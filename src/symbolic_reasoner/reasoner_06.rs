impl SymbolicReasoner {
    /// 前向饱和推导报告：多跳新事实（含前提与规则来源）
    pub fn derived_facts_report(&self, limit: usize) -> String {
        if !self.config.enable_resolution {
            return "归结推理未启用（config.toml [reasoning] enable_resolution = false）。"
                .to_string();
        }
        // 报告与回答共用同一套可见性：被否定或被槽压制的推导不再作为
        // 「推理发现」展示，否则用户会在报告里看到回答已经排除的幽灵结论。
        let derived: Vec<DerivedFact> = self
            .resolver
            .forward_chain(3, 100)
            .into_iter()
            .filter(|fact| !self.derived_fact_is_blocked(fact))
            .collect();
        let mut output = String::from("推理发现（前向饱和，多跳优先；均可回溯前提）：\n");
        output.push_str(&format!(
            "  现有 {} 条事实、{} 条规则，本轮推出 {} 条新事实。\n",
            self.resolver.fact_count(),
            self.resolver.clause_count(),
            derived.len()
        ));
        if derived.is_empty() {
            output.push_str("  （没有可推导的新事实：可增加事实/规则或放宽规则阈值）\n");
            return output;
        }
        let mut items: Vec<(&DerivedFact, String)> = derived
            .iter()
            .map(|item| (item, item.atom.display()))
            .collect();
        items.sort_by(|(a, a_display), (b, b_display)| {
            b.depth
                .cmp(&a.depth)
                .then_with(|| {
                    b.confidence
                        .partial_cmp(&a.confidence)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| a_display.cmp(b_display))
        });
        for (index, (item, _)) in items.into_iter().take(limit.max(1)).enumerate() {
            let premises = item
                .premises
                .iter()
                .map(|premise| self.render_atom_nl(premise))
                .collect::<Vec<_>>()
                .join("；");
            output.push_str(&format!(
                "  {}. {}（第 {} 轮推导；前提：{}；规则：{}；置信度 {:.2}）\n",
                index + 1,
                self.render_atom_nl(&item.atom),
                item.depth,
                premises,
                item.rule_source,
                item.confidence
            ));
        }
        output.trim_end().to_string()
    }

    fn is_known_entity(&self, name: &str) -> bool {
        if self.aligner.kind_of(name) == Some(TermKind::Entity) {
            return true;
        }
        self.positive_session_atoms()
            .chain(self.session_negative_facts.iter())
            .any(|atom| {
                atom.terms
                    .iter()
                    .any(|term| !term.is_var() && term.name() == name)
            })
    }

    fn is_known_predicate(&self, name: &str) -> bool {
        let canonical = self.aligner.canonical_form(name);
        let generated = self.generator.canonical_predicate(name);
        self.aligner.kind_of(name) == Some(TermKind::Relation)
            || self.inverse_query_map.contains_key(name)
            || self.inverse_query_map.contains_key(canonical.as_str())
            || self.predictor.has_predicate(name)
            // 正向别名（同义词）按规范化槽位回查预测器
            || (canonical != name && self.predictor.has_predicate(&canonical))
            || generated
                .as_deref()
                .is_some_and(|slot| self.predictor.has_predicate(slot))
            || self
                .positive_session_atoms()
                .chain(self.session_negative_facts.iter())
                .any(|atom| {
                    atom.predicate.as_ref() == name || atom.predicate.as_ref() == canonical
                })
    }

    fn split_chain(&self, segments: &[String]) -> Option<(String, Vec<ChainPredicate>)> {
        if segments.is_empty() {
            return None;
        }
        // 在"精确登记"的实体名上做最长前缀切分
        for take in (1..=segments.len()).rev() {
            let candidate = segments[..take].join("的");
            if self.aligner.kind_of(&candidate) == Some(TermKind::Entity) {
                let predicates = segments[take..]
                    .iter()
                    .map(|segment| {
                        let (slot, inverted) = self.resolve_question_predicate(segment);
                        ChainPredicate {
                            surface: segment.trim().to_string(),
                            slot,
                            inverted,
                        }
                    })
                    .collect();
                return Some((self.aligner.canonical_form(&candidate), predicates));
            }
        }
        let entity = self.aligner.canonical_form(&segments[0]);
        let predicates = segments[1..]
            .iter()
            .map(|segment| {
                let (slot, inverted) = self.resolve_question_predicate(segment);
                ChainPredicate {
                    surface: segment.trim().to_string(),
                    slot,
                    inverted,
                }
            })
            .collect();
        Some((entity, predicates))
    }

    /// 取一个节点沿谓词的命中：正向给出宾语集合，逆向给出主语集合。
    /// 逆向命中以槽位反方向搜索（无预测分支：符号推断仅支持正向出边）。
    fn node_hits(&self, node: &str, predicate: &ChainPredicate) -> Vec<ValueHit> {
        let mut hits: Vec<ValueHit> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        let objects: Vec<String> = if predicate.inverted {
            self.predictor.subjects_of(&predicate.slot, node)
        } else {
            self.predictor.objects_of(node, &predicate.slot)
        };
        for object in objects {
            if seen.insert(object.clone()) {
                hits.push(ValueHit {
                    value: object,
                    prediction: None,
                    law: None,
                    construction: None,
                });
            }
        }
        // 会话事实（用户确认的补全、采纳的假设）与图谱事实同权参与当前轮的
        // 直接命中；它们不写图谱，但必须能在下一轮回答里被用上。
        for atom in self.positive_session_atoms() {
            if atom.predicate.as_ref() != predicate.slot {
                continue;
            }
            let [subject, object] = atom.terms.as_slice() else {
                continue;
            };
            let value = match (predicate.inverted, subject, object) {
                (false, Term::Const(subject), Term::Const(object)) if subject.as_ref() == node => {
                    object.as_ref()
                }
                (true, Term::Const(subject), Term::Const(object)) if object.as_ref() == node => {
                    subject.as_ref()
                }
                _ => continue,
            };
            if seen.insert(value.to_string()) {
                hits.push(ValueHit {
                    value: value.to_string(),
                    prediction: None,
                    law: None,
                    construction: None,
                });
            }
        }
        // 实体属性（非关系型三元组）同样沿属性槽位命中：属性查询与关系查询
        // 走同一条链，逆向（反查主体）暂不支持属性。
        if !predicate.inverted {
            let mut keys = vec![predicate.slot.clone()];
            // 图谱里常见英文属性键（JSON-LD 的 name/age/height…），
            // 用户却按中文词面提问（姓名/年龄/身高…）。仅在中文槽位
            // 无命中时回退到对照表，避免覆盖真实的中文谓词。
            if let Some(canonical) =
                crate::symbolic_generator::canonical_predicate_key(&predicate.surface)
            {
                keys.push(canonical.to_string());
            }
            // 父属性查询召回子属性事实（「身高」→「净身高」）。
            keys.extend(self.attribute_profiles.lookup_keys(&predicate.surface));
            keys.sort();
            keys.dedup();
            for key in keys {
                let Some(values) = self.attribute_index.get(&(node.to_string(), key)) else {
                    continue;
                };
                for value in values {
                    if seen.insert(value.clone()) {
                        hits.push(ValueHit {
                            value: value.clone(),
                            prediction: None,
                            law: None,
                            construction: None,
                        });
                    }
                }
            }
        }
        if !predicate.inverted && self.config.enable_link_prediction {
            let limit = self.config.max_predictions.saturating_mul(4).max(4);
            for prediction in self.predictions_for_entity(node, Some(&predicate.slot), limit) {
                // 只取 node 作为主体的出边预测
                if prediction.triple.subject != node {
                    continue;
                }
                let value = prediction.triple.object.clone();
                if seen.insert(value.clone()) {
                    hits.push(ValueHit {
                        value,
                        prediction: Some(prediction),
                        law: None,
                        construction: None,
                    });
                }
            }
        }
        // 同一取值可能从两条通道到达：关系索引保存的是本地名，而实体属性
        // 保存的是导入时的宾语 IRI（如 `kg:entity/李四`）。显示与计数前把
        // kg 实体 IRI 还原成本地名并按值去重；外部 IRI 原样保留，绝不猜测。
        for hit in &mut hits {
            if let Some(local) =
                crate::identity::entity_local_name(crate::identity::DEFAULT_BASE_NS, &hit.value)
            {
                hit.value = local;
            }
        }
        let mut distinct: HashSet<String> = HashSet::new();
        hits.retain(|hit| distinct.insert(hit.value.clone()));
        // 槽级压制只影响视图：当会话槽已有赢家时，图谱写下的同一槽旧值
        // 不再进入答案；旧值仍保留在图谱与账本里（压制不等于否定）。
        if !predicate.inverted {
            let (slot_subject, slot_predicate) = self.slot_identity(node, &predicate.slot);
            let accepted = self.slots.accepted(&slot_subject, &slot_predicate);
            if !accepted.is_empty() {
                let suppressed: HashSet<String> = self
                    .slots
                    .suppressed(&slot_subject, &slot_predicate)
                    .into_iter()
                    .collect();
                hits.retain(|hit| {
                    !suppressed.contains(&hit.value)
                        || accepted.iter().any(|value| value == &hit.value)
                });
            }
        }
        hits.sort_by(|a, b| a.value.cmp(&b.value));
        hits
    }

    /// 主动声明律带来的查询扩展：传递闭包 / 对称反向 / 互逆反向。
    /// 只从已观测事实做有界遍历，不调用链接预测，因此不会编造未登记取值。
    fn law_hits(
        &self,
        node: &str,
        predicate: &ChainPredicate,
        direct_empty: bool,
    ) -> (Vec<ValueHit>, bool) {
        let mut hits: Vec<ValueHit> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        let mut truncated = false;
        let neighbors = |current: &str| {
            if predicate.inverted {
                self.predictor.subjects_of(&predicate.slot, current)
            } else {
                self.predictor.objects_of(current, &predicate.slot)
            }
        };

        if let Some(law) = self.laws.active_transitive(&predicate.slot) {
            let mut visited: HashSet<String> = HashSet::from([node.to_string()]);
            let mut frontier = vec![node.to_string()];
            let mut expanded = 0usize;
            let mut depth = 0usize;
            loop {
                if depth >= LAW_CLOSURE_MAX_DEPTH {
                    truncated = frontier.iter().any(|current| {
                        neighbors(current)
                            .into_iter()
                            .any(|value| value != node && !visited.contains(&value))
                    });
                    break;
                }
                depth += 1;
                let mut next: Vec<String> = Vec::new();
                for current in &frontier {
                    for value in neighbors(current) {
                        if value == node || !visited.insert(value.clone()) {
                            continue;
                        }
                        expanded += 1;
                        if expanded > LAW_CLOSURE_MAX_NODES {
                            truncated = true;
                            break;
                        }
                        next.push(value);
                    }
                    if truncated {
                        break;
                    }
                }
                if truncated || next.is_empty() {
                    break;
                }
                for value in &next {
                    if seen.insert(value.clone()) {
                        hits.push(ValueHit {
                            value: value.clone(),
                            prediction: None,
                            law: Some(law.id.clone()),
                            construction: None,
                        });
                    }
                }
                frontier = next;
            }
        }

        if direct_empty {
            if let Some(law) = self.laws.active_symmetric(&predicate.slot) {
                let values = if predicate.inverted {
                    self.predictor.objects_of(node, &predicate.slot)
                } else {
                    self.predictor.subjects_of(&predicate.slot, node)
                };
                for value in values {
                    if value != node && seen.insert(value.clone()) {
                        hits.push(ValueHit {
                            value,
                            prediction: None,
                            law: Some(law.id.clone()),
                            construction: None,
                        });
                    }
                }
            }
        }

        if direct_empty {
            if let Some(other) = self.laws.inverse_partner(&predicate.slot) {
                if let Some(law) = self.laws.active_inverse(&predicate.slot, other) {
                    let values = if predicate.inverted {
                        self.predictor.objects_of(node, other)
                    } else {
                        self.predictor.subjects_of(other, node)
                    };
                    for value in values {
                        if value != node && seen.insert(value.clone()) {
                            hits.push(ValueHit {
                                value,
                                prediction: None,
                                law: Some(law.id.clone()),
                                construction: None,
                            });
                        }
                    }
                }
            }
        }

        hits.sort_by(|a, b| a.value.cmp(&b.value));
        (hits, truncated)
    }

    /// 链上所有谓词是否都是同一个声明的传递谓词（仅词面重复时启用闭包选路）。
    fn is_transitive_chain(&self, predicates: &[ChainPredicate]) -> bool {
        if predicates.len() < 2 {
            return false;
        }
        let first = &predicates[0];
        if self.laws.active_transitive(&first.slot).is_none() {
            return false;
        }
        predicates
            .iter()
            .all(|predicate| predicate.slot == first.slot && predicate.inverted == first.inverted)
    }

    fn run_chain(&self, entity: &str, predicates: &[ChainPredicate]) -> Option<ChainRun> {
        self.run_chain_mode(entity, predicates, false)
    }

    /// 指定律闭包通道：重复传递谓词走精确步数链，其余情况回退到逐跳律扩展。
    fn run_chain_via_laws(&self, entity: &str, predicates: &[ChainPredicate]) -> Option<ChainRun> {
        if self.is_transitive_chain(predicates) {
            return self.run_transitive_chain_exact(entity, predicates);
        }
        self.run_chain_mode(entity, predicates, true)
    }

    /// 传递链的精确语义：`P(P(x))` 是恰好两跳，而不是「所有可达后代」。
    /// 传递律在这里负责授权与归因；闭包可达性不能替代步数约束。
    fn run_transitive_chain_exact(
        &self,
        entity: &str,
        predicates: &[ChainPredicate],
    ) -> Option<ChainRun> {
        let law = self.laws.active_transitive(&predicates[0].slot)?;
        let mut frontier = vec![entity.to_string()];
        let mut steps: Vec<ChainStep> = Vec::new();
        let mut last_hits: Vec<ValueHit> = Vec::new();
        for predicate in predicates {
            let mut next_frontier: Vec<String> = Vec::new();
            let mut round_hits: Vec<ValueHit> = Vec::new();
            let mut seen_values: HashSet<String> = HashSet::new();
            for node in &frontier {
                let mut hits = self.node_hits(node, predicate);
                for hit in &mut hits {
                    if hit.prediction.is_none() {
                        hit.law = Some(law.id.clone());
                    }
                }
                if hits.is_empty() {
                    continue;
                }
                for hit in &hits {
                    if seen_values.insert(hit.value.clone()) {
                        next_frontier.push(hit.value.clone());
                        round_hits.push(hit.clone());
                    }
                }
                steps.push(ChainStep {
                    subject: node.clone(),
                    predicate: predicate.surface.clone(),
                    hits,
                });
            }
            if next_frontier.is_empty() {
                return None;
            }
            frontier = next_frontier;
            last_hits = round_hits;
        }
        if last_hits.is_empty() {
            return None;
        }
        Some(ChainRun {
            steps,
            last_hits,
            closure_truncated: false,
            construction_truncated: false,
            construction_facts: 0,
        })
    }

    fn new_construction_reasoner(&self) -> Option<ConstructiveReasoner> {
        if !self.config.constructive.enabled {
            return None;
        }
        let mut observed = ObservedGraph::default();
        for triple in self.predictor.triples() {
            // 会话否定事实是显式阻断边：被否认的前提不得再作为建设性推断的
            // 观察依据，否则否定只挡住了直接链，却从虚拟事实侧绕了回来。
            if self.session_negative_blocks(&triple.subject, &triple.predicate, &triple.object) {
                continue;
            }
            observed.insert(&triple.subject, &triple.predicate, &triple.object);
        }
        for atom in self.positive_session_atoms() {
            if let [Term::Const(subject), Term::Const(object)] = atom.terms.as_slice() {
                if self.session_negative_blocks(subject, &atom.predicate, object) {
                    continue;
                }
                observed.insert(subject, &atom.predicate, object);
            }
        }
        let forbidden: Vec<FactKey> = self
            .session_negative_facts
            .iter()
            .filter_map(|atom| {
                let [Term::Const(subject), Term::Const(object)] = atom.terms.as_slice() else {
                    return None;
                };
                Some(FactKey::new(
                    subject.to_string(),
                    format!("{}", atom.predicate),
                    object.to_string(),
                ))
            })
            .collect();
        Some(
            ConstructiveReasoner::new(
                self.config.constructive.clone(),
                self.revision,
                observed,
                self.predictor
                    .rules()
                    .iter()
                    .enumerate()
                    .filter_map(|(index, rule)| {
                        crate::constructive::ConstructionRule::from_horn(rule, index)
                    })
                    .collect(),
            )
            .with_forbidden(forbidden),
        )
    }

    fn constructed_hit(node: &str, inverted: bool, fact: VirtualFact) -> ValueHit {
        let value = if inverted {
            fact.key.subject.clone()
        } else {
            fact.key.object.clone()
        };
        let _ = node;
        ValueHit {
            value,
            prediction: None,
            law: None,
            construction: Some(fact),
        }
    }

    fn run_chain_mode(
        &self,
        entity: &str,
        predicates: &[ChainPredicate],
        force_closure: bool,
    ) -> Option<ChainRun> {
        if predicates.is_empty() {
            return None;
        }
        let mut frontier = vec![entity.to_string()];
        let mut steps: Vec<ChainStep> = Vec::new();
        let mut last_hits: Vec<ValueHit> = Vec::new();
        let mut closure_truncated = false;
        let mut construction = self.new_construction_reasoner();
        let mut construction_truncated = false;
        let mut construction_facts = 0usize;
        for predicate in predicates {
            let mut next_frontier: Vec<String> = Vec::new();
            let mut round_hits: Vec<ValueHit> = Vec::new();
            // 多个分支可能命中同一实体，边收集边去重：两个集合插入条件相同，
            // 合并为一个即可（保留首次出现顺序）
            let mut seen_values: HashSet<String> = HashSet::new();
            for node in &frontier {
                let direct = self.node_hits(node, predicate);
                let mut hits = if force_closure || direct.is_empty() {
                    let (expanded, expanded_truncated) =
                        self.law_hits(node, predicate, direct.is_empty());
                    closure_truncated |= expanded_truncated;
                    if force_closure && !expanded.is_empty() {
                        // 闭包结果优先保留律归属；直接事实只补闭包未覆盖的取值。
                        let mut merged = expanded;
                        let mut seen_hits: HashSet<String> =
                            merged.iter().map(|hit| hit.value.clone()).collect();
                        for hit in direct {
                            if seen_hits.insert(hit.value.clone()) {
                                merged.push(hit);
                            }
                        }
                        merged.sort_by(|a, b| a.value.cmp(&b.value));
                        merged
                    } else if direct.is_empty() {
                        expanded
                    } else {
                        direct
                    }
                } else {
                    direct
                };
                if hits.is_empty() {
                    if let Some(engine) = construction.as_mut() {
                        let report = engine.derive(node, &predicate.slot, predicate.inverted);
                        construction_truncated |= report.truncated;
                        if !report.target_facts.is_empty() {
                            construction_facts += report.target_facts.len();
                            hits = report
                                .target_facts
                                .into_iter()
                                .map(|fact| Self::constructed_hit(node, predicate.inverted, fact))
                                .collect();
                        }
                    }
                }
                if hits.is_empty() {
                    continue;
                }
                for hit in &hits {
                    if seen_values.insert(hit.value.clone()) {
                        next_frontier.push(hit.value.clone());
                        round_hits.push(hit.clone());
                    }
                }
                steps.push(ChainStep {
                    subject: node.clone(),
                    predicate: predicate.surface.clone(),
                    hits,
                });
            }
            if next_frontier.is_empty() {
                return None;
            }
            frontier = next_frontier;
            last_hits = round_hits;
        }
        if last_hits.is_empty() {
            return None;
        }
        Some(ChainRun {
            steps,
            last_hits,
            closure_truncated,
            construction_truncated,
            construction_facts,
        })
    }

    fn discourse_marker(index: usize, total: usize) -> &'static str {
        if total <= 1 {
            return "";
        }
        if index == 0 {
            "首先"
        } else if index + 1 == total {
            "最后"
        } else {
            "此外"
        }
    }

    // 期望答案类型（形式语义中的问题类型约束）：谁→人物、哪里→地点、何时→时间
    // 疑问词识别与关键词表由虚词算符表提供（`function_words::answer_type_keywords`）
    fn expected_type_keywords(question: &str) -> Vec<&'static str> {
        crate::function_words::answer_type_keywords(question)
    }

    fn value_matches_expectation(&self, value: &str, keywords: &[&str]) -> bool {
        let Some(etype) = self.generator.entity_type_of(value) else {
            return false;
        };
        let etype_lower = etype.to_lowercase();
        // 期望类型关键词均为小写（或中文）
        keywords
            .iter()
            .any(|keyword| etype.contains(*keyword) || etype_lower.contains(*keyword))
    }

    fn generate_find_object(
        &self,
        entity: &str,
        predicates: &[ChainPredicate],
        steps: &[ChainStep],
        last_hits: &[ValueHit],
        question: &str,
        closure_truncated: bool,
        construction_truncated: bool,
        construction_facts: usize,
    ) -> String {
        let expectations = Self::expected_type_keywords(question);
        let mut ordered: Vec<&ValueHit> = last_hits.iter().collect();
        if !expectations.is_empty() {
            ordered.sort_by_cached_key(|hit| {
                if self.value_matches_expectation(&hit.value, &expectations) {
                    0usize
                } else {
                    1usize
                }
            });
        }
        let reordered = ordered
            .iter()
            .zip(last_hits.iter())
            .any(|(a, b)| !std::ptr::eq(*a, b));
        // 只复述真正用到的律；直接事实不冒充「推理得出」。
        let law_notes: Vec<String> = steps
            .iter()
            .flat_map(|step| step.hits.iter())
            .chain(last_hits.iter())
            .filter_map(|hit| hit.law.as_deref())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .filter_map(|id| self.laws.explain(id))
            .collect();
        let chain_text = format!(
            "{} 的 {}",
            entity,
            predicates
                .iter()
                .map(|p| p.surface.as_str())
                .collect::<Vec<_>>()
                .join(" 的 ")
        );
        let mut output = String::from("根据知识图谱");
        if steps.len() > 1 {
            output.push_str("（多跳推理）");
        }
        output.push_str("：\n");
        if steps.len() > 1 {
            for step in steps {
                let values = step
                    .hits
                    .iter()
                    .map(|hit| {
                        if let Some(fact) = &hit.construction {
                            format!(
                                "{}（建设性推断，待确认，置信度 {:.2}）",
                                hit.value,
                                fact.confidence()
                            )
                        } else {
                            hit.value.clone()
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("、");
                output.push_str(&format!(
                    "  {} 的 {} 是 {}。\n",
                    step.subject, step.predicate, values
                ));
            }
            let construction_notes: Vec<String> = steps
                .iter()
                .flat_map(|step| step.hits.iter())
                .filter_map(|hit| hit.construction.as_ref())
                .map(|fact| {
                    format!(
                        "{}（置信度 {:.2}，前提：{}）",
                        fact.rule,
                        fact.confidence(),
                        fact.support
                            .iter()
                            .map(|key| key.render())
                            .collect::<Vec<_>>()
                            .join("、")
                    )
                })
                .take(2)
                .collect();
            if !construction_notes.is_empty() {
                output.push_str(&format!("  推断依据：{}\n", construction_notes.join("；")));
            }
            output.push_str("  综合得出：\n");
        }
        if last_hits.len() == 1 {
            let hit = &last_hits[0];
            output.push_str(&format!("  {} 是 {}。", chain_text, hit.value));
            if let Some(prediction) = &hit.prediction {
                output.push_str(&format!(
                    "\n  推理依据（推断，待确认，置信度 {:.2}）：{}",
                    prediction.confidence,
                    self.predictor.explain(prediction)
                ));
            } else if let Some(fact) = &hit.construction {
                output.push_str(&format!(
                    "\n  推理依据（{}，置信度 {:.2}，{}）：{}；前提：{}",
                    fact.state.label(),
                    fact.confidence(),
                    fact.operator.label(),
                    fact.rule,
                    fact.support
                        .iter()
                        .map(|key| key.render())
                        .collect::<Vec<_>>()
                        .join("、")
                ));
            } else if let Some(law_id) = &hit.law {
                if let Some(explain) = self.laws.explain(law_id) {
                    output.push_str(&format!("\n  推理依据：{}", explain));
                }
            } else if !law_notes.is_empty() {
                output.push_str(&format!("\n  推理依据：{}", law_notes.join("；")));
            }
        } else {
            for (index, hit) in ordered.iter().enumerate() {
                let marker = Self::discourse_marker(index, ordered.len());
                output.push_str(&format!(
                    "  {}{}{} 是 {}。",
                    marker,
                    if marker.is_empty() { "" } else { "，" },
                    chain_text,
                    hit.value
                ));
                if let Some(prediction) = &hit.prediction {
                    output.push_str(&format!("（推断，置信度 {:.2}）", prediction.confidence));
                }
                if let Some(fact) = &hit.construction {
                    output.push_str(&format!(
                        "（建设性推断，待确认，置信度 {:.2}）",
                        fact.confidence()
                    ));
                }
                output.push('\n');
            }
            if last_hits
                .iter()
                .any(|hit| hit.prediction.is_some() || hit.construction.is_some())
            {
                let evidence: Vec<String> = last_hits
                    .iter()
                    .filter_map(|hit| {
                        hit.prediction
                            .as_ref()
                            .map(|prediction| self.predictor.explain(prediction))
                            .or_else(|| {
                                hit.construction.as_ref().map(|fact| {
                                    format!(
                                        "{}（建设性推断，置信度 {:.2}，前提：{}）",
                                        fact.rule,
                                        fact.confidence(),
                                        fact.support
                                            .iter()
                                            .map(|key| key.render())
                                            .collect::<Vec<_>>()
                                            .join("、")
                                    )
                                })
                            })
                    })
                    .take(2)
                    .collect();
                if !evidence.is_empty() {
                    output.push_str(&format!("  推理依据：{}", evidence.join("；")));
                }
            }
            if !law_notes.is_empty() {
                output.push_str(&format!("  推理依据：{}\n", law_notes.join("；")));
            }
            if reordered {
                output.push_str(&format!(
                    "\n  说明：已按问题期望的答案类型（{}）优先排序。",
                    expectations.join("/")
                ));
            }
        }
        if closure_truncated {
            output.push_str("\n  说明：传递闭包扩展达到深度或节点上限，未列出的取值仍可能存在。");
        }
        if construction_facts > 0 {
            output.push_str(
                "\n  说明：建设性推断只在本次查询内有效，未写入知识图谱；确认后才进入会话事实。",
            );
            let confirmable = steps
                .iter()
                .flat_map(|step| step.hits.iter())
                .chain(last_hits.iter())
                .filter_map(|hit| hit.construction.as_ref())
                .find_map(|fact| {
                    let key = &fact.key;
                    if key.subject.is_empty() || key.predicate.is_empty() || key.object.is_empty() {
                        None
                    } else {
                        Some(format!("{}|{}|{}", key.subject, key.predicate, key.object))
                    }
                });
            if let Some(spec) = confirmable {
                output.push_str(&format!(
                    "\n  如需采纳，回复「确认推断：{}」；否认请回复「否定推断：{}」。",
                    spec, spec
                ));
            }
        }
        if construction_truncated {
            output
                .push_str("\n  说明：建设性搜索达到跳深、候选或展开预算，未列出的推断仍可能存在。");
        }
        output
    }

    fn traversal_graph(&self) -> TraversalGraph {
        let mut graph = TraversalGraph::new();
        // 主语实体上的 valid_from/valid_to 等字段给出其出边的有效时间；
        // 没有字段的边保持 Unknown，带时间约束的查询按宽松策略标注 caveat。
        let mut bounds: BTreeMap<&str, (Option<&str>, Option<&str>)> = BTreeMap::new();
        for (entity, attribute, value) in &self.attributes {
            let entry = bounds.entry(entity.as_str()).or_default();
            if TEMPORAL_FROM_FIELDS.contains(&attribute.as_str()) {
                entry.0 = Some(value.as_str());
            } else if TEMPORAL_TO_FIELDS.contains(&attribute.as_str()) {
                entry.1 = Some(value.as_str());
            }
        }
        let mut validity_by_entity: BTreeMap<&str, Validity> = bounds
            .into_iter()
            .filter_map(|(entity, (start, end))| {
                let validity = validity_from_bounds(start, end);
                (!validity.is_unknown()).then_some((entity, validity))
            })
            .collect();
        // 实体键可能是 IRI 尾部，路径端点则用 name 属性字面量；登记别名，
        // 否则从知识图谱导入的历史有效期会在路径上丢失。
        let aliases: Vec<(&str, &str)> = self
            .attributes
            .iter()
            .filter(|(_, attribute, _)| attribute == "name")
            .map(|(entity, _, value)| (entity.as_str(), value.as_str()))
            .collect();
        for (entity, alias) in aliases {
            if let Some(validity) = validity_by_entity.get(entity).copied() {
                validity_by_entity.insert(alias, validity);
            }
        }
        for triple in self.predictor.triples() {
            if self.is_suppressed_slot_value(&triple.subject, &triple.predicate, &triple.object) {
                continue;
            }
            let validity = validity_by_entity
                .get(triple.subject.as_str())
                .copied()
                .unwrap_or(Validity::Unknown);
            graph.add_edge_record(
                crate::path_memory::PathEdge::new(
                    &triple.subject,
                    &triple.predicate,
                    &triple.object,
                )
                .with_validity(validity),
            );
        }
        for (entity, attribute, value) in &self.attributes {
            if TEMPORAL_FROM_FIELDS.contains(&attribute.as_str())
                || TEMPORAL_TO_FIELDS.contains(&attribute.as_str())
            {
                continue;
            }
            if self.is_suppressed_slot_value(entity, attribute, value) {
                continue;
            }
            let validity = validity_by_entity
                .get(entity.as_str())
                .copied()
                .unwrap_or(Validity::Unknown);
            graph.add_edge_record(
                crate::path_memory::PathEdge::new(entity, attribute, value)
                    .with_source(EdgeSource::Attribute)
                    .with_validity(validity),
            );
        }
        for atom in &self.session_facts {
            if atom.terms.len() == 2 {
                if let (Term::Const(subject), Term::Const(object)) =
                    (&atom.terms[0], &atom.terms[1])
                {
                    graph.add_edge_with_source(
                        subject,
                        &atom.predicate,
                        object,
                        EdgeSource::Session,
                    );
                }
            }
        }
        if let Some(world) = &self.assumption_world {
            for atom in &world.assumptions {
                if atom.terms.len() == 2 {
                    if let (Term::Const(subject), Term::Const(object)) =
                        (&atom.terms[0], &atom.terms[1])
                    {
                        graph.add_edge_with_source(
                            subject,
                            &atom.predicate,
                            object,
                            EdgeSource::Assumption,
                        );
                    }
                }
            }
        }
        for atom in &self.session_negative_facts {
            if atom.terms.len() == 2 {
                if let (Term::Const(subject), Term::Const(object)) =
                    (&atom.terms[0], &atom.terms[1])
                {
                    graph.add_negative_edge(subject, &atom.predicate, object);
                }
            }
        }
        for (entity, entity_type) in self.generator.typed_entities() {
            graph.add_entity_type(&entity, &entity_type);
        }
        graph
    }

    fn session_negative_blocks(&self, subject: &str, predicate: &str, object: &str) -> bool {
        self.session_negative_facts.iter().any(|atom| {
            &*atom.predicate == predicate
                && atom.terms.len() == 2
                && atom.terms[0].name() == subject
                && atom.terms[1].name() == object
        })
    }

    fn path_entity(&self, raw: &str) -> String {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return String::new();
        }
        if self.is_known_entity(trimmed) || self.predictor.participates(trimmed) {
            trimmed.to_string()
        } else {
            self.aligner.canonical_form(trimmed)
        }
    }

    fn path_options(&self, max_depth: usize) -> TraversalOptions {
        TraversalOptions {
            max_depth: max_depth.clamp(1, 6),
            max_paths: 12,
            ..TraversalOptions::default()
        }
    }

    fn path_options_for_question(&self, max_depth: usize, question: &str) -> TraversalOptions {
        let mut options = self.path_options(max_depth);
        if let Some(span) = temporal::parse_time_span(question) {
            options.temporal = TemporalConstraint::from_span(span);
            // 现有知识库大多没有 valid-time 元数据：未知时间只作背景并标注，
            // 但已经写明有效期的边仍会被时间合一硬性阻断。
            options.unknown_time = UnknownTimePolicy::Background;
        }
        let keywords = Self::expected_type_keywords(question);
        if keywords.is_empty() {
            return options;
        }
        let mut expected = BTreeSet::new();
        for (_, entity_type) in self.generator.typed_entities() {
            let lower = entity_type.to_lowercase();
            if keywords
                .iter()
                .any(|keyword| entity_type.contains(*keyword) || lower.contains(*keyword))
            {
                expected.insert(entity_type);
            }
        }
        options.expected_end_types = expected;
        options
    }

    fn answer_chain_with_paths(
        &self,
        entity: &str,
        predicates: &[ChainPredicate],
        question: &str,
    ) -> Option<String> {
        if predicates.len() < 2 {
            return None;
        }
        let graph = self.traversal_graph();
        let chain: Vec<PathPredicate> = predicates
            .iter()
            .map(|predicate| PathPredicate {
                predicate: predicate.slot.clone(),
                inverted: predicate.inverted,
            })
            .collect();
        let options = self.path_options_for_question(predicates.len(), question);
        let bundle = graph.enumerate_chain(entity, &chain, &options);
        if bundle.witnesses.is_empty() {
            if bundle.blocked.iter().any(|blocked| {
                matches!(
                    blocked.reason,
                    crate::path_memory::BlockReason::NegatedEdge { .. }
                )
            }) {
                return Some(bundle.render_text());
            }
            return None;
        }

        let expectations = Self::expected_type_keywords(question);
        let mut groups: Vec<(String, Vec<usize>)> = bundle.answer_groups().into_iter().collect();
        groups.sort_by(|(left, _), (right, _)| {
            let left_rank =
                if expectations.is_empty() || self.value_matches_expectation(left, &expectations) {
                    0
                } else {
                    1
                };
            let right_rank = if expectations.is_empty()
                || self.value_matches_expectation(right, &expectations)
            {
                0
            } else {
                1
            };
            left_rank.cmp(&right_rank).then(left.cmp(right))
        });

        let chain_text = format!(
            "{} 的 {}",
            entity,
            predicates
                .iter()
                .map(|predicate| predicate.surface.as_str())
                .collect::<Vec<_>>()
                .join(" 的 ")
        );
        let mut output = format!(
            "根据知识图谱（多跳路径，共 {} 条）：\n",
            bundle.witnesses.len()
        );
        let mut path_index = 0usize;
        for (value, indices) in &groups {
            output.push_str(&format!("  结论：{} 是 {}。\n", chain_text, value));
            for index in indices {
                if let Some(witness) = bundle.witnesses.get(*index) {
                    path_index += 1;
                    output.push_str(&format!(
                        "    路径 {}（{} 跳）：{}\n",
                        path_index,
                        witness.hop_count(),
                        witness.render()
                    ));
                }
            }
        }
        if !bundle.blocked.is_empty() {
            let blockers: Vec<String> = bundle
                .blocked
                .iter()
                .take(3)
                .map(|blocked| blocked.render())
                .collect();
            output.push_str(&format!("  阻断：{}\n", blockers.join("；")));
        }
        let law_notes: Vec<String> = predicates
            .iter()
            .filter_map(|predicate| self.laws.active_transitive(&predicate.slot))
            .map(|law| law.id.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .filter_map(|id| self.laws.explain(&id))
            .collect();
        if !law_notes.is_empty() {
            output.push_str(&format!("  推理依据：{}\n", law_notes.join("；")));
        }
        if bundle.truncated {
            output.push_str("  说明：路径结果已按上限截断。\n");
        }
        Some(output.trim_end().to_string())
    }

    fn answer_path_between(&self, start: &str, goal: &str, max_hops: usize) -> Option<String> {
        let start = self.path_entity(start);
        let goal = self.path_entity(goal);
        if start.is_empty() || goal.is_empty() {
            return None;
        }
        let graph = self.traversal_graph();
        let bundle = graph.enumerate_between(&start, &goal, &self.path_options(max_hops));
        Some(bundle.render_text())
    }

    fn answer_common_neighbor(&self, left: &str, right: &str) -> Option<String> {
        let left = self.path_entity(left);
        let right = self.path_entity(right);
        if left.is_empty() || right.is_empty() {
            return None;
        }
        let graph = self.traversal_graph();
        let bundle = graph.common_neighbors(&left, &right, &self.path_options(2));
        if bundle.witnesses.is_empty() {
            return Some(bundle.render_text());
        }
        let mut groups: Vec<(String, Vec<usize>)> =
            bundle.intermediary_groups().into_iter().collect();
        groups.sort_by(|(left, left_paths), (right, right_paths)| {
            right_paths
                .len()
                .cmp(&left_paths.len())
                .then(left.cmp(right))
        });
        let mut output = format!("共同邻居（{} 和 {}）：\n", left, right);
        for (hub, indices) in groups {
            output.push_str(&format!("  {}（{} 条路径）\n", hub, indices.len()));
            for index in indices {
                if let Some(witness) = bundle.witnesses.get(index) {
                    output.push_str(&format!("    {}\n", witness.render()));
                }
            }
        }
        if bundle.truncated {
            output.push_str("  说明：结果已按上限截断。\n");
        }
        Some(output.trim_end().to_string())
    }

    fn answer_intermediary(&self, start: &str, goal: &str, max_hops: usize) -> Option<String> {
        let start = self.path_entity(start);
        let goal = self.path_entity(goal);
        if start.is_empty() || goal.is_empty() {
            return None;
        }
        let graph = self.traversal_graph();
        let bundle = graph.enumerate_between(&start, &goal, &self.path_options(max_hops));
        if bundle.witnesses.is_empty() {
            return Some(bundle.render_text());
        }
        let mut groups: Vec<(String, Vec<usize>)> =
            bundle.intermediary_groups().into_iter().collect();
        groups.sort_by(|(left, left_paths), (right, right_paths)| {
            right_paths
                .len()
                .cmp(&left_paths.len())
                .then(left.cmp(right))
        });
        let mut output = format!("从「{}」到「{}」的路径：\n", start, goal);
        if groups.is_empty() {
            output.push_str("  直接相连，不需要中间人。\n");
        } else {
            for (hub, indices) in groups {
                output.push_str(&format!("  中间人：{}（{} 条路径）\n", hub, indices.len()));
                for index in indices {
                    if let Some(witness) = bundle.witnesses.get(index) {
                        output.push_str(&format!("    {}\n", witness.render()));
                    }
                }
            }
        }
        output.push_str(&format!(
            "  路径枚举：{} 条，扩展 {} 次，访问 {} 个节点。\n",
            bundle.witnesses.len(),
            bundle.expansions,
            bundle.visited
        ));
        if !bundle.blocked.is_empty() {
            let blockers: Vec<String> = bundle
                .blocked
                .iter()
                .take(3)
                .map(|blocked| blocked.render())
                .collect();
            output.push_str(&format!("  阻断：{}\n", blockers.join("；")));
        }
        Some(output.trim_end().to_string())
    }

    fn answer_graph_question(&self, question: &str) -> Option<String> {
        let lexicon = self.graph_lexicon();
        let facts = self.graph_facts();
        let options = MatchOptions::default();
        let mut guard = self.graph_session.lock();
        if let Some(session) = guard.as_mut() {
            if let Some(intent) = parse_repair_intent(question) {
                return Some(match intent {
                    RepairIntent::RestoreOriginal => {
                        self.restore_graph_original(session, &facts, options)
                    }
                    RepairIntent::Adopt(index) => {
                        self.adopt_graph_repair(session, index, &facts, options)
                    }
                });
            }
            if let Some(result) = session
                .conversation
                .interpret(question, &lexicon, &facts, options)
            {
                return Some(match result {
                    Ok(reply) => {
                        session.compiled.pattern = session.conversation.pattern().clone();
                        let mut answer = self.render_graph_reply(
                            &session.compiled,
                            &session.conversation,
                            &reply,
                            &facts,
                        );
                        if let Some(note) = session.ledger.note(session.conversation.pattern()) {
                            answer = format!("{}{}", note, answer);
                        }
                        if let Some((failure, repairs)) =
                            self.graph_failure_parts(&session.conversation, &facts, options)
                        {
                            answer.push('\n');
                            answer.push_str(&failure);
                            session.repairs = repairs;
                        } else {
                            session.repairs.clear();
                        }
                        answer
                    }
                    Err(PatternError::NoRowsToExplain) => {
                        let mut answer = "当前没有匹配结果，无法给出见证子图。".to_string();
                        if let Some((failure, repairs)) =
                            self.graph_failure_parts(&session.conversation, &facts, options)
                        {
                            answer.push('\n');
                            answer.push_str(&failure);
                            session.repairs = repairs;
                        }
                        answer
                    }
                    Err(error) => format!("图谱追问无法执行：{}。", error),
                });
            }
        }
        let compiled = match lexicon.compile(question) {
            Some(compiled) => compiled,
            None => {
                // 结构上像合取问句、但有一处没读懂时，明确说出缺口；
                // 不认识词与「图谱里没有」是两种不同的失败，不能都落到
                // 其他管线后给出一份看似完整的答案。
                return lexicon.compile_gap(question).map(|gap| gap.render());
            }
        };
        let conversation =
            ConversationalGraph::new(compiled.pattern.clone(), &facts, options).ok()?;
        let ledger = RepairLedger::new(compiled.pattern.clone());
        let mut answer = self.render_graph_question(&compiled, conversation.outcome(), &facts);
        let (failure, repairs) = match self.graph_failure_parts(&conversation, &facts, options) {
            Some(parts) => parts,
            None => (String::new(), Vec::new()),
        };
        if !failure.is_empty() {
            answer.push('\n');
            answer.push_str(&failure);
        }
        *guard = Some(GraphSession {
            compiled,
            conversation,
            ledger,
            repairs,
        });
        Some(answer)
    }

    /// 零解时的「失败诊断 + 有界放宽候选」；有解、预算未定或无法诊断时返回 `None`。
    fn graph_failure_parts(
        &self,
        conversation: &ConversationalGraph,
        facts: &[GraphFact],
        options: MatchOptions,
    ) -> Option<(String, Vec<RepairSuggestion>)> {
        if !conversation.outcome().is_empty() {
            return None;
        }
        let diagnosis = diagnose_failure(conversation.pattern(), facts, options)?;
        let mut text = diagnosis.render(conversation.pattern());
        let repairs = suggest_repairs(conversation.pattern(), facts, options, &diagnosis, self, 3);
        if !repairs.is_empty() {
            text.push('\n');
            text.push_str(&self.render_graph_repairs(&repairs));
        }
        Some((text, repairs))
    }

    fn render_graph_repairs(&self, repairs: &[RepairSuggestion]) -> String {
        let mut output = String::from("可尝试的放宽（尚未生效，回复「放宽 1」确认）：");
        for (index, repair) in repairs.iter().enumerate() {
            let lower_bound = if repair.lower_bound { "，下界" } else { "" };
            let examples = if repair.sample_rows.is_empty() {
                String::new()
            } else {
                format!("，例如 {}", repair.sample_rows.join("、"))
            };
            output.push_str(&format!(
                "\n  {}. {}：{}（{} 项{}{}）",
                index + 1,
                repair.label(),
                repair.rationale,
                repair.total,
                lower_bound,
                examples
            ));
        }
        output
    }

    fn adopt_graph_repair(
        &self,
        session: &mut GraphSession,
        index: usize,
        facts: &[GraphFact],
        options: MatchOptions,
    ) -> String {
        let Some(suggestion) = session.repairs.get(index).cloned() else {
            if session.repairs.is_empty() {
                return "当前没有待确认的放宽候选；请先把问题问完整。".to_string();
            }
            return format!(
                "没有第 {} 条放宽候选；当前有 {} 条，可回复「放宽 1」到「放宽 {}」。",
                index + 1,
                session.repairs.len(),
                session.repairs.len()
            );
        };
        let Some(relaxed_pattern) = suggestion.op.apply(session.conversation.pattern()) else {
            return "这条放宽已经无法应用到当前模式，请重新提问。".to_string();
        };
        match session.conversation.apply(
            GraphCommand::Replace(relaxed_pattern.clone()),
            facts,
            options,
        ) {
            Ok(_) => {
                session.ledger.adopt(&suggestion, relaxed_pattern.clone());
                session.compiled.pattern = relaxed_pattern;
                session.repairs.clear();
                let mut answer = String::from(
                    "已采用放宽。以下结果基于放宽后的条件，不是原问题在当前知识下的严格答案。\n",
                );
                if let Some(note) = session.ledger.note(session.conversation.pattern()) {
                    answer.push_str(&note);
                }
                answer.push_str(&self.render_graph_question(
                    &session.compiled,
                    session.conversation.outcome(),
                    facts,
                ));
                if let Some((failure, repairs)) =
                    self.graph_failure_parts(&session.conversation, facts, options)
                {
                    answer.push('\n');
                    answer.push_str(&failure);
                    session.repairs = repairs;
                }
                answer
            }
            Err(error) => format!("放宽无法执行：{}。", error),
        }
    }

    fn restore_graph_original(
        &self,
        session: &mut GraphSession,
        facts: &[GraphFact],
        options: MatchOptions,
    ) -> String {
        match session.conversation.apply(
            GraphCommand::Replace(session.ledger.restore()),
            facts,
            options,
        ) {
            Ok(_) => {
                session.compiled.pattern = session.ledger.base().clone();
                session.repairs.clear();
                let mut answer = String::from("已恢复原条件。\n");
                answer.push_str(&self.render_graph_question(
                    &session.compiled,
                    session.conversation.outcome(),
                    facts,
                ));
                if let Some((failure, repairs)) =
                    self.graph_failure_parts(&session.conversation, facts, options)
                {
                    answer.push('\n');
                    answer.push_str(&failure);
                    session.repairs = repairs;
                }
                answer
            }
            Err(error) => format!("恢复原条件失败：{}。", error),
        }
    }

    fn render_graph_reply(
        &self,
        compiled: &CompiledGraphQuestion,
        conversation: &ConversationalGraph,
        reply: &GraphReply,
        facts: &[GraphFact],
    ) -> String {
        match reply {
            GraphReply::Page {
                page,
                pages,
                total,
                lower_bound,
                rows,
            } => {
                let select = &conversation.pattern().select;
                let values: Vec<String> = rows
                    .iter()
                    .map(|solution| solution.row(select))
                    .filter(|row| !row.is_empty())
                    .collect();
                let mut output = if values.is_empty() {
                    format!(
                        "第 {}/{} 页没有更多结果（共 {} 项）。",
                        page + 1,
                        pages,
                        total
                    )
                } else {
                    format!(
                        "第 {}/{} 页（共 {} 项）：{}。",
                        page + 1,
                        pages,
                        total,
                        values.join("、")
                    )
                };
                if *lower_bound {
                    output.push_str("（搜索预算已用尽，结果为下界）");
                }
                output
            }
            GraphReply::Explanation(text) => text.clone(),
            GraphReply::Count { count, lower_bound } => {
                if *lower_bound {
                    format!("当前至少匹配 {} 项（搜索预算已用尽，结果为下界）。", count)
                } else {
                    format!("当前共匹配 {} 项。", count)
                }
            }
            GraphReply::Changed { .. } => {
                self.render_graph_question(compiled, conversation.outcome(), facts)
            }
        }
    }
}

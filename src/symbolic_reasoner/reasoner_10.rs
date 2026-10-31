impl SymbolicReasoner {
    /// 忘记一条会话规则：子句库与教学原句两处一致撤回（图账本由调用方重同步）。
    ///
    /// 教学原句若解析出同一规则，整句一并撤回——否则重启时 `rebuild_problog`
    /// 重放原句会让规则经「教学」通道复活。
    pub fn forget_rule_report(&mut self, sentence_raw: &str) -> String {
        if !self.config.enable_resolution {
            return "归结推理未启用（config.toml [reasoning] enable_resolution = false）。"
                .to_string();
        }
        let sentence = sentence_raw.trim();
        if sentence.is_empty() {
            return "请给出要忘记的规则，例如：忘记规则：如果X的师傅是Y，那么X的老师是Y"
                .to_string();
        }
        let outcome = chinese_reasoning_nlu::parse_lessons(sentence);
        let targets: Vec<Clause> = outcome
            .lessons
            .iter()
            .filter_map(chinese_reasoning_nlu::lesson_to_clause)
            .collect();
        if targets.is_empty() {
            let mut output = String::from("没能把这句话解析成可忘记的规则。支持句式：\n");
            output.push_str("  • 如果S的P是O，那么S的Q是O（与「规则：」同款，占位符自动变量化）\n");
            for error in &outcome.errors {
                output.push_str(&format!("  失败原因：{}\n", error));
            }
            return output.trim_end().to_string();
        }
        let same_clause =
            |left: &Clause, right: &Clause| left.head == right.head && left.body == right.body;

        // 写入前守门：只对确实存在的会话规则做撤回预检。
        let existing_targets: Vec<Clause> = targets
            .iter()
            .filter(|target| {
                self.session_clauses
                    .iter()
                    .any(|clause| same_clause(target, clause))
            })
            .cloned()
            .collect();
        let mut impact = None;
        if !existing_targets.is_empty() {
            let mut preflight = ChangeSet::new(format!("忘记规则：{}", sentence));
            for clause in &existing_targets {
                preflight.push(ChangeOp::RemoveRule(clause.clone()));
            }
            let plan = self.revision_guard().plan(preflight);
            if plan.is_blocked() {
                return self.render_guard_block(&plan);
            }
            impact = Self::render_guard_impact(&plan);
        }

        let existing_clauses = std::mem::take(&mut self.session_clauses);
        let mut removed_targets: Vec<Clause> = Vec::new();
        let mut kept_clauses: Vec<Clause> = Vec::new();
        for clause in existing_clauses {
            match targets.iter().find(|target| same_clause(target, &clause)) {
                Some(_) => removed_targets.push(clause),
                None => kept_clauses.push(clause),
            }
        }
        self.session_clauses = kept_clauses;

        // 教学原句撤回：解析结果命中同一规则的整句一并移除
        let mut lessons_removed = 0usize;
        let mut removed_lessons: Vec<String> = Vec::new();
        let mut kept_lessons: Vec<String> = Vec::new();
        for lesson_sentence in &self.lessons {
            let matches = chinese_reasoning_nlu::parse_lessons(lesson_sentence)
                .lessons
                .iter()
                .filter_map(chinese_reasoning_nlu::lesson_to_clause)
                .any(|parsed| targets.iter().any(|target| same_clause(target, &parsed)));
            if matches {
                lessons_removed += 1;
                removed_lessons.push(lesson_sentence.clone());
            } else {
                kept_lessons.push(lesson_sentence.clone());
            }
        }
        if lessons_removed > 0 {
            self.lessons = kept_lessons;
            self.lessons_parsed = self
                .lessons
                .iter()
                .flat_map(|lesson_sentence| {
                    chinese_reasoning_nlu::parse_lessons(lesson_sentence).lessons
                })
                .collect();
            if self.config.enable_problog && self.config.enable_soft_logic {
                self.rebuild_problog();
            }
        }

        let mut output = String::new();
        if removed_targets.is_empty() {
            output.push_str("会话规则库中没有匹配的规则（可能未学过或已被忘记）。\n");
        } else {
            output.push_str(&format!("已忘记 {} 条会话规则：\n", removed_targets.len()));
            for (index, clause) in removed_targets.iter().enumerate() {
                output.push_str(&format!(
                    "  {}. {} :- {}\n",
                    index + 1,
                    clause.head.display(),
                    clause
                        .body
                        .iter()
                        .map(|atom| atom.display())
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
            self.rebuild_resolver();
        }
        if lessons_removed > 0 {
            output.push_str(&format!(
                "同时撤回 {} 条对应教学原句（防止重启重放复活）。\n",
                lessons_removed
            ));
        }
        if !removed_targets.is_empty() || !removed_lessons.is_empty() {
            let ops = removed_targets
                .iter()
                .cloned()
                .map(ChangeOp::RemoveRule)
                .collect::<Vec<_>>();
            self.record_session_revision(
                format!("忘记规则：{}", sentence),
                ops,
                removed_lessons,
                Vec::new(),
            );
            if let Some(impact) = impact {
                output.push_str(&impact);
            }
            if let Some(receipt) = self.session_receipts.last() {
                output.push_str(&format!(
                    "修订回执：{}（可用「撤销修订」恢复本次撤回）。\n",
                    receipt.id
                ));
            }
        }
        output.push_str(&format!(
            "会话规则库现有 {} 条子句。",
            self.session_clauses.len()
        ));
        output
    }

    /// 前向饱和的派生事实 → 图账本条目（含前提、规则来源、轮次、置信度与物化戳）。
    ///
    /// 与「推理发现」报告同排序（轮次降序 → 置信度降序 → 原子字典序），
    /// 截断到 `limit` 条；物化戳取当前知识版本与当前规则集指纹。
    pub fn derived_ledger_entries(&self, limit: usize) -> Vec<graph_ledger::DerivedLedgerEntry> {
        if !self.config.enable_resolution || limit == 0 {
            return Vec::new();
        }
        let fingerprint = self.rule_fingerprint();
        let derived: Vec<DerivedFact> = self
            .resolver
            .forward_chain(3, 100)
            .into_iter()
            .filter(|fact| !self.derived_fact_is_blocked(fact))
            .collect();
        let mut items: Vec<&DerivedFact> = derived.iter().collect();
        items.sort_by(|a, b| {
            b.depth
                .cmp(&a.depth)
                .then_with(|| {
                    b.confidence
                        .partial_cmp(&a.confidence)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| a.atom.display().cmp(&b.atom.display()))
        });
        items.truncate(limit);
        items
            .into_iter()
            .map(|item| {
                let atom = ledger_atom(&item.atom);
                graph_ledger::DerivedLedgerEntry {
                    id: graph_ledger::derived_entry_id(&atom),
                    atom,
                    rule_source: item.rule_source.clone(),
                    premises: item.premises.iter().map(ledger_atom).collect(),
                    depth: item.depth,
                    confidence: item.confidence,
                    revision: self.revision,
                    fingerprint: fingerprint.clone(),
                }
            })
            .collect()
    }

    /// ProbLog 风格可能世界精确推理报告：
    /// 概率事实/规则/注释析取 → BDD 加权模型计数（无采样），并列出证明路径与联合概率
    pub fn problog_report(&self, target_raw: &str) -> String {
        if !self.config.enable_problog {
            return "概率可能世界推理未启用（config.toml [reasoning] enable_problog = false）。"
                .to_string();
        }
        if !self.config.enable_soft_logic {
            return "概率可能世界推理依赖加权软逻辑（图谱事实与规则同源），请先启用 enable_soft_logic。"
                .to_string();
        }
        let target = target_raw.trim();
        if target.is_empty() {
            return "请输入要精确查询的对象，例如：概率推理：李四的师祖 或 概率推理：李四"
                .to_string();
        }
        let (entity, predicate, known_entity) = match self.split_entity_predicate(target) {
            Some((entity, predicate)) => (entity, predicate, true),
            None => {
                let trimmed = target.trim();
                match trimmed.split_once('的') {
                    Some((entity, predicate)) if !entity.trim().is_empty() => (
                        entity.trim().to_string(),
                        Some(predicate.trim().to_string())
                            .filter(|predicate| !predicate.is_empty()),
                        false,
                    ),
                    _ if !trimmed.is_empty() => (trimmed.to_string(), None, false),
                    _ => return format!("无法在「{}」中识别出已知实体。", target),
                }
            }
        };
        let patterns = if known_entity {
            self.problog_patterns(&entity, predicate.as_deref())
        } else {
            let mut patterns = self.problog.patterns_for_subject(&entity);
            if let Some(predicate) = predicate.as_deref() {
                patterns.retain(|atom| {
                    atom.predicate.contains(predicate)
                        || predicate.contains(atom.predicate.as_ref())
                });
            }
            patterns.truncate(6);
            patterns
        };
        if patterns.is_empty() {
            return if known_entity {
                format!(
                    "没有找到关于「{}」可用于概率查询的目标（可先用「教学：」教一条知识，或确认实体已在知识图谱中）。",
                    entity
                )
            } else {
                format!(
                    "无法在「{}」中识别出已知实体（也没有教过与「{}」相关的概率知识）。",
                    target, entity
                )
            };
        }
        let mut merged: HashMap<String, ProblogAnswer> = HashMap::new();
        let mut stats: Option<(usize, usize, usize, usize, bool)> = None;
        for pattern in &patterns {
            let query =
                self.problog
                    .query(pattern, 8, self.config.soft_logic_min_probability.max(0.05));
            stats.get_or_insert((
                query.choice_variables,
                query.bdd_nodes,
                query.ground_instances,
                query.ground_rounds,
                query.exact,
            ));
            for answer in query.answers {
                merged
                    .entry(answer.atom.display())
                    .and_modify(|existing| {
                        if answer.probability > existing.probability {
                            *existing = answer.clone();
                        }
                    })
                    .or_insert(answer);
            }
        }
        let mut answers: Vec<ProblogAnswer> = merged.into_values().collect();
        answers.sort_by(|a, b| {
            b.probability
                .partial_cmp(&a.probability)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.atom.display().cmp(&b.atom.display()))
        });
        answers.truncate(8);
        if answers.is_empty() {
            return format!("关于「{}」目前没有概率 ≥ 阈值的精确推断结果。", entity);
        }
        let (choice_variables, bdd_nodes, ground_instances, ground_rounds, exact) =
            stats.unwrap_or((0, 0, 0, 0, true));
        let mut output = format!(
            "可能世界精确推理（ProbLog 风格：{} 轮基化得到 {} 条规则实例，{} 个选择变量编译成 {} 个 BDD 节点；加权模型计数，无采样、不训练）\n",
            ground_rounds, ground_instances, choice_variables, bdd_nodes
        );
        if !exact {
            output.push_str("  注意：BDD 超出编译上限，概率为证明路径联合概率上界近似。\n");
        }
        let soft_probabilities = self.soft.probability_map();
        for (index, answer) in answers.iter().enumerate() {
            let source_note = if answer.from_fact {
                if self.soft.has_fact(&answer.atom) {
                    "图谱事实"
                } else {
                    "概率事实/会话事实"
                }
            } else {
                "由概率规则编译推出"
            };
            output.push_str(&format!(
                "  {}. {} {}（P={:.3}，{}）\n",
                index + 1,
                self.render_atom_nl(&answer.atom),
                MarkovLogicEngine::hedge(answer.probability),
                answer.probability,
                source_note
            ));
            let soft_probability = if self.soft.has_fact(&answer.atom) {
                1.0
            } else {
                soft_probabilities.get(&answer.atom).copied().unwrap_or(0.0)
            };
            if soft_probability > 1e-9 {
                output.push_str(&format!(
                    "     与软逻辑（MLN-lite）对照：{:.3}，Δ={:+.3}（差异来自两套语义：log-odds 求和 vs 可能世界共享选择变量）\n",
                    soft_probability,
                    answer.probability - soft_probability
                ));
            }
            if answer.proofs.is_empty() {
                if !answer.from_fact {
                    output.push_str("     证明路径：（存在递归/循环依赖，未展开有限证明路径）\n");
                }
                continue;
            }
            for (proof_index, proof) in answer.proofs.iter().take(3).enumerate() {
                let steps = proof
                    .steps
                    .iter()
                    .map(|step| {
                        let premises = if step.premises.is_empty() {
                            "（无条件）".to_string()
                        } else {
                            step.premises
                                .iter()
                                .map(|premise| self.render_atom_nl(premise))
                                .collect::<Vec<_>>()
                                .join(" 且 ")
                        };
                        format!(
                            "{} —[{:.2}]→ {}（来源：{}）",
                            premises,
                            step.probability,
                            self.render_atom_nl(&step.head),
                            step.source
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("；");
                output.push_str(&format!(
                    "     证明路径 {}（联合概率 {:.3}）：{}\n",
                    proof_index + 1,
                    proof.probability,
                    steps
                ));
            }
            let path_sum: f64 = answer.proofs.iter().map(|proof| proof.probability).sum();
            if answer.proofs.len() > 1 && path_sum > answer.probability + 1e-6 {
                output.push_str(&format!(
                    "     路径联合概率之和 {:.3} > 精确概率 {:.3}：各路径共享同一批选择变量（证明重叠），精确值已做去重\n",
                    path_sum, answer.probability
                ));
            }
        }
        output.trim_end().to_string()
    }

    /// 自然语言情态问句 → ProbLog 概率算子。
    ///
    /// 只接管**明确带情态线索**的问句；普通问句返回 `None`，继续走原有管线。
    /// 输出永远是双账本：先给 PQI 的理解后验，再给 ProbLog 的知识概率，
    /// 两者都不相乘、不互相替代。
    pub fn answer_probabilistic_intent(&self, question: &str) -> Option<String> {
        if !self.config.enable_problog || !self.config.enable_soft_logic {
            return None;
        }
        let intent = ProbabilisticQueryIntent::analyze(question)?;
        let target = match intent.target.clone() {
            Some(target) => self.normalize_epistemic_target(target),
            None => self.epistemic_target_from_core(&intent.core_question)?,
        };
        self.render_probabilistic_intent(&intent, &target)
    }

    fn canonical_epistemic_entity(&self, raw: &str) -> String {
        let raw = raw.trim();
        if !raw.is_empty() && self.is_known_entity(raw) {
            self.aligner.canonical_form(raw)
        } else {
            raw.to_string()
        }
    }

    /// 词面目标 → 图谱槽位；逆向谓词交换主宾语，而不是改写否定作用域。
    fn normalize_epistemic_claim(&self, claim: &Claim) -> Claim {
        let subject = self.canonical_epistemic_entity(&claim.subject);
        let object = self.canonical_epistemic_entity(&claim.object);
        let (predicate, inverted) = self.resolve_question_predicate(&claim.predicate);
        let (subject, object) = if inverted {
            (object, subject)
        } else {
            (subject, object)
        };
        Claim {
            subject,
            predicate,
            object,
            negated: claim.negated,
        }
    }

    fn orient_epistemic_target(&self, subject: &str, predicate: &str) -> EpistemicTarget {
        let subject = self.canonical_epistemic_entity(subject);
        let (predicate, inverted) = self.resolve_question_predicate(predicate);
        if inverted {
            EpistemicTarget::SubjectsOf {
                object: subject,
                predicate,
            }
        } else {
            EpistemicTarget::ObjectsOf { subject, predicate }
        }
    }

    fn normalize_epistemic_target(&self, target: EpistemicTarget) -> EpistemicTarget {
        match target {
            EpistemicTarget::GroundClaim(claim) => {
                EpistemicTarget::GroundClaim(self.normalize_epistemic_claim(&claim))
            }
            EpistemicTarget::ObjectsOf { subject, predicate } => {
                self.orient_epistemic_target(&subject, &predicate)
            }
            EpistemicTarget::SubjectsOf { object, predicate } => {
                let (predicate, inverted) = self.resolve_question_predicate(&predicate);
                let object = self.canonical_epistemic_entity(&object);
                if inverted {
                    EpistemicTarget::ObjectsOf {
                        subject: object,
                        predicate,
                    }
                } else {
                    EpistemicTarget::SubjectsOf { object, predicate }
                }
            }
            EpistemicTarget::Comparison { left, right } => EpistemicTarget::Comparison {
                left: self.normalize_epistemic_claim(&left),
                right: self.normalize_epistemic_claim(&right),
            },
        }
    }

    /// 情态剥离后的指称问句 → 结构化目标。只接受单跳目标；多跳链让位给
    /// 原有推理管线，避免把中间实体的概率误当成整条链的概率。
    fn epistemic_target_from_core(&self, question: &str) -> Option<EpistemicTarget> {
        if let Some(plan) = self.planner.parse(question) {
            if let Some(target) = self.epistemic_target_from_plan(plan) {
                return Some(target);
            }
        }
        self.epistemic_target_from_proposition(question)
    }

    fn epistemic_target_from_plan(&self, plan: QueryPlan) -> Option<EpistemicTarget> {
        match plan {
            QueryPlan::FindObject { segments } => {
                let (entity, predicates) = self.split_chain(&segments)?;
                if predicates.len() != 1 {
                    return None;
                }
                let predicate = &predicates[0];
                if predicate.inverted {
                    Some(EpistemicTarget::SubjectsOf {
                        object: entity,
                        predicate: predicate.slot.clone(),
                    })
                } else {
                    Some(EpistemicTarget::ObjectsOf {
                        subject: entity,
                        predicate: predicate.slot.clone(),
                    })
                }
            }
            QueryPlan::FindSubject { object, predicate } => {
                let (slot, inverted) = self.resolve_question_predicate(&predicate);
                let object = self.canonical_epistemic_entity(&object);
                if inverted {
                    Some(EpistemicTarget::ObjectsOf {
                        subject: object,
                        predicate: slot,
                    })
                } else {
                    Some(EpistemicTarget::SubjectsOf {
                        object,
                        predicate: slot,
                    })
                }
            }
            QueryPlan::CheckTriple {
                subject_segments,
                object,
            } => {
                let (entity, predicates) = self.split_chain(&subject_segments)?;
                if predicates.len() != 1 {
                    return None;
                }
                let predicate = &predicates[0];
                let object = self.canonical_epistemic_entity(&object);
                let (subject, object) = if predicate.inverted {
                    (object, entity)
                } else {
                    (entity, object)
                };
                Some(EpistemicTarget::GroundClaim(Claim {
                    subject,
                    predicate: predicate.slot.clone(),
                    object,
                    negated: false,
                }))
            }
            QueryPlan::CheckRelation {
                subject,
                predicate,
                object,
                negated,
            } => Some(EpistemicTarget::GroundClaim(
                self.normalize_epistemic_claim(&Claim {
                    subject,
                    predicate,
                    object,
                    negated,
                }),
            )),
            _ => None,
        }
    }

    /// 无「的」链、但有已知关系词面的直陈句（「李四认识王五」）用与语义
    /// 蕴含相同的命题解析器兜底，保证概率通道和判断通道读同一套词表。
    fn epistemic_target_from_proposition(&self, question: &str) -> Option<EpistemicTarget> {
        if question.contains('的') && !question.contains('是') {
            return None;
        }
        let propositions = self.proposition_parser().parse(question);
        if propositions.len() != 1 {
            return None;
        }
        let proposition = &propositions[0];
        if !matches!(
            proposition.kind,
            crate::semantic_entailment::PropositionKind::Relation
                | crate::semantic_entailment::PropositionKind::Attribute
        ) {
            return None;
        }
        Some(EpistemicTarget::GroundClaim(Claim {
            subject: proposition.subject.clone(),
            predicate: proposition.predicate.clone(),
            object: proposition.object.clone(),
            negated: proposition.negative,
        }))
    }

    fn render_probabilistic_intent(
        &self,
        intent: &ProbabilisticQueryIntent,
        target: &EpistemicTarget,
    ) -> Option<String> {
        match target {
            EpistemicTarget::GroundClaim(claim) => self.render_epistemic_claim(intent, claim),
            EpistemicTarget::ObjectsOf { subject, predicate } => {
                Some(self.render_epistemic_ranking(intent, Some(subject), None, predicate))
            }
            EpistemicTarget::SubjectsOf { object, predicate } => {
                Some(self.render_epistemic_ranking(intent, None, Some(object), predicate))
            }
            EpistemicTarget::Comparison { left, right } => {
                self.render_epistemic_comparison(intent, left, right)
            }
        }
    }

    fn epistemic_label(probability: f64) -> &'static str {
        if probability >= 0.999 {
            "已确证"
        } else if probability <= 0.001 {
            "已排除"
        } else {
            MarkovLogicEngine::hedge(probability)
        }
    }

    fn render_problog_stats(&self, stats: (usize, usize, usize, usize, bool)) -> String {
        let (choice_variables, bdd_nodes, ground_instances, ground_rounds, exact) = stats;
        format!(
            "ProbLog 编译：{} 轮基化 / {} 条规则实例 / {} 个选择变量 / {} 个 BDD 节点；{}",
            ground_rounds,
            ground_instances,
            choice_variables,
            bdd_nodes,
            if exact {
                "加权模型计数精确"
            } else {
                "超出编译上限，按证明路径上界近似"
            }
        )
    }

    fn render_epistemic_proofs(&self, answer: &ProblogAnswer, limit: usize) -> String {
        if answer.proofs.is_empty() {
            return String::new();
        }
        let mut output = String::from("  依据：\n");
        for (index, proof) in answer.proofs.iter().take(limit).enumerate() {
            let steps = proof
                .steps
                .iter()
                .map(|step| {
                    let premises = if step.premises.is_empty() {
                        "（无条件）".to_string()
                    } else {
                        step.premises
                            .iter()
                            .map(|premise| self.render_atom_nl(premise))
                            .collect::<Vec<_>>()
                            .join(" 且 ")
                    };
                    format!(
                        "{} —[P={:.2}]→ {}（来源：{}）",
                        premises,
                        step.probability,
                        self.render_atom_nl(&step.head),
                        step.source
                    )
                })
                .collect::<Vec<_>>()
                .join("；");
            output.push_str(&format!(
                "   • 证明 {}（联合概率 {:.3}）：{}\n",
                index + 1,
                proof.probability,
                steps
            ));
        }
        output
    }

    /// 对一个基原子求精确概率；已知谓词/实体但无证明时按当前概率程序记 P=0，
    /// 未知谓词或未知实体则返回 `None`（交给普通管线，不把「没学过」说成「很假」）。
    fn epistemic_claim_probability(
        &self,
        claim: &Claim,
    ) -> Option<(
        f64,
        Option<ProblogAnswer>,
        (usize, usize, usize, usize, bool),
    )> {
        let atom = Atom::rel(&claim.predicate, &claim.subject, &claim.object);
        let query = self.problog.query(&atom, 4, 0.0);
        let stats = (
            query.choice_variables,
            query.bdd_nodes,
            query.ground_instances,
            query.ground_rounds,
            query.exact,
        );
        if let Some(answer) = query.answers.into_iter().next() {
            return Some((answer.probability, Some(answer), stats));
        }
        let known = self.is_known_predicate(&claim.predicate)
            && self.is_known_entity(&claim.subject)
            && self.is_known_entity(&claim.object);
        known.then_some((0.0, None, stats))
    }

    /// 判定「P 与阈值差多近」才算边界样本。只有比较类情态才有阈值语义，
    /// 纯概率值问没有边界，直接按满边界处理。
    fn confidence_near_boundary(
        &self,
        intent: &ProbabilisticQueryIntent,
        probability: f64,
    ) -> bool {
        if !self.config.confidence.enabled {
            return false;
        }
        let threshold_relevant = intent.threshold_explicit
            || matches!(
                intent.mode,
                EpistemicMode::Certainty | EpistemicMode::Possibility
            );
        threshold_relevant
            && (probability - intent.threshold).abs() <= self.config.confidence.near_margin
    }

    /// 边界证据分：贴阈值时为 0，离开边界带为 1，中间线性过渡。
    fn confidence_margin_score(&self, intent: &ProbabilisticQueryIntent, probability: f64) -> f64 {
        let threshold_relevant = intent.threshold_explicit
            || matches!(
                intent.mode,
                EpistemicMode::Certainty | EpistemicMode::Possibility
            );
        if !threshold_relevant {
            return 1.0;
        }
        let margin = self.config.confidence.near_margin.max(1e-9);
        ((probability - intent.threshold).abs() / margin).clamp(0.0, 1.0)
    }

    /// 只给已配置的来源提供观测；没配置的来源保持缺失，不被算成 0。
    fn confidence_observations(
        &self,
        intent: &ProbabilisticQueryIntent,
        exact: bool,
        margin: f64,
    ) -> Vec<(&str, f64)> {
        let evidence = if exact {
            1.0
        } else {
            self.config.confidence.approximate_evidence
        };
        self.config
            .confidence
            .sources
            .iter()
            .filter_map(|source| match source.name.as_str() {
                "intent" => Some(("intent", intent.confidence)),
                "evidence" => Some(("evidence", evidence)),
                "margin" => Some(("margin", margin)),
                _ => None,
            })
            .collect()
    }

    /// 统一的四档决策入口；配置关闭时返回 None，旧路径逐字节不变。
    fn confidence_record(
        &self,
        intent: &ProbabilisticQueryIntent,
        probability: f64,
        exact: bool,
    ) -> Option<DecisionRecord> {
        if !self.config.confidence.enabled {
            return None;
        }
        let near_boundary = self.confidence_near_boundary(intent, probability);
        let margin = self.confidence_margin_score(intent, probability);
        self.confidence_record_with_margin(intent, exact, margin, near_boundary)
    }

    /// 排序/比较路径可以自己给「边界强度」，不套用单原子的阈值距离。
    fn confidence_record_with_margin(
        &self,
        intent: &ProbabilisticQueryIntent,
        exact: bool,
        margin: f64,
        near_boundary: bool,
    ) -> Option<DecisionRecord> {
        if !self.config.confidence.enabled {
            return None;
        }
        let observations = self.confidence_observations(intent, exact, margin.clamp(0.0, 1.0));
        Some(
            self.config
                .confidence
                .decide(intent.confidence, &observations, near_boundary),
        )
    }

    /// 决策为澄清/拒答时，不再把后面的结论当作确定事实铺开。
    fn confidence_blocked_note(record: &DecisionRecord, core_question: &str) -> Option<String> {
        match record.decision {
            ConfidenceDecision::Clarify => Some(format!(
                "  为避免误导，先确认你问的是不是「{}」，或换一种更明确的问法。",
                core_question
            )),
            ConfidenceDecision::Abstain => {
                Some("  证据不足以给出可靠结论：本次拒答，不把猜测伪装成答案。".to_string())
            }
            _ => None,
        }
    }

    fn render_epistemic_claim(
        &self,
        intent: &ProbabilisticQueryIntent,
        claim: &Claim,
    ) -> Option<String> {
        let (atom_probability, answer, stats) = self.epistemic_claim_probability(claim)?;
        let probability = if claim.negated {
            1.0 - atom_probability
        } else {
            atom_probability
        };
        let record = self.confidence_record(intent, probability, stats.4);
        if let Some(record) = &record {
            if let Some(note) = Self::confidence_blocked_note(record, &intent.core_question) {
                let mut output = intent.render_ledger();
                output.push('\n');
                output.push_str(&format!("情态算子：{}\n", intent.operator_note()));
                output.push_str(&format!("  {}\n", record.render_receipt()));
                output.push_str(&note);
                return Some(output.trim_end().to_string());
            }
        }
        let mut output = intent.render_ledger();
        output.push('\n');
        output.push_str(&format!("情态算子：{}\n", intent.operator_note()));
        output.push_str("知识概率（ProbLog 加权模型计数，无采样）：\n");
        if claim.negated {
            output.push_str(&format!(
                "  P({} {} {})={:.3}；因此 P({})={:.3}（{}）\n",
                claim.subject,
                claim.predicate,
                claim.object,
                atom_probability,
                claim.render(),
                probability,
                Self::epistemic_label(probability)
            ));
        } else {
            output.push_str(&format!(
                "  {} — {}（P={:.3}）\n",
                claim.render(),
                Self::epistemic_label(probability),
                probability
            ));
        }
        if let Some(answer) = &answer {
            let source = if answer.from_fact {
                if self.soft.has_fact(&answer.atom) {
                    "图谱事实"
                } else {
                    "概率事实/会话事实"
                }
            } else {
                "由概率规则编译推出"
            };
            output.push_str(&format!("  来源：{}\n", source));
            output.push_str(&self.render_epistemic_proofs(answer, 2));
        } else {
            output.push_str(
                "  来源：当前概率程序中没有该原子的证明（图谱是开放世界，P=0 表示在此程序中不可证，不表示现实一定为假）\n",
            );
        }
        if intent.threshold_explicit {
            let met = if intent.threshold_at_least {
                probability >= intent.threshold
            } else {
                probability <= intent.threshold
            };
            output.push_str(&format!(
                "  显式阈值检查：P={:.3} {} {:.2} → {}\n",
                probability,
                if intent.threshold_at_least {
                    "≥"
                } else {
                    "≤"
                },
                intent.threshold,
                if met {
                    "条件成立"
                } else {
                    "条件不成立"
                }
            ));
            if intent.mode == EpistemicMode::Probability {
                output.push_str(
                    "  概率值问：以上为基原子的精确概率；理解置信度与知识概率分账展示。\n",
                );
            }
        } else {
            match intent.mode {
                EpistemicMode::Certainty => {
                    let confirmed = probability >= intent.threshold;
                    output.push_str(&format!(
                        "  确定检查：P={:.3} {} 阈值 {:.2} → {}\n",
                        probability,
                        if confirmed { "≥" } else { "<" },
                        intent.threshold,
                        if confirmed {
                            "可以确认"
                        } else {
                            "不足以确认，只能保留为可能"
                        }
                    ));
                }
                EpistemicMode::Possibility if intent.negated => {
                    let consistent = probability <= intent.threshold;
                    output.push_str(&format!(
                        "  否定可能检查：P={:.3} {} 阈值 {:.2} → {}\n",
                        probability,
                        if consistent { "≤" } else { ">" },
                        intent.threshold,
                        if consistent {
                            "与「不可能」一致"
                        } else {
                            "概率并不低，不能断言「不可能」"
                        }
                    ));
                }
                EpistemicMode::Possibility => {
                    let reached = probability >= intent.threshold;
                    output.push_str(&format!(
                        "  可能阈值：P={:.3} {} τ={:.2} → {}\n",
                        probability,
                        if reached { "≥" } else { "<" },
                        intent.threshold,
                        if reached {
                            "进入可能候选"
                        } else {
                            "低于阈值，只能算弱可能"
                        }
                    ));
                }
                EpistemicMode::Probability => {
                    output.push_str(
                        "  概率值问：以上为基原子的精确概率；理解置信度与知识概率分账展示。\n",
                    );
                }
                EpistemicMode::MostLikely => {}
                EpistemicMode::Plain | EpistemicMode::Comparison => {}
            }
        }
        if let Some(record) = &record {
            output.push_str(&format!("  {}\n", record.render_receipt()));
            if record.decision == ConfidenceDecision::Hedge {
                output.push_str("  说明：置信度一般，以上结论请结合来源核对，不宜当作确定事实。\n");
            }
        }
        output.push_str(&format!("  {}\n", self.render_problog_stats(stats)));
        Some(output.trim_end().to_string())
    }

    fn render_epistemic_ranking(
        &self,
        intent: &ProbabilisticQueryIntent,
        subject: Option<&str>,
        object: Option<&str>,
        predicate: &str,
    ) -> String {
        let fixed = subject.or(object);
        let pattern = match (subject, object) {
            (Some(subject), None) => {
                Atom::pattern(predicate, Term::constant(subject), Term::var("X"))
            }
            (None, Some(object)) => {
                Atom::pattern(predicate, Term::var("X"), Term::constant(object))
            }
            _ => {
                return format!(
                    "{}\n知识概率：候选目标不完整，无法排序。",
                    intent.render_ledger()
                )
            }
        };
        let query = self.problog.query(&pattern, 8, 0.0);
        let stats = (
            query.choice_variables,
            query.bdd_nodes,
            query.ground_instances,
            query.ground_rounds,
            query.exact,
        );
        let mut answers = query.answers;
        answers.retain(|answer| answer.atom.arity() == pattern.arity());
        let mut seen_atoms: HashSet<String> = HashSet::new();
        answers.retain(|answer| seen_atoms.insert(answer.atom.display()));
        let mut output = intent.render_ledger();
        output.push('\n');
        output.push_str(&format!("情态算子：{}\n", intent.operator_note()));
        output.push_str("知识概率（ProbLog 加权模型计数，无采样）：\n");
        if answers.is_empty() {
            output.push_str("  没有可排序的候选：当前概率程序中该谓词没有可证实例。\n");
            output.push_str(&format!("  {}\n", self.render_problog_stats(stats)));
            return output.trim_end().to_string();
        }
        // 每个候选只选一次：MostLikely 曾在这里连续填充两遍，导致
        // 「最可能：李四 / 次选：李四」这种同义重复。阈值过滤与尾部兜底
        // 也合并成一次决策，避免多个分支互相覆盖。
        let filtered: Vec<&ProblogAnswer> = match intent.mode {
            EpistemicMode::Possibility if intent.negated => answers
                .iter()
                .filter(|answer| answer.probability <= intent.threshold)
                .collect(),
            EpistemicMode::Possibility => answers
                .iter()
                .filter(|answer| answer.probability >= intent.threshold)
                .collect(),
            EpistemicMode::MostLikely => answers.iter().take(3).collect(),
            _ => answers.iter().take(5).collect(),
        };
        let threshold_missed = filtered.is_empty() && !answers.is_empty();
        let kept: Vec<&ProblogAnswer> = if filtered.is_empty() {
            answers.iter().take(3).collect()
        } else {
            filtered
        };
        if threshold_missed {
            output.push_str(&format!(
                "  没有候选达到 τ={:.2}；以下只是当前概率最高的弱候选\n",
                intent.threshold
            ));
        }
        let top_probability = kept[0].probability;
        let (margin, near_boundary) = if intent.mode == EpistemicMode::MostLikely {
            match kept.get(1) {
                Some(second) => {
                    let gap = (top_probability - second.probability).max(0.0);
                    let window = self.config.confidence.near_margin.max(1e-9);
                    ((gap / window).clamp(0.0, 1.0), gap <= window)
                }
                None => (1.0, false),
            }
        } else {
            (
                self.confidence_margin_score(intent, top_probability),
                self.confidence_near_boundary(intent, top_probability),
            )
        };
        let record = self.confidence_record_with_margin(intent, stats.4, margin, near_boundary);
        if let Some(record) = &record {
            if let Some(note) = Self::confidence_blocked_note(record, &intent.core_question) {
                output.push_str(&format!("  {}\n", record.render_receipt()));
                output.push_str(&note);
                return output.trim_end().to_string();
            }
        }
        if intent.mode == EpistemicMode::MostLikely {
            let top = kept[0];
            output.push_str(&format!(
                "  最可能：{}（P={:.3}，{}）\n",
                self.epistemic_candidate(top, fixed),
                top.probability,
                Self::epistemic_label(top.probability)
            ));
            if kept.len() > 1 {
                let alternatives = kept
                    .iter()
                    .skip(1)
                    .map(|answer| {
                        format!(
                            "{}（P={:.3}）",
                            self.epistemic_candidate(answer, fixed),
                            answer.probability
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("，");
                output.push_str(&format!("  次选：{}\n", alternatives));
            }
            output.push_str(&self.render_epistemic_proofs(top, 1));
        } else {
            for (index, answer) in kept.iter().take(5).enumerate() {
                output.push_str(&format!(
                    "  {}. {}（P={:.3}，{}）\n",
                    index + 1,
                    self.epistemic_candidate(answer, fixed),
                    answer.probability,
                    Self::epistemic_label(answer.probability)
                ));
            }
            if let Some(top) = kept.first() {
                output.push_str(&self.render_epistemic_proofs(top, 1));
            }
        }
        if let Some(record) = &record {
            output.push_str(&format!("  {}\n", record.render_receipt()));
            if record.decision == ConfidenceDecision::Hedge {
                output.push_str("  说明：候选之间或与阈值过于接近，以上排序请结合来源核对。\n");
            }
        }
        output.push_str(&format!("  {}\n", self.render_problog_stats(stats)));
        output.trim_end().to_string()
    }

    /// 候选出现在变量槽，不能永远读最后一个词项：`师傅(?X, 李四)` 的答案
    /// 是第一个词项；读末尾会把固定锚点「李四」当成候选，反向问句永远
    /// 只能答出锚点本身。
    fn epistemic_candidate(&self, answer: &ProblogAnswer, fixed: Option<&str>) -> String {
        let term = fixed
            .and_then(|anchor| answer.atom.terms.iter().find(|term| term.name() != anchor))
            .or_else(|| answer.atom.terms.last())
            .map(|term| self.canonical_epistemic_entity(term.name()))
            .unwrap_or_else(|| self.render_atom_nl(&answer.atom));
        term
    }

    fn render_epistemic_comparison(
        &self,
        intent: &ProbabilisticQueryIntent,
        left: &Claim,
        right: &Claim,
    ) -> Option<String> {
        let (left_atom, left_answer, left_stats) = self.epistemic_claim_probability(left)?;
        let (right_atom, right_answer, right_stats) = self.epistemic_claim_probability(right)?;
        let left_probability = if left.negated {
            1.0 - left_atom
        } else {
            left_atom
        };
        let right_probability = if right.negated {
            1.0 - right_atom
        } else {
            right_atom
        };
        let mut output = intent.render_ledger();
        output.push('\n');
        output.push_str(&format!("情态算子：{}\n", intent.operator_note()));
        output.push_str("知识概率（ProbLog 加权模型计数，无采样）：\n");
        for (claim, answer, probability) in [
            (left, &left_answer, left_probability),
            (right, &right_answer, right_probability),
        ] {
            output.push_str(&format!(
                "  {} — {}（P={:.3}）\n",
                claim.render(),
                Self::epistemic_label(probability),
                probability
            ));
            if let Some(answer) = answer {
                output.push_str(&self.render_epistemic_proofs(answer, 1));
            }
        }
        let winner = if (left_probability - right_probability).abs() <= 1e-9 {
            None
        } else if left_probability > right_probability {
            Some((left, left_probability, right_probability))
        } else {
            Some((right, right_probability, left_probability))
        };
        match winner {
            Some((claim, winner_p, loser_p)) => output.push_str(&format!(
                "  比较结论：更可能是「{}」（P={:.3} vs {:.3}）。\n",
                claim.render(),
                winner_p,
                loser_p
            )),
            None => output.push_str(&format!(
                "  比较结论：两者概率相同（P={:.3} vs {:.3}），不做任意拍板。\n",
                left_probability, right_probability
            )),
        }
        let stats = (
            left_stats.0.max(right_stats.0),
            left_stats.1.max(right_stats.1),
            left_stats.2.max(right_stats.2),
            left_stats.3.max(right_stats.3),
            left_stats.4 && right_stats.4,
        );
        let gap = (left_probability - right_probability).abs();
        let window = self.config.confidence.near_margin.max(1e-9);
        let record = self.confidence_record_with_margin(
            intent,
            stats.4,
            (gap / window).clamp(0.0, 1.0),
            gap <= window,
        );
        if let Some(record) = &record {
            output.push_str(&format!("  {}\n", record.render_receipt()));
            if let Some(note) = Self::confidence_blocked_note(record, &intent.core_question) {
                output.push_str(&note);
                output.push('\n');
            } else if record.decision == ConfidenceDecision::Hedge {
                output.push_str("  说明：两者概率过于接近，比较结论请结合来源核对。\n");
            }
        }
        output.push_str(&format!("  {}\n", self.render_problog_stats(stats)));
        Some(output.trim_end().to_string())
    }

    /// 从当前会话知识构建默认理论：图谱三元组/属性/实体类型 + 会话事实为事实，
    /// 教过的「除非」默认句为默认规则（cross-session 由 lessons 重放，无新增快照字段）
    fn build_default_theory(&self) -> DefaultTheory {
        let mut theory = DefaultTheory::new(DEFAULT_LOGIC_MAX_ROUNDS);
        for triple in self.predictor.triples() {
            theory.add_fact_atom(Atom::rel(
                &triple.predicate,
                &triple.subject,
                &triple.object,
            ));
        }
        for (subject, key, value) in &self.attributes {
            theory.add_fact_atom(Atom::rel(key, subject, value));
        }
        for (entity, entity_type) in self.generator.typed_entities() {
            if !entity.trim().is_empty() && !entity_type.trim().is_empty() {
                theory.add_fact_atom(Atom::rel("是", &entity, &entity_type));
            }
        }
        for atom in self.positive_session_atoms() {
            theory.add_fact_atom(atom.clone());
        }
        for lesson in &self.lessons_parsed {
            if lesson.kind != LessonKind::DefaultRule {
                continue;
            }
            let Some(exception) = &lesson.exception else {
                continue;
            };
            let prerequisite: Vec<Literal> = lesson
                .body
                .iter()
                .map(|atom| Literal::positive(chinese_reasoning_nlu::generalize_atom(atom)))
                .collect();
            let justifications = vec![Literal::negative(chinese_reasoning_nlu::generalize_atom(
                exception,
            ))];
            let consequent =
                Literal::positive(chinese_reasoning_nlu::generalize_atom(&lesson.head));
            theory.add_default(prerequisite, justifications, consequent, &lesson.source);
        }
        theory
    }

    /// 默认逻辑查询：把会话知识接地成良基扩展，报告触发/例外阻断，
    /// 并用 ProbLog 对同一目标给出「概率视图」对照（不建模例外的乐观倾向）
    pub fn default_logic_report(&mut self, target_raw: &str) -> String {
        let theory = self.build_default_theory();
        let closure = self.discourse.narrator_mut().closure().clone();
        let extension = theory.extend(&closure);
        let target_text = target_raw.trim();
        let target_atom = if target_text.is_empty() {
            None
        } else {
            chinese_reasoning_nlu::parse_proposition(target_text)
        };
        let target = target_atom
            .as_ref()
            .map(|atom| Literal::positive(atom.clone()));
        let mut output = default_logic::render_extension_zh(&theory, &extension, target.as_ref());
        if !target_text.is_empty() && target.is_none() {
            output.push_str(&format!(
                "\n无法把「{}」形式化为可查询命题（支持：S的P是O / S是O / S会V / 谓词(参数)）。",
                target_text
            ));
        }
        if let (Some(atom), true, true) = (
            target_atom.as_ref(),
            self.config.enable_problog,
            self.config.enable_soft_logic,
        ) {
            let query = self.problog.query(atom, 3, 0.0);
            match query.answers.first() {
                Some(answer) => output.push_str(&format!(
                    "\n概率视图（ProbLog 精确可能世界，不建模「除非」例外，仅作对照）：{} {}（P={:.3}）。",
                    self.render_atom_nl(&answer.atom),
                    MarkovLogicEngine::hedge(answer.probability),
                    answer.probability
                )),
                None => output
                    .push_str("\n概率视图（ProbLog）：概率程序中没有与目标相关的证据。"),
            }
        }
        output
    }

    /// 默认逻辑补充证据（结果抽取兜底入口）：主问答路径未命中「没有记录」或
    /// 无法作答时，用「除非」默认知识给出非单调补充；无默认规则、目标无法
    /// 形式化或默认扩展中无证据时返回 None（不产生任何输出与状态变化）
    pub fn default_logic_supplement(&mut self, question: &str) -> Option<String> {
        if !self
            .lessons_parsed
            .iter()
            .any(|lesson| lesson.kind == LessonKind::DefaultRule)
        {
            return None;
        }
        // 自然问句去掉句首礼貌语与句末疑问/标点，让「麻雀会飞吗」也能走默认推理
        let trimmed = question.trim();
        let trimmed = trimmed.strip_prefix("请问").unwrap_or(trimmed);
        let cleaned = trimmed
            .trim_end_matches(|c| {
                matches!(
                    c,
                    '吗' | '呢' | '吧' | '么' | '呀' | '？' | '?' | '。' | '！' | '!'
                )
            })
            .trim();
        let atom = chinese_reasoning_nlu::parse_proposition(cleaned)?;
        let target = Literal::positive(atom);
        let theory = self.build_default_theory();
        let closure = self.discourse.narrator_mut().closure().clone();
        let extension = theory.extend(&closure);
        if let Some(application) = extension.application_for(&target) {
            let mut output = format!(
                "默认推理补充：「{}」通常成立（{}；未发现例外）。",
                default_logic::Literal::render_zh(&target),
                default_logic::render_application_zh(application)
            );
            if extension.truncated {
                output.push_str("（注意：接地推导达到轮数上限，结果可能不完整。）");
            }
            return Some(output);
        }
        if let Some(blocked) = extension.blocked_for(&target).first() {
            return Some(format!(
                "默认推理补充：不能推出「{}」。{}",
                default_logic::Literal::render_zh(&target),
                default_logic::render_blocked_zh(blocked)
            ));
        }
        None
    }

    /// 概率查询模式：有谓词时二元/一元各试一次；无谓词时枚举主语相关的概率原子
    fn problog_patterns(&self, entity: &str, predicate: Option<&str>) -> Vec<Atom> {
        if let Some(pred) = predicate.filter(|pred| !pred.is_empty()) {
            let mut patterns = vec![Atom::pattern(pred, Term::constant(entity), Term::var("X"))];
            patterns.push(Atom::unary(pred, Term::constant(entity)));
            return patterns;
        }
        let mut patterns = self.problog.patterns_for_subject(entity);
        if patterns.is_empty() {
            for fact in self.collect_soft_facts_for_entity(entity, 6) {
                let pattern = if fact.atom.arity() == 2 && fact.atom.terms[0].name() == entity {
                    Atom::pattern(
                        fact.atom.predicate.as_ref(),
                        Term::constant(entity),
                        Term::var("X"),
                    )
                } else {
                    fact.atom.clone()
                };
                if !patterns.contains(&pattern) {
                    patterns.push(pattern);
                }
            }
        }
        patterns.truncate(6);
        patterns
    }

    /// 把「实体[的谓词]」拆成规范实体与可选谓词（最长前缀匹配，支持同义词）
    fn split_entity_predicate(&self, target: &str) -> Option<(String, Option<String>)> {
        let mut best: Option<(usize, String)> = None;
        let mut consider = |name: &str, canonical: &str| {
            if name.is_empty() || !target.starts_with(name) {
                return;
            }
            let length = name.chars().count();
            if best
                .as_ref()
                .map(|(best_length, _)| length > *best_length)
                .unwrap_or(true)
            {
                best = Some((length, canonical.to_string()));
            }
        };
        for profile in self.aligner.profiles() {
            // 前缀预筛：只处理可能命中目标前缀的实体画像
            if profile.kind == TermKind::Entity
                && !profile.term.is_empty()
                && target.starts_with(&profile.term)
            {
                let canonical = self.aligner.canonical_form(&profile.term);
                consider(&profile.term, &canonical);
            }
        }
        for (alias, canonical) in self.aligner.synonyms() {
            if target.starts_with(alias.as_str())
                && self.aligner.kind_of(canonical) == Some(TermKind::Entity)
            {
                consider(alias, canonical);
            }
        }
        let (length, entity) = best?;
        let remainder: String = target.chars().skip(length).collect();
        let remainder = remainder
            .trim()
            .trim_start_matches(['的', '：', ':', ' ', '是']);
        let predicate = if remainder.is_empty() {
            None
        } else {
            Some(self.aligner.canonical_form(remainder))
        };
        Some((entity, predicate))
    }

    pub fn dialogue_summary(&self) -> String {
        let focus = if self.context.focus_entities().is_empty() {
            "（空）".to_string()
        } else {
            self.context.focus_entities().join(" → ")
        };
        let predicate = self
            .context
            .last_predicate()
            .unwrap_or("（无）")
            .to_string();
        let generation = self
            .context
            .generation_state()
            .map(|state| {
                let order = match state.order {
                    GenerationOrder::Relevance => "信息量排序",
                    GenerationOrder::Alphabetical => "主题排序",
                };
                format!("「{}」（已讲 {} 句，{}）", state.entity, state.shown, order)
            })
            .unwrap_or_else(|| "（无）".to_string());
        let session_info =
            if self.session_facts.is_empty() && self.session_negative_facts.is_empty() {
                "（无）".to_string()
            } else {
                format!(
                    "{} 条，会话否定 {} 条",
                    self.session_facts.len(),
                    self.session_negative_facts.len()
                )
            };
        let clarify = self
            .clarification_progress()
            .map(|progress| format!("进行中（{}）", progress))
            .unwrap_or_else(|| "（无）".to_string());
        let lesson_info = if self.lessons.is_empty() {
            "（无）".to_string()
        } else {
            format!("{} 条", self.lessons.len())
        };
        let frame_info = {
            let clarification = match self.frames.clarification() {
                Some(request) => format!("，待绑定澄清「{}」", request.raw),
                None => String::new(),
            };
            match self.frames.last() {
                Some(frame) => format!(
                    "上一帧「{}」、挂账 {} 条{}",
                    frame.display(),
                    self.frames.pending().len(),
                    clarification
                ),
                None => format!(
                    "上一帧（无）、挂账 {} 条{}",
                    self.frames.pending().len(),
                    clarification
                ),
            }
        };
        format!(
            "对话记忆：焦点实体栈 [{}]，上一轮谓词「{}」，上一段生成 {}，历史问题 {} 条，会话假设 {}，会话规则 {} 条，澄清 {}，中文概率知识 {}，话语帧 {}，悬问 {} 条（搁置 {}）。",
            focus,
            predicate,
            generation,
            self.context.history().len(),
            session_info,
            self.session_clauses.len(),
            clarify,
            lesson_info,
            frame_info,
            self.pending_questions.len(),
            self.pending_questions.shelved_len()
        )
    }

    /// 会话记忆的完整内存快照（焦点/历史/生成状态/会话事实/教学/规则/话语帧）。
    ///
    /// 与 [`Self::save_dialogue_state_with_contrast`] 落盘的内容一致（除对比仲裁
    /// 记忆外），供服务层在不触碰文件系统的前提下导出/导入会话。
    pub fn dialogue_snapshot(&self) -> DialogueSnapshot {
        let mut snapshot = self.context.snapshot();
        snapshot.session_facts = self
            .session_facts
            .iter()
            .filter(|atom| atom.is_ground())
            .map(|atom| SnapshotAtom {
                predicate: atom.predicate.to_string(),
                terms: atom
                    .terms
                    .iter()
                    .map(|term| term.name().to_string())
                    .collect(),
            })
            .collect();
        snapshot.session_negative_facts = self
            .session_negative_facts
            .iter()
            .filter(|atom| atom.is_ground())
            .map(|atom| SnapshotAtom {
                predicate: atom.predicate.to_string(),
                terms: atom
                    .terms
                    .iter()
                    .map(|term| term.name().to_string())
                    .collect(),
            })
            .collect();
        snapshot.slots = (!self.slots.is_empty()).then(|| self.slots.clone());
        snapshot.lessons = self.lessons.clone();
        snapshot.session_clauses = self.session_clauses.iter().map(snapshot_clause).collect();
        snapshot.frames = Some(self.frames.clone());
        snapshot.pending_questions =
            (!self.pending_questions.is_empty()).then(|| self.pending_questions.clone());
        snapshot.batch_pending = self.batch_pending.clone();
        snapshot.ontology = self.discourse.ontology_capsule();
        snapshot.completion = (!self.completion.is_empty()).then(|| self.completion.clone());
        snapshot.predictive = (!self.predictive.is_empty()).then(|| self.predictive.clone());
        snapshot.foil = self.foil_ledger.has_decisions().then(|| self.foil_ledger.clone());
        snapshot.assumptions = self
            .assumption_world
            .iter()
            .flat_map(|world| world.assumptions.iter())
            .filter(|atom| atom.is_ground())
            .map(|atom| SnapshotAtom {
                predicate: atom.predicate.to_string(),
                terms: atom
                    .terms
                    .iter()
                    .map(|term| term.name().to_string())
                    .collect(),
            })
            .collect();
        snapshot.session_receipts = self
            .session_receipts
            .iter()
            .map(SessionRevisionReceipt::to_snapshot)
            .collect();
        snapshot.pending_hypotheses = self
            .abduction
            .pending()
            .iter()
            .map(snapshot_hypothesis)
            .collect();
        snapshot.clarification = self.clarification.as_ref().map(snapshot_clarification);
        snapshot.last_abduction_goal = self
            .last_abduction_goal
            .iter()
            .flat_map(|goals| goals.iter())
            .map(snapshot_clause_atom)
            .collect();
        snapshot.last_abduction_vars = self.last_abduction_vars.clone();
        snapshot.alignment =
            (!self.linker.alignment().is_empty()).then(|| self.linker.alignment().clone());
        snapshot.epistemic_turns =
            (!self.epistemic_turns.is_empty()).then(|| self.epistemic_turns.clone());
        snapshot.beliefs = (!self.beliefs.is_empty()).then(|| self.beliefs.clone());
        snapshot
    }
}

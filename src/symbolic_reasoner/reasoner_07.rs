impl SymbolicReasoner {
    /// 图谱问句词表：实体、类型、谓词的规范形与已确认别名。
    ///
    /// 别名只在“规范形确实是图中的实体/类型/谓词”时才登记，避免把
    /// 关系别名误当实体、或把未落图的配置名伪装成已知词。
    fn graph_lexicon(&self) -> PatternLexicon {
        let mut lexicon = PatternLexicon::new();
        let predicates = self.generator.all_predicates();
        let mut known_types: BTreeSet<String> = self.generator.all_types().into_iter().collect();
        known_types.extend(
            self.generator
                .typed_entities()
                .into_iter()
                .map(|(_, entity_type)| entity_type)
                .filter(|entity_type| !entity_type.trim().is_empty()),
        );
        for triple in self.predictor.triples() {
            if triple.predicate == "type" && !triple.object.trim().is_empty() {
                known_types.insert(triple.object.clone());
            }
        }
        // 己登记的子类父类型也要进词表，否则「电影」问句会在编译期就失配，
        // 根本走不到类型闭包已经支持的那一步。
        {
            let retrieval = self.retrieval.lock();
            let supertypes: Vec<String> = known_types
                .iter()
                .flat_map(|known| retrieval.type_ancestors(known))
                .collect();
            known_types.extend(supertypes);
        }
        for entity in self.generator.all_entities() {
            lexicon.add_entity(&entity, &entity);
        }
        for entity_type in &known_types {
            lexicon.add_type(entity_type, entity_type);
        }
        for (entity, entity_type) in self.generator.typed_entities() {
            if !entity_type.trim().is_empty() {
                lexicon.add_type(&entity_type, &entity_type);
            }
            lexicon.add_entity(&entity, &entity);
        }
        for predicate in &predicates {
            lexicon.add_predicate(predicate, predicate);
        }
        for (surface, canonical) in self.generator.predicate_surfaces() {
            lexicon.add_predicate(surface, canonical);
        }
        for (surface, (canonical, inverted)) in &self.inverse_query_map {
            if !*inverted {
                lexicon.add_predicate(surface, canonical);
            }
        }
        for (alias, canonical) in self.aligner.synonyms() {
            if self.is_known_entity(canonical) {
                lexicon.add_entity(alias, canonical);
            } else if known_types.contains(canonical) {
                lexicon.add_type(alias, canonical);
            } else if predicates.iter().any(|predicate| predicate == canonical)
                || self.predictor.has_predicate(canonical)
            {
                lexicon.add_predicate(alias, canonical);
            }
        }
        lexicon
    }

    fn graph_facts(&self) -> Vec<GraphFact> {
        let mut facts = GraphFact::from_triples(self.predictor.triples());
        for (entity, entity_type) in self.generator.typed_entities() {
            if !entity_type.trim().is_empty() {
                facts.push(GraphFact::type_assertion(entity, entity_type));
            }
        }
        for ((entity, key), values) in &self.attribute_index {
            for value in values {
                facts.push(GraphFact::attribute(entity, key, value));
            }
        }
        for atom in self.positive_session_atoms() {
            if atom.terms.len() == 2
                && !atom.terms[0].is_var()
                && !atom.terms[1].is_var()
                && atom.predicate.as_ref() != PROOF_TYPE
            {
                facts.push(GraphFact::relation(
                    atom.terms[0].name(),
                    atom.predicate.as_ref(),
                    atom.terms[1].name(),
                ));
            }
        }
        // 类型闭包只在本轮图谱匹配中物化，不写回知识图谱：查询「电影」
        // 能命中显式断言的「科幻片」，依据仍是检索层已登记的子类边。
        let mut inferred = Vec::new();
        {
            let retrieval = self.retrieval.lock();
            for fact in &facts {
                if fact.predicate != "type" {
                    continue;
                }
                for ancestor in retrieval.type_ancestors(&fact.object) {
                    inferred.push(GraphFact::type_assertion(&fact.subject, ancestor));
                }
            }
        }
        facts.extend(inferred);
        // 槽级压制与否定阻断同样作用于结构化图谱问答：会话赢家存在时，
        // 图谱旧值不参与匹配；它仍保留在图谱/账本里，撤回改口后自动恢复。
        // 过滤必须放在类型闭包之后，否则「某实体不是某祖先类型」仍会从
        // 显式子类边物化出被否定的祖先。
        facts.retain(|fact| {
            let atom = Atom::rel(&fact.predicate, &fact.subject, &fact.object);
            // 类型断言在图谱里用内部谓词 `type`，而自然语言否定解析成「是」；
            // 两种写法必须指向同一条可阻断边。
            let type_negative = fact.predicate == "type"
                && self.negative_atom_blocks(&Atom::rel("是", &fact.subject, &fact.object));
            !self.is_suppressed_slot_value(&fact.subject, &fact.predicate, &fact.object)
                && !self.negative_atom_blocks(&atom)
                && !type_negative
        });
        facts.sort_by(|left, right| {
            (
                &left.subject,
                &left.predicate,
                &left.object,
                left.source as u8,
            )
                .cmp(&(
                    &right.subject,
                    &right.predicate,
                    &right.object,
                    right.source as u8,
                ))
        });
        facts.dedup();
        facts
    }

    fn render_graph_question(
        &self,
        compiled: &CompiledGraphQuestion,
        outcome: &PatternOutcome,
        facts: &[GraphFact],
    ) -> String {
        let select = &compiled.pattern.select;
        let total = outcome.solutions.len();
        if compiled.asks_for_count() {
            let mut output = match compiled.kind {
                GraphQuestionKind::Conjunction => {
                    if total == 0 {
                        "没有找到同时满足这些条件的实体。".to_string()
                    } else {
                        format!("满足全部条件的共有 {} 个。", total)
                    }
                }
                GraphQuestionKind::Existential => {
                    if total == 0 {
                        "没有找到满足这些条件的实体。".to_string()
                    } else {
                        format!("满足条件的实体共有 {} 个。", total)
                    }
                }
                GraphQuestionKind::Equality => {
                    let subjects = describe_graph_subjects(&compiled.subject_variables);
                    let predicate = compiled.focus_predicate.as_deref().unwrap_or("共同值");
                    if total == 0 {
                        format!("{}没有共同的{}。", subjects, predicate)
                    } else {
                        format!("{}共有 {} 个共同的{}。", subjects, total, predicate)
                    }
                }
                GraphQuestionKind::SharedObject => {
                    let subjects = compiled.subject_variables.join("、");
                    let predicate = compiled.focus_predicate.as_deref().unwrap_or("指向");
                    if total == 0 {
                        format!("没有找到「{}」都{}的对象。", subjects, predicate)
                    } else {
                        format!("「{}」都{}的对象共有 {} 个。", subjects, predicate, total)
                    }
                }
            };
            if outcome.truncated {
                output.push_str("（搜索预算已用尽，结果为下界）");
            }
            return output;
        }
        let mut rows: Vec<String> = outcome
            .solutions
            .iter()
            .map(|solution| solution.row(select))
            .filter(|row| !row.is_empty())
            .collect();
        // 首屏与后续翻页共用同一页大小，避免「还有呢」重复展示上一屏末尾。
        const MAX_ROWS: usize = crate::conversational_graph::DEFAULT_PAGE_SIZE;
        let hidden = rows.len().saturating_sub(MAX_ROWS);
        rows.truncate(MAX_ROWS);
        let mut output = match compiled.kind {
            GraphQuestionKind::Conjunction => {
                if rows.is_empty() {
                    "没有找到同时满足这些条件的实体。".to_string()
                } else {
                    format!("满足全部条件的有 {} 个：{}。", total, rows.join("、"))
                }
            }
            GraphQuestionKind::Existential => {
                if rows.is_empty() {
                    "没有找到满足这些条件的实体。".to_string()
                } else {
                    format!("存在满足条件的实体 {} 个：{}。", total, rows.join("、"))
                }
            }
            GraphQuestionKind::Equality => {
                let subjects = describe_graph_subjects(&compiled.subject_variables);
                let predicate = compiled.focus_predicate.as_deref().unwrap_or("共同值");
                if rows.is_empty() {
                    format!("没有找到{}共同的{}。", subjects, predicate)
                } else {
                    format!("{}共同的{}：{}。", subjects, predicate, rows.join("、"))
                }
            }
            GraphQuestionKind::SharedObject => {
                let subjects = compiled.subject_variables.join("、");
                let predicate = compiled.focus_predicate.as_deref().unwrap_or("指向");
                if rows.is_empty() {
                    format!("没有找到「{}」都{}的对象。", subjects, predicate)
                } else {
                    format!("「{}」都{}：{}。", subjects, predicate, rows.join("、"))
                }
            }
        };
        if hidden > 0 {
            output.push_str(&format!("（另有 {} 项未展开）", hidden));
        }
        if outcome.truncated {
            output.push_str("（搜索预算已用尽，结果为下界）");
        }
        if total <= 3 && !outcome.solutions.is_empty() {
            let lines = outcome.solutions[0].witness_lines(facts);
            if !lines.is_empty() {
                output.push_str("\n依据：");
                output.push_str(&lines.join("；"));
            }
        }
        output
    }

    /// 直接链没有命中时，把前向推理的产物当作**本会话内的推导**回答。
    ///
    /// 推导只读、按需重算：不写回图谱、会话事实或槽账本。前提被更正后，
    /// 旧结论会随下一次 `forward_chain` 自然消失，而不是留下无法撤销的幽灵
    /// 事实。这里只接受单谓词问句，避免把多跳链的中间结论误当成最终答案。
    fn answer_by_derivation(
        &self,
        entity: &str,
        predicates: &[ChainPredicate],
        _question: &str,
    ) -> Option<String> {
        if !self.config.enable_resolution || predicates.len() != 1 {
            return None;
        }
        let predicate = &predicates[0];
        if entity.trim().is_empty() || predicate.slot.trim().is_empty() {
            return None;
        }
        let derived = self.resolver.forward_chain(3, 100);
        if derived.is_empty() {
            return None;
        }
        // 只为解释「这个结论从哪来」保留一层前提→推导的索引，不把推导当事实。
        let mut by_atom: HashMap<&Atom, &DerivedFact> = HashMap::new();
        for fact in &derived {
            by_atom.entry(&fact.atom).or_insert(fact);
        }
        let mut matches: Vec<(&DerivedFact, String)> = derived
            .iter()
            .filter_map(|fact| {
                self.derived_value_for(fact, entity, predicate)
                    .map(|value| (fact, value))
            })
            .collect();
        if matches.is_empty() {
            return None;
        }
        matches.sort_by(|(a, a_value), (b, b_value)| {
            a.depth
                .cmp(&b.depth)
                .then_with(|| {
                    b.confidence
                        .partial_cmp(&a.confidence)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| a_value.cmp(b_value))
        });
        let mut seen: HashSet<String> = HashSet::new();
        matches.retain(|(_, value)| seen.insert(value.clone()));
        let limit = 3usize;
        let mut output = String::from("根据当前规则现算的推导（未写入知识图谱）：\n");
        for (index, (fact, value)) in matches.iter().take(limit).enumerate() {
            if matches.len() == 1 {
                output.push_str(&format!(
                    "  {} 的 {} 是 {}。\n",
                    entity, predicate.surface, value
                ));
            } else {
                output.push_str(&format!(
                    "  {}. {} 的 {} 是 {}。\n",
                    index + 1,
                    entity,
                    predicate.surface,
                    value
                ));
            }
            let supports: Vec<String> = fact
                .premises
                .iter()
                .map(|premise| {
                    let rendered = self.render_atom_nl(premise);
                    match by_atom.get(premise) {
                        Some(parent) if parent.depth >= 2 => {
                            format!("{}（第 {} 轮推出）", rendered, parent.depth)
                        }
                        _ => format!("{}（事实）", rendered),
                    }
                })
                .collect();
            output.push_str(&format!("     依据：{}。\n", supports.join("；")));
            output.push_str(&format!(
                "     规则：{}（第 {} 轮推导，按需重算）。\n",
                fact.rule_source, fact.depth
            ));
        }
        output.push_str(
            "  说明：推导只对本次回答有效；前提被更正后会重新证明，不写入图谱或会话事实。",
        );
        Some(output)
    }

    /// 一条推导是否以 `entity` 为锚点、目标谓词，并返回用户要的取值。
    fn derived_value_for(
        &self,
        fact: &DerivedFact,
        entity: &str,
        predicate: &ChainPredicate,
    ) -> Option<String> {
        // 统计挖掘出的链接/关联规则只产生候选，不构成「已经证明」的答案；
        // 它们继续走原有的候选通道，不能被包装成确定性推导。
        if !Self::derived_rule_is_assertoric(&fact.rule_source) {
            return None;
        }
        // 会话否定事实是阻断边：被否定的前提不能推出结论，被否定的结论
        // 也不能因为规则链而复活。
        if self.derived_fact_is_blocked(fact) {
            return None;
        }
        let fact_predicate = self.aligner.canonical_form(fact.atom.predicate.as_ref());
        let slot_key = self.aligner.canonical_form(&predicate.slot);
        let surface_key = self.aligner.canonical_form(&predicate.surface);
        if fact_predicate != slot_key && fact_predicate != surface_key {
            return None;
        }
        match fact.atom.terms.as_slice() {
            [Term::Const(left), Term::Const(right)] => {
                let anchor = if predicate.inverted { right } else { left };
                if self.aligner.canonical_form(anchor) != self.aligner.canonical_form(entity) {
                    return None;
                }
                Some(if predicate.inverted {
                    left.as_ref().to_string()
                } else {
                    right.as_ref().to_string()
                })
            }
            _ => None,
        }
    }

    /// 只有教学/补全/手工规则算「证明」；统计挖掘规则仍只算候选。
    fn derived_rule_is_assertoric(source: &str) -> bool {
        !source.starts_with("链接预测规则") && !source.starts_with("关联规则")
    }

    /// 推导的结论或任一前提是否已被否定事实/槽压制排除。
    fn derived_fact_is_blocked(&self, fact: &DerivedFact) -> bool {
        std::iter::once(&fact.atom)
            .chain(fact.premises.iter())
            .any(|atom| self.negative_atom_blocks(atom) || self.atom_suppressed_by_slot(atom))
    }

    /// 槽账本里已有更新的赢家时，图谱写下的旧值只是被压制，不是被删除；
    /// 推导层必须和直接回答看到同一套可见取值，否则会把旧值重新推出来。
    fn atom_suppressed_by_slot(&self, atom: &Atom) -> bool {
        let [Term::Const(subject), Term::Const(object)] = atom.terms.as_slice() else {
            return false;
        };
        let (slot_subject, slot_predicate) = self.slot_identity(subject, atom.predicate.as_ref());
        let accepted = self.slots.accepted(&slot_subject, &slot_predicate);
        if accepted.is_empty() {
            return false;
        }
        let object = self.aligner.canonical_form(object);
        self.slots
            .suppressed(&slot_subject, &slot_predicate)
            .iter()
            .any(|value| self.aligner.canonical_form(value) == object)
            && !accepted
                .iter()
                .any(|value| self.aligner.canonical_form(value) == object)
    }

    /// 规范化比较的否定阻断：词面同义词不应让阻断边漏过去。
    fn negative_atom_blocks(&self, atom: &Atom) -> bool {
        let [Term::Const(subject), Term::Const(object)] = atom.terms.as_slice() else {
            return false;
        };
        let subject = self.aligner.canonical_form(subject);
        let predicate = self.aligner.canonical_form(atom.predicate.as_ref());
        let object = self.aligner.canonical_form(object);
        self.session_negative_facts.iter().any(|negative| {
            negative.terms.len() == 2
                && self.aligner.canonical_form(negative.predicate.as_ref()) == predicate
                && self.aligner.canonical_form(negative.terms[0].name()) == subject
                && self.aligner.canonical_form(negative.terms[1].name()) == object
        })
    }

    /// 在当前前向闭包里找一条能证明 `entity 的 predicate 是 expected` 的推导。
    fn derived_fact_for(
        &self,
        entity: &str,
        predicate: &ChainPredicate,
        expected: &str,
    ) -> Option<DerivedFact> {
        if !self.config.enable_resolution {
            return None;
        }
        let expected_key = self.aligner.canonical_form(expected);
        self.resolver
            .forward_chain(3, 100)
            .into_iter()
            .find(|fact| {
                self.derived_value_for(fact, entity, predicate)
                    .map(|value| self.aligner.canonical_form(&value) == expected_key)
                    .unwrap_or(false)
            })
    }

    /// 判断/是非问句里的一行推导回执：证明依据与规则来源，不写回知识。
    fn render_derived_support(&self, fact: &DerivedFact) -> String {
        let supports: Vec<String> = fact
            .premises
            .iter()
            .map(|premise| self.render_atom_nl(premise))
            .collect();
        format!(
            "  依据：{}。\n  规则：{}（第 {} 轮推导，按需重算）。",
            supports.join("；"),
            fact.rule_source,
            fact.depth
        )
    }

    fn answer_planned(&self, question: &str) -> Option<String> {
        let compiled = self.answer_compiled(question);
        let should_anchor = match compiled.as_deref() {
            None => true,
            Some(answer) => Self::is_missing_record(answer),
        };
        if !should_anchor {
            return compiled;
        }
        // 已采纳的习得重写是显式规则：在通用锚定/实体画像回退之前竞争，
        // 避免一条本该走规则的问题被实体画像吞掉。
        let mut rewrite_retrieval_attempted = false;
        if self.config.rewrite_retrieval.enabled && self.learned_rewrite_available(question) {
            rewrite_retrieval_attempted = true;
            if let Some(answer) = self.answer_rewrite_retrieval(question) {
                return Some(answer);
            }
        }
        // 精确路径只给出「没有记录」时，错别字/别名问句仍应有机会走锚定；
        // 歧义、无法证明等失败不进入，避免绕过澄清或证明边界。若锚定也
        // 拿不到确证答案，再保留原失败答复。
        if let Some(answer) = self.answer_anchored(question) {
            return Some(answer);
        }
        // 最后让多个候选读法用稠密检索证据竞争；出厂配置启用，且只在
        // 精确/锚定路径失败后接管，因此不会改变已能确证的旧行为。
        if self.config.rewrite_retrieval.enabled && !rewrite_retrieval_attempted {
            if let Some(answer) = self.answer_rewrite_retrieval(question) {
                return Some(answer);
            }
        }
        compiled
    }

    fn learned_rewrite_available(&self, question: &str) -> bool {
        self.rewrite_book
            .as_ref()
            .is_some_and(|book| book.resolve(question, &self.relation_lexicon()).is_some())
    }

    pub(crate) fn answer_learned_rewrite(&self, question: &str) -> Option<String> {
        if !self.config.rewrite_retrieval.enabled || !self.learned_rewrite_available(question) {
            return None;
        }
        self.answer_rewrite_retrieval(question)
    }

    /// 收集问句的候选读法：精确计划、句法重写、指称锚定。
    ///
    /// 顺序只影响报告的可读性；去重与排序交给
    /// [`rewrite_retrieval::assess`]，同一计划保留最可靠的来源。
    fn rewrite_retrieval_context_bonus(&self, plan: &QueryPlan) -> f64 {
        let last_predicate = self.context.last_predicate();
        let focus = self.context.focus_entities();
        let mut bonus: f64 = 0.0;
        for predicate in rewrite_retrieval::plan_predicates(plan) {
            if last_predicate.is_some_and(|last| last == predicate) {
                bonus += 0.5;
            }
        }
        for entity in rewrite_retrieval::plan_entities(plan) {
            if focus.iter().any(|focused| focused == entity) {
                bonus += 0.25;
            }
        }
        bonus.clamp(-2.0, 2.0)
    }

    pub fn rewrite_retrieval_candidates(&self, question: &str) -> Vec<HypothesisCandidate> {
        let mut candidates = Vec::new();
        if let Some(plan) = self.planner.parse(question) {
            let bonus = self.rewrite_retrieval_context_bonus(&plan);
            candidates.push(
                HypothesisCandidate::new(HypothesisSource::ExactPlan, "planner", question, plan)
                    .with_context_bonus(bonus),
            );
        }
        let lexicon = self.relation_lexicon();
        if let Some(rewrite) = query_rewrite::rewrite(question, &lexicon) {
            let bonus = self.rewrite_retrieval_context_bonus(&rewrite.plan);
            candidates.push(
                HypothesisCandidate::new(
                    HypothesisSource::SyntaxRewrite,
                    rewrite.rule,
                    question,
                    rewrite.plan,
                )
                .with_context_bonus(bonus),
            );
        }
        if let Some(book) = &self.rewrite_book {
            if let Some(application) = book.resolve(question, &lexicon) {
                let plan = self.planner.parse(&application.text).or_else(|| {
                    query_rewrite::rewrite(&application.text, &lexicon).map(|rewrite| rewrite.plan)
                });
                if let Some(plan) = plan {
                    let bonus = self.rewrite_retrieval_context_bonus(&plan);
                    candidates.push(
                        HypothesisCandidate::new(
                            HypothesisSource::LearnedRewrite,
                            application.rules.join("+"),
                            application.text,
                            plan,
                        )
                        .with_context_bonus(bonus),
                    );
                }
            }
        }
        let hints = self.link_hints(question);
        let capsule = self.discourse.ontology_capsule();
        let evidence = self.anchor_evidence(capsule.as_ref());
        for variant in
            query_anchor::anchor_questions(question, &self.linker, &hints, &lexicon, &evidence)
        {
            let Some(plan) = variant.preferred_plan() else {
                continue;
            };
            let bonus = self.rewrite_retrieval_context_bonus(plan);
            let rule = variant
                .replacements
                .first()
                .map(|replacement| format!("anchor:{}", replacement.surface))
                .unwrap_or_else(|| "anchor".to_string());
            candidates.push(
                HypothesisCandidate::new(
                    HypothesisSource::AnchoredPlan,
                    rule,
                    variant.question.clone(),
                    plan.clone(),
                )
                .with_context_bonus(bonus),
            );
        }
        candidates
    }

    /// 对候选读法做证据裁决（只读；不执行计划、不写知识库）。
    pub fn rewrite_retrieval_assessment(
        &self,
        question: &str,
    ) -> Option<rewrite_retrieval::Assessment> {
        let candidates = self.rewrite_retrieval_candidates(question);
        if candidates.is_empty() {
            return None;
        }
        let schema = crate::query_canonical::CanonicalSchema::from_law_book(&self.laws);
        Some(rewrite_retrieval::assess_with_schema(
            question,
            candidates,
            &self.config.rewrite_retrieval,
            &schema,
            |query, k, constraints| {
                self.evidence_search(query, k, constraints, SearchBudget::default())
            },
        ))
    }

    /// 读法裁决的可打印回执（只读审计）。
    pub fn rewrite_retrieval_report(&self, question: &str) -> Option<String> {
        self.rewrite_retrieval_assessment(question)
            .map(|assessment| assessment.render_report())
    }

    /// 执行一个读法候选，返回最终回答（不写知识库）。
    pub fn answer_hypothesis(&self, hypothesis: &HypothesisCandidate) -> Option<String> {
        self.answer_plan(&hypothesis.plan, &hypothesis.question)
    }

    /// 把可执行的候选读法转成会话澄清帧，让下一轮序号选择真正落地。
    fn rewrite_retrieval_clarification(
        &self,
        question: &str,
        assessment: &rewrite_retrieval::Assessment,
    ) -> Option<FormalClarification> {
        let mut options = Vec::new();
        for hypothesis in assessment
            .hypotheses
            .iter()
            .filter(|hypothesis| hypothesis.answer.is_some())
        {
            if options.len() >= 3 {
                break;
            }
            let Some(answer) = self.answer_hypothesis(&hypothesis.candidate) else {
                continue;
            };
            if Self::is_pending_failure(&answer) {
                continue;
            }
            let evidence = hypothesis
                .answers
                .iter()
                .take(2)
                .map(|cluster| format!("{}（{:.2}）", cluster.answer, cluster.weight))
                .collect();
            options.push(FormalClarificationOption {
                logical_form: hypothesis.candidate.receipt(),
                answer,
                evidence,
            });
        }
        (options.len() >= 2).then(|| FormalClarification {
            question: question.trim().to_string(),
            options,
        })
    }

    fn render_rewrite_retrieval_clarification(
        &self,
        assessment: &rewrite_retrieval::Assessment,
    ) -> String {
        let mut output =
            String::from("这个问句有不止一种理解，也有不止一种有证据的读法，需要先确认：\n");
        for (index, hypothesis) in assessment
            .hypotheses
            .iter()
            .filter(|hypothesis| hypothesis.answer.is_some())
            .take(3)
            .enumerate()
        {
            output.push_str(&format!(
                "  {}. {}（答案：{}；置信度 {:.2}）\n",
                index + 1,
                hypothesis.candidate.receipt(),
                hypothesis.answer.as_deref().unwrap_or("未定"),
                hypothesis.confidence
            ));
        }
        output.push_str("请补充主语/谓词，或直接说明你指的是哪一种。");
        output
    }

    /// 证据裁决后的读法执行：只有 `Accept` 才执行计划，`Clarify` 先问一句。
    fn answer_rewrite_retrieval(&self, question: &str) -> Option<String> {
        let assessment = self.rewrite_retrieval_assessment(question)?;
        self.semantic.record_assessment(&assessment);
        if self.semantic.advice().prefer_clarification
            && assessment.decision == RewriteRetrievalDecision::Accept
        {
            if let Some(clarification) =
                self.rewrite_retrieval_clarification(question, &assessment)
            {
                self.perf.bump("semantic_control_clarify");
                let prompt = clarification.prompt();
                *self.formal_clarification.lock() = Some(clarification);
                return Some(prompt);
            }
        }
        match assessment.decision {
            RewriteRetrievalDecision::Accept => {
                let selected = assessment.selected_hypothesis()?;
                let answer = self.answer_hypothesis(&selected.candidate)?;
                if Self::is_pending_failure(&answer) {
                    return None;
                }
                self.perf.bump("rewrite_retrieval_accept");
                Some(answer)
            }
            RewriteRetrievalDecision::Clarify => {
                self.perf.bump("rewrite_retrieval_clarify");
                if let Some(clarification) =
                    self.rewrite_retrieval_clarification(question, &assessment)
                {
                    let prompt = clarification.prompt();
                    *self.formal_clarification.lock() = Some(clarification);
                    return Some(prompt);
                }
                Some(self.render_rewrite_retrieval_clarification(&assessment))
            }
            RewriteRetrievalDecision::Abstain => None,
        }
    }

    /// 精确词表路径：句法规划器优先，其次句法重写器。
    fn answer_compiled(&self, question: &str) -> Option<String> {
        let parsed = self.planner.parse(question);
        if let Some(answer) = parsed
            .as_ref()
            .and_then(|plan| self.answer_plan(plan, question))
        {
            return Some(answer);
        }
        // 已有计划可能只解析了句子的前半段（例如「X 有没有 V Y」被误读成
        // 存在性问句）。此时用句法重写器再编译一次，词表校验不过就弃权。
        let rewritten = query_rewrite::rewrite(question, &self.relation_lexicon())?;
        if parsed.as_ref() == Some(&rewritten.plan) {
            return None;
        }
        self.answer_plan(&rewritten.plan, question)
    }

    /// 精确路径失败后的指称锚定回退：用实体链接器的候选把错别字/别名
    /// 替换成规范名，再走同一套规划与执行。歧义、类型互斥或只得到
    /// 「没有记录」时一律放弃，让后续预测/证据路径继续工作。
    fn answer_anchored(&self, question: &str) -> Option<String> {
        let hints = self.link_hints(question);
        let capsule = self.discourse.ontology_capsule();
        let evidence = self.anchor_evidence(capsule.as_ref());
        let lexicon = self.relation_lexicon();
        let variants =
            query_anchor::anchor_questions(question, &self.linker, &hints, &lexicon, &evidence);
        for variant in variants {
            let mut answer = variant
                .planner_plan
                .as_ref()
                .and_then(|plan| self.answer_plan(plan, &variant.question));
            if answer.is_none() {
                if let Some(rewrite) = &variant.rewrite {
                    if variant.planner_plan.as_ref() != Some(&rewrite.plan) {
                        answer = self.answer_plan(&rewrite.plan, &variant.question);
                    }
                }
            }
            let Some(answer) = answer else {
                continue;
            };
            if Self::is_pending_failure(&answer) {
                continue;
            }
            self.perf.bump("query_anchor_accept");
            return Some(answer);
        }
        None
    }

    /// 锚定重写回执（只读审计；无可用变体时返回 None）。
    pub fn query_anchor_receipt(&self, question: &str) -> Option<String> {
        let hints = self.link_hints(question);
        let capsule = self.discourse.ontology_capsule();
        let evidence = self.anchor_evidence(capsule.as_ref());
        let lexicon = self.relation_lexicon();
        query_anchor::anchor_questions(question, &self.linker, &hints, &lexicon, &evidence)
            .first()
            .map(|variant| variant.receipt())
    }

    fn answer_plan(&self, plan: &QueryPlan, question: &str) -> Option<String> {
        let plan = plan.clone();
        match plan {
            QueryPlan::FindObject { segments } => {
                let (entity, predicates) = self.split_chain(&segments)?;
                if predicates.is_empty() {
                    return None;
                }
                if self.config.enable_clarification
                    && predicates.len() == 1
                    && self.is_known_entity(&entity)
                {
                    if let Some(decision) =
                        self.attribute_alignment_decision(&entity, &predicates[0].surface)
                    {
                        if decision.should_clarify() {
                            return Some(self.render_alignment_clarification(
                                &predicates[0].surface,
                                &decision,
                            ));
                        }
                    }
                }
                if predicates.len() >= 2 {
                    if let Some(answer) =
                        self.answer_chain_with_paths(&entity, &predicates, question)
                    {
                        return Some(answer);
                    }
                }
                // 单个传递谓词的直接问句也必须并入有界闭包：否则
                // 「A 的 P 是谁」只返回直接边，声明的传递性在回答里没有兑现。
                let single_transitive = predicates.len() == 1
                    && self.laws.active_transitive(&predicates[0].slot).is_some();
                let chain_result = if self.is_transitive_chain(&predicates) || single_transitive {
                    // 重复的传递谓词：用 Closure 算子压缩链，trace 会引用该律。
                    self.run_chain_via_laws(&entity, &predicates)
                } else {
                    self.run_chain(&entity, &predicates).or_else(|| {
                        if self.laws.has_inference_laws() {
                            self.run_chain_via_laws(&entity, &predicates)
                        } else {
                            None
                        }
                    })
                };
                if let Some(run) = chain_result {
                    return Some(self.generate_find_object(
                        &entity,
                        &predicates,
                        &run.steps,
                        &run.last_hits,
                        question,
                        run.closure_truncated,
                        run.construction_truncated,
                        run.construction_facts,
                    ));
                }
                // 直接边/链接预测都没有命中时，再问一次归结引擎：规则推出来的
                // 结论也可以当答案，但只作为本轮推导，不物化成事实。
                if let Some(answer) = self.answer_by_derivation(&entity, &predicates, question) {
                    return Some(answer);
                }
                if self.is_known_entity(&entity)
                    && predicates
                        .iter()
                        .all(|p| self.is_known_predicate(&p.surface))
                {
                    return Some(format!(
                        "根据现有知识库，没有记录「{} 的 {}」。",
                        entity,
                        predicates
                            .iter()
                            .map(|p| p.surface.as_str())
                            .collect::<Vec<_>>()
                            .join(" 的 ")
                    ));
                }
                None
            }
            QueryPlan::Count { subject, predicate } => {
                let classifier = crate::quantifier::requested_classifier(question)
                    .or_else(|| self.learned_count_classifier(&predicate));
                self.answer_count(&subject, &predicate, false, classifier.as_deref())
            }
            QueryPlan::Exist { subject, predicate } => {
                let classifier = crate::quantifier::requested_classifier(question)
                    .or_else(|| self.learned_count_classifier(&predicate));
                self.answer_count(&subject, &predicate, true, classifier.as_deref())
            }
            QueryPlan::Closure {
                subject,
                predicate,
                operator,
                count_only,
                then,
            } => self.answer_closure(&subject, &predicate, operator, count_only, &then),
            QueryPlan::CheckTriple {
                subject_segments,
                object,
            } => self.answer_check(&subject_segments, &object),
            QueryPlan::CheckRelation {
                subject,
                predicate,
                object,
                negated,
            } => self.answer_check_relation(&subject, &predicate, &object, negated),
            QueryPlan::FindSubject { object, predicate } => {
                self.answer_find_subject(&object, &predicate)
            }
            QueryPlan::FindObjectByPredicate { subject, predicate } => {
                self.answer_relation_objects(&subject, &predicate)
            }
            QueryPlan::FindObjectExcept {
                subject,
                predicate,
                excluded,
            } => self.answer_relation_objects_filtered(&subject, &predicate, &excluded),
            QueryPlan::FindSubjectExcept {
                object,
                predicate,
                excluded,
            } => self.answer_find_subject_filtered(&object, &predicate, &excluded),
            QueryPlan::PathBetween {
                start,
                goal,
                max_hops,
            } => self.answer_path_between(&start, &goal, max_hops),
            QueryPlan::CommonNeighbor { left, right } => self.answer_common_neighbor(&left, &right),
            QueryPlan::Intermediary {
                start,
                goal,
                max_hops,
            } => self.answer_intermediary(&start, &goal, max_hops),
            QueryPlan::Similar { entity } => self.answer_similar(&entity),
            QueryPlan::Analogy { a, b, c } => self.answer_analogy(&a, &b, &c),
        }
    }

    fn closure_adjacency(&self, slot: &str, inverted: bool) -> BTreeMap<String, BTreeSet<String>> {
        fn add_edge(adjacency: &mut BTreeMap<String, BTreeSet<String>>, from: &str, to: &str) {
            adjacency
                .entry(from.to_string())
                .or_default()
                .insert(to.to_string());
        }

        let mut adjacency: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for triple in self.predictor.triples() {
            if triple.predicate != slot {
                continue;
            }
            if inverted {
                add_edge(&mut adjacency, &triple.object, &triple.subject);
            } else {
                add_edge(&mut adjacency, &triple.subject, &triple.object);
            }
        }
        for atom in self.positive_session_atoms() {
            if atom.predicate.as_ref() != slot || atom.terms.len() != 2 {
                continue;
            }
            if let (Term::Const(subject), Term::Const(object)) = (&atom.terms[0], &atom.terms[1]) {
                if inverted {
                    add_edge(&mut adjacency, object, subject);
                } else {
                    add_edge(&mut adjacency, subject, object);
                }
            }
        }
        for atom in &self.session_negative_facts {
            if atom.predicate.as_ref() != slot || atom.terms.len() != 2 {
                continue;
            }
            if let (Term::Const(subject), Term::Const(object)) = (&atom.terms[0], &atom.terms[1]) {
                let (from, to) = if inverted {
                    (object.as_ref(), subject.as_ref())
                } else {
                    (subject.as_ref(), object.as_ref())
                };
                if let Some(values) = adjacency.get_mut(from) {
                    values.remove(to);
                }
            }
        }
        adjacency
    }

    fn answer_closure(
        &self,
        subject: &str,
        predicate: &str,
        operator: ClosureOp,
        count_only: bool,
        then: &[String],
    ) -> Option<String> {
        let entity = self.aligner.canonical_form(subject.trim());
        let surface = predicate.trim();
        if entity.is_empty() || surface.is_empty() {
            return None;
        }
        if !self.is_known_entity(&entity) && !self.predictor.participates(&entity) {
            return None;
        }
        let (slot, inverted) = self.resolve_question_predicate(surface);
        if slot.is_empty() || (!self.is_known_predicate(&slot) && !self.is_known_predicate(surface))
        {
            return None;
        }
        let mut then_links: Vec<(String, String, bool)> = Vec::new();
        for raw in then {
            let raw = raw.trim();
            let (then_slot, then_inverted) = self.resolve_question_predicate(raw);
            if then_slot.is_empty()
                || (!self.is_known_predicate(&then_slot) && !self.is_known_predicate(raw))
            {
                return None;
            }
            then_links.push((raw.to_string(), then_slot, then_inverted));
        }

        let adjacency = self.closure_adjacency(&slot, inverted);
        let law = if operator.requires_transitive_law() {
            match self.laws.active_transitive(&slot) {
                Some(law) => Some(law.id.clone()),
                None => {
                    let direct: Vec<String> = adjacency
                        .get(&entity)
                        .map(|values| values.iter().cloned().collect())
                        .unwrap_or_default();
                    let direct_text = if direct.is_empty() {
                        "当前没有直接记录".to_string()
                    } else {
                        format!("当前只有直接记录：{}", direct.join("、"))
                    };
                    return Some(format!(
                        "无法安全回答「{}」的所有「{}」：谓词「{}」尚未声明为传递关系。{}。若要让「所有」按可达闭包回答，请先在语义律中启用传递律。",
                        entity, surface, slot, direct_text
                    ));
                }
            }
        } else {
            None
        };

        let mut budget = ClosureBudget::default();
        if let ClosureOp::Exact(hops) = operator {
            budget.max_depth = budget.max_depth.max(hops);
        }
        if !then_links.is_empty() {
            budget.max_witnesses = budget.max_nodes;
        }
        let outcome = evaluate_closure(&entity, operator, budget, law.as_deref(), |node| {
            adjacency
                .get(node)
                .map(|values| values.iter().cloned().collect())
                .unwrap_or_default()
        })
        .ok()?;
        if then_links.is_empty() {
            return Some(self.render_closure_answer(&entity, surface, &outcome, count_only));
        }
        let composed = self.extend_closure_targets(&outcome, &then_links);
        Some(self.render_closure_chain_answer(
            &entity,
            surface,
            &then_links,
            &outcome,
            count_only,
            &composed,
        ))
    }

    fn extend_closure_targets(
        &self,
        outcome: &ClosureOutcome,
        then: &[(String, String, bool)],
    ) -> Vec<(String, Vec<String>)> {
        let mut current: BTreeMap<String, Vec<String>> = outcome
            .witnesses
            .iter()
            .map(|(target, path)| (target.clone(), path.clone()))
            .collect();
        for (_, slot, inverted) in then {
            let adjacency = self.closure_adjacency(slot, *inverted);
            let mut ordered: Vec<(String, Vec<String>)> = current.into_iter().collect();
            ordered.sort_by(|(left_name, left_path), (right_name, right_path)| {
                left_path
                    .len()
                    .cmp(&right_path.len())
                    .then_with(|| left_name.cmp(right_name))
            });
            let mut next: BTreeMap<String, Vec<String>> = BTreeMap::new();
            for (node, path) in ordered {
                let values: Vec<String> = adjacency
                    .get(&node)
                    .map(|neighbors| neighbors.iter().cloned().collect())
                    .unwrap_or_default();
                for value in values {
                    let mut extended = path.clone();
                    extended.push(value.clone());
                    let replace = next
                        .get(&value)
                        .map_or(true, |existing| existing.len() > extended.len());
                    if replace {
                        next.insert(value, extended);
                    }
                }
            }
            current = next;
            if current.is_empty() {
                break;
            }
        }
        let mut result: Vec<(String, Vec<String>)> = current.into_iter().collect();
        result.sort_by(|(left_name, left_path), (right_name, right_path)| {
            left_path
                .len()
                .cmp(&right_path.len())
                .then_with(|| left_name.cmp(right_name))
        });
        result
    }

    fn render_closure_chain_answer(
        &self,
        entity: &str,
        surface: &str,
        then: &[(String, String, bool)],
        outcome: &ClosureOutcome,
        count_only: bool,
        composed: &[(String, Vec<String>)],
    ) -> String {
        let suffix = then
            .iter()
            .map(|(name, _, _)| format!("的「{}」", name))
            .collect::<String>();
        let target_phrase = format!("所有「{}」{}", surface, suffix);
        let basis = outcome
            .law
            .as_deref()
            .map(|law| format!("；依据 {}", law))
            .unwrap_or_default();
        if composed.is_empty() {
            if outcome.complete() {
                return format!("根据现有知识库，「{}」没有{}。", entity, target_phrase);
            }
            let mut output = format!(
                "根据知识图谱（{}{}）：「{}」暂未找到{}。",
                outcome.operator.label(),
                basis,
                entity,
                target_phrase
            );
            if let Some(truncation) = outcome.truncated {
                output.push_str(&format!(
                    "
  说明：闭包搜索在{}处停止，不能排除仍有更多结果。",
                    truncation.label()
                ));
            }
            return output;
        }

        let count = if outcome.count_is_lower_bound() {
            format!("至少 {} 个", composed.len())
        } else {
            format!("共 {} 个", composed.len())
        };
        let mut output = format!(
            "根据知识图谱（{}{}）：「{}」的{} {}。",
            outcome.operator.label(),
            basis,
            entity,
            target_phrase,
            count
        );
        if !count_only {
            for (index, (target, path)) in composed.iter().take(12).enumerate() {
                output.push_str(&format!(
                    "
  {}. {}（{} 跳：{}）",
                    index + 1,
                    target,
                    path.len().saturating_sub(1),
                    path.join(" → ")
                ));
            }
            if composed.len() > 12 {
                output.push_str(&format!(
                    "
  说明：其余 {} 个结果未展开。",
                    composed.len() - 12
                ));
            }
        } else {
            output.push_str(
                "
  计数口径：不同最终节点，不按路径重复计数。",
            );
        }
        if let Some(truncation) = outcome.truncated {
            output.push_str(&format!(
                "
  说明：闭包搜索在{}处停止，结果不完整，上述计数是下界（至少 {} 个）。",
                truncation.label(),
                composed.len()
            ));
        }
        output
    }

    fn render_closure_answer(
        &self,
        entity: &str,
        surface: &str,
        outcome: &ClosureOutcome,
        count_only: bool,
    ) -> String {
        let target_phrase = match outcome.operator {
            ClosureOp::Exact(hops) => format!("恰好 {} 跳的「{}」", hops, surface),
            ClosureOp::Plus => format!("所有「{}」", surface),
            ClosureOp::Star => format!("所有（含自身）「{}」", surface),
        };
        if outcome.reached_count == 0 && outcome.complete() {
            return format!("根据现有知识库，「{}」没有{}。", entity, target_phrase);
        }

        let basis = outcome
            .law
            .as_deref()
            .map(|law| format!("；依据 {}", law))
            .unwrap_or_default();
        let count = if outcome.count_is_lower_bound() {
            format!("至少 {} 个", outcome.reached_count)
        } else {
            format!("共 {} 个", outcome.reached_count)
        };
        let mut output = format!(
            "根据知识图谱（{}{}）：「{}」的{} {}。",
            outcome.operator.label(),
            basis,
            entity,
            target_phrase,
            count
        );
        if !count_only {
            for (index, (target, hops)) in outcome.targets_by_distance().into_iter().enumerate() {
                let witness = outcome
                    .render_witness(target)
                    .unwrap_or_else(|| target.to_string());
                output.push_str(&format!(
                    "
  {}. {}（{} 跳：{}）",
                    index + 1,
                    target,
                    hops,
                    witness
                ));
            }
        } else {
            output.push_str(
                "
  计数口径：不同可达节点，不按路径重复计数。",
            );
        }
        if outcome.witnesses_truncated {
            output.push_str(&format!(
                "
  说明：见证仅保留前 {} 个，计数仍按不同节点去重。",
                outcome.witnesses.len()
            ));
        }
        if let Some(truncation) = outcome.truncated {
            output.push_str(&format!(
                "
  说明：闭包搜索在{}处停止，结果不完整，上述计数是下界（至少 {} 个）。",
                truncation.label(),
                outcome.reached_count
            ));
        }
        output
    }

    /// 从词法记忆里取谓词期待类型最可信的量词；没有证据时返回 `None`。
    fn learned_count_classifier(&self, predicate: &str) -> Option<String> {
        let expected = self
            .lexical_memory
            .type_expectation(predicate)
            .map(|expectation| expectation.expected_type)
            .or_else(|| {
                crate::identity::infer_suffix_type(predicate)
                    .map(|(_, expected)| expected.to_string())
            })?;
        self.lexical_memory
            .best_classifier_for_type(&expected, 1)
            .filter(|preference| !preference.conflict)
            .map(|preference| preference.classifier)
    }

    /// 用命中值的实体类型反推分类词：`李四/王五` 都是学生时优先说「位/名」。
    fn learned_count_classifier_for_values(
        &self,
        values: &[String],
        predicate: &str,
    ) -> Option<String> {
        let expected = values
            .iter()
            .find_map(|value| self.generator.entity_type_of(value))
            .map(str::to_string)
            .or_else(|| {
                self.lexical_memory
                    .type_expectation(predicate)
                    .map(|expectation| expectation.expected_type)
            })
            .or_else(|| {
                crate::identity::infer_suffix_type(predicate)
                    .map(|(_, expected)| expected.to_string())
            })?;
        let category = crate::quantifier::semantic_category(&expected)
            .unwrap_or(expected.as_str())
            .to_string();
        self.lexical_memory
            .best_classifier_for_type(&category, 1)
            .or_else(|| self.lexical_memory.best_classifier_for_type(&expected, 1))
            .filter(|preference| !preference.conflict)
            .map(|preference| preference.classifier)
    }

    fn answer_count(
        &self,
        subject: &str,
        predicate: &str,
        exist: bool,
        classifier: Option<&str>,
    ) -> Option<String> {
        let entity = self.aligner.canonical_form(subject.trim());
        if entity.is_empty() {
            return None;
        }
        let (slot, inverted) = self.resolve_question_predicate(predicate);
        if slot.is_empty() {
            return None;
        }
        if !self.is_known_entity(&entity) && !self.predictor.participates(&entity) {
            return None;
        }
        let chain_predicate = ChainPredicate {
            surface: predicate.trim().to_string(),
            slot: slot.clone(),
            inverted,
        };
        let hits = self.node_hits(&entity, &chain_predicate);
        if hits.is_empty() {
            if self.is_known_predicate(&chain_predicate.surface) {
                if exist {
                    return Some(format!(
                        "根据现有知识库，「{}」没有「{}」的记录。",
                        entity, chain_predicate.surface
                    ));
                }
                return Some(format!(
                    "根据现有知识库，「{}」的「{}」数量为 0（没有相关记录）。",
                    entity, chain_predicate.surface
                ));
            }
            return None;
        }
        let values: Vec<String> = hits.iter().map(|hit| hit.value.clone()).collect();
        let learned = classifier
            .is_none()
            .then(|| self.learned_count_classifier_for_values(&values, predicate))
            .flatten();
        let classifier = classifier
            .or(learned.as_deref())
            .map(str::trim)
            .filter(|classifier| !classifier.is_empty())
            .unwrap_or("个");
        let mut output = if exist {
            format!(
                "是的，根据知识图谱，「{}」有 {}（共 {} {}：{}）。",
                entity,
                chain_predicate.surface,
                values.len(),
                classifier,
                values.join("、")
            )
        } else {
            format!(
                "根据知识图谱：「{}」的「{}」共有 {} {}：{}。",
                entity,
                chain_predicate.surface,
                values.len(),
                classifier,
                values.join("、")
            )
        };
        if let Some(prediction) = hits.iter().find_map(|hit| hit.prediction.as_ref()) {
            output.push_str(&format!(
                "\n其中为符号推断结果（例如置信度 {:.2}）：{}",
                prediction.confidence,
                self.predictor.explain(prediction)
            ));
        }
        Some(output)
    }

    fn answer_check(&self, subject_segments: &[String], object_raw: &str) -> Option<String> {
        if subject_segments.len() < 2 {
            return None;
        }
        let predicate_raw = subject_segments.last()?.clone();
        let (entity, mut intermediate) =
            self.split_chain(&subject_segments[..subject_segments.len() - 1])?;
        let (slot, inverted) = self.resolve_question_predicate(&predicate_raw);
        let object = self.aligner.canonical_form(object_raw.trim());
        if entity.is_empty() || slot.is_empty() || object.is_empty() {
            return None;
        }
        let surface = self.display_predicate_surface(&predicate_raw);
        intermediate.push(ChainPredicate {
            surface: surface.clone(),
            slot: slot.clone(),
            inverted,
        });

        if intermediate.len() == 1 {
            let (negative_subject, negative_object) = if inverted {
                (object.as_str(), entity.as_str())
            } else {
                (entity.as_str(), object.as_str())
            };
            if self.session_negative_blocks(negative_subject, &slot, negative_object) {
                return Some(format!(
                    "不是。会话中记录了否定事实：{} 的 {} 不是 {}。",
                    entity, surface, object
                ));
            }
        }

        // 单跳直查：逆向谓词按槽位反方向核对（triple(object, slot, entity)）
        if intermediate.len() == 1
            && (if inverted {
                self.predictor.has_triple(&object, &slot, &entity)
            } else {
                self.predictor.has_triple(&entity, &slot, &object)
            })
        {
            return Some(format!(
                "是的。根据知识图谱：{} 的 {} 是 {}。",
                entity, surface, object
            ));
        }

        if intermediate.len() >= 2 {
            let graph = self.traversal_graph();
            let chain: Vec<PathPredicate> = intermediate
                .iter()
                .map(|predicate| PathPredicate {
                    predicate: predicate.slot.clone(),
                    inverted: predicate.inverted,
                })
                .collect();
            let bundle =
                graph.enumerate_chain(&entity, &chain, &self.path_options(intermediate.len()));
            let negated_target = bundle.blocked.iter().any(|blocked| {
                matches!(
                    &blocked.reason,
                    crate::path_memory::BlockReason::NegatedEdge { to, .. } if to == &object
                )
            });
            if negated_target {
                return Some(format!(
                    "不是。会话中记录的否定事实阻断了通往 {} 的路径。",
                    object
                ));
            }
            let matched: Vec<&crate::path_memory::PathWitness> = bundle
                .witnesses
                .iter()
                .filter(|witness| witness.end() == Some(object.as_str()))
                .collect();
            if !matched.is_empty() {
                let chain_text = format!(
                    "{} 的 {}",
                    entity,
                    intermediate
                        .iter()
                        .map(|predicate| predicate.surface.as_str())
                        .collect::<Vec<_>>()
                        .join(" 的 ")
                );
                let mut output = format!("是的。根据知识图谱：{} 是 {}。\n", chain_text, object);
                for witness in matched.iter().take(3) {
                    output.push_str(&format!(
                        "  路径（{} 跳）：{}\n",
                        witness.hop_count(),
                        witness.render()
                    ));
                }
                return Some(output.trim_end().to_string());
            }
            if bundle.witnesses.is_empty()
                && bundle.blocked.iter().any(|blocked| {
                    matches!(
                        blocked.reason,
                        crate::path_memory::BlockReason::NegatedEdge { .. }
                    )
                })
            {
                return Some(bundle.render_text());
            }
        }

        let mut chain_result = self.run_chain(&entity, &intermediate);
        let need_laws = self.laws.has_inference_laws()
            && chain_result.as_ref().map_or(true, |run| {
                !run.last_hits
                    .iter()
                    .any(|hit| hit.value == object && hit.construction.is_none())
            });
        if need_laws {
            if let Some(expanded) = self.run_chain_via_laws(&entity, &intermediate) {
                chain_result = Some(expanded);
            }
        }
        if let Some(run) = chain_result {
            if let Some(hit) = run.last_hits.iter().find(|hit| hit.value == object) {
                let step = run.steps.last()?;
                if let Some(fact) = &hit.construction {
                    let confirmable = run
                        .steps
                        .iter()
                        .flat_map(|step| step.hits.iter())
                        .chain(run.last_hits.iter())
                        .filter_map(|candidate| candidate.construction.as_ref())
                        .map(|candidate| &candidate.key)
                        .next()
                        .unwrap_or(&fact.key);
                    return Some(format!(
                        "很可能，但目前只是建设性推断（待确认，置信度 {:.2}）：{} 的 {} 是 {}。\n  推断依据：{}；前提：{}。\n  说明：推断只在本次查询内有效，未写入知识图谱；确认请回复「确认推断：{}|{}|{}」。",
                        fact.confidence(),
                        step.subject,
                        step.predicate,
                        object,
                        fact.rule,
                        fact.support
                            .iter()
                            .map(|premise| premise.render())
                            .collect::<Vec<_>>()
                            .join("、"),
                        confirmable.subject,
                        confirmable.predicate,
                        confirmable.object
                    ));
                }
                return Some(match &hit.prediction {
                    None => match &hit.law {
                        Some(law_id) => format!(
                            "是的。{}",
                            self.laws.explain(law_id).unwrap_or_else(|| {
                                format!(
                                    "根据知识图谱：{} 的 {} 是 {}。",
                                    step.subject, step.predicate, object
                                )
                            })
                        ),
                        None => format!(
                            "是的。根据知识图谱：{} 的 {} 是 {}。",
                            step.subject, step.predicate, object
                        ),
                    },
                    Some(prediction) => format!(
                        "很可能是的（符号推断）。{}",
                        self.predictor.explain(prediction)
                    ),
                });
            }
            let known_values: Vec<String> = run
                .last_hits
                .iter()
                .filter(|hit| hit.prediction.is_none() && hit.construction.is_none())
                .map(|hit| hit.value.clone())
                .collect();
            let constructed_values: Vec<String> = run
                .last_hits
                .iter()
                .filter(|hit| hit.construction.is_some())
                .map(|hit| hit.value.clone())
                .collect();
            if !known_values.is_empty() {
                let step = run.steps.last()?;
                if run.closure_truncated {
                    return Some(format!(
                        "无法确定。传递闭包扩展达到上限，未列出通往 {} 的路径；目前记录的是：{} 的 {} 是 {}。",
                        object,
                        step.subject,
                        step.predicate,
                        known_values.join("、")
                    ));
                }
                return Some(format!(
                    "不是。知识图谱记录的是：{} 的 {} 是 {}。",
                    step.subject,
                    step.predicate,
                    known_values.join("、")
                ));
            }
            if !constructed_values.is_empty() {
                let step = run.steps.last()?;
                return Some(format!(
                    "无法确定。知识图谱没有直接记录，建设性推断给出的是：{} 的 {} 是 {}（待确认，未写入知识图谱）。",
                    step.subject,
                    step.predicate,
                    constructed_values.join("、")
                ));
            }
        } else if intermediate.len() == 1 {
            let known_values: Vec<String> = if inverted {
                self.predictor.subjects_of(&slot, &entity)
            } else {
                self.predictor.objects_of(&entity, &slot)
            };
            if !known_values.is_empty() {
                return Some(format!(
                    "不是。知识图谱记录的是：{} 的 {} 是 {}。",
                    entity,
                    surface,
                    known_values.join("、")
                ));
            }
        }

        // 直接边与图谱取值都没有命中；若规则能现算出目标取值，按推导确认，
        // 同时把前提和规则来源一起交代，避免「是」成为无出处的判断。
        if intermediate.len() == 1 {
            if let Some(fact) = self.derived_fact_for(&entity, &intermediate[0], &object) {
                return Some(format!(
                    "是的。根据当前规则现算的推导（未写入知识图谱）：{} 的 {} 是 {}。\n{}",
                    entity,
                    surface,
                    object,
                    self.render_derived_support(&fact)
                ));
            }
        }

        if self.is_known_entity(&entity) && self.is_known_predicate(&surface) {
            return Some(format!(
                "根据现有知识库，无法确认「{} 的 {} 是 {}」（没有相关记录）。",
                entity, surface, object
            ));
        }
        None
    }
}

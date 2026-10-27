impl SymbolicReasoner {
    /// 更正预演：在提交到图谱之前，先展示受影响的事实/派生结论与冲突。
    /// 只计算、不修改知识库；真正的写入仍由宿主的更正路径完成。
    pub fn belief_revision_preview(
        &self,
        entity: &str,
        predicate: &str,
        old_value: &str,
        new_value: &str,
    ) -> String {
        let entity = self.aligner.canonical_form(entity.trim());
        let predicate = self
            .canonical_attribute_name(predicate)
            .unwrap_or_else(|| predicate.trim().to_string());
        let state = self.belief_state();
        let changeset = ChangeSet::new(format!(
            "预演更正：{}.{} 由「{}」改为「{}」",
            entity, predicate, old_value, new_value
        ))
        .retract(&predicate, &entity, old_value)
        .assert(&predicate, &entity, new_value);
        state.plan(changeset).render_zh()
    }

    fn session_revision_constraints(&self) -> RevisionConstraints {
        match &self.law_registry {
            Some(registry) => RevisionConstraints::from_registry(registry),
            None => RevisionConstraints::from_law_book(&self.laws),
        }
    }

    /// 会话写入守门器：以当前归结引擎为基座，先计划再决定是否落笔。
    pub fn revision_guard(&self) -> RevisionGuard {
        RevisionGuard::new(self.resolver.clone(), self.session_revision_constraints())
    }

    /// 预检一批会话级变更（事实/规则），返回带最小冲突核的计划报告。
    pub fn preview_session_change(&self, label: &str, ops: Vec<ChangeOp>) -> GuardPlan {
        let mut changeset = ChangeSet::new(label.to_string());
        for op in ops {
            changeset.push(op);
        }
        self.revision_guard().plan(changeset)
    }

    fn render_guard_block(&self, plan: &GuardPlan) -> String {
        format!(
            "{}\n本次写入未提交：会话事实、会话规则与教学记忆均未改变。",
            plan.render_zh()
        )
    }

    fn render_guard_impact(plan: &GuardPlan) -> Option<String> {
        let counts = &plan.plan.impact.counts;
        if counts.derived_added == 0 && counts.derived_retracted == 0 {
            return None;
        }
        Some(format!(
            "预检影响：派生 +{} / -{}，净相信 +{} / -{}。\n",
            counts.derived_added,
            counts.derived_retracted,
            counts.newly_believed,
            counts.no_longer_believed
        ))
    }

    fn record_session_revision(
        &mut self,
        label: impl Into<String>,
        ops: Vec<ChangeOp>,
        restore_lessons: Vec<String>,
        remove_lessons: Vec<String>,
    ) {
        if ops.is_empty() && restore_lessons.is_empty() && remove_lessons.is_empty() {
            return;
        }
        let revision = self
            .session_receipts
            .last()
            .map(|receipt| receipt.revision + 1)
            .unwrap_or(1);
        let material = ops
            .iter()
            .map(ChangeOp::describe)
            .collect::<Vec<_>>()
            .join("|");
        let digest = material.bytes().fold(2_166_136_261u32, |acc, byte| {
            (acc ^ u32::from(byte)).wrapping_mul(16_777_619)
        });
        let id = format!("session:{}:{:08x}", revision, digest);
        if self.session_receipts.len() >= MAX_SESSION_RECEIPTS {
            self.session_receipts.remove(0);
        }
        self.session_receipts.push(SessionRevisionReceipt {
            id,
            label: label.into(),
            revision,
            ops,
            restore_lessons,
            remove_lessons,
        });
    }

    /// 会话修订回执：最近的可撤销写入，以及每笔写入的真实操作数。
    pub fn session_revision_report(&self) -> String {
        let mut lines = vec![format!(
            "会话修订回执：{} 条（最多保留 {} 条）",
            self.session_receipts.len(),
            MAX_SESSION_RECEIPTS
        )];
        if self.session_receipts.is_empty() {
            lines.push("  （还没有可撤销的会话写入）".to_string());
            return lines.join("\n");
        }
        for receipt in self.session_receipts.iter().rev().take(5) {
            let mut summary = format!(
                "  · {} 〔{}；{} 个操作；版本 {}",
                receipt.id,
                receipt.label,
                receipt.ops.len(),
                receipt.revision
            );
            if !receipt.restore_lessons.is_empty() || !receipt.remove_lessons.is_empty() {
                summary.push_str(&format!(
                    "；教学原句 +{} / -{}",
                    receipt.restore_lessons.len(),
                    receipt.remove_lessons.len()
                ));
            }
            summary.push('〕');
            lines.push(summary);
            for op in receipt.ops.iter().take(4) {
                lines.push(format!("      {}", op.describe()));
            }
            if receipt.ops.len() > 4 {
                lines.push(format!("      …另有 {} 个操作", receipt.ops.len() - 4));
            }
        }
        lines.push("可用「撤销修订」撤回最新一条回执。".to_string());
        lines.join("\n")
    }

    fn remove_session_clause(&mut self, clause: &Clause) -> bool {
        match self
            .session_clauses
            .iter()
            .position(|existing| existing.head == clause.head && existing.body == clause.body)
        {
            Some(position) => {
                self.session_clauses.remove(position);
                true
            }
            None => false,
        }
    }

    fn add_session_clause(&mut self, clause: &Clause) -> bool {
        if self
            .session_clauses
            .iter()
            .any(|existing| existing.head == clause.head && existing.body == clause.body)
        {
            return false;
        }
        let mut new_predicates = false;
        for atom in std::iter::once(&clause.head).chain(clause.body.iter()) {
            if !self.aligner.is_registered(&atom.predicate) {
                self.aligner
                    .register_term(&atom.predicate, TermKind::Relation);
                new_predicates = true;
            }
        }
        self.session_clauses.push(clause.clone());
        if new_predicates {
            self.rebuild_knowledge_caches();
        }
        true
    }

    /// 撤销最新一条会话修订回执：按逆操作恢复事实/规则/教学原句。
    /// 撤销本身也记为新回执，历史不回写。
    pub fn undo_last_session_revision(&mut self) -> String {
        let Some(receipt) = self.session_receipts.last().cloned() else {
            return "没有可撤销的会话修订。".to_string();
        };
        let mut inverse_ops: Vec<ChangeOp> = Vec::new();
        let mut facts_changed = 0usize;
        let mut clauses_changed = 0usize;
        let mut clause_touched = false;
        for op in receipt.ops.iter().rev() {
            match op {
                ChangeOp::AssertFact(atom) => {
                    if self.retract_atom(atom) {
                        self.note_slot_retract(atom);
                        facts_changed += 1;
                        inverse_ops.push(ChangeOp::RetractFact(atom.clone()));
                    }
                }
                ChangeOp::RetractFact(atom) => {
                    self.register_atom_vocabulary(atom);
                    if self.adopt_atom(atom.clone()) {
                        self.note_slot_adopt(atom, AssertionStrength::Correction, "撤销修订");
                        facts_changed += 1;
                        inverse_ops.push(ChangeOp::AssertFact(atom.clone()));
                    }
                }
                ChangeOp::AddRule(clause) => {
                    if self.remove_session_clause(clause) {
                        clauses_changed += 1;
                        clause_touched = true;
                        inverse_ops.push(ChangeOp::RemoveRule(clause.clone()));
                    }
                }
                ChangeOp::RemoveRule(clause) => {
                    if self.add_session_clause(clause) {
                        clauses_changed += 1;
                        clause_touched = true;
                        inverse_ops.push(ChangeOp::AddRule(clause.clone()));
                    }
                }
            }
        }
        if clause_touched {
            self.rebuild_resolver();
        }
        let mut lessons_restored = 0usize;
        for lesson in &receipt.restore_lessons {
            if !self.lessons.contains(lesson) {
                self.lessons.push(lesson.clone());
                lessons_restored += 1;
            }
        }
        let mut lessons_removed = 0usize;
        for lesson in &receipt.remove_lessons {
            if let Some(position) = self.lessons.iter().position(|item| item == lesson) {
                self.lessons.remove(position);
                lessons_removed += 1;
            }
        }
        if lessons_restored > 0 || lessons_removed > 0 {
            self.lessons_parsed = self
                .lessons
                .iter()
                .flat_map(|sentence| chinese_reasoning_nlu::parse_lessons(sentence).lessons)
                .collect();
            if self.config.enable_problog && self.config.enable_soft_logic {
                self.rebuild_problog();
            }
        }
        if inverse_ops.is_empty() && lessons_restored == 0 && lessons_removed == 0 {
            return format!(
                "回执 {} 在当前会话没有可恢复的改动（可能已被后续操作覆盖）。",
                receipt.id
            );
        }
        self.record_session_revision(
            format!("撤销 {}", receipt.id),
            inverse_ops,
            receipt.remove_lessons.clone(),
            receipt.restore_lessons.clone(),
        );
        let new_id = self
            .session_receipts
            .last()
            .map(|item| item.id.clone())
            .unwrap_or_default();
        format!(
            "已撤销会话修订 {}（{}）：恢复事实 {} 条、规则 {} 条、教学原句 +{} / -{} 条。新回执：{}。",
            receipt.id,
            receipt.label,
            facts_changed,
            clauses_changed,
            lessons_restored,
            lessons_removed,
            new_id
        )
    }

    fn rebuild_laws(&mut self) {
        let Some(registry) = &self.law_registry else {
            self.laws = self.persisted_laws.clone().unwrap_or_default();
            return;
        };
        let mut book = LawBook::from_registry(registry);
        book.add_mined_rules(self.predictor.rules());
        if let Some(persisted) = &self.persisted_laws {
            book.apply_persisted(persisted);
        }
        self.laws = book;
    }

    /// 问句谓词解析：对齐器归一后，若为反向词面则返回 (槽位, true)。
    /// 与「谓词画像」约定一致：逆向词面的答案是槽位三元组的反方向。
    ///
    /// 生成器在观察事实时登记了「图谱键 ↔ 中文词面」对照（如 knows ↔ 认识）。
    /// 这类词面没有经过对齐记忆，但同样是已登记谓词；在此统一解析后，正向
    /// 查询与「谁认识李四」这类反查主体问句才能落到同一个槽位。
    fn resolve_question_predicate(&self, raw: &str) -> (String, bool) {
        let raw = raw.trim();
        if let Some((slot, inverted)) = self.inverse_query_map.get(raw) {
            return (slot.clone(), *inverted);
        }
        let canonical = self.aligner.canonical_form(raw);
        if let Some((slot, inverted)) = self.inverse_query_map.get(canonical.as_str()) {
            return (slot.clone(), *inverted);
        }
        if let Some(generated) = self.generator.canonical_predicate(raw) {
            return (generated, false);
        }
        (canonical, false)
    }

    /// 回答措辞里的谓词面：用户原话优先；只有原话就是图谱规范键
    /// （如 knows/age）时，才回退到生成器登记的中文词面，避免回答里
    /// 突然出现英文键。
    fn display_predicate_surface(&self, raw: &str) -> String {
        let raw = raw.trim();
        let slot = self.resolve_question_predicate(raw).0;
        if raw == slot {
            if let Some(surface) = self.generator.preferred_predicate_surface(&slot) {
                return surface.to_string();
            }
        }
        raw.to_string()
    }

    pub fn observe_facts(&mut self, facts: &[PrologFact]) {
        self.perf.bump("observe_calls");
        self.perf.bump_by("observe_facts_input", facts.len() as u64);
        // 统一证据索引与知识图谱同源观察（内部去重；重建按需惰性触发）
        self.retrieval.lock().observe_facts(facts);
        // 话语引擎：本体叙事与 DCG 生成共用同一批事实（内部去重，可反复观察）
        if self.config.enable_discourse {
            self.discourse.observe_facts(facts);
        }
        // 溯因统计只累积计数（谓词取值、类型分布、特征），不参与学习
        self.abduction.observe_facts(facts);
        let mut relation_seen = 0usize;
        let mut entity_new = false;
        let mut entity_type_changed = false;
        let mut association_changed = false;
        let mut attribute_changed = false;
        let triples_before = self.predictor.triple_count();
        for fact in facts {
            match fact {
                PrologFact::Relation { from, rel_type, to } => {
                    self.aligner.observe_triple(from, rel_type, to);
                    self.predictor.add_triple(from, rel_type, to);
                    self.linker.observe_triple(from, rel_type, to);
                    for endpoint in [from, to] {
                        let etype = self.linker.entity_type(endpoint).map(str::to_string);
                        self.mention_lexicon.confirm(
                            endpoint,
                            endpoint,
                            etype.as_deref(),
                            "kb_relation",
                        );
                    }
                    self.generator.observe_triple(from, rel_type, to);
                    if self.config.enable_semantic_embedding {
                        self.embedding.observe_triple(from, rel_type, to);
                    }
                    if self.config.enable_graph_walk {
                        self.walks.observe_triple(from, rel_type, to);
                    }
                    if self.config.enable_holographic_memory {
                        self.holo.observe_triple(from, rel_type, to);
                    }
                    association_changed |= self.associations.observe_triple(from, rel_type);
                    relation_seen += 1;
                }
                PrologFact::Entity {
                    name,
                    entity_type,
                    properties,
                } => {
                    self.aligner.register_term(name, TermKind::Entity);
                    self.aligner.register_term(entity_type, TermKind::Type);
                    for property in properties.keys() {
                        self.aligner.register_term(property, TermKind::Attribute);
                    }
                    // 新实体同样会改变实体提及/链接剖面，必须触发词表缓存重建
                    if !self.linker.is_known_entity(name) {
                        entity_new = true;
                    }
                    self.linker
                        .register_entity(name, Some(entity_type.as_str()));
                    self.mention_lexicon
                        .confirm(name, name, Some(entity_type.as_str()), "kb");
                    // 生成器的实体类型会被后续观察覆盖，归结引擎的 __类型__ 事实依赖它：
                    // 类型变化时必须触发重建（关联篮子的变化无法覆盖此情形）
                    if !entity_type.trim().is_empty()
                        && self.generator.entity_type_of(name) != Some(entity_type.as_str())
                    {
                        entity_type_changed = true;
                    }
                    self.generator
                        .observe_entity_map(name, entity_type, properties);
                    if self.config.enable_semantic_embedding {
                        self.embedding
                            .observe_entity_map(name, entity_type, properties);
                    }
                    association_changed |=
                        self.associations
                            .observe_entity_map(name, entity_type, properties);
                    for (key, value) in properties {
                        if self
                            .attributes
                            .insert((name.clone(), key.clone(), value.clone()))
                        {
                            attribute_changed = true;
                        }
                        if self.config.enable_holographic_memory {
                            self.holo.observe_triple(name, key, value);
                        }
                    }
                }
            }
        }
        let triples_added = self.predictor.triple_count() > triples_before;
        self.perf.bump_by("observe_relations", relation_seen as u64);
        // 版本化增量：内容确有变化 → 版本 +1、依赖挂牌、派生重建；
        // 重复观察（快照重放）→ 版本不动、memo 不清、派生不重建
        let changed = triples_added
            || association_changed
            || attribute_changed
            || entity_type_changed
            || entity_new;
        if !changed {
            self.perf.bump("observe_no_change");
            return;
        }
        // 图谱变化后，上一张模式的见证下标与结果页不再可信：整体作废。
        *self.graph_session.lock() = None;
        self.revision += 1;
        self.perf.bump("observe_changed");
        if triples_added {
            self.perf.bump("observe_new_triples");
            self.stamp_dependency("triples");
        }
        if attribute_changed {
            self.stamp_dependency("attributes");
        }
        if entity_type_changed {
            self.stamp_dependency("entity_types");
        }
        if association_changed {
            self.stamp_dependency("associations");
        }
        // 内容变化意味着链接剖面可能变化，单条记忆整体失效（在变更检测后统一做）
        self.link_memo.lock().clear();
        self.entity_mentions_memo.lock().clear();
        self.predicate_canonicals_memo.lock().clear();
        self.prediction_drafts.lock().clear();
        // 规则只依赖已积累的三元组与固定阈值：有新增三元组时才重新挖掘
        if triples_added && self.config.enable_link_prediction {
            self.predictor.mine_rules(
                self.config.link_prediction_min_support,
                self.config.link_prediction_min_confidence,
            );
            self.stamp_rebuild("link_rules");
        }
        // 关联规则与前向/反向归结都只依赖已观察的篮子与事实：有新增时才重建
        if self.config.enable_association_rules && (association_changed || triples_added) {
            self.associations.mine();
            self.stamp_rebuild("associations");
        }
        if self.config.enable_resolution
            && (association_changed || attribute_changed || entity_type_changed || triples_added)
        {
            self.rebuild_resolver();
        }
        // 加权软逻辑与归结引擎同源（同一批事实与规则），触发条件保持一致
        if self.config.enable_soft_logic
            && (association_changed || attribute_changed || entity_type_changed || triples_added)
        {
            self.rebuild_soft_logic();
        }
        // 图游走向量采用惰性重建：新边在 observe_triple 内标记失效，
        // 首次需要嵌入时才做全量幂迭代（查询结果与即时重建完全一致）
        if self.config.enable_semantic_embedding {
            if self.embedding.is_dirty() {
                self.perf.bump("embedding_rebuild");
                self.embedding.rebuild();
            } else {
                self.perf.bump("embedding_rebuild_skipped");
            }
        }
        // 谓词清单与命题表面积索引随剖面/规则变化，统一在变更后重建一次
        self.rebuild_knowledge_caches();
    }

    // 用当前知识（事实 + 链接预测规则 + 关联规则）重建归结引擎
    fn rebuild_resolver(&mut self) {
        let mut engine = ResolutionEngine::new(
            self.config.resolution_max_depth,
            self.config.resolution_max_solutions,
        );
        for triple in self.predictor.triples() {
            engine.add_fact(&triple.predicate, &triple.subject, &triple.object);
            engine.add_fact(PROOF_HAS, &triple.subject, &triple.predicate);
        }
        for (subject, key, value) in &self.attributes {
            engine.add_fact(key, subject, value);
            engine.add_fact(PROOF_HAS, subject, key);
        }
        for (entity, etype) in self.generator.typed_entities() {
            engine.add_fact(PROOF_TYPE, &entity, &etype);
        }
        if self.config.enable_association_rules {
            for clause in self.associations.to_clauses(PROOF_HAS, PROOF_TYPE) {
                engine.add_clause(clause);
            }
        }
        for rule in self.predictor.rules() {
            engine.add_clause(clause_from_path_rule(rule));
        }
        // 会话级假设事实必须在重建后重新注入（否则刷新知识图谱会丢失"采纳的假设"）
        for atom in self.positive_session_atoms() {
            engine.add_fact_atom(atom.clone());
        }
        // 会话级 Horn 规则同样在重建后重新注入（教过的一般规则不能因刷新而丢失）
        for clause in &self.session_clauses {
            engine.add_clause(clause.clone());
        }
        // 只有用户显式晋升到全局的补全规则才进入全局归结；本批/本例/
        // 命名空间规则留在回合内，避免没有事实级命名空间标签的引擎跨域 join。
        for clause in self.completion.reasoning_clauses() {
            engine.add_clause(clause);
        }
        self.resolver = engine;
        self.stamp_rebuild("resolver");
    }

    /// 用当前知识重建加权软逻辑引擎：事实同上，规则权重 = 置信度的 log-odds
    fn rebuild_soft_logic(&mut self) {
        let mut engine = MarkovLogicEngine::new(
            self.config.soft_logic_max_iterations,
            self.config.soft_logic_min_probability,
        );
        for triple in self.predictor.triples() {
            engine.add_fact(Atom::rel(
                &triple.predicate,
                &triple.subject,
                &triple.object,
            ));
            engine.add_fact(Atom::rel(PROOF_HAS, &triple.subject, &triple.predicate));
        }
        for (subject, key, value) in &self.attributes {
            engine.add_fact(Atom::rel(key, subject, value));
            engine.add_fact(Atom::rel(PROOF_HAS, subject, key));
        }
        for (entity, etype) in self.generator.typed_entities() {
            engine.add_fact(Atom::rel(PROOF_TYPE, &entity, &etype));
        }
        for atom in self.positive_session_atoms() {
            engine.add_fact(atom.clone());
        }
        if self.config.enable_association_rules {
            for clause in self.associations.to_clauses(PROOF_HAS, PROOF_TYPE) {
                engine.add_rule(WeightedRule::new(
                    clause.head,
                    clause.body,
                    MarkovLogicEngine::weight_from_confidence(clause.confidence),
                    &clause.source,
                ));
            }
        }
        for rule in self.predictor.rules() {
            let clause = clause_from_path_rule(rule);
            engine.add_rule(WeightedRule::new(
                clause.head,
                clause.body,
                MarkovLogicEngine::weight_from_confidence(clause.confidence),
                &clause.source,
            ));
        }
        for clause in self.completion.reasoning_clauses() {
            engine.add_rule(WeightedRule::new(
                clause.head,
                clause.body,
                MarkovLogicEngine::weight_from_confidence(clause.confidence),
                &clause.source,
            ));
        }
        self.soft = engine;
        self.stamp_rebuild("soft_logic");
        // 概率逻辑程序与软逻辑同源：软逻辑重建后同步转换（含重放教过的中文知识）
        if self.config.enable_problog {
            self.rebuild_problog();
        }
    }

    /// 重建 ProbLog 风格概率程序：MLN-lite 规则按 σ(w) 转成概率规则，
    /// 再重放用户用中文教过的概率知识（会话级，不修改知识图谱）
    fn rebuild_problog(&mut self) {
        let mut program = ProblogEngine::from_markov_engine(
            &self.soft,
            self.config.problog_max_depth,
            self.config.problog_max_proofs,
        );
        // 重放缓存好的解析结果
        for lesson in &self.lessons_parsed {
            chinese_reasoning_nlu::apply_lesson(&mut program, lesson);
        }
        self.problog = program;
        self.stamp_rebuild("problog");
    }

    pub fn resolver(&self) -> &ResolutionEngine {
        &self.resolver
    }

    pub fn associations(&self) -> &AssociationMiner {
        &self.associations
    }

    pub fn contextualize(&mut self, input: &str) -> Option<String> {
        self.contextualize_with_receipt(input)
            .map(|rewrite| rewrite.text)
    }

    /// 与 `contextualize` 相同，但附带来源收据（哪个洞由哪一轮的哪个槽补上）。
    ///
    /// 处理顺序：代词消解 → 话语帧合一（前置尝试，命中才接管）→ 省略规则
    /// → 代词改写。帧通道不合法的合一逐字回退。
    pub fn contextualize_with_receipt(&mut self, input: &str) -> Option<ContextRewrite> {
        self.frames.begin_turn();
        let pronoun_resolved = self.context.resolve_pronouns(input);
        let base = pronoun_resolved.as_deref().unwrap_or(input);
        let rewrite = self
            .try_frame_rewrite(base)
            .or_else(|| {
                self.resolve_ellipsis(base)
                    .map(|text| ContextRewrite::from_text(text))
            })
            .or_else(|| pronoun_resolved.map(ContextRewrite::from_text));
        self.record_vocabulary(input);
        if let Some(rewrite) = &rewrite {
            self.record_vocabulary(&rewrite.text);
        }
        if let Some(rewrite) = &rewrite {
            self.record_plan_kind_from_text(&rewrite.text);
        } else {
            self.record_plan_kind_from_text(input);
        }
        rewrite
    }

    /// 记录本轮成功编译的查询计划类型；省略句/纯寒暄没有计划时不覆盖
    /// 上一轮类型，保证「那李四呢」仍能接着上一轮的结构补全。
    fn record_plan_kind_from_text(&mut self, text: &str) {
        let plan = self.planner.parse(text).or_else(|| {
            query_rewrite::rewrite(text, &self.relation_lexicon()).map(|rewrite| rewrite.plan)
        });
        if let Some(plan) = plan {
            self.context.record_plan_kind(QueryPlanner::kind(&plan));
        }
    }

    /// 话语帧通道：复合问句拆帧（第一问照答、其余挂账）、省略追问合一补全、
    /// 完整 AskAttr 问句登记为上一帧。命中返回规范问句与来源收据。
    fn try_frame_rewrite(&mut self, input: &str) -> Option<ContextRewrite> {
        let turn = self.frames.turn();
        if let Some(compound) = dialogue_frame::parse_compound(input, &FrameOntology(self), turn) {
            let frames = compound.frames;
            let first_question = frames.first().and_then(|frame| frame.question())?;
            let mut notes = Vec::new();
            if frames.len() > 1 {
                notes.push(format!(
                    "话语帧：先答「{}」，其余挂账等待追问：{}",
                    first_question,
                    frames[1..]
                        .iter()
                        .map(|frame| frame.display())
                        .collect::<Vec<_>>()
                        .join("；")
                ));
            }
            self.frames.record_compound(frames);
            return Some(ContextRewrite {
                text: first_question,
                notes,
            });
        }

        let fragment =
            dialogue_frame::parse_followup(input, &FrameOntology(self), turn).or_else(|| {
                dialogue_frame::parse_bare_attribute_question(input, &FrameOntology(self), turn)
            });
        if let Some(fragment) = fragment {
            let plan = self.frames.plan_followup(&fragment)?;
            if !self.frame_slots_known(&plan.frame) {
                return None;
            }
            let resolution = self.frames.commit_followup(plan);
            return Some(ContextRewrite {
                text: resolution.question,
                notes: vec![resolution.note],
            });
        }

        if let Some(frame) = self.parse_full_ask_frame(input) {
            self.frames.record_last(frame);
        }
        None
    }

    /// 完整 AskAttr 问句（单跳、实体与谓词均在本体登记）→ 登记为上一帧
    fn parse_full_ask_frame(&self, input: &str) -> Option<AskAttributeFrame> {
        let plan = self.planner.parse(input)?;
        let QueryPlan::FindObject { segments } = plan else {
            return None;
        };
        let (entity, predicates) = self.split_chain(&segments)?;
        if predicates.len() != 1 {
            return None;
        }
        if !self.is_known_entity(&entity) {
            return None;
        }
        let attribute = self.canonical_attribute_name(&predicates[0].surface)?;
        Some(AskAttributeFrame::new(
            input.trim(),
            self.frames.turn(),
            Some(entity),
            Some(attribute),
        ))
    }

    /// 合一产物的排序复核：实体/属性都必须是已登记项
    fn frame_slots_known(&self, frame: &AskAttributeFrame) -> bool {
        let entity_known = frame
            .entity
            .as_deref()
            .map(|entity| self.is_known_entity(entity))
            .unwrap_or(false);
        let attribute_known = frame
            .attribute
            .as_deref()
            .and_then(|attribute| self.canonical_attribute_name(attribute))
            .is_some();
        entity_known && attribute_known
    }

    /// 归一化属性词面：保留逆向词面的表述方向（「徒弟」问句 ↔「师傅」事实）
    fn canonical_attribute_name(&self, raw: &str) -> Option<String> {
        let raw = raw.trim();
        if raw.is_empty() {
            return None;
        }
        if self.inverse_query_map.contains_key(raw) {
            return Some(raw.to_string());
        }
        if let Some(aligned) = self.alignment_predicate_canonicalize(raw) {
            let known = matches!(
                self.aligner.kind_of(&aligned),
                Some(TermKind::Relation) | Some(TermKind::Attribute)
            ) || self.predictor.has_predicate(&aligned)
                || self.inverse_query_map.contains_key(aligned.as_str());
            if known {
                return Some(aligned);
            }
        }
        let canonical = self.aligner.canonical_form(raw);
        if self.inverse_query_map.contains_key(canonical.as_str()) {
            return Some(canonical);
        }
        let known = matches!(
            self.aligner.kind_of(&canonical),
            Some(TermKind::Relation) | Some(TermKind::Attribute)
        ) || self.predictor.has_predicate(&canonical);
        if known {
            Some(canonical)
        } else {
            None
        }
    }

    fn resolve_ellipsis(&self, input: &str) -> Option<String> {
        let trimmed = input.trim().trim_end_matches(['？', '?']).trim();
        let core = if let Some(rest) = trimmed.strip_prefix("那") {
            let rest = rest.strip_prefix("么").unwrap_or(rest);
            rest.strip_suffix("呢")?
        } else if trimmed.ends_with('呢') && trimmed.chars().count() <= 12 {
            trimmed.strip_suffix("呢")?
        } else {
            return None;
        };
        let core = core.trim();
        if core.is_empty() || core.contains(' ') {
            return None;
        }
        if core.contains('的') {
            return Some(format!("{}是什么", core));
        }
        let predicate = self.context.last_predicate()?;
        // 换实体的省略追问必须跟着上一轮的计划类型走：
        // 「张三认识谁」后的「李四呢」是「李四认识谁」，不是
        // 「李四的认识是什么」；反查句则替换宾语。
        if self.is_known_entity(core) {
            match self.context.last_plan_kind() {
                Some("FindObjectByPredicate") | Some("FindObjectExcept") => {
                    return Some(format!("{}{}谁", core, predicate));
                }
                Some("FindSubject") | Some("FindSubjectExcept") => {
                    return Some(format!("谁{}{}", predicate, core));
                }
                _ => {}
            }
        }
        Some(format!("{}的{}是什么", core, predicate))
    }

    fn record_vocabulary(&mut self, text: &str) {
        let mut entities = self.entities_mentioned(text);
        if self.config.enable_generation {
            for entity in self.linked_entities(text) {
                if !entities.contains(&entity) {
                    entities.push(entity);
                }
            }
        }
        if !entities.is_empty() {
            // 焦点实体带类型证据入栈：代词消解据此做类型闸门（未知则不设限）
            let typed: Vec<(String, Option<String>)> = entities
                .iter()
                .map(|entity| {
                    (
                        entity.clone(),
                        self.linker.entity_type(entity).map(str::to_string),
                    )
                })
                .collect();
            self.context.record_entities_typed(&typed);
            if self.config.enable_holographic_memory {
                for entity in &entities {
                    self.holo.push_focus(entity);
                }
            }
        }
        let normalized_text = self.aligner.normalize_cow(text);
        let mentioned_canonicals = self.mentioned_predicate_canonicals(text);
        let predicate =
            self.first_mentioned_known_predicate(normalized_text.as_ref(), &mentioned_canonicals);
        if let Some(predicate) = predicate {
            self.context.record_predicate(&predicate);
        }
        self.context.record_question(text);
    }

    fn collect_known_predicates(&self) -> Vec<String> {
        let mut predicates: Vec<String> = self
            .aligner
            .profiles()
            .iter()
            .filter(|p| p.kind == TermKind::Relation)
            .map(|p| p.term.clone())
            .collect();
        let mut seen: HashSet<String> = predicates.iter().cloned().collect();
        for rule in self.predictor.rules() {
            if seen.insert(rule.head.clone()) {
                predicates.push(rule.head.clone());
            }
        }
        predicates
    }

    pub fn entities_mentioned(&self, text: &str) -> Vec<String> {
        // 同一文本会在一轮对话中被多次询问，做多槽记忆复用；
        // 实体剖面或同义词表只在 observe_facts 中变化，届时（revision 推进）整体失效
        if let Some(found) = self
            .entity_mentions_memo
            .lock()
            .get(self.revision, text)
            .cloned()
        {
            self.perf.bump("memo_entity_mentions_hit");
            return found;
        }
        self.perf.bump("memo_entity_mentions_miss");
        let found = self.entities_mentioned_uncached(text);
        self.entity_mentions_memo
            .lock()
            .put(self.revision, text.to_string(), found.clone());
        found
    }

    fn entities_mentioned_uncached(&self, text: &str) -> Vec<String> {
        let normalized_text = self.aligner.normalize_cow(text);
        let mut found: Vec<(usize, String)> = Vec::new();
        for (i, profile) in self.aligner.profiles().iter().enumerate() {
            if profile.kind != TermKind::Entity {
                continue;
            }
            let Some(term_norm) = self.aligner.normalized_term_at(i) else {
                continue;
            };
            if term_norm.is_empty() {
                continue;
            }
            if normalized_text.contains(term_norm) {
                found.push((
                    profile.term.chars().count(),
                    self.aligner.canonical_form(&profile.term),
                ));
            }
        }
        for (alias, canonical) in self.aligner.synonyms() {
            if canonical.is_empty() {
                continue;
            }
            if text.contains(alias.as_str())
                && self.aligner.kind_of(canonical) == Some(TermKind::Entity)
            {
                found.push((alias.chars().count(), canonical.clone()));
            }
        }
        // 镜像对齐指称：整段词面（如「远山科技公司」）已在链接器里落地到
        // 已知身份；这里同步进实体提及，让上下文焦点与问答计划看到同一结果。
        for mention in self.linker.alignment_mentions(text) {
            if let Some(best) = mention.best() {
                found.push((mention.mention.chars().count(), best.entity.clone()));
            }
        }
        found.sort_by(|a, b| b.0.cmp(&a.0));
        let mut seen = HashSet::new();
        found
            .into_iter()
            .map(|(_, entity)| entity)
            .filter(|entity| seen.insert(entity.clone()))
            .collect()
    }

    /// 收集文本中出现的同义词对应的规范谓词集合
    fn mentioned_predicate_canonicals(&self, text: &str) -> Arc<HashSet<String>> {
        if let Some(canonicals) = self
            .predicate_canonicals_memo
            .lock()
            .get(self.revision, text)
            .cloned()
        {
            self.perf.bump("memo_predicate_canonicals_hit");
            return canonicals;
        }
        self.perf.bump("memo_predicate_canonicals_miss");
        let mut mentioned: HashSet<String> = HashSet::new();
        for (alias, canonical) in self.aligner.synonyms() {
            if text.contains(alias.as_str()) {
                mentioned.insert(canonical.clone());
            }
        }
        let mentioned = Arc::new(mentioned);
        self.predicate_canonicals_memo.lock().put(
            self.revision,
            text.to_string(),
            mentioned.clone(),
        );
        mentioned
    }

    /// 按缓存顺序返回首个被文本提及的已知谓词（原形）
    fn first_mentioned_known_predicate(
        &self,
        normalized_text: &str,
        mentioned_canonicals: &HashSet<String>,
    ) -> Option<String> {
        self.known_predicates_normalized_cache
            .iter()
            .find_map(|(predicate, predicate_norm)| {
                if mentioned_canonicals.contains(predicate)
                    || (!predicate_norm.is_empty() && normalized_text.contains(predicate_norm))
                {
                    Some(predicate.clone())
                } else {
                    None
                }
            })
    }

    /// 从问句结构提取**目标谓词**（最后一个「的」后的中心词）。
    ///
    /// 例：「姚明的配偶的体重」→「体重」，「姚明的配偶是谁」→「配偶」。
    /// 泛化中心词（资料/信息/关系……）不构成目标限制，返回 None。
    /// 属性谓词未必出现在 [`Self::known_predicates_normalized_cache`] 中，
    /// 因此这里按词面结构判断，而不是只查已知关系谓词。
    fn target_predicate_surface(&self, normalized_text: &str) -> Option<String> {
        let mut core = normalized_text
            .trim()
            .trim_end_matches(['？', '?', '。', '！', '!', ' '])
            .trim();
        let mut stripped = true;
        while stripped {
            stripped = false;
            for suffix in [
                "是多少",
                "是什么",
                "是谁",
                "是哪里",
                "是哪个",
                "是哪儿",
                "是哪位",
                "是哪些",
                "是几",
                "呢",
                "吗",
            ] {
                if let Some(rest) = core.strip_suffix(suffix) {
                    core = rest.trim();
                    stripped = true;
                    break;
                }
            }
        }
        // 没有「的」的句子（如「姚明和叶莉什么关系」）不构成链式目标限制。
        if !core.contains('的') {
            return None;
        }
        let segment = core.rsplit('的').next()?.trim();
        if segment.is_empty() {
            return None;
        }
        const GENERIC: &[&str] = &[
            "资料", "信息", "情况", "介绍", "关系", "内容", "属性", "特征", "数据", "详情", "细节",
            "知识", "事实", "谁", "什么", "多少", "哪儿", "哪里",
        ];
        if GENERIC.contains(&segment) {
            return None;
        }
        Some(self.aligner.canonical_form(segment))
    }

    /// 判断谓词是否被文本提及（等价于 predicate_mentioned_normalized）
    fn predicate_mentioned_normalized_with(
        &self,
        normalized_text: &str,
        predicate: &str,
        mentioned_canonicals: &HashSet<String>,
    ) -> bool {
        if mentioned_canonicals.contains(predicate) {
            return true;
        }
        let predicate_norm = self.aligner.normalize_cow(predicate);
        !predicate_norm.is_empty() && normalized_text.contains(predicate_norm.as_ref())
    }

    // 实体链接：候选生成 + 模糊容错 + 上下文消歧（详见 entity_linker 模块）
    pub const LINK_ACCEPT_THRESHOLD: f64 = 0.5;

    /// 接地清单：词面 ↔ 实体 ↔ 类型（与链接路径同源，供回答附引用）。
    pub fn grounding_mentions(&self, text: &str) -> Vec<crate::entity_linker::GroundedMention> {
        let hints = self.link_hints(text);
        self.linker.grounded_mentions(text, &hints)
    }

    /// 显式准入一个 OOV 词面（提名 → 用户确认 → 词表增长）。
    ///
    /// 返回（词面，类型）；只改会话内词表（链接缓存即时失效）；
    /// 持久化由调用方写入知识库后重新观察事实完成。
    pub fn admit_vocabulary(&mut self, surface: &str) -> Option<(String, String)> {
        let admitted = self.linker.admit_entity(surface)?;
        self.mention_lexicon
            .confirm(&admitted.0, &admitted.0, Some(&admitted.1), "admitted");
        self.link_memo.lock().clear();
        Some(admitted)
    }

    /// 会话级镜像对齐报告：规则、类型守卫、否证墓碑与回声词面。
    pub fn alignment_memory_report(&self) -> String {
        self.linker.alignment().render_report()
    }

    /// 解析一个可落地锚点：规范名本身或唯一别名；歧义/未登记返回 None。
    fn alignment_anchor(&self, surface: &str) -> Option<String> {
        let surface = surface.trim();
        if surface.is_empty() {
            return None;
        }
        if self.linker.is_known_entity(surface) {
            return Some(surface.to_string());
        }
        let mut candidates: Vec<String> = self
            .linker
            .surface_candidates(surface)
            .into_iter()
            .map(|candidate| candidate.entity)
            .collect();
        candidates.sort();
        candidates.dedup();
        (candidates.len() == 1).then(|| candidates.remove(0))
    }

    /// 学习一次实体词形对齐（会话级）。返回人类可读回执。
    pub fn learn_entity_alignment(&mut self, surface: &str, canonical: &str) -> String {
        self.learn_alignment_with_kind(surface, canonical, AlignmentKind::Entity)
    }

    /// 学习一次谓词词形对齐（会话级；只影响对齐记忆本身与审计）。
    pub fn learn_predicate_alignment(&mut self, surface: &str, canonical: &str) -> String {
        self.learn_alignment_with_kind(surface, canonical, AlignmentKind::Predicate)
    }

    fn learn_alignment_with_kind(
        &mut self,
        surface: &str,
        canonical: &str,
        kind: AlignmentKind,
    ) -> String {
        let surface = surface.trim();
        if surface.is_empty() {
            return "请给出用户词面，例如：对齐纠正：未名科技公司 = 未名科技".to_string();
        }
        let anchor = match kind {
            AlignmentKind::Entity => self.alignment_anchor(canonical),
            AlignmentKind::Predicate => {
                let canonical = self.aligner.canonical_form(canonical);
                (self.aligner.kind_of(&canonical) == Some(TermKind::Relation)).then_some(canonical)
            }
            AlignmentKind::Type => {
                let canonical = self.aligner.canonical_form(canonical);
                (self.aligner.kind_of(&canonical) == Some(TermKind::Type)).then_some(canonical)
            }
        };
        let Some(canonical) = anchor else {
            return format!(
                "对齐未生效：「{}」不是已确认的{}身份（落地律要求结果必须能被现有词表接住）。",
                canonical.trim(),
                kind.label()
            );
        };
        let expected_type = match kind {
            AlignmentKind::Entity => self
                .linker
                .entity_type(&canonical)
                .map(str::to_string)
                .or_else(|| {
                    self.generator
                        .entity_type_of(&canonical)
                        .map(str::to_string)
                }),
            _ => None,
        };
        let turn = self.linker.alignment().turn.saturating_add(1);
        let scope = AlignmentScope::Session;
        let outcome = self.linker.learn_alignment(
            turn,
            surface,
            &canonical,
            kind,
            expected_type.as_deref(),
            "会话纠正",
            scope,
        );
        if outcome.changed() {
            self.link_memo.lock().clear();
            self.entity_mentions_memo.lock().clear();
        }
        match outcome {
            LearnOutcome::Learned { rule_id, created } => format!(
                "{}镜像对齐：{}「{}」⇒「{}」（规则 {}，类型守卫 {}）。",
                if created { "已记住" } else { "已加强" },
                kind.label(),
                surface,
                canonical,
                rule_id,
                expected_type.as_deref().unwrap_or("未声明")
            ),
            LearnOutcome::Refuted { rule_id } => format!(
                "「{}」⇒「{}」的规则 {} 已被否证；本次只记支持，不复活墓碑（可用「对齐查询」复核）。",
                surface, canonical, rule_id
            ),
            LearnOutcome::Conflict { existing } => format!(
                "对齐冲突：「{}」已被规则 {} 指向另一个身份；未静默替换，请先用「对齐否决」清理旧规则。",
                surface, existing
            ),
            LearnOutcome::Unchanged => format!("「{}」与「{}」相同或无法形成词形规则。", surface, canonical),
        }
    }

    /// 反向选面：生成端用用户自己的说法，且选择的词面必须能再对齐回原身份。
    pub fn alignment_surface_choice(&self, canonical: &str) -> Option<(String, String)> {
        let canonical = self.alignment_anchor(canonical)?;
        let turn = self.linker.alignment().turn;
        let resolve = |surface: &str| -> Vec<String> {
            let mut entities: Vec<String> = self
                .linker
                .surface_candidates(surface)
                .into_iter()
                .map(|candidate| candidate.entity)
                .collect();
            entities.sort();
            entities.dedup();
            entities
        };
        let type_of = |entity: &str| -> Option<String> {
            self.linker.entity_type(entity).map(str::to_string)
        };
        self.linker
            .alignment()
            .surface_for_with(&canonical, turn, resolve, type_of)
    }

    /// 单次前向对齐查询（供交互命令/生成复核使用）。
    pub fn alignment_canonicalize(
        &self,
        surface: &str,
    ) -> Option<crate::alignment_memory::AlignmentMatch> {
        let turn = self.linker.alignment().turn;
        let resolve = |candidate: &str| -> Vec<String> {
            let mut entities: Vec<String> = self
                .linker
                .surface_candidates(candidate)
                .into_iter()
                .map(|linked| linked.entity)
                .collect();
            entities.sort();
            entities.dedup();
            entities
        };
        let type_of = |entity: &str| -> Option<String> {
            self.linker.entity_type(entity).map(str::to_string)
        };
        self.linker.alignment().canonicalize_with(
            surface,
            AlignmentKind::Entity,
            turn,
            resolve,
            type_of,
        )
    }

    /// 单次谓词侧对齐：把用户说法收束到已登记的关系/属性名。
    pub fn alignment_predicate_canonicalize(&self, surface: &str) -> Option<String> {
        let surface = surface.trim();
        if surface.is_empty() {
            return None;
        }
        let turn = self.linker.alignment().turn;
        let resolve = |candidate: &str| -> Vec<String> {
            let known = self.aligner.kind_of(candidate) == Some(TermKind::Relation)
                || self.aligner.kind_of(candidate) == Some(TermKind::Attribute)
                || self.predictor.has_predicate(candidate)
                || self.inverse_query_map.contains_key(candidate);
            (known && !candidate.trim().is_empty())
                .then(|| vec![candidate.to_string()])
                .unwrap_or_default()
        };
        let type_of = |_candidate: &str| -> Option<String> { None };
        self.linker
            .alignment()
            .canonicalize_with(surface, AlignmentKind::Predicate, turn, resolve, type_of)
            .map(|matched| matched.canonical)
    }

    /// 把问句中的会话谓词说法替换为已登记谓词；未知实体词面不参与改写。
    fn rewrite_predicate_alignment(&self, text: &str) -> Option<String> {
        let turn = self.linker.alignment().turn;
        let resolve = |candidate: &str| -> Vec<String> {
            let known = self.aligner.kind_of(candidate) == Some(TermKind::Relation)
                || self.aligner.kind_of(candidate) == Some(TermKind::Attribute)
                || self.predictor.has_predicate(candidate)
                || self.inverse_query_map.contains_key(candidate);
            (known && !candidate.trim().is_empty())
                .then(|| vec![candidate.to_string()])
                .unwrap_or_default()
        };
        let type_of = |_candidate: &str| -> Option<String> { None };
        let spans: Vec<_> = self
            .linker
            .alignment()
            .mention_spans_with(text, AlignmentKind::Predicate, turn, resolve, type_of)
            .into_iter()
            .filter(|matched| !self.is_known_entity(&matched.surface))
            .collect();
        if spans.is_empty() {
            return None;
        }
        let mut chars: Vec<char> = text.chars().collect();
        let mut spans = spans;
        spans.sort_by(|left, right| right.start.cmp(&left.start));
        for matched in spans {
            if matched.end > chars.len()
                || matched.start > matched.end
                || matched.surface.chars().count() != matched.end - matched.start
            {
                continue;
            }
            chars.splice(matched.start..matched.end, matched.canonical.chars());
        }
        Some(chars.into_iter().collect())
    }

    /// 交互入口：镜像对齐的查看、教学、否证、查询与反向选面。
    pub fn handle_alignment_command(&mut self, input: &str) -> Option<String> {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return None;
        }
        if matches!(
            trimmed,
            "对齐记忆" | "镜像对齐" | "对齐台账" | "镜像记忆" | "词形对齐"
        ) {
            return Some(self.alignment_memory_report());
        }
        for prefix in [
            "对齐纠正：",
            "对齐纠正:",
            "镜像对齐：",
            "镜像对齐:",
            "对齐：",
            "对齐:",
        ] {
            if let Some(spec) = trimmed.strip_prefix(prefix) {
                let Some((surface, canonical)) = parse_alignment_pair(spec) else {
                    return Some(
                        "请用「对齐纠正：用户词面 = 规范身份」的格式，例如：对齐纠正：未名科技公司 = 未名科技"
                            .to_string(),
                    );
                };
                return Some(self.learn_entity_alignment(surface, canonical));
            }
        }
        for prefix in ["对齐谓词：", "对齐谓词:", "谓词对齐：", "谓词对齐:"] {
            if let Some(spec) = trimmed.strip_prefix(prefix) {
                let Some((surface, canonical)) = parse_alignment_pair(spec) else {
                    return Some(
                        "请用「对齐谓词：用户说法 = 规范谓词」的格式，例如：对齐谓词：老婆 = 妻子"
                            .to_string(),
                    );
                };
                return Some(self.learn_predicate_alignment(surface, canonical));
            }
        }
        for prefix in ["对齐否决：", "对齐否决:", "否证对齐：", "否证对齐:"] {
            if let Some(spec) = trimmed.strip_prefix(prefix) {
                let Some((surface, canonical)) = parse_alignment_pair(spec) else {
                    return Some("请用「对齐否决：用户词面 = 规范身份」的格式。".to_string());
                };
                let turn = self.linker.alignment().turn.saturating_add(1);
                let count = self
                    .linker
                    .alignment_mut()
                    .refute_pair(turn, surface, canonical);
                if count == 0 {
                    self.link_memo.lock().clear();
                    self.entity_mentions_memo.lock().clear();
                    return Some(format!(
                        "没有找到生效中的「{}」⇒「{}」对齐规则；无需否证。",
                        surface.trim(),
                        canonical.trim()
                    ));
                }
                self.link_memo.lock().clear();
                self.entity_mentions_memo.lock().clear();
                return Some(format!(
                    "已否证「{}」⇒「{}」的 {} 条对齐规则；墓碑保留，后续同类观察不会自动复活。",
                    surface.trim(),
                    canonical.trim(),
                    count
                ));
            }
        }
        for prefix in ["对齐查询：", "对齐查询:", "镜像查询：", "镜像查询:"] {
            if let Some(surface) = trimmed.strip_prefix(prefix) {
                let surface = surface.trim();
                return Some(match self.alignment_canonicalize(surface) {
                    Some(matched) => format!(
                        "「{}」⇒「{}」（规则 {}，支持 {}，{}）",
                        matched.surface,
                        matched.canonical,
                        matched.rule_id,
                        matched.support,
                        matched.reason
                    ),
                    None => format!(
                        "「{}」没有可落地的镜像对齐（未登记、歧义、类型不符或词面本身已是另一身份）。",
                        surface
                    ),
                });
            }
        }
        for prefix in ["对齐答法：", "对齐答法:", "镜像答法：", "镜像答法:"] {
            if let Some(canonical) = trimmed.strip_prefix(prefix) {
                let canonical = canonical.trim();
                return Some(match self.alignment_surface_choice(canonical) {
                    Some((surface, reason)) => format!(
                        "生成端将用「{}」称呼「{}」（{}；可再对齐回同一身份）。",
                        surface, canonical, reason
                    ),
                    None => format!(
                        "「{}」没有可逆的镜像答法；生成端继续使用规范名。",
                        canonical
                    ),
                });
            }
        }
        None
    }

    /// 提及格全局解码报告（只读审计面：不改变实际链接路径）。
    pub fn mention_lattice_report(&self, text: &str) -> String {
        let hints = self.link_hints(text);
        self.linker.lattice_report(text, &hints)
    }

    /// 身份台账：规范身份 ↔ 别名 ↔ 歧义（URI 身份的只读审计视图，确定排序）。
    pub fn identity_report(&self) -> String {
        self.linker.identity_registry().audit_lines().join("\n")
    }

    /// 文本的实体链接结果：一次查询流程内同一文本会被多处链接，做单条记忆复用
    fn linked_mentions(&self, text: &str) -> Arc<Vec<LinkedMention>> {
        if let Some(mentions) = self.link_memo.lock().get(self.revision, text).cloned() {
            self.perf.bump("memo_link_hit");
            return mentions;
        }
        self.perf.bump("memo_link_miss");
        let hints = self.link_hints(text);
        let mentions = Arc::new(self.linker.link_with_context(text, &hints));
        self.link_memo
            .lock()
            .put(self.revision, text.to_string(), mentions.clone());
        mentions
    }

    pub fn linked_entities(&self, text: &str) -> Vec<String> {
        let mut seen = HashSet::new();
        self.linked_mentions(text)
            .iter()
            .filter_map(|mention| mention.best())
            .filter(|candidate| candidate.score >= Self::LINK_ACCEPT_THRESHOLD)
            .map(|candidate| candidate.entity.clone())
            .filter(|entity| seen.insert(entity.clone()))
            .collect()
    }

    pub fn best_linked_entity(&self, text: &str) -> Option<String> {
        self.linked_mentions(text)
            .iter()
            .filter_map(|mention| {
                mention
                    .best()
                    .map(|candidate| (candidate.score, candidate.entity.clone()))
            })
            .filter(|(score, _)| *score >= Self::LINK_ACCEPT_THRESHOLD)
            .max_by(|a, b| {
                a.0.partial_cmp(&b.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| b.1.cmp(&a.1))
            })
            .map(|(_, entity)| entity)
    }

    /// 问题谓词/类型线索（只读；供语境指称消歧表决使用）。
    pub fn reference_hints(&self, text: &str) -> Vec<String> {
        self.link_hints(text)
    }

    pub fn entity_link_report(&self, text: &str) -> String {
        let mentions = self.linked_mentions(text);
        self.linker.report_for_mentions(text, mentions.as_slice())
    }

    // 仅返回"模糊匹配"的链接说明，用于在回答前向用户提示容错链接
    pub fn entity_link_notes(&self, text: &str) -> Vec<String> {
        self.linked_mentions(text)
            .iter()
            .filter_map(|mention| {
                let candidate = mention.best()?;
                if candidate.method != LinkMethod::Fuzzy {
                    return None;
                }
                Some(format!(
                    "实体链接：「{}」→「{}」（置信度 {:.2}）",
                    mention.mention, candidate.entity, candidate.score
                ))
            })
            .collect()
    }

    /// 提及账本：每个指称一张证据收据（边界 + 类型证据 × 疑问词期待 + 焦点先验）。
    ///
    /// 只读审计面：不改变链接与作答行为，输出确定可 golden。
    pub fn mention_ledger_report(&self, text: &str) -> String {
        let hints = self.link_hints(text);
        let expectations = crate::function_words::answer_type_keywords(text);
        let receipts =
            self.linker
                .mention_receipts(text, &hints, &expectations, self.context.focus());
        let suspects = crate::entity_linker::oov_suspects(text, &receipts);
        crate::entity_linker::render_mention_ledger(text, &receipts, &expectations, &suspects)
    }

    fn link_hints(&self, text: &str) -> Vec<String> {
        let normalized_text = self.aligner.normalize_cow(text);
        let mentioned_canonicals = self.mentioned_predicate_canonicals(text);
        self.known_predicates_normalized_cache
            .iter()
            .filter(|(predicate, predicate_norm)| {
                mentioned_canonicals.contains(predicate)
                    || (!predicate_norm.is_empty() && normalized_text.contains(predicate_norm))
            })
            .map(|(predicate, _)| predicate.clone())
            .collect()
    }
}

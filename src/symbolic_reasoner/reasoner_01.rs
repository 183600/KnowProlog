impl SymbolicReasoner {
    /// 把本体的实体/谓词登记转成句法重写器可用的词表视图：
    /// 只有已登记项才放行，未登记的一律返回 None（保守弃权）。
    fn relation_lexicon(&self) -> impl RelationLexicon + '_ {
        struct View<'a>(&'a SymbolicReasoner);

        impl RelationLexicon for View<'_> {
            fn canonical_entity(&self, raw: &str) -> Option<String> {
                // 只接受精确登记的词面/会话事实或显式同义词；不用模糊对齐
                // 把整个短语（如「张三的师傅」）误吸附到一个短实体上。
                let canonical = self.0.aligner.canonical_form(raw);
                if self.0.is_known_entity(raw) {
                    return Some(canonical);
                }
                let alias = canonical != raw
                    && self
                        .0
                        .aligner
                        .synonyms()
                        .get(raw)
                        .is_some_and(|target| target == &canonical);
                (alias && self.0.is_known_entity(&canonical)).then_some(canonical)
            }

            fn canonical_predicate(&self, raw: &str) -> Option<String> {
                // 谓词同样先做精确判定：已登记关系、精确事实谓词、逆谓词、
                // 同义词别名或会话镜像对齐，任一成立才归一；模糊对齐不参与边界切分。
                let aligned = self.0.alignment_predicate_canonicalize(raw);
                let canonical = aligned
                    .clone()
                    .unwrap_or_else(|| self.0.aligner.canonical_form(raw));
                let alias = canonical != raw
                    && self
                        .0
                        .aligner
                        .synonyms()
                        .get(raw)
                        .is_some_and(|target| target == &canonical);
                // 生成器词面（knows → 认识）是观察事实时登记的可用谓词，
                // 与对齐记忆、预测器索引同权参与句法边界校验。
                let generated = self.0.generator.canonical_predicate(raw);
                let exact = aligned.is_some()
                    || generated.is_some()
                    || self.0.predictor.has_predicate(raw)
                    || self.0.aligner.kind_of(raw) == Some(TermKind::Relation)
                    || self.0.inverse_query_map.contains_key(raw)
                    || alias;
                exact.then(|| {
                    let base = if aligned.is_some() {
                        canonical.as_str()
                    } else {
                        raw
                    };
                    self.0.resolve_question_predicate(base).0
                })
            }
        }

        View(self)
    }

    /// 装载已采纳的习得重写规则；只在读法候选层生效。
    pub fn set_rewrite_book(&mut self, book: RewriteBook) {
        self.rewrite_book = Some(book);
    }

    pub fn rewrite_book(&self) -> Option<&RewriteBook> {
        self.rewrite_book.as_ref()
    }

    /// 锚定重写的只读证据视图：链接器类型 + 谓词画像 + 本体闭包层级。
    fn anchor_evidence<'a>(
        &'a self,
        capsule: Option<&'a OntologyCapsule>,
    ) -> AnchorEvidenceView<'a> {
        AnchorEvidenceView {
            linker: &self.linker,
            registry: self.law_registry.as_ref(),
            capsule,
        }
    }
    /// 审计辅助：词条是否是任一实体的属性键、取值或实体名本身。
    /// 供对话忠信度检查（回答引用必须可回溯）使用。
    pub fn attribute_token_known(&self, token: &str) -> bool {
        self.attribute_index
            .iter()
            .any(|((entity, attribute), values)| {
                entity == token
                    || attribute == token
                    || crate::symbolic_generator::localized_predicate_key(attribute) == Some(token)
                    || values.iter().any(|value| value == token)
            })
    }

    pub fn new(config: ReasoningConfig, synonyms: HashMap<String, String>) -> Self {
        Self::with_perf_ledger(config, synonyms, Arc::new(PerfLedger::new()))
    }

    /// 与 `new` 相同，但可注入共享的性能台账（会话级操作计数）。
    pub fn with_perf_ledger(
        config: ReasoningConfig,
        synonyms: HashMap<String, String>,
        perf: Arc<PerfLedger>,
    ) -> Self {
        let semantic = SemanticMonitor::with_options(
            config.semantic_monitor.window,
            config.semantic_monitor.budget.clone(),
        );
        semantic.set_mode(config.semantic_monitor.mode);
        let aligner = OntologyAligner::new(synonyms, config.ontology_alignment_threshold);
        let embedding = SymbolicEmbedding::new(config.embedding_similarity_floor);
        let mut linker = EntityLinker::new();
        for (alias, canonical) in aligner.synonyms() {
            linker.register_alias(alias, canonical);
        }
        let generator =
            SymbolicGenerator::new(config.max_profile_sentences, config.completion_min_score);
        let associations = AssociationMiner::new(
            config.association_min_support,
            config.association_min_confidence,
            config.association_min_lift,
        );
        let resolver =
            ResolutionEngine::new(config.resolution_max_depth, config.resolution_max_solutions);
        let abduction =
            AbductionEngine::new(config.resolution_max_depth, config.abduction_max_hypotheses);
        let mut abduction = abduction;
        abduction.set_max_scenario_size(config.abduction_max_scenario_size);
        let walks = GraphWalkEmbedding::new(
            config.graph_walk_damping,
            config.graph_walk_iterations,
            config.embedding_similarity_floor,
        );
        let soft = MarkovLogicEngine::new(
            config.soft_logic_max_iterations,
            config.soft_logic_min_probability,
        );
        let problog = ProblogEngine::new(config.problog_max_depth, config.problog_max_proofs);
        let discourse =
            DiscoursePlanner::new(config.discourse_max_rounds, config.discourse_max_sentences);
        let holo = HolographicMemory::new(
            config.holographic_dimension,
            config.holographic_similarity_floor,
            config.holographic_max_hits,
        );
        let mut reasoner = Self {
            config,
            aligner,
            predictor: LinkPredictor::new(),
            prediction_drafts: parking_lot::Mutex::new(PredictionDrafts::new(
                PREDICTION_DRAFT_CAPACITY,
            )),
            completion: CompletionLedger::new(),
            predictive: PredictiveElaboration::new(),
            laws: LawBook::new(),
            law_registry: None,
            persisted_laws: None,
            context: DialogueContext::new(5),
            epistemic_turns: EpistemicTurnBook::new(),
            embedding,
            schema_view: SchemaView::new(),
            walks,
            soft,
            problog,
            lessons: Vec::new(),
            lessons_parsed: Vec::new(),
            planner: QueryPlanner::new(),
            linker,
            generator,
            rewrite_book: None,
            retrieval: parking_lot::Mutex::new(DenseRetriever::new(8, 0.85, 12)),
            contextual_last: parking_lot::Mutex::new(None),
            formal_clarification: parking_lot::Mutex::new(None),
            contextual_memory: ReadingMemory::new(512),
            attributes: HashSet::new(),
            attribute_index: HashMap::new(),
            associations,
            resolver,
            abduction,
            frames: FrameMemory::new(),
            pending_questions: PendingQuestionLedger::new(),
            graph_session: parking_lot::Mutex::new(None),
            session_facts: Vec::new(),
            slots: SlotLedger::new(),
            assumption_world: None,
            session_negative_facts: Vec::new(),
            session_clauses: Vec::new(),
            session_receipts: Vec::new(),
            clarification: None,
            last_abduction_goal: None,
            last_abduction_vars: Vec::new(),
            known_predicates_cache: Vec::new(),
            known_predicates_normalized_cache: Vec::new(),
            proposition_surfaces: Vec::new(),
            link_memo: parking_lot::Mutex::new(RevisionLru::new(MEMO_CAPACITY)),
            entity_mentions_memo: parking_lot::Mutex::new(RevisionLru::new(MEMO_CAPACITY)),
            predicate_canonicals_memo: parking_lot::Mutex::new(RevisionLru::new(MEMO_CAPACITY)),
            inverse_query_map: HashMap::new(),
            discourse,
            holo,
            revision: 0,
            derived_stamps: BTreeMap::new(),
            dependency_stamps: BTreeMap::new(),
            perf,
            nlp_config: None,
            read_traces: ReadTraceLedger::new(),
            semantic,
            lexical_memory: LexicalMemory::default(),
            mention_lexicon: MentionLexicon::default(),
            engine_fault: None,
        };
        reasoner.rebuild_knowledge_caches();
        reasoner
    }

    /// 从预热基座分叉一个新会话：知识派生状态按值深拷贝，会话态清零。
    ///
    /// 正确性契约：分叉结果与「新建实例 + 重放同一批 `observe_facts`」在固定
    /// 查询集上逐字节一致（同一知识版本、同一配置、同一同义词表），
    /// 由 `fork_matches_fresh_observe_byte_for_byte` 等测试守护。
    /// 分叉基座以 `Arc<str>` 共享，可跨线程使用（`Send`）。
    pub fn fork_session(&self) -> Self {
        let mut fork = self.clone();
        fork.reset_session_state();
        fork
    }

    /// 用一份新的图谱快照替换推理器的全局知识，同时保留会话态。
    ///
    /// `observe_facts` 只增不减；这里以同一配置、同义词与性能台账重建基座，
    /// 再恢复会话快照（焦点/历史/会话事实/规则/帧），
    /// 使查询立即看到更新后的图谱。
    pub fn rebuild_from_graph(&mut self, facts: &[PrologFact]) {
        let synonyms = self.aligner.synonyms().clone();
        let snapshot = self.dialogue_snapshot();
        let law_registry = self.law_registry.clone();
        let persisted_laws = self.persisted_laws.clone();
        let clarification = self.clarification.clone();
        let last_goal = self.last_abduction_goal.clone();
        let last_vars = self.last_abduction_vars.clone();
        let engine_fault = self.engine_fault.clone();
        let lexical_memory = self.lexical_memory.clone();
        let mention_lexicon = self.mention_lexicon.clone();
        let contextual_memory = self.contextual_memory.clone();
        let rewrite_book = self.rewrite_book.clone();
        let schema_view = self.schema_view.clone();
        let mut fresh =
            Self::with_perf_ledger(self.config.clone(), synonyms, Arc::clone(&self.perf));
        fresh.schema_view = schema_view.clone();
        fresh.embedding.set_schema(Some(schema_view));
        // 词表是词法记忆：待定提名与墓碑不能因图谱重建而消失；
        // 当前图谱实体仍由随后的 observe_facts 重新确认。
        fresh.mention_lexicon = mention_lexicon;
        fresh.rewrite_book = rewrite_book;
        fresh.observe_facts(facts);
        fresh.lexical_memory = lexical_memory;
        fresh.contextual_memory = contextual_memory;
        // 快照来自自身：胶囊公理摘要必然通过（除非内存中的快照被外部改写）。
        // 万一被拒，也保留新图谱与其余会话态，不让重建失败丢掉知识。
        let _ = fresh.restore_dialogue_snapshot(&snapshot);
        if let Some(registry) = law_registry {
            fresh.configure_laws(&registry);
        }
        if let Some(persisted) = persisted_laws {
            fresh.load_persisted_laws(persisted);
        }
        fresh.clarification = clarification;
        fresh.last_abduction_goal = last_goal;
        fresh.last_abduction_vars = last_vars;
        fresh.engine_fault = engine_fault;
        *self = fresh;
    }

    /// 清空会话态：焦点/历史、会话事实与临时规则、澄清、教学、记忆缓存、
    /// 注入故障。知识派生状态（剖面/规则/向量/台账）原样保留。
    fn reset_session_state(&mut self) {
        self.context.clear();
        self.epistemic_turns.clear();
        *self.contextual_last.lock() = None;
        *self.formal_clarification.lock() = None;
        self.contextual_memory.clear();
        // 假设世界是会话态：分叉/重置时整体丢弃，不带走任何临时前提。
        self.discard_assumption_world();
        let lessons_were_taught = !self.lessons_parsed.is_empty();
        self.lessons.clear();
        self.lessons_parsed.clear();
        self.frames = FrameMemory::new();
        self.pending_questions.clear();
        *self.graph_session.lock() = None;
        // 会话事实可能已被索引为证据；分叉/重置时把非图谱来源的条目撤出索引。
        let session_facts = std::mem::take(&mut self.session_facts);
        for atom in &session_facts {
            if !self.graph_supports_atom(atom) {
                self.unindex_session_atom(atom);
            }
        }
        self.session_negative_facts.clear();
        self.slots = SlotLedger::new();
        self.session_clauses.clear();
        // 补全回合属于会话态：分叉/重置不得把已确认候选与规则带进新会话。
        self.completion = CompletionLedger::new();
        self.predictive = PredictiveElaboration::new();
        self.clarification = None;
        self.last_abduction_goal = None;
        self.last_abduction_vars.clear();
        self.abduction.reset_session();
        self.link_memo.lock().clear();
        self.entity_mentions_memo.lock().clear();
        self.predicate_canonicals_memo.lock().clear();
        self.prediction_drafts.lock().clear();
        self.read_traces.clear();
        self.semantic.clear();
        self.engine_fault = None;
        // 会话事实/规则可能已注入归结、软逻辑与概率程序；分叉后必须用
        // 纯图谱基座重建，否则新会话仍能推出上一会话的临时前提。
        if self.config.enable_resolution {
            self.rebuild_resolver();
        }
        if self.config.enable_soft_logic {
            self.rebuild_soft_logic();
        } else if lessons_were_taught && self.config.enable_problog {
            self.rebuild_problog();
        }
    }

    /// 注入引擎层故障（宿主在 Prolog / 存储失败后调用）：
    /// 之后对话管理器的知识问句会走 EngineFault 路径。
    pub fn note_engine_fault(&mut self, message: impl Into<String>) {
        let message = message.into();
        let message = message.trim();
        if !message.is_empty() {
            self.engine_fault = Some(message.to_string());
        }
    }

    /// 清除引擎层故障（默认实现里故障是粘性的，仅在宿主确认恢复后清除）。
    pub fn clear_engine_fault(&mut self) {
        self.engine_fault = None;
    }

    /// 当前引擎层故障（无则 None）。
    pub fn engine_fault(&self) -> Option<String> {
        self.engine_fault.clone()
    }

    /// 当前知识版本（每次内容确有变化的观察 +1）。
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// 共享的性能台账（与 `OxigraphManager` 可为同一实例）。
    pub fn perf_ledger(&self) -> Arc<PerfLedger> {
        Arc::clone(&self.perf)
    }

    /// 注入 NLP 配置，启用读法观测。失败轮会留下只读轨迹回执；
    /// 成功轮不额外分词。未注入时失败轮保持原有的零开销回答路径。
    pub fn set_nlp_config(&mut self, config: NlpConfig) {
        self.nlp_config = Some(config);
        self.read_traces.clear();
    }

    /// 应用热替换的语言策略：同义词、分词/NLU 配置、生成预算与双向语义规则。
    ///
    /// 只触碰「怎么说」这一层，不重建事实、焦点、会话假设或已学规则。
    /// 同义词是单调追加：删除旧映射不会在会话中撤销已经学到的指称；
    /// 这类撤销应由重启按新配置重建。
    pub fn apply_language_policy(&mut self, config: &SystemConfig) -> LanguagePolicySync {
        let mut sync = LanguagePolicySync::default();
        if let Some(synonyms) = &config.synonyms {
            for (alias, canonical) in &synonyms.mappings {
                let (alias, canonical) = (alias.trim(), canonical.trim());
                if alias.is_empty() || canonical.is_empty() || alias == canonical {
                    continue;
                }
                self.linker.register_alias(alias, canonical);
                if self.aligner.register_synonym(alias, canonical) {
                    sync.synonyms_added += 1;
                }
            }
        }
        if sync.synonyms_added > 0 {
            self.rebuild_knowledge_caches();
        }

        let enable_generation = self.config.enable_generation;
        let budget = (
            self.config.max_profile_sentences,
            self.config.completion_min_score,
        );
        self.config.enable_generation = config.reasoning.enable_generation;
        self.config.max_profile_sentences = config.reasoning.max_profile_sentences;
        self.config.completion_min_score = config.reasoning.completion_min_score;
        sync.generation_updated = enable_generation != self.config.enable_generation
            || budget
                != (
                    self.config.max_profile_sentences,
                    self.config.completion_min_score,
                );
        self.generator.set_generation_budget(
            config.reasoning.max_profile_sentences,
            config.reasoning.completion_min_score,
        );

        if config.semantic_grammar.enabled {
            self.generator
                .attach_semantic_grammar(crate::semantic_grammar::RulePack::from_config(config));
            sync.semantic_grammar = true;
        } else {
            self.generator.detach_semantic_grammar();
            sync.semantic_grammar = false;
        }
        if config.frame_semantics.enabled {
            self.generator
                .attach_frame_book(crate::frame_semantics::FrameBook::from_config(
                    &config.frame_semantics,
                ));
        } else {
            self.generator.detach_frame_book();
        }

        self.set_nlp_config(config.nlp.clone());
        sync.nlp_updated = true;
        sync
    }

    /// 恢复跨会话词法记忆（宿主从命名图读取后注入）。
    pub fn set_lexical_memory(&mut self, memory: LexicalMemory) {
        self.lexical_memory = memory;
    }

    /// 当前词法记忆（宿主据此落图）。
    pub fn lexical_memory(&self) -> &LexicalMemory {
        &self.lexical_memory
    }

    /// 当前提及词表（只读；图谱实体、确认别名与显式准入的切分记忆）。
    pub fn mention_lexicon(&self) -> &MentionLexicon {
        &self.mention_lexicon
    }

    /// 用当前提及词表做覆盖层切分；词表为空时与旧路径逐 token 相同。
    pub fn segment_with_lexicon(
        &self,
        text: &str,
        config: &NlpConfig,
    ) -> Vec<crate::nlp::TaggedWord> {
        crate::nlp::preprocess_with_lexicon(text, config, &self.mention_lexicon)
    }

    /// 与 [`Self::segment_with_lexicon`] 同序，保留边界与来源证据。
    pub fn segment_annotated_with_lexicon(
        &self,
        text: &str,
        config: &NlpConfig,
    ) -> Vec<crate::nlp::AnnotatedWord> {
        crate::nlp::preprocess_annotated_with_lexicon(text, config, &self.mention_lexicon)
    }

    /// 词表完整报告（确认/待定/墓碑与来源，按词面排序）。
    pub fn mention_lexicon_report(&self) -> String {
        self.mention_lexicon.render_report()
    }

    /// 词表在指定文本上的覆盖选路报告（只读审计）。
    pub fn mention_lexicon_match_report(&self, text: &str) -> String {
        self.mention_lexicon.render_match_report(text)
    }

    /// 提名一个 OOV 词面（只进待定，不参与切分，也不写图谱）。
    pub fn nominate_mention(&mut self, surface: &str, etype: Option<&str>) -> bool {
        self.mention_lexicon.nominate(surface, etype, "oov")
    }

    /// 拒绝一个词面：墓碑挡板，后续观察不得重新确认，也不再覆盖切分。
    pub fn reject_mention(&mut self, surface: &str) -> bool {
        let changed = self.mention_lexicon.reject(surface, "user");
        if changed {
            self.link_memo.lock().clear();
        }
        changed
    }

    /// 观察一轮原文，更新词形证据与量词期待；未注入 NLP 配置时不增加成本。
    pub fn observe_lexical_text(&mut self, text: &str) -> Option<ObservationSummary> {
        let config = self.nlp_config.as_ref()?;
        let turn = self.lexical_memory.turn.saturating_add(1);
        Some(self.lexical_memory.observe_text(turn, text, config))
    }

    /// 某个词面的弱类型期待（不写 `rdf:type`，只作软约束）。
    pub fn lexical_type_expectation(&self, surface: &str) -> Option<TypeExpectation> {
        self.lexical_memory
            .type_expectation(surface)
            .or_else(|| self.lexical_memory.classifier_expectation(surface))
    }

    /// 构建一次知识感知的读法轨迹（只读；不执行回答）。
    pub fn lexical_trace(&self, question: &str) -> Option<LexicalTrace> {
        let config = self.nlp_config.as_ref()?;
        let plan = self.planner.parse(question).or_else(|| {
            query_rewrite::rewrite(question, &self.relation_lexicon()).map(|rewrite| rewrite.plan)
        });
        let mut trace = lexical_trace::trace_question_with_mention_lexicon(
            question,
            config,
            plan.as_ref(),
            Some(&self.mention_lexicon),
        );
        trace.fidelity = TraceFidelity::Reconstructed;
        if let Some(plan) = &plan {
            self.enrich_trace_plan_terms(&mut trace, plan, question);
        }
        self.enrich_trace_context(&mut trace);
        self.enrich_trace_oov(&mut trace, question);
        Some(trace)
    }

    /// 用会话上下文给「省略/缺实体」残差补充候选：上一轮谓词可以补
    /// 「李四呢」的谓词，当前焦点可以补「体重是多少」的实体。只补候选，
    /// 不替用户做选择；是否采用由调用方/追问流程决定。
    fn enrich_trace_context(&self, trace: &mut LexicalTrace) {
        if let Some(predicate) = self.context.last_predicate() {
            trace
                .reading
                .add_candidates(ResidualKind::Ellipsis, [predicate.to_string()]);
        }
        if let Some(focus) = self.context.focus() {
            trace
                .reading
                .add_candidates(ResidualKind::MissingEntity, [focus.to_string()]);
        }
    }

    /// 只返回读法残差表（不渲染 token 全文），供 API/评测直接使用。
    pub fn read_question(&self, question: &str) -> Option<Reading> {
        self.lexical_trace(question).map(|trace| trace.reading)
    }

    /// 读法追踪报告：token/来源/边界/计划/失败分类，并标注回答管线是否命中。
    pub fn lexical_trace_report(&self, question: &str) -> String {
        let Some(mut trace) = self.lexical_trace(question) else {
            return "读法追踪未启用：推理器没有注入 NLP 配置。".to_string();
        };
        let answered = self.answer_impl(question).is_some();
        trace.answer_available = Some(answered);
        if !answered
            && trace.plan.is_some()
            && !trace.failures.iter().any(|failure| {
                matches!(
                    failure.kind,
                    ReadFailureKind::UnknownEntity
                        | ReadFailureKind::UnknownPredicate
                        | ReadFailureKind::ClosureMarkerInToken
                )
            })
        {
            trace.push_failure(
                ReadStage::Execute,
                ReadFailureKind::ExecutionMiss,
                "计划已识别，但回答管线没有产生结果（事实缺口/缺少推理律/截断需在推理侧继续区分）",
            );
            trace.reading.push(
                Residual::new(
                    ResidualKind::ExecutionMiss,
                    ResidualStage::Execute,
                    ResidualSeverity::Blocking,
                    RepairKind::AcquireFact,
                    "计划已识别，但执行阶段没有产生结果",
                )
                .expected("事实、推理律或可执行预算"),
            );
        }
        self.perf.bump("lexical_trace_reports");
        self.perf.bump_by(
            "lexical_trace_unknown_tokens",
            trace.unknown_tokens().count() as u64,
        );
        self.perf.bump_by(
            "lexical_trace_marker_violations",
            trace.boundary_violations() as u64,
        );
        self.perf
            .bump_by("lexical_trace_oov", trace.oov.len() as u64);
        trace.render_report()
    }

    /// 有界失败回执台账（不含原文）；进程重启即失，不进入会话快照。
    pub fn read_trace_ledger_report(&self) -> String {
        self.read_traces.report()
    }

    pub fn read_trace_receipts(&self) -> Vec<TraceReceipt> {
        self.read_traces.snapshot()
    }

    pub fn semantic_metrics(&self) -> WindowMetrics {
        self.semantic.snapshot()
    }

    pub fn semantic_observations(&self) -> Vec<SemanticObservation> {
        self.semantic.observations()
    }

    pub fn semantic_advice(&self) -> SearchAdvice {
        self.semantic.advice()
    }

    pub fn semantic_report(&self) -> String {
        self.semantic.report()
    }

    pub fn set_semantic_control_mode(&self, mode: ControlMode) {
        self.semantic.set_mode(mode);
    }

    pub fn clear_semantic_monitor(&self) {
        self.semantic.clear();
    }

    fn enrich_trace_plan_terms(&self, trace: &mut LexicalTrace, plan: &QueryPlan, question: &str) {
        let (entities, predicates) = lexical_trace::plan_terms(plan);
        for entity in entities {
            let resolution = self.aligner.resolve_candidates(&entity, 3);
            let candidates: Vec<String> = resolution
                .candidates
                .iter()
                .map(|candidate| candidate.target().to_string())
                .filter(|candidate| candidate != &entity && self.is_known_entity(candidate))
                .collect();
            if self.is_known_entity(&entity) {
                // 只有真正存在多个可链接读法时才报歧义；模糊匹配到的单一候选
                // 只是对齐证据，不构成「需要澄清」。
                if resolution.state == AlignmentState::Ambiguous && candidates.len() >= 2 {
                    let mut residual = Residual::new(
                        ResidualKind::AmbiguousEntity,
                        ResidualStage::Link,
                        ResidualSeverity::Blocking,
                        RepairKind::ChooseCandidate,
                        "实体有多个会改变答案的候选读法",
                    )
                    .expected("一个确定的实体")
                    .candidates(candidates);
                    if let Some(span) = term_span(question, &entity) {
                        residual = residual.surface(question, span);
                    }
                    trace.reading.push(residual);
                }
                continue;
            }

            trace.push_failure(
                ReadStage::Link,
                ReadFailureKind::UnknownEntity,
                format!("实体槽「{}」未在知识图谱登记", entity),
            );
            let repair = if candidates.is_empty() {
                RepairKind::SupplyEntity
            } else {
                RepairKind::ChooseCandidate
            };
            let mut residual = Residual::new(
                ResidualKind::UnknownEntity,
                ResidualStage::Link,
                ResidualSeverity::Blocking,
                repair,
                format!("实体槽「{}」未在知识图谱登记", entity),
            )
            .expected("已登记实体")
            .candidates(candidates);
            if let Some(span) = term_span(question, &entity) {
                residual = residual.surface(question, span);
            }
            trace.reading.push(residual);
        }
        for predicate in predicates {
            if self.is_known_predicate(&predicate) {
                continue;
            }
            trace.push_failure(
                ReadStage::Link,
                ReadFailureKind::UnknownPredicate,
                format!("谓词槽「{}」未在谓词画像/事实中登记", predicate),
            );
            let resolution = self.aligner.resolve_candidates(&predicate, 3);
            let candidates: Vec<String> = resolution
                .candidates
                .iter()
                .map(|candidate| candidate.target().to_string())
                .filter(|candidate| candidate != &predicate && self.is_known_predicate(candidate))
                .collect();
            let repair = if candidates.is_empty() {
                RepairKind::SupplyPredicate
            } else {
                RepairKind::ChooseCandidate
            };
            let mut residual = Residual::new(
                ResidualKind::UnknownPredicate,
                ResidualStage::Link,
                ResidualSeverity::Blocking,
                repair,
                format!("谓词槽「{}」未在谓词画像/事实中登记", predicate),
            )
            .expected("已登记谓词/关系")
            .candidates(candidates);
            if let Some(span) = term_span(question, &predicate) {
                residual = residual.surface(question, span);
            }
            trace.reading.push(residual);
        }
    }

    fn enrich_trace_oov(&self, trace: &mut LexicalTrace, question: &str) {
        let expectations = crate::function_words::answer_type_keywords(question);
        let hints = self.link_hints(question);
        let receipts =
            self.linker
                .mention_receipts(question, &hints, &expectations, self.context.focus());
        trace.oov = crate::entity_linker::oov_suspects(question, &receipts)
            .into_iter()
            .map(|suspect| crate::lexical_trace::OovTrace {
                text: suspect.text,
                start: suspect.start,
                kind: match suspect.kind {
                    crate::entity_linker::OovKind::Quoted => "quoted".to_string(),
                    crate::entity_linker::OovKind::Suffix => "suffix".to_string(),
                },
                reason: suspect.reason,
                suggested_type: suspect.suggested_type,
            })
            .collect();
        for oov in &mut trace.oov {
            if oov.suggested_type.is_some() {
                continue;
            }
            if let Some(expectation) = self.lexical_memory.type_expectation(&oov.text) {
                oov.suggested_type = Some(expectation.expected_type);
            }
        }
    }

    /// 失败轮的只读轨迹：有界回执 + 有限枚举指标。成功轮不调用。
    fn record_read_failure(&self, question: &str) {
        let Some(mut trace) = self.lexical_trace(question) else {
            return;
        };
        trace.answer_available = Some(false);
        if trace.plan.is_some()
            && trace.failures.iter().all(|failure| {
                !matches!(
                    failure.kind,
                    ReadFailureKind::UnknownEntity
                        | ReadFailureKind::UnknownPredicate
                        | ReadFailureKind::NoQuestionShape
                )
            })
        {
            trace.push_failure(
                ReadStage::Execute,
                ReadFailureKind::ExecutionMiss,
                "计划已识别，但回答管线没有产生结果",
            );
            trace.reading.push(
                Residual::new(
                    ResidualKind::ExecutionMiss,
                    ResidualStage::Execute,
                    ResidualSeverity::Blocking,
                    RepairKind::AcquireFact,
                    "计划已识别，但执行阶段没有产生结果",
                )
                .expected("事实、推理律或可执行预算"),
            );
        }
        let receipt = self.read_traces.record(&trace);
        self.perf.bump("lexical_trace_failures");
        self.perf.bump_by(
            "lexical_trace_unknown_tokens",
            trace.unknown_tokens().count() as u64,
        );
        self.perf.bump_by(
            "lexical_trace_marker_violations",
            trace.boundary_violations() as u64,
        );
        self.perf
            .bump_by("lexical_trace_oov", trace.oov.len() as u64);
        for stage in &receipt.failure_stages {
            self.perf.bump(&format!("lexical_trace_stage_{}", stage));
        }
        for kind in &receipt.failure_kinds {
            self.perf.bump(&format!("lexical_trace_failure_{}", kind));
        }
    }

    /// 记录一次派生重建：派生器戳上当前知识版本，并在性能台账计一次数。
    fn stamp_rebuild(&mut self, deriver: &'static str) {
        self.derived_stamps.insert(deriver, self.revision);
        self.perf.bump(match deriver {
            "link_rules" => "rebuild_link_rules",
            "associations" => "rebuild_associations",
            "resolver" => "rebuild_resolver",
            "soft_logic" => "rebuild_soft_logic",
            "problog" => "rebuild_problog",
            "knowledge_caches" => "rebuild_knowledge_caches",
            _ => "rebuild_other",
        });
    }

    /// 记录某类依赖发生变化的版本。
    fn stamp_dependency(&mut self, dependency: &'static str) {
        self.dependency_stamps.insert(dependency, self.revision);
    }

    /// 派生台账报告：知识版本 + 各派生器最后重建版本 + 各依赖最后变化版本。
    ///
    /// 「重复观察不触发重建」因此可审计：版本不动 → 台账不动。
    pub fn derivation_report(&self) -> String {
        let mut output = format!("派生台账：知识版本 {}。\n", self.revision);
        if self.dependency_stamps.is_empty() {
            output.push_str("依赖变化：暂无。\n");
        } else {
            output.push_str("依赖变化（最后版本）：");
            let items: Vec<String> = self
                .dependency_stamps
                .iter()
                .map(|(name, revision)| format!("{}={}", name, revision))
                .collect();
            output.push_str(&items.join("；"));
            output.push('\n');
        }
        if self.derived_stamps.is_empty() {
            output.push_str("派生重建：暂无。\n");
        } else {
            output.push_str("派生重建（最后版本）：");
            let items: Vec<String> = self
                .derived_stamps
                .iter()
                .map(|(name, revision)| format!("{}={}", name, revision))
                .collect();
            output.push_str(&items.join("；"));
            output.push('\n');
        }
        output
    }

    /// 重建谓词清单与命题表面积索引（二者仅依赖对齐器/生成器/预测规则的状态）
    fn rebuild_knowledge_caches(&mut self) {
        self.known_predicates_cache = self.collect_known_predicates();
        self.known_predicates_normalized_cache = self
            .known_predicates_cache
            .iter()
            .map(|predicate| (predicate.clone(), self.aligner.normalize(predicate)))
            .collect();
        self.proposition_surfaces = semantic_entailment::build_predicate_surfaces(
            &self.aligner,
            self.generator.predicate_surfaces(),
        );
        // 属性查询索引按规范槽位重建（同义词/谓词别名变化后仍可命中）
        let mut attribute_index: HashMap<(String, String), Vec<String>> = HashMap::new();
        for (subject, key, value) in &self.attributes {
            let slot = self.aligner.canonical_form(key);
            attribute_index
                .entry((subject.clone(), slot))
                .or_default()
                .push(value.clone());
        }
        for values in attribute_index.values_mut() {
            values.sort();
            values.dedup();
        }
        self.attribute_index = attribute_index;
        // 词表变化后实体链接/实体提及的缓存结果可能过时，同步失效记忆缓存
        self.link_memo.lock().clear();
        self.entity_mentions_memo.lock().clear();
        self.predicate_canonicals_memo.lock().clear();
        // 律典是谓词画像与当前挖掘规则的纯派生：知识/规则变化后同批重建。
        self.rebuild_laws();
        self.stamp_rebuild("knowledge_caches");
    }

    pub fn context(&self) -> &DialogueContext {
        &self.context
    }

    /// 话语帧记忆（只读）：上一完整帧 + 挂账（未竟之问）
    pub fn frame_memory(&self) -> &FrameMemory {
        &self.frames
    }

    /// 话语帧记忆报告：帧、洞填充来源、挂账与剩余轮数（确定性、可断言）
    pub fn frame_memory_report(&self) -> String {
        self.frames.report()
    }

    /// 帧澄清是否进行中（无源可填的洞已转为等待一次显式绑定）
    pub fn frame_clarification_active(&self) -> bool {
        self.frames.clarification().is_some()
    }

    /// 帧澄清提示（候选编号列表；确定性渲染）
    pub fn frame_clarification_prompt(&self) -> Option<String> {
        self.frames
            .clarification()
            .map(dialogue_frame::ClarificationRequest::prompt)
    }

    /// 帧澄清入口：无上下文的「体重是多少」这类带洞问句在常规通道必然失败时，
    /// 不落进「无法理解」，而是挂成一次显式绑定请求（变量求解的第 4 种结局）。
    ///
    /// 开启条件（全部满足才接管）：
    ///   1) 配置启用澄清，且没有进行中的溯因/帧澄清（至多一条）；
    ///   2) 问句是无主语属性问句（`属性+是多少/是什么`），但不含「呢」
    ///      （「体重呢」属于常规省略/对话管理器通道）；
    ///   3) 常规通道必然失败：上一完整帧为空、挂账为空（否则应走合一补全）；
    ///   4) 候选非空：焦点实体优先，其余是知识库中确实拥有该属性的实体。
    pub fn start_frame_clarification(&mut self, input: &str) -> Option<String> {
        if !self.config.enable_clarification || self.clarification_active() {
            return None;
        }
        if self.frames.clarification().is_some()
            || self.frames.last().is_some()
            || !self.frames.pending().is_empty()
        {
            return None;
        }
        let trimmed = input
            .trim()
            .trim_end_matches(['？', '?', '。', '！', '!', ' '])
            .trim();
        if trimmed.ends_with('呢') {
            return None;
        }
        let turn = self.frames.turn();
        let fragment =
            dialogue_frame::parse_bare_attribute_question(input, &FrameOntology(self), turn)?;
        let attribute = fragment.attribute.clone()?;
        let candidates = self.frame_clarification_candidates(&attribute);
        if candidates.is_empty() {
            return None;
        }
        let request = self
            .frames
            .begin_clarification(input.trim(), &attribute, candidates);
        Some(request.prompt())
    }

    /// 处理对帧澄清的短答：序号/候选名 → 绑定；越界 → 重问；都不是 → 取消。
    /// 已知实体名（即使不在候选表里）同样接受为一次显式绑定；
    /// 其余回答返回 None（放行给常规管线，零状态变化）。
    pub fn answer_frame_clarification(
        &mut self,
        text: &str,
    ) -> Option<dialogue_frame::ClarificationOutcome> {
        let request = self.frames.clarification()?.clone();
        match request.parse_reply(text) {
            dialogue_frame::ClarificationReply::Candidate(entity) => {
                self.frames.commit_clarification(&entity)
            }
            dialogue_frame::ClarificationReply::OutOfRange => {
                Some(dialogue_frame::ClarificationOutcome::Reprompt {
                    prompt: request.prompt(),
                })
            }
            dialogue_frame::ClarificationReply::Cancel => self.frames.cancel_clarification(),
            dialogue_frame::ClarificationReply::Unmatched => {
                let entity = FrameOntology(self).canonical_entity(text)?;
                self.frames.commit_clarification(&entity)
            }
        }
    }

    /// 澄清候选：焦点实体 ∩ 有该属性的实体 → 有该属性的实体（名称序）→ 其余焦点实体。
    /// 上限 5 条；顺序确定（可 golden），焦点只作为先验来源，不引入新猜测。
    fn frame_clarification_candidates(&self, attribute: &str) -> Vec<String> {
        const MAX_CANDIDATES: usize = 5;
        let mut with_attribute: Vec<String> = self
            .attribute_index
            .iter()
            .filter(|((_, slot), values)| slot == attribute && !values.is_empty())
            .map(|((entity, _), _)| entity.clone())
            .collect();
        with_attribute.sort();
        with_attribute.dedup();
        let available: HashSet<&str> = with_attribute.iter().map(String::as_str).collect();
        let mut ranked: Vec<String> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for focus in self.context.focus_entities() {
            if available.contains(focus.as_str()) && seen.insert(focus.clone()) {
                ranked.push(focus.clone());
            }
        }
        for entity in &with_attribute {
            if ranked.len() >= MAX_CANDIDATES {
                break;
            }
            if seen.insert(entity.clone()) {
                ranked.push(entity.clone());
            }
        }
        for focus in self.context.focus_entities() {
            if ranked.len() >= MAX_CANDIDATES {
                break;
            }
            if self.is_known_entity(focus) && seen.insert(focus.clone()) {
                ranked.push(focus.clone());
            }
        }
        ranked.truncate(MAX_CANDIDATES);
        ranked
    }

    pub fn embedding(&self) -> &SymbolicEmbedding {
        &self.embedding
    }

    pub fn walks(&self) -> &GraphWalkEmbedding {
        &self.walks
    }

    /// 全息符号记忆（VSA/HDC 软召回通道 + 三元组硬核查通道）
    pub fn holographic(&self) -> &HolographicMemory {
        &self.holo
    }

    /// 把一段文本里的已知实体压入全息时序记忆（多轮对话的"注意力轨迹"）
    pub fn note_holo_context(&mut self, text: &str) {
        if !self.config.enable_holographic_memory {
            return;
        }
        for entity in self.entities_mentioned(text) {
            self.holo.push_focus(&entity);
        }
    }

    pub fn soft_logic(&self) -> &MarkovLogicEngine {
        &self.soft
    }

    pub fn problog(&self) -> &ProblogEngine {
        &self.problog
    }

    /// 已学的中文概率知识原句（教学记忆）
    pub fn lessons(&self) -> &[String] {
        &self.lessons
    }

    pub fn linker(&self) -> &EntityLinker {
        &self.linker
    }

    pub fn generator(&self) -> &SymbolicGenerator {
        &self.generator
    }

    /// 让生成器复用配置编译出的双向语义规则（抽取与生成同源）。
    pub fn attach_semantic_grammar(&mut self, grammar: crate::semantic_grammar::RulePack) {
        self.generator.attach_semantic_grammar(grammar);
    }

    /// 挂接导入可逆帧：生成侧复用导入句措辞，回答侧作为最后一道保守回退。
    pub fn attach_import_frames(&mut self, frames: crate::import_frames::ImportFrameLedger) {
        self.generator.attach_import_frames(frames);
    }

    /// 挂接谓词中心语义角色帧。
    pub fn attach_frame_book(&mut self, book: crate::frame_semantics::FrameBook) {
        self.generator.attach_frame_book(book);
    }

    /// 当前语义角色帧台账（未挂接时为 `None`）。
    pub fn frame_book(&self) -> Option<&crate::frame_semantics::FrameBook> {
        self.generator.frame_book()
    }

    /// 摘除导入可逆帧（撤销全部导入或热加载关闭时使用）。
    pub fn detach_import_frames(&mut self) {
        self.generator.detach_import_frames();
    }

    /// 当前导入帧台账（未挂接时为 `None`）。
    pub fn import_frames(&self) -> Option<&crate::import_frames::ImportFrameLedger> {
        self.generator.import_frames()
    }

    /// 类型化配方查询：按方向（实体→答案 / 答案→实体）取关系值，
    /// 用当前单一本体的物化闭包补齐逆关系与子类推断，并按答案类型做
    /// **开放世界**过滤——未知类型放行，只有显式互斥的已知冲突才拒答。
    ///
    /// 返回的 `evidence` 逐条复述事实边/公理推导理由，供插件模板 `{evidence}`
    /// 接地生成；没有取值时返回 `None`，调用方应让位核心管线。
    pub fn recipe_lookup(
        &mut self,
        entity: &str,
        attribute: &str,
        direction: RecipeDirection,
        answer_types: &[String],
    ) -> Option<RecipeLookup> {
        let entity = entity.trim();
        let attribute = attribute.trim();
        if entity.is_empty() || attribute.is_empty() {
            return None;
        }
        let mut predicates: BTreeSet<String> = BTreeSet::new();
        predicates.insert(attribute.to_string());
        let canonical = self.aligner.canonical_form(attribute);
        if !canonical.trim().is_empty() {
            predicates.insert(canonical.trim().to_string());
        }

        let closure = self.discourse.narrator_mut().closure().clone();
        // 主语侧类型约束：正向问句的实体应落在谓词定义域内，逆向问句的
        // 实体应落在值域内。由「关系定义域/值域反推」得到的类型是循环
        // 证据（类型恰由被查关系自身反推），不作数；开放世界下有效类型
        // 未知即放行，已知类型与全部允许类互斥才拒答——减少「关系同名
        // 但类型不符」的误答。
        let mut expected_anchor_types: BTreeSet<String> = BTreeSet::new();
        for predicate in &predicates {
            if direction.is_inverse() {
                expected_anchor_types.extend(closure.range_classes(predicate));
            } else {
                expected_anchor_types.extend(closure.domain_classes(predicate));
            }
        }
        if !expected_anchor_types.is_empty() {
            let declared_types: Vec<String> = closure
                .types_of(entity)
                .into_iter()
                .filter(|class| {
                    !closure
                        .why_instance(entity, class)
                        .map(|reason| reason.contains("的定义域是") || reason.contains("的值域是"))
                        .unwrap_or(false)
                })
                .collect();
            if !declared_types.is_empty() {
                let compatible = declared_types.iter().any(|actual| {
                    expected_anchor_types
                        .iter()
                        .any(|expected| actual == expected || closure.is_subclass(actual, expected))
                });
                if !compatible
                    && declared_types.iter().any(|actual| {
                        expected_anchor_types
                            .iter()
                            .any(|expected| closure.are_disjoint(actual, expected))
                    })
                {
                    return None;
                }
            }
        }

        let mut raw_values: BTreeSet<String> = BTreeSet::new();
        for predicate in &predicates {
            if direction.is_inverse() {
                raw_values.extend(self.generator.relation_values_in(entity, predicate));
            } else {
                raw_values.extend(self.generator.attribute_values(entity, predicate));
                raw_values.extend(self.generator.relation_values(entity, predicate));
            }
        }

        for (predicate, object) in closure.properties_of(entity) {
            if predicates.contains(&predicate) {
                raw_values.insert(object);
            }
        }
        if direction.is_inverse() {
            for (subject, predicate, object) in closure.relations_snapshot() {
                if object == entity && predicates.contains(&predicate) {
                    raw_values.insert(subject);
                }
            }
        }

        if raw_values.is_empty() {
            return None;
        }
        if !answer_types.is_empty() {
            raw_values.retain(|value| {
                let known_types = closure.types_of(value);
                if known_types.is_empty() {
                    return true;
                }
                let compatible = answer_types.iter().any(|expected| {
                    closure.is_instance(value, expected)
                        || known_types
                            .iter()
                            .any(|actual| closure.is_subclass(actual, expected))
                });
                if compatible {
                    return true;
                }
                // 开放世界：已知类型但无互斥证据时仍放行，只有显式冲突才拒答。
                !known_types.iter().any(|actual| {
                    answer_types
                        .iter()
                        .any(|expected| closure.are_disjoint(actual, expected))
                })
            });
            if raw_values.is_empty() {
                return None;
            }
        }

        let mut evidence: Vec<String> = Vec::new();
        for (subject, predicate, object) in closure.relations_snapshot() {
            if !predicates.contains(&predicate) {
                continue;
            }
            let (anchor, answer) = if direction.is_inverse() {
                (object.as_str(), subject.clone())
            } else {
                (subject.as_str(), object.clone())
            };
            if anchor != entity || !raw_values.contains(&answer) {
                continue;
            }
            let why = closure
                .why_relation(&subject, &predicate, &object)
                .unwrap_or("直接声明");
            evidence.push(format!("{} —{}→ {}（{}）", subject, predicate, object, why));
        }
        if evidence.is_empty() {
            evidence.push(format!("图谱属性记录：{}的{}", entity, attribute));
        }
        // 事实边优先保留，类型证明只在剩余额度内补充；不能因为答案多值
        // 就把「哪条事实支持了这个值」挤出依据列表。
        evidence.sort();
        evidence.dedup();
        evidence.truncate(6);
        // 类型约束用的不是问答模板里的猜测：把每个答案的已知类型及其推导理由
        // 一并复述，用户能看到「为什么这个值符合 answer_types」。
        if !answer_types.is_empty() {
            let mut type_evidence = Vec::new();
            for value in &raw_values {
                let known_types = closure.types_of(value);
                for actual in &known_types {
                    let compatible = answer_types.iter().any(|expected| {
                        actual == expected || closure.is_subclass(actual, expected)
                    });
                    if !compatible {
                        continue;
                    }
                    let most_specific = !known_types
                        .iter()
                        .any(|other| other != actual && closure.is_subclass(other, actual));
                    if !most_specific {
                        continue;
                    }
                    let why = closure.why_instance(value, actual).unwrap_or("类型继承");
                    type_evidence.push(format!("{} 属于 {}（{}）", value, actual, why));
                }
            }
            type_evidence.sort();
            type_evidence.dedup();
            for line in type_evidence {
                if evidence.len() >= 8 {
                    break;
                }
                if !evidence.contains(&line) {
                    evidence.push(line);
                }
            }
        }

        let values: Vec<String> = raw_values
            .into_iter()
            .map(|value| {
                crate::identity::local_name(crate::identity::DEFAULT_BASE_NS, &value)
                    .unwrap_or(value)
            })
            .collect();
        Some(RecipeLookup { values, evidence })
    }

    /// 执行一次插件配方：回指焦点解析 → 类型化查值（含逆关系/子类/开放世界
    /// 类型约束）→ 核心管线兜底 → 模板接地。CLI 与 HTTP 都只调用这里，
    /// 避免两条入口在方向、依据或回指处理上各自演化。
    pub fn answer_plugin_recipe(
        &mut self,
        outcome: &PluginOutcome,
        runtime: &mut PluginRuntime,
        focus_entity: Option<&str>,
    ) -> Option<PluginRecipeReply> {
        let recipe = outcome.trace.recipe.clone()?;
        let raw_entity = recipe.slots.get("entity").cloned();
        let entity = raw_entity.as_ref().map(|raw| {
            if runtime.is_context_reference(raw) {
                focus_entity
                    .map(str::to_string)
                    .unwrap_or_else(|| raw.clone())
            } else {
                raw.clone()
            }
        });
        // 回指替换后同步改写问句，核心管线兜底时不再拿「他」去检索。
        let question = match (raw_entity.as_deref(), entity.as_deref()) {
            (Some(raw), Some(resolved)) if raw != resolved => {
                recipe.question.replace(raw, resolved)
            }
            _ => recipe.question.clone(),
        };
        let mut answer = None;
        let mut answer_values = Vec::new();
        let mut evidence = Vec::new();
        if let (Some(entity), Some(attribute)) = (entity.as_deref(), recipe.attribute.as_deref()) {
            if let Some(lookup) =
                self.recipe_lookup(entity, attribute, recipe.direction, &recipe.answer_types)
            {
                if !lookup.values.is_empty() {
                    answer_values = lookup.values.clone();
                    answer = Some(lookup.values.join("、"));
                    evidence = lookup.evidence;
                }
            }
        }
        if answer.is_none() {
            answer = self.answer(&question);
        }
        let answer = answer?;
        // 正向问句的答案通常是下一轮的话题（阿凡达 → 卡梅隆）；逆向问句
        // 的主语本来就是话题（卡梅隆 → 阿凡达），不能被答案覆盖。
        let focus = if recipe.direction.is_inverse() {
            entity.clone()
        } else {
            answer_values.first().cloned().or_else(|| entity.clone())
        };
        let composed = runtime.compose_answer_with_evidence(outcome, &answer, &evidence);
        let plan = vec![format!("plugin:{}:{}", recipe.plugin, recipe.contribution)];
        let provenance = vec![format!(
            "插件 {} / 配方 {} / 问句 {}",
            recipe.plugin, recipe.contribution, question
        )];
        Some(PluginRecipeReply {
            text: composed.text,
            plan,
            provenance,
            entity,
            attribute: recipe.attribute,
            question,
            kind: "attribute",
            focus,
        })
    }

    /// 在当前知识快照上执行带约束/预算的证据检索（含可审计回执）。
    ///
    /// 索引随 `observe_facts` 增量观察，首次查询时惰性重建倒排表；
    /// 结果与独立调用 `DenseRetriever::observe_facts + search` 一致。
    pub fn evidence_search(
        &self,
        query: &str,
        k: usize,
        constraints: &SearchConstraints,
        budget: SearchBudget,
    ) -> EvidenceSearch {
        let mut search = self.retrieval.lock().search(query, k, constraints, budget);
        // 证据索引同样尊重槽级压制：功能槽已有会话赢家时，图谱旧值不再
        // 作为候选证据；旧值仍在图谱与账本里，可审计、可恢复。
        search.hits.retain(|hit| {
            !self.is_suppressed_slot_value(&hit.subject, &hit.predicate, &hit.object)
        });
        self.semantic.record_search(query, &search);
        search
    }

    /// 某个 `(主语, 谓词, 取值)` 是否被当前会话槽赢家压制。
    ///
    /// 只有「该槽确有会话赢家且新值不是它」时才算压制；槽为空时图谱值
    /// 正常可见（撤回改口后自然回落到图谱）。
    pub fn is_suppressed_slot_value(&self, subject: &str, predicate: &str, value: &str) -> bool {
        let (subject, predicate) = self.slot_identity(subject, predicate);
        let accepted = self.slots.accepted(&subject, &predicate);
        if accepted.is_empty() || accepted.iter().any(|candidate| candidate == value) {
            return false;
        }
        // 功能槽只允许一个视图赢家：会话已有赢家时，图谱后来新增/变化的同槽
        // 取值也不得绕过压制进入回答；非功能槽只压制账本明确记录的落选值。
        self.slots.is_functional(&predicate)
            || self
                .slots
                .suppressed(&subject, &predicate)
                .iter()
                .any(|candidate| candidate == value)
    }

    /// 证据检索的文本视图：排序证据 + 检索回执。
    pub fn evidence_report(
        &self,
        query: &str,
        k: usize,
        constraints: &SearchConstraints,
        budget: SearchBudget,
    ) -> String {
        let search = self.evidence_search(query, k, constraints, budget);
        format!(
            "{}\n\n{}",
            self.retrieval.lock().format_hits(query, &search.hits),
            search.receipt.render_text()
        )
    }

    /// 运行时登记谓词别名（来自谓词画像准入）：查询解析与生成词面同步归一。
    pub fn register_predicate_alias(&mut self, alias: &str, canonical: &str) -> bool {
        if alias.trim().is_empty() || canonical.trim().is_empty() || alias == canonical {
            return false;
        }
        self.aligner.register_term(canonical, TermKind::Relation);
        let synonym_added = self.aligner.register_synonym(alias, canonical);
        let surface_added = self.generator.register_predicate_alias(alias, canonical);
        if synonym_added || surface_added {
            self.rebuild_knowledge_caches();
        }
        synonym_added || surface_added
    }

    /// 登记实体别名（来自实体身份注册表已确认的持久别名）：链接器按别名识别，
    /// 与 KB 里的规范名同权；歧义词面如实保留多候选，不静默择一。
    ///
    /// 已确认别名同时进对齐器：问答路径按规范名归一实体（「确认身份」签过字
    /// 的说法要能直接答出来），链接器仍负责候选枚举与接地引用，两处同源。
    pub fn register_entity_alias(&mut self, alias: &str, canonical: &str) -> bool {
        let (alias, canonical) = (alias.trim(), canonical.trim());
        if alias.is_empty() || canonical.is_empty() || alias == canonical {
            return false;
        }
        let etype = self.linker.entity_type(canonical).map(str::to_string);
        self.linker.register_alias(alias, canonical);
        self.aligner.register_term(canonical, TermKind::Entity);
        self.aligner.register_term(alias, TermKind::Entity);
        self.aligner.register_synonym(alias, canonical);
        self.mention_lexicon
            .confirm(alias, canonical, etype.as_deref(), "confirmed_alias");
        self.link_memo.lock().clear();
        true
    }

    /// 登记逆向谓词词面：问句用该词面发问时，命中规范槽位三元组的反方向
    ///（「徒弟」问句 ↔「师傅」事实）。只影响查询换向，不改词面/叙事表。
    /// 返回是否新增/改变；重复登记返回 `false`。
    pub fn register_predicate_query_surface(&mut self, surface: &str, slot: &str) -> bool {
        let (surface, slot) = (surface.trim(), slot.trim());
        if surface.is_empty() || slot.is_empty() || surface == slot {
            return false;
        }
        self.aligner.register_term(surface, TermKind::Relation);
        self.aligner.register_term(slot, TermKind::Relation);
        let changed = self.inverse_query_map.get(surface) != Some(&(slot.to_string(), true));
        self.inverse_query_map
            .insert(surface.to_string(), (slot.to_string(), true));
        if changed {
            self.rebuild_knowledge_caches();
        }
        changed
    }

    /// 把已启用插件的本体贡献单调并入同一个推理器：谓词别名、互逆词面与
    /// 结构性公理。返回本次真正新增的条数；停用插件的撤回不在本会话发生。
    pub fn apply_plugin_runtime(&mut self, runtime: &PluginRuntime) -> PluginOntologySync {
        let mut sync = PluginOntologySync::default();
        for (alias, canonical) in runtime.ontology_aliases() {
            if self.register_predicate_alias(&alias, &canonical) {
                sync.aliases += 1;
            }
        }
        for (surface, slot) in runtime.ontology_inverse_surfaces() {
            if self.register_predicate_query_surface(&surface, &slot) {
                sync.inverse_surfaces += 1;
            }
        }
        sync.lessons = self.configure_ontology_program(&runtime.ontology_program());
        sync
    }

    /// 谓词对齐批量注入（交互与服务共用同一入口）：三元组
    /// `(词面, 规范槽位, 是否换向)`；正向走别名归一，换向只登记查询转置。
    /// 返回本次真正生效的条数（重复/退化条目不计）。
    pub fn align_predicate_pairs(&mut self, pairs: &[(String, String, bool)]) -> usize {
        let mut applied = 0usize;
        for (surface, slot, inverted) in pairs {
            if surface == slot {
                continue;
            }
            let registered = if *inverted {
                self.register_predicate_query_surface(surface, slot)
            } else {
                self.register_predicate_alias(surface, slot)
            };
            if registered {
                applied += 1;
            }
        }
        applied
    }

    /// 实体身份别名批量注入（已确认的持久别名与规范名同权），返回生效条数。
    pub fn align_entity_aliases(&mut self, aliases: &[(String, String)]) -> usize {
        let mut applied = 0usize;
        for (surface, canonical) in aliases {
            if self.register_entity_alias(surface, canonical) {
                applied += 1;
            }
        }
        applied
    }

    /// 配置律典：把谓词画像中的显式声明编译为 Active 律，并吸收当前挖掘规则。
    /// 可反复调用；知识变化后由 `rebuild_knowledge_caches` 自动重放同一来源。
    pub fn configure_laws(&mut self, registry: &PredicateRegistry) {
        self.law_registry = Some(registry.clone());
        self.rebuild_laws();
    }

    /// 载入持久化账本中的显式采纳/退役决定，并立即重放律典。
    /// 返回载入的显式决定条数（Active/Retired）。
    pub fn load_persisted_laws(&mut self, persisted: LawBook) -> usize {
        let loaded = persisted
            .laws()
            .filter(|law| matches!(law.status, LawStatus::Active | LawStatus::Retired))
            .count();
        self.persisted_laws = Some(persisted);
        self.rebuild_laws();
        loaded
    }

    /// 律典快照（只读派生；调用方审计时在其副本上取反例）。
    pub fn law_book_snapshot(&self) -> LawBook {
        self.laws.clone()
    }

    /// 当前律典 + 对现有事实的确定性审计（交互命令与评测共用）。
    pub fn law_report(&self) -> String {
        let mut book = self.laws.clone();
        let audit = book.audit(self.predictor.triples());
        format!("{}{}", book.render_text(), audit.render_text())
    }

    /// 当前会话的信念快照：复用归结引擎与已声明的谓词约束，不复制另一套事实。
    pub fn belief_state(&self) -> BeliefState {
        BeliefState::with_constraints(self.resolver.clone(), self.session_revision_constraints())
    }

    /// 只读一致性报告：功能谓词多值、定义域/值域冲突与规则形状问题。
    pub fn belief_check_report(&self) -> String {
        self.belief_state().check().render_zh()
    }

    /// 规则提交预演：把中文规则解析成 Horn 子句后，先检查不安全变量、重复规则、
    /// 功能谓词冲突与派生影响；只计算，不把规则放进会话规则库。
    pub fn belief_rule_preview(&self, sentence_raw: &str) -> String {
        let sentence = sentence_raw.trim();
        if sentence.is_empty() {
            return "请给出中文规则，例如：规则预演：如果X的师傅是Y，那么X的老师是Y".to_string();
        }
        let outcome = chinese_reasoning_nlu::parse_lessons(sentence);
        let clauses: Vec<Clause> = outcome
            .lessons
            .iter()
            .filter_map(chinese_reasoning_nlu::lesson_to_clause)
            .collect();
        if clauses.is_empty() {
            let mut output = String::from("没能把这句话解析成规则：\n");
            for error in &outcome.errors {
                output.push_str(&format!("  · {}\n", error));
            }
            return output.trim_end().to_string();
        }
        let state = self.belief_state();
        let mut changeset = ChangeSet::new(format!("规则预演：{}", sentence));
        for clause in clauses {
            changeset.push(crate::belief_revision::ChangeOp::AddRule(clause));
        }
        state.plan(changeset).render_zh()
    }
}

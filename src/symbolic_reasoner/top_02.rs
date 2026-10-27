/// 深拷贝：所有子引擎必须实现 `Clone`（锁/原子量以值重建，不共享可变状态）。
/// 内存记忆缓存（提及/链接/谓词）按内容复制，语义不变。
impl Clone for SymbolicReasoner {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            aligner: self.aligner.clone(),
            predictor: self.predictor.clone(),
            // 草稿是纯派生缓存：分叉时留空，由新会话按需重建。
            prediction_drafts: parking_lot::Mutex::new(PredictionDrafts::new(
                PREDICTION_DRAFT_CAPACITY,
            )),
            completion: self.completion.clone(),
            predictive: self.predictive.clone(),
            laws: self.laws.clone(),
            law_registry: self.law_registry.clone(),
            persisted_laws: self.persisted_laws.clone(),
            context: self.context.clone(),
            epistemic_turns: self.epistemic_turns.clone(),
            embedding: self.embedding.clone(),
            schema_view: self.schema_view.clone(),
            walks: self.walks.clone(),
            soft: self.soft.clone(),
            problog: self.problog.clone(),
            lessons: self.lessons.clone(),
            lessons_parsed: self.lessons_parsed.clone(),
            planner: self.planner.clone(),
            linker: self.linker.clone(),
            generator: self.generator.clone(),
            rewrite_book: self.rewrite_book.clone(),
            retrieval: parking_lot::Mutex::new(self.retrieval.lock().clone()),
            contextual_last: parking_lot::Mutex::new(self.contextual_last.lock().clone()),
            formal_clarification: parking_lot::Mutex::new(self.formal_clarification.lock().clone()),
            contextual_memory: self.contextual_memory.clone(),
            attributes: self.attributes.clone(),
            attribute_index: self.attribute_index.clone(),
            associations: self.associations.clone(),
            resolver: self.resolver.clone(),
            abduction: self.abduction.clone(),
            frames: self.frames.clone(),
            pending_questions: self.pending_questions.clone(),
            graph_session: parking_lot::Mutex::new(self.graph_session.lock().clone()),
            session_facts: self.session_facts.clone(),
            slots: self.slots.clone(),
            assumption_world: self.assumption_world.clone(),
            session_negative_facts: self.session_negative_facts.clone(),
            session_clauses: self.session_clauses.clone(),
            session_receipts: self.session_receipts.clone(),
            clarification: self.clarification.clone(),
            last_abduction_goal: self.last_abduction_goal.clone(),
            last_abduction_vars: self.last_abduction_vars.clone(),
            known_predicates_cache: self.known_predicates_cache.clone(),
            known_predicates_normalized_cache: self.known_predicates_normalized_cache.clone(),
            proposition_surfaces: self.proposition_surfaces.clone(),
            // 记忆缓存是纯派生数据：分叉时留空，由新会话按需重建
            // （空缓存与「重建后的缓存」在固定纪元下读数一致）
            link_memo: parking_lot::Mutex::new(RevisionLru::new(MEMO_CAPACITY)),
            entity_mentions_memo: parking_lot::Mutex::new(RevisionLru::new(MEMO_CAPACITY)),
            predicate_canonicals_memo: parking_lot::Mutex::new(RevisionLru::new(MEMO_CAPACITY)),
            inverse_query_map: self.inverse_query_map.clone(),
            discourse: self.discourse.clone(),
            holo: self.holo.clone(),
            revision: self.revision,
            derived_stamps: self.derived_stamps.clone(),
            dependency_stamps: self.dependency_stamps.clone(),
            perf: Arc::clone(&self.perf),
            nlp_config: self.nlp_config.clone(),
            // 轨迹不是记忆：分叉会话从空台账开始，避免把上一会话的失败带入
            read_traces: ReadTraceLedger::new(),
            semantic: self.semantic.clone(),
            lexical_memory: self.lexical_memory.clone(),
            mention_lexicon: self.mention_lexicon.clone(),
            engine_fault: self.engine_fault.clone(),
        }
    }
}

/// 语境指称消歧（CRD）的只读义项来源：复用生成器（类型/谓词/取值）与
/// 链接器（邻居锚点/度数），不新建词表、不重写链接打分。
const SENSE_CARD_ANCHOR_LIMIT: usize = 8;

impl SenseSource for SymbolicReasoner {
    fn sense_card(&self, entity: &str) -> Option<SenseCard> {
        let known = self.generator.is_known_entity(entity) || self.linker.is_known_entity(entity);
        if !known {
            return None;
        }
        let etype = self
            .linker
            .entity_type(entity)
            .map(str::to_string)
            .or_else(|| self.generator.entity_type_of(entity).map(str::to_string));
        let mut out_predicates = self.generator.relation_predicates(entity);
        out_predicates.extend(self.generator.attribute_keys(entity));
        out_predicates.sort();
        out_predicates.dedup();
        let in_predicates = self.generator.relation_predicates_in(entity);
        let mut anchors = self.linker.neighbors_of(entity);
        anchors.truncate(SENSE_CARD_ANCHOR_LIMIT);
        Some(SenseCard {
            entity: entity.to_string(),
            etype,
            out_predicates,
            in_predicates,
            anchors,
            degree: self.linker.degree_of(entity),
        })
    }

    fn predicate_values(&self, entity: &str, predicate: &str) -> Vec<String> {
        self.generator.relation_values(entity, predicate)
    }

    fn predicate_values_in(&self, entity: &str, predicate: &str) -> Vec<String> {
        self.generator.relation_values_in(entity, predicate)
    }
}

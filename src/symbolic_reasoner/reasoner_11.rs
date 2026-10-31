impl SymbolicReasoner {
    /// 把 [`Self::dialogue_snapshot`] 的内容恢复到当前推理器上。
    ///
    /// 恢复只重建会话级派生数据（会话事实/教学/会话规则/话语帧/本体胶囊），
    /// 全局知识图谱不受影响；同一快照恢复两次结果一致（幂等去重）。
    /// 本体胶囊的版本门或公理摘要不通过时整体拒绝并返回 `Err`（零半应用）。
    pub fn restore_dialogue_snapshot(&mut self, snapshot: &DialogueSnapshot) -> Result<(), String> {
        self.apply_dialogue_snapshot(snapshot)
    }

    pub fn save_dialogue_state(&self, path: &str) -> std::io::Result<()> {
        self.save_dialogue_state_with_contrast(path, None)
    }

    /// 保存对话状态，并额外写入神经符号对比仲裁记忆（校准统计/待定推测/否证集）。
    ///
    /// 落盘为**规范字节**（对象键递归排序）：同一会话状态永远同一字节串，
    /// 跨运行 diff / 审计不受 `HashMap` 迭代序影响（L2 字节不动点）。
    pub fn save_dialogue_state_with_contrast(
        &self,
        path: &str,
        contrast: Option<&ContrastSnapshot>,
    ) -> std::io::Result<()> {
        if let Some(parent) = std::path::Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let mut snapshot = self.dialogue_snapshot();
        snapshot.contrast = contrast.cloned();
        let bytes = canonical::canonical_bytes(&snapshot)
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::Other, error.to_string()))?;
        std::fs::write(path, bytes)
    }

    /// 读取并解码对话状态（不做任何应用），返回快照与解码回执。
    ///
    /// 读宽容：pretty 排布的文件仍可解码（回执 `canonical == false`）。
    pub fn read_dialogue_state(
        path: &str,
    ) -> Result<(DialogueSnapshot, DecodeReceipt), DecodeError> {
        let bytes = std::fs::read(path).map_err(|error| DecodeError::Io(error.to_string()))?;
        canonical::decode_json::<DialogueSnapshot>(
            &bytes,
            "dialogue_state",
            1,
            canonical::DEFAULT_MAX_DECODE_BYTES,
        )
    }

    pub fn load_dialogue_state(&mut self, path: &str) -> bool {
        self.load_dialogue_state_with_contrast(path).0
    }

    /// 加载对话状态，同时返回其中的对比仲裁记忆快照（无则 None）
    pub fn load_dialogue_state_with_contrast(
        &mut self,
        path: &str,
    ) -> (bool, Option<ContrastSnapshot>) {
        let Ok((snapshot, _)) = Self::read_dialogue_state(path) else {
            return (false, None);
        };
        let contrast = snapshot.contrast.clone();
        // 校验失败（版本门 / 公理摘要被改写）时零半应用，按「未加载」上报。
        match self.apply_dialogue_snapshot(&snapshot) {
            Ok(()) => (true, contrast),
            Err(_) => (false, None),
        }
    }

    fn apply_dialogue_snapshot(&mut self, snapshot: &DialogueSnapshot) -> Result<(), String> {
        // 快照会整体替换会话世界；旧图谱追问的结果页与见证下标不再可信。
        self.invalidate_graph_session();
        self.search_session.lock().clear();
        // 先丢弃旧假设世界：快照要么恢复一组新的，要么保持无世界。
        self.discard_assumption_world();
        // 本体胶囊最先恢复：版本门 / 公理摘要不通过就整体拒绝，
        // 绝不留下「会话态已换、本体还是旧的」这种半应用状态。
        if let Some(capsule) = &snapshot.ontology {
            self.discourse.restore_ontology_capsule(capsule)?;
            // 胶囊公理与检索类型层级同源：恢复后同步一次（幂等）
            self.sync_retrieval_type_hierarchy();
        }
        self.context.restore(snapshot);
        // 复核台账是会话态：旧快照缺省为空，避免把上一会话的待办带进来。
        self.epistemic_turns = snapshot.epistemic_turns.clone().unwrap_or_default();
        // 跨轮信念账本同样只在快照声明时恢复；旧快照保持空账本。
        self.beliefs = snapshot.beliefs.clone().unwrap_or_default();
        // 镜像对齐：旧快照没有该字段时清空，避免把上一会话的词形规则串进来。
        let alignment = snapshot.alignment.clone().unwrap_or_default();
        *self.linker.alignment_mut() = alignment;
        self.link_memo.lock().clear();
        self.entity_mentions_memo.lock().clear();
        // 补全回合：先恢复台账，后续会话事实/规则重建时即可带上已晋升的全局规则。
        // 旧快照没有该字段时清空，避免把上一个会话的回合串进新会话。
        self.completion = snapshot.completion.clone().unwrap_or_default();
        self.completion.clamp_limits();
        self.predictive = snapshot.predictive.clone().unwrap_or_default();
        let regression = self.completion.enforce_regression();
        if !regression.passed() {
            self.completion.note(format!(
                "快照恢复时运行期回归未通过：{}；相关规则已隔离",
                regression.summary()
            ));
        }
        // 话语帧记忆：恢复上一帧与挂账（重启后接着补全未竟之问）
        if let Some(frames) = &snapshot.frames {
            self.frames = frames.clone();
        }
        // 悬问账本：旧快照没有该字段时按空账本恢复；损坏的超额条目在 replace 内裁剪。
        let mut pending_questions = PendingQuestionLedger::new();
        if let Some(saved) = &snapshot.pending_questions {
            pending_questions.restore_from(saved);
        }
        self.pending_questions = pending_questions;
        self.batch_pending = snapshot.batch_pending.clone();
        // 跨会话临时信念：恢复被采纳的会话假设（仍不写知识图谱）
        let mut restored: Vec<Atom> = Vec::new();
        for entry in &snapshot.session_facts {
            if entry.predicate.is_empty() || entry.terms.is_empty() {
                continue;
            }
            let terms: Vec<Term> = entry
                .terms
                .iter()
                .map(|name| Term::constant(name))
                .collect();
            let atom = Atom::new(&entry.predicate, terms);
            if atom.is_ground() && !restored.contains(&atom) {
                restored.push(atom);
            }
        }
        // 旧会话事实在加载快照时必须被替换，而不是与新快照并集；
        // 仅当快照为空时才走这条清理路径（非空快照下面整体覆盖）。
        if restored.is_empty() && !self.session_facts.is_empty() {
            let stale = std::mem::take(&mut self.session_facts);
            for atom in &stale {
                if !self.graph_supports_atom(atom) {
                    self.unindex_session_atom(atom);
                    if self.config.enable_resolution {
                        self.resolver.remove_fact_atom(atom);
                    }
                }
            }
            if self.config.enable_resolution {
                self.rebuild_resolver();
            }
        }
        if !restored.is_empty() {
            // 与 adopt_atom 保持一致：恢复的会话事实同样注入软逻辑/概率程序，
            // 否则恢复后 soft_inference_report 与 problog_report 缺少这些事实
            for atom in &restored {
                if self.config.enable_soft_logic {
                    self.soft.add_fact(atom.clone());
                }
                if self.config.enable_problog {
                    self.problog.add_deterministic_fact(atom.clone());
                }
            }
            self.abduction.restore_adopted(restored.clone());
            for atom in &restored {
                self.index_session_atom(atom);
            }
            self.session_facts = restored;
            if self.config.enable_resolution {
                self.rebuild_resolver();
            }
        }
        // 槽账本：新快照直接恢复历史；旧快照从会话事实重建当前投影。
        // 重建只保证「账本 = 事实集」逐字一致，不伪造历史。
        self.slots = match &snapshot.slots {
            Some(ledger) => ledger.clone(),
            None => {
                let mut policy = SlotPolicy::with_defaults();
                policy.adopt_functional(self.session_revision_constraints().functional.iter());
                let facts: Vec<(String, String, String)> = self
                    .session_facts
                    .iter()
                    .filter_map(|atom| {
                        let (subject, predicate, object) = Self::session_atom_triple(atom)?;
                        let (subject, predicate) = self.slot_identity(subject, predicate);
                        Some((subject, predicate, object.to_string()))
                    })
                    .collect();
                SlotLedger::rebuild_from_facts(facts, policy)
            }
        };
        // 否定事实：恢复为路径阻断边（不进入正向归结，因此不触发引擎重建）
        let mut restored_negative: Vec<Atom> = Vec::new();
        for entry in &snapshot.session_negative_facts {
            if entry.predicate.is_empty() || entry.terms.len() != 2 {
                continue;
            }
            let terms: Vec<Term> = entry
                .terms
                .iter()
                .map(|name| Term::constant(name))
                .collect();
            let atom = Atom::new(&entry.predicate, terms);
            if atom.is_ground() && !restored_negative.contains(&atom) {
                restored_negative.push(atom);
            }
        }
        if !restored_negative.is_empty() {
            for atom in &restored_negative {
                self.register_atom_vocabulary(atom);
            }
            self.session_negative_facts = restored_negative;
        }
        // 教学记忆：恢复中文概率知识原句并在概率程序中重放（不写知识图谱）
        if !snapshot.lessons.is_empty() {
            self.lessons = snapshot.lessons.clone();
            // 恢复原句时重建解析缓存
            self.lessons_parsed = self
                .lessons
                .iter()
                .flat_map(|sentence| chinese_reasoning_nlu::parse_lessons(sentence).lessons)
                .collect();
            if self.config.enable_problog && self.config.enable_soft_logic {
                self.rebuild_problog();
            }
        }
        // 会话 Horn 规则：跨会话还原后向链规则库（谓词词表随之恢复）
        if !snapshot.session_clauses.is_empty() {
            let mut restored: Vec<Clause> = Vec::new();
            let mut new_predicates: Vec<String> = Vec::new();
            for entry in &snapshot.session_clauses {
                let Some(clause) = restore_clause(entry) else {
                    continue;
                };
                if restored.iter().any(|existing: &Clause| {
                    existing.head == clause.head && existing.body == clause.body
                }) {
                    continue;
                }
                for atom in std::iter::once(&clause.head).chain(clause.body.iter()) {
                    if !self.aligner.is_registered(&atom.predicate)
                        && !new_predicates
                            .iter()
                            .any(|p| p.as_str() == atom.predicate.as_ref())
                    {
                        new_predicates.push(atom.predicate.to_string());
                    }
                }
                restored.push(clause);
            }
            if !restored.is_empty() {
                for predicate in &new_predicates {
                    self.aligner.register_term(predicate, TermKind::Relation);
                }
                self.session_clauses = restored;
                if !new_predicates.is_empty() {
                    self.rebuild_knowledge_caches();
                }
                if self.config.enable_resolution {
                    self.rebuild_resolver();
                }
            }
        }
        // 假设世界：恢复为与图谱/会话事实分离的临时前提，可继续追问或退出。
        let mut restored_assumptions: Vec<Atom> = Vec::new();
        for entry in &snapshot.assumptions {
            if entry.predicate.is_empty() || entry.terms.is_empty() {
                continue;
            }
            let terms: Vec<Term> = entry
                .terms
                .iter()
                .map(|name| Term::constant(name))
                .collect();
            let atom = Atom::new(&entry.predicate, terms);
            if atom.is_ground() && !restored_assumptions.contains(&atom) {
                restored_assumptions.push(atom);
            }
        }
        self.restore_assumption_world(restored_assumptions);
        // 只有补全台账、没有会话事实/规则时，前面的分支不会触发重建；
        // 这里补一次，确保已晋升的全局规则立即进入归结与软逻辑。
        if snapshot.completion.is_some() {
            if self.config.enable_resolution {
                self.rebuild_resolver();
            }
            if self.config.enable_soft_logic {
                self.rebuild_soft_logic();
            }
        }
        // 修订回执：旧快照没有该字段时清空；损坏的单条回执跳过，不影响会话恢复。
        self.session_receipts = snapshot
            .session_receipts
            .iter()
            .filter_map(SessionRevisionReceipt::from_snapshot)
            .collect();
        if self.session_receipts.len() > MAX_SESSION_RECEIPTS {
            let excess = self.session_receipts.len() - MAX_SESSION_RECEIPTS;
            self.session_receipts.drain(0..excess);
        }
        // 溯因接口状态：待定假设、澄清会话与最近目标都随快照恢复；
        // 旧快照没有这些字段时按空状态处理（字节兼容）。
        let pending: Vec<Hypothesis> = snapshot
            .pending_hypotheses
            .iter()
            .filter_map(restore_hypothesis)
            .collect();
        self.abduction.set_pending(pending);
        self.clarification = snapshot
            .clarification
            .as_ref()
            .map(restore_clarification)
            .filter(|session| !session.is_finished());
        self.last_abduction_goal = {
            let goals: Vec<Atom> = snapshot
                .last_abduction_goal
                .iter()
                .filter_map(restore_snapshot_clause_atom)
                .collect();
            (!goals.is_empty()).then_some(goals)
        };
        self.last_abduction_vars = snapshot.last_abduction_vars.clone();
        // 解释空间不进入快照：恢复时按当前知识重算，只保留快照中的承诺与问题顺序。
        self.refresh_clarification_space();
        Ok(())
    }

    pub fn alignment_report(&self) -> String {
        let pairs = self.aligner.align_all();
        let mut output = format!(
            "本体对齐报告：已登记 {} 个术语，发现 {} 组对齐关系。\n",
            self.aligner.term_count(),
            pairs.len()
        );
        for pair in pairs.iter().take(20) {
            output.push_str(&format!(
                "  • {} ⇒ {}（{:?}，置信度 {:.2}，证据：{}）\n",
                pair.source,
                pair.target,
                pair.kind,
                pair.confidence,
                pair.evidence.join("；")
            ));
        }
        if pairs.is_empty() {
            output.push_str("  （未发现跨词表对齐，可补充同义词或更多关系数据）\n");
        }
        let ambiguous = self.aligner.ambiguous_terms(8);
        if ambiguous.is_empty() {
            output.push_str("对齐歧义：未发现候选分数接近的术语。\n");
        } else {
            output.push_str(&format!(
                "对齐歧义：{} 个术语存在候选分数接近的对齐（可用「对齐候选：术语」查看）。\n",
                ambiguous.len()
            ));
            for resolution in ambiguous {
                output.push_str(&format!(
                    "  • {} → {}\n",
                    resolution.source,
                    resolution
                        .targets()
                        .into_iter()
                        .take(3)
                        .collect::<Vec<_>>()
                        .join(" / ")
                ));
            }
        }
        output
    }

    /// 只读访问本体对齐器（诊断 / 外部知识属性准入等只读消费者）。
    pub fn aligner(&self) -> &OntologyAligner {
        &self.aligner
    }

    /// 候选格报告：展示 top-k 对齐候选、证据与（有上下文实体时的）答案等价性。
    pub fn alignment_candidates_report(&self, term_raw: &str) -> String {
        let term = term_raw.trim();
        if term.is_empty() {
            return "请提供要对齐的术语，例如：对齐候选：老婆".to_string();
        }
        let resolution = self.aligner.resolve_candidates(term, 5);
        let mut output = format!(
            "对齐候选「{}」：状态 {}，规范化形「{}」。\n",
            term,
            resolution.state.label_zh(),
            resolution.normalized
        );
        if resolution.is_empty() {
            if let Some(kind) = self.aligner.kind_of(term) {
                output.push_str(&format!("  精确登记为 {:?}，没有其他接近候选。\n", kind));
            } else {
                output.push_str("  （没有达到阈值的候选）\n");
            }
            return output;
        }
        for (index, candidate) in resolution.candidates.iter().enumerate() {
            output.push_str(&format!(
                "  {}. {}（{:?}，置信度 {:.2}，证据：{}）\n",
                index + 1,
                candidate.target(),
                candidate.kind(),
                candidate.confidence(),
                candidate.pair.evidence.join("；")
            ));
        }

        let Some(entity) = self.context.focus() else {
            return output;
        };
        let mut answerable = resolution.clone();
        answerable.retain(|candidate| self.is_answerable_predicate(candidate.target()));
        if answerable.is_empty() {
            return output;
        }
        let decision = self.aligner.probe_answers(answerable, |target| {
            self.predicate_values_for_entity(entity, target)
        });
        output.push_str(&format!(
            "\n答案等价性（上下文实体「{}」）：{}\n",
            entity,
            if decision.equivalent_answers {
                "候选给出相同答案，不需要澄清"
            } else if decision.should_clarify() {
                "候选给出不同答案，值得澄清"
            } else {
                "只有一种解释有答案，按最佳候选作答"
            }
        ));
        for answer in &decision.answers {
            output.push_str(&format!(
                "  • 「{}」→ {}\n",
                answer.target,
                if answer.answers.is_empty() {
                    "（没有记录）".to_string()
                } else {
                    answer.answers.join("、")
                }
            ));
        }
        output
    }

    /// 谓词是否能在当前知识库中产生答案（关系、属性或预测规则）。
    fn is_answerable_predicate(&self, target: &str) -> bool {
        self.canonical_attribute_name(target).is_some() || self.is_known_predicate(target)
    }

    /// 某实体在一个候选谓词下的全部取值（与正常查询同一条 `node_hits` 路径）。
    fn predicate_values_for_entity(&self, entity: &str, target: &str) -> Vec<String> {
        let (slot, inverted) = self.resolve_question_predicate(target);
        if slot.is_empty() {
            return Vec::new();
        }
        let predicate = ChainPredicate {
            surface: target.to_string(),
            slot,
            inverted,
        };
        self.node_hits(entity, &predicate)
            .into_iter()
            .map(|hit| hit.value)
            .collect()
    }

    /// 属性问句的对齐决策：只保留本库可作答的候选，再用答案集合判断是否真歧义。
    /// 精确登记的谓词是用户明确给出的词面，不因模糊候选而追问。
    fn attribute_alignment_decision(
        &self,
        entity: &str,
        surface: &str,
    ) -> Option<AlignmentDecision> {
        if self.aligner.is_registered(surface) && self.aligner.canonical_form(surface) == surface {
            return None;
        }
        let mut resolution = self.aligner.resolve_candidates(surface, 6);
        if resolution.state != AlignmentState::Ambiguous {
            return None;
        }
        resolution.retain(|candidate| self.is_answerable_predicate(candidate.target()));
        if resolution.candidates.len() < 2 {
            return None;
        }
        Some(self.aligner.probe_answers(resolution, |target| {
            self.predicate_values_for_entity(entity, target)
        }))
    }

    fn render_alignment_clarification(
        &self,
        surface: &str,
        decision: &AlignmentDecision,
    ) -> String {
        let mut output = format!(
            "「{}」有不止一种理解，而且在现有知识中会得到不同答案：\n",
            surface
        );
        for answer in decision.answers.iter().take(4) {
            output.push_str(&format!(
                "  • 「{}」→ {}\n",
                answer.target,
                if answer.answers.is_empty() {
                    "（没有记录）".to_string()
                } else {
                    answer.answers.join("、")
                }
            ));
        }
        output.push_str("请确认你指的是哪一种（可以换一种更具体的说法）。");
        output
    }

    pub fn rules_report(&self) -> String {
        let rules = self.predictor.rules();
        let mut output = format!(
            "已从 {} 条关系事实中挖掘出 {} 条符号推理规则（支持度阈值 {}，置信度阈值 {:.2}）：\n",
            self.predictor.triple_count(),
            rules.len(),
            self.config.link_prediction_min_support,
            self.config.link_prediction_min_confidence
        );
        for rule in rules.iter().take(30) {
            output.push_str(&format!(
                "  • [{}] {}（支持 {} 例，置信度 {:.2}，头部覆盖率 {:.2}）\n",
                rule.template.name(),
                rule.rule_text(),
                rule.support,
                rule.confidence,
                rule.head_coverage
            ));
        }
        if rules.is_empty() {
            output.push_str("  （暂无满足阈值的规则，可降低支持度/置信度阈值或增加关系数据）\n");
        }
        output
    }
}

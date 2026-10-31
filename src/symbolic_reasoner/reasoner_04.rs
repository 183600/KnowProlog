impl SymbolicReasoner {
    /// 会话事实/假设世界变化会改变 `graph_facts` 的输入，上一张图谱模式的
    /// 结果页与见证下标必须整体失效，不能继续按旧事实回答追问。
    fn invalidate_graph_session(&self) {
        *self.graph_session.lock() = None;
        // 会话事实/假设可能已直接注入归结引擎：旧证明不再可复用。
        self.proof_memory.lock().clear();
    }

    fn unindex_session_atom(&mut self, atom: &Atom) {
        if let Some((subject, predicate, object)) = Self::session_atom_triple(atom) {
            self.retrieval
                .lock()
                .remove_triple(subject, predicate, object);
        }
    }

    /// 把一条基事实从会话事实集中撤回（不触碰知识图谱）
    fn retract_atom(&mut self, atom: &Atom) -> bool {
        if !self.session_facts.iter().any(|existing| existing == atom) {
            return false;
        }
        self.session_facts.retain(|existing| existing != atom);
        self.invalidate_graph_session();
        // 图谱事实始终保留；仅当图谱不再支持该原子时才从归结引擎与证据索引
        // 增量删除，避免每次撤回都重建整库（同名的图谱事实与会话事实互不误删）。
        if !self.graph_supports_atom(atom) {
            self.unindex_session_atom(atom);
            if self.config.enable_resolution {
                self.resolver.remove_fact_atom(atom);
            }
        }
        // 软逻辑与概率程序同源：撤回后整体重建（概率程序随软逻辑一起重放）
        if self.config.enable_soft_logic {
            self.rebuild_soft_logic();
        }
        true
    }

    /// 把一条基事实加入会话事实集（不写知识图谱；归结引擎重建后依然保留）
    fn adopt_atom(&mut self, atom: Atom) -> bool {
        if !atom.is_ground() {
            return false;
        }
        let mut added = false;
        let mut session_changed = false;
        if self.config.enable_resolution && self.resolver.add_fact_atom(atom.clone()) {
            added = true;
        }
        if !self.session_facts.contains(&atom) {
            self.session_facts.push(atom.clone());
            self.index_session_atom(&atom);
            added = true;
            session_changed = true;
        }
        if session_changed {
            self.invalidate_graph_session();
        }
        // 会话假设同步注入软逻辑引擎
        if added && self.config.enable_soft_logic {
            self.soft.add_fact(atom.clone());
        }
        // 概率程序同步注入（ProbLog 的确定事实），保持两套语义事实一致
        if added && self.config.enable_problog {
            self.problog.add_deterministic_fact(atom);
        }
        added
    }

    /// 在会话事实层写入一条断言：先经 [`RevisionGuard`] 预检，冲突时只返回
    /// 最小冲突核且不改变任何状态；通过后才镜像到会话事实并记修订回执。
    ///
    /// 返回值：`Ok(true)` 表示确有新增，`Ok(false)` 表示幂等命中已有事实，
    /// `Err(text)` 表示预检拒绝（调用方必须原样展示并可重试）。
    fn commit_session_assertion(&mut self, label: String, atom: &Atom) -> Result<bool, String> {
        self.commit_session_assertions(label, std::slice::from_ref(atom))
            .map(|added| added > 0)
    }

    /// 批量版本：整批一次预检、一次回执；任一条含变量或触发冲突都零写入。
    fn commit_session_assertions(
        &mut self,
        label: String,
        atoms: &[Atom],
    ) -> Result<usize, String> {
        if atoms.iter().any(|atom| !atom.is_ground()) {
            return Err("存在含未绑定变量的假设，无法写入会话事实。".to_string());
        }
        let mut changeset = ChangeSet::new(label.clone());
        for atom in atoms {
            changeset.push(ChangeOp::AssertFact(atom.clone()));
        }
        let plan = self.revision_guard().plan(changeset);
        if plan.is_blocked() {
            return Err(self.render_guard_block(&plan));
        }
        let mut added_ops = Vec::new();
        for atom in atoms {
            self.register_atom_vocabulary(atom);
            // 确认/采纳是显式裁决：功能槽上以「更正」语义替换旧赢家，
            // 不让两个取值同时进入会话事实。
            self.note_slot_adopt(atom, AssertionStrength::Correction, "采纳假设");
            if self.adopt_atom(atom.clone()) {
                added_ops.push(ChangeOp::AssertFact(atom.clone()));
            }
        }
        let added = added_ops.len();
        if !added_ops.is_empty() {
            self.record_session_revision(label, added_ops, Vec::new(), Vec::new());
        }
        Ok(added)
    }

    /// 采纳第 index（1 起）条待定假设：加入会话归结事实集（不写知识图谱）
    pub fn adopt_hypothesis(&mut self, index: usize) -> Option<String> {
        // 先检查可采纳性再取用：不可采纳的假设不能从待定列表中丢失
        let position = index.checked_sub(1)?;
        if !self.abduction.pending().get(position)?.is_adoptable() {
            let hypothesis = &self.abduction.pending()[position];
            let rendered = self.render_atom_nl(&hypothesis.atom);
            return Some(format!(
                "假设 {}「{}」仍含未绑定变量，无法直接采纳；请提供更具体的陈述。",
                index, rendered
            ));
        }
        // 预检失败时保留待定假设：冲突核需要用户裁决，不能静默丢弃。
        let hypothesis = self.abduction.pending().get(position)?.clone();
        let atom = hypothesis.atom.clone();
        let rendered = self.render_atom_nl(&atom);
        let label = format!("采纳假设 {}：{}", index, rendered);
        match self.commit_session_assertion(label, &atom) {
            Err(blocked) => Some(format!(
                "无法采纳假设 {}「{}」：\n{}",
                index, rendered, blocked
            )),
            Ok(added) => {
                let reasons = hypothesis.reasons.join("；");
                self.abduction.take_pending(index);
                self.abduction.adopt(hypothesis);
                let status = if added {
                    "该事实已加入本轮会话的归结事实集（未写入知识图谱），后续「证明/推理链/溯因」会使用它。"
                } else {
                    "该事实已在会话事实集中；本次为幂等采纳，未重复写入。"
                };
                let receipt = self
                    .session_receipts
                    .last()
                    .map(|receipt| receipt.id.clone())
                    .filter(|_| added)
                    .map(|id| format!("\n  修订回执：{}（可用「撤销修订」撤回）", id))
                    .unwrap_or_default();
                Some(format!(
                    "已采纳假设 {}：{}。{}{}\n  依据：{}",
                    index, rendered, status, receipt, reasons
                ))
            }
        }
    }

    pub fn pending_hypotheses_report(&self) -> String {
        let render = |atom: &Atom| self.render_atom_nl(atom);
        abduction::render_pending(self.abduction.pending(), &render)
    }

    /// 供外部查询当前待定假设
    pub fn pending_hypotheses(&self) -> &[Hypothesis] {
        self.abduction.pending()
    }

    /// 会话级假设事实（被采纳、跨会话持久化的临时信念）
    pub fn session_facts(&self) -> &[Atom] {
        &self.session_facts
    }

    /// 槽账本报告：每个槽的当前赢家、被压制值与仅证据值。
    pub fn slot_ledger_report(&self) -> String {
        self.slots.render_report()
    }

    /// 单个槽的改口史（第几次记录、before/after、压制而非否定）。
    pub fn slot_history_report(&self, subject: &str, predicate: &str) -> String {
        let (subject, predicate) = self.slot_identity(subject, predicate);
        self.slots.render_history(&subject, &predicate)
    }

    /// 显式改口：用「更正」语义替换功能槽的当前赢家，并记录 before/after。
    ///
    /// 与 `assert_nl_fact` 的分工：直陈遇到功能槽冲突只问不写；本方法代表
    /// 用户已经作出裁决，因此允许压制旧值。旧值仍在账本里，可撤销、可解释。
    pub fn correct_nl_fact(&mut self, sentence_raw: &str) -> String {
        let sentence = sentence_raw.trim();
        if sentence.is_empty() {
            return "请给出改口内容，例如：改口：姚明的身高是229".to_string();
        }
        let (_, content) = AssertionStrength::classify(sentence);
        let outcome = chinese_reasoning_nlu::parse_lessons(content);
        if outcome.lessons.is_empty() {
            return format!("没能把「{}」解析为可更正的确定事实。", sentence);
        }
        let before_facts = self.session_facts.clone();
        let mut transitions: Vec<SlotTransition> = Vec::new();
        let mut refused: Vec<String> = Vec::new();
        for lesson in &outcome.lessons {
            if lesson.kind != LessonKind::DeterministicFact || !lesson.head.is_ground() {
                refused.push(format!("「{}」不是可更正的确定事实", lesson.original));
                continue;
            }
            self.register_atom_vocabulary(&lesson.head);
            let Some((subject, predicate, object)) =
                Self::session_atom_triple(&lesson.head).map(|(subject, predicate, object)| {
                    (
                        subject.to_string(),
                        predicate.to_string(),
                        object.to_string(),
                    )
                })
            else {
                refused.push(format!("「{}」不是二元事实，无法按槽更正", lesson.original));
                continue;
            };
            match self.apply_session_slot(
                &subject,
                &predicate,
                &object,
                AssertionStrength::Correction,
                "改口",
            ) {
                Ok(transition) => {
                    if transition.changed_state() {
                        self.sync_slot_projection(&subject, &predicate);
                        if let Some(position) = self
                            .session_negative_facts
                            .iter()
                            .position(|negative| negative == &lesson.head)
                        {
                            self.session_negative_facts.remove(position);
                        }
                        self.adopt_atom(lesson.head.clone());
                    }
                    transitions.push(transition);
                }
                Err(error) => refused.push(format!("「{}」未写入：{}", lesson.original, error)),
            }
        }
        // 以事实集的实际增减生成回执，确保撤销能逐字恢复。
        let mut revision_ops: Vec<ChangeOp> = Vec::new();
        for atom in &before_facts {
            if !self.session_facts.contains(atom) {
                revision_ops.push(ChangeOp::RetractFact(atom.clone()));
            }
        }
        for atom in &self.session_facts {
            if !before_facts.contains(atom) {
                revision_ops.push(ChangeOp::AssertFact(atom.clone()));
            }
        }
        if !revision_ops.is_empty() {
            self.record_session_revision(
                format!("改口：{}", sentence),
                revision_ops,
                Vec::new(),
                Vec::new(),
            );
        }
        let mut output = String::new();
        if transitions.is_empty() {
            output.push_str("没有产生任何槽状态转移。\n");
        }
        for transition in &transitions {
            output.push_str(&transition.render_zh());
            output.push('\n');
        }
        for reason in &refused {
            output.push_str(&format!("  跳过：{}\n", reason));
        }
        if let Some(first) = transitions.first() {
            output.push('\n');
            output.push_str(&self.slot_history_report(&first.key.subject, &first.key.predicate));
            output.push('\n');
        }
        if let Some(receipt) = self.session_receipts.last() {
            if receipt.label == format!("改口：{}", sentence) {
                output.push_str(&format!(
                    "修订回执：{}（可用「撤销修订」恢复改口前的取值）。\n",
                    receipt.id
                ));
            }
        }
        output.push_str(&format!("会话事实共 {} 条。", self.session_facts.len()));
        output
    }

    /// 传闻处理：只进槽账本的证据区，不写会话事实、不进入回答。
    fn note_hearsay_nl_fact(&mut self, sentence: &str) -> String {
        let outcome = chinese_reasoning_nlu::parse_lessons(sentence);
        let mut recorded: Vec<SlotTransition> = Vec::new();
        for lesson in &outcome.lessons {
            if lesson.kind != LessonKind::DeterministicFact || !lesson.head.is_ground() {
                continue;
            }
            let Some((subject, predicate, object)) =
                Self::session_atom_triple(&lesson.head).map(|(subject, predicate, object)| {
                    (
                        subject.to_string(),
                        predicate.to_string(),
                        object.to_string(),
                    )
                })
            else {
                continue;
            };
            if let Ok(transition) = self.apply_session_slot(
                &subject,
                &predicate,
                &object,
                AssertionStrength::Hearsay,
                "传闻",
            ) {
                recorded.push(transition);
            }
        }
        if recorded.is_empty() {
            return format!("没能把「{}」记录为可追溯的传闻证据。", sentence);
        }
        let mut output = String::from("已把传闻记为证据（不进入回答、不覆盖当前取值）：\n");
        for transition in &recorded {
            output.push_str(&format!("  {}\n", transition.render_zh()));
        }
        output.trim_end().to_string()
    }

    // ------------------------- 对话式澄清（以问代猜） -------------------------

    /// 失败问题自动进入澄清：先做溯因拿到当前问题的失败前沿与候选解释，
    /// 再直接返回最有判别力的提问，而不是把候选假设清单丢给用户。
    ///
    /// 与 [`Self::hypothesis_answer`] 的分工：
    /// - `hypothesis_answer` 面向「我想看有哪些猜测」的显式溯因命令；
    /// - 本方法面向「问题没答出来，下一步该问什么」的自动化通道。
    ///
    /// 目标已可证、溯因被禁用或无法形式化时返回 `None`，调用方应保留原回答。
    pub fn begin_clarification_for_question(&mut self, question: &str) -> Option<String> {
        if !self.config.enable_abduction || !self.config.enable_clarification {
            return None;
        }
        if self.clarification_active() {
            return Some(self.render_clarification_prompt());
        }
        // 始终按当前问题重算待定池：上一题残留的假设不能劫持这一题的提问。
        let _ = self.hypothesis_answer(question)?;
        self.start_clarification()
    }

    /// 开始澄清对话：把待定假设逐条逆转为「是否成立」的提问
    pub fn start_clarification(&mut self) -> Option<String> {
        if !self.config.enable_abduction || !self.config.enable_clarification {
            return None;
        }
        // 目标在生成待定假设之后可能已被新事实证明：这时再追问只会浪费用户
        // 的确认，直接给证明并清空过期的假设池。
        if let Some((goal, solution)) = self.clarified_goal_solution() {
            self.abduction.clear_pending();
            self.last_abduction_goal = None;
            self.last_abduction_vars.clear();
            let mut output = format!(
                "【对话式澄清】原目标「{}」已可证明（置信度 {:.2}），无需继续追问。",
                goal, solution.confidence
            );
            if let Some(answer) = solution.bindings.get("答案") {
                if !answer.is_empty() {
                    output.push_str(&format!("\n  答案是「{}」。", answer));
                }
            }
            return Some(output);
        }
        let mut session = match &self.last_abduction_goal {
            Some(goals) if !goals.is_empty() => ClarificationSession::start_with_goal(
                self.abduction.pending(),
                &self.resolver,
                goals,
                &self.last_abduction_vars,
            ),
            _ => ClarificationSession::start(self.abduction.pending()),
        };
        if session.is_finished() {
            return None;
        }
        if let Some(goals) = self.last_abduction_goal.clone() {
            let space =
                self.abduction
                    .explanation_space(&self.resolver, &goals, &self.last_abduction_vars);
            session.attach_explanation_space(space);
            // 场景成员可能比待定池更新：先并入，再按解释空间判别力排序。
            session.include_explanation_hypotheses();
            // 解释空间优先：能砍掉更多解释的假设先问，能单独证明目标的最先问。
            session.reprioritize(&self.resolver, &goals, &self.last_abduction_vars);
        }
        self.clarification = Some(session);
        Some(self.render_clarification_prompt())
    }

    fn render_clarification_prompt(&self) -> String {
        let render = |atom: &Atom| self.render_atom_nl(atom);
        match &self.clarification {
            Some(session) => abduction::render_clarification_question(session, &render),
            None => "【对话式澄清】没有进行中的澄清会话。".to_string(),
        }
    }

    pub fn clarification_active(&self) -> bool {
        self.clarification
            .as_ref()
            .map(|session| !session.is_finished())
            .unwrap_or(false)
    }

    /// 处理用户对当前澄清问题的回答（是 / 不是 / 不知道）
    pub fn answer_clarification(&mut self, text: &str) -> Option<String> {
        let reply = {
            let session = self.clarification.as_mut()?;
            session.respond(text)
        };
        match reply {
            ClarificationReply::Confirmed(hypothesis) => {
                let rendered = self.render_atom_nl(&hypothesis.atom);
                let label = format!("澄清确认：{}", rendered);
                match self.commit_session_assertion(label, &hypothesis.atom) {
                    Err(blocked) => {
                        if let Some(session) = self.clarification.as_mut() {
                            session.retry(hypothesis);
                        }
                        Some(format!(
                            "该假设未写入会话事实，澄清问题保留：\n{}\n{}",
                            blocked,
                            self.render_clarification_prompt()
                        ))
                    }
                    Ok(added) => {
                        let mut output = format!(
                            "✓ 已确认假设：{}。{}",
                            rendered,
                            if added {
                                "已加入会话事实集（未写知识图谱）。"
                            } else {
                                "该事实已在会话事实集中（幂等确认）。"
                            }
                        );
                        if !hypothesis.reasons.is_empty() {
                            output.push_str(&format!(
                                "\n  依据：{}",
                                hypothesis.reasons.join("；")
                            ));
                        }
                        if added {
                            if let Some(receipt) = self.session_receipts.last() {
                                output.push_str(&format!(
                                    "\n  修订回执：{}（可用「撤销修订」撤回）",
                                    receipt.id
                                ));
                            }
                        }
                        self.reprioritize_clarification();
                        output.push_str(&self.clarification_follow_up());
                        Some(output)
                    }
                }
            }
            ClarificationReply::Corrected {
                rejected,
                confirmed,
            } => {
                let rejected_rendered = self.render_atom_nl(&rejected.atom);
                let rendered = self.render_atom_nl(&confirmed.atom);
                let label = format!("澄清更正：{}", rendered);
                match self.commit_session_assertion(label, &confirmed.atom) {
                    Err(blocked) => {
                        if let Some(session) = self.clarification.as_mut() {
                            session.retry(confirmed);
                        }
                        Some(format!(
                            "✗ 已排除原假设：{}。\n但更正后的假设未写入会话事实，问题保留：\n{}\n{}",
                            rejected_rendered,
                            blocked,
                            self.render_clarification_prompt()
                        ))
                    }
                    Ok(added) => {
                        let mut output = format!(
                            "✗ 已排除原假设：{}。\n✓ 已确认更正：{}。{}",
                            rejected_rendered,
                            rendered,
                            if added {
                                "已加入会话事实集（未写知识图谱）。"
                            } else {
                                "该事实已在会话事实集中（幂等确认）。"
                            }
                        );
                        if !confirmed.reasons.is_empty() {
                            output.push_str(&format!(
                                "\n  依据：{}",
                                confirmed.reasons.join("；")
                            ));
                        }
                        if added {
                            if let Some(receipt) = self.session_receipts.last() {
                                output.push_str(&format!(
                                    "\n  修订回执：{}（可用「撤销修订」撤回）",
                                    receipt.id
                                ));
                            }
                        }
                        self.reprioritize_clarification();
                        output.push_str(&self.clarification_follow_up());
                        Some(output)
                    }
                }
            }
            ClarificationReply::ConfirmedAll(batch) => {
                let atoms: Vec<Atom> = batch.iter().map(|h| h.atom.clone()).collect();
                match self.commit_session_assertions("澄清批量确认".to_string(), &atoms) {
                    Err(blocked) => {
                        if let Some(session) = self.clarification.as_mut() {
                            session.restore_batch(batch);
                        }
                        Some(format!(
                            "批量确认未写入会话事实（整批零写入），问题保留：\n{}\n{}",
                            blocked,
                            self.render_clarification_prompt()
                        ))
                    }
                    Ok(added) => {
                        let mut output = format!("✓ 已批量确认 {} 条假设：", batch.len());
                        for hypothesis in &batch {
                            output.push_str(&format!(
                                "\n  ✓ {}",
                                self.render_atom_nl(&hypothesis.atom)
                            ));
                        }
                        output.push_str(if added > 0 {
                            "\n已加入会话事实集（未写知识图谱）。"
                        } else {
                            "\n这些事实均已在会话事实集中（幂等确认）。"
                        });
                        if added > 0 {
                            if let Some(receipt) = self.session_receipts.last() {
                                output.push_str(&format!(
                                    "\n  修订回执：{}（可用「撤销修订」撤回）",
                                    receipt.id
                                ));
                            }
                        }
                        self.reprioritize_clarification();
                        output.push_str(&self.clarification_follow_up());
                        Some(output)
                    }
                }
            }
            ClarificationReply::RejectedAll(batch) => {
                let mut output = format!("✗ 已批量排除 {} 条假设：", batch.len());
                for hypothesis in &batch {
                    output.push_str(&format!(
                        "\n  ✗ {}",
                        self.render_atom_nl(&hypothesis.atom)
                    ));
                }
                self.reprioritize_clarification();
                output.push_str(&self.clarification_follow_up());
                Some(output)
            }
            ClarificationReply::Rejected(hypothesis) => {
                self.reprioritize_clarification();
                Some(format!(
                    "✗ 已排除假设：{}。{}",
                    self.render_atom_nl(&hypothesis.atom),
                    self.clarification_follow_up()
                ))
            }
            ClarificationReply::Deferred(hypothesis) => {
                self.reprioritize_clarification();
                Some(format!(
                    "？暂记为不确定：{}。{}",
                    self.render_atom_nl(&hypothesis.atom),
                    self.clarification_follow_up()
                ))
            }
            ClarificationReply::Unrecognized => Some(format!(
                "没有理解回答（请用「是 / 不是 / 不知道」，「不是，是<取值>」更正，或「全部确认」/「全部否定」批量处理）。\n{}",
                self.render_clarification_prompt()
            )),
            ClarificationReply::Finished => Some(self.finish_clarification_session()),
        }
    }

    /// 按最新会话世界重排剩余澄清问题（解释空间判别力优先，可信度为平手）。
    ///
    /// 解释空间是派生状态：每轮从当前知识重新枚举，再用用户的否定承诺过滤，
    /// 因此确认/否定后不会残留已经被事实或回答淘汰的旧解释。
    fn reprioritize_clarification(&mut self) {
        let Some(goals) = self.last_abduction_goal.clone() else {
            return;
        };
        let vars = self.last_abduction_vars.clone();
        let frontier = self
            .clarification
            .as_ref()
            .and_then(ClarificationSession::explanation_space)
            .map(|space| space.frontier().to_vec())
            .unwrap_or_default();
        let space = self.abduction.explanation_space_with_frontier(
            &self.resolver,
            &goals,
            &vars,
            &frontier,
        );
        if let Some(session) = self.clarification.as_mut() {
            session.refresh_explanation_space(space);
            session.include_explanation_hypotheses();
            session.reprioritize(&self.resolver, &goals, &vars);
        }
    }

    /// 只刷新解释空间、不改变当前问题顺序（快照恢复用：派生状态重算，承诺不变）。
    fn refresh_clarification_space(&mut self) {
        let Some(goals) = self.last_abduction_goal.clone() else {
            return;
        };
        if self.clarification.is_none() {
            return;
        }
        let vars = self.last_abduction_vars.clone();
        let frontier = self
            .clarification
            .as_ref()
            .and_then(ClarificationSession::explanation_space)
            .map(|space| space.frontier().to_vec())
            .unwrap_or_default();
        let space = self.abduction.explanation_space_with_frontier(
            &self.resolver,
            &goals,
            &vars,
            &frontier,
        );
        if let Some(session) = self.clarification.as_mut() {
            session.refresh_explanation_space(space);
        }
    }

    /// 澄清推进：若原始目标已得证则汇报答案并结束会话；否则继续下一问。
    ///
    /// 三种提前收束：目标得证、解释空间崩塌、达到最大追问轮数。后两种直接给
    /// 条件化/崩塌结论，避免在已经没有判别力的问题上继续纠缠用户。
    fn clarification_follow_up(&mut self) -> String {
        if let Some((goal, solution)) = self.clarified_goal_solution() {
            let mut text = format!(
                "\n  → 目标「{}」已可证明（置信度 {:.2}）",
                goal, solution.confidence
            );
            if let Some(answer) = solution.bindings.get("答案") {
                if !answer.is_empty() {
                    text.push_str(&format!("：答案是「{}」", answer));
                }
            }
            text.push_str("。\n");
            text.push_str(&self.finish_clarification_session());
            return text;
        }
        let collapsed = self
            .clarification
            .as_ref()
            .map(ClarificationSession::explanation_collapsed)
            .unwrap_or(false);
        if collapsed {
            return format!(
                "\n  已停止追问：继续确认已无法挽救当前解释空间。\n{}",
                self.finish_clarification_session()
            );
        }
        let max_reached = self
            .clarification
            .as_ref()
            .map(ClarificationSession::max_questions_reached)
            .unwrap_or(false);
        if max_reached {
            return format!(
                "\n  已达到本轮最大追问数 {} 轮，先给条件化结论。\n{}",
                self.clarification
                    .as_ref()
                    .map(ClarificationSession::max_questions)
                    .unwrap_or(0),
                self.finish_clarification_session()
            );
        }
        if self.clarification_active() {
            format!("\n{}", self.render_clarification_prompt())
        } else {
            format!("\n{}", self.finish_clarification_session())
        }
    }

    /// 原溯因目标合取是否已被会话事实证明（含"答案"绑定）；返回渲染后的目标描述
    fn clarified_goal_solution(&self) -> Option<(String, Solution)> {
        let goals = self.last_abduction_goal.clone()?;
        if goals.is_empty() {
            return None;
        }
        let vars = self.last_abduction_vars.clone();
        let solution = self.resolver.solve(&goals, &vars).into_iter().next()?;
        let description = goals
            .iter()
            .map(|goal| self.render_atom_nl(goal))
            .collect::<Vec<_>>()
            .join(" 且 ");
        Some((description, solution))
    }

    fn finish_clarification_session(&mut self) -> String {
        let Some(session) = self.clarification.take() else {
            return "【对话式澄清】没有进行中的澄清会话。".to_string();
        };
        let render = |atom: &Atom| self.render_atom_nl(atom);
        let mut output = abduction::render_clarification_summary(&session, &render);
        let proved = self.clarified_goal_solution();
        // 目标已得证时解释空间会因"无需解释"而变空；此时不能再报崩塌。
        if proved.is_none() {
            if let Some(conclusion) =
                session.explanation_conclusion(&self.last_abduction_vars, &render)
            {
                output.push('\n');
                output.push_str(&conclusion);
            }
        }
        if let Some((goal, solution)) = proved {
            output.push_str(&format!(
                "\n  → 当前目标「{}」已可证明（置信度 {:.2}）",
                goal, solution.confidence
            ));
            if let Some(answer) = solution.bindings.get("答案") {
                if !answer.is_empty() {
                    output.push_str(&format!("：答案是「{}」", answer));
                }
            }
            output.push('。');
        }
        output
    }

    /// 手动结束澄清会话：输出已确认/否定/不确定清单
    pub fn stop_clarification(&mut self) -> Option<String> {
        self.clarification.as_ref()?;
        Some(self.finish_clarification_session())
    }

    pub fn clarification_progress(&self) -> Option<String> {
        self.clarification
            .as_ref()
            .map(ClarificationSession::progress_text)
    }

    /// 反事实模拟："如果……那么……" 把条件当临时假设，正向链推导后用后自动回滚
    pub fn what_if_report(&mut self, text: &str) -> Option<String> {
        if !self.config.enable_abduction {
            return None;
        }
        let trimmed = text
            .trim()
            .trim_end_matches(['。', '！', '!', '？', '?', ' ']);
        if trimmed.is_empty() {
            return None;
        }
        // 只取条件部分："如果A那么B" / "如果A，就B" / "如果A则B" → A
        let condition = trimmed
            .strip_prefix("如果")
            .or_else(|| trimmed.strip_prefix("假如"))
            .or_else(|| trimmed.strip_prefix("假设"))
            .unwrap_or(trimmed);
        let condition = ["那么", "则", "，就", ",就"]
            .iter()
            .filter_map(|separator| condition.find(separator))
            .min()
            .map(|index| &condition[..index])
            .unwrap_or(condition)
            .trim()
            .trim_end_matches(['，', ','])
            .trim();
        if condition.is_empty() {
            return None;
        }
        let assumptions = self.condition_atoms(condition);
        if assumptions.is_empty() {
            return None;
        }
        let result =
            self.abduction
                .what_if(&self.resolver, &assumptions, self.resolver.max_depth(), 20);
        let render = |atom: &Atom| self.render_atom_nl(atom);
        Some(abduction::render_what_if(&result, &render))
    }

    /// 当前假设世界（只读）。
    pub fn assumption_world(&self) -> Option<&AssumptionWorld> {
        self.assumption_world.as_ref()
    }

    /// 会话事实 + 假设世界的正向原子（提供者顺序稳定，便于确定性遍历）。
    fn positive_session_atoms(&self) -> impl Iterator<Item = &Atom> {
        self.session_facts.iter().chain(
            self.assumption_world
                .iter()
                .flat_map(|world| world.assumptions.iter()),
        )
    }

    fn inject_assumption_atom(&mut self, atom: &Atom) {
        self.register_atom_vocabulary(atom);
        if self.config.enable_resolution {
            self.resolver.add_fact_atom(atom.clone());
        }
        if self.config.enable_soft_logic {
            self.soft.add_fact(atom.clone());
        }
        if self.config.enable_problog {
            self.problog.add_deterministic_fact(atom.clone());
        }
        self.index_session_atom(atom);
        self.invalidate_graph_session();
    }

    /// 丢弃当前假设世界：只撤回本世界注入且不被图谱/会话事实支持的原子。
    fn discard_assumption_world(&mut self) -> Option<AssumptionWorld> {
        let world = self.assumption_world.take()?;
        self.invalidate_graph_session();
        for atom in &world.assumptions {
            if self.graph_supports_atom(atom) || self.session_facts.contains(atom) {
                continue;
            }
            self.unindex_session_atom(atom);
            if self.config.enable_resolution {
                self.resolver.remove_fact_atom(atom);
            }
        }
        if self.config.enable_soft_logic {
            self.rebuild_soft_logic();
        }
        Some(world)
    }

    fn assumption_world_id(atoms: &[Atom]) -> String {
        let mut keys: Vec<String> = atoms.iter().map(Atom::display).collect();
        keys.sort();
        keys.dedup();
        let digest = canonical::canonical_digest(&keys).unwrap_or_default();
        format!("w-{}", &digest[..digest.len().min(8)])
    }

    fn restore_assumption_world(&mut self, atoms: Vec<Atom>) {
        if atoms.is_empty() {
            return;
        }
        let mut unique: Vec<Atom> = Vec::new();
        for atom in atoms {
            if atom.is_ground() && !unique.contains(&atom) {
                unique.push(atom);
            }
        }
        if unique.is_empty() {
            return;
        }
        let id = Self::assumption_world_id(&unique);
        for atom in &unique {
            self.inject_assumption_atom(atom);
        }
        self.assumption_world = Some(AssumptionWorld {
            id,
            assumptions: unique,
        });
    }

    /// 进入一个跨轮假设世界：假设只在本世界内参与推理，退出即整体丢弃。
    ///
    /// 支持句式与反事实模拟一致（「S的P是O」「S是类型」等），条件之间可用
    /// `；` / `，` 分隔。返回可见回执；解析失败时返回可读原因。
    pub fn enter_assumption_world(&mut self, condition: &str) -> Result<String, String> {
        let condition = condition
            .trim()
            .trim_matches(['，', ',', '：', ':', '。', ' ']);
        if condition.is_empty() {
            return Err("请给出假设内容，例如：假设张三是李四的师傅。".to_string());
        }
        let mut atoms: Vec<Atom> = Vec::new();
        for atom in self.condition_atoms(condition) {
            if !atom.is_ground() {
                return Err(format!(
                    "假设「{}」含未绑定变量，无法作为跨轮前提。",
                    self.render_atom_nl(&atom)
                ));
            }
            if !atoms.contains(&atom) {
                atoms.push(atom);
            }
        }
        if atoms.is_empty() {
            return Err(format!(
                "无法把「{}」形式化为可模拟的假设。支持句式：假设张三是李四的师傅。",
                condition
            ));
        }
        self.discard_assumption_world();
        for atom in &atoms {
            self.inject_assumption_atom(atom);
        }
        let id = Self::assumption_world_id(&atoms);
        let rendered = atoms
            .iter()
            .map(|atom| self.render_atom_nl(atom))
            .collect::<Vec<_>>()
            .join("；");
        self.assumption_world = Some(AssumptionWorld {
            id: id.clone(),
            assumptions: atoms,
        });
        Ok(format!(
            "已进入假设世界 {}（假设：{}）。\n  本世界内的后续追问会把它当作临时前提；问答与图谱均不写入。\n  可以继续追问，或说「退出假设」丢弃，或说「记住这个假设」并入会话层。",
            id, rendered
        ))
    }

    /// 退出当前假设世界：临时前提整体丢弃，基础图谱与会话事实不受影响。
    pub fn exit_assumption_world(&mut self) -> Option<String> {
        let world = self.discard_assumption_world()?;
        Some(format!(
            "已退出假设世界 {}，{} 条临时假设已丢弃（未写入图谱）。",
            world.id,
            world.assumptions.len()
        ))
    }

    /// 把当前假设世界的临时前提并入会话事实层（仍不写知识图谱）。
    pub fn remember_assumption_world(&mut self) -> Option<String> {
        // 世界先保留：预检失败时用户仍可修正或退出，不静默丢弃假设。
        let world = self.assumption_world.as_ref()?.clone();
        let label = format!("记住假设世界 {}", world.id);
        match self.commit_session_assertions(label, &world.assumptions) {
            Err(blocked) => Some(format!(
                "假设世界未并入会话事实（世界仍保留）：\n{}",
                blocked
            )),
            Ok(added) => {
                self.assumption_world = None;
                let receipt = self
                    .session_receipts
                    .last()
                    .filter(|_| added > 0)
                    .map(|receipt| format!("；修订回执 {}", receipt.id))
                    .unwrap_or_default();
                Some(format!(
                    "已把 {} 条假设并入会话事实层（新增 {} 条，仍未写入图谱；来源：假设世界 {}{}）。后续回答会把它当作会话前提。",
                    world.assumptions.len(),
                    added,
                    world.id,
                    receipt
                ))
            }
        }
    }

    /// 假设世界的可见状态（无世界时也给出明确说明）。
    pub fn assumption_world_report(&self) -> String {
        match &self.assumption_world {
            Some(world) => {
                let rendered = world
                    .assumptions
                    .iter()
                    .map(|atom| self.render_atom_nl(atom))
                    .collect::<Vec<_>>()
                    .join("；");
                format!(
                    "假设世界 {}（{} 条临时假设）：{}。\n  该世界内的事实不写图谱；可说「退出假设」或「记住这个假设」。",
                    world.id,
                    world.assumptions.len(),
                    rendered
                )
            }
            None => {
                "当前不在假设世界。可用「假设：张三是李四的师傅」进入一个跨轮假设世界。".to_string()
            }
        }
    }

    /// 解析假设世界命令：进入 / 退出 / 记住 / 查看。
    ///
    /// 带「那么/则」的一次性反事实句不在此接管，仍交给 `what_if_report`。
    pub fn handle_assumption_command(&mut self, input: &str) -> Option<String> {
        let trimmed = input
            .trim()
            .trim_end_matches(['。', '！', '!', '？', '?', ' '])
            .trim();
        if trimmed.is_empty() {
            return None;
        }
        if matches!(
            trimmed,
            "退出假设" | "退出假设世界" | "回到现实" | "离开假设" | "退出假定"
        ) {
            return Some(
                self.exit_assumption_world()
                    .unwrap_or_else(|| "当前不在假设世界，无需退出。".to_string()),
            );
        }
        if matches!(
            trimmed,
            "记住假设" | "记住这个假设" | "采纳假设世界" | "采纳这个假设" | "确认这个假设"
        ) {
            return Some(
                self.remember_assumption_world()
                    .unwrap_or_else(|| "当前没有可记住的假设世界。".to_string()),
            );
        }
        if matches!(trimmed, "假设世界" | "当前假设" | "假设状态" | "查看假设") {
            return Some(self.assumption_world_report());
        }
        for prefix in [
            "假设世界：",
            "假设世界:",
            "进入假设：",
            "进入假设:",
            "假设：",
            "假设:",
        ] {
            if let Some(condition) = trimmed.strip_prefix(prefix) {
                if condition.contains("那么") || condition.contains("则") {
                    return None;
                }
                return Some(match self.enter_assumption_world(condition) {
                    Ok(text) => text,
                    Err(reason) => reason,
                });
            }
        }
        for prefix in ["假设", "假如"] {
            if let Some(condition) = trimmed.strip_prefix(prefix) {
                if condition.contains("那么")
                    || condition.contains("则")
                    || condition.contains('？')
                    || condition.contains('?')
                {
                    return None;
                }
                if condition.trim().is_empty() {
                    return Some(self.assumption_world_report());
                }
                // 裸「假设」可能与普通名词短语冲突：解析失败时让位给常规管线，
                // 只有成功形式化后才接管。
                return self.enter_assumption_world(condition).ok();
            }
        }
        None
    }

    /// 给回答追加假设世界标签（没有世界时逐字节不变）。
    fn annotate_assumption_world(&self, answer: String) -> String {
        let Some(world) = &self.assumption_world else {
            return answer;
        };
        let rendered = world
            .assumptions
            .iter()
            .map(|atom| self.render_atom_nl(atom))
            .collect::<Vec<_>>()
            .join("；");
        format!(
            "{}\n（在假设世界 {} 内；临时前提：{}；未写入图谱）",
            answer, world.id, rendered
        )
    }

    /// 会话增广的知识视图：蕴含判定、自检与生成看到同一批会话事实。
    ///
    /// 只在会话层非空时克隆一次图谱视图；会话三元组同时登记为关系和属性，
    /// 以适配形式化解析器对谓词类型的不同归类。
    fn session_aware_knowledge(&self) -> (LinkPredictor, SymbolicGenerator) {
        let mut predictor = self.predictor.clone();
        let mut generator = self.generator.clone();
        let mut triples: BTreeSet<(String, String, String)> = BTreeSet::new();
        for atom in &self.session_facts {
            if let Some((subject, predicate, object)) = Self::session_atom_triple(atom) {
                triples.insert((
                    subject.to_string(),
                    predicate.to_string(),
                    object.to_string(),
                ));
            }
        }
        for (subject, predicate, object) in self.slots.projection() {
            triples.insert((subject, predicate, object));
        }
        for (subject, predicate, object) in triples {
            predictor.add_triple(&subject, &predicate, &object);
            generator.observe_triple(&subject, &predicate, &object);
            let etype = self
                .generator
                .entity_type_of(&subject)
                .unwrap_or("")
                .to_string();
            generator.observe_entity(&subject, &etype, &[(predicate.clone(), object.clone())]);
        }
        (predictor, generator)
    }

    // 生成审计：逐句回译为命题并验证是否被知识图谱（∪ 会话层）蕴含。
    // `trusted` 是账本背书的会话层原句：整句精确命中即视为已证，
    // 避免把「刚教的事实」误报为幻觉。
    //
    // 与旧版自检的关键区别：
    //   • 一句话形式化出多个子句时必须**全部**被蕴含，任一子句失败即整句
    //     丢弃，堵住「真事实 + 假事实」拼接的幻觉后门；
    //   • 无法形式化且不在 trusted 白名单里的句子不再静默跳过，而是计入
    //     失败并丢弃（宁可少说，不冒充事实）；
    //   • 未通过的句子不进入最终输出，审计只报告实际保留下来的句子。
    fn audit_generation(&self, text: &str, trusted: &[String]) -> GenerationAudit {
        if !self.config.enable_entailment {
            return GenerationAudit {
                text: text.to_string(),
                note: None,
            };
        }
        let parser = self.proposition_parser();
        let augmented = if self.session_facts.is_empty() && self.slots.is_empty() {
            None
        } else {
            Some(self.session_aware_knowledge())
        };
        let knowledge = match &augmented {
            Some((predictor, generator)) => semantic_entailment::Knowledge {
                predictor,
                generator,
                min_rule_confidence: self.config.entailment_min_confidence,
                max_rule_candidates: 16,
                slot_policy: Some(self.slots.policy()),
            },
            None => self.knowledge(),
        };
        let mut kept: Vec<&str> = Vec::new();
        let mut total = 0usize;
        let mut entailed = 0usize;
        let mut failed: Vec<String> = Vec::new();
        for raw_line in text.lines() {
            let sentence = semantic_entailment::strip_discourse_markers(raw_line.trim());
            if !sentence.ends_with('。') {
                kept.push(raw_line);
                continue;
            }
            // 更正句是账本差分的元叙述，不是对新事实的断言。
            if sentence.starts_with("更正：") || sentence.starts_with("补充：") {
                kept.push(raw_line);
                continue;
            }
            let normalized = sentence.trim_end_matches('。');
            let trusted_hit = trusted
                .iter()
                .any(|known| known.trim().trim_end_matches('。') == normalized);
            let propositions = parser.parse(sentence);
            let supported = if trusted_hit {
                Some(())
            } else if propositions.is_empty() {
                None
            } else if propositions.iter().all(|proposition| {
                semantic_entailment::entail(proposition, &knowledge).verdict == Verdict::Entailed
            }) {
                Some(())
            } else {
                None
            };
            total += 1;
            if supported.is_some() {
                entailed += 1;
                kept.push(raw_line);
            } else {
                failed.push(sentence.to_string());
            }
        }
        let text = kept.join("\n");
        let note = if total == 0 {
            None
        } else if failed.is_empty() {
            Some(format!(
                "（生成自检：{}/{} 句可由知识图谱蕴含，零幻觉）",
                entailed, total
            ))
        } else {
            Some(format!(
                "（生成自检：{}/{} 句可由知识图谱蕴含，未通过 {} 句并已丢弃：{}）",
                entailed,
                total,
                failed.len(),
                failed.join("；")
            ))
        };
        GenerationAudit { text, note }
    }

    // 符号续写：对「X的」做谓词预测，支持实体级→类型级→全局回退
    pub fn predict_continuation(&self, prefix: &str) -> Option<String> {
        if !self.config.enable_generation {
            return None;
        }
        let trimmed = prefix.trim();
        let normalized = trimmed.strip_suffix('的').map(|root| {
            let root = root.trim();
            let canonical = self.aligner.canonical_form(root);
            if !canonical.is_empty() && canonical != root {
                format!("{}的", canonical)
            } else {
                trimmed.to_string()
            }
        });
        let base = self
            .generator
            .complete(normalized.as_deref().unwrap_or(trimmed))?;
        // 统计回退之外再挂一层"关联规则联想"：由已有特征推荐缺失特征（纯计数）
        if self.config.enable_association_rules {
            if let Some(root) = trimmed.strip_suffix('的') {
                if let Some(extra) = self.association_suggestions(root.trim()) {
                    let mut output = base;
                    output.push('\n');
                    output.push_str(&extra);
                    return Some(output);
                }
            }
        }
        Some(base)
    }

    // ===== 归结推理：证明、推理链与关联规则 =====

    fn resolve_known_entity(&self, raw: &str) -> Option<String> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return None;
        }
        let canonical = self.aligner.canonical_form(trimmed);
        let candidate = if canonical.is_empty() {
            trimmed.to_string()
        } else {
            canonical
        };
        if self.generator.is_known_entity(&candidate)
            || self.associations.basket(&candidate).is_some()
        {
            return Some(candidate);
        }
        self.best_linked_entity(trimmed).filter(|entity| {
            self.generator.is_known_entity(entity) || self.associations.basket(entity).is_some()
        })
    }

    // 解析 "S的P1的P2...是O"（谓词串可为空 ⇒ 类型断言）
    fn parse_chain_statement(&self, text: &str) -> Option<(String, Vec<ChainPredicate>, String)> {
        let trimmed = text.trim().trim_end_matches(['。', '！', '!']).trim();
        let (subject_part, object_part) = trimmed.rsplit_once('是')?;
        let object = self.aligner.canonical_form(object_part.trim());
        if object.is_empty() {
            return None;
        }
        let segments: Vec<String> = subject_part
            .split('的')
            .map(|segment| segment.trim().to_string())
            .filter(|segment| !segment.is_empty())
            .collect();
        if segments.is_empty() {
            return None;
        }
        if segments.len() == 1 {
            let entity = self.aligner.canonical_form(&segments[0]);
            if entity.is_empty() {
                return None;
            }
            return Some((entity, Vec::new(), object));
        }
        let (entity, predicates) = self.split_chain(&segments)?;
        if predicates.is_empty() {
            return None;
        }
        Some((entity, predicates, object))
    }

    fn render_atom_nl(&self, atom: &Atom) -> String {
        if !atom.is_ground() {
            return atom.display();
        }
        let names: Vec<String> = atom
            .terms
            .iter()
            .map(|term| {
                let canonical = render_term(term);
                self.alignment_surface_choice(&canonical)
                    .map(|(surface, _reason)| surface)
                    .unwrap_or(canonical)
            })
            .collect();
        if &*atom.predicate == PROOF_HAS && names.len() == 2 {
            return format!("{} 具有 {}", names[0], names[1]);
        }
        if &*atom.predicate == PROOF_TYPE && names.len() == 2 {
            return format!("{} 属于类型 {}", names[0], names[1]);
        }
        if names.len() == 2 {
            let sentence =
                self.generator
                    .render_fact(&names[0], &atom.predicate, &[names[1].clone()]);
            return sentence.trim_end_matches('。').to_string();
        }
        atom.display()
    }

    fn render_proof_tree(&self, node: &ProofNode, lines: &mut Vec<String>, indent: usize) {
        let padding = "    ".repeat(indent);
        let fact = self.render_atom_nl(&node.atom);
        match &node.source {
            ProofSource::Fact => lines.push(format!("{}{}（事实）", padding, fact)),
            ProofSource::Rule { source } => lines.push(format!(
                "{}{}（由 {}；本步置信度 {:.2}）",
                padding, fact, source, node.confidence
            )),
        }
        for child in &node.children {
            self.render_proof_tree(child, lines, indent + 1);
        }
    }

    /// 结构化证明视图：与报告共用同一套解析/求解路径，供可视化消费。
    /// 不变量：每棵树的节点集 ≡ `Solution.proofs` 展平集合（不增不减）。
    fn build_resolution_view(&self, phrase: String, goals: Vec<Atom>) -> ResolutionProofView {
        // 目标中出现的变量即查询变量（按首现顺序，确定）：绑定表随视图返回
        let mut query_vars: Vec<String> = Vec::new();
        for goal in &goals {
            for term in &goal.terms {
                if term.is_var() {
                    let name = term.name().to_string();
                    if !query_vars.contains(&name) {
                        query_vars.push(name);
                    }
                }
            }
        }
        let outcome = self.resolver.solve_outcome(&goals, &query_vars);
        let goal_display = goals
            .iter()
            .map(|goal| goal.display())
            .collect::<Vec<_>>()
            .join("，");
        let solutions = outcome
            .solutions
            .iter()
            .map(|solution| {
                let mut bindings: Vec<(String, String)> = solution
                    .bindings
                    .iter()
                    .map(|(name, value)| (name.clone(), value.clone()))
                    .collect();
                bindings.sort();
                ProofSolutionView {
                    bindings,
                    confidence: solution.confidence,
                    proofs: solution
                        .proofs
                        .iter()
                        .map(|proof| {
                            ProofViewNode::from_proof_with(proof, |node| {
                                self.render_atom_nl(&node.atom)
                            })
                        })
                        .collect(),
                }
            })
            .collect();
        ResolutionProofView {
            phrase,
            goal_display,
            facts: self.resolver.fact_count(),
            clauses: self.resolver.clause_count(),
            max_depth: self.resolver.max_depth(),
            search_status: outcome.status,
            search_truncated: outcome.is_truncated(),
            frontier: outcome.frontier,
            stats: outcome.stats,
            solutions,
        }
    }

    /// 陈述句目标（"S有P" / "S的P…是O" / "S是类型"）→ (短语, 目标原子)。
    fn statement_goals(&self, text: &str) -> Option<(String, Vec<Atom>)> {
        if let Some(goals) = self.existence_goals(text) {
            return Some(goals);
        }
        let trimmed = text.trim().trim_end_matches(['。', '！', '!']).trim();
        let (entity, predicates, object) = self.parse_chain_statement(trimmed)?;
        let surfaces: Vec<&str> = predicates.iter().map(|p| p.surface.as_str()).collect();
        let (goals, phrase) = if predicates.is_empty() {
            (
                vec![Atom::rel(PROOF_TYPE, &entity, &object)],
                format!("{} 属于类型 {}", entity, object),
            )
        } else if predicates.len() == 1 {
            let predicate = &predicates[0];
            // 逆向谓词的陈述按槽位反方向编码（slot(object, entity)）
            let atom = if predicate.inverted {
                Atom::rel(&predicate.slot, &object, &entity)
            } else {
                Atom::rel(&predicate.slot, &entity, &object)
            };
            (
                vec![atom],
                format!("{}的{}是{}", entity, predicate.surface, object),
            )
        } else {
            let mut goals = Vec::new();
            let mut previous = Term::constant(&entity);
            for (index, predicate) in predicates.iter().enumerate() {
                let next = if index + 1 == predicates.len() {
                    Term::constant(&object)
                } else {
                    Term::var(&format!("V{}", index + 1))
                };
                let atom = if predicate.inverted {
                    Atom::pattern(&predicate.slot, next.clone(), previous.clone())
                } else {
                    Atom::pattern(&predicate.slot, previous.clone(), next.clone())
                };
                goals.push(atom);
                previous = next;
            }
            (
                goals,
                format!("{}的{}是{}", entity, surfaces.join("的"), object),
            )
        };
        Some((phrase, goals))
    }

    /// "X有P" 形式的存在性目标（由关联规则/前向推导证明）。
    fn existence_goals(&self, text: &str) -> Option<(String, Vec<Atom>)> {
        let (subject_part, object_part) = text.split_once('有')?;
        let subject_part = subject_part.trim();
        let object_part = object_part.trim();
        if subject_part.is_empty() || object_part.is_empty() {
            return None;
        }
        if subject_part.contains('的') || object_part.contains('的') {
            return None;
        }
        let entity = self.resolve_known_entity(subject_part)?;
        let predicate = self.aligner.canonical_form(object_part);
        if predicate.is_empty() {
            return None;
        }
        let goals = vec![Atom::rel(PROOF_HAS, &entity, &predicate)];
        Some((format!("{} 具有 {}", entity, predicate), goals))
    }

    /// 把主谓宾/被字句/把字句计划直接转成归结目标，供「为什么」重建证明。
    fn relation_plan_view(&self, plan: &QueryPlan) -> Option<ResolutionProofView> {
        let answer = || Term::var("答案");
        let (phrase, goals) = match plan {
            QueryPlan::FindObjectByPredicate { subject, predicate } => {
                let entity = self.resolve_known_entity(subject)?;
                let (raw_slot, inverted) = self.resolve_question_predicate(predicate);
                let slot = self.proof_predicate_slot(predicate, &raw_slot, inverted);
                if slot.is_empty() {
                    return None;
                }
                let goal = if inverted {
                    Atom::pattern(&slot, answer(), Term::constant(&entity))
                } else {
                    Atom::pattern(&slot, Term::constant(&entity), answer())
                };
                (format!("{} {} 谁", entity, predicate), vec![goal])
            }
            QueryPlan::FindSubject { object, predicate } => {
                let target = self.resolve_known_entity(object)?;
                let (raw_slot, inverted) = self.resolve_question_predicate(predicate);
                let slot = self.proof_predicate_slot(predicate, &raw_slot, inverted);
                if slot.is_empty() {
                    return None;
                }
                let goal = if inverted {
                    Atom::pattern(&slot, Term::constant(&target), answer())
                } else {
                    Atom::pattern(&slot, answer(), Term::constant(&target))
                };
                (format!("谁 {} {}", predicate, target), vec![goal])
            }
            QueryPlan::CheckRelation {
                subject,
                predicate,
                object,
                negated,
            } => {
                if *negated {
                    // 否定问句的证明需要否定即失败语义；共享的归结视图
                    // 只重建肯定事实，这里保守弃权。
                    return None;
                }
                let subject = self.resolve_known_entity(subject)?;
                let object = self.resolve_known_entity(object)?;
                let (raw_slot, inverted) = self.resolve_question_predicate(predicate);
                let slot = self.proof_predicate_slot(predicate, &raw_slot, inverted);
                if slot.is_empty() {
                    return None;
                }
                let goal = if inverted {
                    Atom::rel(&slot, &object, &subject)
                } else {
                    Atom::rel(&slot, &subject, &object)
                };
                (format!("{} {} {}", subject, predicate, object), vec![goal])
            }
            _ => return None,
        };
        Some(self.build_resolution_view(phrase, goals))
    }
}

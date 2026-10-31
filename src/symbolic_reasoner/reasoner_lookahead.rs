impl SymbolicReasoner {
    fn clarification_lookahead_enabled(&self) -> bool {
        self.config.enable_clarification_lookahead
            && self.config.enable_clarification
            && self.config.enable_abduction
            && self.config.enable_resolution
    }

    /// 当前澄清问题的封印：任何一项变化都不得复用旧分支证据。
    fn clarification_branch_seal(&self, session: &ClarificationSession) -> BranchSeal {
        let engine = lookahead::engine_digest(&self.resolver);
        let mut goals: Vec<String> = self
            .last_abduction_goal
            .iter()
            .flatten()
            .map(Atom::display)
            .collect();
        goals.extend(
            self.last_abduction_vars
                .iter()
                .map(|var| format!("?{}", var)),
        );
        let pending: Vec<String> = self
            .abduction
            .pending()
            .iter()
            .map(|hypothesis| hypothesis.atom.display())
            .collect();
        let commitments: Vec<String> = session
            .confirmed
            .iter()
            .chain(session.rejected.iter())
            .chain(session.deferred.iter())
            .map(|hypothesis| hypothesis.atom.display())
            .collect();
        let current = session
            .current()
            .map(|hypothesis| hypothesis.atom.display());
        let queue: Vec<String> = session
            .queue()
            .iter()
            .map(|hypothesis| hypothesis.atom.display())
            .collect();
        let mut space: Vec<String> = Vec::new();
        if let Some(explanations) = session.explanation_space() {
            space.push(format!("bounded:{}", explanations.is_bounded()));
            space.extend(
                explanations
                    .frontier()
                    .iter()
                    .map(|atom| format!("frontier:{}", atom.display())),
            );
            for scenario in explanations.explanations() {
                space.push("scenario".to_string());
                space.push(format!("verified:{}", scenario.verified));
                space.push(format!(
                    "confidence:{}",
                    scenario.joint_confidence.to_bits()
                ));
                space.push(format!("members:{}", scenario.hypotheses.len()));
                space.extend(
                    scenario
                        .atoms()
                        .iter()
                        .map(|atom| format!("member:{}", atom.display())),
                );
            }
        }
        BranchSeal::compute(
            self.revision,
            session.questions_asked(),
            self.context.focus().unwrap_or_default(),
            engine,
            &goals,
            &pending,
            &commitments,
            current.as_deref(),
            &queue,
            &space,
        )
    }

    /// 预计算下一句常见短答的派生证据包；不修改真实会话、不写图谱。
    fn prefetch_clarification_lookahead(&self) {
        if !self.clarification_lookahead_enabled() {
            self.clarification_lookahead.lock().clear();
            return;
        }
        let Some(session) = self.clarification.as_ref() else {
            self.clarification_lookahead.lock().clear();
            return;
        };
        if session.is_finished() {
            self.clarification_lookahead.lock().clear();
            return;
        }
        let seal = self.clarification_branch_seal(session);
        let seeds =
            lookahead::seed_branch_answers(session, self.config.clarification_lookahead_branches);
        if seeds.is_empty() {
            self.clarification_lookahead.lock().clear();
            return;
        }
        let goals = self.last_abduction_goal.clone().unwrap_or_default();
        let vars = self.last_abduction_vars.clone();
        let mut prepared: Vec<(BranchKey, BranchEvidence)> = Vec::with_capacity(seeds.len());
        for (key, answer) in seeds {
            let mut probe = session.clone();
            let reply = probe.respond(answer);
            if BranchKey::from_reply(&reply).as_ref() != Some(&key) {
                continue;
            }
            // 探针引擎只重放真实提交会做的事实注入；软逻辑/概率程序不影响解释空间。
            let mut engine = self.resolver.clone();
            match &reply {
                ClarificationReply::Confirmed(hypothesis) => {
                    engine.add_fact_atom(hypothesis.atom.clone());
                }
                ClarificationReply::Corrected { confirmed, .. } => {
                    engine.add_fact_atom(confirmed.atom.clone());
                }
                ClarificationReply::ConfirmedAll(batch) => {
                    for hypothesis in batch {
                        engine.add_fact_atom(hypothesis.atom.clone());
                    }
                }
                ClarificationReply::Rejected(_)
                | ClarificationReply::Deferred(_)
                | ClarificationReply::RejectedAll(_)
                | ClarificationReply::Unrecognized
                | ClarificationReply::Finished => {}
            }
            let state = lookahead::session_state_digest(&probe);
            if !goals.is_empty() {
                let frontier = probe
                    .explanation_space()
                    .map(|space| space.frontier().to_vec())
                    .unwrap_or_default();
                let space = self
                    .abduction
                    .explanation_space_with_frontier(&engine, &goals, &vars, &frontier);
                probe.refresh_explanation_space(space);
                probe.include_explanation_hypotheses();
                probe.reprioritize(&engine, &goals, &vars);
            }
            let goal_proved = !goals.is_empty() && !engine.solve(&goals, &vars).is_empty();
            let digest = lookahead::engine_digest(&engine);
            prepared.push((
                key,
                BranchEvidence::with_state(probe, digest, state, goal_proved),
            ));
        }
        let mut board = self.clarification_lookahead.lock();
        board.clear();
        let count = prepared.len() as u64;
        for (key, evidence) in prepared {
            board.insert(&seal, key, evidence);
        }
        drop(board);
        self.perf
            .bump_by("clarification_lookahead_prefetched", count);
    }

    /// 读取当前回答对应的分支证据；未命中/禁用时返回 `None`。
    fn take_clarification_lookahead(&self, text: &str) -> Option<(BranchKey, BranchEvidence)> {
        if !self.clarification_lookahead_enabled() {
            return None;
        }
        let session = self.clarification.as_ref()?;
        let key = lookahead::preview_key(session, text)?;
        let seal = self.clarification_branch_seal(session);
        let mut board = self.clarification_lookahead.lock();
        let evidence = board.take(&seal, &key);
        drop(board);
        if evidence.is_some() {
            self.perf.bump("clarification_lookahead_hits");
        } else {
            self.perf.bump("clarification_lookahead_misses");
        }
        evidence.map(|evidence| (key, evidence))
    }

    /// 提交成功后应用分支证据：会话承诺与引擎摘要都逐位一致才采用。
    fn apply_clarification_lookahead(
        &mut self,
        evidence: Option<(BranchKey, BranchEvidence)>,
    ) -> bool {
        let Some((key, evidence)) = evidence else {
            return false;
        };
        let engine_matches = lookahead::engine_digest(&self.resolver) == evidence.engine;
        let state_matches = self
            .clarification
            .as_ref()
            .map(|session| lookahead::session_state_digest(session) == evidence.state)
            .unwrap_or(false);
        if !engine_matches || !state_matches {
            self.perf.bump("clarification_lookahead_rejected");
            return false;
        }
        self.clarification = Some(evidence.session);
        self.perf.bump("clarification_lookahead_applied");
        self.perf
            .bump(&format!("clarification_lookahead_applied_{}", key.code()));
        true
    }

    /// 前瞻报告：只读、确定性，展示封印与命中/应用计数。
    pub fn clarification_lookahead_report(&self) -> String {
        let board = self.clarification_lookahead.lock();
        let mut output = format!(
            "【澄清分支前瞻】已预计算 {} 条（容量 {}）",
            board.len(),
            board.capacity()
        );
        if let Some(seal) = board.seal() {
            output.push_str(&format!(
                "\n  封印：版本 {}、轮次 {}、焦点「{}」、引擎 {:016x}、当前「{}」",
                seal.revision, seal.turn, seal.focus, seal.engine, seal.current
            ));
        }
        drop(board);
        output.push_str(&format!(
            "\n  命中 {} 次 / 未命中 {} 次 / 应用 {} 次 / 拒绝 {} 次 / 预计算 {} 条",
            self.perf.count("clarification_lookahead_hits"),
            self.perf.count("clarification_lookahead_misses"),
            self.perf.count("clarification_lookahead_applied"),
            self.perf.count("clarification_lookahead_rejected"),
            self.perf.count("clarification_lookahead_prefetched"),
        ));
        output
    }

    /// 清空前瞻缓存；返回被清除的条目数。
    pub fn clear_clarification_lookahead(&self) -> usize {
        let mut board = self.clarification_lookahead.lock();
        let count = board.len();
        board.clear();
        count
    }
}

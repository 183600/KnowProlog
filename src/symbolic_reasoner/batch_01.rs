impl SymbolicReasoner {
    /// 当前挂起的批次裁决（无则 None）。
    pub fn pending_batch_turn(&self) -> Option<&BatchPendingTurn> {
        self.batch_pending.as_ref()
    }

    /// 待裁决批次的用户可读报告。
    pub fn pending_batch_report(&self) -> Option<String> {
        self.batch_pending
            .as_ref()
            .map(BatchPendingTurn::render_zh)
    }

    /// 显式批处理入口：即使只有一条 claim 也按批次规范化（例如「张三30岁」）。
    pub fn assert_nl_batch(&mut self, text: &str, mode: BatchMode) -> String {
        let text = text.trim();
        if text.is_empty() {
            return "请给出要批量记住的事实，例如：批量记住：张三30岁；李四32岁".to_string();
        }
        let outcome = batch_conflict::parse_batch_claims(text);
        if outcome.claims.is_empty() {
            return format!(
                "没能从「{}」解析出可裁决的 claim。{}",
                text,
                render_batch_errors(&outcome.errors)
            );
        }
        self.assert_batch_outcome(text, outcome, mode)
    }

    /// 自动批处理：多 claim 时启用；单 claim 返回 None 以保持旧路径不变。
    fn try_assert_nl_batch(&mut self, sentence: &str, forced: Option<BatchMode>) -> Option<String> {
        let outcome = batch_conflict::parse_batch_claims(sentence);
        if outcome.claims.is_empty() {
            return None;
        }
        if outcome.claims.len() < 2 && forced.is_none() {
            let not_but_correction = outcome.claims.len() == 1
                && outcome.claims[0].correction_cue.as_deref() == Some("不是…而是");
            if !not_but_correction {
                return None;
            }
        }
        Some(self.assert_batch_outcome(sentence, outcome, forced.unwrap_or(BatchMode::Progressive)))
    }

    fn apply_batch_evidence(
        &mut self,
        claims: &[BatchClaim],
        indices: &[usize],
    ) -> Vec<String> {
        let mut notes = Vec::new();
        for &index in indices {
            let Some(claim) = claims.get(index) else {
                continue;
            };
            if let Ok(transition) = self.apply_session_slot(
                &claim.subject,
                &claim.predicate,
                &claim.object,
                AssertionStrength::Hearsay,
                "批量传闻",
            ) {
                notes.push(transition.render_zh());
            }
        }
        notes
    }

    fn assert_batch_outcome(
        &mut self,
        text: &str,
        outcome: BatchParseOutcome,
        mode: BatchMode,
    ) -> String {
        if let Some(pending) = &self.batch_pending {
            return format!(
                "上一批仍有 {} 处待确认，请先「批答：…」或「批答：取消」。\n{}",
                pending.questions.len(),
                pending.render_zh()
            );
        }
        let label = format!("批量记住：{}", text);
        let mut claims = outcome.claims;
        for claim in &mut claims {
            claim.canonicalize_with(|subject, predicate| self.slot_identity(subject, predicate));
        }

        let mut functional: BTreeSet<String> = self
            .slots
            .policy()
            .functional_predicates()
            .map(str::to_string)
            .collect();
        functional.extend(self.session_revision_constraints().functional.iter().cloned());
        let mut base_values: BTreeMap<SlotKey, Vec<String>> = BTreeMap::new();
        for claim in &claims {
            base_values
                .entry(claim.key())
                .or_insert_with(|| self.slot_base_values(&claim.subject, &claim.predicate));
        }
        let options = BatchOptions {
            functional_predicates: functional,
            base_values,
            mode,
        };
        let plan = batch_conflict::plan_batch(claims, &options);

        // 传闻只进证据区：不写会话事实，也不参与冲突提问。
        // 原子模式有未决冲突时连证据也暂缓，保证「裁决前零写入」。
        let defer_evidence = plan.mode == BatchMode::Atomic && !plan.questions.is_empty();
        let evidence_notes = if defer_evidence {
            Vec::new()
        } else {
            self.apply_batch_evidence(&plan.claims, &plan.evidence)
        };

        let commit = self.commit_batch_claims(&label, &plan.claims, &plan.accepted);
        let mut output = String::new();
        output.push_str(&format!(
            "批次记住：解析 {} 条候选，提交 {} 条（新增 {} 条），隔离 {} 条，待确认 {} 处。\n",
            plan.claims.len(),
            commit.handled.len(),
            commit.added.len(),
            plan.isolated.len(),
            plan.questions.len()
        ));
        for &index in &commit.handled {
            let claim = &plan.claims[index];
            let added = commit.added.contains(&index);
            output.push_str(&format!(
                "  {} {}（{}；{}）\n",
                if added { "✓" } else { "=" },
                claim.render_zh(),
                claim.source,
                if added { "新增" } else { "已存在" }
            ));
        }
        for isolation in &plan.isolated {
            let claim = &plan.claims[isolation.claim];
            output.push_str(&format!(
                "  ! {}（{}）：{}\n",
                claim.render_zh(),
                claim.source,
                isolation.reason
            ));
        }
        for note in &evidence_notes {
            output.push_str(&format!("  · {}\n", note.trim_end_matches('\n')));
        }
        for reason in &commit.refused {
            output.push_str(&format!("  跳过：{}\n", reason.trim_end_matches('\n')));
        }
        if !plan.questions.is_empty() {
            let pending = BatchPendingTurn {
                label: label.clone(),
                mode: plan.mode,
                claims: plan.claims.clone(),
                questions: plan.questions.clone(),
                committed: commit.handled.clone(),
                deferred: plan.deferred.clone(),
                evidence: if defer_evidence {
                    plan.evidence.clone()
                } else {
                    Vec::new()
                },
            };
            output.push_str(&pending.render_zh());
            output.push('\n');
            self.batch_pending = Some(pending);
        }
        if let Some(errors) = render_batch_errors_opt(&outcome.errors) {
            output.push_str(&errors);
        }
        if let Some(receipt) = &commit.receipt_id {
            output.push_str(&format!(
                "修订回执：{}（可用「撤销修订」撤回本批提交）。\n",
                receipt
            ));
        }
        output.push_str(&format!("会话事实共 {} 条。", self.session_facts.len()));
        output
    }

    /// 回答待裁决批次：逐项选择、全部同意、都按后者、取消。
    pub fn resolve_batch_answer(&mut self, answer: &str) -> String {
        let Some(pending) = self.batch_pending.take() else {
            return "当前没有待裁决批次。".to_string();
        };
        let parsed = batch_conflict::parse_batch_answer(answer, &pending.questions);
        if parsed.cancelled {
            return format!(
                "已取消待裁决批次「{}」；此前已提交的无冲突事实保留。",
                pending.label
            );
        }
        let mut selected: Vec<usize> = Vec::new();
        let mut remaining_questions: Vec<BatchQuestion> = Vec::new();
        for (index, question) in pending.questions.iter().enumerate() {
            match parsed.selections.get(index).copied().flatten() {
                Some(option_index) => {
                    if let Some(option) = question.options.get(option_index) {
                        selected.extend(option.claims.iter().copied());
                    }
                }
                None => remaining_questions.push(question.clone()),
            }
        }
        let all_resolved = remaining_questions.is_empty();
        let conflict_selected: BTreeSet<usize> = selected.iter().copied().collect();
        if all_resolved {
            selected.extend(pending.deferred.iter().copied());
        }
        selected.sort_unstable();
        selected.dedup();
        let committed_set: BTreeSet<usize> = pending.committed.iter().copied().collect();
        let mut to_commit: Vec<usize> = selected
            .into_iter()
            .filter(|index| !committed_set.contains(index))
            .collect();
        if pending.mode == BatchMode::Atomic && !all_resolved {
            to_commit.clear();
        }
        // 只有用户在冲突表上显式选择的 claim 升级为「更正」；
        // 原子模式暂缓的无冲突 claim 保持原强度，避免误压制非功能槽的旧值。
        let mut resolved_claims = pending.claims.clone();
        for &index in &to_commit {
            if !conflict_selected.contains(&index) {
                continue;
            }
            if let Some(claim) = resolved_claims.get_mut(index) {
                claim.strength = AssertionStrength::Correction;
            }
        }
        let commit = self.commit_batch_claims(&pending.label, &resolved_claims, &to_commit);
        let evidence_notes = if all_resolved {
            self.apply_batch_evidence(&pending.claims, &pending.evidence)
        } else {
            Vec::new()
        };
        let mut output = String::new();
        if !parsed.notes.is_empty() {
            for note in &parsed.notes {
                output.push_str(&format!("裁决提示：{}\n", note));
            }
        }
        if !commit.handled.is_empty() {
            output.push_str(&format!(
                "本批裁决提交 {} 条（新增 {} 条）。\n",
                commit.handled.len(),
                commit.added.len()
            ));
            for &index in &commit.handled {
                if let Some(claim) = pending.claims.get(index) {
                    output.push_str(&format!(
                        "  ✓ {}（{}）\n",
                        claim.render_zh(),
                        claim.source
                    ));
                }
            }
        }
        for note in &evidence_notes {
            output.push_str(&format!("  · {}\n", note.trim_end_matches('\n')));
        }
        for reason in &commit.refused {
            output.push_str(&format!("  跳过：{}\n", reason.trim_end_matches('\n')));
        }
        if let Some(receipt) = &commit.receipt_id {
            output.push_str(&format!(
                "修订回执：{}（可用「撤销修订」撤回本次裁决）。\n",
                receipt
            ));
        }
        if !remaining_questions.is_empty() {
            let mut committed = pending.committed.clone();
            committed.extend(commit.handled.iter().copied());
            committed.sort_unstable();
            committed.dedup();
            let next = BatchPendingTurn {
                label: pending.label.clone(),
                mode: pending.mode,
                claims: pending.claims.clone(),
                questions: remaining_questions,
                committed,
                deferred: pending.deferred.clone(),
                evidence: pending.evidence.clone(),
            };
            output.push_str(&next.render_zh());
            output.push('\n');
            self.batch_pending = Some(next);
        } else if commit.refused.is_empty() {
            output.push_str("本批冲突已全部裁决。\n");
        } else {
            output.push_str("本批冲突已裁决，但写入被修订守门器阻止；请按提示修正后重试。\n");
        }
        output.push_str(&format!("会话事实共 {} 条。", self.session_facts.len()));
        output
    }

    /// 一次预检、一次提交、一次回执地写入一批 claim。
    fn commit_batch_claims(
        &mut self,
        label: &str,
        claims: &[BatchClaim],
        indices: &[usize],
    ) -> BatchCommitOutcome {
        let mut outcome = BatchCommitOutcome::default();
        let mut seen = BTreeSet::new();
        let selected: Vec<usize> = indices
            .iter()
            .copied()
            .filter(|index| seen.insert(*index))
            .collect();
        if selected.is_empty() {
            return outcome;
        }
        let mut changeset = ChangeSet::new(label.to_string());
        for &index in &selected {
            let Some(claim) = claims.get(index) else {
                continue;
            };
            changeset.push(ChangeOp::AssertFact(Atom::rel(
                &claim.predicate,
                &claim.subject,
                &claim.object,
            )));
        }
        let plan = self.revision_guard().plan(changeset);
        if plan.is_blocked() {
            outcome.refused.push(self.render_guard_block(&plan));
            return outcome;
        }
        let before_facts = self.session_facts.clone();
        for &index in &selected {
            let Some(claim) = claims.get(index) else {
                continue;
            };
            let atom = Atom::rel(&claim.predicate, &claim.subject, &claim.object);
            if let Some(position) = self
                .session_negative_facts
                .iter()
                .position(|negative| negative == &atom)
            {
                self.session_negative_facts.remove(position);
                self.invalidate_graph_session();
            }
            self.register_atom_vocabulary(&atom);
            match self.apply_session_slot(
                &claim.subject,
                &claim.predicate,
                &claim.object,
                claim.strength,
                "批量记住",
            ) {
                Ok(transition) if transition.should_confirm() => {
                    outcome.refused.push(transition.render_zh());
                    continue;
                }
                Ok(_) => {
                    self.sync_slot_projection(&claim.subject, &claim.predicate);
                }
                Err(error) => {
                    outcome
                        .refused
                        .push(format!("「{}」未写入：{}", claim.render_zh(), error));
                    continue;
                }
            }
            let existed = self.session_facts.contains(&atom);
            if self.adopt_atom(atom) && !existed {
                outcome.added.push(index);
            }
            outcome.handled.push(index);
        }
        let mut ops: Vec<ChangeOp> = Vec::new();
        for atom in &before_facts {
            if !self.session_facts.contains(atom) {
                ops.push(ChangeOp::RetractFact(atom.clone()));
            }
        }
        for atom in &self.session_facts {
            if !before_facts.contains(atom) {
                ops.push(ChangeOp::AssertFact(atom.clone()));
            }
        }
        if !ops.is_empty() {
            self.record_session_revision(label.to_string(), ops, Vec::new(), Vec::new());
            outcome.receipt_id = self
                .session_receipts
                .last()
                .map(|receipt| receipt.id.clone());
        }
        outcome
    }
}

#[derive(Debug, Default)]
struct BatchCommitOutcome {
    handled: Vec<usize>,
    added: Vec<usize>,
    refused: Vec<String>,
    receipt_id: Option<String>,
}

fn render_batch_errors(errors: &[String]) -> String {
    render_batch_errors_opt(errors).unwrap_or_default()
}

fn render_batch_errors_opt(errors: &[String]) -> Option<String> {
    if errors.is_empty() {
        return None;
    }
    let mut output = String::from("未解析：\n");
    for error in errors {
        output.push_str(&format!("  · {}\n", error));
    }
    Some(output)
}

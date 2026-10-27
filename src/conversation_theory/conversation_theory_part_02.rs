impl ConversationTheory {
    pub fn build(chronicle: &Chronicle) -> Self {
        Self::build_with_config(chronicle, TheoryConfig::default())
    }

    pub fn build_with_config(chronicle: &Chronicle, config: TheoryConfig) -> Self {
        let total_entries = chronicle.len();
        let all_entries = chronicle.entries();
        let start = all_entries.len().saturating_sub(config.max_events.max(1));
        let entries = &all_entries[start..];
        let window_start_seq = entries.first().map(|entry| entry.seq).unwrap_or(0);
        let window_end_seq = entries.last().map(|entry| entry.seq).unwrap_or(0);
        let initial_revision = entries
            .first()
            .map(|entry| entry.kb_revision)
            .unwrap_or_else(|| chronicle.kb_revision());

        let mut events: Vec<TheoryEvent> = Vec::with_capacity(entries.len());
        let mut situations = Vec::with_capacity(entries.len() + 1);
        situations.push(Situation {
            id: "s0".to_string(),
            seq: 0,
            revision: initial_revision,
            focus: None,
            question: None,
            answer: None,
            kind: EventKind::Other,
        });
        let mut answers: Vec<AnswerRecord> = Vec::new();
        let mut pending: Option<PendingTurn> = None;
        let mut belief_state = BeliefState::default();
        let mut snapshots: BTreeMap<u64, Vec<BeliefFact>> = BTreeMap::new();

        for entry in entries {
            let event = TheoryEvent::from_entry(entry);
            belief_state.apply(&event);
            let mut situation = Situation {
                id: format!("s{}", entry.seq),
                seq: entry.seq,
                revision: entry.kb_revision,
                focus: event.focus.clone(),
                question: None,
                answer: None,
                kind: event.kind,
            };
            match event.kind {
                EventKind::Turn => {
                    let mode = entry
                        .payload
                        .get("mode")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    let user = entry
                        .payload
                        .get("user")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .trim();
                    if mode == "query" && !user.is_empty() {
                        pending = Some(PendingTurn {
                            seq: entry.seq,
                            question: user.to_string(),
                            focus: event.focus.clone(),
                        });
                        situation.question = Some(user.to_string());
                        situation.focus = event.focus.clone();
                    }
                }
                EventKind::Reply => {
                    let answer = entry
                        .payload
                        .get("answer")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .trim()
                        .to_string();
                    let payload_question = entry
                        .payload
                        .get("question")
                        .and_then(Value::as_str)
                        .map(str::trim)
                        .filter(|question| !question.is_empty())
                        .map(str::to_string);
                    let turn = pending.take();
                    let question = payload_question
                        .or_else(|| turn.as_ref().map(|turn| turn.question.clone()));
                    if let Some(question) = question {
                        let key = entry
                            .payload
                            .get("question_key")
                            .and_then(Value::as_str)
                            .map(str::to_string)
                            .unwrap_or_else(|| question_key(&question));
                        let question_seq = turn.as_ref().map(|turn| turn.seq).unwrap_or(entry.seq);
                        let focus = turn
                            .as_ref()
                            .and_then(|turn| turn.focus.clone())
                            .or_else(|| event.focus.clone());
                        answers.push(AnswerRecord {
                            question: question.clone(),
                            question_key: key,
                            question_seq,
                            reply_seq: entry.seq,
                            answer: answer.clone(),
                            answer_key: answer_key(&answer),
                            revision: entry.kb_revision,
                            situation: situation.id.clone(),
                            focus: focus.clone(),
                            proof_id: entry
                                .payload
                                .get("proof_id")
                                .and_then(Value::as_str)
                                .map(str::to_string),
                            observed_seq: entry
                                .payload
                                .get("observed_seq")
                                .and_then(Value::as_u64)
                                .unwrap_or(question_seq),
                            read_revision: entry
                                .payload
                                .get("read_revision")
                                .and_then(Value::as_u64)
                                .unwrap_or(entry.kb_revision),
                            commit_revision: entry
                                .payload
                                .get("commit_revision")
                                .and_then(Value::as_u64)
                                .unwrap_or(entry.kb_revision),
                        });
                        situation.question = Some(question);
                        situation.focus = focus;
                    }
                    if !answer.is_empty() {
                        situation.answer = Some(answer);
                    }
                    snapshots.insert(entry.seq, belief_state.snapshot());
                }
                _ => {}
            }
            events.push(event);
            situations.push(situation);
        }

        let changes = compute_changes(&answers);
        let (program, reverse, evaluation_error, truncated_belief_facts) = compile_program(
            &events,
            &answers,
            &changes,
            &snapshots,
            config.max_belief_facts,
        );

        Self {
            events,
            situations,
            answers,
            changes,
            snapshots,
            program,
            reverse,
            evaluation_error,
            window_start_seq,
            window_end_seq,
            total_entries,
            truncated_belief_facts,
        }
    }

    pub fn program(&self) -> &LogicProgram {
        &self.program
    }

    pub fn evaluation_error(&self) -> Option<&str> {
        self.evaluation_error.as_deref()
    }

    pub fn events(&self) -> &[TheoryEvent] {
        &self.events
    }

    pub fn situations(&self) -> &[Situation] {
        &self.situations
    }

    pub fn answers(&self) -> &[AnswerRecord] {
        &self.answers
    }

    pub fn changes(&self) -> &[AnswerChange] {
        &self.changes
    }

    pub fn window(&self) -> (u64, u64) {
        (self.window_start_seq, self.window_end_seq)
    }

    pub fn total_entries(&self) -> usize {
        self.total_entries
    }

    pub fn truncated_belief_facts(&self) -> bool {
        self.truncated_belief_facts
    }

    /// 解析内部常量（`n_姚明` 等）回人类可读文本；未知常量原样返回。
    pub fn resolve(&self, constant: &str) -> String {
        self.reverse
            .get(constant)
            .cloned()
            .unwrap_or_else(|| constant.to_string())
    }

    pub fn matching_answers(&self, query: &str) -> Vec<AnswerRecord> {
        let query = query.trim();
        if query.is_empty() {
            return self.answers.clone();
        }
        let key = question_key(query);
        let exact: Vec<AnswerRecord> = self
            .answers
            .iter()
            .filter(|answer| answer.question_key == key)
            .cloned()
            .collect();
        if !exact.is_empty() {
            return exact;
        }
        let needle = query.to_lowercase();
        self.answers
            .iter()
            .filter(|answer| {
                answer.question.to_lowercase().contains(&needle)
                    || answer.question_key.contains(&needle)
            })
            .cloned()
            .collect()
    }

    /// 某情景（或情景之前）的可枚举信念。不是整库快照，只折叠语义事件效果。
    pub fn beliefs_at(&self, seq: u64) -> Vec<BeliefFact> {
        if let Some(snapshot) = self.snapshots.get(&seq) {
            return snapshot.clone();
        }
        let mut state = BeliefState::default();
        for event in self.events.iter().filter(|event| event.seq <= seq) {
            state.apply(event);
        }
        state.snapshot()
    }

    pub fn latest_change(&self, query: &str) -> Option<AnswerChange> {
        let matched = self.matching_answers(query);
        let keys: BTreeSet<&str> = matched
            .iter()
            .map(|answer| answer.question_key.as_str())
            .collect();
        self.changes
            .iter()
            .rev()
            .find(|change| {
                keys.contains(change.question_key.as_str())
                    || change.question.contains(query.trim())
            })
            .cloned()
    }

    /// 区间归因：只提名「触碰了问题/回答词面或当轮焦点」的语义事件。
    pub fn causes(&self, query: &str) -> CauseReport {
        let Some(change) = self.latest_change(query) else {
            return CauseReport {
                change: None,
                candidates: Vec::new(),
                ignored_events: 0,
                non_enumerable_events: 0,
                note: if self.matching_answers(query).is_empty() {
                    format!("编年史里没有「{}」的回答，无法归因。", query.trim())
                } else {
                    format!("「{}」没有改口记录：相邻回答一致。", query.trim())
                },
            };
        };
        let report = self.causes_for_change(&change);
        report
    }

    pub fn causes_for_change(&self, change: &AnswerChange) -> CauseReport {
        let mut haystack = format!(
            "{} {} {}",
            change.question, change.previous.answer, change.current.answer
        )
        .to_lowercase();
        if let Some(focus) = change
            .current
            .focus
            .as_deref()
            .or(change.previous.focus.as_deref())
        {
            haystack.push(' ');
            haystack.push_str(&focus.to_lowercase());
        }

        let mut candidates = Vec::new();
        let mut ignored_events = 0usize;
        let mut non_enumerable_events = 0usize;
        for event in &self.events {
            if !event.kind.is_causal_candidate() {
                continue;
            }
            if event.seq <= change.previous.reply_seq || event.seq >= change.current.reply_seq {
                continue;
            }
            if event.effects.is_empty() {
                non_enumerable_events += 1;
                if event.kind == EventKind::Rule {
                    candidates.push(CandidateCause {
                        event: event.clone(),
                        score: 1,
                        reasons: vec!["区间内的规则变更（影响推导，效果不可完全枚举）".to_string()],
                    });
                }
                continue;
            }
            let mut score = 0u32;
            let mut reasons = Vec::new();
            for effect in &event.effects {
                let entity = effect.entity.to_lowercase();
                let predicate = effect.predicate.to_lowercase();
                let value = effect.value.to_lowercase();
                if !entity.is_empty() && haystack.contains(&entity) {
                    score += 2;
                    reasons.push(format!("触及实体「{}」", effect.entity));
                }
                if !predicate.is_empty() && haystack.contains(&predicate) {
                    score += 1;
                    reasons.push(format!("触及属性「{}」", effect.predicate));
                }
                if !value.is_empty()
                    && (change.current.answer.contains(&effect.value)
                        || change.previous.answer.contains(&effect.value))
                {
                    score += 4;
                    reasons.push(format!("改变回答中出现的取值「{}」", effect.value));
                }
            }
            if event.kind == EventKind::Rule && score == 0 {
                score = 1;
                reasons.push("区间内的规则变更（影响推导）".to_string());
            }
            reasons.sort();
            reasons.dedup();
            if score == 0 {
                ignored_events += 1;
                continue;
            }
            candidates.push(CandidateCause {
                event: event.clone(),
                score,
                reasons,
            });
        }
        candidates.sort_by(|left, right| {
            right
                .score
                .cmp(&left.score)
                .then(right.event.seq.cmp(&left.event.seq))
        });
        let note = if candidates.is_empty() {
            if non_enumerable_events > 0 {
                "没有可核对词面的候选原因；区间内有不可枚举事件，无法进一步归因。".to_string()
            } else {
                "无法归因：区间内没有触碰该问题/回答词面的语义事件。".to_string()
            }
        } else if non_enumerable_events > 0 {
            "候选只覆盖可枚举效果；区间内还有不可枚举事件，结论不声称穷尽原因。".to_string()
        } else {
            "候选按词面命中和取值变化打分；这是归因提示，不是因果证明。".to_string()
        };
        CauseReport {
            change: Some(change.clone()),
            candidates,
            ignored_events,
            non_enumerable_events,
            note,
        }
    }

    /// 有界反事实：撤销最近一条可枚举语义事件，看当前回答是否失去支持。
    pub fn counterfactual_undo_latest(&self, query: &str) -> CounterfactualReport {
        let matched = self.matching_answers(query);
        let Some(answer) = matched.last().cloned() else {
            return CounterfactualReport {
                question: query.to_string(),
                answer: None,
                target_event: None,
                verdict: CounterfactualVerdict::Unknown,
                affected: Vec::new(),
                remaining_support: 0,
                reasons: vec!["编年史里没有这条问题的回答。".to_string()],
            };
        };
        let event = self
            .events
            .iter()
            .rev()
            .find(|event| event.seq < answer.reply_seq && event.kind.is_semantic())
            .cloned();
        match event {
            Some(event) => self.counterfactual_undo_event(&event, &answer),
            None => CounterfactualReport {
                question: answer.question.clone(),
                answer: Some(answer),
                target_event: None,
                verdict: CounterfactualVerdict::Unknown,
                affected: Vec::new(),
                remaining_support: 0,
                reasons: vec!["该回答之前没有语义事件可撤销。".to_string()],
            },
        }
    }

    pub fn counterfactual_undo_seq(&self, seq: u64, query: &str) -> CounterfactualReport {
        let matched = self.matching_answers(query);
        let Some(answer) = matched.last().cloned() else {
            return CounterfactualReport {
                question: query.to_string(),
                answer: None,
                target_event: None,
                verdict: CounterfactualVerdict::Unknown,
                affected: Vec::new(),
                remaining_support: 0,
                reasons: vec!["编年史里没有这条问题的回答。".to_string()],
            };
        };
        match self.events.iter().find(|event| event.seq == seq).cloned() {
            Some(event) => self.counterfactual_undo_event(&event, &answer),
            None => CounterfactualReport {
                question: answer.question.clone(),
                answer: Some(answer),
                target_event: None,
                verdict: CounterfactualVerdict::Unknown,
                affected: Vec::new(),
                remaining_support: 0,
                reasons: vec![format!("窗口内没有 seq {} 的事件。", seq)],
            },
        }
    }

    fn counterfactual_undo_event(
        &self,
        event: &TheoryEvent,
        answer: &AnswerRecord,
    ) -> CounterfactualReport {
        if event.seq >= answer.reply_seq {
            return CounterfactualReport {
                question: answer.question.clone(),
                answer: Some(answer.clone()),
                target_event: Some(event.clone()),
                verdict: CounterfactualVerdict::Unchanged,
                affected: Vec::new(),
                remaining_support: 0,
                reasons: vec!["事件发生在该回答之后，撤销它不会改变历史回答。".to_string()],
            };
        }
        if !event.kind.is_semantic() {
            return CounterfactualReport {
                question: answer.question.clone(),
                answer: Some(answer.clone()),
                target_event: Some(event.clone()),
                verdict: CounterfactualVerdict::Unknown,
                affected: Vec::new(),
                remaining_support: 0,
                reasons: vec!["这不是语义事件（只是对话输入/回答），没有可重放效果。".to_string()],
            };
        }
        if !event.enumerable || event.effects.is_empty() {
            return CounterfactualReport {
                question: answer.question.clone(),
                answer: Some(answer.clone()),
                target_event: Some(event.clone()),
                verdict: CounterfactualVerdict::Unknown,
                affected: Vec::new(),
                remaining_support: 0,
                reasons: vec![
                    "该事件的效果不可完全枚举（规则/外链/批量导入等），拒绝给出反事实结论。"
                        .to_string(),
                ],
            };
        }

        let haystack = format!("{} {}", answer.question, answer.answer).to_lowercase();
        let focus = answer.focus.clone().unwrap_or_default().to_lowercase();
        let affected_effects: Vec<TheoryEffect> = event
            .effects
            .iter()
            .filter(|effect| {
                let entity = effect.entity.to_lowercase();
                let predicate = effect.predicate.to_lowercase();
                let value = effect.value.to_lowercase();
                let value_hit = !value.is_empty() && haystack.contains(&value);
                let entity_hit = !entity.is_empty()
                    && (haystack.contains(&entity)
                        || (!focus.is_empty() && focus.contains(&entity)));
                let predicate_hit = !predicate.is_empty() && haystack.contains(&predicate);
                value_hit || (entity_hit && predicate_hit) || (entity_hit && value.is_empty())
            })
            .cloned()
            .collect();

        if affected_effects.is_empty() {
            return CounterfactualReport {
                question: answer.question.clone(),
                answer: Some(answer.clone()),
                target_event: Some(event.clone()),
                verdict: CounterfactualVerdict::Unchanged,
                affected: Vec::new(),
                remaining_support: 0,
                reasons: vec![
                    "该事件的可枚举效果没有出现在问题、回答或焦点中：按效果判断不会改变这条回答。"
                        .to_string(),
                ],
            };
        }

        let beliefs = self.beliefs_at(answer.reply_seq);
        let mut flips = Vec::new();
        let mut uncertain = Vec::new();
        let mut remaining_support = 0usize;
        for effect in &affected_effects {
            match effect.op {
                EffectOp::Add => {
                    if effect.value.is_empty() {
                        uncertain.push(effect.clone());
                        continue;
                    }
                    if !answer.answer.contains(&effect.value) {
                        uncertain.push(effect.clone());
                        continue;
                    }
                    let alternative = beliefs.iter().any(|fact| {
                        fact.entity == effect.entity
                            && fact.predicate == effect.predicate
                            && fact.value == effect.value
                            && fact.source_seq != event.seq
                    });
                    if alternative {
                        remaining_support += 1;
                    } else {
                        flips.push(effect.clone());
                    }
                }
                EffectOp::Remove => {
                    if !effect.value.is_empty() && answer.answer.contains(&effect.value) {
                        remaining_support += 1;
                        continue;
                    }
                    let earlier_has_value = self
                        .answers
                        .iter()
                        .filter(|record| {
                            record.question_key == answer.question_key
                                && record.reply_seq < answer.reply_seq
                        })
                        .any(|record| record.answer.contains(&effect.value));
                    if earlier_has_value {
                        flips.push(effect.clone());
                    } else {
                        uncertain.push(effect.clone());
                    }
                }
            }
        }

        let (verdict, mut reasons) = if !flips.is_empty() {
            (
                CounterfactualVerdict::Flips,
                vec![format!(
                    "撤销该事件会移除回答所依赖的 {} 条取值来源（无替代来源）。",
                    flips.len()
                )],
            )
        } else if !uncertain.is_empty() {
            (
                CounterfactualVerdict::Unknown,
                vec!["事件触及问题，但回答没有暴露可核对的取值，无法确定翻转方向。".to_string()],
            )
        } else {
            (
                CounterfactualVerdict::Unchanged,
                vec![format!(
                    "相关取值仍有 {} 条替代来源，撤销该事件不会改变这条回答。",
                    remaining_support
                )],
            )
        };
        reasons.push("只按编年史中可枚举的效果推断，没有重放整库查询。".to_string());
        CounterfactualReport {
            question: answer.question.clone(),
            answer: Some(answer.clone()),
            target_event: Some(event.clone()),
            verdict,
            affected: affected_effects,
            remaining_support,
            reasons,
        }
    }

    /// 在编译后的理论里执行 Datalog 查询，返回变量绑定（常量已解析为文本）。
    pub fn query(&self, pattern: &str) -> Result<Vec<BTreeMap<String, String>>, String> {
        let envs = self
            .program
            .query(pattern)
            .map_err(|error| format!("查询解析失败：{}", error))?;
        Ok(envs
            .into_iter()
            .map(|env| {
                env.into_iter()
                    .map(|(name, value)| (name, self.resolve(&value)))
                    .collect()
            })
            .collect())
    }

    pub fn render_report(&self) -> String {
        let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
        for event in &self.events {
            *counts.entry(event.kind.as_str()).or_default() += 1;
        }
        let count_text = counts
            .iter()
            .map(|(kind, count)| format!("{} {}", kind, count))
            .collect::<Vec<_>>()
            .join("、");
        let mut lines = vec![format!(
            "对话情景理论（CST）：窗口 seq {}..{}（编译 {} 条，日志共 {} 条）；情景 {} 个，问答 {} 条，改口 {} 处。",
            self.window_start_seq,
            self.window_end_seq,
            self.events.len(),
            self.total_entries,
            self.situations.len(),
            self.answers.len(),
            self.changes.len()
        )];
        lines.push(format!(
            "语义事件：{}；理论事实 {} 条，规则 {} 条。",
            if count_text.is_empty() {
                "无"
            } else {
                &count_text
            },
            self.program.fact_count(),
            self.program.rule_count()
        ));
        if let Some(error) = &self.evaluation_error {
            lines.push(format!("理论求值警告：{}", error));
        }
        lines.push(
            "可查询谓词：situation/1、next_situation/2、before/2、event_kind/2、revision_at/2、focus_at/2、asked_at/3、answered_at/4、answer_change/3、change_at/4、touches/3、belief_at/5、candidate_cause/2。"
                .to_string(),
        );
        lines.push("最近情景（新 → 旧，最多 8 条）：".to_string());
        for situation in self.situations.iter().rev().filter(|s| s.seq != 0).take(8) {
            let focus = situation
                .focus
                .as_deref()
                .map(|focus| format!("焦点 {}；", focus))
                .unwrap_or_default();
            let question = situation
                .question
                .as_deref()
                .map(|question| format!("问 {}；", truncate(question, 32)))
                .unwrap_or_default();
            let answer = situation
                .answer
                .as_deref()
                .map(|answer| format!("答 {}；", truncate(answer, 40)))
                .unwrap_or_default();
            let event_text = self
                .events
                .iter()
                .find(|event| event.seq == situation.seq)
                .filter(|event| !matches!(event.kind, EventKind::Turn | EventKind::Reply))
                .map(|event| format!("{}；", truncate(&event.label, 56)))
                .unwrap_or_default();
            lines.push(format!(
                "- {}（seq {}，rev {}，{}）：{}{}{}{}",
                situation.id,
                situation.seq,
                situation.revision,
                situation.kind.as_str(),
                focus,
                question,
                answer,
                event_text
            ));
        }
        if self.truncated_belief_facts {
            lines.push(
                "注意：belief_at 事实超过预算，已截断；beliefs_at 接口仍可完整折叠。".to_string(),
            );
        }
        lines.join("\n")
    }

    pub fn render_situation(&self, seq: u64) -> String {
        let Some(event) = self.events.iter().find(|event| event.seq == seq) else {
            return format!("窗口内没有 seq {} 的情景。", seq);
        };
        let Some(situation) = self
            .situations
            .iter()
            .find(|situation| situation.seq == seq)
        else {
            return format!("窗口内没有 seq {} 的情景。", seq);
        };
        let mut lines = vec![format!(
            "情景 {}（seq {}，{}，rev {}）",
            situation.id,
            situation.seq,
            event.kind.as_str(),
            situation.revision
        )];
        if let Some(focus) = &situation.focus {
            lines.push(format!("焦点：{}", focus));
        }
        if let Some(question) = &situation.question {
            lines.push(format!("问题：{}", question));
        }
        if let Some(answer) = &situation.answer {
            lines.push(format!("回答：{}", answer));
        }
        lines.push(format!("事件：{}", event.label));
        if event.effects.is_empty() {
            lines.push("可枚举效果：无".to_string());
        } else {
            lines.push(format!(
                "可枚举效果（{} 条{}）：",
                event.effect_count,
                if event.enumerable {
                    ""
                } else {
                    "，未完全枚举"
                }
            ));
            for effect in event.effects.iter().take(12) {
                lines.push(format!(
                    "  - {} {}.{} = {}",
                    effect.op.as_str(),
                    effect.entity,
                    effect.predicate,
                    if effect.value.is_empty() {
                        "（空）"
                    } else {
                        &effect.value
                    }
                ));
            }
            if event.effects.len() > 12 {
                lines.push(format!("  …其余 {} 条见导入日志", event.effects.len() - 12));
            }
        }
        if !event.corrections.is_empty() {
            lines.push(format!("更正（{} 条）：", event.corrections.len()));
            for correction in event.corrections.iter().take(8) {
                lines.push(format!(
                    "  - {}.{}：「{}」→「{}」",
                    correction.entity, correction.predicate, correction.from, correction.to
                ));
            }
        }
        if !event.conflicts.is_empty() {
            lines.push(format!("未裁决冲突（{} 条）：", event.conflicts.len()));
            for conflict in event.conflicts.iter().take(8) {
                lines.push(format!(
                    "  - {}.{}：已有「{}」，新值「{}」",
                    conflict.entity,
                    conflict.predicate,
                    conflict.existing.join("、"),
                    conflict.incoming
                ));
            }
        }
        if event.kind == EventKind::Reply {
            let beliefs = self.beliefs_at(seq);
            lines.push(format!(
                "该情景的可枚举信念：{} 条（不是整库快照）。",
                beliefs.len()
            ));
            for belief in beliefs.iter().take(8) {
                lines.push(format!(
                    "  - {}.{} = {}（来源 e{}）",
                    belief.entity, belief.predicate, belief.value, belief.source_seq
                ));
            }
        }
        lines.join("\n")
    }

    pub fn render_history(&self, query: &str) -> String {
        let records = self.matching_answers(query);
        if records.is_empty() {
            return format!("编年史里没有与「{}」匹配的已回答问题。", query.trim());
        }
        let mut lines = vec![format!(
            "情景历史「{}」：{} 次回答（按时间顺序）",
            query.trim(),
            records.len()
        )];
        for record in &records {
            let proof = record
                .proof_id
                .as_deref()
                .map(|id| format!("；证明 {}", id))
                .unwrap_or_default();
            let focus = record
                .focus
                .as_deref()
                .map(|focus| format!("；焦点 {}", focus))
                .unwrap_or_default();
            lines.push(format!(
                "- 第 {} 条（{}, rev {}）：问「{}」→ 答「{}」{}",
                record.reply_seq,
                record.situation,
                record.revision,
                truncate(&record.question, 36),
                truncate(&record.answer, 60),
                format!("{}{}", focus, proof)
            ));
        }
        let changes = self
            .changes
            .iter()
            .filter(|change| {
                records
                    .iter()
                    .any(|record| record.question_key == change.question_key)
            })
            .count();
        lines.push(format!(
            "同一问题改口 {} 处；用「情景归因：{}」查看候选原因。",
            changes,
            query.trim()
        ));
        lines.join("\n")
    }

    pub fn render_changes(&self) -> String {
        if self.changes.is_empty() {
            return "情景理论里没有改口记录（同一问题相邻回答一致）。".to_string();
        }
        let mut lines = vec![format!(
            "情景改口：{} 处（列出最近 12 处，附最多 3 个候选原因）",
            self.changes.len()
        )];
        for change in self.changes.iter().rev().take(12) {
            lines.push(format!(
                "- 「{}」：{}（rev {}，seq {}）→ {}（rev {}，seq {}）；{}",
                truncate(&change.question, 36),
                truncate(&change.previous.answer, 42),
                change.previous.revision,
                change.previous.reply_seq,
                truncate(&change.current.answer, 42),
                change.current.revision,
                change.current.reply_seq,
                change.kind.label()
            ));
            let report = self.causes_for_change(change);
            for candidate in report.candidates.iter().take(3) {
                lines.push(format!(
                    "    ↳ e{} {}（score {}）：{}",
                    candidate.event.seq,
                    candidate.event.kind.as_str(),
                    candidate.score,
                    candidate.reasons.join("；")
                ));
            }
            if report.candidates.is_empty() {
                lines.push(format!("    ↳ {}", report.note));
            }
        }
        lines.join("\n")
    }

    pub fn render_causes(&self, query: &str) -> String {
        let report = self.causes(query);
        let Some(change) = &report.change else {
            return report.note;
        };
        let mut lines = vec![format!(
            "情景归因「{}」：{}（rev {}）→ {}（rev {}）；{}",
            truncate(&change.question, 40),
            truncate(&change.previous.answer, 48),
            change.previous.revision,
            truncate(&change.current.answer, 48),
            change.current.revision,
            change.kind.label()
        )];
        if report.candidates.is_empty() {
            lines.push(report.note.clone());
        } else {
            lines.push("候选原因（按词面命中与取值变化排序）：".to_string());
            for candidate in &report.candidates {
                lines.push(format!(
                    "- e{}（seq {}，{}）：score {}；{}",
                    candidate.event.seq,
                    candidate.event.seq,
                    candidate.event.kind.as_str(),
                    candidate.score,
                    candidate.reasons.join("；")
                ));
                lines.push(format!(
                    "    事件：{}",
                    truncate(&candidate.event.label, 72)
                ));
            }
            lines.push(report.note.clone());
        }
        if report.ignored_events > 0 || report.non_enumerable_events > 0 {
            lines.push(format!(
                "区间内另有 {} 条事件未命中词面、{} 条效果不可枚举。",
                report.ignored_events, report.non_enumerable_events
            ));
        }
        lines.join("\n")
    }

    pub fn render_counterfactual(&self, query: &str) -> String {
        let report = self.counterfactual_undo_latest(query);
        self.render_counterfactual_report(&report)
    }

    pub fn render_counterfactual_report(&self, report: &CounterfactualReport) -> String {
        let mut lines = vec![format!(
            "情景反事实「{}」：{}",
            truncate(&report.question, 40),
            report.verdict.label()
        )];
        if let Some(answer) = &report.answer {
            lines.push(format!(
                "当前回答（第 {} 条，rev {}）：{}",
                answer.reply_seq,
                answer.revision,
                truncate(&answer.answer, 72)
            ));
        }
        if let Some(event) = &report.target_event {
            lines.push(format!(
                "撤销目标：e{}（{}）{}",
                event.seq,
                event.kind.as_str(),
                truncate(&event.label, 72)
            ));
        }
        if !report.affected.is_empty() {
            lines.push("相关可枚举效果：".to_string());
            for effect in report.affected.iter().take(8) {
                lines.push(format!(
                    "  - {} {}.{} = {}",
                    effect.op.as_str(),
                    effect.entity,
                    effect.predicate,
                    if effect.value.is_empty() {
                        "（空）"
                    } else {
                        &effect.value
                    }
                ));
            }
        }
        for reason in &report.reasons {
            lines.push(format!("· {}", reason));
        }
        lines.join("\n")
    }

    pub fn render_query(&self, pattern: &str) -> String {
        match self.query(pattern) {
            Ok(envs) => {
                if envs.is_empty() {
                    return format!("情景理论查询「{}」：无解。", pattern.trim());
                }
                let mut lines = vec![format!(
                    "情景理论查询「{}」：{} 个解",
                    pattern.trim(),
                    envs.len()
                )];
                for env in envs.iter().take(20) {
                    if env.is_empty() {
                        lines.push("  - true".to_string());
                    } else {
                        let binding = env
                            .iter()
                            .map(|(name, value)| format!("{} = {}", name, value))
                            .collect::<Vec<_>>()
                            .join("；");
                        lines.push(format!("  - {}", binding));
                    }
                }
                if envs.len() > 20 {
                    lines.push(format!("  …其余 {} 个解省略", envs.len() - 20));
                }
                lines.join("\n")
            }
            Err(error) => error,
        }
    }
}
fn compute_changes(answers: &[AnswerRecord]) -> Vec<AnswerChange> {
    let mut by_key: BTreeMap<&str, Vec<&AnswerRecord>> = BTreeMap::new();
    for answer in answers {
        if !answer.question_key.is_empty() {
            by_key
                .entry(answer.question_key.as_str())
                .or_default()
                .push(answer);
        }
    }
    let mut changes = Vec::new();
    for records in by_key.values() {
        for pair in records.windows(2) {
            let previous = pair[0];
            let current = pair[1];
            if previous.answer_key == current.answer_key {
                continue;
            }
            let kind = if previous.revision == current.revision {
                ChangeKind::SameRevisionConflict
            } else {
                ChangeKind::RevisionChanged
            };
            changes.push(AnswerChange {
                question: current.question.clone(),
                question_key: current.question_key.clone(),
                previous: previous.clone(),
                current: current.clone(),
                kind,
            });
        }
    }
    changes.sort_by_key(|change| change.current.reply_seq);
    changes
}
#[derive(Debug, Default)]
struct ConstantTable {
    reverse: BTreeMap<String, String>,
}
impl ConstantTable {
    fn intern(&mut self, prefix: &str, text: &str) -> String {
        let text = text.trim();
        let hash = fnv1a_hex(text.as_bytes());
        let sanitized: String = text
            .chars()
            .map(|ch| {
                if ch.is_alphanumeric() || ch == '_' || ch == '-' {
                    ch
                } else {
                    '_'
                }
            })
            .take(32)
            .collect();
        let sanitized = sanitized.trim_matches('_');
        let mut candidate = if sanitized.is_empty() {
            format!("{}_{}", prefix, &hash[..12.min(hash.len())])
        } else if sanitized == text {
            format!("{}_{}", prefix, sanitized)
        } else {
            format!("{}_{}_{}", prefix, sanitized, &hash[..8.min(hash.len())])
        };
        if let Some(existing) = self.reverse.get(&candidate) {
            if existing != text {
                candidate = format!("{}_{}_{}", prefix, sanitized, hash);
            }
        }
        self.reverse.insert(candidate.clone(), text.to_string());
        candidate
    }
}
fn fact(predicate: &str, terms: Vec<String>) -> LAtom {
    LAtom::new(
        predicate,
        terms.into_iter().map(LTerm::atom).collect::<Vec<_>>(),
    )
}
fn compile_program(
    events: &[TheoryEvent],
    answers: &[AnswerRecord],
    changes: &[AnswerChange],
    snapshots: &BTreeMap<u64, Vec<BeliefFact>>,
    max_belief_facts: usize,
) -> (LogicProgram, BTreeMap<String, String>, Option<String>, bool) {
    let mut table = ConstantTable::default();
    let mut program = LogicProgram::new();

    program.add_fact(fact("situation", vec!["s0".to_string()]));
    let mut previous_situation = "s0".to_string();
    for event in events {
        let situation = format!("s{}", event.seq);
        let event_id = format!("e{}", event.seq);
        program.add_fact(fact("situation", vec![situation.clone()]));
        program.add_fact(fact(
            "next_situation",
            vec![previous_situation.clone(), situation.clone()],
        ));
        program.add_fact(fact("event", vec![event_id.clone()]));
        program.add_fact(fact(
            "event_kind",
            vec![event_id.clone(), event.kind.as_str().to_string()],
        ));
        program.add_fact(fact(
            "event_seq",
            vec![event_id.clone(), event.seq.to_string()],
        ));
        program.add_fact(fact(
            "event_after",
            vec![event_id.clone(), situation.clone()],
        ));
        program.add_fact(fact(
            "event_revision",
            vec![event_id.clone(), event.revision.to_string()],
        ));
        program.add_fact(fact(
            "revision_at",
            vec![situation.clone(), event.revision.to_string()],
        ));
        if let Some(focus) = event.focus.as_deref().filter(|focus| !focus.is_empty()) {
            let focus = table.intern("n", focus);
            program.add_fact(fact("focus_at", vec![situation.clone(), focus]));
        }
        for effect in &event.effects {
            let entity = table.intern("n", &effect.entity);
            let predicate = table.intern("n", &effect.predicate);
            let value = table.intern("n", &effect.value);
            program.add_fact(fact(
                "effect",
                vec![
                    event_id.clone(),
                    effect.op.as_str().to_string(),
                    entity.clone(),
                    predicate.clone(),
                    value,
                ],
            ));
        }
        for (entity, predicate) in event.touches() {
            let entity = table.intern("n", &entity);
            let predicate = table.intern("n", &predicate);
            program.add_fact(fact("touches", vec![event_id.clone(), entity, predicate]));
        }
        previous_situation = situation;
    }

    for answer in answers {
        let question = table.intern("q", &answer.question_key);
        let answer_id = format!("a{}", answer.reply_seq);
        let proof = answer
            .proof_id
            .as_deref()
            .filter(|proof| !proof.is_empty())
            .map(|proof| table.intern("p", proof))
            .unwrap_or_else(|| "none".to_string());
        let situation = format!("s{}", answer.reply_seq);
        program.add_fact(fact(
            "asked_at",
            vec![
                format!("s{}", answer.question_seq),
                question.clone(),
                format!("e{}", answer.question_seq),
            ],
        ));
        program.add_fact(fact(
            "answered_at",
            vec![
                situation.clone(),
                question.clone(),
                answer_id.clone(),
                proof,
            ],
        ));
        program.add_fact(fact(
            "answer_at",
            vec![answer_id.clone(), situation.clone()],
        ));
        program.add_fact(fact(
            "answer_text",
            vec![answer_id, table.intern("a", &answer.answer)],
        ));
    }

    let mut by_key: BTreeMap<&str, Vec<&AnswerRecord>> = BTreeMap::new();
    for answer in answers {
        by_key
            .entry(answer.question_key.as_str())
            .or_default()
            .push(answer);
    }
    for (key, records) in &by_key {
        let question = table.intern("q", key);
        for pair in records.windows(2) {
            let first = format!("a{}", pair[0].reply_seq);
            let second = format!("a{}", pair[1].reply_seq);
            program.add_fact(fact(
                "next_answer",
                vec![question.clone(), first.clone(), second.clone()],
            ));
            if pair[0].answer_key != pair[1].answer_key {
                program.add_fact(fact("answer_differ", vec![first, second]));
            }
        }
    }

    for change in changes {
        let question = table.intern("q", &change.question_key);
        let first = format!("a{}", change.previous.reply_seq);
        let second = format!("a{}", change.current.reply_seq);
        program.add_fact(fact("answer_change", vec![question.clone(), first, second]));
        program.add_fact(fact(
            "change_at",
            vec![
                question.clone(),
                format!("s{}", change.previous.reply_seq),
                format!("s{}", change.current.reply_seq),
            ],
        ));
        for event in events {
            if !event.kind.is_causal_candidate() {
                continue;
            }
            if event.seq <= change.previous.reply_seq || event.seq >= change.current.reply_seq {
                continue;
            }
            if event.touches().is_empty() && event.kind != EventKind::Rule {
                continue;
            }
            program.add_fact(fact(
                "candidate_cause",
                vec![format!("e{}", event.seq), question.clone()],
            ));
        }
    }

    let mut belief_facts = 0usize;
    let mut truncated_belief_facts = false;
    'snapshots: for (seq, snapshot) in snapshots {
        let situation = format!("s{}", seq);
        for belief in snapshot {
            if belief_facts >= max_belief_facts {
                truncated_belief_facts = true;
                program.add_fact(fact("theory_truncated", vec!["belief_at".to_string()]));
                break 'snapshots;
            }
            let entity = table.intern("n", &belief.entity);
            let predicate = table.intern("n", &belief.predicate);
            let value = table.intern("n", &belief.value);
            let source = format!("e{}", belief.source_seq);
            program.add_fact(fact(
                "belief_at",
                vec![situation.clone(), entity, predicate, value, source],
            ));
            belief_facts += 1;
        }
    }
    program.add_fact(fact(
        "theory_window",
        vec![
            events
                .first()
                .map(|event| event.seq)
                .unwrap_or(0)
                .to_string(),
            events
                .last()
                .map(|event| event.seq)
                .unwrap_or(0)
                .to_string(),
            events.len().to_string(),
        ],
    ));

    for source in [
        "before(S1, S2) :- next_situation(S1, S2).",
        "before(S1, S2) :- next_situation(S1, S3), before(S3, S2).",
        "answer_change(Q, A1, A2) :- next_answer(Q, A1, A2), answer_differ(A1, A2).",
        "change_at(Q, S1, S2) :- answer_change(Q, A1, A2), answer_at(A1, S1), answer_at(A2, S2).",
    ] {
        match parse_rule(source) {
            Ok(rule) => {
                let _ = program.add_rule(rule);
            }
            Err(error) => {
                return (
                    program,
                    table.reverse,
                    Some(format!("理论规则解析失败：{}", error)),
                    truncated_belief_facts,
                )
            }
        }
    }
    let evaluation_error = program.evaluate().err();
    (
        program,
        table.reverse,
        evaluation_error,
        truncated_belief_facts,
    )
}
/// 从导入回执提取可折叠效果（IRI 本地名化，保留可读实体/谓词）。
pub fn receipt_effects(receipt: &ImportReceipt, op: EffectOp) -> Vec<TheoryEffect> {
    let mut effects = BTreeSet::new();
    for action in &receipt.actions {
        match action {
            ImportAction::Named {
                subject,
                predicate,
                object,
            } => {
                effects.insert(TheoryEffect {
                    op,
                    entity: local_name(subject),
                    predicate: local_name(predicate),
                    value: local_name(object),
                });
            }
            ImportAction::Literal {
                subject,
                predicate,
                value,
            } => {
                effects.insert(TheoryEffect {
                    op,
                    entity: local_name(subject),
                    predicate: local_name(predicate),
                    value: value.clone(),
                });
            }
        }
    }
    effects.into_iter().collect()
}
/// 从可解析的中文断言提取效果（只取确定、基事实；其余诚实留空）。
pub fn effects_from_text(text: &str, op: EffectOp) -> Vec<TheoryEffect> {
    let outcome = parse_lessons(text.trim());
    let mut effects = BTreeSet::new();
    for lesson in &outcome.lessons {
        if lesson.kind != LessonKind::DeterministicFact || !lesson.head.is_ground() {
            continue;
        }
        let atom = &lesson.head;
        match atom.terms.as_slice() {
            [subject, object] => {
                effects.insert(TheoryEffect {
                    op,
                    entity: subject.name().to_string(),
                    predicate: atom.predicate.to_string(),
                    value: object.name().to_string(),
                });
            }
            [subject] => {
                effects.insert(TheoryEffect {
                    op,
                    entity: subject.name().to_string(),
                    predicate: atom.predicate.to_string(),
                    value: String::new(),
                });
            }
            _ => {}
        }
        if effects.len() >= MAX_EFFECTS_PER_EVENT {
            break;
        }
    }
    effects.into_iter().collect()
}
fn local_name(iri: &str) -> String {
    if iri == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" {
        return "type".to_string();
    }
    if let Some(name) = crate::identity::local_name(crate::identity::DEFAULT_BASE_NS, iri) {
        return name;
    }
    iri.rsplit(|ch| ch == '#' || ch == '/')
        .next()
        .unwrap_or(iri)
        .to_string()
}
fn effects_json(effects: &[TheoryEffect]) -> Vec<Value> {
    effects
        .iter()
        .map(|effect| {
            json!({
                "op": effect.op.as_str(),
                "entity": effect.entity,
                "predicate": effect.predicate,
                "value": effect.value,
            })
        })
        .collect()
}
fn receipt_digest(receipt: &ImportReceipt) -> String {
    let effects = receipt_effects(receipt, EffectOp::Add);
    fnv1a_hex(canonical_json(&Value::Array(effects_json(&effects))).as_bytes())
}
fn effects_digest(effects: &[TheoryEffect]) -> String {
    fnv1a_hex(canonical_json(&Value::Array(effects_json(effects))).as_bytes())
}
fn event_payload(
    summary: &str,
    source: &str,
    revision_after: u64,
    effects: &[TheoryEffect],
    enumerable: bool,
    digest: Option<String>,
    removes_digest: Option<String>,
    effect_count: usize,
) -> Value {
    let mut payload = json!({
        "summary": summary.lines().next().unwrap_or(summary),
        "source": source,
        "revision_after": revision_after,
        "effects": effects_json(effects),
        "effect_count": effect_count,
        "effects_truncated": effect_count > effects.len(),
        "enumerable": enumerable && !effects.is_empty() && effect_count == effects.len(),
    });
    if let Some(digest) = digest {
        payload["digest"] = Value::from(digest);
    }
    if let Some(digest) = removes_digest {
        payload["removes_digest"] = Value::from(digest);
    }
    payload
}
/// 追加文本导入事件（效果可枚举，可反事实重放）。
pub fn append_import(
    chronicle: &mut Chronicle,
    receipt: &ImportReceipt,
    source: &str,
    revision_after: u64,
) -> io::Result<u64> {
    let all_effects = receipt_effects(receipt, EffectOp::Add);
    let effect_count = all_effects.len();
    let effects: Vec<TheoryEffect> = all_effects
        .iter()
        .take(MAX_EFFECTS_PER_EVENT)
        .cloned()
        .collect();
    let digest = receipt_digest(receipt);
    chronicle.append(
        "import",
        event_payload(
            &receipt.summary(),
            source,
            revision_after,
            &effects,
            true,
            Some(digest),
            None,
            effect_count,
        ),
    )
}
/// 追加导入冲突事件（保留被拒入值；不改变可枚举信念）。
pub fn append_conflict(
    chronicle: &mut Chronicle,
    conflicts: &[crate::predicate_registry::ImportConflict],
    source: &str,
    revision_after: u64,
) -> io::Result<u64> {
    let items: Vec<Value> = conflicts
        .iter()
        .map(|conflict| {
            json!({
                "entity": local_name(&conflict.subject),
                "predicate": local_name(&conflict.predicate),
                "existing": conflict.existing.iter().map(|value| local_name(value)).collect::<Vec<_>>(),
                "incoming": local_name(&conflict.incoming),
            })
        })
        .collect();
    let first = conflicts.first();
    let summary = first
        .map(|conflict| {
            format!(
                "导入冲突：{}.{} 保留「{}」，拒绝「{}」",
                local_name(&conflict.subject),
                local_name(&conflict.predicate),
                conflict.existing.join("、"),
                conflict.incoming
            )
        })
        .unwrap_or_else(|| "导入冲突".to_string());
    let mut payload = event_payload(&summary, source, revision_after, &[], false, None, None, 0);
    payload["conflicts"] = Value::Array(items);
    chronicle.append("conflict", payload)
}

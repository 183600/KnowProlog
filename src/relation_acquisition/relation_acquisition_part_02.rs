impl RelationAcquisition {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn turn_count(&self) -> usize {
        self.turn
    }

    pub fn active_goal(&self) -> Option<&RelationGoal> {
        self.active_goal.as_ref()
    }

    pub fn has_pending_goal(&self) -> bool {
        self.active_goal.is_some()
    }

    pub fn clear_goal(&mut self) {
        self.active_goal = None;
    }

    pub fn claims(&self) -> impl Iterator<Item = &AcquiredClaim> {
        self.claims.values()
    }

    pub fn claim(&self, id: &str) -> Option<&AcquiredClaim> {
        self.claims.get(id)
    }

    pub fn patterns(&self) -> impl Iterator<Item = &DialogueInducedPattern> {
        self.patterns.values()
    }

    pub fn status_count(&self, status: AcquisitionStatus) -> usize {
        self.claims
            .values()
            .filter(|claim| claim.status == status)
            .count()
    }

    pub fn overlay(&self) -> SessionOverlay {
        let mut rows = Vec::new();
        for claim in self.claims.values() {
            if !claim.status.is_committed() {
                continue;
            }
            rows.push(SessionOverlayRow {
                subject: claim.claim.subject.clone(),
                predicate: claim.claim.predicate.clone(),
                object: claim.claim.object.text(),
                claim_id: claim.id.clone(),
                status: claim.status,
            });
        }
        rows.sort_by(|left, right| {
            (&left.subject, &left.predicate, &left.object, &left.claim_id).cmp(&(
                &right.subject,
                &right.predicate,
                &right.object,
                &right.claim_id,
            ))
        });
        SessionOverlay { rows }
    }

    pub fn digest(&self) -> String {
        let value = serde_json::to_value(self.snapshot()).unwrap_or(serde_json::Value::Null);
        fnv1a_hex(canonical_json(&value).as_bytes())
    }

    pub fn snapshot(&self) -> AcquisitionSnapshot {
        AcquisitionSnapshot {
            dialect: ACQUISITION_DIALECT.to_string(),
            turn: self.turn,
            last_claim: self.last_claim.clone(),
            active_goal: self.active_goal.clone(),
            claims: self.claims.clone(),
            patterns: self.patterns.clone(),
        }
    }

    pub fn restore(&mut self, snapshot: &AcquisitionSnapshot) -> bool {
        if snapshot.dialect != ACQUISITION_DIALECT {
            return false;
        }
        self.turn = snapshot.turn;
        self.last_claim = snapshot.last_claim.clone();
        self.active_goal = snapshot.active_goal.clone();
        self.claims = snapshot.claims.clone();
        self.patterns = snapshot.patterns.clone();
        true
    }

    pub fn save_to_path(&self, path: &str) -> std::io::Result<()> {
        if let Some(parent) = Path::new(path).parent() {
            fs::create_dir_all(parent)?;
        }
        let text = serde_json::to_string_pretty(&self.snapshot())
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::Other, error))?;
        fs::write(path, format!("{}\n", text))
    }

    pub fn load_from_path(path: &str) -> Option<Self> {
        let text = fs::read_to_string(path).ok()?;
        let snapshot: AcquisitionSnapshot = serde_json::from_str(&text).ok()?;
        let mut acquisition = Self::new();
        acquisition.restore(&snapshot).then_some(acquisition)
    }

    // ---------------------------------------------------------------------
    // 问题理解与目标生成
    // ---------------------------------------------------------------------

    pub fn looks_like_question(input: &str) -> bool {
        let text = input.trim();
        text.ends_with('？')
            || text.ends_with('?')
            || text.ends_with('吗')
            || text.contains("是谁")
            || text.contains("是什么")
            || text.contains("是多少")
            || text.contains("哪一位")
            || text.contains("哪一个")
            || text.contains("在哪")
            || text.contains("什么关系")
    }

    pub fn looks_like_answer(input: &str) -> bool {
        let text = input.trim();
        !text.is_empty() && !Self::looks_like_question(text)
    }

    pub fn parse_question(input: &str) -> Option<ParsedQuestion> {
        let mut text = input
            .trim()
            .trim_end_matches(['？', '?', '。', '！', '!', ' '])
            .trim()
            .to_string();
        for prefix in [
            "请问",
            "问一下",
            "追问：",
            "追问:",
            "关系获取：",
            "关系获取:",
            "问答获取：",
            "问答获取:",
        ] {
            if let Some(rest) = text.strip_prefix(prefix) {
                text = rest.trim().to_string();
                break;
            }
        }
        if text.is_empty() {
            return None;
        }

        if let Some((subject, predicate)) = parse_where_work_question(&text) {
            return Some(ParsedQuestion {
                subject,
                predicate,
                form: QuestionForm::Where,
            });
        }

        let reverse = text
            .strip_prefix("谁")
            .and_then(|rest| rest.strip_prefix("是").or_else(|| rest.strip_prefix("为")));
        if let Some(rest) = reverse {
            let (subject, predicate) = split_possession(rest)?;
            return Some(ParsedQuestion {
                subject,
                predicate,
                form: QuestionForm::Who,
            });
        }

        let (body, form) = split_question_marker(&text);
        if form == QuestionForm::Plain {
            return None;
        }
        let (subject, predicate) = split_possession(body)?;
        Some(ParsedQuestion {
            subject,
            predicate,
            form,
        })
    }

    pub fn start_question(
        &mut self,
        input: &str,
        facts: &[PrologFact],
        registry: &PredicateRegistry,
        types: &TypeSystem,
        shapes: &ShapeSet,
    ) -> GoalOpen {
        let Some(parsed) = Self::parse_question(input) else {
            return GoalOpen::NoRelation {
                reason: "请使用「S 的 P 是谁/是什么」句式，或显式输入「关系获取：S 的 P」。"
                    .to_string(),
            };
        };
        self.start_relation_with_form(
            &parsed.subject,
            &parsed.predicate,
            parsed.form,
            GoalOrigin::UserRequest,
            facts,
            registry,
            types,
            shapes,
        )
    }

    /// 显式命令入口：既接受问句，也接受「S 的 P」这种不带疑问词的槽位声明。
    /// 由形状修复动作打开一个缺口追问（`RepairAction::AskProperty` 的落点）。
    pub fn start_shape_gap(
        &mut self,
        subject: &str,
        predicate: &str,
        facts: &[PrologFact],
        registry: &PredicateRegistry,
        types: &TypeSystem,
        shapes: &ShapeSet,
    ) -> GoalOpen {
        self.start_relation_with_form(
            subject,
            predicate,
            QuestionForm::Plain,
            GoalOrigin::ShapeRepair,
            facts,
            registry,
            types,
            shapes,
        )
    }

    pub fn start_command(
        &mut self,
        input: &str,
        facts: &[PrologFact],
        registry: &PredicateRegistry,
        types: &TypeSystem,
        shapes: &ShapeSet,
    ) -> GoalOpen {
        if let Some(parsed) = Self::parse_question(input) {
            return self.start_relation_with_form(
                &parsed.subject,
                &parsed.predicate,
                parsed.form,
                GoalOrigin::UserRequest,
                facts,
                registry,
                types,
                shapes,
            );
        }
        if let Some((subject, predicate)) = split_possession(input) {
            return self.start_relation_with_form(
                &subject,
                &predicate,
                QuestionForm::Plain,
                GoalOrigin::UserRequest,
                facts,
                registry,
                types,
                shapes,
            );
        }
        self.start_question(input, facts, registry, types, shapes)
    }

    pub fn start_relation(
        &mut self,
        subject: &str,
        predicate: &str,
        facts: &[PrologFact],
        registry: &PredicateRegistry,
        types: &TypeSystem,
        shapes: &ShapeSet,
    ) -> GoalOpen {
        self.start_relation_with_form(
            subject,
            predicate,
            QuestionForm::Plain,
            GoalOrigin::UserRequest,
            facts,
            registry,
            types,
            shapes,
        )
    }

    fn start_relation_with_form(
        &mut self,
        subject: &str,
        predicate: &str,
        form: QuestionForm,
        origin: GoalOrigin,
        facts: &[PrologFact],
        registry: &PredicateRegistry,
        types: &TypeSystem,
        shapes: &ShapeSet,
    ) -> GoalOpen {
        let subject = subject.trim();
        let surface = predicate.trim();
        if subject.is_empty() || surface.is_empty() {
            return GoalOpen::NoRelation {
                reason: "主语与谓词都不能为空。".to_string(),
            };
        }
        let resolution = registry.resolve(surface);
        let canonical = resolution.canonical.trim().to_string();
        if canonical.is_empty() {
            return GoalOpen::NoRelation {
                reason: "谓词为空。".to_string(),
            };
        }
        let known_path = registry.is_known(&canonical)
            || shapes.relations.iter().any(|shape| shape.path == canonical)
            || graph_path_exists(facts, registry, &canonical);
        if !known_path {
            return GoalOpen::UnknownPredicate {
                predicate: surface.to_string(),
            };
        }

        let graph_values =
            graph_values_for(facts, registry, subject, &canonical, resolution.inverted);
        if !graph_values.is_empty() {
            return GoalOpen::AlreadyKnown {
                subject: subject.to_string(),
                predicate: canonical,
                values: graph_values,
                source: "知识图谱".to_string(),
            };
        }
        let overlay_values = self.overlay_values_for(subject, &canonical, resolution.inverted);
        if !overlay_values.is_empty() {
            return GoalOpen::AlreadyKnown {
                subject: subject.to_string(),
                predicate: canonical,
                values: overlay_values,
                source: "会话覆盖层".to_string(),
            };
        }

        let slot_claims = self
            .claims
            .values()
            .filter(|claim| {
                claim.claim.subject == subject
                    && claim.claim.predicate == canonical
                    && claim.status.is_live()
            })
            .count();
        if slot_claims >= MAX_CLAIMS_PER_SLOT {
            return GoalOpen::NoRelation {
                reason: format!(
                    "槽位「{} 的 {}」已有 {} 条候选，达到上限；请先确认、否认或撤回旧候选。",
                    subject, canonical, slot_claims
                ),
            };
        }

        self.turn += 1;
        let profile = registry.profile(&canonical);
        let expectation = expectation_for(&canonical, resolution.inverted, profile, shapes, types);
        let question = question_text(subject, surface, &expectation);
        let asked = self
            .claims
            .values()
            .filter(|claim| claim.claim.subject == subject && claim.claim.predicate == canonical)
            .count()
            + 1;
        let id = format!(
            "qg_{}",
            fnv1a_hex(
                canonical_json(&serde_json::json!({
                    "subject": subject,
                    "predicate": canonical,
                    "surface": surface,
                    "form": form,
                    "turn": self.turn,
                }))
                .as_bytes(),
            )
        );
        let goal = RelationGoal {
            id,
            subject: subject.to_string(),
            predicate: canonical,
            surface: surface.to_string(),
            expectation,
            origin,
            form,
            question,
            turn: self.turn,
            asked,
        };
        self.active_goal = Some(goal.clone());
        GoalOpen::Opened { goal }
    }

    // ---------------------------------------------------------------------
    // 回答分类与账本写入
    // ---------------------------------------------------------------------

    /// 自由陈述入口：把已抽取、已归一方向的断言写入账本。
    ///
    /// 与 `interpret_answer` 的区别是：这里没有槽位目标，用户陈述本身
    /// 就是候选；是否确认仍由形状闸门与用户下一轮回答决定。
    #[allow(clippy::too_many_arguments)]
    pub fn record_assertion(
        &mut self,
        claim: Claim,
        source: AcquisitionSource,
        raw: &str,
        rule: &str,
        facts: &[PrologFact],
        registry: &PredicateRegistry,
        types: &TypeSystem,
        shapes: &ShapeSet,
    ) -> AcquisitionOutcome {
        self.turn += 1;
        self.record_claim(
            None, claim, source, raw, rule, facts, registry, types, shapes,
        )
    }

    pub fn interpret_answer(
        &mut self,
        input: &str,
        facts: &[PrologFact],
        registry: &PredicateRegistry,
        types: &TypeSystem,
        shapes: &ShapeSet,
    ) -> AcquisitionOutcome {
        let Some(goal) = self.active_goal.clone() else {
            return AcquisitionOutcome::message(
                AnswerAct::Clarify,
                "当前没有待填的关系槽位；请先用「关系获取：S 的 P」打开一个目标。",
            );
        };
        self.turn += 1;
        let classified = classify_answer(input);
        match classified.act {
            AnswerAct::Unknown => {
                self.active_goal = None;
                AcquisitionOutcome::message(
                    AnswerAct::Unknown,
                    format!(
                        "已记下：你暂时不知道「{}」的「{}」。该槽位保持未确认，不会写入覆盖层。",
                        goal.subject, goal.surface
                    ),
                )
            }
            AnswerAct::Refuse => {
                self.active_goal = None;
                AcquisitionOutcome::message(
                    AnswerAct::Refuse,
                    format!(
                        "已尊重你的选择：不再追问「{}」的「{}」。",
                        goal.subject, goal.surface
                    ),
                )
            }
            AnswerAct::Clarify => AcquisitionOutcome {
                follow_up: Some(goal.question.clone()),
                ..AcquisitionOutcome::message(
                    AnswerAct::Clarify,
                    format!(
                        "我需要一个具体取值来填「{}」的「{}」。\n{}",
                        goal.subject, goal.surface, goal.question
                    ),
                )
            },
            AnswerAct::Confirm => {
                if !classified.values.is_empty() {
                    let mut outcomes = Vec::new();
                    for value in &classified.values {
                        outcomes.push(self.record_value(
                            &goal,
                            value,
                            AcquisitionSource::UserConfirmation,
                            input,
                            "确认回答",
                            facts,
                            registry,
                            types,
                            shapes,
                        ));
                    }
                    return merge_outcomes(AnswerAct::Confirm, outcomes);
                }
                self.confirm(None)
            }
            AnswerAct::Deny => {
                let mut outcome = self.deny(None);
                outcome.act = AnswerAct::Deny;
                outcome
            }
            AnswerAct::Correct => {
                let mut messages = Vec::new();
                if let Some(target) = self.last_proposed_for_goal(&goal.id) {
                    if let Some(claim) = self.claims.get_mut(&target) {
                        claim.status = AcquisitionStatus::Refuted;
                        claim.notes.push(format!("用户纠错否证：{}", input.trim()));
                        messages.push(format!("已否证候选 {}", target));
                    }
                }
                if classified.values.is_empty() {
                    return AcquisitionOutcome::message(
                        AnswerAct::Correct,
                        format!(
                            "{}。请给出修正后的取值，例如「不是叶莉，是邓紫棋」。",
                            messages.join("；")
                        ),
                    );
                }
                let mut outcomes = Vec::new();
                for value in &classified.values {
                    outcomes.push(self.record_value(
                        &goal,
                        value,
                        AcquisitionSource::UserCorrection,
                        input,
                        "纠错回答",
                        facts,
                        registry,
                        types,
                        shapes,
                    ));
                }
                let mut merged = merge_outcomes(AnswerAct::Correct, outcomes);
                if !messages.is_empty() {
                    merged.message = format!("{}；{}", messages.join("；"), merged.message);
                }
                merged
            }
            AnswerAct::Provide => {
                if classified.values.is_empty() {
                    return AcquisitionOutcome {
                        follow_up: Some(goal.question.clone()),
                        ..AcquisitionOutcome::message(
                            AnswerAct::Clarify,
                            format!("没有从「{}」中解析出可用的取值。", input.trim()),
                        )
                    };
                }
                let mut outcomes = Vec::new();
                for value in &classified.values {
                    outcomes.push(self.record_value(
                        &goal,
                        value,
                        AcquisitionSource::UserAnswer,
                        input,
                        "直接回答",
                        facts,
                        registry,
                        types,
                        shapes,
                    ));
                }
                merge_outcomes(AnswerAct::Provide, outcomes)
            }
        }
    }

    fn record_value(
        &mut self,
        goal: &RelationGoal,
        value: &str,
        source: AcquisitionSource,
        raw: &str,
        rule: &str,
        facts: &[PrologFact],
        registry: &PredicateRegistry,
        types: &TypeSystem,
        shapes: &ShapeSet,
    ) -> AcquisitionOutcome {
        let value = clean_value(value);
        if value.is_empty() {
            return AcquisitionOutcome::message(
                AnswerAct::Clarify,
                "取值为空，请重新给出一个具体值。",
            );
        }
        let claim = build_claim_for_goal(goal, &value);
        self.record_claim(
            Some(goal),
            claim,
            source,
            raw,
            rule,
            facts,
            registry,
            types,
            shapes,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn record_claim(
        &mut self,
        goal: Option<&RelationGoal>,
        claim: Claim,
        source: AcquisitionSource,
        raw: &str,
        rule: &str,
        facts: &[PrologFact],
        registry: &PredicateRegistry,
        types: &TypeSystem,
        shapes: &ShapeSet,
    ) -> AcquisitionOutcome {
        let claim = normalize_claim(claim, registry);
        let id = claim_id(&claim);
        let candidate = InterpretationCandidate {
            claim: claim.clone(),
            score: 100,
            segmentation: Vec::new(),
        };
        let decision = shapes.decide_candidates(&[candidate], facts, registry, types);
        let Some(verdict) = decision.verdicts.first() else {
            return AcquisitionOutcome::message(
                AnswerAct::Provide,
                "形状闸门没有返回裁决；本轮不写入账本。",
            );
        };

        let mut notes = Vec::new();
        let mut conflicts: BTreeSet<String> = BTreeSet::new();
        let mut status = match verdict.status {
            CandidateStatus::Accepted => {
                if matches!(
                    source,
                    AcquisitionSource::UserConfirmation | AcquisitionSource::UserCorrection
                ) {
                    AcquisitionStatus::Confirmed
                } else {
                    AcquisitionStatus::Proposed
                }
            }
            CandidateStatus::UnknownEntity => {
                notes.push(verdict.detail.clone());
                notes.push("新词面先作为候选，需确认后才进入覆盖层。".to_string());
                AcquisitionStatus::Proposed
            }
            CandidateStatus::UnknownPredicate => {
                notes.push(verdict.detail.clone());
                AcquisitionStatus::Rejected
            }
            CandidateStatus::TypeMismatch => {
                notes.push(verdict.detail.clone());
                AcquisitionStatus::Conflicted
            }
            CandidateStatus::ValueConflict => {
                notes.push(verdict.detail.clone());
                if let Some(RepairAction::DiscloseConflict { values, .. }) = &verdict.repair {
                    conflicts.extend(values.iter().cloned());
                }
                AcquisitionStatus::Conflicted
            }
        };
        if let Some(RepairAction::DiscloseConflict { values, .. }) = &verdict.repair {
            conflicts.extend(values.iter().cloned());
        }

        let object = claim.object.text();
        let functional = goal
            .map(|goal| goal.expectation.functional)
            .unwrap_or_else(|| registry.is_functional(&claim.predicate));
        if functional && status != AcquisitionStatus::Rejected {
            let mut others: BTreeSet<String> = self
                .claims
                .values()
                .filter(|other| {
                    other.id != id
                        && other.status.is_live()
                        && other.claim.subject == claim.subject
                        && other.claim.predicate == claim.predicate
                        && other.claim.object.text() != object
                })
                .map(|other| other.claim.object.text())
                .collect();
            if !others.is_empty() {
                status = AcquisitionStatus::Conflicted;
                notes.push("单值槽出现不同取值，需用户裁定。".to_string());
                conflicts.append(&mut others);
            }
        }

        let outcome = if let Some(existing) = self.claims.get_mut(&id) {
            let independent = existing.turn != self.turn || existing.raw_span != raw;
            if independent {
                existing.support += 1;
            }
            if source == AcquisitionSource::UserConfirmation {
                existing.confirmations += 1;
            }
            if existing.status == AcquisitionStatus::Refuted
                || existing.status == AcquisitionStatus::Rejected
            {
                existing.status = AcquisitionStatus::Proposed;
                existing.conflicts.clear();
                existing.notes.clear();
            }
            if status == AcquisitionStatus::Conflicted {
                existing.status = AcquisitionStatus::Conflicted;
                existing.conflicts.extend(conflicts.iter().cloned());
            } else if matches!(
                source,
                AcquisitionSource::UserConfirmation | AcquisitionSource::UserCorrection
            ) && existing.conflicts.is_empty()
            {
                existing.status = AcquisitionStatus::Confirmed;
            } else if existing.support >= CONFIRMATION_SUPPORT
                && existing.status == AcquisitionStatus::Proposed
                && existing.conflicts.is_empty()
            {
                existing.status = AcquisitionStatus::Confirmed;
            }
            for note in &notes {
                if !existing.notes.contains(note) {
                    existing.notes.push(note.clone());
                }
            }
            existing.turn = self.turn;
            existing.raw_span = raw.trim().to_string();
            existing.source = source;
            existing.rule = rule.to_string();
            existing.status
        } else {
            let entry = AcquiredClaim {
                id: id.clone(),
                goal_id: goal.map(|goal| goal.id.clone()),
                claim: claim.clone(),
                status,
                source,
                turn: self.turn,
                raw_span: raw.trim().to_string(),
                rule: rule.to_string(),
                support: 1,
                confirmations: usize::from(source == AcquisitionSource::UserConfirmation),
                conflicts,
                notes,
            };
            let status = entry.status;
            self.claims.insert(id.clone(), entry);
            status
        };

        self.last_claim = Some(id.clone());
        if outcome.is_committed() {
            if let Some(goal) = goal {
                self.induce_pattern(goal, &claim);
            }
        }

        let claim_ref = self.claims.get(&id);
        let mut message =
            match outcome {
                AcquisitionStatus::Proposed => format!(
                    "已记为候选：{} 的 {} = {}。若正确请回复「确认」，否则请纠错。",
                    claim.subject,
                    claim.predicate,
                    claim.object.text()
                ),
                AcquisitionStatus::Confirmed => format!(
                    "已确认：{} 的 {} = {}（仅会话覆盖层，未写入知识图谱）。",
                    claim.subject,
                    claim.predicate,
                    claim.object.text()
                ),
                AcquisitionStatus::Conflicted => format!(
                "发现冲突：{} 的 {} = {}；已有取值 {}。请用「不是…，是…」纠错或输入「确认」裁定。",
                claim.subject,
                claim.predicate,
                claim.object.text(),
                claim_ref
                    .map(|claim| claim.conflicts.iter().cloned().collect::<Vec<_>>().join("、"))
                    .unwrap_or_default()
            ),
                AcquisitionStatus::Refuted => format!(
                    "已否证：{} 的 {} = {}。",
                    claim.subject,
                    claim.predicate,
                    claim.object.text()
                ),
                AcquisitionStatus::Admitted => format!(
                    "已采纳：{} 的 {} = {}（已加入会话事实集）。",
                    claim.subject,
                    claim.predicate,
                    claim.object.text()
                ),
                AcquisitionStatus::Rejected => format!(
                    "未写入账本：{} 的 {} = {}（形状闸门拒绝）。",
                    claim.subject,
                    claim.predicate,
                    claim.object.text()
                ),
            };
        if let Some(claim) = claim_ref {
            if !claim.notes.is_empty() {
                message.push_str(&format!("\n  说明：{}", claim.notes.join("；")));
            }
        }
        let commit = if outcome.is_committed() {
            Some(CommitProposal {
                claim_id: id.clone(),
                sentence: commit_sentence(&claim),
                status: outcome,
            })
        } else {
            None
        };
        let follow_up = match outcome {
            AcquisitionStatus::Proposed => Some(
                "如果这条关系正确，请回复「确认」；如果不对，请回复「不是X，是Y」。".to_string(),
            ),
            AcquisitionStatus::Conflicted => Some("请先裁定冲突值，再写入会话覆盖层。".to_string()),
            AcquisitionStatus::Confirmed => Some(format!(
                "可输入「采纳获取 {}」把它加入会话事实集（仍不写知识图谱）。",
                id
            )),
            _ => None,
        };
        AcquisitionOutcome {
            act: AnswerAct::Provide,
            status: Some(outcome),
            claim_id: Some(id),
            message,
            commit,
            follow_up,
        }
    }

    // ---------------------------------------------------------------------
    // 确认 / 否认 / 撤回 / 采纳
    // ---------------------------------------------------------------------

    pub fn confirm(&mut self, id: Option<&str>) -> AcquisitionOutcome {
        let Some(id) = id
            .map(|id| id.trim().to_string())
            .or_else(|| self.last_proposed_for_goal_current())
            .or_else(|| self.last_claim.clone())
        else {
            return AcquisitionOutcome::message(
                AnswerAct::Confirm,
                "账本中没有可确认的候选；请先回答一个关系槽位。",
            );
        };
        let Some(claim) = self.claims.get(&id) else {
            return AcquisitionOutcome::message(
                AnswerAct::Confirm,
                format!("账本中没有条目 {}。", id),
            );
        };
        if matches!(
            claim.status,
            AcquisitionStatus::Refuted | AcquisitionStatus::Rejected
        ) {
            return AcquisitionOutcome::message(
                AnswerAct::Confirm,
                format!(
                    "条目 {} 已被否证或拒绝，不能直接确认；请重新回答该槽位。",
                    id
                ),
            );
        }
        let subject = claim.claim.subject.clone();
        let predicate = claim.claim.predicate.clone();
        let object = claim.claim.object.text();
        let goal = self
            .active_goal
            .clone()
            .filter(|goal| goal.subject == subject || goal.predicate == predicate);
        let mut refuted = Vec::new();
        if claim.status == AcquisitionStatus::Conflicted {
            let conflicting: Vec<String> = self
                .claims
                .values()
                .filter(|other| {
                    other.id != id
                        && other.claim.subject == subject
                        && other.claim.predicate == predicate
                        && other.claim.object.text() != object
                        && other.status.is_live()
                })
                .map(|other| other.id.clone())
                .collect();
            for other in conflicting {
                if let Some(other) = self.claims.get_mut(&other) {
                    other.status = AcquisitionStatus::Refuted;
                    other.notes.push(format!("用户确认 {} 后退役冲突值。", id));
                }
                refuted.push(other);
            }
        }
        let Some(claim) = self.claims.get_mut(&id) else {
            return AcquisitionOutcome::message(AnswerAct::Confirm, "账本条目已消失。");
        };
        claim.status = AcquisitionStatus::Confirmed;
        claim.confirmations += 1;
        claim.support += 1;
        claim.conflicts.clear();
        claim.notes.push("用户显式确认。".to_string());
        claim.turn = self.turn;
        let sentence = commit_sentence(&claim.claim);
        self.last_claim = Some(id.clone());
        let goal_for_pattern = goal.clone();
        let claim_for_pattern = claim.claim.clone();
        if let Some(goal) = goal_for_pattern.as_ref() {
            self.induce_pattern(goal, &claim_for_pattern);
        }
        let mut message = format!(
            "已确认：{} 的 {} = {}（仅会话覆盖层，未写入知识图谱）。",
            subject, predicate, object
        );
        if !refuted.is_empty() {
            message.push_str(&format!(
                "\n  已按你的裁定退役 {} 条冲突候选。",
                refuted.len()
            ));
        }
        AcquisitionOutcome {
            act: AnswerAct::Confirm,
            status: Some(AcquisitionStatus::Confirmed),
            claim_id: Some(id.clone()),
            message,
            commit: Some(CommitProposal {
                claim_id: id.clone(),
                sentence,
                status: AcquisitionStatus::Confirmed,
            }),
            follow_up: Some(format!(
                "可输入「采纳获取 {}」把它加入会话事实集（仍不写知识图谱）。",
                id
            )),
        }
    }

    pub fn deny(&mut self, id: Option<&str>) -> AcquisitionOutcome {
        let Some(id) = id
            .map(|id| id.trim().to_string())
            .or_else(|| self.last_proposed_for_goal_current())
            .or_else(|| self.last_claim.clone())
        else {
            return AcquisitionOutcome::message(AnswerAct::Deny, "账本中没有可否认的候选。");
        };
        let Some(claim) = self.claims.get_mut(&id) else {
            return AcquisitionOutcome::message(
                AnswerAct::Deny,
                format!("账本中没有条目 {}。", id),
            );
        };
        claim.status = AcquisitionStatus::Refuted;
        claim.notes.push("用户否认。".to_string());
        let rendered = format!(
            "{} 的 {} = {}",
            claim.claim.subject,
            claim.claim.predicate,
            claim.claim.object.text()
        );
        if self
            .active_goal
            .as_ref()
            .map(|goal| self.last_proposed_for_goal(&goal.id).is_none())
            .unwrap_or(false)
        {
            self.active_goal = None;
        }
        AcquisitionOutcome {
            act: AnswerAct::Deny,
            status: Some(AcquisitionStatus::Refuted),
            claim_id: Some(id),
            message: format!("已否证候选：{}。", rendered),
            commit: None,
            follow_up: None,
        }
    }

    pub fn retract(&mut self, id: Option<&str>) -> AcquisitionOutcome {
        let Some(id) = id
            .map(|id| id.trim().to_string())
            .or_else(|| self.last_claim.clone())
        else {
            return AcquisitionOutcome::message(AnswerAct::Deny, "账本中没有可撤回的条目。");
        };
        let Some(claim) = self.claims.get_mut(&id) else {
            return AcquisitionOutcome::message(
                AnswerAct::Deny,
                format!("账本中没有条目 {}。", id),
            );
        };
        claim.status = AcquisitionStatus::Refuted;
        claim.notes.push("用户撤回。".to_string());
        let rendered = format!(
            "{} 的 {} = {}",
            claim.claim.subject,
            claim.claim.predicate,
            claim.claim.object.text()
        );
        AcquisitionOutcome {
            act: AnswerAct::Deny,
            status: Some(AcquisitionStatus::Refuted),
            claim_id: Some(id),
            message: format!("已撤回会话覆盖层条目：{}。", rendered),
            commit: None,
            follow_up: None,
        }
    }

    pub fn commit_proposal(&self, id: Option<&str>) -> Option<CommitProposal> {
        let id = id
            .map(|id| id.trim().to_string())
            .or_else(|| self.last_claim.clone())?;
        let claim = self.claims.get(&id)?;
        if !claim.status.is_committed() {
            return None;
        }
        Some(CommitProposal {
            claim_id: id,
            sentence: commit_sentence(&claim.claim),
            status: claim.status,
        })
    }

    pub fn mark_admitted(&mut self, id: &str) -> bool {
        let Some(claim) = self.claims.get_mut(id) else {
            return false;
        };
        if !claim.status.is_committed() {
            return false;
        }
        claim.status = AcquisitionStatus::Admitted;
        claim.notes.push("已采纳到会话事实集。".to_string());
        true
    }

    pub fn retire_pattern(&mut self, id: &str) -> bool {
        let Some(pattern) = self.patterns.get_mut(id) else {
            return false;
        };
        pattern.retired = true;
        pattern.active = false;
        true
    }

    // ---------------------------------------------------------------------
    // 覆盖层查询与模式诱导
    // ---------------------------------------------------------------------

    fn overlay_values_for(&self, subject: &str, predicate: &str, inverted: bool) -> Vec<String> {
        let mut values = Vec::new();
        for claim in self.claims.values() {
            if !claim.status.is_committed() || claim.claim.predicate != predicate {
                continue;
            }
            if inverted {
                if claim.claim.object.text() == subject {
                    values.push(claim.claim.subject.clone());
                }
            } else if claim.claim.subject == subject {
                values.push(claim.claim.object.text());
            }
        }
        values.sort();
        values.dedup();
        values
    }

    pub fn answer_from_overlay(
        &self,
        input: &str,
        registry: &PredicateRegistry,
    ) -> Option<OverlayAnswer> {
        let parsed = Self::parse_question(input)?;
        let resolution = registry.resolve(&parsed.predicate);
        let canonical = resolution.canonical.trim();
        if canonical.is_empty() {
            return None;
        }
        let mut values = Vec::new();
        let mut claim_ids = Vec::new();
        for claim in self.claims.values() {
            if !claim.status.is_committed() || claim.claim.predicate != canonical {
                continue;
            }
            let matches = if resolution.inverted {
                claim.claim.object.text() == parsed.subject
            } else {
                claim.claim.subject == parsed.subject
            };
            if matches {
                values.push(if resolution.inverted {
                    claim.claim.subject.clone()
                } else {
                    claim.claim.object.text()
                });
                claim_ids.push(claim.id.clone());
            }
        }
        if values.is_empty() {
            return None;
        }
        values.sort();
        values.dedup();
        claim_ids.sort();
        claim_ids.dedup();
        Some(OverlayAnswer {
            subject: parsed.subject,
            predicate: parsed.predicate,
            values,
            claim_ids,
        })
    }

    fn last_proposed_for_goal(&self, goal_id: &str) -> Option<String> {
        self.claims
            .values()
            .filter(|claim| claim.goal_id.as_deref() == Some(goal_id))
            .filter(|claim| {
                matches!(
                    claim.status,
                    AcquisitionStatus::Proposed | AcquisitionStatus::Conflicted
                )
            })
            .max_by(|left, right| {
                left.turn
                    .cmp(&right.turn)
                    .then_with(|| left.id.cmp(&right.id))
            })
            .map(|claim| claim.id.clone())
    }

    fn last_proposed_for_goal_current(&self) -> Option<String> {
        let goal = self.active_goal.as_ref()?;
        self.last_proposed_for_goal(&goal.id)
    }

    fn induce_pattern(&mut self, goal: &RelationGoal, claim: &Claim) {
        let key = format!("{}|{}", claim.predicate, goal.form.label());
        let id = format!("dip_{}", fnv1a_hex(key.as_bytes()));
        let source_claim = claim_id(claim);
        let entry = self
            .patterns
            .entry(id.clone())
            .or_insert_with(|| DialogueInducedPattern {
                id: id.clone(),
                predicate: claim.predicate.clone(),
                surface: goal.surface.clone(),
                form: goal.form,
                question_template: if goal.form == QuestionForm::Plain {
                    format!("{{subject}}的{}是什么", goal.surface)
                } else {
                    format!("{{subject}}的{}是{}", goal.surface, goal.form.label())
                },
                statement_template: format!("{{subject}}的{}是{{object}}", goal.surface),
                support: 0,
                active: false,
                retired: false,
                source_claims: BTreeSet::new(),
                examples: BTreeSet::new(),
            });
        if !entry.source_claims.insert(source_claim) {
            return;
        }
        entry.support += 1;
        entry.examples.insert(format!(
            "{}的{}是{}",
            claim.subject,
            goal.surface,
            claim.object.text()
        ));
        if entry.support >= CONFIRMATION_SUPPORT && !entry.retired {
            entry.active = true;
        }
    }

    pub fn scan_text(&self, text: &str, registry: &PredicateRegistry) -> PatternScan {
        let active: Vec<&DialogueInducedPattern> = self
            .patterns
            .values()
            .filter(|pattern| pattern.active)
            .take(MAX_MATCHED_PATTERNS)
            .collect();
        if active.is_empty() {
            return PatternScan::default();
        }
        let mut claims = Vec::new();
        let mut used = BTreeSet::new();
        for segment in text.split(|ch| matches!(ch, '。' | '；' | ';' | '！' | '!' | '\n')) {
            let segment = segment.trim();
            if segment.is_empty() || segment.contains('？') || segment.contains('?') {
                continue;
            }
            let Some((subject, rest)) = segment.split_once('的') else {
                continue;
            };
            let Some((predicate_surface, object)) = rest.split_once('是') else {
                continue;
            };
            let object = clean_value(object);
            if subject.trim().is_empty() || object.is_empty() {
                continue;
            }
            let resolution = registry.resolve(predicate_surface.trim());
            let canonical = resolution.canonical.trim();
            for pattern in &active {
                if pattern.predicate != canonical {
                    continue;
                }
                let claim = if resolution.inverted {
                    Claim::entity(object.clone(), canonical, subject.trim())
                } else {
                    Claim::entity(subject.trim(), canonical, object.clone())
                };
                used.insert(pattern.id.clone());
                claims.push(claim);
            }
        }
        claims.sort_by(|left, right| claim_key(left).cmp(&claim_key(right)));
        claims.dedup_by(|left, right| claim_key(left) == claim_key(right));
        PatternScan {
            claims,
            patterns: used,
        }
    }

    pub fn absorb_pattern_scan(
        &mut self,
        text: &str,
        facts: &[PrologFact],
        registry: &PredicateRegistry,
        types: &TypeSystem,
        shapes: &ShapeSet,
    ) -> AcquisitionOutcome {
        self.turn += 1;
        let scan = self.scan_text(text, registry);
        if scan.is_empty() {
            return AcquisitionOutcome {
                act: AnswerAct::Provide,
                status: None,
                claim_id: None,
                message: scan.render_text(),
                commit: None,
                follow_up: None,
            };
        }
        let claims = scan.claims.clone();
        let mut outcomes = Vec::new();
        for claim in claims {
            outcomes.push(self.record_claim(
                None,
                claim,
                AcquisitionSource::InducedPattern,
                text,
                "对话诱导模式扫描",
                facts,
                registry,
                types,
                shapes,
            ));
        }
        let mut merged = merge_outcomes(AnswerAct::Provide, outcomes);
        merged.message = format!("{}\n{}", scan.render_text(), merged.message);
        merged
    }

    // ---------------------------------------------------------------------
    // 报告
    // ---------------------------------------------------------------------

    pub fn report(&self) -> String {
        let mut lines = vec![format!(
            "问答式关系获取（QRA v{}）：轮次 {}，账本 {} 条（候选 {} / 已确认 {} / 冲突 {} / 已否证 {} / 已拒绝 {} / 已采纳 {}），模式 {} 个（激活 {}）",
            ACQUISITION_DIALECT.trim_start_matches("knowprolog-qra/"),
            self.turn,
            self.claims.len(),
            self.status_count(AcquisitionStatus::Proposed),
            self.status_count(AcquisitionStatus::Confirmed),
            self.status_count(AcquisitionStatus::Conflicted),
            self.status_count(AcquisitionStatus::Refuted),
            self.status_count(AcquisitionStatus::Rejected),
            self.status_count(AcquisitionStatus::Admitted),
            self.patterns.len(),
            self.patterns.values().filter(|pattern| pattern.active).count()
        )];
        match &self.active_goal {
            Some(goal) => lines.push(format!("待填槽位：{}", goal.render_text())),
            None => lines.push("待填槽位：（无）".to_string()),
        }
        if self.claims.is_empty() {
            lines.push("账本：（空；输入「关系获取：姚明的女儿是谁」开始）".to_string());
        } else {
            lines.push("账本：".to_string());
            for claim in self.claims.values() {
                lines.push(claim.render_text());
            }
        }
        lines.push(self.overlay().render_text());
        lines.push(self.patterns_report());
        lines.push(format!("账本摘要：{}", self.digest()));
        lines.join("\n")
    }

    pub fn patterns_report(&self) -> String {
        let mut lines = vec![format!(
            "对话诱导模式（DIP）：{} 个（激活 {}）",
            self.patterns.len(),
            self.patterns
                .values()
                .filter(|pattern| pattern.active)
                .count()
        )];
        if self.patterns.is_empty() {
            lines.push("  （空；同一谓词被两次确认后自动诱导）".to_string());
        } else {
            for pattern in self.patterns.values() {
                lines.push(pattern.render_text());
            }
        }
        lines.join("\n")
    }
}
/// 「S 在哪(里/儿/个单位)工作」：中文里最常见的任职问法，不是所有格。
/// 归一为规范谓词「任职于」，让覆盖层与图谱走同一方向。
fn parse_where_work_question(text: &str) -> Option<(String, String)> {
    let suffix = ["工作", "上班", "任职", "就职", "供职"]
        .iter()
        .find(|suffix| text.ends_with(*suffix))?;
    let body = text[..text.len() - suffix.len()]
        .trim_end_matches(['了', '过'])
        .trim();
    let position = body.rfind('在')?;
    let tail = &body[position + '在'.len_utf8()..];
    if !(tail.is_empty() || tail.starts_with('哪') || tail.starts_with("什么")) {
        return None;
    }
    let subject = body[..position].trim();
    if subject.is_empty() {
        return None;
    }
    Some((subject.to_string(), "任职于".to_string()))
}
fn split_possession(text: &str) -> Option<(String, String)> {
    let (subject, predicate) = text.rsplit_once('的')?;
    let subject = subject.trim();
    let predicate = predicate.trim();
    if subject.is_empty() || predicate.is_empty() {
        return None;
    }
    Some((subject.to_string(), predicate.to_string()))
}
fn split_question_marker(text: &str) -> (&str, QuestionForm) {
    let markers: [(&str, QuestionForm); 18] = [
        ("是谁", QuestionForm::Who),
        ("是哪一位", QuestionForm::Who),
        ("是哪一个", QuestionForm::Who),
        ("是什么人", QuestionForm::Who),
        ("叫什么", QuestionForm::Who),
        ("是什么", QuestionForm::What),
        ("是啥", QuestionForm::What),
        ("是多少", QuestionForm::Value),
        ("是几", QuestionForm::Value),
        ("是哪里", QuestionForm::Where),
        ("在哪", QuestionForm::Where),
        ("呢", QuestionForm::What),
        ("吗", QuestionForm::What),
        ("谁", QuestionForm::Who),
        ("什么", QuestionForm::What),
        ("多少", QuestionForm::Value),
        ("哪里", QuestionForm::Where),
        ("哪儿", QuestionForm::Where),
    ];
    for (marker, form) in markers {
        if let Some(body) = text.strip_suffix(marker) {
            let body = body.trim();
            if !body.is_empty() {
                return (body, form);
            }
        }
    }
    (text, QuestionForm::Plain)
}
fn graph_path_exists(facts: &[PrologFact], registry: &PredicateRegistry, canonical: &str) -> bool {
    facts.iter().any(|fact| match fact {
        PrologFact::Entity { properties, .. } => properties
            .keys()
            .any(|key| registry.resolve(key).canonical == canonical),
        PrologFact::Relation { rel_type, .. } => registry.resolve(rel_type).canonical == canonical,
    })
}
fn graph_values_for(
    facts: &[PrologFact],
    registry: &PredicateRegistry,
    subject: &str,
    canonical: &str,
    inverted: bool,
) -> Vec<String> {
    let mut values = Vec::new();
    for fact in facts {
        match fact {
            PrologFact::Relation { from, rel_type, to } => {
                let resolution = registry.resolve(rel_type);
                if resolution.canonical != canonical {
                    continue;
                }
                let (canonical_subject, canonical_object) = if resolution.inverted {
                    (to.as_str(), from.as_str())
                } else {
                    (from.as_str(), to.as_str())
                };
                if inverted {
                    if canonical_object == subject {
                        values.push(canonical_subject.to_string());
                    }
                } else if canonical_subject == subject {
                    values.push(canonical_object.to_string());
                }
            }
            PrologFact::Entity {
                name, properties, ..
            } => {
                if inverted || name != subject {
                    continue;
                }
                for (key, value) in properties {
                    if registry.resolve(key).canonical == canonical {
                        values.push(value.clone());
                    }
                }
            }
        }
    }
    values.sort();
    values.dedup();
    values
}

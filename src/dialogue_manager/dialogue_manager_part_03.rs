impl DialogueManager {
    /// 执行策略：固化 ABox → 逻辑程序求值 → 选策略 → 组合生成 → 记录对话实例。
    fn execute(
        &mut self,
        frame: UtteranceFrame,
        knowledge: &dyn DialogueKnowledge,
    ) -> Option<DialogueReply> {
        self.last_recovery = None;
        let mut identity_card: Option<IdentityCard> = None;
        let mut entity = frame.entity.clone();
        let pure_dialogue = matches!(
            frame.kind,
            MoveKind::Greet
                | MoveKind::Thanks
                | MoveKind::Farewell
                | MoveKind::MetaRecap
                | MoveKind::MetaReview
                | MoveKind::MetaWhy
        );
        if entity.is_none() && !pure_dialogue {
            entity = self.focus.last().cloned();
        }
        let known = entity
            .as_deref()
            .map(|name| knowledge.known_entity(name))
            .unwrap_or(false);
        // 只有属性没有实体（且无焦点）时不外抛：由策略层生成澄清问题；
        // 实体存在但图谱不认识时让位给常规 NLU / 外部知识管线。
        let attribute_only = matches!(frame.kind, MoveKind::AskAttribute)
            && entity.is_none()
            && frame.attribute.is_some();

        // —— 失败恢复：引擎层故障优先于一切知识判定 ——
        // On 模式下故障必须显式说「我出错了」；Observe 只入台账；Off 不介入。
        let mut recovery_attempt: Option<recovery::RecoveryAttempt> = None;
        let ask_like = matches!(
            frame.kind,
            MoveKind::AskAttribute | MoveKind::AskIdentity | MoveKind::AskBinary
        );
        if !self.recovery_mode.eq(&RecoveryMode::Off) && ask_like {
            self.recovery.begin_turn();
            if let Some(fault) = knowledge.engine_fault() {
                let fault = fault.trim().to_string();
                let request = RecoveryRequest {
                    status: EpistemicStatus::EngineFault,
                    entity: entity.as_deref(),
                    attribute: frame.attribute.as_deref(),
                    detail: &fault,
                };
                let attempt = recovery::attempt(
                    &mut self.recovery,
                    self.recovery_mode,
                    &request,
                    &KnowledgeRecovery { knowledge },
                );
                self.note_recovery(&attempt);
                if self.recovery_mode == RecoveryMode::On {
                    let text = if attempt.text.is_empty() {
                        crate::epistemic::render_statement(
                            EpistemicStatus::EngineFault,
                            entity.as_deref(),
                            frame.attribute.as_deref(),
                            None,
                        )
                    } else {
                        attempt.text.clone()
                    };
                    let plans = vec!["engine_fault".to_string()];
                    self.turns.push(DialogueTurn {
                        index: self.turns.len() + 1,
                        user: raw_input(&frame),
                        kind: frame.kind.clone(),
                        entity: entity.clone(),
                        attribute: frame.attribute.clone(),
                        answer_excerpt: None,
                        provenance: vec![format!("引擎故障：{}", fault)],
                        plan: plans.clone(),
                        warrant: None,
                        kb_revision: self.quality.as_ref().map(|index| index.revision),
                    });
                    if let Some(entity) = &entity {
                        self.push_focus(entity);
                    }
                    self.last_reply = plans.clone();
                    if self.externals.last().map(String::as_str) == Some(frame.raw.trim()) {
                        self.externals.pop();
                    }
                    return Some(DialogueReply {
                        text,
                        plan: plans,
                        kind: frame.kind.clone(),
                        derivation: None,
                    });
                }
                recovery_attempt = Some(attempt);
            }
        }

        if matches!(frame.kind, MoveKind::AskAttribute | MoveKind::AskIdentity)
            && !known
            && !attribute_only
        {
            return None;
        }

        let known_attributes: Vec<String> = match (&entity, known) {
            (Some(name), true) => knowledge.known_attributes(name),
            _ => Vec::new(),
        };
        let asked = frame.attribute.clone();
        let more_attributes: Vec<String> = known_attributes
            .iter()
            .filter(|attribute| Some(attribute.as_str()) != asked.as_deref())
            .filter(|attribute| !self.already_asked(&entity, attribute))
            .cloned()
            .collect();
        let mut answer: Option<String> = if known {
            frame.question.as_deref().and_then(|question| {
                if self.recovery_mode.acts() {
                    knowledge.answer_certain(question)
                } else {
                    knowledge.answer(question)
                }
            })
        } else {
            None
        };
        // 结构化兜底：链接预测/推导链答不上来时，直接检索字面量属性取值
        if answer.is_none() && known {
            if let (Some(name), Some(attribute)) = (entity.as_deref(), frame.attribute.as_deref()) {
                answer = self.lookup_attribute(knowledge, name, attribute);
            }
        }
        // 身份问句兜底：用符号生成器输出结构化画像段落
        if answer.is_none() && known && matches!(frame.kind, MoveKind::AskIdentity) {
            if let Some(name) = entity.as_deref() {
                answer = knowledge.describe(name);
            }
        }
        // 实体身份卡：结构化 claim 随回答追加（首个实质行仍是旧答案，兼容旧口径）。
        if known && matches!(frame.kind, MoveKind::AskIdentity) {
            if let Some(name) = entity.as_deref() {
                identity_card = Some(identity_dialogue::entity_card(knowledge, name));
            }
        }
        // On 模式：用户更正后的会话内覆盖值优先于图谱旧值（改口在本会话内立即生效）。
        let mut used_correction = false;
        if self.trace_mode.is_on() {
            if let (Some(name), Some(attribute)) = (entity.as_deref(), frame.attribute.as_deref()) {
                if let Some(value) = self.trace.value_override(name, attribute) {
                    answer = Some(value.to_string());
                    used_correction = true;
                }
            }
        }
        // 会话层确认事实优先于图谱与预测：用户确认/更正过的取值在后续轮次生效。
        let mut used_confirmation = false;
        if known {
            if let (Some(name), Some(attribute)) = (entity.as_deref(), frame.attribute.as_deref()) {
                if let Some(fact) = self.recovery.confirmed_value(name, attribute) {
                    answer = Some(fact.value.clone());
                    used_confirmation = true;
                }
            }
        }

        // —— ABox：本轮的信息状态 ——
        let turn_id = format!("t{}", self.turns.len() + 1);
        let kind_atom = match frame.kind {
            MoveKind::MetaRecap => "meta_recap",
            MoveKind::MetaReview => "meta_review",
            MoveKind::MetaWhy => "meta_why",
            MoveKind::Greet => "greet",
            MoveKind::Thanks => "thanks",
            MoveKind::Farewell => "farewell",
            MoveKind::ContinueRequest => "continue",
            MoveKind::ElaborateRequest => "elaborate",
            MoveKind::AskAttribute | MoveKind::AskIdentity | MoveKind::AskBinary => "ask",
            MoveKind::Assert => "assert",
            MoveKind::Other => "other",
        };
        let mut abox = vec![format!("kind({}, {})", turn_id, kind_atom)];
        if !self.focus.is_empty() {
            abox.push(format!("has_focus({})", turn_id));
        }
        if self.focus.len() >= 2 {
            abox.push(format!("multiple_focus({})", turn_id));
        }
        if self.last_answered_turn().is_some() {
            abox.push(format!("has_evidence({})", turn_id));
        }
        if frame.question.as_deref() == Some("why_not") {
            abox.push(format!("why_not({})", turn_id));
        }
        if frame
            .question
            .as_deref()
            .is_some_and(|question| question.starts_with("step:"))
        {
            abox.push(format!("step_request({})", turn_id));
        }
        if known {
            abox.push(format!("entity_known({})", turn_id));
        }
        if answer.is_some() {
            abox.push(format!("answer_found({})", turn_id));
        }
        if frame.attribute.is_some() {
            abox.push(format!("has_attribute({})", turn_id));
        }
        if frame.entity.is_some() && !frame.inherited {
            abox.push(format!("has_explicit_entity({})", turn_id));
        }
        if !more_attributes.is_empty() {
            abox.push(format!("has_more_attributes({})", turn_id));
        }

        self.policy.clear_facts();
        for fact in &abox {
            let _ = self.policy.add_fact_str(fact);
        }
        if self.policy.evaluate().is_err() {
            return None;
        }
        let mut plans: Vec<String> = self
            .policy
            .query("respond(X)")
            .ok()?
            .into_iter()
            .filter_map(|env| env.get("X").cloned())
            .collect();
        plans.sort_by_key(|name| {
            REPLY_PRIORITY
                .iter()
                .position(|candidate| candidate == name)
                .unwrap_or(usize::MAX)
        });
        plans.dedup();
        let chosen = plans.first()?.clone();
        let warrant = self.quality.as_ref().and_then(|index| {
            entity
                .as_deref()
                .and_then(|name| index.warrant_for(name, frame.attribute.as_deref()))
        });

        let mut explanation = None;
        let text = match chosen.as_str() {
            "recap" => self.recap_text(),
            "review_answer" => {
                let (text, tag) = self.review_text(knowledge);
                plans.push(tag.to_string());
                text
            }
            "explain_why" => {
                let (text, derivation) = if frame.question.as_deref() == Some("why_not") {
                    self.explain_last_failure_capture(knowledge)
                } else if let Some(step) = frame
                    .question
                    .as_deref()
                    .and_then(|question| question.strip_prefix("step:"))
                    .and_then(|step| step.parse::<usize>().ok())
                {
                    self.explain_step_capture(knowledge, step)
                } else {
                    self.explain_last_inner(Some(knowledge))
                };
                explanation = derivation;
                text
            }
            "why_no_evidence" => self
                .failure_self_description()
                .unwrap_or_else(|| {
                    "我还没有答过什么问题，暂时没有可解释的依据。你可以先问我一个知识问题，再问我「你怎么知道的」。"
                        .to_string()
                }),
            "continue_topic" => self.continue_text(entity.as_deref(), knowledge),
            "continue_no_focus" => {
                "我还不知道你想接着聊哪个实体。先给我一个名字吧，比如「姚明」。".to_string()
            }
            "elaborate" => self.elaborate_text(entity.as_deref(), knowledge),
            "clarify_entity" => self.clarify_text(frame.attribute.as_deref()),
            "clarify_entity_no_focus" => format!(
                "你想问谁的「{}」？先告诉我实体名吧。",
                frame.attribute.as_deref().unwrap_or("这个属性")
            ),
            "direct_answer" => self.compose_direct(
                answer.as_deref().unwrap_or(""),
                entity.as_deref().unwrap_or(""),
                &more_attributes,
            ),
            "admit_ignorance" => {
                // 失败恢复：缺记录时走 L2（候选验证）/ L3（一问澄清）/ L4（承认）。
                if recovery_attempt.is_none() && !self.recovery_mode.eq(&RecoveryMode::Off) {
                    let status = if known {
                        EpistemicStatus::Absent
                    } else {
                        EpistemicStatus::Unknown
                    };
                    let request = RecoveryRequest {
                        status,
                        entity: entity.as_deref(),
                        attribute: frame.attribute.as_deref(),
                        detail: "",
                    };
                    let attempt = recovery::attempt(
                        &mut self.recovery,
                        self.recovery_mode,
                        &request,
                        &KnowledgeRecovery { knowledge },
                    );
                    self.note_recovery(&attempt);
                    recovery_attempt = Some(attempt);
                }
                self.compose_ignorance(
                    entity.as_deref().unwrap_or(""),
                    frame.attribute.as_deref(),
                    &more_attributes,
                )
            }
            "greet" => self.greet_text(),
            "thanks" => self.thanks_text(),
            "farewell" => self.farewell_text(),
            _ => return None,
        };
        // 提示资产接管：接地闸门通过则用模板渲染（On），双跑模式计数（Parity），
        // 任一 required 槽位未绑定则回退上面的组合生成器（结构保证不幻觉）。
        let text = self.apply_prompt_spec(
            &chosen,
            &frame,
            entity.as_deref(),
            answer.as_deref(),
            &more_attributes,
            text,
        );
        // On 模式：恢复引擎的文本（含证据链/追问/回执 id）取代生成器措辞；
        // Observe 模式只记台账，文本不变。
        let text = match &recovery_attempt {
            Some(attempt) if self.recovery_mode == RecoveryMode::On && !attempt.text.is_empty() => {
                attempt.text.clone()
            }
            _ => text,
        };
        let (text, plans) =
            self.apply_warrant(text, plans, warrant.as_ref(), &frame.kind, answer.is_some());
        let mut text = text;
        if let Some(card) = &identity_card {
            let extension = card.render();
            if !extension.is_empty() && !text.contains("【身份卡·实体】") {
                if !text.trim().is_empty() {
                    text.push_str("\n\n");
                }
                text.push_str(&extension);
            }
            self.last_identity_card = Some(card.clone());
        }

        if used_confirmation && !text.contains("本会话") {
            if !text.trim().is_empty() {
                text.push('\n');
            }
            text.push_str("（来源：本会话确认；未写入图谱）");
        }

        let mut provenance = answer
            .as_deref()
            .map(classify_provenance)
            .unwrap_or_default();
        if used_correction {
            provenance.push("用户更正".to_string());
        }
        if used_confirmation {
            provenance.push("本会话确认".to_string());
        }
        // 知识/续聊轮进入对话本体实例；元话语与寒暄只影响策略，不记入"聊了什么"
        let content_turn = matches!(
            frame.kind,
            MoveKind::AskAttribute
                | MoveKind::AskIdentity
                | MoveKind::AskBinary
                | MoveKind::ContinueRequest
                | MoveKind::ElaborateRequest
        );
        if content_turn {
            self.turns.push(DialogueTurn {
                index: self.turns.len() + 1,
                user: raw_input(&frame),
                kind: frame.kind.clone(),
                entity: entity.clone(),
                attribute: frame.attribute.clone(),
                answer_excerpt: answer.as_deref().map(|text| excerpt(text, 40)),
                provenance: provenance.clone(),
                plan: plans.clone(),
                warrant,
                kb_revision: self.quality.as_ref().map(|index| index.revision),
            });
        }
        if let Some(entity) = &entity {
            self.push_focus(entity);
        }
        self.last_reply = plans.clone();
        if self.externals.last().map(String::as_str) == Some(frame.raw.trim()) {
            self.externals.pop();
        }
        let reply = DialogueReply {
            text: text.clone(),
            plan: plans,
            kind: frame.kind.clone(),
            derivation: explanation,
        };
        self.record_turn_trace(&frame, &entity, Some(&reply), &provenance);
        Some(reply)
    }

    /// 记录一次恢复动作的对外视图。
    fn note_recovery(&mut self, attempt: &recovery::RecoveryAttempt) {
        self.last_recovery = Some(RecoveryNotice {
            status: attempt.status,
            level: attempt.level.label().to_string(),
            receipt_id: attempt.receipt_id.clone(),
            evidence: attempt.evidence.clone(),
            question: attempt.question.clone(),
        });
    }

    fn already_asked(&self, entity: &Option<String>, attribute: &str) -> bool {
        let Some(entity) = entity else {
            return false;
        };
        self.turns.iter().any(|turn| {
            turn.entity.as_deref() == Some(entity.as_str())
                && turn.attribute.as_deref() == Some(attribute)
        })
    }

    fn last_answered_turn(&self) -> Option<&DialogueTurn> {
        self.turns
            .iter()
            .rev()
            .find(|turn| turn.answer_excerpt.is_some())
    }

    fn push_focus(&mut self, entity: &str) {
        if entity.is_empty() || self.focus.last().map(String::as_str) == Some(entity) {
            return;
        }
        self.focus.retain(|focus| focus != entity);
        self.focus.push(entity.to_string());
        while self.focus.len() > 5 {
            self.focus.remove(0);
        }
    }

    /// 用户改口拦截（仅 On 模式）：识别「不对，叶莉的身高是 190」这类更正，
    /// 覆盖会话内取值并把上一轮相关回复判为失效。
    fn try_trace_correction(
        &mut self,
        input: &str,
        knowledge: &dyn DialogueKnowledge,
    ) -> Option<DialogueReply> {
        if !self.trace_mode.is_on() {
            return None;
        }
        let stripped = input
            .trim()
            .trim_end_matches(['？', '?', '。', '！', '!', ' '])
            .trim();
        if stripped.is_empty() || stripped.chars().count() > 40 {
            return None;
        }
        const PREFIXES: [&str; 6] = ["不对，", "不是，", "错了，", "更正：", "改成", "应该是"];
        let body = PREFIXES
            .iter()
            .find_map(|prefix| stripped.strip_prefix(prefix))
            .map(str::trim)?;
        if body.is_empty() || body.contains('？') || body.contains('?') {
            return None;
        }
        // 上一轮必须是有实体+属性的知识回答，改口才有对象。
        let previous = self
            .turns
            .iter()
            .rev()
            .find(|turn| {
                turn.answer_excerpt.is_some() && turn.entity.is_some() && turn.attribute.is_some()
            })
            .cloned()?;
        let entity = previous.entity.clone()?;
        let attribute = previous.attribute.clone()?;
        // 从句式里抽出更正值：「…是 190」/「…为 190」/「…190」。
        let value = body
            .split(['是', '为'])
            .next_back()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| body.to_string());
        let previous_answer = previous.answer_excerpt.clone().unwrap_or_default();
        let turn_no = self.trace.turn_count() + 1;
        let (correction, marked) = self.trace.record_correction(
            turn_no,
            previous.index,
            &entity,
            &attribute,
            &value,
            &previous_answer,
        );
        // 更正轮也进入对话本体实例。
        self.turns.push(DialogueTurn {
            index: self.turns.len() + 1,
            user: stripped.to_string(),
            kind: MoveKind::Assert,
            entity: Some(entity.clone()),
            attribute: Some(attribute.clone()),
            answer_excerpt: Some(excerpt(&value, 40)),
            provenance: vec!["用户更正".to_string()],
            plan: vec!["trace_correction".to_string()],
            warrant: None,
            kb_revision: knowledge.revision().into(),
        });
        self.push_focus(&entity);
        let text = if marked > 0 {
            format!(
                "收到，已把「{}」的「{}」更新为「{}」，上一轮的说法作废。",
                entity, attribute, value
            )
        } else {
            format!(
                "收到，已记下「{}」的「{}」为「{}」。",
                entity, attribute, value
            )
        };
        let plan = vec!["trace_correction".to_string()];
        self.last_reply = plan.clone();
        if self.externals.last().map(String::as_str) == Some(stripped) {
            self.externals.pop();
        }
        let _ = correction;
        let reply = DialogueReply {
            text,
            plan,
            kind: MoveKind::Assert,
            derivation: None,
        };
        let frame = UtteranceFrame {
            raw: input.trim().to_string(),
            kind: MoveKind::Assert,
            entity: Some(entity),
            attribute: Some(attribute),
            question: None,
            inherited: false,
        };
        self.record_turn_trace(
            &frame,
            &frame.entity,
            Some(&reply),
            &["用户更正".to_string()],
        );
        Some(reply)
    }

    /// 把本轮决策写入分层追踪图（仅 Off 之外的模式）。
    ///
    /// 记录顺序：L0 表层 → L1 解读 → L2 指代/问据（谓词承继）→ L2 回答/让位。
    /// `reply == None` 表示让位给其它管线（记 defer 节点）。
    fn record_turn_trace(
        &mut self,
        frame: &UtteranceFrame,
        entity: &Option<String>,
        reply: Option<&DialogueReply>,
        provenance: &[String],
    ) {
        if self.trace_mode.is_off() {
            return;
        }
        // 若本轮已有节点（如更正已开启本轮），复用该轮号，避免产生空轮迹。
        let turn_no = if self
            .trace
            .turns()
            .last()
            .map(|turn| !turn.nodes.is_empty())
            .unwrap_or(false)
        {
            self.trace.current_turn()
        } else {
            self.trace.turn_count() + 1
        };
        let surface = self.trace.record_surface(turn_no, &frame.raw);
        let kind = format!("{:?}", frame.kind);
        let interpret = self.trace.record_interpret(
            turn_no,
            surface,
            &kind,
            entity.as_deref(),
            frame.attribute.as_deref(),
            frame.question.as_deref(),
            frame.inherited,
        );
        // 指代/省略登记：显式实体、代词、省略各有出处。
        if let Some(entity) = entity.as_deref() {
            let source = if frame.inherited {
                if contains_pronoun(&frame.raw) {
                    ReferentSource::Pronoun
                } else {
                    ReferentSource::Ellipsis
                }
            } else {
                ReferentSource::Explicit
            };
            let surface_form = if frame.inherited {
                frame.raw.trim().to_string()
            } else {
                entity.to_string()
            };
            self.trace.record_mention(
                turn_no,
                interpret,
                &surface_form,
                entity,
                source,
                frame.question.as_deref().unwrap_or(""),
            );
            self.trace.record_qud(
                turn_no,
                interpret,
                match frame.kind {
                    MoveKind::AskIdentity => QudKind::AskIdentity,
                    MoveKind::AskBinary => QudKind::AskBinary,
                    _ => QudKind::AskAttribute,
                },
                Some(entity),
                frame.attribute.as_deref(),
                frame.inherited,
            );
            if frame.inherited {
                if let Some(attribute) = frame.attribute.as_deref() {
                    let origin = self
                        .trace
                        .predicate_focus()
                        .unwrap_or("谓词焦点")
                        .to_string();
                    self.trace
                        .record_inheritance(turn_no, interpret, attribute, &origin);
                }
            }
        }
        match reply {
            Some(reply) => {
                let excerpt = excerpt(&reply.text, 80);
                self.trace.record_reply(
                    turn_no,
                    entity.as_deref(),
                    frame.attribute.as_deref(),
                    vec![interpret],
                    provenance,
                    Some(&excerpt),
                );
                self.trace.resolve_open_qud();
            }
            None => {
                self.trace
                    .record_defer(turn_no, interpret, "知识管线未接管本轮输入");
            }
        }
    }

    /// 复核上一轮回答：比较当时保存的担保与当前知识版本。
    ///
    /// 关键约束：删掉的事实绝不能被说成「现在仍为真」；新增冲突必须显式
    /// 披露；版本推进但事实未变时只更新「仍成立」而不制造虚假变化。
    fn review_text(&self, knowledge: &dyn DialogueKnowledge) -> (String, &'static str) {
        let Some(turn) = self.last_answered_turn() else {
            return (
                "我还没有给出过可复核的知识回答。先问我一个具体问题吧。".to_string(),
                "review_empty",
            );
        };
        let Some(entity) = turn.entity.as_deref() else {
            return (
                "上一轮没有落到具体实体，暂时无法复核。".to_string(),
                "review_empty",
            );
        };
        let attribute = turn.attribute.as_deref();
        let current_answer = if let Some(attribute) = attribute {
            self.lookup_attribute(knowledge, entity, attribute)
        } else {
            knowledge
                .answer(&format!("{}是什么", entity))
                .or_else(|| knowledge.describe(entity))
        };
        let current_warrant = self
            .quality
            .as_ref()
            .and_then(|index| index.warrant_for(entity, attribute));
        let current_revision = self.quality_revision();
        let subject = match attribute {
            Some(attribute) => format!("「{}」的「{}」", entity, attribute),
            None => format!("「{}」", entity),
        };
        if current_answer.is_none() {
            return (
                format!(
                    "复核结果：{}现在已没有记录，上一轮的说法不再作为当前事实。",
                    subject
                ),
                "review_stale",
            );
        }
        if let (Some(old), Some(new)) = (turn.warrant.as_ref(), current_warrant.as_ref()) {
            if old.known_values != new.known_values {
                let old_values = if old.known_values.is_empty() {
                    "（无）".to_string()
                } else {
                    old.known_values.join("、")
                };
                let new_values = if new.known_values.is_empty() {
                    "（无）".to_string()
                } else {
                    new.known_values.join("、")
                };
                return (
                    format!(
                        "复核结果：知识已变化。{}上一轮记录为「{}」，现在记录为「{}」，以现在为准。",
                        subject, old_values, new_values
                    ),
                    "review_changed",
                );
            }
            let new_conflicts: Vec<&crate::predicate_registry::ImportConflict> = new
                .conflicts
                .iter()
                .filter(|conflict| !old.conflicts.contains(*conflict))
                .collect();
            if !new_conflicts.is_empty() {
                let incoming = new_conflicts
                    .iter()
                    .map(|conflict| conflict.incoming.clone())
                    .collect::<Vec<_>>()
                    .join("、");
                return (
                    format!(
                        "复核结果：{}仍为「{}」，但出现了新的不同说法「{}」，需要你确认哪个为准。",
                        subject,
                        old.known_values.join("、"),
                        incoming
                    ),
                    "review_conflict",
                );
            }
            // 缺口变化也是知识变化：删掉必备属性/类型后，答案本身可能仍为真，
            // 但「当时完整、现在有缺口」必须可见，不能只说「版本推进了」。
            let conflict_gap = |gap: &&String| gap.starts_with("存在未裁决冲突");
            let new_gaps: Vec<&String> = new
                .gaps
                .iter()
                .filter(|gap| !old.gaps.contains(gap) && !conflict_gap(gap))
                .collect();
            let resolved_gaps: Vec<&String> = old
                .gaps
                .iter()
                .filter(|gap| !new.gaps.contains(gap) && !conflict_gap(gap))
                .collect();
            if !new_gaps.is_empty() || !resolved_gaps.is_empty() {
                let mut changes = Vec::new();
                if !new_gaps.is_empty() {
                    changes.push(format!(
                        "新增缺口「{}」",
                        new_gaps
                            .iter()
                            .take(2)
                            .map(|gap| gap.as_str())
                            .collect::<Vec<_>>()
                            .join("」「")
                    ));
                }
                if !resolved_gaps.is_empty() {
                    changes.push(format!(
                        "已补齐「{}」",
                        resolved_gaps
                            .iter()
                            .take(2)
                            .map(|gap| gap.as_str())
                            .collect::<Vec<_>>()
                            .join("」「")
                    ));
                }
                let still = if old.known_values.is_empty() {
                    "该回答仍成立".to_string()
                } else {
                    format!("仍为「{}」", old.known_values.join("、"))
                };
                return (
                    format!(
                        "复核结果：{}{}，但知识质量有变化：{}。事实本身不变，缺口可用于下一次补全。",
                        subject,
                        still,
                        changes.join("；")
                    ),
                    "review_gap",
                );
            }
            if current_revision != Some(old.revision) {
                return (
                    format!(
                        "复核结果：知识版本已从 {} 推进到 {}，但{}仍然成立。",
                        old.revision,
                        current_revision.unwrap_or(old.revision),
                        subject
                    ),
                    "review_confirmed",
                );
            }
            return (
                format!("复核结果：知识版本没有变化，{}仍然成立。", subject),
                "review_confirmed",
            );
        }
        let excerpt = current_answer
            .as_deref()
            .map(|answer| excerpt(answer, 60))
            .unwrap_or_default();
        (
            format!("复核结果：现有记录仍支持上一轮的回答：{}", excerpt),
            "review_confirmed",
        )
    }

    /// 回顾所用的有界窗口：当前焦点相关轮优先，最近轮兜底，并按字符预算封顶。
    pub fn recap_window(&self, budget: WindowBudget) -> WindowSelection {
        let cues: Vec<ContextCue> = self
            .turns
            .iter()
            .map(|turn| {
                let mut cue = ContextCue::new(turn.index, turn.user.clone())
                    .open(turn.answer_excerpt.is_none());
                if let Some(entity) = &turn.entity {
                    cue.focus.push(entity.clone());
                }
                if let Some(attribute) = &turn.attribute {
                    cue.predicates.push(attribute.clone());
                }
                cue
            })
            .collect();
        let focus: Vec<String> = self.focus.last().cloned().into_iter().collect();
        let predicates: Vec<String> = self
            .turns
            .last()
            .and_then(|turn| turn.attribute.clone())
            .into_iter()
            .collect();
        select_window(&cues, &focus, &predicates, budget)
    }

    /// 多轮回顾：把对话本体实例组织成一段连贯自述。
    pub fn recap_text(&self) -> String {
        if self.turns.is_empty() {
            let mut text = String::from("我们还没有正式开始知识问答。");
            if !self.externals.is_empty() {
                text.push_str(&format!(
                    "你刚才输入了 {} 条指令，但还没有聊到具体实体。",
                    self.externals.len()
                ));
            }
            text.push_str("你可以问我某个人物的属性，比如「姚明的身高是多少」。");
            return text;
        }
        let mut text = format!("我们一共聊了 {} 轮。", self.turns.len());
        let window = self.recap_window(WindowBudget::new(3, 240, 1));
        let recent: Vec<String> = self
            .turns
            .iter()
            .filter(|turn| window.keep(turn.index))
            .map(|turn| {
                let topic = match (&turn.entity, &turn.attribute) {
                    (Some(entity), Some(attribute)) => {
                        format!("「{}」的「{}」", entity, attribute)
                    }
                    (Some(entity), None) => format!("「{}」", entity),
                    _ => format!("「{}」", excerpt(&turn.user, 16)),
                };
                format!("第 {} 轮你问的是{}", turn.index, topic)
            })
            .collect();
        text.push_str(&recent.join("；"));
        text.push('。');
        if let Some(focus) = self.focus.last() {
            text.push_str(&format!("当前话题焦点是「{}」。", focus));
        }
        if !self.externals.is_empty() {
            text.push_str(&format!(
                "另外你还输入了 {} 条其他指令。",
                self.externals.len()
            ));
        }
        text
    }

    /// 失败自省：恢复模式开启且台账非空时，把失败回执讲给用户听。
    ///
    /// 绝不编造依据：没有成功回答时不假装有，只有回执可引用。
    fn failure_self_description(&self) -> Option<String> {
        if self.recovery_mode.eq(&RecoveryMode::Off) || self.recovery.ledger.is_empty() {
            return None;
        }
        let receipt = self
            .recovery
            .ledger
            .open()
            .next()
            .or_else(|| self.recovery.ledger.last())?;
        let framing = if receipt.is_open() {
            "还有一条没结清的失败回执"
        } else {
            "最近一条失败回执"
        };
        Some(format!(
            "我还没有答过带依据的问题；这里有一条失败回执——{}：{}。失败回执和成功回答一样会被记录（状态、档位、证据都可回溯），我不会为此编造依据。",
            framing,
            receipt.render_line()
        ))
    }

    /// 元话语：解释上一轮回答的依据（把推理出处「讲」出来）。
    ///
    /// 该入口不依赖知识源，保留给只读调用方；带知识源的路径见
    /// [`Self::explain_last_with`]，会优先重建一条可展开的推导。
    pub fn explain_last(&self) -> String {
        self.explain_last_inner(None).0
    }

    /// 带知识源的依据解释：先尝试从上一轮问句重建有界推导，
    /// 重建不到（或证明与回答不匹配）时诚实退回出处标签。
    pub fn explain_last_with(&self, knowledge: &dyn DialogueKnowledge) -> String {
        self.explain_last_inner(Some(knowledge)).0
    }

    fn explain_last_inner(
        &self,
        knowledge: Option<&dyn DialogueKnowledge>,
    ) -> (String, Option<Derivation>) {
        let Some(turn) = self.last_answered_turn() else {
            return (
                self.failure_self_description().unwrap_or_else(|| {
                    "我还没有答过什么问题，暂时没有可解释的依据。先问我一个知识问题吧。".to_string()
                }),
                None,
            );
        };
        let provenance = if turn.provenance.is_empty() {
            vec!["知识图谱中的事实".to_string()]
        } else {
            turn.provenance.clone()
        };
        let topic = match (&turn.entity, &turn.attribute) {
            (Some(entity), Some(attribute)) => format!("「{}」的「{}」", entity, attribute),
            (Some(entity), None) => format!("「{}」", entity),
            _ => format!("「{}」", excerpt(&turn.user, 16)),
        };
        if let Some(knowledge) = knowledge {
            if let Some(derivation) = self.explain_derivation(turn, knowledge) {
                let mut text = format!(
                    "刚才关于{}的回答，{}",
                    topic,
                    derivation.render_explanation()
                );
                if let Some(recorded) = turn.kb_revision {
                    let current = knowledge.revision();
                    if recorded != current {
                        text.push_str(&format!(
                            "\n说明：这条证明按当前知识版本 v{} 重建（回答时是 v{}）；若知识已变化，原回答仍保留在对话记忆里。",
                            current, recorded
                        ));
                    }
                }
                text.push_str(&format!("\n出处标签：{}。", provenance.join("、")));
                return (text, Some(derivation));
            }
        }
        (
            format!(
                "刚才关于{}的回答，依据是：{}。这些结论都来自知识图谱与规则程序（逐条符号匹配、逻辑推导），没有任何神经网络参与。",
                topic,
                provenance.join("、")
            ),
            None,
        )
    }

    /// 「为什么没有/查不到」：不改口，也不把失败说成不存在；
    /// 只展开最近一次未答上的知识问句所对应的失败前沿。
    pub fn explain_last_failure(&self, knowledge: &dyn DialogueKnowledge) -> String {
        self.explain_last_failure_capture(knowledge).0
    }

    fn explain_last_failure_capture(
        &self,
        knowledge: &dyn DialogueKnowledge,
    ) -> (String, Option<Derivation>) {
        let Some(turn) = self.turns.iter().rev().find(|turn| {
            turn.answer_excerpt.is_none()
                && matches!(
                    turn.kind,
                    MoveKind::AskAttribute | MoveKind::AskIdentity | MoveKind::AskBinary
                )
        }) else {
            return (
                "最近没有没答上的知识问句；上一轮是有结论的，可以问「怎么推出来的」。".to_string(),
                None,
            );
        };
        let topic = match (&turn.entity, &turn.attribute) {
            (Some(entity), Some(attribute)) => format!("「{}」的「{}」", entity, attribute),
            (Some(entity), None) => format!("「{}」", entity),
            _ => format!("「{}」", excerpt(&turn.user, 16)),
        };
        let Some(question) = turn_question(turn) else {
            return (
                format!(
                    "刚才关于{}的问题没有结论；这个问句没有可形式化的实体与属性槽位，因此没有可展开的失败前沿。",
                    topic
                ),
                None,
            );
        };
        match knowledge.derivation(&question, None) {
            Some(derivation) => {
                let mut text = format!(
                    "刚才关于{}没有作答，这里不改口：{}",
                    topic,
                    derivation.render_explanation()
                );
                if let Some(recorded) = turn.kb_revision {
                    let current = knowledge.revision();
                    if recorded != current {
                        text.push_str(&format!(
                            "\n说明：失败前沿按当前知识版本 v{} 重建（当时是 v{}）。",
                            current, recorded
                        ));
                    }
                }
                text.push_str("\n失败前沿只说明当前事实或规则的缺口，不是「它不存在」的证明。");
                (text, Some(derivation))
            }
            None => (
                format!(
                    "刚才关于{}没有作答；这个问句无法形式化为归结目标，因此没有可展开的失败前沿（属于检索或模板管线未命中）。",
                    topic
                ),
                None,
            ),
        }
    }

    /// 单步追问：只展开上一轮推导的第 `step` 步（从 1 开始）。
    pub fn explain_step(&self, knowledge: &dyn DialogueKnowledge, step: usize) -> String {
        self.explain_step_capture(knowledge, step).0
    }

    fn explain_step_capture(
        &self,
        knowledge: &dyn DialogueKnowledge,
        step: usize,
    ) -> (String, Option<Derivation>) {
        let Some(derivation) = self.last_derivation(knowledge) else {
            return (
                "上一轮没有可展开的推导，暂时不能按步骤解释。可以先问「怎么推出来的」。"
                    .to_string(),
                None,
            );
        };
        let total = derivation.steps.len();
        let text = match derivation.render_step(step) {
            Some(line) => {
                let mut text = line;
                if derivation.alternatives > 0 {
                    text.push_str(&format!(
                        "\n完整证明共 {} 步；另有 {} 条独立证明未展开。",
                        total, derivation.alternatives
                    ));
                } else {
                    text.push_str(&format!("\n完整证明共 {} 步。", total));
                }
                if derivation.truncated || derivation.omitted_steps > 0 {
                    text.push_str("部分节点因预算未展开，步骤数可能少于真实证明。");
                }
                text
            }
            None => format!(
                "上一轮推导只展开了 {} 步，没有第 {} 步；可以问「怎么推出来的」查看全貌。",
                total, step
            ),
        };
        (text, Some(derivation))
    }

    /// 只读：为最近一条已答轮次重建推导（与 [`Self::explain_last_with`] 同源）。
    /// 供 HTTP/CLI 在「为什么」轮次返回结构化证明，不写入会话状态。
    pub fn last_derivation(&self, knowledge: &dyn DialogueKnowledge) -> Option<Derivation> {
        let turn = self.last_answered_turn()?;
        self.explain_derivation(turn, knowledge)
    }

    /// 为上一轮回答重建一条与实际答案锚定的推导：
    /// 先走知识源的归结证明；没有形式证明时，再按实体属性/关系边
    /// 组装直接事实链（只取回答文本中确实出现的取值）。
    fn explain_derivation(
        &self,
        turn: &DialogueTurn,
        knowledge: &dyn DialogueKnowledge,
    ) -> Option<Derivation> {
        let question = turn_question(turn);
        let expected = turn.answer_excerpt.as_deref();
        if let Some(question) = question.as_deref() {
            if let Some(derivation) = knowledge.derivation(question, expected) {
                if derivation.is_proved() {
                    return Some(derivation);
                }
            }
        }
        self.direct_fact_derivation(turn, knowledge)
    }

    fn direct_fact_derivation(
        &self,
        turn: &DialogueTurn,
        knowledge: &dyn DialogueKnowledge,
    ) -> Option<Derivation> {
        let entity = turn.entity.as_deref()?;
        let attribute = turn.attribute.as_deref()?;
        let expected = turn.answer_excerpt.as_deref()?;
        let mut facts: Vec<(String, String, String)> = Vec::new();

        for value in knowledge.attribute_values(entity, attribute) {
            if !value.is_empty() && expected.contains(value.as_str()) {
                facts.push((
                    entity.to_string(),
                    surface_attribute(attribute).to_string(),
                    value,
                ));
            }
        }
        let canonical = knowledge
            .canonical_predicate(attribute)
            .unwrap_or_else(|| attribute.to_string());
        for (predicate, object) in knowledge.entity_relations(entity) {
            if object.is_empty() || !expected.contains(object.as_str()) {
                continue;
            }
            let label = knowledge.predicate_label(&predicate);
            if predicate == attribute || label == attribute || predicate == canonical {
                facts.push((entity.to_string(), label, object));
            }
        }
        facts.sort();
        facts.dedup();
        if facts.is_empty() {
            return None;
        }
        Some(Derivation::from_direct_facts(
            format!("{} 的 {}", entity, surface_attribute(attribute)),
            facts,
        ))
    }

    /// 单条属性的结构化作答：先走推导链/预测，再退到字面量取值检索。
    fn lookup_attribute(
        &self,
        knowledge: &dyn DialogueKnowledge,
        entity: &str,
        attribute: &str,
    ) -> Option<String> {
        let question = format!("{}的{}是什么", entity, attribute);
        let answer = if self.recovery_mode.acts() {
            knowledge.answer_certain(&question)
        } else {
            knowledge.answer(&question)
        };
        if let Some(answer) = answer {
            return Some(answer);
        }
        let values = knowledge.attribute_values(entity, attribute);
        if values.is_empty() {
            None
        } else {
            Some(format!(
                "「{}」的「{}」是{}。",
                entity,
                attribute,
                values.join("、")
            ))
        }
    }

    fn continue_text(&self, entity: Option<&str>, knowledge: &dyn DialogueKnowledge) -> String {
        let Some(entity) = entity else {
            return "先给我一个实体名，我们就有话题可继续了。".to_string();
        };
        let attributes = knowledge.known_attributes(entity);
        for attribute in attributes.iter().filter(|attribute| {
            !self.turns.iter().any(|turn| {
                turn.entity.as_deref() == Some(entity)
                    && turn.attribute.as_deref() == Some(attribute.as_str())
            })
        }) {
            if let Some(answer) = self.lookup_attribute(knowledge, entity, attribute) {
                return format!(
                    "接着说说「{}」的「{}」：{}",
                    entity,
                    attribute,
                    answer.trim()
                );
            }
        }
        format!(
            "关于「{}」我知道的都讲得差不多了；换个实体，或者直接考我一个问题吧。",
            entity
        )
    }

    fn elaborate_text(&self, entity: Option<&str>, knowledge: &dyn DialogueKnowledge) -> String {
        let Some(entity) = entity else {
            return "先告诉我你想展开讲哪个实体。".to_string();
        };
        let mut parts = Vec::new();
        for attribute in knowledge.known_attributes(entity).iter().take(3) {
            if let Some(answer) = self.lookup_attribute(knowledge, entity, attribute) {
                parts.push(format!(
                    "{}：{}",
                    surface_attribute(attribute),
                    answer.trim()
                ));
            }
        }
        if parts.is_empty() {
            return format!("关于「{}」我暂时没有可以展开的细节。", entity);
        }
        let body = parts
            .iter()
            .map(|part| format!("  • {}", part))
            .collect::<Vec<_>>()
            .join("\n");
        format!("关于「{}」，我知道这些：\n{}", entity, body)
    }

    fn clarify_text(&self, attribute: Option<&str>) -> String {
        let attribute = attribute.unwrap_or("这个属性");
        let candidates: Vec<String> = self.focus.iter().rev().take(3).cloned().collect();
        match candidates.split_last() {
            Some((last, rest)) if !rest.is_empty() => format!(
                "你想问的「{}」是指「{}」还是「{}」？告诉我名字我就接着答。",
                attribute,
                rest.join("」「"),
                last
            ),
            _ => format!("你想问谁的「{}」？先告诉我实体名吧。", attribute),
        }
    }

    /// 把质量担保翻译成话语行为：冲突披露 → 类型澄清 → 一个缺口 → 引用方式。
    ///
    /// 顺序固定，最多追加四条（每类至多一条），因此回答不会因质量问题无限膨胀。
    /// 担保只改「怎么说」，不改答案本身，也不把低分/缺来源说成事实为假。
    fn apply_warrant(
        &self,
        mut text: String,
        mut plans: Vec<String>,
        warrant: Option<&ClaimWarrant>,
        kind: &MoveKind,
        answered: bool,
    ) -> (String, Vec<String>) {
        let Some(warrant) = warrant else {
            return (text, plans);
        };
        let ask_like = matches!(
            kind,
            MoveKind::AskAttribute | MoveKind::AskIdentity | MoveKind::AskBinary
        );
        if !ask_like {
            return (text, plans);
        }
        if let Some(note) = warrant.conflict_note() {
            text.push('\n');
            text.push_str(&note);
            plans.push("quality_disclose_conflict".to_string());
        }
        if warrant.needs_type_clarification() && matches!(kind, MoveKind::AskIdentity) {
            text.push_str("\n另外，我还不能确定它的类型（人物/地点/组织等），你能补充一下吗？");
            plans.push("quality_clarify_type".to_string());
        }
        if let Some(gap) = warrant.actionable_gap() {
            text.push_str(&format!("\n补充线索：{}。", gap));
            plans.push("quality_offer_gap".to_string());
        }
        if answered {
            text.push_str(&warrant.citation_note());
            plans.push(if warrant.has_source() {
                "quality_cite_source".to_string()
            } else {
                "quality_no_source".to_string()
            });
        }
        (text, plans)
    }

    fn compose_direct(&self, answer: &str, entity: &str, more: &[String]) -> String {
        let mut text = answer.trim().to_string();
        if !more.is_empty() {
            let listed = more
                .iter()
                .take(3)
                .map(|attribute| surface_attribute(attribute).to_string())
                .collect::<Vec<_>>()
                .join("」「");
            let variants = [
                format!(
                    "我还知道它的「{}」等 {} 项信息，想了解可以直接问。",
                    listed,
                    more.len()
                ),
                format!(
                    "关于「{}」，我还能补充「{}」等 {} 项。",
                    entity,
                    listed,
                    more.len()
                ),
                format!(
                    "除此之外，「{}」的「{}」我也知道（共 {} 项），继续问我就好。",
                    entity,
                    listed,
                    more.len()
                ),
            ];
            text.push('\n');
            text.push_str(&variants[self.turns.len() % variants.len()]);
        }
        text
    }

    fn compose_ignorance(&self, entity: &str, attribute: Option<&str>, more: &[String]) -> String {
        let mut text = match attribute {
            Some(attribute) => format!(
                "关于「{}」的「{}」，我的知识图谱里暂时没有记录。",
                entity, attribute
            ),
            None => format!("我的知识图谱里暂时没有关于「{}」的介绍信息。", entity),
        };
        if !more.is_empty() {
            let listed = more
                .iter()
                .take(3)
                .map(|attribute| surface_attribute(attribute).to_string())
                .collect::<Vec<_>>()
                .join("」「");
            text.push_str(&format!(
                "\n不过我知道它的「{}」等 {} 项信息，你想了解哪一个？",
                listed,
                more.len()
            ));
        } else {
            text.push_str("\n你可以换一种问法，或者直接告诉我相关事实，我会把它记住。");
        }
        text
    }

    /// 选策略 → 选 spec 变体 → 绑定槽位；未外化的策略返回 None。
    ///
    /// 槽位值必须与组合生成器使用的是同一批运行时值，
    /// 这样 Parity 模式的逐字比对才有意义（差异 = 模板资产的真差异）。
    fn prompt_bindings(
        &self,
        policy: &str,
        frame: &UtteranceFrame,
        entity: Option<&str>,
        answer: Option<&str>,
        more: &[String],
    ) -> Option<(&'static str, BindingSet)> {
        let listed = || {
            more.iter()
                .take(3)
                .map(|attribute| surface_attribute(attribute).to_string())
                .collect::<Vec<_>>()
                .join("」「")
        };
        match policy {
            "direct_answer" => {
                if more.is_empty() {
                    let mut bindings = BindingSet::new();
                    if let Some(answer) = answer {
                        bindings.bind("answer", answer.trim());
                    }
                    Some(("direct_answer/plain", bindings))
                } else {
                    let ids = [
                        "direct_answer/more/0",
                        "direct_answer/more/1",
                        "direct_answer/more/2",
                    ];
                    let id = ids[self.turns.len() % ids.len()];
                    let mut bindings = BindingSet::new();
                    if let Some(answer) = answer {
                        bindings.bind("answer", answer.trim());
                    }
                    bindings.bind("entity", entity.unwrap_or(""));
                    bindings.bind("listed", &listed());
                    bindings.bind("count", &more.len().to_string());
                    Some((id, bindings))
                }
            }
            "admit_ignorance" => {
                let id = match (frame.attribute.is_some(), more.is_empty()) {
                    (true, true) => "admit_ignorance/attribute/plain",
                    (true, false) => "admit_ignorance/attribute/more",
                    (false, true) => "admit_ignorance/identity/plain",
                    (false, false) => "admit_ignorance/identity/more",
                };
                let mut bindings = BindingSet::new();
                bindings.bind("entity", entity.unwrap_or(""));
                if let Some(attribute) = frame.attribute.as_deref() {
                    bindings.bind("attribute", attribute);
                }
                if !more.is_empty() {
                    bindings.bind("listed", &listed());
                    bindings.bind("count", &more.len().to_string());
                }
                Some((id, bindings))
            }
            "clarify_entity" => {
                let candidates: Vec<String> = self.focus.iter().rev().take(3).cloned().collect();
                match candidates.split_last() {
                    Some((last, rest)) if !rest.is_empty() => {
                        let mut bindings = BindingSet::new();
                        if let Some(attribute) = frame.attribute.as_deref() {
                            bindings.bind("attribute", attribute);
                        }
                        bindings.bind("candidates", &rest.join("」「"));
                        bindings.bind("last", last);
                        Some(("clarify_entity/multi", bindings))
                    }
                    _ => {
                        let mut bindings = BindingSet::new();
                        if let Some(attribute) = frame.attribute.as_deref() {
                            bindings.bind("attribute", attribute);
                        }
                        Some(("clarify_entity/single", bindings))
                    }
                }
            }
            "clarify_entity_no_focus" => {
                let mut bindings = BindingSet::new();
                if let Some(attribute) = frame.attribute.as_deref() {
                    bindings.bind("attribute", attribute);
                }
                Some(("clarify_entity_no_focus/plain", bindings))
            }
            "greet" => {
                if self.turns.is_empty() {
                    Some(("greet/first", BindingSet::new()))
                } else if let Some(focus) = self.focus.last() {
                    let mut bindings = BindingSet::new();
                    bindings.bind("focus", focus);
                    Some(("greet/returning_focus", bindings))
                } else {
                    Some(("greet/returning_plain", BindingSet::new()))
                }
            }
            "thanks" => match self.focus.last() {
                Some(focus) => {
                    let mut bindings = BindingSet::new();
                    bindings.bind("focus", focus);
                    Some(("thanks/focus", bindings))
                }
                None => Some(("thanks/plain", BindingSet::new())),
            },
            "farewell" => {
                let mut bindings = BindingSet::new();
                bindings.bind("turn_count", &self.turns.len().to_string());
                Some(("farewell/plain", bindings))
            }
            "why_no_evidence" => Some(("why_no_evidence/plain", BindingSet::new())),
            "continue_no_focus" => Some(("continue_no_focus/plain", BindingSet::new())),
            "explain_why" => {
                if frame.question.as_deref() == Some("why_not")
                    || frame
                        .question
                        .as_deref()
                        .is_some_and(|question| question.starts_with("step:"))
                {
                    return None;
                }
                let turn = self.last_answered_turn()?;
                let provenance = if turn.provenance.is_empty() {
                    vec!["知识图谱中的事实".to_string()]
                } else {
                    turn.provenance.clone()
                };
                let topic = match (&turn.entity, &turn.attribute) {
                    (Some(entity), Some(attribute)) => format!("「{}」的「{}」", entity, attribute),
                    (Some(entity), None) => format!("「{}」", entity),
                    _ => format!("「{}」", excerpt(&turn.user, 16)),
                };
                let mut bindings = BindingSet::new();
                bindings.bind("topic", &topic);
                bindings.bind("provenance", &provenance.join("、"));
                Some(("explain_why/plain", bindings))
            }
            _ => None,
        }
    }
}

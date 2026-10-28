impl SymbolicReasoner {
    /// 属性倒排反查：给定属性槽位与取值，返回属性值等于它的实体。
    ///
    /// literal 属性（身高/年龄/国籍…）不会成为关系边，因此 `predictor`
    /// 里查不到，只能扫 `attribute_index`。槽位键先用对齐后的规范词面，
    /// 再回退英文属性键对照表（与 `node_hits` 的正向回退同一套规则）。
    fn attribute_subjects(&self, slot: &str, surface: &str, value: &str) -> Vec<String> {
        let value = value.trim();
        if slot.is_empty() || value.is_empty() {
            return Vec::new();
        }
        let mut keys = vec![slot.to_string()];
        if let Some(canonical) = crate::symbolic_generator::canonical_predicate_key(surface) {
            if canonical != slot {
                keys.push(canonical.to_string());
            }
        }
        let mut subjects: Vec<String> = self
            .attribute_index
            .iter()
            .filter(|((_, key), values)| {
                keys.iter().any(|candidate| candidate == key)
                    && values.iter().any(|stored| stored.trim() == value)
            })
            .map(|((entity, _), _)| entity.clone())
            .collect();
        subjects.sort();
        subjects.dedup();
        subjects
    }

    fn answer_relation_objects(&self, subject_raw: &str, predicate_raw: &str) -> Option<String> {
        self.answer_relation_objects_filtered(subject_raw, predicate_raw, &[])
    }

    fn answer_relation_objects_filtered(
        &self,
        subject_raw: &str,
        predicate_raw: &str,
        excluded: &[String],
    ) -> Option<String> {
        let entity = self.aligner.canonical_form(subject_raw.trim());
        let (slot, inverted) = self.resolve_question_predicate(predicate_raw);
        let surface = self.display_predicate_surface(predicate_raw);
        if entity.is_empty() || slot.is_empty() {
            return None;
        }
        if !self.is_known_entity(&entity) || !self.is_known_predicate(&surface) {
            return None;
        }
        let excluded_display: Vec<String> = excluded.to_vec();
        let excluded_set: HashSet<String> = excluded
            .iter()
            .map(|item| self.aligner.canonical_form(item.trim()))
            .collect();
        let predicate = ChainPredicate {
            surface: surface.clone(),
            slot,
            inverted,
        };
        let mut hits = self.node_hits(&entity, &predicate);
        hits.sort_by(|left, right| left.value.cmp(&right.value));
        hits.dedup_by(|left, right| left.value == right.value);
        let had_hits = !hits.is_empty();
        if !excluded_set.is_empty() {
            hits.retain(|hit| !excluded_set.contains(&hit.value));
        }
        if hits.is_empty() {
            // 直接命中为空时，规则推导的取值同样可以回答「A 的 P 是谁」，
            // 但只返回本次推导与依据，不把新取值写进图谱。
            if excluded_set.is_empty() {
                if let Some(answer) =
                    self.answer_by_derivation(&entity, std::slice::from_ref(&predicate), "")
                {
                    return Some(answer);
                }
            }
            if had_hits && !excluded_display.is_empty() {
                return Some(format!(
                    "根据现有知识库，除「{}」外，没有其他记录「{}{}…」。",
                    excluded_display.join("、"),
                    entity,
                    surface
                ));
            }
            return Some(format!(
                "根据现有知识库，没有记录「{}{}…」。",
                entity, surface
            ));
        }
        let values: Vec<String> = hits.iter().map(|hit| hit.value.clone()).collect();
        let mut output = if inverted {
            format!(
                "根据知识图谱：「{}」的「{}」是：{}。",
                entity,
                surface,
                values.join("、")
            )
        } else if values.len() == 1 {
            format!("根据知识图谱：{}{}{}。", entity, surface, values[0])
        } else {
            format!(
                "根据知识图谱：{}{}的有：{}。",
                entity,
                surface,
                values.join("、")
            )
        };
        if !excluded_display.is_empty() {
            output.push_str(&format!("\n已排除：{}。", excluded_display.join("、")));
        }
        if let Some(prediction) = hits.iter().find_map(|hit| hit.prediction.as_ref()) {
            output.push_str(&format!(
                "\n其中为符号推断结果（例如置信度 {:.2}）：{}",
                prediction.confidence,
                self.predictor.explain(prediction)
            ));
        }
        Some(output)
    }

    fn answer_check_relation(
        &self,
        subject_raw: &str,
        predicate_raw: &str,
        object_raw: &str,
        negated: bool,
    ) -> Option<String> {
        let subject = self.aligner.canonical_form(subject_raw.trim());
        let object = self.aligner.canonical_form(object_raw.trim());
        let (slot, inverted) = self.resolve_question_predicate(predicate_raw);
        let surface = self.display_predicate_surface(predicate_raw);
        if subject.is_empty() || object.is_empty() || slot.is_empty() {
            return None;
        }
        if !self.is_known_entity(&subject)
            || !self.is_known_entity(&object)
            || !self.is_known_predicate(&surface)
        {
            return None;
        }
        let (negative_subject, negative_object) = if inverted {
            (object.as_str(), subject.as_str())
        } else {
            (subject.as_str(), object.as_str())
        };
        if self.session_negative_blocks(negative_subject, &slot, negative_object) {
            return Some(if negated {
                format!(
                    "是的。会话中记录了否定事实：{} {} {}。",
                    subject, surface, object
                )
            } else {
                format!(
                    "不是。会话中记录了否定事实：{} {} {}。",
                    subject, surface, object
                )
            });
        }
        let holds = if inverted {
            self.predictor.has_triple(&object, &slot, &subject)
        } else {
            self.predictor.has_triple(&subject, &slot, &object)
        };
        if !holds {
            let predicate = ChainPredicate {
                surface: surface.clone(),
                slot: slot.clone(),
                inverted,
            };
            if let Some(fact) = self.derived_fact_for(&subject, &predicate, &object) {
                let verdict = if negated { "不是" } else { "是的" };
                return Some(format!(
                    "{}。根据当前规则现算的推导（未写入知识图谱），该关系成立：{} {} {}。\n{}",
                    verdict,
                    subject,
                    surface,
                    object,
                    self.render_derived_support(&fact)
                ));
            }
        }
        if negated {
            return Some(if holds {
                format!(
                    "不是。根据知识图谱，该关系成立：{} {} {}。",
                    subject, surface, object
                )
            } else {
                format!(
                    "是的。根据现有知识图谱，没有记录「{} {} {}」，与否定问法一致。",
                    subject, surface, object
                )
            });
        }
        if holds {
            return Some(format!(
                "是的。根据知识图谱，该关系成立：{} {} {}。",
                subject, surface, object
            ));
        }
        Some(format!(
            "不是。根据现有知识图谱，没有记录「{} {} {}」。",
            subject, surface, object
        ))
    }

    fn answer_find_subject(&self, object_raw: &str, predicate_raw: &str) -> Option<String> {
        self.answer_find_subject_filtered(object_raw, predicate_raw, &[])
    }

    fn answer_find_subject_filtered(
        &self,
        object_raw: &str,
        predicate_raw: &str,
        excluded: &[String],
    ) -> Option<String> {
        let object = self.aligner.canonical_form(object_raw.trim());
        let (slot, inverted) = self.resolve_question_predicate(predicate_raw);
        if object.is_empty() || slot.is_empty() {
            return None;
        }
        let surface = self.display_predicate_surface(predicate_raw);
        let excluded_display: Vec<String> = excluded.to_vec();
        let excluded_set: HashSet<String> = excluded
            .iter()
            .map(|item| self.aligner.canonical_form(item.trim()))
            .collect();
        // 反查主体：正向找槽位三元组的主语；逆向词面（如「徒弟」）语义即
        // "谁的槽位值是 object"，等价于找槽位三元组的宾语（object 作为主语）
        let mut subjects: Vec<String> = if inverted {
            self.predictor.objects_of(&object, &slot)
        } else {
            self.predictor.subjects_of(&slot, &object)
        };
        // 实体属性（身高/年龄/国籍…）不进关系三元组索引，literal 取值只能
        // 走属性倒排反查；没有这一步，「谁的身高是226」永远查不到人。
        if !inverted {
            subjects.extend(self.attribute_subjects(&slot, &surface, &object));
        }
        subjects.sort();
        subjects.dedup();
        let mut evidence: Vec<String> = Vec::new();
        if !inverted && self.config.enable_link_prediction {
            let limit = self.config.max_predictions.saturating_mul(4).max(4);
            // 用哈希集合做成员检查
            let mut seen_subjects: HashSet<String> = subjects.iter().cloned().collect();
            for prediction in self.predictions_for_entity(&object, Some(&slot), limit) {
                if prediction.triple.object == object
                    && prediction.triple.subject != object
                    && !excluded_set.contains(&prediction.triple.subject)
                {
                    if seen_subjects.insert(prediction.triple.subject.clone()) {
                        subjects.push(prediction.triple.subject.clone());
                    }
                    evidence.push(self.predictor.explain(&prediction));
                }
            }
        }
        if !excluded_set.is_empty() {
            subjects.retain(|subject| !excluded_set.contains(subject));
        }
        if subjects.is_empty() {
            if !excluded_display.is_empty()
                && self.is_known_entity(&object)
                && self.is_known_predicate(&surface)
            {
                return Some(format!(
                    "根据现有知识库，除「{}」外，没有其他记录「谁的 {} 是 {}」。",
                    excluded_display.join("、"),
                    surface,
                    object
                ));
            }
            if self.is_known_entity(&object) && self.is_known_predicate(&surface) {
                return Some(format!(
                    "根据现有知识库，没有记录「谁的 {} 是 {}」。",
                    surface, object
                ));
            }
            return None;
        }
        subjects.sort();
        let mut output = format!(
            "根据知识图谱：{} 的 {} 是 {}。",
            subjects.join("、"),
            surface,
            object
        );
        if !excluded_display.is_empty() {
            output.push_str(&format!("\n已排除：{}。", excluded_display.join("、")));
        }
        if !evidence.is_empty() {
            output.push_str(&format!("\n推理依据：{}", evidence.join("；")));
        }
        Some(output)
    }

    fn answer_similar(&self, entity_raw: &str) -> Option<String> {
        let entity = self.aligner.canonical_form(entity_raw.trim());
        if entity.is_empty() || !self.embedding.is_embedded(&entity) {
            return None;
        }
        let neighbors = self.embedding.nearest(&entity, 3);
        if neighbors.is_empty() {
            return None;
        }
        let mut output = format!(
            "符号嵌入近邻检索（PPMI 角色向量，无神经网络）：与「{}」语义最接近的实体：\n",
            entity
        );
        for (index, (name, _)) in neighbors.iter().enumerate() {
            output.push_str(&format!(
                "  {}. {}\n",
                index + 1,
                self.embedding.explain_similarity(&entity, name)
            ));
        }
        Some(output.trim_end().to_string())
    }

    fn answer_analogy(&self, a_raw: &str, b_raw: &str, c_raw: &str) -> Option<String> {
        let a = self.aligner.canonical_form(a_raw.trim());
        let b = self.aligner.canonical_form(b_raw.trim());
        let c = self.aligner.canonical_form(c_raw.trim());
        if a.is_empty() || b.is_empty() || c.is_empty() {
            return None;
        }
        let results = self.embedding.analogy(&a, &b, &c, 3);
        if results.is_empty() {
            return None;
        }
        let direction = self.embedding.describe_direction(&a, &b);
        let mut output = format!(
            "向量算术类比（{} : {} :: {} : ?）：\n  从「{}」到「{}」的语义方向是{}，应用到「{}」：\n",
            a, b, c, a, b,
            if direction.is_empty() {
                String::new()
            } else {
                format!("「{}」", direction)
            },
            c
        );
        for (index, (name, score)) in results.iter().enumerate() {
            let shared = self.embedding.shared_feature_labels(&b, name, 2);
            let shared_text = if shared.is_empty() {
                String::new()
            } else {
                format!("，与「{}」共享 {}", b, shared.join("、"))
            };
            output.push_str(&format!(
                "  {}. 「{}」（语义偏移相似度 {:.2}{}）\n",
                index + 1,
                name,
                score,
                shared_text
            ));
        }
        Some(output.trim_end().to_string())
    }

    pub fn answer(&self, question: &str) -> Option<String> {
        self.semantic.begin(question);
        let rewritten = self.rewrite_predicate_alignment(question);
        let normalized = rewritten.as_deref().unwrap_or(question);
        let answer = self
            .answer_impl(normalized)
            .map(|answer| self.annotate_assumption_world(answer));
        if answer.is_none() {
            // 只观测失败：成功轮不做额外分词，回答内容与原有路径逐字节一致。
            self.record_read_failure(question);
        }
        self.finish_semantic_turn(if answer.is_some() {
            TurnOutcome::Answered
        } else {
            TurnOutcome::Deferred
        });
        answer
    }

    /// 只要可确证的回答：不把链接预测列表当作确定答案。
    ///
    /// 对话层用它与恢复阶梯衔接：确证失败后由 L2 提名、验证、追问，
    /// 而不是把候选列表直接当结论输出。
    pub fn answer_certain(&self, question: &str) -> Option<String> {
        self.semantic.begin(question);
        let answer = self.answer_certain_impl(question);
        self.finish_semantic_turn(if answer.is_some() {
            TurnOutcome::Answered
        } else {
            TurnOutcome::Deferred
        });
        answer
    }

    fn finish_semantic_turn(
        &self,
        outcome: TurnOutcome,
    ) -> Option<crate::semantic_monitor::SemanticObservation> {
        let observation = self.semantic.finish(outcome)?;
        self.perf.bump("semantic_observations");
        self.perf.bump(match outcome {
            TurnOutcome::Answered => "semantic_outcome_answered",
            TurnOutcome::Deferred => "semantic_outcome_deferred",
            TurnOutcome::Clarified => "semantic_outcome_clarified",
            TurnOutcome::Fault => "semantic_outcome_fault",
        });
        self.perf
            .bump_by("semantic_search_attempts", observation.search.searches);
        self.perf
            .bump_by("semantic_search_hits", observation.search.returned);
        self.perf
            .bump_by("semantic_assemblies", observation.assembly.assemblies);
        self.perf
            .bump_by(
                "semantic_assembly_selected",
                observation.assembly.selected_assemblies,
            );
        for code in &observation.codes {
            self.perf.bump(&format!("semantic_code_{code}"));
        }
        Some(observation)
    }

    fn answer_certain_impl(&self, question: &str) -> Option<String> {
        let rewritten = self.rewrite_predicate_alignment(question);
        let question = rewritten.as_deref().unwrap_or(question);
        if let Some(answer) = self.answer_learned_rewrite(question) {
            return Some(self.annotate_assumption_world(answer));
        }
        // 情态问句的答案不是「确证事实」，但它把不确定性与阈值显式写在脸上；
        // 对话恢复层（answer_certain）同样必须看得见它，否则「可能」会被
        // 悄悄降级成普通确证回答。
        if let Some(answer) = self.answer_probabilistic_intent(question) {
            return Some(answer);
        }
        if let Some(answer) = crate::type_memory::answer_with_schema(
            &self.generator,
            Some(self.schema_view()),
            question,
        ) {
            return Some(self.annotate_assumption_world(answer.text));
        }
        if let Some(answer) = crate::entity_inventory::answer(&self.generator, question) {
            return Some(self.annotate_assumption_world(answer));
        }
        if let Some(answer) = self.answer_planned(question) {
            if !answer.contains("置信度") && !answer.contains("链接预测") {
                return Some(self.annotate_assumption_world(answer));
            }
        }
        if let Some(answer) = self.answer_graph_question(question) {
            return Some(self.annotate_assumption_world(answer));
        }
        if let Some(answer) = self.answer_by_evidence(question) {
            return Some(self.annotate_assumption_world(answer));
        }
        self.answer_contextual(question)
            .or_else(|| self.formal_semantics_answer(question))
            .or_else(|| self.generator.answer_import_frame(question))
            .map(|answer| self.annotate_assumption_world(answer))
    }

    /// 是否还有未答之问（悬问）。
    pub fn has_pending_question(&self) -> bool {
        self.pending_questions.has_active()
    }

    /// 悬问台账报告：列出「哪个目标缺哪条前提」。
    pub fn pending_question_report(&self) -> String {
        self.pending_questions.render_report()
    }

    /// 原问句已在正常路径上得到确证回答：关闭同题悬问（不影响其他悬问）。
    pub fn dismiss_pending_question(&mut self, question: &str) -> bool {
        self.pending_questions.dismiss_question(question)
    }

    /// 显式放弃全部悬问。忘记必须是显式动作，不设 TTL。
    pub fn abandon_pending_questions(&mut self) -> String {
        let count = self.pending_questions.len() + self.pending_questions.shelved_len();
        self.pending_questions.clear();
        if count == 0 {
            "悬问台账本来就是空的。".to_string()
        } else {
            format!(
                "已放弃 {} 条悬问；原问题不会自动重开，相关知识仍留在图谱与会话事实里。",
                count
            )
        }
    }

    /// 显式重开用户先前说「不知道」而搁置的悬问，并重新追问当前洞。
    pub fn reopen_pending_questions(&mut self) -> String {
        let count = self.pending_questions.reopen_shelved();
        if count == 0 {
            return "没有搁置的悬问；当前台账里的问题仍在追问中。".to_string();
        }
        let Some(pending) = self.pending_questions.active().cloned() else {
            return "已恢复搁置的悬问。".to_string();
        };
        self.pending_questions.mark_asked(pending.id);
        format!(
            "已重新打开 {} 条搁置的悬问。\n{}",
            count,
            PendingQuestionLedger::render_ask(&pending)
        )
    }

    /// 本轮没有把输入当作短答：关闭「仅紧接一轮可裸答」的窗口。
    pub fn clear_pending_awaiting(&mut self) {
        self.pending_questions.clear_awaiting();
    }

    /// 回答悬问：把本轮输入绑定到当前追问的洞，补入会话事实后重解原目标。
    ///
    /// 只有在系统上一轮刚追问过某个洞时才接受裸取值；其他轮次只接受
    /// 「回答悬问：…」这样的显式回填。无法判定为回答时返回 `None`，
    /// 让位给正常对话管线（绝不劫持新问句/命令）。
    pub fn try_answer_pending_question(&mut self, input: &str) -> Option<String> {
        let pending = self.pending_questions.active()?.clone();
        let binding = pending_question::interpret_answer(input, &pending.hole);
        let explicit = pending_question::has_explicit_answer_prefix(input);
        if !self.pending_questions.awaiting_answer() && !explicit {
            self.pending_questions.clear_awaiting();
            return None;
        }
        match binding {
            AnswerBinding::NotAnAnswer => {
                self.pending_questions.clear_awaiting();
                None
            }
            AnswerBinding::Unknown => {
                self.pending_questions.suspend(pending.id);
                self.pending_questions.clear_awaiting();
                Some(format!(
                    "好，「{}」先悬着并移入搁置区。\n「不知道」只是暂时没有信息，不会被记成否定事实；输入「重开悬问」可以继续追问。",
                    pending.hole.phrase()
                ))
            }
            AnswerBinding::Fill(object) => {
                let adopted = self.adopt_hole_fill(&pending.hole, &object);
                self.pending_questions.remove(pending.id);
                self.pending_questions.clear_awaiting();
                // 延期回答只走可确证的计划路径：证据索引的邻近事实、
                // 链接预测候选与上下文叙述都不能冒充原目标的答案。
                let answer = self
                    .answer_planned(&pending.question)
                    .filter(|text| !Self::is_pending_failure(text));
                let Some(answer) = answer else {
                    // 补上一条前提仍不足以证明：按当前失败前沿继续找下一个洞。
                    if let Some(prompt) = self.open_pending_question(&pending.question) {
                        return Some(format!(
                            "已记住「{}」（会话事实，不写知识图谱）。\n原问题还差前提：\n{}",
                            pending.hole.fill_sentence(&object),
                            prompt
                        ));
                    }
                    return Some(format!(
                        "收到「{}」，但原问题仍无法证明；没有把推测当作答案。",
                        pending.hole.fill_sentence(&object)
                    ));
                };
                let mut output = format!(
                    "你刚才问的「{}」，补上「{}」后现在可以回答了：\n{}",
                    pending.question,
                    pending.hole.fill_sentence(&object),
                    answer
                );
                let fill_sentence = pending.hole.fill_sentence(&object);
                if !answer.contains("路径")
                    && !answer.contains("依据")
                    && !answer.contains(&fill_sentence)
                {
                    let mut support = pending.support.clone();
                    support.push(fill_sentence);
                    output.push_str(&format!("\n依据：{}。", support.join("；")));
                }
                if !adopted
                    && !self
                        .session_facts
                        .iter()
                        .any(|atom| atom.terms.iter().any(|term| term.name() == object))
                {
                    output.push_str("\n（该取值已在图谱或已有会话事实中，未重复写入。）");
                }
                Some(output)
            }
        }
    }

    /// 失败目标 → 悬问：从链式目标的第一个缺口生成一条「目标 + 洞」。
    ///
    /// 单跳缺边与多跳缺中间前提同样处理；只在回答确实失败时调用，
    /// 返回追问话术并写入悬问账本。
    pub fn open_pending_question(&mut self, question: &str) -> Option<String> {
        let (goal, hole, support) = self.diagnose_chain_hole(question)?;
        let pending = self
            .pending_questions
            .record(question, &goal, hole, support);
        self.pending_questions.mark_asked(pending.id);
        Some(PendingQuestionLedger::render_ask(&pending))
    }

    /// 任何新断言到达后，重放悬问：原目标已可证则给出延期回答并关闭悬问。
    ///
    /// 不新增事实，也不把链接预测候选当成答案；只使用可确证的证明路径。
    pub fn resolve_pending_after_assertion(&mut self) -> Option<String> {
        if !self.pending_questions.has_active() {
            return None;
        }
        let pending = self.pending_questions.questions().to_vec();
        let mut unresolved: Vec<PendingQuestion> = Vec::new();
        let mut resolved: Vec<String> = Vec::new();
        for question in pending {
            // 只用原目标的可确证计划回答，避免把「证据索引」的邻近事实
            // 伪装成延期回答；候选列表也不进入此路径。
            match self.answer_planned(&question.question) {
                Some(answer) if !Self::is_pending_failure(&answer) => {
                    resolved.push(format!(
                        "你之前问的「{}」，现在可以回答了：\n{}",
                        question.question, answer
                    ));
                }
                _ => unresolved.push(question),
            }
        }
        self.pending_questions.clear_awaiting();
        self.pending_questions.replace(unresolved);
        (!resolved.is_empty()).then(|| resolved.join("\n"))
    }

    /// 只把「查无记录」视为可继续尝试锚定的失败；歧义/未证明等失败
    /// 必须原样返回，不能靠改写绕过。
    fn is_missing_record(text: &str) -> bool {
        text.contains("没有记录")
            || text.contains("没有相关记录")
            || text.contains("没有可用依据")
            || text.contains("数量为 0")
            || text.contains("未能回答")
    }

    fn is_pending_failure(text: &str) -> bool {
        text.contains("没有记录")
            || text.contains("没有相关记录")
            || text.contains("没有可用依据")
            || text.contains("未能回答")
            || text.contains("数量为 0")
            || text.contains("无法证明")
            || text.contains("不止一种理解")
            // 链接预测/溯因候选不是已证明的答案：悬问只接受确证证明。
            || text.contains("（推断")
            || text.contains("链接预测")
    }

    /// 把洞的取值写入会话事实（逆向谓词按存储方向翻转）。
    fn adopt_hole_fill(&mut self, hole: &PendingHole, object: &str) -> bool {
        if hole.inverted {
            self.adopt_session_triple(object, &hole.slot, &hole.subject)
        } else {
            self.adopt_session_triple(&hole.subject, &hole.slot, object)
        }
    }

    fn pending_expected_type(&self, slot: &str) -> Option<String> {
        let registry = self.law_registry.as_ref()?;
        let profile = registry.profile(slot)?;
        profile.range.iter().next().cloned()
    }

    /// 模拟一次多跳链，返回（目标串，缺口，已走通的边）。
    fn diagnose_chain_hole(&self, question: &str) -> Option<(String, PendingHole, Vec<String>)> {
        let plan = self.planner.parse(question)?;
        let QueryPlan::FindObject { segments } = plan else {
            return None;
        };
        let (entity, predicates) = self.split_chain(&segments)?;
        if predicates.is_empty()
            || entity.is_empty()
            || predicates.iter().any(|predicate| predicate.slot.is_empty())
            || !self.is_known_entity(&entity)
            || !predicates
                .iter()
                .all(|predicate| self.is_known_predicate(&predicate.surface))
        {
            return None;
        }
        let goal = format!(
            "{} 的 {}",
            entity,
            predicates
                .iter()
                .map(|predicate| predicate.surface.as_str())
                .collect::<Vec<_>>()
                .join(" 的 ")
        );
        let mut frontier = vec![entity.clone()];
        let mut support: Vec<String> = Vec::new();
        for (index, predicate) in predicates.iter().enumerate() {
            let mut next: Vec<String> = Vec::new();
            let mut step_support: Vec<String> = Vec::new();
            for node in &frontier {
                let hits = self.pending_simulation_hits(node, predicate);
                for hit in hits {
                    if !next.iter().any(|value| value == &hit.value) {
                        next.push(hit.value.clone());
                    }
                    step_support.push(format!(
                        "{} 的 {} 是 {}",
                        node, predicate.surface, hit.value
                    ));
                }
            }
            if next.is_empty() {
                let subject = if index == 0 {
                    entity.clone()
                } else {
                    let mut candidates = frontier.clone();
                    candidates.sort();
                    candidates.into_iter().next()?
                };
                let mut hole = PendingHole::new(
                    &subject,
                    &predicate.surface,
                    &predicate.slot,
                    predicate.inverted,
                );
                hole.expected_type = self.pending_expected_type(&predicate.slot);
                return Some((goal, hole, support));
            }
            for line in step_support {
                if !support.contains(&line) {
                    support.push(line);
                }
            }
            frontier = next;
        }
        None
    }

    /// 悬问诊断用的边：只认已知事实/会话事实/声明的律，不认链接预测候选。
    fn pending_simulation_hits(&self, node: &str, predicate: &ChainPredicate) -> Vec<ValueHit> {
        let mut hits: Vec<ValueHit> = self
            .node_hits(node, predicate)
            .into_iter()
            .filter(|hit| hit.prediction.is_none())
            .collect();
        if hits.is_empty() && self.laws.has_inference_laws() {
            let (mut expanded, _) = self.law_hits(node, predicate, true);
            expanded.retain(|hit| hit.prediction.is_none());
            hits = expanded;
        }
        hits
    }

    /// 句法重写回执：展示「哪种句法外壳被压成哪个逻辑计划」。
    /// 只读；未命中或词表校验失败时返回 None。
    pub fn query_rewrite_receipt(&self, question: &str) -> Option<String> {
        query_rewrite::rewrite(question, &self.relation_lexicon()).map(|rewrite| rewrite.receipt())
    }

    /// 为问句重建有界推导视图（与回答共用同一份事实、规则与归结搜索）。
    ///
    /// 这是惰性通道：只在用户追问依据时调用，不进入回答热路径，也不写入
    /// 会话快照；没有可形式化的证明时返回 None，由调用方诚实降级。
    pub fn answer_derivation(
        &self,
        question: &str,
        expected_answer: Option<&str>,
    ) -> Option<Derivation> {
        let view = self.resolution_proof_view(question)?;
        Derivation::from_resolution_view(&view, expected_answer)
    }

    /// 无观测副作用的回答实现（报告路径复用，避免失败轨迹递归）。
    fn answer_impl(&self, question: &str) -> Option<String> {
        if let Some(answer) = self.answer_learned_rewrite(question) {
            return Some(answer);
        }
        if let Some(answer) = self.answer_probabilistic_intent(question) {
            return Some(answer);
        }
        if let Some(answer) = crate::type_memory::answer_with_schema(
            &self.generator,
            Some(self.schema_view()),
            question,
        ) {
            return Some(answer.text);
        }
        if let Some(answer) = crate::entity_inventory::answer(&self.generator, question) {
            return Some(answer);
        }
        if let Some(answer) = self.answer_planned(question) {
            return Some(answer);
        }
        if let Some(answer) = self.answer_graph_question(question) {
            return Some(answer);
        }
        if let Some(answer) = self.answer_by_prediction(question) {
            return Some(answer);
        }
        if let Some(answer) = self.answer_by_evidence(question) {
            return Some(answer);
        }
        self.answer_contextual(question)
            .or_else(|| self.formal_semantics_attempt(question))
            .or_else(|| self.generator.answer_import_frame(question))
    }

    /// 对话级证据装配（默认关闭）：把问句、焦点栈、上一谓词与已引用证据
    /// 编译成查询帧，融合稠密检索与全息召回，返回逐条引用的证据束。
    pub fn contextual_evidence(&self, question: &str) -> Option<EvidenceAssembly> {
        if !self.config.enable_contextual_rag
            || !self.config.enable_generation
            || self.retrieval.lock().is_empty()
            || !contextual_rag::looks_like_question(question)
        {
            return None;
        }
        let normalized_question = self.aligner.normalize_cow(question);
        let mentioned_canonicals = self.mentioned_predicate_canonicals(question);
        let explicit_predicate = self
            .first_mentioned_known_predicate(normalized_question.as_ref(), &mentioned_canonicals);
        let explicit_anchors = self.entities_mentioned(question);
        let continuation = contextual_rag::is_continuation(question);
        let referenced = if continuation {
            self.contextual_last
                .lock()
                .as_ref()
                .map(|(_, keys)| keys.clone())
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        let focus = self.context.focus_entities().to_vec();
        let frame = QueryFrame::compile(FrameInput {
            question,
            explicit_anchors,
            explicit_predicate,
            focus: &focus,
            last_predicate: self.context.last_predicate(),
            referenced: &referenced,
        });
        if frame.is_empty_request() {
            return None;
        }
        let base_max_items = self.config.max_predictions.clamp(3, 8);
        let advice = self.semantic.advice();
        let max_items = advice
            .max_items
            .unwrap_or(base_max_items)
            .clamp(3, base_max_items);
        if advice.max_items.is_some() && advice.max_items != Some(base_max_items) {
            self.perf.bump("semantic_control_max_items");
        }
        let query = if frame.explicit_anchors.is_empty() {
            match (frame.anchors.first(), frame.predicate.as_deref()) {
                (Some(anchor), Some(predicate)) => format!("{} {}", anchor, predicate),
                (Some(anchor), None) => anchor.clone(),
                (None, Some(predicate)) => predicate.to_string(),
                (None, None) => question.trim().to_string(),
            }
        } else {
            question.trim().to_string()
        };
        let mut dense_hits = Vec::new();
        let mut holo_hits = Vec::new();
        let two_anchors = frame.anchors.len() >= 2;
        for anchor in frame.anchors.iter().take(2) {
            let constraints = SearchConstraints {
                subject: Some(anchor.clone()),
                predicate: frame.predicate.clone(),
                ..SearchConstraints::default()
            };
            dense_hits.extend(
                self.evidence_search(
                    &query,
                    max_items.max(4),
                    &constraints,
                    SearchBudget::unlimited(),
                )
                .hits,
            );
            // 双锚点路径/对比：补一次入边检索，避免只看到锚点的出向事实。
            if two_anchors && frame.predicate.is_none() {
                let incoming = SearchConstraints {
                    object: Some(anchor.clone()),
                    ..SearchConstraints::default()
                };
                dense_hits.extend(
                    self.evidence_search(
                        &query,
                        max_items.max(4),
                        &incoming,
                        SearchBudget::unlimited(),
                    )
                    .hits,
                );
            }
            if self.config.enable_holographic_memory {
                if let Some(predicate) = &frame.predicate {
                    for hit in self.holo.recall_object(anchor, predicate) {
                        holo_hits.push(HoloEvidence {
                            subject: anchor.clone(),
                            predicate: predicate.clone(),
                            object: hit.symbol,
                            score: hit.score,
                            verified: hit.verified,
                        });
                    }
                }
            }
        }
        if frame.anchors.is_empty() {
            if let Some(predicate) = &frame.predicate {
                let constraints = SearchConstraints {
                    predicate: Some(predicate.clone()),
                    ..SearchConstraints::default()
                };
                dense_hits.extend(
                    self.evidence_search(
                        &query,
                        max_items.max(4),
                        &constraints,
                        SearchBudget::unlimited(),
                    )
                    .hits,
                );
            }
        }
        let mut seen: HashSet<(String, String, String)> = HashSet::new();
        dense_hits.retain(|hit| {
            seen.insert((
                hit.subject.clone(),
                hit.predicate.clone(),
                hit.object.clone(),
            ))
        });
        let mut type_hints: HashMap<String, String> = HashMap::new();
        for hit in &dense_hits {
            if let Some(entity_type) = self.linker.entity_type(&hit.object) {
                type_hints
                    .entry(hit.object.clone())
                    .or_insert_with(|| entity_type.to_string());
            }
        }
        let assembly =
            contextual_rag::assemble(&frame, &dense_hits, &holo_hits, &type_hints, max_items);
        self.semantic.record_assembly(&assembly);
        if assembly.bundle.is_none() {
            return None;
        }
        let keys = assembly.selected_keys();
        if !keys.is_empty() {
            *self.contextual_last.lock() = Some((question.trim().to_string(), keys));
        }
        Some(assembly)
    }

    /// 对话级证据装配的文本回答：只引用已核实证据，证据不足时弃权。
    pub fn answer_contextual(&self, question: &str) -> Option<String> {
        self.contextual_evidence(question)
            .and_then(|assembly| assembly.answer().map(str::to_string))
    }

    /// 最后一道接地回答：计划与链接预测都无结论时，用统一证据索引取回
    /// 「已知实体 + 被提及谓词」的事实，并附检索回执。只引用索引中的真实
    /// 三元组，不生成无出处内容；没有同时满足锚定与谓词条件的证据就弃权。
    fn answer_by_evidence(&self, question: &str) -> Option<String> {
        if !self.config.enable_generation || self.retrieval.lock().is_empty() {
            return None;
        }
        let entities = self.entities_mentioned(question);
        if entities.is_empty() {
            return None;
        }
        let normalized_question = self.aligner.normalize_cow(question);
        let mentioned_canonicals = self.mentioned_predicate_canonicals(question);
        let mentions_predicate = !mentioned_canonicals.is_empty()
            || self
                .known_predicates_normalized_cache
                .iter()
                .any(|(_, predicate_norm)| {
                    !predicate_norm.is_empty()
                        && normalized_question.contains(predicate_norm.as_str())
                });
        if !mentions_predicate {
            return None;
        }
        // 话题谓词（如「配偶」）不等于目标谓词（如「体重」）。证据索引
        // 只允许用问句最后一个「的」后的中心词作答，否则会把
        // 「配偶是谁」的证据当成「配偶的体重」的答案，制造无证明的确定回答。
        let target_predicate = self.target_predicate_surface(normalized_question.as_ref());

        let base_k = self.config.max_predictions.clamp(1, 5);
        let search_k = self
            .semantic
            .advice()
            .max_items
            .map_or(base_k, |max_items| base_k.min(max_items));
        if search_k < base_k {
            self.perf.bump("semantic_control_evidence_k");
        }
        let search = self.evidence_search(
            question,
            search_k,
            &SearchConstraints::default(),
            SearchBudget::default(),
        );
        let anchors: HashSet<&str> = entities.iter().map(String::as_str).collect();
        let hits: Vec<_> = search
            .hits
            .iter()
            .filter(|hit| {
                let anchored =
                    anchors.contains(hit.subject.as_str()) || anchors.contains(hit.object.as_str());
                if !anchored || hit.score < 0.2 {
                    return false;
                }
                let canonical = self.aligner.canonical_form(&hit.predicate);
                let predicate_norm = self.aligner.normalize_cow(&hit.predicate);
                let matches_target = target_predicate.as_ref().map_or(true, |target| {
                    canonical.as_str() == target.as_str()
                        || hit.predicate.as_str() == target.as_str()
                });
                matches_target
                    && (mentioned_canonicals.contains(&hit.predicate)
                        || mentioned_canonicals.contains(&canonical)
                        || (!predicate_norm.is_empty()
                            && normalized_question.contains(predicate_norm.as_ref() as &str)))
            })
            .take(3)
            .collect();
        if hits.is_empty() {
            return None;
        }

        let mut output = String::from("根据证据索引（带检索回执的符号检索）：\n");
        for (index, hit) in hits.iter().enumerate() {
            output.push_str(&format!(
                "  {}. {}（{}）\n",
                index + 1,
                hit.text(),
                hit.breakdown()
            ));
        }
        output.push_str(&format!(
            "  检索回执：候选 {}，打分 {}，上界剪枝 {}；{}。",
            search.receipt.candidates,
            search.receipt.scanned,
            search.receipt.bound_pruned,
            if search.receipt.certified_top_k {
                "top-k 已证明"
            } else {
                "未证明 top-k（结果按当前预算返回）"
            }
        ));
        Some(output.trim_end().to_string())
    }

    fn answer_by_prediction(&self, question: &str) -> Option<String> {
        if !self.config.enable_link_prediction || self.predictor.rules().is_empty() {
            return None;
        }
        let mut entities = self.entities_mentioned(question);
        if entities.is_empty() && self.config.enable_generation {
            // 精确匹配失败时，退回到带容错的实体链接结果（别名词表之外也能命中）
            entities = self.linked_entities(question);
        }
        if entities.is_empty() {
            return None;
        }

        let normalized_question = self.aligner.normalize_cow(question);
        let mentioned_canonicals = self.mentioned_predicate_canonicals(question);
        let mut candidates: Vec<Prediction> = Vec::new();
        for entity in &entities {
            for prediction in
                self.predictions_for_entity(entity, None, self.config.max_predictions * 4)
            {
                if self.is_suppressed_slot_value(
                    &prediction.triple.subject,
                    &prediction.triple.predicate,
                    &prediction.triple.object,
                ) {
                    continue;
                }
                if self.predicate_mentioned_normalized_with(
                    normalized_question.as_ref(),
                    &prediction.triple.predicate,
                    &mentioned_canonicals,
                ) {
                    candidates.push(prediction);
                }
            }
        }
        if candidates.is_empty() {
            return None;
        }

        candidates.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.triple.subject.cmp(&b.triple.subject))
                .then_with(|| a.triple.object.cmp(&b.triple.object))
        });
        let mut seen: HashSet<Triple> = HashSet::new();
        candidates.retain(|p| seen.insert(p.triple.clone()));

        let limit = self.config.max_predictions.clamp(1, 5);
        let mut output =
            String::from("根据知识图谱的链接预测（符号推理；以下为待确认候选，不是确定答案）：\n");
        for prediction in candidates.iter().take(limit) {
            output.push_str("  • ");
            output.push_str(&self.predictor.explain(prediction));
            output.push('\n');
        }
        Some(output)
    }

    pub fn similarity_report(&self, entity_raw: &str) -> String {
        let entity = self.aligner.canonical_form(entity_raw.trim());
        if entity.is_empty() {
            return "请输入要检索的实体名，例如：相似实体：张三".to_string();
        }
        if !self.embedding.is_embedded(&entity) {
            return format!(
                "语义嵌入中没有「{}」的角色特征（可能是孤立节点或尚未导入关系数据）。",
                entity
            );
        }
        let neighbors = self.embedding.nearest(&entity, 5);
        if neighbors.is_empty() {
            return format!(
                "没有与「{}」相似度达到阈值 {:.2} 的实体。",
                entity,
                self.embedding.similarity_floor()
            );
        }
        let mut output = format!(
            "符号嵌入近邻（PPMI 角色向量；已嵌入 {} 个实体、{} 个特征）：\n",
            self.embedding.entity_count(),
            self.embedding.feature_count()
        );
        for (index, (name, _)) in neighbors.iter().enumerate() {
            output.push_str(&format!(
                "  {}. {}\n",
                index + 1,
                self.embedding.explain_similarity(&entity, name)
            ));
        }
        output.trim_end().to_string()
    }

    /// 结构联想：RWR 图嵌入检索 + 结构角色近邻 + 关联路径解释
    pub fn structural_related_report(&self, entity_raw: &str) -> String {
        if !self.config.enable_graph_walk {
            return "结构联想（图游走）未启用（config.toml [reasoning] enable_graph_walk = false）。"
                .to_string();
        }
        let raw = entity_raw.trim();
        if raw.is_empty() {
            return "请输入要联想的实体，例如：结构联想：张三".to_string();
        }
        let entity = self.aligner.canonical_form(raw);
        if !self.walks.is_embedded(&entity) {
            if let Some(linked) = self.best_linked_entity(raw) {
                if self.walks.is_embedded(&linked) {
                    return self.walks.report(&linked, 5);
                }
            }
            return format!(
                "结构联想中没有「{}」：它没有出现在任何关系三元组中。",
                entity
            );
        }
        self.walks.report(&entity, 5)
    }

    /// 话题延伸：以对话焦点实体栈为多源种子做重启随机游走（上下文敏感的联想）
    pub fn topic_extension_report(&self) -> String {
        if !self.config.enable_graph_walk {
            return "话题延伸（图游走）未启用（config.toml [reasoning] enable_graph_walk = false）。"
                .to_string();
        }
        let seeds: Vec<String> = self.context.focus_entities().to_vec();
        if seeds.is_empty() {
            return "还没有对话焦点：先聊到一个实体，再输入「话题延伸」。".to_string();
        }
        let related = self.walks.context_related(&seeds, 6);
        if related.is_empty() {
            return format!(
                "焦点实体「{}」不在关系图中（或无相邻实体），暂时无法延伸话题。",
                seeds.join("、")
            );
        }
        let mut output = format!(
            "话题延伸（以对话焦点 [{}] 为多源种子做 RWR 图游走；图规模 {} 实体 / {} 边）：\n",
            seeds.join(" → "),
            self.walks.entity_count(),
            self.walks.edge_count()
        );
        for (index, (name, weight)) in related.iter().enumerate() {
            let mut line = format!("  {}. {}（联合驻留概率 {:.3}）", index + 1, name, weight);
            if let Some((seed, path)) = seeds.iter().find_map(|seed| {
                self.walks
                    .best_path(seed, name, 4)
                    .map(|path| (seed.clone(), path))
            }) {
                line.push_str(&format!(
                    "；路径：{}",
                    GraphWalkEmbedding::render_path(&seed, &path)
                ));
            }
            output.push_str(&line);
            output.push('\n');
        }
        if let Some(focus) = seeds.first() {
            let similar = self.walks.nearest(focus, 3);
            if !similar.is_empty() {
                output.push_str(&format!("\n  与「{}」结构角色相近：", focus));
                output.push_str(
                    &similar
                        .iter()
                        .map(|(name, score)| format!("{}({:.2})", name, score))
                        .collect::<Vec<_>>()
                        .join("、"),
                );
            }
        }
        output
    }

    /// 全局软推理报告：加权规则推出的候选事实（按概率排序）
    pub fn soft_facts_report(&self, limit: usize) -> String {
        if !self.config.enable_soft_logic {
            return "加权软逻辑未启用（config.toml [reasoning] enable_soft_logic = false）。"
                .to_string();
        }
        let limit = limit.clamp(1, 20);
        let all: Vec<SoftFact> = self.soft.all_predictions(limit * 4);
        // 优先展示"推出的新事实"；若没有（规则不足），退化为展示已知事实
        let inferred: Vec<SoftFact> = all.iter().filter(|fact| !fact.from_fact).cloned().collect();
        let showing_inferred = !inferred.is_empty();
        let mut predictions = if showing_inferred { inferred } else { all };
        predictions.truncate(limit);
        let mut output = format!(
            "加权软推理（MLN-lite：log-odds 累加 + sigmoid 归一；事实 {} 条，加权规则 {} 条，阈值 P≥{:.2}）：\n",
            self.soft.fact_count(),
            self.soft.rule_count(),
            self.soft.min_probability()
        );
        if predictions.is_empty() {
            output.push_str("  （暂无可推出的软事实：可增加事实/规则或降低阈值）\n");
            return output.trim_end().to_string();
        }
        if !showing_inferred {
            output.push_str("  （暂无可加权推出的新事实，以下为已确认事实）\n");
        }
        for (index, fact) in predictions.iter().enumerate() {
            output.push_str(&format!(
                "  {}. {}\n",
                index + 1,
                self.render_soft_fact(fact)
            ));
        }
        output.trim_end().to_string()
    }

    /// 针对某个实体（可带「的谓词」）的软推理报告
    pub fn soft_inference_report(&self, target_raw: &str) -> String {
        if !self.config.enable_soft_logic {
            return "加权软逻辑未启用（config.toml [reasoning] enable_soft_logic = false）。"
                .to_string();
        }
        let target = target_raw.trim();
        if target.is_empty() {
            return "请输入要软推理的对象，例如：软推理：张三 或 软推理：张三的师傅".to_string();
        }
        let Some((entity, predicate)) = self.split_entity_predicate(target) else {
            return format!("无法在「{}」中识别出已知实体。", target);
        };
        let facts: Vec<SoftFact> = match predicate {
            Some(predicate) if !predicate.is_empty() => {
                let forward = self.soft.infer(
                    &Atom::pattern(&predicate, Term::constant(&entity), Term::var("X")),
                    8,
                );
                if !forward.is_empty() {
                    forward
                } else {
                    self.soft.infer(
                        &Atom::pattern(&predicate, Term::var("X"), Term::constant(&entity)),
                        8,
                    )
                }
            }
            _ => self.collect_soft_facts_for_entity(&entity, 8),
        };
        if facts.is_empty() {
            return format!(
                "软推理没有找到关于「{}」达到阈值（P≥{:.2}）的候选：可增加同类事实/规则，或降低阈值。",
                entity,
                self.soft.min_probability()
            );
        }
        let mut output = format!(
            "加权软推理（关于「{}」；事实 {} 条，加权规则 {} 条，无神经网络）：\n",
            entity,
            self.soft.fact_count(),
            self.soft.rule_count()
        );
        for (index, fact) in facts.iter().enumerate() {
            output.push_str(&format!(
                "  {}. {}\n",
                index + 1,
                self.render_soft_fact(fact)
            ));
            if let Some(evidence) = fact.evidence.first() {
                output.push_str(&format!(
                    "     依据：{}（权重 {:+.2}）\n",
                    evidence.source, evidence.weight
                ));
                if !evidence.premises.is_empty() {
                    let premises = evidence
                        .premises
                        .iter()
                        .map(|premise| self.render_atom_nl(premise))
                        .collect::<Vec<_>>()
                        .join(" 且 ");
                    output.push_str(&format!("     前提：{}\n", premises));
                }
            }
        }
        output.trim_end().to_string()
    }

    fn collect_soft_facts_for_entity(&self, entity: &str, limit: usize) -> Vec<SoftFact> {
        let mut merged: HashMap<String, SoftFact> = HashMap::new();
        for atom in self.soft.facts() {
            if atom.predicate.starts_with("__") {
                continue;
            }
            if atom.terms.iter().any(|term| term.name() == entity) {
                merged.insert(
                    atom.display(),
                    SoftFact {
                        atom: atom.clone(),
                        probability: 1.0,
                        log_odds: 6.0,
                        evidence: Vec::new(),
                        from_fact: true,
                    },
                );
            }
        }
        for fact in self.soft.all_predictions(256) {
            if fact.from_fact && fact.atom.predicate.starts_with("__") {
                continue;
            }
            if !fact.atom.terms.iter().any(|term| term.name() == entity) {
                continue;
            }
            merged
                .entry(fact.atom.display())
                .and_modify(|existing| {
                    if fact.probability > existing.probability {
                        *existing = fact.clone();
                    }
                })
                .or_insert(fact);
        }
        let mut results: Vec<SoftFact> = merged.into_values().collect();
        results.sort_by(|a, b| {
            b.probability
                .partial_cmp(&a.probability)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.atom.display().cmp(&b.atom.display()))
        });
        results.truncate(limit.max(1));
        results
    }

    fn render_soft_fact(&self, fact: &SoftFact) -> String {
        let mut text = format!(
            "{} {}（P={:.2}",
            self.render_atom_nl(&fact.atom),
            MarkovLogicEngine::hedge(fact.probability),
            fact.probability
        );
        if fact.from_fact {
            text.push_str("，图谱事实");
        }
        if let Some(evidence) = fact.evidence.first() {
            text.push_str(&format!("；依据：{}", evidence.source));
        }
        text.push('）');
        text
    }

    /// 收集可用于因果解释的目标（优先非事实的加权推测，其次图谱事实）
    fn causal_targets(&self, entity: &str, predicate: Option<&str>, limit: usize) -> Vec<SoftFact> {
        let predicate_matches = |atom: &Atom| -> bool {
            match predicate {
                Some(pred) if !pred.is_empty() => atom.predicate.as_ref() == pred,
                _ => !atom.predicate.starts_with("__"),
            }
        };
        let mentions_entity = |atom: &Atom| atom.terms.iter().any(|term| term.name() == entity);
        let mut merged: HashMap<String, SoftFact> = HashMap::new();
        for atom in self.soft.facts() {
            if !mentions_entity(atom) || !predicate_matches(atom) {
                continue;
            }
            merged.insert(
                atom.display(),
                SoftFact {
                    atom: atom.clone(),
                    probability: 1.0,
                    log_odds: 6.0,
                    evidence: Vec::new(),
                    from_fact: true,
                },
            );
        }
        for fact in self.soft.all_predictions(256) {
            if !mentions_entity(&fact.atom) || !predicate_matches(&fact.atom) {
                continue;
            }
            merged
                .entry(fact.atom.display())
                .and_modify(|existing| {
                    if fact.probability > existing.probability {
                        *existing = fact.clone();
                    }
                })
                .or_insert(fact);
        }
        let mut results: Vec<SoftFact> = merged.into_values().collect();
        results.sort_by(|a, b| {
            a.from_fact
                .cmp(&b.from_fact)
                .then_with(|| {
                    b.probability
                        .partial_cmp(&a.probability)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| a.atom.display().cmp(&b.atom.display()))
        });
        results.truncate(limit.max(1));
        results
    }

    /// 找出某个「实体[的谓词]」对应的可干预原子（优先图谱事实，其次加权推测；
    /// 实体作主语优先——"李四的师傅"应干预 师傅(李四, ?)，结果确定可复现）
    fn find_causal_atom(&self, entity: &str, predicate: Option<&str>) -> Option<Atom> {
        let predicate_matches = |atom: &Atom| -> bool {
            match predicate {
                Some(pred) if !pred.is_empty() => atom.predicate.as_ref() == pred,
                _ => !atom.predicate.starts_with("__"),
            }
        };
        let subject_matches = |atom: &Atom| -> bool {
            atom.terms
                .first()
                .map(|term| term.name() == entity)
                .unwrap_or(false)
                && predicate_matches(atom)
        };
        let any_matches = |atom: &Atom| -> bool {
            atom.terms.iter().any(|term| term.name() == entity) && predicate_matches(atom)
        };
        if let Some(atom) = self
            .soft
            .facts()
            .find(|atom| subject_matches(atom))
            .or_else(|| self.soft.facts().find(|atom| any_matches(atom)))
        {
            return Some(atom.clone());
        }
        // 软预测最多计算一次，两次查找共用同一结果
        let predictions = self.soft.all_predictions(256);
        predictions
            .iter()
            .find(|fact| subject_matches(&fact.atom))
            .or_else(|| predictions.iter().find(|fact| any_matches(&fact.atom)))
            .map(|fact| fact.atom.clone())
    }

    /// 因果推理报告：因果图 + 直接原因 + 事实根因链 + 混杂因子（纯符号 do-演算视角）
    pub fn causal_report(&self, target_raw: &str) -> String {
        if !self.config.enable_causal_reasoning {
            return "因果推理未启用（config.toml [reasoning] enable_causal_reasoning = false）。"
                .to_string();
        }
        if !self.config.enable_soft_logic {
            return "因果推理依赖加权软逻辑，请先启用 enable_soft_logic。".to_string();
        }
        let target = target_raw.trim();
        if target.is_empty() {
            return "请输入要因果解释的对象，例如：因果：李四的师祖 或 因果：张三".to_string();
        }
        let Some((entity, predicate)) = self.split_entity_predicate(target) else {
            return format!("无法在「{}」中识别出已知实体。", target);
        };
        let candidates = self.causal_targets(&entity, predicate.as_deref(), 3);
        if candidates.is_empty() {
            return format!("没有找到关于「{}」可用于因果解释的事实或加权推测。", entity);
        }
        // 概率视图与因果图供所有候选目标共用
        let probabilities = self.soft.probability_map();
        let analyzer = CausalAnalyzer::from_engine_with_probabilities(
            &self.soft,
            &probabilities,
            self.config.causal_max_depth,
        );
        let graph = analyzer.graph();
        let mut output = format!(
            "因果推理（因果图 {} 节点 / {} 条有符号机制；正权促进、负权抑制；规则基化自 {} 条事实）：\n",
            graph.node_count(),
            graph.edge_count(),
            self.soft.fact_count()
        );
        for (index, fact) in candidates.iter().enumerate() {
            let explanation =
                analyzer.explain_with_probabilities(&self.soft, &probabilities, &fact.atom, 3);
            output.push_str(&format!(
                "  {}. {} {}（P={:.2}）\n",
                index + 1,
                self.render_atom_nl(&explanation.target),
                if explanation.probability >= 0.999 {
                    "是图谱事实"
                } else {
                    MarkovLogicEngine::hedge(explanation.probability)
                },
                explanation.probability
            ));
            if explanation.direct_causes.is_empty() {
                output.push_str("     直接原因：（没有规则指向它，可能是一手事实）\n");
            } else {
                output.push_str("     直接原因：\n");
                for link in &explanation.direct_causes {
                    output.push_str(&format!(
                        "       • {}（P={:.2}）—[{:+.2}{}]→ 本结论；依据：{}\n",
                        self.render_atom_nl(&link.cause),
                        link.cause_probability,
                        link.weight,
                        if link.is_inhibitory() {
                            "，抑制"
                        } else {
                            ""
                        },
                        link.source
                    ));
                }
            }
            if let Some(chain) = explanation.chains.first() {
                output.push_str(&format!("     事实根因链：{}\n", chain.render()));
            }
            if !explanation.confounders.is_empty() {
                let names = explanation
                    .confounders
                    .iter()
                    .map(|atom| self.render_atom_nl(atom))
                    .collect::<Vec<_>>()
                    .join("、");
                output.push_str(&format!(
                    "     注意混杂因子（同为原因，相关≠因果）：{}\n",
                    names
                ));
            }
        }
        output.trim_end().to_string()
    }

    /// 因果反事实报告：do(干预原子=真/假) 后，下游结论在"事实世界"与"干预世界"的概率差
    pub fn counterfactual_report(&self, target_raw: &str, value: bool) -> String {
        if !self.config.enable_causal_reasoning {
            return "因果推理未启用（config.toml [reasoning] enable_causal_reasoning = false）。"
                .to_string();
        }
        if !self.config.enable_soft_logic {
            return "反事实推理依赖加权软逻辑，请先启用 enable_soft_logic。".to_string();
        }
        let target = target_raw.trim();
        if target.is_empty() {
            return "请输入要干预的对象，例如：干预：张三的师傅 或 干预：张三的师傅=真".to_string();
        }
        let Some((entity, predicate)) = self.split_entity_predicate(target) else {
            return format!("无法在「{}」中识别出已知实体。", target);
        };
        let Some(intervention) = self.find_causal_atom(&entity, predicate.as_deref()) else {
            return format!(
                "没有找到关于「{}」的可干预原子（需要先有该事实或能推出它的规则）。",
                entity
            );
        };
        // 因果图基化使用同一份概率视图
        let probabilities = self.soft.probability_map();
        let analyzer = CausalAnalyzer::from_engine_with_probabilities(
            &self.soft,
            &probabilities,
            self.config.causal_max_depth,
        );
        let probes = analyzer.direct_effect_probes(&intervention);
        if probes.is_empty() {
            return format!(
                "干预「{}」在当前因果图中没有直接下游（没有规则以它为前提），因此反事实世界与事实世界相同。",
                intervention.display()
            );
        }
        let mut deltas: Vec<CounterfactualDelta> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        // 事实世界与干预世界共用软真值传播结果
        let mut fixed = HashMap::new();
        fixed.insert(intervention.clone(), if value { 1.0 } else { 0.0 });
        let factual_beliefs = self.soft.propagate_with_fixed(&HashMap::new());
        let counterfactual_beliefs = self.soft.propagate_with_fixed(&fixed);
        for probe in probes.iter().take(6) {
            for delta in analyzer
                .counterfactual_from_beliefs(
                    &self.soft,
                    &factual_beliefs,
                    &counterfactual_beliefs,
                    &fixed,
                    probe,
                    &intervention,
                    value,
                    4,
                )
                .deltas
            {
                if seen.insert(delta.atom.display()) {
                    deltas.push(delta);
                }
            }
        }
        deltas.sort_by(|a, b| {
            b.delta
                .abs()
                .partial_cmp(&a.delta.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.atom.display().cmp(&b.atom.display()))
        });
        deltas.truncate(8);
        let mut output = format!(
            "反事实推理（do：干预「{}」={}；符号实现 = 钉住真值 + 切断指向它的规则，其余规则重新软传播）：\n",
            self.render_atom_nl(&intervention),
            if value { "真" } else { "假" }
        );
        output.push_str("  事实世界 → 干预世界（Δ=干预后−事实）：\n");
        for (index, delta) in deltas.iter().enumerate() {
            let direction = if delta.delta > 1e-9 {
                "上升"
            } else if delta.delta < -1e-9 {
                "下降"
            } else {
                "不变"
            };
            let factual_hedge = if delta.factual <= 1e-9 {
                "未推出"
            } else {
                MarkovLogicEngine::hedge(delta.factual)
            };
            let counterfactual_hedge = if delta.counterfactual <= 1e-9 {
                "未推出"
            } else {
                MarkovLogicEngine::hedge(delta.counterfactual)
            };
            output.push_str(&format!(
                "  {}. {}：P {:.2}（{}）→ {:.2}（{}），Δ={:+.2}（{}）\n",
                index + 1,
                self.render_atom_nl(&delta.atom),
                delta.factual,
                factual_hedge,
                delta.counterfactual,
                counterfactual_hedge,
                delta.delta,
                direction
            ));
        }
        output.trim_end().to_string()
    }
}

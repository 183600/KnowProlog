impl SymbolicReasoner {
    /// 把会话事实与槽账本裁决投影为生成叠加层。
    ///
    /// 生成器本身只持有图谱视图；会话层在每次生成时合并进来，功能槽以
    /// 会话赢家为准，其余槽取并集并删掉被更正/否定的取值。这样两份记忆
    /// 不会互相写入或产生撤回歧义。
    fn generation_overlay(&self) -> GenerationOverlay {
        let mut overlay = GenerationOverlay::default();
        for atom in &self.session_facts {
            if let Some((subject, predicate, object)) = Self::session_atom_triple(atom) {
                overlay
                    .add
                    .entry((subject.to_string(), predicate.to_string()))
                    .or_default()
                    .insert(object.to_string());
            }
        }
        for key in self.slots.keys() {
            let accepted = self.slots.accepted(&key.subject, &key.predicate);
            if accepted.is_empty() {
                continue;
            }
            let slot = (key.subject.clone(), key.predicate.clone());
            if self.slots.is_functional(&key.predicate) {
                overlay.replace.insert(slot.clone(), accepted.clone());
            } else {
                overlay
                    .add
                    .entry(slot.clone())
                    .or_default()
                    .extend(accepted.iter().cloned());
            }
            for suppressed in self.slots.suppressed(&key.subject, &key.predicate) {
                overlay
                    .suppress
                    .entry(slot.clone())
                    .or_default()
                    .insert(suppressed);
            }
        }
        for atom in &self.session_negative_facts {
            if let Some((subject, predicate, object)) = Self::session_atom_triple(atom) {
                overlay
                    .suppress
                    .entry((subject.to_string(), predicate.to_string()))
                    .or_default()
                    .insert(object.to_string());
            }
        }
        overlay
    }

    // 生成式介绍：把实体的相关事实组合成连贯段落（无神经网络）
    // 生成后自动做语义蕴含自检，并记录"上一段生成"以便多轮续说。
    pub fn generate_about(&mut self, entity_raw: &str) -> Option<String> {
        if !self.config.enable_generation {
            return None;
        }
        let trimmed = entity_raw.trim();
        if trimmed.is_empty() {
            return None;
        }
        let canonical = self.aligner.canonical_form(trimmed);
        let entity = if canonical.is_empty() {
            trimmed.to_string()
        } else {
            canonical
        };
        let overlay = self.generation_overlay();
        let clauses = self.generator.profile_clauses_with_overlay(
            &entity,
            GenerationOrder::Relevance,
            &overlay,
        )?;
        let plan = self.build_answer_plan(&format!("介绍：{}", entity), &entity, &clauses);
        let mut cursor = GenerationCursor::new(&entity, GenerationOrder::Relevance);
        let page = self.generator.cursor_page(
            &entity,
            GenerationOrder::Relevance,
            &overlay,
            &mut cursor,
        )?;
        self.observe_epistemic_plan(&plan);
        self.context.record_generation_with_plan(cursor, plan);
        let audit = self.audit_generation(&page.text, &page.trusted);
        let mut text = audit.text;
        if let Some(note) = audit.note {
            if !text.is_empty() {
                text.push('\n');
            }
            text.push_str(&note);
        }
        Some(text)
    }

    /// 观察一份新计划：只有启用 PCC 时才写入逐命题复核台账。
    fn observe_epistemic_plan(&mut self, plan: &AnswerPlan) {
        if !self.config.confidence.enabled {
            return;
        }
        let policy = TurnPolicy::from_confidence(&self.config.confidence);
        let turn = self.epistemic_turns.turn.saturating_add(1);
        let envelope = TurnEnvelope::from_plan(turn, plan);
        self.epistemic_turns.observe_turn(&envelope, &policy);
        self.beliefs.observe_turn(turn, &plan.subject, &envelope);
        self.beliefs.advance_to(turn);
    }

    /// 从生成命题的来源等级与槽账本事件构造逐命题置信区间。
    ///
    /// 关闭 PCC 时不附加区间，旧计划与快照字节保持不变。启用后：
    /// 图谱断言以下界较高起步；会话直陈/更正降低下界；槽账本中的冲突
    /// 同时压低上界，佐证按递减收益抬升下界。所有推导都是整数运算。
    fn claim_envelope(&self, clause: &ProfileClause) -> ClaimEnvelope {
        if !self.config.confidence.enabled {
            return ClaimEnvelope::point(0.5);
        }
        let history = self.slots.history(&clause.key.entity, clause.predicate());
        let grade = if clause.from_overlay {
            match history.last().map(|event| event.strength) {
                Some(AssertionStrength::Correction) => EnvelopeGrade::SessionCorrection,
                Some(AssertionStrength::Hearsay) => EnvelopeGrade::SessionHearsay,
                _ => EnvelopeGrade::SessionDirect,
            }
        } else {
            EnvelopeGrade::Graph
        };
        let corroborations = history
            .iter()
            .filter(|event| {
                matches!(
                    event.kind,
                    crate::slot_ledger::SlotEventKind::Corroborate
                        | crate::slot_ledger::SlotEventKind::Evidence
                )
            })
            .count();
        let conflicts = history
            .iter()
            .filter(|event| {
                matches!(
                    event.kind,
                    crate::slot_ledger::SlotEventKind::Conflict
                        | crate::slot_ledger::SlotEventKind::Suppress
                )
            })
            .count();
        ClaimEnvelope::from_grade(grade, corroborations, conflicts)
    }

    /// 把生成器命题转成回答计划：先做一次整批生成自检，再按原顺序保留。
    ///
    /// 自检失败的句子不进入计划，避免「详细版」重新说出初次生成时已被
    /// 丢弃的句子；容量上限由 [`AnswerPlan::new`] 统一执行。
    fn build_answer_plan(
        &self,
        question: &str,
        entity: &str,
        clauses: &[ProfileClause],
    ) -> AnswerPlan {
        let accepted: Vec<&ProfileClause> = if self.config.enable_entailment && !clauses.is_empty()
        {
            let raw = clauses
                .iter()
                .map(|clause| clause.sentence.as_str())
                .collect::<Vec<_>>()
                .join("\n");
            let trusted: Vec<String> = clauses
                .iter()
                .filter(|clause| clause.from_overlay)
                .map(|clause| clause.sentence.clone())
                .collect();
            let audited = self.audit_generation(&raw, &trusted);
            let mut kept: HashMap<String, usize> = HashMap::new();
            for line in audited.text.lines() {
                *kept.entry(line.trim().to_string()).or_default() += 1;
            }
            clauses
                .iter()
                .filter(|clause| {
                    let key = clause.sentence.trim();
                    match kept.get_mut(key) {
                        Some(count) if *count > 0 => {
                            *count -= 1;
                            true
                        }
                        _ => false,
                    }
                })
                .collect()
        } else {
            clauses.iter().collect()
        };
        let suppressed = clauses.len().saturating_sub(accepted.len());
        let omitted = accepted.len().saturating_sub(MAX_PLAN_CLAIMS);
        let claims = accepted
            .iter()
            .take(MAX_PLAN_CLAIMS)
            .map(|clause| {
                let mut claim = PlanClaim::new(
                    clause.slot_id(),
                    clause.predicate(),
                    clause.sentence.clone(),
                    clause.from_overlay,
                );
                if !clause.evidence.is_empty() {
                    claim = claim.with_evidence(
                        clause
                            .evidence
                            .iter()
                            .map(|(subject, predicate, object)| {
                                ClaimEvidence::new(subject, predicate, object, clause.from_overlay)
                            })
                            .collect(),
                    );
                }
                if self.config.confidence.enabled {
                    claim.with_envelope(self.claim_envelope(clause))
                } else {
                    claim
                }
            })
            .collect();
        AnswerPlan::new(question, entity, self.revision, claims, omitted, suppressed)
    }

    /// 对上一段生成的计划做风格化重述：只选择与呈现，不重新检索。
    ///
    /// 计划缺失、结构版本不支持或知识版本已变化时返回明确提示；绝不把
    /// 两代知识拼进同一段回答。
    pub fn restyle_generation(&mut self, directive: StyleDirective) -> Option<String> {
        if !self.config.enable_generation {
            return None;
        }
        let Some(state) = self.context.generation_state().cloned() else {
            return Some(
                "还没有可复述的回答计划。先用「介绍：实体名」生成一段介绍，再说「简短点」或「完整讲讲」。"
                    .to_string(),
            );
        };
        let Some(plan) = state.plan.as_ref() else {
            return Some(format!(
                "最近一次生成回答（关于「{}」）没有可复述的计划；请重新输入「介绍：{}」。",
                state.entity, state.entity
            ));
        };
        if !plan.is_current_schema() {
            return Some(format!(
                "最近一份生成计划的结构版本 {} 超出当前支持范围；请重新生成「介绍：{}」。",
                plan.version, state.entity
            ));
        }
        if !plan.is_fresh_for(self.revision) {
            return Some(format!(
                "最近一份生成计划基于知识版本 v{}，当前已是 v{}；为避免混用两代事实，请重新输入「介绍：{}」。",
                plan.revision, self.revision, state.entity
            ));
        }
        match plan.realize(directive) {
            Ok(text) => Some(text),
            Err(error) => Some(error.to_string()),
        }
    }

    /// 用当前知识重建同一实体的计划，并与上一份计划逐点对照。
    ///
    /// 计划本身只读：不重新解释原问题、不改写已经说出口的回答，也不替换
    /// 已保存的旧计划。复核会推进逐命题置信台账（应用待复核项、生成掉档
    /// 更正），但不写知识图谱。缺失/过期/结构不兼容都给出明确回执；实体在
    /// 图谱与会话层都不存在时，当前计划为空，差异会逐点标为「移除」。
    pub fn review_generation_plan(&mut self) -> String {
        if !self.config.enable_generation {
            return "生成功能未启用，无法复核回答计划。".to_string();
        }
        let Some(state) = self.context.generation_state().cloned() else {
            return "还没有回答计划。先用「介绍：实体名」生成一段介绍，再说「复核计划」。"
                .to_string();
        };
        let Some(plan) = state.plan.as_ref() else {
            return format!(
                "最近一次生成回答（关于「{}」）没有可复述的计划；请重新输入「介绍：{}」。",
                state.entity, state.entity
            );
        };
        if !plan.is_current_schema() {
            return format!(
                "最近一份生成计划的结构版本 {} 超出当前支持范围；请重新输入「介绍：{}」。",
                plan.version, state.entity
            );
        }
        let overlay = self.generation_overlay();
        let clauses = self
            .generator
            .profile_clauses_with_overlay(&state.entity, GenerationOrder::Relevance, &overlay)
            .unwrap_or_default();
        let question = if plan.question.is_empty() {
            format!("介绍：{}", state.entity)
        } else {
            plan.question.clone()
        };
        let current = self.build_answer_plan(&question, &state.entity, &clauses);
        let mut output = format!("复核对象：最近一次生成（关于「{}」）。\n", state.entity);
        output.push_str(&plan.diff(&current).render_zh(true));
        if self.config.confidence.enabled && !self.epistemic_turns.is_empty() {
            let policy = TurnPolicy::from_confidence(&self.config.confidence);
            let turn = self.epistemic_turns.turn.saturating_add(1);
            let envelope = TurnEnvelope::from_plan(turn, &current);
            let report = self.epistemic_turns.reconcile(&envelope, &policy);
            self.beliefs.observe_turn(turn, &current.subject, &envelope);
            self.beliefs.advance_to(turn);
            let text = report.render_zh();
            if !text.is_empty() {
                output.push_str("\n置信复核：\n");
                output.push_str(&text);
                output.push('\n');
            }
        }
        if !plan.is_fresh_for(self.revision) {
            output.push_str(&format!(
                "\n旧计划仍保留为历史对照；如需基于当前知识重新生成，请输入「介绍：{}」。",
                state.entity
            ));
        }
        output
    }

    /// 当前回答计划的审计视图：id、知识版本、逐点命题与可用风格指令。
    pub fn answer_plan_report(&self) -> String {
        let Some(state) = self.context.generation_state() else {
            return "还没有回答计划。先用「介绍：实体名」生成一段介绍，再说「简短点」或「完整讲讲」。"
                .to_string();
        };
        let Some(plan) = state.plan.as_ref() else {
            return format!(
                "最近一次生成回答（关于「{}」）没有可复述的计划；请重新输入「介绍：{}」。",
                state.entity, state.entity
            );
        };
        if !plan.is_current_schema() {
            return format!(
                "最近一份生成计划的结构版本 {} 超出当前支持范围；请重新输入「介绍：{}」。",
                plan.version, state.entity
            );
        }
        let freshness = if plan.is_fresh_for(self.revision) {
            "与当前知识版本一致".to_string()
        } else {
            format!(
                "已过期：计划基于 v{}，当前为 v{}（输入「复核计划」查看逐点差异）",
                plan.revision, self.revision
            )
        };
        let mut output = format!(
            "回答计划 {}（主体：{}；{} 点；{}；{}）：\n",
            plan.id,
            plan.subject,
            plan.claim_count(),
            freshness,
            if plan.question.is_empty() {
                "来源未记录".to_string()
            } else {
                format!("问句「{}」", plan.question)
            }
        );
        for (index, claim) in plan.claims.iter().enumerate() {
            let envelope = claim
                .envelope
                .map(|envelope| {
                    format!(
                        "；置信区间：{}",
                        self.config.confidence.render_envelope(envelope)
                    )
                })
                .unwrap_or_default();
            output.push_str(&format!(
                "  {}. {}（谓词：{}；来源：{}{}）\n",
                index + 1,
                claim.text.trim(),
                claim.predicate,
                if claim.session {
                    "会话事实"
                } else {
                    "图谱事实"
                },
                envelope
            ));
        }
        if plan.suppressed > 0 {
            output.push_str(&format!(
                "  另有 {} 句未通过生成自检，未纳入计划。\n",
                plan.suppressed
            ));
        }
        if plan.omitted > 0 {
            output.push_str(&format!(
                "  计划容量上限 {} 点，另有 {} 点未纳入。\n",
                MAX_PLAN_CLAIMS, plan.omitted
            ));
        }
        output.push_str(
            "  可输入「简短点」「完整讲讲」「换种表达」「第N点是什么」或「预算：120字」重述同一份计划。",
        );
        if self.config.confidence.enabled {
            let pending = self.epistemic_turns.render_pending_zh(3);
            if !pending.is_empty() {
                output.push('\n');
                output.push_str(&pending);
            }
            if !self.beliefs.is_empty() {
                output.push('\n');
                output.push_str(&self.beliefs.render_report(self.beliefs.turn()));
                output.push('\n');
            }
        }
        output
    }

    /// 跨模块/跨轮信念账本的只读视图；未启用 PCC 时为空。
    pub fn belief_ledger(&self) -> &BeliefLedger {
        &self.beliefs
    }

    pub fn belief_ledger_report(&self) -> String {
        if self.beliefs.is_empty() {
            return "信念账本为空：先生成一份带置信区间的回答计划。".to_string();
        }
        self.beliefs.render_report(self.beliefs.turn())
    }

    pub fn discourse(&self) -> &DiscoursePlanner {
        &self.discourse
    }

    // 符号主义叙述：OWL 本体定义开场 + 层级/继承补充 + DCG 文法实现的多句连贯段落
    pub fn narrate(&mut self, entity_raw: &str) -> Option<String> {
        if !self.config.enable_generation || !self.config.enable_discourse {
            return None;
        }
        let trimmed = entity_raw.trim();
        if trimmed.is_empty() {
            return None;
        }
        let canonical = self.aligner.canonical_form(trimmed);
        let entity = if canonical.is_empty() {
            trimmed.to_string()
        } else {
            canonical
        };
        let focus = self.context.focus().map(|focus| focus.to_string());
        let text = self.discourse.narrate(
            &entity,
            focus.as_deref(),
            None,
            self.config.discourse_max_sentences,
            self.generator.predicate_surfaces(),
        )?;
        self.context.record_entities(&[entity.clone()]);
        self.context.record_question(&format!("叙述：{}", entity));
        Some(text)
    }

    // 围绕话题的叙述：命中话题的谓词优先（回答"关于X的P"式追问）
    pub fn narrate_topic(&mut self, entity_raw: &str, topic: &str) -> Option<String> {
        if !self.config.enable_generation || !self.config.enable_discourse {
            return None;
        }
        let trimmed = entity_raw.trim();
        let topic = topic.trim();
        if trimmed.is_empty() || topic.is_empty() {
            return None;
        }
        let canonical = self.aligner.canonical_form(trimmed);
        let entity = if canonical.is_empty() {
            trimmed.to_string()
        } else {
            canonical
        };
        let focus = self.context.focus().map(|focus| focus.to_string());
        self.discourse.narrate(
            &entity,
            focus.as_deref(),
            Some(topic),
            self.config.discourse_max_sentences,
            self.generator.predicate_surfaces(),
        )
    }

    // DCG 文法复述：同一事实的多种合规句式（纯符号改写，不改变语义）
    pub fn paraphrase_fact(
        &mut self,
        entity_raw: &str,
        predicate: &str,
        limit: usize,
    ) -> Option<String> {
        if !self.config.enable_discourse {
            return None;
        }
        let canonical = self.aligner.canonical_form(entity_raw.trim());
        let entity = if canonical.is_empty() {
            entity_raw.trim().to_string()
        } else {
            canonical
        };
        let variants = self.discourse.paraphrase(
            &entity,
            predicate.trim(),
            limit.max(2),
            self.generator.predicate_surfaces(),
        );
        if variants.is_empty() {
            return None;
        }
        let mut text = String::new();
        for (index, variant) in variants.iter().enumerate() {
            if index > 0 {
                text.push('\n');
            }
            text.push_str(&format!("{}. {}", index + 1, variant));
        }
        Some(text)
    }

    // OWL 本体教学：把中文公理加入本体（子类/互斥/定义域/固有属性……）
    pub fn teach_ontology(&mut self, axiom: &str) -> Result<String, String> {
        if !self.config.enable_discourse {
            return Err("话语引擎未启用（enable_discourse = false）".to_string());
        }
        let trimmed = axiom.trim();
        if trimmed.is_empty() {
            return Err("请提供本体公理，例如：人是一种动物".to_string());
        }
        let summary = self.discourse.teach_axiom(trimmed)?;
        self.sync_retrieval_type_hierarchy();
        self.refresh_schema_view();
        Ok(summary)
    }

    /// 把单一本体（教学态 + 配置态）的子类/互斥公理同步进稠密检索的类型层级：
    /// 类型约束按 RDFS 展开（互斥是唯一硬剪证据），与本体推理器同源同簿。
    /// 观察是幂等的（集合插入），重复同步不产生重复边。
    fn sync_retrieval_type_hierarchy(&mut self) {
        let lessons: Vec<OntologyLesson> = self.discourse.taught_lessons().to_vec();
        let mut retrieval = self.retrieval.lock();
        for lesson in &lessons {
            match lesson {
                OntologyLesson::Subclass { sub, sup, .. } => retrieval.observe_subclass(sub, sup),
                OntologyLesson::Disjoint { left, right, .. } => {
                    retrieval.observe_disjoint(left, right)
                }
                _ => {}
            }
        }
    }

    /// 合并一份外部 RDFS 模式视图（配置别名/子类/互斥/谓词画像），
    /// 并让符号嵌入按新闭包重放。返回内容是否实际变化。
    pub fn configure_schema_view(&mut self, view: &SchemaView) -> bool {
        let merged = self.schema_view.merge(view);
        let digest = self.schema_view.digest();
        let needs_install =
            merged || self.embedding.schema_digest().as_deref() != Some(digest.as_str());
        if needs_install {
            self.embedding.set_schema(Some(self.schema_view.clone()));
        }
        merged
    }

    /// 当前 RDFS 模式视图（只读；嵌入与类型审计同源）。
    pub fn schema_view(&self) -> &SchemaView {
        &self.schema_view
    }

    /// 把已登记的全部本体公理重建成模式视图并装进嵌入。
    /// 公理集单调，因此迁移只增不减；重复调用由摘要去重。
    fn refresh_schema_view(&mut self) {
        let program = OntologyProgram::from_lessons(self.discourse.taught_lessons().to_vec());
        let lessons_view = SchemaView::from_ontology_program(&program);
        self.schema_view.merge(&lessons_view);
        self.embedding.set_schema(Some(self.schema_view.clone()));
    }

    /// 配置态本体并入**单一本体**：`[type_system]` 与 `[[predicate_profiles]]`
    /// 归约出的公理与教学公理共用同一个本体推理器与同一个胶囊，
    /// 落盘/重建/分叉都不丢（配置不再是第二套本体）。
    /// 公理集合是单调的：已存在的条目按整句（含原句）去重，重复调用幂等。
    /// 返回本次新登记的公理条数。
    pub fn configure_ontology_program(&mut self, program: &OntologyProgram) -> usize {
        let before = self.discourse.taught_lessons().len();
        for lesson in &program.lessons {
            self.discourse.install_lesson(lesson);
        }
        let added = self.discourse.taught_lessons().len().saturating_sub(before);
        if added > 0 {
            self.perf.bump_by("ontology_config_lessons", added as u64);
            self.sync_retrieval_type_hierarchy();
        }
        self.refresh_schema_view();
        added
    }

    /// 已登记的本体公理（配置态 + 教学态同簿；供审计与断言）。
    pub fn taught_ontology_lessons(&self) -> &[OntologyLesson] {
        self.discourse.taught_lessons()
    }

    // 多轮生成：接着上一段介绍继续讲尚未说过的命题，并先播更正/撤回
    pub fn continue_generation(&mut self) -> Option<String> {
        if !self.config.enable_generation {
            return None;
        }
        let state = self.context.generation_state()?.clone();
        let overlay = self.generation_overlay();
        let mut cursor = match state.cursor.clone() {
            Some(cursor) if cursor.entity == state.entity => cursor,
            _ => {
                // 旧快照只有位置：把当时已讲过的前 N 句补记为已说，避免重复。
                let mut cursor = GenerationCursor::new(&state.entity, state.order);
                if let Some(clauses) = self.generator.profile_clauses_with_overlay(
                    &state.entity,
                    state.order,
                    &overlay,
                ) {
                    cursor.seed_spoken(&clauses, state.shown);
                }
                cursor
            }
        };
        let page = self
            .generator
            .cursor_page(&state.entity, state.order, &overlay, &mut cursor)?;
        if page.shown == 0 {
            return Some(format!(
                "关于「{}」的知识已经介绍完毕。可以输入「换个角度说」按不同顺序重述，或介绍其它实体。",
                state.entity
            ));
        }
        let plan = self
            .generator
            .profile_clauses_with_overlay(&state.entity, state.order, &overlay)
            .map(|clauses| {
                self.build_answer_plan(
                    &format!("继续介绍：{}", state.entity),
                    &state.entity,
                    &clauses,
                )
            });
        match plan {
            Some(plan) => self.context.record_generation_with_plan(cursor, plan),
            None => self.context.record_generation_with_cursor(cursor),
        }
        let audit = self.audit_generation(&page.text, &page.trusted);
        let mut text = audit.text;
        if let Some(note) = audit.note {
            if !text.is_empty() {
                text.push('\n');
            }
            text.push_str(&note);
        }
        Some(text)
    }

    // 多轮生成：换一个排序角度重述同一实体的介绍
    pub fn switch_generation_angle(&mut self) -> Option<String> {
        if !self.config.enable_generation {
            return None;
        }
        let state = self.context.generation_state()?.clone();
        let order = match state.order {
            GenerationOrder::Relevance => GenerationOrder::Alphabetical,
            GenerationOrder::Alphabetical => GenerationOrder::Relevance,
        };
        let overlay = self.generation_overlay();
        let mut cursor = state
            .cursor
            .clone()
            .unwrap_or_else(|| GenerationCursor::new(&state.entity, order));
        // 「换个角度说」是显式重述：账本从零开始，但排序角度更新。
        cursor.reset_for_order(order);
        let page = self
            .generator
            .cursor_page(&state.entity, order, &overlay, &mut cursor)?;
        if page.shown == 0 {
            return Some(format!("关于「{}」暂无可重述的事实。", state.entity));
        }
        let plan = self
            .generator
            .profile_clauses_with_overlay(&state.entity, order, &overlay)
            .map(|clauses| {
                self.build_answer_plan(
                    &format!("换个角度介绍：{}", state.entity),
                    &state.entity,
                    &clauses,
                )
            });
        match plan {
            Some(plan) => self.context.record_generation_with_plan(cursor, plan),
            None => self.context.record_generation_with_cursor(cursor),
        }
        let label = match order {
            GenerationOrder::Relevance => "信息量排序（由稀有谓词到常见谓词）",
            GenerationOrder::Alphabetical => "主题排序（按谓词字典序）",
        };
        let heading = format!("换个角度介绍「{}」（{}）：", state.entity, label);
        let audit = self.audit_generation(&page.text, &page.trusted);
        let mut text = if audit.text.is_empty() {
            heading
        } else {
            format!("{}\n{}", heading, audit.text)
        };
        if let Some(note) = audit.note {
            text.push('\n');
            text.push_str(&note);
        }
        Some(text)
    }

    fn knowledge(&self) -> Knowledge<'_> {
        Knowledge {
            predictor: &self.predictor,
            generator: &self.generator,
            min_rule_confidence: self.config.entailment_min_confidence,
            max_rule_candidates: 16,
            slot_policy: Some(self.slots.policy()),
        }
    }

    fn proposition_parser(&self) -> PropositionParser<'_> {
        PropositionParser::with_surfaces(&self.aligner, &self.linker, &self.proposition_surfaces)
    }

    /// 把「图谱 ∪ 会话层」投影为形式语义的词库与微型模型。
    ///
    /// 只读取生成器/预测器/槽账本的既有投影，不新建事实源：
    /// 关系三元组、实体类型、属性键值与子类层级都按同一份知识版本构建。
    fn formal_semantics_view(
        &self,
    ) -> (
        crate::formal_semantics::FormalSemantics,
        crate::formal_semantics::TinyModel,
    ) {
        use crate::formal_semantics::{FormalSemantics, TinyModel};
        let synthesized = if self.session_facts.is_empty() && self.slots.is_empty() {
            None
        } else {
            Some(self.session_aware_knowledge())
        };
        let (predictor, generator) = match &synthesized {
            Some((predictor, generator)) => (predictor, generator),
            None => (&self.predictor, &self.generator),
        };
        let mut semantics = FormalSemantics::new();
        let mut model = TinyModel::new(self.aligner.synonyms().clone());
        let mut canonical_to_surface: HashMap<String, String> = HashMap::new();
        for (surface, canonical) in generator.predicate_surfaces() {
            canonical_to_surface
                .entry(canonical.clone())
                .or_insert_with(|| surface.clone());
        }
        semantics.set_predicate_surfaces(&canonical_to_surface);
        for entity in generator.all_entities() {
            semantics.register_entity(&entity);
            let etype = generator.entity_type_of(&entity).unwrap_or("");
            let pairs = generator.attribute_pairs(&entity);
            if pairs.is_empty() && etype.trim().is_empty() {
                model.observe_entity(&entity, "", &[]);
            } else {
                model.observe_entity(&entity, etype, &pairs);
            }
            if !etype.trim().is_empty() {
                semantics.register_class(etype);
            }
            for (key, _) in &pairs {
                semantics.register_unary_predicate(key);
            }
        }
        for triple in predictor.triples() {
            model.observe_relation(&triple.subject, &triple.predicate, &triple.object);
            semantics.register_binary_predicate(&triple.predicate);
            semantics.register_entity(&triple.subject);
            semantics.register_entity(&triple.object);
            if crate::formal_semantics::is_subclass_relation(&triple.predicate) {
                semantics.register_class(&triple.subject);
                semantics.register_class(&triple.object);
            }
        }
        for key in self.attribute_index.keys() {
            semantics.register_entity(&key.0);
            semantics.register_unary_predicate(&key.1);
        }
        (semantics, model)
    }

    /// 候选森林 + 模型接地的可读报告（「语义分析」命令）。
    pub fn formal_semantics_report(&self, text: &str) -> String {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return "请输入要分析的句子，例如：语义分析：张三的年龄是多少".to_string();
        }
        let (semantics, model) = self.formal_semantics_view();
        let evaluation = semantics.evaluate(trimmed, &model);
        semantics.render_forest_report(&evaluation)
    }

    /// 旧回答路径全部失败后的语义回退：只回答唯一且由正面证据支持的读法。
    fn formal_semantics_answer(&self, text: &str) -> Option<String> {
        let (semantics, _model, evaluation) = self.formal_semantics_evaluation(text)?;
        semantics.answer_evaluation(&evaluation, true)
    }

    fn formal_semantics_evaluation(
        &self,
        text: &str,
    ) -> Option<(
        crate::formal_semantics::FormalSemantics,
        crate::formal_semantics::TinyModel,
        crate::formal_semantics::ForestEvaluation,
    )> {
        if !self.config.enable_entailment {
            return None;
        }
        let trimmed = text.trim();
        if trimmed.is_empty()
            || (!crate::function_words::contains_question_marker(trimmed)
                && !trimmed.ends_with('?')
                && !trimmed.ends_with('？'))
        {
            return None;
        }
        let (semantics, model) = self.formal_semantics_view();
        let evaluation = semantics.evaluate(trimmed, &model);
        Some((semantics, model, evaluation))
    }

    /// 正常回答路径的回退：唯一读法直接作答；多个有证据读法则建立澄清帧。
    fn formal_semantics_attempt(&self, text: &str) -> Option<String> {
        let (semantics, _model, evaluation) = self.formal_semantics_evaluation(text)?;
        match &evaluation.decision {
            crate::formal_semantics::ForestDecision::Answer { .. } => {
                semantics.answer_evaluation(&evaluation, true)
            }
            crate::formal_semantics::ForestDecision::Ambiguous { indexes } => {
                let options: Vec<FormalClarificationOption> = indexes
                    .iter()
                    .filter_map(|index| evaluation.grounded.get(*index))
                    .map(|grounded| {
                        let evidence = match &grounded.outcome {
                            crate::formal_semantics::QueryOutcome::Values { evidence, .. }
                            | crate::formal_semantics::QueryOutcome::Entities {
                                evidence, ..
                            }
                            | crate::formal_semantics::QueryOutcome::EntitiesByObject {
                                evidence,
                                ..
                            }
                            | crate::formal_semantics::QueryOutcome::Truth { evidence, .. } => {
                                evidence.clone()
                            }
                            crate::formal_semantics::QueryOutcome::Unsupported(_) => Vec::new(),
                        };
                        FormalClarificationOption {
                            logical_form: grounded.candidate.form.render(),
                            answer: semantics.render_outcome(&grounded.outcome),
                            evidence,
                        }
                    })
                    .collect();
                if options.len() < 2 {
                    return semantics.answer_evaluation(&evaluation, true);
                }
                let clarification = FormalClarification {
                    question: text.trim().to_string(),
                    options,
                };
                let prompt = clarification.prompt();
                *self.formal_clarification.lock() = Some(clarification);
                Some(prompt)
            }
            crate::formal_semantics::ForestDecision::Unknown { .. } => None,
        }
    }

    pub fn formal_clarification_active(&self) -> bool {
        self.formal_clarification.lock().is_some()
    }

    pub fn formal_clarification_prompt(&self) -> Option<String> {
        self.formal_clarification
            .lock()
            .as_ref()
            .map(FormalClarification::prompt)
    }

    /// 处理对语义歧义澄清的短答：序号/前者/后者 → 选定读法；都不是 → 放弃。
    /// 非回答输入清空澄清帧并返回 `NotAnAnswer`，放行常规管线。
    pub fn answer_formal_clarification(&mut self, text: &str) -> FormalClarificationOutcome {
        let Some(clarification) = self.formal_clarification.lock().clone() else {
            return FormalClarificationOutcome::NotAnAnswer;
        };
        match clarification.parse_reply(text) {
            Some(FormalReply::Choice(index)) => {
                let option = &clarification.options[index];
                let answer = format!(
                    "按你选择的读法（{}）：\n{}",
                    option.logical_form,
                    option.full_answer()
                );
                *self.formal_clarification.lock() = None;
                FormalClarificationOutcome::Answered(answer)
            }
            Some(FormalReply::Cancel) => {
                *self.formal_clarification.lock() = None;
                FormalClarificationOutcome::Cancelled(format!(
                    "已放弃「{}」的歧义消解；没有把任何候选当作事实。",
                    clarification.question
                ))
            }
            Some(FormalReply::OutOfRange) => {
                FormalClarificationOutcome::Reprompt(clarification.prompt())
            }
            None => {
                *self.formal_clarification.lock() = None;
                FormalClarificationOutcome::NotAnAnswer
            }
        }
    }

    // 形式化语义 + 语义蕴含：把陈述句解析为命题并逐条判定蕴含/矛盾/未知。
    // 判定视图与生成自检共用同一份「图谱 ∪ 会话层」，刚记住的事实应能立即
    // 通过校验，而不是生成时可见、判断时却报未知。
    pub fn check_statement(&self, text: &str) -> Vec<EntailmentResult> {
        if !self.config.enable_entailment {
            return Vec::new();
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
        parser
            .parse(text)
            .into_iter()
            .map(|proposition| semantic_entailment::entail(&proposition, &knowledge))
            .collect()
    }

    /// 语义蕴含判未知/形式化失败时的后向链补充判定：把陈述转成归结目标，
    /// 用「图谱事实 + 会话规则」做 SLD 反驳，得证则输出可追溯的证明树
    fn backward_chain_appendix(&self, text: &str) -> Option<String> {
        if !self.config.enable_resolution {
            return None;
        }
        let atom = self
            .statement_atom(text)
            .or_else(|| chinese_reasoning_nlu::parse_proposition(text))
            .filter(Atom::is_ground)?;
        let solutions = self.resolver.solve(std::slice::from_ref(&atom), &[]);
        let solution = solutions.first()?;
        let mut lines = Vec::new();
        for proof in &solution.proofs {
            self.render_proof_tree(proof, &mut lines, 2);
        }
        Some(format!(
            "【后向链补充判定】语义蕴含未能直接判定，但归结引擎证明了「{}」（证明置信度 {:.2}）：\n  后向链证明树：\n{}",
            self.render_atom_nl(&atom),
            solution.confidence,
            lines.join("\n")
        ))
    }

    pub fn entailment_report(&self, text: &str) -> String {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return "请输入要判定的陈述句，例如：判断：张三是李四的师傅".to_string();
        }
        let results = self.check_statement(trimmed);
        if results.is_empty() {
            if let Some(appendix) = self.backward_chain_appendix(trimmed) {
                return appendix;
            }
            return format!(
                "形式化失败：未能把「{}」解析为可判定命题。\n  可用句式：S的P是O / O是S的P / S是类型 / S P O（P 为图谱已知谓词）。",
                trimmed
            );
        }
        let mut output = format!(
            "形式化语义分析：「{}」解析出 {} 个候选命题，逐条给出蕴含判定（证据可追溯）：\n",
            trimmed,
            results.len()
        );
        output.push_str(&semantic_entailment::render_results(&results));
        if results
            .iter()
            .all(|result| result.verdict == Verdict::Unknown)
        {
            if let Some(appendix) = self.backward_chain_appendix(trimmed) {
                output.push('\n');
                output.push_str(&appendix);
            }
        }
        output
    }

    // ------------------------- 溯因推理·假设生成 -------------------------

    fn proposition_atom(&self, proposition: &semantic_entailment::Proposition) -> Atom {
        match proposition.kind {
            semantic_entailment::PropositionKind::TypeMembership => {
                Atom::rel(PROOF_TYPE, &proposition.subject, &proposition.object)
            }
            _ => Atom::rel(
                &proposition.predicate,
                &proposition.subject,
                &proposition.object,
            ),
        }
    }

    /// 把陈述句形式化为归结原子（否定句返回 None）
    fn statement_atom(&self, text: &str) -> Option<Atom> {
        let parser = self.proposition_parser();
        let propositions = parser.parse(text);
        let proposition = propositions.iter().find(|item| !item.negative)?;
        Some(self.proposition_atom(proposition))
    }

    /// 把（可能含多个条件的）文本切分为归结原子列表
    fn condition_atoms(&self, text: &str) -> Vec<Atom> {
        let mut atoms: Vec<Atom> = Vec::new();
        let mut candidates: Vec<String> = text
            .split(['；', ';', '，', ','])
            .map(|part| part.trim().to_string())
            .filter(|part| !part.is_empty())
            .collect();
        if candidates.is_empty() {
            candidates.push(text.trim().to_string());
        }
        let parser = self.proposition_parser();
        for part in &candidates {
            for proposition in parser.parse(part) {
                if proposition.negative {
                    continue;
                }
                let atom = self.proposition_atom(&proposition);
                if !atoms.contains(&atom) {
                    atoms.push(atom);
                }
            }
        }
        if atoms.is_empty() {
            if let Some(atom) = self.statement_atom(text) {
                atoms.push(atom);
            }
        }
        atoms
    }

    /// 把一条谓词链落成归结目标；`final_term` 为链尾（常量或答案变量）。
    fn chain_goals_to(entity: &str, predicates: &[ChainPredicate], final_term: Term) -> Vec<Atom> {
        let mut goals = Vec::new();
        let mut previous = Term::constant(entity);
        for (index, predicate) in predicates.iter().enumerate() {
            let next = if index + 1 == predicates.len() {
                final_term.clone()
            } else {
                Term::var(&format!("V{}", index + 2))
            };
            let atom = if predicate.inverted {
                Atom::pattern(&predicate.slot, next.clone(), previous.clone())
            } else {
                Atom::pattern(&predicate.slot, previous.clone(), next.clone())
            };
            goals.push(atom);
            previous = next;
        }
        goals
    }

    /// 把问句/陈述解析为溯因目标（开放问句 → 谓词链目标；陈述句 → 单原子目标）。
    ///
    /// 覆盖的计划类型刻意与普通问答一致：开放问句、校验句、反查句和计数句
    /// 都能在证不出时进入同一条失败前沿，而不是只有 `FindObject/Exist` 能溯因。
    fn abduction_target(&self, trimmed: &str) -> Option<(Vec<Atom>, Vec<String>, String)> {
        if let Some(plan) = self.planner.parse(trimmed) {
            match plan {
                QueryPlan::FindObject { segments } => {
                    if let Some((entity, predicates)) = self.split_chain(&segments) {
                        if !predicates.is_empty()
                            && self.is_known_entity(&entity)
                            && predicates
                                .iter()
                                .all(|p| self.is_known_predicate(&p.surface))
                        {
                            let answer_var = "答案".to_string();
                            let goals =
                                Self::chain_goals_to(&entity, &predicates, Term::var(&answer_var));
                            let phrase = format!(
                                "{}的{}是谁",
                                entity,
                                predicates
                                    .iter()
                                    .map(|p| p.surface.as_str())
                                    .collect::<Vec<_>>()
                                    .join("的")
                            );
                            return Some((goals, vec![answer_var], phrase));
                        }
                    }
                }
                QueryPlan::Exist { subject, predicate } => {
                    let entity = self.aligner.canonical_form(subject.trim());
                    let (slot, inverted) = self.resolve_question_predicate(&predicate);
                    if !entity.is_empty() && !slot.is_empty() {
                        let atom = if inverted {
                            Atom::pattern(&slot, Term::var("答案"), Term::constant(&entity))
                        } else {
                            Atom::pattern(&slot, Term::constant(&entity), Term::var("答案"))
                        };
                        return Some((
                            vec![atom],
                            vec!["答案".to_string()],
                            format!("{}的{}是否存在", entity, slot),
                        ));
                    }
                }
                QueryPlan::FindObjectByPredicate { subject, predicate } => {
                    let entity = self.aligner.canonical_form(subject.trim());
                    let (slot, inverted) = self.resolve_question_predicate(&predicate);
                    if !entity.is_empty()
                        && !slot.is_empty()
                        && self.is_known_entity(&entity)
                        && self.is_known_predicate(&predicate)
                    {
                        let atom = if inverted {
                            Atom::pattern(&slot, Term::var("答案"), Term::constant(&entity))
                        } else {
                            Atom::pattern(&slot, Term::constant(&entity), Term::var("答案"))
                        };
                        return Some((
                            vec![atom],
                            vec!["答案".to_string()],
                            format!("{}的{}是谁", entity, predicate.trim()),
                        ));
                    }
                }
                QueryPlan::FindSubject { object, predicate } => {
                    let object = self.aligner.canonical_form(object.trim());
                    let (slot, inverted) = self.resolve_question_predicate(&predicate);
                    if !object.is_empty()
                        && !slot.is_empty()
                        && self.is_known_entity(&object)
                        && self.is_known_predicate(&predicate)
                    {
                        // 正常词面：答案在主语位；逆向词面（徒弟↔师傅）：答案在宾语位。
                        let atom = if inverted {
                            Atom::pattern(&slot, Term::constant(&object), Term::var("答案"))
                        } else {
                            Atom::pattern(&slot, Term::var("答案"), Term::constant(&object))
                        };
                        return Some((
                            vec![atom],
                            vec!["答案".to_string()],
                            format!("谁的{}是{}", predicate.trim(), object),
                        ));
                    }
                }
                QueryPlan::Count { subject, predicate } => {
                    let entity = self.aligner.canonical_form(subject.trim());
                    let (slot, inverted) = self.resolve_question_predicate(&predicate);
                    if !entity.is_empty()
                        && !slot.is_empty()
                        && self.is_known_entity(&entity)
                        && self.is_known_predicate(&predicate)
                    {
                        let atom = if inverted {
                            Atom::pattern(&slot, Term::var("答案"), Term::constant(&entity))
                        } else {
                            Atom::pattern(&slot, Term::constant(&entity), Term::var("答案"))
                        };
                        return Some((
                            vec![atom],
                            vec!["答案".to_string()],
                            format!("{}的{}有几个", entity, predicate.trim()),
                        ));
                    }
                }
                QueryPlan::CheckTriple {
                    subject_segments,
                    object,
                } => {
                    let statement = format!("{}是{}", subject_segments.join("的"), object);
                    if let Some((entity, predicates, object)) =
                        self.parse_chain_statement(&statement)
                    {
                        if !predicates.is_empty()
                            && self.is_known_entity(&entity)
                            && predicates
                                .iter()
                                .all(|predicate| self.is_known_predicate(&predicate.surface))
                        {
                            let phrase = format!(
                                "{}的{}是{}",
                                entity,
                                predicates
                                    .iter()
                                    .map(|predicate| predicate.surface.as_str())
                                    .collect::<Vec<_>>()
                                    .join("的"),
                                object
                            );
                            let goals =
                                Self::chain_goals_to(&entity, &predicates, Term::constant(&object));
                            return Some((goals, Vec::new(), phrase));
                        }
                    }
                }
                QueryPlan::CheckRelation {
                    subject,
                    predicate,
                    object,
                    negated,
                } => {
                    // 否定校验不能靠“补一条事实”解释：缺的是否定证据，不是正向前提。
                    if !negated {
                        let subject = self.aligner.canonical_form(subject.trim());
                        let object = self.aligner.canonical_form(object.trim());
                        let (slot, inverted) = self.resolve_question_predicate(&predicate);
                        if !subject.is_empty()
                            && !object.is_empty()
                            && !slot.is_empty()
                            && self.is_known_entity(&subject)
                            && self.is_known_entity(&object)
                            && self.is_known_predicate(&predicate)
                        {
                            let atom = if inverted {
                                Atom::pattern(
                                    &slot,
                                    Term::constant(&object),
                                    Term::constant(&subject),
                                )
                            } else {
                                Atom::pattern(
                                    &slot,
                                    Term::constant(&subject),
                                    Term::constant(&object),
                                )
                            };
                            return Some((
                                vec![atom],
                                Vec::new(),
                                format!("{}的{}是{}", subject, predicate.trim(), object),
                            ));
                        }
                    }
                }
                _ => {}
            }
        }
        let atom = self.statement_atom(trimmed)?;
        let phrase = self.render_atom_nl(&atom);
        Some((vec![atom], Vec::new(), phrase))
    }

    /// 统一渲染溯因报告：非空假设（含联合场景成员）存为待定，供「采纳假设 N / 澄清」
    fn finish_abduction_report(
        &mut self,
        report: AbductionReport,
        phrase: &str,
        goals: &[Atom],
        query_vars: &[String],
    ) -> String {
        if report.provable {
            // 目标已可证：上一轮的待定假设与澄清问题全部过期，不能继续追问。
            self.abduction.clear_pending();
            self.clarification = None;
            self.last_abduction_goal = None;
            self.last_abduction_vars.clear();
            let render = |atom: &Atom| self.render_atom_nl(atom);
            let mut output = abduction::render_report(&report, &render);
            if !phrase.is_empty() {
                output = format!("（问题：{}）\n{}", phrase, output);
            }
            return output;
        }
        if !goals.is_empty() {
            self.last_abduction_goal = Some(goals.to_vec());
            self.last_abduction_vars = query_vars.to_vec();
        }
        // 联合解释场景（多假设协同 + 注入验证）：成员也并入待定池
        //（`report.provable == false` 已确认目标不可证）
        let scenarios =
            self.abduction
                .scenarios_assuming_unprovable(&self.resolver, goals, query_vars);
        // 循环假设过滤："假设目标本身成立"不算解释，不进入待定池/澄清队列
        let mut hypotheses: Vec<abduction::Hypothesis> = report
            .hypotheses
            .iter()
            .filter(|hypothesis| !goals.iter().any(|goal| goal == &hypothesis.atom))
            .cloned()
            .collect();
        for scenario in &scenarios {
            for hypothesis in &scenario.hypotheses {
                if !hypotheses
                    .iter()
                    .any(|existing| existing.atom == hypothesis.atom)
                {
                    hypotheses.push(hypothesis.clone());
                }
            }
        }
        self.abduction.set_pending(hypotheses);
        let render = |atom: &Atom| self.render_atom_nl(atom);
        let mut output = abduction::render_report(&report, &render);
        // 只有当"多条假设协同"确实能验证目标时才展示场景，避免噪音
        let joint: Vec<Scenario> = scenarios
            .into_iter()
            .filter(|scenario| scenario.verified && scenario.size() >= 2)
            .collect();
        if !joint.is_empty() {
            output.push('\n');
            output.push_str(&abduction::render_scenarios(
                &report.goal,
                &report.frontier,
                &joint,
                &render,
            ));
        }
        if !phrase.is_empty() {
            output = format!("（问题：{}）\n{}", phrase, output);
        }
        output
    }

    /// 溯因陈述：能证明就给证明；不能证明就做失败前沿分析并生成候选假设
    pub fn abduce_statement(&mut self, text: &str) -> Option<String> {
        if !self.config.enable_abduction {
            return None;
        }
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return None;
        }
        let atom = self.statement_atom(trimmed)?;
        let goals = vec![atom.clone()];
        let report = self.abduction.analyze(&self.resolver, &goals, &[]);
        Some(self.finish_abduction_report(report, "", &goals, &[]))
    }

    /// 溯因问句：把开放问句拆成谓词链目标，失败前沿生成候选答案假设
    pub fn hypothesis_answer(&mut self, question: &str) -> Option<String> {
        if !self.config.enable_abduction {
            return None;
        }
        let trimmed = question
            .trim()
            .trim_end_matches(['？', '?', '。', '！', '!', ' ']);
        if trimmed.is_empty() {
            return None;
        }
        let (goals, query_vars, phrase) = self.abduction_target(trimmed)?;
        let report = self.abduction.analyze(&self.resolver, &goals, &query_vars);
        let solution = report.solutions.first().cloned();
        let mut output = self.finish_abduction_report(report, &phrase, &goals, &query_vars);
        if let Some(solution) = solution {
            output = if let Some(answer) = solution.bindings.get("答案") {
                format!(
                    "后向链推理给出答案：「{}」（证明置信度 {:.2}）\n{}",
                    answer, solution.confidence, output
                )
            } else {
                // 陈述句目标没有答案变量，展示"已证明"而不是伪造「?」答案
                format!(
                    "后向链推理已证明该陈述（证明置信度 {:.2}）\n{}",
                    solution.confidence, output
                )
            };
        }
        Some(output)
    }

    /// 生成式联合解释：枚举假设组合 + 注入克隆引擎验证 + 输出可证场景
    pub fn scenario_report(&mut self, text: &str) -> String {
        if !self.config.enable_abduction {
            return "溯因推理未启用（config.toml [reasoning] enable_abduction = false）。"
                .to_string();
        }
        let trimmed = text
            .trim()
            .trim_end_matches(['？', '?', '。', '！', '!', ' ']);
        if trimmed.is_empty() {
            return "请输入要解释的陈述或问句，例如：场景：张三的师祖是赵六".to_string();
        }
        let Some((goals, query_vars, phrase)) = self.abduction_target(trimmed) else {
            return format!(
                "无法把「{}」形式化为可联合解释的目标。支持句式：S的P是O / S的P是谁。",
                trimmed
            );
        };
        let report = self.abduction.analyze(&self.resolver, &goals, &query_vars);
        if report.provable {
            let render = |atom: &Atom| self.render_atom_nl(atom);
            return format!(
                "目标「{}」已可由现有知识证明，无需联合假设：\n{}",
                phrase,
                abduction::render_report(&report, &render)
            );
        }
        self.last_abduction_goal = Some(goals.clone());
        self.last_abduction_vars = query_vars.clone();
        //（`report.provable == false` 已确认目标不可证）
        let scenarios =
            self.abduction
                .scenarios_assuming_unprovable(&self.resolver, &goals, &query_vars);
        // 循环假设过滤："假设目标本身成立"不算解释，不进入待定池/澄清队列
        let mut pending: Vec<Hypothesis> = report
            .hypotheses
            .iter()
            .filter(|hypothesis| !goals.iter().any(|goal| goal == &hypothesis.atom))
            .cloned()
            .collect();
        for scenario in &scenarios {
            for hypothesis in &scenario.hypotheses {
                if !pending
                    .iter()
                    .any(|existing| existing.atom == hypothesis.atom)
                {
                    pending.push(hypothesis.clone());
                }
            }
        }
        self.abduction.set_pending(pending);
        let render = |atom: &Atom| self.render_atom_nl(atom);
        abduction::render_scenarios(&report.goal, &report.frontier, &scenarios, &render)
    }

    /// 把断言事实中的新实体/新谓词登记进词表：后续解析、意图规划与实体链接才能识别它们
    fn register_atom_vocabulary(&mut self, atom: &Atom) -> bool {
        let mut changed = false;
        if !atom.predicate.trim().is_empty() && !self.aligner.is_registered(&atom.predicate) {
            self.aligner
                .register_term(&atom.predicate, TermKind::Relation);
            changed = true;
        }
        for term in &atom.terms {
            if let Term::Const(name) = term {
                if !name.trim().is_empty() && !self.aligner.is_registered(name) {
                    self.aligner.register_term(name, TermKind::Entity);
                    self.linker.register_entity(name, None);
                    changed = true;
                }
            }
        }
        if changed {
            self.rebuild_knowledge_caches();
        }
        changed
    }

    /// 判断基原子是否已由知识图谱独立支持（三元组、属性或类型）。
    /// 会话事实可能与图谱事实同名；撤回会话副本时不能把图谱那一份一并删掉。
    fn graph_supports_atom(&self, atom: &Atom) -> bool {
        let [subject, object] = atom.terms.as_slice() else {
            return false;
        };
        let (Term::Const(subject), Term::Const(object)) = (subject, object) else {
            return false;
        };
        let subject: &str = subject;
        let object: &str = object;
        let predicate: &str = &atom.predicate;
        if predicate == PROOF_HAS {
            return self
                .predictor
                .triples()
                .iter()
                .any(|triple| triple.subject == subject && triple.predicate == object)
                || self
                    .attributes
                    .iter()
                    .any(|(entity, key, _)| entity == subject && key == object);
        }
        if predicate == PROOF_TYPE {
            return self
                .generator
                .typed_entities()
                .iter()
                .any(|(entity, etype)| entity == subject && etype == object);
        }
        self.predictor.triples().iter().any(|triple| {
            triple.subject == subject && triple.predicate == predicate && triple.object == object
        }) || self.attributes.contains(&(
            subject.to_string(),
            predicate.to_string(),
            object.to_string(),
        ))
    }

    /// 二元常原子 → 证据三元组；其他元数或含变量的原子不进入证据索引。
    fn session_atom_triple(atom: &Atom) -> Option<(&str, &str, &str)> {
        match atom.terms.as_slice() {
            [Term::Const(subject), Term::Const(object)] => {
                Some((subject.as_ref(), atom.predicate.as_ref(), object.as_ref()))
            }
            _ => None,
        }
    }

    fn index_session_atom(&mut self, atom: &Atom) {
        if let Some((subject, predicate, object)) = Self::session_atom_triple(atom) {
            self.retrieval
                .lock()
                .observe_triple(subject, predicate, object);
        }
    }
}

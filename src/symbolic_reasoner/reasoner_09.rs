impl SymbolicReasoner {
    /// 符号梯度归因报告（"可微推理"的纯符号实现：解析求导，只解释与排序求证，不训练）
    pub fn gradient_report(&self, target_raw: &str) -> String {
        if !self.config.enable_gradient_attribution {
            return "符号梯度归因未启用（config.toml [reasoning] enable_gradient_attribution = false）。"
                .to_string();
        }
        if !self.config.enable_soft_logic {
            return "符号梯度归因依赖加权软逻辑，请先启用 enable_soft_logic。".to_string();
        }
        let target = target_raw.trim();
        if target.is_empty() {
            return "请输入要归因的对象，例如：梯度归因：李四的师祖 或 梯度归因：张三".to_string();
        }
        let Some((entity, predicate)) = self.split_entity_predicate(target) else {
            return format!("无法在「{}」中识别出已知实体。", target);
        };
        let pattern = match predicate.as_deref() {
            Some(pred) if !pred.is_empty() => {
                Atom::pattern(pred, Term::constant(&entity), Term::var("X"))
            }
            _ => {
                let Some(candidate) = self
                    .collect_soft_facts_for_entity(&entity, 8)
                    .into_iter()
                    .find(|fact| !fact.from_fact)
                else {
                    return format!(
                        "「{}」目前只有图谱事实、没有加权推测，无需梯度归因。",
                        entity
                    );
                };
                candidate.atom
            }
        };
        let mut attributions = self.soft.gradient(&pattern, 3);
        if attributions.is_empty() {
            if let Some(pred) = predicate.as_deref().filter(|pred| !pred.is_empty()) {
                attributions = self.soft.gradient(
                    &Atom::pattern(pred, Term::var("X"), Term::constant(&entity)),
                    3,
                );
            }
        }
        if attributions.is_empty() {
            return format!(
                "没有可做梯度归因的加权推测（「{}」未被任何规则支持）。",
                entity
            );
        }
        let mut output = format!(
            "符号梯度归因（可微推理的纯符号版：∂P/∂w = P(1−P)·前件满足度；解析求导、不训练、无神经网络）\n"
        );
        for (index, attribution) in attributions.iter().enumerate() {
            output.push_str(&format!(
                "  {}. {} {}（P={:.3}，log-odds {:+.3}）\n",
                index + 1,
                self.render_atom_nl(&attribution.atom),
                MarkovLogicEngine::hedge(attribution.probability),
                attribution.probability,
                attribution.log_odds
            ));
            for contribution in attribution.contributions.iter().take(4) {
                let premises = if contribution.premises.is_empty() {
                    "（无前件）".to_string()
                } else {
                    contribution
                        .premises
                        .iter()
                        .map(|premise| self.render_atom_nl(premise))
                        .collect::<Vec<_>>()
                        .join(" 且 ")
                };
                output.push_str(&format!(
                    "     规则贡献 ∂P/∂w={:+.4}（权重 {:+.2}，前件满足度 {:.2}）：{}\n       来源：{}\n",
                    contribution.gradient,
                    contribution.weight,
                    contribution.satisfaction,
                    premises,
                    contribution.source
                ));
            }
            let uncertain: Vec<_> = attribution
                .premise_sensitivity
                .iter()
                .filter(|sensitivity| sensitivity.truth < 0.999)
                .take(3)
                .collect();
            if !uncertain.is_empty() {
                output.push_str("     最值得求证的前提（按 |∂P/∂s|·(1−真值) 排序）：\n");
                for sensitivity in uncertain {
                    output.push_str(&format!(
                        "       • {}（当前软真值 {:.2}，∂P/∂s={:+.3}，求证优先级 {:.3}）\n",
                        self.render_atom_nl(&sensitivity.premise),
                        sensitivity.truth,
                        sensitivity.gradient,
                        sensitivity.verify_priority
                    ));
                }
            }
        }
        output.trim_end().to_string()
    }

    /// 从本体对齐器、生成器、会话事实与因果图构建因果分词器：
    /// 词表锚（实体/类型/谓词/同义词）+ 每条因果机制的权重质量
    fn build_causal_tokenizer(&self, graph: &CausalGraph) -> CausalTokenizer {
        let mut tokenizer = CausalTokenizer::new(MAX_TOKEN_CHARS);
        // 本体对齐器词条：实体/类型/关系/属性
        let mut profiles: Vec<(&str, AnchorKind, String)> = self
            .aligner
            .profiles()
            .iter()
            .map(|profile| {
                let kind = match profile.kind {
                    TermKind::Entity => AnchorKind::Entity,
                    TermKind::Type => AnchorKind::Type,
                    TermKind::Relation | TermKind::Attribute => AnchorKind::Predicate,
                };
                (
                    profile.term.as_str(),
                    kind,
                    self.aligner.canonical_form(&profile.term),
                )
            })
            .collect();
        profiles.sort_by(|a, b| a.0.cmp(b.0));
        for (term, kind, canonical) in profiles {
            tokenizer.add_anchor(term, &canonical, kind, 0.95);
            if canonical != term {
                tokenizer.add_anchor(&canonical, &canonical, kind, 1.0);
            }
        }
        // 同义词表：别名 → 规范词
        let mut aliases: Vec<(&String, &String)> = self.aligner.synonyms().iter().collect();
        aliases.sort();
        for (alias, canonical) in aliases {
            tokenizer.add_alias(alias, canonical);
        }
        // 生成器的谓词表面形式（动词化表述也参与锚定）
        let mut surfaces: Vec<(&String, &String)> =
            self.generator.predicate_surfaces().iter().collect();
        surfaces.sort();
        for (surface, canonical) in surfaces {
            tokenizer.add_anchor(surface, canonical, AnchorKind::Predicate, 0.7);
        }
        // 图谱事实：孤立事实同样具有词表地位
        for atom in self.soft.facts() {
            tokenizer.observe_fact(atom);
        }
        // 因果图：每条有符号机制为两侧原子累计因果质量
        for edge in graph.edges() {
            tokenizer.observe_causal_edge(edge);
        }
        tokenizer
    }

    /// 语境分词：把词表、类型、谓词、三元组与当前焦点投影到同一符号坐标系，
    /// 再用字符格 k-best 解码；只读，不写图谱，不改写回答主路径。
    pub fn contextual_segment_report(&self, input_raw: &str) -> String {
        let text = input_raw.trim();
        if text.is_empty() {
            return "请输入要分词的文本，例如：语境分词：张三的师傅是谁".to_string();
        }
        let (focus_entities, expected_types) = self.contextual_focus();
        let options = self.contextual_segment_options(&focus_entities, &expected_types);
        let mut segmenter = self.build_contextual_segmenter();
        let result = segmenter.segment(text, &options);
        let mut output = format!(
            "语境分词（符号子词嵌入 + 字符格 k-best + 焦点语境；词表 {} 个词面 / 类型 {} / 谓词 {} / 三元组 {}；纯符号、零训练）\n",
            segmenter.surface_count(),
            segmenter.type_count(),
            segmenter.predicate_count(),
            segmenter.triple_count()
        );
        if focus_entities.is_empty() {
            output.push_str("焦点：∅\n");
        } else {
            output.push_str(&format!(
                "焦点：{}；期望类型：{}\n",
                focus_entities.join("、"),
                if expected_types.is_empty() {
                    "∅".to_string()
                } else {
                    expected_types.join("、")
                }
            ));
        }
        output.push_str(&result.render_report());
        output
    }

    /// 联合解析：先抽关系帧，再让帧的定义域/值域与参数跨度反向约束分词，
    /// 最后用类型裁决、读法归一与失败回灌收敛。只读诊断，不改回答主路径。
    pub fn joint_parse_report(&self, input_raw: &str) -> String {
        let (focus, expected) = self.contextual_focus();
        self.joint_parse_report_with_context(
            input_raw,
            &focus,
            &expected,
            self.context.last_predicate(),
            crate::contextual_segmentation::DEFAULT_AMBIGUITY_MARGIN,
        )
    }

    /// 联合解析并把无硬冲突的帧回写到对话上下文，供后续指代与生成复用。
    pub fn joint_parse_report_and_observe(&mut self, input_raw: &str) -> String {
        let (focus, expected) = self.contextual_focus();
        let predicate = self.context.last_predicate().map(str::to_string);
        let parsed = self.joint_parse_result(
            input_raw,
            &focus,
            &expected,
            predicate.as_deref(),
            crate::contextual_segmentation::DEFAULT_AMBIGUITY_MARGIN,
        );
        let healthy: Vec<&JointFrame> = parsed
            .frames
            .iter()
            .filter(|frame| !frame.is_hard_failure())
            .collect();
        if !healthy.is_empty() {
            let mut entities = Vec::with_capacity(healthy.len() * 2);
            for frame in &healthy {
                entities.push((
                    frame.subject.canonical.clone(),
                    frame.subject_types.first().cloned(),
                ));
                entities.push((
                    frame.object.canonical.clone(),
                    frame.object_types.first().cloned(),
                ));
            }
            self.context.record_entities_typed(&entities);
            if let Some(frame) = healthy.iter().max_by(|left, right| {
                left.original
                    .confidence
                    .total_cmp(&right.original.confidence)
                    .then_with(|| right.original.id.cmp(&left.original.id))
            }) {
                self.context.record_predicate(&frame.normalized.predicate);
            }
        }
        parsed.render_report()
    }

    /// 与 [`Self::joint_parse_report`] 相同，但允许命令行显式给出焦点/类型/谓词。
    pub fn joint_parse_report_with_context(
        &self,
        input_raw: &str,
        focus: &[String],
        expected_types: &[String],
        predicate: Option<&str>,
        margin: f64,
    ) -> String {
        self.joint_parse_result(input_raw, focus, expected_types, predicate, margin)
            .render_report()
    }

    /// 返回结构化联合解析结果，供 CLI `--json` 与生成回译验证复用。
    pub fn joint_parse_result(
        &self,
        input_raw: &str,
        focus: &[String],
        expected_types: &[String],
        predicate: Option<&str>,
        margin: f64,
    ) -> JointParse {
        let text = input_raw.trim();
        let context = JointContext::new(
            focus.to_vec(),
            expected_types.to_vec(),
            predicate.map(str::to_string),
        );
        let options = JointParseOptions {
            context,
            type_hints: self.joint_type_hints(),
            ambiguity_margin: margin.clamp(0.0, 1.0),
            ..JointParseOptions::default()
        };
        let mut segmenter = self.build_contextual_segmenter();
        joint_parse(&mut segmenter, &self.schema_view, text, &options)
    }

    /// 用现有生成器把肯定直陈帧渲染成候选句；否定/模态帧暂不猜测措辞。
    pub fn joint_generation_candidate(&self, frames: &[RelationFrame]) -> Option<String> {
        if frames.is_empty() {
            return None;
        }
        let mut sentences = Vec::with_capacity(frames.len());
        for frame in frames {
            if frame.polarity != crate::relation_frame::Polarity::Affirmative
                || frame.modality != crate::relation_frame::Modality::Asserted
            {
                return None;
            }
            let rendered = self.generator.render_fact(
                &frame.subject,
                &frame.predicate,
                &[frame.object.clone()],
            );
            let rendered = rendered.trim().trim_end_matches('。');
            if rendered.is_empty() {
                return None;
            }
            sentences.push(format!("{}。", rendered));
        }
        Some(sentences.join(""))
    }

    /// 生成候选并立即回译验证；返回候选句与验证结果。
    pub fn joint_generate_and_verify(
        &self,
        expected: &[RelationFrame],
    ) -> Option<(String, FrameVerification)> {
        let (focus, expected_types) = self.contextual_focus();
        self.joint_generate_and_verify_with_context(
            expected,
            &focus,
            &expected_types,
            self.context.last_predicate(),
        )
    }

    /// 与 [`Self::joint_generate_and_verify`] 相同，但允许显式给出联合上下文。
    pub fn joint_generate_and_verify_with_context(
        &self,
        expected: &[RelationFrame],
        focus: &[String],
        expected_types: &[String],
        predicate: Option<&str>,
    ) -> Option<(String, FrameVerification)> {
        let candidate = self.joint_generation_candidate(expected)?;
        let verification = self.verify_joint_generation_with_context(
            &candidate,
            expected,
            focus,
            expected_types,
            predicate,
        );
        Some((candidate, verification))
    }

    /// 生成侧回译验证：把候选句重新联合解析，要求恢复同一组关系帧。
    pub fn verify_joint_generation(
        &self,
        candidate: &str,
        expected: &[RelationFrame],
    ) -> FrameVerification {
        let (focus, expected_types) = self.contextual_focus();
        self.verify_joint_generation_with_context(
            candidate,
            expected,
            &focus,
            &expected_types,
            self.context.last_predicate(),
        )
    }

    /// 与 [`Self::verify_joint_generation`] 相同，但允许显式给出联合上下文。
    pub fn verify_joint_generation_with_context(
        &self,
        candidate: &str,
        expected: &[RelationFrame],
        focus: &[String],
        expected_types: &[String],
        predicate: Option<&str>,
    ) -> FrameVerification {
        let parsed = self.joint_parse_result(
            candidate,
            focus,
            expected_types,
            predicate,
            crate::contextual_segmentation::DEFAULT_AMBIGUITY_MARGIN,
        );
        verify_frames(&self.schema_view, &parsed, expected)
    }

    /// 把生成器/对齐器/链接器里的实体类型证据投影成只读类型提示。
    fn joint_type_hints(&self) -> BTreeMap<String, BTreeSet<String>> {
        let mut hints: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for (entity, etype) in self.generator.typed_entities() {
            insert_type_hint(&mut hints, &entity, &etype);
        }
        for entity in self.context.focus_entities() {
            if let Some(etype) = self.context.focus_type(entity) {
                insert_type_hint(&mut hints, entity, etype);
            }
        }
        for profile in self.aligner.profiles() {
            if profile.kind != TermKind::Entity {
                continue;
            }
            let canonical = self.aligner.canonical_form(&profile.term);
            let canonical = if canonical.is_empty() {
                profile.term.as_str()
            } else {
                canonical.as_str()
            };
            for etype in &profile.types {
                insert_type_hint(&mut hints, canonical, etype);
                insert_type_hint(&mut hints, &profile.term, etype);
            }
            if let Some(etype) = self
                .generator
                .entity_type_of(canonical)
                .or_else(|| self.linker.entity_type(canonical))
            {
                insert_type_hint(&mut hints, canonical, etype);
                insert_type_hint(&mut hints, &profile.term, etype);
            }
        }
        let mut aliases: Vec<(&String, &String)> = self.aligner.synonyms().iter().collect();
        aliases.sort();
        for (alias, canonical) in aliases {
            if self.aligner.kind_of(canonical) != Some(TermKind::Entity) {
                continue;
            }
            if let Some(types) = hints.get(canonical).cloned() {
                for etype in types {
                    insert_type_hint(&mut hints, alias, &etype);
                }
            }
        }
        hints
    }

    /// 确认当前语境下某个词面的读法；写入会话级读法记忆。
    pub fn confirm_contextual_reading(&mut self, surface: &str, canonical: &str) -> String {
        let surface = surface.trim();
        let canonical = canonical.trim();
        if surface.is_empty() || canonical.is_empty() {
            return "请提供「词面=规范读法」，例如：语境分词确认：苹果=苹果公司".to_string();
        }
        let (focus, expected) = self.contextual_focus();
        let options = self.contextual_segment_options(&focus, &expected);
        let mut segmenter = self.build_contextual_segmenter();
        let result = segmenter.segment(surface, &options);
        let kind = segmenter
            .reading_kind(surface, canonical)
            .unwrap_or(ReadingKind::Entity);
        self.contextual_memory
            .confirm(surface, canonical, kind, &result.context.fingerprint);
        format!(
            "读法记忆已确认：「{}」→「{}」（{}）；语境 {}。",
            surface,
            canonical,
            kind.label(),
            if result.context.fingerprint.is_empty() {
                "∅".to_string()
            } else {
                result.context.fingerprint
            }
        )
    }

    /// 纠正当前语境下某个词面的读法：否定旧读法，采纳新读法。
    pub fn correct_contextual_reading(
        &mut self,
        surface: &str,
        rejected: &str,
        accepted: &str,
    ) -> String {
        let surface = surface.trim();
        let rejected = rejected.trim();
        let accepted = accepted.trim();
        if surface.is_empty() || rejected.is_empty() || accepted.is_empty() {
            return "请提供「词面=旧读法>新读法」，例如：语境分词纠正：苹果=苹果>苹果公司"
                .to_string();
        }
        let (focus, expected) = self.contextual_focus();
        let options = self.contextual_segment_options(&focus, &expected);
        let mut segmenter = self.build_contextual_segmenter();
        let result = segmenter.segment(surface, &options);
        let rejected_kind = segmenter
            .reading_kind(surface, rejected)
            .unwrap_or(ReadingKind::Entity);
        let accepted_kind = segmenter
            .reading_kind(surface, accepted)
            .unwrap_or(rejected_kind);
        self.contextual_memory.correct(
            surface,
            rejected,
            rejected_kind,
            accepted,
            accepted_kind,
            &result.context.fingerprint,
        );
        format!(
            "读法记忆已纠正：「{}」：{} → {}；语境 {}。",
            surface,
            rejected,
            accepted,
            if result.context.fingerprint.is_empty() {
                "∅".to_string()
            } else {
                result.context.fingerprint
            }
        )
    }

    pub fn contextual_reading_memory_report(&self) -> String {
        self.contextual_memory.render_report()
    }

    pub fn clear_contextual_reading_memory(&mut self) -> String {
        self.contextual_memory.clear();
        "读法记忆已清空。".to_string()
    }

    fn contextual_focus(&self) -> (Vec<String>, Vec<String>) {
        // 焦点只取显式对话焦点栈：若拿输入文本中的实体回填，歧义词面会被
        // 自身同义词反向锚定，读法记忆也就失去「由语境裁决」的意义。
        let focus_entities = self.context.focus_entities().to_vec();
        let mut expected_types: Vec<String> = Vec::new();
        for entity in &focus_entities {
            if let Some(etype) = self
                .context
                .focus_type(entity)
                .or_else(|| self.generator.entity_type_of(entity))
                .or_else(|| self.linker.entity_type(entity))
            {
                if !expected_types.iter().any(|known| known == etype) {
                    expected_types.push(etype.to_string());
                }
            }
        }
        (focus_entities, expected_types)
    }

    fn contextual_segment_options(&self, focus: &[String], expected: &[String]) -> SegmentOptions {
        SegmentOptions {
            context_terms: focus.to_vec(),
            expected_types: expected.to_vec(),
            predicate: self.context.last_predicate().map(str::to_string),
            ..SegmentOptions::default()
        }
    }

    fn build_contextual_segmenter(&self) -> ContextualSegmenter {
        let mut segmenter = ContextualSegmenter::new();
        segmenter.set_schema(self.schema_view.clone());
        *segmenter.memory_mut() = self.contextual_memory.clone();
        let mut profiles: Vec<_> = self.aligner.profiles().iter().collect();
        profiles.sort_by(|left, right| left.term.cmp(&right.term));
        for profile in profiles {
            let canonical = self.aligner.canonical_form(&profile.term);
            let canonical = if canonical.is_empty() {
                profile.term.as_str()
            } else {
                canonical.as_str()
            };
            match profile.kind {
                TermKind::Entity => {
                    let fallback_type = profile.types.iter().min().cloned();
                    let etype = self
                        .generator
                        .entity_type_of(canonical)
                        .or_else(|| self.linker.entity_type(canonical))
                        .or(fallback_type.as_deref());
                    segmenter.observe_entity(canonical, etype);
                }
                TermKind::Type => segmenter.observe_type(canonical),
                TermKind::Relation | TermKind::Attribute => {
                    segmenter.observe_predicate(canonical);
                }
            }
        }
        let mut synonyms: Vec<_> = self.aligner.synonyms().iter().collect();
        synonyms.sort_by(|left, right| left.0.cmp(right.0));
        for (alias, canonical) in synonyms {
            match self.aligner.kind_of(canonical) {
                Some(TermKind::Entity) => segmenter.observe_alias(alias, canonical),
                Some(TermKind::Type) => segmenter.observe_type_alias(alias, canonical),
                Some(TermKind::Relation | TermKind::Attribute) => {
                    segmenter.observe_predicate_alias(alias, canonical);
                }
                None => {}
            }
        }
        let mut surfaces: Vec<_> = self.generator.predicate_surfaces().iter().collect();
        surfaces.sort_by(|left, right| left.0.cmp(right.0));
        for (surface, canonical) in surfaces {
            segmenter.observe_predicate_alias(surface, canonical);
        }
        for triple in self.predictor.triples() {
            segmenter.observe_triple(&triple.subject, &triple.predicate, &triple.object);
        }
        segmenter
    }

    /// 因果分词报告：知识/因果图引导的 token 格 k-best DP 分词 + token 级因果归因
    /// （纯符号 attention：结论的概率质量沿因果链回溯摊到输入 token 上）
    pub fn causal_tokenize_report(&self, input_raw: &str) -> String {
        if !self.config.enable_causal_tokenization {
            return "因果分词未启用（config.toml [reasoning] enable_causal_tokenization = false）。"
                .to_string();
        }
        let text = input_raw.trim();
        if text.is_empty() {
            return "请输入要分词的文本，例如：因果分词：张三的师父是谁".to_string();
        }
        let probabilities = self.soft.probability_map();
        let analyzer = CausalAnalyzer::from_engine_with_probabilities(
            &self.soft,
            &probabilities,
            self.config.causal_max_depth,
        );
        let graph = analyzer.graph();
        let tokenizer = self.build_causal_tokenizer(graph);
        let tokenization = tokenizer.tokenize(text);
        let mut output = format!(
            "因果分词（知识/因果图引导的 token 格 k-best DP：锚定支持度 + 因果质量加成；词表 {} 个锚、因果图 {} 条机制；无神经网络）\n输入：{}\n最优分词（得分 {:.2}）：{}\n",
            tokenizer.anchor_count(),
            graph.edge_count(),
            text,
            tokenization.score,
            tokenization.render_segmentation()
        );
        for token in &tokenization.tokens {
            match &token.anchor {
                Some(anchor) => output.push_str(&format!(
                    "   • {} → {}（锚定 {:.2}，机制 {} 条 / 因果质量 {:.2}）\n",
                    token.text,
                    anchor.text(),
                    anchor.support,
                    anchor.causal_edges,
                    anchor.causal_mass
                )),
                None => output.push_str(&format!("   • {} → 未锚定（回退）\n", token.text)),
            }
        }
        if !tokenization.alternatives.is_empty() {
            let rendered: Vec<String> = tokenization
                .alternatives
                .iter()
                .take(2)
                .map(|alt| format!("{}（{:.2}）", alt.tokens.join(" | "), alt.score))
                .collect();
            output.push_str(&format!("备选分词：{}\n", rendered.join("；")));
        }
        // 目标原子：优先按「实体[的谓词]」解析，退而取第一个实体锚
        let target = self
            .split_entity_predicate(text)
            .and_then(|(entity, predicate)| self.find_causal_atom(&entity, predicate.as_deref()))
            .or_else(|| {
                tokenization
                    .anchored_tokens()
                    .filter(|token| {
                        token
                            .anchor
                            .as_ref()
                            .map(|anchor| anchor.kind == AnchorKind::Entity)
                            .unwrap_or(false)
                    })
                    .map(|token| {
                        token
                            .anchor
                            .as_ref()
                            .expect("已按锚定过滤")
                            .canonical
                            .clone()
                    })
                    .next()
                    .and_then(|entity| self.find_causal_atom(&entity, None))
            });
        match target {
            Some(target) => {
                let attributions = attribute_tokens(&tokenization, &self.soft, graph, &target);
                output.push_str(&format!(
                    "token 因果归因（目标：{}）：\n",
                    self.render_atom_nl(&target)
                ));
                if attributions.is_empty() {
                    output.push_str(
                        "   • 目标未落在输入 token 的锚定路径上（结论来自输入之外的图谱事实）。\n",
                    );
                } else {
                    for item in &attributions {
                        output.push_str(&format!(
                            "   • 「{}」 +{:.3}（{:>5.1}%）锚：{}\n",
                            item.token,
                            item.contribution,
                            item.share * 100.0,
                            item.anchor
                        ));
                    }
                    output.push_str(
                        "归因说明：概率质量沿因果链回溯到锚定 token，纯符号 attention、逐行可回放。\n",
                    );
                }
            }
            None => {
                output.push_str("（未在知识库中找到与输入对应的可归因目标）\n");
            }
        }
        output.trim_end().to_string()
    }

    /// 中文概率知识教学：把「如果…那么…」「八成…」等句子解析成 ProbLog 概率知识，
    /// 只写入会话级概率程序（不修改知识图谱），可随对话状态持久化并在下次会话重放
    pub fn teach_report(&mut self, sentence_raw: &str) -> String {
        if !self.config.enable_problog {
            return "概率逻辑教学未启用（config.toml [reasoning] enable_problog = false）。"
                .to_string();
        }
        if !self.config.enable_soft_logic {
            return "概率逻辑教学依赖加权软逻辑（事实与规则同源），请先启用 enable_soft_logic。"
                .to_string();
        }
        let sentence = sentence_raw.trim();
        if sentence.is_empty() {
            return "请用中文教一条概率知识，例如：教学：如果张三的师傅是李四，那么张三的师祖是王五。"
                .to_string();
        }
        if self.lessons.iter().any(|existing| existing == sentence) {
            return format!(
                "这条知识已经学过（重复教学不会重复计概率）：{}。可用「概率推理：实体」查看推断结果。",
                sentence
            );
        }
        let outcome = chinese_reasoning_nlu::parse_lessons(sentence);
        if outcome.lessons.is_empty() {
            let mut output = String::from("没能把这句话解析成概率知识。支持句式：\n");
            output.push_str(
                "  • 条件句：如果S的P是O，那么S的P是O（可带 八成/很可能/百分之八十 等概率词）\n",
            );
            output.push_str("  • 因果句：因为S的P是O，所以S的P是O\n");
            output.push_str("  • 比例句：八成的S会V / S的P是O的可能性是百分之八十\n");
            output.push_str("  • 默认句（可反驳常识）：如果X是鸟，那么X会飞，除非X是企鹅\n");
            output.push_str("  • 直陈句：S的P是O（作为确定事实）");
            if !outcome.errors.is_empty() {
                output.push_str("\n  失败原因：\n");
                for error in &outcome.errors {
                    output.push_str(&format!("   • {}\n", error));
                }
            }
            return output.trim_end().to_string();
        }
        let mut output = format!(
            "已学会 {} 条中文概率知识（纯符号解析 → ProbLog 程序，可回放、不训练）：\n",
            outcome.lessons.len()
        );
        for (index, lesson) in outcome.lessons.iter().enumerate() {
            chinese_reasoning_nlu::apply_lesson(&mut self.problog, lesson);
            output.push_str(&format!(
                "  {}. 原始：{}\n     逻辑形式：{}\n     知识类型：{}；规则来源：{}\n",
                index + 1,
                lesson.original,
                chinese_reasoning_nlu::render_lesson_zh(lesson),
                lesson_kind_label(lesson.kind),
                lesson.source
            ));
        }
        for error in &outcome.errors {
            output.push_str(&format!("  未学会：{}\n", error));
        }
        // 条件/因果知识同时登记为会话 Horn 子句：后向链（判断/证明/为什么/推理链）直接可用
        let added_clauses = self.register_lesson_clauses(&outcome.lessons);
        if !added_clauses.is_empty() {
            output.push_str(&format!(
                "  其中 {} 条规则已进入会话后向链规则库（含占位符的自动变量化，可用「判断/为什么/推理链」验证）。\n",
                added_clauses.len()
            ));
        }
        self.lessons.push(sentence.to_string());
        // 与上面 apply_lesson 的重放保持一致
        self.lessons_parsed.extend(outcome.lessons.iter().cloned());
        output.push_str(&format!(
            "  概率程序现有 {} 条确定事实、{} 条概率事实、{} 条概率规则；用「概率推理：实体」做可能世界精确查询，用「对话记忆」查看教学记忆。",
            self.problog.deterministic_fact_count(),
            self.problog.probabilistic_fact_count(),
            self.problog.rule_count()
        ));
        output
    }

    /// 把教过的条件/因果知识登记为会话 Horn 子句（含占位符的自动变量化），
    /// 重建归结算子库；返回实际新增的子句（供修订回执精确记录）。
    fn register_lesson_clauses(
        &mut self,
        lessons: &[chinese_reasoning_nlu::ChineseLesson],
    ) -> Vec<Clause> {
        if !self.config.enable_resolution {
            return Vec::new();
        }
        let mut added: Vec<Clause> = Vec::new();
        let mut new_predicates: Vec<String> = Vec::new();
        for lesson in lessons {
            let Some(clause) = chinese_reasoning_nlu::lesson_to_clause(lesson) else {
                continue;
            };
            let duplicate = self
                .session_clauses
                .iter()
                .any(|existing| existing.head == clause.head && existing.body == clause.body);
            if duplicate {
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
            self.session_clauses.push(clause.clone());
            added.push(clause);
        }
        if !added.is_empty() {
            // 新谓词登记为关系：命题解析/意图规划才能识别规则里的谓词（如"老师"）
            for predicate in &new_predicates {
                self.aligner.register_term(predicate, TermKind::Relation);
            }
            if !new_predicates.is_empty() {
                self.rebuild_knowledge_caches();
            }
            self.rebuild_resolver();
        }
        added
    }

    fn render_clause_zh(&self, clause: &Clause) -> String {
        let head = chinese_reasoning_nlu::render_atom_zh(&clause.head);
        let body = clause
            .body
            .iter()
            .map(chinese_reasoning_nlu::render_atom_zh)
            .collect::<Vec<_>>()
            .join(" 且 ");
        format!("只要「{}」，就「{}」", body, head)
    }

    /// 中文通用规则学习：把「规则：如果X的师傅是Y，那么X的老师是Y」直接解析成
    /// Horn 子句（X/Y/Z、甲乙丙、A/B/C 等占位符自动成为逻辑变量），进入会话后向链规则库
    pub fn learn_rule_report(&mut self, sentence_raw: &str) -> String {
        if !self.config.enable_resolution {
            return "归结推理未启用（config.toml [reasoning] enable_resolution = false）。"
                .to_string();
        }
        let sentence = sentence_raw.trim();
        if sentence.is_empty() {
            return "请给出中文规则，例如：规则：如果X的师傅是Y，那么X的老师是Y".to_string();
        }
        let outcome = chinese_reasoning_nlu::parse_lessons(sentence);
        let rules: Vec<(&chinese_reasoning_nlu::ChineseLesson, Clause)> = outcome
            .lessons
            .iter()
            .filter_map(|lesson| {
                chinese_reasoning_nlu::lesson_to_clause(lesson).map(|clause| (lesson, clause))
            })
            .collect();
        if rules.is_empty() {
            let mut output = String::from("没能把这句话解析成规则。支持句式：\n");
            output.push_str(
                "  • 如果S的P是O，那么S的Q是O（S/O 用 X/Y/Z、甲乙丙、A/B/C 等占位符表示变量）\n",
            );
            output.push_str("  • 因为S的P是O，所以S的Q是O\n");
            for error in &outcome.errors {
                output.push_str(&format!("  失败原因：{}\n", error));
            }
            return output.trim_end().to_string();
        }
        // 写入前守门：规则可能推出与现有事实冲突的结论（如功能谓词第二取值），
        // 冲突时只问一个最小问题，绝不先教后报。
        let mut preview = ChangeSet::new(format!("学习规则：{}", sentence));
        for (_, clause) in &rules {
            preview.push(ChangeOp::AddRule(clause.clone()));
        }
        let plan = self.revision_guard().plan(preview);
        if plan.is_blocked() {
            return self.render_guard_block(&plan);
        }
        let impact = Self::render_guard_impact(&plan);
        let mut output =
            String::from("已学习中文规则（纯符号解析 → Horn 子句，直接进入后向链）：\n");
        for (index, (lesson, clause)) in rules.iter().enumerate() {
            output.push_str(&format!(
                "  {}. 原始：{}\n     逻辑形式：{} :- {}\n     规则性质：{}；置信度 {:.2}\n",
                index + 1,
                lesson.original,
                clause.head.display(),
                clause
                    .body
                    .iter()
                    .map(|atom| atom.display())
                    .collect::<Vec<_>>()
                    .join(", "),
                if chinese_reasoning_nlu::lesson_has_variables(lesson) {
                    "通用规则（含逻辑变量，可套用到任意实体）"
                } else {
                    "具体规则（仅对提到的实体生效）"
                },
                clause.confidence
            ));
        }
        for error in &outcome.errors {
            output.push_str(&format!("  未学习：{}\n", error));
        }
        let added = self.register_lesson_clauses(&outcome.lessons);
        if added.is_empty() {
            output.push_str("  （以上规则已存在，未重复添加）\n");
        } else {
            output.push_str(&format!("  新增 {} 条会话规则。", added.len()));
            let ops = added.into_iter().map(ChangeOp::AddRule).collect::<Vec<_>>();
            self.record_session_revision(
                format!("学习规则：{}", sentence),
                ops,
                Vec::new(),
                Vec::new(),
            );
            if let Some(receipt) = self.session_receipts.last() {
                output.push_str(&format!("  修订回执：{}。", receipt.id));
            }
        }
        if let Some(impact) = impact {
            output.push_str(&format!("  {}", impact));
        }
        output.push_str(&format!(
            "\n会话规则库现有 {} 条子句、{} 条事实；可用「判断/为什么/推理链/推理发现」验证后向链。",
            self.resolver.clause_count(),
            self.resolver.fact_count()
        ));
        output
    }

    /// 记住一条中文事实：解析为确定事实并加入会话归结事实集（不写知识图谱）
    /// 解析「A 不是 B 的 P」「A 不认识 B」这类否定陈述为被阻断的正向原子。
    fn parse_negative_atom(sentence: &str) -> Option<Atom> {
        let sentence = sentence.trim();
        let positive = if let Some(position) = sentence.find("不是") {
            let mut text = String::with_capacity(sentence.len());
            text.push_str(&sentence[..position]);
            text.push('是');
            text.push_str(&sentence[position + "不是".len()..]);
            text
        } else {
            let (position, _) = sentence
                .char_indices()
                .find(|(index, character)| *index > 0 && *character == '不')?;
            let after = &sentence[position + '不'.len_utf8()..];
            if ![
                "认识",
                "喜欢",
                "讨厌",
                "属于",
                "位于",
                "出生于",
                "隶属于",
                "工作于",
            ]
            .iter()
            .any(|verb| after.starts_with(verb))
            {
                return None;
            }
            let mut text = String::with_capacity(sentence.len());
            text.push_str(&sentence[..position]);
            text.push_str(after);
            text
        };
        let atom = chinese_reasoning_nlu::parse_proposition(&positive)?;
        if atom.terms.len() != 2 || !atom.is_ground() {
            return None;
        }
        Some(atom)
    }

    fn assert_negative_atom(&mut self, atom: Atom) -> String {
        self.register_atom_vocabulary(&atom);
        let rendered = self.render_atom_nl(&atom);
        let removed_positive = self.retract_atom(&atom);
        if removed_positive {
            self.note_slot_retract(&atom);
        }
        let added = if self.session_negative_facts.contains(&atom) {
            false
        } else {
            self.session_negative_facts.push(atom);
            self.invalidate_graph_session();
            true
        };
        let mut output = if added {
            format!(
                "已记住否定事实：{}。后续多跳路径会把它作为阻断边（未写知识图谱）。",
                rendered
            )
        } else {
            format!("否定事实「{}」已经记住。", rendered)
        };
        if removed_positive {
            output.push_str("已同时撤销同名会话肯定事实。");
        }
        output
    }

    /// 记住一条否定陈述：只作为会话级阻断边，不写知识图谱。
    pub fn assert_negative_nl_fact(&mut self, sentence_raw: &str) -> String {
        let sentence = sentence_raw.trim();
        let Some(atom) = Self::parse_negative_atom(sentence) else {
            return "没能把这句话解析为可阻断的否定事实；例如：李四不是王五的师傅".to_string();
        };
        self.assert_negative_atom(atom)
    }

    /// 槽身份：把实体/谓词归一成读模型使用的规范词面，保证「写入的槽」
    /// 与 `node_hits` 查询时看到的槽是同一个键。
    fn slot_identity(&self, subject: &str, predicate: &str) -> (String, String) {
        let subject = self.aligner.canonical_form(subject.trim());
        let predicate = self.aligner.canonical_form(predicate.trim());
        let predicate = self.canonical_attribute_name(&predicate).unwrap_or(predicate);
        (subject, predicate)
    }

    /// 同一槽上来自知识图谱的只读基值（不把会话事实算进去）。
    fn slot_base_values(&self, subject: &str, predicate: &str) -> Vec<String> {
        let (subject, predicate) = self.slot_identity(subject, predicate);
        let mut values: BTreeSet<String> = BTreeSet::new();
        for triple in self.predictor.triples() {
            if self.aligner.canonical_form(&triple.subject) == subject
                && self.aligner.canonical_form(&triple.predicate) == predicate
            {
                values.insert(triple.object.clone());
            }
        }
        if let Some(stored) = self
            .attribute_index
            .get(&(subject.clone(), predicate.clone()))
        {
            values.extend(stored.iter().cloned());
        }
        for (entity, attribute, value) in &self.attributes {
            if self.aligner.canonical_form(entity) == subject
                && self.aligner.canonical_form(attribute) == predicate
            {
                values.insert(value.clone());
            }
        }
        values.into_iter().collect()
    }

    /// 在槽账本上应用一条断言，返回状态转移；不直接改会话事实。
    fn apply_session_slot(
        &mut self,
        subject: &str,
        predicate: &str,
        object: &str,
        strength: AssertionStrength,
        source: &str,
    ) -> Result<SlotTransition, String> {
        let (canonical_subject, canonical_predicate) = self.slot_identity(subject, predicate);
        let base = self.slot_base_values(&canonical_subject, &canonical_predicate);
        let assertion = SlotAssertion::new(
            &canonical_subject,
            &canonical_predicate,
            object,
            strength,
            source,
        );
        self.slots.apply(assertion, &base)
    }

    /// 把槽账本投影回会话事实：删掉该槽上不再是当前赢家的旧原子。
    ///
    /// 这是「`session_facts` 是槽账本投影」的落地点；调用方随后只需把新的
    /// 赢家原子加入会话事实。
    fn sync_slot_projection(&mut self, subject: &str, predicate: &str) {
        let (canonical_subject, canonical_predicate) = self.slot_identity(subject, predicate);
        let accepted = self
            .slots
            .accepted(&canonical_subject, &canonical_predicate);
        let stale: Vec<Atom> = self
            .session_facts
            .iter()
            .filter(|atom| {
                if !atom.is_ground() {
                    return false;
                }
                let Some((atom_subject, atom_predicate, atom_object)) =
                    Self::session_atom_triple(atom)
                else {
                    return false;
                };
                self.aligner.canonical_form(atom_subject) == canonical_subject
                    && self.aligner.canonical_form(atom_predicate) == canonical_predicate
                    && !accepted.iter().any(|value| value == atom_object)
            })
            .cloned()
            .collect();
        for atom in stale {
            self.retract_atom(&atom);
        }
    }

    /// 会话事实被撤回后同步槽账本（不重写事实）。
    fn note_slot_retract(&mut self, atom: &Atom) {
        let Some((subject, predicate, object)) = Self::session_atom_triple(atom) else {
            return;
        };
        let (subject, predicate) = self.slot_identity(subject, predicate);
        self.slots.retract(&subject, &predicate, Some(object));
    }

    /// 会话事实被写入后同步槽账本，并把功能槽的此前赢家撤出事实集。
    fn note_slot_adopt(&mut self, atom: &Atom, strength: AssertionStrength, source: &str) {
        let Some((subject, predicate, object)) = Self::session_atom_triple(atom) else {
            return;
        };
        let Ok(transition) = self.apply_session_slot(subject, predicate, object, strength, source)
        else {
            return;
        };
        if transition.should_confirm() {
            return;
        }
        self.sync_slot_projection(subject, predicate);
    }

    /// 把一条结构化会话事实加入推理前提（不写知识图谱）。
    ///
    /// 供对话层的确认/更正回执桥接到推理层：用户确认的取值不仅覆盖直接
    /// 属性回答，还会与图谱事实同权参与多跳链、证明与会话快照恢复。
    /// 功能槽上的新取值会以「更正」语义替换旧赢家；返回 `true` 表示本次
    /// 确实新增了会话事实（重复调用幂等）。
    pub fn adopt_session_triple(&mut self, subject: &str, predicate: &str, object: &str) -> bool {
        let subject = subject.trim();
        let predicate = predicate.trim();
        let object = object.trim();
        if subject.is_empty() || predicate.is_empty() || object.is_empty() {
            return false;
        }
        if self.session_facts.iter().any(|atom| {
            atom.predicate.as_ref() == predicate
                && atom.terms.len() == 2
                && atom.terms[0].name() == subject
                && atom.terms[1].name() == object
        }) {
            return false;
        }
        let Ok(transition) = self.apply_session_slot(
            subject,
            predicate,
            object,
            AssertionStrength::Correction,
            "采纳",
        ) else {
            return false;
        };
        if transition.should_confirm() || !transition.after.iter().any(|value| value == object) {
            return false;
        }
        self.sync_slot_projection(subject, predicate);
        let atom = Atom::rel(predicate, subject, object);
        self.register_atom_vocabulary(&atom);
        self.adopt_atom(atom)
    }

    pub fn assert_nl_fact(&mut self, sentence_raw: &str) -> String {
        let sentence = sentence_raw.trim();
        let sentence = ["记住：", "记住:", "记录：", "记录:", "断言：", "断言:"]
            .iter()
            .find_map(|prefix| sentence.strip_prefix(prefix))
            .unwrap_or(sentence)
            .trim();
        if sentence.is_empty() {
            return "请给出要记住的事实，例如：记住：张三是李四的师傅".to_string();
        }
        if let Some(atom) = Self::parse_negative_atom(sentence) {
            return self.assert_negative_atom(atom);
        }
        let (strength, content) = AssertionStrength::classify(sentence);
        if strength == AssertionStrength::Hearsay {
            return self.note_hearsay_nl_fact(content);
        }
        if strength == AssertionStrength::Correction && content != sentence {
            return self.correct_nl_fact(content);
        }
        let outcome = chinese_reasoning_nlu::parse_lessons(content);
        // 写入前守门：先按当前会话世界试算，冲突时只问一个最小问题。
        let mut preflight = ChangeSet::new(format!("记住：{}", sentence));
        for lesson in &outcome.lessons {
            if lesson.kind == LessonKind::DeterministicFact && lesson.head.is_ground() {
                preflight.push(ChangeOp::AssertFact(lesson.head.clone()));
            }
        }
        let mut impact = None;
        if !preflight.is_empty() {
            let plan = self.revision_guard().plan(preflight);
            if plan.is_blocked() {
                return self.render_guard_block(&plan);
            }
            impact = Self::render_guard_impact(&plan);
        }
        let mut remembered: Vec<String> = Vec::new();
        let mut refused: Vec<String> = Vec::new();
        let mut superseded_negatives: Vec<String> = Vec::new();
        let mut revision_ops: Vec<ChangeOp> = Vec::new();
        for lesson in &outcome.lessons {
            if lesson.kind != LessonKind::DeterministicFact {
                let hint = match lesson.kind {
                    LessonKind::DefaultRule => "默认句请用「教学：」学习",
                    LessonKind::ConditionalRule | LessonKind::CausalRule => {
                        "条件句请用「规则：」或「教学：」学习"
                    }
                    LessonKind::ProbabilisticFact => "概率句请用「教学：」学习",
                    LessonKind::DeterministicFact => "",
                };
                refused.push(format!("「{}」不是确定事实（{}）", lesson.original, hint));
                continue;
            }
            if !lesson.head.is_ground() {
                refused.push(format!(
                    "「{}」含占位符，无法作为具体事实；请改用「规则：」",
                    lesson.original
                ));
                continue;
            }
            self.register_atom_vocabulary(&lesson.head);
            let rendered = self.render_atom_nl(&lesson.head);
            if let Some(position) = self
                .session_negative_facts
                .iter()
                .position(|negative| negative == &lesson.head)
            {
                self.session_negative_facts.remove(position);
                self.invalidate_graph_session();
                superseded_negatives.push(rendered.clone());
            }
            // 槽级门控：功能槽上的直陈遇到不同取值时只问一句、零写入；
            // 显式更正（「改口：」）才允许替换赢家。非二元原子回退到原子追加。
            let triple =
                Self::session_atom_triple(&lesson.head).map(|(subject, predicate, object)| {
                    (
                        subject.to_string(),
                        predicate.to_string(),
                        object.to_string(),
                    )
                });
            let has_slot_triple = triple.is_some();
            if let Some((subject, predicate, object)) = triple {
                match self.apply_session_slot(
                    &subject,
                    &predicate,
                    &object,
                    AssertionStrength::Direct,
                    "记住",
                ) {
                    Ok(transition) if transition.should_confirm() => {
                        refused.push(transition.render_zh());
                        continue;
                    }
                    Ok(_) => {
                        self.sync_slot_projection(&subject, &predicate);
                    }
                    Err(error) => {
                        refused.push(format!("「{}」未写入：{}", rendered, error));
                        continue;
                    }
                }
            }
            if self.adopt_atom(lesson.head.clone()) {
                revision_ops.push(ChangeOp::AssertFact(lesson.head.clone()));
                remembered.push(rendered);
            } else if has_slot_triple {
                refused.push(format!("「{}」已经记住（本次为幂等写入）", rendered));
            } else {
                refused.push(format!("「{}」已经记住", rendered));
            }
        }
        if !revision_ops.is_empty() {
            self.record_session_revision(
                format!("记住：{}", sentence),
                revision_ops,
                Vec::new(),
                Vec::new(),
            );
        }
        let mut output = String::new();
        if remembered.is_empty() {
            output.push_str("没有新的会话事实被记住。\n");
        } else {
            output.push_str(&format!(
                "已记住 {} 条会话事实（不写知识图谱，随对话记忆持久化）：\n",
                remembered.len()
            ));
            for (index, fact) in remembered.iter().enumerate() {
                output.push_str(&format!("  {}. {}\n", index + 1, fact));
            }
        }
        for reason in &refused {
            output.push_str(&format!("  跳过：{}\n", reason));
        }
        if !superseded_negatives.is_empty() {
            output.push_str("已撤销同名会话否定事实：");
            output.push_str(&superseded_negatives.join("；"));
            output.push_str("。\n");
        }
        if let Some(impact) = impact {
            output.push_str(&impact);
        }
        for error in &outcome.errors {
            output.push_str(&format!("  未解析：{}\n", error));
        }
        if let Some(receipt) = self.session_receipts.last() {
            if receipt.label == format!("记住：{}", sentence) {
                output.push_str(&format!(
                    "修订回执：{}（可用「撤销修订」撤回本次写入）。\n",
                    receipt.id
                ));
            }
        }
        output.push_str(&format!(
            "会话事实共 {} 条；后续「判断/为什么/推理链」会使用它们。",
            self.session_facts.len()
        ));
        output
    }

    /// 忘记一条会话事实：从会话归结事实集撤回（不触碰知识图谱）
    pub fn retract_nl_fact(&mut self, sentence_raw: &str) -> String {
        let sentence = sentence_raw.trim();
        let sentence = ["忘记：", "忘记:", "遗忘：", "遗忘:", "撤回：", "撤回:"]
            .iter()
            .find_map(|prefix| sentence.strip_prefix(prefix))
            .unwrap_or(sentence)
            .trim();
        if sentence.is_empty() {
            return "请给出要忘记的事实，例如：忘记：张三是李四的师傅".to_string();
        }
        if let Some(atom) = Self::parse_negative_atom(sentence) {
            let rendered = self.render_atom_nl(&atom);
            if let Some(position) = self
                .session_negative_facts
                .iter()
                .position(|existing| existing == &atom)
            {
                self.session_negative_facts.remove(position);
                self.invalidate_graph_session();
                return format!("已撤销否定事实：{}。", rendered);
            }
            return format!("否定事实中没有「{}」。", rendered);
        }
        let outcome = chinese_reasoning_nlu::parse_lessons(sentence);
        // 写入前守门：撤回后若规则仍能推出同一事实，先问清是撤事实还是停规则。
        let mut preflight = ChangeSet::new(format!("忘记：{}", sentence));
        for lesson in &outcome.lessons {
            if lesson.kind == LessonKind::DeterministicFact
                && lesson.head.is_ground()
                && self.session_facts.contains(&lesson.head)
            {
                preflight.push(ChangeOp::RetractFact(lesson.head.clone()));
            }
        }
        let mut impact = None;
        if !preflight.is_empty() {
            let plan = self.revision_guard().plan(preflight);
            if plan.is_blocked() {
                return self.render_guard_block(&plan);
            }
            impact = Self::render_guard_impact(&plan);
        }
        let mut removed: Vec<String> = Vec::new();
        let mut missing: Vec<String> = Vec::new();
        let mut revision_ops: Vec<ChangeOp> = Vec::new();
        for lesson in &outcome.lessons {
            if lesson.kind != LessonKind::DeterministicFact || !lesson.head.is_ground() {
                continue;
            }
            let rendered = self.render_atom_nl(&lesson.head);
            let removed_negative = self
                .session_negative_facts
                .iter()
                .position(|negative| negative == &lesson.head)
                .map(|position| self.session_negative_facts.remove(position))
                .is_some();
            if removed_negative {
                self.invalidate_graph_session();
            }
            if self.retract_atom(&lesson.head) {
                self.note_slot_retract(&lesson.head);
                revision_ops.push(ChangeOp::RetractFact(lesson.head.clone()));
                removed.push(rendered);
            } else if removed_negative {
                removed.push(rendered);
            } else {
                missing.push(rendered);
            }
        }
        if !revision_ops.is_empty() {
            self.record_session_revision(
                format!("忘记：{}", sentence),
                revision_ops,
                Vec::new(),
                Vec::new(),
            );
        }
        let mut output = String::new();
        if removed.is_empty() && missing.is_empty() {
            output.push_str(
                "没能把这句话解析为可撤回的事实（会话事实不会来自知识图谱，只撤回会话级记忆）。\n",
            );
        }
        for fact in &removed {
            output.push_str(&format!("已忘记会话事实：{}\n", fact));
        }
        for fact in &missing {
            output.push_str(&format!(
                "会话事实中没有「{}」（知识图谱事实不在此处撤回）。\n",
                fact
            ));
        }
        if let Some(impact) = impact {
            output.push_str(&impact);
        }
        for error in &outcome.errors {
            output.push_str(&format!("未解析：{}\n", error));
        }
        if let Some(receipt) = self.session_receipts.last() {
            if receipt.label == format!("忘记：{}", sentence) {
                output.push_str(&format!(
                    "修订回执：{}（可用「撤销修订」恢复本次撤回）。\n",
                    receipt.id
                ));
            }
        }
        output.push_str(&format!("会话事实剩余 {} 条。", self.session_facts.len()));
        output
    }

    /// 会话规则库报告：教过的 Horn 规则 + 记住的会话事实
    pub fn session_rules_report(&self) -> String {
        let mut output = format!(
            "会话后向链规则库（纯符号 Horn 子句，{} 条；事实 {} 条，其中教过的规则 {} 条）：\n",
            self.resolver.clause_count(),
            self.resolver.fact_count(),
            self.session_clauses.len()
        );
        if self.session_clauses.is_empty() {
            output.push_str(
                "  （还没有学习过中文规则；可输入「规则：如果X的师傅是Y，那么X的老师是Y」）\n",
            );
        }
        for (index, clause) in self.session_clauses.iter().enumerate() {
            output.push_str(&format!(
                "  {}. {}  〔{} :- {}；来源：{}；置信度 {:.2}〕\n",
                index + 1,
                self.render_clause_zh(clause),
                clause.head.display(),
                clause
                    .body
                    .iter()
                    .map(|atom| atom.display())
                    .collect::<Vec<_>>()
                    .join(", "),
                clause.source,
                clause.confidence
            ));
        }
        if self.session_facts.is_empty() {
            output.push_str("  会话事实：（无；可输入「记住：张三是李四的师傅」）");
        } else {
            output.push_str("  会话事实（未写知识图谱）：");
            for (index, atom) in self.session_facts.iter().enumerate() {
                if index > 0 {
                    output.push('；');
                }
                output.push_str(&self.render_atom_nl(atom));
            }
        }
        if !self.session_negative_facts.is_empty() {
            output.push_str("\n  会话否定事实（多跳路径阻断边）：");
            for (index, atom) in self.session_negative_facts.iter().enumerate() {
                if index > 0 {
                    output.push('；');
                }
                output.push_str(&self.render_atom_nl(atom));
            }
        }
        output
    }

    /// 会话规则 → 图账本条目（规则即数据：稳定 id、规范逻辑形式、来源、置信度）。
    pub fn rule_ledger_entries(&self) -> Vec<graph_ledger::RuleLedgerEntry> {
        self.session_clauses
            .iter()
            .map(|clause| {
                let head = ledger_atom(&clause.head);
                let body: Vec<graph_ledger::LedgerAtom> =
                    clause.body.iter().map(ledger_atom).collect();
                graph_ledger::RuleLedgerEntry {
                    id: graph_ledger::rule_entry_id(&head, &body),
                    head,
                    body,
                    confidence: clause.confidence,
                    source: clause.source.clone(),
                }
            })
            .collect()
    }

    /// 当前会话规则集指纹（派生账本过期判定的依据之一；与插入顺序无关）。
    pub fn rule_fingerprint(&self) -> String {
        graph_ledger::rule_fingerprint(&self.rule_ledger_entries())
    }

    /// 从规则账本恢复会话规则（启动时合并；重复规则不重复添加）。返回新增条数。
    ///
    /// 与 `apply_dialogue_snapshot` 的规则恢复同语义：新谓词登记为关系、
    /// 词表缓存与归结引擎整体重建，保证恢复的规则立即可用于后向链。
    pub fn restore_rules_from_ledger(
        &mut self,
        entries: &[graph_ledger::RuleLedgerEntry],
    ) -> usize {
        if !self.config.enable_resolution {
            return 0;
        }
        let mut added = 0usize;
        let mut new_predicates: Vec<String> = Vec::new();
        for entry in entries {
            let Some(head) = atom_from_ledger(&entry.head) else {
                continue;
            };
            let mut body = Vec::with_capacity(entry.body.len());
            let mut valid = true;
            for atom in &entry.body {
                match atom_from_ledger(atom) {
                    Some(restored) => body.push(restored),
                    None => {
                        valid = false;
                        break;
                    }
                }
            }
            if !valid {
                continue;
            }
            if self
                .session_clauses
                .iter()
                .any(|existing| existing.head == head && existing.body == body)
            {
                continue;
            }
            for atom in std::iter::once(&head).chain(body.iter()) {
                if !self.aligner.is_registered(&atom.predicate)
                    && !new_predicates
                        .iter()
                        .any(|p| p.as_str() == atom.predicate.as_ref())
                {
                    new_predicates.push(atom.predicate.to_string());
                }
            }
            self.session_clauses
                .push(Clause::new(head, body, entry.confidence, &entry.source));
            added += 1;
        }
        if added > 0 {
            for predicate in &new_predicates {
                self.aligner.register_term(predicate, TermKind::Relation);
            }
            if !new_predicates.is_empty() {
                self.rebuild_knowledge_caches();
            }
            self.rebuild_resolver();
        }
        added
    }
}

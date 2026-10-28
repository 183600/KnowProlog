{
                // 待确认的模式联想：先结算「是/不是」短答，再把原问句重放一次。
                // 未识别为短答时静默丢弃待确认状态，按新输入继续处理。
                let mut case_replay: Option<String> = None;
                if let Some((origin, suggestion)) = pending_case.take() {
                    let mut handled = true;
                    match pattern_ir::classify_case_reply(input) {
                        pattern_ir::CaseReply::Accept => {
                            match pattern_library.accept_case(&suggestion) {
                                Ok(entry) => {
                                    pattern_episodes.confirm_case(&suggestion);
                                    println!(
                                        "已确认案例：{}（来源 {}，路由 {}）。",
                                        entry.name, entry.source, entry.query_type
                                    );
                                    match oxigraph_manager
                                        .sync_pattern_ledger(pattern_library.entries())
                                        .await
                                    {
                                        Ok(count) => println!("模式落图：{} 条已同步。", count),
                                        Err(error) => {
                                            eprintln!("模式落图失败：{}", error);
                                        }
                                    }
                                    match oxigraph_manager
                                        .sync_pattern_episodes(pattern_episodes.entries())
                                        .await
                                    {
                                        Ok(count) => {
                                            println!("经验落图：{} 条已同步。", count)
                                        }
                                        Err(error) => {
                                            eprintln!("经验落图失败：{}", error);
                                        }
                                    }
                                    println!("正在用确认后的模式重新理解原问题…");
                                    case_replay = Some(origin);
                                    handled = false;
                                }
                                Err(error) => println!("确认联想失败：{}", error),
                            }
                        }
                        pattern_ir::CaseReply::Reject => {
                            if pattern_episodes.reject_case(&suggestion) {
                                println!("好的，已记下这次否定；该骨架不会再被主动推荐。");
                            } else {
                                println!("好的，已关闭这次模式联想。");
                            }
                            if let Err(error) = oxigraph_manager
                                .sync_pattern_episodes(pattern_episodes.entries())
                                .await
                            {
                                eprintln!("经验落图失败：{}", error);
                            }
                        }
                        pattern_ir::CaseReply::Unclear => {
                            println!("好的，先不确认；这次联想已关闭。");
                        }
                        pattern_ir::CaseReply::NewInput => {
                            handled = false;
                        }
                    }
                    if handled {
                        println!();
                        continue;
                    }
                }
                let input = case_replay.as_deref().unwrap_or(input);

                // 语境指称消歧（CRD）：先结算上一轮反问，再对本轮做绑定/改写/追问。
                crd_ledger.begin_turn();
                let mut crd_input: Option<String> = None;
                if let Some(pending) = crd_pending.take() {
                    match parse_clarify_act(input, &pending.question) {
                        Some(ClarifyAct::Choose { entity, .. })
                        | Some(ClarifyAct::Restate { entity }) => {
                            crd_ledger.bind(
                                &pending.key,
                                &entity,
                                BindingScope::Topic,
                                &format!("澄清回答：{}", input),
                            );
                            println!("（语境绑定：{} ⇒ {}）", pending.mention, entity);
                            crd_input = Some(rewrite_at(
                                &pending.original_input,
                                pending.start,
                                pending.mention.chars().count(),
                                &entity,
                            ));
                            let _ = crd_ledger.save_to_path(&reference_binding_path);
                        }
                        Some(ClarifyAct::Correct { to, .. }) => {
                            crd_ledger.correct(&pending.key, &to, &format!("澄清纠正：{}", input));
                            let _ = symbolic_reasoner.learn_entity_alignment(&pending.mention, &to);
                            println!("（语境纠错：{} ⇒ {}）", pending.mention, to);
                            crd_input = Some(rewrite_at(
                                &pending.original_input,
                                pending.start,
                                pending.mention.chars().count(),
                                &to,
                            ));
                            let _ = crd_ledger.save_to_path(&reference_binding_path);
                        }
                        Some(ClarifyAct::Unknown) | Some(ClarifyAct::Decline) => {
                            println!("（已跳过这次澄清，仍按原句理解。）");
                        }
                        None => {
                            // 不是澄清回答：丢弃待定问题，按新输入继续。
                        }
                    }
                }
                // 二次确认的绑定 → 别名提名（类型闸门 + 人工复核，不自动改库）
                if crd_input.is_some() {
                    let nominations = crd_ledger.take_alias_nominations();
                    if !nominations.is_empty() {
                        let candidates: Vec<know_prolog::entity_registry::AliasCandidate> =
                            nominations
                                .iter()
                                .map(|nomination| know_prolog::entity_registry::AliasCandidate {
                                    surface: nomination.surface.clone(),
                                    canonical: nomination.canonical.clone(),
                                    pattern: "语境指称绑定".to_string(),
                                    evidence: nomination.evidence.clone(),
                                })
                                .collect();
                        let outcomes = oxigraph_manager.nominate_dialogue_aliases(&candidates);
                        for outcome in outcomes {
                            match outcome.kind {
                                know_prolog::entity_registry::NominationKind::Confirmed => {
                                    println!(
                                        "（语境绑定已获二次确认：「{}」≡「{}」，已登记为别名。）",
                                        outcome.surface, outcome.canonical
                                    );
                                    let _ = symbolic_reasoner.register_entity_alias(
                                        &outcome.surface,
                                        &outcome.canonical,
                                    );
                                }
                                know_prolog::entity_registry::NominationKind::Pending => {
                                    println!(
                                        "（语境绑定提名待确认：「{}」≡「{}」，用「身份台账」复核。）",
                                        outcome.surface, outcome.canonical
                                    );
                                }
                                know_prolog::entity_registry::NominationKind::TypeConflict => {
                                    println!(
                                        "（语境绑定提名类型冲突：「{}」≡「{}」，需人工裁决。）",
                                        outcome.surface, outcome.canonical
                                    );
                                }
                                _ => {}
                            }
                        }
                        let _ = oxigraph_manager.save_entity_registry(&entity_registry_path);
                    }
                }
                if crd_input.is_none()
                    && (contains_question_marker(input)
                        || input.contains('？')
                        || input.contains('?'))
                {
                    let crd_hints = symbolic_reasoner.reference_hints(input);
                    let outcome = crd_resolver.resolve_text_with_memory(
                        input,
                        symbolic_reasoner.linker(),
                        &symbolic_reasoner,
                        &crd_ledger,
                        &crd_hints,
                        dialogue_manager.focus_entity(),
                        Some(symbolic_reasoner.lexical_memory()),
                    );
                    // 只在确有竞争/绑定/追问/描述改写时展示证据，避免每问都刷屏
                    let show_notes = outcome.ask.is_some()
                        || outcome.rewrite.is_some()
                        || outcome.resolutions.iter().any(|resolution| {
                            resolution.candidates.len() > 1
                                || resolution.verdict
                                    == know_prolog::contextual_reference::Verdict::Bound
                        });
                    if show_notes {
                        for note in &outcome.notes {
                            println!("（{}）", note);
                        }
                    }
                    if let Some(ask) = outcome.ask {
                        let resolution = outcome.resolutions.iter().find(|resolution| {
                            resolution.sketch.start == ask.start && resolution.key == ask.key
                        });
                        let report = reference_probe::probe_clarification(
                            &ask,
                            resolution,
                            symbolic_reasoner.revision(),
                            reference_probe::MAX_PROBE_CANDIDATES,
                            |entity, rewritten| {
                                probe_reference_answer(&symbolic_reasoner, entity, rewritten)
                            },
                        );
                        grounding_ledger.push(report.certificate.clone());
                        if let Err(error) =
                            grounding_ledger.save_to_path(&grounding_certificate_path)
                        {
                            eprintln!("接地证书落盘失败（不影响本轮回答）：{}", error);
                        }
                        for note in &report.notes {
                            println!("（{}）", note);
                        }
                        match report.decision {
                            ProbeDecision::Bind { entity, .. } => {
                                let rewrite = rewrite_at(
                                    &ask.original_input,
                                    ask.start,
                                    ask.mention.chars().count(),
                                    &entity,
                                );
                                if rewrite != input {
                                    println!("（语境改写：{}）", rewrite);
                                }
                                crd_input = Some(rewrite);
                            }
                            _ => {
                                println!("{}", report.question.render());
                                crd_pending = Some(ClarifyRequest {
                                    question: report.question,
                                    ..ask
                                });
                                println!();
                                continue;
                            }
                        }
                    }
                    if let Some(rewrite) = outcome.rewrite {
                        if rewrite != input {
                            println!("（语境改写：{}）", rewrite);
                        }
                        crd_input = Some(rewrite);
                    }
                }
                let crd_base_input = crd_input.unwrap_or_else(|| input.to_string());

                // 多轮对话上下文：指代消解与省略补全（话语帧合一，符号方式，无神经网络）
                let contextual_input =
                    match symbolic_reasoner.contextualize_with_receipt(&crd_base_input) {
                        Some(rewrite) => {
                            for note in &rewrite.notes {
                                println!("（{}）", note);
                            }
                            println!("（结合对话上下文理解为：{}）", rewrite.text);
                            rewrite.text
                        }
                        None => {
                            // 无源可填的实体洞：不落进「无法理解」，转为一次显式绑定请求
                            // （澄清 = 请求一次绑定；常规通道必然失败时才开启）。
                            if let Some(prompt) =
                                symbolic_reasoner.start_frame_clarification(&crd_base_input)
                            {
                                println!("{}", prompt);
                                println!();
                                continue;
                            }
                            crd_base_input
                        }
                    };
                let input = contextual_input.as_str();

                for note in symbolic_reasoner.entity_link_notes(input) {
                    println!("（{}）", note);
                }

                // 会话覆盖层优先：只回答本会话已确认的关系，绝不冒充图谱事实。
                if let Some(answer) = acquisition
                    .answer_from_overlay(input, &oxigraph_manager.predicate_registry_snapshot())
                {
                    println!("{}", answer.render_text());
                    log_chronicle_reply(&mut chronicle, &answer.render_text(), &[], &[]);
                    println!();
                    continue;
                }

                // 插件配方：只读既有知识源，模板必须全部槽位接地；取不到值则让位核心管线。
                if let Some(outcome) = plugin_outcome.as_ref() {
                    let focus = dialogue_manager.focus_entity().map(str::to_string);
                    if let Some(reply) = symbolic_reasoner.answer_plugin_recipe(
                        outcome,
                        &mut plugin_runtime,
                        focus.as_deref(),
                    ) {
                        println!("{}", reply.text);
                        if dialogue_enabled {
                            dialogue_manager.note_plugin_turn(
                                raw_input,
                                MoveKind::AskAttribute,
                                reply.entity.as_deref(),
                                reply.attribute.as_deref(),
                                &reply.text,
                                reply.plan.clone(),
                                reply.provenance.clone(),
                            );
                            if let Some(focus) = reply.focus.as_deref() {
                                dialogue_manager.note_plugin_focus(focus);
                            }
                        }
                        log_chronicle_reply(
                            &mut chronicle,
                            &reply.text,
                            &reply.plan,
                            &reply.provenance,
                        );
                        log_chronicle_trace(&mut chronicle, &dialogue_manager);
                        println!();
                        continue;
                    }
                }

                // 本体化对话管理器：实体已知的知识问句给出"结论 + 出处 + 追问建议"式组合回答
                if dialogue_enabled {
                    refresh_dialogue_quality(
                        &mut dialogue_manager,
                        &oxigraph_manager,
                        &language_config,
                        symbolic_reasoner.revision(),
                        &store_path,
                    )
                    .await;
                    if let Some(reply) = dialogue_manager.plan_reply(input, &symbolic_reasoner) {
                        let mut text = plugin_runtime.style_wrap(&reply.text);
                        if is_no_record_text(&text) {
                            if let Some(prompt) = symbolic_reasoner.open_pending_question(input) {
                                text.push('\n');
                                text.push_str(&prompt);
                            }
                        } else {
                            symbolic_reasoner.dismiss_pending_question(input);
                        }
                        println!("{}", text);
                        let provenance: Vec<String> = dialogue_manager
                            .turns()
                            .last()
                            .map(|turn| turn.provenance.clone())
                            .unwrap_or_default();
                        log_chronicle_reply(&mut chronicle, &text, &reply.plan, &provenance);
                        log_chronicle_trace(&mut chronicle, &dialogue_manager);
                        println!();
                        continue;
                    }
                }

                let tagged_words = preprocess(input, &plugin_nlp);
                let precomputed_query_entities =
                    extract_query_entities_from_tagged(&tagged_words, &language_config);
                let nlu_tagged_words = if language_config.nlp.enable_nlu.unwrap_or(false) {
                    Some(tagged_words.clone())
                } else {
                    None
                };
                let normalized_words = normalize_query_words(
                    tagged_words.clone(),
                    &synonym_graph,
                    &symbolic_reasoner,
                    &language_config,
                    &synonyms,
                );
                // 感知层覆盖：已确认词面（图谱实体/确认别名/显式准入）保留为
                // 单一 token；旧切分仍是回退路径，覆盖层只增加边界证据。
                let lexicon_tagged_words =
                    symbolic_reasoner.segment_with_lexicon(input, &plugin_nlp);
                let lexicon_normalized_words = if lexicon_tagged_words == tagged_words {
                    normalized_words.clone()
                } else {
                    normalize_query_words(
                        lexicon_tagged_words,
                        &synonym_graph,
                        &symbolic_reasoner,
                        &language_config,
                        &synonyms,
                    )
                };
                // 路由上下文：让带守卫的模式只在相应对话状态下参与匹配。
                let route_context = dialogue_pattern_context(&dialogue_manager, &symbolic_reasoner);

                // 使用NLU进行语义理解
                if language_config.nlp.enable_nlu.unwrap_or(false) {
                    let semantic_analysis = match nlu_tagged_words {
                        Some(tagged) => nlu_engine.analyze_tagged(input, tagged),
                        None => nlu_engine.analyze(input),
                    };

                    if !semantic_analysis.semantics.is_empty() {
                        let best_semantic = &semantic_analysis.semantics[0];
                        let confidence_threshold =
                            language_config.nlp.nlu_confidence_threshold.unwrap_or(0.6);

                        if best_semantic.score >= confidence_threshold {
                            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                            println!("NLU语义理解结果:");
                            println!("  领域: {}", best_semantic.domain);
                            println!("  意图: {}", best_semantic.intent);
                            println!("  置信度: {:.2}", best_semantic.score);
                            if !best_semantic.slots.is_empty() {
                                println!("  词槽:");
                                for (slot_name, slot_value) in &best_semantic.slots {
                                    println!("    {}: {}", slot_name, slot_value);
                                }
                            }
                            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

                            // 根据NLU结果处理查询
                            match best_semantic.domain.as_str() {
                                "天气" => {
                                    let city = best_semantic
                                        .slots
                                        .iter()
                                        .find(|(name, _)| name == "城市")
                                        .map(|(_, value)| value.clone())
                                        .unwrap_or_else(|| "未知城市".to_string());
                                    let date = best_semantic
                                        .slots
                                        .iter()
                                        .find(|(name, _)| name == "日期")
                                        .map(|(_, value)| value.clone())
                                        .unwrap_or_else(|| "今天".to_string());

                                    match best_semantic.intent.as_str() {
                                        "雨" | "天气查询" => {
                                            println!("正在查询{}{}的天气信息...", city, date);
                                            println!("（天气API未接入，尝试从本地知识库回答）");
                                        }
                                        _ => {
                                            println!(
                                                "暂不支持该天气查询类型: {}，尝试本地符号推理...",
                                                best_semantic.intent
                                            );
                                        }
                                    }
                                    // 天气 API 不可用时不能丢问题：回落到本地图谱/符号推理
                                    if let Some(answer) = try_symbolic_reasoning(
                                        &mut symbolic_reasoner,
                                        &mut contrast_arbiter,
                                        &oxigraph_manager,
                                        input,
                                        &precomputed_query_entities,
                                    )
                                    .await
                                    {
                                        println!("{}", answer);
                                        log_chronicle_reply(&mut chronicle, &answer, &[], &[]);
                                    } else {
                                        println!("本地知识库中没有相关天气记录。");
                                    }
                                }
                                "人物" => {
                                    let person = best_semantic
                                        .slots
                                        .iter()
                                        .find(|(name, _)| name == "人名")
                                        .map(|(_, value)| value.clone());

                                    if let Some(person_name) = person {
                                        match best_semantic.intent.as_str() {
                                            "身高查询" => {
                                                println!("正在查询{}的身高信息...", person_name);
                                                // 本地适配器返回了实体但没有身高/高度属性时，
                                                // 不能就此收尾：回落到本地图谱与符号推理
                                                let found_height = match local_semantic_adapter
                                                    .get_entity_knowledge(&person_name)
                                                    .await
                                                {
                                                    Ok(response) => {
                                                        println!(
                                                            "─────────────────────────────────"
                                                        );
                                                        println!("实体: {}", response.data.entity);
                                                        println!("描述: {}", response.data.desc);
                                                        let mut found = false;
                                                        for avp in &response.data.avp {
                                                            if avp.len() >= 2
                                                                && (avp[0].contains("身高")
                                                                    || avp[0].contains("高度"))
                                                            {
                                                                println!("身高: {}", avp[1]);
                                                                found = true;
                                                            }
                                                        }
                                                        println!(
                                                            "─────────────────────────────────"
                                                        );
                                                        found
                                                    }
                                                    Err(e) => {
                                                        println!("获取{}身高信息时出错: {}，回退到本地符号推理...", person_name, e);
                                                        false
                                                    }
                                                };
                                                if !found_height {
                                                    if let Some(answer) = try_symbolic_reasoning(
                                                        &mut symbolic_reasoner,
                                                        &mut contrast_arbiter,
                                                        &oxigraph_manager,
                                                        input,
                                                        &precomputed_query_entities,
                                                    )
                                                    .await
                                                    {
                                                        println!("{}", answer);
                                                        log_chronicle_reply(
                                                            &mut chronicle,
                                                            &answer,
                                                            &[],
                                                            &[],
                                                        );
                                                    } else {
                                                        println!(
                                                            "知识库中没有{}的身高记录。",
                                                            person_name
                                                        );
                                                    }
                                                }
                                            }
                                            "年龄查询" => {
                                                println!("正在查询{}的年龄信息...", person_name);
                                                if let Some(answer) = try_symbolic_reasoning(
                                                    &mut symbolic_reasoner,
                                                    &mut contrast_arbiter,
                                                    &oxigraph_manager,
                                                    input,
                                                    &precomputed_query_entities,
                                                )
                                                .await
                                                {
                                                    println!("{}", answer);
                                                    log_chronicle_reply(
                                                        &mut chronicle,
                                                        &answer,
                                                        &[],
                                                        &[],
                                                    );
                                                } else {
                                                    println!(
                                                        "知识库中没有{}的年龄记录。",
                                                        person_name
                                                    );
                                                }
                                            }
                                            _ => {
                                                println!("暂不支持该人物查询类型: {}，尝试本地符号推理...", best_semantic.intent);
                                                if let Some(answer) = try_symbolic_reasoning(
                                                    &mut symbolic_reasoner,
                                                    &mut contrast_arbiter,
                                                    &oxigraph_manager,
                                                    input,
                                                    &precomputed_query_entities,
                                                )
                                                .await
                                                {
                                                    println!("{}", answer);
                                                    log_chronicle_reply(
                                                        &mut chronicle,
                                                        &answer,
                                                        &[],
                                                        &[],
                                                    );
                                                }
                                            }
                                        }
                                    } else {
                                        println!("未能识别人名，请重新输入");
                                    }
                                }
                                "聊天" => {
                                    let chat_reply = match best_semantic.intent.as_str() {
                                        "问候" => {
                                            "你好！我是KnowProlog知识图谱系统，很高兴为您服务！"
                                        }
                                        "感谢" => "不客气！如果您有其他问题，随时可以问我。",
                                        _ => "很高兴和您聊天！",
                                    };
                                    println!("{}", chat_reply);
                                    log_chronicle_reply(&mut chronicle, chat_reply, &[], &[]);
                                }
                                "知识查询" => {
                                    // 继续使用查询逻辑
                                    if let Some((query_type, entities, pattern_note)) =
                                        route_query_with_lexicon(
                                            &pattern_library,
                                            input,
                                            &lexicon_normalized_words,
                                            &normalized_words,
                                            &mut pattern_episodes,
                                            &route_context,
                                            &language_config.reasoning.soft_route,
                                        )
                                    {
                                        // 执行查询流程
                                        if let Some(note) = &pattern_note {
                                            println!("命中模式: {}", note);
                                            record_forge_hit(&mut pattern_forge, note, &forge_path);
                                        }
                                        continue_original_query(
                                            &language_config,
                                            &oxigraph_manager,
                                            &mut prolog_engine,
                                            &response_formatter,
                                            &local_semantic_adapter,
                                            &external_knowledge_service,
                                            &mut symbolic_reasoner,
                                            &mut contrast_arbiter,
                                            &mut plugin_runtime,
                                            input,
                                            &query_type,
                                            &entities,
                                            &precomputed_query_entities,
                                        )
                                        .await;
                                    } else if let Some(answer) = try_symbolic_reasoning(
                                        &mut symbolic_reasoner,
                                        &mut contrast_arbiter,
                                        &oxigraph_manager,
                                        input,
                                        &precomputed_query_entities,
                                    )
                                    .await
                                    {
                                        println!("{}", answer);
                                        log_chronicle_reply(&mut chronicle, &answer, &[], &[]);
                                    } else if let Some((text, plan)) =
                                        plugin_policy_text(&mut plugin_runtime, input)
                                    {
                                        println!("{}", text);
                                        log_chronicle_reply(
                                            &mut chronicle,
                                            &text,
                                            &[plan.clone()],
                                            &[format!("插件策略 {}", plan)],
                                        );
                                    } else {
                                        let fallback = "无法理解您的查询，请重新描述您的问题。";
                                        println!("{}", fallback);
                                        log_chronicle_reply(&mut chronicle, fallback, &[], &[]);
                                    }
                                }
                                _ => {
                                    println!("暂不支持该领域: {}", best_semantic.domain);
                                    // 如果NLU不能处理，回退到查询流程
                                    if let Some((query_type, entities, pattern_note)) =
                                        route_query_with_lexicon(
                                            &pattern_library,
                                            input,
                                            &lexicon_normalized_words,
                                            &normalized_words,
                                            &mut pattern_episodes,
                                            &route_context,
                                            &language_config.reasoning.soft_route,
                                        )
                                    {
                                        if let Some(note) = &pattern_note {
                                            println!("命中模式: {}", note);
                                            record_forge_hit(&mut pattern_forge, note, &forge_path);
                                        }
                                        continue_original_query(
                                            &language_config,
                                            &oxigraph_manager,
                                            &mut prolog_engine,
                                            &response_formatter,
                                            &local_semantic_adapter,
                                            &external_knowledge_service,
                                            &mut symbolic_reasoner,
                                            &mut contrast_arbiter,
                                            &mut plugin_runtime,
                                            input,
                                            &query_type,
                                            &entities,
                                            &precomputed_query_entities,
                                        )
                                        .await;
                                    } else if let Some(answer) = try_symbolic_reasoning(
                                        &mut symbolic_reasoner,
                                        &mut contrast_arbiter,
                                        &oxigraph_manager,
                                        input,
                                        &precomputed_query_entities,
                                    )
                                    .await
                                    {
                                        println!("{}", answer);
                                        log_chronicle_reply(&mut chronicle, &answer, &[], &[]);
                                    } else if let Some((text, plan)) =
                                        plugin_policy_text(&mut plugin_runtime, input)
                                    {
                                        println!("{}", text);
                                        log_chronicle_reply(
                                            &mut chronicle,
                                            &text,
                                            &[plan.clone()],
                                            &[format!("插件策略 {}", plan)],
                                        );
                                    } else {
                                        println!("{}", language_config.messages.cannot_understand);
                                        log_chronicle_reply(
                                            &mut chronicle,
                                            &language_config.messages.cannot_understand,
                                            &[],
                                            &[],
                                        );
                                    }
                                }
                            }
                            continue; // 跳过后续查询逻辑
                        }
                    }

                    // 如果NLU置信度不够，使用传统模式处理
                    println!("NLU置信度不足，使用传统模式处理查询...");
                }

                // 检查是否是直接的实体知识查询（例如："刘德华的信息" 或 "告诉我关于刘德华"）
                if input.contains("的信息")
                    || input.contains("告诉我关于")
                    || input.contains("什么是")
                {
                    // 提取实体名
                    let entity_name = input
                        .replace("的信息", "")
                        .replace("告诉我关于", "")
                        .replace("什么是", "")
                        .trim()
                        .to_string();

                    if !entity_name.is_empty() {
                        println!("正在获取实体 '{}' 的知识...", entity_name);

                        match local_semantic_adapter
                            .get_entity_knowledge(&entity_name)
                            .await
                        {
                            Ok(response) => {
                                println!("─────────────────────────────────");
                                println!("实体: {}", response.data.entity);
                                println!("描述: {}", response.data.desc);
                                if !response.data.tag.is_empty() {
                                    println!("标签: {}", response.data.tag.join(", "));
                                }
                                println!("\n属性信息:");
                                for avp in &response.data.avp {
                                    if avp.len() >= 2 {
                                        println!("  {}: {}", avp[0], avp[1]);
                                    }
                                }
                                println!("─────────────────────────────────");
                                continue;
                            }
                            Err(e) => {
                                println!("获取实体知识时出错: {}，尝试本地查询...", e);
                            }
                        }
                    }
                }

                // 检查是否是实体属性查询（例如："刘德华的年龄" 或 "苹果的颜色"）
                let attribute_re = ENTITY_ATTRIBUTE_RE
                    .get_or_init(|| regex::Regex::new(r"(.+?)的(.+?)$").unwrap());
                if let Some(captures) = attribute_re.captures(input) {
                    if let (Some(entity_match), Some(attr_match)) =
                        (captures.get(1), captures.get(2))
                    {
                        let entity_name = entity_match.as_str().trim();
                        let attribute = attr_match.as_str().trim();

                        // 排除已经处理过的模式与符号推理问句（谁/吗/几/多少等）
                        if !input.contains("信息")
                            && !input.contains("什么")
                            && !input.contains("谁")
                            && !input.contains("吗")
                            && !input.contains("呢")
                            && !input.contains("几")
                            && !input.contains("多少")
                        {
                            println!("正在查询实体 '{}' 的属性 '{}'...", entity_name, attribute);

                            match local_semantic_adapter
                                .get_entity_attribute_value(entity_name, attribute)
                                .await
                            {
                                Ok(response) => {
                                    println!("─────────────────────────────────");
                                    println!("实体: {}", response.data.entity);
                                    println!("属性: {}", response.data.attribute);
                                    println!("值: {}", response.data.value.join(", "));
                                    println!("─────────────────────────────────");
                                    continue;
                                }
                                Err(e) => {
                                    println!("获取属性值时出错: {}，尝试本地查询...", e);
                                }
                            }
                        }
                    }
                }

                if let Some((query_type, entities, pattern_note)) = route_query_with_lexicon(
                    &pattern_library,
                    input,
                    &lexicon_normalized_words,
                    &normalized_words,
                    &mut pattern_episodes,
                    &route_context,
                    &language_config.reasoning.soft_route,
                ) {
                    println!("查询类型: {}, 实体: {:?}", query_type, entities);
                    if let Some(note) = &pattern_note {
                        println!("命中模式: {}", note);
                        record_forge_hit(&mut pattern_forge, note, &forge_path);
                    }

                    // 提取查询中的实体
                    let mut seen_entities = std::collections::HashSet::new();
                    let mut query_entities: Vec<String> =
                        Vec::with_capacity(precomputed_query_entities.len() + entities.len());
                    for entity in precomputed_query_entities.iter().chain(entities.iter()) {
                        if seen_entities.insert(entity.as_str()) {
                            query_entities.push(entity.clone());
                        }
                    }

                    // === 歧义解析 ===
                    if let Some(api_config) = &language_config.api {
                        if api_config.enable_entity_disambiguation {
                            let resolved_entities = resolve_query_entities_with_disambiguation(
                                input,
                                &query_entities,
                                api_config,
                                &local_semantic_adapter,
                            )
                            .await;

                            // 使用解析后的实体进行查询
                            let final_entities = if !resolved_entities.is_empty() {
                                resolved_entities
                            } else {
                                entities.clone()
                            };

                            // 仅加载相关知识子集
                            let max_hops = 2; // 可配置的跳跃数
                            match oxigraph_manager
                                .get_relevant_facts(&final_entities, max_hops)
                                .await
                            {
                                Ok(facts) => {
                                    println!("加载了 {} 个相关事实", facts.len());
                                    prolog_engine.load_relevant_facts(facts);

                                    // 先尝试直接从RDF数据获取答案
                                    if let Some(answer) = oxigraph_manager
                                        .get_direct_answer(&query_type, &final_entities)
                                        .await
                                    {
                                        println!("{}", answer);
                                        log_chronicle_reply(&mut chronicle, &answer, &[], &[]);
                                    } else if let Some(prolog_query) =
                                        PrologEngine::generate_prolog_query(
                                            &query_type,
                                            &final_entities,
                                        )
                                    {
                                        println!("Prolog查询: {}", prolog_query);

                                        let response = run_prolog_query(
                                            &mut prolog_engine,
                                            &mut symbolic_reasoner,
                                            &prolog_query,
                                            &query_type,
                                            &response_formatter,
                                        );
                                        println!("{}", response);
                                        log_chronicle_reply(&mut chronicle, &response, &[], &[]);
                                    } else {
                                        println!("无法生成有效的Prolog查询");
                                    }
                                }
                                Err(e) => {
                                    println!("{}: {}", language_config.messages.process_error, e);
                                }
                            }
                        } else {
                            // 歧义解析功能被禁用，直接查询
                            let max_hops = 2;
                            match oxigraph_manager
                                .get_relevant_facts(&query_entities, max_hops)
                                .await
                            {
                                Ok(facts) => {
                                    println!("加载了 {} 个相关事实", facts.len());
                                    prolog_engine.load_relevant_facts(facts);

                                    if let Some(answer) = oxigraph_manager
                                        .get_direct_answer(&query_type, &entities)
                                        .await
                                    {
                                        println!("{}", answer);
                                        log_chronicle_reply(&mut chronicle, &answer, &[], &[]);
                                    } else if let Some(prolog_query) =
                                        PrologEngine::generate_prolog_query(&query_type, &entities)
                                    {
                                        println!("Prolog查询: {}", prolog_query);

                                        let response = run_prolog_query(
                                            &mut prolog_engine,
                                            &mut symbolic_reasoner,
                                            &prolog_query,
                                            &query_type,
                                            &response_formatter,
                                        );
                                        println!("{}", response);
                                        log_chronicle_reply(&mut chronicle, &response, &[], &[]);
                                    } else {
                                        println!("无法生成有效的Prolog查询");
                                    }
                                }
                                Err(e) => {
                                    println!("{}: {}", language_config.messages.process_error, e);
                                }
                            }
                        }
                    } else {
                        // 没有API配置，直接查询
                        let max_hops = 2;
                        match oxigraph_manager
                            .get_relevant_facts(&query_entities, max_hops)
                            .await
                        {
                            Ok(facts) => {
                                println!("加载了 {} 个相关事实", facts.len());
                                prolog_engine.load_relevant_facts(facts);

                                if let Some(answer) = oxigraph_manager
                                    .get_direct_answer(&query_type, &entities)
                                    .await
                                {
                                    println!("{}", answer);
                                    log_chronicle_reply(&mut chronicle, &answer, &[], &[]);
                                } else if let Some(prolog_query) =
                                    PrologEngine::generate_prolog_query(&query_type, &entities)
                                {
                                    println!("Prolog查询: {}", prolog_query);

                                    let response = run_prolog_query(
                                        &mut prolog_engine,
                                        &mut symbolic_reasoner,
                                        &prolog_query,
                                        &query_type,
                                        &response_formatter,
                                    );
                                    println!("{}", response);
                                    log_chronicle_reply(&mut chronicle, &response, &[], &[]);
                                } else {
                                    println!("无法生成有效的Prolog查询");
                                }
                            }
                            Err(e) => {
                                println!("{}: {}", language_config.messages.process_error, e);
                            }
                        }
                    }
                } else if let Some(answer) = try_symbolic_reasoning(
                    &mut symbolic_reasoner,
                    &mut contrast_arbiter,
                    &oxigraph_manager,
                    input,
                    &precomputed_query_entities,
                )
                .await
                {
                    println!("{}", answer);
                    log_chronicle_reply(&mut chronicle, &answer, &[], &[]);
                } else {
                    // 先区分「实体未知」与「问法不支持」：库里完全没有提到过的实体，
                    // 坦白说没有记录，比笼统的「不理解」更接近人类对话
                    let mut missing_entities: Vec<String> = Vec::new();
                    for entity in &precomputed_query_entities {
                        if entity.chars().count() < 2 {
                            continue;
                        }
                        match oxigraph_manager
                            .get_relevant_facts(std::slice::from_ref(entity), 0)
                            .await
                        {
                            Ok(facts) if facts.is_empty() => missing_entities.push(entity.clone()),
                            _ => {}
                        }
                    }
                    if missing_entities.is_empty() {
                        println!("{}", language_config.messages.cannot_understand);
                    } else {
                        let names = missing_entities
                            .iter()
                            .map(|name| format!("「{}」", name))
                            .collect::<Vec<_>>()
                            .join("、");
                        println!(
                            "我的知识库里暂时没有{}的记录。如果你想教我，可以说「录入：{}…」。",
                            names,
                            missing_entities.first().map(String::as_str).unwrap_or("")
                        );
                    }

                    // 问法未命中但存在相似成功经验：给出可确认的候选骨架，
                    // 把「没听懂」变成一次可复用的上下文学习机会。
                    if let Some(suggestion) = pattern_episodes.suggest_case_with_context(
                        &normalized_words,
                        &route_context,
                        0.2,
                    ) {
                        println!(
                            "模式联想：{}（你是在问这个意思吗？答「是」确认，答「不是」否定；也可用「确认联想：{}」）",
                            suggestion.pattern.render(),
                            input
                        );
                        pending_case = Some((input.to_string(), suggestion));
                    }

                    // 插件声明的兜底策略先于外链/常识别源；只改措辞，不伪造事实。
                    if let Some((policy_text, policy_plan)) = plugin_runtime.policy_reply(raw_input)
                    {
                        let text = plugin_runtime.style_wrap(&policy_text);
                        println!("{}", text);
                        log_chronicle_reply(
                            &mut chronicle,
                            &text,
                            &[policy_plan.clone()],
                            &[format!("插件策略 {}", policy_plan)],
                        );
                    } else if let Some(direct_answer) = get_direct_answer(input) {
                        println!("根据常识知识库：{}", direct_answer);
                        log_chronicle_reply(
                            &mut chronicle,
                            &format!("根据常识知识库：{}", direct_answer),
                            &[],
                            &[],
                        );
                    } else {
                        // 尝试使用外链知识（名册优先，未命中才打网，回答带出处）
                        match external_knowledge_service
                            .answer_question_with_entities(
                                input,
                                &external_candidates(&[], &precomputed_query_entities),
                            )
                            .await
                        {
                            Ok(outcome) => {
                                if let Some(chronicle) = chronicle.as_mut() {
                                    match &outcome {
                                        AnswerOutcome::Answered(answer) => {
                                            let _ = conversation_theory::append_external(
                                                chronicle,
                                                &answer.entity,
                                                "回答",
                                                &answer.source_uri,
                                                kb_revision,
                                            );
                                        }
                                        AnswerOutcome::KnownAbsent {
                                            entity, source_uri, ..
                                        } => {
                                            let _ = conversation_theory::append_external(
                                                chronicle,
                                                entity,
                                                "确认未收录",
                                                source_uri,
                                                kb_revision,
                                            );
                                        }
                                        AnswerOutcome::Unknown => {}
                                    }
                                }
                                match external_outcome_text(&outcome) {
                                    Some(text) => {
                                        println!("{}", text);
                                        log_chronicle_reply(&mut chronicle, &text, &[], &[]);
                                    }
                                    None => {
                                        let fallback =
                                            "很抱歉，我无法理解您的问题。请尝试重新描述。";
                                        println!("{}", fallback);
                                        log_chronicle_reply(&mut chronicle, fallback, &[], &[]);
                                    }
                                }
                            }
                            Err(e) => {
                                println!("查询外部知识时出错：{}。请检查网络连接。", e);
                            }
                        }
                    }
                }
}

/// ask 的多行会话与交互/服务共用同一套对话前通道配置：
/// 恢复策略之外，还必须编译并注入配置态绑定帧，否则槽位澄清在脚本入口失效。
fn ask_dialogue(config: &SystemConfig, recovery_mode: RecoveryMode) -> DialogueManager {
    let mut dialogue = DialogueManager::new();
    dialogue.set_recovery_mode(recovery_mode);
    dialogue.set_frame_program(compile_frame_program(config));
    dialogue
}

async fn run_ask(
    config: &SystemConfig,
    oxigraph_manager: &OxigraphManager,
    store_path: &str,
    recovery_mode: RecoveryMode,
    question: Option<String>,
    json: bool,
    strict: bool,
    explain: bool,
    clarify: bool,
) {
    let facts = match serve_facts(oxigraph_manager).await {
        Ok(facts) => facts,
        Err(error) => {
            eprintln!("读取知识库失败: {}", error);
            std::process::exit(1);
        }
    };
    let questions: Vec<String> = match question {
        Some(question) => vec![question],
        None => {
            let mut buffer = String::new();
            if let Err(error) = std::io::stdin().read_to_string(&mut buffer) {
                eprintln!("读取标准输入失败: {}", error);
                std::process::exit(1);
            }
            buffer
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .map(str::to_string)
                .collect()
        }
    };
    if questions.is_empty() {
        eprintln!("没有问题可回答：请提供问句参数，或从标准输入逐行输入。");
        std::process::exit(1);
    }

    let mut reasoner = build_store_reasoner(config, oxigraph_manager, store_path, &facts).await;

    // 与交互/服务同一条对话前通道：查询板、指代、澄清与身份回合都在
    // reasoner.answer 之前；一次 ask 的多行输入共享同一会话记忆。
    let mut dialogue = ask_dialogue(config, recovery_mode);
    let mut lens_cache: Option<(viewpoint::LensCorpus, BTreeSet<String>)> = None;
    let mut import_episodes: Option<Vec<know_prolog::import_episode::ImportEpisode>> = None;
    let mut answers: Vec<AskAnswer> = Vec::with_capacity(questions.len());
    for question in questions {
        // 管道输入常沿用交互脚本的收尾行：不要把它当成待回答的问题。
        if is_ask_terminator(&question) {
            break;
        }
        // 风格指令与计划审计优先于澄清/悬问回填：脚本里的「简短点」
        // 不会被误当成对上一轮缺失前提的短答。
        if let Some(directive) = StyleDirective::parse(&question) {
            reasoner.clear_pending_awaiting();
            let text = reasoner
                .restyle_generation(directive)
                .unwrap_or_else(|| "生成功能未启用，无法按回答计划重述。".to_string());
            answers.push(AskAnswer {
                question,
                status: "answered",
                answer: Some(text),
                derivation: None,
            });
            continue;
        }
        if matches!(
            question.as_str(),
            "回答计划" | "计划台账" | "生成计划" | "回答方案"
        ) {
            reasoner.clear_pending_awaiting();
            answers.push(AskAnswer {
                question,
                status: "answered",
                answer: Some(reasoner.answer_plan_report()),
                derivation: None,
            });
            continue;
        }
        if matches!(
            question.as_str(),
            "复核计划" | "计划复核" | "核对计划" | "检查计划"
        ) {
            reasoner.clear_pending_awaiting();
            answers.push(AskAnswer {
                question,
                status: "answered",
                answer: Some(reasoner.review_generation_plan()),
                derivation: None,
            });
            continue;
        }
        if matches!(
            question.as_str(),
            "置信账本" | "信念账本" | "跨轮信念" | "belief-ledger"
        ) {
            reasoner.clear_pending_awaiting();
            answers.push(AskAnswer {
                question,
                status: "answered",
                answer: Some(reasoner.belief_ledger_report()),
                derivation: None,
            });
            continue;
        }
        // 批次线索优先于澄清/悬问回填：问「这批」时不应被当成上一轮短答。
        if know_prolog::import_episode::looks_like_episode_reference(&question) {
            if import_episodes.is_none() {
                import_episodes = Some(
                    oxigraph_manager
                        .load_import_episodes()
                        .await
                        .unwrap_or_else(|error| {
                            eprintln!("读取导入批次失败: {}", error);
                            Vec::new()
                        }),
                );
            }
            let episodes = import_episodes.as_deref().unwrap_or(&[]);
            match oxigraph_manager
                .answer_import_episode_question(&question, episodes)
                .await
            {
                Ok(Some(answer)) => {
                    reasoner.clear_pending_awaiting();
                    answers.push(AskAnswer {
                        question,
                        status: "answered",
                        answer: Some(answer),
                        derivation: None,
                    });
                    continue;
                }
                Ok(None) => {}
                Err(error) => {
                    eprintln!("读取导入批次失败: {}", error);
                    answers.push(AskAnswer {
                        question,
                        status: "unanswered",
                        answer: None,
                        derivation: None,
                    });
                    continue;
                }
            }
        }
        // 语义歧义澄清中的下一行：序号/前者/后者选定读法，都不是则放弃。
        if reasoner.formal_clarification_active() {
            match reasoner.answer_formal_clarification(&question) {
                FormalClarificationOutcome::Answered(text)
                | FormalClarificationOutcome::Cancelled(text) => {
                    answers.push(AskAnswer {
                        question,
                        status: "answered",
                        answer: Some(text),
                        derivation: None,
                    });
                    continue;
                }
                FormalClarificationOutcome::Reprompt(prompt) => {
                    answers.push(AskAnswer {
                        question,
                        status: "needs_clarification",
                        answer: Some(prompt),
                        derivation: None,
                    });
                    continue;
                }
                FormalClarificationOutcome::NotAnAnswer => {}
            }
        }
        // 澄清对话中的下一行按回答处理：脚本可以用多行 stdin 完成
        // 「问题 → 是/不是 → 条件化结论」的整段解释空间收敛。
        if clarify && reasoner.clarification_active() {
            if matches!(question.as_str(), "结束澄清" | "停止澄清" | "退出澄清") {
                if let Some(text) = reasoner.stop_clarification() {
                    answers.push(AskAnswer {
                        question,
                        status: "answered",
                        answer: Some(text),
                        derivation: None,
                    });
                    continue;
                }
            }
            if let Some(text) = reasoner.answer_clarification(&question) {
                answers.push(AskAnswer {
                    question,
                    status: if reasoner.clarification_active() {
                        "needs_clarification"
                    } else {
                        "answered"
                    },
                    answer: Some(text),
                    derivation: None,
                });
                continue;
            }
        }
        // 悬问回填：脚本输入里的下一行短答直接绑定到上一轮缺失前提。
        if reasoner.has_pending_question() {
            if let Some(deferred) = reasoner.try_answer_pending_question(&question) {
                answers.push(AskAnswer {
                    question,
                    status: "answered",
                    answer: Some(deferred),
                    derivation: None,
                });
                continue;
            }
        }
        // 一次性问答同样只给紧接的一轮开裸回答窗口。
        reasoner.clear_pending_awaiting();
        if let Some(reply) =
            ask_with_lens(&question, oxigraph_manager, config, &mut lens_cache).await
        {
            answers.push(AskAnswer {
                question,
                status: "answered",
                answer: Some(reply),
                derivation: None,
            });
            continue;
        }
        let followup = generation_followup(&question);
        let intro_entity = if followup.is_none() {
            intro_request_entity(&question, &reasoner)
        } else {
            None
        };
        let mut answer = if let Some(request) = followup {
            match request {
                GenerationFollowup::Continue => reasoner.continue_generation(),
                GenerationFollowup::SwitchAngle => reasoner.switch_generation_angle(),
            }
        } else if let Some(entity) = intro_entity {
            let generated = reasoner
                .generate_about(&entity)
                .or_else(|| reasoner.narrate(&entity))
                .or_else(|| Some(format!("知识图谱中没有关于「{}」的可生成事实。", entity)));
            if generated.is_some() {
                // 介绍轮也推进焦点栈：下一轮「它的人口呢」必须能接上实体。
                dialogue.note_plugin_focus(&entity);
            }
            generated
        } else if let Some(reply) = dialogue.plan_pure_dialogue(&question, &reasoner) {
            // 元话语/情态承接优先，避免把「还有呢」误当实体省略补全。
            Some(reply.text)
        } else {
            // 与交互模式共用同一条上下文前通道：管道输入里的
            // 「那李四呢」「他的师傅是谁」也按上一轮计划/谓词补全。
            let contextual = reasoner.contextualize_with_receipt(&question);
            let effective = contextual
                .as_ref()
                .map(|rewrite| rewrite.text.as_str())
                .unwrap_or(question.as_str());
            dialogue
                .plan_reply(effective, &reasoner)
                .map(|reply| reply.text)
                .or_else(|| reasoner.answer(effective))
        };
        if answer
            .as_deref()
            .map_or(false, |text| !is_no_record_text(text))
        {
            // 一次性问答也做轮末预取：下一行继续追问时可命中同纪元草稿。
            reasoner.prefetch_answer_drafts();
        }
        let mut status = if answer.is_some() {
            "answered"
        } else {
            "unanswered"
        };
        let mut clarification_started = false;
        if clarify {
            let unanswered = match answer.as_deref() {
                Some(text) => is_no_record_text(text),
                None => true,
            };
            if unanswered {
                if let Some(prompt) = reasoner.begin_clarification_for_question(&question) {
                    answer = Some(match answer {
                        Some(text) if !text.trim().is_empty() => {
                            format!("{}\n{}", text.trim_end(), prompt)
                        }
                        _ => prompt,
                    });
                    status = "needs_clarification";
                    clarification_started = true;
                }
            }
        }
        if !clarification_started {
            if let Some(text) = &answer {
                if is_no_record_text(text) {
                    if let Some(prompt) = reasoner.open_pending_question(&question) {
                        answer = Some(format!("{}\n{}", text, prompt));
                    }
                } else {
                    reasoner.dismiss_pending_question(&question);
                }
            }
        }
        // 追问依据是惰性通道：--explain 才重建推导，回答热路径保持零开销。
        let derivation = if explain && status == "answered" {
            answer
                .as_deref()
                .and_then(|text| reasoner.answer_derivation(&question, Some(text)))
        } else {
            None
        };
        answers.push(AskAnswer {
            question,
            status,
            answer,
            derivation,
        });
    }

    if json {
        match serde_json::to_string_pretty(&answers) {
            Ok(text) => println!("{}", text),
            Err(error) => {
                eprintln!("回答序列化失败: {}", error);
                std::process::exit(1);
            }
        }
    } else {
        for (index, answer) in answers.iter().enumerate() {
            if index > 0 {
                println!();
            }
            println!("> {}", answer.question);
            match &answer.answer {
                Some(text) => println!("{}", text),
                None => println!("未能回答：当前知识库中没有可用依据。"),
            }
            if let Some(derivation) = &answer.derivation {
                println!("{}", derivation.render_explanation());
            } else if explain && answer.answer.is_some() {
                println!("（没有可展开的归结证明：这条回答来自检索、生成或对话通道。）");
            }
        }
    }
    if strict && answers.iter().any(|answer| answer.answer.is_none()) {
        std::process::exit(1);
    }
}

async fn ask_with_lens(
    question: &str,
    manager: &OxigraphManager,
    config: &SystemConfig,
    cache: &mut Option<(viewpoint::LensCorpus, BTreeSet<String>)>,
) -> Option<String> {
    if !viewpoint::looks_contextual(question) {
        return None;
    }
    if cache.is_none() {
        let ledger = manager.load_fusion_ledger().await.unwrap_or_default();
        let default_facts = manager.load_fact_refs().await.unwrap_or_default();
        let corpus = viewpoint::LensCorpus::new(&default_facts, &ledger);
        let mut functional: BTreeSet<String> = manager
            .predicate_registry_snapshot()
            .functional_predicates()
            .into_iter()
            .collect();
        for profile in &config.predicate_profiles {
            if profile.functional {
                functional.insert(profile.name.clone());
            }
        }
        *cache = Some((corpus, functional));
    }
    let (corpus, functional) = cache.as_ref()?;
    let aliases = corpus.aliases();
    let cue = viewpoint::detect_lens_cue(question, &aliases)?;
    let Some(mut query) = viewpoint::extract_slot_query(&cue.question) else {
        return match cue.scope {
            viewpoint::LensScope::Default | viewpoint::LensScope::All => None,
            _ => Some(format!(
                "已识别到语境限定「{}」，但没能把问句编译成槽位查询；请用 `lens --subject … --predicate … --view …` 明确指定。",
                cue.surface
            )),
        };
    };
    let registry = manager.predicate_registry_snapshot();
    let resolution = registry.resolve(&query.predicate);
    let mut variants = vec![query.predicate.clone()];
    if resolution.known {
        variants.push(resolution.canonical.clone());
        variants.extend(registry.aliases_of(&resolution.canonical));
    }
    query = query.with_predicate_aliases(variants);
    let is_functional = query
        .predicates
        .iter()
        .any(|predicate| functional.contains(predicate));
    Some(
        corpus
            .query_slot(&cue.scope, &query, is_functional)
            .render(),
    )
}

#[allow(clippy::too_many_arguments)]
async fn run_variants(
    config: &SystemConfig,
    oxigraph_manager: &OxigraphManager,
    surface: Option<String>,
    canonical: Option<String>,
    expected_type: Option<String>,
    predicate: Option<String>,
    domain: Option<String>,
    range: Option<String>,
    year: Option<i32>,
    echoes: &[String],
    morphology: bool,
    include_pending: bool,
    seed_config: bool,
    json: bool,
) {
    let mut graph = match oxigraph_manager.load_synonym_graph().await {
        Ok(graph) => graph,
        Err(error) => {
            eprintln!("读取同义词图失败: {}", error);
            std::process::exit(1);
        }
    };
    let turn = graph.turn.max(1);
    let mut configured = None;
    if seed_config {
        oxigraph_manager.configure_entities(&config.entity_profiles);
        let seeded = SynonymGraph::from_config(config, turn);
        graph.merge_from(&seeded);
        configured = Some(seeded);
    }
    let registry = oxigraph_manager.entity_registry_snapshot();
    let mut closure = VariantClosure::from_registry(&registry);
    if let Some(configured) = configured.as_ref() {
        closure.add_synonym_graph(configured);
    }
    closure.add_synonym_graph_including(&graph, include_pending);

    let mut context = VariantContext::new().including_pending_conditional(include_pending);
    if let Some(expected_type) = expected_type.as_deref() {
        context = context.with_expected_type(expected_type);
    }
    if let Some(predicate) = predicate.as_deref() {
        context = context.with_predicate(predicate);
    }
    if let Some(domain) = domain.as_deref() {
        context = context.with_domain(domain);
    }
    if let Some(range) = range.as_deref() {
        context = context.with_range(range);
    }
    if let Some(year) = year {
        context = context.at_year(year);
    }

    if let Some(surface) = surface.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        let resolution = if morphology {
            closure.resolve_with_morphology(surface, &context)
        } else {
            closure.resolve(surface, &context)
        };
        if json {
            match serde_json::to_string_pretty(&resolution) {
                Ok(text) => println!("{}", text),
                Err(error) => {
                    eprintln!("名变体解析序列化失败: {}", error);
                    std::process::exit(1);
                }
            }
        } else {
            println!("{}", resolution.render());
        }
        return;
    }

    if let Some(canonical) = canonical.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        let choice = closure.surface_for(canonical, &context, echoes);
        if json {
            match serde_json::to_string_pretty(&choice) {
                Ok(text) => println!("{}", text),
                Err(error) => {
                    eprintln!("名变体反向生成序列化失败: {}", error);
                    std::process::exit(1);
                }
            }
        } else {
            println!("{}", choice.render());
        }
        return;
    }

    if json {
        println!(
            "{}",
            serde_json::json!({
                "anchors": closure.anchor_count(),
                "edges": closure.edge_count(),
                "report": closure.report(),
            })
        );
    } else {
        println!("{}", closure.report());
    }
}

async fn run_synonyms(
    config: &SystemConfig,
    oxigraph_manager: &OxigraphManager,
    surface: Option<String>,
    canonical: Option<String>,
    kind: &str,
    expected_type: Option<String>,
    predicate: Option<String>,
    domain: Option<String>,
    range: Option<String>,
    echoes: &[String],
    confirm: bool,
    refute: bool,
    seed_config: bool,
    json: bool,
) {
    let Some(kind) = SynonymKind::parse(kind) else {
        eprintln!("未知同义层级「{}」：可用 entity / predicate / type。", kind);
        std::process::exit(2);
    };
    if confirm && refute {
        eprintln!("--confirm 与 --refute 不能同时使用。");
        std::process::exit(2);
    }
    let mut graph = match oxigraph_manager.load_synonym_graph().await {
        Ok(graph) => graph,
        Err(error) => {
            eprintln!("读取同义词图失败: {}", error);
            std::process::exit(1);
        }
    };
    let turn = graph.turn.max(1);
    if seed_config {
        let configured = SynonymGraph::from_config(config, turn);
        graph.merge_from(&configured);
    }
    let mut context = SynonymContext::new();
    if let Some(expected_type) = expected_type.as_deref() {
        context = context.with_expected_type(expected_type);
    }
    if let Some(predicate) = predicate.as_deref() {
        context = context.with_predicate(predicate);
    }
    if let Some(domain) = domain.as_deref() {
        context = context.with_domain(domain);
    }
    if let Some(range) = range.as_deref() {
        context = context.with_range(range);
    }
    let sense = SynonymSense::new(
        expected_type.as_deref(),
        domain.as_deref().or(predicate.as_deref()),
        range.as_deref(),
    );

    let persist = seed_config;
    if confirm || refute {
        let Some(surface) = surface.as_deref().map(str::trim).filter(|s| !s.is_empty()) else {
            eprintln!("--confirm/--refute 需要 --surface。");
            std::process::exit(2);
        };
        let Some(canonical) = canonical
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        else {
            eprintln!("--confirm/--refute 需要 --canonical。");
            std::process::exit(2);
        };
        let existing = graph
            .entries()
            .find(|entry| {
                entry.surface == surface && entry.canonical == canonical && entry.kind == kind
            })
            .map(|entry| entry.id.clone());
        if refute {
            let Some(id) = existing else {
                eprintln!(
                    "同义词图中没有「{}」→「{}」（{}），无法否证。",
                    surface,
                    canonical,
                    kind.label()
                );
                std::process::exit(1);
            };
            graph.refute(&id, turn, Some("CLI 显式否证"));
            if let Err(error) = oxigraph_manager.sync_synonym_graph(&graph).await {
                eprintln!("写入同义词图失败: {}", error);
                std::process::exit(1);
            }
            if json {
                println!(
                    "{}",
                    serde_json::json!({
                        "action": "refute",
                        "id": id,
                        "surface": surface,
                        "canonical": canonical,
                        "kind": kind,
                    })
                );
            } else {
                println!("已否证「{}」→「{}」（{}）。", surface, canonical, kind.label());
            }
            return;
        }
        let id = match existing {
            Some(id) => {
                graph.confirm(&id, turn);
                id
            }
            None => {
                let outcome = graph.observe(
                    turn,
                    surface,
                    canonical,
                    kind,
                    sense,
                    "cli",
                    1.0,
                );
                let Some(id) = outcome.id().map(str::to_string) else {
                    eprintln!("无法登记「{}」→「{}」：词面或规范名为空/过长。", surface, canonical);
                    std::process::exit(1);
                };
                graph.confirm(&id, turn);
                id
            }
        };
        if let Err(error) = oxigraph_manager.sync_synonym_graph(&graph).await {
            eprintln!("写入同义词图失败: {}", error);
            std::process::exit(1);
        }
        if json {
            println!(
                "{}",
                serde_json::json!({
                    "action": "confirm",
                    "id": id,
                    "surface": surface,
                    "canonical": canonical,
                    "kind": kind,
                })
            );
        } else {
            println!("已确认「{}」→「{}」（{}）。", surface, canonical, kind.label());
        }
        return;
    }

    if let Some(surface) = surface.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        let resolution = graph.canonicalize(surface, kind, &context);
        if json {
            match serde_json::to_string_pretty(&resolution) {
                Ok(text) => println!("{}", text),
                Err(error) => {
                    eprintln!("同义解析序列化失败: {}", error);
                    std::process::exit(1);
                }
            }
        } else {
            match &resolution {
                SynonymResolution::Resolved(reading) => {
                    if reading.changes_surface() {
                        println!(
                            "「{}」→「{}」（{}）",
                            reading.surface,
                            reading.canonical,
                            reading.reason
                        );
                    } else {
                        println!(
                            "「{}」保持原词面（{}，词义 {}）",
                            reading.surface,
                            reading.reason,
                            reading.sense.label()
                        );
                    }
                }
                SynonymResolution::Ambiguous {
                    surface,
                    candidates,
                    ..
                } => {
                    println!("「{}」有多个词义，保持原词面：", surface);
                    for candidate in candidates {
                        println!(
                            "  - 「{}」{}（{}，得分 {:.2}）",
                            candidate.canonical,
                            candidate.kind.label(),
                            candidate.sense.label(),
                            candidate.score
                        );
                    }
                }
                SynonymResolution::Unresolved {
                    surface, reason, ..
                } => {
                    println!("「{}」不改写：{}。", surface, reason.label());
                }
            }
        }
    } else if let Some(canonical) = canonical
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        let choice = graph.surface_for(canonical, kind, &context, echoes);
        if json {
            match serde_json::to_string_pretty(&choice) {
                Ok(text) => println!("{}", text),
                Err(error) => {
                    eprintln!("反向词面序列化失败: {}", error);
                    std::process::exit(1);
                }
            }
        } else {
            println!(
                "「{}」→「{}」（{}，{}）",
                choice.canonical,
                choice.surface,
                choice.source,
                if choice.reversible {
                    "可逆"
                } else {
                    "不可逆"
                }
            );
        }
    } else {
        if json {
            println!(
                "{}",
                serde_json::json!({
                    "turn": graph.turn,
                    "capacity": graph.capacity,
                    "entries": graph.entries().collect::<Vec<_>>(),
                    "conflicts": graph.conflicts(),
                    "fingerprint": graph.fingerprint(),
                })
            );
        } else {
            print!("{}", graph.render_report());
        }
    }
    if persist {
        if let Err(error) = oxigraph_manager.sync_synonym_graph(&graph).await {
            eprintln!("写入同义词图失败: {}", error);
            std::process::exit(1);
        }
    }
}

async fn run_types(
    config: &SystemConfig,
    oxigraph_manager: &OxigraphManager,
    type_name: Option<String>,
    compare: Option<Vec<String>>,
    members: bool,
    offset: usize,
    limit: usize,
    json: bool,
) {
    let facts = match load_graph_snapshot(oxigraph_manager).await {
        Ok(facts) => facts,
        Err(error) => {
            eprintln!("读取知识库失败: {}", error);
            std::process::exit(1);
        }
    };
    let mut generator = SymbolicGenerator::new(
        config.reasoning.max_profile_sentences,
        config.reasoning.completion_min_score,
    );
    for fact in facts.iter() {
        match fact {
            PrologFact::Entity {
                name,
                entity_type,
                properties,
            } => generator.observe_entity_map(name, entity_type, properties),
            PrologFact::Relation { from, rel_type, to } => {
                generator.observe_triple(from, rel_type, to)
            }
        }
    }

    let available = || generator.all_types().join("、");
    let resolve = |mention: &str| {
        generator.resolve_type_name(mention).unwrap_or_else(|| {
            let names = available();
            if names.is_empty() {
                eprintln!("知识图谱中没有带类型的实体。");
            } else {
                eprintln!("未知类型「{}」。当前类型：{}", mention, names);
            }
            std::process::exit(1);
        })
    };

    if let Some(pair) = compare {
        if pair.len() != 2 {
            eprintln!("--compare 需要正好两个类型。");
            std::process::exit(2);
        }
        let left = resolve(&pair[0]);
        let right = resolve(&pair[1]);
        if left == right {
            eprintln!("--compare 的两个类型不能相同：{}", left);
            std::process::exit(2);
        }
        let Some(comparison) = generator.compare_types(&left, &right) else {
            eprintln!("类型「{}」或「{}」没有可用画像。", left, right);
            std::process::exit(1);
        };
        if json {
            match serde_json::to_string_pretty(&comparison) {
                Ok(text) => println!("{}", text),
                Err(error) => {
                    eprintln!("类型对比序列化失败: {}", error);
                    std::process::exit(1);
                }
            }
        } else {
            println!("{}", comparison.render());
        }
        return;
    }

    if let Some(mention) = type_name {
        let etype = resolve(&mention);
        if members {
            let Some(page) = generator.type_member_page(&etype, offset, limit) else {
                eprintln!("类型「{}」没有成员。", etype);
                std::process::exit(1);
            };
            if json {
                match serde_json::to_string_pretty(&page) {
                    Ok(text) => println!("{}", text),
                    Err(error) => {
                        eprintln!("成员分页序列化失败: {}", error);
                        std::process::exit(1);
                    }
                }
            } else {
                println!("{}", page.render());
            }
            return;
        }
        if offset != 0 || limit != 20 {
            eprintln!("提示：--offset/--limit 仅在 --members 时生效。");
        }
        let Some(profile) = generator.type_profile(&etype) else {
            eprintln!("类型「{}」没有可用画像。", etype);
            std::process::exit(1);
        };
        if json {
            match serde_json::to_string_pretty(&profile) {
                Ok(text) => println!("{}", text),
                Err(error) => {
                    eprintln!("类型画像序列化失败: {}", error);
                    std::process::exit(1);
                }
            }
        } else {
            println!("{}", profile.render());
        }
        return;
    }

    let types = generator.all_types();
    let profiles: Vec<_> = types
        .iter()
        .filter_map(|etype| generator.type_profile(etype))
        .collect();
    if json {
        match serde_json::to_string_pretty(&profiles) {
            Ok(text) => println!("{}", text),
            Err(error) => {
                eprintln!("类型总览序列化失败: {}", error);
                std::process::exit(1);
            }
        }
        return;
    }
    if profiles.is_empty() {
        println!("知识图谱中没有带类型的实体。");
        return;
    }
    println!("类型总览（{} 类；按显式 rdf:type 统计）：", profiles.len());
    for profile in &profiles {
        println!(
            "  - {}：{} 个成员；{} 种谓词，覆盖 {}/{} 个成员",
            profile.type_name,
            profile.total_members,
            profile.total_predicates,
            profile.covered_members,
            profile.total_members
        );
    }
    println!(
        "使用 --type TYPE 查看画像，--type TYPE --members 查看成员，--compare LEFT RIGHT 对比。"
    );
}

/// 编译并执行形状校验（`eval-shapes` 与交互命令共用同一确定性路径）。
async fn shape_report_for(
    oxigraph_manager: &OxigraphManager,
    config: &know_prolog::SystemConfig,
) -> Result<(ShapeSet, ShapeReport), Box<dyn std::error::Error + Send + Sync>> {
    let facts = oxigraph_manager.load_graph_snapshot().await?;
    let registry = oxigraph_manager.predicate_registry_snapshot();
    let types = oxigraph_manager.type_system_snapshot();
    let mut shapes = ShapeSet::from_schema(&config.entity_types, &registry, &types);
    let attributes =
        know_prolog::attribute_bridge::AttributeRegistry::from_config(&config.attributes);
    shapes.apply_attribute_profiles(&attributes);
    let report = shapes.validate(&facts, &registry, &types);
    Ok((shapes, report))
}

/// 关系获取的只读上下文：图快照 + 谓词画像 + 类型体系 + 形状集。
async fn acquisition_context(
    oxigraph_manager: &OxigraphManager,
    config: &know_prolog::SystemConfig,
) -> Result<
    (
        Arc<Vec<PrologFact>>,
        know_prolog::predicate_registry::PredicateRegistry,
        know_prolog::type_system::TypeSystem,
        ShapeSet,
    ),
    Box<dyn std::error::Error + Send + Sync>,
> {
    let facts = load_graph_snapshot(oxigraph_manager).await?;
    let registry = oxigraph_manager.predicate_registry_snapshot();
    let types = oxigraph_manager.type_system_snapshot();
    let mut shapes = ShapeSet::from_schema(&config.entity_types, &registry, &types);
    let attributes =
        know_prolog::attribute_bridge::AttributeRegistry::from_config(&config.attributes);
    shapes.apply_attribute_profiles(&attributes);
    Ok((facts, registry, types, shapes))
}

fn persist_acquisition(acquisition: &RelationAcquisition, path: &str) {
    if let Err(error) = acquisition.save_to_path(path) {
        eprintln!("关系获取账本保存失败：{}", error);
    }
}

fn persist_assertion_reader(reader: &TeachableAssertionReader, path: &str) {
    if let Err(error) = reader.save_to_path(path) {
        eprintln!("断言阅读器保存失败：{}", error);
    }
}

fn persist_pattern_forge(forge: &PatternForge, path: &str) {
    if let Err(error) = forge.save_to_path(path) {
        eprintln!("锻造台账保存失败：{}", error);
    }
}

fn persist_pattern_evolution(ledger: &EvolutionLedger, path: &str) {
    if let Err(error) = ledger.save_to_path(path) {
        eprintln!("模式演化台账保存失败：{}", error);
    }
}

/// 模式命中回执里若来源是锻造规则，则给该规则记一次成功使用。
///
/// 只做计数，不改变回答；台账按需落盘，避免每轮都写 JSON。
fn record_forge_hit(forge: &mut PatternForge, note: &str, path: &str) {
    let Some(rest) = note.split("来源 ").nth(1) else {
        return;
    };
    let source = rest
        .split('，')
        .next()
        .unwrap_or(rest)
        .trim()
        .trim_end_matches(['）', ')']);
    if source.starts_with(pattern_forge::FORGE_SOURCE_PREFIX) && forge.record_hit(source) {
        persist_pattern_forge(forge, path);
    }
}

fn persist_shape_dialogue(dialogue: &ShapeDialogue, path: &str) {
    if let Err(error) = dialogue.save_to_path(path) {
        eprintln!("形状对话状态保存失败：{}", error);
    }
}

/// 把质量护照注入对话管理器；仅在知识版本推进时重建（离线派生，不写图谱）。
async fn refresh_dialogue_quality(
    dialogue_manager: &mut DialogueManager,
    oxigraph_manager: &OxigraphManager,
    config: &know_prolog::SystemConfig,
    revision: u64,
    store_path: &str,
) {
    if dialogue_manager.quality_revision() == Some(revision) {
        return;
    }
    let Ok(facts) = oxigraph_manager.load_graph_snapshot().await else {
        return;
    };
    let registry = oxigraph_manager.predicate_registry_snapshot();
    let types = oxigraph_manager.type_system_snapshot();
    let ledger = provenance_seed_for_store(Some(store_path), None);
    let mut report = build_report_with_provenance(
        &facts,
        &registry,
        &types,
        &config.entity_types,
        Some(&ledger),
    );
    if let Ok(fact_refs) = oxigraph_manager.load_fact_refs().await {
        know_prolog::quality::apply_witnessed_conflicts(
            &mut report,
            &fact_refs,
            Some(&ledger),
            &registry,
            &types,
        );
    }
    let index =
        QualityIndex::from_report_with_provenance(&report, &facts, &registry, Some(&ledger))
            .with_revision(revision);
    dialogue_manager.set_quality_index(Some(index));
}

/// 把一份语言配置视图装回所有语言消费者。
///
/// 只重建配置派生层（抽取器、措辞器、NLU、模式库配置层、插件词性线索、
/// 同义词表、本地语义开关与推理器语言策略）；对话焦点、编年史、会话事实、
/// 教学模式与承诺全部保持原对象不变。
#[allow(clippy::type_complexity)]
fn apply_language_snapshot(
    config: &Arc<SystemConfig>,
    knowledge_extractor: &mut KnowledgeExtractor,
    response_formatter: &mut ResponseFormatter,
    nlu_engine: &mut NLUEngine,
    pattern_library: &mut PatternLibrary,
    plugin_runtime: &PluginRuntime,
    plugin_nlp: &mut know_prolog::NlpConfig,
    synonyms: &mut std::collections::HashMap<String, String>,
    local_semantic_adapter: &mut LocalSemanticAdapter,
    symbolic_reasoner: &mut SymbolicReasoner,
) -> (pattern_ir::PatternLayerRefresh, LanguagePolicySync, usize) {
    *knowledge_extractor = KnowledgeExtractor::new(Arc::clone(config));
    *response_formatter = ResponseFormatter::new(Arc::clone(config));
    *nlu_engine = NLUEngine::new(config);
    let patterns = pattern_library.replace_config_layer(config);
    *plugin_nlp = plugin_runtime.augmented_nlp_config(&config.nlp);
    let plugin_entries = plugin_nlp
        .pos_dict
        .len()
        .saturating_sub(config.nlp.pos_dict.len());
    // 同义词是会话内单调追加：新配置可登记/改写映射，但删除旧映射需重启
    // 才能同时清掉已学指称与推理器里的别名缓存。
    if let Some(configured) = &config.synonyms {
        for (alias, canonical) in &configured.mappings {
            synonyms.insert(alias.clone(), canonical.clone());
        }
    }
    *local_semantic_adapter = LocalSemanticAdapter::new(
        config.nlp.enable_local_semantic.unwrap_or(true)
            && config.nlp.disable_ownthink_api.unwrap_or(false),
    )
    .with_context_window_size(config.nlp.context_window_size);
    let language = symbolic_reasoner.apply_language_policy(config);
    (patterns, language, plugin_entries)
}

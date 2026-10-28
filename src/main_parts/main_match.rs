    match cli.command {
        Some(Commands::ImportJsonld { file_path }) => {
            load_persistent_registries(&config, &oxigraph_manager, &store_path);
            let import_result = if file_path == "-" {
                println!("正在从标准输入导入 JSON-LD...");
                let mut content = String::new();
                if let Err(error) = std::io::stdin().read_to_string(&mut content) {
                    eprintln!("读取标准输入失败: {}", error);
                    std::process::exit(1);
                }
                oxigraph_manager
                    .import_jsonld_batch(&content, "stdin")
                    .await
            } else {
                println!("正在导入 JSON-LD 文件: {}", file_path);
                oxigraph_manager
                    .import_jsonld_file_batch(&file_path)
                    .await
            };

            match import_result {
                Ok(receipt) => {
                    println!("JSON-LD 导入成功：{}", receipt.summary());
                    println!(
                        "批次 B{}｜id {}｜批次图 {}",
                        receipt.episode.seq,
                        receipt.episode.id,
                        receipt.episode.batch_graph
                    );
                    if receipt.episode.blank_nodes {
                        println!(
                            "注意：本批含空白节点；同一文档重复导入可能得到不同批次 id。"
                        );
                    }
                    if let Err(error) =
                        persist_registry_observations(&oxigraph_manager, &store_path).await
                    {
                        eprintln!("导入已写入，但画像同步失败：{}", error);
                    }
                }
                Err(e) => {
                    eprintln!("JSON-LD 导入失败: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Some(Commands::ImportJsonldContent { content }) => {
            load_persistent_registries(&config, &oxigraph_manager, &store_path);
            println!("正在导入 JSON-LD 内容...");

            match oxigraph_manager
                .import_jsonld_batch(&content, "内容")
                .await
            {
                Ok(receipt) => {
                    println!("JSON-LD 内容导入成功：{}", receipt.summary());
                    println!(
                        "批次 B{}｜id {}｜批次图 {}",
                        receipt.episode.seq,
                        receipt.episode.id,
                        receipt.episode.batch_graph
                    );
                    if receipt.episode.blank_nodes {
                        println!(
                            "注意：本批含空白节点；同一内容重复导入可能得到不同批次 id。"
                        );
                    }
                    if let Err(error) =
                        persist_registry_observations(&oxigraph_manager, &store_path).await
                    {
                        eprintln!("导入已写入，但画像同步失败：{}", error);
                    }
                }
                Err(e) => {
                    eprintln!("JSON-LD 内容导入失败: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Some(Commands::ImportRdf { file_path, format }) => {
            load_persistent_registries(&config, &oxigraph_manager, &store_path);
            let import_result = if file_path == "-" {
                let Some(format_name) = format.as_deref() else {
                    eprintln!(
                        "从标准输入导入 RDF 时必须用 --format 指定格式（ttl / nt / nq / trig / rdf / xml）"
                    );
                    std::process::exit(2);
                };
                let format = match OxigraphManager::parse_rdf_format(format_name) {
                    Ok(format) => format,
                    Err(error) => {
                        eprintln!("{}", error);
                        std::process::exit(2);
                    }
                };
                println!("正在从标准输入导入 RDF（{}）...", format_name);
                let mut content = Vec::new();
                if let Err(error) = std::io::stdin().read_to_end(&mut content) {
                    eprintln!("读取标准输入失败: {}", error);
                    std::process::exit(1);
                }
                oxigraph_manager
                    .import_rdf_batch(format, content.as_slice(), "stdin")
                    .await
            } else {
                println!("正在导入 RDF 文件: {}", file_path);
                if let Some(format_name) = format.as_deref() {
                    let format = match OxigraphManager::parse_rdf_format(format_name) {
                        Ok(format) => format,
                        Err(error) => {
                            eprintln!("{}", error);
                            std::process::exit(2);
                        }
                    };
                    match std::fs::File::open(&file_path) {
                        Ok(file) => {
                            oxigraph_manager
                                .import_rdf_batch(format, file, &file_path)
                                .await
                        }
                        Err(error) => Err(error.into()),
                    }
                } else {
                    oxigraph_manager.import_rdf_file_batch(&file_path).await
                }
            };

            match import_result {
                Ok(receipt) => {
                    println!("RDF 导入成功：新增 {} 条三元组！", receipt.added);
                    println!("{}", receipt.summary());
                    println!(
                        "批次 B{}｜id {}｜批次图 {}",
                        receipt.episode.seq,
                        receipt.episode.id,
                        receipt.episode.batch_graph
                    );
                    if receipt.episode.blank_nodes {
                        println!(
                            "注意：本批含空白节点；解析器会重命名空白节点，同一文档重复导入可能得到不同批次 id。"
                        );
                    }
                    if let Err(error) =
                        persist_registry_observations(&oxigraph_manager, &store_path).await
                    {
                        eprintln!("导入已写入，但画像同步失败：{}", error);
                    }
                }
                Err(e) => {
                    eprintln!("RDF 导入失败: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Some(Commands::Episodes {
            reference,
            json,
            export,
        }) => {
            let episodes = match oxigraph_manager.load_import_episodes().await {
                Ok(episodes) => episodes,
                Err(error) => {
                    eprintln!("读取导入批次失败：{}", error);
                    std::process::exit(1);
                }
            };
            if let Some(file_path) = export {
                let Some(reference) = reference.as_deref() else {
                    eprintln!("--export 需要同时给出批次引用，例如：episodes 第2批 --export batch2.ttl");
                    std::process::exit(2);
                };
                let Some(episode) =
                    know_prolog::import_episode::resolve_episode_reference(reference, &episodes)
                else {
                    eprintln!(
                        "没有匹配的导入批次：{}（可用 episodes 查看当前批次）",
                        reference
                    );
                    std::process::exit(1);
                };
                match oxigraph_manager
                    .export_import_batch_file(episode, &file_path)
                    .await
                {
                    Ok(count) => println!(
                        "批次 B{} 导出成功：{} 条断言 → {}",
                        episode.seq, count, file_path
                    ),
                    Err(error) => {
                        eprintln!("批次导出失败：{}", error);
                        std::process::exit(1);
                    }
                }
                return;
            }
            if let Some(reference) = reference {
                if json {
                    match know_prolog::import_episode::resolve_episode_reference(
                        &reference,
                        &episodes,
                    ) {
                        Some(episode) => match serde_json::to_string_pretty(episode) {
                            Ok(text) => println!("{}", text),
                            Err(error) => {
                                eprintln!("JSON 序列化失败：{}", error);
                                std::process::exit(1);
                            }
                        },
                        None => {
                            eprintln!(
                                "没有匹配的导入批次：{}（可用 episodes 查看当前批次）",
                                reference
                            );
                            std::process::exit(1);
                        }
                    }
                } else {
                    match oxigraph_manager
                        .answer_import_episode_question(&reference, &episodes)
                        .await
                    {
                        Ok(Some(answer)) => println!("{}", answer),
                        Ok(None) => {
                            eprintln!(
                                "没有匹配的导入批次：{}（可用 episodes 查看当前批次）",
                                reference
                            );
                            std::process::exit(1);
                        }
                        Err(error) => {
                            eprintln!("读取导入批次失败：{}", error);
                            std::process::exit(1);
                        }
                    }
                }
            } else if json {
                match serde_json::to_string_pretty(&episodes) {
                    Ok(text) => println!("{}", text),
                    Err(error) => {
                        eprintln!("JSON 序列化失败：{}", error);
                        std::process::exit(1);
                    }
                }
            } else {
                println!(
                    "{}",
                    know_prolog::import_episode::render_episode_ledger(&episodes)
                );
            }
        }
        Some(Commands::RetractBatch { reference }) => {
            let episodes = match oxigraph_manager.load_import_episodes().await {
                Ok(episodes) => episodes,
                Err(error) => {
                    eprintln!("读取导入批次失败：{}", error);
                    std::process::exit(1);
                }
            };
            let Some(episode) =
                know_prolog::import_episode::resolve_episode_reference(&reference, &episodes)
            else {
                eprintln!(
                    "没有匹配的导入批次：{}（可用 episodes 查看当前批次）",
                    reference
                );
                std::process::exit(1);
            };
            match oxigraph_manager.retract_import_batch(episode).await {
                Ok(retraction) => println!("{}", retraction.summary()),
                Err(error) => {
                    eprintln!("撤销批次失败：{}", error);
                    std::process::exit(1);
                }
            }
            if let Err(error) =
                persist_registry_observations(&oxigraph_manager, &store_path).await
            {
                eprintln!("批次已撤销，但画像同步失败：{}", error);
            }
        }
        Some(Commands::BatchImport {
            file_path,
            dry_run,
            json,
        }) => {
            run_batch_import(
                &config,
                &oxigraph_manager,
                &store_path,
                &file_path,
                dry_run,
                json,
            )
            .await;
        }
        Some(Commands::Owl { args }) => {
            run_owl_command(&config, &oxigraph_manager, &store_path, args).await;
        }
        Some(Commands::ExportRdf { file_path }) => {
            if file_path == "-" {
                let stdout = std::io::stdout();
                match oxigraph_manager
                    .export_rdf_to_writer(oxigraph::io::RdfFormat::Turtle, stdout.lock())
                    .await
                {
                    Ok(count) => eprintln!("已导出 {} 条三元组（Turtle → stdout）", count),
                    Err(e) => {
                        eprintln!("RDF 导出失败: {}", e);
                        std::process::exit(1);
                    }
                }
            } else {
                match oxigraph_manager.export_rdf_file(&file_path).await {
                    Ok(count) => {
                        println!("RDF 文件导出成功：{} 条三元组 → {}", count, file_path);
                    }
                    Err(e) => {
                        eprintln!("RDF 文件导出失败: {}", e);
                        std::process::exit(1);
                    }
                }
            }
        }
        Some(Commands::Predicates {
            json,
            confirm,
            reject,
            alias,
            inverse,
            functional,
            non_functional,
        }) => {
            run_predicates(
                &config,
                &oxigraph_manager,
                &store_path,
                PredicateEdits {
                    confirm,
                    reject,
                    alias,
                    inverse,
                    functional,
                    non_functional,
                },
                json,
            )
            .await;
        }
        Some(Commands::LexicalMemory {
            learn,
            query,
            forget,
            json,
        }) => {
            let mut memory = match oxigraph_manager.load_lexical_memory().await {
                Ok(memory) => memory,
                Err(error) => {
                    eprintln!("词法记忆读取失败: {}", error);
                    std::process::exit(1);
                }
            };
            let previous_turn = memory.turn;
            if forget {
                memory = lexical_memory::LexicalMemory::new(memory.capacity);
                memory.turn = previous_turn;
            }
            let mut summaries = Vec::with_capacity(learn.len());
            for text in &learn {
                let turn = memory.turn.saturating_add(1);
                summaries.push(memory.observe_text(turn, text, &config.nlp));
            }
            if forget || !learn.is_empty() {
                if let Err(error) = oxigraph_manager.sync_lexical_memory(&memory).await {
                    eprintln!("词法记忆写入失败: {}", error);
                    std::process::exit(1);
                }
            }
            if let Some(word) = query {
                let expectation = memory
                    .type_expectation(&word)
                    .or_else(|| memory.classifier_expectation(&word));
                if json {
                    let payload = serde_json::json!({
                        "query": word,
                        "expectation": expectation,
                        "tokens": memory.len(),
                        "classifiers": memory.classifiers.len(),
                    });
                    match serde_json::to_string_pretty(&payload) {
                        Ok(text) => println!("{}", text),
                        Err(error) => eprintln!("JSON 序列化失败: {}", error),
                    }
                } else {
                    match expectation {
                        Some(expectation) => println!(
                            "「{}」→ {}（{}/{}，支持度 {:.2}{}）",
                            word,
                            expectation.expected_type,
                            expectation.votes,
                            expectation.total,
                            expectation.confidence(),
                            if expectation.conflict {
                                "，冲突"
                            } else {
                                ""
                            }
                        ),
                        None => println!("「{}」没有词法类型证据。", word),
                    }
                }
            } else if json {
                let payload = serde_json::json!({
                    "summary": summaries,
                    "forgot": forget,
                    "memory": memory,
                });
                match serde_json::to_string_pretty(&payload) {
                    Ok(text) => println!("{}", text),
                    Err(error) => eprintln!("JSON 序列化失败: {}", error),
                }
            } else {
                print!("{}", memory.render_report());
            }
        }
        Some(Commands::Sparql { query, json }) => {
            let query_text = if query == "-" {
                let mut buffer = String::new();
                if let Err(e) = std::io::stdin().read_to_string(&mut buffer) {
                    eprintln!("读取标准输入失败: {}", e);
                    std::process::exit(1);
                }
                buffer
            } else {
                query
            };
            match oxigraph_manager.query_sparql(&query_text).await {
                Ok(result) => {
                    if json {
                        match serde_json::to_string_pretty(&result) {
                            Ok(text) => println!("{}", text),
                            Err(e) => {
                                eprintln!("SPARQL 结果序列化失败: {}", e);
                                std::process::exit(1);
                            }
                        }
                    } else {
                        println!("{}", result.render_text());
                    }
                }
                Err(e) => {
                    eprintln!("SPARQL 查询失败: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Some(Commands::Ask {
            question,
            json,
            strict,
            explain,
            clarify,
        }) => {
            run_ask(
                &config,
                &oxigraph_manager,
                &store_path,
                recovery_mode,
                question,
                json,
                strict,
                explain,
                clarify,
            )
            .await;
        }
        Some(Commands::Types {
            type_name,
            compare,
            members,
            offset,
            limit,
            json,
        }) => {
            let registry_path = format!("{}/predicate_registry.json", store_path);
            let _ = oxigraph_manager.load_predicate_registry(&registry_path);
            oxigraph_manager.configure_predicates(
                &config.predicate_profiles,
                config.reasoning.require_predicate_admission,
            );
            oxigraph_manager.configure_type_system(&config.type_system);
            run_types(
                &config,
                &oxigraph_manager,
                type_name,
                compare,
                members,
                offset,
                limit,
                json,
            )
            .await;
        }
        Some(Commands::Segment {
            text,
            context,
            expected_types,
            predicate,
            margin,
            memory,
            confirm,
            correct,
            json,
        }) => {
            run_contextual_segment(
                &config,
                &oxigraph_manager,
                text.as_deref(),
                &context,
                &expected_types,
                predicate.as_deref(),
                margin,
                memory.as_deref(),
                &confirm,
                &correct,
                json,
            )
            .await;
        }
        Some(Commands::JointParse {
            text,
            context,
            expected_types,
            predicate,
            margin,
            json,
        }) => {
            run_joint_parse(
                &config,
                &oxigraph_manager,
                &store_path,
                text.as_deref(),
                &context,
                &expected_types,
                predicate.as_deref(),
                margin,
                json,
            )
            .await;
        }
        Some(Commands::JointVerify {
            text,
            fact,
            context,
            expected_types,
            predicate,
            json,
        }) => {
            run_joint_verify(
                &config,
                &oxigraph_manager,
                &store_path,
                text.as_deref(),
                &fact,
                &context,
                &expected_types,
                predicate.as_deref(),
                json,
            )
            .await;
        }
        Some(Commands::JointGenerate {
            fact,
            context,
            expected_types,
            predicate,
            json,
        }) => {
            run_joint_generate(
                &config,
                &oxigraph_manager,
                &store_path,
                &fact,
                &context,
                &expected_types,
                predicate.as_deref(),
                json,
            )
            .await;
        }
        Some(Commands::Frames {
            text,
            file,
            limit,
            samples,
            json,
        }) => {
            // 与 Laws 相同的只读装配：先载入持久画像/律典，再从当前存储取事实。
            let registry_path = format!("{}/predicate_registry.json", store_path);
            let _ = oxigraph_manager.load_predicate_registry(&registry_path);
            oxigraph_manager.configure_predicates(
                &config.predicate_profiles,
                config.reasoning.require_predicate_admission,
            );
            oxigraph_manager.configure_type_system(&config.type_system);
            let registry = oxigraph_manager.predicate_registry_snapshot();
            let types = oxigraph_manager.type_system_snapshot();
            let mut laws = LawBook::from_registry(&registry);
            if let Some(persisted) =
                LawBook::load_from_path(&format!("{}/law_book.json", store_path))
            {
                laws.apply_persisted(&persisted);
            }
            let facts = match oxigraph_manager.load_graph_snapshot().await {
                Ok(facts) => facts,
                Err(e) => {
                    eprintln!("读取知识库失败: {}", e);
                    std::process::exit(1);
                }
            };
            let source = match (text, file) {
                (Some(text), _) => text,
                (None, Some(path)) => match std::fs::read_to_string(&path) {
                    Ok(content) => content,
                    Err(e) => {
                        eprintln!("读取文本失败（{}）: {}", path, e);
                        std::process::exit(1);
                    }
                },
                (None, None) => match load_frame_corpus() {
                    Ok(content) => content,
                    Err(e) => {
                        eprintln!("{}", e);
                        std::process::exit(1);
                    }
                },
            };
            let (mut frames, stats) = FrameExtractor::new().extract_with_stats(&source);
            if limit > 0 {
                frames.truncate(limit);
            }
            let gate = FrameGate::from_facts(facts.as_slice(), registry, types, laws);
            let diagnostics = diagnose_frames(&frames, &gate, samples);
            if json {
                let reports: Vec<serde_json::Value> = frames
                    .iter()
                    .map(|frame| {
                        let report = gate.evaluate(frame);
                        serde_json::json!({
                            "frame": frame,
                            "report": report,
                            "narration": narrate_frame(frame, &report),
                        })
                    })
                    .collect();
                let payload = serde_json::json!({
                    "dialect": FRAME_DIALECT,
                    "extraction": stats,
                    "gate_facts": gate.fact_count(),
                    "diagnostics": diagnostics,
                    "frames": reports,
                });
                match serde_json::to_string_pretty(&payload) {
                    Ok(text) => println!("{}", text),
                    Err(e) => {
                        eprintln!("关系帧序列化失败: {}", e);
                        std::process::exit(1);
                    }
                }
            } else {
                print!("{}", diagnostics.render_text());
                println!(
                    "抽取统计：{} 句，{} 条帧，解析失败 {} 句；闸门事实 {} 条。",
                    stats.sentences,
                    stats.frames,
                    stats.parse_errors,
                    gate.fact_count()
                );
                for sample in &diagnostics.samples {
                    println!("\n{}", sample.frame.render_text());
                    println!("  证据：{}", sample.frame.evidence.sentence);
                    println!("  {}", narrate_frame(&sample.frame, &sample.report));
                }
            }
        }
        Some(Commands::BuildKg {
            text,
            file,
            min_support,
            min_purity,
            limit,
            feedback,
            apply,
            json,
        }) => {
            let registry_path = format!("{}/predicate_registry.json", store_path);
            let _ = oxigraph_manager.load_predicate_registry(&registry_path);
            oxigraph_manager.configure_predicates(
                &config.predicate_profiles,
                config.reasoning.require_predicate_admission,
            );
            oxigraph_manager.configure_type_system(&config.type_system);
            let registry = oxigraph_manager.predicate_registry_snapshot();
            let types = oxigraph_manager.type_system_snapshot();
            let mut laws = LawBook::from_registry(&registry);
            if let Some(persisted) =
                LawBook::load_from_path(&format!("{}/law_book.json", store_path))
            {
                laws.apply_persisted(&persisted);
            }
            let facts = match oxigraph_manager.load_graph_snapshot().await {
                Ok(facts) => facts,
                Err(e) => {
                    eprintln!("读取知识库失败: {}", e);
                    std::process::exit(1);
                }
            };
            let source = match (text, file) {
                (Some(text), _) => text,
                (None, Some(path)) => match std::fs::read_to_string(&path) {
                    Ok(content) => content,
                    Err(e) => {
                        eprintln!("读取文本失败（{}）: {}", path, e);
                        std::process::exit(1);
                    }
                },
                (None, None) => match load_frame_corpus() {
                    Ok(content) => content,
                    Err(e) => {
                        eprintln!("{}", e);
                        std::process::exit(1);
                    }
                },
            };
            let options = ConstructionOptions {
                min_pattern_support: min_support,
                min_pattern_purity: min_purity.min(1000),
                max_candidates: limit,
                ..ConstructionOptions::default()
            };
            let feedback_items = match feedback {
                Some(path) => match std::fs::read_to_string(&path) {
                    Ok(text) => {
                        let (items, errors) = parse_feedback(&text);
                        for error in errors {
                            eprintln!("反馈文件 {}：{}", path, error);
                        }
                        items
                    }
                    Err(e) => {
                        eprintln!("读取反馈文件失败（{}）: {}", path, e);
                        std::process::exit(1);
                    }
                },
                None => Vec::new(),
            };
            let constructor = WeakConstructor::new(facts.as_slice(), registry, types, laws);
            let report = constructor.build_with_feedback(&source, &options, &feedback_items);
            if apply {
                let items: Vec<KnowledgeItem> = report
                    .candidates
                    .iter()
                    .filter(|candidate| {
                        candidate.decision
                            == know_prolog::weak_construction::CandidateDecision::Accepted
                            && candidate.known_subject
                            && candidate.known_object
                    })
                    .map(|candidate| KnowledgeItem::Relationship {
                        from_label: String::new(),
                        from_id: candidate.subject.clone(),
                        to_label: String::new(),
                        to_id: candidate.object.clone(),
                        rel_type: candidate.predicate.clone(),
                        properties: std::collections::HashMap::new(),
                    })
                    .collect();
                if items.is_empty() {
                    eprintln!("没有可落库的已接受候选（端点必须已在图谱中）。");
                } else {
                    match oxigraph_manager.save_knowledge_with_receipt(items).await {
                        Ok(receipt) => eprintln!("{}", receipt.summary()),
                        Err(e) => {
                            eprintln!("写入知识库失败: {}", e);
                            std::process::exit(1);
                        }
                    }
                }
            }
            if json {
                match serde_json::to_string_pretty(&report) {
                    Ok(text) => println!("{}", text),
                    Err(e) => {
                        eprintln!("构建报告序列化失败: {}", e);
                        std::process::exit(1);
                    }
                }
            } else {
                println!("{}", report.render_text());
            }
        }
        Some(Commands::Paths {
            from,
            to,
            via,
            common_with,
            max_hops,
            max_paths,
            as_of,
            json,
        }) => {
            let facts = match oxigraph_manager.load_graph_snapshot().await {
                Ok(facts) => facts,
                Err(e) => {
                    eprintln!("读取知识库失败: {}", e);
                    std::process::exit(1);
                }
            };
            let temporal_constraint = match as_of.as_deref() {
                Some(text) => match temporal::parse_time_span(text) {
                    Some(span) => TemporalConstraint::from_span(span),
                    None => {
                        eprintln!("无法解析时间「{}」（示例：1995、1995-06、1995年）", text);
                        std::process::exit(2);
                    }
                },
                None => TemporalConstraint::Any,
            };
            let mut validity_by_entity: std::collections::BTreeMap<String, Validity> =
                std::collections::BTreeMap::new();
            for fact in facts.iter() {
                if let PrologFact::Entity {
                    name, properties, ..
                } = fact
                {
                    if let Some(validity) = temporal::validity_from_properties(properties) {
                        validity_by_entity.insert(name.clone(), validity);
                        if let Some(alias) = properties.get("name") {
                            validity_by_entity.insert(alias.clone(), validity);
                        }
                    }
                }
            }
            let mut graph = TraversalGraph::new();
            for fact in facts.iter() {
                match fact {
                    PrologFact::Relation {
                        from: subject,
                        rel_type,
                        to: object,
                    } => {
                        let validity = validity_by_entity
                            .get(subject.as_str())
                            .copied()
                            .unwrap_or(Validity::Unknown);
                        graph.add_edge_record(
                            PathEdge::new(subject.as_str(), rel_type.as_str(), object.as_str())
                                .with_validity(validity),
                        );
                    }
                    PrologFact::Entity {
                        name,
                        entity_type,
                        properties,
                    } => {
                        graph.add_entity_type(name.as_str(), entity_type.as_str());
                        for (key, value) in properties {
                            if temporal::is_temporal_field(key) {
                                continue;
                            }
                            graph.add_edge_with_source(
                                name.as_str(),
                                key.as_str(),
                                value.as_str(),
                                EdgeSource::Attribute,
                            );
                        }
                    }
                }
            }
            let options = TraversalOptions {
                max_depth: max_hops.clamp(1, 8),
                max_paths: max_paths.clamp(1, 64),
                temporal: temporal_constraint,
                unknown_time: if as_of.is_some() {
                    UnknownTimePolicy::Background
                } else {
                    UnknownTimePolicy::Strict
                },
                ..TraversalOptions::default()
            };
            let bundle = if let Some(right) = common_with.as_deref() {
                graph.common_neighbors(&from, right, &options)
            } else if !via.is_empty() {
                let chain: Vec<PathPredicate> = via
                    .iter()
                    .map(|predicate| PathPredicate::forward(predicate.trim()))
                    .collect();
                let chain_options = TraversalOptions {
                    max_depth: chain.len().clamp(1, 8),
                    ..options.clone()
                };
                graph.enumerate_chain(&from, &chain, &chain_options)
            } else if let Some(goal) = to.as_deref() {
                graph.enumerate_between(&from, goal, &options)
            } else {
                eprintln!("请提供 --to、--via 或 --common-with 之一。");
                std::process::exit(2);
            };
            if json {
                match serde_json::to_string_pretty(&bundle) {
                    Ok(text) => println!("{}", text),
                    Err(e) => {
                        eprintln!("路径结果序列化失败: {}", e);
                        std::process::exit(1);
                    }
                }
            } else {
                println!("{}", bundle.render_text());
            }
        }
        Some(Commands::Evidence {
            query,
            k,
            subject,
            predicate,
            object,
            entity_type,
            max_scanned,
            json,
        }) => {
            let facts = match oxigraph_manager.load_graph_snapshot().await {
                Ok(facts) => facts,
                Err(e) => {
                    eprintln!("读取知识库失败: {}", e);
                    std::process::exit(1);
                }
            };
            let mut retriever = DenseRetriever::new(k, 0.85, 12);
            retriever.observe_facts(&facts);
            let constraints = SearchConstraints {
                subject: subject.map(|value| value.trim().to_string()),
                predicate: predicate.map(|value| value.trim().to_string()),
                object: object.map(|value| value.trim().to_string()),
                entity_type: entity_type.map(|value| value.trim().to_string()),
            };
            let search = retriever.search(&query, k, &constraints, SearchBudget { max_scanned });
            if json {
                let payload = serde_json::json!({
                    "query": query,
                    "constraints": constraints,
                    "hits": search.hits,
                    "receipt": search.receipt,
                });
                match serde_json::to_string_pretty(&payload) {
                    Ok(text) => println!("{}", text),
                    Err(e) => {
                        eprintln!("证据检索结果序列化失败: {}", e);
                        std::process::exit(1);
                    }
                }
            } else {
                println!("{}", retriever.format_hits(&query, &search.hits));
                println!();
                println!("{}", search.receipt.render_text());
            }
        }
        Some(Commands::RewriteRetrieval {
            question,
            book,
            json,
        }) => {
            let facts = match oxigraph_manager.load_graph_snapshot().await {
                Ok(facts) => facts,
                Err(error) => {
                    eprintln!("读取知识库失败: {}", error);
                    std::process::exit(1);
                }
            };
            let mut reasoner =
                build_store_reasoner(config.as_ref(), &oxigraph_manager, &store_path, &facts).await;
            if let Some(path) = book {
                match RewriteBook::load_from_path(path) {
                    Ok(loaded) => reasoner.set_rewrite_book(loaded),
                    Err(error) => {
                        eprintln!("读取重写手册失败: {}", error);
                        std::process::exit(1);
                    }
                }
            }
            match reasoner.rewrite_retrieval_assessment(&question) {
                Some(assessment) => {
                    if json {
                        match serde_json::to_string_pretty(&assessment) {
                            Ok(text) => println!("{}", text),
                            Err(error) => {
                                eprintln!("读法裁决序列化失败: {}", error);
                                std::process::exit(1);
                            }
                        }
                    } else {
                        println!("{}", assessment.render_report());
                    }
                }
                None => println!("没有可编译的候选读法：{}", question.trim()),
            }
        }
        Some(Commands::RewriteRetrievalEval {
            corpus,
            book,
            strict,
            json,
        }) => {
            let text = match std::fs::read_to_string(&corpus) {
                Ok(text) => text,
                Err(error) => {
                    eprintln!("读取评测语料失败: {}", error);
                    std::process::exit(1);
                }
            };
            let cases = rewrite_retrieval::parse_corpus(&text);
            if cases.is_empty() {
                eprintln!(
                    "评测语料为空：每行应为 `问句<TAB>期望答案`（期望答案可用 | 分隔多个候选）。"
                );
                std::process::exit(1);
            }
            let facts = match oxigraph_manager.load_graph_snapshot().await {
                Ok(facts) => facts,
                Err(error) => {
                    eprintln!("读取知识库失败: {}", error);
                    std::process::exit(1);
                }
            };
            let mut reasoner =
                build_store_reasoner(config.as_ref(), &oxigraph_manager, &store_path, &facts).await;
            if let Some(path) = book {
                match RewriteBook::load_from_path(path) {
                    Ok(loaded) => reasoner.set_rewrite_book(loaded),
                    Err(error) => {
                        eprintln!("读取重写手册失败: {}", error);
                        std::process::exit(1);
                    }
                }
            }
            let report = rewrite_retrieval::evaluate_with(
                &cases,
                &config.reasoning.rewrite_retrieval,
                |question| reasoner.rewrite_retrieval_candidates(question),
                |query, k, constraints| {
                    reasoner.evidence_search(query, k, constraints, SearchBudget::default())
                },
                |hypothesis| reasoner.answer_hypothesis(&hypothesis.candidate),
            );
            if json {
                match serde_json::to_string_pretty(&report) {
                    Ok(text) => println!("{}", text),
                    Err(error) => {
                        eprintln!("评测报告序列化失败: {}", error);
                        std::process::exit(1);
                    }
                }
            } else {
                println!("{}", report.render_text());
            }
            if strict && report.accepted_wrong > 0 {
                std::process::exit(1);
            }
        }
        Some(Commands::Snapshot {
            subject,
            predicate,
            all,
            limit,
            json,
        }) => {
            let registry_path = format!("{}/predicate_registry.json", store_path);
            oxigraph_manager.load_predicate_registry(&registry_path);
            oxigraph_manager.configure_predicates(
                &config.predicate_profiles,
                config.reasoning.require_predicate_admission,
            );
            let snapshot = match oxigraph_manager.load_fact_snapshot().await {
                Ok(snapshot) => snapshot,
                Err(error) => {
                    eprintln!("读取事实快照失败: {}", error);
                    std::process::exit(1);
                }
            };
            if let (Some(subject), Some(predicate)) = (&subject, &predicate) {
                let view = snapshot.lookup(subject.as_str(), predicate.as_str());
                if json {
                    let payload = serde_json::json!({
                        "revision": snapshot.revision(),
                        "functional": snapshot.is_functional(predicate.as_str()),
                        "slot": view,
                    });
                    match serde_json::to_string_pretty(&payload) {
                        Ok(text) => println!("{}", text),
                        Err(error) => {
                            eprintln!("快照序列化失败: {}", error);
                            std::process::exit(1);
                        }
                    }
                } else {
                    match view {
                        Some(view) => {
                            println!(
                                "槽位 {}({}, ?)：{} 条记录，{} 个有效取值{}",
                                predicate,
                                subject,
                                view.all.len(),
                                view.effective.len(),
                                if view.conflict.is_some() {
                                    "（存在冲突）"
                                } else {
                                    ""
                                }
                            );
                            let rows = if all { &view.all } else { &view.effective };
                            for record in rows {
                                println!("  {}", record.display());
                            }
                            if let Some(conflict) = &view.conflict {
                                println!("  {}", conflict.render());
                            }
                        }
                        None => println!("槽位 {}({}, ?) 没有事实。", predicate, subject),
                    }
                }
            } else if let Some(subject) = &subject {
                let records = snapshot.subject_records(subject);
                if json {
                    let payload = serde_json::json!({
                        "revision": snapshot.revision(),
                        "subject": subject,
                        "record_count": records.len(),
                        "records": records,
                    });
                    match serde_json::to_string_pretty(&payload) {
                        Ok(text) => println!("{}", text),
                        Err(error) => {
                            eprintln!("快照序列化失败: {}", error);
                            std::process::exit(1);
                        }
                    }
                } else if records.is_empty() {
                    println!("实体「{}」没有事实。", subject);
                } else {
                    println!("实体「{}」：{} 条记录", subject, records.len());
                    for record in records.iter().take(limit) {
                        println!("  {}", record.display());
                    }
                    if records.len() > limit {
                        println!("  …另有 {} 条（--limit 调整）", records.len() - limit);
                    }
                }
            } else if json {
                let payload = serde_json::json!({
                    "version": snapshot.version,
                    "revision": snapshot.revision(),
                    "grammar_digest": snapshot.grammar_digest(),
                    "functional": snapshot.functional,
                    "conflicts": snapshot.conflicts(),
                    "records": snapshot.effective_records().into_iter().take(limit).collect::<Vec<_>>(),
                    "record_count": snapshot.record_count(),
                });
                match serde_json::to_string_pretty(&payload) {
                    Ok(text) => println!("{}", text),
                    Err(error) => {
                        eprintln!("快照序列化失败: {}", error);
                        std::process::exit(1);
                    }
                }
            } else {
                println!("{}", snapshot.render());
                println!();
                for record in snapshot.effective_records().into_iter().take(limit) {
                    println!("- {}", record.display());
                }
                if snapshot.effective_records().len() > limit {
                    println!("（仅显示前 {} 条；用 --limit 调整）", limit);
                }
            }
        }
        Some(Commands::Conflicts {
            provenance,
            latent,
            namespaces,
            limit,
            strict,
            json,
        }) => {
            let registry_path = format!("{}/predicate_registry.json", store_path);
            oxigraph_manager.load_predicate_registry(&registry_path);
            oxigraph_manager.configure_predicates(
                &config.predicate_profiles,
                config.reasoning.require_predicate_admission,
            );
            oxigraph_manager.configure_type_system(&config.type_system);
            let mut facts = match oxigraph_manager.load_fact_refs().await {
                Ok(facts) => facts,
                Err(error) => {
                    eprintln!("读取知识库失败: {}", error);
                    std::process::exit(1);
                }
            };
            let registry = oxigraph_manager.predicate_registry_snapshot();
            let types = oxigraph_manager.type_system_snapshot();
            let mut schema = ConflictSchema::from_registry_and_types(&registry, &types);
            for fact in &facts {
                if schema.type_predicates.contains(&fact.predicate) {
                    schema.observe_type(&types, &fact.value);
                }
            }

            let mut ledger = ProvenanceLedger::new();
            // 跨会话导入日志：给文本导入的事实补上「第几笔 + 内容摘要」身份。
            let journal_path = format!("{}/import_journal.json", store_path);
            if let JournalLoad::Loaded(journal) = ImportJournal::load(&journal_path) {
                ledger.seed_from_import_journal(&journal);
            }
            if namespaces {
                match oxigraph_manager.load_fusion_ledger().await {
                    Ok(fusion) => {
                        ledger.seed_from_fusion_ledger(&fusion);
                        for assertion in &fusion.assertions {
                            facts.push(provenance::FactRef::new(
                                &assertion.subject,
                                &assertion.predicate,
                                &assertion.object,
                            ));
                        }
                        facts.sort();
                        facts.dedup();
                        for fact in &facts {
                            if schema.type_predicates.contains(&fact.predicate) {
                                schema.observe_type(&types, &fact.value);
                            }
                        }
                    }
                    Err(error) => eprintln!("读取融合账本失败：{}", error),
                }
            }
            let default_ledger_path = format!("{}/provenance.json", store_path);
            if let ProvenanceLoad::Loaded(loaded) = ProvenanceLedger::load(&default_ledger_path) {
                ledger.merge(&loaded);
            }
            if let Some(path) = provenance.as_deref() {
                if path != default_ledger_path {
                    match ProvenanceLedger::load(path) {
                        ProvenanceLoad::Loaded(loaded) => ledger.merge(&loaded),
                        ProvenanceLoad::Missing => {
                            eprintln!("来源台账不存在：{}（按已回填来源报告）", path)
                        }
                        ProvenanceLoad::Damaged(error) => {
                            eprintln!("来源台账损坏：{}（按已回填来源报告）", error)
                        }
                    }
                }
            }
            let rules = oxigraph_manager
                .load_rule_ledger()
                .await
                .unwrap_or_else(|error| {
                    eprintln!("读取规则账本失败：{}", error);
                    know_prolog::graph_ledger::RuleLedger {
                        synced: false,
                        fingerprint: String::new(),
                        entries: Vec::new(),
                    }
                });
            let derived = oxigraph_manager
                .load_derived_ledger()
                .await
                .unwrap_or_else(|error| {
                    eprintln!("读取派生账本失败：{}", error);
                    know_prolog::graph_ledger::DerivedLedger {
                        materialized: false,
                        revision: 0,
                        fingerprint: String::new(),
                        entries: Vec::new(),
                    }
                });
            let supports = provenance::build_supports(&facts, &ledger, &rules, &derived);
            // 派生事实通常只在命名图里：把它们也纳入已见证冲突的候选结论。
            facts.extend(supports.values().map(|support| support.fact.clone()));
            facts.sort();
            facts.dedup();
            let mut certificates = provenance::detect_witnessed(&facts, &supports, &schema);
            if latent {
                let clauses: Vec<know_prolog::resolution_engine::Clause> =
                    rules.entries.iter().map(ledger_entry_to_clause).collect();
                certificates.extend(provenance::detect_latent(&clauses, &schema));
            }
            let coverage = provenance::Coverage::from_supports(&facts, &supports);
            let report = ConflictReport::new(certificates, coverage);
            if json {
                println!("{}", report.to_json());
            } else {
                let mut shown = report.clone();
                if limit > 0 && shown.certificates.len() > limit {
                    shown.certificates.truncate(limit);
                }
                println!("{}", shown.render_text());
                if shown.certificates.len() < report.certificates.len() {
                    println!(
                        "（仅显示前 {} 个证书；用 --limit 0 查看全部）",
                        shown.certificates.len()
                    );
                }
            }
            if strict && !report.certificates.is_empty() {
                std::process::exit(1);
            }
        }
        Some(Commands::Provenance {
            record,
            fact,
            path,
            namespaces,
            json,
        }) => {
            let ledger_path = path
                .clone()
                .unwrap_or_else(|| format!("{}/provenance.json", store_path));
            let mut ledger = ProvenanceLedger::new();
            match ProvenanceLedger::load(&ledger_path) {
                ProvenanceLoad::Loaded(loaded) => ledger = loaded,
                ProvenanceLoad::Missing => {}
                ProvenanceLoad::Damaged(error) => {
                    eprintln!("来源台账损坏：{}", error);
                    if record.is_some() {
                        eprintln!("拒绝在损坏台账上追加记录；请先修复或另用 --path。");
                        std::process::exit(1);
                    }
                }
            }
            if let Some(file) = record.as_deref() {
                let text = match std::fs::read_to_string(file) {
                    Ok(text) => text,
                    Err(error) => {
                        eprintln!("读取来源记录失败：{}", error);
                        std::process::exit(1);
                    }
                };
                let rows = match parse_provenance_tsv(&text) {
                    Ok(rows) => rows,
                    Err(error) => {
                        eprintln!("来源记录格式错误：{}", error);
                        std::process::exit(2);
                    }
                };
                let mut added = 0usize;
                for (fact, source) in rows {
                    if ledger.record(fact, source) {
                        added += 1;
                    }
                }
                if let Err(error) = ledger.save(&ledger_path) {
                    eprintln!("来源台账写入失败：{}", error);
                    std::process::exit(1);
                }
                println!(
                    "来源台账已登记 {} 条新来源（{}，共 {} 条事实有登记）。",
                    added,
                    ledger_path,
                    ledger.len()
                );
            }

            // 只读视图：台账 + 导入日志 + 可选命名空间（不回写磁盘）。
            let mut view = ledger.clone();
            let journal_path = format!("{}/import_journal.json", store_path);
            if let JournalLoad::Loaded(journal) = ImportJournal::load(&journal_path) {
                view.seed_from_import_journal(&journal);
            }
            let mut facts = match oxigraph_manager.load_fact_refs().await {
                Ok(facts) => facts,
                Err(error) => {
                    eprintln!("读取知识库失败：{}", error);
                    std::process::exit(1);
                }
            };
            if namespaces {
                match oxigraph_manager.load_fusion_ledger().await {
                    Ok(fusion) => {
                        view.seed_from_fusion_ledger(&fusion);
                        for assertion in &fusion.assertions {
                            facts.push(provenance::FactRef::new(
                                &assertion.subject,
                                &assertion.predicate,
                                &assertion.object,
                            ));
                        }
                        facts.sort();
                        facts.dedup();
                    }
                    Err(error) => eprintln!("读取融合账本失败：{}", error),
                }
            }
            let coverage = view.coverage(&facts);
            if let Some(selector) = fact.as_deref() {
                let parts: Vec<&str> = selector.split('|').collect();
                if parts.is_empty() || parts.len() > 3 || parts[0].trim().is_empty() {
                    eprintln!("--fact 格式应为 subject|predicate|value（value 可省略）");
                    std::process::exit(2);
                }
                let matched: Vec<&provenance::FactRef> = facts
                    .iter()
                    .filter(|candidate| {
                        candidate.subject == parts[0]
                            && (parts.len() < 2 || candidate.predicate == parts[1])
                            && (parts.len() < 3 || candidate.value == parts[2])
                    })
                    .collect();
                if json {
                    let rows: Vec<serde_json::Value> = matched
                        .iter()
                        .map(|candidate| {
                            let sources: Vec<String> = view
                                .sources(candidate)
                                .iter()
                                .map(|source| source.render())
                                .collect();
                            serde_json::json!({
                                "fact": candidate,
                                "display": candidate.display(),
                                "sources": sources,
                            })
                        })
                        .collect();
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&rows).unwrap_or_else(|_| "[]".into())
                    );
                } else if matched.is_empty() {
                    println!("没有匹配的事实：{}", selector);
                } else {
                    for candidate in matched {
                        let sources = view.sources(candidate);
                        let rendered: Vec<String> =
                            sources.iter().map(|source| source.render()).collect();
                        println!(
                            "{}：{}",
                            candidate.display(),
                            if rendered.is_empty() {
                                "来源未登记".to_string()
                            } else {
                                rendered.join("、")
                            }
                        );
                    }
                }
            } else if json {
                let value = serde_json::json!({
                    "ledger_path": ledger_path,
                    "ledger_entries": ledger.len(),
                    "view_entries": view.len(),
                    "digest": ledger.digest(),
                    "coverage": coverage,
                });
                println!(
                    "{}",
                    serde_json::to_string_pretty(&value).unwrap_or_else(|_| "{}".into())
                );
            } else {
                println!(
                    "来源台账：{} 条事实有登记（{}）；导入日志/命名空间回填只用于本次查看，不写盘。",
                    view.len(),
                    ledger_path
                );
                println!("{}", coverage.render_zh());
            }
        }
        Some(Commands::Temporal {
            entity,
            at,
            functional,
            json,
        }) => {
            let facts = match oxigraph_manager.load_graph_snapshot().await {
                Ok(facts) => facts,
                Err(e) => {
                    eprintln!("读取知识库失败: {}", e);
                    std::process::exit(1);
                }
            };
            let index = TemporalIndex::from_prolog_facts(&facts);
            let at_span = match at.as_deref() {
                Some(text) => match temporal::parse_time_span(text) {
                    Some(span) => Some(span),
                    None => {
                        eprintln!("无法解析时间「{}」（示例：1868、1868-06、1868年）", text);
                        std::process::exit(2);
                    }
                },
                None => None,
            };
            let at_facts: Vec<_> = at_span
                .as_ref()
                .map(|span| index.overlapping(span))
                .unwrap_or_default();
            let entity_facts = entity
                .as_deref()
                .map(|name| index.for_entity(name))
                .unwrap_or_default();
            let functional: std::collections::BTreeSet<String> = if functional.is_empty() {
                ["capital", "spouse", "president", "首都", "配偶", "总统"]
                    .iter()
                    .map(|name| name.to_string())
                    .collect()
            } else {
                functional.into_iter().collect()
            };
            let conflicts = index.functional_conflicts(&functional);
            if json {
                let payload = serde_json::json!({
                    "dialect": know_prolog::temporal::TEMPORAL_DIALECT,
                    "facts": index.len(),
                    "nodes": index.nodes().len(),
                    "unknown_time": index.unknown_time_facts().len(),
                    "entity": entity,
                    "entity_facts": entity_facts,
                    "at": at,
                    "at_facts": at_facts,
                    "functional_predicates": functional,
                    "conflicts": conflicts,
                });
                match serde_json::to_string_pretty(&payload) {
                    Ok(text) => println!("{}", text),
                    Err(e) => {
                        eprintln!("时序索引序列化失败: {}", e);
                        std::process::exit(1);
                    }
                }
            } else {
                println!(
                    "时序索引：{} 条关系，{} 个实体，{} 条时间未知。",
                    index.len(),
                    index.nodes().len(),
                    index.unknown_time_facts().len()
                );
                for fact in index.timeline() {
                    println!("  {}", fact.render());
                }
                if let Some(name) = entity.as_deref() {
                    println!("\n实体「{}」：{} 条相关事实。", name, entity_facts.len());
                    for fact in &entity_facts {
                        println!("  {}", fact.render());
                    }
                }
                if let Some(text) = at.as_deref() {
                    println!("\n时间「{}」内成立：{} 条。", text, at_facts.len());
                    for fact in &at_facts {
                        println!("  {}", fact.render());
                    }
                }
                if conflicts.is_empty() {
                    println!("\n时序单值冲突：无。");
                } else {
                    println!("\n时序单值冲突：{} 处。", conflicts.len());
                    for conflict in &conflicts {
                        println!("  {}", conflict.render());
                    }
                }
            }
        }
        Some(Commands::Laws {
            json,
            save,
            adopt,
            retire,
        }) => {
            let registry_path = format!("{}/predicate_registry.json", store_path);
            let _ = oxigraph_manager.load_predicate_registry(&registry_path);
            oxigraph_manager.configure_predicates(
                &config.predicate_profiles,
                config.reasoning.require_predicate_admission,
            );
            if let Err(e) = oxigraph_manager.refresh_predicate_registry().await {
                eprintln!("刷新谓词画像失败: {}", e);
                std::process::exit(1);
            }
            let relations = match oxigraph_manager.load_all_relations().await {
                Ok(relations) => relations,
                Err(e) => {
                    eprintln!("读取知识库失败: {}", e);
                    std::process::exit(1);
                }
            };
            let triples: Vec<LawTriple> = relations
                .iter()
                .filter_map(|fact| match fact {
                    PrologFact::Relation { from, rel_type, to } => {
                        Some(LawTriple::new(from, rel_type, to))
                    }
                    _ => None,
                })
                .collect();
            let mut predictor = LinkPredictor::new();
            for triple in &triples {
                predictor.add_triple(&triple.subject, &triple.predicate, &triple.object);
            }
            predictor.mine_rules(
                config.reasoning.link_prediction_min_support,
                config.reasoning.link_prediction_min_confidence,
            );
            let registry = oxigraph_manager.predicate_registry_snapshot();
            let path = format!("{}/law_book.json", store_path);
            let mut book = LawBook::from_registry(&registry);
            if let Some(persisted) = LawBook::load_from_path(&path) {
                let changed = book.apply_persisted(&persisted);
                if changed > 0 && !json {
                    eprintln!("载入持久化律典：{} 条显式决定生效", changed);
                }
            }
            book.add_mined_rules(predictor.rules());

            let mut unknown: Vec<&str> = Vec::new();
            for id in adopt.iter().chain(retire.iter()) {
                if book.law(id).is_none() {
                    unknown.push(id);
                }
            }
            if !unknown.is_empty() {
                eprintln!("未知定律 id：{}", unknown.join(", "));
                std::process::exit(1);
            }
            let adopted = adopt.iter().filter(|id| book.adopt(id)).count();
            let retired = retire.iter().filter(|id| book.retire(id)).count();
            if (adopted > 0 || retired > 0) && !save {
                eprintln!("提示：未加 --save，采纳/退役决定仅在本次运行生效");
            }
            let audit = book.audit(&triples);
            if save {
                match book.save_to_path(&path) {
                    Ok(()) => eprintln!(
                        "律典已写入 {}（采纳 {} 条，退役 {} 条）",
                        path, adopted, retired
                    ),
                    Err(e) => {
                        eprintln!("律典写入失败: {}", e);
                        std::process::exit(1);
                    }
                }
            }
            if json {
                let payload = serde_json::json!({ "book": &book, "audit": &audit });
                match serde_json::to_string_pretty(&payload) {
                    Ok(text) => println!("{}", text),
                    Err(e) => {
                        eprintln!("律典序列化失败: {}", e);
                        std::process::exit(1);
                    }
                }
            } else {
                print!("{}{}", book.render_text(), audit.render_text());
            }
        }
        Some(Commands::Eval {
            test_ratio,
            baseline,
            update_baseline,
            allow_regression,
            rrf_k,
        }) => match oxigraph_manager.load_all_facts().await {
            Ok(facts) => {
                let triples = triples_from_prolog_facts(&facts);
                let entity_types = entity_types_from_prolog_facts(&facts);
                if triples.len() < 2 {
                    eprintln!(
                        "知识库中的关系三元组不足（{} 条），无法评估。可先导入 JSON-LD 知识库。",
                        triples.len()
                    );
                    std::process::exit(1);
                }
                let eval_config = EvalConfig {
                    test_ratio_per_mille: test_ratio,
                    rrf_k: rrf_k.max(1.0),
                    schema: Some(SchemaView::from_config(
                        &config.type_system,
                        &config.predicate_profiles,
                    )),
                    entity_types,
                    ..Default::default()
                };
                let report = evaluate(&triples, &eval_config);
                println!("{}", report.render_text(&eval_config));

                let baseline_path = baseline
                    .clone()
                    .unwrap_or_else(|| "eval_baseline.json".to_string());
                if update_baseline {
                    let new_baseline = EvalBaseline::from_report(&report);
                    match new_baseline.save(&baseline_path) {
                        Ok(()) => println!("基线已写入 {}", baseline_path),
                        Err(e) => {
                            eprintln!("基线写入失败: {}", e);
                            std::process::exit(1);
                        }
                    }
                } else if baseline.is_some() {
                    match EvalBaseline::load(&baseline_path) {
                        Ok(old_baseline) => {
                            let verdict = compare_baseline(&old_baseline, &report, 1e-6);
                            print!("{}", verdict.render_text());
                            if !verdict.passed && !allow_regression {
                                std::process::exit(1);
                            }
                        }
                        Err(e) => {
                            eprintln!("基线读取失败: {}", e);
                            std::process::exit(1);
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("读取知识库失败: {}", e);
                std::process::exit(1);
            }
        },
        // 已在打开存储前处理（见 main 顶部早期分派，无需知识存储）
        Some(Commands::EvalNlu { .. }) => {}
        Some(Commands::EvalLink { .. }) => {}
        Some(Commands::EvalGrounding { .. }) => {}
        Some(Commands::EvalConstruction { .. }) => {}
        Some(Commands::EvalIdentity { .. }) => {}
        Some(Commands::EvalRecovery { .. }) => {}
        Some(Commands::ProofBench { .. }) => {}
        Some(Commands::ReleaseCheck { .. }) => {}
        // 已在打开存储前处理（见 main 顶部早期分派）
        Some(Commands::PromptLint { .. }) => {}
        // 已在打开存储前处理（见 main 顶部早期分派，无需知识存储）
        Some(Commands::Events { .. }) => {}
        Some(Commands::ReadTrace { .. }) => {}
        Some(Commands::Grammar { .. }) => {}
        Some(Commands::Frontier { .. }) => {}
        Some(Commands::EvalRecall { .. }) => {}
        Some(Commands::Forge { .. }) => {}
        // 已在加载配置前处理（见 main 顶部早期分派）
        Some(Commands::RewriteMine { .. }) | Some(Commands::RewriteRules { .. }) => {}
        Some(Commands::BatchTeach { .. }) => {}
        // 已在加载配置前处理（见 main 顶部早期分派）
        Some(Commands::ConfigLint { .. }) => {}
        // 已在加载默认配置前处理（见 main 顶部早期分派）
        Some(Commands::ConfigFulfillment { .. }) => {}
        // 已在加载配置前处理（见 main 顶部早期分派）
        Some(Commands::Roles { .. }) => {}
        // 已在加载配置前处理（见 main 顶部早期分派）
        Some(Commands::OpenClosure { .. }) => {}
        // 已在加载配置前处理（见 main 顶部早期分派）
        Some(Commands::Privacy { .. }) => {}
        Some(Commands::EvalQuality {
            baseline,
            update_baseline,
            allow_regression,
            provenance,
            json,
        }) => {
            let registry_path = format!("{}/predicate_registry.json", store_path);
            oxigraph_manager.load_predicate_registry(&registry_path);
            oxigraph_manager.configure_predicates(
                &config.predicate_profiles,
                config.reasoning.require_predicate_admission,
            );
            oxigraph_manager.configure_type_system(&config.type_system);
            let ledger =
                provenance
                    .as_deref()
                    .and_then(|path| match ProvenanceLedger::load(path) {
                        ProvenanceLoad::Loaded(ledger) => Some(ledger),
                        ProvenanceLoad::Missing => {
                            eprintln!("来源台账不存在：{}（按未登记来源计分）", path);
                            None
                        }
                        ProvenanceLoad::Damaged(error) => {
                            eprintln!("来源台账损坏：{}（按未登记来源计分）", error);
                            None
                        }
                    });
            let report = match quality_report_for_with_provenance(
                &oxigraph_manager,
                &config,
                ledger.as_ref(),
                Some(&store_path),
            )
            .await
            {
                Ok(report) => report,
                Err(e) => {
                    eprintln!("读取知识库失败: {}", e);
                    std::process::exit(1);
                }
            };
            if json {
                match serde_json::to_string_pretty(&report) {
                    Ok(text) => println!("{}", text),
                    Err(e) => {
                        eprintln!("质量报告序列化失败: {}", e);
                        std::process::exit(1);
                    }
                }
            } else {
                println!("{}", report.render_text());
            }

            let baseline_path = baseline
                .clone()
                .unwrap_or_else(|| "quality_baseline.json".to_string());
            if update_baseline {
                let new_baseline = QualityBaseline::from_report(&report);
                match new_baseline.save(&baseline_path) {
                    Ok(()) => println!("质量基线已写入 {}", baseline_path),
                    Err(e) => {
                        eprintln!("基线写入失败: {}", e);
                        std::process::exit(1);
                    }
                }
            } else if baseline.is_some() {
                match QualityBaseline::load(&baseline_path) {
                    Ok(old_baseline) => {
                        let verdict = compare_quality_baseline(&old_baseline, &report, 1e-6);
                        print!("{}", verdict.render_text());
                        if !verdict.passed && !allow_regression {
                            std::process::exit(1);
                        }
                    }
                    Err(e) => {
                        eprintln!("基线读取失败: {}", e);
                        std::process::exit(1);
                    }
                }
            }
        }
        Some(Commands::EvalShapes { json, strict }) => {
            let registry_path = format!("{}/predicate_registry.json", store_path);
            oxigraph_manager.load_predicate_registry(&registry_path);
            oxigraph_manager.configure_predicates(
                &config.predicate_profiles,
                config.reasoning.require_predicate_admission,
            );
            oxigraph_manager.configure_type_system(&config.type_system);
            match shape_report_for(&oxigraph_manager, &config).await {
                Ok((_, report)) => {
                    if json {
                        match serde_json::to_string_pretty(&report) {
                            Ok(text) => println!("{}", text),
                            Err(e) => {
                                eprintln!("形状报告序列化失败: {}", e);
                                std::process::exit(1);
                            }
                        }
                    } else {
                        println!("{}", report.render_text());
                    }
                    if strict && !report.conforms {
                        std::process::exit(1);
                    }
                }
                Err(e) => {
                    eprintln!("读取知识库失败: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Some(Commands::EvalShapeDialogue {
            baseline,
            update_baseline,
            allow_regression,
            json,
        }) => {
            let report = evaluate_shape_dialogue();
            if json {
                match serde_json::to_string_pretty(&report) {
                    Ok(text) => println!("{}", text),
                    Err(error) => {
                        eprintln!("形状对话评测序列化失败: {}", error);
                        std::process::exit(1);
                    }
                }
            } else {
                println!("{}", report.render_text());
            }
            if update_baseline {
                let path = baseline
                    .clone()
                    .unwrap_or_else(|| "shape_dialogue_baseline.json".to_string());
                let baseline_value = ShapeEvalBaseline::from_report(&report);
                match baseline_value.save(&path) {
                    Ok(_) => println!("评测基线已写入 {}", path),
                    Err(error) => {
                        eprintln!("评测基线写入失败: {}", error);
                        std::process::exit(1);
                    }
                }
            } else if let Some(path) = baseline.as_deref() {
                match ShapeEvalBaseline::load(path) {
                    Ok(old_baseline) => {
                        let verdict = compare_shape_eval_baseline(&old_baseline, &report, 1e-9);
                        println!("{}", verdict.render_text());
                        if !verdict.passed && !allow_regression {
                            std::process::exit(1);
                        }
                    }
                    Err(error) => {
                        eprintln!("基线读取失败: {}", error);
                        std::process::exit(1);
                    }
                }
            }
        }
        Some(Commands::Fuse {
            apply,
            prefer,
            arbitrate,
            json,
        }) => {
            if let Err(e) =
                run_fuse(&oxigraph_manager, &config, apply, &prefer, arbitrate, json).await
            {
                eprintln!("知识融合失败: {}", e);
                std::process::exit(1);
            }
        }
        Some(Commands::Namespace {
            source,
            label,
            trust,
            kind,
            file_path,
            list,
            status,
            json,
        }) => {
            let result = if list {
                run_namespace_list(&oxigraph_manager, json).await
            } else if let Some(status) = status.as_deref() {
                run_namespace_status(
                    &oxigraph_manager,
                    source.as_deref().unwrap_or_default(),
                    status,
                    json,
                )
                .await
            } else {
                run_namespace_import(
                    &oxigraph_manager,
                    source.as_deref().unwrap_or_default(),
                    label.as_deref(),
                    &trust,
                    &kind,
                    file_path.as_deref().unwrap_or_default(),
                    json,
                )
                .await
            };
            if let Err(e) = result {
                eprintln!("命名空间操作失败: {}", e);
                std::process::exit(1);
            }
        }
        Some(Commands::Lens {
            question,
            subject,
            predicate,
            view,
            functional,
            contexts,
            facts,
            dialogue_state,
            json,
        }) => {
            if let Err(e) = run_lens(
                &oxigraph_manager,
                &config,
                &store_path,
                question,
                subject,
                predicate,
                view,
                functional,
                contexts,
                facts,
                dialogue_state,
                json,
            )
            .await
            {
                eprintln!("语境镜头查询失败: {}", e);
                std::process::exit(1);
            }
        }
        Some(Commands::Artifacts { fix, json, paths }) => {
            run_artifact_audit(&config, &store_path, fix, json, &paths);
        }
        Some(Commands::Serve {
            addr,
            max_sessions,
            session_dir,
            prompt_mode,
            recovery,
        }) => {

            include!("serve_arm.rs");

        }
        Some(Commands::Plugins { .. }) => {
            // 已在打开存储前处理；此处仅保持 match 穷尽。
        }
        Some(Commands::FuzzDialogue { .. }) => {
            // 已在打开存储前处理；此处仅保持 match 穷尽。
        }
        Some(Commands::Scenario { .. }) => {
            // 已在打开存储前处理；此处仅保持 match 穷尽。
        }
        Some(Commands::Interactive) | None => {
            // 运行交互模式
            run_interactive_mode(
                config,
                oxigraph_manager,
                perf_ledger,
                recovery_mode,
                trace_mode,
                &store_path,
                &cli.config,
            )
            .await;
        }
    }

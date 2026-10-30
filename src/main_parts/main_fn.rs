#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    // 配置契约 lint 必须最先处理：它恰恰要在 config.toml 损坏时可用，
    // 因此不依赖默认配置加载，也不打开知识存储。
    if let Some(Commands::ConfigLint { file, json, strict }) = &cli.command {
        run_config_lint(Some(file.as_deref().unwrap_or(&cli.config)), *json, *strict);
        return;
    }

    // 重写手册完全离线：语料是显式问答对，挖掘与留出评估都不需要存储。
    if let Some(Commands::RewriteMine {
        episodes,
        holdout_predicates,
        min_support,
        save,
        strict,
        json,
    }) = &cli.command
    {
        run_rewrite_mine(
            episodes.as_deref(),
            holdout_predicates,
            *min_support,
            save.as_deref(),
            *strict,
            *json,
        );
        return;
    }
    if let Some(Commands::RewriteRules {
        book,
        adopt,
        retire,
        json,
    }) = &cli.command
    {
        run_rewrite_rules(book, adopt, retire, *json);
        return;
    }

    // 配置兑现只读配置与图：在加载默认配置/打开交互存储之前处理，
    // 这样空图或存储路径覆盖也能得到一份可复现的报告。
    if let Some(Commands::ConfigFulfillment {
        file,
        json,
        strict,
        verbose,
        max_obligations,
    }) = &cli.command
    {
        run_config_fulfillment(
            Some(file.as_deref().unwrap_or(&cli.config)),
            cli.store.as_deref(),
            *json,
            *strict,
            *verbose,
            max_obligations.unwrap_or(fulfillment::DEFAULT_MAX_OBLIGATIONS),
        )
        .await;
        return;
    }

    // 发布检查只读配置与插件清单：在加载默认配置 / 打开存储之前处理，
    // 这样即使存储不可用也能完整跑完离线门禁。
    if let Some(Commands::ReleaseCheck {
        config_file,
        lock,
        update_lock,
        language_version,
        skip_evals,
        allow_regression,
        json,
    }) = &cli.command
    {
        run_release_check(
            if config_file == "config.toml" {
                &cli.config
            } else {
                config_file
            },
            lock,
            *update_lock,
            language_version.clone(),
            *skip_evals,
            *allow_regression,
            *json,
        );
        return;
    }

    // 质量差分模糊测试完全离线：不依赖 config.toml / 存储，先于任何 I/O 处理。
    if let Some(Commands::FuzzDialogue {
        cases,
        seed,
        baseline,
        update_baseline,
        allow_regression,
        json,
    }) = &cli.command
    {
        run_fuzz_dialogue(
            *cases,
            *seed,
            *json,
            baseline.as_deref(),
            *update_baseline,
            *allow_regression,
        );
        return;
    }

    // 查询剧本完全离线（自带事实）：先于配置加载与存储打开处理。
    if let Some(Commands::Scenario { file, json }) = &cli.command {
        run_query_scenario(file, *json);
        return;
    }

    // 开放闭包只操作会话账本，不打开知识存储，也不写律典。
    if let Some(Commands::OpenClosure {
        text,
        fact,
        ask,
        confirm,
        reject,
        retract,
        turn,
        session,
        max_depth,
        max_nodes,
        max_expansions,
        json,
    }) = &cli.command
    {
        run_open_closure(
            text,
            fact,
            ask,
            confirm,
            reject,
            retract,
            turn,
            session.as_deref(),
            *max_depth,
            *max_nodes,
            *max_expansions,
            *json,
        );
        return;
    }

    // 语义角色帧完全离线：配置里的 schema 优先，缺省使用内置核心种子。
    if let Some(Commands::Roles {
        text,
        predicate,
        json,
    }) = &cli.command
    {
        run_roles(text.as_deref(), predicate.as_deref(), *json, &cli.config);
        return;
    }

    // 依存脊完全离线：分析、归一或在临时索引上回答 who/what 问句。
    if let Some(Commands::Spine {
        text,
        fact,
        question,
        json,
    }) = &cli.command
    {
        run_spine(text.as_deref(), fact, question.as_deref(), *json);
        return;
    }

    // 隐私裁决完全离线：只读策略/会话台账，不打开知识存储。
    if let Some(Commands::Privacy {
        text,
        fact,
        facts_file,
        from_store,
        max_facts,
        question,
        purpose,
        precision,
        k,
        enforce,
        subject,
        utterance,
        grant,
        grant_once,
        revoke,
        forget,
        policy,
        session,
        json,
    }) = &cli.command
    {
        run_privacy(
            text.as_deref(),
            fact,
            facts_file.as_deref(),
            *from_store,
            *max_facts,
            cli.store.as_deref(),
            question.as_deref(),
            purpose,
            precision,
            *k,
            *enforce,
            subject,
            utterance,
            grant,
            grant_once,
            revoke,
            forget,
            policy.as_deref(),
            session.as_deref(),
            &cli.config,
            *json,
        )
        .await;
        return;
    }

    let recovery_mode = parse_recovery_mode(&cli.recovery);

    // 启动分期台账：config_load / store_open / facts_load / warm_build / bind
    // 各相耗时只观测不断言；serve 分支打印摘要，/metrics 输出明细。
    let startup_ledger = StartupLedger::shared();
    let config_start = Instant::now();
    let mut config = match load_config_from(&cli.config).await {
        Ok(cfg) => {
            startup_ledger.record_since("config_load", config_start);
            cfg
        }
        Err(e) => {
            eprintln!("加载配置文件失败（{}）: {}", cli.config, e);
            return;
        }
    };
    // 存储路径可被 --store 覆盖（便于对独立知识库做导入/评估而不动主库）。
    // 覆盖时相对对话状态路径也要落到新存储下，否则会话记忆会跨库串台。
    let store_path = cli
        .store
        .clone()
        .unwrap_or_else(|| config.system.oxigraph_store_path.clone());
    if cli.store.is_some() {
        isolate_dialogue_state_path(&mut config.reasoning.dialogue_state_path, &store_path);
    }
    let config = Arc::new(config);
    if let Some(Commands::DialogueArtifacts {
        selector,
        chronicle,
        session,
    }) = &cli.command
    {
        run_dialogue_artifacts(
            &config,
            &store_path,
            selector.as_deref(),
            chronicle.as_deref(),
            session.as_deref(),
        );
        return;
    }
    if let Some(Commands::Transduce {
        text,
        echo,
        kind,
        json,
    }) = &cli.command
    {
        run_transduce(
            &config,
            text.as_deref(),
            echo.as_deref(),
            kind.as_deref(),
            *json,
        );
        return;
    }
    if let Some(Commands::BatchTeach {
        facts,
        corpus,
        negatives,
        min_support,
        min_purity,
        max_candidates,
        adopt,
        retire,
        ledger,
        list,
        json,
    }) = &cli.command
    {
        run_batch_teach(
            &config,
            &store_path,
            facts.as_deref(),
            corpus.as_deref(),
            negatives.as_deref(),
            *min_support,
            *min_purity,
            *max_candidates,
            adopt,
            retire,
            ledger.as_deref(),
            *list,
            *json,
        );
        return;
    }
    // 分层追踪模式：命令行 `--trace` 优先，缺省回落 config.toml 的 [trace_graph]。
    let trace_mode = match cli.trace.as_deref() {
        Some(value) => parse_trace_mode(value),
        None => config.trace_graph.parse_mode(),
    };

    // 模式锻造只读配置与例子：不打开知识存储，适合脚本化预览与回归。
    if let Some(Commands::Forge {
        name,
        route,
        positives,
        negatives,
        priority,
        activate,
        json,
    }) = &cli.command
    {
        run_pattern_forge(
            &config, name, route, positives, negatives, *priority, *activate, *json,
        );
        return;
    }

    // 读法追踪只读配置与文本：不打开知识存储，适合在词表/分词变更前后做差分。
    if let Some(Commands::ReadTrace {
        question,
        corpus,
        lexicon,
        definitions,
        json,
    }) = &cli.command
    {
        run_read_trace(
            question.as_deref(),
            corpus.as_deref(),
            *lexicon,
            definitions.as_deref(),
            *json,
            &config,
        );
        return;
    }

    // 双向语义语法只读配置与文本：不打开知识存储，可检查抽取/生成往返。
    if let Some(Commands::Grammar {
        text,
        fact,
        roundtrip,
        json,
    }) = &cli.command
    {
        run_semantic_grammar(text.as_deref(), fact.as_deref(), *roundtrip, *json, &config);
        return;
    }

    // 决策前沿只读配置与文本：复用运行时路由的同一匹配器，不打开存储。
    if let Some(Commands::Frontier {
        question,
        corpus,
        near_miss,
        json,
    }) = &cli.command
    {
        run_frontier(
            question.as_deref(),
            corpus.as_deref(),
            *near_miss,
            *json,
            &config,
        );
        return;
    }

    // 构念召回评估只读配置：分母由规则与同义词表生成，不打开存储。
    if let Some(Commands::EvalRecall {
        baseline,
        update_baseline,
        allow_regression,
        near_miss,
        json,
    }) = &cli.command
    {
        run_eval_recall(
            baseline.as_deref(),
            *update_baseline,
            *allow_regression,
            *near_miss,
            *json,
            &config,
        );
        return;
    }

    // 插件清点/评测同样不依赖知识存储：清单是纯声明式资产。
    if let Some(Commands::Plugins { dir, json, eval }) = &cli.command {
        run_plugins(&config.plugins, dir.as_deref(), *json, *eval);
        return;
    }

    // 无需知识存储的子命令先行处理：只跑提示词 lint / NLU 评测时，
    // 即使 Oxigraph 库路径不可用也不受影响（不创建存储、不读知识库）。
    match &cli.command {
        Some(Commands::Events {
            text,
            file,
            question,
            limit,
            json,
            prolog,
            nquads,
            graph,
        }) => {
            run_events(
                text.as_deref(),
                file.as_deref(),
                question.as_deref(),
                *limit,
                *json,
                *prolog,
                *nquads,
                graph.as_deref(),
            );
            return;
        }
        Some(Commands::PromptLint { json }) => {
            run_prompt_lint(*json);
            return;
        }
        Some(Commands::EvalNlu {
            knowledge,
            prompt_mode,
            baseline,
            update_baseline,
            allow_regression,
            json,
        }) => {
            run_eval_nlu(
                knowledge,
                prompt_mode,
                baseline.clone(),
                *update_baseline,
                *allow_regression,
                *json,
            );
            return;
        }
        Some(Commands::EvalLink {
            baseline,
            update_baseline,
            allow_regression,
            json,
        }) => {
            run_eval_link(baseline.clone(), *update_baseline, *allow_regression, *json);
            return;
        }
        Some(Commands::EvalGrounding {
            baseline,
            update_baseline,
            allow_regression,
            json,
        }) => {
            run_eval_grounding(baseline.clone(), *update_baseline, *allow_regression, *json);
            return;
        }
        Some(Commands::EvalConstruction { json, strict }) => {
            run_eval_construction(*json, *strict);
            return;
        }
        Some(Commands::EvalIdentity {
            baseline,
            update_baseline,
            allow_regression,
            json,
        }) => {
            run_eval_identity(baseline.clone(), *update_baseline, *allow_regression, *json);
            return;
        }
        Some(Commands::EvalRecovery {
            baseline,
            update_baseline,
            allow_regression,
            json,
        }) => {
            run_eval_recovery(baseline.clone(), *update_baseline, *allow_regression, *json);
            return;
        }
        Some(Commands::ProofBench {
            baseline,
            update_baseline,
            allow_regression,
            emit_prolog,
            json,
        }) => {
            run_proof_bench(
                baseline.clone(),
                *update_baseline,
                *allow_regression,
                emit_prolog.as_deref(),
                *json,
            );
            return;
        }
        Some(Commands::FuzzDialogue {
            cases,
            seed,
            baseline,
            update_baseline,
            allow_regression,
            json,
        }) => {
            run_fuzz_dialogue(
                *cases,
                *seed,
                *json,
                baseline.as_deref(),
                *update_baseline,
                *allow_regression,
            );
            return;
        }
        _ => {}
    }

    // 会话级性能台账：存储与推理共享同一实例，报告覆盖整条链路
    let perf_ledger = PerfLedger::shared();

    // 创建Oxigraph存储
    let store_start = Instant::now();
    let oxigraph_manager = match OxigraphManager::with_perf_ledger(&store_path, perf_ledger.clone())
    {
        Ok(manager) => {
            startup_ledger.record_since("store_open", store_start);
            manager
        }
        Err(e) => {
            eprintln!("创建Oxigraph存储失败: {}", e);
            return;
        }
    };

    // 处理命令行参数

    include!("main_match.rs");

}

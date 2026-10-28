async fn run_interactive_mode(
    config: Arc<know_prolog::SystemConfig>,
    oxigraph_manager: OxigraphManager,
    perf_ledger: Arc<PerfLedger>,
    recovery_mode: RecoveryMode,
    trace_mode: TraceMode,
    store_path: &str,
    config_path: &str,
) {
    let mut lang_runtime = LangRuntime::new(config_path, (*config).clone());
    // 本会话生效的语言视图：热节来自最近的已提交快照，需重启的节沿用启动值。
    // `config` 始终是启动结构；后续语言读取一律走 `language_config`。
    let mut language_config = Arc::clone(&config);
    println!("{}", config.messages.startup);
    warn_config_contract(&config);
    let frame_program = compile_frame_program(&config);
    if !recovery_mode.eq(&RecoveryMode::Off) {
        println!(
            "失败恢复模式：{}（失败会生成结构化回执）。",
            recovery_mode.label()
        );
    }
    if !trace_mode.is_off() {
        println!(
            "分层追踪模式：{}（每轮决策写入追踪图）。",
            trace_mode.label()
        );
    }

    // 谓词画像注册表：跨会话加载已确认的准入/别名/互逆，再叠加配置声明
    let registry_path = format!("{}/predicate_registry.json", store_path);
    if oxigraph_manager.load_predicate_registry(&registry_path) {
        println!("已恢复谓词画像注册表。可用「谓词报告」查看。");
    }
    oxigraph_manager.configure_predicates(
        &config.predicate_profiles,
        config.reasoning.require_predicate_admission,
    );

    // 实体身份注册表：跨会话加载已确认的别名/重定向，再叠加配置声明；
    // 已确认别名同步进链接器（持久别名词面与规范名同权）。
    let entity_registry_path = format!("{}/entity_registry.json", store_path);
    if oxigraph_manager.load_entity_registry(&entity_registry_path) {
        println!("已恢复实体身份注册表。可用「身份台账」查看。");
    }
    oxigraph_manager.configure_entities(&config.entity_profiles);

    // 类型体系：别名/子类/互斥声明 + 模式闸门闭包档（config.toml [type_system]）
    oxigraph_manager.configure_type_system(&config.type_system);

    let mut knowledge_extractor = KnowledgeExtractor::new(config.clone());
    let batch_teaching_path = format!("{}/{}", store_path, batch_teaching::LEDGER_FILE);
    let mut batch_teaching = match BatchTeachingLedger::load(&batch_teaching_path) {
        BatchLedgerLoad::Loaded(ledger) => ledger,
        BatchLedgerLoad::Missing => BatchTeachingLedger::default(),
        BatchLedgerLoad::Damaged(error) => {
            eprintln!("批量教学台账损坏（{}）：{}", batch_teaching_path, error);
            BatchTeachingLedger::default()
        }
    };
    if !batch_teaching.active_rules().is_empty() {
        println!(
            "已恢复批量教学规则 {} 条。可用 batch-teach --list 查看。",
            batch_teaching.active_rules().len()
        );
    }
    // 文法式关系抽取器：DCG 语言器官，词库由谓词画像注册表注入
    let grammar_extractor = GrammarExtractor::new();
    let mut response_formatter = ResponseFormatter::new(config.clone());
    let mut nlu_engine = NLUEngine::new(&config);
    // 统一模式库：配置查询规则 + NLU 意图（同库展示、不路由）+ 跨会话教学模式。
    let mut pattern_library = PatternLibrary::from_config(&config);
    match oxigraph_manager.load_pattern_ledger().await {
        Ok(entries) if !entries.is_empty() => {
            let merged = pattern_library.merge_taught(&entries);
            if merged > 0 {
                println!(
                    "已恢复 {} 条教学模式（模式落图）。可用「模式路由」查看。",
                    merged
                );
            }
        }
        _ => {}
    }
    // 锻造台账：以会话账本的 active 状态为准，重建模式库里的锻造来源。
    // 模式落图可能比锻造台账旧（例如上次激活后未同步），先清掉再按 active
    // 重新物化，避免已退役规则在重启后“复活”。
    let forge_path = format!("{}/pattern_forge.json", store_path);
    let mut pattern_forge = PatternForge::load_from_path(&forge_path).unwrap_or_default();
    pattern_library.remove_source_prefix(pattern_forge::FORGE_SOURCE_PREFIX);
    for entry in pattern_forge.active_entries() {
        pattern_library.add(entry);
    }
    if !pattern_forge.is_empty() {
        println!(
            "已恢复锻造台账：{} 条（激活 {} 条）。可用「锻造台账」查看。",
            pattern_forge.len(),
            pattern_forge.active_count()
        );
        // 图谱可能还留着上次退役/遗忘前的锻造条目；按台账重建后再同步一次，
        // 让「退役后重启又复活」在数据层也不存在。
        match oxigraph_manager
            .sync_pattern_ledger(pattern_library.entries())
            .await
        {
            Ok(count) => println!("锻造模式落图校准：{} 条已同步。", count),
            Err(error) => eprintln!("锻造模式落图校准失败：{}", error),
        }
    }
    // 模式经验：跨会话恢复「上次怎么理解」的记录（独立于模式定义图）。
    let mut pattern_episodes = pattern_ir::PatternEpisodeLog::default();
    match oxigraph_manager.load_pattern_episodes().await {
        Ok(episodes) if !episodes.is_empty() => {
            let merged = pattern_episodes.merge_persisted(&episodes);
            if merged > 0 {
                println!(
                    "已恢复 {} 条模式经验（经验落图）。可用「模式经验」查看。",
                    merged
                );
            }
        }
        _ => {}
    }
    // 经验检查点：交互中每 8 轮把变化过的经验台账落图一次，
    // 避免进程被中断时丢掉整段会话的确认/否定反馈（退出时仍会兜底落图）。
    let mut synced_episodes_revision = pattern_episodes.revision();
    let mut turns_since_episode_sync = 0usize;

    let mut prolog_engine = PrologEngine::new();
    let mut local_semantic_adapter = {
        let nlp_config = &config.nlp;
        LocalSemanticAdapter::new(
            nlp_config.enable_local_semantic.unwrap_or(true)
                && nlp_config.disable_ownthink_api.unwrap_or(false),
        )
        .with_context_window_size(nlp_config.context_window_size)
    };

    // 创建外部知识服务：外链名册为外部知识留册（出处 + 逻辑时刻 + 时限），
    // 跨轮、跨重启记住"从哪里、在第几轮、读到了什么"，避免重复打网。
    let external_ledger_path = config
        .api
        .as_ref()
        .and_then(|api| api.external_ledger_path.clone())
        .filter(|path| !path.trim().is_empty())
        .unwrap_or_else(|| format!("{}/external_ledger.json", store_path));
    let external_ledger_capacity = config
        .api
        .as_ref()
        .map(|api| api.external_ledger_capacity)
        .unwrap_or(64);
    let external_ledger_ttl = config
        .api
        .as_ref()
        .map(|api| api.external_ledger_ttl_ticks)
        .unwrap_or(0);
    let external_knowledge_service = {
        let ledger = ExternalLedger::with_config(
            external_ledger_capacity,
            external_ledger_ttl,
            perf_ledger.clone(),
        );
        let service = ExternalKnowledgeService::with_ledger(ledger);
        let service = if let Some(api_config) = &config.api {
            service.with_endpoint(
                api_config.ownthink_base_url.clone(),
                api_config.ownthink_timeout_seconds,
            )
        } else {
            service
        };
        match service.load_ledger(&external_ledger_path) {
            LedgerLoad::Loaded(restored) if restored > 0 => {
                println!(
                    "已恢复外链名册：{} 条（「外链台账」查看，逻辑时刻 tick={}）。",
                    restored,
                    service.ledger_tick()
                );
            }
            LedgerLoad::Loaded(_) => {}
            LedgerLoad::Damaged(message) => {
                eprintln!("外链名册损坏，已忽略（不影响本地知识）：{}", message);
            }
            LedgerLoad::Missing => {}
        }
        service
    };

    let mut synonyms = if let Some(synonyms_config) = &config.synonyms {
        synonyms_config.mappings.clone()
    } else {
        create_default_synonyms()
    };

    // 词义感知同义词图：先恢复跨会话证据，再合并配置资产（实体/谓词画像、
    // 类型别名、扁平同义词）。墓碑不复活；新增配置证据幂等落图。
    let mut synonym_graph = match oxigraph_manager.load_synonym_graph().await {
        Ok(graph) => graph,
        Err(error) => {
            eprintln!("读取同义词图失败，按空图启动：{}", error);
            SynonymGraph::default()
        }
    };
    let configured_synonyms = SynonymGraph::from_config(&config, synonym_graph.turn.max(1));
    let merged_synonyms = synonym_graph.merge_from(&configured_synonyms);
    if merged_synonyms > 0 || synonym_graph.turn == 0 {
        synonym_graph.begin_turn(1);
        if let Err(error) = oxigraph_manager.sync_synonym_graph(&synonym_graph).await {
            eprintln!("同义词图落图失败（不影响本轮对话）：{}", error);
        }
    }
    if !synonym_graph.is_empty() {
        println!(
            "已恢复同义词图：{} 条（「synonyms」命令查看解析与反向词面）。",
            synonym_graph.len()
        );
    }

    // 神经符号对比仲裁器：联想通道（软逻辑+关联+图游走）与符号通道（蕴含+归结）
    // 交叉验证，输出带对冲词的结论，并保留会话级校准记忆
    let mut contrast_arbiter = ContrastArbiter::from_reasoning_config(&config.reasoning);
    // 对比追问进行中：记录当前提问对应的待定推测编号，用户答「是/不是」即结算
    let mut contrast_followup: Option<usize> = None;
    // 模式联想待确认：一次未命中解析给出的候选骨架。下一轮用户可直接答
    // 「是/不是」来转正或否决；否定记入负向反馈，避免同一骨架反复打扰。
    let mut pending_case: Option<(String, pattern_ir::CaseSuggestion)> = None;

    // 符号推理器：对话上下文 + 本体对齐 + 链接预测 + 符号嵌入
    let mut symbolic_reasoner = SymbolicReasoner::with_perf_ledger(
        config.reasoning.clone(),
        synonyms.clone(),
        perf_ledger.clone(),
    );
    // 读法观测：注入与交互/导入路径相同的 NLP 配置；失败轮才留轨迹。
    symbolic_reasoner.set_nlp_config(config.nlp.clone());
    if config.semantic_grammar.enabled {
        symbolic_reasoner.attach_semantic_grammar(
            know_prolog::semantic_grammar::RulePack::from_config(&config),
        );
    }
    if config.frame_semantics.enabled {
        symbolic_reasoner.attach_frame_book(
            know_prolog::frame_semantics::FrameBook::from_config(&config.frame_semantics),
        );
    }
    // 跨会话词法记忆：从独立命名图恢复词性证据与量词期待（只作软约束）。
    match oxigraph_manager.load_lexical_memory().await {
        Ok(memory) if !memory.is_empty() => {
            println!(
                "已恢复词法记忆：{} 词条，{} 量词期待（「lexical-memory」查看）。",
                memory.len(),
                memory.classifiers.len()
            );
            symbolic_reasoner.set_lexical_memory(memory);
        }
        Ok(_) => {}
        Err(error) => {
            eprintln!("词法记忆读取失败（已忽略，不影响知识库）：{}", error);
        }
    }
    // 谓词归一：把画像里的别名/互逆词面同步到查询解析与生成词面。
    // 正向词面走别名（alias → 槽位，可参与生成叙事）；逆向词面只登记
    // 查询换向（如「徒弟」「学生」命中 师傅 槽位时主宾对调），不改词面表。
    sync_predicate_pairs(&mut symbolic_reasoner, &oxigraph_manager);
    // 单一本体：配置里的类型层级与谓词画像并入教学本体（同一推理器、同一胶囊）
    let (ontology_program, owl_lessons) = owl_annotation::ontology_program_with_annotations(
        &OntologyProgram::from_config(&config.type_system, &config.predicate_profiles),
        &format!("{}/{}", store_path, owl_annotation::LEDGER_FILE),
    );
    let config_lessons = symbolic_reasoner.configure_ontology_program(&ontology_program);
    symbolic_reasoner.configure_schema_view(&SchemaView::from_config(
        &config.type_system,
        &config.predicate_profiles,
    ));
    if config_lessons > 0 {
        println!(
            "已并入本体公理：{} 条（类型层级、谓词画像与已接受的 OWL 人工反馈 {} 条）。",
            config_lessons, owl_lessons
        );
    }
    // 律典：显式采纳/退役决定跨重启生效；缺失文件不影响声明律。
    if let Some(persisted) = LawBook::load_from_path(&format!("{}/law_book.json", store_path)) {
        symbolic_reasoner.load_persisted_laws(persisted);
    }
    if let Some(book) = load_rewrite_book(store_path) {
        symbolic_reasoner.set_rewrite_book(book);
    }
    // 实体身份：已确认的持久别名同步进链接器（别名与规范名同权）
    sync_entity_aliases(&mut symbolic_reasoner, &oxigraph_manager);
    // 语境指称消歧（CRD）：跨轮「说法 ↔ 实体」绑定账本 + 区分性追问；
    // 账本落盘 {store}/reference_bindings.json，重启后仍记得澄清过的指称。
    let reference_binding_path = format!("{}/reference_bindings.json", store_path);
    let mut crd_ledger = match BindingLedger::load_from_path(&reference_binding_path) {
        Some(ledger) if !ledger.is_empty() => {
            println!(
                "已恢复指称绑定账本：{} 条（「绑定账本」查看）。",
                ledger.len()
            );
            ledger
        }
        _ => BindingLedger::new(),
    };
    let crd_resolver = ContextualResolver::new();
    // 进行中的澄清请求：下一轮优先按「选哪个/不是A是B」结算
    let mut crd_pending: Option<ClarifyRequest> = None;
    let grounding_certificate_path = format!("{}/grounding_certificates.json", store_path);
    let mut grounding_ledger =
        GroundingLedger::load_from_path(&grounding_certificate_path).unwrap_or_default();
    // 导入日志：跨重启的有界撤销栈
    let journal_path = format!("{}/import_journal.json", store_path);
    let mut import_journal = match ImportJournal::load(&journal_path) {
        JournalLoad::Loaded(journal) => {
            if !journal.is_empty() {
                println!(
                    "已恢复导入日志：{} 笔可撤销（「导入历史」查看）。",
                    journal.len()
                );
            }
            journal
        }
        JournalLoad::Damaged(message) => {
            eprintln!("导入日志损坏，已忽略（不影响知识库）：{}", message);
            ImportJournal::new()
        }
        JournalLoad::Missing => ImportJournal::new(),
    };
    // 导入可逆帧：从成功落库的陈述句学习「怎么说/怎么问」，随导入日志一起撤销。
    let import_frames_path = format!("{}/import_frames.json", store_path);
    let mut import_frames = match ImportFrameLedger::load(&import_frames_path) {
        ImportFrameLoad::Loaded(ledger) => {
            if !ledger.is_empty() {
                println!("已恢复导入帧：{} 条（「导入帧」查看）。", ledger.len());
                symbolic_reasoner.attach_import_frames(ledger.clone());
            }
            ledger
        }
        ImportFrameLoad::Damaged(message) => {
            eprintln!("导入帧台账损坏，已忽略（不影响知识库）：{}", message);
            ImportFrameLedger::new()
        }
        ImportFrameLoad::Missing => ImportFrameLedger::new(),
    };
    // 批量导入期间把整图刷新推迟到真正需要时（见循环顶部）
    let mut reasoner_stale = false;
    let mut graph_refreshed = false;
    if let Ok(facts) = load_graph_snapshot(&oxigraph_manager).await {
        symbolic_reasoner.observe_facts(&facts);
        oxigraph_manager.observe_entity_facts(&facts);
    }
    // 图谱观测落盘：重启后身份台账与图谱一致（观测是幂等的）
    let _ = oxigraph_manager.save_entity_registry(&entity_registry_path);
    if let Some(state_path) = config.reasoning.dialogue_state_path.as_deref() {
        if std::path::Path::new(state_path).exists() {
            match SymbolicReasoner::read_dialogue_state(state_path) {
                Ok((snapshot, receipt)) => {
                    if !receipt.canonical {
                        eprintln!(
                            "对话记忆字节非规范（已宽容读取，可用 `artifacts --fix` 规范化）：{}",
                            state_path
                        );
                    }
                    let contrast_snapshot = snapshot.contrast.clone();
                    match symbolic_reasoner.restore_dialogue_snapshot(&snapshot) {
                        Ok(()) => {
                            println!(
                                "已恢复上次对话记忆：{}",
                                symbolic_reasoner.dialogue_summary()
                            );
                        }
                        Err(error) => eprintln!(
                            "对话记忆拒绝（本体胶囊校验失败，其余状态未应用）：{}",
                            error
                        ),
                    }
                    if let Some(snapshot) = contrast_snapshot {
                        if contrast_arbiter.restore_snapshot(&snapshot) {
                            println!(
                                "已恢复上次对比校准记忆：{}",
                                contrast_arbiter.memory_summary()
                            );
                        }
                    }
                }
                Err(error) => {
                    eprintln!(
                        "对话记忆读取失败（已忽略，不影响知识库）：{} [decode_{}]",
                        error,
                        error.code()
                    );
                }
            }
        }
    }
    // 规则图账本：命名图物化视图恢复会话规则（跨进程）；账本写入不推进知识版本
    let stored_rule_ledger = match oxigraph_manager.load_rule_ledger().await {
        Ok(ledger) => Some(ledger),
        Err(error) => {
            eprintln!("规则图账本读取失败（已忽略，不影响知识库）：{}", error);
            None
        }
    };
    if let Some(ledger) = stored_rule_ledger.as_ref() {
        if !ledger.entries.is_empty() {
            let restored = symbolic_reasoner.restore_rules_from_ledger(&ledger.entries);
            if restored > 0 {
                println!(
                    "已从规则图账本恢复 {} 条会话规则（命名图 kg:graphs/rules）。",
                    restored
                );
            }
        }
    }
    // 仅当本会话确有规则时回写，避免 enable_resolution=false 时清空账本
    let startup_rule_entries = symbolic_reasoner.rule_ledger_entries();
    if !startup_rule_entries.is_empty() {
        let needs_sync = stored_rule_ledger
            .as_ref()
            .map(|ledger| ledger.fingerprint != symbolic_reasoner.rule_fingerprint())
            .unwrap_or(true);
        if needs_sync {
            match oxigraph_manager
                .sync_rule_ledger(&startup_rule_entries)
                .await
            {
                Ok(count) => println!(
                    "规则图账本已同步：{} 条规则 → 命名图 kg:graphs/rules。",
                    count
                ),
                Err(error) => {
                    eprintln!("规则图账本同步失败（已忽略，不影响会话）：{}", error);
                }
            }
        }
    }

    // 本体化对话管理器：把每轮对话实例化为对话本体，用 Horn 规则策略推导回答方式
    let dialogue_enabled = config.reasoning.enable_dialogue_manager;
    let mut dialogue_manager = DialogueManager::new();
    dialogue_manager.set_frame_program(frame_program);
    dialogue_manager.set_recovery_mode(recovery_mode);
    dialogue_manager.set_trace_mode(trace_mode);
    *dialogue_manager.trace_store_mut() = config.trace_graph.build_store();
    if dialogue_enabled {
        if let Some(state_path) = config.reasoning.dialogue_state_path.as_deref() {
            let manager_path = format!("{}.manager.json", state_path);
            if std::path::Path::new(&manager_path).exists() {
                match dialogue_manager.restore_from_path_report(&manager_path) {
                    Ok(receipt) => {
                        if !receipt.canonical {
                            eprintln!(
                                "对话本体记忆字节非规范（已宽容读取，可用 `artifacts --fix` 规范化）：{}",
                                manager_path
                            );
                        }
                        println!("已恢复对话本体记忆：{}", dialogue_manager.summary());
                    }
                    Err(error) => {
                        eprintln!(
                            "对话本体记忆读取失败（已忽略，不影响会话）：{} [decode_{}]",
                            error,
                            error.code()
                        );
                    }
                }
            }
        }
    }

    // 声明式插件：清单只在启动时合并；运行中切换的是改写/配方/模板/风格的
    // 启用模式。用户显式覆盖落盘到 <store>/plugin_state.json，重启后继续生效。
    let plugin_state_path = format!("{}/plugin_state.json", store_path);
    let plugin_registry = PluginRegistry::load_dir(&config.plugins.directory);
    let mut plugin_runtime = PluginRuntime::with_settings(plugin_registry, &config.plugins);
    if let Some(state) = PluginSessionState::load(&plugin_state_path) {
        let restored = plugin_runtime.restore_session_state(state);
        if restored > 0 {
            println!("已恢复插件会话状态：{} 个模式覆盖。", restored);
        }
    }
    if plugin_runtime.installed_count() > 0 {
        println!(
            "插件目录「{}」：已安装 {} 个，启用 {} 个（「插件」查看，「启用插件：<id>」切换）。",
            config.plugins.directory,
            plugin_runtime.installed_count(),
            plugin_runtime.enabled_count()
        );
    }
    for issue in plugin_runtime.issues() {
        if issue.severity == know_prolog::plugin::PluginSeverity::Error {
            eprintln!("插件：{}", issue.render());
        }
    }
    // 本体贡献按启用集合并进既有谓词对齐器与统一本体推理器（单调、幂等）。
    let plugin_sync = symbolic_reasoner.apply_plugin_runtime(&plugin_runtime);
    if plugin_sync.aliases > 0 {
        println!("插件本体已合并：{} 条谓词别名。", plugin_sync.aliases);
    }
    if plugin_sync.lessons > 0 {
        println!("插件本体已并入统一推理器：{} 条公理。", plugin_sync.lessons);
    }
    // 互逆谓词的逆向词面：发问时主宾对调（「执导」问句 ↔「导演」事实）。
    if plugin_sync.inverse_surfaces > 0 {
        println!(
            "插件本体已合并：{} 条逆向查询词面。",
            plugin_sync.inverse_surfaces
        );
    }
    // 指称线索在启动时补入分词/词性词典，与本体别名一样只在启动时合并。
    let mut plugin_nlp = plugin_runtime.augmented_nlp_config(&config.nlp);
    let plugin_reference_entries = plugin_nlp
        .pos_dict
        .len()
        .saturating_sub(config.nlp.pos_dict.len());
    if plugin_reference_entries > 0 {
        println!(
            "插件指称线索已合并：{} 条词性提示。",
            plugin_reference_entries
        );
    }

    // 问答式关系获取（QRA）：会话账本 + 覆盖层 + 对话诱导模式。
    // 只读图谱与画像，确认后才进入覆盖层；采纳后才进入会话事实集。
    let acquisition_path = format!("{}/acquisition_ledger.json", store_path);
    let mut acquisition =
        RelationAcquisition::load_from_path(&acquisition_path).unwrap_or_default();
    if acquisition.turn_count() > 0 {
        println!(
            "已恢复关系获取账本：{} 条，诱导模式 {} 个。",
            acquisition.claims().count(),
            acquisition.patterns().count()
        );
    }
    let assertion_reader_path = format!("{}/teachable_assertions.json", store_path);
    let mut assertion_reader =
        TeachableAssertionReader::load_from_path(&assertion_reader_path).unwrap_or_default();
    if assertion_reader.turn_count() > 0 {
        println!(
            "已恢复断言阅读器：提及卡 {} 张，{}。",
            assertion_reader.mentions().count(),
            if assertion_reader.has_pending() {
                "有一条待确认断言"
            } else {
                "没有待确认断言"
            }
        );
    }
    let shape_dialogue_path = format!("{}/shape_dialogue.json", store_path);
    let mut shape_dialogue =
        ShapeDialogue::load_from_path(&shape_dialogue_path).unwrap_or_default();
    if shape_dialogue.turn_count() > 0 {
        println!(
            "已恢复形状填充对话：轮次 {}，{}。",
            shape_dialogue.turn_count(),
            if shape_dialogue.has_pending() {
                "有一个待答问题"
            } else {
                "没有待答问题"
            }
        );
    }

    // 对话编年史：追加式哈希链档案，是对话的真相源；快照只是检查点。
    // 每轮输入在任何处理之前先落盘（写前日志），崩溃最多丢进行中的一轮；
    // 坏行只废尾巴不废全档：可信前缀保留、原子重写截断后续写。
    let chronicle_path = config
        .reasoning
        .dialogue_state_path
        .as_deref()
        .map(|path| format!("{}.chronicle.jsonl", path))
        .unwrap_or_else(|| format!("{}/dialogue.chronicle.jsonl", store_path));
    let mut chronicle = match Chronicle::open(&chronicle_path, FsyncPolicy::EachAppend) {
        Ok((mut chronicle, audit)) => {
            if !audit.is_clean() {
                eprintln!("对话编年史{}", audit.verdict());
                match chronicle.repair_tail() {
                    Ok(_) => {
                        println!(
                            "编年史已截断损坏尾行并保留可信前缀：{} 条记录。",
                            chronicle.len()
                        );
                        Some(chronicle)
                    }
                    Err(error) => {
                        eprintln!("编年史尾行重写失败（本轮不落史以免污染链条）：{}", error);
                        None
                    }
                }
            } else {
                if !chronicle.is_empty() {
                    println!(
                        "已恢复对话编年史：{} 条记录，水位 seq {}（「编年史」查看 / 「回放 N」重建）。",
                        chronicle.len(),
                        chronicle.last_seq()
                    );
                }
                Some(chronicle)
            }
        }
        Err(error) => {
            eprintln!("对话编年史打开失败（本轮不落史）：{}", error);
            None
        }
    };
    // 跨会话焦点兜底：只有快照没有恢复出焦点时，才用编年史最近的实体线程。
    // 显式会话状态永远优先；这里只移动下一轮的指代先验，不新增对话轮。
    if dialogue_enabled && dialogue_manager.focus_entity().is_none() {
        if let Some(chronicle) = chronicle.as_ref() {
            let registry = oxigraph_manager.entity_registry_snapshot();
            let digest = EntityThreadDigest::build(chronicle, &registry);
            let candidate = digest
                .threads
                .values()
                .filter(|thread| thread.last_seq.is_some())
                .max_by(|left, right| {
                    left.last_seq
                        .cmp(&right.last_seq)
                        .then_with(|| right.canonical.cmp(&left.canonical))
                });
            if let Some(thread) = candidate {
                if dialogue_manager.note_restored_focus(&thread.canonical) {
                    println!(
                        "已从编年史恢复最近实体焦点：{}（seq {}；无会话快照时兜底）。",
                        thread.canonical,
                        thread.last_seq.unwrap_or(0)
                    );
                }
            }
        }
    }

    // KB 修订号：知识库每发生一次批量变更（导入/撤销）自增，随编年史记录，
    // 用于区分「重演」时的知识漂移（挂钟时间不可断言，序号才是语义时戳）。
    let mut kb_revision: u64 = chronicle
        .as_ref()
        .map(|chronicle| chronicle.kb_revision())
        .unwrap_or(0);

    // 分层追踪回放：把编年史里的 `kind="trace"` 逐轮重建为追踪工作记忆
    //（问据栈/指代轨迹/更正覆盖），使重启后仍能承继谓词焦点与改口。
    if !trace_mode.is_off() {
        if let Some(chronicle) = chronicle.as_ref() {
            let mut replayed = 0usize;
            for entry in chronicle.replay_to(chronicle.last_seq()) {
                if entry.kind != "trace" {
                    continue;
                }
                if let Ok(trace) = serde_json::from_value::<know_prolog::trace_graph::TurnTrace>(
                    entry.payload.clone(),
                ) {
                    dialogue_manager.replay_turn_trace(trace);
                    replayed += 1;
                }
            }
            if replayed > 0 {
                println!("已回放分层追踪：{} 轮（「追踪报告」查看）。", replayed);
            }
        }
    }

    println!("{}", config.messages.ready);
    println!("{}", config.messages.query_prompt);
    println!("{}", config.messages.import_prompt);
    println!("{}", config.messages.quit_prompt);
    println!();

    let stdin_reader = std::io::stdin();
    let mut event_dialogue: Option<EventDialogue> = None;
    let mut lexical_memory_dirty = false;
    loop {
        if lexical_memory_dirty {
            let memory = symbolic_reasoner.lexical_memory().clone();
            match oxigraph_manager.sync_lexical_memory(&memory).await {
                Ok(_) => lexical_memory_dirty = false,
                Err(error) => {
                    eprintln!("词法记忆检查点落图失败（下次重试）：{}", error);
                }
            }
        }
        turns_since_episode_sync += 1;
        if turns_since_episode_sync >= 8 {
            turns_since_episode_sync = 0;
            let revision = pattern_episodes.revision();
            if revision != synced_episodes_revision {
                match oxigraph_manager
                    .sync_pattern_episodes(pattern_episodes.entries())
                    .await
                {
                    Ok(_) => synced_episodes_revision = revision,
                    Err(error) => {
                        eprintln!("模式经验检查点落图失败（下次重试）：{}", error);
                    }
                }
            }
        }
        print!("> ");
        std::io::stdout().flush().unwrap();

        let mut input = String::new();
        match stdin_reader.lock().read_line(&mut input) {
            Ok(0) => break, // EOF reached, exit the loop
            Ok(_) => {}
            Err(e) => {
                eprintln!("Error reading input: {}", e);
                break;
            }
        }
        let input = input.trim();

        include!("loop_a.rs");

        let enabled_before = plugin_runtime.enabled_count();
        if let Some(outcome) = plugin_runtime.handle_command(input) {
            println!("{}", outcome.text);
            if outcome.state_changed {
                let sync = symbolic_reasoner.apply_plugin_runtime(&plugin_runtime);
                if sync.aliases > 0 || sync.lessons > 0 || sync.inverse_surfaces > 0 {
                    println!(
                        "插件本体已增量合并：{} 条别名、{} 条公理、{} 条逆向词面。",
                        sync.aliases, sync.lessons, sync.inverse_surfaces
                    );
                } else if plugin_runtime.enabled_count() < enabled_before {
                    println!(
                        "提示：停用插件不撤回本会话已合并的本体与别名，重启后按当前启用集重建。"
                    );
                }
                if let Err(error) = plugin_runtime.session_state().save(&plugin_state_path) {
                    eprintln!("插件会话状态保存失败（不影响本次回答）：{}", error);
                }
            }
            println!();
            continue;
        }

        // 插件改写只作用于自然语言查询：命令、导入行与帮助保持逐字原样。
        let raw_input = input;
        let plugin_outcome = if !raw_input.is_empty()
            && matches!(
                detect_input_mode(raw_input, &language_config.nlp.import_keywords),
                InputMode::Query
            )
            && !is_command_like(raw_input)
        {
            Some(plugin_runtime.rewrite(raw_input))
        } else {
            None
        };
        let input = plugin_outcome
            .as_ref()
            .map(|outcome| outcome.text.as_str())
            .unwrap_or(raw_input);

        // 外链名册的逻辑时刻：每个非空用户轮次推进一次（TTL 的计量单位）
        if !input.is_empty() {
            external_knowledge_service.advance_tick();
        }

        if dialogue_enabled && !raw_input.is_empty() {
            dialogue_manager.observe_input(raw_input);
        }

        // 全息时序记忆：把本轮输入中提及的实体压入置换编码的焦点叠加（多轮"注意力轨迹"）
        if config.reasoning.enable_holographic_memory && !raw_input.is_empty() {
            symbolic_reasoner.note_holo_context(raw_input);
        }

        include!("loop_b1.rs");

        include!("loop_b2.rs");

        let mut frame_replay: Option<String> = None;
        match handle_frame_clarification(&mut symbolic_reasoner, input) {
            FrameClarificationStep::Replay(question) => frame_replay = Some(question),
            FrameClarificationStep::Handled => continue,
            FrameClarificationStep::Pass => {}
        }
        let input = frame_replay.as_deref().unwrap_or(input);

        include!("loop_c1.rs");

        include!("loop_c2.rs");

        let mode = detect_input_mode(input, &language_config.nlp.import_keywords);

        // 自由陈述先过可教断言阅读器：高精度候选进入 QRA 账本等待确认；

        match mode {
            InputMode::KnowledgeImport => include!("import_arm.rs"),
            InputMode::Query => include!("query_arm.rs"),
        }

        println!();
    }

    if let Some(state_path) = config.reasoning.dialogue_state_path.as_deref() {
        match symbolic_reasoner
            .save_dialogue_state_with_contrast(state_path, Some(&contrast_arbiter.snapshot()))
        {
            Ok(_) => println!(
                "对话记忆已保存到 {}（含对比校准记忆：{}）",
                state_path,
                contrast_arbiter.memory_summary()
            ),
            Err(e) => eprintln!("保存对话记忆失败: {}", e),
        }
    }

    if dialogue_enabled {
        if let Some(state_path) = config.reasoning.dialogue_state_path.as_deref() {
            let manager_path = format!("{}.manager.json", state_path);
            match dialogue_manager.save_to_path(&manager_path) {
                Ok(_) => println!(
                    "对话本体记忆已保存到 {}（{}）",
                    manager_path,
                    dialogue_manager.summary()
                ),
                Err(e) => eprintln!("保存对话本体记忆失败: {}", e),
            }
        }
    }

    // 模式经验落图：会话中积累的命中/未命中经验跨重启可检索。
    if !pattern_episodes.is_empty() {
        match oxigraph_manager
            .sync_pattern_episodes(pattern_episodes.entries())
            .await
        {
            Ok(count) => println!("模式经验已保存（经验落图：{} 条）。", count),
            Err(error) => eprintln!("保存模式经验失败：{}", error),
        }
    }

    // 外链名册落盘：跨重启保留取回的外部知识（出处 + 逻辑时刻）
    match external_knowledge_service.save_ledger(&external_ledger_path) {
        Ok(_) => println!(
            "外链名册已保存到 {}（{} 条条目）。",
            external_ledger_path,
            external_knowledge_service.ledger_len()
        ),
        Err(e) => eprintln!("保存外链名册失败: {}", e),
    }

    persist_acquisition(&acquisition, &acquisition_path);
    persist_shape_dialogue(&shape_dialogue, &shape_dialogue_path);
    println!("系统已退出。");
}

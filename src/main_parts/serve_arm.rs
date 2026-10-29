{
            let prompt_mode = parse_prompt_mode(&prompt_mode);
            let recovery_mode = parse_recovery_mode(&recovery);
            warn_config_contract(&config);
            let frame_program = compile_frame_program(&config);
            let facts_start = Instant::now();
            let facts = match serve_facts(&oxigraph_manager).await {
                Ok(facts) => {
                    startup_ledger.record_since("facts_load", facts_start);
                    facts
                }
                Err(e) => {
                    eprintln!("读取知识库失败: {}", e);
                    std::process::exit(1);
                }
            };
            let synonyms = if let Some(synonyms_config) = &config.synonyms {
                synonyms_config.mappings.clone()
            } else {
                create_default_synonyms()
            };
            // 质量护照索引：与交互模式同一确定性构建路径，供每个会话的
            // 「带出处 / 披露冲突 / 追问缺口」话语行为使用（只读派生，不写图谱）。
            let registry_path = format!("{}/predicate_registry.json", store_path);
            oxigraph_manager.load_predicate_registry(&registry_path);
            oxigraph_manager.configure_predicates(
                &config.predicate_profiles,
                config.reasoning.require_predicate_admission,
            );
            // 实体身份注册表：CLI 里签过字的持久别名同样要对齐进服务暖基座，
            // 否则同一个指称在命令行能解、走 HTTP 却查无此人。
            let entity_registry_path = format!("{}/entity_registry.json", store_path);
            if oxigraph_manager.load_entity_registry(&entity_registry_path) {
                println!("已恢复实体身份注册表（服务会话同源携带）。");
            }
            oxigraph_manager.configure_entities(&config.entity_profiles);
            oxigraph_manager.configure_type_system(&config.type_system);
            let registry = oxigraph_manager.predicate_registry_snapshot();
            let types = oxigraph_manager.type_system_snapshot();
            let ledger = provenance_seed_for_store(Some(&store_path), None);
            let mut quality_report = build_report_with_provenance(
                &facts,
                &registry,
                &types,
                &config.entity_types,
                Some(&ledger),
            );
            if let Ok(fact_refs) = oxigraph_manager.load_fact_refs().await {
                know_prolog::quality::apply_witnessed_conflicts(
                    &mut quality_report,
                    &fact_refs,
                    Some(&ledger),
                    &registry,
                    &types,
                );
            }
            let quality_index = QualityIndex::from_report_with_provenance(
                &quality_report,
                &facts,
                &registry,
                Some(&ledger),
            );
            println!(
                "KnowProlog 会话服务已启动：http://{}（{} 条事实，最多 {} 个驻留会话{}，提示模式 {}，失败恢复 {}）",
                addr,
                facts.len(),
                max_sessions,
                session_dir
                    .as_deref()
                    .map(|dir| format!("，冷会话目录 {}", dir))
                    .unwrap_or_default(),
                prompt_mode.label(),
                recovery_mode.label()
            );
            let mut api_state = know_prolog::serve_api::ApiState::new(
                facts,
                config.reasoning.clone(),
                synonyms,
                perf_ledger.clone(),
                max_sessions,
                startup_ledger.clone(),
            );
            // 读法观测：服务会话的失败轮同样留下有界回执，指标进共享性能台账。
            api_state.set_nlp_config(config.nlp.clone());
            if config.semantic_grammar.enabled {
                api_state.set_semantic_grammar(
                    know_prolog::semantic_grammar::RulePack::from_config(&config),
                );
            }
            api_state.set_frame_program(frame_program.clone());
            // 跨会话词法记忆：与交互启动一致地恢复词性证据与量词期待。
            match oxigraph_manager.load_lexical_memory().await {
                Ok(memory) if !memory.is_empty() => {
                    let restored = api_state.apply_lexical_memory(memory);
                    println!("服务已恢复词法记忆：{} 词条。", restored);
                }
                Ok(_) => {}
                Err(error) => {
                    eprintln!("词法记忆读取失败（已忽略，不影响知识库）：{}", error);
                }
            }
            // 单一本体：配置态本体注入暖基座，之后分叉的每个会话同源携带。
            let (ontology_program, owl_lessons) = owl_annotation::ontology_program_with_annotations(
                &OntologyProgram::from_config(&config.type_system, &config.predicate_profiles),
                &format!("{}/{}", store_path, owl_annotation::LEDGER_FILE),
            );
            let api_config_lessons = api_state.configure_ontology(&ontology_program);
            api_state.configure_schema(&SchemaView::from_config(
                &config.type_system,
                &config.predicate_profiles,
            ));
            if api_config_lessons > 0 {
                println!(
                    "服务已注入本体公理：{} 条（含 OWL 人工反馈 {} 条）。",
                    api_config_lessons, owl_lessons
                );
            }
            // 知识对齐：谓词别名/换向词面 + 画像律典 + 实体持久别名 + 律典的
            // 跨重启采纳/退役决定，与交互启动同一套注册入口，建会话前完成。
            let aligned_predicates = api_state
                .align_predicate_pairs(&oxigraph_manager.predicate_query_pairs(), &registry);
            let aligned_entities =
                api_state.align_entity_aliases(&oxigraph_manager.confirmed_entity_aliases());
            if aligned_predicates > 0 || aligned_entities > 0 {
                println!(
                    "服务已对齐知识注册表：{} 条谓词别名/换向词面、{} 条实体别名。",
                    aligned_predicates, aligned_entities
                );
            }
            if let Some(persisted) =
                LawBook::load_from_path(&format!("{}/law_book.json", store_path))
            {
                let decided = api_state.load_law_book(persisted);
                if decided > 0 {
                    println!("服务已载入律典显式决定：{} 条。", decided);
                }
            }
            if let ImportFrameLoad::Loaded(frames) =
                ImportFrameLedger::load(&format!("{}/import_frames.json", store_path))
            {
                let restored = api_state.load_import_frames(frames);
                if restored > 0 {
                    println!("服务已恢复导入措辞/问法：{} 条。", restored);
                }
            }
            if let Some(book) = load_rewrite_book(&store_path) {
                let restored = api_state.load_rewrite_book(book);
                if restored > 0 {
                    println!("服务已恢复重写手册：{} 条规则。", restored);
                }
            }
            let quality_index = quality_index.with_revision(api_state.warm_revision());
            api_state.set_quality_index(Some(Arc::new(quality_index)));
            if let Some(dir) = session_dir {
                api_state.set_session_dir(dir);
            }
            api_state.set_prompt_mode(prompt_mode);
            api_state.set_recovery_mode(recovery_mode);
            // 服务级插件运行时：多会话共享改写/配方/模板/风格，模式取自
            // 配置与 <store>/plugin_state.json；HTTP 不提供热切换。
            let plugin_state_path = format!("{}/plugin_state.json", store_path);
            let plugin_registry = PluginRegistry::load_dir(&config.plugins.directory);
            let mut plugin_runtime = PluginRuntime::with_settings(plugin_registry, &config.plugins);
            if let Some(state) = PluginSessionState::load(&plugin_state_path) {
                plugin_runtime.restore_session_state(state);
            }
            if plugin_runtime.installed_count() > 0 {
                println!(
                    "插件目录「{}」：已安装 {} 个，启用 {} 个（服务级共享）。",
                    config.plugins.directory,
                    plugin_runtime.installed_count(),
                    plugin_runtime.enabled_count()
                );
            }
            let plugin_receipt = api_state.configure_plugin_runtime(&plugin_runtime);
            if !plugin_receipt.is_empty() {
                println!(
                    "插件本体已并入服务统一推理器：{} 条公理、{} 条别名、{} 条逆向词面。",
                    plugin_receipt.lessons, plugin_receipt.aliases, plugin_receipt.inverse_surfaces
                );
            }
            let plugin_runtime = (plugin_runtime.installed_count() > 0)
                .then(|| Arc::new(parking_lot::Mutex::new(plugin_runtime)));
            api_state.set_plugin_runtime(plugin_runtime);
            let state = Arc::new(api_state);
            if let Err(e) = know_prolog::serve_api::serve(state, &addr).await {
                eprintln!("服务退出: {}", e);
                std::process::exit(1);
            }
}

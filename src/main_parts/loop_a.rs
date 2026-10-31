{

        if matches!(input, "quit" | "退出" | "exit") {
            break;
        }

        if matches!(input, "帮助" | "help" | "?" | "？" | "h") {
            println!("{}", interactive_help());
            println!();
            continue;
        }

        if let Some(summary) = symbolic_reasoner.observe_lexical_text(input) {
            if summary.tokens > 0 || summary.classifiers > 0 {
                lexical_memory_dirty = true;
            }
        }

        if let Some(text) = strip_any_prefix(input, &["事件图：", "事件图:"]) {
            let text = text.trim();
            if text.is_empty() {
                println!("用法：事件图：<文本>");
                println!();
                continue;
            }
            let graph = EventGraph::build(text);
            print!("{}", graph.render_text());
            event_dialogue = Some(EventDialogue::new(graph));
            println!();
            continue;
        }

        if let Some(question) = strip_any_prefix(
            input,
            &[
                "事件问答：",
                "事件问答:",
                "事件问：",
                "事件问:",
                "事件追问：",
                "事件追问:",
            ],
        ) {
            let question = question.trim();
            if question.is_empty() {
                println!("用法：事件问答：<问题>");
                println!();
                continue;
            }
            match event_dialogue.as_mut() {
                Some(dialogue) => {
                    let answer = dialogue.ask(question);
                    print!("{}", answer.render_text());
                }
                None => {
                    println!("还没有事件图。先用「事件图：<文本>」建立事件焦点。");
                }
            }
            println!();
            continue;
        }

        if matches!(input, "事件事实" | "事件rdf" | "事件RDF" | "事件nquads") {
            match event_dialogue.as_ref() {
                Some(dialogue) => print!("{}", dialogue.graph().to_nquads()),
                None => println!("还没有事件图。先用「事件图：<文本>」建立事件焦点。"),
            }
            println!();
            continue;
        }

        if matches!(input, "事件焦点" | "事件焦点栈") {
            match event_dialogue.as_ref() {
                Some(dialogue) if !dialogue.focus().is_empty() => {
                    println!("事件焦点（最近在下）：");
                    for id in dialogue.focus().ids() {
                        match dialogue.graph().event(id) {
                            Some(event) => println!("  [{}] {}", id, event.render_text()),
                            None => println!("  [{}]（图内已无此事件）", id),
                        }
                    }
                }
                Some(_) => println!("事件焦点为空。"),
                None => println!("还没有事件图。先用「事件图：<文本>」建立事件焦点。"),
            }
            println!();
            continue;
        }

        // 配置契约体检：对本会话实际生效的语言视图做静态检查。
        if matches!(input, "配置契约" | "配置检查" | "配置体检" | "config-lint") {
            let issues = config_contract::lint(&language_config);
            println!("{}", config_contract::render_text(&issues));
            continue;
        }

        // 配置热加载：一次事务式修订。lint 有 Error 整次拒绝；只有语言节
        // 立即接管，结构节留待重启；会话记忆不随配置重建。
        if matches!(
            input,
            "重载配置" | "配置重载" | "重新加载配置" | "reload-config" | "config-reload"
        ) {
            match lang_runtime.reload() {
                Ok(LangReloadOutcome::Applied(apply)) => {
                    let pattern_before_reload = pattern_library.clone();
                    language_config = Arc::clone(lang_runtime.current().language_config());
                    let (patterns, language, plugin_entries) = apply_language_snapshot(
                        &language_config,
                        &mut knowledge_extractor,
                        &mut response_formatter,
                        &mut nlu_engine,
                        &mut pattern_library,
                        &plugin_runtime,
                        &mut plugin_nlp,
                        &mut synonyms,
                        &mut local_semantic_adapter,
                        &mut symbolic_reasoner,
                    );
                    println!(
                        "配置已重载：epoch {} → {}。",
                        apply.previous_epoch, apply.epoch
                    );
                    println!(
                        "  已热替换：{}。",
                        lang_runtime::render_sections(&apply.diff.hot)
                    );
                    if !apply.diff.restart_required.is_empty() {
                        println!(
                            "  需重启生效：{}。",
                            lang_runtime::render_sections(&apply.diff.restart_required)
                        );
                    }
                    println!(
                        "  模式层：新增 {} 条、移除 {} 条，保留会话模式 {} 条；同义词新增 {} 条；插件指称 {} 条。",
                        patterns.added,
                        patterns.removed,
                        patterns.kept,
                        language.synonyms_added,
                        plugin_entries
                    );
                    if apply.warnings > 0 {
                        println!(
                            "  配置契约警告 {} 条（输入「配置契约」查看）。",
                            apply.warnings
                        );
                    }
                    println!("  对话焦点、编年史、会话事实与已学模式保持不变。");
                    let clock = EvolutionClock::new(
                        apply.epoch,
                        kb_revision,
                        pattern_evolution
                            .current()
                            .map(|generation| generation.generation)
                            .unwrap_or(0),
                        pattern_episodes.revision(),
                    );
                    match gate_config_change(
                        &mut pattern_evolution,
                        &pattern_before_reload,
                        &pattern_library,
                        pattern_episodes.entries(),
                        &language_config,
                        clock,
                        &evolution_path,
                        "配置重载",
                    ) {
                        Ok(receipt) => {
                            println!(
                                "  演化闸门：{}（回放 {} 例，回归 {}，留出新增命中 {}）。",
                                receipt.verdict.label(),
                                receipt.replay.cases,
                                receipt.replay.regressions,
                                receipt.replay.held_out_gains
                            );
                            if !receipt.verdict.may_commit() {
                                pattern_library = pattern_before_reload;
                                println!("  模式层未通过闸门，已恢复重载前状态：");
                                println!("{}", receipt.render());
                            } else if let Err(error) = oxigraph_manager
                                .sync_pattern_ledger(pattern_library.entries())
                                .await
                            {
                                eprintln!("  模式层已更新，但模式落图失败：{}", error);
                            }
                        }
                        Err(error) => {
                            pattern_library = pattern_before_reload;
                            println!("  演化台账无法记录，模式层已恢复重载前状态：{}", error);
                        }
                    }
                }
                Ok(LangReloadOutcome::Unchanged { epoch }) => {
                    println!("配置未发生语义变化（epoch {}）。", epoch);
                }
                Ok(LangReloadOutcome::Rejected { reason, issues }) => {
                    println!("{}；当前仍为 epoch {}。", reason, lang_runtime.epoch());
                    println!("{}", config_contract::render_text(&issues));
                    println!("（会话记忆未受影响。）");
                }
                Err(error) => eprintln!("{}", error),
            }
            println!();
            continue;
        }

        // 配置回滚：把上一份快照的内容重新提交为新 epoch，而不是倒转时间。
        if matches!(
            input,
            "回滚配置" | "配置回滚" | "config-rollback" | "rollback-config"
        ) {
            match lang_runtime.rollback() {
                Ok(rollback) => {
                    let pattern_before_rollback = pattern_library.clone();
                    language_config = Arc::clone(lang_runtime.current().language_config());
                    let (patterns, language, plugin_entries) = apply_language_snapshot(
                        &language_config,
                        &mut knowledge_extractor,
                        &mut response_formatter,
                        &mut nlu_engine,
                        &mut pattern_library,
                        &plugin_runtime,
                        &mut plugin_nlp,
                        &mut synonyms,
                        &mut local_semantic_adapter,
                        &mut symbolic_reasoner,
                    );
                    println!(
                        "配置已回滚：恢复 epoch {} 的内容，提交为 epoch {}。",
                        rollback.restored_epoch, rollback.epoch
                    );
                    println!(
                        "  已热替换：{}。",
                        lang_runtime::render_sections(&rollback.diff.hot)
                    );
                    if !rollback.diff.restart_required.is_empty() {
                        println!(
                            "  需重启生效：{}。",
                            lang_runtime::render_sections(&rollback.diff.restart_required)
                        );
                    }
                    println!(
                        "  模式层：新增 {} 条、移除 {} 条，保留会话模式 {} 条；同义词新增 {} 条；插件指称 {} 条。",
                        patterns.added,
                        patterns.removed,
                        patterns.kept,
                        language.synonyms_added,
                        plugin_entries
                    );
                    println!("  同义词按会话单调追加，已登记映射需重启才完全撤销。");
                    println!("  对话焦点、编年史、会话事实与已学模式保持不变。");
                    let clock = EvolutionClock::new(
                        rollback.epoch,
                        kb_revision,
                        pattern_evolution
                            .current()
                            .map(|generation| generation.generation)
                            .unwrap_or(0),
                        pattern_episodes.revision(),
                    );
                    match gate_config_change(
                        &mut pattern_evolution,
                        &pattern_before_rollback,
                        &pattern_library,
                        pattern_episodes.entries(),
                        &language_config,
                        clock,
                        &evolution_path,
                        "配置回滚",
                    ) {
                        Ok(receipt) => {
                            println!(
                                "  演化闸门：{}（回放 {} 例，回归 {}，留出新增命中 {}）。",
                                receipt.verdict.label(),
                                receipt.replay.cases,
                                receipt.replay.regressions,
                                receipt.replay.held_out_gains
                            );
                            if !receipt.verdict.may_commit() {
                                pattern_library = pattern_before_rollback;
                                println!("  模式层未通过闸门，已恢复回滚前状态：");
                                println!("{}", receipt.render());
                            } else if let Err(error) = oxigraph_manager
                                .sync_pattern_ledger(pattern_library.entries())
                                .await
                            {
                                eprintln!("  模式层已更新，但模式落图失败：{}", error);
                            }
                        }
                        Err(error) => {
                            pattern_library = pattern_before_rollback;
                            println!("  演化台账无法记录，模式层已恢复回滚前状态：{}", error);
                        }
                    }
                }
                Err(error) => {
                    println!("{}（epoch {} 保持）。", error, lang_runtime.epoch());
                }
            }
            println!();
            continue;
        }

        // 声明式插件：启停、模式、诊断与自述先于一切查询处理。
}

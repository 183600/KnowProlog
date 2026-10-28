{

        // 导入行不立即刷新：进入查询/命令前统一观察整图
        if reasoner_stale
            && !matches!(
                detect_input_mode(input, &language_config.nlp.import_keywords),
                InputMode::KnowledgeImport
            )
        {
            refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
            reasoner_stale = false;
            graph_refreshed = true;
            kb_revision += 1;
            if let Some(chronicle) = chronicle.as_mut() {
                chronicle.set_kb_revision(kb_revision);
            }
        }
        // 对话层的确认/更正必须成为推理层前提：刷新图谱后再同步，避免被重建覆盖。
        if dialogue_enabled {
            sync_recovery_confirmed_facts(&dialogue_manager, &mut symbolic_reasoner);
        }

        // 编年史写前日志：用户输入先落盘再处理；记录当轮开始时的焦点与对话水位，
        // 重启后据此回放/重建上下文（崩溃语义：最多丢进行中的一轮）。
        if !raw_input.is_empty() {
            if let Some(chronicle) = chronicle.as_mut() {
                let mode = match detect_input_mode(raw_input, &language_config.nlp.import_keywords)
                {
                    InputMode::KnowledgeImport => "import",
                    InputMode::Query => "query",
                };
                let kind = if is_command_like(raw_input) {
                    "command"
                } else {
                    mode
                };
                let focus = dialogue_manager.focus_entity();
                // 写端只追加词面与写时身份；合并/改名由读端按当前注册表折叠。
                let refs = if kind == "command" {
                    Vec::new()
                } else {
                    chronicle_entity_refs(&symbolic_reasoner, raw_input)
                };
                let _ = chronicle.append_turn_with_refs(
                    raw_input,
                    kind,
                    focus,
                    dialogue_manager.turn_count(),
                    &refs,
                );
            }
        }

        // 关系获取：把「答不出」编译成带期望的槽位问题（以问代抽）。
        if let Some(raw) = strip_any_prefix(
            input,
            &[
                "关系获取：",
                "关系获取:",
                "问答获取：",
                "问答获取:",
                "获取关系：",
                "获取关系:",
            ],
        ) {
            let raw = raw.trim();
            if raw.is_empty() {
                println!("用法：关系获取：<实体> 的 <谓词>，例如「关系获取：姚明的女儿是谁」。");
            } else {
                match acquisition_context(&oxigraph_manager, &language_config).await {
                    Ok((facts, registry, types, shapes)) => {
                        let opening =
                            acquisition.start_command(raw, &facts, &registry, &types, &shapes);
                        println!("{}", opening.render_text());
                        persist_acquisition(&acquisition, &acquisition_path);
                    }
                    Err(error) => eprintln!("加载关系获取上下文失败：{}", error),
                }
            }
            println!();
            continue;
        }

        // 形状缺口 → 下一个问题：把节点的必备属性缺口编译成槽位追问。
        if matches!(input, "获取缺口" | "关系缺口" | "追问缺口" | "缺口追问") {
            match acquisition_context(&oxigraph_manager, &language_config).await {
                Ok((facts, registry, types, shapes)) => {
                    let report = shapes.validate(&facts, &registry, &types);
                    let mut opened = None;
                    for violation in &report.violations {
                        if let RepairAction::AskProperty { entity, property } = &violation.repair {
                            let candidate = acquisition.start_shape_gap(
                                entity, property, &facts, &registry, &types, &shapes,
                            );
                            if matches!(&candidate, GoalOpen::Opened { .. }) {
                                opened = Some(candidate);
                                break;
                            }
                        }
                    }
                    match opened {
                        Some(opening) => {
                            println!("{}", opening.render_text());
                            persist_acquisition(&acquisition, &acquisition_path);
                        }
                        None => {
                            println!("形状校验没有可追问的必备属性缺口（或缺口值已在覆盖层）。")
                        }
                    }
                }
                Err(error) => eprintln!("加载形状缺口失败：{}", error),
            }
            println!();
            continue;
        }

        // 关系获取台账：候选 / 已确认 / 冲突 / 否证 + 覆盖层 + 模式库
        // 关系获取台账：候选 / 已确认 / 冲突 / 否证 + 覆盖层 + 模式库
        if matches!(
            input,
            "获取台账" | "获取报告" | "关系获取报告" | "问答获取报告" | "获取状态"
        ) {
            println!("{}", acquisition.report());
            println!();
            continue;
        }

        // 统一模式库（Pattern IR）：配置查询 + NLU 意图 + 教学模式同库视图
        if matches!(input, "模式路由" | "查询模式库" | "统一模式库") {
            println!("{}", pattern_ir::render_library_report(&pattern_library));
            let findings = pattern_ir::lint(&pattern_library);
            if findings.is_empty() {
                println!("体检：无重复/冲突/遮蔽。");
            } else {
                println!("体检：{} 条待处理", findings.len());
                for finding in findings.iter().take(12) {
                    println!("  - {}", finding.detail);
                }
            }
            println!();
            continue;
        }

        // 模式关系图：泛化/冗余/冲突落成可查询的边
        if matches!(
            input,
            "模式关系" | "查询模式关系" | "模式关系图" | "模式图体检"
        ) {
            println!("{}", pattern_ir::render_relation_report(&pattern_library));
            println!();
            continue;
        }

        // 模式经验：命中/未命中的可检索记忆（独立命名图）
        if matches!(input, "模式经验" | "模式经验报告" | "经验台账") {
            println!("{}", pattern_ir::render_episode_report(&pattern_episodes));
            println!();
            continue;
        }

        // 模式联想：用相似成功经验泛化出候选骨架，只确认、不直接路由
        if let Some(text) = strip_any_prefix(
            input,
            &[
                "模式联想：",
                "模式联想:",
                "模式联想 ",
                "案例联想：",
                "案例联想:",
            ],
        ) {
            let text = text.trim();
            if text.is_empty() {
                println!("用法：模式联想：<句子>，例如「模式联想：德国的首都在哪里」。");
            } else {
                let context = dialogue_pattern_context(&dialogue_manager, &symbolic_reasoner);
                match suggest_case_for_text(
                    text,
                    &language_config,
                    &synonyms,
                    &pattern_episodes,
                    &context,
                ) {
                    Some(suggestion) => {
                        println!("{}", pattern_ir::render_case_suggestion(text, &suggestion));
                        println!("（可直接回答「是」确认，或回答「不是」否定这次联想。）");
                        pending_case = Some((text.to_string(), suggestion));
                    }
                    None => println!(
                        "模式联想：没有足够相似的成功经验；先处理几条同类问句，或用「教模式」显式教学。"
                    ),
                }
            }
            println!();
            continue;
        }

        // 确认联想：把当前输入的候选骨架转正为可路由模式并落图
        if let Some(text) = strip_any_prefix(
            input,
            &[
                "确认联想：",
                "确认联想:",
                "确认联想 ",
                "采纳联想：",
                "采纳联想:",
                "采纳联想 ",
                "案例转正：",
                "案例转正:",
                "案例转正 ",
            ],
        ) {
            let text = text.trim();
            if text.is_empty() {
                println!("用法：确认联想：<句子>，例如「确认联想：德国的首都在哪里」。");
            } else {
                let context = dialogue_pattern_context(&dialogue_manager, &symbolic_reasoner);
                match suggest_case_for_text(
                    text,
                    &language_config,
                    &synonyms,
                    &pattern_episodes,
                    &context,
                ) {
                    Some(suggestion) => match pattern_library.accept_case(&suggestion) {
                        Ok(entry) => {
                            println!(
                                "已确认案例：{}（来源 {}，路由 {}，优先级 {}）。",
                                entry.name,
                                entry.source,
                                entry.query_type,
                                entry.priority
                            );
                            println!("候选模式：{}", entry.render());
                            match oxigraph_manager
                                .sync_pattern_ledger(pattern_library.entries())
                                .await
                            {
                                Ok(count) => {
                                    println!(
                                        "模式落图：{} 条已同步到 {}。",
                                        count,
                                        pattern_ir::PATTERNS_GRAPH
                                    )
                                }
                                Err(error) => println!("确认后落图失败：{}", error),
                            }
                        }
                        Err(error) => println!("确认联想失败：{}", error),
                    },
                    None => println!(
                        "确认联想：没有足够相似的成功经验；请先用「模式联想：<句子>」确认候选，或使用「教模式」显式教学。"
                    ),
                }
            }
            println!();
            continue;
        }

        // 忘经验：按短 id / 完整 id / 原句删除单条经验（撤销误确认或误否定）
        if let Some(needle) = strip_any_prefix(
            input,
            &["忘经验：", "忘经验:", "忘经验 ", "遗忘经验：", "遗忘经验:"],
        ) {
            let needle = needle.trim();
            if needle.is_empty() {
                println!("用法：忘经验：<短 id 或原句>，短 id 见「模式经验」。");
                println!();
                continue;
            }
            let removed = pattern_episodes.forget(needle);
            match oxigraph_manager
                .sync_pattern_episodes(pattern_episodes.entries())
                .await
            {
                Ok(count) => println!("已删除 {} 条模式经验（经验落图：{} 条）。", removed, count),
                Err(error) => println!("删除 {} 条模式经验，但经验落图失败：{}", removed, error),
            }
            println!();
            continue;
        }

        if matches!(input, "清空模式经验" | "遗忘模式经验") {
            pattern_episodes.clear();
            match oxigraph_manager
                .sync_pattern_episodes(pattern_episodes.entries())
                .await
            {
                Ok(count) => println!("已清空模式经验（经验落图：{} 条）。", count),
                Err(error) => println!("清空模式经验失败：{}", error),
            }
            println!();
            continue;
        }

        // 模式落图：把当前模式库与关系边同步到 kg:graphs/patterns 并读回校验
        if matches!(input, "模式落图" | "同步模式图" | "保存模式库") {
            match oxigraph_manager
                .sync_pattern_ledger(pattern_library.entries())
                .await
            {
                Ok(count) => {
                    println!(
                        "模式落图：{} 条已同步到 {}。",
                        count,
                        pattern_ir::PATTERNS_GRAPH
                    );
                    match oxigraph_manager.load_pattern_ledger().await {
                        Ok(loaded) => {
                            println!("读回校验：{} 条条目。", loaded.len());
                        }
                        Err(error) => println!("读回失败：{}", error),
                    }
                    let edges = pattern_ir::relation_edges(pattern_library.entries());
                    println!(
                        "关系边：{} 条（泛化/冗余/冲突各按图查询可复核）。",
                        edges.len()
                    );
                }
                Err(error) => println!("模式落图失败：{}", error),
            }
            match oxigraph_manager
                .sync_pattern_episodes(pattern_episodes.entries())
                .await
            {
                Ok(count) => {
                    println!(
                        "经验落图：{} 条已同步到 {}。",
                        count,
                        pattern_ir::EPISODES_GRAPH
                    );
                    match oxigraph_manager.load_pattern_episodes().await {
                        Ok(loaded) => println!("经验读回校验：{} 条。", loaded.len()),
                        Err(error) => println!("经验读回失败：{}", error),
                    }
                }
                Err(error) => println!("经验落图失败：{}", error),
            }
            println!();
            continue;
        }

        // 软路由解释：精确路由失败时，哪些近失解释在竞争、差多少、为何接受/澄清/回退
        if let Some(text) = strip_any_prefix(
            input,
            &["软路由：", "软路由:", "软路由 ", "软匹配：", "软匹配:"],
        ) {
            let text = text.trim();
            if text.is_empty() {
                println!("用法：软路由：<句子>，例如「软路由：法国首都是哪里」。");
            } else {
                let tagged = preprocess(text, &language_config.nlp);
                let normalized = normalize_words(tagged, &synonyms);
                let context = dialogue_pattern_context(&dialogue_manager, &symbolic_reasoner);
                let outcome = know_prolog::soft_route::route_tokens(
                    text,
                    &pattern_library,
                    &normalized,
                    &context,
                    &language_config.reasoning.soft_route,
                );
                println!("{}", outcome.render_report(text));
                let exact = pattern_library
                    .match_route_with_context(&normalized, &context)
                    .or_else(|| pattern_library.match_fallback_with_context(&normalized, &context));
                match exact {
                    Some(matched) => println!(
                        "精确路由已命中「{}」：软路由仅作解释，不会改变回答路径。",
                        matched.entry.name
                    ),
                    None => println!(
                        "精确路由未命中；自动接管：{}。",
                        if language_config.reasoning.soft_route.enabled {
                            "开启"
                        } else {
                            "关闭（仅手动查看）"
                        }
                    ),
                }
            }
            println!();
            continue;
        }

        // 软路由体检：配置构念变体上的恢复率、误接受与澄清账目（离线，不打开存储）
        if matches!(input, "软路由体检" | "软路由评估" | "软匹配体检") {
            println!(
                "{}",
                know_prolog::soft_route::evaluate_constructed(&language_config).render_text()
            );
            println!();
            continue;
        }

        if matches!(input, "意图续体" | "意图澄清" | "待裁决意图" | "pending-intent") {
            println!("{}", intent_continuation.render_report());
            println!("（状态文件：{}）", intent_continuation_path);
            println!();
            continue;
        }

        if matches!(
            input,
            "放弃意图澄清" | "取消意图澄清" | "清空意图续体" | "clear-pending-intent"
        ) {
            match intent_continuation.cancel() {
                Some(pending) => {
                    println!("已放弃「{}」的意图澄清。", pending.original_input);
                    if let Err(error) =
                        intent_continuation.save_to_path(&intent_continuation_path)
                    {
                        eprintln!("意图续体保存失败：{}", error);
                    }
                }
                None => println!("当前没有待裁决意图。"),
            }
            println!();
            continue;
        }

        // 模式匹配解释：对一句话给出胜者、命中榜与落选原因
        if let Some(text) = strip_any_prefix(
            input,
            &[
                "模式匹配：",
                "模式匹配:",
                "模式匹配 ",
                "匹配模式：",
                "匹配模式:",
            ],
        ) {
            let text = text.trim();
            if text.is_empty() {
                println!("用法：模式匹配：<句子>，例如「模式匹配：法国的首都是哪里」。");
            } else {
                let tagged = preprocess(text, &language_config.nlp);
                let normalized = normalize_words(tagged, &synonyms);
                let context = dialogue_pattern_context(&dialogue_manager, &symbolic_reasoner);
                let report = pattern_library.rank_with_context(&normalized, &context);
                println!("{}", pattern_ir::render_match_report(text, &report));
                let scan = pattern_library.scan_route_with_context(&normalized, &context);
                println!(
                    "快速路由扫描：完整评估 {} 条，首要素预筛跳过 {} 条（可路由 {} 条，扫描 {} 条）。",
                    scan.evaluated,
                    scan.skipped,
                    pattern_library.route_count(),
                    scan.scanned()
                );
            }
            println!();
            continue;
        }

        if matches!(input, "批量教学" | "批量教学台账" | "教学规则" | "属性教学")
        {
            println!("{}", batch_teaching.render_ledger());
            println!();
            continue;
        }
        if let Some(selector) = strip_any_prefix(
            input,
            &[
                "采纳教学：",
                "采纳教学:",
                "采纳教学 ",
                "确认教学：",
                "确认教学:",
                "确认教学 ",
            ],
        ) {
            let selector = selector.trim();
            if selector.is_empty() {
                println!("用法：采纳教学 <规则id>（用「批量教学」查看；all 采纳全部可采纳候选）。");
            } else {
                match batch_teaching.adopt(selector, &language_config) {
                    Ok(ids) if ids.is_empty() => println!("没有可采纳的候选规则。"),
                    Ok(ids) => match batch_teaching.save(&batch_teaching_path) {
                        Ok(()) => println!(
                            "已采纳 {} 条批量教学规则：{}（立即用于文本录入）。",
                            ids.len(),
                            ids.join(", ")
                        ),
                        Err(error) => eprintln!(
                            "采纳已生效，但写入台账失败（{}）：{}",
                            batch_teaching_path, error
                        ),
                    },
                    Err(error) => println!("采纳教学失败：{}", error),
                }
            }
            println!();
            continue;
        }
        if let Some(selector) = strip_any_prefix(
            input,
            &[
                "退役教学：",
                "退役教学:",
                "退役教学 ",
                "撤销教学：",
                "撤销教学:",
                "撤销教学 ",
            ],
        ) {
            let selector = selector.trim();
            if selector.is_empty() {
                println!("用法：退役教学 <规则id>（用「批量教学」查看活动规则）。");
            } else {
                match batch_teaching.retire(selector) {
                    Ok(ids) if ids.is_empty() => println!("没有匹配的规则。"),
                    Ok(ids) => match batch_teaching.save(&batch_teaching_path) {
                        Ok(()) => println!("已退役 {} 条规则：{}。", ids.len(), ids.join(", ")),
                        Err(error) => eprintln!(
                            "退役已生效，但写入台账失败（{}）：{}",
                            batch_teaching_path, error
                        ),
                    },
                    Err(error) => println!("退役教学失败：{}", error),
                }
            }
            println!();
            continue;
        }

        // 锻造模式：从正例/反例归纳 token 正则，候选需显式确认才参与路由。
        if matches!(input, "锻造模式" | "模式锻造") {
            println!("用法：锻造模式：<名> | <正例(多个用;分隔)> | <反例(可空，多个用;分隔)> | <路由> | <优先级(可选)> | <守卫(可选)>");
            println!("示例：锻造模式：女儿问法 | 姚明的女儿是谁 | 姚明今年几岁 | daughter_query");
            println!();
            continue;
        }
        if let Some(spec) = strip_any_prefix(
            input,
            &[
                "锻造模式：",
                "锻造模式:",
                "锻造模式 ",
                "模式锻造：",
                "模式锻造:",
            ],
        ) {
            match pattern_forge::parse_forge_spec(spec.trim())
                .and_then(|request| pattern_forge.forge(&request, &language_config.nlp))
            {
                Ok(outcome) => {
                    println!("{}", outcome.render());
                    persist_pattern_forge(&pattern_forge, &forge_path);
                }
                Err(error) => println!("锻造失败：{}", error),
            }
            println!();
            continue;
        }

        // 确认锻造：候选转正，展开为无分支模式并入统一模式库。
        if let Some(rest) = strip_any_prefix(
            input,
            &[
                "确认锻造：",
                "确认锻造:",
                "确认锻造 ",
                "采纳锻造：",
                "采纳锻造:",
            ],
        ) {
            let key = rest
                .trim_start_matches(|ch: char| ch == ' ' || ch == '：' || ch == ':')
                .trim();
            if key.is_empty() {
                println!("用法：确认锻造 <id>（先用「锻造台账」查看候选 id）。");
            } else {
                match pattern_forge.activate(key) {
                    Ok(activation) => {
                        let backend_known = pattern_library.entries().iter().any(|entry| {
                            entry.route
                                && entry.query_type == activation.pattern.query_type
                                && !entry.source.starts_with(pattern_forge::FORGE_SOURCE_PREFIX)
                        });
                        pattern_library.remove_source(&activation.pattern.source());
                        for entry in &activation.entries {
                            pattern_library.add(entry.clone());
                        }
                        println!(
                            "已激活锻造规则 {}「{}」：{} 条展开参与路由（仅本会话来源，可用「退役锻造」撤销）。",
                            activation.pattern.id,
                            activation.pattern.name,
                            activation.entries.len()
                        );
                        if !backend_known {
                            println!(
                                "  提示：路由「{}」目前没有其它可执行模式。锻造只让这类问法被认出，后端查询仍需在 config.toml 或 PL 规则中提供。",
                                activation.pattern.query_type
                            );
                        }
                        for entry in &activation.entries {
                            println!("  - {}（{}）", entry.name, entry.render());
                        }
                        match oxigraph_manager
                            .sync_pattern_ledger(pattern_library.entries())
                            .await
                        {
                            Ok(count) => println!("模式落图：{} 条已同步。", count),
                            Err(error) => eprintln!("模式落图失败：{}", error),
                        }
                        persist_pattern_forge(&pattern_forge, &forge_path);
                    }
                    Err(error) => println!("确认锻造失败：{}", error),
                }
            }
            println!();
            continue;
        }

        // 否认锻造：记录一次反驳；达到阈值自动退役并撤出模式库。
        if let Some(rest) = strip_any_prefix(
            input,
            &[
                "否认锻造：",
                "否认锻造:",
                "否认锻造 ",
                "否决锻造：",
                "否决锻造:",
            ],
        ) {
            let key = rest
                .trim_start_matches(|ch: char| ch == ' ' || ch == '：' || ch == ':')
                .trim();
            match pattern_forge.refute(key) {
                Ok((pattern, retired)) => {
                    if retired {
                        let removed = pattern_library.remove_source(&pattern.source());
                        println!(
                            "锻造规则 {}「{}」累计被反驳 {} 次，已自动退役并移除 {} 条展开。",
                            pattern.id, pattern.name, pattern.refutations, removed
                        );
                        match oxigraph_manager
                            .sync_pattern_ledger(pattern_library.entries())
                            .await
                        {
                            Ok(count) => println!("模式落图：{} 条已同步。", count),
                            Err(error) => eprintln!("模式落图失败：{}", error),
                        }
                    } else {
                        println!(
                            "已记下对锻造规则 {}「{}」的反驳：{}/{} 次。",
                            pattern.id,
                            pattern.name,
                            pattern.refutations,
                            pattern_forge::MAX_REFUTATIONS
                        );
                    }
                    persist_pattern_forge(&pattern_forge, &forge_path);
                }
                Err(error) => println!("否认锻造失败：{}", error),
            }
            println!();
            continue;
        }

        // 退役锻造：停止路由并保留出生档案（可再遗忘）。
        if let Some(rest) = strip_any_prefix(
            input,
            &[
                "退役锻造：",
                "退役锻造:",
                "退役锻造 ",
                "停用锻造：",
                "停用锻造:",
            ],
        ) {
            let key = rest
                .trim_start_matches(|ch: char| ch == ' ' || ch == '：' || ch == ':')
                .trim();
            match pattern_forge.retire(key) {
                Ok(pattern) => {
                    let removed = pattern_library.remove_source(&pattern.source());
                    println!(
                        "已退役锻造规则 {}「{}」；从模式库移除 {} 条展开。",
                        pattern.id, pattern.name, removed
                    );
                    match oxigraph_manager
                        .sync_pattern_ledger(pattern_library.entries())
                        .await
                    {
                        Ok(count) => println!("模式落图：{} 条已同步。", count),
                        Err(error) => eprintln!("模式落图失败：{}", error),
                    }
                    persist_pattern_forge(&pattern_forge, &forge_path);
                }
                Err(error) => println!("退役锻造失败：{}", error),
            }
            println!();
            continue;
        }

        // 遗忘锻造：彻底删除非激活记录（激活规则必须先退役）。
        if let Some(rest) = strip_any_prefix(
            input,
            &[
                "遗忘锻造：",
                "遗忘锻造:",
                "遗忘锻造 ",
                "删除锻造：",
                "删除锻造:",
            ],
        ) {
            let key = rest
                .trim_start_matches(|ch: char| ch == ' ' || ch == '：' || ch == ':')
                .trim();
            match pattern_forge.forget(key) {
                Ok(pattern) => {
                    let removed = pattern_library.remove_source(&pattern.source());
                    println!(
                        "已遗忘锻造规则 {}「{}」（移除 {} 条展开）。",
                        pattern.id, pattern.name, removed
                    );
                    match oxigraph_manager
                        .sync_pattern_ledger(pattern_library.entries())
                        .await
                    {
                        Ok(count) => println!("模式落图：{} 条已同步。", count),
                        Err(error) => eprintln!("模式落图失败：{}", error),
                    }
                    persist_pattern_forge(&pattern_forge, &forge_path);
                }
                Err(error) => println!("遗忘锻造失败：{}", error),
            }
            println!();
            continue;
        }

        // 锻造探针：重新跑正例/反例/逆生成句，防止正则在分词变化后悄悄失效。
        if matches!(input, "锻造探针" | "复核锻造" | "锻造探针报告") {
            match pattern_forge.verify_probes(None, &language_config.nlp) {
                Ok(report) => {
                    println!("{}", report.render(None));
                    persist_pattern_forge(&pattern_forge, &forge_path);
                }
                Err(error) => println!("锻造探针失败：{}", error),
            }
            println!();
            continue;
        }
        if let Some(rest) = strip_any_prefix(
            input,
            &[
                "锻造探针：",
                "锻造探针:",
                "锻造探针 ",
                "复核锻造：",
                "复核锻造:",
            ],
        ) {
            let key = rest
                .trim_start_matches(|ch: char| ch == ' ' || ch == '：' || ch == ':')
                .trim();
            let key = (!key.is_empty()).then_some(key);
            match pattern_forge.verify_probes(key, &language_config.nlp) {
                Ok(report) => {
                    println!("{}", report.render(key));
                    persist_pattern_forge(&pattern_forge, &forge_path);
                }
                Err(error) => println!("锻造探针失败：{}", error),
            }
            println!();
            continue;
        }

        if matches!(input, "锻造台账" | "锻造报告" | "锻造状态" | "模式锻造台账")
        {
            println!("{}", pattern_forge.report());
            println!();
            continue;
        }

        // 教模式：<名> | <要素 要素 …> | <路由> | <优先级>
        if let Some(spec) = strip_any_prefix(
            input,
            &["教模式：", "教模式:", "教模式 ", "教学模式：", "教学模式:"],
        ) {
            let parts: Vec<&str> = spec.split('|').map(str::trim).collect();
            if parts.len() < 3 {
                println!("用法：教模式：<名> | <要素 要素 …> | <路由> | <优先级(可选)> | <守卫(可选)>\n  要素：词性码（nr/ns/n/m/v…）、词面、{{槽名}}、{{槽名:排序}}、_ 任意\n  守卫：焦点 / 焦点:人物 / 上轮:ask_attribute / 待澄清 / 无待澄清，多项用逗号分隔");
            } else {
                let name = parts[0];
                let pattern = parts[1];
                let route = parts[2];
                let priority = parts
                    .get(3)
                    .and_then(|p| p.parse::<i32>().ok())
                    .unwrap_or(10);
                let guard = match parts.get(4).filter(|part| !part.is_empty()) {
                    Some(spec) => match pattern_ir::parse_context_guard(spec) {
                        Ok(guard) => Some(guard),
                        Err(error) => {
                            println!("教学模式失败：{}", error);
                            println!();
                            continue;
                        }
                    },
                    None => None,
                };
                match pattern_library.teach_with_guard(name, pattern, route, priority, guard) {
                    Ok(entry) => {
                        println!(
                            "已教模式「{}」→ {}（优先级 {}，IR {}{}）",
                            entry.name,
                            entry.query_type,
                            entry.priority,
                            entry.render(),
                            entry
                                .guard
                                .as_ref()
                                .map(|guard| format!("，守卫：{}", guard.render()))
                                .unwrap_or_default()
                        );
                        match oxigraph_manager
                            .sync_pattern_ledger(pattern_library.entries())
                            .await
                        {
                            Ok(count) => println!("模式落图：{} 条已同步。", count),
                            Err(error) => eprintln!("模式落图失败：{}", error),
                        }
                    }
                    Err(error) => println!("教学模式失败：{}", error),
                }
            }
            println!();
            continue;
        }

        // 忘模式：<名 或 id 前缀>（只删教学模式来源）
        if let Some(name) = strip_any_prefix(
            input,
            &["忘模式：", "忘模式:", "忘模式 ", "遗忘模式：", "遗忘模式:"],
        ) {
            let name = name.trim();
            if name.is_empty() {
                println!("用法：忘模式：<名 或 id 前缀>");
            } else {
                match pattern_library.forget(name) {
                    Some(entry) => {
                        println!(
                            "已遗忘教学模式「{}」（IR {}）。",
                            entry.name,
                            entry.render()
                        );
                        match oxigraph_manager
                            .sync_pattern_ledger(pattern_library.entries())
                            .await
                        {
                            Ok(count) => println!("模式落图：{} 条已同步。", count),
                            Err(error) => eprintln!("模式落图失败：{}", error),
                        }
                    }
                    None => println!("没有找到可遗忘的教学模式「{}」（只删教学来源）。", name),
                }
            }
            println!();
            continue;
        }

        if matches!(input, "获取模式" | "关系获取模式" | "诱导模式" | "模式库") {
            println!("{}", acquisition.patterns_report());
            println!();
            continue;
        }

        if matches!(input, "获取覆盖" | "会话覆盖层" | "覆盖层") {
            println!("{}", acquisition.overlay().render_text());
            println!();
            continue;
        }

        if matches!(
            input,
            "断言阅读" | "断言阅读报告" | "待确认断言" | "提及卡" | "断言台账"
        ) {
            println!("{}", assertion_reader.report());
            println!();
            continue;
        }

        if matches!(input, "清空断言" | "取消断言" | "放弃断言") {
            assertion_reader.clear_pending();
            persist_assertion_reader(&assertion_reader, &assertion_reader_path);
            println!("已清除待确认断言（会话覆盖层与知识图谱不变）。");
            println!();
            continue;
        }

        if matches!(input, "确认断言" | "确认当前断言") {
            match assertion_reader.pending_claim_id().map(str::to_string) {
                Some(claim_id) => {
                    let outcome = acquisition.confirm(Some(&claim_id));
                    println!("{}", outcome.message);
                    if let Some(proposal) = outcome.commit.as_ref() {
                        let answer = symbolic_reasoner.assert_nl_fact(&proposal.sentence);
                        if acquisition.mark_admitted(&proposal.claim_id) {
                            println!(
                                "已记入会话事实：{}（未写入知识图谱）。",
                                proposal.sentence
                            );
                        }
                        log_chronicle_reply(&mut chronicle, &answer, &[], &[]);
                    }
                    assertion_reader.clear_pending();
                    persist_acquisition(&acquisition, &acquisition_path);
                    persist_assertion_reader(&assertion_reader, &assertion_reader_path);
                }
                None => println!("当前没有待确认的断言（可先输入自由陈述句）。"),
            }
            println!();
            continue;
        }

        if matches!(input, "否认断言" | "否认当前断言") {
            match assertion_reader.pending_claim_id().map(str::to_string) {
                Some(claim_id) => {
                    let outcome = acquisition.deny(Some(&claim_id));
                    println!("{}", outcome.message);
                    assertion_reader.clear_pending();
                    persist_acquisition(&acquisition, &acquisition_path);
                    persist_assertion_reader(&assertion_reader, &assertion_reader_path);
                }
                None => println!("当前没有待否认的断言。"),
            }
            println!();
            continue;
        }

        if let Some(rest) = strip_any_prefix(input, &["确认获取", "确认关系获取", "确认问答获取"])
        {
            let id = rest
                .trim_start_matches(|ch: char| ch == ' ' || ch == '：' || ch == ':')
                .trim();
            let outcome = acquisition.confirm(if id.is_empty() { None } else { Some(id) });
            println!("{}", outcome.message);
            if let Some(follow_up) = outcome.follow_up {
                println!("{}", follow_up);
            }
            persist_acquisition(&acquisition, &acquisition_path);
            println!();
            continue;
        }

        if let Some(rest) = strip_any_prefix(input, &["否认获取", "否认关系获取", "拒绝获取"])
        {
            let id = rest
                .trim_start_matches(|ch: char| ch == ' ' || ch == '：' || ch == ':')
                .trim();
            let outcome = acquisition.deny(if id.is_empty() { None } else { Some(id) });
            println!("{}", outcome.message);
            persist_acquisition(&acquisition, &acquisition_path);
            println!();
            continue;
        }

        if let Some(rest) = strip_any_prefix(input, &["撤回获取", "撤销获取", "撤回关系获取"])
        {
            let id = rest
                .trim_start_matches(|ch: char| ch == ' ' || ch == '：' || ch == ':')
                .trim();
            let outcome = acquisition.retract(if id.is_empty() { None } else { Some(id) });
            println!("{}", outcome.message);
            persist_acquisition(&acquisition, &acquisition_path);
            println!();
            continue;
        }

        // 采纳：Confirmed/Admitted 的条目进入会话事实集（仍不写知识图谱）。
        if let Some(rest) = strip_any_prefix(input, &["采纳获取", "采纳关系获取", "提交获取"])
        {
            let id = rest
                .trim_start_matches(|ch: char| ch == ' ' || ch == '：' || ch == ':')
                .trim();
            match acquisition.commit_proposal(if id.is_empty() { None } else { Some(id) }) {
                Some(proposal) => {
                    let answer = symbolic_reasoner.assert_nl_fact(&proposal.sentence);
                    println!("{}", answer);
                    if let Some(deferred) = symbolic_reasoner.resolve_pending_after_assertion() {
                        println!("{}", deferred);
                    }
                    if acquisition.mark_admitted(&proposal.claim_id) {
                        println!(
                            "已采纳 {}：{}（会话覆盖层仍可回溯；未写入知识图谱）。",
                            proposal.claim_id, proposal.sentence
                        );
                    }
                    log_chronicle_reply(&mut chronicle, &answer, &[], &[]);
                    persist_acquisition(&acquisition, &acquisition_path);
                }
                None => println!(
                    "没有可采纳的条目；只有「已确认」或「已采纳」的关系才能进入会话事实集。"
                ),
            }
            println!();
            continue;
        }

        // 对话诱导模式扫描：只有被两次确认激活的模式才会命中。
        if let Some(text) = strip_any_prefix(input, &["模式扫描：", "模式扫描:", "模式扫描 "])
        {
            let text = text.trim();
            if text.is_empty() {
                println!("用法：模式扫描：<文本>，例如「模式扫描：王强的女儿是赵敏」。");
            } else {
                match acquisition_context(&oxigraph_manager, &language_config).await {
                    Ok((facts, registry, types, shapes)) => {
                        let outcome = acquisition
                            .absorb_pattern_scan(text, &facts, &registry, &types, &shapes);
                        println!("{}", outcome.message);
                        persist_acquisition(&acquisition, &acquisition_path);
                    }
                    Err(error) => eprintln!("加载模式扫描上下文失败：{}", error),
                }
            }
            println!();
            continue;
        }

        // 归纳模式：把已激活的对话诱导模式（DIP）提升为统一模式库条目。
}

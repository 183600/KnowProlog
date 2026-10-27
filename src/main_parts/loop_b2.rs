{
        if matches!(input, "归纳模式" | "诱导入库" | "模式入库" | "归纳规律") {
            let active: Vec<pattern_ir::PatternEntry> = acquisition
                .patterns()
                .filter(|pattern| pattern.active && !pattern.retired)
                .map(|pattern| {
                    let ir = format!("{{主体}} 的 {} 是 {{客体}}", pattern.surface);
                    pattern.to_pattern_entry(&ir)
                })
                .collect();
            if active.is_empty() {
                println!("没有可归纳的激活模式（同一谓词被两次确认后自动激活）。");
            } else {
                let mut added = 0usize;
                for entry in &active {
                    if pattern_library.add(entry.clone()) {
                        added += 1;
                    }
                }
                println!(
                    "归纳入库：{} 条激活诱导模式，新增 {} 条（来源 induced#…）。",
                    active.len(),
                    added
                );
                for entry in &active {
                    println!(
                        "  - {}（{}）→ {}",
                        entry.name,
                        entry.render(),
                        entry.query_type
                    );
                }
                match oxigraph_manager
                    .sync_pattern_ledger(pattern_library.entries())
                    .await
                {
                    Ok(count) => println!("模式落图：{} 条已同步。", count),
                    Err(error) => eprintln!("模式落图失败：{}", error),
                }
            }
            println!();
            continue;
        }

        if let Some(id) = strip_any_prefix(input, &["退役模式：", "退役模式:", "退役模式 "])
        {
            let id = id.trim();
            if acquisition.retire_pattern(id) {
                println!("已退役诱导模式 {}。", id);
                persist_acquisition(&acquisition, &acquisition_path);
            } else {
                println!("没有找到诱导模式 {}。", id);
            }
            println!();
            continue;
        }

        // 形状填充对话：把形状报告编译成议程，问到缺口后回到原问题。
        if let Some(entity) = strip_any_prefix(
            input,
            &["形状画像：", "形状画像:", "对话画像：", "对话画像:"],
        ) {
            let entity = entity.trim();
            if entity.is_empty() {
                println!("用法：形状画像：<实体>，例如「形状画像：刘德华」。");
            } else {
                match acquisition_context(&oxigraph_manager, &language_config).await {
                    Ok((facts, registry, types, shapes)) => {
                        let question = format!("{}是谁？", entity);
                        let turn = shape_dialogue.start_profile(
                            entity,
                            &question,
                            &facts,
                            &registry,
                            &types,
                            &shapes,
                            &mut acquisition,
                        );
                        println!("{}", turn.message);
                        persist_acquisition(&acquisition, &acquisition_path);
                        persist_shape_dialogue(&shape_dialogue, &shape_dialogue_path);
                    }
                    Err(error) => eprintln!("加载形状画像上下文失败：{}", error),
                }
            }
            println!();
            continue;
        }

        // 会话级冲突裁定：只收敛本会话画像，不写图谱。
        if let Some(body) = strip_any_prefix(
            input,
            &[
                "裁定：",
                "裁定:",
                "冲突裁定：",
                "冲突裁定:",
                "会话裁定：",
                "会话裁定:",
            ],
        ) {
            match know_prolog::shape_dialogue::parse_ruling(body) {
                Some((entity, predicate, value)) => {
                    match acquisition_context(&oxigraph_manager, &language_config).await {
                        Ok((facts, registry, types, shapes)) => {
                            match shape_dialogue.resolve_conflict(
                                &entity,
                                &predicate,
                                &value,
                                &facts,
                                &registry,
                                &types,
                                &shapes,
                                &mut acquisition,
                            ) {
                                Ok(turn) => println!("{}", turn.message),
                                Err(message) => println!("{}", message),
                            }
                            persist_acquisition(&acquisition, &acquisition_path);
                            persist_shape_dialogue(&shape_dialogue, &shape_dialogue_path);
                        }
                        Err(error) => eprintln!("加载形状画像上下文失败：{}", error),
                    }
                }
                None => println!(
                    "用法：裁定：<实体> 的 <谓词> = <值>，例如「裁定：刘德华 的 职业 = 演员」。"
                ),
            }
            println!();
            continue;
        }

        // 口语化裁定：只在上一条画像留下待裁定冲突且没有待答缺口时接管。
        if !shape_dialogue.has_pending()
            && ShapeDialogue::looks_like_conflict_ruling(input)
            && !is_command_like(input)
        {
            match acquisition_context(&oxigraph_manager, &language_config).await {
                Ok((facts, registry, types, shapes)) => {
                    if let Some(result) = shape_dialogue.resolve_conflict_text(
                        input,
                        &facts,
                        &registry,
                        &types,
                        &shapes,
                        &mut acquisition,
                    ) {
                        match result {
                            Ok(turn) => println!("{}", turn.message),
                            Err(message) => println!("{}", message),
                        }
                        persist_acquisition(&acquisition, &acquisition_path);
                        persist_shape_dialogue(&shape_dialogue, &shape_dialogue_path);
                        println!();
                        continue;
                    }
                }
                Err(error) => eprintln!("加载形状画像上下文失败：{}", error),
            }
        }

        if matches!(input, "形状对话报告" | "形状对话状态" | "形状对话") {
            println!("{}", shape_dialogue.report_text());
            println!();
            continue;
        }

        if matches!(input, "形状议程" | "形状缺口议程") {
            match acquisition_context(&oxigraph_manager, &language_config).await {
                Ok((facts, registry, types, shapes)) => {
                    let (ruled_facts, ruled_overlay) = know_prolog::shape_dialogue::apply_rulings(
                        &facts,
                        &acquisition.overlay(),
                        shape_dialogue.rulings(),
                        &registry,
                    );
                    let effective =
                        know_prolog::shape_dialogue::effective_facts(&ruled_facts, &ruled_overlay);
                    let report = shapes.validate(&effective, &registry, &types);
                    let agenda =
                        know_prolog::shape_dialogue::ShapeAgenda::compile(&report, &registry);
                    println!("{}", agenda.render_text());
                }
                Err(error) => eprintln!("加载形状议程失败：{}", error),
            }
            println!();
            continue;
        }

        if matches!(input, "下一个形状问题" | "下一个缺口问题") {
            match acquisition_context(&oxigraph_manager, &language_config).await {
                Ok((facts, registry, types, shapes)) => {
                    match shape_dialogue.ask_next(
                        &facts,
                        &registry,
                        &types,
                        &shapes,
                        &mut acquisition,
                    ) {
                        Some(turn) => {
                            println!("{}", turn.message);
                            persist_acquisition(&acquisition, &acquisition_path);
                            persist_shape_dialogue(&shape_dialogue, &shape_dialogue_path);
                        }
                        None => println!(
                            "形状议程没有可追问的缺口（可能已完成、已达提问上限，或已有待答问题）。"
                        ),
                    }
                }
                Err(error) => eprintln!("加载形状议程失败：{}", error),
            }
            println!();
            continue;
        }

        if input == "形状对话重置" {
            shape_dialogue.reset(&mut acquisition);
            persist_acquisition(&acquisition, &acquisition_path);
            persist_shape_dialogue(&shape_dialogue, &shape_dialogue_path);
            println!("形状填充对话状态已重置（关系获取账本与知识图谱保持不变）。");
            println!();
            continue;
        }

        // 可教断言：上一轮刚问过「对/不对」或「是哪类实体」，本轮短答优先落在这里。
        if assertion_reader.has_pending()
            && !is_command_like(input)
            && !matches!(
                detect_input_mode(input, &language_config.nlp.import_keywords),
                InputMode::KnowledgeImport
            )
        {
            let types_snapshot = oxigraph_manager.type_system_snapshot();
            match assertion_reader.respond(input, &types_snapshot) {
                TarResponse::Confirm { claim_id } => {
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
                    persist_acquisition(&acquisition, &acquisition_path);
                    persist_assertion_reader(&assertion_reader, &assertion_reader_path);
                    println!();
                    continue;
                }
                TarResponse::Deny { claim_id } => {
                    let outcome = acquisition.deny(Some(&claim_id));
                    println!("{}", outcome.message);
                    persist_acquisition(&acquisition, &acquisition_path);
                    persist_assertion_reader(&assertion_reader, &assertion_reader_path);
                    println!();
                    continue;
                }
                TarResponse::TypeResolved {
                    frame,
                    mention,
                    type_name,
                } => {
                    match acquisition_context(&oxigraph_manager, &language_config).await {
                        Ok((facts, registry, types, shapes)) => {
                            let source_text = frame.source_text.clone();
                            let outcome = acquisition.record_assertion(
                                frame.to_claim(),
                                AcquisitionSource::UserAssertion,
                                &source_text,
                                "断言阅读",
                                &facts,
                                &registry,
                                &types,
                                &shapes,
                            );
                            assertion_reader
                                .mark_proposed(frame.clone(), outcome.claim_id.clone());
                            println!(
                                "好，我按「{}」是{}理解。我理解为：{}。要记入本轮会话吗？（对/不对）",
                                mention,
                                type_name,
                                frame.render_text()
                            );
                            if let Some(claim_id) = outcome.claim_id.as_deref() {
                                if let Some(claim) = acquisition.claim(claim_id) {
                                    if !claim.notes.is_empty() {
                                        println!("（{}）", claim.notes.join("；"));
                                    }
                                }
                            }
                            persist_acquisition(&acquisition, &acquisition_path);
                            persist_assertion_reader(&assertion_reader, &assertion_reader_path);
                        }
                        Err(error) => eprintln!("加载断言阅读上下文失败：{}", error),
                    }
                    println!();
                    continue;
                }
                TarResponse::Cancelled => {
                    println!("好，这条关系先不记。");
                    persist_assertion_reader(&assertion_reader, &assertion_reader_path);
                    println!();
                    continue;
                }
                TarResponse::None => {
                    persist_assertion_reader(&assertion_reader, &assertion_reader_path);
                }
            }
        }

        if shape_dialogue.has_pending()
            && !is_command_like(input)
            && RelationAcquisition::looks_like_answer(input)
            && !matches!(
                detect_input_mode(input, &language_config.nlp.import_keywords),
                InputMode::KnowledgeImport
            )
        {
            match acquisition_context(&oxigraph_manager, &language_config).await {
                Ok((facts, registry, types, shapes)) => {
                    if let Some(turn) = shape_dialogue.respond(
                        input,
                        &facts,
                        &registry,
                        &types,
                        &shapes,
                        &mut acquisition,
                    ) {
                        println!("{}", turn.message);
                    }
                    persist_acquisition(&acquisition, &acquisition_path);
                    persist_shape_dialogue(&shape_dialogue, &shape_dialogue_path);
                }
                Err(error) => eprintln!("加载形状画像上下文失败：{}", error),
            }
            println!();
            continue;
        }

        // 待填槽位的短答：先做对话行为分类，再进形状闸门与账本。
        // 命令、导入行与新问题不拦截，保持既有管线不变。
        if acquisition.has_pending_goal()
            && !is_command_like(input)
            && RelationAcquisition::looks_like_answer(input)
            && !matches!(
                detect_input_mode(input, &language_config.nlp.import_keywords),
                InputMode::KnowledgeImport
            )
        {
            match acquisition_context(&oxigraph_manager, &language_config).await {
                Ok((facts, registry, types, shapes)) => {
                    let outcome =
                        acquisition.interpret_answer(input, &facts, &registry, &types, &shapes);
                    println!("{}", outcome.message);
                    if let Some(follow_up) = outcome.follow_up {
                        println!("{}", follow_up);
                    }
                    persist_acquisition(&acquisition, &acquisition_path);
                }
                Err(error) => eprintln!("加载关系获取上下文失败：{}", error),
            }
            println!();
            continue;
        }

        if let Some(term) = input
            .strip_prefix("对齐候选：")
            .or_else(|| input.strip_prefix("对齐候选:"))
        {
            let term = term.trim();
            if term.is_empty() {
                println!("请提供要对齐的术语，例如：对齐候选：老婆");
            } else {
                match load_graph_snapshot(&oxigraph_manager).await {
                    Ok(facts) => {
                        symbolic_reasoner.observe_facts(&facts);
                        println!("{}", symbolic_reasoner.alignment_candidates_report(term));
                    }
                    Err(e) => println!("加载知识图谱失败: {}", e),
                }
            }
            println!();
            continue;
        }

        if input == "对齐报告" || input == "本体对齐" {
            match load_graph_snapshot(&oxigraph_manager).await {
                Ok(facts) => {
                    symbolic_reasoner.observe_facts(&facts);
                    println!("{}", symbolic_reasoner.alignment_report());
                }
                Err(e) => println!("加载知识图谱失败: {}", e),
            }
            println!();
            continue;
        }

        if input == "推理规则" || input == "链接预测规则" {
            match load_graph_snapshot(&oxigraph_manager).await {
                Ok(facts) => {
                    symbolic_reasoner.observe_facts(&facts);
                    println!("{}", symbolic_reasoner.rules_report());
                }
                Err(e) => println!("加载知识图谱失败: {}", e),
            }
            println!();
            continue;
        }

        if let Some(entity) = input
            .strip_prefix("相似实体：")
            .or_else(|| input.strip_prefix("相似实体:"))
            .or_else(|| input.strip_prefix("语义相似："))
            .or_else(|| input.strip_prefix("语义相似:"))
        {
            match load_graph_snapshot(&oxigraph_manager).await {
                Ok(facts) => symbolic_reasoner.observe_facts(&facts),
                Err(e) => println!("加载知识图谱失败: {}", e),
            }
            println!("{}", symbolic_reasoner.similarity_report(entity));
            println!();
            continue;
        }

        // 证据检索：符号化稠密排序 + 精确约束 + 上界剪枝，并给出可审计回执
        if let Some(query) = strip_any_prefix(
            input,
            &["证据检索：", "证据检索:", "检索证据：", "检索证据:"],
        ) {
            let query = query.trim();
            if query.is_empty() {
                println!("请提供要检索的问题，例如：证据检索：张三的师傅是谁");
            } else {
                match load_graph_snapshot(&oxigraph_manager).await {
                    Ok(facts) => {
                        symbolic_reasoner.observe_facts(&facts);
                        println!(
                            "{}",
                            symbolic_reasoner.evidence_report(
                                query,
                                8,
                                &SearchConstraints::default(),
                                SearchBudget::default(),
                            )
                        );
                    }
                    Err(e) => println!("加载知识图谱失败: {}", e),
                }
            }
            println!();
            continue;
        }

        if input == "对话记忆" {
            println!("{}", symbolic_reasoner.dialogue_summary());
            println!();
            continue;
        }

        // 话语帧记忆：带洞查询帧 + 挂账（合一式多轮消解，可审计）
        if matches!(input, "帧记忆" | "话语帧" | "挂账") {
            println!("{}", symbolic_reasoner.frame_memory_report());
            println!();
            continue;
        }

        // 对话本体状态：展示对话管理器实例与显式的策略规则程序
        if input == "对话本体" || input == "对话状态" {
            println!("{}", dialogue_manager.summary());
            println!("{}", DialogueManager::policy_source());
            println!();
            continue;
        }

        // 分层追踪：问据栈 / 指代轨迹 / 逐轮决策（只读；On 模式才会累积）
        if matches!(input, "追踪报告" | "追踪状态" | "追踪图" | "trace") {
            println!("{}", dialogue_manager.trace_report());
            println!();
            continue;
        }

        // 指代轨迹：当前会话的指代绑定与版本链
        if matches!(input, "指代轨迹" | "指代报告") {
            println!("{}", dialogue_manager.trace_store().referent_report());
            println!();
            continue;
        }

        // 问据栈：当前开放/已决的问据（QUD）
        if matches!(input, "问据栈" | "问据报告" | "QUD") {
            println!("{}", dialogue_manager.trace_store().qud_stack().report());
            println!();
            continue;
        }

        // 追踪为什么：沿追踪图 inputs 边回溯某编号节点的决策链
        if let Some(id) = strip_any_prefix(input, &["追踪为什么：", "追踪为什么:", "追踪为什么"])
        {
            match id.trim().parse::<u64>() {
                Ok(id) => match dialogue_manager.trace_why(id) {
                    Some(text) => println!("{}", text),
                    None => println!("追踪图里没有编号为 {} 的节点。", id),
                },
                Err(_) => println!("请给出节点编号，例如「追踪为什么 4」。"),
            }
            println!();
            continue;
        }

        if matches!(input, "查询板" | "查询状态" | "工作记忆") {
            println!("{}", dialogue_manager.board_summary());
            if let Some(frame) = dialogue_manager.board().current() {
                println!(
                    "当前查询：{}；{} 行；筛选/排序/翻页/选行都可用自然语言继续编辑。",
                    frame.program,
                    frame.rows.len()
                );
            } else {
                println!("可以试试：「列出所有人的身高」「哪些实体」。");
            }
            println!();
            continue;
        }

        // 查询剧本：把多轮交互式查询写成可执行的回合轨迹契约（自包含事实，不依赖知识库）
        if let Some(path) = strip_any_prefix(
            input,
            &["查询剧本：", "查询剧本:", "运行剧本：", "运行剧本:"],
        ) {
            let path = path.trim();
            if path.is_empty() {
                println!("用法：查询剧本：<剧本文件.toml>（文件自带实体事实与每轮期望）");
            } else {
                match know_prolog::query_scenario::run_file(path) {
                    Ok(report) => println!("{}", report.render_text()),
                    Err(error) => println!("查询剧本运行失败：{}", error),
                }
            }
            println!();
            continue;
        }

        // 对话编年史：追加式哈希链档案（真相源；快照只是检查点）
        if matches!(input, "编年史" | "对话档案" | "对话编年史") {
            match chronicle.as_ref() {
                Some(chronicle) => {
                    println!(
                        "对话编年史（追加式哈希链写前日志）：{} 条记录，水位 seq {}，链尾 {}，KB 修订 {}",
                        chronicle.len(),
                        chronicle.last_seq(),
                        if chronicle.last_hash().is_empty() {
                            "（空）"
                        } else {
                            chronicle.last_hash()
                        },
                        chronicle.kb_revision()
                    );
                    println!("  文件：{}", chronicle.path().display());
                    let recent: Vec<TimelineTurn> = chronicle
                        .entries()
                        .iter()
                        .rev()
                        .take(8)
                        .rev()
                        .filter_map(TimelineTurn::from_entry)
                        .collect();
                    if !recent.is_empty() {
                        println!("{}", render_timeline_markdown(&recent));
                    }
                }
                None => println!("编年史不可用（打开失败，见启动警告）。"),
            }
            println!();
            continue;
        }

        // 编年史自检：三类损坏（截尾/翻字节/版本拔高）分别可辨
        if matches!(input, "编年史校验" | "档案校验" | "校验编年史") {
            match Chronicle::audit_file(&chronicle_path) {
                Ok(audit) => println!("编年史自检：{}", audit.verdict()),
                Err(error) => println!("编年史自检失败：{}", error),
            }
            println!();
            continue;
        }

        // 回放：重建到第 N 条记录（调试器视图；不改变当前会话状态）
        if input == "回放"
            || input.starts_with("回放 ")
            || input.starts_with("回放：")
            || input.starts_with("回放:")
        {
            let spec = input
                .trim_start_matches("回放")
                .trim_start_matches([' ', '：', ':'])
                .trim();
            match (chronicle.as_ref(), spec.parse::<u64>().ok()) {
                (Some(chronicle), until) => {
                    let until = until.unwrap_or(chronicle.last_seq());
                    let entries = chronicle.replay_to(until);
                    let turns: Vec<TimelineTurn> = entries
                        .iter()
                        .filter_map(|entry| TimelineTurn::from_entry(entry))
                        .collect();
                    println!(
                        "回放到 seq {}（记录 {} 条；当前状态不变，仅视图重建）：",
                        until,
                        entries.len()
                    );
                    println!("{}", render_timeline_markdown(&turns));
                    if (1..=12).contains(&turns.len()) {
                        println!();
                        println!("```mermaid");
                        println!("{}", render_timeline_flow(&turns));
                        println!("```");
                    }
                }
                (None, _) => println!("编年史不可用（打开失败，见启动警告）。"),
            }
            println!();
            continue;
        }

        // 编年史检索：先按实体身份（别名感知、跨合并）查线程，再退回全文命中。
        if let Some(query) =
            strip_any_prefix(input, &["聊过：", "聊过:", "聊过 ", "查档案：", "查档案:"])
        {
            let query = query.trim();
            match (chronicle.as_ref(), query.is_empty()) {
                (Some(chronicle), false) => {
                    let registry = oxigraph_manager.entity_registry_snapshot();
                    let digest = EntityThreadDigest::build(chronicle, &registry);
                    match digest.lookup(query) {
                        ThreadLookup::Found(thread) => {
                            println!(
                                "编年史检索「{}」→ 实体「{}」：{} 条记录（seq {}–{}；别名感知，含读端折叠）：",
                                query,
                                thread.canonical,
                                thread.total_events,
                                thread.first_seq.unwrap_or(0),
                                thread.last_seq.unwrap_or(0)
                            );
                            for event in thread.events.iter().rev().take(8).rev() {
                                println!("  {}", render_event_line(thread, event));
                            }
                            let omitted =
                                thread.omitted_events + thread.events.len().saturating_sub(8);
                            if omitted > 0 {
                                println!(
                                    "  …另有 {} 条较早记录已折叠；用「实体足迹：{}」查看概览。",
                                    omitted, thread.canonical
                                );
                            }
                        }
                        ThreadLookup::Ambiguous(candidates) => {
                            println!(
                                "「{}」可解析到多个身份：{}。请用规范名再查（不会静默归并）。",
                                query,
                                candidates.join(" / ")
                            );
                        }
                        ThreadLookup::Unknown => {
                            let matches = chronicle.search(query);
                            if matches.is_empty() {
                                println!("编年史里没有聊过「{}」。", query);
                            } else {
                                println!(
                                    "编年史检索「{}」：命中 {} 条（按序号，可审计）：",
                                    query,
                                    matches.len()
                                );
                                for item in matches.iter().rev().take(8).rev() {
                                    println!(
                                        "  第 {} 条（{}，命中 {} 次）{}",
                                        item.seq, item.kind, item.hits, item.excerpt
                                    );
                                }
                            }
                        }
                    }
                }
                (Some(_), true) => println!("用法：聊过：<关键词>，例如：聊过：姚明"),
                (None, _) => println!("编年史不可用（打开失败，见启动警告）。"),
            }
            println!();
            continue;
        }

        // 实体足迹：跨会话提及、回答、身份折叠与当前图谱现状。
        if input == "实体足迹"
            || input.starts_with("实体足迹：")
            || input.starts_with("实体足迹:")
            || input.starts_with("实体足迹 ")
        {
            let spec = input
                .trim_start_matches("实体足迹")
                .trim_start_matches([' ', '：', ':'])
                .trim();
            match chronicle.as_ref() {
                Some(chronicle) => {
                    let (digest, facts) = entity_thread_context(chronicle, &oxigraph_manager).await;
                    println!("{}", digest.render_footprint(spec, facts.as_slice()));
                    if let Some(changes) = entity_related_changes(&digest, chronicle, spec) {
                        println!();
                        println!("{}", changes);
                    }
                }
                None => println!("编年史不可用（打开失败，见启动警告）。"),
            }
            println!();
            continue;
        }

        // 实体线索索引：活跃线程、静默身份、旧 URI 足迹与未解析提及。
        if matches!(input, "实体索引" | "实体线程" | "实体线索" | "实体审计") {
            match chronicle.as_ref() {
                Some(chronicle) => {
                    let registry = oxigraph_manager.entity_registry_snapshot();
                    let digest = EntityThreadDigest::build(chronicle, &registry);
                    println!("{}", digest.render_audit());
                }
                None => println!("编年史不可用（打开失败，见启动警告）。"),
            }
            println!();
            continue;
        }

        if matches!(input, "未解析提及" | "未解析实体" | "实体歧义审计") {
            match chronicle.as_ref() {
                Some(chronicle) => {
                    let registry = oxigraph_manager.entity_registry_snapshot();
                    let digest = EntityThreadDigest::build(chronicle, &registry);
                    println!("{}", digest.render_unresolved());
                }
                None => println!("编年史不可用（打开失败，见启动警告）。"),
            }
            println!();
            continue;
        }

        // 对话时间线：焦点迁移 / 指代补全 / 出处逐轮可见
        if matches!(input, "时间线" | "对话时间线") {
            match chronicle.as_ref() {
                Some(chronicle) => {
                    let turns: Vec<TimelineTurn> = chronicle
                        .entries()
                        .iter()
                        .filter_map(TimelineTurn::from_entry)
                        .collect();
                    println!("对话时间线（编年史投影）：");
                    println!("{}", render_timeline_markdown(&turns));
                    if (1..=12).contains(&turns.len()) {
                        println!();
                        println!("```mermaid");
                        println!("{}", render_timeline_flow(&turns));
                        println!("```");
                    }
                }
                None => println!("编年史不可用（打开失败，见启动警告）。"),
            }
            println!();
            continue;
        }

        // 证明日志：从编年史回放已具体化的证明凭证（稳定 id，可再叙述）
        if matches!(input, "证明日志" | "证明档案" | "证明记录") {
            match chronicle.as_ref() {
                Some(chronicle) => {
                    let projection = JournalProjection::from_chronicle(chronicle);
                    if projection.proofs.is_empty() {
                        println!(
                            "证明日志为空。先用「证明图：<目标>」生成证明，凭证会写入编年史。"
                        );
                    } else {
                        println!(
                            "证明日志：{} 条凭证（按写入顺序；id 只依赖规范逻辑形式）：",
                            projection.proofs.len()
                        );
                        for proof in projection.proofs.iter().rev().take(10).rev() {
                            let confidence = proof
                                .confidence
                                .map(|value| format!("，置信度 {:.2}", value))
                                .unwrap_or_else(|| "，无置信度".to_string());
                            let truncation = if proof.truncated {
                                format!("，已折叠 {} 步", proof.omitted_nodes)
                            } else {
                                String::new()
                            };
                            println!(
                                "  {}  {}（{} 步{}）{}",
                                proof.id, proof.goal, proof.step_count, confidence, truncation
                            );
                        }
                        println!("用法：证明编号：<id> 回放叙述。");
                    }
                }
                None => println!("编年史不可用（打开失败，见启动警告）。"),
            }
            println!();
            continue;
        }

        // 按 id 回放证明叙述（Brief/Normal/Deep 由内容自适应：默认 Deep）
        if let Some(id) = strip_any_prefix(input, &["证明编号：", "证明编号:", "证明编号 "])
        {
            let id = id.trim();
            match chronicle.as_ref() {
                Some(chronicle) if !id.is_empty() => {
                    let projection = JournalProjection::from_chronicle(chronicle);
                    match projection.find_proof(id) {
                        Some(proof) => println!("{}", proof.narrate(ProofDetail::Deep)),
                        None => println!("证明日志中没有 id 为「{}」的凭证。", id),
                    }
                }
                Some(_) => println!("用法：证明编号：<id>，例如：证明编号：proof:0123456789abcdef"),
                None => println!("编年史不可用（打开失败，见启动警告）。"),
            }
            println!();
            continue;
        }

        // 日志推导的一致性：同题异答 / 知识版本变更 / 过期回答 / 未答问题
        if matches!(
            input,
            "一致性" | "回答一致性" | "改口检查" | "改口" | "日志视图"
        ) {
            match chronicle.as_ref() {
                Some(chronicle) => {
                    let projection = JournalProjection::from_chronicle(chronicle);
                    println!(
                        "{}",
                        projection.consistency(chronicle.kb_revision()).render()
                    );
                }
                None => println!("编年史不可用（打开失败，见启动警告）。"),
            }
            println!();
            continue;
        }

        // 对话情景理论：把编年史编译成可查询的有限 Horn 理论（CST）
        if matches!(input, "情景" | "对话情景" | "情景理论") {
            match chronicle.as_ref() {
                Some(chronicle) => {
                    let theory = ConversationTheory::build(chronicle);
                    println!("{}", theory.render_report());
                    if !theory.changes().is_empty() {
                        println!();
                        println!("{}", theory.render_changes());
                    }
                }
                None => println!("编年史不可用（打开失败，见启动警告）。"),
            }
            println!();
            continue;
        }

        if let Some(spec) = strip_any_prefix(input, &["情景：", "情景:"]) {
            match (chronicle.as_ref(), spec.trim().parse::<u64>()) {
                (Some(chronicle), Ok(seq)) => {
                    let theory = ConversationTheory::build(chronicle);
                    println!("{}", theory.render_situation(seq));
                }
                (Some(_), Err(_)) => println!("用法：情景：<seq>，例如：情景：12"),
                (None, _) => println!("编年史不可用（打开失败，见启动警告）。"),
            }
            println!();
            continue;
        }

        if let Some(query) = strip_any_prefix(input, &["情景历史：", "情景历史:", "情景历史 "])
        {
            match chronicle.as_ref() {
                Some(chronicle) => {
                    let theory = ConversationTheory::build(chronicle);
                    if query.trim().is_empty() {
                        println!("{}", theory.render_report());
                    } else {
                        println!("{}", theory.render_history(query.trim()));
                    }
                }
                None => println!("编年史不可用（打开失败，见启动警告）。"),
            }
            println!();
            continue;
        }

        if let Some(query) = strip_any_prefix(input, &["情景归因：", "情景归因:", "情景归因 "])
        {
            match chronicle.as_ref() {
                Some(chronicle) => {
                    let theory = ConversationTheory::build(chronicle);
                    println!("{}", theory.render_causes(query.trim()));
                }
                None => println!("编年史不可用（打开失败，见启动警告）。"),
            }
            println!();
            continue;
        }

        if let Some(spec) = strip_any_prefix(input, &["情景反事实：", "情景反事实:", "情景反事实 "])
        {
            let spec = spec.trim();
            let (seq, query) = match spec.split_once(['：', ':']) {
                Some((left, right)) if left.trim().parse::<u64>().is_ok() => {
                    (left.trim().parse::<u64>().ok(), right.trim())
                }
                _ => (None, spec),
            };
            match chronicle.as_ref() {
                Some(chronicle) => {
                    let theory = ConversationTheory::build(chronicle);
                    match seq {
                        Some(seq) => {
                            let report = theory.counterfactual_undo_seq(seq, query);
                            println!("{}", theory.render_counterfactual_report(&report));
                        }
                        None => println!("{}", theory.render_counterfactual(query)),
                    }
                }
                None => println!("编年史不可用（打开失败，见启动警告）。"),
            }
            println!();
            continue;
        }

        if let Some(pattern) = strip_any_prefix(input, &["情景查询：", "情景查询:", "情景查询 "])
        {
            match chronicle.as_ref() {
                Some(chronicle) => {
                    let theory = ConversationTheory::build(chronicle);
                    println!("{}", theory.render_query(pattern.trim()));
                }
                None => println!("编年史不可用（打开失败，见启动警告）。"),
            }
            println!();
            continue;
        }

        // 历史可信度：沿支持图回答「哪条回答依赖了错值、经哪条边失效」
        if matches!(input, "历史可信度" | "回答可信度" | "信任报告") {
            match chronicle.as_ref() {
                Some(chronicle) => {
                    let assessment = TrustAssessment::from_chronicle(chronicle);
                    println!("{}", assessment.render(None));
                }
                None => println!("编年史不可用（打开失败，见启动警告）。"),
            }
            println!();
            continue;
        }

        if let Some(query) = strip_any_prefix(
            input,
            &[
                "历史可信度：",
                "历史可信度:",
                "历史可信度 ",
                "可信度：",
                "可信度:",
                "信任：",
                "信任:",
            ],
        ) {
            match chronicle.as_ref() {
                Some(chronicle) => {
                    let assessment = TrustAssessment::from_chronicle(chronicle);
                    println!("{}", assessment.render(Some(query.trim())));
                }
                None => println!("编年史不可用（打开失败，见启动警告）。"),
            }
            println!();
            continue;
        }

        // 证明图：把归结证明树渲染成 Mermaid/ASCII（看得见的推理）
        if let Some(target) = strip_any_prefix(
            input,
            &[
                "证明图：",
                "证明图:",
                "证明树：",
                "证明树:",
                "画证明：",
                "画证明:",
            ],
        ) {
            let target = target.trim();
            if target.is_empty() {
                println!("请提供要证明的目标，例如：证明图：张三的师傅的师傅是王五");
            } else {
                refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                match symbolic_reasoner.resolution_proof_view(target) {
                    Some(view) => {
                        println!(
                            "证明图：{}  〔{}；知识库 {} 条事实 / {} 条规则〕",
                            view.phrase, view.goal_display, view.facts, view.clauses
                        );
                        if view.solutions.is_empty() {
                            if view.search_status == SearchStatus::Incomplete {
                                println!(
                                    "未得证：深度上限 {} 内搜索被截断，不能据此断言目标不存在（开世界假设）。",
                                    view.max_depth
                                );
                            } else {
                                println!(
                                    "未得证：搜索已穷尽所有分支，未找到归结反驳（开世界假设：推不出 ≠ 为假）。"
                                );
                            }
                            println!(
                                "搜索状态：{}{}",
                                view.search_status.label(),
                                if view.search_truncated {
                                    "（结果可能不完整）"
                                } else {
                                    ""
                                }
                            );
                            println!(
                                "搜索统计：{} 个目标、{} 次目标展开、{} 次事实检查、{} 次规则尝试",
                                view.stats.goals,
                                view.stats.goal_attempts,
                                view.stats.fact_checks,
                                view.stats.rule_attempts
                            );
                            const SHOWN_GAPS: usize = 5;
                            for entry in view.frontier.iter().take(SHOWN_GAPS) {
                                println!("知识缺口：{}", entry.describe());
                            }
                            let omitted = view.frontier.len().saturating_sub(SHOWN_GAPS)
                                + view.stats.frontier_omitted;
                            if omitted > 0 {
                                println!("……另有 {} 条知识缺口未展开", omitted);
                            }
                        }
                        let certificates =
                            ProofCertificate::from_view(&view, ProofLimits::default());
                        if let Some(chronicle) = chronicle.as_mut() {
                            let _ =
                                know_prolog::proof_journal::record_proofs(chronicle, &certificates);
                        }
                        for (index, solution) in view.solutions.iter().enumerate() {
                            let bindings = if solution.bindings.is_empty() {
                                String::new()
                            } else {
                                format!(
                                    "，绑定 {}",
                                    solution
                                        .bindings
                                        .iter()
                                        .map(|(name, value)| format!("{}={}", name, value))
                                        .collect::<Vec<_>>()
                                        .join("；")
                                )
                            };
                            println!(
                                "解 {}（置信度 {:.2}{}）：",
                                index + 1,
                                solution.confidence,
                                bindings
                            );
                            println!("{}", render_proof_ascii(&solution.proofs));
                            println!("```mermaid");
                            println!("{}", render_proof_mermaid(&solution.proofs));
                            println!("```");
                            if let Some(certificate) = certificates.get(index) {
                                println!(
                                    "证明编号：{}（已写入证明日志；用「证明编号：{}」回放）",
                                    certificate.id, certificate.id
                                );
                            }
                        }
                    }
                    None => println!(
                        "无法把「{}」解析为可归结目标；可用「证明：」查看文本报告。",
                        target
                    ),
                }
            }
            println!();
            continue;
        }

        // 知识子图：焦点实体二跳邻域的确定性分层视图（★ 焦点，==> 本轮出处边）
        if let Some(entity) = strip_any_prefix(
            input,
            &[
                "知识子图：",
                "知识子图:",
                "子图：",
                "子图:",
                "邻域：",
                "邻域:",
            ],
        ) {
            let entity = entity.trim();
            if entity.is_empty() {
                println!("请提供焦点实体，例如：知识子图：姚明");
            } else {
                match load_graph_snapshot(&oxigraph_manager).await {
                    Ok(facts) => {
                        let edges = prolog_fact_edges(&facts);
                        let highlight = latest_used_edges(&dialogue_manager, &edges);
                        let subgraph = KnowledgeSubgraph::bfs(entity, &edges, 2, &highlight);
                        println!(
                            "知识子图「{}」：{} 个节点，{} 条边（★ 焦点，==> 本轮推理用到）：",
                            entity,
                            subgraph.node_count(),
                            subgraph.edge_count()
                        );
                        println!("```mermaid");
                        println!("{}", subgraph.render_mermaid());
                        println!("```");
                        println!("JSON：{}", subgraph.render_json());
                    }
                    Err(e) => println!("加载知识图谱失败: {}", e),
                }
            }
            println!();
            continue;
        }

        // 覆盖报告：悬空引用 / 孤立实体 / 未登记谓词（不依赖 schema）
        if matches!(input, "覆盖报告" | "质量报告" | "覆盖检查") {
            match load_graph_snapshot(&oxigraph_manager).await {
                Ok(facts) => {
                    let mut coverage = CoverageInput::default();
                    for fact in facts.iter() {
                        match fact {
                            PrologFact::Entity {
                                name, properties, ..
                            } => {
                                coverage.entities.push(name.clone());
                                for key in properties.keys() {
                                    coverage.attributes.push((name.clone(), key.clone()));
                                }
                            }
                            PrologFact::Relation { from, rel_type, to } => {
                                coverage
                                    .edges
                                    .push((from.clone(), rel_type.clone(), to.clone()));
                            }
                        }
                    }
                    for profile in &language_config.predicate_profiles {
                        coverage.registered_predicates.push(profile.name.clone());
                        coverage
                            .registered_predicates
                            .extend(profile.aliases.iter().cloned());
                        if let Some(inverse) = &profile.inverse {
                            coverage.registered_predicates.push(inverse.clone());
                        }
                    }
                    println!("{}", render_coverage_report(&coverage));
                }
                Err(e) => println!("加载知识图谱失败: {}", e),
            }
            println!();
            continue;
        }

        // 虚词算符表：功能词唯一正本 + 投影快照（语言地基审计视图）
        if matches!(input, "虚词表" | "功能词表" | "虚词报告" | "算符表") {
            println!("{}", know_prolog::function_words::render_registry_report());
            println!();
            continue;
        }

        // 神经符号对比：对任意陈述做联想通道与符号通道的交叉验证与仲裁
        if let Some(target) = strip_any_prefix(
            input,
            &[
                "对比：",
                "对比:",
                "神经符号对比：",
                "神经符号对比:",
                "双通道对比：",
                "双通道对比:",
                "仲裁：",
                "仲裁:",
            ],
        ) {
            let target = target.trim();
            if target.is_empty() {
                println!("请提供要对比的陈述句，例如：对比：李四的国籍是德国");
            } else {
                refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                let case = contrast_arbiter.contrast_statement(target, &symbolic_reasoner);
                println!("{}", contrast_arbiter.render_case(&case));
            }
            println!();
            continue;
        }

        // 对比校准报告：会话级一致率/冲突率与最近案例
        if matches!(
            input,
            "对比报告"
                | "对比统计"
                | "校准报告"
                | "对比校准"
                | "仲裁报告"
                | "仲裁统计"
                | "对比记忆"
                | "校准记忆"
        ) {
            println!("{}", contrast_arbiter.memory_summary());
            println!("{}", contrast_arbiter.calibration_report());
            println!();
            continue;
        }

        // 待定联想推测清单
        if matches!(input, "待定推测" | "对比推测" | "联想推测") {
            println!("{}", contrast_arbiter.pending_report());
            println!();
            continue;
        }

        // 确认联想推测：采信为会话事实（不修改知识图谱）
        if let Some(target) = strip_any_prefix(
            input,
            &[
                "对比确认",
                "对比确认：",
                "对比确认:",
                "对比采纳",
                "对比采纳：",
                "对比采纳:",
            ],
        ) {
            match parse_hypothesis_index(target) {
                Some(index) => {
                    println!(
                        "{}",
                        contrast_arbiter
                            .confirm_guess(index, &mut symbolic_reasoner)
                            .unwrap_or_else(|| format!(
                                "没有待定的联想推测 #{}；可用「待定推测」查看列表。",
                                index
                            ))
                    );
                }
                None => println!(
                    "请给出推测编号，例如：对比确认 1。当前待定推测：\n{}",
                    contrast_arbiter.pending_report()
                ),
            }
            println!();
            continue;
        }

        // 否认联想推测：加入屏蔽集，同类推测被抑制
        if let Some(target) = strip_any_prefix(
            input,
            &[
                "对比否认",
                "对比否认：",
                "对比否认:",
                "对比否决",
                "对比否决：",
                "对比否决:",
            ],
        ) {
            match parse_hypothesis_index(target) {
                Some(index) => {
                    println!(
                        "{}",
                        contrast_arbiter
                            .refute_guess(index)
                            .unwrap_or_else(|| format!(
                                "没有待定的联想推测 #{}；可用「待定推测」查看列表。",
                                index
                            ))
                    );
                }
                None => println!(
                    "请给出推测编号，例如：对比否认 1。当前待定推测：\n{}",
                    contrast_arbiter.pending_report()
                ),
            }
            println!();
            continue;
        }

        // 对比追问：系统主动就最旧的待定推测提问，把联想直觉逆转为澄清对话
        if matches!(input, "对比追问" | "对比提问" | "对比核实" | "逐条核实") {
            match contrast_arbiter.next_followup() {
                Some(followup) => {
                    println!("{}", followup.question);
                    contrast_followup = Some(followup.guess_id);
                }
                None => println!(
                    "当前没有待定联想推测。可输入「对比：<陈述句>」生成新的双通道对比，或「待定推测」查看清单。"
                ),
            }
            println!();
            continue;
        }

        // 对比追问进行中：把「是 / 不是 / 跳过」直接结算为确认或否认（多轮对话）
        if let Some(guess_id) = contrast_followup {
            let answer = input.trim().trim_end_matches(['。', '！', '!', '~']);
            let affirmative = [
                "是", "是的", "对", "对的", "没错", "属实", "确认", "yes", "y",
            ]
            .contains(&answer);
            let negative = ["不是", "不对", "否", "不属实", "错", "no", "n"].contains(&answer);
            if affirmative || negative {
                let outcome = if affirmative {
                    contrast_arbiter.confirm_guess(guess_id, &mut symbolic_reasoner)
                } else {
                    contrast_arbiter.refute_guess(guess_id)
                };
                println!(
                    "{}",
                    outcome.unwrap_or_else(|| format!(
                        "推测 #{} 已不在待定列表（可能已被处理）。",
                        guess_id
                    ))
                );
                contrast_followup = None;
                println!();
                continue;
            }
            if matches!(answer, "跳过" | "下一个" | "不知道" | "不确定") {
                println!("已跳过推测 #{}；可稍后再输入「对比追问」。", guess_id);
                contrast_followup = None;
                println!();
                continue;
            }
            // 非作答输入：结束本轮追问，按普通输入继续处理
            contrast_followup = None;
        }

        // 语义歧义澄清进行中：序号/前者/后者选定读法；都不是放弃；其它输入放行。
        if handle_formal_clarification(&mut symbolic_reasoner, input) {
            continue;
        }

        // 帧澄清进行中：短答（序号 / 实体名 / 都不是）→ 确定性绑定 → 重放原问句。
        // 绑定有出处（候选 + 轮次），无源可填的洞不猜；非回答输入零状态变化、放行。
}

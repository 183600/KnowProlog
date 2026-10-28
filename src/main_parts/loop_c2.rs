{
        if matches!(input, "槽账本" | "槽位账本" | "改口账本") {
            println!("{}", symbolic_reasoner.slot_ledger_report());
            println!();
            continue;
        }

        // 单个槽的改口史：改口史：姚明 身高 / 改口史：姚明的身高
        if let Some(target) =
            strip_any_prefix(input, &["改口史：", "改口史:", "槽历史：", "槽历史:"])
        {
            let target = target.trim();
            let pair = target
                .split_once('|')
                .or_else(|| target.split_once('的'))
                .map(|(subject, predicate)| (subject.trim(), predicate.trim()))
                .filter(|(subject, predicate)| !subject.is_empty() && !predicate.is_empty());
            match pair {
                Some((subject, predicate)) => {
                    println!(
                        "{}",
                        symbolic_reasoner.slot_history_report(subject, predicate)
                    );
                }
                None => {
                    println!("请给出主语和谓词，例如：改口史：姚明 身高（或 改口史：姚明的身高）")
                }
            }
            println!();
            continue;
        }

        // 修订预演：更正落库前先展示受影响事实、派生结论与阻止级冲突；只计算不写图
        if let Some(target) = strip_any_prefix(
            input,
            &["修订预演：", "修订预演:", "提交预演：", "提交预演:"],
        ) {
            let target = target.trim();
            match parse_correction_spec(target) {
                None => println!(
                    "请给出实体、属性、旧值与新值，例如：修订预演：姚明的身高从226改为229（或 修订预演：姚明|height|226|229）。"
                ),
                Some((entity, predicate, old_value, new_value)) => {
                    refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                    println!(
                        "{}",
                        symbolic_reasoner.belief_revision_preview(
                            &entity,
                            &predicate,
                            &old_value,
                            &new_value
                        )
                    );
                }
            }
            println!();
            continue;
        }

        // 更正知识：删除旧值、写入新值，并把「旧值错了」写成可追踪的事件
        if let Some(target) = strip_any_prefix(
            input,
            &["更正：", "更正:", "纠正：", "纠正:", "修正：", "修正:"],
        ) {
            let target = target.trim();
            match parse_correction_spec(target) {
                None => println!(
                    "请给出实体、属性、旧值与新值，例如：更正：姚明的身高从226改为229（或 更正：姚明|height|226|229）。"
                ),
                Some((entity, predicate, old_value, new_value)) => {
                    match oxigraph_manager
                        .correct_object_value(&entity, &predicate, &old_value, &new_value)
                        .await
                    {
                        Ok((removed, inserted)) if removed > 0 => {
                            let correction = conversation_theory::TheoryCorrection {
                                entity: entity.clone(),
                                predicate: predicate.clone(),
                                from: old_value.clone(),
                                to: new_value.clone(),
                            };
                            let answer = format!(
                                "已更正：{}.{} 由「{}」改为「{}」（移除 {} 条旧三元组，写入 {} 条新值）。",
                                entity,
                                predicate,
                                old_value,
                                new_value,
                                removed,
                                if inserted { 1 } else { 0 }
                            );
                            if let Some(chronicle) = chronicle.as_mut() {
                                let _ = conversation_theory::append_correction(
                                    chronicle,
                                    target,
                                    &[correction],
                                    kb_revision + 1,
                                );
                            }
                            log_chronicle_reply(&mut chronicle, &answer, &[], &[]);
                            // 推理器只增不减：用更正后的图快照重建，使回答基于更新后的图谱
                            rebuild_symbolic_reasoner(
                                &mut symbolic_reasoner,
                                &oxigraph_manager,
                            )
                            .await;
                            reasoner_stale = true;
                            let _ = oxigraph_manager.save_predicate_registry(&registry_path);
                            println!("{}", answer);
                        }
                        Ok(_) => println!(
                            "没有找到「{}」的 {} = 「{}」：更正是「旧值错了」，不是无条件覆盖；未改动知识库。",
                            entity, predicate, old_value
                        ),
                        Err(error) => eprintln!("更正失败：{}", error),
                    }
                }
            }
            println!();
            continue;
        }

        // 记住会话事实：断言为会话级事实（不写知识图谱，可随对话记忆持久化）
        if let Some(target) = strip_any_prefix(
            input,
            &["记住：", "记住:", "记录：", "记录:", "断言：", "断言:"],
        ) {
            let target = target.trim();
            if target.is_empty() {
                println!("请给出要记住的事实，例如：记住：张三是李四的师傅");
            } else {
                refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                let answer = symbolic_reasoner.assert_nl_fact(target);
                println!("{}", answer);
                if let Some(deferred) = symbolic_reasoner.resolve_pending_after_assertion() {
                    println!("{}", deferred);
                    log_chronicle_reply(&mut chronicle, &deferred, &[], &[]);
                }
                let effects = conversation_theory::effects_from_text(
                    target,
                    conversation_theory::EffectOp::Add,
                );
                if let Some(chronicle) = chronicle.as_mut() {
                    let _ = conversation_theory::append_assert(
                        chronicle,
                        target,
                        &effects,
                        kb_revision,
                    );
                }
                log_chronicle_reply(&mut chronicle, &answer, &[], &[]);
            }
            println!();
            continue;
        }

        // 忘记会话事实：从会话事实集撤回（不触碰知识图谱）
        if let Some(target) = strip_any_prefix(
            input,
            &["忘记：", "忘记:", "遗忘：", "遗忘:", "撤回：", "撤回:"],
        ) {
            let target = target.trim();
            if target.is_empty() {
                println!("请给出要忘记的事实，例如：忘记：张三是李四的师傅");
            } else {
                let answer = symbolic_reasoner.retract_nl_fact(target);
                println!("{}", answer);
                let effects = conversation_theory::effects_from_text(
                    target,
                    conversation_theory::EffectOp::Remove,
                );
                if let Some(chronicle) = chronicle.as_mut() {
                    let _ = conversation_theory::append_forget(
                        chronicle,
                        target,
                        &effects,
                        kb_revision,
                    );
                }
                log_chronicle_reply(&mut chronicle, &answer, &[], &[]);
            }
            println!();
            continue;
        }

        // 忘记会话规则：子句库与教学原句一致撤回，并重同步规则图账本
        if let Some(target) = strip_any_prefix(
            input,
            &["忘记规则：", "忘记规则:", "撤回规则：", "撤回规则:"],
        ) {
            let answer = symbolic_reasoner.forget_rule_report(target);
            println!("{}", answer);
            log_chronicle_reply(&mut chronicle, &answer, &[], &[]);
            sync_rule_ledger_report(&symbolic_reasoner, &oxigraph_manager).await;
            println!();
            continue;
        }

        // 会话规则库：列出学过的 Horn 规则与会话事实
        if matches!(input, "会话规则" | "规则库" | "后向链规则" | "规则列表") {
            println!("{}", symbolic_reasoner.session_rules_report());
            println!();
            continue;
        }

        // 谓词画像：注册表报告 + 知识库观测 + 候选发现（功能/对称/互逆）
        if matches!(input, "谓词报告" | "谓词画像" | "谓词注册表" | "谓词列表") {
            if let Err(e) = oxigraph_manager.refresh_predicate_registry().await {
                eprintln!("刷新谓词画像失败: {}", e);
            }
            println!("{}", oxigraph_manager.predicate_registry_report());
            let _ = oxigraph_manager.save_predicate_registry(&registry_path);
            println!();
            continue;
        }

        // 律典：声明律 + 挖掘规则的统一台账，附确定性审计（矛盾/缺口分开）
        if matches!(input, "律典" | "定律台账" | "语义定律" | "law-report") {
            if let Err(e) = oxigraph_manager.refresh_predicate_registry().await {
                eprintln!("刷新谓词画像失败: {}", e);
            }
            sync_predicate_pairs(&mut symbolic_reasoner, &oxigraph_manager);
            if let Some(persisted) =
                LawBook::load_from_path(&format!("{}/law_book.json", store_path))
            {
                symbolic_reasoner.load_persisted_laws(persisted);
            }
            println!("{}", symbolic_reasoner.law_report());
            println!();
            continue;
        }

        // 质量护照：确定性可信向量（低分只用于排序/措辞/追问，不写回、不删事实）
        if matches!(input, "质量护照" | "质量总览" | "护照总览") {
            match quality_report_for(&oxigraph_manager, &language_config, &store_path).await {
                Ok(report) => println!("{}", report.render_text()),
                Err(e) => eprintln!("构建质量护照失败: {}", e),
            }
            println!();
            continue;
        }

        // 形状校验：必备属性 / 基数 / 数据类型 / 定义域值域（只读派生）
        if matches!(
            input,
            "形状报告" | "形状校验" | "SHACL报告" | "shacl报告" | "shapes"
        ) {
            match shape_report_for(&oxigraph_manager, &language_config).await {
                Ok((_, report)) => println!("{}", report.render_text()),
                Err(e) => eprintln!("构建形状报告失败: {}", e),
            }
            println!();
            continue;
        }

        // 单实体质量护照：类型来源 / 四维分数 / 可行动缺口
        if let Some(entity) = strip_any_prefix(input, &["质量护照：", "质量护照:"]) {
            let entity = entity.trim();
            if entity.is_empty() {
                println!("请给出实体名，例如：质量护照：张三");
            } else {
                match quality_report_for(&oxigraph_manager, &language_config, &store_path).await {
                    Ok(report) => match report.passport_for(entity) {
                        Some(passport) => println!("{}", render_entity_passport(passport)),
                        None => println!(
                            "知识库中没有实体「{}」（护照只覆盖图谱中存在的实体）。",
                            entity
                        ),
                    },
                    Err(e) => eprintln!("构建质量护照失败: {}", e),
                }
            }
            println!();
            continue;
        }

        // 类型体系：别名/子类/互斥声明与闭包档状态（config.toml [type_system]）
        if matches!(input, "类型体系" | "类型报告" | "类型格") {
            let report = oxigraph_manager.type_system_report();
            if report.trim().is_empty() {
                println!("未声明类型体系（config.toml 的 [type_system] 段可选）。");
            } else {
                println!("{}", report);
            }
            println!();
            continue;
        }

        // 登记谓词别名：别名 = 规范谓词（查询/生成/导入共用同一词面）
        if let Some(spec) = strip_any_prefix(
            input,
            &[
                "登记别名：",
                "登记别名:",
                "谓词别名：",
                "谓词别名:",
                "同义谓词：",
                "同义谓词:",
            ],
        ) {
            match parse_predicate_spec(spec) {
                Some((alias, canonical)) => {
                    if oxigraph_manager.register_predicate_alias(&alias, &canonical) {
                        sync_predicate_pairs(&mut symbolic_reasoner, &oxigraph_manager);
                        let _ = oxigraph_manager.save_predicate_registry(&registry_path);
                        println!(
                            "已登记谓词别名：{} → {}（问「{}」与「{}」将命中同一事实）",
                            alias, canonical, alias, canonical
                        );
                    } else {
                        println!("未发生变化（别名已存在或与规范名相同）。");
                    }
                }
                None => {
                    println!("请用「登记别名：别名 = 规范谓词」的格式，例如：登记别名：老师 = 师傅")
                }
            }
            println!();
            continue;
        }

        // 登记互逆谓词：左 = 右（p(a,b) 与 q(b,a) 等价，导入自动统一方向）
        if let Some(spec) = strip_any_prefix(
            input,
            &[
                "登记逆谓词：",
                "登记逆谓词:",
                "逆谓词：",
                "逆谓词:",
                "互逆谓词：",
                "互逆谓词:",
            ],
        ) {
            match parse_predicate_spec(spec) {
                Some((left, right)) => {
                    if oxigraph_manager.register_predicate_inverse(&left, &right) {
                        sync_predicate_pairs(&mut symbolic_reasoner, &oxigraph_manager);
                        let _ = oxigraph_manager.save_predicate_registry(&registry_path);
                        println!(
                            "已登记互逆谓词：{} ↔ {}（导入时统一为规范方向，问句自动换算）",
                            left, right
                        );
                    } else {
                        println!("未发生变化（互逆关系已存在或两侧相同）。");
                    }
                }
                None => {
                    println!("请用「登记逆谓词：左 = 右」的格式，例如：登记逆谓词：师傅 = 徒弟")
                }
            }
            println!();
            continue;
        }

        // 标记功能谓词：同主语只允许唯一取值，冲突不覆盖、只标记
        if let Some(predicate) = strip_any_prefix(input, &["功能谓词：", "功能谓词:"]) {
            let predicate = predicate.trim();
            if predicate.is_empty() {
                println!("请给出谓词名，例如：功能谓词：年龄");
            } else if oxigraph_manager.mark_predicate_functional(predicate, true) {
                let _ = oxigraph_manager.save_predicate_registry(&registry_path);
                println!(
                    "已标记功能谓词「{}」：同主语的新值将与原值冲突并等待裁决。",
                    predicate
                );
            } else {
                println!("「{}」已是功能谓词。", predicate);
            }
            println!();
            continue;
        }

        if let Some(predicate) = strip_any_prefix(input, &["取消功能谓词：", "取消功能谓词:"])
        {
            let predicate = predicate.trim();
            if !predicate.is_empty() && oxigraph_manager.mark_predicate_functional(predicate, false)
            {
                let _ = oxigraph_manager.save_predicate_registry(&registry_path);
                println!("已取消「{}」的功能谓词标记。", predicate);
            } else {
                println!("「{}」不是功能谓词。", predicate);
            }
            println!();
            continue;
        }

        // 实体身份：确认/拒绝/记住/忘记/合并/铸造检查（谓词注册表的实体侧对偶）
        if let Some(spec) = strip_any_prefix(input, &["确认身份：", "确认身份:"]) {
            match parse_predicate_spec(spec) {
                Some((surface, canonical)) => {
                    if oxigraph_manager.confirm_entity_alias(&surface, &canonical) {
                        symbolic_reasoner.register_entity_alias(&surface, &canonical);
                        let _ = oxigraph_manager.save_entity_registry(&entity_registry_path);
                        println!(
                            "已确认身份：「{}」≡「{}」（持久别名，重启后仍生效）。",
                            surface, canonical
                        );
                    } else {
                        println!("确认未有改动（词面为空、词面即规范名或已确认）。");
                    }
                }
                None => println!("请给出身份对，例如：确认身份：苹果 = 苹果公司"),
            }
            println!();
            continue;
        }

        if let Some(spec) = strip_any_prefix(input, &["记住身份：", "记住身份:"]) {
            match parse_predicate_spec(spec) {
                Some((surface, canonical)) => {
                    if oxigraph_manager.confirm_entity_alias(&surface, &canonical) {
                        symbolic_reasoner.register_entity_alias(&surface, &canonical);
                        let _ = oxigraph_manager.save_entity_registry(&entity_registry_path);
                        println!(
                            "已记住身份：「{}」≡「{}」（提名即时确认并落盘）。",
                            surface, canonical
                        );
                    } else {
                        println!("记住未有改动（词面为空、词面即规范名或已确认）。");
                    }
                }
                None => println!("请给出身份对，例如：记住身份：苹果 = 苹果公司"),
            }
            println!();
            continue;
        }

        if let Some(spec) = strip_any_prefix(input, &["拒绝身份：", "拒绝身份:"]) {
            match parse_predicate_spec(spec) {
                Some((surface, canonical)) => {
                    if oxigraph_manager.reject_entity_alias(&surface, &canonical) {
                        let _ = oxigraph_manager.save_entity_registry(&entity_registry_path);
                        println!(
                            "已拒绝身份：「{}」↛「{}」（立碑：不予自动提名、不参与解析；重启后生效）。",
                            surface, canonical
                        );
                    } else {
                        println!("拒绝未有改动（未找到该提名，或该词面即规范名）。");
                    }
                }
                None => println!("请给出身份对，例如：拒绝身份：苹果 = 苹果果实"),
            }
            println!();
            continue;
        }

        if let Some(surface) = strip_any_prefix(input, &["忘记别名：", "忘记别名:"]) {
            let surface = surface.trim();
            if surface.is_empty() {
                println!("请给出要忘记的词面，例如：忘记别名：苹果");
            } else {
                let affected = oxigraph_manager.forget_entity_alias(surface);
                if affected.is_empty() {
                    println!("没有找到已确认别名「{}」。", surface);
                } else {
                    let _ = oxigraph_manager.save_entity_registry(&entity_registry_path);
                    println!(
                        "已忘记别名「{}」（影响 {} 个身份；墓碑已立，重启后链接器彻底忘记）。",
                        surface,
                        affected.len()
                    );
                }
            }
            println!();
            continue;
        }

        if let Some(spec) = strip_any_prefix(input, &["合并身份：", "合并身份:"]) {
            match parse_predicate_spec(spec) {
                Some((from, into)) => {
                    match oxigraph_manager.merge_entity_identities(&from, &into) {
                        Ok(record) => {
                            symbolic_reasoner.register_entity_alias(&from, &into);
                            let _ = oxigraph_manager.save_entity_registry(&entity_registry_path);
                            println!(
                            "已合并身份：「{}」→「{}」（迁移 {} 个别名；旧名永久重定向，URI 不回收）。",
                            record.from,
                            record.into,
                            record.moved_aliases
                        );
                        }
                        Err(e) => println!("合并失败：{}", e),
                    }
                }
                None => println!("请给出合并对，例如：合并身份：苹果公司 = 苹果"),
            }
            println!();
            continue;
        }

        if let Some(name) = strip_any_prefix(input, &["铸造检查：", "铸造检查:"]) {
            let name = name.trim();
            if name.is_empty() {
                println!("请给出要检查的名称，例如：铸造检查：苹果公司");
            } else {
                let verdict = oxigraph_manager.entity_mint_check(name);
                println!("{}", verdict.line(name));
            }
            println!();
            continue;
        }

        // 确认/拒绝待定谓词：准入对话的裁决入口
        if let Some(predicate) = strip_any_prefix(input, &["确认谓词：", "确认谓词:"]) {
            let predicate = predicate.trim();
            if !predicate.is_empty() && oxigraph_manager.confirm_predicate(predicate) {
                let _ = oxigraph_manager.save_predicate_registry(&registry_path);
                println!(
                    "已确认谓词「{}」：画像转为已声明，可用于稳定推理。",
                    predicate
                );
            } else {
                println!("「{}」不是待确认谓词（可能已确认或未登记）。", predicate);
            }
            println!();
            continue;
        }

        if let Some(predicate) = strip_any_prefix(input, &["拒绝谓词：", "拒绝谓词:"]) {
            let predicate = predicate.trim();
            if predicate.is_empty() {
                println!("请给出谓词名，例如：拒绝谓词：师从");
            } else {
                match oxigraph_manager.reject_predicate(predicate).await {
                    Ok(removed) if removed > 0 => {
                        reasoner_stale = true;
                        sync_predicate_pairs(&mut symbolic_reasoner, &oxigraph_manager);
                        let _ = oxigraph_manager.save_predicate_registry(&registry_path);
                        println!(
                            "已拒绝谓词「{}」并移除 {} 条相关三元组。",
                            predicate, removed
                        );
                    }
                    Ok(_) => println!(
                        "没有可拒绝的待定谓词「{}」（仅自动准入且未确认的谓词可拒绝）。",
                        predicate
                    ),
                    Err(e) => eprintln!("拒绝谓词失败: {}", e),
                }
            }
            println!();
            continue;
        }

        // 撤销最近一次文本导入：逐条移除本次写入的三元组（重启后仍可撤销）
        if matches!(input, "撤销导入" | "撤销上次导入" | "撤回导入") {
            match import_journal.pop() {
                Some(receipt) => match oxigraph_manager.undo_import(&receipt).await {
                    Ok(removed) => {
                        if let Some(chronicle) = chronicle.as_mut() {
                            let _ = conversation_theory::append_retract(
                                chronicle,
                                &receipt,
                                "撤销导入",
                                kb_revision + 1,
                            );
                        }
                        reasoner_stale = true;
                        let retracted_frames = import_frames.retract_import(&receipt);
                        if retracted_frames > 0 {
                            if let Err(error) = import_frames.save(&import_frames_path) {
                                eprintln!("保存导入帧失败：{}", error);
                            }
                            symbolic_reasoner.attach_import_frames(import_frames.clone());
                        }
                        let _ = oxigraph_manager.save_predicate_registry(&registry_path);
                        let _ = import_journal.save(&journal_path);
                        println!(
                            "已撤销上次导入：移除 {} 条三元组（日志余 {} 笔，导入帧退场 {} 条）。",
                            removed,
                            import_journal.len(),
                            retracted_frames
                        );
                    }
                    Err(e) => {
                        import_journal.push(receipt);
                        eprintln!("撤销导入失败: {}", e);
                    }
                },
                None => println!("没有可撤销的导入记录。"),
            }
            println!();
            continue;
        }

        // 上次导入回执：新增/合并/重复/冲突/待确认谓词
        if matches!(input, "导入回执" | "上次导入" | "导入记录") {
            match import_journal.last() {
                Some(receipt) => println!("{}", receipt.summary()),
                None => println!("还没有可查询的文本导入记录。"),
            }
            println!();
            continue;
        }

        // 导入历史：跨会话撤销栈（新 → 旧）
        if matches!(input, "导入历史" | "导入日志" | "撤销历史") {
            println!("{}", import_journal.summary());
            println!();
            continue;
        }

        // 导入帧：从导入陈述句学到的措辞、问法与原句见证
        if matches!(input, "导入帧" | "语言帧" | "帧台账") {
            println!("{}", import_frames.report());
            println!();
            continue;
        }

        // 外链名册：外部知识的出处、取回时刻与时限（跨轮、跨重启）
        if matches!(input, "外链台账" | "外链名册" | "外链报告" | "外链统计") {
            println!("{}", external_knowledge_service.ledger_report());
            println!();
            continue;
        }

        if let Some(entity) = strip_any_prefix(
            input,
            &["外链查看：", "外链查看:", "外链详情：", "外链详情:"],
        ) {
            println!(
                "{}",
                external_knowledge_service.ledger_entry_report(entity.trim())
            );
            println!();
            continue;
        }

        // 显式刷新：删除缓存页，下次提问重新取回（外链数据不在本地写权限内，
        // 用"逻辑时刻 + 显式刷新"代替时间戳轮询）
        if let Some(entity) = strip_any_prefix(
            input,
            &["外链刷新：", "外链刷新:", "外链重取：", "外链重取:"],
        ) {
            let entity = entity.trim();
            if entity.is_empty() {
                println!("请给出实体名，例如：外链刷新：法国");
            } else if external_knowledge_service.refresh_entity(entity) {
                let _ = external_knowledge_service.save_ledger(&external_ledger_path);
                if let Some(chronicle) = chronicle.as_mut() {
                    let _ = conversation_theory::append_external(
                        chronicle,
                        entity,
                        "标记刷新",
                        "外链名册",
                        kb_revision,
                    );
                }
                println!("已标记「{}」待刷新：下次提问将重新取回外部知识。", entity);
            } else {
                println!("外链名册中没有「{}」的条目；下次提问将直接取回。", entity);
            }
            println!();
            continue;
        }

        if let Some(entity) = strip_any_prefix(
            input,
            &["外链删除：", "外链删除:", "外链遗忘：", "外链遗忘:"],
        ) {
            let entity = entity.trim();
            if external_knowledge_service.forget_entity(entity) {
                let _ = external_knowledge_service.save_ledger(&external_ledger_path);
                if let Some(chronicle) = chronicle.as_mut() {
                    let _ = conversation_theory::append_external(
                        chronicle,
                        entity,
                        "删除缓存",
                        "外链名册",
                        kb_revision,
                    );
                }
                println!("已从外链名册删除「{}」。", entity);
            } else {
                println!("外链名册中没有「{}」的条目。", entity);
            }
            println!();
            continue;
        }

        // 确认同义链接：把推测记为别名知识（纯符号的 sameAs 学习，可显式撤销）
        if let Some(body) = strip_any_prefix(
            input,
            &["同义链接：", "同义链接:", "外链别名：", "外链别名:"],
        ) {
            match parse_alias_link(body) {
                Some((alias, target)) => {
                    if external_knowledge_service.confirm_alias(alias, target) {
                        let _ = external_knowledge_service.save_ledger(&external_ledger_path);
                        if let Some(chronicle) = chronicle.as_mut() {
                            let _ = conversation_theory::append_external(
                                chronicle,
                                alias,
                                &format!("同义链接 = {}", target),
                                "外链名册",
                                kb_revision,
                            );
                        }
                        println!("已确认同义链接：「{}」= 「{}」。", alias, target);
                    } else {
                        println!("同义链接不成立：别名与规范实体不能相同或为空。");
                    }
                }
                None => println!(
                    "请使用「同义链接：别名 = 规范实体」格式，例如：同义链接：姚明 = ownthink:姚明"
                ),
            }
            println!();
            continue;
        }

        // 性能报告：存储/推理共享台账（缓存命中、图扫描、版本推进等）
        if matches!(input, "性能报告" | "性能台账" | "运行统计") {
            println!(
                "{}",
                perf_ledger.report_with_title("性能台账（本会话累计）")
            );
            println!();
            continue;
        }

        // 派生台账：知识版本、重建/依赖打戳与推理器内部计数
        if matches!(input, "派生台账" | "版本台账" | "增量报告") {
            println!("{}", symbolic_reasoner.derivation_report());
            println!();
            continue;
        }

        // 推理发现：正向饱和推出多跳新事实（含前提与规则来源）
        if input == "推理发现" || input == "推导发现" {
            refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
            println!("{}", symbolic_reasoner.derived_facts_report(10));
            println!();
            continue;
        }

        // 物化推理：前向饱和的派生事实写入命名图（含来源/前提/物化戳，可审计）
        if matches!(input, "物化推理" | "推理物化" | "物化派生") {
            refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
            let entries = symbolic_reasoner.derived_ledger_entries(100);
            let revision = oxigraph_manager.revision();
            let fingerprint = symbolic_reasoner.rule_fingerprint();
            match oxigraph_manager
                .materialize_derived_ledger(&entries, revision, &fingerprint)
                .await
            {
                Ok(0) => println!(
                    "没有可物化的派生事实（先教学规则，或启用 enable_resolution）。命名图 kg:graphs/derived 已置空。"
                ),
                Ok(count) => {
                    println!(
                        "已物化 {} 条派生事实 → 命名图 kg:graphs/derived（知识版本 {}，规则指纹 {}）。",
                        count, revision, fingerprint
                    );
                    for (index, entry) in entries.iter().take(5).enumerate() {
                        println!(
                            "  {}. {} ← {}（轮次 {}，置信度 {:.2}，前提 {} 条）",
                            index + 1,
                            entry.atom.display(),
                            entry.rule_source,
                            entry.depth,
                            entry.confidence,
                            entry.premises.len()
                        );
                    }
                    if entries.len() > 5 {
                        println!("  …共 {} 条，用「图账本」查看完整出处。", entries.len());
                    }
                }
                Err(error) => eprintln!("派生账本物化失败：{}", error),
            }
            println!();
            continue;
        }

        // 图账本：命名图物化视图审计（规则账本 + 派生账本 + 过期判定）
        if matches!(input, "图账本" | "规则账本" | "规则图账本" | "派生图账本") {
            println!(
                "{}",
                graph_ledger_report(&symbolic_reasoner, &oxigraph_manager).await
            );
            println!();
            continue;
        }

        // 知识融合：来源命名空间视图（只读，不写默认图）
        if matches!(input, "融合" | "融合视图" | "知识融合" | "来源台账") {
            match oxigraph_manager.load_fusion_ledger().await {
                Ok(ledger) => println!("{}", ledger.render_zh()),
                Err(e) => println!("读取融合账本失败：{}", e),
            }
            match oxigraph_manager
                .fusion_plan(
                    &FusionPolicy::new().with_functional(functional_predicates(&language_config)),
                )
                .await
            {
                Ok(plan) => println!("{}", plan.render_zh()),
                Err(e) => println!("计算融合视图失败：{}", e),
            }
            println!();
            continue;
        }

        // 知识融合物化：把当前视图的赢家写入默认图（可重复执行，幂等）
        if matches!(input, "融合应用" | "物化融合" | "应用融合") {
            match oxigraph_manager
                .apply_fusion(
                    &FusionPolicy::new().with_functional(functional_predicates(&language_config)),
                )
                .await
            {
                Ok(outcome) => {
                    println!("{}", outcome.plan.render_zh());
                    println!(
                        "已物化：写入 {} 条，回收 {} 条",
                        outcome.inserted, outcome.removed
                    );
                    refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                }
                Err(e) => println!("融合物化失败：{}", e),
            }
            println!();
            continue;
        }

        // 显式裁决：裁决：谓词=来源（只影响声明的槽位，可审计）
        if let Some(body) = strip_any_prefix(input, &["裁决：", "裁决:"]) {
            match parse_alias_link(body) {
                Some((predicate, namespace)) => {
                    let namespace_id =
                        know_prolog::knowledge_fusion::sanitize_namespace_id(namespace);
                    match oxigraph_manager.load_fusion_ledger().await {
                        Ok(mut ledger) if ledger.namespace(&namespace_id).is_some() => {
                            ledger.record_adjudication(Adjudication::new(
                                predicate,
                                &namespace_id,
                                "交互裁决",
                                "user",
                            ));
                            ledger.sort_and_dedup();
                            match oxigraph_manager.sync_fusion_ledger(&ledger).await {
                                Ok(_) => println!(
                                    "已记录裁决：{} 优先来源 {}（「融合应用」后生效）",
                                    predicate, namespace_id
                                ),
                                Err(e) => println!("裁决保存失败：{}", e),
                            }
                        }
                        Ok(_) => println!(
                            "命名空间「{}」不在账本中：先用 `namespace` 子命令导入来源",
                            namespace
                        ),
                        Err(e) => println!("读取融合账本失败：{}", e),
                    }
                }
                None => println!("裁决格式：裁决：谓词=来源，例如「裁决：年龄=friend」"),
            }
            println!();
            continue;
        }

        // 溯因·假设生成：后向链失败前沿 → 候选假设（可证明则直接给证明）
        if let Some(target) = strip_any_prefix(
            input,
            &[
                "溯因：",
                "溯因:",
                "假设生成：",
                "假设生成:",
                "为什么：",
                "为什么:",
            ],
        ) {
            let target = target.trim().trim_end_matches(['？', '?', '。', ' ']);
            if target.is_empty() {
                println!("请提供要溯因的陈述，例如：溯因：张三是赵六的师祖");
            } else {
                refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                match symbolic_reasoner.abduce_statement(target) {
                    Some(text) => println!("{}", text),
                    None => println!(
                        "无法把「{}」形式化为可溯因的目标。支持句式：S的P是O / O是S的P / S是类型。",
                        target
                    ),
                }
            }
            println!();
            continue;
        }

        // 引擎层故障同步：Prolog / 事实加载失败注入推理器，
        // 恢复模式开启时对话管理器会显式说「我出错了」，绝不并入「没有记录」。
        if dialogue_enabled {
            if let Some(fault) = prolog_engine.take_fault() {
                symbolic_reasoner.note_engine_fault(fault);
            }
        }

        // 假设世界命令先于挂账结算：避免「退出假设」「记住这个假设」被
        // 当成澄清答案吞掉。带「那么/则」的一次性反事实句仍走下方通道。
        if let Some(text) = symbolic_reasoner.handle_assumption_command(input) {
            println!("{}", text);
            println!();
            continue;
        }

        // 本体化对话管理器（纯对话行为）：回顾、解释依据、续聊、展开、寒暄。
        // 用对话本体 + 逻辑程序策略推导话语行为，再由模板组合生成回答。
        if dialogue_enabled {
            refresh_dialogue_quality(
                &mut dialogue_manager,
                &oxigraph_manager,
                &language_config,
                symbolic_reasoner.revision(),
                &store_path,
            )
            .await;
            if let Some(reply) = dialogue_manager.plan_pure_dialogue(input, &symbolic_reasoner) {
                let text = plugin_runtime.style_wrap(&reply.text);
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

        // "为什么……？"：先尝试后向链证明，未得证则生成假设解释
        if let Some(target) = input.strip_prefix("为什么") {
            let target = target.trim().trim_end_matches(['？', '?', '。', ' ']);
            if target.is_empty() {
                println!("请提供要解释的结论，例如：为什么张三是王五的师祖？");
            } else {
                refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                match symbolic_reasoner.abduce_statement(target) {
                    Some(text) => println!("{}", text),
                    None => println!(
                        "无法把「{}」形式化为可解释的目标。支持句式：S的P是O / O是S的P / S是类型。",
                        target
                    ),
                }
            }
            println!();
            continue;
        }

        // 反事实模拟："如果……那么……" / "假设：……"，正向链推演后自动回滚
        if input.starts_with("如果")
            || input.starts_with("假如")
            || input.starts_with("假设")
            || strip_any_prefix(
                input,
                &["反事实：", "反事实:", "假设：", "假设:", "假如：", "假如:"],
            )
            .is_some()
        {
            refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
            match symbolic_reasoner.what_if_report(input) {
                Some(text) => println!("{}", text),
                None => println!(
                    "无法把条件形式化为可模拟的假设。支持句式：如果S的P是O那么… / 假设：S是类型。"
                ),
            }
            println!();
            continue;
        }

        // 采纳待定假设：加入本轮会话事实集（不写知识图谱）
        if let Some(target) = strip_any_prefix(
            input,
            &[
                "采纳假设",
                "采纳假设：",
                "采纳假设:",
                "确认假设",
                "确认假设：",
                "确认假设:",
            ],
        ) {
            match parse_hypothesis_index(target) {
                Some(index) => {
                    println!(
                        "{}",
                        symbolic_reasoner
                            .adopt_hypothesis(index)
                            .unwrap_or_else(|| format!(
                                "没有第 {} 条待定假设；可用「待定假设」查看列表。",
                                index
                            ))
                    );
                    if let Some(deferred) = symbolic_reasoner.resolve_pending_after_assertion() {
                        println!("{}", deferred);
                    }
                }
                None => println!(
                    "请给出假设编号，例如：采纳假设 1。当前待定假设：\n{}",
                    symbolic_reasoner.pending_hypotheses_report()
                ),
            }
            println!();
            continue;
        }

        // 待定假设列表
        if matches!(input, "待定假设" | "假设列表" | "候选假设" | "溯因假设") {
            println!("{}", symbolic_reasoner.pending_hypotheses_report());
            println!();
            continue;
        }

        // 导入指令先于输入模式判定解析：简写（如「导入文件：」）不在 import_keywords
        // 中也可用；参数正文只做前缀剥离，路径/JSON 中的关键词不受影响。
        if let Some(directive) = parse_import_directive(input) {
            match directive {
                ImportDirective::JsonldFile(file_path) => {
                    if file_path.is_empty() {
                        println!("请提供 JSON-LD 文件路径，例如：导入jsonld文件：test_data.jsonld");
                        println!();
                        continue;
                    }
                    println!("正在导入 JSON-LD 文件: {}", file_path);
                    match oxigraph_manager.import_jsonld_file_batch(&file_path).await {
                        Ok(receipt) => {
                            println!("JSON-LD 文件导入成功：{}", receipt.summary());
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
                            if let Some(chronicle) = chronicle.as_mut() {
                                let _ = conversation_theory::append_bulk_import(
                                    chronicle,
                                    "JSON-LD",
                                    &file_path,
                                    receipt.stats.triples,
                                    kb_revision + 1,
                                );
                            }
                            reasoner_stale = true;
                        }
                        Err(e) => {
                            eprintln!("JSON-LD 文件导入失败: {}", e);
                        }
                    }
                }
                ImportDirective::JsonldContent(jsonld_content) => {
                    if jsonld_content.is_empty() {
                        println!(
                            "请提供 JSON-LD 内容，例如：导入内容：{{\"@id\": \"…\", \"@type\": \"Person\"}}"
                        );
                        println!();
                        continue;
                    }
                    println!("正在导入 JSON-LD 内容...");
                    match oxigraph_manager
                        .import_jsonld_batch(&jsonld_content, "内容")
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
                            if let Some(chronicle) = chronicle.as_mut() {
                                let _ = conversation_theory::append_bulk_import(
                                    chronicle,
                                    "JSON-LD",
                                    "内容",
                                    receipt.stats.triples,
                                    kb_revision + 1,
                                );
                            }
                            reasoner_stale = true;
                        }
                        Err(e) => {
                            eprintln!("JSON-LD 内容导入失败: {}", e);
                        }
                    }
                }
                ImportDirective::RdfFile(file_path) => {
                    if file_path.is_empty() {
                        println!(
                            "请提供 RDF 文件路径（支持 .ttl / .nt / .nq / .trig / .rdf / .xml）"
                        );
                        println!();
                        continue;
                    }
                    println!("正在导入 RDF 文件: {}", file_path);
                    match oxigraph_manager.import_rdf_file_batch(&file_path).await {
                        Ok(receipt) => {
                            println!("RDF 文件导入成功：新增 {} 条三元组！", receipt.added);
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
                            if let Some(chronicle) = chronicle.as_mut() {
                                let _ = conversation_theory::append_bulk_import(
                                    chronicle,
                                    "RDF",
                                    &file_path,
                                    receipt.added,
                                    kb_revision + 1,
                                );
                            }
                            reasoner_stale = true;
                        }
                        Err(e) => {
                            eprintln!("RDF 文件导入失败: {}", e);
                        }
                    }
                }
            }
            println!();
            continue;
        }

        // RDF 导入情景：批次台账与按批次线索回指（这批 / 上一批 / 第 N 批 / 来源文件）。
        if matches!(input, "批次台账" | "导入批次" | "批次列表" | "批次报告") {
            match oxigraph_manager.load_import_episodes().await {
                Ok(episodes) => println!(
                    "{}",
                    know_prolog::import_episode::render_episode_ledger(&episodes)
                ),
                Err(error) => eprintln!("读取导入批次失败：{}", error),
            }
            println!();
            continue;
        }
        if know_prolog::import_episode::looks_like_episode_reference(input) {
            match oxigraph_manager.load_import_episodes().await {
                Ok(episodes) => {
                    match oxigraph_manager
                        .answer_import_episode_question(input, &episodes)
                        .await
                    {
                        Ok(Some(answer)) => println!("{}", answer),
                        Ok(None) if episodes.is_empty() => {
                            println!("还没有 RDF 导入批次；没有可用于「这批」的阅读记录。")
                        }
                        Ok(None) => println!(
                            "没有匹配的批次「{}」。可用「批次台账」查看现有 {} 批。",
                            input,
                            episodes.len()
                        ),
                        Err(error) => eprintln!("读取导入批次失败：{}", error),
                    }
                }
                Err(error) => eprintln!("读取导入批次失败：{}", error),
            }
            println!();
            continue;
        }

        // 导出默认图为 RDF 文件（格式按扩展名，只读）
        if let Some(directive) = parse_export_directive(input) {
            match directive {
                ExportDirective::RdfFile(file_path) => {
                    if file_path.is_empty() {
                        println!(
                            "请提供导出路径（.ttl / .nt / .nq / .trig / .rdf / .xml / .jsonld）"
                        );
                        println!();
                        continue;
                    }
                    match oxigraph_manager.export_rdf_file(&file_path).await {
                        Ok(count) => println!("导出成功：{} 条三元组 → {}", count, file_path),
                        Err(e) => println!("导出失败：{}", e),
                    }
                }
            }
            println!();
            continue;
        }

        // 任意 SPARQL 1.1 查询（只读，不改变知识库与推理器状态）
        if let Some(sparql_query) = parse_sparql_command(input) {
            if sparql_query.is_empty() {
                println!(
                    "请提供 SPARQL 查询，例如：SPARQL：SELECT ?s ?p ?o WHERE {{ ?s ?p ?o }} LIMIT 10"
                );
                println!();
                continue;
            }
            match oxigraph_manager.query_sparql(sparql_query).await {
                Ok(result) => println!("{}", result.render_text()),
                Err(e) => println!("SPARQL 查询失败：{}", e),
            }
            println!();
            continue;
        }

        // 悬问回填：上一轮追问过「缺哪条前提」，本轮短答直接绑定并延期回答。
        // 命令、新问句与导入行一律放行，避免劫持正常对话。
        if symbolic_reasoner.has_pending_question() && !is_command_like(input) {
            if let Some(deferred) = symbolic_reasoner.try_answer_pending_question(input) {
                println!("{}", deferred);
                log_chronicle_reply(&mut chronicle, &deferred, &[], &[]);
                println!();
                continue;
            }
        }
        // 裸回答窗口只开一轮：本轮没有绑定成功，后续输入不得再被旧追问劫持。
        symbolic_reasoner.clear_pending_awaiting();

        // 图谱刚吸收新知识：重放悬问，能证就立即给出延期回答（仍不猜）。
        if graph_refreshed && symbolic_reasoner.has_pending_question() {
            if let Some(deferred) = symbolic_reasoner.resolve_pending_after_assertion() {
                println!("{}", deferred);
                log_chronicle_reply(&mut chronicle, &deferred, &[], &[]);
                println!();
            }
            graph_refreshed = false;
        }

        // 悬问台账：欠用户答案的问题与缺失前提；可显式放弃（不设 TTL）。
        if matches!(input, "悬问" | "未答之问" | "悬问报告" | "悬问台账") {
            println!("{}", symbolic_reasoner.pending_question_report());
            println!();
            continue;
        }
        if matches!(input, "放弃悬问" | "清空悬问" | "放弃未答之问") {
            println!("{}", symbolic_reasoner.abandon_pending_questions());
            println!();
            continue;
        }
        if matches!(input, "重开悬问" | "恢复悬问" | "重开未答之问") {
            println!("{}", symbolic_reasoner.reopen_pending_questions());
            println!();
            continue;
        }
}

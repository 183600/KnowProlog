{
        // 对话式澄清进行中：优先把输入当作对当前提问的回答（是 / 不是 / 不知道）
        if symbolic_reasoner.clarification_active()
            && !matches!(input, "结束澄清" | "停止澄清" | "退出澄清" | "quit")
            && !is_command_like(input)
        {
            match symbolic_reasoner.answer_clarification(input) {
                Some(text) => println!("{}", text),
                None => println!("（澄清会话已结束）"),
            }
            if let Some(deferred) = symbolic_reasoner.resolve_pending_after_assertion() {
                println!("{}", deferred);
            }
            println!();
            continue;
        }

        // 开始对话式澄清：把待定假设逆转为系统主动提问
        if matches!(input, "澄清" | "开始澄清" | "澄清提问" | "提问确认") {
            match symbolic_reasoner.start_clarification() {
                Some(text) => println!("{}", text),
                None => println!(
                    "当前没有可澄清的待定假设。可先输入「溯因：<陈述>」或「场景：<问句>」生成候选假设。"
                ),
            }
            println!();
            continue;
        }

        // 结束澄清会话：输出已确认/否定/不确定清单
        if matches!(input, "结束澄清" | "停止澄清" | "退出澄清") {
            match symbolic_reasoner.stop_clarification() {
                Some(text) => println!("{}", text),
                None => println!("当前没有进行中的澄清会话。"),
            }
            println!();
            continue;
        }

        // 生成式联合解释：枚举假设组合并注入验证，输出"哪些假设一起才能解释目标"
        if let Some(target) = strip_any_prefix(
            input,
            &[
                "场景：",
                "场景:",
                "联合解释：",
                "联合解释:",
                "解释场景：",
                "解释场景:",
            ],
        ) {
            let target = target.trim();
            if target.is_empty() {
                println!("请提供要联合解释的目标，例如：场景：张三的师祖是赵六");
            } else {
                refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                println!("{}", symbolic_reasoner.scenario_report(target));
            }
            println!();
            continue;
        }

        // 语义森林：同一句子的全部候选读法 + 图谱接地状态（歧义不静默压平）
        if let Some(target) = strip_any_prefix(
            input,
            &["语义分析：", "语义分析:", "语义森林：", "语义森林:"],
        ) {
            let target = target.trim();
            if target.is_empty() {
                println!("请提供要分析的句子，例如：语义分析：张三的年龄是多少");
            } else {
                match load_graph_snapshot(&oxigraph_manager).await {
                    Ok(facts) => symbolic_reasoner.observe_facts(&facts),
                    Err(e) => println!("加载知识图谱失败: {}", e),
                }
                println!("{}", symbolic_reasoner.formal_semantics_report(target));
            }
            println!();
            continue;
        }

        // 形式化语义与语义蕴含：把陈述句解析为逻辑形式并判定蕴含/矛盾/未知
        if let Some(target) = strip_any_prefix(
            input,
            &[
                "判断：",
                "判断:",
                "蕴含：",
                "蕴含:",
                "校验：",
                "校验:",
                "形式化：",
                "形式化:",
            ],
        ) {
            let target = target.trim();
            if target.is_empty() {
                println!("请提供要判定的陈述句，例如：判断：张三是李四的师傅");
            } else {
                match load_graph_snapshot(&oxigraph_manager).await {
                    Ok(facts) => symbolic_reasoner.observe_facts(&facts),
                    Err(e) => println!("加载知识图谱失败: {}", e),
                }
                println!("{}", symbolic_reasoner.entailment_report(target));
            }
            println!();
            continue;
        }

        // 回答计划审计：查看 id、知识版本、逐点命题与来源。
        if matches!(input, "回答计划" | "计划台账" | "生成计划" | "回答方案") {
            println!("{}", symbolic_reasoner.answer_plan_report());
            println!();
            continue;
        }

        // 回答计划复核：用当前知识重建同一实体计划并逐点对照，不改写旧回答。
        if matches!(input, "复核计划" | "计划复核" | "核对计划" | "检查计划") {
            println!("{}", symbolic_reasoner.review_generation_plan());
            println!();
            continue;
        }

        // 多轮生成：按同一份回答计划换风格重述（不重新检索、不新增事实）
        if let Some(directive) = StyleDirective::parse(input) {
            match symbolic_reasoner.restyle_generation(directive) {
                Some(text) => println!("{}", text),
                None => println!("生成功能未启用，无法按回答计划重述。"),
            }
            println!();
            continue;
        }

        // 多轮生成：接着上一段介绍继续讲
        if matches!(
            input,
            "接着说" | "继续说" | "继续介绍" | "继续讲" | "接着讲"
        ) {
            match symbolic_reasoner.continue_generation() {
                Some(text) => println!("{}", text),
                None => println!("还没有可续说的生成内容，请先输入「介绍：实体名」。"),
            }
            println!();
            continue;
        }

        // 多轮生成：换个角度重述
        if matches!(
            input,
            "换个角度说" | "换个说法" | "换个角度" | "换一个角度" | "重新介绍"
        ) {
            match symbolic_reasoner.switch_generation_angle() {
                Some(text) => println!("{}", text),
                None => println!("还没有可重述的生成内容，请先输入「介绍：实体名」。"),
            }
            println!();
            continue;
        }

        // 读法追踪：token/词性来源/字符边界/计划/失败分类（只读，不改回答）
        if let Some(text) = strip_any_prefix(input, &["读法追踪：", "读法追踪:", "读法：", "读法:"])
        {
            refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
            let text = text.trim();
            if text.is_empty() {
                println!("请提供要追踪的问句，例如：读法追踪：张三的所有权归谁");
            } else {
                println!("{}", symbolic_reasoner.lexical_trace_report(text));
            }
            println!();
            continue;
        }

        // 读法台账：失败轮的有限枚举计数与词性来源分布（不保存原文）
        if matches!(input, "读法台账" | "读法报告" | "失败分类台账" | "失败分类")
        {
            println!("{}", symbolic_reasoner.read_trace_ledger_report());
            println!();
            continue;
        }

        // 提及账本：每个指称的证据收据（边界 + 类型证据 × 疑问词期待 + 焦点先验）
        if let Some(text) = strip_any_prefix(input, &["提及账本：", "提及账本:", "账本：", "账本:"])
        {
            refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
            let text = text.trim();
            if text.is_empty() {
                println!("请提供要审计的文本，例如：提及账本：谁写过《背影》");
            } else {
                println!("{}", symbolic_reasoner.mention_ledger_report(text));
            }
            println!();
            continue;
        }

        // 身份台账：规范身份 ↔ 别名 ↔ 歧义（实体 URI 的只读审计视图）
        if matches!(input, "身份台账" | "身份注册表" | "身份报告" | "实体身份") {
            refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
            println!("{}", oxigraph_manager.entity_registry_report());
            println!("{}", symbolic_reasoner.identity_report());
            println!();
            continue;
        }

        // 指称绑定账本：跨轮「说法 ↔ 实体」绑定与澄清裁决记录（CRD）
        if matches!(input, "绑定账本" | "指称账本" | "语境账本") {
            println!("{}", crd_ledger.report());
            println!();
            continue;
        }

        // 指称接地证书：答案探测式消歧留下的可核对记录
        if matches!(input, "接地证书" | "指称证书" | "接地台账" | "指称接地") {
            println!("{}", grounding_ledger.report());
            println!();
            continue;
        }

        // 接地证书重放：用当前知识版本复算最近一张证书的候选答案指纹与动作
        if matches!(input, "接地重放" | "指称重放" | "重放证书" | "证书重放") {
            refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
            match grounding_ledger.latest() {
                Some(certificate) => {
                    if !certificate.verify() {
                        println!("（警告：证书摘要校验失败，台账可能被外部修改。）");
                    }
                    let report =
                        certificate.replay(symbolic_reasoner.revision(), |entity, rewritten| {
                            probe_reference_answer(&symbolic_reasoner, entity, rewritten)
                        });
                    println!("{}", report.render_text());
                }
                None => {
                    println!("（空：暂无可重放的接地证书；先发生一次答案探测式消歧。）");
                }
            }
            println!();
            continue;
        }

        // 提及格全局解码：不重叠最大证据路径（精确/别名/模糊/嵌套统一选路）
        if let Some(text) = strip_any_prefix(input, &["提及格：", "提及格:", "格解码：", "格解码:"])
        {
            refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
            let text = text.trim();
            if text.is_empty() {
                println!("请提供要解码的文本，例如：提及格：北京大学的校长是谁");
            } else {
                println!("{}", symbolic_reasoner.mention_lattice_report(text));
            }
            println!();
            continue;
        }

        // 提及词表覆盖层：已确认词面回灌切分，展示旧/新边界与差异
        if let Some(text) = strip_any_prefix(
            input,
            &[
                "提及词表：",
                "提及词表:",
                "词表感知：",
                "词表感知:",
                "词表切分：",
                "词表切分:",
            ],
        ) {
            refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
            let text = text.trim();
            if text.is_empty() {
                println!("请提供要切分的文本，例如：提及词表：未名科技公司发布了新芯片");
            } else {
                let legacy = preprocess_annotated(text, &plugin_nlp);
                let overlay = symbolic_reasoner.segment_annotated_with_lexicon(text, &plugin_nlp);
                println!("{}", symbolic_reasoner.mention_lexicon_match_report(text));
                let render = |words: &[know_prolog::nlp::AnnotatedWord]| {
                    words
                        .iter()
                        .map(|word| format!("{}[{}]", word.text, word.source.code()))
                        .collect::<Vec<_>>()
                        .join(" | ")
                };
                println!("旧切分：{}", render(&legacy));
                println!("词表切分：{}", render(&overlay));
            }
            println!();
            continue;
        }

        if matches!(input, "词表" | "提及词表" | "词表报告" | "词表台账") {
            println!("{}", symbolic_reasoner.mention_lexicon_report());
            println!();
            continue;
        }

        if let Some(surface) = strip_any_prefix(
            input,
            &["词表提名：", "词表提名:", "提名词表：", "提名词表:"],
        ) {
            let surface = surface.trim();
            if surface.is_empty() {
                println!("请提供要提名的词面，例如：词表提名：未名科技公司");
            } else if symbolic_reasoner.nominate_mention(surface, None) {
                println!(
                    "已提名「{}」（待确认，不参与切分、不写图谱；准入：{} 可确认）。",
                    surface, surface
                );
            } else {
                println!("「{}」未提名：它已确认或已被拒绝。", surface);
            }
            println!();
            continue;
        }

        if let Some(surface) = strip_any_prefix(
            input,
            &["词表拒绝：", "词表拒绝:", "拒绝词表：", "拒绝词表:"],
        ) {
            let surface = surface.trim();
            if surface.is_empty() {
                println!("请提供要拒绝的词面，例如：词表拒绝：未名科技公司");
            } else if symbolic_reasoner.reject_mention(surface) {
                println!(
                    "已拒绝「{}」：墓碑挡板已生效，后续观察不会再确认它。",
                    surface
                );
            } else {
                println!("「{}」已是墓碑，无需重复拒绝。", surface);
            }
            println!();
            continue;
        }

        // OOV 准入：提名 → 用户确认 → 写入知识库（显式词汇增长，可回溯）
        if let Some(target) = strip_any_prefix(
            input,
            &[
                "准入：",
                "准入:",
                "收录：",
                "收录:",
                "确认录入：",
                "确认录入:",
            ],
        ) {
            let target = target.trim();
            if target.is_empty() {
                println!("请提供要准入的词面，例如：准入：未名科技公司");
            } else {
                // 铸 URI 前的宪法检查：已存在/重定向/折叠同形/歧义时只引导，不新铸
                match oxigraph_manager.entity_mint_check(target) {
                    MintVerdict::Fresh => {}
                    verdict => {
                        println!("{}", verdict.line(target));
                        if let MintVerdict::Variant { canonical, .. } = verdict {
                            println!(
                                "若「{}」确指「{}」，请用「记住身份：{} = {}」登记别名。",
                                target, canonical, target, canonical
                            );
                        }
                        println!();
                        continue;
                    }
                }
                match symbolic_reasoner.admit_vocabulary(target) {
                    Some((name, etype)) => {
                        let mut properties = std::collections::HashMap::new();
                        properties.insert("name".to_string(), name.clone());
                        let items = vec![KnowledgeItem::Entity {
                            label: etype.clone(),
                            properties,
                        }];
                        match oxigraph_manager.save_knowledge(items).await {
                            Ok(()) => {
                                refresh_symbolic_reasoner(
                                    &mut symbolic_reasoner,
                                    &oxigraph_manager,
                                )
                                .await;
                                let _ =
                                    oxigraph_manager.save_entity_registry(&entity_registry_path);
                                println!(
                                    "已准入：「{}」→ 类型 {}（已写入知识库，可在后续问句中被识别）。",
                                    name, etype
                                );
                            }
                            Err(error) => println!("准入写入知识库失败：{}", error),
                        }
                    }
                    None => println!("词面为空，无法准入。"),
                }
            }
            println!();
            continue;
        }

        if let Some(entity) = strip_any_prefix(
            input,
            &["实体链接：", "实体链接:", "识别实体：", "识别实体:"],
        ) {
            match load_graph_snapshot(&oxigraph_manager).await {
                Ok(facts) => symbolic_reasoner.observe_facts(&facts),
                Err(e) => println!("加载知识图谱失败: {}", e),
            }
            let entity = entity.trim();
            if entity.is_empty() {
                println!("请提供要链接的文本，例如：实体链接：刘德桦的出生地");
            } else {
                println!("{}", symbolic_reasoner.entity_link_report(entity));
            }
            println!();
            continue;
        }

        if let Some(entity) = strip_any_prefix(
            input,
            &[
                "介绍：",
                "介绍:",
                "生成介绍：",
                "生成介绍:",
                "介绍一下",
                "讲讲",
                "说说",
            ],
        ) {
            let entity = entity.trim();
            if entity.is_empty() {
                println!("请提供要介绍的实体名，例如：介绍：姚明");
                println!();
                continue;
            }
            match load_graph_snapshot(&oxigraph_manager).await {
                Ok(facts) => symbolic_reasoner.observe_facts(&facts),
                Err(e) => println!("加载知识图谱失败: {}", e),
            }
            match symbolic_reasoner.generate_about(entity) {
                Some(text) => println!("{}", text),
                None => println!("知识图谱中没有关于「{}」的可生成事实。", entity),
            }
            println!();
            continue;
        }

        // 符号主义叙述：OWL 本体定义开场 + 层级/继承推理 + DCG 文法实现的多句连贯段落
        if let Some(entity) = strip_any_prefix(
            input,
            &[
                "叙述：",
                "叙述:",
                "讲述：",
                "讲述:",
                "详细叙述：",
                "详细叙述:",
            ],
        ) {
            let entity = entity.trim();
            if entity.is_empty() {
                println!("请提供要叙述的实体名，例如：叙述：姚明");
                println!();
                continue;
            }
            refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
            match symbolic_reasoner.narrate(entity) {
                Some(text) => println!("{}", text),
                None => println!("知识图谱中没有关于「{}」的叙述材料。", entity),
            }
            println!();
            continue;
        }

        // DCG 文法复述：对同一条事实给出多种合规句式（纯符号改写）
        if let Some(spec) = strip_any_prefix(input, &["复述：", "复述:", "换个句式：", "换个句式:"])
        {
            let spec = spec.trim();
            match spec.rsplit_once('的') {
                Some((entity, predicate))
                    if !entity.trim().is_empty() && !predicate.trim().is_empty() =>
                {
                    refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                    match symbolic_reasoner.paraphrase_fact(entity.trim(), predicate.trim(), 4) {
                        Some(text) => println!("{}", text),
                        None => println!(
                            "没有找到「{}」的「{}」相关事实。",
                            entity.trim(),
                            predicate.trim()
                        ),
                    }
                }
                _ => println!("用法：复述：<实体>的<属性>，例如：复述：姚明的身高"),
            }
            println!();
            continue;
        }

        // OWL 本体教学：把中文公理（子类/互斥/定义域/固有属性……）纳入本体推理
        if let Some(axiom) =
            strip_any_prefix(input, &["教本体：", "教本体:", "本体教学：", "本体教学:"])
        {
            let axiom = axiom.trim();
            if axiom.is_empty() {
                println!("请提供本体公理，例如：教本体：人是一种动物");
                println!();
                continue;
            }
            refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
            match symbolic_reasoner.teach_ontology(axiom) {
                Ok(summary) => println!("已学习本体公理：{}", summary),
                Err(e) => println!("本体公理解析失败：{}", e),
            }
            println!();
            continue;
        }

        if let Some(prefix) = strip_any_prefix(
            input,
            &["续写：", "续写:", "接龙：", "接龙:", "补全：", "补全:"],
        ) {
            let prefix = prefix.trim();
            if prefix.is_empty() {
                println!("请提供要续写的前缀，例如：续写：姚明的");
                println!();
                continue;
            }
            match load_graph_snapshot(&oxigraph_manager).await {
                Ok(facts) => symbolic_reasoner.observe_facts(&facts),
                Err(e) => println!("加载知识图谱失败: {}", e),
            }
            match symbolic_reasoner.predict_continuation(prefix) {
                Some(text) => println!("{}", text),
                None => println!("无法为「{}」生成续写（实体未知或不含「的」后缀）。", prefix),
            }
            println!();
            continue;
        }

        // 归结证明：对陈述句做 SLD 反驳并输出证明树（纯符号）
        if let Some(target) = strip_any_prefix(
            input,
            &[
                "证明：",
                "证明:",
                "归结：",
                "归结:",
                "证明目标：",
                "证明目标:",
            ],
        ) {
            let target = target.trim();
            if target.is_empty() {
                println!("请提供要证明的陈述，例如：证明：张三的师祖是王五");
            } else {
                refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                println!("{}", symbolic_reasoner.resolution_proof_report(target));
                if let Some(view) = symbolic_reasoner.resolution_proof_view(target) {
                    let certificates = ProofCertificate::from_view(&view, ProofLimits::default());
                    if let Some(chronicle) = chronicle.as_mut() {
                        let _ = know_prolog::proof_journal::record_proofs(chronicle, &certificates);
                    }
                    if let Some(certificate) = certificates.first() {
                        println!(
                            "证明编号：{}（已写入证明日志；用「证明编号：{}」回放）",
                            certificate.id, certificate.id
                        );
                    }
                }
            }
            println!();
            continue;
        }

        // 推理链：对开放目标求全部归结解，逐解展示证明链
        if let Some(target) =
            strip_any_prefix(input, &["推理链：", "推理链:", "寻找证明：", "寻找证明:"])
        {
            let target = target.trim();
            if target.is_empty() {
                println!("请提供推理目标，例如：推理链：张三的师傅的师傅是谁");
            } else {
                refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                println!("{}", symbolic_reasoner.resolution_query_report(target));
                if let Some(view) = symbolic_reasoner.resolution_proof_view(target) {
                    let certificates = ProofCertificate::from_view(&view, ProofLimits::default());
                    if let Some(chronicle) = chronicle.as_mut() {
                        let _ = know_prolog::proof_journal::record_proofs(chronicle, &certificates);
                    }
                    if let Some(certificate) = certificates.first() {
                        println!(
                            "证明编号：{}（已写入证明日志；用「证明编号：{}」回放）",
                            certificate.id, certificate.id
                        );
                    }
                }
            }
            println!();
            continue;
        }

        // 关联规则报告：支持度/置信度/提升度/杠杆率（可带过滤词）
        if input == "关联规则" || input == "关联规则报告" {
            refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
            println!("{}", symbolic_reasoner.association_report(None));
            println!();
            continue;
        }
        if let Some(filter) = strip_any_prefix(input, &["关联规则：", "关联规则:"]) {
            refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
            println!("{}", symbolic_reasoner.association_report(Some(filter)));
            println!();
            continue;
        }

        // 联想：由关联规则推荐实体可能还具备的缺失特征
        if let Some(entity) =
            strip_any_prefix(input, &["联想：", "联想:", "关联联想：", "关联联想:"])
        {
            let entity = entity.trim();
            if entity.is_empty() {
                println!("请提供要联想的实体，例如：联想：姚明");
            } else {
                refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                match symbolic_reasoner.association_suggestions(entity) {
                    Some(text) => println!("{}", text),
                    None => {
                        println!("没有可用的关联联想（实体未见过，或没有满足阈值的关联规则）。")
                    }
                }
            }
            println!();
            continue;
        }

        // 可修订推测：链接预测只作为生成假设呈现，确认/否定后才影响会话。
        if input.contains("预测")
            || input.starts_with("推测：")
            || input.starts_with("推测:")
            || matches!(input, "推测台账" | "推测报告")
        {
            refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
            if let Some(text) = symbolic_reasoner.handle_predictive_command(input) {
                println!("{}", text);
                println!();
                continue;
            }
        }

        // 镜像对齐：会话级词形规则（用户说法 ↔ 规范身份），可正读、可反写。
        if input.contains("对齐")
            || input.contains("镜像记忆")
            || input.starts_with("镜像查询")
            || input.starts_with("镜像答法")
        {
            refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
            if let Some(text) = symbolic_reasoner.handle_alignment_command(input) {
                println!("{}", text);
                println!();
                continue;
            }
        }

        // 补全回合：打开一次可确认、可归纳、可批量闭包的学习回合。
        // 解析与 HTTP 服务共用同一入口，避免两套命令文法漂移。
        if input.contains("补全") || input.starts_with("学习回合") {
            refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
            if let Some(text) = symbolic_reasoner.handle_completion_command(input) {
                println!("{}", text);
                println!();
                continue;
            }
        }

        // 建设性推断：查询、确认或否认只在本次会话生效的虚拟事实。
        if input.contains("推断") {
            refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
            if let Some(text) = symbolic_reasoner.handle_constructive_command(input) {
                println!("{}", text);
                println!();
                continue;
            }
        }

        // 知识容器层次清单：把回答所依赖的四层前提摊开，供逐层审计。
        if matches!(
            input,
            "知识层次" | "层次清单" | "容器清单" | "会话镜像" | "回答层次"
        ) {
            refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
            println!("{}", symbolic_reasoner.knowledge_layers_report());
            println!();
            continue;
        }

        // 结构联想：RWR 图嵌入（随机游走重启）检索结构相关实体与解释路径
        if let Some(entity) =
            strip_any_prefix(input, &["结构联想：", "结构联想:", "图游走：", "图游走:"])
        {
            let entity = entity.trim();
            if entity.is_empty() {
                println!("请提供要联想的实体，例如：结构联想：张三");
            } else {
                refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                println!("{}", symbolic_reasoner.structural_related_report(entity));
            }
            println!();
            continue;
        }

        // 话题延伸：以对话焦点实体栈为多源种子的 RWR 图游走联想
        if matches!(input, "话题延伸" | "顺着话题" | "话题联想") {
            refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
            println!("{}", symbolic_reasoner.topic_extension_report());
            println!();
            continue;
        }

        // 全息符号记忆（VSA/HDC）：状态报告与召回自检（纯符号、零训练）
        if matches!(input, "全息状态" | "全息记忆" | "全息自检" | "全息报告") {
            refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
            println!("{}", symbolic_reasoner.holographic().report());
            println!();
            continue;
        }

        // 全息检索：软召回（绑定/解缠/净噪）→ 硬核查（三元组登记）
        if let Some(target) = strip_any_prefix(
            input,
            &[
                "全息检索：",
                "全息检索:",
                "全息查询：",
                "全息查询:",
                "全息补全：",
                "全息补全:",
            ],
        ) {
            match parse_holo_pair(target) {
                Some((subject, predicate)) => {
                    refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                    let hits = symbolic_reasoner
                        .holographic()
                        .recall_object(&subject, &predicate);
                    println!(
                        "{}",
                        render_holo_hits(
                            &format!("全息召回「{} --{}--> ?」", subject, predicate),
                            &hits
                        )
                    );
                }
                None => println!(
                    "请给出「主体 谓词」，例如：全息检索：张三 师傅 或 全息检索：张三的师傅"
                ),
            }
            println!();
            continue;
        }

        // 全息反查：(谓词, 客体) → ?主体
        if let Some(target) = strip_any_prefix(input, &["全息反查：", "全息反查:"]) {
            match parse_holo_pair(target) {
                Some((predicate, object)) => {
                    refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                    let hits = symbolic_reasoner
                        .holographic()
                        .recall_subject(&predicate, &object);
                    println!(
                        "{}",
                        render_holo_hits(
                            &format!("全息反查「? --{}--> {}」", predicate, object),
                            &hits
                        )
                    );
                }
                None => println!("请给出「谓词 客体」，例如：全息反查：师傅 李四"),
            }
            println!();
            continue;
        }

        // 全息类比：A:B :: C:? —— 解缠关系谓词（硬核查通过）再迁移绑定
        if let Some(target) = strip_any_prefix(input, &["全息类比：", "全息类比:"]) {
            match parse_holo_terms(target, 3) {
                Some(terms) => {
                    refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                    let hits = symbolic_reasoner
                        .holographic()
                        .analogy(&terms[0], &terms[1], &terms[2]);
                    println!(
                        "{}",
                        render_holo_analogy(&terms[0], &terms[1], &terms[2], &hits)
                    );
                }
                None => println!(
                    "请给出三个词，例如：全息类比：张三 李四 王五（意为 张三:李四 :: 王五:?）"
                ),
            }
            println!();
            continue;
        }

        // 加权软推理：MLN-lite 加权规则 + 软真值传播（输出概率与模糊限定词）
        if matches!(input, "软推理" | "加权推理") {
            refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
            println!("{}", symbolic_reasoner.soft_facts_report(8));
            println!();
            continue;
        }
        if let Some(target) =
            strip_any_prefix(input, &["软推理：", "软推理:", "加权推理：", "加权推理:"])
        {
            let target = target.trim();
            if target.is_empty() {
                println!("请提供软推理对象，例如：软推理：张三 或 软推理：张三的师傅");
            } else {
                refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                println!("{}", symbolic_reasoner.soft_inference_report(target));
            }
            println!();
            continue;
        }

        // 因果反事实/干预：do(X=真/假) 后对比"事实世界 vs 干预世界"的下游概率差
        if let Some(target) = strip_any_prefix(
            input,
            &[
                "因果反事实：",
                "因果反事实:",
                "反事实干预：",
                "反事实干预:",
                "干预：",
                "干预:",
                "do：",
                "do:",
            ],
        ) {
            let (target, value) = parse_intervention_value(target.trim());
            if target.is_empty() {
                println!("请提供要干预的对象，例如：干预：张三的师傅 或 干预：张三的师傅=真");
            } else {
                refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                println!("{}", symbolic_reasoner.counterfactual_report(target, value));
            }
            println!();
            continue;
        }

        // 因果推理：因果图 + 直接原因 + 事实根因链 + 混杂因子（do-演算视角）
        if let Some(target) = strip_any_prefix(
            input,
            &[
                "因果链：",
                "因果链:",
                "因果关系：",
                "因果关系:",
                "因果：",
                "因果:",
            ],
        ) {
            let target = target.trim();
            if target.is_empty() {
                println!("请提供要因果解释的对象，例如：因果：李四的师祖 或 因果：张三");
            } else {
                refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                println!("{}", symbolic_reasoner.causal_report(target));
            }
            println!();
            continue;
        }

        // 符号梯度归因：可微推理的纯符号版（解析求导 + 求证优先级）
        if let Some(target) = strip_any_prefix(
            input,
            &["梯度归因：", "梯度归因:", "可微推理：", "可微推理:"],
        ) {
            let target = target.trim();
            if target.is_empty() {
                println!("请提供要归因的对象，例如：梯度归因：李四的师祖 或 梯度归因：张三");
            } else {
                refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                println!("{}", symbolic_reasoner.gradient_report(target));
            }
            println!();
            continue;
        }

        // 语境分词：符号子词嵌入 + 字符格 k-best + 焦点语境；只读审计，不影响回答主路径
        if let Some(target) = strip_any_prefix(
            input,
            &["语境分词：", "语境分词:", "读法分词：", "读法分词:"],
        ) {
            let target = target.trim();
            if target.is_empty() {
                println!("请提供要分词的文本，例如：语境分词：张三的师傅是谁");
            } else {
                refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                println!("{}", symbolic_reasoner.contextual_segment_report(target));
            }
            println!();
            continue;
        }

        // 联合解析：关系帧反向约束分词（帧锚定焊接 + 类型裁决 + 读法归一）
        if let Some(target) = strip_any_prefix(input, &["联合解析：", "联合解析:"]) {
            let target = target.trim();
            if target.is_empty() {
                println!("请提供要联合解析的文本，例如：联合解析：苹果的创始人是乔布斯");
            } else {
                refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                println!(
                    "{}",
                    symbolic_reasoner.joint_parse_report_and_observe(target)
                );
            }
            println!();
            continue;
        }

        // 语境分词确认/纠正：把用户裁决写进会话级读法记忆（键 = 当前焦点语境指纹）
        if let Some(spec) = strip_any_prefix(
            input,
            &["语境分词确认：", "语境分词确认:", "读法确认：", "读法确认:"],
        ) {
            refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
            match spec.trim().split_once('=') {
                Some((surface, canonical))
                    if !surface.trim().is_empty() && !canonical.trim().is_empty() =>
                {
                    println!(
                        "{}",
                        symbolic_reasoner
                            .confirm_contextual_reading(surface.trim(), canonical.trim())
                    );
                }
                _ => println!("请提供「词面=规范读法」，例如：语境分词确认：苹果=苹果公司"),
            }
            println!();
            continue;
        }

        if let Some(spec) = strip_any_prefix(
            input,
            &["语境分词纠正：", "语境分词纠正:", "读法纠正：", "读法纠正:"],
        ) {
            refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
            let parsed = spec.trim().split_once('=').and_then(|(surface, rest)| {
                rest.split_once('>')
                    .map(|(rejected, accepted)| (surface, rejected, accepted))
            });
            match parsed {
                Some((surface, rejected, accepted))
                    if !surface.trim().is_empty()
                        && !rejected.trim().is_empty()
                        && !accepted.trim().is_empty() =>
                {
                    println!(
                        "{}",
                        symbolic_reasoner.correct_contextual_reading(
                            surface.trim(),
                            rejected.trim(),
                            accepted.trim()
                        )
                    );
                }
                _ => {
                    println!("请提供「词面=旧读法>新读法」，例如：语境分词纠正：苹果=苹果>苹果公司")
                }
            }
            println!();
            continue;
        }

        if matches!(
            input,
            "语境记忆" | "读法记忆" | "语境分词记忆" | "读法记忆报告"
        ) {
            println!("{}", symbolic_reasoner.contextual_reading_memory_report());
            println!();
            continue;
        }

        if matches!(input, "清空语境记忆" | "清空读法记忆" | "忘记读法记忆") {
            println!("{}", symbolic_reasoner.clear_contextual_reading_memory());
            println!();
            continue;
        }

        // 因果分词：知识/因果图引导的 token 格 k-best DP + token 级因果归因（纯符号 attention）
        if let Some(target) = strip_any_prefix(
            input,
            &[
                "因果分词：",
                "因果分词:",
                "分词：",
                "分词:",
                "token归因：",
                "token归因:",
            ],
        ) {
            let target = target.trim();
            if target.is_empty() {
                println!("请提供要分词的文本，例如：因果分词：张三的师父是谁");
            } else {
                refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                println!("{}", symbolic_reasoner.causal_tokenize_report(target));
            }
            println!();
            continue;
        }

        // 概率可能世界精确推理：ProbLog 风格 BDD 加权模型计数（概率事实/规则/注释析取）
        if let Some(target) = strip_any_prefix(
            input,
            &[
                "概率推理：",
                "概率推理:",
                "可能世界：",
                "可能世界:",
                "ProbLog：",
                "ProbLog:",
                "probLog：",
                "probLog:",
            ],
        ) {
            let target = target.trim();
            if target.is_empty() {
                println!("请提供要精确查询的对象，例如：概率推理：李四的师祖 或 概率推理：李四");
            } else {
                refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                println!("{}", symbolic_reasoner.problog_report(target));
            }
            println!();
            continue;
        }

        // 默认逻辑推理：可反驳常识（通常…除非…）的接地良基扩展 + 例外阻断 + 概率视图对照
        if let Some(target) = strip_any_prefix(
            input,
            &["默认推理：", "默认推理:", "默认逻辑：", "默认逻辑:"],
        ) {
            refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
            println!("{}", symbolic_reasoner.default_logic_report(target.trim()));
            println!();
            continue;
        }

        // 中文概率知识教学：自然语言句子 → ProbLog 概率知识（会话级，不写知识图谱）
        if let Some(target) = strip_any_prefix(
            input,
            &[
                "教学：",
                "教学:",
                "教我：",
                "教我:",
                "概率知识：",
                "概率知识:",
            ],
        ) {
            let target = target.trim();
            if target.is_empty() {
                println!(
                    "请给出要教的中文知识，例如：教学：如果张三的师傅是李四，那么张三的师祖是王五"
                );
            } else {
                refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                println!("{}", symbolic_reasoner.teach_report(target));
                sync_rule_ledger_report(&symbolic_reasoner, &oxigraph_manager).await;
            }
            println!();
            continue;
        }

        // 规则预演：解析但不入会话规则库，先看重复/不安全变量/功能冲突与派生影响
        if let Some(target) = strip_any_prefix(
            input,
            &["规则预演：", "规则预演:", "教规则预演：", "教规则预演:"],
        ) {
            let target = target.trim();
            if target.is_empty() {
                println!("请给出中文规则，例如：规则预演：如果X的师傅是Y，那么X的老师是Y");
            } else {
                refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                println!("{}", symbolic_reasoner.belief_rule_preview(target));
            }
            println!();
            continue;
        }

        // 中文规则学习：教一条 Horn 规则（含占位符自动变量化），直接进入后向链
        if let Some(target) = strip_any_prefix(
            input,
            &[
                "规则：",
                "规则:",
                "教规则：",
                "教规则:",
                "学习规则：",
                "学习规则:",
            ],
        ) {
            let target = target.trim();
            if target.is_empty() {
                println!("请给出中文规则，例如：规则：如果X的师傅是Y，那么X的老师是Y");
            } else {
                refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                let report = symbolic_reasoner.learn_rule_report(target);
                let learned = report.contains("已学习中文规则");
                println!("{}", report);
                if let Some(chronicle) = chronicle.as_mut() {
                    let _ =
                        conversation_theory::append_rule(chronicle, target, learned, kb_revision);
                }
                sync_rule_ledger_report(&symbolic_reasoner, &oxigraph_manager).await;
            }
            println!();
            continue;
        }

        // 信念一致性：提交前闸门（功能谓词多值 / 定义域 / 值域 / 规则形状）
        if matches!(input, "信念检查" | "信念一致性" | "推理一致性") {
            refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
            println!("{}", symbolic_reasoner.belief_check_report());
            println!();
            continue;
        }

        // 会话修订回执：查看最近写入及其可撤销操作
        if matches!(input, "修订回执" | "修订记录" | "会话修订" | "写入回执") {
            println!("{}", symbolic_reasoner.session_revision_report());
            println!();
            continue;
        }

        // 撤销最新一次会话修订：按回执逆操作恢复事实/规则/教学原句
        if matches!(input, "撤销修订" | "撤销上次修订" | "撤回修订" | "撤销写入")
        {
            let report = symbolic_reasoner.undo_last_session_revision();
            println!("{}", report);
            sync_rule_ledger_report(&symbolic_reasoner, &oxigraph_manager).await;
            log_chronicle_reply(&mut chronicle, &report, &[], &[]);
            println!();
            continue;
        }

        // 会话改口：显式更正一个功能槽，记录 before/after，不写知识图谱
        if let Some(target) = strip_any_prefix(
            input,
            &[
                "改口：",
                "改口:",
                "更正记忆：",
                "更正记忆:",
                "纠正记忆：",
                "纠正记忆:",
            ],
        ) {
            let target = target.trim();
            if target.is_empty() {
                println!("请给出改口内容，例如：改口：姚明的身高是229cm");
            } else {
                refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                let report = symbolic_reasoner.correct_nl_fact(target);
                println!("{}", report);
                log_chronicle_reply(&mut chronicle, &report, &[], &[]);
            }
            println!();
            continue;
        }

        // 会话槽账本：当前赢家 / 压制值 / 仅证据值
}

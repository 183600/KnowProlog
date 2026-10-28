
async fn load_graph_snapshot(
    oxigraph_manager: &OxigraphManager,
) -> Result<
    Arc<Vec<know_prolog::prolog_engine::PrologFact>>,
    Box<dyn std::error::Error + Send + Sync>,
> {
    oxigraph_manager.load_graph_snapshot().await
}

/// 把对话层已确认的会话事实桥接为推理前提（幂等；永不写知识图谱）。
///
/// 确认/更正发生在对话管理器里，推理器需要同一组事实才能让多跳链、
/// 证明与会话快照看到它们；返回本轮新增条数。
fn sync_recovery_confirmed_facts(
    dialogue_manager: &DialogueManager,
    reasoner: &mut SymbolicReasoner,
) -> usize {
    let state = dialogue_manager.recovery_state();
    for fact in &state.confirmed {
        reasoner.note_dialogue_verdict(
            &fact.entity,
            &fact.attribute,
            &fact.value,
            true,
            &fact.receipt_id,
        );
    }
    for candidate in &state.refuted {
        reasoner.note_dialogue_verdict(
            &candidate.entity,
            &candidate.attribute,
            &candidate.value,
            false,
            &candidate.receipt_id,
        );
    }
    state
        .confirmed
        .iter()
        .filter(|fact| reasoner.adopt_session_triple(&fact.entity, &fact.attribute, &fact.value))
        .count()
}

async fn refresh_symbolic_reasoner(
    symbolic_reasoner: &mut SymbolicReasoner,
    oxigraph_manager: &OxigraphManager,
) {
    if let Ok(facts) = load_graph_snapshot(oxigraph_manager).await {
        symbolic_reasoner.observe_facts(&facts);
        oxigraph_manager.observe_entity_facts(&facts);
    }
}

/// 重建推理器的全局知识（用于删值/更正）：保留会话态，只携带当前三元组。
async fn rebuild_symbolic_reasoner(
    symbolic_reasoner: &mut SymbolicReasoner,
    oxigraph_manager: &OxigraphManager,
) {
    if let Ok(facts) = load_graph_snapshot(oxigraph_manager).await {
        symbolic_reasoner.rebuild_from_graph(&facts);
        oxigraph_manager.observe_entity_facts(&facts);
    }
}

/// 规则图账本同步：把当前会话规则全量重写进命名图（幂等；不推进知识版本）。
async fn sync_rule_ledger_report(
    symbolic_reasoner: &SymbolicReasoner,
    oxigraph_manager: &OxigraphManager,
) {
    let entries = symbolic_reasoner.rule_ledger_entries();
    match oxigraph_manager.sync_rule_ledger(&entries).await {
        Ok(0) => println!("规则图账本已同步：0 条规则（命名图 kg:graphs/rules 已置空）。"),
        Ok(count) => println!(
            "规则图账本已同步：{} 条规则 → 命名图 kg:graphs/rules（指纹 {}）。",
            count,
            symbolic_reasoner.rule_fingerprint()
        ),
        Err(error) => eprintln!("规则图账本同步失败：{}", error),
    }
}

/// 图账本审计：规则账本与派生账本读取 + 与当前会话对账（指纹/版本过期判定）。
async fn graph_ledger_report(
    symbolic_reasoner: &SymbolicReasoner,
    oxigraph_manager: &OxigraphManager,
) -> String {
    let mut output = String::from("图账本审计（命名图物化视图，与知识图谱隔离）：\n");
    match oxigraph_manager.load_rule_ledger().await {
        Ok(ledger) => {
            if !ledger.synced {
                output.push_str(
                    "■ 规则账本 kg:graphs/rules：尚未同步过（教一条「规则：…」即会入账）。\n",
                );
            } else {
                output.push_str(&format!(
                    "■ 规则账本 kg:graphs/rules：{} 条规则，规则集指纹 {}。\n",
                    ledger.entries.len(),
                    ledger.fingerprint
                ));
                for (index, entry) in ledger.entries.iter().take(10).enumerate() {
                    output.push_str(&format!(
                        "  {}. {}（来源：{}；置信度 {:.2}）\n",
                        index + 1,
                        entry.display_zh(),
                        entry.source,
                        entry.confidence
                    ));
                }
                if ledger.entries.len() > 10 {
                    output.push_str(&format!("  …共 {} 条。\n", ledger.entries.len()));
                }
                let current = symbolic_reasoner.rule_fingerprint();
                if ledger.fingerprint == current {
                    output.push_str("  对账：与当前会话规则集一致。\n");
                } else {
                    output.push_str(&format!(
                        "  对账：账本指纹 {} ≠ 当前会话指纹 {}——会话有未同步变更。\n",
                        ledger.fingerprint, current
                    ));
                }
            }
        }
        Err(error) => output.push_str(&format!("■ 规则账本读取失败：{}\n", error)),
    }
    match oxigraph_manager.load_derived_ledger().await {
        Ok(ledger) => {
            if !ledger.materialized {
                output
                    .push_str("■ 派生账本 kg:graphs/derived：尚未物化过（用「物化推理」写入）。\n");
            } else {
                output.push_str(&format!(
                    "■ 派生账本 kg:graphs/derived：{} 条派生事实，物化于知识版本 {}，规则指纹 {}。\n",
                    ledger.entries.len(),
                    ledger.revision,
                    ledger.fingerprint
                ));
                for (index, entry) in ledger.entries.iter().take(10).enumerate() {
                    let premises = entry
                        .premises
                        .iter()
                        .map(|premise| premise.display())
                        .collect::<Vec<_>>()
                        .join("、");
                    output.push_str(&format!(
                        "  {}. {} ← {}（轮次 {}，置信度 {:.2}；前提：{}）\n",
                        index + 1,
                        entry.atom.display(),
                        entry.rule_source,
                        entry.depth,
                        entry.confidence,
                        premises
                    ));
                }
                if ledger.entries.len() > 10 {
                    output.push_str(&format!("  …共 {} 条。\n", ledger.entries.len()));
                }
                let current_revision = oxigraph_manager.revision();
                let current_fingerprint = symbolic_reasoner.rule_fingerprint();
                if ledger.revision == current_revision && ledger.fingerprint == current_fingerprint
                {
                    output.push_str("  对账：物化戳与当前知识版本、规则集一致。\n");
                } else {
                    output.push_str(&format!(
                        "  对账：已过期（物化于版本 {} / 指纹 {}；当前版本 {} / 指纹 {}）——用「物化推理」重写。\n",
                        ledger.revision, ledger.fingerprint, current_revision, current_fingerprint
                    ));
                }
            }
        }
        Err(error) => output.push_str(&format!("■ 派生账本读取失败：{}\n", error)),
    }
    output.trim_end().to_string()
}

/// 从知识图谱快照抽取关系边（子图视图的输入；确定性排序由视图层负责）。
fn prolog_fact_edges(facts: &[PrologFact]) -> Vec<(String, String, String)> {
    facts
        .iter()
        .filter_map(|fact| match fact {
            PrologFact::Relation { from, rel_type, to } => {
                Some((from.clone(), rel_type.clone(), to.clone()))
            }
            PrologFact::Entity { .. } => None,
        })
        .collect()
}

/// 把同一实体线索下问过的问题接到情景理论上，列出跨会话「同题异答」。
///
/// 只复用已有的 AnswerChange 判定，不在足迹里另造冲突规则；没有改口就静默。
fn entity_related_changes(
    digest: &EntityThreadDigest,
    chronicle: &Chronicle,
    query: &str,
) -> Option<String> {
    let ThreadLookup::Found(thread) = digest.lookup(query) else {
        return None;
    };
    let questions: BTreeSet<String> = thread
        .events
        .iter()
        .filter_map(|event| event.question.as_deref())
        .map(str::trim)
        .filter(|question| !question.is_empty())
        .map(str::to_string)
        .collect();
    if questions.is_empty() {
        return None;
    }
    let theory = ConversationTheory::build(chronicle);
    let mut seen_keys = BTreeSet::new();
    let mut lines = Vec::new();
    for question in questions {
        let Some(change) = theory.latest_change(&question) else {
            continue;
        };
        if !seen_keys.insert(change.question_key.clone()) {
            continue;
        }
        lines.push(format!(
            "  「{}」：{}（rev {}，seq {}）→ {}（rev {}，seq {}）；{}",
            clip_text(&change.question, 40),
            clip_text(&change.previous.answer, 48),
            change.previous.revision,
            change.previous.reply_seq,
            clip_text(&change.current.answer, 48),
            change.current.revision,
            change.current.reply_seq,
            change.kind.label()
        ));
    }
    if lines.is_empty() {
        return None;
    }
    let mut output = vec!["相关改口（跨会话同题异答）：".to_string()];
    output.extend(lines);
    output.push("  用「情景归因：<问题>」查看候选原因。".to_string());
    Some(output.join("\n"))
}

fn clip_text(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_string();
    }
    let mut clipped: String = text.chars().take(limit).collect();
    clipped.push('…');
    clipped
}

/// 构建实体脉络投影，并尽量附上当前图谱快照（失败只告警，不影响足迹）。
async fn entity_thread_context(
    chronicle: &Chronicle,
    oxigraph_manager: &OxigraphManager,
) -> (EntityThreadDigest, Arc<Vec<PrologFact>>) {
    let registry = oxigraph_manager.entity_registry_snapshot();
    let digest = EntityThreadDigest::build(chronicle, &registry);
    let facts = match load_graph_snapshot(oxigraph_manager).await {
        Ok(facts) => facts,
        Err(error) => {
            eprintln!("图谱快照读取失败（足迹不显示图谱现状）：{}", error);
            Arc::new(Vec::new())
        }
    };
    (digest, facts)
}

/// 编年史写端的实体引用：只记链接器本轮真正看到的词面/身份。
///
/// 焦点不在这里重复落账：`Chronicle` 的 turn 已带 `focus`，读端只在
/// 代词/继续等「接着聊」轮次把焦点当指代证据，避免新话题污染旧实体足迹。
/// 只写「当时看到的身份」，不写当前显示名；日志永不回写，合并后的投影由
/// EntityThreadDigest 在读端完成。
fn chronicle_entity_refs(reasoner: &SymbolicReasoner, text: &str) -> Vec<EntityRef> {
    let mut refs = Vec::new();
    let mut seen = BTreeSet::new();
    for mention in reasoner.grounding_mentions(text) {
        if mention.score < SymbolicReasoner::LINK_ACCEPT_THRESHOLD {
            continue;
        }
        let surface = mention.mention.trim();
        let entity = mention.entity.trim();
        if surface.is_empty() || entity.is_empty() {
            continue;
        }
        if seen.insert((surface.to_string(), entity.to_string())) {
            refs.push(EntityRef::linked(surface, entity));
        }
    }
    refs
}

/// 编年史回复登记：与写前日志配对，失败不打断主流程。
fn log_chronicle_reply(
    chronicle: &mut Option<Chronicle>,
    answer: &str,
    plan: &[String],
    provenance: &[String],
) {
    if let Some(chronicle) = chronicle.as_mut() {
        let _ = chronicle.append_reply(answer, plan, provenance);
    }
}

/// 编年史轨迹归档：把对话管理器最近一轮的分层追踪写入 `kind="trace"`，
/// 重启后可据此回放问据栈/指代轨迹（不重放决策）。
fn log_chronicle_trace(chronicle: &mut Option<Chronicle>, manager: &DialogueManager) {
    if manager.trace_mode().is_off() {
        return;
    }
    if let (Some(chronicle), Some(trace)) = (chronicle.as_mut(), manager.last_turn_trace_json()) {
        let _ = chronicle.append("trace", trace);
    }
}

/// 外链候选实体：预计算实体在前、本轮解析实体在后，去重保序。
fn external_candidates(entities: &[String], precomputed: &[String]) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut merged = Vec::new();
    for entity in precomputed.iter().chain(entities.iter()) {
        if seen.insert(entity.clone()) {
            merged.push(entity.clone());
        }
    }
    merged
}

/// 外链结果的可展示文本（回答带出处标注）；`None` 表示没有可用信息。
fn external_outcome_text(outcome: &AnswerOutcome) -> Option<String> {
    match outcome {
        AnswerOutcome::Answered(answer) => {
            Some(format!("根据外部知识：{}", answer.with_provenance()))
        }
        AnswerOutcome::KnownAbsent {
            entity,
            source_uri,
            fetched_tick,
        } => Some(format!(
            "外部知识源已确认未收录「{}」（来源：{}，第 {} 轮核验）。",
            entity, source_uri, fetched_tick
        )),
        AnswerOutcome::Unknown => None,
    }
}

/// 交互式导入指令的解析结果。
#[derive(Debug, PartialEq, Eq)]
enum ImportDirective {
    JsonldFile(String),
    JsonldContent(String),
    RdfFile(String),
}

/// 解析「导入…文件：路径」/「导入…内容：内容」指令。
///
/// 兼容全角「：」与半角「:」，并接受 README 中的简写「导入文件：」「导入内容：」；
/// 只剥离前缀关键词，参数正文不做替换（路径/JSON 中再次出现关键词也不会被删改）。
fn parse_import_directive(input: &str) -> Option<ImportDirective> {
    let trimmed = input.trim();
    let lower = trimmed.to_ascii_lowercase();
    let groups: [(&[&str], fn(String) -> ImportDirective); 3] = [
        (
            &[
                "导入jsonld内容",
                "导入内容",
                "import jsonld content",
                "import content",
            ],
            ImportDirective::JsonldContent,
        ),
        (
            &[
                "导入jsonld文件",
                "导入文件",
                "import jsonld file",
                "import jsonld",
            ],
            ImportDirective::JsonldFile,
        ),
        (
            &["导入rdf文件", "导入rdf", "import rdf"],
            ImportDirective::RdfFile,
        ),
    ];
    for (keywords, build) in groups {
        for keyword in keywords {
            let needle = keyword.to_ascii_lowercase();
            for colon in [':', '：'] {
                let prefix = format!("{}{}", needle, colon);
                if lower.starts_with(&prefix) {
                    return Some(build(trimmed[prefix.len()..].trim().to_string()));
                }
            }
        }
    }
    None
}

/// 解析「撤销批次：<引用>」指令（兼容全角/半角冒号与英文 retract batch）。
fn parse_retract_batch_directive(input: &str) -> Option<String> {
    let trimmed = input.trim();
    let lower = trimmed.to_ascii_lowercase();
    for keyword in ["撤销批次", "retract batch"] {
        let needle = keyword.to_ascii_lowercase();
        for colon in [':', '：'] {
            let prefix = format!("{}{}", needle, colon);
            if lower.starts_with(&prefix) {
                let reference = trimmed[prefix.len()..].trim();
                if !reference.is_empty() {
                    return Some(reference.to_string());
                }
            }
        }
    }
    None
}

/// 交互式导出指令的解析结果。
#[derive(Debug, PartialEq, Eq)]
enum ExportDirective {
    RdfFile(String),
}

/// 解析「导出rdf文件：路径」指令（兼容全角/半角冒号与「导出文件：」简写）。
fn parse_export_directive(input: &str) -> Option<ExportDirective> {
    let trimmed = input.trim();
    let lower = trimmed.to_ascii_lowercase();
    for keyword in ["导出rdf文件", "导出rdf", "导出文件", "export rdf"] {
        let needle = keyword.to_ascii_lowercase();
        for colon in [':', '：'] {
            let prefix = format!("{}{}", needle, colon);
            if lower.starts_with(&prefix) {
                return Some(ExportDirective::RdfFile(
                    trimmed[prefix.len()..].trim().to_string(),
                ));
            }
        }
    }
    None
}

/// 解析「SPARQL：<查询>」指令（兼容全角/半角冒号与 `sparql <查询>` 空格形式）。
///
/// 只剥离前缀，查询正文原样保留（其中的冒号/关键词不受影响）。
fn parse_sparql_command(input: &str) -> Option<&str> {
    let trimmed = input.trim();
    let lower = trimmed.to_ascii_lowercase();
    for (needle, suffix_len) in [("sparql：", "sparql：".len()), ("sparql:", "sparql:".len())] {
        if lower.starts_with(needle) {
            return Some(trimmed[suffix_len..].trim());
        }
    }
    if lower.starts_with("sparql") {
        let rest = &trimmed["sparql".len()..];
        if rest.starts_with(|c: char| c.is_whitespace()) {
            return Some(rest.trim());
        }
    }
    // 中文等价的「查询sparql：」（半角/全角冒号长度不同，按实际前缀切分）
    if lower.starts_with("查询sparql：") {
        return Some(trimmed["查询SPARQL：".len()..].trim());
    }
    if lower.starts_with("查询sparql:") {
        return Some(trimmed["查询SPARQL:".len()..].trim());
    }
    None
}

/// 交互模式命令速查（只列常用命令，完整命令集见各分支实现与 CLAUDE.md）。
fn interactive_help() -> String {
    [
        "KnowProlog 交互命令速查（输入「帮助 / help / ?」显示本页）",
        "",
        "知识录入",
        "  录入：<文本> / 添加：<文本>        自然语言抽取实体与关系（如「录入：小明18岁」）",
        "  导入jsonld文件：<路径>              导入 JSON-LD 文件（简写：导入文件：）",
        "  导入jsonld内容：<JSON>              直接导入 JSON-LD 内容（简写：导入内容：）",
        "  导入rdf文件：<路径>                 导入 Turtle/N-Triples/N-Quads/TriG/RDF/XML",
        "  批次台账 / 导入批次                  查看 RDF 导入情景（批次号、来源、新增/重复）",
        "  这批简报 / 上一批 / 第2批           按批次线索回指并复述（样例带 [B序号-序号] 引用）",
        "  导出rdf文件：<路径>                 导出默认图（按扩展名，支持 jsonld；简写：导出文件：）",
        "  撤销导入 / 导入回执 / 导入历史       导入回执与可撤销导入栈",
        "  导入帧 / 语言帧 / 帧台账             导入句学到的措辞、问法与原句见证",
        "  情景 / 对话情景                      把编年史编译成可查询的情景理论（CST）",
        "  情景历史：<问题>                     同一问题在各情景下的回答与修订",
        "  情景归因：<问题>                     改口的候选事件原因（无候选就明说）",
        "  情景反事实：<问题>                   撤销最近语义事件后回答是否翻转（可弃权）",
        "  情景查询：<Datalog原子>              在情景理论上执行查询，如 answered_at(S,Q,A,P)",
        "  事件图：<文本>                      抽取事件节点、角色、时间、因果与先后链接",
        "  事件问答：<问题>                    事件焦点多轮追问（谁/何时/何地/为什么/后来/还有谁/是否）",
        "  事件事实 / 事件焦点                 导出事件命名图 N-Quads / 查看事件焦点栈",
        "",
        "查询与推理",
        "  直接输入自然语言                    如「小明多大？」「张三和李四什么关系？」",
        "  列出所有人的身高 / 哪些实体          查询板：集合查询后用「第二个 / 下一页 / 只看… / 按…排序 / 跟…比 / 依据」继续编辑",
        "  查询板 / 查询状态                    查看当前查询程序、结果行数与知识版本",
        "  查询剧本：<file.toml>              回放查询板剧本（每轮断言算子/行数/页码/选中/证据），自检交互行为",
        "  SPARQL：<查询>                      执行任意 SPARQL 1.1 查询（SELECT/ASK/CONSTRUCT）",
        "  判断：<陈述> / 蕴含：<陈述>         形式化语义 + 语义蕴含校验",
        "  语义分析：<句子>                    候选逻辑形式森林 + 图谱接地（歧义/证据缺口）",
        "  证明：<目标> / 推理链：<目标>        归结证明与后向链推理",
        "  为什么<陈述> / 溯因：<陈述>         证明失败前沿生成候选假设",
        "  场景：<问句> / 联合解释：<问句>     枚举多假设协同并注入克隆引擎验证",
        "  澄清 / 开始澄清                     按解释空间选最有判别力的问题；结束给条件化答案或崩塌诊断",
        "  如果…那么… / 假设：…                 反事实推演；无「那么」时进入跨轮假设世界",
        "  退出假设 / 记住这个假设 / 假设世界    丢弃假设 / 并入会话事实 / 查看当前假设",
        "  默认推理：<陈述> / 概率推理：…       非单调常识与加权可能世界",
        "  规则：<如果X的P是Y，那么X的Q是Y>    教一条 Horn 规则（占位符自动变量化）",
        "  模式路由 / 查询模式库                 统一模式库（配置查询+NLU 意图+教学模式）与体检",
        "  模式关系 / 模式关系图                 泛化/冗余/冲突落成的图边与图体检查询",
        "  模式落图 / 同步模式图                 把模式库、关系边与模式经验同步到命名图",
        "  模式匹配：<句子>                     查看命中模式、绑定与落选原因",
        "  软路由：<句子>                       查看近失解释的软对齐、置信度与接受/澄清/回退决策",
        "  软路由体检                           在配置构念变体上评估近失恢复、错误接受与澄清账目",
        "  模式经验 / 经验台账                  查看命中/未命中的可检索理解经验",
        "  模式联想：<句子>                     用相似成功经验泛化候选骨架（只确认，不直接回答）",
        "  确认联想：<句子>                     用户确认候选骨架，转正为可路由模式并落图",
        "  是 / 不是                            直接结算上一轮模式联想；否定记入负向反馈",
        "  忘经验：<短 id 或原句>               删除单条经验（撤销误确认/误否定）",
        "  清空模式经验                         清空经验台账并同步命名图",
        "  教模式：名 | 要素… | 路由 | 优先级 | 守卫  教一条模式（可选上下文守卫，落图持久化）",
        "  锻造模式：名 | 正例(多个用;分隔) | 反例(可空，多个用;分隔) | 路由 | 优先级 | 守卫",
        "  确认锻造 <id> / 否认锻造 <id> / 退役锻造 <id> / 遗忘锻造 <id>  转正、记反驳、停用、删除",
        "  锻造台账                            例子、正则、探针、命中/反驳与生命周期",
        "  锻造探针 [id]                       重跑正例/反例/逆生成探针，发现悄悄失效的规则",
        "  忘模式：<名>                         遗忘一条教学模式",
        "  记住：<事实> / 忘记：<事实>          会话事实/否定断言与撤回（否定只阻断路径，不写图）",
        "  改口：<事实>                        显式更正会话槽（记录 before/after，可撤销，不写图）",
        "  槽账本 / 改口史：<实体>的<属性>      查看当前赢家、压制值与逐次改口史",
        "  更正：<实体>的<属性>从<旧>改为<新>  删除旧值、写入新值，并记录可追踪的更正事件",
        "  修订预演：<实体>|<属性>|<旧>|<新>   提交前计算影响与冲突（只计算，不写图）",
        "  修订回执 / 撤销修订                查看最近写入回执 / 按逆操作恢复最新一笔",
        "  规则预演：<如果…那么…>              教规则前检查重复/不安全变量/功能冲突与影响",
        "  信念检查 / 信念一致性              功能谓词多值、定义域/值域与规则形状体检",
        "  历史可信度：<问题>                   沿支持图查看回答是可信/可疑/失效/未评估",
        "",
        "台账与报告",
        "  悬问 / 未答之问 / 悬问报告           欠用户答案的目标与缺失前提",
        "  回答悬问：<取值>                     回填上一轮追问的缺失前提",
        "  重开悬问 / 恢复悬问                  重新追问被「不知道」搁置的问题",
        "  放弃悬问 / 清空悬问                  显式放弃全部悬问（不写否定事实）",
        "  对话记忆 / 帧记忆 / 对话本体          对话状态、挂账帧与策略程序",
        "  追踪报告 / 指代轨迹 / 问据栈          分层追踪：问据栈、指代绑定与逐轮决策",
        "  追踪为什么 <编号>                     沿追踪图回溯某决策的依据链",
        "  身份台账 / 谓词报告 / 律典 / 类型体系  实体身份、谓词画像、语义定律与类型格",
        "  提及词表：<句子> / 词表                已确认词面对切分的覆盖、旧/新边界差异与来源",
        "  词表提名：<词面> / 词表拒绝：<词面>     只提名不落库 / 墓碑挡板阻止后续确认",
        "  绑定账本                              语境指称绑定：谁的说法指哪个实体（CRD）",
        "  接地证书                              答案探测式消歧的候选、答案指纹与敏感度证书",
        "  接地重放                              用当前知识版本复算最近一张证书的候选答案与动作",
        "  关联规则 / 推理发现 / 全息状态        联想规则、挖掘发现与全息记忆",
        "  补全回合：<实体>                     打开一次学习回合：收集候选、确认归纳、本批闭包",
        "  确认补全：<序号> / 否定补全：<序号>   裁定候选；确认事实只进会话，不写图谱",
        "  补全规则 / 补全报告                   已归纳规则与本回合候选（默认本批作用域）",
        "  预测展开：<实体>                     事实介绍 + 明确标注「未确认」的推测段落",
        "  推测：<实体> / 预测推测：<实体>       只输出推测、证据与拒绝审计，不混入事实段落",
        "  确认预测：<序号或编号> / 否定预测：…  裁决推测；确认只进会话事实层，否定阻断候选",
        "  预测台账 / 推测报告                   推测生命周期、拒绝原因与待定清单",
        "  补全提问 / 下一个问题                 按效用选一个候选追问（重复提问会降权）",
        "  补全测试 / 补全重放                   运行期裁决回归；重放会把违规规则隔离",
        "  别问了 / 允许提问                     长期静音或恢复主动追问（随会话持久化）",
        "  知识层次 / 层次清单 / 会话镜像        四层知识容器：基础/会话/推测/假设与摘要",
        "  晋升补全规则：<id>|<作用域>           命名空间留在回合内；只有「全局」进入全局归结",
        "  退役补全规则：<id> / 恢复补全规则：<id>  停用或恢复规则；已确认的会话事实保留",
        "  对齐报告 / 对齐候选：<术语>          本体对齐、候选格与答案等价性",
        "  对齐记忆 / 镜像对齐                  会话级可逆词形规则、类型守卫与回声词面",
        "  对齐纠正：<用户词面> = <规范身份>     从一次纠正归纳词形（落地 + 类型守卫后才生效）",
        "  对齐谓词：<说法> = <规范谓词>         记住谓词侧的镜像说法（会话级）",
        "  对齐否决：<词面> = <身份> / 对齐查询：<词面> / 对齐答法：<身份>  否证、前向复核、反向选面",
        "  覆盖报告 / 质量护照                   覆盖缺口与数据质量",
        "  关系获取：S 的 P / 获取台账         以问代抽：槽位追问、会话账本、覆盖层与诱导模式",
        "  获取缺口 / 获取模式 / 模式扫描：…    形状缺口转追问、诱导模式库与文本扫描",
        "  确认获取 / 采纳获取 / 模式扫描：…    确认、写入会话事实集、用诱导模式扫描新文本",
        "  断言阅读 / 待确认断言 / 提及卡       自由陈述的类型证据、确认状态与会话提及",
        "  确认断言 / 否认断言 / 清空断言       对待确认断言表态（只进会话，不写图谱）",
        "  形状报告 / 形状校验                  必备属性、基数、类型与定义域/值域校验",
        "  形状画像：<实体>                     形状完整画像：已知/缺失/冲突/未知 + 一次缺口追问",
        "  裁定：<实体> 的 <谓词> = <值>         会话内收敛冲突（不写图谱）",
        "  下一个形状问题 / 形状议程             按严重度继续追问，或查看当前形状缺口议程",
        "  形状对话报告 / 形状对话重置           会话补全进度与记忆状态",
        "  对比报告 / 待定推测 / 性能报告        神经符号双通道、校准与运行统计",
        "  外链台账 / 图账本                     外部知识留册与命名图物化视图",
        "  融合 / 融合应用 / 裁决：谓词=来源     多来源融合视图、物化与显式裁决",
        "  配置契约 / 配置体检                   config.toml 引用/能力/阈值静态检查",
        "  重载配置 / 回滚配置                   事务式热更新语言节；结构节留待重启",
        "",
        "证明与视图",
        "  证明图：<目标> / 知识子图：<实体>     证明树 / 邻域子图",
        "  证明日志 / 证明编号：<id>             回放带稳定 id 的证明凭证",
        "  一致性 / 改口检查                     同题异答、过期回答与未答问题",
        "  时间线 / 编年史 / 回放 <N>            对话编年史审计与重建",
        "  实体足迹：<实体> / 实体索引           跨会话实体脉络、别名折叠、同题异答与图谱现状",
        "  聊过：<词面> / 未解析提及              别名感知检索、未落地提及审计",
        "  相似实体：<实体> / 介绍：<实体>       相似度排序与生成式介绍",
        "  简短点 / 完整讲讲 / 换种表达        按同一份回答计划重述，不重新检索、不新增事实",
        "  第N点是什么 / 风格：第N点            只展开回答计划中的第 N 点",
        "  回答计划 / 计划台账                  查看计划 id、知识版本、逐点来源与过期状态",
        "  复核计划 / 计划复核                  用当前知识对照旧计划，只报告差异不改写回答",
        "  证据检索：<问题>                    带约束与上界剪枝的证据排序及检索回执",
        "",
        "其他",
        "  帮助 / help / ?                      显示本页",
        "  quit / 退出                          退出系统",
    ]
    .join("\n")
}

/// 解析「同义链接：别名 = 规范实体」的两端（支持半角/全角等号）。
fn parse_alias_link(body: &str) -> Option<(&str, &str)> {
    for separator in ['=', '＝'] {
        if let Some((alias, target)) = body.split_once(separator) {
            let alias = alias.trim();
            let target = target.trim();
            if !alias.is_empty() && !target.is_empty() {
                return Some((alias, target));
            }
        }
    }
    None
}

/// 本轮对话实际用到的边：最近一轮对话实例的焦点实体 + 所问属性。
/// 无属性时退化为「焦点实体触达的边」，让用户至少看到话题邻域。
fn latest_used_edges(
    dialogue_manager: &DialogueManager,
    edges: &[(String, String, String)],
) -> std::collections::HashSet<(String, String, String)> {
    let Some(turn) = dialogue_manager.turns().last() else {
        return std::collections::HashSet::new();
    };
    let entity = turn
        .entity
        .as_deref()
        .or_else(|| dialogue_manager.focus_entity());
    let attribute = turn.attribute.as_deref();
    edges
        .iter()
        .filter(|(subject, predicate, object)| {
            let touches = entity
                .map(|entity| subject == entity || object == entity)
                .unwrap_or(false);
            match attribute {
                Some(attribute) => predicate == attribute && touches,
                None => touches,
            }
        })
        .cloned()
        .collect()
}

async fn try_symbolic_reasoning(
    symbolic_reasoner: &mut SymbolicReasoner,
    contrast_arbiter: &mut ContrastArbiter,
    oxigraph_manager: &OxigraphManager,
    input: &str,
    query_entities: &[String],
) -> Option<String> {
    if !query_entities.is_empty() {
        match oxigraph_manager.get_relevant_facts(query_entities, 2).await {
            Ok(facts) => symbolic_reasoner.observe_facts(&facts),
            Err(error) => {
                symbolic_reasoner.note_engine_fault(format!("知识库读取相关事实失败：{}", error))
            }
        }
    }
    // 陈述句优先：形式化语义 + 语义蕴含判定（校验用户断言）；
    // 判为"未知"时追加溯因分析——后向链失败前沿生成候选假设（可解释的推测）；
    // 最后统一做神经符号对比（联想通道 vs 符号通道），输出仲裁结论并写入会话记忆
    if !contains_question_marker(input) {
        let report = symbolic_reasoner.entailment_report(input);
        if !report.starts_with("形式化失败") {
            let mut combined = report;
            if combined.contains("未知（证据不足）") {
                if let Some(abduction) = symbolic_reasoner.abduce_statement(input) {
                    combined.push('\n');
                    combined.push_str(&abduction);
                }
                // 默认逻辑（可反驳常识）补充：把「通常…除非…」知识用于陈述校验
                if let Some(supplement) = symbolic_reasoner.default_logic_supplement(input) {
                    combined.push('\n');
                    combined.push_str(&supplement);
                }
            }
            let case = contrast_arbiter.contrast_statement(input, symbolic_reasoner);
            combined.push('\n');
            combined.push_str(&contrast_arbiter.render_case_brief(&case));
            return Some(combined);
        }
    }
    if let Some(answer) = symbolic_reasoner.answer(input) {
        // 图谱无记录时，用溯因假设生成候选答案（LLM 式的"猜测 + 理由"），
        // 再追加默认逻辑（可反驳常识「通常…除非…」）补充证据
        if answer.contains("没有记录")
            || answer.contains("没有相关记录")
            || answer.contains("数量为 0")
        {
            let mut combined = answer;
            if let Some(abduction) = symbolic_reasoner.hypothesis_answer(input) {
                combined.push('\n');
                combined.push_str(&abduction);
            }
            if let Some(supplement) = symbolic_reasoner.default_logic_supplement(input) {
                combined.push('\n');
                combined.push_str(&supplement);
            }
            // 失败即立悬问：记住「目标 + 缺哪条前提」，下一轮补上后延期回答。
            if let Some(prompt) = symbolic_reasoner.open_pending_question(input) {
                combined.push('\n');
                combined.push_str(&prompt);
            }
            return Some(combined);
        }
        symbolic_reasoner.dismiss_pending_question(input);
        // 轮末预取：焦点实体已确定，下一轮若继续追问可直接复用链接预测候选。
        symbolic_reasoner.prefetch_answer_drafts();
        return Some(answer);
    }
    // 开放问句无直接答案：尝试溯因生成候选假设
    if let Some(abduction) = symbolic_reasoner.hypothesis_answer(input) {
        return Some(abduction);
    }
    // 仍无结果时：默认逻辑（非单调常识）补充证据，避免结果抽取断层
    if let Some(supplement) = symbolic_reasoner.default_logic_supplement(input) {
        return Some(supplement);
    }
    // 逻辑问答失败时，先记住缺哪条前提（悬问），再尝试连贯叙述。
    if let Some(prompt) = symbolic_reasoner.open_pending_question(input) {
        return Some(prompt);
    }
    // 逻辑问答失败时，优先用 OWL+DCG 话语引擎做连贯叙述，
    // 再回退到"知识图谱即语言模型"的统计生成式介绍
    let entity = symbolic_reasoner.best_linked_entity(input)?;
    symbolic_reasoner
        .narrate(&entity)
        .or_else(|| symbolic_reasoner.generate_about(&entity))
}

/// 同一轮内区分「查无结果」与「引擎故障」：故障显式回执并写入会话故障状态。
fn run_prolog_query(
    prolog_engine: &mut PrologEngine,
    symbolic_reasoner: &mut SymbolicReasoner,
    prolog_query: &str,
    query_type: &str,
    response_formatter: &ResponseFormatter,
) -> String {
    match prolog_engine.query_checked(prolog_query) {
        Ok(results) => response_formatter.format_prolog_results(&results, query_type),
        Err(error) => {
            symbolic_reasoner.note_engine_fault(error.clone());
            format!("推理引擎故障（不是查无记录）：{}", error)
        }
    }
}

fn contains_question_marker(text: &str) -> bool {
    // 疑问词识别走虚词算符表（含同形守卫）
    know_prolog::function_words::contains_question_marker(text)
}

/// 回答是否表示「知识库没有依据」：只有这种失败才立悬问。
fn is_no_record_text(text: &str) -> bool {
    text.contains("没有记录")
        || text.contains("没有相关记录")
        || text.contains("没有可用依据")
        || text.contains("数量为 0")
}

fn probe_reference_answer(
    reasoner: &SymbolicReasoner,
    _entity: &str,
    rewritten: &str,
) -> ProbeStatus {
    match reasoner.answer_certain(rewritten) {
        Some(text) if !text.trim().is_empty() && !is_no_record_text(&text) => {
            ProbeStatus::Answered(vec![text])
        }
        _ => ProbeStatus::NoAnswer,
    }
}

/// 判断输入是否以系统命令开头：澄清进行中时遇到命令应让位给常规处理
/// 插件声明式兜底：返回（可直接打印的文本，贡献路径）。
fn plugin_policy_text(
    runtime: &mut know_prolog::plugin::PluginRuntime,
    input: &str,
) -> Option<(String, String)> {
    let (text, plan) = runtime.policy_reply(input)?;
    Some((runtime.style_wrap(&text), plan))
}

fn is_command_like(input: &str) -> bool {
    // 回答计划风格指令：不参与悬问/澄清的短答回填。
    if StyleDirective::parse(input).is_some() {
        return true;
    }
    // 批次线索（这批 / 上一批 / 第 N 批 / 来源文件）走导入情景，不参与短答回填。
    if know_prolog::import_episode::looks_like_episode_reference(input) {
        return true;
    }
    const COMMANDS: &[&str] = &[
        "SPARQL",
        "sparql",
        "Sparql",
        "查询SPARQL",
        "查询sparql",
        "导出rdf文件",
        "导出rdf",
        "导出文件",
        "export rdf",
        "撤销批次",
        "retract batch",
        "全息",
        "对话本体",
        "对话状态",
        "追踪报告",
        "追踪状态",
        "追踪图",
        "trace",
        "指代轨迹",
        "指代报告",
        "问据栈",
        "问据报告",
        "QUD",
        "追踪为什么",
        "查询板",
        "查询状态",
        "工作记忆",
        "查询剧本",
        "运行剧本",
        "溯因",
        "假设生成",
        "为什么",
        "证明",
        "归结",
        "推理链",
        "寻找证明",
        "场景",
        "联合解释",
        "解释场景",
        "采纳假设",
        "确认假设",
        "待定假设",
        "假设列表",
        "候选假设",
        "介绍",
        "生成介绍",
        "回答计划",
        "计划台账",
        "生成计划",
        "回答方案",
        "复核计划",
        "计划复核",
        "核对计划",
        "检查计划",
        "叙述",
        "讲述",
        "详细叙述",
        "复述",
        "教本体",
        "本体教学",
        "续写",
        "接龙",
        "补全",
        "补全回合",
        "确认补全",
        "否定补全",
        "驳回补全",
        "补全规则",
        "补全报告",
        "补全测试",
        "补全回归",
        "知识回归",
        "补全测试报告",
        "补全重放",
        "重放补全测试",
        "补全测试重放",
        "下一个问题",
        "该问什么",
        "主动提问",
        "别问了",
        "不要追问",
        "静音补全提问",
        "停止提问",
        "允许提问",
        "恢复提问",
        "解除静音",
        "晋升补全规则",
        "退役补全规则",
        "补全规则退役",
        "恢复补全规则",
        "补全规则恢复",
        "知识层次",
        "层次清单",
        "容器清单",
        "会话镜像",
        "回答层次",
        "关联规则",
        "关联联想",
        "联想",
        "推理发现",
        "推导发现",
        "对齐报告",
        "本体对齐",
        "对齐候选",
        "对齐记忆",
        "镜像对齐",
        "对齐台账",
        "镜像记忆",
        "词形对齐",
        "对齐纠正",
        "对齐谓词",
        "谓词对齐",
        "对齐否决",
        "否证对齐",
        "对齐查询",
        "镜像查询",
        "对齐答法",
        "镜像答法",
        "推理规则",
        "链接预测规则",
        "实体链接",
        "提及账本",
        "提及格",
        "格解码",
        "提及词表",
        "词表感知",
        "词表切分",
        "词表报告",
        "词表台账",
        "词表提名",
        "词表拒绝",
        "词表",
        "准入",
        "收录",
        "身份台账",
        "身份注册表",
        "绑定账本",
        "指称账本",
        "语境账本",
        "接地证书",
        "指称证书",
        "接地台账",
        "指称接地",
        "接地重放",
        "指称重放",
        "重放证书",
        "证书重放",
        "身份报告",
        "确认身份",
        "记住身份",
        "拒绝身份",
        "忘记别名",
        "合并身份",
        "铸造检查",
        "识别实体",
        "判断",
        "蕴含",
        "校验",
        "形式化",
        "语义分析",
        "语义森林",
        "相似实体",
        "语义相似",
        "证据检索",
        "检索证据",
        "对话记忆",
        "帧记忆",
        "话语帧",
        "挂账",
        "如果",
        "假如",
        "反事实",
        "澄清",
        "结构联想",
        "图游走",
        "话题延伸",
        "软推理",
        "加权推理",
        "因果",
        "干预",
        "反事实干预",
        "do：",
        "do:",
        "梯度归因",
        "可微推理",
        "因果分词",
        "语境分词",
        "读法分词",
        "读法确认",
        "读法纠正",
        "读法记忆",
        "语境记忆",
        "清空读法记忆",
        "清空语境记忆",
        "忘记读法记忆",
        "分词：",
        "分词:",
        "概率推理",
        "可能世界",
        "默认推理",
        "默认逻辑",
        "教学：",
        "教学:",
        "教我",
        "概率知识",
        "ProbLog",
        "probLog",
        "规则",
        "教规则",
        "学习规则",
        "记住",
        "记录",
        "断言",
        "更正",
        "纠正",
        "修正",
        "改口",
        "更正记忆",
        "纠正记忆",
        "槽账本",
        "槽位账本",
        "改口账本",
        "改口史",
        "槽历史",
        "忘记",
        "遗忘",
        "撤回",
        "撤销导入",
        "导入回执",
        "导入历史",
        "批次台账",
        "导入批次",
        "批次列表",
        "批次报告",
        "导入日志",
        "撤销历史",
        "外链",
        "同义链接",
        "配置契约",
        "配置检查",
        "配置体检",
        "config-lint",
        "重载配置",
        "配置重载",
        "重新加载配置",
        "reload-config",
        "config-reload",
        "回滚配置",
        "配置回滚",
        "config-rollback",
        "rollback-config",
        "性能报告",
        "性能台账",
        "运行统计",
        "派生台账",
        "版本台账",
        "增量报告",
        "图账本",
        "派生图账本",
        "物化推理",
        "推理物化",
        "物化派生",
        "会话规则",
        "信念检查",
        "信念一致性",
        "修订回执",
        "会话修订",
        "撤销修订",
        "对比",
        "校准",
        "仲裁",
        "编年史",
        "对话编年史",
        "对话档案",
        "档案校验",
        "校验编年史",
        "回放",
        "聊过",
        "查档案",
        "实体足迹",
        "实体索引",
        "实体线程",
        "实体线索",
        "实体审计",
        "未解析提及",
        "未解析实体",
        "实体歧义审计",
        "时间线",
        "对话时间线",
        "证明图",
        "证明树",
        "画证明",
        "证明日志",
        "证明档案",
        "证明记录",
        "证明编号",
        "悬问",
        "未答之问",
        "放弃悬问",
        "清空悬问",
        "重开悬问",
        "恢复悬问",
        "重开未答之问",
        "一致性",
        "回答一致性",
        "改口检查",
        "改口",
        "日志视图",
        "情景",
        "对话情景",
        "情景理论",
        "情景历史",
        "情景归因",
        "情景反事实",
        "情景查询",
        "历史可信度",
        "回答可信度",
        "信任报告",
        "可信度",
        "知识子图",
        "子图",
        "邻域",
        "覆盖报告",
        "关系获取",
        "获取关系",
        "获取缺口",
        "关系缺口",
        "追问缺口",
        "缺口追问",
        "问答获取",
        "获取台账",
        "获取报告",
        "关系获取报告",
        "问答获取报告",
        "获取状态",
        "获取模式",
        "关系获取模式",
        "诱导模式",
        "模式库",
        "获取覆盖",
        "会话覆盖层",
        "覆盖层",
        "断言阅读",
        "断言阅读报告",
        "待确认断言",
        "提及卡",
        "断言台账",
        "清空断言",
        "取消断言",
        "放弃断言",
        "确认断言",
        "确认当前断言",
        "否认断言",
        "否认当前断言",
        "确认获取",
        "确认关系获取",
        "否认获取",
        "否认关系获取",
        "拒绝获取",
        "撤回获取",
        "撤销获取",
        "采纳获取",
        "采纳关系获取",
        "提交获取",
        "模式扫描",
        "退役模式",
        "形状报告",
        "形状校验",
        "SHACL报告",
        "shacl报告",
        "shapes",
        "形状画像",
        "对话画像",
        "形状对话",
        "形状议程",
        "形状缺口议程",
        "下一个形状问题",
        "下一个缺口问题",
        "裁定",
        "冲突裁定",
        "会话裁定",
        "质量报告",
        "质量护照",
        "质量总览",
        "护照总览",
        "类型体系",
        "类型报告",
        "类型格",
        "覆盖检查",
        "虚词表",
        "功能词表",
        "虚词报告",
        "算符表",
        // 谓词画像 / 身份裁决
        "谓词报告",
        "谓词画像",
        "谓词注册表",
        "谓词列表",
        "律典",
        "定律台账",
        "语义定律",
        "law-report",
        "登记别名",
        "登记逆谓词",
        "功能谓词",
        "取消功能谓词",
        "确认谓词",
        "拒绝谓词",
        // 对比记忆的追问与待定推测
        "待定推测",
        // 会话规则库别名
        "后向链规则",
        // 澄清子命令（澄清进行中不应被当作回答）
        "开始澄清",
        "停止澄清",
        "结束澄清",
        "退出澄清",
        "提问确认",
        // 对话续说 / 换角度
        "接着说",
        "继续说",
        "继续介绍",
        "继续讲",
        "接着讲",
        "换个角度说",
        "换个说法",
        "换个角度",
        "换一个角度",
        "重新介绍",
        "讲讲",
        "说说",
        // 撤销上次导入
        "撤销上次导入",
        // 声明式插件
        "插件",
        "启用插件",
        "停用插件",
        "禁用插件",
        "为什么这么答",
        // 统一模式库（Pattern IR）
        "模式路由",
        "查询模式库",
        "统一模式库",
        "模式关系",
        "查询模式关系",
        "模式关系图",
        "模式图体检",
        "模式落图",
        "同步模式图",
        "保存模式库",
        "模式经验",
        "模式经验报告",
        "经验台账",
        "模式联想：",
        "模式联想:",
        "模式联想 ",
        "案例联想：",
        "案例联想:",
        "确认联想：",
        "确认联想:",
        "确认联想 ",
        "采纳联想：",
        "采纳联想:",
        "采纳联想 ",
        "案例转正：",
        "案例转正:",
        "案例转正 ",
        "忘经验：",
        "忘经验:",
        "忘经验 ",
        "遗忘经验：",
        "遗忘经验:",
        "清空模式经验",
        "遗忘模式经验",
        "模式匹配：",
        "模式匹配:",
        "模式匹配 ",
        "匹配模式：",
        "匹配模式:",
        "软路由：",
        "软路由:",
        "软路由 ",
        "软匹配：",
        "软匹配:",
        "软路由体检",
        "软路由评估",
        "软匹配体检",
        "教模式：",
        "教模式:",
        "教模式 ",
        "教学模式：",
        "教学模式:",
        "锻造模式：",
        "锻造模式:",
        "锻造模式 ",
        "锻造模式",
        "模式锻造：",
        "模式锻造:",
        "模式锻造",
        "确认锻造：",
        "确认锻造:",
        "确认锻造 ",
        "采纳锻造：",
        "采纳锻造:",
        "否认锻造：",
        "否认锻造:",
        "否认锻造 ",
        "否决锻造：",
        "否决锻造:",
        "退役锻造：",
        "退役锻造:",
        "退役锻造 ",
        "停用锻造：",
        "停用锻造:",
        "遗忘锻造：",
        "遗忘锻造:",
        "遗忘锻造 ",
        "删除锻造：",
        "删除锻造:",
        "锻造台账",
        "锻造报告",
        "锻造状态",
        "模式锻造台账",
        "锻造探针",
        "复核锻造",
        "锻造探针报告",
        "锻造探针：",
        "锻造探针:",
        "锻造探针 ",
        "复核锻造：",
        "复核锻造:",
        "忘模式：",
        "忘模式:",
        "忘模式 ",
        "遗忘模式：",
        "遗忘模式:",
        // 帮助与退出
        "帮助",
        "help",
        "?",
        "退出",
        "exit",
    ];
    COMMANDS.iter().any(|command| input.starts_with(command))
}

/// 语义歧义澄清的一步处理：已消化则返回 true；非回答输入放行常规管线。
fn handle_formal_clarification(reasoner: &mut SymbolicReasoner, input: &str) -> bool {
    if !reasoner.formal_clarification_active() || is_command_like(input) {
        return false;
    }
    match reasoner.answer_formal_clarification(input) {
        FormalClarificationOutcome::Answered(text)
        | FormalClarificationOutcome::Cancelled(text) => println!("{}", text),
        FormalClarificationOutcome::Reprompt(prompt) => println!("{}", prompt),
        FormalClarificationOutcome::NotAnAnswer => return false,
    }
    println!();
    true
}

/// 帧澄清(input) 的一步处理结果：绑定后重放 / 本轮已消化 / 放行常规管线
enum FrameClarificationStep {
    Replay(String),
    Handled,
    Pass,
}

/// 帧澄清进行中的短答处理：序号/候选名/已知实体名 → 确定性绑定；
/// 越界 → 重问（零状态变化）；取消词 → 释放；命令或非回答 → 放行。
fn handle_frame_clarification(
    reasoner: &mut SymbolicReasoner,
    input: &str,
) -> FrameClarificationStep {
    if !reasoner.frame_clarification_active() || is_command_like(input) {
        return FrameClarificationStep::Pass;
    }
    match reasoner.answer_frame_clarification(input) {
        Some(ClarificationOutcome::Bound { question, note }) => {
            println!("（{}）", note);
            FrameClarificationStep::Replay(question)
        }
        Some(ClarificationOutcome::Cancelled { note }) => {
            println!("（{}）", note);
            println!();
            FrameClarificationStep::Handled
        }
        Some(ClarificationOutcome::Reprompt { prompt }) => {
            println!("{}", prompt);
            println!();
            FrameClarificationStep::Handled
        }
        None => FrameClarificationStep::Pass,
    }
}

/// 解析干预命令的后缀 `=真/假`（默认 do(…=假)：移除该事实再观察下游变化）
fn parse_intervention_value(target: &str) -> (&str, bool) {
    for (suffix, value) in [
        ("=真", true),
        ("=对", true),
        ("=true", true),
        ("=1", true),
        ("=假", false),
        ("=错", false),
        ("=false", false),
        ("=0", false),
    ] {
        if let Some(stripped) = target.strip_suffix(suffix) {
            return (stripped.trim(), value);
        }
    }
    (target, false)
}

fn strip_any_prefix<'a>(input: &'a str, prefixes: &[&str]) -> Option<&'a str> {
    prefixes
        .iter()
        .find_map(|prefix| input.strip_prefix(prefix))
}

/// 当前对话状态快照：供模式守卫、经验检索与案例联想共用同一口径。
fn dialogue_pattern_context(
    dialogue_manager: &DialogueManager,
    symbolic_reasoner: &SymbolicReasoner,
) -> pattern_ir::PatternContext {
    let focus = dialogue_manager.focus_entity().map(str::to_string);
    pattern_ir::PatternContext {
        focus_type: focus
            .as_deref()
            .and_then(|entity| symbolic_reasoner.context().focus_type(entity))
            .map(str::to_string),
        focus,
        last_move: dialogue_manager
            .turns()
            .last()
            .map(|turn| turn.kind.label().to_string()),
        pending_clarify: dialogue_manager.pending_reference().is_some(),
        turn: dialogue_manager.turn_count(),
    }
}

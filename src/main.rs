// src/main.rs
use std::io::{BufRead, Write};
use std::sync::{Arc, OnceLock};
use clap::{Parser, Subcommand};

// 从库（know_prolog）中导入所需的功能
use know_prolog::{
    load_config, 
    KnowledgeExtractor, 
    OxigraphManager,
    ResponseFormatter,
    InputMode,
    KnowledgeItem,
    create_prolog_rules,
    detect_input_mode,
    create_default_synonyms,
    extract_query_entities_from_tagged,
    nlp::{preprocess, normalize_words, analyze_query_intent},
    nlu::NLUEngine,
    prolog_engine::PrologEngine,
    local_semantic_adapter::LocalSemanticAdapter,
    external_knowledge::{ExternalKnowledgeService, get_direct_answer},
    symbolic_reasoner::SymbolicReasoner,
    neuro_symbolic_contrast::ContrastArbiter,
};

#[derive(Parser)]
#[command(name = "knowprolog")]
#[command(about = "KnowProlog - Chinese knowledge graph system")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

// 实体属性查询模式的全局正则
static ENTITY_ATTRIBUTE_RE: OnceLock<regex::Regex> = OnceLock::new();

#[derive(Subcommand, Debug)]
enum Commands {
    /// Import JSON-LD knowledge base from file
    ImportJsonld {
        /// Path to the JSON-LD file
        file_path: String,
    },
    /// Import JSON-LD knowledge base from content
    ImportJsonldContent {
        /// JSON-LD content as string
        content: String,
    },
    /// Run interactive mode (default)
    Interactive,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    
    let config = match load_config().await {
        Ok(cfg) => Arc::new(cfg),
        Err(e) => {
            eprintln!("加载配置文件失败: {}", e);
            return;
        }
    };

    // 创建Oxigraph存储
    let oxigraph_manager = match OxigraphManager::new(&config.system.oxigraph_store_path) {
        Ok(manager) => manager,
        Err(e) => {
            eprintln!("创建Oxigraph存储失败: {}", e);
            return;
        }
    };

    // 处理命令行参数
    match cli.command {
        Some(Commands::ImportJsonld { file_path }) => {
            println!("正在导入 JSON-LD 文件: {}", file_path);
            
            match oxigraph_manager.import_jsonld_from_file(&file_path).await {
                Ok(_) => {
                    println!("JSON-LD 文件导入成功！");
                }
                Err(e) => {
                    eprintln!("JSON-LD 文件导入失败: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Some(Commands::ImportJsonldContent { content }) => {
            println!("正在导入 JSON-LD 内容...");
            
            match oxigraph_manager.import_jsonld(&content).await {
                Ok(_) => {
                    println!("JSON-LD 内容导入成功！");
                }
                Err(e) => {
                    eprintln!("JSON-LD 内容导入失败: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Some(Commands::Interactive) | None => {
            // 运行交互模式
            run_interactive_mode(config, oxigraph_manager).await;
        }
    }
}

async fn run_interactive_mode(config: Arc<know_prolog::SystemConfig>, oxigraph_manager: OxigraphManager) {
    println!("{}", config.messages.startup);

    let knowledge_extractor = KnowledgeExtractor::new(config.clone());
    let response_formatter = ResponseFormatter::new(config.clone());
    let nlu_engine = NLUEngine::new(&config);
    let prolog_rules = create_prolog_rules(&config);
    let mut prolog_engine = PrologEngine::new();
    let local_semantic_adapter = {
        let nlp_config = &config.nlp;
        LocalSemanticAdapter::new(
            nlp_config.enable_local_semantic.unwrap_or(true) && 
            nlp_config.disable_ownthink_api.unwrap_or(false)
        )
    };
    
    // 创建外部知识服务
    let external_knowledge_service = ExternalKnowledgeService::new();
    
    let synonyms = if let Some(synonyms_config) = &config.synonyms {
        synonyms_config.mappings.clone()
    } else {
        create_default_synonyms()
    };

    // 神经符号对比仲裁器：联想通道（软逻辑+关联+图游走）与符号通道（蕴含+归结）
    // 交叉验证，输出带对冲词的结论，并保留会话级校准记忆
    let mut contrast_arbiter = ContrastArbiter::from_reasoning_config(&config.reasoning);
    // 对比追问进行中：记录当前提问对应的待定推测编号，用户答「是/不是」即结算
    let mut contrast_followup: Option<usize> = None;

    // 符号推理器：对话上下文 + 本体对齐 + 链接预测 + 符号嵌入
    let mut symbolic_reasoner = SymbolicReasoner::new(config.reasoning.clone(), synonyms.clone());
    // 批量导入期间把整图刷新推迟到真正需要时（见循环顶部），避免逐行 O(n²) 重扫重观察
    let mut reasoner_stale = false;
    if let Ok(facts) = load_graph_snapshot(&oxigraph_manager).await {
        symbolic_reasoner.observe_facts(&facts);
    }
    if let Some(state_path) = config.reasoning.dialogue_state_path.as_deref() {
        let (restored, contrast_snapshot) =
            symbolic_reasoner.load_dialogue_state_with_contrast(state_path);
        if restored {
            println!("已恢复上次对话记忆：{}", symbolic_reasoner.dialogue_summary());
        }
        if let Some(snapshot) = contrast_snapshot {
            if contrast_arbiter.restore_snapshot(&snapshot) {
                println!("已恢复上次对比校准记忆：{}", contrast_arbiter.memory_summary());
            }
        }
    }

    println!("{}", config.messages.ready);
    println!("{}", config.messages.query_prompt);
    println!("{}", config.messages.import_prompt);
    println!("{}", config.messages.quit_prompt);
    println!();

    let stdin_reader = std::io::stdin();
    loop {
        print!("> ");
        std::io::stdout().flush().unwrap();

        let mut input = String::new();
        match stdin_reader.lock().read_line(&mut input) {
            Ok(0) => break, // EOF reached, exit the loop
            Ok(_) => {},
            Err(e) => {
                eprintln!("Error reading input: {}", e);
                break;
            }
        }
        let input = input.trim();

        if input == "quit" {
            break;
        }

        // 导入行不立即刷新：连续导入在真正进入查询/命令前只做一次整图观察
        if reasoner_stale
            && !matches!(
                detect_input_mode(input, &config.nlp.import_keywords),
                InputMode::KnowledgeImport
            )
        {
            refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
            reasoner_stale = false;
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

        if input == "对话记忆" {
            println!("{}", symbolic_reasoner.dialogue_summary());
            println!();
            continue;
        }

        // 神经符号对比：对任意陈述做联想通道与符号通道的交叉验证与仲裁
        if let Some(target) = strip_any_prefix(
            input,
            &[
                "对比：", "对比:", "神经符号对比：", "神经符号对比:", "双通道对比：",
                "双通道对比:", "仲裁：", "仲裁:",
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
            "对比报告" | "对比统计" | "校准报告" | "对比校准" | "仲裁报告" | "仲裁统计"
                | "对比记忆" | "校准记忆"
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
            &["对比确认", "对比确认：", "对比确认:", "对比采纳", "对比采纳：", "对比采纳:"],
        ) {
            match parse_hypothesis_index(target) {
                Some(index) => {
                    println!("{}", contrast_arbiter
                        .confirm_guess(index, &mut symbolic_reasoner)
                        .unwrap_or_else(|| format!(
                            "没有待定的联想推测 #{}；可用「待定推测」查看列表。",
                            index
                        )));
                }
                None => println!(
                    "请给出推测编号，例如：对比确认 1。当前待定推测：\n{}",
                    contrast_arbiter.pending_report()
                ),
            }
            println!();
            continue;
        }

        // 否认联想推测：加入屏蔽集，联想通道之后不再提出
        if let Some(target) = strip_any_prefix(
            input,
            &["对比否认", "对比否认：", "对比否认:", "对比否决", "对比否决：", "对比否决:"],
        ) {
            match parse_hypothesis_index(target) {
                Some(index) => {
                    println!("{}", contrast_arbiter.refute_guess(index)
                        .unwrap_or_else(|| format!(
                            "没有待定的联想推测 #{}；可用「待定推测」查看列表。",
                            index
                        )));
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
            let affirmative = ["是", "是的", "对", "对的", "没错", "属实", "确认", "yes", "y"]
                .contains(&answer);
            let negative =
                ["不是", "不对", "否", "不属实", "错", "no", "n"].contains(&answer);
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

        // 对话式澄清进行中：优先把输入当作对当前提问的回答（是 / 不是 / 不知道）
        if symbolic_reasoner.clarification_active()
            && !matches!(input, "结束澄清" | "停止澄清" | "退出澄清" | "quit")
            && !is_command_like(input)
        {
            match symbolic_reasoner.answer_clarification(input) {
                Some(text) => println!("{}", text),
                None => println!("（澄清会话已结束）"),
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
                "场景：", "场景:", "联合解释：", "联合解释:", "解释场景：", "解释场景:",
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

        // 形式化语义与语义蕴含：把陈述句解析为逻辑形式并判定蕴含/矛盾/未知
        if let Some(target) = strip_any_prefix(
            input,
            &[
                "判断：", "判断:", "蕴含：", "蕴含:", "校验：", "校验:", "形式化：", "形式化:",
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

        // 多轮生成：接着上一段介绍继续讲
        if matches!(input, "接着说" | "继续说" | "继续介绍" | "继续讲" | "接着讲") {
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

        if let Some(entity) = strip_any_prefix(input, &["实体链接：", "实体链接:", "识别实体：", "识别实体:"]) {
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
            &["介绍：", "介绍:", "生成介绍：", "生成介绍:", "介绍一下", "讲讲", "说说"],
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
            &["叙述：", "叙述:", "讲述：", "讲述:", "详细叙述：", "详细叙述:"],
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
        if let Some(spec) = strip_any_prefix(input, &["复述：", "复述:", "换个句式：", "换个句式:"]) {
            let spec = spec.trim();
            match spec.rsplit_once('的') {
                Some((entity, predicate)) if !entity.trim().is_empty() && !predicate.trim().is_empty() => {
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
        if let Some(axiom) = strip_any_prefix(
            input,
            &["教本体：", "教本体:", "本体教学：", "本体教学:"],
        ) {
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
            &["证明：", "证明:", "归结：", "归结:", "证明目标：", "证明目标:"],
        ) {
            let target = target.trim();
            if target.is_empty() {
                println!("请提供要证明的陈述，例如：证明：张三的师祖是王五");
            } else {
                refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                println!("{}", symbolic_reasoner.resolution_proof_report(target));
            }
            println!();
            continue;
        }

        // 推理链：对开放目标求全部归结解，逐解展示证明链
        if let Some(target) = strip_any_prefix(
            input,
            &["推理链：", "推理链:", "寻找证明：", "寻找证明:"],
        ) {
            let target = target.trim();
            if target.is_empty() {
                println!("请提供推理目标，例如：推理链：张三的师傅的师傅是谁");
            } else {
                refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                println!("{}", symbolic_reasoner.resolution_query_report(target));
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
        if let Some(entity) = strip_any_prefix(
            input,
            &["联想：", "联想:", "关联联想：", "关联联想:"],
        ) {
            let entity = entity.trim();
            if entity.is_empty() {
                println!("请提供要联想的实体，例如：联想：姚明");
            } else {
                refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                match symbolic_reasoner.association_suggestions(entity) {
                    Some(text) => println!("{}", text),
                    None => println!(
                        "没有可用的关联联想（实体未见过，或没有满足阈值的关联规则）。"
                    ),
                }
            }
            println!();
            continue;
        }

        // 结构联想：RWR 图嵌入（随机游走重启）检索结构相关实体与解释路径
        if let Some(entity) = strip_any_prefix(
            input,
            &["结构联想：", "结构联想:", "图游走：", "图游走:"],
        ) {
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
                "因果反事实：", "因果反事实:", "反事实干预：", "反事实干预:", "干预：",
                "干预:", "do：", "do:",
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
                "因果链：", "因果链:", "因果关系：", "因果关系:", "因果：", "因果:",
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
        if let Some(target) =
            strip_any_prefix(input, &["梯度归因：", "梯度归因:", "可微推理：", "可微推理:"])
        {
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
                "概率推理：", "概率推理:", "可能世界：", "可能世界:", "ProbLog：", "ProbLog:",
                "probLog：", "probLog:",
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

        // 中文概率知识教学：自然语言句子 → ProbLog 概率知识（会话级，不写知识图谱）
        if let Some(target) = strip_any_prefix(
            input,
            &["教学：", "教学:", "教我：", "教我:", "概率知识：", "概率知识:"],
        ) {
            let target = target.trim();
            if target.is_empty() {
                println!(
                    "请给出要教的中文知识，例如：教学：如果张三的师傅是李四，那么张三的师祖是王五"
                );
            } else {
                refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                println!("{}", symbolic_reasoner.teach_report(target));
            }
            println!();
            continue;
        }

        // 中文规则学习：教一条 Horn 规则（含占位符自动变量化），直接进入后向链
        if let Some(target) = strip_any_prefix(
            input,
            &[
                "规则：", "规则:", "教规则：", "教规则:", "学习规则：", "学习规则:",
            ],
        ) {
            let target = target.trim();
            if target.is_empty() {
                println!("请给出中文规则，例如：规则：如果X的师傅是Y，那么X的老师是Y");
            } else {
                refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
                println!("{}", symbolic_reasoner.learn_rule_report(target));
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
                println!("{}", symbolic_reasoner.assert_nl_fact(target));
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
                println!("{}", symbolic_reasoner.retract_nl_fact(target));
            }
            println!();
            continue;
        }

        // 会话规则库：列出学过的 Horn 规则与会话事实
        if matches!(input, "会话规则" | "规则库" | "后向链规则" | "规则列表") {
            println!("{}", symbolic_reasoner.session_rules_report());
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

        // 溯因·假设生成：后向链失败前沿 → 候选假设（可证明则直接给证明）
        if let Some(target) = strip_any_prefix(
            input,
            &["溯因：", "溯因:", "假设生成：", "假设生成:", "为什么：", "为什么:"],
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
            || strip_any_prefix(input, &["反事实：", "反事实:", "假设：", "假设:", "假如：", "假如:"])
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
            &["采纳假设", "采纳假设：", "采纳假设:", "确认假设", "确认假设：", "确认假设:"],
        ) {
            match parse_hypothesis_index(target) {
                Some(index) => println!("{}", symbolic_reasoner.adopt_hypothesis(index)
                    .unwrap_or_else(|| format!("没有第 {} 条待定假设；可用「待定假设」查看列表。", index))),
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

        let mode = detect_input_mode(input, &config.nlp.import_keywords);

        match mode {
            InputMode::KnowledgeImport => {
                // 检查是否是 JSON-LD 文件导入
                if input.starts_with("导入jsonld文件：") || input.starts_with("import jsonld:") {
                    let file_path = input.replace("导入jsonld文件：", "")
                                        .replace("import jsonld:", "")
                                        .trim()
                                        .to_string();
                    
                    if file_path.is_empty() {
                        println!("请提供 JSON-LD 文件路径");
                        continue;
                    }
                    
                    println!("正在导入 JSON-LD 文件: {}", file_path);
                    
                    match oxigraph_manager.import_jsonld_from_file(&file_path).await {
                        Ok(_) => {
                            println!("JSON-LD 文件导入成功！");
                            reasoner_stale = true;
                        }
                        Err(e) => {
                            eprintln!("JSON-LD 文件导入失败: {}", e);
                        }
                    }
                } else if input.starts_with("导入jsonld内容：") || input.starts_with("import jsonld content:") {
                    let jsonld_content = input.replace("导入jsonld内容：", "")
                                             .replace("import jsonld content:", "")
                                             .trim()
                                             .to_string();
                    
                    if jsonld_content.is_empty() {
                        println!("请提供 JSON-LD 内容");
                        continue;
                    }
                    
                    println!("正在导入 JSON-LD 内容...");
                    
                    match oxigraph_manager.import_jsonld(&jsonld_content).await {
                        Ok(_) => {
                            println!("JSON-LD 内容导入成功！");
                            reasoner_stale = true;
                        }
                        Err(e) => {
                            eprintln!("JSON-LD 内容导入失败: {}", e);
                        }
                    }
                } else {
                    // 知识导入逻辑
                    println!("{}", config.messages.importing);
                    
                    let clean_text = config.nlp.import_keywords.iter()
                        .fold(input.to_string(), |text, keyword| {
                            // 未出现该关键词时不构造替换串、不做整串扫描
                            if !text.contains(keyword.as_str()) {
                                return text;
                            }
                            text.replace(&format!("{}：", keyword), "")
                                .replace(&format!("{}:", keyword), "")
                        })
                        .trim()
                        .to_string();
                    
                    let tagged_words = preprocess(&clean_text, &config.nlp);
                    let normalized_words = normalize_words(tagged_words, &synonyms);
                    let items = knowledge_extractor.extract(&clean_text, &normalized_words);
                    
                    if items.is_empty() {
                        println!("{}", config.messages.no_knowledge_extracted);
                    } else {
                        for item in &items {
                            match item {
                                KnowledgeItem::Entity { label, properties } => {
                                    println!("实体 - 类型: {}, 属性: {:?}", label, properties);
                                }
                                KnowledgeItem::Relationship { from_label, from_id, to_label, to_id, rel_type, .. } => {
                                    println!("关系 - {}({}) -[{}]-> {}({})",
                                        from_label, from_id, rel_type, to_label, to_id);
                                }
                            }
                        }
                        
                        match oxigraph_manager.save_knowledge(items).await {
                            Ok(_) => {
                                println!("知识已成功保存");
                                reasoner_stale = true;
                            }
                            Err(e) => eprintln!("保存失败: {}", e),
                        }
                    }
                }
            }
            InputMode::Query => {
                // 多轮对话上下文：指代消解与省略补全（符号方式，无神经网络）
                let contextual_input = match symbolic_reasoner.contextualize(input) {
                    Some(rewritten) => {
                        println!("（结合对话上下文理解为：{}）", rewritten);
                        rewritten
                    }
                    None => input.to_string(),
                };
                let input = contextual_input.as_str();

                for note in symbolic_reasoner.entity_link_notes(input) {
                    println!("（{}）", note);
                }

                let tagged_words = preprocess(input, &config.nlp);
                let precomputed_query_entities =
                    extract_query_entities_from_tagged(&tagged_words, &config);
                let nlu_tagged_words = if config.nlp.enable_nlu.unwrap_or(false) {
                    Some(tagged_words.clone())
                } else {
                    None
                };
                let normalized_words = normalize_words(tagged_words, &synonyms);
                
                // 使用NLU进行语义理解
                if config.nlp.enable_nlu.unwrap_or(false) {
                    let semantic_analysis = match nlu_tagged_words {
                        Some(tagged) => nlu_engine.analyze_tagged(input, tagged),
                        None => nlu_engine.analyze(input),
                    };
                    
                    if !semantic_analysis.semantics.is_empty() {
                        let best_semantic = &semantic_analysis.semantics[0];
                        let confidence_threshold = config.nlp.nlu_confidence_threshold.unwrap_or(0.6);
                        
                        if best_semantic.score >= confidence_threshold {
                            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                            println!("NLU语义理解结果:");
                            println!("  领域: {}", best_semantic.domain);
                            println!("  意图: {}", best_semantic.intent);
                            println!("  置信度: {:.2}", best_semantic.score);
                            if !best_semantic.slots.is_empty() {
                                println!("  词槽:");
                                for (slot_name, slot_value) in &best_semantic.slots {
                                    println!("    {}: {}", slot_name, slot_value);
                                }
                            }
                            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
                            
                            // 根据NLU结果处理查询
                            match best_semantic.domain.as_str() {
                                "天气" => {
                                    let city = best_semantic.slots.iter()
                                        .find(|(name, _)| name == "城市")
                                        .map(|(_, value)| value.clone())
                                        .unwrap_or_else(|| "未知城市".to_string());
                                    let date = best_semantic.slots.iter()
                                        .find(|(name, _)| name == "日期")
                                        .map(|(_, value)| value.clone())
                                        .unwrap_or_else(|| "今天".to_string());
                                    
                                    match best_semantic.intent.as_str() {
                                        "雨" | "天气查询" => {
                                            println!("正在查询{}{}的天气信息...", city, date);
                                            println!("（此功能需要连接天气API服务）");
                                        }
                                        _ => {
                                            println!("暂不支持该天气查询类型: {}", best_semantic.intent);
                                        }
                                    }
                                }
                                "人物" => {
                                    let person = best_semantic.slots.iter()
                                        .find(|(name, _)| name == "人名")
                                        .map(|(_, value)| value.clone());
                                    
                                    if let Some(person_name) = person {
                                        match best_semantic.intent.as_str() {
                                            "身高查询" => {
                                                println!("正在查询{}的身高信息...", person_name);
                                                // 这里可以调用现有的知识图谱查询逻辑
                                                match local_semantic_adapter.get_entity_knowledge(&person_name).await {
                                                    Ok(response) => {
                                                        println!("─────────────────────────────────");
                                                        println!("实体: {}", response.data.entity);
                                                        println!("描述: {}", response.data.desc);
                                                        for avp in &response.data.avp {
                                                            if avp.len() >= 2 && (avp[0].contains("身高") || avp[0].contains("高度")) {
                                                                println!("身高: {}", avp[1]);
                                                            }
                                                        }
                                                        println!("─────────────────────────────────");
                                                    }
                                                    Err(e) => {
                                                        println!("获取{}身高信息时出错: {}，回退到本地符号推理...", person_name, e);
                                                        if let Some(answer) = try_symbolic_reasoning(&mut symbolic_reasoner, &mut contrast_arbiter, &oxigraph_manager, input, &precomputed_query_entities).await {
                                                            println!("{}", answer);
                                                        }
                                                    }
                                                }
                                            }
                                            "年龄查询" => {
                                                println!("正在查询{}的年龄信息...", person_name);
                                                if let Some(answer) = try_symbolic_reasoning(&mut symbolic_reasoner, &mut contrast_arbiter, &oxigraph_manager, input, &precomputed_query_entities).await {
                                                    println!("{}", answer);
                                                } else {
                                                    println!("知识库中没有{}的年龄记录。", person_name);
                                                }
                                            }
                                            _ => {
                                                println!("暂不支持该人物查询类型: {}，尝试本地符号推理...", best_semantic.intent);
                                                if let Some(answer) = try_symbolic_reasoning(&mut symbolic_reasoner, &mut contrast_arbiter, &oxigraph_manager, input, &precomputed_query_entities).await {
                                                    println!("{}", answer);
                                                }
                                            }
                                        }
                                    } else {
                                        println!("未能识别人名，请重新输入");
                                    }
                                }
                                "聊天" => {
                                    match best_semantic.intent.as_str() {
                                        "问候" => {
                                            println!("你好！我是KnowProlog知识图谱系统，很高兴为您服务！");
                                        }
                                        "感谢" => {
                                            println!("不客气！如果您有其他问题，随时可以问我。");
                                        }
                                        _ => {
                                            println!("很高兴和您聊天！");
                                        }
                                    }
                                }
                                "知识查询" => {
                                    // 继续使用查询逻辑
                                    if let Some((query_type, entities)) = analyze_query_intent(&normalized_words, &prolog_rules) {
                                        // 执行查询流程
                                        continue_original_query(&config, &oxigraph_manager, &mut prolog_engine, &response_formatter, &local_semantic_adapter, &external_knowledge_service, &mut symbolic_reasoner, &mut contrast_arbiter, input, &query_type, &entities, &precomputed_query_entities).await;
                                    } else if let Some(answer) = try_symbolic_reasoning(&mut symbolic_reasoner, &mut contrast_arbiter, &oxigraph_manager, input, &precomputed_query_entities).await {
                                        println!("{}", answer);
                                    } else {
                                        println!("无法理解您的查询，请重新描述您的问题。");
                                    }
                                }
                                _ => {
                                    println!("暂不支持该领域: {}", best_semantic.domain);
                                    // 如果NLU不能处理，回退到查询流程
                                    if let Some((query_type, entities)) = analyze_query_intent(&normalized_words, &prolog_rules) {
                                        continue_original_query(&config, &oxigraph_manager, &mut prolog_engine, &response_formatter, &local_semantic_adapter, &external_knowledge_service, &mut symbolic_reasoner, &mut contrast_arbiter, input, &query_type, &entities, &precomputed_query_entities).await;
                                    } else if let Some(answer) = try_symbolic_reasoning(&mut symbolic_reasoner, &mut contrast_arbiter, &oxigraph_manager, input, &precomputed_query_entities).await {
                                        println!("{}", answer);
                                    } else {
                                        println!("{}", config.messages.cannot_understand);
                                    }
                                }
                            }
                            continue; // 跳过后续查询逻辑
                        }
                    }
                    
                    // 如果NLU置信度不够，使用传统模式处理
                    println!("NLU置信度不足，使用传统模式处理查询...");
                }
                
                // 检查是否是直接的实体知识查询（例如："刘德华的信息" 或 "告诉我关于刘德华"）
                if input.contains("的信息") || input.contains("告诉我关于") || input.contains("什么是") {
                    // 提取实体名
                    let entity_name = input
                        .replace("的信息", "")
                        .replace("告诉我关于", "")
                        .replace("什么是", "")
                        .trim()
                        .to_string();
                    
                    if !entity_name.is_empty() {
                        println!("正在获取实体 '{}' 的知识...", entity_name);
                        
                        match local_semantic_adapter.get_entity_knowledge(&entity_name).await {
                            Ok(response) => {
                                println!("─────────────────────────────────");
                                println!("实体: {}", response.data.entity);
                                println!("描述: {}", response.data.desc);
                                if !response.data.tag.is_empty() {
                                    println!("标签: {}", response.data.tag.join(", "));
                                }
                                println!("\n属性信息:");
                                for avp in &response.data.avp {
                                    if avp.len() >= 2 {
                                        println!("  {}: {}", avp[0], avp[1]);
                                    }
                                }
                                println!("─────────────────────────────────");
                                continue;
                            }
                            Err(e) => {
                                println!("获取实体知识时出错: {}，尝试本地查询...", e);
                            }
                        }
                    }
                }
                
                // 检查是否是实体属性查询（例如："刘德华的年龄" 或 "苹果的颜色"）
                let attribute_re = ENTITY_ATTRIBUTE_RE.get_or_init(|| regex::Regex::new(r"(.+?)的(.+?)$").unwrap());
                if let Some(captures) = attribute_re.captures(input) {
                    if let (Some(entity_match), Some(attr_match)) = (captures.get(1), captures.get(2)) {
                        let entity_name = entity_match.as_str().trim();
                        let attribute = attr_match.as_str().trim();
                        
                        // 排除已经处理过的模式与符号推理问句（谁/吗/几/多少等）
                        if !input.contains("信息")
                            && !input.contains("什么")
                            && !input.contains("谁")
                            && !input.contains("吗")
                            && !input.contains("呢")
                            && !input.contains("几")
                            && !input.contains("多少")
                        {
                            println!("正在查询实体 '{}' 的属性 '{}'...", entity_name, attribute);
                            
                            match local_semantic_adapter.get_entity_attribute_value(entity_name, attribute).await {
                                Ok(response) => {
                                    println!("─────────────────────────────────");
                                    println!("实体: {}", response.data.entity);
                                    println!("属性: {}", response.data.attribute);
                                    println!("值: {}", response.data.value.join(", "));
                                    println!("─────────────────────────────────");
                                    continue;
                                }
                                Err(e) => {
                                    println!("获取属性值时出错: {}，尝试本地查询...", e);
                                }
                            }
                        }
                    }
                }
                
                if let Some((query_type, entities)) = analyze_query_intent(&normalized_words, &prolog_rules) {
                    println!("查询类型: {}, 实体: {:?}", query_type, entities);
                    
                    // 提取查询中的实体
                    let mut seen_entities = std::collections::HashSet::new();
                    let mut query_entities: Vec<String> = Vec::with_capacity(
                        precomputed_query_entities.len() + entities.len(),
                    );
                    for entity in precomputed_query_entities.iter().chain(entities.iter()) {
                        if seen_entities.insert(entity.as_str()) {
                            query_entities.push(entity.clone());
                        }
                    }
                    
                    // === 歧义解析 ===
                    if let Some(api_config) = &config.api {
                        if api_config.enable_entity_disambiguation {
                            println!("正在解析实体歧义...");
                            let mut resolved_entities = Vec::new();
                            let mut found_ambiguous = false;
                            
                            for entity in &query_entities {
                                match local_semantic_adapter.get_ambiguous_entities(entity).await {
                                    Ok(ambiguous_response) => {
                                        if ambiguous_response.data.len() > 1 {
                                            found_ambiguous = true;
                                            println!("发现实体 '{}' 有多种含义:", entity);
                                            for (i, (entity_name, weight)) in ambiguous_response.data.iter()
                                                .take(api_config.max_ambiguous_entities).enumerate() {
                                                println!("  {}. {} (权重: {})", i + 1, entity_name, weight);
                                            }
                                            
                                            // 根据配置决定是否自动选择
                                            if api_config.auto_select_best_entity {
                                                if let Some((best_entity, _)) = ambiguous_response.data.first() {
                                                    println!("自动选择: {}", best_entity);
                                                    resolved_entities.push(best_entity.clone());
                                                    
                                                    // 尝试获取并显示该实体的知识
                                                    match local_semantic_adapter.get_simplified_knowledge(best_entity).await {
                                                        Ok(knowledge) => {
                                                            if let Some(desc) = knowledge.get("description") {
                                                                if !desc.is_empty() {
                                                                    println!("  描述: {}", desc);
                                                                }
                                                            }
                                                            if let Some(tags) = knowledge.get("tags") {
                                                                if !tags.is_empty() {
                                                                    println!("  标签: {}", tags);
                                                                }
                                                            }
                                                        }
                                                        Err(e) => {
                                                            eprintln!("获取实体知识时出错: {}", e);
                                                        }
                                                    }
                                                }
                                            } else {
                                                // TODO: 实现用户交互选择逻辑
                                                if let Some((best_entity, _)) = ambiguous_response.data.first() {
                                                    resolved_entities.push(best_entity.clone());
                                                }
                                            }
                                        } else if ambiguous_response.data.len() == 1 {
                                            // 只有一个匹配，直接使用
                                            if let Some((entity_name, _)) = ambiguous_response.data.first() {
                                                resolved_entities.push(entity_name.clone());
                                            }
                                        } else {
                                            // 没有找到匹配，使用原始实体名
                                            resolved_entities.push(entity.clone());
                                        }
                                    }
                                    Err(e) => {
                                        eprintln!("查询实体 '{}' 的歧义信息时出错: {}", entity, e);
                                        resolved_entities.push(entity.clone());
                                    }
                                }
                            }
                            
                            if found_ambiguous {
                                println!("─────────────────────────────────");
                            }
                            
                            // 使用解析后的实体进行查询
                            let final_entities = if !resolved_entities.is_empty() {
                                resolved_entities
                            } else {
                                entities.clone()
                            };
                            
                            // 仅加载相关知识子集
                            let max_hops = 2; // 可配置的跳跃数
                            match oxigraph_manager.get_relevant_facts(&final_entities, max_hops).await {
                                Ok(facts) => {
                                    println!("加载了 {} 个相关事实", facts.len());
                                    prolog_engine.load_relevant_facts(facts);
                                    
                                    // 先尝试直接从RDF数据获取答案
                                    if let Some(answer) = oxigraph_manager.get_direct_answer(&query_type, &final_entities).await {
                                        println!("{}", answer);
                                    } else if let Some(prolog_query) = PrologEngine::generate_prolog_query(&query_type, &final_entities) {
                                        println!("Prolog查询: {}", prolog_query);
                                        
                                        let results = prolog_engine.query(&prolog_query);
                                        
                                        let response = response_formatter.format_prolog_results(&results, &query_type);
                                        println!("{}", response);
                                    } else {
                                        println!("无法生成有效的Prolog查询");
                                    }
                                }
                                Err(e) => {
                                    println!("{}: {}", config.messages.process_error, e);
                                }
                            }
                        } else {
                            // 歧义解析功能被禁用，直接查询
                            let max_hops = 2;
                            match oxigraph_manager.get_relevant_facts(&query_entities, max_hops).await {
                                Ok(facts) => {
                                    println!("加载了 {} 个相关事实", facts.len());
                                    prolog_engine.load_relevant_facts(facts);
                                    
                                    if let Some(answer) = oxigraph_manager.get_direct_answer(&query_type, &entities).await {
                                        println!("{}", answer);
                                    } else if let Some(prolog_query) = PrologEngine::generate_prolog_query(&query_type, &entities) {
                                        println!("Prolog查询: {}", prolog_query);
                                        
                                        let results = prolog_engine.query(&prolog_query);
                                        
                                        let response = response_formatter.format_prolog_results(&results, &query_type);
                                        println!("{}", response);
                                    } else {
                                        println!("无法生成有效的Prolog查询");
                                    }
                                }
                                Err(e) => {
                                    println!("{}: {}", config.messages.process_error, e);
                                }
                            }
                        }
                    } else {
                        // 没有API配置，直接查询
                        let max_hops = 2;
                        match oxigraph_manager.get_relevant_facts(&query_entities, max_hops).await {
                            Ok(facts) => {
                                println!("加载了 {} 个相关事实", facts.len());
                                prolog_engine.load_relevant_facts(facts);
                                
                                if let Some(answer) = oxigraph_manager.get_direct_answer(&query_type, &entities).await {
                                    println!("{}", answer);
                                } else if let Some(prolog_query) = PrologEngine::generate_prolog_query(&query_type, &entities) {
                                    println!("Prolog查询: {}", prolog_query);
                                    
                                    let results = prolog_engine.query(&prolog_query);
                                    
                                    let response = response_formatter.format_prolog_results(&results, &query_type);
                                    println!("{}", response);
                                } else {
                                    println!("无法生成有效的Prolog查询");
                                }
                            }
                            Err(e) => {
                                println!("{}: {}", config.messages.process_error, e);
                            }
                        }
                    }
                } else if let Some(answer) = try_symbolic_reasoning(
                    &mut symbolic_reasoner,
                    &mut contrast_arbiter,
                    &oxigraph_manager,
                    input,
                    &precomputed_query_entities,
                )
                .await
                {
                    println!("{}", answer);
                } else {
                    println!("{}", config.messages.cannot_understand);
                    
                    // 尝试使用外部知识API
                    if let Some(direct_answer) = get_direct_answer(input) {
                        println!("根据常识知识库：{}", direct_answer);
                    } else {
                        // 尝试使用OwnThink API查询
                        match external_knowledge_service.answer_question(input).await {
                            Ok(Some(answer)) => {
                                println!("根据外部知识：{}", answer);
                            }
                            Ok(None) => {
                                println!("很抱歉，我无法理解您的问题。请尝试重新描述。");
                            }
                            Err(e) => {
                                println!("查询外部知识时出错：{}。请检查网络连接。", e);
                            }
                        }
                    }
                }
            }
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

    println!("系统已退出。");
}

async fn load_graph_snapshot(
    oxigraph_manager: &OxigraphManager,
) -> Result<Arc<Vec<know_prolog::prolog_engine::PrologFact>>, Box<dyn std::error::Error + Send + Sync>> {
    oxigraph_manager.load_graph_snapshot().await
}

async fn refresh_symbolic_reasoner(
    symbolic_reasoner: &mut SymbolicReasoner,
    oxigraph_manager: &OxigraphManager,
) {
    if let Ok(facts) = load_graph_snapshot(oxigraph_manager).await {
        symbolic_reasoner.observe_facts(&facts);
    }
}

async fn try_symbolic_reasoning(
    symbolic_reasoner: &mut SymbolicReasoner,
    contrast_arbiter: &mut ContrastArbiter,
    oxigraph_manager: &OxigraphManager,
    input: &str,
    query_entities: &[String],
) -> Option<String> {
    if !query_entities.is_empty() {
        if let Ok(facts) = oxigraph_manager.get_relevant_facts(query_entities, 2).await {
            symbolic_reasoner.observe_facts(&facts);
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
            }
            let case = contrast_arbiter.contrast_statement(input, symbolic_reasoner);
            combined.push('\n');
            combined.push_str(&contrast_arbiter.render_case_brief(&case));
            return Some(combined);
        }
    }
    if let Some(answer) = symbolic_reasoner.answer(input) {
        // 图谱无记录时，用溯因假设生成候选答案（LLM 式的"猜测 + 理由"）
        if answer.contains("没有记录") || answer.contains("没有相关记录") || answer.contains("数量为 0")
        {
            if let Some(abduction) = symbolic_reasoner.hypothesis_answer(input) {
                return Some(format!("{}\n{}", answer, abduction));
            }
        }
        return Some(answer);
    }
    // 开放问句无直接答案：尝试溯因生成候选假设
    if let Some(abduction) = symbolic_reasoner.hypothesis_answer(input) {
        return Some(abduction);
    }
    // 逻辑问答失败时，优先用 OWL+DCG 话语引擎做连贯叙述，
    // 再回退到"知识图谱即语言模型"的统计生成式介绍
    let entity = symbolic_reasoner.best_linked_entity(input)?;
    symbolic_reasoner
        .narrate(&entity)
        .or_else(|| symbolic_reasoner.generate_about(&entity))
}

fn contains_question_marker(text: &str) -> bool {
    [
        "谁", "什么", "哪", "多少", "几", "怎么", "为什么", "如何", "吗", "呢", "何人", "何时",
    ]
    .iter()
    .any(|marker| text.contains(marker))
}

/// 判断输入是否以系统命令开头：澄清进行中时遇到命令应让位给常规处理
fn is_command_like(input: &str) -> bool {
    const COMMANDS: [&str; 84] = [
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
        "叙述",
        "讲述",
        "详细叙述",
        "复述",
        "教本体",
        "本体教学",
        "续写",
        "接龙",
        "补全",
        "关联规则",
        "关联联想",
        "联想",
        "推理发现",
        "推导发现",
        "对齐报告",
        "本体对齐",
        "推理规则",
        "链接预测规则",
        "实体链接",
        "识别实体",
        "判断",
        "蕴含",
        "校验",
        "形式化",
        "相似实体",
        "语义相似",
        "对话记忆",
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
        "分词：",
        "分词:",
        "概率推理",
        "可能世界",
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
        "忘记",
        "遗忘",
        "撤回",
        "会话规则",
        "对比",
        "校准",
        "仲裁",
    ];
    COMMANDS.iter().any(|command| input.starts_with(command))
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

/// 解析中文数词（1..=99）：十一、二十、二十三、九十九
fn parse_chinese_numeral(text: &str) -> Option<usize> {
    let digit = |ch: char| -> Option<usize> {
        Some(match ch {
            '一' => 1,
            '二' | '两' => 2,
            '三' => 3,
            '四' => 4,
            '五' => 5,
            '六' => 6,
            '七' => 7,
            '八' => 8,
            '九' => 9,
            _ => return None,
        })
    };
    let chars: Vec<char> = text.chars().collect();
    match chars.as_slice() {
        ['十'] => Some(10),
        [single] => digit(*single),
        ['十', ones] => Some(10 + digit(*ones)?),
        [tens, '十'] => Some(digit(*tens)? * 10),
        [tens, '十', ones] => Some(digit(*tens)? * 10 + digit(*ones)?),
        _ => None,
    }
}

/// 解析「采纳假设 N」中的编号：支持阿拉伯数字与「一、二、三…十、十二、二十三」
fn parse_hypothesis_index(text: &str) -> Option<usize> {
    let trimmed = text.trim().trim_start_matches(['：', ':', ' ', '第']);
    let trimmed = trimmed.trim_end_matches('条').trim();
    if let Ok(index) = trimmed.parse::<usize>() {
        return (index > 0).then_some(index);
    }
    let index = parse_chinese_numeral(trimmed)?;
    (index > 0).then_some(index)
}

async fn continue_original_query(
    config: &know_prolog::SystemConfig,
    oxigraph_manager: &OxigraphManager,
    prolog_engine: &mut PrologEngine,
    response_formatter: &ResponseFormatter,
    local_semantic_adapter: &LocalSemanticAdapter,
    external_knowledge_service: &ExternalKnowledgeService,
    symbolic_reasoner: &mut SymbolicReasoner,
    contrast_arbiter: &mut ContrastArbiter,
    input: &str,
    query_type: &str,
    entities: &[String],
    precomputed_query_entities: &[String]
) {
    println!("查询类型: {}, 实体: {:?}", query_type, entities);
    
    // 提取查询中的实体
    let mut seen_entities = std::collections::HashSet::new();
    let mut query_entities: Vec<String> =
        Vec::with_capacity(precomputed_query_entities.len() + entities.len());
    for entity in precomputed_query_entities.iter().chain(entities.iter()) {
        if seen_entities.insert(entity.as_str()) {
            query_entities.push(entity.clone());
        }
    }
    
    // 使用歧义解析功能（如果启用）
    if let Some(api_config) = &config.api {
        if api_config.enable_entity_disambiguation {
            println!("正在解析实体歧义...");
            let mut resolved_entities = Vec::new();
            let mut found_ambiguous = false;
            
            for entity in &query_entities {
                match local_semantic_adapter.get_ambiguous_entities(entity).await {
                    Ok(ambiguous_response) => {
                        if ambiguous_response.data.len() > 1 {
                            found_ambiguous = true;
                            println!("发现实体 '{}' 有多种含义:", entity);
                            for (i, (entity_name, weight)) in ambiguous_response.data.iter()
                                .take(api_config.max_ambiguous_entities).enumerate() {
                                println!("  {}. {} (权重: {})", i + 1, entity_name, weight);
                            }
                            
                            if api_config.auto_select_best_entity {
                                if let Some((best_entity, _)) = ambiguous_response.data.first() {
                                    println!("自动选择: {}", best_entity);
                                    resolved_entities.push(best_entity.clone());
                                    
                                    match local_semantic_adapter.get_simplified_knowledge(best_entity).await {
                                        Ok(knowledge) => {
                                            if let Some(desc) = knowledge.get("description") {
                                                if !desc.is_empty() {
                                                    println!("  描述: {}", desc);
                                                }
                                            }
                                            if let Some(tags) = knowledge.get("tags") {
                                                if !tags.is_empty() {
                                                    println!("  标签: {}", tags);
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            eprintln!("获取实体知识时出错: {}", e);
                                        }
                                    }
                                }
                            } else {
                                if let Some((best_entity, _)) = ambiguous_response.data.first() {
                                    resolved_entities.push(best_entity.clone());
                                }
                            }
                        } else if ambiguous_response.data.len() == 1 {
                            if let Some((entity_name, _)) = ambiguous_response.data.first() {
                                resolved_entities.push(entity_name.clone());
                            }
                        } else {
                            resolved_entities.push(entity.clone());
                        }
                    }
                    Err(e) => {
                        eprintln!("查询实体 '{}' 的歧义信息时出错: {}", entity, e);
                        resolved_entities.push(entity.clone());
                    }
                }
            }
            
            if found_ambiguous {
                println!("─────────────────────────────────");
            }
            
            let final_entities = if !resolved_entities.is_empty() {
                resolved_entities
            } else {
                entities.to_vec()
            };
            
            let max_hops = 2;
            match oxigraph_manager.get_relevant_facts(&final_entities, max_hops).await {
                Ok(facts) => {
                    println!("加载了 {} 个相关事实", facts.len());
                    prolog_engine.load_relevant_facts(facts);
                    
                    if let Some(answer) = oxigraph_manager.get_direct_answer(query_type, &final_entities).await {
                        println!("{}", answer);
                    } else if let Some(prolog_query) = PrologEngine::generate_prolog_query(query_type, &final_entities) {
                        println!("Prolog查询: {}", prolog_query);
                        
                        let results = prolog_engine.query(&prolog_query);
                        let response = response_formatter.format_prolog_results(&results, query_type);
                        println!("{}", response);
                    } else {
                        println!("无法生成有效的Prolog查询");
                    }
                }
                Err(e) => {
                    println!("{}: {}", config.messages.process_error, e);
                    if let Some(answer) =
                        try_symbolic_reasoning(symbolic_reasoner, contrast_arbiter, oxigraph_manager, input, precomputed_query_entities).await
                    {
                        println!("{}", answer);
                    }
                }
            }
        } else {
            // 歧义解析功能被禁用
            let max_hops = 2;
            match oxigraph_manager.get_relevant_facts(&query_entities, max_hops).await {
                Ok(facts) => {
                    println!("加载了 {} 个相关事实", facts.len());
                    prolog_engine.load_relevant_facts(facts);
                    
                    if let Some(answer) = oxigraph_manager.get_direct_answer(query_type, entities).await {
                        println!("{}", answer);
                    } else if let Some(prolog_query) = PrologEngine::generate_prolog_query(query_type, entities) {
                        println!("Prolog查询: {}", prolog_query);
                        
                        let results = prolog_engine.query(&prolog_query);
                        let response = response_formatter.format_prolog_results(&results, query_type);
                        println!("{}", response);
                    } else {
                        println!("无法生成有效的Prolog查询");
                    }
                }
                Err(e) => {
                    println!("{}: {}", config.messages.process_error, e);
                    if let Some(answer) =
                        try_symbolic_reasoning(symbolic_reasoner, contrast_arbiter, oxigraph_manager, input, precomputed_query_entities).await
                    {
                        println!("{}", answer);
                    }
                }
            }
        }
    } else {
        // 没有API配置
        let max_hops = 2;
        match oxigraph_manager.get_relevant_facts(&query_entities, max_hops).await {
            Ok(facts) => {
                println!("加载了 {} 个相关事实", facts.len());
                prolog_engine.load_relevant_facts(facts);
                
                if let Some(answer) = oxigraph_manager.get_direct_answer(query_type, entities).await {
                    println!("{}", answer);
                } else if let Some(prolog_query) = PrologEngine::generate_prolog_query(query_type, entities) {
                    println!("Prolog查询: {}", prolog_query);
                    
                    let results = prolog_engine.query(&prolog_query);
                    let response = response_formatter.format_prolog_results(&results, query_type);
                    println!("{}", response);
                } else {
                    println!("无法生成有效的Prolog查询");
                }
            }
            Err(e) => {
                println!("{}: {}", config.messages.process_error, e);

                if let Some(answer) =
                    try_symbolic_reasoning(symbolic_reasoner, contrast_arbiter, oxigraph_manager, input, precomputed_query_entities).await
                {
                    println!("{}", answer);
                } else {
                    // 本地查询失败时，尝试外部知识API
                    println!("本地查询失败，尝试外部知识源...");

                    if let Some(direct_answer) = get_direct_answer(input) {
                        println!("根据常识知识库：{}", direct_answer);
                    } else {
                        match external_knowledge_service.answer_question(input).await {
                            Ok(Some(answer)) => {
                                println!("根据外部知识：{}", answer);
                            }
                            Ok(None) => {
                                println!("很抱歉，我无法找到相关信息。");
                            }
                            Err(e) => {
                                println!("查询外部知识时出错：{}。", e);
                            }
                        }
                    }
                }
            }
        }
    }
}

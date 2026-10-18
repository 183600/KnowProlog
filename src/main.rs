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
    external_knowledge::{AnswerOutcome, ExternalKnowledgeService, get_direct_answer},
    external_ledger::{ExternalLedger, LedgerLoad},
    symbolic_reasoner::SymbolicReasoner,
    dialogue_frame::ClarificationOutcome,
    neuro_symbolic_contrast::ContrastArbiter,
    dialogue_manager::DialogueManager,
    holographic_memory::{HoloAnalogyHit, HoloHit},
    import_journal::{ImportJournal, JournalLoad},
    perf_ledger::PerfLedger,
    grammar_extractor::GrammarExtractor,
    prolog_engine::PrologFact,
    chronicle::{Chronicle, FsyncPolicy},
    view_render::{
        render_coverage_report, render_proof_ascii, render_proof_mermaid, render_timeline_flow,
        render_timeline_markdown, CoverageInput, KnowledgeSubgraph, TimelineTurn,
    },
    embedding_eval::{
        compare_baseline, evaluate, triples_from_prolog_facts, EvalBaseline, EvalConfig,
    },
    dialogue_eval::{
        compare_baseline as compare_dialogue_baseline, evaluate_seed_with, EvalKnowledge,
        DialogueEvalBaseline,
    },
    prompt_spec::{self, LintSeverity, PromptMode},
};

#[derive(Parser)]
#[command(name = "knowprolog")]
#[command(about = "KnowProlog - Chinese knowledge graph system")]
#[command(version)]
struct Cli {
    /// Override the configured Oxigraph store path (all subcommands)
    #[arg(long, global = true)]
    store: Option<String>,
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
    /// Evaluate symbolic embeddings (link prediction + analogy) with a
    /// deterministic holdout split and regression ratchet
    Eval {
        /// Test split ratio in per mille (0-1000)
        #[arg(long, default_value_t = 200)]
        test_ratio: u32,
        /// Baseline JSON path; enables the regression ratchet
        #[arg(long)]
        baseline: Option<String>,
        /// Write current metrics to the baseline path
        #[arg(long)]
        update_baseline: bool,
        /// Do not fail (exit code) on regressions
        #[arg(long)]
        allow_regression: bool,
        /// RRF fusion constant
        #[arg(long, default_value_t = 60.0)]
        rrf_k: f64,
    },
    /// Evaluate dialogue NLU (intent / entity linking / slots / clarification /
    /// answers / provenance) on the built-in gold corpus with a regression ratchet
    EvalNlu {
        /// Knowledge backend: "reasoner" (end-to-end) or "scripted" (policy layer)
        #[arg(long, default_value = "reasoner")]
        knowledge: String,
        /// Prompt asset mode: "off" (legacy renderer only), "parity" (dual-run
        /// comparison), or "on" (grounded templates with legacy fallback)
        #[arg(long, default_value = "on")]
        prompt_mode: String,
        /// Baseline JSON path; enables the regression ratchet
        #[arg(long)]
        baseline: Option<String>,
        /// Write current metrics to the baseline path
        #[arg(long)]
        update_baseline: bool,
        /// Do not fail (exit code) on regressions
        #[arg(long)]
        allow_regression: bool,
        /// Print the report as JSON
        #[arg(long)]
        json: bool,
    },
    /// Lint the prompt asset registry (executable examples + template checks)
    PromptLint {
        /// Print the lint report as JSON
        #[arg(long)]
        json: bool,
    },
    /// Run the session REST service (multi-turn dialogue over HTTP + SSE)
    Serve {
        /// Listen address
        #[arg(long, default_value = "127.0.0.1:8080")]
        addr: String,
        /// Maximum concurrent sessions
        #[arg(long, default_value_t = 64)]
        max_sessions: usize,
        /// Prompt asset mode for sessions: "off", "parity", or "on"
        #[arg(long, default_value = "on")]
        prompt_mode: String,
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

    // 会话级性能台账：存储与推理共享同一实例，报告覆盖整条链路
    let perf_ledger = PerfLedger::shared();

    // 存储路径可被 --store 覆盖（便于对独立知识库做导入/评估而不动主库）
    let store_path = cli
        .store
        .clone()
        .unwrap_or_else(|| config.system.oxigraph_store_path.clone());

    // 创建Oxigraph存储
    let oxigraph_manager = match OxigraphManager::with_perf_ledger(
        &store_path,
        perf_ledger.clone(),
    ) {
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
        Some(Commands::Eval {
            test_ratio,
            baseline,
            update_baseline,
            allow_regression,
            rrf_k,
        }) => {
            match oxigraph_manager.load_all_relations().await {
                Ok(relations) => {
                    let triples = triples_from_prolog_facts(&relations);
                    if triples.len() < 2 {
                        eprintln!(
                            "知识库中的关系三元组不足（{} 条），无法评估。可先导入 JSON-LD 知识库。",
                            triples.len()
                        );
                        std::process::exit(1);
                    }
                    let eval_config = EvalConfig {
                        test_ratio_per_mille: test_ratio,
                        rrf_k: rrf_k.max(1.0),
                        ..Default::default()
                    };
                    let report = evaluate(&triples, &eval_config);
                    println!("{}", report.render_text(&eval_config));

                    let baseline_path = baseline
                        .clone()
                        .unwrap_or_else(|| "eval_baseline.json".to_string());
                    if update_baseline {
                        let new_baseline = EvalBaseline::from_report(&report);
                        match new_baseline.save(&baseline_path) {
                            Ok(()) => println!("基线已写入 {}", baseline_path),
                            Err(e) => {
                                eprintln!("基线写入失败: {}", e);
                                std::process::exit(1);
                            }
                        }
                    } else if baseline.is_some() {
                        match EvalBaseline::load(&baseline_path) {
                            Ok(old_baseline) => {
                                let verdict = compare_baseline(&old_baseline, &report, 1e-6);
                                print!("{}", verdict.render_text());
                                if !verdict.passed && !allow_regression {
                                    std::process::exit(1);
                                }
                            }
                            Err(e) => {
                                eprintln!("基线读取失败: {}", e);
                                std::process::exit(1);
                            }
                        }
                    }
                }
                Err(e) => {
                    eprintln!("读取知识库失败: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Some(Commands::EvalNlu {
            knowledge,
            prompt_mode,
            baseline,
            update_baseline,
            allow_regression,
            json,
        }) => {
            let mode = match knowledge.as_str() {
                "scripted" => EvalKnowledge::Scripted,
                "reasoner" => EvalKnowledge::Reasoner,
                other => {
                    eprintln!("未知的知识后端「{}」：可选 reasoner / scripted", other);
                    std::process::exit(1);
                }
            };
            let prompt_mode = parse_prompt_mode(&prompt_mode);
            let report = evaluate_seed_with(mode, prompt_mode);
            if json {
                match serde_json::to_string_pretty(&report) {
                    Ok(text) => println!("{}", text),
                    Err(e) => {
                        eprintln!("报告序列化失败: {}", e);
                        std::process::exit(1);
                    }
                }
            } else {
                print!("{}", report.render_text());
            }

            let baseline_path = baseline
                .clone()
                .unwrap_or_else(|| "nlu_eval_baseline.json".to_string());
            if update_baseline {
                let new_baseline = DialogueEvalBaseline::from_report(&report);
                match new_baseline.save(&baseline_path) {
                    Ok(()) => println!("对话 NLU 基线已写入 {}", baseline_path),
                    Err(e) => {
                        eprintln!("基线写入失败: {}", e);
                        std::process::exit(1);
                    }
                }
            } else if baseline.is_some() {
                match DialogueEvalBaseline::load(&baseline_path) {
                    Ok(old_baseline) => {
                        let verdict = compare_dialogue_baseline(&old_baseline, &report, 1e-6);
                        print!("{}", verdict.render_text());
                        if !verdict.passed && !allow_regression {
                            std::process::exit(1);
                        }
                    }
                    Err(e) => {
                        eprintln!("基线读取失败: {}", e);
                        std::process::exit(1);
                    }
                }
            }
        }
        Some(Commands::PromptLint { json }) => {
            run_prompt_lint(json);
        }
        Some(Commands::Serve {
            addr,
            max_sessions,
            prompt_mode,
        }) => {
            let prompt_mode = parse_prompt_mode(&prompt_mode);
            let facts = match oxigraph_manager.load_all_facts().await {
                Ok(facts) => facts,
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
            println!(
                "KnowProlog 会话服务已启动：http://{}（{} 条事实，最多 {} 个会话，提示模式 {}）",
                addr,
                facts.len(),
                max_sessions,
                prompt_mode.label()
            );
            let mut api_state = know_prolog::serve_api::ApiState::new(
                facts,
                config.reasoning.clone(),
                synonyms,
                perf_ledger.clone(),
                max_sessions,
            );
            api_state.set_prompt_mode(prompt_mode);
            let state = Arc::new(api_state);
            if let Err(e) = know_prolog::serve_api::serve(state, &addr).await {
                eprintln!("服务退出: {}", e);
                std::process::exit(1);
            }
        }
        Some(Commands::Interactive) | None => {
            // 运行交互模式
            run_interactive_mode(config, oxigraph_manager, perf_ledger).await;
        }
    }
}

/// 解析 `--prompt-mode`：off / parity / on，非法值立即退出。
fn parse_prompt_mode(value: &str) -> PromptMode {
    PromptMode::parse(value).unwrap_or_else(|| {
        eprintln!("未知的提示模式「{}」：可选 off / parity / on", value);
        std::process::exit(1);
    })
}

/// `prompt-lint`：校验提示资产注册表；存在 Error 时以非零码退出。
fn run_prompt_lint(json: bool) {
    let specs = prompt_spec::default_specs();
    let issues = prompt_spec::lint_default();
    let (covered, legacy) = prompt_spec::coverage();
    let errors = issues
        .iter()
        .filter(|issue| issue.severity == LintSeverity::Error)
        .count();
    if json {
        let value = serde_json::json!({
            "version": prompt_spec::PROMPT_SPEC_VERSION,
            "count": specs.len(),
            "examples_checked": specs.len(),
            "coverage": {
                "covered": covered,
                "legacy": legacy,
            },
            "errors": errors,
            "lint": issues,
        });
        match serde_json::to_string_pretty(&value) {
            Ok(text) => println!("{}", text),
            Err(e) => {
                eprintln!("lint 报告序列化失败: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        println!(
            "提示资产注册表 v{}：{} 条模板，已外化策略 {}/{}",
            prompt_spec::PROMPT_SPEC_VERSION,
            specs.len(),
            covered.len(),
            covered.len() + legacy.len()
        );
        for issue in &issues {
            let severity = match issue.severity {
                LintSeverity::Error => "ERROR",
                LintSeverity::Warning => "WARN ",
            };
            println!("{} [{}] {}", severity, issue.spec_id, issue.message);
        }
        if issues.is_empty() {
            println!("lint 通过：{} 条示例逐字渲染一致，无错误与警告", specs.len());
        }
        if !legacy.is_empty() {
            println!("仍由旧组合生成器负责的策略：{}", legacy.join("、"));
        }
    }
    if errors > 0 {
        std::process::exit(1);
    }
}

async fn run_interactive_mode(
    config: Arc<know_prolog::SystemConfig>,
    oxigraph_manager: OxigraphManager,
    perf_ledger: Arc<PerfLedger>,
) {
    println!("{}", config.messages.startup);

    // 谓词画像注册表：跨会话加载已确认的准入/别名/互逆，再叠加配置声明
    let registry_path = format!("{}/predicate_registry.json", config.system.oxigraph_store_path);
    if oxigraph_manager.load_predicate_registry(&registry_path) {
        println!("已恢复谓词画像注册表。可用「谓词报告」查看。");
    }
    oxigraph_manager.configure_predicates(
        &config.predicate_profiles,
        config.reasoning.require_predicate_admission,
    );

    let knowledge_extractor = KnowledgeExtractor::new(config.clone());
    // 文法式关系抽取器：DCG 语言器官，词库由谓词画像注册表注入
    let grammar_extractor = GrammarExtractor::new();
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
    
    // 创建外部知识服务：外链名册为外部知识留册（出处 + 逻辑时刻 + 时限），
    // 跨轮、跨重启记住"从哪里、在第几轮、读到了什么"，避免重复打网。
    let external_ledger_path = config
        .api
        .as_ref()
        .and_then(|api| api.external_ledger_path.clone())
        .filter(|path| !path.trim().is_empty())
        .unwrap_or_else(|| {
            format!(
                "{}/external_ledger.json",
                config.system.oxigraph_store_path
            )
        });
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
    let mut symbolic_reasoner = SymbolicReasoner::with_perf_ledger(
        config.reasoning.clone(),
        synonyms.clone(),
        perf_ledger.clone(),
    );
    // 谓词归一：把画像里的别名/互逆词面同步到查询解析与生成词面。
    // 正向词面走别名（alias → 槽位，可参与生成叙事）；逆向词面只登记
    // 查询换向（如「徒弟」「学生」命中 师傅 槽位时主宾对调），不改词面表。
    sync_predicate_pairs(&mut symbolic_reasoner, &oxigraph_manager);
    // 导入日志：跨重启的有界撤销栈
    let journal_path = format!("{}/import_journal.json", config.system.oxigraph_store_path);
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
    // 批量导入期间把整图刷新推迟到真正需要时（见循环顶部）
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
            match oxigraph_manager.sync_rule_ledger(&startup_rule_entries).await {
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
    if dialogue_enabled {
        if let Some(state_path) = config.reasoning.dialogue_state_path.as_deref() {
            let manager_path = format!("{}.manager.json", state_path);
            if dialogue_manager.restore_from_path(&manager_path) {
                println!("已恢复对话本体记忆：{}", dialogue_manager.summary());
            }
        }
    }

    // 对话编年史：追加式哈希链档案，是对话的真相源；快照只是检查点。
    // 每轮输入在任何处理之前先落盘（写前日志），崩溃最多丢进行中的一轮；
    // 坏行只废尾巴不废全档：可信前缀保留、原子重写尾修复后续写。
    let chronicle_path = config
        .reasoning
        .dialogue_state_path
        .as_deref()
        .map(|path| format!("{}.chronicle.jsonl", path))
        .unwrap_or_else(|| {
            format!(
                "{}/dialogue.chronicle.jsonl",
                config.system.oxigraph_store_path
            )
        });
    let mut chronicle = match Chronicle::open(&chronicle_path, FsyncPolicy::EachAppend) {
        Ok((chronicle, audit)) => {
            if !audit.is_clean() {
                eprintln!("对话编年史{}", audit.verdict());
                match chronicle.repair_tail() {
                    Ok(_) => {
                        println!(
                            "编年史已修复尾行并保留可信前缀：{} 条记录。",
                            chronicle.len()
                        );
                        Some(chronicle)
                    }
                    Err(error) => {
                        eprintln!("编年史尾修复失败（本轮不落史以免污染链条）：{}", error);
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
    // KB 修订号：知识库每发生一次批量变更（导入/撤销）自增，随编年史记录，
    // 用于区分「重演」时的知识漂移（挂钟时间不可断言，序号才是语义时戳）。
    let mut kb_revision: u64 = chronicle
        .as_ref()
        .map(|chronicle| chronicle.kb_revision())
        .unwrap_or(0);

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

        // 外链名册的逻辑时刻：每个非空用户轮次推进一次（TTL 的计量单位）
        if !input.is_empty() {
            external_knowledge_service.advance_tick();
        }

        if dialogue_enabled && !input.is_empty() {
            dialogue_manager.observe_input(input);
        }

        // 全息时序记忆：把本轮输入中提及的实体压入置换编码的焦点叠加（多轮"注意力轨迹"）
        if config.reasoning.enable_holographic_memory && !input.is_empty() {
            symbolic_reasoner.note_holo_context(input);
        }

        // 导入行不立即刷新：进入查询/命令前统一观察整图
        if reasoner_stale
            && !matches!(
                detect_input_mode(input, &config.nlp.import_keywords),
                InputMode::KnowledgeImport
            )
        {
            refresh_symbolic_reasoner(&mut symbolic_reasoner, &oxigraph_manager).await;
            reasoner_stale = false;
            kb_revision += 1;
            if let Some(chronicle) = chronicle.as_mut() {
                chronicle.set_kb_revision(kb_revision);
            }
        }

        // 编年史写前日志：用户输入先落盘再处理；记录当轮开始时的焦点与对话水位，
        // 重启后据此回放/重建上下文（崩溃语义：最多丢进行中的一轮）。
        if !input.is_empty() {
            if let Some(chronicle) = chronicle.as_mut() {
                let mode = match detect_input_mode(input, &config.nlp.import_keywords) {
                    InputMode::KnowledgeImport => "import",
                    InputMode::Query => "query",
                };
                let kind = if is_command_like(input) { "command" } else { mode };
                let focus = dialogue_manager.focus_entity();
                let _ = chronicle.append_turn(
                    input,
                    kind,
                    focus,
                    dialogue_manager.turn_count(),
                );
            }
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
                    let turns: Vec<TimelineTurn> =
                        entries.iter().filter_map(|entry| TimelineTurn::from_entry(entry)).collect();
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

        // 编年史检索：跨会话长程记忆（命中哪一轮、命中几次，可审计）
        if let Some(query) = strip_any_prefix(
            input,
            &["聊过：", "聊过:", "聊过 ", "查档案：", "查档案:"],
        ) {
            let query = query.trim();
            match (chronicle.as_ref(), query.is_empty()) {
                (Some(chronicle), false) => {
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
                (Some(_), true) => println!("用法：聊过：<关键词>，例如：聊过：姚明"),
                (None, _) => println!("编年史不可用（打开失败，见启动警告）。"),
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

        // 证明图：把归结证明树渲染成 Mermaid/ASCII（看得见的推理）
        if let Some(target) = strip_any_prefix(
            input,
            &[
                "证明图：", "证明图:", "证明树：", "证明树:", "画证明：", "画证明:",
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
                            println!(
                                "未得证：深度上限 {} 内无归结反驳（开世界假设：推不出 ≠ 为假）。",
                                view.max_depth
                            );
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
                "知识子图：", "知识子图:", "子图：", "子图:", "邻域：", "邻域:",
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
                            PrologFact::Entity { name, properties, .. } => {
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
                    for profile in &config.predicate_profiles {
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

        // 帧澄清进行中：短答（序号 / 实体名 / 都不是）→ 确定性绑定 → 重放原问句。
        // 绑定有出处（候选 + 轮次），无源可填的洞不猜；非回答输入零状态变化、放行。
        let mut frame_replay: Option<String> = None;
        match handle_frame_clarification(&mut symbolic_reasoner, input) {
            FrameClarificationStep::Replay(question) => frame_replay = Some(question),
            FrameClarificationStep::Handled => continue,
            FrameClarificationStep::Pass => {}
        }
        let input = frame_replay.as_deref().unwrap_or(input);

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

        // 提及账本：每个指称的证据收据（边界 + 类型证据 × 疑问词期待 + 焦点先验）
        if let Some(text) = strip_any_prefix(input, &["提及账本：", "提及账本:", "账本：", "账本:"]) {
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
                "全息检索：", "全息检索:", "全息查询：", "全息查询:", "全息补全：", "全息补全:",
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
                sync_rule_ledger_report(&symbolic_reasoner, &oxigraph_manager).await;
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
                sync_rule_ledger_report(&symbolic_reasoner, &oxigraph_manager).await;
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

        // 登记谓词别名：别名 = 规范谓词（查询/生成/导入共用同一词面）
        if let Some(spec) = strip_any_prefix(
            input,
            &[
                "登记别名：", "登记别名:", "谓词别名：", "谓词别名:", "同义谓词：", "同义谓词:",
            ],
        ) {
            match parse_predicate_pair(spec) {
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
                None => println!(
                    "请用「登记别名：别名 = 规范谓词」的格式，例如：登记别名：老师 = 师傅"
                ),
            }
            println!();
            continue;
        }

        // 登记互逆谓词：左 = 右（p(a,b) 与 q(b,a) 等价，导入自动统一方向）
        if let Some(spec) = strip_any_prefix(
            input,
            &["登记逆谓词：", "登记逆谓词:", "逆谓词：", "逆谓词:", "互逆谓词：", "互逆谓词:"],
        ) {
            match parse_predicate_pair(spec) {
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
                None => println!(
                    "请用「登记逆谓词：左 = 右」的格式，例如：登记逆谓词：师傅 = 徒弟"
                ),
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
                println!("已标记功能谓词「{}」：同主语的新值将与原值冲突并等待裁决。", predicate);
            } else {
                println!("「{}」已是功能谓词。", predicate);
            }
            println!();
            continue;
        }

        if let Some(predicate) = strip_any_prefix(input, &["取消功能谓词：", "取消功能谓词:"]) {
            let predicate = predicate.trim();
            if !predicate.is_empty() && oxigraph_manager.mark_predicate_functional(predicate, false) {
                let _ = oxigraph_manager.save_predicate_registry(&registry_path);
                println!("已取消「{}」的功能谓词标记。", predicate);
            } else {
                println!("「{}」不是功能谓词。", predicate);
            }
            println!();
            continue;
        }

        // 确认/拒绝待定谓词：准入对话的裁决入口
        if let Some(predicate) = strip_any_prefix(input, &["确认谓词：", "确认谓词:"]) {
            let predicate = predicate.trim();
            if !predicate.is_empty() && oxigraph_manager.confirm_predicate(predicate) {
                let _ = oxigraph_manager.save_predicate_registry(&registry_path);
                println!("已确认谓词「{}」：画像转为已声明，可用于稳定推理。", predicate);
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
                        println!("已拒绝谓词「{}」并移除 {} 条相关三元组。", predicate, removed);
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
                        reasoner_stale = true;
                        let _ = oxigraph_manager.save_predicate_registry(&registry_path);
                        let _ = import_journal.save(&journal_path);
                        println!(
                            "已撤销上次导入：移除 {} 条三元组（日志余 {} 笔）。",
                            removed,
                            import_journal.len()
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

        // 显式刷新：删除旧页，下次提问重新取回（外链数据不在本地写权限内，
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
                println!("已从外链名册删除「{}」。", entity);
            } else {
                println!("外链名册中没有「{}」的条目。", entity);
            }
            println!();
            continue;
        }

        // 确认同义链接：把推测升格为别名知识（纯符号的 sameAs 学习，可显式撤销）
        if let Some(body) = strip_any_prefix(
            input,
            &["同义链接：", "同义链接:", "外链别名：", "外链别名:"],
        ) {
            match parse_alias_link(body) {
                Some((alias, target)) => {
                    if external_knowledge_service.confirm_alias(alias, target) {
                        let _ = external_knowledge_service.save_ledger(&external_ledger_path);
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
            println!("{}", perf_ledger.report_with_title("性能台账（本会话累计）"));
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

        // 本体化对话管理器（纯对话行为）：回顾、解释依据、续聊、展开、寒暄。
        // 用对话本体 + 逻辑程序策略推导话语行为，再由模板组合生成回答。
        if dialogue_enabled {
            if let Some(reply) = dialogue_manager.plan_pure_dialogue(input, &symbolic_reasoner) {
                println!("{}", reply.text);
                let provenance: Vec<String> = dialogue_manager
                    .turns()
                    .last()
                    .map(|turn| turn.provenance.clone())
                    .unwrap_or_default();
                log_chronicle_reply(&mut chronicle, &reply.text, &reply.plan, &provenance);
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
                            // 关键词未出现时原样返回
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
                    let mut items = knowledge_extractor.extract(&clean_text, &normalized_words);
                    // 中文关系句 → 关系三元组（如「张三是李四的师傅」），
                    // 叠加 DCG 文法抽取（被字句/把字句/并列/所有格换向）
                    grammar_extractor
                        .set_lexicon_if_changed(&oxigraph_manager.predicate_query_pairs());
                    items.extend(extract_relationships_from_text(
                        &clean_text,
                        &normalized_words,
                        &grammar_extractor,
                    ));
                    
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
                        
                        match oxigraph_manager.save_knowledge_with_receipt(items).await {
                            Ok(receipt) => {
                                let summary = receipt.summary();
                                println!("{}", summary);
                                log_chronicle_reply(&mut chronicle, &summary, &[], &[]);
                                if receipt.can_undo() {
                                    import_journal.push(receipt);
                                    let _ = import_journal.save(&journal_path);
                                }
                                reasoner_stale = true;
                                let _ = oxigraph_manager.save_predicate_registry(&registry_path);
                            }
                            Err(e) => eprintln!("保存失败: {}", e),
                        }
                    }
                }
            }
            InputMode::Query => {
                // 多轮对话上下文：指代消解与省略补全（话语帧合一，符号方式，无神经网络）
                let contextual_input = match symbolic_reasoner.contextualize_with_receipt(input) {
                    Some(rewrite) => {
                        for note in &rewrite.notes {
                            println!("（{}）", note);
                        }
                        println!("（结合对话上下文理解为：{}）", rewrite.text);
                        rewrite.text
                    }
                    None => {
                        // 无源可填的实体洞：不落进「无法理解」，转为一次显式绑定请求
                        // （澄清 = 请求一次绑定；既有通道必然失败时才开启）。
                        if let Some(prompt) = symbolic_reasoner.start_frame_clarification(input) {
                            println!("{}", prompt);
                            println!();
                            continue;
                        }
                        input.to_string()
                    }
                };
                let input = contextual_input.as_str();

                for note in symbolic_reasoner.entity_link_notes(input) {
                    println!("（{}）", note);
                }

                // 本体化对话管理器：实体已知的知识问句给出"结论 + 出处 + 追问建议"式组合回答
                if dialogue_enabled {
                    if let Some(reply) = dialogue_manager.plan_reply(input, &symbolic_reasoner) {
                        println!("{}", reply.text);
                        let provenance: Vec<String> = dialogue_manager
                            .turns()
                            .last()
                            .map(|turn| turn.provenance.clone())
                            .unwrap_or_default();
                        log_chronicle_reply(&mut chronicle, &reply.text, &reply.plan, &provenance);
                        println!();
                        continue;
                    }
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
                                                            log_chronicle_reply(&mut chronicle, &answer, &[], &[]);
                                                        }
                                                    }
                                                }
                                            }
                                            "年龄查询" => {
                                                println!("正在查询{}的年龄信息...", person_name);
                                                if let Some(answer) = try_symbolic_reasoning(&mut symbolic_reasoner, &mut contrast_arbiter, &oxigraph_manager, input, &precomputed_query_entities).await {
                                                    println!("{}", answer);
                                                    log_chronicle_reply(&mut chronicle, &answer, &[], &[]);
                                                } else {
                                                    println!("知识库中没有{}的年龄记录。", person_name);
                                                }
                                            }
                                            _ => {
                                                println!("暂不支持该人物查询类型: {}，尝试本地符号推理...", best_semantic.intent);
                                                if let Some(answer) = try_symbolic_reasoning(&mut symbolic_reasoner, &mut contrast_arbiter, &oxigraph_manager, input, &precomputed_query_entities).await {
                                                    println!("{}", answer);
                                                    log_chronicle_reply(&mut chronicle, &answer, &[], &[]);
                                                }
                                            }
                                        }
                                    } else {
                                        println!("未能识别人名，请重新输入");
                                    }
                                }
                                "聊天" => {
                                    let chat_reply = match best_semantic.intent.as_str() {
                                        "问候" => "你好！我是KnowProlog知识图谱系统，很高兴为您服务！",
                                        "感谢" => "不客气！如果您有其他问题，随时可以问我。",
                                        _ => "很高兴和您聊天！",
                                    };
                                    println!("{}", chat_reply);
                                    log_chronicle_reply(&mut chronicle, chat_reply, &[], &[]);
                                }
                                "知识查询" => {
                                    // 继续使用查询逻辑
                                    if let Some((query_type, entities)) = analyze_query_intent(&normalized_words, &prolog_rules) {
                                        // 执行查询流程
                                        continue_original_query(&config, &oxigraph_manager, &mut prolog_engine, &response_formatter, &local_semantic_adapter, &external_knowledge_service, &mut symbolic_reasoner, &mut contrast_arbiter, input, &query_type, &entities, &precomputed_query_entities).await;
                                    } else if let Some(answer) = try_symbolic_reasoning(&mut symbolic_reasoner, &mut contrast_arbiter, &oxigraph_manager, input, &precomputed_query_entities).await {
                                        println!("{}", answer);
                                        log_chronicle_reply(&mut chronicle, &answer, &[], &[]);
                                    } else {
                                        let fallback = "无法理解您的查询，请重新描述您的问题。";
                                        println!("{}", fallback);
                                        log_chronicle_reply(&mut chronicle, fallback, &[], &[]);
                                    }
                                }
                                _ => {
                                    println!("暂不支持该领域: {}", best_semantic.domain);
                                    // 如果NLU不能处理，回退到查询流程
                                    if let Some((query_type, entities)) = analyze_query_intent(&normalized_words, &prolog_rules) {
                                        continue_original_query(&config, &oxigraph_manager, &mut prolog_engine, &response_formatter, &local_semantic_adapter, &external_knowledge_service, &mut symbolic_reasoner, &mut contrast_arbiter, input, &query_type, &entities, &precomputed_query_entities).await;
                                    } else if let Some(answer) = try_symbolic_reasoning(&mut symbolic_reasoner, &mut contrast_arbiter, &oxigraph_manager, input, &precomputed_query_entities).await {
                                        println!("{}", answer);
                                        log_chronicle_reply(&mut chronicle, &answer, &[], &[]);
                                    } else {
                                        println!("{}", config.messages.cannot_understand);
                                        log_chronicle_reply(&mut chronicle, &config.messages.cannot_understand, &[], &[]);
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
                                        log_chronicle_reply(&mut chronicle, &answer, &[], &[]);
                                    } else if let Some(prolog_query) = PrologEngine::generate_prolog_query(&query_type, &final_entities) {
                                        println!("Prolog查询: {}", prolog_query);
                                        
                                        let results = prolog_engine.query(&prolog_query);
                                        
                                        let response = response_formatter.format_prolog_results(&results, &query_type);
                                        println!("{}", response);
                                        log_chronicle_reply(&mut chronicle, &response, &[], &[]);
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
                                        log_chronicle_reply(&mut chronicle, &answer, &[], &[]);
                                    } else if let Some(prolog_query) = PrologEngine::generate_prolog_query(&query_type, &entities) {
                                        println!("Prolog查询: {}", prolog_query);
                                        
                                        let results = prolog_engine.query(&prolog_query);
                                        
                                        let response = response_formatter.format_prolog_results(&results, &query_type);
                                        println!("{}", response);
                                        log_chronicle_reply(&mut chronicle, &response, &[], &[]);
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
                                    log_chronicle_reply(&mut chronicle, &answer, &[], &[]);
                                } else if let Some(prolog_query) = PrologEngine::generate_prolog_query(&query_type, &entities) {
                                    println!("Prolog查询: {}", prolog_query);
                                    
                                    let results = prolog_engine.query(&prolog_query);
                                    
                                    let response = response_formatter.format_prolog_results(&results, &query_type);
                                    println!("{}", response);
                                    log_chronicle_reply(&mut chronicle, &response, &[], &[]);
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
                    log_chronicle_reply(&mut chronicle, &answer, &[], &[]);
                } else {
                    println!("{}", config.messages.cannot_understand);
                    
                    // 尝试使用外部知识API
                    if let Some(direct_answer) = get_direct_answer(input) {
                        println!("根据常识知识库：{}", direct_answer);
                        log_chronicle_reply(
                            &mut chronicle,
                            &format!("根据常识知识库：{}", direct_answer),
                            &[],
                            &[],
                        );
                    } else {
                        // 尝试使用外链知识（名册优先，未命中才打网，回答带出处）
                        match external_knowledge_service
                            .answer_question_with_entities(
                                input,
                                &external_candidates(&[], &precomputed_query_entities),
                            )
                            .await
                        {
                            Ok(outcome) => match external_outcome_text(&outcome) {
                                Some(text) => {
                                    println!("{}", text);
                                    log_chronicle_reply(&mut chronicle, &text, &[], &[]);
                                }
                                None => {
                                    let fallback =
                                        "很抱歉，我无法理解您的问题。请尝试重新描述。";
                                    println!("{}", fallback);
                                    log_chronicle_reply(&mut chronicle, fallback, &[], &[]);
                                }
                            },
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

    // 外链名册落盘：跨重启保留取回的外部知识（出处 + 逻辑时刻）
    match external_knowledge_service.save_ledger(&external_ledger_path) {
        Ok(_) => println!(
            "外链名册已保存到 {}（{} 条条目）。",
            external_ledger_path,
            external_knowledge_service.ledger_len()
        ),
        Err(e) => eprintln!("保存外链名册失败: {}", e),
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
                output.push_str(
                    "■ 派生账本 kg:graphs/derived：尚未物化过（用「物化推理」写入）。\n",
                );
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
                if ledger.revision == current_revision && ledger.fingerprint == current_fingerprint {
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
        if answer.contains("没有记录") || answer.contains("没有相关记录") || answer.contains("数量为 0")
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
            return Some(combined);
        }
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
    // 逻辑问答失败时，优先用 OWL+DCG 话语引擎做连贯叙述，
    // 再回退到"知识图谱即语言模型"的统计生成式介绍
    let entity = symbolic_reasoner.best_linked_entity(input)?;
    symbolic_reasoner
        .narrate(&entity)
        .or_else(|| symbolic_reasoner.generate_about(&entity))
}

fn contains_question_marker(text: &str) -> bool {
    // 疑问词识别走虚词算符表（含同形守卫）
    know_prolog::function_words::contains_question_marker(text)
}

/// 判断输入是否以系统命令开头：澄清进行中时遇到命令应让位给常规处理
fn is_command_like(input: &str) -> bool {
    const COMMANDS: [&str; 134] = [
        "全息",
        "对话本体",
        "对话状态",
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
        "提及账本",
        "识别实体",
        "判断",
        "蕴含",
        "校验",
        "形式化",
        "相似实体",
        "语义相似",
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
        "忘记",
        "遗忘",
        "撤回",
        "撤销导入",
        "导入回执",
        "导入历史",
        "导入日志",
        "撤销历史",
        "外链",
        "同义链接",
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
        "时间线",
        "对话时间线",
        "证明图",
        "证明树",
        "画证明",
        "知识子图",
        "子图",
        "邻域",
        "覆盖报告",
        "质量报告",
        "覆盖检查",
        "虚词表",
        "功能词表",
        "虚词报告",
        "算符表",
    ];
    COMMANDS.iter().any(|command| input.starts_with(command))
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

/// 解析「左 = 右」形式的谓词登记参数（支持 = / ＝ / =>）
fn parse_predicate_pair(spec: &str) -> Option<(String, String)> {
    let spec = spec.trim();
    for separator in ["=>", "=", "＝"] {
        if let Some((left, right)) = spec.split_once(separator) {
            let left = left.trim();
            let right = right.trim();
            if !left.is_empty() && !right.is_empty() {
                return Some((left.to_string(), right.to_string()));
            }
        }
    }
    None
}

/// 中文句子 → 关系三元组：只取确定事实、二元、无占位符的原子，
/// 跳过「是」断言（类型断言不落成关系）。
/// 两条抽取通道合并去重：
///   1. 模式解析器 `parse_lessons`（「记住：」同款）；
///   2. DCG 文法抽取（被字句/把字句/并列/所有格换向/词库归一）。
fn extract_relationships_from_text(
    text: &str,
    words: &[know_prolog::nlp::TaggedWord],
    grammar_extractor: &GrammarExtractor,
) -> Vec<KnowledgeItem> {
    use know_prolog::chinese_reasoning_nlu::{parse_lessons, LessonKind};
    let outcome = parse_lessons(text);
    let mut seen: std::collections::BTreeSet<(String, String, String)> =
        std::collections::BTreeSet::new();
    let mut items = Vec::new();
    let mut push_triple = |from: String, predicate: String, to: String| {
        if predicate == "是" || from.is_empty() || to.is_empty() || from == to {
            return;
        }
        let key = (from.clone(), predicate.clone(), to.clone());
        if !seen.insert(key) {
            return;
        }
        items.push(KnowledgeItem::Relationship {
            from_label: String::new(),
            from_id: from,
            to_label: String::new(),
            to_id: to,
            rel_type: predicate,
            properties: std::collections::HashMap::new(),
        });
    };

    for lesson in outcome.lessons {
        if lesson.kind != LessonKind::DeterministicFact || !lesson.head.is_ground() {
            continue;
        }
        let terms = &lesson.head.terms;
        if terms.len() != 2 {
            continue;
        }
        push_triple(
            terms[0].name().trim().to_string(),
            lesson.head.predicate.to_string(),
            terms[1].name().trim().to_string(),
        );
    }

    for triple in grammar_extractor.extract_from_tagged(words) {
        push_triple(triple.subject, triple.predicate, triple.object);
    }

    items
}

/// 把谓词画像注册表的词面同步到符号推理器：
/// 正向词面（别名 → 槽位）走别名通道（对齐同义词 + 生成词面表），
/// 逆向词面只登记查询换向（不改生成叙事词面，避免叙事错向）。
fn sync_predicate_pairs(reasoner: &mut SymbolicReasoner, oxigraph_manager: &OxigraphManager) {
    for (surface, slot, inverted) in oxigraph_manager.predicate_query_pairs() {
        if surface == slot {
            continue;
        } else if inverted {
            reasoner.register_predicate_query_surface(&surface, &slot);
        } else {
            reasoner.register_predicate_alias(&surface, &slot);
        }
    }
}

/// 把全息命令参数切成若干词项：支持空白/中英文逗号/冒号分隔
fn parse_holo_terms(target: &str, expected: usize) -> Option<Vec<String>> {
    let terms: Vec<String> = target
        .split(|ch: char| {
            ch.is_whitespace() || matches!(ch, '，' | ',' | '：' | ':' | '、')
        })
        .filter(|term| !term.is_empty())
        .map(|term| term.to_string())
        .collect();
    if terms.len() == expected {
        Some(terms)
    } else {
        None
    }
}

/// 解析全息检索的两元组：支持「主体 谓词」与「主语的谓词」两种写法
fn parse_holo_pair(target: &str) -> Option<(String, String)> {
    if let Some(terms) = parse_holo_terms(target, 2) {
        return Some((terms[0].clone(), terms[1].clone()));
    }
    let text = target.trim();
    if let Some(index) = text.find('的') {
        let subject = text[..index].trim();
        let predicate = text[index + '的'.len_utf8()..].trim();
        if !subject.is_empty() && !predicate.is_empty() {
            return Some((subject.to_string(), predicate.to_string()));
        }
    }
    None
}

/// 渲染全息召回候选（软召回分数 + 硬核查标记）
fn render_holo_hits(title: &str, hits: &[HoloHit]) -> String {
    if hits.is_empty() {
        return format!(
            "{}：全息图未召回候选（实体/谓词未登记，或该结构在知识中不存在）。",
            title
        );
    }
    let verified = hits.iter().filter(|hit| hit.verified).count();
    let mut out = format!(
        "{}（候选 {} 个，通过三元组核查 {} 个）：",
        title,
        hits.len(),
        verified
    );
    for (index, hit) in hits.iter().enumerate() {
        let mark = if hit.verified {
            "已核查✓"
        } else {
            "未登记（串扰/推测，不入结论）"
        };
        out.push_str(&format!(
            "\n  {}. {}  余弦 {:.3}  [{}]",
            index + 1,
            hit.symbol,
            hit.score,
            mark
        ));
    }
    out
}

/// 渲染全息类比候选（关系迁移路径 + 核查标记）
fn render_holo_analogy(a: &str, b: &str, c: &str, hits: &[HoloAnalogyHit]) -> String {
    if hits.is_empty() {
        return format!(
            "全息类比 {}:{} :: {}:? 未产生候选（需先登记一组对应事实，关系才能被解缠迁移）。",
            a, b, c
        );
    }
    let mut out = format!("全息类比 {}:{} :: {}:?（关系解缠 → 迁移绑定 → 净噪）：", a, b, c);
    for (index, hit) in hits.iter().enumerate() {
        let via: Vec<String> = hit
            .via
            .iter()
            .map(|(predicate, score)| format!("{}（{:.2}）", predicate, score))
            .collect();
        let mark = if hit.verified {
            "已核查✓"
        } else {
            "未登记（全息推测，供参考）"
        };
        out.push_str(&format!(
            "\n  {}. {}  综合分 {:.3}  关系通道：{}  [{}]",
            index + 1,
            hit.symbol,
            hit.score,
            via.join("、"),
            mark
        ));
    }
    out
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
                        match external_knowledge_service
                            .answer_question_with_entities(input, &query_entities)
                            .await
                        {
                            Ok(outcome) => match external_outcome_text(&outcome) {
                                Some(text) => println!("{}", text),
                                None => println!("很抱歉，我无法找到相关信息。"),
                            },
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

#[cfg(test)]
mod session_tests {
    use super::*;
    use std::collections::HashMap;

    /// 新命令必须被 is_command_like 识别，避免澄清追问时被吞掉
    #[test]
    fn command_like_covers_journal_and_perf_commands() {
        for command in [
            "撤销导入",
            "导入回执",
            "导入历史",
            "导入日志",
            "性能报告",
            "性能台账",
            "派生台账",
            "增量报告",
            "图账本",
            "规则账本",
            "规则图账本",
            "派生图账本",
            "物化推理",
            "推理物化",
            "物化派生",
            "忘记规则：如果X的师傅是Y，那么X的老师是Y",
            "撤回规则：如果X的师傅是Y，那么X的老师是Y",
            "虚词表",
            "功能词表",
            "虚词报告",
            "算符表",
            "提及账本",
            "外链台账",
            "外链查看：法国",
            "外链刷新：法国",
            "外链删除：法国",
            "同义链接：姚明 = ownthink:姚明",
            "帧记忆",
            "话语帧",
            "挂账",
            "默认推理：麻雀会飞",
            "默认逻辑：",
        ] {
            assert!(is_command_like(command), "{} 应被识别为系统命令", command);
        }
    }

    /// 「同义链接」命令的解析：半角/全角等号、缺端拒绝
    #[test]
    fn alias_link_parses_both_equals_forms() {
        assert_eq!(
            parse_alias_link("姚明 = ownthink:姚明"),
            Some(("姚明", "ownthink:姚明"))
        );
        assert_eq!(
            parse_alias_link("姚明＝ownthink:姚明"),
            Some(("姚明", "ownthink:姚明"))
        );
        assert_eq!(parse_alias_link("只有一头 = "), None);
        assert_eq!(parse_alias_link("没有等号"), None);
    }

    /// 外链候选实体：预计算在前、去重保序
    #[test]
    fn external_candidates_merge_dedup_and_order() {
        let entities = vec!["B".to_string(), "D".to_string()];
        let precomputed = vec!["A".to_string(), "B".to_string()];
        assert_eq!(
            external_candidates(&entities, &precomputed),
            vec!["A", "B", "D"]
        );
    }

    /// 跨重启撤销：导入 → 日志落盘 → 重新打开存储 → 恢复日志 → 撤销归零
    #[tokio::test]
    async fn journal_survives_restart_and_undoes_import() {
        let dir = tempfile::TempDir::new().unwrap();
        let store_path = dir.path().to_str().unwrap().to_string();
        let journal_path = format!("{}/import_journal.json", store_path);

        {
            let manager = OxigraphManager::with_perf_ledger(
                &store_path,
                PerfLedger::shared(),
            )
            .unwrap();
            let items = vec![KnowledgeItem::Relationship {
                from_label: "人物".to_string(),
                from_id: "张三".to_string(),
                to_label: "人物".to_string(),
                to_id: "李四".to_string(),
                rel_type: "师傅".to_string(),
                properties: HashMap::new(),
            }];
            let receipt = manager.save_knowledge_with_receipt(items).await.unwrap();
            assert!(receipt.can_undo());

            let mut journal = ImportJournal::new();
            journal.push(receipt.clone());
            journal.save(&journal_path).unwrap();
            let snapshot = manager.load_graph_snapshot().await.unwrap();
            assert!(!snapshot.is_empty());
            receipt
        };

        // 模拟进程重启：新开存储 + 从信封恢复日志
        let manager =
            OxigraphManager::with_perf_ledger(&store_path, PerfLedger::shared()).unwrap();
        let JournalLoad::Loaded(mut journal) = ImportJournal::load(&journal_path) else {
            panic!("应能从 ContextImage 信封恢复导入日志");
        };
        let restored = journal.pop().expect("栈顶应有一笔可撤销导入");
        let removed = manager.undo_import(&restored).await.unwrap();
        assert_eq!(removed, restored.actions.len());
        assert_eq!(journal.len(), 0);
        let snapshot = manager.load_graph_snapshot().await.unwrap();
        assert!(
            snapshot.is_empty(),
            "撤销后图谱应回到导入前状态，实际剩余 {} 条",
            snapshot.len()
        );
    }

    /// 帧澄清粘合层：命令/无关输入放行且零状态变化；序号绑定后返回重放问句
    #[test]
    fn frame_clarification_step_binds_replays_and_passes_through() {
        let mut yao = HashMap::new();
        yao.insert("身高".to_string(), "226cm".to_string());
        yao.insert("体重".to_string(), "140kg".to_string());
        let mut ye = HashMap::new();
        ye.insert("身高".to_string(), "190cm".to_string());
        ye.insert("体重".to_string(), "85kg".to_string());
        let facts = vec![
            PrologFact::Entity {
                name: "姚明".to_string(),
                entity_type: "人".to_string(),
                properties: yao,
            },
            PrologFact::Entity {
                name: "叶莉".to_string(),
                entity_type: "人".to_string(),
                properties: ye,
            },
        ];
        let mut reasoner =
            SymbolicReasoner::new(know_prolog::ReasoningConfig::default(), HashMap::new());
        reasoner.observe_facts(&facts);

        // 无源可填的洞 → 澄清（候选人名序：叶莉、姚明）
        reasoner.contextualize("体重是多少");
        let prompt = reasoner
            .start_frame_clarification("体重是多少")
            .expect("应开启帧澄清");
        assert!(prompt.contains("叶莉") && prompt.contains("姚明"), "{}", prompt);

        // 命令让位：放行给常规命令通道，澄清保持
        assert!(matches!(
            handle_frame_clarification(&mut reasoner, "帧记忆"),
            FrameClarificationStep::Pass
        ));
        // 非回答自然语言：放行，澄清保持
        assert!(matches!(
            handle_frame_clarification(&mut reasoner, "北京是哪个省的"),
            FrameClarificationStep::Pass
        ));
        assert!(reasoner.frame_clarification_active());

        // 序号短答：确定性绑定 → 返回重放问句（将由主循环走常规问答）
        match handle_frame_clarification(&mut reasoner, "1") {
            FrameClarificationStep::Replay(question) => {
                assert_eq!(question, "叶莉的体重是什么");
            }
            _ => panic!("序号应绑定并重放"),
        }
        assert!(!reasoner.frame_clarification_active(), "绑定后澄清应关闭");
        let answer = reasoner.answer("叶莉的体重是什么").expect("重放应可作答");
        assert!(answer.contains("85kg"), "{}", answer);
    }
}

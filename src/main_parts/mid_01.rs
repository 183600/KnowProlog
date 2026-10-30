
/// 默认帧诊断语料：仓库根目录随附的文本知识库（按文件名排序，缺失即跳过）。
fn load_frame_corpus() -> Result<String, String> {
    const FILES: &[&str] = &[
        "additional_knowledge.txt",
        "basic_knowledge.txt",
        "comprehensive_knowledge_expansion.txt",
        "knowledge_base.txt",
        "knowledge_base_expanded.txt",
    ];
    let mut material = String::new();
    let mut used = 0usize;
    for path in FILES {
        match std::fs::read_to_string(path) {
            Ok(content) => {
                material.push_str(&content);
                if !material.ends_with('\n') {
                    material.push('\n');
                }
                used += 1;
            }
            Err(_) => continue,
        }
    }
    if used == 0 {
        return Err("未找到默认语料；请用 --text 或 --file 指定要诊断的文本".to_string());
    }
    Ok(material)
}

/// `eval-nlu`：对话 NLU 金标评测 + 回归棘轮（不依赖知识存储）。
fn run_eval_nlu(
    knowledge: &str,
    prompt_mode: &str,
    baseline: Option<String>,
    update_baseline: bool,
    allow_regression: bool,
    json: bool,
) {
    let mode = match knowledge {
        "scripted" => EvalKnowledge::Scripted,
        "reasoner" => EvalKnowledge::Reasoner,
        other => {
            eprintln!("未知的知识后端「{}」：可选 reasoner / scripted", other);
            std::process::exit(1);
        }
    };
    let prompt_mode = parse_prompt_mode(prompt_mode);
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

/// `eval-link`：会话级实体链接金标评测 + 回归棘轮（不依赖知识存储）。
fn run_eval_construction(json: bool, strict: bool) {
    let report = know_prolog::constructive_eval::evaluate();
    if json {
        println!("{}", report.json());
    } else {
        println!("{}", report.render_text());
    }
    if strict
        && (report.metrics.recall_bps() < 10_000
            || report.metrics.precision_bps() < 10_000
            || report.metrics.wrong_target_facts > 0
            || !report.metrics.support_audit_clean)
    {
        std::process::exit(1);
    }
}

fn run_eval_link(
    baseline: Option<String>,
    update_baseline: bool,
    allow_regression: bool,
    json: bool,
) {
    let report = know_prolog::link_eval::evaluate_seed();
    if json {
        match serde_json::to_string_pretty(&report) {
            Ok(text) => println!("{}", text),
            Err(e) => {
                eprintln!("链接评测序列化失败: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        println!("{}", report.render_text());
    }

    let baseline_path = baseline
        .clone()
        .unwrap_or_else(|| "link_eval_baseline.json".to_string());
    if update_baseline {
        let new_baseline = know_prolog::link_eval::LinkEvalBaseline::from_report(&report);
        match new_baseline.save(&baseline_path) {
            Ok(()) => println!("链接基线已写入 {}", baseline_path),
            Err(e) => {
                eprintln!("基线写入失败: {}", e);
                std::process::exit(1);
            }
        }
    } else if baseline.is_some() {
        match know_prolog::link_eval::LinkEvalBaseline::load(&baseline_path) {
            Ok(old_baseline) => {
                let verdict =
                    know_prolog::link_eval::compare_baseline(&old_baseline, &report, 1e-6);
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

/// `eval-grounding`：答案探测式指称消解金标评测 + 回归棘轮。
fn run_eval_grounding(
    baseline: Option<String>,
    update_baseline: bool,
    allow_regression: bool,
    json: bool,
) {
    let report = reference_probe::evaluate_seed();
    if json {
        match serde_json::to_string_pretty(&report) {
            Ok(text) => println!("{}", text),
            Err(e) => {
                eprintln!("接地评测序列化失败: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        println!("{}", report.render_text());
    }

    let baseline_path = baseline
        .clone()
        .unwrap_or_else(|| "grounding_eval_baseline.json".to_string());
    if update_baseline {
        let new_baseline = reference_probe::GroundingEvalBaseline::from_report(&report);
        match new_baseline.save(&baseline_path) {
            Ok(()) => println!("接地基线已写入 {}", baseline_path),
            Err(e) => {
                eprintln!("基线写入失败: {}", e);
                std::process::exit(1);
            }
        }
    } else if baseline.is_some() {
        match reference_probe::GroundingEvalBaseline::load(&baseline_path) {
            Ok(old_baseline) => {
                let verdict = reference_probe::compare_baseline(&old_baseline, &report, 1e-6);
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

/// `eval-identity`：身份作用域 / 身份卡 / 会话台账金标评测 + 回归棘轮。
fn run_eval_identity(
    baseline: Option<String>,
    update_baseline: bool,
    allow_regression: bool,
    json: bool,
) {
    let report = evaluate_identity();
    if json {
        match serde_json::to_string_pretty(&report) {
            Ok(text) => println!("{}", text),
            Err(e) => {
                eprintln!("身份评测序列化失败: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        print!("{}", report.render_text());
    }

    let baseline_path = baseline
        .clone()
        .unwrap_or_else(|| "identity_eval_baseline.json".to_string());
    if update_baseline {
        let new_baseline = IdentityEvalBaseline::from_report(&report);
        match new_baseline.save(&baseline_path) {
            Ok(()) => println!("身份基线已写入 {}", baseline_path),
            Err(e) => {
                eprintln!("基线写入失败: {}", e);
                std::process::exit(1);
            }
        }
    } else if baseline.is_some() {
        match IdentityEvalBaseline::load(&baseline_path) {
            Ok(old_baseline) => {
                let verdict = compare_identity_baseline(&old_baseline, &report, 1e-6);
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

/// `eval-recovery`：失败恢复金标评测 + 回归棘轮（不依赖知识存储）。
fn run_eval_recovery(
    baseline: Option<String>,
    update_baseline: bool,
    allow_regression: bool,
    json: bool,
) {
    let report = know_prolog::eval_recovery::evaluate_seed();
    if json {
        match serde_json::to_string_pretty(&report) {
            Ok(text) => println!("{}", text),
            Err(e) => {
                eprintln!("恢复评测序列化失败: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        println!("{}", report.render_text());
    }

    let baseline_path = baseline
        .clone()
        .unwrap_or_else(|| "recovery_baseline.json".to_string());
    if update_baseline {
        let new_baseline = know_prolog::eval_recovery::EvalRecoveryBaseline::from_report(&report);
        match new_baseline.save(&baseline_path) {
            Ok(()) => println!("恢复基线已写入 {}", baseline_path),
            Err(e) => {
                eprintln!("基线写入失败: {}", e);
                std::process::exit(1);
            }
        }
    } else if baseline.is_some() {
        match know_prolog::eval_recovery::EvalRecoveryBaseline::load(&baseline_path) {
            Ok(old_baseline) => {
                let verdict =
                    know_prolog::eval_recovery::compare_baseline(&old_baseline, &report, 1e-6);
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

/// `proof-bench`：证明义务 / 反事实孪生 / 证距评测 + 回归棘轮（不依赖知识存储）。
fn run_proof_bench(
    baseline: Option<String>,
    update_baseline: bool,
    allow_regression: bool,
    emit_prolog: Option<&str>,
    json: bool,
) {
    let report = proof_bench::evaluate_seed();
    if let Some(path) = emit_prolog {
        let corpus = proof_bench::render_prolog(proof_bench::seed_corpus());
        match std::fs::write(path, corpus) {
            Ok(()) => {
                if !json {
                    println!("可审计 Prolog 语料已写入 {}", path);
                }
            }
            Err(error) => {
                eprintln!("Prolog 语料写入失败: {}", error);
                std::process::exit(1);
            }
        }
    }
    if json {
        match serde_json::to_string_pretty(&report) {
            Ok(text) => println!("{}", text),
            Err(error) => {
                eprintln!("ProofBench 报告序列化失败: {}", error);
                std::process::exit(1);
            }
        }
    } else {
        print!("{}", report.render_text());
    }

    let baseline_path = baseline
        .clone()
        .unwrap_or_else(|| "proof_bench_baseline.json".to_string());
    if update_baseline {
        let new_baseline = ProofBenchBaseline::from_report(&report);
        match new_baseline.save(&baseline_path) {
            Ok(()) => println!("ProofBench 基线已写入 {}", baseline_path),
            Err(error) => {
                eprintln!("基线写入失败: {}", error);
                std::process::exit(1);
            }
        }
    } else if baseline.is_some() {
        match ProofBenchBaseline::load(&baseline_path) {
            Ok(old_baseline) => {
                let verdict = proof_bench::compare_baseline(&old_baseline, &report, 1e-6);
                print!("{}", verdict.render_text());
                if !verdict.passed && !allow_regression {
                    std::process::exit(1);
                }
            }
            Err(error) => {
                eprintln!("基线读取失败: {}", error);
                std::process::exit(1);
            }
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

/// 解析 `--recovery`：off / observe / on，非法值立即退出。
fn parse_recovery_mode(value: &str) -> RecoveryMode {
    RecoveryMode::parse(value).unwrap_or_else(|| {
        eprintln!("未知的失败恢复模式「{}」：可选 off / observe / on", value);
        std::process::exit(1);
    })
}

fn parse_trace_mode(value: &str) -> TraceMode {
    TraceMode::parse(value).unwrap_or_else(|| {
        eprintln!("未知的分层追踪模式「{}」：可选 off / observe / on", value);
        std::process::exit(1);
    })
}

/// `--store` 覆盖存储时，把相对的对话状态路径重定位到新存储根目录。
///
/// 配置默认写成 `oxigraph_data/dialogue_state.json`；若原样保留，独立
/// 存储下的会话会与主库共享记忆（焦点、槽账本、改口史全部串台）。绝对
/// 路径视为用户显式指定，保持不动。
fn isolate_dialogue_state_path(path: &mut Option<String>, store_path: &str) {
    let Some(current) = path.as_deref() else {
        return;
    };
    if current.trim().is_empty() || std::path::Path::new(current).is_absolute() {
        return;
    }
    let file = std::path::Path::new(current)
        .file_name()
        .unwrap_or_else(|| std::ffi::OsStr::new("dialogue_state.json"));
    *path = Some(
        std::path::Path::new(store_path)
            .join(file)
            .to_string_lossy()
            .into_owned(),
    );
}

/// `dialogue-artifacts`：读取编年史并输出稳定工件 URI；不打开知识存储。
fn run_dialogue_artifacts(
    config: &know_prolog::SystemConfig,
    store_path: &str,
    selector: Option<&str>,
    chronicle_override: Option<&str>,
    session_override: Option<&str>,
) {
    let chronicle_path = chronicle_override
        .map(str::to_string)
        .or_else(|| {
            config
                .reasoning
                .dialogue_state_path
                .as_deref()
                .map(|path| format!("{}.chronicle.jsonl", path))
        })
        .unwrap_or_else(|| format!("{}/dialogue.chronicle.jsonl", store_path));
    let session_path = session_override.map(str::to_string).unwrap_or_else(|| {
        format!(
            "{}{}",
            chronicle_path,
            know_prolog::dialogue_uri::SESSION_FILE_SUFFIX
        )
    });
    let session = match DialogueSession::open_or_create(&session_path) {
        Ok(session) => session,
        Err(error) => {
            eprintln!("对话工件会话标识打开失败（{}）：{}", session_path, error);
            std::process::exit(1);
        }
    };
    let (chronicle, audit) = match Chronicle::open(&chronicle_path, FsyncPolicy::None) {
        Ok((chronicle, audit)) => (chronicle, audit),
        Err(error) => {
            eprintln!("对话编年史打开失败（{}）：{}", chronicle_path, error);
            std::process::exit(1);
        }
    };
    if !audit.is_clean() {
        eprintln!("对话编年史{}", audit.verdict());
    }
    let index = DialogueArtifactIndex::from_chronicle(&session, &chronicle);
    match selector {
        Some(selector) => println!("{}", index.render_selector(selector)),
        None => println!("{}", index.render_report()),
    }
}

fn run_transduce(
    config: &know_prolog::SystemConfig,
    text: Option<&str>,
    echo: Option<&str>,
    kind: Option<&str>,
    json: bool,
) {
    use know_prolog::lexical_transducer::{
        EchoLedger, LexicalTransducer, TransducerOptions,
    };
    use know_prolog::mention_lexicon::MentionLexicon;
    use know_prolog::synonym_graph::{SynonymContext, SynonymGraph, SynonymKind};
    use know_prolog::type_system::TypeSystem;

    let graph = SynonymGraph::from_config(config, 1);
    let mut lexicon = MentionLexicon::new();
    for profile in &config.entity_profiles {
        let etype = profile.etype.first().map(String::as_str);
        lexicon.confirm(&profile.name, &profile.name, etype, "config");
        for alias in &profile.aliases {
            lexicon.confirm(alias, &profile.name, etype, "entity_profile");
        }
    }
    let flat = config
        .synonyms
        .as_ref()
        .map(|synonyms| synonyms.mappings.clone())
        .unwrap_or_default();
    let transducer =
        LexicalTransducer::compile(&graph, &lexicon, &flat, TransducerOptions::default());
    let types = TypeSystem::from_config(&config.type_system);
    let context = SynonymContext::new();
    let kind_filter = match kind {
        Some(value) => match SynonymKind::parse(value) {
            Some(kind) => Some(kind),
            None => {
                eprintln!("未知词法层「{}」：可选 entity / predicate / type", value);
                std::process::exit(1);
            }
        },
        None => None,
    };

    let mut inputs: Vec<String> = Vec::new();
    match text {
        Some("-") | None => {
            use std::io::BufRead;
            let stdin = std::io::stdin();
            for line in stdin.lock().lines() {
                match line {
                    Ok(line) => {
                        if !line.trim().is_empty() {
                            inputs.push(line);
                        }
                    }
                    Err(error) => {
                        eprintln!("读取标准输入失败：{}", error);
                        std::process::exit(1);
                    }
                }
            }
        }
        Some(text) => inputs.push(text.to_string()),
    }
    if inputs.is_empty() && echo.is_none() {
        eprintln!("没有可转导的文本；请给出文本参数或通过标准输入提供。");
        std::process::exit(1);
    }

    let mut ledger = EchoLedger::new();
    let mut results = Vec::new();
    for (index, input) in inputs.iter().enumerate() {
        ledger.begin_turn(index as u64 + 1);
        let transduction =
            transducer.transduce_with_kind(input, &context, kind_filter, |left, right| {
                types.types_compatible(left, right)
            });
        ledger.observe_all(&transduction);
        let echo_choice = echo.map(|canonical| {
            let kind = graph
                .entries()
                .find(|entry| {
                    entry.is_confirmed()
                        && entry.canonical == canonical
                        && entry.kind == SynonymKind::Entity
                })
                .map(|entry| entry.kind)
                .unwrap_or(SynonymKind::Entity);
            ledger.choose(&graph, canonical, kind, &context)
        });
        if json {
            results.push(serde_json::json!({
                "text": input,
                "transduction": transduction,
                "echo": echo_choice,
            }));
        } else {
            println!("{}", transduction.render_report());
            if let Some(choice) = &echo_choice {
                println!(
                    "回声：「{}」→「{}」（{}，可逆：{}）",
                    choice.canonical, choice.surface, choice.source, choice.reversible
                );
            }
            if index + 1 < inputs.len() {
                println!();
            }
        }
    }
    if json {
        match serde_json::to_string_pretty(&results) {
            Ok(output) => println!("{}", output),
            Err(error) => {
                eprintln!("转导结果序列化失败：{}", error);
                std::process::exit(1);
            }
        }
    }
}

/// `artifacts`：工件审计（知识版 `rustfmt --check`）。
///
/// 扫描已知持久化工件，按「解码 → 规范重编码 → 与磁盘比对」分类；
/// `--fix` 原子重写非规范工件（损坏工件一律不触碰）。存在损坏或未修好的
/// 非规范工件时以非零码退出，可直接用于 CI 巡检。
fn run_artifact_audit(
    config: &know_prolog::SystemConfig,
    store_path: &str,
    fix: bool,
    json: bool,
    extra_paths: &[String],
) {
    use know_prolog::artifact_audit::{self, ArtifactKind};

    let mut targets: std::collections::BTreeMap<String, bool> = std::collections::BTreeMap::new();
    for path in extra_paths {
        targets.insert(path.clone(), true);
    }
    let mut defaults = vec![
        format!("{}/import_journal.json", store_path),
        format!("{}/external_ledger.json", store_path),
        format!("{}/predicate_registry.json", store_path),
        format!("{}/entity_registry.json", store_path),
        format!("{}/eval_baseline.json", store_path),
        format!("{}/nlu_eval_baseline.json", store_path),
        format!("{}/recovery_baseline.json", store_path),
        format!("{}/quality_baseline.json", store_path),
    ];
    if let Some(state_path) = config.reasoning.dialogue_state_path.as_deref() {
        defaults.push(state_path.to_string());
        defaults.push(format!("{}.manager.json", state_path));
        defaults.push(format!("{}.chronicle.jsonl", state_path));
        defaults.push(format!("{}.chronicle.jsonl.session.json", state_path));
    } else {
        defaults.push(format!("{}/dialogue.chronicle.jsonl", store_path));
        defaults.push(format!(
            "{}/dialogue.chronicle.jsonl.session.json",
            store_path
        ));
    }
    for path in defaults {
        targets.entry(path).or_insert(false);
    }

    let mut audits = Vec::new();
    for (path, explicit) in &targets {
        let kind = ArtifactKind::infer(path);
        let audit = if fix {
            match artifact_audit::fix(path, kind) {
                Ok(audit) => audit,
                Err(error) => {
                    eprintln!("工件修复被拒：{}", error);
                    artifact_audit::audit(path, kind)
                }
            }
        } else {
            artifact_audit::audit(path, kind)
        };
        if audit.status == artifact_audit::ArtifactStatus::Missing && !explicit {
            continue;
        }
        audits.push(audit);
    }

    let (mut canonical, mut non_canonical, mut repairable, mut damaged, mut missing) =
        (0u64, 0u64, 0u64, 0u64, 0u64);
    for audit in &audits {
        match audit.status {
            artifact_audit::ArtifactStatus::Canonical => canonical += 1,
            artifact_audit::ArtifactStatus::NonCanonical => non_canonical += 1,
            artifact_audit::ArtifactStatus::Repairable(_) => repairable += 1,
            artifact_audit::ArtifactStatus::Damaged(_) => damaged += 1,
            artifact_audit::ArtifactStatus::Missing => missing += 1,
        }
    }

    if json {
        let value = serde_json::json!({
            "object": "artifacts.audit",
            "fix": fix,
            "total": audits.len(),
            "canonical": canonical,
            "non_canonical": non_canonical,
            "repairable": repairable,
            "damaged": damaged,
            "missing": missing,
            "artifacts": audits
                .iter()
                .map(|audit| serde_json::json!({
                    "path": audit.path,
                    "kind": audit.kind.label(),
                    "status": audit.status.label(),
                    "detail": audit.status.detail(),
                }))
                .collect::<Vec<_>>(),
        });
        match serde_json::to_string_pretty(&value) {
            Ok(text) => println!("{}", text),
            Err(e) => {
                eprintln!("工件审计序列化失败: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        println!(
            "工件审计：共 {} 件（规范 {}，非规范 {}，可截尾 {}，损坏 {}，缺失 {}）。",
            audits.len(),
            canonical,
            non_canonical,
            repairable,
            damaged,
            missing
        );
        for audit in &audits {
            let detail = audit.status.detail();
            if detail.is_empty() {
                println!(
                    "- {} [{}] {}",
                    audit.path,
                    audit.kind.label(),
                    audit.status.label()
                );
            } else {
                println!(
                    "- {} [{}] {} —— {}",
                    audit.path,
                    audit.kind.label(),
                    audit.status.label(),
                    detail
                );
            }
        }
        if fix {
            println!(
                "提示：--fix 只原子重写非规范工件、只截编年史的损坏尾行；其余损坏工件一律不触碰（宁可不修，也不误改）。"
            );
        }
    }

    if damaged > 0 || non_canonical > 0 || repairable > 0 {
        std::process::exit(1);
    }
}

/// 配置声明的功能谓词（融合冲突判定的默认策略来源）。
fn functional_predicates(config: &know_prolog::SystemConfig) -> Vec<String> {
    config
        .predicate_profiles
        .iter()
        .filter(|profile| profile.functional)
        .map(|profile| profile.name.clone())
        .collect()
}

/// `namespace`：把 RDF 文件导入命名空间源图（默认图不动）。
async fn run_namespace_import(
    manager: &OxigraphManager,
    source: &str,
    label: Option<&str>,
    trust: &str,
    kind: &str,
    file_path: &str,
    json: bool,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let trust = TrustLevel::parse(trust).ok_or_else(|| {
        format!(
            "未知信任层级：{}（可用 verified / curated / imported / external / speculative）",
            trust
        )
    })?;
    let kind = NamespaceKind::parse(kind).ok_or_else(|| {
        format!(
            "未知命名空间种类：{}（可用 global / import / session / hypothesis）",
            kind
        )
    })?;
    let format = OxigraphManager::detect_rdf_format(file_path)?;
    let file = std::fs::File::open(file_path)?;
    let parser = oxigraph::io::RdfParser::from_format(format).for_reader(file);

    let mut assertions = Vec::new();
    let mut skipped = 0usize;
    for quad in parser {
        let quad = quad.map_err(|error| -> Box<dyn std::error::Error + Send + Sync> {
            format!("解析 RDF 失败：{}", error).into()
        })?;
        match OxigraphManager::assertion_from_rdf_quad(&quad, source, file_path) {
            Some(assertion) => assertions.push(assertion),
            None => skipped += 1,
        }
    }

    let namespace = NamespaceRecord::new(source, label.unwrap_or(source), kind, trust);
    let added = manager
        .record_namespace_assertions(&namespace, &assertions)
        .await?;

    if json {
        let report = serde_json::json!({
            "namespace": namespace.id,
            "label": namespace.label,
            "trust": namespace.trust.label_zh(),
            "kind": namespace.kind.label_zh(),
            "graph": namespace.graph_iri(),
            "parsed": assertions.len(),
            "added": added,
            "skipped": skipped,
        });
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!(
            "命名空间 {}（{}，{}，{}）",
            namespace.id,
            namespace.label,
            namespace.kind.label_zh(),
            namespace.trust.label_zh()
        );
        println!("  源图：{}", namespace.graph_iri());
        println!(
            "  解析 {} 条断言（跳过 {}），新增源图四元组 {}",
            assertions.len(),
            skipped,
            added
        );
        println!("  默认图未改动；用 `fuse --apply` 物化融合视图");
    }
    Ok(())
}

/// `namespace --list`：列出全部命名空间与断言计数。
async fn run_namespace_list(
    manager: &OxigraphManager,
    json: bool,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let ledger = manager.load_fusion_ledger().await?;
    if json {
        let summaries: Vec<_> = viewpoint::LensCorpus::new(&[], &ledger)
            .summaries()
            .into_iter()
            .filter(|summary| !summary.context.is_default())
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "namespaces": summaries,
            }))?
        );
    } else {
        println!("{}", ledger.render_zh());
    }
    Ok(())
}

/// `namespace --status`：启用 / 隔离 / 退役一个命名空间。
async fn run_namespace_status(
    manager: &OxigraphManager,
    source: &str,
    status: &str,
    json: bool,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let status = NamespaceStatus::parse(status).ok_or_else(|| {
        format!(
            "未知命名空间状态：{}（可用 active / quarantined / retired）",
            status
        )
    })?;
    if !manager.set_namespace_status(source, status).await? {
        return Err(format!("命名空间不存在：{}", source).into());
    }
    let ledger = manager.load_fusion_ledger().await?;
    if json {
        let summaries: Vec<_> = viewpoint::LensCorpus::new(&[], &ledger)
            .summaries()
            .into_iter()
            .filter(|summary| !summary.context.is_default())
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "namespace": know_prolog::knowledge_fusion::sanitize_namespace_id(source),
                "status": status,
                "namespaces": summaries,
            }))?
        );
    } else {
        println!(
            "命名空间 {} 状态已设为 {}",
            know_prolog::knowledge_fusion::sanitize_namespace_id(source),
            status.label_zh()
        );
        println!("{}", ledger.render_zh());
    }
    Ok(())
}

/// `fuse`：预览 / 物化融合视图，并记录显式裁决。
async fn run_fuse(
    manager: &OxigraphManager,
    config: &know_prolog::SystemConfig,
    apply: bool,
    prefer: &[String],
    arbitrate_view: bool,
    json: bool,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut ledger = manager.load_fusion_ledger().await?;

    let mut recorded = Vec::new();
    for spec in prefer {
        let Some((predicate, namespace)) = spec.split_once('=') else {
            return Err(format!("裁决格式应为 谓词=命名空间：{}", spec).into());
        };
        let predicate = predicate.trim();
        let namespace = namespace.trim();
        if predicate.is_empty() || namespace.is_empty() {
            return Err(format!("裁决格式应为 谓词=命名空间：{}", spec).into());
        }
        let namespace_id = know_prolog::knowledge_fusion::sanitize_namespace_id(namespace);
        if ledger.namespace(&namespace_id).is_none() {
            return Err(
                format!("命名空间「{}」不在账本中：先导入来源，再写裁决", namespace).into(),
            );
        }
        let adjudication = Adjudication::new(predicate, &namespace_id, "命令行裁决", "user");
        ledger.record_adjudication(adjudication.clone());
        recorded.push(adjudication);
    }
    if !recorded.is_empty() {
        ledger.sort_and_dedup();
        manager.sync_fusion_ledger(&ledger).await?;
    }

    let policy = FusionPolicy::new().with_functional(functional_predicates(config));
    let (plan, inserted, removed) = if apply {
        let outcome = manager.apply_fusion(&policy).await?;
        (outcome.plan, outcome.inserted, outcome.removed)
    } else {
        (manager.fusion_plan(&policy).await?, 0, 0)
    };
    if apply {
        ledger = manager.load_fusion_ledger().await?;
    }

    let arbitration = if arbitrate_view {
        let policy = know_prolog::slot_ledger::SlotPolicy::with_defaults()
            .with_functional(functional_predicates(config));
        let provenance = provenance_seed_for_store(Some(&config.system.oxigraph_store_path), None);
        Some(arbitrate(
            &ArbitrationInput::from_ledger(&ledger).with_provenance(&provenance),
            &policy,
        ))
    } else {
        None
    };

    if json {
        let mut report = serde_json::json!({
            "namespaces": ledger.namespaces,
            "assertions": ledger.assertions.len(),
            "adjudications": ledger.adjudications,
            "applied": ledger.applied.len(),
            "inserted": inserted,
            "removed": removed,
            "plan": plan,
        });
        if let Some(arbitration) = &arbitration {
            report["arbitration"] = serde_json::to_value(arbitration)?;
        }
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("{}", ledger.render_zh());
        println!("{}", plan.render_zh());
        if let Some(arbitration) = &arbitration {
            println!("{}", arbitration.render_zh());
        }
        if apply {
            println!("已物化：写入 {} 条，回收 {} 条", inserted, removed);
        } else {
            println!("（预览模式：加 --apply 才会写入默认图）");
        }
    }
    Ok(())
}

/// `lens`：在显式语境镜头下回答一个槽位问题。
///
/// 默认图、融合来源命名图与会话语境共享同一个只读事实视图；镜头内没有
/// 记录时显式披露其他语境，而不是静默回落到默认图。
async fn run_lens(
    manager: &OxigraphManager,
    config: &SystemConfig,
    store_path: &str,
    question: Option<String>,
    subject: Option<String>,
    predicate: Option<String>,
    view: Option<String>,
    functional: Vec<String>,
    contexts: bool,
    facts: bool,
    dialogue_state: Option<String>,
    json: bool,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let ledger = manager.load_fusion_ledger().await?;
    let default_facts = manager.load_fact_refs().await?;
    let corpus = match dialogue_state.as_deref() {
        Some(path) => {
            let (snapshot, _) = SymbolicReasoner::read_dialogue_state(path)
                .map_err(|error| format!("读取对话状态「{}」失败：{}", path, error))?;
            viewpoint::LensCorpus::with_dialogue(&default_facts, &ledger, &snapshot)
        }
        None => viewpoint::LensCorpus::new(&default_facts, &ledger),
    };

    if contexts {
        let summaries = corpus.summaries();
        if json {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "contexts": summaries,
                }))?
            );
        } else {
            println!("语境目录：");
            for summary in &summaries {
                println!(
                    "  • {}：{} 条事实",
                    summary.context.render_short(),
                    summary.facts
                );
            }
        }
        return Ok(());
    }

    let aliases = corpus.aliases();
    if facts {
        let scope = view
            .as_deref()
            .map(|view| viewpoint::parse_lens_scope(view, &aliases))
            .unwrap_or(viewpoint::LensScope::All);
        let listed = corpus.list_facts(&scope, subject.as_deref(), predicate.as_deref())?;
        if json {
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "scope": scope,
                    "count": listed.len(),
                    "facts": listed,
                }))?
            );
        } else {
            let label = scope.label_zh();
            if listed.is_empty() {
                println!("{}中没有匹配的事实。", label);
            } else if matches!(scope, viewpoint::LensScope::Namespace(_)) {
                println!("据{}，共 {} 条事实：", label, listed.len());
            } else {
                println!("{}中的事实（共 {} 条）：", label, listed.len());
            }
            if !listed.is_empty() {
                for fact in &listed {
                    println!(
                        "  • {} 的 {}：{}（出处：{}）",
                        fact.fact.subject,
                        fact.fact.predicate,
                        fact.fact.value,
                        fact.context.render_short()
                    );
                }
            }
        }
        return Ok(());
    }

    if subject.is_some() && predicate.is_none() {
        return Err("--subject 需要与 --predicate 一起使用（列事实请加 --facts）".into());
    }
    let cue = question
        .as_deref()
        .and_then(|question| viewpoint::detect_lens_cue(question, &aliases));
    let scope = if let Some(view) = view.as_deref() {
        viewpoint::parse_lens_scope(view, &aliases)
    } else if let Some(cue) = &cue {
        cue.scope.clone()
    } else {
        return Err(
            "没有识别到语境限定：请用 --view 指定，或把「据 alice 说 / 知识库里 / 假设里」写进问句"
                .into(),
        );
    };

    let query = if let (Some(subject), Some(predicate)) = (&subject, &predicate) {
        viewpoint::SlotQuery::forward(subject, predicate)
    } else if let Some(cue) = &cue {
        viewpoint::extract_slot_query(&cue.question).ok_or_else(|| {
            format!(
                "问句「{}」已限定语境，但没能编译成可查询的槽位；请用 --subject/--predicate 明确指定",
                cue.question
            )
        })?
    } else {
        return Err("请提供问句，或用 --subject/--predicate 指定要查询的槽位".into());
    };

    let registry_path = format!("{}/predicate_registry.json", store_path);
    let _ = manager.load_predicate_registry(&registry_path);
    manager.configure_predicates(
        &config.predicate_profiles,
        config.reasoning.require_predicate_admission,
    );
    let registry = manager.predicate_registry_snapshot();
    let mut functional_set: BTreeSet<String> =
        registry.functional_predicates().into_iter().collect();
    functional_set.extend(
        functional
            .iter()
            .map(|predicate| predicate.trim().to_string())
            .filter(|predicate| !predicate.is_empty()),
    );
    let mut variants = vec![query.predicate.clone()];
    let resolution = registry.resolve(&query.predicate);
    if resolution.known {
        variants.push(resolution.canonical.clone());
        variants.extend(registry.aliases_of(&resolution.canonical));
    }
    let query = query.with_predicate_aliases(variants);
    let is_functional = query
        .predicates
        .iter()
        .any(|predicate| functional_set.contains(predicate));

    let slot_view = corpus.query_slot(&scope, &query, is_functional);
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "state": slot_view.state(),
                "answer": slot_view.render(),
                "view": slot_view,
            }))?
        );
    } else {
        println!("{}", slot_view.render());
    }
    Ok(())
}

/// 导入前恢复持久画像：先载入已确认的谓词/实体/类型声明，再叠加 config，
/// 避免一次 CLI 导入用空注册表覆盖用户此前签过的别名与准入。
fn load_persistent_registries(config: &SystemConfig, manager: &OxigraphManager, store_path: &str) {
    let _ = manager.load_predicate_registry(&format!("{}/predicate_registry.json", store_path));
    manager.configure_predicates(
        &config.predicate_profiles,
        config.reasoning.require_predicate_admission,
    );
    let _ = manager.load_entity_registry(&format!("{}/entity_registry.json", store_path));
    manager.configure_entities(&config.entity_profiles);
    manager.configure_type_system(&config.type_system);
}

/// 导入成功后把默认图观测进谓词/实体注册表并原子落盘。
///
/// RDF/JSON-LD 批量导入是「先有数据、后补画像」的路径：刷新而非准入，
/// 让 `ask`、`laws`、`predicates` 等后续命令看到同一份持久观测。
async fn persist_registry_observations(
    manager: &OxigraphManager,
    store_path: &str,
) -> Result<(), String> {
    manager
        .refresh_predicate_registry()
        .await
        .map_err(|error| format!("刷新谓词画像失败：{}", error))?;
    manager
        .refresh_entity_registry()
        .await
        .map_err(|error| format!("刷新实体身份失败：{}", error))?;
    manager
        .save_predicate_registry(&format!("{}/predicate_registry.json", store_path))
        .map_err(|error| format!("保存谓词画像失败：{}", error))?;
    manager
        .save_entity_registry(&format!("{}/entity_registry.json", store_path))
        .map_err(|error| format!("保存实体身份失败：{}", error))
}

/// 谓词注册表的命令行编辑集（每项都可重复，空集合表示只读查看）。
struct PredicateEdits {
    confirm: Vec<String>,
    reject: Vec<String>,
    alias: Vec<String>,
    inverse: Vec<String>,
    functional: Vec<String>,
    non_functional: Vec<String>,
}

impl PredicateEdits {
    fn is_empty(&self) -> bool {
        self.confirm.is_empty()
            && self.reject.is_empty()
            && self.alias.is_empty()
            && self.inverse.is_empty()
            && self.functional.is_empty()
            && self.non_functional.is_empty()
    }
}

/// 解析 `左=右` 形式的谓词对；任一侧为空或缺少分隔符时返回 None。
fn parse_predicate_spec(spec: &str) -> Option<(&str, &str)> {
    let spec = spec.trim();
    for separator in ["=>", "=", "＝"] {
        if let Some((left, right)) = spec.split_once(separator) {
            let left = left.trim();
            let right = right.trim();
            if !left.is_empty() && !right.is_empty() && left != right {
                return Some((left, right));
            }
        }
    }
    None
}

/// `predicates`：把交互模式的谓词准入/别名/互逆/功能标记开放给脚本。
///
/// 与交互模式同一装配顺序（先加载持久注册表，再叠加 config 声明），
/// 修改后立即原子落盘；只读查看不写任何文件。拒绝会删除该谓词的三元组，
/// 因此报告里逐条给出实际移除数量。
async fn run_predicates(
    config: &SystemConfig,
    manager: &OxigraphManager,
    store_path: &str,
    edits: PredicateEdits,
    json: bool,
) {
    let registry_path = format!("{}/predicate_registry.json", store_path);
    let _ = manager.load_predicate_registry(&registry_path);
    manager.configure_predicates(
        &config.predicate_profiles,
        config.reasoning.require_predicate_admission,
    );
    if let Err(error) = manager.refresh_predicate_registry().await {
        eprintln!("刷新谓词画像失败: {}", error);
        std::process::exit(1);
    }

    let mut confirmed = Vec::new();
    let mut rejected = Vec::new();
    let mut aliases = Vec::new();
    let mut inverses = Vec::new();
    let mut functional = Vec::new();
    let mut invalid = Vec::new();

    for predicate in &edits.confirm {
        let name = predicate.trim();
        if name.is_empty() {
            continue;
        }
        let changed = manager.confirm_predicate(name);
        confirmed.push((name.to_string(), changed));
    }
    for predicate in &edits.reject {
        let name = predicate.trim();
        if name.is_empty() {
            continue;
        }
        let provisional = {
            let registry = manager.predicate_registry_snapshot();
            let canonical = registry.resolve(name).canonical;
            registry
                .profile(&canonical)
                .map_or(false, |profile| profile.provisional)
        };
        if !provisional {
            rejected.push((name.to_string(), 0usize, "not_provisional"));
            continue;
        }
        match manager.reject_predicate(name).await {
            Ok(removed) => rejected.push((name.to_string(), removed, "rejected")),
            Err(error) => {
                eprintln!("拒绝谓词「{}」失败: {}", name, error);
                std::process::exit(1);
            }
        }
    }
    for spec in &edits.alias {
        match parse_predicate_spec(spec) {
            Some((alias, canonical)) => {
                let changed = manager.register_predicate_alias(alias, canonical);
                aliases.push((alias.to_string(), canonical.to_string(), changed));
            }
            None => invalid.push(format!("alias:{}", spec)),
        }
    }
    for spec in &edits.inverse {
        match parse_predicate_spec(spec) {
            Some((left, right)) => {
                let changed = manager.register_predicate_inverse(left, right);
                inverses.push((left.to_string(), right.to_string(), changed));
            }
            None => invalid.push(format!("inverse:{}", spec)),
        }
    }
    for predicate in &edits.functional {
        let name = predicate.trim();
        if name.is_empty() {
            continue;
        }
        let changed = manager.mark_predicate_functional(name, true);
        functional.push((name.to_string(), true, changed));
    }
    for predicate in &edits.non_functional {
        let name = predicate.trim();
        if name.is_empty() {
            continue;
        }
        let changed = manager.mark_predicate_functional(name, false);
        functional.push((name.to_string(), false, changed));
    }

    if !invalid.is_empty() {
        for spec in &invalid {
            eprintln!("无法解析谓词对「{}」（应为 左=右）", spec);
        }
        std::process::exit(1);
    }

    let mutated = !edits.is_empty();
    if mutated {
        if let Err(error) = manager.save_predicate_registry(&registry_path) {
            eprintln!("保存谓词画像失败（{}）：{}", registry_path, error);
            std::process::exit(1);
        }
    }

    let registry = manager.predicate_registry_snapshot();
    if json {
        let registry_json: serde_json::Value =
            serde_json::from_str(&registry.to_json()).unwrap_or(serde_json::Value::Null);
        let payload = serde_json::json!({
            "registry": registry_json,
            "findings": registry.findings_report(),
            "edits": {
                "confirmed": confirmed,
                "rejected": rejected,
                "aliases": aliases,
                "inverses": inverses,
                "functional": functional,
                "saved": mutated,
            },
        });
        match serde_json::to_string_pretty(&payload) {
            Ok(text) => println!("{}", text),
            Err(error) => {
                eprintln!("JSON 序列化失败: {}", error);
                std::process::exit(1);
            }
        }
        return;
    }

    if mutated {
        for (name, changed) in &confirmed {
            println!(
                "{}谓词「{}」{}",
                if *changed {
                    "已确认"
                } else {
                    "无需确认"
                },
                name,
                if *changed {
                    "。"
                } else {
                    "（不是待确认谓词）。"
                }
            );
        }
        for (name, removed, status) in &rejected {
            if *status == "rejected" {
                println!("已拒绝谓词「{}」，移除 {} 条三元组。", name, removed);
            } else {
                println!(
                    "未拒绝「{}」：不是待确认谓词（已声明或已观测谓词不能直接拒绝）。",
                    name
                );
            }
        }
        for (alias, canonical, changed) in &aliases {
            println!(
                "{}别名「{}」→「{}」。",
                if *changed { "已登记" } else { "未变更" },
                alias,
                canonical
            );
        }
        for (left, right, changed) in &inverses {
            println!(
                "{}互逆谓词「{}」↔「{}」。",
                if *changed { "已登记" } else { "未变更" },
                left,
                right
            );
        }
        for (name, functional, changed) in &functional {
            println!(
                "{}「{}」的功能谓词标记{}。",
                if *changed { "已设置" } else { "未变更" },
                name,
                if *functional { "" } else { "（已取消）" }
            );
        }
        println!();
    }
    println!("{}", registry.report());
    println!("{}", registry.findings_report());
}

async fn run_batch_import(
    config: &Arc<SystemConfig>,
    manager: &OxigraphManager,
    store_path: &str,
    file_path: &str,
    dry_run: bool,
    json: bool,
) {
    load_persistent_registries(config, manager, store_path);
    let ledger_path = format!("{}/{}", store_path, batch_teaching::LEDGER_FILE);
    let ledger = match BatchTeachingLedger::load(&ledger_path) {
        BatchLedgerLoad::Loaded(ledger) => ledger,
        BatchLedgerLoad::Missing => {
            eprintln!(
                "没有批量教学台账：{}（先用 batch-teach 学习并 --adopt 采纳规则）",
                ledger_path
            );
            std::process::exit(1);
        }
        BatchLedgerLoad::Damaged(error) => {
            eprintln!("批量教学台账损坏（{}）：{}", ledger_path, error);
            std::process::exit(1);
        }
    };
    if ledger.active_rules().is_empty() {
        eprintln!("批量教学台账里没有已采纳的活动规则（用 batch-teach --adopt 采纳后再导入）");
        std::process::exit(1);
    }
    let text = match std::fs::read_to_string(file_path) {
        Ok(text) => text,
        Err(error) => {
            eprintln!("读取批量语料失败（{}）：{}", file_path, error);
            std::process::exit(1);
        }
    };
    let sentences = batch_teaching::split_sentences(&text);
    if sentences.is_empty() {
        eprintln!("批量语料没有任何句子（{}）", file_path);
        std::process::exit(1);
    }
    let extractor = KnowledgeExtractor::new(config.clone());
    let mut items = Vec::new();
    let mut extracted = 0usize;
    for sentence in &sentences {
        let tokens = batch_teaching::tokenize_sentence(sentence, config);
        let mut sentence_items = extractor.extract(sentence, &tokens);
        sentence_items.retain(
            |item| !matches!(item, KnowledgeItem::Entity { label, .. } if label == "Entity"),
        );
        extracted += sentence_items.len();
        let learned = ledger.extract_active(config, sentence, &tokens);
        extracted += learned.len();
        append_learned_items(&mut sentence_items, learned);
        append_learned_items(&mut items, sentence_items);
    }
    let receipt = if dry_run {
        None
    } else {
        match manager.save_knowledge_with_receipt(items.clone()).await {
            Ok(receipt) => {
                if receipt.can_undo() {
                    let journal_path = format!("{}/import_journal.json", store_path);
                    let mut journal = match ImportJournal::load(&journal_path) {
                        JournalLoad::Loaded(journal) => journal,
                        JournalLoad::Missing => ImportJournal::new(),
                        JournalLoad::Damaged(error) => {
                            eprintln!(
                                "导入日志损坏（{}），本轮按空日志重建：{}",
                                journal_path, error
                            );
                            ImportJournal::new()
                        }
                    };
                    journal.push(receipt.clone());
                    if let Err(error) = journal.save(&journal_path) {
                        eprintln!("批量导入已写入，但保存导入日志失败：{}", error);
                    }
                }
                Some(receipt)
            }
            Err(error) => {
                eprintln!("批量导入写入知识库失败：{}", error);
                std::process::exit(1);
            }
        }
    };
    if !dry_run {
        if let Err(error) = persist_registry_observations(manager, store_path).await {
            eprintln!("批量导入已写入，但画像同步失败：{}", error);
        }
    }
    if json {
        let payload = serde_json::json!({
            "sentences": sentences.len(),
            "extracted": extracted,
            "unique": items.len(),
            "dry_run": dry_run,
            "ledger": ledger_path,
            "receipt": receipt,
        });
        match serde_json::to_string_pretty(&payload) {
            Ok(text) => println!("{}", text),
            Err(error) => eprintln!("批量导入结果序列化失败：{}", error),
        }
        return;
    }
    println!(
        "批量导入{}：{} 句，抽取 {} 条，去重后 {} 条。",
        if dry_run { "（预览）" } else { "" },
        sentences.len(),
        extracted,
        items.len()
    );
    if let Some(receipt) = &receipt {
        println!("{}", receipt.summary());
    } else {
        println!("预览模式：未写入知识库。");
    }
}

#[allow(clippy::too_many_arguments)]
fn run_batch_teach(
    config: &Arc<SystemConfig>,
    store_path: &str,
    facts: Option<&str>,
    corpus: Option<&str>,
    negatives: Option<&str>,
    min_support: usize,
    min_purity: u32,
    max_candidates: usize,
    adopt: &[String],
    retire: &[String],
    ledger_path: Option<&str>,
    list: bool,
    json: bool,
) {
    let path = ledger_path
        .map(str::to_string)
        .unwrap_or_else(|| format!("{}/{}", store_path, batch_teaching::LEDGER_FILE));
    let mut ledger = match BatchTeachingLedger::load(&path) {
        BatchLedgerLoad::Loaded(ledger) => ledger,
        BatchLedgerLoad::Missing => BatchTeachingLedger::default(),
        BatchLedgerLoad::Damaged(error) => {
            eprintln!("批量教学台账损坏（{}）：{}", path, error);
            std::process::exit(1);
        }
    };
    if list {
        if json {
            match serde_json::to_string_pretty(&ledger.visible_rules()) {
                Ok(text) => println!("{}", text),
                Err(error) => {
                    eprintln!("批量教学台账序列化失败：{}", error);
                    std::process::exit(1);
                }
            }
        } else {
            println!("{}", ledger.render_ledger());
        }
        return;
    }
    let teaching = match (facts, corpus) {
        (Some(facts), Some(corpus)) => Some((facts, corpus)),
        (None, None) => None,
        _ => {
            eprintln!("批量教学需要同时提供 --facts 与 --corpus");
            std::process::exit(2);
        }
    };
    let mut episode = None;
    if let Some((facts_path, corpus_path)) = teaching {
        let facts_text = match std::fs::read_to_string(facts_path) {
            Ok(text) => text,
            Err(error) => {
                eprintln!("读取种子事实失败（{}）：{}", facts_path, error);
                std::process::exit(1);
            }
        };
        let corpus_text = match std::fs::read_to_string(corpus_path) {
            Ok(text) => text,
            Err(error) => {
                eprintln!("读取语料失败（{}）：{}", corpus_path, error);
                std::process::exit(1);
            }
        };
        let negatives_text = match negatives {
            Some(path) => match std::fs::read_to_string(path) {
                Ok(text) => text,
                Err(error) => {
                    eprintln!("读取反例失败（{}）：{}", path, error);
                    std::process::exit(1);
                }
            },
            None => String::new(),
        };
        let options = BatchOptions {
            min_support,
            min_purity: min_purity.min(1000),
            max_candidates,
            ..BatchOptions::default()
        };
        let teacher = BatchTeacher::new(config.clone());
        let produced = teacher.teach(&facts_text, &corpus_text, &negatives_text, &options);
        if !json {
            println!("{}", produced.render_text());
        }
        ledger.record(produced.clone());
        match ledger.save(&path) {
            Ok(()) => {
                if !json {
                    println!("台账已写入 {}（候选待显式采纳）。", path);
                }
            }
            Err(error) => {
                eprintln!("写入批量教学台账失败（{}）：{}", path, error);
                std::process::exit(1);
            }
        }
        episode = Some(produced);
    }
    let mut adopted = Vec::new();
    for selector in adopt {
        match ledger.adopt(selector, config.as_ref()) {
            Ok(ids) => adopted.extend(ids),
            Err(error) => {
                let _ = ledger.save(&path);
                eprintln!("采纳失败：{}", error);
                std::process::exit(1);
            }
        }
    }
    let mut retired = Vec::new();
    for selector in retire {
        match ledger.retire(selector) {
            Ok(ids) => retired.extend(ids),
            Err(error) => {
                eprintln!("退役失败：{}", error);
                std::process::exit(1);
            }
        }
    }
    if !adopted.is_empty() || !retired.is_empty() {
        if let Err(error) = ledger.save(&path) {
            eprintln!("写入批量教学台账失败（{}）：{}", path, error);
            std::process::exit(1);
        }
    }
    if json {
        let payload = serde_json::json!({
            "episode": episode,
            "adopted": adopted,
            "retired": retired,
            "active": ledger.active_rules().len(),
            "ledger": path,
        });
        match serde_json::to_string_pretty(&payload) {
            Ok(text) => println!("{}", text),
            Err(error) => eprintln!("批量教学结果序列化失败：{}", error),
        }
    } else {
        if !adopted.is_empty() {
            println!("已采纳：{}", adopted.join(", "));
        }
        if !retired.is_empty() {
            println!("已退役：{}", retired.join(", "));
        }
        println!(
            "活动规则 {} 条，台账：{}（用 batch-teach --list 查看）。",
            ledger.active_rules().len(),
            path
        );
    }
}

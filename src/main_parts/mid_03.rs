fn run_roles(text: Option<&str>, predicate: Option<&str>, json: bool, config_path: &str) {
    use know_prolog::frame_semantics::{FrameAnnotator, FrameBook, FrameInstance};

    let book = std::fs::read_to_string(config_path)
        .ok()
        .and_then(|raw| toml::from_str::<SystemConfig>(&raw).ok())
        .filter(|config| {
            config.frame_semantics.enabled || !config.frame_semantics.schemas.is_empty()
        })
        .map(|config| FrameBook::from_config(&config.frame_semantics))
        .unwrap_or_else(FrameBook::with_core_schemas);

    let inputs: Vec<String> = match text {
        Some(text) if text != "-" => vec![text.to_string()],
        _ => {
            let mut buffer = String::new();
            if let Err(error) = std::io::stdin().read_to_string(&mut buffer) {
                eprintln!("读取标准输入失败: {}", error);
                std::process::exit(1);
            }
            buffer
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .map(str::to_string)
                .collect()
        }
    };
    if inputs.is_empty() {
        eprintln!("没有输入句子：请提供文本参数，或从标准输入逐行输入。");
        std::process::exit(1);
    }

    let annotator = FrameAnnotator::new(&book);
    let mut records = Vec::new();
    let mut missing = 0usize;
    for input in &inputs {
        let frames: Vec<FrameInstance> = match predicate {
            Some(predicate) => annotator
                .annotate_predicate(input, predicate)
                .into_iter()
                .collect(),
            None => annotator.annotate_sentences(input),
        };
        if frames.is_empty() {
            missing += 1;
            if !json {
                eprintln!("未识别谓词：{}（可用 --predicate 指定）", input);
            }
            continue;
        }
        for frame in frames {
            let projection = book.project(&frame);
            let realized = book.realize(&frame);
            if json {
                records.push(serde_json::json!({
                    "text": input,
                    "frame": frame,
                    "projection": projection,
                    "realized": realized,
                }));
            } else {
                println!("输入：{}", input);
                println!("角色：{}", frame.render_roles());
                match &projection {
                    Some(projection) => println!("投影：{}", projection.render_text()),
                    None => println!("投影：核心角色不足，未生成三元组"),
                }
                match &realized {
                    Some(text) => println!("回译：{}", text),
                    None => println!("回译：没有匹配的可逆模板"),
                }
                println!();
            }
        }
    }

    if json {
        match serde_json::to_string_pretty(&records) {
            Ok(rendered) => println!("{}", rendered),
            Err(error) => {
                eprintln!("语义角色帧序列化失败: {}", error);
                std::process::exit(1);
            }
        }
    }
    if missing == inputs.len() {
        std::process::exit(1);
    }
}

async fn run_joint_generate(
    config: &know_prolog::SystemConfig,
    manager: &OxigraphManager,
    store_path: &str,
    facts: &[String],
    context: &[String],
    expected_types: &[String],
    predicate: Option<&str>,
    json: bool,
) {
    let expected = expected_frames_from_specs(facts);
    let snapshot = match manager.load_graph_snapshot().await {
        Ok(facts) => facts,
        Err(error) => {
            eprintln!("读取知识库失败: {}", error);
            std::process::exit(1);
        }
    };
    let reasoner = build_store_reasoner(config, manager, store_path, &snapshot).await;
    let generated = reasoner.joint_generate_and_verify_with_context(
        &expected,
        context,
        expected_types,
        predicate,
    );
    let Some((candidate, verification)) = generated else {
        eprintln!("当前生成器无法为否定/模态帧或空帧生成候选。");
        std::process::exit(1);
    };
    if json {
        let payload = serde_json::json!({
            "candidate": candidate,
            "verification": verification,
        });
        match serde_json::to_string_pretty(&payload) {
            Ok(rendered) => println!("{}", rendered),
            Err(error) => {
                eprintln!("联合生成序列化失败: {}", error);
                std::process::exit(1);
            }
        }
    } else {
        println!("候选：{}", candidate);
        println!("{}", verification.render_report());
    }
}

async fn run_contextual_segment(
    config: &know_prolog::SystemConfig,
    manager: &OxigraphManager,
    text: Option<&str>,
    context_terms: &[String],
    expected_types: &[String],
    predicate: Option<&str>,
    margin: f64,
    memory_path: Option<&str>,
    confirm: &[String],
    correct: &[String],
    json: bool,
) {
    if !(0.0..=1.0).contains(&margin) || !margin.is_finite() {
        eprintln!("--margin 必须在 0 到 1 之间（含端点），实得 {}", margin);
        std::process::exit(2);
    }
    let memory = match memory_path {
        Some(path) => match std::fs::read_to_string(path) {
            Ok(text) => match ReadingMemory::from_json(&text) {
                Ok(memory) => memory,
                Err(error) => {
                    eprintln!("读法记忆解析失败（{}）: {}", path, error);
                    std::process::exit(1);
                }
            },
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => ReadingMemory::new(512),
            Err(error) => {
                eprintln!("读法记忆读取失败（{}）: {}", path, error);
                std::process::exit(1);
            }
        },
        None => ReadingMemory::new(512),
    };

    let mut segmenter = ContextualSegmenter::new();
    segmenter.set_schema(SchemaView::from_config(
        &config.type_system,
        &config.predicate_profiles,
    ));
    for entity_type in &config.entity_types {
        segmenter.observe_type(&entity_type.label);
    }
    for profile in &config.entity_profiles {
        for etype in &profile.etype {
            segmenter.observe_type(etype);
        }
        segmenter.observe_entity(&profile.name, profile.etype.first().map(String::as_str));
        for alias in &profile.aliases {
            segmenter.observe_alias(alias, &profile.name);
        }
    }
    for profile in &config.predicate_profiles {
        segmenter.observe_predicate(&profile.name);
        for alias in &profile.aliases {
            segmenter.observe_predicate_alias(alias, &profile.name);
        }
        for class in profile.domain.iter().chain(profile.range.iter()) {
            segmenter.observe_type(class);
        }
    }

    let facts = match manager.load_graph_snapshot().await {
        Ok(facts) => facts,
        Err(error) => {
            eprintln!("读取知识库失败: {}", error);
            std::process::exit(1);
        }
    };
    for fact in facts.iter() {
        match fact {
            PrologFact::Entity {
                name,
                entity_type,
                properties,
            } => {
                segmenter.observe_entity(name, Some(entity_type));
                for property in properties.keys() {
                    segmenter.observe_predicate(property);
                }
            }
            PrologFact::Relation { from, rel_type, to } => {
                segmenter.observe_triple(from, rel_type, to)
            }
        }
    }

    let source = match text {
        Some(text) => text.to_string(),
        None => {
            let mut input = String::new();
            if let Err(error) = std::io::stdin().read_to_string(&mut input) {
                eprintln!("读取标准输入失败: {}", error);
                std::process::exit(1);
            }
            input.trim_end_matches(['\r', '\n']).to_string()
        }
    };
    if source.trim().is_empty() {
        eprintln!("语境分词需要非空文本（参数或标准输入）。");
        std::process::exit(2);
    }

    *segmenter.memory_mut() = memory;
    let result = segmenter.segment(
        &source,
        &SegmentOptions {
            context_terms: context_terms.to_vec(),
            expected_types: expected_types.to_vec(),
            predicate: predicate.map(str::to_string),
            ambiguity_margin: margin,
            ..Default::default()
        },
    );

    let mut memory_changed = false;
    for spec in confirm {
        let Some((surface, canonical)) = spec.split_once('=') else {
            eprintln!("--confirm 需要 SURFACE=CANONICAL，实得 {}", spec);
            std::process::exit(2);
        };
        let surface = surface.trim();
        let canonical = canonical.trim();
        if surface.is_empty() || canonical.is_empty() {
            eprintln!("--confirm 的词面与规范名都不能为空：{}", spec);
            std::process::exit(2);
        }
        let kind = segmenter
            .reading_kind(surface, canonical)
            .unwrap_or(ReadingKind::Entity);
        memory_changed |=
            segmenter
                .memory_mut()
                .confirm(surface, canonical, kind, &result.context.fingerprint);
    }
    for spec in correct {
        let Some((surface, replacement)) = spec.split_once('=') else {
            eprintln!("--correct 需要 SURFACE=REJECTED>ACCEPTED，实得 {}", spec);
            std::process::exit(2);
        };
        let Some((rejected, accepted)) = replacement.split_once('>') else {
            eprintln!("--correct 需要 SURFACE=REJECTED>ACCEPTED，实得 {}", spec);
            std::process::exit(2);
        };
        let surface = surface.trim();
        let rejected = rejected.trim();
        let accepted = accepted.trim();
        if surface.is_empty() || accepted.is_empty() {
            eprintln!("--correct 的词面与接受读法都不能为空：{}", spec);
            std::process::exit(2);
        }
        let rejected_kind = segmenter
            .reading_kind(surface, rejected)
            .unwrap_or(ReadingKind::Entity);
        let accepted_kind = segmenter
            .reading_kind(surface, accepted)
            .unwrap_or(rejected_kind);
        memory_changed |= segmenter.memory_mut().correct(
            surface,
            rejected,
            rejected_kind,
            accepted,
            accepted_kind,
            &result.context.fingerprint,
        );
    }
    if memory_changed {
        if let Some(path) = memory_path {
            match segmenter.memory().to_json() {
                Ok(rendered) => {
                    if let Err(error) = std::fs::write(path, rendered) {
                        eprintln!("读法记忆写入失败（{}）: {}", path, error);
                        std::process::exit(1);
                    }
                }
                Err(error) => {
                    eprintln!("读法记忆序列化失败: {}", error);
                    std::process::exit(1);
                }
            }
        }
        eprintln!("读法记忆已更新。");
    }
    if json {
        match serde_json::to_string_pretty(&result) {
            Ok(rendered) => println!("{}", rendered),
            Err(error) => {
                eprintln!("语境分词序列化失败: {}", error);
                std::process::exit(1);
            }
        }
    } else {
        print!("{}", result.render_report());
    }
}

/// `frontier`：离线查看每条候选模式停在哪个要素、期望/实见什么、覆盖多少前缀。
///
/// 与运行时同序：先 `preprocess`，再按 `[synonyms.mappings]` 归一，最后调用
/// 模式库的同一匹配器；因此报告里的失败点就是线上失败点，不是另写一遍的近似。
fn run_frontier(
    question: Option<&str>,
    corpus: Option<&str>,
    near_miss: f64,
    json: bool,
    config: &know_prolog::SystemConfig,
) {
    if !(0.0..=1.0).contains(&near_miss) || !near_miss.is_finite() {
        eprintln!(
            "--near-miss 必须在 0 到 1 之间（含端点），实得 {}",
            near_miss
        );
        std::process::exit(2);
    }
    let synonyms = config
        .synonyms
        .as_ref()
        .map(|synonyms| synonyms.mappings.clone())
        .unwrap_or_default();
    let library = PatternLibrary::from_config(config);

    if let Some(path) = corpus {
        let text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(error) => {
                eprintln!("读取语料失败（{}）：{}", path, error);
                return;
            }
        };
        let questions: Vec<&str> = text.lines().collect();
        let report =
            frontier::analyze_corpus(questions, &library, &config.nlp, &synonyms, near_miss);
        if json {
            match serde_json::to_string_pretty(&report) {
                Ok(text) => println!("{}", text),
                Err(error) => eprintln!("序列化决策前沿失败：{}", error),
            }
        } else {
            println!("{}", report.render_report());
        }
        return;
    }

    let question = question.unwrap_or_default();
    let report =
        frontier::analyze_with_threshold(question, &library, &config.nlp, &synonyms, near_miss);
    if json {
        match serde_json::to_string_pretty(&report) {
            Ok(text) => println!("{}", text),
            Err(error) => eprintln!("序列化决策前沿失败：{}", error),
        }
    } else {
        println!("{}", report.render_report());
    }
}

/// `eval-recall`：构念表面召回 + 基线棘轮。
///
/// 分母来自配置声称可答的路由规则及其表面变体；`seen` 与 `unseen` 分开，
/// 避免用「字面命中」冒充泛化。只读配置，不打开知识存储。
fn run_eval_recall(
    baseline: Option<&str>,
    update_baseline: bool,
    allow_regression: bool,
    near_miss: f64,
    json: bool,
    config: &know_prolog::SystemConfig,
) {
    if !(0.0..=1.0).contains(&near_miss) || !near_miss.is_finite() {
        eprintln!(
            "--near-miss 必须在 0 到 1 之间（含端点），实得 {}",
            near_miss
        );
        std::process::exit(2);
    }
    let report = know_prolog::recall_audit::evaluate_config(config, near_miss);
    if json {
        match serde_json::to_string_pretty(&report) {
            Ok(text) => println!("{}", text),
            Err(error) => {
                eprintln!("构念召回序列化失败：{}", error);
                std::process::exit(1);
            }
        }
    } else {
        println!("{}", report.render_text());
    }

    let baseline_path = baseline.unwrap_or("recall_baseline.json");
    if update_baseline {
        let new_baseline = know_prolog::recall_audit::RecallBaseline::from_report(&report);
        match new_baseline.save(baseline_path) {
            Ok(()) => println!("召回基线已写入 {}", baseline_path),
            Err(error) => {
                eprintln!("基线写入失败：{}", error);
                std::process::exit(1);
            }
        }
    } else if baseline.is_some() {
        match know_prolog::recall_audit::RecallBaseline::load(baseline_path) {
            Ok(old_baseline) => {
                let verdict =
                    know_prolog::recall_audit::compare_baseline(&old_baseline, &report, 1e-6);
                print!("{}", verdict.render_text());
                if !verdict.passed && !allow_regression {
                    std::process::exit(1);
                }
            }
            Err(error) => {
                eprintln!("基线读取失败：{}", error);
                std::process::exit(1);
            }
        }
    }
}

/// `read-trace`：离线观察一个问句被切成了什么、词性来自哪里、规划器看到了什么。
/// 提供 `--corpus` 时改为聚合词表缺口（未知词形 + 标记边界），不自动改词表。
/// 加 `--lexicon` 时额外做静态覆盖审计：把「计划可执行但谓词/实体没有定义」
/// 的沉默缺口显式化；`--definitions` 可先通过词门准入 alias/compose 定义。
/// 全过程只读：不写知识库、不改变分词、链接、规划或回答。
fn run_events(
    text: Option<&str>,
    file: Option<&str>,
    question: Option<&str>,
    limit: usize,
    json: bool,
    prolog: bool,
    nquads: bool,
    graph_iri: Option<&str>,
) {
    let source = match (text, file) {
        (Some(text), _) => text.to_string(),
        (None, Some(path)) => match std::fs::read_to_string(path) {
            Ok(content) => content,
            Err(e) => {
                eprintln!("读取文本失败（{}）: {}", path, e);
                std::process::exit(1);
            }
        },
        (None, None) => {
            let mut content = String::new();
            if let Err(e) = std::io::stdin().read_to_string(&mut content) {
                eprintln!("读取标准输入失败: {}", e);
                std::process::exit(1);
            }
            content
        }
    };
    if source.trim().is_empty() {
        eprintln!("事件抽取需要 --text、--file 或标准输入中的非空文本。");
        std::process::exit(2);
    }
    let graph = EventGraph::build_with_limits(&source, limit);
    if nquads {
        let rendered = match graph_iri {
            Some(iri) if !iri.trim().is_empty() => graph.to_nquads_in_graph(iri.trim()),
            _ => graph.to_nquads(),
        };
        print!("{}", rendered);
        if prolog {
            print!("{}", graph.to_prolog_program());
        }
        return;
    }
    if let Some(question) = question {
        let questions: Vec<&str> = question
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .collect();
        let mut dialogue = EventDialogue::new(graph.clone());
        let mut answers = Vec::new();
        for one in &questions {
            let answer = dialogue.ask(one);
            if json {
                answers.push(serde_json::json!({
                    "question": one,
                    "answer": answer,
                }));
            } else {
                println!("问：{}", one);
                print!("{}", answer.render_text());
                println!();
            }
        }
        if json {
            let payload = serde_json::json!({
                "dialect": know_prolog::event_graph::EVENT_DIALECT,
                "source_fingerprint": &graph.source_fingerprint,
                "stats": &graph.stats,
                "answers": answers,
                "focus": dialogue.focus().ids(),
            });
            match serde_json::to_string_pretty(&payload) {
                Ok(rendered) => println!("{}", rendered),
                Err(e) => {
                    eprintln!("事件问答序列化失败: {}", e);
                    std::process::exit(1);
                }
            }
        }
    } else if json {
        match serde_json::to_string_pretty(&graph) {
            Ok(rendered) => println!("{}", rendered),
            Err(e) => {
                eprintln!("事件图序列化失败: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        print!("{}", graph.render_text());
    }
    if prolog {
        print!("{}", graph.to_prolog_program());
    }
}

/// 解析 `--fact`：支持竖线、制表、逗号与空白分隔。
fn parse_semantic_fact(raw: &str) -> Option<know_prolog::provenance::FactRef> {
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }
    let parts: Vec<&str> = if raw.contains('|') {
        raw.splitn(3, '|').collect()
    } else if raw.contains('\t') {
        raw.splitn(3, '\t').collect()
    } else if raw.split_whitespace().count() == 3 {
        raw.split_whitespace().collect()
    } else if raw.matches(',').count() >= 2 {
        raw.splitn(3, ',').collect()
    } else {
        raw.splitn(3, '，').collect()
    };
    let parts: Vec<&str> = parts
        .into_iter()
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .collect();
    if parts.len() != 3 {
        return None;
    }
    Some(know_prolog::provenance::FactRef::new(
        parts[0], parts[1], parts[2],
    ))
}

fn run_semantic_grammar(
    text: Option<&str>,
    fact: Option<&str>,
    roundtrip: bool,
    json: bool,
    config: &know_prolog::SystemConfig,
) {
    let pack = know_prolog::semantic_grammar::RulePack::from_config(config);
    let mut hits = Vec::new();
    let mut trips = Vec::new();
    let mut target_fact = None;

    if let Some(text) = text {
        hits = pack.extract_text(text, &config.nlp);
    } else if let Some(raw) = fact {
        match parse_semantic_fact(raw) {
            Some(parsed) => target_fact = Some(parsed),
            None => {
                eprintln!("无法解析事实「{}」：请写成 subject|predicate|value", raw);
                std::process::exit(1);
            }
        }
    }

    let mut realized = Vec::new();
    if let Some(target) = &target_fact {
        if let Some(realization) = pack.realize_fact(target) {
            realized.push((target.clone(), realization));
        }
    } else {
        for hit in &hits {
            if let Some(realization) = pack.realize_fact(&hit.fact) {
                realized.push((hit.fact.clone(), realization));
            }
        }
    }
    if roundtrip {
        for (target, _) in &realized {
            if let Some(trip) = pack.roundtrip(target, &config.nlp) {
                trips.push(trip);
            }
        }
    }

    let failed_roundtrips = trips.iter().filter(|trip| !trip.consistent).count();
    if json {
        let value = serde_json::json!({
            "enabled": config.semantic_grammar.enabled,
            "rules": pack.len(),
            "errors": pack.errors(),
            "text": text,
            "hits": hits,
            "realizations": realized
                .iter()
                .map(|(_, realization)| realization)
                .collect::<Vec<_>>(),
            "roundtrips": trips,
        });
        match serde_json::to_string_pretty(&value) {
            Ok(output) => println!("{}", output),
            Err(error) => {
                eprintln!("语义语法报告序列化失败：{}", error);
                std::process::exit(1);
            }
        }
    } else {
        println!(
            "双向语义规则包：{} 条规则，运行开关：{}",
            pack.len(),
            if config.semantic_grammar.enabled {
                "开"
            } else {
                "关（预览模式）"
            }
        );
        if text.is_none() && fact.is_none() {
            println!("{}", pack.render());
        }
        for error in pack.errors() {
            eprintln!("编译错误 {}", error.render());
        }
        for hit in &hits {
            println!(
                "- [{}] {}（token {}..{}）",
                hit.rule,
                hit.fact.display(),
                hit.start,
                hit.end
            );
        }
        for (target, realization) in &realized {
            println!(
                "  {} ⇒ {}（规则 {}）",
                target.display(),
                realization.text,
                realization.rule
            );
        }
        for trip in &trips {
            println!(
                "  往返 {}：{} ⇒ {}",
                if trip.consistent {
                    "一致"
                } else {
                    "不一致"
                },
                trip.text,
                trip.recovered
                    .as_ref()
                    .map(|recovered| recovered.display())
                    .unwrap_or_else(|| "（未抽回）".to_string())
            );
        }
        if hits.is_empty() && text.is_some() {
            println!("（没有规则命中该文本）");
        }
        if let Some(target) = &target_fact {
            if !realized.iter().any(|(value, _)| value == target) {
                println!("（没有带 realize 模板的规则可生成该事实）");
            }
        }
    }
    if !pack.errors().is_empty() || failed_roundtrips > 0 {
        std::process::exit(1);
    }
}

fn run_read_trace(
    question: Option<&str>,
    corpus: Option<&str>,
    lexicon: bool,
    definitions: Option<&str>,
    identity: bool,
    json: bool,
    config: &know_prolog::SystemConfig,
) {
    let mut inventory = lexicon.then(|| lexical_trace::LexiconInventory::from_config(config));
    let mut admission = None;
    if let (Some(path), Some(inventory)) = (definitions, inventory.as_mut()) {
        match lex_gate::load_file(path, inventory) {
            Ok(report) => {
                report.apply(inventory);
                admission = Some(report);
            }
            Err(error) => {
                eprintln!("读取定义失败：{}", error);
                return;
            }
        }
    }
    if let Some(admission) = &admission {
        // JSON 模式下报告走 stderr，保证 stdout 仍是单个合法 JSON 文档。
        if json {
            eprintln!("{}", admission.render_report());
        } else {
            println!("{}", admission.render_report());
        }
    }

    if let Some(path) = corpus {
        let text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(error) => {
                eprintln!("读取语料失败（{}）：{}", path, error);
                return;
            }
        };
        let questions: Vec<&str> = text.lines().collect();
        if let Some(inventory) = &inventory {
            let report =
                lexical_trace::coverage_report_with_inventory(questions, &config.nlp, inventory);
            if json {
                match serde_json::to_string_pretty(&report) {
                    Ok(text) => println!("{}", text),
                    Err(error) => eprintln!("序列化词表覆盖报告失败：{}", error),
                }
            } else {
                println!("{}", report.render_report());
            }
            return;
        }
        let report = lexical_trace::lexicon_gap_report(questions, &config.nlp);
        if json {
            match serde_json::to_string_pretty(&report) {
                Ok(text) => println!("{}", text),
                Err(error) => eprintln!("序列化词表缺口报告失败：{}", error),
            }
        } else {
            println!("{}", report.render_report());
        }
        return;
    }
    let question = question.unwrap_or_default();
    let plan = QueryPlanner::new().parse(question);
    let identity_lines = identity.then(|| {
        let mut registry = know_prolog::predicate_registry::PredicateRegistry::new();
        for profile in &config.predicate_profiles {
            registry.declare(profile);
        }
        let book = know_prolog::law_book::LawBook::from_registry(&registry);
        let schema = know_prolog::query_canonical::CanonicalSchema::from_law_book(&book);
        let mut lines = Vec::new();
        let Some(plan) = plan.as_ref() else {
            lines.push("查询身份：没有可编译的计划，无法生成。".to_string());
            return lines;
        };
        match know_prolog::query_canonical::canonicalize_plan(plan, &schema) {
            Ok(canonical) => lines.push(canonical.receipt()),
            Err(error) => lines.push(format!("查询身份：不可归一（{}）。", error)),
        }
        match know_prolog::query_canonical::audit_realization(plan, &schema) {
            Some(audit) => lines.push(audit.receipt()),
            None => lines.push("生成回读：该计划不支持生成中文问句。".to_string()),
        }
        lines
    });
    let trace = match &inventory {
        Some(inventory) => lexical_trace::trace_question_with_inventory(
            question,
            &config.nlp,
            plan.as_ref(),
            inventory,
        ),
        None => lexical_trace::trace_question(question, &config.nlp, plan.as_ref()),
    };
    if json {
        match serde_json::to_string_pretty(&trace) {
            Ok(text) => println!("{}", text),
            Err(error) => eprintln!("序列化读法轨迹失败：{}", error),
        }
    } else {
        println!("{}", trace.render_report());
    }
    if let Some(lines) = identity_lines {
        for line in lines {
            if json {
                eprintln!("{}", line);
            } else {
                println!("{}", line);
            }
        }
    }
}

/// `prompt-lint`：校验提示资产注册表；存在 Error 时以非零码退出。
fn run_prompt_lint(json: bool) {
    let specs = prompt_spec::default_specs();
    let issues = prompt_spec::lint_default();
    let (covered, composed) = prompt_spec::coverage();
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
                "composed": composed,
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
            covered.len() + composed.len()
        );
        for issue in &issues {
            let severity = match issue.severity {
                LintSeverity::Error => "ERROR",
                LintSeverity::Warning => "WARN ",
            };
            println!("{} [{}] {}", severity, issue.spec_id, issue.message);
        }
        if issues.is_empty() {
            println!(
                "lint 通过：{} 条示例逐字渲染一致，无错误与警告",
                specs.len()
            );
        }
        if !composed.is_empty() {
            println!("仍由组合生成器负责的策略：{}", composed.join("、"));
        }
    }
    if errors > 0 {
        std::process::exit(1);
    }
}

/// `scenario`：离线回放查询板剧本；任一步失败以非零码退出（可直接进 CI）。
fn run_query_scenario(file: &str, json: bool) {
    let report = match know_prolog::query_scenario::run_file(file) {
        Ok(report) => report,
        Err(error) => {
            eprintln!("查询剧本运行失败：{}", error);
            std::process::exit(1);
        }
    };
    if json {
        match serde_json::to_string_pretty(&report) {
            Ok(text) => println!("{}", text),
            Err(error) => {
                eprintln!("剧本报告序列化失败：{}", error);
                std::process::exit(1);
            }
        }
    } else {
        println!("{}", report.render_text());
    }
    if !report.passed {
        std::process::exit(1);
    }
}

/// `fuzz-dialogue`：确定性质量差分模糊测试（无配置、无存储、无网络）。
fn run_fuzz_dialogue(
    cases: usize,
    seed: u64,
    json: bool,
    baseline: Option<&str>,
    update_baseline: bool,
    allow_regression: bool,
) {
    let report = dialogue_fuzz::run(&FuzzConfig { cases, seed });
    let baseline_path = baseline
        .map(str::to_string)
        .unwrap_or_else(|| "dialogue_fuzz_baseline.json".to_string());
    let mut verdict: Option<FuzzVerdict> = None;

    if update_baseline {
        let new_baseline = FuzzBaseline::from_report(&report);
        match new_baseline.save(&baseline_path) {
            Ok(()) => {
                if !json {
                    println!(
                        "known-gap 基线已写入 {}（{} 条已知缺口）。",
                        baseline_path,
                        new_baseline.gaps.len()
                    );
                }
            }
            Err(e) => {
                eprintln!("基线写入失败: {}", e);
                std::process::exit(1);
            }
        }
    } else if baseline.is_some() {
        match FuzzBaseline::load(&baseline_path) {
            Ok(old_baseline) => {
                verdict = Some(compare_fuzz_baseline(&old_baseline, &report));
            }
            Err(e) => {
                eprintln!("基线读取失败: {}", e);
                std::process::exit(1);
            }
        }
    }

    if json {
        let value = match &verdict {
            Some(verdict) => serde_json::json!({
                "report": &report,
                "verdict": verdict,
            }),
            None => serde_json::to_value(&report).unwrap_or_default(),
        };
        println!(
            "{}",
            serde_json::to_string_pretty(&value).unwrap_or_default()
        );
    } else {
        println!("{}", report.render_text());
        if let Some(verdict) = &verdict {
            println!("{}", verdict.render_text());
        }
    }

    let regressed = if update_baseline {
        false
    } else {
        verdict
            .as_ref()
            .map(|verdict| !verdict.passed)
            .unwrap_or(report.failed > 0)
    };
    if regressed && !allow_regression {
        std::process::exit(1);
    }
}

/// `config-lint`：把 config.toml 当源码做契约编译；默认 Error 非零退出，
/// `--strict` 连 Warning 也非零退出，便于 CI 棘轮推进。
/// `release-check`：把语言资产编译成 schema 快照并与发布锁做差分，再跑离线
/// 语义门禁。`--update-lock` 只在全部门禁通过后原子地固化新基线。
fn run_release_check(
    config_file: &str,
    lock_path: &str,
    update_lock: bool,
    language_version: Option<String>,
    skip_evals: bool,
    allow_regression: bool,
    json: bool,
) {
    let source = match std::fs::read_to_string(config_file) {
        Ok(source) => source,
        Err(error) => {
            eprintln!("读取配置文件 {} 失败: {}", config_file, error);
            std::process::exit(1);
        }
    };
    let config: know_prolog::SystemConfig = match toml::from_str(&source) {
        Ok(config) => config,
        Err(error) => {
            eprintln!("解析配置文件 {} 失败: {}", config_file, error);
            std::process::exit(1);
        }
    };
    let registry = PluginRegistry::load_dir(&config.plugins.directory);
    let options = CheckOptions {
        config_file,
        lock_path,
        language_version: language_version.as_deref(),
        update_lock,
        skip_evals,
    };
    let report = match release::check(&config, &registry, &options) {
        Ok(report) => report,
        Err(error) => {
            eprintln!("发布检查失败: {}", error);
            std::process::exit(1);
        }
    };
    if json {
        match serde_json::to_string_pretty(&report) {
            Ok(text) => println!("{}", text),
            Err(error) => {
                eprintln!("发布报告序列化失败: {}", error);
                std::process::exit(1);
            }
        }
    } else {
        println!("{}", report.render_text());
    }
    if !report.passed && !allow_regression {
        std::process::exit(1);
    }
}

fn run_config_lint(file: Option<&str>, json: bool, strict: bool) {
    let path = file.unwrap_or("config.toml");
    let source = match std::fs::read_to_string(path) {
        Ok(source) => source,
        Err(e) => {
            eprintln!("读取配置文件 {} 失败: {}", path, e);
            std::process::exit(1);
        }
    };
    let issues = match config_contract::lint_str(&source) {
        Ok(issues) => issues,
        Err(e) => {
            if json {
                let value = serde_json::json!({
                    "version": config_contract::CONFIG_CONTRACT_VERSION,
                    "file": path,
                    "errors": 1,
                    "warnings": 0,
                    "issues": [{
                        "severity": "error",
                        "code": "toml_parse",
                        "path": path,
                        "message": e,
                    }],
                });
                match serde_json::to_string_pretty(&value) {
                    Ok(text) => println!("{}", text),
                    Err(inner) => eprintln!("契约报告序列化失败: {}", inner),
                }
            } else {
                eprintln!(
                    "配置契约 v{}（CC）：TOML 解析失败",
                    config_contract::CONFIG_CONTRACT_VERSION
                );
                eprintln!("ERROR [toml_parse] {}：{}", path, e);
            }
            std::process::exit(1);
        }
    };
    let errors = config_contract::error_count(&issues);
    let warnings = config_contract::warning_count(&issues);
    if json {
        let value = serde_json::json!({
            "version": config_contract::CONFIG_CONTRACT_VERSION,
            "file": path,
            "errors": errors,
            "warnings": warnings,
            "issues": issues,
        });
        match serde_json::to_string_pretty(&value) {
            Ok(text) => println!("{}", text),
            Err(e) => {
                eprintln!("契约报告序列化失败: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        println!("配置文件：{}", path);
        println!("{}", config_contract::render_text(&issues));
    }
    if errors > 0 || (strict && warnings > 0) {
        std::process::exit(1);
    }
}

/// `config-fulfillment`：把配置里的蕴涵声明编译成证明义务，用图上的见证
/// 生成最小问句，再走真实回答管线判定「兑现 / 诚实不知道 / 说错 / 未检验」。
/// 只读：不写知识库、不改变交互路径；图里未登记的谓词单独列出。
async fn run_config_fulfillment(
    file: Option<&str>,
    store: Option<&str>,
    json: bool,
    strict: bool,
    verbose: bool,
    max_obligations: usize,
) {
    let path = file.unwrap_or("config.toml");
    let source = match std::fs::read_to_string(path) {
        Ok(source) => source,
        Err(error) => {
            eprintln!("读取配置文件 {} 失败: {}", path, error);
            std::process::exit(1);
        }
    };
    let config: SystemConfig = match toml::from_str(&source) {
        Ok(config) => config,
        Err(error) => {
            eprintln!("解析配置文件 {} 失败: {}", path, error);
            std::process::exit(1);
        }
    };

    let mut registry = know_prolog::predicate_registry::PredicateRegistry::new();
    registry.set_strict_admission(config.reasoning.require_predicate_admission);
    for profile in &config.predicate_profiles {
        registry.declare(profile);
    }

    let store_path = store.unwrap_or(&config.system.oxigraph_store_path);
    let manager = match OxigraphManager::new(store_path) {
        Ok(manager) => manager,
        Err(error) => {
            eprintln!("打开知识存储 {} 失败: {}", store_path, error);
            std::process::exit(1);
        }
    };
    let facts = match manager.load_graph_snapshot().await {
        Ok(facts) => facts,
        Err(error) => {
            eprintln!("读取知识图谱快照失败: {}", error);
            std::process::exit(1);
        }
    };

    let report = fulfillment::evaluate_with_config(
        &registry,
        facts.as_slice(),
        &config.reasoning,
        max_obligations,
    );

    if json {
        match serde_json::to_string_pretty(&report) {
            Ok(text) => println!("{}", text),
            Err(error) => {
                eprintln!("兑现报告序列化失败: {}", error);
                std::process::exit(1);
            }
        }
    } else {
        println!("配置文件：{}", path);
        println!("知识存储：{}", store_path);
        println!("{}", report.render_text(verbose));
    }

    if !report.is_clean() {
        std::process::exit(1);
    }
    if strict && report.has_gaps() {
        std::process::exit(1);
    }
}

/// `plugins`：清点清单、报告启用模式、跑包内评测；不打开知识存储。
fn run_plugins(settings: &PluginSettings, directory: Option<&str>, json: bool, eval: bool) {
    let directory = directory.unwrap_or(&settings.directory);
    let registry = PluginRegistry::load_dir(directory);
    let runtime = PluginRuntime::with_settings(registry, settings);
    let results = if eval {
        runtime.registry().evaluate()
    } else {
        Vec::new()
    };
    if json {
        let value = serde_json::json!({
            "directory": directory,
            "plugins": runtime.manifests(),
            "modes": runtime.modes(),
            "issues": runtime.issues(),
            "rule_packs": runtime.rule_report(),
            "evals": results,
        });
        match serde_json::to_string_pretty(&value) {
            Ok(text) => println!("{}", text),
            Err(error) => {
                eprintln!("插件报告序列化失败: {}", error);
                std::process::exit(1);
            }
        }
    } else {
        println!("{}", runtime.render_report());
        if eval {
            println!("{}", runtime.registry().render_eval(&results));
        }
    }
    let failed_eval = results.iter().any(|result| !result.passed);
    if runtime.registry().has_errors() || failed_eval {
        std::process::exit(1);
    }
}

/// 启动时的静默断链提醒：只在契约存在 Error 时打一行 stderr，
/// 说明去哪里看明细；不改变任何运行路径（观察，不接管）。
fn compile_frame_program(config: &know_prolog::SystemConfig) -> Option<Arc<FrameProgram>> {
    if !config.dialogue_frames.enabled {
        return None;
    }
    match FrameProgram::compile(&config.dialogue_frames) {
        Ok(program) => Some(Arc::new(program)),
        Err(errors) => {
            eprintln!("绑定帧配置无法编译（已停用，回退常规管线）：");
            for error in errors {
                eprintln!("  {}", error);
            }
            None
        }
    }
}

fn warn_config_contract(config: &know_prolog::SystemConfig) {
    let issues = config_contract::lint(config);
    let errors = config_contract::error_count(&issues);
    if errors > 0 {
        eprintln!(
            "配置契约警告：检测到 {} 个 Error、{} 个 Warning（运行 `knowprolog config-lint` 查看明细；本次启动不改变任何回答路径）",
            errors,
            config_contract::warning_count(&issues)
        );
    }
}

/// 构建质量护照（`eval-quality` 与交互命令共用同一确定性路径）。
/// 规则账本条目 → 归结子句（潜在冲突检测的输入）。
fn ledger_entry_to_clause(
    entry: &know_prolog::graph_ledger::RuleLedgerEntry,
) -> know_prolog::resolution_engine::Clause {
    use know_prolog::resolution_engine::{Atom, Clause, Term};
    fn atom(atom: &know_prolog::graph_ledger::LedgerAtom) -> Atom {
        Atom::new(
            &atom.predicate,
            atom.terms
                .iter()
                .map(|term| {
                    if term.is_var {
                        Term::var(&term.name)
                    } else {
                        Term::constant(&term.name)
                    }
                })
                .collect(),
        )
    }
    Clause::new(
        atom(&entry.head),
        entry.body.iter().map(atom).collect(),
        entry.confidence,
        &entry.source,
    )
}

/// 组装事实级来源：导入日志回填 ∪ `<store>/provenance.json` ∪ 调用方台账。
fn provenance_seed_for_store(
    store_path: Option<&str>,
    extra: Option<&ProvenanceLedger>,
) -> ProvenanceLedger {
    let mut ledger = ProvenanceLedger::new();
    if let Some(store_path) = store_path {
        let journal_path = format!("{}/import_journal.json", store_path);
        if let JournalLoad::Loaded(journal) = ImportJournal::load(&journal_path) {
            ledger.seed_from_import_journal(&journal);
        }
        let ledger_path = format!("{}/provenance.json", store_path);
        if let ProvenanceLoad::Loaded(loaded) = ProvenanceLedger::load(&ledger_path) {
            ledger.merge(&loaded);
        }
    }
    if let Some(extra) = extra {
        ledger.merge(extra);
    }
    ledger
}

async fn quality_report_for(
    oxigraph_manager: &OxigraphManager,
    config: &know_prolog::SystemConfig,
    store_path: &str,
) -> Result<know_prolog::quality::QualityReport, Box<dyn std::error::Error + Send + Sync>> {
    quality_report_for_with_provenance(oxigraph_manager, config, None, Some(store_path)).await
}

async fn quality_report_for_with_provenance(
    oxigraph_manager: &OxigraphManager,
    config: &know_prolog::SystemConfig,
    provenance: Option<&ProvenanceLedger>,
    store_path: Option<&str>,
) -> Result<know_prolog::quality::QualityReport, Box<dyn std::error::Error + Send + Sync>> {
    let facts = oxigraph_manager.load_graph_snapshot().await?;
    let registry = oxigraph_manager.predicate_registry_snapshot();
    let types = oxigraph_manager.type_system_snapshot();
    // 事实级来源：导入日志 ∪ 默认台账 ∪ 外部台账。即使为空也显式传入，
    // 让「谓词被声明过」不再冒充「这条事实有出处」。
    let ledger = provenance_seed_for_store(store_path, provenance);
    let mut report = know_prolog::quality::build_report_with_provenance(
        &facts,
        &registry,
        &types,
        &config.entity_types,
        Some(&ledger),
    );
    // 快照折叠多值；质量护照必须再用无损事实表回填已见证的功能槽冲突。
    if let Ok(fact_refs) = oxigraph_manager.load_fact_refs().await {
        know_prolog::quality::apply_witnessed_conflicts(
            &mut report,
            &fact_refs,
            Some(&ledger),
            &registry,
            &types,
        );
    }
    Ok(report)
}

/// `serve` 的知识装载：与交互模式同源，取整图快照（实体属性 + 关系）。
/// 只喂实体事实会让服务会话的全部关系问法（谁是X的师傅 / 两人什么关系）
/// 与关系派生的索引静默落空，而属性问法却照常有答案，极难被发现。
async fn serve_facts(
    manager: &OxigraphManager,
) -> Result<Vec<PrologFact>, Box<dyn std::error::Error + Send + Sync>> {
    Ok((*manager.load_graph_snapshot().await?).clone())
}

#[derive(Debug, serde::Serialize)]
struct AskAnswer {
    question: String,
    status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    answer: Option<String>,
    /// 以当前知识版本重建的有界推导；None 表示这条回答没有可展开的归结证明。
    #[serde(skip_serializing_if = "Option::is_none")]
    derivation: Option<Derivation>,
}

fn is_ask_terminator(question: &str) -> bool {
    matches!(question.trim(), "quit" | "exit" | "退出" | "q")
}

/// `ask`：把交互模式的核心符号推理做成一次性、无横幅的脚本接口。
///
/// 只读知识存储，不写对话记忆；事实、谓词画像、实体别名与律典的装配顺序
/// 与 `serve`/交互模式一致，保证同一问题在三种入口得到同一结论。
/// ask 一次性会话中的生成控制：接着讲 / 换个角度重述。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GenerationFollowup {
    Continue,
    SwitchAngle,
}

fn generation_followup(input: &str) -> Option<GenerationFollowup> {
    let stripped = input
        .trim()
        .trim_end_matches(['？', '?', '。', '！', '!', ' '])
        .trim();
    const CONTINUE: [&str; 20] = [
        "接着说",
        "继续说",
        "继续介绍",
        "继续讲",
        "接着讲",
        "再讲讲",
        "还有呢",
        "还有吗",
        "再多说点",
        "再多说些",
        "再说说",
        "再说点",
        "展开说说",
        "展开讲",
        "详细一点",
        "详细说说",
        "具体说说",
        "具体一点",
        "说说细节",
        "讲讲细节",
    ];
    const SWITCH: [&str; 6] = [
        "换个角度说",
        "换个说法",
        "换个角度",
        "换一个角度",
        "重新介绍",
        "重新说",
    ];
    if CONTINUE.contains(&stripped) {
        return Some(GenerationFollowup::Continue);
    }
    if SWITCH.contains(&stripped) {
        return Some(GenerationFollowup::SwitchAngle);
    }
    None
}

/// 识别「介绍一下北京 / 北京介绍 / 说说北京」这类纯自然语言介绍请求，
/// 返回已被知识库接地的实体。不是已知实体时返回 None，让位给普通问答，
/// 避免把「介绍一下你自己」误当成实体介绍。
fn intro_request_entity(input: &str, reasoner: &SymbolicReasoner) -> Option<String> {
    let stripped = input
        .trim()
        .trim_end_matches(['？', '?', '。', '！', '!', ' '])
        .trim();
    let mut candidates: Vec<&str> = Vec::new();
    for prefix in [
        "详细介绍",
        "详细介绍一下",
        "概述一下",
        "介绍一下",
        "介绍下",
        "生成介绍",
        "讲讲",
        "说说",
        "叙述一下",
        "叙述",
        "讲述",
        "介绍",
    ] {
        if let Some(rest) = stripped.strip_prefix(prefix) {
            candidates.push(rest.trim());
        }
    }
    if let Some(rest) = stripped.strip_suffix("介绍") {
        candidates.push(rest.trim());
    }
    for rest in candidates {
        let rest = rest
            .trim_start_matches(['：', ':', '，', ',', ' ', '\u{3000}'])
            .trim();
        if rest.is_empty() || rest.chars().count() > 24 {
            continue;
        }
        let mut mentioned = reasoner.entities_mentioned(rest);
        if mentioned.len() == 1 {
            return mentioned.pop();
        }
        if let Some(entity) = reasoner.best_linked_entity(rest) {
            return Some(entity);
        }
    }
    None
}

fn load_rewrite_book(store_path: &str) -> Option<RewriteBook> {
    let store_book = format!("{}/rewrite_book.json", store_path);
    match RewriteBook::load_from_path(&store_book) {
        Ok(book) => return Some(book),
        Err(error) if std::path::Path::new(&store_book).exists() => {
            eprintln!("重写手册读取失败（已忽略）：{}", error);
        }
        Err(_) => {}
    }
    RewriteBook::load_from_path("rewrite_book.json").ok()
}

/// 按 ask/交互/服务一致的装配顺序，从存储事实构建只读推理器。
async fn build_store_reasoner(
    config: &SystemConfig,
    oxigraph_manager: &OxigraphManager,
    store_path: &str,
    facts: &[PrologFact],
) -> SymbolicReasoner {
    let registry_path = format!("{}/predicate_registry.json", store_path);
    let _ = oxigraph_manager.load_predicate_registry(&registry_path);
    oxigraph_manager.configure_predicates(
        &config.predicate_profiles,
        config.reasoning.require_predicate_admission,
    );
    let _ = oxigraph_manager.load_entity_registry(&format!("{}/entity_registry.json", store_path));
    oxigraph_manager.configure_entities(&config.entity_profiles);
    oxigraph_manager.configure_type_system(&config.type_system);

    let synonyms = config
        .synonyms
        .as_ref()
        .map(|synonyms| synonyms.mappings.clone())
        .unwrap_or_else(create_default_synonyms);
    let mut reasoner = SymbolicReasoner::with_perf_ledger(
        config.reasoning.clone(),
        synonyms,
        PerfLedger::shared(),
    );
    reasoner.set_nlp_config(config.nlp.clone());
    if config.semantic_grammar.enabled {
        reasoner
            .attach_semantic_grammar(know_prolog::semantic_grammar::RulePack::from_config(config));
    }
    if config.frame_semantics.enabled {
        reasoner.attach_frame_book(know_prolog::frame_semantics::FrameBook::from_config(
            &config.frame_semantics,
        ));
    }
    if let Ok(memory) = oxigraph_manager.load_lexical_memory().await {
        if !memory.is_empty() {
            reasoner.set_lexical_memory(memory);
        }
    }
    // 导入可逆帧是跨重启的语言记忆：交互模式在导入时学习并落盘，
    // 只读的 ask/联合解析也必须恢复它，否则同一问法在两种入口下不同。
    if let ImportFrameLoad::Loaded(frames) =
        ImportFrameLedger::load(&format!("{}/import_frames.json", store_path))
    {
        if !frames.is_empty() {
            reasoner.attach_import_frames(frames);
        }
    }
    let (ontology_program, _) = owl_annotation::ontology_program_with_annotations(
        &OntologyProgram::from_config(&config.type_system, &config.predicate_profiles),
        &format!("{}/{}", store_path, owl_annotation::LEDGER_FILE),
    );
    reasoner.configure_ontology_program(&ontology_program);
    reasoner.configure_schema_view(&SchemaView::from_config(
        &config.type_system,
        &config.predicate_profiles,
    ));
    // 与交互/服务共用同一注册入口：谓词画像的别名、换向词面与显式律典
    // （传递/对称/互逆）必须一起进入只读推理器；只同步别名会让 ask 忽略配置律。
    sync_predicate_pairs(&mut reasoner, oxigraph_manager);
    reasoner.align_entity_aliases(&oxigraph_manager.confirmed_entity_aliases());
    sync_entity_variant_aliases(&mut reasoner, oxigraph_manager).await;
    if let Some(persisted) = LawBook::load_from_path(&format!("{}/law_book.json", store_path)) {
        reasoner.load_persisted_laws(persisted);
    }
    if let Some(book) = load_rewrite_book(store_path) {
        reasoner.set_rewrite_book(book);
    }
    reasoner.observe_facts(facts);
    reasoner
}

fn run_spine(text: Option<&str>, facts: &[String], question: Option<&str>, json: bool) {
    use know_prolog::dependency_spine::{
        analyze_sentences, compile_frames, extract_frames, SpineIndex,
    };

    let mut index = SpineIndex::new();
    for fact in facts {
        for frame in extract_frames(fact) {
            index.insert_frame(frame);
        }
    }
    index.seal();

    if let Some(question) = question {
        if facts.is_empty() {
            eprintln!("回答依存脊问句至少需要一条 --fact。");
            std::process::exit(1);
        }
        match index.answer_question(question, None) {
            Some(result) => {
                if json {
                    let answers: Vec<serde_json::Value> = result
                        .answers()
                        .into_iter()
                        .map(|(role, value)| serde_json::json!({"role": role, "value": value}))
                        .collect();
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&serde_json::json!({
                            "question": question,
                            "plan": result.plan,
                            "answers": answers,
                            "solutions": result.search.solutions,
                            "receipt": result.search.receipt,
                        }))
                        .unwrap_or_else(|error| format!("{{\"error\": \"{}\"}}", error))
                    );
                } else {
                    println!("{}", result.render_text());
                }
            }
            None => {
                if json {
                    println!(
                        "{}",
                        serde_json::json!({
                            "question": question,
                            "plan": serde_json::Value::Null,
                            "answers": [],
                            "reason": "未找到疑问槽位或谓词",
                        })
                    );
                } else {
                    println!("未把「{}」编译成依存脊问句。", question);
                }
            }
        }
        return;
    }

    let inputs: Vec<String> = match text {
        Some(text) if text != "-" => vec![text.to_string()],
        _ => {
            let mut buffer = String::new();
            if let Err(error) = std::io::stdin().read_to_string(&mut buffer) {
                eprintln!("读取标准输入失败: {}", error);
                std::process::exit(1);
            }
            buffer
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .map(str::to_string)
                .collect()
        }
    };
    if inputs.is_empty() {
        eprintln!("没有输入句子：请提供文本参数，或从标准输入逐行输入。");
        std::process::exit(1);
    }

    if json {
        let records: Vec<serde_json::Value> = inputs
            .iter()
            .map(|input| {
                let trees = analyze_sentences(input);
                let frames: Vec<_> = trees.iter().flat_map(compile_frames).collect();
                serde_json::json!({
                    "text": input,
                    "trees": trees,
                    "frames": frames,
                })
            })
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&records)
                .unwrap_or_else(|error| format!("{{\"error\": \"{}\"}}", error))
        );
        return;
    }

    for input in &inputs {
        for tree in analyze_sentences(input) {
            println!("{}", tree.render_text());
            for frame in compile_frames(&tree) {
                println!("规范框架：{}", frame.render_text());
            }
            println!();
        }
    }
}

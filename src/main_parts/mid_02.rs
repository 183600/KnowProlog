fn load_owl_ledger(path: &str) -> AnnotationLedger {
    match AnnotationLedger::load(path) {
        OwlLedgerLoad::Missing => AnnotationLedger::default(),
        OwlLedgerLoad::Loaded(ledger) => ledger,
        OwlLedgerLoad::Damaged(error) => {
            eprintln!("OWL 人工反馈台账损坏：{}", error);
            std::process::exit(1);
        }
    }
}

async fn run_owl_command(
    config: &Arc<SystemConfig>,
    manager: &OxigraphManager,
    store_path: &str,
    args: Vec<String>,
) {
    let owl_cli = match OwlCli::try_parse_from(std::iter::once("owl".to_string()).chain(args)) {
        Ok(cli) => cli,
        Err(error) => {
            let _ = error.print();
            std::process::exit(2);
        }
    };
    let ledger_path = format!("{}/{}", store_path, owl_annotation::LEDGER_FILE);
    match owl_cli.command {
        OwlCommands::Propose {
            min_support,
            min_purity,
            max_candidates,
            json,
        } => {
            load_persistent_registries(config, manager, store_path);
            let mut ledger = load_owl_ledger(&ledger_path);
            let facts = match manager.load_graph_snapshot().await {
                Ok(facts) => facts,
                Err(error) => {
                    eprintln!("读取知识库失败：{}", error);
                    std::process::exit(1);
                }
            };
            let mut observation = GraphObservation::from_facts(&facts);
            if let Ok(refs) = manager.load_fact_refs().await {
                for fact in refs {
                    if fact.predicate == "type" {
                        observation
                            .instances
                            .push((fact.subject.clone(), fact.value.clone()));
                    }
                }
                observation = GraphObservation::new(observation.instances, observation.relations);
            }
            let mut base =
                OntologyProgram::from_config(&config.type_system, &config.predicate_profiles);
            base.merge(&ledger.accepted_program());
            let options = ProposalOptions {
                min_support,
                min_purity,
                max_candidates,
                ..ProposalOptions::default()
            };
            let proposals = owl_annotation::propose_from_observation(&observation, &base, &options);
            let outcome = ledger.propose(proposals);
            if let Err(error) = ledger.save(&ledger_path) {
                eprintln!("保存 OWL 反馈台账失败：{}", error);
                std::process::exit(1);
            }
            if json {
                let touched: Vec<&str> = outcome
                    .added
                    .iter()
                    .chain(outcome.updated.iter())
                    .map(String::as_str)
                    .collect();
                let candidates: Vec<_> = touched.iter().filter_map(|id| ledger.find(id)).collect();
                let payload = serde_json::json!({
                    "ledger": ledger_path,
                    "observation": {
                        "type_assertions": observation.instances.len(),
                        "relations": observation.relations.len(),
                    },
                    "outcome": outcome,
                    "candidates": candidates,
                });
                match serde_json::to_string_pretty(&payload) {
                    Ok(text) => println!("{}", text),
                    Err(error) => eprintln!("候选序列化失败：{}", error),
                }
                return;
            }
            println!(
                "图谱观测：{} 条类型断言，{} 条关系。",
                observation.instances.len(),
                observation.relations.len()
            );
            if outcome.total_changes() == 0 {
                println!("没有新的可提议公理（驳回记忆会抑制同形候选）。");
            } else {
                for id in outcome.added.iter().chain(outcome.updated.iter()) {
                    if let Some(entry) = ledger.find(id) {
                        println!("{}", entry.render());
                        println!();
                    }
                }
            }
            println!(
                "台账：{}（接受 {}，待裁决 {}，驳回 {}）。用 owl review <id> accept|reject|defer|revise 裁决。",
                ledger_path,
                ledger.accepted_count(),
                ledger.pending_count(),
                ledger.rejected_count()
            );
        }
        OwlCommands::Review {
            id,
            action,
            reviewer,
            note,
            lesson,
        } => {
            let mut ledger = load_owl_ledger(&ledger_path);
            let action = match action {
                OwlReviewAction::Accept => ReviewAction::Accept,
                OwlReviewAction::Reject => ReviewAction::Reject,
                OwlReviewAction::Defer => ReviewAction::Defer,
                OwlReviewAction::Revise => {
                    let Some(text) = lesson.as_deref() else {
                        eprintln!("revise 需要 --lesson，例如 --lesson \"学生是一种人物\"");
                        std::process::exit(2);
                    };
                    let lesson = match parse_ontology_lesson(text) {
                        Ok(lesson) => lesson,
                        Err(error) => {
                            eprintln!("无法解析修正公理：{}", error);
                            std::process::exit(2);
                        }
                    };
                    ReviewAction::Revise { lesson }
                }
            };
            let outcome = match ledger.review(&id, action, &reviewer, note.as_deref().unwrap_or(""))
            {
                Ok(outcome) => outcome,
                Err(error) => {
                    eprintln!("裁决失败：{}", error);
                    std::process::exit(1);
                }
            };
            if let Err(error) = ledger.save(&ledger_path) {
                eprintln!("保存 OWL 反馈台账失败：{}", error);
                std::process::exit(1);
            }
            if let Some(entry) = ledger.find(&outcome.id) {
                println!("{}", entry.render());
            }
            println!(
                "当前已接受公理 {} 条；下次交互/ask/serve 启动时并入本体。台账：{}",
                ledger.accepted_count(),
                ledger_path
            );
        }
        OwlCommands::Report { json } => {
            let ledger = load_owl_ledger(&ledger_path);
            if json {
                match ledger.to_json() {
                    Ok(text) => println!("{}", text),
                    Err(error) => {
                        eprintln!("{}", error);
                        std::process::exit(1);
                    }
                }
            } else {
                println!("{}", ledger.render_report());
                println!("台账：{}", ledger_path);
            }
        }
        OwlCommands::Export {
            format,
            output,
            include_proposed,
        } => {
            let ledger = load_owl_ledger(&ledger_path);
            let format = format.trim().to_ascii_lowercase();
            let text = match format.as_str() {
                "turtle" | "ttl" => match ledger.export_turtle(include_proposed) {
                    Ok(text) => text,
                    Err(error) => {
                        eprintln!("导出 OWL 注解失败：{}", error);
                        std::process::exit(1);
                    }
                },
                "json" => match ledger.to_json() {
                    Ok(text) => text,
                    Err(error) => {
                        eprintln!("导出 OWL 台账失败：{}", error);
                        std::process::exit(1);
                    }
                },
                other => {
                    eprintln!("不支持的导出格式「{}」（应为 turtle 或 json）", other);
                    std::process::exit(2);
                }
            };
            match output.as_deref() {
                None | Some("-") => print!("{}", text),
                Some(path) => {
                    if let Err(error) = std::fs::write(path, text.as_bytes()) {
                        eprintln!("写入导出文件失败（{}）：{}", path, error);
                        std::process::exit(1);
                    }
                    println!("已导出 OWL 人工反馈：{}", path);
                }
            }
        }
        OwlCommands::Import {
            path,
            format,
            replace,
        } => {
            let text = match std::fs::read_to_string(&path) {
                Ok(text) => text,
                Err(error) => {
                    eprintln!("读取导入文件失败（{}）：{}", path, error);
                    std::process::exit(1);
                }
            };
            let format = format.trim().to_ascii_lowercase();
            let format = if format == "auto" {
                match std::path::Path::new(&path)
                    .extension()
                    .and_then(|extension| extension.to_str())
                    .map(|extension| extension.to_ascii_lowercase())
                    .as_deref()
                {
                    Some("json") | Some("jsonld") => "json".to_string(),
                    Some("ttl") | Some("turtle") => "turtle".to_string(),
                    _ if text.trim_start().starts_with('{') => "json".to_string(),
                    _ => "turtle".to_string(),
                }
            } else {
                format
            };
            let imported = match format.as_str() {
                "json" => match serde_json::from_str::<AnnotationLedger>(&text) {
                    Ok(ledger) => ledger,
                    Err(error) => {
                        eprintln!("解析 OWL 台账 JSON 失败：{}", error);
                        std::process::exit(1);
                    }
                },
                "turtle" | "ttl" => match AnnotationLedger::import_turtle(&text) {
                    Ok(ledger) => ledger,
                    Err(error) => {
                        eprintln!("导入 OWL 注解失败：{}", error);
                        std::process::exit(1);
                    }
                },
                other => {
                    eprintln!("不支持的导入格式「{}」（应为 auto、turtle 或 json）", other);
                    std::process::exit(2);
                }
            };
            let mut ledger = if replace {
                AnnotationLedger::default()
            } else {
                load_owl_ledger(&ledger_path)
            };
            let changed = ledger.merge_entries(&imported, true);
            if let Err(error) = ledger.save(&ledger_path) {
                eprintln!("保存 OWL 反馈台账失败：{}", error);
                std::process::exit(1);
            }
            println!(
                "已导入 {} 条候选（更新 {} 条）；当前接受 {}，待裁决 {}，驳回 {}。台账：{}",
                imported.len(),
                changed,
                ledger.accepted_count(),
                ledger.pending_count(),
                ledger.rejected_count(),
                ledger_path
            );
        }
    }
}

/// `rewrite-mine`：从成对问句挖掘表面→规范重写规则，并按留出谓词评估。
/// 语料是显式问答对；规则只在基础解析失败时作为 fallback 参与评估。
fn run_rewrite_mine(
    episodes_path: Option<&str>,
    holdout_predicates: &[String],
    min_support: usize,
    save: Option<&str>,
    strict: bool,
    json: bool,
) {
    let episodes = match episodes_path {
        Some(path) => match rewrite_book::load_episodes(path) {
            Ok(episodes) => episodes,
            Err(error) => {
                eprintln!("{}", error);
                std::process::exit(1);
            }
        },
        None => rewrite_book::seed_episodes(),
    };
    if episodes.is_empty() {
        eprintln!("重写语料为空：请提供至少一条 surface/canonical 问句对。");
        std::process::exit(1);
    }
    let lexicon = rewrite_book::EpisodeLexicon::from_episodes(&episodes);
    let predicates = rewrite_book::corpus_predicates(&episodes, &lexicon);
    let holdout: BTreeSet<String> = if holdout_predicates.is_empty() {
        rewrite_book::default_holdout(&predicates)
    } else {
        holdout_predicates.iter().cloned().collect()
    };
    for predicate in holdout.iter() {
        if !predicates.contains(predicate) {
            eprintln!("提示：留出谓词「{}」不在语料中。", predicate);
        }
    }
    let train: Vec<rewrite_book::RewriteEpisode> = episodes
        .iter()
        .filter(|episode| {
            let involved = rewrite_book::episode_predicates(episode, &lexicon);
            involved.is_empty() || involved.is_disjoint(&holdout)
        })
        .cloned()
        .collect();
    let book = rewrite_book::mine(&train, &episodes, &lexicon, min_support);
    let evaluation = rewrite_book::evaluate(&book, &episodes, &lexicon, &holdout);
    if let Some(path) = save {
        if let Err(error) = book.save_to_path(path) {
            eprintln!("{}", error);
            std::process::exit(1);
        }
    }

    if json {
        let payload = serde_json::json!({
            "episodes": episodes.len(),
            "train_episodes": train.len(),
            "holdout_predicates": holdout.iter().collect::<Vec<_>>(),
            "rules": book.rules,
            "evaluation": evaluation,
            "saved": save,
        });
        match serde_json::to_string_pretty(&payload) {
            Ok(text) => println!("{}", text),
            Err(error) => {
                eprintln!("重写报告序列化失败：{}", error);
                std::process::exit(1);
            }
        }
    } else {
        println!(
            "语料：{} 回合，训练 {} 回合；留出谓词：{}。",
            episodes.len(),
            train.len(),
            if holdout.is_empty() {
                "—".to_string()
            } else {
                holdout.iter().cloned().collect::<Vec<_>>().join("、")
            }
        );
        print!("{}", book.render_text());
        print!("{}", evaluation.render_text());
        if let Some(path) = save {
            println!("已保存重写手册：{}", path);
        }
    }

    let regressed = evaluation.held_out.wrong_rewrites > 0
        || (evaluation.held_out.episodes > 0 && evaluation.held_out.coverage_gain() <= 0.0);
    if strict && regressed {
        std::process::exit(1);
    }
}

/// `rewrite-rules`：查看重写手册，显式采纳/退役候选规则并原子保存。
fn run_rewrite_rules(book_path: &str, adopt: &[String], retire: &[String], json: bool) {
    let mut book = match RewriteBook::load_from_path(book_path) {
        Ok(book) => book,
        Err(error) => {
            eprintln!(
                "{}（可先用 `rewrite-mine --save {}` 生成）",
                error, book_path
            );
            std::process::exit(1);
        }
    };
    let mut changed = false;
    let mut errors = Vec::new();
    let mut adopted = Vec::new();
    let mut retired = Vec::new();
    for selector in adopt {
        match book.adopt(selector) {
            Ok(rule) => {
                adopted.push(rule.id.clone());
                changed = true;
            }
            Err(error) => errors.push(error),
        }
    }
    for selector in retire {
        match book.retire(selector) {
            Ok(rule) => {
                retired.push(rule.id.clone());
                changed = true;
            }
            Err(error) => errors.push(error),
        }
    }
    if changed {
        if let Err(error) = book.save_to_path(book_path) {
            eprintln!("{}", error);
            std::process::exit(1);
        }
    }

    if json {
        let payload = serde_json::json!({
            "book": book,
            "adopted": adopted,
            "retired": retired,
            "errors": errors,
        });
        match serde_json::to_string_pretty(&payload) {
            Ok(text) => println!("{}", text),
            Err(error) => {
                eprintln!("重写手册序列化失败：{}", error);
                std::process::exit(1);
            }
        }
    } else {
        print!("{}", book.render_text());
        for id in &adopted {
            println!("已采纳：{}", id);
        }
        for id in &retired {
            println!("已退役：{}", id);
        }
        for error in &errors {
            eprintln!("{}", error);
        }
    }
    if !errors.is_empty() {
        std::process::exit(1);
    }
}

/// `open-closure`：只操作会话账本，不打开知识存储，也不写律典。
#[allow(clippy::too_many_arguments)]
fn run_open_closure(
    texts: &[String],
    facts: &[String],
    asks: &[String],
    confirm: &[String],
    reject: &[String],
    retract: &[String],
    turns: &[String],
    session_path: Option<&str>,
    max_depth: usize,
    max_nodes: usize,
    max_expansions: usize,
    json: bool,
) {
    let mut parsed_facts = Vec::new();
    for raw in facts {
        match open_closure::parse_fact(raw) {
            Ok(fact) => parsed_facts.push(fact),
            Err(error) => {
                eprintln!("{}", error);
                std::process::exit(2);
            }
        }
    }
    let mut parsed_retractions = Vec::new();
    for raw in retract {
        match open_closure::parse_fact(raw) {
            Ok((subject, predicate, object, _)) => {
                parsed_retractions.push((subject, predicate, object))
            }
            Err(error) => {
                eprintln!("{}", error);
                std::process::exit(2);
            }
        }
    }
    let mut parsed_queries = Vec::new();
    for raw in asks {
        match open_closure::parse_query(raw) {
            Ok(query) => parsed_queries.push(query),
            Err(error) => {
                eprintln!("{}", error);
                std::process::exit(2);
            }
        }
    }

    let mut session = match session_path {
        Some(path) if std::path::Path::new(path).exists() => {
            match OpenClosureSession::load_from_path(path) {
                Some(session) => session,
                None => {
                    eprintln!("读取开放闭包会话失败：{}", path);
                    std::process::exit(1);
                }
            }
        }
        _ => OpenClosureSession::new(),
    };

    let mut warnings = Vec::new();
    let mut retracted = 0usize;
    for (subject, predicate, object) in &parsed_retractions {
        if session.retract_triple(subject, predicate, object) {
            retracted += 1;
        } else {
            warnings.push(format!(
                "未找到可撤回的事实：{}|{}|{}",
                subject, predicate, object
            ));
        }
    }

    let mut observed = 0usize;
    for text in texts {
        observed += session.observe_text(text);
    }
    for (subject, predicate, object, negated) in &parsed_facts {
        let source = "命令行事实";
        let added = if *negated {
            session.observe_negative(subject, predicate, object, source)
        } else {
            session.observe_triple(subject, predicate, object, source)
        };
        if added {
            observed += 1;
        }
    }

    let mut confirmed = 0usize;
    let mut rejected = 0usize;
    for predicate in confirm {
        if session.confirm_transitive(predicate, "命令行 --confirm") {
            confirmed += 1;
        }
    }
    for predicate in reject {
        if session.reject_transitive(predicate, "命令行 --reject") {
            rejected += 1;
        }
    }

    let budget = ClosureBudget {
        max_depth: max_depth.max(1),
        max_nodes: max_nodes.max(1),
        max_expansions: max_expansions.max(1),
        ..ClosureBudget::default()
    };
    let mut turn_outputs = Vec::new();
    if !turns.is_empty() {
        let mut dialogue = OpenClosureDialogue::from_session(session);
        for input in turns {
            let turn = dialogue.process(input, budget);
            match &turn {
                OpenClosureTurn::Observed { added, .. } => observed += added,
                OpenClosureTurn::Confirmed { changed: true, .. } => confirmed += 1,
                OpenClosureTurn::Rejected { changed: true, .. } => rejected += 1,
                _ => {}
            }
            turn_outputs.push((input.clone(), turn));
        }
        session = dialogue.into_session();
    }
    let mut answers = Vec::new();
    for (subject, predicate) in parsed_queries {
        let answer = session.answer(&subject, &predicate, budget);
        answers.push((subject, predicate, answer));
    }

    if let Some(path) = session_path {
        if let Err(error) = session.save_to_path(path) {
            eprintln!("写入开放闭包会话失败（{}）：{}", path, error);
            std::process::exit(1);
        }
    }

    if json {
        let answer_values: Vec<serde_json::Value> = answers
            .iter()
            .map(|(subject, predicate, answer)| {
                serde_json::json!({
                    "subject": subject,
                    "predicate": predicate,
                    "answer": answer,
                    "text": answer.render_text(),
                })
            })
            .collect();
        let turn_values: Vec<serde_json::Value> = turn_outputs
            .iter()
            .map(|(input, turn)| {
                serde_json::json!({
                    "input": input,
                    "turn": turn,
                    "text": turn.render_text(),
                })
            })
            .collect();
        let payload = serde_json::json!({
            "session": &session,
            "turns": turn_values,
            "answers": answer_values,
            "observed": observed,
            "retracted": retracted,
            "confirmed": confirmed,
            "rejected": rejected,
            "warnings": warnings,
        });
        match serde_json::to_string_pretty(&payload) {
            Ok(text) => println!("{}", text),
            Err(error) => {
                eprintln!("开放闭包序列化失败：{}", error);
                std::process::exit(1);
            }
        }
        return;
    }

    print!("{}", session.render_text());
    if observed > 0 || retracted > 0 {
        println!("本次吸收 {} 条记录，撤回 {} 条记录。", observed, retracted);
    }
    if confirmed > 0 || rejected > 0 {
        println!("本次确认 {} 个传递谓词，拒绝 {} 个。", confirmed, rejected);
    }
    for (input, turn) in &turn_outputs {
        println!("\n回合：{}", input);
        print!("{}", turn.render_text());
        if !turn.render_text().ends_with('\n') {
            println!();
        }
    }
    for (subject, predicate, answer) in &answers {
        println!("\n查询：{}|{}", subject, predicate);
        print!("{}", answer.render_text());
    }
    if let Some(path) = session_path {
        println!("会话账本已写入 {}。", path);
    }
    for warning in &warnings {
        eprintln!("{}", warning);
    }
}


/// `privacy`：词性类型化的信息流契约。默认影子审计，`--enforce` 才改写。
#[allow(clippy::too_many_arguments)]
async fn run_privacy(
    text: Option<&str>,
    facts: &[String],
    facts_file: Option<&str>,
    from_store: bool,
    max_facts: Option<usize>,
    store_override: Option<&str>,
    question: Option<&str>,
    purpose: &str,
    precision: &str,
    k: Option<usize>,
    enforce: bool,
    act_subject: &str,
    utterances: &[String],
    grants: &[String],
    grants_once: &[String],
    revokes: &[String],
    forgets: &[String],
    policy_path: Option<&str>,
    session_path: Option<&str>,
    config_path: &str,
    json: bool,
) {
    let mut policy = match policy_path {
        Some(path) => match load_privacy_policy(path) {
            Ok(policy) => policy,
            Err(error) => {
                eprintln!("读取隐私策略失败（{}）：{}", path, error);
                std::process::exit(1);
            }
        },
        None => PrivacyPolicy::default(),
    };
    if enforce {
        policy.mode = PolicyMode::Enforce;
    }
    let requested = match DisclosurePrecision::parse(precision) {
        Some(precision) => precision,
        None => {
            eprintln!("未知精度「{}」：可选 exact / granular / coarse / type / hidden", precision);
            std::process::exit(2);
        }
    };

    let mut parsed_facts = Vec::new();
    for raw in facts {
        match parse_privacy_fact(raw) {
            Ok(fact) => parsed_facts.push(fact),
            Err(error) => {
                eprintln!("{}", error);
                std::process::exit(2);
            }
        }
    }
    if let Some(path) = facts_file {
        match read_privacy_facts(path) {
            Ok(file_facts) => parsed_facts.extend(file_facts),
            Err(error) => {
                eprintln!("{}", error);
                std::process::exit(1);
            }
        }
    }
    if from_store {
        let store_path = match privacy_store_path(config_path, store_override) {
            Ok(path) => path,
            Err(error) => {
                eprintln!("{}", error);
                std::process::exit(1);
            }
        };
        let manager = match OxigraphManager::new(&store_path) {
            Ok(manager) => manager,
            Err(error) => {
                eprintln!("打开知识存储失败（{}）：{}", store_path, error);
                std::process::exit(1);
            }
        };
        let refs = match manager.load_fact_refs().await {
            Ok(refs) => refs,
            Err(error) => {
                eprintln!("读取知识存储事实失败（{}）：{}", store_path, error);
                std::process::exit(1);
            }
        };
        let limit = max_facts.unwrap_or(usize::MAX);
        let mut seen = parsed_facts.iter().cloned().collect::<std::collections::BTreeSet<_>>();
        let mut imported = 0usize;
        for fact in refs {
            if imported >= limit {
                break;
            }
            let key = (fact.subject, fact.predicate, fact.value);
            if seen.insert(key.clone()) {
                parsed_facts.push(key);
                imported += 1;
            }
        }
    }

    let text_owned = match text {
        Some("-") => {
            let mut content = String::new();
            if let Err(error) = std::io::stdin().read_to_string(&mut content) {
                eprintln!("读取标准输入失败：{}", error);
                std::process::exit(1);
            }
            Some(content)
        }
        Some(value) => Some(value.to_string()),
        None => None,
    };
    if text_owned.is_none() && parsed_facts.is_empty() && utterances.is_empty() && !from_store {
        eprintln!("请至少提供 --text、--fact \"主语|谓词|宾语\"、--facts-file、--from-store 或 --utterance");
        std::process::exit(2);
    }

    let mut ledger = match session_path {
        Some(path) if std::path::Path::new(path).exists() => {
            let content = match std::fs::read_to_string(path) {
                Ok(content) => content,
                Err(error) => {
                    eprintln!("读取隐私台账失败（{}）：{}", path, error);
                    std::process::exit(1);
                }
            };
            match PrivacyLedger::from_json(&content) {
                Ok(ledger) => ledger,
                Err(error) => {
                    eprintln!("解析隐私台账失败（{}）：{}", path, error);
                    std::process::exit(1);
                }
            }
        }
        _ => PrivacyLedger::new(),
    };
    ledger.begin_turn();

    for raw in grants {
        match parse_privacy_consent(raw) {
            Ok((subject, category, purpose)) => {
                ledger.grant(&subject, category, &purpose, ConsentScope::Session);
            }
            Err(error) => {
                eprintln!("{}", error);
                std::process::exit(2);
            }
        }
    }
    for raw in grants_once {
        match parse_privacy_consent(raw) {
            Ok((subject, category, purpose)) => {
                ledger.grant(&subject, category, &purpose, ConsentScope::Once);
            }
            Err(error) => {
                eprintln!("{}", error);
                std::process::exit(2);
            }
        }
    }
    for raw in revokes {
        match parse_privacy_consent(raw) {
            Ok((subject, category, purpose)) => {
                ledger.revoke(&subject, category, &purpose);
            }
            Err(error) => {
                eprintln!("{}", error);
                std::process::exit(2);
            }
        }
    }
    for raw in forgets {
        match parse_privacy_forget(raw) {
            Ok((subject, Some(category))) => {
                ledger.forget_category(&subject, category, "命令行 --forget");
            }
            Ok((subject, None)) => {
                ledger.forget_subject(&subject, "命令行 --forget");
            }
            Err(error) => {
                eprintln!("{}", error);
                std::process::exit(2);
            }
        }
    }

    let mut acts = Vec::new();
    for raw in utterances {
        let act = PrivacyAct::parse(raw);
        ledger.apply_act(act_subject, purpose, &act);
        acts.push(act);
    }
    let asks_policy = acts
        .iter()
        .any(|act| matches!(act, PrivacyAct::AskPolicy));

    let annotated = text_owned.as_deref().map(|text| {
        std::fs::read_to_string(config_path)
            .ok()
            .and_then(|content| toml::from_str::<SystemConfig>(&content).ok())
            .map(|config| know_prolog::nlp::preprocess_annotated(text, &config.nlp))
            .unwrap_or_default()
    });
    let spans = text_owned
        .as_deref()
        .map(|text| {
            SpanDetector::new(&policy).scan(
                text,
                annotated.as_deref().unwrap_or(&[]),
            )
        })
        .unwrap_or_default();
    let redaction = text_owned
        .as_deref()
        .map(|text| privacy_flow::redact_spans(text, &spans, &policy));

    let mut graph = PrivacyGraph::new();
    for (subject, predicate, object) in &parsed_facts {
        graph.add_fact(&policy, subject, predicate, object);
    }

    let mut decisions = Vec::new();
    {
        let mut adjudicator = PrivacyAdjudicator::new(&policy, &graph, &mut ledger);
        for (subject, predicate, object) in &parsed_facts {
            let mut request = DisclosureRequest::new(subject, predicate, object)
                .with_purpose(purpose)
                .with_precision(requested);
            if let Some(question) = question {
                request = request.with_question(question);
            }
            if let Some(k) = k {
                request = request.with_k(k);
            }
            decisions.push((
                subject.clone(),
                predicate.clone(),
                object.clone(),
                adjudicator.adjudicate(&request),
            ));
        }
    }

    let audit_summary = if decisions.is_empty() {
        None
    } else {
        Some(privacy_flow::AuditSummary::from_decisions(
            decisions.iter().map(|(_, _, _, decision)| decision),
        ))
    };

    if let Some(path) = session_path {
        let canonical = match ledger.canonical_json() {
            Ok(canonical) => canonical,
            Err(error) => {
                eprintln!("隐私台账序列化失败：{}", error);
                std::process::exit(1);
            }
        };
        if let Some(parent) = std::path::Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                let _ = std::fs::create_dir_all(parent);
            }
        }
        if let Err(error) = std::fs::write(path, canonical) {
            eprintln!("写入隐私台账失败（{}）：{}", path, error);
            std::process::exit(1);
        }
    }

    if json {
        let fact_reports: Vec<serde_json::Value> = decisions
            .iter()
            .map(|(subject, predicate, object, decision)| {
                serde_json::json!({
                    "subject": subject,
                    "predicate": predicate,
                    "object": object,
                    "decision": decision,
                })
            })
            .collect();
        let report = serde_json::json!({
            "mode": policy.mode.code(),
            "from_store": from_store,
            "max_facts": max_facts,
            "requested_precision": requested.code(),
            "spans": spans,
            "redaction": redaction,
            "policy_summary": policy.render_summary(),
            "acts": acts,
            "audit_summary": audit_summary,
            "facts": fact_reports,
            "ledger": ledger,
            "ledger_digest": ledger.digest(),
            "session": session_path,
        });
        match serde_json::to_string_pretty(&report) {
            Ok(text) => println!("{}", text),
            Err(error) => {
                eprintln!("隐私报告序列化失败：{}", error);
                std::process::exit(1);
            }
        }
        return;
    }

    println!("策略模式：{}", policy.mode.label());
    if !acts.is_empty() {
        println!("隐私言语行为（{}）：", acts.len());
        for act in &acts {
            println!("  {}：{}", act.code(), act.render());
        }
    }
    if asks_policy {
        println!("{}", policy.render_summary());
    }
    if from_store && parsed_facts.is_empty() {
        println!("知识存储中没有读到可裁决的事实。");
    }
    if text_owned.is_some() {
        if spans.is_empty() {
            println!("敏感跨度：未发现（强规则与词性弱证据均未命中）");
        } else {
            println!("敏感跨度（{}）：", spans.len());
            for span in &spans {
                println!("  {}", span.render());
            }
        }
    }
    if let Some(redaction) = &redaction {
        let mode = if policy.mode == PolicyMode::Enforce {
            "已执行"
        } else {
            "影子，未生效"
        };
        println!("可披露文本（{}）：{}", mode, redaction.text);
        if !redaction.records.is_empty() {
            println!(
                "改写 {} 处，省略 {} 处；弱证据保留 {} 处",
                redaction.replacements(),
                redaction.suppressed(),
                redaction.kept_weak
            );
            for record in &redaction.records {
                println!(
                    "  {}..{} {}（{}，{}）→ {}",
                    record.start,
                    record.end,
                    record.category.label(),
                    record.strength.label(),
                    record.precision.label(),
                    if record.suppressed {
                        "（已省略）"
                    } else {
                        record.replacement.as_str()
                    }
                );
            }
        }
    }
    if !decisions.is_empty() {
        println!("事实裁决（{}）：", decisions.len());
        for (subject, predicate, object, decision) in &decisions {
            print!(
                "{}",
                render_privacy_decision(subject, predicate, object, decision)
            );
        }
    }
    if let Some(summary) = &audit_summary {
        println!("{}", summary.render());
    }
    println!(
        "台账：第 {} 轮，授权 {} 条，披露 {} 条，墓碑 {} 条（摘要 {}）",
        ledger.turn,
        ledger.consents.len(),
        ledger.disclosures.len(),
        ledger.tombstones.len(),
        ledger.digest()
    );
    if let Some(path) = session_path {
        println!("会话台账已写入 {}。", path);
    }
}

fn render_privacy_decision(
    subject: &str,
    predicate: &str,
    object: &str,
    decision: &privacy_flow::DisclosureDecision,
) -> String {
    let mut output = format!(
        "  {} {} {}
    结论：{}（{}；请求 {} → 实际 {}）
",
        subject,
        predicate,
        object,
        decision.status.label(),
        decision.category.label(),
        decision.requested_precision.label(),
        decision.effective_precision.label()
    );
    match &decision.replacement {
        Some(value) => output.push_str(&format!("    可披露：{}
", value)),
        None => output.push_str("    可披露：（已省略）
"),
    }
    if !decision.codes.is_empty() {
        let codes: Vec<&str> = decision.codes.iter().map(|code| code.code()).collect();
        output.push_str(&format!("    决策码：{}
", codes.join(", ")));
    }
    if !decision.inference_leaks.is_empty() {
        output.push_str("    组合推导风险：
");
        for leak in &decision.inference_leaks {
            output.push_str(&format!("      - {}
", leak.render()));
        }
    }
    output.push_str(&format!(
        "    回执：{} / {} / {}
",
        decision.receipt.status.code(),
        decision.receipt.effective_precision.code(),
        decision.receipt.fact_digest
    ));
    if let Some(audit) = &decision.audit {
        output.push_str(&format!(
            "    影子结论：{}，{}；（执行模式才会生效）
",
            audit.would_be_status.label(),
            audit.would_be_precision.label()
        ));
    }
    output
}

fn load_privacy_policy(path: &str) -> Result<PrivacyPolicy, String> {
    let content = std::fs::read_to_string(path).map_err(|error| error.to_string())?;
    let mut policy = match toml::from_str::<PrivacyPolicy>(&content) {
        Ok(policy) => policy,
        Err(toml_error) => serde_json::from_str::<PrivacyPolicy>(&content).map_err(|json_error| {
            format!("TOML: {}；JSON: {}", toml_error, json_error)
        })?,
    };
    policy.merge_default_rules();
    Ok(policy)
}

fn privacy_store_path(
    config_path: &str,
    store_override: Option<&str>,
) -> Result<String, String> {
    if let Some(path) = store_override {
        return Ok(path.to_string());
    }
    let content = std::fs::read_to_string(config_path)
        .map_err(|error| format!("读取配置失败（{}）：{}", config_path, error))?;
    let config = toml::from_str::<SystemConfig>(&content)
        .map_err(|error| format!("解析配置失败（{}）：{}", config_path, error))?;
    Ok(config.system.oxigraph_store_path)
}

fn read_privacy_facts(path: &str) -> Result<Vec<(String, String, String)>, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|error| format!("读取事实文件失败（{}）：{}", path, error))?;
    let mut facts = Vec::new();
    for (index, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let fact = parse_privacy_fact(trimmed)
            .map_err(|error| format!("{}:{} {}", path, index + 1, error))?;
        facts.push(fact);
    }
    Ok(facts)
}

fn parse_privacy_fact(raw: &str) -> Result<(String, String, String), String> {
    let mut parts = raw.splitn(3, '|');
    let subject = parts.next().unwrap_or_default().trim();
    let predicate = parts.next().unwrap_or_default().trim();
    let object = parts.next().unwrap_or_default().trim();
    if subject.is_empty() || predicate.is_empty() || object.is_empty() {
        return Err(format!(
            "事实格式应为 主语|谓词|宾语（收到「{}」）",
            raw
        ));
    }
    Ok((
        subject.to_string(),
        predicate.to_string(),
        object.to_string(),
    ))
}

fn parse_privacy_consent(raw: &str) -> Result<(String, PrivacyCategory, String), String> {
    let mut parts = raw.splitn(3, '|');
    let subject = parts.next().unwrap_or_default().trim();
    let category = parts.next().unwrap_or_default().trim();
    let purpose = parts.next().unwrap_or_default().trim();
    let category = PrivacyCategory::parse(category)
        .ok_or_else(|| format!("未知隐私类别「{}」（例：contact / 联系方式）", category))?;
    if subject.is_empty() {
        return Err(format!("授权格式应为 主语|类别|用途（收到「{}」）", raw));
    }
    let purpose = if purpose.is_empty() { "问答" } else { purpose };
    Ok((subject.to_string(), category, purpose.to_string()))
}

fn parse_privacy_forget(raw: &str) -> Result<(String, Option<PrivacyCategory>), String> {
    let mut parts = raw.splitn(2, '|');
    let subject = parts.next().unwrap_or_default().trim();
    if subject.is_empty() {
        return Err(format!("遗忘格式应为 主语[|类别]（收到「{}」）", raw));
    }
    let category = match parts.next() {
        Some(value) => Some(PrivacyCategory::parse(value.trim()).ok_or_else(|| {
            format!("未知隐私类别「{}」（例：health / 健康）", value.trim())
        })?),
        None => None,
    };
    Ok((subject.to_string(), category))
}

/// `forge`：离线从正例/反例归纳 token 正则，打印展开与探针结果。
/// 只读配置与例子，不打开存储、不写台账；`--activate` 仅预览物化条目。
#[allow(clippy::too_many_arguments)]
fn run_pattern_forge(
    config: &SystemConfig,
    name: &str,
    route: &str,
    positives: &[String],
    negatives: &[String],
    priority: i32,
    activate: bool,
    json: bool,
) {
    let Some(primary) = positives.first() else {
        eprintln!("锻造至少需要一个 --positive 例子");
        std::process::exit(2);
    };
    let mut request = ForgeRequest::new(name, primary, route).with_priority(priority);
    for positive in positives.iter().skip(1) {
        request = request.with_positive(positive.clone());
    }
    for negative in negatives {
        request = request.with_negative(negative.clone());
    }
    let mut forge = PatternForge::new();
    let outcome = match forge.forge(&request, &config.nlp) {
        Ok(outcome) => outcome,
        Err(error) => {
            eprintln!("锻造失败：{}", error);
            std::process::exit(1);
        }
    };
    let activation = if activate {
        match forge.activate(&outcome.pattern.id) {
            Ok(activation) => Some(activation),
            Err(error) => {
                eprintln!("激活失败：{}", error);
                std::process::exit(1);
            }
        }
    } else {
        None
    };
    let mut outcome = outcome;
    if let Some(activation) = &activation {
        outcome.pattern.status = activation.pattern.status;
    }
    if json {
        let payload = serde_json::json!({
            "created": outcome.created,
            "pattern": &outcome.pattern,
            "active": activation.is_some(),
            "entries": activation.as_ref().map(|activation| &activation.entries),
        });
        match serde_json::to_string_pretty(&payload) {
            Ok(text) => println!("{}", text),
            Err(error) => eprintln!("序列化锻造结果失败：{}", error),
        }
        return;
    }
    println!("{}", outcome.render());
    if let Some(activation) = activation {
        println!(
            "已物化 {} 条无分支模式（预览；未写入存储）：",
            activation.entries.len()
        );
        for entry in &activation.entries {
            println!(
                "  - {}（{}）→ {} · 来源 {}",
                entry.name,
                entry.render(),
                entry.query_type,
                entry.source
            );
        }
    } else {
        println!("（预览：加 --activate 可查看展开后的路由条目）");
    }
}

async fn run_joint_parse(
    config: &know_prolog::SystemConfig,
    manager: &OxigraphManager,
    store_path: &str,
    text: Option<&str>,
    context: &[String],
    expected_types: &[String],
    predicate: Option<&str>,
    margin: f64,
    json: bool,
) {
    if !(0.0..=1.0).contains(&margin) || !margin.is_finite() {
        eprintln!("--margin 必须在 0 到 1 之间（含端点），实得 {}", margin);
        std::process::exit(2);
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
        eprintln!("联合解析需要非空文本（参数或标准输入）。");
        std::process::exit(2);
    }
    let facts = match manager.load_graph_snapshot().await {
        Ok(facts) => facts,
        Err(error) => {
            eprintln!("读取知识库失败: {}", error);
            std::process::exit(1);
        }
    };
    let reasoner = build_store_reasoner(config, manager, store_path, &facts).await;
    let result = reasoner.joint_parse_result(&source, context, expected_types, predicate, margin);
    if json {
        match serde_json::to_string_pretty(&result) {
            Ok(rendered) => println!("{}", rendered),
            Err(error) => {
                eprintln!("联合解析序列化失败: {}", error);
                std::process::exit(1);
            }
        }
    } else {
        println!("{}", result.render_report());
    }
}

fn expected_frames_from_specs(facts: &[String]) -> Vec<RelationFrame> {
    facts
        .iter()
        .map(|spec| {
            let (negative, body) = match spec.strip_prefix('!') {
                Some(body) => (true, body),
                None => (false, spec.as_str()),
            };
            let parts: Vec<&str> = body.split('|').map(str::trim).collect();
            if parts.len() != 3 || parts.iter().any(|part| part.is_empty()) {
                eprintln!(
                    "--fact 需要 SUBJECT|PREDICATE|OBJECT（可选前缀 ! 表示否定），实得 {}",
                    spec
                );
                std::process::exit(2);
            }
            let mut frame = RelationFrame::new(parts[0], parts[1], parts[2]);
            if negative {
                frame = frame.with_polarity(know_prolog::relation_frame::Polarity::Negative);
            }
            frame
        })
        .collect()
}

async fn run_joint_verify(
    config: &know_prolog::SystemConfig,
    manager: &OxigraphManager,
    store_path: &str,
    text: Option<&str>,
    facts: &[String],
    context: &[String],
    expected_types: &[String],
    predicate: Option<&str>,
    json: bool,
) {
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
        eprintln!("联合验证需要非空候选句（参数或标准输入）。");
        std::process::exit(2);
    }
    let expected = expected_frames_from_specs(facts);
    let facts = match manager.load_graph_snapshot().await {
        Ok(facts) => facts,
        Err(error) => {
            eprintln!("读取知识库失败: {}", error);
            std::process::exit(1);
        }
    };
    let reasoner = build_store_reasoner(config, manager, store_path, &facts).await;
    let parsed = reasoner.joint_parse_result(
        &source,
        context,
        expected_types,
        predicate,
        know_prolog::contextual_segmentation::DEFAULT_AMBIGUITY_MARGIN,
    );
    let verification = verify_frames(reasoner.schema_view(), &parsed, &expected);
    if json {
        let payload = serde_json::json!({
            "parse": parsed,
            "verification": verification,
        });
        match serde_json::to_string_pretty(&payload) {
            Ok(rendered) => println!("{}", rendered),
            Err(error) => {
                eprintln!("联合验证序列化失败: {}", error);
                std::process::exit(1);
            }
        }
    } else {
        println!("{}", parsed.render_report());
        println!("{}", verification.render_report());
    }
}

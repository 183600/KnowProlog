{
            let ledger_path = path
                .clone()
                .unwrap_or_else(|| format!("{}/provenance.json", store_path));
            let mut ledger = ProvenanceLedger::new();
            match ProvenanceLedger::load(&ledger_path) {
                ProvenanceLoad::Loaded(loaded) => ledger = loaded,
                ProvenanceLoad::Missing => {}
                ProvenanceLoad::Damaged(error) => {
                    eprintln!("来源台账损坏：{}", error);
                    if record.is_some() {
                        eprintln!("拒绝在损坏台账上追加记录；请先修复或另用 --path。");
                        std::process::exit(1);
                    }
                }
            }
            if let Some(file) = record.as_deref() {
                let text = match std::fs::read_to_string(file) {
                    Ok(text) => text,
                    Err(error) => {
                        eprintln!("读取来源记录失败：{}", error);
                        std::process::exit(1);
                    }
                };
                let rows = match parse_provenance_tsv(&text) {
                    Ok(rows) => rows,
                    Err(error) => {
                        eprintln!("来源记录格式错误：{}", error);
                        std::process::exit(2);
                    }
                };
                let mut added = 0usize;
                for (fact, source) in rows {
                    if ledger.record(fact, source) {
                        added += 1;
                    }
                }
                if let Err(error) = ledger.save(&ledger_path) {
                    eprintln!("来源台账写入失败：{}", error);
                    std::process::exit(1);
                }
                println!(
                    "来源台账已登记 {} 条新来源（{}，共 {} 条事实有登记）。",
                    added,
                    ledger_path,
                    ledger.len()
                );
            }

            // 只读视图：台账 + 导入日志 + 可选命名空间（不回写磁盘）。
            let mut view = ledger.clone();
            let journal_path = format!("{}/import_journal.json", store_path);
            if let JournalLoad::Loaded(journal) = ImportJournal::load(&journal_path) {
                view.seed_from_import_journal(&journal);
            }
            let mut facts = match oxigraph_manager.load_fact_refs().await {
                Ok(facts) => facts,
                Err(error) => {
                    eprintln!("读取知识库失败：{}", error);
                    std::process::exit(1);
                }
            };
            if namespaces {
                match oxigraph_manager.load_fusion_ledger().await {
                    Ok(fusion) => {
                        view.seed_from_fusion_ledger(&fusion);
                        for assertion in &fusion.assertions {
                            facts.push(provenance::FactRef::new(
                                &assertion.subject,
                                &assertion.predicate,
                                &assertion.object,
                            ));
                        }
                        facts.sort();
                        facts.dedup();
                    }
                    Err(error) => eprintln!("读取融合账本失败：{}", error),
                }
            }
            let coverage = view.coverage(&facts);
            if let Some(selector) = fact.as_deref() {
                let parts: Vec<&str> = selector.split('|').collect();
                if parts.is_empty() || parts.len() > 3 || parts[0].trim().is_empty() {
                    eprintln!("--fact 格式应为 subject|predicate|value（value 可省略）");
                    std::process::exit(2);
                }
                let matched: Vec<&provenance::FactRef> = facts
                    .iter()
                    .filter(|candidate| {
                        candidate.subject == parts[0]
                            && (parts.len() < 2 || candidate.predicate == parts[1])
                            && (parts.len() < 3 || candidate.value == parts[2])
                    })
                    .collect();
                if json {
                    let rows: Vec<serde_json::Value> = matched
                        .iter()
                        .map(|candidate| {
                            let sources: Vec<String> = view
                                .sources(candidate)
                                .iter()
                                .map(|source| source.render())
                                .collect();
                            serde_json::json!({
                                "fact": candidate,
                                "display": candidate.display(),
                                "sources": sources,
                            })
                        })
                        .collect();
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&rows).unwrap_or_else(|_| "[]".into())
                    );
                } else if matched.is_empty() {
                    println!("没有匹配的事实：{}", selector);
                } else {
                    for candidate in matched {
                        let sources = view.sources(candidate);
                        let rendered: Vec<String> =
                            sources.iter().map(|source| source.render()).collect();
                        println!(
                            "{}：{}",
                            candidate.display(),
                            if rendered.is_empty() {
                                "来源未登记".to_string()
                            } else {
                                rendered.join("、")
                            }
                        );
                    }
                }
            } else if json {
                let value = serde_json::json!({
                    "ledger_path": ledger_path,
                    "ledger_entries": ledger.len(),
                    "view_entries": view.len(),
                    "digest": ledger.digest(),
                    "coverage": coverage,
                });
                println!(
                    "{}",
                    serde_json::to_string_pretty(&value).unwrap_or_else(|_| "{}".into())
                );
            } else {
                println!(
                    "来源台账：{} 条事实有登记（{}）；导入日志/命名空间回填只用于本次查看，不写盘。",
                    view.len(),
                    ledger_path
                );
                println!("{}", coverage.render_zh());
            }
}

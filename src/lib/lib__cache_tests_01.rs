    use super::*;
    use crate::perf_ledger::PerfLedger;

    /// 有界 LRU：版本失效整体清空、命中触碰、溢出淘汰最久未用
    #[test]
    fn revision_lru_invalidates_touches_and_evicts() {
        let mut cache: RevisionLru<u64> = RevisionLru::new(2);
        assert!(cache.get(1, "a").is_none(), "空缓存不命中");
        assert_eq!(cache.put(1, "a".to_string(), 1), 0);
        assert_eq!(cache.put(1, "b".to_string(), 2), 0);
        assert_eq!(cache.get(1, "a"), Some(&1));
        // 触碰 a 后，写入 c 应淘汰 b
        assert_eq!(cache.put(1, "c".to_string(), 3), 1);
        assert_eq!(cache.get(1, "b"), None);
        assert_eq!(cache.get(1, "a"), Some(&1));
        assert_eq!(cache.get(1, "c"), Some(&3));
        // 版本推进：全部失效，原键不得命中
        assert!(cache.get(2, "a").is_none());
        assert_eq!(cache.put(2, "d".to_string(), 4), 0);
        assert_eq!(cache.len(), 1, "版本变化时先清空");
        assert_eq!(cache.get(2, "d"), Some(&4));
    }

    /// 多槽直答缓存：交替询问两个同版本问题，第二次都可命中
    #[tokio::test]
    async fn direct_answer_cache_keeps_multiple_keys_per_revision() {
        let (manager, _dir) = temp_manager().await;
        manager
            .save_knowledge(vec![
                KnowledgeItem::Entity {
                    label: "City".to_string(),
                    properties: HashMap::from([
                        ("name".to_string(), "北京".to_string()),
                        ("capital".to_string(), "中国".to_string()),
                    ]),
                },
                KnowledgeItem::Entity {
                    label: "Person".to_string(),
                    properties: HashMap::from([("name".to_string(), "张三".to_string())]),
                },
            ])
            .await
            .unwrap();
        let perf = manager.perf_ledger();
        let entities = vec!["北京".to_string()];

        let first = manager.get_direct_answer("capital_query", &entities).await;
        assert_eq!(first.as_deref(), Some("北京的首都是中国"));
        assert!(manager
            .get_direct_answer("author_query", &entities)
            .await
            .is_none());
        let queries = perf.count("direct_answer_query");

        // 两个键都在：来回询问均命中缓存
        assert_eq!(
            manager.get_direct_answer("capital_query", &entities).await,
            first
        );
        assert!(manager
            .get_direct_answer("author_query", &entities)
            .await
            .is_none());
        assert_eq!(perf.count("direct_answer_query"), queries);
        assert_eq!(perf.count("direct_answer_cache_hit"), 2);
    }

    /// 多槽相关事实缓存：同一版本交替询问两个实体，第二次都命中
    #[tokio::test]
    async fn relevant_facts_cache_keeps_multiple_keys_per_revision() {
        let (manager, _dir) = temp_manager().await;
        manager
            .save_knowledge(vec![person("张三"), person("李四")])
            .await
            .unwrap();
        manager.save_knowledge(items("张三", "李四")).await.unwrap();
        let perf = manager.perf_ledger();

        let _ = manager
            .get_relevant_facts(&["张三".to_string()], 2)
            .await
            .unwrap();
        let _ = manager
            .get_relevant_facts(&["李四".to_string()], 2)
            .await
            .unwrap();
        let scans = perf.count("relevant_facts_graph_scan");

        let second_zhang = manager
            .get_relevant_facts(&["张三".to_string()], 2)
            .await
            .unwrap();
        assert!(!second_zhang.is_empty());
        let second_li = manager
            .get_relevant_facts(&["李四".to_string()], 2)
            .await
            .unwrap();
        assert!(!second_li.is_empty());
        assert_eq!(
            perf.count("relevant_facts_graph_scan"),
            scans,
            "两个键同版本都应在缓存中"
        );
        assert_eq!(perf.count("relevant_facts_cache_hit"), 2);
    }

    async fn temp_manager() -> (OxigraphManager, tempfile::TempDir) {
        let dir = tempfile::TempDir::new().unwrap();
        let manager = OxigraphManager::with_perf_ledger(
            dir.path().to_str().unwrap(),
            Arc::new(PerfLedger::new()),
        )
        .unwrap();
        (manager, dir)
    }

    fn items(first: &str, second: &str) -> Vec<KnowledgeItem> {
        vec![KnowledgeItem::Relationship {
            from_label: "人物".to_string(),
            from_id: first.to_string(),
            to_label: "人物".to_string(),
            to_id: second.to_string(),
            rel_type: "师傅".to_string(),
            properties: HashMap::new(),
        }]
    }

    /// 关系宾语在实体属性快照里也必须用本地名，与关系事实的取值一致；
    /// 外部命名空间的 IRI 不能被截断。
    #[tokio::test]
    async fn graph_snapshot_uses_local_names_for_kg_relation_objects() {
        let (manager, _dir) = temp_manager().await;
        let jsonld = r#"{
            "@context": {
                "name": "http://example.org/kg/name",
                "师傅": "http://example.org/kg/师傅",
                "external": "http://example.org/kg/external"
            },
            "@graph": [
                {
                    "@id": "http://example.org/kg/entity/张三",
                    "@type": "http://example.org/kg/Person",
                    "name": "张三",
                    "师傅": { "@id": "http://example.org/kg/entity/李四" },
                    "external": { "@id": "https://other.example/entity/王五" }
                },
                {
                    "@id": "http://example.org/kg/entity/李四",
                    "@type": "http://example.org/kg/Person",
                    "name": "李四"
                }
            ]
        }"#;
        manager.import_jsonld(jsonld).await.unwrap();
        let facts = manager.load_graph_snapshot().await.unwrap();
        let properties = facts
            .iter()
            .find_map(|fact| match fact {
                PrologFact::Entity {
                    name, properties, ..
                } if name == "张三" => Some(properties),
                _ => None,
            })
            .expect("应载入张三");
        assert_eq!(properties.get("师傅").map(String::as_str), Some("李四"));
        assert_eq!(
            properties.get("external").map(String::as_str),
            Some("https://other.example/entity/王五"),
            "外部 IRI 必须原样保留"
        );
        assert!(
            facts.iter().any(|fact| matches!(
                fact,
                PrologFact::Relation { from, rel_type, to }
                    if from == "张三" && rel_type == "师傅" && to == "李四"
            )),
            "关系事实仍应存在：{:?}",
            facts
        );
    }

    fn person(name: &str) -> KnowledgeItem {
        KnowledgeItem::Entity {
            label: "Person".to_string(),
            properties: HashMap::from([("name".to_string(), name.to_string())]),
        }
    }

    /// 快照会把同一谓词的多值折叠进 `HashMap`，无损读取必须保留每一行；
    /// 否则「同一主语两个取值」在冲突检测里结构上不可见。
    #[tokio::test]
    async fn fact_refs_preserve_values_that_snapshot_collapses() {
        let (manager, _dir) = temp_manager().await;
        let turtle = r#"
            @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
            <http://example.org/kg/entity/姚明> rdf:type <http://example.org/kg/Person> .
            <http://example.org/kg/entity/姚明> <http://example.org/kg/name> "姚明" .
            <http://example.org/kg/entity/姚明> <http://example.org/kg/年龄> "38" .
            <http://example.org/kg/entity/姚明> <http://example.org/kg/年龄> "39" .
        "#;
        manager
            .import_rdf_reader(oxigraph::io::RdfFormat::Turtle, turtle.as_bytes())
            .await
            .unwrap();

        let snapshot = manager.load_graph_snapshot().await.unwrap();
        let ages: Vec<&String> = snapshot
            .iter()
            .filter_map(|fact| match fact {
                PrologFact::Entity {
                    name, properties, ..
                } if name == "姚明" => properties.get("年龄"),
                _ => None,
            })
            .collect();
        assert_eq!(ages.len(), 1, "快照只保留最后一个取值（历史兼容行为）");

        let refs = manager.load_fact_refs().await.unwrap();
        let values: BTreeSet<String> = refs
            .iter()
            .filter(|fact| fact.subject == "姚明" && fact.predicate == "年龄")
            .map(|fact| fact.value.clone())
            .collect();
        assert_eq!(
            values,
            ["38".to_string(), "39".to_string()].into_iter().collect(),
            "无损读取必须保留两个取值"
        );
    }

    /// 身份 URI 往返：名字含 `/`、空格时 IRI 合法，且图谱键能无损反解（编码/解码往返律）。
    #[tokio::test]
    async fn entity_uri_roundtrip_handles_awkward_names() {
        let (manager, _dir) = temp_manager().await;
        let name = "北京/大学 附中";
        manager.save_knowledge(vec![person(name)]).await.unwrap();
        let facts = manager.load_all_facts().await.unwrap();
        assert!(
            facts.iter().any(|fact| matches!(
                fact,
                PrologFact::Entity { name: fact_name, .. } if fact_name == name
            )),
            "图谱键应无损往返：{:?}",
            facts
        );
        let iri = crate::identity::entity_uri(crate::identity::DEFAULT_BASE_NS, name);
        assert!(!iri.contains(' '), "URI 不允许空格：{}", iri);
        assert_eq!(
            crate::identity::entity_local_name(crate::identity::DEFAULT_BASE_NS, &iri).as_deref(),
            Some(name)
        );
    }

    /// 版本键查询缓存：同版本重复问同一实体命中缓存；写入后自动失效。
    #[tokio::test]
    async fn relevant_facts_cache_is_revision_keyed() {
        let (manager, _dir) = temp_manager().await;
        manager
            .save_knowledge(vec![person("张三"), person("李四")])
            .await
            .unwrap();
        manager.save_knowledge(items("张三", "李四")).await.unwrap();
        let perf = manager.perf_ledger();

        let first = manager
            .get_relevant_facts(&["张三".to_string()], 2)
            .await
            .unwrap();
        assert!(!first.is_empty());
        let scans_after_first = perf.count("relevant_facts_graph_scan");
        let misses_after_first = perf.count("relevant_facts_cache_miss");

        // 同一版本、同一实体（顺序不同也不影响）：命中缓存
        let second = manager
            .get_relevant_facts(&["张三".to_string()], 2)
            .await
            .unwrap();
        assert_eq!(first, second);
        assert_eq!(perf.count("relevant_facts_graph_scan"), scans_after_first);
        assert_eq!(perf.count("relevant_facts_cache_miss"), misses_after_first);
        assert_eq!(perf.count("relevant_facts_cache_hit"), 1);

        // 写入新关系 → 版本变化 → 必须重新扫描并看到新数据
        manager.save_knowledge(items("李四", "王五")).await.unwrap();
        let third = manager
            .get_relevant_facts(&["张三".to_string()], 2)
            .await
            .unwrap();
        assert!(perf.count("relevant_facts_graph_scan") > scans_after_first);
        let relations: Vec<&str> = third
            .iter()
            .filter_map(|fact| match fact {
                PrologFact::Relation { rel_type, .. } => Some(rel_type.as_str()),
                _ => None,
            })
            .collect();
        assert!(relations.contains(&"师傅"));
    }

    /// 重复导入不推进知识版本。
    #[tokio::test]
    async fn duplicate_import_does_not_bump_revision() {
        let (manager, _dir) = temp_manager().await;
        manager.save_knowledge(items("张三", "李四")).await.unwrap();
        let perf = manager.perf_ledger();
        let bumps = perf.count("revision_bump");

        let replay = manager
            .save_knowledge_with_receipt(items("张三", "李四"))
            .await
            .unwrap();
        assert_eq!(replay.added, 0);
        assert_eq!(perf.count("revision_bump"), bumps, "纯重复导入不应推进版本");

        let _ = manager.load_graph_snapshot().await.unwrap();
        let scans = perf.count("graph_snapshot_scan");
        let _ = manager.load_graph_snapshot().await.unwrap();
        assert_eq!(
            perf.count("graph_snapshot_scan"),
            scans,
            "同版本快照应命中缓存"
        );
        assert!(perf.count("graph_snapshot_cache_hit") >= 1);
    }

    /// 直答缓存：同版本命中（含负缓存）；写入后重查看到新值。
    #[tokio::test]
    async fn direct_answer_cache_is_revision_keyed() {
        let (manager, _dir) = temp_manager().await;
        manager
            .save_knowledge(vec![KnowledgeItem::Entity {
                label: "City".to_string(),
                properties: HashMap::from([
                    ("name".to_string(), "北京".to_string()),
                    ("capital".to_string(), "中国".to_string()),
                ]),
            }])
            .await
            .unwrap();
        let perf = manager.perf_ledger();
        let entities = vec!["北京".to_string()];

        let first = manager.get_direct_answer("capital_query", &entities).await;
        assert_eq!(first.as_deref(), Some("北京的首都是中国"));
        let queries = perf.count("direct_answer_query");

        // 同版本同问句：命中缓存
        let second = manager.get_direct_answer("capital_query", &entities).await;
        assert_eq!(second, first);
        assert_eq!(perf.count("direct_answer_query"), queries);
        assert_eq!(perf.count("direct_answer_cache_hit"), 1);

        // 负缓存：同版本问不存在的属性同样命中缓存
        assert!(manager
            .get_direct_answer("author_query", &entities)
            .await
            .is_none());
        let queries_with_negative = perf.count("direct_answer_query");
        assert!(manager
            .get_direct_answer("author_query", &entities)
            .await
            .is_none());
        assert_eq!(perf.count("direct_answer_query"), queries_with_negative);

        // 写入后版本变化 → 缓存失效并重查
        manager.save_knowledge(vec![person("李四")]).await.unwrap();
        let third = manager.get_direct_answer("capital_query", &entities).await;
        assert_eq!(third.as_deref(), Some("北京的首都是中国"));
        assert!(perf.count("direct_answer_query") > queries_with_negative);
    }

    fn ledger_var(name: &str) -> graph_ledger::LedgerTerm {
        graph_ledger::LedgerTerm {
            name: name.to_string(),
            is_var: true,
        }
    }

    fn ledger_const(name: &str) -> graph_ledger::LedgerTerm {
        graph_ledger::LedgerTerm {
            name: name.to_string(),
            is_var: false,
        }
    }

    fn ledger_rule_entry() -> graph_ledger::RuleLedgerEntry {
        let head = graph_ledger::LedgerAtom {
            predicate: "老师".to_string(),
            terms: vec![ledger_var("X"), ledger_var("Y")],
        };
        let body = vec![graph_ledger::LedgerAtom {
            predicate: "师傅".to_string(),
            terms: vec![ledger_var("X"), ledger_var("Y")],
        }];
        graph_ledger::RuleLedgerEntry {
            id: graph_ledger::rule_entry_id(&head, &body),
            head,
            body,
            confidence: 0.95,
            source: "中文规则".to_string(),
        }
    }

    fn ledger_derived_entry(revision: u64, fingerprint: &str) -> graph_ledger::DerivedLedgerEntry {
        let atom = graph_ledger::LedgerAtom {
            predicate: "老师".to_string(),
            terms: vec![ledger_const("张三"), ledger_const("李四")],
        };
        graph_ledger::DerivedLedgerEntry {
            id: graph_ledger::derived_entry_id(&atom),
            atom,
            rule_source: "中文规则".to_string(),
            premises: vec![graph_ledger::LedgerAtom {
                predicate: "师傅".to_string(),
                terms: vec![ledger_const("张三"), ledger_const("李四")],
            }],
            depth: 1,
            confidence: 0.95,
            revision,
            fingerprint: fingerprint.to_string(),
        }
    }

    /// 图账本与知识图谱隔离：账本写入不进快照/谓词画像，也不推进知识版本。
    #[tokio::test]
    async fn ledger_graphs_are_isolated_from_knowledge_snapshot() {
        let (manager, _dir) = temp_manager().await;
        manager.save_knowledge(items("张三", "李四")).await.unwrap();
        let revision = manager.revision();
        let snapshot_before = manager.load_graph_snapshot().await.unwrap();

        let rule = ledger_rule_entry();
        let fingerprint = graph_ledger::rule_fingerprint(&[rule.clone()]);
        manager.sync_rule_ledger(&[rule]).await.unwrap();
        manager
            .materialize_derived_ledger(
                &[ledger_derived_entry(revision, &fingerprint)],
                revision,
                &fingerprint,
            )
            .await
            .unwrap();

        assert_eq!(manager.revision(), revision, "账本写入不推进知识版本");
        let snapshot_after = manager.load_graph_snapshot().await.unwrap();
        assert_eq!(
            snapshot_before.as_ref(),
            snapshot_after.as_ref(),
            "命名图里的账本四元组不得进入知识快照"
        );

        manager.refresh_predicate_registry().await.unwrap();
        let report = manager.predicate_registry_report();
        assert!(
            !report.contains("ledger"),
            "谓词画像不得观测账本谓词：{}",
            report
        );
    }

    /// 规则账本经存储往返一致；重复同步幂等；空集同步清空账本。
    #[tokio::test]
    async fn rule_ledger_store_roundtrip_and_idempotent_resync() {
        let (manager, _dir) = temp_manager().await;
        let entry = ledger_rule_entry();
        assert_eq!(manager.sync_rule_ledger(&[entry.clone()]).await.unwrap(), 1);

        let loaded = manager.load_rule_ledger().await.unwrap();
        assert!(loaded.synced);
        assert_eq!(loaded.entries, vec![entry.clone()]);
        assert_eq!(
            loaded.fingerprint,
            graph_ledger::rule_fingerprint(&[entry.clone()])
        );

        manager.sync_rule_ledger(&[entry.clone()]).await.unwrap();
        assert_eq!(
            manager.load_rule_ledger().await.unwrap().entries,
            vec![entry.clone()],
            "同内容重复同步应幂等"
        );

        manager.sync_rule_ledger(&[]).await.unwrap();
        let cleared = manager.load_rule_ledger().await.unwrap();
        assert!(cleared.synced);
        assert!(cleared.entries.is_empty(), "空集同步必须清空账本");
    }

    /// 模式账本经存储往返一致；重复同步幂等；空集同步清空账本；不推进知识版本。
    #[tokio::test]
    async fn pattern_ledger_store_roundtrip_and_idempotent_resync() {
        let (manager, _dir) = temp_manager().await;
        manager.save_knowledge(items("张三", "李四")).await.unwrap();
        let revision = manager.revision();

        let config: crate::SystemConfig =
            toml::from_str(include_str!("../../config.toml")).expect("config.toml 必须可解析");
        let mut library = crate::pattern_ir::PatternLibrary::from_config(&config);
        assert!(!library.is_empty());
        library
            .teach_with_guard(
                "省略身高",
                "身高 呢",
                "height_query",
                30,
                Some(crate::pattern_ir::ContextGuard::focus()),
            )
            .expect("守卫模式应可教学");
        // 落图读回按 subject（= id）排序，与库内配置序无关 → 比较前统一按 id 排
        let by_id = |entries: &[crate::pattern_ir::PatternEntry]| {
            let mut v = entries.to_vec();
            v.sort_by(|a, b| a.id().cmp(&b.id()));
            v
        };

        assert_eq!(
            manager
                .sync_pattern_ledger(library.entries())
                .await
                .unwrap(),
            library.len()
        );
        let loaded = manager.load_pattern_ledger().await.unwrap();
        assert_eq!(
            by_id(&loaded),
            by_id(library.entries()),
            "模式账本往返应逐条一致（与顺序无关）"
        );
        assert_eq!(
            loaded
                .iter()
                .find(|entry| entry.name == "省略身高")
                .and_then(|entry| entry.guard.as_ref()),
            Some(&crate::pattern_ir::ContextGuard::focus()),
            "上下文守卫必须随模式账本持久化"
        );
        assert_eq!(manager.revision(), revision, "模式账本写入不推进知识版本");

        manager
            .sync_pattern_ledger(library.entries())
            .await
            .unwrap();
        assert_eq!(
            by_id(&manager.load_pattern_ledger().await.unwrap()),
            by_id(library.entries()),
            "同内容重复同步应幂等"
        );

        manager.sync_pattern_ledger(&[]).await.unwrap();
        let cleared = manager.load_pattern_ledger().await.unwrap();
        assert!(cleared.is_empty(), "空集同步必须清空模式账本");
    }

    /// 模式经验账本经存储往返一致；经验与模式定义分图；空集同步清空。
    #[tokio::test]
    async fn pattern_episode_ledger_store_roundtrip_and_isolated_graph() {
        let (manager, _dir) = temp_manager().await;
        let revision = manager.revision();
        let tokens = vec![crate::nlp::TaggedWord {
            text: "法国".to_string(),
            pos: "ns".to_string(),
        }];
        let mut log = crate::pattern_ir::PatternEpisodeLog::with_capacity(8);
        log.record_unmatched("法国的首都是哪里", &tokens, Some("法国"), 3);
        log.record_unmatched("这是另一句", &tokens, None, 4);

        assert_eq!(
            manager.sync_pattern_episodes(log.entries()).await.unwrap(),
            2
        );
        let loaded = manager.load_pattern_episodes().await.unwrap();
        let mut expected: Vec<_> = log.entries().to_vec();
        expected.sort_by_key(crate::pattern_ir::PatternEpisode::id);
        assert_eq!(loaded, expected, "经验账本往返应逐条一致");
        assert_eq!(manager.revision(), revision, "经验写入不推进知识版本");

        // 模式定义图不受经验图影响
        let config: crate::SystemConfig =
            toml::from_str(include_str!("../../config.toml")).expect("config.toml 必须可解析");
        let library = crate::pattern_ir::PatternLibrary::from_config(&config);
        manager
            .sync_pattern_ledger(library.entries())
            .await
            .unwrap();
        assert_eq!(
            manager.load_pattern_ledger().await.unwrap().len(),
            library.len()
        );
        assert_eq!(manager.load_pattern_episodes().await.unwrap().len(), 2);

        manager.sync_pattern_episodes(&[]).await.unwrap();
        assert!(
            manager.load_pattern_episodes().await.unwrap().is_empty(),
            "空集同步必须清空经验账本"
        );
        assert_eq!(
            manager.load_pattern_ledger().await.unwrap().len(),
            library.len(),
            "清空经验不得影响模式定义"
        );
    }

    /// 派生账本经存储往返一致，物化戳（版本 + 指纹）可审计；重物化全量替换。
    #[tokio::test]
    async fn derived_ledger_store_roundtrip_and_stamp() {
        let (manager, _dir) = temp_manager().await;
        let entry = ledger_derived_entry(5, "fp-one");
        assert_eq!(
            manager
                .materialize_derived_ledger(&[entry.clone()], 5, "fp-one")
                .await
                .unwrap(),
            1
        );
        let loaded = manager.load_derived_ledger().await.unwrap();
        assert!(loaded.materialized);
        assert_eq!(loaded.revision, 5);
        assert_eq!(loaded.fingerprint, "fp-one");
        assert_eq!(loaded.entries, vec![entry.clone()]);

        let replacement = ledger_derived_entry(6, "fp-two");
        manager
            .materialize_derived_ledger(&[replacement.clone()], 6, "fp-two")
            .await
            .unwrap();
        let reloaded = manager.load_derived_ledger().await.unwrap();
        assert_eq!(reloaded.entries, vec![replacement], "重物化应全量替换");
        assert_eq!(reloaded.revision, 6);
        assert_eq!(reloaded.fingerprint, "fp-two");
    }

    /// RDF 格式识别：常见扩展名映射到对应解析器；JSON-LD 与未知扩展名给出可操作错误。
    #[test]
    fn detect_rdf_format_maps_extensions_and_rejects_jsonld() {
        use oxigraph::io::RdfFormat;
        assert_eq!(
            OxigraphManager::detect_rdf_format("kb.ttl").unwrap(),
            RdfFormat::Turtle
        );
        assert_eq!(
            OxigraphManager::detect_rdf_format("KB.TTL").unwrap(),
            RdfFormat::Turtle,
            "扩展名大小写不敏感"
        );
        assert_eq!(
            OxigraphManager::detect_rdf_format("kb.nt").unwrap(),
            RdfFormat::NTriples
        );
        assert_eq!(
            OxigraphManager::detect_rdf_format("kb.nq").unwrap(),
            RdfFormat::NQuads
        );
        assert_eq!(
            OxigraphManager::detect_rdf_format("kb.trig").unwrap(),
            RdfFormat::TriG
        );
        assert_eq!(
            OxigraphManager::detect_rdf_format("kb.rdf").unwrap(),
            RdfFormat::RdfXml
        );
        assert_eq!(
            OxigraphManager::detect_rdf_format("kb.xml").unwrap(),
            RdfFormat::RdfXml
        );

        let jsonld_error = OxigraphManager::detect_rdf_format("kb.jsonld")
            .unwrap_err()
            .to_string();
        assert!(jsonld_error.contains("import-jsonld"), "{}", jsonld_error);
        let unknown_error = OxigraphManager::detect_rdf_format("kb.bin")
            .unwrap_err()
            .to_string();
        assert!(unknown_error.contains("ttl"), "{}", unknown_error);
        assert!(
            OxigraphManager::detect_rdf_format("kb").is_err(),
            "无扩展名应报错"
        );
    }

    /// 标准输入没有扩展名，必须能按显式格式名解析；未知名称要明确报错。
    #[test]
    fn parse_rdf_format_accepts_names_and_rejects_unknown() {
        use oxigraph::io::RdfFormat;
        assert!(matches!(
            OxigraphManager::parse_rdf_format("ttl"),
            Ok(RdfFormat::Turtle)
        ));
        assert!(matches!(
            OxigraphManager::parse_rdf_format("N-Triples"),
            Ok(RdfFormat::NTriples)
        ));
        assert!(matches!(
            OxigraphManager::parse_rdf_format("nq"),
            Ok(RdfFormat::NQuads)
        ));
        assert!(matches!(
            OxigraphManager::parse_rdf_format("trig"),
            Ok(RdfFormat::TriG)
        ));
        assert!(matches!(
            OxigraphManager::parse_rdf_format("rdf/xml"),
            Ok(RdfFormat::RdfXml)
        ));
        let error = OxigraphManager::parse_rdf_format("jsonld").unwrap_err();
        assert!(error.to_string().contains("无法识别"), "{}", error);
    }

    /// RDF 文件导入：新增三元组计数正确、重复导入幂等、数据进入知识快照。
    #[tokio::test]
    async fn import_rdf_file_counts_new_triples_and_feeds_snapshot() {
        let (manager, dir) = temp_manager().await;
        let ttl_path = dir.path().join("sample.ttl");
        fs::write(
            &ttl_path,
            r#"@prefix kg: <http://example.org/kg/> .
<http://example.org/kg/entity/王五> a kg:Person ;
    kg:name "王五" ;
    kg:age "20" .
<http://example.org/kg/entity/王五> kg:师父 <http://example.org/kg/entity/赵六> .
"#,
        )
        .unwrap();

        let added = manager
            .import_rdf_file(ttl_path.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(added, 4, "应报告 4 条新增三元组");

        let facts = manager.load_graph_snapshot().await.unwrap();
        assert!(
            facts.iter().any(|fact| matches!(
                fact,
                PrologFact::Entity { name, properties, .. }
                    if name == "王五" && properties.get("age").map(String::as_str) == Some("20")
            )),
            "导入的实体应进入知识快照：{:?}",
            facts
        );

        let revision_after_first = manager.revision();
        let replay = manager
            .import_rdf_file(ttl_path.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(replay, 0, "重复导入不应新增三元组");
        assert!(
            manager.revision() >= revision_after_first,
            "导入后知识版本推进（严格正版本号）"
        );

        let nt_path = dir.path().join("extra.nt");
        fs::write(
            &nt_path,
            "<http://example.org/kg/entity/赵六> <http://example.org/kg/name> \"赵六\" .\n",
        )
        .unwrap();
        assert_eq!(
            manager
                .import_rdf_file(nt_path.to_str().unwrap())
                .await
                .unwrap(),
            1
        );
    }

    /// N-Quads / TriG 的命名图在通用导入中并入默认图，命名图本身不落库。
    #[tokio::test]
    async fn import_rdf_reader_flattens_named_graphs_into_default_graph() {
        let (manager, _dir) = temp_manager().await;
        let nquads = concat!(
            "<http://example.org/kg/entity/甲> <http://example.org/kg/name> \"甲\" <http://example.org/kg/g1> .\n",
            "<http://example.org/kg/entity/乙> <http://example.org/kg/name> \"乙\" <http://example.org/kg/g2> .\n",
        );
        let added = manager
            .import_rdf_reader(oxigraph::io::RdfFormat::NQuads, nquads.as_bytes())
            .await
            .unwrap();
        assert_eq!(added, 2);

        let trig = concat!(
            "@prefix kg: <http://example.org/kg/> .\n",
            "<http://example.org/kg/g> {\n",
            "  <http://example.org/kg/entity/丙> <http://example.org/kg/name> \"丙\" .\n",
            "}\n",
        );
        assert_eq!(
            manager
                .import_rdf_reader(oxigraph::io::RdfFormat::TriG, trig.as_bytes())
                .await
                .unwrap(),
            1
        );

        let default_graph = manager
            .query_sparql("SELECT ?s WHERE { ?s ?p ?o }")
            .await
            .unwrap();
        assert_eq!(
            default_graph.data.len(),
            3,
            "命名图中的三元组应全部并入默认图：{:?}",
            default_graph.data
        );

        let named = manager
            .query_sparql("SELECT ?g WHERE { GRAPH ?g { ?s ?p ?o } }")
            .await
            .unwrap();
        assert!(
            named.data.is_empty(),
            "通用导入不应保留命名图：{:?}",
            named.data
        );
    }

    /// RDF/XML 经 reader 导入同样落入默认图。
    #[tokio::test]
    async fn import_rdf_reader_accepts_rdfxml() {
        let (manager, _dir) = temp_manager().await;
        let rdfxml = r#"<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:kg="http://example.org/kg/">
  <rdf:Description rdf:about="http://example.org/kg/entity/孙七">
    <rdf:type rdf:resource="http://example.org/kg/Person"/>
    <kg:name>孙七</kg:name>
  </rdf:Description>
</rdf:RDF>
"#;
        let added = manager
            .import_rdf_reader(oxigraph::io::RdfFormat::RdfXml, rdfxml.as_bytes())
            .await
            .unwrap();
        assert_eq!(added, 2);
        let facts = manager.load_graph_snapshot().await.unwrap();
        assert!(facts.iter().any(|fact| matches!(
            fact,
            PrologFact::Entity { name, .. } if name == "孙七"
        )));
    }

    /// 实体身份注册表：保存即观测（name 别名自动确认）、显式裁决、落盘往返。
    #[tokio::test]
    async fn entity_registry_observes_saves_and_roundtrips() {
        let (manager, dir) = temp_manager().await;
        let mut knowledge = vec![person("张三"), person("李四")];
        knowledge.extend(items("张三", "李四"));
        manager.save_knowledge(knowledge).await.unwrap();

        let facts = manager.load_all_facts().await.unwrap();
        assert!(manager.observe_entity_facts(&facts) > 0, "首次观测应有变化");
        assert_eq!(
            manager.resolve_entity_identity("张三"),
            crate::entity_registry::EntityResolution::Canonical("张三".to_string())
        );
        assert!(matches!(
            manager.entity_mint_check("张三"),
            crate::entity_registry::MintVerdict::Exists { .. }
        ));

        // 显式确认别名 → 链接器可见；重复确认无变化
        assert!(manager.confirm_entity_alias("老张", "张三"));
        assert!(!manager.confirm_entity_alias("老张", "张三"));
        assert_eq!(
            manager.resolve_entity_identity("老张"),
            crate::entity_registry::EntityResolution::Canonical("张三".to_string())
        );
        assert_eq!(
            manager.confirmed_entity_aliases(),
            vec![("老张".to_string(), "张三".to_string())]
        );

        // 落盘 → 新实例加载逐字节一致（规范 JSON）；库目录另起（RocksDB 单写锁）
        let path = dir.path().join("entity_registry.json");
        let path_text = path.to_str().unwrap();
        manager.save_entity_registry(path_text).unwrap();
        let (restored, _other_dir) = temp_manager().await;
        assert!(restored.load_entity_registry(path_text));
        assert_eq!(
            restored.entity_registry_snapshot().to_json(),
            manager.entity_registry_snapshot().to_json()
        );

        // 提名 → 拒绝：墓碑阻止解析；合并：改名后重定向
        let nominations =
            manager.nominate_entity_aliases(&[crate::entity_registry::AliasCandidate {
                surface: "小张".to_string(),
                canonical: "张三".to_string(),
                pattern: "简称".to_string(),
                evidence: "测试".to_string(),
            }]);
        assert_eq!(
            nominations[0].kind,
            crate::entity_registry::NominationKind::Pending
        );
        assert!(manager.reject_entity_alias("小张", "张三"));
        assert_eq!(
            manager.resolve_entity_identity("小张"),
            crate::entity_registry::EntityResolution::Unknown
        );
        let merged = manager.merge_entity_identities("张三", "张老师").unwrap();
        assert_eq!(merged.into, "张老师");
        assert_eq!(
            manager.resolve_entity_identity("张三"),
            crate::entity_registry::EntityResolution::Redirected {
                from: "张三".to_string(),
                to: "张老师".to_string(),
            }
        );
        let perf = manager.perf_ledger();
        assert!(perf.count("identity_alias_confirm") >= 1);
        assert_eq!(perf.count("identity_merge"), 1);
    }

    /// JSON-LD 导入统计：实体数 / 命名节点关系数 / 实际新增三元组数；
    /// 重复导入新增为 0（与 RDF 导入一致的确定性口径）。
    #[tokio::test]
    async fn import_jsonld_reports_entity_relation_and_triple_counts() {
        let (manager, _dir) = temp_manager().await;
        let jsonld = r#"{
            "@context": {
                "name": "http://example.org/kg/name",
                "teacher_of": "http://example.org/kg/teacher_of",
                "Person": "http://example.org/kg/Person"
            },
            "@graph": [
                {
                    "@id": "http://example.org/kg/entity/小明",
                    "@type": "Person",
                    "name": "小明"
                },
                {
                    "@id": "http://example.org/kg/entity/张三",
                    "@type": "Person",
                    "name": "张三",
                    "teacher_of": { "@id": "http://example.org/kg/entity/李四" }
                }
            ]
        }"#;

        let stats = manager.import_jsonld(jsonld).await.unwrap();
        assert_eq!(stats.entities, 2, "两个节点");
        assert_eq!(stats.relations, 1, "一条指向命名节点的关系边");
        assert_eq!(stats.triples, 5, "类型 2 + 名字 2 + 关系 1");
        assert_eq!(
            stats.summary(),
            "解析了 2 个实体和 1 个关系，新增 5 条三元组。"
        );

        let replay = manager.import_jsonld(jsonld).await.unwrap();
        assert_eq!(replay.entities, 2);
        assert_eq!(replay.relations, 1);
        assert_eq!(replay.triples, 0, "重复导入不新增三元组");
    }

    /// JSON-LD：前缀/对象上下文、@base 相对 @id，以及带属性的嵌套节点都要落库。
    #[tokio::test]
    async fn import_jsonld_expands_context_and_nested_nodes() {
        let (manager, _dir) = temp_manager().await;
        let jsonld = r#"{
            "@context": {
                "ex": "http://example.org/kg/",
                "Person": { "@id": "ex:Person" },
                "@base": "http://example.org/kg/entity/",
                "name": "ex:name",
                "friend": { "@id": "ex:friend", "@type": "@id" }
            },
            "@graph": [
                {
                    "@id": "张三",
                    "@type": "Person",
                    "name": "张三",
                    "friend": {
                        "@id": "李四",
                        "@type": "Person",
                        "name": "李四"
                    }
                }
            ]
        }"#;

        let stats = manager.import_jsonld(jsonld).await.unwrap();
        assert_eq!(stats.entities, 2, "嵌套命名节点也应计为实体");
        assert_eq!(stats.relations, 1);
        assert_eq!(stats.triples, 5, "类型 2 + 名字 2 + 好友关系 1");

        let result = manager
            .query_sparql(
                r#"PREFIX kg: <http://example.org/kg/>
                SELECT ?friendName WHERE {
                    <http://example.org/kg/entity/张三> kg:friend ?friend .
                    ?friend kg:name ?friendName
                }"#,
            )
            .await
            .unwrap();
        assert_eq!(result.data.len(), 1, "{:?}", result.data);
        assert_eq!(
            result.data[0].get("friendName"),
            Some(&Value::String("李四".to_string()))
        );
    }

    /// JSON-LD 空白节点：`_:label` 作为主语与宾语都落库，重复导入保持幂等。
    #[tokio::test]
    async fn import_jsonld_supports_blank_node_ids_and_references() {
        let (manager, _dir) = temp_manager().await;
        let jsonld = r#"{
            "@context": {
                "name": "http://example.org/kg/name",
                "knows": "http://example.org/kg/knows",
                "Person": "http://example.org/kg/Person"
            },
            "@graph": [
                {
                    "@id": "_:a",
                    "@type": "Person",
                    "name": "匿名甲",
                    "knows": { "@id": "_:b" }
                },
                {
                    "@id": "_:b",
                    "@type": "Person",
                    "name": "匿名乙"
                }
            ]
        }"#;

        let stats = manager.import_jsonld(jsonld).await.unwrap();
        assert_eq!(stats.entities, 2);
        assert_eq!(stats.relations, 1);
        assert_eq!(stats.triples, 5, "类型 2 + 名字 2 + 关系 1");

        let result = manager
            .query_sparql(
                r#"PREFIX kg: <http://example.org/kg/>
                SELECT ?subject ?targetName WHERE {
                    ?subject kg:name "匿名甲" ;
                             kg:knows ?target .
                    ?target kg:name ?targetName
                }"#,
            )
            .await
            .unwrap();
        assert_eq!(result.data.len(), 1, "{:?}", result.data);
        assert_eq!(
            result.data[0].get("subject"),
            Some(&Value::String("_:a".to_string())),
            "空白节点应保留 `_:` 形式：{:?}",
            result.data
        );
        assert_eq!(
            result.data[0].get("targetName"),
            Some(&Value::String("匿名乙".to_string()))
        );

        let replay = manager.import_jsonld(jsonld).await.unwrap();
        assert_eq!(replay.triples, 0, "重复导入不应新增三元组");
    }

    /// JSON-LD @reverse 与 @set：反向边方向正确，集合值逐项展开。
    #[tokio::test]
    async fn import_jsonld_supports_reverse_and_set_values() {
        let (manager, _dir) = temp_manager().await;
        let jsonld = r#"{
            "@context": {
                "name": "http://example.org/kg/name",
                "nick": "http://example.org/kg/nick",
                "parent_of": "http://example.org/kg/parent_of",
                "knows": "http://example.org/kg/knows"
            },
            "@id": "http://example.org/kg/entity/甲",
            "name": "甲",
            "nick": { "@set": ["阿甲", "甲哥"] },
            "@reverse": {
                "parent_of": { "@id": "http://example.org/kg/entity/乙" },
                "knows": {
                    "@id": "http://example.org/kg/entity/丙",
                    "name": "丙"
                }
            }
        }"#;

        let stats = manager.import_jsonld(jsonld).await.unwrap();
        assert_eq!(stats.entities, 2, "甲与带属性的反向节点丙");
        assert_eq!(stats.relations, 2, "两条反向边");
        assert_eq!(stats.triples, 6, "name 2 + nick 2 + 反向边 2");

        let result = manager
            .query_sparql(
                r#"PREFIX kg: <http://example.org/kg/>
                SELECT ?s ?p ?o WHERE {
                    ?s ?p ?o .
                    FILTER(?p IN (kg:parent_of, kg:knows))
                }"#,
            )
            .await
            .unwrap();
        let pairs: Vec<(String, String)> = result
            .data
            .iter()
            .map(|row| {
                (
                    match row.get("s") {
                        Some(Value::String(text)) => text.clone(),
                        other => format!("{:?}", other),
                    },
                    match row.get("o") {
                        Some(Value::String(text)) => text.clone(),
                        other => format!("{:?}", other),
                    },
                )
            })
            .collect();
        assert!(
            pairs.contains(&(
                "http://example.org/kg/entity/乙".to_string(),
                "http://example.org/kg/entity/甲".to_string()
            )),
            "反向边应为 乙 parent_of 甲：{:?}",
            pairs
        );
        assert!(
            pairs.contains(&(
                "http://example.org/kg/entity/丙".to_string(),
                "http://example.org/kg/entity/甲".to_string()
            )),
            "反向边应为 丙 knows 甲：{:?}",
            pairs
        );

        let nicks = manager
            .query_sparql(
                r#"PREFIX kg: <http://example.org/kg/>
                SELECT ?nick WHERE {
                    <http://example.org/kg/entity/甲> kg:nick ?nick
                }"#,
            )
            .await
            .unwrap();
        assert_eq!(nicks.data.len(), 2, "{:?}", nicks.data);
        assert_eq!(
            manager.import_jsonld(jsonld).await.unwrap().triples,
            0,
            "重复导入保持幂等"
        );
    }

    /// JSON-LD @list：编码为 rdf:first/rdf:rest 链，空表指向 rdf:nil，重复导入幂等。
    #[tokio::test]
    async fn import_jsonld_supports_rdf_lists() {
        let (manager, _dir) = temp_manager().await;
        let jsonld = r#"{
            "@context": {
                "items": "http://example.org/kg/items",
                "friend": "http://example.org/kg/friend"
            },
            "@id": "http://example.org/kg/entity/甲",
            "items": { "@list": ["一", "二"] },
            "friend": { "@list": [] }
        }"#;

        let stats = manager.import_jsonld(jsonld).await.unwrap();
        assert_eq!(stats.triples, 6, "非空表 1+4 条，空表 1 条");

        let firsts = manager
            .query_sparql(
                r#"PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                SELECT ?value WHERE { ?cell rdf:first ?value }"#,
            )
            .await
            .unwrap();
        let mut values: Vec<String> = firsts
            .data
            .iter()
            .filter_map(|row| match row.get("value") {
                Some(Value::String(text)) => Some(text.clone()),
                _ => None,
            })
            .collect();
        values.sort();
        assert_eq!(values, vec!["一".to_string(), "二".to_string()]);

        let empty = manager
            .query_sparql(
                r#"PREFIX kg: <http://example.org/kg/>
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                SELECT ?tail WHERE {
                    <http://example.org/kg/entity/甲> kg:friend ?tail .
                    FILTER(?tail = rdf:nil)
                }"#,
            )
            .await
            .unwrap();
        assert_eq!(empty.data.len(), 1, "{:?}", empty.data);

        assert_eq!(
            manager.import_jsonld(jsonld).await.unwrap().triples,
            0,
            "集合单元格按内容命名，重复导入不应新增三元组"
        );
    }

    /// 嵌套 @graph 与 @included：内容展开到默认图，纯图容器不生成实体。
    #[tokio::test]
    async fn import_jsonld_supports_nested_graphs_and_included() {
        let (manager, _dir) = temp_manager().await;
        let jsonld = r#"{
            "@context": { "name": "http://example.org/kg/name" },
            "@graph": [
                {
                    "@id": "http://example.org/kg/graph/outer",
                    "@graph": [
                        { "@id": "http://example.org/kg/entity/甲", "name": "甲" }
                    ]
                },
                {
                    "@id": "http://example.org/kg/entity/乙",
                    "name": "乙",
                    "@included": [
                        { "@id": "http://example.org/kg/entity/丙", "name": "丙" }
                    ]
                }
            ]
        }"#;

        let stats = manager.import_jsonld(jsonld).await.unwrap();
        assert_eq!(stats.entities, 3, "纯图容器不计实体，只统计甲乙丙");
        assert_eq!(stats.triples, 3);

        let names = manager
            .query_sparql(
                r#"PREFIX kg: <http://example.org/kg/>
                SELECT ?name WHERE { ?entity kg:name ?name }"#,
            )
            .await
            .unwrap();
        assert_eq!(names.data.len(), 3, "{:?}", names.data);
        assert_eq!(
            manager.import_jsonld(jsonld).await.unwrap().triples,
            0,
            "重复导入保持幂等"
        );
    }

    /// 节点级 @context 只作用于该节点：嵌套节点可用不同词项映射。
    #[tokio::test]
    async fn import_jsonld_respects_scoped_context() {
        let (manager, _dir) = temp_manager().await;
        let jsonld = r#"{
            "@context": {
                "ex": "http://example.org/kg/",
                "name": "ex:name"
            },
            "@id": "http://example.org/kg/entity/甲",
            "ex:parent": {
                "@context": { "name": "http://example.org/kg/childName" },
                "@id": "http://example.org/kg/entity/乙",
                "name": "乙"
            }
        }"#;

        manager.import_jsonld(jsonld).await.unwrap();
        let result = manager
            .query_sparql(
                r#"PREFIX kg: <http://example.org/kg/>
                SELECT ?childName WHERE {
                    <http://example.org/kg/entity/乙> kg:childName ?childName
                }"#,
            )
            .await
            .unwrap();
        assert_eq!(
            result.data[0].get("childName"),
            Some(&Value::String("乙".to_string()))
        );
    }

    /// 任意 SPARQL SELECT：变量绑定转 JSON，带类型字面量映射为数字/布尔。
    #[tokio::test]
    async fn sparql_select_maps_typed_literals_to_json() {
        let (manager, _dir) = temp_manager().await;
        let jsonld = r#"{
            "@context": {
                "name": "http://example.org/kg/name",
                "age": "http://example.org/kg/age",
                "Person": "http://example.org/kg/Person"
            },
            "@graph": [
                {
                    "@id": "http://example.org/kg/entity/小明",
                    "@type": "Person",
                    "name": "小明",
                    "age": {"@value": "18", "@type": "http://www.w3.org/2001/XMLSchema#integer"}
                }
            ]
        }"#;
        manager.import_jsonld(jsonld).await.unwrap();

        let result = manager
            .query_sparql(
                r#"PREFIX kg: <http://example.org/kg/>
                SELECT ?name ?age WHERE { ?s kg:name ?name . ?s kg:age ?age }"#,
            )
            .await
            .unwrap();

        assert!(result.success);
        assert_eq!(result.data.len(), 1);
        assert_eq!(
            result.data[0].get("name"),
            Some(&Value::String("小明".to_string()))
        );
        assert_eq!(result.data[0].get("age"), Some(&serde_json::json!(18)));
        let rendered = result.render_text();
        assert!(
            rendered.contains("?age") || rendered.contains("age"),
            "{}",
            rendered
        );
        assert!(rendered.contains("小明"), "{}", rendered);
        assert!(rendered.contains("18"), "{}", rendered);
    }

    /// ASK 返回确定布尔；CONSTRUCT 按三元组逐行返回；语法错误不静默为空。
    #[tokio::test]
    async fn sparql_ask_construct_and_syntax_error() {
        let (manager, _dir) = temp_manager().await;
        manager.save_knowledge(vec![person("张三")]).await.unwrap();

        let ask = manager
            .query_sparql(r#"ASK { ?s <http://example.org/kg/name> "张三" }"#)
            .await
            .unwrap();
        assert_eq!(ask.data.len(), 1);
        assert_eq!(ask.data[0].get("boolean"), Some(&Value::Bool(true)));

        let construct = manager
            .query_sparql(
                r#"PREFIX kg: <http://example.org/kg/>
                CONSTRUCT { ?s kg:name ?name } WHERE { ?s kg:name ?name }"#,
            )
            .await
            .unwrap();
        assert!(!construct.data.is_empty());
        for row in &construct.data {
            assert!(row.contains_key("subject"), "{:?}", row);
            assert!(row.contains_key("predicate"), "{:?}", row);
            assert!(row.contains_key("object"), "{:?}", row);
        }

        let error = manager.query_sparql("SELECT ?s WHERE { ?s ?p").await;
        assert!(error.is_err());
        let message = error.unwrap_err().to_string();
        assert!(message.contains("SPARQL 语法错误"), "{}", message);
    }

    /// 无结果的 SELECT 与空表渲染：固定措辞，不伪装成故障。
    #[tokio::test]
    async fn sparql_empty_result_and_render_are_explicit() {
        let (manager, _dir) = temp_manager().await;
        let result = manager
            .query_sparql(r#"SELECT ?s WHERE { ?s <http://example.org/kg/name> "不存在" }"#)
            .await
            .unwrap();
        assert!(result.data.is_empty());
        assert_eq!(result.render_text(), "（查询成功，无结果）");
    }

    /// RDF 导出：Turtle/JSON-LD 逐条写出且能往返导入（导入计数一致）。
    #[tokio::test]
    async fn export_rdf_roundtrips_turtle_and_jsonld() {
        let (manager, dir) = temp_manager().await;
        manager
            .save_knowledge(vec![person("张三"), person("李四")])
            .await
            .unwrap();
        manager.save_knowledge(items("张三", "李四")).await.unwrap();

        let turtle_path = dir.path().join("dump.ttl");
        let exported = manager
            .export_rdf_file(turtle_path.to_str().unwrap())
            .await
            .unwrap();
        assert!(exported >= 4, "实体类型/名字/关系至少 4 条：{}", exported);
        let turtle = std::fs::read_to_string(&turtle_path).unwrap();
        assert!(turtle.contains("张三"), "{}", turtle);

        let (restored, _other) = temp_manager().await;
        let imported = restored
            .import_rdf_file(turtle_path.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(imported, exported, "往返导入应写回同等数量三元组");
        let facts = restored.load_all_facts().await.unwrap();
        assert!(facts
            .iter()
            .any(|fact| matches!(fact, PrologFact::Entity { name, .. } if name == "张三")));

        let jsonld_path = dir.path().join("dump.jsonld");
        let jsonld_exported = manager
            .export_rdf_file(jsonld_path.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(jsonld_exported, exported);
        let text = std::fs::read_to_string(&jsonld_path).unwrap();
        assert!(
            text.contains('\u{5f20}') || text.contains("\\\\u5f20"),
            "JSON-LD 应包含实体文本：{}",
            text
        );
    }

    /// 更正关系值：删旧宾语（实体 IRI）并写新宾语，同时补 kg:name，
    /// 保证推理快照仍能把新宾语识别为关系端点。
    #[tokio::test]
    async fn correct_object_value_replaces_entity_and_keeps_relation_visible() {
        let (manager, _dir) = temp_manager().await;
        manager
            .save_knowledge(vec![person("姚明"), person("226")])
            .await
            .unwrap();
        manager.save_knowledge(items("姚明", "226")).await.unwrap();
        // items() 的谓词是「师傅」，这里直接改用身高关系
        manager
            .save_knowledge(vec![KnowledgeItem::Relationship {
                from_label: "Person".to_string(),
                from_id: "姚明".to_string(),
                to_label: "Person".to_string(),
                to_id: "226".to_string(),
                rel_type: "身高".to_string(),
                properties: HashMap::new(),
            }])
            .await
            .unwrap();
        let (removed, inserted) = manager
            .correct_object_value("姚明", "身高", "226", "229")
            .await
            .unwrap();
        assert_eq!(removed, 1);
        assert!(inserted);
        let facts = manager.load_graph_snapshot().await.unwrap();
        assert!(
            facts.iter().any(|fact| matches!(
                fact,
                PrologFact::Relation { from, rel_type, to }
                    if from == "姚明" && rel_type == "身高" && to == "229"
            )),
            "新关系应可见：{:?}",
            facts
        );
        assert!(
            !facts.iter().any(|fact| matches!(
                fact,
                PrologFact::Relation { from, rel_type, to }
                    if from == "姚明" && rel_type == "身高" && to == "226"
            )),
            "旧关系不应残留：{:?}",
            facts
        );
        // 旧值不存在时拒绝「更正」：不能变成无条件覆盖
        let (removed_again, inserted_again) = manager
            .correct_object_value("姚明", "身高", "226", "230")
            .await
            .unwrap();
        assert_eq!((removed_again, inserted_again), (0, false));
    }

    /// 未知扩展名拒绝导出，绝不猜测格式。
    #[tokio::test]
    async fn export_rdf_rejects_unknown_extension() {
        let (manager, dir) = temp_manager().await;
        let path = dir.path().join("dump.unknown");
        let error = manager.export_rdf_file(path.to_str().unwrap()).await;
        assert!(error.is_err());
        assert!(error.unwrap_err().to_string().contains("识别"));
    }

    /// JSON-LD 原生标量必须保留 XSD 类型：数字/布尔不是字符串，`null` 不落库。
    /// 否则 `SELECT ?age` 只会得到 `"25"`，下游数值比较与类型化查询全部失效。
    #[tokio::test]
    async fn import_jsonld_types_native_scalars_and_skips_null() {
        let (manager, _dir) = temp_manager().await;
        let jsonld = r#"{
            "@context": {
                "ex": "http://example.org/kg/",
                "@vocab": "http://example.org/kg/",
                "name": "ex:name",
                "age": "ex:age",
                "score": "ex:score",
                "active": "ex:active"
            },
            "@id": "http://example.org/kg/entity/小明",
            "@type": "Person",
            "name": "小明",
            "age": 25,
            "score": 1.5,
            "active": true,
            "nickname": null,
            "height": { "@value": 180 },
            "retired": { "@value": false }
        }"#;

        let stats = manager.import_jsonld(jsonld).await.unwrap();
        assert_eq!(stats.entities, 1);
        assert_eq!(stats.relations, 0);
        assert_eq!(stats.triples, 7, "类型 1 + 六个非空属性");

        let result = manager
            .query_sparql(
                r#"PREFIX kg: <http://example.org/kg/>
                SELECT ?age ?score ?active ?height ?retired WHERE {
                    <http://example.org/kg/entity/小明>
                        kg:age ?age ;
                        kg:score ?score ;
                        kg:active ?active ;
                        kg:height ?height ;
                        kg:retired ?retired
                }"#,
            )
            .await
            .unwrap();
        assert_eq!(result.data.len(), 1, "{:?}", result.data);
        assert_eq!(result.data[0].get("age"), Some(&serde_json::json!(25)));
        assert_eq!(result.data[0].get("score"), Some(&serde_json::json!(1.5)));
        assert_eq!(result.data[0].get("active"), Some(&serde_json::json!(true)));
        assert_eq!(result.data[0].get("height"), Some(&serde_json::json!(180)));
        assert_eq!(
            result.data[0].get("retired"),
            Some(&serde_json::json!(false))
        );

        assert_eq!(
            manager.import_jsonld(jsonld).await.unwrap().triples,
            0,
            "重复导入保持幂等"
        );
    }

    /// RDF 批量导入把断言双写进默认图与批次图，并在元数据图留下可回指的情景。
    #[tokio::test]
    async fn import_rdf_batch_records_episode_and_batch_graph() {
        let (manager, _dir) = temp_manager().await;
        let nquads = concat!(
            "<http://example.org/kg/entity/甲> <http://example.org/kg/name> \"甲\" <http://example.org/kg/g1> .\n",
            "<http://example.org/kg/entity/乙> <http://example.org/kg/name> \"乙\" <http://example.org/kg/g2> .\n",
        );
        let first = manager
            .import_rdf_batch(
                oxigraph::io::RdfFormat::NQuads,
                nquads.as_bytes(),
                "data/sample.nq",
            )
            .await
            .unwrap();
        assert!(first.recorded, "首次导入应新建批次");
        assert_eq!(first.added, 2);
        assert_eq!(first.episode.seq, 1);
        assert_eq!(first.episode.total, 2);
        assert_eq!(first.episode.input_named_graphs, 2);
        assert_eq!(first.episode.gist.distinct_subjects, 2);
        assert!(!first.episode.blank_nodes);
        let batch_graph = first.episode.batch_graph.clone();

        let default = manager
            .query_sparql("SELECT ?s WHERE { ?s ?p ?o }")
            .await
            .unwrap();
        assert_eq!(default.data.len(), 2, "默认图仍是主视图");
        let batch = manager
            .query_sparql(&format!(
                "SELECT ?s WHERE {{ GRAPH <{}> {{ ?s ?p ?o }} }}",
                batch_graph
            ))
            .await
            .unwrap();
        assert_eq!(batch.data.len(), 2, "批次命名图保存本批断言");

        let replay = manager
            .import_rdf_batch(
                oxigraph::io::RdfFormat::NQuads,
                nquads.as_bytes(),
                "data/sample.nq",
            )
            .await
            .unwrap();
        assert!(!replay.recorded, "同内容重复导入不新建批次");
        assert_eq!(replay.added, 0);
        assert_eq!(replay.duplicates, 2);
        assert_eq!(replay.episode.id, first.episode.id);
        assert_eq!(manager.load_import_episodes().await.unwrap().len(), 1);

        let second = manager
            .import_rdf_batch(
                oxigraph::io::RdfFormat::Turtle,
                "<http://example.org/kg/entity/丙> <http://example.org/kg/name> \"丙\" ."
                    .as_bytes(),
                "data/second.ttl",
            )
            .await
            .unwrap();
        assert_eq!(second.episode.seq, 2);
        let episodes = manager.load_import_episodes().await.unwrap();
        assert_eq!(episodes.len(), 2);
        assert_eq!(
            crate::import_episode::resolve_episode_reference("上一批", &episodes)
                .map(|episode| episode.seq),
            Some(1)
        );
        assert_eq!(
            crate::import_episode::resolve_episode_reference("第2批", &episodes)
                .map(|episode| episode.seq),
            Some(2)
        );
        assert_eq!(
            crate::import_episode::resolve_episode_reference("second.ttl 讲了什么", &episodes)
                .map(|episode| episode.seq),
            Some(2)
        );
    }

    /// 批次问题只在出现批次线索时回答；类型名单与单主体事实来自批次命名图。
    #[tokio::test]
    async fn import_episode_answers_are_scoped_to_the_batch_graph() {
        let (manager, _dir) = temp_manager().await;
        let turtle = concat!(
            "@prefix kg: <http://example.org/kg/> .\n",
            "<http://example.org/kg/entity/张三> a kg:Person ; kg:name \"张三\" ; kg:age \"30\" .\n",
            "<http://example.org/kg/entity/李四> a kg:Person ; kg:name \"李四\" ; kg:knows <http://example.org/kg/entity/张三> .\n",
            "<http://example.org/kg/entity/北京> a kg:Place ; kg:name \"北京\" .\n",
        );
        manager
            .import_rdf_batch(
                oxigraph::io::RdfFormat::Turtle,
                turtle.as_bytes(),
                "data/people.ttl",
            )
            .await
            .unwrap();
        let episodes = manager.load_import_episodes().await.unwrap();
        assert_eq!(episodes.len(), 1);

        let people = manager
            .answer_import_episode_question("这批里有哪些人物", &episodes)
            .await
            .unwrap()
            .expect("批次问题应有回答");
        assert!(people.contains("张三"), "{}", people);
        assert!(people.contains("李四"), "{}", people);
        assert!(!people.contains("北京"), "{}", people);

        let facts = manager
            .answer_import_episode_question("这批里关于张三有哪些事实", &episodes)
            .await
            .unwrap()
            .expect("批次问题应有回答");
        assert!(facts.contains("张三"), "{}", facts);
        assert!(facts.contains("李四 knows 张三"), "{}", facts);
        assert!(facts.contains("[B1-"), "{}", facts);

        let implicit_facts = manager
            .answer_import_episode_question("这批里张三的关系", &episodes)
            .await
            .unwrap()
            .expect("批次问题应有回答");
        assert!(implicit_facts.contains("李四 knows 张三"), "{}", implicit_facts);

        let missing_subject = manager
            .answer_import_episode_question("这批里关于王五有哪些事实", &episodes)
            .await
            .unwrap()
            .expect("批次问题应有回答");
        assert!(missing_subject.contains("没有找到"), "{}", missing_subject);

        let missing_type = manager
            .answer_import_episode_question("这批里有哪些公司", &episodes)
            .await
            .unwrap()
            .expect("批次问题应有回答");
        assert!(missing_type.contains("没有匹配"), "{}", missing_type);

        let count = manager
            .answer_import_episode_question("这批有多少条断言", &episodes)
            .await
            .unwrap()
            .unwrap();
        assert!(count.contains("批次 B1 共 8 条断言"), "{}", count);

        let types = manager
            .answer_import_episode_question("这批有哪些类型", &episodes)
            .await
            .unwrap()
            .unwrap();
        assert!(types.contains("Person×2"), "{}", types);
        assert!(types.contains("Place×1"), "{}", types);

        let source = manager
            .answer_import_episode_question("这批的来源是什么", &episodes)
            .await
            .unwrap()
            .unwrap();
        assert!(source.contains("data/people.ttl"), "{}", source);

        let changes = manager
            .answer_import_episode_question("这批新增多少、重复多少", &episodes)
            .await
            .unwrap()
            .unwrap();
        assert!(changes.contains("新增 8 条"), "{}", changes);

        assert!(manager
            .answer_import_episode_question("张三多大", &episodes)
            .await
            .unwrap()
            .is_none());
        assert!(manager
            .answer_import_episode_question("这批讲了什么", &[])
            .await
            .unwrap()
            .is_none());
    }

    /// 空白节点是内容寻址的例外：回执必须披露，且解析失败不留下批次。
    #[tokio::test]
    async fn import_rdf_batch_discloses_blank_nodes_and_parse_failures() {
        let (manager, _dir) = temp_manager().await;
        let turtle = "@prefix kg: <http://example.org/kg/> .\n[] kg:name \"匿名\" .\n";
        let first = manager
            .import_rdf_batch(
                oxigraph::io::RdfFormat::Turtle,
                turtle.as_bytes(),
                "blank.ttl",
            )
            .await
            .unwrap();
        assert!(first.episode.blank_nodes);
        assert_eq!(first.added, 1);

        let second = manager
            .import_rdf_batch(
                oxigraph::io::RdfFormat::Turtle,
                turtle.as_bytes(),
                "blank.ttl",
            )
            .await
            .unwrap();
        assert!(second.episode.blank_nodes);
        assert_ne!(
            second.episode.id, first.episode.id,
            "空白节点重命名使同文档批次 id 不稳定（已知例外）"
        );

        let before = manager.load_import_episodes().await.unwrap().len();
        let broken = manager
            .import_rdf_batch(
                oxigraph::io::RdfFormat::Turtle,
                b"<http://example.org/kg/s> <http://example.org/kg/p> .".as_slice(),
                "broken.ttl",
            )
            .await;
        assert!(broken.is_err());
        assert_eq!(manager.load_import_episodes().await.unwrap().len(), before);
    }

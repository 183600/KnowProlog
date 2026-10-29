fn local_person(name: &str) -> KnowledgeItem {
    KnowledgeItem::Entity {
        label: "Person".to_string(),
        properties: HashMap::from([("name".to_string(), name.to_string())]),
    }
}

fn local_relation(from: &str, to: &str) -> KnowledgeItem {
    KnowledgeItem::Relationship {
        from_label: "Person".to_string(),
        from_id: from.to_string(),
        to_label: "Person".to_string(),
        to_id: to.to_string(),
        rel_type: "师傅".to_string(),
        properties: HashMap::new(),
    }
}

async fn local_manager() -> (OxigraphManager, tempfile::TempDir) {
    let dir = tempfile::TempDir::new().unwrap();
    let manager = OxigraphManager::with_perf_ledger(
        dir.path().to_str().unwrap(),
        Arc::new(PerfLedger::new()),
    )
    .unwrap();
    (manager, dir)
}

#[tokio::test]
async fn relevant_facts_cache_survives_unrelated_addition() {
    let (manager, _dir) = local_manager().await;
    manager
        .save_knowledge(vec![local_person("张三"), local_person("李四")])
        .await
        .unwrap();
    manager
        .save_knowledge(vec![local_relation("张三", "李四")])
        .await
        .unwrap();
    let perf = manager.perf_ledger();

    let first = manager
        .get_relevant_facts(&["张三".to_string()], 1)
        .await
        .unwrap();
    assert!(!first.is_empty());
    let scans = perf.count("relevant_facts_graph_scan");
    let hits = perf.count("relevant_facts_cache_hit");

    manager
        .save_knowledge(vec![local_person("王五")])
        .await
        .unwrap();
    let second = manager
        .get_relevant_facts(&["张三".to_string()], 1)
        .await
        .unwrap();

    assert_eq!(first, second, "无关实体入库不应改变已缓存的子图");
    assert_eq!(
        perf.count("relevant_facts_graph_scan"),
        scans,
        "无关增量应保留缓存，而不是整图重扫"
    );
    assert_eq!(perf.count("relevant_facts_cache_hit"), hits + 1);
    assert!(
        perf.count("relevant_cache_delta_retained") >= 1,
        "保留决定应计入增量台账"
    );
}

#[tokio::test]
async fn relevant_facts_cache_recomputes_after_closure_touch() {
    let (manager, _dir) = local_manager().await;
    manager
        .save_knowledge(vec![local_person("张三"), local_person("李四")])
        .await
        .unwrap();
    manager
        .save_knowledge(vec![local_relation("张三", "李四")])
        .await
        .unwrap();
    let perf = manager.perf_ledger();

    let first = manager
        .get_relevant_facts(&["张三".to_string()], 2)
        .await
        .unwrap();
    let scans = perf.count("relevant_facts_graph_scan");
    assert!(!first.is_empty());

    manager
        .save_knowledge(vec![local_person("王五"), local_relation("李四", "王五")])
        .await
        .unwrap();
    let second = manager
        .get_relevant_facts(&["张三".to_string()], 2)
        .await
        .unwrap();

    assert_ne!(first, second, "闭包节点上的新关系必须进入重算结果");
    assert!(
        perf.count("relevant_facts_graph_scan") > scans,
        "触碰闭包后必须重新扫描"
    );
    assert!(
        second.iter().any(|fact| matches!(
            fact,
            PrologFact::Relation { from, to, .. } if from == "李四" && to == "王五"
        )),
        "重算结果应包含新关系：{:?}",
        second
    );
}

fn local_city(name: &str, capital: &str) -> KnowledgeItem {
    KnowledgeItem::Entity {
        label: "City".to_string(),
        properties: HashMap::from([
            ("name".to_string(), name.to_string()),
            ("capital".to_string(), capital.to_string()),
        ]),
    }
}

#[tokio::test]
async fn direct_answer_cache_survives_unrelated_addition() {
    let (manager, _dir) = local_manager().await;
    manager
        .save_knowledge(vec![local_city("北京", "中国")])
        .await
        .unwrap();
    let perf = manager.perf_ledger();
    let entities = vec!["北京".to_string()];

    let first = manager.get_direct_answer("capital_query", &entities).await;
    assert_eq!(first.as_deref(), Some("北京的首都是中国"));
    let queries = perf.count("direct_answer_query");
    let hits = perf.count("direct_answer_cache_hit");

    manager
        .save_knowledge(vec![local_person("王五")])
        .await
        .unwrap();
    let second = manager.get_direct_answer("capital_query", &entities).await;

    assert_eq!(first, second, "无关实体入库不应改变直答");
    assert_eq!(perf.count("direct_answer_query"), queries);
    assert_eq!(perf.count("direct_answer_cache_hit"), hits + 1);
    assert!(
        perf.count("direct_answer_cache_delta_retained") >= 1,
        "直答保留决定应计入增量台账"
    );
}

#[tokio::test]
async fn direct_answer_negative_cache_recomputes_when_entity_gains_property() {
    let (manager, _dir) = local_manager().await;
    manager
        .save_knowledge(vec![local_city("北京", "中国")])
        .await
        .unwrap();
    let perf = manager.perf_ledger();
    let entities = vec!["北京".to_string()];

    assert!(manager
        .get_direct_answer("author_query", &entities)
        .await
        .is_none());
    let queries = perf.count("direct_answer_query");
    assert!(manager
        .get_direct_answer("author_query", &entities)
        .await
        .is_none());
    assert_eq!(
        perf.count("direct_answer_query"),
        queries,
        "同版本负缓存应命中"
    );

    manager
        .save_knowledge(vec![KnowledgeItem::Entity {
            label: "City".to_string(),
            properties: HashMap::from([
                ("name".to_string(), "北京".to_string()),
                ("author".to_string(), "曹雪芹".to_string()),
            ]),
        }])
        .await
        .unwrap();
    let answer = manager.get_direct_answer("author_query", &entities).await;
    assert_eq!(answer.as_deref(), Some("北京的作者是曹雪芹"));
    assert!(perf.count("direct_answer_query") > queries);
}

#[tokio::test]
async fn relevant_facts_empty_result_recomputes_when_query_entity_arrives() {
    let (manager, _dir) = local_manager().await;
    manager
        .save_knowledge(vec![local_person("李四")])
        .await
        .unwrap();
    let perf = manager.perf_ledger();
    let query = vec!["张三".to_string()];

    assert!(
        manager
            .get_relevant_facts(&query, 1)
            .await
            .unwrap()
            .is_empty()
    );
    let scans = perf.count("relevant_facts_graph_scan");
    assert!(
        manager
            .get_relevant_facts(&query, 1)
            .await
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        perf.count("relevant_facts_graph_scan"),
        scans,
        "同版本空结果也应命中缓存"
    );

    manager
        .save_knowledge(vec![local_person("张三")])
        .await
        .unwrap();
    let facts = manager.get_relevant_facts(&query, 1).await.unwrap();

    assert!(!facts.is_empty(), "查询模式新命中时必须重算空缓存");
    assert!(
        perf.count("relevant_facts_graph_scan") > scans,
        "模式命中应触发重新扫描"
    );
    assert!(
        facts
            .iter()
            .any(|fact| matches!(fact, PrologFact::Entity { name, .. } if name == "张三")),
        "重算结果应包含新入库实体：{:?}",
        facts
    );
}

#[tokio::test]
async fn relevant_facts_cache_survives_unrelated_undo() {
    let (manager, _dir) = local_manager().await;
    manager
        .save_knowledge(vec![local_person("张三"), local_person("李四")])
        .await
        .unwrap();
    manager
        .save_knowledge(vec![local_relation("张三", "李四")])
        .await
        .unwrap();
    let receipt = manager
        .save_knowledge_with_receipt(vec![local_person("王五")])
        .await
        .unwrap();
    let perf = manager.perf_ledger();
    let query = vec!["张三".to_string()];

    let first = manager.get_relevant_facts(&query, 1).await.unwrap();
    let scans = perf.count("relevant_facts_graph_scan");
    let retained = perf.count("relevant_cache_delta_retained");
    assert!(manager.undo_import(&receipt).await.unwrap() > 0);
    let second = manager.get_relevant_facts(&query, 1).await.unwrap();

    assert_eq!(first, second, "无关撤回不应改变已缓存的子图");
    assert_eq!(
        perf.count("relevant_facts_graph_scan"),
        scans,
        "无关撤回应保留缓存，而不是整图重扫"
    );
    assert!(
        perf.count("relevant_cache_delta_retained") > retained,
        "无关撤回应计入保留台账"
    );
}

#[tokio::test]
async fn relevant_facts_cache_recomputes_after_removal_in_closure() {
    let (manager, _dir) = local_manager().await;
    manager
        .save_knowledge(vec![local_person("张三"), local_person("李四")])
        .await
        .unwrap();
    let receipt = manager
        .save_knowledge_with_receipt(vec![local_relation("张三", "李四")])
        .await
        .unwrap();
    let perf = manager.perf_ledger();
    let query = vec!["张三".to_string()];

    let first = manager.get_relevant_facts(&query, 2).await.unwrap();
    assert!(first
        .iter()
        .any(|fact| matches!(fact, PrologFact::Relation { .. })));
    let scans = perf.count("relevant_facts_graph_scan");
    assert!(manager.undo_import(&receipt).await.unwrap() > 0);
    let second = manager.get_relevant_facts(&query, 2).await.unwrap();

    assert!(
        perf.count("relevant_facts_graph_scan") > scans,
        "闭包内关系被撤回后必须重新扫描"
    );
    assert!(
        !second
            .iter()
            .any(|fact| matches!(fact, PrologFact::Relation { .. })),
        "重算结果不应保留已撤回的关系：{:?}",
        second
    );
}

#[tokio::test]
async fn direct_answer_cache_recomputes_after_entity_undo() {
    let (manager, _dir) = local_manager().await;
    let receipt = manager
        .save_knowledge_with_receipt(vec![local_city("北京", "中国")])
        .await
        .unwrap();
    let perf = manager.perf_ledger();
    let entities = vec!["北京".to_string()];

    let first = manager.get_direct_answer("capital_query", &entities).await;
    assert_eq!(first.as_deref(), Some("北京的首都是中国"));
    let queries = perf.count("direct_answer_query");
    assert!(manager.undo_import(&receipt).await.unwrap() > 0);
    let second = manager.get_direct_answer("capital_query", &entities).await;

    assert!(second.is_none(), "实体撤回后直答必须失效");
    assert!(
        perf.count("direct_answer_query") > queries,
        "命中撤回足迹的直答必须重算"
    );
}

#[tokio::test]
async fn relevant_facts_cache_survives_unrelated_batch_retraction() {
    let (manager, _dir) = local_manager().await;
    manager
        .save_knowledge(vec![local_person("张三"), local_person("李四")])
        .await
        .unwrap();
    manager
        .save_knowledge(vec![local_relation("张三", "李四")])
        .await
        .unwrap();
    let batch = manager
        .import_rdf_batch(
            oxigraph::io::RdfFormat::Turtle,
            "<http://example.org/kg/entity/王五> <http://example.org/kg/name> \"王五\" ."
                .as_bytes(),
            "unrelated.ttl",
        )
        .await
        .unwrap();
    let perf = manager.perf_ledger();
    let query = vec!["张三".to_string()];

    let first = manager.get_relevant_facts(&query, 1).await.unwrap();
    let scans = perf.count("relevant_facts_graph_scan");
    let retraction = manager.retract_import_batch(&batch.episode).await.unwrap();
    assert_eq!(retraction.removed, 1);
    let second = manager.get_relevant_facts(&query, 1).await.unwrap();

    assert_eq!(first, second, "无关批次撤回不应改变已缓存的子图");
    assert_eq!(
        perf.count("relevant_facts_graph_scan"),
        scans,
        "无关批次撤回应保留缓存，而不是整图重扫"
    );
}

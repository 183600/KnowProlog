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

    /// SELECT 的列顺序必须跟随查询声明，而不是按变量名字典序重排：
    /// `?s ?p ?o` 应渲染成 s、p、o，而不是 o、p、s。
    #[tokio::test]
    async fn sparql_select_preserves_declared_column_order() {
        let (manager, _dir) = temp_manager().await;
        manager.save_knowledge(vec![person("张三")]).await.unwrap();

        let result = manager
            .query_sparql(
                r#"PREFIX kg: <http://example.org/kg/>
                SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 1"#,
            )
            .await
            .unwrap();
        assert_eq!(result.columns, vec!["s", "p", "o"]);
        let rendered = result.render_text();
        let header: Vec<String> = rendered
            .lines()
            .next()
            .unwrap_or_default()
            .split('|')
            .map(|cell| cell.trim().to_string())
            .collect();
        assert_eq!(header, vec!["s", "p", "o"], "{}", rendered);

        let reordered = manager
            .query_sparql(
                r#"PREFIX kg: <http://example.org/kg/>
                SELECT ?o ?s WHERE { ?s kg:name ?o }"#,
            )
            .await
            .unwrap();
        assert_eq!(reordered.columns, vec!["o", "s"]);
        let header: Vec<String> = reordered
            .render_text()
            .lines()
            .next()
            .unwrap_or_default()
            .split('|')
            .map(|cell| cell.trim().to_string())
            .collect();
        assert_eq!(header, vec!["o", "s"]);
    }

    /// 手工构造且未声明列顺序的结果仍按字典序稳定渲染。
    #[test]
    fn query_result_without_declared_columns_is_deterministic() {
        let result = QueryResult {
            success: true,
            columns: Vec::new(),
            data: vec![HashMap::from([
                ("b".to_string(), Value::String("2".to_string())),
                ("a".to_string(), Value::String("1".to_string())),
            ])],
        };
        let header: Vec<String> = result
            .render_text()
            .lines()
            .next()
            .unwrap_or_default()
            .split('|')
            .map(|cell| cell.trim().to_string())
            .collect();
        assert_eq!(header, vec!["a", "b"]);
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
        assert!(!replay.source_aliases_added, "同一来源不重复记别名");
        assert_eq!(replay.added, 0);
        assert_eq!(replay.duplicates, 2);
        assert_eq!(replay.episode.id, first.episode.id);
        assert_eq!(manager.load_import_episodes().await.unwrap().len(), 1);

        let alias = manager
            .import_rdf_batch(
                oxigraph::io::RdfFormat::NQuads,
                nquads.as_bytes(),
                "data/copy.nq",
            )
            .await
            .unwrap();
        assert!(!alias.recorded, "同内容不同来源仍复用批次");
        assert!(alias.source_aliases_added, "新来源应记入批次");
        assert_eq!(alias.episode.seq, 1);
        let episodes = manager.load_import_episodes().await.unwrap();
        assert_eq!(episodes.len(), 1);
        assert_eq!(
            crate::import_episode::resolve_episode_reference("copy.nq 讲了什么", &episodes)
                .map(|episode| episode.seq),
            Some(1)
        );
        let source = manager
            .answer_import_episode_question("这批的来源是什么", &episodes)
            .await
            .unwrap()
            .unwrap();
        assert!(source.contains("data/sample.nq"), "{}", source);
        assert!(source.contains("data/copy.nq"), "{}", source);

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

    /// JSON-LD 批量导入复用同一套情景结构：默认图 + 批次图 + 元数据图。
    #[tokio::test]
    async fn import_jsonld_batch_records_episode_and_batch_graph() {
        let (manager, _dir) = temp_manager().await;
        let jsonld = r#"{
            "@context": {
                "name": "http://example.org/kg/name",
                "member_of": "http://example.org/kg/member_of",
                "Person": "http://example.org/kg/Person",
                "Organization": "http://example.org/kg/Organization"
            },
            "@graph": [
                {
                    "@id": "http://example.org/kg/entity/张三",
                    "@type": "Person",
                    "name": "张三",
                    "member_of": { "@id": "http://example.org/kg/entity/示例公司" }
                },
                {
                    "@id": "http://example.org/kg/entity/示例公司",
                    "@type": "Organization",
                    "name": "示例公司"
                }
            ]
        }"#;
        let first = manager
            .import_jsonld_batch(jsonld, "data/people.jsonld")
            .await
            .unwrap();
        assert!(first.recorded);
        assert_eq!(first.stats.entities, 2);
        assert_eq!(first.stats.relations, 1);
        assert_eq!(first.stats.triples, 5);
        assert_eq!(first.episode.seq, 1);
        assert_eq!(first.episode.format, "JSON-LD");
        assert_eq!(first.episode.total, 5);
        assert_eq!(first.episode.input_named_graphs, 0);
        assert_eq!(first.episode.gist.distinct_subjects, 2);
        assert!(!first.episode.blank_nodes);
        assert_eq!(first.episode.source, "data/people.jsonld");
        assert!(first.summary().contains("批次 B1"));

        let batch = manager
            .query_sparql(&format!(
                "SELECT ?s WHERE {{ GRAPH <{}> {{ ?s ?p ?o }} }}",
                first.episode.batch_graph
            ))
            .await
            .unwrap();
        assert_eq!(batch.data.len(), 5);
        let default = manager
            .query_sparql("SELECT ?s WHERE { ?s ?p ?o }")
            .await
            .unwrap();
        assert_eq!(default.data.len(), 5, "默认图仍是主视图");

        let replay = manager
            .import_jsonld_batch(jsonld, "data/people.jsonld")
            .await
            .unwrap();
        assert!(!replay.recorded, "同内容重复导入不新建批次");
        assert_eq!(replay.stats.triples, 0);
        assert_eq!(replay.episode.added, 5, "批次记录保留首次导入的计数");
        assert_eq!(manager.load_import_episodes().await.unwrap().len(), 1);

        let alias = manager
            .import_jsonld_batch(jsonld, "data/people-copy.jsonld")
            .await
            .unwrap();
        assert!(alias.source_aliases_added);
        let episodes = manager.load_import_episodes().await.unwrap();
        assert_eq!(episodes.len(), 1);
        assert_eq!(
            crate::import_episode::resolve_episode_reference(
                "people-copy.jsonld 讲了什么",
                &episodes
            )
            .map(|episode| episode.seq),
            Some(1)
        );

        let before = manager.load_import_episodes().await.unwrap().len();
        let broken = manager
            .import_jsonld_batch(r#"{"@id": "http://example.org/kg/entity/坏","#, "broken.jsonld")
            .await;
        assert!(broken.is_err());
        assert_eq!(
            manager.load_import_episodes().await.unwrap().len(),
            before,
            "结构化解析失败不留批次"
        );
    }

    /// 没有 @id 的 JSON-LD 节点必须确定性命名：重复导入同一文档保持幂等。
    #[tokio::test]
    async fn import_jsonld_anonymous_nodes_are_deterministic() {
        let (manager, _dir) = temp_manager().await;
        let jsonld = r#"{
            "@context": {
                "name": "http://example.org/kg/name",
                "Person": "http://example.org/kg/Person"
            },
            "@graph": [
                { "@type": "Person", "name": "匿名甲" },
                { "@type": "Person", "name": "匿名乙" }
            ]
        }"#;
        let first = manager.import_jsonld_batch(jsonld, "anon.jsonld").await.unwrap();
        assert_eq!(first.stats.triples, 4);
        assert_eq!(first.episode.total, 4);
        let replay = manager
            .import_jsonld_batch(jsonld, "anon.jsonld")
            .await
            .unwrap();
        assert_eq!(replay.stats.triples, 0, "匿名节点不得因随机命名而重复新增");
        assert!(!replay.recorded);
        assert_eq!(replay.episode.id, first.episode.id);
        assert_eq!(
            manager.load_import_episodes().await.unwrap().len(),
            1,
            "同内容匿名节点文档只应记一个批次"
        );
    }

    /// 普通 JSON-LD 导入保持原统计口径，且暂存失败不写入半批三元组。
    #[tokio::test]
    async fn import_jsonld_stages_before_writing() {
        let (manager, _dir) = temp_manager().await;
        let jsonld = r#"{
            "@context": {"name": "http://example.org/kg/name"},
            "@id": "http://example.org/kg/entity/甲",
            "name": "甲"
        }"#;
        let stats = manager.import_jsonld(jsonld).await.unwrap();
        assert_eq!(stats.triples, 1);
        assert_eq!(stats.entities, 1);
        assert!(manager.load_import_episodes().await.unwrap().is_empty());

        let broken = manager
            .import_jsonld(r#"{"@id": "http://example.org/kg/entity/乙","#)
            .await;
        assert!(broken.is_err());
        let count = manager
            .query_sparql("SELECT ?s WHERE { ?s ?p ?o }")
            .await
            .unwrap();
        assert_eq!(count.data.len(), 1, "失败导入不留下半批三元组");
    }

    /// 批次导出只包含该批断言：数据集格式保留批次图名，图格式压平到默认图。
    #[tokio::test]
    async fn export_import_batch_is_scoped_to_the_batch_graph() {
        let (manager, _dir) = temp_manager().await;
        let first = manager
            .import_rdf_batch(
                oxigraph::io::RdfFormat::Turtle,
                "<http://example.org/kg/entity/甲> <http://example.org/kg/name> \"甲\" ."
                    .as_bytes(),
                "first.ttl",
            )
            .await
            .unwrap();
        manager
            .import_rdf_batch(
                oxigraph::io::RdfFormat::Turtle,
                "<http://example.org/kg/entity/乙> <http://example.org/kg/name> \"乙\" ."
                    .as_bytes(),
                "second.ttl",
            )
            .await
            .unwrap();

        let mut nquads = Vec::new();
        let count = manager
            .export_import_batch_to_writer(
                &first.episode,
                oxigraph::io::RdfFormat::NQuads,
                &mut nquads,
            )
            .await
            .unwrap();
        assert_eq!(count, 1);
        let exported: Vec<Quad> = oxigraph::io::RdfParser::from_format(oxigraph::io::RdfFormat::NQuads)
            .for_reader(nquads.as_slice())
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(exported.len(), 1);
        assert_eq!(
            exported[0].graph_name,
            GraphName::NamedNode(NamedNode::new(&first.episode.batch_graph).unwrap())
        );
        assert!(exported[0].subject.to_string().contains("甲"));

        let mut turtle = Vec::new();
        let count = manager
            .export_import_batch_to_writer(
                &first.episode,
                oxigraph::io::RdfFormat::Turtle,
                &mut turtle,
            )
            .await
            .unwrap();
        assert_eq!(count, 1);
        let exported: Vec<Quad> = oxigraph::io::RdfParser::from_format(oxigraph::io::RdfFormat::Turtle)
            .for_reader(turtle.as_slice())
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(exported.len(), 1);
        assert_eq!(exported[0].graph_name, GraphName::DefaultGraph);
        assert!(exported[0].object.to_string().contains("甲"));
    }

    /// 批次撤销只回撤本批首次新增；共享断言在最后一个持有者撤销时才移除。
    #[tokio::test]
    async fn retract_import_batch_respects_shared_assertions() {
        let (manager, _dir) = temp_manager().await;
        let first = manager
            .import_rdf_batch(
                oxigraph::io::RdfFormat::Turtle,
                "<http://example.org/kg/entity/甲> <http://example.org/kg/name> \"甲\" .
                 <http://example.org/kg/entity/共同> <http://example.org/kg/name> \"共同\" ."
                    .as_bytes(),
                "first.ttl",
            )
            .await
            .unwrap();
        let second = manager
            .import_rdf_batch(
                oxigraph::io::RdfFormat::Turtle,
                "<http://example.org/kg/entity/乙> <http://example.org/kg/name> \"乙\" .
                 <http://example.org/kg/entity/共同> <http://example.org/kg/name> \"共同\" ."
                    .as_bytes(),
                "second.ttl",
            )
            .await
            .unwrap();
        assert_eq!(second.added, 1);
        assert_eq!(second.duplicates, 1);

        let retraction = manager.retract_import_batch(&first.episode).await.unwrap();
        assert_eq!(retraction.removed, 1, "只移除甲");
        assert_eq!(retraction.kept_elsewhere, 1, "共同仍被第二批持有");
        assert_eq!(retraction.batch_assertions, 2);
        assert!(retraction.summary().contains("保留 1 条"));
        let default = manager
            .query_sparql("SELECT ?s WHERE { ?s ?p ?o }")
            .await
            .unwrap();
        assert_eq!(default.data.len(), 2, "{:?}", default.data);
        let batch = manager
            .query_sparql(&format!(
                "SELECT ?s WHERE {{ GRAPH <{}> {{ ?s ?p ?o }} }}",
                first.episode.batch_graph
            ))
            .await
            .unwrap();
        assert!(batch.data.is_empty(), "批次图应被移除");
        let episodes = manager.load_import_episodes().await.unwrap();
        assert_eq!(episodes.len(), 1);
        assert_eq!(episodes[0].seq, second.episode.seq);

        let retraction = manager.retract_import_batch(&second.episode).await.unwrap();
        assert_eq!(retraction.removed, 2, "最后持有者撤销后共享断言也应移除");
        assert_eq!(retraction.kept_elsewhere, 0);
        let default = manager
            .query_sparql("SELECT ?s WHERE { ?s ?p ?o }")
            .await
            .unwrap();
        assert!(default.data.is_empty(), "{:?}", default.data);
        assert!(manager.load_import_episodes().await.unwrap().is_empty());
    }

    /// 导入前已存在的手动断言不能被批次撤销误删。
    #[tokio::test]
    async fn retract_import_batch_preserves_manual_assertions() {
        let (manager, _dir) = temp_manager().await;
        manager
            .import_rdf_reader(
                oxigraph::io::RdfFormat::Turtle,
                "<http://example.org/kg/entity/甲> <http://example.org/kg/name> \"甲\" ."
                    .as_bytes(),
            )
            .await
            .unwrap();
        let batch = manager
            .import_rdf_batch(
                oxigraph::io::RdfFormat::Turtle,
                "<http://example.org/kg/entity/甲> <http://example.org/kg/name> \"甲\" .
                 <http://example.org/kg/entity/乙> <http://example.org/kg/name> \"乙\" ."
                    .as_bytes(),
                "mixed.ttl",
            )
            .await
            .unwrap();
        assert_eq!(batch.added, 1);
        assert_eq!(batch.duplicates, 1);
        let retraction = manager.retract_import_batch(&batch.episode).await.unwrap();
        assert_eq!(retraction.removed, 1, "只移除批次新增的乙");
        let default = manager
            .query_sparql("SELECT ?s WHERE { ?s ?p ?o }")
            .await
            .unwrap();
        assert_eq!(default.data.len(), 1);
        assert!(
            default.data[0]
                .get("s")
                .is_some_and(|value| value.to_string().contains("甲")),
            "{:?}",
            default.data
        );
    }

    /// 未知批次撤销返回错误，且不改变现有数据。
    #[tokio::test]
    async fn retract_import_batch_rejects_unknown_batch() {
        let (manager, _dir) = temp_manager().await;
        let episode = crate::import_episode::ImportEpisode {
            seq: 9,
            id: "missing0000".to_string(),
            source: "missing.ttl".to_string(),
            sources: Vec::new(),
            format: "Turtle".to_string(),
            imported_at: 0,
            total: 0,
            added: 0,
            duplicates: 0,
            blank_nodes: false,
            input_named_graphs: 0,
            batch_graph: crate::import_episode::batch_graph_iri("missing0000"),
            gist: crate::import_episode::BatchGist::default(),
        };
        assert!(manager.retract_import_batch(&episode).await.is_err());
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

    /// 同义词图账本：命名图往返一致、幂等重写、不推进知识版本。
    #[tokio::test]
    async fn synonym_graph_ledger_roundtrip_is_isolated() {
        use crate::synonym_graph::{
            SynonymContext, SynonymGraph, SynonymKind, SynonymSense,
        };

        let (manager, _dir) = temp_manager().await;
        let revision = manager.revision();
        let mut graph = SynonymGraph::new(8);
        let added = graph.observe(
            1,
            "老婆",
            "妻子",
            SynonymKind::Entity,
            SynonymSense::type_only("人物"),
            "test",
            1.0,
        );
        let id = added.id().expect("应写入条目").to_string();
        assert!(graph.confirm(&id, 1));
        assert_eq!(manager.sync_synonym_graph(&graph).await.unwrap(), 1);

        let loaded = manager.load_synonym_graph().await.unwrap();
        assert_eq!(loaded.fingerprint(), graph.fingerprint());
        assert_eq!(
            loaded
                .canonicalize("老婆", SynonymKind::Entity, &SynonymContext::new())
                .canonical(),
            Some("妻子")
        );
        assert_eq!(manager.revision(), revision, "账本写入不推进知识版本");

        assert_eq!(manager.sync_synonym_graph(&graph).await.unwrap(), 1);
        assert_eq!(manager.load_synonym_graph().await.unwrap().len(), 1);

        assert_eq!(
            manager
                .sync_synonym_graph(&SynonymGraph::new(8))
                .await
                .unwrap(),
            0
        );
        assert!(manager.load_synonym_graph().await.unwrap().is_empty());
    }

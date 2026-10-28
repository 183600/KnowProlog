    /// 上下文级 `@language` 应作用于未显式声明语言的字符串字面量；
    /// 节点级上下文可覆盖或（用 null）清除继承的默认语言。
    #[tokio::test]
    async fn import_jsonld_applies_and_scopes_default_language() {
        let (manager, _dir) = temp_manager().await;
        let jsonld = r#"{
            "@context": {
                "@language": "zh",
                "ex": "http://example.org/kg/",
                "name": "ex:name"
            },
            "@graph": [
                { "@id": "ex:entity/甲", "name": "甲" },
                {
                    "@context": { "@language": "en" },
                    "@id": "ex:entity/乙",
                    "name": "乙"
                },
                {
                    "@context": { "@language": null },
                    "@id": "ex:entity/丙",
                    "name": "丙"
                }
            ]
        }"#;
        manager.import_jsonld(jsonld).await.unwrap();
        let result = manager
            .query_sparql(
                r#"PREFIX kg: <http://example.org/kg/>
                SELECT ?s ?lang WHERE {
                    ?s kg:name ?name .
                    BIND(lang(?name) AS ?lang)
                } ORDER BY ?s"#,
            )
            .await
            .unwrap();
        let lang_of = |name: &str| {
            result
                .data
                .iter()
                .find(|row| {
                    row.get("s")
                        == Some(&serde_json::json!(format!(
                            "http://example.org/kg/entity/{}",
                            name
                        )))
                })
                .and_then(|row| row.get("lang"))
                .cloned()
        };
        assert_eq!(lang_of("甲"), Some(serde_json::json!("zh")));
        assert_eq!(lang_of("乙"), Some(serde_json::json!("en")));
        assert_eq!(lang_of("丙"), Some(serde_json::json!("")));
    }

    /// 作用域上下文重新定义词项时，必须撤销父级继承的节点强制；
    /// 否则子节点里的普通字符串会被误判成实体引用。
    #[tokio::test]
    async fn import_jsonld_scoped_context_resets_inherited_coercion() {
        let (manager, _dir) = temp_manager().await;
        let jsonld = r#"{
            "@context": {
                "ex": "http://example.org/kg/",
                "friend": { "@id": "ex:friend", "@type": "@id" }
            },
            "@id": "http://example.org/kg/entity/甲",
            "friend": {
                "@context": { "friend": { "@id": "ex:friend" } },
                "@id": "http://example.org/kg/entity/乙",
                "friend": "李四"
            }
        }"#;

        manager.import_jsonld(jsonld).await.unwrap();
        let result = manager
            .query_sparql(
                r#"PREFIX kg: <http://example.org/kg/>
                SELECT ?friend WHERE {
                    <http://example.org/kg/entity/乙> kg:friend ?friend
                }"#,
            )
            .await
            .unwrap();
        assert_eq!(
            result.data[0].get("friend"),
            Some(&Value::String("李四".to_string())),
            "作用域上下文应清除父级 @type:@id 强制"
        );
    }

    /// 上下文里的 `@type: "@id"` / `"@vocab"` 是节点引用声明：字符串必须
    /// 落成实体边而不是字面量；数组中的 `null` 直接跳过。
    #[tokio::test]
    async fn import_jsonld_coerces_string_values_to_node_references() {
        let (manager, _dir) = temp_manager().await;
        let jsonld = r#"{
            "@context": {
                "ex": "http://example.org/kg/",
                "@base": "http://example.org/kg/entity/",
                "@vocab": "http://example.org/kg/entity/",
                "name": "ex:name",
                "friend": { "@id": "ex:friend", "@type": "@id" },
                "mentor": { "@id": "ex:mentor", "@type": "@vocab" },
                "parent": { "@id": "ex:parent", "@type": "@id" }
            },
            "@id": "http://example.org/kg/entity/张三",
            "name": "张三",
            "friend": "李四",
            "mentor": "王五",
            "parent": ["赵六", null]
        }"#;

        let stats = manager.import_jsonld(jsonld).await.unwrap();
        assert_eq!(stats.entities, 1);
        assert_eq!(stats.relations, 3, "friend + mentor + parent 都是实体边");
        assert_eq!(stats.triples, 4, "名字 1 + 关系 3");

        let result = manager
            .query_sparql(
                r#"PREFIX kg: <http://example.org/kg/>
                SELECT ?friend ?mentor ?parent WHERE {
                    <http://example.org/kg/entity/张三>
                        kg:friend ?friend ;
                        kg:mentor ?mentor ;
                        kg:parent ?parent
                }"#,
            )
            .await
            .unwrap();
        assert_eq!(result.data.len(), 1, "{:?}", result.data);
        assert_eq!(
            result.data[0].get("friend"),
            Some(&serde_json::json!("http://example.org/kg/entity/李四"))
        );
        assert_eq!(
            result.data[0].get("mentor"),
            Some(&serde_json::json!("http://example.org/kg/entity/王五"))
        );
        assert_eq!(
            result.data[0].get("parent"),
            Some(&serde_json::json!("http://example.org/kg/entity/赵六"))
        );

        assert_eq!(
            manager.import_jsonld(jsonld).await.unwrap().triples,
            0,
            "重复导入保持幂等"
        );
    }

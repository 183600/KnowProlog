#[cfg(test)]
mod session_tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn snapshot_accepts_subject_without_predicate() {
        let cli = Cli::try_parse_from(["knowprolog", "snapshot", "--subject", "张三"]).unwrap();
        match cli.command {
            Some(Commands::Snapshot {
                subject: Some(subject),
                predicate: None,
                ..
            }) => assert_eq!(subject, "张三"),
            _ => panic!("应解析为按实体查看快照"),
        }
        assert!(
            Cli::try_parse_from(["knowprolog", "snapshot", "--predicate", "age"]).is_err(),
            "--predicate 离开 --subject 应报错"
        );
    }

    #[test]
    fn privacy_cli_accepts_store_audit_and_speech_acts() {
        let cli = Cli::try_parse_from([
            "knowprolog",
            "privacy",
            "--from-store",
            "--max-facts",
            "12",
            "--subject",
            "张三",
            "--utterance",
            "别把我的电话告诉别人",
        ])
        .unwrap();
        match cli.command {
            Some(Commands::Privacy {
                from_store,
                max_facts,
                subject,
                utterance,
                ..
            }) => {
                assert!(from_store);
                assert_eq!(max_facts, Some(12));
                assert_eq!(subject, "张三");
                assert_eq!(utterance, vec!["别把我的电话告诉别人".to_string()]);
            }
            _ => panic!("应解析为隐私子命令"),
        }
    }

    #[test]
    fn privacy_fact_file_skips_comments_and_reports_bad_lines() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("facts.txt");
        std::fs::write(
            &path,
            "# comment\n\n张三|电话|13800138000\n  李四|地址|北京市  \n",
        )
        .unwrap();
        let facts = read_privacy_facts(path.to_str().unwrap()).unwrap();
        assert_eq!(
            facts,
            vec![
                (
                    "张三".to_string(),
                    "电话".to_string(),
                    "13800138000".to_string()
                ),
                ("李四".to_string(), "地址".to_string(), "北京市".to_string()),
            ]
        );

        std::fs::write(&path, "只有两个|字段\n").unwrap();
        let error = read_privacy_facts(path.to_str().unwrap()).unwrap_err();
        assert!(error.contains(":1"), "错误应带行号：{}", error);
    }

    #[test]
    fn semantic_fact_argument_accepts_common_separators() {
        for raw in [
            "张三|height|226",
            "张三 height 226",
            "张三,height,226",
            "张三，height，226",
        ] {
            let fact = parse_semantic_fact(raw).unwrap();
            assert_eq!(fact.subject, "张三");
            assert_eq!(fact.predicate, "height");
            assert_eq!(fact.value, "226");
        }
        assert!(parse_semantic_fact("只有两个|字段").is_none());
    }

    #[test]
    fn store_override_isolates_relative_dialogue_state() {
        let mut path = Some("oxigraph_data/dialogue_state.json".to_string());
        isolate_dialogue_state_path(&mut path, "/tmp/kp-alt");
        assert_eq!(
            path.as_deref(),
            Some("/tmp/kp-alt/dialogue_state.json"),
            "相对状态路径必须跟随 --store"
        );

        let mut absolute = Some("/var/lib/kp/state.json".to_string());
        isolate_dialogue_state_path(&mut absolute, "/tmp/kp-alt");
        assert_eq!(
            absolute.as_deref(),
            Some("/var/lib/kp/state.json"),
            "绝对路径是用户显式指定，不得改写"
        );

        let mut missing = None;
        isolate_dialogue_state_path(&mut missing, "/tmp/kp-alt");
        assert_eq!(missing, None);
    }

    #[test]
    fn ask_intro_request_targets_known_entity() {
        let mut reasoner = SymbolicReasoner::new(
            know_prolog::ReasoningConfig::default(),
            std::collections::HashMap::new(),
        );
        reasoner.observe_facts(&[PrologFact::Entity {
            name: "北京".to_string(),
            entity_type: "City".to_string(),
            properties: std::collections::HashMap::from([
                ("name".to_string(), "北京".to_string()),
                ("population".to_string(), "2189万".to_string()),
            ]),
        }]);
        for input in ["介绍一下北京", "北京介绍", "说说北京", "介绍：北京"] {
            assert_eq!(
                intro_request_entity(input, &reasoner).as_deref(),
                Some("北京"),
                "{}",
                input
            );
        }
        assert_eq!(intro_request_entity("介绍一下你自己", &reasoner), None);
        assert_eq!(intro_request_entity("介绍不存在的地方", &reasoner), None);

        // 介绍轮把实体推入焦点栈，后续代词问句才能继承。
        let mut dialogue = DialogueManager::new();
        dialogue.note_plugin_focus("北京");
        let frame = dialogue.analyze("它的人口呢？", &reasoner);
        assert_eq!(frame.kind, MoveKind::AskAttribute, "{:?}", frame);
        assert_eq!(frame.entity.as_deref(), Some("北京"));
        assert_eq!(frame.attribute.as_deref(), Some("人口"));
    }

    #[test]
    fn probe_reference_answer_separates_records_from_absence() {
        let mut reasoner = SymbolicReasoner::new(
            know_prolog::ReasoningConfig::default(),
            std::collections::HashMap::new(),
        );
        assert_eq!(
            probe_reference_answer(&reasoner, "苹果", "苹果的颜色是什么"),
            ProbeStatus::NoAnswer
        );
        reasoner.observe_facts(&[PrologFact::Entity {
            name: "苹果".to_string(),
            entity_type: "水果".to_string(),
            properties: std::collections::HashMap::from([
                ("name".to_string(), "苹果".to_string()),
                ("颜色".to_string(), "红色".to_string()),
            ]),
        }]);
        let status = probe_reference_answer(&reasoner, "苹果", "苹果的颜色是什么");
        assert!(
            matches!(status, ProbeStatus::Answered(ref values) if !values.is_empty()),
            "{:?}",
            status
        );
    }

    #[test]
    fn place_degree_question_compiles_to_area() {
        let mut reasoner = SymbolicReasoner::new(
            know_prolog::ReasoningConfig::default(),
            std::collections::HashMap::new(),
        );
        reasoner.observe_facts(&[PrologFact::Entity {
            name: "北京".to_string(),
            entity_type: "City".to_string(),
            properties: std::collections::HashMap::from([
                ("name".to_string(), "北京".to_string()),
                ("area".to_string(), "16410平方公里".to_string()),
            ]),
        }]);
        let dialogue = DialogueManager::new();
        let frame = dialogue.analyze("北京有多大？", &reasoner);
        assert_eq!(frame.kind, MoveKind::AskAttribute, "{:?}", frame);
        assert_eq!(frame.entity.as_deref(), Some("北京"));
        assert_eq!(frame.attribute.as_deref(), Some("面积"));

        reasoner.observe_facts(&[PrologFact::Entity {
            name: "北京".to_string(),
            entity_type: "City".to_string(),
            properties: std::collections::HashMap::from([
                ("name".to_string(), "北京".to_string()),
                ("population".to_string(), "2189万".to_string()),
            ]),
        }]);
        for input in ["北京有多少人？", "北京的人口有多少人？", "北京有多少人口？"]
        {
            let frame = dialogue.analyze(input, &reasoner);
            assert_eq!(frame.kind, MoveKind::AskAttribute, "{} {:?}", input, frame);
            assert_eq!(frame.entity.as_deref(), Some("北京"), "{}", input);
            assert_eq!(frame.attribute.as_deref(), Some("人口"), "{}", input);
        }
    }

    #[test]
    fn ask_generation_followup_recognizes_continue_and_switch() {
        for input in ["继续介绍", "接着说。", "展开说说", "详细一点？"] {
            assert_eq!(
                generation_followup(input),
                Some(GenerationFollowup::Continue),
                "{}",
                input
            );
        }
        for input in ["换个角度说", "重新介绍？"] {
            assert_eq!(
                generation_followup(input),
                Some(GenerationFollowup::SwitchAngle),
                "{}",
                input
            );
        }
        assert_eq!(generation_followup("介绍一下北京"), None);
        assert_eq!(generation_followup("北京的人口是多少"), None);
    }

    #[test]
    fn generation_style_commands_do_not_steal_existing_followups() {
        for input in [
            "简短点",
            "完整讲讲",
            "换种表达",
            "第二点是什么",
            "风格：第2点",
        ] {
            assert!(
                StyleDirective::parse(input).is_some(),
                "应识别风格指令：{}",
                input
            );
            assert!(is_command_like(input), "风格指令不应被当成短答：{}", input);
        }
        for input in [
            "回答计划",
            "计划台账",
            "生成计划",
            "回答方案",
            "复核计划",
            "计划复核",
            "核对计划",
            "检查计划",
        ] {
            assert!(is_command_like(input), "计划命令不应被当成短答：{}", input);
        }
        for input in ["详细一点", "展开说说", "展开讲", "换个说法", "接着说"] {
            assert!(
                StyleDirective::parse(input).is_none(),
                "不得改写既有续说/换角度语义：{}",
                input
            );
        }
    }

    #[test]
    fn import_cli_accepts_stdin_and_explicit_rdf_format() {
        let jsonld = Cli::try_parse_from(["knowprolog", "import-jsonld", "-"]).unwrap();
        assert!(matches!(
            jsonld.command,
            Some(Commands::ImportJsonld { file_path }) if file_path == "-"
        ));

        let rdf =
            Cli::try_parse_from(["knowprolog", "import-rdf", "-", "--format", "turtle"]).unwrap();
        match rdf.command {
            Some(Commands::ImportRdf { file_path, format }) => {
                assert_eq!(file_path, "-");
                assert_eq!(format.as_deref(), Some("turtle"));
            }
            other => panic!("应解析为 import-rdf: {:?}", other),
        }

        let file =
            Cli::try_parse_from(["knowprolog", "import-rdf", "dump.data", "--format", "ttl"])
                .unwrap();
        assert!(matches!(
            file.command,
            Some(Commands::ImportRdf { format: Some(name), .. }) if name == "ttl"
        ));
    }

    #[test]
    fn ask_cli_accepts_question_and_stdin_modes() {
        let cli = Cli::try_parse_from([
            "knowprolog",
            "ask",
            "张三的师傅是谁",
            "--json",
            "--strict",
            "--explain",
            "--clarify",
        ])
        .expect("ask 参数应可解析");
        match cli.command {
            Some(Commands::Ask {
                question,
                json,
                strict,
                explain,
                clarify,
            }) => {
                assert_eq!(question.as_deref(), Some("张三的师傅是谁"));
                assert!(json);
                assert!(strict);
                assert!(explain);
                assert!(clarify);
            }
            _ => panic!("应解析为 Ask 子命令"),
        }
        let cli = Cli::try_parse_from(["knowprolog", "ask"]).expect("无问句时应读 stdin");
        assert!(matches!(
            cli.command,
            Some(Commands::Ask {
                question: None,
                json: false,
                strict: false,
                explain: false,
                clarify: false
            })
        ));
        for terminator in ["quit", "exit", "退出", "q", " 退出 "] {
            assert!(is_ask_terminator(terminator), "{}", terminator);
        }
        assert!(!is_ask_terminator("为什么退出假设"));
    }

    #[test]
    fn rewrite_retrieval_cli_parses_question_and_json() {
        let cli = Cli::try_parse_from([
            "knowprolog",
            "rewrite-retrieval",
            "谁的师傅是李四",
            "--book",
            "rules.json",
            "--json",
        ])
        .expect("rewrite-retrieval 参数应可解析");
        assert!(matches!(
            cli.command,
            Some(Commands::RewriteRetrieval { question, book, json })
                if question == "谁的师傅是李四"
                    && book.as_deref() == Some("rules.json")
                    && json
        ));
    }

    #[test]
    fn rewrite_book_loads_from_store_path() {
        let dir = tempfile::TempDir::new().expect("临时目录");
        let path = dir.path().join("rewrite_book.json");
        RewriteBook::new().save_to_path(&path).expect("写入手册");
        let loaded =
            load_rewrite_book(dir.path().to_str().expect("utf-8 路径")).expect("应读回手册");
        assert_eq!(loaded.dialect, RewriteBook::new().dialect);
    }

    #[test]
    fn rewrite_retrieval_eval_cli_parses_corpus_and_flags() {
        let cli = Cli::try_parse_from([
            "knowprolog",
            "rewrite-retrieval-eval",
            "corpus.tsv",
            "--book",
            "rules.json",
            "--strict",
            "--json",
        ])
        .expect("rewrite-retrieval-eval 参数应可解析");
        assert!(matches!(
            cli.command,
            Some(Commands::RewriteRetrievalEval { corpus, book, strict, json })
                if corpus == "corpus.tsv"
                    && book.as_deref() == Some("rules.json")
                    && strict
                    && json
        ));
    }

    #[tokio::test]
    async fn ask_lens_keeps_source_scope_and_discloses_other_contexts() {
        let dir = tempfile::TempDir::new().unwrap();
        let manager =
            OxigraphManager::with_perf_ledger(dir.path().to_str().unwrap(), PerfLedger::shared())
                .unwrap();
        let namespace = NamespaceRecord::new(
            "alice",
            "alice",
            NamespaceKind::Import,
            TrustLevel::Imported,
        );
        let assertions = vec![know_prolog::knowledge_fusion::ScopedAssertion::new(
            "alice",
            "张三",
            "师傅",
            "李四",
            know_prolog::knowledge_fusion::ObjectKind::Literal,
            "alice.ttl",
        )];
        manager
            .record_namespace_assertions(&namespace, &assertions)
            .await
            .unwrap();
        let config = load_config_from("config.toml").await.unwrap();
        let mut cache = None;

        let reply = ask_with_lens("据alice说，张三的师傅是谁", &manager, &config, &mut cache)
            .await
            .expect("来源镜头应接管");
        assert!(reply.contains("李四"), "{}", reply);
        assert!(reply.contains("alice"), "{}", reply);
        assert!(!reply.contains("是 王五"), "{}", reply);

        let reply = ask_with_lens("知识库里张三的师傅是谁", &manager, &config, &mut cache)
            .await
            .expect("知识库镜头应接管");
        assert!(reply.contains("没有关于"), "{}", reply);
        assert!(reply.contains("alice"), "{}", reply);

        let reply = ask_with_lens("张三的师傅是谁", &manager, &config, &mut cache).await;
        assert!(reply.is_none(), "无语境线索不应接管旧回答路径");
    }

    /// `ask` 的只读推理器必须和交互/服务一样接入配置律典：
    /// 只同步谓词词面而不配置显式传递律，会让「所有」问句错误地拒绝闭包。
    #[tokio::test]
    async fn ask_reasoner_honors_configured_transitive_laws() {
        let dir = tempfile::TempDir::new().unwrap();
        let manager =
            OxigraphManager::with_perf_ledger(dir.path().to_str().unwrap(), PerfLedger::shared())
                .unwrap();
        let jsonld = std::fs::read_to_string("examples/master_disciple.jsonld").unwrap();
        manager.import_jsonld(&jsonld).await.unwrap();
        let facts = manager.load_graph_snapshot().await.unwrap();
        let config = load_config_from("config.toml").await.unwrap();

        let reasoner = build_store_reasoner(
            &config,
            &manager,
            dir.path().to_str().unwrap(),
            &facts,
        )
        .await;
        let answer = reasoner
            .answer("张三的所有师傅是谁")
            .expect("传递律应让闭包回答");
        assert!(answer.contains("law:transitive:师傅"), "{}", answer);
        assert!(answer.contains("钱七"), "{}", answer);
        assert!(answer.contains("4 个"), "{}", answer);
    }

    #[test]
    fn namespace_cli_parses_import_list_and_status() {
        let cli = Cli::try_parse_from(["knowprolog", "namespace", "--source", "alice", "a.ttl"])
            .expect("命名空间导入应可解析");
        match cli.command {
            Some(Commands::Namespace {
                source,
                file_path,
                list,
                status,
                ..
            }) => {
                assert_eq!(source.as_deref(), Some("alice"));
                assert_eq!(file_path.as_deref(), Some("a.ttl"));
                assert!(!list);
                assert!(status.is_none());
            }
            other => panic!("应解析为 Namespace 子命令: {:?}", other),
        }
        let cli = Cli::try_parse_from(["knowprolog", "namespace", "--list"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Commands::Namespace { list: true, .. })
        ));
        let cli = Cli::try_parse_from([
            "knowprolog",
            "namespace",
            "--source",
            "alice",
            "--status",
            "retired",
        ])
        .expect("命名空间状态应可解析");
        match cli.command {
            Some(Commands::Namespace {
                source,
                status,
                file_path,
                ..
            }) => {
                assert_eq!(source.as_deref(), Some("alice"));
                assert_eq!(status.as_deref(), Some("retired"));
                assert!(file_path.is_none());
            }
            other => panic!("应解析为 Namespace 子命令: {:?}", other),
        }
        assert!(Cli::try_parse_from(["knowprolog", "namespace", "--list", "a.ttl"]).is_err());
        assert!(Cli::try_parse_from(["knowprolog", "namespace", "--source", "alice"]).is_err());
    }

    #[test]
    fn lens_cli_parses_question_and_explicit_slot() {
        let cli =
            Cli::try_parse_from(["knowprolog", "lens", "据alice说，张三的师傅是谁", "--json"])
                .expect("lens 问句应可解析");
        match cli.command {
            Some(Commands::Lens {
                question,
                subject,
                predicate,
                json,
                ..
            }) => {
                assert_eq!(question.as_deref(), Some("据alice说，张三的师傅是谁"));
                assert!(subject.is_none());
                assert!(predicate.is_none());
                assert!(json);
            }
            other => panic!("应解析为 Lens 子命令: {:?}", other),
        }
        let cli = Cli::try_parse_from([
            "knowprolog",
            "lens",
            "--subject",
            "张三",
            "--predicate",
            "师傅",
            "--view",
            "alice",
            "--dialogue-state",
            "state.json",
        ])
        .expect("显式槽位应可解析");
        match cli.command {
            Some(Commands::Lens {
                question,
                subject,
                predicate,
                view,
                dialogue_state,
                facts,
                ..
            }) => {
                assert!(question.is_none());
                assert_eq!(subject.as_deref(), Some("张三"));
                assert_eq!(predicate.as_deref(), Some("师傅"));
                assert_eq!(view.as_deref(), Some("alice"));
                assert_eq!(dialogue_state.as_deref(), Some("state.json"));
                assert!(!facts);
            }
            other => panic!("应解析为 Lens 子命令: {:?}", other),
        }
        assert!(Cli::try_parse_from(["knowprolog", "lens"]).is_err());
        // 只给主语时 Clap 仍可解析，run_lens 会要求补 --predicate 或 --facts。
        assert!(Cli::try_parse_from(["knowprolog", "lens", "--subject", "张三"]).is_ok());
        let cli = Cli::try_parse_from([
            "knowprolog",
            "lens",
            "--facts",
            "--view",
            "all",
            "--subject",
            "张三",
        ])
        .expect("事实列表应允许只给主语过滤");
        match cli.command {
            Some(Commands::Lens {
                facts,
                subject,
                predicate,
                ..
            }) => {
                assert!(facts);
                assert_eq!(subject.as_deref(), Some("张三"));
                assert!(predicate.is_none());
            }
            other => panic!("应解析为 Lens 子命令: {:?}", other),
        }
        assert!(Cli::try_parse_from(["knowprolog", "lens", "--facts", "--contexts"]).is_err());
    }

    #[test]
    fn types_cli_parses_profile_members_and_compare() {
        let cli = Cli::try_parse_from([
            "knowprolog",
            "types",
            "--type",
            "人物",
            "--members",
            "--offset",
            "10",
            "--limit",
            "5",
            "--json",
        ])
        .expect("types 成员分页参数应可解析");
        match cli.command {
            Some(Commands::Types {
                type_name,
                compare,
                members,
                offset,
                limit,
                json,
            }) => {
                assert_eq!(type_name.as_deref(), Some("人物"));
                assert!(compare.is_none());
                assert!(members);
                assert_eq!(offset, 10);
                assert_eq!(limit, 5);
                assert!(json);
            }
            other => panic!("应解析为 Types：{:?}", other),
        }

        let cli = Cli::try_parse_from(["knowprolog", "types", "--compare", "人物", "组织"])
            .expect("types 对比参数应可解析");
        assert!(matches!(
            cli.command,
            Some(Commands::Types {
                compare: Some(pair),
                members: false,
                ..
            }) if pair == vec!["人物".to_string(), "组织".to_string()]
        ));

        assert!(Cli::try_parse_from(["knowprolog", "types", "--members"]).is_err());
        assert!(Cli::try_parse_from([
            "knowprolog",
            "types",
            "--type",
            "人物",
            "--compare",
            "人物",
            "组织"
        ])
        .is_err());
    }

    #[test]
    fn predicates_cli_parses_repeatable_edits() {
        let cli = Cli::try_parse_from([
            "knowprolog",
            "predicates",
            "--confirm",
            "认识",
            "--reject",
            "噪音",
            "--alias",
            "老师=师傅",
            "--inverse",
            "师从=徒弟",
            "--functional",
            "年龄",
            "--non-functional",
            "别名",
            "--json",
        ])
        .expect("predicates 参数应可解析");
        match cli.command {
            Some(Commands::Predicates {
                json,
                confirm,
                reject,
                alias,
                inverse,
                functional,
                non_functional,
            }) => {
                assert!(json);
                assert_eq!(confirm, vec!["认识"]);
                assert_eq!(reject, vec!["噪音"]);
                assert_eq!(alias, vec!["老师=师傅"]);
                assert_eq!(inverse, vec!["师从=徒弟"]);
                assert_eq!(functional, vec!["年龄"]);
                assert_eq!(non_functional, vec!["别名"]);
            }
            _ => panic!("应解析为 Predicates 子命令"),
        }
        assert_eq!(parse_predicate_spec("老师=师傅"), Some(("老师", "师傅")));
        assert_eq!(parse_predicate_spec("老师＝师傅"), Some(("老师", "师傅")));
        assert_eq!(parse_predicate_spec("老师=>师傅"), Some(("老师", "师傅")));
        assert_eq!(parse_predicate_spec("老师"), None);
        assert_eq!(parse_predicate_spec("=师傅"), None);
        assert_eq!(parse_predicate_spec("老师="), None);
        assert_eq!(parse_predicate_spec("老师=老师"), None);
    }

    #[test]
    fn ask_pipeline_does_not_turn_copula_statement_into_attribute_query() {
        let mut reasoner = SymbolicReasoner::new(
            know_prolog::ReasoningConfig::default(),
            std::collections::HashMap::new(),
        );
        reasoner.observe_facts(&[
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "师傅".to_string(),
                to: "李四".to_string(),
            },
            PrologFact::Relation {
                from: "李四".to_string(),
                rel_type: "师傅".to_string(),
                to: "王五".to_string(),
            },
        ]);
        let mut manager = DialogueManager::new();
        let frame = manager.analyze("张三是李四的师傅", &reasoner);
        assert_eq!(
            frame.kind,
            MoveKind::Assert,
            "frame={:?} question={:?}",
            frame.kind,
            frame.question
        );
        for input in ["张三是李四的师傅", "张三是李四的师傅吗"] {
            if let Some(reply) = manager.plan_pure_dialogue(input, &reasoner) {
                panic!(
                    "{} 被纯对话层接管：kind={:?} plan={:?} text={}",
                    input, reply.kind, reply.plan, reply.text
                );
            }
            if let Some(reply) = manager.plan_reply(input, &reasoner) {
                panic!(
                    "{} 被对话层接管：kind={:?} plan={:?} text={}",
                    input, reply.kind, reply.plan, reply.text
                );
            }
        }
        let answer = reasoner.answer("张三是李四的师傅").unwrap();
        assert!(answer.contains("不是"), "{}", answer);
    }

    #[test]
    fn config_override_is_global_and_defaults_to_config_toml() {
        let cli = Cli::try_parse_from([
            "knowprolog",
            "--config",
            "conf/alt.toml",
            "ask",
            "张三的师傅是谁",
        ])
        .expect("全局 --config 应可解析");
        assert_eq!(cli.config, "conf/alt.toml");
        let default = Cli::try_parse_from(["knowprolog", "ask", "问题"]).unwrap();
        assert_eq!(default.config, "config.toml");
    }

    /// 各命令必须被 is_command_like 识别，避免澄清追问时被吞掉
    #[test]
    fn command_like_covers_journal_and_perf_commands() {
        for command in [
            "语义分析：张三的年龄是多少",
            "语义森林：张三的年龄是多少",
            "撤销导入",
            "导入回执",
            "导入历史",
            "导入日志",
            "性能报告",
            "性能台账",
            "配置契约",
            "配置体检",
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
            "情景",
            "对话情景",
            "情景理论",
            "情景：12",
            "情景历史：姚明的身高是多少",
            "情景归因：姚明的身高是多少",
            "情景反事实：姚明的身高是多少",
            "情景查询：answered_at(S,Q,A,P)",
            "外链查看：法国",
            "外链刷新：法国",
            "外链删除：法国",
            "同义链接：姚明 = ownthink:姚明",
            "帧记忆",
            "话语帧",
            "挂账",
            "追踪报告",
            "追踪状态",
            "追踪图",
            "指代轨迹",
            "问据栈",
            "追踪为什么 4",
            "查询剧本：examples/query_board_scenario.toml",
            "运行剧本：examples/query_board_scenario.toml",
            "默认推理：麻雀会飞",
            "默认逻辑：",
            "对齐候选：老婆",
            "补全回合：张三",
            "确认补全：1",
            "否定补全：1",
            "补全提问",
            "补全测试",
            "补全重放",
            "别问了",
            "允许提问",
            "晋升补全规则：rule-abc|全局",
            "退役补全规则：rule-abc",
            "恢复补全规则：rule-abc",
        ] {
            assert!(is_command_like(command), "{} 应被识别为系统命令", command);
        }
    }

    /// 各命令（谓词裁决 / 澄清子命令 / 帮助退出等）都要被 is_command_like 识别
    #[test]
    fn command_like_covers_pending_question_commands() {
        for command in [
            "悬问",
            "悬问台账",
            "未答之问",
            "放弃悬问",
            "清空悬问",
            "重开悬问",
            "恢复悬问",
            "重开未答之问",
        ] {
            assert!(is_command_like(command), "{} 应被识别为系统命令", command);
        }
        assert!(is_no_record_text(
            "根据现有知识库，没有记录「张三 的 师傅」。"
        ));
        assert!(is_no_record_text("未能回答：当前知识库中没有可用依据。"));
        assert!(!is_no_record_text("张三 的 师傅 是 李四。"));
    }

    #[test]
    fn command_like_covers_mention_lexicon_commands() {
        for command in [
            "词表",
            "提及词表",
            "提及词表：未名科技公司发布了新芯片",
            "词表感知：未名科技公司",
            "词表提名：未名科技公司",
            "词表拒绝：未名科技公司",
            "词表报告",
        ] {
            assert!(is_command_like(command), "{} 应被识别为系统命令", command);
        }
    }

    #[test]
    fn command_like_covers_help_predicate_and_clarification_commands() {
        for command in [
            "帮助",
            "help",
            "?",
            "退出",
            "exit",
            "谓词报告",
            "谓词画像",
            "谓词注册表",
            "谓词列表",
            "登记别名：师父 = 师傅",
            "登记逆谓词：师傅 = 徒弟",
            "功能谓词：出生地",
            "取消功能谓词：出生地",
            "确认谓词：师傅",
            "拒绝谓词：师傅",
            "待定推测",
            "后向链规则",
            "开始澄清",
            "停止澄清",
            "结束澄清",
            "退出澄清",
            "提问确认",
            "接着说",
            "继续说",
            "继续介绍",
            "继续讲",
            "接着讲",
            "换个角度说",
            "换个说法",
            "换个角度",
            "换一个角度",
            "重新介绍",
            "撤销上次导入",
            "撤销批次：第2批",
        ] {
            assert!(is_command_like(command), "{} 应被识别为系统命令", command);
        }
    }

    /// 导入指令解析：全角/半角冒号、README 简写、参数正文零替换
    #[test]
    fn import_directive_parses_colons_aliases_and_keeps_body() {
        assert_eq!(
            parse_import_directive("导入jsonld文件：test_data.jsonld"),
            Some(ImportDirective::JsonldFile("test_data.jsonld".to_string()))
        );
        assert_eq!(
            parse_import_directive("导入jsonld文件: test_data.jsonld"),
            Some(ImportDirective::JsonldFile("test_data.jsonld".to_string()))
        );
        // README 中的简写
        assert_eq!(
            parse_import_directive("导入文件：examples/master_disciple.jsonld"),
            Some(ImportDirective::JsonldFile(
                "examples/master_disciple.jsonld".to_string()
            ))
        );
        assert_eq!(
            parse_import_directive("导入内容：{\"@id\": \"http://example.org/kg/entity/王五\"}"),
            Some(ImportDirective::JsonldContent(
                "{\"@id\": \"http://example.org/kg/entity/王五\"}".to_string()
            ))
        );
        assert_eq!(
            parse_import_directive("import jsonld content: {\"@id\": \"x\"}"),
            Some(ImportDirective::JsonldContent(
                "{\"@id\": \"x\"}".to_string()
            ))
        );
        assert_eq!(
            parse_import_directive("导入rdf文件：data/sample.ttl"),
            Some(ImportDirective::RdfFile("data/sample.ttl".to_string()))
        );
        assert_eq!(
            parse_import_directive("import rdf: data/sample.nt"),
            Some(ImportDirective::RdfFile("data/sample.nt".to_string()))
        );

        // 参数正文里再出现关键词不得被删改
        assert_eq!(
            parse_import_directive("导入文件：导入文件：备份.jsonld"),
            Some(ImportDirective::JsonldFile(
                "导入文件：备份.jsonld".to_string()
            ))
        );

        // 非导入指令（自然语言条目）不误判
        assert_eq!(parse_import_directive("录入：小明18岁"), None);
        assert_eq!(parse_import_directive("导入jsonld文件"), None);
    }

    /// 撤销批次指令：支持全角/半角冒号与英文别名，空引用不误判。
    #[test]
    fn retract_batch_directive_parses_reference() {
        assert_eq!(
            parse_retract_batch_directive("撤销批次：第2批"),
            Some("第2批".to_string())
        );
        assert_eq!(
            parse_retract_batch_directive("撤销批次: data/sample.ttl"),
            Some("data/sample.ttl".to_string())
        );
        assert_eq!(
            parse_retract_batch_directive("retract batch: B1"),
            Some("B1".to_string())
        );
        assert_eq!(parse_retract_batch_directive("撤销批次："), None);
        assert_eq!(parse_retract_batch_directive("撤销上次导入"), None);
    }

    /// 帮助页覆盖关键入口（录入 / 导入 / 查询 / 退出），避免手册与实现漂移
    #[test]
    fn interactive_help_mentions_key_commands() {
        let help = interactive_help();
        for needle in [
            "录入：",
            "导入jsonld文件：",
            "导入rdf文件：",
            "导出rdf文件：",
            "SPARQL：",
            "为什么",
            "证明：",
            "假设世界",
            "退出假设",
            "追踪报告",
            "改口：",
            "槽账本",
            "接地证书",
            "接地重放",
            "简短点",
            "回答计划",
            "复核计划",
            "帮助",
            "quit",
        ] {
            assert!(help.contains(needle), "帮助页应包含 {}", needle);
        }
    }

    /// SPARQL 指令解析：全角/半角冒号、大小写、空格形式与正文保真
    #[test]
    fn sparql_command_parses_colons_case_and_keeps_body() {
        let select = "SELECT ?s WHERE { ?s ?p ?o }";
        assert_eq!(
            parse_sparql_command(&format!("SPARQL：{}", select)),
            Some(select)
        );
        assert_eq!(
            parse_sparql_command(&format!("sparql:{}", select)),
            Some(select)
        );
        assert_eq!(
            parse_sparql_command(&format!("Sparql {}", select)),
            Some(select)
        );
        assert_eq!(
            parse_sparql_command(&format!("sparql {}", select)),
            Some(select)
        );
        assert_eq!(
            parse_sparql_command(&format!("查询SPARQL：{}", select)),
            Some(select)
        );
        assert_eq!(
            parse_sparql_command(&format!("查询SPARQL:{}", select)),
            Some(select)
        );
        // 半角冒号的短前缀不得按全角长度切分而吃掉正文（含多字节字符）
        assert_eq!(
            parse_sparql_command("查询SPARQL:SELECT ?s WHERE { ?s <http://a/中> }"),
            Some("SELECT ?s WHERE { ?s <http://a/中> }")
        );
        assert_eq!(parse_sparql_command("sparql："), Some(""));
        assert_eq!(parse_sparql_command("小明多大？"), None);
        // 正文里的冒号与关键词不被删改
        assert_eq!(
            parse_sparql_command("SPARQL：SELECT ?s WHERE { ?s <http://a/b> \"x:y\" }"),
            Some("SELECT ?s WHERE { ?s <http://a/b> \"x:y\" }")
        );
        assert!(is_command_like("SPARQL：SELECT ?s WHERE { ?s ?p ?o }"));
        assert!(is_command_like("sparql SELECT ?s WHERE { ?s ?p ?o }"));
    }

    /// 导出指令解析：全角/半角冒号、简写与路径保真；非指令不误判
    #[test]
    fn export_directive_parses_colons_aliases_and_keeps_path() {
        assert_eq!(
            parse_export_directive("导出rdf文件：out/data.ttl"),
            Some(ExportDirective::RdfFile("out/data.ttl".to_string()))
        );
        assert_eq!(
            parse_export_directive("导出文件: backup.jsonld"),
            Some(ExportDirective::RdfFile("backup.jsonld".to_string()))
        );
        assert_eq!(
            parse_export_directive("export rdf: dump.nt"),
            Some(ExportDirective::RdfFile("dump.nt".to_string()))
        );
        assert_eq!(
            parse_export_directive("导出rdf：导出文件：怪名.ttl"),
            Some(ExportDirective::RdfFile("导出文件：怪名.ttl".to_string()))
        );
        assert_eq!(parse_export_directive("导出rdf文件"), None);
        assert_eq!(parse_export_directive("录入：小明18岁"), None);
        assert!(is_command_like("导出文件：dump.ttl"));
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
            let manager =
                OxigraphManager::with_perf_ledger(&store_path, PerfLedger::shared()).unwrap();
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
        let manager = OxigraphManager::with_perf_ledger(&store_path, PerfLedger::shared()).unwrap();
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

    /// 服务会话的知识装载必须含关系事实：只有实体会让关系问法静默落空。
    #[tokio::test]
    async fn serve_facts_include_relations_alongside_entities() {
        let dir = tempfile::TempDir::new().unwrap();
        let store_path = dir.path().to_str().unwrap().to_string();
        let manager = OxigraphManager::with_perf_ledger(&store_path, PerfLedger::shared()).unwrap();
        let items = vec![
            KnowledgeItem::Entity {
                label: "人物".to_string(),
                properties: HashMap::from([
                    ("name".to_string(), "张三".to_string()),
                    ("age".to_string(), "25".to_string()),
                ]),
            },
            KnowledgeItem::Relationship {
                from_label: "人物".to_string(),
                from_id: "张三".to_string(),
                to_label: "人物".to_string(),
                to_id: "李四".to_string(),
                rel_type: "师傅".to_string(),
                properties: HashMap::new(),
            },
        ];
        manager.save_knowledge(items).await.unwrap();

        let facts = serve_facts(&manager).await.unwrap();
        assert!(
            facts
                .iter()
                .any(|fact| matches!(fact, PrologFact::Entity { .. })),
            "实体事实应保留：{:?}",
            facts
        );
        assert!(
            facts.iter().any(
                |fact| matches!(fact, PrologFact::Relation { rel_type, .. } if rel_type == "师傅")
            ),
            "关系事实必须随服务启动一并装载：{:?}",
            facts
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
        assert!(
            prompt.contains("叶莉") && prompt.contains("姚明"),
            "{}",
            prompt
        );

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

    /// 歧义义项选择解析：合法编号映射为 0 基下标；空/非数字/越界回退 None
    #[test]
    fn ambiguous_choice_parses_valid_index_and_rejects_invalid() {
        assert_eq!(parse_ambiguous_choice("1", 5), Some(0));
        assert_eq!(parse_ambiguous_choice(" 3 ", 5), Some(2));
        assert_eq!(parse_ambiguous_choice("5", 5), Some(4));
        assert_eq!(parse_ambiguous_choice("0", 5), None, "编号从 1 开始");
        assert_eq!(parse_ambiguous_choice("6", 5), None, "超出候选数量");
        assert_eq!(parse_ambiguous_choice("", 5), None, "空输入回退");
        assert_eq!(parse_ambiguous_choice("苹果", 5), None, "非数字回退");
        assert_eq!(parse_ambiguous_choice("2", 1), None, "单个候选时越界");
    }

    #[test]
    fn correction_spec_accepts_pipe_and_chinese_forms() {
        assert_eq!(
            parse_correction_spec("姚明|height|226|229"),
            Some((
                "姚明".to_string(),
                "height".to_string(),
                "226".to_string(),
                "229".to_string()
            ))
        );
        assert_eq!(
            parse_correction_spec("姚明的身高从226改为229"),
            Some((
                "姚明".to_string(),
                "身高".to_string(),
                "226".to_string(),
                "229".to_string()
            ))
        );
        assert_eq!(
            parse_correction_spec("把姚明的身高由 226 改成 229。"),
            Some((
                "姚明".to_string(),
                "身高".to_string(),
                "226".to_string(),
                "229".to_string()
            ))
        );
        assert_eq!(parse_correction_spec("姚明的身高是229"), None);
        assert_eq!(parse_correction_spec("姚明|height|226"), None);
    }

    #[test]
    fn paths_cli_supports_between_chain_and_common_modes() {
        let between = Cli::try_parse_from([
            "knowprolog",
            "paths",
            "--from",
            "张三",
            "--to",
            "王五",
            "--max-hops",
            "3",
        ])
        .expect("两点路径参数应可解析");
        match between.command {
            Some(Commands::Paths {
                to,
                via,
                common_with,
                max_hops,
                ..
            }) => {
                assert_eq!(to.as_deref(), Some("王五"));
                assert!(via.is_empty());
                assert!(common_with.is_none());
                assert_eq!(max_hops, 3);
            }
            _ => panic!("应解析为 Paths 子命令"),
        }

        let chain = Cli::try_parse_from([
            "knowprolog",
            "paths",
            "--from",
            "张三",
            "--via",
            "师傅",
            "--via",
            "师傅",
        ])
        .expect("谓词链参数应可解析");
        match chain.command {
            Some(Commands::Paths { to, via, .. }) => {
                assert!(to.is_none());
                assert_eq!(via, vec!["师傅".to_string(), "师傅".to_string()]);
            }
            _ => panic!("应解析为 Paths 子命令"),
        }

        let common = Cli::try_parse_from([
            "knowprolog",
            "paths",
            "--from",
            "张三",
            "--common-with",
            "李四",
        ])
        .expect("共同邻居参数应可解析");
        match common.command {
            Some(Commands::Paths { common_with, .. }) => {
                assert_eq!(common_with.as_deref(), Some("李四"));
            }
            _ => panic!("应解析为 Paths 子命令"),
        }

        assert!(Cli::try_parse_from(["knowprolog", "paths", "--from", "张三"]).is_err());
        assert!(Cli::try_parse_from([
            "knowprolog",
            "paths",
            "--from",
            "张三",
            "--to",
            "王五",
            "--via",
            "师傅",
        ])
        .is_err());
    }

    #[test]
    fn forge_cli_parses_repeated_examples_and_activation() {
        let cli = Cli::try_parse_from([
            "knowprolog",
            "forge",
            "--name",
            "女儿问法",
            "--route",
            "daughter_query",
            "--positive",
            "姚明的女儿是谁",
            "--positive",
            "姚明的闺女是谁",
            "--negative",
            "姚明今年几岁",
            "--priority",
            "30",
            "--activate",
            "--json",
        ])
        .expect("锻造参数应可解析");
        match cli.command {
            Some(Commands::Forge {
                name,
                route,
                positives,
                negatives,
                priority,
                activate,
                json,
            }) => {
                assert_eq!(name, "女儿问法");
                assert_eq!(route, "daughter_query");
                assert_eq!(positives.len(), 2);
                assert_eq!(negatives, vec!["姚明今年几岁".to_string()]);
                assert_eq!(priority, 30);
                assert!(activate);
                assert!(json);
            }
            _ => panic!("应解析为 Forge 子命令"),
        }
        assert!(Cli::try_parse_from(["knowprolog", "forge", "--name", "空"]).is_err());
    }

    #[test]
    fn rewrite_cli_parses_mine_and_rules() {
        let mine = Cli::try_parse_from([
            "knowprolog",
            "rewrite-mine",
            "--episodes",
            "episodes.jsonl",
            "--holdout-predicate",
            "邀请",
            "--holdout-predicate",
            "拜访",
            "--min-support",
            "3",
            "--save",
            "rewrite_book.json",
            "--strict",
            "--json",
        ])
        .expect("rewrite-mine 参数应可解析");
        match mine.command {
            Some(Commands::RewriteMine {
                episodes,
                holdout_predicates,
                min_support,
                save,
                strict,
                json,
            }) => {
                assert_eq!(episodes.as_deref(), Some("episodes.jsonl"));
                assert_eq!(holdout_predicates, vec!["邀请", "拜访"]);
                assert_eq!(min_support, 3);
                assert_eq!(save.as_deref(), Some("rewrite_book.json"));
                assert!(strict);
                assert!(json);
            }
            _ => panic!("应解析为 RewriteMine 子命令"),
        }

        let rules = Cli::try_parse_from([
            "knowprolog",
            "rewrite-rules",
            "--book",
            "book.json",
            "--adopt",
            "rw-1234",
            "--retire",
            "rw-5678",
            "--json",
        ])
        .expect("rewrite-rules 参数应可解析");
        match rules.command {
            Some(Commands::RewriteRules {
                book,
                adopt,
                retire,
                json,
            }) => {
                assert_eq!(book, "book.json");
                assert_eq!(adopt, vec!["rw-1234"]);
                assert_eq!(retire, vec!["rw-5678"]);
                assert!(json);
            }
            _ => panic!("应解析为 RewriteRules 子命令"),
        }
    }

    #[test]
    fn build_kg_cli_parses_feedback_and_apply() {
        let cli = Cli::try_parse_from([
            "knowprolog",
            "build-kg",
            "--text",
            "张三拜李四为师。",
            "--min-support",
            "3",
            "--min-purity",
            "750",
            "--limit",
            "12",
            "--feedback",
            "feedback.txt",
            "--apply",
            "--json",
        ])
        .expect("build-kg 参数应可解析");
        match cli.command {
            Some(Commands::BuildKg {
                text,
                min_support,
                min_purity,
                limit,
                feedback,
                apply,
                json,
                ..
            }) => {
                assert_eq!(text.as_deref(), Some("张三拜李四为师。"));
                assert_eq!(min_support, 3);
                assert_eq!(min_purity, 750);
                assert_eq!(limit, 12);
                assert_eq!(feedback.as_deref(), Some("feedback.txt"));
                assert!(apply);
                assert!(json);
            }
            _ => panic!("应解析为 BuildKg 子命令"),
        }
        assert!(Cli::try_parse_from([
            "knowprolog",
            "build-kg",
            "--text",
            "甲",
            "--file",
            "corpus.txt",
        ])
        .is_err());
    }

    #[test]
    fn owl_cli_forwards_arguments_and_parses_decisions() {
        let cli = Cli::try_parse_from(["knowprolog", "owl", "propose", "--min-support", "3"])
            .expect("owl 参数应转发");
        match cli.command {
            Some(Commands::Owl { args }) => {
                assert_eq!(args, vec!["propose", "--min-support", "3"]);
            }
            _ => panic!("应解析为 Owl 子命令"),
        }
        let cli =
            OwlCli::try_parse_from(["owl", "review", "owl-0001", "accept", "--reviewer", "alice"])
                .expect("owl review 参数应可解析");
        match cli.command {
            OwlCommands::Review {
                id,
                action,
                reviewer,
                ..
            } => {
                assert_eq!(id, "owl-0001");
                assert!(matches!(action, OwlReviewAction::Accept));
                assert_eq!(reviewer, "alice");
            }
            _ => panic!("应解析为 Review 子命令"),
        }
    }

    #[test]
    fn batch_teach_cli_parses_seed_corpus_and_decisions() {
        let cli = Cli::try_parse_from([
            "knowprolog",
            "batch-teach",
            "--facts",
            "facts.tsv",
            "--corpus",
            "corpus.txt",
            "--negatives",
            "negatives.txt",
            "--min-support",
            "3",
            "--min-purity",
            "900",
            "--max-candidates",
            "8",
            "--adopt",
            "bt-1234",
            "--retire",
            "bt-5678",
            "--json",
        ])
        .expect("batch-teach 参数应可解析");
        match cli.command {
            Some(Commands::BatchTeach {
                facts,
                corpus,
                negatives,
                min_support,
                min_purity,
                max_candidates,
                adopt,
                retire,
                json,
                ..
            }) => {
                assert_eq!(facts.as_deref(), Some("facts.tsv"));
                assert_eq!(corpus.as_deref(), Some("corpus.txt"));
                assert_eq!(negatives.as_deref(), Some("negatives.txt"));
                assert_eq!(min_support, 3);
                assert_eq!(min_purity, 900);
                assert_eq!(max_candidates, 8);
                assert_eq!(adopt, vec!["bt-1234".to_string()]);
                assert_eq!(retire, vec!["bt-5678".to_string()]);
                assert!(json);
            }
            _ => panic!("应解析为 BatchTeach 子命令"),
        }

        let cli = Cli::try_parse_from([
            "knowprolog",
            "batch-import",
            "corpus.txt",
            "--dry-run",
            "--json",
        ])
        .expect("batch-import 参数应可解析");
        match cli.command {
            Some(Commands::BatchImport {
                file_path,
                dry_run,
                json,
            }) => {
                assert_eq!(file_path, "corpus.txt");
                assert!(dry_run);
                assert!(json);
            }
            _ => panic!("应解析为 BatchImport 子命令"),
        }
    }

    #[test]
    fn command_like_covers_trust_and_correction_commands() {
        for command in [
            "更正：姚明的身高从226改为229",
            "纠正：姚明|height|226|229",
            "修正：姚明的身高从226改为229",
            "历史可信度",
            "历史可信度：姚明的身高是多少",
            "可信度：姚明的身高是多少",
            "信任报告",
            "改口：姚明的身高是229cm",
            "改口史：姚明 身高",
            "槽账本",
        ] {
            assert!(is_command_like(command), "{} 应被识别为系统命令", command);
        }
    }

    #[test]
    fn recovery_confirmation_bridges_into_reasoner_premises() {
        use know_prolog::recovery::{ConfirmationSource, RecoveryState};

        let mut state = RecoveryState::new();
        state.record_confirmed(
            "王五",
            "师傅",
            "赵六",
            "receipt-1",
            ConfirmationSource::Provisional,
        );
        let mut manager = DialogueManager::new();
        let mut snapshot = manager.snapshot();
        snapshot.recovery = state;
        manager.restore(&snapshot);

        let mut reasoner =
            SymbolicReasoner::new(know_prolog::ReasoningConfig::default(), HashMap::new());
        assert_eq!(
            sync_recovery_confirmed_facts(&manager, &mut reasoner),
            1,
            "首次同步应写入一条会话事实"
        );
        assert!(reasoner
            .session_facts()
            .iter()
            .any(|atom| atom.display() == "师傅(王五, 赵六)"));
        assert_eq!(
            sync_recovery_confirmed_facts(&manager, &mut reasoner),
            0,
            "重复同步必须幂等"
        );
        let proof = reasoner.resolution_proof_report("王五的师傅是赵六");
        assert!(!proof.contains("未得证"), "{}", proof);
    }

    /// 统一模式库命令：库视图 / 匹配解释 / 教学模式 / 忘模式都要被识别为命令。
    #[test]
    fn command_like_covers_pattern_library_commands() {
        for command in [
            "模式路由",
            "查询模式库",
            "统一模式库",
            "模式关系",
            "模式关系图",
            "模式落图",
            "同步模式图",
            "模式匹配：法国的首都是哪里",
            "模式经验",
            "经验台账",
            "模式联想：德国的首都在哪里",
            "确认联想：德国的首都在哪里",
            "采纳联想：德国的首都在哪里",
            "忘经验：a1b2c3d4",
            "遗忘经验：法国首都查询",
            "清空模式经验",
            "教模式：老家 | {人} 的 老家 | hometown_query | 20 | 焦点",
            "忘模式：老家",
            "锻造模式：女儿问法 | 姚明的女儿是谁 | 姚明今年几岁 | daughter_query",
            "锻造模式",
            "确认锻造 forge_abc123",
            "否认锻造 abc123",
            "退役锻造 abc123",
            "遗忘锻造 abc123",
            "锻造台账",
            "锻造探针",
        ] {
            assert!(is_command_like(command), "{} 应被识别为系统命令", command);
        }
    }

    /// RDF 导入后的画像同步必须落盘，且重新打开存储仍能解析导入的谓词与实体。
    #[test]
    fn entity_thread_commands_are_command_like() {
        for command in [
            "实体足迹：张三",
            "实体足迹:张三",
            "实体足迹 张三",
            "实体索引",
            "实体线程",
            "实体审计",
            "未解析提及",
            "未解析实体",
        ] {
            assert!(is_command_like(command), "{} 应视为命令", command);
        }
    }

    #[test]
    fn chronicle_refs_only_write_linked_mentions() {
        let reasoner = SymbolicReasoner::new(
            know_prolog::ReasoningConfig::default(),
            std::collections::HashMap::new(),
        );
        assert!(chronicle_entity_refs(&reasoner, "今天天气不错").is_empty());
    }

    #[test]
    fn entity_footprint_reuses_situational_change_detection() {
        use know_prolog::entity_registry::EntityRegistry;

        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("chronicle.jsonl");
        let (mut chronicle, _) =
            Chronicle::open(path.to_str().unwrap(), FsyncPolicy::EachAppend).unwrap();
        chronicle
            .append_turn("张三的身高是多少", "query", None, 0)
            .unwrap();
        chronicle.append_reply("226 厘米", &[], &[]).unwrap();
        chronicle.set_kb_revision(1);
        chronicle
            .append_turn("张三的身高是多少呢", "query", None, 0)
            .unwrap();
        chronicle.append_reply("228 厘米", &[], &[]).unwrap();

        let mut registry = EntityRegistry::new();
        registry.observe_kb_entity("张三", &BTreeSet::new(), &[], "test");
        let digest = EntityThreadDigest::build(&chronicle, &registry);
        let report = entity_related_changes(&digest, &chronicle, "张三")
            .expect("同一问题两次不同回答应报告改口");
        assert!(report.contains("226 厘米"), "{}", report);
        assert!(report.contains("228 厘米"), "{}", report);
        assert!(report.contains("知识版本变化"), "{}", report);
        assert!(entity_related_changes(&digest, &chronicle, "不存在").is_none());
    }

    #[tokio::test]
    async fn imported_graph_persists_predicate_and_entity_registries() {
        let dir = tempfile::TempDir::new().expect("临时目录");
        let store_path = dir.path().to_str().expect("临时目录路径");
        let rdf_path = dir.path().join("sample.ttl");
        std::fs::write(
            &rdf_path,
            r#"@prefix kg: <http://example.org/kg/> .
<http://example.org/kg/entity/张三> kg:name "张三" .
<http://example.org/kg/entity/张三> <http://example.org/kg/认识> <http://example.org/kg/entity/李四> .
"#,
        )
        .expect("写入测试 RDF");

        let manager = OxigraphManager::new(store_path).expect("打开测试存储");
        let added = manager
            .import_rdf_file(rdf_path.to_str().expect("RDF 路径"))
            .await
            .expect("导入测试 RDF");
        assert_eq!(added, 2);
        persist_registry_observations(&manager, store_path)
            .await
            .expect("同步画像注册表");
        drop(manager);

        let predicate_path = format!("{}/predicate_registry.json", store_path);
        let entity_path = format!("{}/entity_registry.json", store_path);
        assert!(std::path::Path::new(&predicate_path).exists());
        assert!(std::path::Path::new(&entity_path).exists());

        let reopened = OxigraphManager::new(store_path).expect("重新打开测试存储");
        assert!(reopened.load_predicate_registry(&predicate_path));
        assert!(reopened.predicate_registry_snapshot().resolve("认识").known);
        assert!(reopened.load_entity_registry(&entity_path));
        assert_eq!(
            reopened.resolve_entity_identity("张三").canonical(),
            Some("张三")
        );
    }
    #[test]
    fn synonyms_cli_parses_resolution_and_seed() {
        let cli = Cli::try_parse_from([
            "knowprolog",
            "synonyms",
            "--surface",
            "苹果",
            "--kind",
            "entity",
            "--type",
            "组织",
            "--echo",
            "苹果公司",
            "--seed-config",
            "--json",
        ])
        .expect("synonyms 解析参数应可解析");
        match cli.command {
            Some(Commands::Synonyms {
                surface,
                canonical,
                kind,
                expected_type,
                echo,
                seed_config,
                json,
                ..
            }) => {
                assert_eq!(surface.as_deref(), Some("苹果"));
                assert!(canonical.is_none());
                assert_eq!(kind, "entity");
                assert_eq!(expected_type.as_deref(), Some("组织"));
                assert_eq!(echo, vec!["苹果公司".to_string()]);
                assert!(seed_config);
                assert!(json);
            }
            other => panic!("应解析为 Synonyms：{:?}", other),
        }

        assert!(Cli::try_parse_from([
            "knowprolog",
            "synonyms",
            "--canonical",
            "妻子",
            "--kind",
            "predicate",
        ])
        .is_ok());
        assert!(Cli::try_parse_from(["knowprolog", "synonyms", "--kind", "unknown"]).is_ok());
    }

}

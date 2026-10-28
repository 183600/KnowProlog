    fn trace_nlp_config() -> NlpConfig {
        NlpConfig {
            import_keywords: Vec::new(),
            pos_dict: HashMap::new(),
            name_patterns: crate::NamePatterns {
                prefixes: Vec::new(),
                suffixes: Vec::new(),
                full_names: vec!["张三".to_string(), "李四".to_string(), "王五".to_string()],
                profession_suffix: Vec::new(),
            },
            enable_nlu: None,
            nlu_confidence_threshold: None,
            enable_local_semantic: None,
            disable_ownthink_api: None,
            semantic_similarity_threshold: None,
            max_entity_candidates: None,
            enable_advanced_entity_extraction: None,
            enable_relation_extraction: None,
            context_window_size: None,
        }
    }

    #[test]
    fn failed_answer_records_classified_read_trace_without_raw_text() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.set_nlp_config(trace_nlp_config());
        assert!(reasoner.read_trace_receipts().is_empty());

        let answer = reasoner.answer("未名科技公司的老板是谁");
        assert!(answer.is_none(), "未知实体不应被静默猜测：{:?}", answer);
        let receipts = reasoner.read_trace_receipts();
        assert_eq!(receipts.len(), 1, "失败轮应留下一条有界回执");
        let receipt = &receipts[0];
        assert!(
            receipt
                .failure_kinds
                .iter()
                .any(|kind| kind == "unknown_entity"),
            "失败应分类为未知实体：{:?}",
            receipt
        );
        assert_eq!(receipt.answered, Some(false));
        assert!(
            receipt.source_counts.values().sum::<u64>() > 0,
            "回执应记录词性来源分布"
        );
        let report = reasoner.read_trace_ledger_report();
        assert!(!report.contains("未名科技"), "台账不得保存原文：{}", report);

        // 成功轮不做额外失败记录（也只观测失败）。
        let answer = reasoner.answer("张三的师傅是谁");
        assert!(answer.is_some(), "已登记实体应正常回答");
        assert_eq!(reasoner.read_trace_receipts().len(), 1);
    }

    #[test]
    fn reading_residual_locates_unknown_entity_and_names_repair() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.set_nlp_config(trace_nlp_config());

        let reading = reasoner
            .read_question("未名科技公司的老板是谁")
            .expect("应返回读法残差表");
        assert_eq!(reading.status(), crate::reading::ReadingStatus::Partial);
        let residual = reading
            .residuals
            .iter()
            .find(|residual| residual.kind == ResidualKind::UnknownEntity)
            .expect("应定位到未知实体残差");
        assert_eq!(residual.stage, ResidualStage::Link);
        assert_eq!(residual.span, Some(TextSpan::new(0, 6)));
        assert_eq!(residual.surface.as_deref(), Some("未名科技公司"));
        assert_eq!(residual.expected.as_deref(), Some("已登记实体"));
        assert_eq!(residual.repair, RepairKind::SupplyEntity);
        assert!(reading
            .residual_kinds()
            .iter()
            .any(|kind| kind == "unknown_entity"));

        let report = reasoner.lexical_trace_report("未名科技公司的老板是谁");
        assert!(report.contains("读法状态：部分可读"), "{}", report);
        assert!(report.contains("未知实体"), "{}", report);
        assert!(report.contains("0..6"), "{}", report);
    }

    #[test]
    fn reading_uses_dialogue_context_to_fill_residual_candidates() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.set_nlp_config(trace_nlp_config());
        reasoner.contextualize("张三的师傅是谁");

        let ellipsis = reasoner.read_question("李四呢").expect("读法");
        let residual = ellipsis
            .residuals
            .iter()
            .find(|residual| residual.kind == ResidualKind::Ellipsis)
            .expect("省略残差");
        assert!(
            residual
                .candidates
                .iter()
                .any(|candidate| candidate == "师傅"),
            "上一轮谓词应成为候选：{:?}",
            residual.candidates
        );

        let missing = reasoner.read_question("身高是多少").expect("读法");
        let residual = missing
            .residuals
            .iter()
            .find(|residual| residual.kind == ResidualKind::MissingEntity)
            .expect("缺实体残差");
        assert!(
            residual
                .candidates
                .iter()
                .any(|candidate| candidate == "张三"),
            "当前焦点应成为候选：{:?}",
            residual.candidates
        );
    }

    #[test]
    fn degree_question_reaches_attribute_answer_path() {
        let facts = vec![PrologFact::Entity {
            name: "姚明".to_string(),
            entity_type: "Person".to_string(),
            properties: HashMap::from([("身高".to_string(), "226cm".to_string())]),
        }];
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&facts);
        let answer = reasoner.answer("姚明多高").expect("程度问句应可回答");
        assert!(answer.contains("226cm"), "实际回答：{}", answer);
    }

    #[test]
    fn relation_answer_normalizes_entity_iri_and_dedups_channels() {
        let mut facts = master_facts();
        facts.push(PrologFact::Entity {
            name: "张三".to_string(),
            entity_type: "Person".to_string(),
            properties: HashMap::from([
                ("name".to_string(), "张三".to_string()),
                (
                    "师傅".to_string(),
                    crate::identity::entity_uri(crate::identity::DEFAULT_BASE_NS, "李四"),
                ),
            ]),
        });
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&facts);
        let answer = reasoner.answer("张三的师傅是谁").expect("应回答直接关系");
        assert!(
            !answer.contains("http://example.org/kg/entity/李四"),
            "实体 IRI 不应作为取值展示：{}",
            answer
        );
        assert_eq!(
            answer.matches("李四").count(),
            1,
            "关系索引与属性索引的同一取值应去重：{}",
            answer
        );
    }

    #[test]
    fn lexical_trace_report_exposes_marker_boundary_without_changing_answer() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.set_nlp_config(trace_nlp_config());

        let report = reasoner.lexical_trace_report("张三的所有权归公司");
        assert!(report.contains("读法追踪"), "{}", report);
        assert!(report.contains("边界越界"), "{}", report);
        assert!(report.contains("事后重建"), "{}", report);
        assert!(!report.contains("失败分类：无"), "{}", report);

        // 观测不改变回答路径：普通闭包/多跳问句仍走原实现。
        let answer = reasoner.answer("张三的师傅的师傅是谁");
        assert!(answer.is_some(), "原多跳回答不应被追踪改变");
    }

    #[test]
    fn answer_certain_does_not_treat_link_prediction_as_answer() {
        let mut config = test_config();
        config.link_prediction_min_confidence = 0.1;
        let mut facts = master_facts();
        for (from, to) in [("孙八", "周九"), ("周九", "吴十")] {
            facts.push(PrologFact::Relation {
                from: from.to_string(),
                rel_type: "师傅".to_string(),
                to: to.to_string(),
            });
        }
        let reasoner = {
            let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
            reasoner.observe_facts(&facts);
            reasoner
        };
        let full = reasoner.answer("孙八的师祖是谁").expect("预测通道应有候选");
        assert!(
            full.contains("待确认") && full.contains("置信度"),
            "{}",
            full
        );
        assert!(
            reasoner.answer_certain("孙八的师祖是谁").is_none(),
            "确认前的预测不得走确证通道"
        );
    }

    #[test]
    fn evidence_answer_requires_the_target_predicate() {
        let facts = vec![
            PrologFact::Entity {
                name: "姚明".to_string(),
                entity_type: "人物".to_string(),
                properties: HashMap::from([("身高".to_string(), "226".to_string())]),
            },
            PrologFact::Entity {
                name: "叶莉".to_string(),
                entity_type: "人物".to_string(),
                properties: HashMap::from([("身高".to_string(), "190".to_string())]),
            },
            PrologFact::Entity {
                name: "李娜".to_string(),
                entity_type: "人物".to_string(),
                properties: HashMap::from([("体重".to_string(), "55".to_string())]),
            },
            PrologFact::Relation {
                from: "姚明".to_string(),
                rel_type: "配偶".to_string(),
                to: "叶莉".to_string(),
            },
            PrologFact::Relation {
                from: "叶莉".to_string(),
                rel_type: "配偶".to_string(),
                to: "姚明".to_string(),
            },
        ];
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&facts);

        // 「配偶」只是话题；目标是「体重」。不能拿配偶证据冒充确定答案。
        assert!(
            reasoner.answer_certain("姚明的配偶的体重是多少").is_none(),
            "无目标谓词证据时必须弃权"
        );
        assert!(
            reasoner.answer_certain("姚明的女儿的体重是多少").is_none(),
            "同类链式未知也必须弃权"
        );
        // 目标谓词确有证据时仍需正常作答，不能把闸门做成一律弃权。
        let relation = reasoner
            .answer_certain("姚明的配偶是谁")
            .expect("配偶关系本身可证");
        assert!(relation.contains("叶莉"), "{}", relation);
        let attribute = reasoner
            .answer_certain("姚明的配偶的身高是多少")
            .expect("配偶身高可经多跳证明");
        assert!(attribute.contains("190"), "{}", attribute);
    }

    #[test]
    fn dialogue_confirmation_turns_prediction_into_next_turn_premise() {
        let mut config = test_config();
        config.link_prediction_min_confidence = 0.1;
        let mut facts = master_facts();
        for (from, to) in [("孙八", "周九"), ("周九", "吴十")] {
            facts.push(PrologFact::Relation {
                from: from.to_string(),
                rel_type: "师傅".to_string(),
                to: to.to_string(),
            });
        }
        for (a, b, c) in [
            ("郑十一", "钱十二", "孙十三"),
            ("钱十三", "李十四", "周十五"),
            ("吴十六", "郑十七", "王十八"),
        ] {
            for (from, to) in [(a, b), (b, c)] {
                facts.push(PrologFact::Relation {
                    from: from.to_string(),
                    rel_type: "师傅".to_string(),
                    to: to.to_string(),
                });
            }
            facts.push(PrologFact::Relation {
                from: a.to_string(),
                rel_type: "师祖".to_string(),
                to: c.to_string(),
            });
        }
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        reasoner.observe_facts(&facts);

        let mut manager = crate::dialogue_manager::DialogueManager::new();
        manager.set_recovery_mode(crate::recovery::RecoveryMode::On);
        let first = manager
            .plan_reply("孙八的师祖是谁", &reasoner)
            .expect("预测轮应被对话层接管");
        assert!(first.text.contains("推测"), "{}", first.text);
        assert!(first.text.contains("吴十"), "{}", first.text);
        assert_eq!(manager.recovery_state().pending.len(), 1);

        let confirmed = manager
            .plan_pure_dialogue("对", &reasoner)
            .expect("确认应结算挂账");
        assert!(confirmed.text.contains("记入本会话"), "{}", confirmed.text);
        assert_eq!(
            manager
                .recovery_state()
                .confirmed_value("孙八", "师祖")
                .map(|fact| fact.value.as_str()),
            Some("吴十")
        );

        let second = manager
            .plan_reply("孙八的师祖是谁", &reasoner)
            .expect("确认后下一轮应作答");
        assert!(second.text.contains("吴十"), "{}", second.text);
        assert!(second.text.contains("本会话确认"), "{}", second.text);
    }

    #[test]
    fn completion_episode_confirms_batch_and_promotes_to_global_rule() {
        let mut config = test_config();
        config.link_prediction_min_confidence = 0.1;
        let mut facts = master_facts();
        for (from, to) in [
            ("孙八", "周九"),
            ("周九", "吴十"),
            ("郑十一", "钱十二"),
            ("钱十二", "孙十三"),
        ] {
            facts.push(PrologFact::Relation {
                from: from.to_string(),
                rel_type: "师傅".to_string(),
                to: to.to_string(),
            });
        }
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        reasoner.observe_facts(&facts);

        let opened = reasoner.completion_open("孙八");
        assert!(opened.contains("补全回合"), "{}", opened);
        assert!(opened.contains("孙八"), "{}", opened);
        assert!(opened.contains("师祖"), "{}", opened);

        let confirmed = reasoner.completion_confirm("孙八|师祖|吴十");
        assert!(confirmed.contains("已确认补全"), "{}", confirmed);
        assert!(confirmed.contains("本批"), "{}", confirmed);
        assert!(
            confirmed.contains("郑十一"),
            "批次闭包应覆盖同一关系族的其他实体：{}",
            confirmed
        );
        assert!(
            reasoner
                .session_facts()
                .iter()
                .any(|atom| atom.predicate.as_ref() == "师祖"
                    && atom.terms.iter().any(|term| term.name() == "孙八")
                    && atom.terms.iter().any(|term| term.name() == "吴十")),
            "确认的补全应成为会话事实"
        );
        let answer = reasoner.answer("孙八的师祖是谁").expect("确认后应可回答");
        assert!(
            answer.contains("吴十"),
            "确认的补全必须在下一轮回答中生效：{}",
            answer
        );
        assert_eq!(
            reasoner.completion_ledger().reasoning_clauses().len(),
            0,
            "本批规则不应自动进入全局归结"
        );

        let rule_id = reasoner.completion_ledger().rules()[0].id.clone();
        let namespaced = reasoner.completion_promote(&rule_id, "命名空间");
        assert!(namespaced.contains("命名空间"), "{}", namespaced);
        assert!(
            namespaced.contains("尚未进入全局归结"),
            "命名空间晋升不得冒充全局：{}",
            namespaced
        );
        assert_eq!(
            reasoner.completion_ledger().reasoning_clauses().len(),
            0,
            "命名空间规则不得静默进入全局归结"
        );
        assert!(
            !reasoner
                .resolver()
                .all_clauses()
                .iter()
                .any(|clause| clause.source.contains("命名空间")),
            "命名空间规则不得出现在全局归结引擎里"
        );
        let promoted = reasoner.completion_promote(&rule_id, "全局");
        assert!(promoted.contains("全局"), "{}", promoted);
        assert_eq!(
            reasoner.completion_ledger().reasoning_clauses().len(),
            1,
            "显式晋升到全局后规则才进入全局归结"
        );
        assert!(
            reasoner
                .resolver()
                .all_clauses()
                .iter()
                .any(|clause| clause.source.contains("全局")),
            "晋升到全局后规则应真正进入归结引擎"
        );
        assert!(reasoner.completion_rules_report().contains("全局"));

        let short = &rule_id[..rule_id.len().min(21)];
        let retired = reasoner.completion_retire(short);
        assert!(retired.contains("已退役"), "{}", retired);
        assert_eq!(
            reasoner.completion_ledger().reasoning_clauses().len(),
            0,
            "退役后规则不得继续参与全局归结"
        );
        assert!(
            reasoner
                .answer("孙八的师祖是谁")
                .expect("会话事实应保留")
                .contains("吴十"),
            "退役规则不得撤回已确认的会话事实"
        );
        let restored_rule = reasoner.completion_reactivate(&rule_id);
        assert!(restored_rule.contains("已恢复"), "{}", restored_rule);
        assert_eq!(
            reasoner.completion_ledger().reasoning_clauses().len(),
            1,
            "恢复后规则应重新参与全局归结"
        );

        let snapshot = reasoner.dialogue_snapshot();
        assert!(snapshot.completion.is_some(), "非空补全台账应进入会话快照");
        let mut restored = SymbolicReasoner::new(test_config(), HashMap::new());
        restored
            .restore_dialogue_snapshot(&snapshot)
            .expect("补全台账快照应可恢复");
        assert_eq!(
            restored.completion_ledger(),
            reasoner.completion_ledger(),
            "补全回合应原样跨会话恢复"
        );
        assert_eq!(
            restored.completion_ledger().reasoning_clauses().len(),
            1,
            "恢复后已晋升的全局规则仍应参与归结"
        );
        assert!(
            restored
                .answer("孙八的师祖是谁")
                .expect("恢复后应可回答")
                .contains("吴十"),
            "恢复的会话事实必须继续生效"
        );

        let path = std::env::temp_dir().join(format!(
            "knowprolog_completion_state_{}.json",
            std::process::id()
        ));
        let path_str = path.to_string_lossy().to_string();
        reasoner.save_dialogue_state(&path_str).expect("应能保存");
        let mut loaded = SymbolicReasoner::new(test_config(), HashMap::new());
        assert!(
            loaded.load_dialogue_state(&path_str),
            "补全回合应能随对话状态落盘并读回"
        );
        assert_eq!(
            loaded.completion_ledger(),
            reasoner.completion_ledger(),
            "落盘往返后补全台账应保持不变"
        );
        let _ = std::fs::remove_file(&path);

        let fork = reasoner.fork_session();
        assert!(
            fork.completion_ledger().is_empty(),
            "分叉会话不得继承补全回合"
        );
        assert_eq!(
            fork.completion_ledger().reasoning_clauses().len(),
            0,
            "分叉后已晋升规则不得残留"
        );
    }

    #[test]
    fn completion_command_entry_point_matches_cli_semantics() {
        let mut config = test_config();
        config.link_prediction_min_confidence = 0.1;
        let mut facts = master_facts();
        for (from, to) in [
            ("孙八", "周九"),
            ("周九", "吴十"),
            ("郑十一", "钱十二"),
            ("钱十二", "孙十三"),
        ] {
            facts.push(PrologFact::Relation {
                from: from.to_string(),
                rel_type: "师傅".to_string(),
                to: to.to_string(),
            });
        }
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        reasoner.observe_facts(&facts);

        assert!(reasoner
            .handle_completion_command("姚明的身高是多少")
            .is_none());
        let opened = reasoner
            .handle_completion_command("补全回合：孙八")
            .expect("打开回合应被命令入口接管");
        assert!(opened.contains("补全回合"), "{}", opened);
        assert!(opened.contains("师祖"), "{}", opened);

        let confirmed = reasoner
            .handle_completion_command("确认补全：孙八|师祖|吴十")
            .expect("确认命令应被接管");
        assert!(confirmed.contains("已确认补全"), "{}", confirmed);
        assert!(
            reasoner
                .session_facts()
                .iter()
                .any(|atom| atom.predicate.as_ref() == "师祖"
                    && atom.terms.iter().any(|term| term.name() == "孙八")
                    && atom.terms.iter().any(|term| term.name() == "吴十")),
            "命令入口确认的候选必须落地为会话事实"
        );

        let rules = reasoner
            .handle_completion_command("补全规则")
            .expect("规则报告应被接管");
        assert!(rules.contains("补全规则"), "{}", rules);
        assert!(reasoner
            .handle_completion_command("补全报告")
            .expect("回合报告应被接管")
            .contains("孙八"));
        assert!(reasoner
            .handle_completion_command("晋升补全规则：")
            .expect("空参数应给出用法")
            .contains("用法"));
        assert!(
            opened.contains("下一问"),
            "打开回合应给出一个提问：{}",
            opened
        );
        assert!(reasoner
            .handle_completion_command("补全提问")
            .expect("提问命令应被接管")
            .contains("补全提问"));
        assert!(reasoner
            .handle_completion_command("别问了")
            .expect("静音命令应被接管")
            .contains("静音"));
        assert!(reasoner
            .handle_completion_command("补全提问")
            .expect("静音后仍应回执")
            .contains("已静音"));
        assert!(reasoner
            .handle_completion_command("允许提问")
            .expect("恢复提问应被接管")
            .contains("已恢复"));
    }

    #[test]
    fn dialogue_verdict_feeds_completion_ledger_and_stops_renomination() {
        let mut config = test_config();
        config.link_prediction_min_confidence = 0.1;
        let mut facts = master_facts();
        for (from, to) in [
            ("孙八", "周九"),
            ("周九", "吴十"),
            ("郑十一", "钱十二"),
            ("钱十二", "孙十三"),
        ] {
            facts.push(PrologFact::Relation {
                from: from.to_string(),
                rel_type: "师傅".to_string(),
                to: to.to_string(),
            });
        }
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        reasoner.observe_facts(&facts);

        assert!(
            reasoner
                .recovery_nominations("孙八", "师祖")
                .iter()
                .any(|nomination| nomination.value == "吴十"),
            "确认前应可提名吴十"
        );
        assert!(reasoner.note_dialogue_verdict("孙八", "师祖", "吴十", true, "receipt-1"));
        assert_eq!(reasoner.completion_ledger().rules().len(), 1);
        assert!(
            reasoner.recovery_nominations("孙八", "师祖").is_empty(),
            "已确认的候选不应再次追问"
        );
        assert!(reasoner.completion_regression_report().contains("确认 1"));
        assert!(!reasoner.note_dialogue_verdict("孙八", "师祖", "吴十", true, "receipt-1"));

        assert!(reasoner.note_dialogue_verdict("郑十一", "师祖", "孙十三", false, "receipt-2"));
        assert!(
            reasoner
                .recovery_nominations("郑十一", "师祖")
                .iter()
                .all(|nomination| nomination.value != "孙十三"),
            "已否决的候选不得再次提名"
        );
        assert!(reasoner.completion_regression_report().contains("驳回 1"));
    }

    #[test]
    fn knowledge_layer_manifest_separates_base_session_and_speculation() {
        let mut config = test_config();
        config.link_prediction_min_confidence = 0.1;
        let mut facts = master_facts();
        for (from, to) in [
            ("孙八", "周九"),
            ("周九", "吴十"),
            ("郑十一", "钱十二"),
            ("钱十二", "孙十三"),
        ] {
            facts.push(PrologFact::Relation {
                from: from.to_string(),
                rel_type: "师傅".to_string(),
                to: to.to_string(),
            });
        }
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        reasoner.observe_facts(&facts);

        let manifest = reasoner.knowledge_layer_manifest();
        assert_eq!(
            manifest.revision,
            reasoner.revision(),
            "清单必须锚定知识版本"
        );
        assert_eq!(manifest.layers.len(), 4);
        assert_eq!(
            manifest
                .layers
                .iter()
                .map(|layer| layer.id.as_str())
                .collect::<Vec<_>>(),
            vec!["L0", "L1", "L2", "L3"],
            "层次顺序必须稳定"
        );
        let base = manifest.layer("L0").expect("基础层必须存在");
        assert!(base.count > 0, "母版事实应进入基础层");
        assert_eq!(base.digest.len(), 16, "摘要应是 16 位十六进制");
        assert!(manifest.layer("L1").expect("会话层").empty);
        assert_eq!(manifest.digest.len(), 16);
        assert_eq!(
            manifest,
            reasoner.knowledge_layer_manifest(),
            "只读导出必须确定"
        );

        let opened = reasoner
            .handle_completion_command("补全回合：孙八")
            .expect("打开回合");
        assert!(opened.contains("补全回合"), "{}", opened);
        let with_candidates = reasoner.knowledge_layer_manifest();
        assert!(
            with_candidates.layer("L2").expect("推测层").count > 0,
            "待确认候选必须出现在推测层"
        );

        let confirmed = reasoner
            .handle_completion_command("确认补全：孙八|师祖|吴十")
            .expect("确认候选");
        assert!(confirmed.contains("已确认补全"), "{}", confirmed);
        let with_session = reasoner.knowledge_layer_manifest();
        assert!(
            with_session.layer("L1").expect("会话层").count > 0,
            "确认后的会话事实必须与基础层分列"
        );
        assert_ne!(
            manifest.digest, with_session.digest,
            "状态变化必须改变合成摘要"
        );
        let text = reasoner.knowledge_layers_report();
        assert!(text.contains("知识容器层次清单"), "{}", text);
        assert!(text.contains("L3 假设层"), "{}", text);
    }

    #[test]
    fn knowledge_layer_manifest_lists_predictive_hypotheses() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner
            .predictive_elaboration("李四")
            .expect("链接预测应生成推测段落");
        let manifest = reasoner.knowledge_layer_manifest();
        let speculative = manifest.layer("L2").expect("推测层");
        assert!(speculative.count > 0, "推测层应包含待定预测");
        assert!(
            speculative
                .sample
                .iter()
                .any(|entry| entry.contains("预测") && entry.contains("赵六")),
            "{:?}",
            speculative.sample
        );
    }

    #[test]
    fn belief_rule_preview_reports_plan_without_teaching() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let before = reasoner.resolver.clause_count();
        let report = reasoner.belief_rule_preview("如果X的师傅是Y，那么X的老师是Y");
        assert!(report.contains("修订计划"), "{}", report);
        assert!(report.contains("影响"), "{}", report);
        assert_eq!(
            reasoner.resolver.clause_count(),
            before,
            "预演不得把规则放进会话规则库"
        );
    }

    #[test]
    fn assert_nl_fact_blocks_functional_conflict_with_minimal_question() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        let mut registry = PredicateRegistry::new();
        registry.observe("张三", "年龄", "30", None);
        assert!(registry.mark_functional("年龄", true));
        reasoner.configure_laws(&registry);
        let mut properties = HashMap::new();
        properties.insert("年龄".to_string(), "30".to_string());
        reasoner.observe_facts(&[PrologFact::Entity {
            name: "张三".to_string(),
            entity_type: "人".to_string(),
            properties,
        }]);
        let before = reasoner.session_facts().len();

        let blocked = reasoner.assert_nl_fact("张三的年龄是31");
        assert!(blocked.contains("最小冲突核"), "{}", blocked);
        assert!(blocked.contains("未提交"), "{}", blocked);
        assert!(blocked.contains("30"), "{}", blocked);
        assert!(blocked.contains("31"), "{}", blocked);
        assert_eq!(
            reasoner.session_facts().len(),
            before,
            "被阻止的写入不得留下会话事实"
        );
        assert!(
            !reasoner.session_revision_report().contains("session:"),
            "被阻止的写入不得生成修订回执"
        );
        let answer = reasoner.answer("张三的年龄是多少").unwrap_or_default();
        assert!(!answer.contains("31"), "冲突值不得进入回答：{}", answer);

        let accepted = reasoner.assert_nl_fact("张三的年龄是30");
        assert!(accepted.contains("修订回执"), "{}", accepted);
    }

    #[test]
    fn accepted_fact_reports_derived_impact() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let learned = reasoner.learn_rule_report("如果X的师傅是Y，那么X的老师是Y");
        assert!(learned.contains("已学习"), "{}", learned);

        let added = reasoner.assert_nl_fact("钱七是孙八的师傅");
        assert!(added.contains("修订回执"), "{}", added);
        assert!(
            added.contains("预检影响：派生 +1"),
            "接受写入时应报告规则派生影响：{}",
            added
        );
    }

    #[test]
    fn session_revision_undo_restores_fact_and_can_redo() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let added = reasoner.assert_nl_fact("钱七是孙八的师傅");
        assert!(added.contains("修订回执"), "{}", added);
        assert_eq!(reasoner.session_facts().len(), 1);
        assert!(reasoner.session_revision_report().contains("session:"));

        let undone = reasoner.undo_last_session_revision();
        assert!(undone.contains("已撤销会话修订"), "{}", undone);
        assert!(reasoner.session_facts().is_empty(), "撤销后事实应清空");
        let proof = reasoner.resolution_proof_report("孙八的师傅是钱七");
        assert!(proof.contains("未得证"), "撤销后不应再得证：{}", proof);

        let redone = reasoner.undo_last_session_revision();
        assert!(redone.contains("恢复事实 1 条"), "{}", redone);
        assert_eq!(reasoner.session_facts().len(), 1, "再次撤销应恢复事实");
        let proof = reasoner.resolution_proof_report("孙八的师傅是钱七");
        assert!(proof.contains("得证"), "恢复后应重新得证：{}", proof);
    }

    #[test]
    fn session_receipts_survive_snapshot_and_support_undo() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.assert_nl_fact("钱七是孙八的师傅");
        let path =
            std::env::temp_dir().join(format!("knowprolog_receipts_{}.json", std::process::id()));
        let path_str = path.to_string_lossy().to_string();
        reasoner.save_dialogue_state(&path_str).expect("应能保存");

        let mut restored = SymbolicReasoner::new(test_config(), HashMap::new());
        restored.observe_facts(&master_facts());
        assert!(restored.load_dialogue_state(&path_str), "应能恢复对话状态");
        assert_eq!(restored.session_facts().len(), 1);
        assert!(
            restored.session_revision_report().contains("session:"),
            "修订回执应随快照恢复：{}",
            restored.session_revision_report()
        );
        let undone = restored.undo_last_session_revision();
        assert!(undone.contains("已撤销会话修订"), "{}", undone);
        assert!(restored.session_facts().is_empty());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn forget_rule_undo_restores_clause_and_lesson() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.teach_report("如果X的师傅是Y，那么X的老师是Y");
        assert_eq!(reasoner.rule_ledger_entries().len(), 1);
        assert!(!reasoner.lessons().is_empty());

        let forgotten = reasoner.forget_rule_report("如果X的师傅是Y，那么X的老师是Y");
        assert!(forgotten.contains("修订回执"), "{}", forgotten);
        assert!(reasoner.rule_ledger_entries().is_empty());
        assert!(reasoner.lessons().is_empty());

        let undone = reasoner.undo_last_session_revision();
        assert!(undone.contains("恢复事实 0 条"), "{}", undone);
        assert!(undone.contains("规则 1 条"), "{}", undone);
        assert!(undone.contains("教学原句 +1"), "{}", undone);
        assert_eq!(reasoner.rule_ledger_entries().len(), 1);
        assert!(!reasoner.lessons().is_empty());
        let proof = reasoner.resolution_proof_report("李四的老师是王五");
        assert!(proof.contains("得证"), "撤销忘记后规则应恢复：{}", proof);

        let redone = reasoner.undo_last_session_revision();
        assert!(redone.contains("教学原句 +0 / -1"), "{}", redone);
        assert!(
            reasoner.rule_ledger_entries().is_empty(),
            "重做后规则应再次撤回"
        );
        assert!(reasoner.lessons().is_empty(), "重做后教学原句也应撤回");
    }

    fn graph_pattern_facts() -> Vec<PrologFact> {
        let empty = HashMap::new();
        let mut yao = HashMap::new();
        yao.insert("出生地".to_string(), "上海".to_string());
        let mut liu = HashMap::new();
        liu.insert("出生地".to_string(), "上海".to_string());
        vec![
            PrologFact::Entity {
                name: "张三".to_string(),
                entity_type: "Person".to_string(),
                properties: empty.clone(),
            },
            PrologFact::Entity {
                name: "李四".to_string(),
                entity_type: "Person".to_string(),
                properties: empty.clone(),
            },
            PrologFact::Entity {
                name: "张艺谋".to_string(),
                entity_type: "中国人".to_string(),
                properties: empty.clone(),
            },
            PrologFact::Entity {
                name: "电影A".to_string(),
                entity_type: "科幻片".to_string(),
                properties: empty.clone(),
            },
            PrologFact::Entity {
                name: "电影B".to_string(),
                entity_type: "剧情片".to_string(),
                properties: empty.clone(),
            },
            PrologFact::Entity {
                name: "金棕榈".to_string(),
                entity_type: "奖项".to_string(),
                properties: empty.clone(),
            },
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "导演".to_string(),
                to: "电影A".to_string(),
            },
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "获奖".to_string(),
                to: "金棕榈".to_string(),
            },
            PrologFact::Relation {
                from: "李四".to_string(),
                rel_type: "获奖".to_string(),
                to: "金棕榈".to_string(),
            },
            PrologFact::Relation {
                from: "电影A".to_string(),
                rel_type: "主演".to_string(),
                to: "张艺谋".to_string(),
            },
            PrologFact::Relation {
                from: "电影B".to_string(),
                rel_type: "主演".to_string(),
                to: "李四".to_string(),
            },
            PrologFact::Entity {
                name: "姚明".to_string(),
                entity_type: "Person".to_string(),
                properties: yao,
            },
            PrologFact::Entity {
                name: "刘翔".to_string(),
                entity_type: "Person".to_string(),
                properties: liu,
            },
        ]
    }

    fn graph_pattern_reasoner() -> SymbolicReasoner {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.register_predicate_alias("拿过", "获奖");
        reasoner.observe_facts(&graph_pattern_facts());
        reasoner
    }

    #[test]
    fn graph_question_answers_conjunction_over_kg() {
        let reasoner = graph_pattern_reasoner();
        let answer = reasoner
            .answer_graph_question("谁既导演过科幻片，又拿过金棕榈？")
            .expect("合取问句应由图谱模式回答");
        assert!(answer.contains("张三"), "{}", answer);
        assert!(!answer.contains("李四"), "{}", answer);
        assert!(answer.contains("依据"), "{}", answer);
        let via_answer = reasoner
            .answer("谁既导演过科幻片，又拿过金棕榈？")
            .expect("回答管线应接入图谱模式");
        assert!(via_answer.contains("张三"), "{}", via_answer);
    }

    #[test]
    fn graph_question_answers_existential_over_kg() {
        let reasoner = graph_pattern_reasoner();
        let answer = reasoner
            .answer_graph_question("有没有一部科幻片，主演是中国人？")
            .expect("存在问句应由图谱模式回答");
        assert!(answer.contains("电影A"), "{}", answer);
        assert!(!answer.contains("电影B"), "{}", answer);
        assert!(answer.contains("存在"), "{}", answer);
    }

    #[test]
    fn graph_question_answers_equality_over_attributes() {
        let reasoner = graph_pattern_reasoner();
        let answer = reasoner
            .answer_graph_question("姚明和刘翔出生在同一个城市吗？")
            .expect("等值问句应由图谱模式回答");
        assert!(answer.contains("上海"), "{}", answer);
        assert!(answer.contains("共同的出生地"), "{}", answer);
    }

    #[test]
    fn graph_question_answers_shared_object() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&[
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "认识".to_string(),
                to: "王五".to_string(),
            },
            PrologFact::Relation {
                from: "李四".to_string(),
                rel_type: "认识".to_string(),
                to: "王五".to_string(),
            },
            PrologFact::Relation {
                from: "李四".to_string(),
                rel_type: "认识".to_string(),
                to: "赵六".to_string(),
            },
        ]);
        let answer = reasoner
            .answer_graph_question("张三和李四都认识谁？")
            .expect("共享对象问句应由图谱模式回答");
        assert!(answer.contains("王五"), "{}", answer);
        assert!(!answer.contains("赵六"), "{}", answer);
    }

    #[test]
    fn graph_question_answers_count_questions_without_listing_rows() {
        let reasoner = graph_pattern_reasoner();
        let answer = reasoner
            .answer_graph_question("有多少人既导演过科幻片，又拿过金棕榈？")
            .expect("计数问句应由图谱模式回答");
        assert!(answer.contains("共有 1 个"), "{}", answer);
        assert!(!answer.contains("张三"), "{}", answer);
    }

    #[test]
    fn graph_question_uses_type_closure_and_confirmed_entity_aliases() {
        let mut reasoner = graph_pattern_reasoner();
        // 「电影A」显式断言的是「科幻片」；闭包让查询「电影」也能命中。
        reasoner
            .teach_ontology("科幻片是一种电影")
            .expect("可教学子类公理");
        let typed = reasoner
            .answer_graph_question("有没有一部电影，主演是中国人？")
            .expect("父类型问句应可编译");
        assert!(typed.contains("电影A"), "{}", typed);
        assert!(!typed.contains("电影B"), "{}", typed);

        reasoner.register_entity_alias("小巨人", "姚明");
        let aliased = reasoner
            .answer_graph_question("小巨人和刘翔出生在同一个城市吗？")
            .expect("已确认实体别名应进入图谱词表");
        assert!(aliased.contains("上海"), "{}", aliased);
    }

    #[test]
    fn graph_question_reports_why_a_conjunction_fails() {
        let reasoner = graph_pattern_reasoner();
        let answer = reasoner
            .answer_graph_question("谁既导演过科幻片，又主演过中国人？")
            .expect("图谱问句应仍由模式管线回答");
        assert!(answer.contains("失败诊断"), "{}", answer);
        assert!(answer.contains("连接冲突"), "{}", answer);
        assert!(answer.contains("最小冲突组"), "{}", answer);
    }

    #[test]
    fn graph_question_offers_and_adopts_a_bounded_relaxation() {
        let reasoner = graph_pattern_reasoner();
        let first = reasoner
            .answer("谁既导演过科幻片，又主演过中国人？")
            .expect("零解问句应给出诊断与放宽候选");
        assert!(first.contains("可尝试的放宽"), "{}", first);
        let adopted = reasoner.answer("放宽 1").expect("应能确认第一条放宽");
        assert!(adopted.contains("已采用放宽"), "{}", adopted);
        assert!(!adopted.contains("失败诊断"), "{}", adopted);
        let restored = reasoner.answer("恢复原条件").expect("应能恢复原条件");
        assert!(restored.contains("已恢复原条件"), "{}", restored);
        assert!(restored.contains("失败诊断"), "{}", restored);
    }

    #[test]
    fn graph_question_relaxation_survives_followups_and_can_be_withdrawn() {
        let reasoner = graph_pattern_reasoner();
        reasoner
            .answer("谁既导演过科幻片，又主演过中国人？")
            .expect("零解问句应建立会话");
        let adopted = reasoner.answer("放宽 1").expect("应能确认放宽");
        assert!(adopted.contains("已采用放宽"), "{}", adopted);
        let count = reasoner.answer("一共多少个").expect("放宽后仍可计数");
        assert!(count.contains("共匹配"), "{}", count);
        assert!(count.starts_with("（当前为放宽结果"), "{}", count);
        let withdrawn = reasoner.answer("撤回").expect("应能撤回放宽");
        // 撤回只回退一步；放宽标签必须随模式变化同步清除。
        assert!(!withdrawn.contains("当前为放宽结果"), "{}", withdrawn);
    }

    #[test]
    fn graph_question_can_withdraw_a_conflicting_refinement() {
        let reasoner = graph_pattern_reasoner();
        let first = reasoner
            .answer("谁既导演过科幻片，又拿过金棕榈？")
            .expect("首轮应成功");
        assert!(first.contains("张三"), "{}", first);
        let narrowed = reasoner.answer("只看李四").expect("限定追问应执行");
        assert!(narrowed.contains("约束冲突"), "{}", narrowed);
        assert!(narrowed.contains("可尝试的放宽"), "{}", narrowed);
        let adopted = reasoner.answer("放宽 1").expect("应能撤回冲突约束");
        assert!(adopted.contains("已采用放宽"), "{}", adopted);
        assert!(adopted.contains("张三"), "{}", adopted);
        let restored = reasoner.answer("恢复原条件").expect("应能恢复原条件");
        assert!(restored.contains("张三"), "{}", restored);
        assert!(!restored.contains("失败诊断"), "{}", restored);
    }

    #[test]
    fn equality_answers_name_every_compared_entity() {
        assert_eq!(
            describe_graph_subjects(&["张三".to_string(), "李四".to_string()]),
            "「张三」和「李四」"
        );
        assert_eq!(
            describe_graph_subjects(&["张三".to_string(), "李四".to_string(), "王五".to_string()]),
            "「张三」、「李四」、「王五」"
        );
    }

    #[test]
    fn graph_question_ledger_accumulates_adopted_repairs() {
        let reasoner = graph_pattern_reasoner();
        reasoner
            .answer("谁既导演过科幻片，又主演过中国人？")
            .expect("零解问句应建立会话");
        let first = reasoner.answer("放宽 1").expect("应能采用第一条放宽");
        assert!(first.contains("第 1 步"), "{}", first);
        let narrowed = reasoner.answer("只看李四").expect("放宽后仍可继续限定");
        assert!(narrowed.contains("可尝试的放宽"), "{}", narrowed);
        let second = reasoner.answer("放宽 1").expect("应能采用第二步放宽");
        assert!(second.contains("第 1 步"), "{}", second);
        assert!(second.contains("第 2 步"), "{}", second);
        let restored = reasoner.answer("恢复原条件").expect("应能一次恢复到原问题");
        assert!(restored.contains("已恢复原条件"), "{}", restored);
        assert!(!restored.contains("当前为放宽结果"), "{}", restored);
    }

    #[test]
    fn graph_question_explains_a_zero_result_instead_of_erroring() {
        let reasoner = graph_pattern_reasoner();
        reasoner
            .answer("谁既导演过科幻片，又主演过中国人？")
            .expect("零解问句应建立会话");
        let why = reasoner.answer("为什么").expect("零解时也应能回答为什么");
        assert!(why.contains("无法给出见证子图"), "{}", why);
        assert!(why.contains("失败诊断"), "{}", why);
        assert!(why.contains("可尝试的放宽"), "{}", why);
    }

    #[test]
    fn graph_question_explains_a_partially_understood_conjunction() {
        let reasoner = graph_pattern_reasoner();
        let answer = reasoner
            .answer("谁既导演过科幻片，又喜欢姚明？")
            .expect("读懂一半的图谱问句应说明缺口");
        assert!(answer.contains("只读懂了"), "{}", answer);
        assert!(answer.contains("喜欢姚明"), "{}", answer);
        assert!(answer.contains("无法解析"), "{}", answer);
    }

    #[test]
    fn graph_question_keeps_unrelated_questions_on_existing_pipeline() {
        let reasoner = graph_pattern_reasoner();
        assert!(reasoner
            .graph_lexicon()
            .compile("谁既导演过科幻片，又拿过金棕榈？")
            .is_some());
        assert!(reasoner
            .graph_lexicon()
            .compile("这台机器怎么启动？")
            .is_none());
        assert!(reasoner
            .graph_lexicon()
            .compile("张三的师傅是谁？")
            .is_none());
    }

    #[test]
    fn graph_question_supports_structural_followups() {
        let reasoner = graph_pattern_reasoner();
        let first = reasoner
            .answer("谁既导演过科幻片，又拿过金棕榈？")
            .expect("首轮应建立图谱模式会话");
        assert!(first.contains("张三"), "{}", first);
        let explanation = reasoner.answer("为什么").expect("追问依据应成功");
        assert!(explanation.contains("见证子图"), "{}", explanation);
        assert!(explanation.contains("[E1]"), "{}", explanation);
        let narrowed = reasoner.answer("只看张三").expect("限定追问应成功");
        assert!(narrowed.contains("张三"), "{}", narrowed);
        let relaxed = reasoner.answer("去掉这个条件").expect("放松追问应成功");
        assert!(relaxed.contains("张三"), "{}", relaxed);
        let count = reasoner.answer("一共多少个").expect("计数追问应成功");
        assert!(count.contains("共匹配 1 项"), "{}", count);
        reasoner.answer("只看张三").expect("再次限定应成功");
        let withdrawn = reasoner.answer("撤回").expect("撤回应成功");
        assert!(withdrawn.contains("张三"), "{}", withdrawn);
    }

    #[test]
    fn graph_repair_uses_structural_neighbors_only_as_candidates() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&[
            PrologFact::Entity {
                name: "甲".to_string(),
                entity_type: "Person".to_string(),
                properties: HashMap::new(),
            },
            PrologFact::Entity {
                name: "乙".to_string(),
                entity_type: "Person".to_string(),
                properties: HashMap::new(),
            },
            PrologFact::Relation {
                from: "甲".to_string(),
                rel_type: "获奖".to_string(),
                to: "金奖".to_string(),
            },
            PrologFact::Relation {
                from: "乙".to_string(),
                rel_type: "获奖".to_string(),
                to: "金奖".to_string(),
            },
        ]);
        let alternatives = <SymbolicReasoner as RepairHints>::entity_alternatives(&reasoner, "甲");
        assert!(
            alternatives.iter().any(|(name, _)| name == "乙"),
            "结构相同的实体应成为候选：{:?}",
            alternatives
        );
        assert!(
            alternatives
                .iter()
                .all(|(_, reason)| reason.contains("需确认") && reason.contains("相似度")),
            "候选理由必须标明相似度与不确定性：{:?}",
            alternatives
        );
    }

    #[test]
    fn graph_question_session_is_cleared_on_fork() {
        let reasoner = graph_pattern_reasoner();
        reasoner
            .answer("谁既导演过科幻片，又拿过金棕榈？")
            .expect("首轮应建立图谱模式会话");
        let fork = reasoner.fork_session();
        assert!(fork.graph_session.lock().is_none());
        assert!(fork.answer("为什么").is_none());
    }

    #[test]
    fn graph_question_session_is_invalidated_by_new_facts() {
        let mut reasoner = graph_pattern_reasoner();
        reasoner
            .answer("谁既导演过科幻片，又拿过金棕榈？")
            .expect("首轮应建立图谱模式会话");
        reasoner.observe_facts(&[PrologFact::Relation {
            from: "王五".to_string(),
            rel_type: "获奖".to_string(),
            to: "金棕榈".to_string(),
        }]);
        assert!(reasoner.graph_session.lock().is_none());
        assert!(reasoner.answer("为什么").is_none());
    }

    #[test]
    fn graph_question_session_is_invalidated_by_session_fact_changes() {
        let mut reasoner = graph_pattern_reasoner();
        reasoner
            .answer_graph_question("谁既导演过科幻片，又拿过金棕榈？")
            .expect("首轮应建立图谱模式会话");
        assert!(reasoner.graph_session.lock().is_some());
        reasoner.assert_nl_fact("李四的师傅是张三");
        assert!(
            reasoner.graph_session.lock().is_none(),
            "新增会话事实后旧结果页与见证下标必须失效"
        );

        reasoner
            .answer_graph_question("谁既导演过科幻片，又拿过金棕榈？")
            .expect("新会话应可重新建立图谱模式");
        assert!(reasoner.graph_session.lock().is_some());
        let retracted = reasoner.retract_nl_fact("李四的师傅是张三");
        assert!(
            retracted.contains("已忘记") || retracted.contains("已撤回"),
            "{}",
            retracted
        );
        assert!(
            reasoner.graph_session.lock().is_none(),
            "撤回会话事实后旧结果页与见证下标必须失效"
        );
    }

    #[test]
    fn graph_question_session_is_invalidated_by_assumption_world_changes() {
        let mut reasoner = graph_pattern_reasoner();
        reasoner
            .answer_graph_question("谁既导演过科幻片，又拿过金棕榈？")
            .expect("首轮应建立图谱模式会话");
        assert!(reasoner.graph_session.lock().is_some());
        reasoner
            .enter_assumption_world("李四的出生地是北京")
            .expect("应能进入假设世界");
        assert!(
            reasoner.graph_session.lock().is_none(),
            "假设前提进入图谱输入后旧结果必须失效"
        );

        reasoner
            .answer_graph_question("谁既导演过科幻片，又拿过金棕榈？")
            .expect("假设世界内应可重新建立图谱模式");
        assert!(reasoner.graph_session.lock().is_some());
        reasoner.exit_assumption_world().expect("应能退出假设世界");
        assert!(
            reasoner.graph_session.lock().is_none(),
            "假设前提退出图谱输入后旧结果必须失效"
        );
    }

    #[test]
    fn epistemic_possibility_uses_problog_and_keeps_two_ledgers() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let answer = reasoner
            .answer("李四的师傅可能是谁")
            .expect("情态问句应接入 ProbLog");
        assert!(answer.contains("PQI"), "缺少理解账：{}", answer);
        assert!(answer.contains("理解置信度"), "缺少理解账：{}", answer);
        assert!(answer.contains("知识概率"), "缺少知识账：{}", answer);
        assert!(answer.contains("王五"), "应给出候选：{}", answer);
        assert!(answer.contains("P=1.000"), "图谱事实概率应为 1：{}", answer);
        let certain = reasoner
            .answer_certain("李四的师傅可能是谁")
            .expect("对话恢复通道同样应看见情态");
        assert!(certain.contains("PQI"), "恢复通道丢失情态：{}", certain);
    }

    #[test]
    fn epistemic_certainty_compares_probability_with_threshold() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let confirmed = reasoner
            .answer("王五一定是李四的师傅吗")
            .expect("确定问应可回答");
        assert!(confirmed.contains("可以确认"), "{}", confirmed);
        let refuted = reasoner
            .answer("赵六一定是李四的师傅吗")
            .expect("确定问应可回答");
        assert!(refuted.contains("不足以确认"), "{}", refuted);
        assert!(refuted.contains("P=0.000"), "{}", refuted);
    }

    #[test]
    fn epistemic_probability_frame_extracts_the_ground_atom() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let answer = reasoner
            .answer("王五有多大的可能是李四的师傅")
            .expect("概率值问应可回答");
        assert!(answer.contains("概率值问"), "{}", answer);
        assert!(answer.contains("李四 的 师傅 是 王五"), "{}", answer);
        assert!(answer.contains("P=1.000"), "{}", answer);
    }

    #[test]
    fn epistemic_comparison_uses_the_same_ground_predicate() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let answer = reasoner
            .answer("李四和王五谁更可能是张三的师傅")
            .expect("比较问应可回答");
        assert!(answer.contains("比较结论"), "{}", answer);
        assert!(
            answer.contains("更可能是「张三 的 师傅 是 李四"),
            "{}",
            answer
        );
    }

    #[test]
    fn epistemic_negation_stays_in_scope() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let answer = reasoner
            .answer("张三一定不是李四的师傅吗")
            .expect("否定确定问应可回答");
        assert!(
            answer.contains("因此 P(李四 的 师傅 不是 张三)=1.000"),
            "{}",
            answer
        );
        assert!(answer.contains("可以确认"), "{}", answer);
        let denied = reasoner
            .answer("王五有多大的可能不是李四的师傅")
            .expect("否定概率问句应可回答");
        assert!(
            denied.contains("因此 P(李四 的 师傅 不是 王五)=0.000"),
            "{}",
            denied
        );
        assert!(denied.contains("已排除"), "{}", denied);
    }

    #[test]
    fn epistemic_channel_does_not_hijack_plain_questions_or_unknown_entities() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let plain = reasoner.answer("李四的师傅是谁").expect("普通问句应可回答");
        assert!(!plain.contains("PQI"), "普通问句不应进入情态层：{}", plain);
        assert!(
            reasoner
                .answer_probabilistic_intent("无名氏一定不是王五的师傅吗")
                .is_none(),
            "未知实体应让位给普通管线"
        );
    }

    #[test]
    fn epistemic_probability_parses_a_direct_relation() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&[PrologFact::Relation {
            from: "李四".to_string(),
            rel_type: "认识".to_string(),
            to: "王五".to_string(),
        }]);
        let answer = reasoner
            .answer("李四有多大的可能认识王五")
            .expect("直陈关系应能进入概率通道");
        assert!(answer.contains("认识"), "{}", answer);
        assert!(answer.contains("P=1.000"), "{}", answer);
    }

    #[test]
    fn epistemic_probability_reads_taught_probabilistic_facts() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.teach_report("李四的喜欢是赵六的可能性是百分之八十");
        let answer = reasoner
            .answer("李四的喜欢是赵六的概率是多少")
            .expect("教过的概率事实应可回答");
        assert!(answer.contains("P=0.800"), "{}", answer);
        assert!(answer.contains("概率事实"), "{}", answer);
    }

    #[test]
    fn epistemic_inverted_most_likely_reads_the_object_slot() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let answer = reasoner
            .answer("李四最可能是谁的师傅")
            .expect("「谁的师傅」应查反主宾的候选");
        assert!(answer.contains("最可能：张三"), "{}", answer);
        assert!(!answer.contains("最可能：王五"), "{}", answer);
    }

    #[test]
    fn epistemic_explicit_threshold_compares_in_the_requested_direction() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let met = reasoner
            .answer("李四的师傅是王五的概率超过百分之八十吗")
            .expect("显式阈值问应可回答");
        assert!(met.contains("显式阈值检查"), "{}", met);
        assert!(met.contains("P=1.000 ≥ 0.80"), "{}", met);
        assert!(met.contains("条件成立"), "{}", met);
        let missed = reasoner
            .answer("李四的师傅是张三的概率超过百分之八十吗")
            .expect("显式阈值问应可回答");
        assert!(missed.contains("条件不成立"), "{}", missed);
        let below = reasoner
            .answer("李四的师傅是张三的概率低于80%吗")
            .expect("下界方向也应可回答");
        assert!(below.contains("P=0.000 ≤ 0.80"), "{}", below);
        assert!(below.contains("条件成立"), "{}", below);
    }

    #[test]
    fn epistemic_copula_orientation_matches_plain_check() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let plain = reasoner
            .answer("张三是李四的师傅吗")
            .expect("普通是非问应可回答");
        assert!(!plain.contains("是的"), "{}", plain);
        let modal = reasoner
            .answer("张三一定是李四的师傅吗")
            .expect("情态是非问应可回答");
        assert!(
            modal.contains("李四 的 师傅 是 张三"),
            "「A 是 B 的 P」应读成 P(B, A)，与普通校验一致：{}",
            modal
        );
        assert!(modal.contains("P=0.000"), "{}", modal);
        assert!(modal.contains("不足以确认"), "{}", modal);
    }

    #[test]
    fn colloquial_name_question_answers_with_full_chain() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        let mut facts = master_facts();
        for (name, master) in [("张三", "李四"), ("李四", "王五")] {
            facts.push(PrologFact::Entity {
                name: name.to_string(),
                entity_type: "Person".to_string(),
                properties: HashMap::from([
                    ("name".to_string(), name.to_string()),
                    ("师傅".to_string(), master.to_string()),
                ]),
            });
        }
        reasoner.observe_facts(&facts);
        for question in [
            "张三的师傅叫什么名字",
            "张三的师傅的名字是什么",
            "张三的师傅名字是什么",
        ] {
            let answer = reasoner
                .answer(question)
                .unwrap_or_else(|| panic!("姓名问句应可回答：{}", question));
            assert!(answer.contains("张三 的 师傅"), "{} → {}", question, answer);
            assert!(answer.contains("李四"), "{} → {}", question, answer);
            assert!(
                !answer.contains("师傅叫什么名字 是"),
                "{} 不应把问句残片当成谓词：{}",
                question,
                answer
            );
        }
    }

    #[test]
    fn epistemic_pcc_receipt_carries_a_verified_certificate() {
        let mut reasoner = SymbolicReasoner::new(pcc_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let answer = reasoner
            .answer("王五一定是李四的师傅吗")
            .expect("启用了置信度层也应可回答");
        assert!(answer.contains("置信度决策（PCC）"), "{}", answer);
        assert!(answer.contains("决策=回答"), "{}", answer);
        // PQI 的单位置确定线索后验约 0.60，按理解置信度词表应落到「建议核对」；
        // 回答置信度（融合后 0.913）不能冒充理解置信度。
        assert!(answer.contains("理解措辞=建议核对"), "{}", answer);
        assert!(answer.contains("风险证书"), "{}", answer);
        assert!(answer.contains("上界=0.0377"), "{}", answer);
        assert!(answer.contains("已验证"), "{}", answer);
    }

    #[test]
    fn epistemic_pcc_degrades_a_threshold_boundary_answer_to_hedge() {
        let mut reasoner = SymbolicReasoner::new(pcc_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.teach_report("李四的喜欢是赵六的可能性是百分之八十");
        let answer = reasoner
            .answer("李四的喜欢是赵六的概率超过百分之八十吗")
            .expect("边界样本应可回答");
        assert!(answer.contains("边界=接近判定阈值"), "{}", answer);
        assert!(answer.contains("决策=含糊回答"), "{}", answer);
        assert!(answer.contains("结合来源核对"), "{}", answer);
    }

    #[test]
    fn epistemic_pcc_is_off_by_default_and_keeps_legacy_output() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let answer = reasoner
            .answer("王五一定是李四的师傅吗")
            .expect("旧路径应可回答");
        assert!(!answer.contains("置信度决策（PCC）"), "{}", answer);
        assert!(!answer.contains("风险证书"), "{}", answer);
    }

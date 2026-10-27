    #[test]
    fn test_derived_answer_is_recomputed_after_premise_correction() {
        let mut config = test_config();
        config.enable_link_prediction = false;
        config.enable_association_rules = false;
        config.enable_clarification = false;
        config.enable_soft_logic = false;
        config.enable_problog = false;
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        // 「师傅」按功能槽处理，改口才会撤回旧前提；这也是对话里最常见的情形。
        reasoner.slots.policy_mut().declare_functional("师傅", true);
        reasoner.assert_nl_fact("张三的师傅是李四");
        reasoner.assert_nl_fact("李四的师傅是王五");
        reasoner.aligner.register_term("师祖", TermKind::Relation);
        reasoner.session_clauses.push(Clause::new(
            Atom::pattern("师祖", Term::var("X"), Term::var("Z")),
            vec![
                Atom::pattern("师傅", Term::var("X"), Term::var("Y")),
                Atom::pattern("师傅", Term::var("Y"), Term::var("Z")),
            ],
            0.99,
            "测试：师祖是师傅的两跳",
        ));
        reasoner.rebuild_knowledge_caches();
        reasoner.rebuild_resolver();

        let before = reasoner.answer("张三的师祖是谁").expect("两跳规则应能回答");
        assert!(before.contains("王五"), "{}", before);
        assert!(!before.contains("赵六"), "{}", before);

        reasoner.correct_nl_fact("改口：李四的师傅是赵六");
        let after = reasoner
            .answer("张三的师祖是谁")
            .expect("前提更正后应重新证明");
        assert!(after.contains("赵六"), "{}", after);
        assert!(
            !after.contains("王五"),
            "前提被更正后，旧推导不得残留为答案：{}",
            after
        );
        assert!(!reasoner
            .session_facts
            .iter()
            .any(|atom| atom.predicate.as_ref() == "师祖"));
    }

    #[test]
    fn test_answer_requires_mentioned_predicate() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        assert!(reasoner.answer("李四是谁").is_none());
    }

    #[test]
    fn test_answer_returns_none_without_facts() {
        let reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        assert!(reasoner.answer("李四的师祖是谁").is_none());
    }

    #[test]
    fn pending_question_names_missing_premise_for_two_hop_chain() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&pair_facts());
        let baseline = reasoner
            .answer("张三的师傅的师傅是谁")
            .expect("缺中间前提时仍应给出诚实的无记录回答");
        assert!(baseline.contains("没有记录"), "{}", baseline);

        let ask = reasoner
            .open_pending_question("张三的师傅的师傅是谁")
            .expect("两跳缺一条前提时应立悬问");
        assert!(ask.contains("李四 的 师傅"), "{}", ask);
        assert!(ask.contains("张三 的 师傅 是 李四"), "{}", ask);
        assert!(!ask.contains("王五"), "不得把未知取值说成已知：{}", ask);
        assert!(reasoner.has_pending_question());
        assert!(reasoner
            .pending_question_report()
            .contains("张三的师傅的师傅是谁"));
    }

    #[test]
    fn pending_question_bare_answer_defers_and_closes() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&pair_facts());
        reasoner
            .open_pending_question("张三的师傅的师傅是谁")
            .expect("应先立悬问");

        let deferred = reasoner
            .try_answer_pending_question("王五")
            .expect("裸回答应绑定到洞并给出延期回答");
        assert!(deferred.contains("张三的师傅的师傅是谁"), "{}", deferred);
        assert!(deferred.contains("李四 的 师傅 是 王五"), "{}", deferred);
        assert!(deferred.contains("王五"), "{}", deferred);
        assert!(!reasoner.has_pending_question(), "回填成功后悬问应关闭");
        assert!(
            reasoner.session_facts.iter().any(|atom| {
                atom.predicate.as_ref() == "师傅"
                    && atom.terms[0].name() == "李四"
                    && atom.terms[1].name() == "王五"
            }),
            "补上的前提应成为会话事实"
        );
    }

    #[test]
    fn pending_question_does_not_hijack_questions_or_commands() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&pair_facts());
        reasoner
            .open_pending_question("张三的师傅的师傅是谁")
            .expect("应先立悬问");
        assert!(reasoner.try_answer_pending_question("王五是谁").is_none());
        assert!(reasoner
            .try_answer_pending_question("记住：李四的师傅是王五")
            .is_none());
        assert!(reasoner.has_pending_question());
    }

    #[test]
    fn pending_question_unknown_defers_without_negative_fact() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&pair_facts());
        reasoner
            .open_pending_question("张三的师傅的师傅是谁")
            .expect("应先立悬问");
        let reply = reasoner
            .try_answer_pending_question("不知道")
            .expect("明确不知道应显式搁置");
        assert!(reply.contains("先悬着"), "{}", reply);
        assert!(!reasoner.has_pending_question());
        assert!(
            reasoner.session_negative_facts.is_empty(),
            "「不知道」不得写成否定事实"
        );
    }

    #[test]
    fn pending_question_resolves_after_explicit_assertion() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&pair_facts());
        reasoner
            .open_pending_question("张三的师傅的师傅是谁")
            .expect("应先立悬问");
        reasoner.assert_nl_fact("李四的师傅是王五");
        let deferred = reasoner
            .resolve_pending_after_assertion()
            .expect("新断言补上洞后应给出延期回答");
        assert!(deferred.contains("现在可以回答了"), "{}", deferred);
        assert!(deferred.contains("王五"), "{}", deferred);
        assert!(
            deferred.contains("路径") || deferred.contains("结论"),
            "延期回答必须是原目标的证明，而不是邻近证据：{}",
            deferred
        );
        assert!(!reasoner.has_pending_question());
    }

    #[test]
    fn pending_question_accepts_explicit_answer_prefix() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&pair_facts());
        reasoner
            .open_pending_question("张三的师傅的师傅是谁")
            .expect("应先立悬问");
        let deferred = reasoner
            .try_answer_pending_question("回答悬问：王五")
            .expect("显式前缀应可回填");
        assert!(deferred.contains("王五"), "{}", deferred);
        assert!(!reasoner.has_pending_question());
    }

    #[test]
    fn pending_question_handles_single_missing_hop() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&pair_facts());
        let ask = reasoner
            .open_pending_question("李四的师傅是谁")
            .expect("单跳缺边同样应立悬问");
        assert!(ask.contains("李四 的 师傅"), "{}", ask);
        let deferred = reasoner
            .try_answer_pending_question("王五")
            .expect("短答应回填");
        assert!(deferred.contains("李四 的 师傅 是 王五"), "{}", deferred);
        assert!(deferred.contains("王五"), "{}", deferred);
    }

    #[test]
    fn old_snapshot_without_pending_questions_still_loads() {
        let json = r#"{"focus_entities":[],"focus_types":{},"last_predicate":null,"history":[]}"#;
        let snapshot: DialogueSnapshot = serde_json::from_str(json).expect("旧快照应可解码");
        assert!(snapshot.pending_questions.is_none());
        let encoded = serde_json::to_string(&snapshot).expect("应可编码");
        assert!(
            !encoded.contains("pending_questions"),
            "空悬问不得改变旧快照字节：{}",
            encoded
        );
    }

    #[test]
    fn pending_question_survives_snapshot_roundtrip() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&pair_facts());
        reasoner
            .open_pending_question("张三的师傅的师傅是谁")
            .expect("应先立悬问");
        let snapshot = reasoner.dialogue_snapshot();
        assert!(snapshot.pending_questions.is_some(), "悬问应随快照落盘");

        let mut restored = SymbolicReasoner::new(test_config(), HashMap::new());
        restored.observe_facts(&pair_facts());
        restored
            .restore_dialogue_snapshot(&snapshot)
            .expect("快照应能恢复");
        assert!(restored.has_pending_question(), "恢复后仍应记得未答之问");
        let deferred = restored
            .try_answer_pending_question("王五")
            .expect("恢复后短答仍应能回填");
        assert!(deferred.contains("王五"), "{}", deferred);
    }

    #[test]
    fn pending_question_is_visible_in_summary_and_can_be_dismissed() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&pair_facts());
        reasoner
            .open_pending_question("张三的师傅的师傅是谁")
            .expect("应先立悬问");
        assert!(reasoner.dialogue_summary().contains("悬问 1 条"));
        assert!(reasoner.dismiss_pending_question("张三的师傅的师傅是谁"));
        assert!(!reasoner.has_pending_question());
        assert!(reasoner.dialogue_summary().contains("悬问 0 条"));
    }

    #[test]
    fn pending_question_does_not_open_without_a_real_gap() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        assert!(
            reasoner
                .open_pending_question("张三的师傅的师傅是谁")
                .is_none(),
            "链路已通时不得凭空立悬问"
        );
        assert!(!reasoner.has_pending_question());
    }

    #[test]
    fn pending_question_binds_the_last_asked_hole_not_the_oldest() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&pair_facts());
        reasoner
            .open_pending_question("张三的师傅的师傅是谁")
            .expect("第一条悬问");
        reasoner
            .open_pending_question("王五的师傅的师傅是谁")
            .expect("第二条悬问");

        // 后问的是「赵六 的 师傅」，裸回答必须落在这一条上。
        let deferred = reasoner
            .try_answer_pending_question("孙七")
            .expect("裸回答应绑定到最近追问的洞");
        assert!(deferred.contains("孙七"), "{}", deferred);
        assert!(
            reasoner.session_facts.iter().any(|atom| {
                atom.predicate.as_ref() == "师傅"
                    && atom.terms[0].name() == "赵六"
                    && atom.terms[1].name() == "孙七"
            }),
            "回填应落在第二条悬问的洞上"
        );
        assert!(reasoner.has_pending_question(), "第一条悬问仍应挂着");
        assert!(reasoner
            .pending_question_report()
            .contains("张三的师傅的师傅是谁"));

        // 追问窗口已过：裸回答不再绑定，但显式前缀仍可回填旧悬问。
        assert!(reasoner.try_answer_pending_question("王五").is_none());
        let deferred = reasoner
            .try_answer_pending_question("回答悬问：王五")
            .expect("显式前缀应能回填旧悬问");
        assert!(deferred.contains("王五"), "{}", deferred);
        assert!(!reasoner.has_pending_question());
    }

    #[test]
    fn pending_question_bare_answer_only_opens_for_one_turn() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&pair_facts());
        reasoner
            .open_pending_question("张三的师傅的师傅是谁")
            .expect("应先立悬问");

        assert!(reasoner.try_answer_pending_question("张三是谁").is_none());
        assert!(
            reasoner.try_answer_pending_question("王五").is_none(),
            "隔了一轮之后，裸取值不得再被旧追问吞掉"
        );
        assert!(reasoner.has_pending_question());
        assert!(reasoner
            .try_answer_pending_question("回答悬问：王五")
            .is_some());
    }

    #[test]
    fn pending_question_unknown_is_shelved_and_explicitly_reopenable() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&pair_facts());
        reasoner
            .open_pending_question("张三的师傅的师傅是谁")
            .expect("应先立悬问");
        let reply = reasoner
            .try_answer_pending_question("不知道")
            .expect("明确不知道应显式搁置");
        assert!(reply.contains("先悬着"), "{}", reply);
        assert!(!reasoner.has_pending_question());
        assert!(reasoner.pending_question_report().contains("已搁置 1 条"));

        let reopened = reasoner.reopen_pending_questions();
        assert!(reopened.contains("已重新打开 1 条"), "{}", reopened);
        assert!(reopened.contains("李四 的 师傅"), "{}", reopened);
        assert!(reasoner.has_pending_question());
        let deferred = reasoner
            .try_answer_pending_question("王五")
            .expect("重开后短答应可回填");
        assert!(deferred.contains("王五"), "{}", deferred);
        assert!(!reasoner.has_pending_question());
    }

    #[test]
    fn shelved_pending_question_survives_snapshot_roundtrip() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&pair_facts());
        reasoner
            .open_pending_question("张三的师傅的师傅是谁")
            .expect("应先立悬问");
        reasoner.try_answer_pending_question("不知道");
        let snapshot = reasoner.dialogue_snapshot();
        assert!(
            snapshot.pending_questions.is_some(),
            "搁置的悬问也必须随快照落盘"
        );

        let mut restored = SymbolicReasoner::new(test_config(), HashMap::new());
        restored.observe_facts(&pair_facts());
        restored
            .restore_dialogue_snapshot(&snapshot)
            .expect("快照应能恢复");
        assert!(!restored.has_pending_question());
        assert!(restored.pending_question_report().contains("已搁置 1 条"));
        assert!(restored.reopen_pending_questions().contains("已重新打开"));
        assert!(restored
            .try_answer_pending_question("王五")
            .is_some_and(|reply| reply.contains("王五")));
    }

    #[test]
    fn pending_question_walks_multiple_holes_one_at_a_time() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&pair_facts());
        let ask = reasoner
            .open_pending_question("张三的师傅的师傅的师傅的师傅是谁")
            .expect("四跳缺边应立悬问");
        assert!(ask.contains("李四 的 师傅"), "{}", ask);

        let next = reasoner
            .try_answer_pending_question("王五")
            .expect("补上第一条洞后应继续追问");
        assert!(next.contains("还差前提"), "{}", next);
        assert!(next.contains("赵六 的 师傅"), "{}", next);
        assert!(
            reasoner.has_pending_question(),
            "四跳仍未走通，悬问不能关闭"
        );

        let done = reasoner
            .try_answer_pending_question("孙七")
            .expect("补上第二条洞后应回答原问题");
        assert!(done.contains("孙七"), "{}", done);
        assert!(
            done.contains("张三 的 师傅 的 师傅 的 师傅 的 师傅"),
            "{}",
            done
        );
        assert!(!reasoner.has_pending_question());
    }

    #[test]
    fn pending_question_handles_inverted_predicate_hole() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&pair_facts());
        reasoner.register_predicate_query_surface("徒弟", "师傅");
        let ask = reasoner
            .open_pending_question("王五的徒弟是谁")
            .expect("逆向谓词缺边也应立悬问");
        assert!(ask.contains("王五 的 徒弟"), "{}", ask);

        let deferred = reasoner
            .try_answer_pending_question("赵六的师傅是王五")
            .expect("按存储方向陈述也应回填逆向洞");
        assert!(deferred.contains("赵六"), "{}", deferred);
        assert!(
            reasoner.session_facts.iter().any(|atom| {
                atom.predicate.as_ref() == "师傅"
                    && atom.terms[0].name() == "赵六"
                    && atom.terms[1].name() == "王五"
            }),
            "逆向回填应按存储方向写入会话事实"
        );
        assert!(!reasoner.has_pending_question());
    }

    #[test]
    fn shelved_pending_question_is_not_auto_reopened_by_new_facts() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&pair_facts());
        reasoner
            .open_pending_question("张三的师傅的师傅是谁")
            .expect("应先立悬问");
        reasoner
            .try_answer_pending_question("不知道")
            .expect("明确不知道应搁置");
        reasoner.observe_facts(&master_facts());
        assert!(
            reasoner.resolve_pending_after_assertion().is_none(),
            "搁置的悬问不应被新事实自动重开"
        );
        assert!(!reasoner.has_pending_question());
        assert!(reasoner.pending_question_report().contains("已搁置 1 条"));
    }

    #[test]
    fn pending_question_resolves_after_new_graph_facts_arrive() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&pair_facts());
        reasoner
            .open_pending_question("张三的师傅的师傅是谁")
            .expect("应先立悬问");
        reasoner.observe_facts(&master_facts());
        let deferred = reasoner
            .resolve_pending_after_assertion()
            .expect("图谱补上缺边后应给出延期回答");
        assert!(deferred.contains("现在可以回答了"), "{}", deferred);
        assert!(deferred.contains("王五"), "{}", deferred);
        assert!(!reasoner.has_pending_question());
    }

    #[test]
    fn test_synonym_alias_matches_predicate() {
        let mut synonyms = HashMap::new();
        synonyms.insert("祖师".to_string(), "师祖".to_string());
        let mut reasoner = SymbolicReasoner::new(test_config(), synonyms);
        reasoner.observe_facts(&master_facts());
        let answer = reasoner
            .answer("李四的祖师是谁")
            .expect("同义词别名应能命中师祖关系");
        assert!(answer.contains("赵六"));
    }

    #[test]
    fn test_alignment_and_rules_reports() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let rules_report = reasoner.rules_report();
        assert!(rules_report.contains("师傅"));
        let alignment_report = reasoner.alignment_report();
        assert!(alignment_report.contains("本体对齐报告"));
    }

    fn ambiguous_attribute_facts() -> Vec<PrologFact> {
        vec![
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "苹果公司".to_string(),
                to: "甲公司".to_string(),
            },
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "苹果果实".to_string(),
                to: "红富士".to_string(),
            },
        ]
    }

    fn ambiguous_attribute_reasoner() -> SymbolicReasoner {
        let mut config = test_config();
        config.ontology_alignment_threshold = 0.5;
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        reasoner.observe_facts(&ambiguous_attribute_facts());
        reasoner
    }

    #[test]
    fn test_ambiguous_attribute_alignment_asks_instead_of_guessing() {
        let reasoner = ambiguous_attribute_reasoner();
        let answer = reasoner
            .answer("张三的苹果是谁")
            .expect("两个候选谓词都有答案，应进入歧义决策");
        assert!(answer.contains("不止一种理解"), "实际回答：{}", answer);
        assert!(answer.contains("苹果公司"), "实际回答：{}", answer);
        assert!(answer.contains("苹果果实"), "实际回答：{}", answer);
        assert!(answer.contains("甲公司") && answer.contains("红富士"));
    }

    #[test]
    fn test_equivalent_attribute_candidates_do_not_ask() {
        let mut config = test_config();
        config.ontology_alignment_threshold = 0.5;
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        reasoner.observe_facts(&[
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "苹果公司".to_string(),
                to: "红色".to_string(),
            },
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "苹果果实".to_string(),
                to: "红色".to_string(),
            },
        ]);
        let answer = reasoner
            .answer("张三的苹果是谁")
            .expect("答案等价的候选应静默选一个");
        assert!(!answer.contains("不止一种理解"), "实际回答：{}", answer);
        assert!(answer.contains("红色"), "实际回答：{}", answer);
    }

    #[test]
    fn test_alignment_candidates_report_explains_ambiguity() {
        let mut reasoner = ambiguous_attribute_reasoner();
        reasoner.context.record_entities(&["张三".to_string()]);
        let report = reasoner.alignment_candidates_report("苹果");
        assert!(report.contains("有歧义"), "{}", report);
        assert!(report.contains("苹果公司"), "{}", report);
        assert!(report.contains("苹果果实"), "{}", report);
        assert!(report.contains("答案等价性"), "{}", report);
        assert!(report.contains("值得澄清"), "{}", report);
    }

    #[test]
    fn test_planned_multi_hop_question() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let answer = reasoner
            .answer("张三的师傅的师傅是谁")
            .expect("多跳问题应被逻辑计划执行");
        assert!(answer.contains("王五"));
        assert!(answer.contains("多跳"));
    }

    #[test]
    fn test_planned_multi_hop_answer_keeps_path_identity() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&[
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "师傅".to_string(),
                to: "李四".to_string(),
            },
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "师傅".to_string(),
                to: "王五".to_string(),
            },
            PrologFact::Relation {
                from: "李四".to_string(),
                rel_type: "师傅".to_string(),
                to: "赵六".to_string(),
            },
            PrologFact::Relation {
                from: "王五".to_string(),
                rel_type: "师傅".to_string(),
                to: "赵六".to_string(),
            },
        ]);
        let answer = reasoner
            .answer("张三的师傅的师傅是谁")
            .expect("多跳问句应有路径答案");
        assert!(answer.contains("赵六"), "{}", answer);
        assert!(answer.contains("多跳"), "{}", answer);
        assert!(answer.contains("共 2 条"), "{}", answer);
        assert!(answer.contains("张三 --师傅--> 李四"), "{}", answer);
        assert!(answer.contains("张三 --师傅--> 王五"), "{}", answer);
    }

    #[test]
    fn test_planned_path_between_and_common_neighbor() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&[
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "认识".to_string(),
                to: "李四".to_string(),
            },
            PrologFact::Relation {
                from: "李四".to_string(),
                rel_type: "认识".to_string(),
                to: "王五".to_string(),
            },
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "认识".to_string(),
                to: "赵六".to_string(),
            },
            PrologFact::Relation {
                from: "赵六".to_string(),
                rel_type: "认识".to_string(),
                to: "王五".to_string(),
            },
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "认识".to_string(),
                to: "王五".to_string(),
            },
        ]);
        let between = reasoner
            .answer("张三和王五之间是什么关系")
            .expect("路径问句应返回见证路径");
        assert!(
            between.contains("李四") && between.contains("赵六"),
            "{}",
            between
        );
        assert!(between.contains("路径"), "{}", between);

        let common = reasoner
            .answer("张三和赵六共同认识谁")
            .expect("共同邻居问句应返回中间实体");
        assert!(common.contains("王五"), "{}", common);

        let intermediary = reasoner
            .answer("张三通过谁认识王五")
            .expect("中间人问句应返回中间实体");
        assert!(
            intermediary.contains("李四") || intermediary.contains("赵六"),
            "{}",
            intermediary
        );
    }

    #[test]
    fn test_planned_count_question() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let answer = reasoner
            .answer("李四有几个师傅")
            .expect("计数问题应被逻辑计划执行");
        assert!(answer.contains("共有 1 个"));
        assert!(answer.contains("王五"));
    }

    #[test]
    fn test_planned_yes_no_true_and_false() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let yes = reasoner
            .answer("李四的师傅是王五吗")
            .expect("是非问题应被逻辑计划执行");
        assert!(yes.starts_with("是的"), "实际回答：{}", yes);
        let no = reasoner
            .answer("李四的师傅是张三吗")
            .expect("错误的断言也应给出否定回答");
        assert!(no.starts_with("不是"), "实际回答：{}", no);
        assert!(no.contains("王五"));
    }

    #[test]
    fn test_planned_find_subject() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let answer = reasoner
            .answer("谁的师傅是李四")
            .expect("反查主体应被逻辑计划执行");
        assert!(answer.contains("张三"));
    }

    #[test]
    fn test_planned_answer_reports_missing_record() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let answer = reasoner
            .answer("赵六的师祖是谁")
            .expect("已知实体与已知谓词应给出明确的无记录回答");
        assert!(answer.contains("没有记录"), "实际回答：{}", answer);
    }

    #[test]
    fn test_planned_exist_question() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let answer = reasoner
            .answer("张三有没有师傅")
            .expect("存在性问题应被逻辑计划执行");
        assert!(answer.starts_with("是的"), "实际回答：{}", answer);
    }

    #[test]
    fn test_planned_similar_entity_retrieval() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&pair_facts());
        let answer = reasoner
            .answer("谁和张三最像")
            .expect("近邻检索应命中最相似的实体");
        assert!(answer.contains("王五"));
        assert!(answer.contains("相似度"));
    }

    #[test]
    fn test_planned_analogy_question() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&pair_facts());
        let answer = reasoner
            .answer("张三对应李四，王五对应谁")
            .expect("类比问题应由向量算术回答");
        assert!(answer.contains("赵六"), "实际回答：{}", answer);
    }

    #[test]
    fn test_similarity_report_handles_unknown_entity() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&pair_facts());
        let report = reasoner.similarity_report("不存在的人");
        assert!(report.contains("没有"));
        let report = reasoner.similarity_report("张三");
        assert!(report.contains("王五"));
    }

    #[test]
    fn test_dialogue_state_persistence_roundtrip() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.contextualize("张三的师祖是谁");
        reasoner.contextualize("那李四呢");
        let path =
            std::env::temp_dir().join(format!("knowprolog_dialogue_{}.json", std::process::id()));
        let path_str = path.to_string_lossy().to_string();
        reasoner
            .save_dialogue_state(&path_str)
            .expect("应能保存对话状态");

        let mut restored = SymbolicReasoner::new(test_config(), HashMap::new());
        restored.observe_facts(&master_facts());
        assert!(restored.load_dialogue_state(&path_str), "应能恢复对话状态");
        assert_eq!(restored.context().focus(), Some("李四"));
        assert_eq!(restored.context().last_predicate(), Some("师祖"));
        let rewritten = restored
            .contextualize("他的师祖是谁")
            .expect("恢复后的记忆应支持指代消解");
        assert!(rewritten.contains("李四"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn dialogue_state_persistence_is_canonical_bytes_with_receipt() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.contextualize("张三的师祖是谁");
        reasoner.contextualize("那李四呢");
        let path = std::env::temp_dir().join(format!(
            "knowprolog_canonical_state_{}.json",
            std::process::id()
        ));
        let path_str = path.to_string_lossy().to_string();
        reasoner.save_dialogue_state(&path_str).expect("应能保存");

        let bytes = std::fs::read(&path).unwrap();
        let (snapshot, receipt) = SymbolicReasoner::read_dialogue_state(&path_str).unwrap();
        assert!(receipt.canonical, "落盘必须是规范字节");
        assert_eq!(receipt.kind, "dialogue_state");
        // L1/L2/L3/L4：往返恒等 / 字节不动点 / 编码幂等 / 摘要交换
        assert_eq!(canonical::canonical_bytes(&snapshot).unwrap(), bytes);
        assert_eq!(
            canonical::canonical_digest(&snapshot).unwrap(),
            canonical::canonical_digest(&reasoner.dialogue_snapshot()).unwrap()
        );
        let (again, _) = SymbolicReasoner::read_dialogue_state(&path_str).unwrap();
        assert_eq!(
            canonical::canonical_bytes(&again).unwrap(),
            canonical::canonical_bytes(&snapshot).unwrap()
        );

        // 坏字节给 parse 码；缺失文件给 io 码（原因可回答）
        assert_eq!(
            SymbolicReasoner::read_dialogue_state("/nonexistent/state.json")
                .unwrap_err()
                .code(),
            "io"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_dialogue_summary_mentions_focus() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.contextualize("张三的师祖是谁");
        let summary = reasoner.dialogue_summary();
        assert!(summary.contains("张三"));
        assert!(summary.contains("师祖"));
    }

    #[test]
    fn test_embedding_feeds_from_facts() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&pair_facts());
        assert!(reasoner.embedding().is_embedded("张三"));
        assert!(reasoner.embedding().is_embedded("赵六"));
        assert!(reasoner.embedding().similarity("张三", "王五") > 0.9);
    }

    #[test]
    fn test_configure_ontology_feeds_schema_aware_embedding() {
        use crate::ontology_reasoning::{OntologyLesson, OntologyProgram};

        let program = OntologyProgram::from_lessons(vec![
            OntologyLesson::Subclass {
                sub: "学生".into(),
                sup: "人物".into(),
                original: "学生是一种人物".into(),
            },
            OntologyLesson::Subproperty {
                sub: "导师".into(),
                sup: "师傅".into(),
                original: "导师是师傅的子属性".into(),
            },
            OntologyLesson::Domain {
                property: "导师".into(),
                class: "学生".into(),
                original: "导师关系的定义域是学生".into(),
            },
            OntologyLesson::Range {
                property: "导师".into(),
                class: "人物".into(),
                original: "导师关系的值域是人物".into(),
            },
        ]);
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.configure_ontology_program(&program);
        reasoner.observe_facts(&[
            PrologFact::Entity {
                name: "王五".into(),
                entity_type: "人物".into(),
                properties: HashMap::new(),
            },
            PrologFact::Relation {
                from: "张三".into(),
                rel_type: "导师".into(),
                to: "李四".into(),
            },
        ]);
        assert!(reasoner.schema_view().is_subclass("学生", "人物"));
        let shared = reasoner
            .embedding()
            .shared_feature_labels("张三", "王五", 8);
        assert!(
            shared.iter().any(|feature| feature == "类型·人物"),
            "定义域投影应让关系主语与显式类型实体共享类型特征：{:?}",
            shared
        );
    }

    #[test]
    fn test_configure_schema_view_merges_aliases_and_survives_rebuild() {
        use crate::predicate_registry::PredicateProfileConfig;
        use crate::schema_view::SchemaView;
        use crate::type_system::{TypeAliasConfig, TypeSubclassConfig, TypeSystemConfig};

        let view = SchemaView::from_config(
            &TypeSystemConfig {
                aliases: vec![TypeAliasConfig {
                    alias: "Person".into(),
                    canonical: "人物".into(),
                }],
                subclasses: vec![TypeSubclassConfig {
                    sub: "学生".into(),
                    sup: "人物".into(),
                }],
                ..TypeSystemConfig::default()
            },
            &[PredicateProfileConfig {
                name: "师傅".into(),
                domain: vec!["人物".into()],
                range: vec!["人物".into()],
                ..PredicateProfileConfig::default()
            }],
        );
        let facts = vec![
            PrologFact::Entity {
                name: "张三".into(),
                entity_type: "Person".into(),
                properties: HashMap::new(),
            },
            PrologFact::Entity {
                name: "李四".into(),
                entity_type: "学生".into(),
                properties: HashMap::new(),
            },
        ];
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.configure_schema_view(&view);
        reasoner.observe_facts(&facts);
        let shared = reasoner
            .embedding()
            .shared_feature_labels("张三", "李四", 8);
        assert!(
            shared.iter().any(|feature| feature == "类型·人物"),
            "Person 别名与学生的子类闭包应汇合到人物：{:?}",
            shared
        );
        let digest = reasoner.schema_view().digest();

        reasoner.rebuild_from_graph(&facts);
        assert_eq!(reasoner.schema_view().digest(), digest);
        let shared = reasoner
            .embedding()
            .shared_feature_labels("张三", "李四", 8);
        assert!(shared.iter().any(|feature| feature == "类型·人物"));
        assert_eq!(reasoner.fork_session().schema_view().digest(), digest);
    }

    #[test]
    fn test_entity_link_notes_for_typo_mention() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let notes = reasoner.entity_link_notes("张山的师傅是谁");
        assert!(
            notes
                .iter()
                .any(|note| note.contains("张三") && note.contains("→")),
            "错别字指称应产生模糊链接说明：{:?}",
            notes
        );
        assert!(reasoner.linker().is_known_entity("张三"));
    }

    #[test]
    fn test_linked_entity_fallback_recovers_typo_query() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let answer = reasoner
            .answer("李寺的师祖是谁")
            .expect("错别字实体应能通过链接回退命中推理");
        assert!(answer.contains("赵六"), "实际回答：{}", answer);
    }

    #[test]
    fn test_query_anchor_recovers_single_hop_typo() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let receipt = reasoner
            .query_anchor_receipt("张山的师傅是谁")
            .expect("单跳错别字问句应产生锚定回执");
        assert!(receipt.contains("张山"), "{}", receipt);
        assert!(receipt.contains("张三"), "{}", receipt);
        let answer = reasoner
            .answer("张山的师傅是谁")
            .expect("单跳错别字问句应被锚定重写救回");
        assert!(answer.contains("李四"), "实际回答：{}", answer);
    }

    #[test]
    fn test_query_anchor_recovers_after_pending_failure() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        let mut facts = master_facts();
        facts.push(PrologFact::Relation {
            from: "张三".to_string(),
            rel_type: "认识".to_string(),
            to: "李四".to_string(),
        });
        reasoner.observe_facts(&facts);
        let exact = reasoner.answer("张山认识谁");
        assert!(
            exact
                .as_deref()
                .is_some_and(|answer| answer.contains("李四")),
            "精确路径先报无记录时仍应尝试锚定：{:?}",
            exact
        );
    }

    #[test]
    fn test_query_anchor_keeps_exact_answer_unchanged() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let exact = reasoner.answer("张三的师傅是谁").expect("精确问句应可回答");
        assert!(exact.contains("李四"), "实际回答：{}", exact);
        assert!(reasoner.query_anchor_receipt("张三的师傅是谁").is_none());
    }

    #[test]
    fn test_generate_about_combines_facts() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let profile = reasoner.generate_about("张三").expect("应能生成介绍段落");
        assert!(profile.contains("关于「张三」"));
        assert!(profile.contains("师傅"));
        assert!(profile.contains("师祖"));
        assert!(profile.contains("李四"));
        assert!(reasoner.generate_about("不存在的人").is_none());
    }

    #[test]
    fn test_narrate_uses_owl_definition_and_dcg_variants() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        let mut properties = HashMap::new();
        properties.insert("职业".to_string(), "教师".to_string());
        let facts = vec![
            PrologFact::Entity {
                name: "张三".to_string(),
                entity_type: "人".to_string(),
                properties,
            },
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "任教于".to_string(),
                to: "北京大学".to_string(),
            },
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "任教于".to_string(),
                to: "清华大学".to_string(),
            },
        ];
        reasoner.observe_facts(&facts);
        let text = reasoner.narrate("张三").expect("应能生成符号主义叙述");
        assert!(text.contains("是一种人"), "应包含 OWL 定义开场：{}", text);
        assert!(
            text.contains("北京大学") && text.contains("清华大学") && text.contains("、"),
            "多值事实应聚合：{}",
            text
        );
        assert!(reasoner.narrate("不存在的人").is_none());

        let variants = reasoner
            .paraphrase_fact("张三", "职业", 4)
            .expect("应能改写事实句式");
        assert!(
            variants.lines().count() >= 2,
            "应有多种合规句式：{}",
            variants
        );
        assert!(variants.contains("教师"));
    }

    #[test]
    fn test_teach_ontology_axiom_flows_into_narration() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&[PrologFact::Entity {
            name: "苏格拉底".to_string(),
            entity_type: "人".to_string(),
            properties: HashMap::new(),
        }]);
        let summary = reasoner
            .teach_ontology("人是一种动物")
            .expect("本体公理应可学习");
        assert!(
            summary.contains("动物"),
            "教学回执应包含公理内容：{}",
            summary
        );
        let text = reasoner.narrate("苏格拉底").expect("叙述应可用");
        assert!(
            text.contains("动物"),
            "本体继承推理应进入叙述（定义/推导链）：{}",
            text
        );
    }

    /// 教学本体 → 快照胶囊：未教学时快照不携带胶囊（旧字节兼容），
    /// 教学后胶囊随快照走，且能通过自身的版本门与摘要校验。
    #[test]
    fn test_taught_ontology_lands_in_dialogue_snapshot_capsule() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&[PrologFact::Entity {
            name: "苏格拉底".to_string(),
            entity_type: "人".to_string(),
            properties: HashMap::new(),
        }]);
        let plain = reasoner.dialogue_snapshot();
        assert!(
            plain.ontology.is_none(),
            "未教学时不应携带本体胶囊（旧快照字节保持不变）"
        );
        let plain_text = String::from_utf8(canonical::canonical_bytes(&plain).unwrap()).unwrap();
        assert!(
            !plain_text.contains("ontology"),
            "无胶囊时规范字节不得出现 ontology 键：{}",
            plain_text
        );

        reasoner
            .teach_ontology("人是一种动物")
            .expect("本体公理应可学习");
        let snapshot = reasoner.dialogue_snapshot();
        let capsule = snapshot.ontology.expect("教学后快照应携带本体胶囊");
        capsule.verify().expect("胶囊应通过自身校验");
        assert_eq!(capsule.version, 1);
        assert!(!capsule.closure_digest.is_empty(), "闭包缓存应带摘要");
        assert!(
            capsule.closure_matches(reasoner.discourse.narrator_mut().closure()),
            "导出缓存应与当前闭包一致"
        );
    }

    /// 快照恢复到全新推理器：教过的本体 100% 保留（理解 = 反序列化）。
    #[test]
    fn test_taught_ontology_survives_snapshot_restore_on_fresh_reasoner() {
        let facts = [PrologFact::Entity {
            name: "苏格拉底".to_string(),
            entity_type: "人".to_string(),
            properties: HashMap::new(),
        }];
        let mut source = SymbolicReasoner::new(test_config(), HashMap::new());
        source.observe_facts(&facts);
        source.teach_ontology("人是一种动物").expect("可教学");
        let snapshot = source.dialogue_snapshot();

        let mut target = SymbolicReasoner::new(test_config(), HashMap::new());
        target.observe_facts(&facts);
        target
            .restore_dialogue_snapshot(&snapshot)
            .expect("胶囊应通过校验并恢复");
        // 幂等：同一快照恢复两次，快照内容不变
        let after_first =
            canonical::canonical_digest(&target.dialogue_snapshot()).expect("快照可摘要");
        target
            .restore_dialogue_snapshot(&snapshot)
            .expect("重复恢复应幂等成功");
        let after_second =
            canonical::canonical_digest(&target.dialogue_snapshot()).expect("快照可摘要");
        assert_eq!(after_first, after_second, "重复恢复必须幂等");
        let text = target.narrate("苏格拉底").expect("叙述应可用");
        assert!(
            text.contains("动物"),
            "恢复后本体继承推理必须仍在：{}",
            text
        );
    }

    /// 重建基座（rebuild_from_graph）后教过的本体不得消失。
    #[test]
    fn test_taught_ontology_survives_rebuild_from_graph() {
        let facts = vec![
            PrologFact::Entity {
                name: "苏格拉底".to_string(),
                entity_type: "人".to_string(),
                properties: HashMap::new(),
            },
            PrologFact::Relation {
                from: "苏格拉底".to_string(),
                rel_type: "徒弟".to_string(),
                to: "柏拉图".to_string(),
            },
        ];
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&facts);
        reasoner.teach_ontology("人是一种动物").expect("可教学");
        reasoner.rebuild_from_graph(&facts);
        let text = reasoner.narrate("苏格拉底").expect("重建后叙述应可用");
        assert!(
            text.contains("动物"),
            "重建图谱后教过的本体必须保留：{}",
            text
        );
    }

    /// 落盘 → 新进程读回：教过的本体随规范字节持久化。
    #[test]
    fn test_taught_ontology_survives_dialogue_state_file_roundtrip() {
        let dir = tempfile::TempDir::new().expect("临时目录");
        let state_path = dir.path().join("dialogue_state.json");
        let state_path_str = state_path.to_str().expect("路径可表示");
        let facts = [PrologFact::Entity {
            name: "苏格拉底".to_string(),
            entity_type: "人".to_string(),
            properties: HashMap::new(),
        }];
        let mut source = SymbolicReasoner::new(test_config(), HashMap::new());
        source.observe_facts(&facts);
        source.teach_ontology("人是一种动物").expect("可教学");
        source
            .save_dialogue_state(state_path_str)
            .expect("应能保存");

        let mut target = SymbolicReasoner::new(test_config(), HashMap::new());
        target.observe_facts(&facts);
        assert!(
            target.load_dialogue_state(state_path_str),
            "含胶囊的对话状态应可加载"
        );
        let text = target.narrate("苏格拉底").expect("叙述应可用");
        assert!(text.contains("动物"), "落盘恢复后本体应保留：{}", text);
    }

    /// 被改写 / 超前版本的胶囊：整体拒绝，零半应用。
    #[test]
    fn test_tampered_ontology_capsule_is_rejected_without_partial_apply() {
        let facts = [PrologFact::Entity {
            name: "苏格拉底".to_string(),
            entity_type: "人".to_string(),
            properties: HashMap::new(),
        }];
        let mut source = SymbolicReasoner::new(test_config(), HashMap::new());
        source.observe_facts(&facts);
        source.teach_ontology("人是一种动物").expect("可教学");
        source.context.record_question("苏格拉底是谁");
        let mut snapshot = source.dialogue_snapshot();

        let mut tampered = snapshot.clone();
        {
            let capsule = tampered.ontology.as_mut().expect("应有胶囊");
            capsule.lessons.clear();
        }
        let mut target = SymbolicReasoner::new(test_config(), HashMap::new());
        target.observe_facts(&facts);
        let error = target
            .restore_dialogue_snapshot(&tampered)
            .expect_err("改写公理必须被拒绝");
        assert!(error.contains("摘要"), "错误应说明摘要不匹配：{}", error);
        assert!(
            target.dialogue_snapshot().ontology.is_none(),
            "拒绝后不得留下半应用的胶囊"
        );
        assert!(
            target.context.history().is_empty(),
            "拒绝后其余会话态也不得被应用（零半应用）"
        );

        let mut future = snapshot.clone();
        future.ontology.as_mut().expect("应有胶囊").version = 99;
        let error = target
            .restore_dialogue_snapshot(&future)
            .expect_err("超前版本必须被版本门拒绝");
        assert!(error.contains("版本"), "错误应说明版本门：{}", error);

        snapshot.ontology = None;
        target
            .restore_dialogue_snapshot(&snapshot)
            .expect("无胶囊快照不受影响");
    }

    /// 类型化配方：逆方向查询、子类扩展、开放世界类型过滤与依据复述。
    #[test]
    fn recipe_lookup_uses_inverse_subclass_and_open_world_types() {
        use crate::ontology_reasoning::{OntologyLesson, OntologyProgram};

        let facts = vec![
            PrologFact::Entity {
                name: "阿凡达".into(),
                entity_type: "电影".into(),
                properties: HashMap::new(),
            },
            PrologFact::Entity {
                name: "千与千寻".into(),
                entity_type: "动画电影".into(),
                properties: HashMap::new(),
            },
            PrologFact::Entity {
                name: "卡梅隆".into(),
                entity_type: "人物".into(),
                properties: HashMap::new(),
            },
            PrologFact::Entity {
                name: "宫崎骏".into(),
                entity_type: "人物".into(),
                properties: HashMap::new(),
            },
            PrologFact::Relation {
                from: "阿凡达".into(),
                rel_type: "导演".into(),
                to: "卡梅隆".into(),
            },
            PrologFact::Relation {
                from: "千与千寻".into(),
                rel_type: "导演".into(),
                to: "宫崎骏".into(),
            },
        ];
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&facts);
        let program = OntologyProgram::from_lessons(vec![
            OntologyLesson::Subclass {
                sub: "动画电影".into(),
                sup: "电影".into(),
                original: "动画电影是一种电影".into(),
            },
            OntologyLesson::Subclass {
                sub: "电影".into(),
                sup: "影视作品".into(),
                original: "电影是一种影视作品".into(),
            },
            OntologyLesson::Disjoint {
                left: "电影".into(),
                right: "人物".into(),
                original: "电影和人物互斥".into(),
            },
            OntologyLesson::Domain {
                property: "导演".into(),
                class: "电影".into(),
                original: "导演关系的定义域是电影".into(),
            },
            OntologyLesson::Range {
                property: "导演".into(),
                class: "人物".into(),
                original: "导演关系的值域是人物".into(),
            },
            OntologyLesson::Inverse {
                left: "导演".into(),
                right: "执导".into(),
                original: "导演和执导互为逆关系".into(),
            },
        ]);
        assert!(reasoner.configure_ontology_program(&program) > 0);

        let forward = reasoner
            .recipe_lookup("阿凡达", "导演", RecipeDirection::Forward, &["人物".into()])
            .expect("正向应命中导演");
        assert_eq!(forward.values, vec!["卡梅隆".to_string()]);
        assert!(
            forward.evidence.iter().any(|line| line.contains("导演")),
            "依据必须复述事实边：{:?}",
            forward.evidence
        );

        let inverse = reasoner
            .recipe_lookup("宫崎骏", "导演", RecipeDirection::Inverse, &["电影".into()])
            .expect("逆向应命中作品");
        assert_eq!(
            inverse.values,
            vec!["千与千寻".to_string()],
            "动画电影 ⊑ 电影，答案类型必须沿子类扩展"
        );

        let unknown = reasoner
            .recipe_lookup(
                "卡梅隆",
                "导演",
                RecipeDirection::Inverse,
                &["未知类型".into()],
            )
            .expect("开放世界：未知类型必须放行");
        assert_eq!(unknown.values, vec!["阿凡达".to_string()]);

        assert!(
            reasoner
                .recipe_lookup("卡梅隆", "导演", RecipeDirection::Inverse, &["人物".into()])
                .is_none(),
            "已知互斥类型（电影 ⊥ 人物）必须拒答"
        );
    }

    /// 主语侧类型约束：正向问句查定义域、逆向问句查值域；由关系反推的
    /// 循环类型不作数，未知类型放行，已知互斥才拒答。
    #[test]
    fn recipe_lookup_gates_subject_types_by_domain_and_range() {
        use crate::ontology_reasoning::{OntologyLesson, OntologyProgram};

        let facts = vec![
            PrologFact::Entity {
                name: "阿凡达".into(),
                entity_type: "电影".into(),
                properties: HashMap::new(),
            },
            PrologFact::Entity {
                name: "卡梅隆".into(),
                entity_type: "人物".into(),
                properties: HashMap::new(),
            },
            PrologFact::Relation {
                from: "阿凡达".into(),
                rel_type: "导演".into(),
                to: "卡梅隆".into(),
            },
            // 反向错录：把人物记成了导演关系的主语、把电影记成了宾语——
            // 同名关系用在了类型不符的实体上，主语侧门控必须挡住两侧误答。
            PrologFact::Relation {
                from: "卡梅隆".into(),
                rel_type: "导演".into(),
                to: "阿凡达".into(),
            },
            // 类型完全未知的关系主语：开放世界必须放行而不是猜测拒答。
            PrologFact::Relation {
                from: "某无名氏".into(),
                rel_type: "导演".into(),
                to: "卡梅隆".into(),
            },
        ];
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&facts);
        let program = OntologyProgram::from_lessons(vec![
            OntologyLesson::Domain {
                property: "导演".into(),
                class: "电影".into(),
                original: "导演关系的定义域是电影".into(),
            },
            OntologyLesson::Range {
                property: "导演".into(),
                class: "人物".into(),
                original: "导演关系的值域是人物".into(),
            },
            OntologyLesson::Disjoint {
                left: "电影".into(),
                right: "人物".into(),
                original: "电影和人物互斥".into(),
            },
        ]);
        assert!(reasoner.configure_ontology_program(&program) > 0);

        let forward = reasoner
            .recipe_lookup("阿凡达", "导演", RecipeDirection::Forward, &["人物".into()])
            .expect("声明类型与定义域相符的正向查询应命中");
        assert_eq!(forward.values, vec!["卡梅隆".to_string()]);

        let inverse = reasoner
            .recipe_lookup("卡梅隆", "导演", RecipeDirection::Inverse, &["电影".into()])
            .expect("声明类型与值域相符的逆向查询应命中");
        assert!(inverse.values.contains(&"阿凡达".to_string()));

        assert!(
            reasoner
                .recipe_lookup("卡梅隆", "导演", RecipeDirection::Forward, &["人物".into()])
                .is_none(),
            "人物作为导演关系的主语与定义域电影互斥，正向误答必须拒答"
        );
        assert!(
            reasoner
                .recipe_lookup("阿凡达", "导演", RecipeDirection::Inverse, &["电影".into()])
                .is_none(),
            "电影作为导演关系的宾语与值域人物互斥，逆向误答必须拒答"
        );

        let unknown = reasoner
            .recipe_lookup(
                "某无名氏",
                "导演",
                RecipeDirection::Forward,
                &["人物".into()],
            )
            .expect("有效类型未知（关系反推的循环类型不作数）时必须放行");
        assert_eq!(unknown.values, vec!["卡梅隆".to_string()]);
    }

    /// 配方执行单一入口：CLI 与 HTTP 共用；角色指称接续焦点，依据随模板接地。
    #[test]
    fn answer_plugin_recipe_resolves_role_reference_and_grounds_evidence() {
        use crate::plugin::{PluginManifest, PluginRegistry, PluginRuntime, PluginSettings};

        let manifest: PluginManifest = toml::from_str(
            r#"
id = "test.film"
version = "0.1.0"
api = 1
default_mode = "on"

[[contributions]]
kind = "reference"
id = "film.reference"
quantifiers = ["这位", "那位"]
role_words = ["导演"]

[[contributions]]
kind = "recipe"
id = "film.works_by_director"
patterns = ["{entity}还拍过什么", "那{entity}还拍过什么"]
question = "{entity}导演过哪些作品"
attribute = "导演"
direction = "inverse"
answer_types = ["电影"]
template = "film.works.template"

[[contributions]]
kind = "template"
id = "film.works.template"
template = "「{entity}」导演过的作品：{answer}。依据：{evidence}"

[[contributions]]
kind = "recipe"
id = "film.directed_by"
patterns = ["谁导演了{entity}"]
question = "{entity}的导演是什么"
attribute = "导演"
direction = "forward"
answer_types = ["人物"]
template = "film.directed.template"

[[contributions]]
kind = "template"
id = "film.directed.template"
template = "「{entity}」的导演是{answer}。依据：{evidence}"
"#,
        )
        .expect("测试清单应可解析");
        let registry = PluginRegistry::from_manifests(vec![manifest]);
        let mut runtime = PluginRuntime::with_settings(registry, &PluginSettings::default());

        let facts = vec![
            PrologFact::Entity {
                name: "阿凡达".into(),
                entity_type: "电影".into(),
                properties: HashMap::new(),
            },
            PrologFact::Entity {
                name: "卡梅隆".into(),
                entity_type: "人物".into(),
                properties: HashMap::new(),
            },
            PrologFact::Relation {
                from: "阿凡达".into(),
                rel_type: "导演".into(),
                to: "卡梅隆".into(),
            },
        ];
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&facts);

        let outcome = runtime.rewrite("这位导演还拍过什么");
        assert!(outcome.trace.recipe.is_some(), "配方应命中角色指称问法");
        let reply = reasoner
            .answer_plugin_recipe(&outcome, &mut runtime, Some("卡梅隆"))
            .expect("焦点实体应能完成逆向配方");
        assert_eq!(reply.entity.as_deref(), Some("卡梅隆"));
        assert_eq!(reply.question, "卡梅隆导演过哪些作品");
        assert_eq!(
            reply.focus.as_deref(),
            Some("卡梅隆"),
            "逆向问句的焦点应留在被查询的主语"
        );
        assert!(reply.text.contains("阿凡达"), "{}", reply.text);
        assert!(reply.text.contains("依据"), "{}", reply.text);
        assert!(
            reply.text.contains("属于 电影"),
            "依据应复述答案的类型约束：{}",
            reply.text
        );

        let forward = runtime.rewrite("谁导演了阿凡达");
        let forward_reply = reasoner
            .answer_plugin_recipe(&forward, &mut runtime, None)
            .expect("正向配方应命中");
        assert_eq!(
            forward_reply.focus.as_deref(),
            Some("卡梅隆"),
            "正向问句的焦点应转到答案，供「这位导演」继续追问"
        );
        assert!(
            reply
                .plan
                .iter()
                .any(|plan| plan.contains("film.works_by_director")),
            "{:?}",
            reply.plan
        );
    }

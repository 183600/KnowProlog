    #[test]
    fn constructive_command_confirms_into_session_and_refutes_as_blocker() {
        let facts = vec![
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "同事".to_string(),
                to: "王五".to_string(),
            },
            PrologFact::Relation {
                from: "王五".to_string(),
                rel_type: "师傅".to_string(),
                to: "赵六".to_string(),
            },
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "朋友".to_string(),
                to: "赵六".to_string(),
            },
            PrologFact::Relation {
                from: "孙七".to_string(),
                rel_type: "同事".to_string(),
                to: "周八".to_string(),
            },
            PrologFact::Relation {
                from: "周八".to_string(),
                rel_type: "师傅".to_string(),
                to: "吴九".to_string(),
            },
        ];
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&facts);
        reasoner.config.enable_link_prediction = false;
        reasoner.config.constructive.enabled = true;

        let probe = reasoner
            .handle_constructive_command("建设性推断：孙七|朋友")
            .expect("推断命令应被识别");
        assert!(probe.contains("孙七 --朋友--> 吴九"), "实际：{}", probe);
        assert!(probe.contains("回溯到观察事实"), "实际：{}", probe);

        let confirmed = reasoner
            .handle_constructive_command("确认推断：孙七|朋友|吴九")
            .expect("确认命令应被识别");
        assert!(confirmed.contains("会话事实"), "实际：{}", confirmed);
        assert!(reasoner.positive_session_atoms().any(|atom| {
            atom.predicate.as_ref() == "朋友"
                && atom.terms.first().map(|term| term.name()) == Some("孙七")
                && atom.terms.get(1).map(|term| term.name()) == Some("吴九")
        }));
        let after = reasoner
            .answer("孙七的朋友是谁")
            .expect("确认后的会话事实应可直接回答");
        assert!(after.contains("吴九"), "实际：{}", after);
        assert!(
            !after.contains("建设性推断"),
            "确认后不应再待确认：{}",
            after
        );

        let mut refuting = SymbolicReasoner::new(test_config(), HashMap::new());
        refuting.observe_facts(&facts);
        refuting.config.enable_link_prediction = false;
        refuting.config.constructive.enabled = true;
        let denied = refuting
            .handle_constructive_command("否定推断：孙七|同事|周八")
            .expect("否定命令应被识别");
        assert!(denied.contains("否定事实"), "实际：{}", denied);
        let blocked = refuting
            .answer("孙七的朋友的师傅是谁")
            .expect("否定后仍应给出无答案说明");
        assert!(
            !blocked.contains("吴九"),
            "被否认的前提不得复用：{}",
            blocked
        );
        assert!(
            !blocked.contains("建设性推断"),
            "被否认后不应再合成：{}",
            blocked
        );

        let mut target_denied = SymbolicReasoner::new(test_config(), HashMap::new());
        target_denied.observe_facts(&facts);
        target_denied.config.enable_link_prediction = false;
        target_denied.config.constructive.enabled = true;
        target_denied
            .handle_constructive_command("否定推断：孙七|朋友|吴九")
            .expect("否定命令应被识别");
        let target_blocked = target_denied
            .answer("孙七的朋友的师傅是谁")
            .expect("否认目标后仍应给出无答案说明");
        assert!(
            !target_blocked.contains("建设性推断"),
            "被否认的目标不得从另一条规则重新合成：{}",
            target_blocked
        );
        assert!(
            !target_blocked.contains("综合得出"),
            "被否认的目标不得成为下一跳前提：{}",
            target_blocked
        );
    }

    #[test]
    fn constructive_check_is_reported_as_pending_not_graph_fact() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        let facts = vec![
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "同事".to_string(),
                to: "王五".to_string(),
            },
            PrologFact::Relation {
                from: "王五".to_string(),
                rel_type: "师傅".to_string(),
                to: "赵六".to_string(),
            },
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "朋友".to_string(),
                to: "赵六".to_string(),
            },
            PrologFact::Relation {
                from: "孙七".to_string(),
                rel_type: "同事".to_string(),
                to: "周八".to_string(),
            },
            PrologFact::Relation {
                from: "周八".to_string(),
                rel_type: "师傅".to_string(),
                to: "吴九".to_string(),
            },
        ];
        reasoner.observe_facts(&facts);
        reasoner.config.enable_link_prediction = false;
        reasoner.config.constructive.enabled = true;
        let answer = reasoner
            .answer("孙七的朋友的师傅是周八吗")
            .expect("缺中间边的是非问应能给出待确认推断");
        assert!(answer.contains("建设性推断"), "实际：{}", answer);
        assert!(answer.contains("待确认"), "实际：{}", answer);
        assert!(
            !answer.contains("是的。根据知识图谱"),
            "未确认的推断不得冒充图谱事实：{}",
            answer
        );
        assert!(
            answer.contains("确认推断：孙七|朋友|吴九"),
            "应给出可执行的确认入口：{}",
            answer
        );

        let other = reasoner
            .answer("孙七的朋友的师傅是郑十吗")
            .expect("不同取值应给出无法确定而不是伪造否定");
        assert!(other.contains("无法确定"), "实际：{}", other);
        assert!(
            !other.contains("不是。知识图谱记录的是"),
            "只有待确认推断时不得断言图谱否定：{}",
            other
        );
    }

    #[test]
    fn test_check_triple_uses_path_witness() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let answer = reasoner
            .answer("张三的师傅的师傅是王五吗")
            .expect("多跳校验应给出证明路径");
        assert!(answer.contains("是的"), "实际：{}", answer);
        assert!(
            answer.contains("张三 --师傅--> 李四 --师傅--> 王五"),
            "校验应附带路径：{}",
            answer
        );
    }

    #[test]
    fn copula_statement_answer_preserves_direction() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());

        let false_claim = reasoner
            .answer("张三是李四的师傅吗")
            .expect("是非问应给出判定");
        assert!(false_claim.contains("不是"), "实际：{}", false_claim);
        assert!(
            !false_claim.contains("张三 的 师傅 是 李四"),
            "不得把问题方向翻转成另一个查询：{}",
            false_claim
        );

        let true_claim = reasoner
            .answer("李四是张三的师傅吗")
            .expect("反向是非问应给出判定");
        assert!(true_claim.contains("是的"), "实际：{}", true_claim);

        let statement = reasoner
            .answer("张三是李四的师傅")
            .expect("陈述句应给出校验判定");
        assert!(statement.contains("不是"), "实际：{}", statement);
    }

    #[test]
    fn test_session_overlay_extends_paths_and_retract_restores() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&[
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "师傅".to_string(),
                to: "李四".to_string(),
            },
            PrologFact::Relation {
                from: "王五".to_string(),
                rel_type: "师傅".to_string(),
                to: "赵六".to_string(),
            },
        ]);
        let before = reasoner
            .answer("张三的师傅的师傅的师傅是谁")
            .unwrap_or_default();
        assert!(
            !before.contains("赵六"),
            "未断言前不应出现跨分支答案：{}",
            before
        );

        let message = reasoner.assert_nl_fact("王五是李四的师傅");
        assert!(message.contains("已记住"), "实际：{}", message);
        let answer = reasoner
            .answer("张三的师傅的师傅的师傅是谁")
            .expect("会话事实应补齐中间边");
        assert!(answer.contains("赵六"), "实际：{}", answer);
        assert!(answer.contains("路径"), "实际：{}", answer);

        let forgotten = reasoner.retract_nl_fact("王五是李四的师傅");
        assert!(forgotten.contains("已忘记"), "实际：{}", forgotten);
        let after = reasoner
            .answer("张三的师傅的师傅的师傅是谁")
            .unwrap_or_default();
        assert!(!after.contains("赵六"), "撤回后路径应失效：{}", after);
    }

    #[test]
    fn test_negative_session_fact_blocks_paths_and_can_be_retracted() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let message = reasoner.assert_nl_fact("赵六不是王五的师傅");
        assert!(message.contains("已记住否定事实"), "实际：{}", message);

        let blocked = reasoner
            .answer("张三的师傅的师傅的师傅是谁")
            .unwrap_or_default();
        assert!(
            blocked.contains("否定记录阻断"),
            "应报告否定阻断原因：{}",
            blocked
        );

        let checked = reasoner.answer("王五的师傅是赵六吗").unwrap_or_default();
        assert!(
            checked.contains("不是"),
            "直接校验应尊重会话否定：{}",
            checked
        );

        let forgotten = reasoner.retract_nl_fact("赵六不是王五的师傅");
        assert!(forgotten.contains("已撤销否定事实"), "实际：{}", forgotten);
        let restored = reasoner
            .answer("张三的师傅的师傅的师傅是谁")
            .expect("撤销否定后路径应恢复");
        assert!(restored.contains("赵六"), "实际：{}", restored);
    }

    #[test]
    fn test_positive_session_fact_supersedes_negative() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.assert_nl_fact("赵六不是王五的师傅");
        let message = reasoner.assert_nl_fact("王五的师傅是赵六");
        assert!(
            message.contains("已撤销同名会话否定事实"),
            "实际：{}",
            message
        );

        let answer = reasoner
            .answer("张三的师傅的师傅的师傅是谁")
            .expect("肯定事实应恢复路径");
        assert!(answer.contains("赵六"), "实际：{}", answer);
        assert!(!answer.contains("否定记录阻断"), "实际：{}", answer);
        let checked = reasoner.answer("王五的师傅是赵六吗").unwrap_or_default();
        assert!(checked.contains("是的"), "实际：{}", checked);
    }

    #[test]
    fn test_negative_session_fact_roundtrips_snapshot() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.assert_nl_fact("赵六不是王五的师傅");
        let path =
            std::env::temp_dir().join(format!("knowprolog_negative_{}.json", std::process::id()));
        let path_str = path.to_string_lossy().to_string();
        reasoner.save_dialogue_state(&path_str).expect("应能保存");

        let mut restored = SymbolicReasoner::new(test_config(), HashMap::new());
        restored.observe_facts(&master_facts());
        assert!(restored.load_dialogue_state(&path_str), "应能恢复对话状态");
        let answer = restored
            .answer("张三的师傅的师傅的师傅是谁")
            .unwrap_or_default();
        assert!(
            answer.contains("否定记录阻断"),
            "恢复的否定边仍应阻断：{}",
            answer
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_assert_and_retract_session_fact() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let message = reasoner.assert_nl_fact("钱七是孙八的师傅");
        assert!(message.contains("已记住"), "实际：{}", message);
        let proof = reasoner.resolution_proof_report("孙八的师傅是钱七");
        assert!(proof.contains("得证"), "记住的事实应参与后向链：{}", proof);
        let rules = reasoner.session_rules_report();
        assert!(rules.contains("会话事实"), "实际：{}", rules);
        assert!(
            rules.contains("钱七"),
            "会话事实清单应包含新事实：{}",
            rules
        );

        let forgotten = reasoner.retract_nl_fact("钱七是孙八的师傅");
        assert!(forgotten.contains("已忘记"), "实际：{}", forgotten);
        let proof = reasoner.resolution_proof_report("孙八的师傅是钱七");
        assert!(proof.contains("未得证"), "撤回后不应再得证：{}", proof);
    }

    #[test]
    fn retract_session_fact_is_incremental_and_keeps_graph_duplicates() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let perf = reasoner.perf_ledger();
        let rebuilds = perf.count("rebuild_resolver");

        let message = reasoner.assert_nl_fact("钱七是孙八的师傅");
        assert!(message.contains("已记住"), "实际：{}", message);
        assert_eq!(
            perf.count("rebuild_resolver"),
            rebuilds,
            "采纳会话事实不应重建归结引擎"
        );

        // 与图谱同名的会话副本：撤回后图谱事实必须仍然可证
        let duplicate = reasoner.assert_nl_fact("张三是李四的师傅");
        assert!(duplicate.contains("已记住"), "实际：{}", duplicate);
        let forgotten = reasoner.retract_nl_fact("张三是李四的师傅");
        assert!(forgotten.contains("已忘记"), "实际：{}", forgotten);
        let proof = reasoner.resolution_proof_report("李四的师傅是张三");
        assert!(proof.contains("得证"), "同名图谱事实不得被误删：{}", proof);

        let forgotten = reasoner.retract_nl_fact("钱七是孙八的师傅");
        assert!(forgotten.contains("已忘记"), "实际：{}", forgotten);
        let proof = reasoner.resolution_proof_report("孙八的师傅是钱七");
        assert!(proof.contains("未得证"), "撤回后不应再得证：{}", proof);
        assert_eq!(
            perf.count("rebuild_resolver"),
            rebuilds,
            "撤回会话事实应增量删除而不是整库重建"
        );
    }

    #[test]
    fn rebuild_from_graph_drops_old_values_and_keeps_session_facts() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&attribute_facts());
        reasoner.assert_nl_fact("钱七是孙八的师傅");
        let before = reasoner.answer("姚明的身高是什么").expect("重建前应可回答");
        assert!(before.contains("226cm"), "实际：{}", before);

        let mut properties = HashMap::new();
        properties.insert("身高".to_string(), "229cm".to_string());
        reasoner.rebuild_from_graph(&[PrologFact::Entity {
            name: "姚明".to_string(),
            entity_type: "人".to_string(),
            properties,
        }]);

        let after = reasoner.answer("姚明的身高是什么").expect("重建后应可回答");
        assert!(after.contains("229cm"), "实际：{}", after);
        assert!(!after.contains("226cm"), "旧值不应残留：{}", after);
        assert_eq!(reasoner.session_facts().len(), 1, "会话事实应保留");
        let proof = reasoner.resolution_proof_report("孙八的师傅是钱七");
        assert!(
            proof.contains("得证"),
            "会话事实应继续参与后向链：{}",
            proof
        );
    }

    #[test]
    fn test_assert_nl_fact_rejects_conditional_sentence() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let message = reasoner.assert_nl_fact("如果张三的师傅是李四，那么张三的师祖是赵六");
        assert!(message.contains("不是确定事实"), "实际：{}", message);
        assert!(reasoner.session_facts().is_empty());
    }

    #[test]
    fn test_session_clauses_persist_across_snapshot() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.learn_rule_report("如果X的师傅是Y，那么X的老师是Y");
        let path =
            std::env::temp_dir().join(format!("knowprolog_clauses_{}.json", std::process::id()));
        let path_str = path.to_string_lossy().to_string();
        reasoner.save_dialogue_state(&path_str).expect("应能保存");

        let mut restored = SymbolicReasoner::new(test_config(), HashMap::new());
        restored.observe_facts(&master_facts());
        assert!(restored.load_dialogue_state(&path_str), "应能恢复对话状态");
        let proof = restored.resolution_proof_report("李四的老师是王五");
        assert!(
            proof.contains("得证"),
            "恢复后的会话规则应参与后向链：{}",
            proof
        );
        assert!(
            restored.dialogue_summary().contains("会话规则 1 条"),
            "对话记忆应显示会话规则：{}",
            restored.dialogue_summary()
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_rule_ledger_entries_and_fingerprint_track_session_rules() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let empty = reasoner.rule_fingerprint();
        assert!(reasoner.rule_ledger_entries().is_empty());

        reasoner.learn_rule_report("如果X的师傅是Y，那么X的老师是Y");
        let entries = reasoner.rule_ledger_entries();
        assert_eq!(entries.len(), 1, "教过的规则应成为账本条目");
        assert!(
            entries[0].head.display().contains("老师"),
            "实际：{}",
            entries[0].head.display()
        );
        assert_eq!(entries[0].body.len(), 1);
        assert!(
            entries[0].source.contains("中文教学"),
            "来源应可审计：{}",
            entries[0].source
        );
        assert!(!entries[0].id.is_empty());
        assert_eq!(
            reasoner.rule_fingerprint(),
            graph_ledger::rule_fingerprint(&entries),
            "指纹必须由当前规则集确定"
        );
        assert_ne!(reasoner.rule_fingerprint(), empty, "规则增删必须改变指纹");
    }

    #[test]
    fn test_forget_rule_retracts_clause_and_lesson_consistently() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.teach_report("如果X的师傅是Y，那么X的老师是Y");
        assert_eq!(
            reasoner.rule_ledger_entries().len(),
            1,
            "教学规则应立即入账"
        );
        assert!(!reasoner.lessons().is_empty());

        let forgotten = reasoner.forget_rule_report("如果X的师傅是Y，那么X的老师是Y");
        assert!(
            forgotten.contains("已忘记 1 条会话规则"),
            "实际：{}",
            forgotten
        );
        assert!(
            forgotten.contains("撤回 1 条对应教学原句"),
            "实际：{}",
            forgotten
        );
        assert!(
            reasoner.rule_ledger_entries().is_empty(),
            "忘记后账本条目清空"
        );
        assert!(reasoner.lessons().is_empty(), "教学原句应一并撤回");
        let proof = reasoner.resolution_proof_report("李四的老师是王五");
        assert!(proof.contains("未得证"), "忘记后规则不应再得证：{}", proof);

        let again = reasoner.forget_rule_report("如果X的师傅是Y，那么X的老师是Y");
        assert!(
            again.contains("没有匹配的规则"),
            "重复忘记应零状态变化：{}",
            again
        );
    }

    #[test]
    fn test_restore_rules_from_ledger_enables_backward_chain() {
        let mut teacher = SymbolicReasoner::new(test_config(), HashMap::new());
        teacher.observe_facts(&master_facts());
        teacher.learn_rule_report("如果X的师傅是Y，那么X的老师是Y");
        let entries = teacher.rule_ledger_entries();

        let mut learner = SymbolicReasoner::new(test_config(), HashMap::new());
        learner.observe_facts(&master_facts());
        assert_eq!(learner.restore_rules_from_ledger(&entries), 1);
        assert_eq!(
            learner.restore_rules_from_ledger(&entries),
            0,
            "重复恢复不重复添加"
        );
        let proof = learner.resolution_proof_report("李四的老师是王五");
        assert!(proof.contains("得证"), "恢复的规则应参与后向链：{}", proof);
        assert_eq!(
            learner.rule_fingerprint(),
            teacher.rule_fingerprint(),
            "恢复后指纹与原会话一致"
        );
    }

    #[test]
    fn test_derived_ledger_entries_carry_provenance_and_stamp() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&pair_facts());
        reasoner.learn_rule_report("如果X的师傅是Y，那么X的老师是Y");
        let entries = reasoner.derived_ledger_entries(10);
        assert!(!entries.is_empty(), "教过的规则应推出新事实");
        let derived = entries
            .iter()
            .find(|entry| entry.atom.predicate == "老师" && entry.atom.display().contains("李四"))
            .unwrap_or_else(|| panic!("应推出李四的老师：{:?}", entries));
        assert_eq!(derived.depth, 1);
        assert!(
            derived.rule_source.contains("中文教学"),
            "规则来源应可审计：{}",
            derived.rule_source
        );
        assert_eq!(
            derived.premises.len(),
            1,
            "应记录前提：{:?}",
            derived.premises
        );
        assert_eq!(derived.revision, reasoner.revision(), "物化戳应记知识版本");
        assert_eq!(
            derived.fingerprint,
            reasoner.rule_fingerprint(),
            "物化戳应记规则集指纹"
        );
        assert!(
            reasoner.derived_ledger_entries(0).is_empty(),
            "limit=0 不产出条目"
        );
    }

    #[test]
    fn test_problog_multi_sentence_teaching_reports_errors() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let report =
            reasoner.teach_report("如果李四的师傅是王五，那么李四的师祖是赵六。这句话是谁说的？");
        assert!(report.contains("已学会 1 条"), "实际：{}", report);
        assert!(report.contains("未学会"), "应逐条报告失败原因：{}", report);
        assert!(report.contains("疑问句"), "疑问句应被拒绝：{}", report);
    }

    #[test]
    fn test_holographic_memory_feeds_from_facts_and_recalls() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let holo = reasoner.holographic();
        assert_eq!(holo.triple_count(), 4, "全部三元组应进入全息图");
        assert!(holo.dimension() >= 256);

        let objects = holo.recall_object("张三", "师傅");
        assert_eq!(
            objects.first().map(|hit| hit.symbol.as_str()),
            Some("李四"),
            "全息客体召回 top-1 应为李四：{:?}",
            objects
        );
        assert!(objects[0].verified, "事实中的三元组应通过硬核查");

        let subjects = holo.recall_subject("师傅", "王五");
        assert_eq!(
            subjects.first().map(|hit| hit.symbol.as_str()),
            Some("李四")
        );

        let predicates = holo.recall_predicate("张三", "王五");
        assert_eq!(
            predicates.first().map(|hit| hit.symbol.as_str()),
            Some("师祖")
        );

        let analogy = holo.analogy("张三", "李四", "王五");
        assert_eq!(
            analogy.first().map(|hit| hit.symbol.as_str()),
            Some("赵六"),
            "张三:李四 :: 王五:? 应迁移师傅关系召回赵六：{:?}",
            analogy
        );
        assert!(analogy[0].verified);
    }

    #[test]
    fn test_holographic_context_records_mention_trace() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.note_holo_context("张三的师傅是谁？");
        reasoner.note_holo_context("李四的国籍是什么？");
        let holo = reasoner.holographic();
        assert_eq!(holo.context_len(), 2, "两轮不同焦点应产生两条时序记忆");
        let recent = holo.recent_focus(0).expect("应能回读最近焦点");
        assert_eq!(recent.symbol, "李四");
        let previous = holo.recent_focus(1).expect("应能回读上一轮焦点");
        assert_eq!(previous.symbol, "张三");
    }

    #[test]
    fn test_holographic_memory_respects_disabled_config() {
        let mut config = test_config();
        config.enable_holographic_memory = false;
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.note_holo_context("张三的师傅是谁？");
        assert_eq!(reasoner.holographic().triple_count(), 0);
        assert_eq!(reasoner.holographic().context_len(), 0);
    }

    /// VID（版本化增量派生）：重复观察同一批事实必须版本不动、派生不重建。
    #[test]
    fn test_repeat_observation_is_version_stable() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        assert_eq!(reasoner.revision(), 1, "首次观察应产生版本 1");
        let perf = reasoner.perf_ledger();
        let resolver_rebuilds = perf.count("rebuild_resolver");
        let soft_rebuilds = perf.count("rebuild_soft_logic");
        let cache_rebuilds = perf.count("rebuild_knowledge_caches");
        let embedding_rebuilds = perf.count("embedding_rebuild");

        // 整图快照重放：版本与派生台账必须纹丝不动
        reasoner.observe_facts(&master_facts());
        reasoner.observe_facts(&master_facts());
        assert_eq!(reasoner.revision(), 1, "纯重复观察不应推进版本");
        assert_eq!(perf.count("rebuild_resolver"), resolver_rebuilds);
        assert_eq!(perf.count("rebuild_soft_logic"), soft_rebuilds);
        assert_eq!(perf.count("rebuild_knowledge_caches"), cache_rebuilds);
        assert_eq!(perf.count("embedding_rebuild"), embedding_rebuilds);
        assert_eq!(perf.count("observe_no_change"), 2);

        // 加入一条事实：版本推进，派生台账重新挂牌
        let mut extended = master_facts();
        extended.push(PrologFact::Relation {
            from: "王五".to_string(),
            rel_type: "师傅".to_string(),
            to: "钱七".to_string(),
        });
        reasoner.observe_facts(&extended);
        assert_eq!(reasoner.revision(), 2, "新增事实应推进版本");
        assert_eq!(perf.count("rebuild_resolver"), resolver_rebuilds + 1);
        assert!(reasoner.derivation_report().contains("知识版本 2"));
    }

    /// 失效的 memo 必须在知识变化后清空（版本感应的缓存一致性）。
    #[test]
    fn test_memo_invalidated_only_on_change() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let _ = reasoner.entities_mentioned("张三的师傅是谁");
        let perf = reasoner.perf_ledger();
        let hits = perf.count("memo_entity_mentions_hit");
        // 同一文本重复查询：命中记忆
        let _ = reasoner.entities_mentioned("张三的师傅是谁");
        assert_eq!(perf.count("memo_entity_mentions_hit"), hits + 1);

        // 纯重复观察不清缓存：仍应命中
        reasoner.observe_facts(&master_facts());
        let _ = reasoner.entities_mentioned("张三的师傅是谁");
        assert_eq!(perf.count("memo_entity_mentions_hit"), hits + 2);

        // 知识变化后缓存失效：同文本查询不能再命中
        let mut extended = master_facts();
        extended.push(PrologFact::Relation {
            from: "李四".to_string(),
            rel_type: "师傅".to_string(),
            to: "钱七".to_string(),
        });
        reasoner.observe_facts(&extended);
        let _ = reasoner.entities_mentioned("张三的师傅是谁");
        assert_eq!(
            perf.count("memo_entity_mentions_hit"),
            hits + 2,
            "版本变化后过期记忆必须失效"
        );
    }

    /// 多槽记忆：同一知识版本内交替询问多个文本，各自留槽、重复命中；
    /// 版本推进后整体不可命中（纪元失效，不靠 TTL）。
    #[test]
    fn test_memo_keeps_multiple_texts_within_revision() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let perf = reasoner.perf_ledger();

        let _ = reasoner.entities_mentioned("张三的师傅是谁");
        let _ = reasoner.entities_mentioned("李四的师傅是谁");
        let hits = perf.count("memo_entity_mentions_hit");
        // 交替再问：两个文本都应命中（不得互相挤掉）
        let _ = reasoner.entities_mentioned("张三的师傅是谁");
        let _ = reasoner.entities_mentioned("李四的师傅是谁");
        assert_eq!(perf.count("memo_entity_mentions_hit"), hits + 2);

        // 链接记忆同样多槽
        let _ = reasoner.linked_mentions("张三的师傅是谁");
        let _ = reasoner.linked_mentions("李四的师傅是谁");
        let link_hits = perf.count("memo_link_hit");
        let _ = reasoner.linked_mentions("张三的师傅是谁");
        let _ = reasoner.linked_mentions("李四的师傅是谁");
        assert_eq!(perf.count("memo_link_hit"), link_hits + 2);

        // 知识版本推进：原文本不得命中
        let mut extended = master_facts();
        extended.push(PrologFact::Relation {
            from: "王五".to_string(),
            rel_type: "师傅".to_string(),
            to: "钱七".to_string(),
        });
        reasoner.observe_facts(&extended);
        let before = perf.count("memo_entity_mentions_hit");
        let _ = reasoner.entities_mentioned("张三的师傅是谁");
        assert_eq!(perf.count("memo_entity_mentions_hit"), before);
    }

    /// WarmBase 的跨线程前提：分叉基座必须可 `Send`。
    #[test]
    fn reasoner_is_send_for_cross_thread_fork() {
        fn assert_send<T: Send>() {}
        assert_send::<SymbolicReasoner>();
    }

    /// 分叉契约：从「观察完毕的基座」分叉，与「新建 + 重放同一批事实」在
    /// 固定查询集与固定对话序列上逐字节一致（回答、焦点、历史、接地清单）。
    #[test]
    fn fork_matches_fresh_observe_byte_for_byte() {
        let facts = master_facts();
        let config = test_config();

        let mut base = SymbolicReasoner::new(config.clone(), HashMap::new());
        base.observe_facts(&facts);
        let mut forked = base.fork_session();

        let mut fresh = SymbolicReasoner::new(config, HashMap::new());
        fresh.observe_facts(&facts);

        assert_eq!(forked.revision(), fresh.revision(), "知识版本必须一致");

        // 固定查询集：覆盖关系查询、逆向谓词、属性、链式与未知实体
        let questions = [
            "张三的师傅是谁",
            "李四的师祖是谁",
            "王五的师傅是谁",
            "赵六的师傅是谁",
            "张三的师傅的师傅是谁",
            "姚明的身高是什么",
            "秦始皇是谁",
            "张三多大",
        ];
        for question in questions {
            assert_eq!(
                forked.answer(question),
                fresh.answer(question),
                "回答必须逐字节一致：{}",
                question
            );
        }
        // 接地清单（提及解码 + 链接记忆路径）也必须一致
        let forked_grounding = forked.grounding_mentions("张三的师傅是谁");
        let fresh_grounding = fresh.grounding_mentions("张三的师傅是谁");
        assert_eq!(
            format!("{:?}", forked_grounding),
            format!("{:?}", fresh_grounding),
            "接地清单必须一致"
        );

        // 固定对话序列：多轮上下文状态（焦点/历史）必须一致
        for turn in ["张三的师傅是谁", "那李四呢", "他的师傅是谁"] {
            let _ = forked.contextualize(turn);
            let _ = fresh.contextualize(turn);
        }
        assert_eq!(
            forked.contextualize("王五的师傅是谁"),
            fresh.contextualize("王五的师傅是谁")
        );
        assert_eq!(
            format!("{:?}", forked.context.focus_entities()),
            format!("{:?}", fresh.context.focus_entities())
        );
    }

    /// 分叉不得把上一会话的临时事实与假设前提留在归结引擎里。
    #[test]
    fn fork_session_drops_session_and_assumption_premises() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        assert!(reasoner.adopt_session_triple("李四", "师傅", "钱七"));
        reasoner
            .enter_assumption_world("赵六的师傅是钱八")
            .expect("假设世界应可进入");
        let session_atom = Atom::rel("师傅", "李四", "钱七");
        let assumption_atom = Atom::rel("师傅", "赵六", "钱八");
        assert!(reasoner.resolver().has_fact(&session_atom));
        assert!(reasoner.resolver().has_fact(&assumption_atom));

        let forked = reasoner.fork_session();
        assert!(forked.session_facts().is_empty(), "分叉不得继承会话事实");
        assert!(forked.assumption_world().is_none(), "分叉不得继承假设世界");
        assert!(
            !forked.resolver().has_fact(&session_atom),
            "分叉归结引擎不得残留会话事实"
        );
        assert!(
            !forked.resolver().has_fact(&assumption_atom),
            "分叉归结引擎不得残留假设事实"
        );
        assert!(
            !forked
                .answer("李四的师傅是谁")
                .unwrap_or_default()
                .contains("钱七"),
            "分叉后的回答不得引用会话临时事实"
        );
    }

    /// 分叉必须清零会话态：源实例用过的焦点/教学不得泄漏进新会话。
    #[test]
    fn fork_session_clears_session_state_only() {
        let config = test_config();
        let mut used = SymbolicReasoner::new(config, HashMap::new());
        used.observe_facts(&master_facts());
        let _ = used.contextualize("张三的师傅是谁");
        let _ = used.answer("张三的师傅是谁");
        assert!(!used.context.focus_entities().is_empty(), "源实例应有焦点");

        let forked = used.fork_session();
        assert!(
            forked.context.focus_entities().is_empty(),
            "分叉不得继承焦点"
        );
        assert_eq!(forked.revision(), used.revision(), "知识版本随分叉保留");
        // 知识仍在：同一问句回答与源实例一致
        assert_eq!(
            forked.answer("张三的师傅是谁"),
            used.answer("张三的师傅是谁")
        );
    }

    fn transitive_registry(
        predicate: &str,
        symmetric: bool,
    ) -> crate::predicate_registry::PredicateRegistry {
        use crate::predicate_registry::{PredicateProfileConfig, PredicateRegistry};
        let mut registry = PredicateRegistry::new();
        registry.declare(&PredicateProfileConfig {
            name: predicate.to_string(),
            transitive: !symmetric,
            symmetric,
            ..Default::default()
        });
        registry
    }

    fn inverse_registry() -> crate::predicate_registry::PredicateRegistry {
        use crate::predicate_registry::{PredicateProfileConfig, PredicateRegistry};
        let mut registry = PredicateRegistry::new();
        registry.declare(&PredicateProfileConfig {
            name: "师傅".to_string(),
            inverse: Some("徒弟".to_string()),
            ..Default::default()
        });
        registry.declare(&PredicateProfileConfig {
            name: "徒弟".to_string(),
            inverse: Some("师傅".to_string()),
            ..Default::default()
        });
        registry
    }

    /// 重复的传递谓词由 Closure 算子压缩，回答必须引用具体定律。
    #[test]
    fn law_closure_answers_repeated_transitive_chain() {
        let mut config = test_config();
        config.enable_link_prediction = false;
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        reasoner.observe_facts(&[
            PrologFact::Relation {
                from: "甲".to_string(),
                rel_type: "上司".to_string(),
                to: "乙".to_string(),
            },
            PrologFact::Relation {
                from: "乙".to_string(),
                rel_type: "上司".to_string(),
                to: "丙".to_string(),
            },
            PrologFact::Relation {
                from: "丙".to_string(),
                rel_type: "上司".to_string(),
                to: "丁".to_string(),
            },
        ]);
        reasoner.configure_laws(&transitive_registry("上司", false));

        let answer = reasoner
            .answer("甲的上司的上司的上司是谁")
            .expect("传递律应支持重复谓词链");
        assert!(answer.contains("丁"), "{}", answer);
        assert!(answer.contains("推理依据"), "{}", answer);
        assert!(answer.contains("传递律"), "{}", answer);
        assert!(answer.contains("law:transitive:上司"), "{}", answer);

        // 两跳问句必须恰好两跳：不能把三跳才可达的「丁」也算进来。
        let two_hops = reasoner
            .answer("甲的上司的上司是谁")
            .expect("两跳链应有答案");
        assert!(two_hops.contains("丙"), "{}", two_hops);
        assert!(!two_hops.contains("丁"), "{}", two_hops);
    }

    fn closure_reasoner() -> SymbolicReasoner {
        let mut config = test_config();
        config.enable_link_prediction = false;
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        reasoner.observe_facts(&[
            PrologFact::Relation {
                from: "甲".to_string(),
                rel_type: "上司".to_string(),
                to: "乙".to_string(),
            },
            PrologFact::Relation {
                from: "乙".to_string(),
                rel_type: "上司".to_string(),
                to: "丙".to_string(),
            },
            PrologFact::Relation {
                from: "丙".to_string(),
                rel_type: "上司".to_string(),
                to: "丁".to_string(),
            },
        ]);
        reasoner.configure_laws(&transitive_registry("上司", false));
        reasoner
    }

    #[test]
    fn all_quantifier_uses_closure_even_when_direct_edge_exists() {
        let reasoner = closure_reasoner();
        let answer = reasoner
            .answer("甲的所有上司是谁")
            .expect("所有问句应由闭包回答");
        assert!(answer.contains("乙"), "{}", answer);
        assert!(answer.contains("丙"), "{}", answer);
        assert!(answer.contains("丁"), "{}", answer);
        assert!(answer.contains("共 3 个"), "{}", answer);
        assert!(answer.contains("law:transitive:上司"), "{}", answer);
        assert!(answer.contains("甲 → 乙 → 丙"), "{}", answer);
        assert!(!answer.contains("下界"), "{}", answer);
    }

    #[test]
    fn exact_hop_closure_does_not_need_transitive_law() {
        let mut config = test_config();
        config.enable_link_prediction = false;
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        reasoner.observe_facts(&[
            PrologFact::Relation {
                from: "甲".to_string(),
                rel_type: "上司".to_string(),
                to: "乙".to_string(),
            },
            PrologFact::Relation {
                from: "乙".to_string(),
                rel_type: "上司".to_string(),
                to: "丙".to_string(),
            },
            PrologFact::Relation {
                from: "丙".to_string(),
                rel_type: "上司".to_string(),
                to: "丁".to_string(),
            },
        ]);
        let answer = reasoner
            .answer("甲恰好两跳的上司是谁")
            .expect("精确跳数无需传递律");
        assert!(answer.contains("丙"), "{}", answer);
        assert!(!answer.contains("1. 乙"), "{}", answer);
        assert!(!answer.contains("丁"), "{}", answer);
        assert!(answer.contains("共 1 个"), "{}", answer);
        assert!(answer.contains("精确跳数"), "{}", answer);
        assert!(!answer.contains("law:transitive"), "{}", answer);

        let count = reasoner
            .answer("甲恰好 2 跳的上司有几个")
            .expect("精确跳数计数");
        assert!(count.contains("共 1 个"), "{}", count);
    }

    #[test]
    fn closure_respects_session_negative_edges() {
        let mut reasoner = closure_reasoner();
        let message = reasoner.assert_nl_fact("乙不是甲的上司");
        assert!(message.contains("已记住否定事实"), "{}", message);
        let answer = reasoner.answer("甲的所有上司是谁").expect("否定后仍应回答");
        assert!(!answer.contains("乙"), "{}", answer);
        assert!(!answer.contains("丙"), "{}", answer);
        assert!(answer.contains("没有所有「上司」"), "{}", answer);
    }

    #[test]
    fn closure_includes_session_facts_in_reachability() {
        let mut reasoner = closure_reasoner();
        let message = reasoner.assert_nl_fact("丙的上司是丁");
        assert!(message.contains("已记住"), "{}", message);
        reasoner.observe_facts(&[PrologFact::Relation {
            from: "丁".to_string(),
            rel_type: "上司".to_string(),
            to: "戊".to_string(),
        }]);
        let answer = reasoner
            .answer("甲的所有上司是谁")
            .expect("会话事实应参与闭包");
        assert!(answer.contains("戊"), "{}", answer);
        assert!(!answer.contains("共 3 个"), "{}", answer);
    }

    #[test]
    fn all_quantifier_without_transitive_law_refuses_to_claim_closure() {
        let mut config = test_config();
        config.enable_link_prediction = false;
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        reasoner.observe_facts(&[
            PrologFact::Relation {
                from: "甲".to_string(),
                rel_type: "上司".to_string(),
                to: "乙".to_string(),
            },
            PrologFact::Relation {
                from: "乙".to_string(),
                rel_type: "上司".to_string(),
                to: "丙".to_string(),
            },
        ]);
        let answer = reasoner
            .answer("甲的所有上司是谁")
            .expect("无传递律时必须给出可解释的拒绝");
        assert!(answer.contains("尚未声明为传递关系"), "{}", answer);
        assert!(answer.contains("直接记录：乙"), "{}", answer);
        assert!(!answer.contains("共 2 个"), "{}", answer);
    }

    #[test]
    fn closure_truncation_reports_a_lower_bound() {
        let mut config = test_config();
        config.enable_link_prediction = false;
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        let names = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"];
        let facts: Vec<PrologFact> = names
            .windows(2)
            .map(|pair| PrologFact::Relation {
                from: pair[0].to_string(),
                rel_type: "上司".to_string(),
                to: pair[1].to_string(),
            })
            .collect();
        reasoner.observe_facts(&facts);
        reasoner.configure_laws(&transitive_registry("上司", false));

        let answer = reasoner
            .answer("甲的所有上司是谁")
            .expect("超深链仍应给出有界答案");
        assert!(answer.contains("至少"), "{}", answer);
        assert!(answer.contains("深度上限"), "{}", answer);
        assert!(answer.contains("下界"), "{}", answer);
        assert!(answer.contains("至少 8 个"), "{}", answer);
        assert!(answer.contains("壬"), "{}", answer);
        assert!(!answer.contains("癸"), "{}", answer);
    }

    fn inverse_transitive_registry() -> crate::predicate_registry::PredicateRegistry {
        use crate::predicate_registry::{PredicateProfileConfig, PredicateRegistry};
        let mut registry = PredicateRegistry::new();
        registry.declare(&PredicateProfileConfig {
            name: "上司".to_string(),
            transitive: true,
            inverse: Some("下属".to_string()),
            ..Default::default()
        });
        registry.declare(&PredicateProfileConfig {
            name: "下属".to_string(),
            inverse: Some("上司".to_string()),
            ..Default::default()
        });
        registry
    }

    fn transitive_predicate_registry(
        predicate: &str,
    ) -> crate::predicate_registry::PredicateRegistry {
        use crate::predicate_registry::{PredicateProfileConfig, PredicateRegistry};
        let mut registry = PredicateRegistry::new();
        registry.declare(&PredicateProfileConfig {
            name: predicate.to_string(),
            transitive: true,
            ..Default::default()
        });
        registry
    }

    #[test]
    fn truncated_law_closure_is_disclosed_in_chain_answer() {
        let mut config = test_config();
        config.enable_link_prediction = false;
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        let mut facts = vec![PrologFact::Relation {
            from: "甲".to_string(),
            rel_type: "上司".to_string(),
            to: "乙".to_string(),
        }];
        let names = ["乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"];
        for pair in names.windows(2) {
            facts.push(PrologFact::Relation {
                from: pair[0].to_string(),
                rel_type: "上司".to_string(),
                to: pair[1].to_string(),
            });
        }
        facts.push(PrologFact::Relation {
            from: "壬".to_string(),
            rel_type: "师傅".to_string(),
            to: "子".to_string(),
        });
        reasoner.observe_facts(&facts);
        reasoner.configure_laws(&transitive_predicate_registry("上司"));

        let answer = reasoner
            .answer("甲的上司的师傅是谁")
            .expect("混合链应由律闭包补充");
        assert!(
            answer.contains("传递闭包扩展达到深度或节点上限"),
            "{}",
            answer
        );
        assert!(answer.contains("子"), "{}", answer);
        assert!(answer.contains("law:transitive:上司"), "{}", answer);
    }

    #[test]
    fn closure_respects_inverted_predicate_direction() {
        let mut config = test_config();
        config.enable_link_prediction = false;
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        reasoner.observe_facts(&[
            PrologFact::Relation {
                from: "甲".to_string(),
                rel_type: "上司".to_string(),
                to: "乙".to_string(),
            },
            PrologFact::Relation {
                from: "乙".to_string(),
                rel_type: "上司".to_string(),
                to: "丙".to_string(),
            },
        ]);
        reasoner.configure_laws(&inverse_transitive_registry());
        reasoner.register_predicate_query_surface("下属", "上司");
        let answer = reasoner
            .answer("丙的所有下属是谁")
            .expect("逆向闭包应有答案");
        assert!(answer.contains("甲"), "{}", answer);
        assert!(answer.contains("乙"), "{}", answer);
        assert!(answer.contains("共 2 个"), "{}", answer);
        assert!(answer.contains("law:transitive:上司"), "{}", answer);
    }

    #[test]
    fn closure_composes_with_following_relation() {
        let mut reasoner = closure_reasoner();
        reasoner.observe_facts(&[
            PrologFact::Relation {
                from: "乙".to_string(),
                rel_type: "师傅".to_string(),
                to: "戊".to_string(),
            },
            PrologFact::Relation {
                from: "丙".to_string(),
                rel_type: "师傅".to_string(),
                to: "己".to_string(),
            },
            PrologFact::Relation {
                from: "丁".to_string(),
                rel_type: "师傅".to_string(),
                to: "庚".to_string(),
            },
        ]);
        let answer = reasoner
            .answer("甲的所有上司的师傅是谁")
            .expect("闭包后接关系应可复合");
        assert!(answer.contains("共 3 个"), "{}", answer);
        assert!(answer.contains("戊"), "{}", answer);
        assert!(answer.contains("己"), "{}", answer);
        assert!(answer.contains("庚"), "{}", answer);
        assert!(answer.contains("甲 → 乙 → 戊"), "{}", answer);
        assert!(answer.contains("law:transitive:上司"), "{}", answer);

        let count = reasoner
            .answer("甲的所有上司的师傅有几个")
            .expect("复合计数");
        assert!(count.contains("共 3 个"), "{}", count);
        assert!(count.contains("不同最终节点"), "{}", count);

        let blocked = reasoner.assert_nl_fact("戊不是乙的师傅");
        assert!(blocked.contains("已记住否定事实"), "{}", blocked);
        let after_negative = reasoner
            .answer("甲的所有上司的师傅是谁")
            .expect("否定边应阻断复合链");
        assert!(!after_negative.contains("戊"), "{}", after_negative);
        assert!(after_negative.contains("共 2 个"), "{}", after_negative);
        assert!(after_negative.contains("己"), "{}", after_negative);
        assert!(after_negative.contains("庚"), "{}", after_negative);
    }

    #[test]
    fn all_quantifier_count_uses_distinct_nodes() {
        let reasoner = closure_reasoner();
        let answer = reasoner
            .answer("甲的所有上司一共有几个")
            .expect("所有+计数应走闭包计数");
        assert!(answer.contains("共 3 个"), "{}", answer);
        assert!(answer.contains("不同可达节点"), "{}", answer);
        assert!(!answer.contains("1. 乙"), "{}", answer);
    }

    /// 对称律补足单向事实：反向问句给出反方向取值，且说明依据。
    #[test]
    fn symmetric_law_recovers_reverse_direction() {
        let mut config = test_config();
        config.enable_link_prediction = false;
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        reasoner.observe_facts(&[PrologFact::Relation {
            from: "张三".to_string(),
            rel_type: "配偶".to_string(),
            to: "李四".to_string(),
        }]);
        reasoner.configure_laws(&transitive_registry("配偶", true));

        let answer = reasoner
            .answer("李四的配偶是谁")
            .expect("对称律应恢复反向命中");
        assert!(answer.contains("张三"), "{}", answer);
        assert!(answer.contains("对称律"), "{}", answer);

        let check = reasoner
            .answer("李四的配偶是张三吗")
            .expect("校验问句也应使用对称律");
        assert!(check.starts_with("是的"), "{}", check);
        assert!(check.contains("对称律"), "{}", check);
    }

    /// 持久化账本的退役决定必须压过配置声明，并在后续知识重建中保持。
    #[test]
    fn persisted_retirement_disables_inverse_law_and_survives_rebuild() {
        let mut config = test_config();
        config.enable_link_prediction = false;
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        reasoner.observe_facts(&[PrologFact::Relation {
            from: "张三".to_string(),
            rel_type: "师傅".to_string(),
            to: "李四".to_string(),
        }]);
        reasoner.configure_laws(&inverse_registry());

        let before = reasoner
            .answer("李四的徒弟是谁")
            .expect("声明互逆律应支持反向问句");
        assert!(before.contains("张三"), "{}", before);

        let mut persisted = LawBook::from_registry(&inverse_registry());
        assert!(persisted.retire("law:inverse:师傅|徒弟"));
        assert!(reasoner.load_persisted_laws(persisted) >= 1);

        let after = reasoner.answer("李四的徒弟是谁");
        assert!(
            after.as_deref().map_or(true, |text| !text.contains("张三")),
            "{:?}",
            after
        );

        // 新事实触发缓存重建，退役决定不能被重建冲掉。
        reasoner.observe_facts(&[PrologFact::Relation {
            from: "王五".to_string(),
            rel_type: "师傅".to_string(),
            to: "赵六".to_string(),
        }]);
        let rebuilt = reasoner.answer("李四的徒弟是谁");
        assert!(
            rebuilt
                .as_deref()
                .map_or(true, |text| !text.contains("张三")),
            "{:?}",
            rebuilt
        );
    }

    #[test]
    fn query_board_scans_reasoner_entities_and_relations() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&[
            PrologFact::Entity {
                name: "姚明".to_string(),
                entity_type: "Person".to_string(),
                properties: HashMap::from([("身高".to_string(), "226".to_string())]),
            },
            PrologFact::Entity {
                name: "王治郅".to_string(),
                entity_type: "Person".to_string(),
                properties: HashMap::from([("身高".to_string(), "214".to_string())]),
            },
            PrologFact::Relation {
                from: "姚明".to_string(),
                rel_type: "球队".to_string(),
                to: "火箭队".to_string(),
            },
        ]);
        let mut manager = crate::dialogue_manager::DialogueManager::new();
        let reply = manager
            .plan_reply("列出所有人的身高", &reasoner)
            .expect("查询板应读取推理器索引");
        assert!(reply.text.contains("姚明"), "{}", reply.text);
        assert!(reply.text.contains("226"), "{}", reply.text);
        let selected = manager
            .plan_reply("第一个呢", &reasoner)
            .expect("查询板应支持序号选择");
        assert!(selected.text.contains("当前选中"), "{}", selected.text);
        let compare = manager
            .plan_reply("跟王治郅比呢", &reasoner)
            .expect("查询板应支持实体比较");
        assert!(compare.text.contains("王治郅"), "{}", compare.text);
        assert!(compare.text.contains("214"), "{}", compare.text);
    }

    #[test]
    fn svo_and_passive_questions_compile_to_shared_relation_plans() {
        let facts = vec![
            PrologFact::Entity {
                name: "张三".into(),
                entity_type: "人物".into(),
                properties: HashMap::new(),
            },
            PrologFact::Entity {
                name: "李四".into(),
                entity_type: "人物".into(),
                properties: HashMap::new(),
            },
            PrologFact::Entity {
                name: "王五".into(),
                entity_type: "人物".into(),
                properties: HashMap::new(),
            },
            PrologFact::Relation {
                from: "李四".into(),
                rel_type: "认识".into(),
                to: "张三".into(),
            },
            PrologFact::Relation {
                from: "张三".into(),
                rel_type: "认识".into(),
                to: "王五".into(),
            },
        ];
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&facts);

        let object = reasoner.answer("张三认识谁").expect("正向问宾语");
        assert!(object.contains("王五"), "{}", object);
        assert!(!object.contains("李四"), "{}", object);

        let subject = reasoner.answer("谁认识张三").expect("正向问主语");
        assert!(subject.contains("李四"), "{}", subject);

        let yes = reasoner.answer("张三认识王五吗").expect("是非问");
        assert!(yes.contains("是的"), "{}", yes);
        let no = reasoner.answer("张三认识李四吗").expect("是非问");
        assert!(no.contains("不是"), "{}", no);

        let passive_object = reasoner.answer("谁被张三认识").expect("被字句问受事");
        assert!(passive_object.contains("王五"), "{}", passive_object);
        let passive_subject = reasoner.answer("张三被谁认识").expect("被字句问施事");
        assert!(passive_subject.contains("李四"), "{}", passive_subject);
        let passive_yes = reasoner.answer("张三被李四认识吗").expect("被字句是非问");
        assert!(passive_yes.contains("是的"), "{}", passive_yes);
        let existence = reasoner.answer("张三有没有认识王五").expect("有没有是非问");
        assert!(existence.contains("是的"), "{}", existence);
        let relative = reasoner.answer("认识张三的是谁").expect("关系从句问施事");
        assert!(relative.contains("李四"), "{}", relative);
        let passive_relative = reasoner
            .answer("被张三认识的是谁")
            .expect("被字关系从句问受事");
        assert!(passive_relative.contains("王五"), "{}", passive_relative);
        let negated_false = reasoner.answer("张三不认识王五吗").expect("否定是非问");
        assert!(negated_false.starts_with("不是"), "{}", negated_false);
        let negated_true = reasoner.answer("张三不认识李四吗").expect("否定是非问");
        assert!(negated_true.starts_with("是的"), "{}", negated_true);
        let a_not_a = reasoner.answer("张三认不认识王五").expect("A 不 A 是非问");
        assert!(a_not_a.starts_with("是的"), "{}", a_not_a);
        let passive_negated = reasoner
            .answer("张三没有被李四认识吗")
            .expect("被字句标记否定");
        assert!(passive_negated.starts_with("不是"), "{}", passive_negated);

        let synonyms = HashMap::from([("老张".to_string(), "张三".to_string())]);
        let mut aliased = SymbolicReasoner::new(test_config(), synonyms);
        aliased.observe_facts(&facts);
        let alias = aliased.answer("老张认识谁").expect("同义词别名问句");
        assert!(alias.contains("王五"), "{}", alias);

        let receipt = reasoner
            .query_rewrite_receipt("谁被张三认识")
            .expect("重写回执");
        assert!(receipt.contains("passive-wh-patient"), "{}", receipt);
        assert!(receipt.contains("FindObjectByPredicate"), "{}", receipt);

        let derivation = reasoner
            .answer_derivation("张三认识谁", Some("王五"))
            .expect("关系问句应可重建证明");
        assert!(derivation.is_proved(), "{:?}", derivation);
        assert_eq!(derivation.answer_value(), Some("王五"));
    }

    /// 图谱用英文键（knows）而用户说中文词面（认识）时，反查主体问句也应
    /// 走同一条关系计划，而不是退化成「无可用依据」。
    #[test]
    fn generated_predicate_surface_supports_reverse_relation_questions() {
        let facts = vec![
            PrologFact::Entity {
                name: "张三".into(),
                entity_type: "人物".into(),
                properties: HashMap::new(),
            },
            PrologFact::Entity {
                name: "李四".into(),
                entity_type: "人物".into(),
                properties: HashMap::new(),
            },
            PrologFact::Entity {
                name: "王五".into(),
                entity_type: "人物".into(),
                properties: HashMap::new(),
            },
            PrologFact::Relation {
                from: "张三".into(),
                rel_type: "knows".into(),
                to: "李四".into(),
            },
            PrologFact::Relation {
                from: "王五".into(),
                rel_type: "knows".into(),
                to: "李四".into(),
            },
        ];
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&facts);

        let answer = reasoner.answer("谁认识李四").expect("中文词面反查主体");
        assert!(answer.contains("张三"), "{}", answer);
        assert!(answer.contains("王五"), "{}", answer);

        let receipt = reasoner
            .query_rewrite_receipt("谁认识李四")
            .expect("中文词面应可重写");
        assert!(receipt.contains("FindSubject"), "{}", receipt);
    }

    #[test]
    fn quantified_and_relative_questions_use_relation_plans() {
        let facts = vec![
            PrologFact::Entity {
                name: "张三".into(),
                entity_type: "老师".into(),
                properties: HashMap::new(),
            },
            PrologFact::Entity {
                name: "李四".into(),
                entity_type: "学生".into(),
                properties: HashMap::new(),
            },
            PrologFact::Entity {
                name: "王五".into(),
                entity_type: "学生".into(),
                properties: HashMap::new(),
            },
            PrologFact::Relation {
                from: "张三".into(),
                rel_type: "教".into(),
                to: "李四".into(),
            },
            PrologFact::Relation {
                from: "张三".into(),
                rel_type: "教".into(),
                to: "王五".into(),
            },
            PrologFact::Relation {
                from: "李四".into(),
                rel_type: "教".into(),
                to: "张三".into(),
            },
            PrologFact::Relation {
                from: "张三".into(),
                rel_type: "认识".into(),
                to: "李四".into(),
            },
            PrologFact::Relation {
                from: "张三".into(),
                rel_type: "认识".into(),
                to: "王五".into(),
            },
            PrologFact::Relation {
                from: "王五".into(),
                rel_type: "认识".into(),
                to: "李四".into(),
            },
        ];
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&facts);

        let objects = reasoner
            .answer("张三教过哪些学生")
            .expect("量词名词短语问宾语");
        assert!(objects.contains("李四"), "{}", objects);
        assert!(objects.contains("王五"), "{}", objects);

        let subjects = reasoner
            .answer("哪些老师教过张三")
            .expect("量词名词短语问主语");
        assert!(subjects.contains("李四"), "{}", subjects);

        let relative_object = reasoner
            .answer("张三教过的学生是谁")
            .expect("关系从句问宾语");
        assert!(relative_object.contains("李四"), "{}", relative_object);
        assert!(relative_object.contains("王五"), "{}", relative_object);

        let relative_subject = reasoner
            .answer("教过张三的老师有哪些")
            .expect("关系从句问主语");
        assert!(relative_subject.contains("李四"), "{}", relative_subject);

        let count = reasoner.answer("张三教过多少个学生").expect("关系计数问句");
        assert!(count.contains("共有 2 个"), "{}", count);
        assert!(count.contains("李四"), "{}", count);
        assert!(count.contains("王五"), "{}", count);
        let relative_count = reasoner
            .answer("张三教过的学生有多少")
            .expect("关系从句计数问句");
        assert!(relative_count.contains("共有 2 个"), "{}", relative_count);
        assert!(!relative_count.contains("类型「学生」"), "{}", relative_count);
        let measured = reasoner
            .answer("张三教过多少位学生")
            .expect("带量词的关系计数问句");
        assert!(measured.contains("共有 2 位"), "{}", measured);
        let named = reasoner
            .answer("张三教过多少名学生")
            .expect("名量词也应保留");
        assert!(named.contains("共有 2 名"), "{}", named);

        reasoner.set_nlp_config(trace_nlp_config());
        reasoner
            .observe_lexical_text("三位学生")
            .expect("应能观察词法证据");
        let learned = reasoner
            .answer("张三教过多少学生")
            .expect("不写量词时应使用学到的分类词");
        assert!(learned.contains("共有 2 位"), "{}", learned);

        let except_object = reasoner
            .answer("除了李四，张三还认识谁")
            .expect("集合差问宾语");
        assert!(except_object.contains("王五"), "{}", except_object);
        assert!(except_object.contains("已排除：李四"), "{}", except_object);

        let except_subject = reasoner
            .answer("除了张三，还有谁认识李四")
            .expect("集合差问主语");
        assert!(except_subject.contains("王五"), "{}", except_subject);
        assert!(
            except_subject.contains("已排除：张三"),
            "{}",
            except_subject
        );

        let receipt = reasoner
            .query_rewrite_receipt("张三教过的学生是谁")
            .expect("关系从句回执");
        assert!(receipt.contains("typed-relative-object"), "{}", receipt);
        let except_receipt = reasoner
            .query_rewrite_receipt("除了李四，张三还认识谁")
            .expect("集合差回执");
        assert!(
            except_receipt.contains("except-wh-object"),
            "{}",
            except_receipt
        );
    }

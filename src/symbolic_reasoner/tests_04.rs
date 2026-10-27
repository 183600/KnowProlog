    #[test]
    fn test_resolution_proof_view_is_structured_and_complete() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());

        // 陈述句：与报告同路径求解，结构可视图化
        let statement = reasoner
            .resolution_proof_view("张三的师傅的师傅是王五")
            .expect("陈述句应可视图化");
        assert!(statement.phrase.contains("王五"), "{}", statement.phrase);
        assert_eq!(statement.solutions.len(), 1);
        let nodes = crate::view_render::proof_forest_node_count(&statement.solutions[0].proofs);
        assert!(nodes >= 2, "链式证明至少覆盖两个事实：{}", nodes);
        let mermaid = crate::view_render::render_proof_mermaid(&statement.solutions[0].proofs);
        let rendered_nodes = mermaid
            .lines()
            .filter(|line| line.contains('[') && line.trim_start().starts_with('n'))
            .count();
        assert_eq!(
            rendered_nodes, nodes,
            "渲染节点集必须等于展平集合（不增不减）"
        );
        assert_eq!(
            mermaid,
            crate::view_render::render_proof_mermaid(&statement.solutions[0].proofs)
        );

        // 开放问句：绑定含答案变量
        let question = reasoner
            .resolution_proof_view("张三的师傅的师傅是谁")
            .expect("开放问句应可视图化");
        let solution = &question.solutions[0];
        assert!(
            solution
                .bindings
                .iter()
                .any(|(name, value)| name == "答案" && value == "王五"),
            "{:?}",
            solution.bindings
        );

        // 未得证：结构仍在，解为空（开世界假设）
        let unknown = reasoner
            .resolution_proof_view("张三的师傅的师傅是赵六")
            .expect("可解析但未得证");
        assert!(unknown.solutions.is_empty());

        // 不可形式化：None，而不是伪造空证明
        assert!(reasoner.resolution_proof_view("@@@").is_none());
    }

    #[test]
    fn test_answer_derivation_is_bounded_and_anchored_to_answer() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());

        let derivation = reasoner
            .answer_derivation("张三的师傅的师傅是谁", Some("王五是张三的师祖"))
            .expect("应能从同一份归结数据重建推导");
        assert!(derivation.is_proved());
        assert_eq!(derivation.answer_value(), Some("王五"));
        assert!(!derivation.steps.is_empty());
        let text = derivation.render_explanation();
        assert!(text.contains("王五"), "{}", text);
        assert!(text.contains("证明步骤"), "{}", text);

        // 证明与上一轮回答不匹配时宁可不解释
        assert!(reasoner
            .answer_derivation("张三的师傅的师傅是谁", Some("赵六"))
            .is_none());
        // 无法形式化时诚实返回 None
        assert!(reasoner.answer_derivation("@@@", None).is_none());
    }

    #[test]
    fn test_dialogue_why_uses_resolution_proof_end_to_end() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let mut manager = crate::dialogue_manager::DialogueManager::new();
        let answer = manager
            .plan_reply("张三的师傅的师傅是谁", &reasoner)
            .expect("链式问句应被对话管理器接管");
        assert!(answer.text.contains("王五"), "{}", answer.text);

        let why = manager
            .plan_pure_dialogue("怎么推出来的？", &reasoner)
            .expect("推导追问应被接管");
        assert!(why.text.contains("证明步骤"), "{}", why.text);
        assert!(why.text.contains("事实"), "{}", why.text);
        assert!(why.text.contains("王五"), "{}", why.text);
    }

    #[test]
    fn test_resolution_proves_existence_via_association_rule() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&nationality_facts());
        let report = reasoner.resolution_proof_report("赵六有国籍");
        assert!(report.contains("得证"), "实际：{}", report);
        assert!(
            report.contains("关联规则"),
            "应有归结算子句来源：{}",
            report
        );
        assert!(report.contains("国籍"));
        let known = reasoner.resolution_proof_report("张三有国籍");
        assert!(known.contains("得证"));
        assert!(known.contains("事实"));
    }

    #[test]
    fn test_association_report_and_suggestions() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&nationality_facts());
        let report = reasoner.association_report(None);
        assert!(report.contains("支持度"), "实际：{}", report);
        assert!(report.contains("提升度"));
        assert!(report.contains("出生地"));
        let filtered = reasoner.association_report(Some("国籍"));
        assert!(filtered.contains("国籍"), "实际：{}", filtered);
        let suggestions = reasoner
            .association_suggestions("赵六")
            .expect("赵六缺国籍应有关联联想");
        assert!(suggestions.contains("国籍"), "实际：{}", suggestions);
        assert!(suggestions.contains("提升度"));
        assert!(!suggestions.contains("已有特征：国籍"));
    }

    #[test]
    fn test_derived_facts_report_discovers_multi_hop() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let report = reasoner.derived_facts_report(10);
        assert!(report.contains("推理发现"), "实际：{}", report);
        assert!(report.contains("第 1 轮推导"), "应给出推导轮次：{}", report);
        assert!(report.contains("李四") && report.contains("赵六"));
        assert!(report.contains("前提"));
    }

    #[test]
    fn test_continuation_appends_association_suggestions() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&nationality_facts());
        let text = reasoner
            .predict_continuation("赵六的")
            .expect("已知实体应能续写");
        assert!(text.contains("关联规则联想"), "实际：{}", text);
        assert!(text.contains("国籍"));
    }

    #[test]
    fn test_abduce_statement_generates_explained_hypotheses() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        // 张三的师祖是赵六：需要补全 师傅(李四, 赵六) 才能沿规则链证明
        let report = reasoner
            .abduce_statement("张三的师祖是赵六")
            .expect("陈述应可形式化为溯因目标");
        assert!(report.contains("假设生成"), "实际：{}", report);
        assert!(report.contains("失败前沿"), "实际：{}", report);
        assert!(report.contains("师傅") && report.contains("李四") && report.contains("赵六"));
        assert!(report.contains("可信度"));
        let pending = reasoner.pending_hypotheses();
        assert!(!pending.is_empty(), "应暂存候选假设供采纳");
        assert!(pending.iter().all(|hypothesis| hypothesis.atom.is_ground()));
    }

    #[test]
    fn test_adopt_hypothesis_enables_session_proof() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        assert!(
            reasoner.adopt_hypothesis(1).is_none(),
            "无待定假设时应返回 None"
        );
        let _ = reasoner.abduce_statement("张三的师祖是赵六");
        assert!(reasoner
            .resolution_proof_report("张三的师祖是赵六")
            .contains("未得证"));
        let message = reasoner.adopt_hypothesis(1).expect("应能采纳第 1 条假设");
        assert!(message.contains("已采纳假设"), "实际：{}", message);
        // 采纳后：假设进入会话归结事实集，后向链即可证明
        let proof = reasoner.resolution_proof_report("张三的师祖是赵六");
        assert!(proof.contains("得证"), "实际：{}", proof);
        assert!(proof.contains("赵六"));
    }

    #[test]
    fn adopted_hypothesis_is_receipted_and_undoable() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let _ = reasoner.abduce_statement("张三的师祖是赵六");
        let adopted_atom = reasoner.pending_hypotheses()[0].atom.clone();
        let message = reasoner.adopt_hypothesis(1).expect("应能采纳第 1 条假设");
        assert!(message.contains("修订回执"), "采纳应生成回执：{}", message);
        assert!(
            reasoner.session_revision_report().contains("采纳假设 1"),
            "回执应记录假设来源：{}",
            reasoner.session_revision_report()
        );
        assert!(reasoner
            .resolution_proof_report("张三的师祖是赵六")
            .contains("得证"));

        let undone = reasoner.undo_last_session_revision();
        assert!(undone.contains("已撤销会话修订"), "{}", undone);
        assert!(
            reasoner
                .resolution_proof_report("张三的师祖是赵六")
                .contains("未得证"),
            "撤销采纳后不应再得证"
        );
        assert!(
            !reasoner
                .pending_hypotheses()
                .iter()
                .any(|hypothesis| hypothesis.atom == adopted_atom),
            "采纳过的假设不应回到待定池"
        );
    }

    #[test]
    fn conflicting_hypothesis_is_blocked_and_kept_for_reconsideration() {
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
        let hypothesis = Hypothesis {
            atom: Atom::rel("年龄", "张三", "31"),
            goal: Atom::rel("年龄", "张三", "31"),
            kind: HypothesisKind::MissingFact,
            confidence: 0.8,
            reasons: vec!["测试冲突".to_string()],
            conflicts: Vec::new(),
            contradicts_known: false,
        };
        reasoner.abduction.set_pending(vec![hypothesis]);
        let before = reasoner.session_facts().len();
        let blocked = reasoner.adopt_hypothesis(1).expect("应给出拒绝回执");
        assert!(blocked.contains("最小冲突核"), "{}", blocked);
        assert!(
            blocked.contains("30") && blocked.contains("31"),
            "{}",
            blocked
        );
        assert_eq!(reasoner.session_facts().len(), before, "拒绝时零写入");
        assert_eq!(reasoner.pending_hypotheses().len(), 1, "拒绝后假设保留");
        assert!(
            !reasoner.session_revision_report().contains("session:"),
            "拒绝的采纳不得生成回执"
        );
    }

    #[test]
    fn batch_clarification_confirmation_commits_all_and_finishes() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        let pending = vec![
            Hypothesis {
                atom: Atom::rel("颜色", "苹果", "红"),
                goal: Atom::rel("颜色", "苹果", "红"),
                kind: HypothesisKind::MissingFact,
                confidence: 0.8,
                reasons: vec!["测试".to_string()],
                conflicts: Vec::new(),
                contradicts_known: false,
            },
            Hypothesis {
                atom: Atom::rel("颜色", "苹果", "青"),
                goal: Atom::rel("颜色", "苹果", "青"),
                kind: HypothesisKind::MissingFact,
                confidence: 0.6,
                reasons: vec!["测试".to_string()],
                conflicts: Vec::new(),
                contradicts_known: false,
            },
        ];
        reasoner.abduction.set_pending(pending);
        reasoner.start_clarification().expect("应开始澄清");
        let before = reasoner.session_facts().len();
        let output = reasoner
            .answer_clarification("全部确认")
            .expect("应处理批量回答");
        assert!(output.contains("批量确认"), "{}", output);
        assert_eq!(reasoner.session_facts().len(), before + 2);
        assert!(
            !reasoner.clarification_active(),
            "全部确认后会话应结束：{}",
            output
        );
    }

    #[test]
    fn blocked_batch_confirmation_restores_every_question() {
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
        reasoner.abduction.set_pending(vec![
            Hypothesis {
                atom: Atom::rel("年龄", "张三", "31"),
                goal: Atom::rel("年龄", "张三", "31"),
                kind: HypothesisKind::MissingFact,
                confidence: 0.9,
                reasons: vec!["测试冲突".to_string()],
                conflicts: Vec::new(),
                contradicts_known: false,
            },
            Hypothesis {
                atom: Atom::rel("颜色", "苹果", "红"),
                goal: Atom::rel("颜色", "苹果", "红"),
                kind: HypothesisKind::MissingFact,
                confidence: 0.5,
                reasons: vec!["测试".to_string()],
                conflicts: Vec::new(),
                contradicts_known: false,
            },
        ]);
        reasoner.start_clarification().expect("应开始澄清");
        let before = reasoner.session_facts().len();
        let output = reasoner
            .answer_clarification("全部确认")
            .expect("应处理批量回答");
        assert!(output.contains("整批零写入"), "{}", output);
        assert!(
            output.contains("是否成立"),
            "被拒绝后必须重新提问：{}",
            output
        );
        assert_eq!(reasoner.session_facts().len(), before);
        assert!(reasoner.clarification_active(), "整批应恢复为待问");
    }

    #[test]
    fn blocked_clarification_confirmation_reasks_the_same_question() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        let mut registry = PredicateRegistry::new();
        registry.observe("张三", "年龄", "30", None);
        assert!(registry.mark_functional("年龄", true));
        reasoner.configure_laws(&registry);
        reasoner.observe_facts(&[PrologFact::Entity {
            name: "张三".to_string(),
            entity_type: "人".to_string(),
            properties: [("年龄".to_string(), "30".to_string())]
                .into_iter()
                .collect(),
        }]);
        reasoner.abduction.set_pending(vec![Hypothesis {
            atom: Atom::rel("年龄", "张三", "31"),
            goal: Atom::rel("年龄", "张三", "31"),
            kind: HypothesisKind::MissingFact,
            confidence: 0.9,
            reasons: vec!["测试冲突".to_string()],
            conflicts: Vec::new(),
            contradicts_known: false,
        }]);
        reasoner.start_clarification().expect("应开始澄清");
        let output = reasoner.answer_clarification("是").expect("应处理回答");
        assert!(output.contains("最小冲突核"), "{}", output);
        assert!(
            output.contains("是否成立"),
            "写入被阻止后必须再次展示同一个问题：{}",
            output
        );
        assert!(reasoner.clarification_active(), "澄清会话不能静默推进");
    }

    #[test]
    fn clarification_correction_commits_the_alternative() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        let current = Hypothesis {
            atom: Atom::rel("师傅", "张三", "李四"),
            goal: Atom::rel("师傅", "张三", "李四"),
            kind: HypothesisKind::MissingFact,
            confidence: 0.8,
            reasons: vec!["当前假设".to_string()],
            conflicts: Vec::new(),
            contradicts_known: false,
        };
        let alternative = Hypothesis {
            atom: Atom::rel("师傅", "张三", "王五"),
            goal: Atom::rel("师傅", "张三", "王五"),
            kind: HypothesisKind::CandidateValue,
            confidence: 0.5,
            reasons: vec!["候选取值".to_string()],
            conflicts: Vec::new(),
            contradicts_known: false,
        };
        reasoner.abduction.set_pending(vec![current, alternative]);
        let prompt = reasoner.start_clarification().expect("应开始澄清");
        assert!(prompt.contains("李四"), "{}", prompt);

        let output = reasoner
            .answer_clarification("不是，是王五")
            .expect("应处理更正回答");
        assert!(output.contains("更正"), "{}", output);
        assert!(reasoner
            .session_facts()
            .iter()
            .any(|atom| atom == &Atom::rel("师傅", "张三", "王五")));
        assert!(!reasoner
            .session_facts()
            .iter()
            .any(|atom| atom == &Atom::rel("师傅", "张三", "李四")));
    }

    #[test]
    fn abduction_target_covers_check_find_subject_and_count() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        for question in ["赵六的师傅是张三吗", "谁的师祖是赵六", "赵六有几个师傅"]
        {
            let report = reasoner
                .hypothesis_answer(question)
                .unwrap_or_else(|| panic!("「{}」应能进入溯因目标", question));
            assert!(
                report.contains("假设生成")
                    || report.contains("后向链得证")
                    || report.contains("已证明"),
                "「{}」应给出证明或候选解释：{}",
                question,
                report
            );
        }
    }

    #[test]
    fn test_hypothesis_answer_for_missing_link() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        // 赵六没有师傅记录：开放问句应生成候选取值假设
        let report = reasoner
            .hypothesis_answer("赵六的师傅是谁")
            .expect("已知实体+已知谓词应可溯因");
        assert!(report.contains("假设生成"), "实际：{}", report);
        assert!(report.contains("候选假设"), "实际：{}", report);
        assert!(report.contains("师傅"));
        let pending = reasoner.pending_hypotheses();
        assert!(!pending.is_empty());
        let displays: Vec<String> = pending.iter().map(|h| h.atom.display()).collect();
        assert!(
            pending
                .iter()
                .any(|hypothesis| hypothesis.atom.display().contains("师傅")),
            "候选假设应补全师傅关系，实际：{:?}",
            displays
        );
    }

    #[test]
    fn failed_question_enters_clarification_without_dumping_candidates() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&sparse_master_facts());
        reasoner.resolver.add_clause(teacher_rule());
        let prompt = reasoner
            .begin_clarification_for_question("张三的师祖是谁")
            .expect("不可证问句应能直接进入澄清");
        assert!(reasoner.clarification_active());
        assert!(
            prompt.contains("判别理由") && prompt.contains("解释空间"),
            "首问应说明解释空间与判别理由：{}",
            prompt
        );
        assert!(
            !prompt.contains("假设生成"),
            "自动通道不应先把候选清单丢给用户：{}",
            prompt
        );
        // 已在进行中的会话不能被新问题重置：仍返回同一个当前提问。
        let again = reasoner
            .begin_clarification_for_question("钱七的师祖是谁")
            .expect("会话进行中应返回当前提问");
        assert_eq!(again, prompt, "澄清进行中不得被新问题劫持");
    }

    #[test]
    fn provable_question_does_not_enter_clarification() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        assert!(
            reasoner
                .begin_clarification_for_question("张三的师祖是谁")
                .is_none(),
            "已可证问题不应开启澄清"
        );
        assert!(!reasoner.clarification_active());
    }

    #[test]
    fn test_what_if_report_derives_and_rolls_back() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let report = reasoner
            .what_if_report("如果李四的师傅是赵六")
            .expect("条件句应可解析为反事实假设");
        assert!(report.contains("反事实模拟"), "实际：{}", report);
        assert!(report.contains("赵六"), "实际：{}", report);
        assert!(
            report.contains("师祖"),
            "应沿师傅→师祖规则推出新结论：{}",
            report
        );
        // 模拟回滚：原知识库不能被改变
        let proof = reasoner.resolution_proof_report("张三的师祖是赵六");
        assert!(proof.contains("未得证"), "反事实模拟必须回滚：{}", proof);
    }

    #[test]
    fn proved_goal_clears_stale_pending_questions() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let report = reasoner
            .abduce_statement("张三的师祖是赵六")
            .expect("第一次应生成候选");
        assert!(report.contains("假设生成"));
        assert!(!reasoner.pending_hypotheses().is_empty());
        reasoner.observe_facts(&[PrologFact::Relation {
            from: "张三".to_string(),
            rel_type: "师祖".to_string(),
            to: "赵六".to_string(),
        }]);
        let proved = reasoner
            .abduce_statement("张三的师祖是赵六")
            .expect("补全后应可证");
        assert!(proved.contains("得证"), "{}", proved);
        assert!(
            reasoner.pending_hypotheses().is_empty(),
            "目标已证后不得继续保留过期假设"
        );
        assert!(!reasoner.clarification_active());
    }

    #[test]
    fn test_provable_statement_gets_proof_not_hypotheses() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let report = reasoner
            .abduce_statement("张三的师祖是王五")
            .expect("可证陈述应返回证明");
        assert!(report.contains("后向链得证"), "实际：{}", report);
        assert!(report.contains("证明树"));
        assert!(reasoner.pending_hypotheses().is_empty());
    }

    // ---- 生成式联合解释 + 对话式澄清 + 跨会话临时信念 ----

    fn sparse_master_facts() -> Vec<PrologFact> {
        // 只有 李四→王五 一条师傅关系：证明 张三的师祖是赵六 需要同时补两条边
        let mut facts = vec![
            PrologFact::Relation {
                from: "李四".to_string(),
                rel_type: "师傅".to_string(),
                to: "王五".to_string(),
            },
            // 让"师祖"成为已知谓词（解析需要），与目标无关
            PrologFact::Relation {
                from: "钱七".to_string(),
                rel_type: "师祖".to_string(),
                to: "孙八".to_string(),
            },
        ];
        for name in ["张三", "李四", "王五", "赵六", "钱七", "孙八"] {
            facts.push(PrologFact::Entity {
                name: name.to_string(),
                entity_type: "人物".to_string(),
                properties: HashMap::new(),
            });
        }
        facts
    }

    fn teacher_rule() -> Clause {
        Clause::new(
            Atom::pattern("师祖", Term::var("x"), Term::var("y")),
            vec![
                Atom::pattern("师傅", Term::var("x"), Term::var("z")),
                Atom::pattern("师傅", Term::var("z"), Term::var("y")),
            ],
            0.9,
            "测试规则：师傅传递",
        )
    }

    #[test]
    fn test_scenario_report_verifies_joint_explanation() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&sparse_master_facts());
        reasoner.resolver.add_clause(teacher_rule());
        let report = reasoner.scenario_report("张三的师祖是赵六");
        assert!(report.contains("生成式联合解释"), "实际：{}", report);
        assert!(
            report.contains("2 条假设"),
            "应枚举出两条假设的联合场景：{}",
            report
        );
        assert!(
            report.contains("注入后目标可证"),
            "联合场景应通过注入验证：{}",
            report
        );
        assert!(report.contains("验证证明"), "应展示验证证明：{}", report);
        assert!(
            !reasoner.pending_hypotheses().is_empty(),
            "场景成员应进入待定池，供采纳/澄清"
        );
    }

    #[test]
    fn clarification_explains_question_choice_and_collapse() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&sparse_master_facts());
        reasoner.resolver.add_clause(teacher_rule());
        let _ = reasoner.scenario_report("张三的师祖是赵六");
        let prompt = reasoner.start_clarification().expect("应开始澄清");
        assert!(
            prompt.contains("解释空间") && prompt.contains("判别理由"),
            "澄清提问应说明解释空间与判别理由：{}",
            prompt
        );

        let mut last = String::new();
        for _ in 0..10 {
            if !reasoner.clarification_active() {
                break;
            }
            last = reasoner
                .answer_clarification("不是")
                .expect("应处理否定回答");
        }
        assert!(
            last.contains("解释空间崩塌"),
            "连续否定应诊断解释空间崩塌：{}",
            last
        );
        assert!(
            last.contains("不是“知识库查无”"),
            "崩塌不能误报为查无：{}",
            last
        );
        assert!(last.contains("缺少的前提"), "崩塌应给出缺失前提：{}", last);
    }

    #[test]
    fn clarification_skips_questions_when_goal_is_already_proved() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let _ = reasoner.abduce_statement("张三的师祖是赵六");
        assert!(!reasoner.pending_hypotheses().is_empty());
        // 模拟用户在开始澄清前补上了缺失事实。
        reasoner.adopt_session_triple("李四", "师傅", "赵六");
        let output = reasoner
            .start_clarification()
            .expect("应给出无需澄清的说明");
        assert!(output.contains("已可证明"), "{}", output);
        assert!(output.contains("无需继续追问"), "{}", output);
        assert!(reasoner.pending_hypotheses().is_empty(), "过期假设应清空");
        assert!(!reasoner.clarification_active());
    }

    #[test]
    fn clarification_that_proves_goal_does_not_report_collapse() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&sparse_master_facts());
        reasoner.resolver.add_clause(teacher_rule());
        let _ = reasoner.scenario_report("张三的师祖是赵六");
        reasoner.start_clarification().expect("应开始澄清");
        let output = reasoner
            .answer_clarification("全部确认")
            .expect("应处理批量确认");
        assert!(output.contains("已可证明"), "{}", output);
        assert!(
            !output.contains("解释空间崩塌"),
            "目标得证时空解释空间只是“无需解释”，不能误报崩塌：{}",
            output
        );
        assert!(!reasoner.clarification_active());
    }

    #[test]
    fn clarification_stops_after_question_budget() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        let pending: Vec<Hypothesis> = (0..10)
            .map(|index| Hypothesis {
                atom: Atom::rel("颜色", "对象", &format!("取值{}", index)),
                goal: Atom::rel("颜色", "对象", &format!("取值{}", index)),
                kind: HypothesisKind::MissingFact,
                confidence: 0.9 - index as f64 * 0.01,
                reasons: vec!["测试".to_string()],
                conflicts: Vec::new(),
                contradicts_known: false,
            })
            .collect();
        reasoner.abduction.set_pending(pending);
        reasoner.start_clarification().expect("应开始澄清");
        let mut last = String::new();
        for _ in 0..12 {
            if !reasoner.clarification_active() {
                break;
            }
            last = reasoner
                .answer_clarification("不知道")
                .expect("应处理暂缓回答");
        }
        assert!(
            last.contains("最大追问数"),
            "达到预算后应停止追问并给结论：{}",
            last
        );
        assert!(!reasoner.clarification_active());
    }

    #[test]
    fn clarification_finishes_with_conditional_answers() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&sparse_master_facts());
        reasoner.resolver.add_clause(teacher_rule());
        let report = reasoner.scenario_report("张三的师祖是谁");
        assert!(report.contains("生成式联合解释"), "{}", report);
        reasoner.start_clarification().expect("应开始澄清");
        let conclusion = reasoner.stop_clarification().expect("应结束澄清");
        assert!(
            conclusion.contains("条件化回答") || conclusion.contains("解释空间收敛"),
            "提前结束应给出条件化或稳健答案：{}",
            conclusion
        );
        assert!(
            conclusion.contains("若") && conclusion.contains("则答案是"),
            "{}",
            conclusion
        );
    }

    #[test]
    fn clarification_snapshot_recomputes_explanation_space() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&sparse_master_facts());
        reasoner.resolver.add_clause(teacher_rule());
        let _ = reasoner.scenario_report("张三的师祖是赵六");
        let prompt = reasoner.start_clarification().expect("应开始澄清");
        assert!(prompt.contains("解释空间"), "{}", prompt);
        let snapshot = reasoner.dialogue_snapshot();

        let mut restored = SymbolicReasoner::new(test_config(), HashMap::new());
        restored.observe_facts(&sparse_master_facts());
        restored.resolver.add_clause(teacher_rule());
        restored
            .restore_dialogue_snapshot(&snapshot)
            .expect("快照应可恢复");
        assert!(restored.clarification_active(), "澄清会话应继续");
        assert_eq!(
            restored.clarification_progress(),
            reasoner.clarification_progress(),
            "恢复后解释空间应派生重算并与原会话一致"
        );
        let restored_prompt = restored
            .start_clarification()
            .expect("可显式重开以复核解释空间");
        assert!(
            restored_prompt.contains("解释空间"),
            "重开后的提问仍应带解释空间：{}",
            restored_prompt
        );
    }

    #[test]
    fn test_clarification_confirms_hypothesis_and_proves_goal() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let report = reasoner
            .abduce_statement("张三的师祖是赵六")
            .expect("应能溯因");
        assert!(report.contains("假设生成"), "实际：{}", report);
        let question = reasoner.start_clarification().expect("有待定假设即可澄清");
        assert!(
            question.contains("请确认") || question.contains("是否成立"),
            "实际：{}",
            question
        );
        let response = reasoner.answer_clarification("是的").expect("应处理确认");
        assert!(response.contains("已确认假设"), "实际：{}", response);
        assert!(
            response.contains("已可证明"),
            "确认后目标应可证明：{}",
            response
        );
        assert!(!reasoner.clarification_active(), "目标得证后澄清应结束");
        let proof = reasoner.resolution_proof_report("张三的师祖是赵六");
        assert!(proof.contains("得证"), "实际：{}", proof);
        assert_eq!(reasoner.session_facts().len(), 1);
    }

    #[test]
    fn test_clarification_reject_and_defer_flow() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let _ = reasoner
            .abduce_statement("张三的师祖是赵六")
            .expect("应能溯因");
        reasoner.start_clarification().expect("应能开始澄清");
        let rejected = reasoner.answer_clarification("不是").expect("应处理否定");
        assert!(rejected.contains("已排除假设"), "实际：{}", rejected);
        if reasoner.clarification_active() {
            let deferred = reasoner
                .answer_clarification("不确定")
                .expect("应处理不确定");
            assert!(
                deferred.contains("暂记为不确定") || deferred.contains("澄清结束"),
                "实际：{}",
                deferred
            );
        }
        assert!(
            reasoner.session_facts().is_empty(),
            "否定/不确定都不应产生会话事实"
        );
    }

    #[test]
    fn test_session_facts_persist_across_snapshot() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let _ = reasoner.abduce_statement("张三的师祖是赵六");
        let message = reasoner.adopt_hypothesis(1).expect("应能采纳");
        assert!(message.contains("已采纳假设"), "实际：{}", message);
        let path =
            std::env::temp_dir().join(format!("knowprolog_session_{}.json", std::process::id()));
        let path_str = path.to_string_lossy().to_string();
        reasoner.save_dialogue_state(&path_str).expect("应能保存");

        let mut restored = SymbolicReasoner::new(test_config(), HashMap::new());
        restored.observe_facts(&master_facts());
        assert!(restored.load_dialogue_state(&path_str), "应能恢复对话状态");
        let proof = restored.resolution_proof_report("张三的师祖是赵六");
        assert!(
            proof.contains("得证"),
            "跨会话恢复的假设应参与证明：{}",
            proof
        );
        let soft = restored.soft_inference_report("张三");
        assert!(
            soft.contains("师祖") && soft.contains("赵六"),
            "跨会话恢复的假设应注入软逻辑：{}",
            soft
        );
        let problog = restored.problog_report("张三");
        assert!(
            problog.contains("师祖") && problog.contains("赵六"),
            "跨会话恢复的假设应注入概率程序：{}",
            problog
        );
        assert!(
            restored.dialogue_summary().contains("会话假设 1 条"),
            "对话记忆应显示会话假设：{}",
            restored.dialogue_summary()
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn pending_hypotheses_and_clarification_survive_snapshot() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let empty =
            String::from_utf8(canonical::canonical_bytes(&reasoner.dialogue_snapshot()).unwrap())
                .unwrap();
        assert!(
            !empty.contains("pending_hypotheses") && !empty.contains("last_abduction_goal"),
            "空会话不应把溯因字段写进快照：{}",
            empty
        );

        let _ = reasoner
            .abduce_statement("张三的师祖是赵六")
            .expect("应能溯因");
        let pending_before: Vec<String> = reasoner
            .pending_hypotheses()
            .iter()
            .map(|hypothesis| hypothesis.atom.display())
            .collect();
        assert!(!pending_before.is_empty());
        let question = reasoner.start_clarification().expect("应能开始澄清");
        assert!(reasoner.clarification_active());

        let snapshot = reasoner.dialogue_snapshot();
        assert!(!snapshot.pending_hypotheses.is_empty());
        assert!(snapshot.clarification.is_some());

        let mut restored = SymbolicReasoner::new(test_config(), HashMap::new());
        restored.observe_facts(&master_facts());
        restored
            .restore_dialogue_snapshot(&snapshot)
            .expect("快照应可恢复");
        let pending_after: Vec<String> = restored
            .pending_hypotheses()
            .iter()
            .map(|hypothesis| hypothesis.atom.display())
            .collect();
        assert_eq!(pending_before, pending_after, "待定假设应按原序恢复");
        assert!(restored.clarification_active(), "澄清会话应跨快照继续");
        assert_eq!(
            restored.clarification_progress().expect("澄清进度应存在"),
            reasoner.clarification_progress().expect("澄清进度应存在")
        );

        let response = restored
            .answer_clarification("是的")
            .expect("恢复后的澄清应可作答");
        assert!(response.contains("已确认假设"), "{}", response);
        assert!(
            restored
                .resolution_proof_report("张三的师祖是赵六")
                .contains("得证"),
            "恢复的澄清确认应能证出原目标"
        );
        assert!(
            question.contains("请确认") || question.contains("是否成立"),
            "{}",
            question
        );
    }

    #[test]
    fn test_session_facts_survive_resolver_rebuild() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let _ = reasoner.abduce_statement("张三的师祖是赵六");
        reasoner.adopt_hypothesis(1).expect("应能采纳");
        assert!(reasoner
            .resolution_proof_report("张三的师祖是赵六")
            .contains("得证"));
        // 再观察新关系会触发归结引擎重建：会话事实必须在重建后重新注入
        reasoner.observe_facts(&[PrologFact::Relation {
            from: "赵六".to_string(),
            rel_type: "师傅".to_string(),
            to: "钱七".to_string(),
        }]);
        let proof = reasoner.resolution_proof_report("张三的师祖是赵六");
        assert!(
            proof.contains("得证"),
            "重建后采纳的假设不应丢失：{}",
            proof
        );
    }

    #[test]
    fn test_structural_related_report_uses_random_walk() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let report = reasoner.structural_related_report("张三");
        assert!(report.contains("随机游走重启"), "实际：{}", report);
        assert!(
            report.contains("李四"),
            "直接邻居应在结构联想中：{}",
            report
        );
        assert!(report.contains("游走驻留概率"));
    }

    #[test]
    fn test_topic_extension_uses_dialogue_focus() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        assert!(
            reasoner.topic_extension_report().contains("还没有对话焦点"),
            "无焦点时应给出提示"
        );
        let _ = reasoner.contextualize("张三的师傅是谁");
        let report = reasoner.topic_extension_report();
        assert!(report.contains("话题延伸"), "实际：{}", report);
        assert!(
            report.contains("张三"),
            "焦点实体应出现在报告中：{}",
            report
        );
        assert!(report.contains("李四"), "焦点邻居应被延伸出来：{}", report);
    }

    #[test]
    fn test_soft_inference_report_gives_probability_and_evidence() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let known = reasoner.soft_inference_report("张三的师祖");
        assert!(known.contains("王五"), "实际：{}", known);
        assert!(known.contains("P=1.00"), "图谱事实概率应为 1：{}", known);
        let inferred = reasoner.soft_inference_report("李四的师祖");
        assert!(inferred.contains("赵六"), "实际：{}", inferred);
        assert!(inferred.contains("依据"), "软推理应给出证据：{}", inferred);
        assert!(inferred.contains("P="), "软推理应给出概率：{}", inferred);
    }

    #[test]
    fn test_soft_facts_report_lists_weighted_rules() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let report = reasoner.soft_facts_report(5);
        assert!(report.contains("加权软推理"), "实际：{}", report);
        assert!(report.contains("加权规则"), "实际：{}", report);
    }

    #[test]
    fn test_graph_walk_and_soft_logic_respect_disabled_config() {
        let mut config = test_config();
        config.enable_graph_walk = false;
        config.enable_soft_logic = false;
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        reasoner.observe_facts(&master_facts());
        assert!(reasoner
            .structural_related_report("张三")
            .contains("未启用"));
        assert!(reasoner.topic_extension_report().contains("未启用"));
        assert!(reasoner.soft_inference_report("张三").contains("未启用"));
        assert!(reasoner.soft_facts_report(5).contains("未启用"));
    }

    #[test]
    fn test_session_assumption_feeds_soft_inference() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let _ = reasoner.abduce_statement("张三的师祖是赵六");
        reasoner.adopt_hypothesis(1).expect("应能采纳假设");
        let report = reasoner.soft_inference_report("张三的师祖");
        assert!(
            report.contains("赵六"),
            "采纳的会话假设应进入软推理：{}",
            report
        );
    }

    #[test]
    fn test_causal_report_explains_soft_prediction() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let report = reasoner.causal_report("李四的师祖");
        assert!(report.contains("因果推理"), "实际：{}", report);
        assert!(report.contains("赵六"), "应解释推出的师祖：{}", report);
        assert!(report.contains("直接原因"), "应列出直接原因：{}", report);
        assert!(report.contains("依据："), "应给出规则来源：{}", report);
        assert!(
            report.contains("事实根因链") && report.contains("—["),
            "应给出可回放的因果链：{}",
            report
        );
        // 没有可解释目标时优雅降级
        assert!(reasoner.causal_report("不存在的人").contains("无法"));
    }

    #[test]
    fn test_counterfactual_report_quantifies_do_effect() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let report = reasoner.counterfactual_report("李四的师傅", false);
        assert!(report.contains("反事实推理"), "实际：{}", report);
        assert!(report.contains("do"), "应说明干预语义：{}", report);
        assert!(report.contains("师傅"), "应指出被干预的关系：{}", report);
        assert!(report.contains("Δ="), "应给出概率差：{}", report);
        assert!(
            report.contains("下降"),
            "移除师傅应使下游加权推测下降：{}",
            report
        );
        let boosted = reasoner.counterfactual_report("李四的师傅", true);
        assert!(
            boosted.contains("不变") || boosted.contains("上升"),
            "置真干预不应使下游下降：{}",
            boosted
        );
    }

    #[test]
    fn test_gradient_report_shows_symbolic_derivatives() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let report = reasoner.gradient_report("李四的师祖");
        assert!(report.contains("符号梯度归因"), "实际：{}", report);
        assert!(report.contains("∂P/∂w"), "应展示解析梯度：{}", report);
        assert!(
            report.contains("规则贡献"),
            "应逐条列出规则贡献：{}",
            report
        );
        assert!(report.contains("权重"), "应展示规则权重：{}", report);
        assert!(report.contains("来源："), "应展示规则来源：{}", report);
        assert!(report.contains("P="), "应展示概率：{}", report);
        // 未知实体时优雅提示
        assert!(reasoner.gradient_report("钱七").contains("无法"));
    }

    #[test]
    fn test_causal_and_gradient_respect_disabled_config() {
        let mut config = test_config();
        config.enable_causal_reasoning = false;
        config.enable_gradient_attribution = false;
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        reasoner.observe_facts(&master_facts());
        assert!(reasoner.causal_report("李四的师祖").contains("未启用"));
        assert!(reasoner
            .counterfactual_report("张三的师傅", false)
            .contains("未启用"));
        assert!(reasoner.gradient_report("李四的师祖").contains("未启用"));
    }

    #[test]
    fn test_causal_tokenize_report_segments_with_alias_and_attributes() {
        let synonyms = HashMap::from([("师父".to_string(), "师傅".to_string())]);
        let mut reasoner = SymbolicReasoner::new(test_config(), synonyms);
        reasoner.observe_facts(&master_facts());
        let report = reasoner.causal_tokenize_report("张三的师父是谁");
        assert!(report.contains("因果分词"), "实际：{}", report);
        assert!(report.contains("最优分词"), "实际：{}", report);
        assert!(
            report.contains("「师傅」（别名）"),
            "同义词「师父」应锚定到规范谓词「师傅」：{}",
            report
        );
        assert!(report.contains("token 因果归因"), "实际：{}", report);
        assert!(
            report.contains("张三"),
            "输入实体应出现在归因中：{}",
            report
        );
        assert!(
            report.contains("纯符号 attention"),
            "应声明归因是纯符号可回放：{}",
            report
        );
    }

    #[test]
    fn test_causal_tokenization_respects_disabled_config() {
        let mut config = test_config();
        config.enable_causal_tokenization = false;
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        reasoner.observe_facts(&master_facts());
        assert!(reasoner
            .causal_tokenize_report("张三的师傅是谁")
            .contains("未启用"));
    }

    #[test]
    fn test_intervention_boosts_downstream_probability() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        // 干预一个"推导出来的中间节点"（李四的师傅=王五）为真：
        // 王五被钉真后，李四的师祖（赵六）概率应不低于事实世界
        let report = reasoner.counterfactual_report("李四的师傅", true);
        assert!(report.contains("师傅"), "实际：{}", report);
        assert!(report.contains("反事实推理"));
        assert!(
            report.contains("不变") || report.contains("上升"),
            "干预中间节点置真不应降低下游：{}",
            report
        );
    }

    #[test]
    fn test_problog_report_gives_exact_probability_and_proof_paths() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let report = reasoner.problog_report("李四的师祖");
        assert!(report.contains("可能世界精确推理"), "实际：{}", report);
        assert!(report.contains("赵六"), "应推出赵六：{}", report);
        assert!(report.contains("证明路径"), "应给出证明路径：{}", report);
        assert!(
            report.contains("与软逻辑"),
            "应给出与 MLN-lite 的对照：{}",
            report
        );
        assert!(
            report.contains("P=0.") || report.contains("P=1."),
            "应给出概率：{}",
            report
        );
        assert!(reasoner.problog_report("不存在的人").contains("无法"));
    }

    #[test]
    fn test_teach_chinese_conditional_rule_affects_problog() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let taught = reasoner.teach_report("如果张三的师傅是李四，那么张三的师祖是赵六");
        assert!(taught.contains("逻辑形式"), "实际：{}", taught);
        assert!(taught.contains("中文教学·条件句"), "应标注来源：{}", taught);
        assert!(taught.contains("95%"), "默认条件句概率应为 95%：{}", taught);
        let report = reasoner.problog_report("张三的师祖");
        assert!(report.contains("赵六"), "教学规则应推出赵六：{}", report);
        assert!(
            report.contains("中文教学"),
            "证明路径应保留教学来源：{}",
            report
        );
        let again = reasoner.teach_report("如果张三的师傅是李四，那么张三的师祖是赵六");
        assert!(again.contains("已经学过"), "重复教学应被识别：{}", again);
    }

    #[test]
    fn test_teach_probabilistic_fact_percentage_query() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let taught = reasoner.teach_report("李四的喜欢是赵六的可能性是百分之八十");
        assert!(taught.contains("概率事实"), "实际：{}", taught);
        assert!(taught.contains("80%"), "应为百分比概率：{}", taught);
        let report = reasoner.problog_report("李四的喜欢");
        assert!(
            report.contains("P=0.800"),
            "应给出精确概率 0.800：{}",
            report
        );
        assert!(report.contains("概率事实"), "应说明来源：{}", report);
    }

    #[test]
    fn test_problog_and_teaching_respect_disabled_config() {
        let mut config = test_config();
        config.enable_problog = false;
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        reasoner.observe_facts(&master_facts());
        assert!(reasoner.problog_report("李四的师祖").contains("未启用"));
        assert!(reasoner.teach_report("八成的鸟会飞").contains("未启用"));
    }

    #[test]
    fn test_problog_lessons_persist_across_snapshot() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.teach_report("李四的喜欢是赵六的可能性是百分之八十");
        let path =
            std::env::temp_dir().join(format!("knowprolog_lessons_{}.json", std::process::id()));
        let path_str = path.to_string_lossy().to_string();
        reasoner.save_dialogue_state(&path_str).expect("应能保存");

        let mut restored = SymbolicReasoner::new(test_config(), HashMap::new());
        restored.observe_facts(&master_facts());
        assert!(restored.load_dialogue_state(&path_str), "应能恢复对话状态");
        assert_eq!(restored.lessons().len(), 1, "教学记忆应跨会话恢复");
        let report = restored.problog_report("李四的喜欢");
        assert!(
            report.contains("P=0.800"),
            "恢复后应重放概率知识：{}",
            report
        );
        assert!(
            restored.dialogue_summary().contains("中文概率知识 1 条"),
            "对话记忆应显示教学记忆：{}",
            restored.dialogue_summary()
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_problog_query_accepts_taught_entity_outside_graph() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.teach_report("八成的鸟会飞");
        let report = reasoner.problog_report("鸟");
        assert!(report.contains("会飞"), "教学事实应可查询：{}", report);
        assert!(report.contains("P=0.800"), "应给出精确概率：{}", report);
        assert!(report.contains("概率事实"), "应说明来源：{}", report);
    }

    #[test]
    fn test_default_logic_applies_and_blocks_with_exception() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.assert_nl_fact("麻雀是鸟");
        reasoner.assert_nl_fact("蒂娜是鸟");
        reasoner.assert_nl_fact("蒂娜是企鹅");
        let taught = reasoner.teach_report("如果X是鸟，那么X会飞，除非X是企鹅");
        assert!(taught.contains("默认规则"), "应识别默认规则：{}", taught);
        assert!(taught.contains("除非"), "应展示例外：{}", taught);

        let applied = reasoner.default_logic_report("麻雀会飞");
        assert!(
            applied.contains("成立（默认推理触发）"),
            "实际：{}",
            applied
        );
        assert!(applied.contains("通常推出"), "应给出触发默认：{}", applied);
        assert!(
            applied.contains("P=0.900"),
            "应给出概率视图对照：{}",
            applied
        );

        let blocked = reasoner.default_logic_report("蒂娜会飞");
        assert!(
            blocked.contains("未成立（被默认阻断）"),
            "实际：{}",
            blocked
        );
        assert!(blocked.contains("例外"), "应说明例外：{}", blocked);
        assert!(
            blocked.contains("蒂娜是企鹅"),
            "应点名例外事实：{}",
            blocked
        );
    }

    #[test]
    fn test_default_logic_reports_missing_evidence_and_unformalizable() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let unknown = reasoner.default_logic_report("阿猫会飞");
        assert!(unknown.contains("未找到证据"), "实际：{}", unknown);
        let unformalizable = reasoner.default_logic_report("今天天气怎么样");
        assert!(
            unformalizable.contains("无法把"),
            "应提示无法形式化：{}",
            unformalizable
        );
    }

    #[test]
    fn test_default_logic_supplement_only_fires_with_defaults() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.assert_nl_fact("麻雀是鸟");
        reasoner.assert_nl_fact("蒂娜是鸟");
        reasoner.assert_nl_fact("蒂娜是企鹅");
        assert!(
            reasoner.default_logic_supplement("麻雀会飞").is_none(),
            "没有默认规则时不应产生补充"
        );
        reasoner.teach_report("如果X是鸟，那么X会飞，除非X是企鹅");
        let supplement = reasoner
            .default_logic_supplement("麻雀会飞")
            .expect("默认规则就位后应给出补充");
        assert!(supplement.contains("通常成立"), "实际：{}", supplement);
        let question = reasoner
            .default_logic_supplement("请问麻雀会飞吗")
            .expect("疑问句形式也应给出补充");
        assert!(question.contains("通常成立"), "实际：{}", question);
        let blocked = reasoner
            .default_logic_supplement("蒂娜会飞吗")
            .expect("例外应给出阻断补充");
        assert!(blocked.contains("不能推出"), "实际：{}", blocked);
        assert!(
            reasoner
                .default_logic_supplement("今天天气怎么样")
                .is_none(),
            "无法形式化时不应产生补充"
        );
    }

    #[test]
    fn test_default_lessons_persist_across_snapshot() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.assert_nl_fact("麻雀是鸟");
        reasoner.teach_report("如果X是鸟，那么X会飞，除非X是企鹅");
        let path =
            std::env::temp_dir().join(format!("knowprolog_defaults_{}.json", std::process::id()));
        let path_str = path.to_string_lossy().to_string();
        reasoner.save_dialogue_state(&path_str).expect("应能保存");

        let mut restored = SymbolicReasoner::new(test_config(), HashMap::new());
        restored.observe_facts(&master_facts());
        assert!(restored.load_dialogue_state(&path_str), "应能恢复对话状态");
        let report = restored.default_logic_report("麻雀会飞");
        assert!(
            report.contains("成立（默认推理触发）"),
            "恢复后默认规则与会话事实应重放生效：{}",
            report
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_learn_general_rule_enables_backward_chain() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let taught = reasoner.learn_rule_report("如果X的师傅是Y，那么X的老师是Y");
        assert!(
            taught.contains("通用规则"),
            "应识别为含变量的通用规则：{}",
            taught
        );
        assert!(taught.contains("老师"), "应展示逻辑形式：{}", taught);
        // 判断走语义蕴含；图谱里没有"老师"事实时由后向链接管
        let judged = reasoner.entailment_report("李四的老师是王五");
        assert!(judged.contains("后向链补充判定"), "实际：{}", judged);
        assert!(judged.contains("王五"), "证明树应落到王五：{}", judged);
        // 一般规则可套用到任意实体：张三的师傅是李四
        let queried = reasoner.resolution_query_report("张三的老师是谁");
        assert!(queried.contains("答案是「李四」"), "实际：{}", queried);
        let proof = reasoner.resolution_proof_report("李四的老师是王五");
        assert!(proof.contains("得证"), "实际：{}", proof);
        assert!(
            proof.contains("中文教学"),
            "证明树应标出规则来源：{}",
            proof
        );
    }

    #[test]
    fn test_learn_ground_rule_joins_backward_chain() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let taught = reasoner.learn_rule_report("如果张三的师傅是李四，那么张三的师祖是赵六");
        assert!(taught.contains("具体规则"), "应识别为具体规则：{}", taught);
        let proof = reasoner.resolution_proof_report("张三的师祖是赵六");
        assert!(
            proof.contains("得证"),
            "教过的具体规则应参与后向链：{}",
            proof
        );
    }

    #[test]
    fn test_learn_rule_report_rejects_non_rules_gracefully() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let report = reasoner.learn_rule_report("张三是李四的师傅");
        assert!(
            report.contains("没能把这句话解析成规则"),
            "实际：{}",
            report
        );
        let empty = reasoner.learn_rule_report("   ");
        assert!(empty.contains("请给出中文规则"), "实际：{}", empty);
    }

    #[test]
    fn test_chain_answer_carries_witness_paths() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let answer = reasoner
            .answer("张三的师傅的师傅是谁")
            .expect("多跳问句应给出路径回答");
        assert!(answer.contains("路径"), "实际：{}", answer);
        assert!(
            answer.contains("张三 --师傅--> 李四 --师傅--> 王五"),
            "路径应保留每一跳：{}",
            answer
        );
        assert!(
            answer.matches("推理依据").count() <= 1,
            "推理依据不应重复：{}",
            answer
        );
    }

    #[test]
    fn test_constructive_chain_fills_missing_hop() {
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
            PrologFact::Relation {
                from: "吴九".to_string(),
                rel_type: "师傅".to_string(),
                to: "郑十".to_string(),
            },
        ];
        reasoner.observe_facts(&facts);
        reasoner.config.enable_link_prediction = false;
        reasoner.config.constructive.enabled = true;
        let answer = reasoner
            .answer("孙七的朋友的师傅是谁")
            .expect("缺中间边的多跳问句应能给出建设性推断");
        assert!(answer.contains("郑十"), "实际：{}", answer);
        assert!(answer.contains("建设性推断"), "实际：{}", answer);
        assert!(answer.contains("未写入知识图谱"), "实际：{}", answer);
        assert!(
            answer.contains("同事(A,B) ∧ 师傅(B,C) ⇒ 朋友(A,C)"),
            "规则来源应可追溯：{}",
            answer
        );
        assert!(
            answer.contains("确认推断：孙七|朋友|吴九"),
            "应给出可直接复制的确认命令：{}",
            answer
        );
    }

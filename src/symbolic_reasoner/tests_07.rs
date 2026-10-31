    #[test]
    fn epistemic_most_likely_lists_each_candidate_once() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&[
            PrologFact::Relation {
                from: "李四".to_string(),
                rel_type: "师傅".to_string(),
                to: "张三".to_string(),
            },
            PrologFact::Relation {
                from: "李四".to_string(),
                rel_type: "师傅".to_string(),
                to: "王五".to_string(),
            },
        ]);
        let answer = reasoner
            .answer("谁最可能是李四的师傅")
            .expect("最可能问应可排序");
        assert_eq!(
            answer.matches("P=1.000").count(),
            2,
            "两个候选应各出现一次（最可能 + 次选），不能重复填充：{}",
            answer
        );
        assert_eq!(answer.matches("次选：").count(), 1, "{}", answer);
    }

    #[test]
    fn epistemic_comparison_accepts_gen_separator() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let answer = reasoner
            .answer("李四跟王五谁更可能是张三的师傅")
            .expect("「跟」也应读作比较");
        assert!(answer.contains("比较结论"), "{}", answer);
        assert!(
            answer.contains("更可能是「张三 的 师傅 是 李四"),
            "{}",
            answer
        );
    }

    #[test]
    fn epistemic_most_likely_compares_two_named_candidates() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let answer = reasoner
            .answer("李四和王五谁最可能是张三的师傅")
            .expect("两个具名候选应做比较");
        assert!(answer.contains("比较结论"), "{}", answer);
        assert!(
            answer.contains("更可能是「张三 的 师傅 是 李四"),
            "{}",
            answer
        );
        assert!(answer.contains("王五"), "{}", answer);
    }

    #[test]
    fn epistemic_followup_keeps_modality_across_turns() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let mut dialogue = crate::dialogue_manager::DialogueManager::new();
        let first = dialogue
            .plan_pure_dialogue("李四的师傅可能是谁", &reasoner)
            .or_else(|| dialogue.plan_reply("李四的师傅可能是谁", &reasoner))
            .map(|reply| reply.text)
            .or_else(|| reasoner.answer("李四的师傅可能是谁"))
            .expect("首轮情态问句应可回答");
        assert!(first.contains("PQI"), "{}", first);
        let follow = dialogue
            .plan_pure_dialogue("那王五呢", &reasoner)
            .or_else(|| dialogue.plan_reply("那王五呢", &reasoner))
            .map(|reply| reply.text)
            .or_else(|| reasoner.answer("那王五呢"))
            .expect("省略追问应承接情态");
        assert!(follow.contains("PQI"), "{}", follow);
        assert!(follow.contains("知识概率"), "{}", follow);
        assert!(follow.contains("赵六"), "王五的师傅应为赵六：{}", follow);
    }


    #[test]
    fn foil_guards_filter_predictions_and_survive_snapshot() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let before = reasoner.predictions_for_entity("李四", Some("师祖"), 8);
        let target = before.first().expect("应至少有一条链接预测").clone();
        let rule = reasoner
            .predictor
            .rules()
            .get(target.rule_index)
            .expect("预测应引用有效规则");
        let refinement = crate::foil_refinement::RuleRefinement {
            rule_index: target.rule_index,
            rule_text: rule.rule_text(),
            hard_negatives: vec![target.triple.clone()],
            unmatched_negatives: Vec::new(),
            positives: Vec::new(),
            evaluations: Vec::new(),
            guards: vec![crate::foil_refinement::RuleGuard::ExceptInstance {
                subject: target.triple.subject.clone(),
                predicate: target.triple.predicate.clone(),
                object: target.triple.object.clone(),
            }],
            blocked: 1,
            status: crate::foil_refinement::RefinementStatus::Refined,
            note: "测试守卫".to_string(),
        };
        let mut ledger = FoilLedger::new();
        assert!(ledger.adopt_refinement(&refinement));
        reasoner.set_foil_ledger(ledger);
        let after = reasoner.predictions_for_entity("李四", Some("师祖"), 8);
        assert!(!after.iter().any(|prediction| prediction.triple == target.triple));

        let snapshot = reasoner.dialogue_snapshot();
        assert!(snapshot.foil.is_some());
        let mut restored = SymbolicReasoner::new(test_config(), HashMap::new());
        restored.observe_facts(&master_facts());
        restored
            .restore_dialogue_snapshot(&snapshot)
            .expect("反例账本应可恢复");
        let after_restore = restored.predictions_for_entity("李四", Some("师祖"), 8);
        assert!(!after_restore
            .iter()
            .any(|prediction| prediction.triple == target.triple));
    }

    fn alignment_facts() -> Vec<PrologFact> {
        vec![
            PrologFact::Entity {
                name: "未名科技".to_string(),
                entity_type: "组织".to_string(),
                properties: HashMap::new(),
            },
            PrologFact::Entity {
                name: "远山科技".to_string(),
                entity_type: "组织".to_string(),
                properties: HashMap::new(),
            },
            PrologFact::Entity {
                name: "张三".to_string(),
                entity_type: "人物".to_string(),
                properties: HashMap::new(),
            },
        ]
    }

    #[test]
    fn alignment_memory_teaches_and_links_unseen_surface() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&alignment_facts());
        let receipt = reasoner.learn_entity_alignment("未名科技公司", "未名科技");
        assert!(receipt.contains("已记住"), "{}", receipt);
        assert!(reasoner.alignment_memory_report().contains("未名科技公司"));

        let mentions = reasoner
            .linker()
            .alignment_mentions("远山科技公司发布了新芯片");
        assert_eq!(mentions.len(), 1);
        assert_eq!(mentions[0].mention, "远山科技公司");
        assert_eq!(mentions[0].best().unwrap().entity, "远山科技");
        assert_eq!(mentions[0].best().unwrap().method, LinkMethod::Alignment);

        let (surface, reason) = reasoner
            .alignment_surface_choice("未名科技")
            .expect("学过的规范身份应能反向选面");
        assert_eq!(surface, "未名科技公司");
        assert!(reason.contains("回声"));
        let rendered = reasoner.render_atom_nl(&Atom::rel("师傅", "未名科技", "张三"));
        assert!(
            rendered.contains("未名科技公司"),
            "生成端应沿用用户词面：{}",
            rendered
        );

        assert!(
            reasoner.alignment_canonicalize("张三公司").is_none(),
            "组织后缀不得套到人物身上，除非同形规则另有类型锚点"
        );
    }

    #[test]
    fn mention_lexicon_grows_from_facts_and_segments_long_entities() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&alignment_facts());
        let entry = reasoner
            .mention_lexicon()
            .entry("未名科技")
            .expect("图谱实体应回灌提及词表");
        assert_eq!(entry.pos, "nt");
        assert!(reasoner.mention_lexicon_report().contains("未名科技"));

        let words = reasoner.segment_with_lexicon("未名科技发布了新芯片", &trace_nlp_config());
        let entity = words
            .iter()
            .find(|word| word.text == "未名科技")
            .expect("长实体应保留为单一 token");
        assert_eq!(entity.pos, "nt");
    }

    #[test]
    fn confirmed_alias_enters_lexicon_and_tombstone_survives_reobservation() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&alignment_facts());
        assert!(reasoner.register_entity_alias("小巨人", "张三"));
        let alias = reasoner
            .mention_lexicon()
            .entry("小巨人")
            .expect("确认别名应进入提及词表");
        assert_eq!(alias.canonical, "张三");
        assert_eq!(alias.pos, "nr");

        assert!(reasoner.reject_mention("未名科技"));
        reasoner.observe_facts(&alignment_facts());
        let entry = reasoner
            .mention_lexicon()
            .entry("未名科技")
            .expect("墓碑条目应保留");
        assert_eq!(
            entry.status,
            crate::mention_lexicon::LexiconStatus::Tombstoned
        );
        assert!(reasoner
            .mention_lexicon()
            .confirmed_spans("未名科技发布了新芯片")
            .is_empty());
    }

    #[test]
    fn rebuild_preserves_mention_lexicon_decisions() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        assert!(reasoner.nominate_mention("未名科技", None));
        assert!(reasoner.reject_mention("坏词"));
        reasoner.rebuild_from_graph(&[]);
        assert!(
            reasoner
                .mention_lexicon()
                .entry("未名科技")
                .is_some_and(|entry| entry.status == crate::mention_lexicon::LexiconStatus::Pending),
            "重建后待定提名应保留"
        );
        assert!(
            reasoner.mention_lexicon().entry("坏词").is_some_and(
                |entry| entry.status == crate::mention_lexicon::LexiconStatus::Tombstoned
            ),
            "重建后墓碑应保留"
        );
    }

    #[test]
    fn lexical_trace_uses_the_same_mention_lexicon_overlay() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&alignment_facts());
        reasoner.set_nlp_config(trace_nlp_config());
        let trace = reasoner
            .lexical_trace("未名科技发布了新芯片")
            .expect("应能生成读法轨迹");
        assert!(
            trace.tokens.iter().any(|token| {
                token.text == "未名科技" && token.source == crate::nlp::TagSource::MentionLexicon
            }),
            "读法轨迹应复用词表覆盖层：{:?}",
            trace.tokens
        );
    }

    #[test]
    fn predicate_alignment_feeds_query_normalization() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let receipt = reasoner.learn_predicate_alignment("恩师", "师傅");
        assert!(receipt.contains("已记住"), "{}", receipt);
        assert_eq!(
            reasoner.alignment_predicate_canonicalize("恩师").as_deref(),
            Some("师傅")
        );
        assert_eq!(
            reasoner.canonical_attribute_name("恩师").as_deref(),
            Some("师傅")
        );
        let answer = reasoner
            .answer("张三的恩师是谁")
            .expect("谓词对齐应参与问句");
        assert!(answer.contains("李四"), "{}", answer);
    }

    #[test]
    fn alignment_memory_survives_dialogue_snapshot() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&alignment_facts());
        reasoner.learn_entity_alignment("未名科技公司", "未名科技");
        let snapshot = reasoner.dialogue_snapshot();
        assert!(snapshot.alignment.is_some());

        let mut restored = SymbolicReasoner::new(test_config(), HashMap::new());
        restored.observe_facts(&alignment_facts());
        restored
            .restore_dialogue_snapshot(&snapshot)
            .expect("对齐记忆应可恢复");
        let mentions = restored.linker().alignment_mentions("远山科技公司");
        assert_eq!(mentions[0].best().unwrap().entity, "远山科技");
        assert!(restored.alignment_memory_report().contains("未名科技公司"));
    }

    #[test]
    fn alignment_commands_query_refute_and_explain() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&alignment_facts());
        let learned = reasoner
            .handle_alignment_command("对齐纠正：未名科技公司 = 未名科技")
            .unwrap();
        assert!(learned.contains("已记住"), "{}", learned);

        let query = reasoner
            .handle_alignment_command("对齐查询：远山科技公司")
            .unwrap();
        assert!(query.contains("远山科技"), "{}", query);

        let reverse = reasoner
            .handle_alignment_command("对齐答法：未名科技")
            .unwrap();
        assert!(reverse.contains("未名科技公司"), "{}", reverse);

        let refuted = reasoner
            .handle_alignment_command("对齐否决：未名科技公司 = 未名科技")
            .unwrap();
        assert!(refuted.contains("已否证"), "{}", refuted);
        let after = reasoner
            .handle_alignment_command("对齐查询：远山科技公司")
            .unwrap();
        assert!(after.contains("没有可落地"), "{}", after);
    }

    #[test]
    fn alignment_unknown_anchor_is_rejected() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&alignment_facts());
        let receipt = reasoner.learn_entity_alignment("未名科技公司", "不存在公司");
        assert!(receipt.contains("对齐未生效"), "{}", receipt);
        assert!(reasoner.linker().alignment().is_empty());
    }

    #[test]
    fn contextual_segment_report_projects_focus_types_and_graph() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&[
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
            PrologFact::Relation {
                from: "张三".into(),
                rel_type: "师傅".into(),
                to: "李四".into(),
            },
        ]);
        reasoner
            .context
            .record_entities_typed(&[("张三".into(), Some("人物".into()))]);
        reasoner.context.record_predicate("师傅");
        let report = reasoner.contextual_segment_report("张三的师傅是谁");
        assert!(report.contains("焦点：张三"), "{}", report);
        assert!(report.contains("期望类型：人物"), "{}", report);
        assert!(report.contains("师傅"), "{}", report);
        assert!(report.contains("纯符号、零训练"), "{}", report);
        assert!(report.contains("读法："), "{}", report);
    }

    #[test]
    fn joint_parse_report_uses_frame_types_and_verifies_generation() {
        let mut synonyms = HashMap::new();
        synonyms.insert("苹果".to_string(), "苹果公司".to_string());
        let mut reasoner = SymbolicReasoner::new(test_config(), synonyms);
        reasoner.observe_facts(&[
            PrologFact::Entity {
                name: "苹果".into(),
                entity_type: "水果".into(),
                properties: HashMap::new(),
            },
            PrologFact::Entity {
                name: "苹果公司".into(),
                entity_type: "公司".into(),
                properties: HashMap::new(),
            },
            PrologFact::Entity {
                name: "乔布斯".into(),
                entity_type: "人物".into(),
                properties: HashMap::new(),
            },
        ]);
        let mut schema = SchemaView::new();
        schema.declare_domain("创始人", "公司");
        schema.declare_range("创始人", "人物");
        schema.declare_disjoint("水果", "公司");
        reasoner.configure_schema_view(&schema);

        let report = reasoner.joint_parse_report("苹果的创始人是乔布斯");
        assert!(report.contains("联合解析"), "{}", report);
        assert!(report.contains("苹果公司"), "{}", report);
        assert!(report.contains("回写上下文"), "{}", report);
        assert!(!report.contains("冲突：类型互斥"), "{}", report);
        assert!(reasoner.context.focus_entities().is_empty());

        let report = reasoner.joint_parse_report_and_observe("苹果的创始人是乔布斯");
        assert!(report.contains("苹果公司"), "{}", report);
        assert_eq!(
            reasoner
                .context
                .focus_entities()
                .first()
                .map(String::as_str),
            Some("苹果公司")
        );
        assert_eq!(reasoner.context.focus_type("苹果公司"), Some("公司"));
        assert_eq!(reasoner.context.last_predicate(), Some("创始人"));

        let expected = vec![RelationFrame::new("苹果公司", "创始人", "乔布斯")];
        let verification = reasoner.verify_joint_generation("苹果的创始人是乔布斯", &expected);
        assert!(verification.passed, "{:?}", verification);

        let (candidate, verification) = reasoner
            .joint_generate_and_verify(&expected)
            .expect("肯定直陈帧应能生成候选");
        assert!(candidate.contains("创始人"), "{}", candidate);
        assert!(verification.passed, "{:?}", verification);

        let negative = vec![RelationFrame::new("苹果公司", "创始人", "乔布斯")
            .with_polarity(crate::relation_frame::Polarity::Negative)];
        assert!(reasoner.joint_generation_candidate(&negative).is_none());
    }

    #[test]
    fn contextual_reading_memory_flows_into_report_and_resets() {
        let mut synonyms = HashMap::new();
        synonyms.insert("苹果".to_string(), "苹果公司".to_string());
        let mut reasoner = SymbolicReasoner::new(test_config(), synonyms);
        reasoner.observe_facts(&[
            PrologFact::Entity {
                name: "苹果".into(),
                entity_type: "水果".into(),
                properties: HashMap::new(),
            },
            PrologFact::Entity {
                name: "苹果公司".into(),
                entity_type: "公司".into(),
                properties: HashMap::new(),
            },
        ]);
        let confirmed = reasoner.confirm_contextual_reading("苹果", "苹果公司");
        assert!(confirmed.contains("已确认"), "{}", confirmed);
        let report = reasoner.contextual_segment_report("苹果");
        assert!(report.contains("苹果公司"), "{}", report);
        let memory = reasoner.contextual_reading_memory_report();
        assert!(memory.contains("「苹果」@∅"), "{}", memory);
        assert!(!memory.contains('\u{1f}'), "{}", memory);
        let corrected = reasoner.correct_contextual_reading("苹果", "苹果公司", "苹果");
        assert!(corrected.contains("已纠正"), "{}", corrected);
        assert!(reasoner
            .clear_contextual_reading_memory()
            .contains("已清空"));
        assert!(reasoner
            .contextual_reading_memory_report()
            .starts_with("读法记忆：0 个语境桶"));
    }

    #[test]
    fn epistemic_question_survives_dialogue_frame_normalization() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let mut dialogue = crate::dialogue_manager::DialogueManager::new();
        let answer = dialogue
            .plan_pure_dialogue("李四的师傅可能是谁", &reasoner)
            .or_else(|| dialogue.plan_reply("李四的师傅可能是谁", &reasoner))
            .map(|reply| reply.text)
            .or_else(|| reasoner.answer("李四的师傅可能是谁"))
            .expect("对话层应回答情态问句");
        assert!(answer.contains("PQI"), "对话层丢失情态：{}", answer);
        assert!(answer.contains("P=1.000"), "{}", answer);
    }

    #[test]
    fn formal_semantics_report_exposes_gaps_and_supported_readings() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&[
            PrologFact::Entity {
                name: "张三".to_string(),
                entity_type: "中国人".to_string(),
                properties: HashMap::from([("年龄".to_string(), "30".to_string())]),
            },
            PrologFact::Entity {
                name: "李四".to_string(),
                entity_type: "美国人".to_string(),
                properties: HashMap::from([("年龄".to_string(), "28".to_string())]),
            },
        ]);
        let report = reasoner.formal_semantics_report("张三的年龄是多少");
        assert!(report.contains("语义森林"), "report = {}", report);
        assert!(report.contains("唯一可答"), "report = {}", report);
        let gap = reasoner.formal_semantics_report("王五的年龄是多少");
        assert!(gap.contains("证据不足"), "report = {}", gap);
        assert!(gap.contains("王五"), "report = {}", gap);
    }

    #[test]
    fn formal_semantics_fallback_only_answers_supported_readings() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&[
            PrologFact::Entity {
                name: "张三".to_string(),
                entity_type: "中国人".to_string(),
                properties: HashMap::new(),
            },
            PrologFact::Entity {
                name: "李四".to_string(),
                entity_type: "美国人".to_string(),
                properties: HashMap::new(),
            },
        ]);
        let answer = reasoner
            .formal_semantics_answer("谁是中国人？")
            .expect("应能回答有证据的实体查询");
        assert!(answer.contains("张三"), "answer = {}", answer);
        assert!(!answer.contains("李四"), "answer = {}", answer);
        assert!(
            reasoner.formal_semantics_answer("谁是火星人？").is_none(),
            "没有证据时不得把候选当事实"
        );
        let attempt = reasoner
            .formal_semantics_attempt("谁是中国人？")
            .expect("正常回答路径也应能使用唯一读法");
        assert!(attempt.contains("张三"), "attempt = {}", attempt);
        assert!(!reasoner.formal_clarification_active());
    }

    #[test]
    fn formal_semantics_fallback_answers_reverse_relation() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&[
            PrologFact::Entity {
                name: "张三".into(),
                entity_type: "人物".into(),
                properties: HashMap::new(),
            },
            PrologFact::Entity {
                name: "王五".into(),
                entity_type: "人物".into(),
                properties: HashMap::new(),
            },
            PrologFact::Entity {
                name: "李四".into(),
                entity_type: "人物".into(),
                properties: HashMap::new(),
            },
            PrologFact::Entity {
                name: "赵六".into(),
                entity_type: "人物".into(),
                properties: HashMap::new(),
            },
            PrologFact::Relation {
                from: "张三".into(),
                rel_type: "师傅".into(),
                to: "李四".into(),
            },
            PrologFact::Relation {
                from: "王五".into(),
                rel_type: "师傅".into(),
                to: "李四".into(),
            },
            PrologFact::Relation {
                from: "李四".into(),
                rel_type: "师傅".into(),
                to: "赵六".into(),
            },
        ]);
        let answer = reasoner
            .formal_semantics_answer("谁的师傅是李四？")
            .expect("反向关系查询应有答案");
        assert!(answer.contains("张三"), "answer = {}", answer);
        assert!(answer.contains("王五"), "answer = {}", answer);
        assert!(!answer.contains("赵六"), "answer = {}", answer);
        assert!(answer.contains("证据"), "answer = {}", answer);
        assert!(!reasoner.formal_clarification_active());
        let integrated = reasoner
            .answer("谁的师傅是李四？")
            .expect("正常回答路径也应接通反向查询");
        assert!(integrated.contains("张三"), "integrated = {}", integrated);
        assert!(!integrated.contains("赵六"), "integrated = {}", integrated);
    }

    #[test]
    fn formal_clarification_selects_cancels_and_reprompts() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        let frame = || FormalClarification {
            question: "张三的师傅是谁".to_string(),
            options: vec![
                FormalClarificationOption {
                    logical_form: "师傅(张三, 李四)".to_string(),
                    answer: "张三的师傅是李四。".to_string(),
                    evidence: vec!["relation(张三, 师傅, 李四)".to_string()],
                },
                FormalClarificationOption {
                    logical_form: "师傅(李四, 张三)".to_string(),
                    answer: "李四的师傅是张三。".to_string(),
                    evidence: vec!["relation(李四, 师傅, 张三)".to_string()],
                },
            ],
        };

        *reasoner.formal_clarification.lock() = Some(frame());
        let prompt = reasoner
            .formal_clarification_prompt()
            .expect("澄清帧应给出提示");
        assert!(prompt.contains("2 种有证据的读法"), "{}", prompt);
        assert!(
            matches!(
                reasoner.answer_formal_clarification("9"),
                FormalClarificationOutcome::Reprompt(_)
            ),
            "越界应重问且不清空"
        );
        assert!(reasoner.formal_clarification_active());

        match reasoner.answer_formal_clarification("选择第二个") {
            FormalClarificationOutcome::Answered(text) => {
                assert!(text.contains("李四的师傅是张三"), "{}", text);
                assert!(
                    text.contains("证据：relation(李四, 师傅, 张三)"),
                    "{}",
                    text
                );
            }
            other => panic!("应选定第二条读法，got {:?}", other),
        }
        assert!(!reasoner.formal_clarification_active());
        assert!(reasoner.formal_clarification_prompt().is_none());

        *reasoner.formal_clarification.lock() = Some(frame());
        match reasoner.answer_formal_clarification("都不是") {
            FormalClarificationOutcome::Cancelled(text) => {
                assert!(text.contains("没有把任何候选当作事实"), "{}", text);
            }
            other => panic!("应放弃澄清，got {:?}", other),
        }
        assert!(!reasoner.formal_clarification_active());

        *reasoner.formal_clarification.lock() = Some(frame());
        assert_eq!(
            reasoner.answer_formal_clarification("张三是谁"),
            FormalClarificationOutcome::NotAnAnswer
        );
        assert!(
            !reasoner.formal_clarification_active(),
            "非回答输入应清空旧澄清帧"
        );

        *reasoner.formal_clarification.lock() = Some(frame());
        let forked = reasoner.fork_session();
        assert!(
            !forked.formal_clarification_active(),
            "分叉会话不得带走上一会话的歧义澄清帧"
        );
    }

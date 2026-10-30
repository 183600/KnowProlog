    use super::*;

    fn test_config() -> ReasoningConfig {
        ReasoningConfig {
            enable_link_prediction: true,
            link_prediction_min_confidence: 0.5,
            link_prediction_min_support: 1,
            ontology_alignment_threshold: 0.72,
            max_predictions: 5,
            enable_semantic_embedding: true,
            embedding_similarity_floor: 0.6,
            dialogue_state_path: None,
            enable_generation: true,
            max_profile_sentences: 6,
            completion_min_score: 0.3,
            enable_entailment: true,
            entailment_min_confidence: 0.5,
            enable_resolution: true,
            resolution_max_depth: 6,
            resolution_max_solutions: 12,
            enable_association_rules: true,
            association_min_support: 0.2,
            association_min_confidence: 0.5,
            association_min_lift: 1.05,
            enable_abduction: true,
            abduction_max_hypotheses: 5,
            abduction_max_scenario_size: 2,
            enable_clarification: true,
            enable_graph_walk: true,
            graph_walk_damping: 0.85,
            graph_walk_iterations: 12,
            enable_soft_logic: true,
            soft_logic_max_iterations: 4,
            soft_logic_min_probability: 0.5,
            enable_causal_reasoning: true,
            causal_max_depth: 4,
            enable_gradient_attribution: true,
            enable_causal_tokenization: true,
            enable_problog: true,
            problog_max_depth: 6,
            problog_max_proofs: 12,
            enable_dialogue_manager: true,
            enable_discourse: true,
            discourse_max_sentences: 6,
            discourse_max_rounds: 4,
            enable_holographic_memory: true,
            holographic_dimension: 4096,
            holographic_similarity_floor: 0.01,
            holographic_max_hits: 6,
            enable_contextual_rag: false,
            require_predicate_admission: false,
            confidence: crate::confidence::ConfidenceConfig::default(),
            constructive: crate::constructive::ConstructionConfig::default(),
            soft_route: crate::soft_route::SoftRouteConfig::default(),
            rewrite_retrieval: crate::rewrite_retrieval::RewriteRetrievalConfig::default(),
            semantic_monitor: crate::SemanticMonitorConfig::default(),
        }
    }

    fn pcc_config() -> ReasoningConfig {
        let full: crate::SystemConfig =
            toml::from_str(include_str!("../../config.toml")).expect("config.toml 必须可解析");
        let mut config = test_config();
        config.confidence = full.reasoning.confidence;
        config
    }

    fn rewrite_retrieval_config() -> ReasoningConfig {
        let mut config = test_config();
        config.rewrite_retrieval.enabled = true;
        config
    }

    fn master_facts() -> Vec<PrologFact> {
        vec![
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
            PrologFact::Relation {
                from: "王五".to_string(),
                rel_type: "师傅".to_string(),
                to: "赵六".to_string(),
            },
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "师祖".to_string(),
                to: "王五".to_string(),
            },
        ]
    }

    fn pair_facts() -> Vec<PrologFact> {
        vec![
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
        ]
    }

    #[test]
    fn rewrite_retrieval_arbitrates_readings_with_dense_evidence() {
        let mut reasoner = SymbolicReasoner::new(rewrite_retrieval_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());

        let assessment = reasoner
            .rewrite_retrieval_assessment("谁的师傅是李四")
            .expect("应收集到候选读法");
        assert_eq!(
            assessment.decision,
            crate::rewrite_retrieval::Decision::Accept,
            "{}",
            assessment.render_report()
        );
        assert_eq!(assessment.answer(), Some("张三"));

        let report = reasoner
            .rewrite_retrieval_report("谁的师傅是李四")
            .expect("应能打印裁决回执");
        assert!(report.contains("读法裁决"), "{}", report);
        assert!(report.contains("张三"), "{}", report);
    }

    #[test]
    fn rewrite_retrieval_clarification_builds_selectable_frame() {
        let mut reasoner = SymbolicReasoner::new(rewrite_retrieval_config(), HashMap::new());
        reasoner.observe_facts(&[
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "师傅".to_string(),
                to: "李四".to_string(),
            },
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "老师".to_string(),
                to: "王五".to_string(),
            },
        ]);
        let candidates = vec![
            HypothesisCandidate::new(
                HypothesisSource::ExactPlan,
                "master",
                "张三的师傅是谁",
                QueryPlan::FindObjectByPredicate {
                    subject: "张三".to_string(),
                    predicate: "师傅".to_string(),
                },
            ),
            HypothesisCandidate::new(
                HypothesisSource::ExactPlan,
                "teacher",
                "张三的老师是谁",
                QueryPlan::FindObjectByPredicate {
                    subject: "张三".to_string(),
                    predicate: "老师".to_string(),
                },
            ),
        ];
        let assessment = rewrite_retrieval::assess(
            "张三的师傅是谁",
            candidates,
            &reasoner.config.rewrite_retrieval,
            |query, k, constraints| {
                reasoner.evidence_search(query, k, constraints, SearchBudget::default())
            },
        );
        let clarification = reasoner
            .rewrite_retrieval_clarification("张三的师傅是谁", &assessment)
            .expect("两个读法都有可执行答案时应建立澄清帧");
        assert_eq!(clarification.options.len(), 2);
        *reasoner.formal_clarification.lock() = Some(clarification);
        match reasoner.answer_formal_clarification("1") {
            FormalClarificationOutcome::Answered(answer) => {
                assert!(answer.contains("按你选择的读法"), "{}", answer);
            }
            other => panic!("应接受序号选择：{:?}", other),
        }
    }

    #[test]
    fn learned_rewrite_book_contributes_candidate_readings() {
        use crate::rewrite_book::{RewriteRule, RewriteStatus};

        let mut reasoner = SymbolicReasoner::new(rewrite_retrieval_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let mut book = RewriteBook::new();
        book.rules.push(RewriteRule {
            id: "learned-shell".to_string(),
            plan_kind: "FindObjectByPredicate".to_string(),
            surface: "{S}到底{P}是谁".to_string(),
            canonical: "{S}的{P}是谁".to_string(),
            status: RewriteStatus::Active,
            support: 3,
            predicates: vec!["师傅".to_string()],
            entities: Vec::new(),
            examples: Vec::new(),
            refutations: 0,
            source: "test".to_string(),
        });
        reasoner.set_rewrite_book(book);

        let candidates = reasoner.rewrite_retrieval_candidates("张三到底师傅是谁");
        assert!(
            candidates
                .iter()
                .any(|candidate| candidate.source == HypothesisSource::LearnedRewrite),
            "{:?}",
            candidates
                .iter()
                .map(|candidate| candidate.receipt())
                .collect::<Vec<_>>()
        );
        let assessment = reasoner
            .rewrite_retrieval_assessment("张三到底师傅是谁")
            .expect("应收集到候选读法");
        assert_eq!(assessment.answer(), Some("李四"));
        let answer = reasoner
            .answer("张三到底师傅是谁")
            .expect("习得重写应直接回答");
        assert!(answer.contains("李四"), "{}", answer);
        assert!(!answer.contains("知识（类型"), "{}", answer);
    }

    #[test]
    fn rewrite_retrieval_is_disabled_by_default() {
        let config = test_config();
        assert!(!config.rewrite_retrieval.enabled);
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        reasoner.observe_facts(&master_facts());
        assert!(reasoner
            .rewrite_retrieval_assessment("谁的师傅是李四")
            .is_some());
    }

    #[test]
    fn test_pronoun_resolution_uses_focus() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        assert!(reasoner.contextualize("张三的师傅是谁").is_none());
        let rewritten = reasoner
            .contextualize("他的师傅是谁")
            .expect("代词应被解析为上下文焦点实体");
        assert!(rewritten.contains("张三"));
        assert!(!rewritten.contains("他的"));
    }

    #[test]
    fn apply_language_policy_swaps_language_layer_but_keeps_session_memory() {
        let mut config = test_config();
        config.max_profile_sentences = 4;
        config.enable_generation = true;
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        reasoner.observe_facts(&master_facts());
        let revision = reasoner.revision();
        assert!(reasoner.linker.is_known_entity("张三"));

        let mut language: crate::SystemConfig =
            toml::from_str(include_str!("../../config.toml")).expect("config.toml 必须可解析");
        language.reasoning.max_profile_sentences = 2;
        language.reasoning.enable_generation = false;
        language.synonyms = Some(crate::SynonymsConfig {
            mappings: HashMap::from([("师父".to_string(), "师傅".to_string())]),
        });
        language.semantic_grammar.enabled = false;

        let sync = reasoner.apply_language_policy(&language);
        assert!(sync.nlp_updated);
        assert_eq!(sync.synonyms_added, 1);
        assert!(sync.generation_updated);
        assert!(!sync.semantic_grammar);
        assert!(reasoner.aligner.synonyms().contains_key("师父"));
        assert_eq!(reasoner.config.max_profile_sentences, 2);
        assert!(!reasoner.config.enable_generation);
        assert_eq!(reasoner.revision(), revision, "语言热替换不改知识修订号");
        assert!(
            reasoner.linker.is_known_entity("张三"),
            "语言热替换不丢会话实体"
        );
    }

    /// 类型闸门：「他/她」可解析人物（类型未知按开放世界放行），
    /// 「它」绝不指代人物，「该公司」顺位到组织，「前者/后者」按焦点栈序取。
    #[test]
    fn test_pronoun_type_gate_and_ordinals() {
        let mut context = DialogueContext::new(8);
        context.record_entities_typed(&[
            ("张三".to_string(), Some("人".to_string())),
            ("未名科技公司".to_string(), Some("组织".to_string())),
            ("某物".to_string(), None),
        ]);
        assert_eq!(
            context.resolve_pronouns("他的师傅是谁").as_deref(),
            Some("张三的师傅是谁")
        );
        assert_eq!(
            context.resolve_pronouns("它的创始人是谁").as_deref(),
            Some("未名科技公司的创始人是谁"),
            "「它」应优先解析到已知非人物焦点"
        );
        assert_eq!(
            context.resolve_pronouns("前者是谁").as_deref(),
            Some("张三是谁"),
            "「前者」= 焦点栈第 1 个（最近一次提及顺序）"
        );
        assert_eq!(
            context.resolve_pronouns("后者是谁").as_deref(),
            Some("未名科技公司是谁")
        );
        // 「该公司」只认公司/组织：焦点栈顶端是人时顺位到组织
        context.clear();
        context.record_entities_typed(&[
            ("张三".to_string(), Some("人".to_string())),
            ("未名科技公司".to_string(), Some("组织".to_string())),
        ]);
        assert_eq!(
            context.resolve_pronouns("该公司的创始人是谁").as_deref(),
            Some("未名科技公司的创始人是谁")
        );
        // 只有人物焦点时「它」让位（不替换、不乱猜）
        context.clear();
        context.record_entities_typed(&[("张三".to_string(), Some("人".to_string()))]);
        assert_eq!(
            context.resolve_pronouns("他的师傅是谁").as_deref(),
            Some("张三的师傅是谁")
        );
        assert!(
            context.resolve_pronouns("它的师傅是谁").is_none(),
            "「它」不得指代人物"
        );
    }

    #[test]
    fn test_ellipsis_completion_uses_last_predicate() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.contextualize("张三的师祖是谁");
        let rewritten = reasoner
            .contextualize("那李四呢")
            .expect("省略句应结合上文补全");
        assert_eq!(rewritten, "李四的师祖是什么");
    }

    #[test]
    fn test_contextualize_ignores_other_ren() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.contextualize("张三的师傅是谁");
        let rewritten = reasoner.contextualize("其他人的师傅是谁");
        assert!(rewritten.is_none() || rewritten.as_deref() == Some("其他张三的师傅是谁"));
    }

    fn acquaintance_facts() -> Vec<PrologFact> {
        vec![
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "认识".to_string(),
                to: "李四".to_string(),
            },
            PrologFact::Relation {
                from: "王五".to_string(),
                rel_type: "认识".to_string(),
                to: "李四".to_string(),
            },
        ]
    }

    #[test]
    fn test_relation_ellipsis_uses_last_plan_kind() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&acquaintance_facts());
        assert!(reasoner.contextualize("张三认识谁").is_none());
        assert_eq!(
            reasoner.context().last_plan_kind(),
            Some("FindObjectByPredicate")
        );
        let rewritten = reasoner
            .contextualize("李四呢")
            .expect("关系问句后的省略应沿用计划类型");
        assert_eq!(rewritten, "李四认识谁");

        assert!(reasoner.contextualize("谁认识李四").is_none());
        assert_eq!(reasoner.context().last_plan_kind(), Some("FindSubject"));
        let rewritten = reasoner
            .contextualize("张三呢")
            .expect("反查句后的省略应替换宾语");
        assert_eq!(rewritten, "谁认识张三");
    }

    #[test]
    fn test_last_plan_kind_survives_snapshot_roundtrip() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&acquaintance_facts());
        reasoner.contextualize("张三认识谁");
        let snapshot = reasoner.dialogue_snapshot();
        assert_eq!(
            snapshot.last_plan_kind.as_deref(),
            Some("FindObjectByPredicate")
        );

        let mut restored = SymbolicReasoner::new(test_config(), HashMap::new());
        restored.observe_facts(&acquaintance_facts());
        restored
            .restore_dialogue_snapshot(&snapshot)
            .expect("快照应可恢复");
        let rewritten = restored
            .contextualize("李四呢")
            .expect("恢复后省略补全仍应沿用计划类型");
        assert_eq!(rewritten, "李四认识谁");
    }

    fn attribute_facts() -> Vec<PrologFact> {
        let mut properties = HashMap::new();
        properties.insert("身高".to_string(), "226cm".to_string());
        properties.insert("体重".to_string(), "140kg".to_string());
        vec![PrologFact::Entity {
            name: "姚明".to_string(),
            entity_type: "人".to_string(),
            properties,
        }]
    }

    #[test]
    fn attribute_unit_request_converts_stored_quantity() {
        use crate::attribute_bridge::{
            AttributeBridgeConfig, AttributeProfileConfig, AttributeRegistry, AttributeValueType,
        };

        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.attribute_profiles = AttributeRegistry::from_config(&AttributeBridgeConfig {
            enabled: true,
            profiles: vec![AttributeProfileConfig {
                name: "height".to_string(),
                aliases: vec!["身高".to_string()],
                domain: vec!["人".to_string()],
                value_type: AttributeValueType::Quantity,
                dimension: Some("length".to_string()),
                unit: Some("厘米".to_string()),
                ..Default::default()
            }],
        });
        reasoner.observe_facts(&attribute_facts());

        let meters = reasoner.answer("姚明身高多少米").expect("应回答单位化问句");
        assert!(meters.contains("2.26米"), "{meters}");
        let centimeters = reasoner
            .answer("姚明身高多少厘米")
            .expect("应回答单位化问句");
        assert!(centimeters.contains("226厘米"), "{centimeters}");
        assert!(
            reasoner.answer("姚明身高是多少").is_some(),
            "无显式单位仍走既有属性回答路径"
        );
    }

    #[test]
    fn type_memory_answers_profiles_comparisons_and_members() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        let mut person_properties = HashMap::new();
        person_properties.insert("年龄".to_string(), "30".to_string());
        let mut organization_properties = HashMap::new();
        organization_properties.insert("城市".to_string(), "北京".to_string());
        reasoner.observe_facts(&[
            PrologFact::Entity {
                name: "张三".to_string(),
                entity_type: "人物".to_string(),
                properties: person_properties,
            },
            PrologFact::Entity {
                name: "李四".to_string(),
                entity_type: "人物".to_string(),
                properties: HashMap::new(),
            },
            PrologFact::Entity {
                name: "甲公司".to_string(),
                entity_type: "组织".to_string(),
                properties: organization_properties,
            },
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "师傅".to_string(),
                to: "李四".to_string(),
            },
        ]);

        let profile = reasoner
            .answer("人物一般有哪些属性")
            .expect("类型画像应可回答");
        assert!(profile.contains("类型画像「人物」"), "{}", profile);
        assert!(profile.contains("年龄"), "{}", profile);
        assert!(profile.contains("师傅"), "{}", profile);

        let comparison = reasoner
            .answer("人物和组织有什么区别")
            .expect("类型对比应可回答");
        assert!(
            comparison.contains("类型对比「人物」与「组织」"),
            "{}",
            comparison
        );
        assert!(comparison.contains("年龄"), "{}", comparison);
        assert!(comparison.contains("城市"), "{}", comparison);

        let members = reasoner
            .answer_certain("人物有哪些成员")
            .expect("成员列表应可确证回答");
        assert!(members.contains("张三"), "{}", members);
        assert!(members.contains("李四"), "{}", members);

        let entity_type = reasoner
            .answer("张三是什么类型")
            .expect("实体类型问句应可回答");
        assert!(entity_type.contains("「张三」的类型是「人物」"), "{}", entity_type);

        let inventory = reasoner
            .answer("李四有哪些关系")
            .expect("实体关系清单应可回答");
        assert!(inventory.contains("张三 师傅 李四（入向）"), "{}", inventory);

        assert!(!reasoner
            .answer("张三的年龄是多少")
            .unwrap_or_default()
            .contains("类型画像"));
    }

    #[test]
    fn test_frame_full_question_records_last_frame() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        assert!(reasoner.contextualize("张三的师祖是谁").is_none());
        assert_eq!(
            reasoner.frame_memory().last().map(|frame| frame.display()),
            Some("张三 的 师祖".to_string())
        );
    }

    #[test]
    fn test_frame_multi_hop_question_is_not_recorded() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        assert!(reasoner.contextualize("张三的师傅的师傅是谁").is_none());
        assert!(
            reasoner.frame_memory().last().is_none(),
            "多跳问句不是 AskAttr 帧，不应被登记"
        );
    }

    #[test]
    fn test_frame_followup_inherits_attribute_with_receipt() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.contextualize("张三的师祖是谁");
        let rewrite = reasoner
            .contextualize_with_receipt("那李四呢")
            .expect("帧合一应接管省略追问");
        assert_eq!(rewrite.text, "李四的师祖是什么");
        assert!(
            rewrite
                .notes
                .iter()
                .any(|note| note.contains("属性「师祖」←第 1 轮的完整帧")),
            "来源收据应标注属性洞的来源：{:?}",
            rewrite.notes
        );
    }

    #[test]
    fn test_frame_followup_inherits_entity_slot() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.contextualize("张三的师傅是谁");
        let rewrite = reasoner
            .contextualize_with_receipt("师祖呢")
            .expect("帧合一应接管属性追问");
        assert_eq!(rewrite.text, "张三的师祖是什么");
    }

    #[test]
    fn test_frame_unknown_followup_falls_back_to_char_rule() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.contextualize("张三的师傅是谁");
        let rewritten = reasoner
            .contextualize("某某某呢")
            .expect("未知实体仍应走省略规则");
        assert_eq!(rewritten, "某某某的师傅是什么");
    }

    #[test]
    fn test_frame_compound_defers_tail_then_pays_off() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&attribute_facts());
        let first = reasoner
            .contextualize_with_receipt("姚明的身高和体重是多少")
            .expect("复合问句应拆帧并先答第一问");
        assert_eq!(first.text, "姚明的身高是什么");
        assert!(
            first.notes.iter().any(|note| note.contains("挂账")),
            "第一答应提示其余挂账：{:?}",
            first.notes
        );
        assert!(reasoner.frame_memory_report().contains("姚明 的 体重"));

        let answer = reasoner
            .answer(&first.text)
            .expect("第一问应能命中属性索引");
        assert!(answer.contains("226cm"), "第一问答案：{}", answer);

        let second = reasoner
            .contextualize_with_receipt("体重呢")
            .expect("追问应结清挂账");
        assert_eq!(second.text, "姚明的体重是什么");
        assert!(
            second
                .notes
                .iter()
                .any(|note| note.contains("补全第 1 轮挂账")),
            "收据应标注挂账来源：{:?}",
            second.notes
        );
        assert!(reasoner.frame_memory().pending().is_empty());
        let answer2 = reasoner
            .answer(&second.text)
            .expect("结清的问句应能命中属性索引");
        assert!(answer2.contains("140kg"), "第二问答案：{}", answer2);
    }

    #[test]
    fn slot_correction_replaces_graph_value_in_the_session_view() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&attribute_facts());
        assert!(reasoner.adopt_session_triple("姚明", "身高", "229cm"));
        assert!(reasoner
            .session_facts()
            .iter()
            .any(|atom| atom.display() == "身高(姚明, 229cm)"));
        assert!(
            !reasoner
                .session_facts()
                .iter()
                .any(|atom| atom.display() == "身高(姚明, 226cm)"),
            "旧值不应同时留在会话事实里"
        );
        assert!(
            reasoner.is_suppressed_slot_value("姚明", "身高", "230cm"),
            "功能槽有会话赢家时，任何其他同槽取值都应被视图压制"
        );
        let answer = reasoner.answer("姚明的身高是什么").expect("应有属性答案");
        assert!(answer.contains("229cm"), "新值应生效：{}", answer);
        assert!(!answer.contains("226cm"), "旧值应被视图压制：{}", answer);
        let evidence = reasoner.evidence_search(
            "姚明的身高",
            8,
            &SearchConstraints::default(),
            SearchBudget::unlimited(),
        );
        assert!(
            !evidence.hits.iter().any(|hit| hit.object == "226cm"),
            "证据索引不得再返回被压制的旧值：{:?}",
            evidence
                .hits
                .iter()
                .map(|hit| hit.text())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn direct_conflict_asks_then_explicit_correction_records_history() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&attribute_facts());
        let asked = reasoner.assert_nl_fact("姚明的身高是229cm");
        assert!(asked.contains("改口"), "直陈冲突应先问：{}", asked);
        assert!(reasoner.session_facts().is_empty(), "提问阶段必须零写入");

        let corrected = reasoner.correct_nl_fact("姚明的身高是229cm");
        assert!(
            corrected.contains("226cm"),
            "回执应给出 before：{}",
            corrected
        );
        assert!(
            corrected.contains("229cm"),
            "回执应给出 after：{}",
            corrected
        );
        let history = reasoner.slot_history_report("姚明", "身高");
        assert!(history.contains("第 1 次记录"), "{}", history);
        assert!(history.contains("不等于否定"), "{}", history);

        let answer = reasoner.answer("姚明的身高是什么").expect("应有属性答案");
        assert!(answer.contains("229cm"), "{}", answer);
        assert!(!answer.contains("226cm"), "{}", answer);
    }

    #[test]
    fn forgetting_a_correction_falls_back_to_the_graph_value() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&attribute_facts());
        reasoner.correct_nl_fact("姚明的身高是229cm");
        let forgotten = reasoner.retract_nl_fact("姚明的身高是229cm");
        assert!(forgotten.contains("已忘记"), "{}", forgotten);
        let answer = reasoner.answer("姚明的身高是什么").expect("应有图谱答案");
        assert!(answer.contains("226cm"), "应回落到图谱值：{}", answer);
        assert!(!answer.contains("229cm"), "{}", answer);
    }

    #[test]
    fn explicit_correction_clears_the_matching_negative_fact() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&attribute_facts());
        reasoner.assert_negative_nl_fact("姚明的身高不是229cm");
        assert_eq!(reasoner.session_negative_facts.len(), 1);
        reasoner.correct_nl_fact("姚明的身高是229cm");
        assert!(
            reasoner.session_negative_facts.is_empty(),
            "改口为肯定事实时应撤销同一原子的否定"
        );
        assert!(reasoner
            .session_facts()
            .iter()
            .any(|atom| atom.display() == "身高(姚明, 229cm)"));
    }

    #[test]
    fn slot_ledger_and_history_survive_dialogue_snapshot() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&attribute_facts());
        reasoner.correct_nl_fact("姚明的身高是229cm");
        let snapshot = reasoner.dialogue_snapshot();
        assert!(snapshot.slots.is_some(), "非空槽账本应随快照导出");

        let mut restored = SymbolicReasoner::new(test_config(), HashMap::new());
        restored.observe_facts(&attribute_facts());
        restored
            .restore_dialogue_snapshot(&snapshot)
            .expect("快照恢复应成功");
        assert!(restored
            .slot_history_report("姚明", "身高")
            .contains("第 1 次记录"));
        let answer = restored.answer("姚明的身高是什么").expect("应有属性答案");
        assert!(answer.contains("229cm"), "{}", answer);
        assert!(!answer.contains("226cm"), "{}", answer);
    }

    #[test]
    fn hearsay_is_recorded_as_evidence_but_not_answered() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&attribute_facts());
        let output = reasoner.assert_nl_fact("据说姚明的身高是230cm");
        assert!(output.contains("证据"), "{}", output);
        assert!(reasoner.session_facts().is_empty(), "传闻不得写会话事实");
        let answer = reasoner.answer("姚明的身高是什么").expect("应有属性答案");
        assert!(answer.contains("226cm"), "{}", answer);
        assert!(!answer.contains("230cm"), "传闻不得进入回答：{}", answer);
    }

    #[test]
    fn test_attribute_question_answers_from_index() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&attribute_facts());
        let answer = reasoner
            .answer("姚明的身高是什么")
            .expect("属性值应沿属性槽位命中");
        assert!(answer.contains("226cm"), "属性答案：{}", answer);
    }

    #[test]
    fn test_evidence_index_ranks_observed_facts_with_receipt() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let search = reasoner.evidence_search(
            "张三的师傅是谁",
            3,
            &SearchConstraints::default(),
            SearchBudget::unlimited(),
        );
        assert!(!search.hits.is_empty());
        assert_eq!(search.hits[0].subject, "张三");
        assert_eq!(search.hits[0].predicate, "师傅");
        assert!(search.receipt.certified_top_k);
        assert!(search
            .receipt
            .notes
            .iter()
            .any(|note| note.contains("top-k")));
    }

    #[test]
    fn test_evidence_index_honours_exact_constraints() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let constraints = SearchConstraints {
            predicate: Some("师傅".to_string()),
            ..SearchConstraints::default()
        };
        let search = reasoner.evidence_search("师傅", 8, &constraints, SearchBudget::unlimited());
        assert!(!search.hits.is_empty());
        assert!(search.hits.iter().all(|hit| hit.predicate == "师傅"));
        assert!(search.receipt.constraint_pruned > 0);
    }

    #[test]
    fn test_evidence_fallback_answers_with_citation_and_receipt() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let answer = reasoner
            .answer_by_evidence("张三的师傅是谁")
            .expect("已锚定实体且已提及谓词时应由证据索引作答");
        assert!(answer.contains("张三"), "{}", answer);
        assert!(answer.contains("师傅"), "{}", answer);
        assert!(answer.contains("李四"), "{}", answer);
        assert!(answer.contains("检索回执"), "{}", answer);
        assert!(answer.contains("top-k 已证明"), "{}", answer);
    }

    #[test]
    fn test_evidence_fallback_declines_without_anchored_entity() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        assert!(reasoner.answer_by_evidence("师傅是谁").is_none());
    }

    fn contextual_config() -> ReasoningConfig {
        let mut config = test_config();
        config.enable_contextual_rag = true;
        config
    }

    #[test]
    fn test_contextual_rag_followup_uses_focus_and_last_predicate() {
        let mut reasoner = SymbolicReasoner::new(contextual_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.context.record_entities(&["张三".to_string()]);
        reasoner.context.record_predicate("师傅");

        let assembly = reasoner
            .contextual_evidence("还有呢")
            .expect("追问应能沿用焦点与上一谓词");
        let bundle = assembly.bundle.as_ref().expect("应有证据束");
        assert!(bundle.answer.contains("李四"), "{}", bundle.answer);
        assert!(bundle.answer.contains("[E1]"), "{}", bundle.answer);
        assert_eq!(bundle.kind, contextual_rag::BundleKind::Single);

        let no_more = reasoner
            .contextual_evidence("还有呢")
            .expect("重复追问应诚实说明没有新证据");
        assert_eq!(
            no_more.bundle.as_ref().map(|bundle| bundle.kind),
            Some(contextual_rag::BundleKind::NoMore)
        );
        assert!(no_more
            .answer()
            .unwrap_or_default()
            .contains("没有新的已核实证据"));
    }

    #[test]
    fn test_contextual_rag_switches_focus_with_explicit_entity() {
        let mut reasoner = SymbolicReasoner::new(contextual_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.context.record_entities(&["张三".to_string()]);
        reasoner.context.record_predicate("师傅");

        let answer = reasoner
            .answer_contextual("那李四呢")
            .expect("显式实体追问应切到李四并沿用师傅");
        assert!(answer.contains("王五"), "{}", answer);
        assert!(answer.contains("李四"), "{}", answer);
    }

    #[test]
    fn test_contextual_rag_profile_without_predicate() {
        let mut reasoner = SymbolicReasoner::new(contextual_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let answer = reasoner
            .answer_contextual("介绍一下张三")
            .expect("无谓词画像问题应装配实体画像");
        assert!(answer.contains("实体画像"), "{}", answer);
        assert!(answer.contains("李四"), "{}", answer);
        assert!(answer.contains("王五"), "{}", answer);
        assert!(answer.contains("[E1]"), "{}", answer);
    }

    #[test]
    fn test_contextual_rag_records_verified_holographic_channel() {
        let mut reasoner = SymbolicReasoner::new(contextual_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let assembly = reasoner
            .contextual_evidence("张三的师傅是谁")
            .expect("应能装配证据");
        assert!(
            assembly.receipt.holographic_support > 0,
            "同一事实的全息硬核查候选应计入通道回执"
        );
        assert!(!assembly.receipt.certified);
    }

    #[test]
    fn test_contextual_rag_reports_multiple_values_without_collapsing() {
        let mut facts = vec![
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
        ];
        facts.extend(master_facts());
        let mut reasoner = SymbolicReasoner::new(contextual_config(), HashMap::new());
        reasoner.observe_facts(&facts);
        let answer = reasoner
            .answer_contextual("张三的师傅是谁")
            .expect("多值槽位应全部列出");
        assert!(answer.contains("李四"), "{}", answer);
        assert!(answer.contains("王五"), "{}", answer);
        assert!(answer.contains("多个登记取值"), "{}", answer);
    }

    #[test]
    fn test_contextual_rag_path_between_two_anchors() {
        let facts = vec![
            PrologFact::Relation {
                from: "甲".to_string(),
                rel_type: "师傅".to_string(),
                to: "丙".to_string(),
            },
            PrologFact::Relation {
                from: "丙".to_string(),
                rel_type: "师傅".to_string(),
                to: "乙".to_string(),
            },
        ];
        let mut reasoner = SymbolicReasoner::new(contextual_config(), HashMap::new());
        reasoner.observe_facts(&facts);
        let answer = reasoner
            .answer_contextual("甲和乙是什么关系")
            .expect("双锚点应装配显式路径");
        assert!(answer.contains("显式路径"), "{}", answer);
        assert!(answer.contains("甲 —师傅→ 丙"), "{}", answer);
        assert!(answer.contains("丙 —师傅→ 乙"), "{}", answer);
    }

    #[test]
    fn test_contextual_rag_auto_answer_path_is_enabled_by_config() {
        let mut reasoner = SymbolicReasoner::new(contextual_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.context.record_entities(&["张三".to_string()]);
        reasoner.context.record_predicate("师傅");
        let answer = reasoner
            .answer("还有呢")
            .expect("自动回答路径应在配置开启时接管追问");
        assert!(answer.contains("李四"), "{}", answer);
    }

    #[test]
    fn test_contextual_rag_is_off_by_default() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        assert!(reasoner.answer_contextual("张三的师傅是谁").is_none());
    }

    #[test]
    fn test_contextual_rag_abstains_without_anchor_or_predicate() {
        let mut reasoner = SymbolicReasoner::new(contextual_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        assert!(reasoner.answer_contextual("介绍一下不存在的人").is_none());
    }

    #[test]
    fn test_session_facts_enter_and_leave_evidence_index() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.assert_nl_fact("钱七是孙八的师傅");

        let present = reasoner.evidence_search(
            "钱七 孙八 师傅",
            8,
            &SearchConstraints::default(),
            SearchBudget::unlimited(),
        );
        assert!(present.hits.iter().any(|hit| {
            (hit.subject == "钱七" || hit.object == "钱七")
                && (hit.subject == "孙八" || hit.object == "孙八")
        }));

        reasoner.retract_nl_fact("钱七是孙八的师傅");
        let absent = reasoner.evidence_search(
            "钱七 孙八 师傅",
            8,
            &SearchConstraints::default(),
            SearchBudget::unlimited(),
        );
        assert!(!absent.hits.iter().any(|hit| {
            (hit.subject == "钱七" || hit.object == "钱七")
                && (hit.subject == "孙八" || hit.object == "孙八")
        }));
    }

    #[test]
    fn test_adopted_session_triple_feeds_proof_and_snapshot() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        assert!(
            reasoner.adopt_session_triple("王五", "师傅", "赵六"),
            "首次采纳应返回 true"
        );
        assert!(
            !reasoner.adopt_session_triple("王五", "师傅", "赵六"),
            "重复采纳必须幂等"
        );
        let proof = reasoner.resolution_proof_report("王五的师傅是赵六");
        assert!(!proof.contains("未得证"), "{}", proof);

        let snapshot = reasoner.dialogue_snapshot();
        let mut restored = SymbolicReasoner::new(test_config(), HashMap::new());
        restored.observe_facts(&master_facts());
        restored.restore_dialogue_snapshot(&snapshot).unwrap();
        let proof = restored.resolution_proof_report("王五的师傅是赵六");
        assert!(
            !proof.contains("未得证"),
            "跨快照后会话事实应继续参与证明：{}",
            proof
        );
    }

    #[test]
    fn test_assumption_world_supports_followup_and_exit() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let receipt = reasoner
            .enter_assumption_world("赵六的师傅是钱七")
            .expect("假设世界应可进入");
        assert!(receipt.contains("假设世界"), "{}", receipt);
        assert!(reasoner.assumption_world().is_some());

        let answer = reasoner.answer("赵六的师傅是谁").expect("世界内应能回答");
        assert!(answer.contains("钱七"), "{}", answer);
        assert!(
            answer.contains("假设世界"),
            "回答应标注当前世界：{}",
            answer
        );

        let exit = reasoner.exit_assumption_world().expect("应可退出");
        assert!(exit.contains("已退出"), "{}", exit);
        assert!(reasoner.assumption_world().is_none());
        let after = reasoner.answer("赵六的师傅是谁");
        assert!(
            after.as_deref().map_or(true, |text| !text.contains("钱七")),
            "退出后临时前提必须消失：{:?}",
            after
        );
    }

    #[test]
    fn remembered_assumption_world_is_receipted_and_undoable() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.enter_assumption_world("赵六的师傅是钱七").unwrap();
        let remembered = reasoner.remember_assumption_world().expect("应可记住");
        assert!(remembered.contains("修订回执"), "{}", remembered);
        assert!(reasoner.assumption_world().is_none());
        assert!(reasoner
            .session_facts()
            .iter()
            .any(|atom| atom.display() == "师傅(赵六, 钱七)"));

        let undone = reasoner.undo_last_session_revision();
        assert!(undone.contains("已撤销会话修订"), "{}", undone);
        assert!(
            !reasoner
                .session_facts()
                .iter()
                .any(|atom| atom.display() == "师傅(赵六, 钱七)"),
            "撤销后假设世界不再留在会话事实层"
        );
    }

    #[test]
    fn test_assumption_world_survives_snapshot_and_can_be_remembered() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.enter_assumption_world("赵六的师傅是钱七").unwrap();
        let snapshot = reasoner.dialogue_snapshot();
        assert!(!snapshot.assumptions.is_empty());

        let mut restored = SymbolicReasoner::new(test_config(), HashMap::new());
        restored.observe_facts(&master_facts());
        restored.restore_dialogue_snapshot(&snapshot).unwrap();
        assert!(restored.assumption_world().is_some());
        let answer = restored.answer("赵六的师傅是谁").expect("恢复后应能回答");
        assert!(answer.contains("钱七"), "{}", answer);

        let remembered = restored.remember_assumption_world().expect("应可记住");
        assert!(remembered.contains("会话事实层"), "{}", remembered);
        assert!(restored.assumption_world().is_none());
        assert!(restored
            .session_facts()
            .iter()
            .any(|atom| atom.display() == "师傅(赵六, 钱七)"));
        let answer = restored.answer("赵六的师傅是谁").expect("记住后仍应能回答");
        assert!(answer.contains("钱七"), "{}", answer);
        assert!(
            !answer.contains("假设世界"),
            "已并入会话层后不应再标假设：{}",
            answer
        );
    }

    #[test]
    fn test_assumption_world_feeds_soft_logic_and_exit_removes_it() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner
            .enter_assumption_world("赵六的师傅是钱七")
            .expect("假设世界应可进入");
        let inside = reasoner.soft_inference_report("赵六的师傅");
        assert!(inside.contains("钱七"), "假设世界应进入软逻辑：{}", inside);

        reasoner.exit_assumption_world().expect("应可退出");
        let outside = reasoner.soft_inference_report("赵六的师傅");
        assert!(
            !outside.contains("钱七"),
            "退出后软逻辑不得残留假设：{}",
            outside
        );
    }

    #[test]
    fn test_assumption_world_command_does_not_swallow_one_shot_counterfactual() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        assert!(reasoner
            .handle_assumption_command("假设赵六的师傅是钱七，那么李四的师傅是谁？")
            .is_none());
        assert!(reasoner.assumption_world().is_none());
        let report = reasoner
            .handle_assumption_command("假设：赵六的师傅是钱七")
            .expect("带冒号的假设应进入世界");
        assert!(report.contains("假设世界"), "{}", report);
    }

    #[test]
    fn test_frame_keeps_inverted_query_surface() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        reasoner.register_predicate_query_surface("徒弟", "师傅");
        reasoner.contextualize("张三的徒弟是谁");
        assert_eq!(
            reasoner
                .frame_memory()
                .last()
                .and_then(|frame| frame.attribute.clone()),
            Some("徒弟".to_string()),
            "帧属性应保留逆向词面，不能塌缩成槽位「师傅」"
        );
        let rewrite = reasoner
            .contextualize_with_receipt("那李四呢")
            .expect("帧合一应保留逆向方向");
        assert_eq!(rewrite.text, "李四的徒弟是什么");
    }

    #[test]
    fn test_frame_memory_persists_across_sessions() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("dialogue.json");
        let state_path = path.to_str().unwrap();

        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&attribute_facts());
        reasoner.contextualize_with_receipt("姚明的身高和体重是多少");
        reasoner.save_dialogue_state(state_path).unwrap();

        let mut restored = SymbolicReasoner::new(test_config(), HashMap::new());
        restored.observe_facts(&attribute_facts());
        assert!(restored.load_dialogue_state(state_path));
        let rewrite = restored
            .contextualize_with_receipt("体重呢")
            .expect("恢复后挂账应仍可结清");
        assert_eq!(rewrite.text, "姚明的体重是什么");
        assert!(rewrite
            .notes
            .iter()
            .any(|note| note.contains("第 1 轮挂账")));
    }

    fn attribute_candidates_facts() -> Vec<PrologFact> {
        let mut yao = HashMap::new();
        yao.insert("身高".to_string(), "226cm".to_string());
        yao.insert("体重".to_string(), "140kg".to_string());
        let mut ye = HashMap::new();
        ye.insert("身高".to_string(), "190cm".to_string());
        ye.insert("体重".to_string(), "85kg".to_string());
        vec![
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
            // 已知但无体重的实体：用于「已知实体名也可作为一次显式绑定」的测试
            PrologFact::Entity {
                name: "张三".to_string(),
                entity_type: "人".to_string(),
                properties: HashMap::new(),
            },
        ]
    }

    #[test]
    fn test_frame_bare_attribute_without_context_starts_clarification() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&attribute_candidates_facts());
        assert!(
            reasoner.contextualize("体重是多少").is_none(),
            "无上下文时帧通道不应凭空改写"
        );
        let prompt = reasoner
            .start_frame_clarification("体重是多少")
            .expect("无源可填的洞应转为一次显式绑定请求");
        assert!(
            prompt.contains("叶莉") && prompt.contains("姚明"),
            "候选应来自知识库中拥有该属性的实体：{}",
            prompt
        );
        assert!(reasoner.frame_clarification_active());
        let report = reasoner.frame_memory_report();
        assert!(report.contains("等待绑定"), "{}", report);
        assert!(
            report.contains("1.叶莉；2.姚明"),
            "候选顺序应确定：{}",
            report
        );
        assert!(
            !prompt.contains("张三"),
            "无该属性且不在焦点的实体不应凭空成为候选：{}",
            prompt
        );
    }

    #[test]
    fn test_frame_clarification_binding_replays_and_remembers() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&attribute_candidates_facts());
        reasoner.contextualize("体重是多少");
        reasoner
            .start_frame_clarification("体重是多少")
            .expect("应开启澄清");
        let outcome = reasoner
            .answer_frame_clarification("2")
            .expect("序号应被确定性解析为一次绑定");
        let question = match outcome {
            dialogue_frame::ClarificationOutcome::Bound { question, note } => {
                assert!(note.contains("第 1 轮澄清"), "{}", note);
                question
            }
            other => panic!("应为绑定：{:?}", other),
        };
        assert_eq!(question, "姚明的体重是什么");
        assert!(!reasoner.frame_clarification_active(), "绑定后澄清应关闭");
        let answer = reasoner.answer(&question).expect("重放问句应可作答");
        assert!(answer.contains("140kg"), "{}", answer);

        // 记忆回写：重放的问句成为上一帧，后续省略追问自然继承
        assert!(reasoner.contextualize(&question).is_none());
        let followup = reasoner
            .contextualize_with_receipt("身高呢")
            .expect("绑定后的实体槽应可被追问继承");
        assert_eq!(followup.text, "姚明的身高是什么");
    }

    #[test]
    fn test_frame_clarification_accepts_known_entity_outside_candidates() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&attribute_candidates_facts());
        reasoner.contextualize("身高是多少");
        let prompt = reasoner
            .start_frame_clarification("身高是多少")
            .expect("应开启澄清");
        assert!(
            prompt.contains("叶莉") && prompt.contains("姚明"),
            "{}",
            prompt
        );
        // 候选表外的已知实体名（张三没有身高记录）同样接受：绑定决定权在用户
        let outcome = reasoner
            .answer_frame_clarification("张三")
            .expect("已知实体名应可作为显式绑定");
        match outcome {
            dialogue_frame::ClarificationOutcome::Bound { question, .. } => {
                assert_eq!(question, "张三的身高是什么");
            }
            other => panic!("应为绑定：{:?}", other),
        }
    }

    #[test]
    fn test_frame_clarification_reprompts_out_of_range_and_passes_natural_language() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&attribute_candidates_facts());
        reasoner.contextualize("体重是多少");
        reasoner.start_frame_clarification("体重是多少").unwrap();
        // 越界序号：重问，零状态变化
        match reasoner.answer_frame_clarification("9") {
            Some(dialogue_frame::ClarificationOutcome::Reprompt { prompt }) => {
                assert!(prompt.contains("姚明"), "{}", prompt);
            }
            other => panic!("越界序号应重问：{:?}", other),
        }
        assert!(reasoner.frame_clarification_active());
        // 自然语言非回答：放行给常规管线，澄清保持
        assert!(
            reasoner
                .answer_frame_clarification("北京是哪个省的")
                .is_none(),
            "无法确定性解析的输入应放行"
        );
        assert!(reasoner.frame_clarification_active());
        // 都不是：放弃
        match reasoner.answer_frame_clarification("都不是") {
            Some(dialogue_frame::ClarificationOutcome::Cancelled { .. }) => {}
            other => panic!("应取消：{:?}", other),
        }
        assert!(!reasoner.frame_clarification_active());
        assert!(reasoner.frame_memory().last().is_none());
    }

    #[test]
    fn test_frame_clarification_not_started_when_existing_channel_can_fill() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&attribute_candidates_facts());
        reasoner.contextualize("姚明的身高是什么");
        assert!(
            reasoner.start_frame_clarification("体重是多少").is_none(),
            "有上一帧时应走合一补全，不另起澄清"
        );
        let rewrite = reasoner
            .contextualize_with_receipt("体重是多少")
            .expect("合一应补全实体洞");
        assert_eq!(rewrite.text, "姚明的体重是什么");
        assert!(!reasoner.frame_clarification_active());
        // 「呢」省略句属于常规通道，帧澄清不得抢答
        assert!(reasoner.start_frame_clarification("体重呢").is_none());
    }

    #[test]
    fn test_frame_clarification_expires_and_persists_across_sessions() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("dialogue.json");
        let state_path = path.to_str().unwrap();

        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&attribute_candidates_facts());
        reasoner.contextualize("体重是多少");
        reasoner.start_frame_clarification("体重是多少").unwrap();
        reasoner.save_dialogue_state(state_path).unwrap();

        let mut restored = SymbolicReasoner::new(test_config(), HashMap::new());
        restored.observe_facts(&attribute_candidates_facts());
        assert!(restored.load_dialogue_state(state_path));
        assert!(restored.frame_clarification_active(), "澄清应跨会话恢复");
        match restored.answer_frame_clarification("1") {
            Some(dialogue_frame::ClarificationOutcome::Bound { question, .. }) => {
                assert_eq!(question, "叶莉的体重是什么");
            }
            other => panic!("恢复后应可绑定：{:?}", other),
        }

        // TTL：3 轮内有效，超过后过期（零残留）
        let mut expiring = SymbolicReasoner::new(test_config(), HashMap::new());
        expiring.observe_facts(&attribute_candidates_facts());
        expiring.contextualize("体重是多少");
        expiring.start_frame_clarification("体重是多少").unwrap();
        for _ in 0..3 {
            let _ = expiring.contextualize("姚明的身高是什么");
        }
        assert!(
            !expiring.frame_clarification_active(),
            "超过 TTL 的澄清应过期"
        );
        assert!(expiring.frame_memory_report().contains("过期 1"));
    }

    #[test]
    fn test_answer_generates_prediction_with_explanation() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let answer = reasoner
            .answer("李四的师祖是谁")
            .expect("应能通过组合规则预测出师祖");
        assert!(answer.contains("赵六"));
        assert!(answer.contains("师傅"));
        assert!(answer.contains("置信度"));
    }

    #[test]
    fn test_derived_rule_can_answer_without_materializing() {
        let mut config = test_config();
        // 隔离变量：让回答只可能来自会话规则的前向推导，而不是链接预测。
        config.enable_link_prediction = false;
        config.enable_association_rules = false;
        config.enable_clarification = false;
        config.enable_soft_logic = false;
        config.enable_problog = false;
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        reasoner.assert_nl_fact("张三的师傅是李四");
        reasoner.learn_rule_report("如果X的师傅是Y，那么X的师祖是Y");

        let answer = reasoner
            .answer("张三的师祖是谁")
            .expect("规则推导应能回答问题");
        assert!(answer.contains("李四"), "{}", answer);
        assert!(answer.contains("推导"), "{}", answer);
        assert!(answer.contains("未写入知识图谱"), "{}", answer);
        assert!(answer.contains("依据"), "{}", answer);
        // 确证通道也必须接受带证明的推导，而不是把它当成链接预测拒掉。
        let certain = reasoner
            .answer_certain("张三的师祖是谁")
            .expect("推导答案应进入确证通道");
        assert!(certain.contains("李四"), "{}", certain);
        // 是非问句与直接关系查询走同一份前向闭包，不能各说各话。
        let yes = reasoner
            .answer("张三的师祖是李四吗")
            .expect("是非问句应能用推导确认");
        assert!(yes.contains("是的"), "{}", yes);
        assert!(yes.contains("依据"), "{}", yes);
        let no = reasoner
            .answer("张三的师祖是王五吗")
            .expect("未知取值应诚实回答，而不是伪造确认");
        assert!(!no.contains("是的"), "{}", no);
        let objects = reasoner
            .answer_relation_objects("张三", "师祖")
            .expect("直接关系查询也应看到推导");
        assert!(objects.contains("李四"), "{}", objects);
        // 「为什么」通道应能用同一份规则/事实证明同一答案。
        let derivation = reasoner
            .answer_derivation("张三的师祖是谁", Some("李四"))
            .expect("推导答案应可解释");
        assert_eq!(derivation.answer_value(), Some("李四"));
        // 推导只是读模型：不能悄悄变成图谱边或会话事实。
        assert!(!reasoner.predictor.has_triple("张三", "师祖", "李四"));
        assert!(!reasoner
            .session_facts
            .iter()
            .any(|atom| atom.predicate.as_ref() == "师祖"));
    }

    #[test]
    fn test_derived_answer_respects_session_negative_premise() {
        let mut config = test_config();
        config.enable_link_prediction = false;
        config.enable_association_rules = false;
        config.enable_clarification = false;
        config.enable_soft_logic = false;
        config.enable_problog = false;
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        reasoner.assert_nl_fact("张三的师傅是李四");
        reasoner.assert_nl_fact("李四的师傅是王五");
        reasoner.learn_rule_report("如果X的师傅是Y，那么X的师祖是Y");
        // 否定「李四的师傅是王五」：规则仍会在前向闭包里产出中间结论，
        // 但回答层必须把这条被阻断的前提过滤掉。
        reasoner.assert_negative_atom(Atom::rel("师傅", "李四", "王五"));

        let answer = reasoner.answer("张三的师祖是谁");
        assert!(
            answer
                .as_deref()
                .map(|text| !text.contains("王五"))
                .unwrap_or(true),
            "{:?}",
            answer
        );
        let check = reasoner
            .answer("张三的师祖是王五吗")
            .expect("已知实体与谓词应给出可读回答");
        assert!(!check.contains("是的"), "{}", check);
        let report = reasoner.derived_facts_report(10);
        assert!(
            !report.contains("王五"),
            "推理发现不应展示被否定前提推出的结论：{}",
            report
        );
    }

    #[test]
    fn test_derived_answer_respects_suppressed_graph_value() {
        let mut config = test_config();
        config.enable_link_prediction = false;
        config.enable_association_rules = false;
        config.enable_clarification = false;
        config.enable_soft_logic = false;
        config.enable_problog = false;
        let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
        // 王五是图谱里的旧值；改口后它仍在图谱中，只是被会话槽压制。
        reasoner.observe_facts(&[PrologFact::Relation {
            from: "李四".to_string(),
            rel_type: "师傅".to_string(),
            to: "王五".to_string(),
        }]);
        reasoner.slots.policy_mut().declare_functional("师傅", true);
        reasoner.assert_nl_fact("张三的师傅是李四");
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

        let before = reasoner.answer("张三的师祖是谁").expect("应能推出王五");
        assert!(before.contains("王五"), "{}", before);

        reasoner.correct_nl_fact("改口：李四的师傅是赵六");
        let after = reasoner
            .answer("张三的师祖是谁")
            .expect("改口后应能推出新值");
        assert!(after.contains("赵六"), "{}", after);
        assert!(
            !after.contains("王五"),
            "被槽账本压制的图谱旧值不得重新进入推导：{}",
            after
        );
        // 图谱本身没有被推导或改口改写。
        assert!(reasoner.predictor.has_triple("李四", "师傅", "王五"));
        let report = reasoner.derived_facts_report(10);
        assert!(
            !report.contains("王五"),
            "推理发现不应展示被槽压制的旧前提推出的结论：{}",
            report
        );
        assert!(report.contains("赵六"), "{}", report);
    }

    #[test]
    fn semantic_monitor_records_answer_without_question_text() {
        let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
        reasoner.observe_facts(&master_facts());
        let question = "张三的师傅是谁";
        let answer = reasoner.answer(question);
        assert!(answer.is_some(), "测试事实应可回答");
        let metrics = reasoner.semantic_metrics();
        assert_eq!(metrics.observations, 1);
        assert!(metrics.answered.bps.is_some());
        let observations = reasoner.semantic_observations();
        assert_eq!(observations.len(), 1);
        let json = serde_json::to_string(&observations).unwrap();
        assert!(!json.contains(question), "观测不得保存原始问句：{}", json);
        assert!(reasoner.semantic_report().contains("语义检索监控"));
        assert_eq!(reasoner.perf.count("semantic_observations"), 1);
        assert_eq!(reasoner.perf.count("semantic_outcome_answered"), 1);
    }

    #[test]
    fn semantic_control_mode_and_budget_are_configurable() {
        let mut config = test_config();
        config.semantic_monitor.mode = crate::semantic_monitor::ControlMode::Control;
        config.semantic_monitor.budget.initial = 0;
        let reasoner = SymbolicReasoner::new(config, HashMap::new());
        let advice = reasoner.semantic_advice();
        assert_eq!(advice.level, crate::semantic_monitor::PressureLevel::Strict);
        assert_eq!(
            advice.max_items,
            Some(crate::semantic_monitor::STRICT_MAX_ITEMS)
        );
        assert!(advice.prefer_clarification);
    }

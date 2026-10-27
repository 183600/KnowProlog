    use super::*;
    use crate::contextual_reference::SenseCard;
    use std::collections::BTreeMap;

    struct MockKnowledge {
        entities: Vec<String>,
        known: Vec<String>,
        attributes: BTreeMap<String, Vec<String>>,
        answers: BTreeMap<String, String>,
        values: BTreeMap<(String, String), Vec<String>>,
        profiles: BTreeMap<String, String>,
        derivations: BTreeMap<String, Derivation>,
        nominations: BTreeMap<(String, String), Vec<RecoveryNomination>>,
        learned: BTreeMap<String, String>,
    }

    impl MockKnowledge {
        fn new() -> Self {
            MockKnowledge {
                entities: Vec::new(),
                known: Vec::new(),
                attributes: BTreeMap::new(),
                answers: BTreeMap::new(),
                values: BTreeMap::new(),
                profiles: BTreeMap::new(),
                derivations: BTreeMap::new(),
                nominations: BTreeMap::new(),
                learned: BTreeMap::new(),
            }
        }

        fn with_learned_rewrite(mut self, question: &str, answer: &str) -> Self {
            self.learned
                .insert(question.to_string(), answer.to_string());
            self
        }

        fn with_nomination(
            mut self,
            entity: &str,
            attribute: &str,
            value: &str,
            score: f64,
            evidence: &str,
        ) -> Self {
            self.nominations
                .entry((entity.to_string(), attribute.to_string()))
                .or_default()
                .push(RecoveryNomination {
                    value: value.to_string(),
                    score,
                    evidence: vec![evidence.to_string()],
                });
            self
        }

        fn with_value(mut self, entity: &str, attribute: &str, value: &str) -> Self {
            self.values.insert(
                (entity.to_string(), attribute.to_string()),
                vec![value.to_string()],
            );
            self
        }

        fn with_profile(mut self, entity: &str, profile: &str) -> Self {
            self.profiles
                .insert(entity.to_string(), profile.to_string());
            self
        }

        fn with_entity(mut self, entity: &str, attributes: &[&str]) -> Self {
            self.entities.push(entity.to_string());
            self.known.push(entity.to_string());
            self.attributes.insert(
                entity.to_string(),
                attributes.iter().map(|a| a.to_string()).collect(),
            );
            self
        }

        fn with_answer(mut self, question: &str, answer: &str) -> Self {
            self.answers
                .insert(question.to_string(), answer.to_string());
            self
        }

        fn with_derivation(mut self, question: &str, derivation: Derivation) -> Self {
            self.derivations.insert(question.to_string(), derivation);
            self
        }
    }

    impl DialogueKnowledge for MockKnowledge {
        fn entities_in(&self, text: &str) -> Vec<String> {
            let mut found: Vec<String> = self
                .entities
                .iter()
                .filter(|entity| text.contains(entity.as_str()))
                .cloned()
                .collect();
            found.sort_by_key(|entity| std::cmp::Reverse(entity.chars().count()));
            found
        }

        fn known_entity(&self, entity: &str) -> bool {
            self.known.iter().any(|known| known == entity)
        }

        fn answer(&self, question: &str) -> Option<String> {
            self.answers.get(question).cloned()
        }

        fn learned_rewrite_answer(&self, question: &str) -> Option<String> {
            self.learned.get(question).cloned()
        }

        fn derivation(&self, question: &str, expected: Option<&str>) -> Option<Derivation> {
            let derivation = self.derivations.get(question)?;
            if let Some(expected) = expected {
                if let Some(value) = derivation.answer_value() {
                    if !expected.contains(value) {
                        return None;
                    }
                }
            }
            Some(derivation.clone())
        }

        fn known_attributes(&self, entity: &str) -> Vec<String> {
            self.attributes.get(entity).cloned().unwrap_or_default()
        }

        fn attribute_values(&self, entity: &str, attribute: &str) -> Vec<String> {
            self.values
                .get(&(entity.to_string(), attribute.to_string()))
                .cloned()
                .unwrap_or_default()
        }

        fn all_entities(&self) -> Vec<String> {
            let mut entities = self.entities.clone();
            entities.sort();
            entities.dedup();
            entities
        }

        fn describe(&self, entity: &str) -> Option<String> {
            self.profiles.get(entity).cloned()
        }

        fn recovery_nominations(&self, entity: &str, attribute: &str) -> Vec<RecoveryNomination> {
            self.nominations
                .get(&(entity.to_string(), attribute.to_string()))
                .cloned()
                .unwrap_or_default()
        }
    }

    fn mock() -> MockKnowledge {
        MockKnowledge::new()
            .with_entity("姚明", &["身高", "国籍", "出生地"])
            .with_answer("姚明是谁", "姚明是中国篮球运动员。")
            .with_answer("姚明的身高是什么", "姚明的身高是 226 厘米。")
            .with_answer("姚明的国籍是什么", "姚明的国籍是中国。")
    }

    #[test]
    fn learned_rewrite_answer_precedes_profile_generation() {
        let knowledge = MockKnowledge::new()
            .with_entity("张三", &["师傅"])
            .with_answer("张三到底师傅是谁", "关于「张三」的知识（类型：人物）")
            .with_learned_rewrite("张三到底师傅是谁", "李四");
        let mut manager = DialogueManager::new();
        let reply = manager
            .plan_reply("张三到底师傅是谁", &knowledge)
            .expect("习得重写应接管本轮");
        assert_eq!(reply.text, "李四");
        assert_eq!(reply.plan, vec!["learned_rewrite"]);
        assert_eq!(manager.focus_entity(), Some("张三"));
        assert_eq!(manager.turn_count(), 1);
    }

    struct CrdKnowledge {
        base: MockKnowledge,
        linker: EntityLinker,
        cards: BTreeMap<String, SenseCard>,
    }

    impl CrdKnowledge {
        fn apple() -> Self {
            let mut linker = EntityLinker::new();
            linker.register_entity("苹果", Some("水果"));
            linker.register_entity("苹果公司", Some("组织"));
            linker.register_alias("苹果", "苹果公司");
            let mut cards = BTreeMap::new();
            cards.insert(
                "苹果".to_string(),
                SenseCard {
                    entity: "苹果".to_string(),
                    etype: Some("水果".to_string()),
                    ..SenseCard::default()
                },
            );
            cards.insert(
                "苹果公司".to_string(),
                SenseCard {
                    entity: "苹果公司".to_string(),
                    etype: Some("组织".to_string()),
                    ..SenseCard::default()
                },
            );
            let base = MockKnowledge::new()
                .with_entity("苹果", &[])
                .with_entity("苹果公司", &["创始人"])
                .with_answer("苹果公司是什么", "苹果公司是一家组织。")
                .with_answer("苹果公司的创始人是什么", "苹果公司的创始人是乔布斯。");
            CrdKnowledge {
                base,
                linker,
                cards,
            }
        }
    }

    impl SenseSource for CrdKnowledge {
        fn sense_card(&self, entity: &str) -> Option<SenseCard> {
            self.cards.get(entity).cloned()
        }
    }

    impl DialogueKnowledge for CrdKnowledge {
        fn entities_in(&self, text: &str) -> Vec<String> {
            self.base.entities_in(text)
        }

        fn known_entity(&self, entity: &str) -> bool {
            self.base.known_entity(entity)
        }

        fn answer(&self, question: &str) -> Option<String> {
            self.base.answer(question)
        }

        fn known_attributes(&self, entity: &str) -> Vec<String> {
            self.base.known_attributes(entity)
        }

        fn attribute_values(&self, entity: &str, attribute: &str) -> Vec<String> {
            self.base.attribute_values(entity, attribute)
        }

        fn describe(&self, entity: &str) -> Option<String> {
            self.base.describe(entity)
        }

        fn reference_view(&self) -> Option<ReferenceView<'_>> {
            Some(ReferenceView {
                linker: &self.linker,
                source: self,
            })
        }
    }

    #[test]
    fn reference_clarification_binds_and_is_reused() {
        let knowledge = CrdKnowledge::apple();
        let mut manager = DialogueManager::new();
        let clarify = manager
            .plan_reply("苹果是什么", &knowledge)
            .expect("两个义项应升级为澄清");
        assert_eq!(clarify.plan, vec!["clarify_reference".to_string()]);
        assert!(clarify.text.contains("苹果公司"), "{}", clarify.text);
        assert!(manager.pending_reference().is_some());

        let reply = manager
            .plan_reply("2", &knowledge)
            .expect("澄清回答应绑定并继续原问句");
        assert!(reply.text.contains("组织"), "{}", reply.text);
        assert_eq!(
            manager
                .reference_ledger()
                .lookup("苹果")
                .map(|binding| binding.entity.as_str()),
            Some("苹果公司")
        );

        let reply = manager
            .plan_reply("苹果的创始人是谁", &knowledge)
            .expect("绑定后的词面应直接复用");
        assert!(reply.text.contains("乔布斯"), "{}", reply.text);
        assert!(manager.pending_reference().is_none());

        let snapshot = manager.snapshot();
        assert!(!snapshot.references.is_empty());
        let mut restored = DialogueManager::new();
        restored.restore(&snapshot);
        assert_eq!(
            restored
                .reference_ledger()
                .lookup("苹果")
                .map(|binding| binding.entity.as_str()),
            Some("苹果公司")
        );
    }

    #[test]
    fn test_policy_source_is_stratified() {
        let mut program = LogicProgram::from_source(POLICY_SOURCE).expect("策略程序应可解析");
        program.evaluate().expect("策略程序应可求值");
        assert!(program.rule_count() >= 10);
    }

    #[test]
    fn test_analyze_attribute_question() {
        let manager = DialogueManager::new();
        let frame = manager.analyze("姚明的身高是多少？", &mock());
        assert_eq!(frame.kind, MoveKind::AskAttribute);
        assert_eq!(frame.entity.as_deref(), Some("姚明"));
        assert_eq!(frame.attribute.as_deref(), Some("身高"));
    }

    #[test]
    fn copula_relation_question_is_binary_not_attribute() {
        let knowledge = mock();
        let mut manager = DialogueManager::new();
        let frame = manager.analyze("姚明是叶莉的师傅吗", &knowledge);
        assert_eq!(frame.kind, MoveKind::AskBinary, "{:?}", frame);
        assert_eq!(
            frame.question.as_deref(),
            Some("姚明是叶莉的师傅吗"),
            "是非问必须原样交给蕴含管线，不能改写成属性查询"
        );
        let statement = manager.analyze("姚明是叶莉的师傅", &knowledge);
        assert_eq!(statement.kind, MoveKind::Assert, "{:?}", statement);
        assert!(
            manager.plan_reply("姚明是叶莉的师傅", &knowledge).is_none(),
            "查询板不得把系表判断重扫成「叶莉的师傅」"
        );
    }

    #[test]
    fn test_analyze_attribute_question_keeps_explicit_attribute_before_degree_word() {
        let knowledge = mock();
        let manager = DialogueManager::new();
        for (input, attribute) in [
            ("姚明的身高有多高？", "身高"),
            ("姚明的身高多高", "身高"),
            ("姚明的年龄多大？", "年龄"),
            ("姚明的出生地是哪里？", "出生地"),
            ("姚明的师傅有哪些？", "师傅"),
        ] {
            let frame = manager.analyze(input, &knowledge);
            assert_eq!(frame.kind, MoveKind::AskAttribute, "{}", input);
            assert_eq!(frame.entity.as_deref(), Some("姚明"), "{}", input);
            assert_eq!(frame.attribute.as_deref(), Some(attribute), "{}", input);
        }
    }

    #[test]
    fn test_analyze_attribute_question_ignores_emphasis_fillers() {
        let knowledge = mock();
        let manager = DialogueManager::new();
        for input in ["姚明的身高到底是多少？", "姚明的身高究竟是多少呢？"] {
            let frame = manager.analyze(input, &knowledge);
            assert_eq!(frame.kind, MoveKind::AskAttribute, "{}", input);
            assert_eq!(frame.entity.as_deref(), Some("姚明"), "{}", input);
            assert_eq!(frame.attribute.as_deref(), Some("身高"), "{}", input);
        }

        let mut manager = DialogueManager::new();
        let reply = manager
            .plan_reply("姚明的身高到底是多少呢？", &knowledge)
            .expect("填充词不应阻断属性问句");
        assert!(reply.text.contains("226"), "{}", reply.text);
    }

    #[test]
    fn test_note_plugin_turn_updates_focus_and_conversation_body() {
        let mut manager = DialogueManager::new();
        manager.note_plugin_turn(
            "阿凡达是谁导演的",
            MoveKind::AskAttribute,
            Some("阿凡达"),
            Some("导演"),
            "「阿凡达」的导演是詹姆斯·卡梅隆。",
            vec!["plugin:zh.film:film.directed_by".to_string()],
            vec!["插件 zh.film / 配方 film.directed_by".to_string()],
        );
        assert_eq!(manager.turn_count(), 1);
        assert_eq!(manager.focus_entity(), Some("阿凡达"));
        let turn = &manager.turns()[0];
        assert_eq!(turn.entity.as_deref(), Some("阿凡达"));
        assert_eq!(turn.attribute.as_deref(), Some("导演"));
        assert!(turn
            .answer_excerpt
            .as_deref()
            .unwrap_or_default()
            .contains("詹姆斯"));
        assert_eq!(
            manager.last_reply_plan(),
            &["plugin:zh.film:film.directed_by".to_string()]
        );
        assert_eq!(
            manager.analyze("还有呢", &mock()).kind,
            MoveKind::ContinueRequest
        );
    }

    #[test]
    fn test_analyze_meta_and_smalltalk() {
        let manager = DialogueManager::new();
        let knowledge = mock();
        assert_eq!(
            manager.analyze("我们聊了什么？", &knowledge).kind,
            MoveKind::MetaRecap
        );
        assert_eq!(
            manager.analyze("你怎么知道的？", &knowledge).kind,
            MoveKind::MetaWhy
        );
        assert_eq!(manager.analyze("你好", &knowledge).kind, MoveKind::Greet);
        assert_eq!(manager.analyze("谢谢！", &knowledge).kind, MoveKind::Thanks);
        assert_eq!(
            manager.analyze("还有呢", &knowledge).kind,
            MoveKind::ContinueRequest
        );
    }

    #[test]
    fn test_direct_answer_plan_and_followup_offer() {
        let knowledge = mock();
        let mut manager = DialogueManager::new();
        let reply = manager
            .plan_reply("姚明的身高是多少", &knowledge)
            .expect("已知实体且有答案时应接管");
        assert!(
            reply.plan.iter().any(|plan| plan == "direct_answer"),
            "策略：{:?}",
            reply.plan
        );
        assert!(reply.plan.iter().any(|plan| plan == "offer_related"));
        assert!(reply.text.contains("226"));
        assert!(
            reply.text.contains("国籍"),
            "应附带追问建议：{}",
            reply.text
        );
        assert_eq!(manager.turn_count(), 1);
        assert_eq!(manager.focus_entity(), Some("姚明"));
    }

    #[test]
    fn test_admit_ignorance_with_suggestions() {
        let knowledge = mock();
        let mut manager = DialogueManager::new();
        let reply = manager
            .plan_reply("姚明的体重是多少", &knowledge)
            .expect("已知实体即使无答案也应接管（生成式坦白）");
        assert!(
            reply.plan.iter().any(|plan| plan == "admit_ignorance"),
            "策略：{:?}",
            reply.plan
        );
        assert!(reply.plan.iter().any(|plan| plan == "ask_back"));
        assert!(reply.text.contains("没有记录"));
        assert!(reply.text.contains("身高") && reply.text.contains("国籍"));
    }

    #[test]
    fn test_unknown_entity_defers_to_other_pipelines() {
        let knowledge = mock();
        let mut manager = DialogueManager::new();
        assert!(manager
            .plan_reply("诸葛亮的身高是多少", &knowledge)
            .is_none());
        assert_eq!(manager.turn_count(), 0);
    }

    #[test]
    fn test_epistemic_followup_keeps_mode_and_predicate() {
        let knowledge = MockKnowledge::new()
            .with_entity("李四", &[])
            .with_entity("王五", &[])
            .with_entity("张三", &[])
            .with_answer("王五的师傅可能是谁", "PQI：王五的师傅概率")
            .with_answer("张三的师傅可能是谁", "PQI：张三的师傅概率");
        let mut manager = DialogueManager::new();
        assert!(manager
            .plan_pure_dialogue("李四的师傅可能是谁", &knowledge)
            .is_none());
        let reply = manager
            .plan_pure_dialogue("那王五呢", &knowledge)
            .expect("省略追问应承接上一轮情态与谓词");
        assert_eq!(reply.kind, MoveKind::AskBinary);
        assert!(
            reply.plan.iter().any(|plan| plan == "epistemic_followup"),
            "策略：{:?}",
            reply.plan
        );
        assert!(reply.text.contains("PQI"), "{}", reply.text);
        assert_eq!(manager.turn_count(), 1);
        assert_eq!(manager.focus_entity(), Some("王五"));

        // 情态记忆属于可观察对话状态：快照恢复后仍能接住下一条省略追问。
        let snapshot = manager.snapshot();
        let mut restored = DialogueManager::new();
        restored.restore(&snapshot);
        let reply = restored
            .plan_pure_dialogue("那张三呢", &knowledge)
            .expect("恢复后仍应承接情态");
        assert!(reply.text.contains("PQI"), "{}", reply.text);
    }

    #[test]
    fn test_epistemic_followup_does_not_hijack_new_questions() {
        let knowledge = MockKnowledge::new()
            .with_entity("李四", &[])
            .with_entity("王五", &[])
            .with_answer("王五的师傅可能是谁", "PQI：王五的师傅概率");
        let mut manager = DialogueManager::new();
        // 没有上一轮情态记忆时，「那王五呢」仍是普通省略问句。
        assert!(manager.plan_pure_dialogue("那王五呢", &knowledge).is_none());
        assert_eq!(manager.turn_count(), 0);

        assert!(manager
            .plan_pure_dialogue("李四的师傅可能是谁", &knowledge)
            .is_none());
        // 裸「王五呢」没有被「那」标记，不抢普通省略/身份问句。
        assert!(manager.plan_pure_dialogue("王五呢", &knowledge).is_none());
        // 含「的」的追问不是纯实体省略，不能套用上一轮谓词。
        assert!(manager
            .plan_pure_dialogue("那王五的师傅呢", &knowledge)
            .is_none());
        assert_eq!(manager.turn_count(), 0);
    }

    #[test]
    fn test_pronoun_and_ellipsis_followup() {
        let knowledge = mock();
        let mut manager = DialogueManager::new();
        manager
            .plan_reply("姚明是谁", &knowledge)
            .expect("身份问句应可回答");
        let frame = manager.analyze("他的身高呢", &knowledge);
        assert_eq!(frame.kind, MoveKind::AskAttribute);
        assert_eq!(frame.entity.as_deref(), Some("姚明"));
        assert!(frame.inherited);
        let reply = manager
            .plan_reply("他的身高呢", &knowledge)
            .expect("指代消解后应能回答");
        assert!(reply.text.contains("226"));
    }

    #[test]
    fn test_recap_and_explain_why() {
        let knowledge = mock();
        let mut manager = DialogueManager::new();
        manager.plan_reply("姚明的身高是多少", &knowledge).unwrap();
        let recap = manager
            .plan_pure_dialogue("我们聊了什么？", &knowledge)
            .expect("回顾元话语应被接管");
        assert!(recap.text.contains("聊了 1 轮"), "{}", recap.text);
        assert!(recap.text.contains("姚明"));
        let why = manager
            .plan_pure_dialogue("你怎么知道的？", &knowledge)
            .expect("依据元话语应被接管");
        assert!(why.text.contains("依据"));
        assert!(why.text.contains("知识图谱"));
    }

    #[test]
    fn test_recap_window_keeps_focus_relevant_old_turn() {
        let mut manager = DialogueManager::new();
        manager.note_plugin_turn(
            "姚明的身高是多少",
            MoveKind::AskAttribute,
            Some("姚明"),
            Some("身高"),
            "226 厘米",
            Vec::new(),
            Vec::new(),
        );
        for (input, answer) in [
            ("今天天气怎么样", "不知道"),
            ("随便聊聊", "好"),
            ("讲个笑话", "好"),
            ("谢谢", "不客气"),
        ] {
            manager.note_plugin_turn(
                input,
                MoveKind::Other,
                None,
                None,
                answer,
                Vec::new(),
                Vec::new(),
            );
        }
        manager.note_plugin_focus("姚明");
        let window = manager.recap_window(WindowBudget::new(3, 240, 1));
        assert!(window.keep(1), "焦点相关的旧轮应进入窗口");
        assert!(!window.keep(2), "{:?}", window);
        assert!(!window.keep(3), "{:?}", window);
        assert!(window.keep(4));
        assert!(window.keep(5));
        let recap = manager.recap_text();
        assert!(recap.contains("第 1 轮"), "{}", recap);
        assert!(recap.contains("姚明"), "{}", recap);
    }

    #[test]
    fn context_packet_merges_external_evidence_without_mutating_state() {
        let mut manager = DialogueManager::new();
        manager.note_plugin_focus("姚\u{200B}明");
        let evidence = AttentionItem::new(
            "shared:claim:1",
            AttentionKind::Evidence,
            "姚明的身高是 226cm（来源：seed）",
        )
        .with_entities(["姚明"])
        .with_predicates(["身高"])
        .trust(65);
        let packet = manager.context_packet_with(AttentionBudget::default(), [evidence]);
        let shared = packet
            .items
            .iter()
            .find(|item| item.id == "shared:claim:1")
            .expect("外部证据必须能进入上下文包");
        assert!(shared
            .reasons
            .contains(&crate::working_memory::AttentionReason::FocusMatch));
        assert!(!manager
            .context_packet(AttentionBudget::default())
            .items
            .iter()
            .any(|item| item.id.starts_with("shared:")));
    }

    #[test]
    fn context_packet_prioritizes_focus_and_open_questions() {
        let mut manager = DialogueManager::new();
        manager.note_plugin_turn(
            "姚明的身高是多少",
            MoveKind::AskAttribute,
            Some("姚明"),
            Some("身高"),
            "226 厘米",
            Vec::new(),
            Vec::new(),
        );
        for (input, answer) in [("今天天气怎么样", "不知道"), ("谢谢", "不客气")] {
            manager.note_plugin_turn(
                input,
                MoveKind::Other,
                None,
                None,
                answer,
                Vec::new(),
                Vec::new(),
            );
        }
        manager.note_plugin_focus("姚\u{200B}明");
        let fault = crate::turn_runtime::TurnFault::new(
            crate::turn_runtime::TurnFaultKind::Deadline,
            crate::turn_runtime::TurnPhase::Reason,
            "超时",
            0,
        );
        manager.record_open_turn("叶莉的国籍是什么", 9, &fault, Some("叶莉"), Some("国籍"));

        let packet = manager.context_packet(AttentionBudget::default());
        let focused = packet
            .items
            .iter()
            .find(|item| item.id == "turn:1")
            .expect("焦点相关旧轮必须进入上下文包");
        assert!(
            focused.reasons.contains(&crate::working_memory::AttentionReason::FocusMatch),
            "{:?}",
            focused
        );
        assert!(
            packet
                .items
                .iter()
                .any(|item| item.kind == AttentionKind::Focus),
            "{:?}",
            packet
        );
        let open = packet
            .items
            .iter()
            .find(|item| item.id == "open:1")
            .expect("未完成问题必须进入上下文包");
        assert!(open.reasons.contains(&crate::working_memory::AttentionReason::OpenQuestion));
        assert!(open.reasons.contains(&crate::working_memory::AttentionReason::Pinned));
        assert!(packet
            .render_receipt(AttentionBudget::default())
            .contains("工作记忆"));
    }

    #[test]
    fn colloquial_name_questions_are_left_to_the_query_planner() {
        for question in [
            "张三的师傅叫什么名字",
            "张三的师傅叫啥名字",
            "张三叫什么名字",
            "张三叫啥",
        ] {
            assert!(
                split_entity_attribute(question).is_none(),
                "{} 不应被对话层拆成实体+属性",
                question
            );
        }
        assert_eq!(
            split_entity_attribute("张三的姓名是什么"),
            Some(("张三", "姓名"))
        );
    }

    #[test]
    fn test_explain_why_rebuilds_a_real_derivation() {
        let derivation = Derivation::from_direct_facts(
            "姚明的身高",
            vec![(
                "姚明".to_string(),
                "身高".to_string(),
                "226 厘米".to_string(),
            )],
        );
        let knowledge = MockKnowledge::new()
            .with_entity("姚明", &["身高"])
            .with_answer("姚明的身高是什么", "姚明的身高是 226 厘米。")
            .with_derivation("姚明的身高是什么", derivation);
        let mut manager = DialogueManager::new();
        manager
            .plan_reply("姚明的身高是多少", &knowledge)
            .expect("属性问句应被接管");
        let why = manager
            .plan_pure_dialogue("怎么推出来的？", &knowledge)
            .expect("推导追问应被接管");
        assert!(why.text.contains("证明步骤"), "{}", why.text);
        assert!(why.text.contains("226 厘米"), "{}", why.text);
        assert!(why.text.contains("事实"), "{}", why.text);
        assert!(why.text.contains("出处标签"), "{}", why.text);
    }

    #[test]
    fn test_step_followup_explains_one_proof_node() {
        let derivation = Derivation::from_direct_facts(
            "姚明的身高",
            vec![
                (
                    "姚明".to_string(),
                    "身高".to_string(),
                    "226 厘米".to_string(),
                ),
                (
                    "姚明".to_string(),
                    "体重".to_string(),
                    "140 公斤".to_string(),
                ),
            ],
        );
        let knowledge = MockKnowledge::new()
            .with_entity("姚明", &["身高"])
            .with_answer("姚明的身高是什么", "姚明的身高是 226 厘米。")
            .with_derivation("姚明的身高是什么", derivation);
        let mut manager = DialogueManager::new();
        manager
            .plan_reply("姚明的身高是多少", &knowledge)
            .expect("属性问句应被接管");
        let step = manager
            .plan_pure_dialogue("第二步是什么？", &knowledge)
            .expect("步骤追问应被接管");
        assert!(step.text.contains("第 2 步"), "{}", step.text);
        assert!(step.text.contains("140 公斤"), "{}", step.text);
        assert!(step.text.contains("完整证明共 2 步"), "{}", step.text);

        let out_of_range = manager
            .plan_pure_dialogue("第九步呢？", &knowledge)
            .expect("越界步骤也应给出可行动作");
        assert!(
            out_of_range.text.contains("没有第 9 步"),
            "{}",
            out_of_range.text
        );
    }

    #[test]
    fn test_explain_why_falls_back_to_direct_fact_edges() {
        let knowledge = MockKnowledge::new()
            .with_entity("姚明", &["身高"])
            .with_value("姚明", "身高", "226 厘米")
            .with_answer("姚明的身高是什么", "姚明的身高是 226 厘米。");
        let mut manager = DialogueManager::new();
        manager
            .plan_reply("姚明的身高是多少", &knowledge)
            .expect("属性问句应被接管");
        let why = manager
            .plan_pure_dialogue("依据是什么？", &knowledge)
            .expect("依据追问应被接管");
        assert!(why.text.contains("证明步骤"), "{}", why.text);
        assert!(
            why.text.contains("姚明 的 身高 是 226 厘米"),
            "{}",
            why.text
        );
    }

    #[test]
    fn test_explain_why_not_uses_failure_frontier_without_prior_answer() {
        let knowledge = MockKnowledge::new()
            .with_entity("姚明", &["身高"])
            .with_answer("姚明的身高是什么", "姚明的身高是 226 厘米。")
            .with_derivation(
                "姚明的体重是什么",
                Derivation::missing(
                    "姚明的体重",
                    vec!["体重(姚明, ?X)（缺少事实或规则，深度 0）".to_string()],
                ),
            );
        let mut manager = DialogueManager::new();
        let reply = manager
            .plan_reply("姚明的体重是多少", &knowledge)
            .expect("已知实体即使无答案也应坦白");
        assert!(reply.text.contains("没有记录"), "{}", reply.text);
        let why = manager
            .plan_pure_dialogue("为什么没有记录？", &knowledge)
            .expect("失败追问应被接管");
        assert!(why.text.contains("没有作答"), "{}", why.text);
        assert!(why.text.contains("体重"), "{}", why.text);
        assert!(why.text.contains("知识缺口"), "{}", why.text);
        assert!(why.text.contains("不是「它不存在」"), "{}", why.text);
    }

    #[test]
    fn test_continue_topic_picks_unasked_attribute() {
        let knowledge = mock();
        let mut manager = DialogueManager::new();
        manager.plan_reply("姚明的身高是多少", &knowledge).unwrap();
        let reply = manager
            .plan_pure_dialogue("还有呢", &knowledge)
            .expect("续聊应被接管");
        assert!(
            reply.plan.iter().any(|plan| plan == "continue_topic"),
            "{:?}",
            reply.plan
        );
        assert!(
            reply.text.contains("国籍"),
            "应挑未问过的属性：{}",
            reply.text
        );
    }

    #[test]
    fn test_continue_without_focus() {
        let knowledge = mock();
        let mut manager = DialogueManager::new();
        let reply = manager
            .plan_pure_dialogue("还有呢", &knowledge)
            .expect("无焦点时也应给出可行动作");
        assert!(reply.plan.iter().any(|plan| plan == "continue_no_focus"));
        assert!(reply.text.contains("实体"));
    }

    #[test]
    fn test_structured_fallbacks_fill_missing_answers() {
        let knowledge = MockKnowledge::new()
            .with_entity("姚明", &["身高", "国籍"])
            .with_value("姚明", "国籍", "中国")
            .with_profile("姚明", "姚明是中国篮球运动员。");
        let mut manager = DialogueManager::new();
        let reply = manager
            .plan_reply("姚明的国籍是什么", &knowledge)
            .expect("属性取值检索应兜底作答");
        assert!(reply.plan.iter().any(|plan| plan == "direct_answer"));
        assert!(reply.text.contains("中国"));
        let mut manager = DialogueManager::new();
        let reply = manager
            .plan_reply("姚明是谁", &knowledge)
            .expect("身份问句应回落到画像生成");
        assert!(reply.text.contains("中国篮球运动员"));
    }

    #[test]
    fn test_attribute_without_any_entity_asks_which_entity() {
        let knowledge = mock();
        let mut manager = DialogueManager::new();
        let reply = manager
            .plan_reply("首都呢", &knowledge)
            .expect("缺实体时应生成澄清问题而非放弃");
        assert!(
            reply
                .plan
                .iter()
                .any(|plan| plan == "clarify_entity_no_focus"),
            "策略：{:?}",
            reply.plan
        );
        assert!(reply.text.contains("首都"));
        assert!(reply.text.contains("实体名"));
    }

    #[test]
    fn test_snapshot_roundtrip() {
        let knowledge = mock();
        let mut manager = DialogueManager::new();
        manager.plan_reply("姚明是谁", &knowledge).unwrap();
        let path = std::env::temp_dir().join(format!(
            "knowprolog_dialogue_manager_{}.json",
            std::process::id()
        ));
        let path_str = path.to_string_lossy().to_string();
        manager.save_to_path(&path_str).expect("应能保存");
        let mut restored = DialogueManager::new();
        assert!(restored.restore_from_path(&path_str), "应能恢复");
        assert_eq!(restored.turn_count(), 1);
        assert_eq!(restored.focus_entity(), Some("姚明"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_snapshot_roundtrip_writes_canonical_bytes_with_receipt() {
        let knowledge = mock();
        let mut manager = DialogueManager::new();
        manager.plan_reply("姚明是谁", &knowledge).unwrap();
        let path = std::env::temp_dir().join(format!(
            "knowprolog_dialogue_manager_canonical_{}.json",
            std::process::id()
        ));
        let path_str = path.to_string_lossy().to_string();
        manager.save_to_path(&path_str).expect("应能保存");
        let bytes = std::fs::read(&path).unwrap();
        let snapshot = manager.snapshot();
        assert_eq!(
            crate::canonical::canonical_bytes(&snapshot).unwrap(),
            bytes,
            "落盘必须是规范字节（L2 字节不动点）"
        );

        let mut restored = DialogueManager::new();
        let receipt = restored
            .restore_from_path_report(&path_str)
            .expect("应能恢复");
        assert!(receipt.canonical);
        assert_eq!(receipt.kind, "dialogue_manager");
        assert_eq!(restored.snapshot(), snapshot);

        // 失败都带稳定错误码：坏字节/缺失文件分类明确
        let bad = std::env::temp_dir().join(format!(
            "knowprolog_dialogue_manager_bad_{}.json",
            std::process::id()
        ));
        std::fs::write(&bad, b"{broken").unwrap();
        let error = DialogueManager::new()
            .restore_from_path_report(bad.to_str().unwrap())
            .unwrap_err();
        assert_eq!(error.code(), "parse");
        let error = DialogueManager::new()
            .restore_from_path_report("/nonexistent/definitely/not/here.json")
            .unwrap_err();
        assert_eq!(error.code(), "io");
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&bad);
    }

    #[test]
    fn test_snapshot_preserves_observable_state() {
        let knowledge = mock();
        let mut manager = DialogueManager::new();
        manager.plan_reply("姚明的身高是多少", &knowledge).unwrap();
        manager.plan_pure_dialogue("还有呢", &knowledge).unwrap();
        let snapshot = manager.snapshot();
        let mut restored = DialogueManager::new();
        restored.restore(&snapshot);
        assert_eq!(restored.snapshot(), snapshot);
        assert_eq!(restored.summary(), manager.summary());
        assert_eq!(restored.last_reply_plan(), manager.last_reply_plan());
    }

    #[test]
    fn test_snapshot_deserializes_without_last_reply_field() {
        let stored = r#"{
            "turns": [],
            "focus": ["姚明"],
            "externals": []
        }"#;
        let snapshot: DialogueManagerSnapshot =
            serde_json::from_str(stored).expect("缺少 last_reply 字段的快照应可反序列化");
        assert!(snapshot.last_reply.is_empty());
        let mut manager = DialogueManager::new();
        manager.restore(&snapshot);
        assert_eq!(manager.focus_entity(), Some("姚明"));
    }

    #[test]
    fn test_external_inputs_are_tracked_and_consumed() {
        let knowledge = mock();
        let mut manager = DialogueManager::new();
        manager.observe_input("你好");
        manager.observe_input("谢谢");
        manager.plan_pure_dialogue("谢谢", &knowledge).unwrap();
        let summary = manager.summary();
        assert!(summary.contains("外部指令 1 条"), "{}", summary);
    }

    #[test]
    fn test_polite_replies_use_focus() {
        let knowledge = mock();
        let mut manager = DialogueManager::new();
        manager.plan_reply("姚明是谁", &knowledge).unwrap();
        let thanks = manager.plan_pure_dialogue("谢谢！", &knowledge).unwrap();
        assert!(thanks.text.contains("姚明"));
        let farewell = manager.plan_pure_dialogue("再见", &knowledge).unwrap();
        assert!(farewell.text.contains("1 轮"));
    }

    #[test]
    fn test_clarify_entity_with_multiple_focus() {
        let knowledge = MockKnowledge::new()
            .with_entity("姚明", &["身高"])
            .with_entity("叶莉", &["身高"])
            .with_answer("姚明是谁", "姚明是中国篮球运动员。")
            .with_answer("叶莉是谁", "叶莉是中国篮球运动员。");
        let mut manager = DialogueManager::new();
        manager.plan_reply("姚明是谁", &knowledge).unwrap();
        manager.plan_reply("叶莉是谁", &knowledge).unwrap();
        let reply = manager
            .plan_reply("身高呢", &knowledge)
            .expect("属性问句缺实体时应触发消歧");
        assert!(
            reply.plan.iter().any(|plan| plan == "clarify_entity"),
            "策略：{:?}",
            reply.plan
        );
        assert!(reply.text.contains("叶莉") && reply.text.contains("姚明"));
    }

    fn parity_knowledge() -> MockKnowledge {
        MockKnowledge::new()
            .with_entity("姚明", &["身高", "国籍", "出生地"])
            .with_entity("叶莉", &["身高", "国籍"])
            .with_answer("姚明是谁", "姚明是中国篮球运动员。")
            .with_answer("叶莉是谁", "叶莉是中国篮球运动员。")
            .with_answer("姚明的身高是什么", "姚明的身高是 226 厘米。")
            .with_answer("姚明的国籍是什么", "姚明的国籍是中国。")
            .with_answer("叶莉的身高是什么", "叶莉的身高是 190 厘米。")
    }

    const PARITY_TRANSCRIPT: [&str; 15] = [
        "你怎么知道的？",
        "谢谢！",
        "你好",
        "姚明是谁",
        "他的身高呢",
        "你怎么知道的？",
        "我们聊了什么？",
        "还有呢",
        "详细一点",
        "姚明的体重是多少",
        "叶莉是谁",
        "叶莉的身高是多少",
        "身高呢",
        "谢谢！",
        "再见",
    ];

    fn replay_texts(mode: PromptMode) -> Vec<String> {
        let knowledge = parity_knowledge();
        let mut manager = DialogueManager::new();
        manager.set_prompt_mode(mode);
        PARITY_TRANSCRIPT
            .iter()
            .map(|input| {
                manager
                    .plan_pure_dialogue(input, &knowledge)
                    .or_else(|| manager.plan_reply(input, &knowledge))
                    .unwrap_or_else(|| panic!("输入「{}」应被接管", input))
                    .text
            })
            .collect()
    }

    #[test]
    fn test_prompt_specs_reproduce_replies_byte_for_byte() {
        let knowledge = parity_knowledge();
        let mut manager = DialogueManager::new();
        manager.set_prompt_mode(PromptMode::Parity);
        for input in PARITY_TRANSCRIPT {
            manager
                .plan_pure_dialogue(input, &knowledge)
                .or_else(|| manager.plan_reply(input, &knowledge))
                .unwrap_or_else(|| panic!("输入「{}」应被接管", input));
        }
        let stats = manager.prompt_stats();
        assert!(stats.checked >= 10, "覆盖不足：{}", stats.render_text());
        assert_eq!(
            stats.parity_mismatches, 0,
            "模板与组合生成器文本存在差异：{:#?}",
            stats.mismatch_examples
        );
        assert_eq!(stats.ungrounded, 0, "缺槽位：{:?}", stats.missing_slots);
        assert_eq!(
            stats.composed, 3,
            "recap/continue/elaborate 应保持组合生成器"
        );
    }

    #[test]
    fn test_prompt_on_mode_is_byte_identical_on_seed_transcript() {
        let off = replay_texts(PromptMode::Off);
        let on = replay_texts(PromptMode::On);
        let parity = replay_texts(PromptMode::Parity);
        assert_eq!(off, on, "On 模式应逐字等价于 Off 模式渲染");
        assert_eq!(off, parity, "Parity 模式对外输出应与 Off 模式一致");
    }

    #[test]
    fn test_prompt_stats_are_observable() {
        let knowledge = parity_knowledge();
        let mut manager = DialogueManager::new();
        manager.set_prompt_mode(PromptMode::On);
        manager.plan_reply("姚明的身高是多少", &knowledge).unwrap();
        let stats = manager.prompt_stats();
        assert_eq!(stats.grounded, 1);
        assert_eq!(stats.ungrounded, 0);
        assert!((stats.grounding_rate() - 1.0).abs() < 1e-12);
        assert!(manager.prompt_report().contains("提示资产 v1"));
        manager.reset_prompt_stats();
        assert_eq!(manager.prompt_stats().attempted(), 0);
    }

    #[test]
    fn test_query_board_drives_multi_turn_edit_and_snapshot() {
        let knowledge = MockKnowledge::new()
            .with_entity("姚明", &["身高"])
            .with_value("姚明", "身高", "226")
            .with_entity("王治郅", &["身高"])
            .with_value("王治郅", "身高", "214")
            .with_entity("易建联", &["身高"])
            .with_value("易建联", "身高", "213");
        let mut manager = DialogueManager::new();
        let reply = manager
            .plan_reply("列出所有人的身高", &knowledge)
            .expect("查询板应接管集合查询");
        assert!(reply.text.contains("姚明"));
        assert!(manager.board().is_active());

        let selected = manager
            .plan_reply("第二个呢", &knowledge)
            .expect("序号选择应留在查询板内");
        assert!(selected.text.contains("当前选中"));
        assert_eq!(manager.turns().last().unwrap().plan, vec!["board:select"]);

        let snapshot = manager.snapshot();
        let mut restored = DialogueManager::new();
        restored.restore(&snapshot);
        assert!(restored.board().is_active());
        assert_eq!(
            restored.board().frame_count(),
            manager.board().frame_count()
        );
        assert_eq!(restored.turn_count(), manager.turn_count());

        let sorted = restored
            .plan_reply("按身高从低到高排", &knowledge)
            .expect("排序应留在查询板内");
        assert!(sorted.text.contains("易建联"));
        let undone = restored
            .plan_pure_dialogue("回到上一步", &knowledge)
            .expect("撤销应回到上一帧");
        assert!(undone.text.contains("已回到第"));
    }

    #[test]
    fn test_superlative_question_normalizes_real_unit_values() {
        let knowledge = MockKnowledge::new()
            .with_entity("姚明", &["身高"])
            .with_value("姚明", "身高", "226 厘米")
            .with_entity("叶莉", &["身高"])
            .with_value("叶莉", "身高", "1.90 米");
        let mut manager = DialogueManager::new();
        manager
            .plan_reply("姚明的身高是多少", &knowledge)
            .expect("先回答单个属性");
        let reply = manager
            .plan_reply("谁最高", &knowledge)
            .expect("同一量纲的带单位值应能排名");
        assert!(reply.text.contains("姚明"), "{}", reply.text);
        assert!(reply.text.contains("226 厘米"), "{}", reply.text);
        assert!(!reply.text.contains("叶莉」的「身高」最"), "{}", reply.text);
    }

    #[test]
    fn test_superlative_question_uses_previous_attribute_and_ranks() {
        let knowledge = MockKnowledge::new()
            .with_entity("姚明", &["身高"])
            .with_value("姚明", "身高", "226")
            .with_entity("王治郅", &["身高"])
            .with_value("王治郅", "身高", "214")
            .with_entity("易建联", &["身高"])
            .with_value("易建联", "身高", "213");
        let mut manager = DialogueManager::new();
        manager
            .plan_reply("姚明的身高是多少", &knowledge)
            .expect("先回答单个属性");
        let reply = manager
            .plan_reply("谁最高", &knowledge)
            .expect("上一轮属性应补全排名请求");
        assert!(reply.text.contains("姚明"), "{}", reply.text);
        assert!(reply.text.contains("最"), "{}", reply.text);
        assert_eq!(reply.plan, vec!["board:rank"]);

        let second = manager
            .plan_reply("第二高呢", &knowledge)
            .expect("排名后应能取第二名");
        assert!(second.text.contains("当前选中"), "{}", second.text);
    }

    #[test]
    fn test_trace_mode_off_is_byte_compatible_and_records_nothing() {
        let knowledge = mock();
        let mut manager = DialogueManager::new();
        manager.plan_reply("姚明的身高是多少", &knowledge).unwrap();
        assert_eq!(manager.trace_mode(), TraceMode::Off);
        assert!(manager.trace_store().is_empty());
        // 默认 Off 的省略问句仍读作身份问句（既有契约）。
        let frame = manager.analyze("那姚明呢", &knowledge);
        assert_eq!(frame.kind, MoveKind::AskIdentity);
        assert!(!frame.inherited);
    }

    #[test]
    fn test_trace_on_inherits_predicate_focus_for_ellipsis() {
        let knowledge = MockKnowledge::new()
            .with_entity("姚明", &["身高", "国籍"])
            .with_answer("姚明的身高是什么", "姚明的身高是 226 厘米。")
            .with_entity("叶莉", &["身高"])
            .with_answer("叶莉的身高是什么", "叶莉的身高是 190 厘米。");
        let mut manager = DialogueManager::new();
        manager.set_trace_mode(TraceMode::On);
        manager.plan_reply("姚明的身高是多少", &knowledge).unwrap();
        // 「叶莉呢」在属性问据下读作「叶莉的身高呢」。
        let frame = manager.analyze("叶莉呢", &knowledge);
        assert_eq!(frame.kind, MoveKind::AskAttribute, "应承继谓词焦点");
        assert_eq!(frame.entity.as_deref(), Some("叶莉"));
        assert_eq!(frame.attribute.as_deref(), Some("身高"));
        assert!(!frame.inherited, "实体显式出现");
        let reply = manager.plan_reply("叶莉呢", &knowledge).unwrap();
        assert!(reply.text.contains("190"), "{}", reply.text);
        assert_eq!(manager.trace_store().predicate_focus(), Some("身高"));
    }

    #[test]
    fn test_trace_on_pronoun_inherits_predicate_focus() {
        let knowledge = mock();
        let mut manager = DialogueManager::new();
        manager.set_trace_mode(TraceMode::On);
        manager.plan_reply("姚明的身高是多少", &knowledge).unwrap();
        let frame = manager.analyze("他呢", &knowledge);
        assert_eq!(frame.kind, MoveKind::AskAttribute);
        assert_eq!(frame.entity.as_deref(), Some("姚明"));
        assert_eq!(frame.attribute.as_deref(), Some("身高"));
        assert!(frame.inherited);
    }

    #[test]
    fn test_trace_on_correction_overrides_value_and_marks_stale() {
        let knowledge = mock();
        let mut manager = DialogueManager::new();
        manager.set_trace_mode(TraceMode::On);
        let first = manager.plan_reply("姚明的身高是多少", &knowledge).unwrap();
        assert!(first.text.contains("226"));
        let correction = manager
            .plan_reply("不对，姚明的身高是 190", &knowledge)
            .expect("改口应被拦截");
        assert!(correction.text.contains("190"), "{}", correction.text);
        assert!(correction.text.contains("作废"), "{}", correction.text);
        // 会话内覆盖值优先于图谱旧值。
        let again = manager.plan_reply("姚明的身高是多少", &knowledge).unwrap();
        assert!(again.text.contains("190"), "{}", again.text);
        assert!(!again.text.contains("226"), "{}", again.text);
        assert!(manager.trace_report().contains("更正"));
        // 更正与改口回复必须落在同一轮迹（不留空轮、不拆成两轮）。
        let correction_turn = manager
            .trace_store()
            .turns()
            .find(|turn| !turn.corrections.is_empty())
            .expect("应有更正轮迹");
        assert!(
            correction_turn
                .nodes
                .iter()
                .any(|node| node.kind == "reply"),
            "更正轮应同时含回复节点：{:?}",
            correction_turn
                .nodes
                .iter()
                .map(|n| &n.kind)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_trace_snapshot_roundtrip_preserves_focus_and_overrides() {
        let knowledge = mock();
        let mut manager = DialogueManager::new();
        manager.set_trace_mode(TraceMode::On);
        manager.plan_reply("姚明的身高是多少", &knowledge).unwrap();
        manager
            .plan_reply("不对，姚明的身高是 190", &knowledge)
            .unwrap();
        let snapshot = manager.snapshot();
        let mut restored = DialogueManager::new();
        restored.set_trace_mode(TraceMode::On);
        restored.restore(&snapshot);
        assert_eq!(restored.trace_store().predicate_focus(), Some("身高"));
        assert_eq!(
            restored.trace_store().value_override("姚明", "身高"),
            Some("190")
        );
        assert_eq!(
            restored.trace_store().turn_count(),
            manager.trace_store().turn_count()
        );
    }

    #[test]
    fn open_turn_is_recorded_and_survives_snapshot() {
        let mut manager = DialogueManager::new();
        manager.begin_turn();
        let fault = crate::turn_runtime::TurnFault::new(
            crate::turn_runtime::TurnFaultKind::Deadline,
            crate::turn_runtime::TurnPhase::Reason,
            "超出预算",
            12,
        );
        manager.record_open_turn("姚明的身高是多少", 1, &fault, Some("姚明"), Some("身高"));
        let open = manager.latest_open_turn().expect("应记录未完成轮次");
        assert_eq!(open.input, "姚明的身高是多少");
        assert!(manager.summary().contains("未完成轮次"));

        let snapshot = manager.snapshot();
        let mut restored = DialogueManager::new();
        restored.restore(&snapshot);
        assert_eq!(
            restored.latest_open_turn().map(|open| open.fault),
            Some(crate::turn_runtime::TurnFaultKind::Deadline)
        );

        let taken = restored.take_open_turn().expect("应可取走");
        assert_eq!(taken.attempts, 1);
        assert!(restored.latest_open_turn().is_none());
    }

    #[test]
    fn open_turns_expire_after_ttl_ticks() {
        let mut manager = DialogueManager::new();
        manager.begin_turn();
        let fault = crate::turn_runtime::TurnFault::new(
            crate::turn_runtime::TurnFaultKind::Store,
            crate::turn_runtime::TurnPhase::Recall,
            "",
            1,
        );
        manager.record_open_turn("姚明是谁", 1, &fault, None, None);
        for _ in 0..crate::turn_runtime::OPEN_TURN_TTL_TURNS {
            manager.begin_turn();
        }
        assert!(manager.latest_open_turn().is_none());
    }

    #[test]
    fn frame_program_survives_snapshot_and_inherits_entity() {
        let source = r#"
enabled = true

[[frames]]
id = "person.attribute"
patterns = ["{entity}的{attribute}是什么", "{attribute}呢"]
realize = { answer = "{entity}的{attribute}是{values}。" }

[[frames.slots]]
name = "entity"
sort = "entity"
required = true
inherit = true
candidates = "entities"

[[frames.slots]]
name = "attribute"
sort = "attribute"
required = true
inherit = true
candidates = "attributes"

[[frames.slots]]
name = "value"
sort = "text"

[frames.goal]
kind = "attribute"
entity = "entity"
attribute = "attribute"
output = "value"
"#;
        let config: crate::binding_frames::DialogueFramesConfig =
            toml::from_str(source).expect("帧配置应可解析");
        let program = Arc::new(FrameProgram::compile(&config).expect("帧配置应可编译"));
        let knowledge = MockKnowledge::new()
            .with_entity("姚明", &["身高", "体重"])
            .with_value("姚明", "身高", "226")
            .with_value("姚明", "体重", "140");
        let mut manager = DialogueManager::new();
        manager.set_frame_program(Some(program.clone()));
        let reply = manager
            .plan_reply("姚明的身高是什么", &knowledge)
            .expect("完整绑定应直接回答");
        assert!(reply.text.contains("226"), "{}", reply.text);

        let snapshot = manager.snapshot();
        assert!(snapshot.frames.is_some());
        let mut restored = DialogueManager::new();
        restored.restore(&snapshot);
        restored.set_frame_program(Some(program));
        let reply = restored
            .plan_reply("体重呢", &knowledge)
            .expect("恢复后应继承实体");
        assert!(reply.text.contains("140"), "{}", reply.text);
        assert!(restored.frame_report().contains("回答 2 次"));
        assert!(restored.summary().contains("绑定帧记忆"));
    }

    #[test]
    fn identity_scopes_stay_in_the_dialogue_layer() {
        let knowledge = mock();
        let mut manager = DialogueManager::new();

        let system = manager
            .plan_pure_dialogue("你是谁？", &knowledge)
            .expect("系统身份应被接管");
        assert!(system.text.contains("KnowProlog"), "{}", system.text);
        assert_eq!(
            manager.last_identity_card().map(|card| card.scope),
            Some(crate::identity_dialogue::IdentityScope::System)
        );
        assert_eq!(manager.focus_entity(), None, "第二人称不得进入实体焦点");

        let unknown = manager
            .plan_pure_dialogue("我是谁", &knowledge)
            .expect("用户身份应被接管");
        assert!(unknown.text.contains("还没有"), "{}", unknown.text);
        assert_eq!(
            manager.last_identity_card().map(|card| card.status),
            Some(crate::identity_dialogue::IdentityStatus::Unknown)
        );

        let intro = manager
            .plan_pure_dialogue("我叫张三", &knowledge)
            .expect("显式自述应被记住");
        assert!(intro.text.contains("张三"), "{}", intro.text);
        assert_eq!(manager.identity_ledger().current(), Some("张三"));
        assert_eq!(manager.focus_entity(), None, "用户自述不得进入实体焦点");

        let recall = manager
            .plan_pure_dialogue("你记得我叫什么吗", &knowledge)
            .expect("用户身份应可回忆");
        assert!(recall.text.contains("张三"), "{}", recall.text);

        let corrected = manager
            .plan_pure_dialogue("不对，我叫李四", &knowledge)
            .expect("更正应被接受");
        assert!(corrected.text.contains("李四"), "{}", corrected.text);
        assert!(!corrected.text.contains("张三"), "旧称呼应立即失效");
        assert_eq!(
            manager.last_identity_card().map(|card| card.status),
            Some(crate::identity_dialogue::IdentityStatus::Corrected)
        );

        let snapshot = manager.snapshot();
        let mut restored = DialogueManager::new();
        restored.restore(&snapshot);
        let recalled = restored
            .plan_pure_dialogue("我是谁", &knowledge)
            .expect("恢复后应保留用户身份");
        assert!(recalled.text.contains("李四"), "{}", recalled.text);

        let mut fresh = DialogueManager::new();
        let fresh_reply = fresh
            .plan_pure_dialogue("我是谁", &knowledge)
            .expect("新会话也应回答");
        assert!(fresh_reply.text.contains("还没有"), "{}", fresh_reply.text);
        let fresh_forget = fresh
            .plan_pure_dialogue("忘记我的名字", &knowledge)
            .expect("空台账的遗忘也应被确认");
        assert!(
            fresh_forget.text.contains("还没有"),
            "{}",
            fresh_forget.text
        );
        assert!(
            !fresh_forget.text.contains("已经忘记"),
            "{}",
            fresh_forget.text
        );

        let forgotten = manager
            .plan_pure_dialogue("忘记我的名字", &knowledge)
            .expect("遗忘请求应被确认");
        assert!(forgotten.text.contains("忘记"), "{}", forgotten.text);
        assert!(!forgotten.text.contains("李四"), "遗忘后不得再出现旧名字");
        let after = manager
            .plan_pure_dialogue("我是谁", &knowledge)
            .expect("遗忘后仍应回答");
        assert!(!after.text.contains("李四"), "{}", after.text);
        assert_eq!(
            manager.last_identity_card().map(|card| card.status),
            Some(crate::identity_dialogue::IdentityStatus::Forgotten)
        );

        let reintro = manager
            .plan_pure_dialogue("我叫王五", &knowledge)
            .expect("遗忘后可以重新自述");
        assert!(reintro.text.contains("王五"), "{}", reintro.text);
        assert_eq!(manager.identity_ledger().current(), Some("王五"));
    }

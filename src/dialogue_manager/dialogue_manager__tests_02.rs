    #[test]
    fn entity_identity_appends_an_auditable_card_without_changing_the_answer_head() {
        let knowledge = mock()
            .with_entity("姚明", &["身高"])
            .with_value("姚明", "身高", "226")
            .with_profile("姚明", "姚明是中国篮球运动员。");
        let mut manager = DialogueManager::new();
        assert_eq!(
            manager.analyze("你是谁", &knowledge).question.as_deref(),
            Some("identity:system")
        );
        assert_eq!(
            manager.analyze("我是谁", &knowledge).question.as_deref(),
            Some("identity:user")
        );

        let reply = manager
            .plan_reply("姚明是谁", &knowledge)
            .expect("已知实体应回落到画像");
        assert!(
            reply.text.starts_with("姚明是中国篮球运动员。"),
            "旧答案首行必须保留：{}",
            reply.text
        );
        assert!(reply.text.contains("【身份卡·实体】"), "{}", reply.text);
        assert!(reply.text.contains("依据：图谱属性 身高"), "{}", reply.text);
        let card = manager.last_identity_card().expect("应留下结构化身份卡");
        assert_eq!(card.scope, crate::identity_dialogue::IdentityScope::Entity);
        assert_eq!(card.status, crate::identity_dialogue::IdentityStatus::Known);
        assert_eq!(card.scope_leaks(), 0);
        assert_eq!(card.missing_evidence(), 0);
        assert_eq!(manager.focus_entity(), Some("姚明"));
    }

    struct TentativeKnowledge {
        base: MockKnowledge,
    }

    impl DialogueKnowledge for TentativeKnowledge {
        fn entities_in(&self, text: &str) -> Vec<String> {
            self.base.entities_in(text)
        }

        fn known_entity(&self, entity: &str) -> bool {
            self.base.known_entity(entity)
        }

        fn answer(&self, _question: &str) -> Option<String> {
            Some("根据知识图谱的链接预测（符号推理）：\n  • 体重(姚明, 140公斤)".to_string())
        }

        fn answer_certain(&self, _question: &str) -> Option<String> {
            None
        }

        fn known_attributes(&self, entity: &str) -> Vec<String> {
            self.base.known_attributes(entity)
        }

        fn attribute_values(&self, entity: &str, attribute: &str) -> Vec<String> {
            self.base.attribute_values(entity, attribute)
        }

        fn recovery_nominations(&self, entity: &str, attribute: &str) -> Vec<RecoveryNomination> {
            self.base.recovery_nominations(entity, attribute)
        }
    }

    #[test]
    fn prediction_is_not_treated_as_certain_in_recovery_mode() {
        let knowledge = TentativeKnowledge {
            base: mock().with_entity("姚明", &["体重"]).with_nomination(
                "姚明",
                "体重",
                "140公斤",
                0.72,
                "链接预测：分布证据",
            ),
        };

        let mut off = DialogueManager::new();
        let direct = off
            .plan_reply("姚明的体重是多少", &knowledge)
            .expect("Off 模式保持既有输出");
        assert!(direct.text.contains("链接预测"), "{}", direct.text);
        assert!(off.last_recovery().is_none(), "Off 模式不应产生恢复视图");

        let mut on = DialogueManager::new();
        on.set_recovery_mode(RecoveryMode::On);
        let reply = on
            .plan_reply("姚明的体重是多少", &knowledge)
            .expect("推测轮应被接管");
        assert!(
            reply.text.contains("推测") && reply.text.contains("140公斤"),
            "预测必须以推测 + 候选呈现：{}",
            reply.text
        );
        assert!(
            reply.text.contains("依据") && reply.text.contains("链接预测"),
            "推测必须带证据：{}",
            reply.text
        );
        assert!(
            !reply.text.starts_with("根据知识图谱的链接预测"),
            "候选列表不得原样冒充确定答案：{}",
            reply.text
        );
        let notice = on.last_recovery().expect("应产生恢复视图");
        assert_eq!(notice.status, EpistemicStatus::Provisional);
        assert_eq!(notice.level, "L2");
        assert_eq!(on.recovery_state().pending.len(), 1);
    }

    #[test]
    fn denying_the_same_candidate_is_refutation_not_correction() {
        let knowledge = mock().with_entity("姚明", &["体重"]).with_nomination(
            "姚明",
            "体重",
            "140公斤",
            0.72,
            "链接预测：分布证据",
        );
        let mut manager = DialogueManager::new();
        manager.set_recovery_mode(RecoveryMode::On);
        manager
            .plan_reply("姚明的体重是多少", &knowledge)
            .expect("推测轮应被接管");
        let reply = manager
            .plan_pure_dialogue("不是140公斤", &knowledge)
            .expect("否认应结算挂账");
        assert!(reply.text.contains("作废"), "{}", reply.text);
        assert!(
            manager
                .recovery_state()
                .confirmed_value("姚明", "体重")
                .is_none(),
            "单纯否认不得把被否认的候选写进会话事实"
        );
        assert!(manager
            .recovery_state()
            .is_refuted("姚明", "体重", "140公斤"));
    }

    #[test]
    fn session_confirmed_facts_survive_snapshot_roundtrip() {
        let knowledge = mock().with_entity("姚明", &["体重"]).with_nomination(
            "姚明",
            "体重",
            "140公斤",
            0.72,
            "链接预测：分布证据",
        );
        let mut manager = DialogueManager::new();
        manager.set_recovery_mode(RecoveryMode::On);
        manager
            .plan_reply("姚明的体重是多少", &knowledge)
            .expect("推测轮应被接管");
        manager
            .plan_pure_dialogue("对", &knowledge)
            .expect("确认应结算挂账");

        let snapshot = manager.snapshot();
        let mut restored = DialogueManager::new();
        restored.set_recovery_mode(RecoveryMode::On);
        restored.restore(&snapshot);
        assert_eq!(
            restored
                .recovery_state()
                .confirmed_value("姚明", "体重")
                .map(|fact| fact.value.as_str()),
            Some("140公斤")
        );
        let reply = restored
            .plan_reply("姚明的体重是多少", &knowledge)
            .expect("恢复后应直接使用会话事实");
        assert!(reply.text.contains("140公斤"), "{}", reply.text);
        assert!(reply.text.contains("本会话确认"), "{}", reply.text);
    }

    #[test]
    fn confirmed_provisional_becomes_next_turn_premise_and_correction_revises() {
        let knowledge = mock().with_entity("姚明", &["体重"]).with_nomination(
            "姚明",
            "体重",
            "140公斤",
            0.72,
            "链接预测：分布证据",
        );
        let mut manager = DialogueManager::new();
        manager.set_recovery_mode(RecoveryMode::On);

        let first = manager
            .plan_reply("姚明的体重是多少", &knowledge)
            .expect("推测轮应被接管");
        assert!(first.text.contains("140公斤"), "{}", first.text);
        assert!(first.text.contains("推测"), "{}", first.text);

        let correction = manager
            .plan_pure_dialogue("不对，姚明的体重是 135 公斤", &knowledge)
            .expect("更正应结算推测挂账");
        assert!(correction.text.contains("作废"), "{}", correction.text);
        assert!(correction.text.contains("135 公斤"), "{}", correction.text);
        let fact = manager
            .recovery_state()
            .confirmed_value("姚明", "体重")
            .expect("更正必须进入会话事实层");
        assert_eq!(fact.value, "135 公斤");
        assert!(
            manager.recovery_report().contains("会话确认 1 条"),
            "{}",
            manager.recovery_report()
        );

        let revised = manager
            .plan_reply("姚明的体重是多少", &knowledge)
            .expect("确认后的下一轮应直接作答");
        assert!(revised.text.contains("135 公斤"), "{}", revised.text);
        assert!(
            !revised.text.contains("140公斤"),
            "被否认的推测不得再次出现：{}",
            revised.text
        );
        assert!(revised.text.contains("本会话确认"), "{}", revised.text);
        let last = manager.turns.last().expect("应记录修订轮");
        assert!(
            last.provenance.iter().any(|tag| tag == "本会话确认"),
            "{:?}",
            last.provenance
        );
    }

    #[test]
    fn restored_focus_moves_reference_prior_without_new_turn() {
        let mut manager = DialogueManager::new();
        let turns = manager.turn_count();
        assert!(manager.note_restored_focus(" 张三 "));
        assert_eq!(manager.focus_entity(), Some("张三"));
        assert_eq!(manager.turn_count(), turns, "恢复焦点不得伪造对话轮");
        assert!(!manager.note_restored_focus("   "));
        assert_eq!(manager.focus_entity(), Some("张三"));
    }

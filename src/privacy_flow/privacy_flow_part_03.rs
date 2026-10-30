#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> PrivacyPolicy {
        PrivacyPolicy {
            mode: PolicyMode::Enforce,
            k_anonymity: 1,
            ..PrivacyPolicy::default()
        }
    }

    fn ledger() -> PrivacyLedger {
        let mut ledger = PrivacyLedger::new();
        ledger.begin_turn();
        ledger
    }

    fn annotate(text: &str) -> Vec<AnnotatedWord> {
        crate::nlp::preprocess_annotated(text, &test_nlp_config())
    }

    fn test_nlp_config() -> crate::NlpConfig {
        crate::NlpConfig {
            import_keywords: vec!["录入".to_string()],
            pos_dict: Default::default(),
            name_patterns: crate::NamePatterns {
                prefixes: vec!["小".to_string()],
                suffixes: vec!["总".to_string()],
                full_names: vec!["张三".to_string()],
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
    fn strong_regexes_find_contact_credential_and_age() {
        let policy = PrivacyPolicy::default();
        let detector = SpanDetector::new(&policy);
        let text = "电话13800138000，邮箱a.b@example.com，身份证110101199001011234，今年18岁";
        let spans = detector.scan(text, &annotate(text));

        let contact = spans
            .iter()
            .find(|span| span.text == "13800138000")
            .expect("手机号");
        assert_eq!(contact.category, PrivacyCategory::Contact);
        assert_eq!(contact.strength, EvidenceStrength::Strong);
        assert!(contact.is_blocking(&policy));

        assert!(spans.iter().any(
            |span| span.category == PrivacyCategory::Contact && span.text == "a.b@example.com"
        ));
        assert!(spans
            .iter()
            .any(|span| span.category == PrivacyCategory::Credential
                && span.text == "110101199001011234"));
        assert!(spans
            .iter()
            .any(|span| span.category == PrivacyCategory::Demographic && span.text.contains("18")));
    }

    #[test]
    fn names_are_weak_evidence_and_not_blocking_by_default() {
        let policy = PrivacyPolicy::default();
        let detector = SpanDetector::new(&policy);
        let text = "张三在北京工作";
        let spans = detector.scan(text, &annotate(text));
        assert!(spans.iter().any(|span| {
            span.text == "张三"
                && span.category == PrivacyCategory::PersonalIdentity
                && span.strength == EvidenceStrength::Weak
                && !span.is_blocking(&policy)
        }));
    }

    #[test]
    fn contextual_address_is_strong() {
        let policy = PrivacyPolicy::default();
        let detector = SpanDetector::new(&policy);
        let text = "他住在北京市海淀区中关村大街1号";
        let spans = detector.scan(text, &annotate(text));
        let address = spans
            .iter()
            .find(|span| {
                span.category == PrivacyCategory::Location
                    && span.strength == EvidenceStrength::Strong
            })
            .expect("住址");
        assert!(address.text.contains("海淀区"));
    }

    #[test]
    fn redact_replaces_strong_spans_and_keeps_weak_names() {
        let policy = policy();
        let detector = SpanDetector::new(&policy);
        let text = "张三住在北京市海淀区中关村大街1号，电话13800138000。";
        let redaction = detector.redact(text, &annotate(text));

        assert!(redaction.changed());
        assert!(redaction.text.contains("张三"), "默认不对人名脱敏");
        assert!(redaction.text.contains("北京市"));
        assert!(!redaction.text.contains("海淀区"));
        assert!(!redaction.text.contains("13800138000"));
        assert!(redaction.text.contains("某联系方式"));
        assert!(redaction.text.starts_with("张三住在北京市，电话"));
        assert!(redaction.kept_weak >= 1);
        assert!(redaction
            .records
            .iter()
            .any(|record| record.category == PrivacyCategory::Contact
                && record.precision == DisclosurePrecision::TypeOnly));
    }

    #[test]
    fn redact_honors_policy_precision_and_weak_blocking() {
        let mut policy = policy();
        policy
            .category_precision
            .insert(PrivacyCategory::Location, DisclosurePrecision::Granular);
        policy.weak_evidence_blocks = true;
        let detector = SpanDetector::new(&policy);
        let text = "张三住在北京市海淀区中关村大街1号。";
        let redaction = detector.redact(text, &annotate(text));

        assert!(redaction.text.contains("北京市海淀区"));
        assert!(!redaction.text.contains("中关村"));
        assert_eq!(redaction.kept_weak, 0, "弱证据被提升为硬证据后不再原样保留");
        assert!(!redaction.text.contains("张三"));
        assert!(redaction.text.contains("某位人士"));
    }

    #[test]
    fn redact_with_no_spans_is_identity() {
        let policy = policy();
        let detector = SpanDetector::new(&policy);
        let text = "今天天气不错。";
        let redaction = detector.redact(text, &annotate(text));
        assert_eq!(redaction.text, text);
        assert!(!redaction.changed());
        assert!(redaction.records.is_empty());
    }

    #[test]
    fn audit_mode_reports_block_without_changing_answer_or_ledger() {
        let mut policy = PrivacyPolicy::default();
        policy.mode = PolicyMode::Audit;
        policy.k_anonymity = 1;
        let mut graph = PrivacyGraph::new();
        graph.add_fact(&policy, "张三", "电话", "13800138000");
        let mut ledger = ledger();
        let before = ledger.digest();
        let mut adjudicator = PrivacyAdjudicator::new(&policy, &graph, &mut ledger);
        let decision =
            adjudicator.adjudicate(&DisclosureRequest::new("张三", "电话", "13800138000"));

        assert_eq!(decision.status, DisclosureStatus::Allowed);
        assert_eq!(decision.effective_precision, DisclosurePrecision::Exact);
        assert_eq!(
            decision.replacement.as_deref(),
            Some("13800138000"),
            "影子模式绝不能改写回答"
        );
        let audit = decision.audit.expect("审计结论");
        assert_eq!(audit.would_be_status, DisclosureStatus::NeedsConsent);
        assert_eq!(audit.would_be_precision, DisclosurePrecision::TypeOnly);
        assert_eq!(ledger.digest(), before, "审计模式不写台账");
    }

    #[test]
    fn enforce_requires_consent_and_uses_typed_placeholder() {
        let policy = policy();
        let mut graph = PrivacyGraph::new();
        graph.add_fact(&policy, "张三", "电话", "13800138000");
        let mut ledger = ledger();
        let mut adjudicator = PrivacyAdjudicator::new(&policy, &graph, &mut ledger);
        let decision =
            adjudicator.adjudicate(&DisclosureRequest::new("张三", "电话", "13800138000"));

        assert_eq!(decision.status, DisclosureStatus::NeedsConsent);
        assert_eq!(decision.effective_precision, DisclosurePrecision::TypeOnly);
        assert_eq!(decision.replacement.as_deref(), Some("某联系方式"));
        assert!(decision.codes.contains(&DecisionCode::ConsentRequired));
        assert!(!decision
            .receipt
            .codes
            .iter()
            .any(|code| code == "audit_shadow"));
        assert!(!serde_json::to_string(&decision.receipt)
            .unwrap()
            .contains("13800138000"));
    }

    #[test]
    fn one_time_consent_allows_exact_once_then_falls_back() {
        let policy = policy();
        let mut graph = PrivacyGraph::new();
        graph.add_fact(&policy, "张三", "电话", "13800138000");
        let mut ledger = ledger();
        ledger.grant("张三", PrivacyCategory::Contact, "问答", ConsentScope::Once);
        let mut adjudicator = PrivacyAdjudicator::new(&policy, &graph, &mut ledger);
        let request = DisclosureRequest::new("张三", "电话", "13800138000");
        let first = adjudicator.adjudicate(&request);
        assert_eq!(first.status, DisclosureStatus::Allowed);
        assert_eq!(first.effective_precision, DisclosurePrecision::Exact);
        assert!(first.codes.contains(&DecisionCode::ConsentGranted));

        let second = adjudicator.adjudicate(&request);
        assert_eq!(second.status, DisclosureStatus::NeedsConsent);
        assert_eq!(second.effective_precision, DisclosurePrecision::TypeOnly);
    }

    #[test]
    fn session_consent_can_be_revoked() {
        let policy = policy();
        let mut graph = PrivacyGraph::new();
        graph.add_fact(&policy, "张三", "病史", "高血压");
        let mut ledger = ledger();
        ledger.grant(
            "张三",
            PrivacyCategory::Health,
            "问答",
            ConsentScope::Session,
        );
        let mut adjudicator = PrivacyAdjudicator::new(&policy, &graph, &mut ledger);
        let request = DisclosureRequest::new("张三", "病史", "高血压");
        assert_eq!(
            adjudicator.adjudicate(&request).status,
            DisclosureStatus::Allowed
        );
        ledger.revoke("张三", PrivacyCategory::Health, "问答");
        let mut adjudicator = PrivacyAdjudicator::new(&policy, &graph, &mut ledger);
        assert_eq!(
            adjudicator.adjudicate(&request).status,
            DisclosureStatus::NeedsConsent
        );
    }

    #[test]
    fn tombstone_overrides_consent() {
        let policy = policy();
        let mut graph = PrivacyGraph::new();
        graph.add_fact(&policy, "张三", "电话", "13800138000");
        let mut ledger = ledger();
        ledger.grant(
            "张三",
            PrivacyCategory::Contact,
            "问答",
            ConsentScope::Session,
        );
        ledger.forget_subject("张三", "用户要求遗忘");
        let mut adjudicator = PrivacyAdjudicator::new(&policy, &graph, &mut ledger);
        let decision =
            adjudicator.adjudicate(&DisclosureRequest::new("张三", "电话", "13800138000"));
        assert_eq!(decision.status, DisclosureStatus::Blocked);
        assert_eq!(decision.effective_precision, DisclosurePrecision::Hidden);
        assert!(decision.replacement.is_none());
        assert!(decision.codes.contains(&DecisionCode::Tombstoned));
    }

    #[test]
    fn k_anonymity_downgrades_unique_city_to_type_only() {
        let mut policy = policy();
        policy.k_anonymity = 2;
        let mut graph = PrivacyGraph::new();
        graph.add_fact(&policy, "张三", "地址", "北京市海淀区中关村大街1号");
        graph.add_fact(&policy, "李四", "地址", "上海市浦东新区世纪大道2号");
        let mut ledger = ledger();
        let mut adjudicator = PrivacyAdjudicator::new(&policy, &graph, &mut ledger);
        let decision = adjudicator.adjudicate(
            &DisclosureRequest::new("张三", "地址", "北京市海淀区中关村大街1号")
                .with_precision(DisclosurePrecision::Coarse),
        );
        assert_eq!(decision.status, DisclosureStatus::Downgraded);
        assert_eq!(decision.effective_precision, DisclosurePrecision::TypeOnly);
        assert_eq!(decision.replacement.as_deref(), Some("某地"));
        assert!(decision.codes.contains(&DecisionCode::KAnonymity));
    }

    #[test]
    fn k_anonymity_hides_unique_value_in_sparse_graph() {
        let mut policy = policy();
        policy.k_anonymity = 2;
        let mut graph = PrivacyGraph::new();
        graph.add_fact(&policy, "张三", "地址", "北京市海淀区中关村大街1号");
        let mut ledger = ledger();
        let mut adjudicator = PrivacyAdjudicator::new(&policy, &graph, &mut ledger);
        let decision = adjudicator.adjudicate(
            &DisclosureRequest::new("张三", "地址", "北京市海淀区中关村大街1号")
                .with_precision(DisclosurePrecision::Coarse),
        );
        assert_eq!(decision.status, DisclosureStatus::Blocked);
        assert_eq!(decision.effective_precision, DisclosurePrecision::Hidden);
        assert!(decision.codes.contains(&DecisionCode::KAnonymity));
    }

    #[test]
    fn k_anonymity_keeps_shared_city() {
        let mut policy = policy();
        policy.k_anonymity = 2;
        let mut graph = PrivacyGraph::new();
        graph.add_fact(&policy, "张三", "地址", "北京市海淀区中关村大街1号");
        graph.add_fact(&policy, "李四", "地址", "北京市朝阳区建国路2号");
        let mut ledger = ledger();
        let mut adjudicator = PrivacyAdjudicator::new(&policy, &graph, &mut ledger);
        let decision = adjudicator.adjudicate(&DisclosureRequest::new(
            "张三",
            "地址",
            "北京市海淀区中关村大街1号",
        ));
        assert_eq!(decision.status, DisclosureStatus::Downgraded);
        assert_eq!(decision.effective_precision, DisclosurePrecision::Coarse);
        assert_eq!(decision.replacement.as_deref(), Some("北京市"));
    }

    #[test]
    fn prior_disclosure_keeps_granularity_consistent_across_turns() {
        let policy = policy();
        let mut graph = PrivacyGraph::new();
        graph.add_fact(&policy, "张三", "姓名", "张三");
        let mut ledger = ledger();
        let mut adjudicator = PrivacyAdjudicator::new(&policy, &graph, &mut ledger);
        let first = adjudicator.adjudicate(
            &DisclosureRequest::new("张三", "姓名", "张三")
                .with_precision(DisclosurePrecision::TypeOnly),
        );
        assert_eq!(first.status, DisclosureStatus::Allowed);
        assert_eq!(first.replacement.as_deref(), Some("某位人士"));

        let second = adjudicator.adjudicate(&DisclosureRequest::new("张三", "姓名", "张三"));
        assert_eq!(second.status, DisclosureStatus::Downgraded);
        assert_eq!(second.effective_precision, DisclosurePrecision::TypeOnly);
        assert!(second.codes.contains(&DecisionCode::PriorDisclosure));
    }

    #[test]
    fn inference_guard_blocks_school_location_chain() {
        let mut policy = policy();
        policy.inference_rules.push(
            InferenceRule::new(
                "学校位置推断城区",
                "就读学校",
                "位于",
                "所在城区",
                PrivacyCategory::Location,
            )
            .with_max_precision(DisclosurePrecision::TypeOnly),
        );
        let mut graph = PrivacyGraph::new();
        graph.add_fact(&policy, "张三", "就读学校", "阳光小学");
        graph.add_fact(&policy, "阳光小学", "位于", "北京市海淀区");
        let mut ledger = ledger();
        let mut adjudicator = PrivacyAdjudicator::new(&policy, &graph, &mut ledger);
        let decision =
            adjudicator.adjudicate(&DisclosureRequest::new("张三", "就读学校", "阳光小学"));
        assert_eq!(decision.status, DisclosureStatus::Downgraded);
        assert_eq!(decision.effective_precision, DisclosurePrecision::TypeOnly);
        assert!(decision.codes.contains(&DecisionCode::InferenceLeak));
        assert!(decision
            .inference_leaks
            .iter()
            .any(|leak| leak.conclusion == "所在城区"));
    }

    #[test]
    fn truncated_closure_blocks_personal_facts_by_default() {
        let mut policy = policy();
        policy.max_closure_atoms = 2;
        policy.inference_rules.push(
            InferenceRule::new(
                "学校位置推断城区",
                "学校地址",
                "位于",
                "所在城区",
                PrivacyCategory::Location,
            )
            .with_max_precision(DisclosurePrecision::TypeOnly),
        );
        let mut graph = PrivacyGraph::new();
        graph.add_fact(&policy, "张三", "学校地址", "阳光小学");
        graph.add_fact(&policy, "阳光小学", "位于", "北京市海淀区");
        let mut ledger = ledger();
        let mut adjudicator = PrivacyAdjudicator::new(&policy, &graph, &mut ledger);
        let decision =
            adjudicator.adjudicate(&DisclosureRequest::new("张三", "学校地址", "阳光小学"));
        assert_eq!(decision.status, DisclosureStatus::Blocked);
        assert_eq!(decision.effective_precision, DisclosurePrecision::Hidden);
        assert!(decision.codes.contains(&DecisionCode::ClosureTruncated));
    }

    #[test]
    fn truncated_closure_can_be_configured_open() {
        let mut policy = policy();
        policy.max_closure_atoms = 2;
        policy.fail_closed_on_closure_truncation = false;
        policy.inference_rules.push(
            InferenceRule::new(
                "学校位置推断城区",
                "学校地址",
                "位于",
                "所在城区",
                PrivacyCategory::Location,
            )
            .with_max_precision(DisclosurePrecision::TypeOnly),
        );
        let mut graph = PrivacyGraph::new();
        graph.add_fact(&policy, "张三", "学校地址", "阳光小学");
        graph.add_fact(&policy, "阳光小学", "位于", "北京市海淀区");
        let mut ledger = ledger();
        let mut adjudicator = PrivacyAdjudicator::new(&policy, &graph, &mut ledger);
        let decision =
            adjudicator.adjudicate(&DisclosureRequest::new("张三", "学校地址", "阳光小学"));
        assert_ne!(decision.status, DisclosureStatus::Blocked);
        assert!(decision.codes.contains(&DecisionCode::ClosureTruncated));
    }

    #[test]
    fn inference_guard_can_block_when_rule_forbids_disclosure() {
        let mut policy = policy();
        policy.inference_rules.push(
            InferenceRule::new(
                "学校位置推断城区",
                "就读学校",
                "位于",
                "所在城区",
                PrivacyCategory::Location,
            )
            .with_max_precision(DisclosurePrecision::Hidden),
        );
        let mut graph = PrivacyGraph::new();
        graph.add_fact(&policy, "张三", "就读学校", "阳光小学");
        graph.add_fact(&policy, "阳光小学", "位于", "北京市海淀区");
        let mut ledger = ledger();
        let mut adjudicator = PrivacyAdjudicator::new(&policy, &graph, &mut ledger);
        let decision =
            adjudicator.adjudicate(&DisclosureRequest::new("张三", "就读学校", "阳光小学"));
        assert_eq!(decision.status, DisclosureStatus::Blocked);
        assert!(decision.codes.contains(&DecisionCode::InferenceLeak));
    }

    #[test]
    fn typed_placeholder_tracks_question_kind() {
        assert_eq!(
            typed_placeholder(PrivacyCategory::PersonalIdentity, Some(WhKind::Person)),
            "某位人士"
        );
        assert_eq!(
            typed_placeholder(PrivacyCategory::Location, Some(WhKind::Place)),
            "某地"
        );
        assert_eq!(
            typed_placeholder(PrivacyCategory::Demographic, Some(WhKind::Time)),
            "某个时间段"
        );
        assert_eq!(
            coarsen_value(
                PrivacyCategory::Demographic,
                DisclosurePrecision::Granular,
                "18岁",
                Some(WhKind::Quantity)
            )
            .as_deref(),
            Some("15-19岁")
        );
        assert_eq!(
            coarsen_value(
                PrivacyCategory::Demographic,
                DisclosurePrecision::Coarse,
                "1990-01-01",
                Some(WhKind::Time)
            )
            .as_deref(),
            Some("1990年代")
        );
    }

    #[test]
    fn question_kind_is_recorded_on_request() {
        let request =
            DisclosureRequest::new("张三", "地址", "北京市").with_question("张三在哪里？");
        assert_eq!(request.wh, Some(WhKind::Place));
        let decision_policy = PolicyMode::Audit;
        assert_eq!(decision_policy.label(), "审计（影子）");
    }

    #[test]
    fn privacy_acts_parse_grant_revoke_forget_and_policy() {
        assert_eq!(
            PrivacyAct::parse("我同意你告诉别人我的电话"),
            PrivacyAct::Grant {
                category: PrivacyCategory::Contact,
                scope: ConsentScope::Session,
            }
        );
        assert_eq!(
            PrivacyAct::parse("只此一次，可以告诉别人我的邮箱"),
            PrivacyAct::Grant {
                category: PrivacyCategory::Contact,
                scope: ConsentScope::Once,
            }
        );
        assert_eq!(
            PrivacyAct::parse("别把我的地址透露给别人"),
            PrivacyAct::Revoke {
                category: PrivacyCategory::Location,
            }
        );
        assert_eq!(
            PrivacyAct::parse("不要告诉别人我的病史"),
            PrivacyAct::Revoke {
                category: PrivacyCategory::Health,
            }
        );
        assert_eq!(
            PrivacyAct::parse("忘掉关于我的所有信息"),
            PrivacyAct::Forget { category: None }
        );
        assert_eq!(
            PrivacyAct::parse("忘记我的家庭住址"),
            PrivacyAct::Forget {
                category: Some(PrivacyCategory::Location),
            }
        );
        assert_eq!(
            PrivacyAct::parse("你的隐私政策是什么？"),
            PrivacyAct::AskPolicy
        );
        assert_eq!(
            PrivacyAct::parse("我可以告诉别人我的电话吗"),
            PrivacyAct::AskPolicy
        );
        assert_eq!(PrivacyAct::parse("今天天气不错"), PrivacyAct::Other);
    }

    #[test]
    fn apply_act_updates_consent_and_tombstones() {
        let mut ledger = ledger();
        ledger.apply_act(
            "张三",
            "问答",
            &PrivacyAct::Grant {
                category: PrivacyCategory::Contact,
                scope: ConsentScope::Session,
            },
        );
        assert!(ledger.consent_permits("张三", PrivacyCategory::Contact, "问答"));
        ledger.apply_act(
            "张三",
            "问答",
            &PrivacyAct::Revoke {
                category: PrivacyCategory::Contact,
            },
        );
        assert!(!ledger.consent_permits("张三", PrivacyCategory::Contact, "问答"));
        ledger.apply_act(
            "张三",
            "问答",
            &PrivacyAct::Forget {
                category: Some(PrivacyCategory::Location),
            },
        );
        assert!(ledger.is_tombstoned("张三", "地址", "北京市", PrivacyCategory::Location));
        assert!(!ledger.is_tombstoned("张三", "电话", "13800138000", PrivacyCategory::Contact));
    }

    #[test]
    fn policy_summary_is_deterministic_and_value_free() {
        let policy = PrivacyPolicy::default();
        let summary = policy.render_summary();
        assert!(summary.contains("审计（影子）"));
        assert!(summary.contains("k=2"));
        assert!(summary.contains("contact=type"));
        assert!(!summary.contains("13800138000"));
    }

    #[test]
    fn ledger_roundtrip_is_canonical_and_versioned() {
        let mut ledger = ledger();
        ledger.grant(
            "张三",
            PrivacyCategory::Location,
            "*",
            ConsentScope::Session,
        );
        ledger.forget_category("李四", PrivacyCategory::Health, "用户撤回");
        let text = ledger.canonical_json().expect("规范 JSON");
        let restored = PrivacyLedger::from_json(&text).expect("恢复");
        assert_eq!(restored, ledger);
        assert_eq!(restored.digest(), ledger.digest());

        let mut value: serde_json::Value = serde_json::from_str(&text).unwrap();
        value["version"] = serde_json::json!(PRIVACY_LEDGER_VERSION + 1);
        let error = PrivacyLedger::from_json(&value.to_string()).unwrap_err();
        assert!(error.contains("高于支持版本"));
    }

    #[test]
    fn overlay_replaces_or_hides_according_to_decision() {
        use crate::symbolic_generator::GenerationOverlay;
        let mut policy = policy();
        let mut graph = PrivacyGraph::new();
        graph.add_fact(&policy, "张三", "地址", "北京市海淀区中关村大街1号");
        let mut ledger = ledger();
        let mut adjudicator = PrivacyAdjudicator::new(&policy, &graph, &mut ledger);
        let decision = adjudicator.adjudicate(
            &DisclosureRequest::new("张三", "地址", "北京市海淀区中关村大街1号")
                .with_precision(DisclosurePrecision::Coarse)
                .with_k(1),
        );
        let mut overlay = GenerationOverlay::default();
        decision.apply_to_overlay(&mut overlay, "张三", "地址", "北京市海淀区中关村大街1号");
        assert_eq!(
            overlay
                .replace
                .get(&("张三".to_string(), "地址".to_string())),
            Some(&vec!["北京市".to_string()])
        );

        policy.mode = PolicyMode::Enforce;
        ledger.forget_subject("张三", "遗忘");
        let mut adjudicator = PrivacyAdjudicator::new(&policy, &graph, &mut ledger);
        let blocked = adjudicator.adjudicate(&DisclosureRequest::new(
            "张三",
            "地址",
            "北京市海淀区中关村大街1号",
        ));
        let mut overlay = GenerationOverlay::default();
        blocked.apply_to_overlay(&mut overlay, "张三", "地址", "北京市海淀区中关村大街1号");
        assert!(overlay
            .hide
            .contains(&("张三".to_string(), "地址".to_string())));
    }

    #[test]
    fn audit_summary_counts_retention_and_consent() {
        let mut policy = PrivacyPolicy::default();
        policy.mode = PolicyMode::Audit;
        policy.k_anonymity = 1;
        let mut graph = PrivacyGraph::new();
        graph.add_fact(&policy, "张三", "职业", "工程师");
        graph.add_fact(&policy, "张三", "电话", "13800138000");
        graph.add_fact(&policy, "张三", "地址", "北京市海淀区中关村大街1号");
        let mut ledger = ledger();
        let mut adjudicator = PrivacyAdjudicator::new(&policy, &graph, &mut ledger);
        let decisions = vec![
            adjudicator.adjudicate(&DisclosureRequest::new("张三", "职业", "工程师")),
            adjudicator.adjudicate(&DisclosureRequest::new("张三", "电话", "13800138000")),
            adjudicator.adjudicate(&DisclosureRequest::new(
                "张三",
                "地址",
                "北京市海淀区中关村大街1号",
            )),
        ];

        let summary = AuditSummary::from_decisions(&decisions);
        assert_eq!(summary.total, 3);
        assert_eq!(summary.allowed, 1);
        assert_eq!(summary.needs_consent, 1);
        assert_eq!(summary.downgraded, 1);
        assert_eq!(summary.blocked, 0);
        assert_eq!(summary.public_total, 1);
        assert_eq!(summary.public_retained, 1);
        assert_eq!(summary.public_changed, 0);
        assert_eq!(summary.false_block_rate(), 0.0);
        assert_eq!(summary.public_retention(), 1.0);
        assert_eq!(summary.sensitive_total, 2);
        assert_eq!(summary.sensitive_retained, 0);
        assert_eq!(summary.code_counts.get("consent_required"), Some(&1));
        assert_eq!(summary.code_counts.get("policy_ceiling"), Some(&2));
        assert!(summary.render().contains("公开事实保留 1/1"));
    }

    #[test]
    fn audit_summary_flags_public_fact_false_block() {
        let mut policy = PrivacyPolicy::default();
        policy.mode = PolicyMode::Audit;
        policy.k_anonymity = 1;
        let mut graph = PrivacyGraph::new();
        graph.add_fact(&policy, "张三", "职业", "工程师");
        let mut ledger = ledger();
        ledger.forget_subject("张三", "用户要求遗忘");
        let mut adjudicator = PrivacyAdjudicator::new(&policy, &graph, &mut ledger);
        let decision = adjudicator.adjudicate(&DisclosureRequest::new("张三", "职业", "工程师"));

        let summary = AuditSummary::from_decisions([&decision]);
        assert_eq!(summary.public_total, 1);
        assert_eq!(summary.public_changed, 1);
        assert_eq!(summary.false_block_rate(), 1.0);
        assert_eq!(summary.blocked, 1);
        assert_eq!(summary.code_counts.get("tombstoned"), Some(&1));
    }

    #[test]
    fn zero_k_disables_anonymity_check() {
        let mut policy = policy();
        policy.k_anonymity = 0;
        let mut graph = PrivacyGraph::new();
        graph.add_fact(&policy, "张三", "地址", "北京市海淀区中关村大街1号");
        let mut ledger = ledger();
        let mut adjudicator = PrivacyAdjudicator::new(&policy, &graph, &mut ledger);
        let decision = adjudicator.adjudicate(&DisclosureRequest::new(
            "张三",
            "地址",
            "北京市海淀区中关村大街1号",
        ));
        assert_eq!(decision.effective_precision, DisclosurePrecision::Coarse);
        assert!(!decision.codes.contains(&DecisionCode::KAnonymity));
    }

    #[test]
    fn custom_policy_keeps_default_rules_and_predicate_override() {
        let mut policy = PrivacyPolicy::default();
        policy.predicate_rules = vec![PredicateRule {
            contains: "就诊医院".to_string(),
            category: PrivacyCategory::Health,
            max_precision: Some(DisclosurePrecision::TypeOnly),
        }];
        policy.merge_default_rules();
        assert_eq!(
            policy.classify_predicate("联系电话"),
            PrivacyCategory::Contact
        );
        assert_eq!(
            policy.max_precision_for_predicate("就诊医院"),
            Some(DisclosurePrecision::TypeOnly)
        );
        assert_eq!(
            policy.max_precision_for_predicate("电话"),
            None,
            "未显式覆盖时沿用类别上限"
        );
    }

    #[test]
    fn inference_downgrade_uses_question_compatible_category() {
        let mut policy = policy();
        policy.inference_rules.push(
            InferenceRule::new(
                "学校位置推断城区",
                "就读学校",
                "位于",
                "所在城区",
                PrivacyCategory::Location,
            )
            .with_max_precision(DisclosurePrecision::TypeOnly),
        );
        let mut graph = PrivacyGraph::new();
        graph.add_fact(&policy, "张三", "就读学校", "阳光小学");
        graph.add_fact(&policy, "阳光小学", "位于", "北京市海淀区");
        let mut ledger = ledger();
        let mut adjudicator = PrivacyAdjudicator::new(&policy, &graph, &mut ledger);
        let decision = adjudicator.adjudicate(
            &DisclosureRequest::new("张三", "就读学校", "阳光小学")
                .with_question("张三在哪里上学？"),
        );
        assert_eq!(decision.replacement.as_deref(), Some("某地"));
    }

    #[test]
    fn audit_mode_does_not_consume_one_time_consent() {
        let mut policy = PrivacyPolicy::default();
        policy.mode = PolicyMode::Audit;
        policy.k_anonymity = 1;
        let mut graph = PrivacyGraph::new();
        graph.add_fact(&policy, "张三", "电话", "13800138000");
        let mut ledger = ledger();
        ledger.grant("张三", PrivacyCategory::Contact, "问答", ConsentScope::Once);
        let mut adjudicator = PrivacyAdjudicator::new(&policy, &graph, &mut ledger);
        let request = DisclosureRequest::new("张三", "电话", "13800138000");
        let _ = adjudicator.adjudicate(&request);
        assert!(!ledger.consents[0].consumed, "影子模式不消费授权");

        policy.mode = PolicyMode::Enforce;
        let mut adjudicator = PrivacyAdjudicator::new(&policy, &graph, &mut ledger);
        let decision = adjudicator.adjudicate(&request);
        assert_eq!(decision.status, DisclosureStatus::Allowed);
        assert!(ledger.consents[0].consumed);
    }

    #[test]
    fn classify_predicate_prefers_specific_rule() {
        let policy = PrivacyPolicy::default();
        assert_eq!(
            policy.classify_predicate("家庭住址"),
            PrivacyCategory::Location
        );
        assert_eq!(
            policy.classify_predicate("联系电话"),
            PrivacyCategory::Contact
        );
        assert_eq!(policy.classify_predicate("职业"), PrivacyCategory::Other);
    }
}

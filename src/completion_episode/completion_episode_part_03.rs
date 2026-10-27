fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}
/// 稳定短 id：FNV-1a + 输入长度，避免同前缀截断歧义。
fn stable_id(prefix: &str, canonical: &str) -> String {
    format!(
        "{}{:016x}{:x}",
        prefix,
        fnv1a64(canonical.as_bytes()),
        canonical.len()
    )
}
fn short_id(id: &str) -> &str {
    id.get(..id.len().min(21)).unwrap_or(id)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::link_prediction::LinkPredictor;

    fn chain_predictor() -> LinkPredictor {
        let mut predictor = LinkPredictor::new();
        predictor.add_triple("张三", "师傅", "李四");
        predictor.add_triple("李四", "师傅", "王五");
        predictor.add_triple("王五", "师傅", "赵六");
        predictor.add_triple("张三", "师祖", "王五");
        predictor.add_triple("孙八", "师傅", "周九");
        predictor.add_triple("周九", "师傅", "吴十");
        predictor.add_triple("郑十一", "师傅", "钱十二");
        predictor.add_triple("钱十二", "师傅", "孙十三");
        predictor.add_triple("甲", "师傅", "乙");
        predictor.add_triple("乙", "师傅", "丙");
        predictor.add_triple("丁", "师傅", "戊");
        predictor.add_triple("戊", "师傅", "己");
        predictor.mine_rules(1, 0.1);
        predictor
    }

    fn open_with_predictions(
        ledger: &mut CompletionLedger,
        predictor: &LinkPredictor,
        entity: &str,
    ) {
        ledger.open_episode(entity, None);
        let predictions = predictor.predict_for_entity(entity, None, 8);
        for prediction in &predictions {
            let rule = predictor.rules().get(prediction.rule_index);
            let explanation = predictor.explain(prediction);
            ledger.add_prediction(prediction, rule, explanation);
        }
    }

    #[test]
    fn confirm_learns_batch_rule_and_closes_more_entities() {
        let predictor = chain_predictor();
        let facts = predictor.triples().to_vec();
        let mut ledger = CompletionLedger::new();
        open_with_predictions(&mut ledger, &predictor, "孙八");
        ledger.set_batch_scope(
            ["孙八", "周九", "吴十", "郑十一", "钱十二", "孙十三", "甲", "乙", "丙"]
                .into_iter()
                .map(str::to_string),
        );
        let episode = ledger.current_episode().expect("回合应打开");
        assert!(
            episode
                .candidates
                .iter()
                .any(|candidate| candidate.triple == Triple::new("孙八", "师祖", "吴十")),
            "孙八的师祖候选应存在：{:?}",
            episode
                .candidates
                .iter()
                .map(|c| c.display_triple())
                .collect::<Vec<_>>()
        );
        let outcome = ledger
            .confirm("孙八|师祖|吴十", &facts)
            .expect("确认应成功");
        assert!(outcome.learned);
        assert_eq!(outcome.rule_scope, Some(RuleScope::Batch));
        assert_eq!(ledger.rules().len(), 1);
        let rule = &ledger.rules()[0];
        assert_eq!(rule.scope, RuleScope::Batch);
        assert_eq!(rule.confirmations, 1);
        assert!(outcome.new_candidates >= 1, "批次闭包应产生新候选");
        let episode = ledger.current_episode().unwrap();
        let learned = episode
            .candidates
            .iter()
            .find(|candidate| candidate.triple == Triple::new("郑十一", "师祖", "孙十三"))
            .expect("批次内其他实体应被补全");
        assert_eq!(learned.origin, CandidateOrigin::BatchClosure);
        assert_eq!(learned.warrant, Warrant::Induced);
        assert!(learned.explanation.contains("本批"));
        assert!(ledger.reasoning_clauses().is_empty(), "本批规则不应进入全局归结");
    }

    #[test]
    fn repeated_refutation_quarantines_rule_and_removes_pending() {
        let predictor = chain_predictor();
        let facts = predictor.triples().to_vec();
        let mut ledger = CompletionLedger::new();
        open_with_predictions(&mut ledger, &predictor, "孙八");
        ledger.set_batch_scope(
            [
                "孙八", "郑十一", "钱十二", "孙十三", "甲", "乙", "丙", "丁", "戊", "己",
            ]
            .into_iter()
            .map(str::to_string),
        );
        ledger
            .confirm("孙八|师祖|吴十", &facts)
            .expect("确认应成功");
        let first_closure = ledger
            .current_episode()
            .unwrap()
            .candidates
            .iter()
            .find(|candidate| {
                candidate.origin == CandidateOrigin::BatchClosure
                    && candidate.triple == Triple::new("郑十一", "师祖", "孙十三")
            })
            .expect("应有闭包候选")
            .id
            .clone();
        let second_closure = ledger
            .current_episode()
            .unwrap()
            .candidates
            .iter()
            .find(|candidate| {
                candidate.origin == CandidateOrigin::BatchClosure
                    && candidate.triple == Triple::new("甲", "师祖", "丙")
            })
            .expect("应有第二个闭包候选")
            .id
            .clone();
        let first = ledger.refute(&first_closure).expect("第一次驳回应成功");
        assert!(!first.quarantined);
        let second = ledger.refute(&second_closure).expect("第二次驳回应成功");
        assert!(second.quarantined);
        assert_eq!(second.rule_status, Some(RuleStatus::Quarantined));
        assert!(second.removed_pending > 0, "隔离后应清理同一规则的待定候选");
        assert!(ledger.rules()[0]
            .negative_examples
            .contains(&Triple::new("郑十一", "师祖", "孙十三")));
    }

    #[test]
    fn refuted_triple_is_not_reproposed_in_next_episode() {
        let predictor = chain_predictor();
        let mut ledger = CompletionLedger::new();
        open_with_predictions(&mut ledger, &predictor, "孙八");
        ledger
            .refute("孙八|师祖|吴十")
            .expect("驳回应成功");
        open_with_predictions(&mut ledger, &predictor, "孙八");
        assert!(
            !ledger
                .current_episode()
                .unwrap()
                .candidates
                .iter()
                .any(|candidate| candidate.triple == Triple::new("孙八", "师祖", "吴十")),
            "已驳回的候选不应在新回合重新出现"
        );
    }

    #[test]
    fn promotion_separates_namespace_from_global_reasoning() {
        let predictor = chain_predictor();
        let facts = predictor.triples().to_vec();
        let mut ledger = CompletionLedger::new();
        open_with_predictions(&mut ledger, &predictor, "孙八");
        ledger.set_batch_scope(["孙八", "周九", "吴十"].into_iter().map(str::to_string));
        ledger
            .confirm("孙八|师祖|吴十", &facts)
            .expect("第一次确认应成功");
        let rule_id = ledger.rules()[0].id.clone();
        open_with_predictions(&mut ledger, &predictor, "郑十一");
        ledger.set_batch_scope(
            ["郑十一", "钱十二", "孙十三"]
                .into_iter()
                .map(str::to_string),
        );
        ledger
            .confirm("郑十一|师祖|孙十三", &facts)
            .expect("第二次确认应成功");
        let rule = ledger.rule(&rule_id).unwrap();
        assert_eq!(rule.confirmations, 2);
        assert_eq!(rule.status, RuleStatus::Active);
        let outcome = ledger
            .promote(&rule_id, RuleScope::Namespace)
            .expect("晋升应成功");
        assert_eq!(outcome.from, RuleScope::Batch);
        assert_eq!(outcome.to, RuleScope::Namespace);
        assert!(outcome.warning.is_none());
        assert_eq!(ledger.rule(&rule_id).unwrap().scope, RuleScope::Namespace);
        assert!(
            ledger.reasoning_clauses().is_empty(),
            "命名空间规则不得静默进入全局归结"
        );
        let global = ledger
            .promote(&rule_id, RuleScope::Global)
            .expect("再次晋升到全局应成功");
        assert_eq!(global.from, RuleScope::Namespace);
        assert_eq!(ledger.reasoning_clauses().len(), 1);
        assert!(ledger.promote(&rule_id, RuleScope::Batch).is_err());
    }

    #[test]
    fn retire_and_reactivate_rule_with_short_id() {
        let predictor = chain_predictor();
        let facts = predictor.triples().to_vec();
        let mut ledger = CompletionLedger::new();
        open_with_predictions(&mut ledger, &predictor, "孙八");
        ledger.set_batch_scope(["孙八", "周九", "吴十"].into_iter().map(str::to_string));
        ledger
            .confirm("孙八|师祖|吴十", &facts)
            .expect("确认应成功");
        let rule_id = ledger.rules()[0].id.clone();
        ledger
            .promote(&rule_id, RuleScope::Namespace)
            .expect("晋升应成功");
        assert!(
            ledger.reasoning_clauses().is_empty(),
            "命名空间规则只在本回合范围内使用"
        );
        ledger
            .promote(&rule_id, RuleScope::Global)
            .expect("全局晋升应成功");
        assert_eq!(ledger.reasoning_clauses().len(), 1);

        let short = &rule_id[..rule_id.len().min(21)];
        let retired = ledger.retire(short).expect("短 id 应能定位规则");
        assert_eq!(retired.rule_id, rule_id);
        assert_eq!(retired.previous, RuleStatus::Candidate);
        assert_eq!(retired.current, RuleStatus::Retired);
        assert!(ledger.reasoning_clauses().is_empty());
        assert!(matches!(
            ledger.retire(&rule_id),
            Err(EpisodeError::RuleAlreadyRetired(_))
        ));
        assert!(matches!(
            ledger.promote(&rule_id, RuleScope::Global),
            Err(EpisodeError::RuleUnavailable(_))
        ));

        let restored = ledger.reactivate(short).expect("恢复应成功");
        assert_eq!(restored.previous, RuleStatus::Retired);
        assert_eq!(restored.current, RuleStatus::Candidate);
        assert_eq!(ledger.reasoning_clauses().len(), 1);
        assert!(matches!(
            ledger.reactivate(&rule_id),
            Err(EpisodeError::RuleNotRetired(_))
        ));
        assert!(matches!(
            ledger.retire("rule-missing"),
            Err(EpisodeError::RuleNotFound(_))
        ));
        assert!(ledger.render_rules().contains("全局"));
    }

    #[test]
    fn analogical_candidates_are_hedged_and_clamped() {
        let mut ledger = CompletionLedger::new();
        ledger.open_episode("赵六", Some("赵六的国籍是什么"));
        assert!(ledger.add_analogical(
            Triple::new("赵六", "国籍", "中国"),
            0.55,
            vec![Triple::new("张三", "国籍", "中国")],
            "同类实体都有国籍".to_string(),
        ));
        assert!(!ledger.add_analogical(
            Triple::new("赵六", "国籍", "中国"),
            0.8,
            vec![Triple::new("李四", "国籍", "中国")],
            "另一个同类实体也有国籍".to_string(),
        ));
        let episode = ledger.current_episode().unwrap();
        assert_eq!(episode.candidates.len(), 1, "同一三元组不应重复入账");
        let candidate = &episode.candidates[0];
        assert_eq!(candidate.warrant, Warrant::Analogical);
        assert_eq!(candidate.warrant.hedge(), "（类比，仅提示）");
        assert!(candidate.confidence >= 0.8);
        let rendered = ledger.render_episode();
        assert!(rendered.contains("类比"));
        assert!(rendered.contains("赵六"));
    }

    #[test]
    fn closure_respects_existing_facts_and_candidate_limit() {
        let facts = vec![
            Triple::new("a", "r", "b"),
            Triple::new("b", "r", "c"),
            Triple::new("a", "s", "c"),
            Triple::new("d", "r", "e"),
            Triple::new("e", "r", "f"),
        ];
        let spec = RuleSpec {
            pattern: RulePattern::Compose,
            body: vec!["r".to_string(), "r".to_string()],
            head: "s".to_string(),
        };
        let filter = ScopeFilter::All;
        let (hits, truncated) = apply_rule(&spec, &facts, &filter, 8);
        assert!(!truncated);
        assert_eq!(
            hits.iter().map(|hit| hit.triple.clone()).collect::<Vec<_>>(),
            vec![Triple::new("d", "s", "f")],
            "已知的 a-s-c 不应重复生成"
        );
        let (limited, truncated) = apply_rule(&spec, &facts, &filter, 0);
        assert!(limited.is_empty() && !truncated);
    }

    #[test]
    fn clamp_limits_rejects_untrusted_unbounded_snapshot() {
        let mut ledger =
            CompletionLedger::with_limits(usize::MAX, usize::MAX, usize::MAX, usize::MAX);
        ledger.open_episode("甲", None);
        ledger.active = Some(999);
        ledger.clamp_limits();
        assert_eq!(ledger.max_candidates, DEFAULT_MAX_EPISODE_CANDIDATES);
        assert_eq!(ledger.max_closure, DEFAULT_MAX_CLOSURE_CANDIDATES);
        assert_eq!(ledger.max_episodes, DEFAULT_MAX_EPISODES);
        assert_eq!(ledger.max_rules, DEFAULT_MAX_RULES);
        assert_eq!(ledger.max_batch_entities, DEFAULT_MAX_BATCH_ENTITIES);
        assert_eq!(ledger.active, None, "越界活动索引应被清掉");
    }

    #[test]
    fn rule_cap_evicts_before_insert_so_returned_id_stays_live() {
        let mut ledger = CompletionLedger::with_limits(8, 8, 8, 1);
        ledger.open_episode("甲", None);
        let first = RuleSpec {
            pattern: RulePattern::Compose,
            body: vec!["r".to_string(), "r".to_string()],
            head: "s".to_string(),
        };
        let second = RuleSpec {
            pattern: RulePattern::Compose,
            body: vec!["r".to_string(), "r".to_string()],
            head: "t".to_string(),
        };
        let first_id = ledger.upsert_rule(
            first,
            &Triple::new("甲", "s", "乙"),
            "ep-1",
            0.7,
            1,
        );
        let second_id = ledger.upsert_rule(
            second,
            &Triple::new("甲", "t", "乙"),
            "ep-2",
            0.7,
            2,
        );
        assert_eq!(ledger.rules().len(), 1);
        assert_ne!(first_id, second_id);
        assert!(ledger.rule(&second_id).is_some(), "返回值必须是仍在台账中的规则");
        assert!(ledger.rule(&first_id).is_none());
    }

    #[test]
    fn ambiguous_candidate_prefix_is_rejected_instead_of_guessed() {
        let mut ledger = CompletionLedger::new();
        ledger.open_episode("甲", None);
        assert!(ledger.add_candidate(
            Triple::new("甲", "r", "乙"),
            Warrant::Induced,
            0.5,
            Vec::new(),
            "候选一".to_string(),
        ));
        assert!(ledger.add_candidate(
            Triple::new("甲", "r", "丙"),
            Warrant::Induced,
            0.5,
            Vec::new(),
            "候选二".to_string(),
        ));
        {
            let episode = ledger.current_episode_mut().expect("回合应打开");
            episode.candidates[0].id = "cand-shared-a".to_string();
            episode.candidates[1].id = "cand-shared-b".to_string();
        }
        let error = ledger
            .confirm("cand-shared", &[])
            .expect_err("歧义前缀不得静默选择第一条");
        assert!(
            matches!(
                error,
                EpisodeError::AmbiguousSelector { count: 2, .. }
            ),
            "{error:?}"
        );
        let episode = ledger.current_episode().expect("回合应打开");
        assert_eq!(
            episode.candidates[0].status,
            CandidateStatus::Pending,
            "歧义失败不得改变候选状态"
        );
        assert_eq!(episode.candidates[1].status, CandidateStatus::Pending);
    }

    #[test]
    fn refuted_closure_blocks_promotion_and_external_correction_clears_it() {
        let predictor = chain_predictor();
        let facts = predictor.triples().to_vec();
        let mut ledger = CompletionLedger::new();
        open_with_predictions(&mut ledger, &predictor, "孙八");
        ledger.set_batch_scope(
            ["孙八", "周九", "吴十", "郑十一", "钱十二", "孙十三"]
                .into_iter()
                .map(str::to_string),
        );
        ledger
            .confirm("孙八|师祖|吴十", &facts)
            .expect("确认应成功");
        let rule_id = ledger.rules()[0].id.clone();
        ledger
            .refute("郑十一|师祖|孙十三")
            .expect("驳回应成功");
        let regression = ledger.replay_rule(&rule_id).expect("规则应存在");
        assert!(!regression.passed(), "被否决候选应进入运行期测试");
        let blocked = ledger
            .promote(&rule_id, RuleScope::Namespace)
            .expect_err("有未通过测试的规则不得晋升");
        assert!(
            matches!(&blocked, EpisodeError::PromotionBlocked(reason) if reason.contains("运行期测试")),
            "{blocked:?}"
        );

        let reopened = ledger
            .record_external_verdict(
                Triple::new("郑十一", "师祖", "孙十三"),
                true,
                ledger.rule(&rule_id).map(|rule| rule.spec.clone()),
                vec![
                    Triple::new("郑十一", "师傅", "钱十二"),
                    Triple::new("钱十二", "师傅", "孙十三"),
                ],
                "恢复阶梯",
                "receipt-1",
            );
        assert!(reopened, "更正裁决应写入台账");
        assert!(
            ledger
                .replay_rule(&rule_id)
                .expect("规则应存在")
                .passed(),
            "更正后运行期测试应通过"
        );
        ledger
            .promote(&rule_id, RuleScope::Namespace)
            .expect("更正后应可晋升");
    }

    #[test]
    fn question_policy_picks_one_discriminative_candidate_and_respects_mute() {
        let predictor = chain_predictor();
        let mut ledger = CompletionLedger::new();
        open_with_predictions(&mut ledger, &predictor, "孙八");
        let first = ledger.next_question().expect("应有可追问候选");
        assert_eq!(first.triple.subject, "孙八");
        assert!(first.utility >= QUESTION_MIN_UTILITY);
        assert!(first.text.contains("孙八"));
        assert!(matches!(
            first.kind,
            QuestionKind::Confirm | QuestionKind::Choice
        ));
        let count = ledger.current_episode().unwrap().pending_count();
        let mut asked = 1usize;
        while asked < count.saturating_mul(QUESTION_MAX_REPEATS as usize) {
            if ledger.next_question().is_none() {
                break;
            }
            asked += 1;
        }
        assert!(ledger.next_question().is_none(), "重复追问应被预算截停");
        assert!(ledger
            .asked_questions
            .values()
            .all(|count| *count <= QUESTION_MAX_REPEATS));
        assert!(ledger.mute_questions());
        assert!(ledger.next_question().is_none());
    }

    #[test]
    fn question_policy_uses_choice_when_multiple_values_compete() {
        let mut ledger = CompletionLedger::new();
        ledger.open_episode("甲", None);
        for object in ["乙", "丙"] {
            assert!(ledger.add_candidate(
                Triple::new("甲", "朋友", object),
                Warrant::Induced,
                0.6,
                vec![Triple::new(object, "同类", "甲")],
                format!("候选{}", object),
            ));
        }
        let question = ledger.next_question().expect("应有选择问");
        assert_eq!(question.kind, QuestionKind::Choice);
        assert_eq!(question.alternatives.len(), 1);
        assert!(question.text.contains("更可能是"));
        assert!(!ledger.unmute_questions(), "未静音时恢复应无变化");
        assert!(ledger.mute_questions());
        assert!(ledger.unmute_questions());
    }

    #[test]
    fn external_verdict_learns_rule_and_is_idempotent() {
        let predictor = chain_predictor();
        let prediction = predictor
            .predict_for_entity("孙八", None, 8)
            .into_iter()
            .find(|prediction| prediction.triple == Triple::new("孙八", "师祖", "吴十"))
            .expect("应有一条师祖预测");
        let spec = RuleSpec::from_horn(&predictor.rules()[prediction.rule_index])
            .expect("预测规则应可归纳");
        let mut ledger = CompletionLedger::new();
        assert!(ledger.record_external_verdict(
            prediction.triple.clone(),
            true,
            Some(spec.clone()),
            prediction.evidence.clone(),
            "恢复阶梯",
            "receipt-9",
        ));
        assert_eq!(ledger.rules().len(), 1);
        assert_eq!(ledger.rules()[0].confirmations, 1);
        assert_eq!(ledger.rules()[0].scope, RuleScope::Batch);
        assert!(!ledger.record_external_verdict(
            prediction.triple.clone(),
            true,
            Some(spec),
            prediction.evidence.clone(),
            "恢复阶梯",
            "receipt-9",
        ));
        assert_eq!(ledger.rules()[0].confirmations, 1, "同一回执不得重复计数");
        assert!(ledger.is_adjudicated(&prediction.triple));
        assert!(ledger.render_regression_report().contains("确认 1"));
    }

    #[test]
    fn enforce_regression_quarantines_rules_that_contradict_oracle() {
        let predictor = chain_predictor();
        let prediction = predictor
            .predict_for_entity("孙八", None, 8)
            .into_iter()
            .find(|prediction| prediction.triple == Triple::new("孙八", "师祖", "吴十"))
            .expect("应有一条师祖预测");
        let spec = RuleSpec::from_horn(&predictor.rules()[prediction.rule_index])
            .expect("预测规则应可归纳");
        let mut ledger = CompletionLedger::new();
        assert!(ledger.record_external_verdict(
            prediction.triple.clone(),
            true,
            Some(spec.clone()),
            prediction.evidence.clone(),
            "恢复阶梯",
            "receipt-30",
        ));
        assert!(ledger.record_external_verdict(
            prediction.triple.clone(),
            false,
            Some(spec),
            prediction.evidence.clone(),
            "恢复阶梯",
            "receipt-31",
        ));
        assert_eq!(ledger.rules()[0].status, RuleStatus::Candidate);
        let report = ledger.enforce_regression();
        assert!(!report.passed());
        assert_eq!(report.failures[0].rule_id, ledger.rules()[0].id);
        assert_eq!(ledger.rules()[0].status, RuleStatus::Quarantined);
        assert!(ledger.enforce_regression().passed(), "重复重放应稳定");
    }

    #[test]
    fn correction_reinstates_quarantined_rule_and_clears_negative_example() {
        let predictor = chain_predictor();
        let prediction = predictor
            .predict_for_entity("孙八", None, 8)
            .into_iter()
            .find(|prediction| prediction.triple == Triple::new("孙八", "师祖", "吴十"))
            .expect("应有一条师祖预测");
        let spec = RuleSpec::from_horn(&predictor.rules()[prediction.rule_index])
            .expect("预测规则应可归纳");
        let mut ledger = CompletionLedger::new();
        assert!(ledger.record_external_verdict(
            prediction.triple.clone(),
            true,
            Some(spec.clone()),
            prediction.evidence.clone(),
            "恢复阶梯",
            "receipt-20",
        ));
        for receipt in ["receipt-21", "receipt-22"] {
            assert!(ledger.record_external_verdict(
                prediction.triple.clone(),
                false,
                Some(spec.clone()),
                prediction.evidence.clone(),
                "恢复阶梯",
                receipt,
            ));
        }
        assert_eq!(ledger.rules()[0].status, RuleStatus::Quarantined);
        assert!(ledger.record_external_verdict(
            prediction.triple.clone(),
            true,
            Some(spec),
            prediction.evidence.clone(),
            "恢复阶梯",
            "receipt-23",
        ));
        let rule = &ledger.rules()[0];
        assert_eq!(rule.status, RuleStatus::Active, "更正后应解除隔离并重新生效");
        assert_eq!(rule.refutations, 1);
        assert!(!rule.negative_examples.contains(&prediction.triple));
        assert!(
            ledger
                .replay_rule(&rule.id)
                .expect("规则应存在")
                .passed(),
            "更正后运行期测试应通过"
        );
    }

    #[test]
    fn external_refutation_is_never_reproposed_and_blocks_reactivation() {
        let predictor = chain_predictor();
        let prediction = predictor
            .predict_for_entity("孙八", None, 8)
            .into_iter()
            .find(|prediction| prediction.triple == Triple::new("孙八", "师祖", "吴十"))
            .expect("应有一条师祖预测");
        let spec = RuleSpec::from_horn(&predictor.rules()[prediction.rule_index])
            .expect("预测规则应可归纳");
        let mut ledger = CompletionLedger::new();
        assert!(ledger.record_external_verdict(
            prediction.triple.clone(),
            true,
            Some(spec.clone()),
            prediction.evidence.clone(),
            "恢复阶梯",
            "receipt-10",
        ));
        assert!(ledger.record_external_verdict(
            prediction.triple.clone(),
            false,
            Some(spec),
            prediction.evidence.clone(),
            "恢复阶梯",
            "receipt-11",
        ));
        assert!(ledger.is_refuted(&prediction.triple));
        ledger.open_episode("孙八", None);
        assert!(
            !ledger.add_prediction(&prediction, None, String::new()),
            "外部否决过的三元组不得重新成为候选"
        );
        let rule_id = ledger.rules()[0].id.clone();
        ledger.retire(&rule_id).expect("退役应成功");
        let blocked = ledger
            .reactivate(&rule_id)
            .expect_err("重放失败的规则不得恢复");
        assert!(
            matches!(blocked, EpisodeError::RegressionBlocked(_)),
            "{blocked:?}"
        );
    }
}

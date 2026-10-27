fn expectation_for(
    canonical: &str,
    inverted: bool,
    profile: Option<&PredicateProfile>,
    shapes: &ShapeSet,
    types: &TypeSystem,
) -> Expectation {
    let relation_shape = shapes
        .relations
        .iter()
        .find(|shape| shape.path == canonical);
    let property_shape = shapes
        .nodes
        .iter()
        .flat_map(|node| node.properties.iter())
        .find(|shape| shape.path == canonical);

    let mut raw_types = BTreeSet::new();
    if let Some(profile) = profile {
        if inverted {
            raw_types.extend(profile.domain.iter().cloned());
        } else {
            raw_types.extend(profile.range.iter().cloned());
        }
    }
    if let Some(shape) = relation_shape {
        if inverted {
            raw_types.extend(shape.subject_class.iter().cloned());
        } else {
            raw_types.extend(shape.class.iter().cloned());
        }
    }
    if let Some(shape) = property_shape {
        raw_types.extend(shape.class.iter().cloned());
    }
    let entity_types: Vec<String> = raw_types
        .into_iter()
        .map(|name| types.canonical(&name))
        .filter(|name| !name.is_empty())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();

    let datatype = property_shape.and_then(|shape| shape.datatype.clone());
    let functional = profile.map(|profile| profile.functional).unwrap_or(false)
        || relation_shape
            .and_then(|shape| shape.max_count)
            .map(|count| count == 1)
            .unwrap_or(false);
    let required = property_shape
        .and_then(|shape| shape.min_count)
        .map(|count| count > 0)
        .unwrap_or(false);
    let value_shape = if datatype.is_some() {
        ValueShape::Literal
    } else if entity_types.is_empty() {
        ValueShape::Infer
    } else {
        ValueShape::Entity
    };
    Expectation {
        entity_types,
        datatype,
        functional,
        inverse: inverted,
        required,
        value_shape,
    }
}
fn question_text(subject: &str, surface: &str, expectation: &Expectation) -> String {
    let slot = match expectation.value_shape {
        ValueShape::Entity => {
            if expectation
                .entity_types
                .iter()
                .any(|name| name == "人物" || name == "person")
            {
                "谁"
            } else {
                "哪个实体"
            }
        }
        ValueShape::Literal => "什么值",
        ValueShape::Infer => "什么",
    };
    let hint = expectation.render_text();
    if expectation.functional {
        format!(
            "我暂时没有「{}」的「{}」记录。你能告诉我是{}吗？（{}，只接受一个取值）",
            subject, surface, slot, hint
        )
    } else {
        format!(
            "我暂时没有「{}」的「{}」记录。你能告诉我是{}吗？（期望：{}）",
            subject, surface, slot, hint
        )
    }
}
fn clean_value(raw: &str) -> String {
    let mut value = raw
        .trim()
        .trim_matches(|ch: char| {
            matches!(
                ch,
                '。' | '，' | ',' | '；' | ';' | '！' | '!' | ' ' | '“' | '”' | '"' | '\''
            )
        })
        .trim()
        .to_string();
    for prefix in [
        "应该是",
        "其实是",
        "就是",
        "她叫",
        "他叫",
        "叫作",
        "叫做",
        "叫",
        "是",
    ] {
        if let Some(rest) = value.strip_prefix(prefix) {
            value = rest.trim().to_string();
            break;
        }
    }
    value
}
fn classify_answer(input: &str) -> ClassifiedAnswer {
    let text = input.trim().trim_end_matches(['。', '！', '!', ' ']).trim();
    let lower = text.to_ascii_lowercase();
    if text.is_empty() {
        return ClassifiedAnswer::new(AnswerAct::Clarify, Vec::new());
    }
    if ClassifiedAnswer::contains_any(
        text,
        &[
            "不知道",
            "不清楚",
            "不确定",
            "不记得",
            "忘了",
            "没印象",
            "无可奉告",
        ],
    ) {
        return ClassifiedAnswer::new(AnswerAct::Unknown, Vec::new());
    }
    if ClassifiedAnswer::contains_any(
        text,
        &[
            "不想说",
            "不能说",
            "不方便说",
            "保密",
            "拒绝回答",
            "不想回答",
        ],
    ) {
        return ClassifiedAnswer::new(AnswerAct::Refuse, Vec::new());
    }
    if text.ends_with('？') || text.ends_with('?') || text.ends_with('吗') {
        return ClassifiedAnswer::new(AnswerAct::Clarify, Vec::new());
    }

    if is_denial(text) {
        if let Some(replacement) = correction_tail(text) {
            let values = extract_values(&replacement);
            if !values.is_empty() {
                return ClassifiedAnswer::new(AnswerAct::Correct, values);
            }
        }
        return ClassifiedAnswer::new(AnswerAct::Deny, Vec::new());
    }

    if matches!(
        lower.as_str(),
        "对" | "对呀"
            | "对的"
            | "是"
            | "是的"
            | "是的呀"
            | "没错"
            | "正确"
            | "确定"
            | "嗯"
            | "嗯嗯"
            | "好"
            | "可以"
            | "yes"
            | "y"
            | "ok"
            | "okay"
    ) {
        return ClassifiedAnswer::new(AnswerAct::Confirm, Vec::new());
    }
    if let Some(rest) = text
        .strip_prefix("确认")
        .or_else(|| text.strip_prefix("对，"))
        .or_else(|| text.strip_prefix("对,"))
    {
        let values = extract_values(rest);
        return ClassifiedAnswer::new(AnswerAct::Confirm, values);
    }

    let values = extract_values(text);
    if values.is_empty() {
        ClassifiedAnswer::new(AnswerAct::Clarify, Vec::new())
    } else {
        ClassifiedAnswer::new(AnswerAct::Provide, values)
    }
}
fn is_denial(text: &str) -> bool {
    let text = text.trim();
    matches!(
        text,
        "不" | "不是" | "不对" | "错" | "错了" | "否" | "没有" | "没"
    ) || text.starts_with("不是")
        || text.starts_with("不对")
        || text.starts_with("错了")
        || text.starts_with("错，")
        || text.starts_with("错,")
        || text.starts_with("否")
        || text.starts_with("不，")
        || text.starts_with("不,")
}
fn correction_tail(text: &str) -> Option<String> {
    for marker in ["而是", "改是", "应该是", "其实是"] {
        if let Some(index) = text.rfind(marker) {
            let tail = text[index + marker.len()..].trim();
            if !tail.is_empty() {
                return Some(tail.to_string());
            }
        }
    }
    let denial_start = ["不是", "不对", "错了", "错", "不"]
        .iter()
        .filter_map(|marker| text.find(marker))
        .min()?;
    let after_denial = &text[denial_start..];
    let denial = ["不是", "不对", "错了", "错", "不"]
        .iter()
        .find(|marker| after_denial.starts_with(*marker))?;
    let tail_zone = &after_denial[denial.len()..];
    let index = tail_zone.rfind('是')?;
    let tail = tail_zone[index + '是'.len_utf8()..].trim();
    if tail.is_empty() {
        None
    } else {
        Some(tail.to_string())
    }
}
fn extract_values(text: &str) -> Vec<String> {
    let text = clean_value(text);
    if text.is_empty() {
        return Vec::new();
    }
    let mut values = Vec::new();
    // 一个槽位只取主句：逗号后的补充说明（如「叶莉，1980 年出生」）不当作第二取值。
    let head = text.split(['，', ',']).next().unwrap_or(&text);
    for part in head.split(['、', '；', ';']) {
        let value = clean_value(part);
        if value.is_empty() || value == "谁" || value == "什么" {
            continue;
        }
        values.push(value);
    }
    values.sort();
    values.dedup();
    values
}
fn build_claim_for_goal(goal: &RelationGoal, value: &str) -> Claim {
    let literal = match goal.expectation.value_shape {
        ValueShape::Literal => true,
        ValueShape::Entity => false,
        ValueShape::Infer => looks_like_literal(value),
    };
    let normalize_object = |value: &str| {
        if literal {
            ClaimObject::Literal(value.to_string())
        } else {
            ClaimObject::Entity(value.to_string())
        }
    };
    if goal.expectation.inverse {
        Claim {
            subject: value.to_string(),
            predicate: goal.predicate.clone(),
            object: normalize_object(&goal.subject),
        }
    } else {
        Claim {
            subject: goal.subject.clone(),
            predicate: goal.predicate.clone(),
            object: normalize_object(value),
        }
    }
}
fn looks_like_literal(value: &str) -> bool {
    let trimmed = value.trim();
    trimmed.parse::<f64>().is_ok()
        || matches!(trimmed, "是" | "否" | "true" | "false")
        || crate::shape_gate::datatype_conforms("date", trimmed)
        || crate::shape_gate::datatype_conforms("time", trimmed)
        || crate::shape_gate::datatype_conforms("datetime", trimmed)
}
fn normalize_claim(claim: Claim, registry: &PredicateRegistry) -> Claim {
    let resolution = registry.resolve(&claim.predicate);
    if !resolution.inverted {
        Claim {
            subject: claim.subject,
            predicate: resolution.canonical,
            object: claim.object,
        }
    } else {
        match claim.object {
            ClaimObject::Entity(object) => Claim {
                subject: object,
                predicate: resolution.canonical,
                object: ClaimObject::Entity(claim.subject),
            },
            ClaimObject::Literal(value) => Claim {
                subject: claim.subject,
                predicate: resolution.canonical,
                object: ClaimObject::Literal(value),
            },
        }
    }
}
fn claim_key(claim: &Claim) -> String {
    canonical_json(&serde_json::json!({
        "subject": claim.subject,
        "predicate": claim.predicate,
        "object": claim.object,
    }))
}
fn claim_id(claim: &Claim) -> String {
    format!("qc_{}", fnv1a_hex(claim_key(claim).as_bytes()))
}
fn commit_sentence(claim: &Claim) -> String {
    format!(
        "{}的{}是{}",
        claim.subject,
        claim.predicate,
        claim.object.text()
    )
}
fn merge_outcomes(act: AnswerAct, outcomes: Vec<AcquisitionOutcome>) -> AcquisitionOutcome {
    let mut message = Vec::new();
    let mut claim_id = None;
    let mut status = None;
    let mut commit = None;
    let mut follow_up = None;
    for outcome in outcomes {
        if !outcome.message.is_empty() {
            message.push(outcome.message);
        }
        if outcome.claim_id.is_some() {
            claim_id = outcome.claim_id;
        }
        if outcome.status.is_some() {
            status = outcome.status;
        }
        if outcome.commit.is_some() {
            commit = outcome.commit;
        }
        if outcome.follow_up.is_some() {
            follow_up = outcome.follow_up;
        }
    }
    AcquisitionOutcome {
        act,
        status,
        claim_id,
        message: message.join("\n"),
        commit,
        follow_up,
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
struct ClassifiedAnswer {
    act: AnswerAct,
    values: Vec<String>,
}
impl ClassifiedAnswer {
    fn new(act: AnswerAct, values: Vec<String>) -> Self {
        Self { act, values }
    }

    fn contains_any(text: &str, needles: &[&str]) -> bool {
        needles.iter().any(|needle| text.contains(needle))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::predicate_registry::PredicateProfileConfig;
    use crate::type_system::{TypeAliasConfig, TypeDisjointConfig, TypeSystemConfig};
    use std::collections::HashMap;

    fn entity(name: &str, etype: &str, properties: &[(&str, &str)]) -> PrologFact {
        PrologFact::Entity {
            name: name.to_string(),
            entity_type: etype.to_string(),
            properties: properties
                .iter()
                .map(|(key, value)| (key.to_string(), value.to_string()))
                .collect::<HashMap<_, _>>(),
        }
    }

    fn relation(from: &str, predicate: &str, to: &str) -> PrologFact {
        PrologFact::Relation {
            from: from.to_string(),
            rel_type: predicate.to_string(),
            to: to.to_string(),
        }
    }

    fn profile(
        name: &str,
        functional: bool,
        domain: &[&str],
        range: &[&str],
    ) -> PredicateProfileConfig {
        PredicateProfileConfig {
            name: name.to_string(),
            functional,
            domain: domain.iter().map(|value| value.to_string()).collect(),
            range: range.iter().map(|value| value.to_string()).collect(),
            ..Default::default()
        }
    }

    fn registry() -> PredicateRegistry {
        let mut registry = PredicateRegistry::new();
        registry.declare(&profile("女儿", false, &["人物"], &["人物"]));
        registry.declare(&profile("年龄", true, &[], &[]));
        registry
    }

    fn types() -> TypeSystem {
        TypeSystem::from_config(&TypeSystemConfig {
            closure_gate: true,
            aliases: vec![TypeAliasConfig {
                alias: "Person".to_string(),
                canonical: "人物".to_string(),
            }],
            subclasses: Vec::new(),
            disjoint: vec![TypeDisjointConfig {
                left: "人物".to_string(),
                right: "地点".to_string(),
            }],
        })
    }

    fn shapes(registry: &PredicateRegistry, types: &TypeSystem) -> ShapeSet {
        ShapeSet::from_schema(&[], registry, types)
    }

    fn facts() -> Vec<PrologFact> {
        vec![
            entity("姚明", "Person", &[("name", "姚明")]),
            entity("叶莉", "Person", &[("name", "叶莉")]),
            entity("邓紫棋", "Person", &[("name", "邓紫棋")]),
            entity("李明", "Person", &[("name", "李明")]),
        ]
    }

    #[test]
    fn parses_possession_questions_and_reverse_who_questions() {
        let parsed = RelationAcquisition::parse_question("姚明的女儿是谁？").unwrap();
        assert_eq!(parsed.subject, "姚明");
        assert_eq!(parsed.predicate, "女儿");
        assert_eq!(parsed.form, QuestionForm::Who);

        let reverse = RelationAcquisition::parse_question("谁是姚明的女儿").unwrap();
        assert_eq!(reverse.subject, "姚明");
        assert_eq!(reverse.predicate, "女儿");

        assert!(RelationAcquisition::parse_question("姚明的女儿").is_none());
    }

    #[test]
    fn question_opens_goal_with_range_expectation() {
        let registry = registry();
        let types = types();
        let shapes = shapes(&registry, &types);
        let mut acquisition = RelationAcquisition::new();
        let opening =
            acquisition.start_question("姚明的女儿是谁？", &facts(), &registry, &types, &shapes);
        let GoalOpen::Opened { goal } = opening else {
            panic!("应打开关系目标：{:?}", opening);
        };
        assert_eq!(goal.subject, "姚明");
        assert_eq!(goal.predicate, "女儿");
        assert_eq!(goal.expectation.entity_types, vec!["人物".to_string()]);
        assert!(!goal.expectation.functional);
        assert!(goal.question.contains("谁"));
        assert!(acquisition.has_pending_goal());
    }

    #[test]
    fn short_answer_is_proposed_and_confirmation_promotes_to_overlay() {
        let registry = registry();
        let types = types();
        let shapes = shapes(&registry, &types);
        let facts = facts();
        let mut acquisition = RelationAcquisition::new();
        let _ = acquisition.start_question("姚明的女儿是谁？", &facts, &registry, &types, &shapes);

        let outcome = acquisition.interpret_answer("叶莉", &facts, &registry, &types, &shapes);
        assert_eq!(outcome.act, AnswerAct::Provide);
        assert_eq!(outcome.status, Some(AcquisitionStatus::Proposed));
        assert!(acquisition.overlay().is_empty(), "候选不应直接进入覆盖层");

        let confirmed = acquisition.confirm(None);
        assert_eq!(confirmed.status, Some(AcquisitionStatus::Confirmed));
        let overlay = acquisition.overlay();
        assert_eq!(overlay.values("姚明", "女儿").len(), 1);
        assert_eq!(overlay.values("姚明", "女儿")[0].object, "叶莉");

        let answered = acquisition
            .answer_from_overlay("姚明的女儿是谁？", &registry)
            .expect("覆盖层应能回答已确认的关系");
        assert_eq!(answered.values, vec!["叶莉".to_string()]);
        assert!(answered.render_text().contains("未写入知识图谱"));
    }

    #[test]
    fn correction_refutes_previous_candidate_and_replaces_it() {
        let registry = registry();
        let types = types();
        let shapes = shapes(&registry, &types);
        let facts = facts();
        let mut acquisition = RelationAcquisition::new();
        let _ = acquisition.start_question("姚明的女儿是谁？", &facts, &registry, &types, &shapes);
        let first = acquisition.interpret_answer("叶莉", &facts, &registry, &types, &shapes);
        let first_id = first.claim_id.unwrap();

        let corrected =
            acquisition.interpret_answer("不是叶莉，是邓紫棋", &facts, &registry, &types, &shapes);
        assert_eq!(corrected.act, AnswerAct::Correct);
        assert_eq!(
            acquisition.claim(&first_id).unwrap().status,
            AcquisitionStatus::Refuted
        );
        assert_eq!(corrected.status, Some(AcquisitionStatus::Confirmed));
        let overlay = acquisition.overlay();
        let values: Vec<&str> = overlay
            .values("姚明", "女儿")
            .iter()
            .map(|row| row.object.as_str())
            .collect();
        assert_eq!(values, vec!["邓紫棋"]);
    }

    #[test]
    fn unknown_answer_does_not_write_overlay() {
        let registry = registry();
        let types = types();
        let shapes = shapes(&registry, &types);
        let facts = facts();
        let mut acquisition = RelationAcquisition::new();
        let _ = acquisition.start_question("姚明的女儿是谁？", &facts, &registry, &types, &shapes);
        let outcome = acquisition.interpret_answer("我不知道", &facts, &registry, &types, &shapes);
        assert_eq!(outcome.act, AnswerAct::Unknown);
        assert!(acquisition.overlay().is_empty());
        assert!(!acquisition.has_pending_goal());
    }

    #[test]
    fn two_independent_answers_promote_same_value() {
        let registry = registry();
        let types = types();
        let shapes = shapes(&registry, &types);
        let facts = facts();
        let mut acquisition = RelationAcquisition::new();
        let _ = acquisition.start_question("姚明的女儿是谁？", &facts, &registry, &types, &shapes);
        let first = acquisition.interpret_answer("叶莉", &facts, &registry, &types, &shapes);
        assert_eq!(first.status, Some(AcquisitionStatus::Proposed));
        let _ = acquisition.start_question("姚明的女儿是谁？", &facts, &registry, &types, &shapes);
        let second = acquisition.interpret_answer("叶莉", &facts, &registry, &types, &shapes);
        assert_eq!(second.status, Some(AcquisitionStatus::Confirmed));
        assert_eq!(acquisition.overlay().rows.len(), 1);
    }

    #[test]
    fn functional_slot_conflict_is_disclosed_and_user_can_adjudicate() {
        let registry = registry();
        let types = types();
        let shapes = shapes(&registry, &types);
        let facts = facts();
        let mut acquisition = RelationAcquisition::new();
        let goal = RelationGoal {
            id: "qg_test".to_string(),
            subject: "姚明".to_string(),
            predicate: "年龄".to_string(),
            surface: "年龄".to_string(),
            expectation: Expectation {
                entity_types: Vec::new(),
                datatype: None,
                functional: true,
                inverse: false,
                required: false,
                value_shape: ValueShape::Infer,
            },
            origin: GoalOrigin::UserRequest,
            form: QuestionForm::Value,
            question: "姚明的年龄是多少？".to_string(),
            turn: 1,
            asked: 1,
        };
        acquisition.active_goal = Some(goal.clone());
        acquisition.turn = 1;
        let first = acquisition.interpret_answer("18", &facts, &registry, &types, &shapes);
        assert_eq!(first.status, Some(AcquisitionStatus::Proposed));
        acquisition.active_goal = Some(goal);
        acquisition.turn += 1;
        let second = acquisition.interpret_answer("19", &facts, &registry, &types, &shapes);
        assert_eq!(second.status, Some(AcquisitionStatus::Conflicted));
        assert!(second.message.contains("冲突"));

        let adjudicated = acquisition.confirm(second.claim_id.as_deref());
        assert_eq!(adjudicated.status, Some(AcquisitionStatus::Confirmed));
        assert_eq!(
            acquisition
                .claims()
                .filter(|claim| claim.claim.subject == "姚明" && claim.claim.predicate == "年龄")
                .filter(|claim| claim.status == AcquisitionStatus::Refuted)
                .count(),
            1
        );
    }

    #[test]
    fn induced_pattern_activates_after_two_confirmations_and_scans_text() {
        let registry = registry();
        let types = types();
        let shapes = shapes(&registry, &types);
        let facts = facts();
        let mut acquisition = RelationAcquisition::new();
        for (subject, value) in [("姚明", "叶莉"), ("李明", "邓紫棋")] {
            let question = format!("{}的女儿是谁？", subject);
            let _ = acquisition.start_question(&question, &facts, &registry, &types, &shapes);
            let _ = acquisition.interpret_answer(value, &facts, &registry, &types, &shapes);
            let _ = acquisition.confirm(None);
        }
        assert_eq!(acquisition.patterns().count(), 1);
        assert!(acquisition.patterns().next().unwrap().active);

        let scan = acquisition.scan_text("王强的女儿是赵敏。", &registry);
        assert_eq!(scan.claims.len(), 1);
        assert_eq!(scan.claims[0].subject, "王强");
        assert_eq!(scan.claims[0].object.text(), "赵敏");
    }

    #[test]
    fn adopted_claims_remain_visible_in_overlay_and_snapshot_roundtrips() {
        let registry = registry();
        let types = types();
        let shapes = shapes(&registry, &types);
        let facts = facts();
        let mut acquisition = RelationAcquisition::new();
        let _ = acquisition.start_question("姚明的女儿是谁？", &facts, &registry, &types, &shapes);
        let outcome = acquisition.interpret_answer("叶莉", &facts, &registry, &types, &shapes);
        let id = outcome.claim_id.unwrap();
        let _ = acquisition.confirm(Some(&id));
        assert!(acquisition.mark_admitted(&id));
        assert_eq!(
            acquisition.claim(&id).unwrap().status,
            AcquisitionStatus::Admitted
        );
        assert!(!acquisition.overlay().is_empty());

        let snapshot = acquisition.snapshot();
        let mut restored = RelationAcquisition::new();
        assert!(restored.restore(&snapshot));
        assert_eq!(restored.digest(), acquisition.digest());
        assert_eq!(restored.overlay().rows, acquisition.overlay().rows);
    }

    #[test]
    fn already_known_graph_value_does_not_open_goal() {
        let registry = registry();
        let types = types();
        let shapes = shapes(&registry, &types);
        let facts = vec![
            entity("姚明", "Person", &[("name", "姚明")]),
            entity("叶莉", "Person", &[("name", "叶莉")]),
            relation("姚明", "女儿", "叶莉"),
        ];
        let mut acquisition = RelationAcquisition::new();
        let opening =
            acquisition.start_question("姚明的女儿是谁？", &facts, &registry, &types, &shapes);
        assert!(matches!(opening, GoalOpen::AlreadyKnown { .. }));
        assert!(!acquisition.has_pending_goal());
    }

    #[test]
    fn denial_without_replacement_is_not_mistaken_for_correction() {
        let registry = registry();
        let types = types();
        let shapes = shapes(&registry, &types);
        let facts = facts();
        let mut acquisition = RelationAcquisition::new();
        let _ = acquisition.start_question("姚明的女儿是谁？", &facts, &registry, &types, &shapes);
        let _ = acquisition.interpret_answer("叶莉", &facts, &registry, &types, &shapes);
        let outcome = acquisition.interpret_answer("不是叶莉", &facts, &registry, &types, &shapes);
        assert_eq!(outcome.act, AnswerAct::Deny);
        assert!(acquisition.overlay().is_empty());
        assert!(acquisition
            .claims()
            .all(|claim| claim.status == AcquisitionStatus::Refuted));
    }

    #[test]
    fn valid_value_starting_with_bu_is_not_treated_as_denial() {
        let classified = classify_answer("不丹");
        assert_eq!(classified.act, AnswerAct::Provide);
        assert_eq!(classified.values, vec!["不丹".to_string()]);
    }

    #[test]
    fn shape_gap_goal_records_its_origin() {
        let registry = registry();
        let types = types();
        let shapes = shapes(&registry, &types);
        let facts = vec![entity("姚明", "Person", &[])];
        let mut acquisition = RelationAcquisition::new();
        let opening =
            acquisition.start_shape_gap("姚明", "女儿", &facts, &registry, &types, &shapes);
        let GoalOpen::Opened { goal } = opening else {
            panic!("形状缺口应打开目标：{:?}", opening);
        };
        assert_eq!(goal.origin, GoalOrigin::ShapeRepair);
    }

    #[test]
    fn free_assertion_is_proposed_and_confirmation_reaches_overlay() {
        let mut registry = PredicateRegistry::new();
        registry.declare(&profile("任职于", false, &["人物"], &["机构"]));
        let types = types();
        let shapes = shapes(&registry, &types);
        let facts = vec![entity("张三", "Person", &[])];
        let mut acquisition = RelationAcquisition::new();
        let outcome = acquisition.record_assertion(
            Claim::entity("张三", "任职于", "清华大学"),
            AcquisitionSource::UserAssertion,
            "张三在清华大学工作。",
            "断言阅读",
            &facts,
            &registry,
            &types,
            &shapes,
        );
        assert_eq!(outcome.status, Some(AcquisitionStatus::Proposed));
        assert!(acquisition.overlay().is_empty());

        let id = outcome.claim_id.unwrap();
        let confirmed = acquisition.confirm(Some(&id));
        assert_eq!(confirmed.status, Some(AcquisitionStatus::Confirmed));
        let overlay = acquisition.overlay();
        assert_eq!(overlay.values("张三", "任职于").len(), 1);
        assert_eq!(overlay.values("张三", "任职于")[0].object, "清华大学");
    }

    #[test]
    fn where_work_question_parses_and_answers_from_overlay() {
        let mut registry = PredicateRegistry::new();
        registry.declare(&profile("任职于", false, &["人物"], &["机构"]));
        let types = types();
        let shapes = shapes(&registry, &types);
        let facts = vec![entity("张三", "Person", &[])];
        let mut acquisition = RelationAcquisition::new();
        let outcome = acquisition.record_assertion(
            Claim::entity("张三", "任职于", "清华大学"),
            AcquisitionSource::UserAssertion,
            "张三在清华大学工作。",
            "断言阅读",
            &facts,
            &registry,
            &types,
            &shapes,
        );
        let id = outcome.claim_id.unwrap();
        let _ = acquisition.confirm(Some(&id));

        let parsed = RelationAcquisition::parse_question("张三在哪工作？").unwrap();
        assert_eq!(parsed.subject, "张三");
        assert_eq!(parsed.predicate, "任职于");
        assert_eq!(parsed.form, QuestionForm::Where);
        let answer = acquisition
            .answer_from_overlay("张三在哪工作？", &registry)
            .expect("覆盖层应能回答任职问句");
        assert_eq!(answer.values, vec!["清华大学".to_string()]);
    }

    #[test]
    fn inverse_question_is_normalized_to_canonical_direction() {
        let mut registry = PredicateRegistry::new();
        registry.declare(&profile("师傅", false, &["人物"], &["人物"]));
        registry.register_inverse("师傅", "徒弟", "测试");
        let types = types();
        let shapes = shapes(&registry, &types);
        let facts = vec![
            entity("姚明", "Person", &[("name", "姚明")]),
            entity("叶莉", "Person", &[("name", "叶莉")]),
            relation("姚明", "师傅", "叶莉"),
        ];
        let mut acquisition = RelationAcquisition::new();
        let opening =
            acquisition.start_question("叶莉的徒弟是谁？", &facts, &registry, &types, &shapes);
        assert!(matches!(opening, GoalOpen::AlreadyKnown { .. }));
    }
}

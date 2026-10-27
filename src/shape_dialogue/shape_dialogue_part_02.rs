fn compile_gap(violation: &ShapeViolation, registry: &PredicateRegistry) -> Option<ShapeGap> {
    let (kind, property, values, expected) = match &violation.repair {
        RepairAction::AskProperty { property, .. }
            if violation.constraint == ConstraintKind::MinCount =>
        {
            (
                GapKind::Missing,
                property.clone(),
                Vec::new(),
                violation.expected.clone(),
            )
        }
        RepairAction::DiscloseConflict {
            property, values, ..
        } => (
            GapKind::Conflict,
            property.clone(),
            values.clone(),
            violation.expected.clone(),
        ),
        _ => return None,
    };
    let required = violation.severity == Severity::Violation;
    let surface = surface_for(&property, registry);
    let severity_score = match violation.severity {
        Severity::Violation => 300,
        Severity::Warning => 200,
        Severity::Info => 100,
    };
    let score = severity_score
        + if required { 60 } else { 0 }
        + if kind == GapKind::Conflict { 40 } else { 0 };
    let cost = if kind == GapKind::Conflict {
        3
    } else if required {
        1
    } else {
        2
    };
    let question = if kind == GapKind::Conflict {
        format!(
            "关于「{}」的「{}」我听到多个取值：{}；系统不会替你选，需要先裁定。",
            violation.focus_node,
            surface,
            values.join("、")
        )
    } else {
        format!(
            "你知道「{}」的「{}」吗？（期望：{}）",
            violation.focus_node,
            surface,
            if expected.is_empty() {
                "一个具体取值".to_string()
            } else {
                expected.join("、")
            }
        )
    };
    let id = format!(
        "sg_{}",
        fnv1a_hex(
            canonical_json(&serde_json::json!({
                "entity": &violation.focus_node,
                "predicate": &property,
                "kind": kind,
            }))
            .as_bytes(),
        )
    );
    Some(ShapeGap {
        id,
        entity: violation.focus_node.clone(),
        predicate: property,
        surface,
        kind,
        severity: violation.severity,
        required,
        expected,
        values,
        score,
        cost,
        question,
    })
}
fn slot_conflict(predicate: &str, values: &[String], facts: &[AnswerFact]) -> SlotConflict {
    let mut rows = Vec::new();
    for value in values {
        let mut sources = BTreeSet::new();
        for fact in facts {
            if fact.predicate == predicate && fact.object == *value {
                sources.extend(fact.sources.iter().copied());
            }
        }
        rows.push(ConflictValue {
            value: value.clone(),
            sources: sources.into_iter().collect(),
        });
    }
    rows.sort_by(|left, right| left.value.cmp(&right.value));
    rows.dedup_by(|left, right| left.value == right.value);
    SlotConflict {
        predicate: predicate.to_string(),
        surface: display_predicate(predicate),
        values: rows,
    }
}
fn push_fact(
    facts: &mut Vec<AnswerFact>,
    predicate: String,
    object: String,
    direction: FactDirection,
    source: FactSource,
    claim_id: Option<String>,
) {
    if let Some(existing) = facts.iter_mut().find(|fact| {
        fact.predicate == predicate && fact.object == object && fact.direction == direction
    }) {
        if !existing.sources.contains(&source) {
            existing.sources.push(source);
            existing.sources.sort();
        }
        if existing.claim_id.is_none() {
            existing.claim_id = claim_id;
        }
        return;
    }
    facts.push(AnswerFact {
        predicate,
        object,
        direction,
        sources: vec![source],
        claim_id,
    });
}
fn ask_surface(gap: &ShapeGap, registry: &PredicateRegistry) -> String {
    for (alias, canonical, inverted) in registry.alias_pairs() {
        if !inverted && canonical == gap.predicate && !alias.trim().is_empty() {
            return alias;
        }
    }
    if registry.resolve(&gap.surface).canonical == gap.predicate {
        return gap.surface.clone();
    }
    gap.predicate.clone()
}
fn surface_for(predicate: &str, registry: &PredicateRegistry) -> String {
    for (alias, canonical, inverted) in registry.alias_pairs() {
        if !inverted && canonical == predicate && !alias.trim().is_empty() {
            return alias;
        }
    }
    localized_predicate_key(predicate)
        .map(|label| label.to_string())
        .unwrap_or_else(|| predicate.to_string())
}
fn display_predicate(predicate: &str) -> String {
    localized_predicate_key(predicate)
        .map(|label| label.to_string())
        .unwrap_or_else(|| predicate.to_string())
}
fn is_name_key(key: &str) -> bool {
    matches!(
        key.trim().to_ascii_lowercase().as_str(),
        "name" | "姓名" | "名称" | "名字"
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::predicate_registry::PredicateProfileConfig;
    use crate::relation_acquisition::{SessionOverlay, SessionOverlayRow};
    use crate::shape_gate::ShapeSet;
    use crate::type_system::{TypeAliasConfig, TypeDisjointConfig, TypeSystemConfig};
    use crate::{EntityTypeConfig, PropertyConfig};
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
        registry.declare(&profile("出生地", false, &["人物"], &[]));
        registry.declare(&profile("国籍", false, &["人物"], &[]));
        registry.declare(&profile("职业", true, &[], &[]));
        registry.declare(&profile("女儿", false, &["人物"], &["人物"]));
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
        ShapeSet::from_schema(
            &[EntityTypeConfig {
                label: "Person".to_string(),
                examples: Vec::new(),
                properties: vec![
                    PropertyConfig {
                        name: "出生地".to_string(),
                        data_type: "string".to_string(),
                        required: true,
                        extraction_patterns: Vec::new(),
                    },
                    PropertyConfig {
                        name: "职业".to_string(),
                        data_type: "string".to_string(),
                        required: false,
                        extraction_patterns: Vec::new(),
                    },
                ],
            }],
            registry,
            types,
        )
    }

    fn facts() -> Vec<PrologFact> {
        vec![
            entity("刘德华", "Person", &[("name", "刘德华"), ("职业", "演员")]),
            entity("姚明", "Person", &[("name", "姚明"), ("出生地", "上海")]),
            entity("王五", "Person", &[("name", "王五")]),
        ]
    }

    #[test]
    fn partial_profile_asks_one_required_gap_then_confirmation_completes_it() {
        let registry = registry();
        let types = types();
        let shapes = shapes(&registry, &types);
        let facts = facts();
        let mut acquisition = RelationAcquisition::new();
        let mut dialogue = ShapeDialogue::new();

        let first = dialogue.start_profile(
            "刘德华",
            "刘德华是谁？",
            &facts,
            &registry,
            &types,
            &shapes,
            &mut acquisition,
        );
        assert_eq!(first.answer.status, AnswerStatus::Partial);
        assert_eq!(first.answer.missing_predicates(), vec!["出生地"]);
        let consultation = first.consultation.expect("部分画像应追问一个缺口");
        assert_eq!(consultation.gap.predicate, "出生地");
        assert!(consultation.question.contains("出生地"));
        assert!(first.message.contains("只问一个问题"));

        let proposed = dialogue
            .respond("香港", &facts, &registry, &types, &shapes, &mut acquisition)
            .expect("待答状态应处理短答");
        assert_eq!(proposed.answer.status, AnswerStatus::Partial);
        assert!(proposed.answer.value_of("出生地").is_empty());
        assert!(acquisition.overlay().is_empty());

        let confirmed = dialogue
            .respond("确认", &facts, &registry, &types, &shapes, &mut acquisition)
            .expect("待答状态应处理确认");
        assert_eq!(confirmed.answer.status, AnswerStatus::Complete);
        assert_eq!(confirmed.answer.value_of("出生地"), vec!["香港"]);
        assert!(confirmed.answer.render_text().contains("本会话"));
        assert!(confirmed.message.contains("刘德华是谁？"));
        assert!(confirmed.consultation.is_none());
        assert_eq!(acquisition.overlay().values("刘德华", "出生地").len(), 1);
    }

    #[test]
    fn graph_functional_conflict_is_disclosed_and_never_silently_resolved() {
        let registry = registry();
        let types = types();
        let shapes = shapes(&registry, &types);
        let facts = vec![
            entity("刘德华", "Person", &[("name", "刘德华"), ("职业", "演员")]),
            relation("刘德华", "职业", "导演"),
        ];
        let mut acquisition = RelationAcquisition::new();
        let mut dialogue = ShapeDialogue::new();
        let turn = dialogue.start_profile(
            "刘德华",
            "刘德华是谁？",
            &facts,
            &registry,
            &types,
            &shapes,
            &mut acquisition,
        );
        assert_eq!(turn.answer.status, AnswerStatus::Conflict);
        assert_eq!(turn.answer.conflicts.len(), 1);
        let values: Vec<&str> = turn.answer.conflicts[0]
            .values
            .iter()
            .map(|value| value.value.as_str())
            .collect();
        assert_eq!(values, vec!["导演", "演员"]);
        assert!(turn.message.contains("冲突"));
        assert!(turn.message.contains("不替你选"));
        assert!(turn.consultation.is_none(), "冲突不能变成普通填空追问");
    }

    #[test]
    fn dangling_relation_node_is_not_silently_typed() {
        let registry = registry();
        let types = types();
        let shapes = shapes(&registry, &types);
        let facts = vec![relation("刘德华", "出生地", "香港")];
        let (answer, _) = ShapeAnswer::build(
            "香港",
            &facts,
            &SessionOverlay::default(),
            &shapes,
            &registry,
            &types,
        );
        assert!(answer.entity_type.is_none());
        assert!(answer.render_text().contains("类型未声明"));
    }

    #[test]
    fn datatype_violation_is_disclosed_instead_of_claimed_complete() {
        let registry = registry();
        let types = types();
        let shapes = ShapeSet::from_schema(
            &[EntityTypeConfig {
                label: "Person".to_string(),
                examples: Vec::new(),
                properties: vec![PropertyConfig {
                    name: "年龄".to_string(),
                    data_type: "integer".to_string(),
                    required: false,
                    extraction_patterns: Vec::new(),
                }],
            }],
            &registry,
            &types,
        );
        let facts = vec![entity(
            "刘德华",
            "Person",
            &[("name", "刘德华"), ("年龄", "十八")],
        )];
        let mut acquisition = RelationAcquisition::new();
        let mut dialogue = ShapeDialogue::new();
        let turn = dialogue.start_profile(
            "刘德华",
            "刘德华是谁？",
            &facts,
            &registry,
            &types,
            &shapes,
            &mut acquisition,
        );
        assert_eq!(turn.answer.status, AnswerStatus::Conflict);
        assert!(!turn.answer.issues.is_empty());
        assert!(dialogue.agenda().blocked.contains("刘德华"));
        assert!(dialogue.agenda().next_global(&BTreeMap::new()).is_none());
        assert!(turn.message.contains("形状问题"));
        assert!(turn.message.contains("不会把不合形状的取值当作确定事实"));
        assert!(
            turn.consultation.is_none(),
            "类型问题不能被当作普通缺口继续追问"
        );
    }

    #[test]
    fn unknown_entity_abstains_without_fabricating_a_profile() {
        let registry = registry();
        let types = types();
        let shapes = shapes(&registry, &types);
        let facts = facts();
        let mut acquisition = RelationAcquisition::new();
        let mut dialogue = ShapeDialogue::new();
        let turn = dialogue.start_profile(
            "未名科技",
            "未名科技是谁？",
            &facts,
            &registry,
            &types,
            &shapes,
            &mut acquisition,
        );
        assert_eq!(turn.answer.status, AnswerStatus::Unknown);
        assert!(turn.answer.facts.is_empty());
        assert!(turn.message.contains("不会为它编造画像"));
        assert!(turn.consultation.is_none());
        assert!(!acquisition.has_pending_goal());
    }

    #[test]
    fn snapshot_restores_pending_question_and_original_task() {
        let registry = registry();
        let types = types();
        let shapes = shapes(&registry, &types);
        let facts = facts();
        let mut acquisition = RelationAcquisition::new();
        let mut dialogue = ShapeDialogue::new();
        let _ = dialogue.start_profile(
            "王五",
            "王五是谁？",
            &facts,
            &registry,
            &types,
            &shapes,
            &mut acquisition,
        );
        assert!(dialogue.has_pending());
        let snapshot = dialogue.snapshot();
        let acquisition_snapshot = acquisition.snapshot();
        let mut restored = ShapeDialogue::new();
        assert!(restored.restore(&snapshot));
        let mut restored_acquisition = RelationAcquisition::new();
        assert!(restored_acquisition.restore(&acquisition_snapshot));
        assert!(restored.has_pending());
        assert_eq!(restored.original_question(), Some("王五是谁？"));

        let proposed = restored
            .respond(
                "北京",
                &facts,
                &registry,
                &types,
                &shapes,
                &mut restored_acquisition,
            )
            .unwrap();
        assert_eq!(proposed.answer.status, AnswerStatus::Partial);
        let confirmed = restored
            .respond(
                "确认",
                &facts,
                &registry,
                &types,
                &shapes,
                &mut restored_acquisition,
            )
            .unwrap();
        assert_eq!(confirmed.answer.status, AnswerStatus::Complete);
        assert!(confirmed.message.contains("王五是谁？"));
        assert_eq!(confirmed.answer.value_of("出生地"), vec!["北京"]);
    }

    #[test]
    fn session_overlay_does_not_leak_into_a_fresh_session() {
        let registry = registry();
        let types = types();
        let shapes = shapes(&registry, &types);
        let facts = facts();
        let mut acquisition = RelationAcquisition::new();
        let mut dialogue = ShapeDialogue::new();
        let _ = dialogue.start_profile(
            "刘德华",
            "刘德华是谁？",
            &facts,
            &registry,
            &types,
            &shapes,
            &mut acquisition,
        );
        let _ = dialogue.respond("香港", &facts, &registry, &types, &shapes, &mut acquisition);
        let _ = dialogue.respond("确认", &facts, &registry, &types, &shapes, &mut acquisition);
        let (completed, _) = ShapeAnswer::build(
            "刘德华",
            &facts,
            &acquisition.overlay(),
            &shapes,
            &registry,
            &types,
        );
        assert_eq!(completed.status, AnswerStatus::Complete);

        let fresh_acquisition = RelationAcquisition::new();
        let (fresh, _) = ShapeAnswer::build(
            "刘德华",
            &facts,
            &fresh_acquisition.overlay(),
            &shapes,
            &registry,
            &types,
        );
        assert_eq!(fresh.status, AnswerStatus::Partial);
        assert!(fresh.value_of("出生地").is_empty());
    }

    #[test]
    fn agenda_orders_missing_gaps_and_refuses_to_ask_conflicts() {
        let registry = registry();
        let types = types();
        let shapes = shapes(&registry, &types);
        let facts = vec![
            entity(
                "刘德华",
                "Person",
                &[("name", "刘德华"), ("出生地", "香港"), ("职业", "演员")],
            ),
            relation("刘德华", "职业", "导演"),
            entity("王五", "Person", &[("name", "王五")]),
        ];
        let report = shapes.validate(&facts, &registry, &types);
        let agenda = ShapeAgenda::compile(&report, &registry);
        let missing = agenda
            .gaps
            .iter()
            .find(|gap| gap.entity == "王五")
            .expect("王五应编译出缺失缺口");
        assert_eq!(missing.kind, GapKind::Missing);
        assert_eq!(missing.predicate, "出生地");
        assert!(missing.required);
        assert!(agenda
            .gaps
            .iter()
            .any(|gap| gap.entity == "刘德华" && gap.kind == GapKind::Conflict));
        assert!(agenda.blocked.contains("刘德华"));

        let mut attempts = BTreeMap::new();
        assert_eq!(
            agenda.next_for("刘德华", &attempts),
            None,
            "冲突不是可追问的填空缺口"
        );
        attempts.insert(missing.id.clone(), MAX_ASK_ATTEMPTS);
        assert_eq!(agenda.next_for("王五", &attempts), None);
    }

    #[test]
    fn confirmation_advances_to_the_next_gap_without_repeating() {
        let registry = registry();
        let types = types();
        let shapes = ShapeSet::from_schema(
            &[EntityTypeConfig {
                label: "Person".to_string(),
                examples: Vec::new(),
                properties: vec![
                    PropertyConfig {
                        name: "出生地".to_string(),
                        data_type: "string".to_string(),
                        required: true,
                        extraction_patterns: Vec::new(),
                    },
                    PropertyConfig {
                        name: "国籍".to_string(),
                        data_type: "string".to_string(),
                        required: true,
                        extraction_patterns: Vec::new(),
                    },
                ],
            }],
            &registry,
            &types,
        );
        let facts = vec![entity("刘德华", "Person", &[("name", "刘德华")])];
        let mut acquisition = RelationAcquisition::new();
        let mut dialogue = ShapeDialogue::new();
        let first = dialogue.start_profile(
            "刘德华",
            "刘德华是谁？",
            &facts,
            &registry,
            &types,
            &shapes,
            &mut acquisition,
        );
        assert!(first.answer.missing.len() >= 2);
        let first_gap = first.consultation.unwrap().gap.predicate;
        let value = if first_gap == "出生地" {
            "香港"
        } else {
            "中国"
        };
        let _ = dialogue.respond(value, &facts, &registry, &types, &shapes, &mut acquisition);
        let second = dialogue
            .respond("确认", &facts, &registry, &types, &shapes, &mut acquisition)
            .unwrap();
        assert_eq!(second.answer.status, AnswerStatus::Partial);
        let next = second.consultation.expect("补一个缺口后应继续问下一个");
        assert_ne!(next.gap.predicate, first_gap);
        assert_eq!(
            dialogue.attempts.get(&next.gap.id).copied(),
            Some(1),
            "每个缺口只问一次"
        );
    }

    #[test]
    fn global_agenda_question_carries_its_own_original_task() {
        let registry = registry();
        let types = types();
        let shapes = shapes(&registry, &types);
        let facts = facts();
        let mut acquisition = RelationAcquisition::new();
        let mut dialogue = ShapeDialogue::new();
        let turn = dialogue
            .ask_next(&facts, &registry, &types, &shapes, &mut acquisition)
            .expect("全局议程应能打开一个缺口");
        let consultation = turn.consultation.unwrap();
        assert_eq!(dialogue.focus(), Some(consultation.gap.entity.as_str()));
        assert_eq!(
            dialogue.original_question(),
            Some(format!("{}是谁？", consultation.gap.entity).as_str())
        );
        assert!(!acquisition.has_pending_goal() || dialogue.has_pending());
        assert!(
            dialogue
                .ask_next(&facts, &registry, &types, &shapes, &mut acquisition)
                .is_none(),
            "已有待答问题时不应覆盖"
        );
    }

    #[test]
    fn conflict_ruling_requires_a_listed_value_and_converges_only_in_session() {
        let registry = registry();
        let types = types();
        let shapes = shapes(&registry, &types);
        let facts = vec![
            entity(
                "刘德华",
                "Person",
                &[("name", "刘德华"), ("职业", "演员"), ("出生地", "香港")],
            ),
            relation("刘德华", "职业", "导演"),
        ];
        let mut acquisition = RelationAcquisition::new();
        let mut dialogue = ShapeDialogue::new();
        let first = dialogue.start_profile(
            "刘德华",
            "刘德华是谁？",
            &facts,
            &registry,
            &types,
            &shapes,
            &mut acquisition,
        );
        assert_eq!(first.answer.status, AnswerStatus::Conflict);

        let error = dialogue
            .resolve_conflict(
                "刘德华",
                "职业",
                "歌手",
                &facts,
                &registry,
                &types,
                &shapes,
                &mut acquisition,
            )
            .expect_err("不能裁定为不在冲突列表里的值");
        assert!(error.contains("只有这些冲突值"), "{}", error);

        let turn = dialogue
            .resolve_conflict(
                "刘德华",
                "职业",
                "演员",
                &facts,
                &registry,
                &types,
                &shapes,
                &mut acquisition,
            )
            .expect("列出的冲突值应可裁定");
        assert_eq!(turn.answer.status, AnswerStatus::Complete);
        assert_eq!(turn.answer.value_of("职业"), vec!["演员"]);
        assert!(turn
            .answer
            .value_of("职业")
            .iter()
            .all(|value| *value != "导演"));
        assert_eq!(turn.answer.rulings.len(), 1);
        assert!(turn.message.contains("本会话裁定"));
        assert!(turn.answer.render_text().contains("知识图谱未被改写"));
        assert!(acquisition.overlay().is_empty());

        let (raw, _) = ShapeAnswer::build(
            "刘德华",
            &facts,
            &acquisition.overlay(),
            &shapes,
            &registry,
            &types,
        );
        assert_eq!(
            raw.status,
            AnswerStatus::Conflict,
            "会话裁定不能改写图谱事实"
        );

        let snapshot = dialogue.snapshot();
        let mut restored = ShapeDialogue::new();
        assert!(restored.restore(&snapshot));
        assert_eq!(restored.rulings().len(), 1);
        let (resolved, _) = ShapeAnswer::build_resolved(
            "刘德华",
            &facts,
            &acquisition.overlay(),
            &shapes,
            &registry,
            &types,
            restored.rulings(),
        );
        assert_eq!(resolved.status, AnswerStatus::Complete);
        assert_eq!(resolved.value_of("职业"), vec!["演员"]);
    }

    #[test]
    fn colloquial_conflict_ruling_resolves_the_focused_slot() {
        let registry = registry();
        let types = types();
        let shapes = shapes(&registry, &types);
        let facts = vec![
            entity(
                "刘德华",
                "Person",
                &[("name", "刘德华"), ("职业", "演员"), ("出生地", "香港")],
            ),
            relation("刘德华", "职业", "导演"),
        ];
        let mut acquisition = RelationAcquisition::new();
        let mut dialogue = ShapeDialogue::new();
        let first = dialogue.start_profile(
            "刘德华",
            "刘德华是谁？",
            &facts,
            &registry,
            &types,
            &shapes,
            &mut acquisition,
        );
        assert_eq!(first.answer.status, AnswerStatus::Conflict);
        assert!(first.message.contains("裁定："));
        assert!(ShapeDialogue::looks_like_conflict_ruling("以演员为准"));
        assert!(ShapeDialogue::looks_like_conflict_ruling(
            "不是导演，是演员"
        ));
        assert!(!ShapeDialogue::looks_like_conflict_ruling("刘德华是谁？"));

        let turn = dialogue
            .resolve_conflict_text(
                "不是导演，是演员",
                &facts,
                &registry,
                &types,
                &shapes,
                &mut acquisition,
            )
            .expect("应命中当前职业冲突")
            .expect("演员是列出的冲突值");
        assert_eq!(turn.answer.status, AnswerStatus::Complete);
        assert_eq!(turn.answer.value_of("职业"), vec!["演员"]);
        assert_eq!(dialogue.focus(), Some("刘德华"));
    }

    #[test]
    fn ruling_can_select_a_session_value_and_suppress_the_graph_value() {
        let registry = registry();
        let types = types();
        let shapes = shapes(&registry, &types);
        let facts = vec![
            entity(
                "刘德华",
                "Person",
                &[("name", "刘德华"), ("职业", "演员"), ("出生地", "香港")],
            ),
            relation("刘德华", "职业", "导演"),
        ];
        let overlay = SessionOverlay {
            rows: vec![SessionOverlayRow {
                subject: "刘德华".to_string(),
                predicate: "职业".to_string(),
                object: "歌手".to_string(),
                claim_id: "c_session".to_string(),
                status: AcquisitionStatus::Admitted,
            }],
        };
        let rulings = vec![ConflictRuling {
            entity: "刘德华".to_string(),
            predicate: "职业".to_string(),
            value: "歌手".to_string(),
            turn: 1,
        }];
        let (answer, report) = ShapeAnswer::build_resolved(
            "刘德华",
            &facts,
            &overlay,
            &shapes,
            &registry,
            &types,
            &rulings,
        );
        assert_eq!(
            answer.status,
            AnswerStatus::Complete,
            "{}",
            answer.render_text()
        );
        assert_eq!(answer.value_of("职业"), vec!["歌手"]);
        assert_eq!(answer.conflicts.len(), 0);
        assert!(report.conforms);
    }

    #[test]
    fn agenda_after_ruling_does_not_stay_blocked_by_the_resolved_conflict() {
        let registry = registry();
        let types = types();
        let shapes = shapes(&registry, &types);
        let facts = vec![
            entity("刘德华", "Person", &[("name", "刘德华"), ("职业", "演员")]),
            relation("刘德华", "职业", "导演"),
        ];
        let mut acquisition = RelationAcquisition::new();
        let mut dialogue = ShapeDialogue::new();
        let first = dialogue.start_profile(
            "刘德华",
            "刘德华是谁？",
            &facts,
            &registry,
            &types,
            &shapes,
            &mut acquisition,
        );
        assert_eq!(first.answer.status, AnswerStatus::Conflict);
        let ruled = dialogue
            .resolve_conflict(
                "刘德华",
                "职业",
                "演员",
                &facts,
                &registry,
                &types,
                &shapes,
                &mut acquisition,
            )
            .unwrap();
        assert_eq!(ruled.answer.status, AnswerStatus::Partial);
        let next = dialogue
            .ask_next(&facts, &registry, &types, &shapes, &mut acquisition)
            .expect("裁定后应能继续追问剩余缺失槽位");
        assert_eq!(
            next.consultation.unwrap().gap.predicate,
            "出生地",
            "已裁定的职业冲突不应继续阻塞议程"
        );
    }

    #[test]
    fn declined_gap_is_not_asked_again() {
        let registry = registry();
        let types = types();
        let shapes = shapes(&registry, &types);
        let facts = vec![entity("王五", "Person", &[("name", "王五")])];
        let mut acquisition = RelationAcquisition::new();
        let mut dialogue = ShapeDialogue::new();
        let first = dialogue.start_profile(
            "王五",
            "王五是谁？",
            &facts,
            &registry,
            &types,
            &shapes,
            &mut acquisition,
        );
        assert_eq!(first.answer.status, AnswerStatus::Partial);
        let gap_id = first.consultation.unwrap().gap.id;
        let declined = dialogue
            .respond(
                "我不知道",
                &facts,
                &registry,
                &types,
                &shapes,
                &mut acquisition,
            )
            .unwrap();
        assert!(declined.consultation.is_none());
        assert!(declined.message.contains("不会重复追问"));
        assert_eq!(
            dialogue.attempts.get(&gap_id).copied(),
            Some(MAX_ASK_ATTEMPTS)
        );
        assert!(
            dialogue
                .ask_next(&facts, &registry, &types, &shapes, &mut acquisition)
                .is_none(),
            "用户已明确不知道，不应再次把同一缺口抛回去"
        );
    }

    #[test]
    fn parse_ruling_accepts_explicit_command_forms() {
        assert_eq!(
            parse_ruling("刘德华 的 职业 = 演员"),
            Some(("刘德华".to_string(), "职业".to_string(), "演员".to_string()))
        );
        assert_eq!(
            parse_ruling("刘德华的出生地是香港。"),
            Some((
                "刘德华".to_string(),
                "出生地".to_string(),
                "香港".to_string()
            ))
        );
        assert_eq!(
            parse_ruling("刘德华|职业|导演"),
            Some(("刘德华".to_string(), "职业".to_string(), "导演".to_string()))
        );
        assert!(parse_ruling("刘德华").is_none());
        assert!(parse_ruling("刘德华 的 职业 = ").is_none());
    }

    #[test]
    fn persisted_snapshot_roundtrips_on_disk() {
        let registry = registry();
        let types = types();
        let shapes = shapes(&registry, &types);
        let facts = facts();
        let mut acquisition = RelationAcquisition::new();
        let mut dialogue = ShapeDialogue::new();
        let _ = dialogue.start_profile(
            "王五",
            "王五是谁？",
            &facts,
            &registry,
            &types,
            &shapes,
            &mut acquisition,
        );
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sfd.json");
        dialogue.save_to_path(path.to_str().unwrap()).unwrap();
        let restored = ShapeDialogue::load_from_path(path.to_str().unwrap()).unwrap();
        assert_eq!(restored.digest(), dialogue.digest());
        assert_eq!(restored.pending(), dialogue.pending());
        assert_eq!(restored.original_question(), Some("王五是谁？"));
    }
}

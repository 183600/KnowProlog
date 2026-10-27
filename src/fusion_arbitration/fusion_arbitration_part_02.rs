#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge_fusion::{Adjudication, NamespaceKind, NamespaceRecord, TrustLevel};
    use crate::provenance::{SourceId, SourceKind};
    use crate::slot_ledger::{AssertionStrength, SlotAssertion};

    fn record(namespace: &str, trust: TrustLevel) -> NamespaceRecord {
        NamespaceRecord::new(namespace, namespace, NamespaceKind::Import, trust)
    }

    fn assertion(
        namespace: &str,
        subject: &str,
        predicate: &str,
        object: &str,
    ) -> ArbitrationAssertion {
        ArbitrationAssertion::new(namespace, subject, predicate, object, ObjectKind::Literal)
    }

    fn input(namespaces: &[NamespaceRecord], assertions: &[ArbitrationAssertion]) -> ArbitrationInput {
        ArbitrationInput {
            catalog: SourceCatalog::from_records(namespaces),
            assertions: assertions.to_vec(),
            ..ArbitrationInput::default()
        }
    }

    fn policy() -> SlotPolicy {
        SlotPolicy::with_defaults()
    }

    #[test]
    fn single_independent_source_is_single_source() {
        let report = arbitrate(
            &input(
                &[record("a", TrustLevel::Imported)],
                &[assertion("a", "小明", "年龄", "18")],
            ),
            &policy(),
        );
        let slot = report.slot("小明", "年龄").expect("应有槽");
        assert_eq!(slot.mode, DecisionMode::SingleSource);
        assert_eq!(slot.winner_objects(), vec!["18"]);
        assert_eq!(slot.winners[0].independent_sources, 1);
        assert!(slot.wording_zh().contains("一个独立来源"));
    }

    #[test]
    fn two_independent_namespaces_are_consensus() {
        let report = arbitrate(
            &input(
                &[
                    record("a", TrustLevel::Imported),
                    record("b", TrustLevel::External),
                ],
                &[
                    assertion("a", "小明", "年龄", "18"),
                    assertion("b", "小明", "年龄", "18"),
                ],
            ),
            &policy(),
        );
        let slot = report.slot("小明", "年龄").expect("应有槽");
        assert_eq!(slot.mode, DecisionMode::Consensus);
        assert_eq!(slot.winners[0].independent_sources, 2);
        assert!(slot.dissent.is_empty());
    }

    #[test]
    fn same_file_imported_twice_is_not_consensus() {
        let origin = SourceId::new(SourceKind::Import, "people.csv", "小明,18");
        let catalog = SourceCatalog::from_records(&[
            record("batch-a", TrustLevel::Imported),
            record("batch-b", TrustLevel::Imported),
        ])
        .with_profile(
            SourceProfile::from_record(&record("batch-a", TrustLevel::Imported))
                .with_origin(origin.clone()),
        )
        .with_profile(
            SourceProfile::from_record(&record("batch-b", TrustLevel::Imported))
                .with_origin(origin),
        );
        let report = arbitrate(
            &ArbitrationInput {
                catalog,
                assertions: vec![
                    assertion("batch-a", "小明", "年龄", "18"),
                    assertion("batch-b", "小明", "年龄", "18"),
                ],
                ..ArbitrationInput::default()
            },
            &policy(),
        );
        let slot = report.slot("小明", "年龄").expect("应有槽");
        assert_eq!(slot.mode, DecisionMode::SingleSource);
        assert_eq!(slot.winners[0].independent_sources, 1);
    }

    #[test]
    fn mirrored_namespaces_do_not_create_consensus() {
        let report = arbitrate(
            &input(
                &[
                    record("mirror-a", TrustLevel::Imported),
                    record("mirror-b", TrustLevel::Imported),
                ],
                &[
                    assertion("mirror-a", "小明", "年龄", "18"),
                    assertion("mirror-b", "小明", "年龄", "18"),
                ],
            )
            .with_mirror("mirror-b", "mirror-a"),
            &policy(),
        );
        let slot = report.slot("小明", "年龄").expect("应有槽");
        assert_eq!(slot.mode, DecisionMode::SingleSource);
        assert_eq!(slot.winners[0].independent_sources, 1);
    }

    #[test]
    fn multiple_assertions_in_one_namespace_count_once() {
        let report = arbitrate(
            &input(
                &[record("a", TrustLevel::Imported)],
                &[
                    assertion("a", "小明", "年龄", "18"),
                    assertion("a", "小明", "年龄", "18"),
                ],
            ),
            &policy(),
        );
        let slot = report.slot("小明", "年龄").expect("应有槽");
        assert_eq!(slot.mode, DecisionMode::SingleSource);
        assert_eq!(slot.winners[0].independent_sources, 1);
    }

    #[test]
    fn materialized_default_graph_is_not_an_independent_source() {
        let catalog = SourceCatalog::from_records(&[record("kb", TrustLevel::External)]);
        let report = arbitrate(
            &ArbitrationInput {
                catalog,
                applied: vec![AppliedWinner {
                    id: "w1".to_string(),
                    namespace: "kb".to_string(),
                    subject: "小明".to_string(),
                    predicate: "年龄".to_string(),
                    object: "18".to_string(),
                    object_kind: ObjectKind::Literal,
                }],
                ..ArbitrationInput::default()
            },
            &policy(),
        );
        let slot = report.slot("小明", "年龄").expect("应有槽");
        assert_eq!(slot.mode, DecisionMode::KnowledgeBaseOnly);
        assert_eq!(slot.winners[0].independent_sources, 0);
        assert_eq!(slot.winners[0].trust, Some(TrustLevel::External));
        assert!(slot.winners[0].materialized);
        assert!(slot.wording_zh().contains("没有找到独立来源"));
    }

    #[test]
    fn applied_winner_inherits_namespace_trust() {
        let catalog = SourceCatalog::from_records(&[record("kb", TrustLevel::Speculative)]);
        let report = arbitrate(
            &ArbitrationInput {
                catalog,
                applied: vec![AppliedWinner {
                    id: "w1".to_string(),
                    namespace: "kb".to_string(),
                    subject: "小明".to_string(),
                    predicate: "年龄".to_string(),
                    object: "18".to_string(),
                    object_kind: ObjectKind::Literal,
                }],
                ..ArbitrationInput::default()
            },
            &policy(),
        );
        let slot = report.slot("小明", "年龄").expect("应有槽");
        assert_eq!(slot.winners[0].trust, Some(TrustLevel::Speculative));
        assert_ne!(slot.winners[0].trust, Some(TrustLevel::Curated));
    }

    #[test]
    fn higher_trust_beats_more_low_trust_sources() {
        let report = arbitrate(
            &input(
                &[
                    record("kb", TrustLevel::Curated),
                    record("x", TrustLevel::External),
                    record("y", TrustLevel::External),
                ],
                &[
                    assertion("kb", "小明", "年龄", "18"),
                    assertion("x", "小明", "年龄", "19"),
                    assertion("y", "小明", "年龄", "19"),
                ],
            ),
            &policy(),
        );
        let slot = report.slot("小明", "年龄").expect("应有槽");
        assert_eq!(slot.mode, DecisionMode::TrustedWinner);
        assert_eq!(slot.winner_objects(), vec!["18"]);
        assert_eq!(slot.dissent_objects(), vec!["19"]);
        assert_eq!(slot.dissent[0].independent_sources, 2);
    }

    #[test]
    fn equal_strength_conflict_is_contested() {
        let report = arbitrate(
            &input(
                &[
                    record("a", TrustLevel::Imported),
                    record("b", TrustLevel::Imported),
                ],
                &[
                    assertion("a", "小明", "年龄", "18"),
                    assertion("b", "小明", "年龄", "19"),
                ],
            ),
            &policy(),
        );
        let slot = report.slot("小明", "年龄").expect("应有槽");
        assert_eq!(slot.mode, DecisionMode::Contested);
        assert!(slot.winners.is_empty());
        assert_eq!(slot.contested.len(), 2);
        assert!(slot.wording_zh().contains("暂不合并"));
    }

    #[test]
    fn adjudication_overrides_trust_ordering() {
        let mut input = input(
            &[
                record("kb", TrustLevel::Curated),
                record("friend", TrustLevel::External),
            ],
            &[
                assertion("kb", "小明", "年龄", "18"),
                assertion("friend", "小明", "年龄", "19"),
            ],
        );
        input.adjudications.push(Adjudication::new(
            "年龄", "friend", "用户确认", "user",
        ));
        let report = arbitrate(&input, &policy());
        let slot = report.slot("小明", "年龄").expect("应有槽");
        assert_eq!(slot.mode, DecisionMode::Adjudicated);
        assert_eq!(slot.winner_objects(), vec!["19"]);
        assert!(slot.adjudicated_by.is_some());
        assert!(slot.wording_zh().contains("显式裁决"));
    }

    #[test]
    fn session_correction_overrides_graph_and_keeps_old_value_as_dissent() {
        let mut slots = SlotLedger::new();
        slots
            .apply(
                SlotAssertion::new(
                    "小明",
                    "年龄",
                    "18",
                    AssertionStrength::Direct,
                    "图谱",
                ),
                &["18".to_string()],
            )
            .unwrap();
        let transition = slots
            .apply(
                SlotAssertion::new(
                    "小明",
                    "年龄",
                    "19",
                    AssertionStrength::Correction,
                    "用户",
                ),
                &["18".to_string()],
            )
            .unwrap();
        assert!(matches!(
            transition.delta,
            crate::slot_ledger::SlotDelta::Corrected { .. }
        ));
        let mut input = input(
            &[record("kb", TrustLevel::Curated)],
            &[assertion("kb", "小明", "年龄", "18")],
        )
        .with_session_ledger(&slots);
        input.applied.push(AppliedWinner {
            id: "w1".to_string(),
            namespace: "kb".to_string(),
            subject: "小明".to_string(),
            predicate: "年龄".to_string(),
            object: "18".to_string(),
            object_kind: ObjectKind::Literal,
        });
        let report = arbitrate(&input, &policy());
        let slot = report.slot("小明", "年龄").expect("应有槽");
        assert_eq!(slot.mode, DecisionMode::SessionCorrection);
        assert_eq!(slot.winner_objects(), vec!["19"]);
        assert_eq!(slot.dissent_objects(), vec!["18"]);
        assert!(slot.wording_zh().contains("按本会话的更正"));
        assert!(slot.wording_zh().contains("压制但未否定"));
    }

    #[test]
    fn direct_session_assertion_is_not_a_correction() {
        let mut slots = SlotLedger::new();
        slots
            .apply(
                SlotAssertion::new(
                    "小明",
                    "年龄",
                    "19",
                    AssertionStrength::Direct,
                    "用户",
                ),
                &["18".to_string()],
            )
            .unwrap();
        assert!(SessionOverlay::from_ledger(&slots).is_empty());
    }

    #[test]
    fn hearsay_is_not_a_session_correction() {
        let mut slots = SlotLedger::new();
        slots
            .apply(
                SlotAssertion::new(
                    "小明",
                    "年龄",
                    "19",
                    AssertionStrength::Hearsay,
                    "据说",
                ),
                &["18".to_string()],
            )
            .unwrap();
        assert!(SessionOverlay::from_ledger(&slots).is_empty());
    }

    #[test]
    fn dissent_blocks_false_consensus_from_dissenting_certificate() {
        let mut input = input(
            &[record("a", TrustLevel::Imported)],
            &[assertion("a", "小明", "年龄", "18")],
        );
        input = input.with_dissent(DissentEvidence::new(
            "小明",
            "年龄",
            "19",
            "冲突证书 cert-1",
        ));
        let report = arbitrate(&input, &policy());
        let slot = report.slot("小明", "年龄").expect("应有槽");
        assert_ne!(slot.mode, DecisionMode::Consensus);
        assert_eq!(slot.mode, DecisionMode::TrustedWinner);
        assert_eq!(slot.dissent_objects(), vec!["19"]);
    }

    #[test]
    fn provenance_ledger_bridge_replaces_namespace_origins() {
        let mut ledger = FusionLedger::new();
        ledger.upsert_namespace(record("batch-a", TrustLevel::Imported));
        ledger.upsert_namespace(record("batch-b", TrustLevel::Imported));
        ledger.record_assertion(ScopedAssertion::new(
            "batch-a",
            "小明",
            "年龄",
            "18",
            ObjectKind::Literal,
            "a.csv",
        ));
        ledger.record_assertion(ScopedAssertion::new(
            "batch-b",
            "小明",
            "年龄",
            "18",
            ObjectKind::Literal,
            "b.csv",
        ));

        let without = arbitrate(&ArbitrationInput::from_ledger(&ledger), &policy());
        assert_eq!(
            without.slot("小明", "年龄").expect("应有槽").mode,
            DecisionMode::Consensus
        );

        let origin = SourceId::new(SourceKind::Import, "people.csv", "小明,18");
        let fact = FactRef::new("小明", "年龄", "18");
        let mut provenance = ProvenanceLedger::new();
        provenance.record(fact.clone(), origin.clone());
        provenance.record(fact, origin.clone());
        let report = arbitrate(
            &ArbitrationInput::from_ledger(&ledger).with_provenance(&provenance),
            &policy(),
        );
        let slot = report.slot("小明", "年龄").expect("应有槽");
        assert_eq!(slot.mode, DecisionMode::SingleSource);
        assert_eq!(slot.winners[0].independent_sources, 1);
        assert!(slot.winners[0]
            .sources
            .iter()
            .any(|source| source == &origin.render()));
    }

    #[test]
    fn provenance_origins_collapse_duplicate_imports() {
        let origin = SourceId::new(SourceKind::Import, "people.csv", "小明,18");
        let input = input(
            &[
                record("batch-a", TrustLevel::Imported),
                record("batch-b", TrustLevel::Imported),
            ],
            &[
                assertion("batch-a", "小明", "年龄", "18")
                    .with_origin(origin.clone()),
                assertion("batch-b", "小明", "年龄", "18").with_origin(origin),
            ],
        );
        let report = arbitrate(&input, &policy());
        let slot = report.slot("小明", "年龄").expect("应有槽");
        assert_eq!(slot.mode, DecisionMode::SingleSource);
        assert_eq!(slot.winners[0].independent_sources, 1);
    }

    #[test]
    fn rank_orders_session_correction_and_certified_winner_first() {
        let mut slots = SlotLedger::new();
        slots
            .apply(
                SlotAssertion::new("小明", "年龄", "19", AssertionStrength::Correction, "用户"),
                &["18".to_string()],
            )
            .unwrap();
        let input = input(
            &[record("kb", TrustLevel::Curated)],
            &[assertion("kb", "小明", "年龄", "18")],
        )
        .with_session_ledger(&slots);
        let report = arbitrate(&input, &policy());
        assert_eq!(
            report.rank_for("小明", "年龄", "19").band,
            RankBand::SessionCorrection
        );
        assert_eq!(
            report.rank_for("小明", "年龄", "18").band,
            RankBand::Dissent
        );
        assert!(
            report.compare(
                ("小明", "年龄", "19"),
                ("小明", "年龄", "18")
            ) == Ordering::Less
        );
        assert_eq!(
            report.rank_for("小明", "身高", "175").band,
            RankBand::Unsupported
        );
    }

    #[test]
    fn contested_values_share_the_same_rank() {
        let report = arbitrate(
            &input(
                &[
                    record("a", TrustLevel::Imported),
                    record("b", TrustLevel::Imported),
                ],
                &[
                    assertion("a", "小明", "年龄", "18"),
                    assertion("b", "小明", "年龄", "19"),
                ],
            ),
            &policy(),
        );
        assert_eq!(
            report.rank_for("小明", "年龄", "18"),
            report.rank_for("小明", "年龄", "19")
        );
        assert_eq!(
            report.rank_for("小明", "年龄", "18").band,
            RankBand::Contested
        );
    }

    #[test]
    fn materialized_only_value_is_ranked_as_knowledge_base_in_mixed_slot() {
        let catalog = SourceCatalog::from_records(&[
            record("a", TrustLevel::Imported),
            record("kb", TrustLevel::Curated),
        ]);
        let report = arbitrate(
            &ArbitrationInput {
                catalog,
                assertions: vec![assertion("a", "苹果", "颜色", "红色")],
                applied: vec![AppliedWinner {
                    id: "w1".to_string(),
                    namespace: "kb".to_string(),
                    subject: "苹果".to_string(),
                    predicate: "颜色".to_string(),
                    object: "绿色".to_string(),
                    object_kind: ObjectKind::Literal,
                }],
                ..ArbitrationInput::default()
            },
            &policy(),
        );
        let slot = report.slot("苹果", "颜色").expect("应有槽");
        assert_eq!(slot.winners.len(), 2);
        assert_eq!(
            report.rank_for("苹果", "颜色", "红色").band,
            RankBand::Winner
        );
        assert_eq!(
            report.rank_for("苹果", "颜色", "绿色").band,
            RankBand::KnowledgeBaseOnly
        );
    }

    #[test]
    fn non_functional_slot_keeps_compatible_values() {
        let report = arbitrate(
            &input(
                &[record("a", TrustLevel::Imported)],
                &[
                    assertion("a", "苹果", "颜色", "红色"),
                    assertion("a", "苹果", "颜色", "绿色"),
                ],
            ),
            &policy(),
        );
        let slot = report.slot("苹果", "颜色").expect("应有槽");
        assert_eq!(slot.mode, DecisionMode::SingleSource);
        assert_eq!(slot.winners.len(), 2);
        assert!(slot.contested.is_empty());
    }

    #[test]
    fn report_is_deterministic_regardless_of_input_order() {
        let namespaces = [
            record("b", TrustLevel::External),
            record("a", TrustLevel::Imported),
        ];
        let forward = arbitrate(
            &input(
                &namespaces,
                &[
                    assertion("a", "小明", "年龄", "18"),
                    assertion("b", "小明", "年龄", "19"),
                ],
            ),
            &policy(),
        );
        let reverse = arbitrate(
            &input(
                &[namespaces[1].clone(), namespaces[0].clone()],
                &[
                    assertion("b", "小明", "年龄", "19"),
                    assertion("a", "小明", "年龄", "18"),
                ],
            ),
            &policy(),
        );
        assert_eq!(forward, reverse);
    }

    #[test]
    fn unknown_or_inactive_namespaces_do_not_vote() {
        let mut retired = record("old", TrustLevel::Verified);
        retired.status = NamespaceStatus::Retired;
        let report = arbitrate(
            &input(
                &[record("live", TrustLevel::Imported), retired],
                &[
                    assertion("live", "小明", "年龄", "18"),
                    assertion("old", "小明", "年龄", "19"),
                    assertion("ghost", "小明", "年龄", "20"),
                ],
            ),
            &policy(),
        );
        let slot = report.slot("小明", "年龄").expect("应有槽");
        assert_eq!(slot.mode, DecisionMode::SingleSource);
        assert_eq!(slot.winner_objects(), vec!["18"]);
        assert_eq!(slot.arguments.len(), 1);
    }

    #[test]
    fn source_catalog_component_is_stable_and_idempotent() {
        let origin = SourceId::new(SourceKind::Import, "file.csv", "payload");
        let catalog = SourceCatalog::from_records(&[record("a", TrustLevel::Imported)])
            .with_profile(
                SourceProfile::from_record(&record("a", TrustLevel::Imported))
                    .with_origin(origin.clone()),
            )
            .with_mirror("b", "a");
        let first = catalog.component_map();
        let second = catalog.component_map();
        assert_eq!(first, second);
        assert_eq!(first.get("a"), first.get("b"));
    }
}

/// 追加更正事件：旧值被判错，新值生效（效果可枚举、可反事实重放）。
pub fn append_correction(
    chronicle: &mut Chronicle,
    text: &str,
    corrections: &[TheoryCorrection],
    revision_after: u64,
) -> io::Result<u64> {
    let mut effects = BTreeSet::new();
    for correction in corrections {
        if !correction.from.is_empty() && correction.from != correction.to {
            effects.insert(TheoryEffect::remove(
                &correction.entity,
                &correction.predicate,
                &correction.from,
            ));
        }
        if !correction.to.is_empty() {
            effects.insert(TheoryEffect::add(
                &correction.entity,
                &correction.predicate,
                &correction.to,
            ));
        }
    }
    let effects: Vec<TheoryEffect> = effects.into_iter().collect();
    let effect_count = effects.len();
    let digest = effects_digest(&effects);
    let mut payload = event_payload(
        text,
        "用户更正",
        revision_after,
        &effects,
        true,
        Some(digest),
        None,
        effect_count,
    );
    payload["corrections"] = Value::Array(
        corrections
            .iter()
            .map(|correction| {
                json!({
                    "entity": correction.entity,
                    "predicate": correction.predicate,
                    "from": correction.from,
                    "to": correction.to,
                })
            })
            .collect(),
    );
    chronicle.append("correct", payload)
}
/// 追加上次导入的撤销事件（与导入共享回执摘要，可精确移除来源）。
pub fn append_retract(
    chronicle: &mut Chronicle,
    receipt: &ImportReceipt,
    source: &str,
    revision_after: u64,
) -> io::Result<u64> {
    let all_effects = receipt_effects(receipt, EffectOp::Remove);
    let effect_count = all_effects.len();
    let effects: Vec<TheoryEffect> = all_effects
        .iter()
        .take(MAX_EFFECTS_PER_EVENT)
        .cloned()
        .collect();
    let digest = receipt_digest(receipt);
    chronicle.append(
        "retract",
        event_payload(
            &format!(
                "撤销导入：{}",
                receipt.summary().lines().next().unwrap_or("")
            ),
            source,
            revision_after,
            &effects,
            true,
            Some(digest.clone()),
            Some(digest),
            effect_count,
        ),
    )
}
/// 追加批量导入事件（JSON-LD/RDF：新增条数已知，逐条效果未入编年史）。
pub fn append_bulk_import(
    chronicle: &mut Chronicle,
    source: &str,
    target: &str,
    added: usize,
    revision_after: u64,
) -> io::Result<u64> {
    chronicle.append(
        "import",
        event_payload(
            &format!("{}：{} 新增 {} 条", source, target, added),
            source,
            revision_after,
            &[],
            false,
            None,
            None,
            added,
        ),
    )
}
/// 追加用户断言事件。
pub fn append_assert(
    chronicle: &mut Chronicle,
    text: &str,
    effects: &[TheoryEffect],
    revision_after: u64,
) -> io::Result<u64> {
    let effect_count = effects.len();
    let effects: Vec<TheoryEffect> = effects
        .iter()
        .take(MAX_EFFECTS_PER_EVENT)
        .cloned()
        .collect();
    let digest = effects_digest(&effects);
    chronicle.append(
        "assert",
        event_payload(
            text,
            "用户断言",
            revision_after,
            &effects,
            true,
            Some(digest),
            None,
            effect_count,
        ),
    )
}
/// 追加用户撤回断言事件。
pub fn append_forget(
    chronicle: &mut Chronicle,
    text: &str,
    effects: &[TheoryEffect],
    revision_after: u64,
) -> io::Result<u64> {
    let effect_count = effects.len();
    let effects: Vec<TheoryEffect> = effects
        .iter()
        .take(MAX_EFFECTS_PER_EVENT)
        .cloned()
        .collect();
    let target_digest = effects_digest(&effects);
    chronicle.append(
        "forget",
        event_payload(
            text,
            "用户撤回断言",
            revision_after,
            &effects,
            true,
            Some(target_digest.clone()),
            Some(target_digest),
            effect_count,
        ),
    )
}
/// 追加规则教学事件（效果不可完全枚举）。
pub fn append_rule(
    chronicle: &mut Chronicle,
    text: &str,
    learned: bool,
    revision_after: u64,
) -> io::Result<u64> {
    chronicle.append(
        "rule",
        event_payload(
            text,
            if learned {
                "规则教学（已学习）"
            } else {
                "规则教学（未解析）"
            },
            revision_after,
            &[],
            false,
            None,
            None,
            0,
        ),
    )
}
/// 追加外链事件（缓存/刷新/遗忘/别名；效果不由本地编年史枚举）。
pub fn append_external(
    chronicle: &mut Chronicle,
    entity: &str,
    action: &str,
    source_uri: &str,
    revision_after: u64,
) -> io::Result<u64> {
    chronicle.append(
        "external",
        event_payload(
            &format!("外链 {}：{}", action, entity),
            source_uri,
            revision_after,
            &[],
            false,
            None,
            None,
            0,
        ),
    )
}
/// 追加修复事件（截尾重写等）。
pub fn append_repair(chronicle: &mut Chronicle, detail: &str) -> io::Result<u64> {
    chronicle.append(
        "repair",
        event_payload(detail, "编年史修复", 0, &[], false, None, None, 0),
    )
}
fn truncate(text: &str, limit: usize) -> String {
    let mut out: String = text.chars().take(limit).collect();
    if text.chars().count() > limit {
        out.push('…');
    }
    out
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::chronicle::FsyncPolicy;
    use crate::predicate_registry::{ImportAction, ImportReceipt};
    use tempfile::TempDir;

    fn chronicle(dir: &TempDir) -> (Chronicle, String) {
        let path = dir.path().join("theory.chronicle.jsonl");
        let path_text = path.to_string_lossy().to_string();
        let (chronicle, audit) = Chronicle::open(&path_text, FsyncPolicy::None).unwrap();
        assert!(audit.is_clean());
        (chronicle, path_text)
    }

    fn receipt(subject: &str, predicate: &str, value: &str) -> ImportReceipt {
        ImportReceipt {
            added: 1,
            actions: vec![ImportAction::Literal {
                subject: format!("http://example.org/kg/entity/{}", subject),
                predicate: format!("http://example.org/kg/{}", predicate),
                value: value.to_string(),
            }],
            ..Default::default()
        }
    }

    fn ask(chronicle: &mut Chronicle, question: &str) {
        chronicle
            .append_turn(question, "query", Some("姚明"), 0)
            .unwrap();
    }

    fn answer(chronicle: &mut Chronicle, text: &str) -> u64 {
        let seq = chronicle.append_reply(text, &[], &[]).unwrap();
        seq
    }

    #[test]
    fn compiles_situations_answers_and_change_rules() {
        let dir = TempDir::new().unwrap();
        let (mut chronicle, _) = chronicle(&dir);
        ask(&mut chronicle, "姚明的身高是多少？");
        answer(&mut chronicle, "姚明身高 226 厘米。");
        chronicle.set_kb_revision(1);
        ask(&mut chronicle, "姚明的身高是多少呢");
        answer(&mut chronicle, "姚明身高 229 厘米。");

        let theory = ConversationTheory::build(&chronicle);
        assert_eq!(theory.answers().len(), 2);
        assert_eq!(theory.changes().len(), 1);
        assert!(theory.evaluation_error().is_none());
        assert_eq!(
            theory
                .program()
                .query("answered_at(S, Q, A, P)")
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            theory
                .program()
                .query("answer_change(Q, A1, A2)")
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            theory
                .program()
                .query("change_at(Q, S1, S2)")
                .unwrap()
                .len(),
            1
        );
        assert_eq!(theory.program().query("before(s0, s4)").unwrap().len(), 1);
        assert!(theory.render_report().contains("改口 1 处"));
    }

    #[test]
    fn import_effects_are_folded_and_attributed() {
        let dir = TempDir::new().unwrap();
        let (mut chronicle, _) = chronicle(&dir);
        ask(&mut chronicle, "姚明的身高是多少？");
        answer(&mut chronicle, "姚明身高 226 厘米。");
        let import_receipt = receipt("姚明", "height", "229");
        append_import(&mut chronicle, &import_receipt, "文本录入", 1).unwrap();
        chronicle.set_kb_revision(1);
        ask(&mut chronicle, "姚明的身高是多少？");
        answer(&mut chronicle, "姚明身高 229 厘米。");

        let theory = ConversationTheory::build(&chronicle);
        assert_eq!(theory.changes().len(), 1);
        let report = theory.causes("姚明的身高是多少");
        assert_eq!(
            report.change.as_ref().unwrap().previous.answer,
            "姚明身高 226 厘米。"
        );
        assert_eq!(report.candidates.len(), 1, "{:?}", report);
        assert!(report.candidates[0]
            .reasons
            .iter()
            .any(|reason| reason.contains("229")));
        assert!(theory
            .query("belief_at(S, E, P, V, Src)")
            .unwrap()
            .iter()
            .any(|row| row.get("V").map(String::as_str) == Some("229")));
        assert_eq!(
            theory
                .program()
                .query("candidate_cause(E, Q)")
                .unwrap()
                .len(),
            1
        );
        assert!(!theory.render_causes("姚明的身高是多少").is_empty());
    }

    #[test]
    fn non_enumerable_events_are_ignored_for_attribution() {
        let dir = TempDir::new().unwrap();
        let (mut chronicle, _) = chronicle(&dir);
        ask(&mut chronicle, "姚明的身高是多少？");
        answer(&mut chronicle, "姚明身高 226 厘米。");
        append_bulk_import(&mut chronicle, "JSON-LD", "test.jsonld", 12, 1).unwrap();
        chronicle.set_kb_revision(1);
        ask(&mut chronicle, "姚明的身高是多少？");
        answer(&mut chronicle, "姚明身高 229 厘米。");

        let theory = ConversationTheory::build(&chronicle);
        let report = theory.causes("姚明的身高是多少");
        assert!(report.candidates.is_empty());
        assert_eq!(report.non_enumerable_events, 1);
        assert!(report.note.contains("不可枚举"));
    }

    #[test]
    fn counterfactual_flips_when_only_source_is_removed() {
        let dir = TempDir::new().unwrap();
        let (mut chronicle, _) = chronicle(&dir);
        ask(&mut chronicle, "姚明的身高是多少？");
        answer(&mut chronicle, "姚明身高 226 厘米。");
        let import_receipt = receipt("姚明", "height", "229");
        append_import(&mut chronicle, &import_receipt, "文本录入", 1).unwrap();
        chronicle.set_kb_revision(1);
        ask(&mut chronicle, "姚明的身高是多少？");
        answer(&mut chronicle, "姚明身高 229 厘米。");

        let theory = ConversationTheory::build(&chronicle);
        let report = theory.counterfactual_undo_latest("姚明的身高是多少");
        assert_eq!(report.verdict, CounterfactualVerdict::Flips, "{:?}", report);
        assert!(!report.affected.is_empty());
        assert!(theory
            .render_counterfactual("姚明的身高是多少")
            .contains("会翻转"));
    }

    #[test]
    fn counterfactual_keeps_alternative_support() {
        let dir = TempDir::new().unwrap();
        let (mut chronicle, _) = chronicle(&dir);
        let first = receipt("姚明", "height", "229");
        append_import(&mut chronicle, &first, "导入甲", 1).unwrap();
        let second = receipt("姚明", "height", "229");
        append_import(&mut chronicle, &second, "导入乙", 2).unwrap();
        chronicle.set_kb_revision(2);
        ask(&mut chronicle, "姚明的身高是多少？");
        answer(&mut chronicle, "姚明身高 229 厘米。");

        let theory = ConversationTheory::build(&chronicle);
        let report = theory.counterfactual_undo_latest("姚明的身高是多少");
        assert_eq!(
            report.verdict,
            CounterfactualVerdict::Unchanged,
            "{:?}",
            report
        );
        assert!(report.remaining_support >= 1);
    }

    #[test]
    fn counterfactual_abstains_for_non_enumerable_event() {
        let dir = TempDir::new().unwrap();
        let (mut chronicle, _) = chronicle(&dir);
        append_rule(&mut chronicle, "如果X的师傅是Y，那么X的老师是Y", true, 0).unwrap();
        ask(&mut chronicle, "姚明的身高是多少？");
        answer(&mut chronicle, "姚明身高 229 厘米。");

        let theory = ConversationTheory::build(&chronicle);
        let report = theory.counterfactual_undo_latest("姚明的身高是多少");
        assert_eq!(
            report.verdict,
            CounterfactualVerdict::Unknown,
            "{:?}",
            report
        );
        assert!(report
            .reasons
            .iter()
            .any(|reason| reason.contains("不可完全枚举")));
    }

    #[test]
    fn retract_removes_only_matching_import_source() {
        let dir = TempDir::new().unwrap();
        let (mut chronicle, _) = chronicle(&dir);
        let shared = receipt("姚明", "height", "229");
        append_import(&mut chronicle, &shared, "导入甲", 1).unwrap();
        append_import(&mut chronicle, &shared, "导入乙", 2).unwrap();
        append_retract(&mut chronicle, &shared, "撤销导入", 3).unwrap();

        let theory = ConversationTheory::build(&chronicle);
        let beliefs = theory.beliefs_at(chronicle.last_seq());
        assert_eq!(beliefs.len(), 1, "{:?}", beliefs);
        assert_eq!(beliefs[0].value, "229");
    }

    #[test]
    fn forget_removes_matching_assertion() {
        let dir = TempDir::new().unwrap();
        let (mut chronicle, _) = chronicle(&dir);
        let effects = effects_from_text("姚明的身高是229", EffectOp::Add);
        assert!(!effects.is_empty());
        append_assert(&mut chronicle, "姚明的身高是229", &effects, 0).unwrap();
        append_forget(
            &mut chronicle,
            "姚明的身高是229",
            &effects_from_text("姚明的身高是229", EffectOp::Remove),
            0,
        )
        .unwrap();
        let theory = ConversationTheory::build(&chronicle);
        assert!(theory.beliefs_at(chronicle.last_seq()).is_empty());
    }

    #[test]
    fn local_names_are_decoded_from_entity_iris() {
        assert_eq!(
            local_name("http://example.org/kg/entity/%E5%A7%9A%20%E6%98%8E"),
            "姚 明"
        );
        assert_eq!(local_name("http://example.org/kg/height"), "height");
    }

    #[test]
    fn effects_from_text_extracts_ground_facts() {
        let effects = effects_from_text("姚明的身高是226", EffectOp::Add);
        assert!(!effects.is_empty(), "{:?}", effects);
        assert!(effects.iter().any(|effect| effect.value == "226"));
        let forget = effects_from_text("姚明的身高是226", EffectOp::Remove);
        assert_eq!(effects.len(), forget.len());
        assert_eq!(forget[0].op, EffectOp::Remove);
    }

    #[test]
    fn unknown_kind_is_preserved_as_other() {
        let dir = TempDir::new().unwrap();
        let (mut chronicle, _) = chronicle(&dir);
        chronicle
            .append("note", json!({"text": "保持未知事件"}))
            .unwrap();
        let theory = ConversationTheory::build(&chronicle);
        assert_eq!(theory.events().len(), 1);
        assert_eq!(theory.events()[0].kind, EventKind::Other);
        assert!(theory.render_report().contains("other 1"));
    }
}

#[test]
fn batch_clean_claims_commit_together_with_one_receipt() {
    let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
    let message = reasoner.assert_nl_fact("张三的师傅是李四；李四的师傅是王五");
    assert!(message.contains("批次记住"), "{}", message);
    assert_eq!(reasoner.session_facts().len(), 2, "{}", message);
    assert!(reasoner.pending_batch_turn().is_none());
    let report = reasoner.session_revision_report();
    assert!(report.contains("批量记住"), "{}", report);
    assert_eq!(report.matches("session:").count(), 1, "{}", report);
}

#[test]
fn batch_conflict_asks_one_question_and_progressive_keeps_safe_claims() {
    let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
    let message = reasoner.assert_nl_fact(
        "张三的年龄是30；张三的年龄是31；李四的师傅是王五",
    );
    assert!(message.contains("待确认 1 处"), "{}", message);
    assert!(message.contains("张三 的 年龄"), "{}", message);
    assert_eq!(reasoner.session_facts().len(), 1, "{}", message);
    assert_eq!(
        reasoner.pending_batch_turn().map(|turn| turn.questions.len()),
        Some(1)
    );

    let resolved = reasoner.resolve_batch_answer("1A");
    assert!(resolved.contains("本批冲突已全部裁决"), "{}", resolved);
    assert!(reasoner.pending_batch_turn().is_none());
    assert!(reasoner.session_facts().iter().any(|atom| {
        atom.predicate.as_ref() == "年龄"
            && atom.terms.len() == 2
            && atom.terms[1].name() == "30"
    }));
}

#[test]
fn batch_answer_can_replace_graph_base_value() {
    let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
    reasoner.observe_facts(&[PrologFact::Relation {
        from: "张三".to_string(),
        rel_type: "年龄".to_string(),
        to: "30".to_string(),
    }]);
    let message = reasoner.assert_nl_fact("张三的年龄是31；李四的师傅是王五");
    assert!(message.contains("与图谱基值冲突"), "{}", message);
    assert_eq!(reasoner.session_facts().len(), 1, "{}", message);

    let resolved = reasoner.resolve_batch_answer("1B");
    assert!(resolved.contains("本批裁决提交"), "{}", resolved);
    assert!(reasoner.pending_batch_turn().is_none());
    assert!(reasoner.session_facts().iter().any(|atom| {
        atom.predicate.as_ref() == "年龄"
            && atom.terms.len() == 2
            && atom.terms[1].name() == "31"
    }));
}

#[test]
fn atomic_batch_writes_nothing_until_conflict_is_resolved() {
    let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
    let message = reasoner.assert_nl_fact(
        "原子记住：张三的年龄是30；张三的年龄是31；李四的师傅是王五",
    );
    assert!(message.contains("原子模式"), "{}", message);
    assert!(
        reasoner.session_facts().is_empty(),
        "原子模式有未决冲突时必须零写入：{}",
        message
    );
    assert!(reasoner.pending_batch_turn().is_some());

    let resolved = reasoner.resolve_batch_answer("1A");
    assert!(resolved.contains("本批冲突已全部裁决"), "{}", resolved);
    assert_eq!(reasoner.session_facts().len(), 2, "{}", resolved);
}

#[test]
fn correction_cue_inside_one_input_resolves_batch_without_question() {
    let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
    let message = reasoner.assert_nl_fact("张三30岁，其实31岁；李四32岁");
    assert!(message.contains("待确认 0 处"), "{}", message);
    assert!(reasoner.pending_batch_turn().is_none());
    assert_eq!(reasoner.session_facts().len(), 2, "{}", message);
    assert!(reasoner.session_facts().iter().any(|atom| {
        atom.predicate.as_ref() == "年龄"
            && atom.terms[0].name() == "张三"
            && atom.terms[1].name() == "31"
    }));
    assert!(!reasoner.session_facts().iter().any(|atom| {
        atom.predicate.as_ref() == "年龄"
            && atom.terms[0].name() == "张三"
            && atom.terms[1].name() == "30"
    }));
}

#[test]
fn pending_batch_survives_snapshot_roundtrip() {
    let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
    reasoner.assert_nl_fact("张三的年龄是30；张三的年龄是31");
    assert!(reasoner.pending_batch_turn().is_some());
    let snapshot = reasoner.dialogue_snapshot();

    let mut restored = SymbolicReasoner::new(test_config(), HashMap::new());
    restored
        .restore_dialogue_snapshot(&snapshot)
        .expect("批次挂起应随快照恢复");
    assert!(restored.pending_batch_turn().is_some());
    let resolved = restored.resolve_batch_answer("1B");
    assert!(resolved.contains("本批冲突已全部裁决"), "{}", resolved);
    assert!(restored.session_facts().iter().any(|atom| {
        atom.predicate.as_ref() == "年龄" && atom.terms[1].name() == "31"
    }));
}

#[test]
fn single_claim_still_uses_legacy_assertion_path() {
    let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
    let message = reasoner.assert_nl_fact("张三的师傅是李四");
    assert!(!message.contains("批次记住"), "{}", message);
    assert!(message.contains("已记住 1 条会话事实"), "{}", message);
    assert!(reasoner.pending_batch_turn().is_none());
}
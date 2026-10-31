fn proof_reasoner() -> SymbolicReasoner {
    let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
    reasoner.observe_facts(&[
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
    ]);
    reasoner
}

#[test]
fn repeated_proof_query_reuses_cached_proof() {
    let reasoner = proof_reasoner();
    let (first, first_access) = reasoner
        .resolution_proof_view_with_receipt("张三的师傅是谁")
        .expect("应能构建证明视图");
    assert!(!first_access.hit);
    assert!(first_access.stored, "{:?}", first_access);
    assert!(!first.solutions.is_empty());

    let (second, second_access) = reasoner
        .resolution_proof_view_with_receipt("张三的师傅是谁？")
        .expect("应命中证明记忆");
    assert!(second_access.hit, "{:?}", second_access);
    assert_eq!(second_access.hits, 1);
    assert_eq!(first.goal_display, second.goal_display);
    assert_eq!(first.stats, second.stats);
    assert_eq!(first.solutions.len(), second.solutions.len());
}

#[test]
fn knowledge_change_invalidates_cached_proof() {
    let mut reasoner = proof_reasoner();
    let (_, first_access) = reasoner
        .resolution_proof_view_with_receipt("张三的师傅是谁")
        .expect("应能构建证明视图");
    assert!(first_access.stored);

    reasoner.observe_facts(&[PrologFact::Relation {
        from: "张三".to_string(),
        rel_type: "师傅".to_string(),
        to: "赵六".to_string(),
    }]);
    let (view, access) = reasoner
        .resolution_proof_view_with_receipt("张三的师傅是谁")
        .expect("知识变化后仍应能重算");
    assert!(!access.hit, "知识变化后不得复用旧证明");
    assert!(view.solutions.len() >= 2, "新事实应进入重算结果");
}

#[test]
fn forked_session_does_not_share_proof_memory() {
    let reasoner = proof_reasoner();
    let (_, first_access) = reasoner
        .resolution_proof_view_with_receipt("张三的师傅是谁")
        .expect("应能构建证明视图");
    assert!(first_access.stored);

    let forked = reasoner.fork_session();
    let (_, forked_access) = forked
        .resolution_proof_view_with_receipt("张三的师傅是谁")
        .expect("分叉会话应能重算");
    assert!(!forked_access.hit, "证明记忆不得跨会话共享");
    assert!(forked_access.stored);
}

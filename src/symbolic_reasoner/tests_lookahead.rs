#[test]
fn clarification_lookahead_hits_and_preserves_observable_output() {
    let mut on = SymbolicReasoner::new(test_config(), HashMap::new());
    on.observe_facts(&master_facts());
    let _ = on.abduce_statement("张三的师祖是赵六").expect("应能溯因");
    let prompt_on = on.start_clarification().expect("应开始澄清");
    let hits_before = on.perf.count("clarification_lookahead_hits");
    let applied_before = on.perf.count("clarification_lookahead_applied");
    let answer_on = on.answer_clarification("是的").expect("应处理确认");
    assert!(
        on.perf.count("clarification_lookahead_hits") > hits_before,
        "确认分支应命中前瞻证据"
    );
    assert!(
        on.perf.count("clarification_lookahead_applied") > applied_before,
        "引擎与会话承诺一致时应采用前瞻证据"
    );
    assert_eq!(
        on.perf.count("clarification_lookahead_rejected"),
        0,
        "正常路径不应拒绝前瞻证据"
    );

    let mut off_config = test_config();
    off_config.enable_clarification_lookahead = false;
    let mut off = SymbolicReasoner::new(off_config, HashMap::new());
    off.observe_facts(&master_facts());
    let _ = off.abduce_statement("张三的师祖是赵六").expect("应能溯因");
    let prompt_off = off.start_clarification().expect("应开始澄清");
    let answer_off = off.answer_clarification("是的").expect("应处理确认");

    assert_eq!(prompt_on, prompt_off, "前瞻不得改变提问文本");
    // 归结证明在多个等价路径间可能给出不同置信度；比较时只去掉该数字，
    // 其余文本必须逐字节一致。
    let without_confidence = |text: &str| {
        text.lines()
            .map(|line| line.split("置信度").next().unwrap_or(line))
            .collect::<Vec<_>>()
            .join("\n")
    };
    assert_eq!(
        without_confidence(&answer_on),
        without_confidence(&answer_off),
        "前瞻不得改变回答文本"
    );
    assert_eq!(on.session_facts(), off.session_facts());
    assert_eq!(
        on.clarification_active(),
        off.clarification_active(),
        "前瞻不得改变会话生命周期"
    );
    assert!(
        on.clarification_lookahead_report().contains("应用 1 次"),
        "报告应如实显示应用次数：{}",
        on.clarification_lookahead_report()
    );
}

#[test]
fn clarification_lookahead_rejects_when_engine_changed_after_prefetch() {
    let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
    reasoner.observe_facts(&master_facts());
    let _ = reasoner
        .abduce_statement("张三的师祖是赵六")
        .expect("应能溯因");
    reasoner.start_clarification().expect("应开始澄清");
    // 模拟前瞻之后、回答之前发生的知识变化：封印中的引擎摘要不再匹配。
    reasoner.resolver.add_fact("临时", "甲", "乙");
    let misses_before = reasoner.perf.count("clarification_lookahead_misses");
    let output = reasoner.answer_clarification("是的").expect("应处理确认");
    assert!(
        reasoner.perf.count("clarification_lookahead_misses") > misses_before,
        "引擎摘要变化必须让前瞻未命中"
    );
    assert!(
        output.contains("已确认假设"),
        "拒绝前瞻后仍应走正常确认路径：{}",
        output
    );
    assert_eq!(
        reasoner.session_facts().len(),
        1,
        "正常路径仍应写入一条会话事实"
    );
}

#[test]
fn clarification_lookahead_apply_rejects_mismatched_engine_digest() {
    let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
    reasoner.observe_facts(&master_facts());
    let _ = reasoner
        .abduce_statement("张三的师祖是赵六")
        .expect("应能溯因");
    reasoner.start_clarification().expect("应开始澄清");
    let session = reasoner.clarification.clone().expect("应有澄清会话");
    let state = crate::lookahead::session_state_digest(&session);
    let wrong_engine = crate::lookahead::engine_digest(&reasoner.resolver).wrapping_add(1);
    let evidence =
        crate::lookahead::BranchEvidence::with_state(session, wrong_engine, state, false);
    let key = crate::lookahead::BranchKey::Confirm("测试假设".to_string());
    let rejected_before = reasoner.perf.count("clarification_lookahead_rejected");
    assert!(
        !reasoner.apply_clarification_lookahead(Some((key, evidence))),
        "引擎摘要不一致时必须拒绝证据"
    );
    assert!(
        reasoner.perf.count("clarification_lookahead_rejected") > rejected_before,
        "拒绝必须计入台账"
    );
    assert!(reasoner.clarification_active(), "拒绝证据不得改变真实会话");
}

#[test]
fn clarification_lookahead_clears_on_finish_and_reset() {
    let mut reasoner = SymbolicReasoner::new(test_config(), HashMap::new());
    reasoner.observe_facts(&master_facts());
    let _ = reasoner
        .abduce_statement("张三的师祖是赵六")
        .expect("应能溯因");
    reasoner.start_clarification().expect("应开始澄清");
    assert!(
        reasoner
            .clarification_lookahead_report()
            .contains("已预计算 2 条"),
        "默认应为「是/不是」预计算两条：{}",
        reasoner.clarification_lookahead_report()
    );
    reasoner.stop_clarification().expect("应结束澄清");
    assert!(
        reasoner
            .clarification_lookahead_report()
            .contains("已预计算 0 条"),
        "结束澄清应清空前瞻：{}",
        reasoner.clarification_lookahead_report()
    );

    let _ = reasoner
        .abduce_statement("张三的师祖是赵六")
        .expect("应能溯因");
    reasoner.start_clarification().expect("应开始澄清");
    reasoner.fork_session();
    assert!(
        reasoner
            .clarification_lookahead_report()
            .contains("已预计算 2 条"),
        "原实例的前瞻不应被分叉清空：{}",
        reasoner.clarification_lookahead_report()
    );
    let fork = reasoner.fork_session();
    assert!(
        fork.clarification_lookahead_report()
            .contains("已预计算 0 条"),
        "分叉应从空前瞻开始：{}",
        fork.clarification_lookahead_report()
    );
}

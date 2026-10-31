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

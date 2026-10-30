fn check_reasoning(c: &mut Collector, config: &SystemConfig) {
    let r = &config.reasoning;
    check_probability(
        c,
        r.link_prediction_min_confidence,
        "reasoning.link_prediction_min_confidence",
        "链接预测置信度阈值",
    );
    check_probability(
        c,
        r.ontology_alignment_threshold,
        "reasoning.ontology_alignment_threshold",
        "本体对齐阈值",
    );
    check_probability(
        c,
        r.embedding_similarity_floor,
        "reasoning.embedding_similarity_floor",
        "嵌入相似度下限",
    );
    check_probability(
        c,
        r.completion_min_score,
        "reasoning.completion_min_score",
        "统计续写最低支持度",
    );
    check_probability(
        c,
        r.entailment_min_confidence,
        "reasoning.entailment_min_confidence",
        "蕴含证据最低置信度",
    );
    check_probability(
        c,
        r.association_min_support,
        "reasoning.association_min_support",
        "关联规则最小支持度",
    );
    check_probability(
        c,
        r.association_min_confidence,
        "reasoning.association_min_confidence",
        "关联规则最小置信度",
    );
    check_probability(
        c,
        r.soft_logic_min_probability,
        "reasoning.soft_logic_min_probability",
        "软逻辑最低概率",
    );
    check_probability(
        c,
        r.holographic_similarity_floor,
        "reasoning.holographic_similarity_floor",
        "全息召回相似度下限",
    );

    if r.link_prediction_min_support == 0 {
        c.warn(
            "zero_link_prediction_support",
            "reasoning.link_prediction_min_support",
            "链接预测最小支持度为 0：任意偶然共现都会成为规则",
        );
    }
    if r.max_predictions == 0 {
        c.warn(
            "zero_max_predictions",
            "reasoning.max_predictions",
            "max_predictions=0：推理结果将一条都不展示",
        );
    }
    if r.max_profile_sentences == 0 {
        c.warn(
            "zero_profile_sentences",
            "reasoning.max_profile_sentences",
            "max_profile_sentences=0：运行时会被抬升为 1 句",
        );
    }
    if r.association_min_lift < 1.0 {
        c.warn(
            "low_association_lift",
            "reasoning.association_min_lift",
            format!(
                "association_min_lift={} < 1.0：提升度低于 1 表示负相关，规则将失去统计意义",
                r.association_min_lift
            ),
        );
    }
    if !r.graph_walk_damping.is_finite()
        || r.graph_walk_damping <= 0.0
        || r.graph_walk_damping >= 1.0
    {
        c.error(
            "damping_out_of_range",
            "reasoning.graph_walk_damping",
            format!(
                "graph_walk_damping={} 不在 (0,1)：随机游走将退化",
                r.graph_walk_damping
            ),
        );
    } else if !(0.05..=0.99).contains(&r.graph_walk_damping) {
        c.warn(
            "damping_will_be_clamped",
            "reasoning.graph_walk_damping",
            format!(
                "graph_walk_damping={} 超出运行时区间 [0.05,0.99]：将被静默钳制",
                r.graph_walk_damping
            ),
        );
    }
    check_clamped_range(
        c,
        r.graph_walk_iterations,
        1,
        100,
        "reasoning.graph_walk_iterations",
        "随机游走迭代轮数",
    );
    if r.resolution_max_depth == 0 {
        c.warn(
            "zero_resolution_depth",
            "reasoning.resolution_max_depth",
            "resolution_max_depth=0：运行时会被抬升为 1 层",
        );
    }
    check_clamped_range(
        c,
        r.resolution_max_solutions,
        1,
        200,
        "reasoning.resolution_max_solutions",
        "归结查询最大解数",
    );
    check_clamped_range(
        c,
        r.abduction_max_hypotheses,
        1,
        20,
        "reasoning.abduction_max_hypotheses",
        "溯因最大假设数",
    );
    check_clamped_range(
        c,
        r.abduction_max_scenario_size,
        1,
        4,
        "reasoning.abduction_max_scenario_size",
        "溯因场景最大假设数",
    );
    if r.abduction_max_scenario_size > r.abduction_max_hypotheses {
        c.warn(
            "scenario_exceeds_hypotheses",
            "reasoning.abduction_max_scenario_size",
            "溯因场景大小超过最大假设数：多余槽位永远不会用满",
        );
    }
    check_clamped_range(
        c,
        r.soft_logic_max_iterations,
        1,
        32,
        "reasoning.soft_logic_max_iterations",
        "软逻辑迭代轮数",
    );
    check_clamped_range(
        c,
        r.causal_max_depth,
        1,
        8,
        "reasoning.causal_max_depth",
        "因果链最大深度",
    );
    check_clamped_range(
        c,
        r.problog_max_depth,
        1,
        16,
        "reasoning.problog_max_depth",
        "概率证明最大深度",
    );
    check_clamped_range(
        c,
        r.problog_max_proofs,
        1,
        64,
        "reasoning.problog_max_proofs",
        "概率查询最大证明路径数",
    );
    if r.discourse_max_sentences == 0 {
        c.warn(
            "zero_discourse_sentences",
            "reasoning.discourse_max_sentences",
            "discourse_max_sentences=0：运行时会被抬升为 1 句",
        );
    }
    if r.discourse_max_rounds == 0 {
        c.warn(
            "zero_discourse_rounds",
            "reasoning.discourse_max_rounds",
            "discourse_max_rounds=0：运行时会被抬升为 1 轮",
        );
    }
    check_clamped_range(
        c,
        r.holographic_dimension,
        256,
        16384,
        "reasoning.holographic_dimension",
        "全息超向量维度",
    );
    if r.holographic_similarity_floor < 0.0 {
        c.warn(
            "negative_holographic_floor",
            "reasoning.holographic_similarity_floor",
            format!(
                "holographic_similarity_floor={} 为负：运行时会被钳制到 0",
                r.holographic_similarity_floor
            ),
        );
    }
    if r.holographic_max_hits == 0 {
        c.warn(
            "zero_holographic_hits",
            "reasoning.holographic_max_hits",
            "holographic_max_hits=0：全息召回不可见",
        );
    }
    if let Some(path) = &r.dialogue_state_path {
        if path.trim().is_empty() {
            c.warn(
                "empty_dialogue_state_path",
                "reasoning.dialogue_state_path",
                "对话记忆路径为空：跨会话记忆无法落盘",
            );
        }
    }
    check_rewrite_retrieval(c, &r.rewrite_retrieval);
}
fn check_rewrite_retrieval(
    c: &mut Collector,
    r: &crate::rewrite_retrieval::RewriteRetrievalConfig,
) {
    for (value, path, label) in [
        (
            r.accept_threshold,
            "reasoning.rewrite_retrieval.accept_threshold",
            "读法接受阈值",
        ),
        (
            r.clarify_threshold,
            "reasoning.rewrite_retrieval.clarify_threshold",
            "读法澄清阈值",
        ),
        (
            r.accept_margin,
            "reasoning.rewrite_retrieval.accept_margin",
            "读法接受边际",
        ),
        (
            r.min_support,
            "reasoning.rewrite_retrieval.min_support",
            "读法证据支持闸门",
        ),
        (
            r.min_coverage,
            "reasoning.rewrite_retrieval.min_coverage",
            "读法答案覆盖闸门",
        ),
        (
            r.min_margin,
            "reasoning.rewrite_retrieval.min_margin",
            "读法答案边际闸门",
        ),
        (
            r.support_pivot,
            "reasoning.rewrite_retrieval.support_pivot",
            "读法支持度中性点",
        ),
        (
            r.coverage_pivot,
            "reasoning.rewrite_retrieval.coverage_pivot",
            "读法覆盖度中性点",
        ),
        (
            r.margin_pivot,
            "reasoning.rewrite_retrieval.margin_pivot",
            "读法边际中性点",
        ),
        (
            r.consistency_pivot,
            "reasoning.rewrite_retrieval.consistency_pivot",
            "读法一致性中性点",
        ),
        (
            r.partial_discount,
            "reasoning.rewrite_retrieval.partial_discount",
            "读法首跳折扣",
        ),
    ] {
        check_probability(c, value, path, label);
    }
    if r.clarify_threshold > r.accept_threshold {
        c.warn(
            "clarify_above_accept",
            "reasoning.rewrite_retrieval.clarify_threshold",
            "澄清阈值高于接受阈值：运行时会把澄清阈值压到接受阈值",
        );
    }
    check_clamped_range(
        c,
        r.max_hypotheses,
        1,
        8,
        "reasoning.rewrite_retrieval.max_hypotheses",
        "读法候选数",
    );
    check_clamped_range(
        c,
        r.max_probes,
        1,
        4,
        "reasoning.rewrite_retrieval.max_probes",
        "读法探针数",
    );
    check_clamped_range(
        c,
        r.per_probe_k,
        1,
        20,
        "reasoning.rewrite_retrieval.per_probe_k",
        "读法探针召回数",
    );
    for (value, path, label) in [
        (
            r.support_gain,
            "reasoning.rewrite_retrieval.support_gain",
            "读法支持度增益",
        ),
        (
            r.coverage_gain,
            "reasoning.rewrite_retrieval.coverage_gain",
            "读法覆盖度增益",
        ),
        (
            r.margin_gain,
            "reasoning.rewrite_retrieval.margin_gain",
            "读法边际增益",
        ),
        (
            r.consistency_gain,
            "reasoning.rewrite_retrieval.consistency_gain",
            "读法一致性增益",
        ),
    ] {
        if !value.is_finite() {
            c.error(
                "non_finite_gain",
                path.to_string(),
                format!("{}不是有限数（{}）", label, value),
            );
        } else if !(0.0..=10.0).contains(&value) {
            c.warn(
                "gain_will_be_clamped",
                path.to_string(),
                format!("{}={} 超出运行时区间 [0,10]：将被静默钳制", label, value),
            );
        }
    }
}
fn check_soft_route(c: &mut Collector, config: &SystemConfig) {
    if let Err(issue) = config.reasoning.soft_route.validate() {
        c.error("invalid_soft_route", "reasoning.soft_route", issue);
    }
}
fn check_confidence(c: &mut Collector, config: &SystemConfig) {
    for issue in config.reasoning.confidence.validate() {
        match issue.severity {
            crate::confidence::ConfidenceSeverity::Error => c.error(
                &issue.code,
                issue.path,
                issue.message,
            ),
            crate::confidence::ConfidenceSeverity::Warning => c.warn(
                &issue.code,
                issue.path,
                issue.message,
            ),
        }
    }
}
fn check_trace_graph(c: &mut Collector, config: &SystemConfig) {
    let t = &config.trace_graph;
    if trace_graph::TraceMode::parse(&t.mode).is_none() {
        c.error(
            "unknown_trace_mode",
            "trace_graph.mode",
            format!(
                "未知的分层追踪模式「{}」：可选 off / observe / on（非法值将回落为 off）",
                t.mode
            ),
        );
    }
    if t.max_turns == 0 {
        c.warn(
            "zero_trace_turns",
            "trace_graph.max_turns",
            "max_turns=0：追踪图无法保留任何轮迹",
        );
    }
    if t.max_nodes_per_turn == 0 {
        c.warn(
            "zero_trace_nodes",
            "trace_graph.max_nodes_per_turn",
            "max_nodes_per_turn=0：单轮不记录任何决策节点",
        );
    }
    if t.max_events_per_turn == 0 {
        c.warn(
            "zero_trace_events",
            "trace_graph.max_events_per_turn",
            "max_events_per_turn=0：单轮不记录任何指代事件",
        );
    }
}
fn check_dialogue_frames(c: &mut Collector, config: &SystemConfig) {
    let frames = &config.dialogue_frames;
    if frames.frames.is_empty() {
        if frames.enabled {
            c.warn(
                "empty_dialogue_frames",
                "dialogue_frames.frames",
                "enabled=true 但未声明任何帧：绑定帧层不会接管任何输入",
            );
        }
        return;
    }
    if let Err(errors) = FrameProgram::compile(frames) {
        for error in errors {
            c.error(error.code, error.path, error.message);
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    /// 最小可用配置：所有必需字段齐备，可被故障注入测试当作基线。
    const BASE_CONFIG: &str = r#"
[system]
oxigraph_endpoint = "http://localhost:7878"
oxigraph_store_path = "oxigraph_data"
default_language = "zh"

[nlp]
import_keywords = ["录入", "添加"]
nlu_confidence_threshold = 0.6
semantic_similarity_threshold = 0.7
max_entity_candidates = 10
context_window_size = 5

[nlp.pos_dict]
"是" = "v"
"的" = "u"
"年龄" = "n"
"多大" = "r"
"谁" = "r"

[nlp.name_patterns]
prefixes = ["小", "老"]
suffixes = ["总"]
full_names = ["张三"]
profession_suffix = ["医生"]

[[prolog_rules]]
name = "年龄查询"
pattern = ["nr", "多大"]
query_type = "age_query"
priority = 10

[[prolog_rules]]
name = "关系查询"
pattern = ["nr", "和", "nr", "什么", "关系"]
query_type = "relation_query"
priority = 9

[[entity_types]]
label = "Person"
examples = ["张三18岁"]

[[entity_types.properties]]
name = "name"
data_type = "string"
required = true
extraction_patterns = ["pos:nr"]

[[entity_types.properties]]
name = "age"
data_type = "integer"
required = false
extraction_patterns = ["pos:m"]

[[extraction_rules]]
name = "人物信息提取"
entity_type = "Person"
priority = 10

[[extraction_rules.patterns]]
pos_pattern = ["nr", "m", "q"]
property_mappings = { name = "pos:nr", age = "pos:m" }

[messages]
startup = "启动中"
ready = "就绪"
query_prompt = "请输入查询"
import_prompt = "输入导入"
quit_prompt = "输入quit退出"
importing = "导入中"
no_knowledge_extracted = "未提取到知识"
cannot_understand = "无法理解"
process_error = "处理出错"
"#;

    fn base() -> String {
        BASE_CONFIG.to_string()
    }

    fn lint_source(source: &str) -> Vec<ConfigIssue> {
        lint_str(source).expect("测试配置应能解析")
    }

    fn codes(issues: &[ConfigIssue]) -> Vec<String> {
        issues.iter().map(|issue| issue.code.clone()).collect()
    }

    fn assert_code(issues: &[ConfigIssue], code: &str) {
        assert!(
            issues.iter().any(|issue| issue.code == code),
            "期望问题码 {}，实际：{:?}\n{}",
            code,
            codes(issues),
            render_text(issues)
        );
    }

    #[test]
    fn base_config_is_clean() {
        let issues = lint_source(&base());
        assert!(
            !has_errors(&issues),
            "基线配置不应有 Error：{}",
            render_text(&issues)
        );
    }

    #[test]
    fn shipped_config_has_no_errors() {
        let source = include_str!("../../config.toml");
        let issues = lint_source(source);
        assert!(
            !has_errors(&issues),
            "出厂 config.toml 不应有 Error：\n{}",
            render_text(&issues)
        );
    }

    #[test]
    fn semantic_grammar_rules_are_compiled_and_reported() {
        let valid = format!(
            "{}\n[semantic_grammar]\nenabled = true\n\n[[semantic_grammar.rules]]\nname = \"身高\"\npredicate = \"height\"\nentity_type = \"Person\"\npattern = [\"$s:pos:nr\", \"身高\", \"$o:pos:m\"]\nrealize = \"{{s}}身高{{o}}\"\n",
            base()
        );
        let issues = lint_source(&valid);
        assert!(
            !issues
                .iter()
                .any(|issue| issue.code.starts_with("semantic_")),
            "合法双向规则不应报语义问题：{}",
            render_text(&issues)
        );

        let invalid = format!(
            "{}\n[semantic_grammar]\nenabled = true\n\n[[semantic_grammar.rules]]\nname = \"坏规则\"\npredicate = \"height\"\npattern = [\"$s:pos:nr\"]\nrealize = \"{{s}}\"\n",
            base()
        );
        assert_code(&lint_source(&invalid), "semantic_rule_compile_error");

        let disabled = format!(
            "{}\n[semantic_grammar]\nenabled = false\n\n[[semantic_grammar.rules]]\nname = \"预览\"\npredicate = \"height\"\npattern = [\"$s:pos:nr\", \"身高\", \"$o:pos:m\"]\nrealize = \"{{s}}身高{{o}}\"\n",
            base()
        );
        assert_code(&lint_source(&disabled), "semantic_grammar_disabled");
    }

    #[test]
    fn trace_graph_mode_is_validated() {
        // 合法模式：不报错误。
        for mode in ["off", "observe", "on"] {
            let source = format!("{}\n[trace_graph]\nmode = \"{}\"\n", base(), mode);
            let issues = lint_source(&source);
            assert!(
                !issues
                    .iter()
                    .any(|issue| issue.code == "unknown_trace_mode"),
                "模式 {} 不应报 unknown_trace_mode：{}",
                mode,
                render_text(&issues)
            );
        }
        // 非法模式：报 unknown_trace_mode。
        let source = format!("{}\n[trace_graph]\nmode = \"bogus\"\n", base());
        assert_code(&lint_source(&source), "unknown_trace_mode");
    }

    #[test]
    fn trace_graph_zero_limits_are_warnings() {
        let source = format!(
            "{}\n[trace_graph]\nmode = \"off\"\nmax_turns = 0\nmax_nodes_per_turn = 0\nmax_events_per_turn = 0\n",
            base()
        );
        let issues = lint_source(&source);
        assert_code(&issues, "zero_trace_turns");
        assert_code(&issues, "zero_trace_nodes");
        assert_code(&issues, "zero_trace_events");
    }

    #[test]
    fn shipped_config_has_no_errors_placeholder() {
        let source = include_str!("../../config.toml");
        let issues = lint_source(source);
        let config: SystemConfig = toml::from_str(source).expect("config.toml 应能解析");
        let _ = &config;
        assert!(
            !has_errors(&issues),
            "出厂 config.toml 不应有 Error：\n{}",
            render_text(&issues)
        );
    }

    #[test]
    fn shipped_config_query_types_are_all_supported() {
        let source = include_str!("../../config.toml");
        let config: SystemConfig = toml::from_str(source).expect("config.toml 应能解析");
        for rule in &config.prolog_rules {
            assert!(
                is_query_type_supported(rule.query_type.trim()),
                "出厂配置的 query_type「{}」应在能力图上",
                rule.query_type
            );
        }
    }

    #[test]
    fn prolog_generated_list_matches_engine() {
        for query_type in GENERATED_QUERY_TYPES {
            let query = crate::prolog_engine::PrologEngine::generate_prolog_query(
                query_type,
                &["测试实体".to_string(), "另一实体".to_string()],
            );
            assert!(
                query.is_some(),
                "能力表声称可生成查询，但引擎未实现：{}",
                query_type
            );
        }
    }

    #[test]
    fn dangling_entity_type_is_error() {
        let issues =
            lint_source(&base().replace("entity_type = \"Person\"", "entity_type = \"Persom\""));
        assert_code(&issues, "dangling_entity_type");
        assert!(issues
            .iter()
            .any(|issue| issue.code == "dangling_entity_type"
                && issue.severity == ContractSeverity::Error));
    }

    #[test]
    fn dangling_query_type_is_error() {
        let source = format!(
            "{}\n[[prolog_rules]]\nname = \"创始人查询\"\npattern = [\"n\", \"创始人\"]\nquery_type = \"founder_query\"\npriority = 1\n",
            base()
        );
        let issues = lint_source(&source);
        assert_code(&issues, "dangling_query_type");
    }

    #[test]
    fn empty_query_pattern_and_token_are_errors() {
        let source = format!(
            "{}\n[[prolog_rules]]\nname = \"坏规则\"\npattern = [\"\"]\nquery_type = \"age_query\"\npriority = 1\n",
            base()
        );
        let issues = lint_source(&source);
        assert_code(&issues, "empty_pattern_token");
        let source = format!(
            "{}\n[[prolog_rules]]\nname = \"空规则\"\npattern = []\nquery_type = \"age_query\"\npriority = 1\n",
            base()
        );
        let issues = lint_source(&source);
        assert_code(&issues, "empty_query_pattern");
    }

    #[test]
    fn insufficient_entity_slots_is_warning() {
        let source = format!(
            "{}\n[[prolog_rules]]\nname = \"关系查询2\"\npattern = [\"nr\", \"和\", \"什么\", \"关系\"]\nquery_type = \"relation_query\"\npriority = 1\n",
            base()
        );
        let issues = lint_source(&source);
        assert_code(&issues, "insufficient_entity_slots");
        assert!(issues
            .iter()
            .all(|issue| issue.code != "insufficient_entity_slots"
                || issue.severity == ContractSeverity::Warning));
    }

    #[test]
    fn ascii_pattern_token_treated_as_word_is_warning() {
        let source = format!(
            "{}\n[[prolog_rules]]\nname = \"笔误规则\"\npattern = [\"nn\", \"年龄\"]\nquery_type = \"age_query\"\npriority = 1\n",
            base()
        );
        let issues = lint_source(&source);
        assert_code(&issues, "pattern_token_treated_as_word");
    }

    #[test]
    fn duplicate_entity_type_and_property_are_errors() {
        let source = format!(
            "{}\n[[entity_types]]\nlabel = \"Person\"\nexamples = []\n\n[[entity_types.properties]]\nname = \"name\"\ndata_type = \"string\"\nrequired = true\nextraction_patterns = [\"pos:nr\"]\n",
            base()
        );
        let issues = lint_source(&source);
        assert_code(&issues, "duplicate_entity_type");

        let source = base().replace(
            "[[entity_types.properties]]\nname = \"age\"",
            "[[entity_types.properties]]\nname = \"name\"",
        );
        let issues = lint_source(&source);
        assert_code(&issues, "duplicate_property");
    }

    #[test]
    fn undeclared_property_mapping_is_warning() {
        let source = base().replace(
            "property_mappings = { name = \"pos:nr\", age = \"pos:m\" }",
            "property_mappings = { name = \"pos:nr\", height = \"pos:m\" }",
        );
        let issues = lint_source(&source);
        assert_code(&issues, "undeclared_property_mapping");
    }

    #[test]
    fn property_extraction_patterns_are_optional_but_inert() {
        let issues = lint_source(&base());
        assert_code(&issues, "inert_property_extraction_patterns");

        let without = base()
            .replace("extraction_patterns = [\"pos:nr\"]\n", "")
            .replace("extraction_patterns = [\"pos:m\"]\n", "");
        let issues = lint_source(&without);
        assert!(
            !codes(&issues)
                .iter()
                .any(|code| code == "inert_property_extraction_patterns"),
            "省略 extraction_patterns 后不应再提示：{}",
            render_text(&issues)
        );
    }

    #[test]
    fn predicate_alias_conflicts_are_errors() {
        let source = format!(
            "{}\n[[predicate_profiles]]\nname = \"师傅\"\naliases = [\"老师\"]\ninverse = \"徒弟\"\n\n[[predicate_profiles]]\nname = \"徒弟\"\ninverse = \"师傅\"\n\n[[predicate_profiles]]\nname = \"师父\"\naliases = [\"老师\"]\n\n[[predicate_profiles]]\nname = \"导师\"\naliases = [\"徒弟\"]\n\n[[predicate_profiles]]\nname = \"甲\"\ninverse = \"乙\"\n\n[[predicate_profiles]]\nname = \"丙\"\naliases = [\"乙\"]\n",
            base()
        );
        let issues = lint_source(&source);
        assert_code(&issues, "predicate_alias_conflict");
        assert_code(&issues, "alias_shadows_canonical");
        assert_code(&issues, "alias_conflicts_with_inverse");
    }

    #[test]
    fn predicate_inverse_checks() {
        let source = format!(
            "{}\n[[predicate_profiles]]\nname = \"甲\"\ninverse = \"甲\"\n\n[[predicate_profiles]]\nname = \"乙\"\ninverse = \"丙\"\n\n[[predicate_profiles]]\nname = \"丙\"\ninverse = \"乙\"\n\n[[predicate_profiles]]\nname = \"丁\"\ninverse = \"戊\"\n\n[[predicate_profiles]]\nname = \"戊\"\ninverse = \"丁\"\n",
            base()
        );
        let issues = lint_source(&source);
        assert_code(&issues, "self_inverse");
        // 乙↔丙 互逆一致；丁→戊 未互报：戊声明丁，方向一致，不应误报
        assert!(
            !issues
                .iter()
                .any(|issue| issue.code == "inverse_not_reciprocated"),
            "互逆一致不应误报：{}",
            render_text(&issues)
        );

        let source = format!(
            "{}\n[[predicate_profiles]]\nname = \"甲\"\ninverse = \"乙\"\n\n[[predicate_profiles]]\nname = \"乙\"\ninverse = \"丙\"\n\n[[predicate_profiles]]\nname = \"丙\"\n",
            base()
        );
        let issues = lint_source(&source);
        assert_code(&issues, "inverse_not_reciprocated");
    }

    #[test]
    fn unknown_domain_type_is_warning() {
        let source = format!(
            "{}\n[[predicate_profiles]]\nname = \"师傅\"\ndomain = [\"人间\"]\nrange = [\"Person\"]\n",
            base()
        );
        let issues = lint_source(&source);
        assert_code(&issues, "unknown_domain_type");
        assert!(
            !issues
                .iter()
                .any(|issue| issue.code == "unknown_range_type"),
            "已知类型 Person 不应误报：{}",
            render_text(&issues)
        );
    }

    #[test]
    fn type_alias_cycle_is_error() {
        let source = format!(
            "{}\n[type_system]\n\n[[type_system.aliases]]\nalias = \"A\"\ncanonical = \"B\"\n\n[[type_system.aliases]]\nalias = \"B\"\ncanonical = \"A\"\n",
            base()
        );
        let issues = lint_source(&source);
        assert_code(&issues, "type_alias_cycle");
    }

    #[test]
    fn subclass_cycle_is_error() {
        let source = format!(
            "{}\n[[type_system.subclasses]]\nsub = \"学生\"\nsup = \"人物\"\n\n[[type_system.subclasses]]\nsub = \"人物\"\nsup = \"学生\"\n",
            base()
        );
        let issues = lint_source(&source);
        assert_code(&issues, "subclass_cycle");
    }

    #[test]
    fn contradictory_disjoint_is_error() {
        let source = format!(
            "{}\n[[type_system.aliases]]\nalias = \"Person\"\ncanonical = \"人物\"\n\n[[type_system.disjoint]]\nleft = \"人物\"\nright = \"Person\"\n",
            base()
        );
        let issues = lint_source(&source);
        assert_code(&issues, "contradictory_disjoint");
    }

    #[test]
    fn synonym_self_chain_and_cycle_are_warnings() {
        let source = format!(
            "{}\n[synonyms.mappings]\n\"甲\" = \"甲\"\n\"乙\" = \"丙\"\n\"丙\" = \"丁\"\n\"戊\" = \"己\"\n\"己\" = \"戊\"\n",
            base()
        );
        let issues = lint_source(&source);
        assert_code(&issues, "self_synonym");
        assert_code(&issues, "synonym_chain");
        assert_code(&issues, "synonym_cycle");
    }

    #[test]
    fn probability_out_of_range_is_error() {
        let source = base().replace(
            "[nlp]\nimport_keywords",
            "[reasoning]\nsoft_logic_min_probability = 1.5\n\n[nlp]\nimport_keywords",
        );
        let source = source.replace(
            "nlu_confidence_threshold = 0.6",
            "nlu_confidence_threshold = -0.2",
        );
        let issues = lint_source(&source);
        assert_code(&issues, "threshold_out_of_range");
        assert_eq!(error_count(&issues), 2, "{}", render_text(&issues));
    }

    #[test]
    fn rewrite_retrieval_thresholds_are_checked() {
        let source = format!(
            "{}\n[reasoning.rewrite_retrieval]\naccept_threshold = 1.5\nclarify_threshold = 2.0\nmax_hypotheses = 0\nsupport_gain = 50.0\npartial_discount = -0.5\n",
            base()
        );
        let issues = lint_source(&source);
        assert_code(&issues, "threshold_out_of_range");
        assert_code(&issues, "clarify_above_accept");
        assert_code(&issues, "value_will_be_clamped");
        assert_code(&issues, "gain_will_be_clamped");
    }

    #[test]
    fn damping_out_of_range_is_error() {
        let source = base().replace(
            "[nlp]\nimport_keywords",
            "[reasoning]\ngraph_walk_damping = 1.5\n\n[nlp]\nimport_keywords",
        );
        let issues = lint_source(&source);
        assert_code(&issues, "damping_out_of_range");
    }

    #[test]
    fn clamping_band_is_warning() {
        let source = base().replace(
            "[nlp]\nimport_keywords",
            "[reasoning]\ncausal_max_depth = 99\nholographic_dimension = 128\n\n[nlp]\nimport_keywords",
        );
        let issues = lint_source(&source);
        assert_code(&issues, "value_will_be_clamped");
        assert!(issues
            .iter()
            .filter(|issue| issue.code == "value_will_be_clamped")
            .all(|issue| issue.severity == ContractSeverity::Warning));
    }

    #[test]
    fn inert_oxigraph_endpoint_is_warning() {
        let issues = lint_source(&base());
        assert_code(&issues, "inert_oxigraph_endpoint");
        assert!(issues
            .iter()
            .filter(|issue| issue.code == "inert_oxigraph_endpoint")
            .all(|issue| issue.severity == ContractSeverity::Warning));

        let blanked = base().replace(
            "oxigraph_endpoint = \"http://localhost:7878\"",
            "oxigraph_endpoint = \"\"",
        );
        let issues = lint_source(&blanked);
        assert!(
            !codes(&issues)
                .iter()
                .any(|code| code == "inert_oxigraph_endpoint"),
            "留空后不应再提示：{}",
            render_text(&issues)
        );
    }

    #[test]
    fn strict_admission_without_profiles_is_warning() {
        let source = base().replace(
            "[nlp]\nimport_keywords",
            "[reasoning]\nrequire_predicate_admission = true\n\n[nlp]\nimport_keywords",
        );
        let issues = lint_source(&source);
        assert_code(&issues, "strict_admission_without_profiles");
    }

    #[test]
    fn entity_profile_conflicts_are_errors() {
        let source = format!(
            "{}\n[[entity_profiles]]\nname = \"张三\"\naliases = [\"老张\"]\netype = [\"Person\"]\n\n[[entity_profiles]]\nname = \"李四\"\naliases = [\"老张\"]\netype = [\"外星人\"]\n",
            base()
        );
        let issues = lint_source(&source);
        assert_code(&issues, "entity_alias_conflict");
        assert_code(&issues, "unknown_entity_profile_type");
    }

    #[test]
    fn stale_confidence_certificate_is_an_error() {
        let source = format!(
            "{}\n[reasoning.confidence]\nenabled = true\nanswer_threshold = 0.86\nhedge_threshold = 0.62\nclarify_threshold = 0.55\ncertificate = {{ n = 240, errors = 4, alpha = 0.05, delta = 0.001, max_error_rate = 0.05, claimed_upper_bound = 0.0100 }}\n\n[[reasoning.confidence.sources]]\nname = \"intent\"\ngamma = 1.0\ncalibration = [{{ score = 0.5, probability = 0.7 }}, {{ score = 0.9, probability = 0.9 }}]\n",
            base()
        );
        let issues = lint_source(&source);
        assert_code(&issues, "confidence_certificate_invalid");
    }

    #[test]
    fn attribute_profiles_report_duplicates_and_alias_conflicts() {
        let source = format!(
            "{}\n[attributes]\nenabled = true\n\n[[attributes.profiles]]\nname = \"height\"\naliases = [\"身高\", \"身高\"]\n\n[[attributes.profiles]]\nname = \"height\"\n\n[[attributes.profiles]]\nname = \"weight\"\naliases = [\"height\"]\n",
            base()
        );
        let issues = lint_source(&source);
        assert_code(&issues, "duplicate_attribute_profile");
        assert_code(&issues, "duplicate_attribute_alias");
        assert_code(&issues, "attribute_alias_shadows_canonical");
    }

    #[test]
    fn attribute_profiles_report_schema_and_unit_problems() {
        let source = format!(
            "{}\n[attributes]\nenabled = true\n\n[[attributes.profiles]]\nname = \"height\"\ndomain = [\"Persom\"]\nvalue_type = \"quantity\"\ndimension = \"length\"\nunit = \"千克\"\nparent = \"missing\"\n\n[[attributes.profiles]]\nname = \"甲\"\nparent = \"乙\"\n\n[[attributes.profiles]]\nname = \"乙\"\nparent = \"甲\"\n\n[[attributes.profiles]]\nname = \"width\"\ndimension = \"warp\"\nunit = \"不存在单位\"\n",
            base()
        );
        let issues = lint_source(&source);
        assert_code(&issues, "unknown_attribute_domain_type");
        assert_code(&issues, "attribute_unit_dimension_mismatch");
        assert_code(&issues, "unknown_attribute_parent");
        assert_code(&issues, "attribute_parent_cycle");
        assert_code(&issues, "unknown_attribute_dimension");
        assert_code(&issues, "unknown_attribute_unit");
    }

    #[test]
    fn attribute_and_predicate_surface_divergence_is_reported() {
        let source = format!(
            "{}\n[[predicate_profiles]]\nname = \"年龄\"\naliases = [\"岁数\"]\n\n[attributes]\nenabled = true\n\n[[attributes.profiles]]\nname = \"age\"\naliases = [\"年龄\"]\ndomain = [\"Person\"]\nvalue_type = \"quantity\"\ndimension = \"duration\"\nunit = \"岁\"\n",
            base()
        );
        let issues = lint_source(&source);
        assert_code(&issues, "attribute_predicate_surface_conflict");
    }

    #[test]
    fn parse_errors_are_returned() {
        assert!(lint_str("this is not toml = = =").is_err());
    }

    #[test]
    fn lint_is_deterministic() {
        let source = include_str!("../../config.toml");
        let first = lint_source(source);
        let second = lint_source(source);
        assert_eq!(first, second);
        assert_eq!(render_text(&first), render_text(&second));
    }

    #[test]
    fn issues_are_json_round_trippable() {
        let issues = lint_source(&base());
        let json = serde_json::to_string(&issues).expect("问题列表应能序列化");
        let back: Vec<ConfigIssue> = serde_json::from_str(&json).expect("问题列表应能反序列化");
        assert_eq!(issues, back);
    }
}

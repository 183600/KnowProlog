/// 对单个已解析计划做静态覆盖审计。
pub fn audit_lexical_coverage(
    question: &str,
    tokens: &[TraceToken],
    plan: &QueryPlan,
    inventory: &LexiconInventory,
) -> LexicalCoverage {
    let chars: Vec<char> = question.chars().collect();
    let mut used = vec![false; chars.len()];
    let mut gaps = Vec::new();
    let mut resolved = Vec::new();
    let mut known_terms = 0u64;
    let mut introduced_terms = 0u64;

    let (entities, predicates) = plan_terms(plan);
    for predicate in &predicates {
        let span = locate_term(&chars, predicate, &mut used);
        if inventory.is_predicate(predicate) {
            known_terms += 1;
            if let Some(expansion) = inventory.definition_expansion(predicate) {
                resolved.push(LexResolvedTerm {
                    text: predicate.clone(),
                    expansion: expansion.to_vec(),
                });
            }
            continue;
        }
        let Some(span) = span else {
            introduced_terms += 1;
            gaps.push(LexCoverageGap {
                text: predicate.clone(),
                span: None,
                layer: LexGapLayer::Sense,
                role: LexGapRole::Predicate,
                severity: ResidualSeverity::Warning,
                repair: RepairKind::SupplyPredicate,
                evidence: format!(
                    "计划引入的谓词「{}」不在谓词画像/查询规则/词性词典中；可能是属性归一结果，但没有可执行定义",
                    predicate
                ),
            });
            continue;
        };
        let unknown = span_unknown_surfaces(tokens, span);
        let evidence = if unknown.is_empty() {
            format!(
                "谓词槽「{}」不在静态词表（谓词画像、查询规则、抽取关键词、词性词典）中；槽位已填满但没有可执行定义",
                predicate
            )
        } else {
            format!(
                "谓词槽「{}」包含未登录 token「{}」，且不在静态词表中；槽位已填满但没有可执行定义",
                predicate,
                unknown.join("」「")
            )
        };
        gaps.push(LexCoverageGap {
            text: predicate.clone(),
            span: Some(span),
            layer: LexGapLayer::Sense,
            role: LexGapRole::Predicate,
            severity: ResidualSeverity::Blocking,
            repair: RepairKind::SupplyPredicate,
            evidence,
        });
    }

    for entity in &entities {
        let Some(span) = locate_term(&chars, entity, &mut used) else {
            continue;
        };
        if inventory.is_entity(entity)
            || span_has_name_evidence(tokens, span)
            || !span_has_unknown_token(tokens, span)
        {
            // 没有静态词表命中、但 token 全部有词性或其中有人名证据时，
            // 不能离线断言世界缺口：实体可能已存在于知识库，或已由规划器
            // 的跨度恢复消费。只有未登录词形才足以构成世界缺口证据。
            known_terms += 1;
            continue;
        }
        let unknown = span_unknown_surfaces(tokens, span);
        gaps.push(LexCoverageGap {
            text: entity.clone(),
            span: Some(span),
            layer: LexGapLayer::World,
            role: LexGapRole::Entity,
            severity: ResidualSeverity::Warning,
            repair: RepairKind::SupplyEntity,
            evidence: format!(
                "实体槽「{}」包含未登录 token「{}」，且不在实体名册/姓名模式中；需要登记实体或补充事实",
                entity,
                unknown.join("」「")
            ),
        });
    }

    let mut unbound_tokens = 0u64;
    for token in tokens {
        let span = TextSpan::new(token.start, token.end);
        if (span.start..span.end).any(|index| used.get(index).copied().unwrap_or(false)) {
            continue;
        }
        if ignorable_token(&token.text) {
            continue;
        }
        unbound_tokens += 1;
        if token.source != TagSource::Unknown {
            continue;
        }
        gaps.push(LexCoverageGap {
            text: token.text.clone(),
            span: Some(span),
            layer: LexGapLayer::Form,
            role: LexGapRole::Token,
            severity: ResidualSeverity::Warning,
            repair: RepairKind::AddWord,
            evidence: format!(
                "内容 token「{}」既没有被任何计划槽位消费，也不在静态词表中；可能需要补词或调整分词",
                token.text
            ),
        });
    }

    gaps.sort_by(|a, b| {
        a.layer
            .cmp(&b.layer)
            .then(a.role.cmp(&b.role))
            .then(b.severity.cmp(&a.severity))
            .then(a.text.cmp(&b.text))
    });
    gaps.dedup_by(|a, b| {
        a.text == b.text && a.layer == b.layer && a.role == b.role && a.span == b.span
    });
    LexicalCoverage {
        version: LEXICAL_COVERAGE_VERSION,
        known_terms,
        introduced_terms,
        unbound_tokens,
        resolved,
        gaps,
    }
}
/// 带静态词表审计的读法追踪：默认行为不变，只是额外生成 [`LexicalCoverage`]
/// 并把缺口并入读法残差（仍不改变回答管线）。
pub fn trace_question_with_lexicon(
    question: &str,
    config: &SystemConfig,
    plan: Option<&QueryPlan>,
) -> LexicalTrace {
    let inventory = LexiconInventory::from_config(config);
    trace_question_with_inventory(question, &config.nlp, plan, &inventory)
}
pub fn trace_question_with_inventory(
    question: &str,
    nlp: &NlpConfig,
    plan: Option<&QueryPlan>,
    inventory: &LexiconInventory,
) -> LexicalTrace {
    let mut trace = trace_question(question, nlp, plan);
    let Some(plan) = plan else {
        return trace;
    };
    let trimmed = question.trim();
    let coverage = audit_lexical_coverage(trimmed, &trace.tokens, plan, inventory);
    for gap in &coverage.gaps {
        let (kind, stage, expected) = match (gap.layer, gap.role) {
            (LexGapLayer::Sense, LexGapRole::Predicate) => (
                ResidualKind::UnknownPredicate,
                ResidualStage::Plan,
                "谓词画像/查询规则中的可执行定义",
            ),
            (LexGapLayer::World, LexGapRole::Entity) => (
                ResidualKind::UnknownEntity,
                ResidualStage::Link,
                "已登记实体或实体名册/姓名模式",
            ),
            _ => (
                ResidualKind::UnknownToken,
                ResidualStage::Tag,
                "词表或可对齐的分词边界",
            ),
        };
        let mut residual = Residual::new(
            kind,
            stage,
            gap.severity,
            gap.repair,
            format!("{}（静态词表覆盖审计）", gap.evidence),
        )
        .expected(expected);
        if let Some(span) = gap.span {
            residual = residual.surface(trimmed, span);
        }
        trace.reading.push(residual);
    }
    trace.coverage = Some(coverage);
    trace
}
/// 语料级静态覆盖条目：按（层级、角色、词面）聚合。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LexCoverageEntry {
    pub text: String,
    pub layer: LexGapLayer,
    pub role: LexGapRole,
    pub severity: ResidualSeverity,
    pub repair: RepairKind,
    pub count: u64,
    pub first_question: usize,
    pub evidence: String,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LexCoverageReport {
    pub version: u32,
    pub questions: usize,
    pub plan_ready: u64,
    pub plan_missing: u64,
    pub unknown_tokens: u64,
    pub blocking_gaps: u64,
    #[serde(default)]
    pub reading_statuses: BTreeMap<String, u64>,
    pub entries: Vec<LexCoverageEntry>,
}
impl LexCoverageReport {
    pub fn render_report(&self) -> String {
        let mut output = format!(
            "词表覆盖审计：{} 个问句，可执行计划 {}，无计划 {}；缺口条目 {}，阻断 {} 条，未登录 token {} 个。\n",
            self.questions,
            self.plan_ready,
            self.plan_missing,
            self.entries.len(),
            self.blocking_gaps,
            self.unknown_tokens
        );
        output.push_str(&format!(
            "读法状态：{}。\n",
            render_counts(&self.reading_statuses)
        ));
        if self.entries.is_empty() {
            output.push_str("覆盖结论：没有发现计划内缺口。\n");
            return output.trim_end().to_string();
        }
        for layer in [LexGapLayer::Sense, LexGapLayer::World, LexGapLayer::Form] {
            let items: Vec<&LexCoverageEntry> = self
                .entries
                .iter()
                .filter(|entry| entry.layer == layer)
                .collect();
            if items.is_empty() {
                continue;
            }
            output.push_str(&format!("{}缺口（{}）：\n", layer.label(), items.len()));
            for entry in items {
                output.push_str(&format!(
                    "  • 「{}」/{}/{}×{}（首见第 {} 句，修复：{}）：{}\n",
                    entry.text,
                    entry.role.label(),
                    entry.severity.label(),
                    entry.count,
                    entry.first_question,
                    entry.repair.label(),
                    entry.evidence
                ));
            }
        }
        output.trim_end().to_string()
    }
}
/// 对固定语料做一次只读覆盖审计；只审计已解析计划，未解析问句单独计数。
pub fn coverage_report<'a, I>(questions: I, config: &SystemConfig) -> LexCoverageReport
where
    I: IntoIterator<Item = &'a str>,
{
    let inventory = LexiconInventory::from_config(config);
    coverage_report_with_inventory(questions, &config.nlp, &inventory)
}
pub fn coverage_report_with_inventory<'a, I>(
    questions: I,
    nlp: &NlpConfig,
    inventory: &LexiconInventory,
) -> LexCoverageReport
where
    I: IntoIterator<Item = &'a str>,
{
    let planner = QueryPlanner::new();
    let mut questions_count = 0usize;
    let mut plan_ready = 0u64;
    let mut plan_missing = 0u64;
    let mut unknown_tokens = 0u64;
    let mut blocking_gaps = 0u64;
    let mut reading_statuses: BTreeMap<String, u64> = BTreeMap::new();
    let mut entries: BTreeMap<(LexGapLayer, LexGapRole, String), LexCoverageEntry> =
        BTreeMap::new();
    for (index, question) in questions.into_iter().enumerate() {
        let question = question.trim();
        if question.is_empty() {
            continue;
        }
        questions_count += 1;
        let plan = planner.parse(question);
        let Some(plan) = plan else {
            plan_missing += 1;
            let trace = trace_question(question, nlp, None);
            unknown_tokens += trace
                .unknown_tokens()
                .filter(|token| token.text.chars().any(char::is_alphanumeric))
                .count() as u64;
            *reading_statuses
                .entry(trace.reading.status().code().to_string())
                .or_insert(0) += 1;
            continue;
        };
        plan_ready += 1;
        let trace = trace_question_with_inventory(question, nlp, Some(&plan), inventory);
        *reading_statuses
            .entry(trace.reading.status().code().to_string())
            .or_insert(0) += 1;
        unknown_tokens += trace
            .unknown_tokens()
            .filter(|token| token.text.chars().any(char::is_alphanumeric))
            .count() as u64;
        if let Some(coverage) = &trace.coverage {
            blocking_gaps += coverage.blocking_count() as u64;
            for gap in &coverage.gaps {
                let key = (gap.layer, gap.role, gap.text.clone());
                let entry = entries.entry(key).or_insert_with(|| LexCoverageEntry {
                    text: gap.text.clone(),
                    layer: gap.layer,
                    role: gap.role,
                    severity: gap.severity,
                    repair: gap.repair,
                    count: 0,
                    first_question: index + 1,
                    evidence: gap.evidence.clone(),
                });
                entry.count += 1;
            }
        }
    }
    let mut entries: Vec<LexCoverageEntry> = entries.into_values().collect();
    entries.sort_by(|a, b| {
        a.layer
            .cmp(&b.layer)
            .then(a.role.cmp(&b.role))
            .then(b.severity.cmp(&a.severity))
            .then(b.count.cmp(&a.count))
            .then(a.text.cmp(&b.text))
    });
    LexCoverageReport {
        version: LEXICAL_COVERAGE_VERSION,
        questions: questions_count,
        plan_ready,
        plan_missing,
        unknown_tokens,
        blocking_gaps,
        reading_statuses,
        entries,
    }
}
/// 词表缺口来源：未知词形，或闭包标记词落在 token 内部。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GapKind {
    UnknownToken,
    MarkerBoundary,
}
impl GapKind {
    pub fn code(self) -> &'static str {
        match self {
            GapKind::UnknownToken => "unknown_token",
            GapKind::MarkerBoundary => "marker_boundary",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            GapKind::UnknownToken => "未知词形",
            GapKind::MarkerBoundary => "标记边界",
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LexiconGapEntry {
    pub text: String,
    pub kind: GapKind,
    pub count: u64,
    /// 首次出现的问句序号（从 1 开始，便于回到语料定位）。
    pub first_question: usize,
    pub suggested_type: Option<String>,
}
/// 固定语料上的词表缺口：只提名、不自动改词表；采纳与否由人决定。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LexiconGapReport {
    pub questions: usize,
    pub unknown_tokens: u64,
    pub marker_violations: u64,
    pub plan_missing: u64,
    /// 读法状态分布（ready / partial / unreadable）。
    #[serde(default)]
    pub reading_statuses: BTreeMap<String, u64>,
    /// 残差类别分布（稳定闭集代码）。
    #[serde(default)]
    pub residual_kinds: BTreeMap<String, u64>,
    /// 修复动作分布（稳定闭集代码）。
    #[serde(default)]
    pub repair_kinds: BTreeMap<String, u64>,
    pub entries: Vec<LexiconGapEntry>,
}
impl LexiconGapReport {
    pub fn render_report(&self) -> String {
        let mut output = format!(
            "词表缺口报告：{} 个问句，{} 个不同词形/边界项；未知词 {} 次，标记越界 {} 次，无计划 {} 次。\n",
            self.questions,
            self.entries.len(),
            self.unknown_tokens,
            self.marker_violations,
            self.plan_missing
        );
        output.push_str(&format!(
            "读法状态：{}；残差：{}；修复：{}。\n",
            render_counts(&self.reading_statuses),
            render_counts(&self.residual_kinds),
            render_counts(&self.repair_kinds)
        ));
        if self.entries.is_empty() {
            output.push_str("没有发现词表缺口。\n");
            return output.trim_end().to_string();
        }
        for kind in [GapKind::UnknownToken, GapKind::MarkerBoundary] {
            let items: Vec<&LexiconGapEntry> = self
                .entries
                .iter()
                .filter(|entry| entry.kind == kind)
                .collect();
            if items.is_empty() {
                continue;
            }
            output.push_str(&format!("{}（{}）：\n", kind.label(), items.len()));
            for entry in items {
                let suggested = entry
                    .suggested_type
                    .as_deref()
                    .map(|etype| format!("；可能类型 {}", etype))
                    .unwrap_or_default();
                output.push_str(&format!(
                    "  • 「{}」×{}（首见第 {} 句{}）\n",
                    entry.text, entry.count, entry.first_question, suggested
                ));
            }
        }
        output.trim_end().to_string()
    }
}
/// 对固定语料做一次只读词表缺口扫描。同一问句列表必然得到同一排序：
/// 先按缺口类别，再按次数降序，最后按词形字典序。
pub fn lexicon_gap_report<'a, I>(questions: I, config: &NlpConfig) -> LexiconGapReport
where
    I: IntoIterator<Item = &'a str>,
{
    let planner = QueryPlanner::new();
    let mut entries: BTreeMap<(GapKind, String), LexiconGapEntry> = BTreeMap::new();
    let mut total_questions = 0usize;
    let mut unknown_tokens = 0u64;
    let mut marker_violations = 0u64;
    let mut plan_missing = 0u64;
    let mut reading_statuses: BTreeMap<String, u64> = BTreeMap::new();
    let mut residual_kinds: BTreeMap<String, u64> = BTreeMap::new();
    let mut repair_kinds: BTreeMap<String, u64> = BTreeMap::new();
    let mut record = |kind: GapKind, text: &str, question_index: usize| {
        let key = (kind, text.to_string());
        let entry = entries.entry(key).or_insert_with(|| LexiconGapEntry {
            text: text.to_string(),
            kind,
            count: 0,
            first_question: question_index,
            suggested_type: crate::identity::infer_suffix_type(text)
                .map(|(_, etype)| etype.to_string()),
        });
        entry.count += 1;
    };
    for (index, question) in questions.into_iter().enumerate() {
        let question = question.trim();
        if question.is_empty() {
            continue;
        }
        total_questions += 1;
        let plan = planner.parse(question);
        if plan.is_none() {
            plan_missing += 1;
        }
        let trace = trace_question(question, config, plan.as_ref());
        *reading_statuses
            .entry(trace.reading.status().code().to_string())
            .or_insert(0) += 1;
        for kind in trace.reading.residual_kinds() {
            *residual_kinds.entry(kind).or_insert(0) += 1;
        }
        for repair in trace.reading.repair_kinds() {
            *repair_kinds.entry(repair).or_insert(0) += 1;
        }
        for token in trace.unknown_tokens() {
            if token.text.chars().any(|ch| ch.is_alphanumeric()) {
                unknown_tokens += 1;
                record(GapKind::UnknownToken, &token.text, index + 1);
            }
        }
        for marker in trace
            .markers
            .iter()
            .filter(|marker| marker.boundary_violation)
        {
            marker_violations += 1;
            let text = marker
                .token_text
                .clone()
                .unwrap_or_else(|| marker.marker.clone());
            record(GapKind::MarkerBoundary, &text, index + 1);
        }
    }
    let mut entries: Vec<LexiconGapEntry> = entries.into_values().collect();
    entries.sort_by(|a, b| {
        a.kind
            .cmp(&b.kind)
            .then(b.count.cmp(&a.count))
            .then(a.text.cmp(&b.text))
    });
    LexiconGapReport {
        questions: total_questions,
        unknown_tokens,
        marker_violations,
        plan_missing,
        reading_statuses,
        residual_kinds,
        repair_kinds,
        entries,
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::nlp::TagSource;

    fn config() -> NlpConfig {
        NlpConfig {
            import_keywords: Vec::new(),
            pos_dict: [
                ("的".to_string(), "u".to_string()),
                ("所有权".to_string(), "n".to_string()),
                ("老板".to_string(), "n".to_string()),
            ]
            .into_iter()
            .collect(),
            name_patterns: crate::NamePatterns {
                prefixes: vec!["小".to_string()],
                suffixes: Vec::new(),
                full_names: vec!["张三".to_string()],
                profession_suffix: Vec::new(),
            },
            enable_nlu: None,
            nlu_confidence_threshold: None,
            enable_local_semantic: None,
            disable_ownthink_api: None,
            semantic_similarity_threshold: None,
            max_entity_candidates: None,
            enable_advanced_entity_extraction: None,
            enable_relation_extraction: None,
            context_window_size: None,
        }
    }

    #[test]
    fn trace_records_token_sources_and_spans_deterministically() {
        let config = config();
        let first = trace_question("张三的所有权归谁", &config, None);
        let second = trace_question("张三的所有权归谁", &config, None);
        assert_eq!(first, second);
        assert_eq!(first.fidelity, TraceFidelity::Reconstructed);
        assert_eq!(first.tokens.len(), second.tokens.len());
        assert!(first
            .tokens
            .iter()
            .any(|token| token.text == "所有权" && token.source == TagSource::ConfigPos));
        assert_eq!(&first.lexicon_digest, &second.lexicon_digest);
        assert_eq!(first.failures[0].kind, ReadFailureKind::NoQuestionShape);
        assert!(first
            .warnings
            .iter()
            .any(|warning| warning.kind == ReadWarningKind::ClosureMarkerInToken));
    }

    #[test]
    fn marker_boundary_violation_is_reported_without_changing_plan() {
        let config = config();
        let plan = QueryPlanner::new()
            .parse("张三的所有权归公司")
            .expect("当前规划器按字符子串解析");
        let trace = trace_question("张三的所有权归公司", &config, Some(&plan));
        assert!(matches!(plan, QueryPlan::Closure { .. }));
        let marker = trace
            .markers
            .iter()
            .find(|marker| marker.marker == "所有")
            .expect("应记录子串标记");
        assert!(marker.boundary_violation);
        assert!(marker.plan_consumed);
        assert_eq!(marker.token_text.as_deref(), Some("所有权"));
        let receipt = trace.receipt();
        assert_eq!(receipt.closure_marker_violations, 1);
        assert_eq!(receipt.plan_kind.as_deref(), Some("Closure"));
    }

    #[test]
    fn unplanned_question_reports_unknown_token_with_span() {
        let config = config();
        let trace = trace_question("所有权归谁", &config, None);
        let residual = trace
            .reading
            .residuals
            .iter()
            .find(|residual| residual.kind == ResidualKind::UnknownToken)
            .expect("无计划时应定位未登录词");
        assert_eq!(residual.stage, ResidualStage::Tag);
        assert_eq!(residual.span, Some(TextSpan::new(3, 4)));
        assert_eq!(residual.surface.as_deref(), Some("归"));
        assert_eq!(residual.severity, ResidualSeverity::Warning);
        assert_eq!(residual.repair, RepairKind::RewriteQuestion);
    }

    #[test]
    fn token_aligned_closure_has_no_boundary_warning() {
        let config = config();
        let plan = QueryPlanner::new()
            .parse("张三的所有老板是谁")
            .expect("闭包问句应被解析");
        let trace = trace_question("张三的所有老板是谁", &config, Some(&plan));
        assert!(trace.boundary_violations() == 0, "{:?}", trace.markers);
        assert!(trace.failures.is_empty());
    }

    #[test]
    fn ledger_is_bounded_and_keeps_only_stable_receipts() {
        let config = config();
        let ledger = ReadTraceLedger::with_capacity(2);
        for question in ["张三的所有权归谁", "所有权归谁", "谁"] {
            let trace = trace_question(question, &config, None);
            ledger.record(&trace);
        }
        assert_eq!(ledger.len(), 2);
        let report = ledger.report();
        assert!(report.contains("读法台账：2 条有界回执"));
        assert!(!report.contains("张三"), "台账不得保存原始问句：{}", report);
        let snapshot = ledger.snapshot();
        assert_eq!(snapshot.len(), 2);
        assert!(snapshot
            .iter()
            .all(|receipt| !receipt.question_digest.is_empty()));
    }

    #[test]
    fn plan_terms_do_not_treat_literal_objects_as_entities() {
        let config = config();
        let plan = QueryPlanner::new().parse("张三的老板是谁").unwrap();
        let trace = trace_question("张三的老板是谁", &config, Some(&plan));
        let plan = trace.plan.as_ref().unwrap();
        assert_eq!(plan.entities, vec!["张三".to_string()]);
        assert_eq!(plan.predicates, vec!["老板".to_string()]);
    }

    #[test]
    fn lexicon_gap_report_groups_unknown_tokens_and_boundary_risks() {
        let config = config();
        let questions = ["张三的所有权归谁", "李四的所有权归公司", "所有权归谁", ""];
        let first = lexicon_gap_report(questions, &config);
        let second = lexicon_gap_report(questions, &config);
        assert_eq!(first, second, "同一语料必须得到同一缺口报告");
        assert_eq!(first.questions, 3);
        assert!(first.unknown_tokens > 0);
        assert!(first.marker_violations > 0);
        let ownership = first
            .entries
            .iter()
            .find(|entry| entry.text == "所有权" && entry.kind == GapKind::MarkerBoundary)
            .expect("应提名包含标记词的 token");
        assert_eq!(ownership.first_question, 1);
        let rendered = first.render_report();
        assert!(rendered.contains("词表缺口报告"));
        assert!(rendered.contains("标记边界"));
        assert!(rendered.contains("读法状态"));
        assert!(first.residual_kinds.contains_key("token_boundary"));
        assert!(first.repair_kinds.contains_key("split_token"));
    }

    fn system_config() -> SystemConfig {
        toml::from_str(include_str!("../../config.toml")).expect("config.toml 必须可解析")
    }

    #[test]
    fn lexicon_audit_flags_silent_oov_predicate_without_changing_default_trace() {
        let config = system_config();
        let question = "姚明的曾祖父是谁？";
        let plan = QueryPlanner::new()
            .parse(question)
            .expect("成功计划是沉默缺口的前提");
        let default = trace_question(question, &config.nlp, Some(&plan));
        assert!(default.coverage.is_none(), "默认追踪不应生成覆盖审计");
        assert!(default.reading.is_ready(), "默认读法保持可执行");

        let trace = trace_question_with_lexicon(question, &config, Some(&plan));
        let coverage = trace.coverage.as_ref().expect("应生成覆盖审计");
        let gap = coverage
            .gaps
            .iter()
            .find(|gap| gap.text == "曾祖父")
            .expect("曾祖父未被词表解释");
        assert_eq!(gap.layer, LexGapLayer::Sense);
        assert_eq!(gap.role, LexGapRole::Predicate);
        assert_eq!(gap.severity, ResidualSeverity::Blocking);
        assert!(gap.span.is_some());
        assert_eq!(coverage.blocking_count(), 1);
        assert!(trace.reading.blocking_count() >= 1);
        assert!(trace.render_report().contains("词表覆盖审计"));
    }

    #[test]
    fn lexicon_audit_accepts_a_configured_predicate() {
        let config = system_config();
        let plan = QueryPlanner::new()
            .parse("姚明的配偶是谁")
            .expect("配偶问句应可解析");
        let trace = trace_question_with_lexicon("姚明的配偶是谁", &config, Some(&plan));
        let coverage = trace.coverage.as_ref().unwrap();
        assert!(
            coverage.gaps.is_empty(),
            "配偶已在谓词画像中：{:?}",
            coverage.gaps
        );
        assert!(coverage.known_terms >= 1);
    }

    #[test]
    fn admitted_alias_resolves_a_silent_predicate_gap() {
        let config = system_config();
        let mut inventory = LexiconInventory::from_config(&config);
        let admission = crate::lex_gate::admit_json(
            r#"[{"surface": "先生", "kind": "alias", "canonical": "配偶"}]"#,
            &inventory,
        )
        .expect("定义文件应可解析");
        assert_eq!(admission.accepted.len(), 1, "{:?}", admission.rejected);
        admission.apply(&mut inventory);

        let question = "叶莉的先生的身高是多少？";
        let plan = QueryPlanner::new().parse(question).unwrap();
        let trace =
            trace_question_with_inventory(question, &config.nlp, Some(&plan), &inventory);
        let coverage = trace.coverage.as_ref().unwrap();
        assert!(
            !coverage.gaps.iter().any(|gap| gap.text == "先生"),
            "已准入的定义应解析掉先生：{:?}",
            coverage.gaps
        );
        let resolved = coverage
            .resolved
            .iter()
            .find(|term| term.text == "先生")
            .expect("应记录定义解析证据");
        assert_eq!(resolved.expansion, vec!["配偶".to_string()]);
        assert!(coverage.blocking_count() == 0);
        assert!(trace.render_report().contains("先生 → 配偶"));
    }

    #[test]
    fn lexicon_audit_separates_world_gap_from_sense_gap() {
        let config = system_config();
        let question = "未名科技公司的 CEO 是谁？";
        let plan = QueryPlanner::new().parse(question).unwrap();
        let coverage = trace_question_with_lexicon(question, &config, Some(&plan))
            .coverage
            .unwrap();
        let entity = coverage
            .gaps
            .iter()
            .find(|gap| gap.text == "未名科技公司")
            .expect("未登记实体是独立的世界缺口");
        let predicate = coverage
            .gaps
            .iter()
            .find(|gap| gap.text == "CEO")
            .expect("CEO 没有可执行定义");
        assert_eq!(entity.layer, LexGapLayer::World);
        assert_eq!(entity.role, LexGapRole::Entity);
        assert_eq!(entity.severity, ResidualSeverity::Warning);
        assert_eq!(predicate.layer, LexGapLayer::Sense);
        assert_eq!(predicate.role, LexGapRole::Predicate);
        assert_eq!(predicate.severity, ResidualSeverity::Blocking);
        assert_eq!(coverage.blocking_count(), 1);
    }

    #[test]
    fn lexicon_audit_does_not_report_recovered_name_fragment() {
        let config = system_config();
        let question = "姚明和叶莉是什么关系？";
        let plan = QueryPlanner::new().parse(question).unwrap();
        let coverage = trace_question_with_lexicon(question, &config, Some(&plan))
            .coverage
            .unwrap();
        assert!(
            coverage.gaps.is_empty(),
            "已由实体跨度恢复消费的人名碎片不应报世界缺口：{:?}",
            coverage.gaps
        );
    }

    #[test]
    fn coverage_report_aggregates_layers_and_is_deterministic() {
        let config = system_config();
        let questions = [
            "姚明的曾祖父是谁？",
            "姚明的曾祖父是谁？",
            "未名科技公司的 CEO 是谁？",
            "姚明的配偶是谁",
            "",
        ];
        let first = coverage_report(questions, &config);
        let second = coverage_report(questions, &config);
        assert_eq!(first, second, "同一语料必须得到同一覆盖报告");
        assert_eq!(first.questions, 4);
        assert_eq!(first.plan_ready, 4);
        assert_eq!(first.plan_missing, 0);
        assert!(first.blocking_gaps >= 3);
        let ancestor = first
            .entries
            .iter()
            .find(|entry| entry.text == "曾祖父")
            .expect("聚合应保留首次出现序号");
        assert_eq!(ancestor.count, 2);
        assert_eq!(ancestor.first_question, 1);
        assert_eq!(ancestor.layer, LexGapLayer::Sense);
        assert!(first.entries.iter().any(|entry| {
            entry.text == "未名科技公司" && entry.layer == LexGapLayer::World
        }));
        let rendered = first.render_report();
        assert!(rendered.contains("词表覆盖审计"));
        assert!(rendered.contains("词义缺口"));
        assert!(rendered.contains("世界缺口"));
    }
}

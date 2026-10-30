use std::collections::{BTreeMap, BTreeSet, VecDeque};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use crate::canonical;
use crate::nlp::{preprocess_annotated, TagSource};
use crate::reading::{self, Reading, RepairKind, Residual, ResidualKind, ResidualSeverity, ResidualStage, TextSpan};
use crate::query_planner::{QueryPlan, QueryPlanner};
use crate::{NlpConfig, SystemConfig};
pub const LEXICAL_TRACE_VERSION: u32 = 1;
pub const DEFAULT_FAILURE_TRACE_CAPACITY: usize = 32;
/// 闭包/精确跳数标记词：出现位置与计划解读不一致时，轨迹必须能指出来。
const CLOSURE_MARKERS: [&str; 10] = [
    "的任何",
    "的一切",
    "的所有",
    "的全部",
    "的任意",
    "所有",
    "全部",
    "恰好",
    "正好",
    "刚好",
];
/// 轨迹保真度。第一版只做「事后按同一纯函数重跑」，不冒充当时的执行决策。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceFidelity {
    /// 事后重跑纯函数得到的（默认；不是当时的决策现场）。
    Reconstructed,
    /// 在真实阶段边界记录的。
    Instrumented,
    /// 记录分支/候选/拒绝理由的决策轨迹。
    Decision,
}
impl TraceFidelity {
    pub fn label(self) -> &'static str {
        match self {
            TraceFidelity::Reconstructed => "事后重建",
            TraceFidelity::Instrumented => "阶段实测",
            TraceFidelity::Decision => "决策现场",
        }
    }
}
/// 读过程的阶段：失败必须能落到一个阶段，而不是折叠成「不知道」。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadStage {
    Input,
    Tokenize,
    Tag,
    Normalize,
    Plan,
    Link,
    Execute,
    Generate,
    Ground,
}
impl ReadStage {
    pub fn code(self) -> &'static str {
        match self {
            ReadStage::Input => "input",
            ReadStage::Tokenize => "tokenize",
            ReadStage::Tag => "tag",
            ReadStage::Normalize => "normalize",
            ReadStage::Plan => "plan",
            ReadStage::Link => "link",
            ReadStage::Execute => "execute",
            ReadStage::Generate => "generate",
            ReadStage::Ground => "ground",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            ReadStage::Input => "输入",
            ReadStage::Tokenize => "切词",
            ReadStage::Tag => "词性",
            ReadStage::Normalize => "归一",
            ReadStage::Plan => "规划",
            ReadStage::Link => "链接",
            ReadStage::Execute => "执行",
            ReadStage::Generate => "生成",
            ReadStage::Ground => "接地",
        }
    }
}
/// 读失败的分类。知识失败与读失败分开，回答策略才能分开。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadFailureKind {
    EmptyInput,
    InputTooLong,
    NoQuestionShape,
    UnknownEntity,
    UnknownPredicate,
    ClosureMarkerInToken,
    ExecutionMiss,
    FactGap,
    MissingLaw,
    Truncated,
    GenerationUngrounded,
    Unclassified,
}
impl ReadFailureKind {
    pub fn code(self) -> &'static str {
        match self {
            ReadFailureKind::EmptyInput => "empty_input",
            ReadFailureKind::InputTooLong => "input_too_long",
            ReadFailureKind::NoQuestionShape => "no_question_shape",
            ReadFailureKind::UnknownEntity => "unknown_entity",
            ReadFailureKind::UnknownPredicate => "unknown_predicate",
            ReadFailureKind::ClosureMarkerInToken => "closure_marker_in_token",
            ReadFailureKind::ExecutionMiss => "execution_miss",
            ReadFailureKind::FactGap => "fact_gap",
            ReadFailureKind::MissingLaw => "missing_law",
            ReadFailureKind::Truncated => "truncated",
            ReadFailureKind::GenerationUngrounded => "generation_ungrounded",
            ReadFailureKind::Unclassified => "unclassified",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            ReadFailureKind::EmptyInput => "空输入",
            ReadFailureKind::InputTooLong => "输入过长",
            ReadFailureKind::NoQuestionShape => "没有问句形状",
            ReadFailureKind::UnknownEntity => "未知实体",
            ReadFailureKind::UnknownPredicate => "未知谓词",
            ReadFailureKind::ClosureMarkerInToken => "标记词落在 token 内部",
            ReadFailureKind::ExecutionMiss => "计划已知但执行未命中",
            ReadFailureKind::FactGap => "事实缺口",
            ReadFailureKind::MissingLaw => "缺少推理律",
            ReadFailureKind::Truncated => "搜索被截断",
            ReadFailureKind::GenerationUngrounded => "生成不接地",
            ReadFailureKind::Unclassified => "未分类",
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadWarningKind {
    ClosureMarkerInToken,
    UnknownToken,
}
impl ReadWarningKind {
    pub fn code(self) -> &'static str {
        match self {
            ReadWarningKind::ClosureMarkerInToken => "closure_marker_in_token",
            ReadWarningKind::UnknownToken => "unknown_token",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            ReadWarningKind::ClosureMarkerInToken => "标记词边界",
            ReadWarningKind::UnknownToken => "未知词",
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceToken {
    pub text: String,
    pub pos: String,
    pub source: TagSource,
    pub start: usize,
    pub end: usize,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PosDivergence {
    pub token_index: usize,
    pub text: String,
    pub chosen: String,
    pub source: TagSource,
    pub jieba: String,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MarkerTrace {
    pub marker: String,
    pub start: usize,
    pub end: usize,
    pub token_index: Option<usize>,
    pub token_text: Option<String>,
    /// 标记词没有从 token 边界开始或结束（例如「所有权」内部的「所有」）。
    pub boundary_violation: bool,
    /// 当前计划确实是闭包计划，且该标记词属于闭包触发词族。
    pub plan_consumed: bool,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OovTrace {
    pub text: String,
    pub start: usize,
    pub kind: String,
    pub reason: String,
    pub suggested_type: Option<String>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReadFailure {
    pub stage: ReadStage,
    pub kind: ReadFailureKind,
    pub detail: String,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReadWarning {
    pub kind: ReadWarningKind,
    pub detail: String,
}
/// 查询计划投影：类型、人类可读形，以及用于链接检查的实体/谓词槽。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanTrace {
    pub kind: String,
    pub display: String,
    pub entities: Vec<String>,
    pub predicates: Vec<String>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LexicalTrace {
    pub version: u32,
    pub question_digest: String,
    pub question_chars: usize,
    pub lexicon_digest: String,
    pub fidelity: TraceFidelity,
    pub tokens: Vec<TraceToken>,
    pub divergences: Vec<PosDivergence>,
    pub plan: Option<PlanTrace>,
    pub markers: Vec<MarkerTrace>,
    pub oov: Vec<OovTrace>,
    pub failures: Vec<ReadFailure>,
    pub warnings: Vec<ReadWarning>,
    /// 读法残差表：带 span/候选/修复动作的部分理解；与 failures 并存，
    /// failures 保持旧回执兼容，reading 供新诊断与修复计划使用。
    #[serde(default)]
    pub reading: Reading,
    /// 静态词表覆盖审计（仅在显式请求 `--lexicon` 时生成）：把「计划可执行
    /// 但谓词/实体没有定义」的沉默缺口显式化。默认追踪为 None，保持旧行为。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coverage: Option<LexicalCoverage>,
    /// 宿主是否观察到回答；纯文本追踪（未访问知识库）为 None。
    pub answer_available: Option<bool>,
}
impl LexicalTrace {
    /// 未知词 token（TagSource::Unknown）。
    pub fn unknown_tokens(&self) -> impl Iterator<Item = &TraceToken> {
        self.tokens
            .iter()
            .filter(|token| token.source == TagSource::Unknown)
    }

    /// 边界越界的 token 数（同一 token 内的兄弟标记词不重复计数）。
    pub fn boundary_violations(&self) -> usize {
        let mut seen = std::collections::BTreeSet::new();
        for marker in self
            .markers
            .iter()
            .filter(|marker| marker.boundary_violation)
        {
            let key = marker
                .token_index
                .unwrap_or_else(|| 1_000_000 + marker.start);
            seen.insert(key);
        }
        seen.len()
    }

    /// 追加失败分类；同一（阶段、类别、详情）只保留一次，避免重复扫描膨胀。
    pub fn push_failure(
        &mut self,
        stage: ReadStage,
        kind: ReadFailureKind,
        detail: impl Into<String>,
    ) {
        let detail = detail.into();
        if self.failures.iter().any(|failure| {
            failure.stage == stage && failure.kind == kind && failure.detail == detail
        }) {
            return;
        }
        self.failures.push(ReadFailure {
            stage,
            kind,
            detail,
        });
    }

    /// 稳定回执：不含原始问句与 token 文本，可进入有界台账/指标。
    pub fn receipt(&self) -> TraceReceipt {
        let mut source_counts: BTreeMap<String, u64> = BTreeMap::new();
        for token in &self.tokens {
            *source_counts
                .entry(token.source.code().to_string())
                .or_insert(0) += 1;
        }
        let mut failure_kinds: Vec<String> = self
            .failures
            .iter()
            .map(|failure| failure.kind.code().to_string())
            .collect();
        failure_kinds.sort();
        failure_kinds.dedup();
        let mut failure_stages: Vec<String> = self
            .failures
            .iter()
            .map(|failure| failure.stage.code().to_string())
            .collect();
        failure_stages.sort();
        failure_stages.dedup();
        let mut warning_kinds: Vec<String> = self
            .warnings
            .iter()
            .map(|warning| warning.kind.code().to_string())
            .collect();
        warning_kinds.sort();
        warning_kinds.dedup();
        TraceReceipt {
            version: self.version,
            question_digest: self.question_digest.clone(),
            question_chars: self.question_chars,
            token_count: self.tokens.len(),
            unknown_tokens: self.unknown_tokens().count() as u64,
            source_counts,
            plan_kind: self.plan.as_ref().map(|plan| plan.kind.clone()),
            failure_kinds,
            failure_stages,
            warning_kinds,
            closure_marker_violations: self.boundary_violations() as u64,
            oov_count: self.oov.len() as u64,
            reading_status: self.reading.status().code().to_string(),
            residual_kinds: self.reading.residual_kinds(),
            residual_stages: self.reading.residual_stages(),
            blocking_residuals: self.reading.blocking_count() as u64,
            repair_kinds: self.reading.repair_kinds(),
            coverage_gaps: self.coverage.as_ref().map_or(0, |coverage| coverage.gaps.len() as u64),
            coverage_blocking: self
                .coverage
                .as_ref()
                .map_or(0, |coverage| coverage.blocking_count() as u64),
            fidelity: self.fidelity,
            answered: self.answer_available,
        }
    }

    /// 确定性文本视图（可 golden）。原始 token 只在此本地渲染。
    pub fn render_report(&self) -> String {
        let mut output = String::new();
        output.push_str(&format!(
            "读法追踪：{} 字，{} 个 token，词表指纹 {}。\n",
            self.question_chars,
            self.tokens.len(),
            self.lexicon_digest
        ));
        output.push_str(&format!(
            "证据保真度：{}（事后按同一纯函数重跑，不是当时的执行现场）。\n",
            self.fidelity.label()
        ));
        if let Some(answered) = self.answer_available {
            output.push_str(&format!(
                "回答管线：{}。\n",
                if answered { "有结果" } else { "未命中" }
            ));
        }

        if self.tokens.is_empty() {
            output.push_str("token：无。\n");
        } else {
            output.push_str(&format!("token（{}）：\n", self.tokens.len()));
            for (index, token) in self.tokens.iter().enumerate() {
                output.push_str(&format!(
                    "  {}. 「{}」{}/{} {}..{}\n",
                    index + 1,
                    token.text,
                    token.pos,
                    token.source.code(),
                    token.start,
                    token.end
                ));
            }
        }

        if !self.divergences.is_empty() {
            output.push_str(&format!("词性分歧（{}）：\n", self.divergences.len()));
            for divergence in &self.divergences {
                output.push_str(&format!(
                    "  • 「{}」取 {}（{}），结巴原标签是 {}。\n",
                    divergence.text,
                    divergence.chosen,
                    divergence.source.code(),
                    divergence.jieba
                ));
            }
        }

        match &self.plan {
            Some(plan) => {
                output.push_str(&format!("查询计划：{}（{}）。\n", plan.kind, plan.display));
                if !plan.entities.is_empty() {
                    output.push_str(&format!("  实体槽：{}\n", plan.entities.join("、")));
                }
                if !plan.predicates.is_empty() {
                    output.push_str(&format!("  谓词槽：{}\n", plan.predicates.join("、")));
                }
            }
            None => output.push_str("查询计划：无（没有可执行的问句形状）。\n"),
        }

        if !self.markers.is_empty() {
            output.push_str(&format!("闭包/跳数标记（{}）：\n", self.markers.len()));
            for marker in &self.markers {
                let mut line = format!("  • 「{}」{}..{}", marker.marker, marker.start, marker.end);
                if let Some(token) = &marker.token_text {
                    line.push_str(&format!("，位于 token「{}」", token));
                } else {
                    line.push_str("，不在任何 token 内");
                }
                if marker.boundary_violation {
                    line.push_str(" ← 边界越界");
                }
                if marker.plan_consumed {
                    line.push_str("；当前闭包计划使用了该词簇");
                }
                line.push('\n');
                output.push_str(&line);
            }
        }

        if !self.oov.is_empty() {
            output.push_str(&format!("OOV 提名（{}，只报告不落库）：\n", self.oov.len()));
            for suspect in &self.oov {
                let suggested = suspect
                    .suggested_type
                    .as_deref()
                    .map(|etype| format!("；暂定类型 {}", etype))
                    .unwrap_or_default();
                output.push_str(&format!(
                    "  • 「{}」{}..{}（{}{}）\n",
                    suspect.text, suspect.start, suspect.reason, suspect.kind, suggested
                ));
            }
        }

        for line in self.reading.render_lines() {
            output.push_str(&line);
            output.push('\n');
        }

        if let Some(coverage) = &self.coverage {
            for line in coverage.render_lines() {
                output.push_str(&line);
                output.push('\n');
            }
        }

        if self.failures.is_empty() {
            output.push_str("失败分类：无（本轨迹只做观测）。\n");
        } else {
            output.push_str(&format!("失败分类（{}）：\n", self.failures.len()));
            for failure in &self.failures {
                output.push_str(&format!(
                    "  • {} / {}：{}\n",
                    failure.stage.label(),
                    failure.kind.label(),
                    failure.detail
                ));
            }
        }

        if !self.warnings.is_empty() {
            output.push_str(&format!("风险提示（{}）：\n", self.warnings.len()));
            for warning in &self.warnings {
                output.push_str(&format!(
                    "  • {}：{}\n",
                    warning.kind.label(),
                    warning.detail
                ));
            }
        }

        output.trim_end().to_string()
    }
}
/// 有界读法回执：只保留稳定字段，不保存原始问句与 token 文本。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceReceipt {
    pub version: u32,
    pub question_digest: String,
    pub question_chars: usize,
    pub token_count: usize,
    pub unknown_tokens: u64,
    pub source_counts: BTreeMap<String, u64>,
    pub plan_kind: Option<String>,
    pub failure_kinds: Vec<String>,
    pub failure_stages: Vec<String>,
    pub warning_kinds: Vec<String>,
    pub closure_marker_violations: u64,
    pub oov_count: u64,
    /// 读法状态：ready / partial / unreadable（旧快照缺省为空）。
    #[serde(default)]
    pub reading_status: String,
    /// 残差类别（稳定闭集代码，已排序去重）。
    #[serde(default)]
    pub residual_kinds: Vec<String>,
    /// 残差阶段（稳定闭集代码，已排序去重）。
    #[serde(default)]
    pub residual_stages: Vec<String>,
    /// 阻断性残差条数。
    #[serde(default)]
    pub blocking_residuals: u64,
    /// 可执行的修复动作（稳定闭集代码，已排序去重）。
    #[serde(default)]
    pub repair_kinds: Vec<String>,
    /// 静态词表覆盖缺口条数（仅在显式审计时非零）。
    #[serde(default)]
    pub coverage_gaps: u64,
    /// 静态词表覆盖阻断缺口条数。
    #[serde(default)]
    pub coverage_blocking: u64,
    pub fidelity: TraceFidelity,
    pub answered: Option<bool>,
}
#[derive(Debug)]
pub struct ReadTraceLedger {
    capacity: usize,
    receipts: Mutex<VecDeque<TraceReceipt>>,
}
impl Default for ReadTraceLedger {
    fn default() -> Self {
        Self::with_capacity(DEFAULT_FAILURE_TRACE_CAPACITY)
    }
}
impl ReadTraceLedger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            receipts: Mutex::new(VecDeque::new()),
        }
    }

    pub fn record(&self, trace: &LexicalTrace) -> TraceReceipt {
        let receipt = trace.receipt();
        let mut receipts = self.receipts.lock();
        while receipts.len() >= self.capacity {
            receipts.pop_front();
        }
        receipts.push_back(receipt.clone());
        receipt
    }

    pub fn clear(&self) {
        self.receipts.lock().clear();
    }

    pub fn len(&self) -> usize {
        self.receipts.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn snapshot(&self) -> Vec<TraceReceipt> {
        self.receipts.lock().iter().cloned().collect()
    }

    /// 聚合视图：按阶段不可直接得到（回执只保留类别），因此按类别/来源/计划
    /// 聚合；失败阶段在单条轨迹里可读，聚合只做有限枚举计数。
    pub fn report(&self) -> String {
        let receipts = self.snapshot();
        if receipts.is_empty() {
            return "读法台账：暂无失败轨迹。".to_string();
        }
        let mut failures: BTreeMap<String, u64> = BTreeMap::new();
        let mut stages: BTreeMap<String, u64> = BTreeMap::new();
        let mut plans: BTreeMap<String, u64> = BTreeMap::new();
        let mut sources: BTreeMap<String, u64> = BTreeMap::new();
        let mut statuses: BTreeMap<String, u64> = BTreeMap::new();
        let mut residual_kinds: BTreeMap<String, u64> = BTreeMap::new();
        let mut repairs: BTreeMap<String, u64> = BTreeMap::new();
        let mut unknown_tokens = 0u64;
        let mut marker_violations = 0u64;
        let mut oov = 0u64;
        let mut blocking_residuals = 0u64;
        let mut coverage_gaps = 0u64;
        let mut coverage_blocking = 0u64;
        for receipt in &receipts {
            for kind in &receipt.failure_kinds {
                *failures.entry(kind.clone()).or_insert(0) += 1;
            }
            for stage in &receipt.failure_stages {
                *stages.entry(stage.clone()).or_insert(0) += 1;
            }
            let plan = receipt
                .plan_kind
                .clone()
                .unwrap_or_else(|| "无计划".to_string());
            *plans.entry(plan).or_insert(0) += 1;
            for (source, count) in &receipt.source_counts {
                *sources.entry(source.clone()).or_insert(0) += count;
            }
            *statuses
                .entry(if receipt.reading_status.is_empty() {
                    "unknown".to_string()
                } else {
                    receipt.reading_status.clone()
                })
                .or_insert(0) += 1;
            for kind in &receipt.residual_kinds {
                *residual_kinds.entry(kind.clone()).or_insert(0) += 1;
            }
            for repair in &receipt.repair_kinds {
                *repairs.entry(repair.clone()).or_insert(0) += 1;
            }
            unknown_tokens += receipt.unknown_tokens;
            marker_violations += receipt.closure_marker_violations;
            oov += receipt.oov_count;
            blocking_residuals += receipt.blocking_residuals;
            coverage_gaps += receipt.coverage_gaps;
            coverage_blocking += receipt.coverage_blocking;
        }
        let mut output = format!(
            "读法台账：{} 条有界回执（容量 {}，不保存原文）。\n",
            receipts.len(),
            self.capacity
        );
        output.push_str("失败阶段（条次）：");
        output.push_str(&render_counts(&stages));
        output.push('\n');
        output.push_str("失败分类（条次）：");
        output.push_str(&render_counts(&failures));
        output.push('\n');
        output.push_str("计划分布（条次）：");
        output.push_str(&render_counts(&plans));
        output.push('\n');
        output.push_str("读法状态（条次）：");
        output.push_str(&render_counts(&statuses));
        output.push('\n');
        output.push_str("残差类别（条次）：");
        output.push_str(&render_counts(&residual_kinds));
        output.push('\n');
        output.push_str("修复动作（条次）：");
        output.push_str(&render_counts(&repairs));
        output.push('\n');
        output.push_str("词性来源（token 数）：");
        output.push_str(&render_counts(&sources));
        output.push('\n');
        output.push_str(&format!(
            "未知词：{}；标记边界越界：{}；OOV 提名：{}；阻断残差：{}。\n",
            unknown_tokens, marker_violations, oov, blocking_residuals
        ));
        if coverage_gaps > 0 {
            output.push_str(&format!(
                "静态覆盖缺口：{} 条（阻断 {} 条）。\n",
                coverage_gaps, coverage_blocking
            ));
        }
        if let Some(last) = receipts.last() {
            output.push_str(&format!(
                "最近一条：digest={}，计划={}，失败={}。\n",
                last.question_digest,
                last.plan_kind.as_deref().unwrap_or("无"),
                if last.failure_kinds.is_empty() {
                    "无".to_string()
                } else {
                    last.failure_kinds.join("、")
                }
            ));
        }
        output.trim_end().to_string()
    }
}
fn render_counts(counts: &BTreeMap<String, u64>) -> String {
    if counts.is_empty() {
        return "无".to_string();
    }
    counts
        .iter()
        .map(|(key, value)| format!("{}={}", key, value))
        .collect::<Vec<_>>()
        .join("，")
}
/// 词表指纹：同一配置必得同一摘要；不依赖 HashMap 迭代序。
pub fn lexicon_digest(config: &NlpConfig) -> String {
    #[derive(Serialize)]
    struct Lexicon<'a> {
        tokenizer: &'static str,
        pos: BTreeMap<&'a str, &'a str>,
        name_prefixes: &'a [String],
        name_suffixes: &'a [String],
        full_names: &'a [String],
        profession_suffix: &'a [String],
    }
    let lexicon = Lexicon {
        tokenizer: "jieba-rs-0.7",
        pos: config
            .pos_dict
            .iter()
            .map(|(word, tag)| (word.as_str(), tag.as_str()))
            .collect(),
        name_prefixes: &config.name_patterns.prefixes,
        name_suffixes: &config.name_patterns.suffixes,
        full_names: &config.name_patterns.full_names,
        profession_suffix: &config.name_patterns.profession_suffix,
    };
    canonical::canonical_digest(&lexicon).unwrap_or_default()
}
/// 从计划中抽取需要链接检查的槽位。
///
/// CheckTriple 的宾语可能是字面量（「年龄是 18」），因此不把它当作实体槽；
/// 谓词槽取主语链中第一个之后的段。这样失败分类不会把合法取值误报成未知实体。
pub fn plan_terms(plan: &QueryPlan) -> (Vec<String>, Vec<String>) {
    let mut entities = Vec::new();
    let mut predicates = Vec::new();
    let mut push_entity = |value: &str| {
        let value = value.trim();
        if !value.is_empty() && !value.chars().all(|ch| ch.is_numeric()) {
            entities.push(value.to_string());
        }
    };
    match plan {
        QueryPlan::FindObject { segments } => {
            if let Some(subject) = segments.first() {
                push_entity(subject);
            }
            predicates.extend(segments.iter().skip(1).cloned());
        }
        QueryPlan::FindSubject { object, predicate } => {
            push_entity(object);
            predicates.push(predicate.clone());
        }
        QueryPlan::FindObjectByPredicate { subject, predicate } => {
            push_entity(subject);
            predicates.push(predicate.clone());
        }
        QueryPlan::FindObjectExcept {
            subject,
            predicate,
            excluded,
        } => {
            push_entity(subject);
            predicates.push(predicate.clone());
            for entity in excluded {
                push_entity(entity);
            }
        }
        QueryPlan::FindSubjectExcept {
            object,
            predicate,
            excluded,
        } => {
            push_entity(object);
            predicates.push(predicate.clone());
            for entity in excluded {
                push_entity(entity);
            }
        }
        QueryPlan::PathBetween { start, goal, .. } => {
            push_entity(start);
            push_entity(goal);
        }
        QueryPlan::CommonNeighbor { left, right } => {
            push_entity(left);
            push_entity(right);
        }
        QueryPlan::Intermediary { start, goal, .. } => {
            push_entity(start);
            push_entity(goal);
        }
        QueryPlan::CheckTriple {
            subject_segments, ..
        } => {
            if let Some(subject) = subject_segments.first() {
                push_entity(subject);
            }
            predicates.extend(subject_segments.iter().skip(1).cloned());
        }
        QueryPlan::CheckRelation {
            subject,
            predicate,
            object,
            negated: _,
        } => {
            push_entity(subject);
            push_entity(object);
            predicates.push(predicate.clone());
        }
        QueryPlan::Count { subject, predicate } | QueryPlan::Exist { subject, predicate } => {
            push_entity(subject);
            predicates.push(predicate.clone());
        }
        QueryPlan::Closure {
            subject,
            predicate,
            then,
            ..
        } => {
            push_entity(subject);
            predicates.push(predicate.clone());
            predicates.extend(then.iter().cloned());
        }
        QueryPlan::Similar { entity } => push_entity(entity),
        QueryPlan::Analogy { a, b, c } => {
            push_entity(a);
            push_entity(b);
            push_entity(c);
        }
    }
    predicates.retain(|predicate| !predicate.trim().is_empty());
    entities.sort();
    entities.dedup();
    predicates.sort();
    predicates.dedup();
    (entities, predicates)
}
fn plan_trace(plan: &QueryPlan) -> PlanTrace {
    let (entities, predicates) = plan_terms(plan);
    PlanTrace {
        kind: QueryPlanner::kind(plan).to_string(),
        display: QueryPlanner::format_plan(plan),
        entities,
        predicates,
    }
}
/// 读法追踪主入口：只读取文本与配置，不访问知识库、不改变任何状态。
pub fn trace_question(
    question: &str,
    config: &NlpConfig,
    plan: Option<&QueryPlan>,
) -> LexicalTrace {
    trace_question_with_mention_lexicon(question, config, plan, None)
}
/// 与 [`trace_question`] 相同，但可注入已确认提及词表：
/// 审计切分与真实路由使用同一覆盖层，避免「报告里一种读法、执行时另一种读法」。
pub fn trace_question_with_mention_lexicon(
    question: &str,
    config: &NlpConfig,
    plan: Option<&QueryPlan>,
    lexicon: Option<&crate::mention_lexicon::MentionLexicon>,
) -> LexicalTrace {
    let trimmed = question.trim();
    let question_chars = trimmed.chars().count();
    let annotated = match lexicon {
        Some(lexicon) => crate::nlp::preprocess_annotated_with_lexicon(trimmed, config, lexicon),
        None => preprocess_annotated(trimmed, config),
    };
    let tokens: Vec<TraceToken> = annotated
        .iter()
        .map(|word| TraceToken {
            text: word.text.clone(),
            pos: word.pos.clone(),
            source: word.source,
            start: word.start,
            end: word.end,
        })
        .collect();

    let mut divergences = Vec::new();
    for (index, word) in annotated.iter().enumerate() {
        if let Some(jieba) = &word.jieba_pos {
            if word.source != TagSource::JiebaTag && jieba != &word.pos {
                divergences.push(PosDivergence {
                    token_index: index,
                    text: word.text.clone(),
                    chosen: word.pos.clone(),
                    source: word.source,
                    jieba: jieba.clone(),
                });
            }
        }
    }

    let plan_trace = plan.map(plan_trace);
    let plan_is_closure = matches!(plan, Some(QueryPlan::Closure { .. }));
    let mut markers = Vec::new();
    for marker in CLOSURE_MARKERS {
        for (byte_index, _) in trimmed.match_indices(marker) {
            let start = trimmed[..byte_index].chars().count();
            let end = start + marker.chars().count();
            let token_at = tokens
                .iter()
                .position(|token| token.start <= end - 1 && end - 1 < token.end);
            let start_aligned = tokens.iter().any(|token| token.start == start);
            let end_aligned = tokens.iter().any(|token| token.end == end);
            markers.push(MarkerTrace {
                marker: marker.to_string(),
                start,
                end,
                token_index: token_at,
                token_text: token_at.map(|index| tokens[index].text.clone()),
                boundary_violation: !start_aligned || !end_aligned,
                plan_consumed: plan_is_closure,
            });
        }
    }
    markers.sort_by(|a, b| {
        a.start
            .cmp(&b.start)
            .then(b.marker.len().cmp(&a.marker.len()))
            .then(a.marker.cmp(&b.marker))
    });

    let mut warnings = Vec::new();
    for marker in markers.iter().filter(|marker| marker.boundary_violation) {
        warnings.push(ReadWarning {
            kind: ReadWarningKind::ClosureMarkerInToken,
            detail: format!(
                "「{}」{}..{} 未对齐 token 边界{}",
                marker.marker,
                marker.start,
                marker.end,
                marker
                    .token_text
                    .as_deref()
                    .map(|token| format!("（token「{}」）", token))
                    .unwrap_or_default()
            ),
        });
    }

    let mut reading_state = reading::diagnose(trimmed, plan);
    if plan.is_none() {
        // 没有计划时，未登录词是「为什么读不懂」的直接证据；只在此分支
        // 报告，成功计划里的程度词/量词不会制造噪声。结构助词不是词表缺口。
        for token in tokens.iter().filter(|token| token.source == TagSource::Unknown) {
            let text = token.text.trim();
            if text.is_empty()
                || !text.chars().any(char::is_alphanumeric)
                || matches!(
                    text,
                    "的" | "呢" | "吗" | "吧" | "啊" | "呀" | "了" | "着" | "过" | "是" | "有"
                )
            {
                continue;
            }
            reading_state.push(
                Residual::new(
                    ResidualKind::UnknownToken,
                    ResidualStage::Tag,
                    ResidualSeverity::Warning,
                    RepairKind::RewriteQuestion,
                    format!("未登录词「{}」不在词表/本体中", text),
                )
                .surface(trimmed, TextSpan::new(token.start, token.end))
                .expected("词表/本体中的实体或谓词"),
            );
        }
    }
    for marker in markers.iter().filter(|marker| marker.boundary_violation) {
        // 标记词落在别的 token 内部却已经被当前计划消费，说明计划本身可能
        // 建立在子串误读上（「所有权」不是「所有」）；这种残差必须阻断，
        // 而普通边界提示只需警告。
        let consumed = marker.plan_consumed;
        reading_state.push(
            Residual::new(
                ResidualKind::TokenBoundary,
                ResidualStage::Tokenize,
                if consumed {
                    ResidualSeverity::Blocking
                } else {
                    ResidualSeverity::Warning
                },
                RepairKind::SplitToken,
                format!(
                    "标记词「{}」没有对齐 token 边界，{}",
                    marker.marker,
                    if consumed {
                        "但已被当前闭包计划消费，读法可疑"
                    } else {
                        "可能把词内子串当成语法标记"
                    }
                ),
            )
            .surface(trimmed, TextSpan::new(marker.start, marker.end))
            .expected("与标记词语义一致的 token 边界"),
        );
    }

    let mut failures = Vec::new();
    if trimmed.is_empty() {
        failures.push(ReadFailure {
            stage: ReadStage::Input,
            kind: ReadFailureKind::EmptyInput,
            detail: "问句为空".to_string(),
        });
    } else if question_chars > reading::MAX_QUESTION_CHARS {
        failures.push(ReadFailure {
            stage: ReadStage::Input,
            kind: ReadFailureKind::InputTooLong,
            detail: format!(
                "问句 {} 字，超过 {} 字上限",
                question_chars,
                reading::MAX_QUESTION_CHARS
            ),
        });
    } else if plan.is_none() {
        failures.push(ReadFailure {
            stage: ReadStage::Plan,
            kind: ReadFailureKind::NoQuestionShape,
            detail: "查询规划器没有找到可执行形状".to_string(),
        });
    }

    LexicalTrace {
        version: LEXICAL_TRACE_VERSION,
        question_digest: canonical::canonical_digest(&trimmed).unwrap_or_default(),
        question_chars,
        lexicon_digest: lexicon_digest(config),
        fidelity: TraceFidelity::Reconstructed,
        tokens,
        divergences,
        plan: plan_trace,
        markers,
        oov: Vec::new(),
        failures,
        warnings,
        reading: reading_state,
        coverage: None,
        answer_available: None,
    }
}

/// 静态词表覆盖审计（Lexical Coverage）。
///
/// `trace_question` 只报告「没有计划时」的未登录词，因此成功计划里的生词
/// （「曾祖父」被结巴标成 nr、「儿子叫」被拼成一个谓词）不会留下证据。
/// 覆盖审计把计划槽位重新对回文本 span，并用配置里可执行的词表（谓词画像、
/// 查询规则字面量、抽取关键词、词性词典、同义词）判定：
///
///   - [`LexGapLayer::Sense`]：词形认识，但没有可执行定义（谓词缺口）；
///   - [`LexGapLayer::World`]：词形未登记，需要登记实体/补事实（世界缺口）；
///   - [`LexGapLayer::Form`]：切词/词形本身没有被解释（词表缺口）。
///
/// 这是只读审计：不改变分词、链接、规划或回答；只有显式请求时才把缺口并入
/// 读法残差，便于 `read-trace --lexicon` 与语料报告使用。
pub const LEXICAL_COVERAGE_VERSION: u32 = 1;
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LexGapLayer {
    Form,
    Sense,
    World,
}
impl LexGapLayer {
    pub fn code(self) -> &'static str {
        match self {
            LexGapLayer::Form => "form",
            LexGapLayer::Sense => "sense",
            LexGapLayer::World => "world",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            LexGapLayer::Form => "词形",
            LexGapLayer::Sense => "词义",
            LexGapLayer::World => "世界",
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LexGapRole {
    Entity,
    Predicate,
    Type,
    Token,
}
impl LexGapRole {
    pub fn code(self) -> &'static str {
        match self {
            LexGapRole::Entity => "entity",
            LexGapRole::Predicate => "predicate",
            LexGapRole::Type => "type",
            LexGapRole::Token => "token",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            LexGapRole::Entity => "实体",
            LexGapRole::Predicate => "谓词",
            LexGapRole::Type => "类型",
            LexGapRole::Token => "未归属词",
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LexCoverageGap {
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub span: Option<TextSpan>,
    pub layer: LexGapLayer,
    pub role: LexGapRole,
    pub severity: ResidualSeverity,
    pub repair: RepairKind,
    /// 为什么判定为缺口（确定性文本，不含随机构造）。
    pub evidence: String,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LexResolvedTerm {
    pub text: String,
    /// 离线准入的规范谓词链（「先生」→ [「配偶」]，「师祖」→ [「师傅」,「师傅」]）。
    pub expansion: Vec<String>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LexicalCoverage {
    pub version: u32,
    /// 计划槽位中能在静态词表里找到定义的词数。
    pub known_terms: u64,
    /// 计划引入、原句中没有字面出现的词数（如「多大」→「年龄」）。
    pub introduced_terms: u64,
    /// 未被任何计划槽位消费的内容 token 数。
    pub unbound_tokens: u64,
    /// 由离线定义准入解析掉的词面（仅 `--definitions` 时非空）。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub resolved: Vec<LexResolvedTerm>,
    pub gaps: Vec<LexCoverageGap>,
}
impl LexicalCoverage {
    pub fn blocking_count(&self) -> usize {
        self.gaps
            .iter()
            .filter(|gap| gap.severity == ResidualSeverity::Blocking)
            .count()
    }

    pub fn warning_count(&self) -> usize {
        self.gaps
            .iter()
            .filter(|gap| gap.severity == ResidualSeverity::Warning)
            .count()
    }

    pub fn is_clean(&self) -> bool {
        self.gaps.is_empty()
    }

    pub fn render_lines(&self) -> Vec<String> {
        let mut lines = vec![format!(
            "词表覆盖审计：{} 个已知槽位，{} 个计划引入词，{} 个未归属 token；缺口 {} 条（阻断 {}）。",
            self.known_terms,
            self.introduced_terms,
            self.unbound_tokens,
            self.gaps.len(),
            self.blocking_count()
        )];
        if !self.resolved.is_empty() {
            let resolved = self
                .resolved
                .iter()
                .map(|term| format!("{} → {}", term.text, term.expansion.join(" ∘ ")))
                .collect::<Vec<_>>()
                .join("；");
            lines.push(format!("定义解析：{}。", resolved));
        }
        if self.gaps.is_empty() {
            lines.push("覆盖结论：计划槽位均能在静态词表中找到定义。".to_string());
            return lines;
        }
        for gap in &self.gaps {
            let span = gap
                .span
                .map(|span| format!(" {}..{}", span.start, span.end))
                .unwrap_or_default();
            lines.push(format!(
                "  • 「{}」{} {} /{}（{}，修复：{}）：{}",
                gap.text,
                span,
                gap.layer.label(),
                gap.role.label(),
                gap.severity.label(),
                gap.repair.label(),
                gap.evidence
            ));
        }
        lines
    }
}
/// 静态词表：从配置中提取「系统已经承诺认识」的词。
///
/// 这不是完整本体：它只包含配置里能离线读到的可执行词表。知识库里的实体/
/// 谓词仍需由链接阶段确认，因此世界缺口只作警告，词义缺口才阻断。
#[derive(Debug, Clone)]
pub struct LexiconInventory {
    predicates: BTreeSet<String>,
    entities: BTreeSet<String>,
    types: BTreeSet<String>,
    /// 离线准入的定义展开（surface -> 规范谓词链），来自 [`crate::lex_gate`]。
    definitions: BTreeMap<String, Vec<String>>,
}
const QUESTION_MARKERS: [&str; 34] = [
    "谁", "什么", "哪里", "哪儿", "哪个", "哪些", "多少", "几", "怎么", "怎样", "如何",
    "为什么", "为何", "什么样", "什么时候", "多大", "多久", "多高", "多重", "多远", "吗",
    "呢", "吧", "啊", "呀", "嘛", "请问", "是否", "是不是", "有没有", "对不对", "好不好",
    "行不行", "可以吗",
];
const FUNCTION_WORDS: [&str; 40] = [
    "的", "是", "和", "与", "及", "以及", "或", "或者", "在", "有", "被", "把", "对", "给",
    "让", "从", "到", "向", "往", "为", "了", "着", "过", "得", "地", "最", "更", "还", "也",
    "都", "就", "才", "又", "再", "个", "位", "名", "次", "个", "种",
];
fn is_content_pos(tag: &str) -> bool {
    matches!(
        tag,
        "n" | "nr" | "ns" | "nt" | "nz" | "ng" | "v" | "vn" | "vd" | "a" | "an"
            | "i" | "l" | "j" | "eng"
    )
}
fn has_cjk(text: &str) -> bool {
    text.chars().any(|ch| ('\u{4e00}'..='\u{9fff}').contains(&ch))
}
fn is_content_word(text: &str) -> bool {
    let text = text.trim();
    !text.is_empty()
        && text.chars().any(char::is_alphanumeric)
        && !FUNCTION_WORDS.contains(&text)
        && !QUESTION_MARKERS.contains(&text)
}
impl LexiconInventory {
    pub fn from_config(config: &SystemConfig) -> Self {
        let mut predicates = BTreeSet::new();
        let mut entities = BTreeSet::new();
        let mut types = BTreeSet::new();

        // 1) 谓词画像：规范名与别名是系统最明确的谓词承诺。
        for profile in &config.predicate_profiles {
            for word in std::iter::once(&profile.name).chain(profile.aliases.iter()) {
                if is_content_word(word) {
                    predicates.insert(word.clone());
                }
            }
            types.extend(profile.domain.iter().cloned());
            types.extend(profile.range.iter().cloned());
        }

        // 2) 查询规则里的中文字面量（「首都」「作者」「身高」「关系」等）。
        //    POS 标签（nr/n/ns...）不是词条，不进入词表。
        for rule in &config.prolog_rules {
            for pattern in &rule.pattern {
                if has_cjk(pattern) && is_content_word(pattern) {
                    predicates.insert(pattern.clone());
                }
            }
        }

        // 3) 抽取规则关键词与属性名。
        for rule in &config.extraction_rules {
            for pattern in &rule.patterns {
                if let Some(keywords) = &pattern.keyword_patterns {
                    for keyword in keywords {
                        if is_content_word(keyword) {
                            predicates.insert(keyword.clone());
                        }
                    }
                }
                for property in pattern.property_mappings.keys() {
                    if is_content_word(property) {
                        predicates.insert(property.clone());
                    }
                }
            }
        }
        for entity_type in &config.entity_types {
            types.insert(entity_type.label.clone());
            for property in &entity_type.properties {
                if is_content_word(&property.name) {
                    predicates.insert(property.name.clone());
                }
            }
        }

        // 4) 词性词典中的内容词：这是分词器之外系统显式承诺认识的词。
        for (word, tag) in &config.nlp.pos_dict {
            if is_content_pos(tag) && is_content_word(word) {
                predicates.insert(word.clone());
            }
        }

        // 5) 同义词映射：两侧都是系统认识的词形。
        if let Some(synonyms) = &config.synonyms {
            for (from, to) in &synonyms.mappings {
                if is_content_word(from) {
                    predicates.insert(from.clone());
                }
                if is_content_word(to) {
                    predicates.insert(to.clone());
                }
            }
        }

        // 6) 实体身份：显式声明的规范名/别名与姓名模式。
        for profile in &config.entity_profiles {
            for word in std::iter::once(&profile.name).chain(profile.aliases.iter()) {
                if is_content_word(word) {
                    entities.insert(word.clone());
                }
            }
            types.extend(profile.etype.iter().cloned());
        }
        for name in &config.nlp.name_patterns.full_names {
            if is_content_word(name) {
                entities.insert(name.clone());
            }
        }
        for alias in &config.type_system.aliases {
            types.insert(alias.alias.clone());
            types.insert(alias.canonical.clone());
        }

        predicates.retain(|word| word.chars().count() > 0);
        entities.retain(|word| word.chars().count() > 0);
        types.retain(|word| word.chars().count() > 0);
        Self {
            predicates,
            entities,
            types,
            definitions: BTreeMap::new(),
        }
    }

    /// 登记一条已通过准入的定义展开；surface 同时成为可识别的谓词词面。
    pub fn insert_definition(&mut self, surface: &str, expansion: Vec<String>) {
        let surface = surface.trim();
        if surface.is_empty() || expansion.is_empty() {
            return;
        }
        self.predicates.insert(surface.to_string());
        self.definitions.insert(surface.to_string(), expansion);
    }

    pub fn definition_expansion(&self, term: &str) -> Option<&[String]> {
        self.definitions.get(term.trim()).map(Vec::as_slice)
    }

    pub fn definition_count(&self) -> usize {
        self.definitions.len()
    }

    pub fn predicate_count(&self) -> usize {
        self.predicates.len()
    }

    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    pub fn type_count(&self) -> usize {
        self.types.len()
    }

    pub fn is_predicate(&self, term: &str) -> bool {
        let term = term.trim();
        self.predicates.contains(term) || self.predicates.contains(term.trim_end_matches('的'))
    }

    pub fn is_entity(&self, term: &str) -> bool {
        self.entities.contains(term.trim())
    }

    pub fn is_type(&self, term: &str) -> bool {
        self.types.contains(term.trim())
    }
}
fn ignorable_token(text: &str) -> bool {
    let text = text.trim();
    text.is_empty()
        || !text.chars().any(char::is_alphanumeric)
        || FUNCTION_WORDS.contains(&text)
        || QUESTION_MARKERS.contains(&text)
        || text.chars().all(|ch| ch.is_ascii_digit())
}
fn locate_term(chars: &[char], term: &str, used: &mut [bool]) -> Option<TextSpan> {
    let term_chars: Vec<char> = term.trim().chars().collect();
    if term_chars.is_empty() || term_chars.len() > chars.len() {
        return None;
    }
    for start in 0..=chars.len() - term_chars.len() {
        let matches = term_chars
            .iter()
            .enumerate()
            .all(|(offset, ch)| chars[start + offset] == *ch && !used[start + offset]);
        if !matches {
            continue;
        }
        for offset in 0..term_chars.len() {
            used[start + offset] = true;
        }
        return Some(TextSpan::new(start, start + term_chars.len()));
    }
    None
}
fn tokens_in_span<'a>(
    tokens: &'a [TraceToken],
    span: TextSpan,
) -> impl Iterator<Item = &'a TraceToken> {
    tokens
        .iter()
        .filter(move |token| token.start < span.end && span.start < token.end)
}
fn span_has_unknown_token(tokens: &[TraceToken], span: TextSpan) -> bool {
    tokens_in_span(tokens, span).any(|token| {
        token.source == TagSource::Unknown && token.text.chars().any(char::is_alphanumeric)
    })
}
/// 实体 span 是否得到姓名证据支持：规划器的跨度恢复把「叶」「莉」拼回
/// 「叶莉」时，其中一个 token 已被词性标注为人名；离线审计不应再把这种
/// 已被消费的碎片报成世界缺口。
fn span_has_name_evidence(tokens: &[TraceToken], span: TextSpan) -> bool {
    tokens_in_span(tokens, span).any(|token| crate::pos_ontology::tag_matches("nr", &token.pos))
}
fn span_unknown_surfaces(tokens: &[TraceToken], span: TextSpan) -> Vec<String> {
    let mut surfaces: Vec<String> = tokens_in_span(tokens, span)
        .filter(|token| {
            token.source == TagSource::Unknown && token.text.chars().any(char::is_alphanumeric)
        })
        .map(|token| token.text.clone())
        .collect();
    surfaces.sort();
    surfaces.dedup();
    surfaces
}

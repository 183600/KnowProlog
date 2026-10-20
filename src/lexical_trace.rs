//! READ：读法追踪（Lexical Trace）——让系统看见自己「怎么读」。
//!
//! 推理侧已有证明、闭包、搜索状态等可观测量，但「输入被读成了什么」长期
//! 是隐形的：token 边界、词性来源、规划器命中的标记词、未登录词，都只在
//! 各自模块里短暂存在。本模块把这些证据收拢成一份确定性的只读轨迹：
//!
//! - 每个 token 的字符区间、词性、[`TagSource`]（配置 / 姓名模式 / 结巴 / 未知）；
//! - 配置词性与结巴原始标签的分歧（可用来校准词表，而不是猜）；
//! - 查询计划、实体槽与谓词槽（用于判断失败发生在链接前还是执行后）；
//! - 闭包标记词的字符区间是否落在 token 内部（「所有权」不是「所有」）；
//! - OOV 提名（由宿主注入，只报告不落库）；
//! - 失败分类：没有问句形状、未知实体、未知谓词、执行未命中……
//!
//! 设计约束：
//!   - 只读：不改变分词、链接、规划或回答的默认行为；
//!   - 诚实：轨迹由事后重跑纯函数得到，保真度标为 [`TraceFidelity::Reconstructed`]；
//!   - 有界：失败回执进入固定容量环形缓冲，进程重启即失，不写会话快照；
//!   - 稳定：报告排序确定、摘要只含稳定字段；原始问句只出现在本地渲染，
//!     回执只保留 digest、计数与有限枚举。

use std::collections::{BTreeMap, VecDeque};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::canonical;
use crate::nlp::{preprocess_annotated, TagSource};
use crate::query_planner::{QueryPlan, QueryPlanner};
use crate::NlpConfig;

pub const LEXICAL_TRACE_VERSION: u32 = 1;
pub const DEFAULT_FAILURE_TRACE_CAPACITY: usize = 32;

/// 闭包/精确跳数标记词：出现位置与计划解读不一致时，轨迹必须能指出来。
const CLOSURE_MARKERS: [&str; 10] = [
    "的任何", "的一切", "的所有", "的全部", "的任意", "所有", "全部", "恰好", "正好", "刚好",
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
    pub fn push_failure(&mut self, stage: ReadStage, kind: ReadFailureKind, detail: impl Into<String>) {
        let detail = detail.into();
        if self
            .failures
            .iter()
            .any(|failure| failure.stage == stage && failure.kind == kind && failure.detail == detail)
        {
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
            warning_kinds,
            closure_marker_violations: self.boundary_violations() as u64,
            oov_count: self.oov.len() as u64,
            fidelity: self.fidelity,
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
                let mut line = format!(
                    "  • 「{}」{}..{}",
                    marker.marker, marker.start, marker.end
                );
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
                output.push_str(&format!("  • {}：{}\n", warning.kind.label(), warning.detail));
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
    pub warning_kinds: Vec<String>,
    pub closure_marker_violations: u64,
    pub oov_count: u64,
    pub fidelity: TraceFidelity,
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
        let mut plans: BTreeMap<String, u64> = BTreeMap::new();
        let mut sources: BTreeMap<String, u64> = BTreeMap::new();
        let mut unknown_tokens = 0u64;
        let mut marker_violations = 0u64;
        let mut oov = 0u64;
        for receipt in &receipts {
            for kind in &receipt.failure_kinds {
                *failures.entry(kind.clone()).or_insert(0) += 1;
            }
            let plan = receipt
                .plan_kind
                .clone()
                .unwrap_or_else(|| "无计划".to_string());
            *plans.entry(plan).or_insert(0) += 1;
            for (source, count) in &receipt.source_counts {
                *sources.entry(source.clone()).or_insert(0) += count;
            }
            unknown_tokens += receipt.unknown_tokens;
            marker_violations += receipt.closure_marker_violations;
            oov += receipt.oov_count;
        }
        let mut output = format!(
            "读法台账：{} 条有界回执（容量 {}，不保存原文）。\n",
            receipts.len(),
            self.capacity
        );
        output.push_str("失败分类（条次）：");
        output.push_str(&render_counts(&failures));
        output.push('\n');
        output.push_str("计划分布（条次）：");
        output.push_str(&render_counts(&plans));
        output.push('\n');
        output.push_str("词性来源（token 数）：");
        output.push_str(&render_counts(&sources));
        output.push('\n');
        output.push_str(&format!(
            "未知词：{}；标记边界越界：{}；OOV 提名：{}。\n",
            unknown_tokens, marker_violations, oov
        ));
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
    let trimmed = question.trim();
    let question_chars = trimmed.chars().count();
    let annotated = preprocess_annotated(trimmed, config);
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

    let mut failures = Vec::new();
    if trimmed.is_empty() {
        failures.push(ReadFailure {
            stage: ReadStage::Input,
            kind: ReadFailureKind::EmptyInput,
            detail: "问句为空".to_string(),
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
        assert!(snapshot.iter().all(|receipt| !receipt.question_digest.is_empty()));
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
}

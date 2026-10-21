//! READ：读法残差表（Reading Residuals）——把「没读懂」变成可修复的诊断。
//!
//! 现有的词法追溯 (`lexical_trace`) 能看见 token、词性来源和查询计划，但
//! 规划器返回 `Option`：失败时，调用方只知道「没有计划」，不知道缺的是
//! 实体、谓词、论元、分词边界，还是问法本身不支持。本模块在「读法」这一
//! 层保留部分理解与未解决项：
//!
//! - [`Reading`]：一次读法的有界残差表，而不是一个 `Option`；
//! - [`Residual`]：带阶段、字符区间、期待类型、候选值、严重度与修复动作的
//!   可操作诊断；没有 span 的失败只是日志，没有候选的残差只能开放式追问；
//! - [`ResidualKind`] / [`RepairKind`]：稳定闭集，便于回执聚合与回归测试；
//! - [`diagnose`]：纯函数、确定性、零知识库访问的句面诊断；
//! - [`Reading::add_*`]：供推理侧在链接/执行阶段补充知识感知的残差。
//!
//! 设计约束：
//!   - 部分优于空无：即使没有计划也返回一个 [`Reading`]，保留已有分类；
//!   - 有界：残差数、候选数、detail 长度都有上限，避免异常输入撑爆台账；
//!   - 稳定：排序只依赖枚举与字符串，不依赖 HashMap 迭代序或浮点序；
//!   - 只读：诊断不改变回答内容，修复动作由调用方显式执行；
//!   - 诚实：句面诊断只报告能从文本确定的事情，知识缺口由推理侧注入。

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::query_planner::{QueryPlan, QueryPlanner};

pub const READING_VERSION: u32 = 1;
/// 单次读法最多保留的残差数（按插入顺序，重复项先去重）。
pub const MAX_RESIDUALS: usize = 12;
/// 单条残差最多保留的候选数。
pub const MAX_CANDIDATES: usize = 5;
/// 单条残差 detail 的字符上限。
pub const MAX_DETAIL_CHARS: usize = 200;
/// 与查询规划器保持一致的输入长度上限。
pub const MAX_QUESTION_CHARS: usize = 80;
/// 省略问句的核心长度上限（「那李四呢」仍然可补全）。
const MAX_ELLIPSIS_CORE_CHARS: usize = 12;

/// 读法整体状态：回答管线据此决定是直接作答、带披露作答，还是追问/弃权。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadingStatus {
    /// 已有可执行计划，没有阻断性残差（可以有警告）。
    Ready,
    /// 读到了一部分，但仍有阻断性残差（链接/执行/知识缺口）。
    Partial,
    /// 连可执行形状都没有：需要补全、改写或澄清。
    Unreadable,
}

impl ReadingStatus {
    pub fn code(self) -> &'static str {
        match self {
            ReadingStatus::Ready => "ready",
            ReadingStatus::Partial => "partial",
            ReadingStatus::Unreadable => "unreadable",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            ReadingStatus::Ready => "可执行",
            ReadingStatus::Partial => "部分可读",
            ReadingStatus::Unreadable => "不可读",
        }
    }

    pub fn is_ready(self) -> bool {
        matches!(self, ReadingStatus::Ready)
    }
}

/// 残差发生的阶段。与词法追溯的阶段同名，但本模块不依赖 `lexical_trace`，
/// 保持读法对象可独立使用、可单独序列化。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResidualStage {
    Input,
    Tokenize,
    Tag,
    Normalize,
    Plan,
    Link,
    Execute,
    Generate,
}

impl ResidualStage {
    pub fn code(self) -> &'static str {
        match self {
            ResidualStage::Input => "input",
            ResidualStage::Tokenize => "tokenize",
            ResidualStage::Tag => "tag",
            ResidualStage::Normalize => "normalize",
            ResidualStage::Plan => "plan",
            ResidualStage::Link => "link",
            ResidualStage::Execute => "execute",
            ResidualStage::Generate => "generate",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            ResidualStage::Input => "输入",
            ResidualStage::Tokenize => "切词",
            ResidualStage::Tag => "词性",
            ResidualStage::Normalize => "归一",
            ResidualStage::Plan => "规划",
            ResidualStage::Link => "链接",
            ResidualStage::Execute => "执行",
            ResidualStage::Generate => "生成",
        }
    }
}

/// 残差类别闭集。新增类别必须有提取规则、修复动作和测试，否则放字段。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResidualKind {
    EmptyInput,
    InputTooLong,
    Ellipsis,
    UnsupportedQuestion,
    MissingEntity,
    AmbiguousEntity,
    UnknownEntity,
    MissingPredicate,
    AmbiguousPredicate,
    UnknownPredicate,
    MissingArgument,
    TokenBoundary,
    TypeConflict,
    BudgetExhausted,
    EngineFault,
    FactGap,
    ExecutionMiss,
    NoQuestionShape,
    Unclassified,
}

impl ResidualKind {
    pub fn code(self) -> &'static str {
        match self {
            ResidualKind::EmptyInput => "empty_input",
            ResidualKind::InputTooLong => "input_too_long",
            ResidualKind::Ellipsis => "ellipsis",
            ResidualKind::UnsupportedQuestion => "unsupported_question",
            ResidualKind::MissingEntity => "missing_entity",
            ResidualKind::AmbiguousEntity => "ambiguous_entity",
            ResidualKind::UnknownEntity => "unknown_entity",
            ResidualKind::MissingPredicate => "missing_predicate",
            ResidualKind::AmbiguousPredicate => "ambiguous_predicate",
            ResidualKind::UnknownPredicate => "unknown_predicate",
            ResidualKind::MissingArgument => "missing_argument",
            ResidualKind::TokenBoundary => "token_boundary",
            ResidualKind::TypeConflict => "type_conflict",
            ResidualKind::BudgetExhausted => "budget_exhausted",
            ResidualKind::EngineFault => "engine_fault",
            ResidualKind::FactGap => "fact_gap",
            ResidualKind::ExecutionMiss => "execution_miss",
            ResidualKind::NoQuestionShape => "no_question_shape",
            ResidualKind::Unclassified => "unclassified",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            ResidualKind::EmptyInput => "空输入",
            ResidualKind::InputTooLong => "输入过长",
            ResidualKind::Ellipsis => "省略",
            ResidualKind::UnsupportedQuestion => "不支持的问法",
            ResidualKind::MissingEntity => "缺实体",
            ResidualKind::AmbiguousEntity => "实体歧义",
            ResidualKind::UnknownEntity => "未知实体",
            ResidualKind::MissingPredicate => "缺谓词",
            ResidualKind::AmbiguousPredicate => "谓词歧义",
            ResidualKind::UnknownPredicate => "未知谓词",
            ResidualKind::MissingArgument => "缺论元",
            ResidualKind::TokenBoundary => "词边界",
            ResidualKind::TypeConflict => "类型冲突",
            ResidualKind::BudgetExhausted => "预算耗尽",
            ResidualKind::EngineFault => "引擎故障",
            ResidualKind::FactGap => "事实缺口",
            ResidualKind::ExecutionMiss => "执行未命中",
            ResidualKind::NoQuestionShape => "没有问句形状",
            ResidualKind::Unclassified => "未分类",
        }
    }
}

/// 残差严重度：阻断性问题不能静默忽略；警告性问题可以作答但应披露。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResidualSeverity {
    Info,
    Warning,
    Blocking,
}

impl ResidualSeverity {
    pub fn code(self) -> &'static str {
        match self {
            ResidualSeverity::Info => "info",
            ResidualSeverity::Warning => "warning",
            ResidualSeverity::Blocking => "blocking",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            ResidualSeverity::Info => "提示",
            ResidualSeverity::Warning => "警告",
            ResidualSeverity::Blocking => "阻断",
        }
    }
}

/// 修复动作闭集。每轮最多采用一个；优先静默修复，其次才是追问。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RepairKind {
    None,
    BindAttribute,
    BindEllipsis,
    SupplyEntity,
    SupplyPredicate,
    ChooseCandidate,
    ReversePredicate,
    SplitToken,
    MergeToken,
    AddParticle,
    RewriteQuestion,
    AskClarification,
    AcquireFact,
    CheckEngine,
    ShortenInput,
    Abstain,
}

impl RepairKind {
    pub fn code(self) -> &'static str {
        match self {
            RepairKind::None => "none",
            RepairKind::BindAttribute => "bind_attribute",
            RepairKind::BindEllipsis => "bind_ellipsis",
            RepairKind::SupplyEntity => "supply_entity",
            RepairKind::SupplyPredicate => "supply_predicate",
            RepairKind::ChooseCandidate => "choose_candidate",
            RepairKind::ReversePredicate => "reverse_predicate",
            RepairKind::SplitToken => "split_token",
            RepairKind::MergeToken => "merge_token",
            RepairKind::AddParticle => "add_particle",
            RepairKind::RewriteQuestion => "rewrite_question",
            RepairKind::AskClarification => "ask_clarification",
            RepairKind::AcquireFact => "acquire_fact",
            RepairKind::CheckEngine => "check_engine",
            RepairKind::ShortenInput => "shorten_input",
            RepairKind::Abstain => "abstain",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            RepairKind::None => "无需修复",
            RepairKind::BindAttribute => "补属性",
            RepairKind::BindEllipsis => "补省略",
            RepairKind::SupplyEntity => "补实体",
            RepairKind::SupplyPredicate => "补谓词",
            RepairKind::ChooseCandidate => "选择候选",
            RepairKind::ReversePredicate => "逆向谓词",
            RepairKind::SplitToken => "拆分 token",
            RepairKind::MergeToken => "合并 token",
            RepairKind::AddParticle => "补结构助词",
            RepairKind::RewriteQuestion => "改写问句",
            RepairKind::AskClarification => "追问澄清",
            RepairKind::AcquireFact => "补充事实",
            RepairKind::CheckEngine => "检查引擎",
            RepairKind::ShortenInput => "缩短输入",
            RepairKind::Abstain => "弃权",
        }
    }
}

/// 字符区间（相对原始问句的 Unicode 标量序号，左闭右开）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TextSpan {
    pub start: usize,
    pub end: usize,
}

impl TextSpan {
    pub fn new(start: usize, end: usize) -> Self {
        Self {
            start: start.min(end),
            end: end.max(start),
        }
    }

    pub fn chars(self) -> usize {
        self.end.saturating_sub(self.start)
    }
}

/// 一条读法残差：有位置、有期待、有候选、有修复动作。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Residual {
    pub kind: ResidualKind,
    pub stage: ResidualStage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub span: Option<TextSpan>,
    /// 原句中的局部文本（仅本地诊断；回执只保留类别）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub surface: Option<String>,
    /// 这个位置本来期待什么（类型、角色或结构）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected: Option<String>,
    /// 可枚举的候选（已去重、有界）。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub candidates: Vec<String>,
    pub severity: ResidualSeverity,
    pub repair: RepairKind,
    pub detail: String,
}

impl Residual {
    pub fn new(
        kind: ResidualKind,
        stage: ResidualStage,
        severity: ResidualSeverity,
        repair: RepairKind,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            kind,
            stage,
            span: None,
            surface: None,
            expected: None,
            candidates: Vec::new(),
            severity,
            repair,
            detail: clamp_chars(detail.into(), MAX_DETAIL_CHARS),
        }
    }

    pub fn at(mut self, span: TextSpan) -> Self {
        self.span = Some(span);
        self
    }

    pub fn surface(mut self, question: &str, span: TextSpan) -> Self {
        self.surface = span_text(question, span);
        self.span = Some(span);
        self
    }

    pub fn expected(mut self, expected: impl Into<String>) -> Self {
        self.expected = Some(clamp_chars(expected.into(), MAX_DETAIL_CHARS));
        self
    }

    pub fn candidates<I, S>(mut self, candidates: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut seen = BTreeSet::new();
        for candidate in candidates {
            let candidate = candidate.into();
            let candidate = candidate.trim();
            if candidate.is_empty() {
                continue;
            }
            seen.insert(candidate.to_string());
            if seen.len() >= MAX_CANDIDATES {
                break;
            }
        }
        self.candidates = seen.into_iter().collect();
        self
    }

    pub fn is_blocking(&self) -> bool {
        self.severity == ResidualSeverity::Blocking
    }
}

/// 一次读法的残差表。`residuals` 有界、去重，顺序稳定（插入序）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Reading {
    pub version: u32,
    pub question_chars: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan_kind: Option<String>,
    pub residuals: Vec<Residual>,
}

impl Default for Reading {
    fn default() -> Self {
        Self {
            version: READING_VERSION,
            question_chars: 0,
            plan_kind: None,
            residuals: Vec::new(),
        }
    }
}

impl Reading {
    pub fn new(question_chars: usize, plan: Option<&QueryPlan>) -> Self {
        Self {
            version: READING_VERSION,
            question_chars,
            plan_kind: plan.map(|plan| QueryPlanner::kind(plan).to_string()),
            residuals: Vec::new(),
        }
    }

    /// 追加残差：同一（类别、阶段、区间、局部文本、detail）只保留一次；
    /// 达到上限后丢弃后续项并保持确定性（不替换已有项）。
    pub fn push(&mut self, residual: Residual) {
        if self.residuals.iter().any(|existing| {
            existing.kind == residual.kind
                && existing.stage == residual.stage
                && existing.span == residual.span
                && existing.surface == residual.surface
                && existing.detail == residual.detail
        }) {
            return;
        }
        if self.residuals.len() < MAX_RESIDUALS {
            self.residuals.push(residual);
        }
    }

    pub fn extend<I>(&mut self, residuals: I)
    where
        I: IntoIterator<Item = Residual>,
    {
        for residual in residuals {
            self.push(residual);
        }
    }

    /// 读法状态：有阻断残差 → Partial；没有计划 → Unreadable；否则 Ready。
    pub fn status(&self) -> ReadingStatus {
        if self.plan_kind.is_none()
            && self.residuals.iter().any(|residual| {
                residual.is_blocking()
                    && matches!(residual.stage, ResidualStage::Input | ResidualStage::Plan)
            })
        {
            return ReadingStatus::Unreadable;
        }
        if self.residuals.iter().any(Residual::is_blocking) {
            return ReadingStatus::Partial;
        }
        if self.plan_kind.is_none()
            && self.residuals.iter().any(|residual| {
                matches!(
                    residual.kind,
                    ResidualKind::Ellipsis
                        | ResidualKind::MissingEntity
                        | ResidualKind::MissingPredicate
                        | ResidualKind::UnsupportedQuestion
                        | ResidualKind::NoQuestionShape
                )
            })
        {
            return ReadingStatus::Unreadable;
        }
        ReadingStatus::Ready
    }

    pub fn is_ready(&self) -> bool {
        self.status().is_ready()
    }

    pub fn blocking_count(&self) -> usize {
        self.residuals
            .iter()
            .filter(|residual| residual.is_blocking())
            .count()
    }

    /// 稳定回执字段：类别/阶段/修复动作排序去重，不含原文。
    pub fn residual_kinds(&self) -> Vec<String> {
        let mut kinds: Vec<String> = self
            .residuals
            .iter()
            .map(|residual| residual.kind.code().to_string())
            .collect();
        kinds.sort();
        kinds.dedup();
        kinds
    }

    pub fn residual_stages(&self) -> Vec<String> {
        let mut stages: Vec<String> = self
            .residuals
            .iter()
            .map(|residual| residual.stage.code().to_string())
            .collect();
        stages.sort();
        stages.dedup();
        stages
    }

    pub fn repair_kinds(&self) -> Vec<String> {
        let mut repairs: Vec<String> = self
            .residuals
            .iter()
            .map(|residual| residual.repair.code().to_string())
            .filter(|repair| repair != RepairKind::None.code())
            .collect();
        repairs.sort();
        repairs.dedup();
        repairs
    }

    /// 下一步修复建议：只取第一个阻断残差，没有阻断时取第一个警告。
    pub fn next_repair(&self) -> Option<&Residual> {
        self.residuals
            .iter()
            .find(|residual| residual.severity == ResidualSeverity::Blocking)
            .or_else(|| {
                self.residuals
                    .iter()
                    .find(|residual| residual.severity == ResidualSeverity::Warning)
            })
    }

    /// 本地渲染（可含原句局部文本）。回执只使用上面的稳定字段。
    pub fn render_lines(&self) -> Vec<String> {
        let mut lines = Vec::new();
        lines.push(format!(
            "读法状态：{}（计划 {}，残差 {}，阻断 {}）。",
            self.status().label(),
            self.plan_kind.as_deref().unwrap_or("无"),
            self.residuals.len(),
            self.blocking_count()
        ));
        for residual in &self.residuals {
            let location = match (residual.span, residual.surface.as_deref()) {
                (Some(span), Some(surface)) if !surface.is_empty() => {
                    format!("{}..{}「{}」", span.start, span.end, surface)
                }
                (Some(span), _) => format!("{}..{}", span.start, span.end),
                _ => "无位置".to_string(),
            };
            let expected = residual
                .expected
                .as_deref()
                .map(|expected| format!("；期待 {}", expected))
                .unwrap_or_default();
            let candidates = if residual.candidates.is_empty() {
                String::new()
            } else {
                format!("；候选 {}", residual.candidates.join("、"))
            };
            lines.push(format!(
                "  • [{}] {} {}（{}）{}{}；修复：{}；{}",
                residual.stage.code(),
                residual.kind.label(),
                location,
                residual.severity.label(),
                expected,
                candidates,
                residual.repair.label(),
                residual.detail
            ));
        }
        lines
    }
}

/// 句面诊断：只根据文本与已有计划分类，不访问知识库。
pub fn diagnose(question: &str, plan: Option<&QueryPlan>) -> Reading {
    let trimmed = question.trim();
    let chars = trimmed.chars().count();
    let mut reading = Reading::new(chars, plan);

    if trimmed.is_empty() {
        reading.push(
            Residual::new(
                ResidualKind::EmptyInput,
                ResidualStage::Input,
                ResidualSeverity::Blocking,
                RepairKind::Abstain,
                "问句为空，没有可读内容",
            )
            .at(TextSpan::new(0, 0)),
        );
        return reading;
    }
    if chars > MAX_QUESTION_CHARS {
        reading.push(
            Residual::new(
                ResidualKind::InputTooLong,
                ResidualStage::Input,
                ResidualSeverity::Blocking,
                RepairKind::ShortenInput,
                format!("问句 {} 字，超过 {} 字上限", chars, MAX_QUESTION_CHARS),
            )
            .surface(trimmed, TextSpan::new(0, chars)),
        );
        return reading;
    }

    if plan.is_some() {
        return reading;
    }

    if let Some((span, detail)) = ellipsis_span(trimmed) {
        reading.push(
            Residual::new(
                ResidualKind::Ellipsis,
                ResidualStage::Plan,
                ResidualSeverity::Blocking,
                RepairKind::BindEllipsis,
                detail,
            )
            .surface(trimmed, span)
            .expected("上一轮的谓词/属性"),
        );
        return reading;
    }

    if let Some((span, prefix)) = attribute_only_span(trimmed) {
        reading.push(
            Residual::new(
                ResidualKind::MissingEntity,
                ResidualStage::Plan,
                ResidualSeverity::Blocking,
                RepairKind::AskClarification,
                "只给出了属性，没有给出被问的实体",
            )
            .surface(trimmed, span)
            .expected(format!("已登记实体（属性：{}）", prefix)),
        );
        return reading;
    }

    if let Some((marker, span)) = find_marker(
        trimmed,
        &[
            "为什么",
            "怎么",
            "如何",
            "原因",
            "解释",
            "区别",
            "原理",
            "证明",
        ],
    ) {
        reading.push(
            Residual::new(
                ResidualKind::UnsupportedQuestion,
                ResidualStage::Plan,
                ResidualSeverity::Blocking,
                RepairKind::RewriteQuestion,
                "当前符号管线只支持事实/关系查询，不支持开放式解释问法",
            )
            .surface(trimmed, span)
            .expected(format!("可执行的事实查询（触发词：{}）", marker)),
        );
        return reading;
    }

    if trimmed.ends_with("是谁") && !trimmed.starts_with("谁") {
        let span = TextSpan::new(0, chars);
        reading.push(
            Residual::new(
                ResidualKind::UnsupportedQuestion,
                ResidualStage::Plan,
                ResidualSeverity::Blocking,
                RepairKind::RewriteQuestion,
                "「X 是谁」是定义/介绍问句，不是图谱中的关系查询",
            )
            .surface(trimmed, span)
            .expected("「X 的 <属性> 是什么」或「X 和 Y 是什么关系」"),
        );
        return reading;
    }

    if let Some((marker, span)) = find_marker(
        trimmed,
        &["吗", "是不是", "有没有", "是否", "吧"],
    ) {
        reading.push(
            Residual::new(
                ResidualKind::MissingPredicate,
                ResidualStage::Plan,
                ResidualSeverity::Blocking,
                RepairKind::AddParticle,
                "是非问缺少可校验的谓词或关系",
            )
            .surface(trimmed, span)
            .expected("「X 的 <属性> 是 Y 吗」或「X <关系> Y 吗」"),
        );
        return reading;
    }

    if let Some((marker, span)) = find_marker(
        trimmed,
        &["谁", "什么", "哪个", "哪家", "哪里", "多少", "几", "何人", "啥"],
    ) {
        reading.push(
            Residual::new(
                ResidualKind::MissingPredicate,
                ResidualStage::Plan,
                ResidualSeverity::Blocking,
                RepairKind::SupplyPredicate,
                "有疑问词，但没有找到可执行的谓词槽",
            )
            .surface(trimmed, span)
            .expected(format!("疑问词「{}」所问的谓词", marker)),
        );
        return reading;
    }

    let span = TextSpan::new(0, chars);
    reading.push(
        Residual::new(
            ResidualKind::NoQuestionShape,
            ResidualStage::Plan,
            ResidualSeverity::Blocking,
            RepairKind::RewriteQuestion,
            "查询规划器没有找到可执行形状",
        )
        .surface(trimmed, span)
        .expected("实体 + 谓词 + 疑问位置"),
    );
    reading
}

/// 「李四呢 / 那李四呢」：整句省略，缺的是谓词而不是实体。
fn ellipsis_span(question: &str) -> Option<(TextSpan, String)> {
    let core = question.trim_end_matches(['？', '?', '。', '！', '!', ' ']);
    let core = core.strip_suffix('呢')?.trim_end();
    let chars = core.chars().count();
    if chars == 0 || chars > MAX_ELLIPSIS_CORE_CHARS {
        return None;
    }
    if core.chars().any(char::is_whitespace) || core.chars().any(|ch| "？?。！!，,；;".contains(ch))
    {
        return None;
    }
    if ["谁", "什么", "哪个", "多少", "几", "为什么", "怎么"]
        .iter()
        .any(|word| core.contains(word))
    {
        return None;
    }
    Some((
        TextSpan::new(0, chars),
        "省略问句只保留了话题，缺少本轮要问的谓词/属性".to_string(),
    ))
}

/// 「体重是多少 / 身高呢」这类只给属性、不给实体的问句。
fn attribute_only_span(question: &str) -> Option<(TextSpan, String)> {
    let body = question.trim_end_matches(['？', '?', '。', '！', '!', ' ']);
    for tail in ["是多少", "是什么", "为多少", "有多少", "是几"] {
        if let Some(prefix) = body.strip_suffix(tail) {
            let prefix = prefix.trim();
            if is_attribute_hint(prefix) {
                let span = TextSpan::new(0, prefix.chars().count());
                return Some((span, prefix.to_string()));
            }
        }
    }
    None
}

fn is_attribute_hint(text: &str) -> bool {
    const ATTRIBUTES: [&str; 28] = [
        "身高",
        "体重",
        "年龄",
        "岁数",
        "生日",
        "出生日期",
        "出生地",
        "籍贯",
        "国籍",
        "民族",
        "职业",
        "职位",
        "性别",
        "电话",
        "手机",
        "邮箱",
        "地址",
        "价格",
        "颜色",
        "品牌",
        "型号",
        "身高体重",
        "成立时间",
        "成立年份",
        "创始人",
        "员工数",
        "人口",
        "面积",
    ];
    !text.is_empty()
        && !text.contains('的')
        && !text.chars().any(char::is_whitespace)
        && ATTRIBUTES.contains(&text)
}

fn find_marker<'a>(text: &'a str, markers: &[&'a str]) -> Option<(&'a str, TextSpan)> {
    let mut best: Option<(&str, TextSpan)> = None;
    for marker in markers {
        if let Some(byte_index) = text.find(marker) {
            let start = text[..byte_index].chars().count();
            let span = TextSpan::new(start, start + marker.chars().count());
            match best {
                Some((_, current)) if current.start <= span.start => {}
                _ => best = Some((marker, span)),
            }
        }
    }
    best
}

fn span_text(question: &str, span: TextSpan) -> Option<String> {
    if span.start >= span.end {
        return Some(String::new());
    }
    let text: String = question
        .chars()
        .skip(span.start)
        .take(span.chars())
        .collect();
    let text = text.trim();
    if text.is_empty() {
        None
    } else {
        Some(clamp_chars(text.to_string(), MAX_DETAIL_CHARS))
    }
}

fn clamp_chars(text: String, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text;
    }
    let mut clamped: String = text.chars().take(limit.saturating_sub(1)).collect();
    clamped.push('…');
    clamped
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query_planner::QueryPlanner;

    #[test]
    fn empty_and_overlong_inputs_are_blocking_with_spans() {
        let empty = diagnose("   ", None);
        assert_eq!(empty.status(), ReadingStatus::Unreadable);
        assert_eq!(empty.residuals[0].kind, ResidualKind::EmptyInput);
        assert_eq!(empty.residuals[0].span, Some(TextSpan::new(0, 0)));

        let long = "张".repeat(MAX_QUESTION_CHARS + 1);
        let reading = diagnose(&long, None);
        assert_eq!(reading.residuals[0].kind, ResidualKind::InputTooLong);
        assert_eq!(reading.residuals[0].repair, RepairKind::ShortenInput);
        assert_eq!(reading.residuals[0].span, Some(TextSpan::new(0, long.chars().count())));
    }

    #[test]
    fn ellipsis_keeps_topic_span_and_repair_action() {
        let reading = diagnose("李四呢", None);
        assert_eq!(reading.status(), ReadingStatus::Unreadable);
        let residual = &reading.residuals[0];
        assert_eq!(residual.kind, ResidualKind::Ellipsis);
        assert_eq!(residual.stage, ResidualStage::Plan);
        assert_eq!(residual.span, Some(TextSpan::new(0, 2)));
        assert_eq!(residual.surface.as_deref(), Some("李四"));
        assert_eq!(residual.repair, RepairKind::BindEllipsis);
        assert_eq!(residual.expected.as_deref(), Some("上一轮的谓词/属性"));
    }

    #[test]
    fn attribute_only_question_asks_for_entity_with_span() {
        let reading = diagnose("体重是多少", None);
        let residual = &reading.residuals[0];
        assert_eq!(residual.kind, ResidualKind::MissingEntity);
        assert_eq!(residual.span, Some(TextSpan::new(0, 2)));
        assert_eq!(residual.surface.as_deref(), Some("体重"));
        assert_eq!(residual.repair, RepairKind::AskClarification);
        assert!(residual.expected.as_deref().unwrap_or_default().contains("实体"));
    }

    #[test]
    fn explanatory_and_definition_questions_are_distinguished() {
        let why = diagnose("为什么天是蓝的", None);
        assert_eq!(why.residuals[0].kind, ResidualKind::UnsupportedQuestion);
        assert_eq!(why.residuals[0].repair, RepairKind::RewriteQuestion);
        assert_eq!(why.residuals[0].span, Some(TextSpan::new(0, 3)));

        let who = diagnose("秦始皇是谁", None);
        assert_eq!(who.residuals[0].kind, ResidualKind::UnsupportedQuestion);
        assert_eq!(who.residuals[0].span, Some(TextSpan::new(0, 5)));
    }

    #[test]
    fn planned_degree_question_is_ready_and_keeps_plan_kind() {
        let plan = QueryPlanner::new()
            .parse("姚明多高")
            .expect("程度问句现在应可编译");
        let reading = diagnose("姚明多高", Some(&plan));
        assert_eq!(reading.status(), ReadingStatus::Ready);
        assert!(reading.residuals.is_empty());
        assert_eq!(reading.plan_kind.as_deref(), Some("FindObject"));
    }

    #[test]
    fn duplicate_and_excess_residuals_are_bounded() {
        let mut reading = Reading::default();
        for _ in 0..5 {
            reading.push(Residual::new(
                ResidualKind::UnknownEntity,
                ResidualStage::Link,
                ResidualSeverity::Blocking,
                RepairKind::ChooseCandidate,
                "同一残差",
            ));
        }
        assert_eq!(reading.residuals.len(), 1);

        for index in 0..(MAX_RESIDUALS + 4) {
            reading.push(
                Residual::new(
                    ResidualKind::UnknownPredicate,
                    ResidualStage::Link,
                    ResidualSeverity::Warning,
                    RepairKind::SupplyPredicate,
                    format!("不同残差 {}", index),
                )
                .at(TextSpan::new(index, index + 1)),
            );
        }
        assert_eq!(reading.residuals.len(), MAX_RESIDUALS);
    }

    #[test]
    fn candidates_are_sorted_deduplicated_and_bounded() {
        let residual = Residual::new(
            ResidualKind::AmbiguousEntity,
            ResidualStage::Link,
            ResidualSeverity::Blocking,
            RepairKind::ChooseCandidate,
            "候选",
        )
        .candidates(["乙", "甲", "乙", "", "丙", "丁", "戊", "己"]);
        assert_eq!(residual.candidates, vec!["丁", "丙", "乙", "戊", "甲"]);
        assert_eq!(residual.candidates.len(), MAX_CANDIDATES);
    }

    #[test]
    fn receipt_fields_are_stable_and_text_free() {
        let reading = diagnose("李四呢", None);
        assert_eq!(reading.residual_kinds(), vec!["ellipsis"]);
        assert_eq!(reading.residual_stages(), vec!["plan"]);
        assert_eq!(reading.repair_kinds(), vec!["bind_ellipsis"]);
        assert_eq!(reading.blocking_count(), 1);
        assert!(reading
            .render_lines()
            .iter()
            .any(|line| line.contains("读法状态：不可读")));
    }
}

//! 溯因式重写-检索互证（Abductive Rewrite–Retrieval，ARR）。
//!
//! 查询重写器、指称锚定器与精确规划器都能为同一问句提出「读法」；稠密检索
//! 器已经能给出可解释的证据分数。本模块把两者接成一个裁决层：
//!
//! 1. **假设生成**：每个候选读法是一个 `(来源, 规则, 逻辑计划)`；
//! 2. **证据探针**：把计划翻译成精确约束（主体/谓词/客体），向证据索引
//!    取回带分数的三元组；
//! 3. **四信号评估**：支持度（最强证据）、覆盖度（主答案簇占比）、边际
//!    （第一与第二答案的差距）、一致性（不同探针是否指向同一答案）；
//! 4. **三档决策**：`Accept`（领先且证据足）、`Clarify`（多个读法都像）、
//!    `Abstain`（没有读法过闸门）。
//!
//! 关键约束与既有软路由一致：
//!
//! - 纯符号、无训练参数；所有分数都是可复现的规则分；
//! - 默认关闭，调用方只在显式启用时使用；
//! - 不执行计划、不写知识库；只对候选读法做证据裁决；
//! - 所有决策都带可打印回执，能逐项复核「为什么选它」。

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::dense_retrieval::{EvidenceSearch, RetrievalHit, SearchConstraints};
use crate::query_planner::{QueryPlan, QueryPlanner};

/// 报告方言：读取方据此拒绝把旧版报告当新版解释。
pub const DIALECT: &str = "knowprolog-rewrite-retrieval/1";

/// 候选读法的来源。先验只反映「该来源在精确路径上的可靠度」，不是学习权重。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HypothesisSource {
    /// QueryPlanner 精确编译出的计划。
    ExactPlan,
    /// query_rewrite 句法重写编译出的计划。
    SyntaxRewrite,
    /// query_anchor 指称锚定后编译出的计划。
    AnchoredPlan,
    /// rewrite_book 中已显式采纳的学习规则。
    LearnedRewrite,
    /// 上下文帧补全的省略读法。
    ContextFrame,
    /// 调用方手工注入的候选（评测/调试）。
    Manual,
}

impl HypothesisSource {
    pub fn label(self) -> &'static str {
        match self {
            HypothesisSource::ExactPlan => "精确计划",
            HypothesisSource::SyntaxRewrite => "句法重写",
            HypothesisSource::AnchoredPlan => "指称锚定",
            HypothesisSource::LearnedRewrite => "学习规则",
            HypothesisSource::ContextFrame => "上下文补全",
            HypothesisSource::Manual => "手工候选",
        }
    }

    /// 来源先验：只用于 log-odds 起点，证据可以推翻它。
    pub fn prior(self) -> f64 {
        match self {
            HypothesisSource::ExactPlan => 0.72,
            HypothesisSource::SyntaxRewrite => 0.62,
            HypothesisSource::AnchoredPlan => 0.58,
            HypothesisSource::LearnedRewrite => 0.60,
            HypothesisSource::ContextFrame => 0.55,
            HypothesisSource::Manual => 0.65,
        }
    }

    /// 同一计划被多个来源提出时，保留更可靠的那个来源。
    fn priority(self) -> u8 {
        match self {
            HypothesisSource::ExactPlan => 6,
            HypothesisSource::Manual => 5,
            HypothesisSource::SyntaxRewrite => 4,
            HypothesisSource::LearnedRewrite => 3,
            HypothesisSource::AnchoredPlan => 2,
            HypothesisSource::ContextFrame => 1,
        }
    }
}

/// 一个待裁决的读法。
#[derive(Debug, Clone)]
pub struct HypothesisCandidate {
    pub source: HypothesisSource,
    /// 稳定的规则 id（精确计划用 `planner`，重写用规则 id，锚定用变体规则）。
    pub rule: String,
    /// 该读法对应的问句（锚定后可能是替换过实体的变体）。
    pub question: String,
    pub plan: QueryPlan,
    /// 调用方注入的上下文先验（log-odds 加项；例如焦点实体/上一轮谓词连续）。
    pub context_bonus: f64,
}

impl Serialize for HypothesisCandidate {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("HypothesisCandidate", 6)?;
        state.serialize_field("source", &self.source)?;
        state.serialize_field("rule", &self.rule)?;
        state.serialize_field("question", &self.question)?;
        state.serialize_field("plan_kind", QueryPlanner::kind(&self.plan))?;
        state.serialize_field("plan", &QueryPlanner::format_plan(&self.plan))?;
        state.serialize_field("context_bonus", &self.context_bonus)?;
        state.end()
    }
}

impl HypothesisCandidate {
    pub fn new(
        source: HypothesisSource,
        rule: impl Into<String>,
        question: impl Into<String>,
        plan: QueryPlan,
    ) -> Self {
        Self {
            source,
            rule: rule.into(),
            question: question.into(),
            plan,
            context_bonus: 0.0,
        }
    }

    /// 注入上下文先验（log-odds 加项，内部会夹在 ±2 之间）。
    pub fn with_context_bonus(mut self, bonus: f64) -> Self {
        self.context_bonus = bonus;
        self
    }

    /// 去重键：同一逻辑计划只让最可靠的来源参赛。
    pub fn key(&self) -> String {
        format!(
            "{}|{}",
            QueryPlanner::kind(&self.plan),
            QueryPlanner::format_plan(&self.plan)
        )
    }

    pub fn receipt(&self) -> String {
        format!(
            "[{}] {}：{} → {}",
            self.source.label(),
            self.rule,
            QueryPlanner::kind(&self.plan),
            QueryPlanner::format_plan(&self.plan)
        )
    }
}

/// 探针要回答的角色：正向查对象、反向查主体、校验事实。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProbeRole {
    Object,
    Subject,
    Truth,
}

impl ProbeRole {
    pub fn label(self) -> &'static str {
        match self {
            ProbeRole::Object => "问对象",
            ProbeRole::Subject => "问主体",
            ProbeRole::Truth => "校验",
        }
    }

    fn answer_from(self, hit: &RetrievalHit) -> String {
        match self {
            ProbeRole::Object => hit.object.clone(),
            ProbeRole::Subject => hit.subject.clone(),
            ProbeRole::Truth => "是".to_string(),
        }
    }
}

/// 从逻辑计划导出的一个检索探针。
#[derive(Debug, Clone, Serialize)]
pub struct Probe {
    pub label: String,
    pub query: String,
    pub constraints: SearchConstraints,
    pub role: ProbeRole,
    /// 计划还有后续跳时，本探针只验证第一跳（置信度会打折）。
    pub partial: bool,
}

/// 把逻辑计划压成精确约束探针；无法从证据索引验证的计划返回空列表。
///
/// 这里刻意不执行完整计划：证据只回答「这个读法的第一跳有没有知识支持」，
/// 最终答案仍由 `SymbolicReasoner::answer_plan` 执行并核查。
pub fn probes_for(plan: &QueryPlan) -> Vec<Probe> {
    fn probe(
        label: impl Into<String>,
        query: impl Into<String>,
        constraints: SearchConstraints,
        role: ProbeRole,
        partial: bool,
    ) -> Probe {
        Probe {
            label: label.into(),
            query: query.into(),
            constraints,
            role,
            partial,
        }
    }

    match plan {
        QueryPlan::FindObjectByPredicate { subject, predicate } => vec![probe(
            "正向关系",
            format!("{} {}", subject, predicate),
            SearchConstraints {
                subject: Some(subject.clone()),
                predicate: Some(predicate.clone()),
                ..SearchConstraints::default()
            },
            ProbeRole::Object,
            false,
        )],
        QueryPlan::FindSubject { object, predicate } => vec![probe(
            "反向关系",
            format!("{} {}", object, predicate),
            SearchConstraints {
                predicate: Some(predicate.clone()),
                object: Some(object.clone()),
                ..SearchConstraints::default()
            },
            ProbeRole::Subject,
            false,
        )],
        QueryPlan::CheckRelation {
            subject,
            predicate,
            object,
            ..
        } => vec![probe(
            "关系校验",
            format!("{} {} {}", subject, predicate, object),
            SearchConstraints {
                subject: Some(subject.clone()),
                predicate: Some(predicate.clone()),
                object: Some(object.clone()),
                ..SearchConstraints::default()
            },
            ProbeRole::Truth,
            false,
        )],
        QueryPlan::CheckTriple {
            subject_segments,
            object,
        } => {
            if subject_segments.len() < 2 {
                return Vec::new();
            }
            let subject = subject_segments.first().cloned().unwrap_or_default();
            let predicate = subject_segments.last().cloned().unwrap_or_default();
            vec![probe(
                "链式校验",
                format!("{} {} {}", subject, predicate, object),
                SearchConstraints {
                    subject: Some(subject),
                    predicate: Some(predicate),
                    object: Some(object.clone()),
                    ..SearchConstraints::default()
                },
                ProbeRole::Truth,
                subject_segments.len() > 2,
            )]
        }
        QueryPlan::Count { subject, predicate } | QueryPlan::Exist { subject, predicate } => {
            vec![probe(
                "存在/计数",
                format!("{} {}", subject, predicate),
                SearchConstraints {
                    subject: Some(subject.clone()),
                    predicate: Some(predicate.clone()),
                    ..SearchConstraints::default()
                },
                ProbeRole::Object,
                false,
            )]
        }
        QueryPlan::Closure {
            subject, predicate, ..
        } => vec![probe(
            "闭包首跳",
            format!("{} {}", subject, predicate),
            SearchConstraints {
                subject: Some(subject.clone()),
                predicate: Some(predicate.clone()),
                ..SearchConstraints::default()
            },
            ProbeRole::Object,
            true,
        )],
        QueryPlan::FindObject { segments } => {
            if segments.len() < 2 {
                return Vec::new();
            }
            vec![probe(
                "链式首跳",
                format!("{} {}", segments[0], segments[1]),
                SearchConstraints {
                    subject: Some(segments[0].clone()),
                    predicate: Some(segments[1].clone()),
                    ..SearchConstraints::default()
                },
                ProbeRole::Object,
                segments.len() > 2,
            )]
        }
        QueryPlan::FindObjectExcept {
            subject, predicate, ..
        } => vec![probe(
            "正向关系（排除）",
            format!("{} {}", subject, predicate),
            SearchConstraints {
                subject: Some(subject.clone()),
                predicate: Some(predicate.clone()),
                ..SearchConstraints::default()
            },
            ProbeRole::Object,
            false,
        )],
        QueryPlan::FindSubjectExcept {
            object, predicate, ..
        } => vec![probe(
            "反向关系（排除）",
            format!("{} {}", object, predicate),
            SearchConstraints {
                predicate: Some(predicate.clone()),
                object: Some(object.clone()),
                ..SearchConstraints::default()
            },
            ProbeRole::Subject,
            false,
        )],
        QueryPlan::PathBetween { .. }
        | QueryPlan::CommonNeighbor { .. }
        | QueryPlan::Intermediary { .. }
        | QueryPlan::Similar { .. }
        | QueryPlan::Analogy { .. } => Vec::new(),
    }
}

/// 计划中出现的实体词面（用于上下文连续性先验）。
pub fn plan_entities(plan: &QueryPlan) -> Vec<&str> {
    match plan {
        QueryPlan::FindObject { segments } => segments.iter().map(String::as_str).collect(),
        QueryPlan::FindSubject { object, .. } => vec![object.as_str()],
        QueryPlan::FindObjectByPredicate { subject, .. } => vec![subject.as_str()],
        QueryPlan::FindObjectExcept {
            subject, excluded, ..
        } => std::iter::once(subject.as_str())
            .chain(excluded.iter().map(String::as_str))
            .collect(),
        QueryPlan::FindSubjectExcept {
            object, excluded, ..
        } => std::iter::once(object.as_str())
            .chain(excluded.iter().map(String::as_str))
            .collect(),
        QueryPlan::PathBetween { start, goal, .. }
        | QueryPlan::Intermediary { start, goal, .. } => vec![start.as_str(), goal.as_str()],
        QueryPlan::CommonNeighbor { left, right } => vec![left.as_str(), right.as_str()],
        QueryPlan::CheckTriple {
            subject_segments,
            object,
        } => subject_segments
            .iter()
            .map(String::as_str)
            .chain(std::iter::once(object.as_str()))
            .collect(),
        QueryPlan::CheckRelation {
            subject, object, ..
        } => vec![subject.as_str(), object.as_str()],
        QueryPlan::Count { subject, .. }
        | QueryPlan::Exist { subject, .. }
        | QueryPlan::Closure { subject, .. } => vec![subject.as_str()],
        QueryPlan::Similar { entity } => vec![entity.as_str()],
        QueryPlan::Analogy { a, b, c } => vec![a.as_str(), b.as_str(), c.as_str()],
    }
}

/// 计划中出现的谓词词面（用于上下文连续性先验）。
pub fn plan_predicates(plan: &QueryPlan) -> Vec<&str> {
    match plan {
        QueryPlan::FindObject { segments } => {
            segments.iter().skip(1).map(String::as_str).collect()
        }
        QueryPlan::FindSubject { predicate, .. }
        | QueryPlan::FindObjectByPredicate { predicate, .. }
        | QueryPlan::FindObjectExcept { predicate, .. }
        | QueryPlan::FindSubjectExcept { predicate, .. }
        | QueryPlan::CheckRelation { predicate, .. }
        | QueryPlan::Count { predicate, .. }
        | QueryPlan::Exist { predicate, .. }
        | QueryPlan::Closure { predicate, .. } => vec![predicate.as_str()],
        QueryPlan::CheckTriple {
            subject_segments, ..
        } => subject_segments
            .last()
            .map(|predicate| vec![predicate.as_str()])
            .unwrap_or_default(),
        QueryPlan::PathBetween { .. }
        | QueryPlan::CommonNeighbor { .. }
        | QueryPlan::Intermediary { .. }
        | QueryPlan::Similar { .. }
        | QueryPlan::Analogy { .. } => Vec::new(),
    }
}

fn default_max_hypotheses() -> usize {
    4
}

fn default_max_probes() -> usize {
    2
}

fn default_per_probe_k() -> usize {
    4
}

fn default_accept_threshold() -> f64 {
    0.72
}

fn default_clarify_threshold() -> f64 {
    0.48
}

fn default_accept_margin() -> f64 {
    0.08
}

fn default_min_support() -> f64 {
    0.18
}

fn default_min_coverage() -> f64 {
    0.55
}

fn default_min_margin() -> f64 {
    0.08
}

fn default_support_pivot() -> f64 {
    0.25
}

fn default_coverage_pivot() -> f64 {
    0.60
}

fn default_margin_pivot() -> f64 {
    0.30
}

fn default_consistency_pivot() -> f64 {
    0.60
}

fn default_support_gain() -> f64 {
    3.0
}

fn default_coverage_gain() -> f64 {
    1.8
}

fn default_margin_gain() -> f64 {
    1.2
}

fn default_consistency_gain() -> f64 {
    1.0
}

fn default_partial_discount() -> f64 {
    0.65
}

/// 重写-检索裁决参数。所有阈值都是显式先验，不做在线学习。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RewriteRetrievalConfig {
    /// 是否允许回答路径使用本模块；默认关闭，关闭时旧路径逐字节不变。
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_max_hypotheses")]
    pub max_hypotheses: usize,
    #[serde(default = "default_max_probes")]
    pub max_probes: usize,
    #[serde(default = "default_per_probe_k")]
    pub per_probe_k: usize,
    #[serde(default = "default_accept_threshold")]
    pub accept_threshold: f64,
    #[serde(default = "default_clarify_threshold")]
    pub clarify_threshold: f64,
    /// 接受时与「不同答案」的最佳竞争读法所需的最小置信度差。
    #[serde(default = "default_accept_margin")]
    pub accept_margin: f64,
    #[serde(default = "default_min_support")]
    pub min_support: f64,
    #[serde(default = "default_min_coverage")]
    pub min_coverage: f64,
    #[serde(default = "default_min_margin")]
    pub min_margin: f64,
    #[serde(default = "default_support_pivot")]
    pub support_pivot: f64,
    #[serde(default = "default_coverage_pivot")]
    pub coverage_pivot: f64,
    #[serde(default = "default_margin_pivot")]
    pub margin_pivot: f64,
    #[serde(default = "default_consistency_pivot")]
    pub consistency_pivot: f64,
    #[serde(default = "default_support_gain")]
    pub support_gain: f64,
    #[serde(default = "default_coverage_gain")]
    pub coverage_gain: f64,
    #[serde(default = "default_margin_gain")]
    pub margin_gain: f64,
    #[serde(default = "default_consistency_gain")]
    pub consistency_gain: f64,
    /// 只验证第一跳时对证据分的折扣。
    #[serde(default = "default_partial_discount")]
    pub partial_discount: f64,
}

impl Default for RewriteRetrievalConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_hypotheses: default_max_hypotheses(),
            max_probes: default_max_probes(),
            per_probe_k: default_per_probe_k(),
            accept_threshold: default_accept_threshold(),
            clarify_threshold: default_clarify_threshold(),
            accept_margin: default_accept_margin(),
            min_support: default_min_support(),
            min_coverage: default_min_coverage(),
            min_margin: default_min_margin(),
            support_pivot: default_support_pivot(),
            coverage_pivot: default_coverage_pivot(),
            margin_pivot: default_margin_pivot(),
            consistency_pivot: default_consistency_pivot(),
            support_gain: default_support_gain(),
            coverage_gain: default_coverage_gain(),
            margin_gain: default_margin_gain(),
            consistency_gain: default_consistency_gain(),
            partial_discount: default_partial_discount(),
        }
    }
}

impl RewriteRetrievalConfig {
    fn clamped(&self) -> Self {
        let mut config = self.clone();
        config.max_hypotheses = config.max_hypotheses.clamp(1, 8);
        config.max_probes = config.max_probes.clamp(1, 4);
        config.per_probe_k = config.per_probe_k.clamp(1, 20);
        config.partial_discount = config.partial_discount.clamp(0.0, 1.0);
        config
    }
}

/// 一个探针的检索回执。
#[derive(Debug, Clone, Serialize)]
pub struct ProbeReport {
    pub label: String,
    pub role: ProbeRole,
    pub query: String,
    pub partial: bool,
    pub hits: usize,
    pub top_score: f64,
    pub certified_top_k: bool,
    pub top_hits: Vec<RetrievalHit>,
    pub notes: Vec<String>,
}

/// 一个答案簇：同一读法下多条证据指向的同一答案。
#[derive(Debug, Clone, Serialize)]
pub struct AnswerCluster {
    pub answer: String,
    pub weight: f64,
    pub best_score: f64,
    pub hits: usize,
}

/// 单个读法的证据评估。
#[derive(Debug, Clone, Serialize)]
pub struct HypothesisAssessment {
    pub candidate: HypothesisCandidate,
    pub probes: Vec<ProbeReport>,
    pub support: f64,
    pub coverage: f64,
    pub margin: f64,
    pub consistency: f64,
    pub confidence: f64,
    pub answer: Option<String>,
    pub answers: Vec<AnswerCluster>,
    pub notes: Vec<String>,
}

impl HypothesisAssessment {
    pub fn receipt(&self) -> String {
        format!(
            "{}；置信度 {:.3}（支持 {:.3}，覆盖 {:.3}，边际 {:.3}，一致 {:.3}）{}",
            self.candidate.receipt(),
            self.confidence,
            self.support,
            self.coverage,
            self.margin,
            self.consistency,
            self.answer
                .as_ref()
                .map(|answer| format!("；答案「{}」", answer))
                .unwrap_or_default()
        )
    }
}

/// 三档裁决。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Decision {
    Accept,
    Clarify,
    Abstain,
}

impl Decision {
    pub fn label(self) -> &'static str {
        match self {
            Decision::Accept => "接受",
            Decision::Clarify => "澄清",
            Decision::Abstain => "弃权",
        }
    }
}

/// 一次读法裁决的完整回执。
#[derive(Debug, Clone, Serialize)]
pub struct Assessment {
    pub dialect: String,
    pub question: String,
    pub decision: Decision,
    /// 接受时指向 `hypotheses` 中胜出的读法。
    pub selected: Option<usize>,
    pub hypotheses: Vec<HypothesisAssessment>,
    pub notes: Vec<String>,
}

impl Assessment {
    pub fn selected_hypothesis(&self) -> Option<&HypothesisAssessment> {
        self.selected.and_then(|index| self.hypotheses.get(index))
    }

    pub fn answer(&self) -> Option<&str> {
        self.selected_hypothesis()
            .and_then(|hypothesis| hypothesis.answer.as_deref())
    }

    /// 可打印报告：每个读法的四信号、答案簇与探针回执。
    pub fn render_report(&self) -> String {
        let mut output = format!(
            "读法裁决（{}）：{}\n决策：{}",
            self.dialect,
            self.question,
            self.decision.label()
        );
        if let Some(selected) = self.selected_hypothesis() {
            output.push_str(&format!("；选中：{}", selected.candidate.receipt()));
        }
        for (index, hypothesis) in self.hypotheses.iter().enumerate() {
            output.push_str(&format!("\n  {}. {}", index + 1, hypothesis.receipt()));
            for probe in &hypothesis.probes {
                output.push_str(&format!(
                    "\n     探针「{}」（{}，{}）：命中 {}，最高 {:.3}{}",
                    probe.label,
                    probe.role.label(),
                    probe.query,
                    probe.hits,
                    probe.top_score,
                    if probe.certified_top_k {
                        ""
                    } else {
                        "（未证明 top-k）"
                    }
                ));
            }
            if !hypothesis.answers.is_empty() {
                let clusters = hypothesis
                    .answers
                    .iter()
                    .take(3)
                    .map(|cluster| {
                        format!(
                            "{}（权重 {:.2}，{} 条）",
                            cluster.answer, cluster.weight, cluster.hits
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("；");
                output.push_str(&format!("\n     答案簇：{}", clusters));
            }
            for note in &hypothesis.notes {
                output.push_str(&format!("\n     说明：{}", note));
            }
        }
        for note in &self.notes {
            output.push_str(&format!("\n说明：{}", note));
        }
        output
    }
}

fn logit(probability: f64) -> f64 {
    let probability = probability.clamp(1e-6, 1.0 - 1e-6);
    (probability / (1.0 - probability)).ln()
}

fn sigmoid(value: f64) -> f64 {
    1.0 / (1.0 + (-value).exp())
}

fn normalise_candidates(candidates: Vec<HypothesisCandidate>) -> Vec<HypothesisCandidate> {
    let mut by_key: BTreeMap<String, HypothesisCandidate> = BTreeMap::new();
    for candidate in candidates {
        match by_key.get(&candidate.key()) {
            Some(existing) if existing.source.priority() >= candidate.source.priority() => {}
            _ => {
                by_key.insert(candidate.key(), candidate);
            }
        }
    }
    by_key.into_values().collect()
}

fn clamp01(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

fn evaluate_hypothesis<F>(
    candidate: HypothesisCandidate,
    config: &RewriteRetrievalConfig,
    search: &mut F,
) -> Option<HypothesisAssessment>
where
    F: FnMut(&str, usize, &SearchConstraints) -> EvidenceSearch,
{
    let probes = probes_for(&candidate.plan);
    if probes.is_empty() {
        return None;
    }
    let mut probe_reports = Vec::new();
    let mut clusters: BTreeMap<String, (f64, f64, usize)> = BTreeMap::new();
    let mut notes = Vec::new();
    let mut best_support: f64 = 0.0;
    let mut probe_answers: Vec<Option<String>> = Vec::new();
    for probe in probes.into_iter().take(config.max_probes) {
        let evidence = search(&probe.query, config.per_probe_k, &probe.constraints);
        let mut local: BTreeMap<String, (f64, f64, usize)> = BTreeMap::new();
        for hit in &evidence.hits {
            if hit.score <= 0.0 {
                continue;
            }
            let answer = probe.role.answer_from(hit);
            if answer.trim().is_empty() {
                continue;
            }
            let weight = hit.score
                * if probe.partial {
                    config.partial_discount
                } else {
                    1.0
                };
            best_support = best_support.max(weight);
            let entry = local.entry(answer.clone()).or_insert((0.0, 0.0, 0));
            entry.0 += weight;
            entry.1 = entry.1.max(hit.score);
            entry.2 += 1;
        }
        let local_answer = local
            .iter()
            .max_by(|left, right| {
                left.1
                     .0
                    .partial_cmp(&right.1 .0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| right.0.cmp(left.0))
            })
            .map(|(answer, _)| answer.clone());
        for (answer, (weight, best, hits)) in local {
            let entry = clusters.entry(answer).or_insert((0.0, 0.0, 0));
            entry.0 += weight;
            entry.1 = entry.1.max(best);
            entry.2 += hits;
        }
        probe_answers.push(local_answer);
        let mut probe_notes = evidence.receipt.notes.clone();
        if probe.partial {
            probe_notes.push(format!(
                "该探针只验证计划首跳，证据分按 {:.2} 折扣。",
                config.partial_discount
            ));
        }
        if !evidence.receipt.certified_top_k {
            probe_notes.push("检索未证明 top-k，裁决只使用当前预算内的证据。".to_string());
        }
        probe_reports.push(ProbeReport {
            label: probe.label,
            role: probe.role,
            query: probe.query,
            partial: probe.partial,
            hits: evidence.hits.len(),
            top_score: evidence
                .hits
                .iter()
                .map(|hit| hit.score)
                .fold(0.0_f64, f64::max),
            certified_top_k: evidence.receipt.certified_top_k,
            top_hits: evidence.hits.into_iter().take(3).collect(),
            notes: probe_notes,
        });
    }

    if clusters.is_empty() {
        return Some(HypothesisAssessment {
            candidate,
            probes: probe_reports,
            support: 0.0,
            coverage: 0.0,
            margin: 0.0,
            consistency: 0.0,
            confidence: 0.0,
            answer: None,
            answers: Vec::new(),
            notes: vec!["没有探针命中可解释的证据。".to_string()],
        });
    }

    let mut answers: Vec<AnswerCluster> = clusters
        .into_iter()
        .map(|(answer, (weight, best_score, hits))| AnswerCluster {
            answer,
            weight,
            best_score,
            hits,
        })
        .collect();
    answers.sort_by(|left, right| {
        right
            .weight
            .partial_cmp(&left.weight)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                right
                    .best_score
                    .partial_cmp(&left.best_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| left.answer.cmp(&right.answer))
    });
    let total_weight: f64 = answers.iter().map(|cluster| cluster.weight).sum();
    let top = &answers[0];
    let coverage = if total_weight > 0.0 {
        clamp01(top.weight / total_weight)
    } else {
        0.0
    };
    let second_weight = answers.get(1).map(|cluster| cluster.weight).unwrap_or(0.0);
    let margin = if top.weight > 0.0 {
        clamp01((top.weight - second_weight) / top.weight)
    } else {
        0.0
    };
    let with_hits = probe_answers
        .iter()
        .filter(|answer| answer.is_some())
        .count();
    let agreeing = probe_answers
        .iter()
        .filter(|answer| answer.as_deref() == Some(top.answer.as_str()))
        .count();
    let consistency = if with_hits == 0 {
        0.0
    } else {
        agreeing as f64 / with_hits as f64
    };
    let support = clamp01(best_support);
    let context_bonus = candidate.context_bonus.clamp(-2.0, 2.0);
    let log_odds = logit(candidate.source.prior())
        + context_bonus
        + config.support_gain * (support - config.support_pivot)
        + config.coverage_gain * (coverage - config.coverage_pivot)
        + config.margin_gain * (margin - config.margin_pivot)
        + config.consistency_gain * (consistency - config.consistency_pivot);
    let confidence = sigmoid(log_odds);
    if context_bonus.abs() > f64::EPSILON {
        notes.push(format!("上下文先验 {:+.2} 已计入 log-odds。", context_bonus));
    }
    if answers.len() > 1 {
        notes.push(format!(
            "证据分成 {} 个答案簇，主簇占比 {:.2}。",
            answers.len(),
            coverage
        ));
    }
    if probe_reports.len() > 1 && consistency < 1.0 {
        notes.push(format!(
            "{} 个有命中探针中只有 {} 个与主答案一致。",
            with_hits, agreeing
        ));
    }
    Some(HypothesisAssessment {
        candidate,
        probes: probe_reports,
        support,
        coverage,
        margin,
        consistency,
        confidence,
        answer: Some(top.answer.clone()),
        answers,
        notes,
    })
}

/// 对候选读法做证据裁决。
///
/// `search(query, k, constraints)` 由调用方注入（通常是
/// [`crate::symbolic_reasoner::SymbolicReasoner::evidence_search`] 的适配闭包），
/// 因此本模块不依赖任何具体存储，也可在离线评测里用独立检索器复现。
pub fn assess<F>(
    question: &str,
    candidates: Vec<HypothesisCandidate>,
    config: &RewriteRetrievalConfig,
    mut search: F,
) -> Assessment
where
    F: FnMut(&str, usize, &SearchConstraints) -> EvidenceSearch,
{
    let config = config.clamped();
    let candidates = normalise_candidates(candidates);
    let mut hypotheses: Vec<HypothesisAssessment> = candidates
        .into_iter()
        .filter_map(|candidate| evaluate_hypothesis(candidate, &config, &mut search))
        .take(config.max_hypotheses)
        .collect();
    hypotheses.sort_by(|left, right| {
        right
            .confidence
            .partial_cmp(&left.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                right
                    .support
                    .partial_cmp(&left.support)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| {
                right
                    .candidate
                    .source
                    .priority()
                    .cmp(&left.candidate.source.priority())
            })
            .then_with(|| left.candidate.key().cmp(&right.candidate.key()))
    });

    let mut notes = Vec::new();
    let Some(top_index) = hypotheses
        .iter()
        .position(|hypothesis| hypothesis.answer.is_some())
    else {
        return Assessment {
            dialect: DIALECT.to_string(),
            question: question.trim().to_string(),
            decision: Decision::Abstain,
            selected: None,
            hypotheses,
            notes: vec!["没有候选读法得到可解释的证据支持。".to_string()],
        };
    };
    let top = &hypotheses[top_index];
    let runner_up = hypotheses
        .iter()
        .enumerate()
        .filter(|(index, hypothesis)| {
            *index != top_index && hypothesis.answer.is_some() && hypothesis.answer != top.answer
        })
        .max_by(|left, right| {
            left.1
                .confidence
                .partial_cmp(&right.1.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    let different_answer_lead = runner_up
        .as_ref()
        .map(|(_, hypothesis)| top.confidence - hypothesis.confidence)
        .unwrap_or(1.0);
    let evidence_ok = top.support >= config.min_support
        && top.coverage >= config.min_coverage
        && top.margin >= config.min_margin;
    let leading = top.confidence >= config.accept_threshold && evidence_ok;
    let separated = different_answer_lead >= config.accept_margin;
    let decision = if leading && separated {
        Decision::Accept
    } else if top.confidence >= config.clarify_threshold
        && (top.support >= config.min_support)
        && (!separated || !evidence_ok)
    {
        Decision::Clarify
    } else {
        Decision::Abstain
    };
    if !evidence_ok {
        notes.push(format!(
            "主读法未过证据闸门：支持 {:.3}（≥{:.2}），覆盖 {:.3}（≥{:.2}），边际 {:.3}（≥{:.2}）。",
            top.support,
            config.min_support,
            top.coverage,
            config.min_coverage,
            top.margin,
            config.min_margin
        ));
    }
    if !separated {
        if let Some((index, competitor)) = &runner_up {
            notes.push(format!(
                "与不同答案的竞争读法 {}（{}）置信度只差 {:.3}，低于接受边际 {:.2}。",
                index + 1,
                competitor.candidate.rule,
                different_answer_lead,
                config.accept_margin
            ));
        }
    }
    if decision == Decision::Clarify {
        notes.push("多个读法都能解释部分证据，先澄清比硬选更诚实。".to_string());
    }
    Assessment {
        dialect: DIALECT.to_string(),
        question: question.trim().to_string(),
        decision,
        selected: (decision == Decision::Accept).then_some(top_index),
        hypotheses,
        notes,
    }
}

/// 把一个已编译计划包装成候选；便于调用方在不依赖内部类型的情况下构造输入。
pub fn candidate_from_plan(
    question: &str,
    source: HypothesisSource,
    rule: &str,
    plan: QueryPlan,
) -> HypothesisCandidate {
    HypothesisCandidate::new(source, rule, question, plan)
}

/// 离线评测样例：问句与期望答案（期望答案可写多个候选用 `|` 分隔）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvalCase {
    pub question: String,
    pub expected: String,
}

impl EvalCase {
    pub fn new(question: impl Into<String>, expected: impl Into<String>) -> Self {
        Self {
            question: question.into(),
            expected: expected.into(),
        }
    }

    /// 一行样例：`问句<TAB>期望答案`，也接受 `问句 => 期望答案`。
    pub fn parse_line(line: &str) -> Option<Self> {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            return None;
        }
        let (question, expected) = line
            .split_once('\t')
            .or_else(|| line.split_once("=>"))?;
        let question = question.trim();
        let expected = expected.trim();
        if question.is_empty() || expected.is_empty() {
            return None;
        }
        Some(Self::new(question, expected))
    }

    fn answer_matches(&self, answer: &str) -> bool {
        let answer = answer.trim();
        if answer.is_empty() {
            return false;
        }
        self.expected.split('|').any(|expected| {
            let expected = expected.trim();
            !expected.is_empty()
                && (answer == expected
                    || answer.contains(expected)
                    || expected.contains(answer))
        })
    }
}

/// 解析离线评测语料；非法行被跳过，空行与 `#` 注释被忽略。
pub fn parse_corpus(text: &str) -> Vec<EvalCase> {
    text.lines().filter_map(EvalCase::parse_line).collect()
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct SourceEval {
    pub selected: usize,
    pub correct: usize,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct EvalReport {
    pub total: usize,
    pub accepted: usize,
    pub accepted_correct: usize,
    pub accepted_wrong: usize,
    pub clarified: usize,
    pub abstained: usize,
    pub no_candidates: usize,
    pub by_source: BTreeMap<String, SourceEval>,
}

impl EvalReport {
    /// 接受率：接受读法的题数 / 总题数。
    pub fn coverage(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.accepted as f64 / self.total as f64
        }
    }

    /// 接受读法的答案精确率。
    pub fn precision(&self) -> f64 {
        if self.accepted == 0 {
            0.0
        } else {
            self.accepted_correct as f64 / self.accepted as f64
        }
    }

    pub fn render_text(&self) -> String {
        let mut output = format!(
            "读法裁决离线评测：{} 题；接受 {}（正确 {} / 错误 {}，覆盖 {:.2}，精确率 {:.2}），澄清 {}，弃权 {}，无候选 {}。",
            self.total,
            self.accepted,
            self.accepted_correct,
            self.accepted_wrong,
            self.coverage(),
            self.precision(),
            self.clarified,
            self.abstained,
            self.no_candidates
        );
        for (source, stats) in &self.by_source {
            output.push_str(&format!(
                "\n  {}：选中 {}，正确 {}",
                source, stats.selected, stats.correct
            ));
        }
        output
    }
}

/// 用候选生成器与证据检索器跑一遍离线评测。
///
/// 评测只比较「裁决层选中的证据答案」与期望答案；不执行完整计划，
/// 因此适合快速比较不同阈值/来源先验，而不会把生成措辞算进误差。
pub fn evaluate<C, S>(
    cases: &[EvalCase],
    config: &RewriteRetrievalConfig,
    candidates_for: C,
    search: S,
) -> EvalReport
where
    C: FnMut(&str) -> Vec<HypothesisCandidate>,
    S: FnMut(&str, usize, &SearchConstraints) -> EvidenceSearch,
{
    evaluate_with(cases, config, candidates_for, search, |hypothesis| {
        hypothesis.answer.clone()
    })
}

/// 与 [`evaluate`] 相同，但用 `answer_for` 执行选中读法并比较真实答案。
///
/// 探针答案可能只是首跳/中间实体；需要评估端到端回答质量时传入执行器。
pub fn evaluate_with<C, S, A>(
    cases: &[EvalCase],
    config: &RewriteRetrievalConfig,
    mut candidates_for: C,
    mut search: S,
    mut answer_for: A,
) -> EvalReport
where
    C: FnMut(&str) -> Vec<HypothesisCandidate>,
    S: FnMut(&str, usize, &SearchConstraints) -> EvidenceSearch,
    A: FnMut(&HypothesisAssessment) -> Option<String>,
{
    let mut report = EvalReport {
        total: cases.len(),
        ..EvalReport::default()
    };
    for case in cases {
        let candidates = candidates_for(&case.question);
        if candidates.is_empty() {
            report.no_candidates += 1;
            continue;
        }
        let assessment = assess(&case.question, candidates, config, &mut search);
        match assessment.decision {
            Decision::Accept => {
                report.accepted += 1;
                let selected = assessment.selected_hypothesis();
                let answer = selected.and_then(|hypothesis| answer_for(hypothesis));
                let correct = answer
                    .as_deref()
                    .map(|answer| case.answer_matches(answer))
                    .unwrap_or(false);
                if correct {
                    report.accepted_correct += 1;
                } else {
                    report.accepted_wrong += 1;
                }
                if let Some(hypothesis) = selected {
                    let source = hypothesis.candidate.source.label().to_string();
                    let entry = report.by_source.entry(source).or_default();
                    entry.selected += 1;
                    if correct {
                        entry.correct += 1;
                    }
                }
            }
            Decision::Clarify => report.clarified += 1,
            Decision::Abstain => report.abstained += 1,
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense_retrieval::DenseRetriever;
    use crate::prolog_engine::PrologFact;
    use std::collections::HashMap;

    fn facts() -> Vec<PrologFact> {
        vec![
            PrologFact::Entity {
                name: "张三".to_string(),
                entity_type: "人物".to_string(),
                properties: HashMap::from([("师傅".to_string(), "李四".to_string())]),
            },
            PrologFact::Entity {
                name: "王五".to_string(),
                entity_type: "人物".to_string(),
                properties: HashMap::from([("师傅".to_string(), "张三".to_string())]),
            },
            PrologFact::Entity {
                name: "赵六".to_string(),
                entity_type: "人物".to_string(),
                properties: HashMap::from([("师傅".to_string(), "李四".to_string())]),
            },
        ]
    }

    fn retriever() -> DenseRetriever {
        let mut retriever = DenseRetriever::new(8, 0.85, 12);
        retriever.observe_facts(&facts());
        retriever
    }

    fn assess_with_retriever(question: &str, candidates: Vec<HypothesisCandidate>) -> Assessment {
        let mut retriever = retriever();
        assess(
            question,
            candidates,
            &RewriteRetrievalConfig::default(),
            |query, k, constraints| retriever.search(query, k, constraints, Default::default()),
        )
    }

    fn candidate(source: HypothesisSource, subject: &str, predicate: &str) -> HypothesisCandidate {
        HypothesisCandidate::new(
            source,
            "test",
            "测试问句",
            QueryPlan::FindObjectByPredicate {
                subject: subject.to_string(),
                predicate: predicate.to_string(),
            },
        )
    }

    #[test]
    fn probes_follow_the_answer_role() {
        let probes = probes_for(&QueryPlan::FindSubject {
            object: "李四".to_string(),
            predicate: "师傅".to_string(),
        });
        assert_eq!(probes.len(), 1);
        assert_eq!(probes[0].role, ProbeRole::Subject);
        assert_eq!(probes[0].constraints.object.as_deref(), Some("李四"));
        assert_eq!(probes[0].constraints.predicate.as_deref(), Some("师傅"));

        let probes = probes_for(&QueryPlan::FindObject {
            segments: vec!["张三".to_string(), "师傅".to_string(), "师傅".to_string()],
        });
        assert_eq!(probes.len(), 1);
        assert!(probes[0].partial);
        assert_eq!(probes[0].constraints.subject.as_deref(), Some("张三"));
    }

    #[test]
    fn evidence_accepts_the_supported_reading() {
        let assessment = assess_with_retriever(
            "张三的师傅是谁",
            vec![candidate(HypothesisSource::ExactPlan, "张三", "师傅")],
        );
        assert_eq!(
            assessment.decision,
            Decision::Accept,
            "{}",
            assessment.render_report()
        );
        assert_eq!(assessment.answer(), Some("李四"));
        let top = assessment.selected_hypothesis().unwrap();
        assert!(top.support > 0.0);
        assert!(top.coverage > 0.0);
    }

    #[test]
    fn contradictory_readings_trigger_clarification_or_abstention() {
        let assessment = assess_with_retriever(
            "张三和谁有关",
            vec![
                candidate(HypothesisSource::ExactPlan, "张三", "师傅"),
                candidate(HypothesisSource::SyntaxRewrite, "王五", "师傅"),
            ],
        );
        assert_ne!(
            assessment.decision,
            Decision::Accept,
            "{}",
            assessment.render_report()
        );
    }

    #[test]
    fn same_plan_from_multiple_sources_is_deduplicated() {
        let assessment = assess_with_retriever(
            "张三的师傅是谁",
            vec![
                candidate(HypothesisSource::SyntaxRewrite, "张三", "师傅"),
                candidate(HypothesisSource::ExactPlan, "张三", "师傅"),
            ],
        );
        assert_eq!(assessment.hypotheses.len(), 1);
        assert_eq!(
            assessment.hypotheses[0].candidate.source,
            HypothesisSource::ExactPlan
        );
    }

    #[test]
    fn no_evidence_abstains() {
        let assessment = assess_with_retriever(
            "未名的师傅是谁",
            vec![candidate(HypothesisSource::ExactPlan, "未名", "师傅")],
        );
        assert_eq!(assessment.decision, Decision::Abstain);
        assert_eq!(assessment.answer(), None);
    }

    #[test]
    fn unverifiable_plans_have_no_probes() {
        assert!(probes_for(&QueryPlan::Similar {
            entity: "张三".to_string()
        })
        .is_empty());
        assert!(probes_for(&QueryPlan::Analogy {
            a: "张三".to_string(),
            b: "李四".to_string(),
            c: "王五".to_string()
        })
        .is_empty());
    }

    #[test]
    fn corpus_parsing_skips_comments_and_malformed_lines() {
        let cases = parse_corpus(
            "# 注释\n张三的师傅是谁\t李四\n\n谁的师傅是李四 => 张三\n没有分隔符\n",
        );
        assert_eq!(cases.len(), 2);
        assert_eq!(cases[0].question, "张三的师傅是谁");
        assert_eq!(cases[1].expected, "张三");
    }

    #[test]
    fn offline_evaluation_counts_acceptance_and_accuracy() {
        let cases = vec![
            EvalCase::new("张三的师傅是谁", "李四"),
            EvalCase::new("未名的师傅是谁", "某人"),
        ];
        let mut retriever = retriever();
        let report = evaluate(
            &cases,
            &RewriteRetrievalConfig::default(),
            |question| {
                vec![candidate(
                    HypothesisSource::ExactPlan,
                    question.split('的').next().unwrap_or(question),
                    "师傅",
                )]
            },
            |query, k, constraints| retriever.search(query, k, constraints, Default::default()),
        );
        assert_eq!(report.total, 2);
        assert_eq!(report.accepted, 1);
        assert_eq!(report.accepted_correct, 1);
        assert_eq!(report.accepted_wrong, 0);
        assert_eq!(report.abstained, 1);
        assert!((report.coverage() - 0.5).abs() < 1e-9);
        assert!((report.precision() - 1.0).abs() < 1e-9);
        assert!(report.render_text().contains("精确率 1.00"));
    }

    #[test]
    fn offline_evaluation_can_execute_selected_plan() {
        let cases = vec![EvalCase::new("张三的师傅是谁", "李四")];
        let mut retriever = retriever();
        let report = evaluate_with(
            &cases,
            &RewriteRetrievalConfig::default(),
            |question| vec![candidate(HypothesisSource::ExactPlan, "张三", "师傅")],
            |query, k, constraints| retriever.search(query, k, constraints, Default::default()),
            |_hypothesis| Some("王五".to_string()),
        );
        assert_eq!(report.accepted, 1);
        assert_eq!(report.accepted_correct, 0);
        assert_eq!(report.accepted_wrong, 1);
        assert!((report.precision() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn unverifiable_candidates_do_not_consume_hypothesis_budget() {
        let mut retriever = retriever();
        let mut config = RewriteRetrievalConfig::default();
        config.max_hypotheses = 1;
        let assessment = assess(
            "张三的师傅是谁",
            vec![
                HypothesisCandidate::new(
                    HypothesisSource::ExactPlan,
                    "similar",
                    "张三的师傅是谁",
                    QueryPlan::Similar {
                        entity: "张三".to_string(),
                    },
                ),
                candidate(HypothesisSource::ExactPlan, "张三", "师傅"),
            ],
            &config,
            |query, k, constraints| retriever.search(query, k, constraints, Default::default()),
        );
        assert_eq!(assessment.hypotheses.len(), 1);
        assert_eq!(assessment.answer(), Some("李四"));
    }

    #[test]
    fn context_bonus_raises_confidence_without_changing_evidence() {
        let mut plain_retriever = retriever();
        let plain = assess(
            "张三的师傅是谁",
            vec![candidate(HypothesisSource::ExactPlan, "张三", "师傅")],
            &RewriteRetrievalConfig::default(),
            |query, k, constraints| {
                plain_retriever.search(query, k, constraints, Default::default())
            },
        );
        let mut retriever2 = retriever();
        let contextual = assess(
            "张三的师傅是谁",
            vec![candidate(HypothesisSource::ExactPlan, "张三", "师傅").with_context_bonus(1.0)],
            &RewriteRetrievalConfig::default(),
            |query, k, constraints| retriever2.search(query, k, constraints, Default::default()),
        );
        assert!(contextual.hypotheses[0].confidence > plain.hypotheses[0].confidence);
        assert_eq!(contextual.answer(), plain.answer());
    }
}

//! 软路由（Soft Routing）：允许近失的解释竞争，而不是「第一条命中即答案」。
//!
//! 既有 [`crate::pattern_ir::PatternLibrary`] 的查询路由是**硬**的：按优先级
//! 做锚定匹配，第一条成功的模式直接胜出；一条要素之差就会让整句落回
//! 「无法理解」。`frontier` 能看到「差多少」，但近失候选不参与路由；
//! `ContextGuard` 只回答守卫是否满足，不回答「有多满足」。
//!
//! 本模块补上这两块积木之间的连接：
//!
//! 1. **软对齐**：用动态规划把模式要素与词序列对齐，允许固定词近义/前缀、
//!    可选成分缺失、虚词与语序噪声；同时保留硬约束（守卫、覆盖闸门）；
//! 2. **证据合成**：把对齐质量、模式先验、上下文连续性放进 log-odds 相加，
//!    再经 sigmoid 得到「这句话被该模式解释」的后验（与
//!    [`crate::markov_logic`] 同一套语义，不训练、无梯度）；
//! 3. **三档决策**：接受 / 澄清 / 回退。只有领先且过闸门的候选才被接受；
//!    两个不同路由难分时输出可澄清的候选，而不是硬答。
//!
//! 关键约束：
//!
//! - **不改变精确路径**：调用方只在精确路由失败后才咨询本模块；
//! - **硬闸门先于软排序**：上下文守卫、最小对齐覆盖、token 覆盖都在打分前
//!   排除候选；软层只排序已经过闸门的解释；
//! - **可解释**：每个候选都带要素/token 覆盖、缺失、额外、部分匹配与
//!   分数贡献，报告可逐项复核；
//! - **默认关闭**：`enabled = false` 时调用方不会使用本模块，旧行为不变。

use std::cmp::Ordering;
use std::fmt::Write;

use serde::{Deserialize, Serialize};

use crate::bindings::BindingSet;
use crate::nlp::TaggedWord;
use crate::pattern_ir::{
    match_entry_detailed, Elem, PatternContext, PatternEntry, PatternLibrary, PatternMatch, Span,
};

/// 报告方言：读取方据此拒绝把旧版报告当新版解释。
pub const DIALECT: &str = "knowprolog-soft-route/1";

/// 软路由候选的决策档位。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SoftDecision {
    /// 存在领先且过闸门的解释，可以继续回答。
    Accept,
    /// 多个解释接近，先问一句比硬答更诚实。
    Clarify,
    /// 没有足够证据，交回旧路径（回退/拒答）。
    Reject,
}

impl SoftDecision {
    pub fn label(self) -> &'static str {
        match self {
            SoftDecision::Accept => "接受",
            SoftDecision::Clarify => "澄清",
            SoftDecision::Reject => "回退",
        }
    }
}

fn default_true() -> bool {
    true
}

fn default_accept_threshold() -> f64 {
    0.74
}

fn default_clarify_threshold() -> f64 {
    0.48
}

fn default_accept_margin() -> f64 {
    0.08
}

fn default_min_alignment() -> f64 {
    0.62
}

fn default_min_token_coverage() -> f64 {
    0.65
}

fn default_alignment_pivot() -> f64 {
    0.72
}

fn default_alignment_scale() -> f64 {
    0.12
}

fn default_alignment_gain() -> f64 {
    1.6
}

fn default_priority_gain() -> f64 {
    0.35
}

fn default_specificity_gain() -> f64 {
    0.25
}

fn default_context_bonus() -> f64 {
    0.30
}

fn default_fallback_penalty() -> f64 {
    0.30
}

fn default_extra_token_penalty() -> f64 {
    0.06
}

fn default_fuzzy_floor() -> f64 {
    0.55
}

fn default_max_alternatives() -> usize {
    4
}

fn default_clarify_before_fallback() -> bool {
    true
}

/// 软路由参数。所有权重都是**可配置先验**，不做在线学习。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SoftRouteConfig {
    /// 是否允许调用方在精确路由失败后使用软路由。
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// 接受一档解释的最低后验。
    #[serde(default = "default_accept_threshold")]
    pub accept_threshold: f64,
    /// 进入澄清档的最低后验。
    #[serde(default = "default_clarify_threshold")]
    pub clarify_threshold: f64,
    /// 与第二候选的最小后验差（不同路由时生效）。
    #[serde(default = "default_accept_margin")]
    pub accept_margin: f64,
    /// 最小对齐得分（硬闸门）。
    #[serde(default = "default_min_alignment")]
    pub min_alignment: f64,
    /// 最小 token 覆盖（硬闸门，防止短模式吞长句）。
    #[serde(default = "default_min_token_coverage")]
    pub min_token_coverage: f64,
    /// 对齐证据的中性点：高于它才增加 log-odds。
    #[serde(default = "default_alignment_pivot")]
    pub alignment_pivot: f64,
    /// 对齐证据的尺度（越小越敏感）。
    #[serde(default = "default_alignment_scale")]
    pub alignment_scale: f64,
    /// 对齐证据的 log-odds 增益。
    #[serde(default = "default_alignment_gain")]
    pub alignment_gain: f64,
    /// 模式优先级先验的增益。
    #[serde(default = "default_priority_gain")]
    pub priority_gain: f64,
    /// 模式特化度先验的增益。
    #[serde(default = "default_specificity_gain")]
    pub specificity_gain: f64,
    /// 守卫被满足时的上下文证据。
    #[serde(default = "default_context_bonus")]
    pub context_bonus: f64,
    /// NLU 回退候选的折扣（配置查询优先）。
    #[serde(default = "default_fallback_penalty")]
    pub fallback_penalty: f64,
    /// 每个未对齐 token 的惩罚（标点/空白不计）。
    #[serde(default = "default_extra_token_penalty")]
    pub extra_token_penalty: f64,
    /// 单要素对齐的最低可接受质量。
    #[serde(default = "default_fuzzy_floor")]
    pub fuzzy_floor: f64,
    /// 报告里最多展示多少个候选。
    #[serde(default = "default_max_alternatives")]
    pub max_alternatives: usize,
    /// 软层判为澄清且存在回退候选时，是否先向用户澄清。
    #[serde(default = "default_clarify_before_fallback")]
    pub clarify_before_fallback: bool,
}

impl Default for SoftRouteConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            accept_threshold: default_accept_threshold(),
            clarify_threshold: default_clarify_threshold(),
            accept_margin: default_accept_margin(),
            min_alignment: default_min_alignment(),
            min_token_coverage: default_min_token_coverage(),
            alignment_pivot: default_alignment_pivot(),
            alignment_scale: default_alignment_scale(),
            alignment_gain: default_alignment_gain(),
            priority_gain: default_priority_gain(),
            specificity_gain: default_specificity_gain(),
            context_bonus: default_context_bonus(),
            fallback_penalty: default_fallback_penalty(),
            extra_token_penalty: default_extra_token_penalty(),
            fuzzy_floor: default_fuzzy_floor(),
            max_alternatives: default_max_alternatives(),
            clarify_before_fallback: default_clarify_before_fallback(),
        }
    }
}

impl SoftRouteConfig {
    /// 配置自检：阈值必须在 [0,1]、排序关系不能反转、尺度必须为正。
    pub fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("accept_threshold", self.accept_threshold),
            ("clarify_threshold", self.clarify_threshold),
            ("accept_margin", self.accept_margin),
            ("min_alignment", self.min_alignment),
            ("min_token_coverage", self.min_token_coverage),
            ("alignment_pivot", self.alignment_pivot),
            ("fuzzy_floor", self.fuzzy_floor),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(format!("{}={} 不在 [0,1]", name, value));
            }
        }
        if self.clarify_threshold > self.accept_threshold {
            return Err("clarify_threshold 不能高于 accept_threshold".to_string());
        }
        if self.alignment_scale <= 0.0 || !self.alignment_scale.is_finite() {
            return Err("alignment_scale 必须为正".to_string());
        }
        for (name, value) in [
            ("alignment_gain", self.alignment_gain),
            ("priority_gain", self.priority_gain),
            ("specificity_gain", self.specificity_gain),
            ("context_bonus", self.context_bonus),
            ("fallback_penalty", self.fallback_penalty),
            ("extra_token_penalty", self.extra_token_penalty),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(format!("{}={} 必须为非负有限数", name, value));
            }
        }
        if self.max_alternatives == 0 {
            return Err("max_alternatives 至少为 1".to_string());
        }
        Ok(())
    }
}

/// 一次软对齐的完整账目。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Alignment {
    /// 模式要素总数。
    pub total_elems: usize,
    /// 对齐上的要素数（可选未占位不计）。
    pub matched_elems: usize,
    /// 其中质量 < 1 的近失匹配数。
    pub partial_elems: usize,
    /// 对齐到的 token 数（跨词要素按实际 token 计）。
    pub matched_tokens: usize,
    /// 非标点 token 总数。
    pub total_tokens: usize,
    /// 未对齐的非标点 token 数。
    pub extra_tokens: usize,
    /// 对齐得分（0..=1）。
    pub score: f64,
    /// 是否走了精确匹配器（逐字同一实现）。
    pub exact: bool,
    /// 逐要素区间（`None` = 可选未占位或未对齐）。
    pub elem_spans: Vec<Option<Span>>,
    pub entity_spans: Vec<Span>,
    pub entities: Vec<String>,
    pub bindings: BindingSet,
}

impl Alignment {
    /// token 覆盖率：对齐 token / 非标点 token。
    pub fn token_coverage(&self) -> f64 {
        if self.total_tokens == 0 {
            1.0
        } else {
            self.matched_tokens as f64 / self.total_tokens as f64
        }
    }

    /// 要素覆盖率：对齐要素 / 全部要素。
    pub fn elem_coverage(&self) -> f64 {
        if self.total_elems == 0 {
            1.0
        } else {
            self.matched_elems as f64 / self.total_elems as f64
        }
    }

    /// 一个未通过硬闸门的解释仍然可以被报告，但不能被接受。
    pub fn gate(&self, config: &SoftRouteConfig) -> Option<String> {
        if self.matched_elems == 0 {
            return Some("没有任何模式要素对齐".to_string());
        }
        if self.score < config.min_alignment {
            return Some(format!(
                "对齐得分 {:.2} 低于闸门 {:.2}",
                self.score, config.min_alignment
            ));
        }
        if self.token_coverage() < config.min_token_coverage {
            return Some(format!(
                "token 覆盖 {:.2} 低于闸门 {:.2}",
                self.token_coverage(),
                config.min_token_coverage
            ));
        }
        None
    }
}

/// 一个被软路由考虑的候选解释。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SoftCandidate {
    pub entry: PatternEntry,
    /// 解释后验 P(该模式是这句话的意图 | 证据)。
    pub confidence: f64,
    /// 对齐账目。
    pub alignment: Alignment,
    /// 因硬闸门落选时的原因；`None` 表示可参与接受/澄清。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blocked: Option<String>,
    /// log-odds 分项（审计用）。
    #[serde(default)]
    pub evidence: Vec<EvidenceTerm>,
}

impl SoftCandidate {
    pub fn query_type(&self) -> &str {
        &self.entry.query_type
    }

    pub fn name(&self) -> &str {
        &self.entry.name
    }

    pub fn eligible(&self) -> bool {
        self.blocked.is_none()
    }

    /// 把软对齐结果投影成既有路由/经验账本使用的 [`PatternMatch`]。
    pub fn to_pattern_match(&self) -> PatternMatch {
        let span = self
            .alignment
            .elem_spans
            .iter()
            .flatten()
            .fold(None::<Span>, |acc, span| {
                Some(match acc {
                    Some(existing) => Span::new(
                        existing.start.min(span.start),
                        existing.end.max(span.end),
                    ),
                    None => *span,
                })
            })
            .unwrap_or_else(|| Span::new(0, 0));
        PatternMatch {
            entry: self.entry.clone(),
            bindings: self.alignment.bindings.clone(),
            entities: self.alignment.entities.clone(),
            entity_spans: self.alignment.entity_spans.clone(),
            elem_spans: self.alignment.elem_spans.clone(),
            span,
        }
    }
}

/// log-odds 的一个可审计分项。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceTerm {
    pub name: String,
    pub value: f64,
}

/// 一次软路由的决策账目。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SoftRouteOutcome {
    pub dialect: String,
    pub decision: SoftDecision,
    /// 接受/澄清时排第一的候选。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected: Option<SoftCandidate>,
    /// 除 `selected` 外按后验降序的候选（有界）。
    #[serde(default)]
    pub alternatives: Vec<SoftCandidate>,
    /// 决策原因（人类可读）。
    pub reason: String,
}

impl SoftRouteOutcome {
    /// 是否给出了可执行的软路由。
    pub fn accepted(&self) -> bool {
        self.decision == SoftDecision::Accept && self.selected.is_some()
    }

    /// 需要澄清时返回可选项（已按后验排序）。
    pub fn clarification_options(&self) -> Vec<&SoftCandidate> {
        if self.decision != SoftDecision::Clarify {
            return Vec::new();
        }
        self.selected.iter().chain(self.alternatives.iter()).collect()
    }

    pub fn render_report(&self, input: &str) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "软路由 {}「{}」", self.dialect, input);
        match (&self.decision, &self.selected) {
            (SoftDecision::Accept, Some(selected)) => {
                let _ = writeln!(
                    out,
                    "判定：接受「{}」→ {}（理解置信度 {:.2}）",
                    selected.name(),
                    selected.query_type(),
                    selected.confidence
                );
            }
            (SoftDecision::Clarify, Some(selected)) => {
                let _ = writeln!(
                    out,
                    "判定：澄清（首选「{}」→ {}，置信度 {:.2}）",
                    selected.name(),
                    selected.query_type(),
                    selected.confidence
                );
            }
            _ => {
                let _ = writeln!(out, "判定：回退（软层没有足够证据）");
            }
        }
        let _ = writeln!(out, "理由：{}", self.reason);
        if self.decision == SoftDecision::Clarify {
            let options = self.clarification_options();
            if !options.is_empty() {
                let _ = writeln!(out, "可选解释：");
                for (index, option) in options.iter().take(4).enumerate() {
                    let _ = writeln!(
                        out,
                        "  {}. {} → {}（{:.2}）",
                        index + 1,
                        option.name(),
                        option.query_type(),
                        option.confidence
                    );
                }
            }
        }
        let candidates: Vec<&SoftCandidate> = self
            .selected
            .iter()
            .chain(self.alternatives.iter())
            .collect();
        if !candidates.is_empty() {
            let _ = writeln!(out, "候选（按后验降序）：");
            for (index, candidate) in candidates.iter().enumerate() {
                let status = match (&candidate.blocked, candidate.eligible()) {
                    (Some(reason), _) => format!("闸门：{}", reason),
                    (None, true) if index == 0 => "候选".to_string(),
                    _ => "次选".to_string(),
                };
                let _ = writeln!(
                    out,
                    "  {}. {}（{}）{:.2}  [{}；{}]",
                    index + 1,
                    candidate.name(),
                    candidate.query_type(),
                    candidate.confidence,
                    status,
                    if candidate.entry.fallback { "回退层" } else { "配置层" }
                );
                let _ = writeln!(
                    out,
                    "     对齐：要素 {}/{}，token {}/{}（覆盖 {:.0}%），部分 {}，额外 {}，得分 {:.2}{}",
                    candidate.alignment.matched_elems,
                    candidate.alignment.total_elems,
                    candidate.alignment.matched_tokens,
                    candidate.alignment.total_tokens,
                    candidate.alignment.token_coverage() * 100.0,
                    candidate.alignment.partial_elems,
                    candidate.alignment.extra_tokens,
                    candidate.alignment.score,
                    if candidate.alignment.exact {
                        "，精确"
                    } else {
                        ""
                    }
                );
                if let Some(guard) = &candidate.entry.guard {
                    let _ = writeln!(out, "     守卫：{}", guard.render());
                }
                let terms: Vec<String> = candidate
                    .evidence
                    .iter()
                    .map(|term| format!("{}={:+.2}", term.name, term.value))
                    .collect();
                if !terms.is_empty() {
                    let _ = writeln!(out, "     log-odds：{}", terms.join("；"));
                }
            }
        }
        out
    }
}

/// 软路由入口（已有词序列）。
///
/// 精确匹配仍然由 [`match_entry_detailed`] 给出；本函数只负责在它失败后，
/// 为**全部**通过守卫的候选计算软对齐并竞争。调用方不应在精确路由成功时
/// 再调用本函数。
pub fn route_tokens(
    _input: &str,
    library: &PatternLibrary,
    tokens: &[TaggedWord],
    context: &PatternContext,
    config: &SoftRouteConfig,
) -> SoftRouteOutcome {
    let mut candidates: Vec<SoftCandidate> = Vec::new();
    let max_priority = library
        .entries()
        .iter()
        .filter(|entry| entry.route || entry.fallback)
        .map(|entry| entry.priority)
        .max()
        .unwrap_or(0);
    let min_priority = library
        .entries()
        .iter()
        .filter(|entry| entry.route || entry.fallback)
        .map(|entry| entry.priority)
        .min()
        .unwrap_or(0);
    let priority_span = (max_priority - min_priority).max(1) as f64;

    for entry in library.entries() {
        if !(entry.route || entry.fallback) {
            continue;
        }
        if !entry.context_satisfied(context) {
            continue;
        }
        let alignment = align(entry, tokens, config);
        let mut evidence = Vec::new();
        let priority_norm =
            (entry.priority - min_priority) as f64 / priority_span;
        let prior = 0.45 + config.priority_gain * (priority_norm - 0.5)
            + config.specificity_gain * (entry.specificity() - 0.5);
        let prior = prior.clamp(0.05, 0.95);
        let prior_logit = logit(prior);
        evidence.push(EvidenceTerm {
            name: "先验".to_string(),
            value: prior_logit,
        });
        let alignment_evidence =
            (alignment.score - config.alignment_pivot) / config.alignment_scale;
        let alignment_logit = config.alignment_gain * alignment_evidence;
        evidence.push(EvidenceTerm {
            name: "对齐".to_string(),
            value: alignment_logit,
        });
        let mut total = prior_logit + alignment_logit;
        if entry.guard.is_some() {
            total += config.context_bonus;
            evidence.push(EvidenceTerm {
                name: "上下文".to_string(),
                value: config.context_bonus,
            });
        }
        if entry.fallback {
            total -= config.fallback_penalty;
            evidence.push(EvidenceTerm {
                name: "回退折扣".to_string(),
                value: -config.fallback_penalty,
            });
        }
        if alignment.extra_tokens > 0 {
            let penalty = config.extra_token_penalty * alignment.extra_tokens as f64;
            total -= penalty;
            evidence.push(EvidenceTerm {
                name: "额外词".to_string(),
                value: -penalty,
            });
        }
        let blocked = alignment.gate(config);
        candidates.push(SoftCandidate {
            entry: entry.clone(),
            confidence: sigmoid(total),
            alignment,
            blocked,
            evidence,
        });
    }

    candidates.sort_by(compare_candidates);
    // 分层：配置路由先获得一次完整决策；只有它完全没有可用解释时，
    // 才允许 NLU 回退候选接管。否则一个泛化的回退槽位会压过更具体的
    // 配置模式（例如「属性查询」抢走「哲学思想查询」）。
    let route_eligible: Vec<&SoftCandidate> = candidates
        .iter()
        .filter(|candidate| candidate.eligible() && candidate.entry.route)
        .collect();
    let fallback_eligible: Vec<&SoftCandidate> = candidates
        .iter()
        .filter(|candidate| candidate.eligible() && candidate.entry.fallback)
        .collect();
    let route_decision = decide(&route_eligible, config);
    let (eligible, decision) = if route_decision == SoftDecision::Reject
        && !fallback_eligible.is_empty()
    {
        let fallback_decision = decide(&fallback_eligible, config);
        (fallback_eligible, fallback_decision)
    } else {
        (route_eligible, route_decision)
    };
    let selected = match decision {
        SoftDecision::Accept | SoftDecision::Clarify => eligible.first().map(|c| (*c).clone()),
        SoftDecision::Reject => None,
    };
    let alternatives = candidates
        .iter()
        .filter(|candidate| {
            selected
                .as_ref()
                .map(|selected| candidate.entry.id() != selected.entry.id())
                .unwrap_or(true)
        })
        .take(config.max_alternatives)
        .cloned()
        .collect();
    let reason = decision_reason(&decision, &eligible, config);
    SoftRouteOutcome {
        dialect: DIALECT.to_string(),
        decision,
        selected,
        alternatives,
        reason,
    }
}

/// 软路由入口（原句；与运行时同序做分词与同义词归一）。
pub fn route(
    input: &str,
    library: &PatternLibrary,
    nlp: &crate::NlpConfig,
    synonyms: &std::collections::HashMap<String, String>,
    context: &PatternContext,
    config: &SoftRouteConfig,
) -> SoftRouteOutcome {
    let tagged = crate::nlp::preprocess(input, nlp);
    let tokens = crate::nlp::normalize_words(tagged, synonyms);
    route_tokens(input, library, &tokens, context, config)
}

fn compare_candidates(left: &SoftCandidate, right: &SoftCandidate) -> Ordering {
    right
        .confidence
        .partial_cmp(&left.confidence)
        .unwrap_or(Ordering::Equal)
        .then_with(|| {
            right
                .alignment
                .score
                .partial_cmp(&left.alignment.score)
                .unwrap_or(Ordering::Equal)
        })
        .then_with(|| right.entry.priority.cmp(&left.entry.priority))
        .then_with(|| left.entry.name.cmp(&right.entry.name))
}

fn decide(eligible: &[&SoftCandidate], config: &SoftRouteConfig) -> SoftDecision {
    let Some(top) = eligible.first() else {
        return SoftDecision::Reject;
    };
    if top.confidence < config.clarify_threshold {
        return SoftDecision::Reject;
    }
    let second = eligible.get(1);
    let conflicting = second
        .map(|second| second.query_type() != top.query_type())
        .unwrap_or(false);
    if top.confidence >= config.accept_threshold {
        let margin = second
            .map(|second| top.confidence - second.confidence)
            .unwrap_or(1.0);
        if !conflicting || margin >= config.accept_margin {
            return SoftDecision::Accept;
        }
    }
    if conflicting {
        SoftDecision::Clarify
    } else {
        SoftDecision::Reject
    }
}

fn decision_reason(
    decision: &SoftDecision,
    eligible: &[&SoftCandidate],
    config: &SoftRouteConfig,
) -> String {
    let Some(top) = eligible.first() else {
        return "没有通过守卫与对齐闸门的候选".to_string();
    };
    match decision {
        SoftDecision::Accept => {
            let margin = eligible
                .get(1)
                .map(|second| top.confidence - second.confidence)
                .unwrap_or(1.0);
            format!(
                "「{}」后验 {:.2} ≥ 接受阈值 {:.2}，同层领先 {:.2}（阈值 {:.2}）",
                top.name(),
                top.confidence,
                config.accept_threshold,
                margin,
                config.accept_margin
            )
        }
        SoftDecision::Clarify => {
            let second = eligible.get(1);
            match second {
                Some(second) if second.query_type() != top.query_type() => format!(
                    "「{}」({:.2}) 与「{}」({:.2}) 路由不同且差距不足 {:.2}",
                    top.name(),
                    top.confidence,
                    second.name(),
                    second.confidence,
                    config.accept_margin
                ),
                _ => format!(
                    "「{}」后验 {:.2} 未达接受阈值 {:.2}",
                    top.name(),
                    top.confidence,
                    config.accept_threshold
                ),
            }
        }
        SoftDecision::Reject => format!(
            "最佳候选「{}」后验 {:.2} 低于澄清阈值 {:.2}",
            top.name(),
            top.confidence,
            config.clarify_threshold
        ),
    }
}

fn logit(probability: f64) -> f64 {
    let p = probability.clamp(1e-6, 1.0 - 1e-6);
    (p / (1.0 - p)).ln()
}

fn sigmoid(value: f64) -> f64 {
    1.0 / (1.0 + (-value).exp())
}

/// 对一条模式做软对齐。精确匹配优先，失败后走动态规划。
pub fn align(entry: &PatternEntry, tokens: &[TaggedWord], config: &SoftRouteConfig) -> Alignment {
    if let Ok(matched) = match_entry_detailed(entry, tokens) {
        let non_skippable = tokens.iter().filter(|token| !is_skippable(&token.text)).count();
        let matched_tokens = non_skippable;
        return Alignment {
            total_elems: entry.elems.len(),
            matched_elems: matched.elem_spans.iter().filter(|span| span.is_some()).count(),
            partial_elems: 0,
            matched_tokens,
            total_tokens: non_skippable,
            extra_tokens: 0,
            score: 1.0,
            exact: true,
            elem_spans: matched.elem_spans,
            entity_spans: matched.entity_spans,
            entities: matched.entities,
            bindings: matched.bindings,
        };
    }
    align_soft(entry, tokens, config)
}

#[derive(Debug, Clone)]
struct State {
    score: f64,
    matched_elems: usize,
    matched_tokens: usize,
    action: Action,
}

#[derive(Debug, Clone)]
enum Action {
    Start,
    Match {
        len: usize,
        quality: f64,
        span: Span,
        entity: Option<String>,
        binding: Option<(String, String)>,
    },
    SkipElem,
    SkipToken,
}

#[derive(Debug, Clone)]
struct ElemHit {
    len: usize,
    quality: f64,
    entity: Option<String>,
    binding: Option<(String, String)>,
}

fn align_soft(entry: &PatternEntry, tokens: &[TaggedWord], config: &SoftRouteConfig) -> Alignment {
    let elems = &entry.elems;
    let total_tokens = tokens.iter().filter(|token| !is_skippable(&token.text)).count();
    let total_elems = elems.len();
    let weights: Vec<f64> = elems.iter().map(elem_weight).collect();
    let total_weight: f64 = weights.iter().sum();
    let (rows, cols) = (total_elems + 1, tokens.len() + 1);
    let mut states: Vec<Vec<Option<State>>> = vec![vec![None; cols]; rows];
    states[0][0] = Some(State {
        score: 0.0,
        matched_elems: 0,
        matched_tokens: 0,
        action: Action::Start,
    });

    for i in 0..=total_elems {
        for j in 0..=tokens.len() {
            let Some(state) = states[i][j].clone() else {
                continue;
            };
            if i < total_elems {
                let next = State {
                    score: state.score,
                    matched_elems: state.matched_elems,
                    matched_tokens: state.matched_tokens,
                    action: Action::SkipElem,
                };
                update_state(&mut states[i + 1][j], next);
                if j < tokens.len() && !is_skippable(&tokens[j].text) {
                    if let Some(hit) = match_elem(elems[i].clone(), tokens, j, config) {
                        let next = State {
                            score: state.score + hit.quality * weights[i],
                            matched_elems: state.matched_elems + 1,
                            matched_tokens: state.matched_tokens + hit.len,
                            action: Action::Match {
                                len: hit.len,
                                quality: hit.quality,
                                span: Span::new(j, j + hit.len),
                                entity: hit.entity,
                                binding: hit.binding,
                            },
                        };
                        update_state(&mut states[i + 1][j + hit.len], next);
                    }
                }
            }
            if j < tokens.len() {
                let skippable = is_skippable(&tokens[j].text);
                let next = State {
                    score: state.score
                        - if skippable {
                            0.0
                        } else {
                            config.extra_token_penalty
                        },
                    matched_elems: state.matched_elems,
                    matched_tokens: state.matched_tokens,
                    action: Action::SkipToken,
                };
                update_state(&mut states[i][j + 1], next);
            }
        }
    }

    let mut best: Option<(usize, State)> = None;
    for (j, state) in states[total_elems].iter().enumerate() {
        let Some(state) = state else {
            continue;
        };
        let remaining_extra = tokens[j..]
            .iter()
            .filter(|token| !is_skippable(&token.text))
            .count();
        let adjusted = State {
            score: state.score - config.extra_token_penalty * remaining_extra as f64,
            matched_elems: state.matched_elems,
            matched_tokens: state.matched_tokens,
            action: state.action.clone(),
        };
        let replace = match &best {
            None => true,
            Some((best_j, best_state)) => {
                if adjusted.score > best_state.score + 1e-9 {
                    true
                } else if (adjusted.score - best_state.score).abs() <= 1e-9 {
                    adjusted.matched_tokens > best_state.matched_tokens
                        || (adjusted.matched_tokens == best_state.matched_tokens && j < *best_j)
                } else {
                    false
                }
            }
        };
        if replace {
            best = Some((j, adjusted));
        }
    }

    let (end_j, best) = best.unwrap_or((0, State {
        score: 0.0,
        matched_elems: 0,
        matched_tokens: 0,
        action: Action::Start,
    }));
    let mut elem_spans: Vec<Option<Span>> = vec![None; total_elems];
    let mut entities = Vec::new();
    let mut entity_spans = Vec::new();
    let mut bindings = BindingSet::new();
    let mut partial_elems = 0usize;
    let mut i = total_elems;
    let mut j = end_j;
    while i > 0 {
        let action = states[i][j]
            .as_ref()
            .map(|state| state.action.clone())
            .unwrap_or(Action::SkipElem);
        match action {
            Action::Match {
                len,
                quality,
                span,
                entity,
                binding,
            } => {
                elem_spans[i - 1] = Some(span);
                if quality < 0.999 {
                    partial_elems += 1;
                }
                if let Some(entity) = entity {
                    let key = format!("entity.{}", entities.len() + 1);
                    bindings.bind(&key, &entity);
                    entities.push(entity);
                    entity_spans.push(span);
                }
                if let Some((name, value)) = binding {
                    bindings.bind(name, &value);
                }
                i -= 1;
                j = j.saturating_sub(len);
            }
            Action::SkipElem | Action::Start => {
                i = i.saturating_sub(1);
            }
            Action::SkipToken => {
                j = j.saturating_sub(1);
            }
        }
    }

    let score = if total_weight > 0.0 {
        (best.score / total_weight).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let extra_tokens = total_tokens.saturating_sub(best.matched_tokens);
    Alignment {
        total_elems,
        matched_elems: best.matched_elems,
        partial_elems,
        matched_tokens: best.matched_tokens,
        total_tokens,
        extra_tokens,
        score,
        exact: false,
        elem_spans,
        entity_spans,
        entities,
        bindings,
    }
}

fn update_state(slot: &mut Option<State>, candidate: State) {
    let replace = match slot {
        None => true,
        Some(existing) => {
            candidate.score > existing.score + 1e-9
                || ((candidate.score - existing.score).abs() <= 1e-9
                    && (candidate.matched_elems > existing.matched_elems
                        || (candidate.matched_elems == existing.matched_elems
                            && candidate.matched_tokens > existing.matched_tokens)))
        }
    };
    if replace {
        *slot = Some(candidate);
    }
}

fn elem_weight(elem: &Elem) -> f64 {
    match elem {
        Elem::Word { surface } if is_function_word(surface) => 0.45,
        Elem::Word { .. } => 1.0,
        Elem::Pos { .. } => 0.8,
        Elem::Slot { .. } => 0.9,
        Elem::Any => 0.3,
        Elem::Optional(inner) => 0.5 * elem_weight(inner),
    }
}

/// 虚词/语气词：省略它们不应像漏掉实义词那样重罚。
fn is_function_word(surface: &str) -> bool {
    matches!(
        surface,
        "的" | "了" | "着" | "地" | "得" | "吗" | "呢" | "吧" | "啊" | "呀" | "嘛"
    )
}

fn match_elem(
    elem: Elem,
    tokens: &[TaggedWord],
    index: usize,
    config: &SoftRouteConfig,
) -> Option<ElemHit> {
    match elem {
        Elem::Optional(inner) => match_elem(*inner, tokens, index, config),
        Elem::Word { surface } => word_match(&surface, tokens, index, config),
        Elem::Pos { pos } => pos_match(&pos, &tokens[index]),
        Elem::Slot { name, sort, hints } => slot_match(&name, &sort, &hints, &tokens[index]),
        Elem::Any => Some(ElemHit {
            len: 1,
            quality: 0.6,
            entity: None,
            binding: None,
        }),
    }
}

fn word_match(
    surface: &str,
    tokens: &[TaggedWord],
    index: usize,
    config: &SoftRouteConfig,
) -> Option<ElemHit> {
    let text = tokens[index].text.as_str();
    if text == surface {
        return Some(ElemHit {
            len: 1,
            quality: 1.0,
            entity: None,
            binding: None,
        });
    }
    let surface_len = surface.chars().count();
    let text_len = text.chars().count();
    if surface_len > text_len && !text.is_empty() {
        let mut concat = String::new();
        let limit = (index + 6).min(tokens.len());
        for (offset, token) in tokens[index..limit].iter().enumerate() {
            concat.push_str(&token.text);
            if concat == surface {
                return Some(ElemHit {
                    len: offset + 1,
                    quality: 0.97,
                    entity: None,
                    binding: None,
                });
            }
            if !surface.starts_with(&concat) {
                break;
            }
        }
    }
    if text.starts_with(surface) && !surface.is_empty() {
        let ratio = surface_len as f64 / text_len as f64;
        let quality = 0.65 + 0.30 * ratio;
        if quality >= config.fuzzy_floor {
            return Some(ElemHit {
                len: 1,
                quality,
                entity: None,
                binding: None,
            });
        }
    }
    if surface_len >= 2 && text.contains(surface) {
        return Some(ElemHit {
            len: 1,
            quality: 0.75,
            entity: None,
            binding: None,
        });
    }
    if surface_len >= 2 && surface.contains(text) && !text.is_empty() {
        let ratio = text_len as f64 / surface_len as f64;
        let quality = 0.55 + 0.35 * ratio;
        if quality >= config.fuzzy_floor {
            return Some(ElemHit {
                len: 1,
                quality,
                entity: None,
                binding: None,
            });
        }
    }
    if surface_len >= 2 && text_len >= 2 {
        let distance = edit_distance(text, surface);
        if distance <= 1 {
            let quality = 0.62 - 0.05 * distance as f64;
            if quality >= config.fuzzy_floor {
                return Some(ElemHit {
                    len: 1,
                    quality,
                    entity: None,
                    binding: None,
                });
            }
        }
    }
    None
}

fn pos_match(expected: &str, token: &TaggedWord) -> Option<ElemHit> {
    let actual = token.pos.as_str();
    let quality = if expected == actual {
        1.0
    } else {
        match (expected, actual) {
            ("nr", "unknown") => 1.0,
            ("n", "nz") | ("n", "vn") => 0.85,
            ("n", "nr") | ("n", "ns") => 0.75,
            ("ns", "n") => 0.65,
            ("q", "m") => 0.85,
            ("r", "unknown") => 0.4,
            _ if crate::pos_ontology::tag_matches(expected, actual) => 0.8,
            _ => return None,
        }
    };
    let entity = (matches!(expected, "n" | "nr" | "ns")
        || crate::pos_ontology::captures_entity(expected))
    .then(|| token.text.clone());
    Some(ElemHit {
        len: 1,
        quality,
        entity,
        binding: None,
    })
}

fn slot_match(
    name: &str,
    sort: &str,
    hints: &[String],
    token: &TaggedWord,
) -> Option<ElemHit> {
    if hints.iter().any(|hint| {
        !hint.is_empty() && (token.text.contains(hint.as_str()) || hint.contains(&token.text))
    }) {
        return Some(ElemHit {
            len: 1,
            quality: 1.0,
            entity: Some(token.text.clone()),
            binding: Some((name.to_string(), token.text.clone())),
        });
    }
    let quality = match sort {
        "城市" | "地点" => {
            if crate::pos_ontology::tag_matches("ns", &token.pos) || is_city_literal(&token.text) {
                1.0
            } else if crate::pos_ontology::profile(&token.pos).has_sort(crate::pos_ontology::Sort::Place) {
                0.9
            } else if crate::pos_ontology::tag_matches("n", &token.pos) {
                0.85
            } else {
                return None;
            }
        }
        "日期" | "时间" => {
            if crate::pos_ontology::tag_matches("m", &token.pos)
                && (token.text.contains('天')
                    || token.text.contains('月')
                    || token.text.contains('年')
                    || token.text.contains('日'))
            {
                1.0
            } else if crate::pos_ontology::tag_matches("m", &token.pos) {
                0.8
            } else if crate::pos_ontology::profile(&token.pos).has_sort(crate::pos_ontology::Sort::Time) {
                0.9
            } else {
                return None;
            }
        }
        "人名" => {
            if crate::pos_ontology::tag_matches("nr", &token.pos) {
                1.0
            } else if token.pos == "unknown" {
                0.8
            } else {
                return None;
            }
        }
        "数量" => {
            if crate::pos_ontology::tag_matches("m", &token.pos)
                || (!token.text.is_empty() && token.text.chars().all(|ch| ch.is_numeric()))
            {
                1.0
            } else if crate::pos_ontology::tag_matches("q", &token.pos) {
                0.8
            } else {
                return None;
            }
        }
        "实体" => {
            if matches!(token.pos.as_str(), "nr" | "ns" | "n") {
                1.0
            } else if crate::pos_ontology::is_entity_tag(&token.pos) {
                0.85
            } else {
                return None;
            }
        }
        "属性" => {
            if matches!(token.pos.as_str(), "n" | "nz" | "vn") {
                1.0
            } else if crate::pos_ontology::profile(&token.pos).has_sort(crate::pos_ontology::Sort::Attribute) {
                0.8
            } else if crate::pos_ontology::tag_matches("v", &token.pos) {
                0.7
            } else {
                return None;
            }
        }
        "" | "*" => 0.6,
        _ => 0.7,
    };
    Some(ElemHit {
        len: 1,
        quality,
        entity: Some(token.text.clone()),
        binding: Some((name.to_string(), token.text.clone())),
    })
}

fn is_city_literal(text: &str) -> bool {
    matches!(
        text,
        "厦门" | "北京" | "上海" | "广州" | "深圳" | "杭州" | "南京" | "武汉" | "成都" | "重庆"
    )
}

fn is_skippable(text: &str) -> bool {
    text.trim().is_empty() || matches!(text, "？" | "。" | "，" | "、" | "！" | "；" | ":" | "：")
}

fn edit_distance(left: &str, right: &str) -> usize {
    let left: Vec<char> = left.chars().collect();
    let right: Vec<char> = right.chars().collect();
    if left.is_empty() {
        return right.len();
    }
    if right.is_empty() {
        return left.len();
    }
    let mut previous: Vec<usize> = (0..=right.len()).collect();
    for (i, left_char) in left.iter().enumerate() {
        let mut current = vec![i + 1; right.len() + 1];
        for (j, right_char) in right.iter().enumerate() {
            let cost = if left_char == right_char { 0 } else { 1 };
            current[j + 1] = (previous[j] + cost)
                .min(previous[j + 1] + 1)
                .min(current[j] + 1);
        }
        previous = current;
    }
    previous[right.len()]
}

/// 一条构念变体上的软路由结果（只保留需要复核的样例）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SoftEvalCase {
    pub question: String,
    pub expected_query_type: String,
    pub actual_query_type: Option<String>,
    pub variant: String,
    pub confidence: f64,
    pub alignment: f64,
}

/// 构念变体上的软路由体检：精确主路由基线、近失恢复、错误接受与澄清。
///
/// 分母来自配置规则的表面变体（同义词/实体替换/尾缀/礼貌前缀），
/// **不是人工转述**；它只回答「软层有没有在原有精确主路由之外制造新错误」，
/// 不能替代真实用户语料的评估。NLU 回退排在软层之后，不计入这里的精确基线。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SoftEvalReport {
    pub dialect: String,
    pub cases: usize,
    pub exact_correct: usize,
    pub exact_wrong: usize,
    pub exact_unanswered: usize,
    pub soft_recovered: usize,
    pub soft_wrong_accept: usize,
    pub soft_clarify: usize,
    pub soft_reject: usize,
    pub correct: usize,
    pub recoveries: Vec<SoftEvalCase>,
    pub false_accepts: Vec<SoftEvalCase>,
    pub clarifications: Vec<SoftEvalCase>,
}

impl SoftEvalReport {
    /// 精确路由之后的增量召回。
    pub fn soft_recall_gain(&self) -> f64 {
        if self.cases == 0 {
            0.0
        } else {
            self.soft_recovered as f64 / self.cases as f64
        }
    }

    /// 软层在「精确未命中」上接受时的错误率。
    pub fn soft_false_accept_rate(&self) -> f64 {
        let accepted = self.soft_recovered + self.soft_wrong_accept;
        if accepted == 0 {
            0.0
        } else {
            self.soft_wrong_accept as f64 / accepted as f64
        }
    }

    pub fn render_text(&self) -> String {
        let mut output = String::new();
        let _ = writeln!(output, "软路由体检 {}", self.dialect);
        let _ = writeln!(
            output,
            "分母：{} 条构念变体（配置规则表面变体，非人工转述）",
            self.cases
        );
        let _ = writeln!(
            output,
            "精确主路由：正确 {}，错误 {}，未命中 {}",
            self.exact_correct, self.exact_wrong, self.exact_unanswered
        );
        let _ = writeln!(
            output,
            "软层增量：恢复 {}（+{:.1} 个百分点），错误接受 {}（未命中上的错误率 {:.1}%），澄清 {}，回退 {}",
            self.soft_recovered,
            self.soft_recall_gain() * 100.0,
            self.soft_wrong_accept,
            self.soft_false_accept_rate() * 100.0,
            self.soft_clarify,
            self.soft_reject
        );
        let _ = writeln!(
            output,
            "总命中：{} / {}（{:.1}%）",
            self.correct,
            self.cases,
            if self.cases == 0 {
                0.0
            } else {
                self.correct as f64 / self.cases as f64 * 100.0
            }
        );
        if !self.recoveries.is_empty() {
            let _ = writeln!(output, "近失恢复样例：");
            for case in self.recoveries.iter().take(8) {
                let _ = writeln!(
                    output,
                    "  · {} → {}（{}，{:.2}，对齐 {:.0}%）",
                    case.question,
                    case.expected_query_type,
                    case.variant,
                    case.confidence,
                    case.alignment * 100.0
                );
            }
        }
        if !self.false_accepts.is_empty() {
            let _ = writeln!(output, "错误接受样例：");
            for case in self.false_accepts.iter().take(8) {
                let _ = writeln!(
                    output,
                    "  · {} → 期望 {}，实得 {}（{}，{:.2}）",
                    case.question,
                    case.expected_query_type,
                    case.actual_query_type.as_deref().unwrap_or("<无>"),
                    case.variant,
                    case.confidence
                );
            }
        }
        if !self.clarifications.is_empty() {
            let _ = writeln!(output, "澄清样例：");
            for case in self.clarifications.iter().take(8) {
                let _ = writeln!(
                    output,
                    "  · {} → 期望 {}（首选 {}，{:.2}）",
                    case.question,
                    case.expected_query_type,
                    case.actual_query_type.as_deref().unwrap_or("<无>"),
                    case.confidence
                );
            }
        }
        output
    }
}

#[derive(Default)]
struct SoftEvalTally {
    recovered: usize,
    wrong_accept: usize,
    clarify: usize,
    reject: usize,
    recoveries: Vec<SoftEvalCase>,
    false_accepts: Vec<SoftEvalCase>,
    clarifications: Vec<SoftEvalCase>,
}

impl SoftEvalTally {
    fn record(&mut self, case: &crate::recall_audit::RecallCase, outcome: &SoftRouteOutcome) {
        match outcome.decision {
            SoftDecision::Accept => {
                let Some(selected) = &outcome.selected else {
                    self.reject += 1;
                    return;
                };
                let entry = SoftEvalCase {
                    question: case.question.clone(),
                    expected_query_type: case.expected_query_type.clone(),
                    actual_query_type: Some(selected.query_type().to_string()),
                    variant: case.variant.label().to_string(),
                    confidence: selected.confidence,
                    alignment: selected.alignment.score,
                };
                if selected.query_type() == case.expected_query_type {
                    self.recovered += 1;
                    if self.recoveries.len() < 40 {
                        self.recoveries.push(entry);
                    }
                } else {
                    self.wrong_accept += 1;
                    if self.false_accepts.len() < 40 {
                        self.false_accepts.push(entry);
                    }
                }
            }
            SoftDecision::Clarify => {
                self.clarify += 1;
                if self.clarifications.len() < 40 {
                    self.clarifications.push(SoftEvalCase {
                        question: case.question.clone(),
                        expected_query_type: case.expected_query_type.clone(),
                        actual_query_type: outcome
                            .selected
                            .as_ref()
                            .map(|candidate| candidate.query_type().to_string()),
                        variant: case.variant.label().to_string(),
                        confidence: outcome
                            .selected
                            .as_ref()
                            .map(|candidate| candidate.confidence)
                            .unwrap_or(0.0),
                        alignment: outcome
                            .selected
                            .as_ref()
                            .map(|candidate| candidate.alignment.score)
                            .unwrap_or(0.0),
                    });
                }
            }
            SoftDecision::Reject => self.reject += 1,
        }
    }
}

/// 在配置构念变体上评估软路由（离线、不打开存储）。
pub fn evaluate_constructed(config: &crate::SystemConfig) -> SoftEvalReport {
    use crate::nlp::{normalize_words, preprocess};

    let library = PatternLibrary::from_config(config);
    let synonyms = config
        .synonyms
        .as_ref()
        .map(|synonyms| synonyms.mappings.clone())
        .unwrap_or_default();
    let (canonical_cases, _) = crate::recall_audit::construct_cases(&library, &synonyms);
    let mut cases = canonical_cases.clone();
    // 在字面句上删除一个虚词，构造可控的近失；这些样例只用于回归，
    // 不冒充真实用户转述。
    for case in &canonical_cases {
        if case.variant != crate::recall_audit::CaseVariant::Canonical {
            continue;
        }
        for particle in ["的", "是", "了"] {
            let Some(position) = case.question.find(particle) else {
                continue;
            };
            let question = format!(
                "{}{}",
                &case.question[..position],
                &case.question[position + particle.len()..]
            );
            if question == case.question || cases.iter().any(|item| item.question == question) {
                continue;
            }
            let mut near_miss = case.clone();
            near_miss.question = question;
            near_miss.seen = false;
            cases.push(near_miss);
        }
    }
    let context = PatternContext::default();
    let mut exact_correct = 0usize;
    let mut exact_wrong = 0usize;
    let mut exact_unanswered = 0usize;
    let mut tally = SoftEvalTally::default();

    for case in &cases {
        let tagged = preprocess(&case.question, &config.nlp);
        let tokens = normalize_words(tagged, &synonyms);
        let exact = library.match_route_with_context(&tokens, &context);

        let mut soft_outcome = None;
        if let Some(matched) = &exact {
            if matched.entry.defer_to_soft && config.reasoning.soft_route.enabled {
                let outcome = route_tokens(
                    &case.question,
                    &library,
                    &tokens,
                    &context,
                    &config.reasoning.soft_route,
                );
                let replaces = outcome.decision == SoftDecision::Accept
                    && outcome
                        .selected
                        .as_ref()
                        .is_some_and(|selected| selected.entry.id() != matched.entry.id());
                if replaces {
                    soft_outcome = Some(outcome);
                }
            }
        }

        if let Some(outcome) = soft_outcome {
            exact_unanswered += 1;
            tally.record(case, &outcome);
            continue;
        }

        if let Some(matched) = exact {
            if matched.entry.query_type == case.expected_query_type {
                exact_correct += 1;
            } else {
                exact_wrong += 1;
            }
            continue;
        }

        exact_unanswered += 1;
        let outcome = route_tokens(
            &case.question,
            &library,
            &tokens,
            &context,
            &config.reasoning.soft_route,
        );
        tally.record(case, &outcome);
    }

    SoftEvalReport {
        dialect: DIALECT.to_string(),
        cases: cases.len(),
        exact_correct,
        exact_wrong,
        exact_unanswered,
        soft_recovered: tally.recovered,
        soft_wrong_accept: tally.wrong_accept,
        soft_clarify: tally.clarify,
        soft_reject: tally.reject,
        correct: exact_correct + tally.recovered,
        recoveries: tally.recoveries,
        false_accepts: tally.false_accepts,
        clarifications: tally.clarifications,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pattern_ir::PatternEntry;

    fn tokens(items: &[(&str, &str)]) -> Vec<TaggedWord> {
        items
            .iter()
            .map(|(text, pos)| TaggedWord {
                text: (*text).to_string(),
                pos: (*pos).to_string(),
            })
            .collect()
    }

    fn library(patterns: &[(&str, &str, Vec<Elem>)]) -> PatternLibrary {
        let mut library = PatternLibrary::new();
        for (name, query_type, elems) in patterns {
            library.add(PatternEntry::new(
                *name,
                format!("test#{}", name),
                *query_type,
                10,
                true,
                elems.clone(),
            ));
        }
        library
    }

    fn capital_library() -> PatternLibrary {
        library(&[
            (
                "首都查询",
                "capital_query",
                vec![
                    Elem::pos("ns"),
                    Elem::word("的"),
                    Elem::word("首都"),
                    Elem::word("是"),
                    Elem::word("哪里"),
                ],
            ),
            (
                "首都查询2",
                "capital_query",
                vec![Elem::pos("ns"), Elem::word("首都")],
            ),
            (
                "海拔查询",
                "altitude_query",
                vec![
                    Elem::pos("ns"),
                    Elem::word("的"),
                    Elem::word("海拔"),
                    Elem::word("是"),
                    Elem::word("多少"),
                ],
            ),
        ])
    }

    #[test]
    fn pos_closure_accepts_specific_noun_and_classifier_tags() {
        let token = TaggedWord {
            text: "某组织".to_string(),
            pos: "nt".to_string(),
        };
        assert!(pos_match("n", &token).is_some());
        let mq = TaggedWord {
            text: "三个".to_string(),
            pos: "mq".to_string(),
        };
        assert!(pos_match("q", &mq).is_some());
        let place = TaggedWord {
            text: "某地".to_string(),
            pos: "s".to_string(),
        };
        assert_eq!(slot_match("城市", "地点", &[], &place).map(|hit| hit.quality), Some(0.9));
    }

    #[test]
    fn exact_match_keeps_existing_semantics() {
        let library = capital_library();
        let words = tokens(&[
            ("法国", "ns"),
            ("的", "u"),
            ("首都", "n"),
            ("是", "v"),
            ("哪里", "r"),
        ]);
        let config = SoftRouteConfig::default();
        let outcome = route_tokens("法国的首都是哪里", &library, &words, &PatternContext::default(), &config);
        assert_eq!(outcome.decision, SoftDecision::Accept);
        let selected = outcome.selected.expect("应接受精确候选");
        assert!(selected.alignment.exact);
        assert_eq!(selected.query_type(), "capital_query");
        assert_eq!(selected.alignment.entities, vec!["法国"]);
        assert_eq!(selected.confidence, selected.confidence.clamp(0.0, 1.0));
    }

    #[test]
    fn near_miss_with_missing_particle_is_recoverable() {
        let library = library(&[(
            "完整首都查询",
            "capital_query",
            vec![
                Elem::pos("ns"),
                Elem::word("的"),
                Elem::word("首都"),
                Elem::word("是"),
                Elem::word("哪里"),
            ],
        )]);
        let words = tokens(&[
            ("法国", "ns"),
            ("首都", "n"),
            ("是", "v"),
            ("哪里", "r"),
        ]);
        let config = SoftRouteConfig::default();
        let outcome = route_tokens("法国首都是哪里", &library, &words, &PatternContext::default(), &config);
        assert_eq!(outcome.decision, SoftDecision::Accept, "{}", outcome.render_report("法国首都是哪里"));
        let selected = outcome.selected.expect("应接受近失候选");
        assert!(!selected.alignment.exact);
        assert_eq!(selected.query_type(), "capital_query");
        assert_eq!(selected.alignment.matched_elems, 4);
        assert_eq!(selected.alignment.extra_tokens, 0);
    }

    #[test]
    fn short_entity_only_input_is_rejected() {
        let library = capital_library();
        let words = tokens(&[("法国", "ns")]);
        let config = SoftRouteConfig::default();
        let outcome = route_tokens("法国", &library, &words, &PatternContext::default(), &config);
        assert_eq!(outcome.decision, SoftDecision::Reject);
        assert!(outcome.selected.is_none());
    }

    #[test]
    fn ambiguous_routes_ask_for_clarification() {
        let library = library(&[
            (
                "询问首都",
                "capital_query",
                vec![Elem::pos("ns"), Elem::word("哪里")],
            ),
            (
                "询问位置",
                "location_query",
                vec![Elem::pos("ns"), Elem::word("哪里")],
            ),
            (
                "询问人口",
                "population_query",
                vec![Elem::pos("ns"), Elem::word("多少")],
            ),
        ]);
        let words = tokens(&[("北京", "ns"), ("哪里", "r")]);
        let config = SoftRouteConfig::default();
        let outcome = route_tokens("北京哪里", &library, &words, &PatternContext::default(), &config);
        assert_eq!(outcome.decision, SoftDecision::Clarify);
        assert!(outcome.clarification_options().len() >= 2);
    }

    #[test]
    fn guard_is_a_hard_gate_not_a_soft_hint() {
        let mut library = PatternLibrary::new();
        library.add(
            PatternEntry::new(
                "省略首都",
                "test#ellipsis",
                "capital_query",
                10,
                true,
                vec![Elem::word("首都"), Elem::word("哪里")],
            )
            .with_guard(crate::pattern_ir::ContextGuard::focus()),
        );
        let words = tokens(&[("首都", "n"), ("哪里", "r")]);
        let config = SoftRouteConfig::default();
        let empty = route_tokens("首都哪里", &library, &words, &PatternContext::default(), &config);
        assert_eq!(empty.decision, SoftDecision::Reject);
        let mut context = PatternContext::default();
        context.focus = Some("北京".to_string());
        let with_focus = route_tokens("首都哪里", &library, &words, &context, &config);
        assert_eq!(with_focus.decision, SoftDecision::Accept);
        assert!(with_focus.selected.unwrap().evidence.iter().any(|term| term.name == "上下文"));
    }

    #[test]
    fn explicit_forms_prefer_route_over_fallback() {
        let mut library = PatternLibrary::new();
        library.add(PatternEntry::new(
            "配置首都",
            "config#capital",
            "capital_query",
            10,
            true,
            vec![Elem::pos("ns"), Elem::word("首都")],
        ));
        let mut fallback = PatternEntry::new(
            "NLU首都",
            "nlu#geo#capital",
            "capital_query",
            10,
            false,
            vec![Elem::pos("ns"), Elem::word("首都")],
        );
        fallback.fallback = true;
        library.add(fallback);
        let words = tokens(&[("法国", "ns"), ("首都", "n")]);
        let config = SoftRouteConfig::default();
        let outcome = route_tokens("法国首都", &library, &words, &PatternContext::default(), &config);
        assert_eq!(outcome.decision, SoftDecision::Accept);
        assert_eq!(outcome.selected.unwrap().entry.source, "config#capital");
    }

    #[test]
    fn soft_layer_precedes_generic_fallback() {
        let mut library = PatternLibrary::new();
        library.add(PatternEntry::new(
            "完整海拔查询",
            "config#altitude",
            "altitude_query",
            10,
            true,
            vec![
                Elem::pos("ns"),
                Elem::word("海拔"),
                Elem::word("多少"),
            ],
        ));
        let mut fallback = PatternEntry::new(
            "泛化属性回退",
            "nlu#kg#property",
            "property_query",
            10,
            false,
            vec![Elem::pos("ns"), Elem::word("海拔")],
        );
        fallback.fallback = true;
        library.add(fallback);
        let words = tokens(&[
            ("北京", "ns"),
            ("海拔", "n"),
            ("高度", "n"),
            ("多少", "r"),
        ]);
        let context = PatternContext::default();
        assert!(library.match_route_with_context(&words, &context).is_none());
        assert_eq!(
            library
                .match_fallback_with_context(&words, &context)
                .expect("回退应命中泛化属性")
                .entry
                .query_type,
            "property_query"
        );
        let outcome = route_tokens(
            "北京海拔高度多少",
            &library,
            &words,
            &context,
            &SoftRouteConfig::default(),
        );
        assert_eq!(outcome.decision, SoftDecision::Accept);
        assert_eq!(outcome.selected.unwrap().query_type(), "altitude_query");
    }

    #[test]
    fn generic_property_rule_opts_into_soft_recheck() {
        let config: crate::SystemConfig =
            toml::from_str(include_str!("../config.toml")).expect("config.toml 必须可解析");
        let library = PatternLibrary::from_config(&config);
        let entry = library
            .entries()
            .iter()
            .find(|entry| entry.source == "config.toml#属性查询")
            .expect("配置中应有属性查询规则");
        assert!(entry.route);
        assert!(entry.defer_to_soft);
    }

    #[test]
    fn report_is_readable_and_auditable() {
        let library = capital_library();
        let words = tokens(&[
            ("法国", "ns"),
            ("首都", "n"),
            ("是", "v"),
            ("哪里", "r"),
        ]);
        let config = SoftRouteConfig::default();
        let outcome = route_tokens("法国首都是哪里", &library, &words, &PatternContext::default(), &config);
        let report = outcome.render_report("法国首都是哪里");
        assert!(report.contains("软路由"), "{}", report);
        assert!(report.contains("判定：接受"), "{}", report);
        assert!(report.contains("对齐：要素"), "{}", report);
        assert!(report.contains("log-odds"), "{}", report);
    }

    #[test]
    fn constructed_eval_reports_incremental_recall_without_counting_errors_as_hits() {
        let config: crate::SystemConfig =
            toml::from_str(include_str!("../config.toml")).expect("config.toml 必须可解析");
        let report = evaluate_constructed(&config);
        assert!(report.cases > 0);
        assert!(report.soft_recovered > 0, "{}", report.render_text());
        assert_eq!(report.soft_wrong_accept, 0, "{}", report.render_text());
        assert_eq!(report.correct, report.exact_correct + report.soft_recovered);
        assert_eq!(
            report.cases,
            report.exact_correct
                + report.exact_wrong
                + report.exact_unanswered
        );
        assert_eq!(
            report.exact_unanswered,
            report.soft_recovered
                + report.soft_wrong_accept
                + report.soft_clarify
                + report.soft_reject
        );
        assert!(report.soft_false_accept_rate() >= 0.0);
        assert!(report.render_text().contains("软路由体检"));
    }

    #[test]
    fn config_validation_rejects_inverted_thresholds() {
        let mut config = SoftRouteConfig::default();
        config.clarify_threshold = 0.9;
        config.accept_threshold = 0.5;
        assert!(config.validate().is_err());
        config.clarify_threshold = 0.2;
        config.accept_threshold = 0.7;
        assert!(config.validate().is_ok());
        config.alignment_scale = 0.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn clarify_before_fallback_defaults_on_for_legacy_configs() {
        let legacy: SoftRouteConfig = toml::from_str("").expect("空配置应使用全部默认值");
        assert!(legacy.clarify_before_fallback);
        let explicit: SoftRouteConfig =
            toml::from_str("clarify_before_fallback = false").expect("显式关闭应可解析");
        assert!(!explicit.clarify_before_fallback);
    }
}

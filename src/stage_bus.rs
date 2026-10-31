//! 阶段总线（Stage Bus）：让模块之间传递「值 + 证据 + 置信区间 + 残差」。
//!
//! 系统里已经有很多成熟的单点能力：`soft_route` 给出意图后验，`plugin` 给
//! 出声明式改写，`epistemic_turn` 给逐命题复核队列，`confidence` 给出
//! PCC 的校准、融合与四档决策。它们各自都自洽，但彼此之间传的仍然只是
//! 「值」。本模块补的是一层很薄的契约，不重写任何已有模块：
//!
//! - [`ModuleCard`]：模块声明自己处在哪个阶段、提供/依赖哪些能力、成本、
//!   是否为确定性计算、对应哪条置信来源、失败时回退给谁；
//! - [`StageEnvelope`]：跨阶段传递的统一信封（值、原始分、区间、见证、
//!   残差、成本）；
//! - [`EvidenceBus`]：把各阶段的观测交给既有 PCC 融合，保留逐阶段读数、
//!   冲突与失败原因；同一来源不重复计票；
//! - [`BeliefLedger`]：跨轮保存逐槽位信念，按回合衰减、有界去重；
//! - [`ClarificationPlanner`]：在候选读法中挑选「最能劈开候选集」的澄清
//!   问题，而不是把全部选项原样抛给用户。
//!
//! 设计约束：
//!
//! - **默认不接管**：本模块是纯类型与纯函数层，调用方不接入时旧路径不变；
//! - **复用 PCC**：校准、log-odds 融合与四档决策全部走
//!   [`crate::confidence::ConfidenceConfig`]，这里不另造概率论；
//! - **区间优先**：两个候选冲突时保留「争议」状态，不把矛盾平均成一个
//!   看似确定的数；
//! - **可复算**：衰减、覆盖与澄清收益都用整数万分比，结果可逐项对账；
//! - **有界**：信念账本与见证列表都有容量上限，不随会话无限增长。

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::answer_plan::AnswerPlan;
use crate::confidence::{
    ClaimEnvelope, ConfidenceConfig, DecisionRecord, EnvelopeGrade, EnvelopeVerdict,
    ENVELOPE_SCALE,
};
use crate::epistemic_turn::TurnEnvelope;
use crate::plugin::PluginOutcome;
use crate::soft_route::{SoftDecision, SoftRouteOutcome};

/// 报告方言：读取方据此拒绝把旧版账本当新版解释。
pub const DIALECT: &str = "knowprolog-stage-bus/1";

/// 信念每经过一轮没有新证据时的衰减量（万分比）。
///
/// 只作用于会话层的跨轮视图，不改写原始命题信封；衰减后下界下降、上界
/// 向 1 靠近，表达的是「越久没复核，越不该当作确定事实」。
pub const BELIEF_DECAY_BPS: u16 = 200;

/// 信念账本默认容量（按槽位计）。
pub const DEFAULT_BELIEF_CAPACITY: usize = 64;

/// 单条信封最多保留的见证数。
pub const MAX_WITNESSES: usize = 8;

/// 一次回答的生命周期阶段。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    /// 意图与问句理解。
    Intent,
    /// 词表/问法改写。
    Rewrite,
    /// 证据召回。
    Retrieve,
    /// 符号推理与一致性检查。
    Reason,
    /// 生成与措辞。
    Compose,
}

impl Stage {
    pub fn code(self) -> &'static str {
        match self {
            Stage::Intent => "intent",
            Stage::Rewrite => "rewrite",
            Stage::Retrieve => "retrieve",
            Stage::Reason => "reason",
            Stage::Compose => "compose",
        }
    }

    pub fn label_zh(self) -> &'static str {
        match self {
            Stage::Intent => "意图",
            Stage::Rewrite => "改写",
            Stage::Retrieve => "召回",
            Stage::Reason => "推理",
            Stage::Compose => "生成",
        }
    }
}

/// 模块能力声明。
///
/// 这是「模块市场」的最小单元：宿主只读取声明，不执行模块代码。`source`
/// 指向 `[reasoning.confidence.sources]` 里的来源名；`fallback` 是失败时
/// 应退回的模块 id（例如 `soft_route` → `pattern`）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModuleCard {
    pub id: String,
    pub stage: Stage,
    #[serde(default)]
    pub provides: Vec<String>,
    #[serde(default)]
    pub requires: Vec<String>,
    #[serde(default = "default_true")]
    pub deterministic: bool,
    #[serde(default)]
    pub cost_units: u16,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fallback: Option<String>,
}

fn default_true() -> bool {
    true
}

impl ModuleCard {
    pub fn new(id: impl Into<String>, stage: Stage) -> Self {
        Self {
            id: id.into(),
            stage,
            provides: Vec::new(),
            requires: Vec::new(),
            deterministic: true,
            cost_units: 0,
            source: None,
            fallback: None,
        }
    }

    pub fn with_provides<I, S>(mut self, provides: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.provides = clean_terms(provides);
        self
    }

    pub fn with_requires<I, S>(mut self, requires: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.requires = clean_terms(requires);
        self
    }

    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        let source = source.into();
        self.source = (!source.trim().is_empty()).then(|| source.trim().to_string());
        self
    }

    pub fn with_fallback(mut self, fallback: impl Into<String>) -> Self {
        let fallback = fallback.into();
        self.fallback = (!fallback.trim().is_empty()).then(|| fallback.trim().to_string());
        self
    }

    pub fn with_cost(mut self, cost_units: u16) -> Self {
        self.cost_units = cost_units;
        self
    }

    pub fn non_deterministic(mut self) -> Self {
        self.deterministic = false;
        self
    }

    pub fn covers(&self, stage: Stage) -> bool {
        self.stage == stage
    }

    /// 静态校验：空 id、空能力名、自依赖都返回稳定的问题码。
    pub fn validate(&self) -> Vec<String> {
        let mut issues = Vec::new();
        if self.id.trim().is_empty() {
            issues.push("module_empty_id".to_string());
        }
        if self.provides.iter().any(|item| item.is_empty()) {
            issues.push("module_empty_capability".to_string());
        }
        if self
            .requires
            .iter()
            .any(|item| item == &self.id || item.is_empty())
        {
            issues.push("module_invalid_requirement".to_string());
        }
        if self
            .source
            .as_ref()
            .is_some_and(|source| source.trim().is_empty())
        {
            issues.push("module_empty_source".to_string());
        }
        issues
    }
}

fn clean_terms<I, S>(items: I) -> Vec<String>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut terms: Vec<String> = items
        .into_iter()
        .map(Into::into)
        .map(|item: String| item.trim().to_string())
        .filter(|item| !item.is_empty())
        .collect();
    terms.sort();
    terms.dedup();
    terms
}

/// 小型模块目录：登记、按阶段查找、报告缺失依赖。
///
/// 它不是任务调度器；阶段顺序仍由现有主路径固定。目录只负责把「谁提供了
/// 什么」变成可检查的数据，避免新增能力时到处改分支。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModuleCatalog {
    #[serde(default)]
    cards: BTreeMap<String, ModuleCard>,
}

impl ModuleCatalog {
    pub fn new() -> Self {
        Self::default()
    }

    /// 登记一个模块；重复 id 会被拒绝并返回问题码。
    pub fn register(&mut self, card: ModuleCard) -> Vec<String> {
        let mut issues = card.validate();
        if self.cards.contains_key(&card.id) {
            issues.push("module_duplicate_id".to_string());
        }
        if issues.is_empty() {
            self.cards.insert(card.id.clone(), card);
        }
        issues
    }

    pub fn get(&self, id: &str) -> Option<&ModuleCard> {
        self.cards.get(id)
    }

    pub fn cards(&self) -> impl Iterator<Item = &ModuleCard> {
        self.cards.values()
    }

    pub fn for_stage(&self, stage: Stage) -> Vec<&ModuleCard> {
        self.cards
            .values()
            .filter(|card| card.covers(stage))
            .collect()
    }

    /// 返回所有「声明了依赖但目录里没有」的 `(module, missing)`。
    pub fn missing_dependencies(&self) -> Vec<(String, String)> {
        let mut missing = Vec::new();
        for card in self.cards.values() {
            for required in &card.requires {
                if !self.cards.contains_key(required) {
                    missing.push((card.id.clone(), required.clone()));
                }
            }
        }
        missing.sort();
        missing
    }

    pub fn render_zh(&self) -> String {
        let mut out = format!("模块目录：{} 个模块。", self.cards.len());
        for card in self.cards.values() {
            out.push_str(&format!(
                "\n  {} [{}] 提供：{}；依赖：{}；来源：{}；成本：{}",
                card.id,
                card.stage.label_zh(),
                join_or_dash(&card.provides),
                join_or_dash(&card.requires),
                card.source.as_deref().unwrap_or("-"),
                card.cost_units
            ));
        }
        let missing = self.missing_dependencies();
        if !missing.is_empty() {
            out.push_str("\n缺失依赖：");
            for (module, required) in missing {
                out.push_str(&format!("\n  {} → {}", module, required));
            }
        }
        out
    }
}

fn join_or_dash(items: &[String]) -> String {
    if items.is_empty() {
        "-".to_string()
    } else {
        items.join("、")
    }
}

/// 一条可核查的见证：哪个来源、提供了什么细节。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Witness {
    pub source: String,
    pub detail: String,
}

impl Witness {
    pub fn new(source: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            source: source.into().trim().to_string(),
            detail: truncate_chars(&detail.into(), 160),
        }
    }
}

/// 跨阶段传递的统一信封。
///
/// `score` 是模块自己的原始分（0..1）；`envelope` 是已经过来源等级/冲突
/// 修正的逐命题区间。两者都保留，因为「模块多有把握」和「结论多可信」
/// 不是同一个问题。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StageEnvelope<T> {
    pub stage: Stage,
    pub module: String,
    pub value: T,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
    pub envelope: ClaimEnvelope,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub witnesses: Vec<Witness>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub residual: Option<String>,
    #[serde(default)]
    pub cost_units: u16,
}

impl<T> StageEnvelope<T> {
    pub fn new(stage: Stage, module: impl Into<String>, value: T, envelope: ClaimEnvelope) -> Self {
        Self {
            stage,
            module: module.into(),
            value,
            score: None,
            envelope,
            witnesses: Vec::new(),
            residual: None,
            cost_units: 0,
        }
    }

    /// 用点估计构造信封；调用方应只在没有更细证据时使用。
    pub fn from_point(
        stage: Stage,
        module: impl Into<String>,
        value: T,
        score: f64,
    ) -> Self {
        let score = clamp_score(score);
        Self {
            stage,
            module: module.into(),
            value,
            score: Some(score),
            envelope: ClaimEnvelope::point(score),
            witnesses: Vec::new(),
            residual: None,
            cost_units: 0,
        }
    }

    pub fn with_score(mut self, score: f64) -> Self {
        self.score = Some(clamp_score(score));
        self
    }

    pub fn with_witness(mut self, witness: Witness) -> Self {
        if self.witnesses.len() < MAX_WITNESSES
            && !self
                .witnesses
                .iter()
                .any(|existing| existing == &witness)
        {
            self.witnesses.push(witness);
        }
        self
    }

    pub fn with_residual(mut self, residual: impl Into<String>) -> Self {
        let residual = residual.into();
        self.residual = (!residual.trim().is_empty())
            .then(|| truncate_chars(residual.trim(), 200));
        self
    }

    pub fn with_cost(mut self, cost_units: u16) -> Self {
        self.cost_units = cost_units;
        self
    }

    pub fn raw(&self) -> Option<f64> {
        self.score.map(clamp_score)
    }

    pub fn verdict(&self, config: &ConfidenceConfig) -> EnvelopeVerdict {
        config.envelope_verdict(self.envelope)
    }

    pub fn assertable(&self, config: &ConfidenceConfig) -> bool {
        self.verdict(config).asserts()
    }

    pub fn render_zh(&self, config: &ConfidenceConfig) -> String {
        let raw = self
            .score
            .map(|score| format!("，原始分 {:.2}", clamp_score(score)))
            .unwrap_or_default();
        format!(
            "{}段「{}」：{}{}；区间 {}",
            self.stage.label_zh(),
            self.module,
            config.render_envelope(self.envelope),
            raw,
            self.envelope.render_interval()
        )
    }
}

/// EvidenceBus 记录的一条观测。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Observation {
    pub stage: Stage,
    pub module: String,
    pub source: String,
    pub raw: f64,
    pub envelope: ClaimEnvelope,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub witnesses: Vec<Witness>,
    #[serde(default)]
    pub cost_units: u16,
}

/// 某个阶段明确失败的原因；失败不是「没有证据」，必须单独保留。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageFailure {
    pub stage: Stage,
    pub module: String,
    pub reason: String,
}

impl StageFailure {
    pub fn new(stage: Stage, module: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            stage,
            module: module.into(),
            reason: truncate_chars(&reason.into(), 200),
        }
    }
}

/// 逐阶段读数：保留原始模块、来源、见证与区间，便于逐行对账。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StageReading {
    pub stage: Stage,
    pub module: String,
    pub source: String,
    pub raw: f64,
    pub envelope: ClaimEnvelope,
    pub witnesses: Vec<Witness>,
}

/// 同阶段两个模块的区间关系。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IntervalRelation {
    /// 左侧下界已经高于右侧上界。
    Dominates,
    /// 右侧下界已经高于左侧上界。
    Dominated,
    /// 区间完全一致。
    Equal,
    /// 区间重叠且互不支配。
    Contested,
}

impl IntervalRelation {
    pub fn label_zh(self) -> &'static str {
        match self {
            IntervalRelation::Dominates => "支配",
            IntervalRelation::Dominated => "被支配",
            IntervalRelation::Equal => "相同",
            IntervalRelation::Contested => "争议",
        }
    }
}

/// 比较两个区间：只有严格分离才算支配；重叠保留争议。
pub fn compare_envelopes(left: ClaimEnvelope, right: ClaimEnvelope) -> IntervalRelation {
    if left == right {
        return IntervalRelation::Equal;
    }
    if left.lower_bps() > right.upper_bps() {
        IntervalRelation::Dominates
    } else if right.lower_bps() > left.upper_bps() {
        IntervalRelation::Dominated
    } else {
        IntervalRelation::Contested
    }
}

/// 同阶段两个读数之间的冲突记录。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageConflict {
    pub stage: Stage,
    pub left: String,
    pub right: String,
    pub relation: IntervalRelation,
}

/// 一次阶段总线的决策结果。
#[derive(Debug, Clone, PartialEq)]
pub struct BusDecision {
    pub record: DecisionRecord,
    pub readings: Vec<StageReading>,
    pub conflicts: Vec<StageConflict>,
    pub failures: Vec<StageFailure>,
    pub cost_units: u16,
}

impl BusDecision {
    /// 决策是否被证据总线抬到可断言；拒答与澄清都返回 false。
    pub fn asserts(&self) -> bool {
        self.record.decision.asserts()
    }

    /// 失败回执：第一段给用户看的说明必须明确「这是故障，不是没记录」。
    pub fn failure_note(&self) -> Option<String> {
        self.failures.first().map(|failure| {
            format!(
                "{}阶段「{}」未能完成（{}）；这不是知识库中没有记录。",
                failure.stage.label_zh(),
                failure.module,
                failure.reason
            )
        })
    }

    pub fn render_receipt(&self) -> String {
        let mut out = format!(
            "阶段总线 {}：{}；读数 {} 条；冲突 {} 条；失败 {} 条；成本 {}",
            DIALECT,
            self.record.render_receipt(),
            self.readings.len(),
            self.conflicts.len(),
            self.failures.len(),
            self.cost_units
        );
        for reading in &self.readings {
            out.push_str(&format!(
                "\n  {}·{}（来源 {}，原始分 {:.2}）：{}",
                reading.stage.label_zh(),
                reading.module,
                reading.source,
                clamp_score(reading.raw),
                reading.envelope.render_interval()
            ));
        }
        for conflict in &self.conflicts {
            out.push_str(&format!(
                "\n  冲突：{} 与 {} {}",
                conflict.left,
                conflict.right,
                conflict.relation.label_zh()
            ));
        }
        for failure in &self.failures {
            out.push_str(&format!(
                "\n  失败：{}·{}：{}",
                failure.stage.label_zh(),
                failure.module,
                failure.reason
            ));
        }
        out
    }
}

/// 置信度携带的模块总线。
#[derive(Debug, Clone, PartialEq)]
pub struct EvidenceBus {
    understanding: f64,
    near_boundary: bool,
    observations: Vec<Observation>,
    failures: Vec<StageFailure>,
}

impl EvidenceBus {
    pub fn new(understanding: f64) -> Self {
        Self {
            understanding: clamp_score(understanding),
            near_boundary: false,
            observations: Vec::new(),
            failures: Vec::new(),
        }
    }

    pub fn with_near_boundary(mut self, near_boundary: bool) -> Self {
        self.near_boundary = near_boundary;
        self
    }

    pub fn understanding(&self) -> f64 {
        self.understanding
    }

    pub fn observations(&self) -> &[Observation] {
        &self.observations
    }

    pub fn failures(&self) -> &[StageFailure] {
        &self.failures
    }

    /// 观察一个信封。没有原始分或来源为空时不产出观测，避免把「无证据」
    /// 当成「证据为 0」。
    pub fn observe<T>(&mut self, envelope: &StageEnvelope<T>, source: Option<&str>) -> bool {
        let Some(raw) = envelope.raw() else {
            return false;
        };
        let source = source
            .map(str::trim)
            .filter(|source| !source.is_empty())
            .map(str::to_string)
            .or_else(|| envelope_source(&envelope.module));
        let Some(source) = source else {
            return false;
        };
        self.observations.push(Observation {
            stage: envelope.stage,
            module: envelope.module.clone(),
            source,
            raw,
            envelope: envelope.envelope,
            witnesses: envelope.witnesses.clone(),
            cost_units: envelope.cost_units,
        });
        true
    }

    pub fn observe_scored(
        &mut self,
        stage: Stage,
        module: impl Into<String>,
        source: impl Into<String>,
        raw: f64,
        envelope: ClaimEnvelope,
    ) -> bool {
        let source = source.into();
        if source.trim().is_empty() {
            return false;
        }
        self.observations.push(Observation {
            stage,
            module: module.into(),
            source: source.trim().to_string(),
            raw: clamp_score(raw),
            envelope,
            witnesses: Vec::new(),
            cost_units: 0,
        });
        true
    }

    pub fn fail(&mut self, failure: StageFailure) -> &mut Self {
        self.failures.push(failure);
        self
    }

    pub fn clear(&mut self) {
        self.observations.clear();
        self.failures.clear();
    }

    /// 把观测交给 PCC。同一来源只保留最强读数（不重复计票），但所有
    /// 逐阶段读数、见证与冲突都保留在回执里。
    pub fn decide(&self, config: &ConfidenceConfig) -> BusDecision {
        let mut strongest: BTreeMap<&str, f64> = BTreeMap::new();
        for observation in &self.observations {
            strongest
                .entry(observation.source.as_str())
                .and_modify(|raw| *raw = raw.max(observation.raw))
                .or_insert(observation.raw);
        }
        let observations: Vec<(&str, f64)> = strongest.into_iter().collect();
        let record = config.decide(self.understanding, &observations, self.near_boundary);
        let readings = self
            .observations
            .iter()
            .map(|observation| StageReading {
                stage: observation.stage,
                module: observation.module.clone(),
                source: observation.source.clone(),
                raw: observation.raw,
                envelope: observation.envelope,
                witnesses: observation.witnesses.clone(),
            })
            .collect();
        let mut conflicts = Vec::new();
        for left in 0..self.observations.len() {
            for right in left + 1..self.observations.len() {
                let first = &self.observations[left];
                let second = &self.observations[right];
                if first.stage != second.stage || first.module == second.module {
                    continue;
                }
                let relation = compare_envelopes(first.envelope, second.envelope);
                if relation == IntervalRelation::Contested {
                    conflicts.push(StageConflict {
                        stage: first.stage,
                        left: first.module.clone(),
                        right: second.module.clone(),
                        relation,
                    });
                }
            }
        }
        let cost_units = self
            .observations
            .iter()
            .map(|observation| observation.cost_units)
            .sum();
        BusDecision {
            record,
            readings,
            conflicts,
            failures: self.failures.clone(),
            cost_units,
        }
    }
}

fn envelope_source(module: &str) -> Option<String> {
    let module = module.trim();
    (!module.is_empty()).then(|| module.to_string())
}

/// 候选区间裁决：严格支配才产生赢家，重叠保留争议。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Arbitration {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub winner: Option<String>,
    pub relation: IntervalRelation,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub contenders: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub envelope: Option<ClaimEnvelope>,
}

/// 对一组具名候选做区间支配裁决。输入顺序不影响输出。
pub fn arbitrate<I>(candidates: I) -> Arbitration
where
    I: IntoIterator<Item = (String, ClaimEnvelope)>,
{
    let mut candidates: Vec<(String, ClaimEnvelope)> = candidates.into_iter().collect();
    if candidates.is_empty() {
        return Arbitration {
            winner: None,
            relation: IntervalRelation::Contested,
            contenders: Vec::new(),
            envelope: None,
        };
    }
    candidates.sort_by(|left, right| {
        right
            .1
            .lower_bps()
            .cmp(&left.1.lower_bps())
            .then_with(|| right.1.upper_bps().cmp(&left.1.upper_bps()))
            .then_with(|| left.0.cmp(&right.0))
    });
    let top = &candidates[0];
    let dominated = candidates[1..]
        .iter()
        .all(|candidate| top.1.lower_bps() > candidate.1.upper_bps());
    if dominated {
        return Arbitration {
            winner: Some(top.0.clone()),
            relation: IntervalRelation::Dominates,
            contenders: vec![top.0.clone()],
            envelope: Some(top.1),
        };
    }
    let contenders: Vec<String> = candidates
        .iter()
        .filter(|candidate| {
            !candidates.iter().any(|other| {
                other.0 != candidate.0 && other.1.lower_bps() > candidate.1.upper_bps()
            })
        })
        .map(|candidate| candidate.0.clone())
        .collect();
    let lower = candidates
        .iter()
        .map(|candidate| candidate.1.lower_bps())
        .min()
        .unwrap_or(0);
    let upper = candidates
        .iter()
        .map(|candidate| candidate.1.upper_bps())
        .max()
        .unwrap_or(0);
    Arbitration {
        winner: None,
        relation: IntervalRelation::Contested,
        contenders,
        envelope: Some(ClaimEnvelope::new(lower, upper)),
    }
}

/// 槽键：`(主语, 谓词)`。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SlotKey {
    pub subject: String,
    pub predicate: String,
}

impl SlotKey {
    pub fn new(subject: impl Into<String>, predicate: impl Into<String>) -> Self {
        Self {
            subject: subject.into().trim().to_string(),
            predicate: predicate.into().trim().to_string(),
        }
    }
}

/// 一次信念观测。`grade` 决定保守起点，`envelope` 是来源给出的区间。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BeliefObservation {
    pub subject: String,
    pub predicate: String,
    pub value: String,
    pub source: String,
    pub grade: EnvelopeGrade,
    pub envelope: ClaimEnvelope,
}

impl BeliefObservation {
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        value: impl Into<String>,
        source: impl Into<String>,
        grade: EnvelopeGrade,
        envelope: ClaimEnvelope,
    ) -> Self {
        Self {
            subject: subject.into().trim().to_string(),
            predicate: predicate.into().trim().to_string(),
            value: value.into().trim().to_string(),
            source: source.into().trim().to_string(),
            grade,
            envelope,
        }
    }

    pub fn from_grade(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        value: impl Into<String>,
        source: impl Into<String>,
        grade: EnvelopeGrade,
    ) -> Self {
        let envelope = ClaimEnvelope::from_grade(grade, 0, 0);
        Self::new(subject, predicate, value, source, grade, envelope)
    }
}

/// 账本中的一条信念。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlotBelief {
    pub value: String,
    pub source: String,
    pub grade: EnvelopeGrade,
    pub envelope: ClaimEnvelope,
    pub first_turn: u64,
    pub last_turn: u64,
    #[serde(default)]
    pub corroborations: u16,
    #[serde(default)]
    pub conflicts: u16,
}

impl SlotBelief {
    /// 按当前回合衰减后的区间；原始信封保持不变。
    pub fn envelope_at(&self, turn: u64) -> ClaimEnvelope {
        decay_envelope(self.envelope, turn.saturating_sub(self.last_turn))
    }

    pub fn age(&self, turn: u64) -> u64 {
        turn.saturating_sub(self.last_turn)
    }
}

/// 当前槽位的裁决视图。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlotResolution {
    pub slot: SlotKey,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub winner: Option<SlotBelief>,
    pub relation: IntervalRelation,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub contenders: Vec<SlotBelief>,
}

impl SlotResolution {
    pub fn is_contested(&self) -> bool {
        self.winner.is_none() && self.contenders.len() > 1
    }

    pub fn render_zh(&self, turn: u64) -> String {
        let slot = format!("{} 的 {}", self.slot.subject, self.slot.predicate);
        match &self.winner {
            Some(winner) => format!(
                "{}：{}（{}，{}）",
                slot,
                winner.value,
                winner.envelope_at(turn).render_interval(),
                winner.source
            ),
            None if self.contenders.is_empty() => format!("{}：没有已登记信念", slot),
            None => {
                let mut out = format!("{}：有争议，未选边", slot);
                for belief in &self.contenders {
                    out.push_str(&format!(
                        "\n  {}（{}，{}）",
                        belief.value,
                        belief.envelope_at(turn).render_interval(),
                        belief.source
                    ));
                }
                out
            }
        }
    }
}

/// 跨轮信念账本：按槽位去重、按回合衰减、容量有界。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BeliefLedger {
    #[serde(default = "default_capacity")]
    capacity: usize,
    #[serde(default)]
    turn: u64,
    #[serde(default, with = "belief_map")]
    beliefs: BTreeMap<SlotKey, Vec<SlotBelief>>,
}

/// JSON 对象的键必须是字符串，而槽键是 `(主语, 谓词)`。这里把映射编码
/// 成有序的 `[槽键, 信念列表]` 数组，既保持 JSON 可读，也保留确定序。
mod belief_map {
    use super::{BeliefLedger, SlotBelief, SlotKey};
    use serde::de::Error as _;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::collections::BTreeMap;

    #[derive(Deserialize)]
    struct Entry {
        key: SlotKey,
        beliefs: Vec<SlotBelief>,
    }

    pub fn serialize<S>(
        map: &BTreeMap<SlotKey, Vec<SlotBelief>>,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        #[derive(Serialize)]
        struct EntryRef<'a> {
            key: &'a SlotKey,
            beliefs: &'a [SlotBelief],
        }
        let entries: Vec<EntryRef<'_>> = map
            .iter()
            .map(|(key, beliefs)| EntryRef { key, beliefs })
            .collect();
        entries.serialize(serializer)
    }

    pub fn deserialize<'de, D>(
        deserializer: D,
    ) -> Result<BTreeMap<SlotKey, Vec<SlotBelief>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let entries = Vec::<Entry>::deserialize(deserializer)?;
        let mut map = BTreeMap::new();
        for entry in entries {
            if entry.key.subject.trim().is_empty() || entry.key.predicate.trim().is_empty() {
                return Err(D::Error::custom("信念槽键不能为空"));
            }
            if map.insert(entry.key, entry.beliefs).is_some() {
                return Err(D::Error::custom("信念槽键重复"));
            }
        }
        Ok(map)
    }

    // 让 rustc 看到 `BeliefLedger` 的导入只用于路径消歧，不参与代码生成。
    #[allow(dead_code)]
    fn _assert_ledger_type(_: &BeliefLedger) {}
}

fn default_capacity() -> usize {
    DEFAULT_BELIEF_CAPACITY
}

impl Default for BeliefLedger {
    fn default() -> Self {
        Self::new(DEFAULT_BELIEF_CAPACITY)
    }
}

impl BeliefLedger {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            turn: 0,
            beliefs: BTreeMap::new(),
        }
    }

    pub fn turn(&self) -> u64 {
        self.turn
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn is_empty(&self) -> bool {
        self.beliefs.is_empty()
    }

    pub fn len(&self) -> usize {
        self.beliefs.values().map(Vec::len).sum()
    }

    pub fn slots(&self) -> impl Iterator<Item = &SlotKey> {
        self.beliefs.keys()
    }

    pub fn slot_beliefs(&self, subject: &str, predicate: &str) -> &[SlotBelief] {
        self.beliefs
            .get(&SlotKey::new(subject, predicate))
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// 记录一次观测。同一取值佐证递增；不同取值互相攻击；容量满时淘汰
    /// 最久未更新的信念（按槽键/取值确定性排序）。
    pub fn observe(&mut self, turn: u64, observation: BeliefObservation) -> bool {
        if self.capacity == 0 {
            return false;
        }
        let observation = normalized(observation);
        if observation.subject.is_empty()
            || observation.predicate.is_empty()
            || observation.value.is_empty()
        {
            return false;
        }
        self.turn = self.turn.max(turn);
        let key = SlotKey::new(&observation.subject, &observation.predicate);
        let beliefs = self.beliefs.entry(key.clone()).or_default();
        if let Some(existing) = beliefs
            .iter_mut()
            .find(|belief| belief.value == observation.value)
        {
            let current = existing.envelope_at(turn);
            existing.envelope = merge_support(current, observation.envelope);
            existing.last_turn = turn;
            existing.corroborations = existing.corroborations.saturating_add(1);
            if observation.grade > existing.grade {
                existing.grade = observation.grade;
            }
            if !observation.source.is_empty() {
                existing.source = observation.source;
            }
        } else {
            let conflicts = beliefs.len() as u16;
            if !beliefs.is_empty() {
                for belief in beliefs.iter_mut() {
                    belief.envelope = apply_attack(belief.envelope_at(turn), 1);
                    belief.conflicts = belief.conflicts.saturating_add(1);
                    belief.last_turn = turn;
                }
            }
            beliefs.push(SlotBelief {
                value: observation.value,
                source: observation.source,
                grade: observation.grade,
                envelope: apply_attack(observation.envelope, conflicts),
                first_turn: turn,
                last_turn: turn,
                corroborations: 0,
                conflicts,
            });
        }
        self.enforce_capacity();
        true
    }

    fn enforce_capacity(&mut self) {
        while self.len() > self.capacity {
            let victim = self
                .beliefs
                .iter()
                .flat_map(|(key, beliefs)| {
                    beliefs
                        .iter()
                        .map(move |belief| (key.clone(), belief.value.clone(), belief.last_turn))
                })
                .min_by(|left, right| {
                    left.2
                        .cmp(&right.2)
                        .then_with(|| left.0.cmp(&right.0))
                        .then_with(|| left.1.cmp(&right.1))
                });
            let Some((key, value, _)) = victim else {
                break;
            };
            if let Some(beliefs) = self.beliefs.get_mut(&key) {
                beliefs.retain(|belief| belief.value != value);
                if beliefs.is_empty() {
                    self.beliefs.remove(&key);
                }
            }
        }
    }

    /// 当前槽位裁决：严格支配才给赢家，否则返回争议候选。
    pub fn resolve(&self, subject: &str, predicate: &str, turn: u64) -> Option<SlotResolution> {
        let key = SlotKey::new(subject, predicate);
        let beliefs = self.beliefs.get(&key)?;
        if beliefs.is_empty() {
            return None;
        }
        let candidates: Vec<(String, ClaimEnvelope)> = beliefs
            .iter()
            .map(|belief| (belief.value.clone(), belief.envelope_at(turn)))
            .collect();
        let arbitration = arbitrate(candidates);
        let winner = arbitration.winner.as_ref().and_then(|winner| {
            beliefs
                .iter()
                .find(|belief| &belief.value == winner)
                .cloned()
        });
        let contenders = if winner.is_some() {
            Vec::new()
        } else {
            let mut contenders: Vec<SlotBelief> = beliefs
                .iter()
                .filter(|belief| {
                    !beliefs.iter().any(|other| {
                        other.value != belief.value
                            && other.envelope_at(turn).lower_bps()
                                > belief.envelope_at(turn).upper_bps()
                    })
                })
                .cloned()
                .collect();
            contenders.sort_by(|left, right| {
                right
                    .envelope_at(turn)
                    .lower_bps()
                    .cmp(&left.envelope_at(turn).lower_bps())
                    .then_with(|| left.value.cmp(&right.value))
            });
            contenders
        };
        Some(SlotResolution {
            slot: key,
            winner,
            relation: arbitration.relation,
            contenders,
        })
    }

    /// 从回答计划吸收带信封的命题；没有信封的命题不进入账本。
    pub fn observe_plan(&mut self, turn: u64, plan: &AnswerPlan) -> usize {
        let mut observed = 0;
        for claim in &plan.claims {
            let Some(envelope) = claim.envelope else {
                continue;
            };
            let grade = if claim.session {
                EnvelopeGrade::SessionDirect
            } else {
                EnvelopeGrade::Graph
            };
            let observation = BeliefObservation::new(
                &plan.subject,
                &claim.predicate,
                &claim.text,
                "answer_plan",
                grade,
                envelope,
            );
            if self.observe(turn, observation) {
                observed += 1;
            }
        }
        observed
    }

    /// 从逐命题复核信封吸收观测；`subject` 由调用方提供计划主体。
    pub fn observe_turn(&mut self, turn: u64, subject: &str, envelope: &TurnEnvelope) -> usize {
        let mut observed = 0;
        for claim in &envelope.claims {
            let grade = if claim.session {
                EnvelopeGrade::SessionDirect
            } else {
                EnvelopeGrade::Graph
            };
            let observation = BeliefObservation::new(
                subject,
                &claim.predicate,
                &claim.text,
                "epistemic_turn",
                grade,
                claim.envelope,
            );
            if self.observe(turn, observation) {
                observed += 1;
            }
        }
        observed
    }

    /// 显式衰减到某个回合（只影响读取视图，不改写原始信封）。
    pub fn advance_to(&mut self, turn: u64) {
        self.turn = self.turn.max(turn);
    }

    pub fn render_report(&self, turn: u64) -> String {
        let mut out = format!(
            "信念账本 {}：{} 个槽位，{} 条信念（回合 {}）。",
            DIALECT,
            self.beliefs.len(),
            self.len(),
            turn
        );
        for (key, beliefs) in &self.beliefs {
            out.push_str(&format!(
                "\n{} 的 {}：",
                key.subject,
                key.predicate
            ));
            let mut sorted = beliefs.clone();
            sorted.sort_by(|left, right| {
                right
                    .envelope_at(turn)
                    .lower_bps()
                    .cmp(&left.envelope_at(turn).lower_bps())
                    .then_with(|| left.value.cmp(&right.value))
            });
            for belief in sorted {
                out.push_str(&format!(
                    "\n  {}（{}，{}，佐证 {}，冲突 {}，{}）",
                    belief.value,
                    belief.envelope_at(turn).render_interval(),
                    belief.source,
                    belief.corroborations,
                    belief.conflicts,
                    belief.grade.label_zh()
                ));
            }
        }
        out
    }
}

fn normalized(mut observation: BeliefObservation) -> BeliefObservation {
    observation.source = observation.source.trim().to_string();
    observation
}

fn merge_support(left: ClaimEnvelope, right: ClaimEnvelope) -> ClaimEnvelope {
    let lower = left.lower_bps().max(right.lower_bps());
    let upper = left.upper_bps().min(right.upper_bps());
    if lower <= upper {
        ClaimEnvelope::new(lower, upper)
    } else {
        // 两个「支持同一取值」的区间却互不相容：说明至少一方高估了自己，
        // 用包络而不是硬选一个端点。
        ClaimEnvelope::new(
            left.lower_bps().min(right.lower_bps()),
            left.upper_bps().max(right.upper_bps()),
        )
    }
}

fn apply_attack(envelope: ClaimEnvelope, rounds: u16) -> ClaimEnvelope {
    let mut lower = u32::from(envelope.lower_bps());
    let mut upper = u32::from(envelope.upper_bps());
    for _ in 0..rounds.min(8) {
        lower = lower * 8_500 / u32::from(ENVELOPE_SCALE);
        upper = upper * 8_500 / u32::from(ENVELOPE_SCALE);
    }
    ClaimEnvelope::new(lower as u16, upper as u16)
}

fn decay_envelope(envelope: ClaimEnvelope, age: u64) -> ClaimEnvelope {
    if age == 0 {
        return envelope;
    }
    let mut lower = u32::from(envelope.lower_bps());
    let mut upper = u32::from(envelope.upper_bps());
    let scale = u32::from(ENVELOPE_SCALE);
    for _ in 0..age.min(64) {
        lower = lower * (scale - u32::from(BELIEF_DECAY_BPS)) / scale;
        upper = upper + (scale - upper) * u32::from(BELIEF_DECAY_BPS) / scale;
    }
    ClaimEnvelope::new(lower as u16, upper as u16)
}

/// 一个候选读法及其后验质量（万分比）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReadingCandidate {
    pub id: String,
    pub label: String,
    pub mass_bps: u16,
}

impl ReadingCandidate {
    pub fn new(id: impl Into<String>, label: impl Into<String>, mass_bps: u16) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            mass_bps: mass_bps.min(ENVELOPE_SCALE),
        }
    }

    pub fn from_soft_candidate(candidate: &crate::soft_route::SoftCandidate) -> Self {
        Self::new(
            candidate.entry.id(),
            format!("{} → {}", candidate.name(), candidate.query_type()),
            to_bps(candidate.confidence),
        )
    }
}

/// 一个可提出的澄清问题：`yes` 是回答「是」时仍然成立的候选 id 集合。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClarificationQuestion {
    pub id: String,
    pub text: String,
    pub yes: BTreeSet<String>,
    #[serde(default)]
    pub cost_units: u16,
}

impl ClarificationQuestion {
    pub fn new<I, S>(id: impl Into<String>, text: impl Into<String>, yes: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            id: id.into(),
            text: text.into(),
            yes: yes.into_iter().map(Into::into).collect(),
            cost_units: 0,
        }
    }

    pub fn with_cost(mut self, cost_units: u16) -> Self {
        self.cost_units = cost_units;
        self
    }
}

/// 澄清规划结果：问题、预期收益与预期剩余质量。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClarificationChoice {
    pub question_id: String,
    pub text: String,
    pub prior_max_bps: u16,
    pub expected_max_bps: u16,
    pub gain_bps: u16,
    pub cost_units: u16,
}

impl ClarificationChoice {
    pub fn render_zh(&self) -> String {
        format!(
            "建议澄清「{}」：预期最大后验从 {} 降到 {}，收益 {} 个万分点，提问成本 {}。",
            self.text,
            self.prior_max_bps,
            self.expected_max_bps,
            self.gain_bps,
            self.cost_units
        )
    }
}

/// 选一个最能劈开候选集的问题：最大化「最大后验质量」的预期下降；
/// 同收益时优先成本更低的问题，再按 id 稳定排序。
pub fn plan_clarification(
    candidates: &[ReadingCandidate],
    questions: &[ClarificationQuestion],
) -> Option<ClarificationChoice> {
    let candidates: Vec<&ReadingCandidate> = candidates
        .iter()
        .filter(|candidate| candidate.mass_bps > 0)
        .collect();
    if candidates.len() < 2 {
        return None;
    }
    let total: u32 = candidates.iter().map(|c| u32::from(c.mass_bps)).sum();
    if total == 0 {
        return None;
    }
    let prior_max = candidates.iter().map(|c| c.mass_bps).max().unwrap_or(0);
    let mut best: Option<ClarificationChoice> = None;
    for question in questions {
        let yes_mass: u32 = candidates
            .iter()
            .filter(|candidate| question.yes.contains(&candidate.id))
            .map(|candidate| u32::from(candidate.mass_bps))
            .sum();
        if yes_mass == 0 || yes_mass == total {
            continue;
        }
        let no_mass = total - yes_mass;
        let yes_max = candidates
            .iter()
            .filter(|candidate| question.yes.contains(&candidate.id))
            .map(|candidate| candidate.mass_bps)
            .max()
            .unwrap_or(0);
        let no_max = candidates
            .iter()
            .filter(|candidate| !question.yes.contains(&candidate.id))
            .map(|candidate| candidate.mass_bps)
            .max()
            .unwrap_or(0);
        let expected = (yes_mass * u32::from(yes_max) + no_mass * u32::from(no_max)) / total;
        let expected = expected.min(u32::from(ENVELOPE_SCALE)) as u16;
        let gain = prior_max.saturating_sub(expected);
        if gain == 0 {
            continue;
        }
        let choice = ClarificationChoice {
            question_id: question.id.clone(),
            text: question.text.clone(),
            prior_max_bps: prior_max,
            expected_max_bps: expected,
            gain_bps: gain,
            cost_units: question.cost_units,
        };
        let better = match &best {
            None => true,
            Some(current) => {
                choice.gain_bps > current.gain_bps
                    || (choice.gain_bps == current.gain_bps
                        && (choice.cost_units < current.cost_units
                            || (choice.cost_units == current.cost_units
                                && choice.question_id < current.question_id)))
            }
        };
        if better {
            best = Some(choice);
        }
    }
    best
}

/// 把软路由结果装进阶段信封：值保留完整结果，原始分取当前首选的后验。
pub fn soft_route_envelope(outcome: &SoftRouteOutcome) -> StageEnvelope<SoftRouteOutcome> {
    let candidate = outcome
        .selected
        .as_ref()
        .or_else(|| outcome.alternatives.first());
    let mut envelope = match candidate {
        Some(candidate) => StageEnvelope::from_point(
            Stage::Intent,
            "soft_route",
            outcome.clone(),
            candidate.confidence,
        ),
        None => StageEnvelope::new(
            Stage::Intent,
            "soft_route",
            outcome.clone(),
            ClaimEnvelope::new(0, 5_000),
        ),
    }
    .with_witness(Witness::new("soft_route", outcome.reason.clone()))
    .with_cost(2);
    let residual = match outcome.decision {
        SoftDecision::Accept => None,
        SoftDecision::Clarify => {
            let labels: Vec<String> = outcome
                .clarification_options()
                .iter()
                .map(|candidate| candidate.query_type().to_string())
                .collect();
            Some(format!("仍需澄清：{}", labels.join("、")))
        }
        SoftDecision::Reject => Some("软路由没有可接受的解释".to_string()),
    };
    if let Some(residual) = residual {
        envelope = envelope.with_residual(residual);
    }
    envelope
}

/// 把插件改写结果装进阶段信封。
///
/// 插件没有历史准确率，因此这里不伪造原始分：只保留「命中了哪些插件、
/// 生效文本是什么」。调用方若要接入 PCC，应显式提供一个已校准来源。
pub fn plugin_envelope(outcome: &PluginOutcome) -> StageEnvelope<PluginOutcome> {
    let mut envelope = StageEnvelope::new(
        Stage::Rewrite,
        "plugin",
        outcome.clone(),
        ClaimEnvelope::from_grade(EnvelopeGrade::Inference, 0, 0),
    )
    .with_cost(
        outcome
            .trace
            .matches
            .iter()
            .filter(|entry| entry.applied)
            .count()
            .min(u16::MAX as usize) as u16,
    );
    for entry in outcome.trace.matches.iter().filter(|entry| entry.applied) {
        envelope = envelope.with_witness(Witness::new(
            &entry.plugin,
            format!("{} {}", entry.kind, entry.contribution),
        ));
    }
    let residual = if outcome.trace.notes.is_empty() {
        None
    } else {
        Some(outcome.trace.notes.join("；"))
    };
    if let Some(residual) = residual {
        envelope = envelope.with_residual(residual);
    }
    envelope
}

fn clamp_score(score: f64) -> f64 {
    if score.is_finite() {
        score.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn to_bps(probability: f64) -> u16 {
    (clamp_score(probability) * f64::from(ENVELOPE_SCALE)).round() as u16
}

fn truncate_chars(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_string();
    }
    let mut out: String = text.chars().take(limit).collect();
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::confidence::{ConfidenceSourceConfig, EnvelopeGrade};

    fn test_config() -> ConfidenceConfig {
        ConfidenceConfig {
            enabled: true,
            sources: vec![
                ConfidenceSourceConfig {
                    name: "intent".to_string(),
                    gamma: 0.5,
                    calibration: Vec::new(),
                },
                ConfidenceSourceConfig {
                    name: "evidence".to_string(),
                    gamma: 0.4,
                    calibration: Vec::new(),
                },
            ],
            ..ConfidenceConfig::default()
        }
    }

    #[test]
    fn module_catalog_registers_and_reports_missing_dependencies() {
        let mut catalog = ModuleCatalog::new();
        let card = ModuleCard::new("soft_route", Stage::Intent)
            .with_provides(["intent"])
            .with_requires(["pattern_library"])
            .with_source("intent")
            .with_cost(2);
        assert!(catalog.register(card).is_empty());
        assert_eq!(catalog.for_stage(Stage::Intent).len(), 1);
        assert_eq!(
            catalog.missing_dependencies(),
            vec![("soft_route".to_string(), "pattern_library".to_string())]
        );
        let duplicate = ModuleCard::new("soft_route", Stage::Intent);
        assert!(catalog.register(duplicate).contains(&"module_duplicate_id".to_string()));
    }

    #[test]
    fn interval_comparison_is_strict_and_preserves_contested() {
        let low = ClaimEnvelope::new(4_000, 5_000);
        let high = ClaimEnvelope::new(6_000, 7_000);
        assert_eq!(
            compare_envelopes(high, low),
            IntervalRelation::Dominates
        );
        assert_eq!(
            compare_envelopes(low, high),
            IntervalRelation::Dominated
        );
        assert_eq!(
            compare_envelopes(ClaimEnvelope::new(5_000, 7_000), high),
            IntervalRelation::Contested
        );
        assert_eq!(compare_envelopes(high, high), IntervalRelation::Equal);
    }

    #[test]
    fn arbitration_only_returns_a_winner_on_dominance() {
        let result = arbitrate(vec![
            ("a".to_string(), ClaimEnvelope::new(8_000, 9_000)),
            ("b".to_string(), ClaimEnvelope::new(4_000, 6_000)),
        ]);
        assert_eq!(result.winner.as_deref(), Some("a"));
        assert_eq!(result.relation, IntervalRelation::Dominates);

        let contested = arbitrate(vec![
            ("a".to_string(), ClaimEnvelope::new(5_000, 8_000)),
            ("b".to_string(), ClaimEnvelope::new(6_000, 9_000)),
        ]);
        assert!(contested.winner.is_none());
        assert_eq!(contested.relation, IntervalRelation::Contested);
        assert_eq!(
            contested.envelope,
            Some(ClaimEnvelope::new(5_000, 9_000))
        );
        assert_eq!(contested.contenders.len(), 2);
    }

    #[test]
    fn bus_fuses_by_source_and_keeps_stage_readings() {
        let mut bus = EvidenceBus::new(0.9);
        assert!(bus.observe_scored(
            Stage::Intent,
            "soft_route",
            "intent",
            0.8,
            ClaimEnvelope::new(7_500, 8_500),
        ));
        assert!(bus.observe_scored(
            Stage::Intent,
            "soft_route",
            "intent",
            0.9,
            ClaimEnvelope::new(8_000, 9_000),
        ));
        assert!(bus.observe_scored(
            Stage::Reason,
            "problog",
            "evidence",
            0.85,
            ClaimEnvelope::new(8_000, 9_000),
        ));
        assert!(!bus.observe_scored(
            Stage::Rewrite,
            "plugin",
            "",
            0.7,
            ClaimEnvelope::point(0.7),
        ));
        let decision = bus.decide(&test_config());
        assert_eq!(decision.readings.len(), 3);
        assert_eq!(decision.record.confidence, decision.record.fused.probability);
        assert_eq!(
            decision
                .record
                .fused
                .contributions
                .iter()
                .filter(|contribution| contribution.raw.is_some())
                .count(),
            2
        );
    }

    #[test]
    fn bus_records_failures_without_turning_them_into_zero_evidence() {
        let mut bus = EvidenceBus::new(0.9);
        bus.fail(StageFailure::new(
            Stage::Retrieve,
            "sparql",
            "存储读取失败",
        ));
        let decision = bus.decide(&test_config());
        assert_eq!(decision.failures.len(), 1);
        assert!(decision
            .failure_note()
            .as_deref()
            .unwrap()
            .contains("存储读取失败"));
    }

    #[test]
    fn belief_ledger_corroborates_conflicts_and_decays() {
        let mut ledger = BeliefLedger::new(8);
        let direct = ClaimEnvelope::from_grade(EnvelopeGrade::SessionDirect, 0, 0);
        assert!(ledger.observe(
            1,
            BeliefObservation::new("小明", "年龄", "18", "用户", EnvelopeGrade::SessionDirect, direct),
        ));
        assert!(ledger.observe(
            2,
            BeliefObservation::new("小明", "年龄", "18", "用户", EnvelopeGrade::SessionDirect, direct),
        ));
        let beliefs = ledger.slot_beliefs("小明", "年龄");
        assert_eq!(beliefs.len(), 1);
        assert_eq!(beliefs[0].corroborations, 1);
        assert_eq!(ledger.resolve("小明", "年龄", 2).unwrap().winner.unwrap().value, "18");

        assert!(ledger.observe(
            3,
            BeliefObservation::new("小明", "年龄", "19", "更正", EnvelopeGrade::SessionCorrection, ClaimEnvelope::new(8_000, 9_000)),
        ));
        let resolution = ledger.resolve("小明", "年龄", 3).unwrap();
        assert!(resolution.winner.is_none(), "{:?}", resolution);
        assert_eq!(resolution.contenders.len(), 2);

        let before = ledger.slot_beliefs("小明", "年龄")[0].envelope_at(3);
        let after = ledger.slot_beliefs("小明", "年龄")[0].envelope_at(63);
        assert!(after.lower_bps() < before.lower_bps());
        assert!(after.upper_bps() > before.upper_bps());
    }

    #[test]
    fn belief_ledger_is_bounded_and_drops_oldest() {
        let mut ledger = BeliefLedger::new(2);
        for (turn, subject) in [(1, "甲"), (2, "乙"), (3, "丙")] {
            ledger.observe(
                turn,
                BeliefObservation::from_grade(
                    subject,
                    "属性",
                    "值",
                    "测试",
                    EnvelopeGrade::Graph,
                ),
            );
        }
        assert_eq!(ledger.len(), 2);
        assert!(ledger.slot_beliefs("甲", "属性").is_empty());
        assert!(!ledger.slot_beliefs("丙", "属性").is_empty());
    }

    #[test]
    fn clarification_planner_prefers_the_question_that_splits_mass() {
        let candidates = vec![
            ReadingCandidate::new("a", "读法 A", 4_000),
            ReadingCandidate::new("b", "读法 B", 4_000),
            ReadingCandidate::new("c", "读法 C", 2_000),
        ];
        let weak = ClarificationQuestion::new("q1", "是 A 吗？", ["a"]);
        let strong = ClarificationQuestion::new("q2", "是 A 或 B 吗？", ["a", "b"]);
        let choice = plan_clarification(&candidates, &[weak, strong]).unwrap();
        assert_eq!(choice.question_id, "q2");
        assert!(choice.gain_bps > 0);
        assert_eq!(choice.prior_max_bps, 4_000);
    }

    #[test]
    fn clarification_planner_rejects_questions_that_split_nothing() {
        let candidates = vec![
            ReadingCandidate::new("a", "读法 A", 5_000),
            ReadingCandidate::new("b", "读法 B", 5_000),
        ];
        assert!(plan_clarification(
            &candidates,
            &[ClarificationQuestion::new("all", "是 A 或 B 吗？", ["a", "b"])]
        )
        .is_none());
        assert!(plan_clarification(&candidates[..1], &[]).is_none());
    }

    #[test]
    fn belief_ledger_observes_plans_without_envelopes() {
        let plan = AnswerPlan::new("介绍：小明", "小明", 1, Vec::new(), 0, 0);
        let mut ledger = BeliefLedger::default();
        assert_eq!(ledger.observe_plan(1, &plan), 0);
        assert!(ledger.is_empty());
    }

    #[test]
    fn soft_route_envelope_keeps_a_residual_for_clarification() {
        let outcome = SoftRouteOutcome {
            dialect: "test".to_string(),
            decision: SoftDecision::Clarify,
            selected: None,
            alternatives: Vec::new(),
            reason: "候选接近".to_string(),
        };
        let envelope = soft_route_envelope(&outcome);
        assert_eq!(envelope.stage, Stage::Intent);
        assert_eq!(envelope.raw(), None);
        assert!(envelope.residual.as_deref().unwrap().contains("仍需澄清"));
    }

    #[test]
    fn witness_list_is_bounded_and_deduplicated() {
        let mut envelope = StageEnvelope::from_point(
            Stage::Retrieve,
            "sparql",
            "value",
            0.8,
        );
        for _ in 0..(MAX_WITNESSES + 4) {
            envelope = envelope.with_witness(Witness::new("sparql", "同一见证"));
        }
        envelope = envelope.with_witness(Witness::new("other", "另一见证"));
        assert_eq!(envelope.witnesses.len(), 2);
    }

    #[test]
    fn catalog_and_ledger_roundtrip_through_json() {
        let mut catalog = ModuleCatalog::new();
        catalog.register(ModuleCard::new("soft_route", Stage::Intent));
        let encoded = serde_json::to_string(&catalog).unwrap();
        let decoded: ModuleCatalog = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, catalog);

        let mut ledger = BeliefLedger::new(4);
        ledger.observe(
            1,
            BeliefObservation::from_grade("小明", "年龄", "18", "用户", EnvelopeGrade::SessionDirect),
        );
        let encoded = serde_json::to_string(&ledger).unwrap();
        let decoded: BeliefLedger = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, ledger);
    }
}

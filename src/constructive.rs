//! 建设性多跳推理：在知识缺口处合成有证明的虚拟事实，并让它们参与后续跳。
//!
//! 与既有的链接预测、溯因和补全回合不同，这里的单位是**查询局部的虚拟事实**：
//!
//! ```text
//! 观察事实/会话事实 → 规则合成候选 → 硬约束与预算裁剪
//!   → 虚拟事实（Hypothesis）→ 作为下一跳前提继续合成
//!   → 生成带证明的回答 → 用户确认/否认 → 会话事实/阻断边
//! ```
//!
//! 关键不变量：
//!
//! - **无证明不事实**：每条虚拟事实都带支持边；支持图必须最终落在观察事实上，
//!   且不能出现自支持环。
//! - **无确认不晋升**：查询作用域的推断随查询结束丢弃；只有显式确认才进入
//!   会话作用域，且永不写基础图谱。
//! - **无预算不搜索**：跳深、往返、候选数、虚拟事实总数与展开次数都有上限；
//!   触发上限时报告显式标记截断，绝不假装完整。
//! - **无失效不记忆**：知识版本变化或前提被否认时，依赖图级联失效。
//!
//! 本模块默认不接入回答路径；只有配置显式启用后才在链式查询的缺口处使用。

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::link_prediction::{HornRule, RuleTemplate, Triple};

pub const DEFAULT_MAX_DEPTH: usize = 4;
pub const DEFAULT_MAX_VIRTUAL_FACTS: usize = 96;
pub const DEFAULT_MAX_CANDIDATES_PER_ROUND: usize = 32;
pub const DEFAULT_MAX_EXPANSIONS: usize = 8_000;
pub const DEFAULT_BEAM_WIDTH: usize = 32;
pub const DEFAULT_MIN_CONFIDENCE_BPS: u16 = 3_500;
pub const DEFAULT_SESSION_CAPACITY: usize = 64;
pub const DEFAULT_PROMOTION_THRESHOLD_BPS: u16 = 8_000;
pub const VIRTUAL_DAMPING_BPS: u16 = 9_000;
pub const VIRTUAL_FACT_PENALTY_BPS: u16 = 9_500;
pub const OBSERVED_CONFIDENCE_BPS: u16 = 10_000;
const MAX_VIRTUAL_CONFIDENCE_BPS: u16 = 9_999;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ConstructionConfig {
    pub enabled: bool,
    pub max_depth: usize,
    pub max_virtual_facts: usize,
    pub max_candidates_per_round: usize,
    pub max_expansions: usize,
    pub beam_width: usize,
    pub min_confidence_bps: u16,
    pub session_capacity: usize,
    pub promotion_threshold_bps: u16,
}

impl Default for ConstructionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_depth: DEFAULT_MAX_DEPTH,
            max_virtual_facts: DEFAULT_MAX_VIRTUAL_FACTS,
            max_candidates_per_round: DEFAULT_MAX_CANDIDATES_PER_ROUND,
            max_expansions: DEFAULT_MAX_EXPANSIONS,
            beam_width: DEFAULT_BEAM_WIDTH,
            min_confidence_bps: DEFAULT_MIN_CONFIDENCE_BPS,
            session_capacity: DEFAULT_SESSION_CAPACITY,
            promotion_threshold_bps: DEFAULT_PROMOTION_THRESHOLD_BPS,
        }
    }
}

impl ConstructionConfig {
    fn normalized(mut self) -> Self {
        self.max_depth = self.max_depth.clamp(1, 16);
        self.max_virtual_facts = self.max_virtual_facts.clamp(1, 4_096);
        self.max_candidates_per_round = self.max_candidates_per_round.clamp(1, 1_024);
        self.max_expansions = self.max_expansions.clamp(1, 1_000_000);
        self.beam_width = self.beam_width.clamp(1, self.max_candidates_per_round);
        self.session_capacity = self.session_capacity.clamp(1, 4_096);
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstructionOperator {
    Inverse,
    Compose,
    CommonNeighbor,
    CommonObject,
}

impl ConstructionOperator {
    pub fn label(self) -> &'static str {
        match self {
            ConstructionOperator::Inverse => "逆向",
            ConstructionOperator::Compose => "组合",
            ConstructionOperator::CommonNeighbor => "共同邻居",
            ConstructionOperator::CommonObject => "共同主体",
        }
    }

    fn from_template(template: RuleTemplate) -> Option<Self> {
        match template {
            RuleTemplate::Inverse => Some(ConstructionOperator::Inverse),
            RuleTemplate::Compose => Some(ConstructionOperator::Compose),
            RuleTemplate::CommonNeighbor => Some(ConstructionOperator::CommonNeighbor),
            RuleTemplate::CommonObject => Some(ConstructionOperator::CommonObject),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VirtualState {
    Hypothesis,
    Confirmed,
    Refuted,
    Invalidated,
}

impl VirtualState {
    pub fn label(self) -> &'static str {
        match self {
            VirtualState::Hypothesis => "推断（待确认）",
            VirtualState::Confirmed => "已确认",
            VirtualState::Refuted => "已否认",
            VirtualState::Invalidated => "已失效",
        }
    }

    pub fn asserts(self) -> bool {
        matches!(self, VirtualState::Confirmed)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FactScope {
    Query,
    Session,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct FactKey {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

impl FactKey {
    pub fn new(subject: impl Into<String>, predicate: impl Into<String>, object: impl Into<String>) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
        }
    }

    pub fn render(&self) -> String {
        format!("{} --{}--> {}", self.subject, self.predicate, self.object)
    }

    pub fn canonical(&self) -> String {
        format!("{}({}, {})", self.predicate, self.subject, self.object)
    }
}

impl fmt::Display for FactKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.render())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VirtualFact {
    pub key: FactKey,
    pub state: VirtualState,
    pub scope: FactScope,
    pub confidence_bps: u16,
    pub depth: usize,
    pub operator: ConstructionOperator,
    pub rule: String,
    pub support: Vec<FactKey>,
    pub revision: u64,
    #[serde(default)]
    pub touched: u64,
}

impl VirtualFact {
    pub fn confidence(&self) -> f64 {
        f64::from(self.confidence_bps) / 10_000.0
    }

    pub fn is_asserted(&self) -> bool {
        self.state.asserts()
    }

    pub fn render(&self) -> String {
        format!(
            "{}（{}，置信度 {:.2}，{} 跳，{}；前提：{}）",
            self.key.render(),
            self.state.label(),
            self.confidence(),
            self.depth,
            self.operator.label(),
            if self.support.is_empty() {
                "无".to_string()
            } else {
                self.support
                    .iter()
                    .map(FactKey::render)
                    .collect::<Vec<_>>()
                    .join("；")
            }
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstructionRule {
    pub id: String,
    pub operator: ConstructionOperator,
    pub body: Vec<String>,
    pub head: String,
    pub confidence_bps: u16,
}

impl ConstructionRule {
    pub fn from_horn(rule: &HornRule, index: usize) -> Option<Self> {
        let operator = ConstructionOperator::from_template(rule.template)?;
        if rule.body.is_empty() || rule.head.is_empty() {
            return None;
        }
        let confidence_bps = quantize(rule.confidence);
        if confidence_bps == 0 {
            return None;
        }
        Some(Self {
            id: format!("规则#{}[{}]", index, rule.rule_text()),
            operator,
            body: rule.body.clone(),
            head: rule.head.clone(),
            confidence_bps,
        })
    }

    pub fn render(&self) -> String {
        match self.operator {
            ConstructionOperator::Inverse => format!(
                "{}(A,B) ⇒ {}(B,A)",
                self.body.first().map(String::as_str).unwrap_or(""),
                self.head
            ),
            ConstructionOperator::Compose => format!(
                "{}(A,B) ∧ {}(B,C) ⇒ {}(A,C)",
                self.body.first().map(String::as_str).unwrap_or(""),
                self.body.get(1).map(String::as_str).unwrap_or(""),
                self.head
            ),
            ConstructionOperator::CommonNeighbor => format!(
                "{}(A,C) ∧ {}(B,C) ⇒ {}(A,B)",
                self.body.first().map(String::as_str).unwrap_or(""),
                self.body.get(1).map(String::as_str).unwrap_or(""),
                self.head
            ),
            ConstructionOperator::CommonObject => format!(
                "{}(C,A) ∧ {}(C,B) ⇒ {}(A,B)",
                self.body.first().map(String::as_str).unwrap_or(""),
                self.body.get(1).map(String::as_str).unwrap_or(""),
                self.head
            ),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ObservedGraph {
    out: BTreeMap<(String, String), BTreeSet<String>>,
    incoming: BTreeMap<(String, String), BTreeSet<String>>,
    incident: BTreeMap<String, BTreeSet<FactKey>>,
    count: usize,
}

impl ObservedGraph {
    pub fn from_triples(triples: &[Triple]) -> Self {
        let mut graph = Self::default();
        for triple in triples {
            graph.insert(&triple.subject, &triple.predicate, &triple.object);
        }
        graph
    }

    pub fn insert(&mut self, subject: &str, predicate: &str, object: &str) {
        let key = FactKey::new(subject, predicate, object);
        if self
            .incident
            .get(subject)
            .is_some_and(|facts| facts.contains(&key))
        {
            return;
        }
        self.out
            .entry((subject.to_string(), predicate.to_string()))
            .or_default()
            .insert(object.to_string());
        self.incoming
            .entry((predicate.to_string(), object.to_string()))
            .or_default()
            .insert(subject.to_string());
        self.incident
            .entry(subject.to_string())
            .or_default()
            .insert(key.clone());
        self.incident
            .entry(object.to_string())
            .or_default()
            .insert(key);
        self.count += 1;
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn contains(&self, key: &FactKey) -> bool {
        self.incident
            .get(&key.subject)
            .is_some_and(|facts| facts.contains(key))
    }

    pub fn objects(&self, subject: &str, predicate: &str) -> Vec<String> {
        self.out
            .get(&(subject.to_string(), predicate.to_string()))
            .map(|objects| objects.iter().cloned().collect())
            .unwrap_or_default()
    }

    pub fn subjects(&self, predicate: &str, object: &str) -> Vec<String> {
        self.incoming
            .get(&(predicate.to_string(), object.to_string()))
            .map(|subjects| subjects.iter().cloned().collect())
            .unwrap_or_default()
    }

    pub fn incident(&self, entity: &str) -> Vec<FactKey> {
        self.incident
            .get(entity)
            .map(|facts| facts.iter().cloned().collect())
            .unwrap_or_default()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstructionStatus {
    Disabled,
    AlreadyObserved,
    Constructed,
    Exhausted,
    Truncated,
}

impl ConstructionStatus {
    pub fn label(self) -> &'static str {
        match self {
            ConstructionStatus::Disabled => "未启用",
            ConstructionStatus::AlreadyObserved => "已有事实",
            ConstructionStatus::Constructed => "已合成",
            ConstructionStatus::Exhausted => "未找到可合成路径",
            ConstructionStatus::Truncated => "已截断",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConstructionReport {
    pub seed: String,
    pub target_predicate: String,
    pub inverted: bool,
    pub status: ConstructionStatus,
    pub target_facts: Vec<VirtualFact>,
    pub facts: Vec<VirtualFact>,
    pub truncated: bool,
    pub truncation_reasons: Vec<&'static str>,
    pub expansions: usize,
    pub virtual_total: usize,
    pub revision: u64,
}

impl ConstructionReport {
    pub fn is_empty(&self) -> bool {
        self.target_facts.is_empty()
    }

    pub fn render_text(&self) -> String {
        let direction = if self.inverted { "逆向" } else { "正向" };
        let mut output = format!(
            "建设性多跳（起点 {}，{}目标 {}）：{}；合成 {} 条，目标 {} 条。\n",
            self.seed,
            direction,
            self.target_predicate,
            self.status.label(),
            self.facts.len(),
            self.target_facts.len()
        );
        for fact in self.target_facts.iter().take(8) {
            output.push_str(&format!("  • {}\n", fact.render()));
        }
        if self.target_facts.len() > 8 {
            output.push_str(&format!(
                "  • ……另有 {} 条目标推断未展开。\n",
                self.target_facts.len() - 8
            ));
        }
        if self.truncated {
            output.push_str(&format!(
                "  说明：搜索被预算截断（{}）；未展开的分支不能视为不存在。\n",
                if self.truncation_reasons.is_empty() {
                    "达到上限".to_string()
                } else {
                    self.truncation_reasons.join("、")
                }
            ));
        }
        output.push_str(&format!(
            "  作用域：查询局部（知识版本 {}，展开 {} 次，虚拟事实共 {} 条）；未写基础图谱。",
            self.revision, self.expansions, self.virtual_total
        ));
        output
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SupportAudit {
    pub inspected: usize,
    pub invalid: Vec<FactKey>,
    pub cyclic: Vec<FactKey>,
}

impl SupportAudit {
    pub fn is_clean(&self) -> bool {
        self.invalid.is_empty() && self.cyclic.is_empty()
    }
}

#[derive(Debug, Clone)]
struct FactRef {
    key: FactKey,
    confidence_bps: u16,
    depth: usize,
}

#[derive(Debug, Clone)]
struct Candidate {
    key: FactKey,
    confidence_bps: u16,
    depth: usize,
    operator: ConstructionOperator,
    rule: String,
    support: Vec<FactKey>,
    order: usize,
}

#[derive(Debug, Clone)]
pub struct ConstructiveReasoner {
    config: ConstructionConfig,
    revision: u64,
    observed: ObservedGraph,
    rules: Vec<ConstructionRule>,
    virtuals: BTreeMap<FactKey, VirtualFact>,
    refuted: BTreeSet<FactKey>,
    dependents: BTreeMap<FactKey, BTreeSet<FactKey>>,
    active: Vec<FactKey>,
    loaded_entities: BTreeSet<String>,
    queued: HashSet<FactKey>,
    tick: u64,
    expansions: usize,
    truncated: bool,
    truncation_reasons: Vec<&'static str>,
}

impl ConstructiveReasoner {
    pub fn new(
        config: ConstructionConfig,
        revision: u64,
        observed: ObservedGraph,
        rules: Vec<ConstructionRule>,
    ) -> Self {
        Self {
            config: config.normalized(),
            revision,
            observed,
            rules,
            virtuals: BTreeMap::new(),
            refuted: BTreeSet::new(),
            dependents: BTreeMap::new(),
            active: Vec::new(),
            loaded_entities: BTreeSet::new(),
            queued: HashSet::new(),
            tick: 0,
            expansions: 0,
            truncated: false,
            truncation_reasons: Vec::new(),
        }
    }

    pub fn from_predictor(
        config: ConstructionConfig,
        revision: u64,
        triples: &[Triple],
        rules: &[HornRule],
    ) -> Self {
        let rules = rules
            .iter()
            .enumerate()
            .filter_map(|(index, rule)| ConstructionRule::from_horn(rule, index))
            .collect();
        Self::new(config, revision, ObservedGraph::from_triples(triples), rules)
    }

    pub fn enabled(&self) -> bool {
        self.config.enabled
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn virtual_facts(&self) -> impl Iterator<Item = &VirtualFact> {
        self.virtuals.values()
    }

    pub fn observed(&self) -> &ObservedGraph {
        &self.observed
    }

    pub fn rules(&self) -> &[ConstructionRule] {
        &self.rules
    }

    /// 为 `seed` 的 `predicate` 目标做有界前向饱和。
    ///
    /// `inverted = true` 时目标形如 `?X predicate seed`；返回的
    /// [`VirtualFact`] 始终保持原始三元组方向，调用方按需取另一端。
    pub fn derive(
        &mut self,
        seed: &str,
        predicate: &str,
        inverted: bool,
    ) -> ConstructionReport {
        if !self.config.enabled {
            return self.report(seed, predicate, inverted, ConstructionStatus::Disabled, 0);
        }
        let direct = if inverted {
            !self.observed.subjects(predicate, seed).is_empty()
        } else {
            !self.observed.objects(seed, predicate).is_empty()
        };
        if direct {
            return self.report(
                seed,
                predicate,
                inverted,
                ConstructionStatus::AlreadyObserved,
                0,
            );
        }

        self.seed_entity(seed);
        self.saturate();

        let mut target_facts: Vec<VirtualFact> = self
            .virtuals
            .values()
            .filter(|fact| fact.key.predicate == predicate)
            .filter(|fact| {
                if inverted {
                    fact.key.object == seed
                } else {
                    fact.key.subject == seed
                }
            })
            .filter(|fact| fact.state != VirtualState::Invalidated && fact.state != VirtualState::Refuted)
            .cloned()
            .collect();
        target_facts.sort_by(|left, right| {
            right
                .confidence_bps
                .cmp(&left.confidence_bps)
                .then_with(|| left.depth.cmp(&right.depth))
                .then_with(|| left.key.cmp(&right.key))
        });
        let status = if target_facts.is_empty() {
            if self.truncated {
                ConstructionStatus::Truncated
            } else {
                ConstructionStatus::Exhausted
            }
        } else if self.truncated {
            ConstructionStatus::Truncated
        } else {
            ConstructionStatus::Constructed
        };
        let facts = self.report_facts(seed, &target_facts);
        let mut report = self.report(seed, predicate, inverted, status, facts.len());
        report.target_facts = target_facts;
        report.facts = facts;
        report.truncated = self.truncated;
        report.truncation_reasons = self.truncation_reasons.clone();
        report
    }

    fn report(
        &self,
        seed: &str,
        predicate: &str,
        inverted: bool,
        status: ConstructionStatus,
        fact_count: usize,
    ) -> ConstructionReport {
        ConstructionReport {
            seed: seed.to_string(),
            target_predicate: predicate.to_string(),
            inverted,
            status,
            target_facts: Vec::new(),
            facts: Vec::with_capacity(fact_count),
            truncated: self.truncated,
            truncation_reasons: self.truncation_reasons.clone(),
            expansions: self.expansions,
            virtual_total: self.virtuals.len(),
            revision: self.revision,
        }
    }

    fn report_facts(&self, seed: &str, targets: &[VirtualFact]) -> Vec<VirtualFact> {
        let mut keys: BTreeSet<FactKey> = BTreeSet::new();
        for target in targets {
            keys.insert(target.key.clone());
            for key in self.support_closure(&target.key) {
                keys.insert(key);
            }
        }
        let _ = seed;
        keys.into_iter()
            .filter_map(|key| self.virtuals.get(&key).cloned())
            .collect()
    }

    fn mark_truncated(&mut self, reason: &'static str) {
        self.truncated = true;
        if !self.truncation_reasons.contains(&reason) {
            self.truncation_reasons.push(reason);
        }
    }

    fn seed_entity(&mut self, entity: &str) {
        self.active.clear();
        self.queued.clear();
        self.loaded_entities.insert(entity.to_string());
        for key in self.observed.incident(entity) {
            self.enqueue(key);
        }
        let virtuals: Vec<FactKey> = self.virtuals.keys().cloned().collect();
        for key in virtuals {
            if self
                .virtuals
                .get(&key)
                .is_some_and(|fact| matches!(fact.state, VirtualState::Hypothesis | VirtualState::Confirmed))
            {
                self.enqueue(key);
            }
        }
    }

    fn load_entity(&mut self, entity: &str) {
        if !self.loaded_entities.insert(entity.to_string()) {
            return;
        }
        for key in self.observed.incident(entity) {
            self.enqueue(key);
        }
    }

    fn enqueue(&mut self, key: FactKey) {
        if self.queued.insert(key.clone()) {
            self.active.push(key);
        }
    }

    fn enqueue_without_seed(&mut self, key: FactKey) {
        self.enqueue(key);
    }

    fn fact_ref(&self, key: &FactKey) -> Option<FactRef> {
        if let Some(fact) = self.virtuals.get(key) {
            if fact.state == VirtualState::Hypothesis || fact.state == VirtualState::Confirmed {
                return Some(FactRef {
                    key: key.clone(),
                    confidence_bps: fact.confidence_bps,
                    depth: fact.depth,
                });
            }
            return None;
        }
        if self.observed.contains(key) {
            return Some(FactRef {
                key: key.clone(),
                confidence_bps: OBSERVED_CONFIDENCE_BPS,
                depth: 0,
            });
        }
        None
    }

    fn lookup(&self, subject: &str, predicate: &str) -> Vec<FactRef> {
        let mut refs: Vec<FactRef> = self
            .observed
            .objects(subject, predicate)
            .into_iter()
            .map(|object| FactRef {
                key: FactKey::new(subject, predicate, object),
                confidence_bps: OBSERVED_CONFIDENCE_BPS,
                depth: 0,
            })
            .collect();
        for fact in self.virtuals.values() {
            if fact.key.subject == subject
                && fact.key.predicate == predicate
                && matches!(fact.state, VirtualState::Hypothesis | VirtualState::Confirmed)
            {
                refs.push(FactRef {
                    key: fact.key.clone(),
                    confidence_bps: fact.confidence_bps,
                    depth: fact.depth,
                });
            }
        }
        refs.sort_by(|left, right| {
            right
                .confidence_bps
                .cmp(&left.confidence_bps)
                .then_with(|| left.depth.cmp(&right.depth))
                .then_with(|| left.key.cmp(&right.key))
        });
        refs
    }

    fn lookup_incoming(&self, predicate: &str, object: &str) -> Vec<FactRef> {
        let mut refs: Vec<FactRef> = self
            .observed
            .subjects(predicate, object)
            .into_iter()
            .map(|subject| FactRef {
                key: FactKey::new(subject, predicate, object),
                confidence_bps: OBSERVED_CONFIDENCE_BPS,
                depth: 0,
            })
            .collect();
        for fact in self.virtuals.values() {
            if fact.key.object == object
                && fact.key.predicate == predicate
                && matches!(fact.state, VirtualState::Hypothesis | VirtualState::Confirmed)
            {
                refs.push(FactRef {
                    key: fact.key.clone(),
                    confidence_bps: fact.confidence_bps,
                    depth: fact.depth,
                });
            }
        }
        refs.sort_by(|left, right| {
            right
                .confidence_bps
                .cmp(&left.confidence_bps)
                .then_with(|| left.depth.cmp(&right.depth))
                .then_with(|| left.key.cmp(&right.key))
        });
        refs
    }

    fn saturate(&mut self) {
        let mut round = 0usize;
        let round_limit = self.config.max_depth.saturating_add(2);
        while !self.active.is_empty() && round < round_limit {
            round += 1;
            let frontier: Vec<FactKey> = std::mem::take(&mut self.active);
            self.queued.clear();
            let mut candidates: BTreeMap<FactKey, Candidate> = BTreeMap::new();
            let mut order = 0usize;
            let mut budget_hit = false;
            'frontier: for key in frontier {
                let Some(premise) = self.fact_ref(&key) else {
                    continue;
                };
                for rule in &self.rules {
                    if self.expansions >= self.config.max_expansions {
                        budget_hit = true;
                        break 'frontier;
                    }
                    self.expansions += 1;
                    for candidate in self.expand_rule(rule, &premise, &mut order) {
                        if candidate.depth > self.config.max_depth {
                            continue;
                        }
                        if candidate.confidence_bps < self.config.min_confidence_bps {
                            continue;
                        }
                        match candidates.get(&candidate.key) {
                            Some(existing) if !better_candidate(&candidate, existing) => {}
                            _ => {
                                candidates.insert(candidate.key.clone(), candidate);
                            }
                        }
                    }
                }
            }

            let mut ranked: Vec<Candidate> = candidates.into_values().collect();
            ranked.sort_by(compare_candidates);
            let omitted = ranked.len().saturating_sub(self.config.max_candidates_per_round);
            let accepted = ranked.into_iter().take(self.config.max_candidates_per_round);
            let mut inserted = 0usize;
            for candidate in accepted {
                if self.virtuals.len() >= self.config.max_virtual_facts {
                    budget_hit = true;
                    break;
                }
                if self.observed.contains(&candidate.key)
                    || self.virtuals.contains_key(&candidate.key)
                    || self.refuted.contains(&candidate.key)
                {
                    continue;
                }
                if candidate.support.iter().any(|premise| premise == &candidate.key) {
                    continue;
                }
                if inserted >= self.config.beam_width {
                    budget_hit = true;
                    break;
                }
                self.insert_virtual(candidate);
                inserted += 1;
            }
            if omitted > 0 {
                self.mark_truncated("候选数上限");
            }
            if budget_hit {
                self.mark_truncated("预算上限");
            }
            if inserted == 0 {
                break;
            }
        }
        if round >= round_limit && !self.active.is_empty() {
            self.mark_truncated("跳深上限");
        }
    }

    fn expand_rule(
        &self,
        rule: &ConstructionRule,
        premise: &FactRef,
        order: &mut usize,
    ) -> Vec<Candidate> {
        let fact = &premise.key;
        let mut out = Vec::new();
        match rule.operator {
            ConstructionOperator::Inverse => {
                if fact.predicate == rule.body[0] {
                    *order += 1;
                    out.push(self.candidate(
                        FactKey::new(&fact.object, &rule.head, &fact.subject),
                        rule,
                        vec![fact.clone()],
                        premise,
                        *order,
                    ));
                }
            }
            ConstructionOperator::Compose => {
                if rule.body.len() < 2 {
                    return out;
                }
                if fact.predicate == rule.body[0] {
                    for other in self.lookup(&fact.object, &rule.body[1]) {
                        *order += 1;
                        out.push(self.candidate(
                            FactKey::new(&fact.subject, &rule.head, &other.key.object),
                            rule,
                            vec![fact.clone(), other.key.clone()],
                            premise,
                            *order,
                        ));
                    }
                }
                if fact.predicate == rule.body[1] {
                    for other in self.lookup_incoming(&rule.body[0], &fact.subject) {
                        *order += 1;
                        out.push(self.candidate(
                            FactKey::new(&other.key.subject, &rule.head, &fact.object),
                            rule,
                            vec![other.key.clone(), fact.clone()],
                            premise,
                            *order,
                        ));
                    }
                }
            }
            ConstructionOperator::CommonNeighbor => {
                if rule.body.len() < 2 {
                    return out;
                }
                if fact.predicate == rule.body[0] {
                    for other in self.lookup_incoming(&rule.body[1], &fact.object) {
                        *order += 1;
                        out.push(self.candidate(
                            FactKey::new(&fact.subject, &rule.head, &other.key.subject),
                            rule,
                            vec![fact.clone(), other.key.clone()],
                            premise,
                            *order,
                        ));
                    }
                }
                if fact.predicate == rule.body[1] {
                    for other in self.lookup_incoming(&rule.body[0], &fact.object) {
                        *order += 1;
                        out.push(self.candidate(
                            FactKey::new(&other.key.subject, &rule.head, &fact.subject),
                            rule,
                            vec![other.key.clone(), fact.clone()],
                            premise,
                            *order,
                        ));
                    }
                }
            }
            ConstructionOperator::CommonObject => {
                if rule.body.len() < 2 {
                    return out;
                }
                if fact.predicate == rule.body[0] {
                    for other in self.lookup(&fact.subject, &rule.body[1]) {
                        *order += 1;
                        out.push(self.candidate(
                            FactKey::new(&fact.object, &rule.head, &other.key.object),
                            rule,
                            vec![fact.clone(), other.key.clone()],
                            premise,
                            *order,
                        ));
                    }
                }
                if fact.predicate == rule.body[1] {
                    for other in self.lookup(&fact.subject, &rule.body[0]) {
                        *order += 1;
                        out.push(self.candidate(
                            FactKey::new(&fact.object, &rule.head, &other.key.object),
                            rule,
                            vec![other.key.clone(), fact.clone()],
                            premise,
                            *order,
                        ));
                    }
                }
            }
        }
        out
    }

    fn candidate(
        &self,
        key: FactKey,
        rule: &ConstructionRule,
        support: Vec<FactKey>,
        premise: &FactRef,
        order: usize,
    ) -> Candidate {
        let support = dedupe_support(key.clone(), support);
        let depth = support
            .iter()
            .filter_map(|key| self.fact_ref(key))
            .map(|fact| fact.depth)
            .max()
            .unwrap_or(0)
            .saturating_add(1);
        let mut confidence = u32::from(rule.confidence_bps)
            .saturating_mul(u32::from(VIRTUAL_FACT_PENALTY_BPS))
            / 10_000;
        for key in &support {
            let bps = if key == &premise.key {
                premise.confidence_bps
            } else {
                self.fact_ref(key)
                    .map(|fact| fact.confidence_bps)
                    .unwrap_or(0)
            };
            confidence = confidence.saturating_mul(u32::from(bps)) / 10_000;
        }
        let virtual_premises = support
            .iter()
            .filter(|key| self.virtuals.contains_key(*key))
            .count();
        for _ in 0..virtual_premises {
            confidence = confidence.saturating_mul(u32::from(VIRTUAL_DAMPING_BPS)) / 10_000;
        }
        let confidence_bps = confidence
            .min(u32::from(MAX_VIRTUAL_CONFIDENCE_BPS)) as u16;
        Candidate {
            key,
            confidence_bps,
            depth,
            operator: rule.operator,
            rule: rule.id.clone(),
            support,
            order,
        }
    }

    fn insert_virtual(&mut self, candidate: Candidate) {
        let fact = VirtualFact {
            key: candidate.key.clone(),
            state: VirtualState::Hypothesis,
            scope: FactScope::Query,
            confidence_bps: candidate.confidence_bps,
            depth: candidate.depth,
            operator: candidate.operator,
            rule: candidate.rule,
            support: candidate.support.clone(),
            revision: self.revision,
            touched: self.next_tick(),
        };
        for premise in &candidate.support {
            self.dependents
                .entry(premise.clone())
                .or_default()
                .insert(candidate.key.clone());
        }
        let subject = candidate.key.subject.clone();
        let object = candidate.key.object.clone();
        self.virtuals.insert(candidate.key.clone(), fact);
        self.enqueue_without_seed(candidate.key);
        self.load_entity(&subject);
        self.load_entity(&object);
    }

    fn next_tick(&mut self) -> u64 {
        self.tick = self.tick.saturating_add(1);
        self.tick
    }

    pub fn has_virtual(&self, key: &FactKey) -> Option<&VirtualFact> {
        self.virtuals.get(key)
    }

    pub fn confirm(&mut self, key: &FactKey) -> bool {
        let Some(fact) = self.virtuals.get_mut(key) else {
            return false;
        };
        if fact.state == VirtualState::Refuted || fact.state == VirtualState::Invalidated {
            return false;
        }
        let tick = {
            self.tick = self.tick.saturating_add(1);
            self.tick
        };
        fact.state = VirtualState::Confirmed;
        fact.scope = FactScope::Session;
        fact.revision = self.revision;
        fact.touched = tick;
        self.enforce_session_capacity();
        true
    }

    pub fn refute(&mut self, key: &FactKey) -> usize {
        let tick = {
            self.tick = self.tick.saturating_add(1);
            self.tick
        };
        self.refuted.insert(key.clone());
        if let Some(fact) = self.virtuals.get_mut(key) {
            fact.state = VirtualState::Refuted;
            fact.scope = FactScope::Session;
            fact.touched = tick;
        }
        self.invalidate_dependents(key)
    }

    pub fn invalidate_dependents(&mut self, key: &FactKey) -> usize {
        let mut stack: Vec<FactKey> = self
            .dependents
            .get(key)
            .map(|keys| keys.iter().cloned().collect())
            .unwrap_or_default();
        let mut visited: HashSet<FactKey> = HashSet::new();
        let mut invalidated = 0usize;
        while let Some(dependent) = stack.pop() {
            if !visited.insert(dependent.clone()) {
                continue;
            }
            if let Some(fact) = self.virtuals.get_mut(&dependent) {
                if matches!(
                    fact.state,
                    VirtualState::Hypothesis | VirtualState::Confirmed
                ) {
                    fact.state = VirtualState::Invalidated;
                    invalidated += 1;
                }
            }
            if let Some(children) = self.dependents.get(&dependent) {
                for child in children {
                    if !visited.contains(child) {
                        stack.push(child.clone());
                    }
                }
            }
        }
        invalidated
    }

    pub fn end_query(&mut self) -> usize {
        let keys: Vec<FactKey> = self
            .virtuals
            .iter()
            .filter(|(_, fact)| fact.scope == FactScope::Query)
            .map(|(key, _)| key.clone())
            .collect();
        for key in &keys {
            self.virtuals.remove(key);
        }
        for dependents in self.dependents.values_mut() {
            for key in &keys {
                dependents.remove(key);
            }
        }
        self.dependents.retain(|_, dependents| !dependents.is_empty());
        self.active.clear();
        self.queued.clear();
        self.loaded_entities.clear();
        keys.len()
    }

    pub fn on_revision(&mut self, revision: u64) -> usize {
        if revision <= self.revision {
            return 0;
        }
        self.revision = revision;
        self.end_query();
        let mut invalidated = 0usize;
        for fact in self.virtuals.values_mut() {
            if fact.scope == FactScope::Session && fact.state == VirtualState::Hypothesis {
                fact.state = VirtualState::Invalidated;
                invalidated += 1;
            } else if fact.scope == FactScope::Session {
                fact.revision = revision;
            }
        }
        invalidated
    }

    pub fn session_facts(&self) -> Vec<&VirtualFact> {
        self.virtuals
            .values()
            .filter(|fact| fact.scope == FactScope::Session)
            .collect()
    }

    pub fn promotable(&self) -> Vec<&VirtualFact> {
        self.session_facts()
            .into_iter()
            .filter(|fact| {
                fact.state == VirtualState::Confirmed
                    && fact.confidence_bps >= self.config.promotion_threshold_bps
            })
            .collect()
    }

    fn enforce_session_capacity(&mut self) {
        let mut session: Vec<(FactKey, u64, bool)> = self
            .virtuals
            .values()
            .filter(|fact| fact.scope == FactScope::Session)
            .map(|fact| (fact.key.clone(), fact.touched, fact.state == VirtualState::Confirmed))
            .collect();
        if session.len() <= self.config.session_capacity {
            return;
        }
        session.sort_by(|left, right| {
            left.2
                .cmp(&right.2)
                .then_with(|| left.1.cmp(&right.1))
                .then_with(|| left.0.cmp(&right.0))
        });
        let excess = session.len() - self.config.session_capacity;
        for (key, _, _) in session.into_iter().take(excess) {
            self.virtuals.remove(&key);
            for dependents in self.dependents.values_mut() {
                dependents.remove(&key);
            }
        }
    }

    pub fn support_audit(&self) -> SupportAudit {
        let mut audit = SupportAudit {
            inspected: self.virtuals.len(),
            invalid: Vec::new(),
            cyclic: Vec::new(),
        };
        for (key, fact) in &self.virtuals {
            if fact.state == VirtualState::Refuted {
                continue;
            }
            let mut visiting = HashSet::new();
            let mut memo = HashSet::new();
            if has_cycle(self, key, &mut visiting) {
                audit.cyclic.push(key.clone());
            } else if !has_observed_leaf(self, key, &mut memo) {
                audit.invalid.push(key.clone());
            }
        }
        audit
    }

    fn support_closure(&self, key: &FactKey) -> BTreeSet<FactKey> {
        let mut closure = BTreeSet::new();
        let mut stack = vec![key.clone()];
        while let Some(current) = stack.pop() {
            let Some(fact) = self.virtuals.get(&current) else {
                continue;
            };
            for premise in &fact.support {
                if self.virtuals.contains_key(premise) && closure.insert(premise.clone()) {
                    stack.push(premise.clone());
                }
            }
        }
        closure
    }
}

fn has_observed_leaf(
    reasoner: &ConstructiveReasoner,
    key: &FactKey,
    memo: &mut HashSet<FactKey>,
) -> bool {
    if reasoner.observed.contains(key) {
        return true;
    }
    if !memo.insert(key.clone()) {
        return false;
    }
    let Some(fact) = reasoner.virtuals.get(key) else {
        return false;
    };
    if fact.support.is_empty() {
        return false;
    }
    fact.support
        .iter()
        .any(|premise| has_observed_leaf(reasoner, premise, memo))
}

fn has_cycle(
    reasoner: &ConstructiveReasoner,
    key: &FactKey,
    visiting: &mut HashSet<FactKey>,
) -> bool {
    if !visiting.insert(key.clone()) {
        return true;
    }
    let Some(fact) = reasoner.virtuals.get(key) else {
        visiting.remove(key);
        return false;
    };
    for premise in &fact.support {
        if reasoner.virtuals.contains_key(premise) && has_cycle(reasoner, premise, visiting) {
            return true;
        }
    }
    visiting.remove(key);
    false
}

fn dedupe_support(key: FactKey, support: Vec<FactKey>) -> Vec<FactKey> {
    let mut seen = BTreeSet::new();
    support
        .into_iter()
        .filter(|premise| premise != &key)
        .filter(|premise| seen.insert(premise.clone()))
        .collect()
}

fn better_candidate(candidate: &Candidate, existing: &Candidate) -> bool {
    compare_candidates(candidate, existing) == Ordering::Less
}

fn compare_candidates(left: &Candidate, right: &Candidate) -> Ordering {
    right
        .confidence_bps
        .cmp(&left.confidence_bps)
        .then_with(|| left.depth.cmp(&right.depth))
        .then_with(|| left.key.cmp(&right.key))
        .then_with(|| left.operator.cmp(&right.operator))
        .then_with(|| left.rule.cmp(&right.rule))
        .then_with(|| left.order.cmp(&right.order))
}

fn quantize(value: f64) -> u16 {
    if !value.is_finite() {
        return 0;
    }
    (value.clamp(0.0, 1.0) * 10_000.0).round() as u16
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::link_prediction::{HornRule, RuleTemplate, Triple};

    fn rule(
        operator: ConstructionOperator,
        body: &[&str],
        head: &str,
        confidence: u16,
    ) -> ConstructionRule {
        ConstructionRule {
            id: format!("测试规则-{head}"),
            operator,
            body: body.iter().map(|item| item.to_string()).collect(),
            head: head.to_string(),
            confidence_bps: confidence,
        }
    }

    fn engine(triples: &[(&str, &str, &str)], rules: Vec<ConstructionRule>) -> ConstructiveReasoner {
        let triples: Vec<Triple> = triples
            .iter()
            .map(|(s, p, o)| Triple::new(s, p, o))
            .collect();
        let config = ConstructionConfig {
            enabled: true,
            ..ConstructionConfig::default()
        };
        ConstructiveReasoner::new(
            config,
            1,
            ObservedGraph::from_triples(&triples),
            rules,
        )
    }

    #[test]
    fn compose_fills_missing_target_edge() {
        let mut reasoner = engine(
            &[("甲", "同事", "丙"), ("丙", "师傅", "乙")],
            vec![rule(ConstructionOperator::Compose, &["同事", "师傅"], "师傅", 9_000)],
        );
        let report = reasoner.derive("甲", "师傅", false);
        assert_eq!(report.status, ConstructionStatus::Constructed);
        assert_eq!(report.target_facts.len(), 1);
        let fact = &report.target_facts[0];
        assert_eq!(fact.key, FactKey::new("甲", "师傅", "乙"));
        assert_eq!(fact.depth, 1);
        assert!(fact.confidence_bps > 7_000 && fact.confidence_bps < 9_000);
        assert!(fact.support.contains(&FactKey::new("甲", "同事", "丙")));
        assert!(fact.support.contains(&FactKey::new("丙", "师傅", "乙")));
        assert!(reasoner.support_audit().is_clean());
    }

    #[test]
    fn virtual_fact_can_feed_the_next_hop() {
        let mut reasoner = engine(
            &[
                ("甲", "认识", "乙"),
                ("乙", "认识", "丙"),
                ("丙", "认识", "丁"),
            ],
            vec![
                rule(ConstructionOperator::Compose, &["认识", "认识"], "熟识", 9_500),
                rule(ConstructionOperator::Compose, &["熟识", "认识"], "熟识", 9_500),
            ],
        );
        let report = reasoner.derive("甲", "熟识", false);
        let values: BTreeSet<String> = report
            .target_facts
            .iter()
            .map(|fact| fact.key.object.clone())
            .collect();
        assert!(values.contains("丙"));
        assert!(values.contains("丁"), "二跳虚拟事实必须继续参与合成");

        let deeper = report
            .target_facts
            .iter()
            .find(|fact| fact.key.object == "丁")
            .expect("应能由虚拟事实再推一跳");
        assert_eq!(deeper.depth, 2);
        assert!(deeper
            .support
            .iter()
            .any(|key| key == &FactKey::new("甲", "熟识", "丙")));
        assert!(reasoner.support_audit().is_clean());
    }

    #[test]
    fn inverse_rule_derives_incoming_target() {
        let mut reasoner = engine(
            &[("甲", "师傅", "乙")],
            vec![rule(ConstructionOperator::Inverse, &["师傅"], "徒弟", 9_000)],
        );
        let report = reasoner.derive("乙", "徒弟", false);
        assert_eq!(report.status, ConstructionStatus::Constructed);
        assert_eq!(report.target_facts[0].key, FactKey::new("乙", "徒弟", "甲"));
    }

    #[test]
    fn refuting_a_premise_invalidates_the_dependents() {
        let mut reasoner = engine(
            &[("甲", "同事", "丙"), ("丙", "师傅", "乙")],
            vec![rule(ConstructionOperator::Compose, &["同事", "师傅"], "师傅", 9_000)],
        );
        let report = reasoner.derive("甲", "师傅", false);
        let derived = report.target_facts[0].key.clone();
        assert_eq!(reasoner.refute(&FactKey::new("甲", "同事", "丙")), 1);
        assert_eq!(
            reasoner.has_virtual(&derived).unwrap().state,
            VirtualState::Invalidated
        );
    }

    #[test]
    fn confirmation_promotes_only_session_facts() {
        let mut reasoner = engine(
            &[("甲", "同事", "丙"), ("丙", "师傅", "乙")],
            vec![rule(ConstructionOperator::Compose, &["同事", "师傅"], "师傅", 9_000)],
        );
        let key = reasoner.derive("甲", "师傅", false).target_facts[0].key.clone();
        assert!(reasoner.confirm(&key));
        assert_eq!(reasoner.session_facts().len(), 1);
        assert_eq!(reasoner.promotable().len(), 1);
        assert_eq!(reasoner.end_query(), 0);
        assert_eq!(
            reasoner.has_virtual(&key).unwrap().state,
            VirtualState::Confirmed
        );
    }

    #[test]
    fn revision_drops_query_facts_and_invalidates_session_hypotheses() {
        let mut reasoner = engine(
            &[("甲", "同事", "丙"), ("丙", "师傅", "乙")],
            vec![rule(ConstructionOperator::Compose, &["同事", "师傅"], "师傅", 9_000)],
        );
        let query_key = reasoner.derive("甲", "师傅", false).target_facts[0].key.clone();
        reasoner.confirm(&query_key);
        let second = reasoner.derive("乙", "徒弟", true);
        assert!(second.is_empty() || second.target_facts.iter().all(|fact| fact.state == VirtualState::Hypothesis));
        assert_eq!(reasoner.on_revision(2), 0);
        assert!(reasoner.session_facts().iter().all(|fact| fact.state == VirtualState::Confirmed));
        assert_eq!(reasoner.revision(), 2);
    }

    #[test]
    fn disabled_engine_never_derives() {
        let triples = vec![Triple::new("甲", "同事", "丙"), Triple::new("丙", "师傅", "乙")];
        let mut reasoner = ConstructiveReasoner::from_predictor(
            ConstructionConfig::default(),
            1,
            &triples,
            &[HornRule {
                template: RuleTemplate::Compose,
                body: vec!["同事".to_string(), "师傅".to_string()],
                head: "师傅".to_string(),
                support: 3,
                body_count: 3,
                confidence: 0.9,
                head_coverage: 0.8,
            }],
        );
        let report = reasoner.derive("甲", "师傅", false);
        assert_eq!(report.status, ConstructionStatus::Disabled);
        assert!(report.target_facts.is_empty());
    }

    #[test]
    fn reports_are_deterministic_and_render_epistemic_status() {
        let build = || {
            engine(
                &[("甲", "同事", "丙"), ("丙", "师傅", "乙")],
                vec![rule(ConstructionOperator::Compose, &["同事", "师傅"], "师傅", 9_000)],
            )
        };
        let mut left = build();
        let mut right = build();
        let left_report = left.derive("甲", "师傅", false);
        let right_report = right.derive("甲", "师傅", false);
        assert_eq!(left_report, right_report);
        let text = left_report.render_text();
        assert!(text.contains("推断（待确认）"), "{}", text);
        assert!(text.contains("未写基础图谱"), "{}", text);
    }

    #[test]
    fn config_limits_and_min_confidence_are_enforced() {
        let triples = vec![Triple::new("甲", "同事", "丙"), Triple::new("丙", "师傅", "乙")];
        let config = ConstructionConfig {
            enabled: true,
            min_confidence_bps: 9_999,
            ..ConstructionConfig::default()
        };
        let mut reasoner = ConstructiveReasoner::new(
            config,
            1,
            ObservedGraph::from_triples(&triples),
            vec![rule(ConstructionOperator::Compose, &["同事", "师傅"], "师傅", 9_000)],
        );
        let report = reasoner.derive("甲", "师傅", false);
        assert!(report.target_facts.is_empty());
        assert_eq!(report.status, ConstructionStatus::Exhausted);
    }

    #[test]
    fn observed_target_is_not_duplicated_as_a_virtual_fact() {
        let mut reasoner = engine(
            &[("甲", "师傅", "乙")],
            vec![rule(ConstructionOperator::Inverse, &["师傅"], "徒弟", 9_000)],
        );
        let report = reasoner.derive("甲", "师傅", false);
        assert_eq!(report.status, ConstructionStatus::AlreadyObserved);
        assert!(report.target_facts.is_empty());
        assert!(reasoner.virtual_facts().next().is_none());
    }

    #[test]
    fn common_neighbor_rule_joins_two_subjects() {
        let mut reasoner = engine(
            &[("甲", "朋友", "丙"), ("乙", "朋友", "丙")],
            vec![rule(
                ConstructionOperator::CommonNeighbor,
                &["朋友", "朋友"],
                "同友",
                8_000,
            )],
        );
        let report = reasoner.derive("甲", "同友", false);
        assert!(report
            .target_facts
            .iter()
            .any(|fact| fact.key == FactKey::new("甲", "同友", "乙")));
    }
}

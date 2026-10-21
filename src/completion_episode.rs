//! 补全回合（Completion Episode）：把一次「确认」变成下一次推理的前提。
//!
//! 链接预测、关联挖掘与统计续写都能给出候选，但它们只解释、不记忆：
//! 用户确认一条推测后，系统既不归纳规则，也不在同类实体上批量补全。
//! 本模块补上这段生命周期：
//!
//! ```text
//! 打开回合 → 收集候选 → 用户确认/驳回
//!          → 从候选证据归纳候选规则（默认作用域=本批）
//!          → 在批次范围内做有界闭包，得到一批新候选
//!          → 用户可把规则显式晋升到命名空间/全局
//! ```
//!
//! 纪律：
//! - 只有 `Entailed` 是确定答案；`Induced`/`Analogical` 必须带 hedge；
//! - 归纳规则默认 `Batch` 作用域，不自动晋升为全局；
//! - 开放世界：缺失不是否定；负例只来自用户驳回；
//! - 闭包有候选数、扫描数与规则数上限，溢出时明确标注「本批未补全」；
//! - 候选 id 与批次 id 由规范形式哈希得出，与插入顺序无关。

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::link_prediction::{HornRule, Prediction, RuleTemplate, Triple};
use crate::resolution_engine::{Atom, Clause, Term};

/// 一个回合最多保留的候选数。
pub const DEFAULT_MAX_EPISODE_CANDIDATES: usize = 24;
/// 一次闭包最多新增的候选数。
pub const DEFAULT_MAX_CLOSURE_CANDIDATES: usize = 48;
/// 单个规则一次闭包最多扫描的候选/边数（按候选数的倍数计）。
pub const CLOSURE_SCAN_FACTOR: usize = 20;
/// 同时保留的回合数。
pub const DEFAULT_MAX_EPISODES: usize = 32;
/// 已归纳规则的数量上限。
pub const DEFAULT_MAX_RULES: usize = 64;
/// 批次实体的数量上限。
pub const DEFAULT_MAX_BATCH_ENTITIES: usize = 256;

/// 一条补全的证据强度。强度顺序：Entailed > Induced > Analogical。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Warrant {
    /// 由已激活规则与事实必然推出，可直接作为确定答案。
    Entailed,
    /// 由归纳/统计规则推出，必须标注「归纳，待确认」。
    Induced,
    /// 由类型/关联/路径类比推出，只能提示或追问。
    Analogical,
}

impl Warrant {
    pub fn label(self) -> &'static str {
        match self {
            Warrant::Entailed => "必然",
            Warrant::Induced => "归纳",
            Warrant::Analogical => "类比",
        }
    }

    /// 回答时的限定语：只有必然项可以为空。
    pub fn hedge(self) -> &'static str {
        match self {
            Warrant::Entailed => "",
            Warrant::Induced => "（归纳，待确认）",
            Warrant::Analogical => "（类比，仅提示）",
        }
    }

    pub fn is_certain(self) -> bool {
        matches!(self, Warrant::Entailed)
    }

    fn strength(self) -> u8 {
        match self {
            Warrant::Entailed => 3,
            Warrant::Induced => 2,
            Warrant::Analogical => 1,
        }
    }

    fn stronger(self, other: Self) -> Self {
        if self.strength() >= other.strength() {
            self
        } else {
            other
        }
    }
}

/// 候选规则的作用域。默认 `Batch`，只有用户显式晋升才扩大。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuleScope {
    /// 只解释当前例子。
    Example,
    /// 只作用于当前回合/当前批次的实体。
    Batch,
    /// 作用于同一来源命名空间。
    Namespace,
    /// 全局生效。
    Global,
}

impl RuleScope {
    pub fn label(self) -> &'static str {
        match self {
            RuleScope::Example => "本例",
            RuleScope::Batch => "本批",
            RuleScope::Namespace => "命名空间",
            RuleScope::Global => "全局",
        }
    }

    pub fn rank(self) -> u8 {
        match self {
            RuleScope::Example => 0,
            RuleScope::Batch => 1,
            RuleScope::Namespace => 2,
            RuleScope::Global => 3,
        }
    }

    /// 晋升到该作用域建议的最少独立确认回合数。
    pub fn recommended_episodes(self) -> usize {
        match self {
            RuleScope::Example => 1,
            RuleScope::Batch => 1,
            RuleScope::Namespace => 2,
            RuleScope::Global => 3,
        }
    }
}

/// 规则的生命周期状态。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuleStatus {
    Candidate,
    Active,
    Quarantined,
    Retired,
}

impl RuleStatus {
    pub fn label(self) -> &'static str {
        match self {
            RuleStatus::Candidate => "候选",
            RuleStatus::Active => "生效",
            RuleStatus::Quarantined => "隔离",
            RuleStatus::Retired => "退役",
        }
    }
}

/// 单条候选的状态。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateStatus {
    Pending,
    Confirmed,
    Refuted,
}

impl CandidateStatus {
    pub fn label(self) -> &'static str {
        match self {
            CandidateStatus::Pending => "待确认",
            CandidateStatus::Confirmed => "已确认",
            CandidateStatus::Refuted => "已驳回",
        }
    }
}

/// 候选的来源通道（审计用）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateOrigin {
    PathPrediction,
    BatchClosure,
    Analogical,
    Manual,
}

impl CandidateOrigin {
    pub fn label(self) -> &'static str {
        match self {
            CandidateOrigin::PathPrediction => "链接预测",
            CandidateOrigin::BatchClosure => "批次闭包",
            CandidateOrigin::Analogical => "类比联想",
            CandidateOrigin::Manual => "手工提名",
        }
    }
}

/// 规则模板（与 `LinkPredictor` 的模板同构，但可独立序列化）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RulePattern {
    Compose,
    Inverse,
    CommonNeighbor,
    CommonObject,
}

impl RulePattern {
    pub fn from_template(template: RuleTemplate) -> Self {
        match template {
            RuleTemplate::Compose => RulePattern::Compose,
            RuleTemplate::Inverse => RulePattern::Inverse,
            RuleTemplate::CommonNeighbor => RulePattern::CommonNeighbor,
            RuleTemplate::CommonObject => RulePattern::CommonObject,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            RulePattern::Compose => "组合",
            RulePattern::Inverse => "逆向",
            RulePattern::CommonNeighbor => "共同邻居",
            RulePattern::CommonObject => "共同主体",
        }
    }
}

/// 归纳规则的结构化表示。真正交给归结引擎时再编译成 [`Clause`]。
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RuleSpec {
    pub pattern: RulePattern,
    pub body: Vec<String>,
    pub head: String,
}

impl RuleSpec {
    pub fn from_horn(rule: &HornRule) -> Option<Self> {
        let pattern = RulePattern::from_template(rule.template);
        let body: Vec<String> = rule.body.iter().cloned().collect();
        let spec = Self {
            pattern,
            body,
            head: rule.head.clone(),
        };
        spec.is_valid().then_some(spec)
    }

    pub fn is_valid(&self) -> bool {
        if self.head.trim().is_empty() {
            return false;
        }
        match self.pattern {
            RulePattern::Compose | RulePattern::CommonNeighbor | RulePattern::CommonObject => {
                self.body.len() == 2
                    && self.body.iter().all(|predicate| !predicate.trim().is_empty())
            }
            RulePattern::Inverse => {
                self.body.len() == 1 && !self.body[0].trim().is_empty()
            }
        }
    }

    /// 稳定 id：同一模板/体/头恒定，与插入顺序无关。
    pub fn id(&self) -> String {
        stable_id(
            "rule-",
            &format!(
                "{}:-{}({})",
                self.head,
                self.body.join("&"),
                self.pattern.label()
            ),
        )
    }

    pub fn rule_text(&self) -> String {
        match self.pattern {
            RulePattern::Compose => format!(
                "{}(A,B) ∧ {}(B,C) ⇒ {}(A,C)",
                self.body.first().map(String::as_str).unwrap_or(""),
                self.body.get(1).map(String::as_str).unwrap_or(""),
                self.head
            ),
            RulePattern::Inverse => format!(
                "{}(A,B) ⇒ {}(B,A)",
                self.body.first().map(String::as_str).unwrap_or(""),
                self.head
            ),
            RulePattern::CommonNeighbor => format!(
                "{}(A,C) ∧ {}(B,C) ⇒ {}(A,B)",
                self.body.first().map(String::as_str).unwrap_or(""),
                self.body.get(1).map(String::as_str).unwrap_or(""),
                self.head
            ),
            RulePattern::CommonObject => format!(
                "{}(C,A) ∧ {}(C,B) ⇒ {}(A,B)",
                self.body.first().map(String::as_str).unwrap_or(""),
                self.body.get(1).map(String::as_str).unwrap_or(""),
                self.head
            ),
        }
    }

    /// 编译成归结引擎唯一的规则表示。
    pub fn to_clause(&self, confidence: f64, source: &str) -> Option<Clause> {
        if !self.is_valid() {
            return None;
        }
        let a = || Term::var("A");
        let b = || Term::var("B");
        let c = || Term::var("C");
        let clause = match self.pattern {
            RulePattern::Compose => Clause::new(
                Atom::pattern(&self.head, a(), c()),
                vec![
                    Atom::pattern(&self.body[0], a(), b()),
                    Atom::pattern(&self.body[1], b(), c()),
                ],
                confidence,
                source,
            ),
            RulePattern::Inverse => Clause::new(
                Atom::pattern(&self.head, b(), a()),
                vec![Atom::pattern(&self.body[0], a(), b())],
                confidence,
                source,
            ),
            RulePattern::CommonNeighbor => Clause::new(
                Atom::pattern(&self.head, a(), b()),
                vec![
                    Atom::pattern(&self.body[0], a(), c()),
                    Atom::pattern(&self.body[1], b(), c()),
                ],
                confidence,
                source,
            ),
            RulePattern::CommonObject => Clause::new(
                Atom::pattern(&self.head, a(), b()),
                vec![
                    Atom::pattern(&self.body[0], c(), a()),
                    Atom::pattern(&self.body[1], c(), b()),
                ],
                confidence,
                source,
            ),
        };
        Some(clause)
    }
}

/// 一个补全候选。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompletionCandidate {
    pub id: String,
    pub triple: Triple,
    pub warrant: Warrant,
    pub status: CandidateStatus,
    pub confidence: f64,
    pub evidence: Vec<Triple>,
    pub explanation: String,
    pub rule_id: Option<String>,
    pub rule: Option<RuleSpec>,
    pub origin: CandidateOrigin,
    pub added_tick: u64,
}

impl CompletionCandidate {
    pub fn display_triple(&self) -> String {
        format!(
            "{}({}, {})",
            self.triple.predicate, self.triple.subject, self.triple.object
        )
    }
}

/// 一个回合：一次提问/导入打开，确认与驳回都落在其中。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompletionEpisode {
    pub id: String,
    pub focus: String,
    pub question: Option<String>,
    pub opened_tick: u64,
    pub batch_entities: BTreeSet<String>,
    pub namespace: String,
    pub candidates: Vec<CompletionCandidate>,
    pub notes: Vec<String>,
    pub truncated: bool,
}

impl CompletionEpisode {
    pub fn pending_count(&self) -> usize {
        self.candidates
            .iter()
            .filter(|candidate| candidate.status == CandidateStatus::Pending)
            .count()
    }

    pub fn confirmed_count(&self) -> usize {
        self.candidates
            .iter()
            .filter(|candidate| candidate.status == CandidateStatus::Confirmed)
            .count()
    }

    pub fn refuted_count(&self) -> usize {
        self.candidates
            .iter()
            .filter(|candidate| candidate.status == CandidateStatus::Refuted)
            .count()
    }

    pub fn candidate(&self, index: usize) -> Option<&CompletionCandidate> {
        self.candidates.get(index)
    }
}

/// 从确认例子归纳出的候选规则及其证据。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LearnedRule {
    pub id: String,
    pub spec: RuleSpec,
    pub scope: RuleScope,
    pub status: RuleStatus,
    /// 在多少个不同回合被确认（重复确认同一回合只计一次）。
    pub confirmations: usize,
    pub refutations: usize,
    pub episodes: BTreeSet<String>,
    pub positive_examples: Vec<Triple>,
    pub negative_examples: Vec<Triple>,
    pub confidence: f64,
    pub created_tick: u64,
    pub updated_tick: u64,
    pub source_episode: String,
    pub promoted_tick: Option<u64>,
}

impl LearnedRule {
    pub fn display_line(&self) -> String {
        format!(
            "[{} · {}] {}（确认 {} 次 / 驳回 {} 次，{} 个回合）",
            self.status.label(),
            self.scope.label(),
            self.spec.rule_text(),
            self.confirmations,
            self.refutations,
            self.episodes.len()
        )
    }

    pub fn is_usable(&self) -> bool {
        matches!(
            self.status,
            RuleStatus::Candidate | RuleStatus::Active
        )
    }

    fn note_confirmation(&mut self, episode_id: &str, example: &Triple, tick: u64, confidence: f64) {
        if self.episodes.insert(episode_id.to_string()) {
            self.confirmations += 1;
        }
        if !self.positive_examples.contains(example) {
            self.positive_examples.push(example.clone());
            self.positive_examples.sort();
        }
        self.confidence = self.confidence.max(confidence);
        self.updated_tick = tick;
        if self.status == RuleStatus::Candidate && self.confirmations >= 2 {
            self.status = RuleStatus::Active;
        }
    }

    fn note_refutation(&mut self, example: &Triple, tick: u64) -> bool {
        if !self.negative_examples.contains(example) {
            self.negative_examples.push(example.clone());
            self.negative_examples.sort();
        }
        self.refutations += 1;
        self.updated_tick = tick;
        if self.refutations >= 2 && self.refutations >= self.confirmations {
            self.status = RuleStatus::Quarantined;
            true
        } else {
            false
        }
    }
}

/// 确认一条候选后的回执。
#[derive(Debug, Clone, PartialEq)]
pub struct ConfirmOutcome {
    pub candidate: Triple,
    pub rule_id: Option<String>,
    pub rule_scope: Option<RuleScope>,
    pub learned: bool,
    pub new_candidates: usize,
    pub closure_truncated: bool,
}

/// 驳回一条候选后的回执。
#[derive(Debug, Clone, PartialEq)]
pub struct RefuteOutcome {
    pub candidate: Triple,
    pub rule_id: Option<String>,
    pub rule_status: Option<RuleStatus>,
    pub quarantined: bool,
    pub removed_pending: usize,
}

/// 规则晋升回执。
#[derive(Debug, Clone, PartialEq)]
pub struct PromotionOutcome {
    pub rule_id: String,
    pub from: RuleScope,
    pub to: RuleScope,
    pub warning: Option<String>,
}

/// 回合操作错误。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EpisodeError {
    NoActiveEpisode,
    CandidateNotFound(String),
    CandidateAlreadyClosed(String),
    RuleNotFound(String),
    RuleUnavailable(String),
    PromotionNotHigher {
        from: RuleScope,
        to: RuleScope,
    },
    PromotionBlocked(String),
}

impl fmt::Display for EpisodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EpisodeError::NoActiveEpisode => {
                write!(f, "还没有打开的补全回合，请先执行「补全回合：<实体>」")
            }
            EpisodeError::CandidateNotFound(selector) => {
                write!(f, "找不到候选「{}」（可用序号、候选 id 或 主体|谓词|客体）", selector)
            }
            EpisodeError::CandidateAlreadyClosed(selector) => {
                write!(f, "候选「{}」已有确认/驳回结论，不能再次裁定", selector)
            }
            EpisodeError::RuleNotFound(id) => write!(f, "找不到补全规则「{}」", id),
            EpisodeError::RuleUnavailable(id) => {
                write!(f, "补全规则「{}」已隔离或退役，不能继续使用", id)
            }
            EpisodeError::PromotionNotHigher { from, to } => write!(
                f,
                "规则作用域只能扩大：当前是「{}」，不能降为「{}」",
                from.label(),
                to.label()
            ),
            EpisodeError::PromotionBlocked(reason) => write!(f, "规则晋升被拒绝：{}", reason),
        }
    }
}

impl std::error::Error for EpisodeError {}

/// 有界的补全回合台账。可随会话快照序列化，但不自动写入知识图谱。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompletionLedger {
    episodes: Vec<CompletionEpisode>,
    active: Option<usize>,
    rules: Vec<LearnedRule>,
    refuted_triples: BTreeSet<Triple>,
    tick: u64,
    max_candidates: usize,
    max_closure: usize,
    max_episodes: usize,
    max_rules: usize,
    max_batch_entities: usize,
}

impl Default for CompletionLedger {
    fn default() -> Self {
        Self::new()
    }
}

impl CompletionLedger {
    pub fn new() -> Self {
        Self::with_limits(
            DEFAULT_MAX_EPISODE_CANDIDATES,
            DEFAULT_MAX_CLOSURE_CANDIDATES,
            DEFAULT_MAX_EPISODES,
            DEFAULT_MAX_RULES,
        )
    }

    pub fn with_limits(
        max_candidates: usize,
        max_closure: usize,
        max_episodes: usize,
        max_rules: usize,
    ) -> Self {
        Self {
            episodes: Vec::new(),
            active: None,
            rules: Vec::new(),
            refuted_triples: BTreeSet::new(),
            tick: 0,
            max_candidates: max_candidates.max(1),
            max_closure: max_closure.max(1),
            max_episodes: max_episodes.max(1),
            max_rules: max_rules.max(1),
            max_batch_entities: DEFAULT_MAX_BATCH_ENTITIES,
        }
    }

    pub fn tick(&self) -> u64 {
        self.tick
    }

    pub fn is_empty(&self) -> bool {
        self.episodes.is_empty() && self.rules.is_empty() && self.refuted_triples.is_empty()
    }

    /// 从持久化快照恢复后重新夹紧上限：快照文件不可信，不能让其中的
    /// 超大上限把有界闭包变成无界计算。
    pub fn clamp_limits(&mut self) {
        self.max_candidates = self
            .max_candidates
            .clamp(1, DEFAULT_MAX_EPISODE_CANDIDATES);
        self.max_closure = self.max_closure.clamp(1, DEFAULT_MAX_CLOSURE_CANDIDATES);
        self.max_episodes = self.max_episodes.clamp(1, DEFAULT_MAX_EPISODES);
        self.max_rules = self.max_rules.clamp(1, DEFAULT_MAX_RULES);
        self.max_batch_entities = self
            .max_batch_entities
            .clamp(1, DEFAULT_MAX_BATCH_ENTITIES);
        self.episodes.truncate(self.max_episodes);
        self.rules.truncate(self.max_rules);
        self.active = self.active.filter(|index| *index < self.episodes.len());
    }

    pub fn rules(&self) -> &[LearnedRule] {
        &self.rules
    }

    pub fn episodes(&self) -> &[CompletionEpisode] {
        &self.episodes
    }

    pub fn rule(&self, rule_id: &str) -> Option<&LearnedRule> {
        self.rules.iter().find(|rule| rule.id == rule_id)
    }

    pub fn current_episode(&self) -> Option<&CompletionEpisode> {
        self.active.and_then(|index| self.episodes.get(index))
    }

    pub fn current_episode_mut(&mut self) -> Option<&mut CompletionEpisode> {
        self.active
            .and_then(move |index| self.episodes.get_mut(index))
    }

    pub fn refuted_triples(&self) -> &BTreeSet<Triple> {
        &self.refuted_triples
    }

    /// 推进逻辑轮次（无挂钟，TTL/审计均用此计数）。
    pub fn begin_turn(&mut self) {
        self.tick += 1;
    }

    /// 打开一个新回合，返回其在台账中的索引。旧回合超出上限时淘汰最旧的。
    pub fn open_episode(&mut self, focus: &str, question: Option<&str>) -> usize {
        self.tick += 1;
        if self.episodes.len() >= self.max_episodes {
            self.episodes.remove(0);
            self.active = self.active.and_then(|index| index.checked_sub(1));
        }
        let id = stable_id(
            "ep-",
            &format!("{}@{}:{}", focus.trim(), self.tick, question.unwrap_or("")),
        );
        let index = self.episodes.len();
        self.episodes.push(CompletionEpisode {
            id,
            focus: focus.trim().to_string(),
            question: question.map(|value| value.trim().to_string()),
            opened_tick: self.tick,
            batch_entities: BTreeSet::new(),
            namespace: namespace_of(focus),
            candidates: Vec::new(),
            notes: Vec::new(),
            truncated: false,
        });
        self.active = Some(index);
        index
    }

    /// 设置本批实体范围（超出上限时截断并记录）。
    pub fn set_batch_scope<I>(&mut self, entities: I)
    where
        I: IntoIterator<Item = String>,
    {
        let max_batch_entities = self.max_batch_entities;
        let Some(episode) = self.current_episode_mut() else {
            return;
        };
        episode.batch_entities.clear();
        for entity in entities {
            let entity = entity.trim();
            if entity.is_empty() {
                continue;
            }
            if episode.batch_entities.len() >= max_batch_entities {
                episode.truncated = true;
                episode
                    .notes
                    .push(format!("本批实体超过 {} 个，已截断", max_batch_entities));
                break;
            }
            episode.batch_entities.insert(entity.to_string());
        }
    }

    pub fn note(&mut self, text: impl Into<String>) {
        if let Some(episode) = self.current_episode_mut() {
            episode.notes.push(text.into());
        }
    }

    /// 追加一条链接预测候选（warrant=Induced）。
    pub fn add_prediction(
        &mut self,
        prediction: &Prediction,
        rule: Option<&HornRule>,
        explanation: String,
    ) -> bool {
        let rule_spec = rule.and_then(RuleSpec::from_horn);
        let rule_id = rule_spec.as_ref().map(RuleSpec::id);
        self.insert_candidate(
            prediction.triple.clone(),
            Warrant::Induced,
            prediction.confidence,
            prediction.evidence.clone(),
            explanation,
            rule_spec,
            rule_id,
            CandidateOrigin::PathPrediction,
        )
    }

    /// 追加一条类比候选（warrant=Analogical，只能提示）。
    pub fn add_analogical(
        &mut self,
        triple: Triple,
        confidence: f64,
        evidence: Vec<Triple>,
        explanation: String,
    ) -> bool {
        self.insert_candidate(
            triple,
            Warrant::Analogical,
            confidence,
            evidence,
            explanation,
            None,
            None,
            CandidateOrigin::Analogical,
        )
    }

    /// 手工提名（warrant 由调用方负责，适合接入其他候选来源）。
    pub fn add_candidate(
        &mut self,
        triple: Triple,
        warrant: Warrant,
        confidence: f64,
        evidence: Vec<Triple>,
        explanation: String,
    ) -> bool {
        self.insert_candidate(
            triple,
            warrant,
            confidence,
            evidence,
            explanation,
            None,
            None,
            CandidateOrigin::Manual,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn insert_candidate(
        &mut self,
        triple: Triple,
        warrant: Warrant,
        confidence: f64,
        evidence: Vec<Triple>,
        explanation: String,
        rule: Option<RuleSpec>,
        rule_id: Option<String>,
        origin: CandidateOrigin,
    ) -> bool {
        if self.refuted_triples.contains(&triple) {
            return false;
        }
        let tick = self.tick;
        let max_candidates = self.max_candidates;
        let id = stable_id(
            "cand-",
            &format!("{}|{}|{}", triple.subject, triple.predicate, triple.object),
        );
        let Some(episode) = self.current_episode_mut() else {
            return false;
        };
        if let Some(existing) = episode
            .candidates
            .iter_mut()
            .find(|candidate| candidate.triple == triple)
        {
            if existing.status != CandidateStatus::Refuted {
                existing.warrant = existing.warrant.stronger(warrant);
                existing.confidence = existing.confidence.max(confidence);
                if explanation.len() > existing.explanation.len() {
                    existing.explanation = explanation;
                }
                merge_evidence(&mut existing.evidence, evidence);
                if existing.rule.is_none() {
                    existing.rule = rule;
                }
                if existing.rule_id.is_none() {
                    existing.rule_id = rule_id;
                }
            }
            return false;
        }
        if episode.candidates.len() >= max_candidates {
            episode.truncated = true;
            return false;
        }
        episode.candidates.push(CompletionCandidate {
            id,
            triple,
            warrant,
            status: CandidateStatus::Pending,
            confidence: confidence.clamp(0.0, 1.0),
            evidence,
            explanation,
            rule_id,
            rule,
            origin,
            added_tick: tick,
        });
        true
    }

    fn find_candidate_index(&self, selector: &str) -> Result<usize, EpisodeError> {
        let episode = self
            .current_episode()
            .ok_or(EpisodeError::NoActiveEpisode)?;
        let selector = selector.trim();
        if selector.is_empty() {
            return Err(EpisodeError::CandidateNotFound(selector.to_string()));
        }
        if let Ok(position) = selector.trim_start_matches('#').parse::<usize>() {
            if position == 0 || position > episode.candidates.len() {
                return Err(EpisodeError::CandidateNotFound(selector.to_string()));
            }
            return Ok(position - 1);
        }
        if let Some(position) = episode
            .candidates
            .iter()
            .position(|candidate| candidate.id == selector)
        {
            return Ok(position);
        }
        if let Some(position) = episode
            .candidates
            .iter()
            .position(|candidate| candidate.id.starts_with(selector))
        {
            return Ok(position);
        }
        if let Some((subject, predicate, object)) = parse_triple_selector(selector) {
            if let Some(position) = episode.candidates.iter().position(|candidate| {
                candidate.triple.subject == subject
                    && candidate.triple.predicate == predicate
                    && candidate.triple.object == object
            }) {
                return Ok(position);
            }
        }
        Err(EpisodeError::CandidateNotFound(selector.to_string()))
    }

    /// 确认一条候选：标记状态、归纳规则，并在本批范围上做有界闭包。
    pub fn confirm(
        &mut self,
        selector: &str,
        facts: &[Triple],
    ) -> Result<ConfirmOutcome, EpisodeError> {
        let position = self.find_candidate_index(selector)?;
        let active = self.active.ok_or(EpisodeError::NoActiveEpisode)?;
        let candidate = self.episodes[active].candidates[position].clone();
        if candidate.status != CandidateStatus::Pending {
            return Err(EpisodeError::CandidateAlreadyClosed(selector.to_string()));
        }
        let tick = self.tick;
        let episode_id = self.episodes[active].id.clone();
        let rule_id = if let Some(spec) = candidate.rule.clone() {
            let id = self.upsert_rule(
                spec,
                &candidate.triple,
                &episode_id,
                candidate.confidence,
                tick,
            );
            Some(id)
        } else {
            None
        };
        self.episodes[active].candidates[position].status = CandidateStatus::Confirmed;
        self.episodes[active].candidates[position].rule_id = rule_id.clone();
        let rule_scope = rule_id
            .as_deref()
            .and_then(|id| self.rule(id))
            .map(|rule| rule.scope);

        let mut closure_truncated = false;
        let mut new_candidates = 0usize;
        let learned: Vec<LearnedRule> = self
            .rules
            .iter()
            .filter(|rule| rule.is_usable())
            .cloned()
            .collect();
        let remaining = self.max_closure;
        let mut produced = 0usize;
        for rule in &learned {
            if produced >= remaining {
                closure_truncated = true;
                break;
            }
            let filter = self.scope_filter(rule.scope, active, facts);
            let (hits, truncated) = apply_rule(
                &rule.spec,
                facts,
                &filter,
                remaining.saturating_sub(produced),
            );
            closure_truncated |= truncated;
            for hit in hits {
                if produced >= remaining {
                    closure_truncated = true;
                    break;
                }
                let explanation = format!(
                    "由本回合规则「{}」（{}作用域）推出；证据：{}",
                    rule.spec.rule_text(),
                    rule.scope.label(),
                    hit.evidence
                        .iter()
                        .map(display_triple)
                        .collect::<Vec<_>>()
                        .join("；")
                );
                let accepted = self.insert_candidate(
                    hit.triple,
                    Warrant::Induced,
                    rule.confidence,
                    hit.evidence,
                    explanation,
                    Some(rule.spec.clone()),
                    Some(rule.id.clone()),
                    CandidateOrigin::BatchClosure,
                );
                if accepted {
                    new_candidates += 1;
                    produced += 1;
                }
            }
        }
        if closure_truncated {
            if let Some(episode) = self.current_episode_mut() {
                episode.truncated = true;
                episode
                    .notes
                    .push("本批闭包达到上限，未补全部分不会假装完整".to_string());
            }
        }
        Ok(ConfirmOutcome {
            candidate: candidate.triple,
            rule_id,
            rule_scope,
            learned: candidate.rule.is_some(),
            new_candidates,
            closure_truncated,
        })
    }

    /// 驳回一条候选：写入负例；同一规则被反复驳回时自动隔离。
    pub fn refute(&mut self, selector: &str) -> Result<RefuteOutcome, EpisodeError> {
        let position = self.find_candidate_index(selector)?;
        let active = self.active.ok_or(EpisodeError::NoActiveEpisode)?;
        let candidate = self.episodes[active].candidates[position].clone();
        if candidate.status != CandidateStatus::Pending {
            return Err(EpisodeError::CandidateAlreadyClosed(selector.to_string()));
        }
        let tick = self.tick;
        self.episodes[active].candidates[position].status = CandidateStatus::Refuted;
        self.refuted_triples.insert(candidate.triple.clone());
        let mut quarantined = false;
        let mut rule_id = candidate.rule_id.clone();
        if rule_id.is_none() {
            rule_id = candidate.rule.as_ref().map(RuleSpec::id);
        }
        if let Some(id) = &rule_id {
            if let Some(rule) = self.rules.iter_mut().find(|rule| &rule.id == id) {
                quarantined = rule.note_refutation(&candidate.triple, tick);
            }
        }
        let mut removed_pending = 0usize;
        if quarantined {
            let rule_id = rule_id.clone();
            if let Some(episode) = self.episodes.get_mut(active) {
                let before = episode.candidates.len();
                episode.candidates.retain(|candidate| {
                    !(candidate.status == CandidateStatus::Pending
                        && candidate.rule_id.as_deref() == rule_id.as_deref())
                });
                removed_pending = before - episode.candidates.len();
            }
        }
        let rule_status = rule_id
            .as_deref()
            .and_then(|id| self.rule(id))
            .map(|rule| rule.status);
        Ok(RefuteOutcome {
            candidate: candidate.triple,
            rule_id,
            rule_status,
            quarantined,
            removed_pending,
        })
    }

    /// 显式晋升规则作用域。晋升是用户动作，但仍拒绝已隔离/退役的规则。
    pub fn promote(
        &mut self,
        rule_id: &str,
        target: RuleScope,
    ) -> Result<PromotionOutcome, EpisodeError> {
        let tick = self.tick;
        let rule = self
            .rules
            .iter_mut()
            .find(|rule| rule.id == rule_id)
            .ok_or_else(|| EpisodeError::RuleNotFound(rule_id.to_string()))?;
        if !rule.is_usable() {
            return Err(EpisodeError::RuleUnavailable(rule_id.to_string()));
        }
        if target.rank() <= rule.scope.rank() {
            return Err(EpisodeError::PromotionNotHigher {
                from: rule.scope,
                to: target,
            });
        }
        if target == RuleScope::Global && rule.refutations > 0 {
            return Err(EpisodeError::PromotionBlocked(format!(
                "全局晋升要求零反例，但该规则已有 {} 条驳回",
                rule.refutations
            )));
        }
        let from = rule.scope;
        let recommended = target.recommended_episodes();
        let warning = if rule.episodes.len() < recommended {
            Some(format!(
                "证据偏薄：只在 {} 个回合确认，建议至少 {} 个",
                rule.episodes.len(),
                recommended
            ))
        } else {
            None
        };
        rule.scope = target;
        rule.promoted_tick = Some(tick);
        rule.updated_tick = tick;
        if rule.confirmations >= 2 {
            rule.status = RuleStatus::Active;
        }
        Ok(PromotionOutcome {
            rule_id: rule.id.clone(),
            from,
            to: target,
            warning,
        })
    }

    /// 晋升到命名空间/全局的规则才进入全局归结与软逻辑；本批规则留在回合内。
    pub fn reasoning_clauses(&self) -> Vec<Clause> {
        self.rules
            .iter()
            .filter(|rule| {
                rule.is_usable()
                    && matches!(rule.scope, RuleScope::Namespace | RuleScope::Global)
            })
            .filter_map(|rule| {
                rule.spec.to_clause(
                    rule.confidence,
                    &format!(
                        "补全回合规则 {}（{}）",
                        rule.spec.rule_text(),
                        rule.scope.label()
                    ),
                )
            })
            .collect()
    }

    fn upsert_rule(
        &mut self,
        spec: RuleSpec,
        example: &Triple,
        episode_id: &str,
        confidence: f64,
        tick: u64,
    ) -> String {
        let id = spec.id();
        if let Some(rule) = self.rules.iter_mut().find(|rule| rule.id == id) {
            rule.note_confirmation(episode_id, example, tick, confidence);
            return id;
        }
        if self.rules.len() >= self.max_rules {
            if let Some(position) = self
                .rules
                .iter()
                .position(|rule| matches!(rule.status, RuleStatus::Quarantined | RuleStatus::Retired))
            {
                self.rules.remove(position);
            }
        }
        self.rules.push(LearnedRule {
            id: id.clone(),
            spec,
            scope: RuleScope::Batch,
            status: RuleStatus::Candidate,
            confirmations: 1,
            refutations: 0,
            episodes: BTreeSet::from([episode_id.to_string()]),
            positive_examples: vec![example.clone()],
            negative_examples: Vec::new(),
            confidence: confidence.clamp(0.0, 1.0),
            created_tick: tick,
            updated_tick: tick,
            source_episode: episode_id.to_string(),
            promoted_tick: None,
        });
        if self.rules.len() > self.max_rules {
            self.rules.remove(0);
        }
        id
    }

    fn scope_filter(
        &self,
        scope: RuleScope,
        active: usize,
        facts: &[Triple],
    ) -> ScopeFilter {
        let Some(episode) = self.episodes.get(active) else {
            return ScopeFilter::Entities(HashSet::new());
        };
        match scope {
            RuleScope::Example => ScopeFilter::Entities(HashSet::from([episode.focus.clone()])),
            RuleScope::Batch => {
                if episode.batch_entities.is_empty() {
                    ScopeFilter::Entities(HashSet::from([episode.focus.clone()]))
                } else {
                    ScopeFilter::Entities(
                        episode.batch_entities.iter().cloned().collect::<HashSet<_>>(),
                    )
                }
            }
            RuleScope::Namespace => {
                if episode.namespace.trim().is_empty() {
                    // 无显式命名空间时退回本批，绝不意外放大到全图。
                    return self.scope_filter(RuleScope::Batch, active, facts);
                }
                let entities = facts
                    .iter()
                    .flat_map(|triple| [triple.subject.as_str(), triple.object.as_str()])
                    .filter(|entity| namespace_of(entity) == episode.namespace)
                    .map(str::to_string)
                    .collect();
                ScopeFilter::Entities(entities)
            }
            RuleScope::Global => ScopeFilter::All,
        }
    }

    pub fn render_episode(&self) -> String {
        let Some(episode) = self.current_episode() else {
            return "还没有打开的补全回合。用「补全回合：<实体>」开始一次学习回合。"
                .to_string();
        };
        let mut output = format!(
            "补全回合 {}（焦点「{}」，候选 {}，待确认 {}，已确认 {}，已驳回 {}）\n",
            short_id(&episode.id),
            episode.focus,
            episode.candidates.len(),
            episode.pending_count(),
            episode.confirmed_count(),
            episode.refuted_count()
        );
        if !episode.batch_entities.is_empty() {
            let shown: Vec<&str> = episode
                .batch_entities
                .iter()
                .take(8)
                .map(String::as_str)
                .collect();
            output.push_str(&format!(
                "  本批实体 {} 个：{}{}\n",
                episode.batch_entities.len(),
                shown.join("、"),
                if episode.batch_entities.len() > shown.len() {
                    "……"
                } else {
                    ""
                }
            ));
        }
        if episode.candidates.is_empty() {
            output.push_str("  （没有候选：可先导入更多关系，或降低链接预测阈值）\n");
        }
        for (index, candidate) in episode.candidates.iter().enumerate() {
            let marker = match candidate.status {
                CandidateStatus::Pending => format!(
                    "[{} · {}]",
                    candidate.warrant.label(),
                    candidate.warrant.hedge().trim_matches(['（', '）'])
                ),
                CandidateStatus::Confirmed => "[已确认]".to_string(),
                CandidateStatus::Refuted => "[已驳回]".to_string(),
            };
            output.push_str(&format!(
                "  {}. {} {}（{}，来源：{}，置信度 {:.2}）\n",
                index + 1,
                marker,
                candidate.display_triple(),
                candidate.status.label(),
                candidate.origin.label(),
                candidate.confidence
            ));
            if !candidate.explanation.trim().is_empty() {
                output.push_str(&format!("     {}\n", candidate.explanation.trim()));
            }
        }
        for note in &episode.notes {
            output.push_str(&format!("  注：{}\n", note));
        }
        if episode.truncated {
            output.push_str("  本批结果已截断：未补全部分不等于不存在。\n");
        }
        output.push_str(
            "  操作：确认补全：<序号> / 否定补全：<序号> / 补全规则 / 晋升补全规则：<规则id>|<作用域>",
        );
        output.trim_end().to_string()
    }

    pub fn render_rules(&self) -> String {
        if self.rules.is_empty() {
            return "还没有归纳出的补全规则。先打开回合并确认一条候选。".to_string();
        }
        let mut output = format!("补全规则（{} 条，默认本批作用域）\n", self.rules.len());
        for (index, rule) in self.rules.iter().enumerate() {
            output.push_str(&format!(
                "  {}. {} id={}\n",
                index + 1,
                rule.display_line(),
                short_id(&rule.id)
            ));
            if !rule.positive_examples.is_empty() {
                let examples: Vec<String> = rule
                    .positive_examples
                    .iter()
                    .take(3)
                    .map(display_triple)
                    .collect();
                output.push_str(&format!("     正例：{}\n", examples.join("；")));
            }
            if !rule.negative_examples.is_empty() {
                let examples: Vec<String> = rule
                    .negative_examples
                    .iter()
                    .take(3)
                    .map(display_triple)
                    .collect();
                output.push_str(&format!("     负例：{}\n", examples.join("；")));
            }
        }
        output.trim_end().to_string()
    }
}

#[derive(Debug, Clone)]
enum ScopeFilter {
    Entities(HashSet<String>),
    All,
}

impl ScopeFilter {
    fn allows(&self, entity: &str) -> bool {
        match self {
            ScopeFilter::Entities(entities) => entities.contains(entity),
            ScopeFilter::All => true,
        }
    }
}

#[derive(Debug, Clone)]
struct ClosureHit {
    triple: Triple,
    evidence: Vec<Triple>,
}

/// 在一个明确的作用域内做规则闭包；返回（新三元组, 是否因上限截断）。
fn apply_rule(
    spec: &RuleSpec,
    facts: &[Triple],
    filter: &ScopeFilter,
    max: usize,
) -> (Vec<ClosureHit>, bool) {
    if !spec.is_valid() || max == 0 {
        return (Vec::new(), false);
    }
    let mut out: HashMap<&str, HashMap<&str, HashSet<&str>>> = HashMap::new();
    let mut into: HashMap<&str, HashMap<&str, HashSet<&str>>> = HashMap::new();
    let mut pairs: HashMap<&str, Vec<(&str, &str)>> = HashMap::new();
    for triple in facts {
        let subject = triple.subject.as_str();
        let object = triple.object.as_str();
        let predicate = triple.predicate.as_str();
        out.entry(predicate)
            .or_default()
            .entry(subject)
            .or_default()
            .insert(object);
        into.entry(predicate)
            .or_default()
            .entry(object)
            .or_default()
            .insert(subject);
        pairs
            .entry(predicate)
            .or_default()
            .push((subject, object));
    }

    let max_scan = max.saturating_mul(CLOSURE_SCAN_FACTOR);
    let mut scanned = 0usize;
    let mut truncated = false;
    let mut hits: BTreeMap<Triple, Vec<Triple>> = BTreeMap::new();
    let already_exists = |subject: &str, object: &str| {
        out.get(spec.head.as_str())
            .and_then(|by_subject| by_subject.get(subject))
            .map_or(false, |objects| objects.contains(object))
    };
    let push = |triple: Triple, evidence: Vec<Triple>, hits: &mut BTreeMap<Triple, Vec<Triple>>| {
        match hits.get_mut(&triple) {
            Some(current) if &evidence < current => {
                *current = evidence;
            }
            Some(_) => {}
            None => {
                hits.insert(triple, evidence);
            }
        }
    };

    match spec.pattern {
        RulePattern::Compose => {
            let Some(first) = out.get(spec.body[0].as_str()) else {
                return (Vec::new(), false);
            };
            let Some(second) = out.get(spec.body[1].as_str()) else {
                return (Vec::new(), false);
            };
            'outer: for (subject, middle_entities) in first {
                if !filter.allows(subject) {
                    continue;
                }
                for middle in middle_entities {
                    let Some(objects) = second.get(*middle) else {
                        continue;
                    };
                    for object in objects {
                        scanned += 1;
                        if scanned > max_scan {
                            truncated = true;
                            break 'outer;
                        }
                        if subject == object || already_exists(subject, object) {
                            continue;
                        }
                        push(
                            Triple::new(subject, &spec.head, object),
                            vec![
                                Triple::new(subject, &spec.body[0], middle),
                                Triple::new(middle, &spec.body[1], object),
                            ],
                            &mut hits,
                        );
                        if hits.len() >= max {
                            truncated = true;
                            break 'outer;
                        }
                    }
                }
            }
        }
        RulePattern::Inverse => {
            let Some(body_pairs) = pairs.get(spec.body[0].as_str()) else {
                return (Vec::new(), false);
            };
            for (subject, object) in body_pairs {
                scanned += 1;
                if scanned > max_scan {
                    truncated = true;
                    break;
                }
                if subject == object
                    || !filter.allows(object)
                    || already_exists(object, subject)
                {
                    continue;
                }
                push(
                    Triple::new(object, &spec.head, subject),
                    vec![Triple::new(subject, &spec.body[0], object)],
                    &mut hits,
                );
                if hits.len() >= max {
                    truncated = true;
                    break;
                }
            }
        }
        RulePattern::CommonNeighbor => {
            let Some(first) = into.get(spec.body[0].as_str()) else {
                return (Vec::new(), false);
            };
            let Some(second) = into.get(spec.body[1].as_str()) else {
                return (Vec::new(), false);
            };
            'outer: for (anchor, subjects) in first {
                let Some(others) = second.get(*anchor) else {
                    continue;
                };
                for subject in subjects {
                    if !filter.allows(subject) {
                        continue;
                    }
                    for other in others {
                        scanned += 1;
                        if scanned > max_scan {
                            truncated = true;
                            break 'outer;
                        }
                        if subject == other || already_exists(subject, other) {
                            continue;
                        }
                        push(
                            Triple::new(subject, &spec.head, other),
                            vec![
                                Triple::new(subject, &spec.body[0], anchor),
                                Triple::new(other, &spec.body[1], anchor),
                            ],
                            &mut hits,
                        );
                        if hits.len() >= max {
                            truncated = true;
                            break 'outer;
                        }
                    }
                }
            }
        }
        RulePattern::CommonObject => {
            let Some(first) = out.get(spec.body[0].as_str()) else {
                return (Vec::new(), false);
            };
            let Some(second) = out.get(spec.body[1].as_str()) else {
                return (Vec::new(), false);
            };
            'outer: for (anchor, subjects) in first {
                let Some(others) = second.get(*anchor) else {
                    continue;
                };
                for subject in subjects {
                    if !filter.allows(subject) {
                        continue;
                    }
                    for other in others {
                        scanned += 1;
                        if scanned > max_scan {
                            truncated = true;
                            break 'outer;
                        }
                        if subject == other || already_exists(subject, other) {
                            continue;
                        }
                        push(
                            Triple::new(subject, &spec.head, other),
                            vec![
                                Triple::new(anchor, &spec.body[0], subject),
                                Triple::new(anchor, &spec.body[1], other),
                            ],
                            &mut hits,
                        );
                        if hits.len() >= max {
                            truncated = true;
                            break 'outer;
                        }
                    }
                }
            }
        }
    }

    (
        hits.into_iter()
            .map(|(triple, evidence)| ClosureHit { triple, evidence })
            .collect(),
        truncated,
    )
}

fn merge_evidence(current: &mut Vec<Triple>, incoming: Vec<Triple>) {
    for triple in incoming {
        if !current.contains(&triple) {
            current.push(triple);
        }
    }
    current.sort();
}

fn parse_triple_selector(selector: &str) -> Option<(String, String, String)> {
    if let Some((subject, rest)) = selector.split_once('|') {
        if let Some((predicate, object)) = rest.split_once('|') {
            let subject = subject.trim();
            let predicate = predicate.trim();
            let object = object.trim();
            if !subject.is_empty() && !predicate.is_empty() && !object.is_empty() {
                return Some((
                    subject.to_string(),
                    predicate.to_string(),
                    object.to_string(),
                ));
            }
        }
    }
    let parts: Vec<&str> = selector.split_whitespace().collect();
    if parts.len() == 3 && parts.iter().all(|part| !part.is_empty()) {
        return Some((
            parts[0].to_string(),
            parts[1].to_string(),
            parts[2].to_string(),
        ));
    }
    None
}

/// 命名空间前缀：取最后一个 `:`、`/` 或 `#` 之前的部分；无分隔符则为空。
fn namespace_of(entity: &str) -> String {
    let entity = entity.trim();
    match entity.rfind([':', '/', '#']) {
        Some(index) if index > 0 => entity[..=index].to_string(),
        _ => String::new(),
    }
}

fn display_triple(triple: &Triple) -> String {
    format!(
        "{}({}, {})",
        triple.predicate, triple.subject, triple.object
    )
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// 稳定短 id：FNV-1a + 输入长度，避免同前缀截断歧义。
fn stable_id(prefix: &str, canonical: &str) -> String {
    format!(
        "{}{:016x}{:x}",
        prefix,
        fnv1a64(canonical.as_bytes()),
        canonical.len()
    )
}

fn short_id(id: &str) -> &str {
    id.get(..id.len().min(21)).unwrap_or(id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::link_prediction::LinkPredictor;

    fn chain_predictor() -> LinkPredictor {
        let mut predictor = LinkPredictor::new();
        predictor.add_triple("张三", "师傅", "李四");
        predictor.add_triple("李四", "师傅", "王五");
        predictor.add_triple("王五", "师傅", "赵六");
        predictor.add_triple("张三", "师祖", "王五");
        predictor.add_triple("孙八", "师傅", "周九");
        predictor.add_triple("周九", "师傅", "吴十");
        predictor.add_triple("郑十一", "师傅", "钱十二");
        predictor.add_triple("钱十二", "师傅", "孙十三");
        predictor.add_triple("甲", "师傅", "乙");
        predictor.add_triple("乙", "师傅", "丙");
        predictor.add_triple("丁", "师傅", "戊");
        predictor.add_triple("戊", "师傅", "己");
        predictor.mine_rules(1, 0.1);
        predictor
    }

    fn open_with_predictions(
        ledger: &mut CompletionLedger,
        predictor: &LinkPredictor,
        entity: &str,
    ) {
        ledger.open_episode(entity, None);
        let predictions = predictor.predict_for_entity(entity, None, 8);
        for prediction in &predictions {
            let rule = predictor.rules().get(prediction.rule_index);
            let explanation = predictor.explain(prediction);
            ledger.add_prediction(prediction, rule, explanation);
        }
    }

    #[test]
    fn confirm_learns_batch_rule_and_closes_more_entities() {
        let predictor = chain_predictor();
        let facts = predictor.triples().to_vec();
        let mut ledger = CompletionLedger::new();
        open_with_predictions(&mut ledger, &predictor, "孙八");
        ledger.set_batch_scope(
            ["孙八", "周九", "吴十", "郑十一", "钱十二", "孙十三", "甲", "乙", "丙"]
                .into_iter()
                .map(str::to_string),
        );
        let episode = ledger.current_episode().expect("回合应打开");
        assert!(
            episode
                .candidates
                .iter()
                .any(|candidate| candidate.triple == Triple::new("孙八", "师祖", "吴十")),
            "孙八的师祖候选应存在：{:?}",
            episode
                .candidates
                .iter()
                .map(|c| c.display_triple())
                .collect::<Vec<_>>()
        );
        let outcome = ledger
            .confirm("孙八|师祖|吴十", &facts)
            .expect("确认应成功");
        assert!(outcome.learned);
        assert_eq!(outcome.rule_scope, Some(RuleScope::Batch));
        assert_eq!(ledger.rules().len(), 1);
        let rule = &ledger.rules()[0];
        assert_eq!(rule.scope, RuleScope::Batch);
        assert_eq!(rule.confirmations, 1);
        assert!(outcome.new_candidates >= 1, "批次闭包应产生新候选");
        let episode = ledger.current_episode().unwrap();
        let learned = episode
            .candidates
            .iter()
            .find(|candidate| candidate.triple == Triple::new("郑十一", "师祖", "孙十三"))
            .expect("批次内其他实体应被补全");
        assert_eq!(learned.origin, CandidateOrigin::BatchClosure);
        assert_eq!(learned.warrant, Warrant::Induced);
        assert!(learned.explanation.contains("本批"));
        assert!(ledger.reasoning_clauses().is_empty(), "本批规则不应进入全局归结");
    }

    #[test]
    fn repeated_refutation_quarantines_rule_and_removes_pending() {
        let predictor = chain_predictor();
        let facts = predictor.triples().to_vec();
        let mut ledger = CompletionLedger::new();
        open_with_predictions(&mut ledger, &predictor, "孙八");
        ledger.set_batch_scope(
            [
                "孙八", "郑十一", "钱十二", "孙十三", "甲", "乙", "丙", "丁", "戊", "己",
            ]
            .into_iter()
            .map(str::to_string),
        );
        ledger
            .confirm("孙八|师祖|吴十", &facts)
            .expect("确认应成功");
        let first_closure = ledger
            .current_episode()
            .unwrap()
            .candidates
            .iter()
            .find(|candidate| {
                candidate.origin == CandidateOrigin::BatchClosure
                    && candidate.triple == Triple::new("郑十一", "师祖", "孙十三")
            })
            .expect("应有闭包候选")
            .id
            .clone();
        let second_closure = ledger
            .current_episode()
            .unwrap()
            .candidates
            .iter()
            .find(|candidate| {
                candidate.origin == CandidateOrigin::BatchClosure
                    && candidate.triple == Triple::new("甲", "师祖", "丙")
            })
            .expect("应有第二个闭包候选")
            .id
            .clone();
        let first = ledger.refute(&first_closure).expect("第一次驳回应成功");
        assert!(!first.quarantined);
        let second = ledger.refute(&second_closure).expect("第二次驳回应成功");
        assert!(second.quarantined);
        assert_eq!(second.rule_status, Some(RuleStatus::Quarantined));
        assert!(second.removed_pending > 0, "隔离后应清理同一规则的待定候选");
        assert!(ledger.rules()[0]
            .negative_examples
            .contains(&Triple::new("郑十一", "师祖", "孙十三")));
    }

    #[test]
    fn refuted_triple_is_not_reproposed_in_next_episode() {
        let predictor = chain_predictor();
        let mut ledger = CompletionLedger::new();
        open_with_predictions(&mut ledger, &predictor, "孙八");
        ledger
            .refute("孙八|师祖|吴十")
            .expect("驳回应成功");
        open_with_predictions(&mut ledger, &predictor, "孙八");
        assert!(
            !ledger
                .current_episode()
                .unwrap()
                .candidates
                .iter()
                .any(|candidate| candidate.triple == Triple::new("孙八", "师祖", "吴十")),
            "已驳回的候选不应在新回合重新出现"
        );
    }

    #[test]
    fn promotion_expands_scope_and_feeds_global_reasoning() {
        let predictor = chain_predictor();
        let facts = predictor.triples().to_vec();
        let mut ledger = CompletionLedger::new();
        open_with_predictions(&mut ledger, &predictor, "孙八");
        ledger.set_batch_scope(["孙八", "周九", "吴十"].into_iter().map(str::to_string));
        ledger
            .confirm("孙八|师祖|吴十", &facts)
            .expect("第一次确认应成功");
        let rule_id = ledger.rules()[0].id.clone();
        open_with_predictions(&mut ledger, &predictor, "郑十一");
        ledger.set_batch_scope(
            ["郑十一", "钱十二", "孙十三"]
                .into_iter()
                .map(str::to_string),
        );
        ledger
            .confirm("郑十一|师祖|孙十三", &facts)
            .expect("第二次确认应成功");
        let rule = ledger.rule(&rule_id).unwrap();
        assert_eq!(rule.confirmations, 2);
        assert_eq!(rule.status, RuleStatus::Active);
        let outcome = ledger
            .promote(&rule_id, RuleScope::Namespace)
            .expect("晋升应成功");
        assert_eq!(outcome.from, RuleScope::Batch);
        assert_eq!(outcome.to, RuleScope::Namespace);
        assert!(outcome.warning.is_none());
        assert_eq!(ledger.rule(&rule_id).unwrap().scope, RuleScope::Namespace);
        assert_eq!(ledger.reasoning_clauses().len(), 1);
        assert!(ledger.promote(&rule_id, RuleScope::Batch).is_err());
    }

    #[test]
    fn analogical_candidates_are_hedged_and_clamped() {
        let mut ledger = CompletionLedger::new();
        ledger.open_episode("赵六", Some("赵六的国籍是什么"));
        assert!(ledger.add_analogical(
            Triple::new("赵六", "国籍", "中国"),
            0.55,
            vec![Triple::new("张三", "国籍", "中国")],
            "同类实体都有国籍".to_string(),
        ));
        assert!(!ledger.add_analogical(
            Triple::new("赵六", "国籍", "中国"),
            0.8,
            vec![Triple::new("李四", "国籍", "中国")],
            "另一个同类实体也有国籍".to_string(),
        ));
        let episode = ledger.current_episode().unwrap();
        assert_eq!(episode.candidates.len(), 1, "同一三元组不应重复入账");
        let candidate = &episode.candidates[0];
        assert_eq!(candidate.warrant, Warrant::Analogical);
        assert_eq!(candidate.warrant.hedge(), "（类比，仅提示）");
        assert!(candidate.confidence >= 0.8);
        let rendered = ledger.render_episode();
        assert!(rendered.contains("类比"));
        assert!(rendered.contains("赵六"));
    }

    #[test]
    fn closure_respects_existing_facts_and_candidate_limit() {
        let facts = vec![
            Triple::new("a", "r", "b"),
            Triple::new("b", "r", "c"),
            Triple::new("a", "s", "c"),
            Triple::new("d", "r", "e"),
            Triple::new("e", "r", "f"),
        ];
        let spec = RuleSpec {
            pattern: RulePattern::Compose,
            body: vec!["r".to_string(), "r".to_string()],
            head: "s".to_string(),
        };
        let filter = ScopeFilter::All;
        let (hits, truncated) = apply_rule(&spec, &facts, &filter, 8);
        assert!(!truncated);
        assert_eq!(
            hits.iter().map(|hit| hit.triple.clone()).collect::<Vec<_>>(),
            vec![Triple::new("d", "s", "f")],
            "已知的 a-s-c 不应重复生成"
        );
        let (limited, truncated) = apply_rule(&spec, &facts, &filter, 0);
        assert!(limited.is_empty() && !truncated);
    }

    #[test]
    fn clamp_limits_rejects_untrusted_unbounded_snapshot() {
        let mut ledger =
            CompletionLedger::with_limits(usize::MAX, usize::MAX, usize::MAX, usize::MAX);
        ledger.open_episode("甲", None);
        ledger.active = Some(999);
        ledger.clamp_limits();
        assert_eq!(ledger.max_candidates, DEFAULT_MAX_EPISODE_CANDIDATES);
        assert_eq!(ledger.max_closure, DEFAULT_MAX_CLOSURE_CANDIDATES);
        assert_eq!(ledger.max_episodes, DEFAULT_MAX_EPISODES);
        assert_eq!(ledger.max_rules, DEFAULT_MAX_RULES);
        assert_eq!(ledger.max_batch_entities, DEFAULT_MAX_BATCH_ENTITIES);
        assert_eq!(ledger.active, None, "越界活动索引应被清掉");
    }
}

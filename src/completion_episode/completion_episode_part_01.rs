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
/// 运行期裁决测试集的数量上限。
pub const DEFAULT_MAX_TESTS: usize = 256;
/// 低于该效用的候选不值得打扰用户。
pub const QUESTION_MIN_UTILITY: f64 = 12.0;
/// 每重复提问一次扣减的效用。
pub const QUESTION_REPEAT_PENALTY: f64 = 18.0;
/// 同一候选最多主动追问的次数。
pub const QUESTION_MAX_REPEATS: u64 = 2;
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

    /// 用户更正了此前的否决：撤回对应负例，并在证据允许时解除隔离。
    fn note_reinstatement(&mut self, example: &Triple, tick: u64) {
        self.negative_examples.retain(|triple| triple != example);
        self.refutations = self.refutations.saturating_sub(1);
        self.updated_tick = tick;
        if self.status == RuleStatus::Quarantined
            && (self.refutations < 2 || self.refutations < self.confirmations)
        {
            self.status = if self.confirmations >= 2 {
                RuleStatus::Active
            } else {
                RuleStatus::Candidate
            };
        }
    }
}
/// 用户裁决的方向。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RegressionVerdict {
    Confirmed,
    Refuted,
}
impl RegressionVerdict {
    pub fn label(self) -> &'static str {
        match self {
            RegressionVerdict::Confirmed => "确认",
            RegressionVerdict::Refuted => "驳回",
        }
    }
}
/// 一次可重放的运行期知识测试。
///
/// 测试不是开发期回归，而是用户裁决留下的「错误记忆」：规则晋升、
/// 恢复或导入时重放，防止系统重新提出已被否决的候选，也防止规则
/// 悄悄丢掉已经确认的依据。
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CompletionTest {
    pub id: String,
    pub triple: Triple,
    pub verdict: RegressionVerdict,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rule: Option<RuleSpec>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence: Vec<Triple>,
    pub source: String,
    pub source_id: String,
    pub created_tick: u64,
}
impl CompletionTest {
    pub fn display_line(&self) -> String {
        format!(
            "[{}] {}({}, {})",
            self.verdict.label(), self.triple.predicate, self.triple.subject, self.triple.object
        )
    }
}
/// 提问形式：是非、二选一，或没有候选时的开放问。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuestionKind {
    Confirm,
    Choice,
    Open,
}
impl QuestionKind {
    pub fn label(self) -> &'static str {
        match self {
            QuestionKind::Confirm => "是非",
            QuestionKind::Choice => "选择",
            QuestionKind::Open => "开放",
        }
    }
}
/// 一次经过效用排序的提问。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompletionQuestion {
    pub candidate_id: String,
    pub triple: Triple,
    pub kind: QuestionKind,
    pub utility: f64,
    pub text: String,
    pub alternatives: Vec<String>,
}
/// 一次重放中的失败项。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegressionFailure {
    pub rule_id: String,
    pub test_id: String,
    pub triple: Triple,
    pub reason: String,
}
/// 规则重放结果。
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct RegressionReport {
    pub checked: usize,
    pub failures: Vec<RegressionFailure>,
}
impl RegressionReport {
    pub fn passed(&self) -> bool {
        self.failures.is_empty()
    }

    pub fn summary(&self) -> String {
        if self.passed() {
            return format!("运行期测试 {} 条全部通过", self.checked);
        }
        let details = self
            .failures
            .iter()
            .take(3)
            .map(|failure| format!("{}：{}", failure.triple.object, failure.reason))
            .collect::<Vec<_>>()
            .join("；");
        format!(
            "运行期测试 {} 条中有 {} 条失败（{}）",
            self.checked,
            self.failures.len(),
            details
        )
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
/// 规则退役 / 恢复的结果：保留前后状态，便于回执审计。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuleLifecycleOutcome {
    pub rule_id: String,
    pub scope: RuleScope,
    pub previous: RuleStatus,
    pub current: RuleStatus,
}
/// 回合操作错误。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EpisodeError {
    NoActiveEpisode,
    CandidateNotFound(String),
    AmbiguousSelector { selector: String, count: usize },
    CandidateAlreadyClosed(String),
    RuleNotFound(String),
    AmbiguousRuleSelector { selector: String, count: usize },
    RuleUnavailable(String),
    RuleAlreadyRetired(String),
    RuleNotRetired(String),
    PromotionNotHigher {
        from: RuleScope,
        to: RuleScope,
    },
    PromotionBlocked(String),
    RegressionBlocked(String),
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
            EpisodeError::AmbiguousSelector { selector, count } => write!(
                f,
                "候选前缀「{}」匹配 {} 条，请改用序号或更长的候选 id",
                selector, count
            ),
            EpisodeError::CandidateAlreadyClosed(selector) => {
                write!(f, "候选「{}」已有确认/驳回结论，不能再次裁定", selector)
            }
            EpisodeError::RuleNotFound(id) => write!(f, "找不到补全规则「{}」", id),
            EpisodeError::AmbiguousRuleSelector { selector, count } => write!(
                f,
                "规则前缀「{}」匹配 {} 条，请改用更长的规则 id",
                selector, count
            ),
            EpisodeError::RuleUnavailable(id) => {
                write!(f, "补全规则「{}」已隔离或退役，不能继续使用", id)
            }
            EpisodeError::RuleAlreadyRetired(id) => {
                write!(f, "补全规则「{}」已经退役", id)
            }
            EpisodeError::RuleNotRetired(id) => {
                write!(f, "补全规则「{}」未退役，无需恢复", id)
            }
            EpisodeError::PromotionNotHigher { from, to } => write!(
                f,
                "规则作用域只能扩大：当前是「{}」，不能降为「{}」",
                from.label(),
                to.label()
            ),
            EpisodeError::PromotionBlocked(reason) => write!(f, "规则晋升被拒绝：{}", reason),
            EpisodeError::RegressionBlocked(reason) => {
                write!(f, "运行期知识回归未通过：{}", reason)
            }
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
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    confirmed_triples: BTreeSet<Triple>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    regression_tests: Vec<CompletionTest>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    asked_questions: BTreeMap<String, u64>,
    #[serde(default)]
    question_muted: bool,
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

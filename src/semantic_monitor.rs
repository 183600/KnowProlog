use std::collections::{BTreeMap, BTreeSet, VecDeque};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::context_image::fnv1a_hex;
use crate::contextual_rag::{AssemblyReceipt, EvidenceAssembly};
use crate::dense_retrieval::EvidenceSearch;
use crate::rewrite_retrieval::{Assessment, Decision};

pub const DEFAULT_WINDOW: usize = 64;
pub const MIN_ALERT_SAMPLE: u64 = 8;
pub const MIN_ASSESSMENT_SAMPLE: u64 = 5;
pub const LOW_MARGIN_BPS: u32 = 100;
pub const ZERO_RECALL_ALERT_BPS: u32 = 300;
pub const LOW_RECALL_ALERT_BPS: u32 = 500;
pub const LOW_CERTIFICATE_ALERT_BPS: u32 = 5000;
pub const BUDGET_PRESSURE_ALERT_BPS: u32 = 250;
pub const LOW_MARGIN_ALERT_BPS: u32 = 300;
pub const CLARIFY_HEAVY_ALERT_BPS: u32 = 400;
pub const ABSTAIN_HEAVY_ALERT_BPS: u32 = 400;
pub const ASSEMBLY_EMPTY_ALERT_BPS: u32 = 300;
pub const DEFERRAL_ALERT_BPS: u32 = 300;
pub const NO_RETRIEVAL_ALERT_BPS: u32 = 100;
pub const CAUTIOUS_MAX_ITEMS: usize = 4;
pub const STRICT_MAX_ITEMS: usize = 3;

pub const CODE_ANSWERED: &str = "answered";
pub const CODE_DEFERRED: &str = "deferred";
pub const CODE_FAULT: &str = "fault";
pub const CODE_CLARIFY: &str = "clarify";
pub const CODE_ABSTAIN: &str = "abstain";
pub const CODE_LOW_MARGIN: &str = "low_margin";
pub const CODE_ZERO_RECALL: &str = "zero_recall";
pub const CODE_UNCERTIFIED: &str = "uncertified_top_k";
pub const CODE_BUDGET_EXHAUSTED: &str = "budget_exhausted";
pub const CODE_INDEX_EMPTY: &str = "index_empty";
pub const CODE_NO_RETRIEVAL: &str = "no_retrieval";
pub const CODE_ASSEMBLY_EMPTY: &str = "assembly_empty";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnOutcome {
    Answered,
    Deferred,
    Clarified,
    Fault,
}

impl TurnOutcome {
    pub fn code(self) -> &'static str {
        match self {
            Self::Answered => CODE_ANSWERED,
            Self::Deferred => CODE_DEFERRED,
            Self::Clarified => CODE_CLARIFY,
            Self::Fault => CODE_FAULT,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControlMode {
    #[default]
    Observe,
    Control,
}

impl ControlMode {
    pub fn label(self) -> &'static str {
        match self {
            Self::Observe => "observe",
            Self::Control => "control",
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PressureLevel {
    #[default]
    Healthy,
    Cautious,
    Strict,
}

impl PressureLevel {
    pub fn label(self) -> &'static str {
        match self {
            Self::Healthy => "healthy",
            Self::Cautious => "cautious",
            Self::Strict => "strict",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RepairAction {
    AcquireAlias,
    AcquireVocabulary,
    AcquireFact,
    RaiseRetrievalDepth,
    RaiseScanBudget,
    PreferClarification,
    WidenEvidence,
    RebuildIndex,
    InspectRouting,
    InspectEngine,
}

impl RepairAction {
    pub fn label(self) -> &'static str {
        match self {
            Self::AcquireAlias => "补登记实体别名",
            Self::AcquireVocabulary => "补词表或谓词词面",
            Self::AcquireFact => "补充图谱事实",
            Self::RaiseRetrievalDepth => "提高检索深度",
            Self::RaiseScanBudget => "提高扫描预算",
            Self::PreferClarification => "优先请求澄清",
            Self::WidenEvidence => "放宽证据装配范围",
            Self::RebuildIndex => "重建证据索引",
            Self::InspectRouting => "检查检索路由",
            Self::InspectEngine => "检查推理引擎",
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Ratio {
    pub numerator: u64,
    pub denominator: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bps: Option<u32>,
}

impl Ratio {
    pub fn new(numerator: u64, denominator: u64) -> Self {
        let bps = (denominator > 0).then(|| {
            let scaled = (u128::from(numerator) * 10_000 + u128::from(denominator) / 2)
                / u128::from(denominator);
            scaled.min(10_000) as u32
        });
        Self {
            numerator,
            denominator,
            bps,
        }
    }

    pub fn measured(self) -> bool {
        self.denominator > 0
    }

    pub fn as_fraction(self) -> Option<f64> {
        self.measured()
            .then(|| self.numerator as f64 / self.denominator as f64)
    }

    pub fn merge(&mut self, other: Self) {
        *self = Self::new(
            self.numerator.saturating_add(other.numerator),
            self.denominator.saturating_add(other.denominator),
        );
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchFold {
    pub searches: u64,
    pub hit_searches: u64,
    pub returned: u64,
    pub docs: usize,
    pub query_terms: usize,
    pub candidates: u64,
    pub scanned: u64,
    pub constraint_pruned: u64,
    pub bound_pruned: u64,
    pub omitted: u64,
    pub truncated_searches: u64,
    pub budget_exhausted_searches: u64,
    pub certified_searches: u64,
    pub empty_searches: u64,
    pub index_empty_searches: u64,
    pub max_returned: usize,
}

impl SearchFold {
    fn observe(&mut self, search: &EvidenceSearch) {
        let receipt = &search.receipt;
        let returned = search.hits.len();
        self.searches += 1;
        self.returned += returned as u64;
        self.hit_searches += u64::from(returned > 0);
        self.empty_searches += u64::from(returned == 0);
        self.index_empty_searches += u64::from(receipt.docs == 0);
        self.certified_searches += u64::from(receipt.certified_top_k);
        self.budget_exhausted_searches += u64::from(receipt.budget_exhausted);
        self.truncated_searches += u64::from(
            receipt.omitted_by_truncation > 0 || receipt.budget_exhausted,
        );
        self.docs = self.docs.max(receipt.docs);
        self.query_terms = self.query_terms.max(receipt.query_terms);
        self.max_returned = self.max_returned.max(returned);
        self.candidates += receipt.candidates as u64;
        self.scanned += receipt.scanned as u64;
        self.constraint_pruned += receipt.constraint_pruned as u64;
        self.bound_pruned += receipt.bound_pruned as u64;
        self.omitted += receipt.omitted_by_truncation as u64;
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssemblyFold {
    pub assemblies: u64,
    pub selected_assemblies: u64,
    pub empty_assemblies: u64,
    pub dense_candidates: u64,
    pub holographic_candidates: u64,
    pub merged: u64,
    pub anchor_pruned: u64,
    pub unverified_dropped: u64,
    pub referenced_dropped: u64,
    pub weak_dropped: u64,
    pub supported_assemblies: u64,
    pub certified_assemblies: u64,
}

impl AssemblyFold {
    fn observe(&mut self, receipt: &AssemblyReceipt) {
        self.assemblies += 1;
        self.selected_assemblies += u64::from(receipt.selected > 0);
        self.empty_assemblies += u64::from(receipt.selected == 0);
        self.dense_candidates += receipt.dense_candidates as u64;
        self.holographic_candidates += receipt.holo_candidates as u64;
        self.merged += receipt.merged as u64;
        self.anchor_pruned += receipt.anchor_pruned as u64;
        self.unverified_dropped += receipt.unverified_dropped as u64;
        self.referenced_dropped += receipt.referenced_dropped as u64;
        self.weak_dropped += receipt.weak_dropped as u64;
        let support = receipt.lexical_support
            + receipt.structural_support
            + receipt.predicate_support
            + receipt.holographic_support
            + receipt.contextual_support;
        self.supported_assemblies += u64::from(support > 0);
        self.certified_assemblies += u64::from(receipt.certified);
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssessmentFold {
    pub assessments: u64,
    pub accept: u64,
    pub clarify: u64,
    pub abstain: u64,
    pub low_margin: u64,
    pub selected: u64,
}

impl AssessmentFold {
    fn observe(&mut self, assessment: &Assessment) {
        self.assessments += 1;
        match assessment.decision {
            Decision::Accept => self.accept += 1,
            Decision::Clarify => self.clarify += 1,
            Decision::Abstain => self.abstain += 1,
        }
        self.selected += u64::from(assessment.selected.is_some());
        let top = assessment
            .selected_hypothesis()
            .or_else(|| assessment.hypotheses.first());
        if let Some(top) = top {
            let margin_bps = (top.margin.clamp(0.0, 1.0) * 10_000.0).round() as u32;
            self.low_margin += u64::from(margin_bps < LOW_MARGIN_BPS);
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BudgetConfig {
    pub capacity: i64,
    pub initial: i64,
    pub cautious_below: i64,
    pub strict_below: i64,
    pub cautious_recover: i64,
    pub strict_recover: i64,
    pub success_reward: i64,
    pub grounded_reward: i64,
    pub clarify_cost: i64,
    pub abstain_cost: i64,
    pub deferred_cost: i64,
    pub fault_cost: i64,
    pub zero_recall_cost: i64,
    pub uncertified_cost: i64,
    pub budget_exhausted_cost: i64,
}

impl Default for BudgetConfig {
    fn default() -> Self {
        Self {
            capacity: 100,
            initial: 60,
            cautious_below: 50,
            strict_below: 30,
            cautious_recover: 60,
            strict_recover: 40,
            success_reward: 3,
            grounded_reward: 1,
            clarify_cost: 3,
            abstain_cost: 5,
            deferred_cost: 6,
            fault_cost: 10,
            zero_recall_cost: 2,
            uncertified_cost: 1,
            budget_exhausted_cost: 1,
        }
    }
}

impl BudgetConfig {
    fn clamped(&self) -> Self {
        let capacity = self.capacity.max(1);
        let initial = self.initial.clamp(0, capacity);
        let strict_below = self.strict_below.clamp(0, capacity);
        let cautious_below = self.cautious_below.clamp(strict_below, capacity);
        let strict_recover = self.strict_recover.clamp(strict_below, cautious_below);
        let cautious_recover = self.cautious_recover.clamp(strict_recover, capacity);
        Self {
            capacity,
            initial,
            cautious_below,
            strict_below,
            cautious_recover,
            strict_recover,
            ..*self
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BudgetSnapshot {
    pub mode: ControlMode,
    pub level: PressureLevel,
    pub score: i64,
    pub capacity: i64,
}

impl Default for BudgetSnapshot {
    fn default() -> Self {
        let config = BudgetConfig::default().clamped();
        Self {
            mode: ControlMode::Observe,
            level: PressureLevel::Healthy,
            score: config.initial,
            capacity: config.capacity,
        }
    }
}

impl BudgetSnapshot {
    pub fn fraction_bps(self) -> u32 {
        if self.capacity <= 0 {
            return 0;
        }
        ((i128::from(self.score) * 10_000) / i128::from(self.capacity)).clamp(0, 10_000) as u32
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchAdvice {
    pub level: PressureLevel,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_items: Option<usize>,
    pub prefer_clarification: bool,
}

impl SearchAdvice {
    fn from_budget(budget: &BudgetState, mode: ControlMode) -> Self {
        if mode == ControlMode::Observe {
            return Self {
                level: budget.level,
                max_items: None,
                prefer_clarification: false,
            };
        }
        match budget.level {
            PressureLevel::Healthy => Self {
                level: PressureLevel::Healthy,
                max_items: None,
                prefer_clarification: false,
            },
            PressureLevel::Cautious => Self {
                level: PressureLevel::Cautious,
                max_items: Some(CAUTIOUS_MAX_ITEMS),
                prefer_clarification: true,
            },
            PressureLevel::Strict => Self {
                level: PressureLevel::Strict,
                max_items: Some(STRICT_MAX_ITEMS),
                prefer_clarification: true,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticObservation {
    pub sequence: u64,
    pub question_digest: String,
    pub question_chars: usize,
    pub token_chars: usize,
    pub outcome: TurnOutcome,
    pub search: SearchFold,
    pub assembly: AssemblyFold,
    pub assessment: AssessmentFold,
    pub codes: Vec<String>,
    pub budget: BudgetSnapshot,
    pub advice: SearchAdvice,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Alert {
    pub code: String,
    pub severity: AlertSeverity,
    pub numerator: u64,
    pub denominator: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bps: Option<u32>,
    pub message: String,
    pub repair: RepairAction,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WindowMetrics {
    pub capacity: usize,
    pub observations: u64,
    pub searches: u64,
    pub answered: Ratio,
    pub deferred: Ratio,
    pub faults: Ratio,
    pub retrieval_share: Ratio,
    pub retrieval_recall: Ratio,
    pub search_recall: Ratio,
    pub zero_recall_searches: Ratio,
    pub certified_searches: Ratio,
    pub truncated_searches: Ratio,
    pub budget_exhausted_searches: Ratio,
    pub index_empty_searches: Ratio,
    pub assessment_accept: Ratio,
    pub assessment_clarify: Ratio,
    pub assessment_abstain: Ratio,
    pub low_margin: Ratio,
    pub assembly_selected: Ratio,
    pub assembly_empty: Ratio,
    pub answers_at_least_once: bool,
    pub codes: BTreeMap<String, u64>,
    pub budget: BudgetSnapshot,
    pub advice: SearchAdvice,
    pub alerts: Vec<Alert>,
}

impl Default for WindowMetrics {
    fn default() -> Self {
        let budget = BudgetSnapshot::default();
        Self {
            capacity: DEFAULT_WINDOW,
            observations: 0,
            searches: 0,
            answered: Ratio::default(),
            deferred: Ratio::default(),
            faults: Ratio::default(),
            retrieval_share: Ratio::default(),
            retrieval_recall: Ratio::default(),
            search_recall: Ratio::default(),
            zero_recall_searches: Ratio::default(),
            certified_searches: Ratio::default(),
            truncated_searches: Ratio::default(),
            budget_exhausted_searches: Ratio::default(),
            index_empty_searches: Ratio::default(),
            assessment_accept: Ratio::default(),
            assessment_clarify: Ratio::default(),
            assessment_abstain: Ratio::default(),
            low_margin: Ratio::default(),
            assembly_selected: Ratio::default(),
            assembly_empty: Ratio::default(),
            answers_at_least_once: false,
            codes: BTreeMap::new(),
            budget,
            advice: SearchAdvice::from_budget(
                &BudgetState {
                    level: budget.level,
                },
                budget.mode,
            ),
            alerts: Vec::new(),
        }
    }
}

impl WindowMetrics {
    pub fn render_text(&self) -> String {
        let mut output = format!(
            "语义检索监控：窗口 {}/{}，预算 {}/{}（{}，{}）。\n",
            self.observations,
            self.capacity,
            self.budget.score,
            self.budget.capacity,
            self.budget.level.label(),
            self.budget.mode.label()
        );
        for (label, ratio) in [
            ("回答", self.answered),
            ("让位", self.deferred),
            ("故障", self.faults),
            ("走检索", self.retrieval_share),
            ("轮次召回", self.retrieval_recall),
            ("检索召回", self.search_recall),
            ("检索零召回", self.zero_recall_searches),
            ("top-k 证书", self.certified_searches),
            ("截断检索", self.truncated_searches),
            ("预算耗尽", self.budget_exhausted_searches),
            ("空索引检索", self.index_empty_searches),
            ("读法接受", self.assessment_accept),
            ("读法澄清", self.assessment_clarify),
            ("读法弃权", self.assessment_abstain),
            ("低边际", self.low_margin),
            ("装配选中", self.assembly_selected),
            ("装配为空", self.assembly_empty),
        ] {
            output.push_str(&format!(
                "- {}：{}/{}（{}）。\n",
                label,
                ratio.numerator,
                ratio.denominator,
                ratio
                    .bps
                    .map(|bps| format!("{}‰", bps))
                    .unwrap_or_else(|| "无样本".to_string())
            ));
        }
        if self.alerts.is_empty() {
            output.push_str("- 告警：无。\n");
        } else {
            for alert in &self.alerts {
                output.push_str(&format!(
                    "- 告警 [{}] {}：{}；修复：{}。\n",
                    alert.code,
                    match alert.severity {
                        AlertSeverity::Info => "提示",
                        AlertSeverity::Warning => "警告",
                        AlertSeverity::Critical => "严重",
                    },
                    alert.message,
                    alert.repair.label()
                ));
            }
        }
        output
    }
}

impl WindowMetrics {
    /// 把另一会话窗口并入当前窗口：比率按分子/分母重新计算，
    /// 只合并派生计数与告警码，不复制任何问句文本。
    pub fn merge(&mut self, other: &Self) {
        self.capacity = self.capacity.max(other.capacity);
        self.observations = self.observations.saturating_add(other.observations);
        self.searches = self.searches.saturating_add(other.searches);
        self.answered.merge(other.answered);
        self.deferred.merge(other.deferred);
        self.faults.merge(other.faults);
        self.retrieval_share.merge(other.retrieval_share);
        self.retrieval_recall.merge(other.retrieval_recall);
        self.search_recall.merge(other.search_recall);
        self.zero_recall_searches.merge(other.zero_recall_searches);
        self.certified_searches.merge(other.certified_searches);
        self.truncated_searches.merge(other.truncated_searches);
        self.budget_exhausted_searches
            .merge(other.budget_exhausted_searches);
        self.index_empty_searches.merge(other.index_empty_searches);
        self.assessment_accept.merge(other.assessment_accept);
        self.assessment_clarify.merge(other.assessment_clarify);
        self.assessment_abstain.merge(other.assessment_abstain);
        self.low_margin.merge(other.low_margin);
        self.assembly_selected.merge(other.assembly_selected);
        self.assembly_empty.merge(other.assembly_empty);
        self.answers_at_least_once |= other.answers_at_least_once;
        for (code, count) in &other.codes {
            *self.codes.entry(code.clone()).or_insert(0) += count;
        }
        for alert in &other.alerts {
            if !self.alerts.iter().any(|known| known.code == alert.code) {
                self.alerts.push(alert.clone());
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemanticFleetMetrics {
    pub sessions: usize,
    pub window: WindowMetrics,
    pub alerting_sessions: BTreeMap<String, u64>,
}

impl Default for SemanticFleetMetrics {
    fn default() -> Self {
        Self {
            sessions: 0,
            window: WindowMetrics::default(),
            alerting_sessions: BTreeMap::new(),
        }
    }
}

impl SemanticFleetMetrics {
    pub fn from_windows(windows: impl IntoIterator<Item = WindowMetrics>) -> Self {
        let mut fleet = Self::default();
        for window in windows {
            fleet.sessions += 1;
            fleet.window.merge(&window);
            for alert in &window.alerts {
                *fleet.alerting_sessions.entry(alert.code.clone()).or_insert(0) += 1;
            }
        }
        fleet
    }
}

#[derive(Debug, Clone)]
struct ActiveObservation {
    sequence: u64,
    question_digest: String,
    question_chars: usize,
    token_chars: usize,
    search: SearchFold,
    assembly: AssemblyFold,
    assessment: AssessmentFold,
    codes: BTreeSet<String>,
    nesting: u32,
}

#[derive(Debug, Clone)]
struct BudgetState {
    level: PressureLevel,
}

fn initial_level(config: &BudgetConfig) -> PressureLevel {
    if config.initial < config.strict_below {
        PressureLevel::Strict
    } else if config.initial < config.cautious_below {
        PressureLevel::Cautious
    } else {
        PressureLevel::Healthy
    }
}

#[derive(Debug, Clone)]
struct MonitorState {
    next_sequence: u64,
    observations: VecDeque<SemanticObservation>,
    active: Option<ActiveObservation>,
    score: i64,
    level: PressureLevel,
    mode: ControlMode,
}

impl MonitorState {
    fn new(config: &BudgetConfig) -> Self {
        Self {
            next_sequence: 0,
            observations: VecDeque::new(),
            active: None,
            score: config.initial,
            level: initial_level(config),
            mode: ControlMode::Observe,
        }
    }

    fn reset(&mut self, config: &BudgetConfig) {
        self.next_sequence = 0;
        self.observations.clear();
        self.active = None;
        self.score = config.initial;
        self.level = initial_level(config);
    }
}

#[derive(Debug)]
pub struct SemanticMonitor {
    window: usize,
    config: BudgetConfig,
    state: Mutex<MonitorState>,
}

impl Default for SemanticMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for SemanticMonitor {
    fn clone(&self) -> Self {
        let state = self.state.lock().clone();
        Self {
            window: self.window,
            config: self.config.clone(),
            state: Mutex::new(state),
        }
    }
}

impl SemanticMonitor {
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_WINDOW)
    }

    pub fn with_capacity(window: usize) -> Self {
        let config = BudgetConfig::default().clamped();
        Self {
            window: window.max(1),
            state: Mutex::new(MonitorState::new(&config)),
            config,
        }
    }

    pub fn with_budget(config: BudgetConfig) -> Self {
        Self::with_options(DEFAULT_WINDOW, config)
    }

    pub fn with_options(window: usize, budget: BudgetConfig) -> Self {
        let config = budget.clamped();
        Self {
            window: window.max(1),
            state: Mutex::new(MonitorState::new(&config)),
            config,
        }
    }

    pub fn window_capacity(&self) -> usize {
        self.window
    }

    pub fn set_mode(&self, mode: ControlMode) {
        self.state.lock().mode = mode;
    }

    pub fn mode(&self) -> ControlMode {
        self.state.lock().mode
    }

    pub fn advice(&self) -> SearchAdvice {
        let state = self.state.lock();
        SearchAdvice::from_budget(
            &BudgetState {
                level: state.level,
            },
            state.mode,
        )
    }

    pub fn clear(&self) {
        self.state.lock().reset(&self.config);
    }

    pub fn begin(&self, question: &str) {
        let mut state = self.state.lock();
        if let Some(active) = state.active.as_mut() {
            active.nesting = active.nesting.saturating_add(1);
            return;
        }
        state.next_sequence = state.next_sequence.saturating_add(1);
        let trimmed = question.trim();
        state.active = Some(ActiveObservation {
            sequence: state.next_sequence,
            question_digest: fnv1a_hex(trimmed.as_bytes()),
            question_chars: trimmed.chars().count(),
            token_chars: trimmed.chars().filter(|c| !c.is_whitespace()).count(),
            search: SearchFold::default(),
            assembly: AssemblyFold::default(),
            assessment: AssessmentFold::default(),
            codes: BTreeSet::new(),
            nesting: 0,
        });
    }

    pub fn record_search(&self, _query: &str, search: &EvidenceSearch) -> bool {
        let mut state = self.state.lock();
        let Some(active) = state.active.as_mut() else {
            return false;
        };
        active.search.observe(search);
        true
    }

    pub fn record_assembly(&self, assembly: &EvidenceAssembly) -> bool {
        self.record_assembly_receipt(&assembly.receipt)
    }

    pub fn record_assembly_receipt(&self, receipt: &AssemblyReceipt) -> bool {
        let mut state = self.state.lock();
        let Some(active) = state.active.as_mut() else {
            return false;
        };
        active.assembly.observe(receipt);
        true
    }

    pub fn record_assessment(&self, assessment: &Assessment) -> bool {
        let mut state = self.state.lock();
        let Some(active) = state.active.as_mut() else {
            return false;
        };
        active.assessment.observe(assessment);
        true
    }

    pub fn finish(&self, outcome: TurnOutcome) -> Option<SemanticObservation> {
        let mut state = self.state.lock();
        let mut active = state.active.take()?;
        if active.nesting > 0 {
            active.nesting -= 1;
            state.active = Some(active);
            return None;
        }
        active.codes.insert(outcome.code().to_string());
        if active.search.searches == 0 {
            active.codes.insert(CODE_NO_RETRIEVAL.to_string());
        } else {
            if active.search.returned == 0 {
                active.codes.insert(CODE_ZERO_RECALL.to_string());
            }
            if active.search.certified_searches < active.search.searches {
                active.codes.insert(CODE_UNCERTIFIED.to_string());
            }
            if active.search.budget_exhausted_searches > 0 {
                active.codes.insert(CODE_BUDGET_EXHAUSTED.to_string());
            }
            if active.search.index_empty_searches > 0 {
                active.codes.insert(CODE_INDEX_EMPTY.to_string());
            }
        }
        if active.assessment.abstain > 0 {
            active.codes.insert(CODE_ABSTAIN.to_string());
        }
        if active.assessment.clarify > 0 {
            active.codes.insert(CODE_CLARIFY.to_string());
        }
        if active.assessment.low_margin > 0 {
            active.codes.insert(CODE_LOW_MARGIN.to_string());
        }
        if active.assembly.empty_assemblies > 0 {
            active.codes.insert(CODE_ASSEMBLY_EMPTY.to_string());
        }
        let mode = state.mode;
        Self::settle(&self.config, &mut state, outcome, &active);
        let budget = BudgetSnapshot {
            mode,
            level: state.level,
            score: state.score,
            capacity: self.config.capacity,
        };
        let advice = SearchAdvice::from_budget(
            &BudgetState {
                level: state.level,
            },
            mode,
        );
        let observation = SemanticObservation {
            sequence: active.sequence,
            question_digest: active.question_digest,
            question_chars: active.question_chars,
            token_chars: active.token_chars,
            outcome,
            search: active.search,
            assembly: active.assembly,
            assessment: active.assessment,
            codes: active.codes.into_iter().collect(),
            budget,
            advice,
        };
        state.observations.push_back(observation.clone());
        while state.observations.len() > self.window {
            state.observations.pop_front();
        }
        Some(observation)
    }

    fn settle(
        config: &BudgetConfig,
        state: &mut MonitorState,
        outcome: TurnOutcome,
        active: &ActiveObservation,
    ) {
        let mut delta = match outcome {
            TurnOutcome::Answered => {
                if active.search.returned > 0 {
                    config.success_reward
                } else {
                    config.grounded_reward
                }
            }
            TurnOutcome::Deferred => -config.deferred_cost,
            TurnOutcome::Clarified => -config.clarify_cost,
            TurnOutcome::Fault => -config.fault_cost,
        };
        if active.assessment.abstain > 0 {
            delta -= config.abstain_cost;
        }
        if active.assessment.clarify > 0 {
            delta -= config.clarify_cost;
        }
        if active.search.searches > 0 && active.search.returned == 0 {
            delta -= config.zero_recall_cost;
        }
        if active.search.searches > 0
            && active.search.certified_searches < active.search.searches
        {
            delta -= config.uncertified_cost;
        }
        if active.search.budget_exhausted_searches > 0 {
            delta -= config.budget_exhausted_cost;
        }
        state.score = (state.score + delta).clamp(0, config.capacity);
        let score = state.score;
        state.level = match state.level {
            PressureLevel::Healthy => {
                if score < config.strict_below {
                    PressureLevel::Strict
                } else if score < config.cautious_below {
                    PressureLevel::Cautious
                } else {
                    PressureLevel::Healthy
                }
            }
            PressureLevel::Cautious => {
                if score < config.strict_below {
                    PressureLevel::Strict
                } else if score >= config.cautious_recover {
                    PressureLevel::Healthy
                } else {
                    PressureLevel::Cautious
                }
            }
            PressureLevel::Strict => {
                if score >= config.strict_recover {
                    PressureLevel::Cautious
                } else {
                    PressureLevel::Strict
                }
            }
        };
    }

    pub fn observations(&self) -> Vec<SemanticObservation> {
        self.state.lock().observations.iter().cloned().collect()
    }

    pub fn latest(&self) -> Option<SemanticObservation> {
        self.state.lock().observations.back().cloned()
    }

    pub fn snapshot(&self) -> WindowMetrics {
        let state = self.state.lock();
        let observations: Vec<&SemanticObservation> = state.observations.iter().collect();
        let total = observations.len() as u64;
        let mut answered = 0;
        let mut deferred = 0;
        let mut faults = 0;
        let mut retrieval_turns = 0;
        let mut retrieval_hits = 0;
        let mut searches = 0;
        let mut hit_searches = 0;
        let mut certified_searches = 0;
        let mut truncated_searches = 0;
        let mut budget_exhausted_searches = 0;
        let mut index_empty_searches = 0;
        let mut assessments = 0;
        let mut accepts = 0;
        let mut clarifies = 0;
        let mut abstains = 0;
        let mut low_margins = 0;
        let mut assemblies = 0;
        let mut selected_assemblies = 0;
        let mut empty_assemblies = 0;
        let mut codes: BTreeMap<String, u64> = BTreeMap::new();
        for observation in &observations {
            match observation.outcome {
                TurnOutcome::Answered | TurnOutcome::Clarified => answered += 1,
                TurnOutcome::Deferred => deferred += 1,
                TurnOutcome::Fault => faults += 1,
            }
            if observation.search.searches > 0 {
                retrieval_turns += 1;
                retrieval_hits += u64::from(observation.search.returned > 0);
            }
            searches += observation.search.searches;
            hit_searches += observation.search.hit_searches;
            certified_searches += observation.search.certified_searches;
            truncated_searches += observation.search.truncated_searches;
            budget_exhausted_searches += observation.search.budget_exhausted_searches;
            index_empty_searches += observation.search.index_empty_searches;
            assessments += observation.assessment.assessments;
            accepts += observation.assessment.accept;
            clarifies += observation.assessment.clarify;
            abstains += observation.assessment.abstain;
            low_margins += observation.assessment.low_margin;
            assemblies += observation.assembly.assemblies;
            selected_assemblies += observation.assembly.selected_assemblies;
            empty_assemblies += observation.assembly.empty_assemblies;
            for code in &observation.codes {
                *codes.entry(code.clone()).or_insert(0) += 1;
            }
        }
        let mode = state.mode;
        let budget = BudgetSnapshot {
            mode,
            level: state.level,
            score: state.score,
            capacity: self.config.capacity,
        };
        let advice = SearchAdvice::from_budget(
            &BudgetState {
                level: state.level,
            },
            mode,
        );
        let mut metrics = WindowMetrics {
            capacity: self.window,
            observations: total,
            searches,
            answered: Ratio::new(answered, total),
            deferred: Ratio::new(deferred, total),
            faults: Ratio::new(faults, total),
            retrieval_share: Ratio::new(retrieval_turns, total),
            retrieval_recall: Ratio::new(retrieval_hits, retrieval_turns),
            search_recall: Ratio::new(hit_searches, searches),
            zero_recall_searches: Ratio::new(searches.saturating_sub(hit_searches), searches),
            certified_searches: Ratio::new(certified_searches, searches),
            truncated_searches: Ratio::new(truncated_searches, searches),
            budget_exhausted_searches: Ratio::new(budget_exhausted_searches, searches),
            index_empty_searches: Ratio::new(index_empty_searches, searches),
            assessment_accept: Ratio::new(accepts, assessments),
            assessment_clarify: Ratio::new(clarifies, assessments),
            assessment_abstain: Ratio::new(abstains, assessments),
            low_margin: Ratio::new(low_margins, assessments),
            assembly_selected: Ratio::new(selected_assemblies, assemblies),
            assembly_empty: Ratio::new(empty_assemblies, assemblies),
            answers_at_least_once: answered > 0,
            codes,
            budget,
            advice,
            alerts: Vec::new(),
        };
        metrics.alerts = build_alerts(&metrics);
        metrics
    }

    pub fn report(&self) -> String {
        self.snapshot().render_text()
    }
}

fn ratio_bps(ratio: Ratio) -> Option<u32> {
    ratio.bps
}

fn build_alerts(metrics: &WindowMetrics) -> Vec<Alert> {
    let mut alerts = Vec::new();
    if metrics.observations >= MIN_ALERT_SAMPLE {
        if let Some(bps) = ratio_bps(metrics.retrieval_recall) {
            if bps <= LOW_RECALL_ALERT_BPS {
                alerts.push(Alert {
                    code: "low_retrieval_recall".to_string(),
                    severity: AlertSeverity::Warning,
                    numerator: metrics.retrieval_recall.numerator,
                    denominator: metrics.retrieval_recall.denominator,
                    bps: Some(bps),
                    message: format!(
                        "走检索的轮次里只有 {}‰ 至少命中一条证据。",
                        bps
                    ),
                    repair: RepairAction::AcquireFact,
                });
            }
        }
        if let Some(bps) = ratio_bps(metrics.zero_recall_searches) {
            if bps >= ZERO_RECALL_ALERT_BPS {
                alerts.push(Alert {
                    code: "zero_recall_spike".to_string(),
                    severity: AlertSeverity::Critical,
                    numerator: metrics.zero_recall_searches.numerator,
                    denominator: metrics.zero_recall_searches.denominator,
                    bps: Some(bps),
                    message: format!(
                        "最近 {} 次检索里零召回占 {}‰。",
                        metrics.searches, bps
                    ),
                    repair: RepairAction::AcquireFact,
                });
            }
        }
        if let Some(bps) = ratio_bps(metrics.deferred) {
            if bps >= DEFERRAL_ALERT_BPS {
                alerts.push(Alert {
                    code: "deferral_spike".to_string(),
                    severity: AlertSeverity::Warning,
                    numerator: metrics.deferred.numerator,
                    denominator: metrics.deferred.denominator,
                    bps: Some(bps),
                    message: format!("最近 {} 轮里让位占 {}‰。", metrics.observations, bps),
                    repair: RepairAction::AcquireVocabulary,
                });
            }
        }
        if let Some(bps) = ratio_bps(metrics.retrieval_share) {
            if bps <= NO_RETRIEVAL_ALERT_BPS {
                alerts.push(Alert {
                    code: "retrieval_not_exercised".to_string(),
                    severity: AlertSeverity::Info,
                    numerator: metrics.retrieval_share.numerator,
                    denominator: metrics.retrieval_share.denominator,
                    bps: Some(bps),
                    message: format!(
                        "最近 {} 轮里只有 {}‰ 真正走检索。",
                        metrics.observations, bps
                    ),
                    repair: RepairAction::InspectRouting,
                });
            }
        }
    }
    if metrics.searches >= MIN_ALERT_SAMPLE {
        for (label, ratio, code, repair) in [
            (
                "top-k 证书覆盖率",
                metrics.certified_searches,
                "low_top_k_certificate",
                RepairAction::RaiseScanBudget,
            ),
            (
                "截断检索占比",
                metrics.truncated_searches,
                "retrieval_truncation",
                RepairAction::RaiseScanBudget,
            ),
            (
                "预算耗尽占比",
                metrics.budget_exhausted_searches,
                "retrieval_budget_pressure",
                RepairAction::RaiseRetrievalDepth,
            ),
            (
                "空索引检索占比",
                metrics.index_empty_searches,
                "empty_retrieval_index",
                RepairAction::RebuildIndex,
            ),
        ] {
            let Some(bps) = ratio_bps(ratio) else {
                continue;
            };
            let fired = match code {
                "low_top_k_certificate" => bps <= LOW_CERTIFICATE_ALERT_BPS,
                "empty_retrieval_index" => ratio.numerator > 0,
                _ => bps >= BUDGET_PRESSURE_ALERT_BPS,
            };
            if fired {
                alerts.push(Alert {
                    code: code.to_string(),
                    severity: if code == "empty_retrieval_index" {
                        AlertSeverity::Critical
                    } else {
                        AlertSeverity::Warning
                    },
                    numerator: ratio.numerator,
                    denominator: ratio.denominator,
                    bps: Some(bps),
                    message: format!("{}：{}/{}（{}‰）。", label, ratio.numerator, ratio.denominator, bps),
                    repair,
                });
            }
        }
    }
    if metrics.assessment_accept.denominator >= MIN_ASSESSMENT_SAMPLE {
        for (label, ratio, code, repair, threshold) in [
            (
                "低边际读法",
                metrics.low_margin,
                "low_reading_margin",
                RepairAction::PreferClarification,
                LOW_MARGIN_ALERT_BPS,
            ),
            (
                "读法澄清",
                metrics.assessment_clarify,
                "clarify_heavy",
                RepairAction::AcquireAlias,
                CLARIFY_HEAVY_ALERT_BPS,
            ),
            (
                "读法弃权",
                metrics.assessment_abstain,
                "abstain_heavy",
                RepairAction::AcquireFact,
                ABSTAIN_HEAVY_ALERT_BPS,
            ),
        ] {
            let Some(bps) = ratio_bps(ratio) else {
                continue;
            };
            if bps >= threshold {
                alerts.push(Alert {
                    code: code.to_string(),
                    severity: AlertSeverity::Warning,
                    numerator: ratio.numerator,
                    denominator: ratio.denominator,
                    bps: Some(bps),
                    message: format!("{}：{}/{}（{}‰）。", label, ratio.numerator, ratio.denominator, bps),
                    repair,
                });
            }
        }
    }
    if metrics.assembly_empty.denominator >= MIN_ASSESSMENT_SAMPLE {
        if let Some(bps) = ratio_bps(metrics.assembly_empty) {
            if bps >= ASSEMBLY_EMPTY_ALERT_BPS {
                alerts.push(Alert {
                    code: "assembly_empty_heavy".to_string(),
                    severity: AlertSeverity::Warning,
                    numerator: metrics.assembly_empty.numerator,
                    denominator: metrics.assembly_empty.denominator,
                    bps: Some(bps),
                    message: format!(
                        "证据装配为空：{}/{}（{}‰）。",
                        metrics.assembly_empty.numerator,
                        metrics.assembly_empty.denominator,
                        bps
                    ),
                    repair: RepairAction::WidenEvidence,
                });
            }
        }
    }
    alerts.sort_by(|left, right| {
        right
            .severity
            .cmp(&left.severity)
            .then_with(|| left.code.cmp(&right.code))
    });
    alerts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense_retrieval::{RetrievalHit, SearchReceipt};
    use crate::query_planner::QueryPlan;
    use crate::rewrite_retrieval::{
        HypothesisAssessment, HypothesisCandidate, HypothesisSource,
    };

    fn hit(subject: &str) -> RetrievalHit {
        RetrievalHit {
            subject: subject.to_string(),
            predicate: "师傅".to_string(),
            object: "李四".to_string(),
            score: 0.8,
            lexical: 0.5,
            structural: 0.2,
            predicate_match: 0.1,
            reasons: Vec::new(),
        }
    }

    fn search(hits: Vec<RetrievalHit>, certified: bool, docs: usize) -> EvidenceSearch {
        EvidenceSearch {
            receipt: SearchReceipt {
                docs,
                query_terms: 1,
                candidates: 3,
                scanned: 3,
                returned: hits.len(),
                certified_top_k: certified,
                top_k: 2,
                ..SearchReceipt::default()
            },
            hits,
        }
    }

    fn assessment(margin: f64, decision: Decision) -> Assessment {
        let candidate = HypothesisCandidate::new(
            HypothesisSource::ExactPlan,
            "planner",
            "张三的师傅是谁",
            QueryPlan::FindObject {
                segments: vec!["张三".to_string(), "师傅".to_string()],
            },
        );
        Assessment {
            dialect: "test".to_string(),
            question: "张三的师傅是谁".to_string(),
            decision,
            selected: (decision == Decision::Accept).then_some(0),
            hypotheses: vec![HypothesisAssessment {
                candidate,
                probes: Vec::new(),
                support: 0.8,
                coverage: 0.8,
                margin,
                consistency: 0.8,
                confidence: 0.8,
                answer: Some("李四".to_string()),
                answers: Vec::new(),
                notes: Vec::new(),
            }],
            notes: Vec::new(),
        }
    }

    #[test]
    fn observation_folds_receipts_without_raw_question() {
        let monitor = SemanticMonitor::new();
        monitor.begin("秘密问题甲乙丙");
        assert!(monitor.record_search("张三 师傅", &search(vec![hit("张三")], true, 10)));
        assert!(monitor.record_assessment(&assessment(0.4, Decision::Accept)));
        let observation = monitor.finish(TurnOutcome::Answered).expect("应完成观测");
        assert_eq!(observation.search.searches, 1);
        assert_eq!(observation.search.returned, 1);
        assert!(observation.codes.contains(&CODE_ANSWERED.to_string()));
        let json = serde_json::to_string(&observation).unwrap();
        assert!(!json.contains("秘密问题甲乙丙"));
        assert_eq!(observation.question_chars, "秘密问题甲乙丙".chars().count());
    }

    #[test]
    fn fleet_metrics_merge_ratios_without_double_counting_denominators() {
        let mut first = WindowMetrics {
            observations: 10,
            answered: Ratio::new(8, 10),
            ..WindowMetrics::default()
        };
        first.codes.insert(CODE_ANSWERED.to_string(), 8);
        first.alerts.push(Alert {
            code: "low_retrieval_recall".to_string(),
            severity: AlertSeverity::Warning,
            numerator: 4,
            denominator: 10,
            bps: Some(4000),
            message: "测试告警".to_string(),
            repair: RepairAction::WidenEvidence,
        });
        let second = WindowMetrics {
            observations: 5,
            answered: Ratio::new(2, 5),
            ..WindowMetrics::default()
        };
        let fleet = SemanticFleetMetrics::from_windows(vec![first, second]);
        assert_eq!(fleet.sessions, 2);
        assert_eq!(fleet.window.observations, 15);
        assert_eq!(fleet.window.answered.numerator, 10);
        assert_eq!(fleet.window.answered.denominator, 15);
        assert_eq!(fleet.window.answered.bps, Some(6667));
        assert_eq!(fleet.window.codes.get(CODE_ANSWERED), Some(&8));
        assert_eq!(fleet.window.alerts.len(), 1);
        assert_eq!(
            fleet.alerting_sessions.get("low_retrieval_recall"),
            Some(&1)
        );
    }

    #[test]
    fn cold_start_ratios_are_unmeasured_and_alert_free() {
        let monitor = SemanticMonitor::new();
        let metrics = monitor.snapshot();
        assert_eq!(metrics.observations, 0);
        assert!(metrics.retrieval_recall.bps.is_none());
        assert!(metrics.alerts.is_empty());
        monitor.begin("你好");
        monitor.finish(TurnOutcome::Deferred);
        let metrics = monitor.snapshot();
        assert_eq!(metrics.observations, 1);
        assert!(metrics.alerts.is_empty(), "冷启动不应告警：{:?}", metrics.alerts);
    }

    #[test]
    fn zero_recall_spike_produces_actionable_alert() {
        let monitor = SemanticMonitor::new();
        for _ in 0..MIN_ALERT_SAMPLE {
            monitor.begin("张三的师傅是谁");
            monitor.record_search("张三 师傅", &search(Vec::new(), true, 10));
            monitor.finish(TurnOutcome::Answered);
        }
        let metrics = monitor.snapshot();
        assert_eq!(metrics.search_recall.bps, Some(0));
        let alert = metrics
            .alerts
            .iter()
            .find(|alert| alert.code == "zero_recall_spike")
            .expect("应产生零召回告警");
        assert_eq!(alert.repair, RepairAction::AcquireFact);
        assert_eq!(alert.numerator, MIN_ALERT_SAMPLE);
        assert_eq!(alert.denominator, MIN_ALERT_SAMPLE);
    }

    #[test]
    fn window_is_bounded_and_sequence_monotonic() {
        let monitor = SemanticMonitor::with_capacity(2);
        for index in 0..5 {
            monitor.begin(&format!("问题 {}", index));
            monitor.finish(TurnOutcome::Answered);
        }
        let observations = monitor.observations();
        assert_eq!(observations.len(), 2);
        assert_eq!(observations[0].sequence, 4);
        assert_eq!(observations[1].sequence, 5);
        assert_eq!(monitor.snapshot().observations, 2);
    }

    #[test]
    fn budget_enters_strict_with_hysteresis_and_recovers_stepwise() {
        let config = BudgetConfig {
            initial: 60,
            success_reward: 3,
            deferred_cost: 20,
            ..BudgetConfig::default()
        };
        let monitor = SemanticMonitor::with_budget(config);
        monitor.set_mode(ControlMode::Control);
        for _ in 0..2 {
            monitor.begin("不知道");
            monitor.finish(TurnOutcome::Deferred);
        }
        assert_eq!(monitor.snapshot().budget.level, PressureLevel::Strict);
        let advice = monitor.snapshot().advice;
        assert_eq!(advice.max_items, Some(STRICT_MAX_ITEMS));
        assert!(advice.prefer_clarification);

        for _ in 0..7 {
            monitor.begin("张三的师傅是谁");
            monitor.record_search("张三 师傅", &search(vec![hit("张三")], true, 10));
            monitor.finish(TurnOutcome::Answered);
        }
        assert_eq!(
            monitor.snapshot().budget.level,
            PressureLevel::Cautious,
            "严格档一次不能跳回健康档"
        );
    }

    #[test]
    fn low_margin_and_abstain_are_folded() {
        let monitor = SemanticMonitor::new();
        monitor.begin("张三的师傅到底是谁");
        monitor.record_assessment(&assessment(0.005, Decision::Abstain));
        let observation = monitor.finish(TurnOutcome::Deferred).expect("应完成观测");
        assert!(observation.codes.contains(&CODE_LOW_MARGIN.to_string()));
        assert!(observation.codes.contains(&CODE_ABSTAIN.to_string()));
        assert!(monitor.snapshot().budget.score < 60);
    }
}

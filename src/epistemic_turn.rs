//! EBT：逐命题置信回合（Epistemic Turn）。
//!
//! 传统回答把整句话压成一个置信度；本模块保留**命题级信封**，把低置信
//! 命题放进有界复核队列，并用「基础修订 + 单写者应用」处理异步返回的证据：
//!
//! - [`TurnEnvelope`]：一轮回答的只读快照（计划 id、修订、逐命题区间）；
//! - [`DeliberationTask`]：待复核项，优先级由区间下界与决策档位决定；
//! - [`DeliberationProposal`]：慢通道返回的候选更新，必须带观测修订；
//! - [`EpistemicTurnBook`]：单写者台账；过期提案拒绝，重复提案去重，
//!   每批最多应用一条「可断言 → 不可断言」的更正，避免突发刷屏。
//!
//! 队列满时任务进入 backlog；应用与排序全部按 `(优先级, 槽, id)` 确定性执行。
//! 本模块不启动线程，也不写图谱：调用方负责把慢通道结果作为 proposal 交回。

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::answer_plan::AnswerPlan;
use crate::confidence::{ClaimEnvelope, ConfidenceConfig, EnvelopeVerdict};
use crate::context_image::fnv1a_hex;

/// 活跃队列容量。
pub const PENDING_CAPACITY: usize = 8;
/// backlog 容量；超出后丢弃优先级最低且最旧的任务。
pub const BACKLOG_CAPACITY: usize = 32;
/// 每轮新生成计划最多入队任务数。
pub const MAX_TASKS_PER_TURN: usize = 1;
/// 每批最多应用的「可断言 → 不可断言」更正数。
pub const MAX_CORRECTIONS_PER_BATCH: usize = 1;
/// 已应用 id 的保留上限（去重窗口）。
pub const APPLIED_CAPACITY: usize = 64;
/// 提案说明的字符上限。
pub const NOTE_LIMIT: usize = 160;

/// 复核策略：阈值与队列预算。默认值对应 `[reasoning.confidence]` 的出厂阈值。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TurnPolicy {
    pub answer_bps: u16,
    pub hedge_bps: u16,
    pub max_tasks_per_turn: usize,
    pub pending_capacity: usize,
    pub max_corrections_per_batch: usize,
}

impl Default for TurnPolicy {
    fn default() -> Self {
        Self {
            answer_bps: 8_600,
            hedge_bps: 6_200,
            max_tasks_per_turn: MAX_TASKS_PER_TURN,
            pending_capacity: PENDING_CAPACITY,
            max_corrections_per_batch: MAX_CORRECTIONS_PER_BATCH,
        }
    }
}

impl TurnPolicy {
    pub fn from_confidence(config: &ConfidenceConfig) -> Self {
        let (answer_bps, hedge_bps) = config.envelope_thresholds_bps();
        Self {
            answer_bps,
            hedge_bps,
            ..Self::default()
        }
    }

    pub fn verdict(&self, envelope: ClaimEnvelope) -> EnvelopeVerdict {
        envelope.verdict(self.answer_bps, self.hedge_bps)
    }
}

/// 一条被观测的命题。文本随计划保存，便于更正时逐字对照。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservedClaim {
    pub slot: String,
    pub predicate: String,
    pub text: String,
    #[serde(default)]
    pub session: bool,
    pub envelope: ClaimEnvelope,
}

/// 一轮回答的只读信封。只有带 `ClaimEnvelope` 的命题才进入复核视野。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnEnvelope {
    pub turn: u64,
    pub revision: u64,
    pub plan_id: String,
    #[serde(default)]
    pub claims: Vec<ObservedClaim>,
}

impl TurnEnvelope {
    pub fn from_plan(turn: u64, plan: &AnswerPlan) -> Self {
        Self {
            turn,
            revision: plan.revision,
            plan_id: plan.id.clone(),
            claims: plan
                .claims
                .iter()
                .filter_map(|claim| {
                    claim.envelope.map(|envelope| ObservedClaim {
                        slot: claim.slot.clone(),
                        predicate: claim.predicate.clone(),
                        text: claim.text.clone(),
                        session: claim.session,
                        envelope,
                    })
                })
                .collect(),
        }
    }

    pub fn claim(&self, slot: &str) -> Option<&ObservedClaim> {
        self.claims.iter().find(|claim| claim.slot == slot)
    }

    pub fn is_empty(&self) -> bool {
        self.claims.is_empty()
    }
}

/// 一个待复核命题。id 由「计划内容 + 槽 + 修订」确定，重复观测不会重复入队。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeliberationTask {
    pub id: String,
    pub turn: u64,
    pub plan_id: String,
    pub base_revision: u64,
    pub slot: String,
    pub predicate: String,
    pub text: String,
    #[serde(default)]
    pub session: bool,
    pub envelope: ClaimEnvelope,
    pub verdict: EnvelopeVerdict,
    pub priority: u16,
}

impl DeliberationTask {
    fn from_claim(
        turn: u64,
        plan_id: &str,
        revision: u64,
        claim: &ObservedClaim,
        policy: &TurnPolicy,
    ) -> Self {
        let verdict = policy.verdict(claim.envelope);
        let priority = claim
            .envelope
            .deliberation_priority(policy.answer_bps, policy.hedge_bps);
        let id = format!(
            "et-{}",
            fnv1a_hex(format!("{}|{}|{}", plan_id, claim.slot, revision).as_bytes())
        );
        Self {
            id,
            turn,
            plan_id: plan_id.to_string(),
            base_revision: revision,
            slot: claim.slot.clone(),
            predicate: claim.predicate.clone(),
            text: claim.text.clone(),
            session: claim.session,
            envelope: claim.envelope,
            verdict,
            priority,
        }
    }

    /// 是否需要慢通道复核：只有区间跨过判定带或落在带内才有信息价值；
    /// 已经可断言或确定证据不足的命题不占用队列。
    pub fn is_actionable(&self) -> bool {
        self.priority > 0
    }

    pub fn render_zh(&self) -> String {
        format!(
            "待复核「{}」：{}（优先级 {}）",
            self.text,
            self.verdict.label_zh(),
            self.priority
        )
    }
}

/// 慢通道返回的候选更新。携带 `observed_revision` 供单写者做过期栅栏。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeliberationProposal {
    pub task_id: String,
    pub turn: u64,
    pub plan_id: String,
    pub base_revision: u64,
    pub observed_revision: u64,
    pub slot: String,
    pub envelope: ClaimEnvelope,
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub note: String,
    #[serde(default)]
    pub source: String,
}

impl DeliberationProposal {
    pub fn new(
        task: &DeliberationTask,
        observed_revision: u64,
        envelope: ClaimEnvelope,
        text: &str,
        source: &str,
    ) -> Self {
        Self {
            task_id: task.id.clone(),
            turn: task.turn,
            plan_id: task.plan_id.clone(),
            base_revision: task.base_revision,
            observed_revision,
            slot: task.slot.clone(),
            envelope,
            text: text.to_string(),
            note: String::new(),
            source: source.to_string(),
        }
    }

    pub fn with_note(mut self, note: &str) -> Self {
        self.note = truncate_chars(note, NOTE_LIMIT);
        self
    }
}

/// 一次更新对命题语义的影响等级。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimUpdateKind {
    /// 结论、语气、文本均未变。
    Unchanged,
    /// 区间或文本变了，但决策档位没变。
    Revised,
    /// 非可断言 → 可断言。
    Confirmed,
    /// 可断言 → 非可断言：必须向用户明说。
    Corrected,
}

impl ClaimUpdateKind {
    pub fn label(self) -> &'static str {
        match self {
            ClaimUpdateKind::Unchanged => "无变化",
            ClaimUpdateKind::Revised => "更新",
            ClaimUpdateKind::Confirmed => "确认",
            ClaimUpdateKind::Corrected => "更正",
        }
    }
}

/// 已应用的逐命题更新。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaimUpdate {
    pub task_id: String,
    pub turn: u64,
    pub slot: String,
    pub predicate: String,
    pub before_text: String,
    pub after_text: String,
    pub before_verdict: EnvelopeVerdict,
    pub after_verdict: EnvelopeVerdict,
    pub before_envelope: ClaimEnvelope,
    pub after_envelope: ClaimEnvelope,
    #[serde(default = "default_answer_bps")]
    pub answer_bps: u16,
    #[serde(default = "default_hedge_bps")]
    pub hedge_bps: u16,
    pub kind: ClaimUpdateKind,
    #[serde(default)]
    pub note: String,
    #[serde(default)]
    pub source: String,
}

impl ClaimUpdate {
    pub fn render_zh(&self) -> String {
        let before = self
            .before_envelope
            .render_zh(self.answer_bps, self.hedge_bps);
        let after = self
            .after_envelope
            .render_zh(self.answer_bps, self.hedge_bps);
        match self.kind {
            ClaimUpdateKind::Corrected => format!(
                "第{}轮复核更正「{}」：上一轮判为{}（{}），当前证据为{}（{}）。来源：{}。",
                self.turn,
                self.after_text,
                self.before_verdict.label_zh(),
                before,
                self.after_verdict.label_zh(),
                after,
                if self.source.is_empty() {
                    "复核"
                } else {
                    self.source.as_str()
                }
            ),
            ClaimUpdateKind::Confirmed => format!(
                "第{}轮复核确认「{}」：原判{}（{}），当前证据达到{}（{}）。",
                self.turn,
                self.after_text,
                self.before_verdict.label_zh(),
                before,
                self.after_verdict.label_zh(),
                after
            ),
            ClaimUpdateKind::Revised => format!(
                "第{}轮复核更新「{}」：{}（{}）→ {}（{}）。",
                self.turn,
                self.after_text,
                self.before_verdict.label_zh(),
                before,
                self.after_verdict.label_zh(),
                after
            ),
            ClaimUpdateKind::Unchanged => format!(
                "第{}轮复核「{}」无变化（{}）。",
                self.turn, self.after_text, after
            ),
        }
    }

    pub fn is_visible(&self) -> bool {
        self.kind != ClaimUpdateKind::Unchanged
    }
}

/// 提案被拒绝的原因。拒绝不会消耗任务。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RejectionReason {
    /// 观测修订与当前修订不一致：慢通道结果已过期。
    StaleRevision,
    /// 任务不在台账中（可能已被另一写者应用或从未入队）。
    UnknownTask,
    /// 同一任务已应用过。
    Duplicate,
    /// 提案槽位与任务不一致。
    InvalidTask,
    /// 本批更正预算已用完。
    CorrectionBudget,
}

impl RejectionReason {
    pub fn label(self) -> &'static str {
        match self {
            RejectionReason::StaleRevision => "过期修订",
            RejectionReason::UnknownTask => "未知任务",
            RejectionReason::Duplicate => "重复提案",
            RejectionReason::InvalidTask => "任务不匹配",
            RejectionReason::CorrectionBudget => "更正预算用尽",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProposalRejection {
    pub task_id: String,
    pub reason: RejectionReason,
}

/// 单批应用结果。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApplyReport {
    #[serde(default)]
    pub updates: Vec<ClaimUpdate>,
    #[serde(default)]
    pub rejections: Vec<ProposalRejection>,
}

impl ApplyReport {
    pub fn is_empty(&self) -> bool {
        self.updates.is_empty() && self.rejections.is_empty()
    }

    pub fn visible_updates(&self) -> impl Iterator<Item = &ClaimUpdate> {
        self.updates.iter().filter(|update| update.is_visible())
    }

    pub fn render_zh(&self) -> String {
        let mut lines: Vec<String> = self.visible_updates().map(ClaimUpdate::render_zh).collect();
        let rejected = self.rejections.len();
        if rejected > 0 {
            lines.push(format!("另有 {} 条复核提案未应用。", rejected));
        }
        lines.join("\n")
    }
}

/// 观测一轮计划后的队列报告。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScheduleReport {
    #[serde(default)]
    pub scheduled: usize,
    #[serde(default)]
    pub deferred: usize,
    #[serde(default)]
    pub dropped: usize,
    #[serde(default)]
    pub pending: usize,
}

/// 复核报告：应用结果 + 新一轮调度。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReconcileReport {
    #[serde(default)]
    pub applied: ApplyReport,
    #[serde(default)]
    pub scheduled: usize,
    #[serde(default)]
    pub pending: usize,
}

impl ReconcileReport {
    pub fn render_zh(&self) -> String {
        let mut parts = Vec::new();
        let applied = self.applied.render_zh();
        if !applied.is_empty() {
            parts.push(applied);
        }
        if self.scheduled > 0 {
            parts.push(format!("新增待复核 {} 项。", self.scheduled));
        }
        if self.pending > 0 {
            parts.push(format!("队列剩余 {} 项。", self.pending));
        }
        parts.join("\n")
    }
}

/// 有界台账指标（只增计数，便于快照与诊断）。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnMetrics {
    #[serde(default)]
    pub observed: u64,
    #[serde(default)]
    pub scheduled: u64,
    #[serde(default)]
    pub deferred: u64,
    #[serde(default)]
    pub dropped: u64,
    #[serde(default)]
    pub applied: u64,
    #[serde(default)]
    pub corrections: u64,
    #[serde(default)]
    pub stale: u64,
    #[serde(default)]
    pub duplicates: u64,
    #[serde(default)]
    pub budget_rejected: u64,
}

/// 单写者复核台账。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EpistemicTurnBook {
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub turn: u64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pending: Vec<DeliberationTask>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub backlog: Vec<DeliberationTask>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub last_claims: BTreeMap<String, ObservedClaim>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub last_plan_id: String,
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub last_revision: u64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub applied_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "is_zero_metrics")]
    pub metrics: TurnMetrics,
}

fn default_answer_bps() -> u16 {
    8_600
}

fn default_hedge_bps() -> u16 {
    6_200
}

fn is_zero_metrics(metrics: &TurnMetrics) -> bool {
    *metrics == TurnMetrics::default()
}

fn is_zero_u64(value: &u64) -> bool {
    *value == 0
}

impl EpistemicTurnBook {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.turn == 0
            && self.pending.is_empty()
            && self.backlog.is_empty()
            && self.last_claims.is_empty()
            && self.last_plan_id.is_empty()
            && self.last_revision == 0
            && self.applied_ids.is_empty()
            && self.metrics == TurnMetrics::default()
    }

    pub fn clear(&mut self) {
        *self = Self::default();
    }

    pub fn pending(&self) -> &[DeliberationTask] {
        &self.pending
    }

    pub fn backlog(&self) -> &[DeliberationTask] {
        &self.backlog
    }

    pub fn pending_count(&self) -> usize {
        self.pending.len() + self.backlog.len()
    }

    pub fn last_claim(&self, slot: &str) -> Option<&ObservedClaim> {
        self.last_claims.get(slot)
    }

    pub fn metrics(&self) -> &TurnMetrics {
        &self.metrics
    }

    /// 观察一轮计划：刷新 last_claims，并把非可断言命题放进有界队列。
    ///
    /// 同一计划重复观测不会产生新任务；每轮最多加入
    /// `policy.max_tasks_per_turn` 个新任务，其余进入 backlog。
    pub fn observe_turn(&mut self, current: &TurnEnvelope, policy: &TurnPolicy) -> ScheduleReport {
        self.metrics.observed += 1;
        self.turn = self.turn.max(current.turn);
        self.last_plan_id = current.plan_id.clone();
        self.last_revision = current.revision;
        // 上一轮可断言的命题一旦掉档，先保留旧判决，等复核回合逐字更正；
        // 否则这里直接覆盖 last_claims 会把「更正」静默成普通更新。
        let previous_claims = self.last_claims.clone();
        let mut deferred = BTreeSet::new();
        for (slot, previous) in &previous_claims {
            let before = policy.verdict(previous.envelope);
            let after = current
                .claim(slot)
                .map(|claim| policy.verdict(claim.envelope))
                .unwrap_or(EnvelopeVerdict::Abstain);
            if before == EnvelopeVerdict::Assert && after != EnvelopeVerdict::Assert {
                deferred.insert(slot.clone());
            }
        }
        self.last_claims = current
            .claims
            .iter()
            .map(|claim| (claim.slot.clone(), claim.clone()))
            .collect();
        for slot in &deferred {
            if let Some(previous) = previous_claims.get(slot) {
                self.last_claims.insert(slot.clone(), previous.clone());
            }
        }

        let mut known: BTreeSet<String> = self
            .pending
            .iter()
            .chain(self.backlog.iter())
            .map(|task| task.id.clone())
            .chain(self.applied_ids.iter().cloned())
            .collect();
        let mut candidates: Vec<DeliberationTask> = Vec::new();
        let mut new_count = 0;
        for claim in &current.claims {
            let task = DeliberationTask::from_claim(
                current.turn,
                &current.plan_id,
                current.revision,
                claim,
                policy,
            );
            if !task.is_actionable() || known.contains(&task.id) {
                continue;
            }
            if new_count >= policy.max_tasks_per_turn {
                continue;
            }
            new_count += 1;
            known.insert(task.id.clone());
            candidates.push(task);
        }
        let scheduled = candidates.len();
        self.metrics.scheduled += scheduled as u64;

        let mut all: Vec<DeliberationTask> = self
            .pending
            .drain(..)
            .chain(self.backlog.drain(..))
            .chain(candidates)
            .collect();
        all.sort_by(task_order);
        all.dedup_by(|left, right| left.id == right.id);
        let capacity = policy.pending_capacity.max(1);
        let split = all.len().min(capacity);
        let deferred = all.len() - split;
        self.pending = all.drain(..split).collect();
        self.backlog = all;
        let dropped = self.trim_backlog();
        self.metrics.deferred += deferred as u64;
        self.metrics.dropped += dropped as u64;
        ScheduleReport {
            scheduled,
            deferred,
            dropped,
            pending: self.pending_count(),
        }
    }

    fn trim_backlog(&mut self) -> usize {
        if self.backlog.len() <= BACKLOG_CAPACITY {
            return 0;
        }
        self.backlog.sort_by(task_order);
        let dropped = self.backlog.len() - BACKLOG_CAPACITY;
        self.backlog.truncate(BACKLOG_CAPACITY);
        dropped
    }

    fn take_task(&mut self, task_id: &str) -> Option<DeliberationTask> {
        if let Some(index) = self.pending.iter().position(|task| task.id == task_id) {
            return Some(self.pending.remove(index));
        }
        if let Some(index) = self.backlog.iter().position(|task| task.id == task_id) {
            return Some(self.backlog.remove(index));
        }
        None
    }

    fn remember_applied(&mut self, task_id: &str) {
        if self.applied_ids.iter().any(|id| id == task_id) {
            return;
        }
        self.applied_ids.push(task_id.to_string());
        if self.applied_ids.len() > APPLIED_CAPACITY {
            let overflow = self.applied_ids.len() - APPLIED_CAPACITY;
            self.applied_ids.drain(..overflow);
        }
    }

    /// 应用一批提案。调用方负责提供**当前**修订；过期提案整体拒绝。
    pub fn apply_batch(
        &mut self,
        proposals: Vec<DeliberationProposal>,
        current_revision: u64,
        policy: &TurnPolicy,
    ) -> ApplyReport {
        let mut proposals = proposals;
        proposals.sort_by(|left, right| {
            (left.turn, left.task_id.as_str()).cmp(&(right.turn, right.task_id.as_str()))
        });
        let mut report = ApplyReport::default();
        let mut corrections = 0usize;
        for proposal in proposals {
            let Some(task) = self
                .pending
                .iter()
                .chain(self.backlog.iter())
                .find(|task| task.id == proposal.task_id)
                .cloned()
            else {
                let duplicate = self.applied_ids.iter().any(|id| id == &proposal.task_id);
                if duplicate {
                    self.metrics.duplicates += 1;
                }
                report.rejections.push(ProposalRejection {
                    task_id: proposal.task_id,
                    reason: if duplicate {
                        RejectionReason::Duplicate
                    } else {
                        RejectionReason::UnknownTask
                    },
                });
                continue;
            };
            if proposal.observed_revision != current_revision {
                self.metrics.stale += 1;
                report.rejections.push(ProposalRejection {
                    task_id: proposal.task_id,
                    reason: RejectionReason::StaleRevision,
                });
                continue;
            }
            if proposal.slot != task.slot
                || proposal.envelope.lower_bps() > proposal.envelope.upper_bps()
            {
                report.rejections.push(ProposalRejection {
                    task_id: proposal.task_id,
                    reason: RejectionReason::InvalidTask,
                });
                continue;
            }
            let previous = self
                .last_claims
                .get(&task.slot)
                .cloned()
                .unwrap_or_else(|| ObservedClaim {
                    slot: task.slot.clone(),
                    predicate: task.predicate.clone(),
                    text: task.text.clone(),
                    session: task.session,
                    envelope: task.envelope,
                });
            let after_verdict = policy.verdict(proposal.envelope);
            let before_verdict = policy.verdict(previous.envelope);
            let after_text = if proposal.text.trim().is_empty() {
                previous.text.clone()
            } else {
                proposal.text.clone()
            };
            let kind = classify_update(
                &previous,
                before_verdict,
                after_verdict,
                proposal.envelope,
                &after_text,
            );
            if kind == ClaimUpdateKind::Corrected && corrections >= policy.max_corrections_per_batch
            {
                self.metrics.budget_rejected += 1;
                report.rejections.push(ProposalRejection {
                    task_id: proposal.task_id,
                    reason: RejectionReason::CorrectionBudget,
                });
                continue;
            }
            self.take_task(&proposal.task_id);
            self.remember_applied(&proposal.task_id);
            self.metrics.applied += 1;
            if kind == ClaimUpdateKind::Corrected {
                corrections += 1;
                self.metrics.corrections += 1;
            }
            let update = ClaimUpdate {
                task_id: proposal.task_id.clone(),
                turn: proposal.turn,
                slot: task.slot.clone(),
                predicate: task.predicate.clone(),
                before_text: previous.text.clone(),
                after_text: after_text.clone(),
                before_verdict,
                after_verdict,
                before_envelope: previous.envelope,
                after_envelope: proposal.envelope,
                answer_bps: policy.answer_bps,
                hedge_bps: policy.hedge_bps,
                kind,
                note: proposal.note.clone(),
                source: proposal.source.clone(),
            };
            self.last_claims.insert(
                task.slot.clone(),
                ObservedClaim {
                    slot: task.slot,
                    predicate: task.predicate,
                    text: after_text,
                    session: task.session,
                    envelope: proposal.envelope,
                },
            );
            report.updates.push(update);
        }
        report
    }

    pub fn apply_proposal(
        &mut self,
        proposal: DeliberationProposal,
        current_revision: u64,
        policy: &TurnPolicy,
    ) -> ApplyReport {
        self.apply_batch(vec![proposal], current_revision, policy)
    }

    /// 用当前计划复核所有待办，并顺手检查上一轮每条已观测命题。
    ///
    /// 待办提案走 `apply_batch`（过期修订拒绝）；上一轮已断言、但当前证据
    /// 掉档的命题即使没有待办任务也会生成更正。每轮最多应用一条更正，
    /// 未应用的更正保留旧状态，下一次复核可以继续处理。
    ///
    /// 这是同步版慢通道；异步宿主可跳过本方法，直接调用 `apply_proposal`。
    pub fn reconcile(&mut self, current: &TurnEnvelope, policy: &TurnPolicy) -> ReconcileReport {
        let tasks: Vec<DeliberationTask> = self
            .pending
            .iter()
            .chain(self.backlog.iter())
            .cloned()
            .collect();
        let task_slots: BTreeSet<String> = tasks.iter().map(|task| task.slot.clone()).collect();
        let proposals: Vec<DeliberationProposal> = tasks
            .iter()
            .map(|task| match current.claim(&task.slot) {
                Some(claim) => DeliberationProposal::new(
                    task,
                    current.revision,
                    claim.envelope,
                    &claim.text,
                    "当前知识",
                )
                .with_note("复核当前知识"),
                None => {
                    let mut proposal = DeliberationProposal::new(
                        task,
                        current.revision,
                        ClaimEnvelope::point(0.0),
                        "（当前知识不再支持该命题）",
                        "当前知识",
                    );
                    proposal.note = "命题已消失".to_string();
                    proposal
                }
            })
            .collect();
        self.turn = self.turn.max(current.turn);
        let mut applied = self.apply_batch(proposals, current.revision, policy);
        let mut corrections = applied
            .updates
            .iter()
            .filter(|update| update.kind == ClaimUpdateKind::Corrected)
            .count();
        let handled_slots: BTreeSet<String> = applied
            .updates
            .iter()
            .map(|update| update.slot.clone())
            .collect();
        let rejected_slots: BTreeSet<String> = applied
            .rejections
            .iter()
            .filter_map(|rejection| {
                self.pending
                    .iter()
                    .chain(self.backlog.iter())
                    .find(|task| task.id == rejection.task_id)
                    .map(|task| task.slot.clone())
            })
            .collect();
        let previous_claims = self.last_claims.clone();
        let mut blocked: BTreeSet<String> =
            task_slots.intersection(&rejected_slots).cloned().collect();
        let mut updates = Vec::new();
        for (slot, previous) in &previous_claims {
            if handled_slots.contains(slot) || blocked.contains(slot) {
                continue;
            }
            let observed = current.claim(slot);
            let (after_envelope, after_text) = match observed {
                Some(claim) => (claim.envelope, claim.text.clone()),
                None => (
                    ClaimEnvelope::point(0.0),
                    "（当前知识不再支持该命题）".to_string(),
                ),
            };
            let before_verdict = policy.verdict(previous.envelope);
            let after_verdict = policy.verdict(after_envelope);
            let kind = classify_update(
                previous,
                before_verdict,
                after_verdict,
                after_envelope,
                &after_text,
            );
            if kind == ClaimUpdateKind::Unchanged {
                continue;
            }
            if kind == ClaimUpdateKind::Corrected && corrections >= policy.max_corrections_per_batch
            {
                blocked.insert(slot.clone());
                self.metrics.budget_rejected += 1;
                applied.rejections.push(ProposalRejection {
                    task_id: format!(
                        "observe-{}",
                        fnv1a_hex(format!("{}|{}", current.plan_id, slot).as_bytes())
                    ),
                    reason: RejectionReason::CorrectionBudget,
                });
                continue;
            }
            let update = ClaimUpdate {
                task_id: format!(
                    "observe-{}",
                    fnv1a_hex(format!("{}|{}", current.plan_id, slot).as_bytes())
                ),
                turn: current.turn,
                slot: slot.clone(),
                predicate: previous.predicate.clone(),
                before_text: previous.text.clone(),
                after_text: after_text.clone(),
                before_verdict,
                after_verdict,
                before_envelope: previous.envelope,
                after_envelope,
                answer_bps: policy.answer_bps,
                hedge_bps: policy.hedge_bps,
                kind,
                note: String::new(),
                source: "当前知识".to_string(),
            };
            if kind == ClaimUpdateKind::Corrected {
                corrections += 1;
                self.metrics.corrections += 1;
            }
            self.metrics.applied += 1;
            self.remember_applied(&update.task_id);
            updates.push(update);
        }
        self.last_plan_id = current.plan_id.clone();
        self.last_revision = current.revision;
        self.last_claims = current
            .claims
            .iter()
            .map(|claim| (claim.slot.clone(), claim.clone()))
            .collect();
        for (slot, observed) in &previous_claims {
            if blocked.contains(slot) {
                self.last_claims.insert(slot.clone(), observed.clone());
            }
        }
        applied.updates.extend(updates);
        ReconcileReport {
            applied,
            scheduled: 0,
            pending: self.pending_count(),
        }
    }

    /// 队列摘要；没有待办时返回空串。
    pub fn render_pending_zh(&self, limit: usize) -> String {
        if self.pending_count() == 0 {
            return String::new();
        }
        let mut lines = vec![format!("待复核 {} 项：", self.pending_count())];
        for task in self.pending.iter().take(limit) {
            lines.push(format!("  - {}", task.render_zh()));
        }
        if self.pending_count() > limit {
            lines.push(format!("  …另有 {} 项", self.pending_count() - limit));
        }
        lines.join("\n")
    }
}

fn classify_update(
    previous: &ObservedClaim,
    before_verdict: EnvelopeVerdict,
    after_verdict: EnvelopeVerdict,
    after_envelope: ClaimEnvelope,
    after_text: &str,
) -> ClaimUpdateKind {
    if previous.envelope == after_envelope
        && previous.text == after_text
        && before_verdict == after_verdict
    {
        ClaimUpdateKind::Unchanged
    } else if before_verdict == EnvelopeVerdict::Assert && after_verdict != EnvelopeVerdict::Assert
    {
        ClaimUpdateKind::Corrected
    } else if before_verdict != EnvelopeVerdict::Assert && after_verdict == EnvelopeVerdict::Assert
    {
        ClaimUpdateKind::Confirmed
    } else {
        ClaimUpdateKind::Revised
    }
}

fn task_order(left: &DeliberationTask, right: &DeliberationTask) -> std::cmp::Ordering {
    right
        .priority
        .cmp(&left.priority)
        .then_with(|| left.turn.cmp(&right.turn))
        .then_with(|| left.slot.cmp(&right.slot))
        .then_with(|| left.id.cmp(&right.id))
}

fn truncate_chars(text: &str, limit: usize) -> String {
    let count = text.chars().count();
    if count <= limit {
        return text.to_string();
    }
    let head: String = text.chars().take(limit.saturating_sub(1)).collect();
    format!("{}…", head)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::answer_plan::PlanClaim;
    use crate::confidence::{ClaimEnvelope, EnvelopeGrade};

    fn policy() -> TurnPolicy {
        TurnPolicy {
            answer_bps: 8_600,
            hedge_bps: 6_200,
            max_tasks_per_turn: 2,
            pending_capacity: 2,
            max_corrections_per_batch: 1,
        }
    }

    fn plan(revision: u64, rows: &[(&str, &str, ClaimEnvelope)]) -> AnswerPlan {
        AnswerPlan::new(
            "张三怎么样",
            "张三",
            revision,
            rows.iter()
                .map(|(slot, text, envelope)| {
                    PlanClaim::new(*slot, "关系", *text, false).with_envelope(*envelope)
                })
                .collect(),
            0,
            0,
        )
    }

    #[test]
    fn a_turn_schedules_only_non_asserted_claims_with_bounded_queue() {
        let plan = plan(
            1,
            &[
                ("s1", "张三是工程师。", ClaimEnvelope::point(0.95)),
                ("s2", "张三是李四的师傅。", ClaimEnvelope::new(4_000, 8_000)),
                ("s3", "张三住在北京。", ClaimEnvelope::new(6_000, 9_500)),
            ],
        );
        let envelope = TurnEnvelope::from_plan(1, &plan);
        let mut book = EpistemicTurnBook::new();
        let report = book.observe_turn(&envelope, &policy());
        assert_eq!(report.scheduled, 2, "每轮最多新增两个任务");
        assert_eq!(book.pending_count(), 2);
        assert!(book.pending().iter().all(|task| task.id.starts_with("et-")));
        assert!(book
            .pending()
            .iter()
            .all(|task| task.verdict != EnvelopeVerdict::Assert));

        let again = book.observe_turn(&envelope, &policy());
        assert_eq!(again.scheduled, 0, "同一计划重复观测不得重复入队");
    }

    #[test]
    fn observing_a_drop_keeps_the_old_assertion_until_review_corrects_it() {
        let old = plan(1, &[("s1", "张三住在北京。", ClaimEnvelope::point(0.95))]);
        let mut book = EpistemicTurnBook::new();
        book.observe_turn(&TurnEnvelope::from_plan(1, &old), &policy());
        let current = plan(
            1,
            &[("s1", "张三住在北京。", ClaimEnvelope::new(4_000, 7_000))],
        );
        book.observe_turn(&TurnEnvelope::from_plan(2, &current), &policy());
        assert_eq!(
            book.last_claim("s1").unwrap().envelope,
            ClaimEnvelope::point(0.95),
            "掉档要先保留旧断言，等待复核更正"
        );
        let report = book.reconcile(&TurnEnvelope::from_plan(3, &current), &policy());
        assert!(report
            .applied
            .updates
            .iter()
            .any(|update| update.kind == ClaimUpdateKind::Corrected));
        assert_eq!(
            book.last_claim("s1").unwrap().envelope,
            ClaimEnvelope::new(4_000, 7_000)
        );
    }

    #[test]
    fn stale_and_duplicate_proposals_are_rejected_without_consuming_tasks() {
        let plan = plan(
            3,
            &[("s1", "张三住在北京。", ClaimEnvelope::new(5_000, 8_000))],
        );
        let envelope = TurnEnvelope::from_plan(1, &plan);
        let mut book = EpistemicTurnBook::new();
        book.observe_turn(&envelope, &policy());
        let task = book.pending()[0].clone();
        let proposal = DeliberationProposal::new(
            &task,
            2,
            ClaimEnvelope::new(8_000, 9_500),
            "张三住在北京。",
            "用户",
        );
        let report = book.apply_proposal(proposal.clone(), 3, &policy());
        assert_eq!(report.rejections[0].reason, RejectionReason::StaleRevision);
        assert_eq!(book.pending_count(), 1);

        let fresh = DeliberationProposal::new(
            &task,
            3,
            ClaimEnvelope::point(0.95),
            "张三住在北京。",
            "用户",
        );
        let first = book.apply_proposal(fresh.clone(), 3, &policy());
        assert_eq!(first.updates[0].kind, ClaimUpdateKind::Confirmed);
        assert_eq!(book.pending_count(), 0);
        let duplicate = book.apply_proposal(fresh, 3, &policy());
        assert_eq!(duplicate.rejections[0].reason, RejectionReason::Duplicate);
    }

    #[test]
    fn a_batch_applies_at_most_one_correction_but_keeps_the_rest_pending() {
        let old = plan(
            1,
            &[
                ("s1", "张三是工程师。", ClaimEnvelope::point(0.95)),
                ("s2", "张三住在北京。", ClaimEnvelope::point(0.95)),
                ("s3", "张三是李四。", ClaimEnvelope::point(0.95)),
            ],
        );
        let mut book = EpistemicTurnBook::new();
        book.observe_turn(&TurnEnvelope::from_plan(1, &old), &policy());
        let current = plan(
            2,
            &[
                ("s1", "张三是工程师。", ClaimEnvelope::new(5_000, 7_000)),
                ("s2", "张三住在北京。", ClaimEnvelope::new(5_000, 7_000)),
                ("s3", "张三是李四。", ClaimEnvelope::new(5_000, 7_000)),
            ],
        );
        let report = book.reconcile(&TurnEnvelope::from_plan(2, &current), &policy());
        assert_eq!(
            report
                .applied
                .updates
                .iter()
                .filter(|update| update.kind == ClaimUpdateKind::Corrected)
                .count(),
            1
        );
        assert_eq!(
            report
                .applied
                .rejections
                .iter()
                .filter(|rejection| rejection.reason == RejectionReason::CorrectionBudget)
                .count(),
            2
        );
        assert_eq!(book.last_claims.len(), 3);
        assert_eq!(
            book.last_claims
                .values()
                .filter(|claim| claim.envelope == ClaimEnvelope::point(0.95))
                .count(),
            2,
            "未获预算的更正必须保留旧状态，下一轮还能补报"
        );
    }

    #[test]
    fn reconcile_uses_current_claims_and_turns_missing_ones_into_abstention() {
        let old = plan(
            1,
            &[
                ("s1", "张三是工程师。", ClaimEnvelope::new(5_000, 7_000)),
                ("s2", "张三住在北京。", ClaimEnvelope::point(0.95)),
            ],
        );
        let envelope = TurnEnvelope::from_plan(1, &old);
        let mut book = EpistemicTurnBook::new();
        book.observe_turn(&envelope, &policy());

        let current = plan(2, &[("s1", "张三是工程师。", ClaimEnvelope::point(0.95))]);
        let report = book.reconcile(&TurnEnvelope::from_plan(2, &current), &policy());
        assert_eq!(report.applied.updates.len(), 2);
        assert!(report
            .applied
            .updates
            .iter()
            .any(|update| update.kind == ClaimUpdateKind::Confirmed));
        assert!(report
            .applied
            .updates
            .iter()
            .any(|update| update.kind == ClaimUpdateKind::Corrected
                && update.after_verdict == EnvelopeVerdict::Abstain));
        assert_eq!(report.pending, 0);
    }

    #[test]
    fn book_is_serde_stable_and_empty_state_serializes_to_no_fields() {
        let empty = EpistemicTurnBook::new();
        let json = serde_json::to_string(&empty).unwrap();
        assert_eq!(json, "{}");
        let restored: EpistemicTurnBook = serde_json::from_str(&json).unwrap();
        assert!(restored.is_empty());

        let plan = plan(
            1,
            &[(
                "s1",
                "张三住在北京。",
                ClaimEnvelope::from_grade(EnvelopeGrade::SessionDirect, 0, 1),
            )],
        );
        let mut book = EpistemicTurnBook::new();
        book.observe_turn(&TurnEnvelope::from_plan(1, &plan), &policy());
        let json = serde_json::to_string(&book).unwrap();
        let restored: EpistemicTurnBook = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, book);
    }
}

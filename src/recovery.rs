//! RCV：失败恢复阶梯（Recovery Ladder）。
//!
//! 神经符号的正确用法是「双过程」：联想通道（A）提名，符号通道（B）仲裁。
//! B 失败（无证明）不是死路，而是 A 的点火信号；提名回到 B 验证
//! （propose→verify）；验证不了就降级表达并请求确认；两条都灭才承认不知道。
//!
//! 恢复阶梯（每级都有固定顺序、固定上限、固定文案）：
//!
//! ```text
//! L0 直接答     确证 → 原样输出
//! L1 换帧重解   未解析 → 换问法重查，命中即恢复为确证
//! L2 提名+验证  A 提名 → B 验证 → 推测（带证据链，可确认/否认）
//! L3 一问挂账   至多一个澄清问题，按 (实体,属性) 去重，TTL 后过期
//! L4 承认不知道 状态化措辞（确无 ≠ 查无 ≠ 引擎故障）
//! ```
//!
//! 设计纪律（纯符号，零训练）：
//! - 阈值不做学习：提名最低分固定为 0.45，候选有上限；
//! - 排序定点化（score × 10⁶ 取整）后按 (分数降序, 值升序)，同输入必同序；
//! - 未经验证的提名永不以肯定语气输出、永不写图谱；
//! - 澄清每轮至多一个，与帧澄清共用「问题预算」的克制原则；
//! - EngineFault 与 Unknown 永不合并（见 [`crate::epistemic`]）。
//!
//! 三档模式与提示资产同构：[`RecoveryMode::Off`] 原样输出、
//! [`RecoveryMode::Observe`] 只记台账不改文本、[`RecoveryMode::On`] 完整阶梯。

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::epistemic::{
    render_statement, EpistemicStatus, FailureLedger, FailureReceipt, ReceiptResolution,
};

/// L2 候选上限。
pub const MAX_NOMINATIONS: usize = 3;
/// L2 提名最低分（与对比仲裁同口径：低于此分不进入验证）。
pub const MIN_NOMINATION_SCORE: f64 = 0.45;
/// 待定澄清 / 推测的有效轮数（TTL）。
pub const CLARIFY_TTL_TURNS: u64 = 3;
/// 同时挂账的问题上限。
pub const MAX_PENDING: usize = 8;
/// 定点化标尺（排序确定性）。
pub const SCORE_FIXED_SCALE: f64 = 1_000_000.0;

/// 恢复三档模式（与提示资产 Off / Observe / On 同构）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecoveryMode {
    /// 原样输出，不做恢复处理（默认）。
    Off,
    /// 只记失败台账，不改动进程文本（先收集真实失败分布）。
    Observe,
    /// 完整恢复阶梯 + 状态化措辞。
    On,
}

impl Default for RecoveryMode {
    fn default() -> Self {
        RecoveryMode::Off
    }
}

impl RecoveryMode {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_lowercase().as_str() {
            "off" => Some(RecoveryMode::Off),
            "observe" => Some(RecoveryMode::Observe),
            "on" => Some(RecoveryMode::On),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            RecoveryMode::Off => "off",
            RecoveryMode::Observe => "observe",
            RecoveryMode::On => "on",
        }
    }

    /// 是否允许执行恢复阶梯的联想/换帧查询（Observe 只记账，不查）。
    pub fn acts(self) -> bool {
        matches!(self, RecoveryMode::On)
    }

    pub fn records(self) -> bool {
        !matches!(self, RecoveryMode::Off)
    }
}

/// 恢复阶梯档位。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecoveryLevel {
    Direct,
    Reframe,
    Associate,
    Clarify,
    Admit,
}

impl RecoveryLevel {
    pub fn label(self) -> &'static str {
        match self {
            RecoveryLevel::Direct => "L0",
            RecoveryLevel::Reframe => "L1",
            RecoveryLevel::Associate => "L2",
            RecoveryLevel::Clarify => "L3",
            RecoveryLevel::Admit => "L4",
        }
    }
}

/// A 通道的一条提名（附证据链，逐字可回溯）。
#[derive(Debug, Clone, PartialEq)]
pub struct RecoveryNomination {
    pub value: String,
    /// 联想通道得分（定点化后排序，避免浮点序不确定）。
    pub score: f64,
    pub evidence: Vec<String>,
}

/// B 通道对提名的验证结论。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationOutcome {
    /// 符号通道确认（仍然只是「有支持」，输出定性为推测）。
    Confirmed,
    /// 符号通道反驳（转为争议披露）。
    Refuted,
    /// 符号通道无结论（降级表达，请求确认）。
    NoVerdict,
}

/// L1 换帧重解的一次尝试。
#[derive(Debug, Clone, PartialEq)]
pub struct ReframedReading {
    pub question: String,
    pub answer: Option<String>,
}

/// 恢复所需的知识通道（由调用方的知识源实现；缺省全部空 = 直接到 L4）。
pub trait RecoveryKnowledge {
    /// L1：换一种问法重新解析（返回按 (问题, 答案) 的确定性序列）。
    fn reframe(&self, entity: &str, attribute: &str) -> Vec<ReframedReading> {
        let _ = (entity, attribute);
        Vec::new()
    }

    /// L2：联想通道提名（调用方负责确定给定顺序，本模块再排序）。
    fn nominations(&self, entity: &str, attribute: &str) -> Vec<RecoveryNomination> {
        let _ = (entity, attribute);
        Vec::new()
    }

    /// L2：符号通道验证一条提名。
    fn verify(&self, entity: &str, attribute: &str, value: &str) -> VerificationOutcome {
        let _ = (entity, attribute, value);
        VerificationOutcome::NoVerdict
    }

    /// L3：实体可追问的相关线索（按确定性序）。
    fn related(&self, entity: &str) -> Vec<String> {
        let _ = entity;
        Vec::new()
    }
}

/// 挂账问题的类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PendingKind {
    /// 清澄问题（等用户提供事实）。
    Clarification,
    /// 推测确认（等用户 是/否 裁定）。
    Provisional,
}

/// 一条挂账问题（有 TTL、可裁定、随快照持久化）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PendingFollowup {
    pub receipt_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entity: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attribute: Option<String>,
    pub question: String,
    pub kind: PendingKind,
    pub status: EpistemicStatus,
    pub created_tick: u64,
}

/// 恢复引擎的会话状态（随对话快照持久化，保证状态充分性）。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecoveryState {
    /// 逻辑轮次水位（TTL 计量单位，无挂钟）。
    #[serde(default)]
    pub tick: u64,
    #[serde(default)]
    pub ledger: FailureLedger,
    #[serde(default)]
    pub pending: BTreeMap<String, PendingFollowup>,
}

impl RecoveryState {
    pub fn new() -> Self {
        Self::default()
    }

    /// 挂账键：实体 + 属性（确定性、可去重）。
    pub fn pending_key(entity: Option<&str>, attribute: Option<&str>) -> String {
        format!("{}\u{1}{}", entity.unwrap_or(""), attribute.unwrap_or(""))
    }

    /// 推进一轮：过期挂账随 TTL 清除（确定性）。
    pub fn begin_turn(&mut self) {
        self.tick += 1;
        let tick = self.tick;
        self.pending
            .retain(|_, followup| tick.saturating_sub(followup.created_tick) < CLARIFY_TTL_TURNS);
    }

    /// 最近挂账（新 → 旧中的最新；并列按会话键升序，保证确定性）。
    pub fn latest_pending(&self) -> Option<&PendingFollowup> {
        self.pending
            .values()
            .max_by(|left, right| {
                left.created_tick
                    .cmp(&right.created_tick)
                    .then_with(|| left.question.cmp(&right.question).reverse())
            })
    }
}

/// 一次待恢复失败的输入描述。
#[derive(Debug, Clone)]
pub struct RecoveryRequest<'a> {
    pub status: EpistemicStatus,
    pub entity: Option<&'a str>,
    pub attribute: Option<&'a str>,
    /// 失败细节（引擎错误摘要 / 解析失败说明），会进回执。
    pub detail: &'a str,
}

impl<'a> RecoveryRequest<'a> {
    pub fn new(status: EpistemicStatus, entity: Option<&'a str>, attribute: Option<&'a str>) -> Self {
        RecoveryRequest {
            status,
            entity,
            attribute,
            detail: "",
        }
    }
}

/// 恢复结果：状态 + 档位 + 文本 + 证据 + 待确认问题 + 回执 id。
#[derive(Debug, Clone, PartialEq)]
pub struct RecoveryAttempt {
    pub status: EpistemicStatus,
    pub level: RecoveryLevel,
    pub text: String,
    pub evidence: Vec<String>,
    pub question: Option<String>,
    pub receipt_id: Option<String>,
    /// 恢复成功时给出的值（换帧答案 / 推测值）。
    pub value: Option<String>,
}

impl RecoveryAttempt {
    fn plain(status: EpistemicStatus, level: RecoveryLevel) -> Self {
        RecoveryAttempt {
            status,
            level,
            text: String::new(),
            evidence: Vec::new(),
            question: None,
            receipt_id: None,
            value: None,
        }
    }

    /// 是否在恢复阶梯中把失败救回来了（最终为确证）。
    pub fn recovered(&self) -> bool {
        !self.status.is_failure()
    }
}

/// 定点化分数键（非有限值视为最低）。
fn score_key(score: f64) -> i64 {
    if !score.is_finite() {
        return i64::MIN;
    }
    (score * SCORE_FIXED_SCALE).round() as i64
}

/// 提名排序 + 去重 + 截断：分数降序（定点）、值升序；低于阈值丢弃。
fn prepare_nominations(mut nominations: Vec<RecoveryNomination>) -> Vec<RecoveryNomination> {
    nominations.retain(|nomination| {
        !nomination.value.trim().is_empty()
            && score_key(nomination.score) >= score_key(MIN_NOMINATION_SCORE)
    });
    nominations.sort_by(|left, right| {
        score_key(right.score)
            .cmp(&score_key(left.score))
            .then_with(|| left.value.cmp(&right.value))
    });
    nominations.dedup_by(|left, right| left.value == right.value);
    nominations.truncate(MAX_NOMINATIONS);
    nominations
}

/// 组合正文：基础措辞 + 证据链 + 追问（固定骨架，无生成）。
fn compose(base: String, evidence: &[String], question: Option<&str>) -> String {
    let mut text = base;
    if !evidence.is_empty() {
        text.push_str(&format!("\n依据：{}", evidence.join("；")));
    }
    if let Some(question) = question {
        if !question.is_empty() {
            text.push('\n');
            text.push_str(question);
        }
    }
    text
}

/// 构造一条回执（id 由台账分配）。
fn build_receipt(
    state: &RecoveryState,
    request: &RecoveryRequest<'_>,
    level: RecoveryLevel,
    evidence: Vec<String>,
    question: Option<String>,
    resolution: Option<ReceiptResolution>,
) -> FailureReceipt {
    FailureReceipt {
        id: String::new(),
        turn: state.tick,
        status: request.status,
        entity: request.entity.map(str::to_string),
        attribute: request.attribute.map(str::to_string),
        level: level.label().to_string(),
        detail: request.detail.to_string(),
        evidence,
        question,
        resolution,
    }
}

/// 执行恢复阶梯。返回的 `text` 仅在 [`RecoveryMode::On`] 下由调用方采用；
/// `Observe` 只入台账，`Off` 不产生任何副作用。
pub fn attempt<K: RecoveryKnowledge + ?Sized>(
    state: &mut RecoveryState,
    mode: RecoveryMode,
    request: &RecoveryRequest<'_>,
    knowledge: &K,
) -> RecoveryAttempt {
    if !mode.records() {
        return RecoveryAttempt::plain(request.status, RecoveryLevel::Direct);
    }

    match request.status {
        EpistemicStatus::Certain => RecoveryAttempt::plain(EpistemicStatus::Certain, RecoveryLevel::Direct),
        EpistemicStatus::EngineFault => {
            let mut attempt = RecoveryAttempt::plain(EpistemicStatus::EngineFault, RecoveryLevel::Admit);
            attempt.text = render_statement(EpistemicStatus::EngineFault, None, None, None);
            let receipt = build_receipt(state, request, RecoveryLevel::Admit, Vec::new(), None, None);
            attempt.receipt_id = Some(state.ledger.record(receipt));
            attempt
        }
        EpistemicStatus::Unparseable => recover_unparseable(state, mode, request, knowledge),
        EpistemicStatus::Absent | EpistemicStatus::Unknown => {
            recover_missing(state, mode, request, knowledge)
        }
        // 推测 / 冲突由恢复引擎自己产出，不作为输入。
        EpistemicStatus::Provisional | EpistemicStatus::Conflict => {
            let mut attempt = RecoveryAttempt::plain(request.status, RecoveryLevel::Admit);
            attempt.text = render_statement(request.status, request.entity, request.attribute, None);
            attempt
        }
    }
}

/// L1：换帧重解；失败则按未解析状态入账（text 交由调用方决定是否采用）。
fn recover_unparseable<K: RecoveryKnowledge + ?Sized>(
    state: &mut RecoveryState,
    mode: RecoveryMode,
    request: &RecoveryRequest<'_>,
    knowledge: &K,
) -> RecoveryAttempt {
    let (entity, attribute) = match (request.entity, request.attribute) {
        (Some(entity), Some(attribute)) if !entity.is_empty() && !attribute.is_empty() => {
            (entity, attribute)
        }
        _ => {
            let mut attempt = RecoveryAttempt::plain(EpistemicStatus::Unparseable, RecoveryLevel::Admit);
            attempt.text = render_statement(EpistemicStatus::Unparseable, None, None, None);
            let receipt = build_receipt(state, request, RecoveryLevel::Admit, Vec::new(), None, None);
            attempt.receipt_id = Some(state.ledger.record(receipt));
            return attempt;
        }
    };

    if mode.acts() {
        let mut readings = knowledge.reframe(entity, attribute);
        readings.sort_by(|left, right| left.question.cmp(&right.question));
        if let Some(recovered) = readings
            .iter()
            .find(|reading| reading.answer.as_deref().map(str::trim).is_some_and(|text| !text.is_empty()))
        {
            let answer = recovered.answer.clone().unwrap_or_default();
            let evidence = vec![format!("换帧重解：{}", recovered.question)];
            let receipt = build_receipt(
                state,
                request,
                RecoveryLevel::Reframe,
                evidence.clone(),
                None,
                Some(ReceiptResolution::Acknowledged),
            );
            let receipt_id = state.ledger.record(receipt);
            return RecoveryAttempt {
                status: EpistemicStatus::Certain,
                level: RecoveryLevel::Reframe,
                text: answer.clone(),
                evidence,
                question: None,
                receipt_id: Some(receipt_id),
                value: Some(answer),
            };
        }
    }

    let mut attempt = RecoveryAttempt::plain(EpistemicStatus::Unparseable, RecoveryLevel::Reframe);
    attempt.text = render_statement(EpistemicStatus::Unparseable, None, None, None);
    let receipt = build_receipt(state, request, RecoveryLevel::Reframe, Vec::new(), None, None);
    attempt.receipt_id = Some(state.ledger.record(receipt));
    attempt
}

/// L2 → L3 → L4：缺记录时的恢复。
fn recover_missing<K: RecoveryKnowledge + ?Sized>(
    state: &mut RecoveryState,
    mode: RecoveryMode,
    request: &RecoveryRequest<'_>,
    knowledge: &K,
) -> RecoveryAttempt {
    let status = request.status;
    let base = render_statement(status, request.entity, request.attribute, None);

    // Observe：只记账，不查联想通道（文本由调用方按原样输出）。
    if !mode.acts() {
        let mut attempt = RecoveryAttempt::plain(status, RecoveryLevel::Admit);
        attempt.text = base;
        let receipt = build_receipt(state, request, RecoveryLevel::Admit, Vec::new(), None, None);
        attempt.receipt_id = Some(state.ledger.record(receipt));
        return attempt;
    }

    let (Some(entity), Some(attribute)) = (request.entity, request.attribute) else {
        // 身份问句 / 无属性：没有可验证的提名，直接 L4。
        let mut attempt = RecoveryAttempt::plain(status, RecoveryLevel::Admit);
        attempt.text = base;
        let receipt = build_receipt(state, request, RecoveryLevel::Admit, Vec::new(), None, None);
        attempt.receipt_id = Some(state.ledger.record(receipt));
        return attempt;
    };

    // L2：A 提名 → B 验证。
    let nominations = prepare_nominations(knowledge.nominations(entity, attribute));
    if !nominations.is_empty() {
        let mut chosen: Option<(&RecoveryNomination, VerificationOutcome)> = None;
        for nomination in &nominations {
            let outcome = knowledge.verify(entity, attribute, &nomination.value);
            if matches!(outcome, VerificationOutcome::Confirmed) {
                chosen = Some((nomination, outcome));
                break;
            }
            if chosen.is_none() {
                chosen = Some((nomination, outcome));
            }
        }
        let (nomination, outcome) = chosen.expect("提名非空");
        let mut evidence = nomination.evidence.clone();
        let (final_status, receipt_status) = match outcome {
            VerificationOutcome::Confirmed => {
                evidence.push("符号通道验证：有支持".to_string());
                (EpistemicStatus::Provisional, EpistemicStatus::Provisional)
            }
            VerificationOutcome::Refuted => {
                evidence.push("符号通道验证：被反驳".to_string());
                (EpistemicStatus::Conflict, EpistemicStatus::Conflict)
            }
            VerificationOutcome::NoVerdict => {
                evidence.push("符号通道验证：无结论".to_string());
                (EpistemicStatus::Provisional, EpistemicStatus::Provisional)
            }
        };
        let question = "是这样吗？如果不对，请直接纠正我。".to_string();
        let text = compose(
            render_statement(final_status, request.entity, request.attribute, Some(&nomination.value)),
            &evidence,
            Some(&question),
        );
        // 入账（回执状态反映对外的认识论定性）。
        let mut receipt_request = request.clone();
        receipt_request.status = receipt_status;
        let receipt = build_receipt(
            state,
            &receipt_request,
            RecoveryLevel::Associate,
            evidence.clone(),
            Some(question.clone()),
            None,
        );
        let receipt_id = state.ledger.record(receipt);
        push_pending(
            state,
            entity,
            attribute,
            &receipt_id,
            &question,
            PendingKind::Provisional,
            receipt_status,
        );
        return RecoveryAttempt {
            status: final_status,
            level: RecoveryLevel::Associate,
            text,
            evidence,
            question: Some(question),
            receipt_id: Some(receipt_id),
            value: Some(nomination.value.clone()),
        };
    }

    // L3：至多一个澄清问题（按 (实体,属性) 去重 + TTL + 挂账上限）。
    let key = RecoveryState::pending_key(Some(entity), Some(attribute));
    let already_pending = state.pending.contains_key(&key);
    if !already_pending && state.pending.len() < MAX_PENDING {
        let question = clarification_question(entity, attribute, knowledge);
        let mut attempt = RecoveryAttempt::plain(status, RecoveryLevel::Clarify);
        attempt.text = compose(base, &[], Some(&question));
        let receipt = build_receipt(
            state,
            request,
            RecoveryLevel::Clarify,
            Vec::new(),
            Some(question.clone()),
            None,
        );
        let receipt_id = state.ledger.record(receipt);
        push_pending(
            state,
            entity,
            attribute,
            &receipt_id,
            &question,
            PendingKind::Clarification,
            status,
        );
        attempt.receipt_id = Some(receipt_id);
        attempt.question = Some(question);
        return attempt;
    }

    // L4：承认不知道（状态化措辞）。
    let mut attempt = RecoveryAttempt::plain(status, RecoveryLevel::Admit);
    attempt.text = base;
    let receipt = build_receipt(state, request, RecoveryLevel::Admit, Vec::new(), None, None);
    attempt.receipt_id = Some(state.ledger.record(receipt));
    attempt
}

/// L3 澄清问题：缺属性时问该属性；若有相关线索，顺带列一条（确定性取首个）。
fn clarification_question<K: RecoveryKnowledge + ?Sized>(
    entity: &str,
    attribute: &str,
    knowledge: &K,
) -> String {
    let related = knowledge.related(entity);
    match related.first() {
        Some(related) if related != attribute => format!(
            "你能帮我确认一下「{}」的「{}」吗？你知道的线索（例如「{}」相关）也可以直接告诉我，我会记进会话。",
            entity, attribute, related
        ),
        _ => format!(
            "你能帮我确认一下「{}」的「{}」吗？告诉我答案，我会记进会话。",
            entity, attribute
        ),
    }
}

/// 挂账（同键覆盖：新问题替代旧问题）。
fn push_pending(
    state: &mut RecoveryState,
    entity: &str,
    attribute: &str,
    receipt_id: &str,
    question: &str,
    kind: PendingKind,
    status: EpistemicStatus,
) {
    let key = RecoveryState::pending_key(Some(entity), Some(attribute));
    state.pending.insert(
        key,
        PendingFollowup {
            receipt_id: receipt_id.to_string(),
            entity: Some(entity.to_string()),
            attribute: Some(attribute.to_string()),
            question: question.to_string(),
            kind,
            status,
            created_tick: state.tick,
        },
    );
}

/// 用户以「是/不是」回应最近的挂账：结算回执、关闭挂账，返回回执文本。
///
/// 澄清型挂账不接受 是/否 作为答案（继续等具体事实，不关闭）。
pub fn resolve_by_yes(state: &mut RecoveryState, confirmed: bool) -> Option<String> {
    let key = {
        let latest = state.latest_pending()?;
        RecoveryState::pending_key(latest.entity.as_deref(), latest.attribute.as_deref())
    };
    let followup = state.pending.remove(&key)?;
    let resolution = if confirmed {
        ReceiptResolution::Confirmed
    } else {
        ReceiptResolution::Refuted
    };
    match followup.kind {
        PendingKind::Provisional => {
            state.ledger.resolve(&followup.receipt_id, resolution);
            Some(if confirmed {
                format!("好的，「{}」我记下了——它是会话级线索，不会写入图谱。", followup.question)
            } else {
                "明白，这条推测作废，我不会再拿它当依据。".to_string()
            })
        }
        PendingKind::Clarification => {
            // 重新挂回（等具体事实），不结算回执。
            state.pending.insert(key, followup.clone());
            Some(format!("我还在等你的具体答案：{}", followup.question))
        }
    }
}

/// 用户提供了具体线索：关闭最近一条澄清挂账并回执。
pub fn resolve_clarification(state: &mut RecoveryState, answer: &str) -> Option<String> {
    let key = {
        let latest = state.latest_pending()?;
        if latest.kind != PendingKind::Clarification {
            return None;
        }
        RecoveryState::pending_key(latest.entity.as_deref(), latest.attribute.as_deref())
    };
    let followup = state.pending.remove(&key)?;
    state
        .ledger
        .resolve(&followup.receipt_id, ReceiptResolution::Acknowledged);
    Some(format!(
        "谢谢，我把「{}」这条线索记在会话里了（{}）。",
        answer.trim(),
        followup.question
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockKnowledge {
        reframed: Vec<ReframedReading>,
        nominations: Vec<RecoveryNomination>,
        verdict: VerificationOutcome,
        related: Vec<String>,
    }

    impl Default for MockKnowledge {
        fn default() -> Self {
            MockKnowledge {
                reframed: Vec::new(),
                nominations: Vec::new(),
                verdict: VerificationOutcome::NoVerdict,
                related: Vec::new(),
            }
        }
    }

    impl RecoveryKnowledge for MockKnowledge {
        fn reframe(&self, _entity: &str, _attribute: &str) -> Vec<ReframedReading> {
            self.reframed.clone()
        }

        fn nominations(&self, _entity: &str, _attribute: &str) -> Vec<RecoveryNomination> {
            self.nominations.clone()
        }

        fn verify(&self, _entity: &str, _attribute: &str, _value: &str) -> VerificationOutcome {
            self.verdict
        }

        fn related(&self, _entity: &str) -> Vec<String> {
            self.related.clone()
        }
    }

    fn request<'a>(status: EpistemicStatus, entity: Option<&'a str>, attribute: Option<&'a str>) -> RecoveryRequest<'a> {
        RecoveryRequest::new(status, entity, attribute)
    }

    #[test]
    fn modes_parse_and_off_is_noop() {
        assert_eq!(RecoveryMode::parse("off"), Some(RecoveryMode::Off));
        assert_eq!(RecoveryMode::parse("OBSERVE"), Some(RecoveryMode::Observe));
        assert_eq!(RecoveryMode::parse(" on "), Some(RecoveryMode::On));
        assert_eq!(RecoveryMode::parse("bogus"), None);
        assert!(!RecoveryMode::Observe.acts());
        assert!(RecoveryMode::On.acts());

        let mut state = RecoveryState::new();
        let attempt = attempt(
            &mut state,
            RecoveryMode::Off,
            &request(EpistemicStatus::Absent, Some("姚明"), Some("体重")),
            &MockKnowledge::default(),
        );
        assert!(state.ledger.is_empty(), "Off 不得留下任何台账");
        assert!(attempt.status.is_failure());
    }

    #[test]
    fn reframe_can_rescue_unparseable() {
        let mut state = RecoveryState::new();
        let knowledge = MockKnowledge {
            reframed: vec![
                ReframedReading {
                    question: "姚明的身高是多少".to_string(),
                    answer: Some("姚明的身高是 226。".to_string()),
                },
                ReframedReading {
                    question: "姚明多高".to_string(),
                    answer: None,
                },
            ],
            ..Default::default()
        };
        let mut state_observer = RecoveryState::new();
        let observed = attempt(
            &mut state_observer,
            RecoveryMode::Observe,
            &request(EpistemicStatus::Unparseable, Some("姚明"), Some("身高")),
            &knowledge,
        );
        assert!(observed.status.is_failure(), "Observe 不改确定性");
        assert_eq!(state_observer.ledger.len(), 1);

        let rescued = attempt(
            &mut state,
            RecoveryMode::On,
            &request(EpistemicStatus::Unparseable, Some("姚明"), Some("身高")),
            &knowledge,
        );
        assert!(!rescued.status.is_failure(), "换帧命中应恢复为确证");
        assert_eq!(rescued.level, RecoveryLevel::Reframe);
        assert!(rescued.text.contains("226"));
        assert!(rescued.evidence[0].contains("换帧重解"));
    }

    #[test]
    fn nominations_are_promoted_as_provisional_with_evidence() {
        let knowledge = MockKnowledge {
            nominations: vec![
                RecoveryNomination {
                    value: "90".to_string(),
                    score: 0.5,
                    evidence: vec!["图游走：姚明—关联—叶莉".to_string()],
                },
                RecoveryNomination {
                    value: "80".to_string(),
                    score: 0.9,
                    evidence: vec!["链接预测：身高".to_string()],
                },
                RecoveryNomination {
                    value: "70".to_string(),
                    score: 0.1,
                    evidence: vec!["低于阈值".to_string()],
                },
            ],
            verdict: VerificationOutcome::NoVerdict,
            ..Default::default()
        };
        let mut state = RecoveryState::new();
        let attempt = attempt(
            &mut state,
            RecoveryMode::On,
            &request(EpistemicStatus::Absent, Some("姚明"), Some("体重")),
            &knowledge,
        );
        assert_eq!(attempt.status, EpistemicStatus::Provisional);
        assert_eq!(attempt.level, RecoveryLevel::Associate);
        assert!(
            attempt.text.contains("推测"),
            "未经符号验证不得以肯定语气：{}",
            attempt.text
        );
        assert!(attempt.text.contains("依据"), "{}", attempt.text);
        assert!(
            attempt.evidence[0].contains("链接预测"),
            "高分提名应排在前：{:?}",
            attempt.evidence
        );
        assert!(!attempt.evidence.iter().any(|line| line.contains("低于阈值")));
        assert_eq!(state.ledger.open_count(), 1);
        assert!(state.latest_pending().is_some());

        // 用户否认：回执关闭，推测作废
        let reply = resolve_by_yes(&mut state, false).unwrap();
        assert!(reply.contains("作废"));
        assert_eq!(state.ledger.open_count(), 0);
        assert!(state.latest_pending().is_none());
    }

    #[test]
    fn refuted_nomination_becomes_conflict() {
        let knowledge = MockKnowledge {
            nominations: vec![RecoveryNomination {
                value: "90".to_string(),
                score: 0.8,
                evidence: vec!["链接预测：体重".to_string()],
            }],
            verdict: VerificationOutcome::Refuted,
            ..Default::default()
        };
        let mut state = RecoveryState::new();
        let attempt = attempt(
            &mut state,
            RecoveryMode::On,
            &request(EpistemicStatus::Absent, Some("姚明"), Some("体重")),
            &knowledge,
        );
        assert_eq!(attempt.status, EpistemicStatus::Conflict);
        assert!(attempt.text.contains("矛盾"), "{}", attempt.text);
    }

    #[test]
    fn clarification_budget_is_one_per_gap_with_ttl() {
        let mut state = RecoveryState::new();
        let knowledge = MockKnowledge {
            related: vec!["身高".to_string()],
            ..Default::default()
        };
        let first = attempt(
            &mut state,
            RecoveryMode::On,
            &request(EpistemicStatus::Absent, Some("姚明"), Some("体重")),
            &knowledge,
        );
        assert_eq!(first.level, RecoveryLevel::Clarify);
        assert!(first.text.contains("姚明") && first.text.contains("体重"));
        assert!(first.question.is_some());
        let first_question = first.question.clone().unwrap();

        // 同缺口重复提问：不挂第二个问题，降级为 L4 状态措辞
        let second = attempt(
            &mut state,
            RecoveryMode::On,
            &request(EpistemicStatus::Absent, Some("姚明"), Some("体重")),
            &knowledge,
        );
        assert_eq!(second.level, RecoveryLevel::Admit);
        assert!(!second.text.contains(&first_question));

        // TTL 到期后可以再问一次
        state.begin_turn();
        state.begin_turn();
        state.begin_turn();
        assert!(state.pending.is_empty(), "超过 TTL 的挂账应被清除");
        let third = attempt(
            &mut state,
            RecoveryMode::On,
            &request(EpistemicStatus::Absent, Some("姚明"), Some("体重")),
            &knowledge,
        );
        assert_eq!(third.level, RecoveryLevel::Clarify);
    }

    #[test]
    fn engine_fault_is_recorded_once_and_never_says_no_record() {
        let mut state = RecoveryState::new();
        let mut request = request(EpistemicStatus::EngineFault, Some("姚明"), Some("身高"));
        request.detail = "consult failed: boom";
        let first = attempt(&mut state, RecoveryMode::On, &request, &MockKnowledge::default());
        let second = attempt(&mut state, RecoveryMode::On, &request, &MockKnowledge::default());
        assert!(!first.text.contains("没有记录"));
        assert!(first.text.contains("出错了"));
        assert_eq!(state.ledger.len(), 1, "同路径故障只报一次");
        assert_eq!(first.receipt_id, second.receipt_id);
        assert!(state
            .ledger
            .last()
            .unwrap()
            .detail
            .contains("consult failed"));
    }

    #[test]
    fn observe_records_distribution_without_actions() {
        let mut state = RecoveryState::new();
        let knowledge = MockKnowledge {
            nominations: vec![RecoveryNomination {
                value: "90".to_string(),
                score: 0.9,
                evidence: vec!["x".to_string()],
            }],
            ..Default::default()
        };
        let observed = attempt(
            &mut state,
            RecoveryMode::Observe,
            &request(EpistemicStatus::Unknown, Some("姚明"), Some("体重")),
            &knowledge,
        );
        assert_eq!(observed.level, RecoveryLevel::Admit, "Observe 不点火联想通道");
        assert_eq!(state.ledger.status_counts(), vec![(EpistemicStatus::Unknown, 1)]);
        assert!(state.pending.is_empty());
    }

    #[test]
    fn recovery_determinism_digest() {
        let knowledge = MockKnowledge {
            nominations: vec![
                RecoveryNomination {
                    value: "a".to_string(),
                    score: 0.5,
                    evidence: vec!["e1".to_string()],
                },
                RecoveryNomination {
                    value: "b".to_string(),
                    score: 0.5,
                    evidence: vec!["e2".to_string()],
                },
            ],
            ..Default::default()
        };
        let run = || {
            let mut state = RecoveryState::new();
            let attempt = attempt(
                &mut state,
                RecoveryMode::On,
                &request(EpistemicStatus::Absent, Some("E"), Some("P")),
                &knowledge,
            );
            (state.ledger.digest(), attempt.text)
        };
        assert_eq!(run(), run());
    }
}

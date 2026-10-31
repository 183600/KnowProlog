//! CLM：命题生命周期与影响面（Claim Lifecycle & Impact）。
//!
//! 现有模块已经能回答「这条回答依据哪些事实」（`provenance` / `answer_plan`）
//! 和「过去某条回答相对于事件流是否可信」（`history_trust`），但缺少一个
//! **提交前**的统一对象：这条即将说出口的断言，是否仍然有活跃支撑？支撑被
//! 撤回、遗忘或过期后，哪些旧计划、旧回答和派生断言会一起失效？
//!
//! 本模块把 claim 而不是 fact 当作生命周期单位：
//!
//! - **身份**：规范结构（主语 / 谓词 / 生成器槽位 / 是否会话事实）+ 排序去重
//!   后的支撑集合。换个说法不改变槽位，仍可落到同一条 claim；换一组证据则
//!   得到新的 claim，必须重新裁决。
//! - **状态**：`planned → emitted → stale / retracted / superseded / expired /
//!   redacted`。终止态不可复活；`stale` 只有在支撑重新核验后才回到 `emitted`。
//! - **谱系**：支撑可以是事实，也可以是父 claim。事实退役与父 claim 退役沿
//!   反向索引传播到所有后代；这里采用保守的 **AND 语义**：任一必需支撑失效，
//!   依赖它的断言就失效，宁可多撤，不把派生结论洗白。
//! - **闸门**：生成前逐 claim 返回 `allow / refresh / deny`。未知 claim、无证据、
//!   链损坏一律不给 `allow`；终止态与墓碑支撑直接 `deny`。
//! - **审计**：只对事件日志做投影，给出 `clean / findings / unknown` 三态。
//!   无证据、无记录、链损坏或重放窗口截断都判 `unknown`，绝不伪装成 clean。
//!
//! 设计边界：不引入概率、训练或外部依赖；不把 claim 生命周期写进知识图谱；
//! 事件链沿用规范 JSON + FNV-1a 64 做完整性自检，不声称密码学安全；编年史
//! 历史字节不可擦除，本模块删除的是**活跃表面**上的可再生成断言。

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use std::io::Write as _;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::answer_plan::{AnswerPlan, ClaimEvidence, PlanClaim};
use crate::context_image::{canonical_json, fnv1a_hex};
use crate::privacy_flow::PrivacyLedger;

/// 生命周期账本的当前结构版本。
pub const CLAIM_LEDGER_VERSION: u32 = 1;

/// 退役原因文本的最大字符数，避免异常输入撑大事件日志。
pub const MAX_REASON_CHARS: usize = 240;

/// 规范 claim 标识前缀。
pub const CLAIM_ID_PREFIX: &str = "clm_";

fn current_version() -> u32 {
    CLAIM_LEDGER_VERSION
}

fn is_false(value: &bool) -> bool {
    !*value
}

fn bounded_reason(reason: &str) -> String {
    let reason = reason.trim();
    if reason.chars().count() <= MAX_REASON_CHARS {
        return reason.to_string();
    }
    let mut bounded = reason.chars().take(MAX_REASON_CHARS).collect::<String>();
    bounded.push('…');
    bounded
}

/// 用于回退身份的规范文本摘要：去掉空白与常见句读，减少排版差异。
fn canonical_statement(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    for ch in text.chars() {
        if ch.is_whitespace() {
            continue;
        }
        if matches!(
            ch,
            '，' | '。'
                | '！'
                | '？'
                | '；'
                | '：'
                | '、'
                | ','
                | '.'
                | '!'
                | '?'
                | ';'
                | ':'
                | '“'
                | '”'
                | '‘'
                | '’'
                | '"'
                | '\''
                | '（'
                | '）'
                | '('
                | ')'
                | '【'
                | '】'
                | '['
                | ']'
                | '《'
                | '》'
        ) {
            continue;
        }
        for lower in ch.to_lowercase() {
            output.push(lower);
        }
    }
    output
}

/// claim 的生命周期状态。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimStatus {
    /// 已登记、尚未说出口。
    Planned,
    /// 已说出口，仍可复述。
    Emitted,
    /// 支撑可能已变化，必须重新核验后才能再说。
    Stale,
    /// 支撑被撤回。
    Retracted,
    /// 被更新版本取代。
    Superseded,
    /// 支撑超过有效期。
    Expired,
    /// 支撑含有被遗忘内容。
    Redacted,
}

impl ClaimStatus {
    pub fn code(self) -> &'static str {
        match self {
            ClaimStatus::Planned => "planned",
            ClaimStatus::Emitted => "emitted",
            ClaimStatus::Stale => "stale",
            ClaimStatus::Retracted => "retracted",
            ClaimStatus::Superseded => "superseded",
            ClaimStatus::Expired => "expired",
            ClaimStatus::Redacted => "redacted",
        }
    }

    pub fn label_zh(self) -> &'static str {
        match self {
            ClaimStatus::Planned => "待说",
            ClaimStatus::Emitted => "已说",
            ClaimStatus::Stale => "待复核",
            ClaimStatus::Retracted => "已撤回",
            ClaimStatus::Superseded => "已被取代",
            ClaimStatus::Expired => "已过期",
            ClaimStatus::Redacted => "已遗忘",
        }
    }

    /// 是否仍被视为活跃（可被复述或等待复核）。
    pub fn is_active(self) -> bool {
        matches!(
            self,
            ClaimStatus::Planned | ClaimStatus::Emitted | ClaimStatus::Stale
        )
    }

    /// 是否终止：终止态不可复活。
    pub fn is_terminal(self) -> bool {
        !self.is_active()
    }

    fn from_retirement(kind: RetirementKind) -> Self {
        match kind {
            RetirementKind::Retracted => ClaimStatus::Retracted,
            RetirementKind::Superseded => ClaimStatus::Superseded,
            RetirementKind::Expired => ClaimStatus::Expired,
            RetirementKind::Redacted => ClaimStatus::Redacted,
        }
    }
}

/// 事实或 claim 的退役种类。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetirementKind {
    /// 不再相信这条断言，但它当时可能为真。
    Retracted,
    /// 被新的断言取代。
    Superseded,
    /// 超过有效期。
    Expired,
    /// 因遗忘 / 墓碑而不得再出现。
    Redacted,
}

impl RetirementKind {
    pub fn code(self) -> &'static str {
        match self {
            RetirementKind::Retracted => "retracted",
            RetirementKind::Superseded => "superseded",
            RetirementKind::Expired => "expired",
            RetirementKind::Redacted => "redacted",
        }
    }

    pub fn label_zh(self) -> &'static str {
        match self {
            RetirementKind::Retracted => "撤回",
            RetirementKind::Superseded => "取代",
            RetirementKind::Expired => "过期",
            RetirementKind::Redacted => "遗忘",
        }
    }

    fn rank(self) -> u8 {
        match self {
            RetirementKind::Redacted => 4,
            RetirementKind::Retracted => 3,
            RetirementKind::Expired => 2,
            RetirementKind::Superseded => 1,
        }
    }
}

/// 事实自身的生命周期状态。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FactStatus {
    Alive,
    Retracted,
    Expired,
    Redacted,
}

impl FactStatus {
    pub fn code(self) -> &'static str {
        match self {
            FactStatus::Alive => "alive",
            FactStatus::Retracted => "retracted",
            FactStatus::Expired => "expired",
            FactStatus::Redacted => "redacted",
        }
    }

    pub fn label_zh(self) -> &'static str {
        match self {
            FactStatus::Alive => "有效",
            FactStatus::Retracted => "已撤回",
            FactStatus::Expired => "已过期",
            FactStatus::Redacted => "已遗忘",
        }
    }

    pub fn is_terminal(self) -> bool {
        self != FactStatus::Alive
    }

    fn from_retirement(kind: RetirementKind) -> Option<Self> {
        match kind {
            RetirementKind::Retracted => Some(FactStatus::Retracted),
            RetirementKind::Expired => Some(FactStatus::Expired),
            RetirementKind::Redacted => Some(FactStatus::Redacted),
            RetirementKind::Superseded => None,
        }
    }
}

/// 一条登记事实的规范身份。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct FactKey {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    #[serde(default, skip_serializing_if = "is_false")]
    pub session: bool,
}

impl FactKey {
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
        session: bool,
    ) -> Self {
        Self {
            subject: subject.into().trim().to_string(),
            predicate: predicate.into().trim().to_string(),
            object: object.into().trim().to_string(),
            session,
        }
    }

    pub fn from_evidence(evidence: &ClaimEvidence) -> Self {
        Self::new(
            &evidence.subject,
            &evidence.predicate,
            &evidence.object,
            evidence.session,
        )
    }

    pub fn render(&self) -> String {
        format!(
            "{} {} {} {}",
            if self.session { "会话" } else { "图谱" },
            self.subject,
            self.predicate,
            self.object
        )
    }

    /// 无歧义的索引键：字段长度前缀消除分隔符碰撞。
    pub fn key(&self) -> String {
        format!(
            "{}|{}:{}|{}:{}|{}:{}",
            if self.session { "s" } else { "g" },
            self.subject.len(),
            self.subject,
            self.predicate.len(),
            self.predicate,
            self.object.len(),
            self.object
        )
    }

    pub fn digest(&self) -> String {
        fnv1a_hex(self.key().as_bytes())
    }
}

/// claim 的一条支撑。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ClaimSupport {
    Fact {
        subject: String,
        predicate: String,
        object: String,
        #[serde(default, skip_serializing_if = "is_false")]
        session: bool,
    },
    Claim {
        id: String,
    },
}

impl ClaimSupport {
    pub fn fact(key: &FactKey) -> Self {
        ClaimSupport::Fact {
            subject: key.subject.clone(),
            predicate: key.predicate.clone(),
            object: key.object.clone(),
            session: key.session,
        }
    }

    pub fn from_evidence(evidence: &ClaimEvidence) -> Self {
        Self::fact(&FactKey::from_evidence(evidence))
    }

    pub fn claim(id: impl Into<String>) -> Self {
        ClaimSupport::Claim {
            id: id.into().trim().to_string(),
        }
    }

    pub fn as_fact(&self) -> Option<FactKey> {
        match self {
            ClaimSupport::Fact {
                subject,
                predicate,
                object,
                session,
            } => Some(FactKey::new(subject, predicate, object, *session)),
            ClaimSupport::Claim { .. } => None,
        }
    }

    pub fn claim_id(&self) -> Option<&str> {
        match self {
            ClaimSupport::Claim { id } => Some(id),
            ClaimSupport::Fact { .. } => None,
        }
    }

    pub fn render(&self) -> String {
        match self {
            ClaimSupport::Fact { .. } => {
                self.as_fact().map(|fact| fact.render()).unwrap_or_default()
            }
            ClaimSupport::Claim { id } => format!("claim:{}", id),
        }
    }
}

/// claim 的规范结构；文本只作为表达变体，不参与身份。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ClaimShape {
    pub subject: String,
    pub predicate: String,
    /// 生成器槽位或规范文本摘要，用于区分同一谓词下的不同命题。
    pub structure: String,
    #[serde(default, skip_serializing_if = "is_false")]
    pub session: bool,
}

impl ClaimShape {
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        structure: impl Into<String>,
        session: bool,
    ) -> Self {
        Self {
            subject: subject.into().trim().to_string(),
            predicate: predicate.into().trim().to_string(),
            structure: structure.into().trim().to_string(),
            session,
        }
    }
}

/// claim 的出处：哪份计划、哪条回答或手工登记。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ClaimOrigin {
    Plan { plan_id: String, slot: String },
    Reply { reply_seq: u64, index: u32 },
    Manual { note: String },
}

impl ClaimOrigin {
    pub fn plan(plan_id: impl Into<String>, slot: impl Into<String>) -> Self {
        ClaimOrigin::Plan {
            plan_id: plan_id.into(),
            slot: slot.into(),
        }
    }

    pub fn reply(reply_seq: u64, index: u32) -> Self {
        ClaimOrigin::Reply { reply_seq, index }
    }

    pub fn manual(note: impl Into<String>) -> Self {
        ClaimOrigin::Manual { note: note.into() }
    }

    pub fn render(&self) -> String {
        match self {
            ClaimOrigin::Plan { plan_id, slot } => format!("计划 {} / {}", plan_id, slot),
            ClaimOrigin::Reply { reply_seq, index } => {
                format!("回答 #{} / 第 {} 句", reply_seq, index)
            }
            ClaimOrigin::Manual { note } => format!("手工：{}", note),
        }
    }
}

/// 退役来源，便于沿谱系解释「为什么失效」。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RetirementCause {
    Fact { fact: FactKey },
    Parent { claim_id: String },
    Manual,
}

/// 一次终止迁移的不可变记录。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Retirement {
    pub kind: RetirementKind,
    pub reason: String,
    pub turn: u64,
    pub seq: u64,
    pub cause: RetirementCause,
}

/// 一条 claim 的完整记录。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaimRecord {
    pub id: String,
    pub shape: ClaimShape,
    /// 当前表达文本的规范摘要；同一 claim 可以有多个表达变体。
    #[serde(default)]
    pub text_digests: BTreeSet<String>,
    pub status: ClaimStatus,
    #[serde(default)]
    pub support: Vec<ClaimSupport>,
    #[serde(default)]
    pub origins: BTreeSet<ClaimOrigin>,
    pub first_turn: u64,
    pub last_turn: u64,
    pub revision: u64,
    pub registered_seq: u64,
    pub updated_seq: u64,
    /// 是否曾经说出口；`stale` 复核后据此恢复 `emitted` 或 `planned`。
    #[serde(default)]
    pub ever_emitted: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retirement: Option<Retirement>,
}

/// 一条事实的当前生命周期。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FactRecord {
    pub key: FactKey,
    pub status: FactStatus,
    pub first_turn: u64,
    pub last_turn: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retirement: Option<Retirement>,
}

/// 事件种类；事实登记也进入同一条链，保证迁移顺序可回放。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimEventKind {
    ClaimRegistered,
    ClaimEmitted,
    ClaimStale,
    ClaimRetracted,
    ClaimSuperseded,
    ClaimExpired,
    ClaimRedacted,
    ClaimRevalidated,
    FactRegistered,
    FactRetired,
}

impl ClaimEventKind {
    pub fn code(self) -> &'static str {
        match self {
            ClaimEventKind::ClaimRegistered => "claim_registered",
            ClaimEventKind::ClaimEmitted => "claim_emitted",
            ClaimEventKind::ClaimStale => "claim_stale",
            ClaimEventKind::ClaimRetracted => "claim_retracted",
            ClaimEventKind::ClaimSuperseded => "claim_superseded",
            ClaimEventKind::ClaimExpired => "claim_expired",
            ClaimEventKind::ClaimRedacted => "claim_redacted",
            ClaimEventKind::ClaimRevalidated => "claim_revalidated",
            ClaimEventKind::FactRegistered => "fact_registered",
            ClaimEventKind::FactRetired => "fact_retired",
        }
    }

    fn from_retirement(kind: RetirementKind) -> Self {
        match kind {
            RetirementKind::Retracted => ClaimEventKind::ClaimRetracted,
            RetirementKind::Superseded => ClaimEventKind::ClaimSuperseded,
            RetirementKind::Expired => ClaimEventKind::ClaimExpired,
            RetirementKind::Redacted => ClaimEventKind::ClaimRedacted,
        }
    }
}

/// 事件作用对象。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ClaimEventTarget {
    Claim { id: String },
    Fact { fact: FactKey },
}

impl ClaimEventTarget {
    pub fn claim(id: impl Into<String>) -> Self {
        ClaimEventTarget::Claim { id: id.into() }
    }

    pub fn fact(fact: &FactKey) -> Self {
        ClaimEventTarget::Fact { fact: fact.clone() }
    }
}

/// 追加式生命周期事件。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaimEvent {
    pub seq: u64,
    pub turn: u64,
    pub revision: u64,
    pub kind: ClaimEventKind,
    pub target: ClaimEventTarget,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub from: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub to: Option<String>,
    pub reason: String,
    #[serde(default)]
    pub prev_hash: String,
    #[serde(default)]
    pub hash: String,
}

impl ClaimEvent {
    fn material(&self) -> String {
        canonical_json(&serde_json::json!({
            "seq": self.seq,
            "turn": self.turn,
            "revision": self.revision,
            "kind": self.kind.code(),
            "target": self.target,
            "from": self.from,
            "to": self.to,
            "reason": self.reason,
            "prev_hash": self.prev_hash,
        }))
    }

    fn compute_hash(&self) -> String {
        fnv1a_hex(self.material().as_bytes())
    }

    pub fn verify_hash(&self) -> bool {
        !self.hash.is_empty() && self.compute_hash() == self.hash
    }
}

/// 链校验结果。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChainReport {
    pub version: u32,
    pub intact: bool,
    pub entries: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub broken_at: Option<u64>,
    pub head: String,
    #[serde(default)]
    pub truncated: bool,
}

impl ChainReport {
    pub fn empty() -> Self {
        Self {
            version: CLAIM_LEDGER_VERSION,
            intact: true,
            entries: 0,
            broken_at: None,
            head: "0".to_string(),
            truncated: false,
        }
    }
}

/// 登记草稿；调用方只需给出结构与支撑，身份由账本计算。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaimDraft {
    pub subject: String,
    pub predicate: String,
    pub structure: String,
    pub text: String,
    pub session: bool,
    #[serde(default)]
    pub support: Vec<ClaimSupport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin: Option<ClaimOrigin>,
}

impl ClaimDraft {
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        structure: impl Into<String>,
        text: impl Into<String>,
        session: bool,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            structure: structure.into(),
            text: text.into(),
            session,
            support: Vec::new(),
            origin: None,
        }
    }

    pub fn with_support(mut self, support: impl IntoIterator<Item = ClaimSupport>) -> Self {
        self.support.extend(support);
        self
    }

    pub fn with_origin(mut self, origin: ClaimOrigin) -> Self {
        self.origin = Some(origin);
        self
    }

    /// 从回答计划中的一条命题构造草稿。
    pub fn from_plan_claim(plan: &AnswerPlan, claim: &PlanClaim) -> Self {
        let mut support: Vec<ClaimSupport> = claim
            .evidence
            .iter()
            .map(ClaimSupport::from_evidence)
            .collect();
        if let Some(justification) = &claim.justification {
            support.extend(
                justification
                    .premises
                    .iter()
                    .map(ClaimSupport::from_evidence),
            );
        }
        support.sort();
        support.dedup();
        let structure = if claim.slot.trim().is_empty() {
            format!(
                "text:{}",
                fnv1a_hex(canonical_statement(&claim.text).as_bytes())
            )
        } else {
            format!("slot:{}", claim.slot.trim())
        };
        Self {
            subject: plan.subject.clone(),
            predicate: claim.predicate.clone(),
            structure,
            text: claim.text.clone(),
            session: claim.session,
            support,
            origin: Some(ClaimOrigin::plan(plan.id.clone(), claim.slot.clone())),
        }
    }
}

/// 登记错误：封闭枚举，避免把不同失败折叠成同一条消息。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClaimError {
    EmptySubject,
    EmptyPredicate,
    UnknownParent(String),
    SelfParent(String),
    UnknownClaim(String),
    InvalidRetirement(String),
    StillUnsupported(String),
    LedgerVersion(u32),
    Serialize(String),
}

impl fmt::Display for ClaimError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ClaimError::EmptySubject => write!(formatter, "claim 的主语不能为空"),
            ClaimError::EmptyPredicate => write!(formatter, "claim 的谓词不能为空"),
            ClaimError::UnknownParent(id) => write!(formatter, "父 claim 不存在：{}", id),
            ClaimError::SelfParent(id) => write!(formatter, "claim 不能支撑自己：{}", id),
            ClaimError::UnknownClaim(id) => write!(formatter, "claim 不存在：{}", id),
            ClaimError::InvalidRetirement(detail) => {
                write!(formatter, "非法的退役请求：{}", detail)
            }
            ClaimError::StillUnsupported(id) => {
                write!(formatter, "claim 的支撑仍失效，不能复核为可用：{}", id)
            }
            ClaimError::LedgerVersion(version) => {
                write!(formatter, "不支持的 claim 生命周期版本：{}", version)
            }
            ClaimError::Serialize(detail) => {
                write!(formatter, "claim 生命周期序列化失败：{}", detail)
            }
        }
    }
}

impl std::error::Error for ClaimError {}

/// 一次登记的结果。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegisteredClaim {
    pub slot: String,
    pub id: String,
    pub status: ClaimStatus,
    pub created: bool,
    /// 在原计划中的零基位置；用于回答出处编号，避免被拒命题造成错位。
    #[serde(default)]
    pub plan_index: usize,
}

/// 计划登记结果；被拒绝的命题显式列出，不静默丢弃。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanRegistration {
    pub plan_id: String,
    pub claims: Vec<RegisteredClaim>,
    pub rejected: Vec<String>,
}

/// 生成闸门裁决。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateVerdict {
    Allow,
    Refresh,
    Deny,
}

impl GateVerdict {
    pub fn code(self) -> &'static str {
        match self {
            GateVerdict::Allow => "allow",
            GateVerdict::Refresh => "refresh",
            GateVerdict::Deny => "deny",
        }
    }

    pub fn label_zh(self) -> &'static str {
        match self {
            GateVerdict::Allow => "允许",
            GateVerdict::Refresh => "重算",
            GateVerdict::Deny => "拒绝",
        }
    }
}

/// 闸门发现码。新增原因必须显式出现在这里。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateFindingCode {
    UnknownClaim,
    UnknownSupport,
    MissingEvidence,
    ChainDamaged,
    LedgerVersion,
    ClaimRetired,
    SupportRetired,
    ParentRetired,
    ExternalTombstone,
    StaleClaim,
}

impl GateFindingCode {
    pub fn code(self) -> &'static str {
        match self {
            GateFindingCode::UnknownClaim => "unknown_claim",
            GateFindingCode::UnknownSupport => "unknown_support",
            GateFindingCode::MissingEvidence => "missing_evidence",
            GateFindingCode::ChainDamaged => "chain_damaged",
            GateFindingCode::LedgerVersion => "ledger_version",
            GateFindingCode::ClaimRetired => "claim_retired",
            GateFindingCode::SupportRetired => "support_retired",
            GateFindingCode::ParentRetired => "parent_retired",
            GateFindingCode::ExternalTombstone => "external_tombstone",
            GateFindingCode::StaleClaim => "stale_claim",
        }
    }

    pub fn label_zh(self) -> &'static str {
        match self {
            GateFindingCode::UnknownClaim => "未登记命题",
            GateFindingCode::UnknownSupport => "未知支撑",
            GateFindingCode::MissingEvidence => "缺少证据",
            GateFindingCode::ChainDamaged => "事件链损坏",
            GateFindingCode::LedgerVersion => "账本版本不支持",
            GateFindingCode::ClaimRetired => "命题已退役",
            GateFindingCode::SupportRetired => "支撑已退役",
            GateFindingCode::ParentRetired => "父命题已退役",
            GateFindingCode::ExternalTombstone => "外部墓碑命中",
            GateFindingCode::StaleClaim => "命题待复核",
        }
    }
}

/// 单条闸门发现。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct GateFinding {
    pub code: GateFindingCode,
    pub detail: String,
}

impl GateFinding {
    fn new(code: GateFindingCode, detail: impl Into<String>) -> Self {
        Self {
            code,
            detail: detail.into(),
        }
    }

    pub fn render_zh(&self) -> String {
        format!("{}：{}", self.code.label_zh(), self.detail)
    }
}

/// 闸门上下文；`extra_retired` 用于接入隐私墓碑等外部退役集合。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GateContext {
    #[serde(default)]
    pub chain_intact: bool,
    #[serde(default)]
    pub extra_retired: BTreeSet<FactKey>,
}

impl Default for GateContext {
    fn default() -> Self {
        Self {
            chain_intact: true,
            extra_retired: BTreeSet::new(),
        }
    }
}

impl GateContext {
    pub fn with_chain_intact(mut self, intact: bool) -> Self {
        self.chain_intact = intact;
        self
    }

    pub fn with_retired_facts(mut self, facts: impl IntoIterator<Item = FactKey>) -> Self {
        self.extra_retired.extend(facts);
        self
    }

    /// 用隐私台账的墓碑扩充外部退役集合。
    pub fn with_privacy_ledger(self, ledger: &PrivacyLedger, facts: &[FactKey]) -> Self {
        let retired = privacy_retired_facts(ledger, facts.iter().cloned());
        self.with_retired_facts(retired)
    }
}

/// 返回隐私台账中命中墓碑的事实；无分类信息时保守匹配。
pub fn privacy_retired_facts(
    ledger: &PrivacyLedger,
    facts: impl IntoIterator<Item = FactKey>,
) -> BTreeSet<FactKey> {
    facts
        .into_iter()
        .filter(|fact| {
            ledger.is_tombstoned_any_category(&fact.subject, &fact.predicate, &fact.object)
        })
        .collect()
}

/// 提交前裁决；digest 覆盖全部字段（不含 digest 自身），可逐字节复核。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GateDecision {
    pub claim_id: String,
    pub verdict: GateVerdict,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<ClaimStatus>,
    pub findings: Vec<GateFinding>,
    pub digest: String,
}

impl GateDecision {
    fn seal(mut self) -> Self {
        let material = canonical_json(&serde_json::json!({
            "claim_id": self.claim_id,
            "verdict": self.verdict.code(),
            "status": self.status.map(ClaimStatus::code),
            "findings": self.findings.iter().map(|finding| {
                serde_json::json!({"code": finding.code.code(), "detail": finding.detail})
            }).collect::<Vec<_>>(),
        }));
        self.digest = fnv1a_hex(material.as_bytes());
        self
    }

    pub fn allowed(&self) -> bool {
        self.verdict == GateVerdict::Allow
    }

    pub fn render_zh(&self) -> String {
        let findings = if self.findings.is_empty() {
            "无".to_string()
        } else {
            self.findings
                .iter()
                .map(GateFinding::render_zh)
                .collect::<Vec<_>>()
                .join("；")
        };
        format!(
            "{}（{}）：{}",
            self.verdict.label_zh(),
            self.claim_id,
            findings
        )
    }
}

/// 一次计划提交的闸门结果。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanEmission {
    pub plan_id: String,
    pub emitted: Vec<String>,
    pub refused: Vec<GateDecision>,
    pub rejected: Vec<String>,
}

impl PlanEmission {
    pub fn complete(&self) -> bool {
        self.refused.is_empty() && self.rejected.is_empty()
    }

    pub fn render_zh(&self) -> String {
        let mut output = format!(
            "计划 {}：提交 {} 条，拒绝 {} 条",
            self.plan_id,
            self.emitted.len(),
            self.refused.len() + self.rejected.len()
        );
        for decision in &self.refused {
            output.push_str("\n  ");
            output.push_str(&decision.render_zh());
        }
        for reason in &self.rejected {
            output.push_str("\n  ");
            output.push_str(reason);
        }
        output
    }
}

/// 影响面中的一条 claim。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImpactEntry {
    pub claim_id: String,
    pub status: ClaimStatus,
    pub depth: usize,
    pub origins: BTreeSet<ClaimOrigin>,
}

/// 从事实或 claim 出发的反向影响面。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Impact {
    pub root: String,
    pub status: String,
    pub affected: Vec<ImpactEntry>,
    pub plans: BTreeSet<String>,
    pub replies: BTreeSet<u64>,
}

impl Impact {
    pub fn is_empty(&self) -> bool {
        self.affected.is_empty()
    }

    pub fn render_zh(&self) -> String {
        let mut output = format!(
            "{}（{}）：影响 {} 条断言",
            self.root,
            self.status,
            self.affected.len()
        );
        for entry in &self.affected {
            output.push_str(&format!(
                "\n  {} [{}] 深度 {}",
                entry.claim_id,
                entry.status.label_zh(),
                entry.depth
            ));
            for origin in &entry.origins {
                output.push_str("\n    ");
                output.push_str(&origin.render());
            }
        }
        if !self.plans.is_empty() {
            output.push_str("\n  计划：");
            output.push_str(&self.plans.iter().cloned().collect::<Vec<_>>().join("、"));
        }
        if !self.replies.is_empty() {
            output.push_str("\n  回答：");
            output.push_str(
                &self
                    .replies
                    .iter()
                    .map(u64::to_string)
                    .collect::<Vec<_>>()
                    .join("、"),
            );
        }
        output
    }
}

/// 审计三态；`Unknown` 永远优先于 `Findings` 与 `Clean`。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditVerdict {
    Clean,
    Findings,
    Unknown,
}

impl AuditVerdict {
    pub fn code(self) -> &'static str {
        match self {
            AuditVerdict::Clean => "clean",
            AuditVerdict::Findings => "findings",
            AuditVerdict::Unknown => "unknown",
        }
    }

    pub fn label_zh(self) -> &'static str {
        match self {
            AuditVerdict::Clean => "干净",
            AuditVerdict::Findings => "发现风险",
            AuditVerdict::Unknown => "无法判定",
        }
    }
}

/// 审计发现。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct AuditFinding {
    pub code: GateFindingCode,
    pub claim_id: String,
    pub detail: String,
}

/// 审计选项；默认不截断，链损坏由调用方显式传入。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditOptions {
    #[serde(default)]
    pub chain_intact: bool,
    #[serde(default)]
    pub extra_retired: BTreeSet<FactKey>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_events: Option<usize>,
}

impl Default for AuditOptions {
    fn default() -> Self {
        Self {
            chain_intact: true,
            extra_retired: BTreeSet::new(),
            max_events: None,
        }
    }
}

impl AuditOptions {
    pub fn with_external_chain(mut self, intact: bool) -> Self {
        self.chain_intact = intact;
        self
    }

    pub fn with_retired_facts(mut self, facts: impl IntoIterator<Item = FactKey>) -> Self {
        self.extra_retired.extend(facts);
        self
    }

    pub fn with_event_window(mut self, max_events: usize) -> Self {
        self.max_events = Some(max_events);
        self
    }
}

/// 确定性审计报告；`digest` 可跨进程逐字节比较。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaimAuditReport {
    pub verdict: AuditVerdict,
    pub chain: ChainReport,
    pub claims: usize,
    pub active: usize,
    pub emitted: usize,
    pub retired: usize,
    /// 已退役但曾经说出口的 claim：只允许作为归档历史存在。
    pub archived: usize,
    pub ungrounded: usize,
    pub late_emissions: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_emissions: Option<usize>,
    pub findings: Vec<AuditFinding>,
    pub unknown_reasons: Vec<String>,
    pub digest: String,
}

impl ClaimAuditReport {
    fn seal(mut self) -> Self {
        let material = canonical_json(&serde_json::json!({
            "verdict": self.verdict.code(),
            "chain": {
                "version": self.chain.version,
                "intact": self.chain.intact,
                "entries": self.chain.entries,
                "broken_at": self.chain.broken_at,
                "head": self.chain.head,
                "truncated": self.chain.truncated,
            },
            "claims": self.claims,
            "active": self.active,
            "emitted": self.emitted,
            "retired": self.retired,
            "archived": self.archived,
            "ungrounded": self.ungrounded,
            "late_emissions": self.late_emissions,
            "expected_emissions": self.expected_emissions,
            "findings": self.findings.iter().map(|finding| {
                serde_json::json!({
                    "code": finding.code.code(),
                    "claim_id": finding.claim_id,
                    "detail": finding.detail,
                })
            }).collect::<Vec<_>>(),
            "unknown_reasons": self.unknown_reasons,
        }));
        self.digest = fnv1a_hex(material.as_bytes());
        self
    }

    pub fn render_zh(&self) -> String {
        let mut output = format!(
            "命题审计：{}（{}）\n  条目 {} / 活跃 {} / 已说 {} / 已退役 {} / 归档 {} / 无证据 {} / 迟发 {}",
            self.verdict.label_zh(),
            self.verdict.code(),
            self.claims,
            self.active,
            self.emitted,
            self.retired,
            self.archived,
            self.ungrounded,
            self.late_emissions
        );
        output.push_str(&format!(
            "\n  事件链：{}（{} 条，head {}）",
            if self.chain.intact {
                "完整"
            } else {
                "损坏"
            },
            self.chain.entries,
            self.chain.head
        ));
        if let Some(expected) = self.expected_emissions {
            output.push_str(&format!("\n  预期已说：{}", expected));
        }
        for finding in &self.findings {
            output.push_str(&format!(
                "\n  [{}] {}：{}",
                finding.code.code(),
                finding.claim_id,
                finding.detail
            ));
        }
        for reason in &self.unknown_reasons {
            output.push_str("\n  [unknown] ");
            output.push_str(reason);
        }
        output.push_str(&format!("\n  digest：{}", self.digest));
        output
    }
}

/// 表面的可擦除性等级。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SurfaceKind {
    /// 活跃表面：计划、工作记忆、证明缓存、当前回答等，删除后不得残留。
    Active,
    /// 归档表面：编年史历史字节等不可物理擦除的追加日志。
    Archival,
}

impl SurfaceKind {
    pub fn code(self) -> &'static str {
        match self {
            SurfaceKind::Active => "active",
            SurfaceKind::Archival => "archival",
        }
    }

    pub fn label_zh(self) -> &'static str {
        match self {
            SurfaceKind::Active => "活跃表面",
            SurfaceKind::Archival => "归档表面",
        }
    }
}

/// 调用方提供的待扫描表面；文本只在本次扫描中使用，不写入账本。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Surface {
    pub name: String,
    pub kind: SurfaceKind,
    pub text: String,
}

impl Surface {
    pub fn active(name: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            kind: SurfaceKind::Active,
            text: text.into(),
        }
    }

    pub fn archival(name: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            kind: SurfaceKind::Archival,
            text: text.into(),
        }
    }
}

/// 一次明文命中。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ResidueHit {
    pub surface: String,
    pub kind: SurfaceKind,
    pub term: String,
    pub count: usize,
}

impl ResidueHit {
    pub fn render_zh(&self) -> String {
        format!(
            "{} {}：{} × {}",
            self.kind.label_zh(),
            self.surface,
            self.term,
            self.count
        )
    }
}

/// 明文残留报告：活跃命中是风险，归档命中只做显式披露。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResidueReport {
    pub verdict: AuditVerdict,
    pub scanned_surfaces: usize,
    pub scanned_terms: usize,
    pub active_hits: Vec<ResidueHit>,
    pub archival_hits: Vec<ResidueHit>,
    pub unknown_reasons: Vec<String>,
    pub digest: String,
}

impl ResidueReport {
    fn seal(mut self) -> Self {
        let material = canonical_json(&serde_json::json!({
            "verdict": self.verdict.code(),
            "scanned_surfaces": self.scanned_surfaces,
            "scanned_terms": self.scanned_terms,
            "active_hits": self.active_hits.iter().map(|hit| {
                serde_json::json!({
                    "surface": hit.surface,
                    "kind": hit.kind.code(),
                    "term": hit.term,
                    "count": hit.count,
                })
            }).collect::<Vec<_>>(),
            "archival_hits": self.archival_hits.iter().map(|hit| {
                serde_json::json!({
                    "surface": hit.surface,
                    "kind": hit.kind.code(),
                    "term": hit.term,
                    "count": hit.count,
                })
            }).collect::<Vec<_>>(),
            "unknown_reasons": self.unknown_reasons,
        }));
        self.digest = fnv1a_hex(material.as_bytes());
        self
    }

    pub fn render_zh(&self) -> String {
        let mut output = format!(
            "明文残留扫描：{}（{}）\n  表面 {} / 词项 {} / 活跃命中 {} / 归档命中 {}",
            self.verdict.label_zh(),
            self.verdict.code(),
            self.scanned_surfaces,
            self.scanned_terms,
            self.active_hits.len(),
            self.archival_hits.len()
        );
        for hit in &self.active_hits {
            output.push_str("\n  [active] ");
            output.push_str(&hit.render_zh());
        }
        for hit in &self.archival_hits {
            output.push_str("\n  [archival] ");
            output.push_str(&hit.render_zh());
        }
        for reason in &self.unknown_reasons {
            output.push_str("\n  [unknown] ");
            output.push_str(reason);
        }
        output.push_str(&format!("\n  digest：{}", self.digest));
        output
    }
}

/// 用调用方提供的明文扫描表面。
///
/// 删除时明文只存在于本次调用；账本不保存原文。活跃表面命中判 `findings`，
/// 归档表面命中单独报告为 archival，不会被忽略，也不会被误判为干净。
pub fn scan_residue(surfaces: &[Surface], terms: &[String]) -> ResidueReport {
    let mut terms: Vec<String> = terms
        .iter()
        .map(|term| term.trim().to_string())
        .filter(|term| !term.is_empty())
        .collect();
    terms.sort();
    terms.dedup();
    let mut active_hits = Vec::new();
    let mut archival_hits = Vec::new();
    let mut unknown_reasons = Vec::new();
    if terms.is_empty() {
        unknown_reasons.push("没有提供要扫描的明文".to_string());
    }
    if surfaces.is_empty() {
        unknown_reasons.push("没有提供任何表面".to_string());
    }
    if !surfaces.is_empty()
        && !surfaces
            .iter()
            .any(|surface| surface.kind == SurfaceKind::Active)
    {
        unknown_reasons.push("没有提供活跃表面，无法判断残留".to_string());
    }
    for surface in surfaces {
        for term in &terms {
            let count = surface.text.matches(term.as_str()).count();
            if count == 0 {
                continue;
            }
            let hit = ResidueHit {
                surface: surface.name.clone(),
                kind: surface.kind,
                term: term.clone(),
                count,
            };
            match surface.kind {
                SurfaceKind::Active => active_hits.push(hit),
                SurfaceKind::Archival => archival_hits.push(hit),
            }
        }
    }
    active_hits.sort();
    archival_hits.sort();
    let verdict = if !unknown_reasons.is_empty() {
        AuditVerdict::Unknown
    } else if !active_hits.is_empty() {
        AuditVerdict::Findings
    } else {
        AuditVerdict::Clean
    };
    ResidueReport {
        verdict,
        scanned_surfaces: surfaces.len(),
        scanned_terms: terms.len(),
        active_hits,
        archival_hits,
        unknown_reasons,
        digest: String::new(),
    }
    .seal()
}

/// claim 生命周期账本。
///
/// 字段公开以便审计读取；修改应通过方法完成，避免索引与记录脱节。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaimLedger {
    #[serde(default = "current_version")]
    pub version: u32,
    #[serde(default)]
    pub claims: BTreeMap<String, ClaimRecord>,
    #[serde(default)]
    pub facts: BTreeMap<String, FactRecord>,
    #[serde(default)]
    pub events: Vec<ClaimEvent>,
    #[serde(default)]
    pub fact_claims: BTreeMap<String, BTreeSet<String>>,
    #[serde(default)]
    pub parents: BTreeMap<String, BTreeSet<String>>,
    #[serde(default)]
    pub children: BTreeMap<String, BTreeSet<String>>,
}

impl Default for ClaimLedger {
    fn default() -> Self {
        Self::new()
    }
}

impl ClaimLedger {
    pub fn new() -> Self {
        Self {
            version: CLAIM_LEDGER_VERSION,
            claims: BTreeMap::new(),
            facts: BTreeMap::new(),
            events: Vec::new(),
            fact_claims: BTreeMap::new(),
            parents: BTreeMap::new(),
            children: BTreeMap::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.claims.is_empty()
    }

    pub fn len(&self) -> usize {
        self.claims.len()
    }

    pub fn claim(&self, id: &str) -> Option<&ClaimRecord> {
        self.claims.get(id)
    }

    pub fn fact(&self, key: &FactKey) -> Option<&FactRecord> {
        self.facts.get(&key.key())
    }

    /// claim 身份：规范结构 + 排序后的支撑集合。
    pub fn claim_id(shape: &ClaimShape, support: &[ClaimSupport]) -> String {
        let mut support = support.to_vec();
        support.sort();
        support.dedup();
        let material = canonical_json(&serde_json::json!({
            "version": CLAIM_LEDGER_VERSION,
            "shape": shape,
            "support": support,
        }));
        format!(
            "{}{}",
            CLAIM_ID_PREFIX,
            &fnv1a_hex(material.as_bytes())[..16]
        )
    }

    fn last_hash(&self) -> String {
        self.events
            .last()
            .map(|event| event.hash.clone())
            .filter(|hash| !hash.is_empty())
            .unwrap_or_else(|| "0".to_string())
    }

    fn append_event(
        &mut self,
        kind: ClaimEventKind,
        target: ClaimEventTarget,
        from: Option<&str>,
        to: Option<&str>,
        reason: &str,
        turn: u64,
        revision: u64,
    ) -> u64 {
        let seq = self.events.len() as u64 + 1;
        let prev_hash = self.last_hash();
        let mut event = ClaimEvent {
            seq,
            turn,
            revision,
            kind,
            target,
            from: from.map(str::to_string),
            to: to.map(str::to_string),
            reason: bounded_reason(reason),
            prev_hash,
            hash: String::new(),
        };
        event.hash = event.compute_hash();
        self.events.push(event);
        seq
    }

    fn ensure_fact(&mut self, key: &FactKey, turn: u64, revision: u64) -> bool {
        if self.facts.contains_key(&key.key()) {
            return false;
        }
        self.facts.insert(
            key.key(),
            FactRecord {
                key: key.clone(),
                status: FactStatus::Alive,
                first_turn: turn,
                last_turn: turn,
                retirement: None,
            },
        );
        self.append_event(
            ClaimEventKind::FactRegistered,
            ClaimEventTarget::fact(key),
            None,
            Some(FactStatus::Alive.code()),
            "登记事实",
            turn,
            revision,
        );
        true
    }

    /// 登记一条 claim。重复登记不会重置终止状态；新证据得到新的身份。
    pub fn register(
        &mut self,
        draft: ClaimDraft,
        turn: u64,
        revision: u64,
    ) -> Result<RegisteredClaim, ClaimError> {
        if draft.subject.trim().is_empty() {
            return Err(ClaimError::EmptySubject);
        }
        if draft.predicate.trim().is_empty() {
            return Err(ClaimError::EmptyPredicate);
        }
        let mut support = draft.support;
        support.sort();
        support.dedup();
        let mut seen_facts = BTreeSet::new();
        let mut seen_claims = BTreeSet::new();
        for item in &support {
            match item {
                ClaimSupport::Fact { .. } => {
                    if let Some(fact) = item.as_fact() {
                        seen_facts.insert(fact);
                    }
                }
                ClaimSupport::Claim { id } => {
                    if id.trim().is_empty() {
                        return Err(ClaimError::UnknownParent(id.clone()));
                    }
                    if !self.claims.contains_key(id) {
                        return Err(ClaimError::UnknownParent(id.clone()));
                    }
                    seen_claims.insert(id.clone());
                }
            }
        }
        let structure = if draft.structure.trim().is_empty() {
            format!(
                "text:{}",
                fnv1a_hex(canonical_statement(&draft.text).as_bytes())
            )
        } else {
            draft.structure.trim().to_string()
        };
        let shape = ClaimShape::new(draft.subject, draft.predicate, structure, draft.session);
        let id = Self::claim_id(&shape, &support);
        if seen_claims.contains(&id) {
            return Err(ClaimError::SelfParent(id));
        }
        let text_digest = fnv1a_hex(canonical_statement(&draft.text).as_bytes());
        if let Some(record) = self.claims.get_mut(&id) {
            let created = false;
            if record.text_digests.insert(text_digest) {
                record.updated_seq = self.events.len() as u64;
            }
            if let Some(origin) = draft.origin {
                if record.origins.insert(origin) {
                    record.updated_seq = self.events.len() as u64;
                }
            }
            record.last_turn = record.last_turn.max(turn);
            record.revision = record.revision.max(revision);
            return Ok(RegisteredClaim {
                slot: record.shape.structure.clone(),
                id,
                status: record.status,
                created,
                plan_index: 0,
            });
        }

        for fact in &seen_facts {
            self.ensure_fact(fact, turn, revision);
        }
        for parent in &seen_claims {
            if parent == &id {
                return Err(ClaimError::SelfParent(id));
            }
        }

        let mut inherited: Option<(RetirementKind, RetirementCause)> = None;
        for item in &support {
            match item {
                ClaimSupport::Fact { .. } => {
                    if let Some(fact) = item.as_fact() {
                        if let Some(record) = self.facts.get(&fact.key()) {
                            if let Some(kind) = match record.status {
                                FactStatus::Alive => None,
                                FactStatus::Retracted => Some(RetirementKind::Retracted),
                                FactStatus::Expired => Some(RetirementKind::Expired),
                                FactStatus::Redacted => Some(RetirementKind::Redacted),
                            } {
                                let cause = RetirementCause::Fact { fact: fact.clone() };
                                if inherited
                                    .as_ref()
                                    .map(|(current, _)| kind.rank() > current.rank())
                                    .unwrap_or(true)
                                {
                                    inherited = Some((kind, cause));
                                }
                            }
                        }
                    }
                }
                ClaimSupport::Claim { id: parent } => {
                    if let Some(record) = self.claims.get(parent) {
                        if let Some(retirement) = &record.retirement {
                            let kind = retirement.kind;
                            let cause = RetirementCause::Parent {
                                claim_id: parent.clone(),
                            };
                            if inherited
                                .as_ref()
                                .map(|(current, _)| kind.rank() > current.rank())
                                .unwrap_or(true)
                            {
                                inherited = Some((kind, cause));
                            }
                        }
                    }
                }
            }
        }

        let status = inherited
            .as_ref()
            .map(|(kind, _)| ClaimStatus::from_retirement(*kind))
            .unwrap_or(ClaimStatus::Planned);
        let seq = self.events.len() as u64 + 1;
        let retirement = inherited.map(|(kind, cause)| Retirement {
            kind,
            reason: "登记时继承支撑失效".to_string(),
            turn,
            seq,
            cause,
        });
        let record = ClaimRecord {
            id: id.clone(),
            shape,
            text_digests: BTreeSet::from([text_digest]),
            status,
            support: support.clone(),
            origins: draft.origin.into_iter().collect(),
            first_turn: turn,
            last_turn: turn,
            revision,
            registered_seq: seq,
            updated_seq: seq,
            ever_emitted: false,
            retirement,
        };
        let slot = record.shape.structure.clone();
        self.claims.insert(id.clone(), record);
        for item in &support {
            match item {
                ClaimSupport::Fact { .. } => {
                    if let Some(fact) = item.as_fact() {
                        self.fact_claims
                            .entry(fact.key())
                            .or_default()
                            .insert(id.clone());
                    }
                }
                ClaimSupport::Claim { id: parent } => {
                    self.parents
                        .entry(id.clone())
                        .or_default()
                        .insert(parent.clone());
                    self.children
                        .entry(parent.clone())
                        .or_default()
                        .insert(id.clone());
                }
            }
        }
        self.append_event(
            ClaimEventKind::ClaimRegistered,
            ClaimEventTarget::claim(&id),
            None,
            Some(status.code()),
            "登记命题",
            turn,
            revision,
        );
        Ok(RegisteredClaim {
            slot,
            id,
            status,
            created: true,
            plan_index: 0,
        })
    }

    /// 登记一份回答计划中的全部命题。
    pub fn register_plan(
        &mut self,
        plan: &AnswerPlan,
        turn: u64,
        revision: u64,
    ) -> PlanRegistration {
        let mut claims = Vec::with_capacity(plan.claims.len());
        let mut rejected = Vec::new();
        for (index, claim) in plan.claims.iter().enumerate() {
            let draft = ClaimDraft::from_plan_claim(plan, claim);
            match self.register(draft, turn, revision) {
                Ok(mut registered) => {
                    registered.plan_index = index;
                    claims.push(registered);
                }
                Err(error) => rejected.push(format!("槽位 {}：{}", claim.slot, error)),
            }
        }
        PlanRegistration {
            plan_id: plan.id.clone(),
            claims,
            rejected,
        }
    }

    fn support_problem(&self, record: &ClaimRecord) -> Option<(GateFindingCode, String)> {
        if record.support.is_empty() {
            return Some((
                GateFindingCode::MissingEvidence,
                "命题没有登记任何支撑".to_string(),
            ));
        }
        for item in &record.support {
            match item {
                ClaimSupport::Fact { .. } => {
                    let Some(fact) = item.as_fact() else {
                        continue;
                    };
                    let Some(fact_record) = self.facts.get(&fact.key()) else {
                        return Some((
                            GateFindingCode::UnknownSupport,
                            format!("事实未登记：{}", fact.render()),
                        ));
                    };
                    if fact_record.status.is_terminal() {
                        return Some((
                            GateFindingCode::SupportRetired,
                            format!("事实{}：{}", fact_record.status.label_zh(), fact.render()),
                        ));
                    }
                }
                ClaimSupport::Claim { id } => {
                    let Some(parent) = self.claims.get(id) else {
                        return Some((
                            GateFindingCode::UnknownSupport,
                            format!("父 claim 未登记：{}", id),
                        ));
                    };
                    if parent.status.is_terminal() {
                        return Some((
                            GateFindingCode::ParentRetired,
                            format!("父 claim {} {}", id, parent.status.label_zh()),
                        ));
                    }
                }
            }
        }
        None
    }

    /// 提交前闸门；不改状态，只给出 allow / refresh / deny 与可复核发现。
    pub fn gate(&self, id: &str, context: &GateContext) -> GateDecision {
        let mut findings = Vec::new();
        if self.version != CLAIM_LEDGER_VERSION {
            findings.push(GateFinding::new(
                GateFindingCode::LedgerVersion,
                format!("账本版本 {}，支持 {}", self.version, CLAIM_LEDGER_VERSION),
            ));
        }
        if !context.chain_intact || !self.verify_chain(None).intact {
            findings.push(GateFinding::new(
                GateFindingCode::ChainDamaged,
                "事件链校验失败",
            ));
        }
        let Some(record) = self.claims.get(id) else {
            findings.push(GateFinding::new(
                GateFindingCode::UnknownClaim,
                "命题未登记，无法核验支撑",
            ));
            return GateDecision {
                claim_id: id.to_string(),
                verdict: GateVerdict::Refresh,
                status: None,
                findings,
                digest: String::new(),
            }
            .seal();
        };
        if record.status.is_terminal() {
            findings.push(GateFinding::new(
                GateFindingCode::ClaimRetired,
                record.status.label_zh(),
            ));
            return GateDecision {
                claim_id: id.to_string(),
                verdict: GateVerdict::Deny,
                status: Some(record.status),
                findings,
                digest: String::new(),
            }
            .seal();
        }
        if record.status == ClaimStatus::Stale {
            findings.push(GateFinding::new(
                GateFindingCode::StaleClaim,
                "命题需要重新核验",
            ));
        }
        if let Some((code, detail)) = self.support_problem(record) {
            findings.push(GateFinding::new(code, detail));
        }
        for item in &record.support {
            if let Some(fact) = item.as_fact() {
                if context.extra_retired.contains(&fact) {
                    findings.push(GateFinding::new(
                        GateFindingCode::ExternalTombstone,
                        fact.render(),
                    ));
                }
            }
        }
        findings.sort();
        findings.dedup();
        let verdict = if findings.iter().any(|finding| {
            matches!(
                finding.code,
                GateFindingCode::ClaimRetired
                    | GateFindingCode::SupportRetired
                    | GateFindingCode::ParentRetired
                    | GateFindingCode::ExternalTombstone
            )
        }) {
            GateVerdict::Deny
        } else if findings.is_empty() {
            GateVerdict::Allow
        } else {
            GateVerdict::Refresh
        };
        GateDecision {
            claim_id: id.to_string(),
            verdict,
            status: Some(record.status),
            findings,
            digest: String::new(),
        }
        .seal()
    }

    /// 通过闸门的 claim 才记为已说；终止态永远不会被改回活跃。
    pub fn authorize_emission(
        &mut self,
        id: &str,
        origin: ClaimOrigin,
        turn: u64,
        revision: u64,
        context: &GateContext,
    ) -> GateDecision {
        let decision = self.gate(id, context);
        if !decision.allowed() {
            return decision;
        }
        let Some(record) = self.claims.get_mut(id) else {
            return decision;
        };
        let from = record.status;
        record.status = ClaimStatus::Emitted;
        record.ever_emitted = true;
        record.origins.insert(origin);
        record.last_turn = record.last_turn.max(turn);
        record.revision = record.revision.max(revision);
        record.updated_seq = self.events.len() as u64 + 1;
        self.append_event(
            ClaimEventKind::ClaimEmitted,
            ClaimEventTarget::claim(id),
            Some(from.code()),
            Some(ClaimStatus::Emitted.code()),
            "通过提交闸门",
            turn,
            revision,
        );
        decision
    }

    /// 登记计划并在提交前逐条过闸；只有 `allow` 的 claim 记为已说。
    pub fn emit_plan(
        &mut self,
        plan: &AnswerPlan,
        reply_seq: u64,
        turn: u64,
        revision: u64,
        context: &GateContext,
    ) -> PlanEmission {
        let registration = self.register_plan(plan, turn, revision);
        let mut emitted = Vec::new();
        let mut refused = Vec::new();
        for claim in &registration.claims {
            let decision = self.authorize_emission(
                &claim.id,
                ClaimOrigin::reply(reply_seq, claim.plan_index as u32 + 1),
                turn,
                revision,
                context,
            );
            if decision.allowed() {
                emitted.push(decision.claim_id);
            } else {
                refused.push(decision);
            }
        }
        PlanEmission {
            plan_id: plan.id.clone(),
            emitted,
            refused,
            rejected: registration.rejected,
        }
    }

    fn retire_claim_internal(
        &mut self,
        id: &str,
        kind: RetirementKind,
        reason: &str,
        cause: RetirementCause,
        turn: u64,
        revision: u64,
    ) -> bool {
        let Some(record) = self.claims.get_mut(id) else {
            return false;
        };
        let from = record.status;
        if from.is_terminal() {
            let current_rank = record
                .retirement
                .as_ref()
                .map(|retirement| retirement.kind.rank())
                .unwrap_or(0);
            if kind.rank() <= current_rank {
                return false;
            }
        }
        let next = ClaimStatus::from_retirement(kind);
        let seq = self.events.len() as u64 + 1;
        record.status = next;
        record.last_turn = record.last_turn.max(turn);
        record.updated_seq = seq;
        record.retirement = Some(Retirement {
            kind,
            reason: bounded_reason(reason),
            turn,
            seq,
            cause,
        });
        self.append_event(
            ClaimEventKind::from_retirement(kind),
            ClaimEventTarget::claim(id),
            Some(from.code()),
            Some(next.code()),
            reason,
            turn,
            revision,
        );
        true
    }

    /// 退役一条事实，并把迁移沿支撑谱系传播到所有后代。
    ///
    /// 采用 AND 语义：任一必需支撑退役，依赖它的 claim 就退役。重复退役是
    /// 幂等的；已经终止的 claim 不会被改写。
    pub fn retire_fact(
        &mut self,
        fact: &FactKey,
        kind: RetirementKind,
        reason: &str,
        turn: u64,
        revision: u64,
    ) -> Result<Impact, ClaimError> {
        let Some(status) = FactStatus::from_retirement(kind) else {
            return Err(ClaimError::InvalidRetirement(format!(
                "事实不支持「{}」迁移",
                kind.label_zh()
            )));
        };
        self.ensure_fact(fact, turn, revision);
        let (already, from, current_rank) = self
            .facts
            .get(&fact.key())
            .map(|record| {
                (
                    record.status.is_terminal(),
                    record.status,
                    record
                        .retirement
                        .as_ref()
                        .map(|retirement| retirement.kind.rank())
                        .unwrap_or(0),
                )
            })
            .unwrap_or((false, FactStatus::Alive, 0));
        if !already || kind.rank() > current_rank {
            let seq = self.events.len() as u64 + 1;
            if let Some(record) = self.facts.get_mut(&fact.key()) {
                record.status = status;
                record.last_turn = record.last_turn.max(turn);
                record.retirement = Some(Retirement {
                    kind,
                    reason: bounded_reason(reason),
                    turn,
                    seq,
                    cause: RetirementCause::Fact { fact: fact.clone() },
                });
            }
            self.append_event(
                ClaimEventKind::FactRetired,
                ClaimEventTarget::fact(fact),
                Some(from.code()),
                Some(status.code()),
                reason,
                turn,
                revision,
            );
            let direct: BTreeSet<String> = self
                .fact_claims
                .get(&fact.key())
                .cloned()
                .unwrap_or_default();
            for id in direct {
                self.retire_claim_internal(
                    &id,
                    kind,
                    reason,
                    RetirementCause::Fact { fact: fact.clone() },
                    turn,
                    revision,
                );
                self.retire_descendants(&id, kind, reason, turn, revision);
            }
        }
        Ok(self.impact_of_fact(fact))
    }

    /// 把隐私台账命中的事实登记为遗忘，并返回每条事实的影响面。
    ///
    /// 未命中的事实不产生事件；重复调用幂等。分类信息缺失时由
    /// [`privacy_retired_facts`] 做保守匹配。
    pub fn retire_with_privacy(
        &mut self,
        ledger: &PrivacyLedger,
        facts: &[FactKey],
        reason: &str,
        turn: u64,
        revision: u64,
    ) -> Vec<Impact> {
        let retired = privacy_retired_facts(ledger, facts.iter().cloned());
        retired
            .into_iter()
            .filter_map(|fact| {
                self.retire_fact(&fact, RetirementKind::Redacted, reason, turn, revision)
                    .ok()
            })
            .collect()
    }

    fn retire_descendants(
        &mut self,
        id: &str,
        kind: RetirementKind,
        reason: &str,
        turn: u64,
        revision: u64,
    ) {
        let children: BTreeSet<String> = self.children.get(id).cloned().unwrap_or_default();
        for child in children {
            self.retire_claim_internal(
                &child,
                kind,
                reason,
                RetirementCause::Parent {
                    claim_id: id.to_string(),
                },
                turn,
                revision,
            );
            self.retire_descendants(&child, kind, reason, turn, revision);
        }
    }

    /// 显式退役一条 claim，并传播到后代。
    pub fn retire_claim(
        &mut self,
        id: &str,
        kind: RetirementKind,
        reason: &str,
        turn: u64,
        revision: u64,
    ) -> Result<Impact, ClaimError> {
        if !self.claims.contains_key(id) {
            return Err(ClaimError::UnknownClaim(id.to_string()));
        }
        self.retire_claim_internal(id, kind, reason, RetirementCause::Manual, turn, revision);
        self.retire_descendants(id, kind, reason, turn, revision);
        Ok(self.impact_of_claim(id))
    }

    /// 用新 claim 取代旧 claim；旧 claim 的后代一并退役。
    pub fn supersede(
        &mut self,
        old: &str,
        new: &str,
        reason: &str,
        turn: u64,
        revision: u64,
    ) -> Result<Impact, ClaimError> {
        if !self.claims.contains_key(old) {
            return Err(ClaimError::UnknownClaim(old.to_string()));
        }
        if !self.claims.contains_key(new) {
            return Err(ClaimError::UnknownClaim(new.to_string()));
        }
        self.retire_claim_internal(
            old,
            RetirementKind::Superseded,
            reason,
            RetirementCause::Manual,
            turn,
            revision,
        );
        self.retire_descendants(old, RetirementKind::Superseded, reason, turn, revision);
        Ok(self.impact_of_claim(old))
    }

    /// 将 `stale` claim 重新核验；支撑仍失效时拒绝复活。
    pub fn revalidate(
        &mut self,
        id: &str,
        turn: u64,
        revision: u64,
    ) -> Result<ClaimStatus, ClaimError> {
        let Some(record) = self.claims.get(id) else {
            return Err(ClaimError::UnknownClaim(id.to_string()));
        };
        if record.status.is_terminal() {
            return Err(ClaimError::StillUnsupported(id.to_string()));
        }
        if self.support_problem(record).is_some() {
            return Err(ClaimError::StillUnsupported(id.to_string()));
        }
        let from = record.status;
        let next = if record.ever_emitted {
            ClaimStatus::Emitted
        } else {
            ClaimStatus::Planned
        };
        if let Some(record) = self.claims.get_mut(id) {
            record.status = next;
            record.last_turn = record.last_turn.max(turn);
            record.updated_seq = self.events.len() as u64 + 1;
        }
        self.append_event(
            ClaimEventKind::ClaimRevalidated,
            ClaimEventTarget::claim(id),
            Some(from.code()),
            Some(next.code()),
            "支撑复核通过",
            turn,
            revision,
        );
        Ok(next)
    }

    fn collect_impact(&self, root: &str, start: BTreeSet<String>) -> Impact {
        let mut affected = Vec::new();
        let mut plans = BTreeSet::new();
        let mut replies = BTreeSet::new();
        let mut visited = BTreeSet::new();
        let mut frontier: VecDeque<(String, usize)> =
            start.into_iter().map(|id| (id, 1usize)).collect();
        while let Some((id, depth)) = frontier.pop_front() {
            if !visited.insert(id.clone()) {
                continue;
            }
            let Some(record) = self.claims.get(&id) else {
                continue;
            };
            for origin in &record.origins {
                match origin {
                    ClaimOrigin::Plan { plan_id, .. } => {
                        plans.insert(plan_id.clone());
                    }
                    ClaimOrigin::Reply { reply_seq, .. } => {
                        replies.insert(*reply_seq);
                    }
                    ClaimOrigin::Manual { .. } => {}
                }
            }
            affected.push(ImpactEntry {
                claim_id: id.clone(),
                status: record.status,
                depth,
                origins: record.origins.clone(),
            });
            if let Some(children) = self.children.get(&id) {
                for child in children {
                    if !visited.contains(child) {
                        frontier.push_back((child.clone(), depth + 1));
                    }
                }
            }
        }
        affected.sort_by(|left, right| {
            left.depth
                .cmp(&right.depth)
                .then_with(|| left.claim_id.cmp(&right.claim_id))
        });
        Impact {
            root: root.to_string(),
            status: "impact".to_string(),
            affected,
            plans,
            replies,
        }
    }

    /// 从一条事实反向查出直接与传递影响的 claim、计划和回答。
    pub fn impact_of_fact(&self, fact: &FactKey) -> Impact {
        let direct = self
            .fact_claims
            .get(&fact.key())
            .cloned()
            .unwrap_or_default();
        let mut impact = self.collect_impact(&fact.render(), direct);
        impact.status = self
            .facts
            .get(&fact.key())
            .map(|record| record.status.code().to_string())
            .unwrap_or_else(|| "unregistered".to_string());
        impact
    }

    /// 从一条 claim 反向查出传递影响面。
    pub fn impact_of_claim(&self, id: &str) -> Impact {
        let mut impact = self.collect_impact(id, BTreeSet::from([id.to_string()]));
        if let Some(record) = self.claims.get(id) {
            impact.status = record.status.code().to_string();
        }
        impact
    }

    /// 校验事件链；`window` 为只回放最近若干条事件时的显式截断窗口。
    pub fn verify_chain(&self, window: Option<usize>) -> ChainReport {
        let start = window
            .map(|window| self.events.len().saturating_sub(window))
            .unwrap_or(0);
        let truncated = start > 0;
        let mut previous = if truncated {
            self.events
                .get(start.saturating_sub(1))
                .map(|event| event.hash.clone())
                .unwrap_or_else(|| "0".to_string())
        } else {
            "0".to_string()
        };
        let mut broken_at = None;
        let mut checked = 0usize;
        for (offset, event) in self.events.iter().enumerate().skip(start) {
            checked += 1;
            let expected_seq = offset as u64 + 1;
            if event.seq != expected_seq || event.prev_hash != previous || !event.verify_hash() {
                broken_at = Some(event.seq);
                break;
            }
            previous = event.hash.clone();
        }
        ChainReport {
            version: self.version,
            intact: broken_at.is_none(),
            entries: checked,
            broken_at,
            head: self.last_hash(),
            truncated,
        }
    }

    /// 重建反向索引；从 JSON 载入后调用，保证索引与记录一致。
    pub fn rebuild_indexes(&mut self) {
        self.fact_claims.clear();
        self.parents.clear();
        self.children.clear();
        for (id, record) in &self.claims {
            for item in &record.support {
                match item {
                    ClaimSupport::Fact { .. } => {
                        if let Some(fact) = item.as_fact() {
                            self.fact_claims
                                .entry(fact.key())
                                .or_default()
                                .insert(id.clone());
                        }
                    }
                    ClaimSupport::Claim { id: parent } => {
                        if self.claims.contains_key(parent) {
                            self.parents
                                .entry(id.clone())
                                .or_default()
                                .insert(parent.clone());
                            self.children
                                .entry(parent.clone())
                                .or_default()
                                .insert(id.clone());
                        }
                    }
                }
            }
        }
    }

    /// 确定性审计。
    ///
    /// `expected_emissions` 是外部可核对的说出口句数；旧回答没有 claim 记录
    /// 时必须传入，否则空账本只能判 `unknown`。`options.chain_intact` 用于接入
    /// 编年史等外部链的校验结果。
    pub fn audit(
        &self,
        expected_emissions: Option<usize>,
        options: &AuditOptions,
    ) -> ClaimAuditReport {
        let chain = self.verify_chain(options.max_events);
        let truncated = chain.truncated
            || options
                .max_events
                .map(|window| self.events.len() > window)
                .unwrap_or(false);
        let mut findings: BTreeSet<AuditFinding> = BTreeSet::new();
        let mut unknown_reasons: BTreeSet<String> = BTreeSet::new();
        if self.version != CLAIM_LEDGER_VERSION {
            unknown_reasons.insert(format!(
                "账本版本 {} 不受支持（当前 {}）",
                self.version, CLAIM_LEDGER_VERSION
            ));
        }
        if !options.chain_intact {
            unknown_reasons.insert("外部事件链校验失败".to_string());
        }
        if !chain.intact {
            unknown_reasons.insert(format!(
                "事件链在序号 {} 处损坏",
                chain
                    .broken_at
                    .map(|seq| seq.to_string())
                    .unwrap_or_else(|| "未知".to_string())
            ));
        }
        if truncated {
            unknown_reasons.insert("重放窗口被截断，旧事件未参与审计".to_string());
        }

        let mut active = 0usize;
        let mut emitted = 0usize;
        let mut retired = 0usize;
        let mut archived = 0usize;
        let mut ungrounded = 0usize;
        let mut late_emissions = 0usize;
        for (id, record) in &self.claims {
            if record.status.is_active() {
                active += 1;
            }
            if record.ever_emitted {
                emitted += 1;
            }
            if record.status.is_terminal() {
                retired += 1;
                if record.ever_emitted {
                    archived += 1;
                }
            }
            if record.support.is_empty() {
                ungrounded += 1;
                if record.status.is_active() || record.ever_emitted {
                    unknown_reasons.insert(format!("{} 没有登记支撑，不能判为干净", id));
                }
            }
            if record.status == ClaimStatus::Stale {
                findings.insert(AuditFinding {
                    code: GateFindingCode::StaleClaim,
                    claim_id: id.clone(),
                    detail: "命题处于待复核状态".to_string(),
                });
            }
            if let Some((code, detail)) = self.support_problem(record) {
                if code == GateFindingCode::UnknownSupport {
                    unknown_reasons.insert(format!("{} 的支撑无法解析：{}", id, detail));
                }
                if record.status.is_active() {
                    findings.insert(AuditFinding {
                        code,
                        claim_id: id.clone(),
                        detail: format!("活跃命题含失效/未知支撑：{}", detail),
                    });
                }
            }
            for item in &record.support {
                if let Some(fact) = item.as_fact() {
                    if options.extra_retired.contains(&fact) {
                        findings.insert(AuditFinding {
                            code: GateFindingCode::ExternalTombstone,
                            claim_id: id.clone(),
                            detail: format!("支撑命中外部墓碑：{}", fact.render()),
                        });
                    }
                }
            }
            let last_emitted_seq = self
                .events
                .iter()
                .rev()
                .find(|event| {
                    event.kind == ClaimEventKind::ClaimEmitted
                        && event.target == ClaimEventTarget::claim(id)
                })
                .map(|event| event.seq);
            if let (Some(retirement), Some(emitted_seq)) = (&record.retirement, last_emitted_seq) {
                if emitted_seq > retirement.seq {
                    late_emissions += 1;
                    findings.insert(AuditFinding {
                        code: GateFindingCode::ClaimRetired,
                        claim_id: id.clone(),
                        detail: format!(
                            "序号 {} 在退役（{}）之后仍被提交",
                            emitted_seq, retirement.seq
                        ),
                    });
                }
            }
        }

        if let Some(expected) = expected_emissions {
            if emitted != expected {
                unknown_reasons.insert(format!(
                    "可核对已说命题 {} 条，与预期 {} 条不一致",
                    emitted, expected
                ));
            }
        } else if self.claims.is_empty() {
            unknown_reasons.insert("没有可审计的 claim 记录".to_string());
        }

        let verdict = if !unknown_reasons.is_empty() {
            AuditVerdict::Unknown
        } else if !findings.is_empty() {
            AuditVerdict::Findings
        } else {
            AuditVerdict::Clean
        };
        ClaimAuditReport {
            verdict,
            chain,
            claims: self.claims.len(),
            active,
            emitted,
            retired,
            archived,
            ungrounded,
            late_emissions,
            expected_emissions,
            findings: findings.into_iter().collect(),
            unknown_reasons: unknown_reasons.into_iter().collect(),
            digest: String::new(),
        }
        .seal()
    }

    /// 稳定 JSON；BTreeMap 与结构字段顺序保证同内容同字节。
    pub fn to_json(&self) -> Result<String, ClaimError> {
        serde_json::to_string(self).map_err(|error| ClaimError::Serialize(error.to_string()))
    }

    /// 从 JSON 载入并重建反向索引；版本高于当前实现时显式拒绝。
    pub fn from_json(text: &str) -> Result<Self, ClaimError> {
        let mut ledger: Self =
            serde_json::from_str(text).map_err(|error| ClaimError::Serialize(error.to_string()))?;
        if ledger.version > CLAIM_LEDGER_VERSION {
            return Err(ClaimError::LedgerVersion(ledger.version));
        }
        ledger.version = CLAIM_LEDGER_VERSION;
        ledger.rebuild_indexes();
        Ok(ledger)
    }

    /// 原子落盘：临时文件写完并同步后再 rename，半截文件不会替换旧账本。
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), ClaimError> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .map_err(|error| ClaimError::Serialize(error.to_string()))?;
            }
        }
        let text = serde_json::to_string_pretty(self)
            .map_err(|error| ClaimError::Serialize(error.to_string()))?;
        let temporary = path.with_extension("tmp");
        {
            let mut file = std::fs::File::create(&temporary)
                .map_err(|error| ClaimError::Serialize(error.to_string()))?;
            file.write_all(text.as_bytes())
                .map_err(|error| ClaimError::Serialize(error.to_string()))?;
            file.write_all(b"\n")
                .map_err(|error| ClaimError::Serialize(error.to_string()))?;
            file.sync_all()
                .map_err(|error| ClaimError::Serialize(error.to_string()))?;
        }
        std::fs::rename(&temporary, path)
            .map_err(|error| ClaimError::Serialize(error.to_string()))?;
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, ClaimError> {
        let text = std::fs::read_to_string(path)
            .map_err(|error| ClaimError::Serialize(error.to_string()))?;
        Self::from_json(&text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::answer_plan::{ClaimEvidence, PlanClaim};

    fn fact(subject: &str, predicate: &str, object: &str) -> FactKey {
        FactKey::new(subject, predicate, object, false)
    }

    fn draft(
        subject: &str,
        predicate: &str,
        slot: &str,
        text: &str,
        support: Vec<ClaimSupport>,
    ) -> ClaimDraft {
        ClaimDraft::new(subject, predicate, slot, text, false).with_support(support)
    }

    fn plan_with(claims: Vec<PlanClaim>) -> AnswerPlan {
        AnswerPlan::new("介绍：张三", "张三", 7, claims, 0, 0)
    }

    fn claim_with_evidence(slot: &str, text: &str, evidence: Vec<(&str, &str, &str)>) -> PlanClaim {
        PlanClaim::new(slot, "职业", text, false).with_evidence(
            evidence
                .into_iter()
                .map(|(subject, predicate, object)| {
                    ClaimEvidence::new(subject, predicate, object, false)
                })
                .collect(),
        )
    }

    #[test]
    fn identity_ignores_wording_but_tracks_evidence() {
        let support = vec![ClaimSupport::from_evidence(&ClaimEvidence::new(
            "张三", "职业", "教师", false,
        ))];
        let first =
            ClaimLedger::claim_id(&ClaimShape::new("张三", "职业", "slot:s1", false), &support);
        let second = ClaimLedger::claim_id(
            &ClaimShape::new(" 张三 ", "职业 ", " slot:s1 ", false),
            &support,
        );
        assert_eq!(first, second, "同一槽位与支撑必须得到同一身份");
        let other = ClaimLedger::claim_id(
            &ClaimShape::new("张三", "职业", "slot:s1", false),
            &[ClaimSupport::from_evidence(&ClaimEvidence::new(
                "张三", "职业", "医生", false,
            ))],
        );
        assert_ne!(first, other, "换证据必须重新判定");
    }

    #[test]
    fn registering_plan_uses_slot_and_support_stably() {
        let plan = plan_with(vec![claim_with_evidence(
            "s1",
            "张三是教师。",
            vec![("张三", "职业", "教师")],
        )]);
        let mut ledger = ClaimLedger::new();
        let first = ledger.register_plan(&plan, 1, 7);
        let second = ledger.register_plan(&plan, 2, 7);
        assert!(first.rejected.is_empty());
        assert_eq!(first.claims[0].id, second.claims[0].id);
        assert!(first.claims[0].created);
        assert!(!second.claims[0].created);
        assert_eq!(ledger.len(), 1);
        assert_eq!(ledger.events.len(), 2, "事实登记 + 命题登记，无重复事件");
    }

    #[test]
    fn rejected_plan_claim_does_not_shift_reply_origins() {
        let mut ledger = ClaimLedger::new();
        let plan = plan_with(vec![
            PlanClaim::new("s0", "", "缺谓词。", false),
            claim_with_evidence("s1", "张三是教师。", vec![("张三", "职业", "教师")]),
        ]);
        let emission = ledger.emit_plan(&plan, 9, 1, 7, &GateContext::default());
        assert_eq!(emission.rejected.len(), 1);
        assert_eq!(emission.emitted.len(), 1);
        let record = ledger
            .claim(&emission.emitted[0])
            .expect("emitted claim should exist");
        assert!(record.origins.iter().any(|origin| matches!(
            origin,
            ClaimOrigin::Reply {
                reply_seq: 9,
                index: 2
            }
        )));
    }

    #[test]
    fn fact_retirement_propagates_and_reports_origins() {
        let mut ledger = ClaimLedger::new();
        let fact_key = fact("张三", "职业", "教师");
        let parent = ledger
            .register(
                draft(
                    "张三",
                    "职业",
                    "s1",
                    "张三是教师。",
                    vec![ClaimSupport::fact(&fact_key)],
                )
                .with_origin(ClaimOrigin::plan("plan-1", "s1")),
                1,
                7,
            )
            .unwrap();
        let child_fact = fact("教师", "属于", "教育行业");
        let child = ledger
            .register(
                draft(
                    "张三",
                    "行业",
                    "s2",
                    "张三从事教育行业。",
                    vec![
                        ClaimSupport::claim(&parent.id),
                        ClaimSupport::fact(&child_fact),
                    ],
                )
                .with_origin(ClaimOrigin::plan("plan-2", "s2")),
                2,
                7,
            )
            .unwrap();

        let impact = ledger
            .retire_fact(&fact_key, RetirementKind::Retracted, "来源撤回", 3, 8)
            .unwrap();
        assert_eq!(impact.status, "retracted");
        assert_eq!(impact.affected.len(), 2);
        assert_eq!(impact.plans.len(), 2);
        assert_eq!(
            ledger.claim(&parent.id).unwrap().status,
            ClaimStatus::Retracted
        );
        assert_eq!(
            ledger.claim(&child.id).unwrap().status,
            ClaimStatus::Retracted
        );
        assert!(matches!(
            ledger
                .claim(&child.id)
                .unwrap()
                .retirement
                .as_ref()
                .unwrap()
                .cause,
            RetirementCause::Parent { .. }
        ));
        let decision = ledger.gate(&child.id, &GateContext::default());
        assert_eq!(decision.verdict, GateVerdict::Deny);
        assert!(decision
            .findings
            .iter()
            .any(|finding| finding.code == GateFindingCode::ClaimRetired));
    }

    #[test]
    fn stronger_retirement_upgrades_facts_claims_and_descendants() {
        let mut ledger = ClaimLedger::new();
        let root = fact("王五", "住址", "旧地址");
        let parent = ledger
            .register(
                draft(
                    "王五",
                    "住址",
                    "s1",
                    "王五住在旧地址。",
                    vec![ClaimSupport::fact(&root)],
                ),
                1,
                1,
            )
            .unwrap();
        let child = ledger
            .register(
                draft(
                    "王五",
                    "通勤",
                    "s2",
                    "王五通勤距离很短。",
                    vec![ClaimSupport::claim(&parent.id)],
                ),
                2,
                1,
            )
            .unwrap();

        ledger
            .retire_fact(&root, RetirementKind::Expired, "超过有效期", 3, 2)
            .unwrap();
        assert_eq!(ledger.fact(&root).unwrap().status, FactStatus::Expired);

        ledger
            .retire_fact(&root, RetirementKind::Redacted, "用户遗忘", 4, 3)
            .unwrap();
        assert_eq!(ledger.fact(&root).unwrap().status, FactStatus::Redacted);
        assert_eq!(
            ledger.claim(&parent.id).unwrap().status,
            ClaimStatus::Redacted
        );
        assert_eq!(
            ledger.claim(&child.id).unwrap().status,
            ClaimStatus::Redacted
        );

        ledger
            .retire_fact(&root, RetirementKind::Retracted, "来源撤回", 5, 4)
            .unwrap();
        assert_eq!(ledger.fact(&root).unwrap().status, FactStatus::Redacted);
        assert_eq!(
            ledger.claim(&parent.id).unwrap().status,
            ClaimStatus::Redacted
        );
        assert!(ledger.verify_chain(None).intact);
    }

    #[test]
    fn registration_against_retired_fact_cannot_launder() {
        let mut ledger = ClaimLedger::new();
        let fact_key = fact("李四", "电话", "13800138000");
        ledger
            .retire_fact(&fact_key, RetirementKind::Redacted, "用户遗忘", 1, 3)
            .unwrap();
        let registered = ledger
            .register(
                draft(
                    "李四",
                    "电话",
                    "s1",
                    "李四的电话是 13800138000。",
                    vec![ClaimSupport::fact(&fact_key)],
                ),
                2,
                4,
            )
            .unwrap();
        assert_eq!(registered.status, ClaimStatus::Redacted);
        assert_eq!(
            ledger.gate(&registered.id, &GateContext::default()).verdict,
            GateVerdict::Deny
        );
        assert_eq!(
            ledger.audit(Some(0), &AuditOptions::default()).verdict,
            AuditVerdict::Clean
        );
    }

    #[test]
    fn gate_allows_grounded_claim_and_refreshes_without_evidence() {
        let mut ledger = ClaimLedger::new();
        let fact_key = fact("王五", "职业", "医生");
        let grounded = ledger
            .register(
                draft(
                    "王五",
                    "职业",
                    "s1",
                    "王五是医生。",
                    vec![ClaimSupport::fact(&fact_key)],
                ),
                1,
                1,
            )
            .unwrap();
        let ungrounded = ledger
            .register(
                draft("王五", "职业", "s2", "王五可能还是老师。", Vec::new()),
                1,
                1,
            )
            .unwrap();
        assert_eq!(
            ledger.gate(&grounded.id, &GateContext::default()).verdict,
            GateVerdict::Allow
        );
        let decision = ledger.gate(&ungrounded.id, &GateContext::default());
        assert_eq!(decision.verdict, GateVerdict::Refresh);
        assert!(decision
            .findings
            .iter()
            .any(|finding| finding.code == GateFindingCode::MissingEvidence));
    }

    #[test]
    fn emission_through_gate_tracks_reply_and_blocks_retired_claim() {
        let mut ledger = ClaimLedger::new();
        let fact_key = fact("赵六", "职业", "工程师");
        let registered = ledger
            .register(
                draft(
                    "赵六",
                    "职业",
                    "s1",
                    "赵六是工程师。",
                    vec![ClaimSupport::fact(&fact_key)],
                ),
                1,
                1,
            )
            .unwrap();
        let decision = ledger.authorize_emission(
            &registered.id,
            ClaimOrigin::reply(11, 1),
            1,
            1,
            &GateContext::default(),
        );
        assert!(decision.allowed());
        assert_eq!(
            ledger.claim(&registered.id).unwrap().status,
            ClaimStatus::Emitted
        );
        assert!(ledger.claim(&registered.id).unwrap().ever_emitted);

        ledger
            .retire_fact(&fact_key, RetirementKind::Retracted, "更正", 2, 2)
            .unwrap();
        let denied = ledger.authorize_emission(
            &registered.id,
            ClaimOrigin::reply(12, 1),
            2,
            2,
            &GateContext::default(),
        );
        assert_eq!(denied.verdict, GateVerdict::Deny);
        assert_eq!(
            ledger.claim(&registered.id).unwrap().status,
            ClaimStatus::Retracted
        );
        assert!(!ledger
            .claim(&registered.id)
            .unwrap()
            .origins
            .iter()
            .any(|origin| matches!(origin, ClaimOrigin::Reply { reply_seq: 12, .. })));
    }

    #[test]
    fn emit_plan_only_emits_grounded_claims() {
        let plan = plan_with(vec![
            claim_with_evidence("s1", "张三是教师。", vec![("张三", "职业", "教师")]),
            PlanClaim::new("s2", "职业", "张三也写小说。", false),
        ]);
        let mut ledger = ClaimLedger::new();
        let report = ledger.emit_plan(&plan, 3, 1, 7, &GateContext::default());
        assert_eq!(report.emitted.len(), 1);
        assert_eq!(report.refused.len(), 1);
        assert!(report.refused[0]
            .findings
            .iter()
            .any(|finding| finding.code == GateFindingCode::MissingEvidence));
    }

    #[test]
    fn supersede_retires_old_and_leaves_replacement_usable() {
        let mut ledger = ClaimLedger::new();
        let old_fact = fact("张三", "职业", "教师");
        let new_fact = fact("张三", "职业", "作家");
        let old = ledger
            .register(
                draft(
                    "张三",
                    "职业",
                    "s1",
                    "张三是教师。",
                    vec![ClaimSupport::fact(&old_fact)],
                ),
                1,
                1,
            )
            .unwrap();
        let new = ledger
            .register(
                draft(
                    "张三",
                    "职业",
                    "s1",
                    "张三是作家。",
                    vec![ClaimSupport::fact(&new_fact)],
                ),
                2,
                2,
            )
            .unwrap();
        ledger
            .supersede(&old.id, &new.id, "用户更正", 2, 2)
            .unwrap();
        assert_eq!(
            ledger.gate(&old.id, &GateContext::default()).verdict,
            GateVerdict::Deny
        );
        assert_eq!(
            ledger.gate(&new.id, &GateContext::default()).verdict,
            GateVerdict::Allow
        );
    }

    #[test]
    fn audit_is_unknown_without_evidence_and_clean_when_grounded() {
        let mut ledger = ClaimLedger::new();
        let empty_report = ledger.audit(Some(0), &AuditOptions::default());
        assert_eq!(empty_report.verdict, AuditVerdict::Clean);

        let fact_key = fact("钱七", "职业", "厨师");
        let claim = ledger
            .register(
                draft(
                    "钱七",
                    "职业",
                    "s1",
                    "钱七是厨师。",
                    vec![ClaimSupport::fact(&fact_key)],
                ),
                1,
                1,
            )
            .unwrap();
        ledger.authorize_emission(
            &claim.id,
            ClaimOrigin::reply(1, 1),
            1,
            1,
            &GateContext::default(),
        );
        let report = ledger.audit(Some(1), &AuditOptions::default());
        assert_eq!(report.verdict, AuditVerdict::Clean);
        assert_eq!(report.emitted, 1);
        assert_eq!(report.ungrounded, 0);
        assert_eq!(
            ledger.audit(Some(0), &AuditOptions::default()).verdict,
            AuditVerdict::Unknown,
            "多出的已说命题同样应触发未知"
        );

        let ungrounded = ledger
            .register(
                draft("钱七", "职业", "s2", "钱七可能兼职。", Vec::new()),
                2,
                2,
            )
            .unwrap();
        let report = ledger.audit(Some(1), &AuditOptions::default());
        assert_eq!(report.verdict, AuditVerdict::Unknown);
        assert!(report
            .unknown_reasons
            .iter()
            .any(|reason| reason.contains(&ungrounded.id)));
    }

    #[test]
    fn audit_digest_is_stable_across_roundtrip() {
        let mut ledger = ClaimLedger::new();
        let fact_key = fact("孙八", "职业", "画家");
        let claim = ledger
            .register(
                draft(
                    "孙八",
                    "职业",
                    "s1",
                    "孙八是画家。",
                    vec![ClaimSupport::fact(&fact_key)],
                ),
                1,
                1,
            )
            .unwrap();
        ledger.authorize_emission(
            &claim.id,
            ClaimOrigin::reply(4, 1),
            1,
            1,
            &GateContext::default(),
        );
        ledger
            .retire_fact(&fact_key, RetirementKind::Retracted, "来源撤回", 2, 2)
            .unwrap();
        let first = ledger.audit(Some(1), &AuditOptions::default());
        assert_eq!(first.verdict, AuditVerdict::Clean);
        assert_eq!(first.archived, 1);
        let encoded = ledger.to_json().unwrap();
        let rebuilt = ClaimLedger::from_json(&encoded).unwrap();
        let second = rebuilt.audit(Some(1), &AuditOptions::default());
        assert_eq!(first, second);
        assert_eq!(first.digest, second.digest);
    }

    #[test]
    fn chain_detects_tampering_and_window_truncation_is_unknown() {
        let mut ledger = ClaimLedger::new();
        let fact_key = fact("周九", "职业", "司机");
        ledger
            .register(
                draft(
                    "周九",
                    "职业",
                    "s1",
                    "周九是司机。",
                    vec![ClaimSupport::fact(&fact_key)],
                ),
                1,
                1,
            )
            .unwrap();
        assert!(ledger.verify_chain(None).intact);
        ledger.events[0].reason = "被篡改".to_string();
        let broken = ledger.verify_chain(None);
        assert!(!broken.intact);
        assert_eq!(broken.broken_at, Some(1));
        let report = ledger.audit(Some(0), &AuditOptions::default());
        assert_eq!(report.verdict, AuditVerdict::Unknown);

        let mut fresh = ClaimLedger::new();
        fresh
            .register(
                draft(
                    "周九",
                    "职业",
                    "s1",
                    "周九是司机。",
                    vec![ClaimSupport::fact(&fact_key)],
                ),
                1,
                1,
            )
            .unwrap();
        let options = AuditOptions::default().with_event_window(1);
        let report = fresh.audit(Some(0), &options);
        assert_eq!(report.verdict, AuditVerdict::Unknown);
        assert!(report.chain.truncated);
    }

    #[test]
    fn external_tombstones_deny_emission_without_mutating_ledger() {
        let mut ledger = ClaimLedger::new();
        let fact_key = fact("吴十", "住址", "某街 1 号");
        let claim = ledger
            .register(
                draft(
                    "吴十",
                    "住址",
                    "s1",
                    "吴十住在某街 1 号。",
                    vec![ClaimSupport::fact(&fact_key)],
                ),
                1,
                1,
            )
            .unwrap();
        let context = GateContext::default().with_retired_facts([fact_key.clone()]);
        let decision = ledger.gate(&claim.id, &context);
        assert_eq!(decision.verdict, GateVerdict::Deny);
        assert!(decision
            .findings
            .iter()
            .any(|finding| finding.code == GateFindingCode::ExternalTombstone));
        assert_eq!(
            ledger.claim(&claim.id).unwrap().status,
            ClaimStatus::Planned
        );
        let audit = ledger.audit(
            Some(0),
            &AuditOptions::default().with_retired_facts([fact_key]),
        );
        assert_eq!(audit.verdict, AuditVerdict::Findings);
    }

    #[test]
    fn session_facts_do_not_merge_with_graph_facts() {
        let graph = fact("郑十一", "职业", "编辑");
        let session = FactKey::new("郑十一", "职业", "编辑", true);
        let mut ledger = ClaimLedger::new();
        let graph_claim = ledger
            .register(
                draft(
                    "郑十一",
                    "职业",
                    "s1",
                    "郑十一是编辑。",
                    vec![ClaimSupport::fact(&graph)],
                ),
                1,
                1,
            )
            .unwrap();
        let session_claim = ledger
            .register(
                ClaimDraft::new("郑十一", "职业", "s1", "郑十一是编辑。", true)
                    .with_support([ClaimSupport::fact(&session)]),
                1,
                1,
            )
            .unwrap();
        assert_ne!(graph_claim.id, session_claim.id);
        ledger
            .retire_fact(&graph, RetirementKind::Retracted, "图谱更正", 2, 2)
            .unwrap();
        assert_eq!(
            ledger.claim(&graph_claim.id).unwrap().status,
            ClaimStatus::Retracted
        );
        assert_eq!(
            ledger.claim(&session_claim.id).unwrap().status,
            ClaimStatus::Planned
        );
    }

    #[test]
    fn revalidation_never_revives_terminal_or_unsupported_claims() {
        let mut ledger = ClaimLedger::new();
        let fact_key = fact("冯十二", "职业", "会计");
        let claim = ledger
            .register(
                draft(
                    "冯十二",
                    "职业",
                    "s1",
                    "冯十二是会计。",
                    vec![ClaimSupport::fact(&fact_key)],
                ),
                1,
                1,
            )
            .unwrap();
        assert_eq!(ledger.revalidate(&claim.id, 2, 2), Ok(ClaimStatus::Planned));
        ledger
            .retire_fact(&fact_key, RetirementKind::Expired, "超过有效期", 3, 3)
            .unwrap();
        assert_eq!(
            ledger.revalidate(&claim.id, 4, 4),
            Err(ClaimError::StillUnsupported(claim.id))
        );
    }

    #[test]
    fn plan_registration_rejects_unknown_parent_instead_of_silently_dropping() {
        let plan = plan_with(vec![PlanClaim::new("s1", "职业", "张三是教师。", false)]);
        let mut ledger = ClaimLedger::new();
        let registration = ledger.register_plan(&plan, 1, 1);
        assert_eq!(registration.claims.len(), 1);
        assert!(registration.rejected.is_empty());
        let mut bad = ClaimDraft::new("张三", "职业", "s2", "张三是教师。", false)
            .with_support([ClaimSupport::claim("clm_deadbeefdeadbeef")]);
        bad.origin = None;
        assert_eq!(
            ledger.register(bad, 2, 2),
            Err(ClaimError::UnknownParent(
                "clm_deadbeefdeadbeef".to_string()
            ))
        );
    }

    #[test]
    fn residue_scan_separates_active_from_archival() {
        let surfaces = vec![
            Surface::active("当前计划", "张三住在某街 1 号，电话 13800138000。"),
            Surface::archival("编年史", "旧回答：李四的电话是 13900139000。"),
        ];
        let terms = vec!["13800138000".to_string(), "13900139000".to_string()];
        let report = scan_residue(&surfaces, &terms);
        assert_eq!(report.verdict, AuditVerdict::Findings);
        assert_eq!(report.active_hits.len(), 1);
        assert_eq!(report.active_hits[0].surface, "当前计划");
        assert_eq!(report.archival_hits.len(), 1);
        assert_eq!(report.archival_hits[0].surface, "编年史");
    }

    #[test]
    fn residue_scan_does_not_claim_clean_without_terms_or_active_surfaces() {
        let surfaces = vec![Surface::archival("编年史", "旧回答")];
        let report = scan_residue(&surfaces, &[]);
        assert_eq!(report.verdict, AuditVerdict::Unknown);
        assert!(report
            .unknown_reasons
            .iter()
            .any(|reason| reason.contains("明文")));

        let report = scan_residue(&[], &["张三".to_string()]);
        assert_eq!(report.verdict, AuditVerdict::Unknown);

        let surfaces = vec![Surface::active("工作记忆", "与目标实体无关的文本")];
        let report = scan_residue(&surfaces, &["张三".to_string()]);
        assert_eq!(report.verdict, AuditVerdict::Clean);
        assert!(report.active_hits.is_empty());
    }

    #[test]
    fn fact_keys_and_claim_ids_disambiguate_embedded_separators() {
        let left = FactKey::new("a b", "c", "d", false);
        let right = FactKey::new("a", "b c", "d", false);
        assert_ne!(left.key(), right.key());
        let left_shape = ClaimShape::new("x", "p", "slot:s", false);
        let right_shape = ClaimShape::new("x", "p", "slot:s", false);
        assert_ne!(
            ClaimLedger::claim_id(&left_shape, &[ClaimSupport::fact(&left)]),
            ClaimLedger::claim_id(&right_shape, &[ClaimSupport::fact(&right)])
        );
    }

    #[test]
    fn privacy_tombstones_retire_supporting_claims_conservatively() {
        let phone = FactKey::new("李四", "电话", "13800138000", false);
        let job = FactKey::new("李四", "职业", "教师", false);
        let mut privacy = PrivacyLedger::new();
        privacy.forget_fact("李四", "电话", "13800138000", "用户要求遗忘联系方式");
        let retired = privacy_retired_facts(&privacy, [phone.clone(), job.clone()]);
        assert!(retired.contains(&phone));
        assert!(!retired.contains(&job));

        let mut ledger = ClaimLedger::new();
        let phone_claim = ledger
            .register(
                draft(
                    "李四",
                    "电话",
                    "s1",
                    "李四的电话是 13800138000。",
                    vec![ClaimSupport::fact(&phone)],
                ),
                1,
                1,
            )
            .unwrap();
        let job_claim = ledger
            .register(
                draft(
                    "李四",
                    "职业",
                    "s2",
                    "李四是教师。",
                    vec![ClaimSupport::fact(&job)],
                ),
                1,
                1,
            )
            .unwrap();
        let impacts =
            ledger.retire_with_privacy(&privacy, &[phone.clone(), job.clone()], "隐私台账", 2, 2);
        assert_eq!(impacts.len(), 1);
        assert_eq!(
            ledger.claim(&phone_claim.id).unwrap().status,
            ClaimStatus::Redacted
        );
        assert_eq!(
            ledger.claim(&job_claim.id).unwrap().status,
            ClaimStatus::Planned
        );
        let context =
            GateContext::default().with_privacy_ledger(&privacy, &[phone.clone(), job.clone()]);
        assert_eq!(
            ledger.gate(&phone_claim.id, &context).verdict,
            GateVerdict::Deny
        );
        assert_eq!(
            ledger.gate(&job_claim.id, &context).verdict,
            GateVerdict::Allow
        );
    }
}

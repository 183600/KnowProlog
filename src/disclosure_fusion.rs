//! 披露融合解码器（Disclosure Fusion Decoder，DFD）。
//!
//! 该模块把「联想通道」与「演绎通道」的候选命题融合为带保证书
//! （warrant）的命题集合，再对整段回答做披露检查与最小修复：
//!
//! - 单命题裁决复用 [`PrivacyAdjudicator`]，沿用策略上限、授权、墓碑与
//!   有界推导闭包；
//! - 联合 k-匿名按准标识符组合计数，而不是只看单个谓词；
//! - 整段回答在「所有可说命题」的并集上再做一次推导闭包，检测单条命题
//!   看不出的组合泄漏；
//! - 修复动作写回 [`GenerationOverlay`]，先隐藏、再替换，绝不先写敏感句
//!   再做字符串替换；
//! - 闭包被预算截断时保守阻断，并显式报告 `closure_truncated`。
//!
//! 所有推导都是确定性的：候选、泄漏、修复与报告均按固定键排序，不依赖
//! 挂钟、随机数或哈希遍历顺序。模块本身不写隐私台账；调用方拿到报告后
//! 自行决定是否落盘。

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::answer_plan::AnswerPlan;
use crate::privacy_flow::{
    coarsen_value, DecisionCode, DisclosureDecision, DisclosurePrecision, DisclosureRequest,
    DisclosureStatus, PolicyMode, PrivacyAdjudicator, PrivacyCategory, PrivacyFact, PrivacyGraph,
    PrivacyLedger, PrivacyPolicy,
};
use crate::symbolic_generator::GenerationOverlay;

/// 一次融合最多保留的命题数，与回答计划上限保持一致。
pub const DEFAULT_MAX_FUSED_CLAIMS: usize = 24;
/// 最小命中集搜索的默认节点预算；超出后回退到确定性贪心。
pub const DEFAULT_HITTING_SET_BUDGET: usize = 20_000;

/// 命题候选来自哪条通道。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceChannel {
    /// 稠密检索、全息记忆、图游走、关联规则等广覆盖召回。
    Associative,
    /// 归结证明、语义蕴含、类型/本体约束等审慎校验。
    Deductive,
}

impl EvidenceChannel {
    pub const fn code(self) -> &'static str {
        match self {
            EvidenceChannel::Associative => "associative",
            EvidenceChannel::Deductive => "deductive",
        }
    }
}

/// 保证书等级。排序只表达「可采纳性」，不做加权平均。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WarrantLevel {
    /// 只有联想召回，未被任何演绎检查确认。
    Associative,
    /// 只有演绎通道。
    Deductive,
    /// 两通道同时到达同一命题。
    Corroborated,
}

impl WarrantLevel {
    pub const fn code(self) -> &'static str {
        match self {
            WarrantLevel::Associative => "associative",
            WarrantLevel::Deductive => "deductive",
            WarrantLevel::Corroborated => "corroborated",
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            WarrantLevel::Associative => "联想召回",
            WarrantLevel::Deductive => "演绎确认",
            WarrantLevel::Corroborated => "双通道互证",
        }
    }

    /// 是否足以进入肯定语气回答。联想召回默认只作候选。
    pub const fn is_admissible(self) -> bool {
        matches!(self, WarrantLevel::Deductive | WarrantLevel::Corroborated)
    }
}

/// 一条命题所依据的三元组。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ClaimAtom {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

impl ClaimAtom {
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
        }
    }

    pub fn slot(&self) -> DisclosureSlot {
        DisclosureSlot {
            subject: self.subject.clone(),
            predicate: self.predicate.clone(),
        }
    }
}

/// 披露槽位：生成叠加层以「主语 + 谓词」为单位隐藏或替换。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct DisclosureSlot {
    pub subject: String,
    pub predicate: String,
}

impl DisclosureSlot {
    pub fn new(subject: impl Into<String>, predicate: impl Into<String>) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
        }
    }

    pub fn render(&self) -> String {
        format!("{} {}", self.subject, self.predicate)
    }
}

/// 一个待融合的命题候选。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaimCandidate {
    /// 稳定命题 id；同一 id 的两通道候选会被融合。
    pub claim_id: String,
    pub text: String,
    pub atoms: Vec<ClaimAtom>,
    pub channel: EvidenceChannel,
    /// 来源标签；排序后参与保证书，不参与打分。
    pub sources: BTreeSet<String>,
}

impl ClaimCandidate {
    pub fn new(
        claim_id: impl Into<String>,
        text: impl Into<String>,
        atoms: Vec<ClaimAtom>,
        channel: EvidenceChannel,
    ) -> Self {
        Self {
            claim_id: claim_id.into(),
            text: text.into(),
            atoms,
            channel,
            sources: BTreeSet::new(),
        }
    }

    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.sources.insert(source.into());
        self
    }

    pub fn with_sources<I, S>(mut self, sources: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.sources.extend(sources.into_iter().map(Into::into));
        self
    }
}

/// 命题保证书：为什么信。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaimWarrant {
    pub level: WarrantLevel,
    pub channels: BTreeSet<EvidenceChannel>,
    pub sources: BTreeSet<String>,
}

impl ClaimWarrant {
    fn from_channels(channels: BTreeSet<EvidenceChannel>, sources: BTreeSet<String>) -> Self {
        let level = match (
            channels.contains(&EvidenceChannel::Associative),
            channels.contains(&EvidenceChannel::Deductive),
        ) {
            (true, true) => WarrantLevel::Corroborated,
            (false, true) => WarrantLevel::Deductive,
            _ => WarrantLevel::Associative,
        };
        Self {
            level,
            channels,
            sources,
        }
    }

    pub fn is_admissible(&self) -> bool {
        self.level.is_admissible()
    }

    /// 演绎通道的保证书。
    pub fn deductive(source: impl Into<String>) -> Self {
        Self::from_channels(
            [EvidenceChannel::Deductive].into_iter().collect(),
            [source.into()].into_iter().collect(),
        )
    }

    /// 联想通道的保证书。
    pub fn associative(source: impl Into<String>) -> Self {
        Self::from_channels(
            [EvidenceChannel::Associative].into_iter().collect(),
            [source.into()].into_iter().collect(),
        )
    }

    /// 两通道互证的保证书。
    pub fn corroborated<I, S>(sources: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self::from_channels(
            [EvidenceChannel::Associative, EvidenceChannel::Deductive]
                .into_iter()
                .collect(),
            sources.into_iter().map(Into::into).collect(),
        )
    }
}

/// 融合后的命题：带保证书与去重后的证据原子。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FusedClaim {
    pub claim_id: String,
    pub text: String,
    pub atoms: Vec<ClaimAtom>,
    pub warrant: ClaimWarrant,
}

impl FusedClaim {
    /// 由演绎通道直接建立命题。
    pub fn deductive(
        claim_id: impl Into<String>,
        text: impl Into<String>,
        atoms: Vec<ClaimAtom>,
        source: impl Into<String>,
    ) -> Self {
        Self {
            claim_id: claim_id.into(),
            text: text.into(),
            atoms,
            warrant: ClaimWarrant::deductive(source),
        }
    }
}

/// 被拒绝的候选及稳定原因码。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RejectedCandidate {
    pub claim_id: String,
    pub channel: EvidenceChannel,
    pub reason: String,
}

/// 融合结果。拒绝项也保留，便于审计召回损失。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FusionReport {
    pub accepted: Vec<FusedClaim>,
    pub rejected: Vec<RejectedCandidate>,
    pub notes: Vec<String>,
}

impl FusionReport {
    pub fn is_empty(&self) -> bool {
        self.accepted.is_empty()
    }

    pub fn render_zh(&self) -> String {
        let mut text = format!(
            "融合：采纳 {} 条，拒绝 {} 条",
            self.accepted.len(),
            self.rejected.len()
        );
        if !self.notes.is_empty() {
            text.push_str(&format!("；注记 {} 条", self.notes.len()));
        }
        text
    }
}

struct MergedCandidate {
    claim_id: String,
    texts: BTreeMap<EvidenceChannel, String>,
    atoms: BTreeSet<ClaimAtom>,
    channels: BTreeSet<EvidenceChannel>,
    sources: BTreeSet<String>,
}

/// 融合联想与演绎候选。
///
/// 默认要求演绎确认：仅有联想召回的候选不进入肯定语气回答，但仍出现在
/// `rejected` 中供后续澄清或再检索。设置 `require_deductive_warrant=false`
/// 可把联想候选以 [`WarrantLevel::Associative`] 放行。
pub fn fuse_candidates(
    associative: &[ClaimCandidate],
    deductive: &[ClaimCandidate],
    require_deductive_warrant: bool,
) -> FusionReport {
    let mut merged: BTreeMap<String, MergedCandidate> = BTreeMap::new();
    let mut rejected = Vec::new();
    let mut notes = Vec::new();

    for candidate in associative.iter().chain(deductive.iter()) {
        if candidate.atoms.is_empty() {
            rejected.push(RejectedCandidate {
                claim_id: candidate.claim_id.clone(),
                channel: candidate.channel,
                reason: "missing_evidence".to_string(),
            });
            continue;
        }
        let entry = merged
            .entry(candidate.claim_id.clone())
            .or_insert_with(|| MergedCandidate {
                claim_id: candidate.claim_id.clone(),
                texts: BTreeMap::new(),
                atoms: BTreeSet::new(),
                channels: BTreeSet::new(),
                sources: BTreeSet::new(),
            });
        if let Some(existing) = entry.texts.get(&candidate.channel) {
            if existing != &candidate.text {
                notes.push(format!(
                    "claim:{}:{}:text_divergence",
                    candidate.claim_id,
                    candidate.channel.code()
                ));
                if candidate.text < *existing {
                    entry
                        .texts
                        .insert(candidate.channel, candidate.text.clone());
                }
            }
        } else {
            entry
                .texts
                .insert(candidate.channel, candidate.text.clone());
        }
        entry.atoms.extend(candidate.atoms.iter().cloned());
        entry.channels.insert(candidate.channel);
        entry.sources.extend(candidate.sources.iter().cloned());
    }

    let mut accepted = Vec::new();
    let mut merged_values: Vec<MergedCandidate> = merged.into_values().collect();
    merged_values.sort_by(|left, right| left.claim_id.cmp(&right.claim_id));

    for entry in merged_values {
        if entry.channels.len() > 1 {
            let associative_text = entry.texts.get(&EvidenceChannel::Associative);
            let deductive_text = entry.texts.get(&EvidenceChannel::Deductive);
            if let (Some(left), Some(right)) = (associative_text, deductive_text) {
                if left != right {
                    notes.push(format!("claim:{}:channel_text_divergence", entry.claim_id));
                }
            }
        }
        let text = entry
            .texts
            .get(&EvidenceChannel::Deductive)
            .or_else(|| entry.texts.get(&EvidenceChannel::Associative))
            .cloned()
            .unwrap_or_default();
        let warrant = ClaimWarrant::from_channels(entry.channels.clone(), entry.sources);
        if !warrant.is_admissible() {
            if require_deductive_warrant {
                for channel in &entry.channels {
                    rejected.push(RejectedCandidate {
                        claim_id: entry.claim_id.clone(),
                        channel: *channel,
                        reason: "needs_deductive_warrant".to_string(),
                    });
                }
                continue;
            }
        }
        accepted.push(FusedClaim {
            claim_id: entry.claim_id,
            text,
            atoms: entry.atoms.into_iter().collect(),
            warrant,
        });
    }

    accepted.sort_by(|left, right| left.claim_id.cmp(&right.claim_id));
    rejected.sort_by(|left, right| {
        left.claim_id
            .cmp(&right.claim_id)
            .then_with(|| left.channel.cmp(&right.channel))
            .then_with(|| left.reason.cmp(&right.reason))
    });
    notes.sort();
    notes.dedup();
    if accepted.len() > DEFAULT_MAX_FUSED_CLAIMS {
        notes.push(format!(
            "fusion_truncated:{}",
            accepted.len() - DEFAULT_MAX_FUSED_CLAIMS
        ));
        accepted.truncate(DEFAULT_MAX_FUSED_CLAIMS);
    }

    FusionReport {
        accepted,
        rejected,
        notes,
    }
}

/// 从回答计划读取命题候选。计划的命题已通过生成自检，归入演绎通道。
pub fn candidates_from_plan(plan: &AnswerPlan, channel: EvidenceChannel) -> Vec<ClaimCandidate> {
    plan.claims
        .iter()
        .enumerate()
        .map(|(index, claim)| {
            let atoms: Vec<ClaimAtom> = claim
                .evidence
                .iter()
                .map(|evidence| {
                    ClaimAtom::new(
                        evidence.subject.clone(),
                        evidence.predicate.clone(),
                        evidence.object.clone(),
                    )
                })
                .collect();
            let claim_id = if claim.slot.is_empty() {
                format!("plan-claim-{:02}", index)
            } else {
                claim.slot.clone()
            };
            ClaimCandidate {
                claim_id,
                text: claim.text.clone(),
                atoms,
                channel,
                sources: ["answer_plan".to_string()].into_iter().collect(),
            }
        })
        .collect()
}

/// 回答级披露配置。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnswerDisclosureConfig {
    /// 对回答中每个事实请求的精度。
    pub requested_precision: DisclosurePrecision,
    /// 参与联合 k-匿名的准标识符谓词；为空时不启用联合检查。
    pub quasi_identifiers: BTreeSet<String>,
    /// 最小命中集搜索的节点预算。
    pub max_hitting_set_nodes: usize,
}

impl Default for AnswerDisclosureConfig {
    fn default() -> Self {
        Self {
            requested_precision: DisclosurePrecision::Exact,
            quasi_identifiers: BTreeSet::new(),
            max_hitting_set_nodes: DEFAULT_HITTING_SET_BUDGET,
        }
    }
}

impl AnswerDisclosureConfig {
    pub fn with_quasi_identifiers<I, S>(mut self, predicates: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.quasi_identifiers = predicates.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_requested_precision(mut self, precision: DisclosurePrecision) -> Self {
        self.requested_precision = precision;
        self
    }
}

/// 单条证据原子的裁决结果。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnswerAtomDecision {
    pub claim_id: String,
    pub atom: ClaimAtom,
    pub decision: DisclosureDecision,
}

/// 联合 k-匿名的检查结果。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JointAnonymityFinding {
    pub subject: String,
    pub predicates: Vec<String>,
    pub class_size: usize,
    pub required: usize,
    pub precision: BTreeMap<String, DisclosurePrecision>,
}

/// 整段回答触发的组合推导泄漏。只描述谓词与前提槽位，不把结论对象
/// 当作可披露内容。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnswerLeak {
    pub rule: String,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub category: PrivacyCategory,
    pub max_precision: DisclosurePrecision,
    /// 推出该结论所依赖的基础事实。
    pub premises: Vec<ClaimAtom>,
}

impl AnswerLeak {
    pub fn render_zh(&self) -> String {
        format!(
            "{}（{} 可推出 {}，上限 {}）",
            self.rule,
            self.category.label(),
            self.predicate,
            self.max_precision.label()
        )
    }
}

/// 修复原因。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RepairReason {
    DisclosureDecision,
    JointKAnonymity,
    InferenceLeak,
    ConservativeBlock,
}

impl RepairReason {
    pub const fn code(self) -> &'static str {
        match self {
            RepairReason::DisclosureDecision => "disclosure_decision",
            RepairReason::JointKAnonymity => "joint_k_anonymity",
            RepairReason::InferenceLeak => "inference_leak",
            RepairReason::ConservativeBlock => "conservative_block",
        }
    }
}

/// 一个槽位的修复动作。`precision == Hidden` 表示隐藏整个槽位。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlotRepair {
    pub slot: DisclosureSlot,
    pub precision: DisclosurePrecision,
    pub replacements: Vec<String>,
    pub reason: RepairReason,
}

/// 整段回答的披露报告。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnswerDisclosureReport {
    pub mode: PolicyMode,
    pub requested_precision: DisclosurePrecision,
    pub claim_count: usize,
    pub atom_decisions: Vec<AnswerAtomDecision>,
    pub joint_findings: Vec<JointAnonymityFinding>,
    pub leaks: Vec<AnswerLeak>,
    pub repairs: Vec<SlotRepair>,
    pub removed_claims: Vec<String>,
    pub blocked: bool,
    pub closure_truncated: bool,
    pub hitting_set_exhausted: bool,
    pub notes: Vec<String>,
}

impl AnswerDisclosureReport {
    /// 是否可在不依赖外部授权的前提下披露。
    pub fn is_safe(&self) -> bool {
        !self.blocked
    }

    /// 把修复动作写回生成叠加层。审计模式不改变生成结果。
    pub fn apply_to_overlay(&self, overlay: &mut GenerationOverlay) {
        if self.mode == PolicyMode::Audit {
            return;
        }
        for repair in &self.repairs {
            let key = (repair.slot.subject.clone(), repair.slot.predicate.clone());
            if repair.precision == DisclosurePrecision::Hidden || self.blocked {
                overlay.hide.insert(key);
                overlay
                    .replace
                    .remove(&(repair.slot.subject.clone(), repair.slot.predicate.clone()));
            } else if !repair.replacements.is_empty() {
                overlay.replace.insert(key, repair.replacements.clone());
            }
        }
    }

    pub fn render_zh(&self) -> String {
        let mut text = format!(
            "答案级披露：{} 条命题，{} 条证据；模式 {}，请求精度 {}；修复 {} 个槽位，移除 {} 条命题",
            self.claim_count,
            self.atom_decisions.len(),
            self.mode.label(),
            self.requested_precision.label(),
            self.repairs.len(),
            self.removed_claims.len()
        );
        if self.blocked {
            text.push_str("；保守阻断（无法在不泄漏的前提下回答）");
        }
        if self.closure_truncated {
            text.push_str("；闭包被预算截断");
        }
        if !self.joint_findings.is_empty() {
            text.push_str(&format!(
                "；联合 k-匿名命中 {} 组",
                self.joint_findings.len()
            ));
        }
        if !self.leaks.is_empty() {
            text.push_str(&format!("；组合泄漏 {} 条", self.leaks.len()));
        }
        if self.hitting_set_exhausted {
            text.push_str("；最小命中集搜索回退到贪心");
        }
        const DETAIL_LIMIT: usize = 8;
        for finding in self.joint_findings.iter().take(DETAIL_LIMIT) {
            text.push_str(&format!(
                "\n  联合 k-匿名：{} 的 {}，等价类 {}/{}",
                finding.subject,
                finding.predicates.join("、"),
                finding.class_size,
                finding.required
            ));
        }
        if self.joint_findings.len() > DETAIL_LIMIT {
            text.push_str(&format!(
                "\n  联合 k-匿名：另有 {} 组未展开",
                self.joint_findings.len() - DETAIL_LIMIT
            ));
        }
        for leak in self.leaks.iter().take(DETAIL_LIMIT) {
            text.push_str(&format!("\n  组合泄漏：{}", leak.render_zh()));
        }
        if self.leaks.len() > DETAIL_LIMIT {
            text.push_str(&format!(
                "\n  组合泄漏：另有 {} 条未展开",
                self.leaks.len() - DETAIL_LIMIT
            ));
        }
        for repair in self.repairs.iter().take(DETAIL_LIMIT) {
            if repair.precision == DisclosurePrecision::Hidden {
                text.push_str(&format!(
                    "\n  修复：隐藏 {}（{}）",
                    repair.slot.render(),
                    repair.reason.code()
                ));
            } else {
                text.push_str(&format!(
                    "\n  修复：{} 降至 {}（{}，{} 个替换值）",
                    repair.slot.render(),
                    repair.precision.label(),
                    repair.reason.code(),
                    repair.replacements.len()
                ));
            }
        }
        if self.repairs.len() > DETAIL_LIMIT {
            text.push_str(&format!(
                "\n  修复：另有 {} 个槽位未展开",
                self.repairs.len() - DETAIL_LIMIT
            ));
        }
        for claim_id in self.removed_claims.iter().take(DETAIL_LIMIT) {
            text.push_str(&format!("\n  移除命题：{}", claim_id));
        }
        if self.removed_claims.len() > DETAIL_LIMIT {
            text.push_str(&format!(
                "\n  移除命题：另有 {} 条未展开",
                self.removed_claims.len() - DETAIL_LIMIT
            ));
        }
        text
    }
}

/// 披露融合解码器：纯函数入口。
///
/// 输入策略、只读图谱、只读台账与已融合命题；输出披露报告与修复动作。
/// 内部使用台账副本做裁决，调用方可以放心地先预览再决定是否提交。
pub fn decode_answer(
    policy: &PrivacyPolicy,
    graph: &PrivacyGraph,
    ledger: &PrivacyLedger,
    claims: &[FusedClaim],
    config: &AnswerDisclosureConfig,
) -> AnswerDisclosureReport {
    let mut notes = Vec::new();
    let mut working_ledger = ledger.clone();

    let mut unique_atoms: BTreeMap<ClaimAtom, Vec<String>> = BTreeMap::new();
    for claim in claims {
        for atom in &claim.atoms {
            unique_atoms
                .entry(atom.clone())
                .or_default()
                .push(claim.claim_id.clone());
        }
    }

    let mut atom_decisions = Vec::new();
    let mut decision_by_atom: BTreeMap<ClaimAtom, (DisclosureStatus, DisclosurePrecision)> =
        BTreeMap::new();
    for (atom, mut claim_ids) in unique_atoms {
        claim_ids.sort();
        claim_ids.dedup();
        let request = DisclosureRequest::new(&atom.subject, &atom.predicate, &atom.object)
            .with_precision(config.requested_precision);
        let decision =
            PrivacyAdjudicator::new(policy, graph, &mut working_ledger).adjudicate(&request);
        let (status, precision, _) = effective_decision(&decision);
        decision_by_atom.insert(atom.clone(), (status, precision));
        for claim_id in claim_ids {
            atom_decisions.push(AnswerAtomDecision {
                claim_id,
                atom: atom.clone(),
                decision: decision.clone(),
            });
        }
    }
    atom_decisions.sort_by(|left, right| {
        left.claim_id
            .cmp(&right.claim_id)
            .then_with(|| left.atom.cmp(&right.atom))
    });

    let mut repairs: BTreeMap<DisclosureSlot, SlotRepair> = BTreeMap::new();
    let mut removed_claims: BTreeSet<String> = BTreeSet::new();
    for item in &atom_decisions {
        let (status, precision, codes) = effective_decision(&item.decision);
        let slot = item.atom.slot();
        if status == DisclosureStatus::Blocked || precision == DisclosurePrecision::Hidden {
            removed_claims.insert(item.claim_id.clone());
            merge_repair(
                &mut repairs,
                SlotRepair {
                    slot,
                    precision: DisclosurePrecision::Hidden,
                    replacements: Vec::new(),
                    reason: RepairReason::DisclosureDecision,
                },
            );
            continue;
        }
        if codes.contains(&DecisionCode::ClosureTruncated) {
            notes.push("individual_closure_truncated".to_string());
        }
        if precision < config.requested_precision {
            let replacement = item.decision.replacement.clone().or_else(|| {
                coarsen_value(item.decision.category, precision, &item.atom.object, None)
            });
            match replacement {
                Some(value) => merge_repair(
                    &mut repairs,
                    SlotRepair {
                        slot,
                        precision,
                        replacements: vec![value],
                        reason: RepairReason::DisclosureDecision,
                    },
                ),
                None => {
                    removed_claims.insert(item.claim_id.clone());
                    merge_repair(
                        &mut repairs,
                        SlotRepair {
                            slot,
                            precision: DisclosurePrecision::Hidden,
                            replacements: Vec::new(),
                            reason: RepairReason::DisclosureDecision,
                        },
                    );
                }
            }
        }
    }

    let mut adjusted_precision: BTreeMap<ClaimAtom, DisclosurePrecision> = BTreeMap::new();
    for atom in decision_by_atom.keys() {
        let precision = decision_by_atom
            .get(atom)
            .map(|(_, precision)| *precision)
            .unwrap_or(DisclosurePrecision::Hidden);
        adjusted_precision.insert(atom.clone(), precision);
    }

    let joint_findings = audit_joint_k(
        policy,
        graph,
        claims,
        &adjusted_precision,
        &config.quasi_identifiers,
        &mut repairs,
        &mut removed_claims,
    );

    let combined = combined_facts(policy, graph, claims, &adjusted_precision);
    let closure = derive_closure(&combined, policy);
    let answer_indexes: BTreeSet<usize> = combined
        .iter()
        .enumerate()
        .filter(|(_, fact)| {
            claims
                .iter()
                .any(|claim| claim.atoms.iter().any(|atom| atom_matches_fact(atom, fact)))
        })
        .map(|(index, _)| index)
        .collect();
    let mut leaks = Vec::new();
    for item in closure
        .items
        .iter()
        .filter(|item| item.derived && item.fact.category.is_personal())
    {
        if item.max_precision >= config.requested_precision {
            continue;
        }
        if combined.iter().any(|known| same_fact(known, &item.fact)) {
            continue;
        }
        if item.provenance.is_disjoint(&answer_indexes) {
            continue;
        }
        let premises: Vec<ClaimAtom> = item
            .provenance
            .iter()
            .filter_map(|index| combined.get(*index))
            .map(|fact| ClaimAtom::new(&fact.subject, &fact.predicate, &fact.object))
            .collect();
        leaks.push(AnswerLeak {
            rule: item.rule.clone().unwrap_or_else(|| "组合推导".to_string()),
            subject: item.fact.subject.clone(),
            predicate: item.fact.predicate.clone(),
            object: item.fact.object.clone(),
            category: item.fact.category,
            max_precision: item.max_precision,
            premises,
        });
    }
    leaks.sort_by(|left, right| {
        left.subject
            .cmp(&right.subject)
            .then_with(|| left.predicate.cmp(&right.predicate))
            .then_with(|| left.rule.cmp(&right.rule))
            .then_with(|| left.object.cmp(&right.object))
    });
    leaks.dedup_by(|left, right| {
        left.rule == right.rule
            && left.subject == right.subject
            && left.predicate == right.predicate
            && left.object == right.object
    });

    let mut blocked = false;
    let mut hitting_set_exhausted = false;
    if !leaks.is_empty() {
        let answer_slots: BTreeMap<ClaimAtom, DisclosureSlot> = claims
            .iter()
            .flat_map(|claim| claim.atoms.iter())
            .filter(|atom| {
                adjusted_precision
                    .get(*atom)
                    .copied()
                    .unwrap_or(DisclosurePrecision::Hidden)
                    != DisclosurePrecision::Hidden
            })
            .map(|atom| (atom.clone(), atom.slot()))
            .collect();
        let mut premise_sets: Vec<BTreeSet<DisclosureSlot>> = Vec::new();
        let mut unrepairable = false;
        for leak in &leaks {
            let mut slots = BTreeSet::new();
            for premise in &leak.premises {
                if let Some(slot) = answer_slots.get(premise) {
                    slots.insert(slot.clone());
                }
            }
            if slots.is_empty() {
                unrepairable = true;
                continue;
            }
            premise_sets.push(slots);
        }
        if unrepairable {
            notes.push("unrepairable:premises_outside_answer".to_string());
            blocked = true;
        }
        if !premise_sets.is_empty() {
            let weights = slot_weights(claims, &answer_slots);
            let (chosen, exhausted) =
                solve_hitting_set(&premise_sets, &weights, config.max_hitting_set_nodes.max(1));
            hitting_set_exhausted = exhausted;
            for slot in &chosen {
                for claim in claims {
                    if claim.atoms.iter().any(|atom| atom.slot() == *slot) {
                        removed_claims.insert(claim.claim_id.clone());
                    }
                }
                merge_repair(
                    &mut repairs,
                    SlotRepair {
                        slot: slot.clone(),
                        precision: DisclosurePrecision::Hidden,
                        replacements: Vec::new(),
                        reason: RepairReason::InferenceLeak,
                    },
                );
            }
        }
    }

    if closure.truncated {
        notes.push("closure_truncated".to_string());
        blocked = true;
    }
    if blocked {
        for claim in claims {
            removed_claims.insert(claim.claim_id.clone());
            for atom in &claim.atoms {
                merge_repair(
                    &mut repairs,
                    SlotRepair {
                        slot: atom.slot(),
                        precision: DisclosurePrecision::Hidden,
                        replacements: Vec::new(),
                        reason: RepairReason::ConservativeBlock,
                    },
                );
            }
        }
    }

    let mut repairs: Vec<SlotRepair> = repairs.into_values().collect();
    repairs.sort_by(|left, right| left.slot.cmp(&right.slot));
    let mut removed_claims: Vec<String> = removed_claims.into_iter().collect();
    removed_claims.sort();
    notes.sort();
    notes.dedup();

    AnswerDisclosureReport {
        mode: policy.mode,
        requested_precision: config.requested_precision,
        claim_count: claims.len(),
        atom_decisions,
        joint_findings,
        leaks,
        repairs,
        removed_claims,
        blocked,
        closure_truncated: closure.truncated,
        hitting_set_exhausted,
        notes,
    }
}

fn effective_decision(
    decision: &DisclosureDecision,
) -> (DisclosureStatus, DisclosurePrecision, &[DecisionCode]) {
    match &decision.audit {
        Some(audit) => (
            audit.would_be_status,
            audit.would_be_precision,
            audit.codes.as_slice(),
        ),
        None => (
            decision.status,
            decision.effective_precision,
            decision.codes.as_slice(),
        ),
    }
}

fn merge_repair(repairs: &mut BTreeMap<DisclosureSlot, SlotRepair>, repair: SlotRepair) {
    use std::collections::btree_map::Entry;
    match repairs.entry(repair.slot.clone()) {
        Entry::Vacant(entry) => {
            entry.insert(repair);
        }
        Entry::Occupied(mut entry) => {
            let existing = entry.get_mut();
            if repair.precision == DisclosurePrecision::Hidden
                || repair.precision < existing.precision
            {
                *existing = repair;
                return;
            }
            if repair.precision == existing.precision {
                existing.replacements.extend(repair.replacements);
                existing.replacements.sort();
                existing.replacements.dedup();
                if repair.reason < existing.reason {
                    existing.reason = repair.reason;
                }
            }
        }
    }
}

fn atom_matches_fact(atom: &ClaimAtom, fact: &PrivacyFact) -> bool {
    atom.subject == fact.subject && atom.predicate == fact.predicate && atom.object == fact.object
}

fn same_fact(left: &PrivacyFact, right: &PrivacyFact) -> bool {
    left.subject == right.subject
        && left.predicate == right.predicate
        && left.object == right.object
}

fn combined_facts(
    policy: &PrivacyPolicy,
    graph: &PrivacyGraph,
    claims: &[FusedClaim],
    adjusted_precision: &BTreeMap<ClaimAtom, DisclosurePrecision>,
) -> Vec<PrivacyFact> {
    let mut facts = Vec::with_capacity(graph.facts.len());
    for fact in &graph.facts {
        let atom = claims
            .iter()
            .flat_map(|claim| claim.atoms.iter())
            .find(|atom| atom_matches_fact(atom, fact));
        match atom.and_then(|atom| adjusted_precision.get(atom)).copied() {
            Some(DisclosurePrecision::Hidden) => continue,
            Some(precision) => {
                let mut fact = fact.clone();
                fact.precision = fact.precision.min(precision);
                facts.push(fact);
            }
            None => facts.push(fact.clone()),
        }
    }
    for atom in claims.iter().flat_map(|claim| claim.atoms.iter()) {
        if facts.iter().any(|fact| atom_matches_fact(atom, fact))
            || graph.facts.iter().any(|fact| atom_matches_fact(atom, fact))
        {
            continue;
        }
        let category = policy.classify_predicate(&atom.predicate);
        facts.push(PrivacyFact {
            subject: atom.subject.clone(),
            predicate: atom.predicate.clone(),
            object: atom.object.clone(),
            category,
            precision: adjusted_precision
                .get(atom)
                .copied()
                .unwrap_or(DisclosurePrecision::Exact),
        });
    }
    facts.sort_by(|left, right| {
        left.subject
            .cmp(&right.subject)
            .then_with(|| left.predicate.cmp(&right.predicate))
            .then_with(|| left.object.cmp(&right.object))
    });
    facts.dedup_by(|left, right| same_fact(left, right));
    facts
}

struct ClosureItem {
    fact: PrivacyFact,
    provenance: BTreeSet<usize>,
    depth: usize,
    rule: Option<String>,
    max_precision: DisclosurePrecision,
    derived: bool,
}

struct ClosureResult {
    items: Vec<ClosureItem>,
    truncated: bool,
}

fn derive_closure(facts: &[PrivacyFact], policy: &PrivacyPolicy) -> ClosureResult {
    let mut items: Vec<ClosureItem> = facts
        .iter()
        .enumerate()
        .map(|(index, fact)| ClosureItem {
            fact: fact.clone(),
            provenance: [index].into_iter().collect(),
            depth: 0,
            rule: None,
            max_precision: fact.precision.min(policy.max_precision(fact.category)),
            derived: false,
        })
        .collect();
    let mut known: BTreeMap<(String, String, String), usize> = items
        .iter()
        .enumerate()
        .map(|(index, item)| {
            (
                (
                    item.fact.subject.clone(),
                    item.fact.predicate.clone(),
                    item.fact.object.clone(),
                ),
                index,
            )
        })
        .collect();
    let mut truncated = false;

    if policy.inference_rules.is_empty() {
        return ClosureResult { items, truncated };
    }

    for _depth in 0..policy.max_closure_depth.max(1) {
        let snapshot_len = items.len();
        let mut candidates: Vec<(
            PrivacyFact,
            BTreeSet<usize>,
            usize,
            String,
            DisclosurePrecision,
        )> = Vec::new();
        for rule in &policy.inference_rules {
            let rule_precision = rule
                .max_precision
                .unwrap_or_else(|| policy.max_precision(rule.category));
            for left_index in 0..snapshot_len {
                if items[left_index].fact.predicate != rule.first {
                    continue;
                }
                for right_index in 0..snapshot_len {
                    if items[right_index].fact.predicate != rule.second {
                        continue;
                    }
                    if items[left_index].fact.object != items[right_index].fact.subject {
                        continue;
                    }
                    let conclusion = PrivacyFact {
                        subject: items[left_index].fact.subject.clone(),
                        predicate: rule.conclusion.clone(),
                        object: items[right_index].fact.object.clone(),
                        category: rule.category,
                        precision: DisclosurePrecision::Exact,
                    };
                    if items.iter().any(|item| same_fact(&item.fact, &conclusion)) {
                        continue;
                    }
                    let mut provenance = items[left_index].provenance.clone();
                    provenance.extend(items[right_index].provenance.iter().copied());
                    let depth = items[left_index]
                        .depth
                        .max(items[right_index].depth)
                        .saturating_add(1);
                    let max_precision = items[left_index]
                        .max_precision
                        .min(items[right_index].max_precision)
                        .min(rule_precision);
                    candidates.push((
                        conclusion,
                        provenance,
                        depth,
                        rule.name.clone(),
                        max_precision,
                    ));
                }
            }
        }
        if candidates.is_empty() {
            break;
        }
        let mut added = 0usize;
        for (fact, provenance, depth, rule, max_precision) in candidates {
            let key = (
                fact.subject.clone(),
                fact.predicate.clone(),
                fact.object.clone(),
            );
            if let Some(index) = known.get(&key).copied() {
                let existing = &mut items[index];
                existing.provenance.extend(provenance);
                existing.max_precision = existing.max_precision.min(max_precision);
                existing.depth = existing.depth.max(depth);
                continue;
            }
            if items.len() >= policy.max_closure_atoms.max(1) {
                truncated = true;
                break;
            }
            known.insert(key, items.len());
            items.push(ClosureItem {
                fact,
                provenance,
                depth,
                rule: Some(rule),
                max_precision,
                derived: true,
            });
            added += 1;
        }
        if truncated || added == 0 {
            break;
        }
    }

    ClosureResult { items, truncated }
}

fn audit_joint_k(
    policy: &PrivacyPolicy,
    graph: &PrivacyGraph,
    claims: &[FusedClaim],
    adjusted_precision: &BTreeMap<ClaimAtom, DisclosurePrecision>,
    quasi_identifiers: &BTreeSet<String>,
    repairs: &mut BTreeMap<DisclosureSlot, SlotRepair>,
    removed_claims: &mut BTreeSet<String>,
) -> Vec<JointAnonymityFinding> {
    let k = policy.k_anonymity;
    if k <= 1 || quasi_identifiers.is_empty() {
        return Vec::new();
    }
    let facts = combined_facts(policy, graph, claims, adjusted_precision);
    let mut by_subject: BTreeMap<String, Vec<&ClaimAtom>> = BTreeMap::new();
    for claim in claims {
        for atom in &claim.atoms {
            if !quasi_identifiers.contains(&atom.predicate) {
                continue;
            }
            if adjusted_precision
                .get(atom)
                .copied()
                .unwrap_or(DisclosurePrecision::Hidden)
                == DisclosurePrecision::Hidden
            {
                continue;
            }
            by_subject
                .entry(atom.subject.clone())
                .or_default()
                .push(atom);
        }
    }

    let mut findings = Vec::new();
    for (subject, atoms) in by_subject {
        let mut predicates: Vec<String> = atoms.iter().map(|atom| atom.predicate.clone()).collect();
        predicates.sort();
        predicates.dedup();
        if predicates.is_empty() {
            continue;
        }
        let mut precision: BTreeMap<String, DisclosurePrecision> = BTreeMap::new();
        for predicate in &predicates {
            let value = atoms
                .iter()
                .filter(|atom| &atom.predicate == predicate)
                .map(|atom| {
                    adjusted_precision
                        .get(*atom)
                        .copied()
                        .unwrap_or(DisclosurePrecision::Exact)
                })
                .min()
                .unwrap_or(DisclosurePrecision::Exact);
            precision.insert(predicate.clone(), value);
        }
        let mut class_size = joint_class_size(&facts, &subject, &predicates, &precision);
        if class_size < k {
            let original = precision.clone();
            let max_rounds = predicates.len().saturating_mul(4).max(1);
            for _ in 0..max_rounds {
                if class_size >= k {
                    break;
                }
                let mut best: Option<(
                    usize,
                    String,
                    DisclosurePrecision,
                    BTreeMap<String, DisclosurePrecision>,
                )> = None;
                for predicate in &predicates {
                    let current = precision
                        .get(predicate)
                        .copied()
                        .unwrap_or(DisclosurePrecision::Hidden);
                    if current == DisclosurePrecision::Hidden {
                        continue;
                    }
                    let next = current.coarsen();
                    let mut trial = precision.clone();
                    trial.insert(predicate.clone(), next);
                    let trial_size = joint_class_size(&facts, &subject, &predicates, &trial);
                    let better = match &best {
                        None => true,
                        Some((best_size, best_predicate, best_next, _)) => {
                            trial_size > *best_size
                                || (trial_size == *best_size
                                    && (next < *best_next
                                        || (next == *best_next && predicate < best_predicate)))
                        }
                    };
                    if better {
                        best = Some((trial_size, predicate.clone(), next, trial));
                    }
                }
                let Some((next_size, _predicate, _next_precision, trial)) = best else {
                    break;
                };
                precision = trial;
                class_size = next_size;
            }
            class_size = joint_class_size(&facts, &subject, &predicates, &precision);
            if class_size < k {
                for predicate in &predicates {
                    precision.insert(predicate.clone(), DisclosurePrecision::Hidden);
                }
                class_size = 0;
            }
            for (predicate, effective) in &precision {
                let previous = original
                    .get(predicate)
                    .copied()
                    .unwrap_or(DisclosurePrecision::Exact);
                if *effective >= previous {
                    continue;
                }
                let slot = DisclosureSlot::new(&subject, predicate);
                if *effective == DisclosurePrecision::Hidden {
                    for claim in claims {
                        if claim
                            .atoms
                            .iter()
                            .any(|atom| atom.subject == subject && &atom.predicate == predicate)
                        {
                            removed_claims.insert(claim.claim_id.clone());
                        }
                    }
                    merge_repair(
                        repairs,
                        SlotRepair {
                            slot,
                            precision: DisclosurePrecision::Hidden,
                            replacements: Vec::new(),
                            reason: RepairReason::JointKAnonymity,
                        },
                    );
                } else {
                    let mut replacements = BTreeSet::new();
                    for atom in atoms.iter().filter(|atom| &atom.predicate == predicate) {
                        let category = policy.classify_predicate(&atom.predicate);
                        if let Some(value) = coarsen_value(category, *effective, &atom.object, None)
                        {
                            replacements.insert(value);
                        }
                    }
                    merge_repair(
                        repairs,
                        SlotRepair {
                            slot,
                            precision: *effective,
                            replacements: replacements.into_iter().collect(),
                            reason: RepairReason::JointKAnonymity,
                        },
                    );
                }
            }
        }
        findings.push(JointAnonymityFinding {
            subject,
            predicates,
            class_size,
            required: k,
            precision,
        });
    }
    findings.sort_by(|left, right| left.subject.cmp(&right.subject));
    findings
}

fn joint_class_size(
    facts: &[PrivacyFact],
    subject: &str,
    predicates: &[String],
    precision: &BTreeMap<String, DisclosurePrecision>,
) -> usize {
    let mut by_subject: BTreeMap<&str, Vec<&PrivacyFact>> = BTreeMap::new();
    for fact in facts {
        by_subject
            .entry(fact.subject.as_str())
            .or_default()
            .push(fact);
    }
    let target = signature(by_subject.get(subject), predicates, precision);
    by_subject
        .values()
        .filter(|subject_facts| signature(Some(subject_facts), predicates, precision) == target)
        .count()
}

fn signature(
    facts: Option<&Vec<&PrivacyFact>>,
    predicates: &[String],
    precision: &BTreeMap<String, DisclosurePrecision>,
) -> String {
    let Some(facts) = facts else {
        return String::new();
    };
    let mut parts = Vec::new();
    for predicate in predicates {
        let value = precision
            .get(predicate)
            .copied()
            .unwrap_or(DisclosurePrecision::Hidden);
        if value == DisclosurePrecision::Hidden {
            continue;
        }
        let mut values: Vec<String> = facts
            .iter()
            .filter(|fact| &fact.predicate == predicate)
            .filter_map(|fact| coarsen_value(fact.category, value, &fact.object, None))
            .collect();
        values.sort();
        values.dedup();
        parts.push(format!("{}={}", predicate, values.join(",")));
    }
    parts.join("\u{1}")
}

fn slot_weights(
    claims: &[FusedClaim],
    answer_slots: &BTreeMap<ClaimAtom, DisclosureSlot>,
) -> BTreeMap<DisclosureSlot, u32> {
    let mut weights: BTreeMap<DisclosureSlot, u32> = BTreeMap::new();
    for claim in claims {
        let mut seen = BTreeSet::new();
        for atom in &claim.atoms {
            if let Some(slot) = answer_slots.get(atom) {
                if seen.insert(slot.clone()) {
                    *weights.entry(slot.clone()).or_insert(0) += 1;
                }
            }
        }
    }
    for (atom, slot) in answer_slots {
        if !atom.predicate.is_empty() {
            let weight = weights.entry(slot.clone()).or_insert(1);
            if *weight == 0 {
                *weight = 1;
            }
        }
    }
    weights
}

fn solve_hitting_set(
    sets: &[BTreeSet<DisclosureSlot>],
    weights: &BTreeMap<DisclosureSlot, u32>,
    budget: usize,
) -> (BTreeSet<DisclosureSlot>, bool) {
    let mut normalized: Vec<BTreeSet<DisclosureSlot>> =
        sets.iter().filter(|set| !set.is_empty()).cloned().collect();
    normalized.sort_by(|left, right| {
        left.len()
            .cmp(&right.len())
            .then_with(|| left.iter().next().cmp(&right.iter().next()))
    });
    normalized.dedup();
    let mut minimal: Vec<BTreeSet<DisclosureSlot>> = Vec::new();
    for set in normalized {
        if minimal.iter().any(|existing| existing.is_subset(&set)) {
            continue;
        }
        minimal.retain(|existing| !set.is_subset(existing));
        minimal.push(set);
    }
    if minimal.is_empty() {
        return (BTreeSet::new(), false);
    }

    let mut solver = HittingSolver {
        sets: &minimal,
        weights,
        budget,
        nodes: 0,
        exhausted: false,
        best_cost: u32::MAX,
        best: BTreeSet::new(),
    };
    let mut chosen = BTreeSet::new();
    solver.search(&mut chosen, 0);
    if solver.best_cost != u32::MAX {
        return (solver.best.clone(), solver.exhausted);
    }
    let fallback = greedy_hitting_set(&minimal, weights);
    (fallback, true)
}

struct HittingSolver<'a> {
    sets: &'a [BTreeSet<DisclosureSlot>],
    weights: &'a BTreeMap<DisclosureSlot, u32>,
    budget: usize,
    nodes: usize,
    exhausted: bool,
    best_cost: u32,
    best: BTreeSet<DisclosureSlot>,
}

impl HittingSolver<'_> {
    fn search(&mut self, chosen: &mut BTreeSet<DisclosureSlot>, cost: u32) {
        if self.nodes >= self.budget {
            self.exhausted = true;
            return;
        }
        self.nodes += 1;
        let uncovered: Vec<&BTreeSet<DisclosureSlot>> = self
            .sets
            .iter()
            .filter(|set| set.is_disjoint(chosen))
            .collect();
        if uncovered.is_empty() {
            if cost < self.best_cost || (cost == self.best_cost && *chosen < self.best) {
                self.best_cost = cost;
                self.best = chosen.clone();
            }
            return;
        }
        if cost >= self.best_cost {
            return;
        }
        let pivot = uncovered
            .iter()
            .min_by(|left, right| {
                left.len()
                    .cmp(&right.len())
                    .then_with(|| left.iter().next().cmp(&right.iter().next()))
            })
            .copied()
            .expect("至少一个未覆盖集合");
        let mut candidates: Vec<&DisclosureSlot> = pivot.iter().collect();
        candidates.sort_by(|left, right| {
            let left_weight = self.weights.get(*left).copied().unwrap_or(1);
            let right_weight = self.weights.get(*right).copied().unwrap_or(1);
            left_weight.cmp(&right_weight).then_with(|| left.cmp(right))
        });
        for candidate in candidates {
            if self.exhausted || self.nodes >= self.budget {
                self.exhausted = true;
                return;
            }
            let weight = self.weights.get(candidate).copied().unwrap_or(1);
            if cost.saturating_add(weight) >= self.best_cost {
                continue;
            }
            chosen.insert(candidate.clone());
            self.search(chosen, cost.saturating_add(weight));
            chosen.remove(candidate);
        }
    }
}

fn greedy_hitting_set(
    sets: &[BTreeSet<DisclosureSlot>],
    weights: &BTreeMap<DisclosureSlot, u32>,
) -> BTreeSet<DisclosureSlot> {
    let mut uncovered: Vec<BTreeSet<DisclosureSlot>> = sets.to_vec();
    let mut chosen = BTreeSet::new();
    while !uncovered.is_empty() {
        let mut counts: BTreeMap<DisclosureSlot, usize> = BTreeMap::new();
        for set in &uncovered {
            for slot in set {
                *counts.entry(slot.clone()).or_insert(0) += 1;
            }
        }
        let best = counts
            .into_iter()
            .min_by(|left, right| {
                let left_weight = weights.get(&left.0).copied().unwrap_or(1) as usize;
                let right_weight = weights.get(&right.0).copied().unwrap_or(1) as usize;
                (right.1 * left_weight)
                    .cmp(&(left.1 * right_weight))
                    .then_with(|| left_weight.cmp(&right_weight))
                    .then_with(|| left.0.cmp(&right.0))
            })
            .map(|(slot, _)| slot);
        let Some(slot) = best else {
            break;
        };
        chosen.insert(slot.clone());
        uncovered.retain(|set| !set.contains(&slot));
    }
    chosen
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::answer_plan::{AnswerPlan, ClaimEvidence, PlanClaim};
    use crate::privacy_flow::{InferenceRule, PolicyMode, PredicateRule, PrivacyCategory};

    fn enforced_policy(k: usize) -> PrivacyPolicy {
        PrivacyPolicy {
            mode: PolicyMode::Enforce,
            k_anonymity: k,
            ..PrivacyPolicy::default()
        }
    }

    #[test]
    fn associative_only_candidates_need_deductive_confirmation() {
        let associative = vec![ClaimCandidate::new(
            "c1",
            "张三可能在北京",
            vec![ClaimAtom::new("张三", "居住", "北京")],
            EvidenceChannel::Associative,
        )];
        let report = fuse_candidates(&associative, &[], true);
        assert!(report.accepted.is_empty());
        assert_eq!(report.rejected.len(), 1);
        assert_eq!(report.rejected[0].reason, "needs_deductive_warrant");

        let relaxed = fuse_candidates(&associative, &[], false);
        assert_eq!(relaxed.accepted.len(), 1);
        assert_eq!(relaxed.accepted[0].warrant.level, WarrantLevel::Associative);
    }

    #[test]
    fn both_channels_corroborate_and_prefer_deductive_text() {
        let associative = vec![ClaimCandidate::new(
            "c1",
            "联想句",
            vec![ClaimAtom::new("张三", "职业", "工程师")],
            EvidenceChannel::Associative,
        )
        .with_source("holo")];
        let deductive = vec![ClaimCandidate::new(
            "c1",
            "演绎句",
            vec![ClaimAtom::new("张三", "职业", "工程师")],
            EvidenceChannel::Deductive,
        )
        .with_source("proof")];
        let report = fuse_candidates(&associative, &deductive, true);
        assert_eq!(report.accepted.len(), 1);
        let fused = &report.accepted[0];
        assert_eq!(fused.text, "演绎句");
        assert_eq!(fused.warrant.level, WarrantLevel::Corroborated);
        assert_eq!(fused.warrant.channels.len(), 2);
        assert!(fused.warrant.sources.contains("holo"));
        assert!(fused.warrant.sources.contains("proof"));
        assert!(report
            .notes
            .iter()
            .any(|note| note.contains("channel_text_divergence")));
    }

    #[test]
    fn candidates_without_evidence_are_rejected() {
        let candidate =
            ClaimCandidate::new("empty", "没有依据", Vec::new(), EvidenceChannel::Deductive);
        let report = fuse_candidates(&[], &[candidate], true);
        assert!(report.accepted.is_empty());
        assert_eq!(report.rejected[0].reason, "missing_evidence");
    }

    #[test]
    fn plan_claims_become_deductive_candidates_with_slots() {
        let plan = AnswerPlan::new(
            "张三住在哪里",
            "张三",
            7,
            vec![
                PlanClaim::new("slot-address", "地址", "张三住在北京市", false)
                    .with_evidence(vec![ClaimEvidence::new("张三", "地址", "北京市", false)]),
            ],
            0,
            0,
        );
        let candidates = candidates_from_plan(&plan, EvidenceChannel::Deductive);
        assert_eq!(candidates[0].claim_id, "slot-address");
        let report = fuse_candidates(&[], &candidates, true);
        let fused = &report.accepted[0];
        assert_eq!(fused.warrant.level, WarrantLevel::Deductive);
        assert_eq!(fused.atoms, vec![ClaimAtom::new("张三", "地址", "北京市")]);
        assert!(fused.warrant.sources.contains("answer_plan"));
    }

    #[test]
    fn joint_k_detects_unique_marginal_combinations() {
        let mut policy = enforced_policy(2);
        policy.inference_rules.clear();
        policy.consent_required.clear();
        let mut graph = PrivacyGraph::new();
        for (subject, city, age) in [
            ("张三", "北京市", "20岁"),
            ("李四", "北京市", "30岁"),
            ("王五", "上海市", "20岁"),
            ("赵六", "上海市", "30岁"),
        ] {
            graph.add_fact(&policy, subject, "地址", city);
            graph.add_fact(&policy, subject, "年龄", age);
        }
        let claims = vec![
            FusedClaim {
                claim_id: "a".to_string(),
                text: "张三：北京市，20岁".to_string(),
                atoms: vec![
                    ClaimAtom::new("张三", "地址", "北京市"),
                    ClaimAtom::new("张三", "年龄", "20岁"),
                ],
                warrant: ClaimWarrant {
                    level: WarrantLevel::Deductive,
                    channels: [EvidenceChannel::Deductive].into_iter().collect(),
                    sources: BTreeSet::new(),
                },
            },
            FusedClaim {
                claim_id: "b".to_string(),
                text: "李四：北京市，30岁".to_string(),
                atoms: vec![
                    ClaimAtom::new("李四", "地址", "北京市"),
                    ClaimAtom::new("李四", "年龄", "30岁"),
                ],
                warrant: ClaimWarrant {
                    level: WarrantLevel::Deductive,
                    channels: [EvidenceChannel::Deductive].into_iter().collect(),
                    sources: BTreeSet::new(),
                },
            },
        ];
        let config = AnswerDisclosureConfig::default()
            .with_requested_precision(DisclosurePrecision::Coarse)
            .with_quasi_identifiers(["地址", "年龄"]);
        let report = decode_answer(&policy, &graph, &PrivacyLedger::new(), &claims, &config);
        assert!(!report.joint_findings.is_empty());
        let finding = &report.joint_findings[0];
        assert_eq!(finding.required, 2);
        assert_eq!(finding.class_size, 2);
        assert!(report
            .repairs
            .iter()
            .any(|repair| repair.reason == RepairReason::JointKAnonymity));
        assert!(!report.blocked);
    }

    #[test]
    fn answer_union_closure_finds_leak_invisible_to_single_atom_checks() {
        let mut policy = enforced_policy(1);
        policy.inference_rules.push(
            InferenceRule::new(
                "学校位置推断城区",
                "就读学校",
                "位于",
                "所在城区",
                PrivacyCategory::Location,
            )
            .with_max_precision(DisclosurePrecision::Coarse),
        );
        let mut graph = PrivacyGraph::new();
        graph.add_fact(&policy, "阳光小学", "位于", "北京市海淀区");
        let claims = vec![FusedClaim {
            claim_id: "school".to_string(),
            text: "张三就读于阳光小学".to_string(),
            atoms: vec![ClaimAtom::new("张三", "就读学校", "阳光小学")],
            warrant: ClaimWarrant {
                level: WarrantLevel::Deductive,
                channels: [EvidenceChannel::Deductive].into_iter().collect(),
                sources: BTreeSet::new(),
            },
        }];
        let config = AnswerDisclosureConfig::default();
        let report = decode_answer(&policy, &graph, &PrivacyLedger::new(), &claims, &config);
        assert_eq!(report.leaks.len(), 1);
        assert_eq!(report.leaks[0].predicate, "所在城区");
        assert!(report
            .repairs
            .iter()
            .any(|repair| repair.slot == DisclosureSlot::new("张三", "就读学校")));
        assert!(report.removed_claims.contains(&"school".to_string()));

        let coarse =
            AnswerDisclosureConfig::default().with_requested_precision(DisclosurePrecision::Coarse);
        let safe = decode_answer(&policy, &graph, &PrivacyLedger::new(), &claims, &coarse);
        assert!(safe.leaks.is_empty());
    }

    #[test]
    fn hidden_atom_is_not_reintroduced_by_graph_seed() {
        let mut policy = enforced_policy(1);
        policy.consent_required.clear();
        policy.predicate_rules.insert(
            0,
            PredicateRule {
                contains: "就读".to_string(),
                category: PrivacyCategory::Location,
                max_precision: Some(DisclosurePrecision::Hidden),
            },
        );
        policy
            .predicate_rules
            .insert(1, PredicateRule::new("位于", PrivacyCategory::Other));
        policy.inference_rules.push(
            InferenceRule::new(
                "学校所在城市",
                "就读",
                "位于",
                "所在城市",
                PrivacyCategory::Location,
            )
            .with_max_precision(DisclosurePrecision::TypeOnly),
        );
        let mut graph = PrivacyGraph::new();
        graph.add_fact(&policy, "阳光小学", "就读", "海淀区");
        graph.add_fact(&policy, "海淀区", "位于", "北京市");
        let claims = vec![
            FusedClaim::deductive(
                "school",
                "阳光小学位于海淀区",
                vec![ClaimAtom::new("阳光小学", "就读", "海淀区")],
                "test",
            ),
            FusedClaim::deductive(
                "district",
                "海淀区位于北京市",
                vec![ClaimAtom::new("海淀区", "位于", "北京市")],
                "test",
            ),
        ];
        let report = decode_answer(
            &policy,
            &graph,
            &PrivacyLedger::new(),
            &claims,
            &AnswerDisclosureConfig::default(),
        );
        let school = report
            .atom_decisions
            .iter()
            .find(|decision| decision.claim_id == "school")
            .expect("school decision");
        assert_eq!(
            school.decision.effective_precision,
            DisclosurePrecision::Hidden
        );
        assert!(report.leaks.is_empty());
        assert!(report.removed_claims.contains(&"school".to_string()));
        assert!(!report.removed_claims.contains(&"district".to_string()));
    }

    #[test]
    fn minimal_hitting_set_prefers_shared_premise() {
        let mut policy = enforced_policy(1);
        policy.inference_rules.push(
            InferenceRule::new(
                "学校位置",
                "就读学校",
                "位于",
                "所在城区",
                PrivacyCategory::Location,
            )
            .with_max_precision(DisclosurePrecision::TypeOnly),
        );
        policy.inference_rules.push(
            InferenceRule::new(
                "学校健康",
                "就读学校",
                "有",
                "健康记录",
                PrivacyCategory::Health,
            )
            .with_max_precision(DisclosurePrecision::TypeOnly),
        );
        let graph = PrivacyGraph::new();
        let claims = vec![
            FusedClaim {
                claim_id: "school".to_string(),
                text: "张三就读于阳光小学".to_string(),
                atoms: vec![ClaimAtom::new("张三", "就读学校", "阳光小学")],
                warrant: WarrantLevel::Deductive.into_warrant(),
            },
            FusedClaim {
                claim_id: "location".to_string(),
                text: "阳光小学位于北京市".to_string(),
                atoms: vec![ClaimAtom::new("阳光小学", "位于", "北京市")],
                warrant: WarrantLevel::Deductive.into_warrant(),
            },
            FusedClaim {
                claim_id: "health".to_string(),
                text: "阳光小学有健康记录".to_string(),
                atoms: vec![ClaimAtom::new("阳光小学", "有", "健康记录")],
                warrant: WarrantLevel::Deductive.into_warrant(),
            },
        ];
        let config = AnswerDisclosureConfig::default();
        let report = decode_answer(&policy, &graph, &PrivacyLedger::new(), &claims, &config);
        assert_eq!(report.leaks.len(), 2);
        let hidden: BTreeSet<DisclosureSlot> = report
            .repairs
            .iter()
            .filter(|repair| repair.precision == DisclosurePrecision::Hidden)
            .map(|repair| repair.slot.clone())
            .collect();
        assert!(hidden.contains(&DisclosureSlot::new("张三", "就读学校")));
        assert!(!hidden.contains(&DisclosureSlot::new("阳光小学", "位于")));
        assert!(!hidden.contains(&DisclosureSlot::new("阳光小学", "有")));
    }

    #[test]
    fn truncated_closure_blocks_instead_of_pretending_safe() {
        let mut policy = enforced_policy(1);
        policy.max_closure_atoms = 2;
        policy.inference_rules.push(
            InferenceRule::new(
                "学校位置",
                "就读学校",
                "位于",
                "所在城区",
                PrivacyCategory::Location,
            )
            .with_max_precision(DisclosurePrecision::Coarse),
        );
        let mut graph = PrivacyGraph::new();
        graph.add_fact(&policy, "阳光小学", "位于", "北京市海淀区");
        graph.add_fact(&policy, "张三", "姓名", "张三");
        let claims = vec![FusedClaim {
            claim_id: "school".to_string(),
            text: "张三就读于阳光小学".to_string(),
            atoms: vec![ClaimAtom::new("张三", "就读学校", "阳光小学")],
            warrant: WarrantLevel::Deductive.into_warrant(),
        }];
        let report = decode_answer(
            &policy,
            &graph,
            &PrivacyLedger::new(),
            &claims,
            &AnswerDisclosureConfig::default(),
        );
        assert!(report.closure_truncated);
        assert!(report.blocked);
        assert!(!report.is_safe());
        assert!(report
            .repairs
            .iter()
            .any(|repair| repair.reason == RepairReason::ConservativeBlock));
    }

    #[test]
    fn audit_mode_reports_without_mutating_overlay() {
        let mut policy = enforced_policy(1);
        policy.mode = PolicyMode::Audit;
        policy.inference_rules.push(
            InferenceRule::new(
                "学校位置",
                "就读学校",
                "位于",
                "所在城区",
                PrivacyCategory::Location,
            )
            .with_max_precision(DisclosurePrecision::Coarse),
        );
        let mut graph = PrivacyGraph::new();
        graph.add_fact(&policy, "阳光小学", "位于", "北京市海淀区");
        let claims = vec![FusedClaim {
            claim_id: "school".to_string(),
            text: "张三就读于阳光小学".to_string(),
            atoms: vec![ClaimAtom::new("张三", "就读学校", "阳光小学")],
            warrant: WarrantLevel::Deductive.into_warrant(),
        }];
        let report = decode_answer(
            &policy,
            &graph,
            &PrivacyLedger::new(),
            &claims,
            &AnswerDisclosureConfig::default(),
        );
        assert_eq!(report.mode, PolicyMode::Audit);
        assert!(!report.leaks.is_empty());
        let mut overlay = GenerationOverlay::default();
        report.apply_to_overlay(&mut overlay);
        assert!(overlay.is_empty(), "影子模式不改写生成叠加层");
    }

    #[test]
    fn render_keeps_objects_out_of_the_summary() {
        let report = AnswerDisclosureReport {
            mode: PolicyMode::Enforce,
            requested_precision: DisclosurePrecision::Exact,
            claim_count: 1,
            atom_decisions: Vec::new(),
            joint_findings: Vec::new(),
            leaks: vec![AnswerLeak {
                rule: "r".to_string(),
                subject: "张三".to_string(),
                predicate: "所在城区".to_string(),
                object: "北京市海淀区".to_string(),
                category: PrivacyCategory::Location,
                max_precision: DisclosurePrecision::Coarse,
                premises: Vec::new(),
            }],
            repairs: Vec::new(),
            removed_claims: Vec::new(),
            blocked: false,
            closure_truncated: false,
            hitting_set_exhausted: false,
            notes: Vec::new(),
        };
        let text = report.render_zh();
        assert!(text.contains("组合泄漏 1 条"));
        assert!(!text.contains("北京市海淀区"));
    }

    impl WarrantLevel {
        fn into_warrant(self) -> ClaimWarrant {
            ClaimWarrant {
                level: self,
                channels: [EvidenceChannel::Deductive].into_iter().collect(),
                sources: BTreeSet::new(),
            }
        }
    }
}

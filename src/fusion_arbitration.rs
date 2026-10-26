//! FAR：融合裁决重排器（Fusion-Arbitrated Reranking）。
//!
//! [`crate::knowledge_fusion`] 把多源断言裁决到默认图，但回答阶段读到的只是
//! 「物化后的结论」。本模块补上回答期的另一半：把每个候选取值看成一条论辩
//! 论证，按可审计的序数强度重排，并给出可生成的说理模式。
//!
//! 核心约束：
//!
//! - **强度是序数对 `(信任层级, 独立来源数)`**，不做加权平均、不训练权重；
//! - **独立来源按来源成分计数**：同一份内容重复导入、命名空间镜像、同一命名
//!   空间的多条断言都只算一个来源，避免「假共识」；
//! - **默认图不是来源**：`AppliedWinner` 只说明该值被物化到默认图，可信度继承
//!   其来源命名空间，但不额外投票；
//! - **显式裁决优先于常规来源**，**会话更正优先于图谱**，且只作用于对应槽；
//! - **同强度互相攻击即未决**：不偷偷选一个，而是返回可展示的分歧；
//! - **确定性**：所有遍历按稳定键排序，报告与重排结果可逐字节断言。
//!
//! 本模块是纯函数层：不接触存储、不修改账本，调用方决定何时启用。

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::knowledge_fusion::{
    sanitize_namespace_id, Adjudication, AppliedWinner, FusionLedger, NamespaceKind,
    NamespaceRecord, NamespaceStatus, ObjectKind, ScopedAssertion, TrustLevel,
};
use crate::provenance::{ConflictCertificate, ConflictPoint, FactRef, ProvenanceLedger, SourceId};
use crate::slot_ledger::{SlotEventKind, SlotKey, SlotLedger, SlotPolicy};

// ---------------------------------------------------------------------------
// 来源目录：把命名空间折叠成「独立来源成分」
// ---------------------------------------------------------------------------

/// 一个来源命名空间的裁决画像。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceProfile {
    pub namespace: String,
    pub label: String,
    pub kind: NamespaceKind,
    pub trust: TrustLevel,
    pub status: NamespaceStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin: Option<SourceId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub component: Option<String>,
}

impl SourceProfile {
    pub fn from_record(record: &NamespaceRecord) -> Self {
        Self {
            namespace: record.id.clone(),
            label: record.label.clone(),
            kind: record.kind,
            trust: record.trust,
            status: record.status,
            origin: None,
            component: None,
        }
    }

    pub fn with_origin(mut self, origin: SourceId) -> Self {
        self.origin = Some(origin);
        self
    }

    pub fn with_component(mut self, component: &str) -> Self {
        self.component = Some(component.to_string());
        self
    }

    pub fn is_active(&self) -> bool {
        self.status == NamespaceStatus::Active
    }
}

/// 来源目录：命名空间、来源身份与镜像关系。
///
/// 成分键的优先级：显式成分 > 已登记 `SourceId`（类别 + 定位符 + 内容摘要）
/// > 命名空间 id。镜像关系按等价类合并，因此环也不会让结果依赖遍历顺序。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SourceCatalog {
    profiles: BTreeMap<String, SourceProfile>,
    mirrors: BTreeMap<String, String>,
}

impl SourceCatalog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_records(records: &[NamespaceRecord]) -> Self {
        let mut catalog = Self::default();
        for record in records {
            catalog.insert(SourceProfile::from_record(record));
        }
        catalog
    }

    pub fn insert(&mut self, profile: SourceProfile) -> &mut Self {
        let namespace = sanitize_namespace_id(&profile.namespace);
        let mut profile = profile;
        profile.namespace = namespace.clone();
        self.profiles.insert(namespace, profile);
        self
    }

    pub fn with_profile(mut self, profile: SourceProfile) -> Self {
        self.insert(profile);
        self
    }

    pub fn declare_mirror(&mut self, namespace: &str, mirror_of: &str) -> &mut Self {
        let namespace = sanitize_namespace_id(namespace);
        let mirror_of = sanitize_namespace_id(mirror_of);
        if namespace != mirror_of {
            self.mirrors.insert(namespace, mirror_of);
        }
        self
    }

    pub fn with_mirror(mut self, namespace: &str, mirror_of: &str) -> Self {
        self.declare_mirror(namespace, mirror_of);
        self
    }

    pub fn profile(&self, namespace: &str) -> Option<&SourceProfile> {
        self.profiles.get(&sanitize_namespace_id(namespace))
    }

    pub fn profiles(&self) -> impl Iterator<Item = &SourceProfile> {
        self.profiles.values()
    }

    /// 每个命名空间所属的独立来源成分。
    pub fn component_map(&self) -> BTreeMap<String, String> {
        let mut ids: BTreeSet<String> = self.profiles.keys().cloned().collect();
        for (left, right) in &self.mirrors {
            ids.insert(left.clone());
            ids.insert(right.clone());
        }
        let mut union = UnionFind::new(&ids);
        for (left, right) in &self.mirrors {
            union.union(left, right);
        }
        let mut by_component: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for profile in self.profiles.values() {
            if let Some(component) = profile.component.as_deref() {
                by_component.entry(component).or_default().push(profile.namespace.as_str());
            }
        }
        for members in by_component.values() {
            for member in members.iter().skip(1) {
                union.union(members[0], member);
            }
        }
        let mut by_origin: BTreeMap<String, Vec<&str>> = BTreeMap::new();
        for profile in self.profiles.values() {
            if let Some(origin) = profile.origin.as_ref().filter(|origin| origin.is_registered()) {
                by_origin
                    .entry(origin_component(origin))
                    .or_default()
                    .push(profile.namespace.as_str());
            }
        }
        for members in by_origin.values() {
            for member in members.iter().skip(1) {
                union.union(members[0], member);
            }
        }
        let mut classes: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for id in &ids {
            classes.entry(union.find(id)).or_default().push(id.clone());
        }
        let mut out = BTreeMap::new();
        for mut members in classes.into_values() {
            members.sort();
            let key = self.class_component(&members);
            for member in members {
                out.insert(member, key.clone());
            }
        }
        out
    }

    pub fn component_of(&self, namespace: &str) -> String {
        self.component_map()
            .get(&sanitize_namespace_id(namespace))
            .cloned()
            .unwrap_or_else(|| format!("namespace:{}", sanitize_namespace_id(namespace)))
    }

    fn class_component(&self, members: &[String]) -> String {
        let mut explicit: Option<&str> = None;
        let mut origin: Option<String> = None;
        for member in members {
            let Some(profile) = self.profiles.get(member) else {
                continue;
            };
            if let Some(component) = profile.component.as_deref() {
                explicit = Some(match explicit {
                    Some(current) if current <= component => current,
                    _ => component,
                });
            }
            if let Some(source) = profile.origin.as_ref().filter(|source| source.is_registered()) {
                let key = origin_component(source);
                origin = Some(match origin {
                    Some(current) if current <= key => current,
                    _ => key,
                });
            }
        }
        if let Some(component) = explicit {
            return format!("component:{component}");
        }
        if let Some(origin) = origin {
            return origin;
        }
        format!("namespace:{}", members.first().map(String::as_str).unwrap_or("source"))
    }
}

fn origin_component(origin: &SourceId) -> String {
    format!(
        "origin:{:?}:{}:{}",
        origin.kind, origin.locator, origin.digest
    )
}

#[derive(Debug, Clone)]
struct UnionFind {
    parent: BTreeMap<String, String>,
}

impl UnionFind {
    fn new(ids: &BTreeSet<String>) -> Self {
        Self {
            parent: ids.iter().map(|id| (id.clone(), id.clone())).collect(),
        }
    }

    fn find(&self, id: &str) -> String {
        let mut current = id;
        let mut guard = 0usize;
        while let Some(parent) = self.parent.get(current) {
            if parent == current || guard > self.parent.len() {
                break;
            }
            current = parent;
            guard += 1;
        }
        current.to_string()
    }

    fn union(&mut self, left: &str, right: &str) {
        let left_root = self.find(left);
        let right_root = self.find(right);
        if left_root == right_root {
            return;
        }
        let (keep, drop) = if left_root <= right_root {
            (left_root, right_root)
        } else {
            (right_root, left_root)
        };
        self.parent.insert(drop, keep);
    }
}

// ---------------------------------------------------------------------------
// 输入：断言 / 裁决 / 物化视图 / 会话覆盖 / 冲突异议
// ---------------------------------------------------------------------------

/// 参与裁决的一条断言。`origins` 非空时优先按来源身份折叠，否则回退到命名空间。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArbitrationAssertion {
    pub namespace: String,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub object_kind: ObjectKind,
    pub origins: Vec<SourceId>,
    pub source: String,
}

impl ArbitrationAssertion {
    pub fn new(
        namespace: &str,
        subject: &str,
        predicate: &str,
        object: &str,
        object_kind: ObjectKind,
    ) -> Self {
        Self {
            namespace: sanitize_namespace_id(namespace),
            subject: subject.trim().to_string(),
            predicate: predicate.trim().to_string(),
            object: object.trim().to_string(),
            object_kind,
            origins: Vec::new(),
            source: String::new(),
        }
    }

    pub fn with_origin(mut self, origin: SourceId) -> Self {
        self.origins.push(origin);
        self
    }

    pub fn with_origins<I>(mut self, origins: I) -> Self
    where
        I: IntoIterator<Item = SourceId>,
    {
        self.origins.extend(origins);
        self
    }

    pub fn with_source(mut self, source: &str) -> Self {
        self.source = source.trim().to_string();
        self
    }

    pub fn slot_key(&self) -> SlotKey {
        SlotKey::new(&self.subject, &self.predicate)
    }

    fn identity(&self) -> (String, String, String, String, ObjectKind) {
        (
            self.namespace.clone(),
            self.subject.clone(),
            self.predicate.clone(),
            self.object.clone(),
            self.object_kind,
        )
    }
}

impl From<&ScopedAssertion> for ArbitrationAssertion {
    fn from(assertion: &ScopedAssertion) -> Self {
        Self {
            namespace: assertion.namespace.clone(),
            subject: assertion.subject.clone(),
            predicate: assertion.predicate.clone(),
            object: assertion.object.clone(),
            object_kind: assertion.object_kind,
            origins: Vec::new(),
            source: assertion.source.clone(),
        }
    }
}

/// 冲突证书带来的异议：不增加支持票，只保证分歧在回答期可见。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DissentEvidence {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub reason: String,
    pub sources: Vec<String>,
}

impl DissentEvidence {
    pub fn new(subject: &str, predicate: &str, object: &str, reason: &str) -> Self {
        Self {
            subject: subject.trim().to_string(),
            predicate: predicate.trim().to_string(),
            object: object.trim().to_string(),
            reason: reason.trim().to_string(),
            sources: Vec::new(),
        }
    }

    pub fn with_sources<I, S>(mut self, sources: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        for source in sources {
            let source = source.into();
            if !source.trim().is_empty() && !self.sources.contains(&source) {
                self.sources.push(source);
            }
        }
        self
    }

    pub fn slot_key(&self) -> SlotKey {
        SlotKey::new(&self.subject, &self.predicate)
    }
}

/// 会话覆盖：只收录「更正」事件，直陈与传闻不替代图谱。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionOverlay {
    slots: BTreeMap<SlotKey, SessionSlot>,
}

/// 一个槽的会话更正状态。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionSlot {
    pub accepted: Vec<String>,
    pub suppressed: Vec<String>,
    pub source: String,
    pub step: u64,
}

impl SessionOverlay {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_ledger(ledger: &SlotLedger) -> Self {
        let mut overlay = Self::default();
        for key in ledger.keys() {
            if !ledger.is_functional(&key.predicate) {
                continue;
            }
            let history = ledger.history(&key.subject, &key.predicate);
            let Some(correction_index) = history
                .iter()
                .rposition(|event| event.kind == SlotEventKind::Correct)
            else {
                continue;
            };
            let reset_after = history[correction_index + 1..].iter().any(|event| {
                matches!(event.kind, SlotEventKind::Retract | SlotEventKind::Restore)
            });
            if reset_after {
                continue;
            }
            let correction = &history[correction_index];
            let current: BTreeSet<String> = ledger
                .accepted(&key.subject, &key.predicate)
                .into_iter()
                .collect();
            let accepted: Vec<String> = correction
                .after
                .iter()
                .filter(|value| current.contains(*value))
                .cloned()
                .collect();
            if accepted.is_empty() {
                continue;
            }
            let mut suppressed: Vec<String> = correction
                .before
                .iter()
                .chain(ledger.suppressed(&key.subject, &key.predicate).iter())
                .filter(|value| !accepted.contains(*value))
                .cloned()
                .collect();
            suppressed.sort();
            suppressed.dedup();
            overlay.slots.insert(
                key.clone(),
                SessionSlot {
                    accepted,
                    suppressed,
                    source: correction.source.clone(),
                    step: correction.step,
                },
            );
        }
        overlay
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn get(&self, subject: &str, predicate: &str) -> Option<&SessionSlot> {
        self.slots.get(&SlotKey::new(subject, predicate))
    }

    pub fn slots(&self) -> impl Iterator<Item = (&SlotKey, &SessionSlot)> {
        self.slots.iter()
    }

    pub fn record_correction(
        &mut self,
        subject: &str,
        predicate: &str,
        accepted: &[&str],
        suppressed: &[&str],
        source: &str,
    ) {
        let key = SlotKey::new(subject, predicate);
        self.slots.insert(
            key,
            SessionSlot {
                accepted: accepted.iter().map(|value| value.to_string()).collect(),
                suppressed: suppressed.iter().map(|value| value.to_string()).collect(),
                source: source.to_string(),
                step: 0,
            },
        );
    }
}

/// 裁决输入：来源目录 + 断言 + 裁决 + 物化赢家 + 会话覆盖 + 异议。
#[derive(Debug, Clone, Default)]
pub struct ArbitrationInput {
    pub catalog: SourceCatalog,
    pub assertions: Vec<ArbitrationAssertion>,
    pub adjudications: Vec<Adjudication>,
    pub applied: Vec<AppliedWinner>,
    pub session: SessionOverlay,
    pub dissent: Vec<DissentEvidence>,
}

impl ArbitrationInput {
    pub fn from_ledger(ledger: &FusionLedger) -> Self {
        Self {
            catalog: SourceCatalog::from_records(&ledger.namespaces),
            assertions: ledger
                .assertions
                .iter()
                .map(ArbitrationAssertion::from)
                .collect(),
            adjudications: ledger.adjudications.clone(),
            applied: ledger.applied.clone(),
            session: SessionOverlay::new(),
            dissent: Vec::new(),
        }
    }

    pub fn with_session(mut self, overlay: SessionOverlay) -> Self {
        self.session = overlay;
        self
    }

    pub fn with_session_ledger(mut self, ledger: &SlotLedger) -> Self {
        self.session = SessionOverlay::from_ledger(ledger);
        self
    }

    pub fn with_provenance(mut self, ledger: &ProvenanceLedger) -> Self {
        for assertion in &mut self.assertions {
            let fact = FactRef::new(&assertion.subject, &assertion.predicate, &assertion.object);
            let origins: Vec<SourceId> = ledger
                .sources(&fact)
                .into_iter()
                .filter(SourceId::is_registered)
                .collect();
            if !origins.is_empty() {
                assertion.origins = origins;
            }
        }
        self
    }

    pub fn with_dissent(mut self, dissent: DissentEvidence) -> Self {
        self.dissent.push(dissent);
        self
    }

    pub fn with_conflict_certificates(mut self, certificates: &[ConflictCertificate]) -> Self {
        for certificate in certificates.iter().filter(|certificate| !certificate.latent) {
            let ConflictPoint::Functional {
                subject,
                predicate,
                left_value,
                right_value,
            } = &certificate.point
            else {
                continue;
            };
            let reason = format!("冲突证书 {}：{}", certificate.id, certificate.point.render_zh());
            for (value, support) in [
                (left_value, &certificate.left),
                (right_value, &certificate.right),
            ] {
                let sources: Vec<String> = support
                    .all_sources()
                    .into_iter()
                    .filter(SourceId::is_registered)
                    .map(|source| source.render())
                    .collect();
                self.dissent.push(
                    DissentEvidence::new(subject, predicate, value, &reason).with_sources(sources),
                );
            }
        }
        self
    }

    pub fn with_source(mut self, namespace: &str, origin: SourceId) -> Self {
        let namespace = sanitize_namespace_id(namespace);
        if let Some(profile) = self.catalog.profiles.get_mut(&namespace) {
            profile.origin = Some(origin);
        } else {
            self.catalog.insert(SourceProfile {
                namespace: namespace.clone(),
                label: namespace,
                kind: NamespaceKind::Import,
                trust: TrustLevel::Imported,
                status: NamespaceStatus::Active,
                origin: Some(origin),
                component: None,
            });
        }
        self
    }

    pub fn with_mirror(mut self, namespace: &str, mirror_of: &str) -> Self {
        self.catalog.declare_mirror(namespace, mirror_of);
        self
    }
}

// ---------------------------------------------------------------------------
// 裁决结果
// ---------------------------------------------------------------------------

/// 回答期的说理模式。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionMode {
    /// 按本会话的显式更正
    SessionCorrection,
    /// 按显式裁决
    Adjudicated,
    /// 多独立来源一致
    Consensus,
    /// 单一独立来源
    SingleSource,
    /// 严格更强的信任/来源数胜出
    TrustedWinner,
    /// 同强度争议，未决
    Contested,
    /// 仅有默认图物化记录
    KnowledgeBaseOnly,
    /// 没有可用证据
    NoEvidence,
}

impl DecisionMode {
    pub fn label_zh(self) -> &'static str {
        match self {
            DecisionMode::SessionCorrection => "按会话更正",
            DecisionMode::Adjudicated => "按显式裁决",
            DecisionMode::Consensus => "多源共识",
            DecisionMode::SingleSource => "单一来源",
            DecisionMode::TrustedWinner => "可信胜出",
            DecisionMode::Contested => "争议未决",
            DecisionMode::KnowledgeBaseOnly => "仅知识库记录",
            DecisionMode::NoEvidence => "无证据",
        }
    }
}

/// 一条候选取值的论辩论证。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValueArgument {
    pub object: String,
    pub object_kind: ObjectKind,
    pub trust: Option<TrustLevel>,
    pub independent_sources: usize,
    pub components: Vec<String>,
    pub namespaces: Vec<String>,
    pub sources: Vec<String>,
    pub materialized: bool,
    pub adjudicated_by: Option<String>,
    pub session_corrected: bool,
    pub session_suppressed: bool,
    pub dissent: Vec<String>,
}

impl ValueArgument {
    /// 序数强度：信任层级优先，独立来源数次之。
    pub fn strength(&self) -> (u8, usize) {
        (self.trust_rank(), self.independent_sources)
    }

    fn trust_rank(&self) -> u8 {
        self.trust.map(|trust| trust.rank() + 1).unwrap_or(0)
    }

    pub fn source_label(&self) -> String {
        if self.sources.is_empty() {
            "来源未登记".to_string()
        } else {
            self.sources.join("、")
        }
    }

    pub fn corroborated(&self) -> bool {
        self.independent_sources >= 2
    }
}

/// 一个槽的裁决。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlotDecision {
    pub subject: String,
    pub predicate: String,
    pub functional: bool,
    pub mode: DecisionMode,
    /// 功能槽的唯一赢家；非功能槽为全部相容取值；争议时为空。
    pub winners: Vec<ValueArgument>,
    /// 同强度、未决的取值。
    pub contested: Vec<ValueArgument>,
    /// 落选的其余取值（含会话压制值）。
    pub dissent: Vec<ValueArgument>,
    /// 全部候选（按对象字典序）。
    pub arguments: Vec<ValueArgument>,
    pub adjudicated_by: Option<String>,
    pub session_source: Option<String>,
    pub note: String,
}

impl SlotDecision {
    pub fn slot_label(&self) -> String {
        format!("{} 的 {}", self.subject, self.predicate)
    }

    pub fn is_contested(&self) -> bool {
        self.mode == DecisionMode::Contested
    }

    pub fn winner_objects(&self) -> Vec<&str> {
        self.winners.iter().map(|value| value.object.as_str()).collect()
    }

    pub fn dissent_objects(&self) -> Vec<&str> {
        self.dissent
            .iter()
            .chain(self.contested.iter())
            .map(|value| value.object.as_str())
            .collect()
    }

    /// 生成侧可直接使用的说理句。
    pub fn wording_zh(&self) -> String {
        let slot = self.slot_label();
        let winners = self
            .winners
            .iter()
            .map(|value| value.object.as_str())
            .collect::<Vec<_>>()
            .join("、");
        let dissent = self
            .dissent
            .iter()
            .map(|value| format!("「{}」（{}）", value.object, value.source_label()))
            .collect::<Vec<_>>()
            .join("、");
        match self.mode {
            DecisionMode::SessionCorrection => format!(
                "按本会话的更正，{} 是「{}」{}。",
                slot,
                winners,
                if dissent.is_empty() {
                    String::new()
                } else {
                    format!("；图谱旧说法「{}」已压制但未否定", dissent)
                }
            ),
            DecisionMode::Adjudicated => format!(
                "按显式裁决{}，{} 采用「{}」{}。",
                self.adjudicated_by
                    .as_ref()
                    .map(|id| format!("（{}）", id))
                    .unwrap_or_default(),
                slot,
                winners,
                if dissent.is_empty() {
                    String::new()
                } else {
                    format!("；另有说法 {}", dissent)
                }
            ),
            DecisionMode::Consensus => {
                let sources = self
                    .winners
                    .iter()
                    .map(|value| value.independent_sources)
                    .max()
                    .unwrap_or(0);
                format!(
                    "据 {} 个独立来源一致记录，{} 是「{}」。",
                    sources, slot, winners
                )
            }
            DecisionMode::SingleSource => format!(
                "据一个独立来源记录，{} 是「{}」。",
                slot, winners
            ),
            DecisionMode::TrustedWinner => format!(
                "综合信任层级与独立来源数，{} 采用「{}」{}。",
                slot,
                winners,
                if dissent.is_empty() {
                    String::new()
                } else {
                    format!("；另有来源记载 {}", dissent)
                }
            ),
            DecisionMode::Contested => format!(
                "关于{}，{} 种说法强度相当，暂不合并：{}。",
                slot,
                self.contested.len(),
                self.contested
                    .iter()
                    .map(|value| format!("「{}」（{}）", value.object, value.source_label()))
                    .collect::<Vec<_>>()
                    .join("、")
            ),
            DecisionMode::KnowledgeBaseOnly => format!(
                "知识库里记录 {} 是「{}」，但没有找到独立来源。",
                slot, winners
            ),
            DecisionMode::NoEvidence => format!("没有关于{}的可用来源证据。", slot),
        }
    }

    fn rank_for(&self, object: &str) -> Option<ArbitrationRank> {
        let argument = self
            .arguments
            .iter()
            .find(|argument| argument.object == object)?;
        let band = if self.contested.iter().any(|value| value.object == object) {
            RankBand::Contested
        } else if self.winners.iter().any(|value| value.object == object) {
            match self.mode {
                DecisionMode::SessionCorrection => RankBand::SessionCorrection,
                DecisionMode::Adjudicated => RankBand::Adjudicated,
                DecisionMode::KnowledgeBaseOnly => RankBand::KnowledgeBaseOnly,
                DecisionMode::NoEvidence => RankBand::NoEvidence,
                _ if argument.independent_sources == 0 && argument.materialized => {
                    RankBand::KnowledgeBaseOnly
                }
                _ => RankBand::Winner,
            }
        } else if self.dissent.iter().any(|value| value.object == object) {
            RankBand::Dissent
        } else {
            RankBand::Unsupported
        };
        Some(ArbitrationRank {
            band,
            trust: argument.trust,
            independent_sources: argument.independent_sources,
        })
    }

    fn render_zh(&self) -> String {
        let mut line = format!(
            "{} → {}：{}",
            self.slot_label(),
            self.mode.label_zh(),
            self.wording_zh()
        );
        if !self.note.is_empty() {
            line.push_str(&format!("（{}）", self.note));
        }
        line
    }
}

/// 重排档位：越小越靠前。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RankBand {
    SessionCorrection,
    Adjudicated,
    Winner,
    Contested,
    Dissent,
    KnowledgeBaseOnly,
    NoEvidence,
    Unsupported,
}

impl RankBand {
    pub fn label_zh(self) -> &'static str {
        match self {
            RankBand::SessionCorrection => "会话更正",
            RankBand::Adjudicated => "显式裁决",
            RankBand::Winner => "胜出",
            RankBand::Contested => "未决",
            RankBand::Dissent => "落选",
            RankBand::KnowledgeBaseOnly => "仅知识库",
            RankBand::NoEvidence => "无证据",
            RankBand::Unsupported => "未登记",
        }
    }
}

/// 一个候选的重排键：档位 → 信任层级 → 独立来源数。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArbitrationRank {
    pub band: RankBand,
    pub trust: Option<TrustLevel>,
    pub independent_sources: usize,
}

impl ArbitrationRank {
    fn trust_rank(&self) -> u8 {
        self.trust.map(|trust| trust.rank() + 1).unwrap_or(0)
    }

    /// 序数优先级（越大越靠前），供重排器做确定性比较。
    pub fn priority_score(&self) -> i64 {
        let band = RankBand::Unsupported as i64 - self.band as i64;
        band * 1_000_000 + i64::from(self.trust_rank()) * 10_000
            + self.independent_sources.min(999) as i64 * 10
    }
}

impl PartialOrd for ArbitrationRank {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ArbitrationRank {
    fn cmp(&self, other: &Self) -> Ordering {
        self.band
            .cmp(&other.band)
            .then_with(|| other.trust_rank().cmp(&self.trust_rank()))
            .then_with(|| other.independent_sources.cmp(&self.independent_sources))
    }
}

/// 一次回答期裁决的完整报告。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArbitrationReport {
    slots: Vec<SlotDecision>,
    notes: Vec<String>,
}

impl ArbitrationReport {
    pub fn slots(&self) -> &[SlotDecision] {
        &self.slots
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    pub fn notes(&self) -> &[String] {
        &self.notes
    }

    pub fn slot(&self, subject: &str, predicate: &str) -> Option<&SlotDecision> {
        self.slots
            .iter()
            .find(|slot| slot.subject == subject && slot.predicate == predicate)
    }

    pub fn rank_for(&self, subject: &str, predicate: &str, object: &str) -> ArbitrationRank {
        self.slot(subject, predicate)
            .and_then(|slot| slot.rank_for(object))
            .unwrap_or(ArbitrationRank {
                band: RankBand::Unsupported,
                trust: None,
                independent_sources: 0,
            })
    }

    pub fn compare(
        &self,
        left: (&str, &str, &str),
        right: (&str, &str, &str),
    ) -> Ordering {
        self.rank_for(left.0, left.1, left.2)
            .cmp(&self.rank_for(right.0, right.1, right.2))
    }

    pub fn render_zh(&self) -> String {
        if self.slots.is_empty() {
            return "融合裁决：没有参与裁决的槽位。".to_string();
        }
        let contested = self
            .slots
            .iter()
            .filter(|slot| slot.mode == DecisionMode::Contested)
            .count();
        let mut lines = vec![format!(
            "融合裁决：{} 个槽位，其中未决 {} 个。",
            self.slots.len(),
            contested
        )];
        for slot in &self.slots {
            lines.push(format!("  {}", slot.render_zh()));
        }
        for note in &self.notes {
            lines.push(format!("  注：{}", note));
        }
        lines.join("\n")
    }
}

// ---------------------------------------------------------------------------
// 裁决算法
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
struct ValueBuilder {
    object: String,
    object_kind: ObjectKind,
    trust: Option<TrustLevel>,
    components: BTreeSet<String>,
    namespaces: BTreeSet<String>,
    sources: BTreeSet<String>,
    materialized: bool,
    adjudicated_by: Option<String>,
    session_corrected: bool,
    session_suppressed: bool,
    dissent: BTreeSet<String>,
}

impl ValueBuilder {
    fn new(object: &str, object_kind: ObjectKind) -> Self {
        Self {
            object: object.to_string(),
            object_kind,
            ..Self::default()
        }
    }

    fn add_trust(&mut self, trust: Option<TrustLevel>) {
        if let Some(trust) = trust {
            self.trust = Some(match self.trust {
                Some(current) if current.rank() >= trust.rank() => current,
                _ => trust,
            });
        }
    }

    fn finish(self) -> ValueArgument {
        ValueArgument {
            object: self.object,
            object_kind: self.object_kind,
            trust: self.trust,
            independent_sources: self.components.len(),
            components: self.components.into_iter().collect(),
            namespaces: self.namespaces.into_iter().collect(),
            sources: self.sources.into_iter().collect(),
            materialized: self.materialized,
            adjudicated_by: self.adjudicated_by,
            session_corrected: self.session_corrected,
            session_suppressed: self.session_suppressed,
            dissent: self.dissent.into_iter().collect(),
        }
    }
}

#[derive(Debug, Default)]
struct SlotBuilder {
    functional: bool,
    values: BTreeMap<String, ValueBuilder>,
    adjudication: Option<Adjudication>,
    session: Option<SessionSlot>,
}

/// 对输入做回答期裁决。
pub fn arbitrate(input: &ArbitrationInput, policy: &SlotPolicy) -> ArbitrationReport {
    let components = input.catalog.component_map();
    let mut slots: BTreeMap<SlotKey, SlotBuilder> = BTreeMap::new();
    let mut notes = Vec::new();

    let mut assertions: Vec<&ArbitrationAssertion> = input.assertions.iter().collect();
    assertions.sort_by(|left, right| left.identity().cmp(&right.identity()));
    let mut seen_assertions: BTreeSet<(String, String, String, String, ObjectKind)> = BTreeSet::new();
    for assertion in assertions {
        if !seen_assertions.insert(assertion.identity()) {
            continue;
        }
        if assertion.subject.is_empty()
            || assertion.predicate.is_empty()
            || assertion.object.is_empty()
        {
            continue;
        }
        let Some(profile) = input.catalog.profile(&assertion.namespace) else {
            continue;
        };
        if !profile.is_active() {
            continue;
        }
        let slot = slots
            .entry(assertion.slot_key())
            .or_insert_with(|| SlotBuilder {
                functional: policy.is_functional(&assertion.predicate),
                ..SlotBuilder::default()
            });
        let value = slot
            .values
            .entry(assertion.object.clone())
            .or_insert_with(|| {
                ValueBuilder::new(&assertion.object, assertion.object_kind)
            });
        value.add_trust(Some(profile.trust));
        value.namespaces.insert(profile.namespace.clone());
        value.sources.insert(profile.label.clone());
        let origins: Vec<&SourceId> = assertion
            .origins
            .iter()
            .filter(|origin| origin.is_registered())
            .collect();
        if origins.is_empty() {
            if let Some(component) = components.get(&profile.namespace) {
                value.components.insert(component.clone());
            }
        } else {
            for origin in origins {
                value.components.insert(origin_component(origin));
                value.sources.insert(origin.render());
            }
        }
    }

    for winner in &input.applied {
        if winner.subject.is_empty() || winner.predicate.is_empty() || winner.object.is_empty() {
            continue;
        }
        let slot = slots
            .entry(SlotKey::new(&winner.subject, &winner.predicate))
            .or_insert_with(|| SlotBuilder {
                functional: policy.is_functional(&winner.predicate),
                ..SlotBuilder::default()
            });
        let value = slot
            .values
            .entry(winner.object.clone())
            .or_insert_with(|| ValueBuilder::new(&winner.object, winner.object_kind));
        value.materialized = true;
        if let Some(profile) = input.catalog.profile(&winner.namespace) {
            value.add_trust(Some(profile.trust));
            value.namespaces.insert(profile.namespace.clone());
        } else {
            notes.push(format!(
                "物化赢家 {} 的来源命名空间 {} 未登记，未继承信任层级。",
                winner.object, winner.namespace
            ));
        }
    }

    let mut adjudications: Vec<&Adjudication> = input.adjudications.iter().collect();
    adjudications.sort_by(|left, right| left.id.cmp(&right.id));
    let mut adjudication_by_predicate: BTreeMap<String, Adjudication> = BTreeMap::new();
    for adjudication in adjudications {
        adjudication_by_predicate
            .entry(adjudication.predicate.clone())
            .or_insert_with(|| adjudication.clone());
    }
    for (key, slot) in slots.iter_mut() {
        slot.adjudication = adjudication_by_predicate.get(&key.predicate).cloned();
    }

    for dissent in &input.dissent {
        let slot = slots
            .entry(dissent.slot_key())
            .or_insert_with(|| SlotBuilder {
                functional: policy.is_functional(&dissent.predicate),
                ..SlotBuilder::default()
            });
        let value = slot
            .values
            .entry(dissent.object.clone())
            .or_insert_with(|| ValueBuilder::new(&dissent.object, ObjectKind::Literal));
        value.dissent.insert(dissent.reason.clone());
        for source in &dissent.sources {
            value.sources.insert(source.clone());
        }
    }

    for (key, overlay) in input.session.slots() {
        let slot = slots.entry(key.clone()).or_insert_with(|| SlotBuilder {
            functional: policy.is_functional(&key.predicate),
            ..SlotBuilder::default()
        });
        slot.session = Some(overlay.clone());
        for object in &overlay.accepted {
            let value = slot
                .values
                .entry(object.clone())
                .or_insert_with(|| ValueBuilder::new(object, ObjectKind::Literal));
            value.session_corrected = true;
        }
        for object in &overlay.suppressed {
            let value = slot
                .values
                .entry(object.clone())
                .or_insert_with(|| ValueBuilder::new(object, ObjectKind::Literal));
            value.session_suppressed = true;
            value.dissent.insert(format!(
                "本会话第 {} 次记录更正为「{}」（来源：{}）",
                overlay.step,
                overlay.accepted.join("、"),
                overlay.source
            ));
        }
    }

    slots.retain(|key, slot| !key.subject.is_empty() && !slot.values.is_empty());
    let mut decisions: Vec<SlotDecision> =
        slots.into_iter().map(|(key, slot)| decide(key, slot)).collect();
    decisions.sort_by(|left, right| {
        left.subject
            .cmp(&right.subject)
            .then_with(|| left.predicate.cmp(&right.predicate))
    });
    ArbitrationReport {
        slots: decisions,
        notes,
    }
}

fn decide(key: SlotKey, mut slot: SlotBuilder) -> SlotDecision {
    let adjudication = slot.adjudication.take();
    if let Some(adjudication) = &adjudication {
        for value in slot.values.values_mut() {
            if value.namespaces.contains(&adjudication.prefer_namespace)
                || value
                    .components
                    .iter()
                    .any(|component| component == &format!("namespace:{}", adjudication.prefer_namespace))
            {
                value.adjudicated_by = Some(adjudication.id.clone());
            }
        }
    }

    let mut arguments: Vec<ValueArgument> = slot.values.into_values().map(ValueBuilder::finish).collect();
    arguments.sort_by(|left, right| {
        right
            .strength()
            .cmp(&left.strength())
            .then_with(|| left.object.cmp(&right.object))
    });

    let session_source = slot.session.as_ref().map(|session| session.source.clone());
    let adjudicated_by = adjudication.as_ref().map(|adjudication| adjudication.id.clone());

    if slot.session.is_some() {
        let corrected: Vec<ValueArgument> = arguments
            .iter()
            .filter(|value| value.session_corrected)
            .cloned()
            .collect();
        if !corrected.is_empty() {
            let winners = if slot.functional {
                corrected.into_iter().take(1).collect()
            } else {
                corrected
            };
            let (dissent, contested) = split_arguments(&arguments, &winners, false);
            return build_decision(
                key,
                slot.functional,
                DecisionMode::SessionCorrection,
                winners,
                contested,
                dissent,
                arguments,
                None,
                session_source,
                "会话更正只压制本槽的旧取值，旧值仍作为证据保留。".to_string(),
            );
        }
    }

    let adjudicated: Vec<ValueArgument> = arguments
        .iter()
        .filter(|value| value.adjudicated_by.is_some())
        .cloned()
        .collect();
    if !adjudicated.is_empty() {
        let top_strength = adjudicated
            .iter()
            .map(ValueArgument::strength)
            .max()
            .unwrap_or((0, 0));
        let top: Vec<ValueArgument> = adjudicated
            .into_iter()
            .filter(|value| value.strength() == top_strength)
            .collect();
        if top.len() > 1 {
            let (dissent, _) = split_arguments(&arguments, &top, true);
            return build_decision(
                key,
                slot.functional,
                DecisionMode::Contested,
                Vec::new(),
                top,
                dissent,
                arguments,
                adjudicated_by,
                session_source,
                "显式裁决指向了多个同强度取值，保持未决。".to_string(),
            );
        }
        let winners = top;
        let (dissent, contested) = split_arguments(&arguments, &winners, false);
        return build_decision(
            key,
            slot.functional,
            DecisionMode::Adjudicated,
            winners,
            contested,
            dissent,
            arguments,
            adjudicated_by.or_else(|| adjudication.map(|value| value.id)),
            session_source,
            "显式裁决优先于常规来源排序。".to_string(),
        );
    }

    if !slot.functional {
        let winners: Vec<ValueArgument> = arguments
            .iter()
            .filter(|value| value.independent_sources > 0 || value.materialized)
            .cloned()
            .collect();
        if winners.is_empty() {
            return build_decision(
                key,
                slot.functional,
                DecisionMode::NoEvidence,
                Vec::new(),
                Vec::new(),
                arguments.clone(),
                arguments,
                None,
                session_source,
                "多值槽没有任何来源支持。".to_string(),
            );
        }
        let all_corroborated = winners.iter().all(ValueArgument::corroborated);
        let any_corroborated = winners.iter().any(ValueArgument::corroborated);
        let any_source = winners.iter().any(|value| value.independent_sources > 0);
        let mode = if all_corroborated {
            DecisionMode::Consensus
        } else if any_corroborated {
            DecisionMode::Consensus
        } else if any_source {
            DecisionMode::SingleSource
        } else {
            DecisionMode::KnowledgeBaseOnly
        };
        let (dissent, contested) = split_arguments(&arguments, &winners, false);
        return build_decision(
            key,
            slot.functional,
            mode,
            winners,
            contested,
            dissent,
            arguments,
            adjudicated_by,
            session_source,
            "非功能槽允许多值并存，不把相容取值当冲突。".to_string(),
        );
    }

    let top_strength = arguments
        .iter()
        .map(ValueArgument::strength)
        .max()
        .unwrap_or((0, 0));
    let top: Vec<ValueArgument> = arguments
        .iter()
        .filter(|value| value.strength() == top_strength)
        .cloned()
        .collect();
    if top.is_empty() {
        return build_decision(
            key,
            slot.functional,
            DecisionMode::NoEvidence,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            arguments,
            adjudicated_by,
            session_source,
            "功能槽没有候选取值。".to_string(),
        );
    }
    if top.len() > 1 {
        let (dissent, _) = split_arguments(&arguments, &top, true);
        return build_decision(
            key,
            slot.functional,
            DecisionMode::Contested,
            Vec::new(),
            top,
            dissent,
            arguments,
            adjudicated_by,
            session_source,
            "同强度取值互相攻击，结果未决。".to_string(),
        );
    }

    let winner = top.into_iter().next().expect("top is non-empty");
    let others = arguments
        .iter()
        .filter(|value| value.object != winner.object)
        .count();
    let mode = if winner.independent_sources == 0 {
        if winner.materialized {
            DecisionMode::KnowledgeBaseOnly
        } else {
            DecisionMode::NoEvidence
        }
    } else if others > 0 {
        DecisionMode::TrustedWinner
    } else if winner.corroborated() {
        DecisionMode::Consensus
    } else {
        DecisionMode::SingleSource
    };
    let (dissent, contested) = split_arguments(&arguments, &[winner.clone()], false);
    let note = if mode == DecisionMode::KnowledgeBaseOnly {
        "默认图只是物化视图，不作为独立来源投票。".to_string()
    } else {
        String::new()
    };
    build_decision(
        key,
        slot.functional,
        mode,
        vec![winner],
        contested,
        dissent,
        arguments,
        adjudicated_by,
        session_source,
        note,
    )
}

fn split_arguments(
    arguments: &[ValueArgument],
    winners: &[ValueArgument],
    contested: bool,
) -> (Vec<ValueArgument>, Vec<ValueArgument>) {
    let winner_objects: BTreeSet<&str> = winners.iter().map(|value| value.object.as_str()).collect();
    if contested {
        let contested_values: Vec<ValueArgument> = winners.to_vec();
        let dissent: Vec<ValueArgument> = arguments
            .iter()
            .filter(|value| !winner_objects.contains(value.object.as_str()))
            .cloned()
            .collect();
        (dissent, contested_values)
    } else {
        let dissent: Vec<ValueArgument> = arguments
            .iter()
            .filter(|value| !winner_objects.contains(value.object.as_str()))
            .cloned()
            .collect();
        (dissent, Vec::new())
    }
}

#[allow(clippy::too_many_arguments)]
fn build_decision(
    key: SlotKey,
    functional: bool,
    mode: DecisionMode,
    winners: Vec<ValueArgument>,
    contested: Vec<ValueArgument>,
    dissent: Vec<ValueArgument>,
    arguments: Vec<ValueArgument>,
    adjudicated_by: Option<String>,
    session_source: Option<String>,
    note: String,
) -> SlotDecision {
    SlotDecision {
        subject: key.subject,
        predicate: key.predicate,
        functional,
        mode,
        winners,
        contested,
        dissent,
        arguments,
        adjudicated_by,
        session_source,
        note,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge_fusion::{Adjudication, NamespaceKind, NamespaceRecord, TrustLevel};
    use crate::provenance::{SourceId, SourceKind};
    use crate::slot_ledger::{AssertionStrength, SlotAssertion};

    fn record(namespace: &str, trust: TrustLevel) -> NamespaceRecord {
        NamespaceRecord::new(namespace, namespace, NamespaceKind::Import, trust)
    }

    fn assertion(
        namespace: &str,
        subject: &str,
        predicate: &str,
        object: &str,
    ) -> ArbitrationAssertion {
        ArbitrationAssertion::new(namespace, subject, predicate, object, ObjectKind::Literal)
    }

    fn input(namespaces: &[NamespaceRecord], assertions: &[ArbitrationAssertion]) -> ArbitrationInput {
        ArbitrationInput {
            catalog: SourceCatalog::from_records(namespaces),
            assertions: assertions.to_vec(),
            ..ArbitrationInput::default()
        }
    }

    fn policy() -> SlotPolicy {
        SlotPolicy::with_defaults()
    }

    #[test]
    fn single_independent_source_is_single_source() {
        let report = arbitrate(
            &input(
                &[record("a", TrustLevel::Imported)],
                &[assertion("a", "小明", "年龄", "18")],
            ),
            &policy(),
        );
        let slot = report.slot("小明", "年龄").expect("应有槽");
        assert_eq!(slot.mode, DecisionMode::SingleSource);
        assert_eq!(slot.winner_objects(), vec!["18"]);
        assert_eq!(slot.winners[0].independent_sources, 1);
        assert!(slot.wording_zh().contains("一个独立来源"));
    }

    #[test]
    fn two_independent_namespaces_are_consensus() {
        let report = arbitrate(
            &input(
                &[
                    record("a", TrustLevel::Imported),
                    record("b", TrustLevel::External),
                ],
                &[
                    assertion("a", "小明", "年龄", "18"),
                    assertion("b", "小明", "年龄", "18"),
                ],
            ),
            &policy(),
        );
        let slot = report.slot("小明", "年龄").expect("应有槽");
        assert_eq!(slot.mode, DecisionMode::Consensus);
        assert_eq!(slot.winners[0].independent_sources, 2);
        assert!(slot.dissent.is_empty());
    }

    #[test]
    fn same_file_imported_twice_is_not_consensus() {
        let origin = SourceId::new(SourceKind::Import, "people.csv", "小明,18");
        let catalog = SourceCatalog::from_records(&[
            record("batch-a", TrustLevel::Imported),
            record("batch-b", TrustLevel::Imported),
        ])
        .with_profile(
            SourceProfile::from_record(&record("batch-a", TrustLevel::Imported))
                .with_origin(origin.clone()),
        )
        .with_profile(
            SourceProfile::from_record(&record("batch-b", TrustLevel::Imported))
                .with_origin(origin),
        );
        let report = arbitrate(
            &ArbitrationInput {
                catalog,
                assertions: vec![
                    assertion("batch-a", "小明", "年龄", "18"),
                    assertion("batch-b", "小明", "年龄", "18"),
                ],
                ..ArbitrationInput::default()
            },
            &policy(),
        );
        let slot = report.slot("小明", "年龄").expect("应有槽");
        assert_eq!(slot.mode, DecisionMode::SingleSource);
        assert_eq!(slot.winners[0].independent_sources, 1);
    }

    #[test]
    fn mirrored_namespaces_do_not_create_consensus() {
        let report = arbitrate(
            &input(
                &[
                    record("mirror-a", TrustLevel::Imported),
                    record("mirror-b", TrustLevel::Imported),
                ],
                &[
                    assertion("mirror-a", "小明", "年龄", "18"),
                    assertion("mirror-b", "小明", "年龄", "18"),
                ],
            )
            .with_mirror("mirror-b", "mirror-a"),
            &policy(),
        );
        let slot = report.slot("小明", "年龄").expect("应有槽");
        assert_eq!(slot.mode, DecisionMode::SingleSource);
        assert_eq!(slot.winners[0].independent_sources, 1);
    }

    #[test]
    fn multiple_assertions_in_one_namespace_count_once() {
        let report = arbitrate(
            &input(
                &[record("a", TrustLevel::Imported)],
                &[
                    assertion("a", "小明", "年龄", "18"),
                    assertion("a", "小明", "年龄", "18"),
                ],
            ),
            &policy(),
        );
        let slot = report.slot("小明", "年龄").expect("应有槽");
        assert_eq!(slot.mode, DecisionMode::SingleSource);
        assert_eq!(slot.winners[0].independent_sources, 1);
    }

    #[test]
    fn materialized_default_graph_is_not_an_independent_source() {
        let catalog = SourceCatalog::from_records(&[record("kb", TrustLevel::External)]);
        let report = arbitrate(
            &ArbitrationInput {
                catalog,
                applied: vec![AppliedWinner {
                    id: "w1".to_string(),
                    namespace: "kb".to_string(),
                    subject: "小明".to_string(),
                    predicate: "年龄".to_string(),
                    object: "18".to_string(),
                    object_kind: ObjectKind::Literal,
                }],
                ..ArbitrationInput::default()
            },
            &policy(),
        );
        let slot = report.slot("小明", "年龄").expect("应有槽");
        assert_eq!(slot.mode, DecisionMode::KnowledgeBaseOnly);
        assert_eq!(slot.winners[0].independent_sources, 0);
        assert_eq!(slot.winners[0].trust, Some(TrustLevel::External));
        assert!(slot.winners[0].materialized);
        assert!(slot.wording_zh().contains("没有找到独立来源"));
    }

    #[test]
    fn applied_winner_inherits_namespace_trust() {
        let catalog = SourceCatalog::from_records(&[record("kb", TrustLevel::Speculative)]);
        let report = arbitrate(
            &ArbitrationInput {
                catalog,
                applied: vec![AppliedWinner {
                    id: "w1".to_string(),
                    namespace: "kb".to_string(),
                    subject: "小明".to_string(),
                    predicate: "年龄".to_string(),
                    object: "18".to_string(),
                    object_kind: ObjectKind::Literal,
                }],
                ..ArbitrationInput::default()
            },
            &policy(),
        );
        let slot = report.slot("小明", "年龄").expect("应有槽");
        assert_eq!(slot.winners[0].trust, Some(TrustLevel::Speculative));
        assert_ne!(slot.winners[0].trust, Some(TrustLevel::Curated));
    }

    #[test]
    fn higher_trust_beats_more_low_trust_sources() {
        let report = arbitrate(
            &input(
                &[
                    record("kb", TrustLevel::Curated),
                    record("x", TrustLevel::External),
                    record("y", TrustLevel::External),
                ],
                &[
                    assertion("kb", "小明", "年龄", "18"),
                    assertion("x", "小明", "年龄", "19"),
                    assertion("y", "小明", "年龄", "19"),
                ],
            ),
            &policy(),
        );
        let slot = report.slot("小明", "年龄").expect("应有槽");
        assert_eq!(slot.mode, DecisionMode::TrustedWinner);
        assert_eq!(slot.winner_objects(), vec!["18"]);
        assert_eq!(slot.dissent_objects(), vec!["19"]);
        assert_eq!(slot.dissent[0].independent_sources, 2);
    }

    #[test]
    fn equal_strength_conflict_is_contested() {
        let report = arbitrate(
            &input(
                &[
                    record("a", TrustLevel::Imported),
                    record("b", TrustLevel::Imported),
                ],
                &[
                    assertion("a", "小明", "年龄", "18"),
                    assertion("b", "小明", "年龄", "19"),
                ],
            ),
            &policy(),
        );
        let slot = report.slot("小明", "年龄").expect("应有槽");
        assert_eq!(slot.mode, DecisionMode::Contested);
        assert!(slot.winners.is_empty());
        assert_eq!(slot.contested.len(), 2);
        assert!(slot.wording_zh().contains("暂不合并"));
    }

    #[test]
    fn adjudication_overrides_trust_ordering() {
        let mut input = input(
            &[
                record("kb", TrustLevel::Curated),
                record("friend", TrustLevel::External),
            ],
            &[
                assertion("kb", "小明", "年龄", "18"),
                assertion("friend", "小明", "年龄", "19"),
            ],
        );
        input.adjudications.push(Adjudication::new(
            "年龄", "friend", "用户确认", "user",
        ));
        let report = arbitrate(&input, &policy());
        let slot = report.slot("小明", "年龄").expect("应有槽");
        assert_eq!(slot.mode, DecisionMode::Adjudicated);
        assert_eq!(slot.winner_objects(), vec!["19"]);
        assert!(slot.adjudicated_by.is_some());
        assert!(slot.wording_zh().contains("显式裁决"));
    }

    #[test]
    fn session_correction_overrides_graph_and_keeps_old_value_as_dissent() {
        let mut slots = SlotLedger::new();
        slots
            .apply(
                SlotAssertion::new(
                    "小明",
                    "年龄",
                    "18",
                    AssertionStrength::Direct,
                    "图谱",
                ),
                &["18".to_string()],
            )
            .unwrap();
        let transition = slots
            .apply(
                SlotAssertion::new(
                    "小明",
                    "年龄",
                    "19",
                    AssertionStrength::Correction,
                    "用户",
                ),
                &["18".to_string()],
            )
            .unwrap();
        assert!(matches!(
            transition.delta,
            crate::slot_ledger::SlotDelta::Corrected { .. }
        ));
        let mut input = input(
            &[record("kb", TrustLevel::Curated)],
            &[assertion("kb", "小明", "年龄", "18")],
        )
        .with_session_ledger(&slots);
        input.applied.push(AppliedWinner {
            id: "w1".to_string(),
            namespace: "kb".to_string(),
            subject: "小明".to_string(),
            predicate: "年龄".to_string(),
            object: "18".to_string(),
            object_kind: ObjectKind::Literal,
        });
        let report = arbitrate(&input, &policy());
        let slot = report.slot("小明", "年龄").expect("应有槽");
        assert_eq!(slot.mode, DecisionMode::SessionCorrection);
        assert_eq!(slot.winner_objects(), vec!["19"]);
        assert_eq!(slot.dissent_objects(), vec!["18"]);
        assert!(slot.wording_zh().contains("按本会话的更正"));
        assert!(slot.wording_zh().contains("压制但未否定"));
    }

    #[test]
    fn direct_session_assertion_is_not_a_correction() {
        let mut slots = SlotLedger::new();
        slots
            .apply(
                SlotAssertion::new(
                    "小明",
                    "年龄",
                    "19",
                    AssertionStrength::Direct,
                    "用户",
                ),
                &["18".to_string()],
            )
            .unwrap();
        assert!(SessionOverlay::from_ledger(&slots).is_empty());
    }

    #[test]
    fn hearsay_is_not_a_session_correction() {
        let mut slots = SlotLedger::new();
        slots
            .apply(
                SlotAssertion::new(
                    "小明",
                    "年龄",
                    "19",
                    AssertionStrength::Hearsay,
                    "据说",
                ),
                &["18".to_string()],
            )
            .unwrap();
        assert!(SessionOverlay::from_ledger(&slots).is_empty());
    }

    #[test]
    fn dissent_blocks_false_consensus_from_dissenting_certificate() {
        let mut input = input(
            &[record("a", TrustLevel::Imported)],
            &[assertion("a", "小明", "年龄", "18")],
        );
        input = input.with_dissent(DissentEvidence::new(
            "小明",
            "年龄",
            "19",
            "冲突证书 cert-1",
        ));
        let report = arbitrate(&input, &policy());
        let slot = report.slot("小明", "年龄").expect("应有槽");
        assert_ne!(slot.mode, DecisionMode::Consensus);
        assert_eq!(slot.mode, DecisionMode::TrustedWinner);
        assert_eq!(slot.dissent_objects(), vec!["19"]);
    }

    #[test]
    fn provenance_origins_collapse_duplicate_imports() {
        let origin = SourceId::new(SourceKind::Import, "people.csv", "小明,18");
        let input = input(
            &[
                record("batch-a", TrustLevel::Imported),
                record("batch-b", TrustLevel::Imported),
            ],
            &[
                assertion("batch-a", "小明", "年龄", "18")
                    .with_origin(origin.clone()),
                assertion("batch-b", "小明", "年龄", "18").with_origin(origin),
            ],
        );
        let report = arbitrate(&input, &policy());
        let slot = report.slot("小明", "年龄").expect("应有槽");
        assert_eq!(slot.mode, DecisionMode::SingleSource);
        assert_eq!(slot.winners[0].independent_sources, 1);
    }

    #[test]
    fn rank_orders_session_correction_and_certified_winner_first() {
        let mut slots = SlotLedger::new();
        slots
            .apply(
                SlotAssertion::new("小明", "年龄", "19", AssertionStrength::Correction, "用户"),
                &["18".to_string()],
            )
            .unwrap();
        let input = input(
            &[record("kb", TrustLevel::Curated)],
            &[assertion("kb", "小明", "年龄", "18")],
        )
        .with_session_ledger(&slots);
        let report = arbitrate(&input, &policy());
        assert_eq!(
            report.rank_for("小明", "年龄", "19").band,
            RankBand::SessionCorrection
        );
        assert_eq!(
            report.rank_for("小明", "年龄", "18").band,
            RankBand::Dissent
        );
        assert!(
            report.compare(
                ("小明", "年龄", "19"),
                ("小明", "年龄", "18")
            ) == Ordering::Less
        );
        assert_eq!(
            report.rank_for("小明", "身高", "175").band,
            RankBand::Unsupported
        );
    }

    #[test]
    fn contested_values_share_the_same_rank() {
        let report = arbitrate(
            &input(
                &[
                    record("a", TrustLevel::Imported),
                    record("b", TrustLevel::Imported),
                ],
                &[
                    assertion("a", "小明", "年龄", "18"),
                    assertion("b", "小明", "年龄", "19"),
                ],
            ),
            &policy(),
        );
        assert_eq!(
            report.rank_for("小明", "年龄", "18"),
            report.rank_for("小明", "年龄", "19")
        );
        assert_eq!(
            report.rank_for("小明", "年龄", "18").band,
            RankBand::Contested
        );
    }

    #[test]
    fn materialized_only_value_is_ranked_as_knowledge_base_in_mixed_slot() {
        let catalog = SourceCatalog::from_records(&[
            record("a", TrustLevel::Imported),
            record("kb", TrustLevel::Curated),
        ]);
        let report = arbitrate(
            &ArbitrationInput {
                catalog,
                assertions: vec![assertion("a", "苹果", "颜色", "红色")],
                applied: vec![AppliedWinner {
                    id: "w1".to_string(),
                    namespace: "kb".to_string(),
                    subject: "苹果".to_string(),
                    predicate: "颜色".to_string(),
                    object: "绿色".to_string(),
                    object_kind: ObjectKind::Literal,
                }],
                ..ArbitrationInput::default()
            },
            &policy(),
        );
        let slot = report.slot("苹果", "颜色").expect("应有槽");
        assert_eq!(slot.winners.len(), 2);
        assert_eq!(
            report.rank_for("苹果", "颜色", "红色").band,
            RankBand::Winner
        );
        assert_eq!(
            report.rank_for("苹果", "颜色", "绿色").band,
            RankBand::KnowledgeBaseOnly
        );
    }

    #[test]
    fn non_functional_slot_keeps_compatible_values() {
        let report = arbitrate(
            &input(
                &[record("a", TrustLevel::Imported)],
                &[
                    assertion("a", "苹果", "颜色", "红色"),
                    assertion("a", "苹果", "颜色", "绿色"),
                ],
            ),
            &policy(),
        );
        let slot = report.slot("苹果", "颜色").expect("应有槽");
        assert_eq!(slot.mode, DecisionMode::SingleSource);
        assert_eq!(slot.winners.len(), 2);
        assert!(slot.contested.is_empty());
    }

    #[test]
    fn report_is_deterministic_regardless_of_input_order() {
        let namespaces = [
            record("b", TrustLevel::External),
            record("a", TrustLevel::Imported),
        ];
        let forward = arbitrate(
            &input(
                &namespaces,
                &[
                    assertion("a", "小明", "年龄", "18"),
                    assertion("b", "小明", "年龄", "19"),
                ],
            ),
            &policy(),
        );
        let reverse = arbitrate(
            &input(
                &[namespaces[1].clone(), namespaces[0].clone()],
                &[
                    assertion("b", "小明", "年龄", "19"),
                    assertion("a", "小明", "年龄", "18"),
                ],
            ),
            &policy(),
        );
        assert_eq!(forward, reverse);
    }

    #[test]
    fn unknown_or_inactive_namespaces_do_not_vote() {
        let mut retired = record("old", TrustLevel::Verified);
        retired.status = NamespaceStatus::Retired;
        let report = arbitrate(
            &input(
                &[record("live", TrustLevel::Imported), retired],
                &[
                    assertion("live", "小明", "年龄", "18"),
                    assertion("old", "小明", "年龄", "19"),
                    assertion("ghost", "小明", "年龄", "20"),
                ],
            ),
            &policy(),
        );
        let slot = report.slot("小明", "年龄").expect("应有槽");
        assert_eq!(slot.mode, DecisionMode::SingleSource);
        assert_eq!(slot.winner_objects(), vec!["18"]);
        assert_eq!(slot.arguments.len(), 1);
    }

    #[test]
    fn source_catalog_component_is_stable_and_idempotent() {
        let origin = SourceId::new(SourceKind::Import, "file.csv", "payload");
        let catalog = SourceCatalog::from_records(&[record("a", TrustLevel::Imported)])
            .with_profile(
                SourceProfile::from_record(&record("a", TrustLevel::Imported))
                    .with_origin(origin.clone()),
            )
            .with_mirror("b", "a");
        let first = catalog.component_map();
        let second = catalog.component_map();
        assert_eq!(first, second);
        assert_eq!(first.get("a"), first.get("b"));
    }
}

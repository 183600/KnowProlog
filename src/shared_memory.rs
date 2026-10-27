//! SHM：并发安全的共享语义白板（Shared Semantic Whiteboard）。
//!
//! 编年史是事后档案，会话快照是单会话最后一帧；它们都不回答一个问题：
//! A 会话显式分享的声明，B 会话能否在**同一次运行**里安全地想起来？本模块
//! 提供一层有界、作用域隔离、不可变快照的共享记忆：
//!
//! - [`MemoryScope`]：默认私有；跨会话可见必须显式选 [`MemoryScope::Global`]
//!   或同会话作用域。全局视图永远不包含其他会话的私有声明。
//! - [`SharedMemoryBoard`]：单写者语义（内部 `RwLock`），发布按
//!   「作用域 + 主体身份 + 谓词身份 + 取值身份」幂等去重；不同取值不会
//!   覆盖旧值，而是留下可审计的 [`ConflictCertificate`]。
//! - [`SharedMemoryView`]：发布后整体替换的不可变快照（`Arc`）；读方拿到
//!   视图后不再持锁，推理期间不会因别的会话发布而阻塞。
//! - [`SharedClaim`]：显示字面与身份键并存；比较、去重、冲突检测一律走
//!   [`crate::text_identity::TextIdentity`] 的身份键，零宽/全角噪声不会造成
//!   「看起来一样、哈希不同」的幽灵。
//!
//! 纪律：共享的是 claim，不是自动写入图谱的 fact；升格为图谱事实仍走既有
//! 审核/教学路径。冲突不替用户拍板：视图同时保留两种说法，并给出稳定证书。

use std::collections::BTreeMap;
use std::sync::Arc;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::context_image::{canonical_json, fnv1a_hex};
use crate::text_identity::TextIdentity;

/// 共享记忆格式版本。
pub const SHARED_MEMORY_VERSION: u32 = 1;
/// 默认容量（声明条数，不是字符数）。
pub const DEFAULT_MAX_CLAIMS: usize = 4096;
/// 单个字段展示文本上限（字符）。
pub const FIELD_TEXT_LIMIT: usize = 256;
/// 同一键下最多保留的冲突取值数（超出仍返回证书，但不无限增长）。
pub const MAX_VALUES_PER_KEY: usize = 16;

/// 共享作用域：默认私有，跨会话共享必须显式。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", content = "session_id", rename_all = "snake_case")]
pub enum MemoryScope {
    /// 所有会话可见。
    Global,
    /// 仅指定会话可见。
    Session(String),
}

impl MemoryScope {
    pub fn session(id: impl Into<String>) -> Self {
        MemoryScope::Session(id.into())
    }

    pub fn is_global(&self) -> bool {
        matches!(self, MemoryScope::Global)
    }

    pub fn session_id(&self) -> Option<&str> {
        match self {
            MemoryScope::Global => None,
            MemoryScope::Session(id) => Some(id),
        }
    }

    pub fn visible_to(&self, session: Option<&str>) -> bool {
        match self {
            MemoryScope::Global => true,
            MemoryScope::Session(id) => session == Some(id.as_str()),
        }
    }

    fn key(&self) -> String {
        match self {
            MemoryScope::Global => "global".to_string(),
            // 长度前缀让任意会话 id（含内部分隔符）都不能与另一个作用域前缀碰撞。
            MemoryScope::Session(id) => {
                format!("session\u{1}{}\u{2}{}", id.chars().count(), id)
            }
        }
    }
}

/// 发布失败原因（类型化，不用字符串折叠）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PublishError {
    /// 必填字段为空（主体 / 谓词 / 取值 / 来源 / 会话 id）。
    EmptyField { field: &'static str },
    /// 作用域字段超长。
    TooLong { field: &'static str, limit: usize },
    /// 白板容量为 0（显式关闭）。
    Disabled,
}

impl PublishError {
    pub fn code(&self) -> &'static str {
        match self {
            PublishError::EmptyField { .. } => "empty_field",
            PublishError::TooLong { .. } => "too_long",
            PublishError::Disabled => "disabled",
        }
    }
}

impl std::fmt::Display for PublishError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PublishError::EmptyField { field } => write!(formatter, "字段「{}」不能为空", field),
            PublishError::TooLong { field, limit } => {
                write!(formatter, "字段「{}」超过 {} 字符", field, limit)
            }
            PublishError::Disabled => write!(formatter, "共享记忆已关闭"),
        }
    }
}

impl std::error::Error for PublishError {}

/// 一条待发布的声明（显示字面 + 显式作用域）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NewClaim {
    pub subject: String,
    pub predicate: String,
    pub value: String,
    #[serde(default)]
    pub source: String,
    pub scope: MemoryScope,
}

impl NewClaim {
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        value: impl Into<String>,
        scope: MemoryScope,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            value: value.into(),
            source: String::new(),
            scope,
        }
    }

    pub fn global(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        Self::new(subject, predicate, value, MemoryScope::Global)
    }

    pub fn session(
        session_id: impl Into<String>,
        subject: impl Into<String>,
        predicate: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        Self::new(
            subject,
            predicate,
            value,
            MemoryScope::Session(session_id.into()),
        )
    }

    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = source.into();
        self
    }

    fn validate(&self) -> Result<(), PublishError> {
        for (field, value) in [
            ("subject", self.subject.as_str()),
            ("predicate", self.predicate.as_str()),
            ("value", self.value.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(PublishError::EmptyField { field });
            }
        }
        if let MemoryScope::Session(id) = &self.scope {
            if id.trim().is_empty() {
                return Err(PublishError::EmptyField {
                    field: "session_id",
                });
            }
            if id.chars().count() > FIELD_TEXT_LIMIT {
                return Err(PublishError::TooLong {
                    field: "session_id",
                    limit: FIELD_TEXT_LIMIT,
                });
            }
        }
        Ok(())
    }
}

/// 已发布的共享声明：显示与身份并存。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SharedClaim {
    pub id: String,
    pub scope: MemoryScope,
    pub subject: String,
    pub subject_key: String,
    pub predicate: String,
    pub predicate_key: String,
    pub value: String,
    pub value_key: String,
    pub source: String,
    /// 发布时的全局修订号（单调、可断言）。
    pub revision: u64,
}

/// 同一作用域、主体、谓词下的冲突证书。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConflictCertificate {
    pub id: String,
    pub scope: MemoryScope,
    pub subject: String,
    pub subject_key: String,
    pub predicate: String,
    pub predicate_key: String,
    /// 全部取值（显示字面，按身份键排序）。
    pub values: Vec<String>,
    /// 对应的身份键。
    pub value_keys: Vec<String>,
    /// 对应的声明 id。
    pub claim_ids: Vec<String>,
    /// 证书生成时的全局修订号。
    pub revision: u64,
}

impl ConflictCertificate {
    pub fn render(&self) -> String {
        format!(
            "「{}」的「{}」存在 {} 种说法：{}。",
            self.subject,
            self.predicate,
            self.values.len(),
            self.values.join("、")
        )
    }
}

/// 发布结果（幂等重复 / 新增 / 冲突新增）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum PublishOutcome {
    Published {
        claim: SharedClaim,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        conflict: Option<ConflictCertificate>,
        revision: u64,
        evicted: u64,
    },
    Duplicate {
        claim_id: String,
        revision: u64,
        values: Vec<String>,
    },
    /// 同一键的取值数已达上限：保留既有声明，返回证书但不写入新值。
    Saturated {
        conflict: ConflictCertificate,
        revision: u64,
    },
}

impl PublishOutcome {
    pub fn status(&self) -> &'static str {
        match self {
            PublishOutcome::Published { conflict: None, .. } => "published",
            PublishOutcome::Published {
                conflict: Some(_), ..
            } => "conflict",
            PublishOutcome::Duplicate { .. } => "duplicate",
            PublishOutcome::Saturated { .. } => "saturated",
        }
    }

    pub fn revision(&self) -> u64 {
        match self {
            PublishOutcome::Published { revision, .. }
            | PublishOutcome::Duplicate { revision, .. }
            | PublishOutcome::Saturated { revision, .. } => *revision,
        }
    }

    pub fn conflict(&self) -> Option<&ConflictCertificate> {
        match self {
            PublishOutcome::Published { conflict, .. } => conflict.as_ref(),
            PublishOutcome::Saturated { conflict, .. } => Some(conflict),
            PublishOutcome::Duplicate { .. } => None,
        }
    }
}

/// 一次读取的不可变视图。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SharedMemoryView {
    pub version: u32,
    pub revision: u64,
    /// 过滤作用域：`None` = 只读全局；`Some(session)` = 全局 + 该会话私有。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    pub claims: Vec<SharedClaim>,
    pub conflicts: Vec<ConflictCertificate>,
    pub evictions: u64,
    /// 覆盖可见声明与冲突的稳定摘要（不含全局修订号）。
    pub digest: String,
}

impl SharedMemoryView {
    pub fn is_empty(&self) -> bool {
        self.claims.is_empty()
    }

    pub fn len(&self) -> usize {
        self.claims.len()
    }

    pub fn conflict_count(&self) -> usize {
        self.conflicts.len()
    }

    pub fn values(&self, subject: &str, predicate: &str) -> Vec<&str> {
        let subject = TextIdentity::new(subject);
        let predicate = TextIdentity::new(predicate);
        self.claims
            .iter()
            .filter(|claim| {
                claim.subject_key == subject.identity()
                    && claim.predicate_key == predicate.identity()
            })
            .map(|claim| claim.value.as_str())
            .collect()
    }

    pub fn render_receipt(&self) -> String {
        format!(
            "共享记忆：{} 条声明，{} 个冲突，修订 {}，摘要 {}。",
            self.claims.len(),
            self.conflicts.len(),
            self.revision,
            self.digest
        )
    }
}

#[derive(Debug)]
struct SharedState {
    revision: u64,
    evictions: u64,
    max_claims: usize,
    total: usize,
    entries: BTreeMap<String, Vec<SharedClaim>>,
}

impl SharedState {
    fn new(max_claims: usize) -> Self {
        Self {
            revision: 0,
            evictions: 0,
            max_claims,
            total: 0,
            entries: BTreeMap::new(),
        }
    }

    fn evict_overflow(&mut self) -> u64 {
        let mut evicted = 0u64;
        while self.total > self.max_claims {
            let oldest = self
                .entries
                .iter()
                .flat_map(|(key, claims)| {
                    claims.iter().enumerate().map(move |(index, claim)| {
                        (key.clone(), index, claim.revision, claim.id.clone())
                    })
                })
                .min_by(|left, right| {
                    left.2
                        .cmp(&right.2)
                        .then_with(|| left.3.cmp(&right.3))
                });
            let Some((key, index, _, _)) = oldest else {
                break;
            };
            if let Some(claims) = self.entries.get_mut(&key) {
                if index < claims.len() {
                    claims.remove(index);
                    self.total = self.total.saturating_sub(1);
                    evicted = evicted.saturating_add(1);
                }
                if claims.is_empty() {
                    self.entries.remove(&key);
                }
            } else {
                break;
            }
        }
        self.evictions = self.evictions.saturating_add(evicted);
        evicted
    }

    fn build_view(&self, session_id: Option<&str>) -> SharedMemoryView {
        let mut claims: Vec<SharedClaim> = self
            .entries
            .values()
            .flatten()
            .filter(|claim| claim.scope.visible_to(session_id))
            .cloned()
            .collect();
        claims.sort_by(|left, right| {
            left.scope
                .cmp(&right.scope)
                .then_with(|| left.subject_key.cmp(&right.subject_key))
                .then_with(|| left.predicate_key.cmp(&right.predicate_key))
                .then_with(|| left.value_key.cmp(&right.value_key))
                .then_with(|| left.id.cmp(&right.id))
        });
        let conflicts = collect_conflicts(&claims, self.revision);
        let digest = view_digest(session_id, &claims, &conflicts);
        SharedMemoryView {
            version: SHARED_MEMORY_VERSION,
            revision: self.revision,
            session_id: session_id.map(str::to_string),
            claims,
            conflicts,
            evictions: self.evictions,
            digest,
        }
    }
}

/// 并发共享白板：发布走写锁，读取返回不可变快照。
#[derive(Debug)]
pub struct SharedMemoryBoard {
    state: RwLock<SharedState>,
}

impl Default for SharedMemoryBoard {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_CLAIMS)
    }
}

impl SharedMemoryBoard {
    pub fn new(max_claims: usize) -> Self {
        Self {
            state: RwLock::new(SharedState::new(max_claims)),
        }
    }

    pub fn max_claims(&self) -> usize {
        self.state.read().max_claims
    }

    pub fn len(&self) -> usize {
        self.state.read().total
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn revision(&self) -> u64 {
        self.state.read().revision
    }

    pub fn evictions(&self) -> u64 {
        self.state.read().evictions
    }

    /// 按 id 撤回一条声明，但只在作用域完全匹配时生效（防止跨作用域越权删除）。
    /// 返回被撤回的声明；有变化时推进一次修订号。
    pub fn retract_scoped(&self, claim_id: &str, scope: &MemoryScope) -> Option<SharedClaim> {
        let mut state = self.state.write();
        let (entry_key, index) = state.entries.iter().find_map(|(key, claims)| {
            claims
                .iter()
                .position(|claim| claim.id == claim_id && &claim.scope == scope)
                .map(|index| (key.clone(), index))
        })?;
        let claims = state.entries.get_mut(&entry_key)?;
        let claim = claims.remove(index);
        if claims.is_empty() {
            state.entries.remove(&entry_key);
        }
        state.total = state.total.saturating_sub(1);
        state.revision = state.revision.saturating_add(1);
        Some(claim)
    }

    /// 删除某会话的全部私有声明（会话关闭/删除时调用，防止 id 复用时泄漏）。
    /// 返回删除的声明条数；有删除时推进一次修订号。
    pub fn purge_session(&self, session_id: &str) -> usize {
        let prefix = format!("{}\u{1}", MemoryScope::session(session_id).key());
        let mut state = self.state.write();
        let keys: Vec<String> = state
            .entries
            .keys()
            .filter(|key| key.starts_with(&prefix))
            .cloned()
            .collect();
        let mut removed = 0usize;
        for key in keys {
            if let Some(claims) = state.entries.remove(&key) {
                removed += claims.len();
            }
        }
        if removed > 0 {
            state.total = state.total.saturating_sub(removed);
            state.revision = state.revision.saturating_add(1);
        }
        removed
    }

    /// 发布一条声明；同键同值幂等，不同值保留并返回冲突证书。
    pub fn publish(&self, claim: NewClaim) -> Result<PublishOutcome, PublishError> {
        claim.validate()?;
        // 先按展示上限截断，再派生身份键：否则「前 256 字相同、尾部不同」的
        // 两个输入会得到相同展示、不同身份键，制造无法解释的幽灵冲突。
        let subject_display = truncate_chars(&claim.subject, FIELD_TEXT_LIMIT);
        let predicate_display = truncate_chars(&claim.predicate, FIELD_TEXT_LIMIT);
        let value_display = truncate_chars(&claim.value, FIELD_TEXT_LIMIT);
        let source_display = truncate_chars(&claim.source, FIELD_TEXT_LIMIT);
        let subject_key = TextIdentity::new(&subject_display).identity().to_string();
        let predicate_key = TextIdentity::new(&predicate_display).identity().to_string();
        let value_key = TextIdentity::new(&value_display).identity().to_string();
        if subject_key.is_empty() {
            return Err(PublishError::EmptyField { field: "subject" });
        }
        if predicate_key.is_empty() {
            return Err(PublishError::EmptyField { field: "predicate" });
        }
        if value_key.is_empty() {
            return Err(PublishError::EmptyField { field: "value" });
        }
        let entry_key = format!(
            "{}\u{1}{}\u{1}{}",
            claim.scope.key(),
            subject_key,
            predicate_key
        );

        let mut state = self.state.write();
        if state.max_claims == 0 {
            return Err(PublishError::Disabled);
        }
        if let Some(existing) = state
            .entries
            .get(&entry_key)
            .and_then(|claims| claims.iter().find(|item| item.value_key == value_key))
        {
            let mut values = state
                .entries
                .get(&entry_key)
                .map(|claims| {
                    claims
                        .iter()
                        .map(|item| item.value.clone())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            values.sort();
            values.dedup();
            return Ok(PublishOutcome::Duplicate {
                claim_id: existing.id.clone(),
                revision: state.revision,
                values,
            });
        }

        let already_full = state
            .entries
            .get(&entry_key)
            .map(|claims| claims.len() >= MAX_VALUES_PER_KEY)
            .unwrap_or(false);
        if already_full {
            let certificate = state
                .entries
                .get(&entry_key)
                .map(|claims| {
                    certificate_for(
                        &claim.scope,
                        &subject_key,
                        &predicate_key,
                        claims,
                        state.revision,
                    )
                })
                .expect("已确认键存在");
            return Ok(PublishOutcome::Saturated {
                conflict: certificate,
                revision: state.revision,
            });
        }

        state.revision = state.revision.saturating_add(1);
        let revision = state.revision;
        let claim = SharedClaim {
            id: claim_id(
                &claim.scope,
                &subject_key,
                &predicate_key,
                &value_key,
            ),
            scope: claim.scope.clone(),
            subject: subject_display,
            subject_key,
            predicate: predicate_display,
            predicate_key,
            value: value_display,
            value_key,
            source: source_display,
            revision,
        };
        state
            .entries
            .entry(entry_key.clone())
            .or_default()
            .push(claim.clone());
        state.total = state.total.saturating_add(1);
        let evicted = state.evict_overflow();
        let conflict = state.entries.get(&entry_key).and_then(|claims| {
            (claims.iter().map(|item| item.value_key.as_str()).collect::<std::collections::BTreeSet<_>>().len() > 1)
                .then(|| {
                    certificate_for(
                        &claim.scope,
                        &claim.subject_key,
                        &claim.predicate_key,
                        claims,
                        state.revision,
                    )
                })
        });
        let outcome = PublishOutcome::Published {
            claim,
            conflict,
            revision: state.revision,
            evicted,
        };
        Ok(outcome)
    }

    /// 全局视图（只含显式共享到全局的声明）。
    pub fn snapshot(&self) -> Arc<SharedMemoryView> {
        Arc::new(self.state.read().build_view(None))
    }

    /// 可见视图：全局 + 指定会话私有。
    pub fn snapshot_for(&self, session_id: &str) -> Arc<SharedMemoryView> {
        Arc::new(self.state.read().build_view(Some(session_id)))
    }

    /// 管理/诊断视图：包含全部作用域，不用于回答。
    pub fn snapshot_all(&self) -> Arc<SharedMemoryView> {
        let state = self.state.read();
        let mut claims: Vec<SharedClaim> =
            state.entries.values().flatten().cloned().collect();
        claims.sort_by(|left, right| {
            left.scope
                .cmp(&right.scope)
                .then_with(|| left.subject_key.cmp(&right.subject_key))
                .then_with(|| left.predicate_key.cmp(&right.predicate_key))
                .then_with(|| left.value_key.cmp(&right.value_key))
                .then_with(|| left.id.cmp(&right.id))
        });
        let conflicts = collect_conflicts(&claims, state.revision);
        let digest = view_digest(None, &claims, &conflicts);
        Arc::new(SharedMemoryView {
            version: SHARED_MEMORY_VERSION,
            revision: state.revision,
            session_id: None,
            claims,
            conflicts,
            evictions: state.evictions,
            digest,
        })
    }

    pub fn clear(&self) {
        let mut state = self.state.write();
        state.entries.clear();
        state.total = 0;
        state.revision = state.revision.saturating_add(1);
    }
}

fn claim_id(
    scope: &MemoryScope,
    subject_key: &str,
    predicate_key: &str,
    value_key: &str,
) -> String {
    let value = serde_json::json!({
        "scope": scope,
        "subject": subject_key,
        "predicate": predicate_key,
        "value": value_key,
    });
    format!("claim:{}", fnv1a_hex(canonical_json(&value).as_bytes()))
}

fn certificate_for(
    scope: &MemoryScope,
    subject_key: &str,
    predicate_key: &str,
    claims: &[SharedClaim],
    revision: u64,
) -> ConflictCertificate {
    let mut values: Vec<(String, &SharedClaim)> = claims
        .iter()
        .map(|claim| (claim.value_key.clone(), claim))
        .collect();
    values.sort_by(|left, right| left.0.cmp(&right.0));
    let subject = claims
        .first()
        .map(|claim| claim.subject.clone())
        .unwrap_or_default();
    let predicate = claims
        .first()
        .map(|claim| claim.predicate.clone())
        .unwrap_or_default();
    let value_keys: Vec<String> = values.iter().map(|(key, _)| key.clone()).collect();
    let claim_ids: Vec<String> = values
        .iter()
        .map(|(_, claim)| claim.id.clone())
        .collect();
    let value_texts: Vec<String> = values
        .iter()
        .map(|(_, claim)| claim.value.clone())
        .collect();
    let id_value = serde_json::json!({
        "scope": scope,
        "subject": subject_key,
        "predicate": predicate_key,
        "values": value_keys,
    });
    ConflictCertificate {
        id: format!("conflict:{}", fnv1a_hex(canonical_json(&id_value).as_bytes())),
        scope: scope.clone(),
        subject,
        subject_key: subject_key.to_string(),
        predicate,
        predicate_key: predicate_key.to_string(),
        values: value_texts,
        value_keys,
        claim_ids,
        revision,
    }
}

fn collect_conflicts(claims: &[SharedClaim], revision: u64) -> Vec<ConflictCertificate> {
    let mut groups: BTreeMap<String, Vec<&SharedClaim>> = BTreeMap::new();
    for claim in claims {
        let key = format!(
            "{}\u{1}{}\u{1}{}",
            claim.scope.key(),
            claim.subject_key,
            claim.predicate_key
        );
        groups.entry(key).or_default().push(claim);
    }
    groups
        .values()
        .filter_map(|group| {
            let distinct: std::collections::BTreeSet<&str> =
                group.iter().map(|claim| claim.value_key.as_str()).collect();
            if distinct.len() <= 1 {
                return None;
            }
            let first = group.first()?;
            Some(certificate_for(
                &first.scope,
                &first.subject_key,
                &first.predicate_key,
                &group.iter().map(|claim| (*claim).clone()).collect::<Vec<_>>(),
                revision,
            ))
        })
        .collect()
}

fn view_digest(
    session_id: Option<&str>,
    claims: &[SharedClaim],
    conflicts: &[ConflictCertificate],
) -> String {
    let value = serde_json::json!({
        "version": SHARED_MEMORY_VERSION,
        "session_id": session_id,
        "claims": claims.iter().map(|claim| serde_json::json!({
            "id": claim.id,
            "scope": claim.scope,
            "subject": claim.subject_key,
            "predicate": claim.predicate_key,
            "value": claim.value_key,
        })).collect::<Vec<_>>(),
        "conflicts": conflicts.iter().map(|conflict| serde_json::json!({
            "id": conflict.id,
            "values": conflict.value_keys,
        })).collect::<Vec<_>>(),
    });
    fnv1a_hex(canonical_json(&value).as_bytes())
}

fn truncate_chars(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_string();
    }
    if limit == 0 {
        return String::new();
    }
    let mut output: String = text.chars().take(limit - 1).collect();
    output.push('…');
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn publish_is_idempotent_and_identity_aware() {
        let board = SharedMemoryBoard::new(16);
        let first = board
            .publish(NewClaim::global("张\u{200B}三", "身高", "226cm").with_source("用户"))
            .unwrap();
        assert_eq!(first.status(), "published");
        assert_eq!(board.len(), 1);
        assert_eq!(board.revision(), 1);

        let duplicate = board
            .publish(NewClaim::global("张三", "身高", "２２６cm"))
            .unwrap();
        assert_eq!(duplicate.status(), "duplicate");
        assert_eq!(board.len(), 1);
        assert_eq!(board.revision(), 1);
        assert!(duplicate.conflict().is_none());
    }

    #[test]
    fn conflicting_values_are_preserved_with_a_certificate() {
        let board = SharedMemoryBoard::new(16);
        board
            .publish(NewClaim::global("姚明", "身高", "226cm"))
            .unwrap();
        let conflict = board
            .publish(NewClaim::global("姚明", "身高", "229cm"))
            .unwrap();
        assert_eq!(conflict.status(), "conflict");
        let certificate = conflict.conflict().expect("应返回冲突证书");
        assert_eq!(certificate.values, vec!["226cm", "229cm"]);
        assert!(certificate.render().contains("2 种说法"));
        let view = board.snapshot();
        assert_eq!(view.len(), 2, "冲突不覆盖旧值");
        assert_eq!(view.conflict_count(), 1);
        assert_eq!(view.values("姚明", "身高"), vec!["226cm", "229cm"]);
    }

    #[test]
    fn session_scope_is_private_and_global_is_visible_everywhere() {
        let board = SharedMemoryBoard::new(16);
        board
            .publish(NewClaim::session("s-1", "叶莉", "身高", "190cm"))
            .unwrap();
        board
            .publish(NewClaim::global("姚明", "身高", "226cm"))
            .unwrap();

        let global = board.snapshot();
        assert_eq!(global.len(), 1);
        assert_eq!(global.claims[0].subject_key, "姚明");

        let session_one = board.snapshot_for("s-1");
        assert_eq!(session_one.len(), 2);
        assert!(session_one.values("叶莉", "身高").contains(&"190cm"));

        let session_two = board.snapshot_for("s-2");
        assert_eq!(session_two.len(), 1);
        assert!(session_two.values("叶莉", "身高").is_empty());
        assert_eq!(board.snapshot_all().len(), 2);
    }

    #[test]
    fn same_surface_in_different_scopes_does_not_conflict() {
        let board = SharedMemoryBoard::new(16);
        board
            .publish(NewClaim::session("s-1", "姚明", "身高", "226cm"))
            .unwrap();
        let second = board
            .publish(NewClaim::session("s-2", "姚明", "身高", "229cm"))
            .unwrap();
        assert_eq!(second.status(), "published");
        assert_eq!(board.snapshot_for("s-1").conflict_count(), 0);
        assert_eq!(board.snapshot_for("s-2").conflict_count(), 0);
    }

    #[test]
    fn capacity_is_bounded_and_evictions_are_reported() {
        let board = SharedMemoryBoard::new(2);
        board.publish(NewClaim::global("甲", "属性", "1")).unwrap();
        board.publish(NewClaim::global("乙", "属性", "2")).unwrap();
        let third = board
            .publish(NewClaim::global("丙", "属性", "3"))
            .unwrap();
        match third {
            PublishOutcome::Published { evicted, .. } => assert_eq!(evicted, 1),
            other => panic!("预期新增并淘汰，实际 {:?}", other),
        }
        assert_eq!(board.len(), 2);
        assert_eq!(board.evictions(), 1);
        let view = board.snapshot();
        assert!(!view.values("甲", "属性").contains(&"1"));
        assert!(view.values("丙", "属性").contains(&"3"));
    }

    #[test]
    fn duplicate_does_not_evict_under_pressure() {
        let board = SharedMemoryBoard::new(1);
        board.publish(NewClaim::global("甲", "属性", "1")).unwrap();
        let duplicate = board.publish(NewClaim::global("甲", "属性", "1")).unwrap();
        assert_eq!(duplicate.status(), "duplicate");
        assert_eq!(board.len(), 1);
        assert_eq!(board.evictions(), 0);
    }

    #[test]
    fn truncation_happens_before_identity_derivation() {
        let board = SharedMemoryBoard::new(4);
        let subject = "甲".repeat(FIELD_TEXT_LIMIT);
        let value = "值".repeat(FIELD_TEXT_LIMIT);
        board
            .publish(NewClaim::global(
                format!("{}甲", subject),
                "属性",
                format!("{}甲", value),
            ))
            .unwrap();
        let outcome = board
            .publish(NewClaim::global(
                format!("{}乙", subject),
                "属性",
                format!("{}乙", value),
            ))
            .unwrap();
        assert_eq!(outcome.status(), "duplicate");
        assert_eq!(board.len(), 1);
        let view = board.snapshot();
        assert_eq!(view.claims[0].subject.chars().count(), FIELD_TEXT_LIMIT);
        assert_eq!(view.claims[0].value.chars().count(), FIELD_TEXT_LIMIT);
        assert_eq!(view.conflict_count(), 0);
    }

    #[test]
    fn validation_rejects_empty_fields_and_private_scope_without_id() {
        let board = SharedMemoryBoard::new(4);
        assert_eq!(
            board
                .publish(NewClaim::global("", "属性", "值"))
                .unwrap_err()
                .code(),
            "empty_field"
        );
        assert_eq!(
            board
                .publish(NewClaim::session("", "主体", "属性", "值"))
                .unwrap_err()
                .code(),
            "empty_field"
        );
    }

    #[test]
    fn view_digest_is_content_addressed_not_publish_order() {
        let left = SharedMemoryBoard::new(16);
        let right = SharedMemoryBoard::new(16);
        left.publish(NewClaim::global("甲", "属性", "1")).unwrap();
        left.publish(NewClaim::global("乙", "属性", "2")).unwrap();
        right.publish(NewClaim::global("乙", "属性", "2")).unwrap();
        right.publish(NewClaim::global("甲", "属性", "1")).unwrap();
        assert_eq!(left.snapshot().digest, right.snapshot().digest);
        assert_eq!(left.snapshot().len(), right.snapshot().len());
    }

    #[test]
    fn concurrent_publishers_never_lose_updates() {
        let board = Arc::new(SharedMemoryBoard::new(4096));
        let mut handles = Vec::new();
        for worker in 0..8 {
            let board = Arc::clone(&board);
            handles.push(std::thread::spawn(move || {
                for index in 0..100 {
                    board
                        .publish(NewClaim::global(
                            format!("实体{}-{}", worker, index),
                            "属性",
                            format!("值{}", index),
                        ))
                        .expect("并发发布不应失败");
                }
            }));
        }
        for handle in handles {
            handle.join().expect("发布线程不应 panic");
        }
        assert_eq!(board.revision(), 800);
        assert_eq!(board.len(), 800);
        let view = board.snapshot();
        assert_eq!(view.len(), 800);
        assert_eq!(view.conflict_count(), 0);
    }

    #[test]
    fn concurrent_conflicts_are_all_visible() {
        let board = Arc::new(SharedMemoryBoard::new(1024));
        let mut handles = Vec::new();
        for worker in 0..8 {
            let board = Arc::clone(&board);
            handles.push(std::thread::spawn(move || {
                for _ in 0..20 {
                    board
                        .publish(NewClaim::global("姚明", "身高", format!("{}cm", worker)))
                        .expect("并发冲突发布不应失败");
                }
            }));
        }
        for handle in handles {
            handle.join().expect("发布线程不应 panic");
        }
        assert_eq!(board.len(), 8);
        let view = board.snapshot();
        assert_eq!(view.conflict_count(), 1);
        assert_eq!(view.conflicts[0].values.len(), 8);
    }

    #[test]
    fn view_roundtrips_through_json() {
        let board = SharedMemoryBoard::new(16);
        board
            .publish(NewClaim::session("s-1", "姚明", "身高", "226cm"))
            .unwrap();
        board
            .publish(NewClaim::global("姚明", "身高", "229cm"))
            .unwrap();
        let view = board.snapshot_for("s-1");
        let json = serde_json::to_string(view.as_ref()).unwrap();
        let restored: SharedMemoryView = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, *view);
    }

    #[test]
    fn session_scope_keys_cannot_prefix_collide() {
        let board = SharedMemoryBoard::new(8);
        board
            .publish(NewClaim::session("a", "姚明", "身高", "226cm"))
            .unwrap();
        board
            .publish(NewClaim::session("a\u{1}b", "姚明", "身高", "229cm"))
            .unwrap();
        assert_eq!(board.purge_session("a"), 1);
        let other = board.snapshot_for("a\u{1}b");
        assert_eq!(other.len(), 1);
        assert_eq!(other.values("姚明", "身高"), vec!["229cm"]);
    }

    #[test]
    fn retraction_is_scoped_and_revisioned() {
        let board = SharedMemoryBoard::new(8);
        board
            .publish(NewClaim::global("姚明", "身高", "226cm"))
            .unwrap();
        board
            .publish(NewClaim::session("s-1", "姚明", "国籍", "中国"))
            .unwrap();
        let global = board.snapshot().claims[0].id.clone();
        let private = board
            .snapshot_for("s-1")
            .claims
            .iter()
            .find(|claim| claim.value == "中国")
            .unwrap()
            .id
            .clone();
        let revision = board.revision();
        assert!(board
            .retract_scoped(&private, &MemoryScope::Global)
            .is_none());
        assert!(board
            .retract_scoped(&global, &MemoryScope::session("s-1"))
            .is_none());
        assert_eq!(board.len(), 2);
        assert_eq!(board.revision(), revision);
        assert_eq!(
            board
                .retract_scoped(&global, &MemoryScope::Global)
                .unwrap()
                .id,
            global
        );
        assert_eq!(board.len(), 1);
        assert_eq!(board.revision(), revision + 1);
        assert!(board
            .retract_scoped(&global, &MemoryScope::Global)
            .is_none());
        assert_eq!(board.revision(), revision + 1);
        assert_eq!(
            board
                .retract_scoped(&private, &MemoryScope::session("s-1"))
                .unwrap()
                .id,
            private
        );
        assert_eq!(board.len(), 0);
    }

    #[test]
    fn purging_a_session_removes_only_that_scope() {
        let board = SharedMemoryBoard::new(16);
        board
            .publish(NewClaim::session("s-1", "姚明", "身高", "226cm"))
            .unwrap();
        board
            .publish(NewClaim::session("s-1", "姚明", "国籍", "中国"))
            .unwrap();
        board
            .publish(NewClaim::session("s-2", "姚明", "身高", "229cm"))
            .unwrap();
        board
            .publish(NewClaim::global("叶莉", "身高", "190cm"))
            .unwrap();
        let revision = board.revision();
        assert_eq!(board.purge_session("s-1"), 2);
        assert_eq!(board.revision(), revision + 1);
        assert_eq!(board.len(), 2);
        assert!(board.snapshot_for("s-1").values("姚明", "身高").is_empty());
        assert_eq!(board.snapshot_for("s-2").values("姚明", "身高"), vec!["229cm"]);
        assert_eq!(board.snapshot().values("叶莉", "身高"), vec!["190cm"]);
        assert_eq!(board.purge_session("s-1"), 0);
        assert_eq!(board.revision(), revision + 1);
    }
}

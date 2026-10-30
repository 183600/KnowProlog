//! 知识融合账本（Knowledge Fusion Ledger）：把「来源 / 会话 / 假设」当作命名空间，
//! 在保留原始断言的前提下维护一个受审的默认视图。
//!
//! 设计立场（与实体身份注册表、谓词画像注册表互补的第三本册子）：
//!   - **原始断言与融合视图分离**：每个来源写入自己的命名图（源图），默认图只放
//!     「当前视图的赢家」；原图是证据，默认图是结论，二者都可查；
//!   - **融合是受审的合并，不是去重**：同一槽位（主语×谓词）的多个取值分成
//!     新增（new）/ 重复（duplicate）/ 佐证（corroborated）/ 冲突（conflict）四类；
//!   - **冲突默认隔离**：功能谓词的少数派取值不静默覆盖、也不静默丢弃，而是
//!     进入隔离区（quarantine），由信任层级排序选出赢家；
//!   - **信任是序数，不是概率**：`verified > curated > imported > external >
//!     speculative` 只用于裁决排序；同层用命名空间优先级、id 字典序稳定破平；
//!   - **裁决即对齐**：用户对冲突的裁决（`Adjudication`）写成显式偏好记录，
//!     只影响声明的谓词槽位，可审计、可撤销，且不训练任何权重；
//!   - **确定性**：条目 id 由规范字符串哈希（FNV-1a）得出，全部遍历按字典序，
//!     与插入顺序无关，报告可逐字节断言；
//!   - **账本即 RDF 命名图**：命名空间、断言、裁决与已物化赢家都落在
//!     `kg:graphs/fusion`，可用 SPARQL 直接审计。

use crate::context_image::short_hash;
use std::cmp::{Ordering, Reverse};
use std::collections::{BTreeMap, BTreeSet};

use oxigraph::model::*;
use serde::{Deserialize, Serialize};

/// 融合账本命名图（命名空间 / 断言 / 裁决 / 已物化赢家）。
pub const FUSION_GRAPH: &str = "http://example.org/kg/graphs/fusion";
/// 源图 IRI 前缀：`{前缀}{规范化命名空间 id}`，每个来源一张独立命名图。
pub const SOURCE_GRAPH_PREFIX: &str = "http://example.org/kg/graphs/source/";
/// 融合账本专用命名空间（与知识图谱 `kg:` 词面隔离）。
pub const FUSION_NS: &str = "http://example.org/kg/fusion/";

const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
const NAMESPACE_CLASS: &str = "http://example.org/kg/fusion/Namespace";
const ASSERTION_CLASS: &str = "http://example.org/kg/fusion/Assertion";
const ADJUDICATION_CLASS: &str = "http://example.org/kg/fusion/Adjudication";
const APPLIED_CLASS: &str = "http://example.org/kg/fusion/Applied";
const STAMP_CLASS: &str = "http://example.org/kg/fusion/Stamp";
const STAMP_SUBJECT: &str = "http://example.org/kg/fusion/stamp";

const P_JSON: &str = "http://example.org/kg/fusion/json";
const P_LABEL: &str = "http://example.org/kg/fusion/label";
const P_KIND: &str = "http://example.org/kg/fusion/kind";
const P_TRUST: &str = "http://example.org/kg/fusion/trust";
const P_STATUS: &str = "http://example.org/kg/fusion/status";
const P_PRIORITY: &str = "http://example.org/kg/fusion/priority";
const P_NAMESPACE: &str = "http://example.org/kg/fusion/namespace";
const P_SUBJECT: &str = "http://example.org/kg/fusion/subject";
const P_PREDICATE: &str = "http://example.org/kg/fusion/predicate";
const P_OBJECT: &str = "http://example.org/kg/fusion/object";
const P_SOURCE: &str = "http://example.org/kg/fusion/source";
const P_PREFER: &str = "http://example.org/kg/fusion/prefer";
const P_RATIONALE: &str = "http://example.org/kg/fusion/rationale";
const P_ORIGIN: &str = "http://example.org/kg/fusion/origin";
const P_ASSERTION: &str = "http://example.org/kg/fusion/assertion";
const P_COUNT: &str = "http://example.org/kg/fusion/count";
const P_NAMESPACES: &str = "http://example.org/kg/fusion/namespaces";
const P_ASSERTIONS: &str = "http://example.org/kg/fusion/assertions";
const P_ADJUDICATIONS: &str = "http://example.org/kg/fusion/adjudications";
const P_APPLIED: &str = "http://example.org/kg/fusion/applied";

type FusionResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

// ---------------------------------------------------------------------------
// 信任层级 / 命名空间
// ---------------------------------------------------------------------------

/// 信任层级：序数而非概率，只用于冲突裁决排序。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum TrustLevel {
    /// 已由人确认（最高）
    Verified,
    /// 受管知识库 / 人工整理
    Curated,
    /// 批量导入
    Imported,
    /// 外部服务缓存
    External,
    /// 推测 / 假设（最低）
    Speculative,
}

impl TrustLevel {
    /// 全部层级（强 → 弱）。
    pub const ALL: [TrustLevel; 5] = [
        TrustLevel::Verified,
        TrustLevel::Curated,
        TrustLevel::Imported,
        TrustLevel::External,
        TrustLevel::Speculative,
    ];

    /// 序数：越大越可信。
    pub fn rank(&self) -> u8 {
        match self {
            TrustLevel::Verified => 4,
            TrustLevel::Curated => 3,
            TrustLevel::Imported => 2,
            TrustLevel::External => 1,
            TrustLevel::Speculative => 0,
        }
    }

    pub fn label_zh(&self) -> &'static str {
        match self {
            TrustLevel::Verified => "已验证",
            TrustLevel::Curated => "受管",
            TrustLevel::Imported => "导入",
            TrustLevel::External => "外部",
            TrustLevel::Speculative => "推测",
        }
    }

    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "verified" | "已验证" => Some(TrustLevel::Verified),
            "curated" | "受管" => Some(TrustLevel::Curated),
            "imported" | "导入" => Some(TrustLevel::Imported),
            "external" | "外部" => Some(TrustLevel::External),
            "speculative" | "推测" => Some(TrustLevel::Speculative),
            _ => None,
        }
    }
}

/// 命名空间种类：来源 / 会话 / 假设的分层依据。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum NamespaceKind {
    /// 全局知识库（默认）
    Global,
    /// 一次导入批次 / 一个来源家族
    Import,
    /// 多轮对话的会话命名空间
    Session,
    /// 假设 / 反事实推理的作用域
    Hypothesis,
}

impl NamespaceKind {
    pub fn label_zh(&self) -> &'static str {
        match self {
            NamespaceKind::Global => "全局",
            NamespaceKind::Import => "导入",
            NamespaceKind::Session => "会话",
            NamespaceKind::Hypothesis => "假设",
        }
    }

    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "global" | "全局" => Some(NamespaceKind::Global),
            "import" | "导入" => Some(NamespaceKind::Import),
            "session" | "会话" => Some(NamespaceKind::Session),
            "hypothesis" | "假设" => Some(NamespaceKind::Hypothesis),
            _ => None,
        }
    }

    /// 默认优先级：信任层级相同时的破平依据（假设 > 会话 > 导入 > 全局）。
    pub fn default_priority(&self) -> i64 {
        match self {
            NamespaceKind::Hypothesis => 30,
            NamespaceKind::Session => 20,
            NamespaceKind::Import => 10,
            NamespaceKind::Global => 0,
        }
    }
}

/// 命名空间状态：启用 / 隔离（保留证据但不参与视图）/ 退役。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum NamespaceStatus {
    Active,
    Quarantined,
    Retired,
}

impl NamespaceStatus {
    pub fn label_zh(&self) -> &'static str {
        match self {
            NamespaceStatus::Active => "启用",
            NamespaceStatus::Quarantined => "隔离",
            NamespaceStatus::Retired => "退役",
        }
    }

    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "active" | "启用" => Some(NamespaceStatus::Active),
            "quarantined" | "隔离" => Some(NamespaceStatus::Quarantined),
            "retired" | "退役" => Some(NamespaceStatus::Retired),
            _ => None,
        }
    }
}

/// 命名空间记录：一个来源 / 会话 / 假设的全部元数据。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NamespaceRecord {
    /// 规范化 id（IRI 安全，见 `sanitize_namespace_id`）。
    pub id: String,
    /// 人类可读来源标签。
    pub label: String,
    pub kind: NamespaceKind,
    pub trust: TrustLevel,
    pub status: NamespaceStatus,
    /// 同信任层内的破平优先级（大者优先）。
    pub priority: i64,
}

impl NamespaceRecord {
    pub fn new(id: &str, label: &str, kind: NamespaceKind, trust: TrustLevel) -> Self {
        Self {
            id: sanitize_namespace_id(id),
            label: label.to_string(),
            kind,
            trust,
            status: NamespaceStatus::Active,
            priority: kind.default_priority(),
        }
    }

    pub fn with_priority(mut self, priority: i64) -> Self {
        self.priority = priority;
        self
    }

    pub fn with_status(mut self, status: NamespaceStatus) -> Self {
        self.status = status;
        self
    }

    /// 排序键：信任降序 → 优先级降序 → id 升序（`Ord` 越小越优先）。
    pub fn precedence(&self) -> (Reverse<u8>, Reverse<i64>, &str) {
        (
            Reverse(self.trust.rank()),
            Reverse(self.priority),
            self.id.as_str(),
        )
    }

    /// 是否参与默认融合视图。
    pub fn participates(&self, include_quarantined: bool) -> bool {
        match self.status {
            NamespaceStatus::Active => true,
            NamespaceStatus::Quarantined => include_quarantined,
            NamespaceStatus::Retired => false,
        }
    }

    /// 该命名空间的源图 IRI。
    pub fn graph_iri(&self) -> String {
        source_graph_iri(&self.id)
    }
}

/// 把任意来源 id 规范化为 IRI 安全 slug（幂等）。
///
/// 允许 `[A-Za-z0-9._-]`；其余字符替换为 `_`，若发生过替换则追加稳定短哈希，
/// 避免「a b」与「a_b」碰撞；超长 id 截断后同样追加哈希。
pub fn sanitize_namespace_id(raw: &str) -> String {
    let trimmed = raw.trim();
    let mut out = String::with_capacity(trimmed.len());
    let mut replaced = false;
    for ch in trimmed.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
            out.push(ch);
        } else {
            out.push('_');
            replaced = true;
        }
    }
    if out.is_empty() {
        return "source".to_string();
    }
    if out.len() > 64 {
        let hash = short_hash(trimmed);
        out.truncate(48);
        out.push('-');
        out.push_str(&hash[..8]);
    } else if replaced {
        let hash = short_hash(trimmed);
        out.push('-');
        out.push_str(&hash[..8]);
    }
    out
}

/// 源图 IRI：`{前缀}{规范化 id}`。
pub fn source_graph_iri(namespace: &str) -> String {
    format!(
        "{}{}",
        SOURCE_GRAPH_PREFIX,
        sanitize_namespace_id(namespace)
    )
}

// ---------------------------------------------------------------------------
// 断言 / 裁决
// ---------------------------------------------------------------------------

/// 宾语种类：字面量还是实体引用（决定写回 RDF 时用 Literal 还是 NamedNode）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ObjectKind {
    Literal,
    Entity,
}

impl Default for ObjectKind {
    fn default() -> Self {
        ObjectKind::Literal
    }
}

impl ObjectKind {
    pub fn tag(&self) -> &'static str {
        match self {
            ObjectKind::Literal => "literal",
            ObjectKind::Entity => "entity",
        }
    }

    pub fn label_zh(&self) -> &'static str {
        match self {
            ObjectKind::Literal => "字面量",
            ObjectKind::Entity => "实体",
        }
    }
}

/// 带来源作用域的断言：`(命名空间, 主语, 谓词, 宾语)`。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScopedAssertion {
    /// 稳定 id（命名空间 + 四元组哈希）。
    pub id: String,
    pub namespace: String,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    #[serde(default)]
    pub object_kind: ObjectKind,
    /// 证据指针：文件路径、对话轮次、URL 等。
    pub source: String,
}

impl ScopedAssertion {
    pub fn new(
        namespace: &str,
        subject: &str,
        predicate: &str,
        object: &str,
        object_kind: ObjectKind,
        source: &str,
    ) -> Self {
        let namespace = sanitize_namespace_id(namespace);
        let id = assertion_id(&namespace, subject, predicate, object, object_kind);
        Self {
            id,
            namespace,
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
            object_kind,
            source: source.to_string(),
        }
    }

    /// 槽位键：同一 (主语, 谓词) 的取值互相竞争 / 佐证。
    pub fn slot(&self) -> (&str, &str) {
        (self.subject.as_str(), self.predicate.as_str())
    }

    /// 事实键：`(主语, 谓词, 宾语)`。
    pub fn key(&self) -> (&str, &str, &str) {
        (
            self.subject.as_str(),
            self.predicate.as_str(),
            self.object.as_str(),
        )
    }
}

/// 断言的稳定 id。
pub fn assertion_id(
    namespace: &str,
    subject: &str,
    predicate: &str,
    object: &str,
    object_kind: ObjectKind,
) -> String {
    short_hash(&format!(
        "{}|{}|{}|{}|{}",
        sanitize_namespace_id(namespace),
        subject,
        predicate,
        object_kind.tag(),
        object
    ))
}

/// 显式裁决：在某个谓词上优先某个命名空间（H4：裁决即对齐）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Adjudication {
    /// 稳定 id。
    pub id: String,
    /// 作用域：仅此谓词。
    pub predicate: String,
    /// 优先采用的命名空间 id。
    pub prefer_namespace: String,
    /// 裁决理由（可审计）。
    pub rationale: String,
    /// 裁决来源：`user`（用户）/ `policy`（策略）等。
    pub origin: String,
}

impl Adjudication {
    pub fn new(predicate: &str, prefer_namespace: &str, rationale: &str, origin: &str) -> Self {
        let prefer_namespace = sanitize_namespace_id(prefer_namespace);
        let id = adjudication_id(predicate, &prefer_namespace);
        Self {
            id,
            predicate: predicate.to_string(),
            prefer_namespace,
            rationale: rationale.to_string(),
            origin: origin.to_string(),
        }
    }
}

/// 裁决的稳定 id。
pub fn adjudication_id(predicate: &str, prefer_namespace: &str) -> String {
    short_hash(&format!(
        "{}|{}",
        predicate,
        sanitize_namespace_id(prefer_namespace)
    ))
}

// ---------------------------------------------------------------------------
// 融合策略 / 结果
// ---------------------------------------------------------------------------

/// 融合策略：哪些谓词按功能谓词处理、是否把「隔离」命名空间也纳入视图。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FusionPolicy {
    functional: BTreeSet<String>,
    include_quarantined: bool,
}

impl FusionPolicy {
    pub fn new() -> Self {
        Self::default()
    }

    /// 追加功能谓词（同槽位只允许一个取值，冲突进入隔离区）。
    pub fn with_functional<I, S>(mut self, predicates: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        for predicate in predicates {
            self.functional.insert(predicate.into());
        }
        self
    }

    pub fn include_quarantined(mut self, include: bool) -> Self {
        self.include_quarantined = include;
        self
    }

    pub fn is_functional(&self, predicate: &str) -> bool {
        self.functional.contains(predicate)
    }

    pub fn functional_predicates(&self) -> Vec<&str> {
        self.functional.iter().map(String::as_str).collect()
    }
}

/// 融合分类：新增 / 重复 / 佐证 / 冲突。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum FusionDelta {
    /// 视图里有新事实待物化
    New,
    /// 默认图已有全部取值
    Duplicate,
    /// 多来源一致（同值 ≥ 2 个命名空间）
    Corroborated,
    /// 功能谓词出现分歧（少数派隔离）
    Conflict,
}

impl FusionDelta {
    pub fn label_zh(&self) -> &'static str {
        match self {
            FusionDelta::New => "新增",
            FusionDelta::Duplicate => "重复",
            FusionDelta::Corroborated => "佐证",
            FusionDelta::Conflict => "冲突",
        }
    }
}

/// 一个取值簇：同一 (主语, 谓词, 宾语) 的全部支撑断言。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValueCluster {
    pub object: String,
    pub object_kind: ObjectKind,
    /// 支撑断言（按命名空间优先级排序，best 在前）。
    pub supporters: Vec<ScopedAssertion>,
    /// 代表断言（最高优先级支撑者）。
    pub best: ScopedAssertion,
    /// 是否多来源一致（≥ 2 个不同命名空间）。
    pub corroborated: bool,
    /// 默认图是否已有该事实。
    pub present: bool,
}

/// 槽位裁决结果：`(主语, 谓词)` 的接受值与隔离值。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlotOutcome {
    pub subject: String,
    pub predicate: String,
    /// 是否按功能谓词处理。
    pub functional: bool,
    pub delta: FusionDelta,
    /// 接受簇内是否存在多源佐证。
    pub corroborated: bool,
    /// 进入默认视图的取值簇（功能谓词只有一个；否则全部接受）。
    pub accepted: Vec<ValueCluster>,
    /// 被隔离的少数派取值簇（默认策略下不入默认图，但保留证据）。
    pub quarantined: Vec<ValueCluster>,
    /// 生效的裁决 id（若由裁决而非信任排序决定赢家）。
    pub adjudicated_by: Option<String>,
}

impl SlotOutcome {
    /// 人类可读槽位名：`小明 的 年龄`。
    pub fn slot_label(&self) -> String {
        format!("{} 的 {}", self.subject, self.predicate)
    }

    /// 某个取值簇的支撑来源列表（去重、字典序）。
    pub fn sources_of(cluster: &ValueCluster) -> Vec<String> {
        cluster
            .supporters
            .iter()
            .map(|assertion| assertion.namespace.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }
}

/// 融合统计。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FusionStats {
    /// 参与的命名空间数
    pub namespaces: usize,
    /// 参与判定的断言数（未知命名空间的孤儿不计）
    pub assertions: usize,
    /// 引用了未知 / 非启用命名空间的断言数
    pub orphans: usize,
    pub slots: usize,
    pub new: usize,
    pub duplicate: usize,
    pub corroborated: usize,
    pub conflict: usize,
    /// 被隔离的取值数（不是断言数）
    pub quarantined: usize,
}

/// 融合计划：受审视图的完整说明 + 待物化断言。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FusionPlan {
    /// 全部槽位（按主语、谓词字典序）。
    pub slots: Vec<SlotOutcome>,
    /// 待物化到默认图的断言（每个接受簇一条代表）。
    pub materialize: Vec<ScopedAssertion>,
    pub stats: FusionStats,
}

impl FusionPlan {
    /// 冲突槽位。
    pub fn conflicts(&self) -> Vec<&SlotOutcome> {
        self.slots
            .iter()
            .filter(|slot| slot.delta == FusionDelta::Conflict)
            .collect()
    }

    /// 待物化事实键集合（供写回时判重）。
    pub fn materialize_keys(&self) -> BTreeSet<(String, String, String, ObjectKind)> {
        self.materialize
            .iter()
            .map(|assertion| {
                (
                    assertion.subject.clone(),
                    assertion.predicate.clone(),
                    assertion.object.clone(),
                    assertion.object_kind,
                )
            })
            .collect()
    }

    /// 确定性中文报告：计数 + 冲突明细 + 待物化条数。
    pub fn render_zh(&self) -> String {
        let stats = &self.stats;
        let mut lines = vec![format!(
            "知识融合视图：{} 个命名空间 / {} 条断言（孤儿 {}）",
            stats.namespaces, stats.assertions, stats.orphans
        )];
        lines.push(format!(
            "  槽位 {}（新增 {} / 重复 {} / 佐证 {} / 冲突 {}），隔离取值 {}",
            stats.slots,
            stats.new,
            stats.duplicate,
            stats.corroborated,
            stats.conflict,
            stats.quarantined
        ));
        for slot in self.conflicts() {
            let accepted = slot
                .accepted
                .first()
                .map(|cluster| {
                    let sources = SlotOutcome::sources_of(cluster).join("、");
                    format!("采用「{}」（来源 {}）", cluster.object, sources)
                })
                .unwrap_or_else(|| "无接受值".to_string());
            let via = slot
                .adjudicated_by
                .as_ref()
                .map(|id| format!("，依据裁决 {}", id))
                .unwrap_or_default();
            lines.push(format!("  ⚔ {}：{}{}", slot.slot_label(), accepted, via));
            for cluster in &slot.quarantined {
                let sources = SlotOutcome::sources_of(cluster).join("、");
                lines.push(format!(
                    "      ↳ 隔离「{}」（来源 {}）",
                    cluster.object, sources
                ));
            }
        }
        lines.push(format!("待物化 {} 条断言", self.materialize.len()));
        lines.join("\n")
    }
}

// ---------------------------------------------------------------------------
// 融合算法（纯函数、确定性）
// ---------------------------------------------------------------------------

/// 按策略计算融合计划。
///
/// - `exists(主语, 谓词, 宾语, 宾语种类)`：默认图是否已有该事实（由调用方决定
///   查询方式，模块本身不接触存储）；
/// - 功能谓词的赢家选择顺序：显式裁决 → 信任层级 → 命名空间优先级 → id；
/// - 非功能谓词接受全部取值（多值事实），裁决不影响接受集合；
/// - 引用了未知 / 退役命名空间的断言计入 `orphans`，不参与视图。
pub fn fusion_plan<F>(
    namespaces: &[NamespaceRecord],
    assertions: &[ScopedAssertion],
    adjudications: &[Adjudication],
    policy: &FusionPolicy,
    exists: F,
) -> FusionPlan
where
    F: Fn(&str, &str, &str, ObjectKind) -> bool,
{
    let ns_map: BTreeMap<&str, &NamespaceRecord> = namespaces
        .iter()
        .filter(|record| record.participates(policy.include_quarantined))
        .map(|record| (record.id.as_str(), record))
        .collect();

    // 裁决按 id 排序后每个谓词只保留最小 id 的一条（确定性、与输入顺序无关）。
    let mut adjudication_list: Vec<&Adjudication> = adjudications.iter().collect();
    adjudication_list.sort_by(|a, b| a.id.cmp(&b.id));
    let mut adjud_by_predicate: BTreeMap<&str, &Adjudication> = BTreeMap::new();
    for adjudication in adjudication_list {
        adjud_by_predicate
            .entry(adjudication.predicate.as_str())
            .or_insert(adjudication);
    }

    let mut seen_assertions: BTreeSet<&str> = BTreeSet::new();
    let mut groups: BTreeMap<(&str, &str), Vec<&ScopedAssertion>> = BTreeMap::new();
    let mut stats = FusionStats {
        namespaces: ns_map.len(),
        ..FusionStats::default()
    };
    for assertion in assertions {
        if !seen_assertions.insert(assertion.id.as_str()) {
            continue; // 同一 id 的重复记录只计一次
        }
        if ns_map.contains_key(assertion.namespace.as_str()) {
            groups.entry(assertion.slot()).or_default().push(assertion);
        } else {
            stats.orphans += 1;
        }
    }

    let mut slots: Vec<SlotOutcome> = Vec::with_capacity(groups.len());
    let mut materialize: Vec<ScopedAssertion> = Vec::new();
    for ((subject, predicate), group) in groups {
        stats.assertions += group.len();
        stats.slots += 1;

        // 取值分簇
        let mut by_object: BTreeMap<&str, Vec<&ScopedAssertion>> = BTreeMap::new();
        for assertion in group {
            by_object
                .entry(assertion.object.as_str())
                .or_default()
                .push(assertion);
        }
        let mut clusters: Vec<ValueCluster> = by_object
            .into_iter()
            .map(|(object, mut supporters)| {
                supporters.sort_by(|a, b| {
                    compare_assertions(a, b, &ns_map).then_with(|| a.id.cmp(&b.id))
                });
                let distinct: BTreeSet<&str> = supporters
                    .iter()
                    .map(|assertion| assertion.namespace.as_str())
                    .collect();
                let best = (*supporters[0]).clone();
                let present = exists(
                    &best.subject,
                    &best.predicate,
                    &best.object,
                    best.object_kind,
                );
                ValueCluster {
                    object: object.to_string(),
                    object_kind: best.object_kind,
                    supporters: supporters
                        .iter()
                        .map(|assertion| (*assertion).clone())
                        .collect(),
                    best,
                    corroborated: distinct.len() >= 2,
                    present,
                }
            })
            .collect();
        clusters.sort_by(|a, b| {
            compare_assertions(&a.best, &b.best, &ns_map).then_with(|| a.object.cmp(&b.object))
        });

        let functional = policy.is_functional(predicate);
        let mut accepted: Vec<ValueCluster> = Vec::new();
        let mut quarantined: Vec<ValueCluster> = Vec::new();
        let mut adjudicated_by: Option<String> = None;
        if functional {
            let mut winner_index = 0usize;
            if let Some(adjudication) = adjud_by_predicate.get(predicate) {
                if let Some(index) = clusters.iter().position(|cluster| {
                    cluster
                        .supporters
                        .iter()
                        .any(|assertion| assertion.namespace == adjudication.prefer_namespace)
                }) {
                    winner_index = index;
                    adjudicated_by = Some(adjudication.id.clone());
                }
            }
            for (index, cluster) in clusters.into_iter().enumerate() {
                if index == winner_index {
                    accepted.push(cluster);
                } else {
                    quarantined.push(cluster);
                }
            }
        } else {
            accepted = clusters;
        }

        let delta = if !quarantined.is_empty() {
            FusionDelta::Conflict
        } else if accepted.iter().any(|cluster| cluster.corroborated) {
            FusionDelta::Corroborated
        } else if accepted.iter().all(|cluster| cluster.present) {
            FusionDelta::Duplicate
        } else {
            FusionDelta::New
        };
        match delta {
            FusionDelta::New => stats.new += 1,
            FusionDelta::Duplicate => stats.duplicate += 1,
            FusionDelta::Corroborated => stats.corroborated += 1,
            FusionDelta::Conflict => stats.conflict += 1,
        }
        stats.quarantined += quarantined.len();
        let corroborated = accepted.iter().any(|cluster| cluster.corroborated);
        for cluster in &accepted {
            materialize.push(cluster.best.clone());
        }
        slots.push(SlotOutcome {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            functional,
            delta,
            corroborated,
            accepted,
            quarantined,
            adjudicated_by,
        });
    }

    FusionPlan {
        slots,
        materialize,
        stats,
    }
}

/// 断言优先级比较：命名空间优先级 → 断言 id。
fn compare_assertions(
    a: &ScopedAssertion,
    b: &ScopedAssertion,
    ns_map: &BTreeMap<&str, &NamespaceRecord>,
) -> Ordering {
    match (
        ns_map.get(a.namespace.as_str()),
        ns_map.get(b.namespace.as_str()),
    ) {
        (Some(left), Some(right)) => left.precedence().cmp(&right.precedence()),
        _ => a.namespace.cmp(&b.namespace),
    }
}

// ---------------------------------------------------------------------------
// 账本：命名空间 / 断言 / 裁决 / 已物化赢家
// ---------------------------------------------------------------------------

/// 已物化到默认图的赢家（用于下次融合时精确回收被替换的旧赢家）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AppliedWinner {
    /// 对应断言 id。
    pub id: String,
    pub namespace: String,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    #[serde(default)]
    pub object_kind: ObjectKind,
}

impl AppliedWinner {
    pub fn from_assertion(assertion: &ScopedAssertion) -> Self {
        Self {
            id: assertion.id.clone(),
            namespace: assertion.namespace.clone(),
            subject: assertion.subject.clone(),
            predicate: assertion.predicate.clone(),
            object: assertion.object.clone(),
            object_kind: assertion.object_kind,
        }
    }

    /// 事实键。
    pub fn key(&self) -> (&str, &str, &str, ObjectKind) {
        (
            self.subject.as_str(),
            self.predicate.as_str(),
            self.object.as_str(),
            self.object_kind,
        )
    }
}

/// 融合账本：全部命名空间、断言的持久台账 + 已物化赢家。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FusionLedger {
    /// 账本是否写过（有账本戳或条目）。
    pub synced: bool,
    pub namespaces: Vec<NamespaceRecord>,
    pub assertions: Vec<ScopedAssertion>,
    pub adjudications: Vec<Adjudication>,
    pub applied: Vec<AppliedWinner>,
}

impl FusionLedger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn namespace(&self, id: &str) -> Option<&NamespaceRecord> {
        self.namespaces
            .iter()
            .find(|record| record.id == sanitize_namespace_id(id))
    }

    /// 新增 / 覆盖命名空间（按 id）。
    pub fn upsert_namespace(&mut self, record: NamespaceRecord) {
        self.namespaces.retain(|existing| existing.id != record.id);
        self.namespaces.push(record);
    }

    /// 新增 / 覆盖断言（按 id）。
    pub fn record_assertion(&mut self, assertion: ScopedAssertion) {
        self.assertions
            .retain(|existing| existing.id != assertion.id);
        self.assertions.push(assertion);
    }

    /// 记录裁决：同一谓词只保留一条（后写覆盖）。
    pub fn record_adjudication(&mut self, adjudication: Adjudication) {
        self.adjudications
            .retain(|existing| existing.predicate != adjudication.predicate);
        self.adjudications.push(adjudication);
    }

    pub fn set_applied(&mut self, winners: Vec<AppliedWinner>) {
        self.applied = winners;
    }

    /// 排序 + 去重：保证账本加载 / 保存的确定性。
    pub fn sort_and_dedup(&mut self) {
        self.namespaces.sort_by(|a, b| a.id.cmp(&b.id));
        self.namespaces.dedup_by(|a, b| a.id == b.id);
        self.assertions.sort_by(|a, b| a.id.cmp(&b.id));
        self.assertions.dedup_by(|a, b| a.id == b.id);
        self.adjudications
            .sort_by(|a, b| a.predicate.cmp(&b.predicate).then_with(|| a.id.cmp(&b.id)));
        self.adjudications
            .dedup_by(|a, b| a.predicate == b.predicate);
        self.applied.sort_by(|a, b| a.id.cmp(&b.id));
        self.applied.dedup_by(|a, b| a.id == b.id);
    }

    /// 按命名空间的断言计数（字典序）。
    pub fn assertions_by_namespace(&self) -> BTreeMap<String, usize> {
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for assertion in &self.assertions {
            *counts.entry(assertion.namespace.clone()).or_insert(0) += 1;
        }
        counts
    }

    /// 确定性中文报告：命名空间、断言数、裁决、已物化。
    pub fn render_zh(&self) -> String {
        let counts = self.assertions_by_namespace();
        let mut lines = vec![format!(
            "知识融合账本：命名空间 {} / 断言 {} / 裁决 {} / 已物化 {}",
            self.namespaces.len(),
            self.assertions.len(),
            self.adjudications.len(),
            self.applied.len()
        )];
        for record in &self.namespaces {
            lines.push(format!(
                "  • {}（{}，{}，{}，优先级 {}）：{} 条断言",
                record.id,
                record.label,
                record.trust.label_zh(),
                record.status.label_zh(),
                record.priority,
                counts.get(&record.id).copied().unwrap_or(0)
            ));
        }
        for adjudication in &self.adjudications {
            lines.push(format!(
                "  裁决：{} 优先来源 {}（{}）",
                adjudication.predicate, adjudication.prefer_namespace, adjudication.origin
            ));
        }
        lines.join("\n")
    }
}

/// 融合执行结果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusionApplyOutcome {
    pub plan: FusionPlan,
    pub inserted: usize,
    pub removed: usize,
}

impl FusionApplyOutcome {
    /// 默认图是否发生实际变化。
    pub fn changed(&self) -> bool {
        self.inserted + self.removed > 0
    }
}

// ---------------------------------------------------------------------------
// RDF 往返
// ---------------------------------------------------------------------------

fn named_node(iri: &str) -> FusionResult<NamedNode> {
    Ok(NamedNode::new(iri)?)
}

fn literal_quad(
    subject: &NamedNode,
    predicate: &str,
    value: &str,
    graph: &GraphName,
) -> FusionResult<Quad> {
    Ok(Quad::new(
        subject.clone(),
        named_node(predicate)?,
        Literal::new_simple_literal(value),
        graph.clone(),
    ))
}

fn class_quad(subject: &NamedNode, class: &str, graph: &GraphName) -> FusionResult<Quad> {
    Ok(Quad::new(
        subject.clone(),
        named_node(RDF_TYPE)?,
        Term::NamedNode(named_node(class)?),
        graph.clone(),
    ))
}

fn namespace_node(id: &str) -> FusionResult<NamedNode> {
    named_node(&format!(
        "{}namespace/{}",
        FUSION_NS,
        sanitize_namespace_id(id)
    ))
}

fn assertion_node(id: &str) -> FusionResult<NamedNode> {
    named_node(&format!("{}assertion/{}", FUSION_NS, id))
}

fn adjudication_node(id: &str) -> FusionResult<NamedNode> {
    named_node(&format!("{}adjudication/{}", FUSION_NS, id))
}

fn applied_node(id: &str) -> FusionResult<NamedNode> {
    named_node(&format!("{}applied/{}", FUSION_NS, id))
}

/// 生成融合账本的全部四元组（含账本戳），写入 `kg:graphs/fusion`。
pub fn fusion_quads(ledger: &FusionLedger) -> FusionResult<Vec<Quad>> {
    let graph = GraphName::NamedNode(named_node(FUSION_GRAPH)?);
    let mut quads: Vec<Quad> = Vec::with_capacity(
        ledger.namespaces.len() * 7
            + ledger.assertions.len() * 8
            + ledger.adjudications.len() * 6
            + ledger.applied.len() * 6
            + 6,
    );
    for record in &ledger.namespaces {
        let subject = namespace_node(&record.id)?;
        quads.push(class_quad(&subject, NAMESPACE_CLASS, &graph)?);
        quads.push(literal_quad(
            &subject,
            P_JSON,
            &serde_json::to_string(record)?,
            &graph,
        )?);
        quads.push(literal_quad(&subject, P_LABEL, &record.label, &graph)?);
        quads.push(literal_quad(
            &subject,
            P_KIND,
            record.kind.label_zh(),
            &graph,
        )?);
        quads.push(literal_quad(
            &subject,
            P_TRUST,
            record.trust.label_zh(),
            &graph,
        )?);
        quads.push(literal_quad(
            &subject,
            P_STATUS,
            record.status.label_zh(),
            &graph,
        )?);
        quads.push(literal_quad(
            &subject,
            P_PRIORITY,
            &record.priority.to_string(),
            &graph,
        )?);
    }
    for assertion in &ledger.assertions {
        let subject = assertion_node(&assertion.id)?;
        quads.push(class_quad(&subject, ASSERTION_CLASS, &graph)?);
        quads.push(literal_quad(
            &subject,
            P_JSON,
            &serde_json::to_string(assertion)?,
            &graph,
        )?);
        quads.push(literal_quad(
            &subject,
            P_NAMESPACE,
            &assertion.namespace,
            &graph,
        )?);
        quads.push(literal_quad(
            &subject,
            P_SUBJECT,
            &assertion.subject,
            &graph,
        )?);
        quads.push(literal_quad(
            &subject,
            P_PREDICATE,
            &assertion.predicate,
            &graph,
        )?);
        quads.push(literal_quad(&subject, P_OBJECT, &assertion.object, &graph)?);
        quads.push(literal_quad(
            &subject,
            P_KIND,
            assertion.object_kind.label_zh(),
            &graph,
        )?);
        quads.push(literal_quad(&subject, P_SOURCE, &assertion.source, &graph)?);
    }
    for adjudication in &ledger.adjudications {
        let subject = adjudication_node(&adjudication.id)?;
        quads.push(class_quad(&subject, ADJUDICATION_CLASS, &graph)?);
        quads.push(literal_quad(
            &subject,
            P_JSON,
            &serde_json::to_string(adjudication)?,
            &graph,
        )?);
        quads.push(literal_quad(
            &subject,
            P_PREDICATE,
            &adjudication.predicate,
            &graph,
        )?);
        quads.push(literal_quad(
            &subject,
            P_PREFER,
            &adjudication.prefer_namespace,
            &graph,
        )?);
        quads.push(literal_quad(
            &subject,
            P_RATIONALE,
            &adjudication.rationale,
            &graph,
        )?);
        quads.push(literal_quad(
            &subject,
            P_ORIGIN,
            &adjudication.origin,
            &graph,
        )?);
    }
    for winner in &ledger.applied {
        let subject = applied_node(&winner.id)?;
        quads.push(class_quad(&subject, APPLIED_CLASS, &graph)?);
        quads.push(literal_quad(
            &subject,
            P_JSON,
            &serde_json::to_string(winner)?,
            &graph,
        )?);
        quads.push(literal_quad(&subject, P_ASSERTION, &winner.id, &graph)?);
        quads.push(literal_quad(
            &subject,
            P_NAMESPACE,
            &winner.namespace,
            &graph,
        )?);
        quads.push(literal_quad(&subject, P_SUBJECT, &winner.subject, &graph)?);
        quads.push(literal_quad(
            &subject,
            P_PREDICATE,
            &winner.predicate,
            &graph,
        )?);
        quads.push(literal_quad(&subject, P_OBJECT, &winner.object, &graph)?);
    }
    let stamp = named_node(STAMP_SUBJECT)?;
    quads.push(class_quad(&stamp, STAMP_CLASS, &graph)?);
    quads.push(literal_quad(
        &stamp,
        P_NAMESPACES,
        &ledger.namespaces.len().to_string(),
        &graph,
    )?);
    quads.push(literal_quad(
        &stamp,
        P_ASSERTIONS,
        &ledger.assertions.len().to_string(),
        &graph,
    )?);
    quads.push(literal_quad(
        &stamp,
        P_ADJUDICATIONS,
        &ledger.adjudications.len().to_string(),
        &graph,
    )?);
    quads.push(literal_quad(
        &stamp,
        P_APPLIED,
        &ledger.applied.len().to_string(),
        &graph,
    )?);
    quads.push(literal_quad(
        &stamp,
        P_COUNT,
        &(ledger.namespaces.len()
            + ledger.assertions.len()
            + ledger.adjudications.len()
            + ledger.applied.len())
        .to_string(),
        &graph,
    )?);
    Ok(quads)
}

/// 从 `kg:graphs/fusion` 的四元组还原账本（按 id 排序去重，确定性）。
pub fn parse_fusion_graph(quads: impl Iterator<Item = Quad>) -> FusionLedger {
    let quads: Vec<Quad> = quads.collect();
    // 先收集 rdf:type，JSON 负载按类型分派解析（避免形态相近的 JSON 互串）
    let mut classes: BTreeMap<String, String> = BTreeMap::new();
    for quad in &quads {
        if quad.predicate.as_str() != RDF_TYPE {
            continue;
        }
        if let (Subject::NamedNode(subject), Term::NamedNode(class)) = (&quad.subject, &quad.object)
        {
            classes.insert(subject.as_str().to_string(), class.as_str().to_string());
        }
    }
    let mut ledger = FusionLedger::default();
    for quad in &quads {
        if let Subject::NamedNode(node) = &quad.subject {
            if node.as_str() == STAMP_SUBJECT {
                ledger.synced = true;
                continue;
            }
        }
        if quad.predicate.as_str() != P_JSON {
            continue;
        }
        let subject = match &quad.subject {
            Subject::NamedNode(node) => node.as_str(),
            _ => continue,
        };
        let Some(class) = classes.get(subject) else {
            continue;
        };
        let Term::Literal(literal) = &quad.object else {
            continue;
        };
        let payload = literal.value();
        match class.as_str() {
            NAMESPACE_CLASS => {
                if let Ok(record) = serde_json::from_str::<NamespaceRecord>(payload) {
                    ledger.namespaces.push(record);
                }
            }
            ASSERTION_CLASS => {
                if let Ok(assertion) = serde_json::from_str::<ScopedAssertion>(payload) {
                    ledger.assertions.push(assertion);
                }
            }
            ADJUDICATION_CLASS => {
                if let Ok(adjudication) = serde_json::from_str::<Adjudication>(payload) {
                    ledger.adjudications.push(adjudication);
                }
            }
            APPLIED_CLASS => {
                if let Ok(winner) = serde_json::from_str::<AppliedWinner>(payload) {
                    ledger.applied.push(winner);
                }
            }
            STAMP_CLASS => {
                ledger.synced = true;
            }
            _ => {}
        }
    }
    if !ledger.synced
        && (!ledger.namespaces.is_empty()
            || !ledger.assertions.is_empty()
            || !ledger.adjudications.is_empty()
            || !ledger.applied.is_empty())
    {
        // 无账本戳的账本：有条目即视为已写过
        ledger.synced = true;
    }
    ledger.sort_and_dedup();
    ledger
}

#[cfg(test)]
mod tests {
    use super::*;

    fn namespace(id: &str, trust: TrustLevel) -> NamespaceRecord {
        NamespaceRecord::new(id, id, NamespaceKind::Import, trust)
    }

    fn literal(namespace: &str, subject: &str, predicate: &str, object: &str) -> ScopedAssertion {
        ScopedAssertion::new(
            namespace,
            subject,
            predicate,
            object,
            ObjectKind::Literal,
            "test",
        )
    }

    fn never(_: &str, _: &str, _: &str, _: ObjectKind) -> bool {
        false
    }

    fn plan_with(
        namespaces: &[NamespaceRecord],
        assertions: &[ScopedAssertion],
        functional: &[&str],
    ) -> FusionPlan {
        let policy = FusionPolicy::new().with_functional(functional.iter().copied());
        fusion_plan(namespaces, assertions, &[], &policy, never)
    }

    #[test]
    fn sanitize_namespace_id_is_idempotent_and_collision_resistant() {
        assert_eq!(sanitize_namespace_id("alice"), "alice");
        assert_eq!(sanitize_namespace_id(" alice "), "alice");
        assert_eq!(sanitize_namespace_id("batch-2026.10_x"), "batch-2026.10_x");
        let spaced = sanitize_namespace_id("a b");
        let underscored = sanitize_namespace_id("a_b");
        assert_ne!(spaced, underscored, "替换必须避免碰撞");
        assert_eq!(sanitize_namespace_id(&spaced), spaced, "幂等");
        assert_eq!(sanitize_namespace_id("   "), "source");
        let long = sanitize_namespace_id(&"源".repeat(100));
        assert!(long.len() <= 64, "超长 id 必须截断：{}", long);
        assert_eq!(sanitize_namespace_id(&long), long, "截断后仍幂等");
    }

    #[test]
    fn source_graph_iri_is_stable_and_isolated() {
        assert_eq!(
            source_graph_iri("alice"),
            "http://example.org/kg/graphs/source/alice"
        );
        assert_eq!(source_graph_iri("alice"), source_graph_iri("alice"));
    }

    #[test]
    fn trust_ordering_is_ordinal_and_stable() {
        assert!(TrustLevel::Verified.rank() > TrustLevel::Curated.rank());
        assert!(TrustLevel::Curated.rank() > TrustLevel::Imported.rank());
        assert!(TrustLevel::Imported.rank() > TrustLevel::External.rank());
        assert!(TrustLevel::External.rank() > TrustLevel::Speculative.rank());
        let high = NamespaceRecord::new("high", "high", NamespaceKind::Import, TrustLevel::Curated);
        let low = NamespaceRecord::new("low", "low", NamespaceKind::Import, TrustLevel::External);
        assert!(high.precedence() < low.precedence(), "受管优先于外部");
    }

    #[test]
    fn assertion_and_adjudication_ids_are_content_addressed() {
        let first = literal("a", "小明", "年龄", "18");
        let second = literal("a", "小明", "年龄", "18");
        assert_eq!(first.id, second.id);
        let other = literal("b", "小明", "年龄", "18");
        assert_ne!(first.id, other.id, "不同命名空间是不同断言");
        let entity = ScopedAssertion::new("a", "小明", "年龄", "18", ObjectKind::Entity, "test");
        assert_ne!(first.id, entity.id, "宾语种类参与身份");
        let adj = Adjudication::new("年龄", "a", "用户裁决", "user");
        assert_eq!(
            adj.id,
            Adjudication::new("年龄", "a", "其他理由", "system").id
        );
        assert_ne!(adj.id, Adjudication::new("身高", "a", "", "").id);
    }

    #[test]
    fn duplicate_and_new_are_classified_by_target_presence() {
        let namespaces = vec![namespace("a", TrustLevel::Curated)];
        let assertions = vec![literal("a", "小明", "年龄", "18")];
        let fresh = plan_with(&namespaces, &assertions, &["年龄"]);
        assert_eq!(fresh.slots[0].delta, FusionDelta::New);
        assert!(!fresh.slots[0].accepted[0].present);
        assert_eq!(fresh.materialize.len(), 1);

        let present = fusion_plan(
            &namespaces,
            &assertions,
            &[],
            &FusionPolicy::new().with_functional(["年龄"]),
            |s, p, o, _| (s, p, o) == ("小明", "年龄", "18"),
        );
        assert_eq!(present.slots[0].delta, FusionDelta::Duplicate);
        assert_eq!(present.materialize.len(), 1, "重复也返回代表，写回时判重");
    }

    #[test]
    fn multi_source_agreement_is_corroborated() {
        let namespaces = vec![
            namespace("a", TrustLevel::Imported),
            namespace("b", TrustLevel::External),
        ];
        let assertions = vec![
            literal("a", "小明", "年龄", "18"),
            literal("b", "小明", "年龄", "18"),
        ];
        let plan = plan_with(&namespaces, &assertions, &["年龄"]);
        assert_eq!(plan.slots[0].delta, FusionDelta::Corroborated);
        assert!(plan.slots[0].corroborated);
        assert_eq!(plan.slots[0].accepted.len(), 1);
        assert!(plan.slots[0].quarantined.is_empty());
    }

    #[test]
    fn functional_conflict_quarantines_minority_and_trust_wins() {
        let namespaces = vec![
            namespace("kb", TrustLevel::Curated),
            namespace("friend", TrustLevel::Imported),
        ];
        let assertions = vec![
            literal("friend", "小明", "年龄", "19"),
            literal("kb", "小明", "年龄", "18"),
        ];
        let plan = plan_with(&namespaces, &assertions, &["年龄"]);
        let slot = &plan.slots[0];
        assert_eq!(slot.delta, FusionDelta::Conflict);
        assert_eq!(slot.accepted.len(), 1);
        assert_eq!(slot.accepted[0].object, "18", "受管来源应获胜");
        assert_eq!(slot.accepted[0].best.namespace, "kb");
        assert_eq!(slot.quarantined.len(), 1);
        assert_eq!(slot.quarantined[0].object, "19");
        assert_eq!(plan.stats.quarantined, 1);
        assert!(slot.adjudicated_by.is_none());
    }

    #[test]
    fn non_functional_predicate_accepts_all_values() {
        let namespaces = vec![namespace("a", TrustLevel::Curated)];
        let assertions = vec![
            literal("a", "苹果", "颜色", "红色"),
            literal("a", "苹果", "颜色", "绿色"),
        ];
        let plan = plan_with(&namespaces, &assertions, &[]);
        let slot = &plan.slots[0];
        assert!(!slot.functional);
        assert_eq!(slot.accepted.len(), 2);
        assert!(slot.quarantined.is_empty());
        assert_eq!(plan.materialize.len(), 2);
        assert_eq!(slot.delta, FusionDelta::New);
    }

    #[test]
    fn adjudication_overrides_trust_and_is_recorded() {
        let namespaces = vec![
            namespace("kb", TrustLevel::Curated),
            namespace("friend", TrustLevel::Imported),
        ];
        let assertions = vec![
            literal("kb", "小明", "年龄", "18"),
            literal("friend", "小明", "年龄", "19"),
        ];
        let adjudication = Adjudication::new("年龄", "friend", "本人确认", "user");
        let policy = FusionPolicy::new().with_functional(["年龄"]);
        let plan = fusion_plan(
            &namespaces,
            &assertions,
            &[adjudication.clone()],
            &policy,
            never,
        );
        let slot = &plan.slots[0];
        assert_eq!(slot.delta, FusionDelta::Conflict);
        assert_eq!(slot.accepted[0].object, "19", "裁决应覆盖信任排序");
        assert_eq!(
            slot.adjudicated_by.as_deref(),
            Some(adjudication.id.as_str())
        );
        assert_eq!(slot.quarantined[0].object, "18");
    }

    #[test]
    fn adjudication_with_inactive_namespace_falls_back_to_trust() {
        let namespaces = vec![namespace("kb", TrustLevel::Curated)];
        let assertions = vec![literal("kb", "小明", "年龄", "18")];
        let adjudication = Adjudication::new("年龄", "ghost", "指向不存在来源", "user");
        let policy = FusionPolicy::new().with_functional(["年龄"]);
        let plan = fusion_plan(&namespaces, &assertions, &[adjudication], &policy, never);
        assert!(plan.slots[0].adjudicated_by.is_none());
        assert_eq!(plan.slots[0].accepted[0].object, "18");
    }

    #[test]
    fn orphan_and_unknown_namespace_assertions_are_counted_not_fused() {
        let namespaces = vec![namespace("a", TrustLevel::Curated)];
        let assertions = vec![
            literal("a", "小明", "年龄", "18"),
            literal("ghost", "小明", "年龄", "20"),
        ];
        let plan = plan_with(&namespaces, &assertions, &["年龄"]);
        assert_eq!(plan.stats.orphans, 1);
        assert_eq!(plan.stats.assertions, 1);
        assert_eq!(plan.materialize.len(), 1);
    }

    #[test]
    fn retired_namespace_never_participates() {
        let mut retired = namespace("old", TrustLevel::Verified);
        retired.status = NamespaceStatus::Retired;
        let namespaces = vec![retired, namespace("new", TrustLevel::Imported)];
        let assertions = vec![
            literal("old", "小明", "年龄", "18"),
            literal("new", "小明", "年龄", "19"),
        ];
        let plan = plan_with(&namespaces, &assertions, &["年龄"]);
        assert_eq!(plan.stats.namespaces, 1);
        assert_eq!(plan.stats.orphans, 1);
        assert_eq!(plan.slots[0].accepted[0].object, "19");
        assert!(plan.slots[0].quarantined.is_empty());
    }

    #[test]
    fn priority_breaks_ties_within_same_trust() {
        let namespaces = vec![
            namespace("session", TrustLevel::Imported).with_priority(20),
            namespace("batch", TrustLevel::Imported).with_priority(10),
        ];
        let assertions = vec![
            literal("batch", "小明", "年龄", "18"),
            literal("session", "小明", "年龄", "19"),
        ];
        let plan = plan_with(&namespaces, &assertions, &["年龄"]);
        assert_eq!(plan.slots[0].accepted[0].object, "19", "会话优先级更高");
        assert_eq!(plan.slots[0].quarantined[0].object, "18");
    }

    #[test]
    fn plan_rendering_is_deterministic_and_mentions_conflicts() {
        let namespaces = vec![
            namespace("kb", TrustLevel::Curated),
            namespace("friend", TrustLevel::Imported),
        ];
        let assertions = vec![
            literal("kb", "小明", "年龄", "18"),
            literal("friend", "小明", "年龄", "19"),
        ];
        let plan = plan_with(&namespaces, &assertions, &["年龄"]);
        let first = plan.render_zh();
        assert_eq!(first, plan.render_zh());
        assert!(first.contains("冲突"));
        assert!(first.contains("采用「18」"));
        assert!(first.contains("隔离「19」"));
    }

    #[test]
    fn ledger_upsert_and_adjudication_replace_are_deterministic() {
        let mut ledger = FusionLedger::new();
        ledger.upsert_namespace(namespace("b", TrustLevel::Imported));
        ledger.upsert_namespace(namespace("a", TrustLevel::Curated));
        ledger.upsert_namespace(namespace("a", TrustLevel::Verified));
        ledger.record_assertion(literal("a", "小明", "年龄", "18"));
        ledger.record_assertion(literal("a", "小明", "年龄", "18"));
        ledger.record_adjudication(Adjudication::new("年龄", "a", "第一版", "user"));
        ledger.record_adjudication(Adjudication::new("年龄", "b", "第二版", "user"));
        ledger.sort_and_dedup();
        assert_eq!(ledger.namespaces.len(), 2);
        assert_eq!(ledger.namespaces[0].id, "a");
        assert_eq!(ledger.namespaces[0].trust, TrustLevel::Verified, "覆盖生效");
        assert_eq!(ledger.assertions.len(), 1);
        assert_eq!(ledger.adjudications.len(), 1);
        assert_eq!(ledger.adjudications[0].prefer_namespace, "b");
    }

    #[test]
    fn fusion_quads_roundtrip_with_stamp_and_payloads() {
        let mut ledger = FusionLedger::new();
        ledger.upsert_namespace(namespace("kb", TrustLevel::Curated));
        ledger.record_assertion(literal("kb", "小明", "年龄", "18"));
        ledger.record_adjudication(Adjudication::new("年龄", "kb", "确认", "user"));
        let assertion = literal("kb", "小明", "年龄", "18");
        ledger.set_applied(vec![AppliedWinner::from_assertion(&assertion)]);
        let quads = fusion_quads(&ledger).unwrap();
        assert!(quads
            .iter()
            .all(|quad| quad.graph_name
                == GraphName::NamedNode(NamedNode::new(FUSION_GRAPH).unwrap())));
        let parsed = parse_fusion_graph(quads.into_iter());
        assert!(parsed.synced);
        let mut expected = ledger.clone();
        expected.synced = true;
        expected.sort_and_dedup();
        assert_eq!(parsed, expected);
    }

    #[test]
    fn parse_dedupes_and_sorts_even_with_duplicate_quads() {
        let mut ledger = FusionLedger::new();
        ledger.upsert_namespace(namespace("b", TrustLevel::Imported));
        ledger.upsert_namespace(namespace("a", TrustLevel::Curated));
        ledger.record_assertion(literal("b", "小明", "年龄", "19"));
        ledger.record_assertion(literal("a", "小明", "年龄", "18"));
        let mut quads = fusion_quads(&ledger).unwrap();
        quads.extend(fusion_quads(&ledger).unwrap());
        let parsed = parse_fusion_graph(quads.into_iter());
        assert_eq!(
            parsed
                .namespaces
                .iter()
                .map(|n| n.id.as_str())
                .collect::<Vec<_>>(),
            vec!["a", "b"]
        );
        assert_eq!(parsed.assertions.len(), 2);
    }

    #[test]
    fn empty_ledger_has_stamp_and_renders_summary() {
        let quads = fusion_quads(&FusionLedger::new()).unwrap();
        let parsed = parse_fusion_graph(quads.into_iter());
        assert!(parsed.synced);
        assert!(parsed.namespaces.is_empty());
        assert!(parsed.render_zh().contains("知识融合账本"));
    }
}

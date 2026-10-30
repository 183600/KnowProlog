//! 共指闭包（Coreference Closure）：把「同指」当图，把闭包当算子。
//!
//! 现有模块各自记录一类证据：`variant_closure` 记词形，`contextual_reference`
//! 记账本，`trace_graph` 记指代事件。本模块只在查询时把它们投影成一张
//! **身份图**，不复制事实源：
//!
//!   - **恒等边**（别名、简称、全称、代词、同位语、省略、更正、绑定）参与
//!     自反—对称—传递闭包，得到若干等价类；
//!   - **非恒等边**（描述、桥接、角色、借代）只留作审计，绝不参与传递；
//!   - 每个等价类选一个规范实体，并保留规范实体到每个成员的**最短见证**；
//!   - 关系图先按等价类做商，再在商图上跑 `pattern_closure::evaluate_closure`；
//!     坍缩成自环的关系边只作为证据保留，不参与遍历。
//!
//! 安全约定：
//!   - 类型不相容的边被拒绝，不自动合并（宁可不答，不可错并）；
//!   - 一个词面同时锚到多个互不相连的实体时，该词面被隔离为显式歧义；
//!   - 更正只撤销词面与旧实体的恒等边，并报告受影响的等价类，供
//!     `trace_graph` 标记 stale；
//!   - 所有输出排序固定，同输入同字节。

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

use serde::{Deserialize, Serialize};

use crate::contextual_reference::BindingLedger;
use crate::entity_registry::{fold_surface, EntityRegistry, EntityStatus};
use crate::pattern_closure::{
    evaluate_closure, ClosureBudget, ClosureError, ClosureOp, ClosureOutcome,
};
use crate::temporal::{TimePoint, Validity};
use crate::trace_graph::{ReferentSource, ReferentStatus, ReferentTrail};
use crate::type_system::{TypeSystem, TypeVerdict};
use crate::variant_closure::{VariantClosure, VariantContext, VariantKind};

/// 身份图方言标识（JSON 快照用）。
pub const COREF_CLOSURE_DIALECT: &str = "knowprolog-coref-closure/1";
/// 默认置信度门槛：低于它的恒等边不参与闭包。
pub const DEFAULT_COREF_MIN_CONFIDENCE: f64 = 0.35;
/// 默认见证最大跳数。
pub const DEFAULT_COREF_MAX_HOPS: usize = 6;
/// 默认见证最大节点数（每个等价类）。
pub const DEFAULT_COREF_MAX_NODES: usize = 256;
/// 默认参与闭包的最大恒等边数。
pub const DEFAULT_COREF_MAX_EDGES: usize = 2_048;

/// 恒等边类型：只表达「这两个词面在当前语境下指同一个实体」。
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum IdentityKind {
    #[default]
    Alias,
    Abbreviation,
    FullName,
    Initialism,
    Transliteration,
    Translation,
    FormerName,
    Honorific,
    Nickname,
    Elision,
    Typo,
    Pronoun,
    Apposition,
    Ellipsis,
    Correction,
    Binding,
    Explicit,
}

impl IdentityKind {
    pub const fn label(self) -> &'static str {
        match self {
            IdentityKind::Alias => "别名",
            IdentityKind::Abbreviation => "简称",
            IdentityKind::FullName => "全称",
            IdentityKind::Initialism => "首字母缩写",
            IdentityKind::Transliteration => "音译",
            IdentityKind::Translation => "翻译",
            IdentityKind::FormerName => "曾用名",
            IdentityKind::Honorific => "尊称",
            IdentityKind::Nickname => "昵称",
            IdentityKind::Elision => "词形省略",
            IdentityKind::Typo => "错别字",
            IdentityKind::Pronoun => "代词",
            IdentityKind::Apposition => "同位语",
            IdentityKind::Ellipsis => "省略补全",
            IdentityKind::Correction => "用户更正",
            IdentityKind::Binding => "语境绑定",
            IdentityKind::Explicit => "显式指称",
        }
    }

    /// 词形闭包的边类型映射；借代不是恒等，返回 `None`。
    pub fn from_variant(kind: VariantKind) -> Option<Self> {
        match kind {
            VariantKind::Alias => Some(IdentityKind::Alias),
            VariantKind::Abbreviation => Some(IdentityKind::Abbreviation),
            VariantKind::FullName => Some(IdentityKind::FullName),
            VariantKind::Initialism => Some(IdentityKind::Initialism),
            VariantKind::Transliteration => Some(IdentityKind::Transliteration),
            VariantKind::Translation => Some(IdentityKind::Translation),
            VariantKind::FormerName => Some(IdentityKind::FormerName),
            VariantKind::Honorific => Some(IdentityKind::Honorific),
            VariantKind::Nickname => Some(IdentityKind::Nickname),
            VariantKind::Elision => Some(IdentityKind::Elision),
            VariantKind::Typo => Some(IdentityKind::Typo),
            VariantKind::Metonymy => None,
        }
    }
}

/// 非恒等边类型：描述「通过某个关系/角色可以找到实体」，但不表达同一性。
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum NonIdentityKind {
    #[default]
    Description,
    Bridging,
    Role,
    Metonymy,
}

impl NonIdentityKind {
    pub const fn label(self) -> &'static str {
        match self {
            NonIdentityKind::Description => "描述性指称",
            NonIdentityKind::Bridging => "桥接指称",
            NonIdentityKind::Role => "角色指称",
            NonIdentityKind::Metonymy => "借代",
        }
    }
}

/// 恒等边状态：活跃 / 被证据反驳 / 已过期（更正或忘记）。
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum IdentityEdgeStatus {
    #[default]
    Active,
    Refuted,
    Expired,
}

impl IdentityEdgeStatus {
    pub const fn label(self) -> &'static str {
        match self {
            IdentityEdgeStatus::Active => "活跃",
            IdentityEdgeStatus::Refuted => "已反驳",
            IdentityEdgeStatus::Expired => "已过期",
        }
    }
}

/// 一条恒等证据。`anchor` 是证据想指向的规范实体（可暂时不在端点里）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IdentityEdge {
    pub from: String,
    pub to: String,
    #[serde(default)]
    pub anchor: String,
    pub kind: IdentityKind,
    #[serde(default)]
    pub source: String,
    #[serde(default = "default_identity_confidence")]
    pub confidence: f64,
    #[serde(default)]
    pub validity: Validity,
    #[serde(default)]
    pub status: IdentityEdgeStatus,
}

fn default_identity_confidence() -> f64 {
    0.9
}

impl IdentityEdge {
    pub fn new(from: &str, to: &str, kind: IdentityKind, anchor: &str) -> Self {
        Self {
            from: from.trim().to_string(),
            to: to.trim().to_string(),
            anchor: anchor.trim().to_string(),
            kind,
            source: "未注明来源".to_string(),
            confidence: default_identity_confidence(),
            validity: Validity::Unknown,
            status: IdentityEdgeStatus::Active,
        }
    }

    pub fn with_source(mut self, source: &str) -> Self {
        let source = source.trim();
        if !source.is_empty() {
            self.source = source.to_string();
        }
        self
    }

    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    pub fn with_validity(mut self, validity: Validity) -> Self {
        self.validity = validity;
        self
    }

    pub fn with_status(mut self, status: IdentityEdgeStatus) -> Self {
        self.status = status;
        self
    }

    pub fn render(&self) -> String {
        format!(
            "「{}」—[{}]→「{}」（{}，{:.2}）",
            self.from,
            self.kind.label(),
            self.to,
            self.source,
            self.confidence
        )
    }
}

/// 一条非恒等证据，只入审计账，不参与等价闭包。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NonIdentityEdge {
    pub from: String,
    pub to: String,
    pub kind: NonIdentityKind,
    #[serde(default)]
    pub source: String,
}

impl NonIdentityEdge {
    pub fn new(from: &str, to: &str, kind: NonIdentityKind, source: &str) -> Self {
        Self {
            from: from.trim().to_string(),
            to: to.trim().to_string(),
            kind,
            source: source.trim().to_string(),
        }
    }

    pub fn render(&self) -> String {
        format!(
            "「{}」—[{}]→「{}」（{}；非恒等，不传递）",
            self.from,
            self.kind.label(),
            self.to,
            self.source
        )
    }
}

/// 关系边：商图的输入。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RelationEdge {
    pub from: String,
    pub to: String,
    pub predicate: String,
    #[serde(default)]
    pub source: String,
    #[serde(default = "default_identity_confidence")]
    pub confidence: f64,
    #[serde(default)]
    pub validity: Validity,
}

impl RelationEdge {
    pub fn new(from: &str, to: &str, predicate: &str) -> Self {
        Self {
            from: from.trim().to_string(),
            to: to.trim().to_string(),
            predicate: predicate.trim().to_string(),
            source: String::new(),
            confidence: default_identity_confidence(),
            validity: Validity::Unknown,
        }
    }

    pub fn with_source(mut self, source: &str) -> Self {
        self.source = source.trim().to_string();
        self
    }

    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    pub fn with_validity(mut self, validity: Validity) -> Self {
        self.validity = validity;
        self
    }
}

/// 闭包预算：边数、见证跳数与节点数都有上限，截断必须显式报告。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CorefBudget {
    pub max_edges: usize,
    pub max_hops: usize,
    pub max_nodes: usize,
    pub min_confidence: f64,
    /// 查询时间点；`None` 表示不做时间过滤。
    pub at: Option<TimePoint>,
}

impl Default for CorefBudget {
    fn default() -> Self {
        Self {
            max_edges: DEFAULT_COREF_MAX_EDGES,
            max_hops: DEFAULT_COREF_MAX_HOPS,
            max_nodes: DEFAULT_COREF_MAX_NODES,
            min_confidence: DEFAULT_COREF_MIN_CONFIDENCE,
            at: None,
        }
    }
}

impl CorefBudget {
    pub fn with_max_edges(mut self, max_edges: usize) -> Self {
        self.max_edges = max_edges.max(1);
        self
    }

    pub fn with_max_hops(mut self, max_hops: usize) -> Self {
        self.max_hops = max_hops.max(1);
        self
    }

    pub fn with_max_nodes(mut self, max_nodes: usize) -> Self {
        self.max_nodes = max_nodes.max(1);
        self
    }

    pub fn with_min_confidence(mut self, min_confidence: f64) -> Self {
        self.min_confidence = min_confidence.clamp(0.0, 1.0);
        self
    }

    pub fn at(mut self, point: TimePoint) -> Self {
        self.at = Some(point);
        self
    }

    pub fn at_year(self, year: i32) -> Self {
        self.at(TimePoint::year(year))
    }
}

/// 全局截断原因（每条等价类另有见证截断标记）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CorefTruncation {
    Edges,
}

impl CorefTruncation {
    pub const fn label(self) -> &'static str {
        match self {
            CorefTruncation::Edges => "恒等边上限",
        }
    }
}

/// 边被拒绝的原因（拒绝也留痕，不静默丢弃）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CorefRejectionReason {
    LowConfidence,
    TypeConflict,
    InvalidValidity,
}

impl CorefRejectionReason {
    pub const fn label(self) -> &'static str {
        match self {
            CorefRejectionReason::LowConfidence => "置信度不足",
            CorefRejectionReason::TypeConflict => "类型不相容",
            CorefRejectionReason::InvalidValidity => "有效期不成立",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorefRejection {
    pub from: String,
    pub to: String,
    pub kind: IdentityKind,
    pub source: String,
    pub reason: CorefRejectionReason,
    pub detail: String,
}

impl CorefRejection {
    fn render(&self) -> String {
        format!(
            "「{}」—[{}]→「{}」（{}）：{}",
            self.from,
            self.kind.label(),
            self.to,
            self.reason.label(),
            self.detail
        )
    }
}

/// 等价类级冲突：类型经传递后不相容，或有效期交集为空。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CorefConflictReason {
    TypeConflict,
    ValidityConflict,
}

impl CorefConflictReason {
    pub const fn label(self) -> &'static str {
        match self {
            CorefConflictReason::TypeConflict => "类内类型冲突",
            CorefConflictReason::ValidityConflict => "类内有效期冲突",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorefConflict {
    pub members: Vec<String>,
    pub reason: CorefConflictReason,
    pub detail: String,
}

impl CorefConflict {
    fn render(&self) -> String {
        format!(
            "{}：{}（{}）",
            self.members.join("、"),
            self.reason.label(),
            self.detail
        )
    }
}

/// 身份见证的一步。`from → to` 是遍历方向，`reversed` 表示原边方向相反。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorefStep {
    pub from: String,
    pub to: String,
    pub kind: IdentityKind,
    pub source: String,
    pub confidence: f64,
    pub reversed: bool,
}

impl CorefStep {
    pub fn render(&self) -> String {
        if self.reversed {
            format!("「{}」←[{}]—「{}」", self.from, self.kind.label(), self.to)
        } else {
            format!("「{}」—[{}]→「{}」", self.from, self.kind.label(), self.to)
        }
    }
}

fn reverse_steps(steps: &[CorefStep]) -> Vec<CorefStep> {
    steps
        .iter()
        .rev()
        .map(|step| CorefStep {
            from: step.to.clone(),
            to: step.from.clone(),
            kind: step.kind,
            source: step.source.clone(),
            confidence: step.confidence,
            reversed: !step.reversed,
        })
        .collect()
}

fn render_steps(steps: &[CorefStep]) -> String {
    steps
        .iter()
        .map(CorefStep::render)
        .collect::<Vec<_>>()
        .join(" ")
}

/// 一个共指等价类：规范实体、成员、类型、置信度与最短见证。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorefClass {
    pub canonical: String,
    /// 稳定排序；包含规范实体本身。
    pub members: Vec<String>,
    pub types: BTreeSet<String>,
    pub confidence: f64,
    pub validity: Validity,
    /// 规范实体 → 成员的最短见证（规范实体自身为空路径）。
    pub witnesses: BTreeMap<String, Vec<CorefStep>>,
    /// 预算不足导致部分成员没有见证。
    #[serde(default)]
    pub witness_truncated: bool,
}

impl CorefClass {
    pub fn aliases(&self) -> impl Iterator<Item = &str> {
        self.members
            .iter()
            .map(String::as_str)
            .filter(move |member| *member != self.canonical)
    }

    pub fn witness_to(&self, surface: &str) -> Option<&[CorefStep]> {
        self.witnesses
            .get(&fold_surface(surface))
            .map(Vec::as_slice)
    }

    pub fn render(&self) -> String {
        let mut out = format!(
            "「{}」← [{}]（类型：{}，置信度：{:.2}，有效期：{}）",
            self.canonical,
            self.members.join("、"),
            if self.types.is_empty() {
                "未知".to_string()
            } else {
                self.types.iter().cloned().collect::<Vec<_>>().join("、")
            },
            self.confidence,
            self.validity.render()
        );
        for member in &self.members {
            if member == &self.canonical {
                continue;
            }
            match self.witnesses.get(member) {
                Some(path) if !path.is_empty() => {
                    out.push_str(&format!(
                        "\n    「{}」→「{}」：{}",
                        member,
                        self.canonical,
                        render_steps(&reverse_steps(path))
                    ));
                }
                _ => out.push_str(&format!(
                    "\n    「{}」→「{}」：（见证被预算截断）",
                    member, self.canonical
                )),
            }
        }
        out
    }
}

/// 指称解析状态。`Ambiguous` 是显式歧义，绝不自动合并。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CorefStatus {
    Canonical,
    Alias,
    Ambiguous,
    Unknown,
}

impl CorefStatus {
    pub const fn label(self) -> &'static str {
        match self {
            CorefStatus::Canonical => "规范身份",
            CorefStatus::Alias => "同一身份",
            CorefStatus::Ambiguous => "显式歧义",
            CorefStatus::Unknown => "无恒等证据",
        }
    }
}

/// 单个词面的共指解析结果。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorefResolution {
    pub surface: String,
    pub status: CorefStatus,
    pub canonical: Option<String>,
    pub candidates: Vec<String>,
    pub members: Vec<String>,
    pub types: BTreeSet<String>,
    pub confidence: f64,
    pub validity: Validity,
    /// 规范实体 → 该词面的最短见证。
    pub witness: Vec<CorefStep>,
    pub witness_truncated: bool,
}

impl CorefResolution {
    pub fn render(&self) -> String {
        match self.status {
            CorefStatus::Canonical => format!(
                "「{}」是规范身份（类内 {} 个词面，类型：{}，置信度：{:.2}）",
                self.surface,
                self.members.len(),
                if self.types.is_empty() {
                    "未知".to_string()
                } else {
                    self.types.iter().cloned().collect::<Vec<_>>().join("、")
                },
                self.confidence
            ),
            CorefStatus::Alias => {
                let canonical = self.canonical.as_deref().unwrap_or("");
                let path = if self.witness.is_empty() {
                    format!("「{}」→「{}」（直接声明）", self.surface, canonical)
                } else {
                    render_steps(&reverse_steps(&self.witness))
                };
                format!(
                    "「{}」→「{}」（{}；置信度：{:.2}{}）",
                    self.surface,
                    canonical,
                    path,
                    self.confidence,
                    if self.witness_truncated {
                        "；见证被预算截断"
                    } else {
                        ""
                    }
                )
            }
            CorefStatus::Ambiguous => format!(
                "「{}」有多个互不相连的身份候选：{}（显式歧义，不自动合并）",
                self.surface,
                self.candidates.join("、")
            ),
            CorefStatus::Unknown => {
                format!("「{}」没有恒等证据（不静默造实体）", self.surface)
            }
        }
    }
}

/// 身份闭包的查询视图。它是只读快照，不是第四份事实库。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorefBuild {
    pub classes: Vec<CorefClass>,
    pub canonical_of: BTreeMap<String, String>,
    pub ambiguous: BTreeMap<String, Vec<String>>,
    pub rejected: Vec<CorefRejection>,
    pub conflicts: Vec<CorefConflict>,
    pub nodes: usize,
    pub active_edges: usize,
    pub truncated: Option<CorefTruncation>,
}

impl CorefBuild {
    pub fn canonical(&self, surface: &str) -> Option<&str> {
        self.canonical_of
            .get(&fold_surface(surface))
            .map(String::as_str)
    }

    pub fn class_of(&self, surface: &str) -> Option<&CorefClass> {
        let canonical = self.canonical(surface)?;
        self.classes
            .iter()
            .find(|class| class.canonical == canonical)
    }

    pub fn resolve(&self, surface: &str) -> CorefResolution {
        let key = fold_surface(surface);
        if let Some(candidates) = self.ambiguous.get(&key) {
            return CorefResolution {
                surface: key,
                status: CorefStatus::Ambiguous,
                canonical: None,
                candidates: candidates.clone(),
                members: Vec::new(),
                types: BTreeSet::new(),
                confidence: 0.0,
                validity: Validity::Unknown,
                witness: Vec::new(),
                witness_truncated: false,
            };
        }
        let Some(canonical) = self.canonical_of.get(&key).cloned() else {
            return CorefResolution {
                surface: key,
                status: CorefStatus::Unknown,
                canonical: None,
                candidates: Vec::new(),
                members: Vec::new(),
                types: BTreeSet::new(),
                confidence: 0.0,
                validity: Validity::Unknown,
                witness: Vec::new(),
                witness_truncated: false,
            };
        };
        let Some(class) = self
            .classes
            .iter()
            .find(|class| class.canonical == canonical)
        else {
            return CorefResolution {
                surface: key,
                status: CorefStatus::Unknown,
                canonical: None,
                candidates: Vec::new(),
                members: Vec::new(),
                types: BTreeSet::new(),
                confidence: 0.0,
                validity: Validity::Unknown,
                witness: Vec::new(),
                witness_truncated: false,
            };
        };
        let status = if key == canonical {
            CorefStatus::Canonical
        } else {
            CorefStatus::Alias
        };
        let witness = class.witnesses.get(&key).cloned().unwrap_or_default();
        CorefResolution {
            surface: key,
            status,
            canonical: Some(canonical),
            candidates: Vec::new(),
            members: class.members.clone(),
            types: class.types.clone(),
            confidence: class.confidence,
            validity: class.validity,
            witness_truncated: status == CorefStatus::Alias
                && witness.is_empty()
                && class.witness_truncated,
            witness,
        }
    }

    pub fn report(&self) -> String {
        let mut out = format!(
            "共指闭包：{} 个等价类，{} 个词面，{} 条活跃恒等边（拒绝 {} 条，歧义 {} 个，冲突 {} 个）",
            self.classes.len(),
            self.canonical_of.len(),
            self.active_edges,
            self.rejected.len(),
            self.ambiguous.len(),
            self.conflicts.len()
        );
        if let Some(truncated) = self.truncated {
            out.push_str(&format!("\n  截断：{}", truncated.label()));
        }
        for class in &self.classes {
            out.push_str(&format!("\n  • {}", class.render()));
        }
        if !self.ambiguous.is_empty() {
            out.push_str("\n  显式歧义：");
            for (surface, candidates) in &self.ambiguous {
                out.push_str(&format!("\n    「{}」→ {}", surface, candidates.join("、")));
            }
        }
        if !self.conflicts.is_empty() {
            out.push_str("\n  类内冲突：");
            for conflict in &self.conflicts {
                out.push_str(&format!("\n    {}", conflict.render()));
            }
        }
        if !self.rejected.is_empty() {
            out.push_str("\n  被拒绝的边：");
            for rejection in &self.rejected {
                out.push_str(&format!("\n    {}", rejection.render()));
            }
        }
        out
    }
}

/// 身份图：恒等边、非恒等边与关系边的统一入口。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CorefGraph {
    #[serde(default)]
    nodes: BTreeMap<String, CorefNode>,
    #[serde(default)]
    identity_edges: Vec<IdentityEdge>,
    #[serde(default)]
    non_identity_edges: Vec<NonIdentityEdge>,
    #[serde(default)]
    relation_edges: Vec<RelationEdge>,
    #[serde(default)]
    next_canonical_order: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    types: Option<TypeSystem>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
struct CorefNode {
    #[serde(default)]
    types: BTreeSet<String>,
    #[serde(default)]
    canonical_order: Option<usize>,
}

impl CorefGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_type_system(mut self, types: TypeSystem) -> Self {
        self.types = Some(types);
        self
    }

    pub fn set_type_system(&mut self, types: TypeSystem) {
        self.types = Some(types);
    }

    pub fn type_system(&self) -> Option<&TypeSystem> {
        self.types.as_ref()
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn identity_edge_count(&self) -> usize {
        self.identity_edges.len()
    }

    pub fn non_identity_edges(&self) -> &[NonIdentityEdge] {
        &self.non_identity_edges
    }

    pub fn identity_edges(&self) -> &[IdentityEdge] {
        &self.identity_edges
    }

    pub fn relation_edges(&self) -> &[RelationEdge] {
        &self.relation_edges
    }

    fn ensure_node(&mut self, surface: &str) -> String {
        let key = fold_surface(surface);
        if !key.is_empty() {
            self.nodes.entry(key.clone()).or_default();
        }
        key
    }

    pub fn declare_canonical(&mut self, surface: &str) -> bool {
        let key = self.ensure_node(surface);
        if key.is_empty() {
            return false;
        }
        let order = self.next_canonical_order;
        let node = self.nodes.get_mut(&key).expect("刚插入的节点");
        if node.canonical_order.is_none() {
            node.canonical_order = Some(order);
            self.next_canonical_order += 1;
            true
        } else {
            false
        }
    }

    pub fn declare_type(&mut self, surface: &str, type_name: &str) -> bool {
        let type_name = type_name.trim();
        if type_name.is_empty() {
            return false;
        }
        let key = self.ensure_node(surface);
        if key.is_empty() {
            return false;
        }
        self.nodes
            .get_mut(&key)
            .expect("刚插入的节点")
            .types
            .insert(type_name.to_string())
    }

    pub fn declare_types(&mut self, surface: &str, types: &BTreeSet<String>) -> bool {
        let key = self.ensure_node(surface);
        if key.is_empty() {
            return false;
        }
        let node = self.nodes.get_mut(&key).expect("刚插入的节点");
        let mut changed = false;
        for type_name in types {
            let type_name = type_name.trim();
            if !type_name.is_empty() && node.types.insert(type_name.to_string()) {
                changed = true;
            }
        }
        changed
    }

    pub fn add_identity_edge(&mut self, mut edge: IdentityEdge) -> bool {
        edge.from = fold_surface(&edge.from);
        edge.to = fold_surface(&edge.to);
        edge.anchor = fold_surface(&edge.anchor);
        if edge.anchor.is_empty() {
            edge.anchor = edge.to.clone();
        }
        if edge.from.is_empty()
            || edge.to.is_empty()
            || edge.anchor.is_empty()
            || edge.from == edge.to
        {
            return false;
        }
        edge.confidence = edge.confidence.clamp(0.0, 1.0);
        let source = edge.source.trim();
        edge.source = if source.is_empty() {
            "未注明来源".to_string()
        } else {
            source.to_string()
        };
        self.ensure_node(&edge.from);
        self.ensure_node(&edge.to);
        self.identity_edges.push(edge);
        true
    }

    /// 便捷入口：`from` 与 `to` 同指，证据锚定 `anchor`。
    pub fn link(&mut self, from: &str, to: &str, kind: IdentityKind, anchor: &str) -> bool {
        self.add_identity_edge(IdentityEdge::new(from, to, kind, anchor))
    }

    pub fn add_non_identity_edge(&mut self, edge: NonIdentityEdge) -> bool {
        let from = fold_surface(&edge.from);
        let to = fold_surface(&edge.to);
        if from.is_empty() || to.is_empty() {
            return false;
        }
        self.ensure_node(&from);
        self.ensure_node(&to);
        self.non_identity_edges.push(NonIdentityEdge {
            from,
            to,
            kind: edge.kind,
            source: if edge.source.trim().is_empty() {
                "未注明来源".to_string()
            } else {
                edge.source.trim().to_string()
            },
        });
        true
    }

    pub fn describe(&mut self, from: &str, to: &str, kind: NonIdentityKind) -> bool {
        self.add_non_identity_edge(NonIdentityEdge::new(from, to, kind, "手工声明"))
    }

    pub fn add_relation_edge(&mut self, edge: RelationEdge) -> bool {
        let from = fold_surface(&edge.from);
        let to = fold_surface(&edge.to);
        let predicate = edge.predicate.trim();
        if from.is_empty() || to.is_empty() || predicate.is_empty() {
            return false;
        }
        self.ensure_node(&from);
        self.ensure_node(&to);
        self.relation_edges.push(RelationEdge {
            from,
            to,
            predicate: predicate.to_string(),
            source: edge.source.trim().to_string(),
            confidence: edge.confidence.clamp(0.0, 1.0),
            validity: edge.validity,
        });
        true
    }

    pub fn relate(&mut self, from: &str, to: &str, predicate: &str) -> bool {
        self.add_relation_edge(RelationEdge::new(from, to, predicate))
    }

    /// 反驳一条恒等边；返回命中的活跃边数。
    pub fn refute_identity_edge(&mut self, from: &str, to: &str, anchor: &str) -> usize {
        let from = fold_surface(from);
        let to = fold_surface(to);
        let anchor = fold_surface(anchor);
        let mut count = 0;
        for edge in &mut self.identity_edges {
            if edge.status == IdentityEdgeStatus::Active
                && edge.from == from
                && edge.to == to
                && (anchor.is_empty() || edge.anchor == anchor)
            {
                edge.status = IdentityEdgeStatus::Refuted;
                count += 1;
            }
        }
        count
    }

    /// 用户更正：撤销 `surface` 与旧实体 `from` 的恒等边，改绑到 `to`。
    ///
    /// 返回受影响的等价类（更正前的规范实体），供上层把旧答案标记为 stale。
    pub fn correct(
        &mut self,
        surface: &str,
        from: &str,
        to: &str,
        source: &str,
    ) -> Result<CorrectionOutcome, String> {
        let surface_key = fold_surface(surface);
        let from_key = fold_surface(from);
        let to_key = fold_surface(to);
        if surface_key.is_empty() {
            return Err("更正词面不能为空".to_string());
        }
        if to_key.is_empty() {
            return Err("更正目标不能为空".to_string());
        }
        if surface_key == to_key {
            return Err("更正词面与目标相同，无需更正".to_string());
        }
        if surface_key == from_key {
            return Err("更正词面不能与被更正实体同名；请传入实际指称（如代词或别名）".to_string());
        }
        let before = self.build(&CorefBudget::default());
        let mut affected = BTreeSet::new();
        if let Some(canonical) = before.canonical(&from_key) {
            affected.insert(canonical.to_string());
        }
        if let Some(canonical) = before.canonical(&surface_key) {
            affected.insert(canonical.to_string());
        }
        let mut expired_edges = 0;
        for edge in &mut self.identity_edges {
            if edge.status != IdentityEdgeStatus::Active {
                continue;
            }
            let incident = edge.from == surface_key || edge.to == surface_key;
            if incident && (edge.anchor == from_key || edge.from == from_key || edge.to == from_key)
            {
                edge.status = IdentityEdgeStatus::Expired;
                expired_edges += 1;
            }
        }
        let mut edge_added = false;
        if before.canonical(&surface_key) != Some(to_key.as_str()) {
            let already = self.identity_edges.iter().any(|edge| {
                edge.status == IdentityEdgeStatus::Active
                    && edge.from == surface_key
                    && edge.to == to_key
                    && edge.kind == IdentityKind::Correction
            });
            if !already {
                let source = if source.trim().is_empty() {
                    "用户更正"
                } else {
                    source.trim()
                };
                edge_added = self.add_identity_edge(
                    IdentityEdge::new(&surface_key, &to_key, IdentityKind::Correction, &to_key)
                        .with_source(source)
                        .with_confidence(0.95),
                );
            }
        }
        Ok(CorrectionOutcome {
            surface: surface_key,
            from: from_key,
            to: to_key,
            expired_edges,
            affected_classes: affected.into_iter().collect(),
            edge_added,
        })
    }

    /// 用户显式否定一个词面：撤销其全部活跃恒等边，但保留审计记录。
    pub fn forget(&mut self, surface: &str) -> ForgetOutcome {
        let key = fold_surface(surface);
        let before = self.build(&CorefBudget::default());
        let mut affected = BTreeSet::new();
        if let Some(canonical) = before.canonical(&key) {
            affected.insert(canonical.to_string());
        }
        let mut expired_edges = 0;
        for edge in &mut self.identity_edges {
            if edge.status == IdentityEdgeStatus::Active && (edge.from == key || edge.to == key) {
                edge.status = IdentityEdgeStatus::Expired;
                expired_edges += 1;
            }
        }
        ForgetOutcome {
            surface: key,
            expired_edges,
            affected_classes: affected.into_iter().collect(),
        }
    }

    /// 吸收实体注册表：已确认别名、类型与显式合并都成为恒等证据。
    pub fn absorb_registry(&mut self, registry: &EntityRegistry) -> usize {
        let mut added = 0;
        for record in registry.records() {
            if record.status == EntityStatus::Rejected {
                continue;
            }
            let canonical = fold_surface(&record.canonical);
            if canonical.is_empty() {
                continue;
            }
            self.ensure_node(&canonical);
            self.declare_types(&canonical, &record.types);
            self.declare_canonical(&canonical);
            if let Some(target) = &record.merged_into {
                let target = fold_surface(target);
                if canonical != target
                    && self.add_identity_edge(
                        IdentityEdge::new(&canonical, &target, IdentityKind::FormerName, &target)
                            .with_source("实体注册表：显式合并")
                            .with_confidence(0.95),
                    )
                {
                    added += 1;
                }
                continue;
            }
            for alias in record.aliases.values() {
                if alias.status != EntityStatus::Confirmed {
                    continue;
                }
                let surface = fold_surface(&alias.surface);
                if surface.is_empty() || surface == canonical {
                    continue;
                }
                let kind = IdentityKind::from_variant(VariantKind::infer(&surface, &canonical))
                    .unwrap_or(IdentityKind::Alias);
                if self.add_identity_edge(
                    IdentityEdge::new(&surface, &canonical, kind, &canonical)
                        .with_source("实体注册表：已确认别名")
                        .with_confidence(0.9),
                ) {
                    added += 1;
                }
                for variant in &alias.variants {
                    let variant = fold_surface(variant);
                    if variant.is_empty() || variant == canonical || variant == surface {
                        continue;
                    }
                    let kind = IdentityKind::from_variant(VariantKind::infer(&variant, &canonical))
                        .unwrap_or(IdentityKind::Typo);
                    if self.add_identity_edge(
                        IdentityEdge::new(&variant, &canonical, kind, &canonical)
                            .with_source("实体注册表：同折叠形写法")
                            .with_confidence(0.85),
                    ) {
                        added += 1;
                    }
                }
            }
        }
        added
    }

    /// 吸收词形闭包：借代进入非恒等账，其余进入身份图。
    pub fn absorb_variant_closure(
        &mut self,
        closure: &VariantClosure,
        context: &VariantContext,
    ) -> usize {
        let mut added = 0;
        for anchor in closure.types().values() {
            self.declare_canonical(&anchor.canonical);
            self.declare_types(&anchor.canonical, &anchor.types);
        }
        for edge in closure.edges() {
            if edge.refuted {
                continue;
            }
            if edge.confidence < context.min_confidence
                || !validity_allows(edge.validity, context.at)
            {
                continue;
            }
            match IdentityKind::from_variant(edge.kind) {
                Some(kind) => {
                    if self.add_identity_edge(
                        IdentityEdge::new(&edge.from, &edge.to, kind, &edge.anchor)
                            .with_source(&edge.source)
                            .with_confidence(edge.confidence)
                            .with_validity(edge.validity),
                    ) {
                        added += 1;
                    }
                }
                None => {
                    self.add_non_identity_edge(NonIdentityEdge::new(
                        &edge.from,
                        &edge.to,
                        NonIdentityKind::Metonymy,
                        &edge.source,
                    ));
                }
            }
        }
        added
    }

    /// 吸收语境绑定账本：`词面 → 实体` 是会话级恒等证据。
    pub fn absorb_bindings(&mut self, ledger: &BindingLedger) -> usize {
        let mut added = 0;
        for binding in ledger.active_bindings() {
            let surface = fold_surface(&binding.key);
            let entity = fold_surface(&binding.entity);
            if surface.is_empty() || entity.is_empty() || surface == entity {
                continue;
            }
            let confidence = (0.55 + 0.1 * binding.confirmations.min(4) as f64).min(0.95);
            let source = format!("指称绑定 {}（{}）", binding.id, binding.scope.label());
            if self.add_identity_edge(
                IdentityEdge::new(&surface, &entity, IdentityKind::Binding, &entity)
                    .with_source(&source)
                    .with_confidence(confidence),
            ) {
                added += 1;
            }
        }
        added
    }

    /// 吸收指代轨迹：代词、省略、更正与显式指称都成为恒等证据。
    pub fn absorb_referent_trail(&mut self, trail: &ReferentTrail) -> usize {
        let mut added = 0;
        for event in trail.events() {
            if event.status != ReferentStatus::Active {
                continue;
            }
            let surface = fold_surface(&event.surface);
            let referent = fold_surface(&event.referent);
            if surface.is_empty() || referent.is_empty() || surface == referent {
                continue;
            }
            let (kind, confidence) = match event.source {
                ReferentSource::Explicit => (IdentityKind::Explicit, 0.85),
                ReferentSource::Pronoun => (IdentityKind::Pronoun, 0.7),
                ReferentSource::Ellipsis => (IdentityKind::Ellipsis, 0.75),
                ReferentSource::Correction => (IdentityKind::Correction, 0.9),
                ReferentSource::Plugin => (IdentityKind::Explicit, 0.8),
            };
            let source = format!(
                "指代轨迹#{}（第 {} 轮，{}）",
                event.id,
                event.turn,
                event.source.label()
            );
            if self.add_identity_edge(
                IdentityEdge::new(&surface, &referent, kind, &referent)
                    .with_source(&source)
                    .with_confidence(confidence),
            ) {
                added += 1;
            }
        }
        added
    }

    pub fn from_registry(registry: &EntityRegistry) -> Self {
        let mut graph = Self::new();
        graph.absorb_registry(registry);
        graph
    }

    pub fn from_variant_closure(closure: &VariantClosure, context: &VariantContext) -> Self {
        let mut graph = Self::new();
        graph.absorb_variant_closure(closure, context);
        graph
    }

    pub fn from_bindings(ledger: &BindingLedger) -> Self {
        let mut graph = Self::new();
        graph.absorb_bindings(ledger);
        graph
    }

    pub fn from_referent_trail(trail: &ReferentTrail) -> Self {
        let mut graph = Self::new();
        graph.absorb_referent_trail(trail);
        graph
    }

    fn node_types<'a>(&'a self, surface: &str) -> &'a BTreeSet<String> {
        match self.nodes.get(&fold_surface(surface)) {
            Some(node) => &node.types,
            None => {
                static EMPTY: std::sync::OnceLock<BTreeSet<String>> = std::sync::OnceLock::new();
                EMPTY.get_or_init(BTreeSet::new)
            }
        }
    }

    fn pair_conflict(&self, left: &BTreeSet<String>, right: &BTreeSet<String>) -> bool {
        if left.is_empty() || right.is_empty() {
            return false;
        }
        match &self.types {
            Some(types) => matches!(types.sets_admit(left, right), TypeVerdict::Disjoint),
            // 没有类型格时按精确标签判定：两边都有类型且无交集，视为冲突。
            None => !left.iter().any(|a| right.iter().any(|b| a == b)),
        }
    }

    fn class_type_conflict(&self, members: &[String]) -> Option<String> {
        for (index, left) in members.iter().enumerate() {
            let left_types = self.node_types(left);
            if left_types.is_empty() {
                continue;
            }
            for right in members.iter().skip(index + 1) {
                let right_types = self.node_types(right);
                if self.pair_conflict(left_types, right_types) {
                    return Some(format!(
                        "「{}」类型 {} 与「{}」类型 {} 不相容",
                        left,
                        left_types.iter().cloned().collect::<Vec<_>>().join("、"),
                        right,
                        right_types.iter().cloned().collect::<Vec<_>>().join("、")
                    ));
                }
            }
        }
        None
    }

    fn choose_canonical(&self, members: &[String], edges: &[&IdentityEdge]) -> String {
        let declared: Vec<&String> = members
            .iter()
            .filter(|member| {
                self.nodes
                    .get(*member)
                    .and_then(|node| node.canonical_order)
                    .is_some()
            })
            .collect();
        if !declared.is_empty() {
            return declared
                .into_iter()
                .min_by_key(|member| {
                    (
                        self.nodes
                            .get(*member)
                            .and_then(|node| node.canonical_order)
                            .unwrap_or(usize::MAX),
                        (*member).clone(),
                    )
                })
                .cloned()
                .unwrap_or_else(|| members[0].clone());
        }
        let anchor_hits = |member: &str| edges.iter().filter(|edge| edge.anchor == member).count();
        let outgoing = |member: &str| {
            edges
                .iter()
                .filter(|edge| edge.from == member && edge.to != member)
                .count()
        };
        let degree = |member: &str| {
            edges
                .iter()
                .filter(|edge| edge.from == member || edge.to == member)
                .count()
        };
        let mut best = members[0].clone();
        let mut best_hits = anchor_hits(&best);
        let mut best_outgoing = outgoing(&best);
        let mut best_degree = degree(&best);
        for member in members.iter().skip(1) {
            let hits = anchor_hits(member);
            let out = outgoing(member);
            let deg = degree(member);
            let better = hits > best_hits
                || (hits == best_hits
                    && (out < best_outgoing
                        || (out == best_outgoing
                            && (deg > best_degree || (deg == best_degree && member < &best)))));
            if better {
                best = member.clone();
                best_hits = hits;
                best_outgoing = out;
                best_degree = deg;
            }
        }
        best
    }

    /// 构建身份闭包视图。纯只读；同样的输入与预算得到同样的字节。
    pub fn build(&self, budget: &CorefBudget) -> CorefBuild {
        let mut rejected = Vec::new();
        let mut filtered: Vec<&IdentityEdge> = Vec::new();
        for edge in &self.identity_edges {
            if edge.status != IdentityEdgeStatus::Active {
                continue;
            }
            if edge.confidence < budget.min_confidence {
                rejected.push(CorefRejection {
                    from: edge.from.clone(),
                    to: edge.to.clone(),
                    kind: edge.kind,
                    source: edge.source.clone(),
                    reason: CorefRejectionReason::LowConfidence,
                    detail: format!(
                        "置信度 {:.2} < 门槛 {:.2}",
                        edge.confidence, budget.min_confidence
                    ),
                });
                continue;
            }
            if !validity_allows(edge.validity, budget.at) {
                rejected.push(CorefRejection {
                    from: edge.from.clone(),
                    to: edge.to.clone(),
                    kind: edge.kind,
                    source: edge.source.clone(),
                    reason: CorefRejectionReason::InvalidValidity,
                    detail: format!(
                        "有效期 {} 不覆盖查询时刻 {}",
                        edge.validity.render(),
                        budget
                            .at
                            .map(|point| point.render())
                            .unwrap_or_else(|| "未指定".to_string())
                    ),
                });
                continue;
            }
            if self.pair_conflict(self.node_types(&edge.from), self.node_types(&edge.to)) {
                rejected.push(CorefRejection {
                    from: edge.from.clone(),
                    to: edge.to.clone(),
                    kind: edge.kind,
                    source: edge.source.clone(),
                    reason: CorefRejectionReason::TypeConflict,
                    detail: format!(
                        "「{}」类型 {} 与「{}」类型 {} 不相容",
                        edge.from,
                        self.node_types(&edge.from)
                            .iter()
                            .cloned()
                            .collect::<Vec<_>>()
                            .join("、"),
                        edge.to,
                        self.node_types(&edge.to)
                            .iter()
                            .cloned()
                            .collect::<Vec<_>>()
                            .join("、")
                    ),
                });
                continue;
            }
            filtered.push(edge);
        }
        filtered.sort_by(|left, right| {
            left.from
                .cmp(&right.from)
                .then_with(|| left.to.cmp(&right.to))
                .then_with(|| left.kind.cmp(&right.kind))
                .then_with(|| left.anchor.cmp(&right.anchor))
                .then_with(|| left.source.cmp(&right.source))
                .then_with(|| {
                    left.confidence
                        .partial_cmp(&right.confidence)
                        .unwrap_or(Ordering::Equal)
                })
        });
        let mut truncated = None;
        if filtered.len() > budget.max_edges {
            filtered.truncate(budget.max_edges);
            truncated = Some(CorefTruncation::Edges);
        }

        let mut relevant: BTreeSet<String> = self
            .nodes
            .iter()
            .filter(|(_, node)| node.canonical_order.is_some())
            .map(|(surface, _)| surface.clone())
            .collect();
        for edge in &filtered {
            relevant.insert(edge.from.clone());
            relevant.insert(edge.to.clone());
        }
        let mut all_nodes = relevant.clone();
        for edge in &filtered {
            all_nodes.insert(edge.anchor.clone());
        }

        let mut ambiguous_nodes: BTreeSet<String> = BTreeSet::new();
        let mut ambiguous_candidates: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        let mut active: Vec<&IdentityEdge> = filtered.clone();
        loop {
            let found = detect_ambiguous(&all_nodes, &active);
            let mut newly: Vec<String> = found
                .keys()
                .filter(|node| !ambiguous_nodes.contains(*node))
                .cloned()
                .collect();
            if newly.is_empty() {
                break;
            }
            newly.sort();
            for node in newly {
                if let Some(candidates) = found.get(&node) {
                    ambiguous_candidates
                        .entry(node.clone())
                        .or_default()
                        .extend(candidates.iter().cloned());
                }
                ambiguous_nodes.insert(node);
            }
            active.retain(|edge| {
                !ambiguous_nodes.contains(&edge.from) && !ambiguous_nodes.contains(&edge.to)
            });
        }

        let mut uf = UnionFind::new(&all_nodes);
        for edge in &active {
            uf.union(&edge.from, &edge.to);
        }
        let mut groups: BTreeMap<usize, Vec<String>> = BTreeMap::new();
        for node in &relevant {
            if ambiguous_nodes.contains(node) {
                continue;
            }
            let Some(root) = uf.find(node) else {
                continue;
            };
            groups.entry(root).or_default().push(node.clone());
        }

        let mut adjacency: BTreeMap<String, Vec<(String, &IdentityEdge, bool)>> = BTreeMap::new();
        for edge in &active {
            adjacency
                .entry(edge.from.clone())
                .or_default()
                .push((edge.to.clone(), edge, false));
            adjacency
                .entry(edge.to.clone())
                .or_default()
                .push((edge.from.clone(), edge, true));
        }
        for neighbours in adjacency.values_mut() {
            neighbours.sort_by(|left, right| {
                left.0
                    .cmp(&right.0)
                    .then_with(|| left.1.kind.cmp(&right.1.kind))
                    .then_with(|| left.1.source.cmp(&right.1.source))
                    .then_with(|| {
                        left.1
                            .confidence
                            .partial_cmp(&right.1.confidence)
                            .unwrap_or(Ordering::Equal)
                    })
                    .then_with(|| left.2.cmp(&right.2))
            });
        }

        let mut classes = Vec::new();
        let mut conflicts = Vec::new();
        for (_, mut members) in groups {
            members.sort();
            members.dedup();
            if members.is_empty() {
                continue;
            }
            let singleton_canonical = members.len() == 1
                && self
                    .nodes
                    .get(&members[0])
                    .and_then(|node| node.canonical_order)
                    .is_some();
            if members.len() == 1 && !singleton_canonical {
                continue;
            }
            let class_edges: Vec<&IdentityEdge> = active
                .iter()
                .copied()
                .filter(|edge| members.contains(&edge.from) && members.contains(&edge.to))
                .collect();
            let canonical = self.choose_canonical(&members, &class_edges);

            let member_set: BTreeSet<&String> = members.iter().collect();
            let mut witnesses: BTreeMap<String, Vec<CorefStep>> = BTreeMap::new();
            witnesses.insert(canonical.clone(), Vec::new());
            let mut visited: BTreeSet<String> = BTreeSet::from([canonical.clone()]);
            let mut queue: VecDeque<(String, Vec<CorefStep>)> =
                VecDeque::from([(canonical.clone(), Vec::new())]);
            let mut witness_truncated = false;
            while let Some((node, path)) = queue.pop_front() {
                let neighbours = adjacency.get(&node);
                if path.len() >= budget.max_hops {
                    if neighbours.is_some_and(|list| {
                        list.iter().any(|(next, _, _)| {
                            member_set.contains(next) && !visited.contains(next)
                        })
                    }) {
                        witness_truncated = true;
                    }
                    continue;
                }
                let Some(neighbours) = neighbours else {
                    continue;
                };
                for (next, edge, reversed) in neighbours {
                    if !member_set.contains(next) || visited.contains(next) {
                        continue;
                    }
                    if visited.len() >= budget.max_nodes {
                        witness_truncated = true;
                        continue;
                    }
                    visited.insert(next.clone());
                    let mut next_path = path.clone();
                    next_path.push(CorefStep {
                        from: node.clone(),
                        to: next.clone(),
                        kind: edge.kind,
                        source: edge.source.clone(),
                        confidence: edge.confidence,
                        reversed: *reversed,
                    });
                    witnesses.insert(next.clone(), next_path.clone());
                    queue.push_back((next.clone(), next_path));
                }
            }
            if members.iter().any(|member| !witnesses.contains_key(member)) {
                witness_truncated = true;
            }

            let mut validity = class_edges
                .first()
                .map(|edge| edge.validity)
                .unwrap_or(Validity::Unknown);
            for edge in &class_edges {
                validity = validity.intersection(edge.validity);
            }
            let mut types = BTreeSet::new();
            for member in &members {
                types.extend(self.node_types(member).iter().cloned());
            }
            if !validity.is_satisfiable() {
                conflicts.push(CorefConflict {
                    members: members.clone(),
                    reason: CorefConflictReason::ValidityConflict,
                    detail: format!("有效期交集为 {}", validity.render()),
                });
                continue;
            }
            if let Some(detail) = self.class_type_conflict(&members) {
                conflicts.push(CorefConflict {
                    members: members.clone(),
                    reason: CorefConflictReason::TypeConflict,
                    detail,
                });
                continue;
            }
            let confidence = class_edges
                .iter()
                .map(|edge| edge.confidence)
                .fold(1.0f64, f64::min);
            classes.push(CorefClass {
                canonical,
                members,
                types,
                confidence,
                validity,
                witnesses,
                witness_truncated,
            });
        }
        classes.sort_by(|left, right| left.canonical.cmp(&right.canonical));

        let mut canonical_of = BTreeMap::new();
        for class in &classes {
            for member in &class.members {
                canonical_of.insert(member.clone(), class.canonical.clone());
            }
        }
        let ambiguous = ambiguous_candidates
            .into_iter()
            .map(|(surface, candidates)| {
                let mapped: BTreeSet<String> = candidates
                    .into_iter()
                    .map(|candidate| canonical_of.get(&candidate).cloned().unwrap_or(candidate))
                    .collect();
                (surface, mapped.into_iter().collect())
            })
            .collect();

        CorefBuild {
            classes,
            canonical_of,
            ambiguous,
            rejected,
            conflicts,
            nodes: self.nodes.len(),
            active_edges: active.len(),
            truncated,
        }
    }

    /// 按身份等价类做商，再保留关系边。自环只作为证据，不参与遍历。
    pub fn quotient(&self, budget: &CorefBudget) -> QuotientGraph {
        let identity = self.build(budget);
        let mut edges: BTreeMap<String, BTreeMap<String, BTreeSet<String>>> = BTreeMap::new();
        let mut self_loops = Vec::new();
        let mut self_loop_keys: BTreeSet<(String, String, String, String)> = BTreeSet::new();
        let mut relation_edges = 0;
        let mut dropped = 0;
        for edge in &self.relation_edges {
            if edge.confidence < budget.min_confidence || !validity_allows(edge.validity, budget.at)
            {
                dropped += 1;
                continue;
            }
            relation_edges += 1;
            let from = identity
                .canonical(&edge.from)
                .unwrap_or(&edge.from)
                .to_string();
            let to = identity.canonical(&edge.to).unwrap_or(&edge.to).to_string();
            if from == to {
                let key = (
                    from.clone(),
                    edge.from.clone(),
                    edge.to.clone(),
                    edge.predicate.clone(),
                );
                if self_loop_keys.insert(key) {
                    self_loops.push(SelfLoopEvidence {
                        node: from,
                        from: edge.from.clone(),
                        to: edge.to.clone(),
                        predicate: edge.predicate.clone(),
                    });
                }
                continue;
            }
            edges
                .entry(from)
                .or_default()
                .entry(to)
                .or_default()
                .insert(edge.predicate.clone());
        }
        self_loops.sort_by(|left, right| {
            left.node
                .cmp(&right.node)
                .then_with(|| left.predicate.cmp(&right.predicate))
                .then_with(|| left.from.cmp(&right.from))
                .then_with(|| left.to.cmp(&right.to))
        });
        QuotientGraph {
            identity,
            edges,
            self_loops,
            relation_edges,
            dropped,
        }
    }

    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|error| error.to_string())
    }

    pub fn from_json(text: &str) -> Result<Self, String> {
        serde_json::from_str(text).map_err(|error| error.to_string())
    }
}

fn validity_allows(validity: Validity, at: Option<TimePoint>) -> bool {
    match (validity, at) {
        (Validity::Never, _) => false,
        (_, None) => true,
        (Validity::Unknown, Some(_)) => true,
        (Validity::Unbounded, _) => true,
        (Validity::Span(span), Some(point)) => span.contains(point),
    }
}

/// 命令行/快照友好的证据输入：字段名短，缺省值明确。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceIdentity {
    pub from: String,
    pub to: String,
    #[serde(default)]
    pub kind: IdentityKind,
    #[serde(default)]
    pub anchor: Option<String>,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub confidence: Option<f64>,
    #[serde(default)]
    pub validity: Option<Validity>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceNonIdentity {
    pub from: String,
    pub to: String,
    #[serde(default)]
    pub kind: NonIdentityKind,
    #[serde(default)]
    pub source: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceRelation {
    pub from: String,
    pub to: String,
    pub predicate: String,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub confidence: Option<f64>,
    #[serde(default)]
    pub validity: Option<Validity>,
}

/// 一组可序列化的证据，直接转成 [`CorefGraph`]。
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CorefEvidence {
    #[serde(default)]
    pub identity: Vec<EvidenceIdentity>,
    #[serde(default)]
    pub non_identity: Vec<EvidenceNonIdentity>,
    #[serde(default)]
    pub relations: Vec<EvidenceRelation>,
}

impl CorefEvidence {
    pub fn from_json(text: &str) -> Result<Self, String> {
        serde_json::from_str(text).map_err(|error| error.to_string())
    }

    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|error| error.to_string())
    }

    pub fn into_graph(self) -> CorefGraph {
        let mut graph = CorefGraph::new();
        for evidence in self.identity {
            let anchor = evidence.anchor.as_deref().unwrap_or(&evidence.to);
            let mut edge = IdentityEdge::new(&evidence.from, &evidence.to, evidence.kind, anchor);
            if let Some(source) = &evidence.source {
                edge = edge.with_source(source);
            }
            if let Some(confidence) = evidence.confidence {
                edge = edge.with_confidence(confidence);
            }
            if let Some(validity) = evidence.validity {
                edge = edge.with_validity(validity);
            }
            graph.add_identity_edge(edge);
        }
        for evidence in self.non_identity {
            let source = evidence.source.as_deref().unwrap_or("证据输入");
            graph.add_non_identity_edge(NonIdentityEdge::new(
                &evidence.from,
                &evidence.to,
                evidence.kind,
                source,
            ));
        }
        for evidence in self.relations {
            let mut edge = RelationEdge::new(&evidence.from, &evidence.to, &evidence.predicate);
            if let Some(source) = &evidence.source {
                edge = edge.with_source(source);
            }
            if let Some(confidence) = evidence.confidence {
                edge = edge.with_confidence(confidence);
            }
            if let Some(validity) = evidence.validity {
                edge = edge.with_validity(validity);
            }
            graph.add_relation_edge(edge);
        }
        graph
    }
}

/// 更正回执：受影响的旧等价类要由上层标记 stale。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorrectionOutcome {
    pub surface: String,
    pub from: String,
    pub to: String,
    pub expired_edges: usize,
    pub affected_classes: Vec<String>,
    pub edge_added: bool,
}

impl CorrectionOutcome {
    pub fn render(&self) -> String {
        let mut out = format!(
            "「{}」从「{}」改指「{}」：撤销 {} 条恒等边{}",
            self.surface,
            self.from,
            self.to,
            self.expired_edges,
            if self.edge_added {
                "，并写入一条更正边"
            } else {
                "，未新增边"
            }
        );
        if !self.affected_classes.is_empty() {
            out.push_str(&format!(
                "；受影响的等价类：{}",
                self.affected_classes.join("、")
            ));
        }
        out
    }
}

/// 忘记回执。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForgetOutcome {
    pub surface: String,
    pub expired_edges: usize,
    pub affected_classes: Vec<String>,
}

impl ForgetOutcome {
    pub fn render(&self) -> String {
        let mut out = format!(
            "已忘记「{}」：撤销 {} 条恒等边",
            self.surface, self.expired_edges
        );
        if !self.affected_classes.is_empty() {
            out.push_str(&format!(
                "；受影响的等价类：{}",
                self.affected_classes.join("、")
            ));
        }
        out
    }
}

/// 商图中坍缩成自环的关系边：保留为证据，不参与传递遍历。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SelfLoopEvidence {
    pub node: String,
    pub from: String,
    pub to: String,
    pub predicate: String,
}

impl SelfLoopEvidence {
    pub fn render(&self) -> String {
        format!(
            "「{}」内的「{}」—[{}]→「{}」（自环，仅证据）",
            self.node, self.from, self.predicate, self.to
        )
    }
}

/// 商图：节点是等价类，边是关系边。无别名时行为与直接遍历一致。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuotientGraph {
    pub identity: CorefBuild,
    /// 规范实体 → 规范实体 → 谓词集合。
    #[serde(default)]
    pub edges: BTreeMap<String, BTreeMap<String, BTreeSet<String>>>,
    #[serde(default)]
    pub self_loops: Vec<SelfLoopEvidence>,
    pub relation_edges: usize,
    pub dropped: usize,
}

impl QuotientGraph {
    pub fn canonical(&self, surface: &str) -> String {
        let key = fold_surface(surface);
        self.identity.canonical(&key).unwrap_or(&key).to_string()
    }

    pub fn neighbors(&self, surface: &str) -> Vec<String> {
        self.neighbors_for(surface, None)
    }

    pub fn neighbors_for(&self, surface: &str, predicate: Option<&str>) -> Vec<String> {
        let canonical = self.canonical(surface);
        let Some(targets) = self.edges.get(&canonical) else {
            return Vec::new();
        };
        targets
            .iter()
            .filter(|(_, predicates)| predicate.is_none_or(|wanted| predicates.contains(wanted)))
            .map(|(target, _)| target.clone())
            .collect()
    }

    pub fn closure(
        &self,
        start: &str,
        operator: ClosureOp,
        budget: ClosureBudget,
        law: Option<&str>,
        predicate: Option<&str>,
    ) -> Result<QuotientOutcome, ClosureError> {
        let start_surface = fold_surface(start);
        let start_canonical = self.canonical(&start_surface);
        let edges = &self.edges;
        let outcome = evaluate_closure(&start_canonical, operator, budget, law, |node| {
            edges
                .get(node)
                .map(|targets| {
                    targets
                        .iter()
                        .filter(|(_, predicates)| {
                            predicate.is_none_or(|wanted| predicates.contains(wanted))
                        })
                        .map(|(target, _)| target.clone())
                        .collect()
                })
                .unwrap_or_default()
        })?;
        let mut hops: BTreeMap<String, Vec<QuotientHop>> = BTreeMap::new();
        for (target, path) in &outcome.witnesses {
            let mut target_hops = Vec::new();
            for pair in path.windows(2) {
                let Some(predicates) = edges
                    .get(&pair[0])
                    .and_then(|targets| targets.get(&pair[1]))
                else {
                    continue;
                };
                target_hops.push(QuotientHop {
                    from: pair[0].clone(),
                    to: pair[1].clone(),
                    predicates: predicates.clone(),
                });
            }
            hops.insert(target.clone(), target_hops);
        }
        let start_witness = self.identity.resolve(&start_surface).witness;
        Ok(QuotientOutcome {
            start_surface,
            start: start_canonical,
            start_witness,
            outcome,
            hops,
        })
    }

    pub fn report(&self) -> String {
        let mut out = format!(
            "商图：{} 个规范实体，{} 条关系边（丢弃 {} 条），{} 条自环证据",
            self.identity.classes.len(),
            self.relation_edges,
            self.dropped,
            self.self_loops.len()
        );
        for (from, targets) in &self.edges {
            for (to, predicates) in targets {
                out.push_str(&format!(
                    "\n  「{}」—[{}]→「{}」",
                    from,
                    predicates.iter().cloned().collect::<Vec<_>>().join("/"),
                    to
                ));
            }
        }
        for evidence in &self.self_loops {
            out.push_str(&format!("\n  {}", evidence.render()));
        }
        out
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuotientHop {
    pub from: String,
    pub to: String,
    pub predicates: BTreeSet<String>,
}

/// 商图闭包结果：关系见证 + 起点身份见证。
#[derive(Debug, Clone, PartialEq)]
pub struct QuotientOutcome {
    pub start_surface: String,
    pub start: String,
    pub start_witness: Vec<CorefStep>,
    pub outcome: ClosureOutcome,
    pub hops: BTreeMap<String, Vec<QuotientHop>>,
}

impl QuotientOutcome {
    pub fn targets(&self) -> Vec<&str> {
        self.outcome
            .targets_by_distance()
            .into_iter()
            .map(|(target, _)| target)
            .collect()
    }

    pub fn render_witness(&self, target: &str) -> Option<String> {
        let path = self.outcome.witnesses.get(target)?;
        let mut text = if self.start_surface != self.start {
            format!(
                "「{}」{}",
                self.start_surface,
                render_steps(&reverse_steps(&self.start_witness))
            )
        } else {
            format!("「{}」", self.start)
        };
        if path.is_empty() {
            return Some(text);
        }
        let hops = self.hops.get(target)?;
        for hop in hops {
            text.push_str(&format!(
                " —[{}]→ 「{}」",
                hop.predicates.iter().cloned().collect::<Vec<_>>().join("/"),
                hop.to
            ));
        }
        Some(text)
    }

    pub fn render(&self) -> String {
        let mut out = format!(
            "{}闭包：从「{}」到达 {} 个目标（扩展 {} 次{}{}）",
            self.outcome.operator.label(),
            self.start_surface,
            self.outcome.reached_count,
            self.outcome.expansions,
            self.outcome
                .truncated
                .map(|truncated| format!("，截断：{}", truncated.label()))
                .unwrap_or_default(),
            if self.outcome.witnesses_truncated {
                "，见证截断"
            } else {
                ""
            }
        );
        for (target, hops) in self.outcome.targets_by_distance() {
            out.push_str(&format!("\n  • {}（{} 跳）", target, hops));
            if let Some(witness) = self.render_witness(target) {
                out.push_str(&format!("\n      {}", witness));
            }
        }
        out
    }
}

/// 找出「同一词面同时指向多个互不相连锚点」的节点。
///
/// 只看该词面的出边：一个词面被声明成多个实体的别名才是歧义；入边只是
/// 别人把它当作目标，属于正常的合并证据。检测时移除该节点的全部关联边，
/// 避免歧义词面自己把两个锚点「连」在一起。
fn detect_ambiguous(
    all_nodes: &BTreeSet<String>,
    active: &[&IdentityEdge],
) -> BTreeMap<String, BTreeSet<String>> {
    let candidates: BTreeSet<&String> = active.iter().map(|edge| &edge.from).collect();
    let mut result = BTreeMap::new();
    for node in candidates {
        let anchors: BTreeSet<&String> = active
            .iter()
            .filter(|edge| &edge.from == node)
            .map(|edge| &edge.anchor)
            .collect();
        if anchors.len() <= 1 {
            continue;
        }
        let mut uf = UnionFind::new(all_nodes);
        for edge in active {
            if &edge.from != node && &edge.to != node {
                uf.union(&edge.from, &edge.to);
            }
        }
        let mut roots = BTreeSet::new();
        for anchor in &anchors {
            if let Some(root) = uf.find(anchor.as_str()) {
                roots.insert(root);
            }
        }
        if roots.len() > 1 {
            let mut names = BTreeSet::new();
            for anchor in anchors {
                if let Some(root) = uf.find(anchor.as_str()) {
                    if let Some(name) = uf.name_of(root) {
                        names.insert(name);
                    }
                }
            }
            result.insert(node.clone(), names);
        }
    }
    result
}

/// 并查集：只用于构建查询视图，不写回任何账本。
struct UnionFind {
    index: BTreeMap<String, usize>,
    parent: Vec<usize>,
}

impl UnionFind {
    fn new(nodes: &BTreeSet<String>) -> Self {
        let names: Vec<String> = nodes.iter().cloned().collect();
        let index = names
            .iter()
            .enumerate()
            .map(|(position, name)| (name.clone(), position))
            .collect();
        Self {
            index,
            parent: (0..names.len()).collect(),
        }
    }

    fn find(&mut self, node: &str) -> Option<usize> {
        let position = *self.index.get(node)?;
        Some(self.find_position(position))
    }

    fn find_position(&mut self, mut position: usize) -> usize {
        while self.parent[position] != position {
            self.parent[position] = self.parent[self.parent[position]];
            position = self.parent[position];
        }
        position
    }

    fn union(&mut self, left: &str, right: &str) {
        let (Some(left), Some(right)) = (
            self.index.get(left).copied(),
            self.index.get(right).copied(),
        ) else {
            return;
        };
        let left_root = self.find_position(left);
        let right_root = self.find_position(right);
        if left_root != right_root {
            self.parent[right_root] = left_root;
        }
    }

    fn name_of(&self, root: usize) -> Option<String> {
        self.index
            .iter()
            .find(|(_, position)| **position == root)
            .map(|(name, _)| name.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contextual_reference::BindingScope;
    use crate::entity_registry::NominationOrigin;
    use crate::variant_closure::VariantEdge;

    fn edge(from: &str, to: &str, kind: IdentityKind, anchor: &str) -> IdentityEdge {
        IdentityEdge::new(from, to, kind, anchor)
    }

    #[test]
    fn identity_closure_is_transitive_and_keeps_shortest_witnesses() {
        let mut graph = CorefGraph::new();
        graph.add_identity_edge(edge("老张", "张三", IdentityKind::Alias, "张三"));
        graph.add_identity_edge(edge("张三", "张老师", IdentityKind::Honorific, "张三"));
        graph.add_identity_edge(edge("张老师", "他", IdentityKind::Pronoun, "张三"));

        let build = graph.build(&CorefBudget::default());
        assert_eq!(build.canonical("老张"), Some("张三"));
        assert_eq!(build.canonical("张老师"), Some("张三"));
        assert_eq!(build.canonical("他"), Some("张三"));

        let class = build.class_of("老张").expect("应有等价类");
        assert_eq!(class.members, vec!["他", "张三", "张老师", "老张"]);
        let resolution = build.resolve("他");
        assert_eq!(resolution.status, CorefStatus::Alias);
        assert_eq!(resolution.canonical.as_deref(), Some("张三"));
        assert!(!resolution.witness.is_empty());
        let rendered = resolution.render();
        assert!(rendered.contains("他"));
        assert!(rendered.contains("张三"));
        assert!(rendered.contains("代词"));
    }

    #[test]
    fn non_identity_edges_never_merge_identities() {
        let mut graph = CorefGraph::new();
        graph.describe("姚明的妻子", "叶莉", NonIdentityKind::Description);
        graph.relate("姚明", "叶莉", "妻子");

        let build = graph.build(&CorefBudget::default());
        assert!(build.classes.is_empty());
        assert_eq!(build.resolve("姚明的妻子").status, CorefStatus::Unknown);
        assert_eq!(build.resolve("叶莉").status, CorefStatus::Unknown);
        assert_eq!(graph.non_identity_edges().len(), 1);
        assert!(graph.non_identity_edges()[0].render().contains("不传递"));
    }

    #[test]
    fn conflicting_types_block_merge_until_type_system_says_compatible() {
        let mut graph = CorefGraph::new();
        graph.declare_type("苹果", "水果");
        graph.declare_type("苹果公司", "组织");
        graph.link("苹果", "苹果公司", IdentityKind::Alias, "苹果公司");

        let build = graph.build(&CorefBudget::default());
        assert!(build.canonical("苹果").is_none());
        assert_eq!(build.rejected.len(), 1);
        assert_eq!(build.rejected[0].reason, CorefRejectionReason::TypeConflict);

        let mut types = TypeSystem::new();
        types.declare_alias("水果", "食物");
        types.declare_alias("组织", "机构");
        let mut compatible = CorefGraph::new().with_type_system(types);
        compatible.declare_type("苹果", "水果");
        compatible.declare_type("苹果公司", "组织");
        compatible.link("苹果", "苹果公司", IdentityKind::Alias, "苹果公司");
        let build = compatible.build(&CorefBudget::default());
        assert_eq!(build.canonical("苹果"), Some("苹果公司"));
    }

    #[test]
    fn ambiguous_surface_is_quarantined_instead_of_merged() {
        let mut graph = CorefGraph::new();
        graph.link("苹果", "苹果公司", IdentityKind::Alias, "苹果公司");
        graph.link("苹果", "苹果果实", IdentityKind::Alias, "苹果果实");

        let build = graph.build(&CorefBudget::default());
        let resolution = build.resolve("苹果");
        assert_eq!(resolution.status, CorefStatus::Ambiguous);
        assert_eq!(
            resolution.candidates,
            vec!["苹果公司".to_string(), "苹果果实".to_string()]
        );
        assert!(build.canonical("苹果").is_none());
        assert!(build.canonical("苹果公司").is_none());
        assert!(build.canonical("苹果果实").is_none());
        assert!(resolution.render().contains("不自动合并"));
    }

    #[test]
    fn correction_moves_only_the_surface_and_reports_affected_classes() {
        let mut graph = CorefGraph::new();
        graph.link("老李", "李四", IdentityKind::Alias, "李四");
        graph.link("老李头", "李四", IdentityKind::Alias, "李四");

        let before = graph.build(&CorefBudget::default());
        assert_eq!(before.canonical("老李"), Some("李四"));

        let outcome = graph
            .correct("老李", "李四", "王五", "用户更正")
            .expect("更正应成功");
        assert_eq!(outcome.affected_classes, vec!["李四".to_string()]);
        assert_eq!(outcome.expired_edges, 1);
        assert!(outcome.edge_added);

        let after = graph.build(&CorefBudget::default());
        assert_eq!(after.canonical("老李"), Some("王五"));
        assert_eq!(after.canonical("老李头"), Some("李四"));
        assert_eq!(
            after.resolve("老李").witness[0].kind,
            IdentityKind::Correction
        );
    }

    #[test]
    fn correction_rejects_using_the_old_entity_as_the_surface() {
        let mut graph = CorefGraph::new();
        graph.link("老李", "李四", IdentityKind::Alias, "李四");
        let error = graph
            .correct("李四", "李四", "王五", "用户更正")
            .expect_err("规范名不能被当成更正词面");
        assert!(error.contains("同名"));
    }

    #[test]
    fn forget_expires_incident_edges_and_reports_affected_class() {
        let mut graph = CorefGraph::new();
        graph.link("老李", "李四", IdentityKind::Alias, "李四");
        let outcome = graph.forget("老李");
        assert_eq!(outcome.expired_edges, 1);
        assert_eq!(outcome.affected_classes, vec!["李四".to_string()]);
        let build = graph.build(&CorefBudget::default());
        assert_eq!(build.resolve("老李").status, CorefStatus::Unknown);
    }

    #[test]
    fn quotient_closure_traverses_aliases_but_ignores_collapsed_self_loops() {
        let mut graph = CorefGraph::new();
        graph.link("老李", "李四", IdentityKind::Alias, "李四");
        graph.relate("李四", "王五", "师傅");
        graph.relate("王五", "赵六", "师傅");
        graph.relate("老李", "李四", "同事");

        let quotient = graph.quotient(&CorefBudget::default());
        assert_eq!(quotient.canonical("老李"), "李四");
        assert_eq!(quotient.self_loops.len(), 1);
        assert_eq!(quotient.self_loops[0].node, "李四");

        let outcome = quotient
            .closure(
                "老李",
                ClosureOp::Plus,
                ClosureBudget::default(),
                Some("law:transitive:师傅"),
                Some("师傅"),
            )
            .expect("有传递律");
        let targets = outcome.targets();
        assert_eq!(targets, vec!["王五", "赵六"]);
        assert!(!targets.contains(&"李四"));
        let witness = outcome.render_witness("赵六").expect("应有见证");
        assert!(witness.contains("老李"));
        assert!(witness.contains("李四"));
        assert!(witness.contains("师傅"));
    }

    #[test]
    fn quotient_without_aliases_matches_plain_closure() {
        let mut graph = CorefGraph::new();
        graph.relate("甲", "乙", "上司");
        graph.relate("乙", "丙", "上司");
        let quotient = graph.quotient(&CorefBudget::default());
        let outcome = quotient
            .closure(
                "甲",
                ClosureOp::Plus,
                ClosureBudget::default(),
                Some("law:transitive:上司"),
                None,
            )
            .expect("有传递律");

        let raw = evaluate_closure(
            "甲",
            ClosureOp::Plus,
            ClosureBudget::default(),
            Some("law:transitive:上司"),
            |node| match node {
                "甲" => vec!["乙".to_string()],
                "乙" => vec!["丙".to_string()],
                _ => Vec::new(),
            },
        )
        .expect("有传递律");
        assert_eq!(
            outcome
                .outcome
                .witnesses
                .keys()
                .cloned()
                .collect::<Vec<_>>(),
            raw.witnesses.keys().cloned().collect::<Vec<_>>()
        );
        assert!(outcome
            .render_witness("丙")
            .expect("应有见证")
            .contains("上司"));
    }

    #[test]
    fn budget_filters_low_confidence_and_expired_validity() {
        let mut graph = CorefGraph::new();
        graph.add_identity_edge(
            edge("老张", "张三", IdentityKind::Alias, "张三").with_confidence(0.1),
        );
        let build = graph.build(&CorefBudget::default());
        assert_eq!(build.rejected.len(), 1);
        assert_eq!(
            build.rejected[0].reason,
            CorefRejectionReason::LowConfidence
        );

        let mut dated = CorefGraph::new();
        dated.add_identity_edge(
            edge("老张", "张三", IdentityKind::Alias, "张三")
                .with_validity(Validity::from_years(2000, 2010)),
        );
        let inside = dated.build(&CorefBudget::default().at_year(2005));
        assert_eq!(inside.canonical("老张"), Some("张三"));
        let outside = dated.build(&CorefBudget::default().at_year(2020));
        assert!(outside.canonical("老张").is_none());
        assert_eq!(
            outside.rejected[0].reason,
            CorefRejectionReason::InvalidValidity
        );
    }

    #[test]
    fn witness_budget_truncates_without_losing_class_membership() {
        let mut graph = CorefGraph::new();
        graph.link("甲", "乙", IdentityKind::Alias, "甲");
        graph.link("乙", "丙", IdentityKind::Alias, "甲");
        graph.link("丙", "丁", IdentityKind::Alias, "甲");

        let build = graph.build(&CorefBudget::default().with_max_hops(1));
        assert_eq!(build.canonical("丁"), Some("甲"));
        let class = build.class_of("甲").expect("应有等价类");
        assert!(class.witness_truncated);
        let resolution = build.resolve("丁");
        assert!(resolution.witness.is_empty());
        assert!(resolution.witness_truncated);
        assert!(resolution.render().contains("截断"));
    }

    #[test]
    fn class_level_type_conflict_is_reported_and_not_merged() {
        let mut graph = CorefGraph::new();
        graph.declare_type("甲", "水果");
        graph.declare_type("丙", "组织");
        graph.link("甲", "乙", IdentityKind::Alias, "甲");
        graph.link("乙", "丙", IdentityKind::Alias, "甲");

        let build = graph.build(&CorefBudget::default());
        assert!(build.canonical("甲").is_none());
        assert_eq!(build.conflicts.len(), 1);
        assert_eq!(build.conflicts[0].reason, CorefConflictReason::TypeConflict);
        assert!(build.report().contains("类内类型冲突"));
    }

    #[test]
    fn registry_variants_and_bindings_share_one_closure() {
        let mut registry = EntityRegistry::new();
        registry.confirm_alias_with_origin("老张", "张三", NominationOrigin::Dialogue);
        registry.confirm_alias_with_origin("张老师", "张三", NominationOrigin::Dialogue);

        let mut ledger = BindingLedger::new();
        ledger.begin_turn();
        ledger
            .bind("他", "张三", BindingScope::Topic, "测试绑定")
            .expect("绑定应成功");

        let mut graph = CorefGraph::from_registry(&registry);
        graph.absorb_bindings(&ledger);
        let build = graph.build(&CorefBudget::default());
        assert_eq!(build.canonical("老张"), Some("张三"));
        assert_eq!(build.canonical("张老师"), Some("张三"));
        assert_eq!(build.canonical("他"), Some("张三"));
        assert_eq!(build.resolve("他").witness[0].kind, IdentityKind::Binding);
    }

    #[test]
    fn variant_closure_feeds_identity_and_metonymy_stays_non_identity() {
        let mut closure = VariantClosure::new();
        let types = BTreeSet::from(["人物".to_string()]);
        closure.declare_anchor_typed("张三", &types);
        closure.declare_anchor("苹果公司");
        closure.add_edge(VariantEdge::new("老张", "张三", VariantKind::Alias, "张三"));
        closure.add_edge(VariantEdge::new(
            "苹果",
            "苹果公司",
            VariantKind::Metonymy,
            "苹果公司",
        ));

        let graph = CorefGraph::from_variant_closure(&closure, &VariantContext::new());
        let build = graph.build(&CorefBudget::default());
        assert_eq!(build.canonical("老张"), Some("张三"));
        assert!(build
            .class_of("张三")
            .expect("应有类")
            .types
            .contains("人物"));
        assert_eq!(build.resolve("苹果").status, CorefStatus::Unknown);
        assert_eq!(graph.non_identity_edges().len(), 1);
    }

    #[test]
    fn referent_trail_pronouns_join_the_identity_closure() {
        let mut trail = ReferentTrail::default();
        trail.record(1, "他", "张三", ReferentSource::Pronoun, "测试");
        let graph = CorefGraph::from_referent_trail(&trail);
        let build = graph.build(&CorefBudget::default());
        assert_eq!(build.canonical("他"), Some("张三"));
        assert_eq!(build.resolve("他").witness[0].kind, IdentityKind::Pronoun);
    }

    #[test]
    fn evidence_json_uses_friendly_field_names() {
        let evidence = CorefEvidence::from_json(
            r#"{
                "identity": [
                    {"from": "老李", "to": "李四", "kind": "alias"},
                    {"from": "他", "to": "李四", "kind": "pronoun", "confidence": 0.8}
                ],
                "non_identity": [
                    {"from": "姚明的妻子", "to": "叶莉", "kind": "description"}
                ],
                "relations": [
                    {"from": "李四", "to": "王五", "predicate": "师傅"}
                ]
            }"#,
        )
        .expect("证据 JSON 应可解析");
        let graph = evidence.into_graph();
        let build = graph.build(&CorefBudget::default());
        assert_eq!(build.canonical("老李"), Some("李四"));
        assert_eq!(build.canonical("他"), Some("李四"));
        assert_eq!(build.resolve("姚明的妻子").status, CorefStatus::Unknown);
        assert_eq!(graph.relation_edges().len(), 1);
    }

    #[test]
    fn report_and_json_are_deterministic() {
        let mut graph = CorefGraph::new();
        graph.link("老张", "张三", IdentityKind::Alias, "张三");
        graph.link("张老师", "张三", IdentityKind::Honorific, "张三");
        graph.relate("张三", "李四", "同事");

        let first = graph.build(&CorefBudget::default()).report();
        let second = graph.build(&CorefBudget::default()).report();
        assert_eq!(first, second);

        let json = graph.to_json().expect("可序列化");
        let restored = CorefGraph::from_json(&json).expect("可反序列化");
        assert_eq!(restored.build(&CorefBudget::default()).report(), first);
    }
}

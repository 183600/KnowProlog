//! 查询身份层（Canonical Query Identity，CQI）。
//!
//! 解析器、句法重写、图谱模式与重写手册各自都能说明「这句话我读成了
//! 什么」，但缺少一个共享的「这两句话是不是同一个问题」。本模块把
//! [`QueryPlan`]、[`GraphPattern`](crate::conversational_graph::GraphPattern)
//! 与 [`KgPattern`] 压到同一个很小的正向合取关系核心，只应用**显式声明**
//! 的等价律：
//!
//! - 别名律：`alias(a, b)` 把词面 `b` 归一为 `a`；
//! - 互逆律：`inverse(p, q)` 允许 `p(a,b) ≡ q(b,a)`，并按稳定方向落形；
//! - 对称律：`symmetric(p)` 允许交换主宾语；
//! - 变量重命名与原子/过滤器声明顺序不改变身份；
//! - 投影是集合语义，排序后参与身份，但不会重排用户可见答案。
//!
//! 设计边界：
//!
//! - **不猜**：没有声明律时只做结构归一；不同规范键报告「未知」，绝不
//!   报告「不等价」；
//! - **不物化定义域/值域**：它们只用于排序，不进入规范键；
//! - **不引入权重**：颜色细化 + 小规模精确枚举，结果确定可复现；
//! - **可审计**：每次归一都带定律回执与规范材料，调用方可逐项复核。
//!
//! 精确标号在颜色类内枚举重命名；组合数超过
//! [`MAX_CANONICAL_LABELINGS`] 时退化为确定性名称序并标记
//! `complete = false`。此时「同键 ⇒ 等价」仍然成立，不同键只表示未知。
//! 规范键还与 [`CanonicalSchema::fingerprint`] 绑定，避免把不同定律
//! 体系下碰巧同形的查询误判为同一身份。

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::context_image::fnv1a_hex;
use crate::conversational_graph::{GraphPattern, PatternConstraint, PatternTerm};
use crate::kg_pattern::KgPattern;
use crate::law_book::{LawBook, LawKind};
use crate::query_planner::{QueryPlan, QueryPlanner};

/// CQI 方言版本；语义或规范材料不兼容变更时递增。
pub const QUERY_CANONICAL_VERSION: &str = "cqi/0.1";

/// 颜色类内最多枚举多少种重命名（8 个同类变量 = 40320）。
pub const MAX_CANONICAL_LABELINGS: u64 = 40_320;

/// 关系核心中的一项：变量或常量。
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum CoreTerm {
    Var(String),
    Const(String),
}

impl CoreTerm {
    pub fn variable(name: impl Into<String>) -> Self {
        Self::Var(name.into())
    }

    pub fn constant(value: impl Into<String>) -> Self {
        Self::Const(value.into())
    }

    pub fn var_name(&self) -> Option<&str> {
        match self {
            Self::Var(name) => Some(name),
            Self::Const(_) => None,
        }
    }

    pub fn constant_value(&self) -> Option<&str> {
        match self {
            Self::Const(value) => Some(value),
            Self::Var(_) => None,
        }
    }
}

/// 一条正向合取原子；谓词目前要求是常量（变量谓词是模板，不是问题身份）。
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct CoreAtom {
    pub subject: CoreTerm,
    pub predicate: String,
    pub object: CoreTerm,
}

impl CoreAtom {
    pub fn new(
        subject: CoreTerm,
        predicate: impl Into<String>,
        object: CoreTerm,
    ) -> Self {
        Self {
            subject,
            predicate: predicate.into(),
            object,
        }
    }
}

/// 关系核心上的过滤器。类型约束保留为过滤器，不物化成肯定前提。
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum CoreFilter {
    Equal { var: String, value: String },
    NotEqual { var: String, value: String },
    Type { var: String, of: String },
    NotType { var: String, of: String },
}

impl CoreFilter {
    pub fn var(&self) -> &str {
        match self {
            Self::Equal { var, .. }
            | Self::NotEqual { var, .. }
            | Self::Type { var, .. }
            | Self::NotType { var, .. } => var,
        }
    }
}

/// 与具体表示无关的正向合取查询核心。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RelationCore {
    pub atoms: Vec<CoreAtom>,
    pub filters: Vec<CoreFilter>,
    pub select: Vec<String>,
    pub count: bool,
    pub distinct: bool,
    /// 仅用于是非问：否定的命题与肯定命题不是同一身份。
    pub negated: bool,
}

impl RelationCore {
    pub fn new(atoms: Vec<CoreAtom>, select: Vec<String>) -> Self {
        Self {
            atoms,
            filters: Vec::new(),
            select,
            count: false,
            distinct: true,
            negated: false,
        }
    }

    pub fn with_filters(mut self, filters: Vec<CoreFilter>) -> Self {
        self.filters = filters;
        self
    }

    pub fn with_count(mut self, count: bool) -> Self {
        self.count = count;
        self
    }

    pub fn with_distinct(mut self, distinct: bool) -> Self {
        self.distinct = distinct;
        self
    }

    pub fn with_negated(mut self, negated: bool) -> Self {
        self.negated = negated;
        self
    }

    pub fn validate(&self) -> Result<(), CanonicalError> {
        if self.atoms.is_empty() {
            return Err(CanonicalError::EmptyCore);
        }
        let mut variables = BTreeSet::new();
        for atom in &self.atoms {
            if atom.predicate.trim().is_empty() {
                return Err(CanonicalError::EmptyPredicate);
            }
            for term in [&atom.subject, &atom.object] {
                if let Some(name) = term.var_name() {
                    variables.insert(name);
                }
            }
        }
        for filter in &self.filters {
            if !variables.contains(filter.var()) {
                return Err(CanonicalError::UnknownVariable(
                    filter.var().to_string(),
                ));
            }
        }
        let mut selected = BTreeSet::new();
        for name in &self.select {
            if !variables.contains(name.as_str()) {
                return Err(CanonicalError::UnknownVariable(name.clone()));
            }
            if !selected.insert(name.as_str()) {
                return Err(CanonicalError::DuplicateSelect(name.clone()));
            }
        }
        if self.negated && (self.count || !self.select.is_empty()) {
            return Err(CanonicalError::UnsupportedPattern(
                "否定只支持是非问，不能同时投影或计数".to_string(),
            ));
        }
        Ok(())
    }
}

impl From<&GraphPattern> for RelationCore {
    fn from(pattern: &GraphPattern) -> Self {
        let term = |term: &PatternTerm| match term {
            PatternTerm::Var(name) => CoreTerm::Var(name.clone()),
            PatternTerm::Const(value) => CoreTerm::Const(value.clone()),
        };
        let atoms = pattern
            .edges
            .iter()
            .map(|edge| CoreAtom::new(term(&edge.subject), edge.predicate.clone(), term(&edge.object)))
            .collect();
        let filters = pattern
            .constraints
            .iter()
            .map(|constraint| match constraint {
                PatternConstraint::Equal { variable, value } => CoreFilter::Equal {
                    var: variable.clone(),
                    value: value.clone(),
                },
                PatternConstraint::NotEqual { variable, value } => CoreFilter::NotEqual {
                    var: variable.clone(),
                    value: value.clone(),
                },
            })
            .collect();
        Self {
            atoms,
            filters,
            select: pattern.select.clone(),
            count: false,
            distinct: pattern.distinct,
            negated: false,
        }
    }
}

/// 归一失败：只说明「这个表示超出 CQI 核心」，不猜测其语义。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CanonicalError {
    EmptyCore,
    EmptyPredicate,
    UnknownVariable(String),
    DuplicateSelect(String),
    UnsupportedPlan(&'static str),
    UnsupportedPattern(String),
}

impl fmt::Display for CanonicalError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyCore => write!(formatter, "关系核心没有原子"),
            Self::EmptyPredicate => write!(formatter, "关系核心含空谓词"),
            Self::UnknownVariable(name) => {
                write!(formatter, "变量「{}」没有出现在任何原子中", name)
            }
            Self::DuplicateSelect(name) => write!(formatter, "投影变量「{}」重复", name),
            Self::UnsupportedPlan(kind) => {
                write!(formatter, "计划类型 {} 暂不能压成关系核心", kind)
            }
            Self::UnsupportedPattern(reason) => write!(formatter, "模式不能压成关系核心：{}", reason),
        }
    }
}

impl std::error::Error for CanonicalError {}

/// 一条被实际使用的等价律回执。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct LawReceipt {
    /// 稳定 id：`alias:...`、`inverse:...`、`symmetric:...`。
    pub id: String,
    pub label: String,
    pub detail: String,
}

impl LawReceipt {
    fn alias(surface: &str, canonical: &str, inverted: bool) -> Self {
        Self {
            id: format!("alias:{}->{}", surface, canonical),
            label: "别名".to_string(),
            detail: if inverted {
                format!("把「{}」归一为「{}」并按互逆方向交换主宾语", surface, canonical)
            } else {
                format!("把「{}」归一为「{}」", surface, canonical)
            },
        }
    }

    fn inverse(left: &str, right: &str) -> Self {
        Self {
            id: format!("inverse:{}|{}", left, right),
            label: "互逆".to_string(),
            detail: format!("「{}」与「{}」互逆，按稳定方向落形", left, right),
        }
    }

    fn symmetric(predicate: &str) -> Self {
        Self {
            id: format!("symmetric:{}", predicate),
            label: "对称".to_string(),
            detail: format!("「{}」对称，允许交换主宾语", predicate),
        }
    }
}

/// 只含已声明等价律的规范标号方案。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CanonicalSchema {
    aliases: BTreeMap<String, (String, bool)>,
    inverses: BTreeMap<String, String>,
    symmetric: BTreeSet<String>,
}

impl CanonicalSchema {
    pub fn new() -> Self {
        Self::default()
    }

    /// 从律典编译：只取 `Active` 声明律，挖掘/候选律不进入等价判定。
    pub fn from_law_book(book: &LawBook) -> Self {
        let mut schema = Self::new();
        for law in book.laws() {
            if !law.is_active() {
                continue;
            }
            match law.kind {
                LawKind::Alias if law.terms.len() == 2 => {
                    schema
                        .aliases
                        .entry(law.terms[1].clone())
                        .or_insert_with(|| (law.terms[0].clone(), false));
                }
                LawKind::Inverse if law.terms.len() == 2 => {
                    schema.declare_inverse(&law.terms[0], &law.terms[1]);
                }
                LawKind::Symmetric if law.terms.len() == 1 => {
                    schema.declare_symmetric(&law.terms[0]);
                }
                _ => {}
            }
        }
        schema
    }

    pub fn is_empty(&self) -> bool {
        self.aliases.is_empty() && self.inverses.is_empty() && self.symmetric.is_empty()
    }

    pub fn alias_count(&self) -> usize {
        self.aliases.len()
    }

    pub fn inverse_count(&self) -> usize {
        self.inverses.len() / 2
    }

    pub fn symmetric_count(&self) -> usize {
        self.symmetric.len()
    }

    pub fn is_symmetric(&self, predicate: &str) -> bool {
        self.symmetric.contains(predicate.trim())
    }

    pub fn inverse_partner(&self, predicate: &str) -> Option<&str> {
        self.inverses.get(predicate.trim()).map(String::as_str)
    }

    pub fn declare_alias(
        &mut self,
        surface: impl Into<String>,
        canonical: impl Into<String>,
    ) -> &mut Self {
        self.declare_alias_with_direction(surface, canonical, false)
    }

    pub fn declare_alias_with_direction(
        &mut self,
        surface: impl Into<String>,
        canonical: impl Into<String>,
        inverted: bool,
    ) -> &mut Self {
        let surface = surface.into().trim().to_string();
        let canonical = canonical.into().trim().to_string();
        if !surface.is_empty() && !canonical.is_empty() && surface != canonical {
            self.aliases.insert(surface, (canonical, inverted));
        }
        self
    }

    pub fn declare_inverse(
        &mut self,
        left: impl Into<String>,
        right: impl Into<String>,
    ) -> &mut Self {
        let left = left.into().trim().to_string();
        let right = right.into().trim().to_string();
        if !left.is_empty() && !right.is_empty() && left != right {
            self.inverses.insert(left.clone(), right.clone());
            self.inverses.insert(right, left);
        }
        self
    }

    pub fn declare_symmetric(&mut self, predicate: impl Into<String>) -> &mut Self {
        let predicate = predicate.into().trim().to_string();
        if !predicate.is_empty() {
            self.symmetric.insert(predicate);
        }
        self
    }

    /// 定律体系指纹：不同 schema 下生成的规范键不可直接比较。
    pub fn fingerprint(&self) -> String {
        let aliases: Vec<String> = self
            .aliases
            .iter()
            .map(|(surface, (canonical, inverted))| {
                format!(
                    "{}>{}:{}",
                    quote(surface),
                    quote(canonical),
                    u8::from(*inverted)
                )
            })
            .collect();
        let inverses: Vec<String> = self
            .inverses
            .iter()
            .filter(|(left, right)| left < right)
            .map(|(left, right)| format!("{}|{}", quote(left), quote(right)))
            .collect();
        let symmetric: Vec<String> = self
            .symmetric
            .iter()
            .map(|predicate| quote(predicate))
            .collect();
        fnv1a_hex(
            format!(
                "cqi-schema/1|aliases:{}|inverses:{}|symmetric:{}",
                aliases.join(";"),
                inverses.join(";"),
                symmetric.join(",")
            )
            .as_bytes(),
        )
    }

    fn normalize_predicate(&self, predicate: &str) -> (String, bool, Vec<LawReceipt>) {
        let mut current = predicate.trim().to_string();
        let mut inverted = false;
        let mut laws = Vec::new();
        let mut seen = BTreeSet::new();
        seen.insert(current.clone());
        loop {
            let Some((canonical, flip)) = self.aliases.get(&current) else {
                break;
            };
            if canonical == &current || !seen.insert(canonical.clone()) {
                break;
            }
            laws.push(LawReceipt::alias(&current, canonical, *flip));
            current = canonical.clone();
            inverted ^= *flip;
        }
        if let Some(partner) = self.inverses.get(&current) {
            if partner < &current {
                laws.push(LawReceipt::inverse(&current, partner));
                current = partner.clone();
                inverted = !inverted;
            }
        }
        (current, inverted, laws)
    }
}

/// 答案形状：实体列表、是非、计数。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CanonicalKind {
    Entities,
    Boolean,
    Count,
}

impl CanonicalKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Entities => "实体投影",
            Self::Boolean => "是非判定",
            Self::Count => "计数",
        }
    }
}

/// 一次规范化的可审计结果。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CanonicalQuery {
    /// 规范键；同 schema 下同键 ⇒ 等价。
    pub key: String,
    /// 规范材料（变量已标号、原子/过滤器已排序）。
    pub material: String,
    /// 是否完成了颜色类内的精确枚举。
    pub complete: bool,
    pub kind: CanonicalKind,
    pub distinct: bool,
    pub negated: bool,
    pub select: Vec<String>,
    pub atom_lines: Vec<String>,
    pub filter_lines: Vec<String>,
    pub laws: Vec<LawReceipt>,
}

impl CanonicalQuery {
    pub fn receipt(&self) -> String {
        let laws = if self.laws.is_empty() {
            "无等价律".to_string()
        } else {
            self.laws
                .iter()
                .map(|law| law.detail.as_str())
                .collect::<Vec<_>>()
                .join("；")
        };
        format!(
            "查询身份 {}（{}，{}{}）：原子 {}；过滤器 {}；投影 {}；{}",
            self.key,
            self.kind.label(),
            if self.complete { "精确" } else { "部分" },
            if self.negated { "，否定" } else { "" },
            if self.atom_lines.is_empty() {
                "—".to_string()
            } else {
                self.atom_lines.join("；")
            },
            if self.filter_lines.is_empty() {
                "—".to_string()
            } else {
                self.filter_lines.join("；")
            },
            if self.select.is_empty() {
                "—".to_string()
            } else {
                self.select.join("、")
            },
            laws
        )
    }
}

/// 两个计划之间的比较结论。「不同键」只表示未证明等价。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanEquivalence {
    Equivalent {
        key: String,
        laws: Vec<LawReceipt>,
    },
    Unknown {
        left: Option<String>,
        right: Option<String>,
        reason: String,
    },
}

impl PlanEquivalence {
    pub fn is_equivalent(&self) -> bool {
        matches!(self, Self::Equivalent { .. })
    }

    pub fn receipt(&self) -> String {
        match self {
            Self::Equivalent { key, laws } => {
                if laws.is_empty() {
                    format!("等价（{}）", key)
                } else {
                    format!(
                        "等价（{}；{}）",
                        key,
                        laws.iter()
                            .map(|law| law.detail.as_str())
                            .collect::<Vec<_>>()
                            .join("；")
                    )
                }
            }
            Self::Unknown { left, right, reason } => format!(
                "未知（左 {}，右 {}）：{}",
                left.as_deref().unwrap_or("超出核心"),
                right.as_deref().unwrap_or("超出核心"),
                reason
            ),
        }
    }
}

/// 把查询计划压成关系核心；超出支持范围时显式返回错误。
pub fn core_from_plan(plan: &QueryPlan) -> Result<RelationCore, CanonicalError> {
    let constant = |value: &str| CoreTerm::Const(value.to_string());
    let variable = |name: &str| CoreTerm::Var(name.to_string());
    match plan {
        QueryPlan::FindObject { segments } => {
            if segments.len() != 2 {
                return Err(CanonicalError::UnsupportedPlan(
                    "FindObject：只有「实体 + 单谓词」能压成关系核心",
                ));
            }
            Ok(RelationCore::new(
                vec![CoreAtom::new(
                    constant(&segments[0]),
                    segments[1].clone(),
                    variable("object"),
                )],
                vec!["object".to_string()],
            ))
        }
        QueryPlan::FindObjectByPredicate { subject, predicate } => Ok(RelationCore::new(
            vec![CoreAtom::new(
                constant(subject),
                predicate.clone(),
                variable("object"),
            )],
            vec!["object".to_string()],
        )),
        QueryPlan::FindSubject { object, predicate } => Ok(RelationCore::new(
            vec![CoreAtom::new(
                variable("subject"),
                predicate.clone(),
                constant(object),
            )],
            vec!["subject".to_string()],
        )),
        QueryPlan::FindObjectExcept {
            subject,
            predicate,
            excluded,
        } => Ok(RelationCore::new(
            vec![CoreAtom::new(
                constant(subject),
                predicate.clone(),
                variable("object"),
            )],
            vec!["object".to_string()],
        )
        .with_filters(
            excluded
                .iter()
                .map(|value| CoreFilter::NotEqual {
                    var: "object".to_string(),
                    value: value.clone(),
                })
                .collect(),
        )),
        QueryPlan::FindSubjectExcept {
            object,
            predicate,
            excluded,
        } => Ok(RelationCore::new(
            vec![CoreAtom::new(
                variable("subject"),
                predicate.clone(),
                constant(object),
            )],
            vec!["subject".to_string()],
        )
        .with_filters(
            excluded
                .iter()
                .map(|value| CoreFilter::NotEqual {
                    var: "subject".to_string(),
                    value: value.clone(),
                })
                .collect(),
        )),
        QueryPlan::CheckRelation {
            subject,
            predicate,
            object,
            negated,
        } => Ok(RelationCore::new(
            vec![CoreAtom::new(
                constant(subject),
                predicate.clone(),
                constant(object),
            )],
            Vec::new(),
        )
        .with_negated(*negated)),
        QueryPlan::CheckTriple {
            subject_segments,
            object,
        } => {
            if subject_segments.len() != 2 {
                return Err(CanonicalError::UnsupportedPlan(
                    "CheckTriple：只有「实体 + 单谓词」能压成关系核心",
                ));
            }
            Ok(RelationCore::new(
                vec![CoreAtom::new(
                    constant(&subject_segments[0]),
                    subject_segments[1].clone(),
                    constant(object),
                )],
                Vec::new(),
            ))
        }
        QueryPlan::Count { subject, predicate } => Ok(RelationCore::new(
            vec![CoreAtom::new(
                constant(subject),
                predicate.clone(),
                variable("object"),
            )],
            vec!["object".to_string()],
        )
        .with_count(true)),
        QueryPlan::Exist { subject, predicate } => Ok(RelationCore::new(
            vec![CoreAtom::new(
                constant(subject),
                predicate.clone(),
                variable("object"),
            )],
            Vec::new(),
        )),
        QueryPlan::Closure { .. } => Err(CanonicalError::UnsupportedPlan(
            "Closure：闭包算子暂不进入关系核心",
        )),
        QueryPlan::PathBetween { .. } => Err(CanonicalError::UnsupportedPlan("PathBetween")),
        QueryPlan::CommonNeighbor { .. } => {
            Err(CanonicalError::UnsupportedPlan("CommonNeighbor"))
        }
        QueryPlan::Intermediary { .. } => Err(CanonicalError::UnsupportedPlan("Intermediary")),
        QueryPlan::Similar { .. } => Err(CanonicalError::UnsupportedPlan("Similar")),
        QueryPlan::Analogy { .. } => Err(CanonicalError::UnsupportedPlan("Analogy")),
    }
}

/// 把关系核心归一为可比较身份。
pub fn canonicalize_core(
    core: &RelationCore,
    schema: &CanonicalSchema,
) -> Result<CanonicalQuery, CanonicalError> {
    core.validate()?;

    let mut laws = Vec::new();
    let mut atoms = Vec::with_capacity(core.atoms.len());
    for atom in &core.atoms {
        let (predicate, inverted, mut receipts) = schema.normalize_predicate(&atom.predicate);
        laws.append(&mut receipts);
        let (subject, object) = if inverted {
            (atom.object.clone(), atom.subject.clone())
        } else {
            (atom.subject.clone(), atom.object.clone())
        };
        atoms.push(CoreAtom::new(subject, predicate, object));
    }
    atoms.sort();
    atoms.dedup();

    let mut filters = core.filters.clone();
    filters.sort();
    filters.dedup();

    let mut select = core.select.clone();
    select.sort();
    select.dedup();

    let render = |mapping: &BTreeMap<String, String>| {
        render_material(&atoms, &filters, &select, core, schema, mapping).material
    };
    let (mapping, complete) = canonical_mapping(&atoms, &filters, &select, render)?;
    let rendered = render_material(&atoms, &filters, &select, core, schema, &mapping);
    let RenderedMaterial {
        material,
        kind,
        select: select_lines,
        atoms: atom_lines,
        filters: filter_lines,
    } = rendered;

    for atom in &atoms {
        if schema.is_symmetric(&atom.predicate) {
            let subject = render_term(&atom.subject, &mapping);
            let object = render_term(&atom.object, &mapping);
            if object < subject {
                laws.push(LawReceipt::symmetric(&atom.predicate));
            }
        }
    }
    laws.sort();
    laws.dedup();

    let key = fnv1a_hex(
        format!(
            "{}|schema:{}|{}",
            QUERY_CANONICAL_VERSION,
            schema.fingerprint(),
            material
        )
        .as_bytes(),
    );
    Ok(CanonicalQuery {
        key,
        material,
        complete,
        kind,
        distinct: core.distinct,
        negated: core.negated,
        select: select_lines,
        atom_lines,
        filter_lines,
        laws,
    })
}

/// 归一查询计划。
pub fn canonicalize_plan(
    plan: &QueryPlan,
    schema: &CanonicalSchema,
) -> Result<CanonicalQuery, CanonicalError> {
    canonicalize_core(&core_from_plan(plan)?, schema)
}

/// 归一图执行层模式。
pub fn canonicalize_graph_pattern(
    pattern: &GraphPattern,
    schema: &CanonicalSchema,
) -> Result<CanonicalQuery, CanonicalError> {
    canonicalize_core(&RelationCore::from(pattern), schema)
}

/// 归一 KGP-IR 模式；变量谓词需要运行时接地，不进入问题身份。
pub fn canonicalize_kg_pattern(
    pattern: &KgPattern,
    schema: &CanonicalSchema,
) -> Result<CanonicalQuery, CanonicalError> {
    canonicalize_core(&pattern.relation_core()?, schema)
}

/// 规范键；超出关系核心时退回原计划渲染键（只用于稳定去重）。
pub fn plan_key(plan: &QueryPlan, schema: &CanonicalSchema) -> String {
    canonicalize_plan(plan, schema)
        .map(|query| query.key)
        .unwrap_or_else(|_| legacy_plan_key(plan))
}

/// 旧式计划键：类型 + 人类可读计划。
pub fn legacy_plan_key(plan: &QueryPlan) -> String {
    format!(
        "{}|{}",
        QueryPlanner::kind(plan),
        QueryPlanner::format_plan(plan)
    )
}

/// 比较两个计划。只有得到同一规范键（或二者完全相同）才报告等价。
pub fn compare_plans(
    left: &QueryPlan,
    right: &QueryPlan,
    schema: &CanonicalSchema,
) -> PlanEquivalence {
    match (canonicalize_plan(left, schema), canonicalize_plan(right, schema)) {
        (Ok(left_query), Ok(right_query)) if left_query.key == right_query.key => {
            let mut laws = left_query.laws;
            laws.extend(right_query.laws);
            laws.sort();
            laws.dedup();
            PlanEquivalence::Equivalent {
                key: left_query.key,
                laws,
            }
        }
        (Ok(left_query), Ok(right_query)) => PlanEquivalence::Unknown {
            left: Some(left_query.key),
            right: Some(right_query.key),
            reason: "规范键不同；未声明足够的等价律".to_string(),
        },
        _ if left == right => PlanEquivalence::Equivalent {
            key: legacy_plan_key(left),
            laws: Vec::new(),
        },
        (left_query, right_query) => PlanEquivalence::Unknown {
            left: left_query.ok().map(|query| query.key),
            right: right_query.ok().map(|query| query.key),
            reason: "计划超出关系核心，无法归一".to_string(),
        },
    }
}

/// 便捷判定：同键才返回 `true`。
pub fn plans_equivalent(
    left: &QueryPlan,
    right: &QueryPlan,
    schema: &CanonicalSchema,
) -> bool {
    compare_plans(left, right, schema).is_equivalent()
}

struct RenderedMaterial {
    material: String,
    kind: CanonicalKind,
    select: Vec<String>,
    atoms: Vec<String>,
    filters: Vec<String>,
}

fn render_material(
    atoms: &[CoreAtom],
    filters: &[CoreFilter],
    select: &[String],
    core: &RelationCore,
    schema: &CanonicalSchema,
    mapping: &BTreeMap<String, String>,
) -> RenderedMaterial {
    let kind = if core.count {
        CanonicalKind::Count
    } else if select.is_empty() {
        CanonicalKind::Boolean
    } else {
        CanonicalKind::Entities
    };
    let mut atom_lines: Vec<String> = atoms
        .iter()
        .map(|atom| render_atom(atom, mapping, schema))
        .collect();
    atom_lines.sort();
    atom_lines.dedup();

    let mut filter_lines: Vec<String> = filters
        .iter()
        .map(|filter| render_filter(filter, mapping))
        .collect();
    filter_lines.sort();
    filter_lines.dedup();

    let mut select_lines: Vec<String> = select
        .iter()
        .map(|name| {
            mapping
                .get(name)
                .cloned()
                .unwrap_or_else(|| format!("?{}", name))
        })
        .collect();
    select_lines.sort();
    select_lines.dedup();

    let material = format!(
        "kind:{}|distinct:{}|negated:{}|select:{}|atoms:{}|filters:{}",
        kind_tag(kind),
        u8::from(core.distinct),
        u8::from(core.negated),
        select_lines.join(","),
        atom_lines.join(";"),
        filter_lines.join(";")
    );
    RenderedMaterial {
        material,
        kind,
        select: select_lines,
        atoms: atom_lines,
        filters: filter_lines,
    }
}

fn kind_tag(kind: CanonicalKind) -> &'static str {
    match kind {
        CanonicalKind::Entities => "entities",
        CanonicalKind::Boolean => "boolean",
        CanonicalKind::Count => "count",
    }
}

fn render_atom(
    atom: &CoreAtom,
    mapping: &BTreeMap<String, String>,
    schema: &CanonicalSchema,
) -> String {
    let mut subject = render_term(&atom.subject, mapping);
    let mut object = render_term(&atom.object, mapping);
    if schema.is_symmetric(&atom.predicate) && object < subject {
        std::mem::swap(&mut subject, &mut object);
    }
    format!("a({},{},{})", subject, quote(&atom.predicate), object)
}

fn render_filter(filter: &CoreFilter, mapping: &BTreeMap<String, String>) -> String {
    let var = |name: &str| {
        mapping
            .get(name)
            .cloned()
            .unwrap_or_else(|| format!("?{}", name))
    };
    match filter {
        CoreFilter::Equal { var: name, value } => {
            format!("equal({},{})", var(name), quote(value))
        }
        CoreFilter::NotEqual { var: name, value } => {
            format!("not_equal({},{})", var(name), quote(value))
        }
        CoreFilter::Type { var: name, of } => format!("type({},{})", var(name), quote(of)),
        CoreFilter::NotType { var: name, of } => {
            format!("not_type({},{})", var(name), quote(of))
        }
    }
}

fn render_term(term: &CoreTerm, mapping: &BTreeMap<String, String>) -> String {
    match term {
        CoreTerm::Var(name) => mapping
            .get(name)
            .cloned()
            .unwrap_or_else(|| format!("?{}", name)),
        CoreTerm::Const(value) => format!("lit:{}", quote(value)),
    }
}

fn quote(value: &str) -> String {
    let mut output = String::with_capacity(value.len() + 2);
    output.push('"');
    for ch in value.chars() {
        match ch {
            '\\' => output.push_str("\\\\"),
            '"' => output.push_str("\\\""),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            _ => output.push(ch),
        }
    }
    output.push('"');
    output
}

fn canonical_mapping<F>(
    atoms: &[CoreAtom],
    filters: &[CoreFilter],
    select: &[String],
    mut render: F,
) -> Result<(BTreeMap<String, String>, bool), CanonicalError>
where
    F: FnMut(&BTreeMap<String, String>) -> String,
{
    let (labels, edges, variables) = build_label_graph(atoms, filters, select);
    let colors = refine_colors(&labels, &edges);
    let mut classes: BTreeMap<usize, Vec<String>> = BTreeMap::new();
    for (name, node) in variables {
        classes.entry(colors[node]).or_default().push(name);
    }
    for names in classes.values_mut() {
        names.sort();
    }
    let class_list: Vec<Vec<String>> = classes.into_values().collect();
    let permutations: Vec<Vec<Vec<String>>> = class_list
        .iter()
        .map(|names| permutations_of(names))
        .collect();
    let mut combinations: u64 = 1;
    for choices in &permutations {
        combinations = combinations.saturating_mul(choices.len() as u64);
    }
    if combinations <= MAX_CANONICAL_LABELINGS {
        let mut best: Option<(String, BTreeMap<String, String>)> = None;
        let mut current = Vec::new();
        enumerate_choices(&permutations, 0, &mut current, &mut |choice| {
            let mapping = mapping_from_choice(choice);
            let material = render(&mapping);
            if best
                .as_ref()
                .map_or(true, |(best_material, _)| material < *best_material)
            {
                best = Some((material, mapping));
            }
        });
        match best {
            Some((_, mapping)) => Ok((mapping, true)),
            None => Ok((BTreeMap::new(), true)),
        }
    } else {
        let choice: Vec<Vec<String>> = permutations
            .iter()
            .map(|choices| choices.first().cloned().unwrap_or_default())
            .collect();
        Ok((mapping_from_choice(&choice), false))
    }
}

fn build_label_graph(
    atoms: &[CoreAtom],
    filters: &[CoreFilter],
    select: &[String],
) -> (Vec<String>, Vec<(usize, &'static str, usize)>, Vec<(String, usize)>) {
    let mut names: BTreeSet<String> = BTreeSet::new();
    for atom in atoms {
        for term in [&atom.subject, &atom.object] {
            if let Some(name) = term.var_name() {
                names.insert(name.to_string());
            }
        }
    }
    for filter in filters {
        names.insert(filter.var().to_string());
    }

    let selected: BTreeSet<&str> = select.iter().map(String::as_str).collect();
    let mut labels = Vec::new();
    let mut variables = Vec::new();
    for name in &names {
        let node = labels.len();
        labels.push(if selected.contains(name.as_str()) {
            "var:selected".to_string()
        } else {
            "var:plain".to_string()
        });
        variables.push((name.clone(), node));
    }
    let var_index: BTreeMap<String, usize> = variables
        .iter()
        .map(|(name, node)| (name.clone(), *node))
        .collect();

    let mut constants: BTreeMap<String, usize> = BTreeMap::new();
    let mut edges = Vec::new();
    for atom in atoms {
        let node = labels.len();
        labels.push("atom".to_string());
        let predicate = term_node(
            &CoreTerm::Const(atom.predicate.clone()),
            &var_index,
            &mut constants,
            &mut labels,
        );
        edges.push((node, "p", predicate));
        for (label, term) in [("s", &atom.subject), ("o", &atom.object)] {
            let target = term_node(term, &var_index, &mut constants, &mut labels);
            edges.push((node, label, target));
        }
    }
    for filter in filters {
        let node = labels.len();
        labels.push(filter_label(filter));
        let target = var_index[filter.var()];
        edges.push((node, "f", target));
    }
    (labels, edges, variables)
}

fn term_node(
    term: &CoreTerm,
    var_index: &BTreeMap<String, usize>,
    constants: &mut BTreeMap<String, usize>,
    labels: &mut Vec<String>,
) -> usize {
    match term {
        CoreTerm::Var(name) => var_index[name.as_str()],
        CoreTerm::Const(value) => match constants.get(value) {
            Some(node) => *node,
            None => {
                let node = labels.len();
                labels.push(format!("const:{}:{}", value.len(), value));
                constants.insert(value.clone(), node);
                node
            }
        },
    }
}

fn filter_label(filter: &CoreFilter) -> String {
    match filter {
        CoreFilter::Equal { value, .. } => format!("filter:equal:{}:{}", value.len(), value),
        CoreFilter::NotEqual { value, .. } => {
            format!("filter:not_equal:{}:{}", value.len(), value)
        }
        CoreFilter::Type { of, .. } => format!("filter:type:{}:{}", of.len(), of),
        CoreFilter::NotType { of, .. } => format!("filter:not_type:{}:{}", of.len(), of),
    }
}

fn refine_colors(labels: &[String], edges: &[(usize, &'static str, usize)]) -> Vec<usize> {
    let unique: BTreeSet<&str> = labels.iter().map(String::as_str).collect();
    let initial: BTreeMap<&str, usize> = unique
        .iter()
        .enumerate()
        .map(|(index, label)| (*label, index))
        .collect();
    let mut colors: Vec<usize> = labels
        .iter()
        .map(|label| initial[label.as_str()])
        .collect();
    loop {
        let mut signatures: Vec<(usize, Vec<(String, usize)>)> = colors
            .iter()
            .map(|color| (*color, Vec::new()))
            .collect();
        for (from, label, to) in edges {
            signatures[*from]
                .1
                .push((format!(">{}", label), colors[*to]));
            signatures[*to]
                .1
                .push((format!("<{}", label), colors[*from]));
        }
        for (_, pairs) in &mut signatures {
            pairs.sort();
        }
        let mut unique_signatures = signatures.clone();
        unique_signatures.sort();
        unique_signatures.dedup();
        let ranks: BTreeMap<(usize, Vec<(String, usize)>), usize> = unique_signatures
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, signature)| (signature, index))
            .collect();
        let new_colors: Vec<usize> = signatures
            .iter()
            .map(|signature| ranks[signature])
            .collect();
        let before = colors.iter().copied().collect::<BTreeSet<_>>().len();
        let after = new_colors.iter().copied().collect::<BTreeSet<_>>().len();
        colors = new_colors;
        if before == after {
            break;
        }
    }
    colors
}

fn permutations_of(items: &[String]) -> Vec<Vec<String>> {
    fn walk(
        items: &[String],
        used: &mut [bool],
        current: &mut Vec<String>,
        output: &mut Vec<Vec<String>>,
    ) {
        if current.len() == items.len() {
            output.push(current.clone());
            return;
        }
        for index in 0..items.len() {
            if used[index] {
                continue;
            }
            used[index] = true;
            current.push(items[index].clone());
            walk(items, used, current, output);
            current.pop();
            used[index] = false;
        }
    }
    let mut output = Vec::new();
    let mut used = vec![false; items.len()];
    let mut current = Vec::new();
    walk(items, &mut used, &mut current, &mut output);
    output
}

fn enumerate_choices<F>(
    choices: &[Vec<Vec<String>>],
    index: usize,
    current: &mut Vec<Vec<String>>,
    visit: &mut F,
) where
    F: FnMut(&[Vec<String>]),
{
    if index == choices.len() {
        visit(current);
        return;
    }
    for permutation in &choices[index] {
        current.push(permutation.clone());
        enumerate_choices(choices, index + 1, current, visit);
        current.pop();
    }
}

fn mapping_from_choice(choice: &[Vec<String>]) -> BTreeMap<String, String> {
    let mut mapping = BTreeMap::new();
    let mut index = 0;
    for class in choice {
        for name in class {
            mapping.insert(name.clone(), format!("v{}", index));
            index += 1;
        }
    }
    mapping
}

/// 规范查询到中文表层的一次回读审计。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RealizationAudit {
    /// 生成的中文问句。
    pub question: String,
    /// 原计划的规范键。
    pub original_key: String,
    /// 回读后计划的关系种类。
    pub readback_kind: Option<String>,
    /// 回读后的规范键；回读失败或不可归一为 `None`。
    pub readback_key: Option<String>,
    /// 回读计划是否与原计划同键。
    pub consistent: bool,
}

impl RealizationAudit {
    pub fn receipt(&self) -> String {
        format!(
            "生成回读「{}」：{}；原键 {}，回读 {}（{}）",
            self.question,
            if self.consistent {
                "同键"
            } else {
                "异键或无法回读"
            },
            self.original_key,
            self.readback_key.as_deref().unwrap_or("—"),
            self.readback_kind.as_deref().unwrap_or("—")
        )
    }
}

/// 为可逆的关系核心生成一个中文问句。
///
/// 只使用解析器能稳定读回的句式；不可逆的计划（路径、类比、否定是非等）
/// 返回 `None`，宁可没有生成物，也不生成无法自检的句子。
pub fn realize_question(plan: &QueryPlan) -> Option<String> {
    match plan {
        QueryPlan::FindObject { segments } if segments.len() >= 2 => {
            Some(format!("{}是谁", segments.join("的")))
        }
        QueryPlan::FindObjectByPredicate { subject, predicate } => {
            Some(format!("{}的{}是谁", subject, predicate))
        }
        QueryPlan::FindSubject { object, predicate } => {
            Some(format!("{}是谁的{}", object, predicate))
        }
        QueryPlan::FindObjectExcept { .. } => None,
        QueryPlan::FindSubjectExcept {
            object,
            predicate,
            excluded,
        } if !excluded.is_empty() => Some(format!(
            "除了{}，还有谁的{}是{}",
            excluded.join("、"),
            predicate,
            object
        )),
        QueryPlan::CheckTriple {
            subject_segments,
            object,
        } if subject_segments.len() >= 2 => Some(format!(
            "{}是{}",
            subject_segments.join("的"),
            object
        )),
        QueryPlan::CheckRelation {
            subject,
            predicate,
            object,
            negated: false,
        } => Some(format!("{}的{}是{}", subject, predicate, object)),
        QueryPlan::CheckRelation { negated: true, .. } => None,
        QueryPlan::Count { subject, predicate } => {
            Some(format!("{}有几个{}", subject, predicate))
        }
        QueryPlan::Exist { subject, predicate } => {
            Some(format!("{}有没有{}", subject, predicate))
        }
        _ => None,
    }
}

/// 生成中文问句并用同一 schema 回读，校验规范键是否保持不变。
pub fn audit_realization(
    plan: &QueryPlan,
    schema: &CanonicalSchema,
) -> Option<RealizationAudit> {
    let original = canonicalize_plan(plan, schema).ok()?;
    let question = realize_question(plan)?;
    let readback = crate::query_planner::QueryPlanner::new().parse(&question);
    let readback_kind = readback
        .as_ref()
        .map(crate::query_planner::QueryPlanner::kind)
        .map(str::to_string);
    let readback_key = readback
        .as_ref()
        .and_then(|readback| canonicalize_plan(readback, schema).ok())
        .map(|canonical| canonical.key);
    let consistent = readback_key.as_deref() == Some(original.key.as_str());
    Some(RealizationAudit {
        question,
        original_key: original.key,
        readback_kind,
        readback_key,
        consistent,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema() -> CanonicalSchema {
        let mut schema = CanonicalSchema::new();
        schema.declare_alias("老师", "师傅");
        schema.declare_inverse("师傅", "徒弟");
        schema.declare_symmetric("配偶");
        schema
    }

    fn find_object(subject: &str, predicate: &str) -> QueryPlan {
        QueryPlan::FindObjectByPredicate {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
        }
    }

    #[test]
    fn schema_from_law_book_uses_declared_laws() {
        use crate::predicate_registry::{PredicateProfileConfig, PredicateRegistry};
        let mut registry = PredicateRegistry::new();
        assert!(registry.declare(&PredicateProfileConfig {
            name: "师傅".to_string(),
            aliases: vec!["老师".to_string()],
            inverse: Some("徒弟".to_string()),
            ..PredicateProfileConfig::default()
        }));
        assert!(registry.declare(&PredicateProfileConfig {
            name: "配偶".to_string(),
            symmetric: true,
            ..PredicateProfileConfig::default()
        }));
        let book = crate::law_book::LawBook::from_registry(&registry);
        let schema = CanonicalSchema::from_law_book(&book);
        assert!(schema.alias_count() >= 1);
        assert!(schema.inverse_partner("师傅").is_some());
        assert!(schema.is_symmetric("配偶"));
        let alias = find_object("张三", "老师");
        let canonical = find_object("张三", "师傅");
        assert!(plans_equivalent(&alias, &canonical, &schema));
        let inverse = QueryPlan::FindSubject {
            object: "李四".to_string(),
            predicate: "徒弟".to_string(),
        };
        let direct = find_object("李四", "师傅");
        assert!(plans_equivalent(&inverse, &direct, &schema));
    }

    #[test]
    fn plan_variants_share_one_identity() {
        let schema = CanonicalSchema::new();
        let chain = QueryPlan::FindObject {
            segments: vec!["张三".to_string(), "师傅".to_string()],
        };
        let direct = find_object("张三", "师傅");
        let left = canonicalize_plan(&chain, &schema).expect("链式单跳应可归一");
        let right = canonicalize_plan(&direct, &schema).expect("关系计划应可归一");
        assert_eq!(left.key, right.key, "{}", left.receipt());
        assert!(plans_equivalent(&chain, &direct, &schema));

        let check_triple = QueryPlan::CheckTriple {
            subject_segments: vec!["张三".to_string(), "师傅".to_string()],
            object: "李四".to_string(),
        };
        let check_relation = QueryPlan::CheckRelation {
            subject: "张三".to_string(),
            predicate: "师傅".to_string(),
            object: "李四".to_string(),
            negated: false,
        };
        assert!(plans_equivalent(&check_triple, &check_relation, &schema));
    }

    #[test]
    fn alias_and_inverse_laws_normalize_orientation() {
        let schema = schema();
        let alias = find_object("张三", "老师");
        let canonical = find_object("张三", "师傅");
        assert!(plans_equivalent(&alias, &canonical, &schema));

        let inverse = QueryPlan::FindSubject {
            object: "李四".to_string(),
            predicate: "徒弟".to_string(),
        };
        let direct = find_object("李四", "师傅");
        let identity = canonicalize_plan(&inverse, &schema).expect("互逆谓词应可归一");
        assert!(plans_equivalent(&inverse, &direct, &schema), "{}", identity.receipt());
        assert!(identity
            .laws
            .iter()
            .any(|law| law.label == "互逆" || law.label == "别名"));
    }

    #[test]
    fn symmetric_law_canonicalizes_orientation() {
        let schema = schema();
        let forward = QueryPlan::CheckRelation {
            subject: "张三".to_string(),
            predicate: "配偶".to_string(),
            object: "李四".to_string(),
            negated: false,
        };
        let backward = QueryPlan::CheckRelation {
            subject: "李四".to_string(),
            predicate: "配偶".to_string(),
            object: "张三".to_string(),
            negated: false,
        };
        assert!(plans_equivalent(&forward, &backward, &schema));
        let identity = canonicalize_plan(&backward, &schema).expect("对称谓词应可归一");
        assert!(identity.laws.iter().any(|law| law.label == "对称"));
    }

    #[test]
    fn different_projection_or_polarity_stays_unknown() {
        let schema = CanonicalSchema::new();
        let objects = find_object("张三", "师傅");
        let subjects = QueryPlan::FindSubject {
            object: "李四".to_string(),
            predicate: "师傅".to_string(),
        };
        assert!(!plans_equivalent(&objects, &subjects, &schema));

        let positive = QueryPlan::CheckRelation {
            subject: "张三".to_string(),
            predicate: "师傅".to_string(),
            object: "李四".to_string(),
            negated: false,
        };
        let negative = QueryPlan::CheckRelation {
            subject: "张三".to_string(),
            predicate: "师傅".to_string(),
            object: "李四".to_string(),
            negated: true,
        };
        assert!(!plans_equivalent(&positive, &negative, &schema));
    }

    #[test]
    fn excluded_values_are_order_insensitive() {
        let schema = CanonicalSchema::new();
        let first = QueryPlan::FindObjectExcept {
            subject: "张三".to_string(),
            predicate: "师傅".to_string(),
            excluded: vec!["李四".to_string(), "王五".to_string()],
        };
        let second = QueryPlan::FindObjectExcept {
            subject: "张三".to_string(),
            predicate: "师傅".to_string(),
            excluded: vec!["王五".to_string(), "李四".to_string()],
        };
        assert!(plans_equivalent(&first, &second, &schema));
    }

    #[test]
    fn graph_and_kg_pattern_share_the_core_identity() {
        let schema = CanonicalSchema::new();
        let graph = GraphPattern::new(
            vec![crate::conversational_graph::EdgePattern::new(
                PatternTerm::var("x"),
                "师傅",
                PatternTerm::var("y"),
            )],
            vec!["y".to_string()],
        );
        let declared = crate::kg_pattern::KgPatternRegistry::from_declarations(&[
            crate::kg_pattern::KgPatternDecl {
                id: "masters".to_string(),
                description: String::new(),
                atoms: vec![crate::kg_pattern::KgAtomDecl {
                    subject: "?person".to_string(),
                    predicate: "师傅".to_string(),
                    object: "?master".to_string(),
                }],
                filters: Vec::new(),
                select: vec!["master".to_string()],
                count: false,
            },
        ])
        .expect("模式应可编译");
        let pattern = declared.get("masters").unwrap();
        let graph_identity = canonicalize_graph_pattern(&graph, &schema).expect("图模式应可归一");
        let pattern_identity =
            canonicalize_kg_pattern(pattern, &schema).expect("KGP 模式应可归一");
        assert_eq!(graph_identity.key, pattern_identity.key);
        assert_eq!(graph_identity.atom_lines, vec!["a(v0,\"师傅\",v1)".to_string()]);
    }

    #[test]
    fn realization_round_trips_supported_shapes() {
        let schema = CanonicalSchema::new();
        let plans = vec![
            QueryPlan::FindObject {
                segments: vec!["张三".to_string(), "师傅".to_string()],
            },
            find_object("张三", "师傅"),
            QueryPlan::FindSubject {
                object: "李四".to_string(),
                predicate: "师傅".to_string(),
            },
            QueryPlan::FindSubjectExcept {
                object: "李四".to_string(),
                predicate: "师傅".to_string(),
                excluded: vec!["王五".to_string(), "赵六".to_string()],
            },
            QueryPlan::CheckTriple {
                subject_segments: vec!["张三".to_string(), "师傅".to_string()],
                object: "李四".to_string(),
            },
            QueryPlan::CheckRelation {
                subject: "张三".to_string(),
                predicate: "师傅".to_string(),
                object: "李四".to_string(),
                negated: false,
            },
            QueryPlan::Count {
                subject: "张三".to_string(),
                predicate: "徒弟".to_string(),
            },
            QueryPlan::Exist {
                subject: "张三".to_string(),
                predicate: "徒弟".to_string(),
            },
        ];
        for plan in plans {
            let audit = audit_realization(&plan, &schema)
                .unwrap_or_else(|| panic!("应能生成并回读：{:?}", plan));
            assert!(audit.consistent, "{}", audit.receipt());
        }
        assert!(audit_realization(
            &QueryPlan::CheckRelation {
                subject: "张三".to_string(),
                predicate: "师傅".to_string(),
                object: "李四".to_string(),
                negated: true,
            },
            &schema,
        )
        .is_none());
    }

    #[test]
    fn schema_fingerprint_separates_law_systems() {
        let empty = CanonicalSchema::new();
        let with_alias = schema();
        let plan = find_object("张三", "老师");
        let left = canonicalize_plan(&plan, &empty).expect("应可归一");
        let right = canonicalize_plan(&plan, &with_alias).expect("应可归一");
        assert_ne!(left.key, right.key, "不同定律体系下的键不能混用");
    }

    #[test]
    fn unsupported_plans_stay_unknown_unless_exactly_equal() {
        let schema = CanonicalSchema::new();
        let path = QueryPlan::PathBetween {
            start: "张三".to_string(),
            goal: "王五".to_string(),
            max_hops: 3,
        };
        let other = QueryPlan::PathBetween {
            start: "张三".to_string(),
            goal: "王五".to_string(),
            max_hops: 4,
        };
        assert!(!plans_equivalent(&path, &other, &schema));
        assert!(plans_equivalent(&path, &path.clone(), &schema));
        assert!(canonicalize_plan(&path, &schema).is_err());
    }
}

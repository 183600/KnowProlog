use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use crate::law_book::{LawBook, LawKind};
use crate::predicate_registry::PredicateRegistry;
use crate::resolution_engine::{Atom, Clause, DerivedFact, ResolutionEngine};
use crate::type_system::{TypeSystem, TypeVerdict};
/// 单次前向推导默认最多跑多少轮。
pub const DEFAULT_MAX_ROUNDS: usize = 24;
/// 单次前向推导默认最多保留多少条派生事实。
pub const DEFAULT_MAX_DERIVED: usize = 8_192;
/// 会话内最多保留多少条提交回执（更早的回执仍留在编年史/账本里）。
pub const MAX_RECEIPTS: usize = 64;
/// 回执中每类原子最多保留多少条（计数仍是全量）。
pub const MAX_RECEIPT_ITEMS: usize = 64;
/// 面向用户渲染时每类最多展示多少条。
pub const MAX_RENDER_ITEMS: usize = 8;
/// 单次理由解释最多展开多少个节点。
pub const MAX_EXPLAIN_NODES: usize = 128;
/// 单次理由解释的最大递归深度。
pub const MAX_EXPLAIN_DEPTH: usize = 10;

// ---------------------------------------------------------------------------
// 变更集
// ---------------------------------------------------------------------------
/// 一次信念修订中的单个操作。
#[derive(Debug, Clone)]
pub enum ChangeOp {
    AssertFact(Atom),
    RetractFact(Atom),
    AddRule(Clause),
    RemoveRule(Clause),
}
impl ChangeOp {
    pub fn describe(&self) -> String {
        match self {
            ChangeOp::AssertFact(atom) => format!("断言 {}", atom_key(atom)),
            ChangeOp::RetractFact(atom) => format!("撤回 {}", atom_key(atom)),
            ChangeOp::AddRule(clause) => format!("加入规则 {}", clause_key(clause)),
            ChangeOp::RemoveRule(clause) => format!("移除规则 {}", clause_key(clause)),
        }
    }

    pub fn inverse(&self) -> Self {
        match self {
            ChangeOp::AssertFact(atom) => ChangeOp::RetractFact(atom.clone()),
            ChangeOp::RetractFact(atom) => ChangeOp::AssertFact(atom.clone()),
            ChangeOp::AddRule(clause) => ChangeOp::RemoveRule(clause.clone()),
            ChangeOp::RemoveRule(clause) => ChangeOp::AddRule(clause.clone()),
        }
    }

    fn canonical(&self) -> String {
        self.describe()
    }
}
/// 一次提交的变更集：操作列表 + 可选的目标版本（乐观并发控制）。
#[derive(Debug, Clone)]
pub struct ChangeSet {
    pub label: String,
    pub ops: Vec<ChangeOp>,
    pub expected_revision: Option<u64>,
}
impl ChangeSet {
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            ops: Vec::new(),
            expected_revision: None,
        }
    }

    pub fn assert(mut self, predicate: &str, subject: &str, object: &str) -> Self {
        self.ops
            .push(ChangeOp::AssertFact(Atom::rel(predicate, subject, object)));
        self
    }

    pub fn retract(mut self, predicate: &str, subject: &str, object: &str) -> Self {
        self.ops
            .push(ChangeOp::RetractFact(Atom::rel(predicate, subject, object)));
        self
    }

    pub fn assert_atom(mut self, atom: Atom) -> Self {
        self.ops.push(ChangeOp::AssertFact(atom));
        self
    }

    pub fn retract_atom(mut self, atom: Atom) -> Self {
        self.ops.push(ChangeOp::RetractFact(atom));
        self
    }

    pub fn add_rule(mut self, clause: Clause) -> Self {
        self.ops.push(ChangeOp::AddRule(clause));
        self
    }

    pub fn remove_rule(mut self, clause: Clause) -> Self {
        self.ops.push(ChangeOp::RemoveRule(clause));
        self
    }

    pub fn with_expected_revision(mut self, revision: u64) -> Self {
        self.expected_revision = Some(revision);
        self
    }

    pub fn push(&mut self, op: ChangeOp) {
        self.ops.push(op);
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    pub fn fingerprint(&self) -> String {
        let mut material = String::new();
        for op in &self.ops {
            material.push_str(&op.canonical());
            material.push('\n');
        }
        format!("cs:{}", fnv1a(material.as_bytes()))
    }

    pub fn render_zh(&self) -> String {
        let mut lines = vec![format!(
            "变更集「{}」：{} 个操作（{}）",
            self.label,
            self.ops.len(),
            self.fingerprint()
        )];
        for op in &self.ops {
            lines.push(format!("  · {}", op.describe()));
        }
        lines.join("\n")
    }
}

// ---------------------------------------------------------------------------
// 约束
// ---------------------------------------------------------------------------
/// 提交闸门使用的声明约束。全部来自已有的谓词画像/类型格，不在这里重新发明语义。
#[derive(Debug, Clone)]
pub struct RevisionConstraints {
    pub functional: BTreeSet<String>,
    pub domain: BTreeMap<String, BTreeSet<String>>,
    pub range: BTreeMap<String, BTreeSet<String>>,
    pub types: TypeSystem,
    /// 哪些谓词表示「实体 → 类型」断言（默认 `type` / `rdf:type`）。
    pub type_predicates: BTreeSet<String>,
}
impl Default for RevisionConstraints {
    fn default() -> Self {
        Self {
            functional: BTreeSet::new(),
            domain: BTreeMap::new(),
            range: BTreeMap::new(),
            types: TypeSystem::new(),
            type_predicates: BTreeSet::from(["type".to_string(), "rdf:type".to_string()]),
        }
    }
}
impl RevisionConstraints {
    pub fn new() -> Self {
        Self::default()
    }

    /// 从谓词画像编译约束：功能/定义域/值域都按规范谓词登记。
    pub fn from_registry(registry: &PredicateRegistry) -> Self {
        let mut constraints = Self::new();
        for canonical in registry.canonical_names() {
            let Some(profile) = registry.profile(canonical) else {
                continue;
            };
            if profile.functional {
                constraints.functional.insert(canonical.to_string());
            }
            if !profile.domain.is_empty() {
                constraints
                    .domain
                    .insert(canonical.to_string(), profile.domain.clone());
            }
            if !profile.range.is_empty() {
                constraints
                    .range
                    .insert(canonical.to_string(), profile.range.clone());
            }
        }
        constraints
    }

    /// 从律典编译可用的提交约束。律典的 Domain/Range 只保存谓词名与说明，
    /// 因此这里只取功能律；类型约束应由调用方用 [`Self::from_registry`] 补充。
    pub fn from_law_book(book: &LawBook) -> Self {
        let mut constraints = Self::new();
        for law in book.laws() {
            if !law.is_active() || law.kind != LawKind::Functional {
                continue;
            }
            if let Some(predicate) = law.terms.first().filter(|term| !term.is_empty()) {
                constraints.functional.insert(predicate.clone());
            }
        }
        constraints
    }

    pub fn require_functional(mut self, predicate: &str) -> Self {
        self.functional.insert(predicate.to_string());
        self
    }

    pub fn require_domain<I, S>(mut self, predicate: &str, types: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.domain.insert(
            predicate.to_string(),
            types.into_iter().map(Into::into).collect(),
        );
        self
    }

    pub fn require_range<I, S>(mut self, predicate: &str, types: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.range.insert(
            predicate.to_string(),
            types.into_iter().map(Into::into).collect(),
        );
        self
    }

    pub fn with_types(mut self, types: TypeSystem) -> Self {
        self.types = types;
        self
    }
}

// ---------------------------------------------------------------------------
// 冲突
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ConflictKind {
    FunctionalViolation,
    DomainViolation,
    RangeViolation,
    NonGroundFact,
    UnsafeRule,
    UnconditionalRule,
    DuplicateRule,
    RuleResurrection,
    MissingRule,
    MissingRetraction,
    DuplicateAssertion,
    StalePlan,
    EmptyChangeSet,
    DerivationBudget,
}
impl ConflictKind {
    pub fn label_zh(self) -> &'static str {
        match self {
            ConflictKind::FunctionalViolation => "功能谓词多值",
            ConflictKind::DomainViolation => "定义域冲突",
            ConflictKind::RangeViolation => "值域冲突",
            ConflictKind::NonGroundFact => "事实含未绑定变量",
            ConflictKind::UnsafeRule => "规则头部变量未在体部绑定",
            ConflictKind::UnconditionalRule => "无条件规则",
            ConflictKind::DuplicateRule => "重复规则",
            ConflictKind::RuleResurrection => "规则复活已撤回事实",
            ConflictKind::MissingRule => "撤回不存在的规则",
            ConflictKind::MissingRetraction => "撤回不存在的事实",
            ConflictKind::DuplicateAssertion => "重复断言",
            ConflictKind::StalePlan => "计划基于旧版本",
            ConflictKind::EmptyChangeSet => "空变更集",
            ConflictKind::DerivationBudget => "推导预算截断",
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ConflictSeverity {
    Warning,
    Blocking,
}
impl ConflictSeverity {
    pub fn label_zh(self) -> &'static str {
        match self {
            ConflictSeverity::Warning => "提示",
            ConflictSeverity::Blocking => "阻止提交",
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RevisionConflict {
    pub kind: ConflictKind,
    pub severity: ConflictSeverity,
    pub detail: String,
    /// 相关的规范原子（稳定排序，便于逐字节断言）。
    pub atoms: Vec<String>,
}
impl RevisionConflict {
    fn new(
        kind: ConflictKind,
        severity: ConflictSeverity,
        detail: impl Into<String>,
        mut atoms: Vec<String>,
    ) -> Self {
        atoms.sort();
        atoms.dedup();
        Self {
            kind,
            severity,
            detail: detail.into(),
            atoms,
        }
    }

    fn signature(&self) -> String {
        format!("{:?}|{}", self.kind, self.atoms.join("|"))
    }
}

// ---------------------------------------------------------------------------
// 影响报告
// ---------------------------------------------------------------------------
/// 影响条目的全量计数（回执裁剪向量时仍保留真实总数）。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ImpactCounts {
    pub direct_added: usize,
    pub direct_removed: usize,
    pub derived_added: usize,
    pub derived_retracted: usize,
    pub newly_believed: usize,
    pub no_longer_believed: usize,
    pub predicted_derived_added: usize,
    pub predicted_derived_retracted: usize,
    pub unpredicted_added: usize,
    pub unpredicted_retracted: usize,
    pub affected_predicates: usize,
}
/// 一次修订的影响。`predicted_*` 来自理由图，`*_added/retracted` 是完整前向
/// 重算的事实；`unpredicted_*` 非空时说明理由图没覆盖住，必须标记回退。
#[derive(Debug, Clone, Default)]
pub struct ImpactReport {
    pub direct_added: Vec<String>,
    pub direct_removed: Vec<String>,
    pub derived_added: Vec<String>,
    pub derived_retracted: Vec<String>,
    pub newly_believed: Vec<String>,
    pub no_longer_believed: Vec<String>,
    pub predicted_derived_added: Vec<String>,
    pub predicted_derived_retracted: Vec<String>,
    pub unpredicted_added: Vec<String>,
    pub unpredicted_retracted: Vec<String>,
    pub affected_predicates: Vec<String>,
    /// 理由图没覆盖住或推导被预算截断时，影响集合仍由完整前向重算给出，
    /// 但不能宣称「局部影响已完备」。
    pub fallback_full_recompute: bool,
    pub complete: bool,
    pub note: Option<String>,
    /// 全量计数；回执裁剪向量后仍可据此报告真实影响规模。
    pub counts: ImpactCounts,
    /// 回执裁剪时省略的条数（计划中的报告不裁剪）。
    pub omitted_items: usize,
}
impl ImpactReport {
    pub fn is_empty(&self) -> bool {
        self.counts.direct_added == 0
            && self.counts.direct_removed == 0
            && self.counts.derived_added == 0
            && self.counts.derived_retracted == 0
    }

    pub fn changed_atom_count(&self) -> usize {
        self.counts.newly_believed + self.counts.no_longer_believed
    }

    fn clipped(mut self, max: usize) -> Self {
        let mut omitted = 0usize;
        for items in [
            &mut self.direct_added,
            &mut self.direct_removed,
            &mut self.derived_added,
            &mut self.derived_retracted,
            &mut self.newly_believed,
            &mut self.no_longer_believed,
            &mut self.predicted_derived_added,
            &mut self.predicted_derived_retracted,
            &mut self.unpredicted_added,
            &mut self.unpredicted_retracted,
            &mut self.affected_predicates,
        ] {
            if items.len() > max {
                omitted += items.len() - max;
                items.truncate(max);
            }
        }
        self.omitted_items = omitted;
        self
    }

    pub fn render_zh(&self) -> String {
        let mut lines = vec![format!(
            "影响：直接 +{} / -{}；派生 +{} / -{}；净相信 +{} / -{}",
            self.counts.direct_added,
            self.counts.direct_removed,
            self.counts.derived_added,
            self.counts.derived_retracted,
            self.counts.newly_believed,
            self.counts.no_longer_believed
        )];
        let push_items =
            |title: &str, items: &[String], total: usize, lines: &mut Vec<String>| {
                if total == 0 {
                    return;
                }
                lines.push(format!("{}（{}）：", title, total));
                for item in items.iter().take(MAX_RENDER_ITEMS) {
                    lines.push(format!("  · {}", item));
                }
                if total > MAX_RENDER_ITEMS {
                    lines.push(format!("  …另有 {} 条", total - MAX_RENDER_ITEMS));
                }
            };
        push_items(
            "新增相信",
            &self.newly_believed,
            self.counts.newly_believed,
            &mut lines,
        );
        push_items(
            "不再相信",
            &self.no_longer_believed,
            self.counts.no_longer_believed,
            &mut lines,
        );
        if !self.affected_predicates.is_empty() {
            lines.push(format!(
                "受影响谓词：{}",
                self.affected_predicates.join("、")
            ));
        }
        if self.counts.unpredicted_added > 0 || self.counts.unpredicted_retracted > 0 {
            lines.push(format!(
                "理由图未覆盖：新增 {} 条、撤回 {} 条（已回退完整重算）",
                self.counts.unpredicted_added, self.counts.unpredicted_retracted
            ));
        }
        lines.push(if self.fallback_full_recompute {
            match &self.note {
                Some(note) => format!("影响完备性：回退完整重算（{}）", note),
                None => "影响完备性：回退完整重算".to_string(),
            }
        } else if self.complete {
            "影响完备性：局部理由图覆盖本次变化".to_string()
        } else {
            "影响完备性：未完整评估".to_string()
        });
        if self.omitted_items > 0 {
            lines.push(format!("（回执裁剪了 {} 条影响条目）", self.omitted_items));
        }
        lines.join("\n")
    }
}

// ---------------------------------------------------------------------------
// 理由图
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub enum JustificationNode {
    Fact { atom: Atom },
    Derived {
        atom: Atom,
        rule_source: String,
        confidence: f64,
    },
}
impl JustificationNode {
    pub fn atom(&self) -> &Atom {
        match self {
            JustificationNode::Fact { atom } | JustificationNode::Derived { atom, .. } => atom,
        }
    }

    pub fn is_fact(&self) -> bool {
        matches!(self, JustificationNode::Fact { .. })
    }
}
/// 理由图：从基事实/派生事实出发，记录「谁支持了谁」。边方向为
/// `前提 → 结论`；`dependents_of` 反向遍历得到所有受影响的结论。
#[derive(Debug, Clone, Default)]
pub struct JustificationGraph {
    nodes: BTreeMap<String, JustificationNode>,
    supports: BTreeMap<String, BTreeSet<String>>,
    dependents: BTreeMap<String, BTreeSet<String>>,
}
impl JustificationGraph {
    fn build(base: &BTreeMap<String, Atom>, derived: &BTreeMap<String, DerivedFact>) -> Self {
        let mut graph = Self::default();
        for (key, atom) in base {
            graph.nodes.insert(
                key.clone(),
                JustificationNode::Fact { atom: atom.clone() },
            );
        }
        for (key, fact) in derived {
            graph.nodes.entry(key.clone()).or_insert_with(|| {
                JustificationNode::Derived {
                    atom: fact.atom.clone(),
                    rule_source: fact.rule_source.clone(),
                    confidence: fact.confidence,
                }
            });
        }
        for (key, fact) in derived {
            for premise in &fact.premises {
                let premise_key = atom_key(premise);
                graph
                    .nodes
                    .entry(premise_key.clone())
                    .or_insert_with(|| JustificationNode::Fact {
                        atom: premise.clone(),
                    });
                if premise_key == *key {
                    continue;
                }
                graph
                    .supports
                    .entry(key.clone())
                    .or_default()
                    .insert(premise_key.clone());
                graph
                    .dependents
                    .entry(premise_key)
                    .or_default()
                    .insert(key.clone());
            }
        }
        graph
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn node(&self, atom: &Atom) -> Option<&JustificationNode> {
        self.nodes.get(&atom_key(atom))
    }

    pub fn supports_of(&self, atom: &Atom) -> Vec<String> {
        self.supports
            .get(&atom_key(atom))
            .map(|set| set.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// 直接或间接依赖该原子的结论（不含自身）。
    pub fn dependents_of(&self, atom: &Atom) -> Vec<String> {
        self.dependents_of_key(&atom_key(atom))
    }

    fn dependents_of_key(&self, key: &str) -> Vec<String> {
        let mut out = BTreeSet::new();
        let mut queue = VecDeque::from([key.to_string()]);
        while let Some(current) = queue.pop_front() {
            if let Some(next) = self.dependents.get(&current) {
                for dependent in next {
                    if out.insert(dependent.clone()) {
                        queue.push_back(dependent.clone());
                    }
                }
            }
        }
        out.into_iter().collect()
    }

    /// 沿支持边展开一条有界、确定性的理由链。
    pub fn explain(&self, atom: &Atom) -> Option<String> {
        let key = atom_key(atom);
        if !self.nodes.contains_key(&key) {
            return None;
        }
        let mut seen = BTreeSet::from([key.clone()]);
        let mut budget = MAX_EXPLAIN_NODES;
        Some(self.explain_key(&key, 0, &mut seen, &mut budget))
    }

    fn explain_key(
        &self,
        key: &str,
        depth: usize,
        seen: &mut BTreeSet<String>,
        budget: &mut usize,
    ) -> String {
        let indent = "  ".repeat(depth);
        if *budget == 0 {
            return format!("{}…（理由节点预算已用尽）", indent);
        }
        *budget -= 1;
        let Some(node) = self.nodes.get(key) else {
            return format!("{}{}（无记录）", indent, key);
        };
        match node {
            JustificationNode::Fact { .. } => format!("{}{}（事实）", indent, key),
            JustificationNode::Derived {
                rule_source,
                confidence,
                ..
            } => {
                let mut lines = vec![format!(
                    "{}{}（由「{}」推出，置信度 {:.2}）",
                    indent,
                    key,
                    clip(rule_source, 72),
                    confidence
                )];
                if depth >= MAX_EXPLAIN_DEPTH {
                    lines.push(format!("{}  …（深度上限）", indent));
                    return lines.join("\n");
                }
                if let Some(supports) = self.supports.get(key) {
                    for support in supports {
                        if !seen.insert(support.clone()) {
                            lines.push(format!("{}  ← {}（已在理由链中）", indent, support));
                            continue;
                        }
                        lines.push(self.explain_key(support, depth + 1, seen, budget));
                    }
                }
                lines.join("\n")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 计划与回执
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
pub struct RevisionPlan {
    pub id: String,
    pub base_revision: u64,
    pub changeset: ChangeSet,
    pub strict: bool,
    pub applied_ops: usize,
    pub conflicts: Vec<RevisionConflict>,
    pub impact: ImpactReport,
    effective_ops: Vec<ChangeOp>,
    candidate: ResolutionEngine,
}
impl RevisionPlan {
    pub fn is_blocked(&self) -> bool {
        self.conflicts
            .iter()
            .any(|conflict| conflict.severity == ConflictSeverity::Blocking)
    }

    pub fn blocking_conflicts(&self) -> impl Iterator<Item = &RevisionConflict> {
        self.conflicts
            .iter()
            .filter(|conflict| conflict.severity == ConflictSeverity::Blocking)
    }

    pub fn render_zh(&self) -> String {
        let mut lines = vec![format!(
            "修订计划 {}（基于版本 {}，{}）",
            self.id,
            self.base_revision,
            if self.is_blocked() {
                "将被阻止"
            } else {
                "可提交"
            }
        )];
        if self.conflicts.is_empty() {
            lines.push("一致性闸门：未发现冲突".to_string());
        } else {
            lines.push(format!("一致性闸门：{} 条", self.conflicts.len()));
            for conflict in self.conflicts.iter().take(MAX_RENDER_ITEMS) {
                lines.push(format!(
                    "  · [{}] {}：{}",
                    conflict.severity.label_zh(),
                    conflict.kind.label_zh(),
                    conflict.detail
                ));
            }
            if self.conflicts.len() > MAX_RENDER_ITEMS {
                lines.push(format!(
                    "  …另有 {} 条冲突",
                    self.conflicts.len() - MAX_RENDER_ITEMS
                ));
            }
        }
        lines.push(self.impact.render_zh());
        lines.join("\n")
    }
}
#[derive(Debug, Clone)]
pub struct CommitReceipt {
    pub id: String,
    pub label: String,
    pub base_revision: u64,
    pub revision: u64,
    pub applied_ops: usize,
    pub impact: ImpactReport,
    pub conflicts: Vec<RevisionConflict>,
    pub ops: Vec<ChangeOp>,
    pub digest: String,
}
impl CommitReceipt {
    pub fn changed_beliefs(&self) -> usize {
        self.impact.changed_atom_count()
    }

    pub fn render_zh(&self) -> String {
        let mut lines = vec![format!(
            "提交回执 {}：{} → 版本 {}（{} 个操作生效）",
            self.id, self.base_revision, self.revision, self.applied_ops
        )];
        lines.push(format!("变更集：{}", self.label));
        lines.push(self.impact.render_zh());
        if !self.conflicts.is_empty() {
            lines.push(format!(
                "提交时保留 {} 条提示（无阻止级冲突）",
                self.conflicts.len()
            ));
        }
        lines.join("\n")
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommitError {
    StalePlan { expected: u64, actual: u64 },
    Blocked { count: usize, first: String },
    NothingToUndo,
    NoReceipt(String),
    NotLatest { receipt_revision: u64, current: u64 },
}
impl fmt::Display for CommitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CommitError::StalePlan { expected, actual } => write!(
                f,
                "计划基于版本 {}，当前已是版本 {}；请重新生成计划",
                expected, actual
            ),
            CommitError::Blocked { count, first } => {
                write!(f, "提交被一致性闸门阻止（{} 条冲突）：{}", count, first)
            }
            CommitError::NothingToUndo => write!(f, "没有可撤销的提交回执"),
            CommitError::NoReceipt(id) => write!(f, "找不到回执 {}", id),
            CommitError::NotLatest {
                receipt_revision,
                current,
            } => write!(
                f,
                "只能撤销最新回执（回执版本 {}，当前版本 {}）",
                receipt_revision, current
            ),
        }
    }
}
impl std::error::Error for CommitError {}

// ---------------------------------------------------------------------------
// 信念状态
// ---------------------------------------------------------------------------
/// 可提交的信念状态：引擎 + 约束 + 版本 + 有界回执史。
#[derive(Clone)]
pub struct BeliefState {
    engine: ResolutionEngine,
    constraints: RevisionConstraints,
    revision: u64,
    receipts: Vec<CommitReceipt>,
    max_rounds: usize,
    max_derived: usize,
    strict: bool,
}
impl BeliefState {
    pub fn new(engine: ResolutionEngine) -> Self {
        Self {
            engine,
            constraints: RevisionConstraints::new(),
            revision: 0,
            receipts: Vec::new(),
            max_rounds: DEFAULT_MAX_ROUNDS,
            max_derived: DEFAULT_MAX_DERIVED,
            strict: true,
        }
    }

    pub fn with_constraints(engine: ResolutionEngine, constraints: RevisionConstraints) -> Self {
        let mut state = Self::new(engine);
        state.constraints = constraints;
        state
    }

    pub fn with_limits(mut self, max_rounds: usize, max_derived: usize) -> Self {
        self.max_rounds = max_rounds.max(1);
        self.max_derived = max_derived.max(1);
        self
    }

    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    pub fn set_strict(&mut self, strict: bool) {
        self.strict = strict;
    }

    pub fn strict(&self) -> bool {
        self.strict
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn engine(&self) -> &ResolutionEngine {
        &self.engine
    }

    pub fn constraints(&self) -> &RevisionConstraints {
        &self.constraints
    }

    pub fn receipts(&self) -> &[CommitReceipt] {
        &self.receipts
    }

    pub fn last_receipt(&self) -> Option<&CommitReceipt> {
        self.receipts.last()
    }

    /// 装配期直接注入事实；不产生回执，也不推进版本（用于构造初始世界）。
    pub fn seed_fact(&mut self, atom: Atom) -> bool {
        self.engine.add_fact_atom(atom)
    }

    /// 装配期直接注入规则；不产生回执，也不推进版本（用于构造初始世界）。
    pub fn seed_rule(&mut self, clause: Clause) {
        self.engine.add_clause(clause);
    }

    pub fn facts(&self) -> Vec<Atom> {
        let mut facts: Vec<Atom> = self.engine.all_facts().cloned().collect();
        facts.sort_by_key(atom_key);
        facts
    }

    pub fn rule_count(&self) -> usize {
        self.engine.all_clauses().len()
    }

    pub fn query(&self, goals: &[Atom], query_vars: &[String]) -> Vec<crate::resolution_engine::Solution> {
        self.engine.solve(goals, query_vars)
    }

    /// 当前状态的一致性报告（只读，不推进版本）。
    pub fn check(&self) -> ConsistencyReport {
        let closure = self.closure(&self.engine);
        let mut conflicts = functional_conflicts(&closure, &self.constraints);
        conflicts.extend(schema_conflicts(&closure, &self.constraints));
        conflicts.extend(rule_shape_conflicts(&self.engine));
        sort_conflicts(&mut conflicts);
        ConsistencyReport {
            conflicts,
            facts: closure.base.len(),
            derived: closure.derived.len(),
            truncated: closure.truncated,
        }
    }

    /// 解释一个当前相信的原子：事实直接说明，派生事实沿理由图展开。
    pub fn explain(&self, atom: &Atom) -> Option<String> {
        let closure = self.closure(&self.engine);
        closure.graph.explain(atom)
    }

    /// 预览一次变更：完整计算，但不修改自身。
    pub fn plan(&self, changeset: ChangeSet) -> RevisionPlan {
        let base_closure = self.closure(&self.engine);

        if let Some(expected) = changeset.expected_revision {
            if expected != self.revision {
                return RevisionPlan {
                    id: format!("plan:{}:stale", self.revision),
                    base_revision: self.revision,
                    changeset,
                    strict: self.strict,
                    applied_ops: 0,
                    conflicts: vec![RevisionConflict::new(
                        ConflictKind::StalePlan,
                        ConflictSeverity::Blocking,
                        format!(
                            "变更集要求版本 {}，当前版本 {}",
                            expected, self.revision
                        ),
                        Vec::new(),
                    )],
                    impact: ImpactReport::default(),
                    effective_ops: Vec::new(),
                    candidate: self.engine.clone(),
                };
            }
        }

        let mut conflicts = Vec::new();
        if changeset.is_empty() {
            conflicts.push(RevisionConflict::new(
                ConflictKind::EmptyChangeSet,
                ConflictSeverity::Warning,
                "变更集没有任何操作",
                Vec::new(),
            ));
        }

        validate_ops(&changeset, &base_closure, &mut conflicts);

        let mut candidate = self.engine.clone();
        let effects = apply_ops(&mut candidate, &changeset.ops);
        let candidate_closure = self.closure(&candidate);
        let impact = build_impact(&base_closure, &candidate_closure, &effects, self.strict);

        let base_functional = functional_conflicts(&base_closure, &self.constraints);
        let base_signatures: BTreeSet<String> = base_functional
            .iter()
            .map(RevisionConflict::signature)
            .collect();
        for conflict in functional_conflicts(&candidate_closure, &self.constraints) {
            if !base_signatures.contains(&conflict.signature()) {
                conflicts.push(conflict);
            }
        }
        let base_schema = schema_conflicts(&base_closure, &self.constraints);
        let base_schema_signatures: BTreeSet<String> = base_schema
            .iter()
            .map(RevisionConflict::signature)
            .collect();
        for conflict in schema_conflicts(&candidate_closure, &self.constraints) {
            if !base_schema_signatures.contains(&conflict.signature()) {
                conflicts.push(conflict);
            }
        }

        let base_rule_conflicts = rule_shape_conflicts(&self.engine);
        let base_rule_signatures: BTreeSet<String> = base_rule_conflicts
            .iter()
            .map(RevisionConflict::signature)
            .collect();
        for conflict in rule_shape_conflicts(&candidate) {
            if !base_rule_signatures.contains(&conflict.signature()) {
                conflicts.push(conflict);
            }
        }
        conflicts.extend(resurrection_conflicts(
            &changeset.ops,
            &candidate_closure,
            self.strict,
        ));
        if candidate_closure.truncated {
            conflicts.push(RevisionConflict::new(
                ConflictKind::DerivationBudget,
                ConflictSeverity::Warning,
                format!(
                    "前向推导达到 {} 条上限，影响集合可能不完整",
                    self.max_derived
                ),
                Vec::new(),
            ));
        }
        if impact.fallback_full_recompute {
            conflicts.push(RevisionConflict::new(
                ConflictKind::DerivationBudget,
                ConflictSeverity::Warning,
                impact
                    .note
                    .clone()
                    .unwrap_or_else(|| "理由图未覆盖全部变化".to_string()),
                Vec::new(),
            ));
        }

        if self.strict {
            for conflict in &mut conflicts {
                if matches!(
                    conflict.kind,
                    ConflictKind::RuleResurrection
                        | ConflictKind::DuplicateRule
                        | ConflictKind::UnconditionalRule
                ) {
                    conflict.severity = ConflictSeverity::Blocking;
                }
            }
        }
        sort_conflicts(&mut conflicts);
        let digest = state_digest(&candidate);
        let id = format!(
            "plan:{}:{}",
            self.revision,
            fnv1a(format!("{}|{}", changeset.fingerprint(), digest).as_bytes())
        );

        RevisionPlan {
            id,
            base_revision: self.revision,
            changeset,
            strict: self.strict,
            applied_ops: effects.applied,
            conflicts,
            impact,
            effective_ops: effects.effective_ops.clone(),
            candidate,
        }
    }

    /// 提交计划：版本不匹配或存在阻止级冲突时拒绝，状态保持不变。
    pub fn commit(&mut self, plan: RevisionPlan) -> Result<CommitReceipt, CommitError> {
        if plan.base_revision != self.revision {
            return Err(CommitError::StalePlan {
                expected: plan.base_revision,
                actual: self.revision,
            });
        }
        if plan.is_blocked() {
            let first = plan
                .conflicts
                .iter()
                .find(|conflict| conflict.severity == ConflictSeverity::Blocking)
                .map(|conflict| {
                    format!("{}：{}", conflict.kind.label_zh(), conflict.detail)
                })
                .unwrap_or_default();
            return Err(CommitError::Blocked {
                count: plan.blocking_conflicts().count(),
                first,
            });
        }

        let new_revision = self.revision.saturating_add(1);
        let digest = state_digest(&plan.candidate);
        let id = format!(
            "commit:{}:{}",
            new_revision,
            fnv1a(format!("{}|{}", plan.changeset.fingerprint(), digest).as_bytes())
        );
        let receipt = CommitReceipt {
            id,
            label: plan.changeset.label.clone(),
            base_revision: self.revision,
            revision: new_revision,
            applied_ops: plan.applied_ops,
            impact: plan.impact.clipped(MAX_RECEIPT_ITEMS),
            conflicts: plan.conflicts.clone(),
            ops: plan.effective_ops.clone(),
            digest,
        };
        self.engine = plan.candidate;
        self.revision = new_revision;
        if self.receipts.len() >= MAX_RECEIPTS {
            self.receipts.remove(0);
        }
        self.receipts.push(receipt.clone());
        Ok(receipt)
    }

    /// 预览并提交的便捷入口。
    pub fn apply(&mut self, changeset: ChangeSet) -> Result<CommitReceipt, CommitError> {
        let plan = self.plan(changeset);
        self.commit(plan)
    }

    /// 撤销最新一条回执：用逆操作再提交一次，历史回执不被改写。
    pub fn undo_last(&mut self) -> Result<CommitReceipt, CommitError> {
        let Some(receipt) = self.receipts.last().cloned() else {
            return Err(CommitError::NothingToUndo);
        };
        if receipt.revision != self.revision {
            return Err(CommitError::NotLatest {
                receipt_revision: receipt.revision,
                current: self.revision,
            });
        }
        let mut changeset = ChangeSet::new(format!("撤销 {}", receipt.id))
            .with_expected_revision(self.revision);
        for op in receipt.ops.iter().rev() {
            changeset.push(op.inverse());
        }
        self.apply(changeset)
    }

    /// 按回执 id 撤销（仅允许最新一条；历史只能查看，不能穿越修改）。
    pub fn undo(&mut self, receipt_id: &str) -> Result<CommitReceipt, CommitError> {
        let Some(receipt) = self.receipts.iter().find(|item| item.id == receipt_id) else {
            return Err(CommitError::NoReceipt(receipt_id.to_string()));
        };
        if receipt.revision != self.revision {
            return Err(CommitError::NotLatest {
                receipt_revision: receipt.revision,
                current: self.revision,
            });
        }
        self.undo_last()
    }

    fn closure(&self, engine: &ResolutionEngine) -> Closure {
        Closure::build(engine, self.max_rounds, self.max_derived)
    }
}

// ---------------------------------------------------------------------------
// 一致性报告
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Default)]
pub struct ConsistencyReport {
    pub conflicts: Vec<RevisionConflict>,
    pub facts: usize,
    pub derived: usize,
    pub truncated: bool,
}
impl ConsistencyReport {
    pub fn is_clean(&self) -> bool {
        self.conflicts.is_empty()
    }

    pub fn blocking_count(&self) -> usize {
        self.conflicts
            .iter()
            .filter(|conflict| conflict.severity == ConflictSeverity::Blocking)
            .count()
    }

    pub fn render_zh(&self) -> String {
        let mut lines = vec![format!(
            "信念一致性：{} 条基事实 / {} 条派生事实 / {} 条冲突{}",
            self.facts,
            self.derived,
            self.conflicts.len(),
            if self.truncated {
                "（推导被预算截断）"
            } else {
                ""
            }
        )];
        if self.conflicts.is_empty() {
            lines.push("未发现功能谓词多值、定义域/值域冲突或规则形状问题。".to_string());
        } else {
            for conflict in self.conflicts.iter().take(MAX_RENDER_ITEMS) {
                lines.push(format!(
                    "  · [{}] {}：{}",
                    conflict.severity.label_zh(),
                    conflict.kind.label_zh(),
                    conflict.detail
                ));
            }
            if self.conflicts.len() > MAX_RENDER_ITEMS {
                lines.push(format!(
                    "  …另有 {} 条冲突",
                    self.conflicts.len() - MAX_RENDER_ITEMS
                ));
            }
        }
        lines.join("\n")
    }
}

// ---------------------------------------------------------------------------
// 内部实现
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Default)]
struct Closure {
    base: BTreeMap<String, Atom>,
    derived: BTreeMap<String, DerivedFact>,
    believed: BTreeMap<String, Atom>,
    graph: JustificationGraph,
    truncated: bool,
}
impl Closure {
    fn build(engine: &ResolutionEngine, max_rounds: usize, max_derived: usize) -> Self {
        let base: BTreeMap<String, Atom> = engine
            .all_facts()
            .map(|atom| (atom_key(atom), atom.clone()))
            .collect();
        let derived_list = engine.forward_chain(max_rounds.max(1), max_derived.max(1));
        let truncated = derived_list.len() >= max_derived.max(1);
        let mut derived = BTreeMap::new();
        for fact in derived_list {
            derived.entry(atom_key(&fact.atom)).or_insert(fact);
        }
        let mut believed = base.clone();
        for (key, fact) in &derived {
            believed.insert(key.clone(), fact.atom.clone());
        }
        let graph = JustificationGraph::build(&base, &derived);
        Self {
            base,
            derived,
            believed,
            graph,
            truncated,
        }
    }
}
#[derive(Debug, Default)]
struct OpEffects {
    effective_ops: Vec<ChangeOp>,
    added_facts: Vec<String>,
    removed_facts: Vec<String>,
    added_rule_sources: BTreeSet<String>,
    removed_rule_sources: BTreeSet<String>,
    added_rule_shapes: Vec<String>,
    removed_rule_shapes: Vec<String>,
    applied: usize,
}
fn apply_ops(engine: &mut ResolutionEngine, ops: &[ChangeOp]) -> OpEffects {
    let mut effects = OpEffects::default();
    for op in ops {
        match op {
            ChangeOp::AssertFact(atom) => {
                if engine.add_fact_atom(atom.clone()) {
                    effects.added_facts.push(atom_key(atom));
                    effects.effective_ops.push(ChangeOp::AssertFact(atom.clone()));
                    effects.applied += 1;
                }
            }
            ChangeOp::RetractFact(atom) => {
                if engine.remove_fact_atom(atom) {
                    effects.removed_facts.push(atom_key(atom));
                    effects.effective_ops.push(ChangeOp::RetractFact(atom.clone()));
                    effects.applied += 1;
                }
            }
            ChangeOp::AddRule(clause) => {
                let key = clause_key(clause);
                let exists = engine
                    .all_clauses()
                    .iter()
                    .any(|existing| clause_key(existing) == key);
                if !exists {
                    engine.add_clause(clause.clone());
                    effects.added_rule_sources.insert(clause.source.clone());
                    effects
                        .added_rule_shapes
                        .push(rule_shape_key(clause));
                    effects.effective_ops.push(ChangeOp::AddRule(clause.clone()));
                    effects.applied += 1;
                }
            }
            ChangeOp::RemoveRule(clause) => {
                let shape = rule_shape_key(clause);
                let matched: Vec<Clause> = engine
                    .all_clauses()
                    .iter()
                    .filter(|existing| rule_shape_key(existing) == shape)
                    .cloned()
                    .collect();
                let removed = engine.remove_clauses(|existing| rule_shape_key(existing) == shape);
                if removed > 0 {
                    effects
                        .removed_rule_sources
                        .extend(matched.iter().map(|existing| existing.source.clone()));
                    effects.removed_rule_shapes.push(shape);
                    for actual in matched {
                        effects.effective_ops.push(ChangeOp::RemoveRule(actual));
                    }
                    effects.applied += removed;
                }
            }
        }
    }
    effects
}
fn validate_ops(changeset: &ChangeSet, base: &Closure, conflicts: &mut Vec<RevisionConflict>) {
    for op in &changeset.ops {
        match op {
            ChangeOp::AssertFact(atom) => {
                if !atom.is_ground() {
                    conflicts.push(RevisionConflict::new(
                        ConflictKind::NonGroundFact,
                        ConflictSeverity::Blocking,
                        format!("事实 {} 含未绑定变量", atom_key(atom)),
                        vec![atom_key(atom)],
                    ));
                } else if base.base.contains_key(&atom_key(atom)) {
                    conflicts.push(RevisionConflict::new(
                        ConflictKind::DuplicateAssertion,
                        ConflictSeverity::Warning,
                        format!("{} 已经是基事实", atom_key(atom)),
                        vec![atom_key(atom)],
                    ));
                }
            }
            ChangeOp::RetractFact(atom) => {
                if !atom.is_ground() {
                    conflicts.push(RevisionConflict::new(
                        ConflictKind::NonGroundFact,
                        ConflictSeverity::Blocking,
                        format!("事实 {} 含未绑定变量", atom_key(atom)),
                        vec![atom_key(atom)],
                    ));
                } else if !base.believed.contains_key(&atom_key(atom)) {
                    conflicts.push(RevisionConflict::new(
                        ConflictKind::MissingRetraction,
                        ConflictSeverity::Warning,
                        format!("{} 当前既不是基事实也不是派生事实", atom_key(atom)),
                        vec![atom_key(atom)],
                    ));
                }
            }
            ChangeOp::AddRule(clause) => {
                if let Some(unbound) = unsafe_head_variables(clause) {
                    conflicts.push(RevisionConflict::new(
                        ConflictKind::UnsafeRule,
                        ConflictSeverity::Blocking,
                        format!(
                            "规则头部变量 {} 未在体部出现，可能推出非基事实",
                            unbound.join("、")
                        ),
                        vec![clause_key(clause)],
                    ));
                }
                if clause.body.is_empty() {
                    conflicts.push(RevisionConflict::new(
                        ConflictKind::UnconditionalRule,
                        ConflictSeverity::Warning,
                        format!("规则 {} 没有体部，等价于无条件事实", clause_key(clause)),
                        vec![clause_key(clause)],
                    ));
                }
            }
            ChangeOp::RemoveRule(_) => {}
        }
    }
}
fn unsafe_head_variables(clause: &Clause) -> Option<Vec<String>> {
    let body_vars: BTreeSet<String> = clause
        .body
        .iter()
        .flat_map(|atom| atom.terms.iter())
        .filter(|term| term.is_var())
        .map(|term| term.name().to_string())
        .collect();
    let unbound: Vec<String> = clause
        .head
        .terms
        .iter()
        .filter(|term| term.is_var())
        .map(|term| term.name().to_string())
        .filter(|name| !body_vars.contains(name))
        .collect();
    if unbound.is_empty() {
        None
    } else {
        Some(unbound)
    }
}
fn resurrection_conflicts(
    ops: &[ChangeOp],
    candidate: &Closure,
    strict: bool,
) -> Vec<RevisionConflict> {
    let mut out = Vec::new();
    for op in ops {
        let ChangeOp::RetractFact(atom) = op else {
            continue;
        };
        let key = atom_key(atom);
        if candidate.base.contains_key(&key) || !candidate.derived.contains_key(&key) {
            continue;
        }
        let source = candidate
            .derived
            .get(&key)
            .map(|fact| fact.rule_source.clone())
            .unwrap_or_else(|| "未知规则".to_string());
        out.push(RevisionConflict::new(
            ConflictKind::RuleResurrection,
            if strict {
                ConflictSeverity::Blocking
            } else {
                ConflictSeverity::Warning
            },
            format!(
                "撤回 {} 后仍可由「{}」推出；事实不会真正消失",
                key, source
            ),
            vec![key],
        ));
    }
    out
}
fn rule_shape_conflicts(engine: &ResolutionEngine) -> Vec<RevisionConflict> {
    let mut by_shape: BTreeMap<String, Vec<&Clause>> = BTreeMap::new();
    for clause in engine.all_clauses() {
        by_shape
            .entry(rule_shape_key(clause))
            .or_default()
            .push(clause);
    }
    let mut out = Vec::new();
    for (shape, clauses) in by_shape {
        if clauses.len() <= 1 {
            continue;
        }
        let sources: Vec<String> = clauses
            .iter()
            .map(|clause| clause.source.clone())
            .collect();
        out.push(RevisionConflict::new(
            ConflictKind::DuplicateRule,
            ConflictSeverity::Warning,
            format!("同一规则形状出现 {} 次（来源：{}）", clauses.len(), sources.join(" / ")),
            vec![shape],
        ));
    }
    for clause in engine.all_clauses() {
        if let Some(unbound) = unsafe_head_variables(clause) {
            out.push(RevisionConflict::new(
                ConflictKind::UnsafeRule,
                ConflictSeverity::Blocking,
                format!(
                    "规则 {} 的头部变量 {} 未在体部绑定",
                    clause_key(clause),
                    unbound.join("、")
                ),
                vec![clause_key(clause)],
            ));
        }
    }
    out
}
fn functional_conflicts(
    closure: &Closure,
    constraints: &RevisionConstraints,
) -> Vec<RevisionConflict> {
    let mut groups: BTreeMap<(String, String), BTreeSet<String>> = BTreeMap::new();
    for atom in closure.believed.values() {
        if atom.terms.len() != 2 || !constraints.functional.contains(&*atom.predicate) {
            continue;
        }
        let (Some(subject), Some(object)) = (constant(&atom.terms[0]), constant(&atom.terms[1]))
        else {
            continue;
        };
        groups
            .entry((atom.predicate.to_string(), subject.to_string()))
            .or_default()
            .insert(object.to_string());
    }
    groups
        .into_iter()
        .filter(|(_, objects)| objects.len() > 1)
        .map(|((predicate, subject), objects)| {
            let values: Vec<String> = objects.iter().cloned().collect();
            let atoms: Vec<String> = values
                .iter()
                .map(|value| format!("{}({}, {})", predicate, subject, value))
                .collect();
            RevisionConflict::new(
                ConflictKind::FunctionalViolation,
                ConflictSeverity::Blocking,
                format!(
                    "功能谓词「{}」的主语「{}」有 {} 个取值：{}",
                    predicate,
                    subject,
                    values.len(),
                    values.join(" / ")
                ),
                atoms,
            )
        })
        .collect()
}

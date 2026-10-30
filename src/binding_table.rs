//! BT：绑定表（Binding Table）——把一次成功求解落成可连接、可反查、可排序、
//! 可失效的**变量视图**。
//!
//! 事实快照回答「世界是什么」，绑定表回答「问题变量怎样接到世界上」：
//!
//! - **形状**：[`TableShape`] 保存变量、谓词骨架与固定输入；同一形状可以
//!   服务不同实体，复用的是**理解结构**，不是旧行。
//! - **覆盖范围**：[`Coverage`] 区分点表、域表、全表与部分表。点表只对固定
//!   输入成立，绝不能把 `张三 → 226` 直接实例化成 `李四 → 226`。
//! - **证据**：每一行携带 [`FactRef`]，失效检查能落到具体事实，而不是只比
//!   版本号。
//! - **版本**：表记录构造时的知识版本；版本变化或支撑事实消失时显式报告
//!   stale，绝不把旧答案当新答案复述。
//!
//! 本模块只做关系代数（过滤、投影、自然连接、排序、截断、计数）与有界
//! 台账，不写基础图谱；持久化复用 [`ContextImage`] 侧车。

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::bindings::{compare_quantity, BoundTerm};
use crate::context_image::{canonical_json, fnv1a_hex, ContextImage};
use crate::provenance::FactRef;
use crate::quantity::Comparison;

/// 绑定表格式版本。
pub const BINDING_TABLE_VERSION: u32 = 1;
/// 绑定表台账在 `ContextImage` 信封中的分节名。
pub const BINDING_TABLES_SECTION: &str = "binding_tables";
/// 默认最多保留多少张表。
pub const DEFAULT_MAX_TABLES: usize = 8;
/// 默认每张表最多保留多少行。
pub const DEFAULT_MAX_ROWS: usize = 256;
/// 报告最多逐行展示多少行。
pub const REPORT_ROW_CAP: usize = 20;

fn default_version() -> u32 {
    BINDING_TABLE_VERSION
}

fn default_max_tables() -> usize {
    DEFAULT_MAX_TABLES
}

fn default_max_rows() -> usize {
    DEFAULT_MAX_ROWS
}

fn canonical_key<T: Serialize>(value: &T) -> String {
    serde_json::to_value(value)
        .map(|value| canonical_json(&value))
        .unwrap_or_default()
}

fn terms_equal(left: &BoundTerm, right: &BoundTerm) -> bool {
    if left == right {
        return true;
    }
    matches!(compare_quantity(left, right), Comparison::Equal)
}

fn term_order(left: &BoundTerm, right: &BoundTerm) -> std::cmp::Ordering {
    match (left.is_bound(), right.is_bound()) {
        (false, false) => std::cmp::Ordering::Equal,
        (false, true) => std::cmp::Ordering::Greater,
        (true, false) => std::cmp::Ordering::Less,
        (true, true) => match compare_quantity(left, right).ordering() {
            Some(ordering) => ordering,
            None => left
                .render_raw()
                .unwrap_or_default()
                .cmp(&right.render_raw().unwrap_or_default()),
        },
    }
}

/// 形状中的一个位置：变量或常量。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ShapeTerm {
    Var { name: String },
    Const { value: String },
}

impl ShapeTerm {
    pub fn var(name: impl Into<String>) -> Self {
        ShapeTerm::Var { name: name.into() }
    }

    pub fn constant(value: impl Into<String>) -> Self {
        ShapeTerm::Const {
            value: value.into(),
        }
    }

    fn variable_name(&self) -> Option<&str> {
        match self {
            ShapeTerm::Var { name } => Some(name.as_str()),
            ShapeTerm::Const { .. } => None,
        }
    }
}

impl fmt::Display for ShapeTerm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShapeTerm::Var { name } => write!(f, "?{}", name),
            ShapeTerm::Const { value } => write!(f, "{}", value),
        }
    }
}

/// 一条谓词骨架。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShapeAtom {
    pub subject: ShapeTerm,
    pub predicate: ShapeTerm,
    pub object: ShapeTerm,
}

impl ShapeAtom {
    pub fn new(subject: ShapeTerm, predicate: ShapeTerm, object: ShapeTerm) -> Self {
        Self {
            subject,
            predicate,
            object,
        }
    }

    fn variables(&self) -> impl Iterator<Item = &str> {
        [
            self.subject.variable_name(),
            self.predicate.variable_name(),
            self.object.variable_name(),
        ]
        .into_iter()
        .flatten()
    }
}

/// 求解问题的形状：变量、谓词骨架与固定输入。
///
/// `pinned` 里的变量是**这次问句已经固定的输入**（例如 `?e = 张三`）。
/// [`Coverage::Point`] 的诚实性由调用方声明，形状本身不擅自把固定输入
/// 外推到其他实体。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TableShape {
    #[serde(default)]
    pub variables: Vec<String>,
    #[serde(default)]
    pub atoms: Vec<ShapeAtom>,
    #[serde(default)]
    pub pinned: BTreeMap<String, BoundTerm>,
}

impl TableShape {
    pub fn new<I, S>(variables: I, atoms: Vec<ShapeAtom>) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut normalized = Vec::new();
        for variable in variables {
            let variable = variable.into();
            let variable = variable.trim();
            if !variable.is_empty() && !normalized.iter().any(|seen| seen == variable) {
                normalized.push(variable.to_string());
            }
        }
        Self {
            variables: normalized,
            atoms,
            pinned: BTreeMap::new(),
        }
    }

    pub fn pin(mut self, variable: impl Into<String>, term: BoundTerm) -> Self {
        self.pinned.insert(variable.into(), term);
        self
    }

    pub fn triple(subject: ShapeTerm, predicate: ShapeTerm, object: ShapeTerm) -> Self {
        Self::new(
            ["s", "p", "o"],
            vec![ShapeAtom::new(subject, predicate, object)],
        )
    }

    pub fn validate(&self) -> Result<(), TableError> {
        if self.variables.is_empty() {
            return Err(TableError::EmptyShape);
        }
        let mut seen = BTreeSet::new();
        for variable in &self.variables {
            if !seen.insert(variable.as_str()) {
                return Err(TableError::DuplicateVariable(variable.clone()));
            }
        }
        for atom in &self.atoms {
            for variable in atom.variables() {
                if !seen.contains(variable) {
                    return Err(TableError::UnknownVariable(variable.to_string()));
                }
            }
        }
        for (variable, term) in &self.pinned {
            if !seen.contains(variable.as_str()) {
                return Err(TableError::UnknownPinned(variable.clone()));
            }
            if !term.is_bound() {
                return Err(TableError::UnboundPinned(variable.clone()));
            }
        }
        Ok(())
    }

    /// 完整形状键：包含固定输入的**取值**。
    pub fn key(&self) -> String {
        canonical_key(self)
    }

    /// 结构键：忽略固定输入的取值，只保留变量名与谓词骨架。
    ///
    /// 「那李四呢」与「张三的身高」共享同一个结构键，因此可以路由到同一
    /// 求解形状后**重新执行**；旧的点表行仍然不能直接套到李四身上。
    pub fn structural_key(&self) -> String {
        let mut skeleton = self.clone();
        skeleton.pinned.clear();
        canonical_key(&skeleton)
    }

    pub fn pinned_label(&self) -> String {
        if self.pinned.is_empty() {
            return "无固定输入".to_string();
        }
        self.pinned
            .iter()
            .map(|(name, term)| format!("?{}={}", name, term))
            .collect::<Vec<_>>()
            .join("，")
    }
}

/// 绑定表的覆盖范围。它是整套方案可信度的关键：
/// 点表只能回答同一个点，不能把一行行数据当成泛化规律。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Coverage {
    /// 只对固定输入成立（例如「张三的身高」）。
    Point,
    /// 在声明的域内穷尽（例如「所有演员的身高」）。
    Domain { label: String },
    /// 形状下的解已穷尽。
    Complete,
    /// 结果被截断/采样，只能对已保留的行做承诺。
    Partial { reason: String },
}

impl Coverage {
    pub fn domain(label: impl Into<String>) -> Self {
        Coverage::Domain {
            label: label.into(),
        }
    }

    pub fn partial(reason: impl Into<String>) -> Self {
        Coverage::Partial {
            reason: reason.into(),
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Coverage::Point => "点表",
            Coverage::Domain { .. } => "域表",
            Coverage::Complete => "全表",
            Coverage::Partial { .. } => "部分表",
        }
    }

    pub fn is_point(&self) -> bool {
        matches!(self, Coverage::Point)
    }

    pub fn is_complete(&self) -> bool {
        matches!(self, Coverage::Complete | Coverage::Domain { .. })
    }

    pub fn rows_reusable(&self) -> bool {
        self.is_complete()
    }

    pub fn rank(&self) -> u8 {
        match self {
            Coverage::Point => 0,
            Coverage::Partial { .. } => 1,
            Coverage::Domain { .. } => 2,
            Coverage::Complete => 3,
        }
    }

    pub fn caveat(&self) -> String {
        match self {
            Coverage::Point => "只对固定输入成立，不得外推到其他实体".to_string(),
            Coverage::Domain { label } => format!("在域「{}」内穷尽", label),
            Coverage::Complete => "形状下的解已穷尽".to_string(),
            Coverage::Partial { reason } => {
                format!("仅覆盖已保留的行，未证明完整（{}）", reason)
            }
        }
    }
}

/// 表来源：只用于措辞与审计，不承担身份。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TableSource {
    #[default]
    Solver,
    SearchPort,
    QueryBoard,
    Import,
    Manual,
}

impl TableSource {
    pub fn label(self) -> &'static str {
        match self {
            TableSource::Solver => "求解器",
            TableSource::SearchPort => "检索端口",
            TableSource::QueryBoard => "查询板",
            TableSource::Import => "导入",
            TableSource::Manual => "手工",
        }
    }
}

/// 一行变量绑定。`cells` 缺省即显式未绑定；`evidence` 是这一行能回溯到的
/// 基础事实。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BindingRow {
    #[serde(default)]
    pub cells: BTreeMap<String, BoundTerm>,
    #[serde(default)]
    pub evidence: Vec<FactRef>,
}

impl BindingRow {
    pub fn new(cells: BTreeMap<String, BoundTerm>) -> Self {
        Self {
            cells,
            evidence: Vec::new(),
        }
    }

    pub fn from_pairs<I, K, V>(pairs: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<BoundTerm>,
    {
        let cells = pairs
            .into_iter()
            .map(|(name, term)| (name.into(), term.into()))
            .collect();
        Self::new(cells)
    }

    pub fn with_evidence<I>(mut self, evidence: I) -> Self
    where
        I: IntoIterator<Item = FactRef>,
    {
        self.evidence = evidence.into_iter().collect();
        self.normalize_evidence();
        self
    }

    pub fn get(&self, variable: &str) -> Option<&BoundTerm> {
        self.cells.get(variable)
    }

    pub fn is_bound(&self, variable: &str) -> bool {
        self.cells
            .get(variable)
            .map(BoundTerm::is_bound)
            .unwrap_or(false)
    }

    pub fn value_text(&self, variable: &str) -> Option<String> {
        self.cells.get(variable).and_then(BoundTerm::render_raw)
    }

    pub fn has_evidence(&self) -> bool {
        !self.evidence.is_empty()
    }

    fn normalize_evidence(&mut self) {
        self.evidence.sort();
        self.evidence.dedup();
    }

    fn key(&self) -> String {
        canonical_key(&self.cells)
    }

    fn merge(&mut self, other: &BindingRow) {
        for fact in &other.evidence {
            if !self.evidence.contains(fact) {
                self.evidence.push(fact.clone());
            }
        }
        self.normalize_evidence();
    }
}

impl From<&crate::bindings::BindingSet> for BindingRow {
    fn from(bindings: &crate::bindings::BindingSet) -> Self {
        let cells = bindings
            .iter()
            .map(|(name, term)| (name.clone(), term.clone()))
            .collect();
        BindingRow::new(cells)
    }
}

/// 不可变绑定表。表只引用事实，不复制事实源；同一输入重复构造得到相同
/// 规范形状。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BindingTable {
    #[serde(default = "default_version")]
    pub version: u32,
    #[serde(default)]
    pub id: String,
    pub shape: TableShape,
    pub coverage: Coverage,
    #[serde(default)]
    pub source: TableSource,
    #[serde(default)]
    pub revision: u64,
    #[serde(default)]
    pub created_turn: u64,
    #[serde(default)]
    pub question: String,
    #[serde(default)]
    pub rows: Vec<BindingRow>,
    #[serde(default)]
    pub invalidated: Option<String>,
}

impl BindingTable {
    pub fn new(shape: TableShape, coverage: Coverage, revision: u64) -> Self {
        let digest = fnv1a_hex(format!("{}\u{1}{}", shape.key(), revision).as_bytes());
        Self {
            version: BINDING_TABLE_VERSION,
            id: format!("bt-{}-{}", revision, &digest[..12]),
            shape,
            coverage,
            source: TableSource::Solver,
            revision,
            created_turn: 0,
            question: String::new(),
            rows: Vec::new(),
            invalidated: None,
        }
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    pub fn with_source(mut self, source: TableSource) -> Self {
        self.source = source;
        self
    }

    pub fn with_question(mut self, question: impl Into<String>) -> Self {
        self.question = question.into();
        self
    }

    pub fn with_turn(mut self, turn: u64) -> Self {
        self.created_turn = turn;
        self
    }

    pub fn add_row(&mut self, mut row: BindingRow) -> Result<&mut Self, TableError> {
        self.shape.validate()?;
        for variable in row.cells.keys() {
            if !self.shape.variables.iter().any(|known| known == variable) {
                return Err(TableError::UnknownVariable(variable.clone()));
            }
        }
        for (variable, expected) in &self.shape.pinned {
            if let Some(found) = row.cells.get(variable) {
                if !terms_equal(expected, found) {
                    return Err(TableError::PinnedMismatch {
                        variable: variable.clone(),
                        expected: expected.render_raw().unwrap_or_default(),
                        found: found.render_raw().unwrap_or_default(),
                    });
                }
            } else {
                row.cells.insert(variable.clone(), expected.clone());
            }
        }
        row.normalize_evidence();
        self.rows.push(row);
        Ok(self)
    }

    pub fn push_row(
        &mut self,
        bindings: &crate::bindings::BindingSet,
        evidence: Vec<FactRef>,
    ) -> Result<&mut Self, TableError> {
        self.add_row(BindingRow::from(bindings).with_evidence(evidence))
    }

    pub fn validate(&self) -> Result<(), TableError> {
        if self.version != BINDING_TABLE_VERSION {
            return Err(TableError::InvalidVersion {
                found: self.version,
                supported: BINDING_TABLE_VERSION,
            });
        }
        self.shape.validate()?;
        for row in &self.rows {
            for variable in row.cells.keys() {
                if !self.shape.variables.iter().any(|known| known == variable) {
                    return Err(TableError::UnknownVariable(variable.clone()));
                }
            }
            for (variable, expected) in &self.shape.pinned {
                if let Some(found) = row.cells.get(variable) {
                    if !terms_equal(expected, found) {
                        return Err(TableError::PinnedMismatch {
                            variable: variable.clone(),
                            expected: expected.render_raw().unwrap_or_default(),
                            found: found.render_raw().unwrap_or_default(),
                        });
                    }
                }
            }
        }
        Ok(())
    }

    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn cell_names(&self) -> &[String] {
        &self.shape.variables
    }

    pub fn shape_key(&self) -> String {
        self.shape.key()
    }

    pub fn structural_key(&self) -> String {
        self.shape.structural_key()
    }

    pub fn evidence_facts(&self) -> BTreeSet<FactRef> {
        self.rows
            .iter()
            .flat_map(|row| row.evidence.iter().cloned())
            .collect()
    }

    pub fn invalidate(&mut self, reason: impl Into<String>) {
        self.invalidated = Some(reason.into());
    }

    pub fn is_fresh(&self, revision: u64, facts: &BTreeSet<FactRef>) -> bool {
        !self.staleness(revision, facts).is_stale()
    }

    /// 行能否直接当作当前答案：必须同版本、覆盖完整，且没有被显式失效。
    pub fn rows_reusable_at(&self, revision: u64) -> bool {
        self.revision == revision && self.invalidated.is_none() && self.coverage.rows_reusable()
    }

    pub fn staleness(&self, revision: u64, facts: &BTreeSet<FactRef>) -> Staleness {
        let missing_evidence: Vec<FactRef> = self
            .evidence_facts()
            .into_iter()
            .filter(|fact| !facts.contains(fact))
            .collect();
        Staleness {
            revision_changed: self.revision != revision,
            invalidated: self.invalidated.clone(),
            missing_evidence,
        }
    }

    pub fn filter_eq(&self, variable: &str, value: &str) -> Result<Self, TableError> {
        self.filter_term(variable, &BoundTerm::classify(value))
    }

    pub fn filter_term(&self, variable: &str, value: &BoundTerm) -> Result<Self, TableError> {
        self.validate()?;
        self.require_variable(variable)?;
        let mut table = self.clone();
        table.rows.retain(|row| {
            row.get(variable)
                .map(|cell| terms_equal(cell, value))
                .unwrap_or(false)
        });
        Ok(table)
    }

    pub fn project(&self, keep: &[&str]) -> Result<Self, TableError> {
        self.validate()?;
        if keep.is_empty() {
            return Err(TableError::EmptyProjection);
        }
        for variable in keep {
            self.require_variable(variable)?;
        }
        let keep_names: Vec<String> = keep.iter().map(|name| (*name).to_string()).collect();
        let pinned: BTreeMap<String, BoundTerm> = self
            .shape
            .pinned
            .iter()
            .filter(|(name, _)| keep_names.iter().any(|kept| kept == *name))
            .map(|(name, term)| (name.clone(), term.clone()))
            .collect();
        let atoms: Vec<ShapeAtom> = self
            .shape
            .atoms
            .iter()
            .filter(|atom| {
                atom.variables()
                    .all(|name| keep_names.iter().any(|kept| kept == name))
            })
            .cloned()
            .collect();
        let shape = TableShape {
            variables: keep_names.clone(),
            atoms,
            pinned,
        };
        let mut table = BindingTable::new(shape, self.coverage.clone(), self.revision);
        table.source = self.source;
        table.created_turn = self.created_turn;
        table.question = self.question.clone();
        table.invalidated = self.invalidated.clone();
        let mut seen: BTreeMap<String, BindingRow> = BTreeMap::new();
        for row in &self.rows {
            let mut cells = BTreeMap::new();
            for variable in &keep_names {
                if let Some(term) = row.get(variable) {
                    cells.insert(variable.clone(), term.clone());
                }
            }
            let projected = BindingRow::new(cells).with_evidence(row.evidence.clone());
            let key = projected.key();
            match seen.get_mut(&key) {
                Some(existing) => existing.merge(&projected),
                None => {
                    seen.insert(key, projected);
                }
            }
        }
        table.rows = seen.into_values().collect();
        Ok(table)
    }

    /// 自然连接：共享变量必须取值相等；任一侧是点表时拒绝，因为点表的行
    /// 不是泛化关系。
    pub fn join(&self, other: &BindingTable) -> Result<Self, TableError> {
        self.validate()?;
        other.validate()?;
        if self.coverage.is_point() || other.coverage.is_point() {
            return Err(TableError::PointJoin);
        }
        if self.revision != other.revision {
            return Err(TableError::RevisionMismatch {
                left: self.revision,
                right: other.revision,
            });
        }
        let mut variables = self.shape.variables.clone();
        for variable in &other.shape.variables {
            if !variables.contains(variable) {
                variables.push(variable.clone());
            }
        }
        let mut pinned = self.shape.pinned.clone();
        for (variable, term) in &other.shape.pinned {
            match pinned.get(variable) {
                Some(existing) if !terms_equal(existing, term) => {
                    return Err(TableError::PinnedConflict {
                        variable: variable.clone(),
                        left: existing.render_raw().unwrap_or_default(),
                        right: term.render_raw().unwrap_or_default(),
                    });
                }
                Some(_) => {}
                None => {
                    pinned.insert(variable.clone(), term.clone());
                }
            }
        }
        let mut atoms = self.shape.atoms.clone();
        for atom in &other.shape.atoms {
            if !atoms.contains(atom) {
                atoms.push(atom.clone());
            }
        }
        let mut coverage = combine_coverage(&self.coverage, &other.coverage);
        let mut table = BindingTable::new(
            TableShape {
                variables,
                atoms,
                pinned,
            },
            coverage.clone(),
            self.revision,
        );
        table.source = self.source;
        table.created_turn = self.created_turn.max(other.created_turn);
        table.question = self.question.clone();
        let shared: Vec<&String> = self
            .shape
            .variables
            .iter()
            .filter(|name| other.shape.variables.contains(name))
            .collect();
        let mut rows: BTreeMap<String, BindingRow> = BTreeMap::new();
        for left in &self.rows {
            for right in &other.rows {
                let compatible =
                    shared
                        .iter()
                        .all(|variable| match (left.get(variable), right.get(variable)) {
                            (Some(left), Some(right)) => terms_equal(left, right),
                            (None, None) => true,
                            _ => false,
                        });
                if !compatible {
                    continue;
                }
                let mut cells = left.cells.clone();
                for (variable, term) in &right.cells {
                    cells.insert(variable.clone(), term.clone());
                }
                let mut joined = BindingRow::new(cells).with_evidence(left.evidence.clone());
                joined.merge(right);
                let key = joined.key();
                match rows.get_mut(&key) {
                    Some(existing) => existing.merge(&joined),
                    None => {
                        rows.insert(key, joined);
                    }
                }
            }
        }
        let mut output: Vec<BindingRow> = rows.into_values().collect();
        if output.len() > DEFAULT_MAX_ROWS {
            output.truncate(DEFAULT_MAX_ROWS);
            coverage = Coverage::partial(format!(
                "连接结果超过每表上限 {} 行，已截断",
                DEFAULT_MAX_ROWS
            ));
        }
        table.coverage = coverage;
        table.rows = output;
        Ok(table)
    }

    pub fn sort_by(&self, variable: &str, direction: SortDirection) -> Result<Self, TableError> {
        self.validate()?;
        self.require_variable(variable)?;
        let mut table = self.clone();
        table.rows.sort_by(
            |left, right| match (left.get(variable), right.get(variable)) {
                (Some(left), Some(right)) => direction.apply(term_order(left, right)),
                (None, None) => std::cmp::Ordering::Equal,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (Some(_), None) => std::cmp::Ordering::Less,
            },
        );
        Ok(table)
    }

    pub fn top_k(&self, k: usize) -> Result<Self, TableError> {
        self.validate()?;
        let mut table = self.clone();
        if k >= table.rows.len() {
            return Ok(table);
        }
        table.rows.truncate(k);
        if !table.coverage.is_point() {
            table.coverage =
                Coverage::partial(format!("截断为前 {} 行（原 {} 行）", k, self.rows.len()));
        }
        Ok(table)
    }

    pub fn distinct(&self) -> Self {
        let mut seen: BTreeMap<String, BindingRow> = BTreeMap::new();
        for row in &self.rows {
            let key = row.key();
            match seen.get_mut(&key) {
                Some(existing) => existing.merge(row),
                None => {
                    seen.insert(key, row.clone());
                }
            }
        }
        let mut table = self.clone();
        table.rows = seen.into_values().collect();
        table
    }

    /// 反查：`FilterEq(?v, value) + Project(?e)`，不需要单独的反向规则。
    pub fn reverse_lookup(
        &self,
        value_variable: &str,
        value: &str,
        entity_variable: &str,
    ) -> Result<Self, TableError> {
        self.filter_eq(value_variable, value)?
            .project(&[entity_variable])
    }

    pub fn counts(&self, variable: &str) -> Result<BTreeMap<String, usize>, TableError> {
        self.validate()?;
        self.require_variable(variable)?;
        let mut counts = BTreeMap::new();
        for row in &self.rows {
            let key = row.value_text(variable).unwrap_or_else(|| "—".to_string());
            *counts.entry(key).or_insert(0) += 1;
        }
        Ok(counts)
    }

    pub fn truncate_rows(&mut self, limit: usize) -> bool {
        let original = self.rows.len();
        if original <= limit {
            return false;
        }
        self.rows.truncate(limit);
        if !self.coverage.is_point() {
            self.coverage = Coverage::partial(format!(
                "超过每表上限 {} 行，已截断（原 {} 行）",
                limit, original
            ));
        }
        true
    }

    pub fn render(&self) -> String {
        let mut lines = vec![format!(
            "绑定表 {}（来源：{}，{}，{} 行，知识版本 v{}）",
            if self.id.is_empty() {
                "未命名"
            } else {
                &self.id
            },
            self.source.label(),
            self.coverage.label(),
            self.rows.len(),
            self.revision
        )];
        if !self.question.trim().is_empty() {
            lines.push(format!("问题：{}", self.question.trim()));
        }
        lines.push(format!("形状：{}", self.shape.pinned_label()));
        lines.push(format!("覆盖：{}", self.coverage.caveat()));
        if let Some(reason) = &self.invalidated {
            lines.push(format!("失效：{}", reason));
        }
        lines.push(format!(
            "列：{}",
            self.shape
                .variables
                .iter()
                .map(|name| format!("?{}", name))
                .collect::<Vec<_>>()
                .join(" | ")
        ));
        let shown = self.rows.len().min(REPORT_ROW_CAP);
        for (index, row) in self.rows.iter().take(shown).enumerate() {
            let cells = self
                .shape
                .variables
                .iter()
                .map(|name| row.value_text(name).unwrap_or_else(|| "—".to_string()))
                .collect::<Vec<_>>()
                .join(" | ");
            let evidence = if row.evidence.is_empty() {
                "无证据".to_string()
            } else {
                row.evidence
                    .iter()
                    .map(FactRef::display)
                    .collect::<Vec<_>>()
                    .join("；")
            };
            lines.push(format!("{}. {} ［{}］", index + 1, cells, evidence));
        }
        if self.rows.len() > shown {
            lines.push(format!("…另有 {} 行未列出。", self.rows.len() - shown));
        }
        lines.join("\n")
    }

    fn require_variable(&self, variable: &str) -> Result<(), TableError> {
        if self.shape.variables.iter().any(|known| known == variable) {
            Ok(())
        } else {
            Err(TableError::UnknownVariable(variable.to_string()))
        }
    }
}

fn combine_coverage(left: &Coverage, right: &Coverage) -> Coverage {
    match (left, right) {
        (Coverage::Partial { reason: left }, Coverage::Partial { reason: right }) => {
            Coverage::partial(format!("{}；{}", left, right))
        }
        (Coverage::Partial { reason }, _) | (_, Coverage::Partial { reason }) => {
            Coverage::partial(reason.clone())
        }
        _ => Coverage::Complete,
    }
}

/// 排序方向。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SortDirection {
    Ascending,
    Descending,
}

impl SortDirection {
    fn apply(self, ordering: std::cmp::Ordering) -> std::cmp::Ordering {
        match self {
            SortDirection::Ascending => ordering,
            SortDirection::Descending => ordering.reverse(),
        }
    }
}

/// 失效报告：版本、显式失效与逐事实证据缺失可以同时出现。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Staleness {
    #[serde(default)]
    pub revision_changed: bool,
    #[serde(default)]
    pub invalidated: Option<String>,
    #[serde(default)]
    pub missing_evidence: Vec<FactRef>,
}

impl Staleness {
    pub fn is_stale(&self) -> bool {
        self.revision_changed || self.invalidated.is_some() || !self.missing_evidence.is_empty()
    }

    pub fn render(&self) -> String {
        if !self.is_stale() {
            return "新鲜：版本与证据均未变化。".to_string();
        }
        let mut reasons = Vec::new();
        if self.revision_changed {
            reasons.push("知识版本已变化".to_string());
        }
        if let Some(reason) = &self.invalidated {
            reasons.push(format!("显式失效：{}", reason));
        }
        if !self.missing_evidence.is_empty() {
            reasons.push(format!(
                "支撑事实缺失 {} 条：{}",
                self.missing_evidence.len(),
                self.missing_evidence
                    .iter()
                    .map(FactRef::display)
                    .collect::<Vec<_>>()
                    .join("；")
            ));
        }
        format!("已失效：{}。", reasons.join("；"))
    }
}

/// 由三元素事实构造通用 `?s ?p ?o` 表（检索端口/查询板的统一适配入口）。
pub fn from_fact_refs(
    facts: &[FactRef],
    coverage: Coverage,
    revision: u64,
    source: TableSource,
    question: &str,
) -> Result<BindingTable, TableError> {
    let shape = TableShape::triple(
        ShapeTerm::var("s"),
        ShapeTerm::var("p"),
        ShapeTerm::var("o"),
    );
    let mut table = BindingTable::new(shape, coverage, revision)
        .with_source(source)
        .with_question(question);
    for fact in facts {
        let row = BindingRow::from_pairs([
            ("s", BoundTerm::text(fact.subject.clone())),
            ("p", BoundTerm::text(fact.predicate.clone())),
            ("o", BoundTerm::classify(&fact.value)),
        ])
        .with_evidence([fact.clone()]);
        table.add_row(row)?;
    }
    Ok(table)
}

/// 表级错误：宁可显式拒绝，也不做不 sound 的操作。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TableError {
    InvalidVersion {
        found: u32,
        supported: u32,
    },
    EmptyShape,
    DuplicateVariable(String),
    UnknownVariable(String),
    UnknownPinned(String),
    UnboundPinned(String),
    PinnedMismatch {
        variable: String,
        expected: String,
        found: String,
    },
    PinnedConflict {
        variable: String,
        left: String,
        right: String,
    },
    EmptyProjection,
    PointJoin,
    RevisionMismatch {
        left: u64,
        right: u64,
    },
    StoreFull,
}

impl fmt::Display for TableError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TableError::InvalidVersion { found, supported } => {
                write!(f, "绑定表版本 {} 高于支持版本 {}", found, supported)
            }
            TableError::EmptyShape => write!(f, "形状没有变量"),
            TableError::DuplicateVariable(name) => write!(f, "变量重复：?{}", name),
            TableError::UnknownVariable(name) => write!(f, "未知变量：?{}", name),
            TableError::UnknownPinned(name) => write!(f, "固定输入引用了未知变量：?{}", name),
            TableError::UnboundPinned(name) => write!(f, "固定输入 ?{} 的值未绑定", name),
            TableError::PinnedMismatch {
                variable,
                expected,
                found,
            } => write!(
                f,
                "固定输入 ?{} 应为「{}」，行内却是「{}」",
                variable, expected, found
            ),
            TableError::PinnedConflict {
                variable,
                left,
                right,
            } => write!(
                f,
                "连接时固定输入 ?{} 冲突：「{}」对「{}」",
                variable, left, right
            ),
            TableError::EmptyProjection => write!(f, "投影不能保留零个变量"),
            TableError::PointJoin => {
                write!(f, "点表只对固定输入成立，不能参与自然连接")
            }
            TableError::RevisionMismatch { left, right } => {
                write!(f, "知识版本不一致（v{} vs v{}），请先重算", left, right)
            }
            TableError::StoreFull => write!(f, "绑定表台账已满且无法驱逐"),
        }
    }
}

impl std::error::Error for TableError {}

/// 插入回执：实际 id、被驱逐的表与截断标记。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoreReceipt {
    pub id: String,
    pub evicted: Vec<String>,
    pub truncated: bool,
}

/// 有界绑定表台账：默认 8 张、每张 256 行。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BindingTableStore {
    #[serde(default = "default_version")]
    pub version: u32,
    #[serde(default = "default_max_tables")]
    pub max_tables: usize,
    #[serde(default = "default_max_rows")]
    pub max_rows: usize,
    #[serde(default)]
    sequence: u64,
    #[serde(default)]
    tables: Vec<BindingTable>,
}

impl Default for BindingTableStore {
    fn default() -> Self {
        Self::new()
    }
}

impl BindingTableStore {
    pub fn new() -> Self {
        Self {
            version: BINDING_TABLE_VERSION,
            max_tables: DEFAULT_MAX_TABLES,
            max_rows: DEFAULT_MAX_ROWS,
            sequence: 0,
            tables: Vec::new(),
        }
    }

    pub fn with_limits(max_tables: usize, max_rows: usize) -> Self {
        let mut store = Self::new();
        store.max_tables = max_tables.max(1);
        store.max_rows = max_rows.max(1);
        store
    }

    pub fn len(&self) -> usize {
        self.tables.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tables.is_empty()
    }

    pub fn tables(&self) -> &[BindingTable] {
        &self.tables
    }

    pub fn get(&self, id: &str) -> Option<&BindingTable> {
        self.tables.iter().find(|table| table.id == id)
    }

    pub fn remove(&mut self, id: &str) -> Option<BindingTable> {
        let index = self.tables.iter().position(|table| table.id == id)?;
        Some(self.tables.remove(index))
    }

    pub fn clear(&mut self) {
        self.tables.clear();
    }

    pub fn insert(&mut self, mut table: BindingTable) -> Result<StoreReceipt, TableError> {
        table.validate()?;
        let mut truncated = false;
        if table.rows.len() > self.max_rows {
            truncated = table.truncate_rows(self.max_rows);
        }
        let base = if table.id.trim().is_empty() {
            self.sequence += 1;
            format!("bt-{}", self.sequence)
        } else {
            table.id.clone()
        };
        let mut id = base.clone();
        while self.tables.iter().any(|existing| existing.id == id) {
            self.sequence += 1;
            id = format!("{}-{}", base, self.sequence);
        }
        table.id = id.clone();
        self.tables.push(table);
        let mut evicted = Vec::new();
        while self.tables.len() > self.max_tables {
            let victim = self
                .tables
                .iter()
                .enumerate()
                .min_by_key(|(_, table)| (table.revision, table.created_turn, table.id.clone()))
                .map(|(index, _)| index);
            match victim {
                Some(index) => {
                    evicted.push(self.tables.remove(index).id);
                }
                None => break,
            }
        }
        Ok(StoreReceipt {
            id,
            evicted,
            truncated,
        })
    }

    /// 按结构键路由到候选表；固定输入的取值被忽略，复用形状后必须重新
    /// 执行。同版本候选排在最前，旧版本形状仍可用于理解结构，但其行不能
    /// 直接当作新答案（见 [`BindingTable::rows_reusable_at`]）。
    pub fn route(&self, shape: &TableShape, revision: u64) -> Vec<&BindingTable> {
        let key = shape.structural_key();
        let mut candidates: Vec<&BindingTable> = self
            .tables
            .iter()
            .filter(|table| table.invalidated.is_none() && table.shape.structural_key() == key)
            .collect();
        candidates.sort_by(|left, right| {
            let left_current = u8::from(left.revision == revision);
            let right_current = u8::from(right.revision == revision);
            right_current
                .cmp(&left_current)
                .then_with(|| right.coverage.rank().cmp(&left.coverage.rank()))
                .then_with(|| right.created_turn.cmp(&left.created_turn))
                .then_with(|| left.id.cmp(&right.id))
        });
        candidates
    }

    pub fn stale_reports(
        &self,
        revision: u64,
        facts: &BTreeSet<FactRef>,
    ) -> Vec<(String, Staleness)> {
        self.tables
            .iter()
            .filter_map(|table| {
                let staleness = table.staleness(revision, facts);
                if staleness.is_stale() {
                    Some((table.id.clone(), staleness))
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn render(&self) -> String {
        if self.tables.is_empty() {
            return "绑定表台账：（空）".to_string();
        }
        let mut lines = vec![format!(
            "绑定表台账 v{}：{} / {} 张，每张上限 {} 行",
            self.version,
            self.tables.len(),
            self.max_tables,
            self.max_rows
        )];
        for table in &self.tables {
            lines.push(format!(
                "  {}｜{}｜{} 行｜v{}｜{}",
                table.id,
                table.coverage.label(),
                table.rows.len(),
                table.revision,
                if table.question.trim().is_empty() {
                    "无问题记录"
                } else {
                    table.question.trim()
                }
            ));
        }
        lines.join("\n")
    }

    pub fn install_into(&self, image: &mut ContextImage) -> Result<(), String> {
        let value = serde_json::to_value(self).map_err(|error| error.to_string())?;
        image.set_section(BINDING_TABLES_SECTION, value);
        Ok(())
    }

    pub fn from_image(image: &ContextImage) -> Result<Self, String> {
        let Some(value) = image.section(BINDING_TABLES_SECTION).cloned() else {
            return Err("缺少 binding_tables 分节".to_string());
        };
        let store: BindingTableStore =
            serde_json::from_value(value).map_err(|error| format!("绑定表解码失败：{}", error))?;
        if store.version != BINDING_TABLE_VERSION {
            return Err(format!(
                "不支持的绑定表版本 {}（期望 {}）",
                store.version, BINDING_TABLE_VERSION
            ));
        }
        let mut seen = BTreeSet::new();
        for table in &store.tables {
            table.validate().map_err(|error| error.to_string())?;
            if !seen.insert(table.id.clone()) {
                return Err(format!("绑定表 id 重复：{}", table.id));
            }
        }
        Ok(store)
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let mut image = ContextImage::new();
        self.install_into(&mut image)
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::Other, error))?;
        image.seal();
        image.save_atomic(path)
    }

    pub fn load(path: &str) -> StoreLoad {
        if !std::path::Path::new(path).exists() {
            return StoreLoad::Missing;
        }
        let image = match ContextImage::load_verified(path) {
            Ok(image) => image,
            Err(error) => return StoreLoad::Damaged(error),
        };
        match Self::from_image(&image) {
            Ok(store) => StoreLoad::Loaded(store),
            Err(error) => StoreLoad::Damaged(error),
        }
    }
}

/// 台账读取结果：缺失与损坏都必须与正常空台账区分。
#[derive(Debug, Clone, PartialEq)]
pub enum StoreLoad {
    Missing,
    Loaded(BindingTableStore),
    Damaged(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn triple(subject: &str, predicate: &str, object: &str) -> FactRef {
        FactRef::new(subject, predicate, object)
    }

    fn height_shape() -> TableShape {
        TableShape::new(
            ["e", "v"],
            vec![ShapeAtom::new(
                ShapeTerm::var("e"),
                ShapeTerm::constant("height"),
                ShapeTerm::var("v"),
            )],
        )
    }

    fn height_table() -> BindingTable {
        let mut table = BindingTable::new(height_shape(), Coverage::Complete, 7)
            .with_question("列出所有演员的身高");
        table
            .add_row(
                BindingRow::from_pairs([
                    ("e", BoundTerm::text("张三")),
                    ("v", BoundTerm::classify("226 厘米")),
                ])
                .with_evidence([triple("张三", "height", "226 厘米")]),
            )
            .unwrap();
        table
            .add_row(
                BindingRow::from_pairs([
                    ("e", BoundTerm::text("李四")),
                    ("v", BoundTerm::classify("1.90 米")),
                ])
                .with_evidence([triple("李四", "height", "1.90 米")]),
            )
            .unwrap();
        table
    }

    fn team_tables() -> (BindingTable, BindingTable) {
        let mut actors = BindingTable::new(
            TableShape::new(
                ["e", "team"],
                vec![ShapeAtom::new(
                    ShapeTerm::var("e"),
                    ShapeTerm::constant("team"),
                    ShapeTerm::var("team"),
                )],
            ),
            Coverage::Complete,
            9,
        );
        actors
            .add_row(
                BindingRow::from_pairs([
                    ("e", BoundTerm::text("张三")),
                    ("team", BoundTerm::text("红队")),
                ])
                .with_evidence([triple("张三", "team", "红队")]),
            )
            .unwrap();
        actors
            .add_row(
                BindingRow::from_pairs([
                    ("e", BoundTerm::text("李四")),
                    ("team", BoundTerm::text("蓝队")),
                ])
                .with_evidence([triple("李四", "team", "蓝队")]),
            )
            .unwrap();
        let mut teams = BindingTable::new(
            TableShape::new(
                ["team", "city"],
                vec![ShapeAtom::new(
                    ShapeTerm::var("team"),
                    ShapeTerm::constant("city"),
                    ShapeTerm::var("city"),
                )],
            ),
            Coverage::Complete,
            9,
        );
        teams
            .add_row(
                BindingRow::from_pairs([
                    ("team", BoundTerm::text("红队")),
                    ("city", BoundTerm::text("北京")),
                ])
                .with_evidence([triple("红队", "city", "北京")]),
            )
            .unwrap();
        teams
            .add_row(
                BindingRow::from_pairs([
                    ("team", BoundTerm::text("蓝队")),
                    ("city", BoundTerm::text("上海")),
                ])
                .with_evidence([triple("蓝队", "city", "上海")]),
            )
            .unwrap();
        (actors, teams)
    }

    #[test]
    fn shape_key_includes_pinned_value_but_structural_key_does_not() {
        let zhangsan = height_shape().pin("e", BoundTerm::text("张三"));
        let lisi = height_shape().pin("e", BoundTerm::text("李四"));
        assert_ne!(zhangsan.key(), lisi.key());
        assert_eq!(zhangsan.structural_key(), lisi.structural_key());
        assert_eq!(
            zhangsan.structural_key(),
            height_shape().structural_key(),
            "有固定输入与无固定输入共享结构"
        );
    }

    #[test]
    fn filter_and_reverse_lookup_carry_evidence() {
        let table = height_table();
        let tall = table.filter_eq("v", "2.26米").unwrap();
        assert_eq!(tall.row_count(), 1);
        assert_eq!(tall.rows[0].value_text("e").as_deref(), Some("张三"));

        let reverse = table.reverse_lookup("v", "190cm", "e").unwrap();
        assert_eq!(reverse.row_count(), 1);
        assert_eq!(reverse.cell_names(), vec!["e".to_string()]);
        assert_eq!(
            reverse.rows[0].evidence,
            vec![triple("李四", "height", "1.90 米")]
        );
        assert!(reverse.coverage.is_complete());
    }

    #[test]
    fn point_tables_refuse_join_even_with_matching_anchors() {
        let point = BindingTable::new(
            height_shape().pin("e", BoundTerm::text("张三")),
            Coverage::Point,
            1,
        );
        let other = BindingTable::new(
            TableShape::new(
                ["e", "team"],
                vec![ShapeAtom::new(
                    ShapeTerm::var("e"),
                    ShapeTerm::constant("team"),
                    ShapeTerm::var("team"),
                )],
            )
            .pin("e", BoundTerm::text("张三")),
            Coverage::Complete,
            1,
        );
        assert_eq!(point.join(&other), Err(TableError::PointJoin));
    }

    #[test]
    fn natural_join_composes_tables_and_merges_evidence() {
        let (actors, teams) = team_tables();
        let joined = actors.join(&teams).unwrap();
        assert_eq!(joined.row_count(), 2);
        assert_eq!(
            joined.cell_names(),
            vec!["e".to_string(), "team".to_string(), "city".to_string()]
        );
        assert!(joined.coverage.is_complete());
        let zhangsan = joined.filter_eq("e", "张三").unwrap();
        assert_eq!(zhangsan.row_count(), 1);
        assert_eq!(zhangsan.rows[0].value_text("city").as_deref(), Some("北京"));
        assert_eq!(zhangsan.rows[0].evidence.len(), 2);
        assert!(joined.validate().is_ok());
    }

    #[test]
    fn join_rejects_revision_mismatch_and_pinned_conflict() {
        let (actors, mut teams) = team_tables();
        teams = BindingTable::new(teams.shape.clone(), teams.coverage.clone(), 10);
        assert!(matches!(
            actors.join(&teams),
            Err(TableError::RevisionMismatch { .. })
        ));

        let left = BindingTable::new(
            height_shape().pin("e", BoundTerm::text("张三")),
            Coverage::Complete,
            1,
        );
        let right = BindingTable::new(
            height_shape().pin("e", BoundTerm::text("李四")),
            Coverage::Complete,
            1,
        );
        assert!(matches!(
            left.join(&right),
            Err(TableError::PinnedConflict { .. })
        ));
    }

    #[test]
    fn numeric_sort_uses_units_and_puts_unbound_last_both_ways() {
        let mut table = height_table();
        table
            .add_row(
                BindingRow::from_pairs([("e", BoundTerm::text("王五"))]).with_evidence([triple(
                    "王五",
                    "height",
                    "未登记",
                )]),
            )
            .unwrap();
        let ascending = table.sort_by("v", SortDirection::Ascending).unwrap();
        let names: Vec<String> = ascending
            .rows
            .iter()
            .filter_map(|row| row.value_text("e"))
            .collect();
        assert_eq!(names, vec!["李四", "张三", "王五"]);

        let descending = table.sort_by("v", SortDirection::Descending).unwrap();
        let names: Vec<String> = descending
            .rows
            .iter()
            .filter_map(|row| row.value_text("e"))
            .collect();
        assert_eq!(names, vec!["张三", "李四", "王五"], "未绑定始终排最后");
    }

    #[test]
    fn top_k_marks_partial_only_when_it_actually_truncates() {
        let complete = height_table();
        assert!(complete.top_k(5).unwrap().coverage.is_complete());
        let truncated = complete.top_k(1).unwrap();
        assert_eq!(truncated.row_count(), 1);
        assert!(matches!(truncated.coverage, Coverage::Partial { .. }));
        assert!(truncated.coverage.caveat().contains("未证明完整"));

        let point = BindingTable::new(height_shape(), Coverage::Point, 1);
        let truncated = point.top_k(0).unwrap();
        assert!(truncated.coverage.is_point(), "点表截断后仍然只对点成立");
    }

    #[test]
    fn projection_dedupes_rows_and_keeps_evidence() {
        let table = height_table();
        let projected = table.project(&["e"]).unwrap();
        assert_eq!(projected.row_count(), 2);
        assert_eq!(projected.cell_names(), vec!["e".to_string()]);
        let mut duplicate = height_table();
        duplicate
            .add_row(
                BindingRow::from_pairs([
                    ("e", BoundTerm::text("张三")),
                    ("v", BoundTerm::classify("226 厘米")),
                ])
                .with_evidence([triple("张三", "height", "226 厘米")]),
            )
            .unwrap();
        let projected = duplicate.project(&["e"]).unwrap();
        assert_eq!(projected.row_count(), 2);
        assert_eq!(projected.filter_eq("e", "张三").unwrap().row_count(), 1);
    }

    #[test]
    fn staleness_reports_revision_missing_evidence_and_invalidation() {
        let table = height_table();
        let facts: BTreeSet<FactRef> = table.evidence_facts();
        assert!(!table.staleness(7, &facts).is_stale());
        assert!(table.staleness(8, &facts).revision_changed);
        let missing = table.staleness(7, &BTreeSet::new());
        assert_eq!(missing.missing_evidence.len(), 2);
        assert!(missing.render().contains("支撑事实缺失"));
        let mut invalid = table.clone();
        invalid.invalidate("用户撤回了导入批次");
        assert!(invalid.staleness(7, &facts).is_stale());
        assert!(invalid.staleness(7, &facts).render().contains("显式失效"));
    }

    #[test]
    fn store_enforces_row_and_table_limits_deterministically() {
        let mut store = BindingTableStore::with_limits(1, 1);
        let receipt = store.insert(height_table()).unwrap();
        assert!(receipt.truncated);
        let first_id = receipt.id;
        let stored = store.get(&first_id).unwrap();
        assert_eq!(stored.row_count(), 1);
        assert!(matches!(stored.coverage, Coverage::Partial { .. }));

        let mut second = height_table().with_id("second").with_turn(5);
        second.revision = 7;
        let receipt = store.insert(second).unwrap();
        assert_eq!(receipt.id, "second");
        assert_eq!(receipt.evicted, vec![first_id]);
        assert_eq!(store.len(), 1);
        assert_eq!(store.get("second").unwrap().created_turn, 5);
    }

    #[test]
    fn store_routes_by_structure_not_by_pinned_value() {
        let mut store = BindingTableStore::new();
        let point = BindingTable::new(
            height_shape().pin("e", BoundTerm::text("张三")),
            Coverage::Point,
            4,
        )
        .with_question("张三的身高")
        .with_turn(1);
        store.insert(point).unwrap();
        let complete = height_table();
        store.insert(complete).unwrap();

        let target = height_shape().pin("e", BoundTerm::text("李四"));
        let routes = store.route(&target, 7);
        assert_eq!(routes.len(), 2);
        assert_eq!(routes[0].coverage, Coverage::Complete, "全表优先");
        assert!(routes
            .iter()
            .all(|table| table.shape.structural_key() == target.structural_key()));
        assert!(routes[0].rows_reusable_at(7));
        assert!(!routes[0].rows_reusable_at(8));

        let stale_routes = store.route(&target, 8);
        assert_eq!(stale_routes.len(), 2, "旧版本仍可复用形状");
        assert!(stale_routes.iter().all(|table| table.revision != 8));
        assert!(stale_routes.iter().all(|table| !table.rows_reusable_at(8)));
    }

    #[test]
    fn store_roundtrip_through_context_image_and_tamper_detection() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("binding_tables.json");
        let path_text = path.to_str().unwrap();
        let mut store = BindingTableStore::new();
        store.insert(height_table()).unwrap();
        store.insert(height_table().with_id("second")).unwrap();
        store.save(path_text).unwrap();

        match BindingTableStore::load(path_text) {
            StoreLoad::Loaded(loaded) => assert_eq!(loaded, store),
            other => panic!("应加载成功：{:?}", other),
        }
        let mut text = std::fs::read_to_string(path_text).unwrap();
        text = text.replace("\"height\"", "\"tampered\"");
        std::fs::write(path_text, text).unwrap();
        assert!(matches!(
            BindingTableStore::load(path_text),
            StoreLoad::Damaged(_)
        ));
        assert!(matches!(
            BindingTableStore::load(dir.path().join("missing.json").to_str().unwrap()),
            StoreLoad::Missing
        ));
    }

    #[test]
    fn from_fact_refs_builds_evidence_backed_table() {
        let facts = vec![
            triple("张三", "team", "红队"),
            triple("李四", "team", "蓝队"),
        ];
        let table = from_fact_refs(
            &facts,
            Coverage::partial("检索端口 top-2"),
            3,
            TableSource::SearchPort,
            "球队",
        )
        .unwrap();
        assert_eq!(table.row_count(), 2);
        assert_eq!(
            table.cell_names(),
            vec!["s".to_string(), "p".to_string(), "o".to_string()]
        );
        assert_eq!(table.rows[0].value_text("s").as_deref(), Some("张三"));
        assert_eq!(table.rows[0].evidence, vec![facts[0].clone()]);
        assert!(table.coverage.caveat().contains("未证明完整"));
        assert!(table.render().contains("检索端口"));
    }

    #[test]
    fn pinned_mismatch_is_rejected_not_silently_rewritten() {
        let shape = height_shape().pin("e", BoundTerm::text("张三"));
        let mut table = BindingTable::new(shape, Coverage::Point, 1);
        let error = table
            .add_row(BindingRow::from_pairs([
                ("e", BoundTerm::text("李四")),
                ("v", BoundTerm::classify("190")),
            ]))
            .unwrap_err();
        assert!(matches!(error, TableError::PinnedMismatch { .. }));
        assert!(table.is_empty());
    }

    #[test]
    fn serde_roundtrip_is_stable_and_coverage_survives() {
        let table = height_table();
        let json = serde_json::to_string(&table).unwrap();
        let restored: BindingTable = serde_json::from_str(&json).unwrap();
        assert_eq!(table, restored);
        let point = BindingTable::new(height_shape(), Coverage::Point, 1);
        let json = serde_json::to_string(&point).unwrap();
        let restored: BindingTable = serde_json::from_str(&json).unwrap();
        assert!(restored.coverage.is_point());
        let domain = BindingTable::new(height_shape(), Coverage::domain("演员"), 1);
        let json = serde_json::to_string(&domain).unwrap();
        let restored: BindingTable = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.coverage, Coverage::domain("演员"));
    }
}

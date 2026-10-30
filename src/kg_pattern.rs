//! 知识图谱模式中间表示（KGP-IR）。
//!
//! 绑定帧、模式库与图问答此前各有一套「这句话在问什么」的表示：帧目标
//! 只有五个封闭枚举，词面模式只透出 `query_type`，图问答的
//! [`GraphPattern`](crate::conversational_graph::GraphPattern) 又只服务于
//! 专用编译器。本模块补上共享的语义层：
//!
//! - [`KgPattern`]：带变量、类型过滤器、投影变量与计数形状的正向合取模式；
//! - [`KgPatternRegistry`]：配置编译、校验、稳定语义指纹与重复模式体检；
//! - [`KgPattern::execute`]：把输入绑定落成 [`GraphPattern`]，复用图执行器，
//!   并把结果归一成「得证 / 无记录 / 未知 / 引擎故障」四态；
//! - 每条结果携带可回溯的事实证据，供答案计划与绑定帧措辞使用。
//!
//! v0.1 只覆盖正向合取、类型过滤与计数。变量谓词必须由运行时绑定接地，
//! 这让「两跳关系」可以由配置表达，而不是再写一个 Rust `match` 分支。

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::answer_plan::ClaimEvidence;
use crate::context_image::fnv1a_hex;
use crate::conversational_graph::{
    EdgePattern, GraphFact, GraphPattern, MatchOptions, PatternConstraint, PatternTerm,
};

/// KGP-IR 版本；语义或指纹算法不兼容变更时递增。
pub const KG_PATTERN_VERSION: &str = "kgp/0.2";

/// `[kg_patterns]` 配置：默认关闭；关闭时仍会编译校验，便于配置体检。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct KgPatternConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub patterns: Vec<KgPatternDecl>,
}

/// 一条图谱模式的声明。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KgPatternDecl {
    pub id: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub atoms: Vec<KgAtomDecl>,
    #[serde(default)]
    pub filters: Vec<KgFilterDecl>,
    #[serde(default)]
    pub select: Vec<String>,
    /// 只要计数，不展开结果行。
    #[serde(default)]
    pub count: bool,
}

/// 一条三元组模式；`?x` 表示变量，其余按常量处理。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KgAtomDecl {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

/// 过滤器声明。类型过滤器会编译成 `?var rdf:type Class` 的额外边；
/// `not_type` 则在执行后按事实集排除，而不是假装成肯定前提。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum KgFilterDecl {
    Equal { var: String, value: String },
    NotEqual { var: String, value: String },
    Type { var: String, of: String },
    NotType { var: String, of: String },
}

/// 编译期错误：稳定机器码 + 配置坐标 + 人话。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KgCompileError {
    pub code: &'static str,
    pub path: String,
    pub message: String,
}

impl fmt::Display for KgCompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}：{}", self.code, self.path, self.message)
    }
}

/// 语义体检项：只报告，不改变路由或执行结果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KgLint {
    pub code: &'static str,
    pub pattern_ids: Vec<String>,
    pub message: String,
}

/// IR 中的一项：变量或常量。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum KgTerm {
    Var(String),
    Const(String),
}

impl KgTerm {
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

    fn parse(raw: &str) -> Result<Self, String> {
        let raw = raw.trim();
        if raw.is_empty() {
            return Err("模式项不能为空".to_string());
        }
        if let Some(name) = raw.strip_prefix("var:") {
            validate_variable(name)?;
            return Ok(Self::Var(name.to_string()));
        }
        if let Some(value) = raw.strip_prefix("lit:") {
            if value.is_empty() {
                return Err("`lit:` 后不能为空".to_string());
            }
            return Ok(Self::Const(value.to_string()));
        }
        if let Some(name) = raw.strip_prefix('?') {
            validate_variable(name)?;
            return Ok(Self::Var(name.to_string()));
        }
        Ok(Self::Const(raw.to_string()))
    }

    fn render(&self) -> String {
        match self {
            Self::Var(name) => format!("?{}", name),
            Self::Const(value) => value.clone(),
        }
    }
}

fn validate_variable(name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("变量名不能为空".to_string());
    }
    if name
        .chars()
        .any(|ch| ch.is_whitespace() || matches!(ch, '?' | ':' | '{' | '}' | ','))
    {
        return Err(format!("变量名「{}」包含保留字符", name));
    }
    Ok(())
}

fn normalize_variable(raw: &str) -> Result<String, String> {
    let raw = raw.trim();
    let name = raw.strip_prefix('?').unwrap_or(raw);
    validate_variable(name)?;
    Ok(name.to_string())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct KgAtom {
    subject: KgTerm,
    predicate: KgTerm,
    object: KgTerm,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
enum KgFilter {
    Equal { var: String, value: String },
    NotEqual { var: String, value: String },
    Type { var: String, of: String },
    NotType { var: String, of: String },
}

impl KgFilter {
    fn compile(decl: &KgFilterDecl, variables: &BTreeSet<String>) -> Result<Self, String> {
        let (kind, var, value) = match decl {
            KgFilterDecl::Equal { var, value } => ("equal", var, value),
            KgFilterDecl::NotEqual { var, value } => ("not_equal", var, value),
            KgFilterDecl::Type { var, of } => ("type", var, of),
            KgFilterDecl::NotType { var, of } => ("not_type", var, of),
        };
        let var = normalize_variable(var)?;
        if !variables.contains(&var) {
            return Err(format!("{} 过滤器引用了未在原子中出现的变量「{}」", kind, var));
        }
        if value.trim().is_empty() {
            return Err(format!("{} 过滤器的取值不能为空", kind));
        }
        let value = value.trim().to_string();
        Ok(match decl {
            KgFilterDecl::Equal { .. } => Self::Equal { var, value },
            KgFilterDecl::NotEqual { .. } => Self::NotEqual { var, value },
            KgFilterDecl::Type { .. } => Self::Type { var, of: value },
            KgFilterDecl::NotType { .. } => Self::NotType { var, of: value },
        })
    }
}

/// 编译后的图谱模式：不可变、可复用、可计算稳定指纹。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KgPattern {
    id: String,
    description: String,
    atoms: Vec<KgAtom>,
    filters: Vec<KgFilter>,
    select: Vec<String>,
    count_only: bool,
}

/// 一次语义标号的结果：指纹、规范材料，以及是否完成精确枚举。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KgPatternIdentity {
    pub fingerprint: String,
    pub canonical: String,
    pub complete: bool,
}

/// 颜色类内最多枚举多少种重命名（8 个同类变量 = 40320）。
const MAX_CANONICAL_LABELINGS: u64 = 40_320;

impl KgPattern {
    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn description(&self) -> &str {
        &self.description
    }

    pub fn count_only(&self) -> bool {
        self.count_only
    }

    pub fn select(&self) -> &[String] {
        &self.select
    }

    /// 模式中出现的全部变量（含谓词变量），按字典序返回。
    pub fn variables(&self) -> BTreeSet<String> {
        let mut variables = BTreeSet::new();
        for atom in &self.atoms {
            for term in [&atom.subject, &atom.predicate, &atom.object] {
                if let Some(name) = term.var_name() {
                    variables.insert(name.to_string());
                }
            }
        }
        variables
    }

    /// 输入变量：除投影变量外的变量；它们可以由帧槽位或调用方接地。
    pub fn input_variables(&self) -> BTreeSet<String> {
        let mut variables = self.variables();
        for name in &self.select {
            variables.remove(name);
        }
        variables
    }

    pub fn is_selected(&self, variable: &str) -> bool {
        let variable = variable.strip_prefix('?').unwrap_or(variable);
        self.select.iter().any(|name| name == variable)
    }

    /// 语义指纹：在变量重命名与原子/过滤器声明顺序下保持不变。
    ///
    /// 多原子模式不能按「首次出现」编号：交换两条原子的声明顺序会改变
    /// 编号。这里把模式看成带标签的二部图（变量、常量、原子、过滤器），
    /// 先用颜色细化求稳定划分，再在颜色类内枚举重命名，取字典序最小的
    /// 规范材料。颜色类内组合数超过 [`MAX_CANONICAL_LABELINGS`] 时退化为
    /// 按名称排序的确定性标号；此时同键仍蕴含等价，但不同键只表示未知。
    pub fn semantic_fingerprint(&self) -> String {
        self.semantic_identity().fingerprint
    }

    /// 可审计的语义身份：指纹、规范材料与标号是否完整。
    pub fn semantic_identity(&self) -> KgPatternIdentity {
        let mut atoms = self.atoms.clone();
        atoms.sort_by_key(render_atom_raw);
        atoms.dedup_by_key(|atom| render_atom_raw(atom));

        let mut filters = self.filters.clone();
        filters.sort_by_key(render_filter_raw);
        filters.dedup_by_key(|filter| render_filter_raw(filter));

        let render = |mapping: &BTreeMap<String, String>| {
            render_pattern_material(&atoms, &filters, &self.select, self.count_only, mapping)
        };
        let (mapping, complete) = canonical_variable_mapping(&atoms, &filters, &self.select, render);
        let canonical = render_pattern_material(
            &atoms,
            &filters,
            &self.select,
            self.count_only,
            &mapping,
        );
        KgPatternIdentity {
            fingerprint: fnv1a_hex(canonical.as_bytes()),
            canonical,
            complete,
        }
    }

    fn compile(decl: &KgPatternDecl, path: &str) -> Result<Self, Vec<KgCompileError>> {
        let mut errors = Vec::new();
        let id = decl.id.trim().to_string();
        if id.is_empty() {
            errors.push(KgCompileError {
                code: "kg_pattern_id_empty",
                path: format!("{}.id", path),
                message: "模式 id 不能为空".to_string(),
            });
        }
        if decl.atoms.is_empty() {
            errors.push(KgCompileError {
                code: "kg_pattern_atoms_empty",
                path: format!("{}.atoms", path),
                message: format!("模式「{}」至少需要一条三元组原子", id),
            });
        }

        let mut atoms = Vec::new();
        for (index, atom) in decl.atoms.iter().enumerate() {
            let atom_path = format!("{}.atoms[{}]", path, index);
            let subject = match KgTerm::parse(&atom.subject) {
                Ok(term) => term,
                Err(message) => {
                    errors.push(KgCompileError {
                        code: "kg_pattern_term_invalid",
                        path: format!("{}.subject", atom_path),
                        message,
                    });
                    continue;
                }
            };
            let predicate = match KgTerm::parse(&atom.predicate) {
                Ok(KgTerm::Const(value)) if value.is_empty() => {
                    errors.push(KgCompileError {
                        code: "kg_pattern_predicate_empty",
                        path: format!("{}.predicate", atom_path),
                        message: "谓词不能为空".to_string(),
                    });
                    continue;
                }
                Ok(term) => term,
                Err(message) => {
                    errors.push(KgCompileError {
                        code: "kg_pattern_term_invalid",
                        path: format!("{}.predicate", atom_path),
                        message,
                    });
                    continue;
                }
            };
            let object = match KgTerm::parse(&atom.object) {
                Ok(term) => term,
                Err(message) => {
                    errors.push(KgCompileError {
                        code: "kg_pattern_term_invalid",
                        path: format!("{}.object", atom_path),
                        message,
                    });
                    continue;
                }
            };
            atoms.push(KgAtom {
                subject,
                predicate,
                object,
            });
        }

        let mut variables = BTreeSet::new();
        for atom in &atoms {
            for term in [&atom.subject, &atom.predicate, &atom.object] {
                if let Some(name) = term.var_name() {
                    variables.insert(name.to_string());
                }
            }
        }

        let mut select = Vec::new();
        let mut seen_select = BTreeSet::new();
        for (index, raw) in decl.select.iter().enumerate() {
            match normalize_variable(raw) {
                Ok(name) => {
                    if !variables.contains(&name) {
                        errors.push(KgCompileError {
                            code: "kg_pattern_select_unbound",
                            path: format!("{}.select[{}]", path, index),
                            message: format!("投影变量「{}」没有出现在任何原子中", name),
                        });
                    } else if !seen_select.insert(name.clone()) {
                        errors.push(KgCompileError {
                            code: "kg_pattern_select_duplicate",
                            path: format!("{}.select[{}]", path, index),
                            message: format!("投影变量「{}」重复出现", name),
                        });
                    } else {
                        select.push(name);
                    }
                }
                Err(message) => errors.push(KgCompileError {
                    code: "kg_pattern_term_invalid",
                    path: format!("{}.select[{}]", path, index),
                    message,
                }),
            }
        }
        if select.is_empty() && !decl.count {
            errors.push(KgCompileError {
                code: "kg_pattern_shape_empty",
                path: format!("{}.select", path),
                message: format!("模式「{}」既没有投影变量也不计数，答案形状为空", id),
            });
        }

        let mut filters = Vec::new();
        for (index, filter) in decl.filters.iter().enumerate() {
            match KgFilter::compile(filter, &variables) {
                Ok(filter) => filters.push(filter),
                Err(message) => errors.push(KgCompileError {
                    code: "kg_pattern_filter_invalid",
                    path: format!("{}.filters[{}]", path, index),
                    message,
                }),
            }
        }
        validate_filter_consistency(&filters, path, &mut errors);

        if !errors.is_empty() {
            return Err(errors);
        }

        Ok(Self {
            id,
            description: decl.description.trim().to_string(),
            atoms,
            filters,
            select,
            count_only: decl.count,
        })
    }

    /// 用输入绑定接地模式，并编译成图执行层的 [`GraphPattern`]。
    fn ground(&self, bindings: &BTreeMap<String, String>) -> Result<GroundedKg, KgGroundError> {
        let variables = self.variables();
        let mut input = BTreeMap::new();
        for (raw_name, value) in bindings {
            let name = raw_name.trim().trim_start_matches('?');
            if !variables.contains(name) {
                return Err(KgGroundError::UnknownBinding(name.to_string()));
            }
            input.insert(name.to_string(), value.clone());
        }

        if let Some(()) = contradiction(&self.filters, &input) {
            return Ok(GroundedKg::contradiction());
        }

        let mut edges = Vec::new();
        for atom in &self.atoms {
            let subject = ground_term(&atom.subject, &input);
            let predicate = match &atom.predicate {
                KgTerm::Const(value) => value.clone(),
                KgTerm::Var(name) => input
                    .get(name)
                    .cloned()
                    .ok_or_else(|| KgGroundError::UnboundPredicate(name.clone()))?,
            };
            if predicate.trim().is_empty() {
                return Err(KgGroundError::EmptyPredicate);
            }
            let object = ground_term(&atom.object, &input);
            edges.push(EdgePattern::new(subject, predicate, object));
        }

        for filter in &self.filters {
            if let KgFilter::Type { var, of } = filter {
                let subject = match input.get(var) {
                    Some(value) => PatternTerm::constant(value.clone()),
                    None => PatternTerm::var(var.clone()),
                };
                edges.push(EdgePattern::new(
                    subject,
                    "type",
                    PatternTerm::constant(of.clone()),
                ));
            }
        }

        let mut constraints = Vec::new();
        for filter in &self.filters {
            match filter {
                KgFilter::Equal { var, value } if !input.contains_key(var) => {
                    constraints.push(PatternConstraint::Equal {
                        variable: var.clone(),
                        value: value.clone(),
                    });
                }
                KgFilter::NotEqual { var, value } if !input.contains_key(var) => {
                    constraints.push(PatternConstraint::NotEqual {
                        variable: var.clone(),
                        value: value.clone(),
                    });
                }
                _ => {}
            }
        }

        let graph_select = self
            .select
            .iter()
            .filter(|name| !input.contains_key(*name))
            .cloned()
            .collect::<Vec<_>>();
        let graph = GraphPattern::new(edges, graph_select).with_constraints(constraints);
        graph
            .validate()
            .map_err(|error| KgGroundError::InvalidPattern(error.to_string()))?;
        let post_filters = self
            .filters
            .iter()
            .filter(|filter| match filter {
                KgFilter::NotEqual { var, .. } => !input.contains_key(var),
                KgFilter::NotType { .. } => true,
                _ => false,
            })
            .cloned()
            .collect();
        Ok(GroundedKg {
            graph,
            input,
            post_filters,
            contradiction: false,
        })
    }

    /// 执行模式。`bindings` 是输入变量或投影变量的显式接地；
    /// `closed_world=true` 时零解才敢报告「无记录」，否则是「未知」。
    pub fn execute(
        &self,
        facts: &[GraphFact],
        bindings: &BTreeMap<String, String>,
        closed_world: bool,
        options: MatchOptions,
    ) -> KgResult {
        let grounded = match self.ground(bindings) {
            Ok(grounded) => grounded,
            Err(error) => {
                return KgResult::fault(self, format!("模式接地失败：{}", error));
            }
        };
        if grounded.contradiction {
            return KgResult {
                pattern_id: self.id.clone(),
                status: if closed_world {
                    KgStatus::NoRecords
                } else {
                    KgStatus::Unknown("输入约束互相矛盾".to_string())
                },
                select: self.select.clone(),
                rows: Vec::new(),
                total: 0,
                truncated: false,
                expansions: 0,
            };
        }

        let outcome = match grounded.graph.evaluate(facts, options) {
            Ok(outcome) => outcome,
            Err(error) => return KgResult::fault(self, format!("图执行失败：{}", error)),
        };
        let type_facts: BTreeSet<(&str, &str)> = facts
            .iter()
            .filter(|fact| fact.predicate == "type")
            .map(|fact| (fact.subject.as_str(), fact.object.as_str()))
            .collect();

        let mut rows = Vec::new();
        for solution in &outcome.solutions {
            let mut full = grounded.input.clone();
            full.extend(solution.binding.clone());
            if grounded
                .post_filters
                .iter()
                .any(|filter| !post_filter_accepts(filter, &full, &type_facts))
            {
                continue;
            }
            let evidence = solution
                .witness
                .iter()
                .filter_map(|index| facts.get(*index))
                .map(|fact| ClaimEvidence::new(&fact.subject, &fact.predicate, &fact.object, false))
                .collect();
            rows.push(KgRow {
                bindings: full,
                evidence,
            });
        }
        let total = rows.len();
        let status = if total > 0 {
            KgStatus::Proven
        } else if closed_world {
            KgStatus::NoRecords
        } else {
            KgStatus::Unknown("事实源为开放世界，未找到不等于不存在".to_string())
        };
        KgResult {
            pattern_id: self.id.clone(),
            status,
            select: self.select.clone(),
            rows,
            total,
            truncated: outcome.truncated,
            expansions: outcome.expansions,
        }
    }
}

fn ground_term(term: &KgTerm, input: &BTreeMap<String, String>) -> PatternTerm {
    match term {
        KgTerm::Const(value) => PatternTerm::constant(value.clone()),
        KgTerm::Var(name) => match input.get(name) {
            Some(value) => PatternTerm::constant(value.clone()),
            None => PatternTerm::var(name.clone()),
        },
    }
}

fn contradiction(filters: &[KgFilter], input: &BTreeMap<String, String>) -> Option<()> {
    for filter in filters {
        match filter {
            KgFilter::Equal { var, value } => {
                if let Some(bound) = input.get(var) {
                    if bound != value {
                        return Some(());
                    }
                }
            }
            KgFilter::NotEqual { var, value } => {
                if input.get(var).map(String::as_str) == Some(value.as_str()) {
                    return Some(());
                }
            }
            _ => {}
        }
    }
    None
}

fn post_filter_accepts(
    filter: &KgFilter,
    binding: &BTreeMap<String, String>,
    type_facts: &BTreeSet<(&str, &str)>,
) -> bool {
    match filter {
        KgFilter::NotEqual { var, value } => binding.get(var).map(String::as_str) != Some(value),
        KgFilter::NotType { var, of } => match binding.get(var) {
            Some(value) => !type_facts.contains(&(value.as_str(), of.as_str())),
            None => true,
        },
        _ => true,
    }
}

fn validate_filter_consistency(
    filters: &[KgFilter],
    path: &str,
    errors: &mut Vec<KgCompileError>,
) {
    let mut equal: BTreeMap<&str, &str> = BTreeMap::new();
    let mut not_equal: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    let mut positive_types: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    let mut negative_types: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
    for filter in filters {
        match filter {
            KgFilter::Equal { var, value } => {
                if let Some(previous) = equal.insert(var, value) {
                    if previous != value {
                        errors.push(KgCompileError {
                            code: "kg_pattern_filter_conflict",
                            path: format!("{}.filters", path),
                            message: format!(
                                "变量「{}」同时被要求等于「{}」和「{}」",
                                var, previous, value
                            ),
                        });
                    }
                }
            }
            KgFilter::NotEqual { var, value } => {
                not_equal.entry(var).or_default().insert(value);
            }
            KgFilter::Type { var, of } => {
                positive_types.entry(var).or_default().insert(of);
            }
            KgFilter::NotType { var, of } => {
                negative_types.entry(var).or_default().insert(of);
            }
        }
    }
    for (var, value) in &equal {
        if not_equal
            .get(var)
            .map(|values| values.contains(value))
            .unwrap_or(false)
        {
            errors.push(KgCompileError {
                code: "kg_pattern_filter_conflict",
                path: format!("{}.filters", path),
                message: format!("变量「{}」同时被要求等于与不等于「{}」", var, value),
            });
        }
    }
    for (var, negatives) in &negative_types {
        if let Some(positives) = positive_types.get(var) {
            for negative in negatives {
                if positives.contains(negative) {
                    errors.push(KgCompileError {
                        code: "kg_pattern_type_conflict",
                        path: format!("{}.filters", path),
                        message: format!(
                            "变量「{}」同时被要求是「{}」与不是「{}」",
                            var, negative, negative
                        ),
                    });
                        }
            }
        }
    }
}

fn render_atom_raw(atom: &KgAtom) -> String {
    format!(
        "a({},{},{})",
        atom.subject.render(),
        atom.predicate.render(),
        atom.object.render()
    )
}

fn render_atom_canonical(atom: &KgAtom, mapping: &BTreeMap<String, String>) -> String {
    format!(
        "a({},{},{})",
        render_term_canonical(&atom.subject, mapping),
        render_term_canonical(&atom.predicate, mapping),
        render_term_canonical(&atom.object, mapping)
    )
}

fn render_term_canonical(term: &KgTerm, mapping: &BTreeMap<String, String>) -> String {
    match term {
        KgTerm::Const(value) => format!("lit:{}", value),
        KgTerm::Var(name) => mapping
            .get(name)
            .cloned()
            .unwrap_or_else(|| format!("?{}", name)),
    }
}

fn render_filter_raw(filter: &KgFilter) -> String {
    match filter {
        KgFilter::Equal { var, value } => format!("equal({},{})", var, value),
        KgFilter::NotEqual { var, value } => format!("not_equal({},{})", var, value),
        KgFilter::Type { var, of } => format!("type({},{})", var, of),
        KgFilter::NotType { var, of } => format!("not_type({},{})", var, of),
    }
}

fn render_filter_canonical(filter: &KgFilter, mapping: &BTreeMap<String, String>) -> String {
    let canonical = |name: &str| {
        mapping
            .get(name)
            .cloned()
            .unwrap_or_else(|| format!("?{}", name))
    };
    match filter {
        KgFilter::Equal { var, value } => format!("equal({},{})", canonical(var), value),
        KgFilter::NotEqual { var, value } => format!("not_equal({},{})", canonical(var), value),
        KgFilter::Type { var, of } => format!("type({},{})", canonical(var), of),
        KgFilter::NotType { var, of } => format!("not_type({},{})", canonical(var), of),
    }
}

fn render_pattern_material(
    atoms: &[KgAtom],
    filters: &[KgFilter],
    select: &[String],
    count_only: bool,
    mapping: &BTreeMap<String, String>,
) -> String {
    let mut atom_lines: Vec<String> = atoms
        .iter()
        .map(|atom| render_atom_canonical(atom, mapping))
        .collect();
    atom_lines.sort();
    atom_lines.dedup();

    let mut filter_lines: Vec<String> = filters
        .iter()
        .map(|filter| render_filter_canonical(filter, mapping))
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

    format!(
        "{}|atoms:{}|filters:{}|select:{}|count:{}",
        KG_PATTERN_VERSION,
        atom_lines.join(";"),
        filter_lines.join(";"),
        select_lines.join(","),
        u8::from(count_only)
    )
}

fn build_label_graph(
    atoms: &[KgAtom],
    filters: &[KgFilter],
    select: &[String],
) -> (Vec<String>, Vec<(usize, &'static str, usize)>, Vec<(String, usize)>) {
    let mut names: BTreeSet<String> = BTreeSet::new();
    for atom in atoms {
        for term in [&atom.subject, &atom.predicate, &atom.object] {
            if let Some(name) = term.var_name() {
                names.insert(name.to_string());
            }
        }
    }
    for filter in filters {
        names.insert(filter_var(filter).to_string());
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
        for (label, term) in [
            ("s", &atom.subject),
            ("p", &atom.predicate),
            ("o", &atom.object),
        ] {
            let target = term_node(term, &var_index, &mut constants, &mut labels);
            edges.push((node, label, target));
        }
    }
    for filter in filters {
        let node = labels.len();
        labels.push(filter_label(filter));
        let target = var_index[filter_var(filter)];
        edges.push((node, "f", target));
    }
    (labels, edges, variables)
}

fn term_node(
    term: &KgTerm,
    var_index: &BTreeMap<String, usize>,
    constants: &mut BTreeMap<String, usize>,
    labels: &mut Vec<String>,
) -> usize {
    match term {
        KgTerm::Var(name) => var_index[name.as_str()],
        KgTerm::Const(value) => match constants.get(value) {
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

fn filter_var(filter: &KgFilter) -> &str {
    match filter {
        KgFilter::Equal { var, .. }
        | KgFilter::NotEqual { var, .. }
        | KgFilter::Type { var, .. }
        | KgFilter::NotType { var, .. } => var,
    }
}

fn filter_label(filter: &KgFilter) -> String {
    match filter {
        KgFilter::Equal { value, .. } => format!("filter:equal:{}:{}", value.len(), value),
        KgFilter::NotEqual { value, .. } => {
            format!("filter:not_equal:{}:{}", value.len(), value)
        }
        KgFilter::Type { of, .. } => format!("filter:type:{}:{}", of.len(), of),
        KgFilter::NotType { of, .. } => format!("filter:not_type:{}:{}", of.len(), of),
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

fn canonical_variable_mapping<F>(
    atoms: &[KgAtom],
    filters: &[KgFilter],
    select: &[String],
    mut render: F,
) -> (BTreeMap<String, String>, bool)
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
            Some((_, mapping)) => (mapping, true),
            None => (BTreeMap::new(), true),
        }
    } else {
        let choice: Vec<Vec<String>> = permutations
            .iter()
            .map(|choices| choices.first().cloned().unwrap_or_default())
            .collect();
        (mapping_from_choice(&choice), false)
    }
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct GroundedKg {
    graph: GraphPattern,
    input: BTreeMap<String, String>,
    post_filters: Vec<KgFilter>,
    contradiction: bool,
}

impl GroundedKg {
    fn contradiction() -> Self {
        Self {
            graph: GraphPattern::default(),
            input: BTreeMap::new(),
            post_filters: Vec::new(),
            contradiction: true,
        }
    }
}

/// 模式接地失败：变量谓词没有绑定、显式绑定了未知变量，或 IR 不自洽。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KgGroundError {
    UnboundPredicate(String),
    UnknownBinding(String),
    EmptyPredicate,
    InvalidPattern(String),
}

impl fmt::Display for KgGroundError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnboundPredicate(name) => {
                write!(f, "谓词变量「?{}」没有输入绑定", name)
            }
            Self::UnknownBinding(name) => write!(f, "输入绑定引用了未知变量「{}」", name),
            Self::EmptyPredicate => write!(f, "谓词接地后为空"),
            Self::InvalidPattern(message) => write!(f, "图模式不合法：{}", message),
        }
    }
}

/// 四态执行结果。零解不自动等于「不存在」。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum KgStatus {
    Proven,
    NoRecords,
    Unknown(String),
    Fault(String),
}

impl KgStatus {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Proven => "得证",
            Self::NoRecords => "无记录",
            Self::Unknown(_) => "未知",
            Self::Fault(_) => "引擎故障",
        }
    }

    pub fn is_proven(&self) -> bool {
        matches!(self, Self::Proven)
    }

    pub fn detail(&self) -> Option<&str> {
        match self {
            Self::Unknown(detail) | Self::Fault(detail) => Some(detail),
            _ => None,
        }
    }
}

/// 一行结果：变量绑定 + 支撑事实。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KgRow {
    pub bindings: BTreeMap<String, String>,
    pub evidence: Vec<ClaimEvidence>,
}

impl KgRow {
    pub fn value(&self, variable: &str) -> Option<&str> {
        self.bindings
            .get(variable.strip_prefix('?').unwrap_or(variable))
            .map(String::as_str)
    }

    pub fn selected_values(&self, select: &[String]) -> Vec<String> {
        select
            .iter()
            .filter_map(|name| self.bindings.get(name).cloned())
            .collect()
    }

    pub fn render_selected(&self, select: &[String]) -> String {
        self.selected_values(select).join("、")
    }
}

/// 一次模式执行的完整结果。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KgResult {
    pub pattern_id: String,
    pub status: KgStatus,
    pub select: Vec<String>,
    pub rows: Vec<KgRow>,
    pub total: usize,
    pub truncated: bool,
    pub expansions: usize,
}

impl KgResult {
    fn fault(pattern: &KgPattern, detail: String) -> Self {
        Self {
            pattern_id: pattern.id.clone(),
            status: KgStatus::Fault(detail),
            select: pattern.select.clone(),
            rows: Vec::new(),
            total: 0,
            truncated: false,
            expansions: 0,
        }
    }

    pub fn is_proven(&self) -> bool {
        self.status.is_proven()
    }

    /// 投影结果：每行按 `select` 拼接，去重后稳定排序。
    pub fn answer_values(&self) -> Vec<String> {
        let values: BTreeSet<String> = self
            .rows
            .iter()
            .map(|row| row.render_selected(&self.select))
            .filter(|value| !value.is_empty())
            .collect();
        values.into_iter().collect()
    }

    pub fn first_value(&self, variable: &str) -> Option<&str> {
        self.rows.iter().find_map(|row| row.value(variable))
    }

    pub fn evidence_lines(&self) -> Vec<String> {
        let mut lines: BTreeSet<String> = BTreeSet::new();
        for row in &self.rows {
            for evidence in &row.evidence {
                lines.insert(evidence.render());
            }
        }
        lines.into_iter().collect()
    }

    pub fn summary(&self) -> String {
        match &self.status {
            KgStatus::Proven => format!(
                "模式「{}」得证：{} 个结果{}。",
                self.pattern_id,
                self.total,
                if self.truncated { "（截断）" } else { "" }
            ),
            KgStatus::NoRecords => format!("模式「{}」在封闭事实集中无记录。", self.pattern_id),
            KgStatus::Unknown(detail) => {
                format!("模式「{}」结果未知：{}。", self.pattern_id, detail)
            }
            KgStatus::Fault(detail) => {
                format!("模式「{}」执行故障：{}。", self.pattern_id, detail)
            }
        }
    }
}

/// 配置编译后的模式库：按配置序保存，按 id 查找。
#[derive(Debug, Clone, Default)]
pub struct KgPatternRegistry {
    enabled: bool,
    patterns: Vec<KgPattern>,
    by_id: BTreeMap<String, usize>,
}

impl KgPatternRegistry {
    pub fn compile(config: &KgPatternConfig) -> Result<Self, Vec<KgCompileError>> {
        let mut errors = Vec::new();
        let mut patterns = Vec::new();
        let mut by_id = BTreeMap::new();
        for (index, decl) in config.patterns.iter().enumerate() {
            let path = format!("kg_patterns[{}]", index);
            match KgPattern::compile(decl, &path) {
                Ok(pattern) => {
                    if let Some(previous) = by_id.get(pattern.id()) {
                        errors.push(KgCompileError {
                            code: "kg_pattern_id_duplicate",
                            path: format!("{}.id", path),
                            message: format!(
                                "模式 id「{}」与第 {} 条重复",
                                pattern.id(),
                                *previous + 1
                            ),
                        });
                        continue;
                    }
                    by_id.insert(pattern.id().to_string(), patterns.len());
                    patterns.push(pattern);
                }
                Err(mut found) => errors.append(&mut found),
            }
        }
        if errors.is_empty() {
            Ok(Self {
                enabled: config.enabled,
                patterns,
                by_id,
            })
        } else {
            Err(errors)
        }
    }

    pub fn from_declarations(patterns: &[KgPatternDecl]) -> Result<Self, Vec<KgCompileError>> {
        Self::compile(&KgPatternConfig {
            enabled: true,
            patterns: patterns.to_vec(),
        })
    }

    pub fn empty() -> Self {
        Self::default()
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }

    pub fn len(&self) -> usize {
        self.patterns.len()
    }

    pub fn get(&self, id: &str) -> Option<&KgPattern> {
        self.by_id.get(id).and_then(|index| self.patterns.get(*index))
    }

    pub fn ids(&self) -> Vec<&str> {
        self.patterns.iter().map(KgPattern::id).collect()
    }

    pub fn patterns(&self) -> &[KgPattern] {
        &self.patterns
    }

    /// 语义体检：重复模式、重复原子。只报告，不改变 registry。
    pub fn lint(&self) -> Vec<KgLint> {
        let mut lints = Vec::new();
        let mut fingerprints: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for pattern in &self.patterns {
            fingerprints
                .entry(pattern.semantic_fingerprint())
                .or_default()
                .push(pattern.id().to_string());
            let mut seen = BTreeSet::new();
            for atom in &pattern.atoms {
                let rendered = render_atom_raw(atom);
                if !seen.insert(rendered) {
                    lints.push(KgLint {
                        code: "kg_pattern_atom_duplicate",
                        pattern_ids: vec![pattern.id().to_string()],
                        message: format!("模式「{}」含重复原子，可删除而不改变语义", pattern.id()),
                    });
                }
            }
        }
        for (fingerprint, ids) in fingerprints {
            if ids.len() > 1 {
                lints.push(KgLint {
                    code: "kg_pattern_semantic_duplicate",
                    pattern_ids: ids.clone(),
                    message: format!(
                        "模式 {} 语义指纹相同（{}），可合并为一条配置",
                        ids.join("、"),
                        fingerprint
                    ),
                });
            }
        }
        lints
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts() -> Vec<GraphFact> {
        vec![
            GraphFact::type_assertion("北京", "City"),
            GraphFact::attribute("北京", "capital", "中国"),
            GraphFact::attribute("巴黎", "capital", "法国"),
            GraphFact::relation("张三", "师傅", "李四"),
            GraphFact::relation("李四", "师傅", "王五"),
            GraphFact::type_assertion("李四", "Person"),
        ]
    }

    fn config(source: &str) -> KgPatternConfig {
        toml::from_str(source).expect("测试配置应可解析")
    }

    fn one_hop() -> KgPattern {
        let registry = KgPatternRegistry::compile(&config(
            r#"
enabled = true

[[patterns]]
id = "capital"
atoms = [{ subject = "?place", predicate = "capital", object = "?capital" }]
select = ["capital"]
"#,
        ))
        .expect("应可编译");
        registry.get("capital").unwrap().clone()
    }

    #[test]
    fn one_hop_returns_values_and_evidence() {
        let result = one_hop().execute(
            &facts(),
            &BTreeMap::from([("place".to_string(), "北京".to_string())]),
            true,
            MatchOptions::default(),
        );
        assert_eq!(result.status, KgStatus::Proven);
        assert_eq!(result.answer_values(), vec!["中国"]);
        assert!(result
            .evidence_lines()
            .iter()
            .any(|line| line.contains("北京") && line.contains("capital")));
    }

    #[test]
    fn two_hop_uses_bound_predicates() {
        let registry = KgPatternRegistry::compile(&config(
            r#"
enabled = true

[[patterns]]
id = "two_hop"
atoms = [
  { subject = "?entity", predicate = "?first", object = "?middle" },
  { subject = "?middle", predicate = "?second", object = "?target" },
]
select = ["target"]
"#,
        ))
        .expect("应可编译");
        let pattern = registry.get("two_hop").unwrap();
        let result = pattern.execute(
            &facts(),
            &BTreeMap::from([
                ("entity".to_string(), "张三".to_string()),
                ("first".to_string(), "师傅".to_string()),
                ("second".to_string(), "师傅".to_string()),
            ]),
            true,
            MatchOptions::default(),
        );
        assert_eq!(result.status, KgStatus::Proven);
        assert_eq!(result.answer_values(), vec!["王五"]);
        assert_eq!(result.evidence_lines().len(), 2);
    }

    #[test]
    fn unbound_predicate_is_a_fault_not_an_answer() {
        let registry = KgPatternRegistry::compile(&config(
            r#"
enabled = true

[[patterns]]
id = "two_hop"
atoms = [
  { subject = "?entity", predicate = "?first", object = "?middle" },
  { subject = "?middle", predicate = "?second", object = "?target" },
]
select = ["target"]
"#,
        ))
        .expect("应可编译");
        let result = registry.get("two_hop").unwrap().execute(
            &facts(),
            &BTreeMap::from([("entity".to_string(), "张三".to_string())]),
            true,
            MatchOptions::default(),
        );
        assert!(matches!(result.status, KgStatus::Fault(_)));
    }

    #[test]
    fn zero_rows_distinguishes_closed_and_open_world() {
        let bindings = BTreeMap::from([("place".to_string(), "东京".to_string())]);
        let open = one_hop().execute(&facts(), &bindings, false, MatchOptions::default());
        assert!(matches!(open.status, KgStatus::Unknown(_)));
        let closed = one_hop().execute(&facts(), &bindings, true, MatchOptions::default());
        assert_eq!(closed.status, KgStatus::NoRecords);
    }

    #[test]
    fn type_and_not_type_filters_use_type_facts() {
        let registry = KgPatternRegistry::compile(&config(
            r#"
enabled = true

[[patterns]]
id = "person_masters"
atoms = [{ subject = "?person", predicate = "师傅", object = "?master" }]
filters = [{ kind = "type", var = "?person", of = "Person" }]
select = ["person"]

[[patterns]]
id = "not_person"
atoms = [{ subject = "?person", predicate = "师傅", object = "?master" }]
filters = [{ kind = "not_type", var = "?person", of = "Person" }]
select = ["person"]
"#,
        ))
        .expect("应可编译");
        let person = registry
            .get("person_masters")
            .unwrap()
            .execute(&facts(), &BTreeMap::new(), true, MatchOptions::default());
        assert_eq!(person.answer_values(), vec!["李四"]);
        let not_person = registry
            .get("not_person")
            .unwrap()
            .execute(&facts(), &BTreeMap::new(), true, MatchOptions::default());
        assert_eq!(not_person.answer_values(), vec!["张三"]);
    }

    #[test]
    fn equal_filter_contradiction_is_no_records_not_empty_answer() {
        let registry = KgPatternRegistry::compile(&config(
            r#"
enabled = true

[[patterns]]
id = "capital_const"
atoms = [{ subject = "?place", predicate = "capital", object = "?capital" }]
filters = [{ kind = "equal", var = "?place", value = "北京" }]
select = ["capital"]
"#,
        ))
        .expect("应可编译");
        let result = registry.get("capital_const").unwrap().execute(
            &facts(),
            &BTreeMap::from([("place".to_string(), "巴黎".to_string())]),
            true,
            MatchOptions::default(),
        );
        assert_eq!(result.status, KgStatus::NoRecords);
    }

    #[test]
    fn semantic_fingerprint_is_alpha_invariant_and_lint_reports_duplicates() {
        let config = config(
            r#"
enabled = true

[[patterns]]
id = "capital_a"
atoms = [{ subject = "?place", predicate = "capital", object = "?value" }]
select = ["value"]

[[patterns]]
id = "capital_b"
atoms = [{ subject = "?city", predicate = "capital", object = "?result" }]
select = ["result"]

[[patterns]]
id = "capital_other"
atoms = [{ subject = "?place", predicate = "author", object = "?value" }]
select = ["value"]
"#,
        );
        let registry = KgPatternRegistry::compile(&config).expect("应可编译");
        let a = registry.get("capital_a").unwrap().semantic_fingerprint();
        let b = registry.get("capital_b").unwrap().semantic_fingerprint();
        let c = registry.get("capital_other").unwrap().semantic_fingerprint();
        assert_eq!(a, b);
        assert_ne!(a, c);
        let lints = registry.lint();
        assert!(lints.iter().any(|lint| {
            lint.code == "kg_pattern_semantic_duplicate"
                && lint.pattern_ids == vec!["capital_a", "capital_b"]
        }));
    }

    #[test]
    fn semantic_fingerprint_is_atom_order_and_renaming_invariant() {
        let registry = KgPatternRegistry::compile(&config(
            r#"
enabled = true

[[patterns]]
id = "chain_a"
atoms = [
  { subject = "?x", predicate = "p", object = "?y" },
  { subject = "?y", predicate = "q", object = "?z" },
]
filters = [
  { kind = "not_equal", var = "?z", value = "排除" },
  { kind = "type", var = "?x", of = "Person" },
]
select = ["z"]

[[patterns]]
id = "chain_b"
atoms = [
  { subject = "?middle", predicate = "q", object = "?target" },
  { subject = "?root", predicate = "p", object = "?middle" },
]
filters = [
  { kind = "type", var = "?root", of = "Person" },
  { kind = "not_equal", var = "?target", value = "排除" },
]
select = ["target"]

[[patterns]]
id = "chain_c"
atoms = [
  { subject = "?x", predicate = "p", object = "?y" },
  { subject = "?y", predicate = "q", object = "?z" },
  { subject = "?y", predicate = "r", object = "?z" },
]
select = ["z"]
"#,
        ))
        .expect("应可编译");

        let a = registry.get("chain_a").unwrap().semantic_identity();
        let b = registry.get("chain_b").unwrap().semantic_identity();
        let c = registry.get("chain_c").unwrap().semantic_identity();
        assert_eq!(a.fingerprint, b.fingerprint, "{}", a.canonical);
        assert_eq!(a.canonical, b.canonical);
        assert_ne!(a.fingerprint, c.fingerprint);
        assert!(a.complete && b.complete);
        assert!(registry.lint().iter().any(|lint| {
            lint.code == "kg_pattern_semantic_duplicate"
                && lint.pattern_ids == vec!["chain_a", "chain_b"]
        }));
    }

    #[test]
    fn compile_rejects_unbound_projection_and_conflicting_filters() {
        let errors = KgPatternRegistry::compile(&config(
            r#"
enabled = true

[[patterns]]
id = "broken"
atoms = [{ subject = "?x", predicate = "p", object = "?y" }]
filters = [
  { kind = "equal", var = "?x", value = "a" },
  { kind = "not_equal", var = "?x", value = "a" },
]
select = ["missing"]
"#,
        ))
        .expect_err("应拒绝无效模式");
        assert!(errors
            .iter()
            .any(|error| error.code == "kg_pattern_select_unbound"));
        assert!(errors
            .iter()
            .any(|error| error.code == "kg_pattern_filter_conflict"));
    }

    #[test]
    fn count_pattern_returns_total() {
        let registry = KgPatternRegistry::compile(&config(
            r#"
enabled = true

[[patterns]]
id = "masters"
atoms = [{ subject = "?person", predicate = "师傅", object = "?master" }]
select = ["person"]
count = true
"#,
        ))
        .expect("应可编译");
        let result = registry.get("masters").unwrap().execute(
            &facts(),
            &BTreeMap::new(),
            true,
            MatchOptions::default(),
        );
        assert_eq!(result.total, 2);
        assert_eq!(result.answer_values(), vec!["张三", "李四"]);
    }
}

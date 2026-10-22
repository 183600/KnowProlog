use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::graph_question::PatternLexicon;
use crate::link_prediction::Triple;
use crate::prolog_engine::PrologFact;

pub const DEFAULT_MAX_SOLUTIONS: usize = 64;
pub const DEFAULT_MAX_EXPANSIONS: usize = 50_000;
pub const DEFAULT_PAGE_SIZE: usize = 10;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PatternError {
    EmptyPattern,
    EmptyPredicate { edge: usize },
    UnknownVariable { name: String, at: String },
    UnboundSelect { name: String },
    DuplicateSelect { name: String },
    NoRowsToExplain,
    NoPreviousPattern,
    CannotDropLastEdge,
    CannotDropLastConstraint,
}

impl fmt::Display for PatternError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PatternError::EmptyPattern => write!(f, "图谱模式至少需要一条边"),
            PatternError::EmptyPredicate { edge } => {
                write!(f, "第 {} 条边缺少谓词", edge + 1)
            }
            PatternError::UnknownVariable { name, at } => {
                write!(f, "{}引用了未出现的变量「{}」", at, name)
            }
            PatternError::UnboundSelect { name } => {
                write!(f, "投影变量「{}」没有出现在任何模式边中", name)
            }
            PatternError::DuplicateSelect { name } => {
                write!(f, "投影变量「{}」重复出现", name)
            }
            PatternError::NoRowsToExplain => write!(f, "当前没有可解释的匹配结果"),
            PatternError::NoPreviousPattern => write!(f, "没有可撤回的上一次模式"),
            PatternError::CannotDropLastEdge => write!(f, "至少保留一条模式边"),
            PatternError::CannotDropLastConstraint => write!(f, "当前没有可去掉的约束"),
        }
    }
}

impl std::error::Error for PatternError {}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum PatternTerm {
    Var(String),
    Const(String),
}

impl PatternTerm {
    pub fn var(name: impl Into<String>) -> Self {
        PatternTerm::Var(name.into())
    }

    pub fn constant(value: impl Into<String>) -> Self {
        PatternTerm::Const(value.into())
    }

    pub fn name(&self) -> &str {
        match self {
            PatternTerm::Var(name) | PatternTerm::Const(name) => name,
        }
    }

    pub fn is_var(&self) -> bool {
        matches!(self, PatternTerm::Var(_))
    }

    fn bound<'a>(&'a self, binding: &'a BTreeMap<String, String>) -> Option<&'a str> {
        match self {
            PatternTerm::Const(value) => Some(value.as_str()),
            PatternTerm::Var(name) => binding.get(name).map(String::as_str),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EdgePattern {
    pub subject: PatternTerm,
    pub predicate: String,
    pub object: PatternTerm,
}

impl EdgePattern {
    pub fn new(subject: PatternTerm, predicate: impl Into<String>, object: PatternTerm) -> Self {
        Self {
            subject,
            predicate: predicate.into(),
            object,
        }
    }

    pub fn var_const(
        variable: impl Into<String>,
        predicate: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        Self::new(
            PatternTerm::var(variable),
            predicate,
            PatternTerm::constant(value),
        )
    }

    pub fn const_var(
        value: impl Into<String>,
        predicate: impl Into<String>,
        variable: impl Into<String>,
    ) -> Self {
        Self::new(
            PatternTerm::constant(value),
            predicate,
            PatternTerm::var(variable),
        )
    }

    pub fn var_var(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
    ) -> Self {
        Self::new(
            PatternTerm::var(subject),
            predicate,
            PatternTerm::var(object),
        )
    }

    pub fn const_const(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
    ) -> Self {
        Self::new(
            PatternTerm::constant(subject),
            predicate,
            PatternTerm::constant(object),
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PatternConstraint {
    Equal { variable: String, value: String },
    NotEqual { variable: String, value: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphPattern {
    pub edges: Vec<EdgePattern>,
    pub constraints: Vec<PatternConstraint>,
    pub select: Vec<String>,
    pub distinct: bool,
}

impl GraphPattern {
    pub fn new(edges: Vec<EdgePattern>, select: Vec<String>) -> Self {
        Self {
            edges,
            constraints: Vec::new(),
            select,
            distinct: true,
        }
    }

    pub fn with_constraints(mut self, constraints: Vec<PatternConstraint>) -> Self {
        self.constraints = constraints;
        self
    }

    pub fn with_distinct(mut self, distinct: bool) -> Self {
        self.distinct = distinct;
        self
    }

    pub fn validate(&self) -> Result<(), PatternError> {
        if self.edges.is_empty() {
            return Err(PatternError::EmptyPattern);
        }
        for (index, edge) in self.edges.iter().enumerate() {
            if edge.predicate.trim().is_empty() {
                return Err(PatternError::EmptyPredicate { edge: index });
            }
        }
        let mut variables = BTreeSet::new();
        for edge in &self.edges {
            if let PatternTerm::Var(name) = &edge.subject {
                variables.insert(name.as_str());
            }
            if let PatternTerm::Var(name) = &edge.object {
                variables.insert(name.as_str());
            }
        }
        for constraint in &self.constraints {
            let (variable, at) = match constraint {
                PatternConstraint::Equal { variable, .. } => (variable, "等值约束"),
                PatternConstraint::NotEqual { variable, .. } => (variable, "排除约束"),
            };
            if !variables.contains(variable.as_str()) {
                return Err(PatternError::UnknownVariable {
                    name: variable.clone(),
                    at: at.to_string(),
                });
            }
        }
        let mut selected = BTreeSet::new();
        for name in &self.select {
            if !variables.contains(name.as_str()) {
                return Err(PatternError::UnboundSelect { name: name.clone() });
            }
            if !selected.insert(name.as_str()) {
                return Err(PatternError::DuplicateSelect { name: name.clone() });
            }
        }
        Ok(())
    }

    pub fn evaluate(
        &self,
        facts: &[GraphFact],
        options: MatchOptions,
    ) -> Result<PatternOutcome, PatternError> {
        self.validate()?;
        let index = GraphIndex::new(facts);
        self.evaluate_indexed(&index, options)
    }

    fn evaluate_indexed(
        &self,
        index: &GraphIndex<'_>,
        options: MatchOptions,
    ) -> Result<PatternOutcome, PatternError> {
        let options = options.normalized();
        let mut evaluator = Evaluator {
            pattern: self,
            index,
            options,
            expansions: 0,
            truncated: false,
            seen: BTreeSet::new(),
            solutions: Vec::new(),
        };
        let mut binding = BTreeMap::new();
        for constraint in &self.constraints {
            if let PatternConstraint::Equal { variable, value } = constraint {
                binding.insert(variable.clone(), value.clone());
            }
        }
        let remaining: Vec<usize> = (0..self.edges.len()).collect();
        evaluator.search(binding, Vec::new(), remaining);
        evaluator.solutions.sort_by(|left, right| {
            left.projected(&self.select)
                .cmp(&right.projected(&self.select))
                .then_with(|| left.binding.cmp(&right.binding))
                .then_with(|| left.witness.cmp(&right.witness))
        });
        Ok(PatternOutcome {
            solutions: evaluator.solutions,
            truncated: evaluator.truncated,
            expansions: evaluator.expansions,
        })
    }

    pub fn exists(&self, facts: &[GraphFact], options: MatchOptions) -> bool {
        let mut limited = self.clone();
        limited.select.clear();
        limited
            .evaluate(
                facts,
                MatchOptions {
                    max_solutions: 1,
                    ..options
                },
            )
            .map(|outcome| !outcome.solutions.is_empty())
            .unwrap_or(false)
    }
}

impl Default for GraphPattern {
    fn default() -> Self {
        Self::new(Vec::new(), Vec::new())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FactSource {
    Relation,
    Type,
    Attribute,
}

impl FactSource {
    pub fn label(self) -> &'static str {
        match self {
            FactSource::Relation => "关系",
            FactSource::Type => "类型",
            FactSource::Attribute => "属性",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GraphFact {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub source: FactSource,
}

impl GraphFact {
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
        source: FactSource,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
            source,
        }
    }

    pub fn relation(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
    ) -> Self {
        Self::new(subject, predicate, object, FactSource::Relation)
    }

    pub fn type_assertion(subject: impl Into<String>, entity_type: impl Into<String>) -> Self {
        Self::new(subject, "type", entity_type, FactSource::Type)
    }

    pub fn attribute(
        subject: impl Into<String>,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        Self::new(subject, key, value, FactSource::Attribute)
    }

    pub fn from_triples(triples: &[Triple]) -> Vec<Self> {
        triples
            .iter()
            .map(|triple| Self::relation(&triple.subject, &triple.predicate, &triple.object))
            .collect()
    }

    pub fn from_prolog_facts(facts: &[PrologFact]) -> Vec<Self> {
        let mut graph = Vec::new();
        for fact in facts {
            match fact {
                PrologFact::Relation { from, rel_type, to } => {
                    graph.push(Self::relation(from, rel_type, to));
                }
                PrologFact::Entity {
                    name,
                    entity_type,
                    properties,
                } => {
                    if !entity_type.trim().is_empty() {
                        graph.push(Self::type_assertion(name, entity_type));
                    }
                    let mut keys: Vec<&String> = properties.keys().collect();
                    keys.sort();
                    for key in keys {
                        if let Some(value) = properties.get(key) {
                            graph.push(Self::attribute(name, key, value));
                        }
                    }
                }
            }
        }
        graph
    }
}

pub struct GraphIndex<'a> {
    facts: &'a [GraphFact],
    by_sp: HashMap<(&'a str, &'a str), Vec<usize>>,
    by_po: HashMap<(&'a str, &'a str), Vec<usize>>,
    by_p: HashMap<&'a str, Vec<usize>>,
}

impl<'a> GraphIndex<'a> {
    pub fn new(facts: &'a [GraphFact]) -> Self {
        let mut by_sp: HashMap<(&str, &str), Vec<usize>> = HashMap::new();
        let mut by_po: HashMap<(&str, &str), Vec<usize>> = HashMap::new();
        let mut by_p: HashMap<&str, Vec<usize>> = HashMap::new();
        for (index, fact) in facts.iter().enumerate() {
            by_sp
                .entry((fact.subject.as_str(), fact.predicate.as_str()))
                .or_default()
                .push(index);
            by_po
                .entry((fact.predicate.as_str(), fact.object.as_str()))
                .or_default()
                .push(index);
            by_p.entry(fact.predicate.as_str()).or_default().push(index);
        }
        Self {
            facts,
            by_sp,
            by_po,
            by_p,
        }
    }

    pub fn len(&self) -> usize {
        self.facts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.facts.is_empty()
    }

    pub fn candidates(
        &self,
        subject: Option<&str>,
        predicate: &str,
        object: Option<&str>,
    ) -> Vec<usize> {
        match (subject, object) {
            (Some(subject), Some(object)) => self
                .by_sp
                .get(&(subject, predicate))
                .map(|indices| {
                    indices
                        .iter()
                        .copied()
                        .filter(|index| self.facts[*index].object == object)
                        .collect()
                })
                .unwrap_or_default(),
            (Some(subject), None) => self
                .by_sp
                .get(&(subject, predicate))
                .cloned()
                .unwrap_or_default(),
            (None, Some(object)) => self
                .by_po
                .get(&(predicate, object))
                .cloned()
                .unwrap_or_default(),
            (None, None) => self.by_p.get(predicate).cloned().unwrap_or_default(),
        }
    }

    pub fn predicate_count(&self, predicate: &str) -> usize {
        self.by_p.get(predicate).map(Vec::len).unwrap_or(0)
    }

    /// 指定主语/谓词的事实数（宾语不参与过滤）。
    pub fn subject_predicate_count(&self, subject: &str, predicate: &str) -> usize {
        self.by_sp
            .get(&(subject, predicate))
            .map(Vec::len)
            .unwrap_or(0)
    }

    /// 指定谓词/宾语的事实数（主语不参与过滤）。
    pub fn predicate_object_count(&self, predicate: &str, object: &str) -> usize {
        self.by_po
            .get(&(predicate, object))
            .map(Vec::len)
            .unwrap_or(0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MatchOptions {
    pub max_solutions: usize,
    pub max_expansions: usize,
}

impl MatchOptions {
    pub fn new(max_solutions: usize, max_expansions: usize) -> Self {
        Self {
            max_solutions,
            max_expansions,
        }
    }

    fn normalized(self) -> Self {
        Self {
            max_solutions: self.max_solutions.max(1),
            max_expansions: self.max_expansions.max(1),
        }
    }
}

impl Default for MatchOptions {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_SOLUTIONS, DEFAULT_MAX_EXPANSIONS)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PatternSolution {
    pub binding: BTreeMap<String, String>,
    pub witness: Vec<usize>,
}

impl PatternSolution {
    pub fn projected(&self, select: &[String]) -> Vec<String> {
        select
            .iter()
            .map(|variable| self.binding.get(variable).cloned().unwrap_or_default())
            .collect()
    }

    pub fn row(&self, select: &[String]) -> String {
        self.projected(select).join("、")
    }

    pub fn witness_lines(&self, facts: &[GraphFact]) -> Vec<String> {
        let mut indices = self.witness.clone();
        indices.sort_unstable();
        indices.dedup();
        indices
            .into_iter()
            .filter_map(|index| facts.get(index))
            .enumerate()
            .map(|(position, fact)| {
                let rendered = match fact.source {
                    FactSource::Type => {
                        format!("{} 的类型是 {}", fact.subject, fact.object)
                    }
                    FactSource::Attribute => {
                        format!("{} 的 {} 是 {}", fact.subject, fact.predicate, fact.object)
                    }
                    FactSource::Relation => {
                        format!("{} —{}→ {}", fact.subject, fact.predicate, fact.object)
                    }
                };
                format!("[E{}] {}", position + 1, rendered)
            })
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PatternOutcome {
    pub solutions: Vec<PatternSolution>,
    pub truncated: bool,
    pub expansions: usize,
}

impl PatternOutcome {
    pub fn len(&self) -> usize {
        self.solutions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.solutions.is_empty()
    }

    pub fn is_complete(&self) -> bool {
        !self.truncated
    }

    pub fn count_is_lower_bound(&self) -> bool {
        self.truncated
    }

    pub fn explain(&self, select: &[String], facts: &[GraphFact]) -> Option<String> {
        self.explain_at(0, select, facts)
    }

    /// 解释指定下标的匹配结果；翻页后「为什么」应指向当前页第一条，
    /// 而不是永远回到整张结果集的第一条。
    pub fn explain_at(
        &self,
        index: usize,
        select: &[String],
        facts: &[GraphFact],
    ) -> Option<String> {
        let solution = self.solutions.get(index)?;
        let mut output = String::new();
        if !select.is_empty() {
            let assignments: Vec<String> = select
                .iter()
                .filter_map(|variable| {
                    solution
                        .binding
                        .get(variable)
                        .map(|value| format!("{} = {}", variable, value))
                })
                .collect();
            if !assignments.is_empty() {
                output.push_str("匹配绑定：");
                output.push_str(&assignments.join("，"));
                output.push('\n');
            }
        }
        output.push_str("见证子图：\n");
        for line in solution.witness_lines(facts) {
            output.push_str("  ");
            output.push_str(&line);
            output.push('\n');
        }
        if self.truncated {
            output.push_str("说明：搜索预算已用尽，当前结果只是下界。\n");
        }
        Some(output.trim_end().to_string())
    }
}

struct Evaluator<'a> {
    pattern: &'a GraphPattern,
    index: &'a GraphIndex<'a>,
    options: MatchOptions,
    expansions: usize,
    truncated: bool,
    seen: BTreeSet<Vec<String>>,
    solutions: Vec<PatternSolution>,
}

impl<'a> Evaluator<'a> {
    fn constraints_hold(&self, binding: &BTreeMap<String, String>) -> bool {
        self.pattern.constraints.iter().all(|constraint| match constraint {
            PatternConstraint::Equal { variable, value } => {
                binding.get(variable) == Some(value)
            }
            PatternConstraint::NotEqual { variable, value } => {
                binding.get(variable) != Some(value)
            }
        })
    }

    fn search(
        &mut self,
        binding: BTreeMap<String, String>,
        witness: Vec<usize>,
        remaining: Vec<usize>,
    ) {
        if self.solutions.len() >= self.options.max_solutions {
            self.truncated = true;
            return;
        }
        if remaining.is_empty() {
            if !self.constraints_hold(&binding) {
                return;
            }
            let row = self.project(&binding);
            if self.pattern.distinct && !self.seen.insert(row) {
                return;
            }
            self.solutions.push(PatternSolution { binding, witness });
            return;
        }

        let edge_index = self.choose_edge(&binding, &remaining);
        let edge = &self.pattern.edges[edge_index];
        let subject = edge.subject.bound(&binding);
        let object = edge.object.bound(&binding);
        let candidates = self.index.candidates(subject, &edge.predicate, object);
        let mut next_remaining = remaining;
        next_remaining.retain(|index| *index != edge_index);

        for fact_index in candidates {
            if self.expansions >= self.options.max_expansions {
                self.truncated = true;
                return;
            }
            self.expansions += 1;
            let fact = &self.index.facts[fact_index];
            let mut next = binding.clone();
            if !bind(&edge.subject, &fact.subject, &mut next) {
                continue;
            }
            if !bind(&edge.object, &fact.object, &mut next) {
                continue;
            }
            let mut next_witness = witness.clone();
            next_witness.push(fact_index);
            self.search(next, next_witness, next_remaining.clone());
            if self.truncated {
                return;
            }
        }
    }

    fn choose_edge(&self, binding: &BTreeMap<String, String>, remaining: &[usize]) -> usize {
        *remaining
            .iter()
            .min_by_key(|index| {
                let edge = &self.pattern.edges[**index];
                let bound = edge.subject.bound(binding).is_some() as usize
                    + edge.object.bound(binding).is_some() as usize;
                (
                    Reverse(bound),
                    self.index.predicate_count(&edge.predicate),
                    Reverse(**index),
                )
            })
            .expect("remaining 非空")
    }

    fn project(&self, binding: &BTreeMap<String, String>) -> Vec<String> {
        self.pattern
            .select
            .iter()
            .map(|variable| binding.get(variable).cloned().unwrap_or_default())
            .collect()
    }
}

fn bind(term: &PatternTerm, value: &str, binding: &mut BTreeMap<String, String>) -> bool {
    match term {
        PatternTerm::Const(expected) => expected == value,
        PatternTerm::Var(name) => match binding.get(name) {
            Some(existing) => existing == value,
            None => {
                binding.insert(name.clone(), value.to_string());
                true
            }
        },
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GraphCommand {
    NextPage,
    PreviousPage,
    Explain,
    Count,
    DropLastEdge,
    DropLastConstraint,
    AddEdge(EdgePattern),
    AddConstraint(PatternConstraint),
    Withdraw,
    Replace(GraphPattern),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GraphReply {
    Page {
        page: usize,
        pages: usize,
        total: usize,
        lower_bound: bool,
        rows: Vec<PatternSolution>,
    },
    Explanation(String),
    Count {
        count: usize,
        lower_bound: bool,
    },
    Changed {
        total: usize,
        lower_bound: bool,
        pattern: GraphPattern,
    },
}

#[derive(Debug, Clone)]
pub struct ConversationalGraph {
    pattern: GraphPattern,
    outcome: PatternOutcome,
    page: usize,
    page_size: usize,
    previous: Vec<GraphPattern>,
}

impl ConversationalGraph {
    pub fn new(
        pattern: GraphPattern,
        facts: &[GraphFact],
        options: MatchOptions,
    ) -> Result<Self, PatternError> {
        let outcome = pattern.evaluate(facts, options)?;
        Ok(Self {
            pattern,
            outcome,
            page: 0,
            page_size: DEFAULT_PAGE_SIZE,
            previous: Vec::new(),
        })
    }

    pub fn with_page_size(mut self, page_size: usize) -> Self {
        self.page_size = page_size.max(1);
        self.page = self.page.min(self.last_page());
        self
    }

    pub fn pattern(&self) -> &GraphPattern {
        &self.pattern
    }

    pub fn outcome(&self) -> &PatternOutcome {
        &self.outcome
    }

    pub fn current_rows(&self) -> &[PatternSolution] {
        let start = self.page.saturating_mul(self.page_size);
        let end = (start + self.page_size).min(self.outcome.solutions.len());
        if start >= end {
            &[]
        } else {
            &self.outcome.solutions[start..end]
        }
    }

    pub fn page(&self) -> usize {
        self.page
    }

    pub fn page_count(&self) -> usize {
        self.last_page() + 1
    }

    fn last_page(&self) -> usize {
        self.outcome
            .solutions
            .len()
            .saturating_sub(1)
            .div_euclid(self.page_size.max(1))
    }

    pub fn apply(
        &mut self,
        command: GraphCommand,
        facts: &[GraphFact],
        options: MatchOptions,
    ) -> Result<GraphReply, PatternError> {
        match command {
            GraphCommand::NextPage => {
                let last = self.last_page();
                if self.page < last {
                    self.page += 1;
                }
                Ok(self.page_reply())
            }
            GraphCommand::PreviousPage => {
                self.page = self.page.saturating_sub(1);
                Ok(self.page_reply())
            }
            GraphCommand::Explain => {
                let index = self.page.saturating_mul(self.page_size);
                self.outcome
                    .explain_at(index, &self.pattern.select, facts)
                    .map(GraphReply::Explanation)
                    .ok_or(PatternError::NoRowsToExplain)
            }
            GraphCommand::Count => {
                let outcome = self.pattern.evaluate(facts, options)?;
                Ok(GraphReply::Count {
                    count: outcome.solutions.len(),
                    lower_bound: outcome.count_is_lower_bound(),
                })
            }
            GraphCommand::DropLastEdge => {
                if self.pattern.edges.len() <= 1 {
                    return Err(PatternError::CannotDropLastEdge);
                }
                let mut next = self.pattern.clone();
                next.edges.pop();
                self.replace(next, facts, options)
            }
            GraphCommand::DropLastConstraint => {
                if self.pattern.constraints.is_empty() {
                    return Err(PatternError::CannotDropLastConstraint);
                }
                let mut next = self.pattern.clone();
                next.constraints.pop();
                self.replace(next, facts, options)
            }
            GraphCommand::AddEdge(edge) => {
                let mut next = self.pattern.clone();
                next.edges.push(edge);
                self.replace(next, facts, options)
            }
            GraphCommand::AddConstraint(constraint) => {
                let mut next = self.pattern.clone();
                next.constraints.push(constraint);
                self.replace(next, facts, options)
            }
            GraphCommand::Withdraw => {
                let next = self.previous.pop().ok_or(PatternError::NoPreviousPattern)?;
                self.commit(next, facts, options)
            }
            GraphCommand::Replace(pattern) => self.replace(pattern, facts, options),
        }
    }

    fn replace(
        &mut self,
        pattern: GraphPattern,
        facts: &[GraphFact],
        options: MatchOptions,
    ) -> Result<GraphReply, PatternError> {
        let previous = self.pattern.clone();
        let outcome = pattern.evaluate(facts, options)?;
        self.previous.push(previous);
        self.pattern = pattern;
        self.outcome = outcome;
        self.page = 0;
        Ok(self.changed_reply())
    }

    fn commit(
        &mut self,
        pattern: GraphPattern,
        facts: &[GraphFact],
        options: MatchOptions,
    ) -> Result<GraphReply, PatternError> {
        let outcome = pattern.evaluate(facts, options)?;
        self.pattern = pattern;
        self.outcome = outcome;
        self.page = 0;
        Ok(self.changed_reply())
    }

    fn changed_reply(&self) -> GraphReply {
        GraphReply::Changed {
            total: self.outcome.solutions.len(),
            lower_bound: self.outcome.count_is_lower_bound(),
            pattern: self.pattern.clone(),
        }
    }

    fn page_reply(&self) -> GraphReply {
        GraphReply::Page {
            page: self.page,
            pages: self.page_count(),
            total: self.outcome.solutions.len(),
            lower_bound: self.outcome.count_is_lower_bound(),
            rows: self.current_rows().to_vec(),
        }
    }

    pub fn interpret(
        &mut self,
        text: &str,
        lexicon: &PatternLexicon,
        facts: &[GraphFact],
        options: MatchOptions,
    ) -> Option<Result<GraphReply, PatternError>> {
        let command = self.command_from_text(text, lexicon)?;
        Some(self.apply(command, facts, options))
    }

    fn command_from_text(&self, text: &str, lexicon: &PatternLexicon) -> Option<GraphCommand> {
        let text = text
            .trim()
            .trim_end_matches(['？', '?', '！', '!', '。', '.', ' ']);
        if text.is_empty() {
            return None;
        }
        if [
            "还有呢",
            "还有吗",
            "还有别的",
            "还有",
            "继续",
            "下一页",
            "下一个",
        ]
        .iter()
        .any(|phrase| text == *phrase || text.starts_with(phrase))
        {
            return Some(GraphCommand::NextPage);
        }
        if ["上一页", "前一页", "往回翻", "返回上一页"]
            .iter()
            .any(|phrase| text == *phrase || text.starts_with(phrase))
        {
            return Some(GraphCommand::PreviousPage);
        }
        if ["为什么", "依据", "证据", "解释", "怎么知道", "凭什么"]
            .iter()
            .any(|phrase| text.contains(phrase))
        {
            return Some(GraphCommand::Explain);
        }
        if ["多少", "几个", "总数", "一共", "数量"]
            .iter()
            .any(|phrase| text.contains(phrase))
        {
            return Some(GraphCommand::Count);
        }
        if ["撤回", "撤销", "回到上一个", "恢复上一个"]
            .iter()
            .any(|phrase| text.contains(phrase))
        {
            return Some(GraphCommand::Withdraw);
        }
        let drops_condition = ["去掉", "删除", "取消", "少一个", "不要"]
            .iter()
            .any(|phrase| text.contains(phrase))
            && ["条件", "约束", "这条", "最后"]
                .iter()
                .any(|phrase| text.contains(phrase));
        if drops_condition {
            if self.pattern.constraints.is_empty() {
                return Some(GraphCommand::DropLastEdge);
            }
            return Some(GraphCommand::DropLastConstraint);
        }
        if let Some(target) = exclusion_target(text) {
            let variable = self.focus_variable()?;
            let (value, is_type) = lexicon.resolve_refinement(target)?;
            if !is_type {
                return Some(GraphCommand::AddConstraint(PatternConstraint::NotEqual {
                    variable,
                    value,
                }));
            }
        }
        for prefix in ["只看", "只要", "限定", "仅看", "只保留"] {
            if let Some(rest) = text.strip_prefix(prefix) {
                return self.refinement_command(rest, lexicon);
            }
        }
        let core = if let Some(rest) = text.strip_prefix("那") {
            rest.strip_suffix('呢')
        } else {
            text.strip_suffix('呢')
        }?;
        self.refinement_command(core, lexicon)
    }

    fn refinement_command(
        &self,
        core: &str,
        lexicon: &PatternLexicon,
    ) -> Option<GraphCommand> {
        let mut core = core.trim();
        for prefix in ["如果", "要是", "若是", "那么"] {
            if let Some(rest) = core.strip_prefix(prefix) {
                core = rest.trim();
            }
        }
        let core = core.trim_start_matches(['是', '有', '要', ' ']);
        let core = core.trim_end_matches(['的', '吧', '呢', ' ']);
        if core.is_empty() {
            return None;
        }
        let variable = self.focus_variable()?;
        let (value, is_type) = lexicon.resolve_refinement(core)?;
        if is_type {
            return Some(GraphCommand::AddEdge(EdgePattern::var_const(
                variable, "type", value,
            )));
        }
        Some(GraphCommand::AddConstraint(PatternConstraint::Equal {
            variable,
            value,
        }))
    }

    fn focus_variable(&self) -> Option<String> {
        if let Some(variable) = self.pattern.select.first() {
            return Some(variable.clone());
        }
        self.pattern
            .edges
            .iter()
            .find_map(|edge| match &edge.subject {
                PatternTerm::Var(name) => Some(name.clone()),
                PatternTerm::Const(_) => match &edge.object {
                    PatternTerm::Var(name) => Some(name.clone()),
                    PatternTerm::Const(_) => None,
                },
            })
    }
}

fn exclusion_target(text: &str) -> Option<&str> {
    let mut rest = None;
    for prefix in ["除了", "除去", "排除", "不要", "不是"] {
        if let Some(stripped) = text.strip_prefix(prefix) {
            rest = Some(stripped.trim());
            break;
        }
    }
    let mut rest = rest?;
    loop {
        let before = rest;
        rest = rest.trim_end_matches(['了', '的', '呢', '吧', '吗', ' ', '？', '?']);
        for suffix in ["还有谁", "还有哪些", "还有别的", "还有", "其他", "其它"] {
            if let Some(stripped) = rest.strip_suffix(suffix) {
                rest = stripped.trim();
                break;
            }
        }
        if rest == before {
            break;
        }
    }
    if rest.is_empty() {
        None
    } else {
        Some(rest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_facts() -> Vec<GraphFact> {
        vec![
            GraphFact::relation("张三", "认识", "李四"),
            GraphFact::relation("张三", "认识", "王五"),
            GraphFact::relation("李四", "认识", "王五"),
            GraphFact::relation("赵六", "认识", "王五"),
            GraphFact::type_assertion("张三", "Person"),
            GraphFact::type_assertion("李四", "Person"),
            GraphFact::relation("张三", "导演", "电影A"),
            GraphFact::relation("李四", "导演", "电影A"),
            GraphFact::relation("赵六", "导演", "电影A"),
            GraphFact::relation("王五", "导演", "电影B"),
            GraphFact::type_assertion("电影A", "科幻片"),
            GraphFact::type_assertion("电影B", "剧情片"),
            GraphFact::relation("张三", "获奖", "金棕榈"),
            GraphFact::relation("李四", "获奖", "金棕榈"),
            GraphFact::relation("王五", "获奖", "金棕榈"),
            GraphFact::relation("姚明", "出生地", "上海"),
            GraphFact::relation("刘翔", "出生地", "上海"),
            GraphFact::relation("王五", "出生地", "北京"),
        ]
    }

    fn solution_values(outcome: &PatternOutcome, select: &[String]) -> Vec<Vec<String>> {
        outcome
            .solutions
            .iter()
            .map(|solution| solution.projected(select))
            .collect()
    }

    #[test]
    fn common_object_is_shared_variable() {
        let pattern = GraphPattern::new(
            vec![
                EdgePattern::const_var("张三", "认识", "friend"),
                EdgePattern::const_var("李四", "认识", "friend"),
            ],
            vec!["friend".to_string()],
        );
        let outcome = pattern
            .evaluate(&sample_facts(), MatchOptions::default())
            .expect("well formed pattern");
        assert_eq!(
            solution_values(&outcome, &pattern.select),
            vec![vec!["王五".to_string()]]
        );
        assert!(outcome.is_complete());
    }

    #[test]
    fn conjunction_requires_all_edges() {
        let pattern = GraphPattern::new(
            vec![
                EdgePattern::var_var("person", "导演", "film"),
                EdgePattern::var_const("film", "type", "科幻片"),
                EdgePattern::var_const("person", "获奖", "金棕榈"),
            ],
            vec!["person".to_string()],
        );
        let outcome = pattern
            .evaluate(&sample_facts(), MatchOptions::default())
            .expect("well formed pattern");
        assert_eq!(
            solution_values(&outcome, &pattern.select),
            vec![vec!["张三".to_string()], vec!["李四".to_string()]]
        );
        assert!(outcome
            .solutions
            .iter()
            .all(|solution| solution.witness.len() == 3));
    }

    #[test]
    fn equality_join_returns_common_city() {
        let pattern = GraphPattern::new(
            vec![
                EdgePattern::const_var("姚明", "出生地", "city"),
                EdgePattern::const_var("刘翔", "出生地", "city"),
            ],
            vec!["city".to_string()],
        );
        let facts = sample_facts();
        let outcome = pattern.evaluate(&facts, MatchOptions::default()).unwrap();
        assert_eq!(outcome.len(), 1);
        let explanation = outcome
            .explain(&pattern.select, &facts)
            .expect("有匹配可解释");
        assert!(explanation.contains("[E1]"));
        assert!(explanation.contains("[E2]"));
        assert!(explanation.contains("city = 上海"));
    }

    #[test]
    fn missing_join_yields_empty_result() {
        let pattern = GraphPattern::new(
            vec![
                EdgePattern::const_var("姚明", "出生地", "city"),
                EdgePattern::const_var("王五", "出生地", "city"),
            ],
            vec!["city".to_string()],
        );
        let outcome = pattern
            .evaluate(&sample_facts(), MatchOptions::default())
            .unwrap();
        assert!(outcome.is_empty());
        assert!(outcome.explain(&pattern.select, &sample_facts()).is_none());
    }

    #[test]
    fn constrained_binding_participates_in_join() {
        let pattern = GraphPattern::new(
            vec![
                EdgePattern::var_var("person", "导演", "film"),
                EdgePattern::var_const("film", "type", "科幻片"),
                EdgePattern::var_const("person", "获奖", "金棕榈"),
            ],
            vec!["person".to_string()],
        )
        .with_constraints(vec![PatternConstraint::Equal {
            variable: "person".to_string(),
            value: "李四".to_string(),
        }]);
        let outcome = pattern
            .evaluate(&sample_facts(), MatchOptions::default())
            .unwrap();
        assert_eq!(
            solution_values(&outcome, &pattern.select),
            vec![vec!["李四".to_string()]]
        );
    }

    #[test]
    fn budget_reports_lower_bound_count() {
        let pattern = GraphPattern::new(
            vec![EdgePattern::var_const("person", "获奖", "金棕榈")],
            vec!["person".to_string()],
        );
        let outcome = pattern
            .evaluate(
                &sample_facts(),
                MatchOptions::new(1, DEFAULT_MAX_EXPANSIONS),
            )
            .unwrap();
        assert_eq!(outcome.len(), 1);
        assert!(outcome.truncated);
        assert!(outcome.count_is_lower_bound());
    }

    #[test]
    fn distinct_default_deduplicates_identical_rows() {
        let mut facts = sample_facts();
        facts.push(GraphFact::relation("张三", "获奖", "金棕榈"));
        let pattern = GraphPattern::new(
            vec![EdgePattern::var_const("person", "获奖", "金棕榈")],
            vec!["person".to_string()],
        );
        let outcome = pattern.evaluate(&facts, MatchOptions::default()).unwrap();
        assert_eq!(outcome.len(), 3);
        let all = pattern
            .clone()
            .with_distinct(false)
            .evaluate(&facts, MatchOptions::default())
            .unwrap();
        assert_eq!(all.len(), 4);
    }

    #[test]
    fn pattern_validation_rejects_bad_references() {
        let unknown_select = GraphPattern::new(
            vec![EdgePattern::var_const("person", "获奖", "金棕榈")],
            vec!["film".to_string()],
        );
        assert!(matches!(
            unknown_select.validate(),
            Err(PatternError::UnboundSelect { .. })
        ));
        let empty = GraphPattern::new(Vec::new(), vec![]);
        assert!(matches!(empty.validate(), Err(PatternError::EmptyPattern)));
        let duplicate_select = GraphPattern::new(
            vec![EdgePattern::var_const("person", "获奖", "金棕榈")],
            vec!["person".to_string(), "person".to_string()],
        );
        assert!(matches!(
            duplicate_select.validate(),
            Err(PatternError::DuplicateSelect { .. })
        ));
        let unknown_constraint = GraphPattern::new(
            vec![EdgePattern::var_const("person", "获奖", "金棕榈")],
            vec!["person".to_string()],
        )
        .with_constraints(vec![PatternConstraint::Equal {
            variable: "film".to_string(),
            value: "电影A".to_string(),
        }]);
        assert!(matches!(
            unknown_constraint.validate(),
            Err(PatternError::UnknownVariable { .. })
        ));
    }

    #[test]
    fn prolog_facts_are_indexed_as_typed_attributes() {
        let mut properties = HashMap::new();
        properties.insert("年龄".to_string(), "18".to_string());
        let facts = vec![
            PrologFact::Entity {
                name: "小明".to_string(),
                entity_type: "Person".to_string(),
                properties,
            },
            PrologFact::Relation {
                from: "小明".to_string(),
                rel_type: "认识".to_string(),
                to: "小红".to_string(),
            },
        ];
        let graph = GraphFact::from_prolog_facts(&facts);
        assert!(graph.contains(&GraphFact::type_assertion("小明", "Person")));
        assert!(graph.contains(&GraphFact::attribute("小明", "年龄", "18")));
        assert!(graph.contains(&GraphFact::relation("小明", "认识", "小红")));
        let pattern = GraphPattern::new(
            vec![
                EdgePattern::const_var("小明", "type", "type"),
                EdgePattern::const_const("小明", "年龄", "18"),
            ],
            vec!["type".to_string()],
        );
        let outcome = pattern.evaluate(&graph, MatchOptions::default()).unwrap();
        assert_eq!(
            solution_values(&outcome, &pattern.select),
            vec![vec!["Person".to_string()]]
        );
    }

    #[test]
    fn paging_navigates_deterministically() {
        let facts: Vec<GraphFact> = (0..7)
            .map(|index| GraphFact::relation(format!("p{}", index), "获奖", "金棕榈"))
            .collect();
        let pattern = GraphPattern::new(
            vec![EdgePattern::var_const("person", "获奖", "金棕榈")],
            vec!["person".to_string()],
        );
        let mut conversation = ConversationalGraph::new(pattern, &facts, MatchOptions::default())
            .unwrap()
            .with_page_size(3);
        let first = conversation
            .apply(GraphCommand::NextPage, &facts, MatchOptions::default())
            .unwrap();
        match first {
            GraphReply::Page {
                page,
                pages,
                total,
                rows,
                ..
            } => {
                assert_eq!(page, 1);
                assert_eq!(pages, 3);
                assert_eq!(total, 7);
                assert_eq!(
                    rows[0].projected(&["person".to_string()]),
                    vec!["p3".to_string()]
                );
            }
            other => panic!("unexpected reply: {:?}", other),
        }
        let last = conversation
            .apply(GraphCommand::NextPage, &facts, MatchOptions::default())
            .unwrap();
        assert!(matches!(
            last,
            GraphReply::Page {
                page: 2,
                pages: 3,
                ..
            }
        ));
        let clamped = conversation
            .apply(GraphCommand::NextPage, &facts, MatchOptions::default())
            .unwrap();
        assert!(matches!(
            clamped,
            GraphReply::Page {
                page: 2,
                pages: 3,
                ..
            }
        ));
        let back = conversation
            .apply(GraphCommand::PreviousPage, &facts, MatchOptions::default())
            .unwrap();
        assert!(matches!(back, GraphReply::Page { page: 1, .. }));
    }

    #[test]
    fn explain_after_paging_targets_the_current_page() {
        let facts: Vec<GraphFact> = (0..7)
            .map(|index| GraphFact::relation(format!("p{}", index), "获奖", "金棕榈"))
            .collect();
        let pattern = GraphPattern::new(
            vec![EdgePattern::var_const("person", "获奖", "金棕榈")],
            vec!["person".to_string()],
        );
        let mut conversation = ConversationalGraph::new(pattern, &facts, MatchOptions::default())
            .unwrap()
            .with_page_size(2);
        conversation
            .apply(GraphCommand::NextPage, &facts, MatchOptions::default())
            .unwrap();
        let explanation = conversation
            .apply(GraphCommand::Explain, &facts, MatchOptions::default())
            .unwrap();
        match explanation {
            GraphReply::Explanation(text) => {
                assert!(text.contains("person = p2"), "{}", text);
                assert!(!text.contains("person = p0"), "{}", text);
            }
            other => panic!("unexpected reply: {:?}", other),
        }
    }

    #[test]
    fn refine_and_withdraw_restore_previous_pattern() {
        let facts = sample_facts();
        let pattern = GraphPattern::new(
            vec![EdgePattern::var_const("person", "获奖", "金棕榈")],
            vec!["person".to_string()],
        );
        let mut conversation =
            ConversationalGraph::new(pattern, &facts, MatchOptions::default()).unwrap();
        let added = conversation
            .apply(
                GraphCommand::AddEdge(EdgePattern::var_var("person", "导演", "film")),
                &facts,
                MatchOptions::default(),
            )
            .unwrap();
        assert!(matches!(added, GraphReply::Changed { total: 3, .. }));
        let narrowed = conversation
            .apply(
                GraphCommand::AddEdge(EdgePattern::var_const("film", "type", "科幻片")),
                &facts,
                MatchOptions::default(),
            )
            .unwrap();
        assert!(matches!(narrowed, GraphReply::Changed { total: 2, .. }));
        let withdrawn = conversation
            .apply(GraphCommand::Withdraw, &facts, MatchOptions::default())
            .unwrap();
        assert!(matches!(withdrawn, GraphReply::Changed { total: 3, .. }));
        assert_eq!(conversation.pattern().edges.len(), 2);
        let withdrawn_again = conversation
            .apply(GraphCommand::Withdraw, &facts, MatchOptions::default())
            .unwrap();
        assert!(matches!(
            withdrawn_again,
            GraphReply::Changed { total: 3, .. }
        ));
        assert_eq!(conversation.pattern().edges.len(), 1);
        assert!(matches!(
            conversation.apply(GraphCommand::Withdraw, &facts, MatchOptions::default()),
            Err(PatternError::NoPreviousPattern)
        ));
    }

    #[test]
    fn drop_edge_and_guard_last_edge() {
        let facts = sample_facts();
        let pattern = GraphPattern::new(
            vec![
                EdgePattern::var_var("person", "导演", "film"),
                EdgePattern::var_const("film", "type", "科幻片"),
            ],
            vec!["person".to_string()],
        );
        let mut conversation =
            ConversationalGraph::new(pattern, &facts, MatchOptions::default()).unwrap();
        let dropped = conversation
            .apply(GraphCommand::DropLastEdge, &facts, MatchOptions::default())
            .unwrap();
        assert!(matches!(dropped, GraphReply::Changed { total: 4, .. }));
        assert!(matches!(
            conversation.apply(GraphCommand::DropLastEdge, &facts, MatchOptions::default()),
            Err(PatternError::CannotDropLastEdge)
        ));
    }

    #[test]
    fn replace_resets_paging_and_count_reports_truncation() {
        let facts = sample_facts();
        let pattern = GraphPattern::new(
            vec![EdgePattern::var_const("person", "获奖", "金棕榈")],
            vec!["person".to_string()],
        );
        let mut conversation =
            ConversationalGraph::new(pattern, &facts, MatchOptions::default()).unwrap();
        let replacement = GraphPattern::new(
            vec![EdgePattern::var_var("person", "出生地", "city")],
            vec!["person".to_string()],
        );
        let changed = conversation
            .apply(
                GraphCommand::Replace(replacement),
                &facts,
                MatchOptions::default(),
            )
            .unwrap();
        assert!(matches!(changed, GraphReply::Changed { total: 3, .. }));
        assert_eq!(conversation.page(), 0);
        let count = conversation
            .apply(
                GraphCommand::Count,
                &facts,
                MatchOptions::new(1, DEFAULT_MAX_EXPANSIONS),
            )
            .unwrap();
        assert!(matches!(
            count,
            GraphReply::Count {
                count: 1,
                lower_bound: true
            }
        ));
    }

    fn interpretation_lexicon() -> PatternLexicon {
        let mut lexicon = PatternLexicon::new();
        for entity in ["张三", "李四", "王五", "赵六", "金棕榈"] {
            lexicon.add_entity(entity, entity);
        }
        lexicon.add_predicate("获奖", "获奖");
        lexicon.add_type("科幻片", "科幻片");
        lexicon
    }

    #[test]
    fn interpret_pages_explains_and_counts() {
        let facts = sample_facts();
        let pattern = GraphPattern::new(
            vec![EdgePattern::var_const("person", "获奖", "金棕榈")],
            vec!["person".to_string()],
        );
        let mut conversation = ConversationalGraph::new(pattern, &facts, MatchOptions::default())
            .unwrap()
            .with_page_size(2);
        let lexicon = interpretation_lexicon();
        let page = conversation
            .interpret("还有呢", &lexicon, &facts, MatchOptions::default())
            .unwrap()
            .unwrap();
        assert!(matches!(page, GraphReply::Page { page: 1, .. }));
        let explanation = conversation
            .interpret("为什么", &lexicon, &facts, MatchOptions::default())
            .unwrap()
            .unwrap();
        assert!(matches!(explanation, GraphReply::Explanation(_)));
        let count = conversation
            .interpret("一共多少个", &lexicon, &facts, MatchOptions::default())
            .unwrap()
            .unwrap();
        assert!(matches!(count, GraphReply::Count { count: 3, .. }));
    }

    #[test]
    fn interpret_restricts_and_drops_conditions() {
        let facts = sample_facts();
        let pattern = GraphPattern::new(
            vec![EdgePattern::var_const("person", "获奖", "金棕榈")],
            vec!["person".to_string()],
        );
        let mut conversation =
            ConversationalGraph::new(pattern, &facts, MatchOptions::default()).unwrap();
        let lexicon = interpretation_lexicon();
        let narrowed = conversation
            .interpret("只看张三", &lexicon, &facts, MatchOptions::default())
            .unwrap()
            .unwrap();
        assert!(matches!(narrowed, GraphReply::Changed { total: 1, .. }));
        assert_eq!(conversation.pattern().constraints.len(), 1);
        let relaxed = conversation
            .interpret("去掉这个条件", &lexicon, &facts, MatchOptions::default())
            .unwrap()
            .unwrap();
        assert!(matches!(relaxed, GraphReply::Changed { total: 3, .. }));
        assert!(conversation.pattern().constraints.is_empty());
        assert!(matches!(
            conversation
                .interpret("去掉这个条件", &lexicon, &facts, MatchOptions::default())
                .unwrap(),
            Err(PatternError::CannotDropLastEdge)
        ));
        assert!(conversation
            .interpret("随便说点什么", &lexicon, &facts, MatchOptions::default())
            .is_none());
    }

    #[test]
    fn interpret_type_restriction_adds_type_edge() {
        let facts = sample_facts();
        let pattern = GraphPattern::new(
            vec![EdgePattern::var_var("film", "type", "kind")],
            vec!["film".to_string()],
        );
        let mut conversation =
            ConversationalGraph::new(pattern, &facts, MatchOptions::default()).unwrap();
        let lexicon = interpretation_lexicon();
        let narrowed = conversation
            .interpret("只看科幻片", &lexicon, &facts, MatchOptions::default())
            .unwrap()
            .unwrap();
        assert!(matches!(narrowed, GraphReply::Changed { total: 1, .. }));
        assert_eq!(conversation.pattern().edges.len(), 2);
        assert_eq!(conversation.pattern().edges[1].predicate, "type");
        assert!(matches!(
            conversation.apply(
                GraphCommand::DropLastConstraint,
                &facts,
                MatchOptions::default()
            ),
            Err(PatternError::CannotDropLastConstraint)
        ));
    }

    #[test]
    fn interpret_elliptical_followup_adds_restriction() {
        let facts = sample_facts();
        let pattern = GraphPattern::new(
            vec![
                EdgePattern::var_var("film", "type", "kind"),
                EdgePattern::var_var("director", "导演", "film"),
            ],
            vec!["film".to_string()],
        );
        let mut conversation =
            ConversationalGraph::new(pattern, &facts, MatchOptions::default()).unwrap();
        let lexicon = interpretation_lexicon();
        let narrowed = conversation
            .interpret("那科幻片呢", &lexicon, &facts, MatchOptions::default())
            .unwrap()
            .unwrap();
        assert!(matches!(narrowed, GraphReply::Changed { total: 1, .. }), "{:?}", narrowed);
        assert_eq!(conversation.pattern().edges.len(), 3);
        assert!(conversation
            .pattern()
            .edges
            .iter()
            .any(|edge| edge.predicate == "type" && edge.object == PatternTerm::constant("科幻片")));
    }

    #[test]
    fn interpret_unique_type_prefix_narrows_followup() {
        let mut facts = sample_facts();
        facts.push(GraphFact::type_assertion("张三", "中国人"));
        let mut lexicon = interpretation_lexicon();
        lexicon.add_type("中国人", "中国人");
        let pattern = GraphPattern::new(
            vec![EdgePattern::var_const("person", "获奖", "金棕榈")],
            vec!["person".to_string()],
        );
        let mut conversation =
            ConversationalGraph::new(pattern, &facts, MatchOptions::default()).unwrap();
        let narrowed = conversation
            .interpret("只看中国的", &lexicon, &facts, MatchOptions::default())
            .unwrap()
            .unwrap();
        assert!(matches!(narrowed, GraphReply::Changed { total: 1, .. }));
        assert!(conversation
            .pattern()
            .edges
            .iter()
            .any(|edge| edge.predicate == "type" && edge.object == PatternTerm::constant("中国人")));
    }

    #[test]
    fn interpret_exclusion_removes_entity_from_current_results() {
        let facts = sample_facts();
        let pattern = GraphPattern::new(
            vec![EdgePattern::var_const("person", "获奖", "金棕榈")],
            vec!["person".to_string()],
        );
        let mut conversation =
            ConversationalGraph::new(pattern, &facts, MatchOptions::default()).unwrap();
        let lexicon = interpretation_lexicon();
        let reduced = conversation
            .interpret("除了张三还有谁", &lexicon, &facts, MatchOptions::default())
            .unwrap()
            .unwrap();
        assert!(matches!(reduced, GraphReply::Changed { total: 2, .. }));
        assert_eq!(conversation.pattern().constraints.len(), 1);
        assert!(conversation
            .pattern()
            .constraints
            .contains(&PatternConstraint::NotEqual {
                variable: "person".to_string(),
                value: "张三".to_string(),
            }));
        let outcome = conversation.outcome();
        assert!(outcome
            .solutions
            .iter()
            .all(|solution| solution.binding.get("person").map(String::as_str) != Some("张三")));
        let restored = conversation
            .interpret("撤回", &lexicon, &facts, MatchOptions::default())
            .unwrap()
            .unwrap();
        assert!(matches!(restored, GraphReply::Changed { total: 3, .. }));
    }

    #[test]
    fn index_marks_constant_constant_lookup() {
        let facts = sample_facts();
        let pattern = GraphPattern::new(
            vec![EdgePattern::const_const("张三", "认识", "王五")],
            Vec::new(),
        );
        let index = GraphIndex::new(&facts);
        assert_eq!(index.len(), facts.len());
        let outcome = pattern.evaluate(&facts, MatchOptions::default()).unwrap();
        assert_eq!(outcome.len(), 1);
        assert!(pattern.exists(&facts, MatchOptions::default()));
    }
}

//! 迷你逻辑程序引擎：Datalog / Prolog 子集的纯符号实现。
//!
//! 能力：
//! - 事实 + Horn 规则（`,` 合取、`not` / `\+` 分层否定），无函数符号，
//!   Herbrand 基有限，求值必然终止；
//! - 分层否定（stratified negation）检查 + 自底向上不动点求值（semi-naive 风格逐层收敛）；
//! - 模式查询（返回变量绑定）与一阶推导树（解释结论从哪来）；
//! - Prolog 风格文本的解析与渲染，便于把「对话策略」等知识显式声明、检查与展示。
//!
//! 设计立场：不引入任何神经网络与权重，所有「思考」都是符号匹配与不动点推导。

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// 逻辑项：变量（大写或下划线开头）或常量原子（其余标识符 / 中文名）。
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum LTerm {
    Var(String),
    Atom(String),
}

impl LTerm {
    pub fn atom(name: impl Into<String>) -> Self {
        LTerm::Atom(name.into())
    }

    pub fn var(name: impl Into<String>) -> Self {
        LTerm::Var(name.into())
    }

    pub fn name(&self) -> &str {
        match self {
            LTerm::Var(name) | LTerm::Atom(name) => name,
        }
    }

    pub fn is_var(&self) -> bool {
        matches!(self, LTerm::Var(_))
    }

    #[allow(dead_code)]
    fn is_anonymous(&self) -> bool {
        matches!(self, LTerm::Var(name) if name.starts_with("_anon_"))
    }
}

/// 原子（谓词 + 项列表），如 `parent(张三, 李四)`。
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LAtom {
    pub predicate: String,
    pub terms: Vec<LTerm>,
}

impl LAtom {
    pub fn new(predicate: impl Into<String>, terms: Vec<LTerm>) -> Self {
        LAtom {
            predicate: predicate.into(),
            terms,
        }
    }

    pub fn is_ground(&self) -> bool {
        self.terms.iter().all(|term| !term.is_var())
    }

    pub fn render(&self) -> String {
        if self.terms.is_empty() {
            return self.predicate.clone();
        }
        let terms: Vec<&str> = self.terms.iter().map(LTerm::name).collect();
        format!("{}({})", self.predicate, terms.join(", "))
    }
}

/// 规则体文字：正文字或分层否定文字。
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum LLiteral {
    Pos(LAtom),
    Neg(LAtom),
}

/// Horn 规则：head :- body.
#[derive(Clone, Debug)]
pub struct LRule {
    pub head: LAtom,
    pub body: Vec<LLiteral>,
}

impl LRule {
    pub fn render(&self) -> String {
        if self.body.is_empty() {
            return format!("{}.", self.head.render());
        }
        let body: Vec<String> = self
            .body
            .iter()
            .map(|literal| match literal {
                LLiteral::Pos(atom) => atom.render(),
                LLiteral::Neg(atom) => format!("not {}", atom.render()),
            })
            .collect();
        format!("{} :- {}.", self.head.render(), body.join(", "))
    }
}

/// 解析错误（带可读信息）。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseError {
    pub message: String,
}

impl ParseError {
    pub fn new(message: impl Into<String>) -> Self {
        ParseError {
            message: message.into(),
        }
    }

    fn with_line(self, line: usize) -> Self {
        ParseError::new(format!("第 {} 行：{}", line, self.message))
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ParseError {}

/// 变量绑定环境。
pub type Env = BTreeMap<String, String>;

/// 推导节点：一个结论、触发它的规则与前提。
#[derive(Clone, Debug)]
pub struct ExplainNode {
    pub fact: LAtom,
    pub rule_index: Option<usize>,
    pub premises: Vec<ExplainNode>,
}

/// 逻辑程序：规则集合 + 事实集合 + 一阶推导记录。
#[derive(Clone, Debug, Default)]
pub struct LogicProgram {
    rules: Vec<LRule>,
    facts: BTreeSet<LAtom>,
    by_predicate: BTreeMap<String, BTreeSet<LAtom>>,
    derivations: BTreeMap<LAtom, usize>,
}

impl LogicProgram {
    pub fn new() -> Self {
        Self::default()
    }

    /// 从 Prolog 风格源码解析程序（每行一条事实或规则，`#`/`%` 起注释）。
    pub fn from_source(source: &str) -> Result<Self, ParseError> {
        let mut program = Self::new();
        for (index, raw_line) in source.lines().enumerate() {
            let line_no = index + 1;
            let line = raw_line
                .split('#')
                .next()
                .unwrap_or("")
                .split('%')
                .next()
                .unwrap_or("")
                .trim();
            if line.is_empty() {
                continue;
            }
            if line.contains(":-") {
                let rule = parse_rule(line).map_err(|error| error.with_line(line_no))?;
                program
                    .add_rule(rule)
                    .map_err(|error| error.with_line(line_no))?;
            } else {
                let fact = parse_atom(line).map_err(|error| error.with_line(line_no))?;
                program.add_fact(fact);
            }
        }
        Ok(program)
    }

    /// 添加一条事实（不触发求值）。
    pub fn add_fact(&mut self, fact: LAtom) {
        self.insert_fact(fact);
    }

    /// 以文本形式添加一条事实，如 `kind(t1, ask)`。
    pub fn add_fact_str(&mut self, text: &str) -> Result<LAtom, ParseError> {
        let fact = parse_atom(text)?;
        self.insert_fact(fact.clone());
        Ok(fact)
    }

    /// 添加一条规则：检查安全性（头部变量必须出现在正文字中；
    /// 否定文字的变量必须由之前的正文字绑定）。
    pub fn add_rule(&mut self, rule: LRule) -> Result<(), ParseError> {
        let mut bound: BTreeSet<String> = BTreeSet::new();
        let mut positive_vars: BTreeSet<String> = BTreeSet::new();
        for literal in &rule.body {
            match literal {
                LLiteral::Pos(atom) => {
                    for term in &atom.terms {
                        if let LTerm::Var(name) = term {
                            bound.insert(name.clone());
                            positive_vars.insert(name.clone());
                        }
                    }
                }
                LLiteral::Neg(atom) => {
                    for term in &atom.terms {
                        if let LTerm::Var(name) = term {
                            if !bound.contains(name) {
                                return Err(ParseError::new(format!(
                                    "否定文字中的变量 {} 未经正文字绑定（规则不安全）",
                                    name
                                )));
                            }
                        }
                    }
                }
            }
        }
        for term in &rule.head.terms {
            if let LTerm::Var(name) = term {
                if !positive_vars.contains(name) {
                    return Err(ParseError::new(format!(
                        "规则头部变量 {} 未出现在正文字中（规则不安全）",
                        name
                    )));
                }
            }
        }
        self.rules.push(rule);
        Ok(())
    }

    /// 清空事实（保留规则），供逐轮重建 ABox 使用。
    pub fn clear_facts(&mut self) {
        self.facts.clear();
        self.by_predicate.clear();
        self.derivations.clear();
    }

    pub fn contains(&self, fact: &LAtom) -> bool {
        self.facts.contains(fact)
    }

    pub fn fact_count(&self) -> usize {
        self.facts.len()
    }

    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    pub fn rules(&self) -> &[LRule] {
        &self.rules
    }

    pub fn facts(&self) -> impl Iterator<Item = &LAtom> {
        self.facts.iter()
    }

    pub fn derived(&self, fact: &LAtom) -> bool {
        self.derivations.contains_key(fact)
    }

    /// 分层检查：否定不能出现在同一依赖强连通分量内（负环无法给出良定义语义）。
    fn compute_strata(&self) -> Result<BTreeMap<String, usize>, String> {
        let mut stratum: BTreeMap<String, usize> = BTreeMap::new();
        for rule in &self.rules {
            stratum.entry(rule.head.predicate.clone()).or_insert(0);
            for literal in &rule.body {
                let atom = match literal {
                    LLiteral::Pos(atom) | LLiteral::Neg(atom) => atom,
                };
                stratum.entry(atom.predicate.clone()).or_insert(0);
            }
        }
        let limit = stratum.len() + 2;
        for _ in 0..limit {
            let mut changed = false;
            for rule in &self.rules {
                let mut required = 0usize;
                for literal in &rule.body {
                    match literal {
                        LLiteral::Pos(atom) => {
                            required = required.max(*stratum.get(&atom.predicate).unwrap_or(&0));
                        }
                        LLiteral::Neg(atom) => {
                            required =
                                required.max(stratum.get(&atom.predicate).copied().unwrap_or(0) + 1);
                        }
                    }
                }
                let head = stratum.entry(rule.head.predicate.clone()).or_insert(0);
                if *head < required {
                    *head = required;
                    changed = true;
                }
            }
            if !changed {
                return Ok(stratum);
            }
        }
        Err("规则程序不是分层的：否定出现在递归依赖环中，无法求值".to_string())
    }

    /// 自底向上不动点求值（按分层从低到高，逐层收敛）。
    pub fn evaluate(&mut self) -> Result<(), String> {
        let strata = self.compute_strata()?;
        let mut rules_by_stratum: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (index, rule) in self.rules.iter().enumerate() {
            let level = strata.get(&rule.head.predicate).copied().unwrap_or(0);
            rules_by_stratum.entry(level).or_default().push(index);
        }
        for (_level, rule_indices) in rules_by_stratum {
            loop {
                let mut derived: Vec<(LAtom, usize)> = Vec::new();
                for &rule_index in &rule_indices {
                    let rule = self.rules[rule_index].clone();
                    for env in self.match_body(&rule.body, vec![Env::new()]) {
                        let head = instantiate_atom(&rule.head, &env);
                        if !self.facts.contains(&head)
                            && !derived.iter().any(|(atom, _)| atom == &head)
                        {
                            derived.push((head, rule_index));
                        }
                    }
                }
                if derived.is_empty() {
                    break;
                }
                for (fact, rule_index) in derived {
                    self.insert_derived(fact, rule_index);
                }
            }
        }
        Ok(())
    }

    /// 模式查询：`respond(X)`，返回全部变量绑定（匿名变量不出现在结果中）。
    pub fn query(&self, pattern: &str) -> Result<Vec<Env>, ParseError> {
        let atom = parse_atom(pattern)?;
        let mut results: Vec<Env> = Vec::new();
        if let Some(facts) = self.by_predicate.get(&atom.predicate) {
            for fact in facts {
                if let Some(env) = unify_atom(&atom, fact, &Env::new()) {
                    let clean: Env = env
                        .into_iter()
                        .filter(|(name, _)| !name.starts_with("_anon_"))
                        .collect();
                    if !results.contains(&clean) {
                        results.push(clean);
                    }
                }
            }
        }
        Ok(results)
    }

    pub fn query_atom(&self, atom: &LAtom) -> Vec<Env> {
        match self.query(&atom.render()) {
            Ok(envs) => envs,
            Err(_) => Vec::new(),
        }
    }

    /// 一阶推导树：结论由哪条规则、哪些前提推出（深度受限）。
    pub fn explain(&self, atom: &LAtom, depth: usize) -> Option<ExplainNode> {
        if depth == 0 {
            return None;
        }
        let rule_index = *self.derivations.get(atom)?;
        let rule = &self.rules[rule_index];
        let head_env = unify_atom(&rule.head, atom, &Env::new())?;
        let env = self
            .match_body(&rule.body, vec![head_env])
            .into_iter()
            .next()?;
        let mut premises = Vec::new();
        for literal in &rule.body {
            if let LLiteral::Pos(premise) = literal {
                let ground = instantiate_atom(premise, &env);
                let node = self.explain(&ground, depth - 1).unwrap_or(ExplainNode {
                    fact: ground,
                    rule_index: None,
                    premises: Vec::new(),
                });
                premises.push(node);
            }
        }
        Some(ExplainNode {
            fact: atom.clone(),
            rule_index: Some(rule_index),
            premises,
        })
    }

    /// 渲染推导树为多行文本（`⟸ 规则 #n`）。
    pub fn explain_lines(&self, atom: &LAtom, depth: usize) -> Vec<String> {
        let mut lines = Vec::new();
        if let Some(node) = self.explain(atom, depth) {
            self.render_explain_node(&node, 0, &mut lines);
        }
        lines
    }

    fn render_explain_node(&self, node: &ExplainNode, indent: usize, out: &mut Vec<String>) {
        let prefix = "  ".repeat(indent);
        match node.rule_index {
            Some(index) => {
                out.push(format!(
                    "{}{}  ⟸ 规则 #{}：{}",
                    prefix,
                    node.fact.render(),
                    index + 1,
                    self.rules[index].render()
                ));
            }
            None => out.push(format!("{}{}  （既有事实）", prefix, node.fact.render())),
        }
        for premise in &node.premises {
            self.render_explain_node(premise, indent + 1, out);
        }
    }

    /// 渲染为可再次解析的 Prolog 风格源码。
    pub fn to_prolog_source(&self) -> String {
        let mut out = String::new();
        for fact in &self.facts {
            if !self.derivations.contains_key(fact) {
                out.push_str(&format!("{}.\n", fact.render()));
            }
        }
        for (fact, rule_index) in &self.derivations {
            out.push_str(&format!(
                "{}  # 由规则 #{} 推出：{}\n",
                fact.render(),
                rule_index + 1,
                self.rules[*rule_index].render()
            ));
        }
        for rule in &self.rules {
            out.push_str(&rule.render());
            out.push('\n');
        }
        out
    }

    fn insert_fact(&mut self, fact: LAtom) {
        if self.facts.insert(fact.clone()) {
            self.by_predicate
                .entry(fact.predicate.clone())
                .or_default()
                .insert(fact);
        }
    }

    fn insert_derived(&mut self, fact: LAtom, rule_index: usize) {
        if self.facts.insert(fact.clone()) {
            self.by_predicate
                .entry(fact.predicate.clone())
                .or_default()
                .insert(fact.clone());
            self.derivations.insert(fact, rule_index);
        }
    }

    /// 从给定初始环境集合出发逐文字展开（正文字做连接，否定文字做失败即否定）。
    fn match_body(&self, body: &[LLiteral], start: Vec<Env>) -> Vec<Env> {
        let mut envs = start;
        for literal in body {
            match literal {
                LLiteral::Pos(atom) => {
                    let mut next = Vec::new();
                    if let Some(facts) = self.by_predicate.get(&atom.predicate) {
                        for env in &envs {
                            for fact in facts {
                                if let Some(extended) = unify_atom(atom, fact, env) {
                                    next.push(extended);
                                }
                            }
                        }
                    }
                    envs = next;
                }
                LLiteral::Neg(atom) => {
                    envs.retain(|env| !self.facts.contains(&instantiate_atom(atom, env)));
                }
            }
            if envs.is_empty() {
                break;
            }
        }
        envs
    }
}

fn instantiate_atom(atom: &LAtom, env: &Env) -> LAtom {
    LAtom {
        predicate: atom.predicate.clone(),
        terms: atom
            .terms
            .iter()
            .map(|term| match term {
                LTerm::Atom(name) => LTerm::Atom(name.clone()),
                LTerm::Var(name) => match env.get(name) {
                    Some(value) => LTerm::Atom(value.clone()),
                    None => LTerm::Var(name.clone()),
                },
            })
            .collect(),
    }
}

fn unify_atom(pattern: &LAtom, fact: &LAtom, env: &Env) -> Option<Env> {
    if pattern.predicate != fact.predicate || pattern.terms.len() != fact.terms.len() {
        return None;
    }
    let mut out = env.clone();
    for (pattern_term, fact_term) in pattern.terms.iter().zip(fact.terms.iter()) {
        let value = fact_term.name();
        match pattern_term {
            LTerm::Atom(name) => {
                if name != value {
                    return None;
                }
            }
            LTerm::Var(name) => match out.get(name) {
                Some(existing) => {
                    if existing != value {
                        return None;
                    }
                }
                None => {
                    out.insert(name.clone(), value.to_string());
                }
            },
        }
    }
    Some(out)
}

fn split_top_level(text: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    for (index, ch) in text.char_indices() {
        match ch {
            '(' | '（' => depth += 1,
            ')' | '）' => depth = depth.saturating_sub(1),
            ',' | '，' if depth == 0 => {
                parts.push(&text[start..index]);
                start = index + ch.len_utf8();
            }
            _ => {}
        }
    }
    parts.push(&text[start..]);
    parts
}

fn validate_ident(name: &str) -> Result<(), ParseError> {
    if name.is_empty() {
        return Err(ParseError::new("遇到空的谓词或项名"));
    }
    let bad = name
        .chars()
        .any(|ch| ch.is_whitespace() || "()（）,，:：.'\"".contains(ch));
    if bad {
        return Err(ParseError::new(format!("非法标识符「{}」", name)));
    }
    Ok(())
}

/// 解析原子，如 `plan(direct_answer, T)`；`_` 为匿名变量（每次出现自动改名）。
pub fn parse_atom(text: &str) -> Result<LAtom, ParseError> {
    let text = text.trim().trim_end_matches('.').trim();
    let open = text
        .find(|ch| ch == '(' || ch == '（')
        .ok_or_else(|| ParseError::new(format!("原子缺少左括号：{}", text)))?;
    let close = text
        .rfind(|ch| ch == ')' || ch == '）')
        .ok_or_else(|| ParseError::new(format!("原子缺少右括号：{}", text)))?;
    if close < open {
        return Err(ParseError::new(format!("括号顺序不合法：{}", text)));
    }
    let predicate = text[..open].trim();
    validate_ident(predicate)?;
    let inner = &text[open + 1..close];
    let rest = text[close + 1..].trim();
    if !rest.is_empty() {
        return Err(ParseError::new(format!("右括号后有多余内容：{}", rest)));
    }
    let mut terms = Vec::new();
    let mut anonymous = 0usize;
    if !inner.trim().is_empty() {
        for raw in split_top_level(inner) {
            let raw = raw.trim();
            if raw.is_empty() {
                return Err(ParseError::new(format!("原子 {} 中存在空项", predicate)));
            }
            if raw == "_" {
                terms.push(LTerm::Var(format!("_anon_{}", anonymous)));
                anonymous += 1;
                continue;
            }
            validate_ident(raw)?;
            let first = raw.chars().next().unwrap_or('_');
            if first.is_uppercase() || first == '_' {
                terms.push(LTerm::Var(raw.to_string()));
            } else {
                terms.push(LTerm::Atom(raw.to_string()));
            }
        }
    }
    Ok(LAtom {
        predicate: predicate.to_string(),
        terms,
    })
}

/// 解析一条规则（无 `:-` 时视为空体规则，由调用方决定当作事实处理）。
pub fn parse_rule(text: &str) -> Result<LRule, ParseError> {
    let text = text.trim().trim_end_matches('.').trim();
    if let Some((head, body)) = text.split_once(":-") {
        let head = parse_atom(head)?;
        let mut literals = Vec::new();
        for raw in split_top_level(body) {
            if raw.trim().is_empty() {
                continue;
            }
            literals.push(parse_literal(raw)?);
        }
        Ok(LRule { head, body: literals })
    } else {
        Ok(LRule {
            head: parse_atom(text)?,
            body: Vec::new(),
        })
    }
}

fn parse_literal(text: &str) -> Result<LLiteral, ParseError> {
    let text = text.trim();
    if let Some(rest) = text.strip_prefix("not ") {
        return Ok(LLiteral::Neg(parse_atom(rest)?));
    }
    if let Some(rest) = text.strip_prefix("\\+") {
        return Ok(LLiteral::Neg(parse_atom(rest)?));
    }
    Ok(LLiteral::Pos(parse_atom(text)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn program_from(source: &str) -> LogicProgram {
        let mut program = LogicProgram::from_source(source).expect("程序应可解析");
        program.evaluate().expect("程序应可求值");
        program
    }

    #[test]
    fn test_parse_and_join() {
        let program = program_from(
            r#"
            parent(张三, 李四).
            parent(李四, 王五).
            ancestor(X, Y) :- parent(X, Y).
            ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).
            "#,
        );
        let mut results = program.query("ancestor(张三, X)").expect("查询应可解析");
        results.sort();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].get("X").map(String::as_str), Some("李四"));
        assert_eq!(results[1].get("X").map(String::as_str), Some("王五"));
    }

    #[test]
    fn test_negation_as_failure() {
        let program = program_from(
            r#"
            candidate(a).
            candidate(b).
            blocked(b).
            available(X) :- candidate(X), not blocked(X).
            "#,
        );
        let results = program.query("available(X)").expect("查询应可解析");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].get("X").map(String::as_str), Some("a"));
    }

    #[test]
    fn test_anonymous_variable_dedup() {
        let program = program_from(
            r#"
            likes(a, b).
            likes(a, c).
            personalized(X) :- likes(X, _).
            "#,
        );
        let results = program.query("personalized(X)").expect("查询应可解析");
        assert_eq!(results.len(), 1);
        assert!(!results[0].contains_key("_anon_0"));
    }

    #[test]
    fn test_unsafe_rule_rejected() {
        let mut program = LogicProgram::new();
        let rule = parse_rule("p(X) :- q(Y).").expect("应可解析");
        assert!(program.add_rule(rule).is_err());
    }

    #[test]
    fn test_negative_cycle_rejected() {
        let error = LogicProgram::from_source(
            r#"
            q(a).
            p(X) :- q(X), not p(X).
            "#,
        )
        .expect("应可解析")
        .evaluate()
        .expect_err("负环应被拒绝");
        assert!(error.contains("分层"), "实际：{}", error);
    }

    #[test]
    fn test_explain_trace() {
        let program = program_from(
            r#"
            parent(张三, 李四).
            parent(李四, 王五).
            ancestor(X, Y) :- parent(X, Y).
            ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).
            "#,
        );
        let atom = parse_atom("ancestor(张三, 王五)").expect("应可解析");
        let lines = program.explain_lines(&atom, 6);
        assert!(!lines.is_empty());
        assert!(
            lines.iter().any(|line| line.contains("规则 #")),
            "推导树应引用规则：{:?}",
            lines
        );
    }

    #[test]
    fn test_prolog_source_roundtrip() {
        let program = program_from(
            r#"
            candidate(a).
            blocked(b).
            candidate(b).
            available(X) :- candidate(X), not blocked(X).
            "#,
        );
        let source = program.to_prolog_source();
        let reparsed = program_from(&source);
        let results = reparsed.query("available(X)").expect("查询应可解析");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_clear_facts_keeps_rules() {
        let mut program = LogicProgram::new();
        program
            .add_rule(parse_rule("p(X) :- q(X).").expect("应可解析"))
            .expect("规则安全");
        program.add_fact_str("q(a)").expect("事实应可解析");
        program.clear_facts();
        assert_eq!(program.fact_count(), 0);
        assert_eq!(program.rule_count(), 1);
    }
}

// 一阶归结推理引擎（纯符号，无神经网络、无权重训练）
//
// 灵感：Robinson 归结原理。知识图谱三元组被编码为基原子（ground atom），
// 链接预测挖掘出的 Horn 规则与关联规则成为子句；问答被转成目标子句后：
//   • 反向（SLD 反驳）：逐目标与事实/规则头做合一，成功即给出变量绑定与证明树；
//   • 正向（半朴素饱和）：从事实出发反复应用规则，推导多跳新事实并保留前提来源。
//
// 全程只有：项、置换、合一、深度上限与搜索预算——每一步都可解释、可回溯。

use std::collections::{HashMap, HashSet};
use std::rc::Rc;

/// 项：逻辑变量或常量（实体名、属性值、谓词名都是常量）
///
/// 字符串用 `Rc<str>` 共享。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Term {
    Var(Rc<str>),
    Const(Rc<str>),
}

impl Term {
    pub fn var(name: &str) -> Self {
        Term::Var(Rc::from(name))
    }

    pub fn constant(name: &str) -> Self {
        Term::Const(Rc::from(name))
    }

    pub fn is_var(&self) -> bool {
        matches!(self, Term::Var(_))
    }

    pub fn name(&self) -> &str {
        match self {
            Term::Var(name) | Term::Const(name) => name,
        }
    }
}

/// 原子：谓词 + 参数（知识图谱二元关系；类型/属性存在性用一元或二元谓词编码）
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Atom {
    pub predicate: Rc<str>,
    pub terms: Vec<Term>,
}

impl Atom {
    pub fn new(predicate: &str, terms: Vec<Term>) -> Self {
        Self {
            predicate: Rc::from(predicate),
            terms,
        }
    }

    pub fn unary(predicate: &str, subject: Term) -> Self {
        Self::new(predicate, vec![subject])
    }

    pub fn pattern(predicate: &str, subject: Term, object: Term) -> Self {
        Self::new(predicate, vec![subject, object])
    }

    /// 基原子（全是常量）
    pub fn rel(predicate: &str, subject: &str, object: &str) -> Self {
        Self::pattern(
            predicate,
            Term::constant(subject),
            Term::constant(object),
        )
    }

    pub fn arity(&self) -> usize {
        self.terms.len()
    }

    pub fn is_ground(&self) -> bool {
        self.terms.iter().all(|term| !term.is_var())
    }

    /// 内部调试表示，如 `师傅(张三, ?X)`
    pub fn display(&self) -> String {
        let mut out = String::with_capacity(
            self.predicate.len()
                + 2
                + self
                    .terms
                    .iter()
                    .map(|term| term.name().len() + 3)
                    .sum::<usize>(),
        );
        out.push_str(&self.predicate);
        out.push('(');
        for (index, term) in self.terms.iter().enumerate() {
            if index > 0 {
                out.push_str(", ");
            }
            match term {
                Term::Var(name) => {
                    out.push('?');
                    out.push_str(name);
                }
                Term::Const(name) => out.push_str(name),
            }
        }
        out.push(')');
        out
    }
}

/// 置换：变量名 → 项
///
/// 用不可变单链表 + `Rc` 持久化表示置换：
/// 合一产生的扩展在常数时间完成。
#[derive(Debug, Clone, Default)]
pub struct Substitution {
    head: Option<Rc<SubstitutionBinding>>,
}

#[derive(Debug)]
struct SubstitutionBinding {
    name: Rc<str>,
    term: Term,
    next: Option<Rc<SubstitutionBinding>>,
}

impl Substitution {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, name: &str) -> Option<&Term> {
        let mut current = self.head.as_deref();
        while let Some(binding) = current {
            if &*binding.name == name {
                return Some(&binding.term);
            }
            current = binding.next.as_deref();
        }
        None
    }

    /// 与 [`Self::get`] 相同，但对同一个 `Rc<str>` 走指针比较快速路径
    fn get_rc(&self, name: &Rc<str>) -> Option<&Term> {
        let mut current = self.head.as_deref();
        while let Some(binding) = current {
            if Rc::ptr_eq(&binding.name, name) || &*binding.name == &**name {
                return Some(&binding.term);
            }
            current = binding.next.as_deref();
        }
        None
    }

    pub fn insert(&mut self, name: impl Into<Rc<str>>, term: Term) {
        self.head = Some(Rc::new(SubstitutionBinding {
            name: name.into(),
            term,
            next: self.head.take(),
        }));
    }
}

/// 证明节点来源
#[derive(Debug, Clone, PartialEq)]
pub enum ProofSource {
    Fact,
    Rule { source: String },
}

/// 证明树节点：原子 + 本步来源 + 本步置信度（事实 1.0；规则=规则置信度×子证明置信度）
#[derive(Debug, Clone)]
pub struct ProofNode {
    pub atom: Atom,
    pub source: ProofSource,
    pub confidence: f64,
    pub children: Vec<ProofNode>,
}

impl ProofNode {
    fn fact(atom: Atom) -> Self {
        Self {
            atom,
            source: ProofSource::Fact,
            confidence: 1.0,
            children: Vec::new(),
        }
    }

    fn is_fact(&self) -> bool {
        matches!(self.source, ProofSource::Fact)
    }
}

/// 归结问题的解：变量绑定 + 每个目标的证明 + 总置信度
#[derive(Debug, Clone)]
pub struct Solution {
    pub bindings: HashMap<String, String>,
    pub proofs: Vec<ProofNode>,
    pub confidence: f64,
}

/// Horn 子句：head :- body（body 为空即事实；置信度用于排序/展示，不参与学习）
#[derive(Debug, Clone)]
pub struct Clause {
    pub head: Atom,
    pub body: Vec<Atom>,
    pub confidence: f64,
    pub source: String,
}

impl Clause {
    pub fn new(head: Atom, body: Vec<Atom>, confidence: f64, source: &str) -> Self {
        Self {
            head,
            body,
            confidence: confidence.clamp(0.0, 1.0),
            source: source.to_string(),
        }
    }
}

/// 前向饱和推导出的新事实（含前提与来源，可解释）
#[derive(Debug, Clone)]
pub struct DerivedFact {
    pub atom: Atom,
    pub rule_source: String,
    pub premises: Vec<Atom>,
    pub confidence: f64,
    /// 第几轮推导（1 = 直接应用规则；≥2 = 用到了已推导事实的多跳）
    pub depth: usize,
}

#[derive(Clone)]
pub struct ResolutionEngine {
    facts: HashSet<Atom>,
    facts_by_pred: HashMap<Rc<str>, Vec<Atom>>,
    clauses: Vec<Clause>,
    rules_by_head: HashMap<Rc<str>, Vec<usize>>,
    max_depth: usize,
    max_solutions: usize,
    max_steps: usize,
}

impl ResolutionEngine {
    pub fn new(max_depth: usize, max_solutions: usize) -> Self {
        Self {
            facts: HashSet::new(),
            facts_by_pred: HashMap::new(),
            clauses: Vec::new(),
            rules_by_head: HashMap::new(),
            max_depth: max_depth.max(1),
            max_solutions: max_solutions.clamp(1, 200),
            max_steps: 60_000,
        }
    }

    pub fn max_depth(&self) -> usize {
        self.max_depth
    }

    pub fn fact_count(&self) -> usize {
        self.facts.len()
    }

    pub fn clause_count(&self) -> usize {
        self.clauses.len()
    }

    /// 某个谓词下的全部基事实（供溯因/假设生成做失败前沿分析）
    pub fn facts_for(&self, predicate: &str) -> &[Atom] {
        self.facts_by_pred
            .get(predicate)
            .map(|atoms| atoms.as_slice())
            .unwrap_or(&[])
    }

    /// 头谓词匹配的子句（规则），供后向链展开
    pub fn clauses_for(&self, predicate: &str) -> Vec<&Clause> {
        self.clauses_iter_for(predicate).collect()
    }

    /// 头谓词匹配的子句迭代器
    pub(crate) fn clauses_iter_for(&self, predicate: &str) -> impl Iterator<Item = &Clause> {
        self.rules_by_head
            .get(predicate)
            .into_iter()
            .flatten()
            .map(move |&index| &self.clauses[index])
    }

    /// 是否为已知基事实
    pub fn has_fact(&self, atom: &Atom) -> bool {
        self.facts.contains(atom)
    }

    /// 全部基事实（供反事实模拟做新旧对比）
    pub fn all_facts(&self) -> impl Iterator<Item = &Atom> {
        self.facts.iter()
    }

    /// 加入基事实；已存在返回 false
    pub fn add_fact_atom(&mut self, atom: Atom) -> bool {
        if !atom.is_ground() {
            return false;
        }
        if !self.facts.insert(atom.clone()) {
            return false;
        }
        self.facts_by_pred
            .entry(atom.predicate.clone())
            .or_default()
            .push(atom);
        true
    }

    /// 加入二元基事实 `predicate(subject, object)`
    pub fn add_fact(&mut self, predicate: &str, subject: &str, object: &str) -> bool {
        self.add_fact_atom(Atom::rel(predicate, subject, object))
    }

    /// 移除基事实（`add_fact_atom` 的逆操作，按原顺序压缩索引，保证迭代顺序可复原）；
    /// 事实不存在返回 false
    pub fn remove_fact_atom(&mut self, atom: &Atom) -> bool {
        if !self.facts.remove(atom) {
            return false;
        }
        if let Some(facts) = self.facts_by_pred.get_mut(&atom.predicate) {
            if let Some(position) = facts.iter().position(|fact| fact == atom) {
                facts.remove(position);
            }
        }
        true
    }

    /// 加入一元基事实 `predicate(subject)`
    pub fn add_unary_fact(&mut self, predicate: &str, subject: &str) -> bool {
        self.add_fact_atom(Atom::unary(predicate, Term::constant(subject)))
    }

    pub fn add_clause(&mut self, mut clause: Clause) {
        canonicalize_clause_vars(&mut clause);
        let index = self.clauses.len();
        self.rules_by_head
            .entry(clause.head.predicate.clone())
            .or_default()
            .push(index);
        self.clauses.push(clause);
    }

    /// 反向 SLD 反驳：求解 goals 的合取，返回变量绑定与证明树
    //
    // 搜索预算按"每个合取目标"独立分配：规则探索耗尽预算时，事实匹配
    // （单位归结）仍会完成，避免前一个目标的规则分支拖垮后续目标。
    pub fn solve(&self, goals: &[Atom], query_vars: &[String]) -> Vec<Solution> {
        if goals.is_empty() {
            return Vec::new();
        }
        let mut counter = 0usize;
        let mut states: Vec<(Substitution, Vec<ProofNode>)> =
            vec![(Substitution::new(), Vec::new())];
        let per_goal_budget = (self.max_steps / goals.len().max(1)).max(500);

        for goal in goals {
            let mut budget = per_goal_budget;
            let mut next: Vec<(Substitution, Vec<ProofNode>)> = Vec::new();
            'states: for (subst, proofs) in &states {
                for (solved, node) in self.prove_goal(goal, 0, subst, &mut counter, &mut budget) {
                    let mut chain = proofs.clone();
                    chain.push(node);
                    next.push((solved, chain));
                    if next.len() >= self.max_solutions.saturating_mul(4) {
                        break 'states;
                    }
                }
                if budget == 0 {
                    break;
                }
            }
            states = next;
            if states.is_empty() {
                break;
            }
        }

        // 同一组绑定可能经不同路径到达：保留置信度最高的一条证明
        let mut best: HashMap<String, Solution> = HashMap::new();
        for (subst, proofs) in states {
            // 渲染变量值：同时用于绑定表与去重键
            let mut bindings: HashMap<String, String> = HashMap::with_capacity(query_vars.len());
            let mut key = String::new();
            for var in query_vars {
                let value = resolve_term(&Term::var(var), &subst);
                let rendered = render_term(&value);
                if !key.is_empty() {
                    key.push('|');
                }
                key.push_str(var);
                key.push('=');
                key.push_str(&rendered);
                bindings.insert(var.clone(), rendered);
            }
            let confidence: f64 = proofs.iter().map(|node| node.confidence).product();
            let candidate = Solution {
                bindings,
                proofs,
                confidence: confidence.clamp(0.0, 1.0),
            };
            match best.get(&key) {
                Some(existing) if existing.confidence >= candidate.confidence => {}
                _ => {
                    best.insert(key, candidate);
                }
            }
        }
        let mut solutions: Vec<Solution> = best.into_values().collect();
        solutions.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        solutions.truncate(self.max_solutions);
        solutions
    }

    fn prove_goal(
        &self,
        goal: &Atom,
        depth: usize,
        subst: &Substitution,
        counter: &mut usize,
        budget: &mut usize,
    ) -> Vec<(Substitution, ProofNode)> {
        let mut results: Vec<(Substitution, ProofNode)> = Vec::new();
        if depth > self.max_depth {
            return results;
        }
        let applied = apply_atom(subst, goal);

        // 1) 单位归结：目标与基事实合一（代价 O(同谓词事实数)，不消耗搜索预算）
        if let Some(facts) = self.facts_by_pred.get(&applied.predicate) {
            for fact in facts {
                if fact.arity() != applied.arity() {
                    continue;
                }
                let mut unified = subst.clone();
                if unify_prepared(&applied, fact, &mut unified) {
                    let ground = apply_atom(&unified, &applied);
                    results.push((unified, ProofNode::fact(ground)));
                    if results.len() >= self.max_solutions {
                        return results;
                    }
                }
            }
        }

        // 2) 与规则头归结：先对子句头做变量换名并合一，成功后再换名规则体
        if let Some(indices) = self.rules_by_head.get(&applied.predicate) {
            for &idx in indices {
                if *budget == 0 {
                    break;
                }
                let clause = &self.clauses[idx];
                let tag = *counter;
                *counter += 1;
                let mut renamer = Renamer::new(tag);
                let fresh_head = renamer.atom(&clause.head);
                let mut unified = subst.clone();
                *budget -= 1;
                if !unify_prepared(&applied, &fresh_head, &mut unified) {
                    continue;
                }
                // 头合一失败时无需换名规则体
                let fresh_body: Vec<Atom> = clause
                    .body
                    .iter()
                    .map(|atom| renamer.atom(atom))
                    .collect();
                for (solved, children) in
                    self.prove_body(&fresh_body, depth + 1, &unified, counter, budget)
                {
                    let ground = apply_atom(&solved, &applied);
                    let child_confidence: f64 =
                        children.iter().map(|child| child.confidence).product();
                    let confidence = (clause.confidence * child_confidence).clamp(0.0, 1.0);
                    results.push((
                        solved,
                        ProofNode {
                            atom: ground,
                            source: ProofSource::Rule {
                                source: clause.source.clone(),
                            },
                            confidence,
                            children,
                        },
                    ));
                    if results.len() >= self.max_solutions {
                        return results;
                    }
                }
            }
        }
        results
    }

    fn prove_body(
        &self,
        goals: &[Atom],
        depth: usize,
        subst: &Substitution,
        counter: &mut usize,
        budget: &mut usize,
    ) -> Vec<(Substitution, Vec<ProofNode>)> {
        if goals.is_empty() {
            return vec![(subst.clone(), Vec::new())];
        }
        let mut out: Vec<(Substitution, Vec<ProofNode>)> = Vec::new();
        for (solved, node) in self.prove_goal(&goals[0], depth, subst, counter, budget) {
            for (final_subst, mut rest) in
                self.prove_body(&goals[1..], depth, &solved, counter, budget)
            {
                let mut proofs = Vec::with_capacity(rest.len() + 1);
                proofs.push(node.clone());
                proofs.append(&mut rest);
                out.push((final_subst, proofs));
                if out.len() >= self.max_solutions {
                    return out;
                }
            }
            if *budget == 0 {
                break;
            }
        }
        out
    }

    /// 前向半朴素饱和：从事实出发反复应用子句，推导新基事实（含前提来源）。
    /// 事实直接引用 `self.facts`/`self.facts_by_pred`，只维护新推出的事实。
    pub fn forward_chain(&self, max_rounds: usize, max_new_facts: usize) -> Vec<DerivedFact> {
        self.forward_chain_with_extra_facts(&[], max_rounds, max_new_facts)
    }

    /// 与 `forward_chain` 相同，但把 `extra_facts` 当作额外基事实参与推导
    /// （用于反事实模拟：不修改自身即可注入假设）
    pub(crate) fn forward_chain_with_extra_facts(
        &self,
        extra_facts: &[Atom],
        max_rounds: usize,
        max_new_facts: usize,
    ) -> Vec<DerivedFact> {
        let mut derived_known: HashSet<Atom> = HashSet::new();
        let mut derived_by_pred: HashMap<Rc<str>, Vec<Atom>> = HashMap::new();
        for atom in extra_facts {
            if !atom.is_ground() || self.facts.contains(atom) {
                continue;
            }
            if derived_known.insert(atom.clone()) {
                derived_by_pred
                    .entry(atom.predicate.clone())
                    .or_default()
                    .push(atom.clone());
            }
        }
        // 基事实未记录置信度时按 1.0 处理
        let mut confidence_of: HashMap<Atom, f64> = HashMap::new();
        let mut derived: Vec<DerivedFact> = Vec::new();
        let mut counter = 0usize;

        for round in 1..=max_rounds.max(1) {
            let mut new_this_round: Vec<DerivedFact> = Vec::new();
            let mut new_this_round_atoms: HashSet<Atom> = HashSet::new();
            for clause in &self.clauses {
                let tag = counter;
                counter += 1;
                let fresh = freshen_clause(clause, tag);
                for (subst, premises) in self.match_body(&fresh.body, &derived_by_pred) {
                    let head = apply_atom(&subst, &fresh.head);
                    if !head.is_ground()
                        || self.facts.contains(&head)
                        || derived_known.contains(&head)
                    {
                        continue;
                    }
                    if !new_this_round_atoms.insert(head.clone()) {
                        continue;
                    }
                    let premise_confidence = premises
                        .iter()
                        .map(|premise| confidence_of.get(premise).copied().unwrap_or(1.0))
                        .fold(1.0f64, f64::min);
                    new_this_round.push(DerivedFact {
                        atom: head,
                        rule_source: clause.source.clone(),
                        premises,
                        confidence: (clause.confidence * premise_confidence).clamp(0.0, 1.0),
                        depth: round,
                    });
                    if derived.len() + new_this_round.len() >= max_new_facts {
                        break;
                    }
                }
                if derived.len() + new_this_round.len() >= max_new_facts {
                    break;
                }
            }
            for item in &new_this_round {
                derived_known.insert(item.atom.clone());
                derived_by_pred
                    .entry(item.atom.predicate.clone())
                    .or_default()
                    .push(item.atom.clone());
                confidence_of.insert(item.atom.clone(), item.confidence);
            }
            derived.append(&mut new_this_round);
            if new_this_round_atoms.is_empty() || derived.len() >= max_new_facts {
                break;
            }
        }
        derived
    }

    fn match_body(
        &self,
        goals: &[Atom],
        derived_by_pred: &HashMap<Rc<str>, Vec<Atom>>,
    ) -> Vec<(Substitution, Vec<Atom>)> {
        let mut states: Vec<(Substitution, Vec<Atom>)> = vec![(Substitution::new(), Vec::new())];
        for goal in goals {
            let mut next: Vec<(Substitution, Vec<Atom>)> = Vec::new();
            for (subst, premises) in &states {
                let applied = apply_atom(subst, goal);
                let base = self.facts_by_pred.get(&applied.predicate);
                let derived_extra = derived_by_pred.get(&applied.predicate);
                if base.is_none() && derived_extra.is_none() {
                    continue;
                }
                let facts = base
                    .into_iter()
                    .flat_map(|facts| facts.iter())
                    .chain(derived_extra.into_iter().flat_map(|facts| facts.iter()));
                for fact in facts {
                    if fact.arity() != applied.arity() {
                        continue;
                    }
                    let mut unified = subst.clone();
                    if unify_prepared(&applied, fact, &mut unified) {
                        let mut chain = Vec::with_capacity(premises.len() + 1);
                        chain.extend_from_slice(premises);
                        chain.push(fact.clone());
                        next.push((unified, chain));
                        if next.len() >= 4_000 {
                            break;
                        }
                    }
                }
                if next.len() >= 4_000 {
                    break;
                }
            }
            states = next;
            if states.is_empty() {
                break;
            }
        }
        states
    }
}

/// 快速预检查：若所有已解析的词项对都是互不相同的常量，则合一必然失败。
/// 用于在克隆全部替换之前尽早跳过不匹配的事实/规则头。
/// 沿置换链解析词项，返回最具体的引用
fn resolve_ref<'a>(term: &'a Term, subst: &'a Substitution) -> &'a Term {
    let mut current = term;
    let mut hops = 0;
    while let Term::Var(name) = current {
        match subst.get_rc(name) {
            Some(next) => {
                current = next;
                hops += 1;
                if hops > 128 {
                    break;
                }
            }
            None => break,
        }
    }
    current
}

fn resolve_term(term: &Term, subst: &Substitution) -> Term {
    resolve_ref(term, subst).clone()
}

/// `term` 必须是 [`resolve_ref`] 已解析过的词项
fn occurs_in(var: &str, term: &Term) -> bool {
    matches!(term, Term::Var(name) if &**name == var)
}

fn unify_terms(left: &Term, right: &Term, subst: &mut Substitution) -> bool {
    let (var, other) = match (resolve_ref(left, subst), resolve_ref(right, subst)) {
        (Term::Var(a), Term::Var(b)) if a == b => return true,
        (Term::Var(var), other) | (other, Term::Var(var)) => (var.clone(), other.clone()),
        (Term::Const(a), Term::Const(b)) => return a == b,
    };
    if occurs_in(&var, &other) {
        false
    } else {
        subst.insert(var, other);
        true
    }
}

/// 两个原子合一（谓词同名、参数同元数，逐项合一）
pub fn unify_atoms(pattern: &Atom, other: &Atom, subst: &mut Substitution) -> bool {
    if pattern.predicate != other.predicate || pattern.arity() != other.arity() {
        return false;
    }
    for (left, right) in pattern.terms.iter().zip(other.terms.iter()) {
        if !unify_terms(left, right, subst) {
            return false;
        }
    }
    true
}

/// 记录本次合一中已处理的变量；返回 true 表示该变量重复出现。
/// 超出小数组容量时同样返回 true，让调用方回退到通用合一（保守但正确）。
#[inline]
fn remember_var(
    touched: &mut [Option<Rc<str>>; 4],
    len: &mut usize,
    name: &Rc<str>,
) -> bool {
    for slot in touched.iter().flatten() {
        if Rc::ptr_eq(slot, name) || **slot == **name {
            return true;
        }
    }
    if *len < touched.len() {
        touched[*len] = Some(name.clone());
        *len += 1;
        false
    } else {
        true
    }
}

/// 合一快速路径，前置条件：
/// - `pattern` 已按 `subst` 完全解析（其中仍为变量的项必然在 `subst` 中未绑定）；
/// - `other` 为基原子，或其变量均未出现在 `subst` 中。
///
/// 满足条件时无需为每一对词项重复扫描置换链，可直接写入新绑定；
/// 同一变量在本次合一中重复出现时回退到通用合一，语义与 [`unify_atoms`] 一致。
fn unify_prepared(pattern: &Atom, other: &Atom, subst: &mut Substitution) -> bool {
    if pattern.predicate != other.predicate || pattern.arity() != other.arity() {
        return false;
    }
    let mut touched: [Option<Rc<str>>; 4] = Default::default();
    let mut touched_len = 0usize;
    for (left, right) in pattern.terms.iter().zip(other.terms.iter()) {
        match (left, right) {
            (Term::Const(a), Term::Const(b)) => {
                if a != b {
                    return false;
                }
            }
            (Term::Var(a), Term::Var(b)) => {
                if a == b {
                    continue;
                }
                // 两个变量都要登记：回退到通用合一时可能绑定其中之一
                // （或经解析链绑定），漏登记会让后续出现被当作新变量而被直接覆盖
                let seen_a = remember_var(&mut touched, &mut touched_len, a);
                let seen_b = remember_var(&mut touched, &mut touched_len, b);
                if seen_a || seen_b {
                    if !unify_terms(left, right, subst) {
                        return false;
                    }
                } else {
                    subst.insert(a.clone(), right.clone());
                }
            }
            (Term::Var(a), _) => {
                if remember_var(&mut touched, &mut touched_len, a) {
                    if !unify_terms(left, right, subst) {
                        return false;
                    }
                } else {
                    subst.insert(a.clone(), right.clone());
                }
            }
            (_, Term::Var(b)) => {
                if remember_var(&mut touched, &mut touched_len, b) {
                    if !unify_terms(left, right, subst) {
                        return false;
                    }
                } else {
                    subst.insert(b.clone(), left.clone());
                }
            }
            _ => {}
        }
    }
    true
}

fn apply_term(subst: &Substitution, term: &Term) -> Term {
    resolve_term(term, subst)
}

pub fn apply_atom(subst: &Substitution, atom: &Atom) -> Atom {
    if subst.head.is_none() {
        // 空置换下解析结果与原原子相同，直接克隆
        return atom.clone();
    }
    Atom {
        predicate: atom.predicate.clone(),
        terms: atom.terms.iter().map(|term| apply_term(subst, term)).collect(),
    }
}

/// 与 `format!("{}#{}", name, tag)` 命名一致
fn fresh_var_name(name: &str, tag: usize) -> String {
    let mut out = String::with_capacity(name.len() + 21);
    out.push_str(name);
    out.push('#');
    let mut digits = [0u8; 20];
    let mut index = digits.len();
    let mut value = tag;
    loop {
        index -= 1;
        digits[index] = b'0' + (value % 10) as u8;
        value /= 10;
        if value == 0 {
            break;
        }
    }
    out.push_str(std::str::from_utf8(&digits[index..]).unwrap_or("0"));
    out
}

/// 让同名的变量在子句内共享同一个 `Rc<str>`
fn canonicalize_clause_vars(clause: &mut Clause) {
    let mut canonical: HashMap<Rc<str>, Rc<str>> = HashMap::new();
    for atom in std::iter::once(&mut clause.head).chain(clause.body.iter_mut()) {
        for term in &mut atom.terms {
            if let Term::Var(name) = term {
                match canonical.get(name) {
                    Some(shared) => *name = shared.clone(),
                    None => {
                        canonical.insert(name.clone(), name.clone());
                    }
                }
            }
        }
    }
}

/// 变量换名器：同一次子句使用中，相同变量名的所有出现共享同一个换名结果。
struct Renamer<'a> {
    tag: usize,
    cache: Vec<(&'a Rc<str>, Rc<str>)>,
}

impl<'a> Renamer<'a> {
    fn new(tag: usize) -> Self {
        Self {
            tag,
            cache: Vec::new(),
        }
    }

    fn fresh(&mut self, name: &'a Rc<str>) -> Rc<str> {
        if let Some((_, fresh)) = self
            .cache
            .iter()
            .find(|(original, _)| Rc::ptr_eq(original, name))
        {
            return fresh.clone();
        }
        let fresh: Rc<str> = fresh_var_name(name, self.tag).into();
        self.cache.push((name, fresh.clone()));
        fresh
    }

    fn atom(&mut self, atom: &'a Atom) -> Atom {
        Atom {
            predicate: atom.predicate.clone(),
            terms: atom
                .terms
                .iter()
                .map(|term| match term {
                    Term::Var(name) => Term::Var(self.fresh(name)),
                    other => other.clone(),
                })
                .collect(),
        }
    }
}

pub fn freshen_clause(clause: &Clause, tag: usize) -> Clause {
    let mut renamer = Renamer::new(tag);
    Clause {
        head: renamer.atom(&clause.head),
        body: clause.body.iter().map(|atom| renamer.atom(atom)).collect(),
        confidence: clause.confidence,
        source: clause.source.clone(),
    }
}

pub fn render_term(term: &Term) -> String {
    match term {
        Term::Var(name) => format!("?{}", name),
        Term::Const(name) => name.to_string(),
    }
}

/// 证明树文本渲染（缩进层级 + 置信度），供报告/日志使用
pub fn render_proof(node: &ProofNode, indent: usize) -> Vec<String> {
    let mut lines = Vec::new();
    render_proof_into(node, indent, &mut lines);
    lines
}

fn render_proof_into(node: &ProofNode, indent: usize, lines: &mut Vec<String>) {
    let padding = "  ".repeat(indent);
    match &node.source {
        ProofSource::Fact => lines.push(format!("{}{}（事实）", padding, node.atom.display())),
        ProofSource::Rule { source } => lines.push(format!(
            "{}{}（规则：{}，置信度 {:.2}）",
            padding,
            node.atom.display(),
            source,
            node.confidence
        )),
    }
    for child in &node.children {
        render_proof_into(child, indent + 1, lines);
    }
}

/// 证明是否为纯事实（无规则参与）
pub fn proof_uses_rules(node: &ProofNode) -> bool {
    !node.is_fact() || node.children.iter().any(proof_uses_rules)
}

/// 证明中出现的规则来源（去重、保持出现顺序）
pub fn proof_rule_sources(node: &ProofNode) -> Vec<String> {
    let mut sources = Vec::new();
    let mut seen: HashSet<&str> = HashSet::new();
    collect_rule_sources(node, &mut sources, &mut seen);
    sources
}

fn collect_rule_sources<'a>(
    node: &'a ProofNode,
    sources: &mut Vec<String>,
    seen: &mut HashSet<&'a str>,
) {
    if let ProofSource::Rule { source } = &node.source {
        if seen.insert(source.as_str()) {
            sources.push(source.clone());
        }
    }
    for child in &node.children {
        collect_rule_sources(child, sources, seen);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn master_engine() -> ResolutionEngine {
        let mut engine = ResolutionEngine::new(6, 10);
        engine.add_fact("师傅", "张三", "李四");
        engine.add_fact("师傅", "李四", "王五");
        engine.add_fact("师傅", "王五", "赵六");
        engine.add_fact("师祖", "张三", "王五");
        engine
    }

    fn compose_rule(confidence: f64) -> Clause {
        Clause::new(
            Atom::pattern("师祖", Term::var("A"), Term::var("C")),
            vec![
                Atom::pattern("师傅", Term::var("A"), Term::var("B")),
                Atom::pattern("师傅", Term::var("B"), Term::var("C")),
            ],
            confidence,
            "组合规则：师傅(A,B)∧师傅(B,C)⇒师祖(A,C)",
        )
    }

    #[test]
    fn test_unify_terms_and_substitution_chains() {
        let mut subst = Substitution::new();
        assert!(unify_terms(&Term::var("X"), &Term::constant("张三"), &mut subst));
        assert_eq!(subst.get("X"), Some(&Term::constant("张三")));
        // 变量链：X→Y 后，X 与常量合一应绑定到 Y，并能沿链解析出常量
        let mut subst = Substitution::new();
        subst.insert("X".to_string(), Term::var("Y"));
        assert!(unify_terms(&Term::var("X"), &Term::constant("李四"), &mut subst));
        assert_eq!(resolve_term(&Term::var("X"), &subst), Term::constant("李四"));
        assert_eq!(resolve_term(&Term::var("Y"), &subst), Term::constant("李四"));
        // 同一变量出现在两个位置时必须绑定同一常量（出现检查的等价约束）
        assert!(!unify_atoms(
            &Atom::pattern("师傅", Term::var("X"), Term::var("X")),
            &Atom::rel("师傅", "张三", "李四"),
            &mut Substitution::new()
        ));
    }

    #[test]
    fn test_solve_ground_goal_from_fact() {
        let engine = master_engine();
        let goal = Atom::rel("师傅", "张三", "李四");
        let solutions = engine.solve(&[goal], &[]);
        assert_eq!(solutions.len(), 1);
        assert!(solutions[0].proofs[0].is_fact());
    }

    #[test]
    fn test_solve_open_goal_binds_variable() {
        let engine = master_engine();
        let goal = Atom::pattern("师傅", Term::constant("张三"), Term::var("X"));
        let solutions = engine.solve(&[goal], &["X".to_string()]);
        assert_eq!(solutions.len(), 1);
        assert_eq!(solutions[0].bindings.get("X").map(String::as_str), Some("李四"));
    }

    #[test]
    fn test_solve_uses_rule_with_proof_chain() {
        let mut engine = master_engine();
        engine.add_clause(compose_rule(0.9));
        let goal = Atom::rel("师祖", "李四", "赵六");
        let solutions = engine.solve(&[goal], &[]);
        assert!(!solutions.is_empty(), "组合规则应能证明李四的师祖是赵六");
        let solution = &solutions[0];
        assert!(proof_uses_rules(&solution.proofs[0]));
        assert!(proof_rule_sources(&solution.proofs[0])[0].contains("组合规则"));
        assert!((solution.confidence - 0.9).abs() < 1e-9);
        assert_eq!(solution.proofs[0].children.len(), 2);
    }

    #[test]
    fn test_solve_chain_goals_with_intermediate_vars() {
        let engine = master_engine();
        let goals = vec![
            Atom::pattern("师傅", Term::constant("张三"), Term::var("V1")),
            Atom::pattern("师傅", Term::var("V1"), Term::var("X")),
        ];
        let solutions = engine.solve(&goals, &["X".to_string()]);
        assert_eq!(solutions.len(), 1);
        assert_eq!(solutions[0].bindings.get("X").map(String::as_str), Some("王五"));
        assert_eq!(solutions[0].proofs.len(), 2);
    }

    #[test]
    fn test_depth_limit_bounds_transitive_search() {
        let mut engine = ResolutionEngine::new(2, 5);
        engine.add_fact("父", "a", "b");
        engine.add_fact("父", "b", "c");
        engine.add_fact("父", "c", "d");
        engine.add_clause(Clause::new(
            Atom::pattern("祖先", Term::var("A"), Term::var("B")),
            vec![Atom::pattern("父", Term::var("A"), Term::var("B"))],
            1.0,
            "祖先基础规则",
        ));
        engine.add_clause(Clause::new(
            Atom::pattern("祖先", Term::var("A"), Term::var("B")),
            vec![
                Atom::pattern("父", Term::var("A"), Term::var("C")),
                Atom::pattern("祖先", Term::var("C"), Term::var("B")),
            ],
            1.0,
            "祖先递归规则",
        ));
        let goal = Atom::pattern("祖先", Term::constant("a"), Term::var("X"));
        let solutions = engine.solve(&[goal], &["X".to_string()]);
        let values: HashSet<String> = solutions
            .iter()
            .map(|solution| solution.bindings.get("X").cloned().unwrap_or_default())
            .collect();
        assert!(values.contains("b"));
        assert!(values.contains("c"));
        // 深度上限必须终止搜索，不允许无限递归
        assert!(solutions.len() <= 5);
    }

    #[test]
    fn test_forward_chain_derives_multi_step_facts() {
        let mut engine = master_engine();
        engine.add_clause(compose_rule(0.8));
        engine.add_clause(Clause::new(
            Atom::pattern("太师祖", Term::var("A"), Term::var("C")),
            vec![
                Atom::pattern("师傅", Term::var("A"), Term::var("B")),
                Atom::pattern("师祖", Term::var("B"), Term::var("C")),
            ],
            0.7,
            "二阶组合规则：师傅(A,B)∧师祖(B,C)⇒太师祖(A,C)",
        ));
        let derived = engine.forward_chain(3, 50);
        let first: Vec<&DerivedFact> = derived.iter().filter(|d| d.depth == 1).collect();
        assert!(first
            .iter()
            .any(|d| d.atom == Atom::rel("师祖", "李四", "赵六")));
        let second: Vec<&DerivedFact> = derived.iter().filter(|d| d.depth >= 2).collect();
        assert!(
            second
                .iter()
                .any(|d| d.atom == Atom::rel("太师祖", "张三", "赵六")),
            "第二轮应能推出太师祖：{:?}",
            second.iter().map(|d| d.atom.display()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_forward_chain_confidence_and_premises() {
        let mut engine = master_engine();
        engine.add_clause(compose_rule(0.6));
        let derived = engine.forward_chain(2, 20);
        let fact = derived
            .iter()
            .find(|d| d.atom == Atom::rel("师祖", "李四", "赵六"))
            .expect("应推出李四的师祖是赵六");
        assert_eq!(fact.premises.len(), 2);
        assert_eq!(fact.premises[0], Atom::rel("师傅", "李四", "王五"));
        assert!((fact.confidence - 0.6).abs() < 1e-9);
    }

    #[test]
    fn test_solution_dedup_keeps_highest_confidence() {
        let mut engine = master_engine();
        engine.add_clause(compose_rule(0.4));
        engine.add_clause(Clause::new(
            Atom::pattern("师祖", Term::var("A"), Term::var("C")),
            vec![
                Atom::pattern("师傅", Term::var("A"), Term::var("B")),
                Atom::pattern("师傅", Term::var("B"), Term::var("C")),
            ],
            0.9,
            "更强组合规则",
        ));
        let solutions = engine.solve(&[Atom::rel("师祖", "李四", "赵六")], &[]);
        assert_eq!(solutions.len(), 1, "同绑定不同路径应去重");
        assert!((solutions[0].confidence - 0.9).abs() < 1e-9);
    }
}

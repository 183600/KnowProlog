// 一阶归结推理引擎（纯符号，无神经网络、无权重训练）
//
// 灵感：Robinson 归结原理。知识图谱三元组被编码为基原子（ground atom），
// 链接预测挖掘出的 Horn 规则与关联规则成为子句；问答被转成目标子句后：
//   • 反向（SLD 反驳）：逐目标与事实/规则头做合一，成功即给出变量绑定与证明树；
//   • 正向（半朴素饱和）：从事实出发反复应用规则，推导多跳新事实并保留前提来源。
//
// 全程只有：项、置换、合一、深度上限与搜索预算——每一步都可解释、可回溯。

use std::collections::{HashMap, HashSet};

/// 项：逻辑变量或常量（实体名、属性值、谓词名都是常量）
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Term {
    Var(String),
    Const(String),
}

impl Term {
    pub fn var(name: &str) -> Self {
        Term::Var(name.to_string())
    }

    pub fn constant(name: &str) -> Self {
        Term::Const(name.to_string())
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
    pub predicate: String,
    pub terms: Vec<Term>,
}

impl Atom {
    pub fn new(predicate: &str, terms: Vec<Term>) -> Self {
        Self {
            predicate: predicate.to_string(),
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
        let args = self
            .terms
            .iter()
            .map(|term| match term {
                Term::Var(name) => format!("?{}", name),
                Term::Const(name) => name.clone(),
            })
            .collect::<Vec<_>>()
            .join(", ");
        format!("{}({})", self.predicate, args)
    }
}

/// 置换：变量名 → 项
pub type Substitution = HashMap<String, Term>;

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

pub struct ResolutionEngine {
    facts: HashSet<Atom>,
    facts_by_pred: HashMap<String, Vec<Atom>>,
    clauses: Vec<Clause>,
    rules_by_head: HashMap<String, Vec<usize>>,
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

    /// 加入一元基事实 `predicate(subject)`
    pub fn add_unary_fact(&mut self, predicate: &str, subject: &str) -> bool {
        self.add_fact_atom(Atom::unary(predicate, Term::constant(subject)))
    }

    pub fn add_clause(&mut self, clause: Clause) {
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
            let bindings: HashMap<String, String> = query_vars
                .iter()
                .map(|var| {
                    let value = resolve_term(&Term::Var(var.clone()), &subst);
                    (var.clone(), render_term(&value))
                })
                .collect();
            let key = query_vars
                .iter()
                .map(|var| format!("{}={}", var, bindings.get(var).cloned().unwrap_or_default()))
                .collect::<Vec<_>>()
                .join("|");
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
                if !may_unify_quick(&applied, fact, subst) {
                    continue;
                }
                let mut unified = subst.clone();
                if unify_atoms(&applied, fact, &mut unified) {
                    let ground = apply_atom(&unified, &applied);
                    results.push((unified, ProofNode::fact(ground)));
                    if results.len() >= self.max_solutions {
                        return results;
                    }
                }
            }
        }

        // 2) 与规则头归结：先对子句做变量换名，再合一，然后递归证明规则体
        if let Some(indices) = self.rules_by_head.get(&applied.predicate) {
            for &idx in indices {
                if *budget == 0 {
                    break;
                }
                let clause = &self.clauses[idx];
                let tag = *counter;
                *counter += 1;
                let fresh = freshen_clause(clause, tag);
                let mut unified = subst.clone();
                *budget -= 1;
                if !unify_atoms(&applied, &fresh.head, &mut unified) {
                    continue;
                }
                for (solved, children) in
                    self.prove_body(&fresh.body, depth + 1, &unified, counter, budget)
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

    /// 前向半朴素饱和：从事实出发反复应用子句，推导新基事实（含前提来源）
    pub fn forward_chain(&self, max_rounds: usize, max_new_facts: usize) -> Vec<DerivedFact> {
        let mut known: HashSet<Atom> = self.facts.clone();
        let mut known_by_pred: HashMap<String, Vec<Atom>> = self.facts_by_pred.clone();
        let mut confidence_of: HashMap<Atom, f64> = HashMap::new();
        for fact in &self.facts {
            confidence_of.insert(fact.clone(), 1.0);
        }
        let mut derived: Vec<DerivedFact> = Vec::new();
        let mut counter = 0usize;

        for round in 1..=max_rounds.max(1) {
            let mut new_this_round: Vec<DerivedFact> = Vec::new();
            let mut new_this_round_atoms: HashSet<Atom> = HashSet::new();
            for clause in &self.clauses {
                let tag = counter;
                counter += 1;
                let fresh = freshen_clause(clause, tag);
                for (subst, premises) in self.match_body(&fresh.body, &known_by_pred) {
                    let head = apply_atom(&subst, &fresh.head);
                    if !head.is_ground() || known.contains(&head) {
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
                known.insert(item.atom.clone());
                known_by_pred
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
        known_by_pred: &HashMap<String, Vec<Atom>>,
    ) -> Vec<(Substitution, Vec<Atom>)> {
        let mut states: Vec<(Substitution, Vec<Atom>)> = vec![(Substitution::new(), Vec::new())];
        for goal in goals {
            let mut next: Vec<(Substitution, Vec<Atom>)> = Vec::new();
            for (subst, premises) in &states {
                let applied = apply_atom(subst, goal);
                let Some(facts) = known_by_pred.get(&applied.predicate) else {
                    continue;
                };
                for fact in facts {
                    if fact.arity() != applied.arity() {
                        continue;
                    }
                    if !may_unify_quick(&applied, fact, subst) {
                        continue;
                    }
                    let mut unified = subst.clone();
                    if unify_atoms(&applied, fact, &mut unified) {
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
fn may_unify_quick(pattern: &Atom, other: &Atom, subst: &Substitution) -> bool {
    for (left, right) in pattern.terms.iter().zip(other.terms.iter()) {
        let resolved_left = resolve_term(left, subst);
        let resolved_right = resolve_term(right, subst);
        if let (Term::Const(a), Term::Const(b)) = (&resolved_left, &resolved_right) {
            if a != b {
                return false;
            }
        }
    }
    true
}

fn resolve_term(term: &Term, subst: &Substitution) -> Term {
    let mut current = term.clone();
    let mut hops = 0;
    while let Term::Var(name) = &current {
        let Some(next) = subst.get(name) else {
            break;
        };
        current = next.clone();
        hops += 1;
        if hops > 128 {
            break;
        }
    }
    current
}

fn occurs_in(var: &str, term: &Term, subst: &Substitution) -> bool {
    match resolve_term(term, subst) {
        Term::Var(name) => name == var,
        Term::Const(_) => false,
    }
}

fn unify_terms(left: &Term, right: &Term, subst: &mut Substitution) -> bool {
    let resolved_left = resolve_term(left, subst);
    let resolved_right = resolve_term(right, subst);
    match (&resolved_left, &resolved_right) {
        (Term::Var(a), Term::Var(b)) if a == b => true,
        (Term::Var(var), other) | (other, Term::Var(var)) => {
            if occurs_in(var, other, subst) {
                false
            } else {
                subst.insert(var.clone(), other.clone());
                true
            }
        }
        (Term::Const(a), Term::Const(b)) => a == b,
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

fn apply_term(subst: &Substitution, term: &Term) -> Term {
    resolve_term(term, subst)
}

fn apply_atom(subst: &Substitution, atom: &Atom) -> Atom {
    Atom {
        predicate: atom.predicate.clone(),
        terms: atom.terms.iter().map(|term| apply_term(subst, term)).collect(),
    }
}

fn freshen_atom(atom: &Atom, tag: usize) -> Atom {
    Atom {
        predicate: atom.predicate.clone(),
        terms: atom
            .terms
            .iter()
            .map(|term| match term {
                Term::Var(name) => Term::Var(format!("{}#{}", name, tag)),
                other => other.clone(),
            })
            .collect(),
    }
}

fn freshen_clause(clause: &Clause, tag: usize) -> Clause {
    Clause {
        head: freshen_atom(&clause.head, tag),
        body: clause
            .body
            .iter()
            .map(|atom| freshen_atom(atom, tag))
            .collect(),
        confidence: clause.confidence,
        source: clause.source.clone(),
    }
}

pub fn render_term(term: &Term) -> String {
    match term {
        Term::Var(name) => format!("?{}", name),
        Term::Const(name) => name.clone(),
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
    collect_rule_sources(node, &mut sources);
    sources
}

fn collect_rule_sources(node: &ProofNode, sources: &mut Vec<String>) {
    if let ProofSource::Rule { source } = &node.source {
        if !sources.contains(source) {
            sources.push(source.clone());
        }
    }
    for child in &node.children {
        collect_rule_sources(child, sources);
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

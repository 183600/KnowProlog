// ProbLog 风格的概率逻辑程序（纯符号：可能世界语义 + 知识编译）
//
// 灵感：ProbLog 把 Prolog 扩展成概率逻辑程序：
//   • 概率事实 `p::fact`；
//   • 概率规则 `p::head :- body`（body 成立时以概率 p 推出 head）；
//   • 注释析取 `p1::h1 ; p2::h2 ; … :- body`（互斥多结果）；
//   • 查询语义 = 可能世界上的加权模型计数（WMC）：
//       P(q) = Σ_{ω ⊨ q} Π_{选择变量 v} (v∈ω ? p_v : 1−p_v)。
//
// 本模块的实现与 ProbLog 的知识编译路线一致，但零神经网络、零采样：
//   1. 基化：把规则在事实论域上半连接展开为基实例（每实例一个独立选择变量）；
//   2. 编译：把每个原子"何时成立"编译成有序二叉决策图（BDD），
//      对肯定程序做 Kleene 最小不动点迭代（单调、可终止）；
//   3. 计数：在 BDD 上做 WMC（带 memo 的线性遍历），得到精确概率；
//   4. 解释：枚举有限条证明路径，给出每条路径的联合概率与规则来源。
//
// 与 MLN-lite（log-odds 加权求和 + sigmoid）不同：这里的合取/析取共享同一批
// 选择变量，天然处理证明重叠（相关源头），给出的是可能世界精确概率而不是近似和。

use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::markov_logic::MarkovLogicEngine;
use crate::resolution_engine::{apply_atom, unify_atoms, Atom, Substitution, Term};

const MAX_GROUNDINGS_PER_RULE: usize = 400;
const TRUE_NODE: u32 = 1;
const FALSE_NODE: u32 = 0;
const TERMINAL_VAR: u32 = u32::MAX;

/// 概率降序、同概率按显示串升序的排序键
#[derive(PartialEq)]
struct ProbabilityDisplayKey {
    probability: f64,
    display: String,
}

impl Eq for ProbabilityDisplayKey {}

impl PartialOrd for ProbabilityDisplayKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ProbabilityDisplayKey {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .probability
            .partial_cmp(&self.probability)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.display.cmp(&other.display))
    }
}

/// 实例排序键：概率降序 → 来源升序 → 前件显示串升序
#[derive(PartialEq)]
struct InstanceOrderKey {
    probability: f64,
    source: String,
    premises: String,
}

impl Eq for InstanceOrderKey {}

impl PartialOrd for InstanceOrderKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for InstanceOrderKey {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .probability
            .partial_cmp(&self.probability)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.source.cmp(&other.source))
            .then_with(|| self.premises.cmp(&other.premises))
    }
}

/// 概率事实：`p::atom`（基原子）
#[derive(Debug, Clone)]
pub struct ProbabilisticFact {
    pub atom: Atom,
    pub probability: f64,
}

/// 概率规则：`p::head :- body`
#[derive(Debug, Clone)]
pub struct ProbabilisticRule {
    pub head: Atom,
    pub body: Vec<Atom>,
    pub probability: f64,
    pub source: String,
}

/// 注释析取：`p1::h1 ; p2::h2 ; … :- body`（各结果互斥，概率自动归一）
#[derive(Debug, Clone)]
pub struct AnnotatedDisjunction {
    pub heads: Vec<(Atom, f64)>,
    pub body: Vec<Atom>,
    pub source: String,
}

/// 基化后的概率实例：前件全基、后件基原子、条件为选择变量的 BDD 公式
#[derive(Debug, Clone)]
struct GroundProbability {
    head: Atom,
    premises: Vec<Atom>,
    condition: u32,
    probability: f64,
    source: String,
}

/// 证明路径上的一步：前件 ⇒ 结论（含规则来源与该步概率）
#[derive(Debug, Clone)]
pub struct ProblogProofStep {
    pub head: Atom,
    pub premises: Vec<Atom>,
    pub source: String,
    pub probability: f64,
}

/// 一条完整证明路径（steps 按"先前件后结论"排列），probability 为路径联合概率
#[derive(Debug, Clone)]
pub struct ProblogProof {
    pub steps: Vec<ProblogProofStep>,
    pub probability: f64,
}

/// 一个查询答案：原子 + 精确概率 + 证明路径
#[derive(Debug, Clone)]
pub struct ProblogAnswer {
    pub atom: Atom,
    pub probability: f64,
    pub from_fact: bool,
    pub proofs: Vec<ProblogProof>,
}

/// 一次查询的汇总（顺带给出编译规模，便于报告展示）
#[derive(Debug, Clone)]
pub struct ProblogQuery {
    pub answers: Vec<ProblogAnswer>,
    pub choice_variables: usize,
    pub bdd_nodes: usize,
    pub ground_instances: usize,
    pub ground_rounds: usize,
    pub exact: bool,
}

/// 有序二叉决策图（BDD）管理器：可能世界编译的核心数据结构
struct BddManager {
    nodes: Vec<(u32, u32, u32)>,
    unique: HashMap<(u32, u32, u32), u32>,
    var_probability: Vec<f64>,
    var_description: Vec<String>,
    apply_memo: HashMap<(u8, u32, u32), u32>,
    wmc_memo: HashMap<u32, f64>,
    max_nodes: usize,
    overflow: bool,
}

impl BddManager {
    fn new(max_nodes: usize) -> Self {
        Self {
            nodes: Vec::new(),
            unique: HashMap::new(),
            var_probability: Vec::new(),
            var_description: Vec::new(),
            apply_memo: HashMap::new(),
            wmc_memo: HashMap::new(),
            max_nodes: max_nodes.max(64),
            overflow: false,
        }
    }

    fn new_var(&mut self, probability: f64, description: String) -> u32 {
        let id = self.var_probability.len() as u32;
        self.var_probability
            .push(probability.clamp(1e-9, 1.0 - 1e-9));
        self.var_description.push(description);
        id
    }

    fn var_count(&self) -> usize {
        self.var_probability.len()
    }

    fn node_count(&self) -> usize {
        self.nodes.len()
    }

    fn overflow(&self) -> bool {
        self.overflow
    }

    fn literal(&mut self, var: u32, positive: bool) -> u32 {
        if positive {
            self.make_node(var, FALSE_NODE, TRUE_NODE)
        } else {
            self.make_node(var, TRUE_NODE, FALSE_NODE)
        }
    }

    fn conjunction(&mut self, a: u32, b: u32) -> u32 {
        self.apply(0, a, b)
    }

    fn disjunction(&mut self, a: u32, b: u32) -> u32 {
        self.apply(1, a, b)
    }

    fn make_node(&mut self, var: u32, low: u32, high: u32) -> u32 {
        if low == high {
            return low;
        }
        if let Some(&existing) = self.unique.get(&(var, low, high)) {
            return existing;
        }
        let id = self.nodes.len() as u32 + 2;
        if self.nodes.len() + 2 > self.max_nodes {
            self.overflow = true;
        }
        self.nodes.push((var, low, high));
        self.unique.insert((var, low, high), id);
        id
    }

    fn node(&self, id: u32) -> Option<(u32, u32, u32)> {
        if id < 2 {
            return None;
        }
        self.nodes.get(id as usize - 2).copied()
    }

    fn top_var(&self, id: u32) -> u32 {
        self.node(id).map(|(var, _, _)| var).unwrap_or(TERMINAL_VAR)
    }

    fn cofactors(&self, id: u32, var: u32) -> (u32, u32) {
        match self.node(id) {
            Some((node_var, low, high)) if node_var == var => (low, high),
            _ => (id, id),
        }
    }

    fn apply(&mut self, op: u8, a: u32, b: u32) -> u32 {
        let (a, b) = if op == 0 && a > b { (b, a) } else { (a, b) };
        match op {
            0 => {
                if a == FALSE_NODE || b == FALSE_NODE {
                    return FALSE_NODE;
                }
                if a == TRUE_NODE {
                    return b;
                }
                if b == TRUE_NODE {
                    return a;
                }
            }
            _ => {
                if a == TRUE_NODE || b == TRUE_NODE {
                    return TRUE_NODE;
                }
                if a == FALSE_NODE {
                    return b;
                }
                if b == FALSE_NODE {
                    return a;
                }
            }
        }
        if a == b {
            return a;
        }
        let key = (op, a, b);
        if let Some(&cached) = self.apply_memo.get(&key) {
            return cached;
        }
        let top = self.top_var(a).min(self.top_var(b));
        let (a_low, a_high) = self.cofactors(a, top);
        let (b_low, b_high) = self.cofactors(b, top);
        let low = self.apply(op, a_low, b_low);
        let high = self.apply(op, a_high, b_high);
        let result = self.make_node(top, low, high);
        self.apply_memo.insert(key, result);
        result
    }

    fn wmc(&mut self, root: u32) -> f64 {
        if root == FALSE_NODE {
            return 0.0;
        }
        if root == TRUE_NODE {
            return 1.0;
        }
        if let Some(&cached) = self.wmc_memo.get(&root) {
            return cached;
        }
        let Some((var, low, high)) = self.node(root) else {
            return 0.0;
        };
        let p = self.var_probability[var as usize];
        let low_weight = self.wmc(low);
        let high_weight = self.wmc(high);
        let value = p * high_weight + (1.0 - p) * low_weight;
        self.wmc_memo.insert(root, value);
        value
    }
}

/// 编译结果：基实例 + 每个原子的 BDD 编号与精确概率
struct Compiled {
    instances: Vec<GroundProbability>,
    instances_by_head: HashMap<Atom, Vec<usize>>,
    atom_probability: HashMap<Atom, f64>,
    deterministic_facts: HashSet<Atom>,
    probabilistic_facts: HashMap<Atom, f64>,
    exact: bool,
    ground_rounds: usize,
    choice_variables: usize,
    bdd_nodes: usize,
    max_depth: usize,
}

impl Compiled {
    fn proofs_for(&self, atom: &Atom, max_proofs: usize) -> Vec<ProblogProof> {
        let mut budget = max_proofs.max(1) * 4;
        let mut path: HashSet<Atom> = HashSet::new();
        self.collect_proofs(atom, max_proofs, &mut path, &mut budget, 0)
    }

    fn collect_proofs(
        &self,
        atom: &Atom,
        max_proofs: usize,
        path: &mut HashSet<Atom>,
        budget: &mut usize,
        depth: usize,
    ) -> Vec<ProblogProof> {
        if *budget == 0 || depth > self.max_depth {
            return Vec::new();
        }
        if self.deterministic_facts.contains(atom) {
            *budget -= 1;
            return vec![ProblogProof {
                steps: Vec::new(),
                probability: 1.0,
            }];
        }
        // 概率事实本身是一条证明；同一原子还可能有规则支持，继续向下收集
        let mut results: Vec<ProblogProof> = Vec::new();
        if let Some(&probability) = self.probabilistic_facts.get(atom) {
            *budget -= 1;
            results.push(ProblogProof {
                steps: Vec::new(),
                probability,
            });
        }
        if path.contains(atom) {
            return results;
        }
        let Some(indices) = self.instances_by_head.get(atom) else {
            return results;
        };
        path.insert(atom.clone());
        for &index in indices.iter().take(6) {
            if *budget == 0 || results.len() >= max_proofs {
                break;
            }
            let instance = &self.instances[index];
            let mut premise_options: Vec<Vec<ProblogProof>> = Vec::new();
            let mut feasible = true;
            for premise in &instance.premises {
                let options = self.collect_proofs(premise, max_proofs, path, budget, depth + 1);
                if options.is_empty() {
                    feasible = false;
                    break;
                }
                premise_options.push(options.into_iter().take(3).collect());
            }
            if !feasible {
                continue;
            }
            let mut combinations: Vec<(f64, Vec<ProblogProofStep>)> = vec![(1.0, Vec::new())];
            for options in &premise_options {
                let mut next: Vec<(f64, Vec<ProblogProofStep>)> = Vec::new();
                for (probability, steps) in &combinations {
                    for option in options {
                        let mut merged = steps.clone();
                        merged.extend(option.steps.clone());
                        next.push((probability * option.probability, merged));
                        if next.len() >= 4 {
                            break;
                        }
                    }
                    if next.len() >= 4 {
                        break;
                    }
                }
                combinations = next;
            }
            for (probability, mut steps) in combinations {
                steps.push(ProblogProofStep {
                    head: atom.clone(),
                    premises: instance.premises.clone(),
                    source: instance.source.clone(),
                    probability: instance.probability,
                });
                results.push(ProblogProof {
                    steps,
                    probability: probability * instance.probability,
                });
                *budget = budget.saturating_sub(1);
                if results.len() >= max_proofs {
                    break;
                }
            }
        }
        path.remove(atom);
        results
    }
}

/// ProbLog 风格引擎：概率事实 + 概率规则 + 注释析取，查询即精确 WMC
#[derive(Clone)]
pub struct ProblogEngine {
    deterministic_facts: HashSet<Atom>,
    probabilistic_facts: Vec<ProbabilisticFact>,
    rules: Vec<ProbabilisticRule>,
    annotated_disjunctions: Vec<AnnotatedDisjunction>,
    max_depth: usize,
    max_proofs: usize,
    max_bdd_nodes: usize,
    compiled: RefCell<Option<Arc<Compiled>>>,
}

impl ProblogEngine {
    pub fn new(max_depth: usize, max_proofs: usize) -> Self {
        Self {
            deterministic_facts: HashSet::new(),
            probabilistic_facts: Vec::new(),
            rules: Vec::new(),
            annotated_disjunctions: Vec::new(),
            max_depth: max_depth.clamp(1, 16),
            max_proofs: max_proofs.clamp(1, 64),
            max_bdd_nodes: 200_000,
            compiled: RefCell::new(None),
        }
    }

    /// 由 MLN-lite 引擎转换：事实保留为硬事实，权重 w 换算成概率 σ(w)
    pub fn from_markov_engine(
        engine: &MarkovLogicEngine,
        max_depth: usize,
        max_proofs: usize,
    ) -> Self {
        let mut program = Self::new(max_depth, max_proofs);
        for fact in engine.facts() {
            program.add_deterministic_fact(fact.clone());
        }
        for rule in engine.rules() {
            let probability = 1.0 / (1.0 + (-rule.weight.clamp(-6.0, 6.0)).exp());
            program.add_rule(
                rule.head.clone(),
                rule.body.clone(),
                probability,
                &rule.source,
            );
        }
        program
    }

    pub fn add_deterministic_fact(&mut self, atom: Atom) {
        if atom.is_ground() {
            self.deterministic_facts.insert(atom);
            self.invalidate();
        }
    }

    pub fn add_probabilistic_fact(&mut self, atom: Atom, probability: f64) {
        if !atom.is_ground() {
            return;
        }
        let probability = probability.clamp(0.0, 1.0);
        if probability >= 1.0 {
            self.add_deterministic_fact(atom);
            return;
        }
        if probability <= 0.0 {
            return;
        }
        self.probabilistic_facts.push(ProbabilisticFact {
            atom,
            probability,
        });
        self.invalidate();
    }

    pub fn add_rule(&mut self, head: Atom, body: Vec<Atom>, probability: f64, source: &str) {
        if body.len() > 4 {
            return;
        }
        let probability = probability.clamp(0.0, 1.0);
        if probability <= 0.0 {
            return;
        }
        self.rules.push(ProbabilisticRule {
            head,
            body,
            probability,
            source: source.to_string(),
        });
        self.invalidate();
    }

    pub fn add_annotated_disjunction(
        &mut self,
        heads: Vec<(Atom, f64)>,
        body: Vec<Atom>,
        source: &str,
    ) {
        if heads.is_empty() || body.len() > 4 {
            return;
        }
        self.annotated_disjunctions.push(AnnotatedDisjunction {
            heads,
            body,
            source: source.to_string(),
        });
        self.invalidate();
    }

    pub fn deterministic_fact_count(&self) -> usize {
        self.deterministic_facts.len()
    }

    pub fn probabilistic_fact_count(&self) -> usize {
        self.probabilistic_facts.len()
    }

    pub fn rule_count(&self) -> usize {
        self.rules.len() + self.annotated_disjunctions.len()
    }

    pub fn has_deterministic_fact(&self, atom: &Atom) -> bool {
        self.deterministic_facts.contains(atom)
    }

    fn invalidate(&mut self) {
        *self.compiled.borrow_mut() = None;
    }

    fn compiled(&self) -> Arc<Compiled> {
        if let Some(existing) = self.compiled.borrow().as_ref() {
            return existing.clone();
        }
        let compiled = Arc::new(self.compile());
        *self.compiled.borrow_mut() = Some(compiled.clone());
        compiled
    }

    /// 精确查询：可能世界加权模型计数（BDD 编译，无采样）
    pub fn query(
        &self,
        pattern: &Atom,
        limit: usize,
        min_probability: f64,
    ) -> ProblogQuery {
        let compiled = self.compiled();
        let mut by_atom: HashMap<Atom, ProblogAnswer> = HashMap::new();
        for fact in &self.deterministic_facts {
            if pattern_matches(pattern, fact) {
                by_atom.insert(
                    fact.clone(),
                    ProblogAnswer {
                        atom: fact.clone(),
                        probability: 1.0,
                        from_fact: true,
                        proofs: Vec::new(),
                    },
                );
            }
        }
        for fact in &self.probabilistic_facts {
            // 同一原子既是确定事实时以确定事实为准（与 compile 的跳过规则一致）
            if self.deterministic_facts.contains(&fact.atom) {
                continue;
            }
            if !pattern_matches(pattern, &fact.atom) {
                continue;
            }
            // 概率事实同时被规则支持时，精确概率是 BDD 合并后的结果，
            // 不能只报裸事实概率（否则与依赖它的原子概率不一致）
            let probability = compiled
                .atom_probability
                .get(&fact.atom)
                .copied()
                .unwrap_or(fact.probability);
            if probability < min_probability {
                continue;
            }
            by_atom.insert(
                fact.atom.clone(),
                ProblogAnswer {
                    atom: fact.atom.clone(),
                    probability,
                    from_fact: true,
                    proofs: Vec::new(),
                },
            );
        }
        for (atom, &probability) in &compiled.atom_probability {
            if !pattern_matches(pattern, atom) || probability < min_probability {
                continue;
            }
            if self.deterministic_facts.contains(atom) || compiled.probabilistic_facts.contains_key(atom) {
                continue;
            }
            by_atom.insert(
                atom.clone(),
                ProblogAnswer {
                    atom: atom.clone(),
                    probability,
                    from_fact: false,
                    proofs: Vec::new(),
                },
            );
        }
        let mut answers: Vec<ProblogAnswer> = by_atom.into_values().collect();
        answers.sort_by_cached_key(|answer| ProbabilityDisplayKey {
            probability: answer.probability,
            display: answer.atom.display(),
        });
        answers.truncate(limit.max(1));

        // 排序与截断只依赖概率与原子显示名，证明在截断后仅为保留的答案补算。
        for answer in &mut answers {
            if answer.from_fact {
                // 确定事实不携带证明；概率事实沿用上限 2 的证明
                if compiled.probabilistic_facts.contains_key(&answer.atom) {
                    answer.proofs = compiled.proofs_for(&answer.atom, 2);
                }
            } else {
                answer.proofs = compiled.proofs_for(&answer.atom, self.max_proofs);
            }
        }
        ProblogQuery {
            answers,
            choice_variables: compiled.choice_variables,
            bdd_nodes: compiled.bdd_nodes,
            ground_instances: compiled.instances.len(),
            ground_rounds: compiled.ground_rounds,
            exact: compiled.exact,
        }
    }

    /// 主语相关的查询模式（供「概率推理：实体」在无谓词时枚举目标）
    pub fn patterns_for_subject(&self, subject: &str) -> Vec<Atom> {
        let compiled = self.compiled();
        let mut patterns: Vec<Atom> = Vec::new();
        let mut seen: HashSet<Atom> = HashSet::new();
        let mut consider = |atom: &Atom| {
            let matches_subject = atom
                .terms
                .first()
                .map(|term| term.name() == subject)
                .unwrap_or(false);
            if !matches_subject {
                return;
            }
            let mut terms = vec![Term::constant(subject)];
            for _ in 1..atom.arity() {
                terms.push(Term::var("X"));
            }
            let pattern = Atom::new(atom.predicate.as_ref(), terms);
            if seen.insert(pattern.clone()) {
                patterns.push(pattern);
            }
        };
        for fact in &self.deterministic_facts {
            consider(fact);
        }
        for fact in &self.probabilistic_facts {
            consider(&fact.atom);
        }
        for atom in compiled.atom_probability.keys() {
            consider(atom);
        }
        patterns.sort_by_cached_key(Atom::display);
        patterns
    }

    pub fn ground_instance_count(&self) -> usize {
        self.compiled().instances.len()
    }

    pub fn choice_variable_count(&self) -> usize {
        self.compiled().choice_variables
    }

    // ---- 编译管线：基化 → BDD 不动点 → WMC ----

    fn compile(&self) -> Compiled {
        let mut bdd = BddManager::new(self.max_bdd_nodes);
        let mut universe: HashMap<String, Vec<Atom>> = HashMap::new();
        let mut deterministic: HashSet<Atom> = HashSet::new();
        for fact in &self.deterministic_facts {
            deterministic.insert(fact.clone());
            push_universe(&mut universe, fact);
        }
        let mut probabilistic: Vec<&ProbabilisticFact> = self.probabilistic_facts.iter().collect();
        probabilistic.sort_by_cached_key(|fact| fact.atom.display());
        let mut prob_fact_probability: HashMap<Atom, f64> = HashMap::new();
        let mut prob_fact_bdd: HashMap<Atom, u32> = HashMap::new();
        for fact in probabilistic {
            if deterministic.contains(&fact.atom) {
                continue;
            }
            push_universe(&mut universe, &fact.atom);
            let var = bdd.new_var(
                fact.probability,
                format!("概率事实 {}={:.2}", fact.atom.display(), fact.probability),
            );
            let node = bdd.literal(var, true);
            // 同一原子的多条概率事实是析取支持：合并 BDD，而不是只保留最后一条
            let merged = prob_fact_bdd
                .get(&fact.atom)
                .map_or(node, |&existing| bdd.disjunction(existing, node));
            prob_fact_probability
                .entry(fact.atom.clone())
                .and_modify(|probability| {
                    *probability = 1.0 - (1.0 - *probability) * (1.0 - fact.probability);
                })
                .or_insert(fact.probability);
            prob_fact_bdd.insert(fact.atom.clone(), merged);
        }
        for atoms in universe.values_mut() {
            atoms.sort_by_cached_key(Atom::display);
            atoms.dedup();
        }

        let mut instances: Vec<GroundProbability> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        let mut ground_rounds = 0;
        for round in 0..self.max_depth.max(1) {
            ground_rounds = round + 1;
            let mut new_heads: Vec<Atom> = Vec::new();
            for (index, rule) in self.rules.iter().enumerate() {
                if instances.len() >= MAX_GROUNDINGS_PER_RULE * 4 {
                    break;
                }
                for (substitution, premises) in ground_rule_body(&rule.body, &universe) {
                    let head = apply_atom(&substitution, &rule.head);
                    if !head.is_ground() {
                        continue;
                    }
                    let key = format!("r{}\u{1}{}\u{1}{}", index, head.display(), render_premises(&premises));
                    if !seen.insert(key) {
                        continue;
                    }
                    let (condition, probability) = if rule.probability >= 1.0 - 1e-9 {
                        (TRUE_NODE, 1.0)
                    } else {
                        let var = bdd.new_var(
                            rule.probability,
                            format!(
                                "规则 {}（{:.2}）",
                                rule.source, rule.probability
                            ),
                        );
                        (bdd.literal(var, true), rule.probability)
                    };
                    new_heads.push(head.clone());
                    instances.push(GroundProbability {
                        head,
                        premises,
                        condition,
                        probability,
                        source: rule.source.clone(),
                    });
                }
            }
            for (index, ad) in self.annotated_disjunctions.iter().enumerate() {
                let total: f64 = ad.heads.iter().map(|(_, p)| p.max(0.0)).sum();
                if total <= 1e-9 {
                    continue;
                }
                for (substitution, premises) in ground_rule_body(&ad.body, &universe) {
                    let grounded: Vec<(Atom, f64)> = ad
                        .heads
                        .iter()
                        .filter_map(|(head, probability)| {
                            let atom = apply_atom(&substitution, head);
                            atom.is_ground()
                                .then_some((atom, probability.max(0.0) / total))
                        })
                        .filter(|(_, probability)| *probability > 1e-6)
                        .collect();
                    if grounded.is_empty() {
                        continue;
                    }
                    let key = format!(
                        "a{}\u{1}{}\u{1}{}",
                        index,
                        grounded
                            .iter()
                            .map(|(atom, _)| atom.display())
                            .collect::<Vec<_>>()
                            .join(";"),
                        render_premises(&premises)
                    );
                    if !seen.insert(key) {
                        continue;
                    }
                    let conditions = encode_annotated_disjunction(&grounded, &mut bdd, &ad.source);
                    for ((atom, probability), condition) in grounded.into_iter().zip(conditions) {
                        new_heads.push(atom.clone());
                        instances.push(GroundProbability {
                            head: atom,
                            premises: premises.clone(),
                            condition,
                            probability,
                            source: ad.source.clone(),
                        });
                    }
                }
            }
            let mut added = false;
            // 与 universe 同步的成员集合（按谓词惰性构建）
            let mut universe_sets: HashMap<String, HashSet<Atom>> = HashMap::new();
            for head in new_heads {
                let predicate = head.predicate.to_string();
                let known = universe_sets.entry(predicate.clone()).or_insert_with(|| {
                    universe
                        .get(&predicate)
                        .map(|atoms| atoms.iter().cloned().collect())
                        .unwrap_or_default()
                });
                if known.insert(head.clone()) {
                    universe.entry(predicate).or_default().push(head);
                    added = true;
                }
            }
            if !added {
                break;
            }
        }

        let mut atom_bdd: HashMap<Atom, u32> = HashMap::new();
        for fact in &deterministic {
            atom_bdd.insert(fact.clone(), TRUE_NODE);
        }
        for (atom, &node) in &prob_fact_bdd {
            atom_bdd.insert(atom.clone(), node);
        }
        for instance in &instances {
            atom_bdd.entry(instance.head.clone()).or_insert(FALSE_NODE);
            for premise in &instance.premises {
                atom_bdd.entry(premise.clone()).or_insert(FALSE_NODE);
            }
        }

        let max_iterations = instances
            .len()
            .saturating_add(atom_bdd.len())
            .saturating_add(4)
            .min(20_000);
        let mut exact = true;
        for _ in 0..max_iterations {
            let mut changed = false;
            for instance in &instances {
                let mut body = TRUE_NODE;
                for premise in &instance.premises {
                    let premise_node = atom_bdd.get(premise).copied().unwrap_or(FALSE_NODE);
                    if premise_node == FALSE_NODE {
                        body = FALSE_NODE;
                        break;
                    }
                    body = bdd.conjunction(body, premise_node);
                }
                if body == FALSE_NODE {
                    continue;
                }
                let contribution = bdd.conjunction(instance.condition, body);
                if contribution == FALSE_NODE {
                    continue;
                }
                let current = atom_bdd.get(&instance.head).copied().unwrap_or(FALSE_NODE);
                let merged = bdd.disjunction(current, contribution);
                if merged != current {
                    atom_bdd.insert(instance.head.clone(), merged);
                    changed = true;
                }
                if bdd.overflow() {
                    break;
                }
            }
            if bdd.overflow() {
                exact = false;
                break;
            }
            if !changed {
                break;
            }
        }
        if bdd.overflow() {
            exact = false;
        }

        let mut atom_probability: HashMap<Atom, f64> = HashMap::new();
        for (atom, &node) in &atom_bdd {
            if deterministic.contains(atom) {
                continue;
            }
            // 概率事实同样计入：事实与规则对同一原子是析取支持，
            // WMC 结果可能高于裸事实概率（查询侧区分 from_fact 展示）
            atom_probability.insert(atom.clone(), bdd.wmc(node));
        }

        let mut instances_by_head: HashMap<Atom, Vec<usize>> = HashMap::new();
        for (index, instance) in instances.iter().enumerate() {
            instances_by_head
                .entry(instance.head.clone())
                .or_default()
                .push(index);
        }
        for indices in instances_by_head.values_mut() {
            indices.sort_by_cached_key(|&index| {
                let instance = &instances[index];
                InstanceOrderKey {
                    probability: instance.probability,
                    source: instance.source.clone(),
                    premises: render_premises(&instance.premises),
                }
            });
        }

        Compiled {
            choice_variables: bdd.var_count(),
            bdd_nodes: bdd.node_count(),
            instances,
            instances_by_head,
            atom_probability,
            deterministic_facts: deterministic,
            probabilistic_facts: prob_fact_probability,
            exact,
            ground_rounds,
            max_depth: self.max_depth,
        }
    }
}

fn push_universe(universe: &mut HashMap<String, Vec<Atom>>, atom: &Atom) {
    // 已有同谓词桶时直接复用
    match universe.get_mut(atom.predicate.as_ref()) {
        Some(atoms) => atoms.push(atom.clone()),
        None => {
            universe.insert(atom.predicate.to_string(), vec![atom.clone()]);
        }
    }
}

fn render_premises(premises: &[Atom]) -> String {
    let mut out = String::new();
    for (index, atom) in premises.iter().enumerate() {
        if index > 0 {
            out.push('&');
        }
        out.push_str(&atom.display());
    }
    out
}

pub(crate) fn pattern_matches(pattern: &Atom, atom: &Atom) -> bool {
    if pattern.predicate != atom.predicate || pattern.arity() != atom.arity() {
        return false;
    }
    pattern
        .terms
        .iter()
        .zip(atom.terms.iter())
        .all(|(left, right)| match left {
            Term::Var(_) => true,
            Term::Const(name) => matches!(right, Term::Const(other) if name == other),
        })
}

/// 注释析取的顺序编码：每个结果一个条件 BDD，概率自动归一且互斥
fn encode_annotated_disjunction(
    outcomes: &[(Atom, f64)],
    bdd: &mut BddManager,
    source: &str,
) -> Vec<u32> {
    let mut conditions = Vec::with_capacity(outcomes.len());
    let mut prefix = TRUE_NODE;
    let mut remaining = 1.0_f64;
    let last = outcomes.len() - 1;
    for (index, (atom, probability)) in outcomes.iter().enumerate() {
        if index == last || remaining <= 1e-9 {
            conditions.push(prefix);
            break;
        }
        let conditional = (probability / remaining).clamp(1e-6, 1.0 - 1e-6);
        let var = bdd.new_var(
            conditional,
            format!(
                "注释析取 {} 的第 {} 个结果 {}",
                source,
                index + 1,
                atom.display()
            ),
        );
        let choice = bdd.literal(var, true);
        conditions.push(bdd.conjunction(prefix, choice));
        let unselected = bdd.literal(var, false);
        prefix = bdd.conjunction(prefix, unselected);
        remaining = (remaining - probability).max(1e-9);
    }
    while conditions.len() < outcomes.len() {
        conditions.push(FALSE_NODE);
    }
    conditions
}

/// 前件的半连接基化：返回（置换, 基化前件）。全基前件无条件通过（未知原子按假处理）。
fn ground_rule_body(
    body: &[Atom],
    universe: &HashMap<String, Vec<Atom>>,
) -> Vec<(Substitution, Vec<Atom>)> {
    if body.is_empty() {
        return vec![(Substitution::new(), Vec::new())];
    }
    if body.iter().all(|atom| atom.is_ground()) {
        return vec![(Substitution::new(), body.to_vec())];
    }
    let mut results = Vec::new();
    let mut substitution = Substitution::new();
    let mut premises = Vec::new();
    ground_step(
        body,
        0,
        &mut substitution,
        &mut premises,
        universe,
        &mut results,
    );
    results
}

fn ground_step(
    body: &[Atom],
    index: usize,
    substitution: &mut Substitution,
    premises: &mut Vec<Atom>,
    universe: &HashMap<String, Vec<Atom>>,
    results: &mut Vec<(Substitution, Vec<Atom>)>,
) {
    if results.len() >= MAX_GROUNDINGS_PER_RULE {
        return;
    }
    if index == body.len() {
        results.push((substitution.clone(), premises.clone()));
        return;
    }
    let pattern = &body[index];
    if pattern.is_ground() {
        premises.push(pattern.clone());
        ground_step(body, index + 1, substitution, premises, universe, results);
        premises.pop();
        return;
    }
    let Some(candidates) = universe.get(pattern.predicate.as_ref()) else {
        return;
    };
    for candidate in candidates {
        if results.len() >= MAX_GROUNDINGS_PER_RULE {
            return;
        }
        let mut extended = substitution.clone();
        if unify_atoms(pattern, candidate, &mut extended) {
            let ground = apply_atom(&extended, pattern);
            if ground.is_ground() {
                premises.push(ground);
                ground_step(body, index + 1, &mut extended, premises, universe, results);
                premises.pop();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fact(predicate: &str, subject: &str, object: &str) -> Atom {
        Atom::rel(predicate, subject, object)
    }

    fn rule_engine() -> ProblogEngine {
        ProblogEngine::new(6, 8)
    }

    #[test]
    fn test_bdd_wmc_matches_brute_force_enumeration() {
        let mut engine = rule_engine();
        engine.add_probabilistic_fact(fact("a", "x", "y"), 0.2);
        engine.add_probabilistic_fact(fact("b", "x", "y"), 0.5);
        engine.add_probabilistic_fact(fact("c", "x", "y"), 0.9);
        let head = Atom::pattern("d", Term::var("A"), Term::var("B"));
        for predicate in ["a", "b", "c"] {
            engine.add_rule(
                head.clone(),
                vec![Atom::pattern(predicate, Term::var("A"), Term::var("B"))],
                1.0,
                "析取规则",
            );
        }
        let query = engine.query(&Atom::pattern("d", Term::var("X"), Term::var("Y")), 5, 0.0);
        assert_eq!(query.answers.len(), 1);
        let expected = 1.0 - (1.0 - 0.2) * (1.0 - 0.5) * (1.0 - 0.9);
        assert!(
            (query.answers[0].probability - expected).abs() < 1e-9,
            "BDD WMC {} 应等于枚举值 {}",
            query.answers[0].probability,
            expected
        );
        assert!(query.exact);
    }

    #[test]
    fn test_probabilistic_rule_chain_multiplies() {
        let mut engine = rule_engine();
        engine.add_deterministic_fact(fact("p", "x", "y"));
        engine.add_rule(
            fact("a", "x", "y"),
            vec![fact("p", "x", "y")],
            0.6,
            "p⇒a",
        );
        engine.add_rule(
            fact("b", "x", "y"),
            vec![fact("a", "x", "y")],
            0.5,
            "a⇒b",
        );
        let query = engine.query(
            &Atom::pattern("b", Term::constant("x"), Term::var("Y")),
            5,
            0.0,
        );
        assert_eq!(query.answers.len(), 1);
        assert!((query.answers[0].probability - 0.3).abs() < 1e-9);
        assert_eq!(query.answers[0].proofs.len(), 1);
        assert!((query.answers[0].proofs[0].probability - 0.3).abs() < 1e-9);
    }

    #[test]
    fn test_probabilistic_premise_fact_contributes_probability_to_proof() {
        let mut engine = rule_engine();
        engine.add_probabilistic_fact(fact("p", "x", "y"), 0.8);
        engine.add_rule(
            fact("a", "x", "y"),
            vec![fact("p", "x", "y")],
            0.5,
            "p⇒a",
        );
        let query = engine.query(&fact("a", "x", "y"), 5, 0.0);
        assert_eq!(query.answers.len(), 1);
        assert!((query.answers[0].probability - 0.4).abs() < 1e-9);
        assert_eq!(query.answers[0].proofs.len(), 1);
        assert!(
            (query.answers[0].proofs[0].probability - 0.4).abs() < 1e-9,
            "证明路径应包含概率事实的 0.8（实得 {}）",
            query.answers[0].proofs[0].probability
        );
    }

    #[test]
    fn test_deterministic_fact_wins_over_probabilistic_in_query() {
        let mut engine = rule_engine();
        engine.add_probabilistic_fact(fact("p", "x", "y"), 0.2);
        engine.add_deterministic_fact(fact("p", "x", "y"));
        let query = engine.query(&fact("p", "x", "y"), 5, 0.0);
        assert_eq!(query.answers.len(), 1);
        assert!((query.answers[0].probability - 1.0).abs() < 1e-9);
        assert!(query.answers[0].from_fact);
    }

    #[test]
    fn test_duplicate_probabilistic_facts_merge_as_disjunction() {
        let mut engine = rule_engine();
        engine.add_probabilistic_fact(fact("q", "x", "y"), 0.3);
        engine.add_probabilistic_fact(fact("q", "x", "y"), 0.4);
        let expected = 1.0 - (1.0 - 0.3) * (1.0 - 0.4);
        let q = engine.query(&fact("q", "x", "y"), 5, 0.0);
        assert_eq!(q.answers.len(), 1);
        assert!(
            (q.answers[0].probability - expected).abs() < 1e-9,
            "实得 {}",
            q.answers[0].probability
        );
        assert!(
            q.answers[0]
                .proofs
                .iter()
                .any(|proof| (proof.probability - expected).abs() < 1e-9),
            "裸事实证明的概率也应按析取合并：{:?}",
            q.answers[0].proofs
        );
    }

    #[test]
    fn test_query_merges_rule_support_for_probabilistic_fact() {
        let mut engine = rule_engine();
        engine.add_probabilistic_fact(fact("q", "x", "y"), 0.3);
        engine.add_deterministic_fact(fact("t", "x", "y"));
        engine.add_rule(
            fact("q", "x", "y"),
            vec![fact("t", "x", "y")],
            0.9,
            "t⇒q",
        );
        engine.add_rule(
            fact("r", "x", "y"),
            vec![fact("q", "x", "y")],
            1.0,
            "q⇒r",
        );
        let expected = 1.0 - (1.0 - 0.3) * (1.0 - 0.9);
        let q = engine.query(&fact("q", "x", "y"), 5, 0.0);
        assert_eq!(q.answers.len(), 1);
        assert!(q.answers[0].from_fact);
        assert!(
            (q.answers[0].probability - expected).abs() < 1e-9,
            "事实与规则应析取合并，实得 {}",
            q.answers[0].probability
        );
        let r = engine.query(&fact("r", "x", "y"), 5, 0.0);
        assert!(
            (r.answers[0].probability - expected).abs() < 1e-9,
            "下游原子概率应与合并后的 q 一致，实得 {}",
            r.answers[0].probability
        );
        // min_probability 过滤应基于合并后的精确概率
        let filtered = engine.query(&fact("q", "x", "y"), 5, 0.5);
        assert_eq!(filtered.answers.len(), 1);
    }

    #[test]
    fn test_annotated_disjunction_is_mutually_exclusive() {
        let mut engine = rule_engine();
        engine.add_annotated_disjunction(
            vec![(fact("晴", "今天", "是"), 0.3), (fact("雨", "今天", "是"), 0.7)],
            Vec::new(),
            "天气注释析取",
        );
        engine.add_rule(
            fact("带伞", "我", "是"),
            vec![fact("雨", "今天", "是")],
            0.5,
            "雨⇒带伞",
        );
        let sunny = engine.query(&fact("晴", "今天", "是"), 1, 0.0);
        let rainy = engine.query(&fact("雨", "今天", "是"), 1, 0.0);
        assert!((sunny.answers[0].probability - 0.3).abs() < 1e-9);
        assert!((rainy.answers[0].probability - 0.7).abs() < 1e-9);
        // 互斥：P(晴)+P(雨)=1（无重叠），下游 = 0.7×0.5
        let umbrella = engine.query(&fact("带伞", "我", "是"), 1, 0.0);
        assert!((umbrella.answers[0].probability - 0.35).abs() < 1e-9);
    }

    #[test]
    fn test_shared_premise_exact_correlation_via_bdd() {
        let mut engine = rule_engine();
        engine.add_probabilistic_fact(fact("c", "x", "y"), 0.5);
        engine.add_rule(
            fact("a", "x", "y"),
            vec![fact("c", "x", "y")],
            0.6,
            "c⇒a",
        );
        engine.add_rule(
            fact("b", "x", "y"),
            vec![fact("c", "x", "y")],
            0.5,
            "c⇒b",
        );
        engine.add_rule(
            fact("ab", "x", "y"),
            vec![fact("a", "x", "y"), fact("b", "x", "y")],
            1.0,
            "a∧b⇒ab",
        );
        let query = engine.query(&fact("ab", "x", "y"), 1, 0.0);
        // 相关源头：P(ab)=P(c)·p1·p2=0.15，而独立假设会得到 0.5·0.3·0.25=0.0375
        assert!(
            (query.answers[0].probability - 0.15).abs() < 1e-9,
            "实得 {}",
            query.answers[0].probability
        );
    }

    #[test]
    fn test_cyclic_positive_program_converges_to_least_fixpoint() {
        let mut engine = rule_engine();
        engine.add_deterministic_fact(fact("base", "x", "y"));
        engine.add_rule(
            fact("a", "x", "y"),
            vec![fact("base", "x", "y")],
            0.5,
            "base⇒a",
        );
        engine.add_rule(
            fact("a", "x", "y"),
            vec![fact("b", "x", "y")],
            0.5,
            "b⇒a",
        );
        engine.add_rule(
            fact("b", "x", "y"),
            vec![fact("a", "x", "y")],
            0.5,
            "a⇒b",
        );
        let query = engine.query(
            &Atom::pattern("a", Term::constant("x"), Term::var("Y")),
            1,
            0.0,
        );
        assert!((query.answers[0].probability - 0.5).abs() < 1e-9);
        let query_b = engine.query(
            &Atom::pattern("b", Term::constant("x"), Term::var("Y")),
            1,
            0.0,
        );
        assert!((query_b.answers[0].probability - 0.25).abs() < 1e-9);
    }

    #[test]
    fn test_from_markov_engine_preserves_probability() {
        use crate::markov_logic::WeightedRule;
        let mut markov = MarkovLogicEngine::new(4, 0.5);
        markov.add_fact(fact("师傅", "张三", "李四"));
        markov.add_rule(WeightedRule::new(
            Atom::pattern("尊敬", Term::var("A"), Term::var("B")),
            vec![Atom::pattern("师傅", Term::var("A"), Term::var("B"))],
            2.0,
            "师傅⇒尊敬",
        ));
        let engine = ProblogEngine::from_markov_engine(&markov, 6, 8);
        let query = engine.query(
            &Atom::pattern("尊敬", Term::var("X"), Term::var("Y")),
            5,
            0.0,
        );
        assert_eq!(query.answers.len(), 1);
        let expected = 1.0 / (1.0 + (-2.0f64).exp());
        assert!((query.answers[0].probability - expected).abs() < 1e-9);
    }

    #[test]
    fn test_query_pattern_and_min_probability_filter() {
        let mut engine = rule_engine();
        engine.add_probabilistic_fact(fact("p", "a", "b"), 0.1);
        engine.add_probabilistic_fact(fact("p", "a", "c"), 0.9);
        let query = engine.query(
            &Atom::pattern("p", Term::constant("a"), Term::var("X")),
            5,
            0.5,
        );
        assert_eq!(query.answers.len(), 1);
        assert_eq!(query.answers[0].atom.display(), "p(a, c)");
    }

    #[test]
    fn test_unary_probabilistic_fact_and_ground_rule() {
        let mut engine = rule_engine();
        engine.add_probabilistic_fact(Atom::unary("会飞", Term::constant("鸟")), 0.8);
        engine.add_rule(
            Atom::unary("能上天", Term::constant("鸟")),
            vec![Atom::unary("会飞", Term::constant("鸟"))],
            0.9,
            "会飞⇒能上天",
        );
        let query = engine.query(
            &Atom::pattern("能上天", Term::constant("鸟"), Term::var("X")),
            5,
            0.0,
        );
        // 查询元数与事实元数不同：不匹配
        assert!(query.answers.is_empty());
        let query = engine.query(
            &Atom::unary("能上天", Term::constant("鸟")),
            5,
            0.0,
        );
        assert!((query.answers[0].probability - 0.72).abs() < 1e-9);
    }
}

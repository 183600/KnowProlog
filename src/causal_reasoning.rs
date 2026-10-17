// 因果推理（纯符号）：因果图、do-演算干预、反事实比较、混杂因子检测
//
// 灵感：
//   • 因果推理：知识图谱里的加权规则天然带方向——把每条规则的
//     前件 → 后件看作一条有符号的因果机制（正权=促进，负权=抑制），
//     就得到一张可解释、可回放的因果图；
//   • do-演算：do(X=x) 的符号实现 = 把 X 的真值钉住 + 切断指向 X 的规则，
//     其余规则继续做 MLN-lite 软真值传播，于是 P(Y|do(X=x)) 直接可算；
//   • 反事实：对同一效应模式分别跑"事实世界"与"干预世界"，比较 ΔP；
//   • 混杂因子：在因果图上找共同原因（Z→X 且 Z→Y），提示相关≠因果。
//
// 全部运算是合一、图搜索与既有软真值传播：无神经网络、无参数训练。

use std::collections::{HashMap, HashSet};

use crate::markov_logic::{Beliefs, GroundedRule, MarkovLogicEngine};
use crate::resolution_engine::Atom;

/// 默认因果链最大长度（防组合爆炸）
pub const DEFAULT_MAX_CAUSAL_DEPTH: usize = 4;

/// 单次查询最多枚举的简单路径数
const MAX_PATHS_PER_QUERY: usize = 64;

/// 强度降序、同强度按渲染串升序的排序键：预先算好避免比较器里重复渲染
#[derive(PartialEq)]
struct StrengthRenderKey {
    strength: f64,
    render: String,
}

impl Eq for StrengthRenderKey {}

impl PartialOrd for StrengthRenderKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for StrengthRenderKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .strength
            .partial_cmp(&self.strength)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| self.render.cmp(&other.render))
    }
}

/// 绝对值降序、同值按显示串升序的排序键
#[derive(PartialEq)]
struct MagnitudeDisplayKey {
    magnitude: f64,
    display: String,
}

impl Eq for MagnitudeDisplayKey {}

impl PartialOrd for MagnitudeDisplayKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MagnitudeDisplayKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .magnitude
            .partial_cmp(&self.magnitude)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| self.display.cmp(&other.display))
    }
}

/// 因果边：一条基化规则贡献的 "前件原子 → 后件原子" 有符号机制
#[derive(Debug, Clone, PartialEq)]
pub struct CausalEdge {
    pub cause: Atom,
    pub effect: Atom,
    /// 对数几率权重：正=促进，负=抑制
    pub weight: f64,
    pub source: String,
    pub rule_text: String,
}

impl CausalEdge {
    pub fn is_inhibitory(&self) -> bool {
        self.weight < 0.0
    }

    /// 形如 `师傅(张三, 李四) —[+2.20]→ 师祖(张三, 王五)`
    pub fn text(&self) -> String {
        format!(
            "{} —[{:+.2}]→ {}",
            self.cause.display(),
            self.weight,
            self.effect.display()
        )
    }
}

/// 一条因果链（节点序列 + 各段有符号权重）
#[derive(Debug, Clone)]
pub struct CausalPath {
    nodes: Vec<Atom>,
    weights: Vec<f64>,
}

impl CausalPath {
    pub fn nodes(&self) -> &[Atom] {
        &self.nodes
    }

    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// 链式因果强度：log-odds 沿路径可加，故取权重和
    pub fn strength(&self) -> f64 {
        self.weights.iter().sum()
    }

    /// 链上是否含抑制机制（负权边）
    pub fn inhibitory(&self) -> bool {
        self.weights.iter().any(|weight| *weight < 0.0)
    }

    pub fn render(&self) -> String {
        let mut out = self
            .nodes
            .first()
            .map(|atom| atom.display())
            .unwrap_or_default();
        for (index, weight) in self.weights.iter().enumerate() {
            out.push_str(&format!(
                " —[{:+.2}]→ {}",
                weight,
                self.nodes[index + 1].display()
            ));
        }
        out
    }
}

/// 因果图：由软逻辑引擎的全部基化规则构建（只含可回放的基原子边）
#[derive(Debug, Clone)]
pub struct CausalGraph {
    edges: Vec<CausalEdge>,
    by_cause: HashMap<Atom, Vec<usize>>,
    by_effect: HashMap<Atom, Vec<usize>>,
    max_depth: usize,
}

impl CausalGraph {
    pub fn from_engine(engine: &MarkovLogicEngine, max_depth: usize) -> Self {
        let mut graph = Self {
            edges: Vec::new(),
            by_cause: HashMap::new(),
            by_effect: HashMap::new(),
            max_depth: max_depth.clamp(1, 8),
        };
        for grounded in engine.ground_all() {
            graph.add_grounded(&grounded);
        }
        graph
    }

    fn add_grounded(&mut self, grounded: &GroundedRule) {
        for premise in &grounded.premises {
            let index = self.edges.len();
            self.by_cause
                .entry(premise.clone())
                .or_default()
                .push(index);
            self.by_effect
                .entry(grounded.head.clone())
                .or_default()
                .push(index);
            self.edges.push(CausalEdge {
                cause: premise.clone(),
                effect: grounded.head.clone(),
                weight: grounded.weight,
                source: grounded.source.clone(),
                rule_text: grounded.rule_text.clone(),
            });
        }
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn node_count(&self) -> usize {
        let mut nodes: HashSet<&Atom> = HashSet::new();
        for edge in &self.edges {
            nodes.insert(&edge.cause);
            nodes.insert(&edge.effect);
        }
        nodes.len()
    }

    pub fn edges(&self) -> &[CausalEdge] {
        &self.edges
    }

    /// 指向 effect 的直接因果输入
    pub fn causes_of(&self, effect: &Atom) -> Vec<&CausalEdge> {
        self.by_effect
            .get(effect)
            .map(|indices| indices.iter().map(|index| &self.edges[*index]).collect())
            .unwrap_or_default()
    }

    /// 由 cause 直接引发的后果
    pub fn effects_of(&self, cause: &Atom) -> Vec<&CausalEdge> {
        self.by_cause
            .get(cause)
            .map(|indices| indices.iter().map(|index| &self.edges[*index]).collect())
            .unwrap_or_default()
    }

    /// 从 from 到 to 的全部简单有向因果链（按链强度降序）
    pub fn causal_paths(&self, from: &Atom, to: &Atom) -> Vec<CausalPath> {
        let mut out = Vec::new();
        let mut visited: HashSet<Atom> = HashSet::new();
        visited.insert(from.clone());
        let mut nodes = vec![from.clone()];
        let mut weights = Vec::new();
        self.walk_forward(to, &mut visited, &mut nodes, &mut weights, &mut out);
        Self::sort_paths(&mut out);
        out
    }

    fn walk_forward(
        &self,
        to: &Atom,
        visited: &mut HashSet<Atom>,
        nodes: &mut Vec<Atom>,
        weights: &mut Vec<f64>,
        out: &mut Vec<CausalPath>,
    ) {
        if out.len() >= MAX_PATHS_PER_QUERY || nodes.len() > self.max_depth + 1 {
            return;
        }
        let current = nodes.last().cloned();
        let Some(current) = current else {
            return;
        };
        let Some(indices) = self.by_cause.get(&current) else {
            return;
        };
        for &index in indices {
            let edge = &self.edges[index];
            if !visited.insert(edge.effect.clone()) {
                continue;
            }
            nodes.push(edge.effect.clone());
            weights.push(edge.weight);
            if edge.effect == *to {
                out.push(CausalPath {
                    nodes: nodes.clone(),
                    weights: weights.clone(),
                });
            } else {
                self.walk_forward(to, visited, nodes, weights, out);
            }
            weights.pop();
            nodes.pop();
            visited.remove(&edge.effect);
        }
    }

    /// 从事实根因（roots）出发到 target 的全部因果链（按强度降序）
    pub fn root_paths(&self, target: &Atom, roots: &HashSet<Atom>) -> Vec<CausalPath> {
        let mut out = Vec::new();
        let mut visited: HashSet<Atom> = HashSet::new();
        visited.insert(target.clone());
        let mut reverse_nodes = vec![target.clone()];
        let mut reverse_weights: Vec<f64> = Vec::new();
        self.walk_backward(
            roots,
            &mut visited,
            &mut reverse_nodes,
            &mut reverse_weights,
            &mut out,
        );
        Self::sort_paths(&mut out);
        out
    }

    fn walk_backward(
        &self,
        roots: &HashSet<Atom>,
        visited: &mut HashSet<Atom>,
        reverse_nodes: &mut Vec<Atom>,
        reverse_weights: &mut Vec<f64>,
        out: &mut Vec<CausalPath>,
    ) {
        if out.len() >= MAX_PATHS_PER_QUERY || reverse_nodes.len() > self.max_depth + 1 {
            return;
        }
        let current = reverse_nodes.last().cloned().expect("递归时序列非空");
        if roots.contains(&current) && reverse_nodes.len() > 1 {
            let nodes: Vec<Atom> = reverse_nodes.iter().rev().cloned().collect();
            let weights: Vec<f64> = reverse_weights.iter().rev().cloned().collect();
            out.push(CausalPath { nodes, weights });
            return;
        }
        let Some(indices) = self.by_effect.get(&current) else {
            return;
        };
        let mut seen_causes: HashSet<Atom> = HashSet::new();
        for &index in indices {
            let edge = &self.edges[index];
            if !seen_causes.insert(edge.cause.clone()) {
                continue;
            }
            if !visited.insert(edge.cause.clone()) {
                continue;
            }
            reverse_nodes.push(edge.cause.clone());
            reverse_weights.push(edge.weight);
            self.walk_backward(roots, visited, reverse_nodes, reverse_weights, out);
            reverse_weights.pop();
            reverse_nodes.pop();
            visited.remove(&edge.cause);
        }
    }

    fn sort_paths(paths: &mut [CausalPath]) {
        paths.sort_by_cached_key(|path| StrengthRenderKey {
            strength: path.strength(),
            render: path.render(),
        });
    }

    /// 混杂因子检测：同时指向 cause 与 effect 的共同原因 Z（Z→cause 且 Z→effect）
    pub fn confounders(&self, cause: &Atom, effect: &Atom) -> Vec<Atom> {
        let mut out = Vec::new();
        let Some(parent_indices) = self.by_effect.get(cause) else {
            return out;
        };
        for &index in parent_indices {
            let candidate = &self.edges[index].cause;
            if candidate == cause || candidate == effect {
                continue;
            }
            let connects_to_effect = self
                .by_cause
                .get(candidate)
                .map(|indices| {
                    indices
                        .iter()
                        .any(|edge_index| self.edges[*edge_index].effect == *effect)
                })
                .unwrap_or(false);
            if connects_to_effect && !out.contains(candidate) {
                out.push(candidate.clone());
            }
        }
        out
    }
}

/// 目标结论的一条直接因果输入
#[derive(Debug, Clone)]
pub struct CausalLink {
    pub cause: Atom,
    pub cause_probability: f64,
    pub weight: f64,
    pub source: String,
    pub rule_text: String,
}

impl CausalLink {
    pub fn is_inhibitory(&self) -> bool {
        self.weight < 0.0
    }
}

/// 结论的因果解释：概率 + 直接原因 + 事实根因链 + 混杂因子
#[derive(Debug, Clone)]
pub struct CausalExplanation {
    pub target: Atom,
    pub probability: f64,
    pub direct_causes: Vec<CausalLink>,
    pub chains: Vec<CausalPath>,
    pub confounders: Vec<Atom>,
}

/// 单条原子的反事实对比
#[derive(Debug, Clone)]
pub struct CounterfactualDelta {
    pub atom: Atom,
    pub factual: f64,
    pub counterfactual: f64,
    pub delta: f64,
}

/// 一次 do(X=x) 干预下的反事实比较
#[derive(Debug, Clone)]
pub struct CounterfactualComparison {
    pub intervention: Atom,
    pub value: bool,
    pub deltas: Vec<CounterfactualDelta>,
}

/// 因果分析器：持有因果图，按需借用软逻辑引擎做概率计算
pub struct CausalAnalyzer {
    graph: CausalGraph,
}

impl CausalAnalyzer {
    pub fn from_engine(engine: &MarkovLogicEngine, max_depth: usize) -> Self {
        Self {
            graph: CausalGraph::from_engine(engine, max_depth),
        }
    }

    /// 与 `from_engine` 相同，但复用调用方已算好的概率视图（省去重复的软真值传播）
    pub fn from_engine_with_probabilities(
        engine: &MarkovLogicEngine,
        probabilities: &HashMap<Atom, f64>,
        max_depth: usize,
    ) -> Self {
        let mut graph = CausalGraph {
            edges: Vec::new(),
            by_cause: HashMap::new(),
            by_effect: HashMap::new(),
            max_depth: max_depth.clamp(1, 8),
        };
        for grounded in engine.ground_all_with_probabilities(probabilities) {
            graph.add_grounded(&grounded);
        }
        Self { graph }
    }

    pub fn graph(&self) -> &CausalGraph {
        &self.graph
    }

    /// 解释一个基原子：当前概率、直接原因（含权重/来源）、事实根因链、混杂因子
    pub fn explain(
        &self,
        engine: &MarkovLogicEngine,
        target: &Atom,
        max_chains: usize,
    ) -> CausalExplanation {
        let probabilities = engine.probability_map();
        self.explain_with_probabilities(engine, &probabilities, target, max_chains)
    }

    /// 与 `explain` 相同，但概率视图由调用方提供，可跨多个目标复用
    pub fn explain_with_probabilities(
        &self,
        engine: &MarkovLogicEngine,
        probabilities: &HashMap<Atom, f64>,
        target: &Atom,
        max_chains: usize,
    ) -> CausalExplanation {
        let probability = if engine.has_fact(target) {
            1.0
        } else {
            probabilities.get(target).copied().unwrap_or(0.0)
        };
        let mut direct_causes: Vec<CausalLink> = self
            .graph
            .causes_of(target)
            .into_iter()
            .map(|edge| {
                let cause_probability = if engine.has_fact(&edge.cause) {
                    1.0
                } else {
                    probabilities.get(&edge.cause).copied().unwrap_or(0.0)
                };
                CausalLink {
                    cause: edge.cause.clone(),
                    cause_probability,
                    weight: edge.weight,
                    source: edge.source.clone(),
                    rule_text: edge.rule_text.clone(),
                }
            })
            .collect();
        direct_causes.sort_by_cached_key(|link| MagnitudeDisplayKey {
            magnitude: link.weight.abs(),
            display: link.cause.display(),
        });
        let roots: HashSet<Atom> = engine.facts().cloned().collect();
        let mut chains = self.graph.root_paths(target, &roots);
        chains.truncate(max_chains.max(1));
        let confounders = direct_causes
            .first()
            .map(|link| self.graph.confounders(&link.cause, target))
            .unwrap_or_default();
        CausalExplanation {
            target: target.clone(),
            probability,
            direct_causes,
            chains,
            confounders,
        }
    }

    /// do(intervention=value) 下的反事实比较：同一效应模式在两个世界里的概率差
    pub fn counterfactual(
        &self,
        engine: &MarkovLogicEngine,
        effect_pattern: &Atom,
        intervention: &Atom,
        value: bool,
        limit: usize,
    ) -> CounterfactualComparison {
        let mut fixed = HashMap::new();
        fixed.insert(intervention.clone(), if value { 1.0 } else { 0.0 });
        let factual_beliefs = engine.propagate_with_fixed(&HashMap::new());
        let counterfactual_beliefs = engine.propagate_with_fixed(&fixed);
        self.counterfactual_from_beliefs(
            engine,
            &factual_beliefs,
            &counterfactual_beliefs,
            &fixed,
            effect_pattern,
            intervention,
            value,
            limit,
        )
    }

    /// 与 `counterfactual` 相同，但两个世界的软真值信念由调用方提供，
    /// 供多个效应探测点复用同一批传播结果
    pub fn counterfactual_from_beliefs(
        &self,
        engine: &MarkovLogicEngine,
        factual_beliefs: &Beliefs,
        counterfactual_beliefs: &Beliefs,
        fixed: &HashMap<Atom, f64>,
        effect_pattern: &Atom,
        intervention: &Atom,
        value: bool,
        limit: usize,
    ) -> CounterfactualComparison {
        let limit = limit.max(1);
        let factual = engine.infer_from_beliefs(
            effect_pattern,
            limit * 4,
            factual_beliefs,
            &HashMap::new(),
        );
        let counterfactual = engine.infer_from_beliefs(
            effect_pattern,
            limit * 4,
            counterfactual_beliefs,
            fixed,
        );
        let mut merged: HashMap<String, CounterfactualDelta> = HashMap::new();
        for fact in &factual {
            merged.insert(
                fact.atom.display(),
                CounterfactualDelta {
                    atom: fact.atom.clone(),
                    factual: fact.probability,
                    counterfactual: 0.0,
                    delta: 0.0,
                },
            );
        }
        for fact in &counterfactual {
            let entry = merged
                .entry(fact.atom.display())
                .or_insert_with(|| CounterfactualDelta {
                    atom: fact.atom.clone(),
                    factual: 0.0,
                    counterfactual: 0.0,
                    delta: 0.0,
                });
            entry.counterfactual = fact.probability;
        }
        for delta in merged.values_mut() {
            delta.delta = delta.counterfactual - delta.factual;
        }
        let mut deltas: Vec<CounterfactualDelta> = merged.into_values().collect();
        deltas.sort_by_cached_key(|delta| MagnitudeDisplayKey {
            magnitude: delta.delta.abs(),
            display: delta.atom.display(),
        });
        deltas.truncate(limit);
        CounterfactualComparison {
            intervention: intervention.clone(),
            value,
            deltas,
        }
    }

    /// 干预的直接下游探测点（去重后的直接后果原子）
    pub fn direct_effect_probes(&self, intervention: &Atom) -> Vec<Atom> {
        let mut probes = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for edge in self.graph.effects_of(intervention) {
            if seen.insert(edge.effect.display()) {
                probes.push(edge.effect.clone());
            }
        }
        probes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::markov_logic::WeightedRule;
    use crate::resolution_engine::Term;

    fn chain_engine() -> MarkovLogicEngine {
        let mut engine = MarkovLogicEngine::new(6, 0.5);
        engine.add_fact(Atom::rel("师傅", "张三", "李四"));
        engine.add_rule(WeightedRule::new(
            Atom::pattern("长辈", Term::var("A"), Term::var("B")),
            vec![Atom::pattern("师傅", Term::var("A"), Term::var("B"))],
            2.0,
            "师傅⇒长辈",
        ));
        engine.add_rule(WeightedRule::new(
            Atom::pattern("尊敬", Term::var("A"), Term::var("B")),
            vec![Atom::pattern("长辈", Term::var("A"), Term::var("B"))],
            1.5,
            "长辈⇒尊敬",
        ));
        engine
    }

    #[test]
    fn test_causal_graph_builds_signed_edges_and_paths() {
        let engine = chain_engine();
        let analyzer = CausalAnalyzer::from_engine(&engine, 4);
        let graph = analyzer.graph();
        assert_eq!(graph.edge_count(), 2);
        let root = Atom::rel("师傅", "张三", "李四");
        let middle = Atom::rel("长辈", "张三", "李四");
        let target = Atom::rel("尊敬", "张三", "李四");
        assert_eq!(graph.causes_of(&target).len(), 1);
        assert_eq!(graph.causes_of(&target)[0].cause, middle);
        let paths = graph.causal_paths(&root, &target);
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].nodes().len(), 3);
        assert!((paths[0].strength() - 3.5).abs() < 1e-9);
        assert!(!paths[0].inhibitory());
        let rendered = paths[0].render();
        assert!(rendered.contains("—[+2.00]→"), "实际：{}", rendered);
        assert!(rendered.contains("长辈(张三, 李四)"));
    }

    #[test]
    fn test_root_paths_and_explain_from_facts() {
        let engine = chain_engine();
        let analyzer = CausalAnalyzer::from_engine(&engine, 4);
        let target = Atom::rel("尊敬", "张三", "李四");
        let explanation = analyzer.explain(&engine, &target, 3);
        assert!(explanation.probability > 0.5 && explanation.probability < 0.9);
        assert_eq!(explanation.direct_causes.len(), 1);
        assert_eq!(
            explanation.direct_causes[0].cause.display(),
            "长辈(张三, 李四)"
        );
        assert_eq!(explanation.chains.len(), 1);
        assert_eq!(
            explanation.chains[0].nodes().first().map(|a| a.display()),
            Some("师傅(张三, 李四)".to_string())
        );
    }

    #[test]
    fn test_confounder_detection() {
        let mut engine = MarkovLogicEngine::new(4, 0.5);
        engine.add_fact(Atom::rel("同事", "张三", "李四"));
        engine.add_rule(WeightedRule::new(
            Atom::pattern("师傅", Term::var("A"), Term::var("B")),
            vec![Atom::pattern("同事", Term::var("A"), Term::var("B"))],
            2.0,
            "同事⇒师傅",
        ));
        engine.add_rule(WeightedRule::new(
            Atom::pattern("师兄弟", Term::var("A"), Term::var("B")),
            vec![Atom::pattern("同事", Term::var("A"), Term::var("B"))],
            2.0,
            "同事⇒师兄弟",
        ));
        engine.add_rule(WeightedRule::new(
            Atom::pattern("师兄弟", Term::var("A"), Term::var("B")),
            vec![Atom::pattern("师傅", Term::var("A"), Term::var("B"))],
            1.0,
            "师傅⇒师兄弟",
        ));
        let analyzer = CausalAnalyzer::from_engine(&engine, 4);
        let cause = Atom::rel("师傅", "张三", "李四");
        let effect = Atom::rel("师兄弟", "张三", "李四");
        let confounders = analyzer.graph().confounders(&cause, &effect);
        assert_eq!(confounders, vec![Atom::rel("同事", "张三", "李四")]);
    }

    #[test]
    fn test_counterfactual_do_false_removes_downstream_effect() {
        let engine = chain_engine();
        let analyzer = CausalAnalyzer::from_engine(&engine, 4);
        let pattern = Atom::pattern("尊敬", Term::var("X"), Term::var("Y"));
        let intervention = Atom::rel("师傅", "张三", "李四");
        let comparison = analyzer.counterfactual(&engine, &pattern, &intervention, false, 5);
        assert!(!comparison.deltas.is_empty());
        let delta = &comparison.deltas[0];
        assert_eq!(delta.atom.display(), "尊敬(张三, 李四)");
        assert!(delta.factual > 0.5);
        assert_eq!(delta.counterfactual, 0.0);
        assert!(
            delta.delta < -0.5,
            "移除原因应大幅降低结果：{}",
            delta.delta
        );
    }

    #[test]
    fn test_counterfactual_do_true_on_intermediate_raises_effect() {
        let engine = chain_engine();
        let analyzer = CausalAnalyzer::from_engine(&engine, 4);
        let pattern = Atom::pattern("尊敬", Term::var("X"), Term::var("Y"));
        let intervention = Atom::rel("长辈", "张三", "李四");
        let comparison = analyzer.counterfactual(&engine, &pattern, &intervention, true, 5);
        let delta = comparison
            .deltas
            .iter()
            .find(|delta| delta.atom.display() == "尊敬(张三, 李四)")
            .expect("干预中间节点应产生反事实结果");
        assert!(delta.counterfactual > delta.factual);
        assert!(delta.delta > 0.0);
        let probes = analyzer.direct_effect_probes(&intervention);
        assert_eq!(probes, vec![Atom::rel("尊敬", "张三", "李四")]);
    }

    #[test]
    fn test_inhibitory_edge_is_signed() {
        let mut engine = MarkovLogicEngine::new(4, 0.05);
        engine.add_fact(Atom::rel("争吵", "张三", "李四"));
        engine.add_rule(WeightedRule::new(
            Atom::pattern("信任", Term::var("A"), Term::var("B")),
            vec![Atom::pattern("争吵", Term::var("A"), Term::var("B"))],
            -2.0,
            "争吵⇒不信任",
        ));
        let analyzer = CausalAnalyzer::from_engine(&engine, 4);
        let target = Atom::rel("信任", "张三", "李四");
        let explanation = analyzer.explain(&engine, &target, 2);
        assert_eq!(explanation.direct_causes.len(), 1);
        assert!(explanation.direct_causes[0].is_inhibitory());
        assert!(explanation.probability < 0.5);
    }
}

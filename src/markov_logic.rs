// 轻量马尔可夫逻辑网络（Markov Logic Network, MLN 的符号化子集）
//
// 灵感：MLN 给一阶逻辑子句配权重，把"硬逻辑"变成"软逻辑"：
//   • 每条加权规则（body ⇒ head, weight）是一个软约束；
//   • 一个主语的 log-odds 由命中规则权重之和决定：P(head) = σ(Σ w_i · 前件满足度)；
//   • 权重可由纯计数估计 w = ln((success+α)/(failure+α))，不需要梯度下降；
//   • MAP 推断用加权局部搜索（MaxWalkSat 风格）翻转候选原子，最大化被满足规则的权重和。
//
// 全部运算是合一、集合交并、对数与 sigmoid —— 没有神经网络、没有权重训练。

use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use crate::resolution_engine::{apply_atom, unify_atoms, Atom, Substitution, Term};

/// 每条规则单轮最多保留的基化数（防组合爆炸）
pub const DEFAULT_MAX_GROUNDINGS_PER_RULE: usize = 500;

/// log-odds 截断，避免极端权重把概率压到 0/1
const LOG_ODDS_CLAMP: f64 = 6.0;

#[derive(Debug, Clone)]
pub struct WeightedRule {
    pub head: Atom,
    pub body: Vec<Atom>,
    /// 对数几率权重：正=支持 head，负=抑制 head
    pub weight: f64,
    pub source: String,
    /// 渲染文本在构造时缓存
    text: String,
}

impl WeightedRule {
    pub fn new(head: Atom, body: Vec<Atom>, weight: f64, source: &str) -> Self {
        let weight = weight.clamp(-LOG_ODDS_CLAMP, LOG_ODDS_CLAMP);
        let text = Self::render_text(&head, &body, weight);
        Self {
            head,
            body,
            weight,
            source: source.to_string(),
            text,
        }
    }

    fn render_text(head: &Atom, body: &[Atom], weight: f64) -> String {
        let body = body
            .iter()
            .map(|atom| atom.display())
            .collect::<Vec<_>>()
            .join(" ∧ ");
        format!("{} ⇒ {}（权重 {:+.2}）", body, head.display(), weight)
    }

    /// 形如 `师傅(?A, ?B) ⇒ 师祖(?A, ?B)（权重 +2.20）`
    pub fn text(&self) -> String {
        self.text.clone()
    }
}

/// 概率降序、同概率按显示串升序的排序键
#[derive(PartialEq)]
struct ProbabilityDisplayKey {
    probability: f64,
    display: String,
}

impl Eq for ProbabilityDisplayKey {}

impl PartialOrd for ProbabilityDisplayKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ProbabilityDisplayKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .probability
            .partial_cmp(&self.probability)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| self.display.cmp(&other.display))
    }
}

/// 验证优先级降序、同优先级按显示串升序的排序键
#[derive(PartialEq)]
struct PriorityDisplayKey {
    priority: f64,
    display: String,
}

impl Eq for PriorityDisplayKey {}

impl PartialOrd for PriorityDisplayKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityDisplayKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .priority
            .partial_cmp(&self.priority)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| self.display.cmp(&other.display))
    }
}

/// 某条规则对某个候选事实的证据
#[derive(Debug, Clone)]
pub struct RuleEvidence {
    pub source: String,
    pub rule_text: String,
    pub weight: f64,
    pub premises: Vec<Atom>,
}

/// 软推理结果：事实 + 概率 + 证据规则
#[derive(Debug, Clone)]
pub struct SoftFact {
    pub atom: Atom,
    pub probability: f64,
    pub log_odds: f64,
    pub evidence: Vec<RuleEvidence>,
    pub from_fact: bool,
}

/// 一条规则的基化实例：前件全为基原子、后件为基原子，附带软合取满足度
#[derive(Debug, Clone)]
pub struct GroundedRule {
    pub head: Atom,
    pub premises: Vec<Atom>,
    pub weight: f64,
    pub source: String,
    pub rule_text: String,
    /// 前件满足度（软合取：各前提软真值取 min）
    pub satisfaction: f64,
}

/// 某条基化规则对结论的符号梯度贡献：∂P/∂w = P(1−P)·s
#[derive(Debug, Clone)]
pub struct RuleContribution {
    pub source: String,
    pub rule_text: String,
    pub weight: f64,
    pub satisfaction: f64,
    /// 结论概率对该规则权重的偏导（纯符号解析求导，不用于训练）
    pub gradient: f64,
    pub premises: Vec<Atom>,
}

/// 前提的敏感度：∂P/∂s = P(1−P)·w，以及"若前提被证实"的期望增益
#[derive(Debug, Clone)]
pub struct PremiseSensitivity {
    pub premise: Atom,
    /// 当前软真值（事实为 1.0）
    pub truth: f64,
    /// 结论概率对前提真值的偏导
    pub gradient: f64,
    /// 求证优先级 = |∂P/∂s|·(1−truth)：越不确定且影响越大的前提前越值得先问
    pub verify_priority: f64,
}

/// 一个结论的符号梯度归因（可微推理的纯符号版本：只解释、不训练）
#[derive(Debug, Clone)]
pub struct GradientAttribution {
    pub atom: Atom,
    pub probability: f64,
    pub log_odds: f64,
    pub contributions: Vec<RuleContribution>,
    pub premise_sensitivity: Vec<PremiseSensitivity>,
}

/// 软真值传播的中间结果：候选原子 → (log-odds, 证据)
pub(crate) type Beliefs = HashMap<Atom, (f64, Vec<RuleEvidence>)>;

pub struct MarkovLogicEngine {
    facts: HashSet<Atom>,
    facts_by_pred: HashMap<Rc<str>, Vec<Atom>>,
    rules: Vec<WeightedRule>,
    max_iterations: usize,
    min_probability: f64,
    max_groundings_per_rule: usize,
}

impl MarkovLogicEngine {
    pub fn new(max_iterations: usize, min_probability: f64) -> Self {
        Self {
            facts: HashSet::new(),
            facts_by_pred: HashMap::new(),
            rules: Vec::new(),
            max_iterations: max_iterations.clamp(1, 32),
            min_probability: min_probability.clamp(0.0, 1.0),
            max_groundings_per_rule: DEFAULT_MAX_GROUNDINGS_PER_RULE,
        }
    }

    pub fn fact_count(&self) -> usize {
        self.facts.len()
    }

    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    pub fn min_probability(&self) -> f64 {
        self.min_probability
    }

    pub fn facts(&self) -> impl Iterator<Item = &Atom> {
        self.facts.iter()
    }

    pub fn rules(&self) -> &[WeightedRule] {
        &self.rules
    }

    pub fn add_fact(&mut self, atom: Atom) {
        if !atom.is_ground() {
            return;
        }
        if self.facts.insert(atom.clone()) {
            self.facts_by_pred
                .entry(atom.predicate.clone())
                .or_default()
                .push(atom);
        }
    }

    pub fn add_rule(&mut self, rule: WeightedRule) {
        if rule.body.len() > 4 {
            return;
        }
        self.rules.push(rule);
    }

    /// 由规则置信度换算 log-odds 权重：c=0.9 → +2.20；c=0.5 → 0
    pub fn weight_from_confidence(confidence: f64) -> f64 {
        let c = confidence.clamp(1e-3, 1.0 - 1e-3);
        (c / (1.0 - c)).ln().clamp(-LOG_ODDS_CLAMP, LOG_ODDS_CLAMP)
    }

    /// 纯计数权重估计（伪似然式）：w = ln((success+α)/(failure+α))
    pub fn weight_from_counts(successes: usize, failures: usize, alpha: f64) -> f64 {
        let alpha = alpha.max(1e-6);
        let successes = successes as f64 + alpha;
        let failures = failures as f64 + alpha;
        (successes / failures)
            .ln()
            .clamp(-LOG_ODDS_CLAMP, LOG_ODDS_CLAMP)
    }

    /// 概率 → 中文模糊限定词（生成"不确定但可解释"的回答）
    pub fn hedge(probability: f64) -> &'static str {
        if probability >= 0.9 {
            "极有可能"
        } else if probability >= 0.75 {
            "很可能"
        } else if probability >= 0.6 {
            "比较可能"
        } else if probability >= 0.5 {
            "有可能"
        } else if probability >= 0.35 {
            "不太确定，或许"
        } else {
            "可能性较低"
        }
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x.clamp(-LOG_ODDS_CLAMP, LOG_ODDS_CLAMP)).exp())
    }

    /// 合并事实、"软真值 ≥ 0.5"的已推出原子与被干预原子，作为下一轮基化的论域。
    ///
    /// 事实由 `self.facts_by_pred` 直接引用，不复制整张事实表；本函数只返回
    /// 事实之外的补充原子（已去重、去事实重复），按谓词分组。
    fn extra_universe(
        &self,
        derived: &HashMap<Atom, f64>,
        fixed: &HashMap<Atom, f64>,
    ) -> HashMap<Rc<str>, Vec<Atom>> {
        let mut extra: HashMap<Rc<str>, Vec<Atom>> = HashMap::new();
        // 去重：同一原子可能既是事实又已被推出（或多条路径推出同一原子），
        // 重复候选会让同一条基化规则被重复计数、人为抬高 log-odds
        let mut seen: HashSet<&Atom> = HashSet::new();
        for (atom, truth) in derived {
            if *truth >= 0.5 && !self.facts.contains(atom) && seen.insert(atom) {
                extra
                    .entry(atom.predicate.clone())
                    .or_default()
                    .push(atom.clone());
            }
        }
        for atom in fixed.keys() {
            if !self.facts.contains(atom) && seen.insert(atom) {
                extra
                    .entry(atom.predicate.clone())
                    .or_default()
                    .push(atom.clone());
            }
        }
        extra
    }

    /// 前件的软真值：被干预原子取干预值 > 事实 1.0 > 已推出原子当前软真值
    fn premise_truth(
        &self,
        premise: &Atom,
        derived: &HashMap<Atom, f64>,
        fixed: &HashMap<Atom, f64>,
    ) -> f64 {
        if let Some(value) = fixed.get(premise) {
            return value.clamp(0.0, 1.0);
        }
        if self.facts.contains(premise) {
            return 1.0;
        }
        derived.get(premise).copied().unwrap_or(0.0)
    }

    /// 在当前信念（加可选干预）下基化全部规则，得到可复用的基化实例
    fn ground_rules_against(
        &self,
        derived: &HashMap<Atom, f64>,
        fixed: &HashMap<Atom, f64>,
    ) -> Vec<GroundedRule> {
        let extra = self.extra_universe(derived, fixed);
        let mut results = Vec::new();
        for rule in &self.rules {
            for (subst, premises) in self.ground_rule(rule, &extra) {
                let head = apply_atom(&subst, &rule.head);
                if !head.is_ground() {
                    continue;
                }
                // 前件满足度：事实=1，已推出原子取当前软真值，被干预原子取干预值，取最小值（软合取）
                let satisfaction = premises
                    .iter()
                    .map(|premise| self.premise_truth(premise, derived, fixed))
                    .fold(1.0f64, f64::min);
                if satisfaction <= 0.0 {
                    continue;
                }
                results.push(GroundedRule {
                    head,
                    premises,
                    weight: rule.weight,
                    source: rule.source.clone(),
                    rule_text: rule.text(),
                    satisfaction,
                });
            }
        }
        results
    }

    /// 当前全部基化规则（供因果图/梯度归因复用同一批基化实例）
    pub fn ground_all(&self) -> Vec<GroundedRule> {
        let beliefs = self.propagate();
        let derived: HashMap<Atom, f64> = beliefs
            .iter()
            .map(|(atom, (log_odds, _))| (atom.clone(), Self::sigmoid(*log_odds)))
            .collect();
        self.ground_rules_against(&derived, &HashMap::new())
    }

    /// 使用调用方已算好的概率视图进行基化
    pub fn ground_all_with_probabilities(
        &self,
        probabilities: &HashMap<Atom, f64>,
    ) -> Vec<GroundedRule> {
        self.ground_rules_against(probabilities, &HashMap::new())
    }

    /// 当前软信念的"概率视图"（事实不在内；事实按需由调用方补 1.0）
    pub fn probability_map(&self) -> HashMap<Atom, f64> {
        self.propagate()
            .into_iter()
            .map(|(atom, (log_odds, _))| (atom, Self::sigmoid(log_odds)))
            .collect()
    }

    pub fn has_fact(&self, atom: &Atom) -> bool {
        self.facts.contains(atom)
    }

    fn log_odds_of(probability: f64) -> f64 {
        if probability <= 1e-9 {
            -LOG_ODDS_CLAMP
        } else if probability >= 1.0 - 1e-9 {
            LOG_ODDS_CLAMP
        } else {
            (probability / (1.0 - probability))
                .ln()
                .clamp(-LOG_ODDS_CLAMP, LOG_ODDS_CLAMP)
        }
    }

    /// 对规则前件做半连接基化：返回 (置换, 基化前提) 列表。
    /// 论域 = 事实（引用 `facts_by_pred`）+ `extra` 中的补充原子。
    fn ground_rule(
        &self,
        rule: &WeightedRule,
        extra: &HashMap<Rc<str>, Vec<Atom>>,
    ) -> Vec<(Substitution, Vec<Atom>)> {
        let mut results = Vec::new();
        let mut subst = Substitution::new();
        let mut premises = Vec::new();
        self.ground_step(
            &rule.body,
            0,
            &mut subst,
            &mut premises,
            extra,
            &mut results,
        );
        results
    }

    fn ground_step(
        &self,
        body: &[Atom],
        index: usize,
        subst: &mut Substitution,
        premises: &mut Vec<Atom>,
        extra: &HashMap<Rc<str>, Vec<Atom>>,
        results: &mut Vec<(Substitution, Vec<Atom>)>,
    ) {
        if results.len() >= self.max_groundings_per_rule {
            return;
        }
        if index == body.len() {
            results.push((subst.clone(), premises.clone()));
            return;
        }
        let pattern = &body[index];
        let base = self.facts_by_pred.get(pattern.predicate.as_ref());
        let supplement = extra.get(pattern.predicate.as_ref());
        if base.is_none() && supplement.is_none() {
            return;
        }
        let candidates = base
            .into_iter()
            .flat_map(|atoms| atoms.iter())
            .chain(supplement.into_iter().flat_map(|atoms| atoms.iter()));
        for candidate in candidates {
            if results.len() >= self.max_groundings_per_rule {
                return;
            }
            let mut extended = subst.clone();
            if unify_atoms(pattern, candidate, &mut extended) {
                let ground = apply_atom(&extended, pattern);
                if ground.is_ground() {
                    premises.push(ground);
                    self.ground_step(body, index + 1, &mut extended, premises, extra, results);
                    premises.pop();
                }
            }
        }
    }

    /// 软真值传播（不动点迭代）：log-odds 累加 + sigmoid 归一
    fn propagate(&self) -> Beliefs {
        self.propagate_with_fixed(&HashMap::new())
    }

    /// 干预语义（do-演算的符号版）：`fixed` 中的原子真值被钉住、
    /// 且指向它们的规则不再贡献（切断因果边），其余规则照常软真值传播。
    pub fn propagate_with_fixed(&self, fixed: &HashMap<Atom, f64>) -> Beliefs {
        let mut derived: HashMap<Atom, f64> = HashMap::new();
        let mut evidence: HashMap<Atom, Vec<RuleEvidence>> = HashMap::new();
        for _ in 0..self.max_iterations {
            let groundings = self.ground_rules_against(&derived, fixed);
            let mut log_odds: HashMap<Atom, f64> = HashMap::new();
            let mut next_evidence: HashMap<Atom, Vec<RuleEvidence>> = HashMap::new();
            for grounded in groundings {
                // do(·)：被干预原子不接受任何规则支持，概率完全由干预值决定
                if fixed.contains_key(&grounded.head) {
                    continue;
                }
                *log_odds.entry(grounded.head.clone()).or_insert(0.0) +=
                    grounded.weight * grounded.satisfaction;
                next_evidence
                    .entry(grounded.head)
                    .or_default()
                    .push(RuleEvidence {
                        source: grounded.source,
                        rule_text: grounded.rule_text,
                        weight: grounded.weight,
                        premises: grounded.premises,
                    });
            }
            let mut next: HashMap<Atom, f64> = HashMap::new();
            for (atom, raw) in &log_odds {
                next.insert(atom.clone(), Self::sigmoid(*raw));
            }
            for (atom, value) in fixed {
                next.insert(atom.clone(), value.clamp(0.0, 1.0));
            }
            let converged = next.len() == derived.len()
                && next.iter().all(|(atom, probability)| {
                    derived
                        .get(atom)
                        .map(|previous| (previous - probability).abs() <= 1e-6)
                        .unwrap_or(false)
                });
            derived = next;
            evidence = next_evidence;
            if derived.is_empty() || converged {
                break;
            }
        }
        derived
            .into_iter()
            .map(|(atom, probability)| {
                let log_odds = fixed
                    .get(&atom)
                    .map(|value| Self::log_odds_of(*value))
                    .unwrap_or_else(|| Self::log_odds_of(probability));
                let evidence = evidence.remove(&atom).unwrap_or_default();
                (atom, (log_odds, evidence))
            })
            .collect()
    }

    /// 符号梯度归因（"可微推理"的纯符号实现）：
    /// 对结论概率 P=σ(Σ wᵢ·sᵢ) 做解析求导——
    ///   • 规则权重梯度 ∂P/∂wᵢ = P(1−P)·sᵢ；
    ///   • 前提真值梯度 ∂P/∂sᵢ = P(1−P)·wᵢ（前件取 min，此处给出上界代理）；
    /// 梯度只用于解释与"下一个该求证哪条前提"（verify_priority），绝不更新权重。
    pub fn gradient(&self, pattern: &Atom, limit: usize) -> Vec<GradientAttribution> {
        let beliefs = self.propagate();
        let derived: HashMap<Atom, f64> = beliefs
            .iter()
            .map(|(atom, (log_odds, _))| (atom.clone(), Self::sigmoid(*log_odds)))
            .collect();
        let groundings = self.ground_rules_against(&derived, &HashMap::new());
        let mut by_head: HashMap<Atom, Vec<&GroundedRule>> = HashMap::new();
        for grounded in &groundings {
            by_head
                .entry(grounded.head.clone())
                .or_default()
                .push(grounded);
        }
        let mut results: Vec<GradientAttribution> = Vec::new();
        for (atom, instances) in by_head {
            if !Self::atom_matches(pattern, &atom) {
                continue;
            }
            let log_odds = beliefs
                .get(&atom)
                .map(|(log_odds, _)| *log_odds)
                .unwrap_or(0.0);
            let probability = Self::sigmoid(log_odds);
            let slope = probability * (1.0 - probability);
            let mut contributions = Vec::new();
            let mut sensitivities: HashMap<Atom, PremiseSensitivity> = HashMap::new();
            for grounded in &instances {
                contributions.push(RuleContribution {
                    source: grounded.source.clone(),
                    rule_text: grounded.rule_text.clone(),
                    weight: grounded.weight,
                    satisfaction: grounded.satisfaction,
                    gradient: slope * grounded.satisfaction,
                    premises: grounded.premises.clone(),
                });
                for premise in &grounded.premises {
                    let truth = self.premise_truth(premise, &derived, &HashMap::new());
                    let entry =
                        sensitivities
                            .entry(premise.clone())
                            .or_insert(PremiseSensitivity {
                                premise: premise.clone(),
                                truth,
                                gradient: 0.0,
                                verify_priority: 0.0,
                            });
                    entry.gradient += slope * grounded.weight;
                    entry.verify_priority = entry.gradient.abs() * (1.0 - entry.truth).max(0.0);
                }
            }
            contributions.sort_by(|a, b| {
                b.gradient
                    .abs()
                    .partial_cmp(&a.gradient.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.source.cmp(&b.source))
            });
            let mut premise_sensitivity: Vec<PremiseSensitivity> =
                sensitivities.into_values().collect();
            premise_sensitivity.sort_by_cached_key(|item| PriorityDisplayKey {
                priority: item.verify_priority,
                display: item.premise.display(),
            });
            results.push(GradientAttribution {
                atom,
                probability,
                log_odds,
                contributions,
                premise_sensitivity,
            });
        }
        results.sort_by_cached_key(|item| ProbabilityDisplayKey {
            probability: item.probability,
            display: item.atom.display(),
        });
        results.truncate(limit.max(1));
        results
    }

    fn atom_matches(pattern: &Atom, atom: &Atom) -> bool {
        if pattern.predicate != atom.predicate || pattern.arity() != atom.arity() {
            return false;
        }
        pattern
            .terms
            .iter()
            .zip(atom.terms.iter())
            .all(|(left, right)| match left {
                Term::Var(_) => true,
                Term::Const(name) => match right {
                    Term::Const(other) => name == other,
                    Term::Var(_) => false,
                },
            })
    }

    /// 对模式做软查询：返回概率 ≥ 阈值的候选（含事实，概率 1.0）
    pub fn infer(&self, pattern: &Atom, limit: usize) -> Vec<SoftFact> {
        let beliefs = self.propagate();
        self.infer_from_beliefs(pattern, limit, &beliefs, &HashMap::new())
    }

    /// 干预下的软查询：do(固定原子=值) 后返回概率 ≥ 阈值的候选
    pub fn infer_with_fixed(
        &self,
        pattern: &Atom,
        limit: usize,
        fixed: &HashMap<Atom, f64>,
    ) -> Vec<SoftFact> {
        let beliefs = self.propagate_with_fixed(fixed);
        self.infer_from_beliefs(pattern, limit, &beliefs, fixed)
    }

    pub(crate) fn infer_from_beliefs(
        &self,
        pattern: &Atom,
        limit: usize,
        beliefs: &Beliefs,
        fixed: &HashMap<Atom, f64>,
    ) -> Vec<SoftFact> {
        // 收集证据，排序截断后再克隆
        let mut merged: HashMap<&Atom, (f64, f64, Option<&Vec<RuleEvidence>>, bool)> =
            HashMap::new();
        for fact in &self.facts {
            if !Self::atom_matches(pattern, fact) {
                continue;
            }
            // do(事实=假)：被介入的事实从结果中移除；do(事实=真) 保持事实语义
            if let Some(value) = fixed.get(fact) {
                if *value < 0.5 {
                    continue;
                }
            }
            merged.insert(fact, (1.0, LOG_ODDS_CLAMP, None, true));
        }
        for (atom, (log_odds, evidence)) in beliefs {
            if !Self::atom_matches(pattern, atom) {
                continue;
            }
            let probability = Self::sigmoid(*log_odds);
            if probability < self.min_probability {
                continue;
            }
            match merged.get_mut(atom) {
                Some(existing) if probability > existing.0 => {
                    *existing = (probability, *log_odds, Some(evidence), false);
                }
                Some(_) => {}
                None => {
                    merged.insert(atom, (probability, *log_odds, Some(evidence), false));
                }
            }
        }
        let mut entries: Vec<(&Atom, f64, f64, Option<&Vec<RuleEvidence>>, bool)> = merged
            .into_iter()
            .map(|(atom, (probability, log_odds, evidence, from_fact))| {
                (atom, probability, log_odds, evidence, from_fact)
            })
            .collect();
        entries.sort_by_cached_key(|entry| ProbabilityDisplayKey {
            probability: entry.1,
            display: entry.0.display(),
        });
        entries.truncate(limit.max(1));
        entries
            .into_iter()
            .map(|(atom, probability, log_odds, evidence, from_fact)| {
                let mut evidence = evidence.cloned().unwrap_or_default();
                evidence.sort_by(|a, b| {
                    b.weight
                        .partial_cmp(&a.weight)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.source.cmp(&b.source))
                });
                SoftFact {
                    atom: atom.clone(),
                    probability,
                    log_odds,
                    evidence,
                    from_fact,
                }
            })
            .collect()
    }

    /// 全部规则头模式的软预测（用于"我来推测一下"式全局展示）
    pub fn all_predictions(&self, limit: usize) -> Vec<SoftFact> {
        let mut patterns: Vec<Atom> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for rule in &self.rules {
            let terms: Vec<Term> = rule
                .head
                .terms
                .iter()
                .enumerate()
                .map(|(index, _)| Term::var(&format!("V{}", index)))
                .collect();
            let pattern = Atom::new(rule.head.predicate.as_ref(), terms);
            let key = pattern.display();
            if seen.insert(key) {
                patterns.push(pattern);
            }
        }
        let mut merged: HashMap<Atom, SoftFact> = HashMap::new();
        // 所有头模式共用同一批信念
        let beliefs = self.propagate();
        for pattern in &patterns {
            for fact in self.infer_from_beliefs(pattern, 64, &beliefs, &HashMap::new()) {
                merged
                    .entry(fact.atom.clone())
                    .and_modify(|existing| {
                        if fact.probability > existing.probability {
                            *existing = fact.clone();
                        }
                    })
                    .or_insert(fact);
            }
        }
        let mut results: Vec<SoftFact> = merged.into_values().collect();
        results.sort_by_cached_key(|item| ProbabilityDisplayKey {
            probability: item.probability,
            display: item.atom.display(),
        });
        results.truncate(limit.max(1));
        results
    }

    /// 加权 MAP 局部搜索（MaxWalkSat 风格）：
    /// 在候选原子集合上翻转真值，最大化"被满足的基规则权重和"。
    /// 事实固定为真；未出现在候选/事实中的原子视为假。
    pub fn map_assignment(&self, candidates: &[Atom], max_flips: usize) -> Vec<(Atom, bool)> {
        if candidates.is_empty() {
            return Vec::new();
        }
        let candidate_set: HashSet<Atom> = candidates.iter().cloned().collect();
        // MAP 论域 = 事实 + 候选原子（候选原子无论当前真值都参与基化）。
        // 事实直接引用 self.facts_by_pred，仅对候选补充原子去重建表。
        let mut extra: HashMap<Rc<str>, Vec<Atom>> = HashMap::new();
        let mut seen: HashSet<&Atom> = HashSet::new();
        for atom in &candidate_set {
            if !self.facts.contains(atom) && seen.insert(atom) {
                extra
                    .entry(atom.predicate.clone())
                    .or_default()
                    .push(atom.clone());
            }
        }
        // 基化所有规则（用候选集合补齐论域中的原子）
        let mut ground_rules: Vec<(f64, Vec<Atom>, Atom)> = Vec::new();
        for rule in &self.rules {
            for (subst, premises) in self.ground_rule(rule, &extra) {
                let head = apply_atom(&subst, &rule.head);
                if head.is_ground() {
                    ground_rules.push((rule.weight, premises, head));
                }
            }
        }
        let mut assignment: Vec<bool> = vec![false; candidates.len()];
        let index_of: HashMap<&Atom, usize> = candidates
            .iter()
            .enumerate()
            .map(|(index, atom)| (atom, index))
            .collect();
        let evaluate = |assignment: &[bool]| -> f64 {
            let mut score = 0.0;
            for (weight, premises, head) in &ground_rules {
                let body_true = premises.iter().all(|premise| {
                    if self.facts.contains(premise) {
                        true
                    } else {
                        index_of
                            .get(premise)
                            .map(|index| assignment[*index])
                            .unwrap_or(false)
                    }
                });
                if !body_true {
                    continue;
                }
                let head_true = if self.facts.contains(head) {
                    true
                } else {
                    index_of
                        .get(head)
                        .map(|index| assignment[*index])
                        .unwrap_or(false)
                };
                // 软蕴含 body ⇒ head：前件真时，头真计正权重；头假且权重为负则计惩罚
                if head_true || *weight < 0.0 {
                    score += *weight;
                }
            }
            score
        };
        let mut current = evaluate(&assignment);
        for _ in 0..max_flips.max(1) {
            let mut best_index: Option<usize> = None;
            let mut best_gain = 1e-9;
            for index in 0..candidates.len() {
                assignment[index] = !assignment[index];
                let score = evaluate(&assignment);
                assignment[index] = !assignment[index];
                let gain = score - current;
                if gain > best_gain {
                    best_gain = gain;
                    best_index = Some(index);
                }
            }
            match best_index {
                Some(index) => {
                    assignment[index] = !assignment[index];
                    current = evaluate(&assignment);
                }
                None => break,
            }
        }
        candidates
            .into_iter()
            .cloned()
            .zip(assignment)
            .map(|(atom, value)| {
                let fixed_true = self.facts.contains(&atom);
                (atom, value || fixed_true)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resolution_engine::Atom;

    fn fact(predicate: &str, subject: &str, object: &str) -> Atom {
        Atom::rel(predicate, subject, object)
    }

    #[test]
    fn test_weight_from_confidence_roundtrip() {
        let weight = MarkovLogicEngine::weight_from_confidence(0.9);
        let probability = 1.0 / (1.0 + (-weight).exp());
        assert!((probability - 0.9).abs() < 1e-6);
        assert_eq!(MarkovLogicEngine::weight_from_confidence(0.5), 0.0);
        let counted = MarkovLogicEngine::weight_from_counts(9, 1, 1.0);
        assert!(counted > 0.0 && counted < 3.0);
    }

    #[test]
    fn test_single_rule_soft_inference() {
        let mut engine = MarkovLogicEngine::new(4, 0.5);
        engine.add_fact(fact("师傅", "张三", "李四"));
        engine.add_rule(WeightedRule::new(
            Atom::pattern("尊敬", Term::var("A"), Term::var("B")),
            vec![Atom::pattern("师傅", Term::var("A"), Term::var("B"))],
            2.0,
            "测试规则",
        ));
        let results = engine.infer(&Atom::pattern("尊敬", Term::var("X"), Term::var("Y")), 5);
        assert_eq!(results.len(), 1);
        let soft = &results[0];
        assert_eq!(soft.atom.display(), "尊敬(张三, 李四)");
        assert!(!soft.from_fact);
        assert!((soft.probability - 0.880797).abs() < 1e-4);
        assert_eq!(soft.evidence.len(), 1);
        assert_eq!(soft.evidence[0].premises[0].display(), "师傅(张三, 李四)");
    }

    #[test]
    fn test_chained_rules_propagate_soft_truth() {
        let mut engine = MarkovLogicEngine::new(6, 0.5);
        engine.add_fact(fact("师傅", "张三", "李四"));
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
        let results = engine.infer(&Atom::pattern("尊敬", Term::var("X"), Term::var("Y")), 5);
        assert_eq!(results.len(), 1);
        let probability = results[0].probability;
        assert!(
            probability > 0.5 && probability < 0.88,
            "软真值应级联衰减，实得 {}",
            probability
        );
    }

    #[test]
    fn test_facts_are_certain_and_filtered_by_pattern() {
        let mut engine = MarkovLogicEngine::new(4, 0.5);
        engine.add_fact(fact("师傅", "张三", "李四"));
        engine.add_fact(fact("师傅", "王五", "赵六"));
        let results = engine.infer(
            &Atom::pattern("师傅", Term::constant("张三"), Term::var("X")),
            5,
        );
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].atom.display(), "师傅(张三, 李四)");
        assert_eq!(results[0].probability, 1.0);
        assert!(results[0].from_fact);
    }

    #[test]
    fn test_map_search_avoids_conflicting_assignment() {
        let mut engine = MarkovLogicEngine::new(4, 0.5);
        let sunny = Atom::new("天晴", vec![Term::constant("现在")]);
        let rainy = Atom::new("下雨", vec![Term::constant("现在")]);
        let conflict = Atom::new("矛盾", vec![Term::constant("现在")]);
        engine.add_rule(WeightedRule::new(sunny.clone(), vec![], 2.0, "晴天偏好"));
        engine.add_rule(WeightedRule::new(rainy.clone(), vec![], 1.0, "雨天偏好"));
        engine.add_rule(WeightedRule::new(
            conflict,
            vec![sunny.clone(), rainy.clone()],
            -5.0,
            "晴雨互斥",
        ));
        let assignment = engine.map_assignment(&[sunny.clone(), rainy.clone()], 8);
        let sunny_true = assignment
            .iter()
            .find(|(atom, _)| *atom == sunny)
            .map(|(_, value)| *value)
            .unwrap_or(false);
        let rainy_true = assignment
            .iter()
            .find(|(atom, _)| *atom == rainy)
            .map(|(_, value)| *value)
            .unwrap_or(false);
        assert!(
            !(sunny_true && rainy_true),
            "MAP 搜索应避免互斥原子同时为真"
        );
        assert!(sunny_true, "权重更高且不冲突的原子应被选中");
    }

    #[test]
    fn test_hedge_phrases_are_monotone() {
        assert_eq!(MarkovLogicEngine::hedge(0.95), "极有可能");
        assert_eq!(MarkovLogicEngine::hedge(0.8), "很可能");
        assert_eq!(MarkovLogicEngine::hedge(0.65), "比较可能");
        assert_eq!(MarkovLogicEngine::hedge(0.55), "有可能");
        assert!(MarkovLogicEngine::hedge(0.1).contains("较低"));
    }

    #[test]
    fn test_gradient_matches_finite_difference() {
        // 单规则：P=σ(w)，解析梯度 ∂P/∂w = P(1−P)；用有限差分交叉验证
        let mut engine = MarkovLogicEngine::new(4, 0.5);
        engine.add_fact(fact("师傅", "张三", "李四"));
        engine.add_rule(WeightedRule::new(
            Atom::pattern("尊敬", Term::var("A"), Term::var("B")),
            vec![Atom::pattern("师傅", Term::var("A"), Term::var("B"))],
            2.0,
            "师傅⇒尊敬",
        ));
        let attributions =
            engine.gradient(&Atom::pattern("尊敬", Term::var("X"), Term::var("Y")), 5);
        assert_eq!(attributions.len(), 1);
        let analytic = attributions[0].contributions[0].gradient;

        let epsilon: f64 = 1e-4;
        let perturbed: f64 = 2.0 + epsilon;
        let probability = 1.0 / (1.0 + (-perturbed).exp());
        let finite_difference = (probability - 1.0 / (1.0 + (-2.0f64).exp())) / epsilon;
        assert!(
            (analytic - finite_difference).abs() < 1e-3,
            "解析梯度 {} 应≈有限差分 {}",
            analytic,
            finite_difference
        );
    }

    #[test]
    fn test_gradient_ranks_contributions_and_verify_priority() {
        let mut engine = MarkovLogicEngine::new(6, 0.5);
        engine.add_fact(fact("师傅", "张三", "李四"));
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
        let attributions =
            engine.gradient(&Atom::pattern("尊敬", Term::var("X"), Term::var("Y")), 5);
        assert_eq!(attributions.len(), 1);
        let attribution = &attributions[0];
        assert!(attribution.probability > 0.5 && attribution.probability < 0.9);
        assert_eq!(attribution.contributions.len(), 1);
        assert_eq!(attribution.contributions[0].source, "长辈⇒尊敬");
        assert!((attribution.contributions[0].satisfaction - 0.880797).abs() < 1e-3);
        // 已确认事实（truth=1）的求证优先级为 0；不确定的推导前提优先级最高
        let sensitivity = &attribution.premise_sensitivity[0];
        assert_eq!(sensitivity.premise.display(), "长辈(张三, 李四)");
        assert!(sensitivity.verify_priority > 0.0);
        assert!(attribution
            .premise_sensitivity
            .iter()
            .all(|item| item.premise.display() != "师傅(张三, 李四)"));
    }

    #[test]
    fn test_propagate_with_fixed_cuts_incoming_edges() {
        let mut engine = MarkovLogicEngine::new(6, 0.5);
        engine.add_fact(fact("师傅", "张三", "李四"));
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
        let pattern = Atom::pattern("尊敬", Term::var("X"), Term::var("Y"));

        let factual = engine.infer(&pattern, 5);
        assert_eq!(factual.len(), 1);
        let factual_probability = factual[0].probability;

        // do(师傅(张三,李四)=假)：切断因果输入，下游全部消失
        let mut cut = HashMap::new();
        cut.insert(fact("师傅", "张三", "李四"), 0.0);
        assert!(engine.infer_with_fixed(&pattern, 5, &cut).is_empty());

        // do(长辈(张三,李四)=真)：干预一个"推导出来的中间节点"，下游概率上升
        let mut boost = HashMap::new();
        boost.insert(fact("长辈", "张三", "李四"), 1.0);
        let intervened = engine.infer_with_fixed(&pattern, 5, &boost);
        assert_eq!(intervened.len(), 1);
        assert!(
            intervened[0].probability > factual_probability,
            "干预中间节点应提升下游概率：{} vs {}",
            intervened[0].probability,
            factual_probability
        );
        assert!(!intervened[0].from_fact);
    }

    #[test]
    fn test_all_predictions_deduplicates_patterns() {
        let mut engine = MarkovLogicEngine::new(4, 0.5);
        engine.add_fact(fact("师傅", "张三", "李四"));
        for _ in 0..2 {
            engine.add_rule(WeightedRule::new(
                Atom::pattern("尊敬", Term::var("A"), Term::var("B")),
                vec![Atom::pattern("师傅", Term::var("A"), Term::var("B"))],
                2.0,
                "重复规则",
            ));
        }
        let predictions = engine.all_predictions(10);
        assert_eq!(predictions.len(), 1);
        assert_eq!(predictions[0].atom.display(), "尊敬(张三, 李四)");
        // 权重叠加：σ(2+2) ≈ 0.982
        assert!(predictions[0].probability > 0.95);
    }
}

// 默认逻辑（Reiter 默认理论的接地、良基式实现；纯符号，零神经网络、零训练）
//
// 灵感：默认逻辑用 α : β1,…,βn / γ 表达“在通常情况下成立，但有例外”的知识：
//   • α（前提）在扩展中成立；
//   • 每个 β（理由）不被反驳（其反面不在扩展中）；
//   • 则把 γ（结论）加入扩展。
// 本实现刻意选一个可解释、确定性的语义（“接地良基扩展”）：
//   1. 事实与硬规则先行；默认规则逐轮接地（变量域 = 事实常量 + 本体论域）；
//   2. 前提成立、结论未定、理由未被反驳的默认成为候选；
//   3. 同轮候选出现矛盾结论时按“具体度”裁决：前提更能推出对方前提的一方胜出，
//      这条蕴含检查直接调用本体闭包（企鹅 ⊑ 鸟 ⇒ “企鹅不会飞”压过“鸟会飞”）；
//   4. 同等具体度互相压制（记录冲突），事实相反则直接阻塞；
//   5. 每步触发/阻塞都留痕，报告可逐条中文复述，并把默认规则导出为
//      ProbLog 概率规则做“概率视图”对照（默认=确定性例外，概率=软权重）。
// 这正是人类常识推理的符号化：非单调、可例外、可解释、可回放。

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use crate::chinese_reasoning_nlu::render_atom_zh;
use crate::ontology_reasoning::OntologyClosure;
use crate::problog::pattern_matches;
use crate::resolution_engine::{Atom, Term};

const MAX_GROUND_ENVS_PER_RULE: usize = 2048;
const MAX_DOMAIN: usize = 160;
const MAX_REPORT_ENTRIES: usize = 20;

/// 文字：原子或它的否定（本项目知识库为肯定 Horn 事实，否定只在默认逻辑内显式携带）
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Literal {
    pub negated: bool,
    pub atom: Atom,
}

impl Literal {
    pub fn positive(atom: Atom) -> Self {
        Self {
            negated: false,
            atom,
        }
    }

    pub fn negative(atom: Atom) -> Self {
        Self { negated: true, atom }
    }

    pub fn complement(&self) -> Self {
        Self {
            negated: !self.negated,
            atom: self.atom.clone(),
        }
    }

    pub fn display(&self) -> String {
        if self.negated {
            format!("¬{}", self.atom.display())
        } else {
            self.atom.display()
        }
    }

    /// 中文回读（用于报告与阻断原因）
    pub fn render_zh(&self) -> String {
        if self.negated {
            format!("并非「{}」", render_atom_zh(&self.atom))
        } else {
            render_atom_zh(&self.atom)
        }
    }
}

/// 默认规则 α : β1,…,βn / γ（justifications 为空即硬规则）
#[derive(Debug, Clone)]
pub struct DefaultRule {
    pub name: String,
    pub prerequisite: Vec<Literal>,
    pub justifications: Vec<Literal>,
    pub consequent: Literal,
    pub source: String,
}

impl DefaultRule {
    pub fn is_hard(&self) -> bool {
        self.justifications.is_empty()
    }
}

/// 一条默认被触发的记录
#[derive(Debug, Clone)]
pub struct DefaultApplication {
    pub rule: String,
    pub source: String,
    pub instance: Literal,
    pub prerequisite: Vec<Literal>,
    pub round: usize,
}

/// 一条默认被阻塞的记录（含原因，供解释“为什么没推出来”）
#[derive(Debug, Clone)]
pub struct BlockedDefault {
    pub rule: String,
    pub instance: Literal,
    pub reason: String,
}

/// 默认逻辑的一次扩展（接地良基扩展）
#[derive(Debug, Clone, Default)]
pub struct DefaultExtension {
    pub literals: HashSet<Literal>,
    pub applications: Vec<DefaultApplication>,
    pub blocked: Vec<BlockedDefault>,
    /// 矛盾结论的对抗记录（同轮冲突、具体度裁决）
    pub conflicts: Vec<String>,
    pub rounds: usize,
    pub domain_size: usize,
    pub truncated: bool,
}

impl DefaultExtension {
    pub fn holds(&self, literal: &Literal) -> bool {
        self.literals.contains(literal)
    }

    pub fn application_for(&self, literal: &Literal) -> Option<&DefaultApplication> {
        self.applications
            .iter()
            .find(|application| &application.instance == literal)
    }

    pub fn blocked_for(&self, literal: &Literal) -> Vec<&BlockedDefault> {
        self.blocked
            .iter()
            .filter(|blocked| &blocked.instance == literal)
            .collect()
    }

    pub fn applied_count(&self) -> usize {
        self.applications.len()
    }

    pub fn blocked_count(&self) -> usize {
        self.blocked.len()
    }

    /// 扩展中匹配模式（变量为通配）的全部文字
    pub fn matching(&self, pattern: &Atom) -> Vec<Literal> {
        let mut matches: Vec<Literal> = self
            .literals
            .iter()
            .filter(|literal| pattern_matches(pattern, &literal.atom))
            .cloned()
            .collect();
        matches.sort_by_cached_key(Literal::display);
        matches
    }
}

/// 一条接地后的默认规则实例（变量已全部替换为域常量）
struct GroundRule<'a> {
    rule: &'a DefaultRule,
    prerequisite: Vec<Literal>,
    justifications: Vec<Literal>,
    consequent: Literal,
}

/// 默认理论：事实 + 默认规则（含硬规则），按需做扩展
#[derive(Debug, Clone)]
pub struct DefaultTheory {
    facts: Vec<Literal>,
    fact_set: HashSet<Literal>,
    rules: Vec<DefaultRule>,
    max_rounds: usize,
}

impl DefaultTheory {
    pub fn new(max_rounds: usize) -> Self {
        Self {
            facts: Vec::new(),
            fact_set: HashSet::new(),
            rules: Vec::new(),
            max_rounds: max_rounds.max(1),
        }
    }

    pub fn add_fact(&mut self, literal: Literal) {
        if !literal.atom.is_ground() || self.fact_set.contains(&literal) {
            return;
        }
        self.fact_set.insert(literal.clone());
        self.facts.push(literal);
    }

    pub fn add_fact_atom(&mut self, atom: Atom) {
        self.add_fact(Literal::positive(atom));
    }

    /// 添加默认规则，自动补全规则名；返回规则名
    pub fn add_default(
        &mut self,
        prerequisite: Vec<Literal>,
        justifications: Vec<Literal>,
        consequent: Literal,
        source: &str,
    ) -> String {
        let name = format!("d#{}", self.rules.len() + 1);
        self.rules.push(DefaultRule {
            name,
            prerequisite,
            justifications,
            consequent,
            source: source.to_string(),
        });
        self.rules.last().map(|rule| rule.name.clone()).unwrap_or_default()
    }

    /// 添加硬规则（无理由，必然应用）
    pub fn add_hard_rule(
        &mut self,
        prerequisite: Vec<Literal>,
        consequent: Literal,
        source: &str,
    ) -> String {
        let name = format!("h#{}", self.rules.len() + 1);
        self.rules.push(DefaultRule {
            name,
            prerequisite,
            justifications: Vec::new(),
            consequent,
            source: source.to_string(),
        });
        self.rules.last().map(|rule| rule.name.clone()).unwrap_or_default()
    }

    /// 添加已具名的默认规则；name 为空时自动补全；返回规则名
    pub fn add_default_rule(&mut self, mut rule: DefaultRule) -> String {
        if rule.name.is_empty() {
            rule.name = if rule.is_hard() {
                format!("h#{}", self.rules.len() + 1)
            } else {
                format!("d#{}", self.rules.len() + 1)
            };
        }
        let name = rule.name.clone();
        self.rules.push(rule);
        name
    }

    pub fn facts(&self) -> &[Literal] {
        &self.facts
    }

    pub fn rules(&self) -> &[DefaultRule] {
        &self.rules
    }

    pub fn fact_count(&self) -> usize {
        self.facts.len()
    }

    pub fn default_count(&self) -> usize {
        self.rules.iter().filter(|rule| !rule.is_hard()).count()
    }

    pub fn hard_rule_count(&self) -> usize {
        self.rules.iter().filter(|rule| rule.is_hard()).count()
    }

    /// 计算接地良基扩展：事实/硬规则先行，默认逐轮接地触发，
    /// 例外（被反驳的理由）与同轮冲突逐条留痕。全程确定性、可回放。
    pub fn extend(&self, closure: &OntologyClosure) -> DefaultExtension {
        let mut extension = DefaultExtension::default();
        extension.literals = self.fact_set.clone();
        let mut truncated = false;
        let domain = self.build_domain(closure, &mut truncated);
        extension.domain_size = domain.len();
        let mut grounded: Vec<GroundRule> = Vec::new();
        for rule in &self.rules {
            let mut instances = Vec::new();
            ground_rule(rule, &domain, &mut instances, &mut truncated);
            grounded.extend(instances);
        }
        let mut decided = vec![false; grounded.len()];
        for round in 1..=self.max_rounds {
            let mut candidates: Vec<usize> = Vec::new();
            for (index, instance) in grounded.iter().enumerate() {
                if decided[index] {
                    continue;
                }
                if !instance
                    .prerequisite
                    .iter()
                    .all(|literal| extension.literals.contains(literal))
                {
                    continue;
                }
                if extension.literals.contains(&instance.consequent) {
                    decided[index] = true;
                    continue;
                }
                if extension.literals.contains(&instance.consequent.complement()) {
                    decided[index] = true;
                    extension.blocked.push(BlockedDefault {
                        rule: instance.rule.name.clone(),
                        instance: instance.consequent.clone(),
                        reason: format!(
                            "结论的反面「{}」已知成立，事实优先",
                            instance.consequent.complement().render_zh()
                        ),
                    });
                    continue;
                }
                let refuted: Vec<String> = instance
                    .justifications
                    .iter()
                    .filter(|justification| {
                        extension.literals.contains(&justification.complement())
                    })
                    .map(|justification| justification.complement().render_zh())
                    .collect();
                if !refuted.is_empty() {
                    decided[index] = true;
                    extension.blocked.push(BlockedDefault {
                        rule: instance.rule.name.clone(),
                        instance: instance.consequent.clone(),
                        reason: format!("例外「{}」成立，理由被反驳", refuted.join("、")),
                    });
                    continue;
                }
                candidates.push(index);
            }
            // 同轮矛盾候选（γ 与 ¬γ）：按前提具体度裁决
            let mut positives: HashMap<String, Vec<usize>> = HashMap::new();
            let mut negatives: HashMap<String, Vec<usize>> = HashMap::new();
            for &index in &candidates {
                let key = grounded[index].consequent.atom.display();
                if grounded[index].consequent.negated {
                    negatives.entry(key).or_default().push(index);
                } else {
                    positives.entry(key).or_default().push(index);
                }
            }
            let mut conflict_keys: Vec<String> = positives
                .keys()
                .filter(|key| negatives.contains_key(*key))
                .cloned()
                .collect();
            conflict_keys.sort();
            let mut suppressed: HashSet<usize> = HashSet::new();
            for key in conflict_keys {
                for &positive in &positives[&key] {
                    for &negative in &negatives[&key] {
                        let left = &grounded[positive];
                        let right = &grounded[negative];
                        let conflict = format!(
                            "默认 {} 与 {} 给出矛盾结论：「{}」与「{}」",
                            left.rule.name,
                            right.rule.name,
                            left.consequent.render_zh(),
                            right.consequent.render_zh()
                        );
                        extension.conflicts.push(conflict.clone());
                        match compare_specificity(left, right, closure) {
                            Ordering::Greater => {
                                if suppressed.insert(negative) {
                                    decided[negative] = true;
                                    extension.blocked.push(BlockedDefault {
                                        rule: right.rule.name.clone(),
                                        instance: right.consequent.clone(),
                                        reason: format!(
                                            "与更具体的默认 {} 冲突（{}）",
                                            left.rule.name, conflict
                                        ),
                                    });
                                }
                            }
                            Ordering::Less => {
                                if suppressed.insert(positive) {
                                    decided[positive] = true;
                                    extension.blocked.push(BlockedDefault {
                                        rule: left.rule.name.clone(),
                                        instance: left.consequent.clone(),
                                        reason: format!(
                                            "与更具体的默认 {} 冲突（{}）",
                                            right.rule.name, conflict
                                        ),
                                    });
                                }
                            }
                            Ordering::Equal => {
                                for (index, loser) in [(positive, left), (negative, right)] {
                                    if suppressed.insert(index) {
                                        decided[index] = true;
                                        extension.blocked.push(BlockedDefault {
                                            rule: loser.rule.name.clone(),
                                            instance: loser.consequent.clone(),
                                            reason: format!(
                                                "同等具体度互相压制（{}）",
                                                conflict
                                            ),
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
            let mut applied = 0usize;
            for &index in &candidates {
                if suppressed.contains(&index) {
                    continue;
                }
                let instance = &grounded[index];
                if !extension.literals.insert(instance.consequent.clone()) {
                    decided[index] = true;
                    continue;
                }
                extension.applications.push(DefaultApplication {
                    rule: instance.rule.name.clone(),
                    source: instance.rule.source.clone(),
                    instance: instance.consequent.clone(),
                    prerequisite: instance.prerequisite.clone(),
                    round,
                });
                decided[index] = true;
                applied += 1;
            }
            if applied == 0 {
                break;
            }
            extension.rounds = round;
        }
        extension.truncated = truncated;
        extension
    }

    /// 变量域：事实常量 + 本体闭包中的个体、类与关系两端（有序、限长）
    fn build_domain(&self, closure: &OntologyClosure, truncated: &mut bool) -> Vec<String> {
        let mut names: HashSet<String> = HashSet::new();
        for literal in &self.facts {
            for term in &literal.atom.terms {
                names.insert(term.name().to_string());
            }
        }
        for (individual, class) in closure.instances_snapshot() {
            names.insert(individual);
            names.insert(class);
        }
        for (subject, _, object) in closure.relations_snapshot() {
            names.insert(subject);
            names.insert(object);
        }
        let mut domain: Vec<String> = names.into_iter().collect();
        domain.sort();
        if domain.len() > MAX_DOMAIN {
            domain.truncate(MAX_DOMAIN);
            *truncated = true;
        }
        domain
    }
}

/// 对一条规则枚举域内全部接地实例（变量按首次出现顺序做笛卡尔积，带上限）
fn ground_rule<'a>(
    rule: &'a DefaultRule,
    domain: &[String],
    output: &mut Vec<GroundRule<'a>>,
    truncated: &mut bool,
) {
    let mut variables: Vec<String> = Vec::new();
    for literal in rule
        .prerequisite
        .iter()
        .chain(rule.justifications.iter())
        .chain(std::iter::once(&rule.consequent))
    {
        for term in &literal.atom.terms {
            if let Term::Var(name) = term {
                if !variables.iter().any(|existing| existing == &**name) {
                    variables.push(name.to_string());
                }
            }
        }
    }
    if variables.is_empty() {
        output.push(GroundRule {
            rule,
            prerequisite: rule.prerequisite.clone(),
            justifications: rule.justifications.clone(),
            consequent: rule.consequent.clone(),
        });
        return;
    }
    let mut substitution: HashMap<String, Term> = HashMap::new();
    enumerate_instances(rule, &variables, 0, domain, &mut substitution, output, truncated);
}

fn enumerate_instances<'a>(
    rule: &'a DefaultRule,
    variables: &[String],
    index: usize,
    domain: &[String],
    substitution: &mut HashMap<String, Term>,
    output: &mut Vec<GroundRule<'a>>,
    truncated: &mut bool,
) {
    if output.len() >= MAX_GROUND_ENVS_PER_RULE {
        *truncated = true;
        return;
    }
    if index == variables.len() {
        output.push(GroundRule {
            rule,
            prerequisite: rule
                .prerequisite
                .iter()
                .map(|literal| instantiate_literal(literal, substitution))
                .collect(),
            justifications: rule
                .justifications
                .iter()
                .map(|literal| instantiate_literal(literal, substitution))
                .collect(),
            consequent: instantiate_literal(&rule.consequent, substitution),
        });
        return;
    }
    for constant in domain {
        substitution.insert(variables[index].clone(), Term::constant(constant));
        enumerate_instances(
            rule,
            variables,
            index + 1,
            domain,
            substitution,
            output,
            truncated,
        );
        if output.len() >= MAX_GROUND_ENVS_PER_RULE {
            *truncated = true;
            break;
        }
    }
    substitution.remove(&variables[index]);
}

fn instantiate_literal(literal: &Literal, substitution: &HashMap<String, Term>) -> Literal {
    Literal {
        negated: literal.negated,
        atom: instantiate_atom(&literal.atom, substitution),
    }
}

fn instantiate_atom(atom: &Atom, substitution: &HashMap<String, Term>) -> Atom {
    Atom {
        predicate: atom.predicate.clone(),
        terms: atom
            .terms
            .iter()
            .map(|term| match term {
                Term::Var(name) => substitution
                    .get(&**name)
                    .cloned()
                    .unwrap_or_else(|| term.clone()),
                other => other.clone(),
            })
            .collect(),
    }
}

/// 具体度比较：前提更能推出（更具体于）对方前提的一方胜出
fn compare_specificity(
    left: &GroundRule,
    right: &GroundRule,
    closure: &OntologyClosure,
) -> Ordering {
    let left_more = prerequisite_entails(&left.prerequisite, &right.prerequisite, closure);
    let right_more = prerequisite_entails(&right.prerequisite, &left.prerequisite, closure);
    match (left_more, right_more) {
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        _ => Ordering::Equal,
    }
}

/// specific 的前提集合是否严格蕴含 general 的前提集合（逐条匹配，至少一条更具体）
fn prerequisite_entails(
    specific: &[Literal],
    general: &[Literal],
    closure: &OntologyClosure,
) -> bool {
    if specific.is_empty() || general.is_empty() {
        return false;
    }
    let mut strict = false;
    for general_literal in general {
        let mut matched = false;
        for specific_literal in specific {
            if specific_literal == general_literal {
                matched = true;
                break;
            }
            if literal_specializes(specific_literal, general_literal, closure) {
                strict = true;
                matched = true;
                break;
            }
        }
        if !matched {
            return false;
        }
    }
    strict
}

/// specific 是否严格比 general 更具体：本体闭包给出子类/实例/子谓词的证据
fn literal_specializes(
    specific: &Literal,
    general: &Literal,
    closure: &OntologyClosure,
) -> bool {
    if specific.negated != general.negated {
        return false;
    }
    let specific_atom = &specific.atom;
    let general_atom = &general.atom;
    if specific_atom.predicate == general_atom.predicate
        && specific_atom.arity() == general_atom.arity()
    {
        let mut strict = false;
        for (specific_term, general_term) in specific_atom
            .terms
            .iter()
            .zip(general_atom.terms.iter())
        {
            if specific_term == general_term {
                continue;
            }
            match (specific_term, general_term) {
                (Term::Const(inner), Term::Const(outer)) if inner != outer => {
                    if closure.is_subclass(inner, outer) && !closure.is_subclass(outer, inner) {
                        strict = true;
                    } else if closure.is_instance(inner, outer) {
                        strict = true;
                    } else {
                        return false;
                    }
                }
                _ => return false,
            }
        }
        return strict;
    }
    if !specific.negated
        && specific_atom.arity() == 1
        && general_atom.arity() == 1
        && specific_atom.terms[0] == general_atom.terms[0]
    {
        return closure.is_subclass(&specific_atom.predicate, &general_atom.predicate)
            && !closure.is_subclass(&general_atom.predicate, &specific_atom.predicate);
    }
    false
}

/// 触发记录的中文回读
pub fn render_application_zh(application: &DefaultApplication) -> String {
    let premises = application
        .prerequisite
        .iter()
        .map(Literal::render_zh)
        .collect::<Vec<_>>()
        .join(" 且 ");
    let premise_text = if premises.is_empty() {
        "（无条件）".to_string()
    } else {
        format!("「{}」", premises)
    };
    format!(
        "默认 {}：从 {} 通常推出「{}」（来源：{}；第 {} 轮）",
        application.rule,
        premise_text,
        application.instance.render_zh(),
        application.source,
        application.round
    )
}

/// 阻断记录的中文回读
pub fn render_blocked_zh(blocked: &BlockedDefault) -> String {
    format!(
        "默认 {}：不能推出「{}」——{}",
        blocked.rule,
        blocked.instance.render_zh(),
        blocked.reason
    )
}

/// 扩展报告的中文回读：可选查询一条具体陈述（否则列出全部触发与阻断）
pub fn render_extension_zh(
    theory: &DefaultTheory,
    extension: &DefaultExtension,
    target: Option<&Literal>,
) -> String {
    let mut output = String::new();
    output.push_str(&format!(
        "默认逻辑（接地良基扩展）：事实 {} 条，默认规则 {} 条（硬规则 {} 条），变量域 {} 个常量，{} 轮推导。\n",
        theory.fact_count(),
        theory.default_count(),
        theory.hard_rule_count(),
        extension.domain_size,
        extension.rounds
    ));
    if extension.truncated {
        output.push_str("注意：接地规模达到上限，扩展可能不完整（可减少实体或规则后再查询）。\n");
    }
    if let Some(target) = target {
        output.push_str(&format!("查询「{}」：", target.render_zh()));
        if extension.holds(target) {
            match extension.application_for(target) {
                Some(application) => {
                    output.push_str("成立（默认推理触发）。\n  ");
                    output.push_str(&render_application_zh(application));
                    output.push('\n');
                }
                None => output.push_str("成立（由事实或硬规则直接给出）。\n"),
            }
        } else {
            let blocked = extension.blocked_for(target);
            if blocked.is_empty() {
                output.push_str("未找到证据（开放世界：不推断为假）。\n");
                output.push_str("  提示：可用「教学：」补充「除非」默认句，或确认实体已录入。\n");
            } else {
                output.push_str("未成立（被默认阻断）：\n");
                for entry in blocked {
                    output.push_str("  ");
                    output.push_str(&render_blocked_zh(entry));
                    output.push('\n');
                }
            }
        }
    }
    if extension.applications.is_empty() && extension.blocked.is_empty() {
        if target.is_none() {
            output.push_str("本次扩展没有触发任何默认，也没有发现阻断（可以先教一句「除非」默认句）。\n");
        }
    } else {
        output.push_str(&format!(
            "触发默认 {} 条，阻断 {} 条。\n",
            extension.applied_count(),
            extension.blocked_count()
        ));
        if target.is_none() {
            if !extension.applications.is_empty() {
                output.push_str("已触发默认：\n");
                for application in extension.applications.iter().take(MAX_REPORT_ENTRIES) {
                    output.push_str("  ");
                    output.push_str(&render_application_zh(application));
                    output.push('\n');
                }
                if extension.applications.len() > MAX_REPORT_ENTRIES {
                    output.push_str(&format!(
                        "  …（还有 {} 条）\n",
                        extension.applications.len() - MAX_REPORT_ENTRIES
                    ));
                }
            }
            if !extension.blocked.is_empty() {
                output.push_str("阻断记录：\n");
                for blocked in extension.blocked.iter().take(MAX_REPORT_ENTRIES) {
                    output.push_str("  ");
                    output.push_str(&render_blocked_zh(blocked));
                    output.push('\n');
                }
                if extension.blocked.len() > MAX_REPORT_ENTRIES {
                    output.push_str(&format!(
                        "  …（还有 {} 条）\n",
                        extension.blocked.len() - MAX_REPORT_ENTRIES
                    ));
                }
            }
            if !extension.conflicts.is_empty() {
                output.push_str("冲突记录：\n");
                for conflict in extension.conflicts.iter().take(MAX_REPORT_ENTRIES) {
                    output.push_str("  ");
                    output.push_str(conflict);
                    output.push('\n');
                }
            }
        }
    }
    output.trim_end().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ontology_reasoning::OntologyReasoner;

    fn binary(predicate: &str, left: &str, right: &str) -> Atom {
        Atom::rel(predicate, left, right)
    }

    fn unary(predicate: &str, subject: &str) -> Atom {
        Atom::unary(predicate, Term::constant(subject))
    }

    fn is_fly_default(theory: &mut DefaultTheory, with_prerequisite: bool) -> String {
        let prerequisite = if with_prerequisite {
            vec![Literal::positive(Atom::new(
                "是",
                vec![Term::var("X"), Term::constant("鸟")],
            ))]
        } else {
            Vec::new()
        };
        theory.add_default(
            prerequisite,
            vec![Literal::negative(Atom::new(
                "是",
                vec![Term::var("X"), Term::constant("企鹅")],
            ))],
            Literal::positive(Atom::unary("会飞", Term::var("X"))),
            "测试默认",
        )
    }

    #[test]
    fn test_extend_applies_default_without_exception() {
        let mut theory = DefaultTheory::new(4);
        theory.add_fact_atom(binary("是", "麻雀", "鸟"));
        let name = is_fly_default(&mut theory, true);
        assert_eq!(name, "d#1");

        let extension = theory.extend(&OntologyClosure::default());
        let target = Literal::positive(unary("会飞", "麻雀"));
        assert!(extension.holds(&target), "麻雀应通常会飞");
        let application = extension.application_for(&target).expect("应有触发记录");
        assert_eq!(application.rule, "d#1");
        assert_eq!(application.round, 1);
        assert!(extension.blocked.is_empty(), "不应有阻断");
        assert_eq!(extension.rounds, 1);
    }

    #[test]
    fn test_exception_blocks_default_and_leaves_trace() {
        let mut theory = DefaultTheory::new(4);
        theory.add_fact_atom(binary("是", "蒂娜", "鸟"));
        theory.add_fact_atom(binary("是", "蒂娜", "企鹅"));
        is_fly_default(&mut theory, true);

        let extension = theory.extend(&OntologyClosure::default());
        let target = Literal::positive(unary("会飞", "蒂娜"));
        assert!(!extension.holds(&target), "企鹅例外应阻断会飞");
        let blocked = extension.blocked_for(&target);
        assert_eq!(blocked.len(), 1, "应有唯一阻断记录");
        assert!(blocked[0].reason.contains("例外"), "{}", blocked[0].reason);
        assert!(blocked[0].reason.contains("蒂娜是企鹅"), "{}", blocked[0].reason);
    }

    fn penguin_closure() -> OntologyClosure {
        let mut ontology = OntologyReasoner::new(4);
        ontology.add_subclass("企鹅", "鸟");
        ontology.materialize()
    }

    #[test]
    fn test_specific_default_defeats_general_default() {
        let mut theory = DefaultTheory::new(4);
        theory.add_fact_atom(binary("是", "阿德利", "鸟"));
        theory.add_fact_atom(binary("是", "阿德利", "企鹅"));
        theory.add_default(
            vec![Literal::positive(Atom::new(
                "是",
                vec![Term::var("X"), Term::constant("鸟")],
            ))],
            vec![Literal::negative(Atom::unary("不能飞", Term::var("X")))],
            Literal::positive(Atom::unary("会飞", Term::var("X"))),
            "测试默认·鸟",
        );
        theory.add_default(
            vec![Literal::positive(Atom::new(
                "是",
                vec![Term::var("X"), Term::constant("企鹅")],
            ))],
            vec![Literal::negative(Atom::unary("会飞", Term::var("X")))],
            Literal::negative(Atom::unary("会飞", Term::var("X"))),
            "测试默认·企鹅",
        );

        let extension = theory.extend(&penguin_closure());
        let fly = Literal::positive(unary("会飞", "阿德利"));
        let not_fly = Literal::negative(unary("会飞", "阿德利"));
        assert!(!extension.holds(&fly), "更具体的企鹅默认应压过鸟默认");
        assert!(extension.holds(&not_fly), "企鹅不会飞应成立");
        let blocked = extension.blocked_for(&fly);
        assert_eq!(blocked.len(), 1);
        assert!(blocked[0].reason.contains("更具体"), "{}", blocked[0].reason);
        assert!(!extension.conflicts.is_empty(), "应记录冲突");
    }

    #[test]
    fn test_equal_specificity_conflict_mutually_suppresses() {
        let mut theory = DefaultTheory::new(4);
        theory.add_fact_atom(unary("左派", "某人"));
        theory.add_fact_atom(unary("右派", "某人"));
        theory.add_default(
            vec![Literal::positive(unary("左派", "某人"))],
            vec![Literal::negative(unary("反对", "某人"))],
            Literal::positive(unary("赞成", "某人")),
            "测试默认·左",
        );
        theory.add_default(
            vec![Literal::positive(unary("右派", "某人"))],
            vec![Literal::negative(unary("赞成", "某人"))],
            Literal::negative(unary("赞成", "某人")),
            "测试默认·右",
        );

        let extension = theory.extend(&OntologyClosure::default());
        assert!(!extension.holds(&Literal::positive(unary("赞成", "某人"))));
        assert!(!extension.holds(&Literal::negative(unary("赞成", "某人"))));
        assert_eq!(extension.blocked.len(), 2, "双方应互相压制");
        assert!(extension
            .blocked
            .iter()
            .all(|entry| entry.reason.contains("互相压制")));
        assert!(!extension.conflicts.is_empty());
    }

    #[test]
    fn test_defaults_chain_across_rounds() {
        let mut theory = DefaultTheory::new(4);
        theory.add_fact_atom(binary("是", "小麻", "麻雀"));
        theory.add_default(
            vec![Literal::positive(Atom::new(
                "是",
                vec![Term::var("X"), Term::constant("麻雀")],
            ))],
            vec![Literal::negative(Atom::new(
                "是",
                vec![Term::var("X"), Term::constant("企鹅")],
            ))],
            Literal::positive(Atom::new(
                "是",
                vec![Term::var("X"), Term::constant("鸟")],
            )),
            "测试默认·麻雀是鸟",
        );
        is_fly_default(&mut theory, true);

        let extension = theory.extend(&OntologyClosure::default());
        assert!(extension.holds(&Literal::positive(unary("会飞", "小麻"))));
        assert_eq!(extension.rounds, 2, "应跨两轮链式推导");
        assert_eq!(extension.applications.len(), 2);
    }

    #[test]
    fn test_domain_overflow_marks_truncated() {
        let mut theory = DefaultTheory::new(2);
        for index in 0..(MAX_DOMAIN + 10) {
            theory.add_fact_atom(unary(&format!("标记{}", index), &format!("实体{}", index)));
        }
        let extension = theory.extend(&OntologyClosure::default());
        assert!(extension.truncated, "域超限应标记截断");
        assert_eq!(extension.domain_size, MAX_DOMAIN);
    }

    #[test]
    fn test_report_renders_target_and_block_reasons() {
        let mut theory = DefaultTheory::new(4);
        theory.add_fact_atom(binary("是", "蒂娜", "鸟"));
        theory.add_fact_atom(binary("是", "蒂娜", "企鹅"));
        is_fly_default(&mut theory, true);
        let extension = theory.extend(&OntologyClosure::default());

        let target = Literal::positive(unary("会飞", "蒂娜"));
        let report = render_extension_zh(&theory, &extension, Some(&target));
        assert!(report.contains("默认逻辑"), "{}", report);
        assert!(report.contains("蒂娜是企鹅"), "{}", report);
        assert!(report.contains("阻断"), "{}", report);

        let overview = render_extension_zh(&theory, &extension, None);
        assert!(overview.contains("默认规则 1 条"), "{}", overview);
        assert!(overview.contains("阻断记录"), "{}", overview);
    }

    #[test]
    fn test_matching_supports_negative_literals() {
        let mut theory = DefaultTheory::new(2);
        theory.add_fact(Literal::negative(unary("会飞", "企鹅一号")));
        let extension = theory.extend(&OntologyClosure::default());
        assert!(extension.holds(&Literal::negative(unary("会飞", "企鹅一号"))));
        // 覆盖渲染分支
        assert!(Literal::negative(unary("会飞", "企鹅一号"))
            .render_zh()
            .contains("并非"));
        assert!(binary("是", "企鹅一号", "企鹅").display().contains("企鹅一号"));
    }
}

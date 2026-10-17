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

use std::collections::{HashMap, HashSet};

use crate::chinese_reasoning_nlu::{parse_proposition, render_atom_zh};
use crate::ontology_reasoning::OntologyClosure;
use crate::problog::{pattern_matches, ProblogEngine};
use crate::resolution_engine::{Atom, Term};

const MAX_GROUND_ENVS_PER_RULE: usize = 2048;
const MAX_DOMAIN: usize = 160;

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
}

//! 图谱模式零解时的失败诊断。
//!
//! 精确子图匹配把「零解」压成一个空 `Vec`，但零解至少有四种完全不同的
//! 原因：图是空的、某个谓词根本没有事实、某个常量不在要求的位置、每条边
//! 单独都有支持却无法通过共享变量连接，或者约束互相冲突。把它们区分开，
//! 上层才能给出可检查的回答，而不是笼统的「没有找到」。
//!
//! 这里只读：诊断不修改模式、不写图谱，也不替用户放宽条件。放宽候选由
//! [`crate::conversational_graph`] 之外的修复层生成，并必须显式确认后才生效。

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::conversational_graph::{
    EdgePattern, GraphFact, GraphIndex, GraphPattern, MatchOptions, PatternConstraint, PatternTerm,
    DEFAULT_MAX_EXPANSIONS,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiagnosisKind {
    /// 图里没有任何事实。
    EmptyGraph,
    /// 某条边的谓词在图中一条事实也没有。
    MissingPredicate,
    /// 谓词有事实，但边上的常量没有出现在要求的主语/宾语位置。
    MissingConstant,
    /// 主语和宾语分别都出现过，但这条具体三元组不存在。
    MissingCombination,
    /// 每条边单独都有支持，合取后却没有公共绑定。
    JoinConflict,
    /// 模式边可匹配，但约束之间互相冲突。
    ConstraintConflict,
    /// 搜索预算先耗尽，不能据此断定无解。
    BudgetExhausted,
}

impl DiagnosisKind {
    pub fn label(self) -> &'static str {
        match self {
            DiagnosisKind::EmptyGraph => "图为空",
            DiagnosisKind::MissingPredicate => "谓词无事实",
            DiagnosisKind::MissingConstant => "常量位置失配",
            DiagnosisKind::MissingCombination => "三元组不存在",
            DiagnosisKind::JoinConflict => "连接冲突",
            DiagnosisKind::ConstraintConflict => "约束冲突",
            DiagnosisKind::BudgetExhausted => "预算耗尽",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeIssue {
    MissingPredicate,
    MissingSubject,
    MissingObject,
    MissingPair,
    Supported,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EdgeDiagnosis {
    pub edge_index: usize,
    pub predicate: String,
    /// 该谓词在图中的事实总数（不受常量过滤影响）。
    pub predicate_facts: usize,
    /// 按该边上的常量过滤后的事实数。
    pub matching_facts: usize,
    pub issue: EdgeIssue,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MatchDiagnosis {
    pub kind: DiagnosisKind,
    pub edges: Vec<EdgeDiagnosis>,
    /// 直接失配的边，或最小连接冲突组。
    pub failing_edges: Vec<usize>,
    /// 冲突的约束下标（仅约束冲突时有值）。
    pub failing_constraints: Vec<usize>,
    pub expansions: usize,
}

impl MatchDiagnosis {
    pub fn is_budget_exhausted(&self) -> bool {
        self.kind == DiagnosisKind::BudgetExhausted
    }

    /// 生成一段可渲染的中文诊断；调用方负责决定是否附在答案后面。
    pub fn render(&self, pattern: &GraphPattern) -> String {
        let mut output = format!("失败诊断：{}。", self.kind.label());
        match self.kind {
            DiagnosisKind::EmptyGraph => {
                output.push_str("当前知识图谱没有可匹配的事实。");
            }
            DiagnosisKind::MissingPredicate => {
                let missing: Vec<String> = self
                    .edges
                    .iter()
                    .filter(|edge| edge.issue == EdgeIssue::MissingPredicate)
                    .map(|edge| format!("「{}」", edge.predicate))
                    .collect();
                if !missing.is_empty() {
                    output.push_str(&format!("图中没有任何{}事实。", missing.join("、")));
                }
            }
            DiagnosisKind::MissingConstant => {
                let details: Vec<String> = self
                    .edges
                    .iter()
                    .filter(|edge| {
                        matches!(
                            edge.issue,
                            EdgeIssue::MissingSubject | EdgeIssue::MissingObject
                        )
                    })
                    .filter_map(|edge| {
                        pattern
                            .edges
                            .get(edge.edge_index)
                            .map(|pattern_edge| self.describe_edge(edge, pattern_edge))
                    })
                    .collect();
                if !details.is_empty() {
                    output.push_str(&format!("{}。", details.join("；")));
                }
            }
            DiagnosisKind::MissingCombination => {
                let details: Vec<String> = self
                    .edges
                    .iter()
                    .filter(|edge| matches!(edge.issue, EdgeIssue::MissingPair))
                    .filter_map(|edge| {
                        pattern
                            .edges
                            .get(edge.edge_index)
                            .map(|pattern_edge| self.describe_edge(edge, pattern_edge))
                    })
                    .collect();
                if !details.is_empty() {
                    output.push_str(&format!("{}。", details.join("；")));
                }
            }
            DiagnosisKind::JoinConflict => {
                let edges: Vec<String> = pattern
                    .edges
                    .iter()
                    .enumerate()
                    .filter(|(index, _)| self.failing_edges.contains(index))
                    .map(|(_, edge)| describe_edge(edge))
                    .collect();
                if !edges.is_empty() {
                    output.push_str(&format!(
                        "每条边单独都有事实支持，但最小冲突组无法共享变量：{}。",
                        edges.join("；")
                    ));
                }
            }
            DiagnosisKind::ConstraintConflict => {
                let constraints: Vec<String> = self
                    .failing_constraints
                    .iter()
                    .filter_map(|index| pattern.constraints.get(*index))
                    .map(describe_constraint)
                    .collect();
                if !constraints.is_empty() {
                    output.push_str(&format!("这些约束无法同时成立：{}。", constraints.join("；")));
                }
            }
            DiagnosisKind::BudgetExhausted => {
                output.push_str("搜索预算在找到结果前用尽，不能据此断定图中无解。");
            }
        }
        output
    }

    fn describe_edge(&self, diagnosis: &EdgeDiagnosis, edge: &EdgePattern) -> String {
        let details = match diagnosis.issue {
            EdgeIssue::MissingSubject => "主语常量在图中没有该谓词的事实",
            EdgeIssue::MissingObject => "宾语常量在图中没有该谓词的事实",
            EdgeIssue::MissingPair => "主语和宾语分别存在，但这条具体三元组不存在",
            EdgeIssue::MissingPredicate => "谓词在图中没有事实",
            EdgeIssue::Supported => "有支持",
        };
        format!(
            "第 {} 条边 {}（{}；谓词事实 {} 条，过滤后 {} 条）",
            diagnosis.edge_index + 1,
            describe_edge(edge),
            details,
            diagnosis.predicate_facts,
            diagnosis.matching_facts
        )
    }
}

/// 诊断一个零解模式；模式有解或无法求值时返回 `None`。
pub fn diagnose_failure(
    pattern: &GraphPattern,
    facts: &[GraphFact],
    options: MatchOptions,
) -> Option<MatchDiagnosis> {
    let outcome = pattern.evaluate(facts, options).ok()?;
    if !outcome.solutions.is_empty() {
        return None;
    }
    let mut diagnosis = MatchDiagnosis {
        kind: DiagnosisKind::EmptyGraph,
        edges: Vec::new(),
        failing_edges: Vec::new(),
        failing_constraints: Vec::new(),
        expansions: outcome.expansions,
    };
    if outcome.truncated {
        diagnosis.kind = DiagnosisKind::BudgetExhausted;
        return Some(diagnosis);
    }
    if facts.is_empty() {
        diagnosis.kind = DiagnosisKind::EmptyGraph;
        return Some(diagnosis);
    }

    let index = GraphIndex::new(facts);
    diagnosis.edges = pattern
        .edges
        .iter()
        .enumerate()
        .map(|(edge_index, edge)| inspect_edge(&index, edge, edge_index))
        .collect();

    let direct: Vec<usize> = diagnosis
        .edges
        .iter()
        .filter(|edge| edge.issue != EdgeIssue::Supported)
        .map(|edge| edge.edge_index)
        .collect();
    if !direct.is_empty() {
        diagnosis.failing_edges = direct;
        diagnosis.kind = classify_direct(&diagnosis.edges);
        return Some(diagnosis);
    }

    let options = MatchOptions::new(
        1,
        options.max_expansions.max(DEFAULT_MAX_EXPANSIONS),
    );
    let without_constraints = with_constraints(pattern, &[]);
    match satisfiable(&without_constraints, facts, options) {
        Some(true) => {
            diagnosis.kind = DiagnosisKind::ConstraintConflict;
            diagnosis.failing_constraints =
                minimal_unsat_constraints(pattern, facts, options);
        }
        Some(false) => {
            diagnosis.kind = DiagnosisKind::JoinConflict;
            let seed: Vec<usize> = (0..pattern.edges.len()).collect();
            diagnosis.failing_edges = minimal_unsat_edges(pattern, facts, options, seed);
        }
        None => {
            diagnosis.kind = DiagnosisKind::BudgetExhausted;
        }
    }
    Some(diagnosis)
}

fn classify_direct(edges: &[EdgeDiagnosis]) -> DiagnosisKind {
    if edges
        .iter()
        .any(|edge| edge.issue == EdgeIssue::MissingPredicate)
    {
        return DiagnosisKind::MissingPredicate;
    }
    if edges
        .iter()
        .any(|edge| edge.issue == EdgeIssue::MissingPair)
    {
        return DiagnosisKind::MissingCombination;
    }
    DiagnosisKind::MissingConstant
}

fn inspect_edge(index: &GraphIndex<'_>, edge: &EdgePattern, edge_index: usize) -> EdgeDiagnosis {
    let subject = const_value(&edge.subject);
    let object = const_value(&edge.object);
    let predicate_facts = index.predicate_count(&edge.predicate);
    let matching_facts = index
        .candidates(subject, &edge.predicate, object)
        .len();
    let issue = if predicate_facts == 0 {
        EdgeIssue::MissingPredicate
    } else if let Some(subject) = subject {
        if index.subject_predicate_count(subject, &edge.predicate) == 0 {
            EdgeIssue::MissingSubject
        } else if let Some(object) = object {
            if index.predicate_object_count(&edge.predicate, object) == 0 {
                EdgeIssue::MissingObject
            } else if matching_facts == 0 {
                EdgeIssue::MissingPair
            } else {
                EdgeIssue::Supported
            }
        } else {
            EdgeIssue::Supported
        }
    } else if let Some(object) = object {
        if index.predicate_object_count(&edge.predicate, object) == 0 {
            EdgeIssue::MissingObject
        } else {
            EdgeIssue::Supported
        }
    } else {
        EdgeIssue::Supported
    };
    EdgeDiagnosis {
        edge_index,
        predicate: edge.predicate.clone(),
        predicate_facts,
        matching_facts,
        issue,
    }
}

fn const_value(term: &PatternTerm) -> Option<&str> {
    match term {
        PatternTerm::Const(value) => Some(value.as_str()),
        PatternTerm::Var(_) => None,
    }
}

fn satisfiable(
    pattern: &GraphPattern,
    facts: &[GraphFact],
    options: MatchOptions,
) -> Option<bool> {
    let mut probe = pattern.clone();
    probe.select.clear();
    let outcome = probe.evaluate(facts, options).ok()?;
    if !outcome.solutions.is_empty() {
        Some(true)
    } else if outcome.truncated {
        None
    } else {
        Some(false)
    }
}

fn with_constraints(pattern: &GraphPattern, keep: &[usize]) -> GraphPattern {
    let constraints: Vec<PatternConstraint> = keep
        .iter()
        .filter_map(|index| pattern.constraints.get(*index).cloned())
        .collect();
    GraphPattern {
        edges: pattern.edges.clone(),
        constraints,
        select: Vec::new(),
        distinct: pattern.distinct,
    }
}

fn with_edges(pattern: &GraphPattern, keep: &[usize]) -> GraphPattern {
    let mut edges = Vec::new();
    let mut variables: BTreeSet<String> = BTreeSet::new();
    for index in keep {
        let Some(edge) = pattern.edges.get(*index) else {
            continue;
        };
        edges.push(edge.clone());
        for term in [&edge.subject, &edge.object] {
            if let PatternTerm::Var(name) = term {
                variables.insert(name.clone());
            }
        }
    }
    let constraints = pattern
        .constraints
        .iter()
        .filter(|constraint| variables.contains(constraint_variable(constraint)))
        .cloned()
        .collect();
    GraphPattern {
        edges,
        constraints,
        select: Vec::new(),
        distinct: pattern.distinct,
    }
}

fn minimal_unsat_edges(
    pattern: &GraphPattern,
    facts: &[GraphFact],
    options: MatchOptions,
    seed: Vec<usize>,
) -> Vec<usize> {
    let mut keep = seed;
    for index in keep.clone() {
        if keep.len() <= 1 {
            break;
        }
        let trial: Vec<usize> = keep.iter().copied().filter(|item| *item != index).collect();
        if trial.is_empty() {
            continue;
        }
        let probe = with_edges(pattern, &trial);
        if satisfiable(&probe, facts, options) == Some(false) {
            keep = trial;
        }
    }
    keep
}

fn minimal_unsat_constraints(
    pattern: &GraphPattern,
    facts: &[GraphFact],
    options: MatchOptions,
) -> Vec<usize> {
    let mut keep: Vec<usize> = (0..pattern.constraints.len()).collect();
    for index in keep.clone() {
        let trial: Vec<usize> = keep.iter().copied().filter(|item| *item != index).collect();
        let probe = with_constraints(pattern, &trial);
        if satisfiable(&probe, facts, options) == Some(false) {
            keep = trial;
        }
    }
    keep
}

fn constraint_variable(constraint: &PatternConstraint) -> &str {
    match constraint {
        PatternConstraint::Equal { variable, .. }
        | PatternConstraint::NotEqual { variable, .. } => variable,
    }
}

fn describe_edge(edge: &EdgePattern) -> String {
    format!(
        "{} —{}→ {}",
        describe_term(&edge.subject),
        edge.predicate,
        describe_term(&edge.object)
    )
}

fn describe_term(term: &PatternTerm) -> String {
    match term {
        PatternTerm::Const(value) => format!("「{}」", value),
        PatternTerm::Var(name) => format!("?{}", name),
    }
}

fn describe_constraint(constraint: &PatternConstraint) -> String {
    match constraint {
        PatternConstraint::Equal { variable, value } => {
            format!("?{} = 「{}」", variable, value)
        }
        PatternConstraint::NotEqual { variable, value } => {
            format!("?{} ≠ 「{}」", variable, value)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversational_graph::{EdgePattern, GraphPattern};

    fn facts() -> Vec<GraphFact> {
        vec![
            GraphFact::relation("张三", "导演", "电影A"),
            GraphFact::relation("李四", "导演", "电影B"),
            GraphFact::type_assertion("电影A", "科幻片"),
            GraphFact::type_assertion("电影B", "剧情片"),
            GraphFact::relation("张三", "获奖", "金棕榈"),
            GraphFact::relation("李四", "获奖", "金棕榈"),
            GraphFact::relation("姚明", "出生地", "上海"),
            GraphFact::relation("刘翔", "出生地", "上海"),
            GraphFact::relation("王五", "出生地", "北京"),
        ]
    }

    #[test]
    fn missing_predicate_is_named() {
        let pattern = GraphPattern::new(
            vec![
                EdgePattern::var_const("person", "执导", "电影A"),
                EdgePattern::var_const("person", "获奖", "金棕榈"),
            ],
            vec!["person".to_string()],
        );
        let diagnosis = diagnose_failure(&pattern, &facts(), MatchOptions::default())
            .expect("零解应产生诊断");
        assert_eq!(diagnosis.kind, DiagnosisKind::MissingPredicate);
        assert_eq!(diagnosis.failing_edges, vec![0]);
        let text = diagnosis.render(&pattern);
        assert!(text.contains("执导"), "{}", text);
    }

    #[test]
    fn missing_constant_names_the_edge() {
        let pattern = GraphPattern::new(
            vec![
                EdgePattern::const_var("姚明", "出生地", "city"),
                EdgePattern::const_var("赵六", "出生地", "city"),
            ],
            vec!["city".to_string()],
        );
        let diagnosis = diagnose_failure(&pattern, &facts(), MatchOptions::default())
            .expect("零解应产生诊断");
        assert_eq!(diagnosis.kind, DiagnosisKind::MissingConstant);
        assert_eq!(diagnosis.failing_edges, vec![1]);
        assert!(diagnosis.render(&pattern).contains("赵六"));
    }

    #[test]
    fn missing_combination_is_distinguished_from_missing_constant() {
        let pattern = GraphPattern::new(
            vec![EdgePattern::const_const("姚明", "出生地", "北京")],
            vec![],
        );
        let diagnosis = diagnose_failure(&pattern, &facts(), MatchOptions::default())
            .expect("零解应产生诊断");
        assert_eq!(diagnosis.kind, DiagnosisKind::MissingCombination);
        assert_eq!(diagnosis.edges[0].issue, EdgeIssue::MissingPair);
    }

    #[test]
    fn join_conflict_finds_a_minimal_edge_group() {
        let pattern = GraphPattern::new(
            vec![
                EdgePattern::const_var("姚明", "出生地", "city"),
                EdgePattern::const_var("王五", "出生地", "city"),
            ],
            vec!["city".to_string()],
        );
        let diagnosis = diagnose_failure(&pattern, &facts(), MatchOptions::default())
            .expect("零解应产生诊断");
        assert_eq!(diagnosis.kind, DiagnosisKind::JoinConflict);
        assert_eq!(diagnosis.failing_edges, vec![0, 1]);
        assert!(diagnosis.render(&pattern).contains("最小冲突组"));
    }

    #[test]
    fn conflicting_constraints_are_reported() {
        let pattern = GraphPattern::new(
            vec![EdgePattern::var_const("person", "获奖", "金棕榈")],
            vec!["person".to_string()],
        )
        .with_constraints(vec![
            PatternConstraint::Equal {
                variable: "person".to_string(),
                value: "张三".to_string(),
            },
            PatternConstraint::NotEqual {
                variable: "person".to_string(),
                value: "张三".to_string(),
            },
        ]);
        let diagnosis = diagnose_failure(&pattern, &facts(), MatchOptions::default())
            .expect("零解应产生诊断");
        assert_eq!(diagnosis.kind, DiagnosisKind::ConstraintConflict);
        assert_eq!(diagnosis.failing_constraints, vec![0, 1]);
    }

    #[test]
    fn satisfiable_patterns_have_no_diagnosis() {
        let pattern = GraphPattern::new(
            vec![EdgePattern::var_const("person", "获奖", "金棕榈")],
            vec!["person".to_string()],
        );
        assert!(diagnose_failure(&pattern, &facts(), MatchOptions::default()).is_none());
    }

    #[test]
    fn exhausted_budget_is_not_reported_as_missing_knowledge() {
        let pattern = GraphPattern::new(
            vec![
                EdgePattern::var_const("person", "获奖", "金棕榈"),
                EdgePattern::var_const("person", "出生地", "上海"),
            ],
            vec!["person".to_string()],
        );
        let diagnosis = diagnose_failure(&pattern, &facts(), MatchOptions::new(1, 1))
            .expect("预算耗尽应产生诊断");
        assert_eq!(diagnosis.kind, DiagnosisKind::BudgetExhausted);
    }
}

//! 图谱零解的有界放宽（溯因式修复）。
//!
//! 精确匹配永远优先：本模块只在零解诊断之后生成候选，每个候选都是对原
//! 模式的一次**显式编辑**（删一条边、去掉一条约束、沿已声明层级替换谓词
//! 或类型）。候选必须重新通过子图匹配才会被返回，且不会自动写回图谱；
//! 只有用户确认后，放宽后的模式才进入当前会话。
//!
//! 设计约束：
//! - 修正是纯函数（[`RepairOp::apply`]），不改全局状态；
//! - 默认只尝试单步修正，组合修正留给后续轮次；
//! - 声明式本体（子属性/子类）排在删边之前，结构证据优先于结果数量；
//! - 放宽结果永远带标签，原条件永远可以通过「恢复原条件」找回。

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::conversational_graph::{
    EdgePattern, GraphFact, GraphPattern, MatchOptions, PatternConstraint, PatternTerm,
};
use crate::graph_diagnosis::{DiagnosisKind, MatchDiagnosis};

/// 放宽候选的外部知识：只提议，不裁决。实现方可以来自声明式本体，
/// 也可以来自嵌入近邻；测试可注入空实现做消融。
pub trait RepairHints {
    /// 替代谓词 + 理由。
    fn predicate_alternatives(&self, _predicate: &str) -> Vec<(String, String)> {
        Vec::new()
    }

    /// 更一般的类型 + 理由。
    fn type_supertypes(&self, _entity_type: &str) -> Vec<(String, String)> {
        Vec::new()
    }
}

/// 不提供任何外部提示：只允许结构性的删边/去约束。
#[derive(Debug, Clone, Copy, Default)]
pub struct NoRepairHints;

impl RepairHints for NoRepairHints {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TermPosition {
    Subject,
    Object,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RepairOp {
    DropEdge {
        edge: usize,
    },
    DropConstraint {
        constraint: usize,
    },
    ReplacePredicate {
        edge: usize,
        predicate: String,
    },
    ReplaceConstant {
        edge: usize,
        position: TermPosition,
        value: String,
    },
}

impl RepairOp {
    /// 单步修正的编辑代价；越小越保守。
    pub fn cost(&self) -> usize {
        match self {
            RepairOp::ReplacePredicate { .. } | RepairOp::ReplaceConstant { .. } => 1,
            RepairOp::DropConstraint { .. } => 2,
            RepairOp::DropEdge { .. } => 3,
        }
    }

    /// 纯函数：返回放宽后的模式；编辑后模式不合法时返回 `None`。
    pub fn apply(&self, pattern: &GraphPattern) -> Option<GraphPattern> {
        let mut next = pattern.clone();
        match self {
            RepairOp::DropEdge { edge } => {
                if next.edges.len() <= 1 || *edge >= next.edges.len() {
                    return None;
                }
                next.edges.remove(*edge);
            }
            RepairOp::DropConstraint { constraint } => {
                if *constraint >= next.constraints.len() {
                    return None;
                }
                next.constraints.remove(*constraint);
            }
            RepairOp::ReplacePredicate { edge, predicate } => {
                let pattern_edge = next.edges.get_mut(*edge)?;
                if predicate.trim().is_empty() || pattern_edge.predicate == *predicate {
                    return None;
                }
                pattern_edge.predicate = predicate.clone();
            }
            RepairOp::ReplaceConstant {
                edge,
                position,
                value,
            } => {
                let pattern_edge = next.edges.get_mut(*edge)?;
                let term = match position {
                    TermPosition::Subject => &mut pattern_edge.subject,
                    TermPosition::Object => &mut pattern_edge.object,
                };
                match term {
                    PatternTerm::Const(existing) if existing != value && !value.trim().is_empty() => {
                        *existing = value.clone();
                    }
                    _ => return None,
                }
            }
        }
        next.validate().ok()?;
        Some(next)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepairSuggestion {
    pub op: RepairOp,
    pub rationale: String,
    /// 放宽模式下的事实数（截断时为下界）。
    pub total: usize,
    pub lower_bound: bool,
    /// 前几行投影，供用户判断是否接受；不冒充原问题答案。
    pub sample_rows: Vec<String>,
}

impl RepairSuggestion {
    pub fn label(&self) -> String {
        match &self.op {
            RepairOp::DropEdge { edge } => format!("去掉第 {} 条模式边", edge + 1),
            RepairOp::DropConstraint { constraint } => {
                format!("去掉第 {} 条约束", constraint + 1)
            }
            RepairOp::ReplacePredicate { edge, predicate } => {
                format!("把第 {} 条边的谓词放宽为「{}」", edge + 1, predicate)
            }
            RepairOp::ReplaceConstant {
                edge,
                position,
                value,
            } => {
                let slot = match position {
                    TermPosition::Subject => "主语",
                    TermPosition::Object => "宾语",
                };
                format!("把第 {} 条边的{}放宽为「{}」", edge + 1, slot, value)
            }
        }
    }
}

/// 根据诊断生成至多 `limit` 条可验证的单步放宽。
///
/// 每个候选都会重新求值；无解或模式非法的候选不会进入结果。
pub fn suggest_repairs(
    pattern: &GraphPattern,
    facts: &[GraphFact],
    options: MatchOptions,
    diagnosis: &MatchDiagnosis,
    hints: &dyn RepairHints,
    limit: usize,
) -> Vec<RepairSuggestion> {
    if diagnosis.kind == DiagnosisKind::BudgetExhausted || limit == 0 {
        return Vec::new();
    }
    let mut candidates: Vec<(RepairOp, String)> = Vec::new();
    let failing: BTreeSet<usize> = diagnosis.failing_edges.iter().copied().collect();

    // 结构性放宽：只对诊断指出的冲突边删边，避免「删到能出结果为止」。
    let edge_pool: BTreeSet<usize> = if failing.is_empty() {
        (0..pattern.edges.len()).collect()
    } else {
        failing
    };
    for edge in edge_pool {
        candidates.push((
            RepairOp::DropEdge { edge },
            format!(
                "去掉冲突边「{}」，先看其余条件能回答什么",
                describe_edge(&pattern.edges[edge])
            ),
        ));
    }
    for constraint in &diagnosis.failing_constraints {
        candidates.push((
            RepairOp::DropConstraint {
                constraint: *constraint,
            },
            format!(
                "撤回约束「{}」，回到未限定的模式",
                describe_constraint(&pattern.constraints[*constraint])
            ),
        ));
    }

    // 声明式本体提示：谓词/类型的单步替换。
    for (edge_index, edge) in pattern.edges.iter().enumerate() {
        for (alternative, reason) in hints.predicate_alternatives(&edge.predicate) {
            if alternative == edge.predicate {
                continue;
            }
            candidates.push((
                RepairOp::ReplacePredicate {
                    edge: edge_index,
                    predicate: alternative.clone(),
                },
                reason,
            ));
        }
        if edge.predicate != "type" {
            continue;
        }
        let PatternTerm::Const(current) = &edge.object else {
            continue;
        };
        for (supertype, reason) in hints.type_supertypes(current) {
            if supertype == *current {
                continue;
            }
            candidates.push((
                RepairOp::ReplaceConstant {
                    edge: edge_index,
                    position: TermPosition::Object,
                    value: supertype,
                },
                reason,
            ));
        }
    }

    let mut seen: BTreeSet<RepairOp> = BTreeSet::new();
    let mut suggestions = Vec::new();
    for (op, rationale) in candidates {
        if !seen.insert(op.clone()) {
            continue;
        }
        let Some(candidate) = op.apply(pattern) else {
            continue;
        };
        let Ok(outcome) = candidate.evaluate(facts, options) else {
            continue;
        };
        if outcome.solutions.is_empty() {
            continue;
        }
        let sample_rows: Vec<String> = outcome
            .solutions
            .iter()
            .take(3)
            .map(|solution| solution.row(&candidate.select))
            .filter(|row| !row.is_empty())
            .collect();
        suggestions.push(RepairSuggestion {
            op,
            rationale,
            total: outcome.solutions.len(),
            lower_bound: outcome.count_is_lower_bound(),
            sample_rows,
        });
    }
    suggestions.sort_by(|left, right| {
        left.op
            .cost()
            .cmp(&right.op.cost())
            // 同等代价下优先结果更窄的放宽：从 3 项变 300 项通常不是用户本意。
            .then(left.total.cmp(&right.total))
            .then(left.rationale.cmp(&right.rationale))
    });
    suggestions.truncate(limit);
    suggestions
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepairIntent {
    Adopt(usize),
    RestoreOriginal,
}

/// 解析用户对放宽候选的确认。只认明确的「放宽/采用/恢复」措辞，
/// 不把普通追问误当确认。
pub fn parse_repair_intent(text: &str) -> Option<RepairIntent> {
    let text = text
        .trim()
        .trim_end_matches(['？', '?', '！', '!', '。', '.', ' ']);
    if text.is_empty() {
        return None;
    }
    for phrase in [
        "恢复原条件",
        "恢复原来的条件",
        "恢复原查询",
        "回到原问题",
        "撤销放宽",
        "不要放宽",
        "取消放宽",
    ] {
        if text == phrase || text.contains(phrase) {
            return Some(RepairIntent::RestoreOriginal);
        }
    }
    let mut rest = text;
    let mut matched_prefix = false;
    for prefix in ["放宽", "采用", "接受", "就用", "选择", "试试", "第"] {
        if let Some(stripped) = rest.strip_prefix(prefix) {
            rest = stripped.trim();
            matched_prefix = true;
            break;
        }
    }
    if !matched_prefix {
        return None;
    }
    if rest.is_empty() {
        return Some(RepairIntent::Adopt(0));
    }
    let rest = rest.trim_start_matches(['第', ' ']);
    let rest = rest.trim_end_matches("方案").trim_end_matches("建议");
    let rest = rest.trim_end_matches(['条', '个', '项', '种', '吧', '呢', ' ']);
    parse_ordinal(rest).map(|index| RepairIntent::Adopt(index.saturating_sub(1)))
}

fn parse_ordinal(text: &str) -> Option<usize> {
    if let Ok(value) = text.parse::<usize>() {
        return (value > 0).then_some(value);
    }
    let mut value = 0usize;
    for ch in text.chars() {
        let digit = match ch {
            '一' => 1,
            '二' | '两' => 2,
            '三' => 3,
            '四' => 4,
            '五' => 5,
            '六' => 6,
            '七' => 7,
            '八' => 8,
            '九' => 9,
            '十' => 10,
            _ => return None,
        };
        value = if ch == '十' && value == 0 {
            10
        } else if ch == '十' {
            value * 10
        } else {
            value + digit
        };
    }
    (value > 0).then_some(value)
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
        PatternConstraint::Equal { variable, value } => format!("?{} = 「{}」", variable, value),
        PatternConstraint::NotEqual { variable, value } => {
            format!("?{} ≠ 「{}」", variable, value)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversational_graph::{EdgePattern, GraphPattern};
    use crate::graph_diagnosis::diagnose_failure;

    fn facts() -> Vec<GraphFact> {
        vec![
            GraphFact::relation("张三", "导演", "电影A"),
            GraphFact::relation("李四", "导演", "电影B"),
            GraphFact::type_assertion("电影A", "科幻片"),
            GraphFact::type_assertion("电影B", "剧情片"),
            GraphFact::relation("张三", "获奖", "金棕榈"),
            GraphFact::relation("李四", "获奖", "金棕榈"),
        ]
    }

    fn failing_pattern() -> GraphPattern {
        GraphPattern::new(
            vec![
                EdgePattern::var_const("person", "导演", "电影C"),
                EdgePattern::var_const("person", "获奖", "金棕榈"),
            ],
            vec!["person".to_string()],
        )
    }

    #[test]
    fn dropping_the_missing_constant_edge_recovers_a_partial_answer() {
        let pattern = failing_pattern();
        let options = MatchOptions::default();
        let diagnosis = diagnose_failure(&pattern, &facts(), options).unwrap();
        let repairs = suggest_repairs(
            &pattern,
            &facts(),
            options,
            &diagnosis,
            &NoRepairHints,
            5,
        );
        let drop = repairs
            .iter()
            .find(|repair| matches!(repair.op, RepairOp::DropEdge { edge: 0 }))
            .expect("删掉失配边应能恢复部分结果");
        assert_eq!(drop.total, 2);
        assert!(drop.sample_rows.contains(&"张三".to_string()));
    }

    #[test]
    fn every_suggestion_is_revalidated_and_labeled() {
        let pattern = failing_pattern();
        let options = MatchOptions::default();
        let diagnosis = diagnose_failure(&pattern, &facts(), options).unwrap();
        let repairs = suggest_repairs(
            &pattern,
            &facts(),
            options,
            &diagnosis,
            &NoRepairHints,
            5,
        );
        assert!(!repairs.is_empty());
        for repair in repairs {
            assert!(repair.total > 0);
            let relaxed = repair.op.apply(&pattern).unwrap();
            let outcome = relaxed.evaluate(&facts(), options).unwrap();
            assert_eq!(outcome.solutions.len(), repair.total);
        }
    }

    #[test]
    fn declared_hints_outrank_structural_drops() {
        struct Hints;
        impl RepairHints for Hints {
            fn predicate_alternatives(&self, predicate: &str) -> Vec<(String, String)> {
                if predicate == "导演" {
                    vec![(
                        "执导".to_string(),
                        "「导演」与「执导」是已声明同义谓词".to_string(),
                    )]
                } else {
                    Vec::new()
                }
            }
        }
        let mut facts = facts();
        facts.push(GraphFact::relation("王五", "执导", "电影C"));
        facts.push(GraphFact::relation("王五", "获奖", "金棕榈"));
        let pattern = failing_pattern();
        let options = MatchOptions::default();
        let diagnosis = diagnose_failure(&pattern, &facts, options).unwrap();
        let repairs = suggest_repairs(&pattern, &facts, options, &diagnosis, &Hints, 5);
        let replaced = repairs
            .iter()
            .position(|repair| matches!(repair.op, RepairOp::ReplacePredicate { .. }))
            .expect("声明式同义谓词应产生替换候选");
        let dropped = repairs
            .iter()
            .position(|repair| matches!(repair.op, RepairOp::DropEdge { .. }))
            .expect("结构性放宽仍应保留");
        assert!(replaced < dropped, "声明式提示应排在删边之前");
    }

    #[test]
    fn constraints_can_be_withdrawn_to_restore_the_unfiltered_mode() {
        let pattern = GraphPattern::new(
            vec![EdgePattern::var_const("person", "获奖", "金棕榈")],
            vec!["person".to_string()],
        )
        .with_constraints(vec![PatternConstraint::Equal {
            variable: "person".to_string(),
            value: "王五".to_string(),
        }]);
        let options = MatchOptions::default();
        let diagnosis = diagnose_failure(&pattern, &facts(), options).unwrap();
        assert_eq!(diagnosis.kind, DiagnosisKind::ConstraintConflict);
        let repairs = suggest_repairs(
            &pattern,
            &facts(),
            options,
            &diagnosis,
            &NoRepairHints,
            5,
        );
        assert!(repairs
            .iter()
            .any(|repair| matches!(repair.op, RepairOp::DropConstraint { .. })));
    }

    #[test]
    fn confirmation_phrases_are_narrow_and_ordinal() {
        assert_eq!(
            parse_repair_intent("放宽 2"),
            Some(RepairIntent::Adopt(1))
        );
        assert_eq!(
            parse_repair_intent("采用第二条"),
            Some(RepairIntent::Adopt(1))
        );
        assert_eq!(
            parse_repair_intent("第1个方案"),
            Some(RepairIntent::Adopt(0))
        );
        assert_eq!(
            parse_repair_intent("恢复原条件"),
            Some(RepairIntent::RestoreOriginal)
        );
        assert_eq!(parse_repair_intent("还有呢"), None);
        assert_eq!(parse_repair_intent("为什么"), None);
    }

    #[test]
    fn negative_ordinals_are_rejected() {
        assert_eq!(parse_repair_intent("放宽 0"), None);
        assert_eq!(parse_repair_intent("放宽 很多"), None);
    }
}

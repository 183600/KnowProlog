//! 会话修订守门器（RevisionGuard）：把 [`BeliefState`] 的事务能力包装成
//! 面向对话写入口的「预检 → 最小冲突核 → 提交/拒绝 → 撤销」。
//!
//! 与 `belief_revision` 的分工：
//!   - `BeliefState` 负责计划、一致性闸门、影响报告与原子提交；
//!   - `RevisionGuard` 负责在写入会话事实/规则之前先试算，冲突时把整批冲突
//!     收缩成**最小操作核**，并生成一个可回答的中文裁决问题；
//!   - 重复规则/重复断言等提示级冲突不阻止写入，真正危险的冲突才会拒绝。
//!
//! 设计边界：冲突核通过「逐个删除操作后重新计划」得到 1-最小核；操作数超过
//! [`MAX_KERNEL_OPS`] 时不再收缩并显式标记 `minimal = false`，不假装最小。

use crate::belief_revision::{
    BeliefState, ChangeOp, ChangeSet, CommitError, CommitReceipt, ConflictKind, ConflictSeverity,
    RevisionConflict, RevisionConstraints, RevisionPlan,
};
use crate::resolution_engine::{Atom, Clause, ResolutionEngine};

/// 冲突核收缩最多处理多少个候选操作；超过则保留整批并标记非最小。
pub const MAX_KERNEL_OPS: usize = 32;
/// 冲突核最多保留多少个（防止一次写入产生海量问句）。
pub const MAX_KERNELS: usize = 8;

/// 一个最小冲突核：让候选世界不一致的最小操作集合 + 支撑原子。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConflictKernel {
    pub kind: ConflictKind,
    pub severity: ConflictSeverity,
    pub detail: String,
    /// 核内操作的中文描述（稳定排序）。
    pub ops: Vec<String>,
    /// 冲突涉及的规范原子（来自 `RevisionConflict::atoms`）。
    pub atoms: Vec<String>,
    /// 该核是否已收缩到 1-最小。
    pub minimal: bool,
    /// 面向用户的单一裁决问题。
    pub question: String,
}

impl ConflictKernel {
    pub fn render_zh(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "最小冲突核{}（{}）：{}",
            if self.minimal { "" } else { "（未收缩）" },
            self.kind.label_zh(),
            self.detail
        ));
        if !self.ops.is_empty() {
            lines.push("  本次操作：".to_string());
            for op in &self.ops {
                lines.push(format!("    · {}", op));
            }
        }
        if !self.atoms.is_empty() {
            lines.push("  支撑原子：".to_string());
            for atom in &self.atoms {
                lines.push(format!("    · {}", atom));
            }
        }
        lines.push(format!("  需要裁决：{}", self.question));
        lines.join("\n")
    }
}

/// 预检结果：原始计划 + 写入级最小冲突核。
#[derive(Debug, Clone)]
pub struct GuardPlan {
    pub plan: RevisionPlan,
    pub kernels: Vec<ConflictKernel>,
}

impl GuardPlan {
    /// 是否存在会阻止写入的冲突（提示级重复不算）。
    pub fn is_blocked(&self) -> bool {
        !self.kernels.is_empty()
    }

    pub fn render_zh(&self) -> String {
        if self.kernels.is_empty() {
            return self.plan.render_zh();
        }
        let mut lines = vec![format!(
            "修订预检：写入被阻止（{} 个最小冲突核），状态未改变。",
            self.kernels.len()
        )];
        for kernel in &self.kernels {
            lines.push(kernel.render_zh());
        }
        if self.plan.impact.changed_atom_count() > 0 {
            lines.push(self.plan.impact.render_zh());
        }
        lines.join("\n")
    }
}

/// 可恢复的提交记录：除标准回执外，额外携带需要一并恢复的教学原句。
#[derive(Debug, Clone)]
pub struct GuardReceipt {
    pub receipt: CommitReceipt,
    pub restore_lessons: Vec<String>,
}

/// 面向会话写入的修订守门器。内部持有一个独立 `BeliefState`，调用方在
/// 预检通过后提交，再把回执中的有效操作镜像到自己的会话状态。
#[derive(Clone)]
pub struct RevisionGuard {
    state: BeliefState,
}

impl RevisionGuard {
    pub fn new(engine: ResolutionEngine, constraints: RevisionConstraints) -> Self {
        Self {
            state: BeliefState::with_constraints(engine, constraints),
        }
    }

    pub fn from_state(state: BeliefState) -> Self {
        Self { state }
    }

    pub fn state(&self) -> &BeliefState {
        &self.state
    }

    pub fn revision(&self) -> u64 {
        self.state.revision()
    }

    /// 预览并计算写入级冲突核；不修改状态。
    pub fn plan(&self, changeset: ChangeSet) -> GuardPlan {
        let mut plan = self.state.plan(changeset);
        // 重复规则只是提示：同一形状已有规则时不应阻止一次幂等写入。
        for conflict in &mut plan.conflicts {
            if conflict.kind == ConflictKind::DuplicateRule {
                conflict.severity = ConflictSeverity::Warning;
            }
        }
        let kernels = extract_kernels(&self.state, &plan);
        GuardPlan { plan, kernels }
    }

    /// 提交预检通过的计划；存在写入级冲突时拒绝且状态不变。
    pub fn commit(&mut self, plan: GuardPlan) -> Result<GuardReceipt, CommitError> {
        if let Some(first) = plan.kernels.first() {
            return Err(CommitError::Blocked {
                count: plan.kernels.len(),
                first: first.detail.clone(),
            });
        }
        let receipt = self.state.commit(plan.plan)?;
        Ok(GuardReceipt {
            receipt,
            restore_lessons: Vec::new(),
        })
    }

    /// 预览并提交的便捷入口。
    pub fn apply(&mut self, changeset: ChangeSet) -> Result<GuardReceipt, CommitError> {
        let plan = self.plan(changeset);
        self.commit(plan)
    }

    pub fn undo_last(&mut self) -> Result<CommitReceipt, CommitError> {
        self.state.undo_last()
    }

    pub fn facts(&self) -> Vec<Atom> {
        self.state.facts()
    }

    pub fn rule_count(&self) -> usize {
        self.state.rule_count()
    }
}

fn write_blocking(conflict: &RevisionConflict) -> bool {
    conflict.severity == ConflictSeverity::Blocking
        && matches!(
            conflict.kind,
            ConflictKind::FunctionalViolation
                | ConflictKind::DomainViolation
                | ConflictKind::RangeViolation
                | ConflictKind::NonGroundFact
                | ConflictKind::UnsafeRule
                | ConflictKind::UnconditionalRule
                | ConflictKind::RuleResurrection
                | ConflictKind::StalePlan
        )
}

fn subset(changeset: &ChangeSet, ops: &[ChangeOp]) -> ChangeSet {
    ChangeSet {
        label: changeset.label.clone(),
        ops: ops.to_vec(),
        expected_revision: changeset.expected_revision,
    }
}

fn extract_kernels(state: &BeliefState, plan: &RevisionPlan) -> Vec<ConflictKernel> {
    if !plan.conflicts.iter().any(write_blocking) {
        return Vec::new();
    }
    let (ops, minimal) = shrink_ops(state, plan);
    let minimized = if ops.len() < plan.changeset.ops.len() {
        state.plan(subset(&plan.changeset, &ops))
    } else {
        plan.clone()
    };
    let mut kernels = Vec::new();
    for conflict in minimized.conflicts.iter().filter(|conflict| write_blocking(conflict)) {
        let involved: Vec<String> = ops
            .iter()
            .filter(|op| op_touches(op, conflict))
            .map(ChangeOp::describe)
            .collect();
        let ops = if involved.is_empty() {
            ops.iter().map(ChangeOp::describe).collect()
        } else {
            involved
        };
        kernels.push(ConflictKernel {
            kind: conflict.kind,
            severity: conflict.severity,
            detail: conflict.detail.clone(),
            ops,
            atoms: conflict.atoms.clone(),
            minimal,
            question: question_for(conflict),
        });
        if kernels.len() >= MAX_KERNELS {
            break;
        }
    }
    kernels
}

/// 删除法收缩：反复尝试删掉一个操作，只要仍有写入级冲突就保留删除结果。
fn shrink_ops(state: &BeliefState, plan: &RevisionPlan) -> (Vec<ChangeOp>, bool) {
    let mut current = plan.changeset.ops.clone();
    if current.len() > MAX_KERNEL_OPS {
        return (current, false);
    }
    let mut index = 0;
    while index < current.len() {
        if current.len() <= 1 {
            break;
        }
        let mut trial = current.clone();
        trial.remove(index);
        let trial_plan = state.plan(subset(&plan.changeset, &trial));
        if trial_plan.conflicts.iter().any(write_blocking) {
            current = trial;
        } else {
            index += 1;
        }
    }
    (current, true)
}

fn op_touches(op: &ChangeOp, conflict: &RevisionConflict) -> bool {
    match op {
        ChangeOp::AssertFact(atom) | ChangeOp::RetractFact(atom) => conflict
            .atoms
            .iter()
            .any(|candidate| candidate == &atom.display()),
        ChangeOp::AddRule(clause) | ChangeOp::RemoveRule(clause) => {
            rule_touches(clause, conflict)
        }
    }
}

fn rule_touches(clause: &Clause, conflict: &RevisionConflict) -> bool {
    let head = clause.head.display();
    conflict
        .atoms
        .iter()
        .any(|candidate| candidate.contains(head.as_str()))
}

fn question_for(conflict: &RevisionConflict) -> String {
    match conflict.kind {
        ConflictKind::FunctionalViolation => {
            if let Some((predicate, subject, values)) = functional_parts(conflict) {
                format!(
                    "「{}」的「{}」出现了多个取值：{}。请选择保留哪一个，或撤回这次写入。",
                    subject,
                    predicate,
                    values.join(" / ")
                )
            } else {
                format!(
                    "{}。请选择保留哪一个取值，或撤回这次写入。",
                    conflict.detail
                )
            }
        }
        ConflictKind::RuleResurrection => format!(
            "{}。要保留该结论并放弃撤回，还是同时停用推出它的规则？",
            conflict.detail
        ),
        ConflictKind::NonGroundFact | ConflictKind::UnsafeRule => format!(
            "{}。请补齐变量绑定后重试，或改写为具体事实。",
            conflict.detail
        ),
        ConflictKind::DomainViolation | ConflictKind::RangeViolation => format!(
            "{}。请确认是采纳这次类型判断，还是修正实体类型？",
            conflict.detail
        ),
        ConflictKind::UnconditionalRule => format!(
            "{}。请补充规则体部条件，或改写为具体事实。",
            conflict.detail
        ),
        _ => format!("{}。请确认是否继续这次写入。", conflict.detail),
    }
}

fn functional_parts(conflict: &RevisionConflict) -> Option<(String, String, Vec<String>)> {
    let first = conflict.atoms.first()?;
    let open = first.find('(')?;
    let predicate = first[..open].to_string();
    let inner = first[open + 1..].strip_suffix(')')?;
    let (subject, _) = inner.split_once(", ")?;
    let values: Vec<String> = conflict
        .atoms
        .iter()
        .filter_map(|atom| {
            let open = atom.find('(')?;
            let inner = atom[open + 1..].strip_suffix(')')?;
            let (_, value) = inner.split_once(", ")?;
            Some(value.to_string())
        })
        .collect();
    Some((predicate, subject.to_string(), values))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine_with_functional_age() -> (ResolutionEngine, RevisionConstraints) {
        let mut engine = ResolutionEngine::new(6, 12);
        engine.add_fact("年龄", "张三", "30");
        engine.add_fact("师傅", "张三", "李四");
        let constraints = RevisionConstraints::new().require_functional("年龄");
        (engine, constraints)
    }

    #[test]
    fn functional_conflict_shrinks_to_one_op_and_blocks_commit() {
        let (engine, constraints) = engine_with_functional_age();
        let mut guard = RevisionGuard::new(engine, constraints);
        let before = guard.facts();
        let changeset = ChangeSet::new("冲突写入")
            .assert("年龄", "张三", "31")
            .assert("师傅", "王五", "赵六");
        let plan = guard.plan(changeset);
        assert!(plan.is_blocked(), "功能谓词多值必须阻止写入");
        assert_eq!(plan.kernels.len(), 1);
        let kernel = &plan.kernels[0];
        assert_eq!(kernel.kind, ConflictKind::FunctionalViolation);
        assert!(kernel.minimal, "操作数未超限时必须收缩");
        assert_eq!(kernel.ops.len(), 1, "只应保留真正冲突的那个操作");
        assert!(kernel.ops[0].contains("年龄"), "{}", kernel.ops[0]);
        assert!(kernel.question.contains("30"), "{}", kernel.question);
        assert!(kernel.question.contains("31"), "{}", kernel.question);

        let error = guard.commit(plan).expect_err("阻止级冲突不得提交");
        assert!(matches!(error, CommitError::Blocked { .. }));
        assert_eq!(guard.revision(), 0, "拒绝提交不得推进版本");
        assert_eq!(guard.facts(), before, "拒绝提交不得留下半套事实");
    }

    #[test]
    fn clean_commit_then_undo_restores_exact_state() {
        let (mut engine, constraints) = engine_with_functional_age();
        engine.add_clause(Clause::new(
            Atom::rel("老师", "X", "Y"),
            vec![Atom::pattern(
                "师傅",
                crate::resolution_engine::Term::var("X"),
                crate::resolution_engine::Term::var("Y"),
            )],
            0.9,
            "测试规则",
        ));
        let mut guard = RevisionGuard::new(engine, constraints);
        let facts_before = guard.facts();
        let rules_before = guard.rule_count();

        let receipt = guard
            .apply(ChangeSet::new("新增事实").assert("师傅", "王五", "赵六"))
            .expect("无冲突写入应提交");
        assert_eq!(receipt.receipt.applied_ops, 1);
        assert_eq!(guard.revision(), 1);
        assert!(guard.facts().iter().any(|atom| atom.display() == "师傅(王五, 赵六)"));

        guard.undo_last().expect("应能撤销最新回执");
        assert_eq!(guard.revision(), 2, "撤销是一次新的提交");
        assert_eq!(guard.facts(), facts_before, "撤销必须逐字恢复事实集");
        assert_eq!(guard.rule_count(), rules_before, "撤销必须恢复规则数");
    }

    #[test]
    fn duplicate_assertion_is_warning_but_not_blocking() {
        let (engine, constraints) = engine_with_functional_age();
        let mut guard = RevisionGuard::new(engine, constraints);
        let plan = guard.plan(ChangeSet::new("重复断言").assert("年龄", "张三", "30"));
        assert!(!plan.is_blocked(), "重复断言只应提示，不应阻止");
        assert!(plan
            .plan
            .conflicts
            .iter()
            .any(|conflict| conflict.kind == ConflictKind::DuplicateAssertion));
        let receipt = guard.commit(plan).expect("重复断言可以幂等提交");
        assert_eq!(receipt.receipt.applied_ops, 0, "重复断言不产生有效操作");
    }

    #[test]
    fn rule_that_resurrects_retracted_fact_is_blocked() {
        let mut engine = ResolutionEngine::new(6, 12);
        engine.add_fact("师傅", "张三", "李四");
        engine.add_fact("同事", "张三", "李四");
        let rule = Clause::new(
            Atom::rel("师傅", "张三", "李四"),
            vec![Atom::pattern(
                "同事",
                crate::resolution_engine::Term::var("X"),
                crate::resolution_engine::Term::var("Y"),
            )],
            0.9,
            "测试规则",
        );
        engine.add_clause(rule);
        let mut guard = RevisionGuard::new(engine, RevisionConstraints::new());
        let plan = guard.plan(ChangeSet::new("撤回事实").retract("师傅", "张三", "李四"));
        assert!(plan.is_blocked(), "规则仍能推出该事实时必须阻止静默撤回");
        let kernel = &plan.kernels[0];
        assert_eq!(kernel.kind, ConflictKind::RuleResurrection);
        assert!(kernel.question.contains("停用"), "{}", kernel.question);
        assert!(guard
            .commit(plan)
            .expect_err("复活冲突不得提交")
            .to_string()
            .contains("阻止"));
    }
}

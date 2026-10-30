fn schema_conflicts(
    closure: &Closure,
    constraints: &RevisionConstraints,
) -> Vec<RevisionConflict> {
    if constraints.domain.is_empty() && constraints.range.is_empty() {
        return Vec::new();
    }
    let types = explicit_types(closure, &constraints.type_predicates);
    let mut out = Vec::new();
    for atom in closure.believed.values() {
        if atom.terms.len() != 2 {
            continue;
        }
        let (Some(subject), Some(object)) = (constant(&atom.terms[0]), constant(&atom.terms[1]))
        else {
            continue;
        };
        let predicate = atom.predicate.as_ref();
        if let Some(allowed) = constraints
            .domain
            .get(predicate)
            .or_else(|| constraints.domain.get(predicate.trim()))
        {
            let actual = types.get(subject).cloned().unwrap_or_default();
            if constraints.types.sets_admit(&actual, allowed) == TypeVerdict::Disjoint {
                out.push(RevisionConflict::new(
                    ConflictKind::DomainViolation,
                    ConflictSeverity::Blocking,
                    format!(
                        "「{}」的主语「{}」类型为 {}，不在定义域 {} 内",
                        predicate,
                        subject,
                        join_set(&actual),
                        join_set(allowed)
                    ),
                    vec![atom_key(atom)],
                ));
            }
        }
        if let Some(allowed) = constraints
            .range
            .get(predicate)
            .or_else(|| constraints.range.get(predicate.trim()))
        {
            let actual = types.get(object).cloned().unwrap_or_default();
            if constraints.types.sets_admit(&actual, allowed) == TypeVerdict::Disjoint {
                out.push(RevisionConflict::new(
                    ConflictKind::RangeViolation,
                    ConflictSeverity::Blocking,
                    format!(
                        "「{}」的宾语「{}」类型为 {}，不在值域 {} 内",
                        predicate,
                        object,
                        join_set(&actual),
                        join_set(allowed)
                    ),
                    vec![atom_key(atom)],
                ));
            }
        }
    }
    out
}
fn explicit_types(
    closure: &Closure,
    type_predicates: &BTreeSet<String>,
) -> BTreeMap<String, BTreeSet<String>> {
    let mut types: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for atom in closure.base.values() {
        if atom.terms.len() != 2 || !type_predicates.contains(&*atom.predicate) {
            continue;
        }
        let (Some(subject), Some(type_name)) =
            (constant(&atom.terms[0]), constant(&atom.terms[1]))
        else {
            continue;
        };
        types
            .entry(subject.to_string())
            .or_default()
            .insert(type_name.to_string());
    }
    types
}
fn build_impact(
    base: &Closure,
    candidate: &Closure,
    effects: &OpEffects,
    strict: bool,
) -> ImpactReport {
    let mut direct_added = effects.added_facts.clone();
    let mut direct_removed = effects.removed_facts.clone();
    direct_added.sort();
    direct_added.dedup();
    direct_removed.sort();
    direct_removed.dedup();

    let mut derived_added: Vec<String> = candidate
        .derived
        .keys()
        .filter(|key| !base.derived.contains_key(*key))
        .cloned()
        .collect();
    let mut derived_retracted: Vec<String> = base
        .derived
        .keys()
        .filter(|key| !candidate.derived.contains_key(*key))
        .cloned()
        .collect();
    let mut newly_believed: Vec<String> = candidate
        .believed
        .keys()
        .filter(|key| !base.believed.contains_key(*key))
        .cloned()
        .collect();
    let mut no_longer_believed: Vec<String> = base
        .believed
        .keys()
        .filter(|key| !candidate.believed.contains_key(*key))
        .cloned()
        .collect();
    derived_added.sort();
    derived_retracted.sort();
    newly_believed.sort();
    no_longer_believed.sort();

    let mut predicted_added = BTreeSet::new();
    for key in &direct_added {
        if let Some(atom) = candidate.base.get(key) {
            predicted_added.extend(candidate.graph.dependents_of(atom));
        }
    }
    for source in &effects.added_rule_sources {
        if source.is_empty() {
            continue;
        }
        for (key, fact) in &candidate.derived {
            if &fact.rule_source != source {
                continue;
            }
            predicted_added.insert(key.clone());
            if let Some(atom) = candidate.believed.get(key) {
                predicted_added.extend(candidate.graph.dependents_of(atom));
            }
        }
    }

    let mut predicted_retracted = BTreeSet::new();
    for key in &direct_removed {
        if let Some(atom) = base.base.get(key) {
            predicted_retracted.extend(base.graph.dependents_of(atom));
        }
    }
    for source in &effects.removed_rule_sources {
        if source.is_empty() {
            continue;
        }
        for (key, fact) in &base.derived {
            if &fact.rule_source != source {
                continue;
            }
            predicted_retracted.insert(key.clone());
            if let Some(atom) = base.believed.get(key) {
                predicted_retracted.extend(base.graph.dependents_of(atom));
            }
        }
    }

    let predicted_added: Vec<String> = predicted_added.into_iter().collect();
    let predicted_retracted: Vec<String> = predicted_retracted.into_iter().collect();
    let unpredicted_added: Vec<String> = derived_added
        .iter()
        .filter(|key| !predicted_added.contains(*key))
        .cloned()
        .collect();
    let unpredicted_retracted: Vec<String> = derived_retracted
        .iter()
        .filter(|key| !predicted_retracted.contains(*key))
        .cloned()
        .collect();

    let mut affected = BTreeSet::new();
    for key in newly_believed
        .iter()
        .chain(no_longer_believed.iter())
        .chain(derived_added.iter())
        .chain(derived_retracted.iter())
    {
        if let Some(atom) = candidate
            .believed
            .get(key)
            .or_else(|| base.believed.get(key))
        {
            affected.insert(atom.predicate.to_string());
        }
    }

    let truncated = base.truncated || candidate.truncated;
    let fallback = truncated || !unpredicted_added.is_empty() || !unpredicted_retracted.is_empty();
    let note = if truncated {
        Some("前向推导达到预算上限".to_string())
    } else if !unpredicted_added.is_empty() || !unpredicted_retracted.is_empty() {
        Some(format!(
            "理由图只保留了每条派生事实的一组支持，{} 条变化需要完整重算确认",
            unpredicted_added.len() + unpredicted_retracted.len()
        ))
    } else if strict && !derived_added.is_empty() && effects.added_rule_sources.is_empty() {
        Some("新增事实导致的派生变化已由理由图覆盖".to_string())
    } else {
        None
    };

    let affected_predicates: Vec<String> = affected.into_iter().collect();
    let counts = ImpactCounts {
        direct_added: direct_added.len(),
        direct_removed: direct_removed.len(),
        derived_added: derived_added.len(),
        derived_retracted: derived_retracted.len(),
        newly_believed: newly_believed.len(),
        no_longer_believed: no_longer_believed.len(),
        predicted_derived_added: predicted_added.len(),
        predicted_derived_retracted: predicted_retracted.len(),
        unpredicted_added: unpredicted_added.len(),
        unpredicted_retracted: unpredicted_retracted.len(),
        affected_predicates: affected_predicates.len(),
    };

    ImpactReport {
        direct_added,
        direct_removed,
        derived_added,
        derived_retracted,
        newly_believed,
        no_longer_believed,
        predicted_derived_added: predicted_added,
        predicted_derived_retracted: predicted_retracted,
        unpredicted_added,
        unpredicted_retracted,
        affected_predicates,
        fallback_full_recompute: fallback,
        complete: !fallback,
        note,
        counts,
        omitted_items: 0,
    }
}
fn sort_conflicts(conflicts: &mut Vec<RevisionConflict>) {
    conflicts.sort_by(|left, right| {
        right
            .severity
            .cmp(&left.severity)
            .then(left.kind.cmp(&right.kind))
            .then(left.detail.cmp(&right.detail))
            .then(left.atoms.cmp(&right.atoms))
    });
    conflicts.dedup_by(|left, right| left.signature() == right.signature());
}
fn state_digest(engine: &ResolutionEngine) -> String {
    let mut facts: Vec<String> = engine.all_facts().map(atom_key).collect();
    facts.sort();
    let mut rules: Vec<String> = engine.all_clauses().iter().map(clause_key).collect();
    rules.sort();
    let mut material = String::new();
    for fact in facts {
        material.push_str(&fact);
        material.push('\n');
    }
    material.push_str("--\n");
    for rule in rules {
        material.push_str(&rule);
        material.push('\n');
    }
    fnv1a_hex(material.as_bytes())
}
fn atom_key(atom: &Atom) -> String {
    atom.display()
}
fn clause_key(clause: &Clause) -> String {
    let body = clause
        .body
        .iter()
        .map(atom_key)
        .collect::<Vec<_>>()
        .join(" & ");
    format!(
        "{} :- {} [{}, {:.4}]",
        atom_key(&clause.head),
        if body.is_empty() { "⊤" } else { body.as_str() },
        clause.source,
        clause.confidence
    )
}
/// 规则形状：头部与体部的逻辑形式，忽略来源与置信度。删除规则按形状匹配。
fn rule_shape_key(clause: &Clause) -> String {
    let body = clause
        .body
        .iter()
        .map(atom_key)
        .collect::<Vec<_>>()
        .join(" & ");
    format!("{} :- {}", atom_key(&clause.head), body)
}
fn constant(term: &crate::resolution_engine::Term) -> Option<&str> {
    match term {
        crate::resolution_engine::Term::Const(name) => Some(name),
        _ => None,
    }
}
fn join_set(set: &BTreeSet<String>) -> String {
    if set.is_empty() {
        "未知".to_string()
    } else {
        set.iter().cloned().collect::<Vec<_>>().join(" / ")
    }
}
fn clip(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let mut out: String = text.chars().take(max_chars).collect();
    out.push('…');
    out
}
// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::resolution_engine::{Atom, Clause, ResolutionEngine, Term};

    fn rule(head: Atom, body: Vec<Atom>, source: &str) -> Clause {
        Clause::new(head, body, 0.95, source)
    }

    fn var_rule(head_predicate: &str, body_predicate: &str, source: &str) -> Clause {
        rule(
            Atom::pattern(
                head_predicate,
                Term::var("X"),
                Term::var("Y"),
            ),
            vec![Atom::pattern(
                body_predicate,
                Term::var("X"),
                Term::var("Y"),
            )],
            source,
        )
    }

    fn engine_with_teaches() -> ResolutionEngine {
        let mut engine = ResolutionEngine::new(16, 64);
        engine.add_fact("teaches", "张三", "李四");
        engine.add_fact("teaches", "李四", "王五");
        engine.add_clause(var_rule("teacher_of", "teaches", "教学规则"));
        engine
    }

    #[test]
    fn correction_is_atomic_and_retracts_dependents() {
        let base = engine_with_teaches();
        let mut state = BeliefState::new(base);
        assert_eq!(state.revision(), 0);

        let plan = state.plan(
            ChangeSet::new("更正师傅")
                .retract("teaches", "张三", "李四")
                .assert("teaches", "张三", "赵六"),
        );
        assert!(!plan.is_blocked(), "{}", plan.render_zh());
        assert!(plan
            .impact
            .derived_retracted
            .iter()
            .any(|atom| atom == "teacher_of(张三, 李四)"));
        assert!(plan
            .impact
            .derived_added
            .iter()
            .any(|atom| atom == "teacher_of(张三, 赵六)"));
        assert!(plan.impact.complete);

        let receipt = state.commit(plan).expect("commit");
        assert_eq!(receipt.base_revision, 0);
        assert_eq!(receipt.revision, 1);
        assert_eq!(state.revision(), 1);
        let facts: Vec<String> = state.facts().iter().map(atom_key).collect();
        assert!(facts.contains(&"teaches(张三, 赵六)".to_string()));
        assert!(!facts.contains(&"teaches(张三, 李四)".to_string()));
        let closure = state.closure(state.engine());
        assert!(closure.believed.contains_key("teacher_of(张三, 赵六)"));
        assert!(!closure.believed.contains_key("teacher_of(张三, 李四)"));
    }

    #[test]
    fn what_if_does_not_mutate_and_commit_is_stale_after_another_commit() {
        let mut state = BeliefState::new(ResolutionEngine::new(8, 16));
        let plan_a = state.plan(ChangeSet::new("a").assert("p", "a", "b"));
        let plan_b = state.plan(ChangeSet::new("b").assert("p", "c", "d"));
        assert!(!state.engine().has_fact(&Atom::rel("p", "a", "b")));
        state.commit(plan_a).expect("first commit");
        let error = state.commit(plan_b).expect_err("stale plan must fail");
        assert_eq!(
            error,
            CommitError::StalePlan {
                expected: 0,
                actual: 1
            }
        );
        assert!(!state.engine().has_fact(&Atom::rel("p", "c", "d")));
    }

    #[test]
    fn strict_mode_blocks_functional_conflicts() {
        let mut engine = ResolutionEngine::new(8, 16);
        engine.add_fact("height", "姚明", "226");
        let mut state = BeliefState::with_constraints(
            engine,
            RevisionConstraints::new().require_functional("height"),
        );
        let plan = state.plan(ChangeSet::new("改身高但不撤旧值").assert("height", "姚明", "229"));
        assert!(plan.is_blocked());
        assert!(plan
            .conflicts
            .iter()
            .any(|conflict| conflict.kind == ConflictKind::FunctionalViolation));
        let error = state.commit(plan).expect_err("must be blocked");
        assert!(matches!(error, CommitError::Blocked { .. }));
        assert!(state.engine().has_fact(&Atom::rel("height", "姚明", "226")));
        assert!(!state.engine().has_fact(&Atom::rel("height", "姚明", "229")));
        assert_eq!(state.revision(), 0);
    }

    #[test]
    fn retract_and_assert_passes_functional_gate() {
        let mut engine = ResolutionEngine::new(8, 16);
        engine.add_fact("height", "姚明", "226");
        let mut state = BeliefState::with_constraints(
            engine,
            RevisionConstraints::new().require_functional("height"),
        );
        let receipt = state
            .apply(
                ChangeSet::new("更正身高")
                    .retract("height", "姚明", "226")
                    .assert("height", "姚明", "229"),
            )
            .expect("commit");
        assert_eq!(receipt.revision, 1);
        assert!(state.engine().has_fact(&Atom::rel("height", "姚明", "229")));
        assert!(!state.engine().has_fact(&Atom::rel("height", "姚明", "226")));
    }

    #[test]
    fn domain_and_range_conflicts_are_checked_against_types() {
        let mut engine = ResolutionEngine::new(8, 16);
        engine.add_fact("type", "张三", "Person");
        engine.add_fact("type", "北京", "City");
        let mut types = TypeSystem::new();
        types.declare_disjoint("Person", "City");
        let constraints = RevisionConstraints::new()
            .require_domain("born_in", ["Person"])
            .require_range("born_in", ["City"])
            .with_types(types);
        let state = BeliefState::with_constraints(engine, constraints);
        let plan = state.plan(ChangeSet::new("类型错误").assert("born_in", "北京", "张三"));
        assert!(plan.is_blocked());
        assert!(plan
            .conflicts
            .iter()
            .any(|conflict| conflict.kind == ConflictKind::DomainViolation
                || conflict.kind == ConflictKind::RangeViolation));
    }

    #[test]
    fn strict_mode_blocks_rule_resurrection() {
        let mut engine = ResolutionEngine::new(8, 16);
        engine.add_fact("teaches", "张三", "李四");
        engine.add_fact("teacher_of", "张三", "李四");
        engine.add_clause(var_rule("teacher_of", "teaches", "教学规则"));
        let mut state = BeliefState::new(engine);
        let plan = state.plan(ChangeSet::new("撤回但规则仍推出").retract("teacher_of", "张三", "李四"));
        assert!(plan.is_blocked());
        assert!(plan
            .conflicts
            .iter()
            .any(|conflict| conflict.kind == ConflictKind::RuleResurrection));
        let error = state.commit(plan).expect_err("strict must block");
        assert!(matches!(error, CommitError::Blocked { .. }));
    }

    #[test]
    fn non_strict_commit_keeps_resurrected_fact_as_derived() {
        let mut engine = ResolutionEngine::new(8, 16);
        engine.add_fact("teaches", "张三", "李四");
        engine.add_fact("teacher_of", "张三", "李四");
        engine.add_clause(var_rule("teacher_of", "teaches", "教学规则"));
        let mut state = BeliefState::new(engine).with_strict(false);
        let receipt = state
            .apply(ChangeSet::new("撤回但保留推导").retract("teacher_of", "张三", "李四"))
            .expect("non-strict commit");
        assert!(receipt
            .conflicts
            .iter()
            .any(|conflict| conflict.kind == ConflictKind::RuleResurrection));
        assert!(!state.engine().has_fact(&Atom::rel("teacher_of", "张三", "李四")));
        let closure = state.closure(state.engine());
        assert!(closure.derived.contains_key("teacher_of(张三, 李四)"));
    }

    #[test]
    fn reason_graph_explains_derived_fact() {
        let state = BeliefState::new(engine_with_teaches());
        let text = state
            .explain(&Atom::rel("teacher_of", "张三", "李四"))
            .expect("derived fact should be explainable");
        assert!(text.contains("教学规则"));
        assert!(text.contains("teaches(张三, 李四)"));
    }

    #[test]
    fn undo_last_restores_previous_beliefs_without_rewriting_history() {
        let base = engine_with_teaches();
        let mut state = BeliefState::new(base);
        let receipt = state
            .apply(ChangeSet::new("加事实").assert("teaches", "王五", "赵六"))
            .expect("commit");
        assert!(state.engine().has_fact(&Atom::rel("teaches", "王五", "赵六")));
        let undo = state.undo_last().expect("undo");
        assert_eq!(undo.revision, receipt.revision + 1);
        assert!(!state.engine().has_fact(&Atom::rel("teaches", "王五", "赵六")));
        assert_eq!(state.receipts().len(), 2);
        assert_eq!(state.receipts()[0].id, receipt.id);
    }

    #[test]
    fn rule_changes_are_reported_and_can_be_removed() {
        let mut state = BeliefState::new(engine_with_teaches());
        let rule = var_rule("teacher_of", "teaches", "教学规则");
        let receipt = state
            .apply(ChangeSet::new("忘记规则").remove_rule(rule))
            .expect("commit");
        assert!(receipt
            .impact
            .no_longer_believed
            .iter()
            .any(|atom| atom == "teacher_of(张三, 李四)"));
        assert_eq!(state.rule_count(), 0);
        let closure = state.closure(state.engine());
        assert!(!closure.derived.contains_key("teacher_of(张三, 李四)"));
    }

    #[test]
    fn undo_does_not_invert_no_op_operations() {
        let mut engine = ResolutionEngine::new(8, 16);
        engine.add_fact("p", "a", "b");
        let mut state = BeliefState::new(engine).with_strict(false);
        let receipt = state
            .apply(ChangeSet::new("重复断言").assert("p", "a", "b"))
            .expect("commit");
        assert_eq!(receipt.applied_ops, 0);
        state.undo_last().expect("undo");
        assert!(state.engine().has_fact(&Atom::rel("p", "a", "b")));
    }

    #[test]
    fn pre_existing_duplicate_rule_does_not_block_unrelated_commit() {
        let mut engine = ResolutionEngine::new(8, 16);
        engine.add_fact("teaches", "张三", "李四");
        engine.add_clause(var_rule("teacher_of", "teaches", "规则甲"));
        engine.add_clause(var_rule("teacher_of", "teaches", "规则乙"));
        let mut state = BeliefState::new(engine);
        assert!(!state.check().is_clean());
        let receipt = state
            .apply(ChangeSet::new("无关新增").assert("teaches", "王五", "赵六"))
            .expect("pre-existing duplicate must not block");
        assert!(!receipt
            .conflicts
            .iter()
            .any(|conflict| conflict.kind == ConflictKind::DuplicateRule));
    }

    #[test]
    fn constraints_compile_from_registry_and_law_book() {
        let mut registry = PredicateRegistry::new();
        registry.declare(&crate::predicate_registry::PredicateProfileConfig {
            name: "mentor".to_string(),
            functional: true,
            domain: vec!["Person".to_string()],
            range: vec!["Person".to_string()],
            ..Default::default()
        });
        let constraints = RevisionConstraints::from_registry(&registry);
        assert!(constraints.functional.contains("mentor"));
        assert_eq!(
            constraints.domain.get("mentor"),
            Some(&BTreeSet::from(["Person".to_string()]))
        );

        let book = LawBook::from_registry(&registry);
        let from_book = RevisionConstraints::from_law_book(&book);
        assert!(from_book.functional.contains("mentor"));
    }

    #[test]
    fn clipped_receipt_keeps_full_impact_counts() {
        let mut state = BeliefState::new(ResolutionEngine::new(8, 16));
        let mut changeset = ChangeSet::new("批量新增");
        for index in 0..100 {
            changeset.push(ChangeOp::AssertFact(Atom::rel(
                "p",
                "a",
                &format!("v{}", index),
            )));
        }
        let receipt = state.apply(changeset).expect("commit");
        assert_eq!(receipt.impact.counts.direct_added, 100);
        assert_eq!(receipt.impact.counts.newly_believed, 100);
        assert!(receipt.impact.direct_added.len() <= MAX_RECEIPT_ITEMS);
        assert_eq!(receipt.changed_beliefs(), 100);
    }

    #[test]
    fn empty_changeset_is_warned_but_commits() {
        let mut state = BeliefState::new(ResolutionEngine::new(8, 16));
        let receipt = state.apply(ChangeSet::new("空")).expect("commit");
        assert!(receipt
            .conflicts
            .iter()
            .any(|conflict| conflict.kind == ConflictKind::EmptyChangeSet));
        assert_eq!(state.revision(), 1);
    }

    #[test]
    fn missing_retraction_and_duplicate_assertion_are_warnings_in_non_strict_mode() {
        let mut engine = ResolutionEngine::new(8, 16);
        engine.add_fact("p", "a", "b");
        let state = BeliefState::new(engine).with_strict(false);
        let plan = state.plan(
            ChangeSet::new("无操作")
                .assert("p", "a", "b")
                .retract("p", "a", "c"),
        );
        assert!(!plan.is_blocked(), "{}", plan.render_zh());
        assert!(plan
            .conflicts
            .iter()
            .any(|conflict| conflict.kind == ConflictKind::DuplicateAssertion));
        assert!(plan
            .conflicts
            .iter()
            .any(|conflict| conflict.kind == ConflictKind::MissingRetraction));
    }
}

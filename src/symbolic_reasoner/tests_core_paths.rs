use crate::query_planner::{QueryPlan, QueryPlanner};
use crate::resolution_engine::{
    apply_atom, unify_atoms, Atom, Clause, ResolutionEngine, Substitution, Term,
};

fn planner() -> QueryPlanner {
    QueryPlanner::new()
}

#[test]
fn parse_find_object_chain() {
    let plan = planner().parse("张三的师傅的师傅是谁").expect("应解析出对象链");
    match plan {
        QueryPlan::FindObject { segments } => {
            assert!(segments.contains(&"张三".to_string()));
            assert!(segments.contains(&"师傅".to_string()));
        }
        other => panic!("期望 FindObject，实际 {:?}", other),
    }
}

#[test]
fn parse_count_question() {
    let plan = planner().parse("李四有几个师傅").expect("应解析出计数计划");
    assert!(matches!(plan, QueryPlan::Count { .. }), "实际 {:?}", plan);
    assert_eq!(QueryPlanner::kind(&plan), "Count");
}

#[test]
fn parse_exist_question() {
    let plan = planner().parse("李四有师傅吗").expect("应解析出存在计划");
    assert!(matches!(plan, QueryPlan::Exist { .. }), "实际 {:?}", plan);
}

#[test]
fn parse_find_subject_question() {
    let plan = planner().parse("谁的师傅是李四").expect("应解析出反查主体计划");
    match plan {
        QueryPlan::FindSubject { object, predicate } => {
            assert_eq!(object, "李四");
            assert_eq!(predicate, "师傅");
        }
        other => panic!("期望 FindSubject，实际 {:?}", other),
    }
}

#[test]
fn parse_analogy_question() {
    let plan = planner().parse("张三对应李四，王五对应谁").expect("应解析出类比计划");
    assert!(matches!(plan, QueryPlan::Analogy { .. }), "实际 {:?}", plan);
}

#[test]
fn parse_rejects_empty_and_overlong_questions() {
    assert!(planner().parse("").is_none(), "空问句不应产生计划");
    let overlong = "张".repeat(120);
    assert!(planner().parse(&overlong).is_none(), "超长问句不应产生计划");
}

#[test]
fn format_plan_is_human_readable() {
    let plan = planner().parse("李四有几个师傅").expect("应解析出计划");
    let rendered = QueryPlanner::format_plan(&plan);
    assert!(rendered.contains("计数") || rendered.contains("几个"), "{}", rendered);
}

#[test]
fn unify_ground_atoms_succeeds() {
    let left = Atom::rel("师傅", "张三", "李四");
    let right = Atom::rel("师傅", "张三", "李四");
    let mut subst = Substitution::new();
    assert!(unify_atoms(&left, &right, &mut subst));
}

#[test]
fn unify_variable_binds_constant() {
    let pattern = Atom::pattern("师傅", Term::var("X"), Term::constant("李四"));
    let fact = Atom::rel("师傅", "张三", "李四");
    let mut subst = Substitution::new();
    assert!(unify_atoms(&pattern, &fact, &mut subst));
    assert_eq!(subst.get("X").map(Term::name), Some("张三"));
}

#[test]
fn unify_conflicting_constants_fails() {
    let left = Atom::rel("师傅", "张三", "李四");
    let right = Atom::rel("师傅", "张三", "王五");
    let mut subst = Substitution::new();
    assert!(!unify_atoms(&left, &right, &mut subst));
}

#[test]
fn unify_arity_mismatch_fails() {
    let left = Atom::unary("师傅", Term::constant("张三"));
    let right = Atom::rel("师傅", "张三", "李四");
    let mut subst = Substitution::new();
    assert!(!unify_atoms(&left, &right, &mut subst));
}

#[test]
fn unify_repeated_variable_requires_same_value() {
    let pattern = Atom::pattern("认识", Term::var("X"), Term::var("X"));
    let fact = Atom::rel("认识", "张三", "李四");
    let mut subst = Substitution::new();
    assert!(!unify_atoms(&pattern, &fact, &mut subst));
}

#[test]
fn apply_atom_grounds_bound_variables() {
    let pattern = Atom::pattern("师傅", Term::var("X"), Term::constant("李四"));
    let fact = Atom::rel("师傅", "张三", "李四");
    let mut subst = Substitution::new();
    assert!(unify_atoms(&pattern, &fact, &mut subst));
    let ground = apply_atom(&subst, &pattern);
    assert!(ground.is_ground());
    assert_eq!(ground.display(), "师傅(张三, 李四)");
}

#[test]
fn resolution_engine_solves_two_hop_rule() {
    let mut engine = ResolutionEngine::new(8, 16);
    engine.add_fact("师傅", "张三", "李四");
    engine.add_fact("师傅", "李四", "王五");
    let head = Atom::pattern("师祖", Term::var("X"), Term::var("Z"));
    let body = vec![
        Atom::pattern("师傅", Term::var("X"), Term::var("Y")),
        Atom::pattern("师傅", Term::var("Y"), Term::var("Z")),
    ];
    engine.add_clause(Clause::new(head, body, 1.0, "test"));
    let goal = Atom::pattern("师祖", Term::constant("张三"), Term::var("Z"));
    let solutions = engine.solve(&[goal], &["Z".to_string()]);
    assert_eq!(solutions.len(), 1);
    assert_eq!(solutions[0].bindings.get("Z").map(String::as_str), Some("王五"));
}

#[test]
fn resolution_engine_reports_complete_absence() {
    let mut engine = ResolutionEngine::new(4, 8);
    engine.add_fact("师傅", "张三", "李四");
    let goal = Atom::pattern("师傅", Term::constant("王五"), Term::var("X"));
    let outcome = engine.solve_outcome(&[goal], &["X".to_string()]);
    assert!(outcome.solutions.is_empty());
    assert!(outcome.is_complete(), "预算内无解应报告为穷尽");
}

#[test]
fn resolution_engine_marks_solution_limit_truncation() {
    let mut engine = ResolutionEngine::new(8, 1);
    engine.add_fact("师傅", "张三", "李四");
    engine.add_fact("师傅", "张三", "王五");
    let goal = Atom::pattern("师傅", Term::constant("张三"), Term::var("X"));
    let outcome = engine.solve_outcome(&[goal], &["X".to_string()]);
    assert_eq!(outcome.solutions.len(), 1, "解数量应被上限截断");
    assert!(outcome.is_truncated(), "达到解上限应标记为截断");
}

#[test]
fn forward_chain_derives_new_fact() {
    let mut engine = ResolutionEngine::new(8, 16);
    engine.add_fact("师傅", "张三", "李四");
    engine.add_fact("师傅", "李四", "王五");
    let head = Atom::pattern("师祖", Term::var("X"), Term::var("Z"));
    let body = vec![
        Atom::pattern("师傅", Term::var("X"), Term::var("Y")),
        Atom::pattern("师傅", Term::var("Y"), Term::var("Z")),
    ];
    engine.add_clause(Clause::new(head, body, 1.0, "test"));
    let derived = engine.forward_chain(4, 16);
    assert!(
        derived.iter().any(|fact| fact.atom.display() == "师祖(张三, 王五)"),
        "前向链应推出师祖(张三, 王五)：{:?}",
        derived.iter().map(|fact| fact.atom.display()).collect::<Vec<_>>()
    );
}

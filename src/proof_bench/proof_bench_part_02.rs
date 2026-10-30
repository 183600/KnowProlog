use crate::context_image::fnv1a_hex;
fn run_scenario(
    scenario: &Scenario,
    facts: &[Fact],
    answer_override: Option<AnswerExpectation<'_>>,
) -> ScenarioRun {
    let prolog_facts = prolog_facts(facts);
    let config = ReasoningConfig {
        enable_link_prediction: false,
        ..ReasoningConfig::default()
    };
    let mut reasoner = SymbolicReasoner::new(config, HashMap::new());
    reasoner.observe_facts(&prolog_facts);
    let mut manager = DialogueManager::new();
    let mut readings: Vec<TurnReading> = Vec::new();
    let probe_index = scenario.probe_index();

    for turn in scenario.turns {
        // 与产品对话路径一致：先解读，再走受控生成以推进焦点栈。
        let frame = manager.analyze(turn.input, &reasoner);
        let _reply = manager
            .plan_pure_dialogue(turn.input, &reasoner)
            .or_else(|| manager.plan_reply(turn.input, &reasoner));
        readings.push(TurnReading {
            input: turn.input.to_string(),
            normalized_question: frame.question,
            entity: frame.entity,
            attribute: frame.attribute,
        });
    }

    let reading = &readings[probe_index];
    let probe = scenario
        .turns
        .get(probe_index)
        .and_then(|turn| turn.probe.as_ref())
        .expect("探针轮必须有 probe");
    let outcome = evaluate_probe(probe, reading, &reasoner, scenario, facts, answer_override);
    ScenarioRun { outcome }
}
fn evaluate_probe(
    probe: &Probe,
    reading: &TurnReading,
    reasoner: &SymbolicReasoner,
    scenario: &Scenario,
    facts: &[Fact],
    answer_override: Option<AnswerExpectation<'_>>,
) -> ProbeOutcome {
    match probe {
        Probe::Answer {
            entity,
            attribute,
            value,
            obligations,
        } => evaluate_answer(
            *entity,
            *attribute,
            *value,
            obligations,
            reading,
            reasoner,
            answer_override,
        ),
        Probe::Unknown {
            entity,
            attribute,
            goals,
        } => evaluate_unknown(*entity, *attribute, goals, reading, reasoner, facts),
        Probe::UnknownEntity { entity } => evaluate_unknown_entity(entity, reasoner, scenario),
    }
}
fn evaluate_answer(
    entity: &str,
    attribute: &str,
    value: &str,
    obligations: &[Obligation],
    reading: &TurnReading,
    reasoner: &SymbolicReasoner,
    answer_override: Option<AnswerExpectation<'_>>,
) -> ProbeOutcome {
    let (value, obligations): (&str, Vec<String>) = match answer_override {
        Some(expectation) => (expectation.value, expectation.obligations.to_vec()),
        None => (
            value,
            obligations.iter().map(Obligation::canonical).collect(),
        ),
    };
    if reading.entity.as_deref() != Some(entity) || reading.attribute.as_deref() != Some(attribute)
    {
        return ProbeOutcome {
            verdict: Verdict::Failed,
            observed: reading.entity.clone(),
            detail: format!(
                "问句被读成实体 {:?} / 属性 {:?}，期望实体 {:?} / 属性 {:?}",
                reading.entity, reading.attribute, entity, attribute
            ),
            distance: None,
        };
    }
    let Some(normalized) = reading.normalized_question.as_deref() else {
        return ProbeOutcome {
            verdict: Verdict::Failed,
            observed: None,
            detail: "没有生成规范化问句，无法进入证明路径".to_string(),
            distance: None,
        };
    };
    let Some(view) = reasoner.resolution_proof_view(normalized) else {
        return ProbeOutcome {
            verdict: Verdict::Unchecked,
            observed: None,
            detail: format!("无法把「{}」形式化为归结目标", normalized),
            distance: None,
        };
    };
    if view.solutions.is_empty() {
        if view.search_status == SearchStatus::Incomplete || view.search_truncated {
            return ProbeOutcome {
                verdict: Verdict::Unchecked,
                observed: None,
                detail: format!("搜索被截断（{}），不能判错", view.goal_display),
                distance: None,
            };
        }
        return ProbeOutcome {
            verdict: Verdict::Failed,
            observed: None,
            detail: format!("知识库未能证明目标：{}", view.goal_display),
            distance: None,
        };
    }

    let mut first_observed: Option<String> = None;
    let mut closest_missing: Vec<String> = Vec::new();
    for solution in &view.solutions {
        let observed = answer_binding(&solution.bindings).map(str::to_string);
        if first_observed.is_none() {
            first_observed = observed.clone();
        }
        let proof_atoms = collect_proof_atoms(&solution.proofs);
        let missing: Vec<String> = obligations
            .iter()
            .filter(|obligation| !proof_atoms.contains(*obligation))
            .cloned()
            .collect();
        if observed.as_deref() == Some(value) && missing.is_empty() {
            return ProbeOutcome {
                verdict: Verdict::Passed,
                observed,
                detail: format!(
                    "证明树覆盖 {} 条义务：{}",
                    obligations.len(),
                    obligations.join("，")
                ),
                distance: None,
            };
        }
        if closest_missing.is_empty() {
            closest_missing = missing;
        }
    }

    let detail = if !closest_missing.is_empty() {
        format!(
            "答案绑定为 {:?}，但证明树缺少义务：{}",
            first_observed,
            closest_missing.join("，")
        )
    } else {
        format!("答案绑定为 {:?}，与期望「{}」不一致", first_observed, value)
    };
    ProbeOutcome {
        verdict: Verdict::Failed,
        observed: first_observed,
        detail,
        distance: None,
    }
}
fn evaluate_unknown(
    entity: &str,
    attribute: &str,
    goals: &[Goal],
    reading: &TurnReading,
    reasoner: &SymbolicReasoner,
    facts: &[Fact],
) -> ProbeOutcome {
    if reading.entity.as_deref() != Some(entity) || reading.attribute.as_deref() != Some(attribute)
    {
        return ProbeOutcome {
            verdict: Verdict::Failed,
            observed: reading.entity.clone(),
            detail: format!(
                "不可答问句被读成实体 {:?} / 属性 {:?}，期望实体 {:?} / 属性 {:?}",
                reading.entity, reading.attribute, entity, attribute
            ),
            distance: None,
        };
    }
    let Some(normalized) = reading.normalized_question.as_deref() else {
        return ProbeOutcome {
            verdict: Verdict::Failed,
            observed: None,
            detail: "没有生成规范化问句，无法核对不可答性".to_string(),
            distance: None,
        };
    };
    let Some(view) = reasoner.resolution_proof_view(normalized) else {
        return ProbeOutcome {
            verdict: Verdict::Unchecked,
            observed: None,
            detail: format!("无法把「{}」形式化为归结目标", normalized),
            distance: Some(compute_distance(facts, goals)),
        };
    };
    if !view.solutions.is_empty() {
        let observed = view
            .solutions
            .first()
            .and_then(|solution| answer_binding(&solution.bindings))
            .map(str::to_string);
        return ProbeOutcome {
            verdict: Verdict::Failed,
            observed,
            detail: format!(
                "知识库本应不可答，却证明了 {} 个解：{}",
                view.solutions.len(),
                view.goal_display
            ),
            distance: Some(compute_distance(facts, goals)),
        };
    }
    if view.search_status == SearchStatus::Incomplete || view.search_truncated {
        return ProbeOutcome {
            verdict: Verdict::Unchecked,
            observed: None,
            detail: format!("搜索被截断（{}），不能断言不可答", view.goal_display),
            distance: Some(compute_distance(facts, goals)),
        };
    }
    // 形式证明为空还不够：回答管线也不能把别的话题证据包装成确定答案。
    if let Some(answer) = reasoner.answer_certain(&reading.input) {
        return ProbeOutcome {
            verdict: Verdict::Failed,
            observed: Some(clip(&answer, 120)),
            detail: format!(
                "归结搜索已穷尽且无解，但回答管线仍给出确定答案（目标：{}）",
                view.goal_display
            ),
            distance: Some(compute_distance(facts, goals)),
        };
    }
    ProbeOutcome {
        verdict: Verdict::Passed,
        observed: None,
        detail: format!("搜索已穷尽且无解，且没有确定回答：{}", view.goal_display),
        distance: Some(compute_distance(facts, goals)),
    }
}
fn evaluate_unknown_entity(
    entity: &str,
    reasoner: &SymbolicReasoner,
    scenario: &Scenario,
) -> ProbeOutcome {
    let known = reasoner.known_entity(entity);
    if known {
        return ProbeOutcome {
            verdict: Verdict::Failed,
            observed: Some(entity.to_string()),
            detail: format!("「{}」本应未知，却被登记为已知实体", entity),
            distance: None,
        };
    }
    let input = scenario
        .turns
        .get(scenario.probe_index())
        .map(|turn| turn.input)
        .unwrap_or("");
    if let Some(answer) = reasoner.answer_certain(input) {
        return ProbeOutcome {
            verdict: Verdict::Failed,
            observed: Some(clip(&answer, 80)),
            detail: format!("未知实体「{}」却给出了确定回答", entity),
            distance: None,
        };
    }
    ProbeOutcome {
        verdict: Verdict::Passed,
        observed: None,
        detail: format!("「{}」未被登记，且没有确定回答", entity),
        distance: None,
    }
}
fn evaluate_perturbation(
    scenario: &Scenario,
    base: &ProbeOutcome,
    mutated: &ProbeOutcome,
) -> PerturbationReport {
    let perturbation = &scenario.perturbation;
    let (passed, detail) = match perturbation.kind {
        PerturbationKind::Flip => {
            let flipped = base.verdict == Verdict::Passed
                && mutated.verdict == Verdict::Passed
                && base.observed.as_deref() == Some(perturbation.expected_before)
                && mutated.observed.as_deref() == Some(perturbation.expected_after)
                && base.observed != mutated.observed;
            let detail = if flipped {
                format!(
                    "答案按预期从「{}」翻转为「{}」",
                    perturbation.expected_before, perturbation.expected_after
                )
            } else {
                format!(
                    "期望「{}」→「{}」，实际 {:?}（{:?}）→ {:?}（{:?}）",
                    perturbation.expected_before,
                    perturbation.expected_after,
                    base.observed,
                    base.verdict,
                    mutated.observed,
                    mutated.verdict
                )
            };
            (flipped, detail)
        }
        PerturbationKind::Distractor => {
            let stable = base.verdict == Verdict::Passed
                && mutated.verdict == Verdict::Passed
                && base.observed == mutated.observed
                && perturbation.expected_before == perturbation.expected_after;
            let detail = if stable {
                format!("加入干扰后结论保持「{}」", perturbation.expected_after)
            } else {
                format!(
                    "期望结论不变（「{}」），实际 {:?}（{:?}）→ {:?}（{:?}）",
                    perturbation.expected_after,
                    base.observed,
                    base.verdict,
                    mutated.observed,
                    mutated.verdict
                )
            };
            (stable, detail)
        }
    };
    PerturbationReport {
        scenario: scenario.name.to_string(),
        label: perturbation.label.to_string(),
        kind: perturbation.kind,
        passed,
        expected_before: perturbation.expected_before.to_string(),
        expected_after: perturbation.expected_after.to_string(),
        base_observed: base.observed.clone(),
        mutated_observed: mutated.observed.clone(),
        detail,
    }
}
fn expected_text(probe: &Probe, perturbation_before: &str) -> Option<String> {
    match probe {
        Probe::Answer { value, .. } => Some((*value).to_string()),
        Probe::Unknown { .. } | Probe::UnknownEntity { .. } => {
            if perturbation_before == "unknown" {
                None
            } else {
                Some(perturbation_before.to_string())
            }
        }
    }
}
fn answer_binding(bindings: &[(String, String)]) -> Option<&str> {
    bindings
        .iter()
        .find(|(name, value)| name == "答案" && !value.is_empty())
        .map(|(_, value)| value.as_str())
        .or_else(|| {
            bindings
                .iter()
                .rev()
                .find(|(name, value)| !name.starts_with('V') && !value.is_empty())
                .map(|(_, value)| value.as_str())
        })
}
fn collect_proof_atoms(nodes: &[ProofViewNode]) -> BTreeSet<String> {
    let mut atoms = BTreeSet::new();
    fn walk(nodes: &[ProofViewNode], atoms: &mut BTreeSet<String>) {
        for node in nodes {
            atoms.insert(node.canonical.clone());
            walk(&node.children, atoms);
        }
    }
    walk(nodes, &mut atoms);
    atoms
}

// ---------------------------------------------------------------------------
// 证距
// ---------------------------------------------------------------------------
/// 有界溯因证距：目标可证则为 0；否则在预算内找最少假设联合场景。
pub fn compute_distance(facts: &[Fact], goals: &[Goal]) -> ProofDistance {
    let engine = resolver_for(facts);
    let atoms: Vec<Atom> = goals.iter().map(Goal::atom).collect();
    let mut query_vars: Vec<String> = Vec::new();
    for goal in goals {
        for term in [goal.subject, goal.object] {
            if let TermSpec::Var(name) = term {
                if !query_vars.iter().any(|var| var == name) {
                    query_vars.push(name.to_string());
                }
            }
        }
    }
    let goal_text = if atoms.is_empty() {
        "（空目标）".to_string()
    } else {
        atoms
            .iter()
            .map(Atom::display)
            .collect::<Vec<_>>()
            .join(" & ")
    };
    if !engine.solve(&atoms, &query_vars).is_empty() {
        return ProofDistance {
            goal: goal_text,
            minimum_hypotheses: Some(0),
            hypotheses: Vec::new(),
            verified: true,
            bounded: true,
        };
    }
    let mut abduction = AbductionEngine::new(4, 8);
    abduction.observe_facts(&prolog_facts(facts));
    let scenarios = abduction.scenarios(&engine, &atoms, &query_vars);
    let best_verified = scenarios
        .iter()
        .filter(|scenario| scenario.verified)
        .min_by_key(|scenario| scenario.size());
    if let Some(scenario) = best_verified {
        return ProofDistance {
            goal: goal_text,
            minimum_hypotheses: Some(scenario.size()),
            hypotheses: hypothesis_atoms(scenario),
            verified: true,
            bounded: true,
        };
    }
    // 没有通过注入验证的组合：仍把失败前沿上的最近候选写出来，
    // 但 minimum_hypotheses 保持 None，绝不把候选方向冒充成可证上界。
    match scenarios.iter().min_by_key(|scenario| scenario.size()) {
        Some(scenario) => ProofDistance {
            goal: goal_text,
            minimum_hypotheses: None,
            hypotheses: hypothesis_atoms(scenario),
            verified: false,
            bounded: true,
        },
        None => ProofDistance {
            goal: goal_text,
            minimum_hypotheses: None,
            hypotheses: Vec::new(),
            verified: false,
            bounded: true,
        },
    }
}
fn hypothesis_atoms(scenario: &crate::abduction::Scenario) -> Vec<String> {
    let mut hypotheses: Vec<String> = scenario.atoms().iter().map(Atom::display).collect();
    hypotheses.sort();
    hypotheses
}
fn apply_perturbation(facts: &[Fact], perturbation: &Perturbation) -> Vec<Fact> {
    let mut result: Vec<Fact> = facts.to_vec();
    for remove in perturbation.remove {
        result.retain(|fact| fact != remove);
    }
    result.extend_from_slice(perturbation.add);
    result
}

// ---------------------------------------------------------------------------
// 汇总 / 摘要 / 导出
// ---------------------------------------------------------------------------
fn ratio(numerator: u32, denominator: u32) -> f64 {
    if denominator == 0 {
        1.0
    } else {
        numerator as f64 / denominator as f64
    }
}
fn flatten(
    probes: u32,
    passed: u32,
    failed: u32,
    unchecked: u32,
    families: &BTreeMap<String, FamilyReport>,
    perturbations: &[PerturbationReport],
    results: &[ProbeReport],
) -> BTreeMap<String, f64> {
    let mut metrics = BTreeMap::new();
    let checked = passed + failed;
    metrics.insert("overall.pass_rate".to_string(), ratio(passed, checked));
    metrics.insert("overall.coverage".to_string(), ratio(checked, probes));
    metrics.insert("overall.passed".to_string(), passed as f64);
    metrics.insert("overall.failed".to_string(), failed as f64);
    metrics.insert("overall.unchecked".to_string(), unchecked as f64);
    for family in Family::ALL {
        if let Some(report) = families.get(family.label()) {
            metrics.insert(
                format!("family.{}.pass_rate", family.label()),
                report.pass_rate,
            );
            metrics.insert(
                format!("family.{}.coverage", family.label()),
                report.coverage,
            );
        }
    }
    let flips: Vec<&PerturbationReport> = perturbations
        .iter()
        .filter(|p| p.kind == PerturbationKind::Flip)
        .collect();
    let flip_passed = flips.iter().filter(|p| p.passed).count() as u32;
    let distractors: Vec<&PerturbationReport> = perturbations
        .iter()
        .filter(|p| p.kind == PerturbationKind::Distractor)
        .collect();
    let distractor_passed = distractors.iter().filter(|p| p.passed).count() as u32;
    metrics.insert(
        "twin.flip_rate".to_string(),
        ratio(flip_passed, flips.len() as u32),
    );
    metrics.insert(
        "twin.distractor_stability".to_string(),
        ratio(distractor_passed, distractors.len() as u32),
    );
    let perturb_passed = perturbations.iter().filter(|p| p.passed).count() as u32;
    metrics.insert(
        "perturbation.pass_rate".to_string(),
        ratio(perturb_passed, perturbations.len() as u32),
    );
    let distances: Vec<&ProofDistance> = results
        .iter()
        .filter_map(|result| result.distance.as_ref())
        .collect();
    let verified_distances: Vec<usize> = distances
        .iter()
        .filter_map(|distance| distance.minimum_hypotheses)
        .collect();
    if !verified_distances.is_empty() {
        metrics.insert(
            "proof_distance.mean".to_string(),
            verified_distances.iter().sum::<usize>() as f64 / verified_distances.len() as f64,
        );
        metrics.insert(
            "proof_distance.observed".to_string(),
            verified_distances.len() as f64,
        );
    }
    metrics.insert("proof_distance.probes".to_string(), distances.len() as f64);
    metrics.insert(
        "proof_distance.unverified".to_string(),
        distances.len().saturating_sub(verified_distances.len()) as f64,
    );
    metrics
}
/// 越大越差的指标（棘轮反向判定）。
pub const LOWER_IS_BETTER: [&str; 3] = [
    "proof_distance.mean",
    "proof_distance.unverified",
    "overall.unchecked",
];
/// 可写入磁盘的基线。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProofBenchBaseline {
    pub version: u32,
    pub corpus_digest: String,
    pub metrics: BTreeMap<String, f64>,
}
pub const PROOF_BENCH_BASELINE_VERSION: u32 = 1;
impl ProofBenchBaseline {
    pub fn from_report(report: &ProofBenchReport) -> Self {
        Self {
            version: PROOF_BENCH_BASELINE_VERSION,
            corpus_digest: report.corpus_digest.clone(),
            metrics: report.flatten_metrics(),
        }
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), std::io::Error> {
        let mut text = serde_json::to_string_pretty(self)?;
        text.push('\n');
        std::fs::write(path, text)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let text = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&text)?)
    }
}
/// 单项指标变化。
#[derive(Debug, Clone, PartialEq)]
pub struct MetricDelta {
    pub metric: String,
    pub baseline: f64,
    pub current: f64,
}
/// 棘轮判定。
#[derive(Debug, Clone, PartialEq)]
pub struct RatchetVerdict {
    pub passed: bool,
    pub digest_matches: bool,
    pub regressions: Vec<MetricDelta>,
    pub gains: Vec<MetricDelta>,
}
impl RatchetVerdict {
    pub fn render_text(&self) -> String {
        let mut out = String::new();
        if self.passed {
            out.push_str("ProofBench 棘轮检查通过：无指标低于基线。\n");
        } else {
            out.push_str("ProofBench 棘轮检查未通过：\n");
            if !self.digest_matches {
                out.push_str("  - 语料摘要不匹配：基准已漂移，基线不可直接比较\n");
            }
            for regression in &self.regressions {
                let direction = if LOWER_IS_BETTER.contains(&regression.metric.as_str()) {
                    "（越低越好）"
                } else {
                    ""
                };
                out.push_str(&format!(
                    "  - {}: {:.6} → {:.6}{}\n",
                    regression.metric, regression.baseline, regression.current, direction
                ));
            }
        }
        if !self.gains.is_empty() {
            out.push_str(&format!("有 {} 项指标高于基线。\n", self.gains.len()));
        }
        out
    }
}
/// 与基线比较：语料摘要不同或任何指标退化超 ε 即不通过。
pub fn compare_baseline(
    baseline: &ProofBenchBaseline,
    current: &ProofBenchReport,
    epsilon: f64,
) -> RatchetVerdict {
    let digest_matches = baseline.corpus_digest == current.corpus_digest;
    let current_metrics = current.flatten_metrics();
    let mut regressions = Vec::new();
    let mut gains = Vec::new();
    for (metric, baseline_value) in &baseline.metrics {
        let current_value = current_metrics.get(metric).copied().unwrap_or(0.0);
        let lower_is_better = LOWER_IS_BETTER.contains(&metric.as_str());
        let (is_regression, is_gain) = if lower_is_better {
            (
                current_value > baseline_value + epsilon,
                current_value < baseline_value - epsilon,
            )
        } else {
            (
                current_value < baseline_value - epsilon,
                current_value > baseline_value + epsilon,
            )
        };
        if is_regression {
            regressions.push(MetricDelta {
                metric: metric.clone(),
                baseline: *baseline_value,
                current: current_value,
            });
        } else if is_gain {
            gains.push(MetricDelta {
                metric: metric.clone(),
                baseline: *baseline_value,
                current: current_value,
            });
        }
    }
    RatchetVerdict {
        passed: digest_matches && regressions.is_empty(),
        digest_matches,
        regressions,
        gains,
    }
}
/// 确定性语料摘要（FNV-1a 64）：知识、问句或义务变化必须让基线失效。
pub fn corpus_digest(scenarios: &[Scenario]) -> String {
    let mut canonical = String::new();
    for scenario in scenarios {
        canonical.push_str("scenario=");
        canonical.push_str(scenario.name);
        canonical.push('|');
        canonical.push_str(scenario.family.label());
        canonical.push('\n');
        for fact in scenario.facts {
            canonical.push_str("fact=");
            match fact {
                Fact::Attribute { entity, key, value } => {
                    canonical.push_str(&format!("attr:{entity}|{key}|{value}"));
                }
                Fact::Relation { from, relation, to } => {
                    canonical.push_str(&format!("rel:{from}|{relation}|{to}"));
                }
            }
            canonical.push('\n');
        }
        for (index, turn) in scenario.turns.iter().enumerate() {
            canonical.push_str(&format!("turn={index}|{}|", turn.input));
            match &turn.probe {
                None => canonical.push_str("context"),
                Some(Probe::Answer {
                    entity,
                    attribute,
                    value,
                    obligations,
                }) => {
                    canonical.push_str(&format!("answer:{entity}|{attribute}|{value}|"));
                    for obligation in *obligations {
                        canonical.push_str(&obligation.canonical());
                        canonical.push(';');
                    }
                }
                Some(Probe::Unknown {
                    entity,
                    attribute,
                    goals,
                }) => {
                    canonical.push_str(&format!("unknown:{entity}|{attribute}|"));
                    for goal in *goals {
                        canonical.push_str(&format!(
                            "{}({},{})",
                            goal.predicate,
                            goal.subject.name(),
                            goal.object.name()
                        ));
                        canonical.push(';');
                    }
                }
                Some(Probe::UnknownEntity { entity }) => {
                    canonical.push_str(&format!("unknown_entity:{entity}"));
                }
            }
            canonical.push('\n');
        }
        let perturbation = &scenario.perturbation;
        canonical.push_str(&format!(
            "perturb={}|{}|{}|{}|{}|{}\n",
            perturbation.label,
            perturbation.kind.label(),
            perturbation.turn,
            perturbation.expected_before,
            perturbation.expected_after,
            scenario.facts.len()
        ));
        for fact in perturbation.remove {
            canonical.push_str(&format!("remove={fact:?}\n"));
        }
        for fact in perturbation.add {
            canonical.push_str(&format!("add={fact:?}\n"));
        }
    }
    fnv1a_hex(canonical.as_bytes())
}
/// 导出为纯 Prolog 事实，供第三方 `consult` 后独立审计。
///
/// 只导出数据与义务，不导出任何评分逻辑：评分口径写在 [`ProofBenchReport`]。
pub fn render_prolog(scenarios: &[Scenario]) -> String {
    let mut out = String::new();
    out.push_str("% ProofBench 语料导出：可 consult 的纯事实（只读审计）。\n");
    out.push_str(&format!("% corpus_digest({}).\n", corpus_digest(scenarios)));
    for scenario in scenarios {
        out.push_str(&format!(
            "\n% ---- {} ({}) ----\n",
            scenario.name,
            scenario.family.label()
        ));
        out.push_str(&format!(
            "bench_scenario('{}', '{}').\n",
            escape_atom(scenario.name),
            scenario.family.label()
        ));
        for fact in scenario.facts {
            match fact {
                Fact::Attribute { entity, key, value } => out.push_str(&format!(
                    "bench_fact('{}', '{}', '{}').\n",
                    escape_atom(entity),
                    escape_atom(key),
                    escape_atom(value)
                )),
                Fact::Relation { from, relation, to } => out.push_str(&format!(
                    "bench_fact('{}', '{}', '{}').\n",
                    escape_atom(from),
                    escape_atom(relation),
                    escape_atom(to)
                )),
            }
        }
        for (index, turn) in scenario.turns.iter().enumerate() {
            let kind = match &turn.probe {
                None => "context",
                Some(Probe::Answer { .. }) => "answer",
                Some(Probe::Unknown { .. }) => "unknown",
                Some(Probe::UnknownEntity { .. }) => "unknown_entity",
            };
            out.push_str(&format!(
                "bench_turn('{}', {}, '{}', '{}').\n",
                escape_atom(scenario.name),
                index,
                escape_atom(turn.input),
                kind
            ));
            match &turn.probe {
                Some(Probe::Answer {
                    value, obligations, ..
                }) => {
                    out.push_str(&format!(
                        "bench_answer('{}', {}, '{}').\n",
                        escape_atom(scenario.name),
                        index,
                        escape_atom(value)
                    ));
                    for obligation in *obligations {
                        out.push_str(&format!(
                            "bench_obligation('{}', {}, '{}').\n",
                            escape_atom(scenario.name),
                            index,
                            escape_atom(&obligation.canonical())
                        ));
                    }
                }
                Some(Probe::Unknown { goals, .. }) => {
                    let goal = goals
                        .iter()
                        .map(|goal| {
                            format!(
                                "{}({},{})",
                                goal.predicate,
                                goal.subject.name(),
                                goal.object.name()
                            )
                        })
                        .collect::<Vec<_>>()
                        .join(" & ");
                    out.push_str(&format!(
                        "bench_query('{}', {}, '{}').\n",
                        escape_atom(scenario.name),
                        index,
                        escape_atom(&goal)
                    ));
                }
                Some(Probe::UnknownEntity { entity }) => {
                    out.push_str(&format!(
                        "bench_unknown_entity('{}', {}, '{}').\n",
                        escape_atom(scenario.name),
                        index,
                        escape_atom(entity)
                    ));
                }
                None => {}
            }
        }
        out.push_str(&format!(
            "bench_perturbation('{}', '{}', '{}', '{}', '{}').\n",
            escape_atom(scenario.name),
            escape_atom(scenario.perturbation.label),
            scenario.perturbation.kind.label(),
            escape_atom(scenario.perturbation.expected_before),
            escape_atom(scenario.perturbation.expected_after)
        ));
    }
    out
}
fn escape_atom(text: &str) -> String {
    text.replace('\'', "''")
}
fn clip(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_string();
    }
    let mut clipped: String = text.chars().take(limit).collect();
    clipped.push('…');
    clipped
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corpus_shape_is_stable() {
        let scenarios = seed_corpus();
        assert_eq!(scenarios.len(), 12);
        for family in Family::ALL {
            assert_eq!(
                scenarios
                    .iter()
                    .filter(|scenario| scenario.family == family)
                    .count(),
                3,
                "每个家族应恰好 3 个场景"
            );
        }
        for scenario in scenarios {
            let probes = scenario
                .turns
                .iter()
                .filter(|turn| turn.probe.is_some())
                .count();
            assert_eq!(probes, 1, "{} 应恰好一个探针", scenario.name);
            assert_eq!(
                scenario.perturbation.turn,
                scenario.probe_index(),
                "{} 的扰动必须指向探针轮",
                scenario.name
            );
        }
        let flips = scenarios
            .iter()
            .filter(|scenario| scenario.perturbation.kind == PerturbationKind::Flip)
            .count();
        let distractors = scenarios.len() - flips;
        assert_eq!(flips, 8);
        assert_eq!(distractors, 4);
    }

    #[test]
    fn corpus_answer_obligations_are_provable() {
        for scenario in seed_corpus() {
            let probe = scenario.turns[scenario.probe_index()]
                .probe
                .as_ref()
                .unwrap();
            if let Probe::Answer { obligations, .. } = probe {
                let engine = resolver_for(scenario.facts);
                for obligation in *obligations {
                    let atom =
                        Atom::rel(obligation.predicate, obligation.subject, obligation.object);
                    assert_eq!(
                        engine.solve(&[atom], &[]).len(),
                        1,
                        "{} 的义务 {} 必须由基础知识库证明",
                        scenario.name,
                        obligation.canonical()
                    );
                }
            }
        }
    }

    #[test]
    fn corpus_unknown_goals_are_genuinely_unprovable() {
        for scenario in seed_corpus() {
            let probe = scenario.turns[scenario.probe_index()]
                .probe
                .as_ref()
                .unwrap();
            if let Probe::Unknown { goals, .. } = probe {
                let engine = resolver_for(scenario.facts);
                let atoms: Vec<Atom> = goals.iter().map(Goal::atom).collect();
                assert!(
                    engine.solve(&atoms, &[]).is_empty(),
                    "{} 的不可答目标不应被基础知识库证明",
                    scenario.name
                );
            }
        }
    }

    #[test]
    fn report_is_deterministic_and_metrics_bounded() {
        let first = evaluate_seed();
        let second = evaluate_seed();
        assert_eq!(first, second, "同一语料必须给出逐字节可复现的报告");
        assert_eq!(first.scenarios, 12);
        assert_eq!(first.probes, 12);
        assert_eq!(first.perturbations.len(), 12);
        assert!(first.passed + first.failed + first.unchecked == first.probes);
        for (metric, value) in &first.metrics {
            if metric.ends_with(".observed")
                || metric.ends_with(".passed")
                || metric.ends_with(".failed")
                || metric.ends_with(".unchecked")
                || metric.ends_with(".probes")
                || metric.ends_with(".unverified")
            {
                continue;
            }
            assert!(
                (0.0..=1.0).contains(value),
                "指标 {} 越界：{}",
                metric,
                value
            );
        }
        assert!(!first.corpus_digest.is_empty());
    }

    #[test]
    fn seed_corpus_exercises_all_mechanisms() {
        let report = evaluate_seed();
        assert!(
            report.families.values().all(|family| family.coverage > 0.0),
            "每个家族都必须至少有一个可检查探针：{}",
            report.render_text()
        );
        let flips = report
            .perturbations
            .iter()
            .filter(|p| p.kind == PerturbationKind::Flip)
            .count();
        let distractors = report
            .perturbations
            .iter()
            .filter(|p| p.kind == PerturbationKind::Distractor)
            .count();
        assert_eq!(flips, 8);
        assert_eq!(distractors, 4);
        let distances = report
            .results
            .iter()
            .filter_map(|result| result.distance.as_ref())
            .collect::<Vec<_>>();
        assert!(!distances.is_empty(), "不可答探针必须报告证距");
        assert!(distances.iter().all(|distance| distance.bounded));
        assert_eq!(distances.len(), 2, "两个 Probe::Unknown 都应给出证距对象");
        assert!(
            distances
                .iter()
                .any(|distance| distance.minimum_hypotheses.is_some()),
            "至少应有一个预算内可验证的证距估计"
        );
    }

    #[test]
    fn unknown_bridge_rejects_off_topic_evidence() {
        let report = evaluate_seed();
        let result = report
            .results
            .iter()
            .find(|result| result.scenario == "未知桥接")
            .expect("语料必须包含未知桥接场景");
        assert_eq!(
            result.verdict,
            Verdict::Passed,
            "「配偶」的话题证据不能冒充「配偶的体重」的确定答案：{}",
            result.detail
        );
        let distance = result.distance.as_ref().expect("应报告证距");
        assert!(distance.verified, "{:?}", distance);
    }

    #[test]
    fn baseline_ratchet_detects_regressions_and_drift() {
        let report = evaluate_seed();
        let baseline = ProofBenchBaseline::from_report(&report);
        let verdict = compare_baseline(&baseline, &report, 0.0);
        assert!(verdict.passed, "{}", verdict.render_text());
        assert!(verdict.digest_matches);

        let mut degraded = report.clone();
        degraded
            .metrics
            .insert("overall.pass_rate".to_string(), 0.0);
        let verdict = compare_baseline(&baseline, &degraded, 0.0);
        assert!(!verdict.passed);
        assert!(verdict
            .regressions
            .iter()
            .any(|delta| delta.metric == "overall.pass_rate"));

        let mut drifted = report.clone();
        drifted.corpus_digest = "0000000000000000".to_string();
        let verdict = compare_baseline(&baseline, &drifted, 0.0);
        assert!(!verdict.passed && !verdict.digest_matches);
    }

    #[test]
    fn baseline_roundtrips_through_disk() {
        let report = evaluate_seed();
        let baseline = ProofBenchBaseline::from_report(&report);
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("proof_bench_baseline.json");
        baseline.save(&path).unwrap();
        let loaded = ProofBenchBaseline::load(&path).unwrap();
        assert_eq!(loaded.version, baseline.version);
        assert_eq!(loaded.corpus_digest, baseline.corpus_digest);
        assert_eq!(loaded.metrics, baseline.metrics);
    }

    #[test]
    fn prolog_export_is_auditable() {
        let text = render_prolog(seed_corpus());
        assert!(text.contains("bench_scenario('篮球家庭', 'coref')."));
        assert!(text.contains("bench_fact('叶莉', '身高', '190')."));
        assert!(text.contains("bench_obligation('三跳师门', 0, '师傅(李四, 王五)')."));
        assert!(text.contains("bench_query('未知桥接'"));
        assert!(text.contains("bench_perturbation('"));
        assert!(text.contains("corpus_digest("));
        assert!(text
            .lines()
            .all(|line| line.is_empty() || line.starts_with('%') || line.ends_with('.')));
    }

    #[test]
    fn proof_distance_reports_a_bounded_gap() {
        let facts = [
            Fact::Attribute {
                entity: "姚明",
                key: "身高",
                value: "226",
            },
            Fact::Attribute {
                entity: "李娜",
                key: "体重",
                value: "55",
            },
        ];
        let goals = [Goal {
            predicate: "体重",
            subject: TermSpec::Entity("姚明"),
            object: TermSpec::Var("答案"),
        }];
        let distance = compute_distance(&facts, &goals);
        assert!(distance.bounded);
        assert!(distance.verified);
        assert_eq!(distance.minimum_hypotheses, Some(1), "{:?}", distance);
        assert!(
            distance
                .hypotheses
                .iter()
                .any(|atom| atom.starts_with("体重(姚明")),
            "{:?}",
            distance
        );
    }
}

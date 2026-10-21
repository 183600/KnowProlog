//! 形状填充对话评测台（SFD Eval）：缺口 × 获取 × 生成 × 会话四层指标。
//!
//! 与 `shape_gate` 的 `eval-shapes` 分工：后者只验证图谱是否合规；本模块
//! 验证「形状缺口是否变成值得问的问题、回答是否真的收敛、补全是否回到原问题、
//! 未知与冲突是否被诚实披露、快照恢复后议程是否还在、新会话是否泄漏旧补充」。

use std::collections::BTreeSet;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::context_image::{canonical_json, fnv1a_hex};
use crate::predicate_registry::{PredicateProfileConfig, PredicateRegistry};
use crate::prolog_engine::PrologFact;
use crate::relation_acquisition::{AcquisitionStatus, RelationAcquisition, SessionOverlay};
use crate::shape_dialogue::{
    AnswerStatus, Consultation, FactDirection, ShapeAnswer, ShapeDialogue,
};
use crate::shape_gate::ShapeSet;
use crate::type_system::{TypeAliasConfig, TypeDisjointConfig, TypeSystem, TypeSystemConfig};
use crate::{EntityTypeConfig, PropertyConfig};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShapeEvalReport {
    pub digest: String,
    pub gap_questions: usize,
    pub targeted_gap_questions: usize,
    pub repeat_questions: usize,
    pub claims_proposed: usize,
    pub claims_confirmed: usize,
    pub wrong_accepts: usize,
    pub answer_claims: usize,
    pub grounded_claims: usize,
    pub required_slots_before: usize,
    pub required_slots_after: usize,
    pub conflicts: usize,
    pub conflicts_disclosed: usize,
    pub unknown_cases: usize,
    pub unknown_abstained: usize,
    pub resumed_original: bool,
    pub snapshot_restored: bool,
    pub session_isolation: bool,
    pub cross_session_leaks: usize,
}

impl ShapeEvalReport {
    pub fn gap_precision(&self) -> f64 {
        ratio(self.targeted_gap_questions, self.gap_questions)
    }

    pub fn repeat_rate(&self) -> f64 {
        ratio(self.repeat_questions, self.gap_questions)
    }

    pub fn claim_precision(&self) -> f64 {
        ratio(
            self.claims_confirmed,
            self.claims_confirmed + self.wrong_accepts,
        )
    }

    pub fn grounded_rate(&self) -> f64 {
        ratio(self.grounded_claims, self.answer_claims)
    }

    pub fn required_coverage_after(&self) -> f64 {
        if self.required_slots_before == 0 {
            return 1.0;
        }
        ratio(
            self.required_slots_before
                .saturating_sub(self.required_slots_after),
            self.required_slots_before,
        )
    }

    pub fn conflict_disclosure_rate(&self) -> f64 {
        ratio(self.conflicts_disclosed, self.conflicts)
    }

    pub fn abstain_rate(&self) -> f64 {
        ratio(self.unknown_abstained, self.unknown_cases)
    }

    pub fn session_check_rate(&self) -> f64 {
        let checks = [
            self.resumed_original,
            self.snapshot_restored,
            self.session_isolation,
        ];
        ratio(checks.iter().filter(|check| **check).count(), checks.len())
    }

    pub fn passed(&self) -> bool {
        self.resumed_original
            && self.snapshot_restored
            && self.session_isolation
            && self.wrong_accepts == 0
            && self.repeat_questions == 0
            && self.cross_session_leaks == 0
            && self.gap_precision() >= 1.0
            && self.claim_precision() >= 1.0
            && self.grounded_rate() >= 1.0
            && self.required_coverage_after() >= 1.0
            && self.conflict_disclosure_rate() >= 1.0
            && self.abstain_rate() >= 1.0
    }

    pub fn render_text(&self) -> String {
        let mut lines = vec![
            "形状填充对话评测（SFD eval）".to_string(),
            format!(
                "缺口层：问缺口 {}/{}（精度 {:.3}）；重复问 {}（{:.3}）",
                self.targeted_gap_questions,
                self.gap_questions,
                self.gap_precision(),
                self.repeat_questions,
                self.repeat_rate()
            ),
            format!(
                "获取层：候选 {}；确认 {}；错误接受 {}（精度 {:.3}）",
                self.claims_proposed,
                self.claims_confirmed,
                self.wrong_accepts,
                self.claim_precision()
            ),
            format!(
                "生成层：陈述 {}/{} 有来源（{:.3}）；必备槽覆盖 {:.3}；冲突披露 {}/{}；未知克制 {}/{}",
                self.grounded_claims,
                self.answer_claims,
                self.grounded_rate(),
                self.required_coverage_after(),
                self.conflicts_disclosed,
                self.conflicts,
                self.unknown_abstained,
                self.unknown_cases
            ),
            format!(
                "会话层：回到原问题 {}；快照恢复 {}；会话隔离 {}；跨会话泄漏 {}（检查通过率 {:.3}）",
                yes_no(self.resumed_original),
                yes_no(self.snapshot_restored),
                yes_no(self.session_isolation),
                self.cross_session_leaks,
                self.session_check_rate()
            ),
            format!("报告摘要：{}", self.digest),
        ];
        lines.insert(
            1,
            format!("判定：{}", if self.passed() { "通过" } else { "未通过" }),
        );
        lines.join("\n")
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShapeEvalBaseline {
    pub digest: String,
    pub gap_precision: f64,
    pub repeat_rate: f64,
    pub claim_precision: f64,
    pub grounded_rate: f64,
    pub required_coverage_after: f64,
    pub conflict_disclosure_rate: f64,
    pub abstain_rate: f64,
    pub session_check_rate: f64,
}

impl ShapeEvalBaseline {
    pub fn from_report(report: &ShapeEvalReport) -> Self {
        Self {
            digest: report.digest.clone(),
            gap_precision: report.gap_precision(),
            repeat_rate: report.repeat_rate(),
            claim_precision: report.claim_precision(),
            grounded_rate: report.grounded_rate(),
            required_coverage_after: report.required_coverage_after(),
            conflict_disclosure_rate: report.conflict_disclosure_rate(),
            abstain_rate: report.abstain_rate(),
            session_check_rate: report.session_check_rate(),
        }
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), std::io::Error> {
        std::fs::write(path, serde_json::to_string_pretty(self).unwrap_or_default())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let text = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&text)?)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShapeEvalDelta {
    pub metric: String,
    pub before: f64,
    pub after: f64,
    pub direction: String,
    pub passed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShapeEvalVerdict {
    pub passed: bool,
    pub digest_matches: bool,
    pub deltas: Vec<ShapeEvalDelta>,
}

impl ShapeEvalVerdict {
    pub fn render_text(&self) -> String {
        let mut lines = vec![format!(
            "eval-shape-dialogue 棘轮：{}（评测摘要{}）。",
            if self.passed { "通过" } else { "未通过" },
            if self.digest_matches {
                "一致"
            } else {
                "已变化：指标按新语料比较"
            }
        )];
        for delta in &self.deltas {
            lines.push(format!(
                "  {} {}：{:.6} → {:.6}（{}）",
                if delta.passed { "✓" } else { "✗" },
                delta.metric,
                delta.before,
                delta.after,
                if delta.direction == "higher_better" {
                    "越高越好"
                } else {
                    "越低越好"
                }
            ));
        }
        lines.join("\n")
    }
}

pub fn compare_baseline(
    baseline: &ShapeEvalBaseline,
    current: &ShapeEvalReport,
    epsilon: f64,
) -> ShapeEvalVerdict {
    let mut deltas = Vec::new();
    let mut push = |metric: &str, before: f64, after: f64, higher_better: bool| {
        let passed = if higher_better {
            after + epsilon >= before
        } else {
            after <= before + epsilon
        };
        deltas.push(ShapeEvalDelta {
            metric: metric.to_string(),
            before,
            after,
            direction: if higher_better {
                "higher_better".to_string()
            } else {
                "lower_better".to_string()
            },
            passed,
        });
    };
    push(
        "gap_precision",
        baseline.gap_precision,
        current.gap_precision(),
        true,
    );
    push(
        "repeat_rate",
        baseline.repeat_rate,
        current.repeat_rate(),
        false,
    );
    push(
        "claim_precision",
        baseline.claim_precision,
        current.claim_precision(),
        true,
    );
    push(
        "grounded_rate",
        baseline.grounded_rate,
        current.grounded_rate(),
        true,
    );
    push(
        "required_coverage_after",
        baseline.required_coverage_after,
        current.required_coverage_after(),
        true,
    );
    push(
        "conflict_disclosure_rate",
        baseline.conflict_disclosure_rate,
        current.conflict_disclosure_rate(),
        true,
    );
    push(
        "abstain_rate",
        baseline.abstain_rate,
        current.abstain_rate(),
        true,
    );
    push(
        "session_check_rate",
        baseline.session_check_rate,
        current.session_check_rate(),
        true,
    );
    let passed = deltas.iter().all(|delta| delta.passed);
    ShapeEvalVerdict {
        passed,
        digest_matches: baseline.digest == current.digest,
        deltas,
    }
}

#[derive(Debug, Default)]
struct EvalTally {
    gap_questions: usize,
    targeted_gap_questions: usize,
    questions: BTreeSet<(String, String, String)>,
    repeat_questions: usize,
    claims_proposed: usize,
    claims_confirmed: usize,
    wrong_accepts: usize,
    answer_claims: usize,
    grounded_claims: usize,
    required_slots_before: usize,
    required_slots_after: usize,
    conflicts: usize,
    conflicts_disclosed: usize,
    unknown_cases: usize,
    unknown_abstained: usize,
    resumed_original: bool,
    snapshot_restored: bool,
    session_isolation: bool,
    cross_session_leaks: usize,
}

impl EvalTally {
    fn record_consultation(
        &mut self,
        session: &str,
        entity: &str,
        consultation: &Consultation,
        answer: &ShapeAnswer,
    ) {
        self.gap_questions += 1;
        if answer
            .missing_predicates()
            .contains(&consultation.gap.predicate.as_str())
        {
            self.targeted_gap_questions += 1;
        }
        let key = (
            session.to_string(),
            entity.to_string(),
            consultation.gap.predicate.clone(),
        );
        if !self.questions.insert(key) {
            self.repeat_questions += 1;
        }
    }

    fn assume_wrong_accept_if(&mut self, condition: bool) {
        if condition {
            self.wrong_accepts += 1;
        }
    }
}

pub fn evaluate_seed() -> ShapeEvalReport {
    let mut tally = EvalTally::default();
    run_completion(&mut tally);
    run_two_gaps(&mut tally);
    run_conflict(&mut tally);
    run_type_issue(&mut tally);
    run_unknown(&mut tally);
    run_snapshot(&mut tally);
    run_isolation(&mut tally);

    let digest = {
        let value = serde_json::json!({
            "gap_questions": tally.gap_questions,
            "targeted": tally.targeted_gap_questions,
            "repeat": tally.repeat_questions,
            "proposed": tally.claims_proposed,
            "confirmed": tally.claims_confirmed,
            "wrong": tally.wrong_accepts,
            "answer_claims": tally.answer_claims,
            "grounded": tally.grounded_claims,
            "before": tally.required_slots_before,
            "after": tally.required_slots_after,
            "conflicts": tally.conflicts,
            "disclosed": tally.conflicts_disclosed,
            "unknown": tally.unknown_cases,
            "abstained": tally.unknown_abstained,
            "resumed": tally.resumed_original,
            "snapshot": tally.snapshot_restored,
            "isolation": tally.session_isolation,
            "leaks": tally.cross_session_leaks,
        });
        fnv1a_hex(canonical_json(&value).as_bytes())
    };
    ShapeEvalReport {
        digest,
        gap_questions: tally.gap_questions,
        targeted_gap_questions: tally.targeted_gap_questions,
        repeat_questions: tally.repeat_questions,
        claims_proposed: tally.claims_proposed,
        claims_confirmed: tally.claims_confirmed,
        wrong_accepts: tally.wrong_accepts,
        answer_claims: tally.answer_claims,
        grounded_claims: tally.grounded_claims,
        required_slots_before: tally.required_slots_before,
        required_slots_after: tally.required_slots_after,
        conflicts: tally.conflicts,
        conflicts_disclosed: tally.conflicts_disclosed,
        unknown_cases: tally.unknown_cases,
        unknown_abstained: tally.unknown_abstained,
        resumed_original: tally.resumed_original,
        snapshot_restored: tally.snapshot_restored,
        session_isolation: tally.session_isolation,
        cross_session_leaks: tally.cross_session_leaks,
    }
}

fn run_completion(tally: &mut EvalTally) {
    let registry = registry();
    let types = types();
    let shapes = base_shapes(&registry, &types);
    let facts = base_facts();
    let mut acquisition = RelationAcquisition::new();
    let mut dialogue = ShapeDialogue::new();
    let question = "刘德华是谁？";
    let first = dialogue.start_profile(
        "刘德华",
        question,
        &facts,
        &registry,
        &types,
        &shapes,
        &mut acquisition,
    );
    tally.required_slots_before += first.answer.missing.len();
    if let Some(consultation) = &first.consultation {
        tally.record_consultation("completion", "刘德华", consultation, &first.answer);
    }
    let proposed = dialogue.respond("香港", &facts, &registry, &types, &shapes, &mut acquisition);
    if let Some(turn) = &proposed {
        if acquisition
            .claims()
            .any(|claim| claim.status == AcquisitionStatus::Proposed)
        {
            tally.claims_proposed += 1;
        }
        tally.assume_wrong_accept_if(!acquisition.overlay().is_empty());
        tally.answer_claims += turn.answer.facts.len();
        tally.grounded_claims += grounded_facts(
            "刘德华",
            &turn.answer,
            &facts,
            &acquisition.overlay(),
            &registry,
        );
    }
    let confirmed = dialogue.respond("确认", &facts, &registry, &types, &shapes, &mut acquisition);
    if let Some(turn) = &confirmed {
        tally.required_slots_after += turn.answer.missing.len();
        tally.claims_confirmed += acquisition.overlay().rows.len();
        tally.answer_claims += turn.answer.facts.len();
        tally.grounded_claims += grounded_facts(
            "刘德华",
            &turn.answer,
            &facts,
            &acquisition.overlay(),
            &registry,
        );
        tally.resumed_original |=
            turn.message.contains(question) && turn.answer.status == AnswerStatus::Complete;
    }
}

fn run_two_gaps(tally: &mut EvalTally) {
    let registry = registry();
    let types = types();
    let shapes = ShapeSet::from_schema(
        &[EntityTypeConfig {
            label: "Person".to_string(),
            examples: Vec::new(),
            properties: vec![required_property("出生地"), required_property("国籍")],
        }],
        &registry,
        &types,
    );
    let facts = vec![
        entity("王五", "Person", &[("name", "王五")]),
        entity("赵六", "Person", &[("name", "赵六")]),
    ];
    let mut acquisition = RelationAcquisition::new();
    let mut dialogue = ShapeDialogue::new();
    let first = dialogue.start_profile(
        "王五",
        "王五是谁？",
        &facts,
        &registry,
        &types,
        &shapes,
        &mut acquisition,
    );
    tally.required_slots_before += first.answer.missing.len();
    let Some(first_consultation) = first.consultation.clone() else {
        return;
    };
    tally.record_consultation("two_gaps", "王五", &first_consultation, &first.answer);
    let first_value = if first_consultation.gap.predicate == "出生地" {
        "香港"
    } else {
        "中国"
    };
    let _ = dialogue.respond(
        first_value,
        &facts,
        &registry,
        &types,
        &shapes,
        &mut acquisition,
    );
    let confirmed = dialogue.respond("确认", &facts, &registry, &types, &shapes, &mut acquisition);
    let Some(second) = confirmed else {
        return;
    };
    if let Some(consultation) = &second.consultation {
        tally.record_consultation("two_gaps", "王五", consultation, &second.answer);
        let value = if consultation.gap.predicate == "出生地" {
            "香港"
        } else {
            "中国"
        };
        let _ = dialogue.respond(value, &facts, &registry, &types, &shapes, &mut acquisition);
        let last = dialogue.respond("确认", &facts, &registry, &types, &shapes, &mut acquisition);
        if let Some(last) = last {
            tally.required_slots_after += last.answer.missing.len();
            tally.claims_confirmed += acquisition
                .overlay()
                .rows
                .iter()
                .filter(|row| row.subject == "王五")
                .count();
            tally.answer_claims += last.answer.facts.len();
            tally.grounded_claims += grounded_facts(
                "王五",
                &last.answer,
                &facts,
                &acquisition.overlay(),
                &registry,
            );
        }
    } else {
        tally.required_slots_after += second.answer.missing.len();
    }
}

fn run_conflict(tally: &mut EvalTally) {
    let registry = registry();
    let types = types();
    let shapes = base_shapes(&registry, &types);
    let facts = vec![
        entity(
            "刘德华",
            "Person",
            &[("name", "刘德华"), ("出生地", "香港"), ("职业", "演员")],
        ),
        relation("刘德华", "职业", "导演"),
    ];
    let mut acquisition = RelationAcquisition::new();
    let mut dialogue = ShapeDialogue::new();
    let turn = dialogue.start_profile(
        "刘德华",
        "刘德华是谁？",
        &facts,
        &registry,
        &types,
        &shapes,
        &mut acquisition,
    );
    tally.conflicts += 1;
    let text = turn.message;
    if turn.answer.status == AnswerStatus::Conflict
        && text.contains("冲突")
        && text.contains("演员")
        && text.contains("导演")
        && text.contains("不替你选")
    {
        tally.conflicts_disclosed += 1;
    }
}

fn run_type_issue(tally: &mut EvalTally) {
    let registry = registry();
    let types = types();
    let shapes = ShapeSet::from_schema(
        &[EntityTypeConfig {
            label: "Person".to_string(),
            examples: Vec::new(),
            properties: vec![optional_property("年龄", "integer")],
        }],
        &registry,
        &types,
    );
    let facts = vec![entity(
        "刘德华",
        "Person",
        &[("name", "刘德华"), ("年龄", "十八")],
    )];
    let mut acquisition = RelationAcquisition::new();
    let mut dialogue = ShapeDialogue::new();
    let turn = dialogue.start_profile(
        "刘德华",
        "刘德华是谁？",
        &facts,
        &registry,
        &types,
        &shapes,
        &mut acquisition,
    );
    tally.conflicts += 1;
    if turn.answer.status == AnswerStatus::Conflict
        && !turn.answer.issues.is_empty()
        && turn.consultation.is_none()
        && turn.message.contains("形状问题")
        && turn.message.contains("不会把不合形状的取值当作确定事实")
        && dialogue.agenda().blocked.contains("刘德华")
    {
        tally.conflicts_disclosed += 1;
    }
}

fn run_unknown(tally: &mut EvalTally) {
    let registry = registry();
    let types = types();
    let shapes = base_shapes(&registry, &types);
    let facts = base_facts();
    let mut acquisition = RelationAcquisition::new();
    let mut dialogue = ShapeDialogue::new();
    let turn = dialogue.start_profile(
        "未名科技",
        "未名科技是谁？",
        &facts,
        &registry,
        &types,
        &shapes,
        &mut acquisition,
    );
    tally.unknown_cases += 1;
    if turn.answer.status == AnswerStatus::Unknown
        && turn.answer.facts.is_empty()
        && turn.consultation.is_none()
        && turn.message.contains("不会为它编造画像")
    {
        tally.unknown_abstained += 1;
    }
}

fn run_snapshot(tally: &mut EvalTally) {
    let registry = registry();
    let types = types();
    let shapes = base_shapes(&registry, &types);
    let facts = base_facts();
    let mut acquisition = RelationAcquisition::new();
    let mut dialogue = ShapeDialogue::new();
    let first = dialogue.start_profile(
        "王五",
        "王五是谁？",
        &facts,
        &registry,
        &types,
        &shapes,
        &mut acquisition,
    );
    tally.required_slots_before += first.answer.missing.len();
    if let Some(consultation) = &first.consultation {
        tally.record_consultation("snapshot", "王五", consultation, &first.answer);
    }
    let snapshot = dialogue.snapshot();
    let acquisition_snapshot = acquisition.snapshot();
    let mut restored = ShapeDialogue::new();
    let mut restored_acquisition = RelationAcquisition::new();
    restored.restore(&snapshot);
    restored_acquisition.restore(&acquisition_snapshot);
    let _ = restored.respond(
        "北京",
        &facts,
        &registry,
        &types,
        &shapes,
        &mut restored_acquisition,
    );
    let confirmed = restored.respond(
        "确认",
        &facts,
        &registry,
        &types,
        &shapes,
        &mut restored_acquisition,
    );
    if let Some(turn) = confirmed {
        tally.required_slots_after += turn.answer.missing.len();
        tally.claims_confirmed += restored_acquisition.overlay().rows.len();
        tally.answer_claims += turn.answer.facts.len();
        tally.grounded_claims += grounded_facts(
            "王五",
            &turn.answer,
            &facts,
            &restored_acquisition.overlay(),
            &registry,
        );
        tally.snapshot_restored =
            turn.answer.status == AnswerStatus::Complete && turn.message.contains("王五是谁？");
    }
}

fn run_isolation(tally: &mut EvalTally) {
    let registry = registry();
    let types = types();
    let shapes = base_shapes(&registry, &types);
    let facts = base_facts();
    let fresh = RelationAcquisition::new();
    let (answer, _) = ShapeAnswer::build(
        "刘德华",
        &facts,
        &fresh.overlay(),
        &shapes,
        &registry,
        &types,
    );
    tally.session_isolation =
        answer.status == AnswerStatus::Partial && answer.value_of("出生地").is_empty();
    if !answer.value_of("出生地").is_empty() {
        tally.cross_session_leaks += 1;
    }
}

fn grounded_facts(
    entity: &str,
    answer: &ShapeAnswer,
    facts: &[PrologFact],
    overlay: &SessionOverlay,
    registry: &PredicateRegistry,
) -> usize {
    answer
        .facts
        .iter()
        .filter(|fact| {
            fact.sources.iter().any(|source| match source {
                crate::shape_dialogue::FactSource::Graph => match fact.direction {
                    FactDirection::Attribute => facts.iter().any(|candidate| match candidate {
                        PrologFact::Entity {
                            name, properties, ..
                        } => {
                            name == entity
                                && properties.iter().any(|(key, value)| {
                                    registry.resolve(key).canonical == fact.predicate
                                        && value == &fact.object
                                })
                        }
                        _ => false,
                    }),
                    FactDirection::Forward => facts.iter().any(|candidate| match candidate {
                        PrologFact::Relation { from, rel_type, to } => {
                            let resolution = registry.resolve(rel_type);
                            let (subject, object) = if resolution.inverted {
                                (to.as_str(), from.as_str())
                            } else {
                                (from.as_str(), to.as_str())
                            };
                            subject == entity
                                && object == fact.object
                                && resolution.canonical == fact.predicate
                        }
                        _ => false,
                    }),
                    FactDirection::Reverse => facts.iter().any(|candidate| match candidate {
                        PrologFact::Relation { from, rel_type, to } => {
                            let resolution = registry.resolve(rel_type);
                            let (subject, object) = if resolution.inverted {
                                (to.as_str(), from.as_str())
                            } else {
                                (from.as_str(), to.as_str())
                            };
                            object == entity
                                && subject == fact.object
                                && resolution.canonical == fact.predicate
                        }
                        _ => false,
                    }),
                },
                crate::shape_dialogue::FactSource::Session => overlay.rows.iter().any(|row| {
                    let resolution = registry.resolve(&row.predicate);
                    match fact.direction {
                        FactDirection::Forward | FactDirection::Attribute => {
                            row.subject == entity
                                && row.object == fact.object
                                && resolution.canonical == fact.predicate
                        }
                        FactDirection::Reverse => {
                            row.object == entity
                                && row.subject == fact.object
                                && resolution.canonical == fact.predicate
                        }
                    }
                }),
            })
        })
        .count()
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        1.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn yes_no(value: bool) -> &'static str {
    if value {
        "是"
    } else {
        "否"
    }
}

fn required_property(name: &str) -> PropertyConfig {
    PropertyConfig {
        name: name.to_string(),
        data_type: "string".to_string(),
        required: true,
        extraction_patterns: Vec::new(),
    }
}

fn optional_property(name: &str, data_type: &str) -> PropertyConfig {
    PropertyConfig {
        name: name.to_string(),
        data_type: data_type.to_string(),
        required: false,
        extraction_patterns: Vec::new(),
    }
}

fn entity(name: &str, etype: &str, properties: &[(&str, &str)]) -> PrologFact {
    PrologFact::Entity {
        name: name.to_string(),
        entity_type: etype.to_string(),
        properties: properties
            .iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect(),
    }
}

fn relation(from: &str, predicate: &str, to: &str) -> PrologFact {
    PrologFact::Relation {
        from: from.to_string(),
        rel_type: predicate.to_string(),
        to: to.to_string(),
    }
}

fn profile(
    name: &str,
    functional: bool,
    domain: &[&str],
    range: &[&str],
) -> PredicateProfileConfig {
    PredicateProfileConfig {
        name: name.to_string(),
        functional,
        domain: domain.iter().map(|value| value.to_string()).collect(),
        range: range.iter().map(|value| value.to_string()).collect(),
        ..Default::default()
    }
}

fn registry() -> PredicateRegistry {
    let mut registry = PredicateRegistry::new();
    registry.declare(&profile("出生地", false, &["人物"], &[]));
    registry.declare(&profile("国籍", false, &["人物"], &[]));
    registry.declare(&profile("职业", true, &[], &[]));
    registry.declare(&profile("女儿", false, &["人物"], &["人物"]));
    registry
}

fn types() -> TypeSystem {
    TypeSystem::from_config(&TypeSystemConfig {
        closure_gate: true,
        aliases: vec![TypeAliasConfig {
            alias: "Person".to_string(),
            canonical: "人物".to_string(),
        }],
        subclasses: Vec::new(),
        disjoint: vec![TypeDisjointConfig {
            left: "人物".to_string(),
            right: "地点".to_string(),
        }],
    })
}

fn base_shapes(registry: &PredicateRegistry, types: &TypeSystem) -> ShapeSet {
    ShapeSet::from_schema(
        &[EntityTypeConfig {
            label: "Person".to_string(),
            examples: Vec::new(),
            properties: vec![
                required_property("出生地"),
                PropertyConfig {
                    name: "职业".to_string(),
                    data_type: "string".to_string(),
                    required: false,
                    extraction_patterns: Vec::new(),
                },
            ],
        }],
        registry,
        types,
    )
}

fn base_facts() -> Vec<PrologFact> {
    vec![
        entity("刘德华", "Person", &[("name", "刘德华"), ("职业", "演员")]),
        entity("姚明", "Person", &[("name", "姚明"), ("出生地", "上海")]),
        entity("王五", "Person", &[("name", "王五")]),
        entity("赵六", "Person", &[("name", "赵六")]),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_eval_covers_all_four_layers_without_regressions() {
        let report = evaluate_seed();
        assert_eq!(report.gap_precision(), 1.0, "{}", report.render_text());
        assert_eq!(report.repeat_rate(), 0.0, "{}", report.render_text());
        assert_eq!(report.claim_precision(), 1.0, "{}", report.render_text());
        assert_eq!(report.grounded_rate(), 1.0, "{}", report.render_text());
        assert_eq!(
            report.required_coverage_after(),
            1.0,
            "{}",
            report.render_text()
        );
        assert_eq!(
            report.conflict_disclosure_rate(),
            1.0,
            "{}",
            report.render_text()
        );
        assert_eq!(report.abstain_rate(), 1.0, "{}", report.render_text());
        assert_eq!(report.session_check_rate(), 1.0, "{}", report.render_text());
        assert_eq!(report.cross_session_leaks, 0);
        assert!(report.gap_questions >= 4, "{}", report.render_text());
        assert!(report.claims_confirmed >= 4, "{}", report.render_text());
        assert!(report.render_text().contains("判定：通过"));
    }

    #[test]
    fn degraded_quality_metrics_fail_the_default_verdict() {
        let mut report = evaluate_seed();
        assert!(report.passed());
        assert!(report.render_text().contains("通过"));
        report.conflicts_disclosed = 0;
        assert!(!report.passed());
        let text = report.render_text();
        assert!(text.contains("未通过"));
        assert!(text.contains("冲突披露 0/"));
    }

    #[test]
    fn baseline_roundtrips_and_ratchets() {
        let report = evaluate_seed();
        let baseline = ShapeEvalBaseline::from_report(&report);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sfd_baseline.json");
        baseline.save(&path).unwrap();
        let loaded = ShapeEvalBaseline::load(&path).unwrap();
        assert_eq!(loaded, baseline);
        let verdict = compare_baseline(&loaded, &report, 1e-9);
        assert!(verdict.passed, "{}", verdict.render_text());
        assert!(verdict.digest_matches);
    }
}

//! SFD：Shape-Filling Dialogue（形状约束驱动的对话与回答合成）。
//!
//! `shape_gate` 回答「数据是否合规、回答是否有据」，`relation_acquisition`
//! 回答「问什么、答什么、信什么」。本模块把两者接成一条对话策略：
//! 形状缺口 -> 议程 -> 一次追问 -> 用户回答 -> 会话覆盖层 -> 重新收敛，
//! 并在补全后回到用户最初的问题，给出带来源、缺失披露与冲突披露的画像。
//!
//! 约束：用户补充只进入 QRA 会话覆盖层，不写图谱；未确认的候选不参与回答。

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::context_image::{canonical_json, fnv1a_hex};
use crate::predicate_registry::PredicateRegistry;
use crate::prolog_engine::PrologFact;
use crate::relation_acquisition::{AcquisitionStatus, AnswerAct, GoalOpen, RelationAcquisition};
use crate::shape_gate::{
    ConstraintKind, RepairAction, Severity, ShapeReport, ShapeSet, ShapeViolation,
};
use crate::symbolic_generator::localized_predicate_key;
use crate::type_system::TypeSystem;

pub const SFD_DIALECT: &str = "knowprolog-sfd/1";
pub const MAX_ASK_ATTEMPTS: usize = 2;
pub const MAX_AGENDA_GAPS: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AnswerStatus {
    Complete,
    Partial,
    Conflict,
    Unknown,
}

impl AnswerStatus {
    pub fn label(self) -> &'static str {
        match self {
            AnswerStatus::Complete => "完整",
            AnswerStatus::Partial => "部分",
            AnswerStatus::Conflict => "冲突",
            AnswerStatus::Unknown => "未知",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GapKind {
    Missing,
    Conflict,
}

impl GapKind {
    pub fn label(self) -> &'static str {
        match self {
            GapKind::Missing => "缺失",
            GapKind::Conflict => "冲突",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FactSource {
    Graph,
    Session,
}

impl FactSource {
    pub fn label(self) -> &'static str {
        match self {
            FactSource::Graph => "知识图谱",
            FactSource::Session => "本会话",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FactDirection {
    Attribute,
    Forward,
    Reverse,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnswerFact {
    pub predicate: String,
    pub object: String,
    pub direction: FactDirection,
    pub sources: Vec<FactSource>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub claim_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MissingSlot {
    pub predicate: String,
    pub surface: String,
    pub required: bool,
    pub expected: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConflictValue {
    pub value: String,
    pub sources: Vec<FactSource>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlotConflict {
    pub predicate: String,
    pub surface: String,
    pub values: Vec<ConflictValue>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShapeAnswer {
    pub entity: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entity_type: Option<String>,
    pub status: AnswerStatus,
    pub facts: Vec<AnswerFact>,
    pub missing: Vec<MissingSlot>,
    pub conflicts: Vec<SlotConflict>,
    pub graph_facts: usize,
    pub session_facts: usize,
    pub digest: String,
}

impl ShapeAnswer {
    pub fn build(
        entity: &str,
        facts: &[PrologFact],
        overlay: &crate::relation_acquisition::SessionOverlay,
        shapes: &ShapeSet,
        registry: &PredicateRegistry,
        types: &TypeSystem,
    ) -> (Self, ShapeReport) {
        let effective = effective_facts(facts, overlay);
        let report = shapes.validate(&effective, registry, types);
        let answer = Self::from_report(entity, facts, overlay, &report, shapes, registry, types);
        (answer, report)
    }

    fn from_report(
        entity: &str,
        facts: &[PrologFact],
        overlay: &crate::relation_acquisition::SessionOverlay,
        report: &ShapeReport,
        shapes: &ShapeSet,
        registry: &PredicateRegistry,
        types: &TypeSystem,
    ) -> Self {
        let mut answer_facts: Vec<AnswerFact> = Vec::new();
        let mut known = false;
        let mut entity_type = None;
        let mut graph_nodes: BTreeSet<String> = BTreeSet::new();

        for fact in facts {
            match fact {
                PrologFact::Entity {
                    name,
                    entity_type: raw_type,
                    properties,
                } => {
                    let name = name.trim();
                    if !name.is_empty() {
                        graph_nodes.insert(name.to_string());
                    }
                    if name != entity {
                        continue;
                    }
                    known = true;
                    if entity_type.is_none() {
                        let canonical = types.canonical(raw_type);
                        if !canonical.is_empty() {
                            entity_type = Some(canonical);
                        } else if !raw_type.trim().is_empty() {
                            entity_type = Some(raw_type.trim().to_string());
                        }
                    }
                    for (key, value) in properties {
                        if value.trim().is_empty() {
                            continue;
                        }
                        if is_name_key(key) && value.trim() == entity {
                            continue;
                        }
                        let predicate = registry.resolve(key).canonical;
                        if predicate.is_empty() {
                            continue;
                        }
                        push_fact(
                            &mut answer_facts,
                            predicate,
                            value.trim().to_string(),
                            FactDirection::Attribute,
                            FactSource::Graph,
                            None,
                        );
                    }
                }
                PrologFact::Relation { from, rel_type, to } => {
                    let from = from.trim();
                    let to = to.trim();
                    if !from.is_empty() {
                        graph_nodes.insert(from.to_string());
                    }
                    if !to.is_empty() {
                        graph_nodes.insert(to.to_string());
                    }
                    if from == entity && to == entity {
                        continue;
                    }
                    let resolution = registry.resolve(rel_type);
                    let (subject, object) = if resolution.inverted {
                        (to, from)
                    } else {
                        (from, to)
                    };
                    let predicate = resolution.canonical;
                    if predicate.is_empty() {
                        continue;
                    }
                    if subject == entity {
                        push_fact(
                            &mut answer_facts,
                            predicate,
                            object.to_string(),
                            FactDirection::Forward,
                            FactSource::Graph,
                            None,
                        );
                    } else if object == entity {
                        push_fact(
                            &mut answer_facts,
                            predicate,
                            subject.to_string(),
                            FactDirection::Reverse,
                            FactSource::Graph,
                            None,
                        );
                    }
                }
            }
        }

        let mut overlay_rows: Vec<_> = overlay
            .rows
            .iter()
            .filter(|row| row.subject == entity || row.object == entity)
            .collect();
        overlay_rows.sort_by(|left, right| {
            (&left.subject, &left.predicate, &left.object, &left.claim_id).cmp(&(
                &right.subject,
                &right.predicate,
                &right.object,
                &right.claim_id,
            ))
        });
        for row in overlay_rows {
            known = true;
            let predicate = registry.resolve(&row.predicate).canonical;
            if predicate.is_empty() {
                continue;
            }
            let direction = if row.subject == entity {
                if shapes
                    .nodes
                    .iter()
                    .flat_map(|node| node.properties.iter())
                    .find(|shape| shape.path == predicate)
                    .and_then(|shape| shape.datatype.as_deref())
                    .is_some()
                {
                    FactDirection::Attribute
                } else {
                    FactDirection::Forward
                }
            } else {
                FactDirection::Reverse
            };
            push_fact(
                &mut answer_facts,
                predicate,
                if row.subject == entity {
                    row.object.clone()
                } else {
                    row.subject.clone()
                },
                direction,
                FactSource::Session,
                Some(row.claim_id.clone()),
            );
        }

        answer_facts.sort_by(|left, right| {
            left.direction
                .cmp(&right.direction)
                .then_with(|| left.predicate.cmp(&right.predicate))
                .then_with(|| left.object.cmp(&right.object))
                .then_with(|| left.sources.cmp(&right.sources))
        });
        answer_facts.dedup_by(|left, right| {
            left.predicate == right.predicate
                && left.object == right.object
                && left.direction == right.direction
                && left.sources == right.sources
        });

        let mut missing = Vec::new();
        let mut conflicts = Vec::new();
        for violation in &report.violations {
            if violation.focus_node != entity {
                continue;
            }
            match &violation.repair {
                RepairAction::AskProperty { property, .. }
                    if violation.constraint == ConstraintKind::MinCount =>
                {
                    missing.push(MissingSlot {
                        predicate: property.clone(),
                        surface: display_predicate(property),
                        required: violation.severity == Severity::Violation,
                        expected: violation.expected.clone(),
                    });
                }
                RepairAction::DiscloseConflict {
                    property, values, ..
                } => {
                    conflicts.push(slot_conflict(property, values, &answer_facts));
                }
                _ => {}
            }
        }
        missing.sort_by(|left, right| left.predicate.cmp(&right.predicate));
        missing.dedup_by(|left, right| left.predicate == right.predicate);
        conflicts.sort_by(|left, right| left.predicate.cmp(&right.predicate));
        conflicts.dedup_by(|left, right| left.predicate == right.predicate);

        known = known || graph_nodes.contains(entity);
        let status = if !known {
            AnswerStatus::Unknown
        } else if !conflicts.is_empty() {
            AnswerStatus::Conflict
        } else if !missing.is_empty() {
            AnswerStatus::Partial
        } else {
            AnswerStatus::Complete
        };
        let graph_facts = answer_facts
            .iter()
            .filter(|fact| fact.sources.contains(&FactSource::Graph))
            .count();
        let session_facts = answer_facts
            .iter()
            .filter(|fact| fact.sources.contains(&FactSource::Session))
            .count();
        let digest = {
            let value = serde_json::json!({
                "entity": entity,
                "entity_type": &entity_type,
                "status": status,
                "facts": &answer_facts,
                "missing": &missing,
                "conflicts": &conflicts,
            });
            fnv1a_hex(canonical_json(&value).as_bytes())
        };
        Self {
            entity: entity.to_string(),
            entity_type,
            status,
            facts: answer_facts,
            missing,
            conflicts,
            graph_facts,
            session_facts,
            digest,
        }
    }

    pub fn missing_predicates(&self) -> Vec<&str> {
        self.missing
            .iter()
            .map(|slot| slot.predicate.as_str())
            .collect()
    }

    pub fn value_of(&self, predicate: &str) -> Vec<&str> {
        self.facts
            .iter()
            .filter(|fact| fact.predicate == predicate)
            .map(|fact| fact.object.as_str())
            .collect()
    }

    pub fn render_text(&self) -> String {
        let type_note = self
            .entity_type
            .as_deref()
            .map(|name| format!(" · {}", name))
            .unwrap_or_default();
        let mut lines = vec![format!(
            "【形状画像】「{}」{} · {}",
            self.entity,
            type_note,
            self.status.label()
        )];
        if self.status == AnswerStatus::Unknown {
            lines.push("  知识图谱与本会话都没有可用事实；系统不会为它编造画像。".to_string());
            lines.push(format!(
                "  依据：知识图谱 {} 条；本会话补充 {} 条。",
                self.graph_facts, self.session_facts
            ));
            return lines.join("\n");
        }
        for (direction, title) in [
            (FactDirection::Attribute, "属性"),
            (FactDirection::Forward, "关系"),
            (FactDirection::Reverse, "反向关系"),
        ] {
            let rows: Vec<&AnswerFact> = self
                .facts
                .iter()
                .filter(|fact| fact.direction == direction)
                .collect();
            if rows.is_empty() {
                continue;
            }
            lines.push(format!("  {}：", title));
            for fact in rows {
                let sources = fact
                    .sources
                    .iter()
                    .map(|source| source.label())
                    .collect::<Vec<_>>()
                    .join("+");
                lines.push(format!(
                    "    {} = {}〔{}〕",
                    display_predicate(&fact.predicate),
                    fact.object,
                    sources
                ));
            }
        }
        if !self.missing.is_empty() {
            let slots = self
                .missing
                .iter()
                .map(|slot| {
                    if slot.required {
                        format!("{}(必备)", slot.surface)
                    } else {
                        slot.surface.clone()
                    }
                })
                .collect::<Vec<_>>()
                .join("、");
            lines.push(format!(
                "  缺失：{} —— 知识图谱与本会话都没有记录，系统不会编造。",
                slots
            ));
        }
        if !self.conflicts.is_empty() {
            for conflict in &self.conflicts {
                let values = conflict
                    .values
                    .iter()
                    .map(|value| {
                        if value.sources.is_empty() {
                            value.value.clone()
                        } else {
                            format!(
                                "{}〔{}〕",
                                value.value,
                                value
                                    .sources
                                    .iter()
                                    .map(|source| source.label())
                                    .collect::<Vec<_>>()
                                    .join("+")
                            )
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("、");
                lines.push(format!(
                    "  冲突：「{}」有多个取值：{}；系统不替你选，需先裁定。",
                    conflict.surface, values
                ));
            }
        }
        lines.push(format!(
            "  依据：知识图谱 {} 条；本会话补充 {} 条。",
            self.graph_facts, self.session_facts
        ));
        lines.join("\n")
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShapeGap {
    pub id: String,
    pub entity: String,
    pub predicate: String,
    pub surface: String,
    pub kind: GapKind,
    pub severity: Severity,
    pub required: bool,
    pub expected: Vec<String>,
    pub values: Vec<String>,
    pub score: i64,
    pub cost: u32,
    pub question: String,
}

impl ShapeGap {
    pub fn is_askable(&self) -> bool {
        self.kind == GapKind::Missing
    }

    pub fn render_text(&self) -> String {
        format!(
            "- {} [{}] {}-[{}]-> ? · 分数 {} · 成本 {} · {}",
            self.id,
            self.kind.label(),
            self.entity,
            self.surface,
            self.score,
            self.cost,
            self.question
        )
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShapeAgenda {
    #[serde(default)]
    pub gaps: Vec<ShapeGap>,
}

impl ShapeAgenda {
    pub fn compile(report: &ShapeReport, registry: &PredicateRegistry) -> Self {
        let mut gaps = Vec::new();
        for violation in &report.violations {
            let Some(gap) = compile_gap(violation, registry) else {
                continue;
            };
            gaps.push(gap);
        }
        gaps.sort_by(|left, right| {
            right
                .score
                .cmp(&left.score)
                .then_with(|| left.cost.cmp(&right.cost))
                .then_with(|| left.entity.cmp(&right.entity))
                .then_with(|| left.predicate.cmp(&right.predicate))
                .then_with(|| left.kind.cmp(&right.kind))
        });
        gaps.dedup_by(|left, right| {
            left.entity == right.entity
                && left.predicate == right.predicate
                && left.kind == right.kind
        });
        gaps.truncate(MAX_AGENDA_GAPS);
        Self { gaps }
    }

    pub fn len(&self) -> usize {
        self.gaps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.gaps.is_empty()
    }

    pub fn next_for(&self, entity: &str, attempts: &BTreeMap<String, usize>) -> Option<&ShapeGap> {
        self.gaps.iter().find(|gap| {
            gap.is_askable()
                && gap.entity == entity
                && attempts.get(&gap.id).copied().unwrap_or(0) < MAX_ASK_ATTEMPTS
        })
    }

    pub fn next_global(&self, attempts: &BTreeMap<String, usize>) -> Option<&ShapeGap> {
        self.gaps.iter().find(|gap| {
            gap.is_askable() && attempts.get(&gap.id).copied().unwrap_or(0) < MAX_ASK_ATTEMPTS
        })
    }

    pub fn for_entity<'a>(&'a self, entity: &'a str) -> impl Iterator<Item = &'a ShapeGap> + 'a {
        self.gaps.iter().filter(move |gap| gap.entity == entity)
    }

    pub fn render_text(&self) -> String {
        let mut lines = vec![format!(
            "形状议程：{} 个缺口（可追问 {} 个）",
            self.gaps.len(),
            self.gaps.iter().filter(|gap| gap.is_askable()).count()
        )];
        if self.gaps.is_empty() {
            lines.push("  （空：当前知识满足所有可编译的形状约束）".to_string());
        }
        for gap in &self.gaps {
            lines.push(gap.render_text());
        }
        lines.join("\n")
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Consultation {
    pub id: String,
    pub gap: ShapeGap,
    pub question: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub original_question: Option<String>,
    pub asked_at: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SfdSnapshot {
    pub dialect: String,
    pub turn: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub focus: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub original_question: Option<String>,
    #[serde(default)]
    pub agenda: ShapeAgenda,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pending: Option<Consultation>,
    #[serde(default)]
    pub attempts: BTreeMap<String, usize>,
    #[serde(default)]
    pub resolved: BTreeSet<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SfdTurn {
    pub answer: ShapeAnswer,
    pub consultation: Option<Consultation>,
    pub message: String,
}

impl SfdTurn {
    pub fn render_text(&self) -> &str {
        &self.message
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ShapeDialogue {
    turn: usize,
    focus: Option<String>,
    original_question: Option<String>,
    agenda: ShapeAgenda,
    pending: Option<Consultation>,
    attempts: BTreeMap<String, usize>,
    resolved: BTreeSet<String>,
}

impl ShapeDialogue {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn turn_count(&self) -> usize {
        self.turn
    }

    pub fn focus(&self) -> Option<&str> {
        self.focus.as_deref()
    }

    pub fn original_question(&self) -> Option<&str> {
        self.original_question.as_deref()
    }

    pub fn agenda(&self) -> &ShapeAgenda {
        &self.agenda
    }

    pub fn pending(&self) -> Option<&Consultation> {
        self.pending.as_ref()
    }

    pub fn has_pending(&self) -> bool {
        self.pending.is_some()
    }

    pub fn clear_pending(&mut self, acquisition: &mut RelationAcquisition) {
        self.pending = None;
        acquisition.clear_goal();
    }

    pub fn reset(&mut self, acquisition: &mut RelationAcquisition) {
        *self = Self::new();
        acquisition.clear_goal();
    }

    pub fn snapshot(&self) -> SfdSnapshot {
        SfdSnapshot {
            dialect: SFD_DIALECT.to_string(),
            turn: self.turn,
            focus: self.focus.clone(),
            original_question: self.original_question.clone(),
            agenda: self.agenda.clone(),
            pending: self.pending.clone(),
            attempts: self.attempts.clone(),
            resolved: self.resolved.clone(),
        }
    }

    pub fn digest(&self) -> String {
        let value = serde_json::to_value(self.snapshot()).unwrap_or(serde_json::Value::Null);
        fnv1a_hex(canonical_json(&value).as_bytes())
    }

    pub fn restore(&mut self, snapshot: &SfdSnapshot) -> bool {
        if snapshot.dialect != SFD_DIALECT {
            return false;
        }
        self.turn = snapshot.turn;
        self.focus = snapshot.focus.clone();
        self.original_question = snapshot.original_question.clone();
        self.agenda = snapshot.agenda.clone();
        self.pending = snapshot.pending.clone();
        self.attempts = snapshot.attempts.clone();
        self.resolved = snapshot.resolved.clone();
        true
    }

    pub fn save_to_path(&self, path: &str) -> std::io::Result<()> {
        if let Some(parent) = Path::new(path).parent() {
            fs::create_dir_all(parent)?;
        }
        let text = serde_json::to_string_pretty(&self.snapshot())
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::Other, error))?;
        fs::write(path, format!("{}\n", text))
    }

    pub fn load_from_path(path: &str) -> Option<Self> {
        let text = fs::read_to_string(path).ok()?;
        let snapshot: SfdSnapshot = serde_json::from_str(&text).ok()?;
        let mut dialogue = Self::new();
        dialogue.restore(&snapshot).then_some(dialogue)
    }

    pub fn start_profile(
        &mut self,
        entity: &str,
        question: &str,
        facts: &[PrologFact],
        registry: &PredicateRegistry,
        types: &TypeSystem,
        shapes: &ShapeSet,
        acquisition: &mut RelationAcquisition,
    ) -> SfdTurn {
        let entity = entity.trim();
        self.turn += 1;
        self.focus = Some(entity.to_string());
        self.original_question = Some(question.trim().to_string());
        self.pending = None;

        let (answer, report) = ShapeAnswer::build(
            entity,
            facts,
            &acquisition.overlay(),
            shapes,
            registry,
            types,
        );
        self.agenda = ShapeAgenda::compile(&report, registry);
        let consultation = if answer.status == AnswerStatus::Partial {
            self.open_next(entity, facts, registry, types, shapes, acquisition)
        } else {
            None
        };
        let mut message = answer.render_text();
        if let Some(consultation) = &consultation {
            message.push_str("\n为了补全这张画像，我只问一个问题：\n");
            message.push_str(&consultation.question);
        }
        SfdTurn {
            answer,
            consultation,
            message,
        }
    }

    pub fn ask_next(
        &mut self,
        facts: &[PrologFact],
        registry: &PredicateRegistry,
        types: &TypeSystem,
        shapes: &ShapeSet,
        acquisition: &mut RelationAcquisition,
    ) -> Option<SfdTurn> {
        self.turn += 1;
        let effective = effective_facts(facts, &acquisition.overlay());
        let report = shapes.validate(&effective, registry, types);
        self.agenda = ShapeAgenda::compile(&report, registry);
        let consultation = self.open_next_global(facts, registry, types, shapes, acquisition)?;
        let entity = consultation.gap.entity.clone();
        let (answer, _) = ShapeAnswer::build(
            &entity,
            facts,
            &acquisition.overlay(),
            shapes,
            registry,
            types,
        );
        self.focus = Some(entity);
        let mut message = consultation.question.clone();
        if let Some(original) = &self.original_question {
            message.push_str(&format!("\n（补全后会回到你之前的问题「{}」。）", original));
        }
        Some(SfdTurn {
            answer,
            consultation: Some(consultation),
            message,
        })
    }

    pub fn respond(
        &mut self,
        input: &str,
        facts: &[PrologFact],
        registry: &PredicateRegistry,
        types: &TypeSystem,
        shapes: &ShapeSet,
        acquisition: &mut RelationAcquisition,
    ) -> Option<SfdTurn> {
        let pending = self.pending.clone()?;
        self.turn += 1;
        let outcome = acquisition.interpret_answer(input, facts, registry, types, shapes);
        let committed = outcome.commit.is_some();
        let status = outcome.status;
        let act = outcome.act;
        let mut message = outcome.message.clone();
        if let Some(follow_up) = &outcome.follow_up {
            message.push('\n');
            message.push_str(follow_up);
        }

        if committed {
            self.pending = None;
            self.resolved.insert(pending.gap.id.clone());
            let (answer, report) = ShapeAnswer::build(
                &pending.gap.entity,
                facts,
                &acquisition.overlay(),
                shapes,
                registry,
                types,
            );
            self.agenda = ShapeAgenda::compile(&report, registry);
            let next = if answer.status == AnswerStatus::Partial {
                self.open_next(
                    &pending.gap.entity,
                    facts,
                    registry,
                    types,
                    shapes,
                    acquisition,
                )
            } else {
                None
            };
            if let Some(original) = &self.original_question {
                message.push_str(&format!(
                    "\n\n回到你刚才的问题「{}」，补全后的画像是：\n{}",
                    original,
                    answer.render_text()
                ));
            } else {
                message.push_str(&format!("\n\n补全后的画像是：\n{}", answer.render_text()));
            }
            if let Some(consultation) = &next {
                message.push_str("\n还有下一个形状缺口：\n");
                message.push_str(&consultation.question);
            }
            return Some(SfdTurn {
                answer,
                consultation: next,
                message,
            });
        }

        if matches!(
            act,
            AnswerAct::Unknown | AnswerAct::Refuse | AnswerAct::Deny
        ) || matches!(status, Some(AcquisitionStatus::Refuted))
        {
            self.pending = None;
            acquisition.clear_goal();
            let (answer, report) = ShapeAnswer::build(
                &pending.gap.entity,
                facts,
                &acquisition.overlay(),
                shapes,
                registry,
                types,
            );
            self.agenda = ShapeAgenda::compile(&report, registry);
            message.push_str("\n本会话画像保持不变；未确认的取值不会进入回答。");
            return Some(SfdTurn {
                answer,
                consultation: None,
                message,
            });
        }

        let (answer, report) = ShapeAnswer::build(
            &pending.gap.entity,
            facts,
            &acquisition.overlay(),
            shapes,
            registry,
            types,
        );
        self.agenda = ShapeAgenda::compile(&report, registry);
        message.push_str("\n（补全尚未生效：只有确认后的取值才会进入本会话画像。）");
        Some(SfdTurn {
            answer,
            consultation: Some(pending),
            message,
        })
    }

    fn open_next(
        &mut self,
        entity: &str,
        facts: &[PrologFact],
        registry: &PredicateRegistry,
        types: &TypeSystem,
        shapes: &ShapeSet,
        acquisition: &mut RelationAcquisition,
    ) -> Option<Consultation> {
        let gap = self.agenda.next_for(entity, &self.attempts)?.clone();
        self.open_gap(gap, facts, registry, types, shapes, acquisition)
    }

    fn open_next_global(
        &mut self,
        facts: &[PrologFact],
        registry: &PredicateRegistry,
        types: &TypeSystem,
        shapes: &ShapeSet,
        acquisition: &mut RelationAcquisition,
    ) -> Option<Consultation> {
        let gap = self.agenda.next_global(&self.attempts)?.clone();
        self.open_gap(gap, facts, registry, types, shapes, acquisition)
    }

    fn open_gap(
        &mut self,
        gap: ShapeGap,
        facts: &[PrologFact],
        registry: &PredicateRegistry,
        types: &TypeSystem,
        shapes: &ShapeSet,
        acquisition: &mut RelationAcquisition,
    ) -> Option<Consultation> {
        if acquisition.has_pending_goal() {
            return None;
        }
        let surface = ask_surface(&gap, registry);
        let opening =
            acquisition.start_shape_gap(&gap.entity, &surface, facts, registry, types, shapes);
        let GoalOpen::Opened { goal } = opening else {
            return None;
        };
        *self.attempts.entry(gap.id.clone()).or_insert(0) += 1;
        let id = format!(
            "sc_{}",
            fnv1a_hex(
                canonical_json(&serde_json::json!({
                    "gap": &gap.id,
                    "turn": self.turn,
                    "question": &goal.question,
                }))
                .as_bytes(),
            )
        );
        let consultation = Consultation {
            id,
            gap,
            question: goal.question,
            original_question: self.original_question.clone(),
            asked_at: self.turn,
        };
        self.pending = Some(consultation.clone());
        Some(consultation)
    }

    pub fn report_text(&self) -> String {
        let mut lines = vec![format!(
            "形状填充对话（SFD v{}）：轮次 {}，焦点 {}，待答问题 {}",
            SFD_DIALECT.trim_start_matches("knowprolog-sfd/"),
            self.turn,
            self.focus.as_deref().unwrap_or("（无）"),
            if self.pending.is_some() { "1" } else { "0" }
        )];
        if let Some(question) = &self.original_question {
            lines.push(format!("原问题：{}", question));
        }
        if let Some(pending) = &self.pending {
            lines.push(format!(
                "待答：{}（缺口 {}；已问 {} 次）",
                pending.question,
                pending.gap.id,
                self.attempts.get(&pending.gap.id).copied().unwrap_or(0)
            ));
        }
        lines.push(self.agenda.render_text());
        lines.push(format!(
            "已收敛缺口 {} 个；提问次数 {}。",
            self.resolved.len(),
            self.attempts.values().sum::<usize>()
        ));
        lines.push(format!("对话摘要：{}", self.digest()));
        lines.join("\n")
    }
}

pub fn effective_facts(
    facts: &[PrologFact],
    overlay: &crate::relation_acquisition::SessionOverlay,
) -> Vec<PrologFact> {
    let mut effective = facts.to_vec();
    for row in &overlay.rows {
        effective.push(PrologFact::Entity {
            name: row.subject.clone(),
            entity_type: String::new(),
            properties: [(row.predicate.clone(), row.object.clone())]
                .into_iter()
                .collect(),
        });
    }
    effective
}

fn compile_gap(violation: &ShapeViolation, registry: &PredicateRegistry) -> Option<ShapeGap> {
    let (kind, property, values, expected) = match &violation.repair {
        RepairAction::AskProperty { property, .. }
            if violation.constraint == ConstraintKind::MinCount =>
        {
            (
                GapKind::Missing,
                property.clone(),
                Vec::new(),
                violation.expected.clone(),
            )
        }
        RepairAction::DiscloseConflict {
            property, values, ..
        } => (
            GapKind::Conflict,
            property.clone(),
            values.clone(),
            violation.expected.clone(),
        ),
        _ => return None,
    };
    let required = violation.severity == Severity::Violation;
    let surface = surface_for(&property, registry);
    let severity_score = match violation.severity {
        Severity::Violation => 300,
        Severity::Warning => 200,
        Severity::Info => 100,
    };
    let score = severity_score
        + if required { 60 } else { 0 }
        + if kind == GapKind::Conflict { 40 } else { 0 };
    let cost = if kind == GapKind::Conflict {
        3
    } else if required {
        1
    } else {
        2
    };
    let question = if kind == GapKind::Conflict {
        format!(
            "关于「{}」的「{}」我听到多个取值：{}；系统不会替你选，需要先裁定。",
            violation.focus_node,
            surface,
            values.join("、")
        )
    } else {
        format!(
            "你知道「{}」的「{}」吗？（期望：{}）",
            violation.focus_node,
            surface,
            if expected.is_empty() {
                "一个具体取值".to_string()
            } else {
                expected.join("、")
            }
        )
    };
    let id = format!(
        "sg_{}",
        fnv1a_hex(
            canonical_json(&serde_json::json!({
                "entity": &violation.focus_node,
                "predicate": &property,
                "kind": kind,
            }))
            .as_bytes(),
        )
    );
    Some(ShapeGap {
        id,
        entity: violation.focus_node.clone(),
        predicate: property,
        surface,
        kind,
        severity: violation.severity,
        required,
        expected,
        values,
        score,
        cost,
        question,
    })
}

fn slot_conflict(predicate: &str, values: &[String], facts: &[AnswerFact]) -> SlotConflict {
    let mut rows = Vec::new();
    for value in values {
        let mut sources = BTreeSet::new();
        for fact in facts {
            if fact.predicate == predicate && fact.object == *value {
                sources.extend(fact.sources.iter().copied());
            }
        }
        rows.push(ConflictValue {
            value: value.clone(),
            sources: sources.into_iter().collect(),
        });
    }
    rows.sort_by(|left, right| left.value.cmp(&right.value));
    rows.dedup_by(|left, right| left.value == right.value);
    SlotConflict {
        predicate: predicate.to_string(),
        surface: display_predicate(predicate),
        values: rows,
    }
}

fn push_fact(
    facts: &mut Vec<AnswerFact>,
    predicate: String,
    object: String,
    direction: FactDirection,
    source: FactSource,
    claim_id: Option<String>,
) {
    if let Some(existing) = facts.iter_mut().find(|fact| {
        fact.predicate == predicate && fact.object == object && fact.direction == direction
    }) {
        if !existing.sources.contains(&source) {
            existing.sources.push(source);
            existing.sources.sort();
        }
        if existing.claim_id.is_none() {
            existing.claim_id = claim_id;
        }
        return;
    }
    facts.push(AnswerFact {
        predicate,
        object,
        direction,
        sources: vec![source],
        claim_id,
    });
}

fn ask_surface(gap: &ShapeGap, registry: &PredicateRegistry) -> String {
    for (alias, canonical, inverted) in registry.alias_pairs() {
        if !inverted && canonical == gap.predicate && !alias.trim().is_empty() {
            return alias;
        }
    }
    if registry.resolve(&gap.surface).canonical == gap.predicate {
        return gap.surface.clone();
    }
    gap.predicate.clone()
}

fn surface_for(predicate: &str, registry: &PredicateRegistry) -> String {
    for (alias, canonical, inverted) in registry.alias_pairs() {
        if !inverted && canonical == predicate && !alias.trim().is_empty() {
            return alias;
        }
    }
    localized_predicate_key(predicate)
        .map(|label| label.to_string())
        .unwrap_or_else(|| predicate.to_string())
}

fn display_predicate(predicate: &str) -> String {
    localized_predicate_key(predicate)
        .map(|label| label.to_string())
        .unwrap_or_else(|| predicate.to_string())
}

fn is_name_key(key: &str) -> bool {
    matches!(
        key.trim().to_ascii_lowercase().as_str(),
        "name" | "姓名" | "名称" | "名字"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::predicate_registry::PredicateProfileConfig;
    use crate::shape_gate::ShapeSet;
    use crate::type_system::{TypeAliasConfig, TypeDisjointConfig, TypeSystemConfig};
    use crate::{EntityTypeConfig, PropertyConfig};
    use std::collections::HashMap;

    fn entity(name: &str, etype: &str, properties: &[(&str, &str)]) -> PrologFact {
        PrologFact::Entity {
            name: name.to_string(),
            entity_type: etype.to_string(),
            properties: properties
                .iter()
                .map(|(key, value)| (key.to_string(), value.to_string()))
                .collect::<HashMap<_, _>>(),
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

    fn shapes(registry: &PredicateRegistry, types: &TypeSystem) -> ShapeSet {
        ShapeSet::from_schema(
            &[EntityTypeConfig {
                label: "Person".to_string(),
                examples: Vec::new(),
                properties: vec![
                    PropertyConfig {
                        name: "出生地".to_string(),
                        data_type: "string".to_string(),
                        required: true,
                        extraction_patterns: Vec::new(),
                    },
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

    fn facts() -> Vec<PrologFact> {
        vec![
            entity("刘德华", "Person", &[("name", "刘德华"), ("职业", "演员")]),
            entity("姚明", "Person", &[("name", "姚明"), ("出生地", "上海")]),
            entity("王五", "Person", &[("name", "王五")]),
        ]
    }

    #[test]
    fn partial_profile_asks_one_required_gap_then_confirmation_completes_it() {
        let registry = registry();
        let types = types();
        let shapes = shapes(&registry, &types);
        let facts = facts();
        let mut acquisition = RelationAcquisition::new();
        let mut dialogue = ShapeDialogue::new();

        let first = dialogue.start_profile(
            "刘德华",
            "刘德华是谁？",
            &facts,
            &registry,
            &types,
            &shapes,
            &mut acquisition,
        );
        assert_eq!(first.answer.status, AnswerStatus::Partial);
        assert_eq!(first.answer.missing_predicates(), vec!["出生地"]);
        let consultation = first.consultation.expect("部分画像应追问一个缺口");
        assert_eq!(consultation.gap.predicate, "出生地");
        assert!(consultation.question.contains("出生地"));
        assert!(first.message.contains("只问一个问题"));

        let proposed = dialogue
            .respond("香港", &facts, &registry, &types, &shapes, &mut acquisition)
            .expect("待答状态应处理短答");
        assert_eq!(proposed.answer.status, AnswerStatus::Partial);
        assert!(proposed.answer.value_of("出生地").is_empty());
        assert!(acquisition.overlay().is_empty());

        let confirmed = dialogue
            .respond("确认", &facts, &registry, &types, &shapes, &mut acquisition)
            .expect("待答状态应处理确认");
        assert_eq!(confirmed.answer.status, AnswerStatus::Complete);
        assert_eq!(confirmed.answer.value_of("出生地"), vec!["香港"]);
        assert!(confirmed.answer.render_text().contains("本会话"));
        assert!(confirmed.message.contains("刘德华是谁？"));
        assert!(confirmed.consultation.is_none());
        assert_eq!(acquisition.overlay().values("刘德华", "出生地").len(), 1);
    }

    #[test]
    fn graph_functional_conflict_is_disclosed_and_never_silently_resolved() {
        let registry = registry();
        let types = types();
        let shapes = shapes(&registry, &types);
        let facts = vec![
            entity("刘德华", "Person", &[("name", "刘德华"), ("职业", "演员")]),
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
        assert_eq!(turn.answer.status, AnswerStatus::Conflict);
        assert_eq!(turn.answer.conflicts.len(), 1);
        let values: Vec<&str> = turn.answer.conflicts[0]
            .values
            .iter()
            .map(|value| value.value.as_str())
            .collect();
        assert_eq!(values, vec!["导演", "演员"]);
        assert!(turn.message.contains("冲突"));
        assert!(turn.message.contains("不替你选"));
        assert!(turn.consultation.is_none(), "冲突不能变成普通填空追问");
    }

    #[test]
    fn unknown_entity_abstains_without_fabricating_a_profile() {
        let registry = registry();
        let types = types();
        let shapes = shapes(&registry, &types);
        let facts = facts();
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
        assert_eq!(turn.answer.status, AnswerStatus::Unknown);
        assert!(turn.answer.facts.is_empty());
        assert!(turn.message.contains("不会为它编造画像"));
        assert!(turn.consultation.is_none());
        assert!(!acquisition.has_pending_goal());
    }

    #[test]
    fn snapshot_restores_pending_question_and_original_task() {
        let registry = registry();
        let types = types();
        let shapes = shapes(&registry, &types);
        let facts = facts();
        let mut acquisition = RelationAcquisition::new();
        let mut dialogue = ShapeDialogue::new();
        let _ = dialogue.start_profile(
            "王五",
            "王五是谁？",
            &facts,
            &registry,
            &types,
            &shapes,
            &mut acquisition,
        );
        assert!(dialogue.has_pending());
        let snapshot = dialogue.snapshot();
        let acquisition_snapshot = acquisition.snapshot();
        let mut restored = ShapeDialogue::new();
        assert!(restored.restore(&snapshot));
        let mut restored_acquisition = RelationAcquisition::new();
        assert!(restored_acquisition.restore(&acquisition_snapshot));
        assert!(restored.has_pending());
        assert_eq!(restored.original_question(), Some("王五是谁？"));

        let proposed = restored
            .respond(
                "北京",
                &facts,
                &registry,
                &types,
                &shapes,
                &mut restored_acquisition,
            )
            .unwrap();
        assert_eq!(proposed.answer.status, AnswerStatus::Partial);
        let confirmed = restored
            .respond(
                "确认",
                &facts,
                &registry,
                &types,
                &shapes,
                &mut restored_acquisition,
            )
            .unwrap();
        assert_eq!(confirmed.answer.status, AnswerStatus::Complete);
        assert!(confirmed.message.contains("王五是谁？"));
        assert_eq!(confirmed.answer.value_of("出生地"), vec!["北京"]);
    }

    #[test]
    fn session_overlay_does_not_leak_into_a_fresh_session() {
        let registry = registry();
        let types = types();
        let shapes = shapes(&registry, &types);
        let facts = facts();
        let mut acquisition = RelationAcquisition::new();
        let mut dialogue = ShapeDialogue::new();
        let _ = dialogue.start_profile(
            "刘德华",
            "刘德华是谁？",
            &facts,
            &registry,
            &types,
            &shapes,
            &mut acquisition,
        );
        let _ = dialogue.respond("香港", &facts, &registry, &types, &shapes, &mut acquisition);
        let _ = dialogue.respond("确认", &facts, &registry, &types, &shapes, &mut acquisition);
        let (completed, _) = ShapeAnswer::build(
            "刘德华",
            &facts,
            &acquisition.overlay(),
            &shapes,
            &registry,
            &types,
        );
        assert_eq!(completed.status, AnswerStatus::Complete);

        let fresh_acquisition = RelationAcquisition::new();
        let (fresh, _) = ShapeAnswer::build(
            "刘德华",
            &facts,
            &fresh_acquisition.overlay(),
            &shapes,
            &registry,
            &types,
        );
        assert_eq!(fresh.status, AnswerStatus::Partial);
        assert!(fresh.value_of("出生地").is_empty());
    }

    #[test]
    fn agenda_orders_missing_gaps_and_refuses_to_ask_conflicts() {
        let registry = registry();
        let types = types();
        let shapes = shapes(&registry, &types);
        let facts = vec![
            entity(
                "刘德华",
                "Person",
                &[("name", "刘德华"), ("出生地", "香港"), ("职业", "演员")],
            ),
            relation("刘德华", "职业", "导演"),
            entity("王五", "Person", &[("name", "王五")]),
        ];
        let report = shapes.validate(&facts, &registry, &types);
        let agenda = ShapeAgenda::compile(&report, &registry);
        let missing = agenda
            .gaps
            .iter()
            .find(|gap| gap.entity == "王五")
            .expect("王五应编译出缺失缺口");
        assert_eq!(missing.kind, GapKind::Missing);
        assert_eq!(missing.predicate, "出生地");
        assert!(missing.required);
        assert!(agenda
            .gaps
            .iter()
            .any(|gap| gap.entity == "刘德华" && gap.kind == GapKind::Conflict));

        let mut attempts = BTreeMap::new();
        assert_eq!(
            agenda.next_for("刘德华", &attempts),
            None,
            "冲突不是可追问的填空缺口"
        );
        attempts.insert(missing.id.clone(), MAX_ASK_ATTEMPTS);
        assert_eq!(agenda.next_for("王五", &attempts), None);
    }

    #[test]
    fn persisted_snapshot_roundtrips_on_disk() {
        let registry = registry();
        let types = types();
        let shapes = shapes(&registry, &types);
        let facts = facts();
        let mut acquisition = RelationAcquisition::new();
        let mut dialogue = ShapeDialogue::new();
        let _ = dialogue.start_profile(
            "王五",
            "王五是谁？",
            &facts,
            &registry,
            &types,
            &shapes,
            &mut acquisition,
        );
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sfd.json");
        dialogue.save_to_path(path.to_str().unwrap()).unwrap();
        let restored = ShapeDialogue::load_from_path(path.to_str().unwrap()).unwrap();
        assert_eq!(restored.digest(), dialogue.digest());
        assert_eq!(restored.pending(), dialogue.pending());
        assert_eq!(restored.original_question(), Some("王五是谁？"));
    }
}

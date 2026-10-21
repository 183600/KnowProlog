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
use crate::relation_acquisition::{
    AcquisitionStatus, AnswerAct, GoalOpen, GoalOrigin, RelationAcquisition,
};
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
pub struct AnswerIssue {
    pub predicate: String,
    pub surface: String,
    pub message: String,
    pub expected: Vec<String>,
    pub actual: Vec<String>,
}

/// 会话级冲突裁定：只影响本会话的画像与回答，不改写知识图谱。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ConflictRuling {
    pub entity: String,
    pub predicate: String,
    pub value: String,
    pub turn: usize,
}

impl ConflictRuling {
    pub fn render_text(&self) -> String {
        format!(
            "会话裁定：「{}」的「{}」采用「{}」（仅本会话，知识图谱未被改写）",
            self.entity,
            display_predicate(&self.predicate),
            self.value
        )
    }
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
    pub issues: Vec<AnswerIssue>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rulings: Vec<ConflictRuling>,
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

    /// 先按会话裁定裁剪互相冲突的取值，再生成画像；图谱本身保持不变。
    pub fn build_resolved(
        entity: &str,
        facts: &[PrologFact],
        overlay: &crate::relation_acquisition::SessionOverlay,
        shapes: &ShapeSet,
        registry: &PredicateRegistry,
        types: &TypeSystem,
        rulings: &[ConflictRuling],
    ) -> (Self, ShapeReport) {
        let (facts, overlay) = apply_rulings(facts, overlay, rulings, registry);
        let (mut answer, report) = Self::build(entity, &facts, &overlay, shapes, registry, types);
        answer.rulings = rulings
            .iter()
            .filter(|ruling| ruling.entity == entity)
            .cloned()
            .collect();
        answer.refresh_digest();
        (answer, report)
    }

    pub fn refresh_digest(&mut self) {
        let value = serde_json::json!({
            "entity": &self.entity,
            "entity_type": &self.entity_type,
            "status": self.status,
            "facts": &self.facts,
            "missing": &self.missing,
            "conflicts": &self.conflicts,
            "issues": &self.issues,
            "rulings": &self.rulings,
        });
        self.digest = fnv1a_hex(canonical_json(&value).as_bytes());
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
        let mut issues = Vec::new();
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
                _ => issues.push(AnswerIssue {
                    predicate: violation.path.clone(),
                    surface: display_predicate(&violation.path),
                    message: violation.message.clone(),
                    expected: violation.expected.clone(),
                    actual: violation.actual.clone(),
                }),
            }
        }
        missing.sort_by(|left, right| left.predicate.cmp(&right.predicate));
        missing.dedup_by(|left, right| left.predicate == right.predicate);
        conflicts.sort_by(|left, right| left.predicate.cmp(&right.predicate));
        conflicts.dedup_by(|left, right| left.predicate == right.predicate);
        issues.sort_by(|left, right| {
            left.predicate
                .cmp(&right.predicate)
                .then_with(|| left.message.cmp(&right.message))
        });
        issues.dedup_by(|left, right| {
            left.predicate == right.predicate && left.message == right.message
        });

        known = known || graph_nodes.contains(entity);
        let status = if !known {
            AnswerStatus::Unknown
        } else if !conflicts.is_empty() || !issues.is_empty() {
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
        let mut answer = Self {
            entity: entity.to_string(),
            entity_type,
            status,
            facts: answer_facts,
            missing,
            conflicts,
            issues,
            rulings: Vec::new(),
            graph_facts,
            session_facts,
            digest: String::new(),
        };
        answer.refresh_digest();
        answer
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
        if self.entity_type.is_none() {
            lines
                .push("  类型未声明：无法对应形状，系统不会替它推断类型或编造必备槽。".to_string());
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
        for issue in &self.issues {
            lines.push(format!(
                "  形状问题：{}（{}）；期望：{}；实际：{}。系统不会把不合形状的取值当作确定事实。",
                issue.surface,
                issue.message,
                issue.expected.join("、"),
                issue.actual.join("、")
            ));
        }
        for ruling in &self.rulings {
            lines.push(format!("  {}", ruling.render_text()));
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
                lines.push(format!(
                    "  可输入「裁定：{} 的 {} = <值>」在本会话内收敛该冲突（不改写知识图谱）。",
                    self.entity, conflict.surface
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
    #[serde(default)]
    pub blocked: BTreeSet<String>,
}

impl ShapeAgenda {
    pub fn compile(report: &ShapeReport, registry: &PredicateRegistry) -> Self {
        let mut gaps = Vec::new();
        let mut blocked = BTreeSet::new();
        for violation in &report.violations {
            let Some(gap) = compile_gap(violation, registry) else {
                blocked.insert(violation.focus_node.clone());
                continue;
            };
            if gap.kind == GapKind::Conflict {
                blocked.insert(gap.entity.clone());
            }
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
        Self { gaps, blocked }
    }

    pub fn len(&self) -> usize {
        self.gaps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.gaps.is_empty()
    }

    pub fn next_for(&self, entity: &str, attempts: &BTreeMap<String, usize>) -> Option<&ShapeGap> {
        if self.blocked.contains(entity) {
            return None;
        }
        self.gaps.iter().find(|gap| {
            gap.is_askable()
                && gap.entity == entity
                && attempts.get(&gap.id).copied().unwrap_or(0) < MAX_ASK_ATTEMPTS
        })
    }

    pub fn next_global(&self, attempts: &BTreeMap<String, usize>) -> Option<&ShapeGap> {
        self.gaps.iter().find(|gap| {
            gap.is_askable()
                && !self.blocked.contains(&gap.entity)
                && attempts.get(&gap.id).copied().unwrap_or(0) < MAX_ASK_ATTEMPTS
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
        if !self.blocked.is_empty() {
            lines.push(format!(
                "  待处理实体：{}（存在冲突或类型问题，先披露、不继续追问）。",
                self.blocked.iter().cloned().collect::<Vec<_>>().join("、")
            ));
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
    #[serde(default)]
    pub rulings: Vec<ConflictRuling>,
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
    rulings: Vec<ConflictRuling>,
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

    pub fn rulings(&self) -> &[ConflictRuling] {
        &self.rulings
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
            rulings: self.rulings.clone(),
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
        self.rulings = snapshot.rulings.clone();
        true
    }

    pub fn save_to_path(&self, path: &str) -> std::io::Result<()> {
        if let Some(parent) = Path::new(path).parent() {
            fs::create_dir_all(parent)?;
        }
        let text = serde_json::to_string_pretty(&self.snapshot()).map_err(std::io::Error::other)?;
        fs::write(path, format!("{}\n", text))
    }

    pub fn load_from_path(path: &str) -> Option<Self> {
        let text = fs::read_to_string(path).ok()?;
        let snapshot: SfdSnapshot = serde_json::from_str(&text).ok()?;
        let mut dialogue = Self::new();
        dialogue.restore(&snapshot).then_some(dialogue)
    }

    #[allow(clippy::too_many_arguments)]
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
        if acquisition
            .active_goal()
            .map(|goal| goal.origin == GoalOrigin::ShapeRepair)
            .unwrap_or(false)
        {
            acquisition.clear_goal();
        }

        let (answer, report) =
            self.build_answer(entity, facts, registry, types, shapes, acquisition);
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
        if self.pending.is_some() {
            return None;
        }
        self.turn += 1;
        let (ruled_facts, ruled_overlay) =
            apply_rulings(facts, &acquisition.overlay(), &self.rulings, registry);
        let effective = effective_facts(&ruled_facts, &ruled_overlay);
        let report = shapes.validate(&effective, registry, types);
        self.agenda = ShapeAgenda::compile(&report, registry);
        let consultation = self.open_next_global(facts, registry, types, shapes, acquisition)?;
        let entity = consultation.gap.entity.clone();
        let (answer, _) = self.build_answer(&entity, facts, registry, types, shapes, acquisition);
        if self.focus.as_deref() != Some(entity.as_str()) || self.original_question.is_none() {
            self.original_question = Some(format!("{}是谁？", entity));
        }
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
            acquisition.clear_goal();
            self.resolved.insert(pending.gap.id.clone());
            let (answer, report) = self.build_answer(
                &pending.gap.entity,
                facts,
                registry,
                types,
                shapes,
                acquisition,
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
            let (answer, report) = self.build_answer(
                &pending.gap.entity,
                facts,
                registry,
                types,
                shapes,
                acquisition,
            );
            self.agenda = ShapeAgenda::compile(&report, registry);
            message.push_str("\n本会话画像保持不变；未确认的取值不会进入回答。");
            return Some(SfdTurn {
                answer,
                consultation: None,
                message,
            });
        }

        let (answer, report) = self.build_answer(
            &pending.gap.entity,
            facts,
            registry,
            types,
            shapes,
            acquisition,
        );
        self.agenda = ShapeAgenda::compile(&report, registry);
        message.push_str("\n（补全尚未生效：只有确认后的取值才会进入本会话画像。）");
        Some(SfdTurn {
            answer,
            consultation: Some(pending),
            message,
        })
    }

    fn build_answer(
        &self,
        entity: &str,
        facts: &[PrologFact],
        registry: &PredicateRegistry,
        types: &TypeSystem,
        shapes: &ShapeSet,
        acquisition: &RelationAcquisition,
    ) -> (ShapeAnswer, ShapeReport) {
        ShapeAnswer::build_resolved(
            entity,
            facts,
            &acquisition.overlay(),
            shapes,
            registry,
            types,
            &self.rulings,
        )
    }

    /// 识别口语化的冲突裁定，例如「以香港为准」「不是深圳，是香港」。
    pub fn looks_like_conflict_ruling(input: &str) -> bool {
        let text = input.trim();
        (text.starts_with('以') && text.contains("为准"))
            || (text.starts_with("不是") && text.contains('是'))
            || text.starts_with("采用")
    }

    /// 把口语裁定落到当前焦点实体的某个冲突槽位；没有命中时返回 `None` 让出常规管线。
    #[allow(clippy::too_many_arguments)]
    pub fn resolve_conflict_text(
        &mut self,
        input: &str,
        facts: &[PrologFact],
        registry: &PredicateRegistry,
        types: &TypeSystem,
        shapes: &ShapeSet,
        acquisition: &mut RelationAcquisition,
    ) -> Option<Result<SfdTurn, String>> {
        let entity = self.focus.clone()?;
        let value = conflict_ruling_value(input)?;
        let (raw, _) = ShapeAnswer::build(
            &entity,
            facts,
            &acquisition.overlay(),
            shapes,
            registry,
            types,
        );
        let conflict = raw.conflicts.iter().find(|conflict| {
            conflict
                .values
                .iter()
                .any(|candidate| candidate.value == value)
        })?;
        let predicate = conflict.predicate.clone();
        Some(self.resolve_conflict(
            &entity,
            &predicate,
            &value,
            facts,
            registry,
            types,
            shapes,
            acquisition,
        ))
    }

    /// 记录一条会话级冲突裁定，并重新生成画像。
    ///
    /// 只允许裁定当前确实存在冲突的取值；裁定不写图谱，也不进入 QRA 覆盖层。
    #[allow(clippy::too_many_arguments)]
    pub fn resolve_conflict(
        &mut self,
        entity: &str,
        predicate: &str,
        value: &str,
        facts: &[PrologFact],
        registry: &PredicateRegistry,
        types: &TypeSystem,
        shapes: &ShapeSet,
        acquisition: &mut RelationAcquisition,
    ) -> Result<SfdTurn, String> {
        let entity = entity.trim();
        let value = value.trim();
        if entity.is_empty() || value.is_empty() {
            return Err("裁定需要实体与取值，例如「裁定：刘德华 的 职业 = 演员」。".to_string());
        }
        let canonical = registry.resolve(predicate).canonical;
        if canonical.is_empty() {
            return Err(format!("谓词「{}」为空，无法裁定。", predicate.trim()));
        }
        let (raw, _) = ShapeAnswer::build(
            entity,
            facts,
            &acquisition.overlay(),
            shapes,
            registry,
            types,
        );
        let conflict = raw
            .conflicts
            .iter()
            .find(|conflict| conflict.predicate == canonical)
            .ok_or_else(|| {
                format!(
                    "「{}」当前没有「{}」的冲突需要裁定。",
                    entity,
                    display_predicate(&canonical)
                )
            })?;
        if !conflict
            .values
            .iter()
            .any(|candidate| candidate.value == value)
        {
            let values = conflict
                .values
                .iter()
                .map(|candidate| candidate.value.as_str())
                .collect::<Vec<_>>()
                .join("、");
            return Err(format!(
                "「{}」的「{}」只有这些冲突值：{}；不能裁定为「{}」。",
                entity, conflict.surface, values, value
            ));
        }

        self.turn += 1;
        self.pending = None;
        acquisition.clear_goal();
        self.focus = Some(entity.to_string());
        if self.original_question.is_none() {
            self.original_question = Some(format!("{}是谁？", entity));
        }
        self.rulings
            .retain(|ruling| !(ruling.entity == entity && ruling.predicate == canonical));
        self.rulings.push(ConflictRuling {
            entity: entity.to_string(),
            predicate: canonical.clone(),
            value: value.to_string(),
            turn: self.turn,
        });
        self.rulings.sort();

        let (answer, report) =
            self.build_answer(entity, facts, registry, types, shapes, acquisition);
        self.agenda = ShapeAgenda::compile(&report, registry);
        let mut message = format!(
            "已在本会话裁定：「{}」的「{}」采用「{}」。知识图谱事实未被改写。",
            entity,
            display_predicate(&canonical),
            value
        );
        match answer.status {
            AnswerStatus::Conflict => {
                message.push_str("\n画像仍有未裁定的冲突或形状问题，系统不会替你做进一步选择。");
            }
            AnswerStatus::Partial => {
                message.push_str(
                    "\n该冲突已收敛，但画像仍有缺失槽位；可继续用「下一个形状问题」补全。",
                );
            }
            AnswerStatus::Unknown => {
                message.push_str("\n裁定后仍没有可用事实。");
            }
            AnswerStatus::Complete => {
                message.push_str("\n该冲突已收敛，画像恢复完整。");
            }
        }
        message.push('\n');
        message.push_str(&answer.render_text());
        Ok(SfdTurn {
            answer,
            consultation: None,
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

/// 从「以香港为准」「不是深圳，是香港」「采用香港」中提取裁定值。
fn conflict_ruling_value(input: &str) -> Option<String> {
    let text = input
        .trim()
        .trim_end_matches(['。', '！', '!', '；', ';'])
        .trim();
    let value = if let Some(rest) = text.strip_prefix('以') {
        rest.strip_suffix("为准")?
    } else if let Some(rest) = text.strip_prefix("不是") {
        let position = rest.find('是')?;
        &rest[position + '是'.len_utf8()..]
    } else {
        text.strip_prefix("采用")?
    };
    let value = value
        .trim()
        .trim_matches(['「', '」', '“', '”', '"', '\'', '，', ',', '：', ':']);
    (!value.is_empty()).then(|| value.to_string())
}

/// 解析显式裁定命令的主体：
/// `实体 的 谓词 = 值`、`实体 的 谓词 是 值` 或 `实体|谓词|值`。
pub fn parse_ruling(input: &str) -> Option<(String, String, String)> {
    let text = input.trim().trim_end_matches(['。', '；', ';']).trim();
    if text.is_empty() {
        return None;
    }
    if let Some((entity, rest)) = text.split_once('|') {
        let (predicate, value) = rest.split_once('|')?;
        return ruling_parts(entity, predicate, value);
    }
    let (entity, rest) = text.split_once('的')?;
    let rest = rest.trim();
    let separators = ["=", "＝", "采用", "改为", "为", "是"];
    let mut split: Option<(usize, &str)> = None;
    for separator in separators {
        if let Some(position) = rest.find(separator) {
            if position == 0 {
                continue;
            }
            let better = match split {
                Some((best, _)) => position < best,
                None => true,
            };
            if better {
                split = Some((position, separator));
            }
        }
    }
    let (position, separator) = split?;
    let predicate = &rest[..position];
    let value = &rest[position + separator.len()..];
    ruling_parts(entity, predicate, value)
}

fn ruling_parts(entity: &str, predicate: &str, value: &str) -> Option<(String, String, String)> {
    let entity = entity.trim();
    let predicate = predicate.trim();
    let value = value.trim();
    if entity.is_empty() || predicate.is_empty() || value.is_empty() {
        return None;
    }
    Some((entity.to_string(), predicate.to_string(), value.to_string()))
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

/// 按会话裁定过滤图谱事实与会话覆盖层：只保留被裁定的取值。
///
/// 返回的新事实集用于形状校验与画像生成；原图谱不会被修改。
pub fn apply_rulings(
    facts: &[PrologFact],
    overlay: &crate::relation_acquisition::SessionOverlay,
    rulings: &[ConflictRuling],
    registry: &PredicateRegistry,
) -> (Vec<PrologFact>, crate::relation_acquisition::SessionOverlay) {
    if rulings.is_empty() {
        return (facts.to_vec(), overlay.clone());
    }
    let mut filtered = Vec::with_capacity(facts.len());
    for fact in facts {
        match fact {
            PrologFact::Entity {
                name,
                entity_type,
                properties,
            } => {
                let mut kept = std::collections::HashMap::new();
                for (key, value) in properties {
                    let canonical = registry.resolve(key).canonical;
                    let ruled = ruling_for(rulings, name, &canonical);
                    if ruled.is_some_and(|ruling| value.trim() != ruling.value) {
                        continue;
                    }
                    kept.insert(key.clone(), value.clone());
                }
                filtered.push(PrologFact::Entity {
                    name: name.clone(),
                    entity_type: entity_type.clone(),
                    properties: kept,
                });
            }
            PrologFact::Relation { from, rel_type, to } => {
                let resolution = registry.resolve(rel_type);
                let (subject, object) = if resolution.inverted {
                    (to.as_str(), from.as_str())
                } else {
                    (from.as_str(), to.as_str())
                };
                if ruling_for(rulings, subject, &resolution.canonical)
                    .is_some_and(|ruling| object.trim() != ruling.value)
                {
                    continue;
                }
                filtered.push(fact.clone());
            }
        }
    }

    let mut rows = Vec::with_capacity(overlay.rows.len());
    for row in &overlay.rows {
        let canonical = registry.resolve(&row.predicate).canonical;
        if ruling_for(rulings, &row.subject, &canonical)
            .is_some_and(|ruling| row.object.trim() != ruling.value)
        {
            continue;
        }
        rows.push(row.clone());
    }
    (
        filtered,
        crate::relation_acquisition::SessionOverlay { rows },
    )
}

fn ruling_for<'a>(
    rulings: &'a [ConflictRuling],
    entity: &str,
    predicate: &str,
) -> Option<&'a ConflictRuling> {
    rulings
        .iter()
        .find(|ruling| ruling.entity == entity && ruling.predicate == predicate)
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
    use crate::relation_acquisition::SessionOverlay;
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
    fn dangling_relation_node_is_not_silently_typed() {
        let registry = registry();
        let types = types();
        let shapes = shapes(&registry, &types);
        let facts = vec![relation("刘德华", "出生地", "香港")];
        let (answer, _) = ShapeAnswer::build(
            "香港",
            &facts,
            &SessionOverlay::default(),
            &shapes,
            &registry,
            &types,
        );
        assert!(answer.entity_type.is_none());
        assert!(answer.render_text().contains("类型未声明"));
    }

    #[test]
    fn datatype_violation_is_disclosed_instead_of_claimed_complete() {
        let registry = registry();
        let types = types();
        let shapes = ShapeSet::from_schema(
            &[EntityTypeConfig {
                label: "Person".to_string(),
                examples: Vec::new(),
                properties: vec![PropertyConfig {
                    name: "年龄".to_string(),
                    data_type: "integer".to_string(),
                    required: false,
                    extraction_patterns: Vec::new(),
                }],
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
        assert_eq!(turn.answer.status, AnswerStatus::Conflict);
        assert!(!turn.answer.issues.is_empty());
        assert!(dialogue.agenda().blocked.contains("刘德华"));
        assert!(dialogue.agenda().next_global(&BTreeMap::new()).is_none());
        assert!(turn.message.contains("形状问题"));
        assert!(turn.message.contains("不会把不合形状的取值当作确定事实"));
        assert!(
            turn.consultation.is_none(),
            "类型问题不能被当作普通缺口继续追问"
        );
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
        assert!(agenda.blocked.contains("刘德华"));

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
    fn confirmation_advances_to_the_next_gap_without_repeating() {
        let registry = registry();
        let types = types();
        let shapes = ShapeSet::from_schema(
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
                        name: "国籍".to_string(),
                        data_type: "string".to_string(),
                        required: true,
                        extraction_patterns: Vec::new(),
                    },
                ],
            }],
            &registry,
            &types,
        );
        let facts = vec![entity("刘德华", "Person", &[("name", "刘德华")])];
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
        assert!(first.answer.missing.len() >= 2);
        let first_gap = first.consultation.unwrap().gap.predicate;
        let value = if first_gap == "出生地" {
            "香港"
        } else {
            "中国"
        };
        let _ = dialogue.respond(value, &facts, &registry, &types, &shapes, &mut acquisition);
        let second = dialogue
            .respond("确认", &facts, &registry, &types, &shapes, &mut acquisition)
            .unwrap();
        assert_eq!(second.answer.status, AnswerStatus::Partial);
        let next = second.consultation.expect("补一个缺口后应继续问下一个");
        assert_ne!(next.gap.predicate, first_gap);
        assert_eq!(
            dialogue.attempts.get(&next.gap.id).copied(),
            Some(1),
            "每个缺口只问一次"
        );
    }

    #[test]
    fn global_agenda_question_carries_its_own_original_task() {
        let registry = registry();
        let types = types();
        let shapes = shapes(&registry, &types);
        let facts = facts();
        let mut acquisition = RelationAcquisition::new();
        let mut dialogue = ShapeDialogue::new();
        let turn = dialogue
            .ask_next(&facts, &registry, &types, &shapes, &mut acquisition)
            .expect("全局议程应能打开一个缺口");
        let consultation = turn.consultation.unwrap();
        assert_eq!(dialogue.focus(), Some(consultation.gap.entity.as_str()));
        assert_eq!(
            dialogue.original_question(),
            Some(format!("{}是谁？", consultation.gap.entity).as_str())
        );
        assert!(!acquisition.has_pending_goal() || dialogue.has_pending());
        assert!(
            dialogue
                .ask_next(&facts, &registry, &types, &shapes, &mut acquisition)
                .is_none(),
            "已有待答问题时不应覆盖"
        );
    }

    #[test]
    fn conflict_ruling_requires_a_listed_value_and_converges_only_in_session() {
        let registry = registry();
        let types = types();
        let shapes = shapes(&registry, &types);
        let facts = vec![
            entity(
                "刘德华",
                "Person",
                &[("name", "刘德华"), ("职业", "演员"), ("出生地", "香港")],
            ),
            relation("刘德华", "职业", "导演"),
        ];
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
        assert_eq!(first.answer.status, AnswerStatus::Conflict);

        let error = dialogue
            .resolve_conflict(
                "刘德华",
                "职业",
                "歌手",
                &facts,
                &registry,
                &types,
                &shapes,
                &mut acquisition,
            )
            .expect_err("不能裁定为不在冲突列表里的值");
        assert!(error.contains("只有这些冲突值"), "{}", error);

        let turn = dialogue
            .resolve_conflict(
                "刘德华",
                "职业",
                "演员",
                &facts,
                &registry,
                &types,
                &shapes,
                &mut acquisition,
            )
            .expect("列出的冲突值应可裁定");
        assert_eq!(turn.answer.status, AnswerStatus::Complete);
        assert_eq!(turn.answer.value_of("职业"), vec!["演员"]);
        assert!(turn
            .answer
            .value_of("职业")
            .iter()
            .all(|value| *value != "导演"));
        assert_eq!(turn.answer.rulings.len(), 1);
        assert!(turn.message.contains("本会话裁定"));
        assert!(turn.answer.render_text().contains("知识图谱未被改写"));
        assert!(acquisition.overlay().is_empty());

        let (raw, _) = ShapeAnswer::build(
            "刘德华",
            &facts,
            &acquisition.overlay(),
            &shapes,
            &registry,
            &types,
        );
        assert_eq!(
            raw.status,
            AnswerStatus::Conflict,
            "会话裁定不能改写图谱事实"
        );

        let snapshot = dialogue.snapshot();
        let mut restored = ShapeDialogue::new();
        assert!(restored.restore(&snapshot));
        assert_eq!(restored.rulings().len(), 1);
        let (resolved, _) = ShapeAnswer::build_resolved(
            "刘德华",
            &facts,
            &acquisition.overlay(),
            &shapes,
            &registry,
            &types,
            restored.rulings(),
        );
        assert_eq!(resolved.status, AnswerStatus::Complete);
        assert_eq!(resolved.value_of("职业"), vec!["演员"]);
    }

    #[test]
    fn colloquial_conflict_ruling_resolves_the_focused_slot() {
        let registry = registry();
        let types = types();
        let shapes = shapes(&registry, &types);
        let facts = vec![
            entity(
                "刘德华",
                "Person",
                &[("name", "刘德华"), ("职业", "演员"), ("出生地", "香港")],
            ),
            relation("刘德华", "职业", "导演"),
        ];
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
        assert_eq!(first.answer.status, AnswerStatus::Conflict);
        assert!(first.message.contains("裁定："));
        assert!(ShapeDialogue::looks_like_conflict_ruling("以演员为准"));
        assert!(ShapeDialogue::looks_like_conflict_ruling(
            "不是导演，是演员"
        ));
        assert!(!ShapeDialogue::looks_like_conflict_ruling("刘德华是谁？"));

        let turn = dialogue
            .resolve_conflict_text(
                "不是导演，是演员",
                &facts,
                &registry,
                &types,
                &shapes,
                &mut acquisition,
            )
            .expect("应命中当前职业冲突")
            .expect("演员是列出的冲突值");
        assert_eq!(turn.answer.status, AnswerStatus::Complete);
        assert_eq!(turn.answer.value_of("职业"), vec!["演员"]);
        assert_eq!(dialogue.focus(), Some("刘德华"));
    }

    #[test]
    fn agenda_after_ruling_does_not_stay_blocked_by_the_resolved_conflict() {
        let registry = registry();
        let types = types();
        let shapes = shapes(&registry, &types);
        let facts = vec![
            entity("刘德华", "Person", &[("name", "刘德华"), ("职业", "演员")]),
            relation("刘德华", "职业", "导演"),
        ];
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
        assert_eq!(first.answer.status, AnswerStatus::Conflict);
        let ruled = dialogue
            .resolve_conflict(
                "刘德华",
                "职业",
                "演员",
                &facts,
                &registry,
                &types,
                &shapes,
                &mut acquisition,
            )
            .unwrap();
        assert_eq!(ruled.answer.status, AnswerStatus::Partial);
        let next = dialogue
            .ask_next(&facts, &registry, &types, &shapes, &mut acquisition)
            .expect("裁定后应能继续追问剩余缺失槽位");
        assert_eq!(
            next.consultation.unwrap().gap.predicate,
            "出生地",
            "已裁定的职业冲突不应继续阻塞议程"
        );
    }

    #[test]
    fn parse_ruling_accepts_explicit_command_forms() {
        assert_eq!(
            parse_ruling("刘德华 的 职业 = 演员"),
            Some(("刘德华".to_string(), "职业".to_string(), "演员".to_string()))
        );
        assert_eq!(
            parse_ruling("刘德华的出生地是香港。"),
            Some((
                "刘德华".to_string(),
                "出生地".to_string(),
                "香港".to_string()
            ))
        );
        assert_eq!(
            parse_ruling("刘德华|职业|导演"),
            Some(("刘德华".to_string(), "职业".to_string(), "导演".to_string()))
        );
        assert!(parse_ruling("刘德华").is_none());
        assert!(parse_ruling("刘德华 的 职业 = ").is_none());
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

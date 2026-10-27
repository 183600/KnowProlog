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
            self.attempts
                .insert(pending.gap.id.clone(), MAX_ASK_ATTEMPTS);
            let (answer, report) = self.build_answer(
                &pending.gap.entity,
                facts,
                registry,
                types,
                shapes,
                acquisition,
            );
            self.agenda = ShapeAgenda::compile(&report, registry);
            message
                .push_str("\n本会话画像保持不变；未确认的取值不会进入回答，也不会重复追问该缺口。");
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
        if !self.rulings.is_empty() {
            lines.push(format!("会话裁定 {} 条：", self.rulings.len()));
            for ruling in &self.rulings {
                lines.push(format!("  {}", ruling.render_text()));
            }
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

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::ontology_reasoning::{
    apply_ontology_lesson, canonicalize_lessons, render_ontology_lesson_zh, OntologyClosure,
    OntologyLesson, OntologyProgram, OntologyReasoner,
};
use crate::prolog_engine::PrologFact;

pub const LEDGER_FILE: &str = "owl_annotations.json";
pub const BASE_NS: &str = "http://example.org/kg/";
pub const HITL_NS: &str = "http://example.org/kg/hitl#";

const RDF_TYPE_IRI: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
const RDFS_SUBCLASS_IRI: &str = "http://www.w3.org/2000/01/rdf-schema#subClassOf";
const RDFS_SUBPROPERTY_IRI: &str = "http://www.w3.org/2000/01/rdf-schema#subPropertyOf";
const RDFS_DOMAIN_IRI: &str = "http://www.w3.org/2000/01/rdf-schema#domain";
const RDFS_RANGE_IRI: &str = "http://www.w3.org/2000/01/rdf-schema#range";
const OWL_DISJOINT_IRI: &str = "http://www.w3.org/2002/07/owl#disjointWith";
const OWL_INVERSE_IRI: &str = "http://www.w3.org/2002/07/owl#inverseOf";
const OWL_TRANSITIVE_IRI: &str = "http://www.w3.org/2002/07/owl#TransitiveProperty";
const OWL_SYMMETRIC_IRI: &str = "http://www.w3.org/2002/07/owl#SymmetricProperty";

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphObservation {
    pub instances: Vec<(String, String)>,
    pub relations: Vec<(String, String, String)>,
}

impl GraphObservation {
    pub fn new(instances: Vec<(String, String)>, relations: Vec<(String, String, String)>) -> Self {
        let mut instances: Vec<(String, String)> = instances
            .into_iter()
            .filter(|(individual, class)| !individual.trim().is_empty() && !class.trim().is_empty())
            .collect();
        instances.sort();
        instances.dedup();
        let mut relations: Vec<(String, String, String)> = relations
            .into_iter()
            .filter(|(subject, predicate, object)| {
                !subject.trim().is_empty()
                    && !predicate.trim().is_empty()
                    && !object.trim().is_empty()
                    && predicate != "name"
            })
            .collect();
        relations.sort();
        relations.dedup();
        Self {
            instances,
            relations,
        }
    }

    pub fn from_facts(facts: &[PrologFact]) -> Self {
        let mut instances = Vec::new();
        let mut relations = Vec::new();
        for fact in facts {
            match fact {
                PrologFact::Entity {
                    name, entity_type, ..
                } => {
                    if !name.trim().is_empty() && !entity_type.trim().is_empty() {
                        instances.push((name.clone(), entity_type.clone()));
                    }
                }
                PrologFact::Relation { from, rel_type, to } => {
                    relations.push((from.clone(), rel_type.clone(), to.clone()));
                }
            }
        }
        Self::new(instances, relations)
    }

    pub fn is_empty(&self) -> bool {
        self.instances.is_empty() && self.relations.is_empty()
    }

    pub fn reasoner(&self, base: &OntologyProgram) -> OntologyReasoner {
        let mut reasoner = OntologyReasoner::new(32);
        for lesson in &base.lessons {
            apply_ontology_lesson(&mut reasoner, lesson);
        }
        for (individual, class) in &self.instances {
            reasoner.add_instance(individual, class);
        }
        for (subject, predicate, object) in &self.relations {
            reasoner.add_relation(subject, predicate, object);
        }
        reasoner
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ImpactPreview {
    pub new_type_assertions: usize,
    pub new_relations: usize,
    pub new_conflicts: usize,
    pub score: i64,
}

impl ImpactPreview {
    pub fn changed(&self) -> bool {
        self.new_type_assertions > 0 || self.new_relations > 0
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Proposal {
    pub key: String,
    pub lesson: OntologyLesson,
    pub rationale: String,
    pub evidence: Vec<String>,
    pub support: usize,
    pub purity: f64,
    pub impact: ImpactPreview,
}

impl Proposal {
    pub fn kind(&self) -> &'static str {
        lesson_kind(&self.lesson)
    }

    pub fn render(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "{}：{}",
            lesson_kind(&self.lesson),
            render_ontology_lesson_zh(&self.lesson)
        ));
        lines.push(format!(
            "证据：支持 {}，纯度 {:.0}%；影响：新增类型 {}、新增关系 {}、新增冲突 {}（评分 {}）",
            self.support,
            self.purity * 100.0,
            self.impact.new_type_assertions,
            self.impact.new_relations,
            self.impact.new_conflicts,
            self.impact.score
        ));
        if !self.rationale.is_empty() {
            lines.push(format!("理由：{}", self.rationale));
        }
        if !self.evidence.is_empty() {
            lines.push(format!("样例：{}", self.evidence.join("；")));
        }
        lines.join("\n")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Decision {
    Proposed,
    Accepted,
    Rejected,
    Deferred,
}

impl Decision {
    pub fn label(self) -> &'static str {
        match self {
            Decision::Proposed => "待裁决",
            Decision::Accepted => "已接受",
            Decision::Rejected => "已驳回",
            Decision::Deferred => "已暂缓",
        }
    }

    fn machine_label(self) -> &'static str {
        match self {
            Decision::Proposed => "proposed",
            Decision::Accepted => "accepted",
            Decision::Rejected => "rejected",
            Decision::Deferred => "deferred",
        }
    }

    fn parse(text: &str) -> Option<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "proposed" => Some(Decision::Proposed),
            "accepted" => Some(Decision::Accepted),
            "rejected" => Some(Decision::Rejected),
            "deferred" => Some(Decision::Deferred),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum ReviewAction {
    Accept,
    Reject,
    Defer,
    Revise { lesson: OntologyLesson },
}

impl ReviewAction {
    fn label(&self) -> &'static str {
        match self {
            ReviewAction::Accept => "accept",
            ReviewAction::Reject => "reject",
            ReviewAction::Defer => "defer",
            ReviewAction::Revise { .. } => "revise",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReviewRecord {
    pub action: ReviewAction,
    pub reviewer: String,
    pub note: String,
    pub at_version: u64,
}

impl ReviewRecord {
    pub fn render(&self) -> String {
        let note = if self.note.is_empty() {
            String::new()
        } else {
            format!("，理由：{}", self.note)
        };
        format!(
            "{}（{}，版本 {}{}）",
            self.action.label(),
            self.reviewer,
            self.at_version,
            note
        )
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReviewEntry {
    pub id: String,
    pub proposal: Proposal,
    pub status: Decision,
    pub effective: OntologyLesson,
    #[serde(default)]
    pub history: Vec<ReviewRecord>,
}

impl ReviewEntry {
    pub fn last_review(&self) -> Option<&ReviewRecord> {
        self.history.last()
    }

    pub fn render(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "[{}] {}（{}）",
            self.id,
            self.status.label(),
            self.proposal.kind()
        ));
        lines.push(self.proposal.render());
        if self.effective != self.proposal.lesson {
            lines.push(format!(
                "生效公理：{}",
                render_ontology_lesson_zh(&self.effective)
            ));
        }
        if let Some(review) = self.last_review() {
            lines.push(format!("最近裁决：{}", review.render()));
        }
        lines.join("\n")
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ProposeOutcome {
    pub added: Vec<String>,
    pub updated: Vec<String>,
    pub suppressed: Vec<String>,
    pub unchanged: usize,
}

impl ProposeOutcome {
    pub fn total_changes(&self) -> usize {
        self.added.len() + self.updated.len()
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ReviewOutcome {
    pub id: String,
    pub status: Decision,
    pub effective: OntologyLesson,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotationLedger {
    #[serde(default)]
    pub version: u64,
    #[serde(default = "default_next_id")]
    pub next_id: u64,
    #[serde(default)]
    pub entries: Vec<ReviewEntry>,
}

fn default_next_id() -> u64 {
    1
}

impl Default for AnnotationLedger {
    fn default() -> Self {
        Self {
            version: 0,
            next_id: default_next_id(),
            entries: Vec::new(),
        }
    }
}

pub enum LedgerLoad {
    Missing,
    Loaded(AnnotationLedger),
    Damaged(String),
}

impl AnnotationLedger {
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn accepted_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.status == Decision::Accepted)
            .count()
    }

    pub fn pending_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.status == Decision::Proposed)
            .count()
    }

    pub fn rejected_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.status == Decision::Rejected)
            .count()
    }

    pub fn accepted_lessons(&self) -> Vec<OntologyLesson> {
        canonicalize_lessons(
            self.entries
                .iter()
                .filter(|entry| entry.status == Decision::Accepted)
                .map(|entry| entry.effective.clone())
                .collect(),
        )
    }

    pub fn accepted_program(&self) -> OntologyProgram {
        OntologyProgram::from_lessons(self.accepted_lessons())
    }

    pub fn find(&self, selector: &str) -> Option<&ReviewEntry> {
        self.entries
            .iter()
            .find(|entry| entry.id == selector || entry.proposal.key == selector)
    }

    fn allocate_id(&mut self) -> String {
        let id = format!("owl-{:04}", self.next_id);
        self.next_id = self.next_id.saturating_add(1);
        id
    }

    fn refresh_next_id(&mut self) {
        let mut max_id = 0u64;
        for entry in &self.entries {
            if let Some(number) = entry
                .id
                .strip_prefix("owl-")
                .and_then(|text| text.parse::<u64>().ok())
            {
                max_id = max_id.max(number);
            }
        }
        self.next_id = self.next_id.max(max_id.saturating_add(1)).max(1);
    }

    pub fn propose(&mut self, proposals: Vec<Proposal>) -> ProposeOutcome {
        let mut outcome = ProposeOutcome::default();
        for proposal in proposals {
            let key = proposal.key.clone();
            let existing = self
                .entries
                .iter()
                .position(|entry| entry.proposal.key == key);
            match existing {
                None => {
                    let id = self.allocate_id();
                    let effective = proposal.lesson.clone();
                    self.entries.push(ReviewEntry {
                        id: id.clone(),
                        proposal,
                        status: Decision::Proposed,
                        effective,
                        history: Vec::new(),
                    });
                    outcome.added.push(id);
                }
                Some(index) => {
                    let old_status = self.entries[index].status;
                    match old_status {
                        Decision::Rejected => {
                            if materially_stronger(&self.entries[index].proposal, &proposal) {
                                let entry = &mut self.entries[index];
                                entry.proposal = proposal;
                                entry.effective = entry.proposal.lesson.clone();
                                entry.status = Decision::Proposed;
                                entry.history.push(ReviewRecord {
                                    action: ReviewAction::Defer,
                                    reviewer: "system:reopen".to_string(),
                                    note: "新证据显著增强，重新提请裁决".to_string(),
                                    at_version: self.version.saturating_add(1),
                                });
                                outcome.updated.push(entry.id.clone());
                            } else {
                                outcome.suppressed.push(self.entries[index].id.clone());
                            }
                        }
                        Decision::Accepted => {
                            outcome.unchanged += 1;
                        }
                        Decision::Proposed | Decision::Deferred => {
                            let entry = &mut self.entries[index];
                            entry.proposal = proposal;
                            entry.effective = entry.proposal.lesson.clone();
                            outcome.updated.push(entry.id.clone());
                        }
                    }
                }
            }
        }
        if outcome.total_changes() > 0 {
            self.version = self.version.saturating_add(1);
        }
        self.entries.sort_by(|left, right| left.id.cmp(&right.id));
        outcome
    }

    pub fn review(
        &mut self,
        selector: &str,
        action: ReviewAction,
        reviewer: &str,
        note: &str,
    ) -> Result<ReviewOutcome, String> {
        let index = self
            .entries
            .iter()
            .position(|entry| entry.id == selector || entry.proposal.key == selector)
            .ok_or_else(|| format!("未找到候选公理：{selector}"))?;
        let version = self.version.saturating_add(1);
        let reviewer = {
            let trimmed = reviewer.trim();
            if trimmed.is_empty() {
                "anonymous"
            } else {
                trimmed
            }
        };
        let status;
        {
            let entry = &mut self.entries[index];
            status = match &action {
                ReviewAction::Accept => {
                    entry.effective = entry.proposal.lesson.clone();
                    Decision::Accepted
                }
                ReviewAction::Reject => Decision::Rejected,
                ReviewAction::Defer => Decision::Deferred,
                ReviewAction::Revise { lesson } => {
                    entry.effective = lesson.clone();
                    Decision::Accepted
                }
            };
            entry.status = status;
            entry.history.push(ReviewRecord {
                action,
                reviewer: reviewer.to_string(),
                note: note.trim().to_string(),
                at_version: version,
            });
        }
        self.version = version;
        let entry = &self.entries[index];
        Ok(ReviewOutcome {
            id: entry.id.clone(),
            status: entry.status,
            effective: entry.effective.clone(),
        })
    }

    pub fn merge_entries(&mut self, other: &AnnotationLedger, imported_wins: bool) -> usize {
        let mut changed = 0usize;
        for incoming in &other.entries {
            if let Some(index) = self
                .entries
                .iter()
                .position(|entry| entry.proposal.key == incoming.proposal.key)
            {
                if imported_wins && incoming.status != Decision::Proposed {
                    let mut replacement = incoming.clone();
                    replacement.id = self.entries[index].id.clone();
                    self.entries[index] = replacement;
                    changed += 1;
                }
                continue;
            }
            let mut imported = incoming.clone();
            if self.entries.iter().any(|entry| entry.id == imported.id) || imported.id.is_empty() {
                imported.id = self.allocate_id();
            }
            self.entries.push(imported);
            changed += 1;
        }
        if changed > 0 {
            self.entries.sort_by(|left, right| left.id.cmp(&right.id));
            self.refresh_next_id();
            self.version = self.version.saturating_add(1);
        }
        changed
    }

    pub fn render_report(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "OWL 人工反馈台账：{} 条候选（接受 {}，待裁决 {}，驳回 {}，暂缓 {}）",
            self.len(),
            self.accepted_count(),
            self.pending_count(),
            self.rejected_count(),
            self.entries
                .iter()
                .filter(|entry| entry.status == Decision::Deferred)
                .count()
        ));
        let accepted = self.accepted_lessons();
        if !accepted.is_empty() {
            lines.push(format!("已接受公理：{} 条", accepted.len()));
            for lesson in &accepted {
                lines.push(format!("  - {}", render_ontology_lesson_zh(lesson)));
            }
        }
        if self.entries.is_empty() {
            lines.push("暂无候选公理；先运行 owl propose 从当前图谱挖掘。".to_string());
            return lines.join("\n");
        }
        lines.push(String::new());
        for entry in &self.entries {
            lines.push(entry.render());
            lines.push(String::new());
        }
        if lines.last().is_some_and(|line| line.is_empty()) {
            lines.pop();
        }
        lines.join("\n")
    }

    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|error| format!("台账序列化失败：{error}"))
    }

    pub fn save(&self, path: &str) -> Result<(), String> {
        let text = self.to_json()?;
        if let Some(parent) = Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|error| {
                    format!("创建台账目录失败（{}）：{error}", parent.display())
                })?;
            }
        }
        let temporary = format!("{path}.tmp");
        fs::write(&temporary, text.as_bytes())
            .map_err(|error| format!("写入临时台账失败（{temporary}）：{error}"))?;
        fs::rename(&temporary, path).map_err(|error| format!("替换台账失败（{path}）：{error}"))?;
        Ok(())
    }

    pub fn load(path: &str) -> LedgerLoad {
        let text = match fs::read_to_string(path) {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return LedgerLoad::Missing
            }
            Err(error) => {
                return LedgerLoad::Damaged(format!("读取台账失败（{path}）：{error}"));
            }
        };
        match serde_json::from_str::<AnnotationLedger>(&text) {
            Ok(mut ledger) => {
                ledger.refresh_next_id();
                LedgerLoad::Loaded(ledger)
            }
            Err(error) => LedgerLoad::Damaged(format!("解析台账失败（{path}）：{error}")),
        }
    }

    pub fn export_turtle(&self, include_proposed: bool) -> Result<String, String> {
        let mut output = String::new();
        output.push_str("@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n");
        output.push_str("@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n");
        output.push_str("@prefix owl: <http://www.w3.org/2002/07/owl#> .\n");
        output.push_str("@prefix kg: <http://example.org/kg/> .\n");
        output.push_str("@prefix hitl: <http://example.org/kg/hitl#> .\n\n");
        let mut exported = 0usize;
        for entry in &self.entries {
            if !include_proposed && entry.status != Decision::Accepted {
                continue;
            }
            exported += 1;
            for triple in axiom_triples(&entry.effective) {
                output.push_str(&triple);
                output.push('\n');
            }
            let node = format!("_:axiom{}", exported);
            let (source, property, target) = annotation_target(&entry.effective);
            output.push_str(&format!(
                "{node} rdf:type owl:Axiom ;\n  owl:annotatedSource {source} ;\n  owl:annotatedProperty {property} ;\n  owl:annotatedTarget {target} ;\n"
            ));
            let entry_json =
                serde_json::to_string(entry).map_err(|error| format!("公理序列化失败：{error}"))?;
            output.push_str(&format!(
                "  hitl:entry {} ;\n  hitl:candidateId {} ;\n  hitl:status \"{}\" ;\n  hitl:support {} ;\n  hitl:purity {} ;\n  hitl:rationale {} ;\n",
                turtle_literal(&entry_json),
                turtle_literal(&entry.id),
                entry.status.machine_label(),
                entry.proposal.support,
                format_number(entry.proposal.purity),
                turtle_literal(&entry.proposal.rationale)
            ));
            if let Some(review) = entry.last_review() {
                output.push_str(&format!(
                    "  hitl:reviewer {} ;\n  hitl:note {} ;\n",
                    turtle_literal(&review.reviewer),
                    turtle_literal(&review.note)
                ));
            }
            output.push_str("  .\n\n");
        }
        Ok(output)
    }

    pub fn import_turtle(text: &str) -> Result<Self, String> {
        use oxigraph::io::{RdfFormat, RdfParser};
        use oxigraph::model::Term;

        let mut annotations: BTreeMap<String, BTreeMap<String, String>> = BTreeMap::new();
        let parser = RdfParser::from_format(RdfFormat::Turtle);
        for quad in parser.for_reader(text.as_bytes()) {
            let quad = quad.map_err(|error| format!("Turtle 解析失败：{error}"))?;
            let predicate = quad.predicate.as_str();
            let Some(local) = predicate.strip_prefix(HITL_NS) else {
                continue;
            };
            let value = match &quad.object {
                Term::Literal(literal) => literal.value().to_string(),
                term => term.to_string(),
            };
            annotations
                .entry(quad.subject.to_string())
                .or_default()
                .insert(local.to_string(), value);
        }

        let mut entries = Vec::new();
        for properties in annotations.values() {
            if let Some(entry_json) = properties.get("entry") {
                let entry: ReviewEntry = serde_json::from_str(entry_json)
                    .map_err(|error| format!("hitl:entry 解析失败：{error}"))?;
                entries.push(entry);
                continue;
            }
            let Some(axiom_json) = properties.get("axiom") else {
                continue;
            };
            let lesson: OntologyLesson = serde_json::from_str(axiom_json)
                .map_err(|error| format!("hitl:axiom 解析失败：{error}"))?;
            let status = properties
                .get("status")
                .and_then(|text| Decision::parse(text))
                .unwrap_or(Decision::Proposed);
            let support = properties
                .get("support")
                .and_then(|text| text.parse::<usize>().ok())
                .unwrap_or(1);
            let purity = properties
                .get("purity")
                .and_then(|text| text.parse::<f64>().ok())
                .unwrap_or(0.0);
            let id = properties
                .get("candidateId")
                .cloned()
                .filter(|text| !text.trim().is_empty())
                .unwrap_or_else(|| format!("owl-import-{}", entries.len() + 1));
            let key = lesson_key(&lesson);
            entries.push(ReviewEntry {
                id,
                proposal: Proposal {
                    key,
                    lesson: lesson.clone(),
                    rationale: properties.get("rationale").cloned().unwrap_or_default(),
                    evidence: Vec::new(),
                    support,
                    purity,
                    impact: ImpactPreview::default(),
                },
                status,
                effective: lesson,
                history: Vec::new(),
            });
        }

        if entries.is_empty() {
            return Err("未找到 hitl:entry 或 hitl:axiom 注解".to_string());
        }
        entries.sort_by(|left, right| left.id.cmp(&right.id));
        entries.dedup_by(|left, right| left.proposal.key == right.proposal.key);
        let mut ledger = AnnotationLedger {
            version: 1,
            next_id: default_next_id(),
            entries,
        };
        ledger.refresh_next_id();
        Ok(ledger)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProposalOptions {
    pub min_support: usize,
    pub min_purity: u32,
    pub max_candidates: usize,
    pub max_classes: usize,
    pub max_properties: usize,
    pub max_examples: usize,
}

impl Default for ProposalOptions {
    fn default() -> Self {
        Self {
            min_support: 2,
            min_purity: 800,
            max_candidates: 32,
            max_classes: 32,
            max_properties: 32,
            max_examples: 3,
        }
    }
}

fn materially_stronger(old: &Proposal, new: &Proposal) -> bool {
    let support_gain = new.support >= old.support.saturating_add(2)
        && new.support as f64 >= old.support as f64 * 1.25;
    let purity_gain = new.purity >= old.purity + 0.05 && new.support >= old.support;
    support_gain || purity_gain
}

pub fn lesson_kind(lesson: &OntologyLesson) -> &'static str {
    match lesson {
        OntologyLesson::Subclass { .. } => "subclass",
        OntologyLesson::Instance { .. } => "instance",
        OntologyLesson::Disjoint { .. } => "disjoint",
        OntologyLesson::Subproperty { .. } => "subproperty",
        OntologyLesson::Transitive { .. } => "transitive",
        OntologyLesson::Symmetric { .. } => "symmetric",
        OntologyLesson::Inverse { .. } => "inverse",
        OntologyLesson::Domain { .. } => "domain",
        OntologyLesson::Range { .. } => "range",
        OntologyLesson::ClassProperty { .. } => "class_property",
    }
}

pub fn lesson_key(lesson: &OntologyLesson) -> String {
    const SEP: char = '\u{1f}';
    match lesson {
        OntologyLesson::Subclass { sub, sup, .. } => format!("subclass{SEP}{sub}{SEP}{sup}"),
        OntologyLesson::Instance {
            individual, class, ..
        } => format!("instance{SEP}{individual}{SEP}{class}"),
        OntologyLesson::Disjoint { left, right, .. } => {
            let (left, right) = if left <= right {
                (left, right)
            } else {
                (right, left)
            };
            format!("disjoint{SEP}{left}{SEP}{right}")
        }
        OntologyLesson::Subproperty { sub, sup, .. } => {
            format!("subproperty{SEP}{sub}{SEP}{sup}")
        }
        OntologyLesson::Transitive { property, .. } => format!("transitive{SEP}{property}"),
        OntologyLesson::Symmetric { property, .. } => format!("symmetric{SEP}{property}"),
        OntologyLesson::Inverse { left, right, .. } => {
            let (left, right) = if left <= right {
                (left, right)
            } else {
                (right, left)
            };
            format!("inverse{SEP}{left}{SEP}{right}")
        }
        OntologyLesson::Domain {
            property, class, ..
        } => {
            format!("domain{SEP}{property}{SEP}{class}")
        }
        OntologyLesson::Range {
            property, class, ..
        } => {
            format!("range{SEP}{property}{SEP}{class}")
        }
        OntologyLesson::ClassProperty {
            class,
            property,
            object,
            ..
        } => format!("class_property{SEP}{class}{SEP}{property}{SEP}{object}"),
    }
}

fn reaches(count: usize, total: usize, min_purity: u32) -> bool {
    total > 0 && count.saturating_mul(1000) >= total.saturating_mul(min_purity.min(1000) as usize)
}

fn ratio(count: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        count as f64 / total as f64
    }
}

fn most_specific_top(
    counts: &BTreeMap<String, usize>,
    closure: &OntologyClosure,
) -> Vec<(String, usize)> {
    let Some(maximum) = counts.values().copied().max() else {
        return Vec::new();
    };
    let top: Vec<(String, usize)> = counts
        .iter()
        .filter(|(_, count)| **count == maximum)
        .map(|(class, count)| (class.clone(), *count))
        .collect();
    top.iter()
        .filter(|(class, _)| {
            !top.iter().any(|(other, _)| {
                other != class && closure.is_subclass(other.as_str(), class.as_str())
            })
        })
        .cloned()
        .collect()
}

fn preview_against(
    base_reasoner: &OntologyReasoner,
    before: &OntologyClosure,
    lesson: &OntologyLesson,
) -> ImpactPreview {
    let mut reasoner = base_reasoner.clone();
    apply_ontology_lesson(&mut reasoner, lesson);
    let after = reasoner.materialize();
    let before_types: HashSet<(String, String)> = before.instances_snapshot().into_iter().collect();
    let after_types: HashSet<(String, String)> = after.instances_snapshot().into_iter().collect();
    let before_relations: HashSet<(String, String, String)> =
        before.relations_snapshot().into_iter().collect();
    let after_relations: HashSet<(String, String, String)> =
        after.relations_snapshot().into_iter().collect();
    let new_type_assertions = after_types.difference(&before_types).count();
    let new_relations = after_relations.difference(&before_relations).count();
    let new_conflicts = after
        .problems()
        .len()
        .saturating_sub(before.problems().len());
    let score =
        (new_type_assertions as i64) * 2 + (new_relations as i64) * 3 - (new_conflicts as i64) * 20;
    ImpactPreview {
        new_type_assertions,
        new_relations,
        new_conflicts,
        score,
    }
}

fn push_candidate(
    drafts: &mut Vec<Proposal>,
    seen: &mut HashSet<String>,
    base_reasoner: &OntologyReasoner,
    before: &OntologyClosure,
    lesson: OntologyLesson,
    rationale: String,
    evidence: Vec<String>,
    support: usize,
    purity: f64,
    max_examples: usize,
) {
    const MAX_DRAFTS: usize = 512;
    if drafts.len() >= MAX_DRAFTS {
        return;
    }
    let key = lesson_key(&lesson);
    if !seen.insert(key.clone()) {
        return;
    }
    let impact = preview_against(base_reasoner, before, &lesson);
    drafts.push(Proposal {
        key,
        lesson,
        rationale,
        evidence: evidence.into_iter().take(max_examples).collect(),
        support,
        purity,
        impact,
    });
}

pub fn propose_from_observation(
    observation: &GraphObservation,
    base: &OntologyProgram,
    options: &ProposalOptions,
) -> Vec<Proposal> {
    if observation.is_empty() {
        return Vec::new();
    }
    let base_reasoner = observation.reasoner(base);
    let before = base_reasoner.materialize();
    let min_support = options.min_support.max(1);
    let min_purity = options.min_purity.min(1000);
    let max_examples = options.max_examples.max(1);
    let mut drafts = Vec::new();
    let mut seen = HashSet::new();

    let mut by_type: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (individual, class) in before.instances_snapshot() {
        by_type.entry(class).or_default().insert(individual);
    }
    let mut classes: Vec<(String, BTreeSet<String>)> = by_type
        .into_iter()
        .filter(|(_, instances)| instances.len() >= min_support)
        .collect();
    if classes.len() > options.max_classes {
        classes.sort_by(|left, right| {
            right
                .1
                .len()
                .cmp(&left.1.len())
                .then_with(|| left.0.cmp(&right.0))
        });
        classes.truncate(options.max_classes);
    }
    classes.sort_by(|left, right| left.0.cmp(&right.0));
    for (sub, sub_instances) in &classes {
        for (sup, sup_instances) in &classes {
            if sub == sup || before.is_subclass(sub, sup) {
                continue;
            }
            if sub_instances.len() == sup_instances.len() && sub >= sup {
                continue;
            }
            let support = sub_instances.intersection(sup_instances).count();
            if support < min_support || !reaches(support, sub_instances.len(), min_purity) {
                continue;
            }
            let evidence = sub_instances
                .intersection(sup_instances)
                .take(max_examples)
                .map(|individual| format!("{individual} 同时属于 {sub} 与 {sup}"))
                .collect();
            push_candidate(
                &mut drafts,
                &mut seen,
                &base_reasoner,
                &before,
                OntologyLesson::Subclass {
                    sub: sub.clone(),
                    sup: sup.clone(),
                    original: format!("{sub}是一种{sup}"),
                },
                format!(
                    "{sub} 的 {support}/{} 个已知实例也属于 {sup}",
                    sub_instances.len()
                ),
                evidence,
                support,
                ratio(support, sub_instances.len()),
                max_examples,
            );
        }
    }

    let mut by_property: BTreeMap<String, BTreeSet<(String, String)>> = BTreeMap::new();
    for (subject, predicate, object) in &observation.relations {
        if subject != object {
            by_property
                .entry(predicate.clone())
                .or_default()
                .insert((subject.clone(), object.clone()));
        }
    }
    let mut properties: Vec<(String, BTreeSet<(String, String)>)> =
        by_property.into_iter().collect();
    if properties.len() > options.max_properties {
        properties.sort_by(|left, right| {
            right
                .1
                .len()
                .cmp(&left.1.len())
                .then_with(|| left.0.cmp(&right.0))
        });
        properties.truncate(options.max_properties);
    }
    properties.sort_by(|left, right| left.0.cmp(&right.0));

    for (property, edges) in &properties {
        if before.domain_classes(property).is_empty() {
            let subjects: BTreeSet<String> =
                edges.iter().map(|(subject, _)| subject.clone()).collect();
            if subjects.len() >= min_support {
                let mut typed = 0usize;
                let mut counts: BTreeMap<String, usize> = BTreeMap::new();
                for subject in &subjects {
                    let types = before.types_of(subject);
                    if types.is_empty() {
                        continue;
                    }
                    typed += 1;
                    for class in types {
                        *counts.entry(class).or_insert(0) += 1;
                    }
                }
                if reaches(typed, subjects.len(), min_purity) {
                    for (class, count) in most_specific_top(&counts, &before) {
                        if count < min_support || !reaches(count, subjects.len(), min_purity) {
                            continue;
                        }
                        let evidence = edges
                            .iter()
                            .filter(|(subject, _)| before.types_of(subject).contains(&class))
                            .take(max_examples)
                            .map(|(subject, object)| {
                                format!("{subject} ∈ {class}；{property}({subject}, {object})")
                            })
                            .collect();
                        push_candidate(
                            &mut drafts,
                            &mut seen,
                            &base_reasoner,
                            &before,
                            OntologyLesson::Domain {
                                property: property.clone(),
                                class: class.clone(),
                                original: format!("{property}关系的定义域是{class}"),
                            },
                            format!(
                                "{property} 的 {count}/{} 个主语属于 {class}",
                                subjects.len()
                            ),
                            evidence,
                            count,
                            ratio(count, subjects.len()),
                            max_examples,
                        );
                    }
                }
            }
        }
        if before.range_classes(property).is_empty() {
            let objects: BTreeSet<String> =
                edges.iter().map(|(_, object)| object.clone()).collect();
            if objects.len() >= min_support {
                let mut typed = 0usize;
                let mut counts: BTreeMap<String, usize> = BTreeMap::new();
                for object in &objects {
                    let types = before.types_of(object);
                    if types.is_empty() {
                        continue;
                    }
                    typed += 1;
                    for class in types {
                        *counts.entry(class).or_insert(0) += 1;
                    }
                }
                if reaches(typed, objects.len(), min_purity) {
                    for (class, count) in most_specific_top(&counts, &before) {
                        if count < min_support || !reaches(count, objects.len(), min_purity) {
                            continue;
                        }
                        let evidence = edges
                            .iter()
                            .filter(|(_, object)| before.types_of(object).contains(&class))
                            .take(max_examples)
                            .map(|(subject, object)| {
                                format!("{object} ∈ {class}；{property}({subject}, {object})")
                            })
                            .collect();
                        push_candidate(
                            &mut drafts,
                            &mut seen,
                            &base_reasoner,
                            &before,
                            OntologyLesson::Range {
                                property: property.clone(),
                                class: class.clone(),
                                original: format!("{property}关系的值域是{class}"),
                            },
                            format!("{property} 的 {count}/{} 个宾语属于 {class}", objects.len()),
                            evidence,
                            count,
                            ratio(count, objects.len()),
                            max_examples,
                        );
                    }
                }
            }
        }

        let profile = base_reasoner.property_profile(property, &before);
        let unordered: BTreeSet<(String, String)> = edges
            .iter()
            .map(|(left, right)| {
                if left <= right {
                    (left.clone(), right.clone())
                } else {
                    (right.clone(), left.clone())
                }
            })
            .collect();
        if unordered.len() >= min_support && !profile.symmetric {
            let bilateral = unordered
                .iter()
                .filter(|(left, right)| {
                    edges.contains(&(left.clone(), right.clone()))
                        && edges.contains(&(right.clone(), left.clone()))
                })
                .count();
            if bilateral >= min_support && reaches(bilateral, unordered.len(), min_purity) {
                let evidence = unordered
                    .iter()
                    .filter(|(left, right)| {
                        edges.contains(&(left.clone(), right.clone()))
                            && edges.contains(&(right.clone(), left.clone()))
                    })
                    .take(max_examples)
                    .map(|(left, right)| {
                        format!("{property}({left}, {right}) 且 {property}({right}, {left})")
                    })
                    .collect();
                push_candidate(
                    &mut drafts,
                    &mut seen,
                    &base_reasoner,
                    &before,
                    OntologyLesson::Symmetric {
                        property: property.clone(),
                        original: format!("{property}是对称的"),
                    },
                    format!(
                        "{property} 的 {bilateral}/{} 个无序对有双向事实",
                        unordered.len()
                    ),
                    evidence,
                    bilateral,
                    ratio(bilateral, unordered.len()),
                    max_examples,
                );
            }
        }

        if !profile.transitive {
            let mut adjacency: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
            for (subject, object) in edges {
                adjacency
                    .entry(subject.clone())
                    .or_default()
                    .insert(object.clone());
            }
            let mut implied: BTreeSet<(String, String)> = BTreeSet::new();
            let mut witness: BTreeMap<(String, String), String> = BTreeMap::new();
            for (subject, objects) in &adjacency {
                for middle in objects {
                    if let Some(targets) = adjacency.get(middle) {
                        for target in targets {
                            if subject == target || subject == middle || middle == target {
                                continue;
                            }
                            if implied.insert((subject.clone(), target.clone())) {
                                witness.insert((subject.clone(), target.clone()), middle.clone());
                            }
                        }
                    }
                }
            }
            let supported = implied.iter().filter(|edge| edges.contains(*edge)).count();
            if implied.len() >= min_support
                && supported >= min_support
                && reaches(supported, implied.len(), min_purity)
            {
                let evidence = implied
                    .iter()
                    .filter(|edge| edges.contains(*edge))
                    .take(max_examples)
                    .filter_map(|(subject, target)| {
                        witness.get(&(subject.clone(), target.clone())).map(|middle| {
                            format!(
                                "{property}({subject}, {middle}) 与 {property}({middle}, {target}) ⇒ {property}({subject}, {target})"
                            )
                        })
                    })
                    .collect();
                push_candidate(
                    &mut drafts,
                    &mut seen,
                    &base_reasoner,
                    &before,
                    OntologyLesson::Transitive {
                        property: property.clone(),
                        original: format!("{property}关系可以传递"),
                    },
                    format!(
                        "{property} 的 {supported}/{} 条两步路径已有直达事实",
                        implied.len()
                    ),
                    evidence,
                    supported,
                    ratio(supported, implied.len()),
                    max_examples,
                );
            }
        }
    }

    for left_index in 0..properties.len() {
        for right_index in (left_index + 1)..properties.len() {
            let (left, left_edges) = &properties[left_index];
            let (right, right_edges) = &properties[right_index];
            let left_profile = base_reasoner.property_profile(left, &before);
            if !left_profile.inverses.contains(right) {
                let support = left_edges
                    .iter()
                    .filter(|(subject, object)| {
                        right_edges.contains(&(object.clone(), subject.clone()))
                    })
                    .count();
                let total = left_edges.len() + right_edges.len();
                if support >= min_support && reaches(support.saturating_mul(2), total, min_purity) {
                    let evidence = left_edges
                        .iter()
                        .filter(|(subject, object)| {
                            right_edges.contains(&(object.clone(), subject.clone()))
                        })
                        .take(max_examples)
                        .map(|(subject, object)| {
                            format!("{left}({subject}, {object}) 与 {right}({object}, {subject})")
                        })
                        .collect();
                    push_candidate(
                        &mut drafts,
                        &mut seen,
                        &base_reasoner,
                        &before,
                        OntologyLesson::Inverse {
                            left: left.clone(),
                            right: right.clone(),
                            original: format!("{left}和{right}互为逆关系"),
                        },
                        format!("{left} 与 {right} 有 {support} 条互逆事实"),
                        evidence,
                        support,
                        ratio(support.saturating_mul(2), total),
                        max_examples,
                    );
                }
            }

            if left_edges.len() == right_edges.len() {
                continue;
            }
            let (sub, sub_edges, sup, sup_edges) = if left_edges.len() < right_edges.len() {
                (left, left_edges, right, right_edges)
            } else {
                (right, right_edges, left, left_edges)
            };
            if before.is_subproperty(sub, sup) {
                continue;
            }
            let support = sub_edges
                .iter()
                .filter(|edge| sup_edges.contains(*edge))
                .count();
            if support < min_support || !reaches(support, sub_edges.len(), min_purity) {
                continue;
            }
            let evidence = sub_edges
                .iter()
                .filter(|edge| sup_edges.contains(*edge))
                .take(max_examples)
                .map(|(subject, object)| {
                    format!("{sub}({subject}, {object}) 与 {sup}({subject}, {object})")
                })
                .collect();
            push_candidate(
                &mut drafts,
                &mut seen,
                &base_reasoner,
                &before,
                OntologyLesson::Subproperty {
                    sub: sub.clone(),
                    sup: sup.clone(),
                    original: format!("{sub}是{sup}的子关系"),
                },
                format!(
                    "{sub} 的 {support}/{} 条事实也以 {sup} 出现",
                    sub_edges.len()
                ),
                evidence,
                support,
                ratio(support, sub_edges.len()),
                max_examples,
            );
        }
    }

    drafts.sort_by(|left, right| {
        right
            .impact
            .score
            .cmp(&left.impact.score)
            .then_with(|| right.support.cmp(&left.support))
            .then_with(|| {
                right
                    .purity
                    .partial_cmp(&left.purity)
                    .unwrap_or(Ordering::Equal)
            })
            .then_with(|| left.key.cmp(&right.key))
    });
    drafts.truncate(options.max_candidates.max(1));
    drafts
}

fn format_number(value: f64) -> String {
    let mut text = format!("{value:.4}");
    while text.contains('.') && text.ends_with('0') {
        text.pop();
    }
    if text.ends_with('.') {
        text.pop();
    }
    text
}

fn iri_local(name: &str) -> String {
    let mut output = String::new();
    for character in name.chars() {
        if character <= ' '
            || matches!(
                character,
                '<' | '>' | '"' | '{' | '}' | '|' | '^' | '`' | '\\'
            )
        {
            let mut buffer = [0u8; 4];
            for byte in character.encode_utf8(&mut buffer).as_bytes() {
                output.push_str(&format!("%{byte:02X}"));
            }
        } else {
            output.push(character);
        }
    }
    output
}

fn kg(name: &str) -> String {
    format!("<{}{}>", BASE_NS, iri_local(name))
}

fn iri(iri: &str) -> String {
    format!("<{iri}>")
}

fn turtle_literal(text: &str) -> String {
    let mut output = String::with_capacity(text.len() + 2);
    output.push('"');
    for character in text.chars() {
        match character {
            '\\' => output.push_str("\\\\"),
            '"' => output.push_str("\\\""),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            other => output.push(other),
        }
    }
    output.push('"');
    output
}

fn axiom_triples(lesson: &OntologyLesson) -> Vec<String> {
    match lesson {
        OntologyLesson::Subclass { sub, sup, .. } => vec![format!(
            "{} {} {} .",
            kg(sub),
            iri(RDFS_SUBCLASS_IRI),
            kg(sup)
        )],
        OntologyLesson::Instance {
            individual, class, ..
        } => vec![format!(
            "{} {} {} .",
            kg(individual),
            iri(RDF_TYPE_IRI),
            kg(class)
        )],
        OntologyLesson::Disjoint { left, right, .. } => vec![format!(
            "{} {} {} .",
            kg(left),
            iri(OWL_DISJOINT_IRI),
            kg(right)
        )],
        OntologyLesson::Subproperty { sub, sup, .. } => vec![format!(
            "{} {} {} .",
            kg(sub),
            iri(RDFS_SUBPROPERTY_IRI),
            kg(sup)
        )],
        OntologyLesson::Transitive { property, .. } => vec![format!(
            "{} {} {} .",
            kg(property),
            iri(RDF_TYPE_IRI),
            iri(OWL_TRANSITIVE_IRI)
        )],
        OntologyLesson::Symmetric { property, .. } => vec![format!(
            "{} {} {} .",
            kg(property),
            iri(RDF_TYPE_IRI),
            iri(OWL_SYMMETRIC_IRI)
        )],
        OntologyLesson::Inverse { left, right, .. } => vec![format!(
            "{} {} {} .",
            kg(left),
            iri(OWL_INVERSE_IRI),
            kg(right)
        )],
        OntologyLesson::Domain {
            property, class, ..
        } => vec![format!(
            "{} {} {} .",
            kg(property),
            iri(RDFS_DOMAIN_IRI),
            kg(class)
        )],
        OntologyLesson::Range {
            property, class, ..
        } => vec![format!(
            "{} {} {} .",
            kg(property),
            iri(RDFS_RANGE_IRI),
            kg(class)
        )],
        OntologyLesson::ClassProperty {
            class,
            property,
            object,
            ..
        } => vec![
            format!(
                "{} {} {} .",
                kg(class),
                iri(&format!("{HITL_NS}classProperty")),
                kg(property)
            ),
            format!(
                "{} {} {} .",
                kg(class),
                iri(&format!("{HITL_NS}propertyValue")),
                turtle_literal(object)
            ),
        ],
    }
}

fn annotation_target(lesson: &OntologyLesson) -> (String, String, String) {
    match lesson {
        OntologyLesson::Subclass { sub, sup, .. } => (kg(sub), iri(RDFS_SUBCLASS_IRI), kg(sup)),
        OntologyLesson::Instance {
            individual, class, ..
        } => (kg(individual), iri(RDF_TYPE_IRI), kg(class)),
        OntologyLesson::Disjoint { left, right, .. } => {
            (kg(left), iri(OWL_DISJOINT_IRI), kg(right))
        }
        OntologyLesson::Subproperty { sub, sup, .. } => {
            (kg(sub), iri(RDFS_SUBPROPERTY_IRI), kg(sup))
        }
        OntologyLesson::Transitive { property, .. } => {
            (kg(property), iri(RDF_TYPE_IRI), iri(OWL_TRANSITIVE_IRI))
        }
        OntologyLesson::Symmetric { property, .. } => {
            (kg(property), iri(RDF_TYPE_IRI), iri(OWL_SYMMETRIC_IRI))
        }
        OntologyLesson::Inverse { left, right, .. } => (kg(left), iri(OWL_INVERSE_IRI), kg(right)),
        OntologyLesson::Domain {
            property, class, ..
        } => (kg(property), iri(RDFS_DOMAIN_IRI), kg(class)),
        OntologyLesson::Range {
            property, class, ..
        } => (kg(property), iri(RDFS_RANGE_IRI), kg(class)),
        OntologyLesson::ClassProperty {
            class, property, ..
        } => (
            kg(class),
            iri(&format!("{HITL_NS}classProperty")),
            kg(property),
        ),
    }
}

pub fn ontology_program_with_annotations(
    config_program: &OntologyProgram,
    ledger_path: &str,
) -> (OntologyProgram, usize) {
    let mut program = config_program.clone();
    let accepted = match AnnotationLedger::load(ledger_path) {
        LedgerLoad::Loaded(ledger) => ledger.accepted_lessons(),
        LedgerLoad::Missing | LedgerLoad::Damaged(_) => Vec::new(),
    };
    let count = accepted.len();
    if !accepted.is_empty() {
        program.merge(&OntologyProgram::from_lessons(accepted));
    }
    (program, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation(
        instances: &[(&str, &str)],
        relations: &[(&str, &str, &str)],
    ) -> GraphObservation {
        GraphObservation::new(
            instances
                .iter()
                .map(|(individual, class)| (individual.to_string(), class.to_string()))
                .collect(),
            relations
                .iter()
                .map(|(subject, predicate, object)| {
                    (
                        subject.to_string(),
                        predicate.to_string(),
                        object.to_string(),
                    )
                })
                .collect(),
        )
    }

    #[test]
    fn test_propose_subclass_with_impact() {
        let graph = observation(
            &[
                ("学生1", "学生"),
                ("学生2", "学生"),
                ("学生3", "学生"),
                ("学生4", "学生"),
                ("学生5", "学生"),
                ("学生1", "人物"),
                ("学生2", "人物"),
                ("学生3", "人物"),
                ("学生4", "人物"),
                ("人物6", "人物"),
                ("人物7", "人物"),
            ],
            &[],
        );
        let proposals = propose_from_observation(
            &graph,
            &OntologyProgram::default(),
            &ProposalOptions::default(),
        );
        let subclass = proposals
            .iter()
            .find(|proposal| {
                matches!(
                    &proposal.lesson,
                    OntologyLesson::Subclass { sub, sup, .. } if sub == "学生" && sup == "人物"
                )
            })
            .expect("应提出 学生 ⊑ 人物");
        assert_eq!(subclass.support, 4);
        assert_eq!(subclass.impact.new_type_assertions, 1);

        let mut ledger = AnnotationLedger::default();
        let outcome = ledger.propose(proposals);
        assert_eq!(outcome.added.len(), 1);
        ledger
            .review(
                &outcome.added[0],
                ReviewAction::Accept,
                "tester",
                "抽查通过",
            )
            .unwrap();
        let program = ledger.accepted_program();
        let reasoner = graph.reasoner(&program);
        let closure = reasoner.materialize();
        assert!(closure.is_instance("学生5", "人物"));
        assert_eq!(ledger.accepted_count(), 1);
    }

    #[test]
    fn test_propose_structural_axioms() {
        let graph = observation(
            &[
                ("甲", "人物"),
                ("乙", "人物"),
                ("丙", "人物"),
                ("丁", "人物"),
                ("戊", "人物"),
                ("己", "人物"),
                ("庚", "人物"),
                ("辛", "人物"),
            ],
            &[
                ("甲", "合作", "乙"),
                ("乙", "合作", "甲"),
                ("丙", "合作", "丁"),
                ("丁", "合作", "丙"),
                ("戊", "合作", "己"),
                ("己", "合作", "戊"),
                ("庚", "合作", "辛"),
                ("辛", "合作", "庚"),
                ("甲", "合作", "丙"),
                ("甲", "祖先", "乙"),
                ("乙", "祖先", "丙"),
                ("甲", "祖先", "丙"),
                ("丙", "祖先", "丁"),
                ("乙", "祖先", "丁"),
                ("甲", "祖先", "丁"),
                ("甲", "师傅", "乙"),
                ("丙", "师傅", "丁"),
                ("乙", "徒弟", "甲"),
                ("丁", "徒弟", "丙"),
                ("甲", "任教", "戊"),
                ("乙", "任教", "己"),
                ("甲", "任职", "戊"),
                ("乙", "任职", "己"),
                ("丙", "任职", "庚"),
            ],
        );
        let proposals = propose_from_observation(
            &graph,
            &OntologyProgram::default(),
            &ProposalOptions::default(),
        );
        assert!(proposals.iter().any(|proposal| matches!(
            &proposal.lesson,
            OntologyLesson::Symmetric { property, .. } if property == "合作"
        )));
        assert!(proposals.iter().any(|proposal| matches!(
            &proposal.lesson,
            OntologyLesson::Transitive { property, .. } if property == "祖先"
        )));
        assert!(proposals.iter().any(|proposal| matches!(
            &proposal.lesson,
            OntologyLesson::Inverse { left, right, .. }
                if (left == "师傅" && right == "徒弟") || (left == "徒弟" && right == "师傅")
        )));
        assert!(proposals.iter().any(|proposal| matches!(
            &proposal.lesson,
            OntologyLesson::Subproperty { sub, sup, .. } if sub == "任教" && sup == "任职"
        )));
    }

    #[test]
    fn test_rejection_memory_and_reopen() {
        let graph = observation(
            &[
                ("学生1", "学生"),
                ("学生2", "学生"),
                ("学生3", "学生"),
                ("学生1", "人物"),
                ("学生2", "人物"),
                ("学生3", "人物"),
                ("人物4", "人物"),
                ("人物5", "人物"),
            ],
            &[],
        );
        let mut ledger = AnnotationLedger::default();
        let proposals = propose_from_observation(
            &graph,
            &OntologyProgram::default(),
            &ProposalOptions::default(),
        );
        let outcome = ledger.propose(proposals);
        assert!(!outcome.added.is_empty());
        let id = outcome.added[0].clone();
        ledger
            .review(&id, ReviewAction::Reject, "tester", "证据不足")
            .unwrap();
        assert_eq!(ledger.rejected_count(), 1);

        let proposals_again = propose_from_observation(
            &graph,
            &OntologyProgram::default(),
            &ProposalOptions::default(),
        );
        let suppressed = ledger.propose(proposals_again);
        assert!(!suppressed.suppressed.is_empty());
        assert!(suppressed.added.is_empty());
        assert!(suppressed.updated.is_empty());

        let mut stronger = ledger.entries[0].proposal.clone();
        stronger.support += 5;
        let reopened = ledger.propose(vec![stronger]);
        assert_eq!(reopened.updated.len(), 1);
        assert_eq!(ledger.find(&id).unwrap().status, Decision::Proposed);
    }

    #[test]
    fn test_revision_overrides_proposal() {
        let graph = observation(
            &[
                ("学生1", "学生"),
                ("学生2", "学生"),
                ("学生3", "学生"),
                ("学生1", "人物"),
                ("学生2", "人物"),
                ("学生3", "人物"),
                ("人物4", "人物"),
                ("人物5", "人物"),
            ],
            &[],
        );
        let mut ledger = AnnotationLedger::default();
        let proposals = propose_from_observation(
            &graph,
            &OntologyProgram::default(),
            &ProposalOptions::default(),
        );
        let outcome = ledger.propose(proposals);
        let id = outcome.added[0].clone();
        let revised = OntologyLesson::Subclass {
            sub: "学生".to_string(),
            sup: "人类".to_string(),
            original: "学生是一种人类".to_string(),
        };
        ledger
            .review(
                &id,
                ReviewAction::Revise {
                    lesson: revised.clone(),
                },
                "tester",
                "改成更精确的类名",
            )
            .unwrap();
        assert_eq!(ledger.find(&id).unwrap().effective, revised);
        let program = ledger.accepted_program();
        assert!(program.lessons.contains(&revised));
    }

    #[test]
    fn test_turtle_roundtrip_preserves_review() {
        let graph = observation(
            &[
                ("学生1", "学生"),
                ("学生2", "学生"),
                ("学生3", "学生"),
                ("学生1", "人物"),
                ("学生2", "人物"),
                ("学生3", "人物"),
                ("人物4", "人物"),
                ("人物5", "人物"),
            ],
            &[],
        );
        let mut ledger = AnnotationLedger::default();
        let proposals = propose_from_observation(
            &graph,
            &OntologyProgram::default(),
            &ProposalOptions::default(),
        );
        let outcome = ledger.propose(proposals);
        let id = outcome.added[0].clone();
        ledger
            .review(&id, ReviewAction::Accept, "tester", "接受并导出")
            .unwrap();
        let turtle = ledger.export_turtle(false).unwrap();
        assert!(turtle.contains("owl:Axiom"));
        assert!(turtle.contains("hitl:entry"));
        let imported = AnnotationLedger::import_turtle(&turtle).unwrap();
        assert_eq!(imported.entries, ledger.entries);
        assert_eq!(imported.accepted_count(), 1);
        assert_eq!(imported.accepted_lessons(), ledger.accepted_lessons());
    }

    #[test]
    fn test_merge_entries_prefers_imported_decision() {
        let mut ledger = AnnotationLedger::default();
        let proposal = Proposal {
            key: lesson_key(&OntologyLesson::Symmetric {
                property: "合作".to_string(),
                original: "合作是对称的".to_string(),
            }),
            lesson: OntologyLesson::Symmetric {
                property: "合作".to_string(),
                original: "合作是对称的".to_string(),
            },
            rationale: String::new(),
            evidence: Vec::new(),
            support: 3,
            purity: 1.0,
            impact: ImpactPreview::default(),
        };
        ledger.propose(vec![proposal]);
        let mut imported = AnnotationLedger::default();
        imported.entries = ledger.entries.clone();
        imported.entries[0].status = Decision::Rejected;
        imported.entries[0].effective = ledger.entries[0].effective.clone();
        assert_eq!(ledger.merge_entries(&imported, true), 1);
        assert_eq!(ledger.rejected_count(), 1);
    }
}

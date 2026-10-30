use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::context_image::{canonical_json, fnv1a_hex};
use crate::predicate_registry::PredicateRegistry;
use crate::prolog_engine::PrologFact;
use crate::type_system::{TypeSystem, TypeVerdict};
use crate::EntityTypeConfig;

pub const SHAPE_DIALECT: &str = "knowprolog-shacl/1";
pub const MAX_RENDERED_VIOLATIONS: usize = 50;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Info,
    Warning,
    Violation,
}

impl Severity {
    pub fn label(self) -> &'static str {
        match self {
            Severity::Info => "提示",
            Severity::Warning => "警告",
            Severity::Violation => "违规",
        }
    }

    pub fn is_violation(self) -> bool {
        self == Severity::Violation
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstraintKind {
    MinCount,
    MaxCount,
    Datatype,
    Class,
    SubjectClass,
    Closed,
}

impl ConstraintKind {
    pub fn code(self) -> &'static str {
        match self {
            ConstraintKind::MinCount => "min_count",
            ConstraintKind::MaxCount => "max_count",
            ConstraintKind::Datatype => "datatype",
            ConstraintKind::Class => "class",
            ConstraintKind::SubjectClass => "subject_class",
            ConstraintKind::Closed => "closed",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            ConstraintKind::MinCount => "最少数量",
            ConstraintKind::MaxCount => "最多数量",
            ConstraintKind::Datatype => "数据类型",
            ConstraintKind::Class => "值域类型",
            ConstraintKind::SubjectClass => "定义域类型",
            ConstraintKind::Closed => "封闭形状",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PropertyShape {
    pub path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_count: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_count: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub datatype: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub class: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub subject_class: Vec<String>,
    pub severity: Severity,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeShape {
    pub target_class: String,
    #[serde(default)]
    pub closed: bool,
    #[serde(default)]
    pub properties: Vec<PropertyShape>,
    pub severity: Severity,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PredicateAlias {
    pub canonical: String,
    pub inverted: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShapeSet {
    #[serde(default)]
    pub nodes: Vec<NodeShape>,
    #[serde(default)]
    pub relations: Vec<PropertyShape>,
    #[serde(default)]
    pub aliases: BTreeMap<String, PredicateAlias>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum RepairAction {
    AskProperty {
        entity: String,
        property: String,
    },
    AskEntity {
        surface: String,
    },
    DiscloseConflict {
        entity: String,
        property: String,
        values: Vec<String>,
    },
    CorrectDatatype {
        entity: String,
        property: String,
        expected: String,
    },
    RetypeEntity {
        entity: String,
        expected: Vec<String>,
        actual: Vec<String>,
    },
    RemoveEdge {
        subject: String,
        predicate: String,
        object: String,
    },
    ReviewClosedShape {
        entity: String,
        unexpected: Vec<String>,
    },
    AdmitPredicate {
        predicate: String,
    },
}

impl RepairAction {
    pub fn label(&self) -> String {
        match self {
            RepairAction::AskProperty { entity, property } => {
                format!("追问「{}」的「{}」", entity, property)
            }
            RepairAction::AskEntity { surface } => format!("追问实体「{}」的身份", surface),
            RepairAction::DiscloseConflict {
                entity,
                property,
                values,
            } => format!(
                "披露「{}」的「{}」有多个取值：{}",
                entity,
                property,
                values.join("、")
            ),
            RepairAction::CorrectDatatype {
                entity,
                property,
                expected,
            } => format!("修正「{}」的「{}」为 {} 类型", entity, property, expected),
            RepairAction::RetypeEntity {
                entity, expected, ..
            } => format!("确认「{}」的类型为 {}", entity, expected.join(" 或 ")),
            RepairAction::RemoveEdge {
                subject,
                predicate,
                object,
            } => format!("删除不合形状的边 {}-[{}]->{}", subject, predicate, object),
            RepairAction::ReviewClosedShape { entity, unexpected } => format!(
                "复核「{}」上未声明的路径：{}",
                entity,
                unexpected.join("、")
            ),
            RepairAction::AdmitPredicate { predicate } => {
                format!("把谓词「{}」纳入准入流程", predicate)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShapeViolation {
    pub shape_id: String,
    pub focus_node: String,
    pub path: String,
    pub constraint: ConstraintKind,
    pub severity: Severity,
    pub message: String,
    pub expected: Vec<String>,
    pub actual: Vec<String>,
    pub repair: RepairAction,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShapeReport {
    pub dialect: String,
    pub shape_digest: String,
    pub data_digest: String,
    pub shape_count: usize,
    pub focus_nodes: usize,
    pub conforms: bool,
    pub violations: Vec<ShapeViolation>,
    pub digest: String,
}

impl ShapeReport {
    pub fn violation_count(&self) -> usize {
        self.violations
            .iter()
            .filter(|violation| violation.severity == Severity::Violation)
            .count()
    }

    pub fn warning_count(&self) -> usize {
        self.violations
            .iter()
            .filter(|violation| violation.severity == Severity::Warning)
            .count()
    }

    pub fn info_count(&self) -> usize {
        self.violations
            .iter()
            .filter(|violation| violation.severity == Severity::Info)
            .count()
    }

    pub fn by_constraint(&self) -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::new();
        for violation in &self.violations {
            *counts
                .entry(violation.constraint.code().to_string())
                .or_insert(0) += 1;
        }
        counts
    }

    pub fn render_text(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!("形状校验（{}）", self.dialect));
        lines.push(format!(
            "形状：{} 条；焦点实体：{} 个；形状摘要：{}；数据摘要：{}",
            self.shape_count, self.focus_nodes, self.shape_digest, self.data_digest
        ));
        if self.conforms {
            lines.push(format!(
                "判定：通过（违规 0；警告 {}；提示 {}）",
                self.warning_count(),
                self.info_count()
            ));
        } else {
            lines.push(format!(
                "判定：不通过（违规 {}；警告 {}；提示 {}）",
                self.violation_count(),
                self.warning_count(),
                self.info_count()
            ));
        }
        let by_constraint = self.by_constraint();
        if !by_constraint.is_empty() {
            let summary: Vec<String> = by_constraint
                .iter()
                .map(|(kind, count)| format!("{}×{}", kind, count))
                .collect();
            lines.push(format!("按约束：{}", summary.join("；")));
        }
        for violation in self.violations.iter().take(MAX_RENDERED_VIOLATIONS) {
            lines.push(format!(
                "- [{}] {} · {} · {}：{}；期望：{}；实际：{}（修复：{}）",
                violation.severity.label(),
                violation.focus_node,
                violation.path,
                violation.constraint.label(),
                violation.message,
                violation.expected.join("、"),
                violation.actual.join("、"),
                violation.repair.label()
            ));
        }
        if self.violations.len() > MAX_RENDERED_VIOLATIONS {
            lines.push(format!(
                "……另有 {} 条未显示（使用 --json 查看完整报告）",
                self.violations.len() - MAX_RENDERED_VIOLATIONS
            ));
        }
        lines.push(format!("报告摘要：{}", self.digest));
        lines.join("\n")
    }
}

impl ShapeSet {
    pub fn from_schema(
        entity_types: &[EntityTypeConfig],
        registry: &PredicateRegistry,
        types: &TypeSystem,
    ) -> Self {
        let mut nodes = Vec::new();
        for entity_type in entity_types {
            let target_class = types.canonical(&entity_type.label);
            if target_class.is_empty() {
                continue;
            }
            let mut properties: BTreeMap<String, PropertyShape> = BTreeMap::new();
            for property in &entity_type.properties {
                let path = registry.resolve(&property.name).canonical;
                if path.is_empty() {
                    continue;
                }
                let datatype = normalize_datatype(&property.data_type);
                if !property.required && datatype.is_none() {
                    continue;
                }
                let severity = if property.required {
                    Severity::Violation
                } else {
                    Severity::Warning
                };
                let shape = properties
                    .entry(path.clone())
                    .or_insert_with(|| PropertyShape {
                        path: path.clone(),
                        min_count: None,
                        max_count: None,
                        datatype: None,
                        class: Vec::new(),
                        subject_class: Vec::new(),
                        severity,
                        message: format!("属性「{}」的形状约束", path),
                    });
                if property.required {
                    shape.min_count = Some(1);
                    shape.severity = Severity::Violation;
                    shape.message =
                        format!("类型「{}」要求必备属性「{}」", entity_type.label, path);
                }
                if shape.datatype.is_none() {
                    shape.datatype = datatype;
                }
            }
            nodes.push(NodeShape {
                target_class,
                closed: false,
                properties: properties.into_values().collect(),
                severity: Severity::Violation,
                message: format!("实体类型「{}」的形状", entity_type.label),
            });
        }
        nodes.sort_by(|left, right| left.target_class.cmp(&right.target_class));

        let mut relations = Vec::new();
        for canonical in registry.canonical_names() {
            let Some(profile) = registry.profile(canonical) else {
                continue;
            };
            let subject_class: Vec<String> = canonical_types(&profile.domain, types);
            let class: Vec<String> = canonical_types(&profile.range, types);
            if !profile.functional && subject_class.is_empty() && class.is_empty() {
                continue;
            }
            relations.push(PropertyShape {
                path: profile.canonical.clone(),
                min_count: None,
                max_count: profile.functional.then_some(1),
                datatype: None,
                class,
                subject_class,
                severity: Severity::Violation,
                message: if profile.functional {
                    format!("功能谓词「{}」至多允许一个取值", profile.canonical)
                } else {
                    format!("谓词「{}」的定义域/值域形状", profile.canonical)
                },
            });
        }
        relations.sort_by(|left, right| left.path.cmp(&right.path));

        let mut aliases = BTreeMap::new();
        for canonical in registry.canonical_names() {
            aliases.insert(
                canonical.to_string(),
                PredicateAlias {
                    canonical: canonical.to_string(),
                    inverted: false,
                },
            );
        }
        for (surface, canonical, inverted) in registry.alias_pairs() {
            aliases.insert(
                surface,
                PredicateAlias {
                    canonical,
                    inverted,
                },
            );
        }

        Self {
            nodes,
            relations,
            aliases,
        }
    }

    pub fn digest(&self) -> String {
        let value = serde_json::to_value(self).unwrap_or(serde_json::Value::Null);
        fnv1a_hex(canonical_json(&value).as_bytes())
    }

    /// 用属性本体桥覆盖/补充形状约束：
    /// - 值类型以属性档案为准（例如 `integer` 改为 `quantity`）；
    /// - 单值属性补上 `max_count = 1`；
    /// - 字符串属性移除过窄的数值类型约束。
    pub fn apply_attribute_profiles(
        &mut self,
        attributes: &crate::attribute_bridge::AttributeRegistry,
    ) {
        if attributes.is_empty() {
            return;
        }
        for shape in self.nodes.iter_mut().flat_map(|node| node.properties.iter_mut()) {
            apply_attribute_shape(shape, attributes);
        }
        for shape in &mut self.relations {
            apply_attribute_shape(shape, attributes);
        }
    }

    pub fn resolve_predicate(&self, predicate: &str) -> (String, bool) {
        match self.aliases.get(predicate.trim()) {
            Some(alias) => (alias.canonical.clone(), alias.inverted),
            None => (predicate.trim().to_string(), false),
        }
    }

    pub fn validate(
        &self,
        facts: &[PrologFact],
        registry: &PredicateRegistry,
        types: &TypeSystem,
    ) -> ShapeReport {
        let snapshot = Snapshot::build(facts, registry, types);
        let mut violations = Vec::new();

        for shape in &self.nodes {
            for node in &snapshot.nodes {
                if !snapshot.matches_target(node, &shape.target_class, types) {
                    continue;
                }
                if shape.closed {
                    let allowed: BTreeSet<&str> = shape
                        .properties
                        .iter()
                        .map(|property| property.path.as_str())
                        .collect();
                    let unexpected: Vec<String> = snapshot
                        .paths(node)
                        .into_iter()
                        .filter(|path| !allowed.contains(path.as_str()))
                        .collect();
                    if !unexpected.is_empty() {
                        violations.push(ShapeViolation {
                            shape_id: format!("node:{}", shape.target_class),
                            focus_node: node.clone(),
                            path: "*".to_string(),
                            constraint: ConstraintKind::Closed,
                            severity: shape.severity,
                            message: shape.message.clone(),
                            expected: allowed.iter().map(|path| path.to_string()).collect(),
                            actual: unexpected.clone(),
                            repair: RepairAction::ReviewClosedShape {
                                entity: node.clone(),
                                unexpected,
                            },
                        });
                    }
                }
                for property in &shape.properties {
                    let values = snapshot.values(node, &property.path);
                    if let Some(min_count) = property.min_count {
                        if values.len() < min_count {
                            violations.push(ShapeViolation {
                                shape_id: format!("node:{}", shape.target_class),
                                focus_node: node.clone(),
                                path: property.path.clone(),
                                constraint: ConstraintKind::MinCount,
                                severity: property.severity,
                                message: property.message.clone(),
                                expected: vec![format!("至少 {} 个", min_count)],
                                actual: vec![format!("{} 个", values.len())],
                                repair: RepairAction::AskProperty {
                                    entity: node.clone(),
                                    property: property.path.clone(),
                                },
                            });
                        }
                    }
                    if let Some(max_count) = property.max_count {
                        if values.len() > max_count {
                            violations.push(ShapeViolation {
                                shape_id: format!("node:{}", shape.target_class),
                                focus_node: node.clone(),
                                path: property.path.clone(),
                                constraint: ConstraintKind::MaxCount,
                                severity: property.severity,
                                message: property.message.clone(),
                                expected: vec![format!("至多 {} 个", max_count)],
                                actual: values.clone(),
                                repair: RepairAction::DiscloseConflict {
                                    entity: node.clone(),
                                    property: property.path.clone(),
                                    values,
                                },
                            });
                        }
                    }
                    if let Some(datatype) = &property.datatype {
                        if let Some(attribute_values) = snapshot
                            .attributes
                            .get(node)
                            .and_then(|attributes| attributes.get(&property.path))
                        {
                            let invalid: Vec<String> = attribute_values
                                .iter()
                                .filter(|value| !datatype_conforms(datatype, value))
                                .cloned()
                                .collect();
                            if !invalid.is_empty() {
                                violations.push(ShapeViolation {
                                    shape_id: format!("node:{}", shape.target_class),
                                    focus_node: node.clone(),
                                    path: property.path.clone(),
                                    constraint: ConstraintKind::Datatype,
                                    severity: property.severity,
                                    message: format!(
                                        "属性「{}」的取值不符合 {} 类型",
                                        property.path, datatype
                                    ),
                                    expected: vec![datatype.clone()],
                                    actual: invalid.clone(),
                                    repair: RepairAction::CorrectDatatype {
                                        entity: node.clone(),
                                        property: property.path.clone(),
                                        expected: datatype.clone(),
                                    },
                                });
                            }
                        }
                    }
                    if !property.class.is_empty() {
                        let allowed: BTreeSet<String> = property.class.iter().cloned().collect();
                        let objects = snapshot
                            .relations
                            .get(&(node.clone(), property.path.clone()))
                            .cloned()
                            .unwrap_or_default();
                        for object in objects {
                            let actual = snapshot.types_of(&object);
                            if types.sets_admit(&actual, &allowed) == TypeVerdict::Disjoint {
                                let actual: Vec<String> = actual.into_iter().collect();
                                violations.push(ShapeViolation {
                                    shape_id: format!("node:{}", shape.target_class),
                                    focus_node: object.clone(),
                                    path: property.path.clone(),
                                    constraint: ConstraintKind::Class,
                                    severity: property.severity,
                                    message: format!(
                                        "「{}」的「{}」取值「{}」不在声明的值域内",
                                        node, property.path, object
                                    ),
                                    expected: property.class.clone(),
                                    actual: actual.clone(),
                                    repair: RepairAction::RetypeEntity {
                                        entity: object.clone(),
                                        expected: property.class.clone(),
                                        actual,
                                    },
                                });
                            }
                        }
                    }
                }
            }
        }

        for shape in &self.relations {
            let mut subjects: BTreeSet<String> = BTreeSet::new();
            for (subject, attributes) in &snapshot.attributes {
                if attributes.contains_key(&shape.path) {
                    subjects.insert(subject.clone());
                }
            }
            for (subject, path) in snapshot.relations.keys() {
                if path == &shape.path {
                    subjects.insert(subject.clone());
                }
            }
            for subject in subjects {
                let values = snapshot.values(&subject, &shape.path);
                if let Some(max_count) = shape.max_count {
                    if values.len() > max_count {
                        violations.push(ShapeViolation {
                            shape_id: format!("relation:{}", shape.path),
                            focus_node: subject.clone(),
                            path: shape.path.clone(),
                            constraint: ConstraintKind::MaxCount,
                            severity: shape.severity,
                            message: shape.message.clone(),
                            expected: vec![format!("至多 {} 个", max_count)],
                            actual: values.clone(),
                            repair: RepairAction::DiscloseConflict {
                                entity: subject.clone(),
                                property: shape.path.clone(),
                                values,
                            },
                        });
                    }
                }
                if !shape.subject_class.is_empty() {
                    let actual = snapshot.types_of(&subject);
                    let allowed: BTreeSet<String> = shape.subject_class.iter().cloned().collect();
                    if types.sets_admit(&actual, &allowed) == TypeVerdict::Disjoint {
                        let actual: Vec<String> = actual.into_iter().collect();
                        violations.push(ShapeViolation {
                            shape_id: format!("relation:{}", shape.path),
                            focus_node: subject.clone(),
                            path: shape.path.clone(),
                            constraint: ConstraintKind::SubjectClass,
                            severity: shape.severity,
                            message: format!(
                                "「{}」的主语类型不在谓词「{}」的定义域内",
                                subject, shape.path
                            ),
                            expected: shape.subject_class.clone(),
                            actual: actual.clone(),
                            repair: RepairAction::RetypeEntity {
                                entity: subject.clone(),
                                expected: shape.subject_class.clone(),
                                actual,
                            },
                        });
                    }
                }
                if !shape.class.is_empty() {
                    let allowed: BTreeSet<String> = shape.class.iter().cloned().collect();
                    let objects = snapshot
                        .relations
                        .get(&(subject.clone(), shape.path.clone()))
                        .cloned()
                        .unwrap_or_default();
                    for object in objects {
                        let actual = snapshot.types_of(&object);
                        if types.sets_admit(&actual, &allowed) == TypeVerdict::Disjoint {
                            let actual: Vec<String> = actual.into_iter().collect();
                            violations.push(ShapeViolation {
                                shape_id: format!("relation:{}", shape.path),
                                focus_node: object.clone(),
                                path: shape.path.clone(),
                                constraint: ConstraintKind::Class,
                                severity: shape.severity,
                                message: format!(
                                    "「{}」的「{}」取值「{}」不在声明的值域内",
                                    subject, shape.path, object
                                ),
                                expected: shape.class.clone(),
                                actual: actual.clone(),
                                repair: RepairAction::RetypeEntity {
                                    entity: object.clone(),
                                    expected: shape.class.clone(),
                                    actual,
                                },
                            });
                        }
                    }
                }
            }
        }

        violations.sort_by(|left, right| {
            left.focus_node
                .cmp(&right.focus_node)
                .then_with(|| left.path.cmp(&right.path))
                .then_with(|| left.constraint.cmp(&right.constraint))
                .then_with(|| right.severity.cmp(&left.severity))
                .then_with(|| left.message.cmp(&right.message))
                .then_with(|| left.actual.cmp(&right.actual))
        });
        violations.dedup_by(|left, right| {
            left.focus_node == right.focus_node
                && left.path == right.path
                && left.constraint == right.constraint
                && left.severity == right.severity
                && left.expected == right.expected
                && left.actual == right.actual
        });

        let shape_digest = self.digest();
        let data_digest = snapshot.digest();
        let conforms = violations
            .iter()
            .all(|violation| !violation.severity.is_violation());
        let digest = {
            let value = serde_json::json!({
                "dialect": SHAPE_DIALECT,
                "shape_digest": &shape_digest,
                "data_digest": &data_digest,
                "violations": &violations,
            });
            fnv1a_hex(canonical_json(&value).as_bytes())
        };
        ShapeReport {
            dialect: SHAPE_DIALECT.to_string(),
            shape_digest,
            data_digest,
            shape_count: self.nodes.len()
                + self.relations.len()
                + self
                    .nodes
                    .iter()
                    .map(|shape| shape.properties.len())
                    .sum::<usize>(),
            focus_nodes: snapshot.nodes.len(),
            conforms,
            violations,
            digest,
        }
    }

    pub fn decide_candidates(
        &self,
        candidates: &[InterpretationCandidate],
        facts: &[PrologFact],
        registry: &PredicateRegistry,
        types: &TypeSystem,
    ) -> CandidateDecision {
        let snapshot = Snapshot::build(facts, registry, types);
        let relation_shapes: BTreeMap<&str, &PropertyShape> = self
            .relations
            .iter()
            .map(|shape| (shape.path.as_str(), shape))
            .collect();

        let mut verdicts = Vec::with_capacity(candidates.len());
        for (index, candidate) in candidates.iter().enumerate() {
            let claim = self.normalize_claim(&candidate.claim);
            let (status, detail, repair) =
                self.judge_candidate(&claim, &snapshot, registry, types, &relation_shapes);
            verdicts.push(CandidateVerdict {
                index,
                status,
                detail,
                repair,
            });
        }

        let mut selected: Option<(usize, i64)> = None;
        for verdict in &verdicts {
            if verdict.status != CandidateStatus::Accepted {
                continue;
            }
            let score = candidates[verdict.index].score;
            match selected {
                Some((best_index, best_score))
                    if score < best_score
                        || (score == best_score && verdict.index > best_index) => {}
                _ => selected = Some((verdict.index, score)),
            }
        }
        let fallback_repair = verdicts
            .iter()
            .filter(|verdict| verdict.status != CandidateStatus::Accepted)
            .max_by(|left, right| {
                let left_score = candidates[left.index].score;
                let right_score = candidates[right.index].score;
                left_score
                    .cmp(&right_score)
                    .then_with(|| right.index.cmp(&left.index))
            })
            .and_then(|verdict| verdict.repair.clone());

        CandidateDecision {
            selected: selected.map(|(index, _)| index),
            accepted: verdicts
                .iter()
                .filter(|verdict| verdict.status == CandidateStatus::Accepted)
                .count(),
            verdicts,
            fallback_repair,
        }
    }

    fn judge_candidate(
        &self,
        claim: &Claim,
        snapshot: &Snapshot,
        registry: &PredicateRegistry,
        types: &TypeSystem,
        relation_shapes: &BTreeMap<&str, &PropertyShape>,
    ) -> (CandidateStatus, String, Option<RepairAction>) {
        let path_known = registry.is_known(&claim.predicate)
            || relation_shapes.contains_key(claim.predicate.as_str())
            || snapshot.path_exists(&claim.predicate);
        if !path_known {
            return (
                CandidateStatus::UnknownPredicate,
                format!("谓词「{}」尚未登记", claim.predicate),
                Some(RepairAction::AdmitPredicate {
                    predicate: claim.predicate.clone(),
                }),
            );
        }
        if !snapshot.nodes.contains(&claim.subject) {
            return (
                CandidateStatus::UnknownEntity,
                format!("主语「{}」不在图谱中", claim.subject),
                Some(RepairAction::AskEntity {
                    surface: claim.subject.clone(),
                }),
            );
        }
        if let ClaimObject::Entity(object) = &claim.object {
            if !snapshot.nodes.contains(object) {
                return (
                    CandidateStatus::UnknownEntity,
                    format!("宾语「{}」不在图谱中", object),
                    Some(RepairAction::AskEntity {
                        surface: object.clone(),
                    }),
                );
            }
        }

        let Some(shape) = relation_shapes.get(claim.predicate.as_str()) else {
            return (CandidateStatus::Accepted, "通过形状约束".to_string(), None);
        };
        if !shape.subject_class.is_empty() {
            let actual = snapshot.types_of(&claim.subject);
            let allowed: BTreeSet<String> = shape.subject_class.iter().cloned().collect();
            if types.sets_admit(&actual, &allowed) == TypeVerdict::Disjoint {
                return (
                    CandidateStatus::TypeMismatch,
                    format!("主语「{}」的类型不在定义域内", claim.subject),
                    Some(RepairAction::RetypeEntity {
                        entity: claim.subject.clone(),
                        expected: shape.subject_class.clone(),
                        actual: actual.into_iter().collect(),
                    }),
                );
            }
        }
        if let ClaimObject::Entity(object) = &claim.object {
            if !shape.class.is_empty() {
                let actual = snapshot.types_of(object);
                let allowed: BTreeSet<String> = shape.class.iter().cloned().collect();
                if types.sets_admit(&actual, &allowed) == TypeVerdict::Disjoint {
                    return (
                        CandidateStatus::TypeMismatch,
                        format!("宾语「{}」的类型不在值域内", object),
                        Some(RepairAction::RetypeEntity {
                            entity: object.clone(),
                            expected: shape.class.clone(),
                            actual: actual.into_iter().collect(),
                        }),
                    );
                }
            }
        }
        if shape.max_count == Some(1) {
            let object_text = claim.object.text();
            let existing = snapshot.values(&claim.subject, &claim.predicate);
            if !existing.is_empty() && !existing.iter().any(|value| value == &object_text) {
                return (
                    CandidateStatus::ValueConflict,
                    format!(
                        "功能谓词「{}」已有取值，候选「{}」与之冲突",
                        claim.predicate, object_text
                    ),
                    Some(RepairAction::DiscloseConflict {
                        entity: claim.subject.clone(),
                        property: claim.predicate.clone(),
                        values: existing,
                    }),
                );
            }
        }
        (CandidateStatus::Accepted, "通过形状约束".to_string(), None)
    }

    fn normalize_claim(&self, claim: &Claim) -> Claim {
        let (canonical, inverted) = self.resolve_predicate(&claim.predicate);
        if !inverted {
            return Claim {
                subject: claim.subject.clone(),
                predicate: canonical,
                object: claim.object.clone(),
            };
        }
        match &claim.object {
            ClaimObject::Entity(object) => Claim {
                subject: object.clone(),
                predicate: canonical,
                object: ClaimObject::Entity(claim.subject.clone()),
            },
            ClaimObject::Literal(_) => Claim {
                subject: claim.subject.clone(),
                predicate: canonical,
                object: claim.object.clone(),
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum ClaimObject {
    Entity(String),
    Literal(String),
}

impl ClaimObject {
    pub fn text(&self) -> String {
        match self {
            ClaimObject::Entity(value) | ClaimObject::Literal(value) => value.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Claim {
    pub subject: String,
    pub predicate: String,
    pub object: ClaimObject,
}

impl Claim {
    pub fn entity(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: ClaimObject::Entity(object.into()),
        }
    }

    pub fn literal(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: ClaimObject::Literal(value.into()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GroundingIssue {
    UnknownSubject {
        subject: String,
    },
    UnknownObject {
        object: String,
    },
    ValueMismatch {
        path: String,
        expected: Vec<String>,
        actual: String,
    },
}

impl GroundingIssue {
    pub fn label(&self) -> String {
        match self {
            GroundingIssue::UnknownSubject { subject } => {
                format!("主语「{}」不在图谱中", subject)
            }
            GroundingIssue::UnknownObject { object } => {
                format!("宾语「{}」不在图谱中", object)
            }
            GroundingIssue::ValueMismatch {
                path,
                expected,
                actual,
            } => format!(
                "「{}」没有取值「{}」（图谱中为：{}）",
                path,
                actual,
                if expected.is_empty() {
                    "（无）".to_string()
                } else {
                    expected.join("、")
                }
            ),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroundingRejection {
    pub claim: Claim,
    pub issue: GroundingIssue,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroundingReport {
    pub total: usize,
    pub grounded: usize,
    pub conforms: bool,
    pub rejections: Vec<GroundingRejection>,
    pub digest: String,
}

impl GroundingReport {
    pub fn render_text(&self) -> String {
        let mut lines = vec![format!(
            "回答接地：{}/{} 条陈述可回指图谱事实；判定：{}",
            self.grounded,
            self.total,
            if self.conforms { "通过" } else { "不通过" }
        )];
        for rejection in &self.rejections {
            lines.push(format!(
                "- {}-[{}]->{}：{}",
                rejection.claim.subject,
                rejection.claim.predicate,
                rejection.claim.object.text(),
                rejection.issue.label()
            ));
        }
        lines.push(format!("报告摘要：{}", self.digest));
        lines.join("\n")
    }
}

pub struct AnswerGate<'a> {
    shapes: &'a ShapeSet,
    snapshot: Snapshot,
}

impl<'a> AnswerGate<'a> {
    pub fn new(
        facts: &[PrologFact],
        shapes: &'a ShapeSet,
        registry: &PredicateRegistry,
        types: &TypeSystem,
    ) -> Self {
        Self {
            shapes,
            snapshot: Snapshot::build(facts, registry, types),
        }
    }

    pub fn values_of(&self, subject: &str, predicate: &str) -> Vec<String> {
        let (canonical, inverted) = self.shapes.resolve_predicate(predicate);
        if !inverted {
            return self.snapshot.values(subject, &canonical);
        }
        let mut values: Vec<String> = self
            .snapshot
            .relations
            .iter()
            .filter(|((_, path), objects)| {
                path == &canonical && objects.iter().any(|object| object == subject)
            })
            .map(|((node, _), _)| node.clone())
            .collect();
        values.sort();
        values.dedup();
        values
    }

    pub fn check(&self, claim: &Claim) -> Result<(), GroundingIssue> {
        let claim = self.shapes.normalize_claim(claim);
        if !self.snapshot.nodes.contains(&claim.subject) {
            return Err(GroundingIssue::UnknownSubject {
                subject: claim.subject,
            });
        }
        if let ClaimObject::Entity(object) = &claim.object {
            if !self.snapshot.nodes.contains(object) {
                return Err(GroundingIssue::UnknownObject {
                    object: object.clone(),
                });
            }
        }
        let object = claim.object.text();
        let values = self.snapshot.values(&claim.subject, &claim.predicate);
        if values.iter().any(|value| value == &object) {
            Ok(())
        } else {
            Err(GroundingIssue::ValueMismatch {
                path: claim.predicate,
                expected: values,
                actual: object,
            })
        }
    }

    pub fn audit(&self, claims: &[Claim]) -> GroundingReport {
        let mut rejections = Vec::new();
        for claim in claims {
            if let Err(issue) = self.check(claim) {
                rejections.push(GroundingRejection {
                    claim: claim.clone(),
                    issue,
                });
            }
        }
        rejections.sort_by(|left, right| {
            (
                &left.claim.subject,
                &left.claim.predicate,
                left.claim.object.text(),
            )
                .cmp(&(
                    &right.claim.subject,
                    &right.claim.predicate,
                    right.claim.object.text(),
                ))
        });
        let grounded = claims.len().saturating_sub(rejections.len());
        let digest = {
            let value = serde_json::json!({
                "total": claims.len(),
                "grounded": grounded,
                "rejections": &rejections,
            });
            fnv1a_hex(canonical_json(&value).as_bytes())
        };
        GroundingReport {
            total: claims.len(),
            grounded,
            conforms: rejections.is_empty(),
            rejections,
            digest,
        }
    }

    pub fn keep_grounded<'c>(&self, claims: &'c [Claim]) -> Vec<&'c Claim> {
        claims
            .iter()
            .filter(|claim| self.check(claim).is_ok())
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InterpretationCandidate {
    pub claim: Claim,
    pub score: i64,
    #[serde(default)]
    pub segmentation: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateStatus {
    Accepted,
    UnknownEntity,
    UnknownPredicate,
    TypeMismatch,
    ValueConflict,
}

impl CandidateStatus {
    pub fn label(self) -> &'static str {
        match self {
            CandidateStatus::Accepted => "通过",
            CandidateStatus::UnknownEntity => "实体未知",
            CandidateStatus::UnknownPredicate => "谓词未知",
            CandidateStatus::TypeMismatch => "类型冲突",
            CandidateStatus::ValueConflict => "取值冲突",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateVerdict {
    pub index: usize,
    pub status: CandidateStatus,
    pub detail: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repair: Option<RepairAction>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateDecision {
    pub selected: Option<usize>,
    pub accepted: usize,
    pub verdicts: Vec<CandidateVerdict>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fallback_repair: Option<RepairAction>,
}

impl CandidateDecision {
    pub fn render_text(&self, candidates: &[InterpretationCandidate]) -> String {
        let mut lines = vec![format!(
            "候选解码：{} 个候选，{} 个通过；选定：{}",
            self.verdicts.len(),
            self.accepted,
            self.selected
                .and_then(|index| candidates.get(index))
                .map(|candidate| candidate.claim.predicate.clone())
                .unwrap_or_else(|| "（无）".to_string())
        )];
        for verdict in &self.verdicts {
            let candidate = &candidates[verdict.index];
            lines.push(format!(
                "- #{} 分数 {} · {}({}-[{}]->{})：{}",
                verdict.index,
                candidate.score,
                verdict.status.label(),
                candidate.claim.subject,
                candidate.claim.predicate,
                candidate.claim.object.text(),
                verdict.detail
            ));
        }
        if let Some(repair) = &self.fallback_repair {
            lines.push(format!("修复建议：{}", repair.label()));
        }
        lines.join("\n")
    }
}

fn canonical_types(values: &BTreeSet<String>, types: &TypeSystem) -> Vec<String> {
    values
        .iter()
        .map(|value| types.canonical(value))
        .filter(|value| !value.is_empty())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn normalize_datatype(raw: &str) -> Option<String> {
    let lower = raw.trim().to_ascii_lowercase();
    let canonical = match lower.as_str() {
        "string" | "text" => "string",
        "integer" | "int" => "integer",
        "float" | "double" | "number" => "number",
        "boolean" | "bool" => "boolean",
        "quantity" | "数量" => "quantity",
        "date" => "date",
        "datetime" => "datetime",
        "time" => "time",
        _ => return None,
    };
    Some(canonical.to_string())
}

fn apply_attribute_shape(
    shape: &mut PropertyShape,
    attributes: &crate::attribute_bridge::AttributeRegistry,
) {
    let Some(profile) = attributes.resolve(&shape.path) else {
        return;
    };
    use crate::attribute_bridge::AttributeValueType;
    shape.datatype = match profile.value_type {
        AttributeValueType::Quantity => Some("quantity".to_string()),
        AttributeValueType::Integer => Some("integer".to_string()),
        AttributeValueType::Number => Some("number".to_string()),
        AttributeValueType::Boolean => Some("boolean".to_string()),
        AttributeValueType::Date => Some("date".to_string()),
        AttributeValueType::Datetime => Some("datetime".to_string()),
        AttributeValueType::String | AttributeValueType::Any => None,
    };
    if profile.functional {
        shape.max_count = Some(1);
    }
}

pub fn datatype_conforms(expected: &str, value: &str) -> bool {
    let trimmed = value.trim();
    match expected.trim().to_ascii_lowercase().as_str() {
        "string" | "text" => true,
        "integer" | "int" => trimmed.parse::<i64>().is_ok(),
        "number" | "float" | "double" => trimmed.parse::<f64>().is_ok(),
        "boolean" | "bool" => matches!(
            trimmed.to_ascii_lowercase().as_str(),
            "true" | "false" | "1" | "0" | "yes" | "no" | "是" | "否"
        ),
        "quantity" => crate::quantity::parse_quantity(trimmed).is_some(),
        "date" => parse_date(trimmed).is_some(),
        "datetime" => {
            let (date, time) = trimmed
                .split_once('T')
                .or_else(|| trimmed.split_once(' '))
                .unwrap_or(("", ""));
            parse_date(date).is_some() && parse_time(time).is_some()
        }
        "time" => parse_time(trimmed).is_some(),
        _ => true,
    }
}

fn parse_date(text: &str) -> Option<(i32, u32, u32)> {
    let normalized = text.replace('/', "-");
    let mut parts = normalized.split('-');
    let year: i32 = parts.next()?.parse().ok()?;
    let month: u32 = parts.next()?.parse().ok()?;
    let day: u32 = parts.next()?.parse().ok()?;
    if parts.next().is_some() || year <= 0 || !(1..=12).contains(&month) || day == 0 || day > 31 {
        return None;
    }
    Some((year, month, day))
}

fn parse_time(text: &str) -> Option<(u32, u32, u32)> {
    let mut parts = text.split(':');
    let hour: u32 = parts.next()?.parse().ok()?;
    let minute: u32 = parts.next()?.parse().ok()?;
    let second: u32 = match parts.next() {
        Some(value) => value.parse().ok()?,
        None => 0,
    };
    if parts.next().is_some() || hour > 23 || minute > 59 || second > 59 {
        return None;
    }
    Some((hour, minute, second))
}

#[derive(Debug, Clone)]
struct Snapshot {
    nodes: BTreeSet<String>,
    types: BTreeMap<String, BTreeSet<String>>,
    attributes: BTreeMap<String, BTreeMap<String, Vec<String>>>,
    relations: BTreeMap<(String, String), Vec<String>>,
}

impl Snapshot {
    fn build(facts: &[PrologFact], registry: &PredicateRegistry, types: &TypeSystem) -> Self {
        let mut snapshot = Self {
            nodes: BTreeSet::new(),
            types: BTreeMap::new(),
            attributes: BTreeMap::new(),
            relations: BTreeMap::new(),
        };
        for fact in facts {
            match fact {
                PrologFact::Entity {
                    name,
                    entity_type,
                    properties,
                } => {
                    let name = name.trim();
                    if name.is_empty() {
                        continue;
                    }
                    snapshot.nodes.insert(name.to_string());
                    let canonical_type = types.canonical(entity_type);
                    if !canonical_type.is_empty() {
                        snapshot
                            .types
                            .entry(name.to_string())
                            .or_default()
                            .insert(canonical_type);
                    }
                    for (key, value) in properties {
                        let path = registry.resolve(key).canonical;
                        if path.is_empty() || value.trim().is_empty() {
                            continue;
                        }
                        let slot = snapshot
                            .attributes
                            .entry(name.to_string())
                            .or_default()
                            .entry(path)
                            .or_default();
                        slot.push(value.trim().to_string());
                    }
                }
                PrologFact::Relation { from, rel_type, to } => {
                    let from = from.trim();
                    let to = to.trim();
                    if from.is_empty() || to.is_empty() {
                        continue;
                    }
                    let resolution = registry.resolve(rel_type);
                    let (subject, object) = if resolution.inverted {
                        (to, from)
                    } else {
                        (from, to)
                    };
                    let path = resolution.canonical.trim();
                    if path.is_empty() {
                        continue;
                    }
                    snapshot.nodes.insert(subject.to_string());
                    snapshot.nodes.insert(object.to_string());
                    snapshot
                        .relations
                        .entry((subject.to_string(), path.to_string()))
                        .or_default()
                        .push(object.to_string());
                }
            }
        }
        for values in snapshot
            .attributes
            .values_mut()
            .flat_map(|map| map.values_mut())
        {
            values.sort();
            values.dedup();
        }
        for values in snapshot.relations.values_mut() {
            values.sort();
            values.dedup();
        }
        snapshot
    }

    fn values(&self, subject: &str, path: &str) -> Vec<String> {
        let mut values = Vec::new();
        if let Some(attributes) = self.attributes.get(subject) {
            if let Some(attribute_values) = attributes.get(path) {
                values.extend(attribute_values.iter().cloned());
            }
        }
        if let Some(relation_values) = self.relations.get(&(subject.to_string(), path.to_string()))
        {
            values.extend(relation_values.iter().cloned());
        }
        values.sort();
        values.dedup();
        values
    }

    fn paths(&self, subject: &str) -> BTreeSet<String> {
        let mut paths = BTreeSet::new();
        if let Some(attributes) = self.attributes.get(subject) {
            paths.extend(attributes.keys().cloned());
        }
        for (node, path) in self.relations.keys() {
            if node == subject {
                paths.insert(path.clone());
            }
        }
        paths
    }

    fn path_exists(&self, path: &str) -> bool {
        self.attributes
            .values()
            .any(|attributes| attributes.contains_key(path))
            || self
                .relations
                .keys()
                .any(|(_, relation_path)| relation_path == path)
    }

    fn types_of(&self, node: &str) -> BTreeSet<String> {
        self.types.get(node).cloned().unwrap_or_default()
    }

    fn matches_target(&self, node: &str, target: &str, types: &TypeSystem) -> bool {
        self.types_of(node)
            .iter()
            .any(|actual| types.is_subclass(actual, target))
    }

    fn digest(&self) -> String {
        let relations: Vec<serde_json::Value> = self
            .relations
            .iter()
            .map(|((subject, predicate), objects)| {
                serde_json::json!({
                    "subject": subject,
                    "predicate": predicate,
                    "objects": objects,
                })
            })
            .collect();
        let value = serde_json::json!({
            "nodes": &self.nodes,
            "types": &self.types,
            "attributes": &self.attributes,
            "relations": relations,
        });
        fnv1a_hex(canonical_json(&value).as_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::predicate_registry::PredicateProfileConfig;
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

    fn entity_type(label: &str, properties: &[(&str, &str, bool)]) -> EntityTypeConfig {
        EntityTypeConfig {
            label: label.to_string(),
            examples: Vec::new(),
            properties: properties
                .iter()
                .map(|(name, datatype, required)| PropertyConfig {
                    name: name.to_string(),
                    data_type: datatype.to_string(),
                    required: *required,
                    extraction_patterns: Vec::new(),
                })
                .collect(),
        }
    }

    fn person_types() -> TypeSystem {
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

    #[test]
    fn compiles_node_and_relation_shapes_from_schema() {
        let mut registry = PredicateRegistry::new();
        registry.declare(&profile("师傅", false, &["人物"], &["人物"]));
        registry.declare(&profile("年龄", true, &[], &[]));
        let shapes = ShapeSet::from_schema(
            &[entity_type(
                "Person",
                &[("name", "string", true), ("age", "integer", false)],
            )],
            &registry,
            &person_types(),
        );

        let node = shapes
            .nodes
            .iter()
            .find(|node| node.target_class == "人物")
            .expect("节点形状应使用类型别名归一");
        assert!(node
            .properties
            .iter()
            .any(|property| property.path == "name" && property.min_count == Some(1)));
        assert!(node.properties.iter().any(|property| {
            property.path == "age" && property.datatype.as_deref() == Some("integer")
        }));
        let master = shapes
            .relations
            .iter()
            .find(|shape| shape.path == "师傅")
            .expect("关系形状应来自谓词画像");
        assert_eq!(master.class, vec!["人物".to_string()]);
        assert_eq!(master.subject_class, vec!["人物".to_string()]);
        assert!(shapes
            .relations
            .iter()
            .any(|shape| shape.path == "年龄" && shape.max_count == Some(1)));
        assert_eq!(shapes.digest(), shapes.digest());
    }

    #[test]
    fn reports_missing_required_property_with_repair() {
        let shapes = ShapeSet::from_schema(
            &[entity_type("Person", &[("name", "string", true)])],
            &PredicateRegistry::new(),
            &TypeSystem::new(),
        );
        let facts = vec![entity("张三", "Person", &[])];
        let report = shapes.validate(&facts, &PredicateRegistry::new(), &TypeSystem::new());

        assert!(!report.conforms);
        let violation = report
            .violations
            .iter()
            .find(|violation| violation.constraint == ConstraintKind::MinCount)
            .expect("缺少必备属性应被检出");
        assert_eq!(violation.focus_node, "张三");
        assert_eq!(violation.path, "name");
        assert!(matches!(violation.repair, RepairAction::AskProperty { .. }));
    }

    #[test]
    fn reports_datatype_violation() {
        let shapes = ShapeSet::from_schema(
            &[entity_type(
                "Person",
                &[("name", "string", true), ("age", "integer", false)],
            )],
            &PredicateRegistry::new(),
            &TypeSystem::new(),
        );
        let facts = vec![entity(
            "张三",
            "Person",
            &[("name", "张三"), ("age", "十八")],
        )];
        let report = shapes.validate(&facts, &PredicateRegistry::new(), &TypeSystem::new());

        let violation = report
            .violations
            .iter()
            .find(|violation| violation.constraint == ConstraintKind::Datatype)
            .expect("非整数值应被检出");
        assert_eq!(violation.path, "age");
        assert!(violation.actual.contains(&"十八".to_string()));
    }

    #[test]
    fn reports_functional_conflict() {
        let mut registry = PredicateRegistry::new();
        registry.declare(&profile("年龄", true, &[], &[]));
        let shapes = ShapeSet::from_schema(&[], &registry, &TypeSystem::new());
        let facts = vec![
            relation("张三", "年龄", "18"),
            relation("张三", "年龄", "19"),
        ];
        let report = shapes.validate(&facts, &registry, &TypeSystem::new());

        let violation = report
            .violations
            .iter()
            .find(|violation| violation.constraint == ConstraintKind::MaxCount)
            .expect("功能谓词多值应被检出");
        assert_eq!(violation.focus_node, "张三");
        assert!(matches!(
            violation.repair,
            RepairAction::DiscloseConflict { .. }
        ));
    }

    #[test]
    fn reports_domain_and_range_type_conflicts() {
        let mut registry = PredicateRegistry::new();
        registry.declare(&profile("师傅", false, &["人物"], &["人物"]));
        let shapes = ShapeSet::from_schema(&[], &registry, &person_types());
        let facts = vec![
            entity("张三", "Person", &[]),
            entity("李四", "地点", &[]),
            relation("张三", "师傅", "李四"),
        ];
        let report = shapes.validate(&facts, &registry, &person_types());

        assert!(report
            .violations
            .iter()
            .any(|violation| violation.constraint == ConstraintKind::Class
                && violation.focus_node == "李四"));
    }

    #[test]
    fn report_digest_is_fact_order_independent() {
        let mut registry = PredicateRegistry::new();
        registry.declare(&profile("年龄", true, &[], &[]));
        let shapes = ShapeSet::from_schema(&[], &registry, &TypeSystem::new());
        let forward = vec![
            entity("张三", "Person", &[("name", "张三")]),
            relation("张三", "年龄", "18"),
            relation("张三", "年龄", "19"),
        ];
        let mut reverse = forward.clone();
        reverse.reverse();

        let first = shapes.validate(&forward, &registry, &TypeSystem::new());
        let second = shapes.validate(&reverse, &registry, &TypeSystem::new());
        assert_eq!(first.data_digest, second.data_digest);
        assert_eq!(first.digest, second.digest);
    }

    #[test]
    fn closed_shape_flags_undeclared_paths() {
        let mut shapes = ShapeSet::from_schema(
            &[entity_type("Person", &[("name", "string", true)])],
            &PredicateRegistry::new(),
            &TypeSystem::new(),
        );
        shapes.nodes[0].closed = true;
        let facts = vec![entity("张三", "Person", &[("name", "张三"), ("age", "18")])];
        let report = shapes.validate(&facts, &PredicateRegistry::new(), &TypeSystem::new());

        assert!(report
            .violations
            .iter()
            .any(|violation| violation.constraint == ConstraintKind::Closed
                && violation.actual.contains(&"age".to_string())));
    }

    #[test]
    fn attribute_profiles_refine_shape_datatypes_and_cardinality() {
        use crate::attribute_bridge::{
            AttributeBridgeConfig, AttributeProfileConfig, AttributeRegistry, AttributeValueType,
        };
        let mut shapes = ShapeSet::from_schema(
            &[entity_type(
                "Person",
                &[("name", "string", true), ("height", "integer", false)],
            )],
            &PredicateRegistry::new(),
            &TypeSystem::new(),
        );
        let attributes = AttributeRegistry::from_config(&AttributeBridgeConfig {
            enabled: true,
            profiles: vec![AttributeProfileConfig {
                name: "height".to_string(),
                aliases: vec!["身高".to_string()],
                domain: vec!["Person".to_string()],
                value_type: AttributeValueType::Quantity,
                dimension: Some("length".to_string()),
                unit: Some("厘米".to_string()),
                functional: true,
                ..AttributeProfileConfig::default()
            }],
        });
        shapes.apply_attribute_profiles(&attributes);
        let height = shapes.nodes[0]
            .properties
            .iter()
            .find(|property| property.path == "height")
            .unwrap();
        assert_eq!(height.datatype.as_deref(), Some("quantity"));
        assert_eq!(height.max_count, Some(1));

        let facts = vec![entity(
            "姚明",
            "Person",
            &[("name", "姚明"), ("height", "226厘米")],
        )];
        let report = shapes.validate(&facts, &PredicateRegistry::new(), &TypeSystem::new());
        assert!(
            !report.violations.iter().any(|violation| {
                violation.path == "height" && violation.constraint == ConstraintKind::Datatype
            }),
            "{:?}",
            report.violations
        );
        assert!(datatype_conforms("quantity", "226厘米"));
        assert!(datatype_conforms("quantity", "约2.26米"));
        assert!(!datatype_conforms("quantity", "很高"));
    }

    #[test]
    fn answer_gate_accepts_only_grounded_claims() {
        let mut registry = PredicateRegistry::new();
        registry.declare(&profile("师傅", false, &[], &[]));
        let shapes = ShapeSet::from_schema(&[], &registry, &TypeSystem::new());
        let facts = vec![
            entity("张三", "Person", &[("name", "张三"), ("age", "18")]),
            entity("李四", "Person", &[("name", "李四")]),
            entity("王五", "Person", &[("name", "王五")]),
            relation("张三", "师傅", "李四"),
        ];
        let gate = AnswerGate::new(&facts, &shapes, &registry, &TypeSystem::new());
        let claims = vec![
            Claim::entity("张三", "师傅", "李四"),
            Claim::literal("张三", "age", "18"),
            Claim::entity("张三", "师傅", "王五"),
            Claim::literal("张三", "age", "99"),
            Claim::entity("赵六", "师傅", "李四"),
        ];
        let report = gate.audit(&claims);

        assert_eq!(report.total, 5);
        assert_eq!(report.grounded, 2);
        assert!(!report.conforms);
        assert_eq!(gate.keep_grounded(&claims).len(), 2);
        assert_eq!(gate.values_of("张三", "师傅"), vec!["李四".to_string()]);
    }

    #[test]
    fn answer_gate_respects_inverse_predicates() {
        let mut registry = PredicateRegistry::new();
        registry.declare(&profile("师傅", false, &[], &[]));
        registry.register_inverse("师傅", "徒弟", "测试");
        let shapes = ShapeSet::from_schema(&[], &registry, &TypeSystem::new());
        let facts = vec![
            entity("张三", "Person", &[("name", "张三")]),
            entity("李四", "Person", &[("name", "李四")]),
            relation("张三", "师傅", "李四"),
        ];
        let gate = AnswerGate::new(&facts, &shapes, &registry, &TypeSystem::new());

        assert_eq!(gate.values_of("李四", "徒弟"), vec!["张三".to_string()]);
        assert!(gate.check(&Claim::entity("李四", "徒弟", "张三")).is_ok());
        assert!(gate.check(&Claim::entity("张三", "师傅", "李四")).is_ok());
    }

    #[test]
    fn candidate_decoding_selects_best_conforming_interpretation() {
        let mut registry = PredicateRegistry::new();
        registry.declare(&profile("师傅", false, &["人物"], &["人物"]));
        let shapes = ShapeSet::from_schema(&[], &registry, &person_types());
        let facts = vec![
            entity("张三", "Person", &[]),
            entity("李四", "Person", &[]),
            entity("王五", "地点", &[]),
            relation("张三", "师傅", "李四"),
        ];
        let candidates = vec![
            InterpretationCandidate {
                claim: Claim::entity("张三", "未登记谓词", "李四"),
                score: 100,
                segmentation: vec!["张三".to_string(), "未登记谓词".to_string()],
            },
            InterpretationCandidate {
                claim: Claim::entity("张三", "师傅", "李四"),
                score: 90,
                segmentation: vec!["张三".to_string(), "师傅".to_string()],
            },
            InterpretationCandidate {
                claim: Claim::entity("张三", "师傅", "王五"),
                score: 80,
                segmentation: vec!["张三".to_string(), "师傅".to_string()],
            },
        ];
        let decision = shapes.decide_candidates(&candidates, &facts, &registry, &person_types());

        assert_eq!(decision.accepted, 1);
        assert_eq!(decision.selected, Some(1));
        assert_eq!(
            decision.verdicts[0].status,
            CandidateStatus::UnknownPredicate
        );
        assert_eq!(decision.verdicts[2].status, CandidateStatus::TypeMismatch);
    }

    #[test]
    fn datatype_check_supports_common_types() {
        assert!(datatype_conforms("integer", "18"));
        assert!(!datatype_conforms("integer", "十八"));
        assert!(datatype_conforms("number", "1.5"));
        assert!(datatype_conforms("boolean", "是"));
        assert!(datatype_conforms("date", "2026-10-19"));
        assert!(!datatype_conforms("date", "2026-13-19"));
        assert!(datatype_conforms("datetime", "2026-10-19T12:30:00"));
        assert!(datatype_conforms("time", "12:30"));
        assert!(!datatype_conforms("time", "25:00"));
    }
}

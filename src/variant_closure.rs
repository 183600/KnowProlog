use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::context_image::{canonical_json, fnv1a_hex};
use crate::entity_registry::{fold_surface, EntityRegistry, EntityStatus};
use crate::synonym_graph::{SynonymGraph, SynonymKind, SynonymStatus};
use crate::temporal::{TimePoint, Validity};
use crate::type_system::{TypeSystem, TypeVerdict};

pub const VARIANT_CLOSURE_DIALECT: &str = "knowprolog-variant-closure/1";
pub const VARIANT_CLOSURE_VERSION: u32 = 1;
pub const DEFAULT_VARIANT_MAX_DEPTH: usize = 6;
pub const DEFAULT_VARIANT_MAX_NODES: usize = 256;
pub const DEFAULT_VARIANT_MAX_EXPANSIONS: usize = 4_096;
pub const DEFAULT_VARIANT_MAX_CANDIDATES: usize = 8;
pub const DEFAULT_VARIANT_MIN_CONFIDENCE: f64 = 0.35;
pub const DEFAULT_VARIANT_AMBIGUITY_MARGIN: f64 = 0.08;
pub const DEFAULT_VARIANT_HOP_DECAY: f64 = 0.92;

const HONORIFIC_SUFFIXES: [&str; 12] = [
    "先生", "女士", "小姐", "老师", "教授", "博士", "医生", "律师", "导演", "总统", "主席", "同志",
];
const ORGANIZATION_SUFFIXES: [&str; 14] = [
    "股份有限公司", "有限责任公司", "有限公司", "公司", "集团", "大学", "学院", "研究院",
    "研究所", "委员会", "银行", "医院", "博物馆", "图书馆",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VariantKind {
    Alias,
    Abbreviation,
    FullName,
    Initialism,
    Transliteration,
    Translation,
    FormerName,
    Honorific,
    Nickname,
    Elision,
    Typo,
    Metonymy,
}

impl VariantKind {
    pub const fn label(self) -> &'static str {
        match self {
            VariantKind::Alias => "别名",
            VariantKind::Abbreviation => "简称",
            VariantKind::FullName => "全称",
            VariantKind::Initialism => "首字母缩写",
            VariantKind::Transliteration => "音译",
            VariantKind::Translation => "翻译",
            VariantKind::FormerName => "曾用名",
            VariantKind::Honorific => "尊称",
            VariantKind::Nickname => "昵称",
            VariantKind::Elision => "词形省略",
            VariantKind::Typo => "错别字",
            VariantKind::Metonymy => "借代",
        }
    }

    pub fn infer(surface: &str, canonical: &str) -> Self {
        let surface = surface.trim();
        let canonical = canonical.trim();
        if surface.is_empty() || canonical.is_empty() || surface == canonical {
            return VariantKind::Alias;
        }
        if HONORIFIC_SUFFIXES
            .iter()
            .any(|suffix| surface.ends_with(suffix))
        {
            return VariantKind::Honorific;
        }
        let ascii_initialism = surface.chars().count() >= 2
            && surface.chars().all(|ch| ch.is_ascii_alphanumeric())
            && surface.chars().any(|ch| ch.is_ascii_alphabetic())
            && surface.chars().all(|ch| !ch.is_ascii_lowercase());
        if ascii_initialism {
            return VariantKind::Initialism;
        }
        if is_ascii_words(surface) && contain_cjk(canonical) {
            return VariantKind::Translation;
        }
        if contain_cjk(surface) && is_ascii_words(canonical) {
            return VariantKind::Translation;
        }
        let surface_len = surface.chars().count();
        let canonical_len = canonical.chars().count();
        if surface_len < canonical_len && canonical.contains(surface) {
            return VariantKind::Abbreviation;
        }
        if surface_len > canonical_len && surface.contains(canonical) {
            return VariantKind::FullName;
        }
        VariantKind::Alias
    }

    pub fn compose(self, next: VariantKind) -> VariantKind {
        match (self, next) {
            (VariantKind::Typo, other) | (other, VariantKind::Typo) => other,
            (VariantKind::Alias, other) => other,
            (other, VariantKind::Alias) => other,
            (VariantKind::FormerName, _) | (_, VariantKind::FormerName) => VariantKind::FormerName,
            (VariantKind::Honorific, _) | (_, VariantKind::Honorific) => VariantKind::Honorific,
            (VariantKind::Initialism, VariantKind::Abbreviation)
            | (VariantKind::Abbreviation, VariantKind::Initialism) => VariantKind::Initialism,
            (VariantKind::FullName, VariantKind::Abbreviation)
            | (VariantKind::Abbreviation, VariantKind::FullName) => VariantKind::Abbreviation,
            (VariantKind::Transliteration, VariantKind::Translation)
            | (VariantKind::Translation, VariantKind::Transliteration) => {
                VariantKind::Transliteration
            }
            (_, other) => other,
        }
    }
}

fn contain_cjk(text: &str) -> bool {
    text.chars().any(|ch| {
        matches!(ch,
            '\u{3400}'..='\u{4DBF}'
            | '\u{4E00}'..='\u{9FFF}'
            | '\u{F900}'..='\u{FAFF}'
        )
    })
}

fn is_ascii_words(text: &str) -> bool {
    !text.is_empty()
        && text
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch.is_ascii_whitespace() || ch == '-' || ch == '\'')
        && text.chars().any(|ch| ch.is_ascii_alphabetic())
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct VariantSense {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub type_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub domain: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub range: Option<String>,
}

impl VariantSense {
    pub fn new(type_name: Option<&str>, domain: Option<&str>, range: Option<&str>) -> Self {
        Self {
            type_name: clean(type_name),
            domain: clean(domain),
            range: clean(range),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.type_name.is_none() && self.domain.is_none() && self.range.is_none()
    }

    pub fn label(&self) -> String {
        let mut parts = Vec::new();
        if let Some(type_name) = &self.type_name {
            parts.push(format!("类型 {}", type_name));
        }
        if let Some(domain) = &self.domain {
            parts.push(format!("定义域 {}", domain));
        }
        if let Some(range) = &self.range {
            parts.push(format!("值域 {}", range));
        }
        if parts.is_empty() {
            "通配词义".to_string()
        } else {
            parts.join("，")
        }
    }

    fn key(&self) -> String {
        format!(
            "{}|{}|{}",
            self.type_name.as_deref().unwrap_or(""),
            self.domain.as_deref().unwrap_or(""),
            self.range.as_deref().unwrap_or("")
        )
    }
}

fn clean(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VariantEdge {
    pub from: String,
    pub to: String,
    pub kind: VariantKind,
    pub anchor: String,
    #[serde(default, skip_serializing_if = "VariantSense::is_empty")]
    pub sense: VariantSense,
    #[serde(default)]
    pub validity: Validity,
    pub source: String,
    pub confidence: f64,
    #[serde(default)]
    pub refuted: bool,
}

impl VariantEdge {
    pub fn new(from: &str, to: &str, kind: VariantKind, anchor: &str) -> Self {
        Self {
            from: from.trim().to_string(),
            to: to.trim().to_string(),
            kind,
            anchor: anchor.trim().to_string(),
            sense: VariantSense::default(),
            validity: Validity::Unknown,
            source: "手工声明".to_string(),
            confidence: 0.9,
            refuted: false,
        }
    }

    pub fn with_sense(mut self, sense: VariantSense) -> Self {
        self.sense = sense;
        self
    }

    pub fn with_validity(mut self, validity: Validity) -> Self {
        self.validity = validity;
        self
    }

    pub fn with_source(mut self, source: &str) -> Self {
        self.source = source.trim().to_string();
        self
    }

    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    pub fn valid_years(mut self, start_year: i32, end_year: i32) -> Self {
        self.validity = Validity::from_years(start_year, end_year);
        self
    }

    pub fn refuted(mut self) -> Self {
        self.refuted = true;
        self
    }

    fn key(&self) -> (String, String, String, String) {
        (
            fold_surface(&self.from),
            fold_surface(&self.to),
            fold_surface(&self.anchor),
            format!("{:?}|{}", self.kind, self.sense.key()),
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VariantAnchor {
    pub canonical: String,
    #[serde(default)]
    pub types: BTreeSet<String>,
    #[serde(default)]
    pub observed: bool,
}

impl VariantAnchor {
    pub fn type_label(&self) -> String {
        if self.types.is_empty() {
            "类型未知".to_string()
        } else {
            self.types.iter().cloned().collect::<Vec<_>>().join("、")
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VariantStep {
    pub from: String,
    pub to: String,
    pub kind: VariantKind,
    pub source: String,
    pub confidence: f64,
    #[serde(default)]
    pub validity: Validity,
}

impl VariantStep {
    fn from_edge(edge: &VariantEdge) -> Self {
        Self {
            from: edge.from.clone(),
            to: edge.to.clone(),
            kind: edge.kind,
            source: edge.source.clone(),
            confidence: edge.confidence,
            validity: edge.validity,
        }
    }

    fn synthetic(from: &str, to: &str, kind: VariantKind, source: &str, confidence: f64) -> Self {
        Self {
            from: from.to_string(),
            to: to.to_string(),
            kind,
            source: source.to_string(),
            confidence,
            validity: Validity::Unknown,
        }
    }

    pub fn render(&self) -> String {
        format!("「{}」—[{}]→「{}」", self.from, self.kind.label(), self.to)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VariantCandidate {
    pub anchor: String,
    pub path: Vec<VariantStep>,
    pub confidence: f64,
    #[serde(default)]
    pub validity: Validity,
    #[serde(default)]
    pub anchor_types: BTreeSet<String>,
}

impl VariantCandidate {
    pub fn hops(&self) -> usize {
        self.path.len()
    }

    pub fn is_canonical(&self) -> bool {
        self.path.is_empty()
    }

    pub fn composed_kind(&self) -> Option<VariantKind> {
        let mut kinds = self.path.iter().map(|step| step.kind);
        let first = kinds.next()?;
        Some(kinds.fold(first, |left, right| left.compose(right)))
    }

    pub fn render_witness(&self) -> String {
        if self.path.is_empty() {
            return format!("「{}」（规范名）", self.anchor);
        }
        let mut text = format!("「{}」", self.path[0].from);
        for step in &self.path {
            text.push_str(&format!("—[{}]→「{}」", step.kind.label(), step.to));
        }
        format!("{}（锚定「{}」）", text, self.anchor)
    }

    pub fn render(&self) -> String {
        let kinds = self
            .composed_kind()
            .map(|kind| kind.label())
            .unwrap_or("规范名");
        format!(
            "{}；路径类型 {}，{} 跳，置信度 {:.2}，类型 {}，有效期 {}",
            self.render_witness(),
            kinds,
            self.hops(),
            self.confidence,
            if self.anchor_types.is_empty() {
                "未知".to_string()
            } else {
                self.anchor_types.iter().cloned().collect::<Vec<_>>().join("、")
            },
            self.validity.render()
        )
    }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
struct VariantSnapshot {
    #[serde(default = "dialect_default")]
    dialect: String,
    #[serde(default = "version_default")]
    version: u32,
    #[serde(default)]
    anchors: BTreeMap<String, VariantAnchor>,
    #[serde(default)]
    edges: Vec<VariantEdge>,
}

fn dialect_default() -> String {
    VARIANT_CLOSURE_DIALECT.to_string()
}

fn version_default() -> u32 {
    VARIANT_CLOSURE_VERSION
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct VariantBudget {
    pub max_depth: usize,
    pub max_nodes: usize,
    pub max_expansions: usize,
    pub max_candidates: usize,
}

impl Default for VariantBudget {
    fn default() -> Self {
        Self {
            max_depth: DEFAULT_VARIANT_MAX_DEPTH,
            max_nodes: DEFAULT_VARIANT_MAX_NODES,
            max_expansions: DEFAULT_VARIANT_MAX_EXPANSIONS,
            max_candidates: DEFAULT_VARIANT_MAX_CANDIDATES,
        }
    }
}

impl VariantBudget {
    pub fn with_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth.max(1);
        self
    }

    pub fn with_nodes(mut self, nodes: usize) -> Self {
        self.max_nodes = nodes.max(1);
        self
    }

    pub fn with_expansions(mut self, expansions: usize) -> Self {
        self.max_expansions = expansions.max(1);
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VariantTruncation {
    Depth,
    Nodes,
    Expansions,
    Candidates,
}

impl VariantTruncation {
    pub const fn label(self) -> &'static str {
        match self {
            VariantTruncation::Depth => "深度上限",
            VariantTruncation::Nodes => "节点上限",
            VariantTruncation::Expansions => "扩展上限",
            VariantTruncation::Candidates => "候选上限",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct VariantContext {
    pub expected_type: Option<String>,
    pub predicate: Option<String>,
    pub domain: Option<String>,
    pub range: Option<String>,
    pub at: Option<TimePoint>,
    pub min_confidence: f64,
    pub ambiguity_margin: f64,
    pub include_pending: bool,
    pub budget: VariantBudget,
}

impl Default for VariantContext {
    fn default() -> Self {
        Self {
            expected_type: None,
            predicate: None,
            domain: None,
            range: None,
            at: None,
            min_confidence: DEFAULT_VARIANT_MIN_CONFIDENCE,
            ambiguity_margin: DEFAULT_VARIANT_AMBIGUITY_MARGIN,
            include_pending: false,
            budget: VariantBudget::default(),
        }
    }
}

impl VariantContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_expected_type(mut self, type_name: &str) -> Self {
        self.expected_type = clean(Some(type_name));
        self
    }

    pub fn with_predicate(mut self, predicate: &str) -> Self {
        self.predicate = clean(Some(predicate));
        self
    }

    pub fn with_domain(mut self, domain: &str) -> Self {
        self.domain = clean(Some(domain));
        self
    }

    pub fn with_range(mut self, range: &str) -> Self {
        self.range = clean(Some(range));
        self
    }

    pub fn at(mut self, point: TimePoint) -> Self {
        self.at = Some(point);
        self
    }

    pub fn at_year(self, year: i32) -> Self {
        self.at(TimePoint::year(year))
    }

    pub fn with_min_confidence(mut self, confidence: f64) -> Self {
        self.min_confidence = confidence.clamp(0.0, 1.0);
        self
    }

    pub fn with_margin(mut self, margin: f64) -> Self {
        self.ambiguity_margin = margin.clamp(0.0, 1.0);
        self
    }

    pub fn including_pending(mut self) -> Self {
        self.include_pending = true;
        self
    }

    pub fn including_pending_conditional(mut self, include: bool) -> Self {
        self.include_pending = include;
        self
    }

    pub fn with_budget(mut self, budget: VariantBudget) -> Self {
        self.budget = budget;
        self
    }

    pub fn has_disambiguating_evidence(&self) -> bool {
        self.expected_type.is_some()
            || self.predicate.is_some()
            || self.domain.is_some()
            || self.range.is_some()
            || self.at.is_some()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VariantUnresolvedReason {
    EmptySurface,
    NoEvidence,
    GuardRejected,
    BudgetExhausted,
}

impl VariantUnresolvedReason {
    pub const fn label(self) -> &'static str {
        match self {
            VariantUnresolvedReason::EmptySurface => "词面为空",
            VariantUnresolvedReason::NoEvidence => "没有名变体证据",
            VariantUnresolvedReason::GuardRejected => "证据被类型、时间或置信度守卫拒绝",
            VariantUnresolvedReason::BudgetExhausted => "搜索预算耗尽",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum VariantResolution {
    Canonical(VariantCandidate),
    Resolved(VariantCandidate),
    Ambiguous {
        surface: String,
        candidates: Vec<VariantCandidate>,
    },
    Unresolved {
        surface: String,
        reason: VariantUnresolvedReason,
    },
}

impl VariantResolution {
    pub fn canonical(&self) -> Option<&str> {
        match self {
            VariantResolution::Canonical(candidate) | VariantResolution::Resolved(candidate) => {
                Some(candidate.anchor.as_str())
            }
            _ => None,
        }
    }

    pub fn candidate(&self) -> Option<&VariantCandidate> {
        match self {
            VariantResolution::Canonical(candidate) | VariantResolution::Resolved(candidate) => {
                Some(candidate)
            }
            _ => None,
        }
    }

    pub fn candidates(&self) -> &[VariantCandidate] {
        match self {
            VariantResolution::Ambiguous { candidates, .. } => candidates,
            _ => &[],
        }
    }

    pub fn is_ambiguous(&self) -> bool {
        matches!(self, VariantResolution::Ambiguous { .. })
    }

    pub fn surface(&self) -> &str {
        match self {
            VariantResolution::Canonical(candidate) | VariantResolution::Resolved(candidate) => {
                candidate
                    .path
                    .first()
                    .map(|step| step.from.as_str())
                    .unwrap_or(candidate.anchor.as_str())
            }
            VariantResolution::Ambiguous { surface, .. }
            | VariantResolution::Unresolved { surface, .. } => surface,
        }
    }

    pub fn render(&self) -> String {
        match self {
            VariantResolution::Canonical(candidate) => {
                format!("「{}」已是规范身份；{}", candidate.anchor, candidate.render())
            }
            VariantResolution::Resolved(candidate) => {
                format!("解析为「{}」：{}", candidate.anchor, candidate.render())
            }
            VariantResolution::Ambiguous { surface, candidates } => {
                let mut text = format!("「{}」有多个锚定候选，需语境或澄清：", surface);
                for candidate in candidates {
                    text.push_str(&format!("\n  - {}", candidate.render()));
                }
                text
            }
            VariantResolution::Unresolved { surface, reason } => {
                format!("「{}」未能解析：{}", surface, reason.label())
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct VariantClosure {
    anchors: BTreeMap<String, VariantAnchor>,
    anchor_lookup: BTreeMap<String, BTreeSet<String>>,
    edges: Vec<VariantEdge>,
    edge_index: BTreeMap<(String, String, String, String), usize>,
    outgoing: BTreeMap<String, Vec<usize>>,
    incoming: BTreeMap<String, Vec<usize>>,
    types: Option<TypeSystem>,
}

struct WalkState {
    key: String,
    display: String,
    path: Vec<VariantStep>,
    confidence: f64,
    validity: Validity,
    anchor: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct VariantSurface {
    pub anchor: String,
    pub surface: String,
    pub path: Vec<VariantStep>,
    pub confidence: f64,
    pub validity: Validity,
}

impl VariantSurface {
    pub fn hops(&self) -> usize {
        self.path.len()
    }

    pub fn render(&self) -> String {
        let mut text = format!("「{}」", self.surface);
        for step in &self.path {
            text.push_str(&format!("—[{}]→「{}」", step.kind.label(), step.to));
        }
        format!("{}（锚定「{}」，{:.2}）", text, self.anchor, self.confidence)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct VariantSurfaceChoice {
    pub anchor: String,
    pub surface: String,
    pub source: String,
    pub confidence: f64,
    pub reversible: bool,
}

impl VariantSurfaceChoice {
    pub fn render(&self) -> String {
        format!(
            "「{}」←「{}」：{}（{:.2}{}）",
            self.anchor,
            self.surface,
            self.source,
            self.confidence,
            if self.reversible { "" } else { "，未通过反向解析" }
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MorphologicalCandidate {
    pub surface: String,
    pub kind: VariantKind,
    pub rule: String,
}

impl Default for VariantClosure {
    fn default() -> Self {
        Self::new()
    }
}

impl VariantClosure {
    pub fn new() -> Self {
        Self {
            anchors: BTreeMap::new(),
            anchor_lookup: BTreeMap::new(),
            edges: Vec::new(),
            edge_index: BTreeMap::new(),
            outgoing: BTreeMap::new(),
            incoming: BTreeMap::new(),
            types: None,
        }
    }

    pub fn from_registry(registry: &EntityRegistry) -> Self {
        let mut closure = Self::new();
        closure.add_registry(registry);
        closure
    }

    pub fn from_parts(registry: &EntityRegistry, synonyms: &SynonymGraph) -> Self {
        let mut closure = Self::from_registry(registry);
        closure.add_synonym_graph(synonyms);
        closure
    }

    pub fn with_type_system(mut self, types: TypeSystem) -> Self {
        self.types = Some(types);
        self
    }

    pub fn type_system(&self) -> Option<&TypeSystem> {
        self.types.as_ref()
    }

    pub fn declare_anchor(&mut self, canonical: &str) -> bool {
        self.declare_anchor_typed(canonical, &BTreeSet::new())
    }

    pub fn declare_anchor_typed(&mut self, canonical: &str, types: &BTreeSet<String>) -> bool {
        let canonical = canonical.trim();
        if canonical.is_empty() {
            return false;
        }
        let key = fold_surface(canonical);
        if key.is_empty() {
            return false;
        }
        let entry = self.anchors.entry(key.clone()).or_insert_with(|| VariantAnchor {
            canonical: canonical.to_string(),
            types: BTreeSet::new(),
            observed: false,
        });
        let mut changed = false;
        if entry.canonical != canonical {
            changed = true;
            entry.canonical = canonical.to_string();
        }
        for type_name in types {
            let type_name = type_name.trim();
            if !type_name.is_empty() && entry.types.insert(type_name.to_string()) {
                changed = true;
            }
        }
        self.anchor_lookup
            .entry(key.clone())
            .or_default()
            .insert(entry.canonical.clone());
        changed
    }

    pub fn mark_observed(&mut self, canonical: &str) -> bool {
        let key = fold_surface(canonical);
        let Some(anchor) = self.anchors.get_mut(&key) else {
            return false;
        };
        if anchor.observed {
            return false;
        }
        anchor.observed = true;
        true
    }

    pub fn anchor(&self, canonical: &str) -> Option<&VariantAnchor> {
        self.anchors.get(&fold_surface(canonical))
    }

    pub fn anchor_for(&self, surface: &str) -> Option<&str> {
        let key = fold_surface(surface);
        if let Some(anchor) = self.anchors.get(&key) {
            return Some(anchor.canonical.as_str());
        }
        let owners = self.anchor_lookup.get(&key)?;
        if owners.len() == 1 {
            owners.iter().next().map(String::as_str)
        } else {
            None
        }
    }

    pub fn add_registry(&mut self, registry: &EntityRegistry) -> usize {
        let mut added = 0usize;
        for record in registry.records() {
            if record.merged_into.is_some() || record.status != EntityStatus::Confirmed {
                continue;
            }
            self.declare_anchor_typed(&record.canonical, &record.types);
            if record.observed {
                self.mark_observed(&record.canonical);
            }
            for alias in record.aliases.values() {
                if alias.status != EntityStatus::Confirmed {
                    continue;
                }
                let mut surfaces = vec![alias.surface.clone()];
                surfaces.extend(alias.variants.iter().cloned());
                for surface in surfaces {
                    if fold_surface(&surface) == fold_surface(&record.canonical) {
                        continue;
                    }
                    let edge = VariantEdge::new(
                        &surface,
                        &record.canonical,
                        VariantKind::infer(&surface, &record.canonical),
                        &record.canonical,
                    )
                    .with_source("实体注册表")
                    .with_confidence(0.95);
                    if self.add_edge(edge) {
                        added += 1;
                    }
                }
            }
        }
        added
    }

    pub fn add_synonym_graph(&mut self, graph: &SynonymGraph) -> usize {
        self.add_synonym_graph_including(graph, false)
    }

    pub fn add_synonym_graph_including(
        &mut self,
        graph: &SynonymGraph,
        include_proposed: bool,
    ) -> usize {
        let mut added = 0usize;
        for entry in graph.entries() {
            if entry.kind != SynonymKind::Entity || entry.is_self_entry() {
                continue;
            }
            if entry.status == SynonymStatus::Refuted {
                continue;
            }
            if !entry.is_confirmed() && !include_proposed {
                continue;
            }
            let Some(anchor) = self.anchor_for(&entry.canonical).map(str::to_string) else {
                continue;
            };
            let mut source = if entry.sources.is_empty() {
                "同义词图".to_string()
            } else {
                format!(
                    "同义词图（{}）",
                    entry
                        .sources
                        .iter()
                        .cloned()
                        .collect::<Vec<_>>()
                        .join("、")
                )
            };
            if entry.status == SynonymStatus::Proposed {
                source.push_str("，待确认");
            }
            let edge = VariantEdge::new(
                &entry.surface,
                &entry.canonical,
                VariantKind::infer(&entry.surface, &entry.canonical),
                &anchor,
            )
            .with_sense(VariantSense::new(
                entry.sense.type_name.as_deref(),
                entry.sense.domain.as_deref(),
                entry.sense.range.as_deref(),
            ))
            .with_source(&source)
            .with_confidence(entry.confidence);
            if self.add_edge(edge) {
                added += 1;
            }
        }
        added
    }

    pub fn add_edge(&mut self, edge: VariantEdge) -> bool {
        let from_key = fold_surface(&edge.from);
        let to_key = fold_surface(&edge.to);
        let anchor_key = fold_surface(&edge.anchor);
        if from_key.is_empty()
            || to_key.is_empty()
            || from_key == to_key
            || !self.anchors.contains_key(&anchor_key)
        {
            return false;
        }
        let canonical = self
            .anchors
            .get(&anchor_key)
            .map(|anchor| anchor.canonical.clone())
            .expect("锚刚检查");
        let mut edge = edge;
        edge.anchor = canonical.clone();
        edge.confidence = edge.confidence.clamp(0.0, 1.0);
        let key = edge.key();
        if let Some(index) = self.edge_index.get(&key).copied() {
            let existing = &mut self.edges[index];
            let mut changed = false;
            if edge.confidence > existing.confidence {
                existing.confidence = edge.confidence;
                changed = true;
            }
            if !edge.source.is_empty() && !existing.source.contains(&edge.source) {
                existing.source.push('、');
                existing.source.push_str(&edge.source);
                changed = true;
            }
            return changed;
        }
        let index = self.edges.len();
        self.edge_index.insert(key, index);
        self.outgoing
            .entry(from_key.clone())
            .or_default()
            .push(index);
        self.incoming
            .entry(to_key.clone())
            .or_default()
            .push(index);
        self.anchor_lookup
            .entry(from_key)
            .or_default()
            .insert(canonical.clone());
        self.edges.push(edge);
        true
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn anchor_count(&self) -> usize {
        self.anchors.len()
    }

    pub fn edges(&self) -> &[VariantEdge] {
        &self.edges
    }

    pub fn types(&self) -> &BTreeMap<String, VariantAnchor> {
        &self.anchors
    }

    pub fn refute_edge(&mut self, from: &str, to: &str, anchor: &str) -> bool {
        self.set_edge_refuted(from, to, anchor, true)
    }

    pub fn restore_edge(&mut self, from: &str, to: &str, anchor: &str) -> bool {
        self.set_edge_refuted(from, to, anchor, false)
    }

    fn set_edge_refuted(&mut self, from: &str, to: &str, anchor: &str, refuted: bool) -> bool {
        let from_key = fold_surface(from);
        let to_key = fold_surface(to);
        let anchor_key = fold_surface(anchor);
        let mut changed = false;
        for edge in &mut self.edges {
            if fold_surface(&edge.from) == from_key
                && fold_surface(&edge.to) == to_key
                && fold_surface(&edge.anchor) == anchor_key
                && edge.refuted != refuted
            {
                edge.refuted = refuted;
                changed = true;
            }
        }
        changed
    }

    pub fn is_edge_refuted(&self, from: &str, to: &str, anchor: &str) -> bool {
        let from_key = fold_surface(from);
        let to_key = fold_surface(to);
        let anchor_key = fold_surface(anchor);
        self.edges.iter().any(|edge| {
            edge.refuted
                && fold_surface(&edge.from) == from_key
                && fold_surface(&edge.to) == to_key
                && fold_surface(&edge.anchor) == anchor_key
        })
    }

    fn snapshot(&self) -> VariantSnapshot {
        VariantSnapshot {
            dialect: VARIANT_CLOSURE_DIALECT.to_string(),
            version: VARIANT_CLOSURE_VERSION,
            anchors: self.anchors.clone(),
            edges: self.edges.clone(),
        }
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(&self.snapshot()).unwrap_or_else(|_| "{}".to_string())
    }

    pub fn from_json(text: &str) -> Option<Self> {
        let snapshot: VariantSnapshot = serde_json::from_str(text).ok()?;
        if snapshot.version > VARIANT_CLOSURE_VERSION {
            return None;
        }
        let mut closure = Self::new();
        for (key, anchor) in snapshot.anchors {
            let folded = fold_surface(&anchor.canonical);
            if folded.is_empty() || folded != key {
                return None;
            }
            closure.declare_anchor_typed(&anchor.canonical, &anchor.types);
            if anchor.observed {
                closure.mark_observed(&anchor.canonical);
            }
        }
        for edge in snapshot.edges {
            if !closure.add_edge(edge.clone()) && !closure.edges.iter().any(|existing| existing.key() == edge.key()) {
                return None;
            }
        }
        Some(closure)
    }

    pub fn save_to_path(&self, path: &str) -> std::io::Result<()> {
        std::fs::write(path, self.to_json())
    }

    pub fn load_from_path(path: &str) -> Option<Self> {
        let text = std::fs::read_to_string(path).ok()?;
        Self::from_json(&text)
    }

    pub fn digest(&self) -> String {
        match serde_json::to_value(self.snapshot()) {
            Ok(value) => fnv1a_hex(canonical_json(&value).as_bytes()),
            Err(_) => String::new(),
        }
    }

    pub fn audit_lines(&self) -> Vec<String> {
        let mut lines = vec![format!(
            "名变体闭包：{} 个身份锚，{} 条变体边（{} 条墓碑）",
            self.anchors.len(),
            self.edges.len(),
            self.edges.iter().filter(|edge| edge.refuted).count()
        )];
        let mut anchors: Vec<&VariantAnchor> = self.anchors.values().collect();
        anchors.sort_by(|left, right| left.canonical.cmp(&right.canonical));
        for anchor in anchors {
            lines.push(format!(
                "  - 「{}」（{}）",
                anchor.canonical,
                anchor.type_label()
            ));
            let mut edges: Vec<&VariantEdge> = self
                .edges
                .iter()
                .filter(|edge| fold_surface(&edge.anchor) == fold_surface(&anchor.canonical))
                .collect();
            edges.sort_by(|left, right| {
                left.from
                    .cmp(&right.from)
                    .then_with(|| left.to.cmp(&right.to))
            });
            for edge in edges {
                lines.push(format!(
                    "      {}「{}」→「{}」（{}，{:.2}{}）",
                    if edge.refuted { "✗ " } else { "" },
                    edge.from,
                    edge.to,
                    edge.kind.label(),
                    edge.confidence,
                    if edge.source.is_empty() {
                        String::new()
                    } else {
                        format!("，{}", edge.source)
                    }
                ));
            }
        }
        lines
    }

    pub fn report(&self) -> String {
        self.audit_lines().join("\n")
    }

    pub fn resolve(&self, surface: &str, context: &VariantContext) -> VariantResolution {
        let surface = surface.trim();
        if surface.is_empty() {
            return VariantResolution::Unresolved {
                surface: surface.to_string(),
                reason: VariantUnresolvedReason::EmptySurface,
            };
        }
        let start_key = fold_surface(surface);
        if self.anchors.contains_key(&start_key) {
            let anchor = self
                .anchors
                .get(&start_key)
                .expect("锚刚存在")
                .clone();
            if !self.anchor_admits(&anchor, context) {
                return VariantResolution::Unresolved {
                    surface: surface.to_string(),
                    reason: VariantUnresolvedReason::GuardRejected,
                };
            }
            return VariantResolution::Canonical(self.candidate_for(anchor.canonical, Vec::new(), 1.0, Validity::Unknown));
        }

        let mut frontier = vec![WalkState {
            key: start_key.clone(),
            display: surface.to_string(),
            path: Vec::new(),
            confidence: 1.0,
            validity: Validity::Unknown,
            anchor: None,
        }];
        let mut best: BTreeMap<(String, String), (f64, usize)> = BTreeMap::new();
        let mut candidates: BTreeMap<String, VariantCandidate> = BTreeMap::new();
        let mut expansions = 0usize;
        let mut visited_nodes: BTreeSet<String> = BTreeSet::new();
        visited_nodes.insert(fold_surface(surface));
        let mut truncated: Option<VariantTruncation> = None;
        let mut guard_rejected = false;

        'walk: while let Some(state) = frontier.pop() {
            let anchor_key = state.anchor.as_deref().map(fold_surface);
            if let Some(anchor_key) = anchor_key.as_deref() {
                if state.key == anchor_key {
                    let anchor = self
                        .anchors
                        .get(&state.key)
                        .map(|anchor| anchor.canonical.clone())
                        .unwrap_or_else(|| state.display.clone());
                    let confidence = state.confidence;
                    let candidate = self.candidate_for(
                        anchor,
                        state.path.clone(),
                        confidence,
                        state.validity,
                    );
                    let replace = candidates
                        .get(&candidate.anchor)
                        .map(|existing| {
                            candidate.confidence > existing.confidence + f64::EPSILON
                                || ((candidate.confidence - existing.confidence).abs()
                                    <= f64::EPSILON
                                    && candidate.path.len() < existing.path.len())
                        })
                        .unwrap_or(true);
                    if replace {
                        candidates.insert(candidate.anchor.clone(), candidate);
                    }
                    continue;
                }
            }
            if self.anchors.contains_key(&state.key) && state.key != start_key {
                continue;
            }
            if state.path.len() >= context.budget.max_depth {
                if state.anchor.is_some() {
                    truncated.get_or_insert(VariantTruncation::Depth);
                }
                continue;
            }
            let Some(indices) = self.outgoing.get(&state.key) else {
                continue;
            };
            for index in indices {
                let edge = &self.edges[*index];
                expansions += 1;
                if expansions > context.budget.max_expansions {
                    truncated = Some(VariantTruncation::Expansions);
                    break 'walk;
                }
                if edge.refuted
                    || edge.confidence < context.min_confidence
                    || !self.edge_admits(edge, context)
                {
                    guard_rejected = true;
                    continue;
                }
                if let Some(existing_anchor) = state.anchor.as_deref() {
                    if fold_surface(existing_anchor) != fold_surface(&edge.anchor) {
                        guard_rejected = true;
                        continue;
                    }
                }
                let next_key = fold_surface(&edge.to);
                if next_key.is_empty() {
                    continue;
                }
                if self.anchors.contains_key(&next_key) {
                    let target = self.anchors.get(&next_key).expect("锚刚存在");
                    if fold_surface(&target.canonical) != fold_surface(&edge.anchor) {
                        guard_rejected = true;
                        continue;
                    }
                    if !self.anchor_admits(target, context) {
                        guard_rejected = true;
                        continue;
                    }
                }
                let validity = combine_validity(state.validity, edge.validity);
                if validity.is_never() || !validity_admits(validity, context.at) {
                    guard_rejected = true;
                    continue;
                }
                let decay = if state.path.is_empty() {
                    1.0
                } else {
                    DEFAULT_VARIANT_HOP_DECAY
                };
                let confidence = (state.confidence * edge.confidence * decay).clamp(0.0, 1.0);
                if confidence < context.min_confidence {
                    guard_rejected = true;
                    continue;
                }
                let mut path = state.path.clone();
                path.push(VariantStep::from_edge(edge));
                let anchor = Some(edge.anchor.clone());
                let visit_key = (next_key.clone(), anchor.clone().unwrap_or_default());
                let replace = best
                    .get(&visit_key)
                    .map(|(score, hops)| {
                        confidence > *score + f64::EPSILON
                            || ((confidence - *score).abs() <= f64::EPSILON
                                && path.len() < *hops)
                    })
                    .unwrap_or(true);
                if !replace {
                    continue;
                }
                best.insert(visit_key, (confidence, path.len()));
                if !visited_nodes.contains(&next_key) {
                    if visited_nodes.len() >= context.budget.max_nodes {
                        truncated = Some(VariantTruncation::Nodes);
                        break 'walk;
                    }
                    visited_nodes.insert(next_key.clone());
                }
                frontier.push(WalkState {
                    key: next_key,
                    display: edge.to.clone(),
                    path,
                    confidence,
                    validity,
                    anchor,
                });
            }
        }

        if candidates.is_empty() {
            let reason = if let Some(truncation) = truncated {
                let _ = truncation;
                VariantUnresolvedReason::BudgetExhausted
            } else if guard_rejected {
                VariantUnresolvedReason::GuardRejected
            } else {
                VariantUnresolvedReason::NoEvidence
            };
            return VariantResolution::Unresolved {
                surface: surface.to_string(),
                reason,
            };
        }

        let mut ranked: Vec<VariantCandidate> = candidates.into_values().collect();
        ranked.sort_by(|left, right| {
            right
                .confidence
                .partial_cmp(&left.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.path.len().cmp(&right.path.len()))
                .then_with(|| left.anchor.cmp(&right.anchor))
        });
        ranked.truncate(context.budget.max_candidates.max(1));
        if let Some(first) = ranked.first() {
            let decisive = ranked.len() == 1
                || ranked
                    .get(1)
                    .map(|second| first.confidence - second.confidence >= context.ambiguity_margin)
                    .unwrap_or(true);
            if decisive && (ranked.len() == 1 || context.has_disambiguating_evidence() || first.confidence >= 0.75) {
                let candidate = ranked.remove(0);
                return VariantResolution::Resolved(candidate);
            }
        }
        VariantResolution::Ambiguous {
            surface: surface.to_string(),
            candidates: ranked,
        }
    }

    pub fn resolve_with_morphology(
        &self,
        surface: &str,
        context: &VariantContext,
    ) -> VariantResolution {
        let base = self.resolve(surface, context);
        if !base.is_ambiguous() && base.candidate().is_some() {
            return base;
        }
        let surface = surface.trim();
        let mut candidates: BTreeMap<String, VariantCandidate> = BTreeMap::new();
        if let VariantResolution::Ambiguous {
            candidates: base_candidates,
            ..
        } = &base
        {
            for candidate in base_candidates {
                candidates.insert(candidate.anchor.clone(), candidate.clone());
            }
        }
        for rule in Self::morphological_candidates(surface) {
            let resolved = self.resolve(&rule.surface, context);
            let Some(candidate) = resolved.candidate() else {
                continue;
            };
            let mut path = Vec::with_capacity(candidate.path.len() + 1);
            let confidence = candidate.confidence * 0.85;
            if confidence < context.min_confidence {
                continue;
            }
            path.push(VariantStep::synthetic(
                surface,
                &rule.surface,
                rule.kind,
                &rule.rule,
                confidence,
            ));
            path.extend(candidate.path.iter().cloned());
            let combined = VariantCandidate {
                anchor: candidate.anchor.clone(),
                path,
                confidence,
                validity: candidate.validity,
                anchor_types: candidate.anchor_types.clone(),
            };
            let replace = candidates
                .get(&combined.anchor)
                .map(|existing| combined.confidence > existing.confidence + f64::EPSILON)
                .unwrap_or(true);
            if replace {
                candidates.insert(combined.anchor.clone(), combined);
            }
        }
        if candidates.is_empty() {
            return base;
        }
        let mut ranked: Vec<VariantCandidate> = candidates.into_values().collect();
        ranked.sort_by(|left, right| {
            right
                .confidence
                .partial_cmp(&left.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.path.len().cmp(&right.path.len()))
                .then_with(|| left.anchor.cmp(&right.anchor))
        });
        ranked.truncate(context.budget.max_candidates.max(1));
        if ranked.len() == 1
            || ranked
                .get(1)
                .map(|second| ranked[0].confidence - second.confidence >= context.ambiguity_margin)
                .unwrap_or(true)
        {
            return VariantResolution::Resolved(ranked.remove(0));
        }
        VariantResolution::Ambiguous {
            surface: surface.to_string(),
            candidates: ranked,
        }
    }

    pub fn variants_of(&self, anchor: &str, context: &VariantContext) -> Vec<VariantSurface> {
        let anchor_key = fold_surface(anchor);
        let Some(anchor_record) = self.anchors.get(&anchor_key) else {
            return Vec::new();
        };
        let canonical = anchor_record.canonical.clone();
        if !self.anchor_admits(anchor_record, context) {
            return Vec::new();
        }
        let mut frontier = vec![WalkState {
            key: anchor_key.clone(),
            display: canonical.clone(),
            path: Vec::new(),
            confidence: 1.0,
            validity: Validity::Unknown,
            anchor: Some(canonical.clone()),
        }];
        let mut best: BTreeMap<String, (VariantSurface, usize)> = BTreeMap::new();
        let mut expansions = 0usize;
        let mut visited: BTreeSet<String> = BTreeSet::new();
        while let Some(state) = frontier.pop() {
            if state.path.len() >= context.budget.max_depth {
                continue;
            }
            let Some(indices) = self.incoming.get(&state.key) else {
                continue;
            };
            for index in indices {
                let edge = &self.edges[*index];
                expansions += 1;
                if expansions > context.budget.max_expansions {
                    break;
                }
                if edge.refuted
                    || edge.confidence < context.min_confidence
                    || fold_surface(&edge.anchor) != anchor_key
                    || !self.edge_admits(edge, context)
                {
                    continue;
                }
                let from_key = fold_surface(&edge.from);
                if from_key.is_empty() || from_key == anchor_key {
                    continue;
                }
                if self.anchors.contains_key(&from_key)
                    && fold_surface(&self.anchors[&from_key].canonical) != anchor_key
                {
                    continue;
                }
                let validity = combine_validity(state.validity, edge.validity);
                if validity.is_never() || !validity_admits(validity, context.at) {
                    continue;
                }
                let decay = if state.path.is_empty() {
                    1.0
                } else {
                    DEFAULT_VARIANT_HOP_DECAY
                };
                let confidence = (state.confidence * edge.confidence * decay).clamp(0.0, 1.0);
                if confidence < context.min_confidence {
                    continue;
                }
                let mut path = Vec::with_capacity(state.path.len() + 1);
                path.push(VariantStep::from_edge(edge));
                path.extend(state.path.iter().cloned());
                let path_for_state = path.clone();
                let surface = VariantSurface {
                    anchor: canonical.clone(),
                    surface: edge.from.clone(),
                    path,
                    confidence,
                    validity,
                };
                let replace = best
                    .get(&from_key)
                    .map(|(existing, existing_hops)| {
                        confidence > existing.confidence + f64::EPSILON
                            || ((confidence - existing.confidence).abs() <= f64::EPSILON
                                && surface.path.len() < *existing_hops)
                    })
                    .unwrap_or(true);
                if replace {
                    best.insert(from_key.clone(), (surface, state.path.len() + 1));
                }
                if !visited.contains(&from_key) {
                    if visited.len() >= context.budget.max_nodes {
                        break;
                    }
                    visited.insert(from_key.clone());
                    frontier.push(WalkState {
                        key: from_key,
                        display: edge.from.clone(),
                        path: path_for_state,
                        confidence,
                        validity,
                        anchor: Some(canonical.clone()),
                    });
                }
            }
        }
        let mut variants: Vec<VariantSurface> = best
            .into_values()
            .map(|(variant, _)| variant)
            .filter(|variant| {
                self.resolve(&variant.surface, context)
                    .canonical()
                    .map(|resolved| resolved == canonical)
                    .unwrap_or(false)
            })
            .collect();
        variants.sort_by(|left, right| {
            right
                .confidence
                .partial_cmp(&left.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.path.len().cmp(&right.path.len()))
                .then_with(|| left.surface.chars().count().cmp(&right.surface.chars().count()))
                .then_with(|| left.surface.cmp(&right.surface))
        });
        variants.truncate(context.budget.max_candidates.max(1));
        variants
    }

    pub fn alias_pairs(&self, context: &VariantContext) -> Vec<(String, String)> {
        let mut surfaces: BTreeSet<String> = BTreeSet::new();
        for edge in &self.edges {
            if !edge.refuted {
                surfaces.insert(edge.from.clone());
            }
        }
        let mut pairs: Vec<(String, String)> = surfaces
            .into_iter()
            .filter_map(|surface| {
                let canonical = self.resolve(&surface, context).canonical()?.to_string();
                (canonical != surface).then_some((surface, canonical))
            })
            .collect();
        pairs.sort();
        pairs.dedup();
        pairs
    }

    pub fn surface_for(
        &self,
        anchor: &str,
        context: &VariantContext,
        echoes: &[String],
    ) -> VariantSurfaceChoice {
        let canonical = self.anchor(anchor).map(|record| record.canonical.clone());
        let Some(canonical) = canonical else {
            return VariantSurfaceChoice {
                anchor: anchor.trim().to_string(),
                surface: anchor.trim().to_string(),
                source: "未登记锚".to_string(),
                confidence: 0.0,
                reversible: false,
            };
        };
        for echo in echoes {
            let echo = echo.trim();
            if echo.is_empty() || echo == canonical {
                continue;
            }
            if self.resolve(echo, context).canonical() == Some(canonical.as_str()) {
                return VariantSurfaceChoice {
                    anchor: canonical,
                    surface: echo.to_string(),
                    source: format!("会话回声「{}」", echo),
                    confidence: 1.0,
                    reversible: true,
                };
            }
        }
        if let Some(variant) = self.variants_of(&canonical, context).into_iter().next() {
            return VariantSurfaceChoice {
                anchor: canonical,
                surface: variant.surface,
                source: format!(
                    "已确认变体（{}）",
                    variant
                        .path
                        .first()
                        .map(|step| step.kind.label())
                        .unwrap_or("别名")
                ),
                confidence: variant.confidence,
                reversible: true,
            };
        }
        VariantSurfaceChoice {
            anchor: canonical.clone(),
            surface: canonical,
            source: "规范名".to_string(),
            confidence: 1.0,
            reversible: true,
        }
    }

    pub fn morphological_candidates(surface: &str) -> Vec<MorphologicalCandidate> {
        let surface = surface.trim();
        if surface.is_empty() {
            return Vec::new();
        }
        let mut out: BTreeMap<String, MorphologicalCandidate> = BTreeMap::new();
        for suffix in HONORIFIC_SUFFIXES {
            if let Some(stem) = surface.strip_suffix(suffix) {
                let stem = stem.trim();
                if stem.chars().count() >= 2 && stem != surface {
                    out.entry(stem.to_string()).or_insert(MorphologicalCandidate {
                        surface: stem.to_string(),
                        kind: VariantKind::Honorific,
                        rule: format!("去除尊称后缀「{}」", suffix),
                    });
                }
            }
        }
        for suffix in ORGANIZATION_SUFFIXES {
            if let Some(stem) = surface.strip_suffix(suffix) {
                let stem = stem.trim();
                if stem.chars().count() >= 2 && stem != surface {
                    out.entry(stem.to_string()).or_insert(MorphologicalCandidate {
                        surface: stem.to_string(),
                        kind: VariantKind::Elision,
                        rule: format!("去除机构通名「{}」", suffix),
                    });
                }
            }
        }
        if let Some(open) = surface.rfind(['（', '(']) {
            if surface.ends_with('）') || surface.ends_with(')') {
                let inner = surface[..open].trim();
                if inner.chars().count() >= 2 && inner != surface {
                    out.entry(inner.to_string()).or_insert(MorphologicalCandidate {
                        surface: inner.to_string(),
                        kind: VariantKind::Elision,
                        rule: "去除括注".to_string(),
                    });
                }
            }
        }
        let words: Vec<&str> = surface
            .split(|ch: char| ch.is_whitespace() || ch == '-' || ch == '\'')
            .filter(|word| !word.is_empty())
            .collect();
        if words.len() >= 2
            && words
                .iter()
                .all(|word| word.chars().next().map(|ch| ch.is_ascii_alphabetic()).unwrap_or(false))
        {
            let initials: String = words
                .iter()
                .filter_map(|word| word.chars().next())
                .flat_map(|ch| ch.to_uppercase())
                .collect();
            if initials.chars().count() >= 2 && initials != surface {
                out.entry(initials.clone()).or_insert(MorphologicalCandidate {
                    surface: initials,
                    kind: VariantKind::Initialism,
                    rule: "取英文首字母".to_string(),
                });
            }
        }
        let mut result: Vec<MorphologicalCandidate> = out.into_values().collect();
        result.sort_by(|left, right| {
            left.surface
                .chars()
                .count()
                .cmp(&right.surface.chars().count())
                .then_with(|| left.surface.cmp(&right.surface))
        });
        result
    }

    fn candidate_for(
        &self,
        anchor: String,
        path: Vec<VariantStep>,
        confidence: f64,
        validity: Validity,
    ) -> VariantCandidate {
        let anchor_types = self
            .anchors
            .get(&fold_surface(&anchor))
            .map(|record| record.types.clone())
            .unwrap_or_default();
        VariantCandidate {
            anchor,
            path,
            confidence: confidence.clamp(0.0, 1.0),
            validity,
            anchor_types,
        }
    }

    fn anchor_admits(&self, anchor: &VariantAnchor, context: &VariantContext) -> bool {
        let Some(expected) = context.expected_type.as_deref() else {
            return true;
        };
        if anchor.types.is_empty() {
            return true;
        }
        if let Some(types) = &self.types {
            return match types.sets_admit(&anchor.types, &BTreeSet::from([expected.to_string()])) {
                TypeVerdict::Compatible => true,
                TypeVerdict::Disjoint => false,
                TypeVerdict::Unknown => anchor
                    .types
                    .iter()
                    .any(|type_name| type_name.eq_ignore_ascii_case(expected)),
            };
        }
        anchor
            .types
            .iter()
            .any(|type_name| type_name.eq_ignore_ascii_case(expected))
    }

    fn edge_admits(&self, edge: &VariantEdge, context: &VariantContext) -> bool {
        if !validity_admits(edge.validity, context.at) {
            return false;
        }
        if let Some(expected) = context.expected_type.as_deref() {
            if let Some(type_name) = edge.sense.type_name.as_deref() {
                if let Some(types) = &self.types {
                    if types.types_disjoint(type_name, expected) {
                        return false;
                    }
                } else if !type_name.eq_ignore_ascii_case(expected) {
                    return false;
                }
            }
        }
        if let Some(predicate) = context.domain.as_deref().or(context.predicate.as_deref()) {
            if let Some(domain) = edge.sense.domain.as_deref() {
                if !domain.eq_ignore_ascii_case(predicate) {
                    return false;
                }
            }
        }
        if let Some(range) = context.range.as_deref() {
            if let Some(edge_range) = edge.sense.range.as_deref() {
                if !edge_range.eq_ignore_ascii_case(range) {
                    return false;
                }
            }
        }
        true
    }
}

fn validity_admits(validity: Validity, at: Option<TimePoint>) -> bool {
    let Some(point) = at else {
        return !validity.is_never();
    };
    match validity {
        Validity::Never => false,
        Validity::Unknown | Validity::Unbounded => true,
        Validity::Span(span) => span.contains(point),
    }
}

fn combine_validity(current: Validity, next: Validity) -> Validity {
    match (current, next) {
        (Validity::Never, _) | (_, Validity::Never) => Validity::Never,
        (Validity::Unknown, value) | (value, Validity::Unknown) => value,
        (left, right) => left.intersection(right),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity_registry::NominationOrigin;
    use crate::synonym_graph::SynonymSense;
    use crate::type_system::TypeSystem;

    fn registry() -> EntityRegistry {
        let mut registry = EntityRegistry::new();
        registry.observe_kb_entity(
            "中国科学院大学",
            &BTreeSet::from(["University".to_string()]),
            &["国科大".to_string()],
            "测试图谱",
        );
        registry
    }

    fn synonym_graph() -> SynonymGraph {
        let mut graph = SynonymGraph::new(32);
        let outcome = graph.observe(
            1,
            "UCAS",
            "国科大",
            SynonymKind::Entity,
            SynonymSense::default(),
            "测试",
            0.9,
        );
        if let Some(id) = outcome.id() {
            graph.confirm(id, 1);
        }
        graph
    }

    #[test]
    fn nested_variant_paths_share_one_anchor() {
        let closure = VariantClosure::from_parts(&registry(), &synonym_graph());
        let resolution = closure.resolve("UCAS", &VariantContext::new());
        let candidate = resolution.candidate().expect("应解析到唯一锚");
        assert_eq!(candidate.anchor, "中国科学院大学");
        assert_eq!(candidate.path.len(), 2);
        assert_eq!(candidate.path[0].from, "UCAS");
        assert_eq!(candidate.path[0].to, "国科大");
        assert_eq!(candidate.path[1].to, "中国科学院大学");
        assert!(candidate.render_witness().contains("首字母缩写"));
    }

    #[test]
    fn shared_variant_surface_stays_ambiguous() {
        let mut registry = EntityRegistry::new();
        registry.confirm_alias_with_origin("北大", "北京大学", NominationOrigin::Dialogue);
        registry.confirm_alias_with_origin("北大", "北大方正", NominationOrigin::Dialogue);
        let closure = VariantClosure::from_registry(&registry);
        let resolution = closure.resolve("北大", &VariantContext::new());
        assert!(resolution.is_ambiguous(), "{}", resolution.render());
        let mut anchors: Vec<&str> = resolution
            .candidates()
            .iter()
            .map(|candidate| candidate.anchor.as_str())
            .collect();
        anchors.sort();
        assert_eq!(anchors, vec!["北京大学", "北大方正"]);
    }

    #[test]
    fn context_type_disambiguates_shared_alias() {
        let mut registry = EntityRegistry::new();
        registry.confirm_alias_with_origin("北大", "北京大学", NominationOrigin::Dialogue);
        registry.confirm_alias_with_origin("北大", "北大方正", NominationOrigin::Dialogue);
        registry.observe_kb_entity(
            "北京大学",
            &BTreeSet::from(["University".to_string()]),
            &[],
            "测试图谱",
        );
        registry.observe_kb_entity(
            "北大方正",
            &BTreeSet::from(["Company".to_string()]),
            &[],
            "测试图谱",
        );
        let types = TypeSystem::new();
        let closure = VariantClosure::from_registry(&registry).with_type_system(types);
        let context = VariantContext::new().with_expected_type("University");
        let resolution = closure.resolve("北大", &context);
        assert_eq!(resolution.canonical(), Some("北京大学"));
    }

    #[test]
    fn temporal_window_guards_former_name() {
        let mut closure = VariantClosure::new();
        closure.declare_anchor("北京");
        assert!(closure.add_edge(
            VariantEdge::new("北平", "北京", VariantKind::FormerName, "北京").valid_years(1928, 1949)
        ));
        assert_eq!(
            closure.resolve("北平", &VariantContext::new().at_year(1935)).canonical(),
            Some("北京")
        );
        let expired = closure.resolve("北平", &VariantContext::new().at_year(2020));
        assert!(matches!(
            expired,
            VariantResolution::Unresolved {
                reason: VariantUnresolvedReason::GuardRejected,
                ..
            }
        ));
    }

    #[test]
    fn reverse_generation_rejects_unreversible_surface() {
        let mut closure = VariantClosure::new();
        closure.declare_anchor("中国科学院大学");
        closure.declare_anchor("北京大学");
        closure.add_edge(
            VariantEdge::new("UCAS", "中国科学院大学", VariantKind::Initialism, "中国科学院大学")
                .with_confidence(0.95),
        );
        closure.add_edge(
            VariantEdge::new("UCAS", "北京大学", VariantKind::Alias, "北京大学")
                .with_confidence(0.95),
        );
        closure.add_edge(
            VariantEdge::new("国科大", "中国科学院大学", VariantKind::Abbreviation, "中国科学院大学")
                .with_confidence(0.95),
        );
        let variants = closure.variants_of("中国科学院大学", &VariantContext::new());
        assert_eq!(variants.len(), 1);
        assert_eq!(variants[0].surface, "国科大");
        let choice = closure.surface_for("中国科学院大学", &VariantContext::new(), &[]);
        assert_eq!(choice.surface, "国科大");
        assert!(choice.reversible);
    }

    #[test]
    fn alias_pairs_expand_nested_variants_but_skip_ambiguous_surfaces() {
        let closure = VariantClosure::from_parts(&registry(), &synonym_graph());
        let pairs = closure.alias_pairs(&VariantContext::new());
        assert!(pairs.contains(&("UCAS".to_string(), "中国科学院大学".to_string())));
        assert!(pairs.contains(&("国科大".to_string(), "中国科学院大学".to_string())));

        let mut registry = EntityRegistry::new();
        registry.confirm_alias_with_origin("北大", "北京大学", NominationOrigin::Dialogue);
        registry.confirm_alias_with_origin("北大", "北大方正", NominationOrigin::Dialogue);
        let ambiguous = VariantClosure::from_registry(&registry).alias_pairs(&VariantContext::new());
        assert!(ambiguous.iter().all(|(surface, _)| surface != "北大"));
    }

    #[test]
    fn reverse_generation_follows_nested_paths() {
        let closure = VariantClosure::from_parts(&registry(), &synonym_graph());
        let variants = closure.variants_of("中国科学院大学", &VariantContext::new());
        let surfaces: Vec<&str> = variants
            .iter()
            .map(|variant| variant.surface.as_str())
            .collect();
        assert!(surfaces.contains(&"国科大"));
        assert!(surfaces.contains(&"UCAS"));
        let ucas = variants
            .iter()
            .find(|variant| variant.surface == "UCAS")
            .expect("嵌套变体应保留完整路径");
        assert_eq!(ucas.path.len(), 2);
        assert_eq!(ucas.path[0].from, "UCAS");
        assert_eq!(ucas.path[1].to, "中国科学院大学");
        assert_eq!(ucas.render().matches("→").count(), 2);
    }

    #[test]
    fn morphology_is_anchored_and_composable() {
        let mut registry = EntityRegistry::new();
        registry.confirm_alias_with_origin("鲁迅", "周树人", NominationOrigin::Dialogue);
        let closure = VariantClosure::from_registry(&registry);
        let resolution = closure.resolve_with_morphology("鲁迅先生", &VariantContext::new());
        let candidate = resolution.candidate().expect("尊称变体应锚定");
        assert_eq!(candidate.anchor, "周树人");
        assert_eq!(candidate.path.len(), 2);
        assert_eq!(candidate.path[0].kind, VariantKind::Honorific);
    }

    #[test]
    fn missing_anchor_keeps_morphology_as_candidate_only() {
        let closure = VariantClosure::new();
        assert!(!VariantClosure::morphological_candidates("鲁迅先生").is_empty());
        let resolution = closure.resolve_with_morphology("鲁迅先生", &VariantContext::new());
        assert!(matches!(
            resolution,
            VariantResolution::Unresolved {
                reason: VariantUnresolvedReason::NoEvidence,
                ..
            }
        ));
    }

    #[test]
    fn refuted_edge_is_a_persistent_tombstone() {
        let mut closure = VariantClosure::new();
        closure.declare_anchor("北京");
        closure.add_edge(VariantEdge::new("北平", "北京", VariantKind::FormerName, "北京"));
        assert_eq!(closure.resolve("北平", &VariantContext::new()).canonical(), Some("北京"));
        assert!(closure.refute_edge("北平", "北京", "北京"));
        assert!(closure.is_edge_refuted("北平", "北京", "北京"));
        let blocked = closure.resolve("北平", &VariantContext::new());
        assert!(matches!(
            blocked,
            VariantResolution::Unresolved {
                reason: VariantUnresolvedReason::GuardRejected,
                ..
            }
        ));
        let restored = VariantClosure::from_json(&closure.to_json()).expect("快照应可解码");
        assert!(restored.is_edge_refuted("北平", "北京", "北京"));
        let mut restored = restored;
        assert!(restored.restore_edge("北平", "北京", "北京"));
        assert_eq!(
            restored.resolve("北平", &VariantContext::new()).canonical(),
            Some("北京")
        );
    }

    #[test]
    fn snapshot_roundtrip_is_stable() {
        let closure = VariantClosure::from_parts(&registry(), &synonym_graph());
        let json = closure.to_json();
        let restored = VariantClosure::from_json(&json).expect("快照应可解码");
        assert_eq!(closure.digest(), restored.digest());
        assert_eq!(closure.edges().len(), restored.edges().len());
        assert_eq!(
            restored.resolve("UCAS", &VariantContext::new()).canonical(),
            Some("中国科学院大学")
        );
        assert!(VariantClosure::from_json("{\"version\": 99}").is_none());
    }

    #[test]
    fn budget_truncation_is_reported() {
        let mut closure = VariantClosure::new();
        closure.declare_anchor("A");
        closure.add_edge(VariantEdge::new("S", "M1", VariantKind::Alias, "A"));
        closure.add_edge(VariantEdge::new("M1", "M2", VariantKind::Alias, "A"));
        closure.add_edge(VariantEdge::new("M2", "A", VariantKind::Alias, "A"));
        let context = VariantContext::new()
            .with_budget(VariantBudget::default().with_depth(1).with_expansions(1));
        let resolution = closure.resolve("S", &context);
        assert!(matches!(
            resolution,
            VariantResolution::Unresolved {
                reason: VariantUnresolvedReason::BudgetExhausted,
                ..
            }
        ));
    }
}

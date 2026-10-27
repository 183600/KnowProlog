use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fmt::Write as _;
use serde::{Deserialize, Serialize};
use crate::nlp::shared_jieba;
use crate::schema_view::SchemaView;
pub const DEFAULT_MAX_TOKEN_CHARS: usize = 8;
pub const DEFAULT_BEAM: usize = 5;
pub const DEFAULT_AMBIGUITY_MARGIN: f64 = 0.05;
const MIN_COMPOSITION_CHARS: usize = 3;
const MAX_COMPOSITION_CHARS: usize = 12;
const MIN_COMPOSITION_COVERAGE: f64 = 0.4;
const MAX_CONTEXT_FEATURES: usize = 32;
const MAX_READING_FEATURES: usize = 48;
const MAX_MEMORY_SURFACES: usize = 512;
const ENTITY_PRIOR: f64 = 1.4;
const TYPE_PRIOR: f64 = 0.9;
const PREDICATE_PRIOR: f64 = 0.8;
const JIEBA_PRIOR: f64 = 0.35;
const UNKNOWN_CHAR_PRIOR: f64 = 0.02;
const PUNCT_PRIOR: f64 = 0.04;
const CONTEXT_WEIGHT: f64 = 0.9;
const SCHEMA_WEIGHT: f64 = 0.55;
const MEMORY_WEIGHT: f64 = 0.35;
const COMPOSITION_BASE: f64 = 0.55;
const COMPOSITION_SUPPORT_WEIGHT: f64 = 0.75;
const COMPOSITION_LENGTH_BONUS: f64 = 0.1;
const COMPOSITION_LENGTH_CAP: f64 = 0.35;
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReadingKind {
    Entity,
    Type,
    Predicate,
    Composition,
    Unknown,
}
impl ReadingKind {
    pub fn code(self) -> &'static str {
        match self {
            ReadingKind::Entity => "entity",
            ReadingKind::Type => "type",
            ReadingKind::Predicate => "predicate",
            ReadingKind::Composition => "composition",
            ReadingKind::Unknown => "unknown",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            ReadingKind::Entity => "实体",
            ReadingKind::Type => "类型",
            ReadingKind::Predicate => "谓词",
            ReadingKind::Composition => "组合未登录词",
            ReadingKind::Unknown => "未登录",
        }
    }

    fn prior(self) -> f64 {
        match self {
            ReadingKind::Entity => ENTITY_PRIOR,
            ReadingKind::Type => TYPE_PRIOR,
            ReadingKind::Predicate => PREDICATE_PRIOR,
            ReadingKind::Composition | ReadingKind::Unknown => JIEBA_PRIOR,
        }
    }
}
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ReadingScore {
    pub lexicon: f64,
    pub context: f64,
    pub schema: f64,
    pub memory: f64,
    pub composition: f64,
    pub length_bonus: f64,
    pub total: f64,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SegmentedToken {
    pub text: String,
    pub start: usize,
    pub end: usize,
    pub canonical: String,
    pub kind: ReadingKind,
    pub etype: Option<String>,
    pub evidence: Vec<String>,
    pub score: ReadingScore,
}
impl SegmentedToken {
    pub fn display_reading(&self) -> String {
        if self.canonical == self.text {
            self.text.clone()
        } else {
            format!("{}（{}）", self.text, self.canonical)
        }
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AlternativePath {
    pub tokens: Vec<String>,
    pub readings: Vec<String>,
    pub score: f64,
    pub gap: f64,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextSnapshot {
    pub terms: Vec<String>,
    pub expected_types: Vec<String>,
    pub predicate: Option<String>,
    pub fingerprint: String,
    pub features: Vec<(String, f64)>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AmbiguityReport {
    pub margin: f64,
    pub threshold: f64,
    pub divergent_index: usize,
    pub content_divergence: bool,
    pub options: Vec<String>,
    pub clarification: Option<String>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SegmentResult {
    pub text: String,
    pub tokens: Vec<SegmentedToken>,
    pub score: f64,
    pub margin: f64,
    pub context: ContextSnapshot,
    pub alternatives: Vec<AlternativePath>,
    pub ambiguity: Option<AmbiguityReport>,
}
impl SegmentResult {
    pub fn token_texts(&self) -> Vec<&str> {
        self.tokens
            .iter()
            .map(|token| token.text.as_str())
            .collect()
    }

    pub fn render_segmentation(&self) -> String {
        self.tokens
            .iter()
            .map(|token| token.text.as_str())
            .collect::<Vec<_>>()
            .join(" | ")
    }

    pub fn render_report(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "分词：{}", self.render_segmentation());
        let _ = writeln!(out, "总分：{:.4}", self.score);
        let _ = writeln!(
            out,
            "语境：{}（{}）",
            if self.context.fingerprint.is_empty() {
                "∅"
            } else {
                self.context.fingerprint.as_str()
            },
            self.context.features.len()
        );
        if !self.context.terms.is_empty() {
            let _ = writeln!(out, "焦点：{}", self.context.terms.join("、"));
        }
        if !self.context.expected_types.is_empty() {
            let _ = writeln!(out, "期望类型：{}", self.context.expected_types.join("、"));
        }
        if let Some(predicate) = &self.context.predicate {
            let _ = writeln!(out, "期望谓词：{}", predicate);
        }
        out.push_str("读法：\n");
        for token in &self.tokens {
            let etype = token
                .etype
                .as_deref()
                .map(|etype| format!("，类型={}", etype))
                .unwrap_or_default();
            let _ = writeln!(
                out,
                "  [{},{}) {:<12} {}「{}」{} 总分={:.4}（词表={:.3} 语境={:.3} 图式={:.3} 记忆={:.3} 组合={:.3} 长度={:.3}）",
                token.start,
                token.end,
                token.text,
                token.kind.label(),
                token.canonical,
                etype,
                token.score.total,
                token.score.lexicon,
                token.score.context,
                token.score.schema,
                token.score.memory,
                token.score.composition,
                token.score.length_bonus
            );
            if !token.evidence.is_empty() {
                let _ = writeln!(out, "      证据：{}", token.evidence.join("；"));
            }
        }
        if !self.alternatives.is_empty() {
            out.push_str("备选：\n");
            for alternative in &self.alternatives {
                let _ = writeln!(
                    out,
                    "  {}（差 {:.4}）",
                    alternative.tokens.join(" | "),
                    alternative.gap.max(0.0)
                );
            }
        }
        if let Some(ambiguity) = &self.ambiguity {
            if let Some(question) = &ambiguity.clarification {
                let _ = writeln!(
                    out,
                    "歧义：边际 {:.4} ≤ 阈值 {:.4}，分歧在内容词 {}",
                    ambiguity.margin,
                    ambiguity.threshold,
                    ambiguity.divergent_index + 1
                );
                let _ = writeln!(out, "澄清：{}", question);
            } else {
                let _ = writeln!(
                    out,
                    "歧义边际：{:.4}（阈值 {:.4}，未触发澄清）",
                    ambiguity.margin, ambiguity.threshold
                );
            }
        } else if !self.alternatives.is_empty() {
            let _ = writeln!(out, "歧义边际：{:.4}（未触发澄清）", self.margin);
        }
        out
    }
}
#[derive(Debug, Clone)]
pub struct SegmentOptions {
    pub context_terms: Vec<String>,
    pub expected_types: Vec<String>,
    pub predicate: Option<String>,
    pub beam: usize,
    pub max_token_chars: usize,
    pub ambiguity_margin: f64,
}
impl Default for SegmentOptions {
    fn default() -> Self {
        Self {
            context_terms: Vec::new(),
            expected_types: Vec::new(),
            predicate: None,
            beam: DEFAULT_BEAM,
            max_token_chars: DEFAULT_MAX_TOKEN_CHARS,
            ambiguity_margin: DEFAULT_AMBIGUITY_MARGIN,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReadingPreference {
    pub canonical: String,
    pub kind: ReadingKind,
    pub confirmations: u32,
    pub corrections: u32,
}
impl ReadingPreference {
    pub fn observations(&self) -> u32 {
        self.confirmations.saturating_add(self.corrections)
    }

    pub fn reliability(&self) -> f64 {
        let confirmed = f64::from(self.confirmations) + 1.0;
        let total = f64::from(self.confirmations) + f64::from(self.corrections) + 2.0;
        confirmed / total
    }

    pub fn preference_score(&self) -> f64 {
        self.reliability() * 2.0 - 1.0
    }
}
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ReadingMemory {
    entries: BTreeMap<String, BTreeMap<String, ReadingPreference>>,
    max_surfaces: usize,
}
impl ReadingMemory {
    pub fn new(max_surfaces: usize) -> Self {
        Self {
            entries: BTreeMap::new(),
            max_surfaces: max_surfaces.max(1),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn confirm(
        &mut self,
        surface: &str,
        canonical: &str,
        kind: ReadingKind,
        context_fingerprint: &str,
    ) -> bool {
        let surface = surface.trim();
        let canonical = canonical.trim();
        if surface.is_empty() || canonical.is_empty() {
            return false;
        }
        let bucket = self.bucket_key(surface, context_fingerprint);
        let accepted_key = reading_key(canonical, kind);
        let readings = self.entries.entry(bucket).or_default();
        let entry = readings
            .entry(accepted_key.clone())
            .or_insert_with(|| ReadingPreference {
                canonical: canonical.to_string(),
                kind,
                confirmations: 0,
                corrections: 0,
            });
        entry.confirmations = entry.confirmations.saturating_add(1);
        entry.kind = kind;
        for (key, candidate) in readings.iter_mut() {
            if key != &accepted_key {
                candidate.corrections = candidate.corrections.saturating_add(1);
            }
        }
        self.enforce_capacity();
        true
    }

    pub fn correct(
        &mut self,
        surface: &str,
        rejected_canonical: &str,
        rejected_kind: ReadingKind,
        accepted_canonical: &str,
        accepted_kind: ReadingKind,
        context_fingerprint: &str,
    ) -> bool {
        let surface = surface.trim();
        let rejected = rejected_canonical.trim();
        let accepted = accepted_canonical.trim();
        if surface.is_empty() || accepted.is_empty() {
            return false;
        }
        let bucket = self.bucket_key(surface, context_fingerprint);
        {
            let readings = self.entries.entry(bucket.clone()).or_default();
            let accepted_entry = readings
                .entry(reading_key(accepted, accepted_kind))
                .or_insert_with(|| ReadingPreference {
                    canonical: accepted.to_string(),
                    kind: accepted_kind,
                    confirmations: 0,
                    corrections: 0,
                });
            accepted_entry.confirmations = accepted_entry.confirmations.saturating_add(1);
            accepted_entry.kind = accepted_kind;
            if !rejected.is_empty() && rejected != accepted {
                let rejected_entry = readings
                    .entry(reading_key(rejected, rejected_kind))
                    .or_insert_with(|| ReadingPreference {
                        canonical: rejected.to_string(),
                        kind: rejected_kind,
                        confirmations: 0,
                        corrections: 0,
                    });
                rejected_entry.corrections = rejected_entry.corrections.saturating_add(1);
                rejected_entry.kind = rejected_kind;
            }
            for (key, candidate) in readings.iter_mut() {
                if key != &reading_key(accepted, accepted_kind)
                    && key != &reading_key(rejected, rejected_kind)
                {
                    candidate.corrections = candidate.corrections.saturating_add(1);
                }
            }
        }
        self.enforce_capacity();
        true
    }

    pub fn preference(
        &self,
        surface: &str,
        canonical: &str,
        kind: ReadingKind,
        context_fingerprint: &str,
    ) -> Option<f64> {
        let bucket = self.bucket_key(surface.trim(), context_fingerprint);
        self.entries
            .get(&bucket)
            .and_then(|readings| readings.get(&reading_key(canonical.trim(), kind)))
            .map(ReadingPreference::preference_score)
    }

    pub fn preferred(
        &self,
        surface: &str,
        context_fingerprint: &str,
    ) -> Option<&ReadingPreference> {
        let bucket = self.bucket_key(surface.trim(), context_fingerprint);
        self.entries.get(&bucket).and_then(|readings| {
            let mut best: Option<&ReadingPreference> = None;
            for preference in readings.values() {
                let replace = match best {
                    None => true,
                    Some(current) => {
                        preference
                            .preference_score()
                            .total_cmp(&current.preference_score())
                            .then_with(|| current.canonical.cmp(&preference.canonical))
                            .then_with(|| current.kind.cmp(&preference.kind))
                            == std::cmp::Ordering::Greater
                    }
                };
                if replace {
                    best = Some(preference);
                }
            }
            best
        })
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    pub fn from_json(text: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(text)
    }

    pub fn render_report(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "读法记忆：{} 个语境桶 / {} 条读法",
            self.entries.len(),
            self.entries.values().map(BTreeMap::len).sum::<usize>()
        );
        for (bucket, readings) in &self.entries {
            let (surface, context) = bucket.split_once('\u{1f}').unwrap_or((bucket.as_str(), ""));
            let context = if context.is_empty() { "∅" } else { context };
            for preference in readings.values() {
                let _ = writeln!(
                    out,
                    "  「{}」@{} {}「{}」确认={} 纠正={} 可靠度={:.3}",
                    surface,
                    context,
                    preference.kind.label(),
                    preference.canonical,
                    preference.confirmations,
                    preference.corrections,
                    preference.reliability()
                );
            }
        }
        out
    }

    fn bucket_key(&self, surface: &str, context_fingerprint: &str) -> String {
        format!("{}\u{1f}{}", surface, context_fingerprint)
    }

    fn enforce_capacity(&mut self) {
        while self.entries.len() > self.max_surfaces {
            let Some(first) = self.entries.keys().next().cloned() else {
                break;
            };
            self.entries.remove(&first);
        }
    }
}
fn reading_key(canonical: &str, kind: ReadingKind) -> String {
    format!("{}:{}", kind.code(), canonical)
}
#[derive(Debug, Clone, Default, PartialEq)]
struct SparseVector {
    weights: BTreeMap<String, f64>,
    norm: f64,
}
impl SparseVector {
    fn from_counts(counts: BTreeMap<String, f64>) -> Self {
        let mut vector = Self {
            weights: counts
                .into_iter()
                .filter(|(key, value)| !key.is_empty() && value.is_finite() && *value > 0.0)
                .collect(),
            norm: 0.0,
        };
        vector.finalize();
        vector
    }

    fn finalize(&mut self) {
        self.norm = self
            .weights
            .values()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        if self.norm > 0.0 {
            for value in self.weights.values_mut() {
                *value /= self.norm;
            }
            self.norm = 1.0;
        }
    }

    fn truncate(&mut self, limit: usize) {
        if self.weights.len() <= limit {
            return;
        }
        let mut entries: Vec<(String, f64)> =
            std::mem::take(&mut self.weights).into_iter().collect();
        entries.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        entries.truncate(limit);
        self.weights = entries.into_iter().collect();
        self.finalize();
    }

    fn cosine(&self, other: &Self) -> f64 {
        if self.weights.is_empty() || other.weights.is_empty() {
            return 0.0;
        }
        let (small, large) = if self.weights.len() <= other.weights.len() {
            (&self.weights, &other.weights)
        } else {
            (&other.weights, &self.weights)
        };
        let mut dot = 0.0;
        for (key, value) in small {
            if let Some(other_value) = large.get(key) {
                dot += value * other_value;
            }
        }
        dot.clamp(-1.0, 1.0)
    }

    fn top(&self, limit: usize) -> Vec<(&str, f64)> {
        let mut entries: Vec<(&str, f64)> = self
            .weights
            .iter()
            .map(|(key, value)| (key.as_str(), *value))
            .collect();
        entries.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0)));
        entries.truncate(limit);
        entries
    }

    fn fingerprint(&self) -> String {
        if self.weights.is_empty() {
            return String::new();
        }
        let mut canonical = String::new();
        for (key, value) in self.top(MAX_CONTEXT_FEATURES) {
            let _ = write!(canonical, "{}={:.4};", key, value);
        }
        format!("{:016x}", fnv1a(canonical.as_bytes()))
    }
}
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct SurfaceReading {
    canonical: String,
    kind: ReadingKind,
}
#[derive(Debug, Clone, Default)]
struct EntityProfile {
    etype: Option<String>,
    predicates: BTreeSet<String>,
    neighbors: BTreeSet<String>,
}
#[derive(Debug, Clone, Default)]
struct NgramDocument {
    features: BTreeMap<String, f64>,
    doc_freq: usize,
    start_count: usize,
    end_count: usize,
    interior_count: usize,
}
impl NgramDocument {
    fn position_admits(&self, at_start: bool, at_end: bool) -> bool {
        let total = self.start_count + self.end_count + self.interior_count;
        if total == 0 {
            return true;
        }
        let start_ratio = self.start_count as f64 / total as f64;
        let end_ratio = self.end_count as f64 / total as f64;
        if at_start && !at_end && end_ratio >= 0.6 && start_ratio < 0.6 {
            return false;
        }
        if at_end && !at_start && start_ratio >= 0.6 && end_ratio < 0.6 {
            return false;
        }
        true
    }
}
#[derive(Debug, Clone)]
struct Composition {
    features: BTreeMap<String, f64>,
    support: f64,
    etype: Option<String>,
    evidence: Vec<String>,
}
#[derive(Debug, Clone)]
struct CandidateSeed {
    canonical: String,
    kind: ReadingKind,
    etype: Option<String>,
    lexicon: f64,
    composition: f64,
    features: BTreeMap<String, f64>,
    evidence: Vec<String>,
}
#[derive(Debug, Clone)]
struct Candidate {
    start: usize,
    end: usize,
    surface: String,
    readings: Vec<ScoredReading>,
}
#[derive(Debug, Clone)]
struct ScoredReading {
    canonical: String,
    kind: ReadingKind,
    etype: Option<String>,
    evidence: Vec<String>,
    score: ReadingScore,
}
#[derive(Debug, Clone)]
struct PathEntry {
    score: f64,
    start: usize,
    prev_rank: usize,
    candidate: usize,
    reading: usize,
}
#[derive(Debug, Clone)]
struct ResolvedPath {
    tokens: Vec<SegmentedToken>,
    score: f64,
}
struct ContextVector {
    vector: SparseVector,
    snapshot: ContextSnapshot,
}
pub struct ContextualSegmenter {
    entities: BTreeMap<String, EntityProfile>,
    surfaces: BTreeMap<String, Vec<SurfaceReading>>,
    ngram_documents: BTreeMap<String, NgramDocument>,
    triples: HashSet<(String, String, String)>,
    indexed_surfaces: usize,
    dirty: bool,
    schema: Option<SchemaView>,
    memory: ReadingMemory,
}
impl Default for ContextualSegmenter {
    fn default() -> Self {
        Self::new()
    }
}
impl ContextualSegmenter {
    pub fn new() -> Self {
        Self {
            entities: BTreeMap::new(),
            surfaces: BTreeMap::new(),
            ngram_documents: BTreeMap::new(),
            triples: HashSet::new(),
            indexed_surfaces: 0,
            dirty: true,
            schema: None,
            memory: ReadingMemory::new(MAX_MEMORY_SURFACES),
        }
    }

    pub fn set_schema(&mut self, schema: SchemaView) {
        self.schema = Some(schema);
    }

    pub fn schema(&self) -> Option<&SchemaView> {
        self.schema.as_ref()
    }

    pub fn memory(&self) -> &ReadingMemory {
        &self.memory
    }

    pub fn memory_mut(&mut self) -> &mut ReadingMemory {
        &mut self.memory
    }

    pub fn reading_kind(&self, surface: &str, canonical: &str) -> Option<ReadingKind> {
        self.surfaces
            .get(surface.trim())?
            .iter()
            .find(|reading| reading.canonical == canonical.trim())
            .map(|reading| reading.kind)
    }

    /// 某个词面在词表中的全部读法（规范名 + 读法类别），按规范名排序。
    /// 只读；供帧约束联合解析在词面层做类型裁决，不暴露内部结构。
    pub fn surface_readings(&self, surface: &str) -> Vec<(String, ReadingKind)> {
        let mut readings: Vec<(String, ReadingKind)> = self
            .surfaces
            .get(surface.trim())
            .map(|entries| {
                entries
                    .iter()
                    .map(|reading| (reading.canonical.clone(), reading.kind))
                    .collect()
            })
            .unwrap_or_default();
        readings.sort();
        readings.dedup();
        readings
    }

    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    pub fn surface_count(&self) -> usize {
        self.surfaces.len()
    }

    pub fn type_count(&self) -> usize {
        self.unique_canonical_count(ReadingKind::Type)
    }

    pub fn predicate_count(&self) -> usize {
        self.unique_canonical_count(ReadingKind::Predicate)
    }

    pub fn triple_count(&self) -> usize {
        self.triples.len()
    }

    fn unique_canonical_count(&self, kind: ReadingKind) -> usize {
        self.surfaces
            .values()
            .flat_map(|readings| readings.iter())
            .filter(|reading| reading.kind == kind)
            .map(|reading| reading.canonical.as_str())
            .collect::<BTreeSet<_>>()
            .len()
    }

    pub fn ngram_count(&self) -> usize {
        self.ngram_documents.len()
    }

    pub fn observe_type(&mut self, etype: &str) {
        let etype = etype.trim();
        if etype.is_empty() {
            return;
        }
        self.add_surface_reading(etype, etype, ReadingKind::Type);
    }

    pub fn observe_type_alias(&mut self, alias: &str, canonical: &str) {
        let alias = alias.trim();
        let canonical = canonical.trim();
        if alias.is_empty() || canonical.is_empty() || alias == canonical {
            return;
        }
        self.observe_type(canonical);
        self.add_surface_reading(alias, canonical, ReadingKind::Type);
    }

    pub fn observe_predicate(&mut self, predicate: &str) {
        let predicate = predicate.trim();
        if predicate.is_empty() {
            return;
        }
        self.add_surface_reading(predicate, predicate, ReadingKind::Predicate);
    }

    pub fn observe_predicate_alias(&mut self, alias: &str, canonical: &str) {
        let alias = alias.trim();
        let canonical = canonical.trim();
        if alias.is_empty() || canonical.is_empty() || alias == canonical {
            return;
        }
        self.observe_predicate(canonical);
        self.add_surface_reading(alias, canonical, ReadingKind::Predicate);
    }

    pub fn observe_entity(&mut self, name: &str, etype: Option<&str>) {
        let name = name.trim();
        if name.is_empty() {
            return;
        }
        let etype = etype.map(str::trim).filter(|etype| !etype.is_empty());
        let profile = self.entities.entry(name.to_string()).or_default();
        if profile.etype.is_none() {
            profile.etype = etype.map(str::to_string);
        }
        if let Some(etype) = etype {
            self.observe_type(etype);
        }
        self.add_surface_reading(name, name, ReadingKind::Entity);
        self.dirty = true;
    }

    pub fn observe_alias(&mut self, alias: &str, canonical: &str) {
        let alias = alias.trim();
        let canonical = canonical.trim();
        if alias.is_empty() || canonical.is_empty() || alias == canonical {
            return;
        }
        self.entities.entry(canonical.to_string()).or_default();
        self.add_surface_reading(alias, canonical, ReadingKind::Entity);
    }

    pub fn observe_triple(&mut self, subject: &str, predicate: &str, object: &str) {
        let subject = subject.trim();
        let predicate = predicate.trim();
        let object = object.trim();
        if subject.is_empty() || predicate.is_empty() || object.is_empty() {
            return;
        }
        if !self.triples.insert((
            subject.to_string(),
            predicate.to_string(),
            object.to_string(),
        )) {
            return;
        }
        self.observe_entity(subject, None);
        self.observe_entity(object, None);
        self.observe_predicate(predicate);
        if let Some(profile) = self.entities.get_mut(subject) {
            profile.predicates.insert(predicate.to_string());
            profile.neighbors.insert(object.to_string());
        }
        if let Some(profile) = self.entities.get_mut(object) {
            profile.predicates.insert(predicate.to_string());
            profile.neighbors.insert(subject.to_string());
        }
        self.dirty = true;
    }

    pub fn rebuild_index(&mut self) {
        if !self.dirty {
            return;
        }
        let surface_features: Vec<(String, BTreeMap<String, f64>)> = self
            .surfaces
            .iter()
            .filter_map(|(surface, readings)| {
                let mut features = BTreeMap::new();
                for reading in readings {
                    self.accumulate_reading_features(
                        &mut features,
                        &reading.canonical,
                        reading.kind,
                        1.0,
                    );
                }
                (!features.is_empty()).then(|| (surface.clone(), features))
            })
            .collect();
        self.ngram_documents.clear();
        let mut doc_freqs: BTreeMap<String, usize> = BTreeMap::new();
        for (surface, features) in &surface_features {
            let mut seen: HashSet<String> = HashSet::new();
            for ngram in surface_ngrams(surface, 1, 3) {
                if !seen.insert(ngram.clone()) {
                    continue;
                }
                let char_len = surface.chars().count();
                let ngram_len = ngram.chars().count();
                let ngram_start = find_char_span(surface, &ngram).map(|(start, _)| start);
                let document = self.ngram_documents.entry(ngram.clone()).or_default();
                for (feature, value) in features {
                    *document.features.entry(feature.clone()).or_insert(0.0) += *value;
                }
                if let Some(start) = ngram_start {
                    let at_start = start == 0;
                    let at_end = start + ngram_len == char_len;
                    if at_start {
                        document.start_count += 1;
                    }
                    if at_end {
                        document.end_count += 1;
                    }
                    if !at_start && !at_end {
                        document.interior_count += 1;
                    }
                }
                *doc_freqs.entry(ngram).or_insert(0) += 1;
            }
        }
        for (ngram, document) in &mut self.ngram_documents {
            document.doc_freq = doc_freqs.get(ngram).copied().unwrap_or(0).max(1);
            let total: f64 = document.features.values().sum();
            if total > 0.0 {
                for value in document.features.values_mut() {
                    *value /= total;
                }
            }
        }
        self.indexed_surfaces = self.surfaces.len();
        self.dirty = false;
    }

    pub fn segment(&mut self, text: &str, options: &SegmentOptions) -> SegmentResult {
        self.rebuild_index();
        let context = self.build_context(options);
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        if n == 0 {
            return SegmentResult {
                text: text.to_string(),
                tokens: Vec::new(),
                score: 0.0,
                margin: 0.0,
                context: context.snapshot,
                alternatives: Vec::new(),
                ambiguity: None,
            };
        }

        let max_token = options.max_token_chars.clamp(1, 16);
        let beam = options.beam.clamp(1, 16);
        let jieba_spans = jieba_spans(text, n);
        let mut candidates: Vec<Candidate> = Vec::new();
        let mut by_start: Vec<Vec<usize>> = vec![Vec::new(); n + 1];
        for start in 0..n {
            let last = (start + max_token).min(n);
            for end in start + 1..=last {
                if let Some(candidate) =
                    self.build_candidate(&chars, start, end, &jieba_spans, &context)
                {
                    let index = candidates.len();
                    by_start[start].push(index);
                    candidates.push(candidate);
                }
            }
        }

        let mut paths: Vec<Vec<PathEntry>> = vec![Vec::new(); n + 1];
        paths[0].push(PathEntry {
            score: 0.0,
            start: 0,
            prev_rank: 0,
            candidate: usize::MAX,
            reading: 0,
        });
        for end in 1..=n {
            let mut entries: Vec<PathEntry> = Vec::new();
            for start in 0..end {
                if paths[start].is_empty() {
                    continue;
                }
                for candidate_index in by_start[start].iter().copied() {
                    let candidate = &candidates[candidate_index];
                    if candidate.end != end {
                        continue;
                    }
                    for (reading_index, reading) in candidate.readings.iter().enumerate() {
                        for (rank, previous) in paths[start].iter().enumerate() {
                            entries.push(PathEntry {
                                score: previous.score + reading.score.total,
                                start,
                                prev_rank: rank,
                                candidate: candidate_index,
                                reading: reading_index,
                            });
                        }
                    }
                }
            }
            entries.sort_by(|a, b| {
                b.score
                    .total_cmp(&a.score)
                    .then_with(|| a.start.cmp(&b.start))
                    .then_with(|| a.candidate.cmp(&b.candidate))
                    .then_with(|| a.reading.cmp(&b.reading))
                    .then_with(|| a.prev_rank.cmp(&b.prev_rank))
            });
            entries.truncate(beam);
            paths[end] = entries;
        }

        let mut resolved: Vec<ResolvedPath> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for rank in 0..paths[n].len() {
            let path = reconstruct_path(&paths, &candidates, &chars, n, rank);
            let key = path_key(&path.tokens);
            if !seen.insert(key) {
                continue;
            }
            resolved.push(path);
            if resolved.len() > 3 {
                break;
            }
        }
        let best = resolved.first().cloned().unwrap_or_else(|| ResolvedPath {
            tokens: Vec::new(),
            score: 0.0,
        });
        let margin = resolved
            .get(1)
            .map(|alternative| (best.score - alternative.score).max(0.0))
            .unwrap_or(0.0);
        let alternatives: Vec<AlternativePath> = resolved
            .iter()
            .skip(1)
            .map(|alternative| AlternativePath {
                tokens: alternative
                    .tokens
                    .iter()
                    .map(|token| token.text.clone())
                    .collect(),
                readings: alternative
                    .tokens
                    .iter()
                    .map(SegmentedToken::display_reading)
                    .collect(),
                score: alternative.score,
                gap: (best.score - alternative.score).max(0.0),
            })
            .collect();
        let ambiguity = resolved.get(1).and_then(|alternative| {
            ambiguity_report(
                text,
                &best.tokens,
                &alternative.tokens,
                margin,
                options.ambiguity_margin,
            )
        });

        SegmentResult {
            text: text.to_string(),
            tokens: best.tokens,
            score: best.score,
            margin,
            context: context.snapshot,
            alternatives,
            ambiguity,
        }
    }

    fn add_surface_reading(&mut self, surface: &str, canonical: &str, kind: ReadingKind) {
        let surface = surface.trim();
        let canonical = canonical.trim();
        if surface.is_empty() || canonical.is_empty() {
            return;
        }
        let readings = self.surfaces.entry(surface.to_string()).or_default();
        let reading = SurfaceReading {
            canonical: canonical.to_string(),
            kind,
        };
        if !readings.contains(&reading) {
            readings.push(reading);
            readings.sort();
            self.dirty = true;
        }
    }

    fn accumulate_reading_features(
        &self,
        target: &mut BTreeMap<String, f64>,
        canonical: &str,
        kind: ReadingKind,
        scale: f64,
    ) {
        match kind {
            ReadingKind::Entity => {
                if let Some(profile) = self.entities.get(canonical) {
                    if let Some(etype) = &profile.etype {
                        *target.entry(format!("类型·{}", etype)).or_insert(0.0) += scale;
                    }
                    for predicate in &profile.predicates {
                        *target.entry(format!("谓词·{}", predicate)).or_insert(0.0) += scale * 0.55;
                    }
                    for neighbor in &profile.neighbors {
                        *target.entry(format!("邻·{}", neighbor)).or_insert(0.0) += scale * 0.35;
                    }
                }
            }
            ReadingKind::Type => {
                *target.entry(format!("类型·{}", canonical)).or_insert(0.0) += scale;
            }
            ReadingKind::Predicate => {
                *target.entry(format!("谓词·{}", canonical)).or_insert(0.0) += scale;
            }
            ReadingKind::Composition | ReadingKind::Unknown => {}
        }
    }

    fn surface_features(&self, surface: &str) -> BTreeMap<String, f64> {
        let surface = surface.trim();
        if surface.is_empty() {
            return BTreeMap::new();
        }
        if let Some(readings) = self.surfaces.get(surface) {
            let mut features = BTreeMap::new();
            let mut seen: HashSet<&SurfaceReading> = HashSet::new();
            for reading in readings {
                if !seen.insert(reading) {
                    continue;
                }
                self.accumulate_reading_features(
                    &mut features,
                    &reading.canonical,
                    reading.kind,
                    1.0,
                );
            }
            if !features.is_empty() {
                return features;
            }
        }
        self.compose_surface(surface)
            .map(|composition| composition.features)
            .unwrap_or_default()
    }

    fn compose_surface(&self, surface: &str) -> Option<Composition> {
        let char_count = surface.chars().count();
        if char_count < MIN_COMPOSITION_CHARS
            || char_count > MAX_COMPOSITION_CHARS
            || surface
                .chars()
                .any(|ch| ch.is_whitespace() || is_punctuation(ch) || is_function_char(ch))
            || self.is_known_partition(surface)
        {
            return None;
        }
        let mut coverage = vec![false; char_count];
        let mut features: BTreeMap<String, f64> = BTreeMap::new();
        let mut evidence: Vec<(f64, String, Option<String>)> = Vec::new();
        let mut matched = false;
        for ngram in surface_ngrams(surface, 2, 3) {
            let Some(document) = self.ngram_documents.get(&ngram) else {
                continue;
            };
            if document.features.is_empty() {
                continue;
            }
            let Some((start, end)) = find_char_span(surface, &ngram) else {
                continue;
            };
            if !document.position_admits(start == 0, end == char_count) {
                continue;
            }
            let idf = ((1.0 + self.indexed_surfaces as f64) / (1.0 + document.doc_freq as f64))
                .ln()
                + 1.0;
            let length_weight = if ngram.chars().count() == 2 {
                0.85
            } else {
                1.0
            };
            let weight = idf * length_weight;
            for (feature, value) in &document.features {
                *features.entry(feature.clone()).or_insert(0.0) += weight * value;
            }
            for flag in coverage.iter_mut().take(end).skip(start) {
                *flag = true;
            }
            let top_type = document
                .features
                .iter()
                .filter(|(feature, _)| feature.starts_with("类型·"))
                .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(a.0)))
                .map(|(feature, _)| feature.trim_start_matches("类型·").to_string());
            evidence.push((weight, ngram, top_type));
            matched = true;
        }
        if !matched {
            return None;
        }
        let covered = coverage.iter().filter(|covered| **covered).count();
        let coverage_ratio = covered as f64 / char_count as f64;
        if coverage_ratio < MIN_COMPOSITION_COVERAGE {
            return None;
        }
        let total_type_weight: f64 = features
            .iter()
            .filter(|(feature, _)| feature.starts_with("类型·"))
            .map(|(_, value)| *value)
            .sum();
        let concentration = features
            .iter()
            .filter(|(feature, _)| feature.starts_with("类型·"))
            .map(|(_, value)| *value)
            .max_by(f64::total_cmp)
            .map(|value| {
                if total_type_weight > 0.0 {
                    value / total_type_weight
                } else {
                    0.0
                }
            })
            .unwrap_or(0.0);
        let etype = features
            .iter()
            .filter(|(feature, _)| feature.starts_with("类型·"))
            .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(a.0)))
            .map(|(feature, _)| feature.trim_start_matches("类型·").to_string());
        evidence.sort_by(|a, b| {
            b.0.total_cmp(&a.0)
                .then_with(|| a.1.cmp(&b.1))
                .then_with(|| a.2.cmp(&b.2))
        });
        evidence.dedup_by(|a, b| a.1 == b.1);
        let evidence = evidence
            .into_iter()
            .take(4)
            .map(|(_, ngram, etype)| match etype {
                Some(etype) => format!("子词「{}」→ {}", ngram, etype),
                None => format!("子词「{}」", ngram),
            })
            .collect();
        let support = coverage_ratio * (0.5 + 0.5 * concentration);
        Some(Composition {
            features,
            support,
            etype,
            evidence,
        })
    }

    fn is_known_partition(&self, surface: &str) -> bool {
        let chars: Vec<char> = surface.chars().collect();
        if chars.len() < 2 {
            return false;
        }
        for split in 1..chars.len() {
            let left: String = chars[..split].iter().collect();
            let right: String = chars[split..].iter().collect();
            if self.surfaces.contains_key(&left) && self.surfaces.contains_key(&right) {
                return true;
            }
        }
        false
    }

    fn build_context(&self, options: &SegmentOptions) -> ContextVector {
        let mut counts: BTreeMap<String, f64> = BTreeMap::new();
        let mut terms: Vec<String> = Vec::new();
        for term in &options.context_terms {
            let term = term.trim();
            if term.is_empty() {
                continue;
            }
            terms.push(term.to_string());
            for (feature, value) in self.surface_features(term) {
                *counts.entry(feature).or_insert(0.0) += value;
            }
        }
        terms.sort();
        terms.dedup();
        let mut expected_types: Vec<String> = options
            .expected_types
            .iter()
            .map(|etype| etype.trim().to_string())
            .filter(|etype| !etype.is_empty())
            .collect();
        expected_types.sort();
        expected_types.dedup();
        for etype in &expected_types {
            *counts.entry(format!("类型·{}", etype)).or_insert(0.0) += 1.0;
        }
        let predicate = options
            .predicate
            .as_deref()
            .map(str::trim)
            .filter(|predicate| !predicate.is_empty())
            .map(str::to_string);
        if let Some(predicate) = &predicate {
            *counts.entry(format!("谓词·{}", predicate)).or_insert(0.0) += 0.8;
        }
        let vector = SparseVector::from_counts(counts);
        let features = vector
            .top(12)
            .into_iter()
            .map(|(feature, value)| (feature.to_string(), value))
            .collect();
        ContextVector {
            snapshot: ContextSnapshot {
                terms,
                expected_types,
                predicate,
                fingerprint: vector.fingerprint(),
                features,
            },
            vector,
        }
    }

    fn build_candidate(
        &self,
        chars: &[char],
        start: usize,
        end: usize,
        jieba_spans: &HashSet<(usize, usize)>,
        context: &ContextVector,
    ) -> Option<Candidate> {
        let surface: String = chars[start..end].iter().collect();
        if surface.trim().is_empty() && surface.chars().count() > 1 {
            return None;
        }
        let mut seeds: Vec<CandidateSeed> = Vec::new();
        if let Some(readings) = self.surfaces.get(&surface) {
            for reading in readings {
                let etype = match reading.kind {
                    ReadingKind::Entity => self
                        .entities
                        .get(&reading.canonical)
                        .and_then(|profile| profile.etype.clone()),
                    ReadingKind::Type => Some(reading.canonical.clone()),
                    _ => None,
                };
                seeds.push(CandidateSeed {
                    canonical: reading.canonical.clone(),
                    kind: reading.kind,
                    etype,
                    lexicon: reading.kind.prior(),
                    composition: 0.0,
                    features: BTreeMap::new(),
                    evidence: Vec::new(),
                });
            }
        } else {
            if end - start == 1 {
                let ch = chars[start];
                let punctuation = is_punctuation(ch) || ch.is_whitespace();
                seeds.push(CandidateSeed {
                    canonical: surface.clone(),
                    kind: ReadingKind::Unknown,
                    etype: None,
                    lexicon: if punctuation {
                        PUNCT_PRIOR
                    } else {
                        UNKNOWN_CHAR_PRIOR
                    },
                    composition: 0.0,
                    features: BTreeMap::new(),
                    evidence: Vec::new(),
                });
            } else if jieba_spans.contains(&(start, end)) {
                seeds.push(CandidateSeed {
                    canonical: surface.clone(),
                    kind: ReadingKind::Unknown,
                    etype: None,
                    lexicon: JIEBA_PRIOR,
                    composition: 0.0,
                    features: BTreeMap::new(),
                    evidence: Vec::new(),
                });
            }
            if let Some(composition) = self.compose_surface(&surface) {
                let length_bonus = (COMPOSITION_LENGTH_BONUS
                    * (surface.chars().count().saturating_sub(2) as f64))
                    .min(COMPOSITION_LENGTH_CAP);
                seeds.push(CandidateSeed {
                    canonical: surface.clone(),
                    kind: ReadingKind::Composition,
                    etype: composition.etype.clone(),
                    lexicon: COMPOSITION_BASE
                        + COMPOSITION_SUPPORT_WEIGHT * composition.support
                        + length_bonus,
                    composition: composition.support,
                    features: composition.features,
                    evidence: composition.evidence,
                });
            }
        }
        if seeds.is_empty() {
            return None;
        }
        let mut readings: Vec<ScoredReading> = Vec::with_capacity(seeds.len());
        for seed in seeds {
            let feature_counts = if seed.features.is_empty() {
                self.surface_features(&seed.canonical)
            } else {
                seed.features
            };
            let mut features = SparseVector::from_counts(feature_counts);
            features.truncate(MAX_READING_FEATURES);
            let context_score = context.vector.cosine(&features);
            let schema_score = self.schema_score(seed.etype.as_deref(), &context.snapshot);
            let memory_score = self
                .memory
                .preference(
                    &surface,
                    &seed.canonical,
                    seed.kind,
                    &context.snapshot.fingerprint,
                )
                .unwrap_or(0.0);
            let length_bonus = if seed.kind == ReadingKind::Composition {
                (COMPOSITION_LENGTH_BONUS * (surface.chars().count().saturating_sub(2) as f64))
                    .min(COMPOSITION_LENGTH_CAP)
            } else {
                0.0
            };
            let score = ReadingScore {
                lexicon: seed.lexicon,
                context: context_score,
                schema: schema_score,
                memory: memory_score,
                composition: seed.composition,
                length_bonus,
                total: seed.lexicon
                    + CONTEXT_WEIGHT * context_score
                    + SCHEMA_WEIGHT * schema_score
                    + MEMORY_WEIGHT * memory_score
                    + length_bonus,
            };
            readings.push(ScoredReading {
                canonical: seed.canonical,
                kind: seed.kind,
                etype: seed.etype,
                evidence: seed.evidence,
                score,
            });
        }
        readings.sort_by(|a, b| {
            b.score
                .total
                .total_cmp(&a.score.total)
                .then_with(|| a.canonical.cmp(&b.canonical))
                .then_with(|| a.kind.cmp(&b.kind))
        });
        readings.dedup_by(|a, b| a.canonical == b.canonical && a.kind == b.kind);
        Some(Candidate {
            start,
            end,
            surface,
            readings,
        })
    }

    fn schema_score(&self, etype: Option<&str>, context: &ContextSnapshot) -> f64 {
        let expected = &context.expected_types;
        if expected.is_empty() {
            return 0.0;
        }
        let Some(etype) = etype else {
            return 0.0;
        };
        if expected.iter().any(|expected| expected == etype) {
            return 1.0;
        }
        let Some(schema) = self.schema.as_ref() else {
            return 0.0;
        };
        if expected
            .iter()
            .any(|expected| schema.types_compatible(expected, etype))
        {
            return 0.8;
        }
        if expected
            .iter()
            .any(|expected| schema.types_disjoint(expected, etype))
        {
            return -1.0;
        }
        0.0
    }
}

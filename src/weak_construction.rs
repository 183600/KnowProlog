//! WSC：双向弱监督构建（Bidirectional Weakly-Supervised Construction）。
//!
//! 已有图谱提供种子事实，语料中「种子主语 … 种子宾语」的上下文同时被归纳为
//! 解析模式与措辞模板；新候选按证据家族投票，而不是把同一条模式在十句话里的
//! 重复当成十份独立证据。未知实体只产生铸型提案，未知谓词只进入复核，硬拒绝
//! 仅来自显式否定、类型互斥与单值谓词冲突。
//!
//! 本模块只读：输出可审计的候选、证据家族、序数等级与铸型提案，不写知识库。

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::context_image::fnv1a_hex;
use crate::law_book::LawBook;
use crate::predicate_registry::PredicateRegistry;
use crate::prolog_engine::PrologFact;
use crate::relation_frame::{
    FrameEdge, FrameExtractionStats, FrameExtractor, FrameGate, Modality, Polarity, RelationFrame,
};
use crate::temporal;
use crate::type_system::{TypeSystem, TypeVerdict};

pub const CONSTRUCTION_DIALECT: &str = "knowprolog-wsc/1";
pub const DEFAULT_MIN_PATTERN_SUPPORT: usize = 2;
pub const DEFAULT_MIN_PATTERN_PURITY: u32 = 600;
pub const MIN_ENDPOINT_CHARS: usize = 2;
pub const MAX_ENDPOINT_CHARS: usize = 32;
pub const DEFAULT_MAX_SEED_EDGES: usize = 4096;
pub const MAX_RENDERED_PROPOSALS: usize = 40;

const MAX_CONTEXT_CHARS: usize = 20;
const PUNCTUATION: &[char] = &[
    '，', ',', '。', '！', '？', '；', '；', '：', '、', '.', '!', '?', ';', ':', '(', ')', '（',
    '）', '[', ']', '【', '】', '《', '》', '“', '”', '"', '\'', '\n', '\r', '\t',
];
const LEADING_DISCOURSE: &[&str] = &[
    "后来", "随后", "之后", "此前", "当时", "于是", "因此", "但是", "不过", "而且", "并且",
    "同时", "如今", "过去", "从前", "据说", "听说",
];
const INGEST_PREFIXES: &[&str] = &["录入：", "录入:", "添加：", "添加:", "新增：", "新增:", "记录：", "记录:", "导入：", "导入:"];

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceFamily {
    Grammar,
    Proposition,
    SurfacePattern,
    KnowledgeGraph,
    TypeShape,
    DialogueFeedback,
}

impl EvidenceFamily {
    pub fn label(self) -> &'static str {
        match self {
            EvidenceFamily::Grammar => "文法",
            EvidenceFamily::Proposition => "命题解析",
            EvidenceFamily::SurfacePattern => "语料模式",
            EvidenceFamily::KnowledgeGraph => "已有图谱",
            EvidenceFamily::TypeShape => "类型形状",
            EvidenceFamily::DialogueFeedback => "对话确认",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HardRebuttal {
    ExplicitNegative,
    NegativeConflict,
    FunctionalConflict,
    SchemaViolation,
    DialogueDenied,
}

impl HardRebuttal {
    pub fn label(self) -> &'static str {
        match self {
            HardRebuttal::ExplicitNegative => "显式否定",
            HardRebuttal::NegativeConflict => "否定记录冲突",
            HardRebuttal::FunctionalConflict => "单值谓词冲突",
            HardRebuttal::SchemaViolation => "类型/形状冲突",
            HardRebuttal::DialogueDenied => "对话否认",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfidenceGrade {
    A,
    B,
    C,
    D,
}

impl ConfidenceGrade {
    pub fn from_families(count: usize) -> Self {
        match count {
            0 => ConfidenceGrade::D,
            1 => ConfidenceGrade::C,
            2 => ConfidenceGrade::B,
            _ => ConfidenceGrade::A,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            ConfidenceGrade::A => "A",
            ConfidenceGrade::B => "B",
            ConfidenceGrade::C => "C",
            ConfidenceGrade::D => "D",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateDecision {
    Accepted,
    Review,
    Rejected,
}

impl CandidateDecision {
    pub fn label(self) -> &'static str {
        match self {
            CandidateDecision::Accepted => "接受",
            CandidateDecision::Review => "复核",
            CandidateDecision::Rejected => "拒绝",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConstructionOptions {
    pub min_pattern_support: usize,
    pub min_pattern_purity: u32,
    pub max_seed_edges: usize,
    pub max_candidates: usize,
}

impl Default for ConstructionOptions {
    fn default() -> Self {
        Self {
            min_pattern_support: DEFAULT_MIN_PATTERN_SUPPORT,
            min_pattern_purity: DEFAULT_MIN_PATTERN_PURITY,
            max_seed_edges: DEFAULT_MAX_SEED_EDGES,
            max_candidates: 500,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConstructionFeedback {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub confirmed: bool,
}

impl ConstructionFeedback {
    pub fn key(&self) -> String {
        format!("{}({}, {})", self.predicate, self.subject, self.object)
    }
}

pub fn parse_feedback(text: &str) -> (Vec<ConstructionFeedback>, Vec<String>) {
    let mut items = Vec::new();
    let mut errors = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() != 4 {
            errors.push(format!("第 {} 行：需要「确认/否认 主语 谓词 宾语」", index + 1));
            continue;
        }
        let confirmed = match fields[0] {
            "确认" | "confirm" | "+" => true,
            "否认" | "deny" | "-" => false,
            _ => {
                errors.push(format!("第 {} 行：未知动作「{}」", index + 1, fields[0]));
                continue;
            }
        };
        items.push(ConstructionFeedback {
            subject: fields[1].to_string(),
            predicate: fields[2].to_string(),
            object: fields[3].to_string(),
            confirmed,
        });
    }
    (items, errors)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SurfacePattern {
    pub id: String,
    pub predicate: String,
    pub prefix: String,
    pub middle: String,
    pub suffix: String,
    pub support_pairs: usize,
    pub context_pairs: usize,
    pub purity_permille: u32,
    pub active: bool,
    pub retired: bool,
    pub examples: Vec<String>,
}

impl SurfacePattern {
    pub fn render_template(&self) -> String {
        format!("{} {{subject}}{}{{object}}{}", self.predicate, self.middle, self.suffix)
    }

    pub fn render_text(&self) -> String {
        format!(
            "{}  [{}]  支持 {} 对，纯度 {:.1}%{}",
            self.render_template(),
            self.id,
            self.support_pairs,
            self.purity_permille as f64 / 10.0,
            if self.retired {
                "，已因对话否认退役"
            } else if self.active {
                "，已激活"
            } else {
                "，未激活"
            }
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MintProposal {
    pub surface: String,
    pub role: String,
    pub candidate: String,
    pub reason: String,
}

impl MintProposal {
    pub fn render_text(&self) -> String {
        format!(
            "铸造提案：{}（{}；来自 {}；{}）",
            self.surface, self.role, self.candidate, self.reason
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConstructionCandidate {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub families: BTreeSet<EvidenceFamily>,
    pub grade: ConfidenceGrade,
    pub decision: CandidateDecision,
    pub rebuttals: BTreeSet<HardRebuttal>,
    pub review_reasons: BTreeSet<String>,
    pub sentences: Vec<String>,
    pub patterns: Vec<String>,
    pub known_subject: bool,
    pub known_object: bool,
    pub known_predicate: bool,
}

impl ConstructionCandidate {
    pub fn key(&self) -> String {
        format!("{}({}, {})", self.predicate, self.subject, self.object)
    }

    pub fn families_text(&self) -> String {
        self.families
            .iter()
            .map(|family| family.label())
            .collect::<Vec<_>>()
            .join("、")
    }

    pub fn render_text(&self) -> String {
        let mut output = format!(
            "[{} {}] {} ｜ 证据家族：{}",
            self.grade.label(),
            self.decision.label(),
            self.key(),
            if self.families.is_empty() {
                "无".to_string()
            } else {
                self.families_text()
            }
        );
        if !self.rebuttals.is_empty() {
            output.push_str(&format!(
                " ｜ 硬反驳：{}",
                self.rebuttals
                    .iter()
                    .map(|item| item.label())
                    .collect::<Vec<_>>()
                    .join("、")
            ));
        }
        if !self.review_reasons.is_empty() {
            output.push_str(&format!(
                " ｜ 复核：{}",
                self.review_reasons.iter().cloned().collect::<Vec<_>>().join("；")
            ));
        }
        for sentence in self.sentences.iter().take(4) {
            output.push_str(&format!("\n    证据：{}", sentence));
        }
        output
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeneratedUtterance {
    pub candidate: String,
    pub pattern: String,
    pub text: String,
}

impl GeneratedUtterance {
    pub fn render_text(&self) -> String {
        format!("{}  ← {}（{}）", self.text, self.candidate, self.pattern)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConstructionStats {
    pub sentences: usize,
    pub seed_edges: usize,
    pub aligned_seed_edges: usize,
    pub patterns_learned: usize,
    pub patterns_active: usize,
    pub candidates: usize,
    pub accepted: usize,
    pub review: usize,
    pub rejected: usize,
    pub mint_proposals: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConstructionReport {
    pub dialect: String,
    pub stats: ConstructionStats,
    pub extraction: FrameExtractionStats,
    pub patterns: Vec<SurfacePattern>,
    pub candidates: Vec<ConstructionCandidate>,
    pub mint_proposals: Vec<MintProposal>,
    pub generations: Vec<GeneratedUtterance>,
}

impl ConstructionReport {
    pub fn render_text(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "WSC 双向弱监督构建：{} 句，种子边 {}，对齐 {}，模式 {}（激活 {}）。",
            self.stats.sentences,
            self.stats.seed_edges,
            self.stats.aligned_seed_edges,
            self.stats.patterns_learned,
            self.stats.patterns_active
        ));
        lines.push(format!(
            "候选 {}：接受 {}，复核 {}，拒绝 {}；铸型提案 {}。抽取帧 {}（解析失败 {} 句）。",
            self.stats.candidates,
            self.stats.accepted,
            self.stats.review,
            self.stats.rejected,
            self.stats.mint_proposals,
            self.extraction.frames,
            self.extraction.parse_errors
        ));
        let active: Vec<&SurfacePattern> = self.patterns.iter().filter(|item| item.active).collect();
        if !active.is_empty() {
            lines.push(String::new());
            lines.push("激活模式：".to_string());
            for pattern in active.iter().take(12) {
                lines.push(format!("  {}", pattern.render_text()));
            }
        }
        if !self.candidates.is_empty() {
            lines.push(String::new());
            lines.push("候选：".to_string());
            for candidate in &self.candidates {
                lines.push(candidate.render_text());
            }
        }
        if !self.mint_proposals.is_empty() {
            lines.push(String::new());
            lines.push("铸型提案：".to_string());
            for proposal in self.mint_proposals.iter().take(MAX_RENDERED_PROPOSALS) {
                lines.push(format!("  {}", proposal.render_text()));
            }
            if self.mint_proposals.len() > MAX_RENDERED_PROPOSALS {
                lines.push(format!(
                    "  …其余 {} 条见 --json",
                    self.mint_proposals.len() - MAX_RENDERED_PROPOSALS
                ));
            }
        }
        if !self.generations.is_empty() {
            lines.push(String::new());
            lines.push("已接受事实的措辞（不发明事实）：".to_string());
            for generation in &self.generations {
                lines.push(format!("  {}", generation.render_text()));
            }
        }
        lines.join("\n")
    }
}

pub struct WeakConstructor {
    seed_edges: BTreeSet<FrameEdge>,
    seed_predicates: BTreeSet<String>,
    nodes: BTreeSet<String>,
    entity_types: BTreeMap<String, BTreeSet<String>>,
    registry: PredicateRegistry,
    types: TypeSystem,
    gate: FrameGate,
}

impl WeakConstructor {
    pub fn new(
        facts: &[PrologFact],
        registry: PredicateRegistry,
        types: TypeSystem,
        laws: LawBook,
    ) -> Self {
        let raw_edges = edges_from_facts(facts, &registry);
        let mut seed_edges = BTreeSet::new();
        let mut seed_predicates = BTreeSet::new();
        for edge in raw_edges {
            if edge.subject.chars().count() < MIN_ENDPOINT_CHARS
                || edge.subject.chars().count() > MAX_ENDPOINT_CHARS
                || edge.object.chars().count() < MIN_ENDPOINT_CHARS
                || edge.object.chars().count() > MAX_ENDPOINT_CHARS
                || edge.subject == edge.object
            {
                continue;
            }
            seed_predicates.insert(edge.predicate.clone());
            seed_edges.insert(edge);
        }
        let nodes = nodes_from_facts(facts);
        let entity_types = entity_types_from_facts(facts);
        let gate = FrameGate::from_facts(facts, registry.clone(), types.clone(), laws);
        Self {
            seed_edges,
            seed_predicates,
            nodes,
            entity_types,
            registry,
            types,
            gate,
        }
    }

    pub fn seed_edge_count(&self) -> usize {
        self.seed_edges.len()
    }

    pub fn build(&self, corpus: &str, options: &ConstructionOptions) -> ConstructionReport {
        self.build_with_feedback(corpus, options, &[])
    }

    pub fn build_with_feedback(
        &self,
        corpus: &str,
        options: &ConstructionOptions,
        feedback: &[ConstructionFeedback],
    ) -> ConstructionReport {
        let mut feedback_map: BTreeMap<(String, String, String), bool> = BTreeMap::new();
        for item in feedback {
            let edge = canonical_edge(
                &item.subject,
                &item.predicate,
                &item.object,
                &self.registry,
            );
            feedback_map.insert(
                (edge.subject, edge.predicate, edge.object),
                item.confirmed,
            );
        }
        let sentences = split_sentences(corpus);
        let (mut patterns, aligned_edges) =
            learn_patterns(&self.seed_edges, corpus, &self.registry, options);
        let active: Vec<&SurfacePattern> = patterns.iter().filter(|item| item.active).collect();
        let (frames, extraction) = FrameExtractor::new().extract_with_stats(corpus);
        let mut candidates: BTreeMap<(String, String, String), CandidateAccumulator> =
            BTreeMap::new();

        self.absorb_frames(&frames, &mut candidates);
        self.absorb_patterns(&active, &sentences, &mut candidates);

        let mut list = Vec::new();
        let mut proposals: BTreeMap<(String, String), MintProposal> = BTreeMap::new();
        for ((subject, predicate, object), accumulator) in candidates {
            let key = (subject.clone(), predicate.clone(), object.clone());
            let candidate =
                self.finalize(&subject, &predicate, &object, accumulator, feedback_map.get(&key).copied());
            if candidate.decision != CandidateDecision::Rejected {
                for (role, surface) in [("主语", &candidate.subject), ("宾语", &candidate.object)] {
                    if self.nodes.contains(surface) {
                        continue;
                    }
                    let key = (role.to_string(), surface.clone());
                    proposals.entry(key).or_insert_with(|| MintProposal {
                        surface: surface.clone(),
                        role: role.to_string(),
                        candidate: candidate.key(),
                        reason: "候选端点不在图谱中，默认进入铸型提案而非硬拒绝".to_string(),
                    });
                }
            }
            list.push(candidate);
        }
        list.sort_by(|left, right| {
            decision_rank(left.decision)
                .cmp(&decision_rank(right.decision))
                .then_with(|| left.grade.cmp(&right.grade))
                .then_with(|| left.subject.cmp(&right.subject))
                .then_with(|| left.predicate.cmp(&right.predicate))
                .then_with(|| left.object.cmp(&right.object))
        });
        if options.max_candidates > 0 {
            list.truncate(options.max_candidates);
        }
        let mut mint_proposals: Vec<MintProposal> = proposals.into_values().collect();
        mint_proposals.sort_by(|left, right| {
            left.surface
                .cmp(&right.surface)
                .then_with(|| left.role.cmp(&right.role))
        });
        let denied_patterns: BTreeSet<String> = list
            .iter()
            .filter(|candidate| candidate.rebuttals.contains(&HardRebuttal::DialogueDenied))
            .flat_map(|candidate| candidate.patterns.iter().cloned())
            .collect();
        for pattern in &mut patterns {
            if denied_patterns.contains(&pattern.id) {
                pattern.active = false;
                pattern.retired = true;
            }
        }
        let mut generations = Vec::new();
        for candidate in &list {
            if candidate.decision != CandidateDecision::Accepted {
                continue;
            }
            let Some(pattern) = patterns.iter().find(|pattern| {
                pattern.active && pattern.predicate == candidate.predicate
            }) else {
                continue;
            };
            generations.push(GeneratedUtterance {
                candidate: candidate.key(),
                pattern: pattern.id.clone(),
                text: format!(
                    "{}{}{}{}{}",
                    pattern.prefix,
                    candidate.subject,
                    pattern.middle,
                    candidate.object,
                    pattern.suffix
                ),
            });
        }
        let mut stats = ConstructionStats {
            sentences: sentences.len(),
            seed_edges: self.seed_edges.len(),
            patterns_learned: patterns.len(),
            patterns_active: patterns.iter().filter(|item| item.active).count(),
            candidates: list.len(),
            mint_proposals: mint_proposals.len(),
            ..ConstructionStats::default()
        };
        for candidate in &list {
            match candidate.decision {
                CandidateDecision::Accepted => stats.accepted += 1,
                CandidateDecision::Review => stats.review += 1,
                CandidateDecision::Rejected => stats.rejected += 1,
            }
        }
        stats.aligned_seed_edges = aligned_edges.len();
        ConstructionReport {
            dialect: CONSTRUCTION_DIALECT.to_string(),
            stats,
            extraction,
            patterns,
            candidates: list,
            mint_proposals,
            generations,
        }
    }

    fn absorb_frames(
        &self,
        frames: &[RelationFrame],
        candidates: &mut BTreeMap<(String, String, String), CandidateAccumulator>,
    ) {
        for frame in frames {
            let edge = canonical_edge(&frame.subject, &frame.predicate, &frame.object, &self.registry);
            if edge.subject.is_empty() || edge.object.is_empty() || edge.subject == edge.object {
                continue;
            }
            let key = (
                edge.subject.clone(),
                edge.predicate.clone(),
                edge.object.clone(),
            );
            let accumulator = candidates.entry(key).or_default();
            accumulator.sentences.insert(frame.evidence.sentence.clone());
            if frame.polarity == Polarity::Negative {
                accumulator.rebuttals.insert(HardRebuttal::ExplicitNegative);
                continue;
            }
            if frame.extraction.iter().any(|path| path == "文法") {
                accumulator.families.insert(EvidenceFamily::Grammar);
            }
            if frame.extraction.iter().any(|path| path != "文法") {
                accumulator.families.insert(EvidenceFamily::Proposition);
            }
            if frame.modality != Modality::Asserted {
                accumulator.review_reasons.insert(format!(
                    "模态「{}」不进入断言事实",
                    frame.modality.label()
                ));
            }
            if let Some(temporal) = &frame.temporal {
                accumulator.review_reasons.insert(format!(
                    "时间限定「{}」只保留在证据中，不写成永远成立",
                    temporal.render_text()
                ));
            }
        }
    }

    fn absorb_patterns(
        &self,
        patterns: &[&SurfacePattern],
        sentences: &[&str],
        candidates: &mut BTreeMap<(String, String, String), CandidateAccumulator>,
    ) {
        for sentence in sentences {
            if sentence.contains('？') || sentence.contains('?') {
                continue;
            }
            for pattern in patterns {
                let Some((subject, object)) = match_pattern(pattern, sentence) else {
                    continue;
                };
                let key = (
                    subject.clone(),
                    pattern.predicate.clone(),
                    object.clone(),
                );
                let accumulator = candidates.entry(key).or_default();
                accumulator.families.insert(EvidenceFamily::SurfacePattern);
                accumulator.patterns.insert(pattern.id.clone());
                accumulator.sentences.insert((*sentence).to_string());
                if has_negation(sentence) {
                    accumulator.rebuttals.insert(HardRebuttal::ExplicitNegative);
                }
                if has_speculation(sentence) {
                    accumulator.review_reasons.insert("传闻/可能语气不进入断言事实".to_string());
                }
                if let Some(temporal) = crate::relation_frame::parse_temporal(sentence) {
                    accumulator.review_reasons.insert(format!(
                        "时间限定「{}」只保留在证据中，不写成永远成立",
                        temporal.render_text()
                    ));
                }
            }
        }
    }

    fn finalize(
        &self,
        subject: &str,
        predicate: &str,
        object: &str,
        accumulator: CandidateAccumulator,
        feedback: Option<bool>,
    ) -> ConstructionCandidate {
        let mut families = accumulator.families;
        let mut rebuttals = accumulator.rebuttals;
        let mut review_reasons = accumulator.review_reasons;
        let predicate_known =
            self.registry.is_known(predicate) || self.seed_predicates.contains(predicate);
        if !predicate_known {
            review_reasons.insert(format!("谓词「{}」未登记，需人工复核", predicate));
        }
        let frame = RelationFrame::new(subject, predicate, object);
        let gate_report = self.gate.evaluate(&frame);
        if gate_report.verdict.is_support() {
            families.insert(EvidenceFamily::KnowledgeGraph);
        }
        for finding in &gate_report.findings {
            use crate::relation_frame::GateFindingKind;
            match finding.kind {
                GateFindingKind::DirectFact | GateFindingKind::InferredFact => {
                    families.insert(EvidenceFamily::KnowledgeGraph);
                }
                GateFindingKind::NegativeConflict => {
                    rebuttals.insert(HardRebuttal::NegativeConflict);
                }
                GateFindingKind::FunctionalConflict => {
                    rebuttals.insert(HardRebuttal::FunctionalConflict);
                }
                GateFindingKind::SchemaViolation => {
                    rebuttals.insert(HardRebuttal::SchemaViolation);
                }
                _ => {}
            }
        }
        self.add_type_evidence(subject, predicate, object, &mut families, &mut rebuttals);
        match feedback {
            Some(true) => {
                families.insert(EvidenceFamily::DialogueFeedback);
            }
            Some(false) => {
                rebuttals.insert(HardRebuttal::DialogueDenied);
            }
            None => {}
        }
        let grade = if rebuttals.is_empty() {
            ConfidenceGrade::from_families(families.len())
        } else {
            ConfidenceGrade::D
        };
        let decision = if !rebuttals.is_empty() {
            CandidateDecision::Rejected
        } else if !review_reasons.is_empty() {
            CandidateDecision::Review
        } else if matches!(grade, ConfidenceGrade::A | ConfidenceGrade::B) {
            CandidateDecision::Accepted
        } else if grade == ConfidenceGrade::C {
            CandidateDecision::Review
        } else {
            CandidateDecision::Rejected
        };
        ConstructionCandidate {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
            families,
            grade,
            decision,
            rebuttals,
            review_reasons,
            sentences: accumulator
                .sentences
                .into_iter()
                .take(8)
                .collect::<Vec<_>>(),
            patterns: accumulator.patterns.into_iter().collect::<Vec<_>>(),
            known_subject: self.nodes.contains(subject),
            known_object: self.nodes.contains(object),
            known_predicate: predicate_known,
        }
    }

    fn add_type_evidence(
        &self,
        subject: &str,
        predicate: &str,
        object: &str,
        families: &mut BTreeSet<EvidenceFamily>,
        rebuttals: &mut BTreeSet<HardRebuttal>,
    ) {
        let Some(profile) = self.registry.profile(predicate) else {
            return;
        };
        let subject_types = self.entity_types.get(subject).cloned().unwrap_or_default();
        let object_types = self.entity_types.get(object).cloned().unwrap_or_default();
        let mut compatible = false;
        if !profile.domain.is_empty() && !subject_types.is_empty() {
            match self.types.sets_admit(&subject_types, &profile.domain) {
                TypeVerdict::Compatible => compatible = true,
                TypeVerdict::Disjoint => {
                    rebuttals.insert(HardRebuttal::SchemaViolation);
                }
                TypeVerdict::Unknown => {}
            }
        }
        if !profile.range.is_empty() && !object_types.is_empty() {
            match self.types.sets_admit(&object_types, &profile.range) {
                TypeVerdict::Compatible => compatible = true,
                TypeVerdict::Disjoint => {
                    rebuttals.insert(HardRebuttal::SchemaViolation);
                }
                TypeVerdict::Unknown => {}
            }
        }
        if compatible {
            families.insert(EvidenceFamily::TypeShape);
        }
    }
}

#[derive(Debug, Default)]
struct CandidateAccumulator {
    families: BTreeSet<EvidenceFamily>,
    rebuttals: BTreeSet<HardRebuttal>,
    review_reasons: BTreeSet<String>,
    patterns: BTreeSet<String>,
    sentences: BTreeSet<String>,
}

#[derive(Debug, Default)]
struct ContextEntry {
    support: BTreeMap<String, BTreeSet<String>>,
    examples: BTreeMap<String, BTreeSet<String>>,
}

fn learn_patterns(
    seed_edges: &BTreeSet<FrameEdge>,
    corpus: &str,
    registry: &PredicateRegistry,
    options: &ConstructionOptions,
) -> (Vec<SurfacePattern>, BTreeSet<String>) {
    let sentences = split_sentences(corpus);
    let mut contexts: BTreeMap<(String, String, String), ContextEntry> = BTreeMap::new();
    let mut aligned_edges = BTreeSet::new();
    for sentence in &sentences {
        let sentence = normalize_sentence(sentence);
        if sentence.is_empty() || sentence.contains('？') || sentence.contains('?') {
            continue;
        }
        for edge in seed_edges.iter().take(options.max_seed_edges) {
            let edge = canonical_edge(&edge.subject, &edge.predicate, &edge.object, registry);
            if !endpoint_ok(&edge.subject) || !endpoint_ok(&edge.object) {
                continue;
            }
            let Some((prefix, middle, suffix)) = align_seed(sentence, &edge.subject, &edge.object)
            else {
                continue;
            };
            if !valid_context(&prefix, &middle, &suffix) {
                continue;
            }
            aligned_edges.insert(format!(
                "{}({}, {})",
                edge.predicate, edge.subject, edge.object
            ));
            let entry = contexts.entry((prefix, middle, suffix)).or_default();
            let pair = format!("{}|{}", edge.subject, edge.object);
            entry
                .support
                .entry(edge.predicate.clone())
                .or_default()
                .insert(pair);
            entry
                .examples
                .entry(edge.predicate.clone())
                .or_default()
                .insert(sentence.to_string());
        }
    }

    let mut patterns = Vec::new();
    for ((prefix, middle, suffix), entry) in contexts {
        let context_pairs: usize = entry.support.values().map(BTreeSet::len).sum();
        for (predicate, pairs) in &entry.support {
            let support_pairs = pairs.len();
            let purity_permille = if context_pairs == 0 {
                0
            } else {
                (support_pairs * 1000 / context_pairs) as u32
            };
            let active = support_pairs >= options.min_pattern_support
                && purity_permille >= options.min_pattern_purity
                && discriminative_context(&middle, &suffix);
            let id_material = format!("{}|{}|{}|{}", predicate, prefix, middle, suffix);
            patterns.push(SurfacePattern {
                id: format!("wsp_{}", fnv1a_hex(id_material.as_bytes())),
                predicate: predicate.clone(),
                prefix: prefix.clone(),
                middle: middle.clone(),
                suffix: suffix.clone(),
                support_pairs,
                context_pairs,
                purity_permille,
                active,
                retired: false,
                examples: entry
                    .examples
                    .get(predicate)
                    .map(|items| items.iter().take(3).cloned().collect())
                    .unwrap_or_default(),
            });
        }
    }
    patterns.sort_by(|left, right| {
        left.predicate
            .cmp(&right.predicate)
            .then_with(|| right.support_pairs.cmp(&left.support_pairs))
            .then_with(|| left.id.cmp(&right.id))
    });
    (patterns, aligned_edges)
}

fn align_seed(sentence: &str, subject: &str, object: &str) -> Option<(String, String, String)> {
    let subject_start = sentence.find(subject)?;
    let subject_end = subject_start + subject.len();
    let object_rel = sentence[subject_end..].find(object)?;
    let object_start = subject_end + object_rel;
    let object_end = object_start + object.len();
    let prefix = clause_prefix(&sentence[..subject_start]).to_string();
    let middle = sentence[subject_end..object_start].trim().to_string();
    let suffix = clause_suffix(&sentence[object_end..]).to_string();
    Some((prefix, middle, suffix))
}

fn match_pattern(pattern: &SurfacePattern, sentence: &str) -> Option<(String, String)> {
    let sentence = normalize_sentence(sentence);
    if sentence.is_empty() || sentence.contains('？') || sentence.contains('?') {
        return None;
    }
    let subject_start = if pattern.prefix.is_empty() {
        skip_leading_discourse(sentence)
    } else {
        let found = sentence.find(&pattern.prefix)?;
        if found > 0 {
            let before = sentence[..found].chars().next_back()?;
            if !PUNCTUATION.contains(&before) && !before.is_whitespace() {
                return None;
            }
        }
        found + pattern.prefix.len()
    };
    if subject_start >= sentence.len() {
        return None;
    }
    let middle_rel = sentence[subject_start..].find(&pattern.middle)?;
    let subject_end = subject_start + middle_rel;
    let subject = sentence[subject_start..subject_end].trim();
    let middle_end = subject_end + pattern.middle.len();
    let object_end = if pattern.suffix.is_empty() {
        clause_end(&sentence[middle_end..])
            .map(|offset| middle_end + offset)
            .unwrap_or(sentence.len())
    } else {
        let suffix_rel = sentence[middle_end..].find(&pattern.suffix)?;
        middle_end + suffix_rel
    };
    let object = sentence[middle_end..object_end].trim();
    if !endpoint_ok(subject) || !endpoint_ok(object) || subject == object {
        return None;
    }
    Some((subject.to_string(), object.to_string()))
}

fn valid_context(prefix: &str, middle: &str, suffix: &str) -> bool {
    if middle.is_empty() {
        return false;
    }
    for part in [prefix, middle, suffix] {
        if part.chars().count() > MAX_CONTEXT_CHARS
            || part.contains('\n')
            || part.contains('\r')
            || part.chars().any(|ch| ch.is_whitespace())
        {
            return false;
        }
    }
    !middle.chars().any(|ch| PUNCTUATION.contains(&ch))
        && !suffix.chars().any(|ch| ch == '\u{3000}')
}

fn discriminative_context(middle: &str, suffix: &str) -> bool {
    let context_chars = middle.chars().count() + suffix.chars().count();
    if context_chars < 2 {
        return false;
    }
    if suffix.is_empty() && matches!(middle, "是" | "的" | "和" | "与" | "及" | "、") {
        return false;
    }
    true
}

fn endpoint_ok(value: &str) -> bool {
    let length = value.chars().count();
    if length < MIN_ENDPOINT_CHARS || length > MAX_ENDPOINT_CHARS {
        return false;
    }
    if value.chars().any(|ch| ch.is_whitespace() || PUNCTUATION.contains(&ch)) {
        return false;
    }
    !matches!(
        value,
        "的" | "是" | "了" | "和" | "与" | "及" | "在" | "有" | "为" | "把" | "被" | "他"
            | "她" | "它" | "这" | "那" | "其" | "我" | "你" | "您"
    )
}

fn normalize_sentence(sentence: &str) -> &str {
    let mut body = sentence.trim();
    for prefix in INGEST_PREFIXES {
        if let Some(rest) = body.strip_prefix(prefix) {
            body = rest.trim_start();
            break;
        }
    }
    body
}

fn clause_prefix(text: &str) -> &str {
    let mut start = 0;
    for (index, ch) in text.char_indices() {
        if PUNCTUATION.contains(&ch) || ch.is_whitespace() {
            start = index + ch.len_utf8();
        }
    }
    text[start..].trim()
}

fn clause_suffix(text: &str) -> &str {
    let end = text
        .char_indices()
        .find(|(_, ch)| PUNCTUATION.contains(ch) || ch.is_whitespace())
        .map(|(index, _)| index)
        .unwrap_or(text.len());
    text[..end].trim()
}

fn clause_end(text: &str) -> Option<usize> {
    text.char_indices()
        .find(|(_, ch)| matches!(ch, '，' | ',' | '；' | ';' | '：' | ':'))
        .map(|(index, _)| index)
}

fn skip_leading_discourse(sentence: &str) -> usize {
    let mut offset = 0;
    loop {
        let rest = &sentence[offset..];
        let mut matched = false;
        for marker in LEADING_DISCOURSE {
            if let Some(rest) = rest.strip_prefix(marker) {
                let leading = rest.len() - rest.trim_start_matches(['，', ',', ' ']).len();
                offset += marker.len() + leading;
                matched = true;
                break;
            }
        }
        if !matched {
            break;
        }
    }
    offset
}

fn has_negation(text: &str) -> bool {
    for cue in [
        "不是", "并非", "没有", "不属于", "不在", "未曾", "从未", "从不", "尚未", "未能", "无法",
    ] {
        if text.contains(cue) {
            return true;
        }
    }
    for (start, _) in text.match_indices('不') {
        let rest = &text[start + '不'.len_utf8()..];
        if ![
            "丹", "锈", "断", "同", "仅", "过", "久", "论", "少", "足", "错", "但", "管", "如",
            "妨", "准",
        ]
        .iter()
        .any(|suffix| rest.starts_with(suffix))
        {
            return true;
        }
    }
    false
}

fn has_speculation(text: &str) -> bool {
    ["可能", "也许", "大概", "据说", "听说", "据称", "未必", "不一定"]
        .iter()
        .any(|cue| text.contains(cue))
}

fn split_sentences(text: &str) -> Vec<&str> {
    let mut sentences = Vec::new();
    let mut start = 0usize;
    for (index, ch) in text.char_indices() {
        if matches!(ch, '。' | '！' | '？' | '；' | '!' | '?' | ';' | '\n') {
            let sentence = text[start..index].trim();
            if !sentence.is_empty() {
                sentences.push(sentence);
            }
            start = index + ch.len_utf8();
        }
    }
    let tail = text[start..].trim();
    if !tail.is_empty() {
        sentences.push(tail);
    }
    sentences
}

fn canonical_edge(
    subject: &str,
    predicate: &str,
    object: &str,
    registry: &PredicateRegistry,
) -> FrameEdge {
    let resolution = registry.resolve(predicate);
    if resolution.inverted {
        FrameEdge::new(object.trim(), resolution.canonical.trim(), subject.trim())
    } else {
        FrameEdge::new(subject.trim(), resolution.canonical.trim(), object.trim())
    }
}

fn edges_from_facts(facts: &[PrologFact], registry: &PredicateRegistry) -> Vec<FrameEdge> {
    let mut edges = Vec::new();
    for fact in facts {
        match fact {
            PrologFact::Entity {
                name,
                properties,
                ..
            } => {
                for (predicate, value) in properties {
                    if predicate.trim().is_empty()
                        || value.trim().is_empty()
                        || temporal::is_temporal_field(predicate)
                        || matches!(
                            predicate.as_str(),
                            "name" | "名称" | "type" | "类型" | "label" | "标签"
                        )
                        || value.starts_with("http://")
                        || value.starts_with("https://")
                        || value.starts_with('<')
                    {
                        continue;
                    }
                    edges.push(canonical_edge(name, predicate, value, registry));
                }
            }
            PrologFact::Relation { from, rel_type, to } => {
                edges.push(canonical_edge(from, rel_type, to, registry));
            }
        }
    }
    edges
}

fn nodes_from_facts(facts: &[PrologFact]) -> BTreeSet<String> {
    let mut nodes = BTreeSet::new();
    for fact in facts {
        match fact {
            PrologFact::Entity { name, properties, .. } => {
                nodes.insert(name.clone());
                if let Some(alias) = properties.get("name") {
                    nodes.insert(alias.clone());
                }
            }
            PrologFact::Relation { from, to, .. } => {
                nodes.insert(from.clone());
                nodes.insert(to.clone());
            }
        }
    }
    nodes
}

fn entity_types_from_facts(facts: &[PrologFact]) -> BTreeMap<String, BTreeSet<String>> {
    let mut types: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for fact in facts {
        if let PrologFact::Entity {
            name,
            entity_type,
            properties,
        } = fact
        {
            if !entity_type.is_empty() {
                types.entry(name.clone()).or_default().insert(entity_type.clone());
            }
            if let Some(alias) = properties.get("name") {
                if !entity_type.is_empty() {
                    types
                        .entry(alias.clone())
                        .or_default()
                        .insert(entity_type.clone());
                }
            }
        }
    }
    types
}

fn decision_rank(decision: CandidateDecision) -> u8 {
    match decision {
        CandidateDecision::Accepted => 0,
        CandidateDecision::Review => 1,
        CandidateDecision::Rejected => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::predicate_registry::PredicateProfileConfig;
    use crate::type_system::TypeSystem;

    fn relation(from: &str, rel_type: &str, to: &str) -> PrologFact {
        PrologFact::Relation {
            from: from.to_string(),
            rel_type: rel_type.to_string(),
            to: to.to_string(),
        }
    }

    fn entity(name: &str, entity_type: &str) -> PrologFact {
        PrologFact::Entity {
            name: name.to_string(),
            entity_type: entity_type.to_string(),
            properties: std::collections::HashMap::new(),
        }
    }

    fn registry(predicate: &str) -> PredicateRegistry {
        let mut registry = PredicateRegistry::new();
        registry.declare(&PredicateProfileConfig {
            name: predicate.to_string(),
            ..PredicateProfileConfig::default()
        });
        registry
    }

    fn constructor(facts: Vec<PrologFact>, predicate: &str) -> WeakConstructor {
        WeakConstructor::new(
            &facts,
            registry(predicate),
            TypeSystem::new(),
            LawBook::new(),
        )
    }

    #[test]
    fn learns_pattern_from_two_seed_pairs_only() {
        let facts = vec![
            relation("张三", "师傅", "李四"),
            relation("王五", "师傅", "赵六"),
            relation("钱七", "师傅", "孙八"),
        ];
        let corpus = "张三拜李四为师。\n王五拜赵六为师。\n";
        let report = constructor(facts, "师傅").build(corpus, &ConstructionOptions::default());
        let pattern = report
            .patterns
            .iter()
            .find(|item| item.predicate == "师傅" && item.middle == "拜")
            .expect("应学到“拜…为师”模式");
        assert_eq!(pattern.support_pairs, 2);
        assert_eq!(pattern.suffix, "为师");
        assert!(pattern.active);
    }

    #[test]
    fn one_seed_pair_does_not_activate_pattern() {
        let facts = vec![relation("张三", "师傅", "李四")];
        let corpus = "张三拜李四为师。\n";
        let report = constructor(facts, "师傅").build(corpus, &ConstructionOptions::default());
        assert!(report.patterns.iter().all(|item| !item.active));
    }

    #[test]
    fn pattern_only_candidate_is_review_not_accepted() {
        let facts = vec![
            relation("张三", "师傅", "李四"),
            relation("王五", "师傅", "赵六"),
        ];
        let corpus = "张三拜李四为师。\n王五拜赵六为师。\n钱七拜孙八为师。\n";
        let report = constructor(facts, "师傅").build(corpus, &ConstructionOptions::default());
        let candidate = report
            .candidates
            .iter()
            .find(|item| item.subject == "钱七" && item.object == "孙八")
            .expect("应抽出新模式候选");
        assert_eq!(candidate.families, BTreeSet::from([EvidenceFamily::SurfacePattern]));
        assert_eq!(candidate.grade, ConfidenceGrade::C);
        assert_eq!(candidate.decision, CandidateDecision::Review);
        assert!(candidate.known_subject == false);
    }

    #[test]
    fn grammar_and_pattern_accept_new_entities_with_mint_proposals() {
        let facts = vec![
            relation("张三", "师傅", "李四"),
            relation("王五", "师傅", "赵六"),
        ];
        let corpus = "张三的师傅是李四。\n王五的师傅是赵六。\n钱七的师傅是孙八。\n";
        let report = constructor(facts, "师傅").build(corpus, &ConstructionOptions::default());
        let candidate = report
            .candidates
            .iter()
            .find(|item| item.subject == "钱七" && item.object == "孙八")
            .expect("应抽出候选");
        assert!(
            candidate.families.contains(&EvidenceFamily::Grammar)
                || candidate.families.contains(&EvidenceFamily::Proposition)
        );
        assert!(candidate.families.contains(&EvidenceFamily::SurfacePattern));
        assert_eq!(candidate.grade, ConfidenceGrade::B);
        assert_eq!(candidate.decision, CandidateDecision::Accepted);
        assert!(!candidate.known_subject);
        assert!(!candidate.known_object);
        let surfaces: Vec<&str> = report
            .mint_proposals
            .iter()
            .map(|item| item.surface.as_str())
            .collect();
        assert!(surfaces.contains(&"钱七") || surfaces.contains(&"孙八"));
    }


    #[test]
    fn functional_conflict_rejects_candidate() {
        let facts = vec![
            relation("张三", "师傅", "李四"),
            relation("王五", "师傅", "赵六"),
        ];
        let mut registry = PredicateRegistry::new();
        registry.declare(&PredicateProfileConfig {
            name: "师傅".to_string(),
            functional: true,
            ..PredicateProfileConfig::default()
        });
        let constructor =
            WeakConstructor::new(&facts, registry, TypeSystem::new(), LawBook::new());
        let corpus = "张三拜李四为师。\n王五拜赵六为师。\n张三拜钱七为师。\n";
        let report = constructor.build(corpus, &ConstructionOptions::default());
        let candidate = report
            .candidates
            .iter()
            .find(|item| item.subject == "张三" && item.object == "钱七")
            .expect("应抽出冲突候选");
        assert_eq!(candidate.decision, CandidateDecision::Rejected);
        assert!(candidate.rebuttals.contains(&HardRebuttal::FunctionalConflict));
    }

    #[test]
    fn type_disjointness_rejects_candidate() {
        let mut facts = vec![
            relation("张三", "师傅", "李四"),
            relation("王五", "师傅", "赵六"),
            entity("张三", "Person"),
            entity("李四", "Person"),
            entity("王五", "Person"),
            entity("赵六", "Person"),
            entity("钱七", "Place"),
        ];
        facts.push(entity("孙八", "Person"));
        let mut registry = PredicateRegistry::new();
        registry.declare(&PredicateProfileConfig {
            name: "师傅".to_string(),
            domain: vec!["Person".to_string()],
            range: vec!["Person".to_string()],
            ..PredicateProfileConfig::default()
        });
        let mut types = TypeSystem::new();
        types.declare_disjoint("Person", "Place");
        let constructor = WeakConstructor::new(&facts, registry, types, LawBook::new());
        let corpus = "张三的师傅是李四。\n王五的师傅是赵六。\n钱七的师傅是孙八。\n";
        let report = constructor.build(corpus, &ConstructionOptions::default());
        let candidate = report
            .candidates
            .iter()
            .find(|item| item.subject == "钱七" && item.object == "孙八")
            .expect("应抽出类型冲突候选");
        assert_eq!(candidate.decision, CandidateDecision::Rejected);
        assert!(candidate.rebuttals.contains(&HardRebuttal::SchemaViolation));
    }

    #[test]
    fn ambiguous_context_is_not_activated() {
        let facts = vec![
            relation("张三", "师傅", "李四"),
            relation("王五", "师傅", "赵六"),
            relation("张三", "老师", "李四"),
            relation("王五", "老师", "赵六"),
        ];
        let corpus = "张三的同伴是李四。\n王五的同伴是赵六。\n";
        let constructor = WeakConstructor::new(
            &facts,
            PredicateRegistry::new(),
            TypeSystem::new(),
            LawBook::new(),
        );
        let report = constructor.build(corpus, &ConstructionOptions::default());
        assert!(report.patterns.iter().all(|item| !item.active));
    }

    #[test]
    fn aligned_seed_edges_are_counted_once() {
        let facts = vec![
            relation("张三", "师傅", "李四"),
            relation("王五", "师傅", "赵六"),
        ];
        let corpus = "张三拜李四为师。\n张三拜李四为师。\n王五拜赵六为师。\n";
        let report = constructor(facts, "师傅").build(corpus, &ConstructionOptions::default());
        assert_eq!(report.stats.aligned_seed_edges, 2);
    }


    #[test]
    fn accepted_candidate_can_be_rendered_from_learned_pattern() {
        let facts = vec![
            relation("张三", "师傅", "李四"),
            relation("王五", "师傅", "赵六"),
        ];
        let corpus = "张三的师傅是李四。\n王五的师傅是赵六。\n钱七的师傅是孙八。\n";
        let report = constructor(facts, "师傅").build(corpus, &ConstructionOptions::default());
        let generated = report
            .generations
            .iter()
            .find(|item| item.candidate.contains("钱七"))
            .expect("已接受候选应能由模式生成措辞");
        assert_eq!(generated.text, "钱七的师傅是孙八");
    }


    #[test]
    fn temporal_scope_forces_review() {
        let facts = vec![
            relation("张三", "师傅", "李四"),
            relation("王五", "师傅", "赵六"),
        ];
        let corpus = "张三的师傅是李四。\n王五的师傅是赵六。\n1991年，钱七的师傅是孙八。\n";
        let report = constructor(facts, "师傅").build(corpus, &ConstructionOptions::default());
        let candidate = report
            .candidates
            .iter()
            .find(|item| item.subject == "钱七")
            .expect("应抽出带时间的候选");
        assert_eq!(candidate.decision, CandidateDecision::Review);
        assert!(
            candidate
                .review_reasons
                .iter()
                .any(|reason| reason.contains("时间限定"))
        );
    }


    #[test]
    fn dialogue_confirmation_promotes_pattern_candidate() {
        let facts = vec![
            relation("张三", "师傅", "李四"),
            relation("王五", "师傅", "赵六"),
        ];
        let corpus = "张三拜李四为师。\n王五拜赵六为师。\n钱七拜孙八为师。\n";
        let report = constructor(facts.clone(), "师傅").build(corpus, &ConstructionOptions::default());
        assert!(
            report
                .candidates
                .iter()
                .any(|item| item.subject == "钱七" && item.decision == CandidateDecision::Review)
        );
        let feedback = vec![ConstructionFeedback {
            subject: "钱七".to_string(),
            predicate: "师傅".to_string(),
            object: "孙八".to_string(),
            confirmed: true,
        }];
        let confirmed = constructor(facts, "师傅").build_with_feedback(
            corpus,
            &ConstructionOptions::default(),
            &feedback,
        );
        let candidate = confirmed
            .candidates
            .iter()
            .find(|item| item.subject == "钱七")
            .expect("应有候选");
        assert!(candidate.families.contains(&EvidenceFamily::DialogueFeedback));
        assert_eq!(candidate.grade, ConfidenceGrade::B);
        assert_eq!(candidate.decision, CandidateDecision::Accepted);
    }

    #[test]
    fn dialogue_denial_rejects_candidate() {
        let facts = vec![
            relation("张三", "师傅", "李四"),
            relation("王五", "师傅", "赵六"),
        ];
        let corpus = "张三的师傅是李四。\n王五的师傅是赵六。\n钱七的师傅是孙八。\n";
        let feedback = vec![ConstructionFeedback {
            subject: "钱七".to_string(),
            predicate: "师傅".to_string(),
            object: "孙八".to_string(),
            confirmed: false,
        }];
        let report = constructor(facts, "师傅").build_with_feedback(
            corpus,
            &ConstructionOptions::default(),
            &feedback,
        );
        let candidate = report
            .candidates
            .iter()
            .find(|item| item.subject == "钱七")
            .expect("应有候选");
        assert_eq!(candidate.decision, CandidateDecision::Rejected);
        assert!(candidate.rebuttals.contains(&HardRebuttal::DialogueDenied));
    }

    #[test]
    fn feedback_parser_reports_bad_lines() {
        let (items, errors) = parse_feedback("确认 张三 师傅 李四\n?? 王五 师傅 赵六\n否认 张三 师傅 王五\n");
        assert_eq!(items.len(), 2);
        assert_eq!(errors.len(), 1);
        assert!(items[0].confirmed);
        assert!(!items[1].confirmed);
    }


    #[test]
    fn denial_retires_pattern() {
        let facts = vec![
            relation("张三", "师傅", "李四"),
            relation("王五", "师傅", "赵六"),
        ];
        let corpus = "张三拜李四为师。\n王五拜赵六为师。\n钱七拜孙八为师。\n";
        let feedback = vec![ConstructionFeedback {
            subject: "钱七".to_string(),
            predicate: "师傅".to_string(),
            object: "孙八".to_string(),
            confirmed: false,
        }];
        let report = constructor(facts, "师傅").build_with_feedback(
            corpus,
            &ConstructionOptions::default(),
            &feedback,
        );
        assert!(report.patterns.iter().any(|item| item.retired && !item.active));
        assert!(report.generations.is_empty());
    }

    #[test]
    fn explicit_negation_is_hard_rebuttal() {
        let facts = vec![
            relation("张三", "师傅", "李四"),
            relation("王五", "师傅", "赵六"),
        ];
        let corpus = "张三的师傅是李四。\n王五的师傅是赵六。\n钱七不是孙八的师傅。\n";
        let report = constructor(facts, "师傅").build(corpus, &ConstructionOptions::default());
        assert!(
            report.candidates.iter().all(|item| {
                !(item.subject == "钱七" && item.object == "孙八"
                    && item.decision == CandidateDecision::Accepted)
            }),
            "否定句不应被接受：{:?}",
            report.candidates
        );
    }
}

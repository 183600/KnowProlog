//! 批量教学：从种子事实与自然语料归纳属性抽取规则。
//!
//! 一条规则只由「主语槽 + 中间词面 + 取值槽」组成，先作为候选落盘；
//! 只有显式采纳的候选才进入活动规则集，并被普通文本录入复用。

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs;
use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::context_image::fnv1a_hex;
use crate::nlp::{merge_name_fragments, normalize_words, preprocess, TaggedWord};
use crate::{KnowledgeExtractor, KnowledgeItem, SystemConfig};

pub const BATCH_DIALECT: &str = "knowprolog-batch-teach/1";
pub const LEDGER_FILE: &str = "batch_teaching.json";
pub const DEFAULT_MIN_SUPPORT: usize = 2;
pub const DEFAULT_MIN_PURITY: u32 = 800;
pub const DEFAULT_MAX_CANDIDATES: usize = 64;
pub const DEFAULT_MAX_EVIDENCE: usize = 3;
pub const MAX_EPISODES: usize = 16;
pub const MAX_ACTIVE_RULES: usize = 16;

const INGEST_PREFIXES: &[&str] = &[
    "录入：", "录入:", "添加：", "添加:", "新增：", "新增:", "记录：", "记录:", "导入：", "导入:",
];
const FUNCTION_POS: &[&str] = &[
    "u", "p", "c", "d", "r", "y", "e", "o", "f", "t", "q", "m", "v", "a", "ad", "an", "vd", "vn",
    "ul", "uj", "uz", "ud", "uv",
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeedFact {
    pub subject: String,
    pub predicate: String,
    pub value: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entity_type: Option<String>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkippedFact {
    pub fact: String,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RuleAtom {
    Subject { pos: String },
    Literal { surface: String },
    Value { pos: String },
}

impl RuleAtom {
    fn render(&self) -> String {
        match self {
            RuleAtom::Subject { pos } => format!("{{主语:{}}}", pos),
            RuleAtom::Literal { surface } => surface.clone(),
            RuleAtom::Value { pos } => format!("{{取值:{}}}", pos),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuleDecision {
    Ready,
    Review,
    Rejected,
}

impl RuleDecision {
    pub fn label(self) -> &'static str {
        match self {
            RuleDecision::Ready => "可采纳",
            RuleDecision::Review => "复核",
            RuleDecision::Rejected => "拒绝",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuleStatus {
    Candidate,
    Active,
    Retired,
}

impl RuleStatus {
    pub fn label(self) -> &'static str {
        match self {
            RuleStatus::Candidate => "候选",
            RuleStatus::Active => "已采纳",
            RuleStatus::Retired => "已退役",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LearnedRule {
    pub id: String,
    pub predicate: String,
    pub entity_type: String,
    pub subject_property: String,
    pub value_property: String,
    pub atoms: Vec<RuleAtom>,
    pub support: usize,
    pub matches: usize,
    pub purity: u32,
    pub evidence: Vec<String>,
    pub unsupported: Vec<String>,
    pub negative_hits: Vec<String>,
    pub decision: RuleDecision,
    pub status: RuleStatus,
    pub reasons: Vec<String>,
}

impl LearnedRule {
    pub fn render_pattern(&self) -> String {
        self.atoms
            .iter()
            .map(RuleAtom::render)
            .collect::<Vec<_>>()
            .join(" ")
    }

    pub fn render_text(&self) -> String {
        let mut text = format!(
            "[{}] {} → {}.{} / {}.{}（支持 {}，命中 {}，纯度 {}‰）",
            self.status.label(),
            self.render_pattern(),
            self.entity_type,
            self.subject_property,
            self.entity_type,
            self.value_property,
            self.support,
            self.matches,
            self.purity
        );
        if !self.reasons.is_empty() {
            text.push_str(&format!("；{}", self.reasons.join("；")));
        }
        for sentence in self.evidence.iter().take(2) {
            text.push_str(&format!("\n    证据：{}", sentence));
        }
        for sentence in self.negative_hits.iter().take(1) {
            text.push_str(&format!("\n    反例：{}", sentence));
        }
        text
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TeachingEpisode {
    pub id: String,
    pub facts_digest: String,
    pub corpus_digest: String,
    pub negatives_digest: String,
    pub seed_facts: usize,
    pub usable_facts: usize,
    pub sentences: usize,
    pub candidates: Vec<LearnedRule>,
    pub skipped: Vec<SkippedFact>,
}

impl TeachingEpisode {
    pub fn ready_count(&self) -> usize {
        self.candidates
            .iter()
            .filter(|rule| rule.decision == RuleDecision::Ready)
            .count()
    }

    pub fn render_text(&self) -> String {
        let mut text = format!(
            "批量教学 {}：种子事实 {} 条（可用 {}），语料 {} 句，候选 {} 条（可采纳 {} 条）。",
            &self.id[..self.id.len().min(12)],
            self.seed_facts,
            self.usable_facts,
            self.sentences,
            self.candidates.len(),
            self.ready_count()
        );
        if !self.skipped.is_empty() {
            text.push_str(&format!("\n跳过 {} 条种子事实：", self.skipped.len()));
            for skipped in self.skipped.iter().take(5) {
                text.push_str(&format!("\n  - {}（{}）", skipped.fact, skipped.reason));
            }
        }
        if self.candidates.is_empty() {
            text.push_str("\n没有归纳出候选规则。");
            return text;
        }
        let mut rules = self.candidates.iter().collect::<Vec<_>>();
        rules.sort_by(|a, b| {
            a.decision
                .cmp(&b.decision)
                .then_with(|| b.support.cmp(&a.support))
                .then_with(|| a.id.cmp(&b.id))
        });
        for rule in rules {
            text.push_str(&format!("\n  {}", rule.render_text().replace('\n', "\n  ")));
        }
        text
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchOptions {
    pub min_support: usize,
    pub min_purity: u32,
    pub max_candidates: usize,
    pub max_evidence: usize,
}

impl Default for BatchOptions {
    fn default() -> Self {
        Self {
            min_support: DEFAULT_MIN_SUPPORT,
            min_purity: DEFAULT_MIN_PURITY,
            max_candidates: DEFAULT_MAX_CANDIDATES,
            max_evidence: DEFAULT_MAX_EVIDENCE,
        }
    }
}

pub struct BatchTeacher {
    config: Arc<SystemConfig>,
}

impl BatchTeacher {
    pub fn new(config: Arc<SystemConfig>) -> Self {
        Self { config }
    }

    pub fn teach(
        &self,
        facts_text: &str,
        corpus_text: &str,
        negatives_text: &str,
        options: &BatchOptions,
    ) -> TeachingEpisode {
        let (facts, parse_errors) = parse_seed_facts(facts_text);
        let mut skipped = parse_errors
            .into_iter()
            .map(|reason| SkippedFact {
                fact: String::new(),
                reason,
            })
            .collect::<Vec<_>>();
        let corpus = split_sentences(corpus_text);
        let corpus_tokens = corpus
            .iter()
            .map(|sentence| (sentence.clone(), tokenize(sentence, &self.config)))
            .collect::<Vec<_>>();
        let negative_sentences = split_sentences(negatives_text);

        let mut groups: BTreeMap<RuleKey, Group> = BTreeMap::new();
        let mut usable = 0usize;
        for fact in &facts {
            match self.induce_fact(fact, &corpus_tokens) {
                Ok((key, subject, value, sentence)) => {
                    usable += 1;
                    let group = groups.entry(key).or_default();
                    group
                        .examples
                        .entry((subject.clone(), value.clone()))
                        .or_insert(sentence);
                    group.pairs.insert((subject, value));
                }
                Err(reason) => skipped.push(SkippedFact {
                    fact: fact_label(fact),
                    reason,
                }),
            }
        }

        let mut candidates = groups
            .into_iter()
            .map(|(key, group)| {
                self.evaluate(key, group, &corpus_tokens, &negative_sentences, options)
            })
            .collect::<Vec<_>>();

        candidates.sort_by(|a, b| {
            a.decision
                .cmp(&b.decision)
                .then_with(|| b.support.cmp(&a.support))
                .then_with(|| a.id.cmp(&b.id))
        });
        if candidates.len() > options.max_candidates.max(1) {
            for rule in candidates.drain(options.max_candidates.max(1)..) {
                skipped.push(SkippedFact {
                    fact: rule.render_pattern(),
                    reason: "候选数量超过上限，未纳入本轮".to_string(),
                });
            }
        }
        candidates.sort_by(|a, b| a.id.cmp(&b.id));

        let id_seed = format!(
            "{}\n{}\n{}",
            facts_text,
            corpus_text,
            format!("{}\n{:?}", negatives_text, options)
        );
        let id = format!("bt-{}", &fnv1a_hex(id_seed.as_bytes())[..12]);
        TeachingEpisode {
            id,
            facts_digest: fnv1a_hex(facts_text.as_bytes()),
            corpus_digest: fnv1a_hex(corpus_text.as_bytes()),
            negatives_digest: fnv1a_hex(negatives_text.as_bytes()),
            seed_facts: facts.len(),
            usable_facts: usable,
            sentences: corpus.len(),
            candidates,
            skipped,
        }
    }

    fn induce_fact(
        &self,
        fact: &SeedFact,
        corpus: &[(String, Vec<TaggedWord>)],
    ) -> Result<(RuleKey, String, String, String), String> {
        if fact.subject.is_empty() || fact.predicate.is_empty() || fact.value.is_empty() {
            return Err("主语、谓词、取值都不能为空".to_string());
        }
        let (entity_type, subject_property, value_property) =
            resolve_property(&self.config, &fact.predicate, fact.entity_type.as_deref())
                .ok_or_else(|| format!("属性「{}」未在 entity_types 中声明", fact.predicate))?;
        for (sentence, tokens) in corpus {
            let subject_spans = find_surface_spans(tokens, &fact.subject, 4);
            let value_spans = find_surface_spans(tokens, &fact.value, 1);
            if subject_spans.len() != 1 || value_spans.len() != 1 {
                continue;
            }
            let (_subject_start, subject_end) = subject_spans[0];
            let (value_start, _value_end) = value_spans[0];
            if subject_end > value_start {
                continue;
            }
            let middle = &tokens[subject_end..value_start];
            if middle.is_empty() {
                continue;
            }
            let mut atoms = vec![RuleAtom::Subject {
                pos: "*".to_string(),
            }];
            for token in middle {
                atoms.push(RuleAtom::Literal {
                    surface: token.text.clone(),
                });
            }
            atoms.push(RuleAtom::Value {
                pos: tokens[value_start].pos.clone(),
            });
            let key = RuleKey {
                predicate: canonical_predicate(&self.config, &fact.predicate),
                entity_type,
                subject_property,
                value_property,
                atoms,
            };
            return Ok((
                key,
                fact.subject.clone(),
                fact.value.clone(),
                sentence.clone(),
            ));
        }
        Err("语料中没有同一句内、主语在前且取值是单个 token 的句子".to_string())
    }

    fn evaluate(
        &self,
        key: RuleKey,
        group: Group,
        corpus: &[(String, Vec<TaggedWord>)],
        negatives: &[String],
        options: &BatchOptions,
    ) -> LearnedRule {
        let mut matched_pairs: BTreeSet<(String, String)> = BTreeSet::new();
        let mut matches = 0usize;
        let mut supported_matches = 0usize;
        let mut evidence = Vec::new();
        let mut unsupported = Vec::new();
        let mut negative_hits = Vec::new();
        for (sentence, tokens) in corpus {
            let found = match_atoms(&key.atoms, tokens)
                .into_iter()
                .collect::<BTreeSet<_>>();
            for (subject, value) in found {
                matches += 1;
                let pair = (subject.clone(), value.clone());
                if group.pairs.contains(&pair) {
                    supported_matches += 1;
                    matched_pairs.insert(pair);
                    if evidence.len() < options.max_evidence.max(1) {
                        push_unique(&mut evidence, sentence.clone());
                    }
                } else if unsupported.len() < options.max_evidence.max(1) {
                    push_unique(&mut unsupported, sentence.clone());
                }
            }
        }
        for sentence in negatives {
            let tokens = tokenize(sentence, &self.config);
            if !match_atoms(&key.atoms, &tokens).is_empty() && negative_hits.len() < 3 {
                push_unique(&mut negative_hits, sentence.clone());
            }
        }
        let support = matched_pairs.len();
        let purity = if matches == 0 {
            0
        } else {
            ((supported_matches * 1000) / matches) as u32
        };
        let mut reasons = Vec::new();
        let mut decision = RuleDecision::Ready;
        if !negative_hits.is_empty() {
            decision = RuleDecision::Rejected;
            reasons.push(format!("命中 {} 条显式反例", negative_hits.len()));
        } else if support < options.min_support {
            decision = RuleDecision::Review;
            reasons.push(format!(
                "支持 {} 条，低于门槛 {}",
                support, options.min_support
            ));
        } else if purity < options.min_purity.min(1000) {
            decision = RuleDecision::Review;
            reasons.push(format!(
                "纯度 {}‰，低于门槛 {}‰",
                purity,
                options.min_purity.min(1000)
            ));
        }
        if decision != RuleDecision::Rejected {
            let (covered, total) = self.base_coverage(&key, &group);
            if total > 0 && covered == total {
                decision = RuleDecision::Review;
                reasons.push("现有抽取规则已覆盖全部支持句".to_string());
            } else if covered > 0 {
                reasons.push(format!("现有抽取规则已覆盖 {}/{} 支持句", covered, total));
            }
        }
        let id_seed = format!(
            "{}\n{}\n{}\n{}\n{}",
            key.predicate,
            key.entity_type,
            key.subject_property,
            key.value_property,
            key.atoms
                .iter()
                .map(RuleAtom::render)
                .collect::<Vec<_>>()
                .join(" ")
        );
        LearnedRule {
            id: format!("bt-{}", &fnv1a_hex(id_seed.as_bytes())[..8]),
            predicate: key.predicate,
            entity_type: key.entity_type,
            subject_property: key.subject_property,
            value_property: key.value_property,
            atoms: key.atoms,
            support,
            matches,
            purity,
            evidence,
            unsupported,
            negative_hits,
            decision,
            status: RuleStatus::Candidate,
            reasons,
        }
    }

    fn base_coverage(&self, key: &RuleKey, group: &Group) -> (usize, usize) {
        let extractor = KnowledgeExtractor::new(self.config.clone());
        let total = group.pairs.len();
        let mut covered = 0usize;
        for ((subject, value), sentence) in &group.examples {
            let tokens = tokenize(sentence, &self.config);
            let items = extractor.extract(sentence, &tokens);
            let found = items.iter().any(|item| {
                matches!(
                    item,
                    KnowledgeItem::Entity { label, properties }
                        if label == &key.entity_type
                            && properties.get(&key.subject_property) == Some(subject)
                            && properties.get(&key.value_property) == Some(value)
                )
            });
            if found {
                covered += 1;
            }
        }
        (covered, total)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct RuleKey {
    predicate: String,
    entity_type: String,
    subject_property: String,
    value_property: String,
    atoms: Vec<RuleAtom>,
}

impl RuleAtom {
    fn sort_key(&self) -> (u8, String, String) {
        match self {
            RuleAtom::Subject { pos } => (0, pos.clone(), String::new()),
            RuleAtom::Literal { surface } => (1, surface.clone(), String::new()),
            RuleAtom::Value { pos } => (2, pos.clone(), String::new()),
        }
    }
}

impl PartialOrd for RuleAtom {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RuleAtom {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.sort_key().cmp(&other.sort_key())
    }
}

#[derive(Debug, Default)]
struct Group {
    pairs: BTreeSet<(String, String)>,
    examples: BTreeMap<(String, String), String>,
}

pub fn parse_seed_facts(text: &str) -> (Vec<SeedFact>, Vec<String>) {
    let mut facts = Vec::new();
    let mut errors = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with('{') {
            match serde_json::from_str::<SeedFact>(line) {
                Ok(fact) => facts.push(fact),
                Err(error) => errors.push(format!("第 {} 行 JSON 无效：{}", index + 1, error)),
            }
            continue;
        }
        let fields = if line.contains('\t') {
            line.split('\t').map(str::trim).collect::<Vec<_>>()
        } else {
            line.split_whitespace().collect::<Vec<_>>()
        };
        if !(3..=4).contains(&fields.len()) {
            errors.push(format!(
                "第 {} 行需要 3~4 列（主语 谓词 取值 [实体类型]）",
                index + 1
            ));
            continue;
        }
        if fields[0].is_empty() || fields[1].is_empty() || fields[2].is_empty() {
            errors.push(format!("第 {} 行存在空字段", index + 1));
            continue;
        }
        facts.push(SeedFact {
            subject: fields[0].to_string(),
            predicate: fields[1].to_string(),
            value: fields[2].to_string(),
            entity_type: fields.get(3).map(|value| value.to_string()),
            source: format!("第 {} 行", index + 1),
        });
    }
    (facts, errors)
}

pub fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if matches!(ch, '。' | '！' | '？' | '!' | '?' | ';' | '；' | '\n' | '\r') {
            let sentence = current.trim();
            if !sentence.is_empty() {
                sentences.push(strip_ingest_prefix(sentence));
            }
            current.clear();
        } else {
            current.push(ch);
        }
    }
    let sentence = current.trim();
    if !sentence.is_empty() {
        sentences.push(strip_ingest_prefix(sentence));
    }
    sentences
}

fn strip_ingest_prefix(sentence: &str) -> String {
    for prefix in INGEST_PREFIXES {
        if let Some(rest) = sentence.strip_prefix(prefix) {
            return rest.trim().to_string();
        }
    }
    sentence.to_string()
}

fn canonical_predicate(config: &SystemConfig, predicate: &str) -> String {
    for profile in &config.predicate_profiles {
        if profile.name == predicate || profile.aliases.iter().any(|alias| alias == predicate) {
            return profile.name.clone();
        }
    }
    predicate.to_string()
}

fn resolve_property(
    config: &SystemConfig,
    predicate: &str,
    explicit_type: Option<&str>,
) -> Option<(String, String, String)> {
    let canonical = canonical_predicate(config, predicate);
    for entity_type in &config.entity_types {
        if let Some(expected) = explicit_type {
            if entity_type.label != expected {
                continue;
            }
        }
        let Some(value_property) = entity_type
            .properties
            .iter()
            .find(|property| property.name == predicate || property.name == canonical)
        else {
            continue;
        };
        if value_property.name == "name" {
            continue;
        }
        let subject_property = entity_type
            .properties
            .iter()
            .find(|property| property.name == "name")
            .or_else(|| entity_type.properties.iter().find(|property| property.required))
            .or_else(|| {
                entity_type
                    .properties
                    .iter()
                    .find(|property| property.name != value_property.name)
            })?;
        if subject_property.name == value_property.name {
            continue;
        }
        return Some((
            entity_type.label.clone(),
            subject_property.name.clone(),
            value_property.name.clone(),
        ));
    }
    None
}

fn tokenize(text: &str, config: &SystemConfig) -> Vec<TaggedWord> {
    let words = merge_name_fragments(&preprocess(text, &config.nlp));
    match &config.synonyms {
        Some(synonyms) => normalize_words(words, &synonyms.mappings),
        None => words,
    }
}

fn find_surface_spans(
    tokens: &[TaggedWord],
    surface: &str,
    max_tokens: usize,
) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    for start in 0..tokens.len() {
        let mut text = String::new();
        for end in start..tokens.len().min(start + max_tokens) {
            if !span_token_ok(&tokens[end]) {
                break;
            }
            text.push_str(&tokens[end].text);
            if text == surface {
                spans.push((start, end + 1));
                break;
            }
            if !surface.starts_with(&text) {
                break;
            }
        }
    }
    spans
}

fn span_token_ok(token: &TaggedWord) -> bool {
    !token.text.is_empty()
        && !token
            .text
            .chars()
            .all(|ch| "，。、；：？！「」『』《》【】（）()[]{}<>\"'`~,.;:!?-_|/\\".contains(ch))
}

fn slot_start_ok(token: &TaggedWord, expected: &str) -> bool {
    if expected.is_empty() || expected == "*" {
        return !FUNCTION_POS.contains(&token.pos.as_str());
    }
    pos_matches(token, expected)
}

fn pos_matches(token: &TaggedWord, expected: &str) -> bool {
    if expected.is_empty() || expected == "*" || expected == "unknown" {
        return !FUNCTION_POS.contains(&token.pos.as_str());
    }
    token.pos == expected
}

fn match_atoms(atoms: &[RuleAtom], tokens: &[TaggedWord]) -> Vec<(String, String)> {
    if atoms.is_empty() {
        return Vec::new();
    }
    let mut matches = Vec::new();
    for start in 0..tokens.len() {
        let subject_expected = match atoms.first() {
            Some(RuleAtom::Subject { pos }) => pos,
            _ => return Vec::new(),
        };
        if !slot_start_ok(&tokens[start], subject_expected) {
            continue;
        }
        let max_end = tokens.len().min(start + 4);
        for subject_end in start + 1..=max_end {
            if !tokens[start..subject_end].iter().all(span_token_ok) {
                break;
            }
            let mut position = subject_end;
            let mut value = None;
            let mut matched = true;
            for atom in &atoms[1..] {
                if position >= tokens.len() {
                    matched = false;
                    break;
                }
                match atom {
                    RuleAtom::Subject { .. } => {
                        matched = false;
                        break;
                    }
                    RuleAtom::Literal { surface } => {
                        if tokens[position].text != *surface {
                            matched = false;
                            break;
                        }
                        position += 1;
                    }
                    RuleAtom::Value { pos } => {
                        if !pos_matches(&tokens[position], pos) {
                            matched = false;
                            break;
                        }
                        value = Some(tokens[position].text.clone());
                        position += 1;
                        let _ = position;
                        break;
                    }
                }
            }
            if matched {
                if let Some(value) = value {
                    let subject = tokens[start..subject_end]
                        .iter()
                        .map(|token| token.text.as_str())
                        .collect::<String>();
                    matches.push((subject, value));
                }
            }
        }
    }
    matches
}

fn push_unique(items: &mut Vec<String>, value: String) {
    if !items.contains(&value) {
        items.push(value);
    }
}

fn fact_label(fact: &SeedFact) -> String {
    format!("{} {} {}", fact.subject, fact.predicate, fact.value)
}

#[derive(Debug)]
pub enum LedgerLoad {
    Missing,
    Loaded(BatchTeachingLedger),
    Damaged(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchTeachingLedger {
    pub dialect: String,
    pub episodes: Vec<TeachingEpisode>,
}

impl Default for BatchTeachingLedger {
    fn default() -> Self {
        Self {
            dialect: BATCH_DIALECT.to_string(),
            episodes: Vec::new(),
        }
    }
}

impl BatchTeachingLedger {
    pub fn load(path: &str) -> LedgerLoad {
        match fs::read_to_string(path) {
            Ok(text) => match serde_json::from_str::<BatchTeachingLedger>(&text) {
                Ok(ledger) if ledger.dialect == BATCH_DIALECT => LedgerLoad::Loaded(ledger),
                Ok(ledger) => LedgerLoad::Damaged(format!(
                    "方言不匹配：{}（期望 {}）",
                    ledger.dialect, BATCH_DIALECT
                )),
                Err(error) => LedgerLoad::Damaged(format!("解析失败：{}", error)),
            },
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => LedgerLoad::Missing,
            Err(error) => LedgerLoad::Damaged(format!("读取失败：{}", error)),
        }
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        if let Some(parent) = Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        let text = serde_json::to_string_pretty(self).map_err(std::io::Error::other)?;
        fs::write(path, text)
    }

    pub fn record(&mut self, episode: TeachingEpisode) {
        if let Some(existing) = self
            .episodes
            .iter_mut()
            .find(|existing| existing.id == episode.id)
        {
            *existing = episode;
        } else {
            self.episodes.push(episode);
        }
        if self.episodes.len() > MAX_EPISODES {
            let excess = self.episodes.len() - MAX_EPISODES;
            self.episodes.drain(..excess);
        }
    }

    pub fn active_rules(&self) -> Vec<&LearnedRule> {
        let mut seen = HashSet::new();
        let mut rules = Vec::new();
        for episode in self.episodes.iter().rev() {
            for rule in &episode.candidates {
                if rule.status == RuleStatus::Active && seen.insert(rule.id.clone()) {
                    rules.push(rule);
                }
            }
        }
        rules.sort_by(|a, b| a.id.cmp(&b.id));
        rules
    }

    pub fn adopt(&mut self, selector: &str, config: &SystemConfig) -> Result<Vec<String>, String> {
        let selectors = self.select_targets(selector)?;
        let active_ids = self
            .active_rules()
            .into_iter()
            .map(|rule| rule.id.clone())
            .collect::<BTreeSet<_>>();
        let additions = selectors
            .iter()
            .filter(|id| !active_ids.contains(*id))
            .count();
        if active_ids.len() + additions > MAX_ACTIVE_RULES {
            return Err(format!(
                "活动规则上限为 {} 条，当前已有 {} 条，本次尝试新增 {} 条",
                MAX_ACTIVE_RULES,
                active_ids.len(),
                additions
            ));
        }
        let mut adopted = Vec::new();
        for id in selectors {
            let latest = self
                .latest_rule(&id)
                .ok_or_else(|| format!("找不到规则 {}", id))?;
            if !rule_declared(config, latest) {
                return Err(format!(
                    "规则 {} 指向的实体类型或属性已不在配置中，不能采纳",
                    id
                ));
            }
            if latest.decision != RuleDecision::Ready {
                return Err(format!(
                    "规则 {} 当前为「{}」，不能采纳：{}",
                    id,
                    latest.decision.label(),
                    latest.reasons.join("；")
                ));
            }
            for episode in &mut self.episodes {
                for rule in &mut episode.candidates {
                    if rule.id == id {
                        rule.status = RuleStatus::Active;
                    }
                }
            }
            adopted.push(id);
        }
        Ok(adopted)
    }

    pub fn retire(&mut self, selector: &str) -> Result<Vec<String>, String> {
        let selectors = self.select_targets(selector)?;
        for episode in &mut self.episodes {
            for rule in &mut episode.candidates {
                if selectors.contains(&rule.id) {
                    rule.status = RuleStatus::Retired;
                }
            }
        }
        Ok(selectors.into_iter().collect())
    }

    pub fn extract_active(
        &self,
        config: &SystemConfig,
        text: &str,
        tagged_words: &[TaggedWord],
    ) -> Vec<KnowledgeItem> {
        let tokens = if tagged_words.is_empty() {
            tokenize(text, config)
        } else {
            merge_name_fragments(tagged_words)
        };
        let mut items = Vec::new();
        let mut seen = HashSet::new();
        for rule in self.active_rules() {
            if !rule_declared(config, rule) {
                continue;
            }
            for (subject, value) in match_atoms(&rule.atoms, &tokens) {
                let key = format!(
                    "{}\u{1f}{}\u{1f}{}\u{1f}{}",
                    rule.entity_type, rule.subject_property, subject, value
                );
                if !seen.insert(key) {
                    continue;
                }
                let mut properties = HashMap::new();
                properties.insert(rule.subject_property.clone(), subject);
                properties.insert(rule.value_property.clone(), value);
                items.push(KnowledgeItem::Entity {
                    label: rule.entity_type.clone(),
                    properties,
                });
            }
        }
        items
    }

    pub fn render_active(&self) -> String {
        let rules = self.active_rules();
        if rules.is_empty() {
            return "批量教学：没有已采纳的活动规则。".to_string();
        }
        let mut text = format!("批量教学：{} 条活动规则。", rules.len());
        for rule in rules {
            text.push_str(&format!("\n  {}", rule.render_text().replace('\n', "\n  ")));
        }
        text
    }

    fn latest_rule(&self, id: &str) -> Option<&LearnedRule> {
        self.episodes
            .iter()
            .rev()
            .flat_map(|episode| episode.candidates.iter().rev())
            .find(|rule| rule.id == id)
    }

    fn select_targets(&self, selector: &str) -> Result<BTreeSet<String>, String> {
        if selector == "all" {
            let mut ids = BTreeSet::new();
            for episode in &self.episodes {
                for rule in &episode.candidates {
                    if rule.decision == RuleDecision::Ready && rule.status == RuleStatus::Candidate
                    {
                        ids.insert(rule.id.clone());
                    }
                }
            }
            return Ok(ids);
        }
        let mut ids = BTreeSet::new();
        for episode in &self.episodes {
            for rule in &episode.candidates {
                if rule.id == selector || rule.id.starts_with(selector) {
                    ids.insert(rule.id.clone());
                }
            }
        }
        if ids.is_empty() {
            return Err(format!("找不到规则「{}」", selector));
        }
        if ids.len() > 1 {
            return Err(format!(
                "规则前缀「{}」有多个匹配：{}",
                selector,
                ids.into_iter().collect::<Vec<_>>().join(", ")
            ));
        }
        Ok(ids)
    }
}

fn rule_declared(config: &SystemConfig, rule: &LearnedRule) -> bool {
    config.entity_types.iter().any(|entity_type| {
        entity_type.label == rule.entity_type
            && entity_type
                .properties
                .iter()
                .any(|property| property.name == rule.subject_property)
            && entity_type
                .properties
                .iter()
                .any(|property| property.name == rule.value_property)
    })
}

pub fn append_learned_items(base: &mut Vec<KnowledgeItem>, learned: Vec<KnowledgeItem>) -> usize {
    let mut added = 0usize;
    for item in learned {
        let duplicate = base.iter().any(|existing| match (existing, &item) {
            (
                KnowledgeItem::Entity {
                    label: left_label,
                    properties: left_properties,
                },
                KnowledgeItem::Entity {
                    label: right_label,
                    properties: right_properties,
                },
            ) => {
                left_label == right_label
                    && right_properties
                        .iter()
                        .all(|(key, value)| left_properties.get(key) == Some(value))
            }
            _ => false,
        });
        if !duplicate {
            base.push(item);
            added += 1;
        }
    }
    added
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: &str = r#"
extraction_rules = []
prolog_rules = []

[system]
oxigraph_endpoint = "http://localhost:7878"
oxigraph_store_path = "./oxigraph_data"
default_language = "zh"

[nlp]
import_keywords = ["录入"]
pos_dict = { "张三" = "nr", "李四" = "nr", "的" = "u", "是" = "v", "岁" = "q" }
[nlp.name_patterns]
prefixes = []
suffixes = []
full_names = []
profession_suffix = []

[[entity_types]]
label = "Person"
examples = []

[[entity_types.properties]]
name = "name"
data_type = "string"
required = true

[[entity_types.properties]]
name = "age"
data_type = "integer"
required = false

[messages]
startup = ""
ready = ""
query_prompt = ""
import_prompt = ""
quit_prompt = ""
importing = ""
no_knowledge_extracted = ""
cannot_understand = ""
process_error = ""
"#;

    fn test_config() -> Arc<SystemConfig> {
        Arc::new(toml::from_str(CONFIG).expect("测试配置应可解析"))
    }

    fn options(min_support: usize) -> BatchOptions {
        BatchOptions {
            min_support,
            min_purity: 800,
            ..BatchOptions::default()
        }
    }

    #[test]
    fn parses_tsv_and_jsonl_seed_facts() {
        let (facts, errors) = parse_seed_facts(
            "张三\tage\t18\n{\"subject\":\"李四\",\"predicate\":\"age\",\"value\":\"19\"}\n",
        );
        assert!(errors.is_empty());
        assert_eq!(facts.len(), 2);
        assert_eq!(facts[1].subject, "李四");
    }

    #[test]
    fn induces_ready_rule_from_shared_surface_pattern() {
        let teacher = BatchTeacher::new(test_config());
        let facts = "张三\tage\t18\n李四\tage\t19\n";
        let corpus = "张三的年龄是18岁。张三的年龄是20岁。李四的年龄是19岁。";
        let episode = teacher.teach(facts, corpus, "", &options(2));
        assert_eq!(episode.usable_facts, 2);
        assert_eq!(episode.candidates.len(), 1);
        let rule = &episode.candidates[0];
        assert_eq!(rule.decision, RuleDecision::Review);
        assert_eq!(rule.support, 2);
        assert!(rule.purity < 1000);
        assert_eq!(rule.render_pattern(), "{主语:*} 的 年龄 是 {取值:m}");
    }

    #[test]
    fn clean_corpus_yields_ready_rule() {
        let teacher = BatchTeacher::new(test_config());
        let facts = "张三\tage\t18\n李四\tage\t19\n";
        let corpus = "张三的年龄是18岁。李四的年龄是19岁。";
        let episode = teacher.teach(facts, corpus, "", &options(2));
        let rule = &episode.candidates[0];
        assert_eq!(rule.decision, RuleDecision::Ready);
        assert_eq!(rule.purity, 1000);
        assert_eq!(rule.status, RuleStatus::Candidate);
    }

    #[test]
    fn explicit_negative_rejects_candidate() {
        let teacher = BatchTeacher::new(test_config());
        let facts = "张三\tage\t18\n李四\tage\t19\n";
        let corpus = "张三的年龄是18岁。李四的年龄是19岁。";
        let episode = teacher.teach(facts, corpus, "张三的年龄是18岁。", &options(2));
        assert_eq!(episode.candidates[0].decision, RuleDecision::Rejected);
        assert_eq!(episode.candidates[0].negative_hits.len(), 1);
    }

    #[test]
    fn multi_token_value_is_skipped() {
        let teacher = BatchTeacher::new(test_config());
        let facts = "张三\tage\t十八岁\n李四\tage\t十九岁\n";
        let corpus = "张三的年龄是十八岁。李四的年龄是十九岁。";
        let episode = teacher.teach(facts, corpus, "", &options(2));
        assert!(episode.candidates.is_empty());
        assert!(!episode.skipped.is_empty());
    }

    #[test]
    fn adopt_persists_and_extracts_unseen_sentence() {
        let config = test_config();
        let teacher = BatchTeacher::new(config.clone());
        let facts = "张三\tage\t18\n李四\tage\t19\n";
        let corpus = "张三的年龄是18岁。李四的年龄是19岁。";
        let episode = teacher.teach(facts, corpus, "", &options(2));
        let id = episode.candidates[0].id.clone();
        let mut ledger = BatchTeachingLedger::default();
        ledger.record(episode);
        let adopted = ledger.adopt(&id, &config).expect("规则应可采纳");
        assert_eq!(adopted, vec![id]);
        assert_eq!(ledger.active_rules().len(), 1);

        let tokens = tokenize("王五的年龄是20岁。", &config);
        let items = ledger.extract_active(&config, "王五的年龄是20岁。", &tokens);
        assert_eq!(items.len(), 1);
        match &items[0] {
            KnowledgeItem::Entity { label, properties } => {
                assert_eq!(label, "Person");
                assert_eq!(properties.get("name"), Some(&"王五".to_string()));
                assert_eq!(properties.get("age"), Some(&"20".to_string()));
            }
            _ => panic!("应抽取实体"),
        }
    }

    #[test]
    fn review_rule_cannot_be_adopted() {
        let config = test_config();
        let teacher = BatchTeacher::new(config.clone());
        let facts = "张三\tage\t18\n";
        let corpus = "张三的年龄是18岁。";
        let episode = teacher.teach(facts, corpus, "", &options(2));
        let id = episode.candidates[0].id.clone();
        assert_eq!(episode.candidates[0].decision, RuleDecision::Review);
        let mut ledger = BatchTeachingLedger::default();
        ledger.record(episode);
        assert!(ledger.adopt(&id, &config).is_err());
        assert!(ledger.adopt("all", &config).unwrap().is_empty());
    }

    #[test]
    fn candidates_are_inert_before_explicit_adoption() {
        let config = test_config();
        let teacher = BatchTeacher::new(config.clone());
        let facts = "张三\tage\t18\n李四\tage\t19\n";
        let corpus = "张三的年龄是18岁。李四的年龄是19岁。";
        let episode = teacher.teach(facts, corpus, "", &options(2));
        let mut ledger = BatchTeachingLedger::default();
        ledger.record(episode);
        let tokens = tokenize("王五的年龄是23岁。", &config);
        assert!(ledger
            .extract_active(&config, "王五的年龄是23岁。", &tokens)
            .is_empty());
    }

    #[test]
    fn base_rules_covering_all_examples_force_review() {
        let config = Arc::new(
            toml::from_str::<SystemConfig>(
                &CONFIG.replace(
                    "extraction_rules = []\nprolog_rules = []",
                    r#"prolog_rules = []
[[extraction_rules]]
name = "已有年龄规则"
entity_type = "Person"
priority = 10

[[extraction_rules.patterns]]
pos_pattern = ["nr", "u", "n", "v", "m", "q"]
property_mappings = { name = "pos:nr", age = "pos:m" }"#,
                ),
            )
            .expect("测试配置应可解析"),
        );
        let teacher = BatchTeacher::new(config);
        let facts = "李四\tage\t18\n";
        let corpus = "李四的年龄是18岁。";
        let episode = teacher.teach(facts, corpus, "", &options(1));
        let rule = &episode.candidates[0];
        assert_eq!(rule.decision, RuleDecision::Review);
        assert!(rule.reasons.iter().any(|reason| reason.contains("现有抽取规则")));
    }

    #[test]
    fn ledger_roundtrip_preserves_active_rules() {
        let config = test_config();
        let teacher = BatchTeacher::new(config.clone());
        let facts = "张三\tage\t18\n李四\tage\t19\n";
        let corpus = "张三的年龄是18岁。李四的年龄是19岁。";
        let episode = teacher.teach(facts, corpus, "", &options(2));
        let id = episode.candidates[0].id.clone();
        let mut ledger = BatchTeachingLedger::default();
        ledger.record(episode);
        ledger.adopt(&id, &config).unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(LEDGER_FILE);
        ledger.save(path.to_str().unwrap()).unwrap();
        let loaded = match BatchTeachingLedger::load(path.to_str().unwrap()) {
            LedgerLoad::Loaded(loaded) => loaded,
            other => panic!("应加载成功：{:?}", other),
        };
        assert_eq!(loaded.active_rules().len(), 1);
        assert_eq!(loaded.active_rules()[0].id, id);
    }

    #[test]
    fn append_learned_items_skips_duplicates() {
        let mut base = vec![KnowledgeItem::Entity {
            label: "Person".to_string(),
            properties: HashMap::from([
                ("name".to_string(), "张三".to_string()),
                ("age".to_string(), "18".to_string()),
            ]),
        }];
        let duplicate = base.clone();
        let added = append_learned_items(&mut base, duplicate);
        assert_eq!(added, 0);
        assert_eq!(base.len(), 1);
    }
}

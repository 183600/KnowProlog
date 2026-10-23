//! PLG：跨会话的词法记忆（Persistent Lexical Graph）。
//!
//! `lexical_trace` 记录「这一轮为什么这样读」，生命周期止于当前请求；
//! 本模块把其中可复用的证据沉淀为 Oxigraph 命名图，让下一轮、下一次启动
//! 仍能回答「这个词以前被读成什么」「这个量词通常期待什么类型」。
//!
//! 三层证据，全部是软约束：
//!   1. **词形证据**：`(词面, 词性, 来源)` 的观察次数与轮次；
//!   2. **类型选票**：词性（nr/ns/nt）与后缀（公司/教授/城市）推出的弱类型；
//!   3. **量词期待**：`量词 → 类型` 的计数，由「量词 + 名词」语境习得，
//!      不写死量词表，也不把弱类型自动写成 `rdf:type`。
//!
//! 设计边界：
//!   - 只读消耗方（理解/消歧/生成）得到的是带票数的期待，冲突显式暴露；
//!   - 泛化量词（个/些/种/类/点…）不学习，避免把噪声当语法；
//!   - 容量有界，按最近使用轮次与观察次数淘汰；
//!   - 落图是幂等全量重写，使用独立命名图与账本命名空间，不污染知识图谱。

use std::collections::BTreeMap;

use oxigraph::model::{GraphName, Literal, NamedNode, Quad, Term};
use serde::{Deserialize, Serialize};

use crate::function_words::{roles_of, Role};
use crate::identity::infer_suffix_type;
use crate::nlp::{preprocess_annotated, AnnotatedWord, TagSource};

/// 词法记忆格式版本。
pub const LEXICAL_MEMORY_VERSION: u32 = 1;
/// 词法记忆命名图。
pub const LEXICAL_MEMORY_GRAPH: &str = "http://example.org/kg/graphs/lexical-memory";
/// 词法记忆专用命名空间（与知识图谱 `kg:` 隔离）。
pub const LEXICAL_MEMORY_NS: &str = "http://example.org/kg/ledger/lexical/";
/// 默认词条容量。
pub const DEFAULT_LEXICAL_CAPACITY: usize = 512;
/// 量词条目容量随词条容量收缩，避免量词表无限膨胀。
const CLASSIFIER_CAPACITY_RATIO: usize = 4;

const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
const WORD_CLASS: &str = "http://example.org/kg/ledger/lexical/WordEvidence";
const CLASSIFIER_CLASS: &str = "http://example.org/kg/ledger/lexical/ClassifierEvidence";
const STAMP_CLASS: &str = "http://example.org/kg/ledger/Stamp";
const STAMP_SUBJECT: &str = "http://example.org/kg/ledger/lexical/stamp";

const P_JSON: &str = "http://example.org/kg/ledger/lexical/json";
const P_SURFACE: &str = "http://example.org/kg/ledger/lexical/surface";
const P_POS: &str = "http://example.org/kg/ledger/lexical/pos";
const P_SOURCE: &str = "http://example.org/kg/ledger/lexical/source";
const P_COUNT: &str = "http://example.org/kg/ledger/lexical/count";
const P_FIRST_TURN: &str = "http://example.org/kg/ledger/lexical/firstTurn";
const P_LAST_TURN: &str = "http://example.org/kg/ledger/lexical/lastTurn";
const P_EXPECTED_TYPE: &str = "http://example.org/kg/ledger/lexical/expectedType";
const P_CONFLICT: &str = "http://example.org/kg/ledger/lexical/conflict";
const P_CLASSIFIER: &str = "http://example.org/kg/ledger/lexical/classifier";
const P_VERSION: &str = "http://example.org/kg/ledger/lexical/version";
const P_FINGERPRINT: &str = "http://example.org/kg/ledger/lexical/fingerprint";
const P_CAPACITY: &str = "http://example.org/kg/ledger/lexical/capacity";

type LexicalResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// 泛化量词：不携带稳定的类型信息，学习它们只会制造噪声。
const GENERIC_CLASSIFIERS: [&str; 9] = ["个", "些", "种", "类", "点", "批", "群", "份", "样"];

/// 一次观察的摘要（用于 CLI/日志；不含原文）。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservationSummary {
    pub tokens: usize,
    pub classifiers: usize,
    pub skipped_classifiers: usize,
}

/// 单个词形的证据：同一词面可以因词性/来源不同而有多条。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenEvidence {
    pub id: String,
    pub surface: String,
    pub pos: String,
    pub source: String,
    pub count: u32,
    pub first_turn: u64,
    pub last_turn: u64,
    /// 弱类型选票：类型 → 次数（来自词性、后缀与量词语境）。
    pub type_votes: BTreeMap<String, u32>,
}

/// 量词期待：`位 → 人物` 这类计数，而不是硬编码表。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClassifierEvidence {
    pub classifier: String,
    pub counts: BTreeMap<String, u32>,
    pub last_turn: u64,
}

impl ClassifierEvidence {
    pub fn total(&self) -> u32 {
        self.counts.values().copied().sum()
    }

    /// 最高票类型；并列时按类型名稳定排序并标记冲突。
    pub fn expectation(&self) -> Option<TypeExpectation> {
        expectation_from_counts(&self.counts, self.total())
    }
}

/// 带票数与冲突标记的类型期待。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TypeExpectation {
    pub expected_type: String,
    pub votes: u32,
    pub total: u32,
    pub conflict: bool,
    pub alternatives: Vec<(String, u32)>,
}

impl TypeExpectation {
    /// 支持度（整数票数 / 总票数；没有总票数时为 0）。
    pub fn confidence(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            f64::from(self.votes) / f64::from(self.total)
        }
    }
}

/// 词法记忆：词形证据 + 量词期待。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LexicalMemory {
    pub version: u32,
    pub turn: u64,
    pub capacity: usize,
    pub tokens: BTreeMap<String, TokenEvidence>,
    pub classifiers: BTreeMap<String, ClassifierEvidence>,
}

impl Default for LexicalMemory {
    fn default() -> Self {
        Self::new(DEFAULT_LEXICAL_CAPACITY)
    }
}

impl LexicalMemory {
    pub fn new(capacity: usize) -> Self {
        Self {
            version: LEXICAL_MEMORY_VERSION,
            turn: 0,
            capacity: capacity.max(1),
            tokens: BTreeMap::new(),
            classifiers: BTreeMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty() && self.classifiers.is_empty()
    }

    /// 观察一段已经标注过的文本。轮次只增不减，保证跨会话的 last_turn 单调。
    pub fn observe_annotated(
        &mut self,
        turn: u64,
        words: &[AnnotatedWord],
    ) -> ObservationSummary {
        self.turn = self.turn.max(turn);
        let mut summary = ObservationSummary::default();

        for word in words {
            if !is_content_word(word) {
                continue;
            }
            self.observe_token(word, turn);
            summary.tokens += 1;
        }

        for (index, word) in words.iter().enumerate() {
            if !is_classifier(word) {
                continue;
            }
            if GENERIC_CLASSIFIERS.contains(&word.text.as_str()) {
                summary.skipped_classifiers += 1;
                continue;
            }
            let Some(next) = words[index + 1..]
                .iter()
                .take(3)
                .find(|candidate| is_nounish(candidate) && is_content_word(candidate))
            else {
                continue;
            };
            let types = infer_word_types(next);
            if types.is_empty() {
                continue;
            }
            let key = token_key(next);
            for expected in types {
                self.record_classifier(&word.text, &expected, turn);
                if let Some(entry) = self.tokens.get_mut(&key) {
                    *entry.type_votes.entry(expected).or_insert(0) += 1;
                }
            }
            summary.classifiers += 1;
        }

        self.prune();
        summary
    }

    /// 分词并观察一段原文；只在宿主显式请求时调用（默认回答路径不增加成本）。
    pub fn observe_text(
        &mut self,
        turn: u64,
        text: &str,
        config: &crate::NlpConfig,
    ) -> ObservationSummary {
        let annotated = preprocess_annotated(text, config);
        self.observe_annotated(turn, &annotated)
    }

    fn observe_token(&mut self, word: &AnnotatedWord, turn: u64) {
        let key = token_key(word);
        let types = infer_word_types(word);
        match self.tokens.get_mut(&key) {
            Some(entry) => {
                entry.count = entry.count.saturating_add(1);
                entry.last_turn = entry.last_turn.max(turn);
                for expected in types {
                    *entry.type_votes.entry(expected).or_insert(0) += 1;
                }
            }
            None => {
                let mut type_votes = BTreeMap::new();
                for expected in types {
                    *type_votes.entry(expected).or_insert(0) += 1;
                }
                self.tokens.insert(
                    key.clone(),
                    TokenEvidence {
                        id: evidence_id(&key),
                        surface: word.text.clone(),
                        pos: word.pos.clone(),
                        source: word.source.code().to_string(),
                        count: 1,
                        first_turn: turn,
                        last_turn: turn,
                        type_votes,
                    },
                );
            }
        }
    }

    fn record_classifier(&mut self, classifier: &str, expected_type: &str, turn: u64) {
        let entry = self
            .classifiers
            .entry(classifier.to_string())
            .or_insert_with(|| ClassifierEvidence {
                classifier: classifier.to_string(),
                counts: BTreeMap::new(),
                last_turn: turn,
            });
        entry.last_turn = entry.last_turn.max(turn);
        *entry
            .counts
            .entry(expected_type.to_string())
            .or_insert(0) += 1;
    }

    /// 删去超出容量的词条：先保留最近使用、再保留高观察次数。
    pub fn prune(&mut self) {
        let capacity = self.capacity.max(1);
        if self.tokens.len() > capacity {
            let mut keys: Vec<(String, u64, u32)> = self
                .tokens
                .iter()
                .map(|(key, entry)| (key.clone(), entry.last_turn, entry.count))
                .collect();
            keys.sort_by(|a, b| a.1.cmp(&b.1).then(a.2.cmp(&b.2)).then(a.0.cmp(&b.0)));
            let excess = self.tokens.len() - capacity;
            for (key, _, _) in keys.into_iter().take(excess) {
                self.tokens.remove(&key);
            }
        }
        let classifier_capacity = (capacity / CLASSIFIER_CAPACITY_RATIO).max(1);
        if self.classifiers.len() > classifier_capacity {
            let mut keys: Vec<(String, u64, u32)> = self
                .classifiers
                .iter()
                .map(|(key, entry)| (key.clone(), entry.last_turn, entry.total()))
                .collect();
            keys.sort_by(|a, b| a.1.cmp(&b.1).then(a.2.cmp(&b.2)).then(a.0.cmp(&b.0)));
            let excess = self.classifiers.len() - classifier_capacity;
            for (key, _, _) in keys.into_iter().take(excess) {
                self.classifiers.remove(&key);
            }
        }
    }

    /// 某个词面的弱类型期待（跨词性/来源汇总票数）。
    pub fn type_expectation(&self, surface: &str) -> Option<TypeExpectation> {
        let surface = surface.trim();
        if surface.is_empty() {
            return None;
        }
        let mut counts: BTreeMap<String, u32> = BTreeMap::new();
        for entry in self.tokens.values().filter(|entry| entry.surface == surface) {
            for (expected, votes) in &entry.type_votes {
                *counts.entry(expected.clone()).or_insert(0) += votes;
            }
        }
        let total = counts.values().copied().sum();
        expectation_from_counts(&counts, total)
    }

    /// 某个量词的期待类型。
    pub fn classifier_expectation(&self, classifier: &str) -> Option<TypeExpectation> {
        self.classifiers
            .get(classifier.trim())
            .and_then(ClassifierEvidence::expectation)
    }

    /// 量词期待中显式记录冲突的条目（用于报告与调试）。
    pub fn classifier_conflicts(&self) -> Vec<(String, TypeExpectation)> {
        self.classifiers
            .values()
            .filter_map(|entry| {
                entry
                    .expectation()
                    .filter(|expectation| expectation.conflict)
                    .map(|expectation| (entry.classifier.clone(), expectation))
            })
            .collect()
    }

    /// 全部词条按 id 升序（已由 BTreeMap 保证）。
    pub fn token_entries(&self) -> impl Iterator<Item = &TokenEvidence> {
        self.tokens.values()
    }

    /// 全部量词期待（按量词升序）。
    pub fn classifier_entries(&self) -> impl Iterator<Item = &ClassifierEvidence> {
        self.classifiers.values()
    }

    /// 记忆指纹：与插入顺序无关，内容变化即变化。
    pub fn fingerprint(&self) -> String {
        let mut lines: Vec<String> = Vec::with_capacity(self.tokens.len() + self.classifiers.len());
        for entry in self.tokens.values() {
            let votes: Vec<String> = entry
                .type_votes
                .iter()
                .map(|(expected, count)| format!("{expected}:{count}"))
                .collect();
            lines.push(format!(
                "w|{}|{}|{}|{}|{}|{}",
                entry.surface,
                entry.pos,
                entry.source,
                entry.count,
                entry.last_turn,
                votes.join(",")
            ));
        }
        for entry in self.classifiers.values() {
            let counts: Vec<String> = entry
                .counts
                .iter()
                .map(|(expected, count)| format!("{expected}:{count}"))
                .collect();
            lines.push(format!(
                "c|{}|{}|{}",
                entry.classifier,
                entry.last_turn,
                counts.join(",")
            ));
        }
        lines.sort();
        short_hash(&lines.join("\n"))
    }

    /// 生成命名图四元组（含物化戳）。
    pub fn to_quads(&self) -> LexicalResult<Vec<Quad>> {
        let graph = GraphName::NamedNode(NamedNode::new(LEXICAL_MEMORY_GRAPH)?);
        let rdf_type = NamedNode::new(RDF_TYPE)?;
        let word_class = Term::NamedNode(NamedNode::new(WORD_CLASS)?);
        let classifier_class = Term::NamedNode(NamedNode::new(CLASSIFIER_CLASS)?);
        let mut quads = Vec::with_capacity(self.tokens.len() * 9 + self.classifiers.len() * 7 + 6);

        for entry in self.tokens.values() {
            let subject = NamedNode::new(format!("{}word/{}", LEXICAL_MEMORY_NS, entry.id))?;
            quads.push(Quad::new(
                subject.clone(),
                rdf_type.clone(),
                word_class.clone(),
                graph.clone(),
            ));
            quads.push(literal_quad(
                &subject,
                P_JSON,
                &serde_json::to_string(entry)?,
                &graph,
            )?);
            quads.push(literal_quad(&subject, P_SURFACE, &entry.surface, &graph)?);
            quads.push(literal_quad(&subject, P_POS, &entry.pos, &graph)?);
            quads.push(literal_quad(&subject, P_SOURCE, &entry.source, &graph)?);
            quads.push(literal_quad(
                &subject,
                P_COUNT,
                &entry.count.to_string(),
                &graph,
            )?);
            quads.push(literal_quad(
                &subject,
                P_FIRST_TURN,
                &entry.first_turn.to_string(),
                &graph,
            )?);
            quads.push(literal_quad(
                &subject,
                P_LAST_TURN,
                &entry.last_turn.to_string(),
                &graph,
            )?);
            if let Some(expectation) = expectation_from_counts(
                &entry.type_votes,
                entry.type_votes.values().copied().sum(),
            ) {
                quads.push(literal_quad(
                    &subject,
                    P_EXPECTED_TYPE,
                    &expectation.expected_type,
                    &graph,
                )?);
                quads.push(literal_quad(
                    &subject,
                    P_CONFLICT,
                    if expectation.conflict { "true" } else { "false" },
                    &graph,
                )?);
            }
        }

        for entry in self.classifiers.values() {
            let subject = NamedNode::new(format!(
                "{}classifier/{}",
                LEXICAL_MEMORY_NS,
                evidence_id(&entry.classifier)
            ))?;
            quads.push(Quad::new(
                subject.clone(),
                rdf_type.clone(),
                classifier_class.clone(),
                graph.clone(),
            ));
            quads.push(literal_quad(
                &subject,
                P_JSON,
                &serde_json::to_string(entry)?,
                &graph,
            )?);
            quads.push(literal_quad(
                &subject,
                P_CLASSIFIER,
                &entry.classifier,
                &graph,
            )?);
            quads.push(literal_quad(
                &subject,
                P_COUNT,
                &entry.total().to_string(),
                &graph,
            )?);
            quads.push(literal_quad(
                &subject,
                P_LAST_TURN,
                &entry.last_turn.to_string(),
                &graph,
            )?);
            if let Some(expectation) = entry.expectation() {
                quads.push(literal_quad(
                    &subject,
                    P_EXPECTED_TYPE,
                    &expectation.expected_type,
                    &graph,
                )?);
                quads.push(literal_quad(
                    &subject,
                    P_CONFLICT,
                    if expectation.conflict { "true" } else { "false" },
                    &graph,
                )?);
            }
        }

        let stamp = NamedNode::new(STAMP_SUBJECT)?;
        quads.push(Quad::new(
            stamp.clone(),
            rdf_type,
            Term::NamedNode(NamedNode::new(STAMP_CLASS)?),
            graph.clone(),
        ));
        quads.push(literal_quad(
            &stamp,
            P_VERSION,
            &self.version.to_string(),
            &graph,
        )?);
        quads.push(literal_quad(
            &stamp,
            P_FINGERPRINT,
            &self.fingerprint(),
            &graph,
        )?);
        quads.push(literal_quad(
            &stamp,
            P_CAPACITY,
            &self.capacity.to_string(),
            &graph,
        )?);
        quads.push(literal_quad(
            &stamp,
            P_COUNT,
            &self.tokens.len().to_string(),
            &graph,
        )?);
        Ok(quads)
    }

    /// 从命名图四元组还原记忆；损坏条目跳过，不让一条坏 JSON 毁掉整张表。
    pub fn from_quads(quads: impl Iterator<Item = Quad>) -> Self {
        let mut memory = Self::new(DEFAULT_LEXICAL_CAPACITY);
        let mut capacity = None;
        for quad in quads {
            let predicate = quad.predicate.as_str();
            if predicate == P_CAPACITY {
                if let Term::Literal(value) = &quad.object {
                    capacity = value.value().parse::<usize>().ok();
                }
                continue;
            }
            if predicate != P_JSON {
                continue;
            }
            let Term::Literal(value) = &quad.object else {
                continue;
            };
            if let Ok(entry) = serde_json::from_str::<TokenEvidence>(value.value()) {
                memory.turn = memory.turn.max(entry.last_turn);
                memory.tokens.entry(entry.id.clone()).or_insert(entry);
            } else if let Ok(entry) = serde_json::from_str::<ClassifierEvidence>(value.value()) {
                memory.turn = memory.turn.max(entry.last_turn);
                memory
                    .classifiers
                    .entry(entry.classifier.clone())
                    .or_insert(entry);
            }
        }
        if let Some(capacity) = capacity {
            memory.capacity = capacity.max(1);
        }
        memory.prune();
        memory
    }

    /// 人读报告（CLI/审计用，确定性排序）。
    pub fn render_report(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "词法记忆：{} 词条，{} 量词期待（第 {} 轮，容量 {}）\n",
            self.tokens.len(),
            self.classifiers.len(),
            self.turn,
            self.capacity
        ));
        if self.tokens.is_empty() && self.classifiers.is_empty() {
            out.push_str("  （空）\n");
            return out;
        }
        for entry in self.tokens.values().take(20) {
            let expectation = expectation_from_counts(
                &entry.type_votes,
                entry.type_votes.values().copied().sum(),
            )
            .map(|expectation| {
                format!(
                    "→ {}（{}/{}）{}",
                    expectation.expected_type,
                    expectation.votes,
                    expectation.total,
                    if expectation.conflict { " ⚠冲突" } else { "" }
                )
            })
            .unwrap_or_else(|| "→ 无类型证据".to_string());
            out.push_str(&format!(
                "  {} [{}/{}] ×{} 轮{}-{} {}\n",
                entry.surface,
                entry.pos,
                entry.source,
                entry.count,
                entry.first_turn,
                entry.last_turn,
                expectation
            ));
        }
        if self.tokens.len() > 20 {
            out.push_str(&format!("  …（还有 {} 条）\n", self.tokens.len() - 20));
        }
        for entry in self.classifiers.values() {
            if let Some(expectation) = entry.expectation() {
                out.push_str(&format!(
                    "  量词「{}」→ {}（{}/{}）{}\n",
                    entry.classifier,
                    expectation.expected_type,
                    expectation.votes,
                    expectation.total,
                    if expectation.conflict { " ⚠冲突" } else { "" }
                ));
            }
        }
        out
    }
}

fn expectation_from_counts(
    counts: &BTreeMap<String, u32>,
    total: u32,
) -> Option<TypeExpectation> {
    let mut ranked: Vec<(String, u32)> = counts
        .iter()
        .filter(|(_, votes)| **votes > 0)
        .map(|(expected, votes)| (expected.clone(), *votes))
        .collect();
    if ranked.is_empty() {
        return None;
    }
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let (expected_type, votes) = ranked.first().cloned()?;
    let conflict = ranked.get(1).is_some_and(|second| second.1 == votes);
    Some(TypeExpectation {
        expected_type,
        votes,
        total,
        conflict,
        alternatives: ranked,
    })
}

fn is_content_word(word: &AnnotatedWord) -> bool {
    let text = word.text.trim();
    if text.is_empty()
        || !text.chars().any(char::is_alphanumeric)
        || roles_of(text).contains(&Role::Quantifier)
        || (crate::function_words::is_registered(text) && !is_classifier(word))
    {
        return false;
    }
    is_content_pos(&word.pos)
}

fn is_content_pos(pos: &str) -> bool {
    pos.starts_with('n') || pos.starts_with('v') || pos.starts_with('a') || pos == "i" || pos == "l"
}

fn is_nounish(word: &AnnotatedWord) -> bool {
    word.pos.starts_with('n') || word.pos == "vn"
}

fn is_classifier(word: &AnnotatedWord) -> bool {
    word.pos == "q" || roles_of(&word.text).contains(&Role::Quantifier)
}

fn infer_word_types(word: &AnnotatedWord) -> Vec<String> {
    let mut types = Vec::new();
    match word.pos.as_str() {
        "nr" => types.push("人物".to_string()),
        "ns" => types.push("地点".to_string()),
        "nt" => types.push("组织".to_string()),
        _ => {}
    }
    if let Some((_, expected)) = infer_suffix_type(&word.text) {
        types.push(expected.to_string());
    }
    types.sort();
    types.dedup();
    types
}

fn token_key(word: &AnnotatedWord) -> String {
    format!(
        "{}\u{1f}{}\u{1f}{}",
        word.text,
        word.pos,
        word.source.code()
    )
}

fn evidence_id(text: &str) -> String {
    short_hash(text)
}

fn short_hash(text: &str) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{:016x}{:x}", hash, text.len())
}

fn literal_quad(
    subject: &NamedNode,
    predicate: &str,
    value: &str,
    graph: &GraphName,
) -> LexicalResult<Quad> {
    Ok(Quad::new(
        subject.clone(),
        NamedNode::new(predicate)?,
        Literal::new_simple_literal(value),
        graph.clone(),
    ))
}

/// 只为测试与 CLI 构造证据：避免调用方依赖 `TagSource` 内部表示。
pub fn annotated_word(
    text: &str,
    pos: &str,
    source: TagSource,
    start: usize,
    end: usize,
) -> AnnotatedWord {
    AnnotatedWord {
        text: text.to_string(),
        pos: pos.to_string(),
        source,
        jieba_pos: None,
        start,
        end,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NamePatterns;
    use crate::NlpConfig;

    fn word(text: &str, pos: &str, source: TagSource) -> AnnotatedWord {
        annotated_word(text, pos, source, 0, text.chars().count())
    }

    fn nlp_config() -> NlpConfig {
        NlpConfig {
            import_keywords: Vec::new(),
            pos_dict: std::collections::HashMap::new(),
            name_patterns: NamePatterns {
                prefixes: Vec::new(),
                suffixes: Vec::new(),
                full_names: Vec::new(),
                profession_suffix: Vec::new(),
            },
            enable_nlu: None,
            nlu_confidence_threshold: None,
            enable_local_semantic: None,
            disable_ownthink_api: None,
            semantic_similarity_threshold: None,
            max_entity_candidates: None,
            enable_advanced_entity_extraction: None,
            enable_relation_extraction: None,
            context_window_size: None,
        }
    }

    #[test]
    fn observes_pos_and_source_with_soft_type() {
        let mut memory = LexicalMemory::new(16);
        let words = vec![
            word("张三", "nr", TagSource::NamePattern),
            word("写", "v", TagSource::JiebaTag),
        ];
        let summary = memory.observe_annotated(1, &words);
        assert_eq!(summary.tokens, 2);
        let expectation = memory.type_expectation("张三").expect("应有类型期待");
        assert_eq!(expectation.expected_type, "人物");
        assert!(!expectation.conflict);
        let evidence = memory
            .token_entries()
            .find(|entry| entry.surface == "张三")
            .expect("应记录词形");
        assert_eq!(evidence.pos, "nr");
        assert_eq!(evidence.source, "name_pattern");
    }

    #[test]
    fn classifier_expectation_is_learned_from_context() {
        let mut memory = LexicalMemory::new(16);
        let words = vec![
            word("这", "r", TagSource::JiebaTag),
            word("位", "q", TagSource::JiebaTag),
            word("教授", "n", TagSource::JiebaTag),
        ];
        let summary = memory.observe_annotated(1, &words);
        assert_eq!(summary.classifiers, 1);
        let expectation = memory
            .classifier_expectation("位")
            .expect("量词期待应被学习");
        assert_eq!(expectation.expected_type, "人物");
        assert!(!expectation.conflict);
    }

    #[test]
    fn generic_classifiers_are_not_learned() {
        let mut memory = LexicalMemory::new(16);
        let words = vec![
            word("三", "m", TagSource::Numeric),
            word("个", "q", TagSource::JiebaTag),
            word("苹果", "n", TagSource::JiebaTag),
        ];
        let summary = memory.observe_annotated(1, &words);
        assert_eq!(summary.classifiers, 0);
        assert!(memory.classifier_expectation("个").is_none());
    }

    #[test]
    fn conflicting_types_are_exposed_not_hidden() {
        let mut memory = LexicalMemory::new(16);
        memory.observe_annotated(1, &[word("苹果", "n", TagSource::JiebaTag)]);
        memory.observe_annotated(2, &[word("苹果", "ns", TagSource::JiebaTag)]);
        // 手工补充一次组织票，制造 1:1 冲突
        let key = memory
            .tokens
            .keys()
            .find(|key| key.starts_with("苹果\u{1f}n\u{1f}"))
            .cloned()
            .expect("应存在苹果词条");
        memory
            .tokens
            .get_mut(&key)
            .expect("应存在苹果词条")
            .type_votes
            .insert("组织".to_string(), 1);
        let expectation = memory.type_expectation("苹果").expect("应有期待");
        assert!(expectation.conflict, "并列票数必须显式标记冲突");
        assert!(expectation.alternatives.len() >= 2);
    }

    #[test]
    fn quads_roundtrip_preserves_evidence() {
        let mut memory = LexicalMemory::new(32);
        memory.observe_annotated(
            3,
            &[
                word("李四", "nr", TagSource::ConfigPos),
                word("家", "q", TagSource::JiebaTag),
                word("公司", "n", TagSource::JiebaTag),
            ],
        );
        let quads = memory.to_quads().expect("四元组应可生成");
        let restored = LexicalMemory::from_quads(quads.into_iter());
        assert_eq!(restored.turn, 3);
        assert_eq!(restored.len(), memory.len());
        assert_eq!(restored.fingerprint(), memory.fingerprint());
        assert_eq!(
            restored.classifier_expectation("家").map(|e| e.expected_type),
            Some("组织".to_string())
        );
    }

    #[test]
    fn capacity_evicts_oldest_lowest_count() {
        let mut memory = LexicalMemory::new(2);
        memory.observe_annotated(1, &[word("旧词", "n", TagSource::JiebaTag)]);
        memory.observe_annotated(2, &[word("中词", "n", TagSource::JiebaTag)]);
        memory.observe_annotated(3, &[word("新词", "n", TagSource::JiebaTag)]);
        assert_eq!(memory.len(), 2);
        assert!(memory.token_entries().all(|entry| entry.surface != "旧词"));
    }

    #[test]
    fn observe_text_uses_shared_annotator() {
        let mut memory = LexicalMemory::new(64);
        let summary = memory.observe_text(1, "张三是老师", &nlp_config());
        assert!(summary.tokens > 0, "应至少观察到一个内容词");
        assert!(!memory.is_empty());
    }
}

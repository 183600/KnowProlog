use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt::Write as _;

use serde::{Deserialize, Serialize};

use crate::mention_lexicon::{LexiconStatus, MentionLexicon};
use crate::synonym_graph::{
    SynonymContext, SynonymGraph, SynonymKind, SynonymSense, SurfaceChoice,
};
use crate::text_anchor::{
    AnchoredSpan, CharSpan, ProjectionReceipt, TextProjection,
};

pub const LEXICAL_TRANSDUCER_VERSION: u32 = 1;
pub const DEFAULT_MIN_SURFACE_CHARS: usize = 2;
pub const DEFAULT_MAX_SURFACE_CHARS: usize = 32;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransducerOptions {
    pub min_surface_chars: usize,
    pub max_surface_chars: usize,
    pub protect_quoted_mentions: bool,
}

impl Default for TransducerOptions {
    fn default() -> Self {
        Self {
            min_surface_chars: DEFAULT_MIN_SURFACE_CHARS,
            max_surface_chars: DEFAULT_MAX_SURFACE_CHARS,
            protect_quoted_mentions: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransducerRule {
    pub surface: String,
    pub canonical: String,
    pub kind: SynonymKind,
    pub pos: String,
    #[serde(default, skip_serializing_if = "SynonymSense::is_empty")]
    pub sense: SynonymSense,
    pub origins: Vec<String>,
    pub evidence: u32,
}

impl TransducerRule {
    pub fn new(
        surface: impl Into<String>,
        canonical: impl Into<String>,
        kind: SynonymKind,
        pos: impl Into<String>,
    ) -> Self {
        Self {
            surface: surface.into(),
            canonical: canonical.into(),
            kind,
            pos: pos.into(),
            sense: SynonymSense::default(),
            origins: Vec::new(),
            evidence: 1,
        }
    }

    pub fn with_sense(mut self, sense: SynonymSense) -> Self {
        self.sense = sense;
        self
    }

    pub fn with_origin(mut self, origin: impl Into<String>) -> Self {
        let origin = origin.into();
        if !origin.trim().is_empty() && !self.origins.iter().any(|known| known == &origin) {
            self.origins.push(origin);
            self.origins.sort();
        }
        self
    }

    pub fn with_evidence(mut self, evidence: u32) -> Self {
        self.evidence = evidence.max(1);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum TransductionDecision {
    Rewrite,
    Ambiguous { candidates: Vec<String> },
    ProtectedMention { delimiter: char },
    ContextMismatch { candidates: Vec<String> },
}

impl TransductionDecision {
    pub fn code(&self) -> &'static str {
        match self {
            TransductionDecision::Rewrite => "rewrite",
            TransductionDecision::Ambiguous { .. } => "ambiguous",
            TransductionDecision::ProtectedMention { .. } => "protected_mention",
            TransductionDecision::ContextMismatch { .. } => "context_mismatch",
        }
    }

    pub fn rewrites(&self) -> bool {
        matches!(self, TransductionDecision::Rewrite)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransducedToken {
    pub surface: String,
    pub sense: String,
    pub kind: SynonymKind,
    pub pos: String,
    pub origin: String,
    pub evidence: u32,
    pub anchor: AnchoredSpan,
    pub decision: TransductionDecision,
}

impl TransducedToken {
    pub fn changes_surface(&self) -> bool {
        self.surface != self.sense
    }

    pub fn char_len(&self) -> usize {
        self.anchor.raw_chars.chars()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransductionReceipt {
    pub version: u32,
    pub raw_chars: usize,
    pub normalized_chars: usize,
    pub rule_count: usize,
    pub rewrites: usize,
    pub ambiguous: usize,
    pub protected_mentions: usize,
    pub context_mismatches: usize,
    pub projection: ProjectionReceipt,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Transduction {
    pub normalized: String,
    pub tokens: Vec<TransducedToken>,
    pub receipt: TransductionReceipt,
}

impl Transduction {
    pub fn changes(&self) -> usize {
        self.tokens
            .iter()
            .filter(|token| token.changes_surface())
            .count()
    }

    pub fn render_report(&self) -> String {
        let mut output = format!(
            "词法转导：{} 字 → {} 字（规则 {}，改写 {}，歧义 {}，引号提及 {}，语境不符 {}）。\n结果：{}",
            self.receipt.raw_chars,
            self.receipt.normalized_chars,
            self.receipt.rule_count,
            self.receipt.rewrites,
            self.receipt.ambiguous,
            self.receipt.protected_mentions,
            self.receipt.context_mismatches,
            self.normalized
        );
        for token in &self.tokens {
            let _ = write!(
                output,
                "\n  • 「{}」[{}..{})",
                token.surface, token.anchor.raw_chars.start, token.anchor.raw_chars.end
            );
            match &token.decision {
                TransductionDecision::Rewrite => {
                    let _ = write!(
                        output,
                        " → 「{}」｜{}｜词性 {}｜来源 {}｜支持 {}",
                        token.sense,
                        token.kind.label(),
                        token.pos,
                        token.origin,
                        token.evidence
                    );
                }
                TransductionDecision::Ambiguous { candidates } => {
                    let _ = write!(
                        output,
                        " 保留原词面｜歧义：{}",
                        candidates.join(" / ")
                    );
                }
                TransductionDecision::ProtectedMention { delimiter } => {
                    let _ = write!(output, " 保留原词面｜位于 {}…{} 引号内", delimiter, delimiter);
                }
                TransductionDecision::ContextMismatch { candidates } => {
                    let _ = write!(
                        output,
                        " 保留原词面｜语境不兼容：{}",
                        candidates.join(" / ")
                    );
                }
            }
        }
        output
    }
}

#[derive(Debug, Clone)]
pub struct LexicalTransducer {
    rules: Vec<TransducerRule>,
    by_surface: BTreeMap<String, Vec<usize>>,
    by_first: HashMap<char, Vec<String>>,
    options: TransducerOptions,
}

impl Default for LexicalTransducer {
    fn default() -> Self {
        Self::new(Vec::new(), TransducerOptions::default())
    }
}

impl LexicalTransducer {
    pub fn new(rules: Vec<TransducerRule>, options: TransducerOptions) -> Self {
        let rules: Vec<TransducerRule> = rules
            .into_iter()
            .filter(|rule| {
                let len = rule.surface.chars().count();
                len >= options.min_surface_chars
                    && len <= options.max_surface_chars
                    && !rule.surface.trim().is_empty()
            })
            .collect();
        let mut by_surface: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (index, rule) in rules.iter().enumerate() {
            by_surface
                .entry(rule.surface.clone())
                .or_default()
                .push(index);
        }
        let mut by_first: HashMap<char, Vec<String>> = HashMap::new();
        for surface in by_surface.keys() {
            if let Some(first) = surface.chars().next() {
                by_first.entry(first).or_default().push(surface.clone());
            }
        }
        for bucket in by_first.values_mut() {
            bucket.sort_by(|left, right| {
                right
                    .chars()
                    .count()
                    .cmp(&left.chars().count())
                    .then_with(|| left.cmp(right))
            });
        }
        Self {
            rules,
            by_surface,
            by_first,
            options,
        }
    }

    pub fn compile(
        graph: &SynonymGraph,
        lexicon: &MentionLexicon,
        flat_fallback: &HashMap<String, String>,
        options: TransducerOptions,
    ) -> Self {
        let mut merged: BTreeMap<(String, String, SynonymKind), TransducerRule> = BTreeMap::new();
        let mut graph_surfaces: BTreeSet<String> = BTreeSet::new();
        for entry in graph.entries() {
            if !entry.is_confirmed() {
                continue;
            }
            graph_surfaces.insert(entry.surface.clone());
            if entry.is_self_entry() {
                continue;
            }
            let key = (
                entry.surface.clone(),
                entry.canonical.clone(),
                entry.kind,
            );
            let pos = match entry.kind {
                SynonymKind::Predicate => "v",
                SynonymKind::Entity | SynonymKind::Type => "n",
            };
            let rule = merged.entry(key).or_insert_with(|| {
                TransducerRule::new(
                    entry.surface.clone(),
                    entry.canonical.clone(),
                    entry.kind,
                    pos,
                )
                .with_sense(entry.sense.clone())
            });
            if entry.sense != SynonymSense::default() {
                rule.sense = entry.sense.clone();
            }
            rule.evidence = rule.evidence.max(entry.observations);
            for origin in &entry.sources {
                if !rule.origins.iter().any(|known| known == origin) {
                    rule.origins.push(origin.clone());
                }
            }
        }
        for entry in lexicon.entries() {
            if entry.status != LexiconStatus::Confirmed || entry.surface == entry.canonical {
                continue;
            }
            let kind = if entry.pos.starts_with('v') {
                SynonymKind::Predicate
            } else {
                SynonymKind::Entity
            };
            let key = (entry.surface.clone(), entry.canonical.clone(), kind);
            let rule = merged.entry(key).or_insert_with(|| {
                TransducerRule::new(
                    entry.surface.clone(),
                    entry.canonical.clone(),
                    kind,
                    entry.pos.clone(),
                )
            });
            rule.pos = entry.pos.clone();
            if let Some(etype) = &entry.etype {
                if rule.sense.type_name.is_none() {
                    rule.sense.type_name = Some(etype.clone());
                }
            }
            rule.evidence = rule.evidence.max(entry.observations);
            let origin = format!("mention_lexicon:{}", entry.origin);
            if !rule.origins.iter().any(|known| known == &origin) {
                rule.origins.push(origin);
            }
        }
        for (surface, canonical) in flat_fallback {
            if surface == canonical || graph_surfaces.contains(surface) {
                continue;
            }
            if merged.keys().any(|(known, _, _)| known == surface) {
                continue;
            }
            merged.insert(
                (
                    surface.clone(),
                    canonical.clone(),
                    SynonymKind::Entity,
                ),
                TransducerRule::new(
                    surface.clone(),
                    canonical.clone(),
                    SynonymKind::Entity,
                    "n",
                )
                .with_origin("config_flat")
                .with_evidence(1),
            );
        }
        let mut rules: Vec<TransducerRule> = merged.into_values().collect();
        for rule in &mut rules {
            rule.origins.sort();
            rule.origins.dedup();
        }
        Self::new(rules, options)
    }

    pub fn len(&self) -> usize {
        self.rules.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    pub fn options(&self) -> &TransducerOptions {
        &self.options
    }

    pub fn rules_for(&self, surface: &str) -> Vec<&TransducerRule> {
        self.by_surface
            .get(surface)
            .map(|indices| indices.iter().map(|index| &self.rules[*index]).collect())
            .unwrap_or_default()
    }

    pub fn transduce<F>(
        &self,
        text: &str,
        context: &SynonymContext,
        type_compatible: F,
    ) -> Transduction
    where
        F: Fn(&str, &str) -> bool,
    {
        self.transduce_with_kind(text, context, None, type_compatible)
    }

    pub fn transduce_with_kind<F>(
        &self,
        text: &str,
        context: &SynonymContext,
        kind_filter: Option<SynonymKind>,
        type_compatible: F,
    ) -> Transduction
    where
        F: Fn(&str, &str) -> bool,
    {
        let projection = TextProjection::new(text, Default::default());
        let normalized = projection.normalized();
        let chars: Vec<char> = normalized.chars().collect();
        let protected = if self.options.protect_quoted_mentions {
            quote_ranges(&chars)
        } else {
            Vec::new()
        };
        let selected = self.select_spans(&chars);
        let mut tokens: Vec<TransducedToken> = Vec::with_capacity(selected.len());
        for (start, end, surface) in selected {
            let rules: Vec<&TransducerRule> = self
                .rules_for(&surface)
                .into_iter()
                .filter(|rule| kind_filter.map_or(true, |kind| rule.kind == kind))
                .collect();
            if rules.is_empty() {
                continue;
            }
            let Some(anchor) = projection
                .surface_chars_to_raw(CharSpan::new(start, end))
                .ok()
            else {
                continue;
            };
            if let Some(quote) = protected
                .iter()
                .find(|range| start >= range.start && end <= range.end)
            {
                tokens.push(TransducedToken {
                    surface,
                    sense: chars[start..end].iter().collect(),
                    kind: rules[0].kind,
                    pos: rules[0].pos.clone(),
                    origin: "quote_guard".to_string(),
                    evidence: 0,
                    anchor,
                    decision: TransductionDecision::ProtectedMention {
                        delimiter: quote.delimiter,
                    },
                });
                continue;
            }
            let mut candidates: BTreeMap<String, &TransducerRule> = BTreeMap::new();
            let mut mismatched: BTreeSet<String> = BTreeSet::new();
            for rule in &rules {
                if !sense_compatible(rule, context, &type_compatible) {
                    mismatched.insert(rule.canonical.clone());
                    continue;
                }
                candidates
                    .entry(rule.canonical.clone())
                    .and_modify(|current| {
                        if stronger(rule, current) {
                            *current = rule;
                        }
                    })
                    .or_insert(rule);
            }
            if candidates.len() == 1 {
                let (sense, rule) = candidates.into_iter().next().expect("one candidate");
                tokens.push(TransducedToken {
                    surface: surface.clone(),
                    sense,
                    kind: rule.kind,
                    pos: rule.pos.clone(),
                    origin: rule.origins.join(" / "),
                    evidence: rule.evidence,
                    anchor,
                    decision: TransductionDecision::Rewrite,
                });
            } else if candidates.is_empty() {
                tokens.push(TransducedToken {
                    surface: surface.clone(),
                    sense: surface,
                    kind: rules[0].kind,
                    pos: rules[0].pos.clone(),
                    origin: "context_guard".to_string(),
                    evidence: 0,
                    anchor,
                    decision: TransductionDecision::ContextMismatch {
                        candidates: mismatched.into_iter().collect(),
                    },
                });
            } else {
                tokens.push(TransducedToken {
                    surface: surface.clone(),
                    sense: surface,
                    kind: rules[0].kind,
                    pos: rules[0].pos.clone(),
                    origin: "ambiguity_guard".to_string(),
                    evidence: 0,
                    anchor,
                    decision: TransductionDecision::Ambiguous {
                        candidates: candidates.keys().cloned().collect(),
                    },
                });
            }
        }
        let output = render_normalized(&chars, &tokens);
        let receipt = TransductionReceipt {
            version: LEXICAL_TRANSDUCER_VERSION,
            raw_chars: text.chars().count(),
            normalized_chars: output.chars().count(),
            rule_count: self.rules.len(),
            rewrites: tokens
                .iter()
                .filter(|token| matches!(token.decision, TransductionDecision::Rewrite))
                .count(),
            ambiguous: tokens
                .iter()
                .filter(|token| matches!(token.decision, TransductionDecision::Ambiguous { .. }))
                .count(),
            protected_mentions: tokens
                .iter()
                .filter(|token| {
                    matches!(
                        token.decision,
                        TransductionDecision::ProtectedMention { .. }
                    )
                })
                .count(),
            context_mismatches: tokens
                .iter()
                .filter(|token| {
                    matches!(
                        token.decision,
                        TransductionDecision::ContextMismatch { .. }
                    )
                })
                .count(),
            projection: projection.receipt(),
        };
        Transduction {
            normalized: output,
            tokens,
            receipt,
        }
    }

    fn select_spans(&self, chars: &[char]) -> Vec<(usize, usize, String)> {
        if chars.is_empty() || self.by_first.is_empty() {
            return Vec::new();
        }
        let mut candidates: Vec<(usize, usize, String)> = Vec::new();
        for start in 0..chars.len() {
            let Some(bucket) = self.by_first.get(&chars[start]) else {
                continue;
            };
            for surface in bucket {
                let surface_chars: Vec<char> = surface.chars().collect();
                let end = start + surface_chars.len();
                if end > chars.len() || chars[start..end] != surface_chars[..] {
                    continue;
                }
                candidates.push((start, end, surface.clone()));
            }
        }
        if candidates.is_empty() {
            return Vec::new();
        }
        candidates.sort_by(|left, right| {
            left.1
                .cmp(&right.1)
                .then_with(|| left.0.cmp(&right.0))
                .then_with(|| (right.1 - right.0).cmp(&(left.1 - left.0)))
                .then_with(|| left.2.cmp(&right.2))
        });
        let n = chars.len();
        let mut by_end: Vec<Vec<usize>> = vec![Vec::new(); n + 1];
        for (index, candidate) in candidates.iter().enumerate() {
            by_end[candidate.1].push(index);
        }
        #[derive(Debug, Clone, Copy)]
        struct Best {
            covered: usize,
            spans: usize,
            candidate: Option<usize>,
        }
        let mut best: Vec<Best> = vec![
            Best {
                covered: 0,
                spans: 0,
                candidate: None,
            };
            n + 1
        ];
        for pos in 1..=n {
            best[pos] = best[pos - 1];
            for &index in &by_end[pos] {
                let candidate = &candidates[index];
                let previous = &best[candidate.0];
                let covered = previous.covered + (candidate.1 - candidate.0);
                let spans = previous.spans + 1;
                let replace = covered > best[pos].covered
                    || (covered == best[pos].covered && spans < best[pos].spans)
                    || (covered == best[pos].covered
                        && spans == best[pos].spans
                        && best[pos]
                            .candidate
                            .map(|current| {
                                (candidate.1 - candidate.0)
                                    > (candidates[current].1 - candidates[current].0)
                            })
                            .unwrap_or(true));
                if replace {
                    best[pos] = Best {
                        covered,
                        spans,
                        candidate: Some(index),
                    };
                }
            }
        }
        let mut selected = Vec::new();
        let mut pos = n;
        while let Some(index) = best[pos].candidate {
            let candidate = &candidates[index];
            selected.push(candidate.clone());
            pos = candidate.0;
        }
        selected.reverse();
        selected
    }
}

#[derive(Debug, Clone, Default)]
pub struct EchoLedger {
    turn: u64,
    recent: BTreeMap<(SynonymKind, String), (String, u64)>,
}

impl EchoLedger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn begin_turn(&mut self, turn: u64) {
        self.turn = self.turn.max(turn);
    }

    pub fn observe(&mut self, token: &TransducedToken) {
        if token.decision.rewrites() && token.changes_surface() {
            self.remember(token.kind, &token.sense, &token.surface);
        }
    }

    pub fn observe_all(&mut self, transduction: &Transduction) {
        for token in &transduction.tokens {
            self.observe(token);
        }
    }

    pub fn remember(
        &mut self,
        kind: SynonymKind,
        canonical: impl Into<String>,
        surface: impl Into<String>,
    ) {
        let canonical = canonical.into();
        let surface = surface.into();
        if canonical.trim().is_empty() || surface.trim().is_empty() {
            return;
        }
        self.recent
            .insert((kind, canonical), (surface, self.turn));
    }

    pub fn forget(&mut self, kind: SynonymKind, canonical: &str) -> bool {
        self.recent
            .remove(&(kind, canonical.trim().to_string()))
            .is_some()
    }

    pub fn recent(&self, kind: SynonymKind, canonical: &str) -> Option<&str> {
        self.recent
            .get(&(kind, canonical.trim().to_string()))
            .map(|(surface, _)| surface.as_str())
    }

    pub fn turn_of(&self, kind: SynonymKind, canonical: &str) -> Option<u64> {
        self.recent
            .get(&(kind, canonical.trim().to_string()))
            .map(|(_, turn)| *turn)
    }

    pub fn choose(
        &self,
        graph: &SynonymGraph,
        canonical: &str,
        kind: SynonymKind,
        context: &SynonymContext,
    ) -> SurfaceChoice {
        let echoes: Vec<String> = self
            .recent(kind, canonical)
            .map(|surface| vec![surface.to_string()])
            .unwrap_or_default();
        graph.surface_for(canonical, kind, context, &echoes)
    }
}

fn stronger(candidate: &TransducerRule, current: &TransducerRule) -> bool {
    candidate
        .evidence
        .cmp(&current.evidence)
        .then_with(|| candidate.origins.join(",").cmp(&current.origins.join(",")))
        .is_gt()
}

fn sense_compatible<F>(rule: &TransducerRule, context: &SynonymContext, compat: &F) -> bool
where
    F: Fn(&str, &str) -> bool,
{
    compatible_slot(
        context.expected_type.as_deref(),
        rule.sense.type_name.as_deref(),
        compat,
    ) && compatible_slot(
        context.domain.as_deref(),
        rule.sense.domain.as_deref(),
        compat,
    ) && compatible_slot(context.range.as_deref(), rule.sense.range.as_deref(), compat)
}

fn compatible_slot<F>(expected: Option<&str>, actual: Option<&str>, compat: &F) -> bool
where
    F: Fn(&str, &str) -> bool,
{
    match (expected, actual) {
        (Some(expected), Some(actual)) => compat(expected, actual) || compat(actual, expected),
        _ => true,
    }
}

#[derive(Debug, Clone, Copy)]
struct QuoteRange {
    start: usize,
    end: usize,
    delimiter: char,
}

fn quote_ranges(chars: &[char]) -> Vec<QuoteRange> {
    const PAIRS: [(char, char); 4] = [
        ('“', '”'),
        ('「', '」'),
        ('『', '』'),
        ('‘', '’'),
    ];
    let mut ranges = Vec::new();
    let mut index = 0;
    while index < chars.len() {
        let mut matched = None;
        for (open, close) in PAIRS {
            if chars[index] == open {
                if let Some(offset) = chars[index + 1..].iter().position(|ch| *ch == close) {
                    matched = Some((open, index + 1, index + 1 + offset));
                }
                break;
            }
        }
        if let Some((delimiter, start, end)) = matched {
            ranges.push(QuoteRange {
                start,
                end,
                delimiter,
            });
            index = end + 1;
        } else {
            index += 1;
        }
    }
    ranges
}

fn render_normalized(chars: &[char], tokens: &[TransducedToken]) -> String {
    let mut output = String::with_capacity(chars.len());
    let mut cursor = 0;
    for token in tokens {
        let start = token.anchor.normalized_chars.start;
        let end = token.anchor.normalized_chars.end;
        if start < cursor || end > chars.len() {
            continue;
        }
        output.extend(&chars[cursor..start]);
        output.push_str(&token.sense);
        cursor = end;
    }
    output.extend(&chars[cursor..]);
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synonym_graph::{ObserveOutcome, SynonymResolution};

    fn rule(surface: &str, canonical: &str) -> TransducerRule {
        TransducerRule::new(surface, canonical, SynonymKind::Entity, "n")
            .with_origin("test")
            .with_evidence(2)
    }

    fn transduce(text: &str) -> Transduction {
        let transducer = LexicalTransducer::new(
            vec![rule("老婆", "妻子"), rule("老公", "丈夫")],
            TransducerOptions::default(),
        );
        transducer.transduce(text, &SynonymContext::new(), |left, right| {
            left.eq_ignore_ascii_case(right)
        })
    }

    fn graph_with(entries: &[(&str, &str, SynonymKind, Option<&str>)]) -> SynonymGraph {
        let mut graph = SynonymGraph::new(64);
        for (surface, canonical, kind, type_name) in entries {
            let sense = type_name
                .map(SynonymSense::type_only)
                .unwrap_or_default();
            let outcome = graph.observe(
                1,
                surface,
                canonical,
                *kind,
                sense,
                "test",
                1.0,
            );
            let id = match outcome {
                ObserveOutcome::Added { id } | ObserveOutcome::Reinforced { id } => id,
                other => panic!("unexpected observation: {other:?}"),
            };
            graph.confirm(&id, 1);
        }
        graph
    }

    #[test]
    fn rewrites_a_multi_character_surface_and_keeps_raw_coordinates() {
        let transduction = transduce("我老婆是老师");
        assert_eq!(transduction.normalized, "我妻子是老师");
        assert_eq!(transduction.tokens.len(), 1);
        let token = &transduction.tokens[0];
        assert_eq!(token.surface, "老婆");
        assert_eq!(token.sense, "妻子");
        assert_eq!(token.anchor.raw_chars, CharSpan::new(1, 3));
        assert!(token.changes_surface());
        assert_eq!(transduction.receipt.rewrites, 1);
    }

    #[test]
    fn quoted_mentions_are_preserved_but_unclosed_quotes_are_not() {
        let quoted = transduce("“老婆”是什么意思");
        assert_eq!(quoted.normalized, "“老婆”是什么意思");
        assert_eq!(quoted.receipt.ambiguous, 0);
        assert_eq!(quoted.receipt.protected_mentions, 1);

        let unclosed = transduce("“老婆是谁");
        assert_eq!(unclosed.normalized, "“妻子是谁");
        assert_eq!(unclosed.receipt.protected_mentions, 0);
    }

    #[test]
    fn multiple_canonicals_stay_ambiguous_and_keep_the_surface() {
        let transducer = LexicalTransducer::new(
            vec![
                rule("苹果", "苹果公司"),
                rule("苹果", "苹果树"),
            ],
            TransducerOptions::default(),
        );
        let transduction = transducer.transduce("苹果很甜", &SynonymContext::new(), |l, r| l == r);
        assert_eq!(transduction.normalized, "苹果很甜");
        assert_eq!(transduction.receipt.ambiguous, 1);
        match &transduction.tokens[0].decision {
            TransductionDecision::Ambiguous { candidates } => {
                assert_eq!(candidates, &["苹果公司".to_string(), "苹果树".to_string()]);
            }
            other => panic!("expected ambiguity, got {other:?}"),
        }
    }

    #[test]
    fn context_type_can_disambiguate_a_typed_alias() {
        let mut graph = SynonymGraph::new(64);
        for (canonical, type_name) in [("苹果公司", "Organization"), ("苹果树", "Plant")] {
            let outcome = graph.observe(
                1,
                "苹果",
                canonical,
                SynonymKind::Entity,
                SynonymSense::type_only(type_name),
                "test",
                1.0,
            );
            let id = match outcome {
                ObserveOutcome::Added { id } | ObserveOutcome::Reinforced { id } => id,
                other => panic!("unexpected observation: {other:?}"),
            };
            graph.confirm(&id, 1);
        }
        let transducer = LexicalTransducer::compile(
            &graph,
            &MentionLexicon::new(),
            &HashMap::new(),
            TransducerOptions::default(),
        );
        let context = SynonymContext::new().with_expected_type("Organization");
        let transduction = transducer.transduce("苹果发布了新品", &context, |left, right| {
            left.eq_ignore_ascii_case(right)
        });
        assert_eq!(transduction.normalized, "苹果公司发布了新品");
        assert_eq!(transduction.receipt.context_mismatches, 0);
    }

    #[test]
    fn incompatible_context_keeps_the_surface() {
        let graph = graph_with(&[("苹果", "苹果公司", SynonymKind::Entity, Some("Organization"))]);
        let transducer = LexicalTransducer::compile(
            &graph,
            &MentionLexicon::new(),
            &HashMap::new(),
            TransducerOptions::default(),
        );
        let context = SynonymContext::new().with_expected_type("Person");
        let transduction = transducer.transduce("苹果很甜", &context, |left, right| left == right);
        assert_eq!(transduction.normalized, "苹果很甜");
        assert_eq!(transduction.receipt.context_mismatches, 1);
        assert!(matches!(
            transduction.tokens[0].decision,
            TransductionDecision::ContextMismatch { .. }
        ));
    }

    #[test]
    fn mention_lexicon_and_flat_fallback_are_compiled_without_duplicates() {
        let mut lexicon = MentionLexicon::new();
        lexicon.confirm("未名科技公司", "未名科技", Some("Organization"), "kb");
        let mut flat = HashMap::new();
        flat.insert("未名科技公司".to_string(), "未名科技".to_string());
        flat.insert("老婆".to_string(), "妻子".to_string());
        let transducer = LexicalTransducer::compile(
            &SynonymGraph::new(8),
            &lexicon,
            &flat,
            TransducerOptions::default(),
        );
        assert_eq!(transducer.rules_for("未名科技公司").len(), 1);
        assert_eq!(transducer.rules_for("老婆").len(), 1);
        assert_eq!(
            transducer.rules_for("未名科技公司")[0].origins,
            vec!["mention_lexicon:kb".to_string()]
        );
    }

    #[test]
    fn orthographic_noise_still_projects_back_to_the_raw_span() {
        let transduction = transduce("老\u{200b}婆是老师");
        assert_eq!(transduction.normalized, "妻子是老师");
        assert_eq!(
            transduction.tokens[0].anchor.raw_chars,
            CharSpan::new(0, 3)
        );
        assert_eq!(transduction.tokens[0].surface, "老婆");
    }

    #[test]
    fn echo_ledger_prefers_a_reversible_recent_surface() {
        let graph = graph_with(&[
            ("老婆", "妻子", SynonymKind::Entity, Some("Person")),
            ("爱人", "妻子", SynonymKind::Entity, Some("Person")),
        ]);
        let mut ledger = EchoLedger::new();
        ledger.begin_turn(3);
        ledger.remember(SynonymKind::Entity, "妻子", "老婆");
        ledger.begin_turn(4);
        ledger.remember(SynonymKind::Entity, "妻子", "爱人");
        assert_eq!(ledger.recent(SynonymKind::Entity, "妻子"), Some("爱人"));
        assert_eq!(ledger.turn_of(SynonymKind::Entity, "妻子"), Some(4));
        let choice = ledger.choose(&graph, "妻子", SynonymKind::Entity, &SynonymContext::new());
        assert_eq!(choice.surface, "爱人");
        assert_eq!(choice.source, "会话回声");
        assert!(choice.reversible);
        assert!(ledger.forget(SynonymKind::Entity, "妻子"));
        let fallback = ledger.choose(&graph, "妻子", SynonymKind::Entity, &SynonymContext::new());
        assert_ne!(fallback.surface, "妻子");
        assert!(fallback.reversible);
    }

    #[test]
    fn kind_filter_resolves_cross_layer_homographs() {
        let transducer = LexicalTransducer::new(
            vec![
                TransducerRule::new("老婆", "妻子", SynonymKind::Entity, "n"),
                TransducerRule::new("老婆", "配偶", SynonymKind::Predicate, "v"),
            ],
            TransducerOptions::default(),
        );
        let entity = transducer.transduce_with_kind(
            "我老婆是老师",
            &SynonymContext::new(),
            Some(SynonymKind::Entity),
            |left, right| left == right,
        );
        assert_eq!(entity.normalized, "我妻子是老师");
        let predicate = transducer.transduce_with_kind(
            "我老婆是老师",
            &SynonymContext::new(),
            Some(SynonymKind::Predicate),
            |left, right| left == right,
        );
        assert_eq!(predicate.normalized, "我配偶是老师");
    }

    #[test]
    fn graph_compile_skips_unconfirmed_entries() {
        let mut graph = SynonymGraph::new(8);
        graph.observe(
            1,
            "老婆",
            "妻子",
            SynonymKind::Entity,
            SynonymSense::default(),
            "test",
            1.0,
        );
        let transducer = LexicalTransducer::compile(
            &graph,
            &MentionLexicon::new(),
            &HashMap::new(),
            TransducerOptions::default(),
        );
        assert!(transducer.is_empty());
        assert!(matches!(
            graph.canonicalize("老婆", SynonymKind::Entity, &SynonymContext::new()),
            SynonymResolution::Unresolved { .. }
        ));
    }
}

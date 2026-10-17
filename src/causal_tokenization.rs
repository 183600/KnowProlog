// 因果分词（纯符号）：知识图谱/因果图引导的 token 格 DP + token 级因果归因
//
// 灵感：
//   • tokenization：中文没有空格，"分词"本质是在字符流上切出能锚定到知识的片段；
//   • 因果推理：图谱里每条基化规则的前件→后件是一条有符号因果机制，
//     于是 token 的质量 = 它能锚定多少知识 + 锚定对象参与多少因果机制。
//   • 用这两个信号给分词做 k-best Viterbi DP：切出来的词不仅"像词"
//     （锚定到实体/谓词/同义词/事实常数），还要"有因果分量"（在规则网络中进出边越多越优先）。
//   • token 级因果归因：把结论及其因果链上每个原子的概率质量，按词表锚定关系
//     回溯摊到输入 token 上——这是纯符号版的 attention/attribution，可逐行回放。
//
// 全部运算为字符格 DP、哈希查找与图搜索：无神经网络、无参数训练。

use std::collections::{HashMap, HashSet};

use crate::causal_reasoning::{CausalEdge, CausalGraph};
use crate::markov_logic::MarkovLogicEngine;
use crate::nlp::shared_jieba;
use crate::resolution_engine::Atom;

/// 单个候选 token 的最大字符数（防组合爆炸）
pub const MAX_TOKEN_CHARS: usize = 8;

/// 每个位置保留的 k-best 路径数
pub const K_BEST: usize = 3;

/// 结巴切出的普通词（未锚定）的基础得分
const JIEBA_TOKEN_SCORE: f64 = 0.03;
/// 标点/空白单字符得分
const PUNCT_SCORE: f64 = 0.01;
/// 未锚定单字得分（保证全串覆盖）
const UNKNOWN_CHAR_SCORE: f64 = 0.0;
/// 因果质量对锚定分的加成系数：bonus = scale · ln(1 + mass)
const CAUSAL_BONUS_SCALE: f64 = 0.15;
/// 因果加成上限（防止高频谓词一家独大）
const CAUSAL_BONUS_CAP: f64 = 0.5;
/// 别名（同义词）锚定的支持度折扣
const ALIAS_DISCOUNT: f64 = 0.95;

/// 锚定类型：token 命中的知识种类
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnchorKind {
    Entity,
    Type,
    Predicate,
}

impl AnchorKind {
    pub fn label(self) -> &'static str {
        match self {
            AnchorKind::Entity => "实体",
            AnchorKind::Type => "类型",
            AnchorKind::Predicate => "谓词",
        }
    }
}

/// 词表锚：一个表面形式锚定到规范知识项
#[derive(Debug, Clone)]
pub struct Anchor {
    pub surface: String,
    pub canonical: String,
    pub kind: AnchorKind,
    /// 基础锚定强度（0..1，越大越可信）
    pub support: f64,
    /// 因果质量：该锚定对象在因果图中涉及的机制权重绝对值之和
    pub causal_mass: f64,
    /// 涉及的因果机制条数
    pub causal_edges: usize,
}

impl Anchor {
    fn causal_bonus(&self) -> f64 {
        (CAUSAL_BONUS_SCALE * (1.0 + self.causal_mass).ln()).min(CAUSAL_BONUS_CAP)
    }

    /// 锚定分 = 支持度 + 因果质量加成
    pub fn score(&self) -> f64 {
        self.support + self.causal_bonus()
    }
}

/// token 的锚定信息（分词结果里对 Anchor 的轻量视图）
#[derive(Debug, Clone)]
pub struct TokenAnchor {
    pub canonical: String,
    pub kind: AnchorKind,
    pub support: f64,
    pub causal_mass: f64,
    pub causal_edges: usize,
    /// 是否经由同义词/别名命中
    pub via_alias: bool,
}

impl TokenAnchor {
    pub fn text(&self) -> String {
        if self.via_alias {
            format!("{}「{}」（别名）", self.kind.label(), self.canonical)
        } else {
            format!("{}「{}」", self.kind.label(), self.canonical)
        }
    }
}

/// 分词结果中的一个 token（字符下标区间 [start, end)）
#[derive(Debug, Clone)]
pub struct TokenSpan {
    pub text: String,
    pub start: usize,
    pub end: usize,
    pub anchor: Option<TokenAnchor>,
}

/// 备选分词方案
#[derive(Debug, Clone)]
pub struct AlternativeSegmentation {
    pub tokens: Vec<String>,
    pub score: f64,
}

/// 一次因果分词的结果
#[derive(Debug, Clone)]
pub struct CausalTokenization {
    pub text: String,
    pub tokens: Vec<TokenSpan>,
    pub score: f64,
    pub alternatives: Vec<AlternativeSegmentation>,
}

impl CausalTokenization {
    pub fn token_texts(&self) -> Vec<&str> {
        self.tokens.iter().map(|token| token.text.as_str()).collect()
    }

    pub fn anchored_tokens(&self) -> impl Iterator<Item = &TokenSpan> {
        self.tokens.iter().filter(|token| token.anchor.is_some())
    }

    pub fn anchored_count(&self) -> usize {
        self.anchored_tokens().count()
    }

    /// `张三 | 的 | 师父 | 是 | 谁`
    pub fn render_segmentation(&self) -> String {
        self.tokens
            .iter()
            .map(|token| token.text.as_str())
            .collect::<Vec<_>>()
            .join(" | ")
    }
}

/// token 级因果归因：某个输入 token 对目标结论的因果贡献
#[derive(Debug, Clone)]
pub struct TokenAttribution {
    pub token: String,
    pub start: usize,
    pub end: usize,
    /// 命中锚的描述（`实体「张三」`）
    pub anchor: String,
    /// 绝对贡献（概率质量 × 锚定支持度）
    pub contribution: f64,
    /// 归一化占比（0..1）
    pub share: f64,
}

/// 因果分词器：词表锚 + 别名 + k-best 字符格 DP
pub struct CausalTokenizer {
    anchors: Vec<Anchor>,
    index: HashMap<String, usize>,
    aliases: HashMap<String, String>,
    max_token_chars: usize,
}

struct ResolvedAnchor<'a> {
    anchor: &'a Anchor,
    via_alias: bool,
}

#[derive(Debug, Clone)]
struct PathEntry {
    score: f64,
    start: usize,
    prev_rank: usize,
}

impl CausalTokenizer {
    pub fn new(max_token_chars: usize) -> Self {
        Self {
            anchors: Vec::new(),
            index: HashMap::new(),
            aliases: HashMap::new(),
            max_token_chars: max_token_chars.clamp(1, 16),
        }
    }

    pub fn anchor_count(&self) -> usize {
        self.anchors.len()
    }

    pub fn alias_count(&self) -> usize {
        self.aliases.len()
    }

    /// 登记词表锚；同表面形式取支持度更高者
    pub fn add_anchor(&mut self, surface: &str, canonical: &str, kind: AnchorKind, support: f64) {
        let surface = surface.trim();
        let canonical = canonical.trim();
        if surface.is_empty() || canonical.is_empty() {
            return;
        }
        let support = support.clamp(0.0, 1.0);
        if let Some(&index) = self.index.get(surface) {
            let existing = &mut self.anchors[index];
            if support > existing.support {
                existing.support = support;
                existing.kind = kind;
                existing.canonical = canonical.to_string();
            }
            return;
        }
        let index = self.anchors.len();
        self.anchors.push(Anchor {
            surface: surface.to_string(),
            canonical: canonical.to_string(),
            kind,
            support,
            causal_mass: 0.0,
            causal_edges: 0,
        });
        self.index.insert(surface.to_string(), index);
    }

    /// 登记别名：别名命中的锚定对象取规范形式的锚
    pub fn add_alias(&mut self, alias: &str, canonical: &str) {
        let alias = alias.trim();
        let canonical = canonical.trim();
        if alias.is_empty() || canonical.is_empty() || alias == canonical {
            return;
        }
        if self.index.contains_key(alias) {
            return;
        }
        self.aliases.insert(alias.to_string(), canonical.to_string());
    }

    /// 观察一条图谱事实：登记其实体/谓词锚（不含因果质量）
    pub fn observe_fact(&mut self, atom: &Atom) {
        if !atom.is_ground() {
            return;
        }
        for term in &atom.terms {
            let name = term.name();
            self.add_anchor(name, name, AnchorKind::Entity, 0.9);
        }
        let predicate = &*atom.predicate;
        self.add_anchor(predicate, predicate, AnchorKind::Predicate, 0.75);
    }

    /// 观察一条因果机制边（前件→后件）：两侧原子累计因果质量
    pub fn observe_causal_edge(&mut self, edge: &CausalEdge) {
        self.observe_causal_atom(&edge.cause, edge.weight);
        self.observe_causal_atom(&edge.effect, edge.weight);
    }

    fn observe_causal_atom(&mut self, atom: &Atom, weight: f64) {
        if !atom.is_ground() {
            return;
        }
        for term in &atom.terms {
            let name = term.name();
            self.add_anchor(name, name, AnchorKind::Entity, 0.85);
            self.accumulate_mass(name, weight);
        }
        let predicate = atom.predicate.to_string();
        self.add_anchor(&predicate, &predicate, AnchorKind::Predicate, 0.7);
        self.accumulate_mass(&predicate, weight);
    }

    fn accumulate_mass(&mut self, surface: &str, weight: f64) {
        if let Some(&index) = self.index.get(surface) {
            self.anchors[index].causal_mass += weight.abs();
            self.anchors[index].causal_edges += 1;
        }
    }

    fn resolve(&self, surface: &str) -> Option<ResolvedAnchor<'_>> {
        if let Some(&index) = self.index.get(surface) {
            return Some(ResolvedAnchor {
                anchor: &self.anchors[index],
                via_alias: false,
            });
        }
        let canonical = self.aliases.get(surface)?;
        let &index = self.index.get(canonical.as_str())?;
        Some(ResolvedAnchor {
            anchor: &self.anchors[index],
            via_alias: true,
        })
    }

    fn token_anchor(&self, surface: &str) -> Option<TokenAnchor> {
        let resolved = self.resolve(surface)?;
        Some(TokenAnchor {
            canonical: resolved.anchor.canonical.clone(),
            kind: resolved.anchor.kind,
            support: if resolved.via_alias {
                resolved.anchor.support * ALIAS_DISCOUNT
            } else {
                resolved.anchor.support
            },
            causal_mass: resolved.anchor.causal_mass,
            causal_edges: resolved.anchor.causal_edges,
            via_alias: resolved.via_alias,
        })
    }

    /// 单个候选区间的得分；不可作为 token 的区间返回 None
    fn span_score(
        &self,
        span: &str,
        start: usize,
        end: usize,
        jieba_spans: &HashSet<(usize, usize)>,
    ) -> Option<f64> {
        if let Some(resolved) = self.resolve(span) {
            let support = if resolved.via_alias {
                resolved.anchor.support * ALIAS_DISCOUNT
            } else {
                resolved.anchor.support
            };
            return Some(support + resolved.anchor.causal_bonus());
        }
        if jieba_spans.contains(&(start, end)) {
            return Some(JIEBA_TOKEN_SCORE);
        }
        if end - start == 1 {
            let ch = span.chars().next()?;
            if ch.is_whitespace() || is_punctuation(ch) {
                return Some(PUNCT_SCORE);
            }
            return Some(UNKNOWN_CHAR_SCORE);
        }
        None
    }

    fn jieba_spans(&self, text: &str, char_len: usize) -> HashSet<(usize, usize)> {
        let mut spans: HashSet<(usize, usize)> = HashSet::new();
        let mut offset = 0usize;
        for word in shared_jieba().cut(text, false) {
            let len = word.chars().count();
            if len > 0 && offset + len <= char_len {
                spans.insert((offset, offset + len));
            }
            offset += len;
        }
        spans
    }

    /// k-best 字符格 DP 分词
    pub fn tokenize(&self, text: &str) -> CausalTokenization {
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        let jieba_spans = self.jieba_spans(text, n);
        let mut paths: Vec<Vec<PathEntry>> = vec![Vec::new(); n + 1];
        paths[0].push(PathEntry {
            score: 0.0,
            start: 0,
            prev_rank: 0,
        });
        for end in 1..=n {
            let min_start = end.saturating_sub(self.max_token_chars);
            let mut candidates: Vec<PathEntry> = Vec::new();
            for start in min_start..end {
                let span: String = chars[start..end].iter().collect();
                let Some(span_score) = self.span_score(&span, start, end, &jieba_spans) else {
                    continue;
                };
                for (rank, entry) in paths[start].iter().enumerate() {
                    candidates.push(PathEntry {
                        score: entry.score + span_score,
                        start,
                        prev_rank: rank,
                    });
                }
            }
            // 得分降序；同分时区间更长者优先、起点更小者优先、前驱 rank 更小者优先
            candidates.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| b.start.cmp(&a.start))
                    .then_with(|| a.prev_rank.cmp(&b.prev_rank))
            });
            candidates.truncate(K_BEST);
            paths[end] = candidates;
        }

        let mut best_tokens: Vec<TokenSpan> = Vec::new();
        let mut best_score = 0.0;
        let mut alternatives: Vec<AlternativeSegmentation> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for (rank, entry) in paths[n].iter().enumerate() {
            let spans = reconstruct(&paths, n, rank);
            let key = spans
                .iter()
                .map(|(start, end)| chars[*start..*end].iter().collect::<String>())
                .collect::<Vec<_>>()
                .join("|");
            if !seen.insert(key) {
                continue;
            }
            let texts: Vec<String> = spans
                .iter()
                .map(|(start, end)| chars[*start..*end].iter().collect())
                .collect();
            if best_tokens.is_empty() {
                best_score = entry.score;
                best_tokens = spans
                    .iter()
                    .zip(texts.iter())
                    .map(|((start, end), text)| TokenSpan {
                        text: text.clone(),
                        start: *start,
                        end: *end,
                        anchor: self.token_anchor(text),
                    })
                    .collect();
            } else {
                alternatives.push(AlternativeSegmentation {
                    tokens: texts,
                    score: entry.score,
                });
            }
        }

        CausalTokenization {
            text: text.to_string(),
            tokens: best_tokens,
            score: best_score,
            alternatives,
        }
    }
}

fn reconstruct(paths: &[Vec<PathEntry>], end: usize, rank: usize) -> Vec<(usize, usize)> {
    let mut spans: Vec<(usize, usize)> = Vec::new();
    let mut cursor = end;
    let mut current_rank = rank;
    while cursor > 0 {
        let entry = &paths[cursor][current_rank];
        spans.push((entry.start, cursor));
        cursor = entry.start;
        current_rank = entry.prev_rank;
    }
    spans.reverse();
    spans
}

fn is_punctuation(ch: char) -> bool {
    ch.is_ascii_punctuation()
        || matches!(
            ch,
            '，' | '。'
                | '？'
                | '！'
                | '、'
                | '；'
                | '：'
                | '“'
                | '”'
                | '‘'
                | '’'
                | '（'
                | '）'
                | '《'
                | '》'
                | '…'
                | '—'
                | '·'
        )
}

fn atom_mentions(atom: &Atom, canonical: &str) -> bool {
    &*atom.predicate == canonical || atom.terms.iter().any(|term| term.name() == canonical)
}

/// token 级因果归因：把目标结论及其因果链上各原子的概率质量，
/// 按词表锚定关系回溯摊到输入 token 上（纯符号 attention）。
pub fn attribute_tokens(
    tokenization: &CausalTokenization,
    engine: &MarkovLogicEngine,
    graph: &CausalGraph,
    target: &Atom,
) -> Vec<TokenAttribution> {
    let probabilities = engine.probability_map();
    let probability_of = |atom: &Atom| -> f64 {
        if engine.has_fact(atom) {
            1.0
        } else {
            probabilities.get(atom).copied().unwrap_or(0.0)
        }
    };
    let mut weights: HashMap<String, (Atom, f64)> = HashMap::new();
    let mut add_atom = |atom: &Atom, weight: f64| {
        weights
            .entry(atom.display())
            .and_modify(|entry| {
                if weight > entry.1 {
                    entry.1 = weight;
                }
            })
            .or_insert_with(|| (atom.clone(), weight));
    };

    // 目标结论：自身概率 + 入边机制强度
    let mut target_weight = probability_of(target);
    let direct = graph.causes_of(target);
    for edge in &direct {
        target_weight += edge.weight.abs();
    }
    add_atom(target, target_weight);
    // 直接原因：自身概率 + 该机制的强度
    for edge in &direct {
        let weight = probability_of(&edge.cause) * (1.0 + edge.weight.abs());
        add_atom(&edge.cause, weight);
    }
    // 事实根因链：链上每个节点按其平均链强度加权
    let roots: HashSet<Atom> = engine.facts().cloned().collect();
    for chain in graph.root_paths(target, &roots) {
        let bonus = chain.strength().abs() / chain.weights().len().max(1) as f64;
        for node in chain.nodes() {
            let weight = probability_of(node) * (1.0 + bonus);
            add_atom(node, weight);
        }
    }
    let mut weight_list: Vec<(Atom, f64)> = weights.into_values().collect();
    weight_list.sort_by(|a, b| a.0.display().cmp(&b.0.display()));

    let mut attributions: Vec<TokenAttribution> = Vec::new();
    let mut total = 0.0;
    for token in &tokenization.tokens {
        let Some(anchor) = &token.anchor else {
            continue;
        };
        let mut contribution = 0.0;
        for (atom, weight) in &weight_list {
            if atom_mentions(atom, &anchor.canonical) {
                contribution += weight * anchor.support;
            }
        }
        if contribution > 0.0 {
            total += contribution;
            attributions.push(TokenAttribution {
                token: token.text.clone(),
                start: token.start,
                end: token.end,
                anchor: anchor.text(),
                contribution,
                share: 0.0,
            });
        }
    }
    for attribution in &mut attributions {
        attribution.share = if total > 0.0 {
            attribution.contribution / total
        } else {
            0.0
        };
    }
    attributions.sort_by(|a, b| {
        b.contribution
            .partial_cmp(&a.contribution)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.token.cmp(&b.token))
    });
    attributions
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::causal_reasoning::CausalAnalyzer;
    use crate::markov_logic::WeightedRule;
    use crate::resolution_engine::Term;

    fn sample_tokenizer() -> CausalTokenizer {
        let mut tokenizer = CausalTokenizer::new(MAX_TOKEN_CHARS);
        tokenizer.add_anchor("张三", "张三", AnchorKind::Entity, 1.0);
        tokenizer.add_anchor("李四", "李四", AnchorKind::Entity, 1.0);
        tokenizer.add_anchor("师傅", "师傅", AnchorKind::Predicate, 0.9);
        tokenizer.add_alias("师父", "师傅");
        tokenizer
    }

    fn chain_engine() -> MarkovLogicEngine {
        let mut engine = MarkovLogicEngine::new(6, 0.5);
        engine.add_fact(Atom::rel("师傅", "张三", "李四"));
        engine.add_rule(WeightedRule::new(
            Atom::pattern("长辈", Term::var("A"), Term::var("B")),
            vec![Atom::pattern("师傅", Term::var("A"), Term::var("B"))],
            2.0,
            "师傅⇒长辈",
        ));
        engine.add_rule(WeightedRule::new(
            Atom::pattern("尊敬", Term::var("A"), Term::var("B")),
            vec![Atom::pattern("长辈", Term::var("A"), Term::var("B"))],
            1.5,
            "长辈⇒尊敬",
        ));
        engine
    }

    fn chain_tokenizer(engine: &MarkovLogicEngine) -> CausalTokenizer {
        let mut tokenizer = CausalTokenizer::new(MAX_TOKEN_CHARS);
        for fact in engine.facts() {
            tokenizer.observe_fact(fact);
        }
        let analyzer = CausalAnalyzer::from_engine(engine, 4);
        for edge in analyzer.graph().edges() {
            tokenizer.observe_causal_edge(edge);
        }
        tokenizer
    }

    #[test]
    fn test_segmentation_prefers_causally_anchored_spans() {
        let tokenizer = sample_tokenizer();
        let result = tokenizer.tokenize("张三的师父是谁");
        let texts = result.token_texts();
        assert!(texts.contains(&"张三"), "应识别实体：{:?}", texts);
        let shifu = result
            .tokens
            .iter()
            .find(|token| token.text == "师父")
            .expect("别名「师父」应作为整体 token");
        let anchor = shifu.anchor.as_ref().expect("别名应锚定到谓词");
        assert_eq!(anchor.canonical, "师傅");
        assert_eq!(anchor.kind, AnchorKind::Predicate);
        assert!(anchor.via_alias, "应标记为别名命中");
        assert!(result.anchored_count() >= 2);
    }

    #[test]
    fn test_long_entity_anchor_beats_jieba_splitting() {
        let mut tokenizer = CausalTokenizer::new(MAX_TOKEN_CHARS);
        tokenizer.add_anchor("爱新觉罗玄烨", "爱新觉罗玄烨", AnchorKind::Entity, 1.0);
        tokenizer.add_anchor("康熙", "爱新觉罗玄烨", AnchorKind::Entity, 0.95);
        let result = tokenizer.tokenize("爱新觉罗玄烨是谁");
        assert_eq!(
            result.tokens.first().map(|token| token.text.as_str()),
            Some("爱新觉罗玄烨"),
            "长实体锚应压过结巴的默认切分：{:?}",
            result.token_texts()
        );
        let anchor = result.tokens[0].anchor.as_ref().unwrap();
        assert_eq!(anchor.canonical, "爱新觉罗玄烨");
    }

    #[test]
    fn test_alternatives_are_deterministic_and_deduped() {
        let tokenizer = sample_tokenizer();
        let first = tokenizer.tokenize("张三的师父是谁");
        let second = tokenizer.tokenize("张三的师父是谁");
        assert_eq!(first.render_segmentation(), second.render_segmentation());
        let alt_first: Vec<String> = first
            .alternatives
            .iter()
            .map(|alt| alt.tokens.join("|"))
            .collect();
        let alt_second: Vec<String> = second
            .alternatives
            .iter()
            .map(|alt| alt.tokens.join("|"))
            .collect();
        assert_eq!(alt_first, alt_second, "备选分词应确定性");
        let deduped: HashSet<&String> = alt_first.iter().collect();
        assert_eq!(alt_first.len(), deduped.len(), "备选不应重复：{:?}", alt_first);
        for alt in &first.alternatives {
            assert!(alt.score <= first.score + 1e-9);
        }
    }

    #[test]
    fn test_unknown_text_falls_back_to_characters() {
        let tokenizer = CausalTokenizer::new(MAX_TOKEN_CHARS);
        let result = tokenizer.tokenize("xyz");
        assert_eq!(result.render_segmentation().replace(" | ", ""), "xyz");
        assert!(result.tokens.iter().all(|token| token.anchor.is_none()));
    }

    #[test]
    fn test_causal_mass_accumulates_on_anchors() {
        let engine = chain_engine();
        let tokenizer = chain_tokenizer(&engine);
        let shifu = tokenizer
            .resolve("师傅")
            .expect("应登记谓词锚");
        assert!(shifu.anchor.causal_mass > 0.0, "谓词应累计因果质量");
        assert!(shifu.anchor.causal_edges > 0);
        let zhang = tokenizer.resolve("张三").unwrap();
        assert!(zhang.anchor.causal_mass > 0.0, "实体也应累计因果质量");
    }

    #[test]
    fn test_attribution_ranks_input_tokens_and_normalizes() {
        let engine = chain_engine();
        let tokenizer = chain_tokenizer(&engine);
        let tokenization = tokenizer.tokenize("张三的师傅是谁");
        let analyzer = CausalAnalyzer::from_engine(&engine, 4);
        let target = Atom::rel("尊敬", "张三", "李四");
        let attributions = attribute_tokens(&tokenization, &engine, analyzer.graph(), &target);
        assert!(!attributions.is_empty(), "应产生 token 归因");
        let total: f64 = attributions.iter().map(|item| item.share).sum();
        assert!((total - 1.0).abs() < 1e-9, "占比应归一：{}", total);
        let first = &attributions[0];
        assert_eq!(first.token, "张三", "张三同时锚定目标与链上原子，应排第一");
        assert!(attributions.iter().all(|item| item.contribution > 0.0));
        assert!(
            attributions.iter().all(|item| item.token != "李四"),
            "未出现在输入中的实体不应被归因"
        );
    }

    #[test]
    fn test_attribution_empty_when_no_token_anchors_target() {
        let engine = chain_engine();
        let tokenizer = chain_tokenizer(&engine);
        let tokenization = tokenizer.tokenize("完全无关的话");
        let analyzer = CausalAnalyzer::from_engine(&engine, 4);
        let target = Atom::rel("尊敬", "张三", "李四");
        let attributions = attribute_tokens(&tokenization, &engine, analyzer.graph(), &target);
        assert!(attributions.is_empty());
    }
}

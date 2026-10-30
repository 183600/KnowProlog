//! DSI-M1：依存脊索引（Dependency Spine Index）。
//!
//! 现有链路在关系抽取后就把句法结构丢掉了：主动句、被动句、把字句和
//! 定语从句即便表达同一事实，也往往落到不同的词面匹配路径。本模块补上
//! 一个纯符号、零训练的句法键层：
//!
//! 1. [`analyze`] 用固定规则从结巴标注构造依存脊，覆盖七类高频构式：
//!    主动、被动（被/由）、把字、判断（是）、领属（的）、关系从句（的）
//!    与并列（和/与/及）；
//! 2. [`compile_frames`] 把依存树归一到 `(主语, 谓词, 宾语)` 规范框架，
//!    同时保留极性、构式、证据跨度和回退原因；
//! 3. [`SpineIndex`] 同时索引规范边与表层边，支持带变量、多跳、带预算的
//!    合取查询；查询计划按倒排表选择性排序，回执记录探测、连接与省略，
//!    不依赖挂钟即可复现同一操作计数。
//!
//! 设计约束与仓库其他模块一致：同输入同字节、不写知识图谱、不训练权重。
//! 低置信结构只作为候选，并显式写入 [`DependencyTree::warnings`]，不假装
//! 句法分析总是确定无疑。

use crate::context_image::fnv1a_hex;
use std::collections::{BTreeMap, BTreeSet, HashMap};

use serde::Serialize;

use crate::nlp::shared_jieba;
use crate::pos_ontology;

/// 规范框架的句法构式。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Construction {
    Active,
    Passive,
    Ba,
    Copula,
    Relative,
    Coordinated,
    NonProjective,
    Unknown,
}

impl Construction {
    pub fn code(self) -> &'static str {
        match self {
            Construction::Active => "active",
            Construction::Passive => "passive",
            Construction::Ba => "ba",
            Construction::Copula => "copula",
            Construction::Relative => "relative",
            Construction::Coordinated => "coordinated",
            Construction::NonProjective => "non_projective",
            Construction::Unknown => "unknown",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Construction::Active => "主动",
            Construction::Passive => "被动",
            Construction::Ba => "把字",
            Construction::Copula => "判断",
            Construction::Relative => "关系从句",
            Construction::Coordinated => "并列",
            Construction::NonProjective => "非投射回退",
            Construction::Unknown => "未知",
        }
    }
}

/// 规范框架的极性。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FramePolarity {
    Affirmative,
    Negative,
}

impl FramePolarity {
    pub fn code(self) -> &'static str {
        match self {
            FramePolarity::Affirmative => "affirmative",
            FramePolarity::Negative => "negative",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            FramePolarity::Affirmative => "肯定",
            FramePolarity::Negative => "否定",
        }
    }
}

/// 依存边标签。标签集刻意保持在 20 个以内，避免规则爆炸。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DepLabel {
    Root,
    NSubj,
    NSubjPass,
    Obj,
    Agent,
    Poss,
    Rel,
    Attr,
    Amod,
    Advmod,
    Neg,
    Aspect,
    Prep,
    Pobj,
    Conj,
    Cc,
    Ba,
    Passive,
    Nummod,
    Topic,
    Other,
}

impl DepLabel {
    pub fn code(self) -> &'static str {
        match self {
            DepLabel::Root => "root",
            DepLabel::NSubj => "nsubj",
            DepLabel::NSubjPass => "nsubjpass",
            DepLabel::Obj => "obj",
            DepLabel::Agent => "agent",
            DepLabel::Poss => "poss",
            DepLabel::Rel => "rel",
            DepLabel::Attr => "attr",
            DepLabel::Amod => "amod",
            DepLabel::Advmod => "advmod",
            DepLabel::Neg => "neg",
            DepLabel::Aspect => "aspect",
            DepLabel::Prep => "prep",
            DepLabel::Pobj => "pobj",
            DepLabel::Conj => "conj",
            DepLabel::Cc => "cc",
            DepLabel::Ba => "ba",
            DepLabel::Passive => "passive",
            DepLabel::Nummod => "nummod",
            DepLabel::Topic => "topic",
            DepLabel::Other => "dep",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            DepLabel::Root => "根",
            DepLabel::NSubj => "主语",
            DepLabel::NSubjPass => "受事主语",
            DepLabel::Obj => "宾语",
            DepLabel::Agent => "施事",
            DepLabel::Poss => "领属",
            DepLabel::Rel => "关系从句",
            DepLabel::Attr => "定语标记",
            DepLabel::Amod => "形容词修饰",
            DepLabel::Advmod => "状语修饰",
            DepLabel::Neg => "否定",
            DepLabel::Aspect => "体标记",
            DepLabel::Prep => "介词",
            DepLabel::Pobj => "介词宾语",
            DepLabel::Conj => "并列",
            DepLabel::Cc => "并列连词",
            DepLabel::Ba => "把字标记",
            DepLabel::Passive => "被动标记",
            DepLabel::Nummod => "数量修饰",
            DepLabel::Topic => "话题",
            DepLabel::Other => "其他依存",
        }
    }
}

/// 词元类别。仅用于确定性规则，不声明词性标注完全正确。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenKind {
    Noun,
    ProperNoun,
    Pronoun,
    Verb,
    Copula,
    Adjective,
    Adverb,
    Preposition,
    Particle,
    Aspect,
    Conjunction,
    Negation,
    PassiveMarker,
    BaMarker,
    Wh,
    Number,
    Measure,
    Punctuation,
    Other,
}

impl TokenKind {
    pub fn code(self) -> &'static str {
        match self {
            TokenKind::Noun => "n",
            TokenKind::ProperNoun => "nr",
            TokenKind::Pronoun => "r",
            TokenKind::Verb => "v",
            TokenKind::Copula => "cop",
            TokenKind::Adjective => "a",
            TokenKind::Adverb => "d",
            TokenKind::Preposition => "p",
            TokenKind::Particle => "u",
            TokenKind::Aspect => "asp",
            TokenKind::Conjunction => "c",
            TokenKind::Negation => "neg",
            TokenKind::PassiveMarker => "pass",
            TokenKind::BaMarker => "ba",
            TokenKind::Wh => "wh",
            TokenKind::Number => "m",
            TokenKind::Measure => "q",
            TokenKind::Punctuation => "w",
            TokenKind::Other => "x",
        }
    }

    pub fn is_nominal(self) -> bool {
        matches!(
            self,
            TokenKind::Noun
                | TokenKind::ProperNoun
                | TokenKind::Pronoun
                | TokenKind::Wh
                | TokenKind::Number
                | TokenKind::Measure
                | TokenKind::Adjective
        )
    }

    pub fn is_content(self) -> bool {
        self.is_nominal()
            || matches!(
                self,
                TokenKind::Verb | TokenKind::Copula | TokenKind::Adverb
            )
    }
}

/// 带字符跨度的词元。
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SpineToken {
    pub text: String,
    pub pos: String,
    pub kind: TokenKind,
    pub start: usize,
    pub end: usize,
}

impl SpineToken {
    pub fn render(&self) -> String {
        format!("{}[{}]", self.text, self.kind.code())
    }
}

/// 一条有向依存边：`head --label--> dep`。
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DependencyEdge {
    pub head: usize,
    pub dep: usize,
    pub label: DepLabel,
    pub confidence: f64,
    pub evidence: Vec<String>,
}

impl DependencyEdge {
    pub fn render(&self, tokens: &[SpineToken]) -> String {
        let head = tokens
            .get(self.head)
            .map(|t| t.text.as_str())
            .unwrap_or("?");
        let dep = tokens.get(self.dep).map(|t| t.text.as_str()).unwrap_or("?");
        format!("{} --{}--> {}", head, self.label.code(), dep)
    }
}

/// 一个句内名词短语（含中心词与字符跨度）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Phrase {
    pub text: String,
    pub head: String,
    pub head_index: usize,
    pub start: usize,
    pub end: usize,
}

impl Phrase {
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }
}

/// 确定性依存分析结果。
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DependencyTree {
    pub sentence: String,
    pub tokens: Vec<SpineToken>,
    pub edges: Vec<DependencyEdge>,
    pub root: Option<usize>,
    pub construction: Construction,
    /// 归一后的主语（被动/把字已换位）。
    pub subject: Option<Phrase>,
    /// 归一后的宾语。
    pub object: Option<Phrase>,
    /// 被动句表层受事主语，用于解释与回执。
    pub surface_subject: Option<Phrase>,
    pub warnings: Vec<String>,
}

impl DependencyTree {
    pub fn empty(sentence: &str) -> Self {
        Self {
            sentence: sentence.to_string(),
            tokens: Vec::new(),
            edges: Vec::new(),
            root: None,
            construction: Construction::Unknown,
            subject: None,
            object: None,
            surface_subject: None,
            warnings: vec!["空句子".to_string()],
        }
    }

    pub fn token(&self, index: usize) -> Option<&SpineToken> {
        self.tokens.get(index)
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn head_of(&self, dep: usize) -> Option<&DependencyEdge> {
        self.edges
            .iter()
            .find(|edge| edge.dep == dep && edge.head != dep)
    }

    pub fn children(&self, head: usize) -> Vec<&DependencyEdge> {
        let mut edges: Vec<&DependencyEdge> = self
            .edges
            .iter()
            .filter(|edge| edge.head == head && edge.dep != head)
            .collect();
        edges.sort_by_key(|edge| (edge.dep, edge.label));
        edges
    }

    pub fn is_clean(&self) -> bool {
        self.warnings.is_empty()
    }

    pub fn render_text(&self) -> String {
        let mut text = format!(
            "句子：{}\n构式：{}（{}）\n",
            self.sentence,
            self.construction.label(),
            self.construction.code()
        );
        if let Some(root) = self.root {
            if let Some(token) = self.token(root) {
                text.push_str(&format!("根词：{}（{}）\n", token.text, token.pos));
            }
        }
        for edge in &self.edges {
            text.push_str("  ");
            text.push_str(&edge.render(&self.tokens));
            text.push('\n');
        }
        if let Some(subject) = &self.subject {
            text.push_str(&format!("规范主语：{}\n", subject.text));
        }
        if let Some(object) = &self.object {
            text.push_str(&format!("规范宾语：{}\n", object.text));
        }
        for warning in &self.warnings {
            text.push_str(&format!("警告：{}\n", warning));
        }
        text
    }
}

/// 规范框架中的一个词项：表层短语、中心词、词性与跨度。
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FrameTerm {
    pub text: String,
    pub head: String,
    pub pos: String,
    pub start: usize,
    pub end: usize,
}

impl FrameTerm {
    pub fn empty() -> Self {
        Self {
            text: String::new(),
            head: String::new(),
            pos: String::new(),
            start: 0,
            end: 0,
        }
    }

    pub fn from_phrase(phrase: &Phrase, tokens: &[SpineToken]) -> Self {
        let pos = tokens
            .get(phrase.head_index)
            .map(|token| token.pos.clone())
            .unwrap_or_default();
        Self {
            text: phrase.text.clone(),
            head: phrase.head.clone(),
            pos,
            start: phrase.start,
            end: phrase.end,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    pub fn keys(&self) -> Vec<&str> {
        let mut keys = vec![self.text.as_str()];
        if self.head != self.text {
            keys.push(self.head.as_str());
        }
        keys
    }
}

/// 框架证据：句子、角色跨度与参与的依存边。
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FrameEvidence {
    pub sentence: String,
    pub root_token: usize,
    pub subject_span: Option<(usize, usize)>,
    pub predicate_span: Option<(usize, usize)>,
    pub object_span: Option<(usize, usize)>,
    pub edges: Vec<usize>,
}

/// 从依存树归一出的规范框架。
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CanonicalFrame {
    pub id: String,
    pub subject: FrameTerm,
    pub predicate: FrameTerm,
    pub object: FrameTerm,
    pub polarity: FramePolarity,
    pub construction: Construction,
    pub flags: Vec<String>,
    pub confidence: f64,
    pub evidence: FrameEvidence,
}

impl CanonicalFrame {
    /// 没有源句时，用三元组合成一个最小规范框架，保证仍可被脊索引查询。
    pub fn from_triple(subject: &str, predicate: &str, object: &str) -> Self {
        let subject = subject.trim();
        let predicate = predicate.trim();
        let object = object.trim();
        let subject_term = FrameTerm {
            text: subject.to_string(),
            head: subject.to_string(),
            pos: "n".to_string(),
            start: 0,
            end: subject.len(),
        };
        let predicate_term = FrameTerm {
            text: predicate.to_string(),
            head: predicate.to_string(),
            pos: "v".to_string(),
            start: 0,
            end: predicate.len(),
        };
        let object_term = FrameTerm {
            text: object.to_string(),
            head: object.to_string(),
            pos: "n".to_string(),
            start: 0,
            end: object.len(),
        };
        let material = format!(
            "synthetic|{}|{}|{}|affirmative|active|0",
            subject, predicate, object
        );
        Self {
            id: format!("dsi:{}", fnv1a_hex(material.as_bytes())),
            subject: subject_term,
            predicate: predicate_term,
            object: object_term,
            polarity: FramePolarity::Affirmative,
            construction: Construction::Active,
            flags: vec!["synthetic_frame".to_string()],
            confidence: 1.0,
            evidence: FrameEvidence {
                sentence: String::new(),
                root_token: 0,
                subject_span: None,
                predicate_span: None,
                object_span: None,
                edges: Vec::new(),
            },
        }
    }

    pub fn is_complete(&self) -> bool {
        !self.subject.is_empty() && !self.predicate.is_empty() && !self.object.is_empty()
    }

    pub fn render_text(&self) -> String {
        let subject = if self.subject.is_empty() {
            "∅"
        } else {
            self.subject.text.as_str()
        };
        let object = if self.object.is_empty() {
            "∅"
        } else {
            self.object.text.as_str()
        };
        format!(
            "{} -[{}]-> {}（{}；{}；{}；置信 {:.2}）",
            subject,
            self.predicate.text,
            object,
            self.construction.label(),
            self.polarity.label(),
            if self.flags.is_empty() {
                "无标记".to_string()
            } else {
                self.flags.join(",")
            },
            self.confidence
        )
    }
}

/// 单句分析。
pub fn analyze(sentence: &str) -> DependencyTree {
    let sentence = sentence.trim();
    if sentence.is_empty() {
        return DependencyTree::empty(sentence);
    }
    let tokens = tokenize(sentence);
    if tokens.is_empty() {
        return DependencyTree::empty(sentence);
    }
    build_tree(sentence, tokens)
}

/// 按中文标点与换行切句后逐句分析。
pub fn analyze_sentences(text: &str) -> Vec<DependencyTree> {
    split_sentences(text)
        .into_iter()
        .filter(|sentence| !sentence.trim().is_empty())
        .map(|sentence| analyze(&sentence))
        .collect()
}

/// 从整段文本抽取规范框架。
pub fn extract_frames(text: &str) -> Vec<CanonicalFrame> {
    let mut frames = Vec::new();
    for tree in analyze_sentences(text) {
        frames.extend(compile_frames(&tree));
    }
    frames
}

fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();
    for (index, ch) in chars.iter().copied().enumerate() {
        let hard_boundary = matches!(ch, '。' | '！' | '？' | '；' | '!' | '?' | ';' | '\n');
        let soft_boundary = matches!(ch, '，' | ',')
            && !chars
                .get(index.wrapping_sub(1))
                .map(|previous| previous.is_ascii_digit())
                .unwrap_or(false)
            && !chars
                .get(index + 1)
                .map(|next| next.is_ascii_digit())
                .unwrap_or(false);
        if hard_boundary || soft_boundary {
            if !current.trim().is_empty() {
                sentences.push(current.trim().to_string());
            }
            current.clear();
        } else {
            current.push(ch);
        }
    }
    if !current.trim().is_empty() {
        sentences.push(current.trim().to_string());
    }
    sentences
}

fn tokenize(sentence: &str) -> Vec<SpineToken> {
    let mut tokens = Vec::new();
    let mut cursor = 0usize;
    for tag in shared_jieba().tag(sentence, false) {
        let word = tag.word;
        if word.trim().is_empty() {
            cursor += word.len();
            continue;
        }
        let start = match sentence[cursor..].find(word) {
            Some(offset) => cursor + offset,
            None => cursor,
        };
        let end = start + word.len();
        cursor = end;
        tokens.push(SpineToken {
            text: word.to_string(),
            pos: tag.tag.to_string(),
            kind: classify_token(word, tag.tag),
            start,
            end,
        });
    }
    tokens
}

fn classify_token(word: &str, pos: &str) -> TokenKind {
    if is_punctuation(word) {
        return TokenKind::Punctuation;
    }
    match word {
        "是" | "为" | "乃" | "系" => return TokenKind::Copula,
        "被" | "由" | "让" | "叫" => return TokenKind::PassiveMarker,
        "把" | "将" => return TokenKind::BaMarker,
        "不" | "没" | "没有" | "未" | "无" | "别" | "莫" => return TokenKind::Negation,
        "和" | "与" | "及" | "跟" | "同" | "以及" => return TokenKind::Conjunction,
        "的" | "地" | "得" => return TokenKind::Particle,
        "了" | "着" | "过" | "已" | "已经" | "曾" | "曾经" => return TokenKind::Aspect,
        "我" | "你" | "他" | "她" | "它" | "我们" | "你们" | "他们" | "她们" | "它们" | "这"
        | "那" | "这个" | "那个" | "这些" | "那些" | "自己" | "其" => {
            return TokenKind::Pronoun
        }
        "谁" | "什么" | "哪" | "哪个" | "哪些" | "哪里" | "哪儿" | "何处" | "何时" | "几时"
        | "多少" | "几" | "怎样" | "怎么" | "如何" | "为什么" => return TokenKind::Wh,
        "在" | "从" | "对" | "向" | "给" | "于" | "用" => return TokenKind::Preposition,
        _ => {}
    }
    if pos.starts_with("nr") || pos.starts_with("nz") || pos.starts_with("ns") {
        return TokenKind::ProperNoun;
    }
    if pos.starts_with('n') {
        return TokenKind::Noun;
    }
    if pos.starts_with('v') {
        return TokenKind::Verb;
    }
    if pos.starts_with('a') {
        return TokenKind::Adjective;
    }
    if pos.starts_with('d') {
        return TokenKind::Adverb;
    }
    if pos.starts_with('p') {
        return TokenKind::Preposition;
    }
    if pos.starts_with('u') {
        return TokenKind::Particle;
    }
    if pos.starts_with('m') {
        return TokenKind::Number;
    }
    if pos_ontology::tag_matches("n", pos) {
        return TokenKind::Noun;
    }
    if pos_ontology::tag_matches("v", pos) {
        return TokenKind::Verb;
    }
    if pos_ontology::tag_matches("a", pos) {
        return TokenKind::Adjective;
    }
    if pos.starts_with('q') {
        return TokenKind::Measure;
    }
    if pos.starts_with('r') {
        return TokenKind::Pronoun;
    }
    if pos.starts_with('c') {
        return TokenKind::Conjunction;
    }
    TokenKind::Other
}

fn is_punctuation(text: &str) -> bool {
    !text.is_empty()
        && text.chars().all(|ch| {
            ch.is_ascii_punctuation()
                || matches!(
                    ch,
                    '，' | '。'
                        | '、'
                        | '；'
                        | '：'
                        | '？'
                        | '！'
                        | '“'
                        | '”'
                        | '‘'
                        | '’'
                        | '（'
                        | '）'
                        | '《'
                        | '》'
                        | '—'
                        | '…'
                        | '·'
                )
        })
}

fn build_tree(sentence: &str, tokens: Vec<SpineToken>) -> DependencyTree {
    let mut warnings = Vec::new();
    let mut edges = Vec::new();
    let mut parents: Vec<Option<usize>> = vec![None; tokens.len()];

    let passive_index = tokens
        .iter()
        .position(|token| token.kind == TokenKind::PassiveMarker);
    let ba_index = tokens
        .iter()
        .position(|token| token.kind == TokenKind::BaMarker);
    let root = choose_root(&tokens);

    let Some(root_index) = root else {
        warnings.push("未找到谓词或中心名词".to_string());
        return DependencyTree {
            sentence: sentence.to_string(),
            tokens,
            edges,
            root: None,
            construction: Construction::Unknown,
            subject: None,
            object: None,
            surface_subject: None,
            warnings,
        };
    };

    let mut construction = match (passive_index, ba_index) {
        (Some(_), Some(_)) => Construction::NonProjective,
        (Some(_), None) => Construction::Passive,
        (None, Some(_)) => Construction::Ba,
        (None, None) if tokens[root_index].kind == TokenKind::Copula => Construction::Copula,
        (None, None) => Construction::Active,
    };
    if passive_index.is_some() && ba_index.is_some() {
        warnings.push("同时出现被动与把字标记，退化为非投射读法".to_string());
    }

    let mut subject = None;
    let mut object = None;
    let mut surface_subject = None;

    match construction {
        Construction::Passive => {
            let marker = passive_index.unwrap_or(root_index);
            if let Some(patient) = noun_phrase_left(&tokens, marker, 0) {
                surface_subject = Some(patient);
            }
            if marker + 1 < root_index {
                if let Some(agent) = noun_phrase_left(&tokens, root_index, marker + 1) {
                    subject = Some(agent);
                } else {
                    warnings.push("被动句未找到施事，保留受事为宾语".to_string());
                }
            } else {
                warnings.push("被动句省略施事".to_string());
            }
            object = surface_subject.clone();
        }
        Construction::Ba => {
            let marker = ba_index.unwrap_or(root_index);
            if let Some(agent) = noun_phrase_left(&tokens, marker, 0) {
                subject = Some(agent);
            }
            if marker + 1 < root_index {
                if let Some(patient) = noun_phrase_left(&tokens, root_index, marker + 1) {
                    object = Some(patient);
                } else {
                    warnings.push("把字句未找到前置宾语".to_string());
                }
            }
        }
        _ => {
            if let Some(subject_phrase) = noun_phrase_left(&tokens, root_index, 0) {
                subject = Some(subject_phrase);
            }
            if let Some(object_phrase) = noun_phrase_right(&tokens, root_index + 1, tokens.len()) {
                object = Some(object_phrase);
            }
            if construction == Construction::Copula && subject.is_none() {
                warnings.push("判断句未找到主语".to_string());
            }
        }
    }

    if construction == Construction::Passive {
        if let Some(agent) = &subject {
            attach_phrase_edges(
                &tokens,
                agent,
                root_index,
                DepLabel::Agent,
                0.72,
                &mut edges,
                &mut parents,
            );
        }
        if let Some(patient) = &surface_subject {
            attach_phrase_edges(
                &tokens,
                patient,
                root_index,
                DepLabel::NSubjPass,
                0.72,
                &mut edges,
                &mut parents,
            );
        }
    } else {
        if let Some(subject_phrase) = &subject {
            attach_phrase_edges(
                &tokens,
                subject_phrase,
                root_index,
                DepLabel::NSubj,
                0.86,
                &mut edges,
                &mut parents,
            );
        }
        if let Some(object_phrase) = &object {
            attach_phrase_edges(
                &tokens,
                object_phrase,
                root_index,
                DepLabel::Obj,
                0.86,
                &mut edges,
                &mut parents,
            );
        }
    }

    if construction == Construction::Passive {
        if let Some(marker) = passive_index {
            add_edge(
                &mut edges,
                &mut parents,
                root_index,
                marker,
                DepLabel::Passive,
                0.9,
                vec!["被动标记".to_string()],
            );
        }
    }
    if construction == Construction::Ba {
        if let Some(marker) = ba_index {
            add_edge(
                &mut edges,
                &mut parents,
                root_index,
                marker,
                DepLabel::Ba,
                0.9,
                vec!["把字标记".to_string()],
            );
        }
    }

    for (index, token) in tokens.iter().enumerate() {
        if index == root_index {
            continue;
        }
        match token.kind {
            TokenKind::Negation => {
                add_edge(
                    &mut edges,
                    &mut parents,
                    root_index,
                    index,
                    DepLabel::Neg,
                    0.92,
                    vec!["否定词直接修饰谓词".to_string()],
                );
            }
            TokenKind::Aspect if index > root_index => {
                add_edge(
                    &mut edges,
                    &mut parents,
                    root_index,
                    index,
                    DepLabel::Aspect,
                    0.85,
                    vec!["体标记依附谓词".to_string()],
                );
            }
            TokenKind::Adverb if index < root_index => {
                add_edge(
                    &mut edges,
                    &mut parents,
                    root_index,
                    index,
                    DepLabel::Advmod,
                    0.7,
                    vec!["状语依附谓词".to_string()],
                );
            }
            TokenKind::Preposition if index < root_index => {
                add_edge(
                    &mut edges,
                    &mut parents,
                    root_index,
                    index,
                    DepLabel::Prep,
                    0.7,
                    vec!["介词短语依附谓词".to_string()],
                );
                if let Some(obj) = noun_phrase_right(&tokens, index + 1, root_index.max(index + 1))
                {
                    add_edge(
                        &mut edges,
                        &mut parents,
                        index,
                        obj.head_index,
                        DepLabel::Pobj,
                        0.75,
                        vec!["介词宾语".to_string()],
                    );
                }
            }
            TokenKind::Conjunction => {
                if let Some(left) = noun_phrase_left(&tokens, index, 0) {
                    add_edge(
                        &mut edges,
                        &mut parents,
                        left.head_index,
                        index,
                        DepLabel::Cc,
                        0.72,
                        vec!["并列连词".to_string()],
                    );
                }
            }
            _ => {}
        }
    }

    // 让所有实词都有归属：未分配的实词挂到根词下，并记录回退。
    for (index, token) in tokens.iter().enumerate() {
        if index == root_index || token.kind == TokenKind::Punctuation {
            continue;
        }
        if parents[index].is_none() {
            add_edge(
                &mut edges,
                &mut parents,
                root_index,
                index,
                DepLabel::Other,
                0.45,
                vec!["规则未覆盖，挂到根词保留候选".to_string()],
            );
        }
    }

    if construction == Construction::Active && subject.is_some() && object.is_none() {
        warnings.push("主动句未找到宾语，框架可能不完整".to_string());
    }
    if construction == Construction::NonProjective {
        warnings.push("non_projective_fallback：保留局部边，不强行造一棵投射树".to_string());
    }
    if edges.iter().any(|edge| edge.label == DepLabel::Rel) {
        construction = Construction::Relative;
    }

    DependencyTree {
        sentence: sentence.to_string(),
        tokens,
        edges,
        root: Some(root_index),
        construction,
        subject,
        object,
        surface_subject,
        warnings,
    }
}

fn choose_root(tokens: &[SpineToken]) -> Option<usize> {
    if let Some(copula) = tokens
        .iter()
        .rposition(|token| token.kind == TokenKind::Copula)
    {
        return Some(copula);
    }
    if let Some(verb) = tokens
        .iter()
        .position(|token| token.kind == TokenKind::Verb)
    {
        return Some(verb);
    }
    let adjective_predicate = tokens.iter().enumerate().find_map(|(index, token)| {
        if token.kind != TokenKind::Adjective {
            return None;
        }
        let next = tokens.get(index + 1).map(|next| next.kind);
        match next {
            Some(TokenKind::Noun | TokenKind::ProperNoun | TokenKind::Particle) | None => None,
            Some(_) => Some(index),
        }
    });
    if let Some(adjective) = adjective_predicate {
        return Some(adjective);
    }
    tokens
        .iter()
        .position(|token| token.kind == TokenKind::Noun || token.kind == TokenKind::ProperNoun)
}

fn noun_phrase_left(tokens: &[SpineToken], end: usize, floor: usize) -> Option<Phrase> {
    if end == 0 {
        return None;
    }
    let mut upper = end.min(tokens.len());
    while upper > floor {
        let token = &tokens[upper - 1];
        if token.kind == TokenKind::Punctuation
            || (token.kind == TokenKind::Particle && token.text != "的")
            || token.kind == TokenKind::Negation
            || token.kind == TokenKind::Adverb
            || token.kind == TokenKind::Aspect
        {
            upper -= 1;
        } else {
            break;
        }
    }
    if upper <= floor {
        return None;
    }

    let mut start = upper;
    let mut saw_content = false;
    let mut after_de = false;
    let mut index = upper;
    while index > floor {
        let token = &tokens[index - 1];
        match token.kind {
            TokenKind::Noun
            | TokenKind::ProperNoun
            | TokenKind::Pronoun
            | TokenKind::Wh
            | TokenKind::Number
            | TokenKind::Measure
            | TokenKind::Adjective
            | TokenKind::Adverb => {
                start = index - 1;
                saw_content = true;
                after_de = false;
                index -= 1;
            }
            TokenKind::Particle if token.text == "的" => {
                start = index - 1;
                after_de = true;
                index -= 1;
            }
            TokenKind::Verb | TokenKind::Copula | TokenKind::Negation if after_de => {
                start = index - 1;
                saw_content = true;
                after_de = false;
                index -= 1;
            }
            TokenKind::Conjunction if saw_content => {
                start = index - 1;
                index -= 1;
            }
            _ => break,
        }
    }

    if !saw_content {
        return None;
    }
    phrase_from_span(tokens, start, upper)
}

fn noun_phrase_right(tokens: &[SpineToken], start: usize, ceiling: usize) -> Option<Phrase> {
    let mut index = start.min(tokens.len());
    let ceiling = ceiling.min(tokens.len());
    while index < ceiling
        && (tokens[index].kind == TokenKind::Aspect
            || (tokens[index].kind == TokenKind::Particle && tokens[index].text != "的"))
    {
        index += 1;
    }
    let phrase_start = index;
    let mut end = index;
    let mut saw_content = false;
    while index < ceiling {
        let token = &tokens[index];
        match token.kind {
            TokenKind::Noun
            | TokenKind::ProperNoun
            | TokenKind::Pronoun
            | TokenKind::Wh
            | TokenKind::Number
            | TokenKind::Measure
            | TokenKind::Adjective
            | TokenKind::Adverb => {
                saw_content = true;
                index += 1;
                end = index;
            }
            TokenKind::Particle if token.text == "的" => {
                index += 1;
                end = index;
            }
            TokenKind::Conjunction => {
                index += 1;
                end = index;
            }
            TokenKind::Verb | TokenKind::Copula => break,
            _ => break,
        }
    }
    if !saw_content {
        return None;
    }
    phrase_from_span(tokens, phrase_start, end)
}

fn phrase_from_span(tokens: &[SpineToken], start: usize, end: usize) -> Option<Phrase> {
    if start >= end || end > tokens.len() {
        return None;
    }
    let head_index = (start..end)
        .rev()
        .find(|index| {
            matches!(
                tokens[*index].kind,
                TokenKind::Noun | TokenKind::ProperNoun | TokenKind::Pronoun
            )
        })
        .unwrap_or_else(|| {
            (start..end)
                .rev()
                .find(|index| tokens[*index].kind.is_nominal())
                .unwrap_or(start)
        });
    let text = tokens[start..end]
        .iter()
        .map(|token| token.text.as_str())
        .collect::<Vec<_>>()
        .join("");
    Some(Phrase {
        text,
        head: tokens[head_index].text.clone(),
        head_index,
        start: tokens[start].start,
        end: tokens[end - 1].end,
    })
}

fn attach_phrase_edges(
    tokens: &[SpineToken],
    phrase: &Phrase,
    root: usize,
    label: DepLabel,
    confidence: f64,
    edges: &mut Vec<DependencyEdge>,
    parents: &mut [Option<usize>],
) {
    add_edge(
        edges,
        parents,
        root,
        phrase.head_index,
        label,
        confidence,
        vec![format!("{}：{}", label.label(), phrase.text)],
    );
    let start = tokens
        .iter()
        .position(|token| token.start == phrase.start)
        .unwrap_or(phrase.head_index);
    let end = tokens
        .iter()
        .position(|token| token.end == phrase.end)
        .map(|index| index + 1)
        .unwrap_or(phrase.head_index + 1);
    attach_phrase_internal(tokens, start, end, phrase.head_index, edges, parents);
}

fn attach_phrase_internal(
    tokens: &[SpineToken],
    start: usize,
    end: usize,
    head: usize,
    edges: &mut Vec<DependencyEdge>,
    parents: &mut [Option<usize>],
) {
    for index in start..end {
        if index == head {
            continue;
        }
        let token = &tokens[index];
        let (label, evidence) = match token.kind {
            TokenKind::Particle if token.text == "的" => {
                (DepLabel::Attr, "定语标记依附中心词".to_string())
            }
            TokenKind::Verb | TokenKind::Copula => {
                (DepLabel::Rel, "关系从句谓词依附中心词".to_string())
            }
            TokenKind::Adjective => (DepLabel::Amod, "形容词修饰中心词".to_string()),
            TokenKind::Adverb => (DepLabel::Advmod, "状语修饰中心词".to_string()),
            TokenKind::Conjunction => (DepLabel::Cc, "并列连词".to_string()),
            TokenKind::Number | TokenKind::Measure => {
                (DepLabel::Nummod, "数量修饰中心词".to_string())
            }
            TokenKind::Negation => (DepLabel::Neg, "否定修饰中心词".to_string()),
            TokenKind::Noun | TokenKind::ProperNoun | TokenKind::Pronoun => {
                (DepLabel::Poss, "名词修饰语依附中心词".to_string())
            }
            _ => (DepLabel::Other, "短语内部回退依存".to_string()),
        };
        add_edge(edges, parents, head, index, label, 0.7, vec![evidence]);
        if matches!(token.kind, TokenKind::Verb | TokenKind::Copula) {
            if let Some(subject) = noun_phrase_left(tokens, index, start) {
                add_edge(
                    edges,
                    parents,
                    index,
                    subject.head_index,
                    DepLabel::NSubj,
                    0.68,
                    vec!["关系从句内部主语".to_string()],
                );
            }
        }
    }
}

fn add_edge(
    edges: &mut Vec<DependencyEdge>,
    parents: &mut [Option<usize>],
    head: usize,
    dep: usize,
    label: DepLabel,
    confidence: f64,
    evidence: Vec<String>,
) {
    if head == dep || dep >= parents.len() || head >= parents.len() {
        return;
    }
    if parents[dep].is_some() {
        return;
    }
    parents[dep] = Some(head);
    edges.push(DependencyEdge {
        head,
        dep,
        label,
        confidence,
        evidence,
    });
}

/// 把一个依存树编译成规范框架；并列短语会展开成多个框架。
fn tree_polarity(tree: &DependencyTree) -> FramePolarity {
    let Some(root_index) = tree.root else {
        return FramePolarity::Affirmative;
    };
    if tree
        .tokens
        .iter()
        .take(root_index)
        .any(|token| token.kind == TokenKind::Negation)
        || tree
            .tokens
            .iter()
            .skip(root_index + 1)
            .take_while(|token| token.kind != TokenKind::Punctuation)
            .any(|token| token.kind == TokenKind::Negation)
    {
        FramePolarity::Negative
    } else {
        FramePolarity::Affirmative
    }
}

pub fn compile_frames(tree: &DependencyTree) -> Vec<CanonicalFrame> {
    let Some(root_index) = tree.root else {
        return Vec::new();
    };
    let Some(root_token) = tree.token(root_index) else {
        return Vec::new();
    };
    let predicate = FrameTerm {
        text: root_token.text.clone(),
        head: root_token.text.clone(),
        pos: root_token.pos.clone(),
        start: root_token.start,
        end: root_token.end,
    };
    let subject_parts = tree
        .subject
        .as_ref()
        .map(|phrase| phrase_parts(tree, phrase))
        .unwrap_or_default();
    let object_parts = tree
        .object
        .as_ref()
        .map(|phrase| phrase_parts(tree, phrase))
        .unwrap_or_default();
    let polarity = tree_polarity(tree);
    let base_confidence = 0.9
        - (tree.warnings.len() as f64 * 0.12).min(0.4)
        - if tree.subject.is_none() { 0.2 } else { 0.0 }
        - if tree.object.is_none() { 0.15 } else { 0.0 };

    let mut frames = Vec::new();
    let subjects = if subject_parts.is_empty() {
        vec![FrameTerm::empty()]
    } else {
        subject_parts
    };
    let objects = if object_parts.is_empty() {
        vec![FrameTerm::empty()]
    } else {
        object_parts
    };
    for subject in subjects {
        for object in objects.clone() {
            if subject.is_empty() && object.is_empty() {
                continue;
            }
            let mut flags = Vec::new();
            if subject_parts_len(&tree) > 1 {
                flags.push("coordinated_subject".to_string());
            }
            if tree
                .object
                .as_ref()
                .map(|phrase| phrase_parts(tree, phrase).len() > 1)
                .unwrap_or(false)
            {
                flags.push("coordinated_object".to_string());
            }
            if tree.subject.is_none() {
                flags.push("subject_omitted".to_string());
            }
            if tree.object.is_none() {
                flags.push("object_omitted".to_string());
            }
            if tree.surface_subject.is_some() {
                flags.push("passive_surface".to_string());
            }
            for warning in &tree.warnings {
                if warning.contains("non_projective") {
                    flags.push("non_projective_fallback".to_string());
                }
            }
            flags.sort();
            flags.dedup();

            let construction = if flags.iter().any(|flag| flag == "coordinated_subject")
                || flags.iter().any(|flag| flag == "coordinated_object")
            {
                Construction::Coordinated
            } else if flags.iter().any(|flag| flag == "non_projective_fallback") {
                Construction::NonProjective
            } else {
                tree.construction
            };

            let id = frame_id(tree, &subject, &predicate, &object, polarity, construction);
            frames.push(CanonicalFrame {
                id,
                subject: subject.clone(),
                predicate: predicate.clone(),
                object,
                polarity,
                construction,
                flags,
                confidence: base_confidence.max(0.25).min(0.99),
                evidence: FrameEvidence {
                    sentence: tree.sentence.clone(),
                    root_token: root_index,
                    subject_span: tree
                        .subject
                        .as_ref()
                        .map(|phrase| (phrase.start, phrase.end)),
                    predicate_span: Some((root_token.start, root_token.end)),
                    object_span: tree
                        .object
                        .as_ref()
                        .map(|phrase| (phrase.start, phrase.end)),
                    edges: (0..tree.edges.len()).collect(),
                },
            });
        }
    }
    frames
}

fn subject_parts_len(tree: &DependencyTree) -> usize {
    tree.subject
        .as_ref()
        .map(|phrase| phrase_parts(tree, phrase).len())
        .unwrap_or(0)
}

fn phrase_parts(tree: &DependencyTree, phrase: &Phrase) -> Vec<FrameTerm> {
    let start = tree
        .tokens
        .iter()
        .position(|token| token.start == phrase.start)
        .unwrap_or(phrase.head_index);
    let end = tree
        .tokens
        .iter()
        .position(|token| token.end == phrase.end)
        .map(|index| index + 1)
        .unwrap_or(phrase.head_index + 1);
    let mut parts = Vec::new();
    let mut part_start = start;
    for index in start..end {
        if tree.tokens[index].kind == TokenKind::Conjunction {
            if part_start < index {
                if let Some(part) = phrase_from_span(&tree.tokens, part_start, index) {
                    parts.push(FrameTerm::from_phrase(&part, &tree.tokens));
                }
            }
            part_start = index + 1;
        }
    }
    if part_start < end {
        if let Some(part) = phrase_from_span(&tree.tokens, part_start, end) {
            parts.push(FrameTerm::from_phrase(&part, &tree.tokens));
        }
    }
    if parts.is_empty() {
        parts.push(FrameTerm::from_phrase(phrase, &tree.tokens));
    }
    parts
}

fn frame_id(
    tree: &DependencyTree,
    subject: &FrameTerm,
    predicate: &FrameTerm,
    object: &FrameTerm,
    polarity: FramePolarity,
    construction: Construction,
) -> String {
    let material = format!(
        "{}|{}|{}|{}|{}|{}|{}",
        tree.sentence,
        subject.text,
        predicate.text,
        object.text,
        polarity.code(),
        construction.code(),
        tree.root.unwrap_or(0)
    );
    format!("dsi:{}", fnv1a_hex(material.as_bytes()))
}

/// 索引中的一条脊边。`canonical` 为真表示来自规范框架，否则来自表层树。
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SpineEdge {
    pub frame: usize,
    pub head: String,
    pub head_aliases: Vec<String>,
    pub head_pos: String,
    pub label: DepLabel,
    pub dep: String,
    pub dep_aliases: Vec<String>,
    pub dep_pos: String,
    pub canonical: bool,
    pub confidence: f64,
}

impl SpineEdge {
    pub fn head_keys(&self) -> Vec<&str> {
        let mut keys = vec![self.head.as_str()];
        keys.extend(self.head_aliases.iter().map(String::as_str));
        keys.sort();
        keys.dedup();
        keys
    }

    pub fn dep_keys(&self) -> Vec<&str> {
        let mut keys = vec![self.dep.as_str()];
        keys.extend(self.dep_aliases.iter().map(String::as_str));
        keys.sort();
        keys.dedup();
        keys
    }

    pub fn render(&self) -> String {
        format!(
            "[{}] {} --{}--> {}",
            if self.canonical { "规范" } else { "表层" },
            self.head,
            self.label.code(),
            self.dep
        )
    }
}

/// 查询模式中的词项。
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TermPattern {
    Any,
    Word(String),
    Pos(String),
    Var(String),
}

impl TermPattern {
    pub fn var(name: &str) -> Self {
        TermPattern::Var(name.to_string())
    }

    pub fn word(text: &str) -> Self {
        TermPattern::Word(text.to_string())
    }

    pub fn pos(pos: &str) -> Self {
        TermPattern::Pos(pos.to_string())
    }

    pub fn is_const(&self) -> bool {
        matches!(self, TermPattern::Word(_) | TermPattern::Pos(_))
    }

    pub fn is_any(&self) -> bool {
        matches!(self, TermPattern::Any)
    }
}

/// 一条脊边查询模式。
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EdgePattern {
    pub head: TermPattern,
    pub label: DepLabel,
    pub dep: TermPattern,
}

impl EdgePattern {
    pub fn new(head: TermPattern, label: DepLabel, dep: TermPattern) -> Self {
        Self { head, label, dep }
    }

    pub fn render(&self) -> String {
        format!(
            "{} --{}--> {}",
            render_term(&self.head),
            self.label.code(),
            render_term(&self.dep)
        )
    }
}

fn render_term(term: &TermPattern) -> String {
    match term {
        TermPattern::Any => "_".to_string(),
        TermPattern::Word(word) => word.clone(),
        TermPattern::Pos(pos) => format!("pos:{}", pos),
        TermPattern::Var(name) => format!("?{}", name),
    }
}

/// 一次查询。所有边模式按合取处理；`same_frame` 为真时要求落在同一规范框架。
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SpineQuery {
    pub edges: Vec<EdgePattern>,
    pub same_frame: bool,
    pub max_results: usize,
    pub max_probes: Option<usize>,
    pub canonical_only: bool,
    /// 限定框架极性；`None` 表示肯定和否定都接受。
    pub polarity: Option<FramePolarity>,
}

impl SpineQuery {
    pub fn new(edges: Vec<EdgePattern>) -> Self {
        Self {
            edges,
            same_frame: true,
            max_results: 32,
            max_probes: None,
            canonical_only: true,
            polarity: None,
        }
    }

    pub fn with_polarity(mut self, polarity: FramePolarity) -> Self {
        self.polarity = Some(polarity);
        self
    }

    pub fn with_max_results(mut self, max_results: usize) -> Self {
        self.max_results = max_results.max(1);
        self
    }

    pub fn with_budget(mut self, max_probes: usize) -> Self {
        self.max_probes = Some(max_probes.max(1));
        self
    }

    pub fn allowing_surface(mut self) -> Self {
        self.canonical_only = false;
        self
    }
}

/// 一个查询解。
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SpineSolution {
    pub bindings: BTreeMap<String, String>,
    pub frames: Vec<usize>,
    pub edges: Vec<usize>,
}

impl SpineSolution {
    pub fn render(&self) -> String {
        let bindings = self
            .bindings
            .iter()
            .map(|(key, value)| format!("{}={}", key, value))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "框架 {:?}，边 {:?}，绑定 {{{}}}",
            self.frames, self.edges, bindings
        )
    }
}

/// 查询计划与执行回执。所有计数均为确定性操作，不含挂钟。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct SpineReceipt {
    pub frames: usize,
    pub edges: usize,
    pub patterns: usize,
    pub planner_order: Vec<usize>,
    pub candidate_sizes: Vec<usize>,
    pub probes: usize,
    pub unifications: usize,
    pub budget: Option<usize>,
    pub budget_exhausted: bool,
    pub truncated: bool,
    pub returned: usize,
    pub certified: bool,
    pub notes: Vec<String>,
}

impl SpineReceipt {
    pub fn render_text(&self) -> String {
        let mut text = format!(
            "DSI 回执：框架 {}，边 {}，模式 {}，探测 {}，合一 {}，返回 {}；{}。",
            self.frames,
            self.edges,
            self.patterns,
            self.probes,
            self.unifications,
            self.returned,
            if self.certified {
                "结果完整"
            } else {
                "结果未证明完整"
            }
        );
        text.push_str(&format!(
            "\n  计划顺序：{:?}；候选规模：{:?}",
            self.planner_order, self.candidate_sizes
        ));
        if self.budget_exhausted {
            text.push_str("\n  - 探测预算耗尽，已返回当前最优候选");
        }
        if self.truncated {
            text.push_str("\n  - 结果数达到上限，未继续枚举");
        }
        for note in &self.notes {
            text.push_str("\n  - ");
            text.push_str(note);
        }
        text
    }
}

/// 查询结果。
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SpineSearch {
    pub solutions: Vec<SpineSolution>,
    pub receipt: SpineReceipt,
}

/// 问句中的一个待填槽位。
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct QuestionSlot {
    pub role: String,
    pub variable: String,
}

/// 由问句编译出的检索计划。
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct QuestionPlan {
    pub question: String,
    pub query: SpineQuery,
    pub slots: Vec<QuestionSlot>,
    pub construction: Construction,
}

impl QuestionPlan {
    pub fn render_text(&self) -> String {
        let slots = self
            .slots
            .iter()
            .map(|slot| format!("{}={}?", slot.role, slot.variable))
            .collect::<Vec<_>>()
            .join("，");
        format!(
            "问句「{}」→ {}（{}）",
            self.question,
            self.query
                .edges
                .iter()
                .map(EdgePattern::render)
                .collect::<Vec<_>>()
                .join(" ∧ "),
            slots
        )
    }
}

/// 问句查询结果。
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct QuestionSearch {
    pub plan: QuestionPlan,
    pub search: SpineSearch,
}

impl QuestionSearch {
    /// 按槽位返回去重后的候选答案；顺序稳定，便于后续裁决。
    pub fn answers(&self) -> Vec<(String, String)> {
        let mut answers = BTreeSet::new();
        for solution in &self.search.solutions {
            for slot in &self.plan.slots {
                if let Some(value) = solution.bindings.get(&slot.variable) {
                    answers.insert((slot.role.clone(), value.clone()));
                }
            }
        }
        answers.into_iter().collect()
    }

    pub fn render_text(&self) -> String {
        let mut text = self.plan.render_text();
        text.push('\n');
        for (role, value) in self.answers() {
            text.push_str(&format!("  {} = {}\n", role, value));
        }
        text.push_str(&self.search.receipt.render_text());
        text
    }
}

/// 依存脊索引。
#[derive(Debug, Clone, Default)]
pub struct SpineIndex {
    frames: Vec<CanonicalFrame>,
    frame_ids: HashMap<String, usize>,
    edges: Vec<SpineEdge>,
    frame_edges: Vec<Vec<usize>>,
    by_exact: HashMap<(String, DepLabel, String), Vec<usize>>,
    by_head: HashMap<(String, DepLabel), Vec<usize>>,
    by_dep: HashMap<(DepLabel, String), Vec<usize>>,
    by_pos: HashMap<(String, DepLabel, String), Vec<usize>>,
    by_label: HashMap<DepLabel, Vec<usize>>,
    sealed: bool,
}

impl SpineIndex {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_frames(frames: Vec<CanonicalFrame>) -> Self {
        let mut index = Self::new();
        for frame in frames {
            index.insert_frame(frame);
        }
        index.seal();
        index
    }

    /// 从文本一次完成分析、归一与建索引。
    pub fn from_text(text: &str) -> Self {
        Self::from_frames(extract_frames(text))
    }

    /// 从无源句三元组建索引：合成规范框架，保持与文本框架同一查询路径。
    pub fn from_triples(triples: &[(String, String, String)]) -> Self {
        let mut index = Self::new();
        for (subject, predicate, object) in triples {
            index.insert_frame(CanonicalFrame::from_triple(subject, predicate, object));
        }
        index.seal();
        index
    }

    pub fn insert_triple(&mut self, subject: &str, predicate: &str, object: &str) {
        self.insert_frame(CanonicalFrame::from_triple(subject, predicate, object));
    }

    /// 插入框架；完全相同的框架 id 只保留第一次插入的证据。
    pub fn insert_frame(&mut self, frame: CanonicalFrame) -> usize {
        if let Some(frame_id) = self.frame_ids.get(&frame.id) {
            return *frame_id;
        }
        self.sealed = false;
        let frame_id = self.frames.len();
        self.frame_ids.insert(frame.id.clone(), frame_id);
        self.frames.push(frame);
        self.frame_edges.push(Vec::new());
        let frame = &self.frames[frame_id];
        let mut edges = Vec::new();

        if !frame.subject.is_empty() {
            edges.push(canonical_edge(
                frame_id,
                &frame.predicate,
                DepLabel::NSubj,
                &frame.subject,
                frame.confidence,
            ));
        }
        if !frame.object.is_empty() {
            edges.push(canonical_edge(
                frame_id,
                &frame.predicate,
                DepLabel::Obj,
                &frame.object,
                frame.confidence,
            ));
        }
        if frame.polarity == FramePolarity::Negative {
            edges.push(SpineEdge {
                frame: frame_id,
                head: frame.predicate.text.clone(),
                head_aliases: Vec::new(),
                head_pos: frame.predicate.pos.clone(),
                label: DepLabel::Neg,
                dep: "否定".to_string(),
                dep_aliases: vec!["不".to_string(), "没".to_string(), "未".to_string()],
                dep_pos: "neg".to_string(),
                canonical: true,
                confidence: frame.confidence,
            });
        }
        for edge in edges {
            let edge_id = self.edges.len();
            self.edges.push(edge);
            self.frame_edges[frame_id].push(edge_id);
        }
        frame_id
    }

    /// 索引一棵表层依存树。重复调用不会删除既有框架。
    pub fn insert_tree(&mut self, tree: &DependencyTree) {
        let Some(root) = tree.root else {
            return;
        };
        let frames = compile_frames(tree);
        if frames.is_empty() {
            return;
        }
        let frame_id = self.insert_frame(frames[0].clone());
        for edge in &tree.edges {
            let Some(head) = tree.token(edge.head) else {
                continue;
            };
            let Some(dep) = tree.token(edge.dep) else {
                continue;
            };
            let edge_id = self.edges.len();
            self.edges.push(SpineEdge {
                frame: frame_id,
                head: head.text.clone(),
                head_aliases: phrase_aliases(tree, edge.head),
                head_pos: head.pos.clone(),
                label: edge.label,
                dep: dep.text.clone(),
                dep_aliases: phrase_aliases(tree, edge.dep),
                dep_pos: dep.pos.clone(),
                canonical: false,
                confidence: edge.confidence,
            });
            self.frame_edges[frame_id].push(edge_id);
        }
        let _ = root;
    }

    pub fn seal(&mut self) {
        if self.sealed {
            return;
        }
        self.by_exact.clear();
        self.by_head.clear();
        self.by_dep.clear();
        self.by_pos.clear();
        self.by_label.clear();
        for (edge_id, edge) in self.edges.iter().enumerate() {
            for head in edge.head_keys() {
                self.by_head
                    .entry((head.to_string(), edge.label))
                    .or_default()
                    .push(edge_id);
                for dep in edge.dep_keys() {
                    self.by_exact
                        .entry((head.to_string(), edge.label, dep.to_string()))
                        .or_default()
                        .push(edge_id);
                }
            }
            for dep in edge.dep_keys() {
                self.by_dep
                    .entry((edge.label, dep.to_string()))
                    .or_default()
                    .push(edge_id);
            }
            self.by_pos
                .entry((edge.head_pos.clone(), edge.label, edge.dep_pos.clone()))
                .or_default()
                .push(edge_id);
            self.by_label.entry(edge.label).or_default().push(edge_id);
        }
        for postings in self
            .by_exact
            .values_mut()
            .chain(self.by_head.values_mut())
            .chain(self.by_dep.values_mut())
            .chain(self.by_pos.values_mut())
            .chain(self.by_label.values_mut())
        {
            postings.sort_unstable();
            postings.dedup();
        }
        self.sealed = true;
    }

    pub fn is_sealed(&self) -> bool {
        self.sealed
    }

    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn frame(&self, index: usize) -> Option<&CanonicalFrame> {
        self.frames.get(index)
    }

    pub fn edge(&self, index: usize) -> Option<&SpineEdge> {
        self.edges.get(index)
    }

    pub fn frame_edges(&self, frame: usize) -> &[usize] {
        self.frame_edges
            .get(frame)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    pub fn frames(&self) -> &[CanonicalFrame] {
        &self.frames
    }

    pub fn edges(&self) -> &[SpineEdge] {
        &self.edges
    }

    /// 按主语/谓词/宾语模式做框架级查询的便捷入口。
    pub fn lookup(
        &mut self,
        subject: TermPattern,
        predicate: TermPattern,
        object: TermPattern,
        max_probes: Option<usize>,
    ) -> SpineSearch {
        let edges = vec![
            EdgePattern::new(predicate.clone(), DepLabel::NSubj, subject),
            EdgePattern::new(predicate, DepLabel::Obj, object),
        ];
        let mut query = SpineQuery::new(edges);
        query.same_frame = true;
        query.max_probes = max_probes.map(|budget| budget.max(1));
        self.query(&query)
    }

    /// 把自然语言问句编译成脊边查询：疑问词变成变量，其余词项保持常量。
    ///
    /// 支持 `谁/什么/哪...` 出现在主语、宾语或被动施事位置的问句；
    /// 是非问与无法定位疑问角色的句子返回 `None`，交由其他通道处理。
    pub fn plan_question(&self, question: &str) -> Option<QuestionPlan> {
        let tree = analyze(question);
        let root = tree.root?;
        let root_token = tree.token(root)?;
        if root_token.kind == TokenKind::Wh {
            return None;
        }
        let subject_is_wh = tree
            .subject
            .as_ref()
            .map(|phrase| phrase_contains_wh(&tree, phrase))
            .unwrap_or(false);
        let object_is_wh = tree
            .object
            .as_ref()
            .map(|phrase| phrase_contains_wh(&tree, phrase))
            .unwrap_or(false);
        if !subject_is_wh && !object_is_wh {
            return None;
        }

        let predicate = TermPattern::word(&root_token.text);
        let subject = match (&tree.subject, subject_is_wh) {
            (Some(_), true) => Some(TermPattern::var("subject")),
            (Some(phrase), false) => Some(TermPattern::word(&phrase.text)),
            (None, _) => None,
        };
        let object = match (&tree.object, object_is_wh) {
            (Some(_), true) => Some(TermPattern::var("object")),
            (Some(phrase), false) => Some(TermPattern::word(&phrase.text)),
            (None, _) => None,
        };
        let mut edges = Vec::new();
        if let Some(subject) = subject {
            edges.push(EdgePattern::new(
                predicate.clone(),
                DepLabel::NSubj,
                subject,
            ));
        }
        if let Some(object) = object {
            edges.push(EdgePattern::new(predicate, DepLabel::Obj, object));
        }
        if edges.is_empty() {
            return None;
        }
        let mut slots = Vec::new();
        if subject_is_wh {
            slots.push(QuestionSlot {
                role: "subject".to_string(),
                variable: "subject".to_string(),
            });
        }
        if object_is_wh {
            slots.push(QuestionSlot {
                role: "object".to_string(),
                variable: "object".to_string(),
            });
        }
        Some(QuestionPlan {
            question: question.to_string(),
            query: SpineQuery::new(edges).with_polarity(tree_polarity(&tree)),
            slots,
            construction: tree.construction,
        })
    }

    /// 编译并执行问句；返回计划与可审计检索结果。
    pub fn answer_question(
        &mut self,
        question: &str,
        max_probes: Option<usize>,
    ) -> Option<QuestionSearch> {
        let mut plan = self.plan_question(question)?;
        plan.query.max_probes = max_probes.map(|budget| budget.max(1));
        let search = self.query(&plan.query);
        Some(QuestionSearch { plan, search })
    }

    /// 执行带变量、多跳与预算的合取查询。
    pub fn query(&mut self, query: &SpineQuery) -> SpineSearch {
        self.seal();
        let mut receipt = SpineReceipt {
            frames: self.frames.len(),
            edges: self.edges.len(),
            patterns: query.edges.len(),
            budget: query.max_probes,
            notes: Vec::new(),
            ..SpineReceipt::default()
        };
        if query.edges.is_empty() {
            receipt.certified = false;
            receipt.notes.push("查询没有边模式".to_string());
            return SpineSearch {
                solutions: Vec::new(),
                receipt,
            };
        }

        let mut plans: Vec<(usize, usize)> = query
            .edges
            .iter()
            .enumerate()
            .map(|(index, pattern)| {
                let candidates = self.candidates(pattern, &BTreeMap::new());
                (index, candidates.len())
            })
            .collect();
        plans.sort_by_key(|(index, size)| (*size, *index));
        receipt.planner_order = plans.iter().map(|(index, _)| *index).collect();
        receipt.candidate_sizes = plans.iter().map(|(_, size)| *size).collect();

        let mut solutions = Vec::new();
        let mut used = 0usize;
        let planner_order = receipt.planner_order.clone();
        self.search_recursive(
            query,
            &planner_order,
            0,
            BTreeMap::new(),
            Vec::new(),
            Vec::new(),
            &mut used,
            &mut solutions,
            &mut receipt,
        );

        solutions.sort_by(|left, right| {
            left.frames
                .cmp(&right.frames)
                .then_with(|| left.edges.cmp(&right.edges))
                .then_with(|| left.bindings.cmp(&right.bindings))
        });
        solutions.dedup();
        if solutions.len() > query.max_results {
            solutions.truncate(query.max_results);
            receipt.truncated = true;
        }
        receipt.returned = solutions.len();
        receipt.certified = !receipt.truncated && !receipt.budget_exhausted;
        if !receipt.certified {
            receipt
                .notes
                .push("未证明完整枚举：结果受预算或上限约束".to_string());
        }
        SpineSearch { solutions, receipt }
    }

    #[allow(clippy::too_many_arguments)]
    fn search_recursive(
        &self,
        query: &SpineQuery,
        order: &[usize],
        depth: usize,
        bindings: BTreeMap<String, String>,
        frame_ids: Vec<usize>,
        edge_ids: Vec<usize>,
        used: &mut usize,
        solutions: &mut Vec<SpineSolution>,
        receipt: &mut SpineReceipt,
    ) {
        if depth == order.len() {
            let mut frames = frame_ids;
            frames.sort_unstable();
            frames.dedup();
            let mut edges = edge_ids;
            edges.sort_unstable();
            edges.dedup();
            let solution = SpineSolution {
                bindings,
                frames,
                edges,
            };
            if !solutions.contains(&solution) {
                solutions.push(solution);
            }
            return;
        }
        if let Some(budget) = query.max_probes {
            if *used >= budget {
                receipt.budget_exhausted = true;
                return;
            }
        }
        let pattern_index = order[depth];
        let pattern = &query.edges[pattern_index];
        let candidates = self.candidates(pattern, &bindings);
        for edge_id in candidates {
            if let Some(budget) = query.max_probes {
                if *used >= budget {
                    receipt.budget_exhausted = true;
                    return;
                }
            }
            *used += 1;
            receipt.probes += 1;
            let Some(edge) = self.edges.get(edge_id) else {
                continue;
            };
            if query.canonical_only && !edge.canonical {
                continue;
            }
            if let Some(polarity) = query.polarity {
                if self.frames.get(edge.frame).map(|frame| frame.polarity) != Some(polarity) {
                    continue;
                }
            }
            if query.same_frame && !frame_ids.is_empty() && !frame_ids.contains(&edge.frame) {
                continue;
            }
            let Some(next_bindings) = unify_edge(pattern, edge, &bindings) else {
                continue;
            };
            receipt.unifications += 1;
            let mut next_frames = frame_ids.clone();
            next_frames.push(edge.frame);
            let mut next_edges = edge_ids.clone();
            next_edges.push(edge_id);
            self.search_recursive(
                query,
                order,
                depth + 1,
                next_bindings,
                next_frames,
                next_edges,
                used,
                solutions,
                receipt,
            );
        }
    }

    fn candidates(&self, pattern: &EdgePattern, bindings: &BTreeMap<String, String>) -> Vec<usize> {
        let head = resolve_pattern_term(&pattern.head, bindings);
        let dep = resolve_pattern_term(&pattern.dep, bindings);
        match (&head, &dep) {
            (ProbeTerm::Word(head), ProbeTerm::Word(dep)) => self
                .by_exact
                .get(&(head.clone(), pattern.label, dep.clone()))
                .cloned()
                .unwrap_or_default(),
            (ProbeTerm::Word(head), _) => self
                .by_head
                .get(&(head.clone(), pattern.label))
                .cloned()
                .unwrap_or_default(),
            (_, ProbeTerm::Word(dep)) => self
                .by_dep
                .get(&(pattern.label, dep.clone()))
                .cloned()
                .unwrap_or_default(),
            (ProbeTerm::Pos(head), ProbeTerm::Pos(dep)) if !head.is_empty() && !dep.is_empty() => {
                self.by_pos
                    .get(&(head.clone(), pattern.label, dep.clone()))
                    .cloned()
                    .unwrap_or_default()
            }
            _ => self
                .by_label
                .get(&pattern.label)
                .cloned()
                .unwrap_or_default(),
        }
    }
}

fn phrase_contains_wh(tree: &DependencyTree, phrase: &Phrase) -> bool {
    tree.tokens.iter().any(|token| {
        token.kind == TokenKind::Wh && token.start >= phrase.start && token.end <= phrase.end
    })
}

fn phrase_aliases(tree: &DependencyTree, token_index: usize) -> Vec<String> {
    for phrase in [
        tree.subject.as_ref(),
        tree.object.as_ref(),
        tree.surface_subject.as_ref(),
    ]
    .into_iter()
    .flatten()
    {
        if phrase.head_index == token_index {
            let mut aliases = vec![phrase.text.clone()];
            if phrase.head != phrase.text {
                aliases.push(phrase.head.clone());
            }
            aliases.sort();
            aliases.dedup();
            return aliases;
        }
    }
    Vec::new()
}

fn canonical_edge(
    frame: usize,
    head: &FrameTerm,
    label: DepLabel,
    dep: &FrameTerm,
    confidence: f64,
) -> SpineEdge {
    let mut head_aliases: Vec<String> = head.keys().into_iter().map(str::to_string).collect();
    let mut dep_aliases: Vec<String> = dep.keys().into_iter().map(str::to_string).collect();
    head_aliases.sort();
    head_aliases.dedup();
    dep_aliases.sort();
    dep_aliases.dedup();
    SpineEdge {
        frame,
        head: head.text.clone(),
        head_aliases,
        head_pos: head.pos.clone(),
        label,
        dep: dep.text.clone(),
        dep_aliases,
        dep_pos: dep.pos.clone(),
        canonical: true,
        confidence,
    }
}

#[derive(Debug, Clone)]
enum ProbeTerm {
    Word(String),
    Pos(String),
    Any,
}

fn resolve_pattern_term(term: &TermPattern, bindings: &BTreeMap<String, String>) -> ProbeTerm {
    match term {
        TermPattern::Any => ProbeTerm::Any,
        TermPattern::Word(word) => ProbeTerm::Word(word.clone()),
        TermPattern::Pos(pos) => ProbeTerm::Pos(pos.clone()),
        TermPattern::Var(name) => match bindings.get(name) {
            Some(value) => ProbeTerm::Word(value.clone()),
            None => ProbeTerm::Any,
        },
    }
}

fn unify_edge(
    pattern: &EdgePattern,
    edge: &SpineEdge,
    bindings: &BTreeMap<String, String>,
) -> Option<BTreeMap<String, String>> {
    let mut next = bindings.clone();
    bind_term(
        &pattern.head,
        &edge.head,
        &edge.head_aliases,
        &edge.head_pos,
        &mut next,
    )?;
    bind_term(
        &pattern.dep,
        &edge.dep,
        &edge.dep_aliases,
        &edge.dep_pos,
        &mut next,
    )?;
    Some(next)
}

fn bind_term(
    pattern: &TermPattern,
    text: &str,
    aliases: &[String],
    pos: &str,
    bindings: &mut BTreeMap<String, String>,
) -> Option<()> {
    match pattern {
        TermPattern::Any => Some(()),
        TermPattern::Word(word) => {
            if text == word || aliases.iter().any(|alias| alias == word) {
                Some(())
            } else {
                None
            }
        }
        TermPattern::Pos(expected) => {
            if pos == expected {
                Some(())
            } else {
                None
            }
        }
        TermPattern::Var(name) => match bindings.get(name) {
            Some(bound) => {
                if text == bound || aliases.iter().any(|alias| alias == bound) {
                    Some(())
                } else {
                    None
                }
            }
            None => {
                bindings.insert(name.clone(), text.to_string());
                Some(())
            }
        },
    }
}

impl SpineIndex {
    /// 索引统计摘要（确定性排序，便于测试与审计）。
    pub fn render_report(&self) -> String {
        let mut labels: BTreeMap<&'static str, usize> = BTreeMap::new();
        for edge in &self.edges {
            *labels.entry(edge.label.code()).or_insert(0) += 1;
        }
        let mut text = format!(
            "DSI 索引：框架 {}，边 {}，{}",
            self.frames.len(),
            self.edges.len(),
            if self.sealed {
                "已封存"
            } else {
                "未封存"
            }
        );
        text.push_str("\n  标签分布：");
        for (label, count) in labels {
            text.push_str(&format!(" {}={}", label, count));
        }
        text.push('\n');
        for (frame_id, frame) in self.frames.iter().enumerate() {
            text.push_str(&format!("  #{} {}\n", frame_id, frame.render_text()));
        }
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(text: &str) -> CanonicalFrame {
        let tree = analyze(text);
        let frames = compile_frames(&tree);
        assert_eq!(frames.len(), 1, "{} => {:?}", text, frames);
        frames.into_iter().next().unwrap()
    }

    #[test]
    fn classify_token_uses_ontology_closure_for_residual_tags() {
        assert_eq!(classify_token("简称", "j"), TokenKind::Noun);
        assert_eq!(classify_token("某地", "s"), TokenKind::Noun);
        assert_eq!(classify_token("名词化", "vn"), TokenKind::Verb);
    }

    #[test]
    fn active_svo_compiles_to_canonical_frame() {
        let parsed = frame("张三批评了李四");
        assert_eq!(parsed.subject.text, "张三");
        assert_eq!(parsed.predicate.text, "批评");
        assert_eq!(parsed.object.text, "李四");
        assert_eq!(parsed.polarity, FramePolarity::Affirmative);
        assert_eq!(parsed.construction, Construction::Active);
    }

    #[test]
    fn passive_and_ba_normalize_to_the_same_frame() {
        let active = frame("张三批评了李四");
        let passive = frame("李四被张三批评了");
        let ba = frame("张三把李四批评了");
        assert_eq!(active.subject.text, passive.subject.text);
        assert_eq!(active.object.text, passive.object.text);
        assert_eq!(passive.construction, Construction::Passive);
        assert_eq!(ba.construction, Construction::Ba);
        assert_eq!(active.subject.head, ba.subject.head);
        assert_eq!(active.object.head, ba.object.head);
    }

    #[test]
    fn copula_and_possessive_keep_phrase_evidence() {
        let parsed = frame("张三是李四的老师");
        assert_eq!(parsed.subject.text, "张三");
        assert_eq!(parsed.predicate.text, "是");
        assert_eq!(parsed.object.text, "李四的老师");
        assert!(parsed.object.head == "老师");
        assert!(parsed
            .flags
            .iter()
            .all(|flag| !flag.contains("non_projective")));
    }

    #[test]
    fn negation_is_recorded_not_erased() {
        let parsed = frame("张三不认识李四");
        assert_eq!(parsed.polarity, FramePolarity::Negative);
        assert_eq!(parsed.subject.text, "张三");
        assert_eq!(parsed.object.text, "李四");
    }

    #[test]
    fn coordination_expands_into_multiple_frames() {
        let tree = analyze("张三和李四是老师");
        let frames = compile_frames(&tree);
        assert_eq!(frames.len(), 2, "{:?}", frames);
        let subjects: BTreeSet<String> = frames
            .iter()
            .map(|frame| frame.subject.text.clone())
            .collect();
        assert!(subjects.contains("张三"));
        assert!(subjects.contains("李四"));
        assert!(frames
            .iter()
            .all(|frame| frame.construction == Construction::Coordinated));
    }

    #[test]
    fn relative_clause_subject_is_kept_as_phrase() {
        let parsed = frame("我认识的那个人是张三");
        assert_eq!(parsed.predicate.text, "是");
        assert_eq!(parsed.subject.text, "我认识的那个人");
        assert_eq!(parsed.object.text, "张三");
        let tree = analyze("我认识的那个人是张三");
        assert!(tree.edges.iter().any(|edge| edge.label == DepLabel::Rel));
    }

    #[test]
    fn index_matches_passive_and_active_with_same_query() {
        let mut index = SpineIndex::from_text("张三批评了李四。李四被张三批评了。");
        assert_eq!(index.frame_count(), 2);
        let search = index.lookup(
            TermPattern::word("张三"),
            TermPattern::Any,
            TermPattern::word("李四"),
            None,
        );
        assert_eq!(search.solutions.len(), 2, "{:?}", search.receipt);
        let frames: BTreeSet<usize> = search
            .solutions
            .iter()
            .flat_map(|solution| solution.frames.iter().copied())
            .collect();
        assert_eq!(frames, BTreeSet::from([0, 1]));
        assert!(search.receipt.certified);
    }

    #[test]
    fn multi_hop_join_binds_shared_variable() {
        let mut index = SpineIndex::from_text("张三批评了李四。李四写了书。");
        let search = index.query(&SpineQuery {
            edges: vec![
                EdgePattern::new(
                    TermPattern::var("event1"),
                    DepLabel::NSubj,
                    TermPattern::var("person"),
                ),
                EdgePattern::new(
                    TermPattern::var("event2"),
                    DepLabel::NSubj,
                    TermPattern::var("person"),
                ),
            ],
            same_frame: false,
            max_results: 16,
            max_probes: None,
            canonical_only: true,
            polarity: None,
        });
        assert!(
            search
                .solutions
                .iter()
                .any(|solution| solution.bindings.get("person") == Some(&"李四".to_string())),
            "{:?}",
            search.receipt
        );
        assert!(search.receipt.certified);
    }

    #[test]
    fn duplicate_frame_is_kept_once() {
        let mut index = SpineIndex::new();
        index.insert_triple("张三", "批评", "李四");
        index.insert_triple("张三", "批评", "李四");
        assert_eq!(index.frame_count(), 1);
        assert_eq!(index.edge_count(), 2);
        let search = index.lookup(
            TermPattern::word("张三"),
            TermPattern::word("批评"),
            TermPattern::word("李四"),
            None,
        );
        assert_eq!(search.solutions.len(), 1, "{:?}", search.receipt);
    }

    #[test]
    fn adjective_predicate_keeps_its_subject() {
        let frames = extract_frames("苹果红了。");
        assert_eq!(frames.len(), 1, "{:?}", frames);
        assert_eq!(frames[0].subject.text, "苹果");
        assert_eq!(frames[0].predicate.text, "红");
        assert!(frames[0].object.is_empty());
    }

    #[test]
    fn intransitive_question_uses_the_available_spine_edge() {
        let mut index = SpineIndex::from_text("李四跑了。王五走了。");
        let result = index
            .answer_question("谁跑了？", None)
            .expect("intransitive question should plan");
        assert_eq!(
            result.answers(),
            vec![("subject".to_string(), "李四".to_string())],
            "{:?}",
            result.search.receipt
        );
    }

    #[test]
    fn negative_question_does_not_mix_affirmative_frames() {
        let mut index = SpineIndex::from_text("张三批评了李四。王五没有批评李四。");
        let positive = index
            .answer_question("谁批评了李四？", None)
            .expect("affirmative question should plan");
        assert_eq!(
            positive.answers(),
            vec![("subject".to_string(), "张三".to_string())],
            "{:?}",
            positive.search.receipt
        );
        let negative = index
            .answer_question("谁没有批评李四？", None)
            .expect("negative question should plan");
        assert_eq!(
            negative.answers(),
            vec![("subject".to_string(), "王五".to_string())],
            "{:?}",
            negative.search.receipt
        );
    }

    #[test]
    fn exact_budget_completion_stays_certified() {
        let mut index = SpineIndex::from_text("张三批评了李四。");
        let search = index.query(
            &SpineQuery::new(vec![EdgePattern::new(
                TermPattern::Any,
                DepLabel::NSubj,
                TermPattern::Any,
            )])
            .with_budget(1),
        );
        assert_eq!(search.solutions.len(), 1, "{:?}", search.receipt);
        assert!(!search.receipt.budget_exhausted, "{:?}", search.receipt);
        assert!(search.receipt.certified, "{:?}", search.receipt);
    }

    #[test]
    fn budget_exhaustion_is_visible_in_receipt() {
        let mut index = SpineIndex::from_text("张三批评了李四。李四写了书。");
        let search = index.query(
            &SpineQuery::new(vec![EdgePattern::new(
                TermPattern::Any,
                DepLabel::NSubj,
                TermPattern::Any,
            )])
            .with_budget(1),
        );
        assert!(search.receipt.budget_exhausted);
        assert!(!search.receipt.certified);
    }

    #[test]
    fn surface_edges_are_optional() {
        let text = "李四被张三批评了";
        let tree = analyze(text);
        let mut index = SpineIndex::new();
        index.insert_tree(&tree);
        index.seal();
        let query = SpineQuery::new(vec![EdgePattern::new(
            TermPattern::Any,
            DepLabel::Agent,
            TermPattern::word("张三"),
        )])
        .allowing_surface();
        let search = index.query(&query);
        assert_eq!(search.solutions.len(), 1, "{:?}", search.receipt);
        let canonical = index.query(&SpineQuery::new(vec![EdgePattern::new(
            TermPattern::Any,
            DepLabel::Agent,
            TermPattern::word("张三"),
        )]));
        assert!(canonical.solutions.is_empty());
    }

    #[test]
    fn same_text_same_index_bytes() {
        let first = SpineIndex::from_text("张三批评了李四。");
        let second = SpineIndex::from_text("张三批评了李四。");
        assert_eq!(first.render_report(), second.render_report());
        assert_eq!(
            first.frames()[0].id,
            second.frames()[0].id,
            "同输入必须同 id"
        );
    }

    #[test]
    fn question_plan_turns_wh_into_bindable_variable() {
        let index = SpineIndex::new();
        let plan = index.plan_question("谁批评了李四？").unwrap();
        assert_eq!(plan.slots.len(), 1);
        assert_eq!(plan.slots[0].variable, "subject");
        assert_eq!(plan.query.edges.len(), 2);
        assert!(plan.query.edges.iter().any(|edge| {
            edge.label == DepLabel::NSubj && edge.dep == TermPattern::var("subject")
        }));
    }

    #[test]
    fn index_answers_who_and_what_questions() {
        let mut index = SpineIndex::from_text("张三批评了李四。李四写了书。");
        let who = index
            .answer_question("谁批评了李四？", None)
            .expect("who question");
        assert!(
            who.answers()
                .contains(&("subject".to_string(), "张三".to_string())),
            "{:?}",
            who
        );
        let what = index
            .answer_question("张三批评了谁？", None)
            .expect("what question");
        assert!(
            what.answers()
                .contains(&("object".to_string(), "李四".to_string())),
            "{:?}",
            what
        );
    }

    #[test]
    fn passive_question_finds_agent() {
        let mut index = SpineIndex::from_text("李四被张三批评了。");
        let answer = index
            .answer_question("李四被谁批评了？", None)
            .expect("passive who question");
        assert!(
            answer
                .answers()
                .contains(&("subject".to_string(), "张三".to_string())),
            "{:?}",
            answer
        );
    }

    #[test]
    fn comma_separated_clauses_still_compile_frames() {
        let frames = extract_frames("张三批评了李四，李四写了书。");
        assert_eq!(frames.len(), 2, "{:?}", frames);
        let predicates: BTreeSet<&str> = frames
            .iter()
            .map(|frame| frame.predicate.text.as_str())
            .collect();
        assert_eq!(predicates, BTreeSet::from(["批评", "写"]));
    }

    #[test]
    fn synthetic_frames_keep_queries_available_without_source_sentences() {
        let triple = ("张三".to_string(), "批评".to_string(), "李四".to_string());
        let mut index = SpineIndex::from_triples(&[triple]);
        let answer = index
            .answer_question("谁批评了李四？", None)
            .expect("synthetic frame should answer");
        assert!(
            answer
                .answers()
                .contains(&("subject".to_string(), "张三".to_string())),
            "{:?}",
            answer
        );
        assert_eq!(index.frames()[0].flags, vec!["synthetic_frame"]);
    }

    #[test]
    fn yes_no_question_has_no_spine_plan() {
        let index = SpineIndex::new();
        assert!(index.plan_question("张三批评了李四吗？").is_none());
    }

    #[test]
    fn empty_or_unparsed_input_does_not_panic() {
        let mut index = SpineIndex::from_text("");
        assert_eq!(index.frame_count(), 0);
        let search = index.query(&SpineQuery::new(vec![EdgePattern::new(
            TermPattern::Any,
            DepLabel::NSubj,
            TermPattern::Any,
        )]));
        assert!(search.solutions.is_empty());
        assert_eq!(analyze("。！？").edges.len(), 0);
    }
}

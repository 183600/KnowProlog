//! PRF：证明携带的关系帧（Proof-carrying Relation Frames）。
//!
//! 现有导入链路是「句子 → 三元组」的一次性映射；本模块在它旁边加一层
//! **只读**的中间表示，让抽取结果带上可检查、可复述的证据：
//!   - 帧：主语、谓词、宾语之外的极性、模态、时间范围、限定语、字符跨度；
//!   - 闸门：把帧与图谱事实、谓词画像、类型体系、声明律放在一起，
//!     做有界局部演绎闭包，给出直接支持 / 推理支持 / 冲突 / 无证据；
//!   - 叙述：把判定与证明路径线性化成中文，供诊断命令与后续生成器复用。
//!
//! 设计约束与仓库其他模块一致：纯符号、零训练、同输入同字节；闭包只围绕
//! 候选帧的端点展开，并用深度/边数上限防止组合爆炸；本模块不写库。
//!
//! 记号：`P(A, B)`；互逆谓词先按谓词画像归一，再做闭包。

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::OnceLock;

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::chinese_reasoning_nlu::{parse_lesson, LessonKind};
use crate::context_image::{canonical_json, fnv1a_hex};
use crate::grammar_extractor::GrammarExtractor;
use crate::law_book::{LawBook, LawKind};
use crate::predicate_registry::PredicateRegistry;
use crate::prolog_engine::PrologFact;
use crate::temporal::{self, TimeGranularity as SpanGranularity, TimePoint, TimeSpan, Validity};
use crate::type_system::TypeSystem;

/// 关系帧方言标识。
pub const FRAME_DIALECT: &str = "knowprolog-prf/1";
/// 默认局部闭包深度（传递链最多再走几跳）。
pub const DEFAULT_CLOSURE_DEPTH: usize = 4;
/// 默认局部闭包最多收集多少条边。
pub const DEFAULT_CLOSURE_LIMIT: usize = 2048;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Polarity {
    Affirmative,
    Negative,
}

impl Default for Polarity {
    fn default() -> Self {
        Polarity::Affirmative
    }
}

impl Polarity {
    pub fn label(self) -> &'static str {
        match self {
            Polarity::Affirmative => "肯定",
            Polarity::Negative => "否定",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Modality {
    Asserted,
    Possible,
    Necessary,
    Uncertain,
}

impl Default for Modality {
    fn default() -> Self {
        Modality::Asserted
    }
}

impl Modality {
    pub fn label(self) -> &'static str {
        match self {
            Modality::Asserted => "直陈",
            Modality::Possible => "可能",
            Modality::Necessary => "必然",
            Modality::Uncertain => "传闻/不确定",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TemporalGranularity {
    Year,
    Month,
    Day,
    Relative,
}

impl TemporalGranularity {
    pub fn label(self) -> &'static str {
        match self {
            TemporalGranularity::Year => "年",
            TemporalGranularity::Month => "月",
            TemporalGranularity::Day => "日",
            TemporalGranularity::Relative => "相对时间",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TemporalRelation {
    Point,
    Range,
    Before,
    After,
}

impl TemporalRelation {
    pub fn label(self) -> &'static str {
        match self {
            TemporalRelation::Point => "时间点",
            TemporalRelation::Range => "时间范围",
            TemporalRelation::Before => "之前",
            TemporalRelation::After => "之后",
        }
    }
}

/// 句子中的时间限定。数值只保留显式写出的年/月/日，不推断当前日期。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemporalScope {
    pub raw: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normalized: Option<String>,
    pub granularity: TemporalGranularity,
    pub relation: TemporalRelation,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start_year: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end_year: Option<i32>,
}

impl TemporalScope {
    pub fn render_text(&self) -> String {
        let mut text = self.raw.clone();
        if let Some(normalized) = &self.normalized {
            if normalized != &self.raw {
                text.push_str(&format!("（{}，{}）", self.relation.label(), normalized));
                return text;
            }
        }
        text.push_str(&format!("（{}）", self.relation.label()));
        text
    }

    fn span_granularity(&self) -> SpanGranularity {
        match self.granularity {
            TemporalGranularity::Year => SpanGranularity::Year,
            TemporalGranularity::Month => SpanGranularity::Month,
            TemporalGranularity::Day => SpanGranularity::Day,
            TemporalGranularity::Relative => SpanGranularity::Year,
        }
    }

    fn anchor_point(&self) -> Option<TimePoint> {
        self.normalized
            .as_deref()
            .and_then(temporal::parse_time_point)
            .or_else(|| self.start_year.map(TimePoint::year))
    }

    /// 把句内时间限定转换为可做区间运算的 [`TimeSpan`]。
    ///
    /// 相对时间（“去年/当时”）不猜系统时钟，返回 `None`，由
    /// [`crate::temporal::TemporalSession`] 依据会话参考时间解析。
    pub fn to_time_span(&self) -> Option<TimeSpan> {
        if self.granularity == TemporalGranularity::Relative {
            return None;
        }
        let point = self.anchor_point()?;
        let granularity = self.span_granularity();
        match self.relation {
            TemporalRelation::Point => Some(TimeSpan {
                start: Some(point.period_start(granularity)),
                end: Some(point.period_end(granularity)),
                granularity,
            }),
            TemporalRelation::Range => {
                let end_year = self.end_year.unwrap_or(point.year);
                TimeSpan::from_bounds(
                    Some(point.period_start(granularity)),
                    Some(TimePoint::new(end_year, 12, 31)),
                    granularity,
                )
            }
            TemporalRelation::Before => Some(TimeSpan::before(point, granularity)),
            TemporalRelation::After => Some(TimeSpan::after(point, granularity)),
        }
    }

    /// 时间限定 → 事实有效期；无法解析时是「未知」而不是「永远」。
    pub fn to_validity(&self) -> Validity {
        self.to_time_span()
            .map(Validity::Span)
            .unwrap_or(Validity::Unknown)
    }
}

/// 帧在源文本中的证据跨度（字节偏移）。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceSpan {
    pub sentence_index: usize,
    pub sentence: String,
    pub start: usize,
    pub end: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subject_span: Option<(usize, usize)>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub object_span: Option<(usize, usize)>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub polarity_span: Option<(usize, usize)>,
}

/// 一条关系帧：三元组 + 证据 + 限定。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RelationFrame {
    pub id: String,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub polarity: Polarity,
    pub modality: Modality,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temporal: Option<TemporalScope>,
    #[serde(default)]
    pub qualifiers: Vec<String>,
    pub evidence: EvidenceSpan,
    #[serde(default)]
    pub extraction: Vec<String>,
    pub confidence: f64,
}

impl RelationFrame {
    pub fn new(subject: &str, predicate: &str, object: &str) -> Self {
        let mut frame = Self {
            id: String::new(),
            subject: subject.trim().to_string(),
            predicate: predicate.trim().to_string(),
            object: object.trim().to_string(),
            polarity: Polarity::Affirmative,
            modality: Modality::Asserted,
            temporal: None,
            qualifiers: Vec::new(),
            evidence: EvidenceSpan::default(),
            extraction: vec!["手工构造".to_string()],
            confidence: 1.0,
        };
        frame.refresh_id();
        frame
    }

    pub fn with_polarity(mut self, polarity: Polarity) -> Self {
        self.polarity = polarity;
        self.refresh_id();
        self
    }

    pub fn with_temporal(mut self, temporal: TemporalScope) -> Self {
        self.temporal = Some(temporal);
        self.refresh_id();
        self
    }

    fn refresh_id(&mut self) {
        let material = format!(
            "{}|{}|{}|{}|{:?}|{:?}|{}",
            self.evidence.sentence_index,
            self.subject,
            self.predicate,
            self.object,
            self.polarity,
            self.modality,
            self.evidence.start
        );
        self.id = format!("frame:{}", fnv1a_hex(material.as_bytes()));
    }

    pub fn render_text(&self) -> String {
        let mut output = format!(
            "{} -[{}]-> {}（{}；{}",
            self.subject,
            self.predicate,
            self.object,
            self.polarity.label(),
            self.modality.label()
        );
        if let Some(temporal) = &self.temporal {
            output.push_str(&format!("；{}", temporal.render_text()));
        }
        if !self.qualifiers.is_empty() {
            output.push_str(&format!("；{}", self.qualifiers.join("、")));
        }
        output.push('）');
        output
    }
}

/// 抽取统计：用于诊断「时间/否定/合取句」是否被看见。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameExtractionStats {
    pub sentences: usize,
    pub frames: usize,
    pub temporal_frames: usize,
    pub negative_frames: usize,
    pub modal_frames: usize,
    pub conjunction_frames: usize,
    pub parse_errors: usize,
}

/// 关系帧抽取器：复用文法抽取与中文命题解析，输出带证据的帧。
pub struct FrameExtractor {
    grammar: GrammarExtractor,
}

impl FrameExtractor {
    pub fn new() -> Self {
        Self {
            grammar: GrammarExtractor::new(),
        }
    }

    pub fn extract(&self, text: &str) -> Vec<RelationFrame> {
        self.extract_with_stats(text).0
    }

    pub fn extract_with_stats(&self, text: &str) -> (Vec<RelationFrame>, FrameExtractionStats) {
        let mut stats = FrameExtractionStats::default();
        let mut frames = Vec::new();
        let mut seen_ids = BTreeSet::new();

        for (sentence_index, (start, end)) in sentence_ranges(text).into_iter().enumerate() {
            stats.sentences += 1;
            let raw_sentence = &text[start..end];
            let body_offset = ingest_prefix_len(raw_sentence);
            let body = raw_sentence[body_offset..].trim();
            if body.is_empty() {
                continue;
            }

            let body_start = start
                + body_offset
                + (raw_sentence[body_offset..].len()
                    - raw_sentence[body_offset..].trim_start().len());
            let body_end = body_start + body.len();
            let temporal = parse_temporal(body);
            let polarity_span = negation_cue(body);
            let polarity = if polarity_span.is_some() {
                Polarity::Negative
            } else {
                Polarity::Affirmative
            };
            let modality = detect_modality(body);
            let conjunction = has_conjunction(body);
            let conditional = has_conditional(body);

            let extraction_text = strip_leading_temporal(body);
            let triples = self.collect_triples(extraction_text, &mut stats);
            for (subject, predicate, object, extraction) in triples {
                let mut frame = RelationFrame::new(&subject, &predicate, &object);
                frame.polarity = polarity;
                frame.modality = modality;
                frame.temporal = temporal.clone();
                frame.evidence = EvidenceSpan {
                    sentence_index,
                    sentence: body.to_string(),
                    start: body_start,
                    end: body_end,
                    subject_span: find_span(body, &subject),
                    object_span: find_span(body, &object),
                    polarity_span: polarity_span
                        .map(|(cue_start, cue_end)| (body_start + cue_start, body_start + cue_end)),
                };
                frame.extraction = extraction;
                frame.qualifiers = qualifiers(
                    polarity,
                    modality,
                    temporal.is_some(),
                    conjunction,
                    conditional,
                );
                frame.confidence = frame_confidence(polarity, modality, &frame.extraction);
                frame.refresh_id();
                if seen_ids.insert(frame.id.clone()) {
                    frames.push(frame);
                }
            }
        }

        stats.frames = frames.len();
        stats.temporal_frames = frames
            .iter()
            .filter(|frame| frame.temporal.is_some())
            .count();
        stats.negative_frames = frames
            .iter()
            .filter(|frame| frame.polarity == Polarity::Negative)
            .count();
        stats.modal_frames = frames
            .iter()
            .filter(|frame| frame.modality != Modality::Asserted)
            .count();
        stats.conjunction_frames = frames
            .iter()
            .filter(|frame| frame.qualifiers.iter().any(|qualifier| qualifier == "合取"))
            .count();
        (frames, stats)
    }

    fn collect_triples(
        &self,
        sentence: &str,
        stats: &mut FrameExtractionStats,
    ) -> Vec<(String, String, String, Vec<String>)> {
        let mut entries: Vec<(String, String, String, Vec<String>)> = Vec::new();
        let mut index: BTreeMap<(String, String, String), usize> = BTreeMap::new();

        for triple in self.grammar.extract(sentence) {
            push_triple(
                &mut entries,
                &mut index,
                &triple.subject,
                &triple.predicate,
                &triple.object,
                "文法",
            );
        }

        if !push_lesson(sentence, "模式", &mut entries, &mut index) {
            stats.parse_errors += 1;
            for clause in sentence.split(['，', ',']) {
                if clause.trim() != sentence.trim() {
                    push_lesson(clause, "模式（分句）", &mut entries, &mut index);
                }
            }
        }
        entries
    }
}

impl Default for FrameExtractor {
    fn default() -> Self {
        Self::new()
    }
}

fn push_lesson(
    text: &str,
    path: &str,
    entries: &mut Vec<(String, String, String, Vec<String>)>,
    index: &mut BTreeMap<(String, String, String), usize>,
) -> bool {
    match parse_lesson(text) {
        Ok(lesson)
            if lesson.kind == LessonKind::DeterministicFact
                && lesson.head.arity() == 2
                && lesson.head.is_ground() =>
        {
            push_triple(
                entries,
                index,
                lesson.head.terms[0].name(),
                &lesson.head.predicate,
                lesson.head.terms[1].name(),
                path,
            );
            true
        }
        _ => false,
    }
}

fn push_triple(
    entries: &mut Vec<(String, String, String, Vec<String>)>,
    index: &mut BTreeMap<(String, String, String), usize>,
    subject: &str,
    predicate: &str,
    object: &str,
    path: &str,
) {
    let subject = trim_negation_suffix(subject.trim());
    let predicate = predicate.trim();
    let object = object.trim();
    if predicate.is_empty()
        || predicate == "是"
        || subject.is_empty()
        || object.is_empty()
        || subject == object
        || subject.chars().count() > 40
        || object.chars().count() > 40
    {
        return;
    }
    let key = (
        subject.to_string(),
        predicate.to_string(),
        object.to_string(),
    );
    if let Some(position) = index.get(&key) {
        if !entries[*position].3.iter().any(|known| known == path) {
            entries[*position].3.push(path.to_string());
        }
        return;
    }
    index.insert(key, entries.len());
    entries.push((
        subject.to_string(),
        predicate.to_string(),
        object.to_string(),
        vec![path.to_string()],
    ));
}

/// 「张三不认识李四」被模式解析器切成「张三不 | 认识」时，去掉主语尾巴上的否定词。
fn trim_negation_suffix(subject: &str) -> &str {
    for suffix in ["没有", "不是", "不", "没", "未"] {
        if let Some(head) = subject.strip_suffix(suffix) {
            if head.chars().count() >= 1 {
                return head.trim_end();
            }
        }
    }
    subject
}

fn sentence_ranges(text: &str) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut start = 0usize;
    for (index, ch) in text.char_indices() {
        if matches!(ch, '。' | '！' | '？' | '；' | '!' | '?' | ';' | '\n') {
            if let Some(range) = trim_range(text, start, index) {
                ranges.push(range);
            }
            start = index + ch.len_utf8();
        }
    }
    if let Some(range) = trim_range(text, start, text.len()) {
        ranges.push(range);
    }
    ranges
}

fn trim_range(text: &str, mut start: usize, mut end: usize) -> Option<(usize, usize)> {
    while start < end {
        let ch = text[start..].chars().next()?;
        if ch.is_whitespace() {
            start += ch.len_utf8();
        } else {
            break;
        }
    }
    while end > start {
        let ch = text[..end].chars().next_back()?;
        if ch.is_whitespace() {
            end -= ch.len_utf8();
        } else {
            break;
        }
    }
    (start < end).then_some((start, end))
}

/// 抽取前去掉句首「1991年，」「去年，」这类时间状语；时间已经单独进入帧。
fn strip_leading_temporal(sentence: &str) -> &str {
    if let Some(capture) = year_regex().find(sentence) {
        if capture.start() == 0 {
            let rest = sentence[capture.end()..].trim_start();
            if rest.starts_with(['，', ',', '；', ';', '：', ':']) {
                return rest
                    .trim_start_matches(['，', ',', '；', ';', '：', ':'])
                    .trim_start();
            }
        }
    }
    for marker in [
        "大前天",
        "大后天",
        "上个月",
        "下个月",
        "上星期",
        "下星期",
        "去年",
        "今年",
        "明年",
        "前天",
        "昨天",
        "今天",
        "明天",
        "后天",
    ] {
        if let Some(rest) = sentence.strip_prefix(marker) {
            let rest = rest.trim_start();
            if rest.starts_with(['，', ',', '；', ';', '：', ':']) {
                return rest
                    .trim_start_matches(['，', ',', '；', ';', '：', ':'])
                    .trim_start();
            }
        }
    }
    sentence
}

fn ingest_prefix_len(sentence: &str) -> usize {
    for prefix in ["录入：", "添加：", "新增：", "记录：", "导入："] {
        if let Some(rest) = sentence.strip_prefix(prefix) {
            let leading = rest.len() - rest.trim_start().len();
            return prefix.len() + leading;
        }
    }
    0
}

fn find_span(text: &str, needle: &str) -> Option<(usize, usize)> {
    text.find(needle).map(|start| (start, start + needle.len()))
}

fn year_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(
            r"(?P<year>\d{4})\s*年(?:\s*(?P<month>\d{1,2})\s*月)?(?:\s*(?P<day>\d{1,2})\s*[日号])?",
        )
        .expect("合法时间正则")
    })
}

/// 解析句内时间表达。只识别显式年份与常见相对时间，不依赖系统时钟。
pub fn parse_temporal(text: &str) -> Option<TemporalScope> {
    const RELATIVE: &[(&str, &str)] = &[
        ("大前天", "相对:大前天"),
        ("大后天", "相对:大后天"),
        ("上个月", "相对:上个月"),
        ("下个月", "相对:下个月"),
        ("上星期", "相对:上星期"),
        ("下星期", "相对:下星期"),
        ("去年", "相对:去年"),
        ("今年", "相对:今年"),
        ("明年", "相对:明年"),
        ("前天", "相对:前天"),
        ("昨天", "相对:昨天"),
        ("今天", "相对:今天"),
        ("明天", "相对:明天"),
        ("后天", "相对:后天"),
    ];
    for (raw, normalized) in RELATIVE {
        if text.contains(raw) {
            return Some(TemporalScope {
                raw: (*raw).to_string(),
                normalized: Some((*normalized).to_string()),
                granularity: TemporalGranularity::Relative,
                relation: TemporalRelation::Point,
                start_year: None,
                end_year: None,
            });
        }
    }

    let captures: Vec<regex::Captures<'_>> = year_regex().captures_iter(text).collect();
    if captures.is_empty() {
        return None;
    }
    let parse = |value: Option<&str>| value.and_then(|value| value.parse::<i32>().ok());
    let first = &captures[0];
    let first_year = parse(first.name("year").map(|m| m.as_str()))?;
    let first_month = parse(first.name("month").map(|m| m.as_str()));
    let first_day = parse(first.name("day").map(|m| m.as_str()));
    let granularity = if first_day.is_some() {
        TemporalGranularity::Day
    } else if first_month.is_some() {
        TemporalGranularity::Month
    } else {
        TemporalGranularity::Year
    };
    let normalized = match (first_month, first_day) {
        (Some(month), Some(day)) => Some(format!("{:04}-{:02}-{:02}", first_year, month, day)),
        (Some(month), None) => Some(format!("{:04}-{:02}", first_year, month)),
        _ => Some(format!("{:04}", first_year)),
    };
    let relation = if text.contains("之前") || text.contains("以前") {
        TemporalRelation::Before
    } else if text.contains("之后") || text.contains("以后") {
        TemporalRelation::After
    } else if captures.len() > 1
        && (text.contains('到') || text.contains('至') || text.contains("从"))
    {
        TemporalRelation::Range
    } else {
        TemporalRelation::Point
    };
    let end_year = captures
        .last()
        .filter(|_| relation == TemporalRelation::Range)
        .and_then(|capture| parse(capture.name("year").map(|m| m.as_str())));
    Some(TemporalScope {
        raw: matches_text(text, first),
        normalized,
        granularity,
        relation,
        start_year: Some(first_year),
        end_year,
    })
}

fn matches_text(text: &str, capture: &regex::Captures<'_>) -> String {
    capture
        .get(0)
        .map(|matched| matched.as_str())
        .unwrap_or(text)
        .trim()
        .to_string()
}

fn negation_cue(text: &str) -> Option<(usize, usize)> {
    for cue in [
        "不是",
        "并非",
        "没有",
        "不属于",
        "不在",
        "未曾",
        "从未",
        "从不",
        "尚未",
        "未能",
        "无法",
    ] {
        if let Some(start) = text.find(cue) {
            return Some((start, start + cue.len()));
        }
    }
    for (start, _) in text.match_indices('不') {
        let rest = &text[start + '不'.len_utf8()..];
        if [
            "丹", "锈", "断", "同", "仅", "过", "久", "论", "少", "足", "错", "但", "管", "如",
            "妨", "准",
        ]
        .iter()
        .any(|suffix| rest.starts_with(suffix))
        {
            continue;
        }
        return Some((start, start + '不'.len_utf8()));
    }
    None
}

fn detect_modality(text: &str) -> Modality {
    if text.contains("不一定")
        || text.contains("未必")
        || text.contains("据说")
        || text.contains("听说")
        || text.contains("据称")
    {
        return Modality::Uncertain;
    }
    if ["可能", "也许", "或许", "大概", "大约", "似乎", "好像"]
        .iter()
        .any(|cue| text.contains(cue))
    {
        return Modality::Possible;
    }
    if ["必须", "必然", "必定", "一定", "肯定"]
        .iter()
        .any(|cue| text.contains(cue))
    {
        return Modality::Necessary;
    }
    Modality::Asserted
}

fn has_conjunction(text: &str) -> bool {
    ["并且", "同时", "而且", "以及", "并非只有", "既", "又"]
        .iter()
        .any(|cue| text.contains(cue))
}

fn has_conditional(text: &str) -> bool {
    ["如果", "若", "只要", "除非", "因为", "由于", "一旦", "假如"]
        .iter()
        .any(|cue| text.contains(cue))
}

fn qualifiers(
    polarity: Polarity,
    modality: Modality,
    temporal: bool,
    conjunction: bool,
    conditional: bool,
) -> Vec<String> {
    let mut qualifiers = Vec::new();
    if polarity == Polarity::Negative {
        qualifiers.push("否定".to_string());
    }
    if modality != Modality::Asserted {
        qualifiers.push(format!("模态:{}", modality.label()));
    }
    if temporal {
        qualifiers.push("时间限定".to_string());
    }
    if conjunction {
        qualifiers.push("合取".to_string());
    }
    if conditional {
        qualifiers.push("条件/因果".to_string());
    }
    qualifiers
}

fn frame_confidence(polarity: Polarity, modality: Modality, extraction: &[String]) -> f64 {
    let mut confidence: f64 = if extraction.iter().any(|path| path == "文法") {
        0.86
    } else {
        0.80
    };
    if extraction.len() > 1 {
        confidence += 0.04;
    }
    if polarity == Polarity::Negative {
        confidence -= 0.05;
    }
    if modality != Modality::Asserted {
        confidence -= 0.08;
    }
    confidence.clamp(0.2, 0.99)
}

/// 闭包中的一个规范化三元组。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct FrameEdge {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

impl FrameEdge {
    pub fn new(subject: &str, predicate: &str, object: &str) -> Self {
        Self {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
        }
    }

    pub fn render_text(&self) -> String {
        format!("{}({}, {})", self.predicate, self.subject, self.object)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateVerdict {
    DirectSupport,
    InferredSupport,
    Conflict,
    Unsupported,
}

impl GateVerdict {
    pub fn label(self) -> &'static str {
        match self {
            GateVerdict::DirectSupport => "直接支持",
            GateVerdict::InferredSupport => "推理支持",
            GateVerdict::Conflict => "冲突",
            GateVerdict::Unsupported => "无证据（开放世界）",
        }
    }

    pub fn is_support(self) -> bool {
        matches!(
            self,
            GateVerdict::DirectSupport | GateVerdict::InferredSupport
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateSeverity {
    Info,
    Warning,
    Violation,
}

impl GateSeverity {
    pub fn label(self) -> &'static str {
        match self {
            GateSeverity::Info => "提示",
            GateSeverity::Warning => "警告",
            GateSeverity::Violation => "违规",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateFindingKind {
    DirectFact,
    ExplicitNegative,
    InferredFact,
    InverseNormalized,
    FunctionalConflict,
    SchemaViolation,
    NegativeConflict,
    UnknownPredicate,
    ClosureGain,
}

impl GateFindingKind {
    pub fn tag(self) -> &'static str {
        match self {
            GateFindingKind::DirectFact => "direct_fact",
            GateFindingKind::ExplicitNegative => "explicit_negative",
            GateFindingKind::InferredFact => "inferred_fact",
            GateFindingKind::InverseNormalized => "inverse_normalized",
            GateFindingKind::FunctionalConflict => "functional_conflict",
            GateFindingKind::SchemaViolation => "schema_violation",
            GateFindingKind::NegativeConflict => "negative_conflict",
            GateFindingKind::UnknownPredicate => "unknown_predicate",
            GateFindingKind::ClosureGain => "closure_gain",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            GateFindingKind::DirectFact => "直接事实",
            GateFindingKind::ExplicitNegative => "显式否定",
            GateFindingKind::InferredFact => "演绎闭包",
            GateFindingKind::InverseNormalized => "互逆归一",
            GateFindingKind::FunctionalConflict => "单值谓词冲突",
            GateFindingKind::SchemaViolation => "定义域/值域冲突",
            GateFindingKind::NegativeConflict => "否定与事实冲突",
            GateFindingKind::UnknownPredicate => "未登记谓词",
            GateFindingKind::ClosureGain => "闭包增益",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GateFinding {
    pub kind: GateFindingKind,
    pub severity: GateSeverity,
    pub detail: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub proof: Vec<FrameEdge>,
}

impl GateFinding {
    fn new(kind: GateFindingKind, severity: GateSeverity, detail: String) -> Self {
        Self {
            kind,
            severity,
            detail,
            proof: Vec::new(),
        }
    }

    fn with_proof(mut self, proof: Vec<FrameEdge>) -> Self {
        self.proof = proof;
        self
    }
}

/// 单帧闸门报告。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameGateReport {
    pub frame_id: String,
    pub normalized: FrameEdge,
    pub inverted: bool,
    pub verdict: GateVerdict,
    pub findings: Vec<GateFinding>,
    pub closure_size: usize,
    pub closure_gain: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub support_path: Vec<FrameEdge>,
}

impl FrameGateReport {
    pub fn render_text(&self) -> String {
        let mut output = format!(
            "判定：{}；闭包 {} 条边，增益 {}。",
            self.verdict.label(),
            self.closure_size,
            self.closure_gain
        );
        for finding in &self.findings {
            output.push_str(&format!(
                "\n  [{}] {}：{}",
                finding.severity.label(),
                finding.kind.label(),
                finding.detail
            ));
        }
        output
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClosureLimits {
    pub max_depth: usize,
    pub max_edges: usize,
}

impl Default for ClosureLimits {
    fn default() -> Self {
        Self {
            max_depth: DEFAULT_CLOSURE_DEPTH,
            max_edges: DEFAULT_CLOSURE_LIMIT,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct ClosureRules {
    transitive: BTreeSet<String>,
    symmetric: BTreeSet<String>,
    inverse: BTreeMap<String, String>,
    functional: BTreeSet<String>,
}

impl ClosureRules {
    fn build(facts: &BTreeSet<FrameEdge>, registry: &PredicateRegistry, laws: &LawBook) -> Self {
        let mut rules = Self::default();
        for law in laws.laws() {
            if !law.is_active() {
                continue;
            }
            match law.kind {
                LawKind::Transitive => {
                    if let Some(predicate) = law.terms.first() {
                        rules.transitive.insert(predicate.clone());
                    }
                }
                LawKind::Symmetric => {
                    if let Some(predicate) = law.terms.first() {
                        rules.symmetric.insert(predicate.clone());
                    }
                }
                LawKind::Inverse => {
                    if law.terms.len() >= 2 {
                        rules
                            .inverse
                            .insert(law.terms[0].clone(), law.terms[1].clone());
                        rules
                            .inverse
                            .insert(law.terms[1].clone(), law.terms[0].clone());
                    }
                }
                LawKind::Functional => {
                    if let Some(predicate) = law.terms.first() {
                        rules.functional.insert(predicate.clone());
                    }
                }
                _ => {}
            }
        }
        for fact in facts {
            if registry.is_transitive(&fact.predicate) {
                rules.transitive.insert(fact.predicate.clone());
            }
            if registry.is_symmetric(&fact.predicate) {
                rules.symmetric.insert(fact.predicate.clone());
            }
            if registry.is_functional(&fact.predicate) {
                rules.functional.insert(fact.predicate.clone());
            }
            if let Some(inverse) = registry.inverse_of(&fact.predicate) {
                rules
                    .inverse
                    .insert(fact.predicate.clone(), inverse.to_string());
                rules
                    .inverse
                    .insert(inverse.to_string(), fact.predicate.clone());
            }
        }
        rules
    }
}

/// 事实索引：只按谓词分组，支持局部展开。
#[derive(Debug, Clone, Default)]
struct ClosureIndex {
    out: BTreeMap<String, BTreeMap<String, BTreeSet<String>>>,
    inc: BTreeMap<String, BTreeMap<String, BTreeSet<String>>>,
}

impl ClosureIndex {
    fn build(facts: &BTreeSet<FrameEdge>) -> Self {
        let mut index = Self::default();
        for fact in facts {
            index
                .out
                .entry(fact.predicate.clone())
                .or_default()
                .entry(fact.subject.clone())
                .or_default()
                .insert(fact.object.clone());
            index
                .inc
                .entry(fact.predicate.clone())
                .or_default()
                .entry(fact.object.clone())
                .or_default()
                .insert(fact.subject.clone());
        }
        index
    }

    fn objects(
        &self,
        predicate: &str,
        subject: &str,
        extra: Option<&FrameEdge>,
    ) -> BTreeSet<String> {
        let mut objects = self
            .out
            .get(predicate)
            .and_then(|by_subject| by_subject.get(subject))
            .cloned()
            .unwrap_or_default();
        if let Some(extra) = extra {
            if extra.predicate == predicate && extra.subject == subject {
                objects.insert(extra.object.clone());
            }
        }
        objects
    }

    fn incident(&self, node: &str) -> BTreeSet<FrameEdge> {
        let mut edges = BTreeSet::new();
        for (predicate, by_subject) in &self.out {
            if let Some(objects) = by_subject.get(node) {
                for object in objects {
                    edges.insert(FrameEdge::new(node, predicate, object));
                }
            }
        }
        for (predicate, by_object) in &self.inc {
            if let Some(subjects) = by_object.get(node) {
                for subject in subjects {
                    edges.insert(FrameEdge::new(subject, predicate, node));
                }
            }
        }
        edges
    }
}

/// 演绎闸门：把关系帧与图谱、谓词画像、类型体系、律典放在一起判定。
pub struct FrameGate {
    facts: BTreeSet<FrameEdge>,
    negative_facts: BTreeSet<FrameEdge>,
    entity_types: BTreeMap<String, BTreeSet<String>>,
    registry: PredicateRegistry,
    types: TypeSystem,
    rules: ClosureRules,
    index: ClosureIndex,
    limits: ClosureLimits,
    /// 事实的有效期（sidecar，不改变三元组身份）；缺失即 Unknown。
    fact_validity: BTreeMap<FrameEdge, Validity>,
}

impl FrameGate {
    pub fn new(
        facts: Vec<FrameEdge>,
        negative_facts: Vec<FrameEdge>,
        entity_types: BTreeMap<String, BTreeSet<String>>,
        registry: PredicateRegistry,
        types: TypeSystem,
        laws: LawBook,
    ) -> Self {
        let normalize = |edge: FrameEdge| normalize_edge(&registry, &laws, &edge);
        let facts: BTreeSet<FrameEdge> = facts.into_iter().map(normalize).collect();
        let negative_facts: BTreeSet<FrameEdge> =
            negative_facts.into_iter().map(normalize).collect();
        let rules = ClosureRules::build(&facts, &registry, &laws);
        let index = ClosureIndex::build(&facts);
        Self {
            facts,
            negative_facts,
            entity_types,
            registry,
            types,
            rules,
            index,
            limits: ClosureLimits::default(),
            fact_validity: BTreeMap::new(),
        }
    }

    /// 注入事实有效期（sidecar）。同一个三元组以最后一次注入为准。
    pub fn with_fact_validity(
        mut self,
        entries: impl IntoIterator<Item = (FrameEdge, Validity)>,
    ) -> Self {
        self.fact_validity = entries.into_iter().collect();
        self
    }

    pub fn from_facts(
        facts: &[PrologFact],
        registry: PredicateRegistry,
        types: TypeSystem,
        laws: LawBook,
    ) -> Self {
        let mut edges = Vec::new();
        let mut entity_types: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        let mut validity_by_entity: BTreeMap<String, Validity> = BTreeMap::new();
        for fact in facts {
            match fact {
                PrologFact::Entity {
                    name,
                    entity_type,
                    properties,
                } => {
                    if !entity_type.is_empty() {
                        entity_types
                            .entry(name.clone())
                            .or_default()
                            .insert(entity_type.clone());
                    }
                    if let Some(validity) = temporal::validity_from_properties(properties) {
                        validity_by_entity.insert(name.clone(), validity);
                        if let Some(alias) = properties.get("name") {
                            validity_by_entity.insert(alias.clone(), validity);
                        }
                    }
                    // 实体属性（年龄、身高、首都等）在图中不是三元组关系，
                    // 但对帧而言同样是二元谓词；只收字面量值，跳过关系 IRI。
                    for (predicate, value) in properties {
                        if predicate.trim().is_empty()
                            || value.trim().is_empty()
                            || value.starts_with("http://")
                            || value.starts_with("https://")
                            || value.starts_with('<')
                        {
                            continue;
                        }
                        edges.push(FrameEdge::new(name, predicate, value));
                    }
                }
                PrologFact::Relation { from, rel_type, to } => {
                    edges.push(FrameEdge::new(from, rel_type, to));
                }
            }
        }
        let validity: Vec<(FrameEdge, Validity)> = edges
            .iter()
            .filter_map(|edge| {
                validity_by_entity
                    .get(&edge.subject)
                    .copied()
                    .filter(Validity::is_known)
                    .map(|validity| (edge.clone(), validity))
            })
            .collect();
        Self::new(edges, Vec::new(), entity_types, registry, types, laws)
            .with_fact_validity(validity)
    }

    pub fn with_limits(mut self, limits: ClosureLimits) -> Self {
        self.limits = limits;
        self
    }

    pub fn fact_count(&self) -> usize {
        self.facts.len()
    }

    pub fn evaluate(&self, frame: &RelationFrame) -> FrameGateReport {
        let (normalized, inverted) =
            normalize_edge_detailed(&self.registry, &self.entity_types, frame);
        let negative = frame.polarity == Polarity::Negative;
        let frame_validity = frame
            .temporal
            .as_ref()
            .map(TemporalScope::to_validity)
            .unwrap_or(Validity::Unknown);
        let direct_positive = self.facts.contains(&normalized)
            && self.validities_compatible(frame_validity, &normalized);
        let direct_negative = self.negative_facts.contains(&normalized);

        let without = self.local_closure(&normalized, None);
        let with = if negative {
            None
        } else {
            Some(self.local_closure(&normalized, Some(&normalized)))
        };
        let inferred_positive = !direct_positive && without.contains_key(&normalized);
        let candidate_novel =
            !direct_positive && !direct_negative && !without.contains_key(&normalized);
        let closure_gain = with
            .as_ref()
            .map(|with| {
                with.len()
                    .saturating_sub(without.len())
                    .saturating_sub(usize::from(candidate_novel))
            })
            .unwrap_or(0);

        let mut findings = Vec::new();
        if inverted {
            findings.push(GateFinding::new(
                GateFindingKind::InverseNormalized,
                GateSeverity::Info,
                format!(
                    "谓词「{}」按互逆方向归一为 {}",
                    frame.predicate, normalized.predicate
                ),
            ));
        }

        let known_predicate = self.registry.lookup(&frame.predicate).is_some();
        if !known_predicate {
            findings.push(GateFinding::new(
                GateFindingKind::UnknownPredicate,
                GateSeverity::Warning,
                format!(
                    "谓词「{}」不在谓词画像中，闸门只做结构检查",
                    frame.predicate
                ),
            ));
        }

        let mut conflict = false;

        if direct_positive {
            if negative {
                conflict = true;
                findings.push(GateFinding::new(
                    GateFindingKind::NegativeConflict,
                    GateSeverity::Violation,
                    format!("否定帧与图谱事实 {} 冲突", normalized.render_text()),
                ));
            } else {
                findings.push(GateFinding::new(
                    GateFindingKind::DirectFact,
                    GateSeverity::Info,
                    format!("图谱直接记录 {}", normalized.render_text()),
                ));
            }
        }
        if direct_negative {
            if negative {
                findings.push(GateFinding::new(
                    GateFindingKind::ExplicitNegative,
                    GateSeverity::Info,
                    format!("存在显式否定记录 {}", normalized.render_text()),
                ));
            } else {
                conflict = true;
                findings.push(GateFinding::new(
                    GateFindingKind::NegativeConflict,
                    GateSeverity::Violation,
                    format!("肯定帧与显式否定记录 {} 冲突", normalized.render_text()),
                ));
            }
        }

        if inferred_positive {
            if negative {
                conflict = true;
                findings.push(
                    GateFinding::new(
                        GateFindingKind::NegativeConflict,
                        GateSeverity::Violation,
                        format!("否定帧与演绎闭包中的 {} 冲突", normalized.render_text()),
                    )
                    .with_proof(without.get(&normalized).cloned().unwrap_or_default()),
                );
            } else {
                findings.push(
                    GateFinding::new(
                        GateFindingKind::InferredFact,
                        GateSeverity::Info,
                        format!(
                            "局部闭包推出 {}（{} 步内）",
                            normalized.render_text(),
                            self.limits.max_depth
                        ),
                    )
                    .with_proof(without.get(&normalized).cloned().unwrap_or_default()),
                );
            }
        }

        if !negative {
            let values = self.functional_conflict_values(&normalized, frame_validity);
            if !values.is_empty() {
                conflict = true;
                findings.push(GateFinding::new(
                    GateFindingKind::FunctionalConflict,
                    GateSeverity::Violation,
                    format!(
                        "单值谓词「{}」对主语「{}」已有其他取值：{}",
                        normalized.predicate,
                        normalized.subject,
                        values.join("、")
                    ),
                ));
            }

            if let Some(violation) = self.schema_violation(&normalized) {
                conflict = true;
                findings.push(GateFinding::new(
                    GateFindingKind::SchemaViolation,
                    GateSeverity::Violation,
                    format!(
                        "{}「{}」不在声明的{}内：期望 {}，实际 {}",
                        violation.slot,
                        if violation.slot == "定义域" {
                            &violation.subject
                        } else {
                            &violation.object
                        },
                        if violation.slot == "定义域" {
                            "定义域"
                        } else {
                            "值域"
                        },
                        violation.expected.join(" / "),
                        violation.actual.join(" / ")
                    ),
                ));
            }
        }

        if closure_gain > 0 {
            findings.push(GateFinding::new(
                GateFindingKind::ClosureGain,
                GateSeverity::Info,
                format!("加入该帧后局部闭包新增 {} 条可复用的边", closure_gain),
            ));
        }

        let verdict = if conflict {
            GateVerdict::Conflict
        } else if direct_positive || direct_negative {
            GateVerdict::DirectSupport
        } else if inferred_positive {
            GateVerdict::InferredSupport
        } else {
            GateVerdict::Unsupported
        };
        let support_path = without.get(&normalized).cloned().unwrap_or_default();
        FrameGateReport {
            frame_id: frame.id.clone(),
            normalized,
            inverted,
            verdict,
            findings,
            closure_size: with
                .as_ref()
                .map(|closure| closure.len())
                .unwrap_or_else(|| without.len()),
            closure_gain,
            support_path,
        }
    }

    fn validities_compatible(&self, frame_validity: Validity, edge: &FrameEdge) -> bool {
        let fact_validity = self
            .fact_validity
            .get(edge)
            .copied()
            .unwrap_or(Validity::Unknown);
        if frame_validity.is_unknown() || fact_validity.is_unknown() {
            return true;
        }
        frame_validity.intersection(fact_validity).is_satisfiable()
    }

    fn functional_conflict_values(
        &self,
        edge: &FrameEdge,
        frame_validity: Validity,
    ) -> Vec<String> {
        let functional = self.rules.functional.contains(&edge.predicate)
            || self
                .registry
                .profile(&edge.predicate)
                .map(|profile| profile.functional)
                .unwrap_or(false);
        if !functional {
            return Vec::new();
        }
        self.facts
            .iter()
            .filter(|fact| {
                fact.subject == edge.subject
                    && fact.predicate == edge.predicate
                    && fact.object != edge.object
                    && self.validities_compatible(frame_validity, fact)
            })
            .map(|fact| fact.object.clone())
            .collect()
    }

    fn schema_violation(
        &self,
        edge: &FrameEdge,
    ) -> Option<crate::predicate_registry::SchemaViolation> {
        let subject_types = self
            .entity_types
            .get(&edge.subject)
            .cloned()
            .unwrap_or_default();
        let object_types = self
            .entity_types
            .get(&edge.object)
            .cloned()
            .unwrap_or_default();
        if self.types.closure_gate() {
            self.registry.check_schema_with_types(
                &edge.subject,
                &edge.predicate,
                &edge.object,
                &subject_types,
                &object_types,
                &self.types,
            )
        } else {
            self.registry.check_schema(
                &edge.subject,
                &edge.predicate,
                &edge.object,
                &subject_types,
                &object_types,
            )
        }
    }

    fn local_closure(
        &self,
        edge: &FrameEdge,
        extra: Option<&FrameEdge>,
    ) -> BTreeMap<FrameEdge, Vec<FrameEdge>> {
        let mut proofs: BTreeMap<FrameEdge, Vec<FrameEdge>> = BTreeMap::new();
        let mut queue: VecDeque<(FrameEdge, usize)> = VecDeque::new();
        let mut seeds = self.index.incident(&edge.subject);
        seeds.extend(self.index.incident(&edge.object));
        if let Some(extra) = extra {
            seeds.insert(extra.clone());
        }
        for seed in seeds {
            if proofs.len() >= self.limits.max_edges {
                break;
            }
            if proofs.insert(seed.clone(), Vec::new()).is_none() {
                queue.push_back((seed, 0));
            }
        }

        while let Some((current, depth)) = queue.pop_front() {
            if depth >= self.limits.max_depth {
                continue;
            }
            for (derived, premises) in self.derive_from(&current, extra) {
                if proofs.len() >= self.limits.max_edges {
                    break;
                }
                // 保留 BFS 中首次（即最短）的证明；后续等价推导不覆盖，
                // 否则互逆边会把传递证明替换成循环引用。
                if proofs.contains_key(&derived) {
                    continue;
                }
                proofs.insert(derived.clone(), premises);
                queue.push_back((derived, depth + 1));
            }
        }
        proofs
    }

    fn derive_from(
        &self,
        edge: &FrameEdge,
        extra: Option<&FrameEdge>,
    ) -> Vec<(FrameEdge, Vec<FrameEdge>)> {
        let mut derived = Vec::new();
        if self.rules.transitive.contains(&edge.predicate) {
            for object in self.index.objects(&edge.predicate, &edge.object, extra) {
                if object == edge.subject {
                    continue;
                }
                let mut premises = vec![edge.clone()];
                let other = FrameEdge::new(&edge.object, &edge.predicate, &object);
                if other != *edge {
                    premises.push(other);
                }
                derived.push((
                    FrameEdge::new(&edge.subject, &edge.predicate, &object),
                    premises,
                ));
            }
        }
        if self.rules.symmetric.contains(&edge.predicate) {
            derived.push((
                FrameEdge::new(&edge.object, &edge.predicate, &edge.subject),
                vec![edge.clone()],
            ));
        }
        if let Some(inverse) = self.rules.inverse.get(&edge.predicate) {
            derived.push((
                FrameEdge::new(&edge.object, inverse, &edge.subject),
                vec![edge.clone()],
            ));
        }
        derived
    }
}

fn normalize_edge(registry: &PredicateRegistry, laws: &LawBook, edge: &FrameEdge) -> FrameEdge {
    let resolution = registry.resolve(&edge.predicate);
    let (canonical, inverted) = if resolution.known {
        (resolution.canonical, resolution.inverted)
    } else if let Some((canonical, inverted)) = laws.alias_map().get(edge.predicate.trim()) {
        (canonical.clone(), *inverted)
    } else {
        (edge.predicate.trim().to_string(), false)
    };
    if inverted {
        FrameEdge::new(&edge.object, &canonical, &edge.subject)
    } else {
        FrameEdge::new(&edge.subject, &canonical, &edge.object)
    }
}

fn normalize_edge_detailed(
    registry: &PredicateRegistry,
    _entity_types: &BTreeMap<String, BTreeSet<String>>,
    frame: &RelationFrame,
) -> (FrameEdge, bool) {
    let edge = FrameEdge::new(&frame.subject, &frame.predicate, &frame.object);
    let resolution = registry.resolve(&frame.predicate);
    if resolution.known {
        let normalized = if resolution.inverted {
            FrameEdge::new(&frame.object, &resolution.canonical, &frame.subject)
        } else {
            FrameEdge::new(&frame.subject, &resolution.canonical, &frame.object)
        };
        (normalized, resolution.inverted)
    } else {
        (edge, false)
    }
}

/// 宾语像属性值（数字、数量、日期等）时，用「X 的 P 是 V」句式；
/// 否则用更中性的「X 与 Y 之间存在 P 关系」，避免把动词关系说成属性。
fn value_like(object: &str) -> bool {
    let object = object.trim();
    if object.is_empty() {
        return false;
    }
    if object.chars().any(|ch| ch.is_ascii_digit()) {
        return true;
    }
    const VALUE_SUFFIXES: &[&str] = &[
        "岁",
        "年",
        "月",
        "日",
        "号",
        "元",
        "万元",
        "亿元",
        "米",
        "千米",
        "公里",
        "平方米",
        "平方公里",
        "千克",
        "公斤",
        "克",
        "吨",
        "人",
        "名",
        "个",
        "次",
        "天",
        "小时",
        "分钟",
        "%",
        "％",
    ];
    VALUE_SUFFIXES.iter().any(|suffix| object.ends_with(suffix))
}

/// 把闸门判定与证据整理成中文叙述。
pub fn narrate_frame(frame: &RelationFrame, report: &FrameGateReport) -> String {
    let mut output = String::new();
    let statement = if value_like(&frame.object) {
        if frame.polarity == Polarity::Negative {
            format!(
                "否定陈述：{}的{}不是{}。",
                frame.subject, frame.predicate, frame.object
            )
        } else {
            format!(
                "陈述：{}的{}是{}。",
                frame.subject, frame.predicate, frame.object
            )
        }
    } else if frame.polarity == Polarity::Negative {
        format!(
            "否定陈述：{}与{}之间不存在「{}」关系。",
            frame.subject, frame.object, frame.predicate
        )
    } else {
        format!(
            "陈述：{}与{}之间存在「{}」关系。",
            frame.subject, frame.object, frame.predicate
        )
    };
    output.push_str(&statement);
    if let Some(temporal) = &frame.temporal {
        output.push_str(&format!("时间限定：{}。", temporal.render_text()));
    }
    if frame.modality != Modality::Asserted {
        output.push_str(&format!("语气：{}。", frame.modality.label()));
    }
    output.push_str(&format!("闸门判定：{}。", report.verdict.label()));
    if !report.support_path.is_empty() {
        output.push_str(&format!(
            "证明路径：{}。",
            report
                .support_path
                .iter()
                .map(FrameEdge::render_text)
                .collect::<Vec<_>>()
                .join("；")
        ));
    }
    for finding in &report.findings {
        if finding.severity == GateSeverity::Info {
            continue;
        }
        output.push_str(&format!("{}：{}。", finding.kind.label(), finding.detail));
    }
    output
}

/// 一帧诊断样本：帧 + 闸门报告。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrameDiagnosticSample {
    pub frame: RelationFrame,
    pub report: FrameGateReport,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FrameDiagnostics {
    pub dialect: String,
    pub frames: usize,
    pub direct_support: usize,
    pub inferred_support: usize,
    pub conflicts: usize,
    pub unsupported: usize,
    pub temporal_frames: usize,
    pub negative_frames: usize,
    pub modal_frames: usize,
    pub conjunction_frames: usize,
    pub closure_gain_total: usize,
    pub finding_counts: BTreeMap<String, usize>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub samples: Vec<FrameDiagnosticSample>,
}

impl FrameDiagnostics {
    pub fn render_text(&self) -> String {
        let mut output = format!(
            "帧诊断：{} 条帧；直接支持 {}，推理支持 {}，冲突 {}，无证据 {}。\n",
            self.frames,
            self.direct_support,
            self.inferred_support,
            self.conflicts,
            self.unsupported
        );
        output.push_str(&format!(
            "限定覆盖：时间 {}，否定 {}，模态 {}，合取 {}；闭包总增益 {}。\n",
            self.temporal_frames,
            self.negative_frames,
            self.modal_frames,
            self.conjunction_frames,
            self.closure_gain_total
        ));
        if !self.finding_counts.is_empty() {
            output.push_str("发现计数：");
            output.push_str(
                &self
                    .finding_counts
                    .iter()
                    .map(|(kind, count)| format!("{}={}", kind, count))
                    .collect::<Vec<_>>()
                    .join("，"),
            );
            output.push('\n');
        }
        output
    }
}

/// 对一批帧跑闸门并汇总诊断。
pub fn diagnose(
    frames: &[RelationFrame],
    gate: &FrameGate,
    sample_limit: usize,
) -> FrameDiagnostics {
    let mut diagnostics = FrameDiagnostics {
        dialect: FRAME_DIALECT.to_string(),
        frames: frames.len(),
        ..Default::default()
    };
    for frame in frames {
        let report = gate.evaluate(frame);
        match report.verdict {
            GateVerdict::DirectSupport => diagnostics.direct_support += 1,
            GateVerdict::InferredSupport => diagnostics.inferred_support += 1,
            GateVerdict::Conflict => diagnostics.conflicts += 1,
            GateVerdict::Unsupported => diagnostics.unsupported += 1,
        }
        diagnostics.closure_gain_total += report.closure_gain;
        if frame.temporal.is_some() {
            diagnostics.temporal_frames += 1;
        }
        if frame.polarity == Polarity::Negative {
            diagnostics.negative_frames += 1;
        }
        if frame.modality != Modality::Asserted {
            diagnostics.modal_frames += 1;
        }
        if frame.qualifiers.iter().any(|qualifier| qualifier == "合取") {
            diagnostics.conjunction_frames += 1;
        }
        for finding in &report.findings {
            *diagnostics
                .finding_counts
                .entry(finding.kind.tag().to_string())
                .or_insert(0) += 1;
        }
        if diagnostics.samples.len() < sample_limit {
            diagnostics.samples.push(FrameDiagnosticSample {
                frame: frame.clone(),
                report,
            });
        }
    }
    diagnostics
}

/// 规范 JSON（用于诊断输出的稳定摘要）。
pub fn diagnostics_digest(diagnostics: &FrameDiagnostics) -> String {
    let value = serde_json::to_value(diagnostics).unwrap_or(serde_json::Value::Null);
    fnv1a_hex(canonical_json(&value).as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::predicate_registry::PredicateProfileConfig;
    use std::collections::HashMap;

    fn profile(name: &str) -> PredicateProfileConfig {
        PredicateProfileConfig {
            name: name.to_string(),
            ..Default::default()
        }
    }

    fn registry() -> PredicateRegistry {
        let mut registry = PredicateRegistry::new();
        let mut master = profile("师傅");
        master.transitive = true;
        master.inverse = Some("徒弟".to_string());
        master.domain = vec!["人物".to_string()];
        master.range = vec!["人物".to_string()];
        registry.declare(&master);
        let mut apprentice = profile("徒弟");
        apprentice.inverse = Some("师傅".to_string());
        registry.declare(&apprentice);
        let mut age = profile("年龄");
        age.functional = true;
        registry.declare(&age);
        registry
    }

    fn gate(edges: &[(&str, &str, &str)]) -> FrameGate {
        let registry = registry();
        let laws = LawBook::from_registry(&registry);
        let facts = edges
            .iter()
            .map(|(s, p, o)| FrameEdge::new(s, p, o))
            .collect();
        FrameGate::new(
            facts,
            Vec::new(),
            BTreeMap::new(),
            registry,
            TypeSystem::new(),
            laws,
        )
    }

    #[test]
    fn parses_explicit_and_relative_temporal_scopes() {
        let year = parse_temporal("公司成立于1999年").expect("year");
        assert_eq!(year.normalized.as_deref(), Some("1999"));
        assert_eq!(year.relation, TemporalRelation::Point);

        let range = parse_temporal("从1991年到1999年").expect("range");
        assert_eq!(range.relation, TemporalRelation::Range);
        assert_eq!(range.start_year, Some(1991));
        assert_eq!(range.end_year, Some(1999));

        let relative = parse_temporal("他去年出生").expect("relative");
        assert_eq!(relative.granularity, TemporalGranularity::Relative);
        assert!(relative.to_time_span().is_none());
        assert_eq!(relative.to_validity(), Validity::Unknown);
        assert!(negation_cue("不丹的首都是廷布").is_none());
        assert!(negation_cue("张三不使用标注数据").is_some());
    }

    #[test]
    fn from_facts_maps_validity_across_name_alias() {
        let mut registry = PredicateRegistry::new();
        let mut capital = profile("首都");
        capital.functional = true;
        registry.declare(&capital);
        let laws = LawBook::from_registry(&registry);
        let mut kyoto: HashMap<String, String> = HashMap::new();
        kyoto.insert("name".to_string(), "京都".to_string());
        kyoto.insert("valid_to".to_string(), "1867".to_string());
        let facts = vec![
            PrologFact::Entity {
                name: "kyoto".to_string(),
                entity_type: "城市".to_string(),
                properties: kyoto,
            },
            PrologFact::Entity {
                name: "japan".to_string(),
                entity_type: "国家".to_string(),
                properties: HashMap::from([("name".to_string(), "日本".to_string())]),
            },
            PrologFact::Relation {
                from: "京都".to_string(),
                rel_type: "首都".to_string(),
                to: "日本".to_string(),
            },
        ];
        let gate = FrameGate::from_facts(&facts, registry, TypeSystem::new(), laws);
        let frame = RelationFrame::new("日本", "首都", "京都")
            .with_temporal(parse_temporal("1868年").expect("year"));
        assert_eq!(gate.evaluate(&frame).verdict, GateVerdict::Unsupported);
    }

    #[test]
    fn functional_conflicts_are_time_aware() {
        let mut registry = PredicateRegistry::new();
        let mut capital = profile("首都");
        capital.functional = true;
        registry.declare(&capital);
        let laws = LawBook::from_registry(&registry);
        let kyoto = FrameEdge::new("日本", "首都", "京都");
        let tokyo = FrameEdge::new("日本", "首都", "东京");
        let gate = FrameGate::new(
            vec![kyoto.clone(), tokyo.clone()],
            Vec::new(),
            BTreeMap::new(),
            registry,
            TypeSystem::new(),
            laws,
        )
        .with_fact_validity([
            (kyoto, Validity::from_years(-794, 1867)),
            (tokyo, Validity::from_years(1868, 9999)),
        ]);

        let frame = RelationFrame::new("日本", "首都", "东京")
            .with_temporal(parse_temporal("1868年").expect("year"));
        let report = gate.evaluate(&frame);
        assert_eq!(report.verdict, GateVerdict::DirectSupport);
        assert!(
            !report
                .findings
                .iter()
                .any(|finding| finding.kind == GateFindingKind::FunctionalConflict),
            "历史更替不应误报单值冲突：{:?}",
            report.findings
        );

        let osaka = RelationFrame::new("日本", "首都", "大阪")
            .with_temporal(parse_temporal("1800年").expect("year"));
        let report = gate.evaluate(&osaka);
        assert_eq!(report.verdict, GateVerdict::Conflict);
        let conflict = report
            .findings
            .iter()
            .find(|finding| finding.kind == GateFindingKind::FunctionalConflict)
            .expect("1800 年与京都有效期重叠，应报冲突");
        assert!(conflict.detail.contains("京都"));
        assert!(!conflict.detail.contains("东京"), "{}", conflict.detail);
    }

    #[test]
    fn temporal_scope_converts_to_interval_algebra() {
        let year = parse_temporal("公司成立于1999年").expect("year");
        let span = year.to_time_span().expect("year span");
        assert!(span.contains(TimePoint::day(1999, 6, 1)));
        assert_eq!(year.to_validity(), Validity::from_years(1999, 1999));

        let range = parse_temporal("从1991年到1999年").expect("range");
        let span = range.to_time_span().expect("range span");
        assert!(span.contains(TimePoint::year(1995)));
        assert!(!span.contains(TimePoint::year(2000)));
        assert_eq!(range.to_validity(), Validity::from_years(1991, 1999));
    }

    #[test]
    fn extracts_frames_with_evidence_and_modality() {
        let (frames, stats) = FrameExtractor::new().extract_with_stats("1991年，张三认识李四。");
        assert!(!frames.is_empty(), "应至少抽取一条帧：{:?}", stats);
        let frame = frames
            .iter()
            .find(|frame| frame.predicate == "认识")
            .expect("认识帧");
        assert_eq!(frame.subject, "张三");
        assert_eq!(frame.object, "李四");
        assert_eq!(
            frame.temporal.as_ref().and_then(|t| t.start_year),
            Some(1991)
        );
        assert_eq!(frame.evidence.sentence, "1991年，张三认识李四");
        assert!(frame.evidence.subject_span.is_some());
        assert_eq!(stats.temporal_frames, 1);

        let (negative, _) = FrameExtractor::new().extract_with_stats("张三不认识赵六。");
        let frame = negative
            .iter()
            .find(|frame| frame.predicate == "认识")
            .expect("否定帧");
        assert_eq!(frame.subject, "张三");
        assert_eq!(frame.polarity, Polarity::Negative);
    }

    #[test]
    fn gate_supports_direct_inferred_and_conflicting_frames() {
        let gate = gate(&[
            ("张三", "师傅", "李四"),
            ("李四", "师傅", "王五"),
            ("张三", "年龄", "18"),
        ]);

        let direct = gate.evaluate(&RelationFrame::new("张三", "师傅", "李四"));
        assert_eq!(direct.verdict, GateVerdict::DirectSupport);

        let inferred = gate.evaluate(&RelationFrame::new("张三", "师傅", "王五"));
        assert_eq!(inferred.verdict, GateVerdict::InferredSupport);
        assert_eq!(
            inferred.support_path,
            vec![
                FrameEdge::new("张三", "师傅", "李四"),
                FrameEdge::new("李四", "师傅", "王五"),
            ],
            "证明路径应保留最短传递证明，而不是互逆边造成的循环引用"
        );

        let conflict = gate.evaluate(&RelationFrame::new("张三", "年龄", "20"));
        assert_eq!(conflict.verdict, GateVerdict::Conflict);
        assert!(conflict
            .findings
            .iter()
            .any(|finding| finding.kind == GateFindingKind::FunctionalConflict));

        let negative = gate.evaluate(
            &RelationFrame::new("张三", "师傅", "李四").with_polarity(Polarity::Negative),
        );
        assert_eq!(negative.verdict, GateVerdict::Conflict);
        assert!(negative
            .findings
            .iter()
            .any(|finding| finding.kind == GateFindingKind::NegativeConflict));
    }

    #[test]
    fn inverse_aliases_are_normalized_before_gating() {
        let direct = registry().resolve("师傅");
        assert!(!direct.inverted, "师傅应为正向槽位：{:?}", direct);
        let inverse = registry().resolve("徒弟");
        assert!(inverse.inverted, "徒弟应为逆向别名：{:?}", inverse);
        let gate = gate(&[("李四", "师傅", "张三")]);
        let report = gate.evaluate(&RelationFrame::new("张三", "徒弟", "李四"));
        assert!(report.inverted);
        assert_eq!(report.normalized.predicate, "师傅");
        assert_eq!(report.verdict, GateVerdict::DirectSupport);
    }

    #[test]
    fn entity_properties_participate_in_direct_support_and_conflicts() {
        use std::collections::HashMap;

        let registry = registry();
        let laws = LawBook::from_registry(&registry);
        let mut properties = HashMap::new();
        properties.insert("年龄".to_string(), "25".to_string());
        properties.insert(
            "师傅".to_string(),
            "http://example.org/kg/entity/李四".to_string(),
        );
        let facts = vec![PrologFact::Entity {
            name: "张三".to_string(),
            entity_type: "人物".to_string(),
            properties,
        }];
        let gate = FrameGate::from_facts(&facts, registry, TypeSystem::new(), laws);
        assert_eq!(gate.fact_count(), 1, "关系 IRI 不应作为属性边重复计入");

        let direct = gate.evaluate(&RelationFrame::new("张三", "年龄", "25"));
        assert_eq!(direct.verdict, GateVerdict::DirectSupport);

        let conflict = gate.evaluate(&RelationFrame::new("张三", "年龄", "30"));
        assert_eq!(conflict.verdict, GateVerdict::Conflict);
        assert!(conflict
            .findings
            .iter()
            .any(|finding| finding.kind == GateFindingKind::FunctionalConflict));
    }

    #[test]
    fn diagnostics_are_deterministic_and_narrated() {
        let gate = gate(&[("张三", "师傅", "李四")]);
        let frames = vec![
            RelationFrame::new("张三", "师傅", "李四"),
            RelationFrame::new("张三", "师傅", "王五"),
        ];
        let first = diagnose(&frames, &gate, 2);
        let second = diagnose(&frames, &gate, 2);
        assert_eq!(first, second);
        assert_eq!(first.direct_support, 1);
        assert_eq!(first.unsupported, 1);
        assert_eq!(first.samples.len(), 2);
        let narration = narrate_frame(&frames[0], &first.samples[0].report);
        assert!(narration.contains("直接支持"));
        assert!(narration.contains("之间存在「师傅」关系"));
        let age = RelationFrame::new("张三", "年龄", "18");
        let age_report = gate.evaluate(&age);
        assert!(narrate_frame(&age, &age_report).contains("张三的年龄是18"));
        assert!(!diagnostics_digest(&first).is_empty());
    }
}

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

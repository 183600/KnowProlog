//! 批次冲突回合（Batch Conflict Turn, BCT）：把用户一次说的一段话编译成
//! 一批 claim，按槽检测冲突、按冲突连通分量提问、按一次事务提交。
//!
//! 设计边界：
//!   - 只做符号规则与显式提示词裁决，不引入概率权重；
//!   - 位置只在解释「前者/后者」与展示来源时使用，不参与真值裁决；
//!   - 单条 claim 仍走既有逐条路径，本模块只在多 claim 或显式批处理时启用；
//!   - 解析失败诚实报错，不猜测缺失的主语/谓词。

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::chinese_reasoning_nlu::{self, LessonKind};
use crate::slot_ledger::{AssertionStrength, SlotKey};

/// 明确的更正提示词。出现在句子任意位置都视为「这次是在改口」。
pub const CORRECTION_CUES: &[&str] = &[
    "其实", "不过", "更正", "纠正", "修正", "改口", "不对", "应该说", "改为", "改成", "而是",
];

/// 批处理提交模式。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BatchMode {
    /// 先提交无冲突部分，冲突项隔离并提问。
    Progressive,
    /// 只要还有未决冲突就零写入；裁决后再整批提交。
    Atomic,
}

impl BatchMode {
    pub fn label_zh(self) -> &'static str {
        match self {
            BatchMode::Progressive => "渐进",
            BatchMode::Atomic => "原子",
        }
    }
}

/// 一条从自然语言中抽出的 claim。`span` 是原文中的字节区间，仅用于解释。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchClaim {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub strength: AssertionStrength,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub correction_cue: Option<String>,
    /// 稳定来源标签，例如「第 2 句」。
    pub source: String,
    /// 解析前的原句（去前缀后）。
    pub raw: String,
    pub span: (usize, usize),
}

impl BatchClaim {
    pub fn key(&self) -> SlotKey {
        SlotKey::new(&self.subject, &self.predicate)
    }

    /// 显式序数：更正 > 直陈 > 传闻。不要依赖 `AssertionStrength` 的枚举顺序。
    pub fn strength_rank(&self) -> u8 {
        match self.strength {
            AssertionStrength::Correction => 3,
            AssertionStrength::Direct => 2,
            AssertionStrength::Hearsay => 1,
        }
    }

    pub fn render_zh(&self) -> String {
        format!("{} = {}", self.key().label_zh(), self.object)
    }

    /// 用调用方的规范名归一主语/谓词；对象保持原词面。
    pub fn canonicalize_with<F>(&mut self, mut canonical: F)
    where
        F: FnMut(&str, &str) -> (String, String),
    {
        let (subject, predicate) = canonical(&self.subject, &self.predicate);
        self.subject = subject;
        self.predicate = predicate;
    }
}

/// 批解析结果：成功 claim 与逐段失败原因。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BatchParseOutcome {
    pub claims: Vec<BatchClaim>,
    pub errors: Vec<String>,
}

/// 冲突类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BatchConflictKind {
    /// 功能槽在同一批里出现多个不同取值，且没有显式更正。
    FunctionalMultiValue,
    /// 功能槽的批次取值与知识图谱基值不同，且没有显式更正。
    GraphBaseConflict,
    /// 多个显式更正给出不同取值，无法用序数决出唯一赢家。
    CorrectionTie,
}

impl BatchConflictKind {
    pub fn label_zh(self) -> &'static str {
        match self {
            BatchConflictKind::FunctionalMultiValue => "功能槽多值",
            BatchConflictKind::GraphBaseConflict => "与图谱基值冲突",
            BatchConflictKind::CorrectionTie => "更正之间仍并列",
        }
    }
}

/// 一个冲突连通分量：同一规范槽上需要一起裁决的 claim 集合。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchConflict {
    pub key: SlotKey,
    pub kind: BatchConflictKind,
    /// claim 在批次中的下标（保持首次出现顺序）。
    pub claims: Vec<usize>,
    pub base_values: Vec<String>,
    /// 候选取值（claim 首次出现顺序，随后追加未重复的基值）。
    pub values: Vec<String>,
}

/// 冲突表中的一个选项。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchOption {
    /// 选择该选项后要提交的取值；`None` 表示保留图谱基值、不写会话事实。
    pub value: Option<String>,
    pub label: char,
    pub description: String,
    /// 支持该取值的 claim 下标。
    pub claims: Vec<usize>,
    pub base: bool,
}

/// 一张待裁决问题（一个冲突分量一张，多个分量合并成冲突表）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchQuestion {
    pub conflict: usize,
    pub key: SlotKey,
    pub options: Vec<BatchOption>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_option: Option<usize>,
    pub prompt: String,
}

impl BatchQuestion {
    pub fn option_by_label(&self, label: char) -> Option<usize> {
        self.options.iter().position(|option| option.label == label)
    }

    pub fn option_by_value(&self, value: &str) -> Option<usize> {
        self.options
            .iter()
            .position(|option| option.value.as_deref() == Some(value))
    }

    pub fn render_zh(&self, number: usize) -> String {
        let mut lines = vec![format!("{}. {}", number, self.prompt)];
        for option in &self.options {
            let suggestion = if self.options.iter().position(|item| item == option)
                == self.default_option
            {
                "（建议）"
            } else {
                ""
            };
            lines.push(format!(
                "   {} {}{}",
                option.label, option.description, suggestion
            ));
        }
        lines.join("\n")
    }
}

/// 被隔离的 claim：未写入，但保留原因与关联问题。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchIsolation {
    pub claim: usize,
    pub reason: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub question: Option<usize>,
}

/// 批裁决计划。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchPlan {
    pub mode: BatchMode,
    pub claims: Vec<BatchClaim>,
    /// 可直接提交的 claim 下标。
    pub accepted: Vec<usize>,
    /// 因原子模式被推迟的无冲突 claim。
    pub deferred: Vec<usize>,
    /// 仅记证据、不进入回答的 claim。
    pub evidence: Vec<usize>,
    pub isolated: Vec<BatchIsolation>,
    pub conflicts: Vec<BatchConflict>,
    pub questions: Vec<BatchQuestion>,
}

impl BatchPlan {
    pub fn has_questions(&self) -> bool {
        !self.questions.is_empty()
    }

    pub fn render_zh(&self) -> String {
        let mut lines = vec![format!(
            "批次裁决（{}模式）：{} 条候选，采用 {} 条，隔离 {} 条，待确认 {} 处。",
            self.mode.label_zh(),
            self.claims.len(),
            self.accepted.len() + self.deferred.len(),
            self.isolated.len(),
            self.questions.len()
        )];
        for &index in &self.accepted {
            if let Some(claim) = self.claims.get(index) {
                lines.push(format!("  ✓ {}（{}）", claim.render_zh(), claim.source));
            }
        }
        for &index in &self.deferred {
            if let Some(claim) = self.claims.get(index) {
                lines.push(format!(
                    "  … {}（{}；原子模式暂缓提交）",
                    claim.render_zh(),
                    claim.source
                ));
            }
        }
        for isolation in &self.isolated {
            if let Some(claim) = self.claims.get(isolation.claim) {
                lines.push(format!(
                    "  ! {}（{}）：{}",
                    claim.render_zh(),
                    claim.source,
                    isolation.reason
                ));
            }
        }
        for &index in &self.evidence {
            if let Some(claim) = self.claims.get(index) {
                lines.push(format!(
                    "  · {}（{}；仅记证据，不进入回答）",
                    claim.render_zh(),
                    claim.source
                ));
            }
        }
        if !self.questions.is_empty() {
            lines.push("待确认（可一次回答，例如「1B，2保留30，其余同意」）：".to_string());
            for (index, question) in self.questions.iter().enumerate() {
                lines.push(question.render_zh(index + 1));
            }
        }
        lines.join("\n")
    }
}

/// 跨轮挂起的批次裁决：保留原 claim、问题与已提交/暂缓下标。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchPendingTurn {
    pub label: String,
    pub mode: BatchMode,
    pub claims: Vec<BatchClaim>,
    pub questions: Vec<BatchQuestion>,
    /// 已经提交过的 claim 下标（渐进模式下为无冲突项）。
    #[serde(default)]
    pub committed: Vec<usize>,
    /// 原子模式下暂缓提交的无冲突项。
    #[serde(default)]
    pub deferred: Vec<usize>,
    /// 原子模式下暂缓写入的传闻证据。
    #[serde(default)]
    pub evidence: Vec<usize>,
}

impl BatchPendingTurn {
    pub fn render_zh(&self) -> String {
        let mut lines = vec![format!(
            "待裁决批次（{}模式）：{} 条 claim，{} 处冲突。",
            self.mode.label_zh(),
            self.claims.len(),
            self.questions.len()
        )];
        if !self.committed.is_empty() {
            lines.push(format!(
                "已提交 {} 条无冲突 claim；裁决只影响剩余冲突项。",
                self.committed.len()
            ));
        }
        if !self.deferred.is_empty() {
            lines.push(format!(
                "暂缓提交 {} 条无冲突 claim（原子模式，裁决后一起写入）。",
                self.deferred.len()
            ));
        }
        if !self.evidence.is_empty() {
            lines.push(format!(
                "暂缓记录 {} 条传闻证据（裁决后写入证据区）。",
                self.evidence.len()
            ));
        }
        for (index, question) in self.questions.iter().enumerate() {
            lines.push(question.render_zh(index + 1));
        }
        lines.push(
            "回复示例：批答：1B，2保留原值；或「批答：全部同意」「批答：取消」。".to_string(),
        );
        lines.join("\n")
    }
}

/// 批裁决输入选项。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BatchOptions {
    pub functional_predicates: BTreeSet<String>,
    pub base_values: BTreeMap<SlotKey, Vec<String>>,
    pub mode: BatchMode,
}

impl Default for BatchMode {
    fn default() -> Self {
        BatchMode::Progressive
    }
}

impl BatchOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_mode(mut self, mode: BatchMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn declare_functional<I, S>(mut self, predicates: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.functional_predicates
            .extend(predicates.into_iter().map(Into::into));
        self
    }

    pub fn with_base_values(mut self, key: SlotKey, values: Vec<String>) -> Self {
        self.base_values.insert(key, values);
        self
    }
}

/// 把一段中文文本保守切分成候选 claim。
///
/// 强边界（。！？；;换行）必切；逗号只有在切完后每一段都能解析成 claim 时
/// 才切，避免把「张三，30岁」切坏。
pub fn parse_batch_claims(text: &str) -> BatchParseOutcome {
    let mut outcome = BatchParseOutcome::default();
    if text.contains('？') || text.contains('?') {
        outcome
            .errors
            .push("疑问句不能作为批次事实；请使用查询路径。".to_string());
        return outcome;
    }
    let mut previous: Option<BatchClaim> = None;
    let mut ordinal = 0usize;
    for (segment, start, end) in strong_segments(text) {
        let pieces = split_commas_if_all_parse(&segment, previous.as_ref());
        for piece in pieces {
            ordinal += 1;
            match parse_segment(&piece, previous.as_ref(), ordinal, start, end) {
                Ok(claim) => {
                    previous = Some(claim.clone());
                    outcome.claims.push(claim);
                }
                Err(reason) => outcome.errors.push(reason),
            }
        }
    }
    outcome
}

fn strong_segments(text: &str) -> Vec<(String, usize, usize)> {
    let mut segments = Vec::new();
    let mut start = 0usize;
    for (index, ch) in text.char_indices() {
        if matches!(ch, '。' | '！' | '!' | '？' | '?' | ';' | '；' | '\n' | '\r') {
            flush_segment(text, start, index, &mut segments);
            start = index + ch.len_utf8();
        }
    }
    flush_segment(text, start, text.len(), &mut segments);
    segments
}

fn flush_segment(text: &str, start: usize, end: usize, segments: &mut Vec<(String, usize, usize)>) {
    let raw = &text[start..end];
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return;
    }
    let offset = raw.find(trimmed).unwrap_or(0);
    segments.push((
        trimmed.to_string(),
        start + offset,
        start + offset + trimmed.len(),
    ));
}

fn split_commas_if_all_parse(
    segment: &str,
    previous: Option<&BatchClaim>,
) -> Vec<String> {
    if contains_conditional_marker(segment) {
        return vec![segment.trim().to_string()];
    }
    let pieces: Vec<String> = segment
        .split(['，', ',', '、'])
        .map(|piece| piece.trim().to_string())
        .filter(|piece| !piece.is_empty())
        .collect();
    if pieces.len() <= 1 {
        return vec![segment.trim().to_string()];
    }
    let mut context = previous.cloned();
    for piece in &pieces {
        match parse_segment(piece, context.as_ref(), 0, 0, 0) {
            Ok(claim) => context = Some(claim),
            Err(_) => return vec![segment.trim().to_string()],
        }
    }
    pieces
}

fn contains_conditional_marker(text: &str) -> bool {
    ["如果", "假如", "只要", "一旦", "若", "因为", "由于", "既然", "那么", "则", "所以"]
        .iter()
        .any(|marker| text.contains(marker))
}

fn parse_segment(
    segment: &str,
    previous: Option<&BatchClaim>,
    ordinal: usize,
    start: usize,
    end: usize,
) -> Result<BatchClaim, String> {
    let raw = segment.trim();
    if raw.is_empty() {
        return Err("空句子".to_string());
    }
    let raw = strip_list_marker(raw);
    if raw.contains("不是") || raw.contains("没有") {
        return Err(format!("「{}」是否定句，不能作为批次肯定事实", raw));
    }
    let (mut strength, content) = AssertionStrength::classify(raw);
    let (cue, mut content) = strip_correction_cue(content);
    if cue.is_some() && strength == AssertionStrength::Direct {
        strength = AssertionStrength::Correction;
    }
    content = content.trim().to_string();
    if content.is_empty() {
        return Err(format!("「{}」只有更正提示，没有可裁决内容", raw));
    }

    // 省略宾语/谓词：更正片段「是31」沿用上一条 claim 的槽。
    if let Some(rest) = content.strip_prefix('是') {
        if let Some(previous) = previous {
            let object = rest.trim();
            if !object.is_empty() && !object.contains('是') {
                return Ok(BatchClaim {
                    subject: previous.subject.clone(),
                    predicate: previous.predicate.clone(),
                    object: object.to_string(),
                    strength,
                    correction_cue: cue,
                    source: format!("第 {} 句", ordinal),
                    raw: raw.to_string(),
                    span: (start, end),
                });
            }
        }
    }

    // 代词主语用批次内前一条 claim 补全。
    let mut content = content;
    if let Some(previous) = previous {
        for pronoun in ["他们", "她们", "它们", "他", "她", "它", "这个人", "该人"] {
            if let Some(rest) = content.strip_prefix(pronoun) {
                content = format!("{}{}", previous.subject, rest);
                break;
            }
        }
    }

    // 年龄省略式：X30岁 / 30岁 / 今年30岁；可回填前一条的主语。
    if let Some((subject, object)) = parse_age_claim(&content) {
        let subject = if subject.is_empty() {
            previous
                .map(|claim| claim.subject.clone())
                .ok_or_else(|| format!("「{}」缺主语，无法确定年龄属于谁", raw))?
        } else {
            subject
        };
        return Ok(BatchClaim {
            subject,
            predicate: "年龄".to_string(),
            object,
            strength,
            correction_cue: cue,
            source: format!("第 {} 句", ordinal),
            raw: raw.to_string(),
            span: (start, end),
        });
    }

    let lesson = chinese_reasoning_nlu::parse_lesson(&content)?;
    if lesson.kind != LessonKind::DeterministicFact {
        return Err(format!(
            "「{}」不是确定事实（{}），不能进入批次裁决",
            raw,
            lesson.kind.kind_label()
        ));
    }
    if !lesson.head.is_ground() {
        return Err(format!("「{}」含占位符，不能作为具体事实", raw));
    }
    let terms = &lesson.head.terms;
    let [subject, object] = terms.as_slice() else {
        return Err(format!(
            "「{}」不是二元事实，无法按槽裁决；请改写为「S的P是O」",
            raw
        ));
    };
    Ok(BatchClaim {
        subject: subject.name().to_string(),
        predicate: lesson.head.predicate.to_string(),
        object: object.name().to_string(),
        strength,
        correction_cue: cue,
        source: format!("第 {} 句", ordinal),
        raw: raw.to_string(),
        span: (start, end),
    })
}

fn strip_list_marker(text: &str) -> &str {
    let trimmed = text.trim_start_matches(|ch: char| {
        ch.is_whitespace() || matches!(ch, '-' | '•' | '·' | '*' | '①' | '②' | '③' | '④' | '⑤')
    });
    let digits = trimmed
        .char_indices()
        .take_while(|(_, ch)| ch.is_ascii_digit())
        .last()
        .map(|(index, ch)| index + ch.len_utf8())
        .unwrap_or(0);
    if digits > 0 {
        let rest = trimmed[digits..].trim_start();
        if let Some(rest) = rest
            .strip_prefix('.')
            .or_else(|| rest.strip_prefix('、'))
            .or_else(|| rest.strip_prefix(')'))
            .or_else(|| rest.strip_prefix('）'))
        {
            return rest.trim();
        }
    }
    trimmed
}

fn strip_correction_cue(text: &str) -> (Option<String>, String) {
    let mut best: Option<(usize, &str)> = None;
    for cue in CORRECTION_CUES {
        if let Some(position) = text.find(cue) {
            if best
                .map(|(best_position, _)| position < best_position)
                .unwrap_or(true)
            {
                best = Some((position, cue));
            }
        }
    }
    let Some((position, cue)) = best else {
        return (None, text.to_string());
    };
    let mut cleaned = String::with_capacity(text.len());
    cleaned.push_str(&text[..position]);
    cleaned.push_str(&text[position + cue.len()..]);
    (
        Some(cue.to_string()),
        cleaned
            .trim()
            .trim_matches(|ch: char| matches!(ch, '，' | ',' | '：' | ':' | ' '))
            .to_string(),
    )
}

fn parse_age_claim(text: &str) -> Option<(String, String)> {
    let text = text
        .trim()
        .trim_end_matches(['了', '。', '！', '!'])
        .trim();
    let body = text
        .strip_suffix("周岁")
        .or_else(|| text.strip_suffix('岁'))?;
    let body = body.trim();
    let ascii_start = body
        .char_indices()
        .rev()
        .take_while(|(_, ch)| ch.is_ascii_digit())
        .last()
        .map(|(index, _)| index);
    let number_start = match ascii_start {
        Some(index) => index,
        None => body
            .char_indices()
            .rev()
            .take_while(|(_, ch)| is_number_char(*ch))
            .last()
            .map(|(index, _)| index)?,
    };
    let number = body[number_start..].trim();
    if number.is_empty() {
        return None;
    }
    let mut subject = body[..number_start]
        .trim()
        .trim_end_matches(['，', ',', ' '])
        .trim()
        .to_string();
    if subject.starts_with("今年") {
        subject = subject["今年".len()..].trim().to_string();
    }
    if subject.ends_with("今年") {
        subject = subject[..subject.len() - "今年".len()].trim().to_string();
    }
    let object = chinese_reasoning_nlu::parse_chinese_number_u64(number)
        .map(|value| value.to_string())
        .unwrap_or_else(|| number.to_string());
    Some((subject, object))
}

fn is_number_char(ch: char) -> bool {
    ch.is_ascii_digit()
        || matches!(
            ch,
            '零' | '〇'
                | '一'
                | '二'
                | '两'
                | '三'
                | '四'
                | '五'
                | '六'
                | '七'
                | '八'
                | '九'
                | '十'
                | '百'
                | '千'
                | '万'
                | '亿'
        )
}

trait LessonKindLabel {
    fn kind_label(&self) -> &'static str;
}

impl LessonKindLabel for LessonKind {
    fn kind_label(&self) -> &'static str {
        match self {
            LessonKind::DeterministicFact => "确定事实",
            LessonKind::ProbabilisticFact => "概率事实",
            LessonKind::ConditionalRule => "条件规则",
            LessonKind::CausalRule => "因果规则",
            LessonKind::DefaultRule => "默认规则",
        }
    }
}

/// 对一个批次做确定性裁决。位置不参与胜负，只在选项说明里展示来源。
pub fn plan_batch(claims: Vec<BatchClaim>, options: &BatchOptions) -> BatchPlan {
    let mut plan = BatchPlan {
        mode: options.mode,
        claims: claims.clone(),
        accepted: Vec::new(),
        deferred: Vec::new(),
        evidence: Vec::new(),
        isolated: Vec::new(),
        conflicts: Vec::new(),
        questions: Vec::new(),
    };
    let mut by_slot: BTreeMap<SlotKey, Vec<usize>> = BTreeMap::new();
    for (index, claim) in claims.iter().enumerate() {
        by_slot.entry(claim.key()).or_default().push(index);
    }

    let mut accepted: Vec<usize> = Vec::new();
    let mut isolated: Vec<BatchIsolation> = Vec::new();
    let mut evidence: Vec<usize> = Vec::new();
    let mut conflicts: Vec<BatchConflict> = Vec::new();
    let mut questions: Vec<BatchQuestion> = Vec::new();

    for (key, indices) in by_slot {
        let functional = options.functional_predicates.contains(&key.predicate);
        let base_values = options
            .base_values
            .get(&key)
            .cloned()
            .unwrap_or_default();
        let hearsay: Vec<usize> = indices
            .iter()
            .copied()
            .filter(|index| claims[*index].strength == AssertionStrength::Hearsay)
            .collect();
        evidence.extend(hearsay);
        let factual: Vec<usize> = indices
            .iter()
            .copied()
            .filter(|index| claims[*index].strength != AssertionStrength::Hearsay)
            .collect();
        if factual.is_empty() {
            continue;
        }

        if !functional {
            let mut by_value: BTreeMap<String, Vec<usize>> = BTreeMap::new();
            for index in factual {
                by_value
                    .entry(claims[index].object.clone())
                    .or_default()
                    .push(index);
            }
            for value_indices in by_value.into_values() {
                accepted.push(best_claim(&claims, &value_indices));
            }
            continue;
        }

        let mut by_value: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for index in &factual {
            by_value
                .entry(claims[*index].object.clone())
                .or_default()
                .push(*index);
        }
        let mut values: Vec<String> = Vec::new();
        for base in &base_values {
            if !values.iter().any(|value| value == base) {
                values.push(base.clone());
            }
        }
        for index in &factual {
            let value = &claims[*index].object;
            if !values.iter().any(|candidate| candidate == value) {
                values.push(value.clone());
            }
        }
        let has_correction = factual
            .iter()
            .any(|index| claims[*index].strength == AssertionStrength::Correction);

        if by_value.len() == 1 {
            let (value, value_indices) = by_value.iter().next().expect("非空");
            let base_differs = !base_values.is_empty()
                && !(base_values.len() == 1 && base_values[0] == *value);
            if base_differs && !has_correction {
                let conflict_index = conflicts.len();
                let conflict = BatchConflict {
                    key: key.clone(),
                    kind: BatchConflictKind::GraphBaseConflict,
                    claims: factual.clone(),
                    base_values: base_values.clone(),
                    values: values.clone(),
                };
                let question = build_question(&claims, conflict_index, &conflict);
                isolated.extend(factual.iter().map(|index| BatchIsolation {
                    claim: *index,
                    reason: format!("{}：需先裁决", BatchConflictKind::GraphBaseConflict.label_zh()),
                    question: Some(conflict_index),
                }));
                conflicts.push(conflict);
                questions.push(question);
            } else {
                accepted.push(best_claim(&claims, value_indices));
            }
            continue;
        }

        let top_rank = factual
            .iter()
            .map(|index| claims[*index].strength_rank())
            .max()
            .unwrap_or(0);
        let top: Vec<usize> = factual
            .iter()
            .copied()
            .filter(|index| claims[*index].strength_rank() == top_rank)
            .collect();
        let top_values: BTreeSet<String> = top
            .iter()
            .map(|index| claims[*index].object.clone())
            .collect();
        if top_rank == 3 && top_values.len() == 1 {
            let value = top_values.iter().next().expect("非空");
            let value_indices: Vec<usize> = top
                .iter()
                .copied()
                .filter(|index| &claims[*index].object == value)
                .collect();
            accepted.push(best_claim(&claims, &value_indices));
            for index in &factual {
                if &claims[*index].object != value {
                    isolated.push(BatchIsolation {
                        claim: *index,
                        reason: format!("被显式更正「{}」覆盖", value),
                        question: None,
                    });
                }
            }
            continue;
        }

        let kind = if top_rank == 3 {
            BatchConflictKind::CorrectionTie
        } else {
            BatchConflictKind::FunctionalMultiValue
        };
        let conflict_index = conflicts.len();
        let conflict = BatchConflict {
            key: key.clone(),
            kind,
            claims: factual.clone(),
            base_values: base_values.clone(),
            values: values.clone(),
        };
        let question = build_question(&claims, conflict_index, &conflict);
        isolated.extend(factual.iter().map(|index| BatchIsolation {
            claim: *index,
            reason: format!("{}，需一次裁决", kind.label_zh()),
            question: Some(conflict_index),
        }));
        conflicts.push(conflict);
        questions.push(question);
    }

    plan.accepted = accepted;
    plan.evidence = evidence;
    plan.isolated = isolated;
    plan.conflicts = conflicts;
    plan.questions = questions;
    if plan.mode == BatchMode::Atomic && plan.has_questions() {
        plan.deferred = std::mem::take(&mut plan.accepted);
    }
    plan
}

fn best_claim(claims: &[BatchClaim], indices: &[usize]) -> usize {
    indices
        .iter()
        .copied()
        .max_by(|left, right| {
            claims[*left]
                .strength_rank()
                .cmp(&claims[*right].strength_rank())
                .then_with(|| claims[*right].source.cmp(&claims[*left].source))
        })
        .unwrap_or(0)
}

fn build_question(
    claims: &[BatchClaim],
    conflict_index: usize,
    conflict: &BatchConflict,
) -> BatchQuestion {
    let mut options: Vec<BatchOption> = Vec::new();
    for value in &conflict.values {
        let claim_indices: Vec<usize> = conflict
            .claims
            .iter()
            .copied()
            .filter(|index| &claims[*index].object == value)
            .collect();
        let base = conflict.base_values.iter().any(|item| item == value);
        let label = option_label(options.len());
        let description = if base {
            format!("保留知识图谱当前值「{}」", value)
        } else {
            let sources: Vec<String> = claim_indices
                .iter()
                .map(|index| {
                    let claim = &claims[*index];
                    if claim.correction_cue.is_some() {
                        format!("{}，{}更正", claim.source, claim.correction_cue.as_deref().unwrap_or(""))
                    } else {
                        format!("{}，{}", claim.source, claim.strength.label_zh())
                    }
                })
                .collect();
            format!("采用「{}」（{}）", value, sources.join("；"))
        };
        options.push(BatchOption {
            value: Some(value.clone()),
            label,
            description,
            claims: claim_indices,
            base,
        });
    }
    if conflict.base_values.is_empty() {
        // 没有图谱基值时，允许用户明确「这次先不写」。
        options.push(BatchOption {
            value: None,
            label: option_label(options.len()),
            description: "本次都不写入，保留待确认".to_string(),
            claims: Vec::new(),
            base: false,
        });
    }
    let default_option = if conflict.base_values.len() == 1 {
        conflict
            .base_values
            .first()
            .and_then(|base| options.iter().position(|option| option.value.as_deref() == Some(base.as_str())))
    } else {
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for index in &conflict.claims {
            *counts.entry(claims[*index].object.clone()).or_default() += 1;
        }
        let mut best: Option<(String, usize)> = None;
        let mut tied = false;
        for (value, count) in counts {
            match &best {
                Some((_, best_count)) if count < *best_count => {}
                Some((_, best_count)) if count == *best_count => tied = true,
                _ => {
                    best = Some((value, count));
                    tied = false;
                }
            }
        }
        if tied {
            None
        } else {
            best.and_then(|(value, _)| {
                options
                    .iter()
                    .position(|option| option.value.as_deref() == Some(value.as_str()))
            })
        }
    };
    let prompt = format!(
        "{}：{}。请选择保留哪一个，或回复「取消」。",
        conflict.key.label_zh(),
        if conflict.base_values.is_empty() {
            conflict
                .values
                .iter()
                .map(|value| format!("「{}」", value))
                .collect::<Vec<_>>()
                .join(" / ")
        } else {
            format!(
                "批次给出 {}；知识图谱当前 {}",
                conflict
                    .values
                    .iter()
                    .filter(|value| !conflict.base_values.contains(value))
                    .map(|value| format!("「{}」", value))
                    .collect::<Vec<_>>()
                    .join(" / "),
                conflict
                    .base_values
                    .iter()
                    .map(|value| format!("「{}」", value))
                    .collect::<Vec<_>>()
                    .join(" / ")
            )
        }
    );
    BatchQuestion {
        conflict: conflict_index,
        key: conflict.key.clone(),
        options,
        default_option,
        prompt,
    }
}

fn option_label(index: usize) -> char {
    if index < 26 {
        (b'A' + index as u8) as char
    } else {
        '?'
    }
}

/// 用户对冲突表的回答。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BatchAnswerOutcome {
    /// 每个问题选中的选项下标；`None` 表示仍未决。
    pub selections: Vec<Option<usize>>,
    pub cancelled: bool,
    pub notes: Vec<String>,
}

impl BatchAnswerOutcome {
    pub fn unresolved(&self) -> Vec<usize> {
        self.selections
            .iter()
            .enumerate()
            .filter_map(|(index, selection)| selection.is_none().then_some(index))
            .collect()
    }

    pub fn is_complete(&self) -> bool {
        !self.cancelled && self.selections.iter().all(Option::is_some)
    }
}

/// 解析批答案：支持「1B，2保留30，其余同意」「都按后者」「全部同意」「取消」。
pub fn parse_batch_answer(answer: &str, questions: &[BatchQuestion]) -> BatchAnswerOutcome {
    let normalized = normalize_answer(answer);
    let mut outcome = BatchAnswerOutcome {
        selections: vec![None; questions.len()],
        cancelled: false,
        notes: Vec::new(),
    };
    if normalized.is_empty() {
        outcome.notes.push("没有收到裁决内容。".to_string());
        return outcome;
    }
    if ["取消", "放弃", "不裁决", "cancel"].iter().any(|cue| normalized.contains(cue)) {
        outcome.cancelled = true;
        return outcome;
    }
    if ["全部同意", "都同意", "全同意", "同意全部", "按建议", "都按建议"]
        .iter()
        .any(|cue| normalized.contains(cue))
    {
        apply_defaults(questions, &mut outcome, 0..questions.len());
        return outcome;
    }
    if ["都按后者", "全部按后者", "按后者", "都选后者"].iter().any(|cue| normalized.contains(cue)) {
        for index in 0..questions.len() {
            if let Some(last) = questions[index].options.len().checked_sub(1) {
                outcome.selections[index] = Some(last);
            }
        }
        return outcome;
    }
    if ["都按前者", "全部按前者", "按前者", "都选前者"].iter().any(|cue| normalized.contains(cue)) {
        for index in 0..questions.len() {
            if !questions[index].options.is_empty() {
                outcome.selections[index] = Some(0);
            }
        }
        return outcome;
    }

    let mut rest_defaults = false;
    let mut rest_last = false;
    for token in normalized.split(['，', ',', '；', ';', '、', ' ']).filter(|token| !token.is_empty()) {
        if token.contains("其余") {
            if token.contains("后者") {
                rest_last = true;
            } else {
                rest_defaults = true;
            }
            continue;
        }
        match parse_answer_token(token, questions) {
            Ok((question, option)) => {
                if let Some(slot) = outcome.selections.get_mut(question) {
                    *slot = Some(option);
                }
            }
            Err(note) => outcome.notes.push(note),
        }
    }
    if rest_defaults {
        let remaining: Vec<usize> = outcome
            .unresolved()
            .into_iter()
            .collect();
        apply_defaults(questions, &mut outcome, remaining);
    } else if rest_last {
        for index in outcome.unresolved() {
            if let Some(last) = questions[index].options.len().checked_sub(1) {
                outcome.selections[index] = Some(last);
            }
        }
    }
    outcome
}

fn apply_defaults(
    questions: &[BatchQuestion],
    outcome: &mut BatchAnswerOutcome,
    indices: impl IntoIterator<Item = usize>,
) {
    for index in indices {
        match questions[index].default_option {
            Some(option) => outcome.selections[index] = Some(option),
            None => outcome
                .notes
                .push(format!("第 {} 处没有默认建议，请明确选择。", index + 1)),
        }
    }
}

fn parse_answer_token(
    token: &str,
    questions: &[BatchQuestion],
) -> Result<(usize, usize), String> {
    let token = token.trim();
    if token.is_empty() {
        return Err("空裁决项。".to_string());
    }
    let token = token.strip_prefix("第").unwrap_or(token);
    let digits_end = token
        .char_indices()
        .take_while(|(_, ch)| ch.is_ascii_digit())
        .last()
        .map(|(index, ch)| index + ch.len_utf8())
        .unwrap_or(0);
    if digits_end == 0 {
        if token.len() == 1 && token.chars().all(|ch| ch.is_ascii_alphabetic()) {
            let label = token.chars().next().unwrap().to_ascii_uppercase();
            for (index, question) in questions.iter().enumerate() {
                if let Some(option) = question.option_by_label(label) {
                    return Ok((index, option));
                }
            }
        }
        return Err(format!("无法识别裁决项「{}」。", token));
    }
    let number: usize = token[..digits_end]
        .parse()
        .map_err(|_| format!("题号「{}」无效。", &token[..digits_end]))?;
    if number == 0 || number > questions.len() {
        return Err(format!("题号 {} 超出范围。", number));
    }
    let question_index = number - 1;
    let rest = token[digits_end..]
        .trim_start_matches(|ch: char| matches!(ch, ':' | '.' | '、' | ')' | '）' | '-'));
    let mut rest = rest;
    for prefix in ["第", "题", "选", "取", "按", "保留", ":"] {
        if let Some(stripped) = rest.strip_prefix(prefix) {
            rest = stripped.trim_start();
        }
    }
    let question = &questions[question_index];
    if let Some(first) = rest.chars().next() {
        if first.is_ascii_alphabetic() {
            let label = first.to_ascii_uppercase();
            if let Some(option) = question.option_by_label(label) {
                return Ok((question_index, option));
            }
            return Err(format!(
                "第 {} 处没有选项 {}。",
                number, label
            ));
        }
    }
    let value = rest.trim_start_matches(|ch: char| matches!(ch, ':' | ' '));
    if value.is_empty() {
        return Err(format!("第 {} 处缺少选择。", number));
    }
    if matches!(value, "保留原值" | "原值" | "保留") {
        if let Some(option) = question.options.iter().position(|option| option.base) {
            return Ok((question_index, option));
        }
    }
    if let Some(option) = question.option_by_value(value) {
        return Ok((question_index, option));
    }
    // 「保留30」允许省掉「保留」后再匹配；也允许匹配选项说明中的取值。
    if let Some(option) = question
        .options
        .iter()
        .position(|option| option.description.contains(value))
    {
        return Ok((question_index, option));
    }
    Err(format!(
        "第 {} 处没有取值「{}」；可选：{}。",
        number,
        value,
        question
            .options
            .iter()
            .map(|option| option.label.to_string())
            .collect::<Vec<_>>()
            .join("/")
    ))
}

fn normalize_answer(answer: &str) -> String {
    let mapped: Vec<char> = answer
        .trim()
        .chars()
        .map(|ch| match ch {
            '０'..='９' => char::from_u32(ch as u32 - '０' as u32 + '0' as u32).unwrap_or(ch),
            'Ａ'..='Ｚ' => char::from_u32(ch as u32 - 'Ａ' as u32 + 'A' as u32).unwrap_or(ch),
            'ａ'..='ｚ' => char::from_u32(ch as u32 - 'ａ' as u32 + 'a' as u32).unwrap_or(ch),
            '：' => ':',
            '；' => ';',
            '，' => ',',
            other => other,
        })
        .collect();
    let mut normalized = String::with_capacity(mapped.len());
    for (index, ch) in mapped.iter().copied().enumerate() {
        if ch == ' '
            && index > 0
            && index + 1 < mapped.len()
            && mapped[index - 1].is_ascii_digit()
            && mapped[index + 1].is_ascii_alphabetic()
        {
            continue;
        }
        normalized.push(ch);
    }
    normalized
}

#[cfg(test)]
mod tests {
    use super::*;

    fn claim(subject: &str, predicate: &str, object: &str, strength: AssertionStrength) -> BatchClaim {
        BatchClaim {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
            strength,
            correction_cue: None,
            source: "第 1 句".to_string(),
            raw: format!("{}的{}是{}", subject, predicate, object),
            span: (0, 0),
        }
    }

    fn functional_options() -> BatchOptions {
        BatchOptions::new().declare_functional(["年龄"])
    }

    #[test]
    fn conservative_segmentation_splits_strong_boundaries_and_parseable_commas() {
        let outcome = parse_batch_claims("张三的师傅是李四，李四的师傅是王五。钱七的年龄是30");
        assert_eq!(outcome.claims.len(), 3, "{:?}", outcome);
        assert_eq!(outcome.claims[0].subject, "张三");
        assert_eq!(outcome.claims[1].subject, "李四");
        assert_eq!(outcome.claims[2].object, "30");
    }

    #[test]
    fn conditional_sentence_is_not_silently_turned_into_facts() {
        let outcome = parse_batch_claims("如果张三的师傅是李四，那么张三的师祖是赵六");
        assert!(outcome.claims.is_empty(), "{:?}", outcome);
        assert!(!outcome.errors.is_empty());
    }

    #[test]
    fn correction_cue_and_ellipsis_fill_previous_slot() {
        let outcome = parse_batch_claims("张三30岁，其实31岁；李四32岁");
        assert_eq!(outcome.claims.len(), 3, "{:?}", outcome);
        assert_eq!(outcome.claims[0].subject, "张三");
        assert_eq!(outcome.claims[0].predicate, "年龄");
        assert_eq!(outcome.claims[0].object, "30");
        assert_eq!(outcome.claims[1].subject, "张三");
        assert_eq!(outcome.claims[1].predicate, "年龄");
        assert_eq!(outcome.claims[1].object, "31");
        assert_eq!(outcome.claims[1].strength, AssertionStrength::Correction);
        assert_eq!(outcome.claims[1].correction_cue.as_deref(), Some("其实"));
        assert_eq!(outcome.claims[2].subject, "李四");
        assert_eq!(outcome.claims[2].object, "32");
    }

    #[test]
    fn comma_kept_when_pieces_do_not_parse() {
        let outcome = parse_batch_claims("张三，30岁");
        assert_eq!(outcome.claims.len(), 1, "{:?}", outcome);
        assert_eq!(outcome.claims[0].subject, "张三");
        assert_eq!(outcome.claims[0].predicate, "年龄");
        assert_eq!(outcome.claims[0].object, "30");
    }

    #[test]
    fn functional_tie_isolates_all_and_asks_one_question() {
        let claims = vec![
            claim("张三", "年龄", "30", AssertionStrength::Direct),
            claim("张三", "年龄", "31", AssertionStrength::Direct),
            claim("李四", "师傅", "王五", AssertionStrength::Direct),
        ];
        let plan = plan_batch(claims, &functional_options());
        assert_eq!(plan.questions.len(), 1);
        assert_eq!(plan.conflicts[0].kind, BatchConflictKind::FunctionalMultiValue);
        assert_eq!(plan.accepted.len(), 1, "无冲突槽应被接受");
        assert!(plan.claims[plan.accepted[0]].subject == "李四");
        assert_eq!(plan.isolated.len(), 2);
        assert!(plan.questions[0].options.len() >= 3);
    }

    #[test]
    fn correction_wins_over_direct_without_a_question() {
        let claims = vec![
            claim("张三", "年龄", "30", AssertionStrength::Direct),
            claim("张三", "年龄", "31", AssertionStrength::Correction),
        ];
        let plan = plan_batch(claims, &functional_options());
        assert!(plan.questions.is_empty());
        assert_eq!(plan.accepted.len(), 1);
        assert_eq!(plan.claims[plan.accepted[0]].object, "31");
        assert_eq!(plan.isolated.len(), 1);
    }

    #[test]
    fn graph_base_conflict_needs_explicit_correction() {
        let claims = vec![claim("张三", "年龄", "31", AssertionStrength::Direct)];
        let options = functional_options()
            .with_base_values(SlotKey::new("张三", "年龄"), vec!["30".to_string()]);
        let plan = plan_batch(claims, &options);
        assert_eq!(plan.questions.len(), 1);
        assert_eq!(plan.conflicts[0].kind, BatchConflictKind::GraphBaseConflict);
        assert_eq!(plan.questions[0].default_option, Some(0), "默认应保留图谱基值");
        assert!(plan.questions[0].options.iter().any(|option| option.base));
    }

    #[test]
    fn non_functional_slot_allows_multiple_values() {
        let claims = vec![
            claim("张三", "职业", "教师", AssertionStrength::Direct),
            claim("张三", "职业", "作家", AssertionStrength::Direct),
        ];
        let plan = plan_batch(claims, &BatchOptions::new());
        assert!(plan.questions.is_empty());
        assert_eq!(plan.accepted.len(), 2);
    }

    #[test]
    fn plan_is_order_independent_for_acceptance_and_isolation() {
        let claims = vec![
            claim("张三", "年龄", "30", AssertionStrength::Direct),
            claim("张三", "年龄", "31", AssertionStrength::Direct),
            claim("李四", "师傅", "王五", AssertionStrength::Direct),
        ];
        let first = plan_batch(claims.clone(), &functional_options());
        let mut reversed = claims;
        reversed.reverse();
        let second = plan_batch(reversed, &functional_options());
        let mut first_values: Vec<String> = first
            .accepted
            .iter()
            .map(|index| first.claims[*index].object.clone())
            .collect();
        let mut second_values: Vec<String> = second
            .accepted
            .iter()
            .map(|index| second.claims[*index].object.clone())
            .collect();
        first_values.sort();
        second_values.sort();
        assert_eq!(first_values, second_values);
        assert_eq!(first.questions.len(), second.questions.len());
        assert_eq!(first.isolated.len(), second.isolated.len());
        assert_eq!(
            first.conflicts[0].values.len(),
            second.conflicts[0].values.len()
        );
    }

    #[test]
    fn atomic_mode_defers_everything_when_questions_remain() {
        let claims = vec![
            claim("张三", "年龄", "30", AssertionStrength::Direct),
            claim("张三", "年龄", "31", AssertionStrength::Direct),
            claim("李四", "师傅", "王五", AssertionStrength::Direct),
        ];
        let options = functional_options().with_mode(BatchMode::Atomic);
        let plan = plan_batch(claims, &options);
        assert!(plan.accepted.is_empty());
        assert_eq!(plan.deferred.len(), 1);
        assert_eq!(plan.questions.len(), 1);
    }

    #[test]
    fn answer_parser_handles_per_question_and_global_forms() {
        let claims = vec![
            claim("张三", "年龄", "30", AssertionStrength::Direct),
            claim("张三", "年龄", "31", AssertionStrength::Direct),
            claim("李四", "年龄", "40", AssertionStrength::Direct),
            claim("李四", "年龄", "41", AssertionStrength::Direct),
        ];
        let plan = plan_batch(claims, &functional_options());
        assert_eq!(plan.questions.len(), 2);
        let answer = parse_batch_answer("1B，2保留40，其余同意", &plan.questions);
        assert_eq!(answer.selections.len(), 2);
        assert!(answer.selections[0].is_some());
        assert!(answer.selections[1].is_some());
        assert!(answer.is_complete());

        let last = parse_batch_answer("都按后者", &plan.questions);
        assert_eq!(last.selections[0], Some(plan.questions[0].options.len() - 1));
        assert_eq!(last.selections[1], Some(plan.questions[1].options.len() - 1));

        let cancelled = parse_batch_answer("取消", &plan.questions);
        assert!(cancelled.cancelled);
    }

    #[test]
    fn independent_conflicts_share_one_table_and_resolve_together() {
        let claims = vec![
            claim("张三", "年龄", "30", AssertionStrength::Direct),
            claim("张三", "年龄", "31", AssertionStrength::Direct),
            claim("李四", "年龄", "40", AssertionStrength::Direct),
            claim("李四", "年龄", "41", AssertionStrength::Direct),
        ];
        let plan = plan_batch(claims, &functional_options());
        assert!(plan.accepted.is_empty(), "{:?}", plan);
        assert_eq!(plan.questions.len(), 2, "{:?}", plan);
        let answer = parse_batch_answer("1A，2B", &plan.questions);
        assert!(answer.is_complete(), "{:?}", answer);
        assert_eq!(answer.selections, vec![Some(0), Some(1)]);
    }

    #[test]
    fn answer_parser_accepts_ordinal_prefix_space_and_keep_base() {
        let claims = vec![
            claim("张三", "年龄", "31", AssertionStrength::Direct),
        ];
        let options = functional_options()
            .with_base_values(SlotKey::new("张三", "年龄"), vec!["30".to_string()]);
        let plan = plan_batch(claims, &options);
        assert_eq!(plan.questions.len(), 1);
        let spaced = parse_batch_answer("第1题 B", &plan.questions);
        assert!(spaced.is_complete(), "{:?}", spaced);
        assert_eq!(spaced.selections[0], Some(1));
        let keep = parse_batch_answer("1保留原值", &plan.questions);
        assert!(keep.is_complete(), "{:?}", keep);
        assert_eq!(keep.selections[0], Some(0));
    }

    #[test]
    fn answer_parser_reports_unknown_selection_without_guessing() {
        let claims = vec![
            claim("张三", "年龄", "30", AssertionStrength::Direct),
            claim("张三", "年龄", "31", AssertionStrength::Direct),
        ];
        let plan = plan_batch(claims, &functional_options());
        let answer = parse_batch_answer("1Z", &plan.questions);
        assert!(answer.selections[0].is_none());
        assert!(!answer.notes.is_empty());
    }
}

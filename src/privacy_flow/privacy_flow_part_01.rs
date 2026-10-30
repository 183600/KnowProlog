use crate::function_words::{wh_kind, WhKind};
use crate::nlp::{AnnotatedWord, TagSource};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;
/// 会话隐私台账的方言版本；写入 JSON 以便迁移。
pub const PRIVACY_LEDGER_VERSION: u32 = 1;
/// 一次闭包最多保留多少个推导事实（含基础事实）。
pub const DEFAULT_MAX_CLOSURE_ATOMS: usize = 512;
/// 一次闭包默认最大连接深度。
pub const DEFAULT_MAX_CLOSURE_DEPTH: usize = 4;
/// 默认 k-匿名阈值。
pub const DEFAULT_K_ANONYMITY: usize = 2;

// ---------------------------------------------------------------------------
// 基础枚举
// ---------------------------------------------------------------------------
/// 披露类别。类别决定默认精度、是否需要授权，以及占位表达的类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrivacyCategory {
    PersonalIdentity,
    Contact,
    Credential,
    Location,
    Health,
    Finance,
    Demographic,
    Relationship,
    Other,
}
impl PrivacyCategory {
    pub const fn code(self) -> &'static str {
        match self {
            PrivacyCategory::PersonalIdentity => "personal_identity",
            PrivacyCategory::Contact => "contact",
            PrivacyCategory::Credential => "credential",
            PrivacyCategory::Location => "location",
            PrivacyCategory::Health => "health",
            PrivacyCategory::Finance => "finance",
            PrivacyCategory::Demographic => "demographic",
            PrivacyCategory::Relationship => "relationship",
            PrivacyCategory::Other => "other",
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            PrivacyCategory::PersonalIdentity => "个人身份",
            PrivacyCategory::Contact => "联系方式",
            PrivacyCategory::Credential => "证件信息",
            PrivacyCategory::Location => "位置信息",
            PrivacyCategory::Health => "健康信息",
            PrivacyCategory::Finance => "财务信息",
            PrivacyCategory::Demographic => "人口属性",
            PrivacyCategory::Relationship => "社会关系",
            PrivacyCategory::Other => "普通信息",
        }
    }

    /// 是否属于「与人关联」的数据。普通谓词不参与 k-匿名检查。
    pub const fn is_personal(self) -> bool {
        !matches!(self, PrivacyCategory::Other)
    }

    /// 默认披露精度上限。人名可以作指称，默认不遮；敏感属性默认只说类型
    /// 或只说到城市/年龄段。
    pub const fn default_precision(self) -> DisclosurePrecision {
        match self {
            PrivacyCategory::PersonalIdentity => DisclosurePrecision::Exact,
            PrivacyCategory::Contact => DisclosurePrecision::TypeOnly,
            PrivacyCategory::Credential => DisclosurePrecision::TypeOnly,
            PrivacyCategory::Location => DisclosurePrecision::Coarse,
            PrivacyCategory::Health => DisclosurePrecision::TypeOnly,
            PrivacyCategory::Finance => DisclosurePrecision::TypeOnly,
            PrivacyCategory::Demographic => DisclosurePrecision::Coarse,
            PrivacyCategory::Relationship => DisclosurePrecision::Coarse,
            PrivacyCategory::Other => DisclosurePrecision::Exact,
        }
    }

    /// 默认需要数据主体显式授权的类别。
    pub const fn requires_consent_by_default(self) -> bool {
        matches!(
            self,
            PrivacyCategory::Contact
                | PrivacyCategory::Credential
                | PrivacyCategory::Health
                | PrivacyCategory::Finance
                | PrivacyCategory::Demographic
        )
    }

    /// 该类别自然回答的疑问类型，用于发现占位表达与问句类型不一致。
    pub const fn answer_kind(self) -> Option<WhKind> {
        match self {
            PrivacyCategory::PersonalIdentity | PrivacyCategory::Relationship => {
                Some(WhKind::Person)
            }
            PrivacyCategory::Location => Some(WhKind::Place),
            PrivacyCategory::Demographic => Some(WhKind::Quantity),
            _ => Some(WhKind::Thing),
        }
    }

    pub fn compatible_with(self, wh: Option<WhKind>) -> bool {
        match wh {
            None | Some(WhKind::Thing) => true,
            Some(WhKind::Time) if self == PrivacyCategory::Demographic => true,
            Some(kind) => self == PrivacyCategory::Other || self.answer_kind() == Some(kind),
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        let key = value.trim().to_ascii_lowercase();
        let parsed = match key.as_str() {
            "personal_identity" | "identity" | "name" | "身份" | "姓名" => {
                PrivacyCategory::PersonalIdentity
            }
            "contact" | "联系方式" | "电话" | "邮箱" => PrivacyCategory::Contact,
            "credential" | "证件" | "证件信息" => PrivacyCategory::Credential,
            "location" | "位置" | "位置信息" | "地址" => PrivacyCategory::Location,
            "health" | "健康" | "健康信息" | "医疗" => PrivacyCategory::Health,
            "finance" | "财务" | "财务信息" | "金融" => PrivacyCategory::Finance,
            "demographic" | "人口属性" | "年龄" => PrivacyCategory::Demographic,
            "relationship" | "社会关系" | "关系" => PrivacyCategory::Relationship,
            "other" | "普通信息" | "其他" => PrivacyCategory::Other,
            _ => return None,
        };
        Some(parsed)
    }

    /// 从自然语言短语里猜类别，供「别提我的电话」这类言语行为使用。
    pub fn infer_from_text(text: &str) -> Option<Self> {
        let lowered = text.to_lowercase();
        let table: &[(&[&str], PrivacyCategory)] = &[
            (
                &["身份证", "护照", "证件", "驾照", "社保号"],
                PrivacyCategory::Credential,
            ),
            (
                &["电话", "手机", "联系方式", "邮箱", "email", "住址电话"],
                PrivacyCategory::Contact,
            ),
            (
                &["病历", "病史", "病情", "诊断", "用药", "过敏", "健康"],
                PrivacyCategory::Health,
            ),
            (
                &["工资", "收入", "银行卡", "余额", "资产", "财务"],
                PrivacyCategory::Finance,
            ),
            (
                &["住址", "地址", "位置", "行踪", "住哪", "住在"],
                PrivacyCategory::Location,
            ),
            (
                &["年龄", "生日", "出生", "岁数"],
                PrivacyCategory::Demographic,
            ),
            (
                &[
                    "父亲", "母亲", "父母", "配偶", "亲属", "家人", "朋友", "同事",
                ],
                PrivacyCategory::Relationship,
            ),
            (
                &["姓名", "名字", "真名", "身份"],
                PrivacyCategory::PersonalIdentity,
            ),
        ];
        table
            .iter()
            .find(|(terms, _)| terms.iter().any(|term| lowered.contains(term)))
            .map(|(_, category)| *category)
    }
}
/// 披露精度，按「从最粗到最细」排序；`Ord` 越小越保守。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DisclosurePrecision {
    /// 完全不披露。
    Hidden,
    /// 只披露类型化占位（某位人士 / 某地 / 某联系方式）。
    #[serde(alias = "type")]
    TypeOnly,
    /// 粗粒度（城市、年龄段、年代）。
    Coarse,
    /// 细粒度（区县、五岁段、掩码后的号码）。
    Granular,
    /// 原值。
    Exact,
}
impl DisclosurePrecision {
    pub const fn code(self) -> &'static str {
        match self {
            DisclosurePrecision::Hidden => "hidden",
            DisclosurePrecision::TypeOnly => "type",
            DisclosurePrecision::Coarse => "coarse",
            DisclosurePrecision::Granular => "granular",
            DisclosurePrecision::Exact => "exact",
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            DisclosurePrecision::Hidden => "隐藏",
            DisclosurePrecision::TypeOnly => "仅类型",
            DisclosurePrecision::Coarse => "粗粒度",
            DisclosurePrecision::Granular => "细粒度",
            DisclosurePrecision::Exact => "精确",
        }
    }

    /// 再降一级；`Hidden` 保持不动。
    pub const fn coarsen(self) -> Self {
        match self {
            DisclosurePrecision::Exact => DisclosurePrecision::Granular,
            DisclosurePrecision::Granular => DisclosurePrecision::Coarse,
            DisclosurePrecision::Coarse => DisclosurePrecision::TypeOnly,
            DisclosurePrecision::TypeOnly | DisclosurePrecision::Hidden => {
                DisclosurePrecision::Hidden
            }
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        let parsed = match value.trim().to_ascii_lowercase().as_str() {
            "hidden" | "hide" | "none" | "隐藏" => DisclosurePrecision::Hidden,
            "type" | "type_only" | "typeonly" | "仅类型" => DisclosurePrecision::TypeOnly,
            "coarse" | "粗粒度" | "粗略" => DisclosurePrecision::Coarse,
            "granular" | "fine" | "细粒度" => DisclosurePrecision::Granular,
            "exact" | "full" | "精确" => DisclosurePrecision::Exact,
            _ => return None,
        };
        Some(parsed)
    }
}
/// 证据强度。弱证据只提示；是否硬拦由策略决定。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceStrength {
    Weak,
    Strong,
}
impl EvidenceStrength {
    pub const fn code(self) -> &'static str {
        match self {
            EvidenceStrength::Weak => "weak",
            EvidenceStrength::Strong => "strong",
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            EvidenceStrength::Weak => "弱",
            EvidenceStrength::Strong => "强",
        }
    }
}
/// 策略模式。`Audit` 只影子计算，不改写回答，也不写会话台账。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PolicyMode {
    #[default]
    Audit,
    Enforce,
}
impl PolicyMode {
    pub const fn code(self) -> &'static str {
        match self {
            PolicyMode::Audit => "audit",
            PolicyMode::Enforce => "enforce",
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            PolicyMode::Audit => "审计（影子）",
            PolicyMode::Enforce => "执行",
        }
    }
}

// ---------------------------------------------------------------------------
// 敏感跨度
// ---------------------------------------------------------------------------
/// 文本中的一个敏感跨度（字符半开区间 `[start, end)`）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SensitiveSpan {
    pub start: usize,
    pub end: usize,
    pub text: String,
    pub category: PrivacyCategory,
    pub strength: EvidenceStrength,
    pub reason: String,
    pub suggested_precision: DisclosurePrecision,
}
impl SensitiveSpan {
    pub fn is_blocking(&self, policy: &PrivacyPolicy) -> bool {
        self.strength == EvidenceStrength::Strong || policy.weak_evidence_blocks
    }

    pub fn render(&self) -> String {
        format!(
            "[{}] {}（{}，位置 {}..{}；建议 {}；证据：{}）",
            self.strength.label(),
            self.text,
            self.category.label(),
            self.start,
            self.end,
            self.suggested_precision.label(),
            self.reason
        )
    }
}
/// 一条类型保持改写的记录。只保存位置、类别、精度与可披露表达，不保存原文，
/// 因而可以直接写进回执或评测日志。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RedactionRecord {
    pub start: usize,
    pub end: usize,
    pub category: PrivacyCategory,
    pub strength: EvidenceStrength,
    pub precision: DisclosurePrecision,
    /// 类型化占位或粗化后的表达；`suppressed` 为真时为空。
    pub replacement: String,
    pub suppressed: bool,
}
/// 文本级改写结果：保持非敏感片段与语序，只替换阻断跨度。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpanRedaction {
    pub text: String,
    #[serde(default)]
    pub records: Vec<RedactionRecord>,
    /// 因弱证据默认只提示而原样保留的跨度数。
    #[serde(default)]
    pub kept_weak: usize,
}
impl SpanRedaction {
    /// 是否发生了实际改写（替换或省略）。
    pub fn changed(&self) -> bool {
        self.records
            .iter()
            .any(|record| record.suppressed || !record.replacement.is_empty())
    }

    pub fn suppressed(&self) -> usize {
        self.records
            .iter()
            .filter(|record| record.suppressed)
            .count()
    }

    pub fn replacements(&self) -> usize {
        self.records.len().saturating_sub(self.suppressed())
    }
}
/// 词性 + 上下文规则驱动的敏感跨度检测器。
///
/// 强证据：配置词表、手机号/身份证/邮箱/银行卡号、住址引导词、健康词表。
/// 弱证据：结巴的人名/地名兜底标签。弱证据默认只提示。
pub struct SpanDetector<'a> {
    policy: &'a PrivacyPolicy,
}
impl<'a> SpanDetector<'a> {
    pub fn new(policy: &'a PrivacyPolicy) -> Self {
        Self { policy }
    }

    /// 跨度的建议精度跟随策略。弱证据只有在被策略提升为硬证据时才收紧到
    /// 类型化表达；默认仍保留人名等指称，避免过度脱敏。
    fn suggested_precision(&self, category: PrivacyCategory, weak: bool) -> DisclosurePrecision {
        let ceiling = self.policy.max_precision(category);
        if weak && self.policy.weak_evidence_blocks && ceiling == DisclosurePrecision::Exact {
            DisclosurePrecision::TypeOnly
        } else {
            ceiling
        }
    }

    pub fn scan(&self, text: &str, annotated: &[AnnotatedWord]) -> Vec<SensitiveSpan> {
        let offsets: Vec<usize> = text.char_indices().map(|(byte, _)| byte).collect();
        let to_char = |byte: usize| match offsets.binary_search(&byte) {
            Ok(index) => index,
            Err(index) => index,
        };
        let mut spans = Vec::new();

        for (category, regex) in strong_regexes() {
            for found in regex.find_iter(text) {
                if !standalone_digits(text, found.start(), found.end()) {
                    continue;
                }
                spans.push(SensitiveSpan {
                    start: to_char(found.start()),
                    end: to_char(found.end()),
                    text: found.as_str().to_string(),
                    category: *category,
                    strength: EvidenceStrength::Strong,
                    reason: "上下文数字/邮箱规则".to_string(),
                    suggested_precision: self.suggested_precision(*category, false),
                });
            }
        }

        for capture in address_regex().captures_iter(text) {
            let Some(found) = capture.get(1) else {
                continue;
            };
            if found.as_str().chars().count() < 2 {
                continue;
            }
            spans.push(SensitiveSpan {
                start: to_char(found.start()),
                end: to_char(found.end()),
                text: found.as_str().to_string(),
                category: PrivacyCategory::Location,
                strength: EvidenceStrength::Strong,
                reason: "住址引导词".to_string(),
                suggested_precision: self.suggested_precision(PrivacyCategory::Location, false),
            });
        }

        for (term, category) in &self.policy.sensitive_terms {
            if term.is_empty() {
                continue;
            }
            for (byte, found) in text.match_indices(term) {
                spans.push(SensitiveSpan {
                    start: to_char(byte),
                    end: to_char(byte + found.len()),
                    text: found.to_string(),
                    category: *category,
                    strength: EvidenceStrength::Strong,
                    reason: format!("敏感词表：{}", term),
                    suggested_precision: self.suggested_precision(*category, false),
                });
            }
        }

        for word in annotated {
            let (category, reason) = match word.pos.as_str() {
                "nr" => (
                    PrivacyCategory::PersonalIdentity,
                    "人名标签（只提示，人名可作指称）",
                ),
                "ns" => (
                    PrivacyCategory::Location,
                    "地名标签（未结合上下文，仅提示）",
                ),
                _ if word.source == TagSource::NamePattern => (
                    PrivacyCategory::PersonalIdentity,
                    "姓名模式（只提示，人名可作指称）",
                ),
                _ => continue,
            };
            spans.push(SensitiveSpan {
                start: word.start,
                end: word.end,
                text: word.text.clone(),
                category,
                strength: EvidenceStrength::Weak,
                reason: reason.to_string(),
                suggested_precision: self.suggested_precision(category, true),
            });
        }

        resolve_overlaps(spans)
    }

    /// 扫描并做类型保持改写。非阻断跨度（默认的弱证据）原样保留。
    pub fn redact(&self, text: &str, annotated: &[AnnotatedWord]) -> SpanRedaction {
        let spans = self.scan(text, annotated);
        redact_spans(text, &spans, self.policy)
    }
}
/// 把已经检测出的跨度改写成同类型的粗粒度表达。
///
/// 与逐事实裁决共享 [`coarsen_value`]，因此“某位人士 / 某地 / 15-19 岁”
/// 这类表达在多轮之间保持稳定；只替换阻断跨度，句子的其余部分与语序不变。
pub fn redact_spans(text: &str, spans: &[SensitiveSpan], policy: &PrivacyPolicy) -> SpanRedaction {
    let chars: Vec<char> = text.chars().collect();
    let mut output = String::with_capacity(text.len());
    let mut records = Vec::new();
    let mut kept_weak = 0usize;
    let mut cursor = 0usize;

    for span in spans {
        if !span.is_blocking(policy) {
            kept_weak += 1;
            continue;
        }
        let start = span.start.min(chars.len());
        let end = span.end.min(chars.len());
        if start < cursor || start >= end {
            continue;
        }
        output.extend(chars[cursor..start].iter());
        let replacement = coarsen_value(span.category, span.suggested_precision, &span.text, None);
        match replacement {
            Some(value) if value != span.text => {
                output.push_str(&value);
                records.push(RedactionRecord {
                    start,
                    end,
                    category: span.category,
                    strength: span.strength,
                    precision: span.suggested_precision,
                    replacement: value,
                    suppressed: false,
                });
            }
            Some(_) => output.extend(chars[start..end].iter()),
            None => {
                records.push(RedactionRecord {
                    start,
                    end,
                    category: span.category,
                    strength: span.strength,
                    precision: DisclosurePrecision::Hidden,
                    replacement: String::new(),
                    suppressed: true,
                });
            }
        }
        cursor = end;
    }
    output.extend(chars[cursor..].iter());

    SpanRedaction {
        text: output,
        records,
        kept_weak,
    }
}
fn strong_regexes() -> &'static [(PrivacyCategory, Regex)] {
    static REGEXES: OnceLock<Vec<(PrivacyCategory, Regex)>> = OnceLock::new();
    REGEXES.get_or_init(|| {
        vec![
            (
                PrivacyCategory::Contact,
                Regex::new(r"(?:\+?86[-\s]?)?1[3-9]\d{9}").expect("手机号正则"),
            ),
            (
                PrivacyCategory::Contact,
                Regex::new(r"0\d{2,3}[-\s]?\d{7,8}").expect("固定电话正则"),
            ),
            (
                PrivacyCategory::Contact,
                Regex::new(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
                    .expect("邮箱正则"),
            ),
            (
                PrivacyCategory::Credential,
                Regex::new(r"[1-9]\d{5}(?:18|19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[0-9Xx]")
                    .expect("身份证正则"),
            ),
            (
                PrivacyCategory::Finance,
                Regex::new(r"\d{16,19}").expect("银行卡候选正则"),
            ),
            (
                PrivacyCategory::Demographic,
                Regex::new(r"\d{1,3}\s*(?:岁|周岁)").expect("年龄正则"),
            ),
        ]
    })
}
fn address_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(
            r"(?:住在|住址(?:是|为)?|地址(?:是|为)?|家住|位于)[:：\s]*([\u4e00-\u9fffA-Za-z0-9#号栋单元室路街巷弄区县市镇村]{2,40})",
        )
        .expect("住址正则")
    })
}
fn standalone_digits(text: &str, start: usize, end: usize) -> bool {
    let before = text[..start].chars().next_back();
    let after = text[end..].chars().next();
    !before
        .map(|character| character.is_ascii_digit())
        .unwrap_or(false)
        && !after
            .map(|character| character.is_ascii_digit())
            .unwrap_or(false)
}
fn category_priority(category: PrivacyCategory) -> u8 {
    match category {
        PrivacyCategory::Credential => 9,
        PrivacyCategory::Contact => 8,
        PrivacyCategory::Health => 7,
        PrivacyCategory::Finance => 6,
        PrivacyCategory::Location => 5,
        PrivacyCategory::Demographic => 4,
        PrivacyCategory::Relationship => 3,
        PrivacyCategory::PersonalIdentity => 2,
        PrivacyCategory::Other => 1,
    }
}
fn resolve_overlaps(mut spans: Vec<SensitiveSpan>) -> Vec<SensitiveSpan> {
    spans.sort_by(|left, right| {
        right
            .strength
            .cmp(&left.strength)
            .then_with(|| (right.end - right.start).cmp(&(left.end - left.start)))
            .then_with(|| category_priority(right.category).cmp(&category_priority(left.category)))
            .then_with(|| left.start.cmp(&right.start))
    });
    let mut kept: Vec<SensitiveSpan> = Vec::new();
    for span in spans {
        let overlaps = kept
            .iter()
            .any(|other| span.start < other.end && other.start < span.end);
        if !overlaps {
            kept.push(span);
        }
    }
    kept.sort_by_key(|span| (span.start, span.end));
    kept
}

// ---------------------------------------------------------------------------
// 策略
// ---------------------------------------------------------------------------
/// 谓词匹配规则：谓词包含 `contains` 时归入 `category`。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PredicateRule {
    pub contains: String,
    pub category: PrivacyCategory,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_precision: Option<DisclosurePrecision>,
}
impl PredicateRule {
    pub fn new(contains: impl Into<String>, category: PrivacyCategory) -> Self {
        Self {
            contains: contains.into(),
            category,
            max_precision: None,
        }
    }
}
/// 组合推导规则：`?s -first-> ?mid -second-> ?o` ⇒ `?s -conclusion-> ?o`。
///
/// 它不负责通用 Horn 推理，只覆盖对话里最常见、也最容易泄漏的二跳组合
/// （学校 + 位置 ⇒ 所在城区）。闭包有深度与原子预算，预算耗尽时保守收紧。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InferenceRule {
    pub name: String,
    pub first: String,
    pub second: String,
    pub conclusion: String,
    pub category: PrivacyCategory,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_precision: Option<DisclosurePrecision>,
}
impl InferenceRule {
    pub fn new(
        name: impl Into<String>,
        first: impl Into<String>,
        second: impl Into<String>,
        conclusion: impl Into<String>,
        category: PrivacyCategory,
    ) -> Self {
        Self {
            name: name.into(),
            first: first.into(),
            second: second.into(),
            conclusion: conclusion.into(),
            category,
            max_precision: None,
        }
    }

    pub fn with_max_precision(mut self, precision: DisclosurePrecision) -> Self {
        self.max_precision = Some(precision);
        self
    }
}
/// 披露策略。所有字段都有默认值，配置文件可以只写需要覆盖的部分。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct PrivacyPolicy {
    pub mode: PolicyMode,
    pub k_anonymity: usize,
    pub max_closure_depth: usize,
    pub max_closure_atoms: usize,
    /// 闭包预算耗尽时是否按「无法证明安全」保守阻断个人事实。
    pub fail_closed_on_closure_truncation: bool,
    pub weak_evidence_blocks: bool,
    pub category_precision: BTreeMap<PrivacyCategory, DisclosurePrecision>,
    pub predicate_rules: Vec<PredicateRule>,
    pub inference_rules: Vec<InferenceRule>,
    pub sensitive_terms: BTreeMap<String, PrivacyCategory>,
    pub consent_required: BTreeSet<PrivacyCategory>,
}
impl Default for PrivacyPolicy {
    fn default() -> Self {
        Self {
            mode: PolicyMode::Audit,
            k_anonymity: DEFAULT_K_ANONYMITY,
            max_closure_depth: DEFAULT_MAX_CLOSURE_DEPTH,
            max_closure_atoms: DEFAULT_MAX_CLOSURE_ATOMS,
            fail_closed_on_closure_truncation: true,
            weak_evidence_blocks: false,
            category_precision: default_category_precision(),
            predicate_rules: default_predicate_rules(),
            inference_rules: Vec::new(),
            sensitive_terms: default_sensitive_terms(),
            consent_required: default_consent_categories(),
        }
    }
}
impl PrivacyPolicy {
    pub fn enforce() -> Self {
        Self {
            mode: PolicyMode::Enforce,
            ..Self::default()
        }
    }

    pub fn max_precision(&self, category: PrivacyCategory) -> DisclosurePrecision {
        self.category_precision
            .get(&category)
            .copied()
            .unwrap_or_else(|| category.default_precision())
    }

    pub fn classify_predicate(&self, predicate: &str) -> PrivacyCategory {
        let lowered = predicate.to_lowercase();
        for rule in &self.predicate_rules {
            if !rule.contains.is_empty() && lowered.contains(&rule.contains.to_lowercase()) {
                return rule.category;
            }
        }
        PrivacyCategory::Other
    }

    pub fn max_precision_for_predicate(&self, predicate: &str) -> Option<DisclosurePrecision> {
        let lowered = predicate.to_lowercase();
        self.predicate_rules
            .iter()
            .find(|rule| {
                rule.max_precision.is_some()
                    && !rule.contains.is_empty()
                    && lowered.contains(&rule.contains.to_lowercase())
            })
            .and_then(|rule| rule.max_precision)
    }

    pub fn requires_consent(&self, category: PrivacyCategory) -> bool {
        self.consent_required.contains(&category)
    }

    /// 把配置里的规则叠加到内置默认规则上：同名词条/谓词先到先得，缺省项
    /// 由默认值补齐。这样策略文件可以只写新增规则，而不会意外丢掉手机号、
    /// 住址等基础保护。
    pub fn merge_default_rules(&mut self) {
        for rule in default_predicate_rules() {
            if !self
                .predicate_rules
                .iter()
                .any(|existing| existing.contains == rule.contains)
            {
                self.predicate_rules.push(rule);
            }
        }
        for (term, category) in default_sensitive_terms() {
            self.sensitive_terms.entry(term).or_insert(category);
        }
        for (category, precision) in default_category_precision() {
            self.category_precision.entry(category).or_insert(precision);
        }
        self.max_closure_depth = self.max_closure_depth.max(1);
        self.max_closure_atoms = self.max_closure_atoms.max(1);
    }

    /// 面向用户的策略摘要。只列规则与上限，不暴露任何事实值。
    pub fn render_summary(&self) -> String {
        let categories = self
            .category_precision
            .iter()
            .map(|(category, precision)| format!("{}={}", category.code(), precision.code()))
            .collect::<Vec<_>>()
            .join(" ");
        let consent = self
            .consent_required
            .iter()
            .map(|category| category.code())
            .collect::<Vec<_>>()
            .join(" ");
        let mut text = format!(
            "隐私策略：模式 {}，k={}，闭包深度 {}，闭包原子上限 {}，截断{}保守阻断，弱证据{}硬拦；类别上限：{}；需授权：{}；敏感词 {} 条，推导规则 {} 条",
            self.mode.label(),
            self.k_anonymity,
            self.max_closure_depth,
            self.max_closure_atoms,
            if self.fail_closed_on_closure_truncation {
                ""
            } else {
                "不"
            },
            if self.weak_evidence_blocks {
                ""
            } else {
                "不"
            },
            if categories.is_empty() {
                "（默认）"
            } else {
                categories.as_str()
            },
            if consent.is_empty() {
                "（无）"
            } else {
                consent.as_str()
            },
            self.sensitive_terms.len(),
            self.inference_rules.len()
        );
        if !self.inference_rules.is_empty() {
            let names = self
                .inference_rules
                .iter()
                .map(|rule| rule.name.as_str())
                .collect::<Vec<_>>()
                .join("、");
            text.push_str(&format!("；推导：{}", names));
        }
        text
    }
}
fn default_category_precision() -> BTreeMap<PrivacyCategory, DisclosurePrecision> {
    [
        PrivacyCategory::PersonalIdentity,
        PrivacyCategory::Contact,
        PrivacyCategory::Credential,
        PrivacyCategory::Location,
        PrivacyCategory::Health,
        PrivacyCategory::Finance,
        PrivacyCategory::Demographic,
        PrivacyCategory::Relationship,
        PrivacyCategory::Other,
    ]
    .into_iter()
    .map(|category| (category, category.default_precision()))
    .collect()
}
fn default_consent_categories() -> BTreeSet<PrivacyCategory> {
    [
        PrivacyCategory::Contact,
        PrivacyCategory::Credential,
        PrivacyCategory::Health,
        PrivacyCategory::Finance,
        PrivacyCategory::Demographic,
    ]
    .into_iter()
    .collect()
}
fn default_sensitive_terms() -> BTreeMap<String, PrivacyCategory> {
    [
        ("病历", PrivacyCategory::Health),
        ("病史", PrivacyCategory::Health),
        ("诊断书", PrivacyCategory::Health),
        ("诊断结果", PrivacyCategory::Health),
        ("处方", PrivacyCategory::Health),
        ("过敏史", PrivacyCategory::Health),
        ("住院记录", PrivacyCategory::Health),
        ("身份证号", PrivacyCategory::Credential),
        ("证件号码", PrivacyCategory::Credential),
        ("护照号码", PrivacyCategory::Credential),
        ("银行卡号", PrivacyCategory::Finance),
        ("工资条", PrivacyCategory::Finance),
        ("家庭住址", PrivacyCategory::Location),
    ]
    .into_iter()
    .map(|(term, category)| (term.to_string(), category))
    .collect()
}
fn default_predicate_rules() -> Vec<PredicateRule> {
    [
        ("电话", PrivacyCategory::Contact),
        ("手机", PrivacyCategory::Contact),
        ("联系方式", PrivacyCategory::Contact),
        ("邮箱", PrivacyCategory::Contact),
        ("email", PrivacyCategory::Contact),
        ("身份证", PrivacyCategory::Credential),
        ("护照", PrivacyCategory::Credential),
        ("证件", PrivacyCategory::Credential),
        ("住址", PrivacyCategory::Location),
        ("地址", PrivacyCategory::Location),
        ("居住", PrivacyCategory::Location),
        ("住在", PrivacyCategory::Location),
        ("病史", PrivacyCategory::Health),
        ("病历", PrivacyCategory::Health),
        ("诊断", PrivacyCategory::Health),
        ("病情", PrivacyCategory::Health),
        ("过敏", PrivacyCategory::Health),
        ("用药", PrivacyCategory::Health),
        ("工资", PrivacyCategory::Finance),
        ("收入", PrivacyCategory::Finance),
        ("银行卡", PrivacyCategory::Finance),
        ("账号", PrivacyCategory::Finance),
        ("余额", PrivacyCategory::Finance),
        ("资产", PrivacyCategory::Finance),
        ("年龄", PrivacyCategory::Demographic),
        ("生日", PrivacyCategory::Demographic),
        ("出生", PrivacyCategory::Demographic),
        ("父亲", PrivacyCategory::Relationship),
        ("母亲", PrivacyCategory::Relationship),
        ("配偶", PrivacyCategory::Relationship),
        ("亲属", PrivacyCategory::Relationship),
        ("朋友", PrivacyCategory::Relationship),
        ("同事", PrivacyCategory::Relationship),
        ("姓名", PrivacyCategory::PersonalIdentity),
        ("名字", PrivacyCategory::PersonalIdentity),
        ("真名", PrivacyCategory::PersonalIdentity),
    ]
    .into_iter()
    .map(|(contains, category)| PredicateRule::new(contains, category))
    .collect()
}

// ---------------------------------------------------------------------------
// 图谱视图
// ---------------------------------------------------------------------------
/// 带披露标签的事实。`precision` 是事实本身的固有粒度，通常为 `Exact`。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrivacyFact {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub category: PrivacyCategory,
    pub precision: DisclosurePrecision,
}
impl PrivacyFact {
    pub fn label(policy: &PrivacyPolicy, subject: &str, predicate: &str, object: &str) -> Self {
        let category = policy.classify_predicate(predicate);
        Self {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
            category,
            precision: DisclosurePrecision::Exact,
        }
    }

    pub fn key(&self) -> (&str, &str, &str) {
        (
            self.subject.as_str(),
            self.predicate.as_str(),
            self.object.as_str(),
        )
    }

    pub fn fact_digest(&self) -> String {
        fact_digest(&self.subject, &self.predicate, &self.object)
    }
}
/// 只读图谱视图：提供 k-匿名等价类与有界组合闭包。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrivacyGraph {
    #[serde(default)]
    pub facts: Vec<PrivacyFact>,
}
impl PrivacyGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_fact(
        &mut self,
        policy: &PrivacyPolicy,
        subject: &str,
        predicate: &str,
        object: &str,
    ) -> &PrivacyFact {
        self.facts
            .push(PrivacyFact::label(policy, subject, predicate, object));
        self.facts.last().expect("刚插入的事实")
    }

    pub fn find(&self, subject: &str, predicate: &str, object: &str) -> Option<&PrivacyFact> {
        self.facts.iter().find(|fact| {
            fact.subject == subject && fact.predicate == predicate && fact.object == object
        })
    }

    fn base_contains(&self, fact: &PrivacyFact) -> bool {
        self.facts.iter().any(|known| known.key() == fact.key())
    }

    /// 同一谓词、同一类别、同一精度桶下的不同数据主体数。
    pub fn class_size(&self, target: &PrivacyFact, precision: DisclosurePrecision) -> usize {
        let bucket = bucket_key(target, precision);
        let mut subjects = BTreeSet::new();
        for fact in &self.facts {
            if fact.predicate != target.predicate || fact.category != target.category {
                continue;
            }
            if bucket_key(fact, precision) == bucket {
                subjects.insert(fact.subject.clone());
            }
        }
        subjects.insert(target.subject.clone());
        subjects.len()
    }

    /// 对目标事实做有界闭包，返回「依赖目标事实的敏感结论」与是否截断。
    pub fn inference_leaks(
        &self,
        target: &PrivacyFact,
        policy: &PrivacyPolicy,
        precision: DisclosurePrecision,
    ) -> (Vec<InferenceLeak>, bool) {
        if precision == DisclosurePrecision::Hidden || policy.inference_rules.is_empty() {
            return (Vec::new(), false);
        }

        let mut base = self.facts.clone();
        let target_index = match base.iter().position(|fact| fact.key() == target.key()) {
            Some(index) => index,
            None => {
                base.push(target.clone());
                base.len() - 1
            }
        };
        let base_len = base.len();

        let mut work: Vec<WorkFact> = base
            .iter()
            .enumerate()
            .map(|(index, fact)| WorkFact {
                fact: fact.clone(),
                provenance: [index].into_iter().collect(),
                depth: 0,
                rule: None,
                max_precision: policy.max_precision(fact.category),
            })
            .collect();
        let mut known: BTreeMap<(String, String, String), usize> = work
            .iter()
            .enumerate()
            .map(|(index, item)| {
                (
                    (
                        item.fact.subject.clone(),
                        item.fact.predicate.clone(),
                        item.fact.object.clone(),
                    ),
                    index,
                )
            })
            .collect();
        let mut truncated = false;

        for _depth in 0..policy.max_closure_depth.max(1) {
            let snapshot_len = work.len();
            let mut candidates: Vec<(
                PrivacyFact,
                BTreeSet<usize>,
                usize,
                String,
                DisclosurePrecision,
            )> = Vec::new();
            for rule in &policy.inference_rules {
                let rule_precision = rule
                    .max_precision
                    .unwrap_or_else(|| policy.max_precision(rule.category));
                for left_index in 0..snapshot_len {
                    if work[left_index].fact.predicate != rule.first {
                        continue;
                    }
                    for right_index in 0..snapshot_len {
                        if work[right_index].fact.predicate != rule.second {
                            continue;
                        }
                        if work[left_index].fact.object != work[right_index].fact.subject {
                            continue;
                        }
                        let conclusion = PrivacyFact {
                            subject: work[left_index].fact.subject.clone(),
                            predicate: rule.conclusion.clone(),
                            object: work[right_index].fact.object.clone(),
                            category: rule.category,
                            precision: DisclosurePrecision::Exact,
                        };
                        if base.contains(&conclusion) {
                            continue;
                        }
                        let mut provenance = work[left_index].provenance.clone();
                        provenance.extend(work[right_index].provenance.iter().copied());
                        let depth = work[left_index]
                            .depth
                            .max(work[right_index].depth)
                            .saturating_add(1);
                        candidates.push((
                            conclusion,
                            provenance,
                            depth,
                            rule.name.clone(),
                            rule_precision,
                        ));
                    }
                }
            }

            if candidates.is_empty() {
                break;
            }

            let mut added = 0usize;
            for (fact, provenance, depth, rule, max_precision) in candidates {
                let key = (
                    fact.subject.clone(),
                    fact.predicate.clone(),
                    fact.object.clone(),
                );
                if let Some(index) = known.get(&key).copied() {
                    let existing = &mut work[index];
                    existing.provenance.extend(provenance);
                    existing.max_precision = existing.max_precision.min(max_precision);
                    existing.depth = existing.depth.max(depth);
                    continue;
                }
                if work.len() >= policy.max_closure_atoms.max(1) {
                    truncated = true;
                    break;
                }
                known.insert(key, work.len());
                work.push(WorkFact {
                    fact,
                    provenance,
                    depth,
                    rule: Some(rule),
                    max_precision,
                });
                added += 1;
            }
            if truncated || added == 0 {
                break;
            }
        }

        let mut leaks: Vec<InferenceLeak> = Vec::new();
        for item in work.iter().skip(base_len) {
            if !item.provenance.contains(&target_index) {
                continue;
            }
            if item.fact.category == PrivacyCategory::Other {
                continue;
            }
            if item.max_precision >= precision {
                continue;
            }
            if self.base_contains(&item.fact) {
                continue;
            }
            let leak = InferenceLeak {
                rule: item.rule.clone().unwrap_or_else(|| "组合推导".to_string()),
                conclusion: item.fact.predicate.clone(),
                category: item.fact.category,
                max_precision: item.max_precision,
            };
            if !leaks.iter().any(|known| {
                known.rule == leak.rule
                    && known.conclusion == leak.conclusion
                    && known.category == leak.category
            }) {
                leaks.push(leak);
            }
        }
        leaks.sort_by(|left, right| {
            left.max_precision
                .cmp(&right.max_precision)
                .then_with(|| left.rule.cmp(&right.rule))
        });
        (leaks, truncated)
    }
}
#[derive(Debug, Clone)]
struct WorkFact {
    fact: PrivacyFact,
    provenance: BTreeSet<usize>,
    depth: usize,
    rule: Option<String>,
    max_precision: DisclosurePrecision,
}
/// 一条被阻断的组合推导；只记录谓词，不记录取值。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InferenceLeak {
    pub rule: String,
    pub conclusion: String,
    pub category: PrivacyCategory,
    pub max_precision: DisclosurePrecision,
}
impl InferenceLeak {
    pub fn render(&self) -> String {
        format!(
            "{}（{} 可推出 {}，上限 {}）",
            self.rule,
            self.category.label(),
            self.conclusion,
            self.max_precision.label()
        )
    }
}
fn bucket_key(fact: &PrivacyFact, precision: DisclosurePrecision) -> String {
    match precision {
        DisclosurePrecision::Exact => format!("exact\u{1}{}", fact.object),
        DisclosurePrecision::Hidden => "hidden".to_string(),
        DisclosurePrecision::TypeOnly => typed_placeholder(fact.category, None).to_string(),
        DisclosurePrecision::Granular | DisclosurePrecision::Coarse => {
            coarsen_value(fact.category, precision, &fact.object, None)
                .unwrap_or_else(|| typed_placeholder(fact.category, None).to_string())
        }
    }
}

// ---------------------------------------------------------------------------
// 会话台账
// ---------------------------------------------------------------------------
/// 授权范围。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsentScope {
    Session,
    Once,
}
impl ConsentScope {
    pub const fn code(self) -> &'static str {
        match self {
            ConsentScope::Session => "session",
            ConsentScope::Once => "once",
        }
    }
}
/// 用户话语里的隐私言语行为。只做确定性关键词识别；主体由调用方的
/// 会话上下文给出，避免把「我 / 他」直接当成图谱实体名。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "act", rename_all = "snake_case")]
pub enum PrivacyAct {
    /// 询问策略本身：隐私政策 / 哪些信息不能说。
    AskPolicy,
    /// 允许披露某类信息。
    Grant {
        category: PrivacyCategory,
        scope: ConsentScope,
    },
    /// 撤回某类披露授权。
    Revoke { category: PrivacyCategory },
    /// 遗忘某类或全部会话信息。
    Forget {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        category: Option<PrivacyCategory>,
    },
    /// 与隐私策略无关的话语。
    Other,
}
impl PrivacyAct {
    pub fn parse(text: &str) -> Self {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return PrivacyAct::Other;
        }
        let category = PrivacyCategory::infer_from_text(trimmed);
        let interrogative = trimmed.ends_with('吗')
            || trimmed.ends_with('？')
            || trimmed.ends_with('?')
            || trimmed.contains("是否");
        let asks_policy = contains_any(
            trimmed,
            &[
                "隐私政策",
                "隐私规则",
                "哪些信息",
                "什么信息不能",
                "记得我什么",
                "你记得我",
                "能说什么",
                "不能说什么",
            ],
        );
        let forgets = contains_any(
            trimmed,
            &[
                "忘记",
                "忘掉",
                "删除",
                "抹掉",
                "清除",
                "不要再记",
                "不要记",
                "别记",
            ],
        );
        let negated = contains_any(
            trimmed,
            &[
                "不要",
                "不能",
                "不准",
                "不许",
                "不得",
                "别告诉",
                "别把",
                "别说",
                "别透露",
                "别公开",
                "别分享",
                "别记",
                "别再",
            ],
        );
        let disclosure_verb = contains_any(trimmed, &["告诉", "透露", "说出去", "公开", "分享"]);
        let revokes = negated && disclosure_verb
            || contains_any(trimmed, &["撤回", "收回", "取消授权", "不同意"]);
        let grants =
            contains_any(trimmed, &["可以告诉", "可以说", "同意", "允许", "授权"]) && !negated;

        if forgets {
            return PrivacyAct::Forget { category };
        }
        if revokes {
            return category
                .map(|category| PrivacyAct::Revoke { category })
                .unwrap_or(PrivacyAct::Other);
        }
        let asks_permission = interrogative
            && !negated
            && contains_any(trimmed, &["可以", "能", "同意", "允许", "授权", "告诉"]);
        if asks_policy || asks_permission {
            return PrivacyAct::AskPolicy;
        }
        if grants {
            let scope = if contains_any(trimmed, &["只此一次", "仅此一次", "这一次", "这次可以"])
            {
                ConsentScope::Once
            } else {
                ConsentScope::Session
            };
            return category
                .map(|category| PrivacyAct::Grant { category, scope })
                .unwrap_or(PrivacyAct::Other);
        }
        PrivacyAct::Other
    }

    pub fn code(&self) -> &'static str {
        match self {
            PrivacyAct::AskPolicy => "ask_policy",
            PrivacyAct::Grant { .. } => "grant",
            PrivacyAct::Revoke { .. } => "revoke",
            PrivacyAct::Forget { .. } => "forget",
            PrivacyAct::Other => "other",
        }
    }

    pub fn is_policy_act(&self) -> bool {
        !matches!(self, PrivacyAct::Other)
    }

    pub fn render(&self) -> String {
        match self {
            PrivacyAct::AskPolicy => "询问隐私策略".to_string(),
            PrivacyAct::Grant { category, scope } => {
                format!("允许披露{}（{}）", category.label(), scope.code())
            }
            PrivacyAct::Revoke { category } => format!("撤回{}授权", category.label()),
            PrivacyAct::Forget {
                category: Some(category),
            } => format!("遗忘{}", category.label()),
            PrivacyAct::Forget { category: None } => "遗忘全部会话信息".to_string(),
            PrivacyAct::Other => "非隐私言语行为".to_string(),
        }
    }
}
fn contains_any(text: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| text.contains(needle))
}
/// 一次显式授权或撤回。最新记录优先；撤回不会删除历史。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConsentRecord {
    pub subject: String,
    pub category: PrivacyCategory,
    pub purpose: String,
    pub scope: ConsentScope,
    pub granted: bool,
    #[serde(default)]
    pub consumed: bool,
    pub turn: u64,
}
/// 已披露内容的最小记录：只记槽位与精度，不记原值。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DisclosureRecord {
    pub subject: String,
    pub predicate: String,
    pub category: PrivacyCategory,
    pub purpose: String,
    pub precision: DisclosurePrecision,
    pub turn: u64,
    pub fact_digest: String,
}
/// 遗忘墓碑。字段全部为 `None` 的墓碑永远不会命中。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tombstone {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subject: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub predicate: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub category: Option<PrivacyCategory>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fact_digest: Option<String>,
    pub reason: String,
    pub turn: u64,
}
impl Tombstone {
    fn matches(
        &self,
        subject: &str,
        predicate: &str,
        category: PrivacyCategory,
        digest: &str,
    ) -> bool {
        let mut constrained = false;
        if let Some(wanted) = &self.subject {
            constrained = true;
            if wanted != subject {
                return false;
            }
        }
        if let Some(wanted) = &self.predicate {
            constrained = true;
            if wanted != predicate {
                return false;
            }
        }
        if let Some(wanted) = self.category {
            constrained = true;
            if wanted != category {
                return false;
            }
        }
        if let Some(wanted) = &self.fact_digest {
            constrained = true;
            if wanted != digest {
                return false;
            }
        }
        constrained
    }
}
/// 会话级隐私台账：授权、已披露精度与墓碑。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrivacyLedger {
    #[serde(default = "default_ledger_version")]
    pub version: u32,
    #[serde(default)]
    pub turn: u64,
    #[serde(default)]
    pub consents: Vec<ConsentRecord>,
    #[serde(default)]
    pub disclosures: Vec<DisclosureRecord>,
    #[serde(default)]
    pub tombstones: Vec<Tombstone>,
}
fn default_ledger_version() -> u32 {
    PRIVACY_LEDGER_VERSION
}
impl Default for PrivacyLedger {
    fn default() -> Self {
        Self {
            version: PRIVACY_LEDGER_VERSION,
            turn: 0,
            consents: Vec::new(),
            disclosures: Vec::new(),
            tombstones: Vec::new(),
        }
    }
}

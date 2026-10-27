//! PIFC：词性类型化的信息流契约（POS-typed Information Flow Contract）。
//!
//! 对话系统并不缺「能说什么」的零件，缺的是「这句话该不该说、说到多细」的
//! 语用裁决。本模块把可披露性放在生成之前，而不是事后做字符串替换：
//!
//! 1. **标注层**：复用 [`crate::nlp::AnnotatedWord`] 的词性、来源与字符区间，
//!    加上少量上下文规则（手机号、身份证、住址引导词等），产出带强弱证据的
//!    敏感跨度。结巴兜底只提示，不默认硬拦。
//! 2. **策略层**：谓词映射到披露类别，类别给出默认精度上限；显式授权可提高
//!    上限，墓碑（遗忘）永远优先。
//! 3. **推理层**：对「就读学校 + 学校位于 → 所在城区」这类组合推导做有界
//!    闭包；若当前事实是敏感结论的前提，就把披露精度收紧。
//! 4. **匿名层**：按 `k` 检查同一谓词下等价类中的不同数据主体数，稀疏图谱里
//!    唯一可识别的值会继续降级。
//! 5. **会话层**：`consent / disclosure / tombstone` 台账让「已模糊」的表达在
//!    后续轮次保持一致，并支持一次性授权与显式遗忘。
//! 6. **回执层**：只记录决策码、类别、精度与事实摘要，不把明文写进回执。
//!
//! 本模块只做最小披露与有界闭包，不声称差分隐私或形式化不可识别性。默认
//! 模式是 [`PolicyMode::Audit`]：影子裁决会算出「本该怎样」，但不改变回答，
//! 便于先度量误拦率再切到 [`PolicyMode::Enforce`]。

use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::function_words::{wh_kind, WhKind};
use crate::nlp::{AnnotatedWord, TagSource};

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
                &["父亲", "母亲", "父母", "配偶", "亲属", "家人", "朋友", "同事"],
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
pub fn redact_spans(
    text: &str,
    spans: &[SensitiveSpan],
    policy: &PrivacyPolicy,
) -> SpanRedaction {
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
            "隐私策略：模式 {}，k={}，闭包深度 {}，闭包原子上限 {}，弱证据{}硬拦；类别上限：{}；需授权：{}；敏感词 {} 条，推导规则 {} 条",
            self.mode.label(),
            self.k_anonymity,
            self.max_closure_depth,
            self.max_closure_atoms,
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
                "不要", "不能", "不准", "不许", "不得", "别告诉", "别把", "别说", "别透露",
                "别公开", "别分享", "别记", "别再",
            ],
        );
        let disclosure_verb = contains_any(trimmed, &["告诉", "透露", "说出去", "公开", "分享"]);
        let revokes = negated && disclosure_verb
            || contains_any(trimmed, &["撤回", "收回", "取消授权", "不同意"]);
        let grants = contains_any(
            trimmed,
            &["可以告诉", "可以说", "同意", "允许", "授权"],
        ) && !negated;

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
            let scope = if contains_any(trimmed, &["只此一次", "仅此一次", "这一次", "这次可以"]) {
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

impl PrivacyLedger {
    pub fn new() -> Self {
        Self::default()
    }

    /// 开启新的一轮；台账中的记录使用该轮号。
    pub fn begin_turn(&mut self) -> u64 {
        self.turn = self.turn.saturating_add(1);
        self.turn
    }

    pub fn grant(
        &mut self,
        subject: &str,
        category: PrivacyCategory,
        purpose: &str,
        scope: ConsentScope,
    ) {
        self.consents.push(ConsentRecord {
            subject: subject.to_string(),
            category,
            purpose: purpose.to_string(),
            scope,
            granted: true,
            consumed: false,
            turn: self.turn,
        });
    }

    pub fn revoke(&mut self, subject: &str, category: PrivacyCategory, purpose: &str) {
        self.consents.push(ConsentRecord {
            subject: subject.to_string(),
            category,
            purpose: purpose.to_string(),
            scope: ConsentScope::Session,
            granted: false,
            consumed: false,
            turn: self.turn,
        });
    }

    /// 把一条隐私言语行为落到台账上。`AskPolicy` 与 `Other` 不改变状态。
    pub fn apply_act(&mut self, subject: &str, purpose: &str, act: &PrivacyAct) {
        match act {
            PrivacyAct::Grant { category, scope } => {
                self.grant(subject, *category, purpose, *scope);
            }
            PrivacyAct::Revoke { category } => {
                self.revoke(subject, *category, purpose);
            }
            PrivacyAct::Forget {
                category: Some(category),
            } => self.forget_category(subject, *category, "对话中的遗忘请求"),
            PrivacyAct::Forget { category: None } => {
                self.forget_subject(subject, "对话中的遗忘请求")
            }
            PrivacyAct::AskPolicy | PrivacyAct::Other => {}
        }
    }

    pub fn forget_subject(&mut self, subject: &str, reason: &str) {
        self.tombstones.push(Tombstone {
            subject: Some(subject.to_string()),
            predicate: None,
            category: None,
            fact_digest: None,
            reason: reason.to_string(),
            turn: self.turn,
        });
    }

    pub fn forget_category(&mut self, subject: &str, category: PrivacyCategory, reason: &str) {
        self.tombstones.push(Tombstone {
            subject: Some(subject.to_string()),
            predicate: None,
            category: Some(category),
            fact_digest: None,
            reason: reason.to_string(),
            turn: self.turn,
        });
    }

    pub fn forget_fact(&mut self, subject: &str, predicate: &str, object: &str, reason: &str) {
        self.tombstones.push(Tombstone {
            subject: Some(subject.to_string()),
            predicate: Some(predicate.to_string()),
            category: None,
            fact_digest: Some(fact_digest(subject, predicate, object)),
            reason: reason.to_string(),
            turn: self.turn,
        });
    }

    pub fn is_tombstoned(
        &self,
        subject: &str,
        predicate: &str,
        object: &str,
        category: PrivacyCategory,
    ) -> bool {
        let digest = fact_digest(subject, predicate, object);
        self.tombstones
            .iter()
            .any(|tombstone| tombstone.matches(subject, predicate, category, &digest))
    }

    /// 返回可用的显式授权（一次性授权未消费才算可用）。
    pub fn consent_permits(&self, subject: &str, category: PrivacyCategory, purpose: &str) -> bool {
        for record in self.consents.iter().rev() {
            if record.subject != subject || record.category != category {
                continue;
            }
            if record.purpose != purpose && record.purpose != "*" {
                continue;
            }
            if !record.granted {
                return false;
            }
            if record.scope == ConsentScope::Once && record.consumed {
                continue;
            }
            return true;
        }
        false
    }

    fn consume_consent(&mut self, subject: &str, category: PrivacyCategory, purpose: &str) {
        for record in self.consents.iter_mut().rev() {
            if record.subject != subject || record.category != category || !record.granted {
                continue;
            }
            if record.purpose != purpose && record.purpose != "*" {
                continue;
            }
            if record.scope == ConsentScope::Once && !record.consumed {
                record.consumed = true;
                return;
            }
        }
    }

    /// 同一槽位此前披露过的最粗精度；后续轮次不得更细。
    pub fn disclosure_cap(&self, subject: &str, predicate: &str) -> Option<DisclosurePrecision> {
        self.disclosures
            .iter()
            .filter(|record| record.subject == subject && record.predicate == predicate)
            .map(|record| record.precision)
            .min()
    }

    fn note_disclosure(
        &mut self,
        subject: &str,
        predicate: &str,
        object: &str,
        category: PrivacyCategory,
        purpose: &str,
        precision: DisclosurePrecision,
    ) {
        self.disclosures.push(DisclosureRecord {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            category,
            purpose: purpose.to_string(),
            precision,
            turn: self.turn,
            fact_digest: fact_digest(subject, predicate, object),
        });
    }

    pub fn canonical_json(&self) -> Result<String, String> {
        crate::canonical::canonical_text(self).map_err(|error| error.to_string())
    }

    pub fn from_json(text: &str) -> Result<Self, String> {
        let ledger: Self = serde_json::from_str(text).map_err(|error| error.to_string())?;
        if ledger.version > PRIVACY_LEDGER_VERSION {
            return Err(format!(
                "隐私台账版本 {} 高于支持版本 {}",
                ledger.version, PRIVACY_LEDGER_VERSION
            ));
        }
        Ok(ledger)
    }

    pub fn digest(&self) -> String {
        crate::canonical::canonical_digest(self).unwrap_or_default()
    }

    pub fn render_text(&self) -> String {
        let mut output = String::new();
        output.push_str(&format!(
            "隐私台账：第 {} 轮，授权 {} 条，披露 {} 条，墓碑 {} 条\n",
            self.turn,
            self.consents.len(),
            self.disclosures.len(),
            self.tombstones.len()
        ));
        for record in &self.consents {
            output.push_str(&format!(
                "  授权：{} 的{}，用途「{}」，范围 {}，{}\n",
                record.subject,
                record.category.label(),
                record.purpose,
                record.scope.code(),
                if record.granted {
                    if record.consumed {
                        "已消费"
                    } else {
                        "有效"
                    }
                } else {
                    "已撤回"
                }
            ));
        }
        for record in &self.disclosures {
            output.push_str(&format!(
                "  披露：{} {}（{}，{}）\n",
                record.subject,
                record.predicate,
                record.category.label(),
                record.precision.label()
            ));
        }
        for tombstone in &self.tombstones {
            output.push_str(&format!(
                "  墓碑：{}{}{}（{}）\n",
                tombstone.subject.as_deref().unwrap_or("*"),
                tombstone
                    .category
                    .map(|category| format!("/{}", category.code()))
                    .unwrap_or_default(),
                tombstone
                    .predicate
                    .as_deref()
                    .map(|predicate| format!("/{}", predicate))
                    .unwrap_or_default(),
                tombstone.reason
            ));
        }
        output
    }
}

// ---------------------------------------------------------------------------
// 裁决
// ---------------------------------------------------------------------------

/// 一次披露请求。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DisclosureRequest {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub purpose: String,
    pub requested_precision: DisclosurePrecision,
    pub wh: Option<WhKind>,
    pub k_required: Option<usize>,
}

impl DisclosureRequest {
    pub fn new(subject: &str, predicate: &str, object: &str) -> Self {
        Self {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
            purpose: "问答".to_string(),
            requested_precision: DisclosurePrecision::Exact,
            wh: None,
            k_required: None,
        }
    }

    pub fn with_purpose(mut self, purpose: &str) -> Self {
        self.purpose = purpose.to_string();
        self
    }

    pub fn with_precision(mut self, precision: DisclosurePrecision) -> Self {
        self.requested_precision = precision;
        self
    }

    pub fn with_question(mut self, question: &str) -> Self {
        self.wh = wh_kind(question);
        self
    }

    pub fn with_k(mut self, k: usize) -> Self {
        self.k_required = Some(k);
        self
    }
}

/// 裁决结论。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DisclosureStatus {
    Allowed,
    Downgraded,
    NeedsConsent,
    Blocked,
}

impl DisclosureStatus {
    pub const fn code(self) -> &'static str {
        match self {
            DisclosureStatus::Allowed => "allowed",
            DisclosureStatus::Downgraded => "downgraded",
            DisclosureStatus::NeedsConsent => "needs_consent",
            DisclosureStatus::Blocked => "blocked",
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            DisclosureStatus::Allowed => "允许",
            DisclosureStatus::Downgraded => "降级披露",
            DisclosureStatus::NeedsConsent => "需要授权",
            DisclosureStatus::Blocked => "阻断",
        }
    }
}

/// 稳定决策码，供回执与指标使用。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionCode {
    AuditShadow,
    Tombstoned,
    ConsentRequired,
    ConsentGranted,
    KAnonymity,
    InferenceLeak,
    PriorDisclosure,
    PolicyCeiling,
    WeakEvidence,
    ClosureTruncated,
    WhMismatch,
}

impl DecisionCode {
    pub const fn code(self) -> &'static str {
        match self {
            DecisionCode::AuditShadow => "audit_shadow",
            DecisionCode::Tombstoned => "tombstoned",
            DecisionCode::ConsentRequired => "consent_required",
            DecisionCode::ConsentGranted => "consent_granted",
            DecisionCode::KAnonymity => "k_anonymity",
            DecisionCode::InferenceLeak => "inference_leak",
            DecisionCode::PriorDisclosure => "prior_disclosure",
            DecisionCode::PolicyCeiling => "policy_ceiling",
            DecisionCode::WeakEvidence => "weak_evidence",
            DecisionCode::ClosureTruncated => "closure_truncated",
            DecisionCode::WhMismatch => "wh_mismatch",
        }
    }
}

/// 审计模式下「如果执行会怎样」的影子结论。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditFinding {
    pub would_be_status: DisclosureStatus,
    pub would_be_precision: DisclosurePrecision,
    pub codes: Vec<DecisionCode>,
}

/// 只含决策码与摘要的隐私回执，不落明文。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrivacyReceipt {
    pub mode: PolicyMode,
    pub category: PrivacyCategory,
    pub status: DisclosureStatus,
    pub requested_precision: DisclosurePrecision,
    pub effective_precision: DisclosurePrecision,
    pub simulated_status: DisclosureStatus,
    pub simulated_precision: DisclosurePrecision,
    pub codes: Vec<String>,
    pub fact_digest: String,
    pub ledger_digest: String,
    pub turn: u64,
}

/// 批量审计摘要。`Audit` 模式使用影子结论统计，`Enforce` 模式使用实际结论，
/// 因而同一份策略可以先在影子模式度量公开事实保留率与误拦率。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditSummary {
    pub total: usize,
    pub allowed: usize,
    pub downgraded: usize,
    pub needs_consent: usize,
    pub blocked: usize,
    /// 非敏感（`Other`）事实数：它们本应原样回答。
    pub public_total: usize,
    pub public_retained: usize,
    /// 非敏感事实中被降级或阻断的数量。
    pub public_changed: usize,
    pub sensitive_total: usize,
    pub sensitive_retained: usize,
    pub closure_truncated: usize,
    pub code_counts: BTreeMap<String, usize>,
}

impl AuditSummary {
    pub fn from_decisions<'a>(
        decisions: impl IntoIterator<Item = &'a DisclosureDecision>,
    ) -> Self {
        let mut summary = Self::default();
        for decision in decisions {
            summary.total += 1;
            let (status, precision) = match &decision.audit {
                Some(audit) => (audit.would_be_status, audit.would_be_precision),
                None => (decision.status, decision.effective_precision),
            };
            match status {
                DisclosureStatus::Allowed => summary.allowed += 1,
                DisclosureStatus::Downgraded => summary.downgraded += 1,
                DisclosureStatus::NeedsConsent => summary.needs_consent += 1,
                DisclosureStatus::Blocked => summary.blocked += 1,
            }
            let retained = precision == decision.requested_precision;
            if decision.category == PrivacyCategory::Other {
                summary.public_total += 1;
                if retained {
                    summary.public_retained += 1;
                } else {
                    summary.public_changed += 1;
                }
            } else {
                summary.sensitive_total += 1;
                if retained {
                    summary.sensitive_retained += 1;
                }
            }
            let codes = match &decision.audit {
                Some(audit) => audit.codes.as_slice(),
                None => decision.codes.as_slice(),
            };
            for code in codes {
                *summary
                    .code_counts
                    .entry(code.code().to_string())
                    .or_insert(0) += 1;
                if *code == DecisionCode::ClosureTruncated {
                    summary.closure_truncated += 1;
                }
            }
        }
        summary
    }

    /// 公开事实按请求精度保留的比例；没有公开事实时视为 1。
    pub fn public_retention(&self) -> f64 {
        ratio(self.public_retained, self.public_total)
    }

    /// 公开事实被误拦的比例；没有公开事实时视为 0。
    pub fn false_block_rate(&self) -> f64 {
        if self.public_total == 0 {
            0.0
        } else {
            (self.public_changed as f64) / (self.public_total as f64)
        }
    }

    pub fn render(&self) -> String {
        let mut text = format!(
            "审计摘要：{} 条；允许 {}，降级 {}，待授权 {}，阻断 {}；公开事实保留 {}/{}（{:.1}%），误降级/误拦 {:.1}%；敏感事实保持请求精度 {}/{}",
            self.total,
            self.allowed,
            self.downgraded,
            self.needs_consent,
            self.blocked,
            self.public_retained,
            self.public_total,
            self.public_retention() * 100.0,
            self.false_block_rate() * 100.0,
            self.sensitive_retained,
            self.sensitive_total
        );
        if self.closure_truncated > 0 {
            text.push_str(&format!(
                "；闭包截断 {} 条（按保守结论计）",
                self.closure_truncated
            ));
        }
        if !self.code_counts.is_empty() {
            let codes = self
                .code_counts
                .iter()
                .map(|(code, count)| format!("{}={}", code, count))
                .collect::<Vec<_>>()
                .join(" ");
            text.push_str(&format!("；决策码：{}", codes));
        }
        text
    }
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        1.0
    } else {
        (numerator as f64) / (denominator as f64)
    }
}

/// 一次披露的完整裁决。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DisclosureDecision {
    pub status: DisclosureStatus,
    pub requested_precision: DisclosurePrecision,
    pub effective_precision: DisclosurePrecision,
    pub category: PrivacyCategory,
    pub replacement: Option<String>,
    pub codes: Vec<DecisionCode>,
    pub inference_leaks: Vec<InferenceLeak>,
    pub audit: Option<AuditFinding>,
    pub receipt: PrivacyReceipt,
}

impl DisclosureDecision {
    pub fn replacement_or(&self, original: &str) -> String {
        self.replacement
            .clone()
            .unwrap_or_else(|| original.to_string())
    }

    /// 把裁决接到生成叠加层：隐藏整个槽位，或用类型化表达替换原值。
    ///
    /// 注意：`Audit` 模式的 `effective_precision` 等于请求精度，因此不会改动
    /// 生成结果；只有 `Enforce` 模式才会真正写入 overlay。
    pub fn apply_to_overlay(
        &self,
        overlay: &mut crate::symbolic_generator::GenerationOverlay,
        subject: &str,
        predicate: &str,
        original: &str,
    ) {
        if self.status == DisclosureStatus::Blocked
            || self.effective_precision == DisclosurePrecision::Hidden
        {
            overlay
                .hide
                .insert((subject.to_string(), predicate.to_string()));
            return;
        }
        if self.effective_precision == DisclosurePrecision::Exact {
            return;
        }
        let replacement = self
            .replacement
            .clone()
            .unwrap_or_else(|| original.to_string());
        if replacement != original {
            overlay.replace.insert(
                (subject.to_string(), predicate.to_string()),
                vec![replacement],
            );
        }
    }
}

/// 隐私裁决器：策略 + 图谱 + 可变的会话台账。
pub struct PrivacyAdjudicator<'a> {
    policy: &'a PrivacyPolicy,
    graph: &'a PrivacyGraph,
    ledger: &'a mut PrivacyLedger,
}

impl<'a> PrivacyAdjudicator<'a> {
    pub fn new(
        policy: &'a PrivacyPolicy,
        graph: &'a PrivacyGraph,
        ledger: &'a mut PrivacyLedger,
    ) -> Self {
        Self {
            policy,
            graph,
            ledger,
        }
    }

    pub fn adjudicate(&mut self, request: &DisclosureRequest) -> DisclosureDecision {
        let fact = self
            .graph
            .find(&request.subject, &request.predicate, &request.object)
            .cloned()
            .unwrap_or_else(|| {
                PrivacyFact::label(
                    self.policy,
                    &request.subject,
                    &request.predicate,
                    &request.object,
                )
            });
        let category = fact.category;
        let requested = fact.precision.min(request.requested_precision);
        let explicit_consent =
            self.ledger
                .consent_permits(&fact.subject, category, &request.purpose);
        let mut effective = requested;
        let mut codes: Vec<DecisionCode> = Vec::new();
        let mut inference_leaks: Vec<InferenceLeak> = Vec::new();

        if !category.compatible_with(request.wh) {
            add_code(&mut codes, DecisionCode::WhMismatch);
        }

        if self
            .ledger
            .is_tombstoned(&fact.subject, &fact.predicate, &fact.object, category)
        {
            effective = DisclosurePrecision::Hidden;
            add_code(&mut codes, DecisionCode::Tombstoned);
        } else {
            if !explicit_consent {
                let ceiling = self
                    .policy
                    .max_precision_for_predicate(&fact.predicate)
                    .unwrap_or_else(|| self.policy.max_precision(category));
                if effective > ceiling {
                    effective = ceiling;
                    add_code(&mut codes, DecisionCode::PolicyCeiling);
                }
                if should_k_anonymize(category, self.policy) {
                    let k = request.k_required.unwrap_or(self.policy.k_anonymity).max(1);
                    while effective != DisclosurePrecision::Hidden
                        && self.graph.class_size(&fact, effective) < k
                    {
                        effective = effective.coarsen();
                        add_code(&mut codes, DecisionCode::KAnonymity);
                    }
                }
            }

            let (leaks, truncated) = self.graph.inference_leaks(&fact, self.policy, effective);
            if truncated {
                add_code(&mut codes, DecisionCode::ClosureTruncated);
            }
            if !leaks.is_empty() {
                let cap = leaks
                    .iter()
                    .map(|leak| leak.max_precision)
                    .min()
                    .unwrap_or(effective);
                if cap < effective {
                    effective = cap;
                    add_code(&mut codes, DecisionCode::InferenceLeak);
                }
                inference_leaks = leaks;
            }

            if self.policy.requires_consent(category) && requested > DisclosurePrecision::TypeOnly {
                if explicit_consent {
                    add_code(&mut codes, DecisionCode::ConsentGranted);
                } else {
                    effective = DisclosurePrecision::TypeOnly.min(effective);
                    add_code(&mut codes, DecisionCode::ConsentRequired);
                }
            }

            if !explicit_consent {
                if let Some(previous) = self.ledger.disclosure_cap(&fact.subject, &fact.predicate) {
                    if previous < effective {
                        effective = previous;
                        add_code(&mut codes, DecisionCode::PriorDisclosure);
                    }
                }
            }
        }

        let would_be_status = if effective == DisclosurePrecision::Hidden {
            if requested == DisclosurePrecision::Hidden {
                DisclosureStatus::Allowed
            } else {
                DisclosureStatus::Blocked
            }
        } else if codes.contains(&DecisionCode::ConsentRequired) {
            DisclosureStatus::NeedsConsent
        } else if effective < requested {
            DisclosureStatus::Downgraded
        } else {
            DisclosureStatus::Allowed
        };

        let audit_mode = self.policy.mode == PolicyMode::Audit;
        let (status, effective_precision, replacement) = if audit_mode {
            add_code(&mut codes, DecisionCode::AuditShadow);
            (
                DisclosureStatus::Allowed,
                requested,
                Some(fact.object.clone()),
            )
        } else {
            let replacement_category = inference_leaks
                .iter()
                .map(|leak| leak.category)
                .find(|candidate| request.wh.is_some() && candidate.compatible_with(request.wh))
                .unwrap_or(category);
            let replacement = match would_be_status {
                DisclosureStatus::Blocked => None,
                _ if effective == DisclosurePrecision::Exact => Some(fact.object.clone()),
                _ => coarsen_value(replacement_category, effective, &fact.object, request.wh),
            };
            (would_be_status, effective, replacement)
        };

        if !audit_mode && would_be_status != DisclosureStatus::Blocked {
            self.ledger.note_disclosure(
                &fact.subject,
                &fact.predicate,
                &fact.object,
                category,
                &request.purpose,
                effective,
            );
            if explicit_consent {
                self.ledger
                    .consume_consent(&fact.subject, category, &request.purpose);
            }
        }

        let receipt = PrivacyReceipt {
            mode: self.policy.mode,
            category,
            status,
            requested_precision: requested,
            effective_precision,
            simulated_status: would_be_status,
            simulated_precision: effective,
            codes: codes.iter().map(|code| code.code().to_string()).collect(),
            fact_digest: fact.fact_digest(),
            ledger_digest: self.ledger.digest(),
            turn: self.ledger.turn,
        };
        let audit = if audit_mode {
            Some(AuditFinding {
                would_be_status,
                would_be_precision: effective,
                codes: codes
                    .iter()
                    .copied()
                    .filter(|code| *code != DecisionCode::AuditShadow)
                    .collect(),
            })
        } else {
            None
        };

        DisclosureDecision {
            status,
            requested_precision: requested,
            effective_precision,
            category,
            replacement,
            codes,
            inference_leaks,
            audit,
            receipt,
        }
    }
}

fn add_code(codes: &mut Vec<DecisionCode>, code: DecisionCode) {
    if !codes.contains(&code) {
        codes.push(code);
    }
}

fn should_k_anonymize(category: PrivacyCategory, policy: &PrivacyPolicy) -> bool {
    category.is_personal()
        && category != PrivacyCategory::PersonalIdentity
        && policy.max_precision(category) < DisclosurePrecision::Exact
}

// ---------------------------------------------------------------------------
// 类型保持改写
// ---------------------------------------------------------------------------

/// 按披露类别与疑问类型给出占位表达。类别优先保证槽位语义，疑问类型
/// 只在该类别本身没有更具体表达时兜底。
pub fn typed_placeholder(category: PrivacyCategory, wh: Option<WhKind>) -> &'static str {
    match category {
        PrivacyCategory::PersonalIdentity => match wh {
            Some(WhKind::Person) | None => "某位人士",
            _ => "某人的身份",
        },
        PrivacyCategory::Contact => "某联系方式",
        PrivacyCategory::Credential => "某证件信息",
        PrivacyCategory::Location => "某地",
        PrivacyCategory::Health => "某健康状况",
        PrivacyCategory::Finance => "某财务信息",
        PrivacyCategory::Demographic => match wh {
            Some(WhKind::Time) => "某个时间段",
            _ => "某个年龄范围",
        },
        PrivacyCategory::Relationship => "某位亲友",
        PrivacyCategory::Other => match wh {
            Some(WhKind::Person) => "某位人士",
            Some(WhKind::Place) => "某地",
            Some(WhKind::Time) => "某个时间",
            Some(WhKind::Quantity) => "某个范围",
            Some(WhKind::Reason) => "相关原因（已省略）",
            Some(WhKind::Thing) | None => "相关信息",
        },
    }
}

/// 类型保持改写：同一类别在同一精度下输出稳定表达，便于多轮指代。
pub fn coarsen_value(
    category: PrivacyCategory,
    precision: DisclosurePrecision,
    value: &str,
    wh: Option<WhKind>,
) -> Option<String> {
    if value.trim().is_empty() {
        return None;
    }
    match precision {
        DisclosurePrecision::Exact => Some(value.to_string()),
        DisclosurePrecision::Hidden => None,
        DisclosurePrecision::TypeOnly => Some(typed_placeholder(category, wh).to_string()),
        DisclosurePrecision::Granular => match category {
            PrivacyCategory::Location => cut_through(value, &['区', '县'])
                .or_else(|| cut_through(value, &['市', '省']))
                .or_else(|| Some(typed_placeholder(category, wh).to_string())),
            PrivacyCategory::Demographic => coarsen_demographic(value, 5),
            PrivacyCategory::PersonalIdentity => {
                let surname: String = value.chars().take(1).collect();
                if surname.is_empty() {
                    Some(typed_placeholder(category, wh).to_string())
                } else {
                    Some(format!("{}某", surname))
                }
            }
            PrivacyCategory::Contact => Some(mask_contact(value)),
            PrivacyCategory::Credential => Some(mask_credential(value)),
            PrivacyCategory::Finance => Some(mask_finance(value)),
            PrivacyCategory::Health => Some("某类健康状况".to_string()),
            PrivacyCategory::Relationship => Some("某位亲友".to_string()),
            PrivacyCategory::Other => Some(value.to_string()),
        },
        DisclosurePrecision::Coarse => match category {
            PrivacyCategory::Location => cut_through(value, &['省'])
                .or_else(|| cut_through(value, &['市']))
                .or_else(|| Some(typed_placeholder(category, wh).to_string())),
            PrivacyCategory::Demographic => coarsen_demographic(value, 10),
            PrivacyCategory::PersonalIdentity => Some(typed_placeholder(category, wh).to_string()),
            PrivacyCategory::Contact => Some("某联系方式".to_string()),
            PrivacyCategory::Credential => Some("某证件信息".to_string()),
            PrivacyCategory::Finance => Some("某财务信息".to_string()),
            PrivacyCategory::Health => Some("某健康状况".to_string()),
            PrivacyCategory::Relationship => Some("某位亲友".to_string()),
            PrivacyCategory::Other => Some(value.to_string()),
        },
    }
}

fn cut_through(value: &str, suffixes: &[char]) -> Option<String> {
    for (index, character) in value.char_indices() {
        if suffixes.contains(&character) {
            return Some(value[..index + character.len_utf8()].to_string());
        }
    }
    None
}

fn coarsen_demographic(value: &str, width: u32) -> Option<String> {
    if value.contains('岁') {
        if let Some(age) = extract_digits(value) {
            if age > 130 {
                return Some(
                    typed_placeholder(PrivacyCategory::Demographic, Some(WhKind::Quantity))
                        .to_string(),
                );
            }
            let start = (age / width) * width;
            return Some(format!("{}-{}岁", start, start + width.saturating_sub(1)));
        }
    }
    if let Some(year) = extract_year(value) {
        if (1000..=2999).contains(&year) {
            if width <= 5 {
                return Some(format!("{}年", year));
            }
            return Some(format!("{}年代", year / 10 * 10));
        }
    }
    Some(typed_placeholder(PrivacyCategory::Demographic, Some(WhKind::Quantity)).to_string())
}

fn extract_year(value: &str) -> Option<u32> {
    let digits: String = value
        .chars()
        .filter(|character| character.is_ascii_digit())
        .collect();
    if digits.len() < 4 {
        return None;
    }
    digits[..4].parse().ok()
}

fn extract_digits(value: &str) -> Option<u32> {
    let digits: String = value
        .chars()
        .filter(|character| character.is_ascii_digit())
        .collect();
    if digits.is_empty() {
        None
    } else {
        digits.parse().ok()
    }
}

fn mask_contact(value: &str) -> String {
    if let Some((local, domain)) = value.split_once('@') {
        let first: String = local.chars().take(1).collect();
        return format!("{}***@{}", first, domain);
    }
    let chars: Vec<char> = value.chars().collect();
    if chars.len() >= 7 {
        let head: String = chars.iter().take(3).collect();
        let tail: String = chars.iter().skip(chars.len() - 2).collect();
        format!("{}****{}", head, tail)
    } else {
        "某联系方式".to_string()
    }
}

fn mask_credential(value: &str) -> String {
    let chars: Vec<char> = value.chars().collect();
    if chars.len() >= 6 {
        let head: String = chars.iter().take(3).collect();
        let tail: String = chars.iter().skip(chars.len() - 1).collect();
        format!("{}****{}", head, tail)
    } else {
        "某证件信息".to_string()
    }
}

fn mask_finance(value: &str) -> String {
    let chars: Vec<char> = value.chars().collect();
    if chars.len() >= 8 {
        let tail: String = chars.iter().skip(chars.len() - 4).collect();
        format!("****{}", tail)
    } else {
        "某财务信息".to_string()
    }
}

fn fact_digest(subject: &str, predicate: &str, object: &str) -> String {
    let mut material = String::with_capacity(subject.len() + predicate.len() + object.len() + 2);
    material.push_str(subject);
    material.push('\u{1}');
    material.push_str(predicate);
    material.push('\u{1}');
    material.push_str(object);
    crate::context_image::fnv1a_hex(material.as_bytes())
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> PrivacyPolicy {
        PrivacyPolicy {
            mode: PolicyMode::Enforce,
            k_anonymity: 1,
            ..PrivacyPolicy::default()
        }
    }

    fn ledger() -> PrivacyLedger {
        let mut ledger = PrivacyLedger::new();
        ledger.begin_turn();
        ledger
    }

    fn annotate(text: &str) -> Vec<AnnotatedWord> {
        crate::nlp::preprocess_annotated(text, &test_nlp_config())
    }

    fn test_nlp_config() -> crate::NlpConfig {
        crate::NlpConfig {
            import_keywords: vec!["录入".to_string()],
            pos_dict: Default::default(),
            name_patterns: crate::NamePatterns {
                prefixes: vec!["小".to_string()],
                suffixes: vec!["总".to_string()],
                full_names: vec!["张三".to_string()],
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
    fn strong_regexes_find_contact_credential_and_age() {
        let policy = PrivacyPolicy::default();
        let detector = SpanDetector::new(&policy);
        let text = "电话13800138000，邮箱a.b@example.com，身份证110101199001011234，今年18岁";
        let spans = detector.scan(text, &annotate(text));

        let contact = spans
            .iter()
            .find(|span| span.text == "13800138000")
            .expect("手机号");
        assert_eq!(contact.category, PrivacyCategory::Contact);
        assert_eq!(contact.strength, EvidenceStrength::Strong);
        assert!(contact.is_blocking(&policy));

        assert!(spans.iter().any(
            |span| span.category == PrivacyCategory::Contact && span.text == "a.b@example.com"
        ));
        assert!(spans
            .iter()
            .any(|span| span.category == PrivacyCategory::Credential
                && span.text == "110101199001011234"));
        assert!(spans
            .iter()
            .any(|span| span.category == PrivacyCategory::Demographic && span.text.contains("18")));
    }

    #[test]
    fn names_are_weak_evidence_and_not_blocking_by_default() {
        let policy = PrivacyPolicy::default();
        let detector = SpanDetector::new(&policy);
        let text = "张三在北京工作";
        let spans = detector.scan(text, &annotate(text));
        assert!(spans.iter().any(|span| {
            span.text == "张三"
                && span.category == PrivacyCategory::PersonalIdentity
                && span.strength == EvidenceStrength::Weak
                && !span.is_blocking(&policy)
        }));
    }

    #[test]
    fn contextual_address_is_strong() {
        let policy = PrivacyPolicy::default();
        let detector = SpanDetector::new(&policy);
        let text = "他住在北京市海淀区中关村大街1号";
        let spans = detector.scan(text, &annotate(text));
        let address = spans
            .iter()
            .find(|span| {
                span.category == PrivacyCategory::Location
                    && span.strength == EvidenceStrength::Strong
            })
            .expect("住址");
        assert!(address.text.contains("海淀区"));
    }

    #[test]
    fn redact_replaces_strong_spans_and_keeps_weak_names() {
        let policy = policy();
        let detector = SpanDetector::new(&policy);
        let text = "张三住在北京市海淀区中关村大街1号，电话13800138000。";
        let redaction = detector.redact(text, &annotate(text));

        assert!(redaction.changed());
        assert!(redaction.text.contains("张三"), "默认不对人名脱敏");
        assert!(redaction.text.contains("北京市"));
        assert!(!redaction.text.contains("海淀区"));
        assert!(!redaction.text.contains("13800138000"));
        assert!(redaction.text.contains("某联系方式"));
        assert!(redaction.text.starts_with("张三住在北京市，电话"));
        assert!(redaction.kept_weak >= 1);
        assert!(redaction
            .records
            .iter()
            .any(|record| record.category == PrivacyCategory::Contact
                && record.precision == DisclosurePrecision::TypeOnly));
    }

    #[test]
    fn redact_honors_policy_precision_and_weak_blocking() {
        let mut policy = policy();
        policy.category_precision.insert(
            PrivacyCategory::Location,
            DisclosurePrecision::Granular,
        );
        policy.weak_evidence_blocks = true;
        let detector = SpanDetector::new(&policy);
        let text = "张三住在北京市海淀区中关村大街1号。";
        let redaction = detector.redact(text, &annotate(text));

        assert!(redaction.text.contains("北京市海淀区"));
        assert!(!redaction.text.contains("中关村"));
        assert_eq!(
            redaction.kept_weak, 0,
            "弱证据被提升为硬证据后不再原样保留"
        );
        assert!(!redaction.text.contains("张三"));
        assert!(redaction.text.contains("某位人士"));
    }

    #[test]
    fn redact_with_no_spans_is_identity() {
        let policy = policy();
        let detector = SpanDetector::new(&policy);
        let text = "今天天气不错。";
        let redaction = detector.redact(text, &annotate(text));
        assert_eq!(redaction.text, text);
        assert!(!redaction.changed());
        assert!(redaction.records.is_empty());
    }

    #[test]
    fn audit_mode_reports_block_without_changing_answer_or_ledger() {
        let mut policy = PrivacyPolicy::default();
        policy.mode = PolicyMode::Audit;
        policy.k_anonymity = 1;
        let mut graph = PrivacyGraph::new();
        graph.add_fact(&policy, "张三", "电话", "13800138000");
        let mut ledger = ledger();
        let before = ledger.digest();
        let mut adjudicator = PrivacyAdjudicator::new(&policy, &graph, &mut ledger);
        let decision =
            adjudicator.adjudicate(&DisclosureRequest::new("张三", "电话", "13800138000"));

        assert_eq!(decision.status, DisclosureStatus::Allowed);
        assert_eq!(decision.effective_precision, DisclosurePrecision::Exact);
        assert_eq!(
            decision.replacement.as_deref(),
            Some("13800138000"),
            "影子模式绝不能改写回答"
        );
        let audit = decision.audit.expect("审计结论");
        assert_eq!(audit.would_be_status, DisclosureStatus::NeedsConsent);
        assert_eq!(audit.would_be_precision, DisclosurePrecision::TypeOnly);
        assert_eq!(ledger.digest(), before, "审计模式不写台账");
    }

    #[test]
    fn enforce_requires_consent_and_uses_typed_placeholder() {
        let policy = policy();
        let mut graph = PrivacyGraph::new();
        graph.add_fact(&policy, "张三", "电话", "13800138000");
        let mut ledger = ledger();
        let mut adjudicator = PrivacyAdjudicator::new(&policy, &graph, &mut ledger);
        let decision =
            adjudicator.adjudicate(&DisclosureRequest::new("张三", "电话", "13800138000"));

        assert_eq!(decision.status, DisclosureStatus::NeedsConsent);
        assert_eq!(decision.effective_precision, DisclosurePrecision::TypeOnly);
        assert_eq!(decision.replacement.as_deref(), Some("某联系方式"));
        assert!(decision.codes.contains(&DecisionCode::ConsentRequired));
        assert!(!decision
            .receipt
            .codes
            .iter()
            .any(|code| code == "audit_shadow"));
        assert!(!serde_json::to_string(&decision.receipt)
            .unwrap()
            .contains("13800138000"));
    }

    #[test]
    fn one_time_consent_allows_exact_once_then_falls_back() {
        let policy = policy();
        let mut graph = PrivacyGraph::new();
        graph.add_fact(&policy, "张三", "电话", "13800138000");
        let mut ledger = ledger();
        ledger.grant("张三", PrivacyCategory::Contact, "问答", ConsentScope::Once);
        let mut adjudicator = PrivacyAdjudicator::new(&policy, &graph, &mut ledger);
        let request = DisclosureRequest::new("张三", "电话", "13800138000");
        let first = adjudicator.adjudicate(&request);
        assert_eq!(first.status, DisclosureStatus::Allowed);
        assert_eq!(first.effective_precision, DisclosurePrecision::Exact);
        assert!(first.codes.contains(&DecisionCode::ConsentGranted));

        let second = adjudicator.adjudicate(&request);
        assert_eq!(second.status, DisclosureStatus::NeedsConsent);
        assert_eq!(second.effective_precision, DisclosurePrecision::TypeOnly);
    }

    #[test]
    fn session_consent_can_be_revoked() {
        let policy = policy();
        let mut graph = PrivacyGraph::new();
        graph.add_fact(&policy, "张三", "病史", "高血压");
        let mut ledger = ledger();
        ledger.grant(
            "张三",
            PrivacyCategory::Health,
            "问答",
            ConsentScope::Session,
        );
        let mut adjudicator = PrivacyAdjudicator::new(&policy, &graph, &mut ledger);
        let request = DisclosureRequest::new("张三", "病史", "高血压");
        assert_eq!(
            adjudicator.adjudicate(&request).status,
            DisclosureStatus::Allowed
        );
        ledger.revoke("张三", PrivacyCategory::Health, "问答");
        let mut adjudicator = PrivacyAdjudicator::new(&policy, &graph, &mut ledger);
        assert_eq!(
            adjudicator.adjudicate(&request).status,
            DisclosureStatus::NeedsConsent
        );
    }

    #[test]
    fn tombstone_overrides_consent() {
        let policy = policy();
        let mut graph = PrivacyGraph::new();
        graph.add_fact(&policy, "张三", "电话", "13800138000");
        let mut ledger = ledger();
        ledger.grant(
            "张三",
            PrivacyCategory::Contact,
            "问答",
            ConsentScope::Session,
        );
        ledger.forget_subject("张三", "用户要求遗忘");
        let mut adjudicator = PrivacyAdjudicator::new(&policy, &graph, &mut ledger);
        let decision =
            adjudicator.adjudicate(&DisclosureRequest::new("张三", "电话", "13800138000"));
        assert_eq!(decision.status, DisclosureStatus::Blocked);
        assert_eq!(decision.effective_precision, DisclosurePrecision::Hidden);
        assert!(decision.replacement.is_none());
        assert!(decision.codes.contains(&DecisionCode::Tombstoned));
    }

    #[test]
    fn k_anonymity_downgrades_unique_city_to_type_only() {
        let mut policy = policy();
        policy.k_anonymity = 2;
        let mut graph = PrivacyGraph::new();
        graph.add_fact(&policy, "张三", "地址", "北京市海淀区中关村大街1号");
        graph.add_fact(&policy, "李四", "地址", "上海市浦东新区世纪大道2号");
        let mut ledger = ledger();
        let mut adjudicator = PrivacyAdjudicator::new(&policy, &graph, &mut ledger);
        let decision = adjudicator.adjudicate(
            &DisclosureRequest::new("张三", "地址", "北京市海淀区中关村大街1号")
                .with_precision(DisclosurePrecision::Coarse),
        );
        assert_eq!(decision.status, DisclosureStatus::Downgraded);
        assert_eq!(decision.effective_precision, DisclosurePrecision::TypeOnly);
        assert_eq!(decision.replacement.as_deref(), Some("某地"));
        assert!(decision.codes.contains(&DecisionCode::KAnonymity));
    }

    #[test]
    fn k_anonymity_hides_unique_value_in_sparse_graph() {
        let mut policy = policy();
        policy.k_anonymity = 2;
        let mut graph = PrivacyGraph::new();
        graph.add_fact(&policy, "张三", "地址", "北京市海淀区中关村大街1号");
        let mut ledger = ledger();
        let mut adjudicator = PrivacyAdjudicator::new(&policy, &graph, &mut ledger);
        let decision = adjudicator.adjudicate(
            &DisclosureRequest::new("张三", "地址", "北京市海淀区中关村大街1号")
                .with_precision(DisclosurePrecision::Coarse),
        );
        assert_eq!(decision.status, DisclosureStatus::Blocked);
        assert_eq!(decision.effective_precision, DisclosurePrecision::Hidden);
        assert!(decision.codes.contains(&DecisionCode::KAnonymity));
    }

    #[test]
    fn k_anonymity_keeps_shared_city() {
        let mut policy = policy();
        policy.k_anonymity = 2;
        let mut graph = PrivacyGraph::new();
        graph.add_fact(&policy, "张三", "地址", "北京市海淀区中关村大街1号");
        graph.add_fact(&policy, "李四", "地址", "北京市朝阳区建国路2号");
        let mut ledger = ledger();
        let mut adjudicator = PrivacyAdjudicator::new(&policy, &graph, &mut ledger);
        let decision = adjudicator.adjudicate(&DisclosureRequest::new(
            "张三",
            "地址",
            "北京市海淀区中关村大街1号",
        ));
        assert_eq!(decision.status, DisclosureStatus::Downgraded);
        assert_eq!(decision.effective_precision, DisclosurePrecision::Coarse);
        assert_eq!(decision.replacement.as_deref(), Some("北京市"));
    }

    #[test]
    fn prior_disclosure_keeps_granularity_consistent_across_turns() {
        let policy = policy();
        let mut graph = PrivacyGraph::new();
        graph.add_fact(&policy, "张三", "姓名", "张三");
        let mut ledger = ledger();
        let mut adjudicator = PrivacyAdjudicator::new(&policy, &graph, &mut ledger);
        let first = adjudicator.adjudicate(
            &DisclosureRequest::new("张三", "姓名", "张三")
                .with_precision(DisclosurePrecision::TypeOnly),
        );
        assert_eq!(first.status, DisclosureStatus::Allowed);
        assert_eq!(first.replacement.as_deref(), Some("某位人士"));

        let second = adjudicator.adjudicate(&DisclosureRequest::new("张三", "姓名", "张三"));
        assert_eq!(second.status, DisclosureStatus::Downgraded);
        assert_eq!(second.effective_precision, DisclosurePrecision::TypeOnly);
        assert!(second.codes.contains(&DecisionCode::PriorDisclosure));
    }

    #[test]
    fn inference_guard_blocks_school_location_chain() {
        let mut policy = policy();
        policy.inference_rules.push(
            InferenceRule::new(
                "学校位置推断城区",
                "就读学校",
                "位于",
                "所在城区",
                PrivacyCategory::Location,
            )
            .with_max_precision(DisclosurePrecision::TypeOnly),
        );
        let mut graph = PrivacyGraph::new();
        graph.add_fact(&policy, "张三", "就读学校", "阳光小学");
        graph.add_fact(&policy, "阳光小学", "位于", "北京市海淀区");
        let mut ledger = ledger();
        let mut adjudicator = PrivacyAdjudicator::new(&policy, &graph, &mut ledger);
        let decision =
            adjudicator.adjudicate(&DisclosureRequest::new("张三", "就读学校", "阳光小学"));
        assert_eq!(decision.status, DisclosureStatus::Downgraded);
        assert_eq!(decision.effective_precision, DisclosurePrecision::TypeOnly);
        assert!(decision.codes.contains(&DecisionCode::InferenceLeak));
        assert!(decision
            .inference_leaks
            .iter()
            .any(|leak| leak.conclusion == "所在城区"));
    }

    #[test]
    fn inference_guard_can_block_when_rule_forbids_disclosure() {
        let mut policy = policy();
        policy.inference_rules.push(
            InferenceRule::new(
                "学校位置推断城区",
                "就读学校",
                "位于",
                "所在城区",
                PrivacyCategory::Location,
            )
            .with_max_precision(DisclosurePrecision::Hidden),
        );
        let mut graph = PrivacyGraph::new();
        graph.add_fact(&policy, "张三", "就读学校", "阳光小学");
        graph.add_fact(&policy, "阳光小学", "位于", "北京市海淀区");
        let mut ledger = ledger();
        let mut adjudicator = PrivacyAdjudicator::new(&policy, &graph, &mut ledger);
        let decision =
            adjudicator.adjudicate(&DisclosureRequest::new("张三", "就读学校", "阳光小学"));
        assert_eq!(decision.status, DisclosureStatus::Blocked);
        assert!(decision.codes.contains(&DecisionCode::InferenceLeak));
    }

    #[test]
    fn typed_placeholder_tracks_question_kind() {
        assert_eq!(
            typed_placeholder(PrivacyCategory::PersonalIdentity, Some(WhKind::Person)),
            "某位人士"
        );
        assert_eq!(
            typed_placeholder(PrivacyCategory::Location, Some(WhKind::Place)),
            "某地"
        );
        assert_eq!(
            typed_placeholder(PrivacyCategory::Demographic, Some(WhKind::Time)),
            "某个时间段"
        );
        assert_eq!(
            coarsen_value(
                PrivacyCategory::Demographic,
                DisclosurePrecision::Granular,
                "18岁",
                Some(WhKind::Quantity)
            )
            .as_deref(),
            Some("15-19岁")
        );
        assert_eq!(
            coarsen_value(
                PrivacyCategory::Demographic,
                DisclosurePrecision::Coarse,
                "1990-01-01",
                Some(WhKind::Time)
            )
            .as_deref(),
            Some("1990年代")
        );
    }

    #[test]
    fn question_kind_is_recorded_on_request() {
        let request =
            DisclosureRequest::new("张三", "地址", "北京市").with_question("张三在哪里？");
        assert_eq!(request.wh, Some(WhKind::Place));
        let decision_policy = PolicyMode::Audit;
        assert_eq!(decision_policy.label(), "审计（影子）");
    }

    #[test]
    fn privacy_acts_parse_grant_revoke_forget_and_policy() {
        assert_eq!(
            PrivacyAct::parse("我同意你告诉别人我的电话"),
            PrivacyAct::Grant {
                category: PrivacyCategory::Contact,
                scope: ConsentScope::Session,
            }
        );
        assert_eq!(
            PrivacyAct::parse("只此一次，可以告诉别人我的邮箱"),
            PrivacyAct::Grant {
                category: PrivacyCategory::Contact,
                scope: ConsentScope::Once,
            }
        );
        assert_eq!(
            PrivacyAct::parse("别把我的地址透露给别人"),
            PrivacyAct::Revoke {
                category: PrivacyCategory::Location,
            }
        );
        assert_eq!(
            PrivacyAct::parse("不要告诉别人我的病史"),
            PrivacyAct::Revoke {
                category: PrivacyCategory::Health,
            }
        );
        assert_eq!(
            PrivacyAct::parse("忘掉关于我的所有信息"),
            PrivacyAct::Forget { category: None }
        );
        assert_eq!(
            PrivacyAct::parse("忘记我的家庭住址"),
            PrivacyAct::Forget {
                category: Some(PrivacyCategory::Location),
            }
        );
        assert_eq!(
            PrivacyAct::parse("你的隐私政策是什么？"),
            PrivacyAct::AskPolicy
        );
        assert_eq!(
            PrivacyAct::parse("我可以告诉别人我的电话吗"),
            PrivacyAct::AskPolicy
        );
        assert_eq!(PrivacyAct::parse("今天天气不错"), PrivacyAct::Other);
    }

    #[test]
    fn apply_act_updates_consent_and_tombstones() {
        let mut ledger = ledger();
        ledger.apply_act(
            "张三",
            "问答",
            &PrivacyAct::Grant {
                category: PrivacyCategory::Contact,
                scope: ConsentScope::Session,
            },
        );
        assert!(ledger.consent_permits("张三", PrivacyCategory::Contact, "问答"));
        ledger.apply_act(
            "张三",
            "问答",
            &PrivacyAct::Revoke {
                category: PrivacyCategory::Contact,
            },
        );
        assert!(!ledger.consent_permits("张三", PrivacyCategory::Contact, "问答"));
        ledger.apply_act(
            "张三",
            "问答",
            &PrivacyAct::Forget {
                category: Some(PrivacyCategory::Location),
            },
        );
        assert!(ledger.is_tombstoned(
            "张三",
            "地址",
            "北京市",
            PrivacyCategory::Location
        ));
        assert!(!ledger.is_tombstoned(
            "张三",
            "电话",
            "13800138000",
            PrivacyCategory::Contact
        ));
    }

    #[test]
    fn policy_summary_is_deterministic_and_value_free() {
        let policy = PrivacyPolicy::default();
        let summary = policy.render_summary();
        assert!(summary.contains("审计（影子）"));
        assert!(summary.contains("k=2"));
        assert!(summary.contains("contact=type"));
        assert!(!summary.contains("13800138000"));
    }

    #[test]
    fn ledger_roundtrip_is_canonical_and_versioned() {
        let mut ledger = ledger();
        ledger.grant(
            "张三",
            PrivacyCategory::Location,
            "*",
            ConsentScope::Session,
        );
        ledger.forget_category("李四", PrivacyCategory::Health, "用户撤回");
        let text = ledger.canonical_json().expect("规范 JSON");
        let restored = PrivacyLedger::from_json(&text).expect("恢复");
        assert_eq!(restored, ledger);
        assert_eq!(restored.digest(), ledger.digest());

        let mut value: serde_json::Value = serde_json::from_str(&text).unwrap();
        value["version"] = serde_json::json!(PRIVACY_LEDGER_VERSION + 1);
        let error = PrivacyLedger::from_json(&value.to_string()).unwrap_err();
        assert!(error.contains("高于支持版本"));
    }

    #[test]
    fn overlay_replaces_or_hides_according_to_decision() {
        use crate::symbolic_generator::GenerationOverlay;
        let mut policy = policy();
        let mut graph = PrivacyGraph::new();
        graph.add_fact(&policy, "张三", "地址", "北京市海淀区中关村大街1号");
        let mut ledger = ledger();
        let mut adjudicator = PrivacyAdjudicator::new(&policy, &graph, &mut ledger);
        let decision = adjudicator.adjudicate(
            &DisclosureRequest::new("张三", "地址", "北京市海淀区中关村大街1号")
                .with_precision(DisclosurePrecision::Coarse)
                .with_k(1),
        );
        let mut overlay = GenerationOverlay::default();
        decision.apply_to_overlay(&mut overlay, "张三", "地址", "北京市海淀区中关村大街1号");
        assert_eq!(
            overlay
                .replace
                .get(&("张三".to_string(), "地址".to_string())),
            Some(&vec!["北京市".to_string()])
        );

        policy.mode = PolicyMode::Enforce;
        ledger.forget_subject("张三", "遗忘");
        let mut adjudicator = PrivacyAdjudicator::new(&policy, &graph, &mut ledger);
        let blocked = adjudicator.adjudicate(&DisclosureRequest::new(
            "张三",
            "地址",
            "北京市海淀区中关村大街1号",
        ));
        let mut overlay = GenerationOverlay::default();
        blocked.apply_to_overlay(&mut overlay, "张三", "地址", "北京市海淀区中关村大街1号");
        assert!(overlay
            .hide
            .contains(&("张三".to_string(), "地址".to_string())));
    }

    #[test]
    fn audit_summary_counts_retention_and_consent() {
        let mut policy = PrivacyPolicy::default();
        policy.mode = PolicyMode::Audit;
        policy.k_anonymity = 1;
        let mut graph = PrivacyGraph::new();
        graph.add_fact(&policy, "张三", "职业", "工程师");
        graph.add_fact(&policy, "张三", "电话", "13800138000");
        graph.add_fact(&policy, "张三", "地址", "北京市海淀区中关村大街1号");
        let mut ledger = ledger();
        let mut adjudicator = PrivacyAdjudicator::new(&policy, &graph, &mut ledger);
        let decisions = vec![
            adjudicator.adjudicate(&DisclosureRequest::new("张三", "职业", "工程师")),
            adjudicator.adjudicate(&DisclosureRequest::new("张三", "电话", "13800138000")),
            adjudicator.adjudicate(&DisclosureRequest::new(
                "张三",
                "地址",
                "北京市海淀区中关村大街1号",
            )),
        ];

        let summary = AuditSummary::from_decisions(&decisions);
        assert_eq!(summary.total, 3);
        assert_eq!(summary.allowed, 1);
        assert_eq!(summary.needs_consent, 1);
        assert_eq!(summary.downgraded, 1);
        assert_eq!(summary.blocked, 0);
        assert_eq!(summary.public_total, 1);
        assert_eq!(summary.public_retained, 1);
        assert_eq!(summary.public_changed, 0);
        assert_eq!(summary.false_block_rate(), 0.0);
        assert_eq!(summary.public_retention(), 1.0);
        assert_eq!(summary.sensitive_total, 2);
        assert_eq!(summary.sensitive_retained, 0);
        assert_eq!(summary.code_counts.get("consent_required"), Some(&1));
        assert_eq!(summary.code_counts.get("policy_ceiling"), Some(&2));
        assert!(summary.render().contains("公开事实保留 1/1"));
    }

    #[test]
    fn audit_summary_flags_public_fact_false_block() {
        let mut policy = PrivacyPolicy::default();
        policy.mode = PolicyMode::Audit;
        policy.k_anonymity = 1;
        let mut graph = PrivacyGraph::new();
        graph.add_fact(&policy, "张三", "职业", "工程师");
        let mut ledger = ledger();
        ledger.forget_subject("张三", "用户要求遗忘");
        let mut adjudicator = PrivacyAdjudicator::new(&policy, &graph, &mut ledger);
        let decision = adjudicator.adjudicate(&DisclosureRequest::new("张三", "职业", "工程师"));

        let summary = AuditSummary::from_decisions([&decision]);
        assert_eq!(summary.public_total, 1);
        assert_eq!(summary.public_changed, 1);
        assert_eq!(summary.false_block_rate(), 1.0);
        assert_eq!(summary.blocked, 1);
        assert_eq!(summary.code_counts.get("tombstoned"), Some(&1));
    }

    #[test]
    fn zero_k_disables_anonymity_check() {
        let mut policy = policy();
        policy.k_anonymity = 0;
        let mut graph = PrivacyGraph::new();
        graph.add_fact(&policy, "张三", "地址", "北京市海淀区中关村大街1号");
        let mut ledger = ledger();
        let mut adjudicator = PrivacyAdjudicator::new(&policy, &graph, &mut ledger);
        let decision = adjudicator.adjudicate(&DisclosureRequest::new(
            "张三",
            "地址",
            "北京市海淀区中关村大街1号",
        ));
        assert_eq!(decision.effective_precision, DisclosurePrecision::Coarse);
        assert!(!decision.codes.contains(&DecisionCode::KAnonymity));
    }

    #[test]
    fn custom_policy_keeps_default_rules_and_predicate_override() {
        let mut policy = PrivacyPolicy::default();
        policy.predicate_rules = vec![PredicateRule {
            contains: "就诊医院".to_string(),
            category: PrivacyCategory::Health,
            max_precision: Some(DisclosurePrecision::TypeOnly),
        }];
        policy.merge_default_rules();
        assert_eq!(
            policy.classify_predicate("联系电话"),
            PrivacyCategory::Contact
        );
        assert_eq!(
            policy.max_precision_for_predicate("就诊医院"),
            Some(DisclosurePrecision::TypeOnly)
        );
        assert_eq!(
            policy.max_precision_for_predicate("电话"),
            None,
            "未显式覆盖时沿用类别上限"
        );
    }

    #[test]
    fn inference_downgrade_uses_question_compatible_category() {
        let mut policy = policy();
        policy.inference_rules.push(
            InferenceRule::new(
                "学校位置推断城区",
                "就读学校",
                "位于",
                "所在城区",
                PrivacyCategory::Location,
            )
            .with_max_precision(DisclosurePrecision::TypeOnly),
        );
        let mut graph = PrivacyGraph::new();
        graph.add_fact(&policy, "张三", "就读学校", "阳光小学");
        graph.add_fact(&policy, "阳光小学", "位于", "北京市海淀区");
        let mut ledger = ledger();
        let mut adjudicator = PrivacyAdjudicator::new(&policy, &graph, &mut ledger);
        let decision = adjudicator.adjudicate(
            &DisclosureRequest::new("张三", "就读学校", "阳光小学")
                .with_question("张三在哪里上学？"),
        );
        assert_eq!(decision.replacement.as_deref(), Some("某地"));
    }

    #[test]
    fn audit_mode_does_not_consume_one_time_consent() {
        let mut policy = PrivacyPolicy::default();
        policy.mode = PolicyMode::Audit;
        policy.k_anonymity = 1;
        let mut graph = PrivacyGraph::new();
        graph.add_fact(&policy, "张三", "电话", "13800138000");
        let mut ledger = ledger();
        ledger.grant("张三", PrivacyCategory::Contact, "问答", ConsentScope::Once);
        let mut adjudicator = PrivacyAdjudicator::new(&policy, &graph, &mut ledger);
        let request = DisclosureRequest::new("张三", "电话", "13800138000");
        let _ = adjudicator.adjudicate(&request);
        assert!(!ledger.consents[0].consumed, "影子模式不消费授权");

        policy.mode = PolicyMode::Enforce;
        let mut adjudicator = PrivacyAdjudicator::new(&policy, &graph, &mut ledger);
        let decision = adjudicator.adjudicate(&request);
        assert_eq!(decision.status, DisclosureStatus::Allowed);
        assert!(ledger.consents[0].consumed);
    }

    #[test]
    fn classify_predicate_prefers_specific_rule() {
        let policy = PrivacyPolicy::default();
        assert_eq!(
            policy.classify_predicate("家庭住址"),
            PrivacyCategory::Location
        );
        assert_eq!(
            policy.classify_predicate("联系电话"),
            PrivacyCategory::Contact
        );
        assert_eq!(policy.classify_predicate("职业"), PrivacyCategory::Other);
    }
}

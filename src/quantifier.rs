//! 量词契约（Quantifier Contract）：把中文量词读成可计数、可校准、可生成的语义对象。
//!
//! 系统里同时存在四类容易被混为一谈的“量词”：
//!   - 分类词/度量词（个/位/家/只）：选择名词的计数单位；
//!   - 限定词/广义量词（所有/有些/大多数/没有）：给出量化力；
//!   - 数量词（三个/至少三个/至多三个）：给出基数区间；
//!   - 频度词（总是/通常/有时/很少）：给出比例或频率。
//!
//! 本模块只做四件事，并且全部是确定性的：
//!   1. 解析句首量词短语（量词、基数区间、限定词）；
//!   2. 按类型/名词选择分类词（学到证据优先，其次静态词表，最后回退“个”）；
//!   3. 用 Wilson 区间把计数校准成“大多数/有些/很少”等词面，避免 1/1 被说成“所有”；
//!   4. 生成中文数词（`2 + 个` → `两个`，`12` → `十二`，`20000` → `两万`）。
//!
//! 边界：不在这里消解辖域歧义，不把开放世界里的“没有反例”当成“所有”；
//! 所有区间与证据状态都显式返回，由调用方决定是否加 hedge。

use crate::chinese_reasoning_nlu::parse_chinese_number_u64;

/// 量词的量化力。比例用千分比表示，保证可比较、可序列化。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantifierForce {
    /// 全称：所有/每个/任何/全部。
    Universal,
    /// 存在：有些/某些/至少一个。
    Existential,
    /// 否定：没有/从不。
    No,
    /// 比例：大多数/通常/有时/很少，`permille` 在 0..=1000。
    Proportional { permille: u16 },
    /// 基数区间：至少/至多/恰好 N 个，`max = None` 表示单侧无界。
    Cardinal { min: u64, max: Option<u64> },
}

impl QuantifierForce {
    pub fn label(self) -> &'static str {
        match self {
            QuantifierForce::Universal => "全称",
            QuantifierForce::Existential => "存在",
            QuantifierForce::No => "否定",
            QuantifierForce::Proportional { .. } => "比例",
            QuantifierForce::Cardinal { .. } => "基数",
        }
    }

    /// 该量化力是否允许“零个反例”之外的开放世界断言。
    pub fn asserts_universal(self) -> bool {
        matches!(self, QuantifierForce::Universal)
    }

    pub fn proportional_ratio(self) -> Option<f64> {
        match self {
            QuantifierForce::Proportional { permille } => Some(f64::from(permille) / 1000.0),
            _ => None,
        }
    }

    /// 从区间取一个保守的代表值；单侧无界返回 `None`。
    pub fn cardinal_representative(self) -> Option<u64> {
        match self {
            QuantifierForce::Cardinal { min, max } => match max {
                Some(max) if max == min => Some(min),
                _ => None,
            },
            _ => None,
        }
    }
}

/// 证据状态：生成时必须区分“闭世界已穷举”“开放世界没记录”“样本统计”。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceStatus {
    /// 已穷举：可以说“所有/没有”。
    Closed,
    /// 开放世界：只能说“已知/目前没有记录”。
    Open,
    /// 样本统计：必须 hedge，不能升级成全称。
    Sample,
}

impl EvidenceStatus {
    pub fn label(self) -> &'static str {
        match self {
            EvidenceStatus::Closed => "闭世界",
            EvidenceStatus::Open => "开放世界",
            EvidenceStatus::Sample => "样本",
        }
    }

    /// 开放世界/样本下，全称量词应降级成“几乎所有/已知的”。
    pub fn hedge(self) -> Option<&'static str> {
        match self {
            EvidenceStatus::Closed => None,
            EvidenceStatus::Open => Some("已知的"),
            EvidenceStatus::Sample => Some("几乎所有"),
        }
    }
}

/// 一次量词解析结果：原词面、量化力、分类词与剩余文本。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantifierReading {
    pub surface: String,
    pub force: QuantifierForce,
    pub classifier: Option<String>,
    pub rest: String,
}

impl QuantifierReading {
    /// 去掉量词后的句子；调用方不得把空串当成完整命题。
    pub fn remainder(&self) -> &str {
        self.rest.as_str()
    }
}

const ALL_PREFIXES: [&str; 7] = ["所有", "全部", "每一个", "每个", "任何", "一切", "凡是"];
const NO_PREFIXES: [&str; 8] = [
    "没有任何",
    "没有一个",
    "从来没有",
    "从未有过",
    "从不",
    "从未",
    "绝不",
    "没有",
];
const EXISTENTIAL_PREFIXES: [&str; 6] = ["有些", "有的", "某些", "一些", "部分", "至少一个"];

/// 比例词表：长词在前，千分比与 `chinese_reasoning_nlu` 的概率表保持同一量级。
const PROPORTIONAL_PREFIXES: [(&str, u16); 18] = [
    ("绝大多数", 950),
    ("几乎所有", 950),
    ("大多数", 850),
    ("大部分", 850),
    ("一般来说", 800),
    ("通常", 800),
    ("往往", 800),
    ("总是", 980),
    ("一直", 950),
    ("多数", 800),
    ("多半", 750),
    ("许多", 700),
    ("有时", 400),
    ("有时候", 400),
    ("偶尔", 250),
    ("少数", 150),
    ("很少", 150),
    ("极少", 100),
];

const CARDINAL_AT_LEAST: [&str; 5] = ["至少", "不少于", "不低于", "超过", "多于"];
const CARDINAL_AT_MOST: [&str; 5] = ["至多", "不超过", "不多于", "不足", "少于"];
const CARDINAL_EXACT: [&str; 4] = ["恰好", "正好", "刚好", "整整"];

/// 常见分类词。顺序只影响同长度匹配的稳定性，不影响语义。
pub const CLASSIFIERS: [&str; 39] = [
    "个", "位", "名", "只", "条", "张", "本", "部", "台", "辆", "头", "匹", "件", "次", "份",
    "家", "所", "座", "枚", "颗", "支", "首", "篇", "项", "种", "人", "间", "组", "对", "双",
    "套", "批", "堆", "群", "列", "行", "页", "章", "节",
];

/// 分类词选择来源，按可信度从高到低。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClassifierSource {
    /// 跨会话词法记忆学到的量词期待。
    Learned,
    /// 本体/类型标签。
    Schema,
    /// 名词后缀启发式。
    Noun,
    /// 默认“个”。
    Default,
}

impl ClassifierSource {
    pub fn label(self) -> &'static str {
        match self {
            ClassifierSource::Learned => "学到",
            ClassifierSource::Schema => "类型",
            ClassifierSource::Noun => "名词",
            ClassifierSource::Default => "默认",
        }
    }
}

/// 一次分类词选择：词面 + 来源 + 理由。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassifierChoice {
    pub classifier: String,
    pub source: ClassifierSource,
    pub reason: String,
}

impl ClassifierChoice {
    fn new(classifier: &str, source: ClassifierSource, reason: impl Into<String>) -> Self {
        Self {
            classifier: classifier.to_string(),
            source,
            reason: reason.into(),
        }
    }
}

const PERSON_NOUNS: [&str; 20] = [
    "人物", "学生", "老师", "教授", "医生", "律师", "作家", "经理", "员工", "朋友", "徒弟",
    "师傅", "孩子", "男孩", "女孩", "演员", "歌手", "官员", "记者", "警察",
];
const ORGANIZATION_NOUNS: [&str; 14] = [
    "组织", "公司", "集团", "大学", "学院", "医院", "银行", "博物馆", "图书馆", "研究院",
    "委员会", "政府", "法院", "机构",
];
const ANIMAL_NOUNS: [&str; 7] = ["动物", "鸟", "猫", "狗", "马", "牛", "鱼"];
const WORK_NOUNS: [&str; 6] = ["作品", "书", "小说", "文章", "论文", "电影"];
const PLACE_NOUNS: [&str; 6] = ["城市", "国家", "省", "山", "桥", "建筑"];
const VEHICLE_NOUNS: [&str; 4] = ["汽车", "车", "飞机", "船"];

/// 按名词/类型选择分类词。静态表只做兜底；学到证据应由调用方优先传入。
pub fn classifier_for_noun(noun: &str) -> ClassifierChoice {
    let noun = noun.trim();
    if noun.is_empty() {
        return ClassifierChoice::new("个", ClassifierSource::Default, "空名词回退「个」");
    }
    if let Some(hit) = longest_suffix_hit(noun, &PERSON_NOUNS) {
        return ClassifierChoice::new("位", ClassifierSource::Noun, format!("人物名词「{hit}」"));
    }
    if let Some(hit) = longest_suffix_hit(noun, &ORGANIZATION_NOUNS) {
        if matches!(hit, "大学" | "学院" | "医院" | "博物馆" | "图书馆" | "研究院") {
            return ClassifierChoice::new("所", ClassifierSource::Noun, format!("机构名词「{hit}」"));
        }
        return ClassifierChoice::new("家", ClassifierSource::Noun, format!("组织名词「{hit}」"));
    }
    if let Some(hit) = longest_suffix_hit(noun, &ANIMAL_NOUNS) {
        return ClassifierChoice::new("只", ClassifierSource::Noun, format!("动物名词「{hit}」"));
    }
    if let Some(hit) = longest_suffix_hit(noun, &WORK_NOUNS) {
        let classifier = if matches!(hit, "文章" | "论文") { "篇" } else { "部" };
        return ClassifierChoice::new(classifier, ClassifierSource::Noun, format!("作品名词「{hit}」"));
    }
    if let Some(hit) = longest_suffix_hit(noun, &VEHICLE_NOUNS) {
        return ClassifierChoice::new("辆", ClassifierSource::Noun, format!("交通工具「{hit}」"));
    }
    if let Some(hit) = longest_suffix_hit(noun, &PLACE_NOUNS) {
        return ClassifierChoice::new("座", ClassifierSource::Noun, format!("地点/建筑「{hit}」"));
    }
    ClassifierChoice::new("个", ClassifierSource::Default, "未识别名词回退「个」")
}

/// 按 RDF 类型标签选择分类词（英文标签按 `localized_type_label` 的同一口径匹配）。
pub fn classifier_for_type(etype: &str) -> ClassifierChoice {
    let label = etype.trim();
    if label.is_empty() {
        return ClassifierChoice::new("个", ClassifierSource::Default, "空类型回退「个」");
    }
    match label.to_ascii_lowercase().as_str() {
        "person" | "people" | "human" => {
            return ClassifierChoice::new("位", ClassifierSource::Schema, "类型 person")
        }
        "organization" | "organisation" | "company" => {
            return ClassifierChoice::new("家", ClassifierSource::Schema, "类型 organization")
        }
        "animal" => return ClassifierChoice::new("只", ClassifierSource::Schema, "类型 animal"),
        "work" => return ClassifierChoice::new("部", ClassifierSource::Schema, "类型 work"),
        "city" | "location" | "place" | "country" => {
            return ClassifierChoice::new("座", ClassifierSource::Schema, "类型 location")
        }
        _ => {}
    }
    let choice = classifier_for_noun(label);
    if choice.source == ClassifierSource::Noun {
        ClassifierChoice::new(&choice.classifier, ClassifierSource::Schema, format!("类型「{label}」"))
    } else {
        choice
    }
}

/// 综合选择：学到证据 > 显式请求 > 类型 > 名词 > 默认。
pub fn select_classifier(
    learned: Option<&str>,
    requested: Option<&str>,
    etype: Option<&str>,
    noun: Option<&str>,
) -> ClassifierChoice {
    if let Some(learned) = learned.and_then(normalize_classifier) {
        return ClassifierChoice::new(&learned, ClassifierSource::Learned, "词法记忆的量词期待");
    }
    if let Some(requested) = requested.and_then(normalize_classifier) {
        return ClassifierChoice::new(&requested, ClassifierSource::Schema, "问句显式量词");
    }
    if let Some(etype) = etype {
        let choice = classifier_for_type(etype);
        if choice.source == ClassifierSource::Schema {
            return choice;
        }
    }
    if let Some(noun) = noun {
        let choice = classifier_for_noun(noun);
        if choice.source == ClassifierSource::Noun {
            return choice;
        }
    }
    ClassifierChoice::new("个", ClassifierSource::Default, "无可用证据回退「个」")
}

fn normalize_classifier(text: &str) -> Option<String> {
    let text = text.trim();
    if text.is_empty() || text.chars().count() > 3 {
        return None;
    }
    if text.chars().any(|ch| ch.is_ascii_digit() || ch.is_whitespace()) {
        return None;
    }
    Some(text.to_string())
}

fn longest_suffix_hit<'a>(text: &str, table: &'a [&str]) -> Option<&'a str> {
    table
        .iter()
        .filter(|entry| text.ends_with(*entry))
        .max_by_key(|entry| entry.chars().count())
        .copied()
}

/// 解析句首量词短语。返回 `None` 表示句首没有可识别的量词。
pub fn parse_quantifier(text: &str) -> Option<QuantifierReading> {
    let trimmed = text.trim_start();
    if trimmed.is_empty() {
        return None;
    }
    for prefix in NO_PREFIXES {
        if let Some(rest) = trimmed.strip_prefix(prefix) {
            let rest = rest.trim_start().to_string();
            return Some(QuantifierReading {
                surface: prefix.to_string(),
                force: QuantifierForce::No,
                classifier: None,
                rest,
            });
        }
    }
    for prefix in ALL_PREFIXES {
        if let Some(rest) = trimmed.strip_prefix(prefix) {
            let rest = rest.trim_start().to_string();
            return Some(QuantifierReading {
                surface: prefix.to_string(),
                force: QuantifierForce::Universal,
                classifier: None,
                rest,
            });
        }
    }
    for prefix in EXISTENTIAL_PREFIXES {
        if let Some(rest) = trimmed.strip_prefix(prefix) {
            let rest = rest.trim_start().to_string();
            return Some(QuantifierReading {
                surface: prefix.to_string(),
                force: QuantifierForce::Existential,
                classifier: None,
                rest,
            });
        }
    }
    let mut proportional: Vec<&(&str, u16)> = PROPORTIONAL_PREFIXES
        .iter()
        .filter(|(prefix, _)| trimmed.starts_with(*prefix))
        .collect();
    proportional.sort_by_key(|(prefix, _)| std::cmp::Reverse(prefix.chars().count()));
    if let Some((prefix, permille)) = proportional.first() {
        let rest = trimmed[prefix.len()..].trim_start().to_string();
        return Some(QuantifierReading {
            surface: (*prefix).to_string(),
            force: QuantifierForce::Proportional {
                permille: *permille,
            },
            classifier: None,
            rest,
        });
    }
    parse_cardinal(trimmed)
}

fn parse_cardinal(text: &str) -> Option<QuantifierReading> {
    for (prefix, kind) in CARDINAL_AT_LEAST
        .iter()
        .map(|prefix| (*prefix, CardinalBound::AtLeast))
        .chain(
            CARDINAL_AT_MOST
                .iter()
                .map(|prefix| (*prefix, CardinalBound::AtMost)),
        )
        .chain(
            CARDINAL_EXACT
                .iter()
                .map(|prefix| (*prefix, CardinalBound::Exact)),
        )
    {
        if let Some(rest) = text.strip_prefix(prefix) {
            let rest = rest.trim_start();
            let (number, classifier, remainder) = split_cardinal(rest)?;
            let force = match kind {
                CardinalBound::AtLeast => QuantifierForce::Cardinal {
                    min: number,
                    max: None,
                },
                CardinalBound::AtMost => QuantifierForce::Cardinal {
                    min: 0,
                    max: Some(number),
                },
                CardinalBound::Exact => QuantifierForce::Cardinal {
                    min: number,
                    max: Some(number),
                },
            };
            return Some(QuantifierReading {
                surface: format!("{}{}", prefix, rest[..rest.len() - remainder.len()].trim()),
                force,
                classifier,
                rest: remainder.trim_start().to_string(),
            });
        }
    }
    let (number, classifier, remainder) = split_cardinal(text)?;
    classifier.as_ref()?;
    let surface_len = text.len() - remainder.len();
    Some(QuantifierReading {
        surface: text[..surface_len].trim().to_string(),
        force: QuantifierForce::Cardinal {
            min: number,
            max: Some(number),
        },
        classifier,
        rest: remainder.trim_start().to_string(),
    })
}

#[derive(Debug, Clone, Copy)]
enum CardinalBound {
    AtLeast,
    AtMost,
    Exact,
}

fn split_cardinal(text: &str) -> Option<(u64, Option<String>, &str)> {
    let mut end = 0;
    let mut seen = false;
    for (index, ch) in text.char_indices() {
        if ch.is_ascii_digit() || is_chinese_number_char(ch) {
            seen = true;
            end = index + ch.len_utf8();
        } else if ch == '.' && seen {
            end = index + ch.len_utf8();
        } else if ch.is_whitespace() && seen {
            end = index + ch.len_utf8();
        } else {
            break;
        }
    }
    if !seen {
        return None;
    }
    let number_text = text[..end].trim();
    let number = parse_chinese_number_u64(number_text)?;
    let after_number = &text[end..];
    let mut rest = after_number;
    let mut classifier = None;
    for candidate in CLASSIFIERS {
        if let Some(stripped) = rest.strip_prefix(candidate) {
            classifier = Some(candidate.to_string());
            rest = stripped;
            break;
        }
    }
    Some((number, classifier, rest))
}

fn is_chinese_number_char(ch: char) -> bool {
    matches!(
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

/// 从计数问句中取出用户显式使用的分类词（「有多少位人物」→「位」）。
pub fn requested_classifier(question: &str) -> Option<String> {
    const CUES: [&str; 9] = [
        "一共有多少",
        "总共有多少",
        "共有多少",
        "有多少",
        "多少",
        "一共有几",
        "一共几位",
        "几",
        "哪几",
    ];
    for cue in CUES {
        let mut offset = 0;
        while let Some(position) = question[offset..].find(cue) {
            let after = &question[offset + position + cue.len()..];
            let after = after.trim_start();
            if let Some(candidate) = CLASSIFIERS
                .iter()
                .find(|candidate| after.starts_with(**candidate))
            {
                return Some((*candidate).to_string());
            }
            offset += position + cue.len();
            if offset >= question.len() {
                break;
            }
        }
    }
    // 「人物有多少位」这类量词在句尾：取问号/语气词前的分类词。
    let tail = question.trim_end_matches(|ch: char| {
        matches!(ch, '？' | '?' | '。' | '！' | '!' | '呢' | '吗' | '啊' | '吧' | ' ' | '的')
    });
    CLASSIFIERS
        .iter()
        .find(|candidate| tail.ends_with(**candidate))
        .map(|candidate| (*candidate).to_string())
}

/// 计数校准结果：比例区间 + 量化力 + 证据状态。
#[derive(Debug, Clone, PartialEq)]
pub struct QuantifierEstimate {
    pub successes: u64,
    pub total: u64,
    pub ratio: f64,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub status: EvidenceStatus,
    pub force: QuantifierForce,
}

impl QuantifierEstimate {
    /// 生成用词面：样本不足时不会说“所有”。
    pub fn label_zh(&self) -> String {
        match self.force {
            QuantifierForce::Universal => match self.status {
                EvidenceStatus::Closed => "所有".to_string(),
                _ => "几乎所有".to_string(),
            },
            QuantifierForce::Existential => "有些".to_string(),
            QuantifierForce::No => "没有".to_string(),
            QuantifierForce::Proportional { permille } => match permille {
                0..=100 => "极少".to_string(),
                101..=250 => "很少".to_string(),
                251..=500 => "有些".to_string(),
                501..=750 => "许多".to_string(),
                751..=900 => "大多数".to_string(),
                _ => "绝大多数".to_string(),
            },
            QuantifierForce::Cardinal { min, max } => match max {
                Some(max) if max == min => format_chinese_count(min, "个"),
                Some(max) => format!("{}到{}个", min, max),
                None => format!("至少{}个", min),
            },
        }
    }

    /// 带分类词的计数答句片段，例如「大多数（9/10）」。
    pub fn describe_zh(&self, classifier: &str) -> String {
        let label = match self.label_zh() {
            label if label.ends_with('个') && classifier != "个" => {
                label.trim_end_matches('个').to_string() + classifier
            }
            label => label,
        };
        format!(
            "{}（{}/{}，Wilson 下界 {:.2}）",
            label, self.successes, self.total, self.lower_bound
        )
    }
}

/// 用 Wilson 得分区间校准计数，避免小样本过度断言。
///
/// `successes == total` 只有在样本足够大（n ≥ 20）且下界 ≥ 0.90 时才升级为
/// 全称；否则保留为 `几乎所有`/比例，并标记为 [`EvidenceStatus::Sample`]。
pub fn calibrate_counts(successes: u64, total: u64) -> Option<QuantifierEstimate> {
    if total == 0 || successes > total {
        return None;
    }
    let n = total as f64;
    let p = successes as f64 / n;
    let z = 1.96_f64;
    let z2 = z * z;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let margin = z * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt()) / denominator;
    let lower_bound = (center - margin).clamp(0.0, 1.0);
    let upper_bound = (center + margin).clamp(0.0, 1.0);
    let force = if successes == 0 {
        QuantifierForce::No
    } else if successes == total && total >= 20 && lower_bound >= 0.90 {
        QuantifierForce::Universal
    } else {
        QuantifierForce::Proportional {
            permille: (p * 1000.0).round().clamp(0.0, 1000.0) as u16,
        }
    };
    let status = if total >= 20 && lower_bound >= 0.90 && successes == total {
        EvidenceStatus::Sample
    } else {
        EvidenceStatus::Sample
    };
    Some(QuantifierEstimate {
        successes,
        total,
        ratio: p,
        lower_bound,
        upper_bound,
        status,
        force,
    })
}

/// 生成中文数词：`0` → `零`，`12` → `十二`，`20000` → `两万`。
pub fn format_chinese_number(value: u64) -> String {
    format_chinese_integer(value)
}

/// 生成「数词 + 分类词」：`2 + 个` → `两个`，`12 + 位` → `十二位`。
pub fn format_chinese_count(value: u64, classifier: &str) -> String {
    let classifier = classifier.trim();
    if classifier.is_empty() {
        return format_chinese_integer(value);
    }
    if value == 2 {
        return format!("两{}", classifier);
    }
    format!("{}{}", format_chinese_integer(value), classifier)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_universal_existential_and_negative_prefixes() {
        let all = parse_quantifier("所有鸟都会飞").expect("全称量词");
        assert_eq!(all.force, QuantifierForce::Universal);
        assert_eq!(all.remainder(), "鸟都会飞");

        let some = parse_quantifier("有些学生会迟到").expect("存在量词");
        assert_eq!(some.force, QuantifierForce::Existential);
        assert_eq!(some.remainder(), "学生会迟到");

        let none = parse_quantifier("没有鸟会飞").expect("否定量词");
        assert_eq!(none.force, QuantifierForce::No);
        assert_eq!(none.remainder(), "鸟会飞");

        let rare = parse_quantifier("很少人会这样做").expect("低频比例");
        assert_eq!(rare.force, QuantifierForce::Proportional { permille: 150 });
        assert_eq!(rare.remainder(), "人会这样做");
    }

    #[test]
    fn parses_cardinal_ranges_with_classifier() {
        let exact = parse_quantifier("三个学生通过了考试").expect("精确基数");
        assert_eq!(
            exact.force,
            QuantifierForce::Cardinal {
                min: 3,
                max: Some(3)
            }
        );
        assert_eq!(exact.classifier.as_deref(), Some("个"));
        assert_eq!(exact.remainder(), "学生通过了考试");

        let at_least = parse_quantifier("至少三名学生通过了考试").expect("下界");
        assert_eq!(
            at_least.force,
            QuantifierForce::Cardinal {
                min: 3,
                max: None
            }
        );
        assert_eq!(at_least.classifier.as_deref(), Some("名"));

        let at_most = parse_quantifier("至多两名学生").expect("上界");
        assert_eq!(
            at_most.force,
            QuantifierForce::Cardinal {
                min: 0,
                max: Some(2)
            }
        );
        assert_eq!(at_most.classifier.as_deref(), Some("名"));
    }

    #[test]
    fn cardinal_parser_rejects_plain_nouns() {
        assert!(parse_quantifier("苹果会飞").is_none());
        assert!(parse_quantifier("三").is_none(), "裸数词不携带可断言的量词");
        assert!(parse_quantifier("三个").is_some());
    }

    #[test]
    fn requested_classifier_keeps_user_measure_word() {
        assert_eq!(requested_classifier("一共有多少位人物？"), Some("位".to_string()));
        assert_eq!(requested_classifier("人物有多少个"), Some("个".to_string()));
        assert_eq!(requested_classifier("张三教过多少名学生"), Some("名".to_string()));
        assert_eq!(requested_classifier("张三教过多少个学生"), Some("个".to_string()));
        assert_eq!(requested_classifier("张三有几个师傅"), Some("个".to_string()));
        assert_eq!(requested_classifier("张三的年龄是多少"), None);
    }

    #[test]
    fn classifier_selection_prefers_learned_then_schema_then_noun() {
        let learned = select_classifier(Some("名"), None, Some("人物"), Some("学生"));
        assert_eq!(learned.classifier, "名");
        assert_eq!(learned.source, ClassifierSource::Learned);

        let requested = select_classifier(None, Some("位"), Some("人物"), None);
        assert_eq!(requested.classifier, "位");

        let schema = select_classifier(None, None, Some("person"), None);
        assert_eq!(schema.classifier, "位");
        assert_eq!(schema.source, ClassifierSource::Schema);

        let noun = select_classifier(None, None, None, Some("公司"));
        assert_eq!(noun.classifier, "家");

        let fallback = select_classifier(None, None, None, Some("量子纠缠"));
        assert_eq!(fallback.classifier, "个");
        assert_eq!(fallback.source, ClassifierSource::Default);
    }

    #[test]
    fn chinese_number_generation_roundtrips() {
        for value in [0_u64, 1, 2, 9, 10, 11, 12, 19, 20, 21, 99, 100, 101, 108, 110, 200,
            1000, 1001, 1010, 2000, 10_000, 12_000, 20_000, 22_000, 100_000, 1_000_000,
            100_000_000, 1_000_000_000]
        {
            let rendered = format_chinese_number(value);
            assert_eq!(
                parse_chinese_number_u64(&rendered),
                Some(value),
                "{value} → {rendered}"
            );
        }
        assert_eq!(format_chinese_number(2), "二");
        assert_eq!(format_chinese_count(2, "个"), "两个");
        assert_eq!(format_chinese_count(12, "位"), "十二位");
        assert_eq!(format_chinese_count(20_000, "个"), "两万个");
        assert_eq!(format_chinese_count(2, ""), "二");
    }

    #[test]
    fn count_calibration_avoids_overclaiming_from_one_sample() {
        let tiny = calibrate_counts(1, 1).expect("1/1 仍可校准");
        assert_eq!(
            tiny.force,
            QuantifierForce::Proportional { permille: 1000 },
            "小样本不能升级成全称"
        );
        assert_eq!(tiny.label_zh(), "绝大多数");

        let nearly = calibrate_counts(20, 20).expect("20/20");
        assert_eq!(
            nearly.force,
            QuantifierForce::Proportional { permille: 1000 },
            "20 个样本的 Wilson 下界仍不足 0.90，不能说全称"
        );
        assert_eq!(nearly.label_zh(), "绝大多数");

        let full = calibrate_counts(40, 40).expect("40/40");
        assert_eq!(full.force, QuantifierForce::Universal);
        assert_eq!(full.status, EvidenceStatus::Sample);
        assert_eq!(full.label_zh(), "几乎所有");

        let most = calibrate_counts(9, 10).expect("9/10");
        assert!(matches!(
            most.force,
            QuantifierForce::Proportional { permille: 900 }
        ));
        assert_eq!(most.label_zh(), "大多数");

        let none = calibrate_counts(0, 10).expect("0/10");
        assert_eq!(none.force, QuantifierForce::No);
        assert_eq!(none.label_zh(), "没有");

        assert!(calibrate_counts(0, 0).is_none());
        assert!(calibrate_counts(11, 10).is_none());
    }

    #[test]
    fn evidence_status_hedges_open_world_universal() {
        assert_eq!(EvidenceStatus::Closed.hedge(), None);
        assert_eq!(EvidenceStatus::Open.hedge(), Some("已知的"));
        assert_eq!(EvidenceStatus::Sample.hedge(), Some("几乎所有"));
    }
}

/// 中文整数生成。内部函数与 `format_chinese_number` 分开，便于按“两”的
/// 口语形态扩展而不影响解析器的标准读法。
pub fn format_chinese_integer(value: u64) -> String {
    if value == 0 {
        return "零".to_string();
    }
    let mut groups: Vec<u64> = Vec::new();
    let mut rest = value;
    while rest > 0 {
        groups.push(rest % 10_000);
        rest /= 10_000;
    }
    let section_units = ["", "万", "亿", "兆", "京"];
    let mut output = String::new();
    for (index, group) in groups.iter().enumerate().rev() {
        if *group == 0 {
            if !output.is_empty()
                && !output.ends_with('零')
                && groups[..index].iter().any(|value| *value > 0)
            {
                output.push('零');
            }
            continue;
        }
        if !output.is_empty() && *group < 1000 && !output.ends_with('零') {
            output.push('零');
        }
        let leading_two = output.is_empty() && (*group >= 200 || index > 0);
        output.push_str(&format_group(*group, leading_two));
        if index > 0 {
            output.push_str(section_units.get(index).copied().unwrap_or(""));
        }
    }
    output
}

fn format_group(group: u64, leading_two: bool) -> String {
    let digits = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"];
    let units = ["千", "百", "十", ""];
    let values = [
        (group / 1000) % 10,
        (group / 100) % 10,
        (group / 10) % 10,
        group % 10,
    ];
    let mut output = String::new();
    let mut pending_zero = false;
    for (index, value) in values.iter().enumerate() {
        let unit = units[index];
        if *value == 0 {
            if !output.is_empty() && index < units.len() - 1 {
                pending_zero = true;
            }
            continue;
        }
        if pending_zero {
            output.push('零');
            pending_zero = false;
        }
        if *value == 2 && leading_two && output.is_empty() {
            output.push('两');
        } else {
            output.push_str(digits[*value as usize]);
        }
        output.push_str(unit);
    }
    // 十位开头省略“一”：十二、十五，而不是一十二。
    if output.starts_with("一十") {
        output.remove(0);
    }
    output
}

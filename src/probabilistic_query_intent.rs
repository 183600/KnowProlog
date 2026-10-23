//! PQI：概率问句意图（Probabilistic Query Intent）。
//!
//! 情态不是噪声，而是施加在查询目标上的算子：
//!
//! - 「李四的师傅可能是谁」：指称计划不变，只要求按知识概率排序候选；
//! - 「李四有多大的可能是王五的师傅」：对**基原子**求概率值；
//! - 「谁最可能是李四的师傅」：在候选集合上取 argmax；
//! - 「李四一定是王五的师傅吗」：把概率与阈值比较；
//! - 「李四和王五谁更可能是张三的师傅」：比较两个基原子的概率。
//!
//! 本模块只做两件事，且都不接触知识库：
//!
//! 1. **最长匹配线索扫描**：识别情态词、概率框架、最可能/比较框架与否定作用域；
//! 2. **计数型贝叶斯后验 + 框架规范化**：给「我有多确定你在问可能」一个可打印的
//!    后验，并把问句压回 [`QueryPlanner`](crate::query_planner::QueryPlanner) 能
//!    接住的指称形式。
//!
//! 这里有两条绝不能混用的账：
//!
//! - `P(M|u)`：**意图后验**，来自线索计数与平滑，回答「我有多懂你在问什么」；
//! - `P(G|KB)`：**知识概率**，由 ProbLog 的可能世界加权模型计数精确给出，
//!   回答「这件事有多可能」。
//!
//! `PQI` 只产出第一条账与一个结构化的目标；第二条账由推理器执行。两者相乘或
//! 互相替代都会把「没听懂」伪装成「事情很假」。
//!
//! 设计取舍：不改 `QueryPlanner::parse` 的返回类型，也不新增第七套意图标签。
//! 情态与「想做什么」正交；普通问句的解析路径逐字节不变。

use std::collections::HashMap;

/// 认识论情态：与「问属性 / 问是非」正交的一个维度。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EpistemicMode {
    /// 普通问句（没有情态线索）。
    Plain,
    /// 可能问：求满足概率阈值的候选，或对候选排序。
    Possibility,
    /// 概率值问：对某个基原子求 P。
    Probability,
    /// 最可能问：在候选集合上取 argmax。
    MostLikely,
    /// 确定问：把 P 与高阈值比较。
    Certainty,
    /// 比较问：比较两个基原子的 P。
    Comparison,
}

impl EpistemicMode {
    pub fn code(self) -> &'static str {
        match self {
            EpistemicMode::Plain => "plain",
            EpistemicMode::Possibility => "possibility",
            EpistemicMode::Probability => "probability",
            EpistemicMode::MostLikely => "most_likely",
            EpistemicMode::Certainty => "certainty",
            EpistemicMode::Comparison => "comparison",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            EpistemicMode::Plain => "直问",
            EpistemicMode::Possibility => "可能问",
            EpistemicMode::Probability => "概率值问",
            EpistemicMode::MostLikely => "最可能问",
            EpistemicMode::Certainty => "确定问",
            EpistemicMode::Comparison => "比较问",
        }
    }

    /// 与 [`Self::code`] 互逆；用于把会话快照里的情态字符串恢复成算子。
    pub fn from_code(code: &str) -> Option<Self> {
        match code {
            "plain" => Some(EpistemicMode::Plain),
            "possibility" => Some(EpistemicMode::Possibility),
            "probability" => Some(EpistemicMode::Probability),
            "most_likely" => Some(EpistemicMode::MostLikely),
            "certainty" => Some(EpistemicMode::Certainty),
            "comparison" => Some(EpistemicMode::Comparison),
            _ => None,
        }
    }
}

/// 线索类别。后验只依赖类别与线索强度，不依赖具体词面。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CueKind {
    Possibility,
    Probability,
    MostLikely,
    Certainty,
    Comparison,
}

impl CueKind {
    fn mode(self) -> EpistemicMode {
        match self {
            CueKind::Possibility => EpistemicMode::Possibility,
            CueKind::Probability => EpistemicMode::Probability,
            CueKind::MostLikely => EpistemicMode::MostLikely,
            CueKind::Certainty => EpistemicMode::Certainty,
            CueKind::Comparison => EpistemicMode::Comparison,
        }
    }
}

/// 一条命中的情态线索。
#[derive(Debug, Clone, PartialEq)]
pub struct EpistemicCue {
    pub phrase: String,
    pub kind: CueKind,
    /// 线索强度（0..1）：越明确的框架越接近 1。
    pub confidence: f64,
    /// 「不可能 / 不太可能」等否定作用域。
    pub negated: bool,
    /// 原句中的字节区间（左闭右开）。
    pub start: usize,
    pub end: usize,
}

/// 一个基原子断言：`predicate(subject, object)`。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Claim {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    /// 断言本身是否带否定（「不是……」）。概率始终按**肯定原子**查询，
    /// 否定断言的概率由 `1 - P(atom)` 得到，绝不在查询前把主语和宾语对调。
    pub negated: bool,
}

impl Claim {
    pub fn render(&self) -> String {
        format!(
            "{} 的 {} {} {}",
            self.subject,
            self.predicate,
            if self.negated { "不是" } else { "是" },
            self.object
        )
    }
}

/// 情态算子作用到的结构化目标。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EpistemicTarget {
    /// 对 `predicate(subject, object)` 求概率 / 与阈值比较。
    GroundClaim(Claim),
    /// 求 `predicate(subject, ?x)` 的概率排序（可能问/最可能问）。
    ObjectsOf { subject: String, predicate: String },
    /// 求 `predicate(?x, object)` 的概率排序。
    SubjectsOf { object: String, predicate: String },
    /// 比较两个基原子的概率。
    Comparison { left: Claim, right: Claim },
}

/// 一次概率问句分析的结果。
#[derive(Debug, Clone)]
pub struct ProbabilisticQueryIntent {
    /// 后验最高的情态。
    pub mode: EpistemicMode,
    /// `P(mode | cues)`：**理解置信度**，不是事实概率。
    pub confidence: f64,
    /// 完整的后验（按概率降序、同概率按 `code` 升序）。
    pub posterior: Vec<(EpistemicMode, f64)>,
    pub cues: Vec<EpistemicCue>,
    /// 当前情态是否落在否定作用域里。
    pub negated: bool,
    /// 去掉情态框架后的指称问句，可直接交给 QueryPlanner。
    pub core_question: String,
    /// 结构化目标；无法结构化时为 `None`，由调用方用 `core_question` 兜底。
    pub target: Option<EpistemicTarget>,
    /// 判定「可能/确定」时使用的阈值；概率值问固定为 0.5 仅作展示。
    pub threshold: f64,
}

impl ProbabilisticQueryIntent {
    /// 分析问句。没有可靠的情态线索时返回 `None`，普通问句路径完全不受影响。
    pub fn analyze(question: &str) -> Option<Self> {
        let text = clean_question(question);
        if text.is_empty() {
            return None;
        }
        let cues = scan_cues(&text);
        if cues.is_empty() {
            return None;
        }
        let posterior = posterior(&cues);
        let (mode, confidence) = posterior.first().copied()?;
        if mode == EpistemicMode::Plain {
            return None;
        }
        let negated = cues
            .iter()
            .filter(|cue| cue.kind.mode() == mode)
            .any(|cue| cue.negated);
        let core_question = normalize_core(&text, &cues, mode);
        let target = extract_target(&text, &cues, mode);
        let threshold = threshold_for(mode, &cues, negated);
        Some(Self {
            mode,
            confidence,
            posterior,
            cues,
            negated,
            core_question,
            target,
            threshold,
        })
    }

    /// 双账本中的第一本账：只陈述理解置信度，绝不冒充知识概率。
    pub fn render_ledger(&self) -> String {
        let cues = self
            .cues
            .iter()
            .map(|cue| {
                let scope = if cue.negated { "（否定作用域）" } else { "" };
                format!("{}{}", cue.phrase, scope)
            })
            .collect::<Vec<_>>()
            .join("、");
        let alternatives = self
            .posterior
            .iter()
            .skip(1)
            .filter(|(_, probability)| *probability >= 0.08)
            .take(2)
            .map(|(mode, probability)| format!("{} {:.2}", mode.label(), probability))
            .collect::<Vec<_>>();
        let mut line = format!(
            "概率问句理解（PQI）：情态={}；意图后验={:.2}（这是理解置信度，不是事实概率）；线索={}",
            self.mode.label(),
            self.confidence,
            cues
        );
        if self.negated {
            line.push_str("；作用域=否定");
        }
        if !alternatives.is_empty() {
            line.push_str(&format!("；备选：{}", alternatives.join("，")));
        }
        line
    }

    /// 中文情态算子的一句话说明（用于 CLI / API 回执）。
    pub fn operator_note(&self) -> String {
        match self.mode {
            EpistemicMode::Plain => "直问：只查确证事实".to_string(),
            EpistemicMode::Possibility => {
                format!("可能算子：取 P ≥ {:.2} 的候选", self.threshold)
            }
            EpistemicMode::Probability => "概率算子：返回基原子的精确 P".to_string(),
            EpistemicMode::MostLikely => "最可能算子：在候选上取 P 的 argmax".to_string(),
            EpistemicMode::Certainty => {
                format!("确定算子：把 P 与阈值 {:.2} 比较", self.threshold)
            }
            EpistemicMode::Comparison => "比较算子：比较两个基原子的 P".to_string(),
        }
    }
}

fn threshold_for(mode: EpistemicMode, cues: &[EpistemicCue], negated: bool) -> f64 {
    match mode {
        EpistemicMode::Certainty => 0.95,
        EpistemicMode::Possibility if !negated => {
            let strongest = cues
                .iter()
                .filter(|cue| cue.kind == CueKind::Possibility)
                .map(|cue| cue.confidence)
                .fold(0.0_f64, f64::max);
            if strongest >= 0.90 {
                0.80
            } else if strongest >= 0.80 {
                0.70
            } else {
                0.50
            }
        }
        _ => 0.50,
    }
}

/// 线索目录：先匹配长词，再匹配短词。只收录明确的框架，宁可漏检不可误检。
const CUE_CATALOG: &[(&str, CueKind, f64)] = &[
    ("有多大的可能是", CueKind::Probability, 0.97),
    ("有多大的概率是", CueKind::Probability, 0.97),
    ("有多大概率是", CueKind::Probability, 0.96),
    ("有多大可能是", CueKind::Probability, 0.96),
    ("有多大的可能", CueKind::Probability, 0.95),
    ("有多大的概率", CueKind::Probability, 0.95),
    ("有多大概率", CueKind::Probability, 0.95),
    ("有多大可能", CueKind::Probability, 0.94),
    ("可能性有多大", CueKind::Probability, 0.95),
    ("可能性是多少", CueKind::Probability, 0.95),
    ("可能性多大", CueKind::Probability, 0.95),
    ("概率是多少", CueKind::Probability, 0.95),
    ("概率有多大", CueKind::Probability, 0.95),
    ("概率多大", CueKind::Probability, 0.95),
    ("的可能性", CueKind::Probability, 0.90),
    ("的概率", CueKind::Probability, 0.90),
    ("最有可能", CueKind::MostLikely, 0.94),
    ("最可能", CueKind::MostLikely, 0.93),
    ("更可能", CueKind::Comparison, 0.88),
    ("不一定", CueKind::Possibility, 0.66),
    ("不见得", CueKind::Possibility, 0.62),
    ("未必", CueKind::Possibility, 0.64),
    ("极有可能", CueKind::Possibility, 0.92),
    ("很有可能", CueKind::Possibility, 0.84),
    ("很可能", CueKind::Possibility, 0.80),
    ("比较可能", CueKind::Possibility, 0.64),
    ("不太可能", CueKind::Possibility, 0.70),
    ("不大可能", CueKind::Possibility, 0.70),
    ("不可能", CueKind::Possibility, 0.86),
    ("说不定", CueKind::Possibility, 0.58),
    ("也许", CueKind::Possibility, 0.62),
    ("或许", CueKind::Possibility, 0.62),
    ("恐怕", CueKind::Possibility, 0.56),
    ("可能", CueKind::Possibility, 0.72),
    ("毫无疑问", CueKind::Certainty, 0.90),
    ("一定", CueKind::Certainty, 0.86),
    ("肯定", CueKind::Certainty, 0.83),
    ("必然", CueKind::Certainty, 0.83),
    ("必定", CueKind::Certainty, 0.83),
    ("绝对", CueKind::Certainty, 0.80),
];

fn scan_cues(text: &str) -> Vec<EpistemicCue> {
    let mut catalog: Vec<&(&str, CueKind, f64)> = CUE_CATALOG.iter().collect();
    // 最长匹配优先；同长度保持目录顺序，结果与 HashMap 迭代序无关。
    catalog.sort_by(|a, b| {
        b.0.chars()
            .count()
            .cmp(&a.0.chars().count())
            .then_with(|| a.0.cmp(b.0))
    });
    let boundaries: Vec<usize> = text
        .char_indices()
        .map(|(index, _)| index)
        .chain(std::iter::once(text.len()))
        .collect();
    let mut cues: Vec<EpistemicCue> = Vec::new();
    let mut cursor = 0usize;
    while cursor + 1 < boundaries.len() {
        let start = boundaries[cursor];
        if inside_quote(text, start) {
            cursor += 1;
            continue;
        }
        let mut matched = None;
        for (phrase, kind, confidence) in &catalog {
            if !text[start..].starts_with(phrase) || !cue_allowed(text, start, phrase) {
                continue;
            }
            matched = Some((*phrase, *kind, *confidence));
            break;
        }
        let Some((phrase, kind, confidence)) = matched else {
            cursor += 1;
            continue;
        };
        let end = start + phrase.len();
        let explicitly_negated = phrase.starts_with('不')
            && !matches!(phrase, "不一定" | "不见得");
        let negated = explicitly_negated || preceding_negation(text, start);
        cues.push(EpistemicCue {
            phrase: phrase.to_string(),
            kind,
            confidence,
            negated,
            start,
            end,
        });
        let width = phrase.chars().count();
        cursor += width;
    }
    cues
}

fn cue_allowed(text: &str, start: usize, phrase: &str) -> bool {
    let rest = &text[start + phrase.len()..];
    match phrase {
        // 「可能性」「可能世界」是名词/术语，不是情态算子。
        "可能" => !rest.starts_with('性') && !rest.starts_with("世界"),
        "一定" => {
            !rest.starts_with('的')
                && !rest.starts_with('程')
                && certainty_followed_by_predicate(rest)
        }
        "肯定" | "必然" | "必定" | "绝对" => {
            !rest.starts_with('性')
                && !rest.starts_with('的')
                && certainty_followed_by_predicate(rest)
        }
        _ => true,
    }
}

fn certainty_followed_by_predicate(rest: &str) -> bool {
    match rest.chars().next() {
        None => true,
        Some(next) => {
            matches!(
                next,
                '是' | '会'
                    | '能'
                    | '要'
                    | '有'
                    | '吗'
                    | '吧'
                    | '呀'
                    | '啊'
                    | '？'
                    | '?'
                    | '！'
                    | '!'
                    | '。'
                    | '.'
                    | '，'
                    | ','
            ) || rest
                .trim_end_matches(['？', '?', '！', '!', '。', '.'])
                .ends_with('吗')
        }
    }
}

fn preceding_negation(text: &str, start: usize) -> bool {
    let window: String = text[..start]
        .chars()
        .rev()
        .filter(|ch| !ch.is_whitespace() && !"，。、；：？！,.;:?!".contains(*ch))
        .take(3)
        .collect();
    window.contains('不')
        || window.contains('没')
        || window.contains('未')
        || window.contains('非')
        || window.contains('别')
}

fn inside_quote(text: &str, byte_pos: usize) -> bool {
    let mut depth = 0i32;
    for ch in text[..byte_pos].chars() {
        match ch {
            '“' | '「' | '『' => depth += 1,
            '”' | '」' | '』' => depth = (depth - 1).max(0),
            '"' => depth = if depth == 0 { 1 } else { 0 },
            _ => {}
        }
    }
    depth > 0
}

/// 计数型后验：先验来自小语料计数，似然只依赖线索类别与强度。
///
/// 这不是训练出来的概率模型，而是一张可打印、可审计的计数表；同分时按
/// [`EpistemicMode::code`] 字典序，保证跨进程可复现。
fn posterior(cues: &[EpistemicCue]) -> Vec<(EpistemicMode, f64)> {
    const MODES: [EpistemicMode; 6] = [
        EpistemicMode::Plain,
        EpistemicMode::Possibility,
        EpistemicMode::Probability,
        EpistemicMode::MostLikely,
        EpistemicMode::Certainty,
        EpistemicMode::Comparison,
    ];
    // 小语料先验计数（并非均匀先验：大多数问句仍是直问）。
    let mut scores: HashMap<EpistemicMode, f64> = HashMap::from([
        (EpistemicMode::Plain, 0.45),
        (EpistemicMode::Possibility, 0.20),
        (EpistemicMode::Probability, 0.13),
        (EpistemicMode::MostLikely, 0.10),
        (EpistemicMode::Certainty, 0.08),
        (EpistemicMode::Comparison, 0.04),
    ]);
    // 同一词面重复出现只计一次，避免复读把后验推到 1。
    let mut seen: Vec<(&str, CueKind)> = Vec::new();
    for cue in cues {
        if seen
            .iter()
            .any(|(phrase, kind)| *phrase == cue.phrase && *kind == cue.kind)
        {
            continue;
        }
        seen.push((&cue.phrase, cue.kind));
        let strength = 0.5 + 0.5 * cue.confidence.clamp(0.0, 1.0);
        for mode in MODES {
            let likelihood = likelihood(mode, cue.kind, cue.negated) * strength;
            if let Some(score) = scores.get_mut(&mode) {
                *score *= likelihood.max(1e-9);
            }
        }
    }
    // 固定按 MODES 顺序累加与取数：HashMap 的迭代序会让浮点求和的最后
    // 一位随进程漂移，破坏「同一问句后验逐位可复现」的承诺。
    let total: f64 = MODES
        .iter()
        .filter_map(|mode| scores.get(mode))
        .sum();
    let mut posterior: Vec<(EpistemicMode, f64)> = MODES
        .iter()
        .map(|mode| {
            let score = scores.get(mode).copied().unwrap_or(0.0);
            (*mode, score / total.max(1e-12))
        })
        .collect();
    posterior.sort_by(|a, b| {
        b.1.total_cmp(&a.1)
            .then_with(|| a.0.code().cmp(b.0.code()))
    });
    posterior
}

fn likelihood(mode: EpistemicMode, observed: CueKind, negated: bool) -> f64 {
    let table: [f64; 6] = match observed {
        CueKind::Possibility => [0.02, 0.82, 0.06, 0.03, 0.02, 0.05],
        CueKind::Probability => [0.01, 0.04, 0.90, 0.02, 0.01, 0.02],
        CueKind::MostLikely => [0.01, 0.02, 0.04, 0.90, 0.01, 0.02],
        CueKind::Certainty => [0.08, 0.03, 0.01, 0.01, 0.84, 0.03],
        CueKind::Comparison => [0.01, 0.01, 0.02, 0.08, 0.00, 0.88],
    };
    let index = match mode {
        EpistemicMode::Plain => 0,
        EpistemicMode::Possibility => 1,
        EpistemicMode::Probability => 2,
        EpistemicMode::MostLikely => 3,
        EpistemicMode::Certainty => 4,
        EpistemicMode::Comparison => 5,
    };
    let mut value = table[index];
    // 否定会削弱「确定」与「最可能」的读数，把质量推向可能问。
    if negated {
        value *= match mode {
            EpistemicMode::Certainty | EpistemicMode::MostLikely => 0.25,
            EpistemicMode::Possibility => 1.20,
            _ => 0.90,
        };
    }
    value.max(1e-9)
}

fn clean_question(question: &str) -> String {
    let mut text = question.trim();
    for prefix in [
        "请问一下",
        "请问",
        "我想问一下",
        "我想问",
        "想问一下",
        "想问",
        "问一下",
        "想知道",
    ] {
        if let Some(rest) = text.strip_prefix(prefix) {
            text = rest.trim_start_matches(['，', ',', '：', ':', ' ']);
            break;
        }
    }
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn strip_question_tail(raw: &str) -> String {
    let mut text = raw
        .trim()
        .trim_end_matches(['？', '?', '！', '!', '。', '.', '，', ',', '；', ';', '：', ':', ' ']);
    loop {
        match text.chars().last() {
            Some('吗') | Some('呢') | Some('吧') | Some('呀') | Some('啊') | Some('嘛') => {
                text = &text[..text.len() - text.chars().last().unwrap().len_utf8()];
                text = text.trim();
            }
            _ => break,
        }
    }
    text.to_string()
}

fn trim_leading_phrases<'a>(raw: &'a str, phrases: &[&str]) -> &'a str {
    let mut text = raw.trim();
    let mut changed = true;
    while changed {
        changed = false;
        for phrase in phrases {
            if let Some(rest) = text.strip_prefix(phrase) {
                text = rest.trim();
                changed = true;
            }
        }
    }
    text
}

fn contains_interrogative(text: &str) -> bool {
    text.chars()
        .any(|ch| matches!(ch, '谁' | '什' | '么' | '哪' | '几' | '多' | '少' | '吗' | '呢'))
}

fn parse_claim(raw: &str) -> Option<Claim> {
    let text = strip_question_tail(raw);
    let (left, right) = text.split_once('是')?;
    let mut left = left.trim();
    let mut negated = false;
    if let Some(rest) = left.strip_suffix('不') {
        left = rest.trim();
        negated = true;
    }
    let right = right.trim();
    if left.is_empty() || right.is_empty() || left.contains('是') || right.contains('是') {
        return None;
    }
    let left_possessive = left.rsplit_once('的');
    let right_possessive = right.rsplit_once('的');
    // 「A 的 P 是 B」→ P(A, B)；「A 是 B 的 P」→ P(B, A)。
    // 后者不能读成 P(A, B)：那会把「李四是王五的师傅」翻成
    // 「李四的师傅是王五」，与 QueryPlanner/图谱的主客方向相反。
    let (subject, predicate, object) = match (left_possessive, right_possessive) {
        (Some((subject, predicate)), None) => (subject, predicate, right),
        (None, Some((subject, predicate))) => (subject, predicate, left),
        // 两侧都带「的」是歧义结构（A 的 P 是 B 的 Q），不强行拍板。
        _ => return None,
    };
    let subject = subject.trim();
    let predicate = predicate.trim();
    let object = object.trim();
    if subject.is_empty() || predicate.is_empty() || object.is_empty() {
        return None;
    }
    if [subject, predicate, object]
        .iter()
        .any(|part| part.contains('的') || contains_interrogative(part))
    {
        return None;
    }
    Some(Claim {
        subject: subject.to_string(),
        predicate: predicate.to_string(),
        object: object.to_string(),
        negated,
    })
}

fn parse_possessive(raw: &str) -> Option<(String, String)> {
    let text = strip_question_tail(raw);
    let text = trim_leading_phrases(&text, &["是", "为", "成为", "当", "做"]);
    let (subject, predicate) = text.rsplit_once('的')?;
    let subject = subject.trim();
    let predicate = predicate.trim();
    if subject.is_empty() || predicate.is_empty() {
        return None;
    }
    if subject.contains('的') || predicate.contains('的') {
        return None;
    }
    Some((subject.to_string(), predicate.to_string()))
}

fn extract_target(
    text: &str,
    cues: &[EpistemicCue],
    mode: EpistemicMode,
) -> Option<EpistemicTarget> {
    match mode {
        EpistemicMode::Comparison => parse_comparison(text, cues),
        EpistemicMode::Probability => parse_probability_target(text),
        EpistemicMode::MostLikely => parse_most_likely_target(text, cues),
        EpistemicMode::Certainty | EpistemicMode::Possibility => {
            claim_without_cues(text, cues).map(EpistemicTarget::GroundClaim)
        }
        EpistemicMode::Plain => None,
    }
}

const PROBABILITY_FRAMES: &[&str] = &[
    "有多大的可能是",
    "有多大的概率是",
    "有多大概率是",
    "有多大可能是",
    "有多大的可能",
    "有多大的概率",
    "有多大概率",
    "有多大可能",
    "可能性有多大",
    "可能性是多少",
    "可能性多大",
    "概率是多少",
    "概率有多大",
    "概率多大",
    "的可能性",
    "的概率",
];

fn parse_probability_target(raw: &str) -> Option<EpistemicTarget> {
    let text = clean_question(raw);
    for frame in PROBABILITY_FRAMES {
        let Some((left, right)) = text.split_once(frame) else {
            continue;
        };
        let left = left.trim();
        let right = right.trim();
        if left.is_empty() {
            if let Some(claim) = parse_claim(right) {
                return Some(EpistemicTarget::GroundClaim(claim));
            }
            continue;
        }
        if right.is_empty() {
            if let Some(claim) = parse_claim(left) {
                return Some(EpistemicTarget::GroundClaim(claim));
            }
            continue;
        }
        let right = trim_leading_phrases(right, &["是", "成为", "当", "做"]);
        // 「李四有多大的可能不是王五的师傅」：否定词属于断言，不能拼成
        // 「李四是不是…」这种双重系词。
        let candidate = if right.starts_with('不') || right.starts_with('没') {
            format!("{}{}", left, right)
        } else {
            format!("{}是{}", left, right)
        };
        if let Some(claim) = parse_claim(&candidate) {
            return Some(EpistemicTarget::GroundClaim(claim));
        }
        if let Some(claim) = parse_claim(left) {
            return Some(EpistemicTarget::GroundClaim(claim));
        }
    }
    claim_without_cues(&text, &[]).map(EpistemicTarget::GroundClaim)
}

fn parse_most_likely_target(raw: &str, cues: &[EpistemicCue]) -> Option<EpistemicTarget> {
    let cue = cues
        .iter()
        .find(|cue| cue.kind == CueKind::MostLikely)?;
    let prefix = clean_question(&raw[..cue.start]);
    let prefix = strip_comparison_question_suffix(&prefix).to_string();
    let suffix = clean_question(&raw[cue.end..]);
    let suffix = trim_leading_phrases(&suffix, &["的是", "是", "为", "成为", "当", "做"]);
    let suffix = strip_most_likely_tail(&suffix).to_string();
    if suffix.is_empty() {
        return None;
    }
    // 「A 和 B 谁最可能是 X 的 P」不是开放式 argmax，而是两个具名候选的
    // 概率比较；把它读成全图排序会丢掉用户点名的 B。
    if let Some((left_subject, right_subject)) = split_comparison_subjects(&prefix) {
        let (anchor, predicate) = parse_possessive(&suffix)?;
        if anchor == "谁" || predicate.is_empty() {
            return None;
        }
        return Some(EpistemicTarget::Comparison {
            left: Claim {
                subject: anchor.clone(),
                predicate: predicate.clone(),
                object: left_subject,
                negated: false,
            },
            right: Claim {
                subject: anchor,
                predicate,
                object: right_subject,
                negated: false,
            },
        });
    }
    // 谁最可能是 X 的 P / 最可能是 X 的 P / 最可能的是 X 的 P
    if prefix.is_empty() || prefix == "谁" {
        if let Some((subject, predicate)) = parse_possessive(&suffix) {
            if subject != "谁" && !contains_interrogative(&subject) && !contains_interrogative(&predicate)
            {
                return Some(EpistemicTarget::ObjectsOf { subject, predicate });
            }
        }
    }
    // 李四最可能是谁的 P：「李四 是 谁的 P」= 谁的 P 是李四 = P(?X, 李四)，
    // 不能读成 P(李四, ?X)；后者会答成「李四的师傅是谁」。
    if let Some(rest) = suffix.strip_prefix("谁") {
        let predicate = trim_leading_phrases(rest, &["的"]);
        if !prefix.is_empty()
            && !prefix.contains('谁')
            && !predicate.is_empty()
            && !contains_interrogative(predicate)
        {
            return Some(EpistemicTarget::SubjectsOf {
                object: prefix,
                predicate: predicate.to_string(),
            });
        }
    }
    // 李四最可能认识谁
    if let Some(predicate) = suffix.strip_suffix("谁") {
        let predicate = predicate.trim();
        if !prefix.is_empty()
            && prefix != "谁"
            && !prefix.contains('谁')
            && !predicate.is_empty()
            && !predicate.contains('的')
        {
            return Some(EpistemicTarget::ObjectsOf {
                subject: prefix,
                predicate: predicate.to_string(),
            });
        }
    }
    // 李四最可能是王五的师傅：单个断言的概率，而不是候选 argmax。
    if !prefix.is_empty()
        && prefix != "谁"
        && !prefix.contains('谁')
        && !suffix.contains('谁')
    {
        return parse_claim(&format!("{}是{}", prefix, suffix)).map(EpistemicTarget::GroundClaim);
    }
    None
}

fn parse_comparison(raw: &str, cues: &[EpistemicCue]) -> Option<EpistemicTarget> {
    let cue = cues
        .iter()
        .find(|cue| cue.kind == CueKind::Comparison)?;
    let cleaned = clean_question(&raw[..cue.start]);
    let prefix = trim_leading_phrases(&cleaned, &["比较", "对比"]);
    let prefix = strip_comparison_question_suffix(prefix);
    let (left_subject, right_subject) = split_comparison_subjects(prefix)?;
    let suffix = clean_question(&raw[cue.end..]);
    let suffix = trim_leading_phrases(&suffix, &["的是", "是", "为", "成为", "当", "做"]);
    let (anchor, predicate) = parse_possessive(&suffix)?;
    if anchor == "谁" || predicate.is_empty() {
        return None;
    }
    let left = Claim {
        subject: anchor.clone(),
        predicate: predicate.clone(),
        object: left_subject,
        negated: false,
    };
    let right = Claim {
        subject: anchor,
        predicate,
        object: right_subject,
        negated: false,
    };
    Some(EpistemicTarget::Comparison { left, right })
}

/// 比较问的两个具名候选：`A和B`、`A跟B`、`A与B`、`A同B`。
/// 只接受非空两侧；不猜测超过两个候选的集合。
fn split_comparison_subjects(prefix: &str) -> Option<(String, String)> {
    for separator in ['和', '跟', '与', '同'] {
        if let Some((left, right)) = prefix.split_once(separator) {
            let left = left.trim();
            let right = right.trim();
            if !left.is_empty()
                && !right.is_empty()
                && !left.contains(['谁', '哪'])
                && !right.contains(['谁', '哪'])
            {
                return Some((left.to_string(), right.to_string()));
            }
        }
    }
    None
}

fn strip_comparison_question_suffix(prefix: &str) -> &str {
    let mut prefix = prefix.trim();
    for suffix in ["谁", "哪一个", "哪个"] {
        if let Some(rest) = prefix.strip_suffix(suffix) {
            prefix = rest.trim();
            break;
        }
    }
    prefix
}

/// 「最可能是李四的师傅**的人是谁**」里的尾部是问句外壳，不是谓词的一部分。
fn strip_most_likely_tail(suffix: &str) -> &str {
    let mut suffix = suffix.trim();
    for tail in ["的那个人是谁", "的人是谁", "的那个人", "的人", "的是谁", "是谁"] {
        if let Some(rest) = suffix.strip_suffix(tail) {
            let rest = rest.trim();
            if !rest.is_empty() {
                suffix = rest;
                break;
            }
        }
    }
    suffix
}

fn claim_without_cues(text: &str, cues: &[EpistemicCue]) -> Option<Claim> {
    let stripped = remove_cue_spans(text, cues);
    parse_claim(&stripped)
}

fn remove_cue_spans(text: &str, cues: &[EpistemicCue]) -> String {
    if cues.is_empty() {
        return text.to_string();
    }
    let mut spans: Vec<(usize, usize)> = cues.iter().map(|cue| (cue.start, cue.end)).collect();
    spans.sort_unstable();
    let mut output = String::with_capacity(text.len());
    let mut cursor = 0usize;
    for (start, end) in spans {
        if start < cursor {
            continue;
        }
        output.push_str(&text[cursor..start]);
        cursor = end;
    }
    output.push_str(&text[cursor..]);
    output.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn normalize_core(text: &str, cues: &[EpistemicCue], mode: EpistemicMode) -> String {
    let stripped = remove_cue_spans(text, cues);
    let stripped = stripped.trim_end_matches([
        '？', '?', '！', '!', '。', '.', '，', ',', '；', ';', '：', ':', ' ',
    ]);
    if !stripped.is_empty() {
        return stripped.to_string();
    }
    // 整句都是情态框架（例如只说了「最可能」）：保留原句，交给调用方弃权。
    if mode != EpistemicMode::Plain {
        return text.to_string();
    }
    text.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_questions_are_not_hijacked() {
        for question in [
            "李四的师傅是谁",
            "李四的师傅是王五吗",
            "张三和李四什么关系",
            "大概有多少人",
            "他一定程度的努力没有白费",
        ] {
            assert!(
                ProbabilisticQueryIntent::analyze(question).is_none(),
                "普通问句不应被情态层接管：{}",
                question
            );
        }
    }

    #[test]
    fn possibility_keeps_the_reference_plan() {
        let intent = ProbabilisticQueryIntent::analyze("李四的师傅可能是谁呢？").unwrap();
        assert_eq!(intent.mode, EpistemicMode::Possibility);
        assert_eq!(intent.core_question, "李四的师傅是谁呢");
        assert!(intent.confidence > 0.7);
        assert!(!intent.negated);
        assert!(intent.target.is_none());
    }

    #[test]
    fn probability_frame_extracts_a_ground_claim() {
        let intent =
            ProbabilisticQueryIntent::analyze("李四有多大的可能是王五的师傅？").unwrap();
        assert_eq!(intent.mode, EpistemicMode::Probability);
        assert_eq!(
            intent.target,
            Some(EpistemicTarget::GroundClaim(Claim {
                subject: "王五".to_string(),
                predicate: "师傅".to_string(),
                object: "李四".to_string(),
                negated: false,
            }))
        );
        assert!(intent.confidence > 0.85);
    }

    #[test]
    fn probability_suffix_extracts_a_ground_claim() {
        let intent =
            ProbabilisticQueryIntent::analyze("李四的师傅是王五的概率是多少").unwrap();
        assert_eq!(intent.mode, EpistemicMode::Probability);
        assert_eq!(
            intent.target,
            Some(EpistemicTarget::GroundClaim(Claim {
                subject: "李四".to_string(),
                predicate: "师傅".to_string(),
                object: "王五".to_string(),
                negated: false,
            }))
        );
    }

    #[test]
    fn certainty_extracts_a_ground_claim_and_uses_a_high_threshold() {
        let intent = ProbabilisticQueryIntent::analyze("李四一定是王五的师傅吗").unwrap();
        assert_eq!(intent.mode, EpistemicMode::Certainty);
        assert_eq!(
            intent.target,
            Some(EpistemicTarget::GroundClaim(Claim {
                subject: "王五".to_string(),
                predicate: "师傅".to_string(),
                object: "李四".to_string(),
                negated: false,
            }))
        );
        assert!(intent.threshold >= 0.95);
    }

    #[test]
    fn most_likely_ranks_the_objects_of_a_predicate() {
        let intent = ProbabilisticQueryIntent::analyze("谁最可能是李四的师傅").unwrap();
        assert_eq!(intent.mode, EpistemicMode::MostLikely);
        assert_eq!(
            intent.target,
            Some(EpistemicTarget::ObjectsOf {
                subject: "李四".to_string(),
                predicate: "师傅".to_string(),
            })
        );
    }

    #[test]
    fn most_likely_accepts_the_shi_de_frame() {
        let intent = ProbabilisticQueryIntent::analyze("最可能的是李四的师傅").unwrap();
        assert_eq!(
            intent.target,
            Some(EpistemicTarget::ObjectsOf {
                subject: "李四".to_string(),
                predicate: "师傅".to_string(),
            })
        );
    }

    #[test]
    fn most_likely_handles_the_inverted_question_word_position() {
        let intent = ProbabilisticQueryIntent::analyze("李四最可能是谁的师傅").unwrap();
        assert_eq!(
            intent.target,
            Some(EpistemicTarget::SubjectsOf {
                object: "李四".to_string(),
                predicate: "师傅".to_string(),
            })
        );
    }

    #[test]
    fn comparison_extracts_two_ground_claims() {
        let intent =
            ProbabilisticQueryIntent::analyze("李四和王五谁更可能是张三的师傅").unwrap();
        assert_eq!(intent.mode, EpistemicMode::Comparison);
        let Some(EpistemicTarget::Comparison { left, right }) = intent.target else {
            panic!("应抽出两个待比较的基原子：{:?}", intent.target);
        };
        assert_eq!(left.subject, "张三");
        assert_eq!(right.subject, "张三");
        assert_eq!(left.predicate, "师傅");
        assert_eq!(left.object, "李四");
        assert_eq!(right.object, "王五");
    }

    #[test]
    fn negation_is_kept_in_scope() {
        let intent =
            ProbabilisticQueryIntent::analyze("李四不可能是王五的师傅").unwrap();
        assert_eq!(intent.mode, EpistemicMode::Possibility);
        assert!(intent.negated);
        assert_eq!(
            intent.target,
            Some(EpistemicTarget::GroundClaim(Claim {
                subject: "王五".to_string(),
                predicate: "师傅".to_string(),
                object: "李四".to_string(),
                negated: false,
            }))
        );
    }

    #[test]
    fn copula_negation_stays_inside_the_claim() {
        let intent = ProbabilisticQueryIntent::analyze("李四一定不是王五的师傅吗").unwrap();
        assert_eq!(intent.mode, EpistemicMode::Certainty);
        let Some(EpistemicTarget::GroundClaim(claim)) = intent.target else {
            panic!("应抽出带否定的基原子");
        };
        assert_eq!(claim.subject, "王五");
        assert_eq!(claim.predicate, "师傅");
        assert_eq!(claim.object, "李四");
        assert!(claim.negated);
    }

    #[test]
    fn probability_frame_keeps_a_negated_copula() {
        let intent =
            ProbabilisticQueryIntent::analyze("李四有多大的可能不是王五的师傅").unwrap();
        let Some(EpistemicTarget::GroundClaim(claim)) = intent.target else {
            panic!("应抽出带否定的基原子");
        };
        assert_eq!(claim.subject, "王五");
        assert_eq!(claim.predicate, "师傅");
        assert_eq!(claim.object, "李四");
        assert!(claim.negated);
    }

    #[test]
    fn comparison_accepts_common_coordinators() {
        for question in [
            "李四和王五谁更可能是张三的师傅",
            "李四跟王五谁更可能是张三的师傅",
            "李四与王五谁更可能是张三的师傅",
            "李四同王五谁更可能是张三的师傅",
        ] {
            let intent = ProbabilisticQueryIntent::analyze(question).unwrap();
            assert_eq!(intent.mode, EpistemicMode::Comparison, "{}", question);
            let Some(EpistemicTarget::Comparison { left, right }) = intent.target else {
                panic!("应抽出两个待比较的基原子：{} => {:?}", question, intent.target);
            };
            assert_eq!(left.subject, "张三");
            assert_eq!(right.subject, "张三");
            assert_eq!(left.object, "李四");
            assert_eq!(right.object, "王五");
        }
    }

    #[test]
    fn most_likely_with_two_named_candidates_is_a_comparison() {
        let intent =
            ProbabilisticQueryIntent::analyze("李四和王五谁最可能是张三的师傅").unwrap();
        assert_eq!(intent.mode, EpistemicMode::MostLikely);
        let Some(EpistemicTarget::Comparison { left, right }) = intent.target else {
            panic!("两个具名候选应做比较而不是全图 argmax：{:?}", intent.target);
        };
        assert_eq!(left.subject, "张三");
        assert_eq!(right.subject, "张三");
        assert_eq!(left.predicate, "师傅");
        assert_eq!(left.object, "李四");
    }

    #[test]
    fn most_likely_accepts_the_person_tail() {
        let intent =
            ProbabilisticQueryIntent::analyze("最可能是李四的师傅的人是谁").unwrap();
        assert_eq!(
            intent.target,
            Some(EpistemicTarget::ObjectsOf {
                subject: "李四".to_string(),
                predicate: "师傅".to_string(),
            })
        );
    }

    #[test]
    fn quoted_modality_is_not_the_users_intent() {
        assert!(ProbabilisticQueryIntent::analyze("他说“可能”是什么意思").is_none());
    }

    #[test]
    fn posterior_is_deterministic_and_normalized() {
        let first = ProbabilisticQueryIntent::analyze("李四的师傅可能是谁").unwrap();
        for _ in 0..16 {
            let again = ProbabilisticQueryIntent::analyze("李四的师傅可能是谁").unwrap();
            assert_eq!(first.posterior, again.posterior);
        }
        let sum: f64 = first.posterior.iter().map(|(_, p)| *p).sum();
        assert!((sum - 1.0).abs() < 1e-9);
        assert_eq!(first.posterior[0].0, first.mode);
    }

    #[test]
    fn ledger_separates_understanding_from_fact_probability() {
        let intent = ProbabilisticQueryIntent::analyze("李四可能是王五的师傅").unwrap();
        let ledger = intent.render_ledger();
        assert!(ledger.contains("理解置信度"));
        assert!(ledger.contains("不是事实概率"));
    }
}

// 中文概率知识理解（符号主义 NLP：连词、模糊量词、百分数、命题结构）
//
// 灵感：中文的概率表达不像英语有统一的 P(...) 记号，而是藏在
//   • 条件连词里：「如果…那么…」「只要…就…」「若…则…」「一旦…就…」；
//   • 因果连词里：「因为…所以…」「由于…因此…」；
//   • 模糊量词里：「总是/一定/大多数/通常/很可能/有时/偶尔/很少/从不」；
//   • 比例表达里：「百分之八十」「八成」「80%」。
// 本模块把这些自然语言句子解析成 ProbLog 风格的符号知识：
//   条件/因果句 → 概率规则 head :- body（概率来自量词，缺省时按连词取默认值）；
//   比例陈述   → 概率事实 p::atom；
//   直陈句     → 确定事实。
// 全过程只有字符串切分、连词表、中文数字解析与合一——没有神经网络、没有训练。

use crate::problog::ProblogEngine;
use crate::resolution_engine::{Atom, Clause, Term};

/// 一条从中文句子解析出的概率知识
#[derive(Debug, Clone, PartialEq)]
pub struct ChineseLesson {
    pub kind: LessonKind,
    pub head: Atom,
    pub body: Vec<Atom>,
    pub probability: f64,
    pub source: String,
    pub original: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LessonKind {
    /// 确定事实：直陈句，无概率词
    DeterministicFact,
    /// 概率事实：比例/量词修饰的单句
    ProbabilisticFact,
    /// 条件规则：如果/只要/若/一旦…
    ConditionalRule,
    /// 因果规则：因为/由于…
    CausalRule,
}

/// 解析结果：成功的知识与逐条失败原因（用于教学报告）
#[derive(Debug, Clone, Default)]
pub struct LessonParseOutcome {
    pub lessons: Vec<ChineseLesson>,
    pub errors: Vec<String>,
}

/// 模糊量词表（长词优先，避免「总是」抢先匹配「几乎总是」的尾巴）
const QUANTIFIERS: &[(&str, f64)] = &[
    ("几乎总是", 0.95),
    ("几乎一定", 0.95),
    ("一般来说", 0.80),
    ("大部分", 0.85),
    ("大多数", 0.85),
    ("很可能", 0.75),
    ("有时候", 0.40),
    ("总是", 0.98),
    ("一定", 0.97),
    ("必然", 0.98),
    ("必定", 0.98),
    ("通常", 0.80),
    ("多半", 0.75),
    ("有时", 0.40),
    ("偶尔", 0.25),
    ("很少", 0.15),
    ("极少", 0.10),
    ("从不", 0.02),
    ("绝不", 0.01),
    ("十成", 1.00),
    ("九成", 0.90),
    ("八成", 0.80),
    ("七成", 0.70),
    ("六成", 0.60),
    ("五成", 0.50),
    ("四成", 0.40),
    ("三成", 0.30),
    ("两成", 0.20),
    ("一成", 0.10),
    ("大概", 0.60),
    ("可能", 0.50),
    ("也许", 0.50),
    ("或许", 0.50),
];

const UNARY_VERBS: [&str; 12] = [
    "会", "能", "要", "喜欢", "讨厌", "经常", "通过", "属于", "位于", "参加", "知道", "相信",
];

fn default_rule_probability(kind: LessonKind) -> f64 {
    match kind {
        LessonKind::ConditionalRule => 0.95,
        LessonKind::CausalRule => 0.90,
        _ => 1.0,
    }
}

fn is_cn_digit(ch: char) -> bool {
    matches!(
        ch,
        '零' | '〇' | '一' | '二' | '两' | '三' | '四' | '五' | '六' | '七' | '八' | '九' | '十'
            | '百' | '千'
    )
}

/// 中文数字 → 整数（支持到千位：十五、二十、八十五、一百零八、九十九…）
pub fn parse_chinese_number(text: &str) -> Option<u32> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.chars().all(|ch| ch.is_ascii_digit()) {
        return trimmed.parse::<u32>().ok();
    }
    let mut result: u32 = 0;
    let mut current: u32 = 0;
    let mut seen_digit = false;
    for ch in trimmed.chars() {
        let digit = match ch {
            '零' | '〇' => {
                seen_digit = true;
                continue;
            }
            '一' => 1,
            '二' | '两' => 2,
            '三' => 3,
            '四' => 4,
            '五' => 5,
            '六' => 6,
            '七' => 7,
            '八' => 8,
            '九' => 9,
            '十' => {
                result += if current == 0 { 10 } else { current * 10 };
                current = 0;
                seen_digit = true;
                continue;
            }
            '百' => {
                result += if current == 0 { 100 } else { current * 100 };
                current = 0;
                seen_digit = true;
                continue;
            }
            '千' => {
                result += if current == 0 { 1000 } else { current * 1000 };
                current = 0;
                seen_digit = true;
                continue;
            }
            _ => return None,
        };
        if seen_digit && current != 0 {
            return None;
        }
        current = digit;
        seen_digit = true;
    }
    Some(result + current)
}

/// 从句子中抽取概率（显式百分数优先于模糊量词），返回（概率, 去掉概率词后的句子）
pub fn extract_probability(text: &str) -> Option<(f64, String)> {
    let mut percent_candidates: Vec<(usize, usize, f64)> = Vec::new();
    let mut word_candidates: Vec<(usize, usize, f64)> = Vec::new();
    // 百分之八十 / 百分之 80
    if let Some(pos) = text.find("百分之") {
        let after_raw = &text[pos + "百分之".len()..];
        // 允许「百分之 80」中的空白：跳过并计入被移除片段的长度
        let whitespace_len: usize = after_raw
            .chars()
            .take_while(|ch| ch.is_whitespace())
            .map(|ch| ch.len_utf8())
            .sum();
        let after = &after_raw[whitespace_len..];
        let cn: String = after.chars().take_while(|ch| is_cn_digit(*ch)).collect();
        if let Some(value) = parse_chinese_number(&cn).filter(|value| *value <= 100) {
            percent_candidates.push((
                pos,
                "百分之".len() + whitespace_len + cn.len(),
                value as f64 / 100.0,
            ));
        } else {
            let digits: String = after.chars().take_while(|ch| ch.is_ascii_digit()).collect();
            if let Some(value) = digits.parse::<u32>().ok().filter(|value| *value <= 100) {
                percent_candidates.push((
                    pos,
                    "百分之".len() + whitespace_len + digits.len(),
                    value as f64 / 100.0,
                ));
            }
        }
    }
    // 80% / 80％
    let mut previous_is_digit = false;
    for (pos, ch) in text.char_indices() {
        let is_digit = ch.is_ascii_digit();
        // 只从数字串的开头匹配，避免「180%」被截成「80%」
        let starts_digit_run = is_digit && !previous_is_digit;
        previous_is_digit = is_digit;
        if !starts_digit_run {
            continue;
        }
        let digits: String = text[pos..].chars().take_while(|ch| ch.is_ascii_digit()).collect();
        let rest = &text[pos + digits.len()..];
        let percent_len = if rest.starts_with('%') {
            1
        } else if rest.starts_with('％') {
            '％'.len_utf8()
        } else {
            0
        };
        if percent_len == 0 {
            continue;
        }
        if let Some(value) = digits.parse::<u32>().ok().filter(|value| *value <= 100) {
            percent_candidates.push((pos, digits.len() + percent_len, value as f64 / 100.0));
        }
    }
    // 模糊量词（跳过被「不/没」否定的词与「可能性」这类词内匹配）
    for &(word, probability) in QUANTIFIERS {
        if let Some(pos) = text.find(word) {
            let negated = text[..pos].ends_with('不') || text[..pos].ends_with('没');
            let inside_word = text[pos + word.len()..].starts_with('性');
            if !negated && !inside_word {
                word_candidates.push((pos, word.len(), probability));
            }
        }
    }
    let pick = |mut candidates: Vec<(usize, usize, f64)>| {
        candidates.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| b.1.cmp(&a.1)));
        candidates.into_iter().next()
    };
    let chosen = pick(percent_candidates).or_else(|| pick(word_candidates))?;
    let (position, length, probability) = chosen;
    let mut cleaned = String::with_capacity(text.len());
    cleaned.push_str(&text[..position]);
    cleaned.push_str(&text[position + length..]);
    Some((probability, cleaned))
}

fn strip_probability_wrappers(text: &str) -> String {
    let mut cleaned = text.to_string();
    for wrapper in [
        "的可能性是",
        "的可能性为",
        "的概率是",
        "的概率为",
        "可能性是",
        "可能性为",
        "概率是",
        "概率为",
        "的可能性",
        "的概率",
    ] {
        // 不存在时 replace 会原样返回并白白分配一次，先检查可跳过
        if cleaned.contains(wrapper) {
            cleaned = cleaned.replace(wrapper, "");
        }
    }
    cleaned
}

fn clean_sentence(text: &str) -> String {
    text.trim()
        .trim_matches(|ch: char| {
            matches!(
                ch,
                '。' | '！' | '!' | '？' | '?' | '，' | ',' | '、' | '：' | ':' | '“' | '”' | '"'
                    | '\'' | '\n' | '\r' | '\t' | ' '
            )
        })
        .to_string()
}

fn contains_question_word(text: &str) -> bool {
    // 多字疑问词不会与普通词冲突，直接子串匹配
    const UNAMBIGUOUS: [&str; 8] = ["吗", "呢", "谁", "什么", "怎么", "为什么", "如何", "多少"];
    if UNAMBIGUOUS.iter().any(|marker| text.contains(marker)) {
        return true;
    }
    // 单字疑问词需排除构词：几（几乎/几何/几内亚）、哪（哪吒）
    for (pos, ch) in text.char_indices() {
        let rest = &text[pos + ch.len_utf8()..];
        match ch {
            '几' => {
                if !(rest.starts_with('乎') || rest.starts_with('何') || rest.starts_with('内')) {
                    return true;
                }
            }
            '哪' => {
                if !(rest.starts_with('吒') || rest.starts_with('咤')) {
                    return true;
                }
            }
            _ => {}
        }
    }
    // 「几何」在句末是疑问（价值几何），在句首/句中通常作名词（几何是数学分支）
    text.ends_with("几何")
}

fn valid_part(part: &str) -> bool {
    let length = part.chars().count();
    (1..=24).contains(&length)
}

/// 中文命题 → 基原子：
///   S的P是O / O是S的P / S是O / S会V（一元）/ 谓词(参数, …) 函数式写法
pub fn parse_proposition(text: &str) -> Option<Atom> {
    let text = clean_sentence(text);
    if text.is_empty() {
        return None;
    }
    if let Some(open) = text.find('(') {
        if let Some(inner) = text.strip_suffix(')') {
            let predicate = text[..open].trim();
            let arguments = &inner[open + 1..];
            let terms: Vec<Term> = arguments
                .split([',', '，'])
                .map(|argument| Term::constant(argument.trim()))
                .collect();
            if !predicate.is_empty()
                && !terms.is_empty()
                && terms.iter().all(|term| !term.name().is_empty())
            {
                return Some(Atom::new(predicate, terms));
            }
        }
    }
    if let Some(position_de) = text.find('的') {
        let after_de = &text[position_de + '的'.len_utf8()..];
        if let Some(position_shi) = after_de.find('是') {
            let subject = text[..position_de].trim();
            let predicate = after_de[..position_shi].trim();
            let object = after_de[position_shi + '是'.len_utf8()..].trim();
            if valid_part(subject) && valid_part(predicate) && valid_part(object) {
                return Some(Atom::rel(predicate, subject, object));
            }
        }
        if let Some(position_shi) = text[..position_de].rfind('是') {
            let object = text[..position_shi].trim();
            let subject = text[position_shi + '是'.len_utf8()..position_de].trim();
            let predicate = after_de.trim();
            if valid_part(subject) && valid_part(predicate) && valid_part(object) {
                return Some(Atom::rel(predicate, subject, object));
            }
        }
    }
    if let Some(position_shi) = text.find('是') {
        let subject = text[..position_shi].trim();
        let object = text[position_shi + '是'.len_utf8()..].trim();
        if valid_part(subject) && valid_part(object) {
            return Some(Atom::rel("是", subject, object));
        }
    }
    for verb in UNARY_VERBS {
        if let Some(position) = text.find(verb) {
            if position == 0 {
                continue;
            }
            let subject = text[..position].trim();
            let predicate = format!("{}{}", verb, text[position + verb.len()..].trim());
            if valid_part(subject) && predicate.chars().count() > verb.chars().count() {
                return Some(Atom::unary(&predicate, Term::constant(subject)));
            }
        }
    }
    None
}

/// 通用规则的占位符识别：X/Y/Z、A/B/C、甲乙丙…、某人/某个
/// 例：「如果X的师傅是Y，那么X的老师是Y」中的 X/Y 应被变量化
pub fn is_placeholder(name: &str) -> bool {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return false;
    }
    if trimmed.starts_with("某人") || trimmed.starts_with("某个") {
        return true;
    }
    let mut chars = trimmed.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if chars.next().is_some() {
        return false;
    }
    first.is_ascii_alphabetic()
        || matches!(
            first,
            '甲' | '乙' | '丙' | '丁' | '戊' | '己' | '庚' | '辛' | '壬' | '癸'
        )
}

/// 把命题中的占位符常量泛化为逻辑变量（同名占位符共享同一变量）
pub fn generalize_atom(atom: &Atom) -> Atom {
    let terms: Vec<Term> = atom
        .terms
        .iter()
        .map(|term| match term {
            Term::Const(name) if is_placeholder(name) => Term::var(name),
            other => other.clone(),
        })
        .collect();
    Atom::new(&atom.predicate, terms)
}

/// 规则中是否存在被泛化的变量
pub fn lesson_has_variables(lesson: &ChineseLesson) -> bool {
    lesson
        .head
        .terms
        .iter()
        .chain(lesson.body.iter().flat_map(|atom| atom.terms.iter()))
        .any(|term| match term {
            Term::Const(name) => is_placeholder(name),
            Term::Var(_) => true,
        })
}

/// 条件/因果知识 → Horn 子句（含占位符时自动变量化，供后向链归结使用）
pub fn lesson_to_clause(lesson: &ChineseLesson) -> Option<Clause> {
    match lesson.kind {
        LessonKind::ConditionalRule | LessonKind::CausalRule => Some(Clause::new(
            generalize_atom(&lesson.head),
            lesson
                .body
                .iter()
                .map(generalize_atom)
                .collect(),
            lesson.probability,
            &lesson.source,
        )),
        _ => None,
    }
}

/// 中文命题的生成式回读：Atom → 自然语言
pub fn render_atom_zh(atom: &Atom) -> String {
    let names: Vec<&str> = atom.terms.iter().map(|term| term.name()).collect();
    match names.as_slice() {
        [subject, object] if &*atom.predicate == "是" => format!("{}是{}", subject, object),
        [subject, object] => format!("{}的{}是{}", subject, atom.predicate, object),
        [subject] => {
            let predicate = &*atom.predicate;
            if UNARY_VERBS.iter().any(|verb| predicate.starts_with(*verb)) {
                format!("{}{}", subject, predicate)
            } else {
                format!("{}是{}", subject, predicate)
            }
        }
        _ => atom.display(),
    }
}

/// 条件/因果句切分：返回（类型, 前件文本, 后件文本）
fn split_conditional(text: &str) -> Option<(LessonKind, String, String)> {
    const PATTERNS: [(&str, LessonKind, &[&str]); 8] = [
        (
            "如果",
            LessonKind::ConditionalRule,
            &["那么", "就会", "就", "则", "，", ","],
        ),
        (
            "假如",
            LessonKind::ConditionalRule,
            &["那么", "就会", "就", "则", "，", ","],
        ),
        (
            "只要",
            LessonKind::ConditionalRule,
            &["那么", "就会", "就", "则", "，", ","],
        ),
        (
            "一旦",
            LessonKind::ConditionalRule,
            &["那么", "就会", "就", "则", "，", ","],
        ),
        (
            "若",
            LessonKind::ConditionalRule,
            &["那么", "就", "则", "，", ","],
        ),
        (
            "因为",
            LessonKind::CausalRule,
            &["所以", "因此", "因而", "故而", "，", ","],
        ),
        (
            "由于",
            LessonKind::CausalRule,
            &["所以", "因此", "因而", "故而", "，", ","],
        ),
        (
            "既然",
            LessonKind::CausalRule,
            &["那么", "就", "所以", "，", ","],
        ),
    ];
    for (prefix, kind, markers) in PATTERNS {
        let Some(rest) = text.strip_prefix(prefix) else {
            continue;
        };
        let mut best: Option<(usize, &str)> = None;
        for &marker in markers {
            if let Some(position) = rest.find(marker) {
                if best.map(|(best_position, _)| position < best_position).unwrap_or(true) {
                    best = Some((position, marker));
                }
            }
        }
        let (position, marker) = best?;
        let mut antecedent = rest[..position].trim().to_string();
        if let Some(stripped) = antecedent.strip_suffix("的话") {
            antecedent = stripped.trim().to_string();
        }
        let mut consequent = rest[position + marker.len()..].trim().to_string();
        // 在逗号处切分后，后件常以连词开头（"，那么B" → B）；依次剥掉
        loop {
            let mut stripped = false;
            for lead in ["那么", "因此", "因而", "所以", "故而", "就", "则", "便"] {
                if let Some(rest) = consequent.strip_prefix(lead) {
                    consequent = rest.trim().to_string();
                    stripped = true;
                    break;
                }
            }
            if !stripped {
                break;
            }
        }
        if antecedent.is_empty() || consequent.is_empty() {
            return None;
        }
        return Some((kind, antecedent, consequent));
    }
    None
}

/// 解析单句中文知识
pub fn parse_lesson(text: &str) -> Result<ChineseLesson, String> {
    let raw = text.trim();
    if raw.contains('？') || raw.contains('?') {
        return Err(format!("「{}」是疑问句，不能作为知识教学", clean_sentence(text)));
    }
    let cleaned = clean_sentence(text);
    if cleaned.is_empty() {
        return Err("空句子".to_string());
    }
    if contains_question_word(&cleaned) {
        return Err(format!("「{}」是疑问句，不能作为知识教学", cleaned));
    }
    let (probability, without_quantifier) = match extract_probability(&cleaned) {
        Some((probability, cleaned)) => (Some(probability), cleaned),
        None => (None, cleaned.clone()),
    };
    let normalized = strip_probability_wrappers(&without_quantifier);
    // 条件/因果句
    if let Some((kind, antecedent, consequent)) =
        split_conditional(&normalized).or_else(|| split_conditional(&cleaned))
    {
        let antecedent = strip_probability_wrappers(&antecedent);
        let consequent = strip_probability_wrappers(&consequent);
        let Some(head) = parse_proposition(&consequent) else {
            return Err(format!("无法解析后件「{}」（支持：S的P是O / S是O / S会V）", consequent));
        };
        let Some(premise) = parse_proposition(&antecedent) else {
            return Err(format!("无法解析前件「{}」（支持：S的P是O / S是O / S会V）", antecedent));
        };
        let probability = probability.unwrap_or_else(|| default_rule_probability(kind));
        let source = match kind {
            LessonKind::ConditionalRule => "中文教学·条件句".to_string(),
            _ => "中文教学·因果句".to_string(),
        };
        return Ok(ChineseLesson {
            kind,
            head,
            body: vec![premise],
            probability,
            source,
            original: cleaned,
        });
    }
    // 比例/量词陈述
    if let Some(probability) = probability {
        let target = normalized.trim().trim_start_matches('的').trim();
        let Some(atom) = parse_proposition(target) else {
            return Err(format!(
                "无法解析「{}」（比例陈述支持：S的P是O / S会V）",
                target
            ));
        };
        return Ok(ChineseLesson {
            kind: LessonKind::ProbabilisticFact,
            head: atom,
            body: Vec::new(),
            probability,
            source: "中文教学·概率事实".to_string(),
            original: cleaned,
        });
    }
    // 直陈句 → 确定事实
    let Some(atom) = parse_proposition(&normalized) else {
        return Err(format!(
            "无法解析「{}」（支持：S的P是O / O是S的P / S是O / S会V / 谓词(参数)）",
            cleaned
        ));
    };
    Ok(ChineseLesson {
        kind: LessonKind::DeterministicFact,
        head: atom,
        body: Vec::new(),
        probability: 1.0,
        source: "中文教学·确定事实".to_string(),
        original: cleaned,
    })
}

/// 解析多句文本（按 。；; 换行 切分），收集成功与失败
pub fn parse_lessons(text: &str) -> LessonParseOutcome {
    let mut outcome = LessonParseOutcome::default();
    let segments: Vec<&str> = text
        .split(['。', '；', ';', '\n'])
        .map(|segment| segment.trim())
        .filter(|segment| !segment.is_empty())
        .collect();
    for segment in segments {
        match parse_lesson(segment) {
            Ok(lesson) => outcome.lessons.push(lesson),
            Err(reason) => outcome.errors.push(reason),
        }
    }
    outcome
}

/// 把解析出的知识写入 ProbLog 程序
pub fn apply_lesson(program: &mut ProblogEngine, lesson: &ChineseLesson) {
    match lesson.kind {
        LessonKind::DeterministicFact => {
            program.add_deterministic_fact(lesson.head.clone());
        }
        LessonKind::ProbabilisticFact => {
            program.add_probabilistic_fact(lesson.head.clone(), lesson.probability);
        }
        LessonKind::ConditionalRule | LessonKind::CausalRule => {
            program.add_rule(
                lesson.head.clone(),
                lesson.body.clone(),
                lesson.probability,
                &lesson.source,
            );
        }
    }
}

/// 知识的生成式回读（用于教学报告）
pub fn render_lesson_zh(lesson: &ChineseLesson) -> String {
    let probability = format!("{:.0}%", lesson.probability * 100.0);
    match lesson.kind {
        LessonKind::DeterministicFact => {
            format!("「{}」为确定事实（概率 100%）", render_atom_zh(&lesson.head))
        }
        LessonKind::ProbabilisticFact => format!(
            "「{}」的可能性为 {}",
            render_atom_zh(&lesson.head),
            probability
        ),
        LessonKind::ConditionalRule => {
            let premises = lesson
                .body
                .iter()
                .map(render_atom_zh)
                .collect::<Vec<_>>()
                .join(" 且 ");
            format!(
                "只要「{}」，就有 {} 的可能「{}」",
                premises,
                probability,
                render_atom_zh(&lesson.head)
            )
        }
        LessonKind::CausalRule => {
            let premises = lesson
                .body
                .iter()
                .map(render_atom_zh)
                .collect::<Vec<_>>()
                .join(" 且 ");
            format!(
                "因为「{}」，所以有 {} 的可能「{}」",
                premises,
                probability,
                render_atom_zh(&lesson.head)
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_chinese_number_basic() {
        assert_eq!(parse_chinese_number("八"), Some(8));
        assert_eq!(parse_chinese_number("十"), Some(10));
        assert_eq!(parse_chinese_number("十五"), Some(15));
        assert_eq!(parse_chinese_number("二十"), Some(20));
        assert_eq!(parse_chinese_number("八十五"), Some(85));
        assert_eq!(parse_chinese_number("九十九"), Some(99));
        assert_eq!(parse_chinese_number("一百"), Some(100));
        assert_eq!(parse_chinese_number("一百零八"), Some(108));
        assert_eq!(parse_chinese_number("80"), Some(80));
        assert_eq!(parse_chinese_number("汉"), None);
    }

    #[test]
    fn test_extract_probability_percent_and_words() {
        let (probability, cleaned) = extract_probability("百分之八十的鸟会飞").unwrap();
        assert!((probability - 0.8).abs() < 1e-9);
        assert_eq!(cleaned, "的鸟会飞");
        let (probability, _) = extract_probability("90%的学生通过考试").unwrap();
        assert!((probability - 0.9).abs() < 1e-9);
        let (probability, _) = extract_probability("八成会下雨").unwrap();
        assert!((probability - 0.8).abs() < 1e-9);
        let (probability, _) = extract_probability("鸟几乎总是会飞").unwrap();
        assert!((probability - 0.95).abs() < 1e-9);
        let (probability, _) = extract_probability("鸟很少会飞").unwrap();
        assert!((probability - 0.15).abs() < 1e-9);
        // 「几乎总是」应优先于「总是」
        let (probability, cleaned) = extract_probability("几乎总是下雨").unwrap();
        assert!((probability - 0.95).abs() < 1e-9);
        assert_eq!(cleaned, "下雨");
        assert!(extract_probability("不可能下雨").is_none());
    }

    #[test]
    fn test_extract_probability_percent_boundaries() {
        // 「180%」不是合法概率，不能从中间的「80%」截取，也不应留下残字
        assert!(extract_probability("180%的学生通过考试").is_none());
        assert!(extract_probability("百分之200的鸟会飞").is_none());
        // 「百分之 80」带空白仍可识别
        let (probability, cleaned) = extract_probability("百分之 80 的学生通过考试").unwrap();
        assert!((probability - 0.8).abs() < 1e-9);
        assert_eq!(cleaned, " 的学生通过考试");
    }

    #[test]
    fn test_parse_conditional_rule_default_probability() {
        let lesson = parse_lesson("如果张三的师傅是李四，那么张三的师祖是王五").unwrap();
        assert_eq!(lesson.kind, LessonKind::ConditionalRule);
        assert_eq!(lesson.head.display(), "师祖(张三, 王五)");
        assert_eq!(lesson.body[0].display(), "师傅(张三, 李四)");
        assert!((lesson.probability - 0.95).abs() < 1e-9);
    }

    #[test]
    fn test_parse_causal_rule_with_percentage() {
        let lesson = parse_lesson("因为张三的师傅是李四，所以八成张三的长辈是李四").unwrap();
        assert_eq!(lesson.kind, LessonKind::CausalRule);
        assert_eq!(lesson.head.display(), "长辈(张三, 李四)");
        assert!((lesson.probability - 0.8).abs() < 1e-9);
    }

    #[test]
    fn test_parse_probabilistic_fact_unary_and_binary() {
        let lesson = parse_lesson("八成的鸟会飞").unwrap();
        assert_eq!(lesson.kind, LessonKind::ProbabilisticFact);
        assert_eq!(lesson.head.display(), "会飞(鸟)");
        assert!((lesson.probability - 0.8).abs() < 1e-9);

        let lesson = parse_lesson("张三的师傅是李四的可能性是百分之八十").unwrap();
        assert_eq!(lesson.kind, LessonKind::ProbabilisticFact);
        assert_eq!(lesson.head.display(), "师傅(张三, 李四)");
        assert!((lesson.probability - 0.8).abs() < 1e-9);
    }

    #[test]
    fn test_parse_deterministic_fact_and_reversed_order() {
        let lesson = parse_lesson("李四是张三的师傅").unwrap();
        assert_eq!(lesson.kind, LessonKind::DeterministicFact);
        assert_eq!(lesson.head.display(), "师傅(张三, 李四)");
        assert!((lesson.probability - 1.0).abs() < 1e-9);
        let lesson = parse_lesson("张三是老师").unwrap();
        assert_eq!(lesson.head.display(), "是(张三, 老师)");
    }

    #[test]
    fn test_parse_lessons_splits_multiple_sentences() {
        let outcome = parse_lessons(
            "如果张三的师傅是李四，那么张三的师祖是王五；八成的鸟会飞。这不是一句知识？",
        );
        assert_eq!(outcome.lessons.len(), 2);
        assert_eq!(outcome.errors.len(), 1);
        assert!(outcome.errors[0].contains("疑问句"));
    }

    #[test]
    fn test_function_style_atom_syntax() {
        let lesson = parse_lesson("父亲(张三, 李四)").unwrap();
        assert_eq!(lesson.kind, LessonKind::DeterministicFact);
        assert_eq!(lesson.head.display(), "父亲(张三, 李四)");
    }

    #[test]
    fn test_render_atom_and_lesson_roundtrip() {
        let atom = Atom::rel("师傅", "张三", "李四");
        assert_eq!(render_atom_zh(&atom), "张三的师傅是李四");
        let atom = Atom::unary("会飞", Term::constant("鸟"));
        assert_eq!(render_atom_zh(&atom), "鸟会飞");
        let lesson = parse_lesson("如果张三的师傅是李四，那么张三的师祖是王五").unwrap();
        let rendered = render_lesson_zh(&lesson);
        assert!(rendered.contains("只要"));
        assert!(rendered.contains("95%"));
    }

    #[test]
    fn test_apply_lesson_into_problog_engine() {
        let mut program = ProblogEngine::new(6, 8);
        let lesson = parse_lesson("八成的鸟会飞").unwrap();
        apply_lesson(&mut program, &lesson);
        let lesson = parse_lesson("如果鸟会飞，那么鸟能上天").unwrap();
        apply_lesson(&mut program, &lesson);
        let query = program.query(&Atom::unary("能上天", Term::constant("鸟")), 3, 0.0);
        assert_eq!(query.answers.len(), 1);
        assert!((query.answers[0].probability - 0.8 * 0.95).abs() < 1e-9);
    }

    #[test]
    fn test_generalize_atom_turns_placeholders_into_variables() {
        let lesson = parse_lesson("如果X的师傅是Y，那么X的老师是Y").unwrap();
        assert_eq!(lesson.kind, LessonKind::ConditionalRule);
        assert_eq!(lesson.head.display(), "老师(X, Y)");
        assert_eq!(lesson.body[0].display(), "师傅(X, Y)");
        assert!(lesson_has_variables(&lesson));
        let clause = lesson_to_clause(&lesson).expect("条件句应可转成 Horn 子句");
        assert_eq!(clause.head.display(), "老师(?X, ?Y)");
        assert_eq!(clause.body[0].display(), "师傅(?X, ?Y)");
        assert!((clause.confidence - 0.95).abs() < 1e-9);
    }

    #[test]
    fn test_generalize_atom_supports_chinese_and_letter_placeholders() {
        for (sentence, expected) in [
            ("如果甲是乙的师傅，那么甲是乙的老师", "老师(?乙, ?甲)"),
            ("如果A的师傅是B，那么A的老师是B", "老师(?A, ?B)"),
        ] {
            let lesson = parse_lesson(sentence).unwrap();
            let clause = lesson_to_clause(&lesson).expect("应可变量化");
            assert_eq!(clause.head.display(), expected, "句子：{}", sentence);
        }
    }

    #[test]
    fn test_ground_rule_stays_ground_clause() {
        let lesson = parse_lesson("如果张三的师傅是李四，那么张三的师祖是王五").unwrap();
        assert!(!lesson_has_variables(&lesson));
        let clause = lesson_to_clause(&lesson).expect("条件句应可转成子句");
        assert!(clause.head.is_ground());
        assert_eq!(clause.head.display(), "师祖(张三, 王五)");
    }

    #[test]
    fn test_non_rule_lessons_have_no_clause() {
        let fact = parse_lesson("张三是老师").unwrap();
        assert!(lesson_to_clause(&fact).is_none());
        let probabilistic = parse_lesson("八成的鸟会飞").unwrap();
        assert!(lesson_to_clause(&probabilistic).is_none());
    }

    #[test]
    fn test_unsupported_sentence_reports_reason() {
        let outcome = parse_lessons("除非下雨，否则比赛照常");
        assert!(outcome.lessons.is_empty());
        assert!(!outcome.errors.is_empty());
    }
}

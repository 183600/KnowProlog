// 句法驱动的查询重写（Query Rewrite）
//
// 目标不是记住句子，而是把同一关系问题的不同句法外壳压到同一个逻辑计划：
//
//   张三认识谁          ┐
//   谁被张三认识        ├→ FindObjectByPredicate { subject: 张三, predicate: 认识 }
//   谁被张三认识呢      ┘
//
//   谁认识张三          ┐
//   张三被谁认识        ├→ FindSubject { object: 张三, predicate: 认识 }
//   张三被谁认识过      ┘
//
//   张三认识李四吗      ┐
//   李四被张三认识吗    ├→ CheckRelation { subject: 张三, predicate: 认识, object: 李四 }
//   张三把李四认识了吗  ┘
//
// 解析只做保守的、可解释的结构切分：被/把标记 + wh 位置 + 词表边界搜索。
// 不依赖固定词性表；每个候选边界都必须同时通过 `RelationLexicon` 的实体/
// 谓词登记校验，多个候选按「主语最长、谓词最长」稳定排序。校验不通过就
// 弃权（返回 None），交回现有管线，绝不硬猜。
// 每次重写都带规则 id 与等价说明，可打印、可审计。

use crate::query_planner::{QueryPlan, QueryPlanner};

/// 关系词表视图：调用方用本体/谓词画像做最终校验。
pub trait RelationLexicon {
    /// 归一化实体名；未登记返回 None。
    fn canonical_entity(&self, raw: &str) -> Option<String>;
    /// 归一化谓词词面；未登记返回 None。
    fn canonical_predicate(&self, raw: &str) -> Option<String>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelationShape {
    ActiveWhSubject,
    ActiveWhObject,
    PassiveWhAgent,
    PassiveWhPatient,
    BaWhSubject,
    BaWhObject,
    ActiveYesNo,
    PassiveYesNo,
    BaYesNo,
    CountRelation,
}

impl RelationShape {
    pub fn label(self) -> &'static str {
        match self {
            RelationShape::ActiveWhSubject => "主动句·问主语",
            RelationShape::ActiveWhObject => "主动句·问宾语",
            RelationShape::PassiveWhAgent => "被字句·问施事",
            RelationShape::PassiveWhPatient => "被字句·问受事",
            RelationShape::BaWhSubject => "把字句·问主语",
            RelationShape::BaWhObject => "把字句·问受事",
            RelationShape::ActiveYesNo => "主动句·是非问",
            RelationShape::PassiveYesNo => "被字句·是非问",
            RelationShape::BaYesNo => "把字句·是非问",
            RelationShape::CountRelation => "关系·计数",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationRewrite {
    pub plan: QueryPlan,
    pub shape: RelationShape,
    /// 稳定的规则 id，用于回执与回归测试。
    pub rule: &'static str,
    /// 原句中的谓词词面（可能带体标记，如「教过」）。
    pub predicate_surface: String,
    /// 词表归一后的谓词。
    pub predicate: String,
}

impl RelationRewrite {
    /// 可打印的等价重写回执：规则、词面、逻辑计划。
    pub fn receipt(&self) -> String {
        format!(
            "{}({})（{}）：{} → {}",
            self.rule,
            QueryPlanner::kind(&self.plan),
            self.shape.label(),
            self.predicate_surface,
            QueryPlanner::format_plan(&self.plan)
        )
    }
}

const WH_PREFIXES: [&str; 8] = ["什么人", "哪些人", "哪位", "哪个", "谁", "何人", "什么", "啥"];
const WH_SUFFIXES: [&str; 9] = [
    "什么人", "哪些人", "哪位", "哪个", "谁", "何人", "什么", "啥", "哪些",
];
/// 疑问词后还可以带类型名词：「哪些学生」「哪位老师」「什么书」。
/// 这类名词只约束答案类型，不参与谓词词表校验。
const WH_NP_HEADS: [&str; 8] = ["哪些", "哪个", "哪位", "哪家", "哪门", "哪几", "什么", "啥"];
/// 关系从句后的问句尾巴：「张三教过的学生有哪些」。
/// 按长度从长到短排列，避免「是哪些人」被「是谁」抢先剥离。
const RELATIVE_WH_TAILS: [&str; 16] = [
    "分别是哪些人",
    "分别是哪些",
    "分别是谁",
    "都有哪些",
    "都有谁",
    "有哪些人",
    "有哪些",
    "是哪些人",
    "是哪些",
    "是哪个",
    "是哪位",
    "是什么",
    "是谁",
    "有谁",
    "哪些人",
    "哪些",
];
const POLITE_PREFIXES: [&str; 11] = [
    "请问一下",
    "请问",
    "请告诉我一下",
    "请告诉我",
    "我想知道",
    "我想问一下",
    "帮我查一下",
    "帮我查查",
    "麻烦问一下",
    "麻烦你",
    "告诉我",
];
const ENTITY_PREFIXES: [&str; 8] = ["的", "了", "着", "过", "于", "给", "向", "往"];
const ENTITY_SUFFIXES: [&str; 8] = ["的", "了", "着", "过", "吗", "呢", "吧", "啊"];
const PREDICATE_PREFIXES: [&str; 12] = [
    "曾经", "已经", "一直", "正在", "也", "都", "还", "就", "将", "会", "能", "可以",
];
const PREDICATE_SUFFIXES: [&str; 6] = ["起来", "下去", "了", "过", "着", "吗"];
const NEGATION_PREFIXES: [&str; 3] = ["没有", "没", "不"];

fn clean_question(question: &str) -> Option<String> {
    let mut text = question.trim().to_string();
    for prefix in POLITE_PREFIXES {
        if let Some(rest) = text.strip_prefix(prefix) {
            text = rest.trim().to_string();
            break;
        }
    }
    let text = text
        .trim_end_matches(|c: char| {
            matches!(
                c,
                '？' | '?' | '。' | '！' | '!' | '，' | ',' | '；' | ';' | '~' | ' '
            )
        })
        .trim();
    if text.is_empty() {
        None
    } else {
        Some(text.to_string())
    }
}

fn strip_yes_no(text: &str) -> (String, bool) {
    let trimmed = text.trim_end();
    for suffix in ["吗", "嘛", "没有"] {
        if let Some(rest) = trimmed.strip_suffix(suffix) {
            return (rest.trim_end().to_string(), true);
        }
    }
    if let Some(rest) = trimmed.strip_suffix('么') {
        // 「什么/怎么/那么/这么/多么/为什么」里的「么」不是是非问语气词。
        let blocked = ["什", "怎", "那", "这", "多", "为"];
        if !blocked.iter().any(|prefix| rest.ends_with(prefix)) {
            return (rest.trim_end().to_string(), true);
        }
    }
    (trimmed.to_string(), false)
}

fn strip_trailing_particles(text: &str) -> String {
    text.trim_end_matches(['呢', '吧', '啊', '呀', ' '])
        .trim()
        .to_string()
}

fn is_wh(text: &str) -> bool {
    WH_SUFFIXES.iter().any(|word| text.contains(word))
}

/// 类型名词的保守边界：短、无功能词、无体标记。
/// 解析器只借它界定「哪些学生」的范围，不要求本体登记。
fn valid_type_noun(text: &str) -> bool {
    let chars = text.chars().count();
    (1..=10).contains(&chars)
        && !text.chars().any(|c| {
            matches!(
                c,
                '的' | '是' | '被' | '把' | '将' | '吗' | '呢' | '吧' | '啊' | '过' | '了'
                    | '着' | '不' | '没' | '和' | '跟' | '与' | '还' | '也' | '都' | '，'
                    | ',' | '？' | '?' | '。' | '！' | '!' | ' '
            )
        })
        && !is_wh(text)
}

/// 关系从句：「V 过/了 的 N」「被 V 的 N」「V N 的 N」。
/// 返回（从句核心，类型名词），无「的」或名词不合法时弃权。
fn split_relative_clause(text: &str) -> Option<(&str, &str)> {
    let (head, noun) = text.rsplit_once('的')?;
    let head = head.trim();
    let noun = noun.trim();
    if head.is_empty() || !valid_type_noun(noun) {
        return None;
    }
    Some((head, noun))
}

/// 「哪些学生」「哪位老师」这类疑问名词短语必须整体消费到句尾；
/// 返回（左侧关系部分，右侧疑问名词短语）。
fn split_wh_np_suffix(text: &str) -> Option<(&str, &str)> {
    let mut best: Option<(usize, &str)> = None;
    for head in WH_NP_HEADS {
        let Some(position) = text.rfind(head) else {
            continue;
        };
        if position == 0 || best.is_some_and(|(current, _)| current >= position) {
            continue;
        }
        let noun = &text[position + head.len()..];
        if valid_type_noun(noun) {
            best = Some((position, head));
        }
    }
    let (position, _) = best?;
    Some((&text[..position], &text[position..]))
}

/// 疑问名词短语在句首时，枚举「疑问词 + 类型名词 + 谓词…」的切法。
/// 只返回通过词表校验的最长名词候选，保证「哪些老师教过张三」不把
/// 「老师教过」整体误当名词。
fn split_wh_np_prefix_candidates<'a>(text: &'a str) -> Vec<(&'a str, &'a str)> {
    let mut candidates: Vec<(&str, &str)> = Vec::new();
    for head in WH_NP_HEADS {
        let Some(rest) = text.strip_prefix(head) else {
            continue;
        };
        for (index, _) in rest.char_indices().skip(1) {
            let noun = &rest[..index];
            let tail = &rest[index..];
            if valid_type_noun(noun) {
                candidates.push((noun, tail));
            }
        }
    }
    candidates.sort_by(|left, right| right.0.chars().count().cmp(&left.0.chars().count()));
    candidates
}

fn split_wh_np_prefix_object<'a>(
    text: &'a str,
    lexicon: &dyn RelationLexicon,
) -> Option<&'a str> {
    split_wh_np_prefix_candidates(text)
        .into_iter()
        .find_map(|(_, tail)| split_predicate_object(tail, lexicon).map(|_| tail))
}

fn split_wh_np_prefix_predicate<'a>(
    text: &'a str,
    lexicon: &dyn RelationLexicon,
) -> Option<&'a str> {
    split_wh_np_prefix_candidates(text)
        .into_iter()
        .find_map(|(_, tail)| predicate(tail, lexicon).map(|_| tail))
}

/// 关系计数：「张三教过多少个学生」「张三教过几个」。
/// 量词后的类型名词只用于说明问的是一类对象，计数仍按关系命中执行；
/// 谓词必须是关系词表登记过的实义关系，避免把「有几个师傅」误当计数。
fn parse_relation_count(
    text: &str,
    lexicon: &dyn RelationLexicon,
) -> Option<RelationRewrite> {
    const MEASURES: [&str; 14] = [
        "个", "位", "名", "本", "部", "家", "门", "只", "种", "条", "张", "所", "件", "次",
    ];
    let mut candidates: Vec<(usize, &str)> = Vec::new();
    for marker in ["多少", "几"] {
        let mut offset = 0;
        while let Some(position) = text[offset..].find(marker) {
            let position = offset + position;
            candidates.push((position, marker));
            offset = position + marker.len();
        }
    }
    candidates.sort_by_key(|(position, _)| *position);
    for (position, marker) in candidates {
        let head = text[..position].trim();
        let mut after = text[position + marker.len()..].trim();
        for measure in MEASURES {
            if let Some(rest) = after.strip_prefix(measure) {
                after = rest.trim();
                break;
            }
        }
        if !after.is_empty() && !valid_type_noun(after) {
            continue;
        }
        let Some((subject, surface, predicate, negated)) =
            split_entity_predicate(head, lexicon)
        else {
            continue;
        };
        if negated || matches!(predicate.as_str(), "有" | "是") {
            continue;
        }
        return Some(build(
            QueryPlan::Count {
                subject,
                predicate: predicate.clone(),
            },
            RelationShape::CountRelation,
            "relation-count",
            surface,
            predicate,
        ));
    }
    None
}

/// 「张三教过的学生是谁」「教过张三的老师有哪些」：
/// 关系从句先压成关系查找，再由现成的关系回答器执行。
fn parse_relative_wh(
    text: &str,
    lexicon: &dyn RelationLexicon,
) -> Option<RelationRewrite> {
    let core = RELATIVE_WH_TAILS
        .iter()
        .find_map(|tail| text.strip_suffix(tail))?;
    let (head, _noun) = split_relative_clause(core)?;

    if let Some(inner) = head.strip_prefix('被') {
        let (agent, surface, predicate, negated) = split_entity_predicate(inner, lexicon)?;
        if negated {
            return None;
        }
        return Some(build(
            QueryPlan::FindObjectByPredicate {
                subject: agent,
                predicate: predicate.clone(),
            },
            RelationShape::PassiveWhPatient,
            "typed-relative-passive-patient",
            surface,
            predicate,
        ));
    }

    if let Some((subject, surface, predicate, negated)) =
        split_entity_predicate(head, lexicon)
    {
        if !negated {
            return Some(build(
                QueryPlan::FindObjectByPredicate {
                    subject,
                    predicate: predicate.clone(),
                },
                RelationShape::ActiveWhObject,
                "typed-relative-object",
                surface,
                predicate,
            ));
        }
    }

    let (surface, predicate, object, negated) = split_predicate_object(head, lexicon)?;
    if negated {
        return None;
    }
    Some(build(
        QueryPlan::FindSubject {
            object,
            predicate: predicate.clone(),
        },
        RelationShape::ActiveWhSubject,
        "typed-relative-subject",
        surface,
        predicate,
    ))
}

/// A 不 A / A 不 AB 型正反问：「认不认识」「教不教」「喜不喜欢」。
/// 这类问句本身不含否定命题，只需要识别为是非问。
fn has_a_not_a(text: &str) -> bool {
    let mut offset = 0;
    while let Some(position) = text[offset..].find('不') {
        let position = offset + position;
        let left: Vec<char> = text[..position].chars().collect();
        let right = &text[position + '不'.len_utf8()..];
        // 只看紧邻的 1~2 个字：A 不 A / A 不 AB 都会让右侧复用左侧词头。
        for len in [2, 1] {
            if left.len() >= len {
                let word: String = left[left.len() - len..].iter().collect();
                if right.starts_with(&word) {
                    return true;
                }
            }
        }
        offset = position + '不'.len_utf8();
    }
    false
}

fn strip_entity_affixes(mut text: &str) -> &str {
    loop {
        if let Some(rest) = ENTITY_PREFIXES
            .iter()
            .find_map(|prefix| text.strip_prefix(prefix))
        {
            text = rest;
            continue;
        }
        if let Some(rest) = ENTITY_SUFFIXES
            .iter()
            .find_map(|suffix| text.strip_suffix(suffix))
        {
            text = rest;
            continue;
        }
        break;
    }
    text
}

fn strip_negation_suffix(text: &str) -> (&str, bool) {
    if let Some(rest) = text.strip_suffix("没有") {
        return (rest.trim_end(), true);
    }
    if let Some(rest) = text.strip_suffix('没') {
        return (rest.trim_end(), true);
    }
    (text, false)
}

fn entity(
    raw: &str,
    lexicon: &dyn RelationLexicon,
) -> Option<String> {
    let raw = raw.trim();
    if raw.is_empty() || is_wh(raw) {
        return None;
    }
    if let Some(canonical) = lexicon.canonical_entity(raw) {
        return Some(canonical);
    }
    let stripped = strip_entity_affixes(raw);
    if stripped != raw && !stripped.is_empty() && !is_wh(stripped) {
        return lexicon.canonical_entity(stripped);
    }
    None
}

fn predicate(
    raw: &str,
    lexicon: &dyn RelationLexicon,
) -> Option<(String, String, bool)> {
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }
    let mut candidates: Vec<(String, bool)> = Vec::new();
    let mut current = raw.to_string();
    candidates.push((current.clone(), false));
    if let Some(position) = current.find('不') {
        let left = &current[..position];
        let right = &current[position + '不'.len_utf8()..];
        if !left.is_empty() && right.starts_with(left) {
            candidates.push((right.to_string(), false));
        }
    }
    for prefix in PREDICATE_PREFIXES {
        if let Some(rest) = current.strip_prefix(prefix) {
            current = rest.trim().to_string();
            candidates.push((current.clone(), false));
        }
    }
    for prefix in NEGATION_PREFIXES {
        if let Some(rest) = current.strip_prefix(prefix) {
            current = rest.trim().to_string();
            candidates.push((current.clone(), true));
            break;
        }
    }
    let mut trimmed = current.clone();
    let negated = candidates.last().is_some_and(|(_, negated)| *negated);
    loop {
        let mut changed = false;
        for suffix in PREDICATE_SUFFIXES {
            if let Some(rest) = trimmed.strip_suffix(suffix) {
                if !rest.is_empty() {
                    trimmed = rest.to_string();
                    candidates.push((trimmed.clone(), negated));
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    for (candidate, negated) in candidates {
        if candidate.is_empty() {
            continue;
        }
        if let Some(canonical) = lexicon.canonical_predicate(&candidate) {
            return Some((candidate, canonical, negated));
        }
    }
    None
}

fn wh_prefix(text: &str) -> Option<usize> {
    WH_PREFIXES
        .iter()
        .find(|word| text.starts_with(*word))
        .map(|word| word.len())
}

fn wh_suffix(text: &str) -> Option<usize> {
    WH_SUFFIXES
        .iter()
        .find(|word| text.ends_with(*word))
        .map(|word| text.len() - word.len())
}

/// 把「谓词 + 宾语」切成两段：谓词在左、宾语在右。
/// 谓词按长度从长到短尝试，优先保留完整词形（如「毕业于」）。
fn split_predicate_object<'a>(
    text: &'a str,
    lexicon: &dyn RelationLexicon,
) -> Option<(String, String, String, bool)> {
    let mut candidates: Vec<(String, String, String, bool)> = Vec::new();
    for (index, _) in text.char_indices().skip(1) {
        let left = &text[..index];
        let right = &text[index..];
        let Some((surface, predicate, negated)) = predicate(left, lexicon) else {
            continue;
        };
        let Some(object) = entity(right, lexicon) else {
            continue;
        };
        candidates.push((surface, predicate, object, negated));
    }
    candidates.sort_by(|left, right| {
        right
            .0
            .chars()
            .count()
            .cmp(&left.0.chars().count())
            .then(left.1.cmp(&right.1))
            .then(left.2.cmp(&right.2))
    });
    candidates.into_iter().next()
}

/// 把「实体 + 谓词」切成两段：实体在左、谓词在右。
fn split_entity_predicate<'a>(
    text: &'a str,
    lexicon: &dyn RelationLexicon,
) -> Option<(String, String, String, bool)> {
    let mut candidates: Vec<(String, String, String, bool)> = Vec::new();
    for (index, _) in text.char_indices().skip(1) {
        let left = &text[..index];
        let right = &text[index..];
        let Some(entity) = entity(left, lexicon) else {
            continue;
        };
        let Some((surface, predicate, negated)) = predicate(right, lexicon) else {
            continue;
        };
        candidates.push((entity, surface, predicate, negated));
    }
    candidates.sort_by(|left, right| {
        right
            .0
            .chars()
            .count()
            .cmp(&left.0.chars().count())
            .then(right.1.chars().count().cmp(&left.1.chars().count()))
    });
    candidates.into_iter().next()
}

/// 把「主语 + 谓词 + 宾语」切成三段（是非问）。
fn split_svo<'a>(
    text: &'a str,
    lexicon: &dyn RelationLexicon,
) -> Option<(String, String, String, String, bool)> {
    let mut candidates: Vec<(String, String, String, String, bool)> = Vec::new();
    let boundaries: Vec<usize> = text
        .char_indices()
        .map(|(index, _)| index)
        .chain(std::iter::once(text.len()))
        .collect();
    for &subject_end in boundaries.iter().skip(1) {
        let Some(subject) = entity(&text[..subject_end], lexicon) else {
            continue;
        };
        for &object_start in boundaries.iter().filter(|&&index| index > subject_end) {
            let predicate_text = &text[subject_end..object_start];
            let Some((surface, predicate, negated)) = predicate(predicate_text, lexicon) else {
                continue;
            };
            let Some(object) = entity(&text[object_start..], lexicon) else {
                continue;
            };
            candidates.push((subject.clone(), surface, predicate, object, negated));
        }
    }
    candidates.sort_by(|left, right| {
        right
            .0
            .chars()
            .count()
            .cmp(&left.0.chars().count())
            .then(right.1.chars().count().cmp(&left.1.chars().count()))
            .then(right.3.chars().count().cmp(&left.3.chars().count()))
    });
    candidates.into_iter().next()
}

fn marker_position(text: &str, markers: &[&str]) -> Option<(usize, usize)> {
    for marker in markers {
        if let Some(position) = text.find(marker) {
            return Some((position, marker.len()));
        }
    }
    None
}

fn build(
    plan: QueryPlan,
    shape: RelationShape,
    rule: &'static str,
    predicate_surface: String,
    predicate: String,
) -> RelationRewrite {
    RelationRewrite {
        plan,
        shape,
        rule,
        predicate_surface,
        predicate,
    }
}

pub fn rewrite(
    question: &str,
    lexicon: &dyn RelationLexicon,
) -> Option<RelationRewrite> {
    let text = clean_question(question)?;
    if text.chars().count() > 80 {
        return None;
    }
    let text = strip_trailing_particles(&text);
    let (mut text, mut yes_no) = strip_yes_no(&text);
    if text.chars().count() < 2 {
        return None;
    }
    yes_no |= has_a_not_a(&text);
    for marker in ["有没有", "是否"] {
        if let Some(position) = text.find(marker) {
            text = format!("{}{}", &text[..position], &text[position + marker.len()..]);
            yes_no = true;
            break;
        }
    }
    text = text.trim().to_string();

    if let Some(rewrite) = parse_relative_wh(&text, lexicon) {
        return Some(rewrite);
    }

    if let Some(rewrite) = parse_relation_count(&text, lexicon) {
        return Some(rewrite);
    }

    for suffix in ["的是哪些人", "的是哪位", "的是哪个", "的是谁"] {
        if let Some(rest) = text.strip_suffix(suffix) {
            if let Some(inner) = rest.strip_prefix('被') {
                let (agent, surface, predicate, negated) = split_entity_predicate(inner, lexicon)?;
                if negated {
                    return None;
                }
                return Some(build(
                    QueryPlan::FindObjectByPredicate {
                        subject: agent,
                        predicate: predicate.clone(),
                    },
                    RelationShape::PassiveWhPatient,
                    "passive-relative-patient",
                    surface,
                    predicate,
                ));
            }
            let (surface, predicate, object, negated) = split_predicate_object(rest, lexicon)?;
            if negated {
                return None;
            }
            return Some(build(
                QueryPlan::FindSubject {
                    object,
                    predicate: predicate.clone(),
                },
                RelationShape::ActiveWhSubject,
                "relative-agent",
                surface,
                predicate,
            ));
        }
    }

    if let Some((position, length)) = marker_position(&text, &["被"]) {
        let (patient_text, marker_negated) = strip_negation_suffix(text[..position].trim());
        let right = text[position + length..].trim();
        if patient_text.is_empty() || right.is_empty() {
            return None;
        }
        if is_wh(patient_text) {
            if marker_negated {
                return None;
            }
            let (agent, surface, predicate, negated) = split_entity_predicate(right, lexicon)?;
            if negated {
                return None;
            }
            return Some(build(
                QueryPlan::FindObjectByPredicate {
                    subject: agent,
                    predicate: predicate.clone(),
                },
                RelationShape::PassiveWhPatient,
                "passive-wh-patient",
                surface,
                predicate,
            ));
        }
        let patient = entity(patient_text, lexicon)?;
        if let Some(rest) = split_wh_np_prefix_predicate(right, lexicon)
            .or_else(|| right.strip_prefix_wh())
        {
            if marker_negated {
                return None;
            }
            let (surface, predicate, negated) = predicate(rest, lexicon)?;
            if negated {
                return None;
            }
            return Some(build(
                QueryPlan::FindSubject {
                    object: patient,
                    predicate: predicate.clone(),
                },
                RelationShape::PassiveWhAgent,
                "passive-wh-agent",
                surface,
                predicate,
            ));
        }
        if !yes_no {
            return None;
        }
        let (agent, surface, predicate, predicate_negated) =
            split_entity_predicate(right, lexicon)?;
        return Some(build(
            QueryPlan::CheckRelation {
                subject: agent,
                predicate: predicate.clone(),
                object: patient,
                negated: predicate_negated ^ marker_negated,
            },
            RelationShape::PassiveYesNo,
            "passive-yes-no",
            surface,
            predicate,
        ));
    }

    if let Some((position, length)) = marker_position(&text, &["把", "将"]) {
        let (subject_text, marker_negated) = strip_negation_suffix(text[..position].trim());
        let right = text[position + length..].trim();
        if subject_text.is_empty() || right.is_empty() {
            return None;
        }
        if is_wh(subject_text) {
            if marker_negated {
                return None;
            }
            let (patient, surface, predicate, negated) = split_entity_predicate(right, lexicon)?;
            if negated {
                return None;
            }
            return Some(build(
                QueryPlan::FindSubject {
                    object: patient,
                    predicate: predicate.clone(),
                },
                RelationShape::BaWhSubject,
                "ba-wh-subject",
                surface,
                predicate,
            ));
        }
        let subject = entity(subject_text, lexicon)?;
        if let Some(rest) = split_wh_np_prefix_predicate(right, lexicon)
            .or_else(|| right.strip_prefix_wh())
        {
            if marker_negated {
                return None;
            }
            let (surface, predicate, negated) = predicate(rest, lexicon)?;
            if negated {
                return None;
            }
            return Some(build(
                QueryPlan::FindObjectByPredicate {
                    subject,
                    predicate: predicate.clone(),
                },
                RelationShape::BaWhObject,
                "ba-wh-object",
                surface,
                predicate,
            ));
        }
        if !yes_no {
            return None;
        }
        let (patient, surface, predicate, predicate_negated) =
            split_entity_predicate(right, lexicon)?;
        return Some(build(
            QueryPlan::CheckRelation {
                subject,
                predicate: predicate.clone(),
                object: patient,
                negated: predicate_negated ^ marker_negated,
            },
            RelationShape::BaYesNo,
            "ba-yes-no",
            surface,
            predicate,
        ));
    }

    if let Some(rest) = split_wh_np_prefix_object(&text, lexicon) {
        let (surface, predicate, object, negated) = split_predicate_object(rest, lexicon)?;
        if negated {
            return None;
        }
        return Some(build(
            QueryPlan::FindSubject {
                object,
                predicate: predicate.clone(),
            },
            RelationShape::ActiveWhSubject,
            "active-wh-np-subject",
            surface,
            predicate,
        ));
    }

    if let Some(prefix) = wh_prefix(&text) {
        let rest = &text[prefix..];
        let (surface, predicate, object, negated) = split_predicate_object(rest, lexicon)?;
        if negated {
            return None;
        }
        return Some(build(
            QueryPlan::FindSubject {
                object,
                predicate: predicate.clone(),
            },
            RelationShape::ActiveWhSubject,
            "active-wh-subject",
            surface,
            predicate,
        ));
    }

    if let Some((head, _np)) = split_wh_np_suffix(&text) {
        if let Some((subject, surface, predicate, negated)) =
            split_entity_predicate(head, lexicon)
        {
            if !negated {
                return Some(build(
                    QueryPlan::FindObjectByPredicate {
                        subject,
                        predicate: predicate.clone(),
                    },
                    RelationShape::ActiveWhObject,
                    "active-wh-np-object",
                    surface,
                    predicate,
                ));
            }
        }
    }

    if let Some(suffix) = wh_suffix(&text) {
        let rest = &text[..suffix];
        let Some((subject, surface, predicate, negated)) = split_entity_predicate(rest, lexicon)
        else {
            return None;
        };
        if negated {
            return None;
        }
        return Some(build(
            QueryPlan::FindObjectByPredicate {
                subject,
                predicate: predicate.clone(),
            },
            RelationShape::ActiveWhObject,
            "active-wh-object",
            surface,
            predicate,
        ));
    }

    if !yes_no {
        return None;
    }
    let (subject, surface, predicate, object, negated) = split_svo(&text, lexicon)?;
    Some(build(
        QueryPlan::CheckRelation {
            subject,
            predicate: predicate.clone(),
            object,
            negated,
        },
        RelationShape::ActiveYesNo,
        "active-yes-no",
        surface,
        predicate,
    ))
}

trait StripWhPrefix {
    fn strip_prefix_wh(&self) -> Option<&str>;
}

impl StripWhPrefix for str {
    fn strip_prefix_wh(&self) -> Option<&str> {
        WH_PREFIXES
            .iter()
            .find_map(|word| self.strip_prefix(*word))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FakeLexicon;

    impl RelationLexicon for FakeLexicon {
        fn canonical_entity(&self, raw: &str) -> Option<String> {
            match raw {
                "张三" | "李四" | "王五" | "赵六" => Some(raw.to_string()),
                _ => None,
            }
        }

        fn canonical_predicate(&self, raw: &str) -> Option<String> {
            match raw {
                "认识" | "教" | "教过" => Some(if raw == "教过" {
                    "教".to_string()
                } else {
                    raw.to_string()
                }),
                _ => None,
            }
        }
    }

    fn parse(question: &str) -> Option<RelationRewrite> {
        rewrite(question, &FakeLexicon)
    }

    #[test]
    fn active_wh_object() {
        let rewrite = parse("张三认识谁").expect("主动问宾语应可重写");
        assert_eq!(rewrite.rule, "active-wh-object");
        assert_eq!(rewrite.predicate, "认识");
        assert_eq!(
            rewrite.plan,
            QueryPlan::FindObjectByPredicate {
                subject: "张三".to_string(),
                predicate: "认识".to_string()
            }
        );
    }

    #[test]
    fn active_wh_subject() {
        let rewrite = parse("谁教过张三").expect("主动问主语应可重写");
        assert_eq!(rewrite.rule, "active-wh-subject");
        assert_eq!(rewrite.predicate_surface, "教过");
        assert_eq!(
            rewrite.plan,
            QueryPlan::FindSubject {
                object: "张三".to_string(),
                predicate: "教".to_string()
            }
        );
    }

    #[test]
    fn passive_both_directions() {
        let patient = parse("谁被张三教过").expect("被字句问受事");
        assert_eq!(patient.rule, "passive-wh-patient");
        assert_eq!(
            patient.plan,
            QueryPlan::FindObjectByPredicate {
                subject: "张三".to_string(),
                predicate: "教".to_string()
            }
        );

        let agent = parse("张三被谁教过").expect("被字句问施事");
        assert_eq!(agent.rule, "passive-wh-agent");
        assert_eq!(
            agent.plan,
            QueryPlan::FindSubject {
                object: "张三".to_string(),
                predicate: "教".to_string()
            }
        );
    }

    #[test]
    fn ba_both_directions() {
        let object = parse("张三把谁教过").expect("把字句问受事");
        assert_eq!(object.rule, "ba-wh-object");
        assert_eq!(
            object.plan,
            QueryPlan::FindObjectByPredicate {
                subject: "张三".to_string(),
                predicate: "教".to_string()
            }
        );

        let subject = parse("谁把张三教过").expect("把字句问主语");
        assert_eq!(subject.rule, "ba-wh-subject");
        assert_eq!(
            subject.plan,
            QueryPlan::FindSubject {
                object: "张三".to_string(),
                predicate: "教".to_string()
            }
        );
    }

    #[test]
    fn yes_no_forms_share_one_plan() {
        let expected = QueryPlan::CheckRelation {
            subject: "张三".to_string(),
            predicate: "教".to_string(),
            object: "李四".to_string(),
            negated: false,
        };
        assert_eq!(parse("张三教过李四吗").unwrap().plan, expected);
        assert_eq!(parse("李四被张三教过吗").unwrap().plan, expected);
        assert_eq!(parse("张三把李四教过吗").unwrap().plan, expected);
    }

    #[test]
    fn marker_negation_before_bei_or_ba_is_preserved() {
        let expected = QueryPlan::CheckRelation {
            subject: "张三".to_string(),
            predicate: "教".to_string(),
            object: "李四".to_string(),
            negated: true,
        };
        assert_eq!(parse("李四没有被张三教过吗").unwrap().plan, expected);
        assert_eq!(parse("张三没有把李四教过吗").unwrap().plan, expected);
        assert!(parse("谁没有被张三教过").is_none());
    }

    #[test]
    fn trailing_meiyou_is_yes_no() {
        let rewrite = parse("张三认识李四没有").expect("句尾「没有」应识别为是非问");
        assert_eq!(
            rewrite.plan,
            QueryPlan::CheckRelation {
                subject: "张三".to_string(),
                predicate: "认识".to_string(),
                object: "李四".to_string(),
                negated: false
            }
        );
    }

    #[test]
    fn a_not_a_questions_are_yes_no_without_negation() {
        let rewrite = parse("张三认不认识李四").expect("A 不 A 应识别为是非问");
        assert_eq!(
            rewrite.plan,
            QueryPlan::CheckRelation {
                subject: "张三".to_string(),
                predicate: "认识".to_string(),
                object: "李四".to_string(),
                negated: false
            }
        );
        assert_eq!(parse("张三教不教李四").expect("单字 A 不 A").predicate, "教");
    }

    #[test]
    fn negated_yes_no_keeps_polarity() {
        let rewrite = parse("张三不认识李四吗").expect("否定是非问应可重写");
        assert_eq!(
            rewrite.plan,
            QueryPlan::CheckRelation {
                subject: "张三".to_string(),
                predicate: "认识".to_string(),
                object: "李四".to_string(),
                negated: true
            }
        );
        let rewrite = parse("张三没有教过李四吗").expect("没有否定是非问应可重写");
        assert_eq!(rewrite.predicate, "教");
        assert!(matches!(
            rewrite.plan,
            QueryPlan::CheckRelation { negated: true, .. }
        ));
    }

    #[test]
    fn negated_wh_questions_abstain() {
        assert!(parse("张三不认识谁").is_none());
        assert!(parse("谁不认识张三").is_none());
    }

    #[test]
    fn passive_relative_clause() {
        let rewrite = parse("被张三认识的是谁").expect("被字关系从句应可重写");
        assert_eq!(rewrite.rule, "passive-relative-patient");
        assert_eq!(rewrite.predicate, "认识");
        assert_eq!(
            rewrite.plan,
            QueryPlan::FindObjectByPredicate {
                subject: "张三".to_string(),
                predicate: "认识".to_string()
            }
        );
    }

    #[test]
    fn existence_marker_and_relative_clause() {
        let existence = parse("张三有没有教过李四").expect("有没有是非问");
        assert_eq!(
            existence.plan,
            QueryPlan::CheckRelation {
                subject: "张三".to_string(),
                predicate: "教".to_string(),
                object: "李四".to_string(),
                negated: false
            }
        );

        let relative = parse("教过张三的是谁").expect("关系从句问施事");
        assert_eq!(relative.rule, "relative-agent");
        assert_eq!(
            relative.plan,
            QueryPlan::FindSubject {
                object: "张三".to_string(),
                predicate: "教".to_string()
            }
        );
    }

    #[test]
    fn quantified_noun_phrases_compile_like_bare_wh() {
        let object = parse("张三教过哪些学生").expect("问宾语的名词短语");
        assert_eq!(object.rule, "active-wh-np-object");
        assert_eq!(
            object.plan,
            QueryPlan::FindObjectByPredicate {
                subject: "张三".to_string(),
                predicate: "教".to_string()
            }
        );

        let subject = parse("哪些老师教过张三").expect("问主语的名词短语");
        assert_eq!(subject.rule, "active-wh-np-subject");
        assert_eq!(
            subject.plan,
            QueryPlan::FindSubject {
                object: "张三".to_string(),
                predicate: "教".to_string()
            }
        );

        let passive = parse("张三被哪些老师教过").expect("被字句中的名词短语");
        assert_eq!(
            passive.plan,
            QueryPlan::FindSubject {
                object: "张三".to_string(),
                predicate: "教".to_string()
            }
        );
        let ba = parse("张三把哪些学生教过").expect("把字句中的名词短语");
        assert_eq!(
            ba.plan,
            QueryPlan::FindObjectByPredicate {
                subject: "张三".to_string(),
                predicate: "教".to_string()
            }
        );
    }

    #[test]
    fn relative_clauses_compile_to_relation_lookups() {
        let object = parse("张三教过的学生是谁").expect("关系从句问宾语");
        assert_eq!(object.rule, "typed-relative-object");
        assert_eq!(
            object.plan,
            QueryPlan::FindObjectByPredicate {
                subject: "张三".to_string(),
                predicate: "教".to_string()
            }
        );

        let subject = parse("教过张三的老师有哪些").expect("关系从句问主语");
        assert_eq!(subject.rule, "typed-relative-subject");
        assert_eq!(
            subject.plan,
            QueryPlan::FindSubject {
                object: "张三".to_string(),
                predicate: "教".to_string()
            }
        );

        let passive = parse("被张三教过的学生是谁").expect("被字关系从句");
        assert_eq!(passive.rule, "typed-relative-passive-patient");
        assert_eq!(
            passive.plan,
            QueryPlan::FindObjectByPredicate {
                subject: "张三".to_string(),
                predicate: "教".to_string()
            }
        );
    }

    #[test]
    fn relation_counts_use_the_count_plan() {
        for question in ["张三教过多少个学生", "张三教过几个学生", "张三教过多少"] {
            let rewrite = parse(question).expect(question);
            assert_eq!(rewrite.rule, "relation-count");
            assert_eq!(rewrite.predicate, "教");
            assert_eq!(
                rewrite.plan,
                QueryPlan::Count {
                    subject: "张三".to_string(),
                    predicate: "教".to_string()
                },
                "{}",
                question
            );
        }
        assert!(parse("张三有几个师傅").is_none());
    }

    #[test]
    fn relative_clause_without_relation_words_abstains() {
        assert!(parse("张三的师傅是谁").is_none());
        assert!(parse("张三教过的学生").is_none());
    }

    #[test]
    fn polite_prefix_and_punctuation_are_ignored() {
        let rewrite = parse("请问谁认识张三？").expect("礼貌前缀应被剥离");
        assert_eq!(
            rewrite.plan,
            QueryPlan::FindSubject {
                object: "张三".to_string(),
                predicate: "认识".to_string()
            }
        );
    }

    #[test]
    fn unknown_terms_are_not_guessed() {
        assert!(parse("张三认识未名").is_none());
        assert!(parse("张三折腾谁").is_none());
    }

    #[test]
    fn non_relation_questions_abstain() {
        assert!(parse("什么是知识图谱").is_none());
        assert!(parse("今天天气怎么样").is_none());
        assert!(parse("张三认识李四").is_none());
    }

    #[test]
    fn receipt_is_human_readable() {
        let rewrite = parse("谁被张三教过").unwrap();
        let receipt = rewrite.receipt();
        assert!(receipt.contains("passive-wh-patient"), "{}", receipt);
        assert!(receipt.contains("张三"), "{}", receipt);
        assert!(receipt.contains("教"), "{}", receipt);
    }
}

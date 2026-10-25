use std::collections::BTreeSet;

use crate::symbolic_generator::{localized_type_label, SymbolicGenerator};

const MEMBER_PAGE_SIZE: usize = 20;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeQuestion {
    Profile { etype: String },
    Members { etype: String },
    Compare { left: String, right: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeAnswer {
    pub text: String,
    pub focus: Option<String>,
}

pub fn answer(generator: &SymbolicGenerator, question: &str) -> Option<TypeAnswer> {
    let parsed = parse(generator, question)?;
    let (text, focus) = match parsed {
        TypeQuestion::Profile { etype } => (generator.type_profile_report(&etype)?, Some(etype)),
        TypeQuestion::Members { etype } => {
            let page = generator.type_member_page(&etype, 0, MEMBER_PAGE_SIZE)?;
            (page.render(), Some(etype))
        }
        TypeQuestion::Compare { left, right } => {
            (generator.type_comparison_report(&left, &right)?, Some(left))
        }
    };
    Some(TypeAnswer { text, focus })
}

pub fn parse(generator: &SymbolicGenerator, question: &str) -> Option<TypeQuestion> {
    let question = normalize(question);
    if question.is_empty() || generator.all_types().is_empty() {
        return None;
    }

    if question.contains("类型画像") || question.contains("类型概览") {
        if let Some(etype) = profile_mention(generator, &question) {
            return Some(TypeQuestion::Profile { etype });
        }
    }

    let mentions = known_type_mentions(generator, &question);
    if mentions.is_empty() {
        return None;
    }

    if has_any(&question, &["区别", "不同", "差异", "比较", "对比", "相比"]) && mentions.len() >= 2
    {
        let left = mentions[0].etype.clone();
        let right = mentions
            .iter()
            .find(|mention| mention.etype != left)
            .map(|mention| mention.etype.clone())?;
        let collides = mentions
            .iter()
            .any(|mention| generator.is_known_entity(&mention.etype));
        if collides && !question.contains("类型") {
            return None;
        }
        return Some(TypeQuestion::Compare { left, right });
    }

    let profile_intent = has_profile_intent(&question);
    let member_intent = has_member_intent(&question, profile_intent)
        || (question.ends_with("有哪些")
            && mentions.first().is_some_and(|mention| mention.start == 0));
    let explicit_marker = question.contains("类型");
    let chosen = mentions
        .iter()
        .find(|mention| !generator.is_known_entity(&mention.etype))
        .or_else(|| mentions.first())?;
    if generator.is_known_entity(&chosen.etype)
        && !explicit_marker
        && !profile_intent
        && !member_intent
    {
        return None;
    }
    if profile_intent {
        return Some(TypeQuestion::Profile {
            etype: chosen.etype.clone(),
        });
    }
    if member_intent {
        return Some(TypeQuestion::Members {
            etype: chosen.etype.clone(),
        });
    }
    None
}

fn normalize(question: &str) -> String {
    question
        .trim()
        .trim_matches(|c: char| {
            c.is_whitespace()
                || matches!(
                    c,
                    '「' | '」'
                        | '“'
                        | '”'
                        | '"'
                        | '\''
                        | '？'
                        | '?'
                        | '！'
                        | '!'
                        | '。'
                        | '，'
                        | ','
                        | '：'
                        | ':'
                )
        })
        .to_string()
}

fn has_any(text: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| text.contains(needle))
}

fn has_profile_intent(question: &str) -> bool {
    let topic = has_any(
        question,
        &["画像", "概览", "概况", "属性", "关系", "特征", "特点"],
    );
    let asks = has_any(
        question,
        &[
            "哪些",
            "什么",
            "有什么",
            "一般",
            "通常",
            "常见",
            "介绍",
            "长什么样",
            "怎么样",
        ],
    );
    (topic && asks)
        || question.ends_with("的属性")
        || question.ends_with("的关系")
        || (question.starts_with("介绍") && question.contains("类型"))
}

fn has_member_intent(question: &str, profile_intent: bool) -> bool {
    if profile_intent {
        return false;
    }
    has_any(
        question,
        &[
            "成员",
            "名单",
            "属于",
            "包括",
            "列出",
            "哪些实体",
            "哪些人",
            "有谁",
            "谁是",
        ],
    )
}

fn profile_mention(generator: &SymbolicGenerator, question: &str) -> Option<String> {
    for quoted in quoted_mentions(question) {
        if let Some(etype) = generator.resolve_type_name(&quoted) {
            return Some(etype);
        }
    }
    let stripped = strip_profile_marker(question);
    if let Some(etype) = generator.resolve_type_name(&stripped) {
        return Some(etype);
    }
    known_type_mentions(generator, question)
        .first()
        .map(|mention| mention.etype.clone())
}

fn strip_profile_marker(question: &str) -> String {
    let mut text = question.to_string();
    for prefix in [
        "请帮我",
        "麻烦你",
        "帮我",
        "麻烦",
        "查看一下",
        "查看",
        "显示一下",
        "显示",
        "生成",
        "介绍一下",
        "介绍",
        "看看",
    ] {
        if let Some(rest) = text.strip_prefix(prefix) {
            text = rest.trim_start().to_string();
        }
    }
    for marker in ["的类型画像", "类型画像", "的类型概览", "类型概览"] {
        text = text.replace(marker, "");
    }
    text.trim_matches(|c: char| {
        c.is_whitespace()
            || matches!(
                c,
                '的' | '是' | '有' | '哪' | '些' | '什' | '么' | '怎' | '样' | '请' | '看'
            )
    })
    .to_string()
}

fn quoted_mentions(text: &str) -> Vec<String> {
    let mut mentions = Vec::new();
    for (open, close) in [('「', '」'), ('“', '”'), ('"', '"')] {
        let mut rest = text;
        while let Some(start) = rest.find(open) {
            let after_open = &rest[start + open.len_utf8()..];
            let Some(end) = after_open.find(close) else {
                break;
            };
            let mention = after_open[..end].trim();
            if !mention.is_empty() {
                mentions.push(mention.to_string());
            }
            rest = &after_open[end + close.len_utf8()..];
        }
    }
    mentions
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TypeMention {
    start: usize,
    end: usize,
    etype: String,
}

fn known_type_mentions(generator: &SymbolicGenerator, text: &str) -> Vec<TypeMention> {
    let haystack = text.to_ascii_lowercase();
    let mut raw = Vec::new();
    for etype in generator.all_types() {
        let mut surfaces: BTreeSet<String> = BTreeSet::new();
        surfaces.insert(etype.clone());
        surfaces.insert(localized_type_label(&etype).to_string());
        for surface in surfaces {
            if surface.chars().count() < 2 {
                continue;
            }
            let needle = surface.to_ascii_lowercase();
            for (start, matched) in haystack.match_indices(&needle) {
                raw.push(TypeMention {
                    start,
                    end: start + matched.len(),
                    etype: etype.clone(),
                });
            }
        }
    }
    raw.sort_by(|left, right| {
        left.start
            .cmp(&right.start)
            .then_with(|| right.end.cmp(&left.end))
    });
    let mut mentions: Vec<TypeMention> = Vec::new();
    let mut last_end = 0usize;
    for mention in raw {
        if mention.start < last_end {
            continue;
        }
        if mentions
            .last()
            .is_some_and(|previous: &TypeMention| previous.etype == mention.etype)
        {
            last_end = mention.end;
            continue;
        }
        last_end = mention.end;
        mentions.push(mention);
    }
    mentions
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generator() -> SymbolicGenerator {
        let mut generator = SymbolicGenerator::new(6, 0.3);
        generator.observe_entity("张三", "人物", &[("年龄".to_string(), "30".to_string())]);
        generator.observe_entity("李四", "人物", &[("年龄".to_string(), "40".to_string())]);
        generator.observe_entity(
            "甲公司",
            "组织",
            &[("城市".to_string(), "北京".to_string())],
        );
        generator.observe_entity("乙公司", "组织", &[]);
        generator.observe_triple("张三", "师傅", "李四");
        generator
    }

    #[test]
    fn profile_question_resolves_labels_and_renders() {
        let generator = generator();
        let parsed = parse(&generator, "人物一般有哪些属性？").unwrap();
        assert_eq!(
            parsed,
            TypeQuestion::Profile {
                etype: "人物".to_string()
            }
        );
        let answer = answer(&generator, "人物一般有哪些属性？").unwrap();
        assert!(answer.text.contains("类型画像「人物」"));
        assert!(answer.text.contains("年龄"));
        assert_eq!(answer.focus.as_deref(), Some("人物"));
    }

    #[test]
    fn entity_name_is_not_hijacked_by_type_name() {
        let mut generator = generator();
        generator.observe_entity("人物", "人物", &[]);
        assert!(parse(&generator, "人物是谁").is_none());
        assert!(parse(&generator, "人物有哪些成员").is_some());
        assert!(parse(&generator, "人物类型画像").is_some());
    }

    #[test]
    fn comparison_keeps_question_order() {
        let generator = generator();
        let parsed = parse(&generator, "比较组织和人物的区别").unwrap();
        assert_eq!(
            parsed,
            TypeQuestion::Compare {
                left: "组织".to_string(),
                right: "人物".to_string(),
            }
        );
        let answer = answer(&generator, "比较组织和人物的区别").unwrap();
        assert!(answer.text.contains("类型对比「组织」与「人物」"));
        assert!(answer.text.contains("城市"));
        assert!(answer.text.contains("师傅"));
        assert!(answer.text.contains("年龄"));
    }

    #[test]
    fn member_questions_return_a_bounded_page() {
        let generator = generator();
        let parsed = parse(&generator, "人物有哪些成员").unwrap();
        assert_eq!(
            parsed,
            TypeQuestion::Members {
                etype: "人物".to_string()
            }
        );
        let answer = answer(&generator, "人物有哪些成员").unwrap();
        assert!(answer.text.contains("张三"));
        assert!(answer.text.contains("李四"));
        assert!(answer.text.contains("已到成员表末尾"));
    }

    #[test]
    fn explicit_profile_marker_accepts_quotes() {
        let generator = generator();
        let parsed = parse(&generator, "查看「人物」的类型画像").unwrap();
        assert_eq!(
            parsed,
            TypeQuestion::Profile {
                etype: "人物".to_string()
            }
        );
    }

    #[test]
    fn unknown_type_and_unrelated_questions_abstain() {
        let generator = generator();
        assert!(parse(&generator, "神兽有哪些属性").is_none());
        assert!(parse(&generator, "张三的年龄是多少").is_none());
        assert!(parse(&generator, "今天天气怎么样").is_none());
        assert!(parse(&generator, "教过张三的老师有哪些").is_none());
    }
}

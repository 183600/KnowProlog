//! 实体事实清单问答：把「X 有哪些属性 / 关系 / 信息」编译成对生成器
//! 既有索引的只读列举。
//!
//! 本模块只遍历 [`SymbolicGenerator`] 已观察到的属性对与关系对，不推断、
//! 不生成新事实；清单按谓词与取值排序，保证同一知识版本输出逐字一致。
//! 问句必须显式出现「属性 / 关系 / 信息」清单提示，且提示紧跟在已知实体
//! 之后，因此「张三的师傅有哪些属性」不会被误当成在问「张三」。

use crate::symbolic_generator::{localized_predicate_key, localized_type_label, SymbolicGenerator};

const MAX_ROWS: usize = 20;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InventoryScope {
    Attributes,
    Relations,
    Both,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InventoryQuestion {
    pub entity: String,
    pub scope: InventoryScope,
}

const ATTRIBUTE_TAILS: &[&str] = &[
    "有哪些属性",
    "有什么属性",
    "都有哪些属性",
    "属性有哪些",
    "属性有什么",
    "的属性有哪些",
    "的属性有什么",
    "有哪些属性值",
];

const RELATION_TAILS: &[&str] = &[
    "有哪些关系",
    "有什么关系",
    "都有哪些关系",
    "关系有哪些",
    "关系有什么",
    "的关系有哪些",
    "的关系有什么",
];

const BOTH_TAILS: &[&str] = &[
    "有哪些信息",
    "有什么信息",
    "都有哪些信息",
    "信息有哪些",
    "有哪些资料",
    "有什么资料",
    "资料有哪些",
];

pub fn parse(generator: &SymbolicGenerator, question: &str) -> Option<InventoryQuestion> {
    let question = normalize(question);
    if question.is_empty() {
        return None;
    }
    let (entity, scope) = match_entity_tail(generator, &question)?;
    Some(InventoryQuestion { entity, scope })
}

pub fn answer(generator: &SymbolicGenerator, question: &str) -> Option<String> {
    let parsed = parse(generator, question)?;
    Some(render(generator, &parsed))
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

fn trim_particles(mut text: &str) -> &str {
    loop {
        let trimmed = text.trim_end_matches(|c: char| {
            matches!(c, '呢' | '啊' | '呀' | '吧' | '吗' | '。' | '？' | '?' | '！' | '!')
        });
        if trimmed == text {
            return text;
        }
        text = trimmed;
    }
}

fn match_entity_tail(
    generator: &SymbolicGenerator,
    question: &str,
) -> Option<(String, InventoryScope)> {
    let mut entities = generator.all_entities();
    entities.sort_by(|left, right| {
        right
            .chars()
            .count()
            .cmp(&left.chars().count())
            .then_with(|| left.cmp(right))
    });
    for entity in entities {
        let entity = entity.trim();
        if entity.is_empty() {
            continue;
        }
        for (start, _) in question.match_indices(entity) {
            let suffix = question[start + entity.len()..]
                .trim_start_matches(|c: char| matches!(c, '」' | '”' | '"' | '\'' | ' '));
            let suffix = trim_particles(suffix);
            if ATTRIBUTE_TAILS.contains(&suffix) {
                return Some((entity.to_string(), InventoryScope::Attributes));
            }
            if RELATION_TAILS.contains(&suffix) {
                return Some((entity.to_string(), InventoryScope::Relations));
            }
            if BOTH_TAILS.contains(&suffix) {
                return Some((entity.to_string(), InventoryScope::Both));
            }
        }
    }
    None
}

fn render(generator: &SymbolicGenerator, question: &InventoryQuestion) -> String {
    let entity = question.entity.as_str();
    let type_note = generator
        .entity_type_of(entity)
        .filter(|etype| !etype.trim().is_empty())
        .map(|etype| format!("（{}）", localized_type_label(etype)))
        .unwrap_or_default();
    let mut sections = Vec::new();
    if matches!(question.scope, InventoryScope::Attributes | InventoryScope::Both) {
        sections.push(render_attributes(generator, entity));
    }
    if matches!(question.scope, InventoryScope::Relations | InventoryScope::Both) {
        sections.push(render_relations(generator, entity));
    }
    format!(
        "「{}」{}的事实清单：\n{}",
        entity,
        type_note,
        sections.join("\n")
    )
}

fn render_attributes(generator: &SymbolicGenerator, entity: &str) -> String {
    let pairs = generator.attribute_pairs(entity);
    let mut output = format!("  属性（{} 项）：", pairs.len());
    if pairs.is_empty() {
        output.push_str("暂无。");
        return output;
    }
    output.push('\n');
    for (predicate, value) in pairs.iter().take(MAX_ROWS) {
        let label = localized_predicate_key(predicate).unwrap_or(predicate);
        output.push_str(&format!("    - {}：{}\n", label, value));
    }
    if pairs.len() > MAX_ROWS {
        output.push_str(&format!("    （其余 {} 项未展开）\n", pairs.len() - MAX_ROWS));
    }
    output.trim_end().to_string()
}

fn render_relations(generator: &SymbolicGenerator, entity: &str) -> String {
    let outgoing = generator.relation_pairs(entity);
    let incoming: Vec<(String, String)> = {
        let mut pairs: Vec<(String, String)> = generator
            .relation_predicates_in(entity)
            .into_iter()
            .flat_map(|predicate| {
                generator
                    .relation_values_in(entity, &predicate)
                    .into_iter()
                    .map(move |subject| (predicate.clone(), subject))
            })
            .collect();
        pairs.sort();
        pairs.dedup();
        pairs
    };
    let total = outgoing.len() + incoming.len();
    let mut output = format!("  关系（{} 项）：", total);
    if total == 0 {
        output.push_str("暂无。");
        return output;
    }
    output.push('\n');
    let mut shown = 0usize;
    for (predicate, object) in &outgoing {
        if shown >= MAX_ROWS {
            break;
        }
        let label = localized_predicate_key(predicate).unwrap_or(predicate);
        output.push_str(&format!("    - {} {} {}\n", entity, label, object));
        shown += 1;
    }
    for (predicate, subject) in &incoming {
        if shown >= MAX_ROWS {
            break;
        }
        let label = localized_predicate_key(predicate).unwrap_or(predicate);
        output.push_str(&format!("    - {} {} {}（入向）\n", subject, label, entity));
        shown += 1;
    }
    if total > shown {
        output.push_str(&format!("    （其余 {} 项未展开）\n", total - shown));
    }
    output.trim_end().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generator() -> SymbolicGenerator {
        let mut generator = SymbolicGenerator::new(6, 0.3);
        generator.observe_entity(
            "李四",
            "Person",
            &[
                ("姓名".to_string(), "李四".to_string()),
                ("年龄".to_string(), "30".to_string()),
            ],
        );
        generator.observe_entity("张三", "Person", &[]);
        generator.observe_triple("张三", "师傅", "李四");
        generator
    }

    #[test]
    fn parses_attribute_relation_and_combined_inventory_questions() {
        let generator = generator();
        assert_eq!(
            parse(&generator, "李四有哪些属性？"),
            Some(InventoryQuestion {
                entity: "李四".to_string(),
                scope: InventoryScope::Attributes,
            })
        );
        assert_eq!(
            parse(&generator, "李四的关系有哪些"),
            Some(InventoryQuestion {
                entity: "李四".to_string(),
                scope: InventoryScope::Relations,
            })
        );
        assert_eq!(
            parse(&generator, "「李四」有哪些信息呢"),
            Some(InventoryQuestion {
                entity: "李四".to_string(),
                scope: InventoryScope::Both,
            })
        );
    }

    #[test]
    fn does_not_steal_attribute_value_or_chained_entity_questions() {
        let generator = generator();
        assert_eq!(parse(&generator, "李四的年龄是多少"), None);
        assert_eq!(parse(&generator, "张三的师傅有哪些属性"), None);
        assert_eq!(parse(&generator, "王五有哪些属性"), None);
    }

    #[test]
    fn renders_sorted_attributes_and_incoming_relations() {
        let generator = generator();
        let attributes = answer(&generator, "李四有哪些属性").unwrap();
        assert!(attributes.contains("属性（2 项）"), "{}", attributes);
        assert!(attributes.contains("年龄：30"), "{}", attributes);
        assert!(attributes.contains("姓名：李四"), "{}", attributes);

        let relations = answer(&generator, "李四有哪些关系").unwrap();
        assert!(relations.contains("张三 师傅 李四（入向）"), "{}", relations);

        let both = answer(&generator, "李四有哪些信息").unwrap();
        assert!(both.contains("属性（2 项）"), "{}", both);
        assert!(both.contains("关系（1 项）"), "{}", both);
    }

    #[test]
    fn unknown_or_empty_slots_are_reported_honestly() {
        let generator = generator();
        let empty = answer(&generator, "张三有哪些属性").unwrap();
        assert!(empty.contains("属性（0 项）：暂无"), "{}", empty);
    }
}

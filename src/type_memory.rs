use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::schema_view::SchemaView;
use crate::symbolic_generator::{
    localized_type_label, SymbolicGenerator, TypeComparison, TypeMemberPage, TypePredicateSummary,
    TypeProfile,
};

const MEMBER_PAGE_SIZE: usize = 20;
const PROFILE_CONTINUATION_PREDICATES: usize = 64;
const TYPE_FOCUS_CAPACITY: usize = 4;
const TYPE_FOCUS_TTL: u32 = 3;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeQuestion {
    Profile { etype: String },
    Members { etype: String },
    Count { etype: String },
    Compare { left: String, right: String },
    /// 查某个已知实体的显式类型：`李四是什么类型`。
    EntityType { entity: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeAnswer {
    pub text: String,
    pub focus: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TypeQuestionKind {
    Profile,
    Members,
    Compare,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TypeFocus {
    pub etype: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peer: Option<String>,
    pub kind: TypeQuestionKind,
    #[serde(default)]
    pub member_offset: usize,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub spoken_slots: BTreeSet<String>,
}

impl TypeFocus {
    fn new(etype: impl Into<String>, kind: TypeQuestionKind) -> Self {
        Self {
            etype: etype.into(),
            peer: None,
            kind,
            member_offset: 0,
            spoken_slots: BTreeSet::new(),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TypeDialogueMemory {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    focus: Vec<TypeFocus>,
    #[serde(default, skip_serializing_if = "is_zero")]
    idle_turns: u32,
}

fn is_zero(value: &u32) -> bool {
    *value == 0
}

impl TypeDialogueMemory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.focus.is_empty()
    }

    pub fn focus(&self) -> Option<&TypeFocus> {
        self.focus.last()
    }

    pub fn focus_type(&self) -> Option<&str> {
        self.focus().map(|focus| focus.etype.as_str())
    }

    pub fn focus_stack(&self) -> &[TypeFocus] {
        &self.focus
    }

    pub fn clear(&mut self) {
        self.focus.clear();
        self.idle_turns = 0;
    }

    pub fn note_non_type_turn(&mut self) {
        if self.focus.is_empty() {
            return;
        }
        self.idle_turns = self.idle_turns.saturating_add(1);
        if self.idle_turns >= TYPE_FOCUS_TTL {
            self.clear();
        }
    }

    fn set_focus(&mut self, focus: TypeFocus) {
        self.focus.retain(|previous| previous.etype != focus.etype);
        self.focus.push(focus);
        if self.focus.len() > TYPE_FOCUS_CAPACITY {
            self.focus.remove(0);
        }
        self.idle_turns = 0;
    }

    fn current_mut(&mut self) -> Option<&mut TypeFocus> {
        self.idle_turns = 0;
        self.focus.last_mut()
    }
}

#[derive(Debug, Clone)]
struct TypeTarget {
    key: String,
    base_types: Vec<String>,
    members: Vec<String>,
    revision: u64,
}

pub fn answer(generator: &SymbolicGenerator, question: &str) -> Option<TypeAnswer> {
    answer_with_schema(generator, None, question)
}

pub fn answer_with_schema(
    generator: &SymbolicGenerator,
    schema: Option<&SchemaView>,
    question: &str,
) -> Option<TypeAnswer> {
    let mut memory = TypeDialogueMemory::new();
    answer_in_context(generator, schema, &mut memory, question)
}

pub fn answer_in_context(
    generator: &SymbolicGenerator,
    schema: Option<&SchemaView>,
    memory: &mut TypeDialogueMemory,
    question: &str,
) -> Option<TypeAnswer> {
    let normalized = normalize(question);
    if is_gap_question(&normalized) {
        let mentioned = known_type_mentions(generator, schema, &normalized)
            .into_iter()
            .find(|mention| !generator.is_known_entity(&mention.etype))
            .map(|mention| mention.etype);
        let target = match mentioned.as_deref() {
            Some(etype) => resolve_target(generator, schema, etype),
            None => {
                let names_entity = generator
                    .all_entities()
                    .iter()
                    .any(|entity| !entity.trim().is_empty() && normalized.contains(entity));
                if names_entity {
                    None
                } else {
                    memory
                        .focus()
                        .and_then(|focus| resolve_target(generator, schema, &focus.etype))
                }
            }
        };
        if let Some(target) = target {
            let text = render_gap(generator, &target);
            if mentioned.is_some() || memory.focus().is_none() {
                memory.set_focus(TypeFocus::new(target.key.clone(), TypeQuestionKind::Profile));
            } else {
                memory.current_mut();
            }
            return Some(TypeAnswer {
                text,
                focus: Some(target.key),
            });
        }
    }

    if let Some((parsed, _kind)) = parse_details(generator, schema, &normalized) {
        return answer_parsed(generator, schema, memory, parsed);
    }

    let focus = memory.focus().cloned()?;

    if is_continuation(&normalized) {
        return continue_focus(generator, schema, memory, &focus);
    }
    if is_deepening(&normalized) {
        return deepen_focus(generator, schema, memory, &focus);
    }
    if !is_switch_cue(&normalized) {
        return None;
    }

    let mentions = known_type_mentions(generator, schema, &normalized);
    if mentions.is_empty() {
        return None;
    }
    let left = mentions
        .iter()
        .find(|mention| !generator.is_known_entity(&mention.etype))
        .or_else(|| mentions.first())?;
    if focus.kind == TypeQuestionKind::Compare && mentions.len() >= 2 {
        let right = mentions
            .iter()
            .find(|mention| mention.etype != left.etype)
            .map(|mention| mention.etype.clone())?;
        return answer_parsed(
            generator,
            schema,
            memory,
            TypeQuestion::Compare {
                left: left.etype.clone(),
                right,
            },
        );
    }
    let kind = match focus.kind {
        TypeQuestionKind::Members => TypeQuestionKind::Members,
        TypeQuestionKind::Profile | TypeQuestionKind::Compare => TypeQuestionKind::Profile,
    };
    let parsed = match kind {
        TypeQuestionKind::Members => TypeQuestion::Members {
            etype: left.etype.clone(),
        },
        _ => TypeQuestion::Profile {
            etype: left.etype.clone(),
        },
    };
    answer_parsed(generator, schema, memory, parsed)
}

fn answer_parsed(
    generator: &SymbolicGenerator,
    schema: Option<&SchemaView>,
    memory: &mut TypeDialogueMemory,
    parsed: TypeQuestion,
) -> Option<TypeAnswer> {
    match parsed {
        TypeQuestion::Profile { etype } => {
            let target = resolve_target(generator, schema, &etype)?;
            let mut focus = TypeFocus::new(target.key.clone(), TypeQuestionKind::Profile);
            let text = match profile_for_target(generator, &target, 12, 6) {
                Some(profile) => {
                    focus.spoken_slots = profile
                        .predicates
                        .iter()
                        .map(|summary| summary.predicate.clone())
                        .collect();
                    profile.render()
                }
                None => empty_profile_text(&target.key),
            };
            let focus_key = target.key.clone();
            memory.set_focus(focus);
            Some(TypeAnswer {
                text,
                focus: Some(focus_key),
            })
        }
        TypeQuestion::Members { etype } => {
            let target = resolve_target(generator, schema, &etype)?;
            let page = member_page(&target, 0, MEMBER_PAGE_SIZE);
            let mut focus = TypeFocus::new(target.key.clone(), TypeQuestionKind::Members);
            focus.member_offset = page.next_offset;
            let text = render_member_page(&target, &page);
            let focus_key = target.key.clone();
            memory.set_focus(focus);
            Some(TypeAnswer {
                text,
                focus: Some(focus_key),
            })
        }
        TypeQuestion::Count { etype } => {
            let target = resolve_target(generator, schema, &etype)?;
            let text = render_type_count(&target);
            // 计数后把焦点留在成员表上：「他们是谁 / 还有呢」直接列出成员，
            // 不需要用户再重复类型名。
            memory.set_focus(TypeFocus::new(
                target.key.clone(),
                TypeQuestionKind::Members,
            ));
            Some(TypeAnswer {
                text,
                focus: Some(target.key),
            })
        }
        TypeQuestion::Compare { left, right } => {
            let left_target = resolve_target(generator, schema, &left)?;
            let right_target = resolve_target(generator, schema, &right)?;
            let mut focus = TypeFocus::new(left_target.key.clone(), TypeQuestionKind::Compare);
            focus.peer = Some(right_target.key.clone());
            let text = render_comparison(generator, schema, &left_target, &right_target);
            let focus_key = left_target.key.clone();
            memory.set_focus(focus);
            Some(TypeAnswer {
                text,
                focus: Some(focus_key),
            })
        }
        TypeQuestion::EntityType { entity } => {
            let Some(stored) = generator.entity_type_of(&entity) else {
                return Some(TypeAnswer {
                    text: format!("「{}」尚未登记显式类型。", entity),
                    focus: None,
                });
            };
            let canonical = schema
                .map(|schema| schema.canonical_type(stored))
                .filter(|canonical| !canonical.is_empty())
                .unwrap_or_else(|| stored.to_string());
            let text = render_entity_type(schema, &entity, &canonical);
            memory.set_focus(TypeFocus::new(canonical.clone(), TypeQuestionKind::Profile));
            Some(TypeAnswer {
                text,
                focus: Some(canonical),
            })
        }
    }
}

fn render_entity_type(
    schema: Option<&SchemaView>,
    entity: &str,
    canonical: &str,
) -> String {
    let label = localized_type_label(canonical);
    let mut text = if label != canonical {
        format!("「{}」的类型是「{}」（{}）。", entity, canonical, label)
    } else {
        format!("「{}」的类型是「{}」。", entity, canonical)
    };
    if let Some(schema) = schema {
        let mut supers: Vec<String> = schema
            .class_ancestors(canonical)
            .into_iter()
            .filter(|class| class != canonical)
            .map(|class| {
                let label = localized_type_label(&class);
                if label != class {
                    format!("{}（{}）", class, label)
                } else {
                    class
                }
            })
            .collect();
        supers.sort();
        if !supers.is_empty() {
            text.push_str(&format!("\n上位类型：{}。", supers.join("、")));
        }
    }
    text
}

fn continue_focus(
    generator: &SymbolicGenerator,
    schema: Option<&SchemaView>,
    memory: &mut TypeDialogueMemory,
    focus: &TypeFocus,
) -> Option<TypeAnswer> {
    match focus.kind {
        TypeQuestionKind::Members => {
            let target = resolve_target(generator, schema, &focus.etype)?;
            if target.members.is_empty() {
                memory.current_mut()?.member_offset = 0;
                return Some(TypeAnswer {
                    text: empty_members_text(&target.key),
                    focus: Some(target.key),
                });
            }
            let page = member_page(&target, focus.member_offset, MEMBER_PAGE_SIZE);
            if page.members.is_empty() {
                let text = format!(
                    "类型「{}」的成员已全部列出（共 {} 个）。",
                    target.key, page.total
                );
                memory.current_mut()?.member_offset = page.next_offset;
                return Some(TypeAnswer {
                    text,
                    focus: Some(target.key),
                });
            }
            let mut focus_key = target.key.clone();
            let text = render_member_page(&target, &page);
            if let Some(current) = memory.current_mut() {
                current.member_offset = page.next_offset;
                focus_key = current.etype.clone();
            }
            Some(TypeAnswer {
                text,
                focus: Some(focus_key),
            })
        }
        TypeQuestionKind::Profile => {
            let target = resolve_target(generator, schema, &focus.etype)?;
            let Some(profile) =
                profile_for_target(generator, &target, PROFILE_CONTINUATION_PREDICATES, 0)
            else {
                return Some(TypeAnswer {
                    text: empty_profile_text(&target.key),
                    focus: Some(target.key),
                });
            };
            let remaining: Vec<&TypePredicateSummary> = profile
                .predicates
                .iter()
                .filter(|summary| !focus.spoken_slots.contains(&summary.predicate))
                .collect();
            let text = if remaining.is_empty() {
                if profile.predicates_truncated {
                    format!(
                        "类型画像「{}」已展开 {} 种谓词，其余 {} 种暂未展开。",
                        target.key,
                        profile.predicates.len(),
                        profile
                            .total_predicates
                            .saturating_sub(profile.predicates.len())
                    )
                } else {
                    format!("类型画像「{}」的谓词已全部展开。", target.key)
                }
            } else {
                render_supplementary_predicates(&profile, &remaining)
            };
            let shown: BTreeSet<String> = profile
                .predicates
                .iter()
                .map(|summary| summary.predicate.clone())
                .collect();
            if let Some(current) = memory.current_mut() {
                current.spoken_slots.extend(shown);
            }
            Some(TypeAnswer {
                text,
                focus: Some(target.key),
            })
        }
        TypeQuestionKind::Compare => {
            let peer = focus.peer.clone();
            let text = match peer {
                Some(peer) => format!(
                    "类型对比「{}」与「{}」已在上一轮给出；如需继续，可以问「{}有哪些成员」。",
                    focus.etype, peer, focus.etype
                ),
                None => format!(
                    "类型对比「{}」已在上一轮给出；如需继续，可以问「{}有哪些成员」。",
                    focus.etype, focus.etype
                ),
            };
            memory.current_mut();
            Some(TypeAnswer {
                text,
                focus: Some(focus.etype.clone()),
            })
        }
    }
}

fn deepen_focus(
    generator: &SymbolicGenerator,
    schema: Option<&SchemaView>,
    memory: &mut TypeDialogueMemory,
    focus: &TypeFocus,
) -> Option<TypeAnswer> {
    match focus.kind {
        TypeQuestionKind::Members => continue_focus(generator, schema, memory, focus),
        TypeQuestionKind::Profile => continue_focus(generator, schema, memory, focus),
        TypeQuestionKind::Compare => {
            let left = resolve_target(generator, schema, &focus.etype)?;
            let Some(peer) = focus.peer.as_deref() else {
                return continue_focus(generator, schema, memory, focus);
            };
            let right = resolve_target(generator, schema, peer)?;
            let text = render_comparison(generator, schema, &left, &right);
            memory.current_mut();
            Some(TypeAnswer {
                text,
                focus: Some(left.key),
            })
        }
    }
}

pub fn parse(generator: &SymbolicGenerator, question: &str) -> Option<TypeQuestion> {
    parse_details(generator, None, question).map(|(question, _kind)| question)
}

fn parse_details(
    generator: &SymbolicGenerator,
    schema: Option<&SchemaView>,
    question: &str,
) -> Option<(TypeQuestion, TypeQuestionKind)> {
    let question = normalize(question);
    let schema_has_classes = schema.is_some_and(|schema| !schema.declared_classes().is_empty());
    if question.is_empty() || (generator.all_types().is_empty() && !schema_has_classes) {
        return None;
    }

    if let Some(entity) = entity_type_subject(generator, &question) {
        return Some((TypeQuestion::EntityType { entity }, TypeQuestionKind::Profile));
    }

    if question.contains("类型画像") || question.contains("类型概览") {
        if let Some(etype) = profile_mention(generator, schema, &question) {
            return Some((TypeQuestion::Profile { etype }, TypeQuestionKind::Profile));
        }
    }

    let mentions = known_type_mentions(generator, schema, &question);
    if mentions.is_empty() {
        // 单字类型别名（如「人」→「人物」）低于提及词表的两字门槛，
        // 但「有多少个人」「人一般有哪些属性」这类问句仍应落到类型上。
        if let Some(subject) = count_subject(&question) {
            if let Some(target) = resolve_target(generator, schema, &subject) {
                return Some((
                    TypeQuestion::Count { etype: target.key },
                    TypeQuestionKind::Members,
                ));
            }
        }
        if has_profile_intent(&question) {
            if let Some(etype) = leading_single_char_type(generator, schema, &question) {
                return Some((TypeQuestion::Profile { etype }, TypeQuestionKind::Profile));
            }
        }
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
        return Some((
            TypeQuestion::Compare { left, right },
            TypeQuestionKind::Compare,
        ));
    }

    let count_intent = count_remainder(&question).is_some();
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
    if count_intent {
        if let Some(remainder) = count_remainder(&question) {
            let stripped = strip_count_tail(remainder);
            // 只有量词/语气词（「人物有多少个」）或余下部分本身解析成类型
            // （「有多少个人物」）才是类型计数；「人物有多少属性」问的是
            // 属性个数，不能被劫持成成员计数。
            let bare = stripped.is_empty();
            let typed = !stripped.is_empty()
                && resolve_target(generator, schema, stripped).is_some();
            if bare || typed {
                let etype = if typed {
                    resolve_target(generator, schema, stripped)
                        .map(|target| target.key)
                        .unwrap_or_else(|| chosen.etype.clone())
                } else {
                    chosen.etype.clone()
                };
                return Some((
                    TypeQuestion::Count { etype },
                    TypeQuestionKind::Members,
                ));
            }
        }
    }
    if profile_intent {
        return Some((
            TypeQuestion::Profile {
                etype: chosen.etype.clone(),
            },
            TypeQuestionKind::Profile,
        ));
    }
    if member_intent {
        return Some((
            TypeQuestion::Members {
                etype: chosen.etype.clone(),
            },
            TypeQuestionKind::Members,
        ));
    }
    None
}

/// 实体类型问句的谓词尾部。必须整段匹配，避免把「张三的师傅是什么类型」
/// 误当成在问「张三」的类型。
const ENTITY_TYPE_TAILS: &[&str] = &[
    "是什么类型",
    "是啥类型",
    "是哪种类型",
    "是哪一种类型",
    "是哪类",
    "是哪一类",
    "是哪类实体",
    "是哪一类实体",
    "是什么类别",
    "是什么类型的人",
    "是什么类型的人物",
    "是什么类型的实体",
    "属于什么类型",
    "属于哪种类型",
    "属于哪一类",
    "属于哪类",
    "属于什么类别",
    "属于哪个类型",
    "属于何种类型",
    "算哪类",
    "算是什么类型",
    "是何种类型",
    "被归为什么类型",
    "被归为哪类",
    "的类型是什么",
    "的实体类型是什么",
    "的类别是什么",
    "什么类型",
    "哪种类型",
    "什么类别",
    "哪类",
];

/// 从问句中找出「问类型」的已知实体。最长实体优先，尾部必须完整命中
/// [`ENTITY_TYPE_TAILS`]，因此属性链问句不会被劫持。
fn entity_type_subject(generator: &SymbolicGenerator, question: &str) -> Option<String> {
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
            let suffix = suffix.trim_end_matches(|c: char| {
                matches!(c, '呢' | '啊' | '呀' | '吧' | '吗' | '。' | '？' | '?' | '！' | '!')
            });
            if ENTITY_TYPE_TAILS.contains(&suffix) {
                return Some(entity.to_string());
            }
        }
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

/// 计数问句的提示词（长的在前，避免「一共有多少」被「有多少」截断）。
const COUNT_CUES: &[&str] = &[
    "一共有多少",
    "总共有多少",
    "共有多少",
    "有多少",
    "一共有几个",
    "共有几个",
    "有几个",
];

/// 取计数提示词之后的部分（保留原词面，供类型解析与量词剥离）。
fn count_remainder(question: &str) -> Option<&str> {
    COUNT_CUES
        .iter()
        .find_map(|cue| question.split_once(cue).map(|(_, rest)| rest.trim()))
}

/// 去掉计数尾部的量词与语气词；返回空串表示「有多少个」这类裸计数。
fn strip_count_tail(text: &str) -> &str {
    let text = text.trim_start_matches(|c: char| {
        matches!(
            c,
            '个' | '位' | '名' | '只' | '条' | '种' | '部' | '所' | '家' | '件' | '项' | '次'
        )
    });
    text.trim_end_matches(|c: char| {
        matches!(
            c,
            '呢' | '吗' | '啊' | '呀' | '吧' | '了' | '的' | '是' | '有' | '共'
        )
    })
}

/// 计数问句里直接出现的类型词（如「有多少个人物」→「人物」）。
fn count_subject(question: &str) -> Option<String> {
    let rest = strip_count_tail(count_remainder(question)?.trim());
    if rest.is_empty() {
        None
    } else {
        Some(rest.to_string())
    }
}

fn has_any(text: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| text.contains(needle))
}

/// 单字类型别名 + 画像提示（如「人一般有哪些属性」）的兜底解析。
///
/// 只接受位于句首的单字，并要求其后紧跟画像提示，避免把普通句子里的
/// 单字误判成类型。
fn leading_single_char_type(
    generator: &SymbolicGenerator,
    schema: Option<&SchemaView>,
    question: &str,
) -> Option<String> {
    const PROFILE_TAILS: &[&str] = &[
        "一般", "通常", "常见", "有哪些", "有什么", "的属性", "的关系", "属性", "关系",
        "画像", "概览", "概况", "特征", "特点", "长什么样", "怎么样",
    ];
    let first = question.chars().next()?;
    let rest = &question[first.len_utf8()..];
    if !PROFILE_TAILS.iter().any(|tail| rest.starts_with(tail)) {
        return None;
    }
    resolve_target(generator, schema, &first.to_string()).map(|target| target.key)
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

fn is_continuation(question: &str) -> bool {
    matches!(
        question,
        "还有呢"
            | "还有吗"
            | "还有"
            | "继续"
            | "接着说"
            | "接下去"
            | "下一页"
            | "再看"
            | "更多"
            | "还有哪些"
            | "还有谁"
            | "继续列"
            | "继续看"
            | "再说说"
    )
}

fn is_gap_question(question: &str) -> bool {
    has_any(
        question,
        &[
            "缺什么",
            "缺少",
            "缺失",
            "覆盖缺口",
            "缺口",
            "哪些没",
            "还差",
            "补全",
        ],
    )
}

fn is_deepening(question: &str) -> bool {
    has_any(
        question,
        &[
            "再详细",
            "详细点",
            "详细说说",
            "具体点",
            "展开",
            "多说",
            "还有别的",
            "还有什么属性",
            "还有什么关系",
        ],
    )
}

fn is_switch_cue(question: &str) -> bool {
    question.starts_with("那")
        || question.starts_with("换成")
        || question.starts_with("换到")
        || question.starts_with("改问")
        || question.starts_with("改成")
        || question.starts_with("说说")
        || question.starts_with("讲讲")
        || question.ends_with('呢')
}

fn profile_mention(
    generator: &SymbolicGenerator,
    schema: Option<&SchemaView>,
    question: &str,
) -> Option<String> {
    for quoted in quoted_mentions(question) {
        if let Some(target) = resolve_target(generator, schema, &quoted) {
            return Some(target.key);
        }
    }
    let stripped = strip_profile_marker(question);
    if let Some(target) = resolve_target(generator, schema, &stripped) {
        return Some(target.key);
    }
    known_type_mentions(generator, schema, question)
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

fn known_type_mentions(
    generator: &SymbolicGenerator,
    schema: Option<&SchemaView>,
    text: &str,
) -> Vec<TypeMention> {
    let haystack = text.to_ascii_lowercase();
    let mut surfaces: BTreeMap<String, (u8, String)> = BTreeMap::new();
    let register =
        |surface: &str, etype: &str, priority: u8, surfaces: &mut BTreeMap<String, (u8, String)>| {
            let surface = surface.trim();
            if surface.chars().count() < 2 {
                return;
            }
            if let Some(target) = resolve_target(generator, schema, etype) {
                let entry = surfaces
                    .entry(surface.to_string())
                    .or_insert_with(|| (priority, target.key.clone()));
                if priority < entry.0 {
                    *entry = (priority, target.key);
                }
            }
        };
    for etype in generator.all_types() {
        register(&etype, &etype, 0, &mut surfaces);
    }
    if let Some(schema) = schema {
        for class in schema.declared_classes() {
            register(&class, &class, 1, &mut surfaces);
        }
        for (alias, _canonical) in schema.type_alias_pairs() {
            register(&alias, &alias, 2, &mut surfaces);
        }
    }
    for etype in generator.all_types() {
        let label = localized_type_label(&etype);
        if label != etype {
            register(label, &etype, 3, &mut surfaces);
        }
    }
    if let Some(schema) = schema {
        for class in schema.declared_classes() {
            let label = localized_type_label(&class);
            if label != class {
                register(label, &class, 3, &mut surfaces);
            }
        }
    }

    let mut raw = Vec::new();
    for (surface, (_priority, etype)) in surfaces {
        let needle = surface.to_ascii_lowercase();
        for (start, matched) in haystack.match_indices(&needle) {
            raw.push(TypeMention {
                start,
                end: start + matched.len(),
                etype: etype.clone(),
            });
        }
    }
    raw.sort_by(|left, right| {
        left.start
            .cmp(&right.start)
            .then_with(|| right.end.cmp(&left.end))
            .then_with(|| left.etype.cmp(&right.etype))
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

fn resolve_target(
    generator: &SymbolicGenerator,
    schema: Option<&SchemaView>,
    mention: &str,
) -> Option<TypeTarget> {
    let mention = mention.trim().trim_matches(|c: char| {
        matches!(
            c,
            '「' | '」' | '"' | '\'' | '“' | '”' | '？' | '?' | '。' | '，' | ',' | '：' | ':'
        )
    });
    if mention.is_empty() {
        return None;
    }
    let direct = generator.resolve_type_name(mention);
    let canonical = schema
        .map(|schema| schema.canonical_type(mention))
        .filter(|canonical| !canonical.is_empty())
        .unwrap_or_else(|| mention.to_string());
    let mut key = canonical;
    let mut base_types = qualifying_types(generator, schema, &key);
    if base_types.is_empty() {
        if let Some(direct) = direct {
            key = direct.clone();
            base_types = vec![direct];
        }
    }
    if base_types.is_empty() {
        let declared = schema.is_some_and(|schema| {
            schema
                .declared_classes()
                .iter()
                .any(|class| class == &key)
        });
        if !declared {
            return None;
        }
    }
    let mut members = BTreeSet::new();
    for base in &base_types {
        members.extend(generator.entities_of_type(base));
    }
    let revision = base_types
        .iter()
        .map(|base| generator.type_revision(base))
        .max()
        .unwrap_or(0);
    Some(TypeTarget {
        key,
        base_types,
        members: members.into_iter().collect(),
        revision,
    })
}

fn qualifying_types(
    generator: &SymbolicGenerator,
    schema: Option<&SchemaView>,
    target: &str,
) -> Vec<String> {
    generator
        .all_types()
        .into_iter()
        .filter(|etype| {
            if etype == target {
                return true;
            }
            let Some(schema) = schema else {
                return false;
            };
            let canonical = schema.canonical_type(etype);
            canonical == target
                || schema.is_subclass(&canonical, target)
                || schema.is_subclass(etype, target)
        })
        .collect()
}

fn profile_for_target(
    generator: &SymbolicGenerator,
    target: &TypeTarget,
    max_predicates: usize,
    max_samples: usize,
) -> Option<TypeProfile> {
    if target.members.is_empty() {
        return None;
    }
    if target.base_types.len() == 1 && target.base_types[0] == target.key {
        return generator.type_profile_bounded(&target.key, max_predicates, max_samples);
    }
    generator.type_profile_for_members(
        &target.key,
        &target.members,
        target.revision,
        max_predicates,
        max_samples,
    )
}

fn member_page(target: &TypeTarget, offset: usize, limit: usize) -> TypeMemberPage {
    let total = target.members.len();
    let limit = limit.max(1);
    let offset = offset.min(total);
    let members: Vec<String> = target
        .members
        .iter()
        .skip(offset)
        .take(limit)
        .cloned()
        .collect();
    let next_offset = offset + members.len();
    TypeMemberPage {
        type_name: target.key.clone(),
        offset,
        next_offset,
        total,
        done: next_offset >= total,
        members,
    }
}

fn compare_targets(
    generator: &SymbolicGenerator,
    left: &TypeTarget,
    right: &TypeTarget,
) -> Option<TypeComparison> {
    let left_profile = profile_for_target(generator, left, usize::MAX, 0)?;
    let right_profile = profile_for_target(generator, right, usize::MAX, 0)?;
    let left_predicates: BTreeSet<String> = left_profile
        .predicates
        .iter()
        .map(|summary| summary.predicate.clone())
        .collect();
    let right_predicates: BTreeSet<String> = right_profile
        .predicates
        .iter()
        .map(|summary| summary.predicate.clone())
        .collect();
    Some(TypeComparison {
        left: left_profile,
        right: right_profile,
        shared_predicates: left_predicates
            .intersection(&right_predicates)
            .cloned()
            .collect(),
        left_only_predicates: left_predicates
            .difference(&right_predicates)
            .cloned()
            .collect(),
        right_only_predicates: right_predicates
            .difference(&left_predicates)
            .cloned()
            .collect(),
    })
}

fn render_comparison(
    generator: &SymbolicGenerator,
    schema: Option<&SchemaView>,
    left: &TypeTarget,
    right: &TypeTarget,
) -> String {
    let mut text = match compare_targets(generator, left, right) {
        Some(comparison) => comparison.render(),
        None => format!(
            "类型对比「{}」与「{}」：\n  成员：{} {} 个；{} {} 个。\n  （至少一方暂无显式成员，暂无可比较的谓词。）",
            left.key,
            right.key,
            left.key,
            left.members.len(),
            right.key,
            right.members.len()
        ),
    };
    if let Some(note) = schema_relation_note(schema, &left.key, &right.key) {
        text.push_str(&format!("\n  {}", note));
    }
    text
}

fn schema_relation_note(
    schema: Option<&SchemaView>,
    left: &str,
    right: &str,
) -> Option<String> {
    let schema = schema?;
    if schema.is_subclass(left, right) {
        return Some(format!("模式关系：{} ⊑ {}。", left, right));
    }
    if schema.is_subclass(right, left) {
        return Some(format!("模式关系：{} ⊑ {}。", right, left));
    }
    if schema.types_disjoint(left, right) {
        return Some(format!("模式关系：{} ⊥ {}（可证明互斥）。", left, right));
    }
    None
}

fn render_type_count(target: &TypeTarget) -> String {
    if target.members.is_empty() {
        format!("类型「{}」当前没有已知成员。", target.key)
    } else {
        format!(
            "类型「{}」共有 {} 个已知成员。",
            target.key,
            target.members.len()
        )
    }
}

fn render_member_page(target: &TypeTarget, page: &TypeMemberPage) -> String {
    if target.members.is_empty() {
        return empty_members_text(&target.key);
    }
    page.render()
}

fn render_supplementary_predicates(
    profile: &TypeProfile,
    remaining: &[&TypePredicateSummary],
) -> String {
    let mut output = format!(
        "类型画像「{}」的补充谓词（{} 种）：\n",
        profile.type_name,
        remaining.len()
    );
    for (index, summary) in remaining.iter().enumerate() {
        output.push_str(&format!(
            "  {}. {}（{}）：{}/{} 个成员（{}%），共 {} 个取值",
            index + 1,
            summary.label,
            summary.kind.label(),
            summary.member_count,
            profile.total_members,
            summary.coverage_milli / 10,
            summary.value_count
        ));
        if let Some(sample) = summary.samples.first() {
            output.push_str(&format!(
                "；样例：{} → {}",
                sample.entity,
                sample.values.join("、")
            ));
        }
        output.push('\n');
    }
    if profile.predicates_truncated {
        output.push_str(&format!(
            "  （其余 {} 种谓词暂未展开。）\n",
            profile
                .total_predicates
                .saturating_sub(profile.predicates.len())
        ));
    }
    output.trim_end().to_string()
}

fn member_has_predicate(generator: &SymbolicGenerator, member: &str, predicate: &str) -> bool {
    !generator.relation_values(member, predicate).is_empty()
        || !generator.attribute_values(member, predicate).is_empty()
}

fn render_gap(generator: &SymbolicGenerator, target: &TypeTarget) -> String {
    let Some(profile) = profile_for_target(generator, target, PROFILE_CONTINUATION_PREDICATES, 0)
    else {
        return empty_profile_text(&target.key);
    };
    let total = profile.total_members;
    let coverage = if total == 0 {
        0
    } else {
        (profile.covered_members * 1000 / total).min(1000)
    };
    let mut gaps = Vec::new();
    for summary in &profile.predicates {
        if summary.coverage_milli >= 1000 {
            continue;
        }
        let mut missing = Vec::new();
        for member in &target.members {
            if !member_has_predicate(generator, member, &summary.predicate) {
                missing.push(member.clone());
                if missing.len() >= 3 {
                    break;
                }
            }
        }
        if missing.is_empty() {
            continue;
        }
        gaps.push((summary, missing));
    }
    let mut output = format!(
        "类型画像「{}」的覆盖缺口（总体：{}/{} 个成员至少有 1 条事实，{}%）：\n",
        target.key,
        profile.covered_members,
        total,
        coverage / 10
    );
    if gaps.is_empty() {
        output.push_str(&format!(
            "  已归纳的 {} 种谓词覆盖全部成员，没有低覆盖缺口。",
            profile.total_predicates
        ));
        return output.trim_end().to_string();
    }
    for (index, (summary, missing)) in gaps.iter().take(8).enumerate() {
        output.push_str(&format!(
            "  {}. {}（{}）：覆盖 {}/{} 个成员（{}%）；例如：{}",
            index + 1,
            summary.label,
            summary.kind.label(),
            summary.member_count,
            total,
            summary.coverage_milli / 10,
            missing.join("、")
        ));
        output.push('\n');
    }
    if gaps.len() > 8 {
        output.push_str(&format!("  （其余 {} 种低覆盖谓词未展开。）\n", gaps.len() - 8));
    }
    output.trim_end().to_string()
}

fn empty_profile_text(etype: &str) -> String {
    format!("类型画像「{}」：暂无显式成员，还没有可归纳的属性。", etype)
}

fn empty_members_text(etype: &str) -> String {
    format!("类型「{}」暂无显式成员。", etype)
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

    fn schema() -> SchemaView {
        let mut schema = SchemaView::new();
        schema.declare_type_alias("组织", "机构");
        schema.declare_type_alias("人", "人物");
        schema.declare_subclass("机构", "实体");
        schema.declare_subclass("人物", "实体");
        schema.declare_subclass("学生", "人物");
        schema.declare_disjoint("人物", "地点");
        schema
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
    fn type_count_questions_report_member_totals() {
        let generator = generator();
        let parsed = parse(&generator, "有多少个人物？").unwrap();
        assert_eq!(
            parsed,
            TypeQuestion::Count {
                etype: "人物".to_string()
            }
        );
        let counted = answer(&generator, "有多少个人物？").unwrap();
        assert!(
            counted.text.contains("共有 2 个已知成员"),
            "{}",
            counted.text
        );
        assert_eq!(counted.focus.as_deref(), Some("人物"));

        assert_eq!(
            parse(&generator, "人物有多少个").unwrap(),
            TypeQuestion::Count {
                etype: "人物".to_string()
            }
        );
        let schema = schema();
        let empty =
            answer_with_schema(&generator, Some(&schema), "学生有多少个").unwrap_or_else(|| {
                panic!("已声明的空类型也应诚实回答");
            });
        assert!(empty.text.contains("0") || empty.text.contains("没有"), "{}", empty.text);
    }

    #[test]
    fn single_char_type_alias_counts_and_attribute_counts_are_not_hijacked() {
        let generator = generator();
        let schema = schema();
        let parsed = super::parse_details(&generator, Some(&schema), "有多少个人")
            .map(|(question, _)| question)
            .unwrap();
        assert_eq!(
            parsed,
            TypeQuestion::Count {
                etype: "人物".to_string()
            }
        );
        let answer =
            answer_with_schema(&generator, Some(&schema), "有多少个人").expect("别名计数应可回答");
        assert!(answer.text.contains("共有 2 个已知成员"), "{}", answer.text);

        assert!(
            super::parse_details(&generator, Some(&schema), "人物有多少个属性").is_none(),
            "问属性个数不能被劫持成成员计数"
        );

        let profile = super::parse_details(&generator, Some(&schema), "人一般有哪些属性")
            .map(|(question, _)| question)
            .unwrap();
        assert_eq!(
            profile,
            TypeQuestion::Profile {
                etype: "人物".to_string()
            }
        );
        let answer = answer_with_schema(&generator, Some(&schema), "人一般有哪些属性")
            .expect("单字别名也应能回答类型画像");
        assert!(answer.text.contains("类型画像「人物」"), "{}", answer.text);
        assert!(answer.text.contains("年龄"), "{}", answer.text);
    }

    #[test]
    fn count_focus_pages_members_on_followup() {
        let generator = generator();
        let mut memory = TypeDialogueMemory::new();
        answer_in_context(&generator, None, &mut memory, "有多少个人物").expect("计数应可回答");
        let members = answer_in_context(&generator, None, &mut memory, "还有呢")
            .expect("计数后追问应列出成员");
        assert!(members.text.contains("张三"), "{}", members.text);
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

    #[test]
    fn alias_and_subclass_share_one_type_view() {
        let generator = generator();
        let schema = schema();
        let parsed = super::parse_details(&generator, Some(&schema), "机构有哪些成员")
            .map(|(question, _)| question)
            .unwrap();
        assert_eq!(
            parsed,
            TypeQuestion::Members {
                etype: "机构".to_string()
            }
        );
        let answer = answer_with_schema(&generator, Some(&schema), "机构一般有哪些属性").unwrap();
        assert!(answer.text.contains("类型画像「机构」"));
        assert!(answer.text.contains("城市"));

        let mut generator = generator;
        generator.observe_entity("王五", "学生", &[("年龄".to_string(), "20".to_string())]);
        let members = answer_with_schema(&generator, Some(&schema), "人物有哪些成员").unwrap();
        assert!(members.text.contains("王五"), "{}", members.text);
        let students = answer_with_schema(&generator, Some(&schema), "学生有哪些成员").unwrap();
        assert!(students.text.contains("王五"));
        assert!(!students.text.contains("张三"), "{}", students.text);
    }

    #[test]
    fn followup_switch_and_member_paging_use_focus() {
        let mut generator = SymbolicGenerator::new(6, 0.3);
        for index in 0..25 {
            generator.observe_entity(
                &format!("人物{:02}", index),
                "人物",
                &[("年龄".to_string(), index.to_string())],
            );
        }
        generator.observe_entity(
            "甲公司",
            "组织",
            &[("城市".to_string(), "北京".to_string())],
        );
        let schema = schema();
        let mut memory = TypeDialogueMemory::new();

        let first = answer_in_context(&generator, Some(&schema), &mut memory, "人物有哪些成员")
            .expect("首屏应可回答");
        assert!(first.text.contains("人物00"));
        assert!(!first.text.contains("人物20"), "{}", first.text);

        let second =
            answer_in_context(&generator, Some(&schema), &mut memory, "还有呢").expect("应翻页");
        assert!(second.text.contains("人物20"), "{}", second.text);
        assert!(!second.text.contains("人物00"), "{}", second.text);

        let switched = answer_in_context(&generator, Some(&schema), &mut memory, "那机构呢")
            .expect("应切换类型焦点");
        assert!(switched.text.contains("类型「机构」的成员"), "{}", switched.text);
        assert_eq!(switched.focus.as_deref(), Some("机构"));

        let profile = answer_in_context(&generator, Some(&schema), &mut memory, "机构有哪些属性")
            .expect("应可回答机构画像");
        assert!(profile.text.contains("类型画像「机构」"), "{}", profile.text);
        let switched = answer_in_context(&generator, Some(&schema), &mut memory, "那人物呢")
            .expect("应沿画像意图切换焦点");
        assert!(switched.text.contains("类型画像「人物」"), "{}", switched.text);
        assert_eq!(switched.focus.as_deref(), Some("人物"));
    }

    #[test]
    fn focus_expires_after_three_unrelated_turns() {
        let generator = generator();
        let schema = schema();
        let mut memory = TypeDialogueMemory::new();
        answer_in_context(&generator, Some(&schema), &mut memory, "人物一般有哪些属性")
            .expect("画像应可回答");
        for _ in 0..TYPE_FOCUS_TTL {
            memory.note_non_type_turn();
        }
        assert!(memory.is_empty());
        assert!(
            answer_in_context(&generator, Some(&schema), &mut memory, "还有呢").is_none(),
            "过期焦点不应继续接管追问"
        );
    }

    #[test]
    fn exhausted_members_are_not_repeated() {
        let generator = generator();
        let mut memory = TypeDialogueMemory::new();
        answer_in_context(&generator, None, &mut memory, "人物有哪些成员").expect("应可回答");
        let answer =
            answer_in_context(&generator, None, &mut memory, "还有呢").expect("应说明已列完");
        assert!(answer.text.contains("已全部列出"), "{}", answer.text);
        assert!(!answer.text.contains("张三"));
    }

    #[test]
    fn declared_but_empty_type_still_answers_honestly() {
        let generator = generator();
        let schema = schema();
        let answer = answer_with_schema(&generator, Some(&schema), "学生一般有哪些属性")
            .expect("已声明的类型应可诚实回答");
        assert!(answer.text.contains("暂无显式成员"), "{}", answer.text);
    }

    #[test]
    fn comparison_reports_schema_subclass_and_disjoint_evidence() {
        let mut generator = generator();
        generator.observe_entity("北京", "地点", &[]);
        let schema = schema();
        let disjoint = answer_with_schema(
            &generator,
            Some(&schema),
            "比较人物和地点的区别",
        )
        .expect("对比应可回答");
        assert!(disjoint.text.contains("⊥"), "{}", disjoint.text);
        assert!(disjoint.text.contains("互斥"), "{}", disjoint.text);

        let subclass = answer_with_schema(
            &generator,
            Some(&schema),
            "比较学生和人物的区别",
        )
        .expect("子类对比应可回答");
        assert!(subclass.text.contains("学生 ⊑ 人物"), "{}", subclass.text);
    }

    #[test]
    fn gap_question_lists_low_coverage_slots_and_missing_members() {
        let generator = generator();
        let mut memory = TypeDialogueMemory::new();
        answer_in_context(&generator, None, &mut memory, "人物一般有哪些属性")
            .expect("画像应可回答");
        let gap = answer_in_context(&generator, None, &mut memory, "还缺什么")
            .expect("缺口问题应可回答");
        assert!(gap.text.contains("覆盖缺口"), "{}", gap.text);
        assert!(gap.text.contains("师傅"), "{}", gap.text);
        assert!(gap.text.contains("李四"), "{}", gap.text);
        assert_eq!(gap.focus.as_deref(), Some("人物"));

        let explicit = answer_with_schema(&generator, None, "人物哪些属性缺失")
            .expect("无会话焦点时也可按显式类型回答");
        assert!(explicit.text.contains("覆盖缺口"), "{}", explicit.text);

        assert!(
            answer_in_context(&generator, None, &mut memory, "张三缺什么").is_none(),
            "实体问句不应被类型缺口劫持"
        );
    }

    #[test]
    fn exact_type_surface_beats_localized_label() {
        let mut generator = SymbolicGenerator::new(6, 0.3);
        generator.observe_entity("张三", "Person", &[]);
        generator.observe_entity("李四", "人物", &[]);
        let parsed = parse(&generator, "人物有哪些成员").unwrap();
        assert_eq!(
            parsed,
            TypeQuestion::Members {
                etype: "人物".to_string()
            }
        );
        let answer = answer(&generator, "人物有哪些成员").unwrap();
        assert!(answer.text.contains("李四"), "{}", answer.text);
        assert!(!answer.text.contains("张三"), "{}", answer.text);
    }

    #[test]
    fn dialogue_manager_persists_type_focus_across_snapshot() {
        use crate::dialogue_manager::DialogueManager;
        use crate::prolog_engine::PrologFact;
        use crate::symbolic_reasoner::SymbolicReasoner;
        use std::collections::HashMap;

        let schema = schema();
        let mut reasoner = SymbolicReasoner::new(crate::ReasoningConfig::default(), HashMap::new());
        reasoner.configure_schema_view(&schema);
        let mut facts = Vec::new();
        for index in 0..25 {
            facts.push(PrologFact::Entity {
                name: format!("人物{:02}", index),
                entity_type: "人物".to_string(),
                properties: HashMap::from([("年龄".to_string(), index.to_string())]),
            });
        }
        facts.push(PrologFact::Entity {
            name: "甲公司".to_string(),
            entity_type: "组织".to_string(),
            properties: HashMap::from([("城市".to_string(), "北京".to_string())]),
        });
        reasoner.observe_facts(&facts);

        let direct = reasoner
            .answer("机构一般有哪些属性")
            .expect("推理器直答也应复用模式视图");
        assert!(direct.contains("类型画像「机构」"), "{}", direct);
        assert!(direct.contains("城市"), "{}", direct);

        let mut manager = DialogueManager::new();
        let first = manager
            .plan_pure_dialogue("人物有哪些成员", &reasoner)
            .expect("对话管理器应接管类型问答");
        assert!(first.text.contains("人物00"), "{}", first.text);
        assert_eq!(manager.turn_count(), 1);

        let second = manager
            .plan_pure_dialogue("还有呢", &reasoner)
            .expect("追问应沿成员游标翻页");
        assert!(second.text.contains("人物20"), "{}", second.text);
        assert!(!second.text.contains("人物00"), "{}", second.text);

        let switched = manager
            .plan_pure_dialogue("那机构呢", &reasoner)
            .expect("追问应切换类型焦点");
        assert!(switched.text.contains("类型「机构」的成员"), "{}", switched.text);

        let snapshot = manager.snapshot();
        assert!(!snapshot.type_memory.is_empty());
        let mut restored = DialogueManager::new();
        restored.restore(&snapshot);
        let continued = restored
            .plan_pure_dialogue("还有呢", &reasoner)
            .expect("恢复后应继续使用类型焦点");
        assert!(continued.text.contains("已全部列出"), "{}", continued.text);
    }
    #[test]
    fn entity_type_question_reports_declared_type_and_supertypes() {
        let generator = generator();
        let schema = schema();
        for question in [
            "张三是什么类型？",
            "张三属于哪一类？",
            "张三的类型是什么",
            "「张三」是哪类实体",
            "请问张三是什么类型的人物呢",
        ] {
            assert_eq!(
                parse(&generator, question),
                Some(TypeQuestion::EntityType {
                    entity: "张三".to_string()
                }),
                "{}",
                question
            );
        }
        let answer = answer_with_schema(&generator, Some(&schema), "张三是什么类型？")
            .expect("实体类型应可回答");
        assert!(answer.text.contains("「张三」的类型是「人物」"), "{}", answer.text);
        assert!(answer.text.contains("上位类型：实体"), "{}", answer.text);
        assert_eq!(answer.focus.as_deref(), Some("人物"));
    }

    #[test]
    fn entity_type_question_does_not_steal_attribute_chain_questions() {
        let generator = generator();
        assert_eq!(parse(&generator, "张三的师傅是什么类型？"), None);
        assert_eq!(parse(&generator, "张三的年龄是什么？"), None);
    }

    #[test]
    fn entity_type_question_reports_missing_type_honestly() {
        let mut generator = generator();
        generator.observe_entity("王五", "", &[]);
        let answer = answer(&generator, "王五是什么类型？").expect("问句本身合法");
        assert!(answer.text.contains("「王五」尚未登记显式类型"), "{}", answer.text);
        assert_eq!(answer.focus, None);
    }

}

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
    Compare { left: String, right: String },
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
    if let Some((parsed, _kind)) = parse_details(generator, schema, question) {
        return answer_parsed(generator, schema, memory, parsed);
    }

    let normalized = normalize(question);
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
        TypeQuestion::Compare { left, right } => {
            let left_target = resolve_target(generator, schema, &left)?;
            let right_target = resolve_target(generator, schema, &right)?;
            let mut focus = TypeFocus::new(left_target.key.clone(), TypeQuestionKind::Compare);
            focus.peer = Some(right_target.key.clone());
            let text = render_comparison(generator, &left_target, &right_target);
            let focus_key = left_target.key.clone();
            memory.set_focus(focus);
            Some(TypeAnswer {
                text,
                focus: Some(focus_key),
            })
        }
    }
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
            let text = render_comparison(generator, &left, &right);
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

    if question.contains("类型画像") || question.contains("类型概览") {
        if let Some(etype) = profile_mention(generator, schema, &question) {
            return Some((TypeQuestion::Profile { etype }, TypeQuestionKind::Profile));
        }
    }

    let mentions = known_type_mentions(generator, schema, &question);
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
        return Some((
            TypeQuestion::Compare { left, right },
            TypeQuestionKind::Compare,
        ));
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
    let mut surfaces: BTreeMap<String, String> = BTreeMap::new();
    let register = |surface: &str, etype: &str, surfaces: &mut BTreeMap<String, String>| {
        let surface = surface.trim();
        if surface.chars().count() < 2 {
            return;
        }
        if let Some(target) = resolve_target(generator, schema, etype) {
            surfaces
                .entry(surface.to_string())
                .or_insert(target.key);
        }
    };
    for etype in generator.all_types() {
        register(&etype, &etype, &mut surfaces);
        let label = localized_type_label(&etype);
        if label != etype {
            register(label, &etype, &mut surfaces);
        }
    }
    if let Some(schema) = schema {
        for (alias, _canonical) in schema.type_alias_pairs() {
            register(&alias, &alias, &mut surfaces);
        }
        for class in schema.declared_classes() {
            register(&class, &class, &mut surfaces);
            let label = localized_type_label(&class);
            if label != class {
                register(label, &class, &mut surfaces);
            }
        }
    }

    let mut raw = Vec::new();
    for (surface, etype) in surfaces {
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
    left: &TypeTarget,
    right: &TypeTarget,
) -> String {
    if let Some(comparison) = compare_targets(generator, left, right) {
        return comparison.render();
    }
    format!(
        "类型对比「{}」与「{}」：\n  成员：{} {} 个；{} {} 个。\n  （至少一方暂无显式成员，暂无可比较的谓词。）",
        left.key,
        right.key,
        left.key,
        left.members.len(),
        right.key,
        right.members.len()
    )
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
}

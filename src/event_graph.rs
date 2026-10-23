//! EPG：事件—投影图（Event–Projection Graph）。
//!
//! 现有的三元组抽取保存「谁和谁有什么关系」，却把「发生了什么事」拆散在
//! 各条边里。本模块把事件物化为独立节点，原三元组只作为事件的一个投影
//! 保留，从而让「何时 / 何地 / 为什么 / 后来 / 还有谁」可以直接查图。
//!
//! 纪律与仓库其他模块一致：纯符号、零训练、同输入同字节。
//!   - 事件身份由来源文本、句序、触发词、角色、时间与极性共同决定；
//!     不凭语义相似自动合并，宁可分开也不误并；
//!   - 时间未知不硬排序；先后只来自显式时间比较或显式的「后来/随后/
//!     然后/接着/最后」等话语标记；
//!   - 因果只来自「因为/由于」与「所以/因此/于是」等显式连接词，
//!     没有链接就明确弃权，绝不按词序猜测原因；
//!   - 每个回答都带事件 id 与证据句，生成文本只是这些证据的线性化。

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use serde::{Deserialize, Serialize};

use crate::context_image::fnv1a_hex;
use crate::prolog_engine::escape_atom;
use crate::relation_frame::{
    parse_temporal, EvidenceSpan, FrameExtractor, Modality, Polarity, RelationFrame, TemporalScope,
};

/// 事件图方言标识（JSON / CLI 输出用）。
pub const EVENT_DIALECT: &str = "knowprolog-epg/1";
/// 默认最多保留的事件数。
pub const DEFAULT_MAX_EVENTS: usize = 256;
/// 默认最多保留的链接数。
pub const DEFAULT_MAX_LINKS: usize = 2048;
/// 事件焦点栈的最大深度。
pub const DEFAULT_FOCUS_DEPTH: usize = 16;
/// 事件图默认落入的命名图前缀（默认图保持只读，不被事件层改写）。
pub const EVENT_GRAPH_NS: &str = "http://example.org/kg/eventGraph/";

const EVENT_PROLOG_RULES: &str = r#"event_participant(E, P) :- event_role(E, agent, P).
event_participant(E, P) :- event_role(E, patient, P).
event_participant(E, P) :- event_role(E, recipient, P).
event_place(E, P) :- event_role(E, location, P).
event_before(A, B) :- event_link(A, before, B).
event_before(A, B) :- event_link(A, subevent, B).
event_before(A, C) :- event_before(A, B), event_before(B, C).
event_after(A, B) :- event_before(B, A).
event_causes(A, B) :- event_link(A, causes, B).
event_at(E, T) :- event_time(E, T).
"#;

const CAUSE_MARKERS: &[&str] = &["因为", "由于", "鉴于"];
const RESULT_MARKERS: &[&str] = &["所以", "因此", "于是", "因而", "故而", "结果"];
const ORDER_MARKERS: &[&str] = &[
    "后来",
    "随后",
    "然后",
    "接着",
    "紧接着",
    "最后",
    "此后",
    "之后",
    "接下来",
];
const LEADING_CONNECTORS: &[&str] = &[
    "因为",
    "由于",
    "鉴于",
    "所以",
    "因此",
    "于是",
    "因而",
    "故而",
    "结果",
    "后来",
    "随后",
    "然后",
    "接着",
    "紧接着",
    "最后",
    "此后",
    "之后",
    "接下来",
    "同时",
    "另外",
    "此外",
];

/// 事件角色。MVP 只区分最容易从现有帧稳定得到的四类。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventRole {
    Agent,
    Patient,
    Recipient,
    Location,
}

impl EventRole {
    pub fn label(self) -> &'static str {
        match self {
            EventRole::Agent => "施事",
            EventRole::Patient => "受事",
            EventRole::Recipient => "接受者",
            EventRole::Location => "地点",
        }
    }

    pub fn tag(self) -> &'static str {
        match self {
            EventRole::Agent => "agent",
            EventRole::Patient => "patient",
            EventRole::Recipient => "recipient",
            EventRole::Location => "location",
        }
    }
}

/// 事件中的一个角色绑定。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct EventArgument {
    pub role: EventRole,
    pub value: String,
}

/// 事件在源文本中的证据。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventEvidence {
    pub sentence_index: usize,
    pub sentence: String,
    /// 证据句在源文本中的起点（字节）。
    pub sentence_start: usize,
    /// 事件触发区域在源文本中的起点（字节）。
    pub start: usize,
    /// 事件触发区域在源文本中的终点（字节）。
    pub end: usize,
    /// 主语在源文本中的起点；用于在句内定位事件先后。
    pub anchor: usize,
}

/// 原三元组在事件上的投影。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventProjection {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub frame_id: String,
}

/// 一个物化事件。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EventNode {
    pub id: String,
    /// 规范事件类型（如「出生」「成立」），缺省等于触发词。
    pub event_type: String,
    /// 原文触发词。
    pub trigger: String,
    pub arguments: Vec<EventArgument>,
    pub polarity: Polarity,
    pub modality: Modality,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub time: Option<TemporalScope>,
    pub projection: EventProjection,
    pub evidence: EventEvidence,
    pub confidence: f64,
}

impl EventNode {
    pub fn argument(&self, role: EventRole) -> Option<&str> {
        self.arguments
            .iter()
            .find(|argument| argument.role == role)
            .map(|argument| argument.value.as_str())
    }

    pub fn arguments_of(&self, role: EventRole) -> Vec<&str> {
        self.arguments
            .iter()
            .filter(|argument| argument.role == role)
            .map(|argument| argument.value.as_str())
            .collect()
    }

    pub fn time_text(&self) -> Option<String> {
        let time = self.time.as_ref()?;
        Some(time.normalized.clone().unwrap_or_else(|| time.raw.clone()))
    }

    pub fn render_text(&self) -> String {
        let mut parts = Vec::new();
        if let Some(agent) = self.argument(EventRole::Agent) {
            parts.push(agent.to_string());
        }
        parts.push(self.trigger.clone());
        if let Some(patient) = self.argument(EventRole::Patient) {
            parts.push(patient.to_string());
        }
        if let Some(recipient) = self.argument(EventRole::Recipient) {
            parts.push(format!("给{}", recipient));
        }
        if let Some(location) = self.argument(EventRole::Location) {
            parts.push(format!("在{}", location));
        }
        let mut text = parts.join("");
        if let Some(time) = &self.time {
            text = format!("{}，{}", time.render_text(), text);
        }
        if self.polarity == Polarity::Negative {
            text.push_str("（否定）");
        }
        if self.modality != Modality::Asserted {
            text.push_str(&format!("（{}）", self.modality.label()));
        }
        text
    }
}

/// 事件之间的显式链接。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventLinkKind {
    Before,
    After,
    Simultaneous,
    Causes,
    Subevent,
}

impl EventLinkKind {
    pub fn label(self) -> &'static str {
        match self {
            EventLinkKind::Before => "先于",
            EventLinkKind::After => "后于",
            EventLinkKind::Simultaneous => "同时",
            EventLinkKind::Causes => "导致",
            EventLinkKind::Subevent => "子事件",
        }
    }

    pub fn tag(self) -> &'static str {
        match self {
            EventLinkKind::Before => "before",
            EventLinkKind::After => "after",
            EventLinkKind::Simultaneous => "simultaneous",
            EventLinkKind::Causes => "causes",
            EventLinkKind::Subevent => "subevent",
        }
    }
}

/// 一条事件链接。`Causes` 的方向是「原因 → 结果」。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EventLink {
    pub from: String,
    pub to: String,
    pub kind: EventLinkKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evidence: Option<String>,
    pub confidence: f64,
}

/// 抽取与建图统计：让「抽到多少 / 丢了多少」可核对。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventBuildStats {
    pub sentences: usize,
    pub frames: usize,
    pub fallback_clauses: usize,
    pub events: usize,
    pub links: usize,
    pub truncated_events: usize,
    pub truncated_links: usize,
}

/// 候选共指：只提出、不自动合并。调用方需自行确认。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EventCoreference {
    pub left: String,
    pub right: String,
    pub score: f64,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventGraph {
    pub dialect: String,
    pub source_fingerprint: String,
    pub events: Vec<EventNode>,
    pub links: Vec<EventLink>,
    pub stats: EventBuildStats,
}

impl EventGraph {
    pub fn build(text: &str) -> Self {
        Self::build_with_limits(text, DEFAULT_MAX_EVENTS)
    }

    pub fn build_with_limits(text: &str, max_events: usize) -> Self {
        let ranges = sentence_ranges(text);
        let extractor = FrameExtractor::new();
        let (mut frames, _) = extractor.extract_with_stats(text);
        let (fallback_frames, fallback_clauses) =
            extract_fallback_clauses(text, &ranges, &extractor);
        frames.extend(fallback_frames);
        frames.extend(extract_trigger_frames(text, &ranges));
        Self::from_frames_internal(text, &ranges, frames, fallback_clauses, max_events)
    }

    /// 从已抽取的关系帧构建事件图（供调用方复用已有抽取结果）。
    pub fn from_frames(text: &str, frames: Vec<RelationFrame>) -> Self {
        let ranges = sentence_ranges(text);
        Self::from_frames_internal(text, &ranges, frames, 0, DEFAULT_MAX_EVENTS)
    }

    fn from_frames_internal(
        text: &str,
        ranges: &[(usize, usize)],
        frames: Vec<RelationFrame>,
        fallback_clauses: usize,
        max_events: usize,
    ) -> Self {
        let source_fingerprint = fnv1a_hex(text.as_bytes());
        let mut stats = EventBuildStats {
            sentences: ranges.len(),
            frames: frames.len(),
            fallback_clauses,
            ..EventBuildStats::default()
        };
        let limit = if max_events == 0 {
            usize::MAX
        } else {
            max_events
        };

        let mut events: Vec<EventNode> = Vec::new();
        let mut seen_ids = BTreeSet::new();
        for frame in &frames {
            if frame.evidence.sentence_index >= ranges.len() {
                continue;
            }
            let (range_start, range_end) = ranges[frame.evidence.sentence_index];
            let raw_sentence = &text[range_start..range_end];
            let sentence = raw_sentence.trim();
            if sentence.is_empty() {
                continue;
            }
            let sentence_start =
                range_start + (raw_sentence.len() - raw_sentence.trim_start().len());
            let (subject, subject_offset) = clean_party(&frame.subject);
            let (object, _) = clean_party(&frame.object);
            if subject.is_empty() || object.is_empty() {
                continue;
            }
            let anchor = frame
                .evidence
                .subject_span
                .map(|(start, _)| frame.evidence.start + start)
                .unwrap_or(frame.evidence.start)
                .saturating_add(subject_offset)
                .max(sentence_start);
            let role = object_role(frame);
            let mut arguments = vec![
                EventArgument {
                    role: EventRole::Agent,
                    value: subject.to_string(),
                },
                EventArgument {
                    role,
                    value: object.to_string(),
                },
            ];
            arguments.retain(|argument| !argument.value.trim().is_empty());
            arguments.sort();
            arguments.dedup();

            let event_type = canonical_event_type(&frame.predicate);
            let material = format!(
                "{}|{}|{}|{}|{}|{}|{}",
                source_fingerprint,
                frame.evidence.sentence_index,
                anchor,
                event_type,
                render_arguments_key(&arguments),
                frame
                    .temporal
                    .as_ref()
                    .map(time_key)
                    .unwrap_or_else(|| "unknown".to_string()),
                polarity_key(frame.polarity),
            );
            let id = format!("evt:{}", fnv1a_hex(material.as_bytes()));
            if !seen_ids.insert(id.clone()) {
                continue;
            }
            events.push(EventNode {
                id,
                event_type,
                trigger: frame.predicate.clone(),
                arguments,
                polarity: frame.polarity,
                modality: frame.modality,
                time: frame.temporal.clone(),
                projection: EventProjection {
                    subject: subject.to_string(),
                    predicate: frame.predicate.clone(),
                    object: object.to_string(),
                    frame_id: frame.id.clone(),
                },
                evidence: EventEvidence {
                    sentence_index: frame.evidence.sentence_index,
                    sentence: sentence.to_string(),
                    sentence_start,
                    start: frame.evidence.start,
                    end: frame.evidence.end,
                    anchor,
                },
                confidence: frame.confidence.clamp(0.0, 1.0),
            });
        }

        events.sort_by(|left, right| {
            (left.evidence.sentence_index, left.evidence.anchor, &left.id).cmp(&(
                right.evidence.sentence_index,
                right.evidence.anchor,
                &right.id,
            ))
        });
        if events.len() > limit {
            stats.truncated_events = events.len() - limit;
            events.truncate(limit);
        }
        stats.events = events.len();

        let mut links = build_links(&events);
        links.sort_by(|left, right| {
            (&left.from, left.kind, &left.to).cmp(&(&right.from, right.kind, &right.to))
        });
        links.dedup_by(|left, right| {
            left.from == right.from && left.to == right.to && left.kind == right.kind
        });
        if links.len() > DEFAULT_MAX_LINKS {
            stats.truncated_links = links.len() - DEFAULT_MAX_LINKS;
            links.truncate(DEFAULT_MAX_LINKS);
        }
        stats.links = links.len();

        Self {
            dialect: EVENT_DIALECT.to_string(),
            source_fingerprint,
            events,
            links,
            stats,
        }
    }

    pub fn event(&self, id: &str) -> Option<&EventNode> {
        self.events.iter().find(|event| event.id == id)
    }

    pub fn events_for(&self, entity: &str) -> Vec<&EventNode> {
        self.events
            .iter()
            .filter(|event| {
                event
                    .arguments
                    .iter()
                    .any(|argument| argument.value == entity)
            })
            .collect()
    }

    pub fn links_from(&self, id: &str) -> Vec<&EventLink> {
        self.links.iter().filter(|link| link.from == id).collect()
    }

    pub fn links_to(&self, id: &str) -> Vec<&EventLink> {
        self.links.iter().filter(|link| link.to == id).collect()
    }

    /// 按显式时间排序；时间未知的事件保持句序并排在已知时间之后。
    pub fn ordered_events(&self) -> Vec<&EventNode> {
        let mut ordered: Vec<&EventNode> = self.events.iter().collect();
        ordered.sort_by(|left, right| {
            (
                event_start_ordinal(left).unwrap_or(i64::MAX),
                left.evidence.sentence_index,
                left.evidence.anchor,
                &left.id,
            )
                .cmp(&(
                    event_start_ordinal(right).unwrap_or(i64::MAX),
                    right.evidence.sentence_index,
                    right.evidence.anchor,
                    &right.id,
                ))
        });
        ordered
    }

    pub fn ask(&self, question: &str, focus: Option<&str>) -> EventAnswer {
        let parsed = EventQuestion::parse(question);
        let Some(resolution) = self.resolve_focus(&parsed, focus) else {
            return EventAnswer::abstained(
                parsed.kind,
                None,
                "没有找到可定位的事件；请写明实体或先建立事件焦点。".to_string(),
            );
        };
        let primary = &self.events[resolution.indices[0]];
        let focused: Vec<&EventNode> = resolution
            .indices
            .iter()
            .map(|index| &self.events[*index])
            .collect();
        match parsed.kind {
            EventQuestionKind::Who => self.answer_who(primary, &focused, resolution.implicit),
            EventQuestionKind::Participants => {
                self.answer_participants(primary, &focused, resolution.implicit)
            }
            EventQuestionKind::When => self.answer_when(primary, &focused, resolution.implicit),
            EventQuestionKind::Where => self.answer_where(primary, &focused, resolution.implicit),
            EventQuestionKind::Why => self.answer_why(primary, resolution.implicit),
            EventQuestionKind::WhatNext => self.answer_next(primary, resolution.implicit),
            EventQuestionKind::Did => self.answer_did(primary, &focused, resolution.implicit),
        }
    }

    fn resolve_focus(
        &self,
        question: &EventQuestion,
        focus: Option<&str>,
    ) -> Option<FocusResolution> {
        let mut scored: Vec<(u32, usize)> = Vec::new();
        for (index, event) in self.events.iter().enumerate() {
            let mut score = 0u32;
            for argument in &event.arguments {
                if argument.value.chars().count() >= 2
                    && !is_pronoun(&argument.value)
                    && question.text.contains(&argument.value)
                {
                    score += 5;
                }
            }
            if question.text.contains(&event.trigger) || question.text.contains(&event.event_type) {
                score += 2;
            }
            if score > 0 {
                scored.push((score, index));
            }
        }
        scored.sort_by(|left, right| right.0.cmp(&left.0).then_with(|| left.1.cmp(&right.1)));
        if !scored.is_empty() {
            let best = scored[0].0;
            return Some(FocusResolution {
                indices: scored
                    .into_iter()
                    .filter(|(score, _)| *score == best)
                    .map(|(_, index)| index)
                    .collect(),
                implicit: false,
            });
        }
        if let Some(id) = focus {
            if let Some(index) = self.events.iter().position(|event| event.id == id) {
                return Some(FocusResolution {
                    indices: vec![index],
                    implicit: false,
                });
            }
        }
        if question.anaphoric && question.kind == EventQuestionKind::WhatNext {
            if let Some(last) = self.ordered_events().last() {
                if let Some(index) = self.events.iter().position(|event| event.id == last.id) {
                    return Some(FocusResolution {
                        indices: vec![index],
                        implicit: true,
                    });
                }
            }
        }
        None
    }

    fn answer_who(
        &self,
        primary: &EventNode,
        focused: &[&EventNode],
        implicit: bool,
    ) -> EventAnswer {
        let mut values = BTreeSet::new();
        for event in focused {
            for value in event.arguments_of(EventRole::Agent) {
                values.insert(value.to_string());
            }
        }
        if values.is_empty() {
            for event in focused {
                for argument in &event.arguments {
                    if argument.role != EventRole::Location {
                        values.insert(argument.value.clone());
                    }
                }
            }
        }
        if values.is_empty() {
            return EventAnswer::abstained(
                EventQuestionKind::Who,
                Some(primary.id.clone()),
                "事件存在，但没有标注任何参与者。".to_string(),
            );
        }
        let text = format!("{}", values.into_iter().collect::<Vec<_>>().join("、"));
        EventAnswer::answered(EventQuestionKind::Who, primary, focused, implicit, text)
    }

    fn answer_participants(
        &self,
        primary: &EventNode,
        focused: &[&EventNode],
        implicit: bool,
    ) -> EventAnswer {
        let roles = [
            EventRole::Agent,
            EventRole::Patient,
            EventRole::Recipient,
            EventRole::Location,
        ];
        let mut parts = Vec::new();
        for role in roles {
            let mut values = BTreeSet::new();
            for event in focused {
                for value in event.arguments_of(role) {
                    values.insert(value.to_string());
                }
            }
            if !values.is_empty() {
                parts.push(format!(
                    "{}：{}",
                    role.label(),
                    values.into_iter().collect::<Vec<_>>().join("、")
                ));
            }
        }
        if parts.is_empty() {
            return EventAnswer::abstained(
                EventQuestionKind::Participants,
                Some(primary.id.clone()),
                "事件存在，但没有可枚举的角色。".to_string(),
            );
        }
        EventAnswer::answered(
            EventQuestionKind::Participants,
            primary,
            focused,
            implicit,
            parts.join("；"),
        )
    }

    fn answer_when(
        &self,
        primary: &EventNode,
        focused: &[&EventNode],
        implicit: bool,
    ) -> EventAnswer {
        let mut values = BTreeSet::new();
        for event in focused {
            if let Some(time) = event.time_text() {
                values.insert(time);
            }
        }
        if values.is_empty() {
            return EventAnswer::abstained(
                EventQuestionKind::When,
                Some(primary.id.clone()),
                "事件没有标注显式时间；不猜系统时钟或上下文时间。".to_string(),
            );
        }
        EventAnswer::answered(
            EventQuestionKind::When,
            primary,
            focused,
            implicit,
            values.into_iter().collect::<Vec<_>>().join("、"),
        )
    }

    fn answer_where(
        &self,
        primary: &EventNode,
        focused: &[&EventNode],
        implicit: bool,
    ) -> EventAnswer {
        let mut values = BTreeSet::new();
        for event in focused {
            for value in event.arguments_of(EventRole::Location) {
                values.insert(value.to_string());
            }
        }
        if values.is_empty() {
            return EventAnswer::abstained(
                EventQuestionKind::Where,
                Some(primary.id.clone()),
                "事件没有标注地点；不根据谓词猜测场所。".to_string(),
            );
        }
        EventAnswer::answered(
            EventQuestionKind::Where,
            primary,
            focused,
            implicit,
            values.into_iter().collect::<Vec<_>>().join("、"),
        )
    }

    fn answer_why(&self, primary: &EventNode, implicit: bool) -> EventAnswer {
        let causes: Vec<&EventNode> = self
            .links
            .iter()
            .filter(|link| link.kind == EventLinkKind::Causes && link.to == primary.id)
            .filter_map(|link| self.event(&link.from))
            .collect();
        if causes.is_empty() {
            return EventAnswer::abstained(
                EventQuestionKind::Why,
                Some(primary.id.clone()),
                "没有显式因果链接；不按词序猜测原因。".to_string(),
            );
        }
        let mut items: Vec<EventAnswerItem> = causes
            .iter()
            .map(|event| EventAnswerItem {
                relation: "原因".to_string(),
                event: (*event).clone(),
            })
            .collect();
        items.push(EventAnswerItem {
            relation: "结果".to_string(),
            event: primary.clone(),
        });
        EventAnswer {
            kind: EventQuestionKind::Why,
            status: EventAnswerStatus::Answered,
            focus: Some(primary.id.clone()),
            implicit_focus: implicit,
            answer: causes
                .iter()
                .map(|event| event.render_text())
                .collect::<Vec<_>>()
                .join("；"),
            events: items,
            reason: None,
        }
    }

    fn answer_next(&self, primary: &EventNode, implicit: bool) -> EventAnswer {
        let next: Vec<&EventNode> = self
            .links
            .iter()
            .filter(|link| {
                (link.from == primary.id
                    && matches!(link.kind, EventLinkKind::Before | EventLinkKind::Subevent))
                    || (link.to == primary.id && link.kind == EventLinkKind::After)
            })
            .filter_map(|link| {
                if link.from == primary.id {
                    self.event(&link.to)
                } else {
                    self.event(&link.from)
                }
            })
            .collect();
        if next.is_empty() {
            return EventAnswer::abstained(
                EventQuestionKind::WhatNext,
                Some(primary.id.clone()),
                "没有显式先后或时间链接，无法判断后来发生了什么。".to_string(),
            );
        }
        let items: Vec<EventAnswerItem> = next
            .iter()
            .map(|event| EventAnswerItem {
                relation: "后续".to_string(),
                event: (*event).clone(),
            })
            .collect();
        EventAnswer {
            kind: EventQuestionKind::WhatNext,
            status: EventAnswerStatus::Answered,
            focus: Some(primary.id.clone()),
            implicit_focus: implicit,
            answer: next
                .iter()
                .map(|event| event.render_text())
                .collect::<Vec<_>>()
                .join("；"),
            events: items,
            reason: None,
        }
    }

    fn answer_did(
        &self,
        primary: &EventNode,
        focused: &[&EventNode],
        implicit: bool,
    ) -> EventAnswer {
        let affirmative = primary.polarity == Polarity::Affirmative;
        let text = if affirmative {
            format!("是：{}", primary.render_text())
        } else {
            format!("没有发生：{}", primary.render_text())
        };
        EventAnswer::answered(EventQuestionKind::Did, primary, focused, implicit, text)
    }

    pub fn events_by_trigger(&self, trigger: &str) -> Vec<&EventNode> {
        self.events
            .iter()
            .filter(|event| event.trigger == trigger || event.event_type == trigger)
            .collect()
    }

    pub fn events_for_time(&self, time: &str) -> Vec<&EventNode> {
        self.events
            .iter()
            .filter(|event| event.time_text().as_deref() == Some(time))
            .collect()
    }

    /// 候选共指：触发词与全部角色相同、极性/模态相同、时间不冲突。
    /// 只记录候选，绝不自动合并；时间未知或一方未知时降低分数并注明。
    pub fn coreference_candidates(&self) -> Vec<EventCoreference> {
        let mut candidates = Vec::new();
        for left in 0..self.events.len() {
            for right in (left + 1)..self.events.len() {
                let a = &self.events[left];
                let b = &self.events[right];
                if a.event_type != b.event_type
                    || a.polarity != b.polarity
                    || a.modality != b.modality
                {
                    continue;
                }
                let left_roles: BTreeSet<(&str, &str)> = a
                    .arguments
                    .iter()
                    .map(|argument| (argument.role.tag(), argument.value.as_str()))
                    .collect();
                let right_roles: BTreeSet<(&str, &str)> = b
                    .arguments
                    .iter()
                    .map(|argument| (argument.role.tag(), argument.value.as_str()))
                    .collect();
                if left_roles.is_empty() || left_roles != right_roles {
                    continue;
                }
                let (score, reason) = match (a.time_text(), b.time_text()) {
                    (Some(left_time), Some(right_time)) if left_time == right_time => {
                        (0.85, "触发词、角色、时间与情态一致，可候选共指")
                    }
                    (Some(_), Some(_)) => continue,
                    (None, None) => (0.6, "触发词与角色一致，但两处时间都未知"),
                    _ => (0.55, "触发词与角色一致，但一方时间未知"),
                };
                candidates.push(EventCoreference {
                    left: a.id.clone(),
                    right: b.id.clone(),
                    score,
                    reason: reason.to_string(),
                });
            }
        }
        candidates
    }

    /// 默认命名图 IRI：按来源文本指纹生成，重复导入可幂等覆盖。
    pub fn default_graph_iri(&self) -> String {
        format!("{}{}", EVENT_GRAPH_NS, self.source_fingerprint)
    }

    /// 事件图 → N-Quads（命名图）。默认图不写，调用方可整图替换。
    pub fn to_nquads(&self) -> String {
        self.to_nquads_in_graph(&self.default_graph_iri())
    }

    pub fn to_nquads_in_graph(&self, graph: &str) -> String {
        let mut output = String::new();
        let graph_term = format!("<{}>", graph);
        for event in &self.events {
            let event_iri = format!("<{}>", event_iri(&event.id));
            let mut triples: Vec<(String, String)> = vec![
                (
                    format!("<{}>", rdf_type_iri()),
                    format!("<{}>", event_class_iri()),
                ),
                (
                    format!("<{}>", event_property_iri("type")),
                    nquad_literal(&event.event_type),
                ),
                (
                    format!("<{}>", event_property_iri("trigger")),
                    nquad_literal(&event.trigger),
                ),
                (
                    format!("<{}>", event_property_iri("polarity")),
                    nquad_literal(polarity_key(event.polarity)),
                ),
                (
                    format!("<{}>", event_property_iri("modality")),
                    nquad_literal(modality_key(event.modality)),
                ),
                (
                    format!("<{}>", event_property_iri("evidence")),
                    nquad_literal(&event.evidence.sentence),
                ),
                (
                    format!("<{}>", event_property_iri("confidence")),
                    nquad_literal(&format!("{:.2}", event.confidence)),
                ),
            ];
            for argument in &event.arguments {
                triples.push((
                    format!("<{}>", event_property_iri(argument.role.tag())),
                    nquad_literal(&argument.value),
                ));
            }
            if let Some(time) = event.time_text() {
                triples.push((
                    format!("<{}>", event_property_iri("time")),
                    nquad_literal(&time),
                ));
            }
            let projection_iri = format!("<{}>", event_projection_iri(&event.id));
            triples.push((
                format!("<{}>", event_property_iri("projectsTo")),
                projection_iri.clone(),
            ));
            triples.push((
                format!("<{}>", rdf_type_iri()),
                format!("<{}>", statement_class_iri()),
            ));
            triples.push((
                format!("<{}>", statement_property_iri("subject")),
                nquad_literal(&event.projection.subject),
            ));
            triples.push((
                format!("<{}>", statement_property_iri("predicate")),
                nquad_literal(&event.projection.predicate),
            ));
            triples.push((
                format!("<{}>", statement_property_iri("object")),
                nquad_literal(&event.projection.object),
            ));
            for (predicate, object) in triples {
                let subject = if predicate == format!("<{}>", rdf_type_iri())
                    && object == format!("<{}>", statement_class_iri())
                {
                    projection_iri.clone()
                } else {
                    event_iri.clone()
                };
                output.push_str(&format!(
                    "{} {} {} {} .\n",
                    subject, predicate, object, graph_term
                ));
            }
        }
        for link in &self.links {
            output.push_str(&format!(
                "<{}> <{}> <{}> {} .\n",
                event_iri(&link.from),
                event_link_iri(link.kind),
                event_iri(&link.to),
                graph_term
            ));
        }
        output
    }

    /// 事件图 → 可装载的 Prolog 程序：事实 + 只读派生规则。
    pub fn to_prolog_program(&self) -> String {
        let mut output = self.to_prolog_lines().join("\n");
        if !output.is_empty() {
            output.push('\n');
        }
        output.push_str(EVENT_PROLOG_RULES);
        output
    }

    /// 事件图 → Prolog 事实行（不含命名图前缀，便于调用方自行决定装载位置）。
    pub fn to_prolog_lines(&self) -> Vec<String> {
        let mut lines = Vec::new();
        for event in &self.events {
            let id = escape_atom(&event.id);
            lines.push(format!("event('{}').", id));
            lines.push(format!(
                "event_type('{}', '{}').",
                id,
                escape_atom(&event.event_type)
            ));
            lines.push(format!(
                "event_trigger('{}', '{}').",
                id,
                escape_atom(&event.trigger)
            ));
            for argument in &event.arguments {
                lines.push(format!(
                    "event_role('{}', {}, '{}').",
                    id,
                    argument.role.tag(),
                    escape_atom(&argument.value)
                ));
            }
            if let Some(time) = event.time_text() {
                lines.push(format!("event_time('{}', '{}').", id, escape_atom(&time)));
            }
            lines.push(format!(
                "event_projects_to('{}', '{}', '{}', '{}').",
                id,
                escape_atom(&event.projection.subject),
                escape_atom(&event.projection.predicate),
                escape_atom(&event.projection.object)
            ));
            lines.push(format!(
                "event_evidence('{}', {}, '{}').",
                id,
                event.evidence.sentence_index,
                escape_atom(&event.evidence.sentence)
            ));
        }
        for link in &self.links {
            lines.push(format!(
                "event_link('{}', {}, '{}').",
                escape_atom(&link.from),
                link.kind.tag(),
                escape_atom(&link.to)
            ));
        }
        lines
    }

    pub fn render_text(&self) -> String {
        let mut output = String::new();
        let _ = writeln!(
            output,
            "事件图（{}）：{} 个事件，{} 条链接；证据指纹 {}。",
            self.dialect, self.stats.events, self.stats.links, self.source_fingerprint
        );
        if self.stats.truncated_events > 0 {
            let _ = writeln!(
                output,
                "事件上限：有 {} 个事件被截断。",
                self.stats.truncated_events
            );
        }
        for (index, event) in self.ordered_events().iter().enumerate() {
            let _ = writeln!(
                output,
                "{}. [{}] {}（置信度 {:.2}）",
                index + 1,
                event.id,
                event.render_text(),
                event.confidence
            );
            let _ = writeln!(
                output,
                "   证据（句 {}）：{}",
                event.evidence.sentence_index + 1,
                event.evidence.sentence
            );
            for link in self.links_from(&event.id) {
                if let Some(target) = self.event(&link.to) {
                    let _ = writeln!(
                        output,
                        "   链接：{} {} [{}]",
                        link.kind.label(),
                        target.render_text(),
                        target.id
                    );
                }
            }
        }
        if self.events.is_empty() {
            output.push_str("（没有抽取到事件；不把空图当作“什么都没发生”。）\n");
        }
        output
    }
}

fn clean_party(value: &str) -> (&str, usize) {
    let stripped = strip_subject_prefixes(value);
    let offset = value.len() - stripped.len();
    let trimmed = stripped.trim();
    let offset = offset + (stripped.len() - stripped.trim_start().len());
    (trimmed, offset)
}

fn polarity_key(polarity: Polarity) -> &'static str {
    match polarity {
        Polarity::Affirmative => "affirmative",
        Polarity::Negative => "negative",
    }
}

fn time_key(scope: &TemporalScope) -> String {
    scope
        .normalized
        .clone()
        .unwrap_or_else(|| scope.raw.clone())
}

fn render_arguments_key(arguments: &[EventArgument]) -> String {
    arguments
        .iter()
        .map(|argument| format!("{}:{}", argument.role.tag(), argument.value))
        .collect::<Vec<_>>()
        .join(",")
}

fn event_iri(id: &str) -> String {
    format!(
        "http://example.org/kg/event/{}",
        id.strip_prefix("evt:").unwrap_or(id)
    )
}

fn event_projection_iri(id: &str) -> String {
    format!(
        "http://example.org/kg/event-projection/{}",
        id.strip_prefix("evt:").unwrap_or(id)
    )
}

fn event_property_iri(name: &str) -> String {
    format!("http://example.org/kg/event#{}", name)
}

fn event_link_iri(kind: EventLinkKind) -> String {
    format!("http://example.org/kg/event#{}", kind.tag())
}

fn event_class_iri() -> String {
    "http://example.org/kg/event#Event".to_string()
}

fn rdf_type_iri() -> String {
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type".to_string()
}

fn statement_class_iri() -> String {
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement".to_string()
}

fn statement_property_iri(name: &str) -> String {
    format!("http://www.w3.org/1999/02/22-rdf-syntax-ns#{}", name)
}

fn modality_key(modality: Modality) -> &'static str {
    match modality {
        Modality::Asserted => "asserted",
        Modality::Possible => "possible",
        Modality::Necessary => "necessary",
        Modality::Uncertain => "uncertain",
    }
}

fn nquad_literal(text: &str) -> String {
    let mut escaped = String::with_capacity(text.len() + 2);
    escaped.push('"');
    for ch in text.chars() {
        match ch {
            '\\' => escaped.push_str("\\\\"),
            '"' => escaped.push_str("\\\""),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            _ => escaped.push(ch),
        }
    }
    escaped.push('"');
    escaped
}

fn event_start_ordinal(event: &EventNode) -> Option<i64> {
    event
        .time
        .as_ref()
        .and_then(TemporalScope::to_time_span)
        .and_then(|span| span.start)
        .map(|point| point.to_ordinal())
}

fn is_pronoun(value: &str) -> bool {
    matches!(
        value,
        "他" | "她" | "它" | "他们" | "她们" | "它们" | "其" | "该" | "这" | "那"
    )
}

fn canonical_event_type(trigger: &str) -> String {
    const GROUPS: &[(&[&str], &str)] = &[
        (&["出生于", "生于", "出生"], "出生"),
        (&["去世", "逝世", "死亡", "病逝"], "去世"),
        (&["成立", "创立", "创建", "建立", "创办"], "成立"),
        (&["加入", "入职", "加盟"], "加入"),
        (&["离开", "离职", "辞职", "退出"], "离开"),
        (&["前往", "到达", "抵达", "迁往", "去往", "来到"], "前往"),
        (&["位于", "坐落于", "地处"], "位于"),
        (&["认识", "结识"], "认识"),
        (&["结婚", "成婚"], "结婚"),
        (&["毕业", "毕业于"], "毕业"),
    ];
    for (surfaces, canonical) in GROUPS {
        if surfaces
            .iter()
            .any(|surface| trigger == *surface || trigger.ends_with(surface))
        {
            return (*canonical).to_string();
        }
    }
    trigger.to_string()
}

fn is_location_trigger(trigger: &str) -> bool {
    const LOCATION: &[&str] = &[
        "位于", "坐落", "地处", "居住", "住在", "出生", "生于", "前往", "到达", "抵达", "去往",
        "来到", "离开", "迁往", "迁居", "来自", "临近", "靠近",
    ];
    LOCATION.iter().any(|surface| trigger.contains(surface))
}

fn is_recipient_trigger(trigger: &str) -> bool {
    const RECIPIENT: &[&str] = &["给", "送给", "交给", "寄给", "借给", "告诉", "通知", "赠"];
    RECIPIENT.iter().any(|surface| trigger.contains(surface))
}

fn object_role(frame: &RelationFrame) -> EventRole {
    if is_location_trigger(&frame.predicate)
        || (is_location_like(&frame.object) && is_movement_trigger(&frame.predicate))
    {
        EventRole::Location
    } else if is_recipient_trigger(&frame.predicate) {
        EventRole::Recipient
    } else {
        EventRole::Patient
    }
}

fn is_movement_trigger(trigger: &str) -> bool {
    [
        "前往", "到达", "抵达", "去", "到", "离开", "迁", "回到", "返回",
    ]
    .iter()
    .any(|surface| trigger.contains(surface))
}

fn is_location_like(value: &str) -> bool {
    const SUFFIXES: &[&str] = &[
        "市", "省", "县", "区", "国", "州", "镇", "村", "路", "街", "山", "河", "湖", "海", "岛",
        "港",
    ];
    const NAMES: &[&str] = &[
        "北京", "上海", "广州", "深圳", "天津", "重庆", "香港", "澳门", "台北", "东京", "纽约",
        "伦敦", "巴黎", "罗马", "中国", "美国", "日本", "英国", "法国",
    ];
    NAMES.contains(&value)
        || (value.chars().count() >= 2 && SUFFIXES.iter().any(|suffix| value.ends_with(suffix)))
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FocusResolution {
    indices: Vec<usize>,
    implicit: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventQuestionKind {
    Who,
    When,
    Where,
    Why,
    WhatNext,
    Participants,
    Did,
}

impl EventQuestionKind {
    pub fn tag(self) -> &'static str {
        match self {
            EventQuestionKind::Who => "who",
            EventQuestionKind::When => "when",
            EventQuestionKind::Where => "where",
            EventQuestionKind::Why => "why",
            EventQuestionKind::WhatNext => "what_next",
            EventQuestionKind::Participants => "participants",
            EventQuestionKind::Did => "did",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventQuestion {
    pub text: String,
    pub kind: EventQuestionKind,
    pub anaphoric: bool,
}

impl EventQuestion {
    pub fn parse(text: &str) -> Self {
        let trimmed = text.trim();
        let kind = if contains_any(trimmed, &["为什么", "为何", "什么原因", "怎么会"]) {
            EventQuestionKind::Why
        } else if contains_any(
            trimmed,
            &["什么时候", "何时", "哪年", "哪一天", "哪月", "几点"],
        ) {
            EventQuestionKind::When
        } else if contains_any(trimmed, &["哪里", "哪儿", "在哪", "何处", "什么地方"]) {
            EventQuestionKind::Where
        } else if contains_any(
            trimmed,
            &["还有谁", "谁参加", "参与者", "都有谁", "哪些人", "谁参与"],
        ) {
            EventQuestionKind::Participants
        } else if contains_any(
            trimmed,
            &["后来", "接下来", "之后", "然后", "下一步", "此后", "最后呢"],
        ) {
            EventQuestionKind::WhatNext
        } else if contains_any(trimmed, &["吗", "是否", "有没有", "真的"]) {
            EventQuestionKind::Did
        } else {
            EventQuestionKind::Who
        };
        let anaphoric = contains_any(
            trimmed,
            &[
                "他",
                "她",
                "它",
                "他们",
                "她们",
                "这件事",
                "该事件",
                "此事",
                "后来",
            ],
        );
        Self {
            text: trimmed.to_string(),
            kind,
            anaphoric,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventAnswerStatus {
    Answered,
    Abstained,
}

impl EventAnswerStatus {
    pub fn tag(self) -> &'static str {
        match self {
            EventAnswerStatus::Answered => "answered",
            EventAnswerStatus::Abstained => "abstained",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EventAnswerItem {
    pub relation: String,
    pub event: EventNode,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EventAnswer {
    pub kind: EventQuestionKind,
    pub status: EventAnswerStatus,
    pub focus: Option<String>,
    pub implicit_focus: bool,
    pub answer: String,
    pub events: Vec<EventAnswerItem>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

impl EventAnswer {
    fn answered(
        kind: EventQuestionKind,
        primary: &EventNode,
        focused: &[&EventNode],
        implicit: bool,
        answer: String,
    ) -> Self {
        let mut events = vec![EventAnswerItem {
            relation: "焦点".to_string(),
            event: primary.clone(),
        }];
        for event in focused {
            if event.id == primary.id {
                continue;
            }
            events.push(EventAnswerItem {
                relation: "相关".to_string(),
                event: (*event).clone(),
            });
        }
        Self {
            kind,
            status: EventAnswerStatus::Answered,
            focus: Some(primary.id.clone()),
            implicit_focus: implicit,
            answer,
            events,
            reason: None,
        }
    }

    fn abstained(kind: EventQuestionKind, focus: Option<String>, reason: String) -> Self {
        Self {
            kind,
            status: EventAnswerStatus::Abstained,
            focus,
            implicit_focus: false,
            answer: String::new(),
            events: Vec::new(),
            reason: Some(reason),
        }
    }

    pub fn render_text(&self) -> String {
        let mut output = String::new();
        match self.status {
            EventAnswerStatus::Answered => {
                let _ = writeln!(output, "{}", self.answer);
                if self.implicit_focus {
                    output.push_str("（焦点来自上一事件，不是新实体。）\n");
                }
                if !self.events.is_empty() {
                    output.push_str("依据：\n");
                    for item in &self.events {
                        let _ = writeln!(
                            output,
                            "  - {} [{}] {}（证据：{}）",
                            item.relation,
                            item.event.id,
                            item.event.render_text(),
                            item.event.evidence.sentence
                        );
                    }
                }
            }
            EventAnswerStatus::Abstained => {
                let _ = writeln!(
                    output,
                    "无法回答（{}）：{}",
                    self.kind.tag(),
                    self.reason.as_deref().unwrap_or("没有证据")
                );
                if let Some(focus) = &self.focus {
                    if let Some(event) = self.events.iter().find(|item| &item.event.id == focus) {
                        let _ = writeln!(
                            output,
                            "焦点事件：[{}] {}",
                            event.event.id,
                            event.event.render_text()
                        );
                    }
                }
            }
        }
        output
    }
}

/// 多轮事件焦点：只记住事件 id，不复制图；同一事件不重复压栈。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventFocus {
    stack: Vec<String>,
    depth: usize,
}

impl EventFocus {
    pub fn new(depth: usize) -> Self {
        Self {
            stack: Vec::new(),
            depth: depth.max(1),
        }
    }

    pub fn push(&mut self, id: impl Into<String>) {
        let id = id.into();
        if self.stack.last() == Some(&id) {
            return;
        }
        if let Some(position) = self.stack.iter().position(|known| known == &id) {
            self.stack.remove(position);
        }
        self.stack.push(id);
        if self.stack.len() > self.depth {
            let overflow = self.stack.len() - self.depth;
            self.stack.drain(0..overflow);
        }
    }

    pub fn last(&self) -> Option<&str> {
        self.stack.last().map(String::as_str)
    }

    pub fn pop(&mut self) -> Option<String> {
        self.stack.pop()
    }

    pub fn clear(&mut self) {
        self.stack.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }

    pub fn depth(&self) -> usize {
        self.depth
    }

    pub fn ids(&self) -> &[String] {
        &self.stack
    }
}

/// 事件对话：把事件图与事件焦点栈绑在一起，支持「后来呢 / 为什么」追问。
#[derive(Debug, Clone)]
pub struct EventDialogue {
    graph: EventGraph,
    focus: EventFocus,
}

impl EventDialogue {
    pub fn new(graph: EventGraph) -> Self {
        Self {
            graph,
            focus: EventFocus::new(DEFAULT_FOCUS_DEPTH),
        }
    }

    pub fn with_focus_depth(graph: EventGraph, depth: usize) -> Self {
        Self {
            graph,
            focus: EventFocus::new(depth),
        }
    }

    pub fn graph(&self) -> &EventGraph {
        &self.graph
    }

    pub fn focus(&self) -> &EventFocus {
        &self.focus
    }

    pub fn ask(&mut self, question: &str) -> EventAnswer {
        let answer = self.graph.ask(question, self.focus.last());
        if answer.status == EventAnswerStatus::Answered {
            let next_focus = if answer.kind == EventQuestionKind::WhatNext {
                answer
                    .events
                    .iter()
                    .rev()
                    .find(|item| item.relation == "后续")
                    .map(|item| item.event.id.clone())
                    .or_else(|| answer.focus.clone())
            } else {
                answer.focus.clone()
            };
            if let Some(focus) = next_focus {
                self.focus.push(focus);
            }
        }
        answer
    }
}

fn build_links(events: &[EventNode]) -> Vec<EventLink> {
    let mut links = Vec::new();
    if events.is_empty() {
        return links;
    }
    let mut by_sentence: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (index, event) in events.iter().enumerate() {
        by_sentence
            .entry(event.evidence.sentence_index)
            .or_default()
            .push(index);
    }
    let sentences: Vec<usize> = by_sentence.keys().copied().collect();

    for (position, sentence_index) in sentences.iter().enumerate() {
        let current = &by_sentence[sentence_index];
        let sentence = &events[current[0]].evidence.sentence;
        let sentence_start = events[current[0]].evidence.sentence_start;
        let cause_position = first_marker(sentence, CAUSE_MARKERS).map(|p| sentence_start + p);
        let result_position = first_marker(sentence, RESULT_MARKERS).map(|p| sentence_start + p);
        let order_position = first_marker(sentence, ORDER_MARKERS).map(|p| sentence_start + p);

        if let Some(result_global) = result_position {
            let causes: Vec<usize> = current
                .iter()
                .copied()
                .filter(|index| {
                    events[*index].evidence.anchor < result_global
                        && cause_position
                            .map(|cause| events[*index].evidence.anchor > cause)
                            .unwrap_or(true)
                })
                .collect();
            let effects: Vec<usize> = current
                .iter()
                .copied()
                .filter(|index| events[*index].evidence.anchor > result_global)
                .collect();
            let causes = if causes.is_empty() {
                previous_events(&sentences, position, &by_sentence)
            } else {
                causes
            };
            for cause in &causes {
                for effect in &effects {
                    push_link(
                        &mut links,
                        events,
                        *cause,
                        *effect,
                        EventLinkKind::Causes,
                        0.9,
                        sentence,
                    );
                }
            }
        } else if let Some(cause_global) = cause_position {
            let causes: Vec<usize> = current
                .iter()
                .copied()
                .filter(|index| events[*index].evidence.anchor > cause_global)
                .collect();
            let effects = next_events(&sentences, position, &by_sentence);
            for cause in &causes {
                for effect in &effects {
                    push_link(
                        &mut links,
                        events,
                        *cause,
                        *effect,
                        EventLinkKind::Causes,
                        0.85,
                        sentence,
                    );
                }
            }
        }

        if let Some(order_global) = order_position {
            let following: Vec<usize> = current
                .iter()
                .copied()
                .filter(|index| events[*index].evidence.anchor > order_global)
                .collect();
            let following = if following.is_empty() {
                current.clone()
            } else {
                following
            };
            let previous = previous_events(&sentences, position, &by_sentence);
            for before in &previous {
                for after in &following {
                    push_link(
                        &mut links,
                        events,
                        *before,
                        *after,
                        EventLinkKind::Before,
                        0.8,
                        sentence,
                    );
                }
            }
        }
    }

    for left in 0..events.len() {
        for right in (left + 1)..events.len() {
            let (Some(left_time), Some(right_time)) = (
                events[left]
                    .time
                    .as_ref()
                    .and_then(TemporalScope::to_time_span),
                events[right]
                    .time
                    .as_ref()
                    .and_then(TemporalScope::to_time_span),
            ) else {
                continue;
            };
            if events[left].evidence.sentence_index == events[right].evidence.sentence_index {
                continue;
            }
            if let (Some(left_end), Some(right_start)) = (left_time.end, right_time.start) {
                if left_end < right_start {
                    push_link(
                        &mut links,
                        events,
                        left,
                        right,
                        EventLinkKind::Before,
                        0.75,
                        "显式时间比较",
                    );
                    continue;
                }
            }
            if let (Some(right_end), Some(left_start)) = (right_time.end, left_time.start) {
                if right_end < left_start {
                    push_link(
                        &mut links,
                        events,
                        right,
                        left,
                        EventLinkKind::Before,
                        0.75,
                        "显式时间比较",
                    );
                    continue;
                }
            }
            if left_time.overlaps(&right_time) {
                push_link(
                    &mut links,
                    events,
                    left,
                    right,
                    EventLinkKind::Simultaneous,
                    0.7,
                    "显式时间重叠",
                );
            }
        }
    }

    links
}

fn push_link(
    links: &mut Vec<EventLink>,
    events: &[EventNode],
    from: usize,
    to: usize,
    kind: EventLinkKind,
    confidence: f64,
    evidence: &str,
) {
    if from == to {
        return;
    }
    if links
        .iter()
        .any(|link| link.from == events[from].id && link.to == events[to].id && link.kind == kind)
    {
        return;
    }
    links.push(EventLink {
        from: events[from].id.clone(),
        to: events[to].id.clone(),
        kind,
        evidence: Some(evidence.chars().take(80).collect()),
        confidence,
    });
}

fn previous_events(
    sentences: &[usize],
    position: usize,
    by_sentence: &BTreeMap<usize, Vec<usize>>,
) -> Vec<usize> {
    if position == 0 {
        return Vec::new();
    }
    by_sentence[&sentences[position - 1]].clone()
}

fn next_events(
    sentences: &[usize],
    position: usize,
    by_sentence: &BTreeMap<usize, Vec<usize>>,
) -> Vec<usize> {
    if position + 1 >= sentences.len() {
        return Vec::new();
    }
    by_sentence[&sentences[position + 1]].clone()
}

fn first_marker(sentence: &str, markers: &[&str]) -> Option<usize> {
    markers
        .iter()
        .filter_map(|marker| sentence.find(marker))
        .min()
}

fn contains_any(text: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| text.contains(needle))
}

fn strip_leading_connectors(mut text: &str) -> &str {
    loop {
        let trimmed = text.trim_start_matches(|ch: char| {
            ch.is_whitespace() || matches!(ch, '，' | ',' | '：' | ':')
        });
        let mut changed = false;
        for connector in LEADING_CONNECTORS {
            if let Some(rest) = trimmed.strip_prefix(connector) {
                text = rest;
                changed = true;
                break;
            }
        }
        if !changed {
            return trimmed;
        }
    }
}

const EVENT_TRIGGERS: &[&str] = &[
    "出生于",
    "毕业于",
    "成立于",
    "坐落于",
    "隶属于",
    "工作于",
    "出生于",
    "去世",
    "逝世",
    "病逝",
    "加入",
    "入职",
    "离职",
    "辞职",
    "加盟",
    "创立",
    "创建",
    "成立",
    "建立",
    "创办",
    "出生",
    "前往",
    "到达",
    "抵达",
    "去往",
    "来到",
    "返回",
    "离开",
    "迁往",
    "迁居",
    "位于",
    "认识",
    "结识",
    "参加",
    "举办",
    "举行",
    "召开",
    "发布",
    "推出",
    "收购",
    "投资",
    "赢得",
    "获得",
    "成为",
    "担任",
    "签订",
    "合作",
    "访问",
    "结婚",
    "离婚",
    "毕业",
    "完成",
    "开始",
    "结束",
    "发生",
    "宣布",
];

const SUBJECT_PREFIXES: &[&str] = &[
    "因为",
    "由于",
    "鉴于",
    "所以",
    "因此",
    "于是",
    "因而",
    "故而",
    "后来",
    "随后",
    "然后",
    "接着",
    "最后",
    "此后",
    "之后",
    "接下来",
    "同时",
    "另外",
    "此外",
    "但是",
    "而且",
    "并且",
    "都",
    "也",
    "还",
    "就",
    "又",
    "曾",
    "已经",
    "正在",
    "将",
    "要",
    "会",
    "可以",
    "能够",
    "决定",
    "计划",
];

const RELATIVE_TIME_PREFIXES: &[&str] = &[
    "大前天",
    "大后天",
    "上个月",
    "下个月",
    "上星期",
    "下星期",
    "去年",
    "今年",
    "明年",
    "前天",
    "昨天",
    "今天",
    "明天",
    "后天",
];

fn extract_trigger_frames(text: &str, ranges: &[(usize, usize)]) -> Vec<RelationFrame> {
    let mut frames = Vec::new();
    for (sentence_index, (range_start, range_end)) in ranges.iter().enumerate() {
        let raw_sentence = &text[*range_start..*range_end];
        let sentence = raw_sentence.trim();
        if sentence.is_empty() {
            continue;
        }
        let sentence_start = *range_start + (raw_sentence.len() - raw_sentence.trim_start().len());
        let sentence_temporal = parse_temporal(sentence);
        for (clause_start, clause_end) in split_clause_ranges(sentence) {
            let clause = &sentence[clause_start..clause_end];
            let clause = clause.trim();
            if clause.is_empty() {
                continue;
            }
            let clause_leading = &sentence[clause_start..clause_end];
            let clause_trim_start =
                clause_start + (clause_leading.len() - clause_leading.trim_start().len());
            let Some((trigger, trigger_position)) = first_trigger(clause) else {
                continue;
            };
            let Some((subject, subject_start)) = subject_before(clause, trigger_position) else {
                continue;
            };
            let Some((object, object_start, locative, recipient)) =
                object_after(clause, trigger_position + trigger.len())
            else {
                continue;
            };
            if subject == object {
                continue;
            }
            let prefix = &clause[..trigger_position];
            let negative = ["没有", "不是", "不", "未", "没"]
                .iter()
                .any(|cue| prefix.contains(cue))
                && !prefix.contains("不丹");
            let modality = if contains_any(sentence, &["据说", "听说", "据称", "未必", "不一定"])
            {
                Modality::Uncertain
            } else if contains_any(sentence, &["可能", "也许", "或许"]) {
                Modality::Possible
            } else if contains_any(sentence, &["一定", "必然", "必定"]) {
                Modality::Necessary
            } else {
                Modality::Asserted
            };
            let temporal = parse_temporal(clause).or_else(|| sentence_temporal.clone());
            let mut frame = RelationFrame::new(&subject, trigger, &object);
            frame.polarity = if negative {
                Polarity::Negative
            } else {
                Polarity::Affirmative
            };
            frame.modality = modality;
            frame.temporal = temporal;
            let subject_global = sentence_start + clause_trim_start + subject_start;
            frame.evidence = EvidenceSpan {
                sentence_index,
                sentence: sentence.to_string(),
                start: subject_global,
                end: sentence_start + clause_trim_start + object_start + object.len(),
                subject_span: Some((0, subject.len())),
                object_span: Some((object_start.saturating_sub(subject_start), object.len())),
                polarity_span: None,
            };
            frame.extraction = vec!["事件触发词".to_string()];
            frame.qualifiers = vec![if locative {
                "地点角色".to_string()
            } else if recipient {
                "接受者角色".to_string()
            } else {
                "受事角色".to_string()
            }];
            frame.confidence = if locative || recipient { 0.78 } else { 0.74 };
            frame.id = format!(
                "frame:{}",
                fnv1a_hex(
                    format!(
                        "trigger|{}|{}|{}|{}|{}",
                        sentence_index,
                        subject_global,
                        trigger,
                        object,
                        text.len()
                    )
                    .as_bytes()
                )
            );
            frames.push(frame);
        }
    }
    frames
}

fn split_clause_ranges(sentence: &str) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut start = 0usize;
    for (index, ch) in sentence.char_indices() {
        if matches!(ch, '，' | ',') {
            ranges.push((start, index));
            start = index + ch.len_utf8();
        }
    }
    ranges.push((start, sentence.len()));
    ranges
}

fn first_trigger(sentence: &str) -> Option<(&'static str, usize)> {
    let mut best: Option<(&'static str, usize)> = None;
    for trigger in EVENT_TRIGGERS {
        if let Some(position) = sentence.find(trigger) {
            let replace = match best {
                None => true,
                Some((known, known_position)) => {
                    position < known_position
                        || (position == known_position && trigger.len() > known.len())
                }
            };
            if replace {
                best = Some((trigger, position));
            }
        }
    }
    best
}

fn subject_before(sentence: &str, trigger_position: usize) -> Option<(String, usize)> {
    let left = &sentence[..trigger_position];
    let mut segment_start = 0usize;
    for (index, ch) in left.char_indices() {
        if matches!(
            ch,
            '，' | ',' | '；' | ';' | '：' | ':' | '。' | '！' | '？' | '!' | '?' | ' ' | '\t'
        ) {
            segment_start = index + ch.len_utf8();
        }
    }
    let segment = &left[segment_start..];
    let stripped = strip_subject_prefixes(segment);
    let mut offset = segment_start + (segment.len() - stripped.len());
    let mut candidate = stripped.trim();
    offset += stripped.len() - stripped.trim_start().len();
    if let Some(position) = candidate
        .char_indices()
        .find_map(|(index, ch)| matches!(ch, '在' | '于' | '到' | '往' | '从').then_some(index))
    {
        if position > 0 {
            candidate = candidate[..position].trim_end();
        }
    }
    let candidate = candidate.trim_matches(|ch: char| {
        ch.is_whitespace()
            || matches!(
                ch,
                '「' | '」' | '『' | '』' | '“' | '”' | '‘' | '’' | '《' | '》' | '\'' | '"'
            )
    });
    if candidate.is_empty() || candidate.chars().count() > 40 {
        return None;
    }
    Some((candidate.to_string(), offset))
}

fn object_after(sentence: &str, after_trigger: usize) -> Option<(String, usize, bool, bool)> {
    if after_trigger >= sentence.len() {
        return None;
    }
    let right = &sentence[after_trigger..];
    let leading = right.len() - right.trim_start().len();
    let mut offset = after_trigger + leading;
    let mut segment = right.trim_start();
    let mut locative = false;
    let mut recipient = false;
    loop {
        let mut changed = false;
        for prefix in [
            "正在", "已经", "了", "着", "过", "于", "在", "到", "往", "给", "向", "对",
        ] {
            if let Some(rest) = segment.strip_prefix(prefix) {
                if matches!(prefix, "于" | "在" | "到" | "往") {
                    locative = true;
                }
                if matches!(prefix, "给" | "向" | "对") {
                    recipient = true;
                }
                let trimmed = rest.trim_start();
                offset += segment.len() - trimmed.len();
                segment = trimmed;
                changed = true;
                break;
            }
        }
        if !changed {
            break;
        }
    }
    let end = segment
        .char_indices()
        .find_map(|(index, ch)| {
            matches!(
                ch,
                '，' | ',' | '；' | ';' | '。' | '！' | '？' | '!' | '?' | '\n'
            )
            .then_some(index)
        })
        .unwrap_or(segment.len());
    let candidate = segment[..end].trim();
    if candidate.is_empty() || candidate.chars().count() > 40 {
        return None;
    }
    Some((candidate.to_string(), offset, locative, recipient))
}

fn strip_subject_prefixes(mut text: &str) -> &str {
    loop {
        let trimmed = text.trim_start_matches(|ch: char| {
            ch.is_whitespace() || matches!(ch, '，' | ',' | '：' | ':' | '、' | '；' | ';')
        });
        let without_time = strip_time_prefix(trimmed);
        if without_time.len() != trimmed.len() {
            text = without_time;
            continue;
        }
        let mut changed = false;
        for prefix in SUBJECT_PREFIXES {
            if let Some(rest) = trimmed.strip_prefix(prefix) {
                text = rest;
                changed = true;
                break;
            }
        }
        if !changed {
            return trimmed;
        }
    }
}

fn strip_time_prefix(text: &str) -> &str {
    for prefix in RELATIVE_TIME_PREFIXES {
        if let Some(rest) = text.strip_prefix(prefix) {
            return rest.trim_start_matches(|ch: char| {
                ch.is_whitespace() || matches!(ch, '，' | ',' | '、' | '：' | ':')
            });
        }
    }
    let Some(nian) = text.find('年') else {
        return text;
    };
    if nian > 12 {
        return text;
    }
    let digits: String = text[..nian]
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect();
    if digits.is_empty() || !digits.chars().all(|ch| ch.is_ascii_digit()) {
        return text;
    }
    let mut rest = &text[nian + '年'.len_utf8()..];
    for unit in ['月', '日', '号'] {
        let digits_len = rest
            .char_indices()
            .take_while(|(_, ch)| ch.is_ascii_digit() || ch.is_whitespace())
            .last()
            .map(|(index, ch)| index + ch.len_utf8())
            .unwrap_or(0);
        if digits_len > 0 && rest[digits_len..].starts_with(unit) {
            rest = &rest[digits_len + unit.len_utf8()..];
        }
    }
    rest.trim_start_matches(|ch: char| {
        ch.is_whitespace() || matches!(ch, '，' | ',' | '、' | '：' | ':')
    })
}

fn extract_fallback_clauses(
    text: &str,
    ranges: &[(usize, usize)],
    extractor: &FrameExtractor,
) -> (Vec<RelationFrame>, usize) {
    let mut frames = Vec::new();
    let mut clauses = 0usize;
    for (sentence_index, (range_start, range_end)) in ranges.iter().enumerate() {
        let raw_sentence = &text[*range_start..*range_end];
        if !contains_any(
            raw_sentence,
            &[
                "因为",
                "由于",
                "所以",
                "因此",
                "于是",
                "后来",
                "随后",
                "然后",
                "接着",
                "最后",
                "此后",
                "之后",
                "接下来",
            ],
        ) {
            continue;
        }
        let sentence = raw_sentence.trim();
        let mut clause_start = *range_start;
        let mut boundary = Vec::new();
        for (offset, ch) in raw_sentence.char_indices() {
            if matches!(ch, '，' | ',') {
                boundary.push((clause_start, *range_start + offset));
                clause_start = *range_start + offset + ch.len_utf8();
            }
        }
        boundary.push((clause_start, *range_end));
        for (start, end) in boundary {
            if start >= end {
                continue;
            }
            let clause_raw = &text[start..end];
            let clause_trimmed = clause_raw.trim();
            if clause_trimmed.is_empty() {
                continue;
            }
            let clause_trim_start = start + (clause_raw.len() - clause_raw.trim_start().len());
            let stripped = strip_leading_connectors(clause_trimmed);
            let body = stripped.trim();
            if body.chars().count() < 3 {
                continue;
            }
            let body_start = clause_trim_start + (clause_trimmed.len() - stripped.len());
            let (mut clause_frames, _) = extractor.extract_with_stats(body);
            if clause_frames.is_empty() {
                continue;
            }
            clauses += 1;
            if body_start + body.len() > text.len() {
                continue;
            }
            for frame in &mut clause_frames {
                frame.evidence.sentence_index = sentence_index;
                frame.evidence.sentence = sentence.to_string();
                frame.evidence.start = body_start;
                frame.evidence.end = body_start + body.len();
            }
            frames.extend(clause_frames);
        }
    }
    (frames, clauses)
}

fn sentence_ranges(text: &str) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut start = 0usize;
    for (index, ch) in text.char_indices() {
        if matches!(ch, '。' | '！' | '？' | '；' | '!' | '?' | ';' | '\n') {
            if !text[start..index].trim().is_empty() {
                ranges.push((start, index));
            }
            start = index + ch.len_utf8();
        }
    }
    if !text[start..].trim().is_empty() {
        ranges.push((start, text.len()));
    }
    ranges
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_events_with_roles_time_and_stable_ids() {
        let text = "1991年，张三认识李四。";
        let first = EventGraph::build(text);
        let second = EventGraph::build(text);
        assert_eq!(first.events.len(), 1, "{:?}", first.stats);
        assert_eq!(first.events[0].id, second.events[0].id);
        let event = &first.events[0];
        assert_eq!(event.argument(EventRole::Agent), Some("张三"));
        assert_eq!(event.argument(EventRole::Patient), Some("李四"));
        assert_eq!(event.time_text().as_deref(), Some("1991"));
        assert!(event.evidence.sentence.contains("张三认识李四"));
        assert!(!event.id.is_empty());
    }

    #[test]
    fn explicit_causal_connectors_create_directed_links() {
        let text = "因为张三认识李四，所以王五加入公司。";
        let graph = EventGraph::build(text);
        assert!(graph.events.iter().any(|event| event.trigger == "认识"));
        assert!(graph.events.iter().any(|event| event.trigger == "加入"));
        let cause = graph
            .events
            .iter()
            .find(|event| event.trigger == "认识")
            .expect("原因事件");
        let effect = graph
            .events
            .iter()
            .find(|event| event.trigger == "加入")
            .expect("结果事件");
        assert!(
            graph.links.iter().any(|link| {
                link.from == cause.id && link.to == effect.id && link.kind == EventLinkKind::Causes
            }),
            "{:?}",
            graph.links
        );
        let answer = graph.ask("为什么王五加入公司？", None);
        assert_eq!(answer.status, EventAnswerStatus::Answered, "{:?}", answer);
        assert!(answer.answer.contains("张三"));
    }

    #[test]
    fn explicit_times_order_events_and_answer_when() {
        let text = "1999年，张三加入公司。1991年，张三认识李四。";
        let graph = EventGraph::build(text);
        assert_eq!(graph.events.len(), 2, "{:?}", graph.stats);
        let ordered = graph.ordered_events();
        assert_eq!(ordered[0].trigger, "认识");
        assert_eq!(ordered[1].trigger, "加入");
        assert!(graph
            .links
            .iter()
            .any(|link| link.kind == EventLinkKind::Before));
        let answer = graph.ask("张三什么时候认识李四？", None);
        assert_eq!(answer.status, EventAnswerStatus::Answered);
        assert!(answer.answer.contains("1991"));
    }

    #[test]
    fn missing_cause_abstains_instead_of_guessing() {
        let graph = EventGraph::build("1991年，张三认识李四。");
        let answer = graph.ask("为什么张三认识李四？", None);
        assert_eq!(answer.status, EventAnswerStatus::Abstained);
        assert!(answer.reason.as_deref().unwrap_or("").contains("显式因果"));
    }

    #[test]
    fn dialogue_focus_carries_across_turns() {
        let graph = EventGraph::build("1991年，张三认识李四。1992年，张三加入公司。");
        let mut dialogue = EventDialogue::new(graph);
        let first = dialogue.ask("谁认识李四？");
        assert_eq!(first.status, EventAnswerStatus::Answered);
        assert!(first.answer.contains("张三"));
        let next = dialogue.ask("后来呢？");
        assert_eq!(next.status, EventAnswerStatus::Answered, "{:?}", next);
        assert!(next.answer.contains("加入"), "{:?}", next);
        assert!(dialogue.focus().ids().len() >= 2);
    }

    #[test]
    fn nquads_materialize_only_into_a_named_event_graph() {
        let graph = EventGraph::build("1991年，张三认识李四。");
        let nquads = graph.to_nquads();
        let graph_iri = graph.default_graph_iri();
        assert!(nquads.contains(&graph_iri));
        assert!(nquads.contains("event#Event"));
        assert!(nquads.contains("rdf-syntax-ns#Statement"));
        assert!(nquads.contains("event#projectsTo"));
        assert!(nquads
            .lines()
            .all(|line| line.trim_end().ends_with(&format!("<{}> .", graph_iri))));
        let custom = graph.to_nquads_in_graph("http://example.org/custom");
        assert!(custom
            .lines()
            .all(|line| line.ends_with("<http://example.org/custom> .")));
    }

    #[test]
    fn coreference_candidates_are_reported_not_merged() {
        let graph =
            EventGraph::build("1991年，张三认识李四。1991年，张三认识李四。1999年，张三认识李四。");
        assert_eq!(graph.events.len(), 3, "{:?}", graph.stats);
        let candidates = graph.coreference_candidates();
        assert!(candidates
            .iter()
            .any(|candidate| candidate.score >= 0.8 && candidate.reason.contains("时间")));
        assert!(candidates
            .iter()
            .all(|candidate| candidate.left != candidate.right));
    }

    #[test]
    fn prolog_program_ships_derived_rules_alongside_facts() {
        let graph = EventGraph::build("1991年，张三认识李四。");
        let program = graph.to_prolog_program();
        assert!(program.contains("event_role("));
        assert!(program.contains("event_before(A, B) :- event_link(A, before, B)."));
        assert!(program.contains("event_participant(E, P)"));
        let mut engine = crate::prolog_engine::PrologEngine::new();
        engine
            .consult_source(program)
            .expect("事件程序应可被 Prolog 引擎装载");
        let participants = engine.query("event_participant(E, P)");
        assert!(!participants.is_empty(), "角色规则应可查询");
    }

    #[test]
    fn prolog_projection_keeps_event_identity_and_evidence() {
        let graph = EventGraph::build("1991年，张三认识李四。");
        let lines = graph.to_prolog_lines();
        assert!(lines.iter().any(|line| line.starts_with("event('evt:")));
        assert!(lines.iter().any(|line| line.contains("event_role(")));
        assert!(lines.iter().any(|line| line.contains("event_projects_to(")));
        assert!(lines.iter().any(|line| line.contains("event_evidence(")));
    }
}

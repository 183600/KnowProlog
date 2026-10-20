//! 对话情景理论（Conversation Situation Theory, CST）。
//!
//! 编年史是事件流，本模块把事件流编译成一部**可查询的有限理论**：
//!
//!   s0 --e1--> s1 --e2--> s2 --e3--> ... --en--> sn
//!
//! 每个 `ei` 是一条编年史记录，每个 `si` 是事件之后的情景。情景、事件、
//! 问题、回答、取值都被具体化为常量，再交给 [`LogicProgram`] 查询。
//! 这样「第 N 轮相信什么」「哪条导入改变了回答」「如果撤销这次导入会怎样」
//! 都是同一套 Horn 目标下的不同查询。
//!
//! 设计边界（诚实优先）：
//!   - 只有效果可枚举的事件（文本导入回执、撤销、用户断言/撤回）参与
//!     反事实推断；规则教学、外链、JSON-LD/RDF 批量导入等显式弃权；
//!   - 信念折叠是「会话可枚举效果」的折叠，不是整库快照；它是情景理论
//!     的 ABox，不冒充 Oxigraph 的真值；
//!   - 归因只提名区间内触碰了问题实体/属性的事件，候选为空就明说
//!     「无法归因」，绝不用知识版本变化冒充原因；
//!   - 所有回答都从编年史投影，本模块不维护第二份可变对话状态。

use std::collections::{BTreeMap, BTreeSet};
use std::io;

use serde_json::{json, Value};

use crate::chinese_reasoning_nlu::{parse_lessons, LessonKind};
use crate::chronicle::{answer_key, question_key, Chronicle, ChronicleEntry};
use crate::context_image::{canonical_json, fnv1a_hex};
use crate::logic_program::{parse_rule, LAtom, LTerm, LogicProgram};
use crate::predicate_registry::{ImportAction, ImportReceipt};
use crate::proof_journal::ChangeKind;

/// 默认只编译最近 256 条编年史记录：在线查询有界，离线可调大。
pub const DEFAULT_MAX_EVENTS: usize = 256;
/// 默认最多物化 4096 条情景信念事实。
pub const DEFAULT_MAX_BELIEF_FACTS: usize = 4096;
/// 单条语义事件最多展开的效果条数（完整动作仍在导入日志里）。
pub const MAX_EFFECTS_PER_EVENT: usize = 256;

/// 情景理论编译参数。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TheoryConfig {
    pub max_events: usize,
    pub max_belief_facts: usize,
}

impl Default for TheoryConfig {
    fn default() -> Self {
        Self {
            max_events: DEFAULT_MAX_EVENTS,
            max_belief_facts: DEFAULT_MAX_BELIEF_FACTS,
        }
    }
}

/// 编年史事件的语义类别。未知 kind 仍保留为 [`EventKind::Other`]，不会被丢弃。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EventKind {
    Turn,
    Reply,
    Import,
    Retract,
    External,
    Rule,
    Assert,
    Forget,
    Conflict,
    Correct,
    Proof,
    Repair,
    Other,
}

impl EventKind {
    pub fn from_str(kind: &str) -> Self {
        match kind {
            "turn" => EventKind::Turn,
            "reply" => EventKind::Reply,
            "import" => EventKind::Import,
            "retract" => EventKind::Retract,
            "external" => EventKind::External,
            "rule" => EventKind::Rule,
            "assert" => EventKind::Assert,
            "forget" => EventKind::Forget,
            "conflict" => EventKind::Conflict,
            "correct" => EventKind::Correct,
            "proof" => EventKind::Proof,
            "repair" => EventKind::Repair,
            _ => EventKind::Other,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            EventKind::Turn => "turn",
            EventKind::Reply => "reply",
            EventKind::Import => "import",
            EventKind::Retract => "retract",
            EventKind::External => "external",
            EventKind::Rule => "rule",
            EventKind::Assert => "assert",
            EventKind::Forget => "forget",
            EventKind::Conflict => "conflict",
            EventKind::Correct => "correct",
            EventKind::Proof => "proof",
            EventKind::Repair => "repair",
            EventKind::Other => "other",
        }
    }

    pub fn is_semantic(self) -> bool {
        matches!(
            self,
            EventKind::Import
                | EventKind::Retract
                | EventKind::External
                | EventKind::Rule
                | EventKind::Assert
                | EventKind::Forget
                | EventKind::Conflict
                | EventKind::Correct
                | EventKind::Proof
                | EventKind::Repair
        )
    }

    pub fn is_causal_candidate(self) -> bool {
        matches!(
            self,
            EventKind::Import
                | EventKind::Retract
                | EventKind::External
                | EventKind::Rule
                | EventKind::Assert
                | EventKind::Forget
                | EventKind::Conflict
                | EventKind::Correct
                | EventKind::Repair
        )
    }
}

/// 可枚举效果的操作方向。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EffectOp {
    Add,
    Remove,
}

impl EffectOp {
    pub fn as_str(self) -> &'static str {
        match self {
            EffectOp::Add => "add",
            EffectOp::Remove => "remove",
        }
    }
}

/// 一条可折叠的信念效果。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct TheoryEffect {
    pub op: EffectOp,
    pub entity: String,
    pub predicate: String,
    pub value: String,
}

impl TheoryEffect {
    pub fn add(
        entity: impl Into<String>,
        predicate: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        Self {
            op: EffectOp::Add,
            entity: entity.into(),
            predicate: predicate.into(),
            value: value.into(),
        }
    }

    pub fn remove(
        entity: impl Into<String>,
        predicate: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        Self {
            op: EffectOp::Remove,
            entity: entity.into(),
            predicate: predicate.into(),
            value: value.into(),
        }
    }
}

/// 一条明确的更正：同一 (entity, predicate) 上旧值被判错、新值生效。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct TheoryCorrection {
    pub entity: String,
    pub predicate: String,
    pub from: String,
    pub to: String,
}

/// 一条未裁决的冲突：两个来源对同一 (entity, predicate) 给出不同取值。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct TheoryConflict {
    pub entity: String,
    pub predicate: String,
    pub existing: Vec<String>,
    pub incoming: String,
}

/// 事件在情景理论中的具体化形态。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TheoryEvent {
    pub seq: u64,
    pub kind: EventKind,
    pub raw_kind: String,
    pub situation: String,
    pub revision: u64,
    pub revision_after: Option<u64>,
    pub label: String,
    pub focus: Option<String>,
    pub effects: Vec<TheoryEffect>,
    pub effect_count: usize,
    pub enumerable: bool,
    pub digest: Option<String>,
    pub removes_digest: Option<String>,
    pub answer: Option<String>,
    pub corrections: Vec<TheoryCorrection>,
    pub conflicts: Vec<TheoryConflict>,
}

impl TheoryEvent {
    fn from_entry(entry: &ChronicleEntry) -> Self {
        let kind = EventKind::from_str(&entry.kind);
        let payload = &entry.payload;
        let effects_value = payload.get("effects").and_then(Value::as_array);
        let mut effects = Vec::new();
        if let Some(items) = effects_value {
            for item in items {
                let op = match item.get("op").and_then(Value::as_str) {
                    Some("remove") => EffectOp::Remove,
                    Some("add") => EffectOp::Add,
                    _ => default_op(kind),
                };
                let entity = item
                    .get("entity")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let predicate = item
                    .get("predicate")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let value = item
                    .get("value")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                if entity.is_empty() && predicate.is_empty() && value.is_empty() {
                    continue;
                }
                effects.push(TheoryEffect {
                    op,
                    entity,
                    predicate,
                    value,
                });
            }
        }
        effects.sort();
        effects.dedup();
        let effect_count = payload
            .get("effect_count")
            .and_then(Value::as_u64)
            .map(|count| count as usize)
            .unwrap_or(effects.len());
        let declared_enumerable = payload
            .get("enumerable")
            .and_then(Value::as_bool)
            .unwrap_or(!effects.is_empty());
        let truncated = payload
            .get("effects_truncated")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let enumerable = declared_enumerable && !truncated && effect_count == effects.len();
        let label = event_label(kind, payload);
        let focus = payload
            .get("focus")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|focus| !focus.is_empty())
            .map(str::to_string);
        let corrections = parse_corrections(payload);
        let conflicts = parse_conflicts(payload);
        Self {
            seq: entry.seq,
            kind,
            raw_kind: entry.kind.clone(),
            situation: format!("s{}", entry.seq),
            revision: entry.kb_revision,
            revision_after: payload.get("revision_after").and_then(Value::as_u64),
            label,
            focus,
            effects,
            effect_count,
            enumerable,
            digest: payload
                .get("digest")
                .and_then(Value::as_str)
                .map(str::to_string),
            removes_digest: payload
                .get("removes_digest")
                .and_then(Value::as_str)
                .map(str::to_string),
            answer: payload
                .get("answer")
                .and_then(Value::as_str)
                .map(str::to_string),
            corrections,
            conflicts,
        }
    }

    pub fn touches(&self) -> Vec<(String, String)> {
        let mut pairs = BTreeSet::new();
        for effect in &self.effects {
            if !effect.entity.is_empty() {
                pairs.insert((effect.entity.clone(), effect.predicate.clone()));
            }
        }
        for correction in &self.corrections {
            if !correction.entity.is_empty() {
                pairs.insert((correction.entity.clone(), correction.predicate.clone()));
            }
        }
        for conflict in &self.conflicts {
            if !conflict.entity.is_empty() {
                pairs.insert((conflict.entity.clone(), conflict.predicate.clone()));
            }
        }
        pairs.into_iter().collect()
    }
}

fn default_op(kind: EventKind) -> EffectOp {
    match kind {
        EventKind::Retract | EventKind::Forget => EffectOp::Remove,
        _ => EffectOp::Add,
    }
}

fn parse_corrections(payload: &Value) -> Vec<TheoryCorrection> {
    let Some(items) = payload.get("corrections").and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut corrections = Vec::new();
    for item in items {
        let entity = item
            .get("entity")
            .and_then(Value::as_str)
            .map(local_name)
            .unwrap_or_default();
        let predicate = item
            .get("predicate")
            .and_then(Value::as_str)
            .map(local_name)
            .unwrap_or_default();
        let from = item
            .get("from")
            .and_then(Value::as_str)
            .map(local_name)
            .unwrap_or_default();
        let to = item
            .get("to")
            .and_then(Value::as_str)
            .map(local_name)
            .unwrap_or_default();
        if entity.is_empty() || predicate.is_empty() || (from.is_empty() && to.is_empty()) {
            continue;
        }
        corrections.push(TheoryCorrection {
            entity,
            predicate,
            from,
            to,
        });
    }
    corrections.sort();
    corrections.dedup();
    corrections
}

fn parse_conflicts(payload: &Value) -> Vec<TheoryConflict> {
    let Some(items) = payload.get("conflicts").and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut conflicts = Vec::new();
    for item in items {
        let entity = item
            .get("entity")
            .and_then(Value::as_str)
            .map(local_name)
            .unwrap_or_default();
        let predicate = item
            .get("predicate")
            .and_then(Value::as_str)
            .map(local_name)
            .unwrap_or_default();
        let incoming = item
            .get("incoming")
            .and_then(Value::as_str)
            .map(local_name)
            .unwrap_or_default();
        let mut existing: Vec<String> = item
            .get("existing")
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(Value::as_str)
                    .map(local_name)
                    .collect()
            })
            .unwrap_or_default();
        existing.sort();
        existing.dedup();
        if entity.is_empty() || predicate.is_empty() || incoming.is_empty() {
            continue;
        }
        conflicts.push(TheoryConflict {
            entity,
            predicate,
            existing,
            incoming,
        });
    }
    conflicts.sort();
    conflicts.dedup();
    conflicts
}

fn event_label(kind: EventKind, payload: &Value) -> String {
    if let Some(text) = payload
        .get("text")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
    {
        return text.to_string();
    }
    if let Some(summary) = payload
        .get("summary")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|summary| !summary.is_empty())
    {
        return summary.lines().next().unwrap_or(summary).to_string();
    }
    if let Some(question) = payload
        .get("question")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|question| !question.is_empty())
    {
        return format!("问：{}", question);
    }
    if let Some(answer) = payload
        .get("answer")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|answer| !answer.is_empty())
    {
        return format!("答：{}", answer);
    }
    if let Some(user) = payload
        .get("user")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|user| !user.is_empty())
    {
        return user.to_string();
    }
    if let Some(goal) = payload
        .get("goal")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|goal| !goal.is_empty())
    {
        return format!("证明：{}", goal);
    }
    match kind {
        EventKind::Import => "导入".to_string(),
        EventKind::Retract => "撤销导入".to_string(),
        EventKind::External => "外链".to_string(),
        EventKind::Rule => "规则".to_string(),
        EventKind::Assert => "断言".to_string(),
        EventKind::Forget => "撤回断言".to_string(),
        EventKind::Conflict => "冲突".to_string(),
        EventKind::Correct => "更正".to_string(),
        EventKind::Proof => "证明".to_string(),
        EventKind::Repair => "修复".to_string(),
        EventKind::Turn => "输入".to_string(),
        EventKind::Reply => "回答".to_string(),
        EventKind::Other => "记录".to_string(),
    }
}

/// 情景：事件之后的世界。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Situation {
    pub id: String,
    pub seq: u64,
    pub revision: u64,
    pub focus: Option<String>,
    pub question: Option<String>,
    pub answer: Option<String>,
    pub kind: EventKind,
}

/// 一条已回答的问题。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnswerRecord {
    pub question: String,
    pub question_key: String,
    pub question_seq: u64,
    pub reply_seq: u64,
    pub answer: String,
    pub answer_key: String,
    pub revision: u64,
    pub situation: String,
    pub focus: Option<String>,
    pub proof_id: Option<String>,
    /// 提问时看到的编年史水位（旧日志缺省为提问序号）。
    pub observed_seq: u64,
    /// 提问时的知识版本。
    pub read_revision: u64,
    /// 提交回答时的知识版本。
    pub commit_revision: u64,
}

/// 一条情景信念。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BeliefFact {
    pub entity: String,
    pub predicate: String,
    pub value: String,
    pub source_seq: u64,
    pub source_kind: EventKind,
    pub digest: Option<String>,
}

/// 一次改口（同一问题相邻两次回答不同）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnswerChange {
    pub question: String,
    pub question_key: String,
    pub previous: AnswerRecord,
    pub current: AnswerRecord,
    pub kind: ChangeKind,
}

/// 归因候选。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateCause {
    pub event: TheoryEvent,
    pub score: u32,
    pub reasons: Vec<String>,
}

/// 归因报告：候选、被忽略事件、无法枚举的事件全部计数，避免「看起来全知道」。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CauseReport {
    pub change: Option<AnswerChange>,
    pub candidates: Vec<CandidateCause>,
    pub ignored_events: usize,
    pub non_enumerable_events: usize,
    pub note: String,
}

/// 反事实结论：只允许三态，不许「错误自信」。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CounterfactualVerdict {
    Flips,
    Unchanged,
    Unknown,
}

impl CounterfactualVerdict {
    pub fn label(self) -> &'static str {
        match self {
            CounterfactualVerdict::Flips => "会翻转",
            CounterfactualVerdict::Unchanged => "不会翻转",
            CounterfactualVerdict::Unknown => "无法判定（弃权）",
        }
    }
}

/// 有界反事实报告。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CounterfactualReport {
    pub question: String,
    pub answer: Option<AnswerRecord>,
    pub target_event: Option<TheoryEvent>,
    pub verdict: CounterfactualVerdict,
    pub affected: Vec<TheoryEffect>,
    pub remaining_support: usize,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, Default)]
struct BeliefState {
    values: BTreeMap<(String, String), Vec<BeliefFact>>,
}

impl BeliefState {
    fn apply(&mut self, event: &TheoryEvent) {
        for effect in &event.effects {
            let key = (effect.entity.clone(), effect.predicate.clone());
            match effect.op {
                EffectOp::Add => {
                    let entries = self.values.entry(key).or_default();
                    let duplicate = entries.iter().any(|fact| {
                        fact.value == effect.value
                            && fact.source_seq == event.seq
                            && fact.digest == event.digest
                    });
                    if !duplicate {
                        entries.push(BeliefFact {
                            entity: effect.entity.clone(),
                            predicate: effect.predicate.clone(),
                            value: effect.value.clone(),
                            source_seq: event.seq,
                            source_kind: event.kind,
                            digest: event.digest.clone(),
                        });
                        entries.sort_by(|left, right| {
                            left.entity
                                .cmp(&right.entity)
                                .then(left.predicate.cmp(&right.predicate))
                                .then(left.value.cmp(&right.value))
                                .then(left.source_seq.cmp(&right.source_seq))
                        });
                    }
                }
                EffectOp::Remove => {
                    if let Some(entries) = self.values.get_mut(&key) {
                        let digest = event.removes_digest.as_deref().or(event.digest.as_deref());
                        let mut index = digest.and_then(|digest| {
                            entries.iter().rposition(|fact| {
                                fact.value == effect.value && fact.digest.as_deref() == Some(digest)
                            })
                        });
                        if index.is_none() {
                            index = entries.iter().rposition(|fact| fact.value == effect.value);
                        }
                        if let Some(index) = index {
                            entries.remove(index);
                        }
                        if entries.is_empty() {
                            self.values.remove(&key);
                        }
                    }
                }
            }
        }
    }

    fn snapshot(&self) -> Vec<BeliefFact> {
        let mut facts = Vec::new();
        for entries in self.values.values() {
            facts.extend(entries.iter().cloned());
        }
        facts.sort_by(|left, right| {
            left.entity
                .cmp(&right.entity)
                .then(left.predicate.cmp(&right.predicate))
                .then(left.value.cmp(&right.value))
                .then(left.source_seq.cmp(&right.source_seq))
        });
        facts.dedup();
        facts
    }
}

#[derive(Debug, Clone)]
struct PendingTurn {
    seq: u64,
    question: String,
    focus: Option<String>,
}

/// 编译后的对话情景理论。
#[derive(Debug, Clone)]
pub struct ConversationTheory {
    events: Vec<TheoryEvent>,
    situations: Vec<Situation>,
    answers: Vec<AnswerRecord>,
    changes: Vec<AnswerChange>,
    snapshots: BTreeMap<u64, Vec<BeliefFact>>,
    program: LogicProgram,
    reverse: BTreeMap<String, String>,
    evaluation_error: Option<String>,
    window_start_seq: u64,
    window_end_seq: u64,
    total_entries: usize,
    truncated_belief_facts: bool,
}

impl ConversationTheory {
    pub fn build(chronicle: &Chronicle) -> Self {
        Self::build_with_config(chronicle, TheoryConfig::default())
    }

    pub fn build_with_config(chronicle: &Chronicle, config: TheoryConfig) -> Self {
        let total_entries = chronicle.len();
        let all_entries = chronicle.entries();
        let start = all_entries.len().saturating_sub(config.max_events.max(1));
        let entries = &all_entries[start..];
        let window_start_seq = entries.first().map(|entry| entry.seq).unwrap_or(0);
        let window_end_seq = entries.last().map(|entry| entry.seq).unwrap_or(0);
        let initial_revision = entries
            .first()
            .map(|entry| entry.kb_revision)
            .unwrap_or_else(|| chronicle.kb_revision());

        let mut events: Vec<TheoryEvent> = Vec::with_capacity(entries.len());
        let mut situations = Vec::with_capacity(entries.len() + 1);
        situations.push(Situation {
            id: "s0".to_string(),
            seq: 0,
            revision: initial_revision,
            focus: None,
            question: None,
            answer: None,
            kind: EventKind::Other,
        });
        let mut answers: Vec<AnswerRecord> = Vec::new();
        let mut pending: Option<PendingTurn> = None;
        let mut belief_state = BeliefState::default();
        let mut snapshots: BTreeMap<u64, Vec<BeliefFact>> = BTreeMap::new();

        for entry in entries {
            let event = TheoryEvent::from_entry(entry);
            belief_state.apply(&event);
            let mut situation = Situation {
                id: format!("s{}", entry.seq),
                seq: entry.seq,
                revision: entry.kb_revision,
                focus: event.focus.clone(),
                question: None,
                answer: None,
                kind: event.kind,
            };
            match event.kind {
                EventKind::Turn => {
                    let mode = entry
                        .payload
                        .get("mode")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    let user = entry
                        .payload
                        .get("user")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .trim();
                    if mode == "query" && !user.is_empty() {
                        pending = Some(PendingTurn {
                            seq: entry.seq,
                            question: user.to_string(),
                            focus: event.focus.clone(),
                        });
                        situation.question = Some(user.to_string());
                        situation.focus = event.focus.clone();
                    }
                }
                EventKind::Reply => {
                    let answer = entry
                        .payload
                        .get("answer")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .trim()
                        .to_string();
                    let payload_question = entry
                        .payload
                        .get("question")
                        .and_then(Value::as_str)
                        .map(str::trim)
                        .filter(|question| !question.is_empty())
                        .map(str::to_string);
                    let turn = pending.take();
                    let question = payload_question
                        .or_else(|| turn.as_ref().map(|turn| turn.question.clone()));
                    if let Some(question) = question {
                        let key = entry
                            .payload
                            .get("question_key")
                            .and_then(Value::as_str)
                            .map(str::to_string)
                            .unwrap_or_else(|| question_key(&question));
                        let question_seq = turn.as_ref().map(|turn| turn.seq).unwrap_or(entry.seq);
                        let focus = turn
                            .as_ref()
                            .and_then(|turn| turn.focus.clone())
                            .or_else(|| event.focus.clone());
                        answers.push(AnswerRecord {
                            question: question.clone(),
                            question_key: key,
                            question_seq,
                            reply_seq: entry.seq,
                            answer: answer.clone(),
                            answer_key: answer_key(&answer),
                            revision: entry.kb_revision,
                            situation: situation.id.clone(),
                            focus: focus.clone(),
                            proof_id: entry
                                .payload
                                .get("proof_id")
                                .and_then(Value::as_str)
                                .map(str::to_string),
                            observed_seq: entry
                                .payload
                                .get("observed_seq")
                                .and_then(Value::as_u64)
                                .unwrap_or(question_seq),
                            read_revision: entry
                                .payload
                                .get("read_revision")
                                .and_then(Value::as_u64)
                                .unwrap_or(entry.kb_revision),
                            commit_revision: entry
                                .payload
                                .get("commit_revision")
                                .and_then(Value::as_u64)
                                .unwrap_or(entry.kb_revision),
                        });
                        situation.question = Some(question);
                        situation.focus = focus;
                    }
                    if !answer.is_empty() {
                        situation.answer = Some(answer);
                    }
                    snapshots.insert(entry.seq, belief_state.snapshot());
                }
                _ => {}
            }
            events.push(event);
            situations.push(situation);
        }

        let changes = compute_changes(&answers);
        let (program, reverse, evaluation_error, truncated_belief_facts) = compile_program(
            &events,
            &answers,
            &changes,
            &snapshots,
            config.max_belief_facts,
        );

        Self {
            events,
            situations,
            answers,
            changes,
            snapshots,
            program,
            reverse,
            evaluation_error,
            window_start_seq,
            window_end_seq,
            total_entries,
            truncated_belief_facts,
        }
    }

    pub fn program(&self) -> &LogicProgram {
        &self.program
    }

    pub fn evaluation_error(&self) -> Option<&str> {
        self.evaluation_error.as_deref()
    }

    pub fn events(&self) -> &[TheoryEvent] {
        &self.events
    }

    pub fn situations(&self) -> &[Situation] {
        &self.situations
    }

    pub fn answers(&self) -> &[AnswerRecord] {
        &self.answers
    }

    pub fn changes(&self) -> &[AnswerChange] {
        &self.changes
    }

    pub fn window(&self) -> (u64, u64) {
        (self.window_start_seq, self.window_end_seq)
    }

    pub fn total_entries(&self) -> usize {
        self.total_entries
    }

    pub fn truncated_belief_facts(&self) -> bool {
        self.truncated_belief_facts
    }

    /// 解析内部常量（`n_姚明` 等）回人类可读文本；未知常量原样返回。
    pub fn resolve(&self, constant: &str) -> String {
        self.reverse
            .get(constant)
            .cloned()
            .unwrap_or_else(|| constant.to_string())
    }

    pub fn matching_answers(&self, query: &str) -> Vec<AnswerRecord> {
        let query = query.trim();
        if query.is_empty() {
            return self.answers.clone();
        }
        let key = question_key(query);
        let exact: Vec<AnswerRecord> = self
            .answers
            .iter()
            .filter(|answer| answer.question_key == key)
            .cloned()
            .collect();
        if !exact.is_empty() {
            return exact;
        }
        let needle = query.to_lowercase();
        self.answers
            .iter()
            .filter(|answer| {
                answer.question.to_lowercase().contains(&needle)
                    || answer.question_key.contains(&needle)
            })
            .cloned()
            .collect()
    }

    /// 某情景（或情景之前）的可枚举信念。不是整库快照，只折叠语义事件效果。
    pub fn beliefs_at(&self, seq: u64) -> Vec<BeliefFact> {
        if let Some(snapshot) = self.snapshots.get(&seq) {
            return snapshot.clone();
        }
        let mut state = BeliefState::default();
        for event in self.events.iter().filter(|event| event.seq <= seq) {
            state.apply(event);
        }
        state.snapshot()
    }

    pub fn latest_change(&self, query: &str) -> Option<AnswerChange> {
        let matched = self.matching_answers(query);
        let keys: BTreeSet<&str> = matched
            .iter()
            .map(|answer| answer.question_key.as_str())
            .collect();
        self.changes
            .iter()
            .rev()
            .find(|change| {
                keys.contains(change.question_key.as_str())
                    || change.question.contains(query.trim())
            })
            .cloned()
    }

    /// 区间归因：只提名「触碰了问题/回答词面或当轮焦点」的语义事件。
    pub fn causes(&self, query: &str) -> CauseReport {
        let Some(change) = self.latest_change(query) else {
            return CauseReport {
                change: None,
                candidates: Vec::new(),
                ignored_events: 0,
                non_enumerable_events: 0,
                note: if self.matching_answers(query).is_empty() {
                    format!("编年史里没有「{}」的回答，无法归因。", query.trim())
                } else {
                    format!("「{}」没有改口记录：相邻回答一致。", query.trim())
                },
            };
        };
        let report = self.causes_for_change(&change);
        report
    }

    pub fn causes_for_change(&self, change: &AnswerChange) -> CauseReport {
        let mut haystack = format!(
            "{} {} {}",
            change.question, change.previous.answer, change.current.answer
        )
        .to_lowercase();
        if let Some(focus) = change
            .current
            .focus
            .as_deref()
            .or(change.previous.focus.as_deref())
        {
            haystack.push(' ');
            haystack.push_str(&focus.to_lowercase());
        }

        let mut candidates = Vec::new();
        let mut ignored_events = 0usize;
        let mut non_enumerable_events = 0usize;
        for event in &self.events {
            if !event.kind.is_causal_candidate() {
                continue;
            }
            if event.seq <= change.previous.reply_seq || event.seq >= change.current.reply_seq {
                continue;
            }
            if event.effects.is_empty() {
                non_enumerable_events += 1;
                if event.kind == EventKind::Rule {
                    candidates.push(CandidateCause {
                        event: event.clone(),
                        score: 1,
                        reasons: vec!["区间内的规则变更（影响推导，效果不可完全枚举）".to_string()],
                    });
                }
                continue;
            }
            let mut score = 0u32;
            let mut reasons = Vec::new();
            for effect in &event.effects {
                let entity = effect.entity.to_lowercase();
                let predicate = effect.predicate.to_lowercase();
                let value = effect.value.to_lowercase();
                if !entity.is_empty() && haystack.contains(&entity) {
                    score += 2;
                    reasons.push(format!("触及实体「{}」", effect.entity));
                }
                if !predicate.is_empty() && haystack.contains(&predicate) {
                    score += 1;
                    reasons.push(format!("触及属性「{}」", effect.predicate));
                }
                if !value.is_empty()
                    && (change.current.answer.contains(&effect.value)
                        || change.previous.answer.contains(&effect.value))
                {
                    score += 4;
                    reasons.push(format!("改变回答中出现的取值「{}」", effect.value));
                }
            }
            if event.kind == EventKind::Rule && score == 0 {
                score = 1;
                reasons.push("区间内的规则变更（影响推导）".to_string());
            }
            reasons.sort();
            reasons.dedup();
            if score == 0 {
                ignored_events += 1;
                continue;
            }
            candidates.push(CandidateCause {
                event: event.clone(),
                score,
                reasons,
            });
        }
        candidates.sort_by(|left, right| {
            right
                .score
                .cmp(&left.score)
                .then(right.event.seq.cmp(&left.event.seq))
        });
        let note = if candidates.is_empty() {
            if non_enumerable_events > 0 {
                "没有可核对词面的候选原因；区间内有不可枚举事件，无法进一步归因。".to_string()
            } else {
                "无法归因：区间内没有触碰该问题/回答词面的语义事件。".to_string()
            }
        } else if non_enumerable_events > 0 {
            "候选只覆盖可枚举效果；区间内还有不可枚举事件，结论不声称穷尽原因。".to_string()
        } else {
            "候选按词面命中和取值变化打分；这是归因提示，不是因果证明。".to_string()
        };
        CauseReport {
            change: Some(change.clone()),
            candidates,
            ignored_events,
            non_enumerable_events,
            note,
        }
    }

    /// 有界反事实：撤销最近一条可枚举语义事件，看当前回答是否失去支持。
    pub fn counterfactual_undo_latest(&self, query: &str) -> CounterfactualReport {
        let matched = self.matching_answers(query);
        let Some(answer) = matched.last().cloned() else {
            return CounterfactualReport {
                question: query.to_string(),
                answer: None,
                target_event: None,
                verdict: CounterfactualVerdict::Unknown,
                affected: Vec::new(),
                remaining_support: 0,
                reasons: vec!["编年史里没有这条问题的回答。".to_string()],
            };
        };
        let event = self
            .events
            .iter()
            .rev()
            .find(|event| event.seq < answer.reply_seq && event.kind.is_semantic())
            .cloned();
        match event {
            Some(event) => self.counterfactual_undo_event(&event, &answer),
            None => CounterfactualReport {
                question: answer.question.clone(),
                answer: Some(answer),
                target_event: None,
                verdict: CounterfactualVerdict::Unknown,
                affected: Vec::new(),
                remaining_support: 0,
                reasons: vec!["该回答之前没有语义事件可撤销。".to_string()],
            },
        }
    }

    pub fn counterfactual_undo_seq(&self, seq: u64, query: &str) -> CounterfactualReport {
        let matched = self.matching_answers(query);
        let Some(answer) = matched.last().cloned() else {
            return CounterfactualReport {
                question: query.to_string(),
                answer: None,
                target_event: None,
                verdict: CounterfactualVerdict::Unknown,
                affected: Vec::new(),
                remaining_support: 0,
                reasons: vec!["编年史里没有这条问题的回答。".to_string()],
            };
        };
        match self.events.iter().find(|event| event.seq == seq).cloned() {
            Some(event) => self.counterfactual_undo_event(&event, &answer),
            None => CounterfactualReport {
                question: answer.question.clone(),
                answer: Some(answer),
                target_event: None,
                verdict: CounterfactualVerdict::Unknown,
                affected: Vec::new(),
                remaining_support: 0,
                reasons: vec![format!("窗口内没有 seq {} 的事件。", seq)],
            },
        }
    }

    fn counterfactual_undo_event(
        &self,
        event: &TheoryEvent,
        answer: &AnswerRecord,
    ) -> CounterfactualReport {
        if event.seq >= answer.reply_seq {
            return CounterfactualReport {
                question: answer.question.clone(),
                answer: Some(answer.clone()),
                target_event: Some(event.clone()),
                verdict: CounterfactualVerdict::Unchanged,
                affected: Vec::new(),
                remaining_support: 0,
                reasons: vec!["事件发生在该回答之后，撤销它不会改变历史回答。".to_string()],
            };
        }
        if !event.kind.is_semantic() {
            return CounterfactualReport {
                question: answer.question.clone(),
                answer: Some(answer.clone()),
                target_event: Some(event.clone()),
                verdict: CounterfactualVerdict::Unknown,
                affected: Vec::new(),
                remaining_support: 0,
                reasons: vec!["这不是语义事件（只是对话输入/回答），没有可重放效果。".to_string()],
            };
        }
        if !event.enumerable || event.effects.is_empty() {
            return CounterfactualReport {
                question: answer.question.clone(),
                answer: Some(answer.clone()),
                target_event: Some(event.clone()),
                verdict: CounterfactualVerdict::Unknown,
                affected: Vec::new(),
                remaining_support: 0,
                reasons: vec![
                    "该事件的效果不可完全枚举（规则/外链/批量导入等），拒绝给出反事实结论。"
                        .to_string(),
                ],
            };
        }

        let haystack = format!("{} {}", answer.question, answer.answer).to_lowercase();
        let focus = answer.focus.clone().unwrap_or_default().to_lowercase();
        let affected_effects: Vec<TheoryEffect> = event
            .effects
            .iter()
            .filter(|effect| {
                let entity = effect.entity.to_lowercase();
                let predicate = effect.predicate.to_lowercase();
                let value = effect.value.to_lowercase();
                let value_hit = !value.is_empty() && haystack.contains(&value);
                let entity_hit = !entity.is_empty()
                    && (haystack.contains(&entity)
                        || (!focus.is_empty() && focus.contains(&entity)));
                let predicate_hit = !predicate.is_empty() && haystack.contains(&predicate);
                value_hit || (entity_hit && predicate_hit) || (entity_hit && value.is_empty())
            })
            .cloned()
            .collect();

        if affected_effects.is_empty() {
            return CounterfactualReport {
                question: answer.question.clone(),
                answer: Some(answer.clone()),
                target_event: Some(event.clone()),
                verdict: CounterfactualVerdict::Unchanged,
                affected: Vec::new(),
                remaining_support: 0,
                reasons: vec![
                    "该事件的可枚举效果没有出现在问题、回答或焦点中：按效果判断不会改变这条回答。"
                        .to_string(),
                ],
            };
        }

        let beliefs = self.beliefs_at(answer.reply_seq);
        let mut flips = Vec::new();
        let mut uncertain = Vec::new();
        let mut remaining_support = 0usize;
        for effect in &affected_effects {
            match effect.op {
                EffectOp::Add => {
                    if effect.value.is_empty() {
                        uncertain.push(effect.clone());
                        continue;
                    }
                    if !answer.answer.contains(&effect.value) {
                        uncertain.push(effect.clone());
                        continue;
                    }
                    let alternative = beliefs.iter().any(|fact| {
                        fact.entity == effect.entity
                            && fact.predicate == effect.predicate
                            && fact.value == effect.value
                            && fact.source_seq != event.seq
                    });
                    if alternative {
                        remaining_support += 1;
                    } else {
                        flips.push(effect.clone());
                    }
                }
                EffectOp::Remove => {
                    if !effect.value.is_empty() && answer.answer.contains(&effect.value) {
                        remaining_support += 1;
                        continue;
                    }
                    let earlier_has_value = self
                        .answers
                        .iter()
                        .filter(|record| {
                            record.question_key == answer.question_key
                                && record.reply_seq < answer.reply_seq
                        })
                        .any(|record| record.answer.contains(&effect.value));
                    if earlier_has_value {
                        flips.push(effect.clone());
                    } else {
                        uncertain.push(effect.clone());
                    }
                }
            }
        }

        let (verdict, mut reasons) = if !flips.is_empty() {
            (
                CounterfactualVerdict::Flips,
                vec![format!(
                    "撤销该事件会移除回答所依赖的 {} 条取值来源（无替代来源）。",
                    flips.len()
                )],
            )
        } else if !uncertain.is_empty() {
            (
                CounterfactualVerdict::Unknown,
                vec!["事件触及问题，但回答没有暴露可核对的取值，无法确定翻转方向。".to_string()],
            )
        } else {
            (
                CounterfactualVerdict::Unchanged,
                vec![format!(
                    "相关取值仍有 {} 条替代来源，撤销该事件不会改变这条回答。",
                    remaining_support
                )],
            )
        };
        reasons.push("只按编年史中可枚举的效果推断，没有重放整库查询。".to_string());
        CounterfactualReport {
            question: answer.question.clone(),
            answer: Some(answer.clone()),
            target_event: Some(event.clone()),
            verdict,
            affected: affected_effects,
            remaining_support,
            reasons,
        }
    }

    /// 在编译后的理论里执行 Datalog 查询，返回变量绑定（常量已解析为文本）。
    pub fn query(&self, pattern: &str) -> Result<Vec<BTreeMap<String, String>>, String> {
        let envs = self
            .program
            .query(pattern)
            .map_err(|error| format!("查询解析失败：{}", error))?;
        Ok(envs
            .into_iter()
            .map(|env| {
                env.into_iter()
                    .map(|(name, value)| (name, self.resolve(&value)))
                    .collect()
            })
            .collect())
    }

    pub fn render_report(&self) -> String {
        let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
        for event in &self.events {
            *counts.entry(event.kind.as_str()).or_default() += 1;
        }
        let count_text = counts
            .iter()
            .map(|(kind, count)| format!("{} {}", kind, count))
            .collect::<Vec<_>>()
            .join("、");
        let mut lines = vec![format!(
            "对话情景理论（CST）：窗口 seq {}..{}（编译 {} 条，日志共 {} 条）；情景 {} 个，问答 {} 条，改口 {} 处。",
            self.window_start_seq,
            self.window_end_seq,
            self.events.len(),
            self.total_entries,
            self.situations.len(),
            self.answers.len(),
            self.changes.len()
        )];
        lines.push(format!(
            "语义事件：{}；理论事实 {} 条，规则 {} 条。",
            if count_text.is_empty() {
                "无"
            } else {
                &count_text
            },
            self.program.fact_count(),
            self.program.rule_count()
        ));
        if let Some(error) = &self.evaluation_error {
            lines.push(format!("理论求值警告：{}", error));
        }
        lines.push(
            "可查询谓词：situation/1、next_situation/2、before/2、event_kind/2、revision_at/2、focus_at/2、asked_at/3、answered_at/4、answer_change/3、change_at/4、touches/3、belief_at/5、candidate_cause/2。"
                .to_string(),
        );
        lines.push("最近情景（新 → 旧，最多 8 条）：".to_string());
        for situation in self.situations.iter().rev().filter(|s| s.seq != 0).take(8) {
            let focus = situation
                .focus
                .as_deref()
                .map(|focus| format!("焦点 {}；", focus))
                .unwrap_or_default();
            let question = situation
                .question
                .as_deref()
                .map(|question| format!("问 {}；", truncate(question, 32)))
                .unwrap_or_default();
            let answer = situation
                .answer
                .as_deref()
                .map(|answer| format!("答 {}；", truncate(answer, 40)))
                .unwrap_or_default();
            let event_text = self
                .events
                .iter()
                .find(|event| event.seq == situation.seq)
                .filter(|event| !matches!(event.kind, EventKind::Turn | EventKind::Reply))
                .map(|event| format!("{}；", truncate(&event.label, 56)))
                .unwrap_or_default();
            lines.push(format!(
                "- {}（seq {}，rev {}，{}）：{}{}{}{}",
                situation.id,
                situation.seq,
                situation.revision,
                situation.kind.as_str(),
                focus,
                question,
                answer,
                event_text
            ));
        }
        if self.truncated_belief_facts {
            lines.push(
                "注意：belief_at 事实超过预算，已截断；beliefs_at 接口仍可完整折叠。".to_string(),
            );
        }
        lines.join("\n")
    }

    pub fn render_situation(&self, seq: u64) -> String {
        let Some(event) = self.events.iter().find(|event| event.seq == seq) else {
            return format!("窗口内没有 seq {} 的情景。", seq);
        };
        let Some(situation) = self
            .situations
            .iter()
            .find(|situation| situation.seq == seq)
        else {
            return format!("窗口内没有 seq {} 的情景。", seq);
        };
        let mut lines = vec![format!(
            "情景 {}（seq {}，{}，rev {}）",
            situation.id,
            situation.seq,
            event.kind.as_str(),
            situation.revision
        )];
        if let Some(focus) = &situation.focus {
            lines.push(format!("焦点：{}", focus));
        }
        if let Some(question) = &situation.question {
            lines.push(format!("问题：{}", question));
        }
        if let Some(answer) = &situation.answer {
            lines.push(format!("回答：{}", answer));
        }
        lines.push(format!("事件：{}", event.label));
        if event.effects.is_empty() {
            lines.push("可枚举效果：无".to_string());
        } else {
            lines.push(format!(
                "可枚举效果（{} 条{}）：",
                event.effect_count,
                if event.enumerable {
                    ""
                } else {
                    "，未完全枚举"
                }
            ));
            for effect in event.effects.iter().take(12) {
                lines.push(format!(
                    "  - {} {}.{} = {}",
                    effect.op.as_str(),
                    effect.entity,
                    effect.predicate,
                    if effect.value.is_empty() {
                        "（空）"
                    } else {
                        &effect.value
                    }
                ));
            }
            if event.effects.len() > 12 {
                lines.push(format!("  …其余 {} 条见导入日志", event.effects.len() - 12));
            }
        }
        if !event.corrections.is_empty() {
            lines.push(format!("更正（{} 条）：", event.corrections.len()));
            for correction in event.corrections.iter().take(8) {
                lines.push(format!(
                    "  - {}.{}：「{}」→「{}」",
                    correction.entity, correction.predicate, correction.from, correction.to
                ));
            }
        }
        if !event.conflicts.is_empty() {
            lines.push(format!("未裁决冲突（{} 条）：", event.conflicts.len()));
            for conflict in event.conflicts.iter().take(8) {
                lines.push(format!(
                    "  - {}.{}：已有「{}」，新值「{}」",
                    conflict.entity,
                    conflict.predicate,
                    conflict.existing.join("、"),
                    conflict.incoming
                ));
            }
        }
        if event.kind == EventKind::Reply {
            let beliefs = self.beliefs_at(seq);
            lines.push(format!(
                "该情景的可枚举信念：{} 条（不是整库快照）。",
                beliefs.len()
            ));
            for belief in beliefs.iter().take(8) {
                lines.push(format!(
                    "  - {}.{} = {}（来源 e{}）",
                    belief.entity, belief.predicate, belief.value, belief.source_seq
                ));
            }
        }
        lines.join("\n")
    }

    pub fn render_history(&self, query: &str) -> String {
        let records = self.matching_answers(query);
        if records.is_empty() {
            return format!("编年史里没有与「{}」匹配的已回答问题。", query.trim());
        }
        let mut lines = vec![format!(
            "情景历史「{}」：{} 次回答（按时间顺序）",
            query.trim(),
            records.len()
        )];
        for record in &records {
            let proof = record
                .proof_id
                .as_deref()
                .map(|id| format!("；证明 {}", id))
                .unwrap_or_default();
            let focus = record
                .focus
                .as_deref()
                .map(|focus| format!("；焦点 {}", focus))
                .unwrap_or_default();
            lines.push(format!(
                "- 第 {} 条（{}, rev {}）：问「{}」→ 答「{}」{}",
                record.reply_seq,
                record.situation,
                record.revision,
                truncate(&record.question, 36),
                truncate(&record.answer, 60),
                format!("{}{}", focus, proof)
            ));
        }
        let changes = self
            .changes
            .iter()
            .filter(|change| {
                records
                    .iter()
                    .any(|record| record.question_key == change.question_key)
            })
            .count();
        lines.push(format!(
            "同一问题改口 {} 处；用「情景归因：{}」查看候选原因。",
            changes,
            query.trim()
        ));
        lines.join("\n")
    }

    pub fn render_changes(&self) -> String {
        if self.changes.is_empty() {
            return "情景理论里没有改口记录（同一问题相邻回答一致）。".to_string();
        }
        let mut lines = vec![format!(
            "情景改口：{} 处（列出最近 12 处，附最多 3 个候选原因）",
            self.changes.len()
        )];
        for change in self.changes.iter().rev().take(12) {
            lines.push(format!(
                "- 「{}」：{}（rev {}，seq {}）→ {}（rev {}，seq {}）；{}",
                truncate(&change.question, 36),
                truncate(&change.previous.answer, 42),
                change.previous.revision,
                change.previous.reply_seq,
                truncate(&change.current.answer, 42),
                change.current.revision,
                change.current.reply_seq,
                change.kind.label()
            ));
            let report = self.causes_for_change(change);
            for candidate in report.candidates.iter().take(3) {
                lines.push(format!(
                    "    ↳ e{} {}（score {}）：{}",
                    candidate.event.seq,
                    candidate.event.kind.as_str(),
                    candidate.score,
                    candidate.reasons.join("；")
                ));
            }
            if report.candidates.is_empty() {
                lines.push(format!("    ↳ {}", report.note));
            }
        }
        lines.join("\n")
    }

    pub fn render_causes(&self, query: &str) -> String {
        let report = self.causes(query);
        let Some(change) = &report.change else {
            return report.note;
        };
        let mut lines = vec![format!(
            "情景归因「{}」：{}（rev {}）→ {}（rev {}）；{}",
            truncate(&change.question, 40),
            truncate(&change.previous.answer, 48),
            change.previous.revision,
            truncate(&change.current.answer, 48),
            change.current.revision,
            change.kind.label()
        )];
        if report.candidates.is_empty() {
            lines.push(report.note.clone());
        } else {
            lines.push("候选原因（按词面命中与取值变化排序）：".to_string());
            for candidate in &report.candidates {
                lines.push(format!(
                    "- e{}（seq {}，{}）：score {}；{}",
                    candidate.event.seq,
                    candidate.event.seq,
                    candidate.event.kind.as_str(),
                    candidate.score,
                    candidate.reasons.join("；")
                ));
                lines.push(format!(
                    "    事件：{}",
                    truncate(&candidate.event.label, 72)
                ));
            }
            lines.push(report.note.clone());
        }
        if report.ignored_events > 0 || report.non_enumerable_events > 0 {
            lines.push(format!(
                "区间内另有 {} 条事件未命中词面、{} 条效果不可枚举。",
                report.ignored_events, report.non_enumerable_events
            ));
        }
        lines.join("\n")
    }

    pub fn render_counterfactual(&self, query: &str) -> String {
        let report = self.counterfactual_undo_latest(query);
        self.render_counterfactual_report(&report)
    }

    pub fn render_counterfactual_report(&self, report: &CounterfactualReport) -> String {
        let mut lines = vec![format!(
            "情景反事实「{}」：{}",
            truncate(&report.question, 40),
            report.verdict.label()
        )];
        if let Some(answer) = &report.answer {
            lines.push(format!(
                "当前回答（第 {} 条，rev {}）：{}",
                answer.reply_seq,
                answer.revision,
                truncate(&answer.answer, 72)
            ));
        }
        if let Some(event) = &report.target_event {
            lines.push(format!(
                "撤销目标：e{}（{}）{}",
                event.seq,
                event.kind.as_str(),
                truncate(&event.label, 72)
            ));
        }
        if !report.affected.is_empty() {
            lines.push("相关可枚举效果：".to_string());
            for effect in report.affected.iter().take(8) {
                lines.push(format!(
                    "  - {} {}.{} = {}",
                    effect.op.as_str(),
                    effect.entity,
                    effect.predicate,
                    if effect.value.is_empty() {
                        "（空）"
                    } else {
                        &effect.value
                    }
                ));
            }
        }
        for reason in &report.reasons {
            lines.push(format!("· {}", reason));
        }
        lines.join("\n")
    }

    pub fn render_query(&self, pattern: &str) -> String {
        match self.query(pattern) {
            Ok(envs) => {
                if envs.is_empty() {
                    return format!("情景理论查询「{}」：无解。", pattern.trim());
                }
                let mut lines = vec![format!(
                    "情景理论查询「{}」：{} 个解",
                    pattern.trim(),
                    envs.len()
                )];
                for env in envs.iter().take(20) {
                    if env.is_empty() {
                        lines.push("  - true".to_string());
                    } else {
                        let binding = env
                            .iter()
                            .map(|(name, value)| format!("{} = {}", name, value))
                            .collect::<Vec<_>>()
                            .join("；");
                        lines.push(format!("  - {}", binding));
                    }
                }
                if envs.len() > 20 {
                    lines.push(format!("  …其余 {} 个解省略", envs.len() - 20));
                }
                lines.join("\n")
            }
            Err(error) => error,
        }
    }
}

fn compute_changes(answers: &[AnswerRecord]) -> Vec<AnswerChange> {
    let mut by_key: BTreeMap<&str, Vec<&AnswerRecord>> = BTreeMap::new();
    for answer in answers {
        if !answer.question_key.is_empty() {
            by_key
                .entry(answer.question_key.as_str())
                .or_default()
                .push(answer);
        }
    }
    let mut changes = Vec::new();
    for records in by_key.values() {
        for pair in records.windows(2) {
            let previous = pair[0];
            let current = pair[1];
            if previous.answer_key == current.answer_key {
                continue;
            }
            let kind = if previous.revision == current.revision {
                ChangeKind::SameRevisionConflict
            } else {
                ChangeKind::RevisionChanged
            };
            changes.push(AnswerChange {
                question: current.question.clone(),
                question_key: current.question_key.clone(),
                previous: previous.clone(),
                current: current.clone(),
                kind,
            });
        }
    }
    changes.sort_by_key(|change| change.current.reply_seq);
    changes
}

#[derive(Debug, Default)]
struct ConstantTable {
    reverse: BTreeMap<String, String>,
}

impl ConstantTable {
    fn intern(&mut self, prefix: &str, text: &str) -> String {
        let text = text.trim();
        let hash = fnv1a_hex(text.as_bytes());
        let sanitized: String = text
            .chars()
            .map(|ch| {
                if ch.is_alphanumeric() || ch == '_' || ch == '-' {
                    ch
                } else {
                    '_'
                }
            })
            .take(32)
            .collect();
        let sanitized = sanitized.trim_matches('_');
        let mut candidate = if sanitized.is_empty() {
            format!("{}_{}", prefix, &hash[..12.min(hash.len())])
        } else if sanitized == text {
            format!("{}_{}", prefix, sanitized)
        } else {
            format!("{}_{}_{}", prefix, sanitized, &hash[..8.min(hash.len())])
        };
        if let Some(existing) = self.reverse.get(&candidate) {
            if existing != text {
                candidate = format!("{}_{}_{}", prefix, sanitized, hash);
            }
        }
        self.reverse.insert(candidate.clone(), text.to_string());
        candidate
    }
}

fn fact(predicate: &str, terms: Vec<String>) -> LAtom {
    LAtom::new(
        predicate,
        terms.into_iter().map(LTerm::atom).collect::<Vec<_>>(),
    )
}

fn compile_program(
    events: &[TheoryEvent],
    answers: &[AnswerRecord],
    changes: &[AnswerChange],
    snapshots: &BTreeMap<u64, Vec<BeliefFact>>,
    max_belief_facts: usize,
) -> (LogicProgram, BTreeMap<String, String>, Option<String>, bool) {
    let mut table = ConstantTable::default();
    let mut program = LogicProgram::new();

    program.add_fact(fact("situation", vec!["s0".to_string()]));
    let mut previous_situation = "s0".to_string();
    for event in events {
        let situation = format!("s{}", event.seq);
        let event_id = format!("e{}", event.seq);
        program.add_fact(fact("situation", vec![situation.clone()]));
        program.add_fact(fact(
            "next_situation",
            vec![previous_situation.clone(), situation.clone()],
        ));
        program.add_fact(fact("event", vec![event_id.clone()]));
        program.add_fact(fact(
            "event_kind",
            vec![event_id.clone(), event.kind.as_str().to_string()],
        ));
        program.add_fact(fact(
            "event_seq",
            vec![event_id.clone(), event.seq.to_string()],
        ));
        program.add_fact(fact(
            "event_after",
            vec![event_id.clone(), situation.clone()],
        ));
        program.add_fact(fact(
            "event_revision",
            vec![event_id.clone(), event.revision.to_string()],
        ));
        program.add_fact(fact(
            "revision_at",
            vec![situation.clone(), event.revision.to_string()],
        ));
        if let Some(focus) = event.focus.as_deref().filter(|focus| !focus.is_empty()) {
            let focus = table.intern("n", focus);
            program.add_fact(fact("focus_at", vec![situation.clone(), focus]));
        }
        for effect in &event.effects {
            let entity = table.intern("n", &effect.entity);
            let predicate = table.intern("n", &effect.predicate);
            let value = table.intern("n", &effect.value);
            program.add_fact(fact(
                "effect",
                vec![
                    event_id.clone(),
                    effect.op.as_str().to_string(),
                    entity.clone(),
                    predicate.clone(),
                    value,
                ],
            ));
        }
        for (entity, predicate) in event.touches() {
            let entity = table.intern("n", &entity);
            let predicate = table.intern("n", &predicate);
            program.add_fact(fact("touches", vec![event_id.clone(), entity, predicate]));
        }
        previous_situation = situation;
    }

    for answer in answers {
        let question = table.intern("q", &answer.question_key);
        let answer_id = format!("a{}", answer.reply_seq);
        let proof = answer
            .proof_id
            .as_deref()
            .filter(|proof| !proof.is_empty())
            .map(|proof| table.intern("p", proof))
            .unwrap_or_else(|| "none".to_string());
        let situation = format!("s{}", answer.reply_seq);
        program.add_fact(fact(
            "asked_at",
            vec![
                format!("s{}", answer.question_seq),
                question.clone(),
                format!("e{}", answer.question_seq),
            ],
        ));
        program.add_fact(fact(
            "answered_at",
            vec![
                situation.clone(),
                question.clone(),
                answer_id.clone(),
                proof,
            ],
        ));
        program.add_fact(fact(
            "answer_at",
            vec![answer_id.clone(), situation.clone()],
        ));
        program.add_fact(fact(
            "answer_text",
            vec![answer_id, table.intern("a", &answer.answer)],
        ));
    }

    let mut by_key: BTreeMap<&str, Vec<&AnswerRecord>> = BTreeMap::new();
    for answer in answers {
        by_key
            .entry(answer.question_key.as_str())
            .or_default()
            .push(answer);
    }
    for (key, records) in &by_key {
        let question = table.intern("q", key);
        for pair in records.windows(2) {
            let first = format!("a{}", pair[0].reply_seq);
            let second = format!("a{}", pair[1].reply_seq);
            program.add_fact(fact(
                "next_answer",
                vec![question.clone(), first.clone(), second.clone()],
            ));
            if pair[0].answer_key != pair[1].answer_key {
                program.add_fact(fact("answer_differ", vec![first, second]));
            }
        }
    }

    for change in changes {
        let question = table.intern("q", &change.question_key);
        let first = format!("a{}", change.previous.reply_seq);
        let second = format!("a{}", change.current.reply_seq);
        program.add_fact(fact("answer_change", vec![question.clone(), first, second]));
        program.add_fact(fact(
            "change_at",
            vec![
                question.clone(),
                format!("s{}", change.previous.reply_seq),
                format!("s{}", change.current.reply_seq),
            ],
        ));
        for event in events {
            if !event.kind.is_causal_candidate() {
                continue;
            }
            if event.seq <= change.previous.reply_seq || event.seq >= change.current.reply_seq {
                continue;
            }
            if event.touches().is_empty() && event.kind != EventKind::Rule {
                continue;
            }
            program.add_fact(fact(
                "candidate_cause",
                vec![format!("e{}", event.seq), question.clone()],
            ));
        }
    }

    let mut belief_facts = 0usize;
    let mut truncated_belief_facts = false;
    'snapshots: for (seq, snapshot) in snapshots {
        let situation = format!("s{}", seq);
        for belief in snapshot {
            if belief_facts >= max_belief_facts {
                truncated_belief_facts = true;
                program.add_fact(fact("theory_truncated", vec!["belief_at".to_string()]));
                break 'snapshots;
            }
            let entity = table.intern("n", &belief.entity);
            let predicate = table.intern("n", &belief.predicate);
            let value = table.intern("n", &belief.value);
            let source = format!("e{}", belief.source_seq);
            program.add_fact(fact(
                "belief_at",
                vec![situation.clone(), entity, predicate, value, source],
            ));
            belief_facts += 1;
        }
    }
    program.add_fact(fact(
        "theory_window",
        vec![
            events
                .first()
                .map(|event| event.seq)
                .unwrap_or(0)
                .to_string(),
            events
                .last()
                .map(|event| event.seq)
                .unwrap_or(0)
                .to_string(),
            events.len().to_string(),
        ],
    ));

    for source in [
        "before(S1, S2) :- next_situation(S1, S2).",
        "before(S1, S2) :- next_situation(S1, S3), before(S3, S2).",
        "answer_change(Q, A1, A2) :- next_answer(Q, A1, A2), answer_differ(A1, A2).",
        "change_at(Q, S1, S2) :- answer_change(Q, A1, A2), answer_at(A1, S1), answer_at(A2, S2).",
    ] {
        match parse_rule(source) {
            Ok(rule) => {
                let _ = program.add_rule(rule);
            }
            Err(error) => {
                return (
                    program,
                    table.reverse,
                    Some(format!("理论规则解析失败：{}", error)),
                    truncated_belief_facts,
                )
            }
        }
    }
    let evaluation_error = program.evaluate().err();
    (
        program,
        table.reverse,
        evaluation_error,
        truncated_belief_facts,
    )
}

/// 从导入回执提取可折叠效果（IRI 本地名化，保留可读实体/谓词）。
pub fn receipt_effects(receipt: &ImportReceipt, op: EffectOp) -> Vec<TheoryEffect> {
    let mut effects = BTreeSet::new();
    for action in &receipt.actions {
        match action {
            ImportAction::Named {
                subject,
                predicate,
                object,
            } => {
                effects.insert(TheoryEffect {
                    op,
                    entity: local_name(subject),
                    predicate: local_name(predicate),
                    value: local_name(object),
                });
            }
            ImportAction::Literal {
                subject,
                predicate,
                value,
            } => {
                effects.insert(TheoryEffect {
                    op,
                    entity: local_name(subject),
                    predicate: local_name(predicate),
                    value: value.clone(),
                });
            }
        }
    }
    effects.into_iter().collect()
}

/// 从可解析的中文断言提取效果（只取确定、基事实；其余诚实留空）。
pub fn effects_from_text(text: &str, op: EffectOp) -> Vec<TheoryEffect> {
    let outcome = parse_lessons(text.trim());
    let mut effects = BTreeSet::new();
    for lesson in &outcome.lessons {
        if lesson.kind != LessonKind::DeterministicFact || !lesson.head.is_ground() {
            continue;
        }
        let atom = &lesson.head;
        match atom.terms.as_slice() {
            [subject, object] => {
                effects.insert(TheoryEffect {
                    op,
                    entity: subject.name().to_string(),
                    predicate: atom.predicate.to_string(),
                    value: object.name().to_string(),
                });
            }
            [subject] => {
                effects.insert(TheoryEffect {
                    op,
                    entity: subject.name().to_string(),
                    predicate: atom.predicate.to_string(),
                    value: String::new(),
                });
            }
            _ => {}
        }
        if effects.len() >= MAX_EFFECTS_PER_EVENT {
            break;
        }
    }
    effects.into_iter().collect()
}

fn local_name(iri: &str) -> String {
    if iri == "http://www.w3.org/1999/02/22-rdf-syntax-ns#type" {
        return "type".to_string();
    }
    if let Some(name) = crate::identity::local_name(crate::identity::DEFAULT_BASE_NS, iri) {
        return name;
    }
    iri.rsplit(|ch| ch == '#' || ch == '/')
        .next()
        .unwrap_or(iri)
        .to_string()
}

fn effects_json(effects: &[TheoryEffect]) -> Vec<Value> {
    effects
        .iter()
        .map(|effect| {
            json!({
                "op": effect.op.as_str(),
                "entity": effect.entity,
                "predicate": effect.predicate,
                "value": effect.value,
            })
        })
        .collect()
}

fn receipt_digest(receipt: &ImportReceipt) -> String {
    let effects = receipt_effects(receipt, EffectOp::Add);
    fnv1a_hex(canonical_json(&Value::Array(effects_json(&effects))).as_bytes())
}

fn effects_digest(effects: &[TheoryEffect]) -> String {
    fnv1a_hex(canonical_json(&Value::Array(effects_json(effects))).as_bytes())
}

fn event_payload(
    summary: &str,
    source: &str,
    revision_after: u64,
    effects: &[TheoryEffect],
    enumerable: bool,
    digest: Option<String>,
    removes_digest: Option<String>,
    effect_count: usize,
) -> Value {
    let mut payload = json!({
        "summary": summary.lines().next().unwrap_or(summary),
        "source": source,
        "revision_after": revision_after,
        "effects": effects_json(effects),
        "effect_count": effect_count,
        "effects_truncated": effect_count > effects.len(),
        "enumerable": enumerable && !effects.is_empty() && effect_count == effects.len(),
    });
    if let Some(digest) = digest {
        payload["digest"] = Value::from(digest);
    }
    if let Some(digest) = removes_digest {
        payload["removes_digest"] = Value::from(digest);
    }
    payload
}

/// 追加文本导入事件（效果可枚举，可反事实重放）。
pub fn append_import(
    chronicle: &mut Chronicle,
    receipt: &ImportReceipt,
    source: &str,
    revision_after: u64,
) -> io::Result<u64> {
    let all_effects = receipt_effects(receipt, EffectOp::Add);
    let effect_count = all_effects.len();
    let effects: Vec<TheoryEffect> = all_effects
        .iter()
        .take(MAX_EFFECTS_PER_EVENT)
        .cloned()
        .collect();
    let digest = receipt_digest(receipt);
    chronicle.append(
        "import",
        event_payload(
            &receipt.summary(),
            source,
            revision_after,
            &effects,
            true,
            Some(digest),
            None,
            effect_count,
        ),
    )
}

/// 追加导入冲突事件（保留被拒入值；不改变可枚举信念）。
pub fn append_conflict(
    chronicle: &mut Chronicle,
    conflicts: &[crate::predicate_registry::ImportConflict],
    source: &str,
    revision_after: u64,
) -> io::Result<u64> {
    let items: Vec<Value> = conflicts
        .iter()
        .map(|conflict| {
            json!({
                "entity": local_name(&conflict.subject),
                "predicate": local_name(&conflict.predicate),
                "existing": conflict.existing.iter().map(|value| local_name(value)).collect::<Vec<_>>(),
                "incoming": local_name(&conflict.incoming),
            })
        })
        .collect();
    let first = conflicts.first();
    let summary = first
        .map(|conflict| {
            format!(
                "导入冲突：{}.{} 保留「{}」，拒绝「{}」",
                local_name(&conflict.subject),
                local_name(&conflict.predicate),
                conflict.existing.join("、"),
                conflict.incoming
            )
        })
        .unwrap_or_else(|| "导入冲突".to_string());
    let mut payload = event_payload(
        &summary,
        source,
        revision_after,
        &[],
        false,
        None,
        None,
        0,
    );
    payload["conflicts"] = Value::Array(items);
    chronicle.append("conflict", payload)
}

/// 追加更正事件：旧值被判错，新值生效（效果可枚举、可反事实重放）。
pub fn append_correction(
    chronicle: &mut Chronicle,
    text: &str,
    corrections: &[TheoryCorrection],
    revision_after: u64,
) -> io::Result<u64> {
    let mut effects = BTreeSet::new();
    for correction in corrections {
        if !correction.from.is_empty() && correction.from != correction.to {
            effects.insert(TheoryEffect::remove(
                &correction.entity,
                &correction.predicate,
                &correction.from,
            ));
        }
        if !correction.to.is_empty() {
            effects.insert(TheoryEffect::add(
                &correction.entity,
                &correction.predicate,
                &correction.to,
            ));
        }
    }
    let effects: Vec<TheoryEffect> = effects.into_iter().collect();
    let effect_count = effects.len();
    let digest = effects_digest(&effects);
    let mut payload = event_payload(
        text,
        "用户更正",
        revision_after,
        &effects,
        true,
        Some(digest),
        None,
        effect_count,
    );
    payload["corrections"] = Value::Array(
        corrections
            .iter()
            .map(|correction| {
                json!({
                    "entity": correction.entity,
                    "predicate": correction.predicate,
                    "from": correction.from,
                    "to": correction.to,
                })
            })
            .collect(),
    );
    chronicle.append("correct", payload)
}

/// 追加上次导入的撤销事件（与导入共享回执摘要，可精确移除来源）。
pub fn append_retract(
    chronicle: &mut Chronicle,
    receipt: &ImportReceipt,
    source: &str,
    revision_after: u64,
) -> io::Result<u64> {
    let all_effects = receipt_effects(receipt, EffectOp::Remove);
    let effect_count = all_effects.len();
    let effects: Vec<TheoryEffect> = all_effects
        .iter()
        .take(MAX_EFFECTS_PER_EVENT)
        .cloned()
        .collect();
    let digest = receipt_digest(receipt);
    chronicle.append(
        "retract",
        event_payload(
            &format!(
                "撤销导入：{}",
                receipt.summary().lines().next().unwrap_or("")
            ),
            source,
            revision_after,
            &effects,
            true,
            Some(digest.clone()),
            Some(digest),
            effect_count,
        ),
    )
}

/// 追加批量导入事件（JSON-LD/RDF：新增条数已知，逐条效果未入编年史）。
pub fn append_bulk_import(
    chronicle: &mut Chronicle,
    source: &str,
    target: &str,
    added: usize,
    revision_after: u64,
) -> io::Result<u64> {
    chronicle.append(
        "import",
        event_payload(
            &format!("{}：{} 新增 {} 条", source, target, added),
            source,
            revision_after,
            &[],
            false,
            None,
            None,
            added,
        ),
    )
}

/// 追加用户断言事件。
pub fn append_assert(
    chronicle: &mut Chronicle,
    text: &str,
    effects: &[TheoryEffect],
    revision_after: u64,
) -> io::Result<u64> {
    let effect_count = effects.len();
    let effects: Vec<TheoryEffect> = effects
        .iter()
        .take(MAX_EFFECTS_PER_EVENT)
        .cloned()
        .collect();
    let digest = effects_digest(&effects);
    chronicle.append(
        "assert",
        event_payload(
            text,
            "用户断言",
            revision_after,
            &effects,
            true,
            Some(digest),
            None,
            effect_count,
        ),
    )
}

/// 追加用户撤回断言事件。
pub fn append_forget(
    chronicle: &mut Chronicle,
    text: &str,
    effects: &[TheoryEffect],
    revision_after: u64,
) -> io::Result<u64> {
    let effect_count = effects.len();
    let effects: Vec<TheoryEffect> = effects
        .iter()
        .take(MAX_EFFECTS_PER_EVENT)
        .cloned()
        .collect();
    let target_digest = effects_digest(&effects);
    chronicle.append(
        "forget",
        event_payload(
            text,
            "用户撤回断言",
            revision_after,
            &effects,
            true,
            Some(target_digest.clone()),
            Some(target_digest),
            effect_count,
        ),
    )
}

/// 追加规则教学事件（效果不可完全枚举）。
pub fn append_rule(
    chronicle: &mut Chronicle,
    text: &str,
    learned: bool,
    revision_after: u64,
) -> io::Result<u64> {
    chronicle.append(
        "rule",
        event_payload(
            text,
            if learned {
                "规则教学（已学习）"
            } else {
                "规则教学（未解析）"
            },
            revision_after,
            &[],
            false,
            None,
            None,
            0,
        ),
    )
}

/// 追加外链事件（缓存/刷新/遗忘/别名；效果不由本地编年史枚举）。
pub fn append_external(
    chronicle: &mut Chronicle,
    entity: &str,
    action: &str,
    source_uri: &str,
    revision_after: u64,
) -> io::Result<u64> {
    chronicle.append(
        "external",
        event_payload(
            &format!("外链 {}：{}", action, entity),
            source_uri,
            revision_after,
            &[],
            false,
            None,
            None,
            0,
        ),
    )
}

/// 追加修复事件（截尾重写等）。
pub fn append_repair(chronicle: &mut Chronicle, detail: &str) -> io::Result<u64> {
    chronicle.append(
        "repair",
        event_payload(detail, "编年史修复", 0, &[], false, None, None, 0),
    )
}

fn truncate(text: &str, limit: usize) -> String {
    let mut out: String = text.chars().take(limit).collect();
    if text.chars().count() > limit {
        out.push('…');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chronicle::FsyncPolicy;
    use crate::predicate_registry::{ImportAction, ImportReceipt};
    use tempfile::TempDir;

    fn chronicle(dir: &TempDir) -> (Chronicle, String) {
        let path = dir.path().join("theory.chronicle.jsonl");
        let path_text = path.to_string_lossy().to_string();
        let (chronicle, audit) = Chronicle::open(&path_text, FsyncPolicy::None).unwrap();
        assert!(audit.is_clean());
        (chronicle, path_text)
    }

    fn receipt(subject: &str, predicate: &str, value: &str) -> ImportReceipt {
        ImportReceipt {
            added: 1,
            actions: vec![ImportAction::Literal {
                subject: format!("http://example.org/kg/entity/{}", subject),
                predicate: format!("http://example.org/kg/{}", predicate),
                value: value.to_string(),
            }],
            ..Default::default()
        }
    }

    fn ask(chronicle: &mut Chronicle, question: &str) {
        chronicle
            .append_turn(question, "query", Some("姚明"), 0)
            .unwrap();
    }

    fn answer(chronicle: &mut Chronicle, text: &str) -> u64 {
        let seq = chronicle.append_reply(text, &[], &[]).unwrap();
        seq
    }

    #[test]
    fn compiles_situations_answers_and_change_rules() {
        let dir = TempDir::new().unwrap();
        let (mut chronicle, _) = chronicle(&dir);
        ask(&mut chronicle, "姚明的身高是多少？");
        answer(&mut chronicle, "姚明身高 226 厘米。");
        chronicle.set_kb_revision(1);
        ask(&mut chronicle, "姚明的身高是多少呢");
        answer(&mut chronicle, "姚明身高 229 厘米。");

        let theory = ConversationTheory::build(&chronicle);
        assert_eq!(theory.answers().len(), 2);
        assert_eq!(theory.changes().len(), 1);
        assert!(theory.evaluation_error().is_none());
        assert_eq!(
            theory
                .program()
                .query("answered_at(S, Q, A, P)")
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            theory
                .program()
                .query("answer_change(Q, A1, A2)")
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            theory
                .program()
                .query("change_at(Q, S1, S2)")
                .unwrap()
                .len(),
            1
        );
        assert_eq!(theory.program().query("before(s0, s4)").unwrap().len(), 1);
        assert!(theory.render_report().contains("改口 1 处"));
    }

    #[test]
    fn import_effects_are_folded_and_attributed() {
        let dir = TempDir::new().unwrap();
        let (mut chronicle, _) = chronicle(&dir);
        ask(&mut chronicle, "姚明的身高是多少？");
        answer(&mut chronicle, "姚明身高 226 厘米。");
        let import_receipt = receipt("姚明", "height", "229");
        append_import(&mut chronicle, &import_receipt, "文本录入", 1).unwrap();
        chronicle.set_kb_revision(1);
        ask(&mut chronicle, "姚明的身高是多少？");
        answer(&mut chronicle, "姚明身高 229 厘米。");

        let theory = ConversationTheory::build(&chronicle);
        assert_eq!(theory.changes().len(), 1);
        let report = theory.causes("姚明的身高是多少");
        assert_eq!(
            report.change.as_ref().unwrap().previous.answer,
            "姚明身高 226 厘米。"
        );
        assert_eq!(report.candidates.len(), 1, "{:?}", report);
        assert!(report.candidates[0]
            .reasons
            .iter()
            .any(|reason| reason.contains("229")));
        assert!(theory
            .query("belief_at(S, E, P, V, Src)")
            .unwrap()
            .iter()
            .any(|row| row.get("V").map(String::as_str) == Some("229")));
        assert_eq!(
            theory
                .program()
                .query("candidate_cause(E, Q)")
                .unwrap()
                .len(),
            1
        );
        assert!(!theory.render_causes("姚明的身高是多少").is_empty());
    }

    #[test]
    fn non_enumerable_events_are_ignored_for_attribution() {
        let dir = TempDir::new().unwrap();
        let (mut chronicle, _) = chronicle(&dir);
        ask(&mut chronicle, "姚明的身高是多少？");
        answer(&mut chronicle, "姚明身高 226 厘米。");
        append_bulk_import(&mut chronicle, "JSON-LD", "test.jsonld", 12, 1).unwrap();
        chronicle.set_kb_revision(1);
        ask(&mut chronicle, "姚明的身高是多少？");
        answer(&mut chronicle, "姚明身高 229 厘米。");

        let theory = ConversationTheory::build(&chronicle);
        let report = theory.causes("姚明的身高是多少");
        assert!(report.candidates.is_empty());
        assert_eq!(report.non_enumerable_events, 1);
        assert!(report.note.contains("不可枚举"));
    }

    #[test]
    fn counterfactual_flips_when_only_source_is_removed() {
        let dir = TempDir::new().unwrap();
        let (mut chronicle, _) = chronicle(&dir);
        ask(&mut chronicle, "姚明的身高是多少？");
        answer(&mut chronicle, "姚明身高 226 厘米。");
        let import_receipt = receipt("姚明", "height", "229");
        append_import(&mut chronicle, &import_receipt, "文本录入", 1).unwrap();
        chronicle.set_kb_revision(1);
        ask(&mut chronicle, "姚明的身高是多少？");
        answer(&mut chronicle, "姚明身高 229 厘米。");

        let theory = ConversationTheory::build(&chronicle);
        let report = theory.counterfactual_undo_latest("姚明的身高是多少");
        assert_eq!(report.verdict, CounterfactualVerdict::Flips, "{:?}", report);
        assert!(!report.affected.is_empty());
        assert!(theory
            .render_counterfactual("姚明的身高是多少")
            .contains("会翻转"));
    }

    #[test]
    fn counterfactual_keeps_alternative_support() {
        let dir = TempDir::new().unwrap();
        let (mut chronicle, _) = chronicle(&dir);
        let first = receipt("姚明", "height", "229");
        append_import(&mut chronicle, &first, "导入甲", 1).unwrap();
        let second = receipt("姚明", "height", "229");
        append_import(&mut chronicle, &second, "导入乙", 2).unwrap();
        chronicle.set_kb_revision(2);
        ask(&mut chronicle, "姚明的身高是多少？");
        answer(&mut chronicle, "姚明身高 229 厘米。");

        let theory = ConversationTheory::build(&chronicle);
        let report = theory.counterfactual_undo_latest("姚明的身高是多少");
        assert_eq!(
            report.verdict,
            CounterfactualVerdict::Unchanged,
            "{:?}",
            report
        );
        assert!(report.remaining_support >= 1);
    }

    #[test]
    fn counterfactual_abstains_for_non_enumerable_event() {
        let dir = TempDir::new().unwrap();
        let (mut chronicle, _) = chronicle(&dir);
        append_rule(&mut chronicle, "如果X的师傅是Y，那么X的老师是Y", true, 0).unwrap();
        ask(&mut chronicle, "姚明的身高是多少？");
        answer(&mut chronicle, "姚明身高 229 厘米。");

        let theory = ConversationTheory::build(&chronicle);
        let report = theory.counterfactual_undo_latest("姚明的身高是多少");
        assert_eq!(
            report.verdict,
            CounterfactualVerdict::Unknown,
            "{:?}",
            report
        );
        assert!(report
            .reasons
            .iter()
            .any(|reason| reason.contains("不可完全枚举")));
    }

    #[test]
    fn retract_removes_only_matching_import_source() {
        let dir = TempDir::new().unwrap();
        let (mut chronicle, _) = chronicle(&dir);
        let shared = receipt("姚明", "height", "229");
        append_import(&mut chronicle, &shared, "导入甲", 1).unwrap();
        append_import(&mut chronicle, &shared, "导入乙", 2).unwrap();
        append_retract(&mut chronicle, &shared, "撤销导入", 3).unwrap();

        let theory = ConversationTheory::build(&chronicle);
        let beliefs = theory.beliefs_at(chronicle.last_seq());
        assert_eq!(beliefs.len(), 1, "{:?}", beliefs);
        assert_eq!(beliefs[0].value, "229");
    }

    #[test]
    fn forget_removes_matching_assertion() {
        let dir = TempDir::new().unwrap();
        let (mut chronicle, _) = chronicle(&dir);
        let effects = effects_from_text("姚明的身高是229", EffectOp::Add);
        assert!(!effects.is_empty());
        append_assert(&mut chronicle, "姚明的身高是229", &effects, 0).unwrap();
        append_forget(
            &mut chronicle,
            "姚明的身高是229",
            &effects_from_text("姚明的身高是229", EffectOp::Remove),
            0,
        )
        .unwrap();
        let theory = ConversationTheory::build(&chronicle);
        assert!(theory.beliefs_at(chronicle.last_seq()).is_empty());
    }

    #[test]
    fn local_names_are_decoded_from_entity_iris() {
        assert_eq!(
            local_name("http://example.org/kg/entity/%E5%A7%9A%20%E6%98%8E"),
            "姚 明"
        );
        assert_eq!(local_name("http://example.org/kg/height"), "height");
    }

    #[test]
    fn effects_from_text_extracts_ground_facts() {
        let effects = effects_from_text("姚明的身高是226", EffectOp::Add);
        assert!(!effects.is_empty(), "{:?}", effects);
        assert!(effects.iter().any(|effect| effect.value == "226"));
        let forget = effects_from_text("姚明的身高是226", EffectOp::Remove);
        assert_eq!(effects.len(), forget.len());
        assert_eq!(forget[0].op, EffectOp::Remove);
    }

    #[test]
    fn unknown_kind_is_preserved_as_other() {
        let dir = TempDir::new().unwrap();
        let (mut chronicle, _) = chronicle(&dir);
        chronicle
            .append("note", json!({"text": "保持未知事件"}))
            .unwrap();
        let theory = ConversationTheory::build(&chronicle);
        assert_eq!(theory.events().len(), 1);
        assert_eq!(theory.events()[0].kind, EventKind::Other);
        assert!(theory.render_report().contains("other 1"));
    }
}

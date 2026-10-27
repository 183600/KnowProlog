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

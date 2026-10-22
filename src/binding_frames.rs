//! 配置编译的绑定帧（Config-Compiled Binding Frames）。
//!
//! 把「一句话能绑定哪些变量、变量如何跨轮继承、缺槽时问什么、绑定完成后
//! 查什么、怎么措辞」从 Rust 硬编码搬到 config.toml，并在启动时编译成
//! 不可变的 [`FrameProgram`]。Rust 只负责通用的匹配、合一、执行与生成，
//! 不再认识「身高」「首都」「师傅」这些具体词。
//!
//! 语义约定：
//! - 模式里的 `{slot}` 是带排序的变量，字面量是锚点；
//! - 一次回答是槽位全部接地的绑定，一次追问是部分绑定；
//! - 澄清把自由变量显式还给用户，用户短答后完成一次绑定；
//! - 答案模板只能渲染已声明且已接地的槽位，生成不会凭空造值；
//! - 任何一步失败都零状态变化，调用方回退常规管线。

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::dialogue_manager::{DialogueKnowledge, MoveKind};
use crate::quantity::{self, Comparison, Quantity};

/// 继承槽位的逻辑轮次时限。
pub const FRAME_ENV_TTL: u64 = 8;
/// 待澄清帧的逻辑轮次时限。
pub const FRAME_PENDING_TTL: u64 = 3;
/// 单次澄清最多展示的候选数。
pub const FRAME_MAX_CANDIDATES: usize = 8;

// ----------------------------- 配置层 -----------------------------

/// `[dialogue_frames]`：默认关闭；关闭时编译仍会做静态检查，但运行时不接管。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DialogueFramesConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub frames: Vec<FrameSchemaConfig>,
}

/// 一个帧声明：模式、槽位、目标与措辞。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameSchemaConfig {
    pub id: String,
    #[serde(default)]
    pub priority: i32,
    #[serde(default)]
    pub patterns: Vec<String>,
    #[serde(default)]
    pub slots: Vec<FrameSlotConfig>,
    pub goal: FrameGoalConfig,
    #[serde(default)]
    pub realize: FrameRealizeConfig,
}

/// 槽位排序：决定捕获值如何归一，以及澄清候选从哪里来。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SlotSort {
    #[default]
    Text,
    Entity,
    Predicate,
    Attribute,
    Number,
}

impl SlotSort {
    fn label(self) -> &'static str {
        match self {
            SlotSort::Text => "信息",
            SlotSort::Entity => "实体",
            SlotSort::Predicate => "关系",
            SlotSort::Attribute => "属性",
            SlotSort::Number => "数值",
        }
    }
}

/// 澄清候选来源。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CandidateSource {
    /// 不提供候选，接受用户自由短答（仍按槽位排序校验）。
    #[default]
    None,
    /// 图谱中的全部实体。
    Entities,
    /// 当前实体已知的属性/关系词面。
    Attributes,
    /// 与 `Attributes` 同源；保留独立名字便于配置表达意图。
    Predicates,
}

/// 槽位声明。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameSlotConfig {
    pub name: String,
    #[serde(default)]
    pub sort: SlotSort,
    /// 必填输入槽：缺了就澄清（或按继承补全）。
    #[serde(default)]
    pub required: bool,
    /// 是否允许从会话绑定环境继承（显式值优先覆盖）。
    #[serde(default)]
    pub inherit: bool,
    /// 澄清提示模板，支持 `{raw}` / `{slot}`。
    #[serde(default)]
    pub ask: Option<String>,
    #[serde(default)]
    pub candidates: CandidateSource,
}

/// 结构化目标：只引用槽名或 `const:` 常量，不拼查询字符串。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FrameGoalConfig {
    Attribute {
        entity: String,
        attribute: String,
        #[serde(default = "default_value_output")]
        output: String,
    },
    Relation {
        subject: String,
        predicate: String,
        #[serde(default = "default_object_output")]
        output: String,
    },
    ReverseRelation {
        object: String,
        predicate: String,
        #[serde(default = "default_subject_output")]
        output: String,
    },
    Count {
        subject: String,
        predicate: String,
        #[serde(default = "default_count_output")]
        output: String,
    },
    Compare {
        left: String,
        right: String,
        attribute: String,
        #[serde(default = "default_winner_output")]
        output: String,
    },
}

fn default_value_output() -> String {
    "value".to_string()
}

fn default_object_output() -> String {
    "object".to_string()
}

fn default_subject_output() -> String {
    "subject".to_string()
}

fn default_count_output() -> String {
    "count".to_string()
}

fn default_winner_output() -> String {
    "winner".to_string()
}

/// 答案措辞：`answer` 为命中模板，`empty` 为澄清绑定后仍无值的兜底。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameRealizeConfig {
    #[serde(default)]
    pub answer: Option<String>,
    #[serde(default)]
    pub empty: Option<String>,
}

// ----------------------------- 编译层 -----------------------------

/// 一条编译期错误：稳定机器码 + 配置坐标 + 人话。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameCompileError {
    pub code: &'static str,
    pub path: String,
    pub message: String,
}

impl fmt::Display for FrameCompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}：{}", self.code, self.path, self.message)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Operand {
    Slot(String),
    Const(String),
}

impl Operand {
    fn slot(&self) -> Option<&str> {
        match self {
            Operand::Slot(name) => Some(name),
            Operand::Const(_) => None,
        }
    }

    fn resolve(&self, bindings: &BTreeMap<String, String>) -> Option<String> {
        match self {
            Operand::Slot(name) => bindings.get(name).cloned(),
            Operand::Const(value) => Some(value.clone()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CompiledGoal {
    Attribute {
        entity: Operand,
        attribute: Operand,
        output: String,
    },
    Relation {
        subject: Operand,
        predicate: Operand,
        output: String,
    },
    ReverseRelation {
        object: Operand,
        predicate: Operand,
        output: String,
    },
    Count {
        subject: Operand,
        predicate: Operand,
        output: String,
    },
    Compare {
        left: Operand,
        right: Operand,
        attribute: Operand,
        output: String,
    },
}

impl CompiledGoal {
    fn kind_label(&self) -> &'static str {
        match self {
            CompiledGoal::Attribute { .. } => "attribute",
            CompiledGoal::Relation { .. } => "relation",
            CompiledGoal::ReverseRelation { .. } => "reverse_relation",
            CompiledGoal::Count { .. } => "count",
            CompiledGoal::Compare { .. } => "compare",
        }
    }

    fn output_slot(&self) -> &str {
        match self {
            CompiledGoal::Attribute { output, .. }
            | CompiledGoal::Relation { output, .. }
            | CompiledGoal::ReverseRelation { output, .. }
            | CompiledGoal::Count { output, .. }
            | CompiledGoal::Compare { output, .. } => output,
        }
    }

    /// 目标读取的输入槽（常量不算）。
    fn input_slots(&self) -> Vec<String> {
        let operands: Vec<&Operand> = match self {
            CompiledGoal::Attribute {
                entity, attribute, ..
            } => vec![entity, attribute],
            CompiledGoal::Relation {
                subject, predicate, ..
            } => vec![subject, predicate],
            CompiledGoal::ReverseRelation {
                object, predicate, ..
            } => vec![object, predicate],
            CompiledGoal::Count {
                subject, predicate, ..
            } => vec![subject, predicate],
            CompiledGoal::Compare {
                left,
                right,
                attribute,
                ..
            } => vec![left, right, attribute],
        };
        operands
            .into_iter()
            .filter_map(|operand| operand.slot().map(str::to_string))
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum PatternPart {
    Literal(String),
    Slot(String),
}

#[derive(Debug, Clone)]
struct CompiledPattern {
    source: String,
    parts: Vec<PatternPart>,
    slots: Vec<String>,
    literal_chars: usize,
}

impl CompiledPattern {
    fn matches(&self, input: &str) -> Option<BTreeMap<String, String>> {
        let mut rest = input;
        let mut captures = BTreeMap::new();
        for (index, part) in self.parts.iter().enumerate() {
            match part {
                PatternPart::Literal(literal) => {
                    rest = rest.strip_prefix(literal.as_str())?;
                }
                PatternPart::Slot(name) => {
                    let next_literal = self.parts[index + 1..].iter().find_map(|part| match part {
                        PatternPart::Literal(literal) => Some(literal.as_str()),
                        PatternPart::Slot(_) => None,
                    });
                    match next_literal {
                        Some(literal) => {
                            let position = rest.find(literal)?;
                            if position == 0 {
                                return None;
                            }
                            captures.insert(name.clone(), rest[..position].to_string());
                            rest = &rest[position..];
                        }
                        None => {
                            if rest.is_empty() {
                                return None;
                            }
                            captures.insert(name.clone(), rest.to_string());
                            rest = "";
                        }
                    }
                }
            }
        }
        rest.is_empty().then_some(captures)
    }
}

#[derive(Debug, Clone)]
struct CompiledFrame {
    id: String,
    priority: i32,
    patterns: Vec<CompiledPattern>,
    slots: BTreeMap<String, FrameSlotConfig>,
    slot_order: Vec<String>,
    goal: CompiledGoal,
    realize_answer: Option<String>,
    realize_empty: Option<String>,
}

impl CompiledFrame {
    fn is_input_slot(&self, name: &str) -> bool {
        self.goal.input_slots().iter().any(|slot| slot == name)
            || self
                .patterns
                .iter()
                .any(|pattern| pattern.slots.iter().any(|slot| slot == name))
    }
}

/// 编译后的不可变程序：同一份配置编译出同一种匹配顺序。
#[derive(Debug, Clone, Default)]
pub struct FrameProgram {
    enabled: bool,
    frames: Vec<CompiledFrame>,
    ranked: Vec<(usize, usize)>,
}

impl FrameProgram {
    pub fn compile(config: &DialogueFramesConfig) -> Result<Self, Vec<FrameCompileError>> {
        let mut errors = Vec::new();
        let mut frames = Vec::new();
        let mut seen_ids = BTreeSet::new();

        for (frame_index, schema) in config.frames.iter().enumerate() {
            let frame = compile_frame(frame_index, schema, &mut seen_ids, &mut errors);
            if let Some(frame) = frame {
                frames.push(frame);
            }
        }

        if !errors.is_empty() {
            return Err(errors);
        }

        let mut ranked = Vec::new();
        for (frame_index, frame) in frames.iter().enumerate() {
            for pattern_index in 0..frame.patterns.len() {
                ranked.push((frame_index, pattern_index));
            }
        }
        ranked.sort_by(|(a_frame, a_pattern), (b_frame, b_pattern)| {
            let a = &frames[*a_frame];
            let b = &frames[*b_frame];
            b.priority
                .cmp(&a.priority)
                .then_with(|| {
                    b.patterns[*b_pattern]
                        .literal_chars
                        .cmp(&a.patterns[*a_pattern].literal_chars)
                })
                .then_with(|| {
                    a.patterns[*a_pattern]
                        .slots
                        .len()
                        .cmp(&b.patterns[*b_pattern].slots.len())
                })
                .then_with(|| a.id.cmp(&b.id))
                .then_with(|| {
                    a.patterns[*a_pattern]
                        .source
                        .cmp(&b.patterns[*b_pattern].source)
                })
        });

        Ok(FrameProgram {
            enabled: config.enabled,
            frames,
            ranked,
        })
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    pub fn pattern_count(&self) -> usize {
        self.ranked.len()
    }

    pub fn frame_ids(&self) -> Vec<&str> {
        self.frames.iter().map(|frame| frame.id.as_str()).collect()
    }

    fn frame(&self, id: &str) -> Option<&CompiledFrame> {
        self.frames.iter().find(|frame| frame.id == id)
    }

    /// 规划一轮：先结算待澄清帧，再按确定序尝试模式；执行成功才提交状态。
    pub fn plan(
        &self,
        memory: &mut FrameMemory,
        input: &str,
        knowledge: &dyn DialogueKnowledge,
        turn: u64,
    ) -> Option<FrameOutcome> {
        if !self.enabled || self.frames.is_empty() {
            return None;
        }
        let normalized = normalize_utterance(input);
        if normalized.is_empty() {
            return None;
        }
        memory.prune(turn);
        if memory.pending.is_some() {
            if let Some(outcome) = self.resolve_pending(memory, &normalized, knowledge, turn) {
                return Some(outcome);
            }
        }

        let mut question: Option<QuestionPlan> = None;
        for (frame_index, pattern_index) in &self.ranked {
            let frame = &self.frames[*frame_index];
            let pattern = &frame.patterns[*pattern_index];
            let Some(captures) = pattern.matches(&normalized) else {
                continue;
            };
            let Some(resolved) = self.resolve_frame(frame, &captures, memory, knowledge, turn)
            else {
                continue;
            };
            if !resolved.missing.is_empty() {
                if question.is_none() {
                    question = self.question_for(frame, &resolved, knowledge, input, turn);
                }
                continue;
            }
            if let Some(executed) = self.execute(frame, &resolved, knowledge) {
                memory.commit(frame, &executed.bindings, &resolved.origins, turn);
                memory.pending = None;
                return Some(self.outcome(frame, pattern, executed, turn));
            }
        }

        let question = question?;
        let schema = question.pending.schema.clone();
        memory.pending = Some(question.pending);
        memory.turn = turn;
        memory.clarifications_asked += 1;
        Some(FrameOutcome {
            text: question.prompt,
            plan: vec![
                "frame_clarification".to_string(),
                format!("frame:{}", schema),
            ],
            provenance: vec![format!("绑定帧「{}」缺槽，已发起一次显式绑定请求", schema)],
            kind: MoveKind::AskAttribute,
            entity: None,
            attribute: None,
        })
    }

    fn resolve_frame(
        &self,
        frame: &CompiledFrame,
        captures: &BTreeMap<String, String>,
        memory: &FrameMemory,
        knowledge: &dyn DialogueKnowledge,
        turn: u64,
    ) -> Option<Resolved> {
        let mut resolved = Resolved::default();
        for (name, raw) in captures {
            let slot = frame.slots.get(name)?;
            let value = normalize_slot_value(slot, raw, knowledge)?;
            resolved.bindings.insert(name.clone(), value);
            resolved.origins.insert(name.clone(), FrameOrigin::Explicit);
        }

        for name in &frame.slot_order {
            if resolved.bindings.contains_key(name) {
                continue;
            }
            let slot = &frame.slots[name];
            if !slot.inherit {
                continue;
            }
            if let Some(stored) = memory.env.get(name) {
                if turn.saturating_sub(stored.turn) <= FRAME_ENV_TTL
                    && normalize_slot_value(slot, &stored.raw, knowledge).is_some()
                {
                    resolved.bindings.insert(name.clone(), stored.raw.clone());
                    resolved
                        .origins
                        .insert(name.clone(), FrameOrigin::Inherited);
                }
            }
        }

        if let Some(snapshot) = memory.last.get(&frame.id) {
            for name in &frame.slot_order {
                if resolved.bindings.contains_key(name) {
                    continue;
                }
                if let Some(raw) = snapshot.bindings.get(name) {
                    let slot = &frame.slots[name];
                    if normalize_slot_value(slot, raw, knowledge).is_some() {
                        resolved.bindings.insert(name.clone(), raw.clone());
                        resolved
                            .origins
                            .insert(name.clone(), FrameOrigin::Inherited);
                    }
                }
            }
        }

        for name in &frame.slot_order {
            let slot = &frame.slots[name];
            if slot.required && !resolved.bindings.contains_key(name) {
                resolved.missing.push(name.clone());
            }
        }
        Some(resolved)
    }

    fn question_for(
        &self,
        frame: &CompiledFrame,
        resolved: &Resolved,
        knowledge: &dyn DialogueKnowledge,
        raw: &str,
        turn: u64,
    ) -> Option<QuestionPlan> {
        let slot_name = resolved.missing.first()?.clone();
        let slot = frame.slots.get(&slot_name)?;
        let candidates = clarification_candidates(frame, slot, resolved, knowledge);
        if slot.ask.is_none() && candidates.is_empty() && slot.sort == SlotSort::Entity {
            return None;
        }

        let mut prompt = match slot.ask.as_deref() {
            Some(template) => template
                .replace("{raw}", raw.trim())
                .replace("{slot}", slot_name.as_str())
                .replace("{slot_label}", slot.sort.label()),
            None => format!(
                "「{}」还缺一个{}。请补充{}。",
                raw.trim(),
                slot.sort.label(),
                slot_name
            ),
        };
        if !candidates.is_empty() {
            prompt.push('\n');
            for (index, candidate) in candidates.iter().enumerate() {
                prompt.push_str(&format!("  {}. {}\n", index + 1, candidate));
            }
            prompt.push_str("回复序号或候选名完成绑定；「都不是」放弃这次追问。");
        } else {
            prompt.push_str("直接回复即可完成绑定；「都不是」放弃这次追问。");
        }

        Some(QuestionPlan {
            prompt: prompt.clone(),
            pending: PendingFrame {
                schema: frame.id.clone(),
                bindings: resolved.bindings.clone(),
                origins: resolved.origins.clone(),
                missing: resolved.missing.clone(),
                candidates,
                raw: raw.trim().to_string(),
                prompt,
                turn,
            },
        })
    }

    fn resolve_pending(
        &self,
        memory: &mut FrameMemory,
        input: &str,
        knowledge: &dyn DialogueKnowledge,
        turn: u64,
    ) -> Option<FrameOutcome> {
        let pending = memory.pending.clone()?;
        if turn.saturating_sub(pending.turn) > FRAME_PENDING_TTL {
            memory.pending = None;
            return None;
        }
        let Some(frame) = self.frame(&pending.schema) else {
            memory.pending = None;
            return None;
        };
        if is_cancel_reply(input) {
            memory.pending = None;
            memory.clarifications_cancelled += 1;
            return Some(FrameOutcome {
                text: "好的，这次追问先放下。".to_string(),
                plan: vec![
                    "frame_clarification_cancel".to_string(),
                    format!("frame:{}", pending.schema),
                ],
                provenance: vec![format!("绑定帧「{}」的澄清被用户撤销", pending.schema)],
                kind: MoveKind::Other,
                entity: None,
                attribute: None,
            });
        }

        let slot_name = pending.missing.first()?.clone();
        let slot = frame.slots.get(&slot_name)?;
        let reply = pending_reply(&pending, slot, input, knowledge);
        let value = match reply {
            PendingReply::Candidate(value) => value,
            PendingReply::OutOfRange => {
                memory.pending = Some(pending.clone());
                return Some(FrameOutcome {
                    text: pending.prompt.clone(),
                    plan: vec![
                        "frame_clarification_reprompt".to_string(),
                        format!("frame:{}", pending.schema),
                    ],
                    provenance: vec!["澄清序号越界，已原样重问（零状态变化）".to_string()],
                    kind: MoveKind::AskAttribute,
                    entity: None,
                    attribute: None,
                });
            }
            PendingReply::Unmatched => return None,
        };

        let mut bindings = pending.bindings.clone();
        let mut origins = pending.origins.clone();
        bindings.insert(slot_name.clone(), value);
        origins.insert(slot_name.clone(), FrameOrigin::Clarified);
        let mut missing = pending.missing.clone();
        missing.remove(0);
        let resolved = Resolved {
            bindings,
            origins,
            missing,
        };

        if !resolved.missing.is_empty() {
            let next = self.question_for(frame, &resolved, knowledge, &pending.raw, turn)?;
            memory.pending = Some(next.pending);
            memory.turn = turn;
            memory.clarifications_asked += 1;
            return Some(FrameOutcome {
                text: next.prompt,
                plan: vec![
                    "frame_clarification".to_string(),
                    format!("frame:{}", pending.schema),
                ],
                provenance: vec![format!(
                    "绑定帧「{}」完成一次澄清，仍有缺槽",
                    pending.schema
                )],
                kind: MoveKind::AskAttribute,
                entity: None,
                attribute: None,
            });
        }

        if let Some(executed) = self.execute(frame, &resolved, knowledge) {
            memory.commit(frame, &executed.bindings, &resolved.origins, turn);
            memory.pending = None;
            memory.clarifications_bound += 1;
            let mut outcome = self.outcome(
                frame,
                &CompiledPattern {
                    source: format!("clarification:{}", pending.raw),
                    parts: Vec::new(),
                    slots: Vec::new(),
                    literal_chars: 0,
                },
                executed,
                turn,
            );
            outcome
                .plan
                .insert(0, "frame_clarification_bound".to_string());
            return Some(outcome);
        }

        memory.pending = None;
        memory.clarifications_bound += 1;
        let text = frame
            .realize_empty
            .as_ref()
            .and_then(|template| {
                let vars = self.render_vars(frame, &resolved.bindings, knowledge);
                render_template(template, &vars)
            })
            .unwrap_or_else(|| format!("没有查到与「{}」相关的记录。", pending.raw));
        Some(FrameOutcome {
            text,
            plan: vec![
                "frame_clarification_empty".to_string(),
                format!("frame:{}", pending.schema),
            ],
            provenance: vec![format!("绑定帧「{}」澄清后目标无值", pending.schema)],
            kind: MoveKind::AskAttribute,
            entity: None,
            attribute: None,
        })
    }

    fn execute(
        &self,
        frame: &CompiledFrame,
        resolved: &Resolved,
        knowledge: &dyn DialogueKnowledge,
    ) -> Option<Executed> {
        match &frame.goal {
            CompiledGoal::Attribute {
                entity,
                attribute,
                output,
            } => {
                let entity = entity.resolve(&resolved.bindings)?;
                let attribute = attribute.resolve(&resolved.bindings)?;
                let mut values = knowledge.attribute_values(&entity, &attribute);
                values.sort();
                values.dedup();
                if values.is_empty() {
                    return None;
                }
                let mut bindings = resolved.bindings.clone();
                bindings.insert(output.clone(), values[0].clone());
                let vars = self.render_vars(frame, &bindings, knowledge);
                let text = frame
                    .realize_answer
                    .as_ref()
                    .and_then(|template| render_template(template, &vars))
                    .unwrap_or_else(|| {
                        format!(
                            "{}的{}是{}。",
                            display_value(&entity, knowledge),
                            display_value(&attribute, knowledge),
                            values.join("、")
                        )
                    });
                let evidence = values
                    .iter()
                    .map(|value| {
                        format!(
                            "实体「{}」的属性「{}」=「{}」",
                            entity,
                            display_value(&attribute, knowledge),
                            value
                        )
                    })
                    .collect();
                Some(Executed {
                    text,
                    bindings,
                    evidence,
                    entity: Some(entity),
                    attribute: Some(attribute),
                })
            }
            CompiledGoal::Relation {
                subject,
                predicate,
                output,
            } => {
                let subject = subject.resolve(&resolved.bindings)?;
                let predicate = predicate.resolve(&resolved.bindings)?;
                let objects = relation_objects(knowledge, &subject, &predicate);
                if objects.is_empty() {
                    return None;
                }
                let mut bindings = resolved.bindings.clone();
                bindings.insert(output.clone(), objects[0].clone());
                let vars = self.render_vars(frame, &bindings, knowledge);
                let text = frame
                    .realize_answer
                    .as_ref()
                    .and_then(|template| render_template(template, &vars))
                    .unwrap_or_else(|| {
                        format!(
                            "{}的{}是{}。",
                            subject,
                            display_value(&predicate, knowledge),
                            objects.join("、")
                        )
                    });
                let evidence = objects
                    .iter()
                    .map(|object| {
                        format!(
                            "关系：{} -[{}]-> {}",
                            subject,
                            display_value(&predicate, knowledge),
                            object
                        )
                    })
                    .collect();
                Some(Executed {
                    text,
                    bindings,
                    evidence,
                    entity: Some(subject),
                    attribute: Some(predicate),
                })
            }
            CompiledGoal::ReverseRelation {
                object,
                predicate,
                output,
            } => {
                let object = object.resolve(&resolved.bindings)?;
                let predicate = predicate.resolve(&resolved.bindings)?;
                let subjects = reverse_relation_objects(knowledge, &object, &predicate);
                if subjects.is_empty() {
                    return None;
                }
                let mut bindings = resolved.bindings.clone();
                bindings.insert(output.clone(), subjects[0].clone());
                let vars = self.render_vars(frame, &bindings, knowledge);
                let text = frame
                    .realize_answer
                    .as_ref()
                    .and_then(|template| render_template(template, &vars))
                    .unwrap_or_else(|| {
                        format!(
                            "{}的{}是{}。",
                            object,
                            display_value(&predicate, knowledge),
                            subjects.join("、")
                        )
                    });
                let evidence = subjects
                    .iter()
                    .map(|subject| {
                        format!(
                            "关系：{} -[{}]-> {}",
                            subject,
                            display_value(&predicate, knowledge),
                            object
                        )
                    })
                    .collect();
                Some(Executed {
                    text,
                    bindings,
                    evidence,
                    entity: Some(object),
                    attribute: Some(predicate),
                })
            }
            CompiledGoal::Count {
                subject,
                predicate,
                output,
            } => {
                let subject = subject.resolve(&resolved.bindings)?;
                let predicate = predicate.resolve(&resolved.bindings)?;
                let objects = relation_objects(knowledge, &subject, &predicate);
                if objects.is_empty() {
                    return None;
                }
                let count = objects.len();
                let mut bindings = resolved.bindings.clone();
                bindings.insert(output.clone(), count.to_string());
                let vars = self.render_vars(frame, &bindings, knowledge);
                let text = frame
                    .realize_answer
                    .as_ref()
                    .and_then(|template| render_template(template, &vars))
                    .unwrap_or_else(|| {
                        format!(
                            "{}共有{}个{}。",
                            subject,
                            count,
                            display_value(&predicate, knowledge)
                        )
                    });
                Some(Executed {
                    text,
                    bindings,
                    evidence: vec![format!(
                        "计数：{} 的「{}」共 {} 个（{}）",
                        subject,
                        display_value(&predicate, knowledge),
                        count,
                        objects.join("、")
                    )],
                    entity: Some(subject),
                    attribute: Some(predicate),
                })
            }
            CompiledGoal::Compare {
                left,
                right,
                attribute,
                output,
            } => {
                let left = left.resolve(&resolved.bindings)?;
                let right = right.resolve(&resolved.bindings)?;
                let attribute = attribute.resolve(&resolved.bindings)?;
                let left_values = knowledge.attribute_values(&left, &attribute);
                let right_values = knowledge.attribute_values(&right, &attribute);
                let left_value = left_values.first()?;
                let right_value = right_values.first()?;
                let attribute_label = display_value(&attribute, knowledge);
                let evaluated =
                    compare_attribute_values(&left, &right, &attribute_label, left_value, right_value);
                let winner = evaluated
                    .winner
                    .clone()
                    .unwrap_or_else(|| "无法比较".to_string());
                let mut bindings = resolved.bindings.clone();
                bindings.insert(output.clone(), winner.clone());
                let vars = self.render_vars(frame, &bindings, knowledge);
                let text = frame
                    .realize_answer
                    .as_ref()
                    .and_then(|template| render_template(template, &vars))
                    .unwrap_or_else(|| {
                        format!(
                            "{}的{}是{}，{}的{}是{}；{}。",
                            left,
                            display_value(&attribute, knowledge),
                            left_value,
                            right,
                            display_value(&attribute, knowledge),
                            right_value,
                            evaluated.comparison
                        )
                    });
                Some(Executed {
                    text,
                    bindings,
                    evidence: vec![format!(
                        "比较：{} 的 {}={}，{} 的 {}={}；{}",
                        left,
                        display_value(&attribute, knowledge),
                        left_value,
                        right,
                        display_value(&attribute, knowledge),
                        right_value,
                        evaluated.comparison
                    )],
                    entity: evaluated.winner,
                    attribute: Some(attribute),
                })
            }
        }
    }

    fn render_vars(
        &self,
        frame: &CompiledFrame,
        bindings: &BTreeMap<String, String>,
        knowledge: &dyn DialogueKnowledge,
    ) -> BTreeMap<String, String> {
        let mut vars = BTreeMap::new();
        for (name, raw) in bindings {
            let value = match frame.slots.get(name).map(|slot| slot.sort) {
                Some(SlotSort::Attribute | SlotSort::Predicate) => display_value(raw, knowledge),
                _ => raw.clone(),
            };
            vars.insert(name.clone(), value);
        }
        let values: Vec<String> = match &frame.goal {
            CompiledGoal::Attribute {
                entity,
                attribute,
                output,
            } => {
                let _ = output;
                entity
                    .resolve(bindings)
                    .zip(attribute.resolve(bindings))
                    .map(|(entity, attribute)| knowledge.attribute_values(&entity, &attribute))
                    .unwrap_or_default()
            }
            CompiledGoal::Relation {
                subject,
                predicate,
                output,
            } => {
                let _ = output;
                subject
                    .resolve(bindings)
                    .zip(predicate.resolve(bindings))
                    .map(|(subject, predicate)| relation_objects(knowledge, &subject, &predicate))
                    .unwrap_or_default()
            }
            CompiledGoal::ReverseRelation {
                object,
                predicate,
                output,
            } => {
                let _ = output;
                object
                    .resolve(bindings)
                    .zip(predicate.resolve(bindings))
                    .map(|(object, predicate)| {
                        reverse_relation_objects(knowledge, &object, &predicate)
                    })
                    .unwrap_or_default()
            }
            CompiledGoal::Count { .. } => Vec::new(),
            CompiledGoal::Compare {
                left,
                right,
                attribute,
                output,
            } => {
                let _ = output;
                left.resolve(bindings)
                    .zip(right.resolve(bindings))
                    .zip(attribute.resolve(bindings))
                    .map(|((left, right), attribute)| {
                        let mut values = knowledge.attribute_values(&left, &attribute);
                        values.extend(knowledge.attribute_values(&right, &attribute));
                        values
                    })
                    .unwrap_or_default()
            }
        };
        vars.insert("values".to_string(), values.join("、"));
        vars.insert(
            "value".to_string(),
            values.first().cloned().unwrap_or_default(),
        );
        vars.insert("objects".to_string(), values.join("、"));
        vars.insert(
            "object".to_string(),
            values.first().cloned().unwrap_or_default(),
        );
        vars.insert("count".to_string(), values.len().to_string());
        if let CompiledGoal::Compare {
            left,
            right,
            attribute,
            output,
        } = &frame.goal
        {
            let _ = output;
            if let (Some(left), Some(right), Some(attribute)) = (
                left.resolve(bindings),
                right.resolve(bindings),
                attribute.resolve(bindings),
            ) {
                if let (Some(left_value), Some(right_value)) = (
                    knowledge.attribute_values(&left, &attribute).first(),
                    knowledge.attribute_values(&right, &attribute).first(),
                ) {
                    let evaluated = compare_attribute_values(
                        &left,
                        &right,
                        &display_value(&attribute, knowledge),
                        left_value,
                        right_value,
                    );
                    vars.insert("left_value".to_string(), left_value.clone());
                    vars.insert("right_value".to_string(), right_value.clone());
                    vars.insert("comparison".to_string(), evaluated.comparison);
                    if let Some(winner) = evaluated.winner {
                        vars.insert("winner".to_string(), winner);
                    }
                }
            }
        }
        vars
    }

    fn outcome(
        &self,
        frame: &CompiledFrame,
        pattern: &CompiledPattern,
        executed: Executed,
        turn: u64,
    ) -> FrameOutcome {
        let plan = vec![
            format!("frame:{}", frame.id),
            format!("goal:{}", frame.goal.kind_label()),
            format!("frame_slots:{}", executed.bindings.len()),
            format!("frame_turn:{}", turn),
        ];
        let mut provenance = vec![format!(
            "绑定帧「{}」命中模式「{}」",
            frame.id, pattern.source
        )];
        provenance.extend(executed.evidence.clone());
        FrameOutcome {
            text: executed.text,
            plan,
            provenance,
            kind: MoveKind::AskAttribute,
            entity: executed.entity,
            attribute: executed.attribute,
        }
    }
}

fn compile_frame(
    frame_index: usize,
    schema: &FrameSchemaConfig,
    seen_ids: &mut BTreeSet<String>,
    errors: &mut Vec<FrameCompileError>,
) -> Option<CompiledFrame> {
    let base = format!("frames[{}]", frame_index);
    let id = schema.id.trim().to_string();
    if id.is_empty() {
        errors.push(FrameCompileError {
            code: "frame_id_empty",
            path: format!("{}.id", base),
            message: "帧 id 不能为空".to_string(),
        });
    } else if !seen_ids.insert(id.clone()) {
        errors.push(FrameCompileError {
            code: "frame_id_duplicate",
            path: format!("{}.id", base),
            message: format!("帧 id「{}」重复；同一配置必须给出稳定唯一的标识", id),
        });
    }

    if schema.patterns.is_empty() {
        errors.push(FrameCompileError {
            code: "frame_patterns_empty",
            path: format!("{}.patterns", base),
            message: format!("帧「{}」没有任何表面模式，永远不会命中", id),
        });
    }

    let mut slots = BTreeMap::new();
    let mut slot_order = Vec::new();
    for (slot_index, slot) in schema.slots.iter().enumerate() {
        let name = slot.name.trim().to_string();
        if name.is_empty() {
            errors.push(FrameCompileError {
                code: "frame_slot_empty",
                path: format!("{}.slots[{}].name", base, slot_index),
                message: "槽位名不能为空".to_string(),
            });
            continue;
        }
        if slots.contains_key(&name) {
            errors.push(FrameCompileError {
                code: "frame_slot_duplicate",
                path: format!("{}.slots[{}].name", base, slot_index),
                message: format!("帧「{}」的槽位「{}」重复声明", id, name),
            });
            continue;
        }
        let mut slot = slot.clone();
        slot.name = name.clone();
        slot_order.push(name.clone());
        slots.insert(name, slot);
    }

    let mut patterns = Vec::new();
    let mut seen_patterns = BTreeSet::new();
    for (pattern_index, source) in schema.patterns.iter().enumerate() {
        let path = format!("{}.patterns[{}]", base, pattern_index);
        let pattern = match compile_pattern(source, &path) {
            Ok(pattern) => pattern,
            Err(error) => {
                errors.push(error);
                continue;
            }
        };
        if !seen_patterns.insert(pattern.source.clone()) {
            errors.push(FrameCompileError {
                code: "frame_pattern_duplicate",
                path,
                message: format!("帧「{}」重复声明模式「{}」", id, pattern.source),
            });
            continue;
        }
        for name in &pattern.slots {
            if !slots.contains_key(name) {
                errors.push(FrameCompileError {
                    code: "frame_slot_undeclared",
                    path: format!("{}.patterns[{}]", base, pattern_index),
                    message: format!(
                        "模式「{}」引用了未声明槽位「{}」；槽位必须先出现在 slots 中",
                        pattern.source, name
                    ),
                });
            }
        }
        patterns.push(pattern);
    }

    let goal = match compile_goal(&schema.goal, &slots, &format!("{}.goal", base), errors) {
        Some(goal) => goal,
        None => return None,
    };
    let output = goal.output_slot().to_string();
    if !slots.contains_key(&output) {
        errors.push(FrameCompileError {
            code: "frame_output_undeclared",
            path: format!("{}.goal.output", base),
            message: format!(
                "目标输出槽「{}」未声明；答案模板无法安全渲染未声明变量",
                output
            ),
        });
    }
    if goal.input_slots().iter().any(|slot| slot == &output) {
        errors.push(FrameCompileError {
            code: "frame_output_is_input",
            path: format!("{}.goal.output", base),
            message: format!("槽位「{}」同时被目标读取与写入", output),
        });
    }
    if slots
        .get(&output)
        .map(|slot| slot.required)
        .unwrap_or(false)
    {
        errors.push(FrameCompileError {
            code: "frame_output_required",
            path: format!("{}.slots", base),
            message: format!("输出槽「{}」不能声明为必填输入槽", output),
        });
    }

    for template in [&schema.realize.answer, &schema.realize.empty]
        .into_iter()
        .flatten()
    {
        match template_names(template) {
            Ok(names) => {
                for name in names {
                    if !is_builtin_var(&name) && !slots.contains_key(&name) {
                        errors.push(FrameCompileError {
                            code: "frame_template_undeclared",
                            path: format!("{}.realize", base),
                            message: format!("答案模板引用了未声明变量「{}」", name),
                        });
                    }
                }
            }
            Err(message) => errors.push(FrameCompileError {
                code: "frame_template_invalid",
                path: format!("{}.realize", base),
                message,
            }),
        }
    }

    let pattern_slots: BTreeSet<String> = patterns
        .iter()
        .flat_map(|pattern| pattern.slots.iter().cloned())
        .collect();
    let goal_inputs = goal.input_slots();
    for name in &slot_order {
        let slot = &slots[name];
        if !slot.required {
            continue;
        }
        if name == &output {
            continue;
        }
        let in_all_patterns = patterns
            .iter()
            .all(|pattern| pattern.slots.iter().any(|slot| slot == name));
        if !in_all_patterns && !slot.inherit && slot.ask.is_none() {
            errors.push(FrameCompileError {
                code: "frame_required_unreachable",
                path: format!("{}.slots", base),
                message: format!(
                    "必填槽「{}」并非每个模式都出现，且未声明 inherit / ask，缺槽时无法补全",
                    name
                ),
            });
        }
        if !pattern_slots.contains(name) && !goal_inputs.contains(name) {
            errors.push(FrameCompileError {
                code: "frame_area_unreachable",
                path: format!("{}.slots", base),
                message: format!(
                    "必填槽「{}」既不出现在模式中，也不被目标读取，永远不会被绑定",
                    name
                ),
            });
        }
    }

    Some(CompiledFrame {
        id,
        priority: schema.priority,
        patterns,
        slots,
        slot_order,
        goal,
        realize_answer: schema.realize.answer.clone(),
        realize_empty: schema.realize.empty.clone(),
    })
}

fn compile_pattern(source: &str, path: &str) -> Result<CompiledPattern, FrameCompileError> {
    let source = source.trim().to_string();
    if source.is_empty() {
        return Err(FrameCompileError {
            code: "frame_pattern_empty",
            path: path.to_string(),
            message: "模式不能为空字符串".to_string(),
        });
    }
    let mut parts = Vec::new();
    let mut literal = String::new();
    let mut slots = Vec::new();
    let mut chars = source.chars().peekable();
    let mut literal_chars = 0;
    while let Some(ch) = chars.next() {
        match ch {
            '{' => {
                if !literal.is_empty() {
                    literal_chars += literal.chars().count();
                    parts.push(PatternPart::Literal(std::mem::take(&mut literal)));
                }
                let mut name = String::new();
                let mut closed = false;
                for next in chars.by_ref() {
                    if next == '}' {
                        closed = true;
                        break;
                    }
                    if next == '{' {
                        return Err(FrameCompileError {
                            code: "frame_pattern_invalid",
                            path: path.to_string(),
                            message: format!("模式「{}」出现嵌套 `{{`", source),
                        });
                    }
                    name.push(next);
                }
                if !closed {
                    return Err(FrameCompileError {
                        code: "frame_pattern_invalid",
                        path: path.to_string(),
                        message: format!("模式「{}」的槽位缺少右花括号", source),
                    });
                }
                let name = name.trim().to_string();
                if name.is_empty() {
                    return Err(FrameCompileError {
                        code: "frame_pattern_invalid",
                        path: path.to_string(),
                        message: format!("模式「{}」出现空槽位 `{{}}`", source),
                    });
                }
                if slots.contains(&name) {
                    return Err(FrameCompileError {
                        code: "frame_pattern_invalid",
                        path: path.to_string(),
                        message: format!("模式「{}」重复引用槽位「{}」", source, name),
                    });
                }
                if matches!(parts.last(), Some(PatternPart::Slot(_))) {
                    return Err(FrameCompileError {
                        code: "frame_pattern_invalid",
                        path: path.to_string(),
                        message: format!("模式「{}」出现相邻槽位，捕获边界不唯一", source),
                    });
                }
                slots.push(name.clone());
                parts.push(PatternPart::Slot(name));
            }
            '}' => {
                return Err(FrameCompileError {
                    code: "frame_pattern_invalid",
                    path: path.to_string(),
                    message: format!("模式「{}」出现未配对的右花括号", source),
                });
            }
            other => literal.push(other),
        }
    }
    if !literal.is_empty() {
        literal_chars += literal.chars().count();
        parts.push(PatternPart::Literal(literal));
    }
    if parts.is_empty() {
        return Err(FrameCompileError {
            code: "frame_pattern_empty",
            path: path.to_string(),
            message: "模式至少要包含一个字面量或槽位".to_string(),
        });
    }
    Ok(CompiledPattern {
        source,
        parts,
        slots,
        literal_chars,
    })
}

fn compile_goal(
    goal: &FrameGoalConfig,
    slots: &BTreeMap<String, FrameSlotConfig>,
    path: &str,
    errors: &mut Vec<FrameCompileError>,
) -> Option<CompiledGoal> {
    let compile_operand = |name: &str, field: &str, errors: &mut Vec<FrameCompileError>| {
        if let Some(constant) = name.strip_prefix("const:") {
            if constant.trim().is_empty() {
                errors.push(FrameCompileError {
                    code: "frame_goal_const_empty",
                    path: format!("{}.{}", path, field),
                    message: "常量操作数 `const:` 后不能为空".to_string(),
                });
                return None;
            }
            return Some(Operand::Const(constant.trim().to_string()));
        }
        if slots.contains_key(name) {
            Some(Operand::Slot(name.to_string()))
        } else {
            errors.push(FrameCompileError {
                code: "frame_goal_slot_unknown",
                path: format!("{}.{}", path, field),
                message: format!(
                    "目标引用了未声明槽位「{}」；常量请显式写成 `const:...`",
                    name
                ),
            });
            None
        }
    };

    Some(match goal {
        FrameGoalConfig::Attribute {
            entity,
            attribute,
            output,
        } => CompiledGoal::Attribute {
            entity: compile_operand(entity, "entity", errors)?,
            attribute: compile_operand(attribute, "attribute", errors)?,
            output: output.clone(),
        },
        FrameGoalConfig::Relation {
            subject,
            predicate,
            output,
        } => CompiledGoal::Relation {
            subject: compile_operand(subject, "subject", errors)?,
            predicate: compile_operand(predicate, "predicate", errors)?,
            output: output.clone(),
        },
        FrameGoalConfig::ReverseRelation {
            object,
            predicate,
            output,
        } => CompiledGoal::ReverseRelation {
            object: compile_operand(object, "object", errors)?,
            predicate: compile_operand(predicate, "predicate", errors)?,
            output: output.clone(),
        },
        FrameGoalConfig::Count {
            subject,
            predicate,
            output,
        } => CompiledGoal::Count {
            subject: compile_operand(subject, "subject", errors)?,
            predicate: compile_operand(predicate, "predicate", errors)?,
            output: output.clone(),
        },
        FrameGoalConfig::Compare {
            left,
            right,
            attribute,
            output,
        } => CompiledGoal::Compare {
            left: compile_operand(left, "left", errors)?,
            right: compile_operand(right, "right", errors)?,
            attribute: compile_operand(attribute, "attribute", errors)?,
            output: output.clone(),
        },
    })
}

fn template_names(template: &str) -> Result<Vec<String>, String> {
    let mut names = Vec::new();
    let mut chars = template.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '{' => {
                let mut name = String::new();
                let mut closed = false;
                for next in chars.by_ref() {
                    if next == '}' {
                        closed = true;
                        break;
                    }
                    if next == '{' {
                        return Err("答案模板出现嵌套 `{`".to_string());
                    }
                    name.push(next);
                }
                if !closed {
                    return Err("答案模板存在未闭合的槽位".to_string());
                }
                let name = name.trim().to_string();
                if name.is_empty() {
                    return Err("答案模板出现空槽位 `{}`".to_string());
                }
                names.push(name);
            }
            '}' => return Err("答案模板出现未配对的右花括号".to_string()),
            _ => {}
        }
    }
    Ok(names)
}

fn is_builtin_var(name: &str) -> bool {
    matches!(
        name,
        "values"
            | "value"
            | "objects"
            | "object"
            | "count"
            | "left_value"
            | "right_value"
            | "comparison"
            | "winner"
            | "raw"
            | "slot"
            | "slot_label"
    )
}

fn render_template(template: &str, vars: &BTreeMap<String, String>) -> Option<String> {
    let mut output = String::new();
    let mut chars = template.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '{' => {
                let mut name = String::new();
                let mut closed = false;
                for next in chars.by_ref() {
                    if next == '}' {
                        closed = true;
                        break;
                    }
                    name.push(next);
                }
                if !closed {
                    return None;
                }
                let value = vars.get(name.trim())?;
                output.push_str(value);
            }
            '}' => return None,
            other => output.push(other),
        }
    }
    Some(output)
}

// ----------------------------- 运行层 -----------------------------

/// 槽位来源：可打印、可审计。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum FrameOrigin {
    #[default]
    Explicit,
    Inherited,
    Clarified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct StoredBinding {
    raw: String,
    origin: FrameOrigin,
    turn: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct FrameSnapshot {
    bindings: BTreeMap<String, String>,
    turn: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct PendingFrame {
    schema: String,
    bindings: BTreeMap<String, String>,
    origins: BTreeMap<String, FrameOrigin>,
    missing: Vec<String>,
    candidates: Vec<String>,
    raw: String,
    prompt: String,
    turn: u64,
}

/// 会话级绑定记忆：环境、上一帧、待澄清帧与计数。
///
/// 空态不写入快照（`DialogueManagerSnapshot` 用 `Option` + skip），
/// 因此未启用绑定帧的会话字节不变。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameMemory {
    #[serde(default)]
    turn: u64,
    #[serde(default)]
    env: BTreeMap<String, StoredBinding>,
    #[serde(default)]
    last: BTreeMap<String, FrameSnapshot>,
    #[serde(default)]
    pending: Option<PendingFrame>,
    #[serde(default)]
    answered: u64,
    #[serde(default)]
    inherited_slots: u64,
    #[serde(default)]
    clarifications_asked: u64,
    #[serde(default)]
    clarifications_bound: u64,
    #[serde(default)]
    clarifications_cancelled: u64,
}

impl Default for FrameMemory {
    fn default() -> Self {
        Self {
            turn: 0,
            env: BTreeMap::new(),
            last: BTreeMap::new(),
            pending: None,
            answered: 0,
            inherited_slots: 0,
            clarifications_asked: 0,
            clarifications_bound: 0,
            clarifications_cancelled: 0,
        }
    }
}

impl FrameMemory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.turn == 0
            && self.env.is_empty()
            && self.last.is_empty()
            && self.pending.is_none()
            && self.answered == 0
            && self.inherited_slots == 0
            && self.clarifications_asked == 0
            && self.clarifications_bound == 0
            && self.clarifications_cancelled == 0
    }

    pub fn turn(&self) -> u64 {
        self.turn
    }

    pub fn answered(&self) -> u64 {
        self.answered
    }

    pub fn pending_prompt(&self) -> Option<&str> {
        self.pending.as_ref().map(|pending| pending.prompt.as_str())
    }

    pub fn report(&self) -> String {
        format!(
            "绑定帧记忆：轮次 {}，回答 {} 次，继承槽 {} 次，澄清 {} 次（绑定 {}、撤销 {}），待澄清 {}。",
            self.turn,
            self.answered,
            self.inherited_slots,
            self.clarifications_asked,
            self.clarifications_bound,
            self.clarifications_cancelled,
            if self.pending.is_some() { 1 } else { 0 }
        )
    }

    fn prune(&mut self, turn: u64) {
        if let Some(pending) = &self.pending {
            if turn.saturating_sub(pending.turn) > FRAME_PENDING_TTL {
                self.pending = None;
            }
        }
        self.env
            .retain(|_, binding| turn.saturating_sub(binding.turn) <= FRAME_ENV_TTL);
        self.last
            .retain(|_, snapshot| turn.saturating_sub(snapshot.turn) <= FRAME_ENV_TTL * 2);
    }

    fn commit(
        &mut self,
        frame: &CompiledFrame,
        bindings: &BTreeMap<String, String>,
        origins: &BTreeMap<String, FrameOrigin>,
        turn: u64,
    ) {
        self.turn = turn;
        self.answered += 1;
        for (name, raw) in bindings {
            if !frame.is_input_slot(name) {
                continue;
            }
            let origin = origins.get(name).copied().unwrap_or(FrameOrigin::Explicit);
            if origin == FrameOrigin::Inherited {
                self.inherited_slots += 1;
            }
            self.env.insert(
                name.clone(),
                StoredBinding {
                    raw: raw.clone(),
                    origin,
                    turn,
                },
            );
        }
        self.last.insert(
            frame.id.clone(),
            FrameSnapshot {
                bindings: bindings.clone(),
                turn,
            },
        );
        while self.env.len() > 64 {
            let oldest = self
                .env
                .iter()
                .min_by_key(|(name, binding)| (binding.turn, name.as_str()))
                .map(|(name, _)| name.clone());
            match oldest {
                Some(name) => {
                    self.env.remove(&name);
                }
                None => break,
            }
        }
    }
}

#[derive(Debug, Clone, Default)]
struct Resolved {
    bindings: BTreeMap<String, String>,
    origins: BTreeMap<String, FrameOrigin>,
    missing: Vec<String>,
}

struct QuestionPlan {
    prompt: String,
    pending: PendingFrame,
}

struct Executed {
    text: String,
    bindings: BTreeMap<String, String>,
    evidence: Vec<String>,
    entity: Option<String>,
    attribute: Option<String>,
}

/// 一轮帧规划结果（对话层据此记账，不直接打印）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameOutcome {
    pub text: String,
    pub plan: Vec<String>,
    pub provenance: Vec<String>,
    pub kind: MoveKind,
    pub entity: Option<String>,
    pub attribute: Option<String>,
}

fn normalize_utterance(input: &str) -> String {
    let mut text = input.trim();
    const POLITE: [&str; 7] = [
        "请问",
        "请告诉我",
        "我想知道",
        "我想问一下",
        "帮我查一下",
        "帮我查查",
        "麻烦问一下",
    ];
    loop {
        let mut changed = false;
        for prefix in POLITE {
            if let Some(rest) = text.strip_prefix(prefix) {
                let rest = rest.trim();
                if !rest.is_empty() {
                    text = rest;
                    changed = true;
                    break;
                }
            }
        }
        if !changed {
            break;
        }
    }
    text.trim()
        .trim_end_matches(|ch: char| {
            matches!(ch, '？' | '?' | '。' | '！' | '!' | ' ' | '～' | '~' | '…')
        })
        .trim()
        .to_string()
}

fn normalize_slot_value(
    slot: &FrameSlotConfig,
    raw: &str,
    knowledge: &dyn DialogueKnowledge,
) -> Option<String> {
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }
    match slot.sort {
        SlotSort::Entity => {
            if knowledge.known_entity(raw) {
                Some(raw.to_string())
            } else {
                knowledge
                    .entities_in(raw)
                    .into_iter()
                    .filter(|entity| entity == raw)
                    .next()
            }
        }
        SlotSort::Number => quantity::parse_quantity(raw).map(|_| raw.to_string()),
        SlotSort::Predicate | SlotSort::Attribute => Some(
            knowledge
                .canonical_predicate(raw)
                .unwrap_or_else(|| raw.to_string()),
        ),
        SlotSort::Text => Some(raw.to_string()),
    }
}

fn display_value(raw: &str, knowledge: &dyn DialogueKnowledge) -> String {
    knowledge.predicate_label(raw)
}

/// 比较结论与胜者。胜者为 `None` 表示不可比或约数区间重叠，调用方不得
/// 把它渲染成任何一方获胜。
struct AttributeComparison {
    winner: Option<String>,
    comparison: String,
}

/// 属性比较的唯一出口：单位换算、量纲校验、约数不确定性与差异量都在这里。
fn compare_attribute_values(
    left: &str,
    right: &str,
    attribute_label: &str,
    left_raw: &str,
    right_raw: &str,
) -> AttributeComparison {
    let (Some(left_quantity), Some(right_quantity)) =
        (quantity::parse_quantity(left_raw), quantity::parse_quantity(right_raw))
    else {
        return AttributeComparison {
            winner: None,
            comparison: format!(
                "「{}」和「{}」的{}至少一侧不是数值，无法比较",
                left, right, attribute_label
            ),
        };
    };
    let word = left_quantity.dimension().comparative_word(attribute_label);
    match left_quantity.compare(&right_quantity) {
        Comparison::Greater => {
            winner_with_difference(left, word, &left_quantity, &right_quantity)
        }
        Comparison::Less => {
            winner_with_difference(right, word, &right_quantity, &left_quantity)
        }
        Comparison::Equal => AttributeComparison {
            winner: Some(left.to_string()),
            comparison: format!("{}和{}一样{}", left, right, word),
        },
        Comparison::Uncertain => AttributeComparison {
            winner: None,
            comparison: format!(
                "「{}」和「{}」的{}约数区间重叠，无法确定谁更{}",
                left, right, attribute_label, word
            ),
        },
        Comparison::Incomparable(reason) => AttributeComparison {
            winner: None,
            comparison: format!(
                "「{}」和「{}」的{}不可比（{}）",
                left,
                right,
                attribute_label,
                reason.describe()
            ),
        },
    }
}

fn winner_with_difference(
    winner: &str,
    word: &str,
    winner_quantity: &Quantity,
    loser_quantity: &Quantity,
) -> AttributeComparison {
    let difference = winner_quantity.difference(loser_quantity).ok().map(|value| {
        (
            value.abs(),
            winner_quantity.dimension().canonical_unit(),
        )
    });
    let detail = difference
        .filter(|(value, _)| *value > 1e-9)
        .map(|(value, unit)| {
            let suffix = if winner_quantity.dimension() == quantity::Dimension::Count
                && !winner_quantity.has_unit()
            {
                ""
            } else {
                unit
            };
            format!("（相差{}{}）", quantity::format_number(value), suffix)
        })
        .unwrap_or_default();
    AttributeComparison {
        winner: Some(winner.to_string()),
        comparison: format!("{}更{}{}", winner, word, detail),
    }
}

fn relation_objects(
    knowledge: &dyn DialogueKnowledge,
    subject: &str,
    predicate: &str,
) -> Vec<String> {
    let mut forward: Vec<String> = knowledge
        .entity_relations(subject)
        .into_iter()
        .filter(|(candidate, _)| predicate_matches(knowledge, candidate, predicate))
        .map(|(_, object)| object)
        .collect();
    forward.sort();
    forward.dedup();
    if !forward.is_empty() {
        return forward;
    }
    reverse_relation_objects(knowledge, subject, predicate)
}

fn reverse_relation_objects(
    knowledge: &dyn DialogueKnowledge,
    object: &str,
    predicate: &str,
) -> Vec<String> {
    let mut subjects: Vec<String> = knowledge
        .all_entities()
        .into_iter()
        .filter(|candidate| {
            knowledge.entity_relations(candidate).into_iter().any(
                |(candidate_predicate, candidate_object)| {
                    candidate_object == object
                        && predicate_matches(knowledge, &candidate_predicate, predicate)
                },
            )
        })
        .collect();
    subjects.sort();
    subjects.dedup();
    subjects
}

fn predicate_matches(knowledge: &dyn DialogueKnowledge, candidate: &str, predicate: &str) -> bool {
    candidate == predicate
        || knowledge.predicate_label(candidate) == predicate
        || knowledge.predicate_label(predicate) == candidate
        || knowledge.canonical_predicate(predicate).as_deref() == Some(candidate)
}

fn clarification_candidates(
    _frame: &CompiledFrame,
    slot: &FrameSlotConfig,
    resolved: &Resolved,
    knowledge: &dyn DialogueKnowledge,
) -> Vec<String> {
    let mut candidates = match slot.candidates {
        CandidateSource::None => Vec::new(),
        CandidateSource::Entities => knowledge.all_entities(),
        CandidateSource::Attributes | CandidateSource::Predicates => resolved
            .bindings
            .values()
            .filter(|value| knowledge.known_entity(value))
            .flat_map(|entity| knowledge.known_attributes(entity))
            .map(|attribute| display_value(&attribute, knowledge))
            .collect(),
    };
    candidates.sort();
    candidates.dedup();
    candidates.truncate(FRAME_MAX_CANDIDATES);
    candidates
}

enum PendingReply {
    Candidate(String),
    OutOfRange,
    Unmatched,
}

fn pending_reply(
    pending: &PendingFrame,
    slot: &FrameSlotConfig,
    input: &str,
    knowledge: &dyn DialogueKnowledge,
) -> PendingReply {
    if let Some(index) = parse_choice_index(input) {
        if index >= 1 {
            if let Some(candidate) = pending.candidates.get(index - 1) {
                if let Some(value) = normalize_slot_value(slot, candidate, knowledge) {
                    return PendingReply::Candidate(value);
                }
            }
        }
        return PendingReply::OutOfRange;
    }
    if let Some(candidate) = pending
        .candidates
        .iter()
        .find(|candidate| candidate.as_str() == input)
    {
        if let Some(value) = normalize_slot_value(slot, candidate, knowledge) {
            return PendingReply::Candidate(value);
        }
    }
    if pending.candidates.is_empty() {
        if let Some(value) = normalize_slot_value(slot, input, knowledge) {
            return PendingReply::Candidate(value);
        }
    }
    PendingReply::Unmatched
}

fn is_cancel_reply(input: &str) -> bool {
    matches!(
        input,
        "都不是"
            | "算了"
            | "取消"
            | "不用了"
            | "不查了"
            | "不问了"
            | "结束澄清"
            | "停止澄清"
            | "退出澄清"
    )
}

fn parse_choice_index(text: &str) -> Option<usize> {
    let spec = text
        .trim()
        .trim_start_matches('第')
        .trim_end_matches(|ch: char| matches!(ch, '.' | '。' | '、' | '个' | ')' | '）' | ' '));
    if spec.is_empty() {
        return None;
    }
    if spec.chars().all(|ch| ch.is_ascii_digit()) {
        return spec.parse().ok();
    }
    match spec {
        "一" => Some(1),
        "二" => Some(2),
        "三" => Some(3),
        "四" => Some(4),
        "五" => Some(5),
        "六" => Some(6),
        "七" => Some(7),
        "八" => Some(8),
        "九" => Some(9),
        "十" => Some(10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(source: &str) -> DialogueFramesConfig {
        toml::from_str(source).expect("测试配置应可解析")
    }

    fn program() -> FrameProgram {
        FrameProgram::compile(&config(r#"
enabled = true

[[frames]]
id = "person.attribute"
priority = 20
patterns = ["{entity}的{attribute}是什么", "{attribute}呢", "{attribute}是多少"]
realize = { answer = "{entity}的{attribute}是{values}。" }

[[frames.slots]]
name = "entity"
sort = "entity"
required = true
inherit = true
candidates = "entities"

[[frames.slots]]
name = "attribute"
sort = "attribute"
required = true
inherit = true
candidates = "attributes"
ask = "你想问谁的「{slot}」？"

[[frames.slots]]
name = "value"
sort = "text"

[frames.goal]
kind = "attribute"
entity = "entity"
attribute = "attribute"
output = "value"

[[frames]]
id = "person.compare"
priority = 30
patterns = ["{left}和{right}的{attribute}谁更高"]
realize = { answer = "{left}的{attribute}是{left_value}，{right}的{attribute}是{right_value}；{comparison}。" }

[[frames.slots]]
name = "left"
sort = "entity"
required = true
candidates = "entities"

[[frames.slots]]
name = "right"
sort = "entity"
required = true
candidates = "entities"

[[frames.slots]]
name = "attribute"
sort = "attribute"
required = true
inherit = true
candidates = "attributes"

[[frames.slots]]
name = "winner"
sort = "text"

[frames.goal]
kind = "compare"
left = "left"
right = "right"
attribute = "attribute"
output = "winner"
"#)).expect("配置应可编译")
    }

    struct FakeKnowledge {
        entities: Vec<String>,
        attributes: BTreeMap<String, Vec<(String, String)>>,
        relations: BTreeMap<String, Vec<(String, String)>>,
    }

    impl FakeKnowledge {
        fn new() -> Self {
            let mut attributes = BTreeMap::new();
            attributes.insert(
                "姚明".to_string(),
                vec![
                    ("height".to_string(), "226".to_string()),
                    ("weight".to_string(), "140".to_string()),
                ],
            );
            attributes.insert(
                "叶莉".to_string(),
                vec![("height".to_string(), "190".to_string())],
            );
            let mut relations = BTreeMap::new();
            relations.insert(
                "张三".to_string(),
                vec![("师傅".to_string(), "李四".to_string())],
            );
            relations.insert(
                "李四".to_string(),
                vec![("师傅".to_string(), "王五".to_string())],
            );
            Self {
                entities: vec![
                    "姚明".to_string(),
                    "叶莉".to_string(),
                    "张三".to_string(),
                    "李四".to_string(),
                    "王五".to_string(),
                ],
                attributes,
                relations,
            }
        }
    }

    impl DialogueKnowledge for FakeKnowledge {
        fn entities_in(&self, text: &str) -> Vec<String> {
            self.entities
                .iter()
                .filter(|entity| text.contains(entity.as_str()))
                .cloned()
                .collect()
        }

        fn known_entity(&self, entity: &str) -> bool {
            self.entities.iter().any(|candidate| candidate == entity)
        }

        fn answer(&self, _question: &str) -> Option<String> {
            None
        }

        fn known_attributes(&self, entity: &str) -> Vec<String> {
            let mut values: Vec<String> = self
                .attributes
                .get(entity)
                .map(|pairs| pairs.iter().map(|(key, _)| key.clone()).collect())
                .unwrap_or_default();
            values.extend(
                self.relations
                    .get(entity)
                    .map(|pairs| {
                        pairs
                            .iter()
                            .map(|(key, _)| key.clone())
                            .collect::<Vec<String>>()
                    })
                    .unwrap_or_default(),
            );
            values.sort();
            values.dedup();
            values
        }

        fn attribute_values(&self, entity: &str, attribute: &str) -> Vec<String> {
            self.attributes
                .get(entity)
                .map(|pairs| {
                    pairs
                        .iter()
                        .filter(|(key, _)| {
                            key == attribute
                                || crate::symbolic_generator::canonical_predicate_key(attribute)
                                    == Some(key.as_str())
                        })
                        .map(|(_, value)| value.clone())
                        .collect()
                })
                .unwrap_or_default()
        }

        fn all_entities(&self) -> Vec<String> {
            self.entities.clone()
        }

        fn entity_relations(&self, entity: &str) -> Vec<(String, String)> {
            self.relations.get(entity).cloned().unwrap_or_default()
        }

        fn canonical_predicate(&self, surface: &str) -> Option<String> {
            crate::symbolic_generator::canonical_predicate_key(surface)
                .map(str::to_string)
                .or_else(|| (surface == "师傅" || surface == "徒弟").then(|| surface.to_string()))
        }

        fn predicate_label(&self, predicate: &str) -> String {
            crate::symbolic_generator::localized_predicate_key(predicate)
                .map(str::to_string)
                .unwrap_or_else(|| predicate.to_string())
        }
    }

    #[test]
    fn compile_rejects_undeclared_template_variable() {
        let config = config(
            r#"
enabled = true
[[frames]]
id = "bad"
patterns = ["{entity}是谁"]
realize = { answer = "{missing}" }
[[frames.slots]]
name = "entity"
sort = "entity"
required = true
[frames.goal]
kind = "attribute"
entity = "entity"
attribute = "const:name"
output = "value"
"#,
        );
        let errors = FrameProgram::compile(&config).expect_err("应拒绝未声明变量");
        assert!(errors
            .iter()
            .any(|error| error.code == "frame_output_undeclared"));
        assert!(errors
            .iter()
            .any(|error| error.code == "frame_template_undeclared"));
    }

    #[test]
    fn compile_rejects_required_slot_without_source() {
        let config = config(
            r#"
enabled = true
[[frames]]
id = "bad"
patterns = ["{attribute}呢"]
[[frames.slots]]
name = "entity"
sort = "entity"
required = true
[[frames.slots]]
name = "attribute"
sort = "attribute"
required = true
[[frames.slots]]
name = "value"
sort = "text"
[frames.goal]
kind = "attribute"
entity = "entity"
attribute = "attribute"
output = "value"
"#,
        );
        let errors = FrameProgram::compile(&config).expect_err("应拒绝不可达必填槽");
        assert!(errors
            .iter()
            .any(|error| error.code == "frame_required_unreachable"));
    }

    #[test]
    fn full_question_executes_attribute_goal() {
        let program = program();
        let mut memory = FrameMemory::new();
        let outcome = program
            .plan(&mut memory, "姚明的身高是什么？", &FakeKnowledge::new(), 1)
            .expect("应回答");
        assert!(outcome.text.contains("226"), "{}", outcome.text);
        assert_eq!(memory.answered(), 1);
        assert!(outcome
            .plan
            .iter()
            .any(|plan| plan == "frame:person.attribute"));
    }

    #[test]
    fn followup_inherits_entity_from_environment() {
        let program = program();
        let mut memory = FrameMemory::new();
        let knowledge = FakeKnowledge::new();
        program
            .plan(&mut memory, "姚明的身高是什么", &knowledge, 1)
            .unwrap();
        let outcome = program
            .plan(&mut memory, "体重呢", &knowledge, 2)
            .expect("应继承实体");
        assert!(outcome.text.contains("140"), "{}", outcome.text);
        assert!(outcome
            .provenance
            .iter()
            .any(|note| note.contains("person.attribute")));
    }

    #[test]
    fn clarification_binds_by_candidate_index() {
        let program = program();
        let mut memory = FrameMemory::new();
        let knowledge = FakeKnowledge::new();
        let question = program
            .plan(&mut memory, "身高是多少", &knowledge, 1)
            .expect("缺实体应澄清");
        assert!(question.text.contains("1. 叶莉"), "{}", question.text);
        let outcome = program
            .plan(&mut memory, "1", &knowledge, 2)
            .expect("应完成绑定");
        assert!(outcome.text.contains("叶莉"), "{}", outcome.text);
        assert!(outcome
            .plan
            .iter()
            .any(|plan| plan == "frame_clarification_bound"));
    }

    #[test]
    fn unknown_entity_falls_back_without_state_change() {
        let program = program();
        let mut memory = FrameMemory::new();
        assert!(program
            .plan(&mut memory, "某甲的身高是什么", &FakeKnowledge::new(), 1)
            .is_none());
        assert!(memory.is_empty());
    }

    #[test]
    fn compare_uses_numeric_order_not_lexicographic() {
        let program = program();
        let mut memory = FrameMemory::new();
        let outcome = program
            .plan(
                &mut memory,
                "姚明和叶莉的身高谁更高",
                &FakeKnowledge::new(),
                1,
            )
            .expect("应比较成功");
        assert!(outcome.text.contains("226"), "{}", outcome.text);
        assert!(outcome.text.contains("190"), "{}", outcome.text);
        assert!(outcome.text.contains("姚明更高"), "{}", outcome.text);
    }

    struct UnitKnowledge;

    impl DialogueKnowledge for UnitKnowledge {
        fn entities_in(&self, text: &str) -> Vec<String> {
            ["姚明", "叶莉"]
                .iter()
                .filter(|entity| text.contains(**entity))
                .map(|entity| (*entity).to_string())
                .collect()
        }

        fn known_entity(&self, entity: &str) -> bool {
            matches!(entity, "姚明" | "叶莉")
        }

        fn answer(&self, _question: &str) -> Option<String> {
            None
        }

        fn known_attributes(&self, entity: &str) -> Vec<String> {
            match entity {
                "姚明" | "叶莉" => vec!["height".to_string()],
                _ => Vec::new(),
            }
        }

        fn attribute_values(&self, entity: &str, attribute: &str) -> Vec<String> {
            match (entity, attribute) {
                ("姚明", "height") => vec!["226 厘米".to_string()],
                ("叶莉", "height") => vec!["1.90 米".to_string()],
                _ => Vec::new(),
            }
        }

        fn all_entities(&self) -> Vec<String> {
            vec!["姚明".to_string(), "叶莉".to_string()]
        }

        fn canonical_predicate(&self, surface: &str) -> Option<String> {
            crate::symbolic_generator::canonical_predicate_key(surface).map(str::to_string)
        }

        fn predicate_label(&self, predicate: &str) -> String {
            crate::symbolic_generator::localized_predicate_key(predicate)
                .map(str::to_string)
                .unwrap_or_else(|| predicate.to_string())
        }
    }

    #[test]
    fn compare_parses_units_from_real_knowledge_values() {
        let program = program();
        let mut memory = FrameMemory::new();
        let outcome = program
            .plan(
                &mut memory,
                "姚明和叶莉的身高谁更高",
                &UnitKnowledge,
                1,
            )
            .expect("带单位的值应能比较");
        assert!(outcome.text.contains("姚明更高"), "{}", outcome.text);
        assert!(
            outcome.text.contains("相差0.36米"),
            "比较应给出同量纲差值：{}",
            outcome.text
        );
    }

    #[test]
    fn compare_refuses_cross_dimension_and_uncertain_approximations() {
        let cross = compare_attribute_values(
            "法国",
            "珠峰",
            "面积",
            "67万平方公里",
            "8848米",
        );
        assert!(cross.winner.is_none());
        assert!(cross.comparison.contains("量纲不同"), "{}", cross.comparison);

        let approximate =
            compare_attribute_values("甲", "乙", "年龄", "46亿年", "约46亿年");
        assert!(approximate.winner.is_none());
        assert!(
            approximate.comparison.contains("无法确定"),
            "{}",
            approximate.comparison
        );
    }

    #[test]
    fn memory_and_pending_survive_serde_roundtrip() {
        let program = program();
        let mut memory = FrameMemory::new();
        let knowledge = FakeKnowledge::new();
        program
            .plan(&mut memory, "身高是多少", &knowledge, 1)
            .unwrap();
        let json = serde_json::to_string(&memory).expect("应可序列化");
        let restored: FrameMemory = serde_json::from_str(&json).expect("应可反序列化");
        assert_eq!(memory, restored);
        assert!(restored.pending_prompt().is_some());
    }

    #[test]
    fn old_frames_without_optional_fields_still_parse() {
        let config = config(
            r#"
enabled = true
[[frames]]
id = "minimal"
patterns = ["{entity}的{attribute}是什么"]
[[frames.slots]]
name = "entity"
sort = "entity"
required = true
[[frames.slots]]
name = "attribute"
sort = "attribute"
required = true
[[frames.slots]]
name = "value"
sort = "text"
[frames.goal]
kind = "attribute"
entity = "entity"
attribute = "attribute"
output = "value"
"#,
        );
        let program = FrameProgram::compile(&config).expect("应可编译");
        let mut memory = FrameMemory::new();
        assert!(program
            .plan(&mut memory, "姚明的高度是什么", &FakeKnowledge::new(), 1)
            .is_none());
    }
}

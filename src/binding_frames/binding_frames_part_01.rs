use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use serde::{Deserialize, Serialize};
use crate::conversational_graph::{GraphFact, MatchOptions};
use crate::dialogue_manager::{DialogueKnowledge, MoveKind};
use crate::kg_pattern::{KgPatternDecl, KgPatternRegistry};
use crate::quantity::{self, Comparison, Quantity};
/// 继承槽位的逻辑轮次时限。
pub const FRAME_ENV_TTL: u64 = 8;
/// 待澄清帧的逻辑轮次时限。
pub const FRAME_PENDING_TTL: u64 = 3;
/// 模板实例（答案集合与游标）的逻辑轮次时限。
pub const FRAME_SESSION_TTL: u64 = 16;
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
    /// 帧目标可引用的图谱模式；与帧一起编译，避免运行期再解析字符串。
    #[serde(default)]
    pub kg_patterns: Vec<KgPatternDecl>,
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
    /// 任意已编译图谱模式：`bind` 把模式变量映射到帧槽位。
    Pattern {
        pattern: String,
        #[serde(default)]
        bind: BTreeMap<String, String>,
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
    Pattern {
        pattern: String,
        bind: BTreeMap<String, String>,
        inputs: Vec<String>,
        outputs: Vec<String>,
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
            CompiledGoal::Pattern { .. } => "pattern",
        }
    }

    fn output_slots(&self) -> Vec<String> {
        match self {
            CompiledGoal::Attribute { output, .. }
            | CompiledGoal::Relation { output, .. }
            | CompiledGoal::ReverseRelation { output, .. }
            | CompiledGoal::Count { output, .. }
            | CompiledGoal::Compare { output, .. } => vec![output.clone()],
            CompiledGoal::Pattern { outputs, .. } => outputs.clone(),
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
            CompiledGoal::Pattern { inputs, .. } => {
                return inputs.clone();
            }
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
    kg_patterns: KgPatternRegistry,
}
impl FrameProgram {
    pub fn compile(config: &DialogueFramesConfig) -> Result<Self, Vec<FrameCompileError>> {
        let mut errors = Vec::new();
        let kg_patterns = match KgPatternRegistry::from_declarations(&config.kg_patterns) {
            Ok(registry) => registry,
            Err(found) => {
                errors.extend(found.into_iter().map(|error| FrameCompileError {
                    code: error.code,
                    path: format!("dialogue_frames.{}", error.path),
                    message: error.message,
                }));
                KgPatternRegistry::empty()
            }
        };

        let mut frames = Vec::new();
        let mut seen_ids = BTreeSet::new();

        for (frame_index, schema) in config.frames.iter().enumerate() {
            let frame = compile_frame(
                frame_index,
                schema,
                &mut seen_ids,
                &kg_patterns,
                &mut errors,
            );
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
            kg_patterns,
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

    pub fn kg_patterns(&self) -> &KgPatternRegistry {
        &self.kg_patterns
    }

    fn frame(&self, id: &str) -> Option<&CompiledFrame> {
        self.frames.iter().find(|frame| frame.id == id)
    }

    /// 规划一轮：先结算待澄清帧与歧义选择，再按确定序尝试模式。
    ///
    /// 与「第一条命中即返回」不同，这里会收集所有表面命中且目标可执行的
    /// 候选，再按优先级与模式特化度仲裁：严格更具体的候选胜出；若多个
    /// 同样具体的候选证明出不同答案，则发起一次显式歧义澄清，不静默择一。
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
        if memory.pending_ambiguity.is_some() {
            if let Some(outcome) = self.resolve_ambiguity(memory, &normalized, turn) {
                return Some(outcome);
            }
        }
        if let Some(outcome) = self.enumerate_session(memory, &normalized, turn) {
            return Some(outcome);
        }

        let mut proven: Vec<(usize, usize, Resolved, Executed)> = Vec::new();
        let mut question: Option<(usize, usize, Resolved)> = None;
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
                    question = Some((*frame_index, *pattern_index, resolved));
                }
                continue;
            }
            if let Some(executed) = self.execute(frame, &resolved, knowledge) {
                proven.push((*frame_index, *pattern_index, resolved, executed));
            }
        }

        if !proven.is_empty() {
            return Some(self.arbitrate_proven(memory, &normalized, proven, turn));
        }

        let (frame_index, _pattern_index, resolved) = question?;
        let frame = &self.frames[frame_index];
        let question = self.question_for(frame, &resolved, knowledge, input, turn)?;
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

    /// 仲裁所有已证明候选。排序键与 `ranked` 一致，但把 id/词面从
    /// 「决胜键」降为「稳定序」：同优先级、同特化度的不同解释必须澄清。
    fn arbitrate_proven(
        &self,
        memory: &mut FrameMemory,
        raw: &str,
        proven: Vec<(usize, usize, Resolved, Executed)>,
        turn: u64,
    ) -> FrameOutcome {
        let top_key = self.specificity(proven[0].0, proven[0].1);
        let tied: Vec<&(usize, usize, Resolved, Executed)> = proven
            .iter()
            .filter(|candidate| self.specificity(candidate.0, candidate.1) == top_key)
            .collect();
        let mut distinct: Vec<&(usize, usize, Resolved, Executed)> = Vec::new();
        let mut signatures = BTreeSet::new();
        for candidate in &tied {
            if signatures.insert(answer_signature(&candidate.3)) {
                distinct.push(candidate);
            }
        }

        if distinct.len() > 1 {
            let shown = distinct.len().min(FRAME_MAX_CANDIDATES);
            let candidates: Vec<StoredCandidate> = distinct
                .iter()
                .take(shown)
                .map(|candidate| self.stored_candidate(candidate))
                .collect();
            let prompt = ambiguity_prompt(raw, &candidates, distinct.len(), distinct.len() > shown);
            memory.pending = None;
            memory.pending_ambiguity = Some(PendingAmbiguity {
                raw: raw.to_string(),
                prompt: prompt.clone(),
                candidates,
                turn,
            });
            memory.turn = turn;
            memory.ambiguities_asked += 1;
            return FrameOutcome {
                text: prompt,
                plan: vec![
                    "frame_ambiguity".to_string(),
                    format!("frame_ambiguity_candidates:{}", distinct.len()),
                ],
                provenance: vec![format!(
                    "问题「{}」有 {} 个同样具体的绑定帧目标都可证明，未静默择一",
                    raw,
                    distinct.len()
                )],
                kind: MoveKind::AskAttribute,
                entity: None,
                attribute: None,
            };
        }

        let tied_len = tied.len();
        let proven_len = proven.len();
        let (frame_index, pattern_index, resolved, executed) = proven.into_iter().next().unwrap();
        let frame = &self.frames[frame_index];
        let pattern = &frame.patterns[pattern_index];
        memory.commit(frame, &executed.bindings, &resolved.origins, turn);
        memory.pending = None;
        memory.pending_ambiguity = None;
        self.remember_session(memory, frame, pattern, &executed, turn);
        let mut outcome = self.outcome(frame, pattern, executed, turn);
        if tied_len > 1 {
            outcome
                .plan
                .push(format!("frame_ambiguous_same_answer:{}", tied_len));
            outcome.provenance.push(format!(
                "另有 {} 个同样具体的候选给出相同答案，已合并采用",
                tied_len - 1
            ));
        } else if proven_len > 1 {
            outcome
                .plan
                .push(format!("frame_candidates:{}", proven_len));
        }
        outcome
    }

    fn specificity(&self, frame_index: usize, pattern_index: usize) -> (i32, usize, usize) {
        let frame = &self.frames[frame_index];
        let pattern = &frame.patterns[pattern_index];
        (
            frame.priority,
            pattern.literal_chars,
            usize::MAX - pattern.slots.len(),
        )
    }

    fn stored_candidate(
        &self,
        candidate: &(usize, usize, Resolved, Executed),
    ) -> StoredCandidate {
        let (frame_index, pattern_index, resolved, executed) = candidate;
        let frame = &self.frames[*frame_index];
        StoredCandidate {
            frame: frame.id.clone(),
            pattern: frame.patterns[*pattern_index].source.clone(),
            bindings: executed.bindings.clone(),
            origins: resolved.origins.clone(),
            text: executed.text.clone(),
            evidence: executed.evidence.clone(),
            entity: executed.entity.clone(),
            attribute: executed.attribute.clone(),
            answer_values: executed.answer_values.clone(),
        }
    }

    /// 结算一次歧义选择：用户回复序号或完整候选答案后，才提交槽位记忆。
    fn resolve_ambiguity(
        &self,
        memory: &mut FrameMemory,
        input: &str,
        turn: u64,
    ) -> Option<FrameOutcome> {
        let pending = memory.pending_ambiguity.clone()?;
        if turn.saturating_sub(pending.turn) > FRAME_PENDING_TTL {
            memory.pending_ambiguity = None;
            return None;
        }
        if is_cancel_reply(input) {
            memory.pending_ambiguity = None;
            memory.ambiguities_cancelled += 1;
            return Some(FrameOutcome {
                text: "好的，这次追问先放下。".to_string(),
                plan: vec![
                    "frame_ambiguity_cancel".to_string(),
                    "frame_ambiguity".to_string(),
                ],
                provenance: vec!["绑定帧的歧义澄清被用户撤销".to_string()],
                kind: MoveKind::Other,
                entity: None,
                attribute: None,
            });
        }
        let index = match parse_choice_index(input) {
            Some(index) if index >= 1 && index <= pending.candidates.len() => index - 1,
            Some(_) => {
                memory.pending_ambiguity = Some(pending.clone());
                return Some(FrameOutcome {
                    text: pending.prompt.clone(),
                    plan: vec![
                        "frame_ambiguity_reprompt".to_string(),
                        "frame_ambiguity".to_string(),
                    ],
                    provenance: vec!["歧义序号越界，已原样重问（零状态变化）".to_string()],
                    kind: MoveKind::AskAttribute,
                    entity: None,
                    attribute: None,
                });
            }
            None => match pending.candidates.iter().position(|candidate| {
                normalize_utterance(&candidate.text) == input
            }) {
                Some(index) => index,
                None => {
                    memory.pending_ambiguity = None;
                    return None;
                }
            },
        };
        let candidate = pending.candidates[index].clone();
        let Some(frame) = self.frame(&candidate.frame) else {
            memory.pending_ambiguity = None;
            return None;
        };
        memory.commit(frame, &candidate.bindings, &candidate.origins, turn);
        memory.pending_ambiguity = None;
        memory.pending = None;
        memory.ambiguities_resolved += 1;
        let pattern = CompiledPattern {
            source: candidate.pattern,
            parts: Vec::new(),
            slots: Vec::new(),
            literal_chars: 0,
        };
        let executed = Executed {
            text: candidate.text,
            bindings: candidate.bindings,
            evidence: candidate.evidence,
            entity: candidate.entity,
            attribute: candidate.attribute,
            answer_values: candidate.answer_values,
        };
        self.remember_session(memory, frame, &pattern, &executed, turn);
        let mut outcome = self.outcome(frame, &pattern, executed, turn);
        outcome
            .plan
            .insert(0, "frame_ambiguity_resolved".to_string());
        Some(outcome)
    }

    /// 记录本轮答案集合与游标，供「还有呢 / 第 2 个 / 全部」等追问复用。
    fn remember_session(
        &self,
        memory: &mut FrameMemory,
        frame: &CompiledFrame,
        pattern: &CompiledPattern,
        executed: &Executed,
        turn: u64,
    ) {
        let values = executed.answer_values.clone();
        if values.is_empty() {
            memory.session = None;
            return;
        }
        let lists_all = frame
            .realize_answer
            .as_deref()
            .map(|template| template.contains("{values}") || template.contains("{objects}"))
            .unwrap_or(true);
        let spoken = if lists_all { values.len() } else { 1 };
        memory.session = Some(FrameSession {
            frame: frame.id.clone(),
            pattern: pattern.source.clone(),
            bindings: executed.bindings.clone(),
            values,
            spoken,
            turn,
            entity: executed.entity.clone(),
            attribute: executed.attribute.clone(),
        });
    }

    /// 枚举追问：只读取上一轮的模板实例，不重新解释问句，也不改写槽位环境。
    fn enumerate_session(
        &self,
        memory: &mut FrameMemory,
        input: &str,
        turn: u64,
    ) -> Option<FrameOutcome> {
        let request = parse_enumeration(input)?;
        let mut session = memory.session.clone()?;
        if turn.saturating_sub(session.turn) > FRAME_SESSION_TTL {
            memory.session = None;
            return None;
        }
        let total = session.values.len();
        let (text, plan_tag) = match request {
            EnumerationRequest::Next => {
                if session.spoken >= total {
                    (
                        format!("没有更多了：之前已列出全部 {} 项。", total),
                        "frame_enumerate_end",
                    )
                } else {
                    let value = session.values[session.spoken].clone();
                    session.spoken += 1;
                    (format!("还有：{}。", value), "frame_enumerate_next")
                }
            }
            EnumerationRequest::All => {
                session.spoken = total;
                (
                    format!("全部 {} 项：{}。", total, session.values.join("、")),
                    "frame_enumerate_all",
                )
            }
            EnumerationRequest::Index(index) => {
                if index == 0 || index > total {
                    (
                        format!("没有第{}项：当前共 {} 项。", index, total),
                        "frame_enumerate_missing",
                    )
                } else {
                    session.spoken = session.spoken.max(index);
                    (
                        format!("第{}项是：{}。", index, session.values[index - 1]),
                        "frame_enumerate_index",
                    )
                }
            }
        };
        let frame_id = session.frame.clone();
        let entity = session.entity.clone();
        let attribute = session.attribute.clone();
        session.turn = turn;
        memory.session = Some(session);
        memory.turn = turn;
        memory.enumerations += 1;
        Some(FrameOutcome {
            text,
            plan: vec![plan_tag.to_string(), format!("frame:{}", frame_id)],
            provenance: vec![format!(
                "沿用绑定帧「{}」上一轮保存的答案集合（{} 项）",
                frame_id, total
            )],
            kind: MoveKind::AskAttribute,
            entity,
            attribute,
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
            let pattern = CompiledPattern {
                source: format!("clarification:{}", pending.raw),
                parts: Vec::new(),
                slots: Vec::new(),
                literal_chars: 0,
            };
            self.remember_session(memory, frame, &pattern, &executed, turn);
            let mut outcome = self.outcome(frame, &pattern, executed, turn);
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

    /// 执行目标并把证明写回措辞：模板可以引用 `{evidence}` / `{proof}`，
    /// 但这些变量只由本轮已得证的事实填充，不会凭空补全。
    fn execute(
        &self,
        frame: &CompiledFrame,
        resolved: &Resolved,
        knowledge: &dyn DialogueKnowledge,
    ) -> Option<Executed> {
        let mut executed = self.execute_inner(frame, resolved, knowledge)?;
        let Some(template) = frame.realize_answer.as_deref() else {
            return Some(executed);
        };
        if !template.contains("{evidence}") && !template.contains("{proof}") {
            return Some(executed);
        }
        let mut vars = self.render_vars(frame, &executed.bindings, knowledge);
        let evidence = executed.evidence.join("；");
        vars.insert("evidence".to_string(), evidence.clone());
        vars.insert(
            "proof".to_string(),
            format!("绑定帧「{}」：{}", frame.id, evidence),
        );
        if let Some(text) = render_template(template, &vars) {
            executed.text = text;
        }
        Some(executed)
    }

    fn execute_inner(
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
                    answer_values: values,
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
                    answer_values: objects,
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
                    answer_values: subjects,
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
                    answer_values: Vec::new(),
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
                    answer_values: Vec::new(),
                })
            }
            CompiledGoal::Pattern {
                pattern,
                bind,
                inputs,
                ..
            } => {
                let spec = self.kg_patterns.get(pattern)?;
                if inputs
                    .iter()
                    .any(|slot| !resolved.bindings.contains_key(slot))
                {
                    return None;
                }
                let mut input = BTreeMap::new();
                for (variable, slot) in bind {
                    if let Some(value) = resolved.bindings.get(slot) {
                        input.insert(variable.clone(), value.clone());
                    }
                }
                let seeds: Vec<String> = inputs
                    .iter()
                    .filter_map(|slot| resolved.bindings.get(slot).cloned())
                    .collect();
                let (facts, complete) = kg_facts_from_knowledge(knowledge, &seeds);
                let result = spec.execute(
                    &facts,
                    &input,
                    complete,
                    MatchOptions::new(KG_FRAME_MAX_SOLUTIONS, KG_FRAME_MAX_EXPANSIONS),
                );
                if !result.is_proven() {
                    return None;
                }
                let values = result.answer_values();
                let mut bindings = resolved.bindings.clone();
                bindings.insert("values".to_string(), values.join("、"));
                bindings.insert("count".to_string(), result.total.to_string());
                for (variable, slot) in bind {
                    if spec.is_selected(variable) {
                        if let Some(value) = result.first_value(variable) {
                            bindings.insert(slot.clone(), value.to_string());
                        }
                    }
                }
                let vars = self.render_vars(frame, &bindings, knowledge);
                let text = frame
                    .realize_answer
                    .as_ref()
                    .and_then(|template| render_template(template, &vars))
                    .unwrap_or_else(|| {
                        if result.total == 0 {
                            "没有找到匹配的记录。".to_string()
                        } else {
                            format!("查询结果：{}。", values.join("、"))
                        }
                    });
                let mut evidence = vec![format!(
                    "图谱模式「{}」{}（指纹 {}）",
                    spec.id(),
                    if spec.description().is_empty() {
                        String::new()
                    } else {
                        format!("：{}", spec.description())
                    },
                    spec.semantic_fingerprint()
                )];
                evidence.extend(result.evidence_lines());
                let entity = inputs
                    .iter()
                    .find(|slot| {
                        frame.slots.get(*slot).map(|slot| slot.sort) == Some(SlotSort::Entity)
                    })
                    .and_then(|slot| bindings.get(slot))
                    .cloned()
                    .or_else(|| {
                        bind.iter()
                            .find(|(variable, _)| spec.is_selected(variable))
                            .and_then(|(_, slot)| bindings.get(slot))
                            .cloned()
                    });
                let attribute = inputs
                    .iter()
                    .find(|slot| {
                        matches!(
                            frame.slots.get(*slot).map(|slot| slot.sort),
                            Some(SlotSort::Predicate | SlotSort::Attribute)
                        )
                    })
                    .and_then(|slot| bindings.get(slot))
                    .cloned();
                Some(Executed {
                    text,
                    bindings,
                    evidence,
                    entity,
                    attribute,
                    answer_values: values,
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
            CompiledGoal::Pattern { .. } => bindings
                .get("values")
                .map(|value| vec![value.clone()])
                .unwrap_or_default(),
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
        vars.insert(
            "count".to_string(),
            match &frame.goal {
                CompiledGoal::Pattern { .. } => bindings
                    .get("count")
                    .cloned()
                    .unwrap_or_else(|| values.len().to_string()),
                _ => values.len().to_string(),
            },
        );
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
    kg_patterns: &KgPatternRegistry,
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

    let goal = match compile_goal(
        &schema.goal,
        &slots,
        kg_patterns,
        &format!("{}.goal", base),
        errors,
    ) {
        Some(goal) => goal,
        None => return None,
    };
    let outputs = goal.output_slots();
    for output in &outputs {
        if !slots.contains_key(output) {
            errors.push(FrameCompileError {
                code: "frame_output_undeclared",
                path: format!("{}.goal.output", base),
                message: format!(
                    "目标输出槽「{}」未声明；答案模板无法安全渲染未声明变量",
                    output
                ),
            });
        }
        if goal.input_slots().iter().any(|slot| slot == output) {
            errors.push(FrameCompileError {
                code: "frame_output_is_input",
                path: format!("{}.goal.output", base),
                message: format!("槽位「{}」同时被目标读取与写入", output),
            });
        }
        if slots
            .get(output)
            .map(|slot| slot.required)
            .unwrap_or(false)
        {
            errors.push(FrameCompileError {
                code: "frame_output_required",
                path: format!("{}.slots", base),
                message: format!("输出槽「{}」不能声明为必填输入槽", output),
            });
        }
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
        if outputs.iter().any(|output| output == name) {
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
    kg_patterns: &KgPatternRegistry,
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
        FrameGoalConfig::Pattern { pattern, bind } => {
            let Some(spec) = kg_patterns.get(pattern.trim()) else {
                errors.push(FrameCompileError {
                    code: "frame_goal_pattern_unknown",
                    path: format!("{}.pattern", path),
                    message: format!(
                        "帧目标引用了未声明的图谱模式「{}」；请先在 dialogue_frames.kg_patterns 中声明",
                        pattern
                    ),
                });
                return None;
            };
            let variables = spec.variables();
            let mut inputs = Vec::new();
            let mut outputs = Vec::new();
            let mut compiled_bind = BTreeMap::new();
            for (variable, slot) in bind {
                let name = variable.trim().trim_start_matches('?');
                if !variables.contains(name) {
                    errors.push(FrameCompileError {
                        code: "frame_goal_pattern_bind_unknown",
                        path: format!("{}.bind.{}", path, variable),
                        message: format!(
                            "绑定引用了图谱模式「{}」中不存在的变量「{}」",
                            spec.id(),
                            name
                        ),
                    });
                    continue;
                }
                if !slots.contains_key(slot) {
                    errors.push(FrameCompileError {
                        code: "frame_goal_slot_unknown",
                        path: format!("{}.bind.{}", path, variable),
                        message: format!("模式绑定引用了未声明槽位「{}」", slot),
                    });
                    continue;
                }
                if spec.is_selected(name) {
                    outputs.push(slot.clone());
                } else {
                    inputs.push(slot.clone());
                }
                compiled_bind.insert(name.to_string(), slot.clone());
            }
            for variable in spec.select() {
                if !compiled_bind.contains_key(variable) {
                    errors.push(FrameCompileError {
                        code: "frame_goal_pattern_output_unbound",
                        path: format!("{}.bind", path),
                        message: format!(
                            "图谱模式「{}」的投影变量「{}」没有绑定到输出槽",
                            spec.id(),
                            variable
                        ),
                    });
                }
            }
            inputs.sort();
            inputs.dedup();
            outputs.sort();
            outputs.dedup();
            CompiledGoal::Pattern {
                pattern: spec.id().to_string(),
                bind: compiled_bind,
                inputs,
                outputs,
            }
        }
    })
}

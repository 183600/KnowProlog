//! 分层追踪图（Layered Trace Graph）。
//!
//! 把多轮对话的“决策”显式化为可回放、可查询的图，而不是事后拼接的日志行：
//! - **L0 表层**：本轮输入摘录（不可再分的观察）；
//! - **L1 语义帧**：话语行为、实体槽、属性槽、规范化问句；
//! - **L2 话语记忆**：指代决策（[`ReferentTrail`]）、问据栈（[`QudStack`]）、
//!   更正与失效传播（值覆盖 + Stale 标记）。
//!
//! 每个 [`TraceNode`] 代表一次不可再分的决策，携带 `inputs`（证据边）、`support`
//! （事实/证明/规则/用户输入）、`status` 与 `alternatives`。「为什么/依据」类问题
//! 沿 `inputs` 边回溯（[`TraceStore::render_why`]），而不是再查一遍知识库。
//!
//! 模式与失败恢复同构：[`TraceMode::Off`] 不记录（默认，行为逐字节兼容）、
//! [`TraceMode::Observe`] 只记录不改回答、[`TraceMode::On`] 让追踪参与理解与生成
//! （谓词焦点继承、更正覆盖与失效标记）。

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};

/// 追踪图结构版本（进快照与编年史回放）。
pub const TRACE_GRAPH_VERSION: u32 = 1;

/// 会话内保留的轮迹上限（工作记忆；编年史另有完整档案）。
pub const DEFAULT_MAX_TURNS: usize = 64;
/// 单轮决策节点上限（超出折叠为摘要节点，保留计数）。
pub const DEFAULT_MAX_NODES_PER_TURN: usize = 64;
/// 单轮指代事件上限。
pub const DEFAULT_MAX_EVENTS_PER_TURN: usize = 16;
/// 问据栈容量。
pub const DEFAULT_QUD_CAP: usize = 32;
/// 指代轨迹容量。
pub const DEFAULT_TRAIL_CAP: usize = 128;

/// 追踪三档模式（与失败恢复 [`crate::recovery::RecoveryMode`] 同构）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceMode {
    /// 不记录、不介入（默认；与引入追踪前的行为逐字节一致）。
    Off,
    /// 只记录决策与状态，不改变回答文本。
    Observe,
    /// 追踪参与理解与生成：谓词焦点继承、更正覆盖、失效传播。
    On,
}

impl Default for TraceMode {
    fn default() -> Self {
        TraceMode::Off
    }
}

impl TraceMode {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_lowercase().as_str() {
            "off" => Some(TraceMode::Off),
            "observe" => Some(TraceMode::Observe),
            "on" => Some(TraceMode::On),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            TraceMode::Off => "off",
            TraceMode::Observe => "observe",
            TraceMode::On => "on",
        }
    }

    /// 是否记录决策节点。
    pub fn records(self) -> bool {
        !matches!(self, TraceMode::Off)
    }

    pub fn is_off(self) -> bool {
        matches!(self, TraceMode::Off)
    }

    /// 是否让追踪参与理解与生成（谓词焦点继承、更正覆盖）。
    pub fn is_on(self) -> bool {
        matches!(self, TraceMode::On)
    }
}

/// 分层身份：每条决策属于且仅属于一层；层是证据责任的契约，不是目录。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceLayer {
    /// L0 表层：输入摘录与观察。
    L0Surface,
    /// L1 语义帧：话语行为与槽位。
    L1Semantic,
    /// L2 话语记忆：问据、指代、更正与回复落地。
    L2Discourse,
}

impl TraceLayer {
    pub fn code(self) -> &'static str {
        match self {
            TraceLayer::L0Surface => "L0",
            TraceLayer::L1Semantic => "L1",
            TraceLayer::L2Discourse => "L2",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            TraceLayer::L0Surface => "表层",
            TraceLayer::L1Semantic => "语义帧",
            TraceLayer::L2Discourse => "话语记忆",
        }
    }
}

/// 决策状态：Ok 通过门禁；Stale 表示被后续更正判为失效（但保留可回放）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceStatus {
    Ok,
    Rejected,
    Deferred,
    Failed,
    Stale,
}

impl TraceStatus {
    pub fn label(self) -> &'static str {
        match self {
            TraceStatus::Ok => "ok",
            TraceStatus::Rejected => "rejected",
            TraceStatus::Deferred => "deferred",
            TraceStatus::Failed => "failed",
            TraceStatus::Stale => "stale",
        }
    }
}

/// 决策节点的稳定编号（跨轮全局唯一）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct NodeId(pub u64);

/// 节点载荷（受 [`TRACE_GRAPH_VERSION`] 约束；BTreeMap 保证确定性序列化）。
pub type NodePayload = BTreeMap<String, String>;

/// 决策的支撑来源：事实、证明、规则、模板、用户输入或上游节点。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "support", rename_all = "snake_case")]
pub enum Support {
    UserInput { excerpt: String },
    Fact {
        entity: String,
        predicate: String,
        value: String,
    },
    Proof { id: String },
    Rule { name: String },
    Template { name: String },
    Node { id: NodeId },
}

/// 被拒绝的候选及拒绝理由（“为什么不是另一个”的基础）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Alternative {
    pub label: String,
    pub reason: String,
}

impl Alternative {
    pub fn new(label: impl Into<String>, reason: impl Into<String>) -> Self {
        Alternative {
            label: label.into(),
            reason: reason.into(),
        }
    }
}

/// 一次不可再分的决策。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceNode {
    pub id: NodeId,
    pub turn: usize,
    pub layer: TraceLayer,
    pub kind: String,
    pub label: String,
    pub inputs: Vec<NodeId>,
    pub support: Vec<Support>,
    pub status: TraceStatus,
    pub alternatives: Vec<Alternative>,
    pub payload: NodePayload,
}

/// 问据（QUD）类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QudKind {
    /// 问某实体的某属性（谓词焦点所在）。
    AskAttribute,
    /// 问某实体的身份。
    AskIdentity,
    /// 是非问句。
    AskBinary,
}

impl QudKind {
    pub fn label(self) -> &'static str {
        match self {
            QudKind::AskAttribute => "问属性",
            QudKind::AskIdentity => "问身份",
            QudKind::AskBinary => "是非问",
        }
    }
}

/// 问据状态：Open 尚未被新问据取代；Superseded 被新问据取代（不再承继）；
/// Resolved 已获回答但仍留在轨迹中（谓词话题仍可承继）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QudStatus {
    Open,
    Resolved,
    Superseded,
}

/// 一条问据：本轮话语在问什么（实体槽 + 属性槽）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QudEntry {
    pub id: u64,
    pub turn: usize,
    pub kind: QudKind,
    pub entity: Option<String>,
    pub attribute: Option<String>,
    pub status: QudStatus,
    /// 实体或属性来自承继（省略/指代补全）而非显式提及。
    pub inherited: bool,
}

impl QudEntry {
    /// 人可读的话题（「实体/属性」或「实体/身份」）。
    pub fn topic(&self) -> String {
        let entity = self.entity.as_deref().unwrap_or("（未定实体）");
        match (&self.kind, self.attribute.as_deref()) {
            (QudKind::AskAttribute, Some(attribute)) => format!("{}/{}", entity, attribute),
            _ => format!("{}/身份", entity),
        }
    }
}

/// 问据栈：对话“正在问什么”的显式记忆，谓词焦点从中承继。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QudStack {
    #[serde(default = "default_qud_cap")]
    cap: usize,
    #[serde(default)]
    entries: VecDeque<QudEntry>,
    #[serde(default)]
    next_id: u64,
}

fn default_qud_cap() -> usize {
    DEFAULT_QUD_CAP
}

impl Default for QudStack {
    fn default() -> Self {
        QudStack {
            cap: DEFAULT_QUD_CAP,
            entries: VecDeque::new(),
            next_id: 1,
        }
    }
}

impl QudStack {
    /// 压入新的开放问据；此前的开放问据一律标记为被取代（不再承继）。
    fn push_open(
        &mut self,
        turn: usize,
        kind: QudKind,
        entity: Option<String>,
        attribute: Option<String>,
        inherited: bool,
    ) -> u64 {
        for entry in self.entries.iter_mut() {
            if entry.status == QudStatus::Open {
                entry.status = QudStatus::Superseded;
            }
        }
        let id = self.next_id;
        self.next_id += 1;
        self.entries.push_back(QudEntry {
            id,
            turn,
            kind,
            entity,
            attribute,
            status: QudStatus::Open,
            inherited,
        });
        while self.entries.len() > self.cap {
            self.entries.pop_front();
        }
        id
    }

    pub fn push_attribute(
        &mut self,
        turn: usize,
        entity: Option<String>,
        attribute: impl Into<String>,
        inherited: bool,
    ) -> u64 {
        self.push_open(turn, QudKind::AskAttribute, entity, Some(attribute.into()), inherited)
    }

    pub fn push_identity(&mut self, turn: usize, entity: Option<String>, inherited: bool) -> u64 {
        self.push_open(turn, QudKind::AskIdentity, entity, None, inherited)
    }

    /// 当前最上方开放问据已获回答：置为 Resolved（仍留在轨迹，可承继谓词话题）。
    pub fn resolve_open(&mut self) {
        if let Some(entry) = self.entries.iter_mut().rev().find(|e| e.status == QudStatus::Open) {
            entry.status = QudStatus::Resolved;
        }
    }

    /// 谓词焦点：最近一条未被取代且带属性的问据的属性。
    ///
    /// 这是「上一轮问身高，这一轮『叶莉呢』仍是问身高」的承继来源。
    pub fn predicate_focus(&self) -> Option<&str> {
        self.entries
            .iter()
            .rev()
            .find(|entry| entry.status != QudStatus::Superseded)
            .and_then(|entry| entry.attribute.as_deref())
    }

    /// 实体焦点：最近一条未被取代且定了实体的问据的实体。
    pub fn entity_focus(&self) -> Option<&str> {
        self.entries
            .iter()
            .rev()
            .find(|entry| entry.status != QudStatus::Superseded)
            .and_then(|entry| entry.entity.as_deref())
    }

    /// 与当前谓词话题同题的问据（用于判断是否已问过同一属性）。
    pub fn same_topic_open(&self, entity: &str, attribute: &str) -> bool {
        self.entries.iter().rev().any(|entry| {
            entry.status == QudStatus::Open
                && entry.entity.as_deref() == Some(entity)
                && entry.attribute.as_deref() == Some(attribute)
        })
    }

    pub fn entries(&self) -> impl Iterator<Item = &QudEntry> {
        self.entries.iter()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn report(&self) -> String {
        if self.entries.is_empty() {
            return "问据栈：（空）".to_string();
        }
        let mut lines = vec![format!("问据栈（{} 条，最近在下）：", self.entries.len())];
        for entry in self.entries.iter().rev() {
            let inherited = if entry.inherited { "（承继）" } else { "" };
            lines.push(format!(
                "  #{} 第 {} 轮 {} {}{} [{}]",
                entry.id,
                entry.turn,
                entry.kind.label(),
                entry.topic(),
                inherited,
                match entry.status {
                    QudStatus::Open => "开放",
                    QudStatus::Resolved => "已答",
                    QudStatus::Superseded => "被取代",
                }
            ));
        }
        lines.join("\n")
    }
}

/// 指代来源：显式提及、代词、省略补全、用户更正、插件代答。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReferentSource {
    Explicit,
    Pronoun,
    Ellipsis,
    Correction,
    Plugin,
}

impl ReferentSource {
    pub fn label(self) -> &'static str {
        match self {
            ReferentSource::Explicit => "显式提及",
            ReferentSource::Pronoun => "代词",
            ReferentSource::Ellipsis => "省略补全",
            ReferentSource::Correction => "用户更正",
            ReferentSource::Plugin => "插件代答",
        }
    }
}

/// 指代事件状态：Active 仍有效；Overridden 被后续绑定覆盖。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReferentStatus {
    Active,
    Overridden,
}

/// 一次指代决策（“这个说法指那个实体”）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReferentEvent {
    pub id: u64,
    pub turn: usize,
    pub surface: String,
    pub referent: String,
    pub source: ReferentSource,
    pub status: ReferentStatus,
    /// 本事件覆盖的旧事件 id（「不是他，是叶莉」的版本链）。
    pub overrides: Option<u64>,
    pub note: String,
}

/// 指代轨迹：按版本记录每次绑定的来源、覆盖关系与备注。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReferentTrail {
    #[serde(default = "default_trail_cap")]
    cap: usize,
    #[serde(default)]
    events: VecDeque<ReferentEvent>,
    #[serde(default)]
    next_id: u64,
}

fn default_trail_cap() -> usize {
    DEFAULT_TRAIL_CAP
}

impl Default for ReferentTrail {
    fn default() -> Self {
        ReferentTrail {
            cap: DEFAULT_TRAIL_CAP,
            events: VecDeque::new(),
            next_id: 1,
        }
    }
}

fn fold_surface(surface: &str) -> String {
    surface.trim().to_lowercase()
}

impl ReferentTrail {
    /// 记录一次绑定；同说法的旧 Active 事件被覆盖（保留版本链）。
    pub fn record(
        &mut self,
        turn: usize,
        surface: &str,
        referent: &str,
        source: ReferentSource,
        note: impl Into<String>,
    ) -> u64 {
        let folded = fold_surface(surface);
        let prior = self
            .events
            .iter_mut()
            .rev()
            .find(|event| event.status == ReferentStatus::Active && fold_surface(&event.surface) == folded)
            .map(|event| {
                event.status = ReferentStatus::Overridden;
                event.id
            });
        let id = self.next_id;
        self.next_id += 1;
        self.events.push_back(ReferentEvent {
            id,
            turn,
            surface: surface.to_string(),
            referent: referent.to_string(),
            source,
            status: ReferentStatus::Active,
            overrides: prior,
            note: note.into(),
        });
        while self.events.len() > self.cap {
            self.events.pop_front();
        }
        id
    }

    /// 当前对某说法的有效指代（最近一条 Active）。
    pub fn resolve(&self, surface: &str) -> Option<&ReferentEvent> {
        let folded = fold_surface(surface);
        self.events
            .iter()
            .rev()
            .find(|event| event.status == ReferentStatus::Active && fold_surface(&event.surface) == folded)
    }

    /// 某说法的完整绑定史（新在前）。
    pub fn history(&self, surface: &str) -> Vec<&ReferentEvent> {
        let folded = fold_surface(surface);
        self.events
            .iter()
            .rev()
            .filter(|event| fold_surface(&event.surface) == folded)
            .collect()
    }

    pub fn events(&self) -> impl Iterator<Item = &ReferentEvent> {
        self.events.iter()
    }

    pub fn active_count(&self) -> usize {
        self.events
            .iter()
            .filter(|event| event.status == ReferentStatus::Active)
            .count()
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    pub fn report(&self) -> String {
        if self.events.is_empty() {
            return "指代轨迹：（空）".to_string();
        }
        let mut lines = vec![format!(
            "指代轨迹（{} 条事件，{} 条有效，最近在下）：",
            self.events.len(),
            self.active_count()
        )];
        for event in self.events.iter().rev() {
            let chain = match event.overrides {
                Some(id) => format!("，覆盖 #{}", id),
                None => String::new(),
            };
            let status = match event.status {
                ReferentStatus::Active => "有效",
                ReferentStatus::Overridden => "已被覆盖",
            };
            let note = if event.note.is_empty() {
                String::new()
            } else {
                format!("（{}）", event.note)
            };
            lines.push(format!(
                "  #{} 第 {} 轮 「{}」→「{}」（{}，{}{}）{}",
                event.id, event.turn, event.surface, event.referent, event.source.label(), status, chain, note
            ));
        }
        lines.join("\n")
    }
}

/// 会话内值更正：On 模式下用户改口后的覆盖值（先于知识图谱取值）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValueCorrection {
    pub turn: usize,
    pub entity: String,
    pub attribute: String,
    pub value: String,
    /// 被更正的旧值（展示用；无法取得时为空串）。
    pub previous: String,
}

/// 一轮的完整轨迹（编年史归档与重启回放的单位）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnTrace {
    pub turn: usize,
    #[serde(default)]
    pub input_excerpt: String,
    #[serde(default)]
    pub nodes: Vec<TraceNode>,
    #[serde(default)]
    pub qud: Vec<QudEntry>,
    #[serde(default)]
    pub referents: Vec<ReferentEvent>,
    #[serde(default)]
    pub corrections: Vec<ValueCorrection>,
    #[serde(default)]
    pub reply_excerpt: Option<String>,
    /// 本轮是否含被后续更正判为失效的回答。
    #[serde(default)]
    pub stale: bool,
}

/// 有界摘要：不复制原文，只给稳定的计数分布（台账/评测用）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceReceipt {
    pub version: u32,
    pub turns: usize,
    pub nodes: usize,
    pub stale_turns: usize,
    pub qud_open: usize,
    pub referents_active: usize,
    pub corrections: usize,
    pub status_counts: BTreeMap<String, usize>,
    pub layer_counts: BTreeMap<String, usize>,
    pub kind_counts: BTreeMap<String, usize>,
}

/// 分层追踪存储：工作记忆（本轮 L0–L2）× 情节记忆（TurnTrace 序列）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceStore {
    #[serde(default = "trace_graph_version")]
    pub version: u32,
    #[serde(default)]
    turn: usize,
    #[serde(default)]
    next_node_id: u64,
    #[serde(default = "default_max_turns")]
    max_turns: usize,
    #[serde(default = "default_max_nodes_per_turn")]
    max_nodes_per_turn: usize,
    #[serde(default = "default_max_events_per_turn")]
    max_events_per_turn: usize,
    #[serde(default)]
    turns: VecDeque<TurnTrace>,
    #[serde(default)]
    qud: QudStack,
    #[serde(default)]
    trail: ReferentTrail,
    /// 会话内更正覆盖，键为 `entity\u{1}attribute`（BTreeMap：确定性序列化）。
    #[serde(default)]
    overrides: BTreeMap<String, String>,
}

fn trace_graph_version() -> u32 {
    TRACE_GRAPH_VERSION
}
fn default_max_turns() -> usize {
    DEFAULT_MAX_TURNS
}
fn default_max_nodes_per_turn() -> usize {
    DEFAULT_MAX_NODES_PER_TURN
}
fn default_max_events_per_turn() -> usize {
    DEFAULT_MAX_EVENTS_PER_TURN
}

fn override_key(entity: &str, attribute: &str) -> String {
    format!("{}\u{1}{}", entity, attribute)
}

impl Default for TraceStore {
    fn default() -> Self {
        Self::new()
    }
}

impl TraceStore {
    pub fn new() -> Self {
        TraceStore {
            version: TRACE_GRAPH_VERSION,
            turn: 0,
            next_node_id: 1,
            max_turns: DEFAULT_MAX_TURNS,
            max_nodes_per_turn: DEFAULT_MAX_NODES_PER_TURN,
            max_events_per_turn: DEFAULT_MAX_EVENTS_PER_TURN,
            turns: VecDeque::new(),
            qud: QudStack::default(),
            trail: ReferentTrail::default(),
            overrides: BTreeMap::new(),
        }
    }

    pub fn with_limits(max_turns: usize, max_nodes_per_turn: usize, max_events_per_turn: usize) -> Self {
        TraceStore {
            max_turns: max_turns.max(1),
            max_nodes_per_turn: max_nodes_per_turn.max(4),
            max_events_per_turn: max_events_per_turn.max(2),
            ..Self::new()
        }
    }

    pub fn is_empty(&self) -> bool {
        self.turns.is_empty()
    }

    pub fn turn(&self) -> usize {
        self.turn
    }

    pub fn turn_count(&self) -> usize {
        self.turns.len()
    }

    pub fn node_count(&self) -> usize {
        self.turns.iter().map(|turn| turn.nodes.len()).sum()
    }

    pub fn qud_stack(&self) -> &QudStack {
        &self.qud
    }

    pub fn referent_trail(&self) -> &ReferentTrail {
        &self.trail
    }

    /// 指代轨迹的可读报告。
    pub fn referent_report(&self) -> String {
        self.trail.report()
    }

    pub fn predicate_focus(&self) -> Option<&str> {
        self.qud.predicate_focus()
    }

    pub fn entity_focus(&self) -> Option<&str> {
        self.qud.entity_focus()
    }

    /// 当前最上方开放问据已获回答：置为 Resolved。
    pub fn resolve_open_qud(&mut self) {
        self.qud.resolve_open();
    }

    /// 会话内更正覆盖值（On 模式下先于知识图谱取值）。
    pub fn value_override(&self, entity: &str, attribute: &str) -> Option<&str> {
        self.overrides.get(&override_key(entity, attribute)).map(String::as_str)
    }

    pub fn corrections(&self) -> impl Iterator<Item = &ValueCorrection> {
        self.turns.iter().flat_map(|turn| turn.corrections.iter())
    }

    /// 打开一轮轨迹（幂等：同轮重复调用复用同一缓冲）。
    pub fn begin_turn(&mut self, turn: usize) {
        self.turn = turn;
        let fresh = self.turns.back().map(|trace| trace.turn != turn).unwrap_or(true);
        if fresh {
            self.turns.push_back(TurnTrace {
                turn,
                input_excerpt: String::new(),
                nodes: Vec::new(),
                qud: Vec::new(),
                referents: Vec::new(),
                corrections: Vec::new(),
                reply_excerpt: None,
                stale: false,
            });
            while self.turns.len() > self.max_turns {
                self.turns.pop_front();
            }
        }
    }

    fn current_turn_mut(&mut self) -> Option<&mut TurnTrace> {
        let turn = self.turn;
        if self.turns.back().map(|trace| trace.turn).unwrap_or(0) != turn {
            self.turns.push_back(TurnTrace {
                turn,
                input_excerpt: String::new(),
                nodes: Vec::new(),
                qud: Vec::new(),
                referents: Vec::new(),
                corrections: Vec::new(),
                reply_excerpt: None,
                stale: false,
            });
        }
        self.turns.back_mut()
    }

    /// 记录一次决策；超过单轮上限时折叠为摘要节点（保留层身份）。
    pub fn record_node(
        &mut self,
        layer: TraceLayer,
        kind: &str,
        label: impl Into<String>,
        inputs: Vec<NodeId>,
        support: Vec<Support>,
        status: TraceStatus,
        payload: NodePayload,
    ) -> NodeId {
        let id = NodeId(self.next_node_id);
        self.next_node_id += 1;
        let turn = self.turn;
        let max_nodes = self.max_nodes_per_turn;
        // 确保当前轮缓冲存在（索引访问拆分借用，避免与计数器更新冲突）
        let needs_fresh = self.turns.back().map(|trace| trace.turn != turn).unwrap_or(true);
        if needs_fresh {
            self.turns.push_back(TurnTrace {
                turn,
                input_excerpt: String::new(),
                nodes: Vec::new(),
                qud: Vec::new(),
                referents: Vec::new(),
                corrections: Vec::new(),
                reply_excerpt: None,
                stale: false,
            });
            while self.turns.len() > self.max_turns {
                self.turns.pop_front();
            }
        }
        let index = self.turns.len() - 1;
        let node_count = self.turns[index].nodes.len();
        if node_count >= max_nodes {
            let folded = self.turns[index]
                .nodes
                .iter()
                .filter(|node| node.kind == "fold")
                .count();
            if folded == 0 {
                let summary_id = NodeId(self.next_node_id);
                self.next_node_id += 1;
                self.turns[index].nodes.push(TraceNode {
                    id: summary_id,
                    turn,
                    layer,
                    kind: "fold".to_string(),
                    label: format!("超出单轮节点上限 {}，后续决策折叠", max_nodes),
                    inputs: Vec::new(),
                    support: Vec::new(),
                    status: TraceStatus::Ok,
                    alternatives: Vec::new(),
                    payload: NodePayload::new(),
                });
            }
            return id;
        }
        self.turns[index].nodes.push(TraceNode {
            id,
            turn,
            layer,
            kind: kind.to_string(),
            label: label.into(),
            inputs,
            support,
            status,
            alternatives: Vec::new(),
            payload,
        });
        id
    }

    /// 给既有节点补充被拒绝的候选。
    pub fn add_alternative(&mut self, id: NodeId, alternative: Alternative) {
        for turn in self.turns.iter_mut() {
            if let Some(node) = turn.nodes.iter_mut().find(|node| node.id == id) {
                node.alternatives.push(alternative);
                return;
            }
        }
    }

    /// 标记某轮的回复节点为失效（更正传播：只失效，不删除）。
    pub fn mark_turn_stale(&mut self, turn_index: usize) -> usize {
        let mut marked = 0;
        for trace in self.turns.iter_mut() {
            if trace.turn == turn_index {
                for node in trace.nodes.iter_mut() {
                    if node.kind == "reply" && node.status == TraceStatus::Ok {
                        node.status = TraceStatus::Stale;
                        marked += 1;
                    }
                }
                if marked > 0 {
                    trace.stale = true;
                }
            }
        }
        marked
    }

    /// L0：记录输入摘录。
    pub fn record_surface(&mut self, turn: usize, input: &str) -> NodeId {
        self.begin_turn(turn);
        let excerpt: String = input.trim().chars().take(80).collect();
        if let Some(current) = self.current_turn_mut() {
            if current.input_excerpt.is_empty() {
                current.input_excerpt = excerpt.clone();
            }
        }
        let mut payload = NodePayload::new();
        payload.insert("input".to_string(), excerpt.clone());
        self.record_node(
            TraceLayer::L0Surface,
            "surface",
            format!("输入「{}」", excerpt),
            Vec::new(),
            vec![Support::UserInput { excerpt }],
            TraceStatus::Ok,
            payload,
        )
    }

    /// L1：记录本轮解读（话语行为 + 槽位）。
    pub fn record_interpret(
        &mut self,
        turn: usize,
        surface: NodeId,
        kind: &str,
        entity: Option<&str>,
        attribute: Option<&str>,
        question: Option<&str>,
        inherited: bool,
    ) -> NodeId {
        self.begin_turn(turn);
        let label = match (entity, attribute) {
            (Some(entity), Some(attribute)) => format!("解读：{} {} / {}", kind, entity, attribute),
            (Some(entity), None) => format!("解读：{} {}", kind, entity),
            _ => format!("解读：{}", kind),
        };
        let mut payload = NodePayload::new();
        payload.insert("kind".to_string(), kind.to_string());
        if let Some(entity) = entity {
            payload.insert("entity".to_string(), entity.to_string());
        }
        if let Some(attribute) = attribute {
            payload.insert("attribute".to_string(), attribute.to_string());
        }
        if let Some(question) = question {
            payload.insert("question".to_string(), question.to_string());
        }
        if inherited {
            payload.insert("inherited".to_string(), "true".to_string());
        }
        self.record_node(
            TraceLayer::L1Semantic,
            "interpret",
            label,
            vec![surface],
            Vec::new(),
            TraceStatus::Ok,
            payload,
        )
    }

    /// L2：谓词/实体承继决策（省略补全为何补成这个槽位）。
    pub fn record_inheritance(
        &mut self,
        turn: usize,
        interpret: NodeId,
        attribute: &str,
        origin: &str,
    ) -> NodeId {
        self.begin_turn(turn);
        let mut payload = NodePayload::new();
        payload.insert("attribute".to_string(), attribute.to_string());
        payload.insert("origin".to_string(), origin.to_string());
        self.record_node(
            TraceLayer::L2Discourse,
            "inherit_predicate",
            format!("承继谓词焦点「{}」（来自{}）", attribute, origin),
            vec![interpret],
            Vec::new(),
            TraceStatus::Ok,
            payload,
        )
    }

    /// L2：指代决策 + 轨迹登记。
    pub fn record_mention(
        &mut self,
        turn: usize,
        interpret: NodeId,
        surface: &str,
        referent: &str,
        source: ReferentSource,
        note: &str,
    ) -> NodeId {
        self.begin_turn(turn);
        let truncated: String = note.chars().take(60).collect();
        let event_id = self.trail.record(turn, surface, referent, source, truncated);
        let event = self
            .trail
            .events()
            .find(|event| event.id == event_id)
            .cloned();
        if let Some(event) = event {
            let max_events = self.max_events_per_turn;
            if let Some(current) = self.current_turn_mut() {
                current.referents.push(event);
                while current.referents.len() > max_events {
                    current.referents.remove(0);
                }
            }
        }
        let mut payload = NodePayload::new();
        payload.insert("surface".to_string(), surface.to_string());
        payload.insert("referent".to_string(), referent.to_string());
        payload.insert("source".to_string(), source.label().to_string());
        payload.insert("event".to_string(), event_id.to_string());
        self.record_node(
            TraceLayer::L2Discourse,
            "resolve_mention",
            format!("「{}」→「{}」（{}）", surface, referent, source.label()),
            vec![interpret],
            Vec::new(),
            TraceStatus::Ok,
            payload,
        )
    }

    /// L2：压入问据并记录节点（旧开放问据在栈内被取代）。
    pub fn record_qud(
        &mut self,
        turn: usize,
        interpret: NodeId,
        kind: QudKind,
        entity: Option<&str>,
        attribute: Option<&str>,
        inherited: bool,
    ) -> NodeId {
        self.begin_turn(turn);
        let qud_id = match (kind, attribute) {
            (QudKind::AskAttribute, Some(attribute)) => self.qud.push_attribute(
                turn,
                entity.map(str::to_string),
                attribute,
                inherited,
            ),
            _ => self.qud.push_identity(turn, entity.map(str::to_string), inherited),
        };
        let entry = self.qud.entries().find(|entry| entry.id == qud_id).cloned();
        if let (Some(entry), Some(current)) = (entry, self.current_turn_mut()) {
            current.qud.push(entry);
        }
        let label = match (entity, attribute) {
            (Some(entity), Some(attribute)) => format!("问据：{} / {}", entity, attribute),
            (Some(entity), None) => format!("问据：{} / 身份", entity),
            _ => "问据：（未定实体）".to_string(),
        };
        let mut payload = NodePayload::new();
        payload.insert("qud".to_string(), qud_id.to_string());
        payload.insert("kind".to_string(), kind.label().to_string());
        if let Some(entity) = entity {
            payload.insert("entity".to_string(), entity.to_string());
        }
        if let Some(attribute) = attribute {
            payload.insert("attribute".to_string(), attribute.to_string());
        }
        self.record_node(
            TraceLayer::L2Discourse,
            "qud_push",
            label,
            vec![interpret],
            Vec::new(),
            TraceStatus::Ok,
            payload,
        )
    }

    /// L2：落地回答（每个回答都应能沿 inputs 回溯到本轮决策）。
    pub fn record_reply(
        &mut self,
        turn: usize,
        entity: Option<&str>,
        attribute: Option<&str>,
        inputs: Vec<NodeId>,
        provenance: &[String],
        answer_excerpt: Option<&str>,
    ) -> NodeId {
        self.begin_turn(turn);
        let excerpt: String = answer_excerpt.unwrap_or("").chars().take(80).collect();
        if let Some(current) = self.current_turn_mut() {
            if current.reply_excerpt.is_none() && !excerpt.is_empty() {
                current.reply_excerpt = Some(excerpt.clone());
            }
        }
        let support = provenance
            .iter()
            .map(|tag| Support::Template { name: tag.clone() })
            .collect();
        let label = format!(
            "回复：{}",
            if excerpt.is_empty() { "（无文本）" } else { &excerpt }
        );
        let mut payload = NodePayload::new();
        if let Some(entity) = entity {
            payload.insert("entity".to_string(), entity.to_string());
        }
        if let Some(attribute) = attribute {
            payload.insert("attribute".to_string(), attribute.to_string());
        }
        if !excerpt.is_empty() {
            payload.insert("answer".to_string(), excerpt);
        }
        payload.insert(
            "provenance".to_string(),
            provenance.join("、"),
        );
        self.record_node(
            TraceLayer::L2Discourse,
            "reply",
            label,
            inputs,
            support,
            TraceStatus::Ok,
            payload,
        )
    }

    /// 记录一次让位（知识管线未接管本轮输入）。
    pub fn record_defer(&mut self, turn: usize, interpret: NodeId, reason: &str) -> NodeId {
        self.begin_turn(turn);
        let mut payload = NodePayload::new();
        payload.insert("reason".to_string(), reason.to_string());
        self.record_node(
            TraceLayer::L2Discourse,
            "defer",
            format!("让位：{}", reason),
            vec![interpret],
            Vec::new(),
            TraceStatus::Deferred,
            payload,
        )
    }

    /// 用户改口：登记覆盖值、把受影响轮（`target_turn`）的回复判为 Stale、记 L2 更正节点。
    ///
    /// 返回（更正记录，被标记失效的回复节点数）。
    pub fn record_correction(
        &mut self,
        turn: usize,
        target_turn: usize,
        entity: &str,
        attribute: &str,
        value: &str,
        previous: &str,
    ) -> (ValueCorrection, usize) {
        self.begin_turn(turn);
        self.overrides.insert(override_key(entity, attribute), value.to_string());
        let correction = ValueCorrection {
            turn,
            entity: entity.to_string(),
            attribute: attribute.to_string(),
            value: value.to_string(),
            previous: previous.to_string(),
        };
        let marked = self.mark_turn_stale(target_turn);
        if let Some(current) = self.current_turn_mut() {
            current.corrections.push(correction.clone());
        }
        let mut payload = NodePayload::new();
        payload.insert("entity".to_string(), entity.to_string());
        payload.insert("attribute".to_string(), attribute.to_string());
        payload.insert("value".to_string(), value.to_string());
        payload.insert("previous".to_string(), previous.to_string());
        payload.insert("stale_replies".to_string(), marked.to_string());
        self.record_node(
            TraceLayer::L2Discourse,
            "correction",
            format!(
                "更正：{} / {} 由「{}」改为「{}」",
                entity, attribute, previous, value
            ),
            Vec::new(),
            vec![Support::UserInput { excerpt: format!("{}→{}", previous, value) }],
            TraceStatus::Ok,
            payload,
        );
        (correction, marked)
    }

    /// 找到最近一轮带回复节点且问同一实体属性的轮号（更正定位用）。
    pub fn last_answered_turn_for(&self, entity: &str, attribute: &str) -> Option<usize> {
        self.turns.iter().rev().find(|trace| {
            trace
                .nodes
                .iter()
                .any(|node| node.kind == "reply"
                    && node.status == TraceStatus::Ok
                    && node.payload.get("entity").map(String::as_str) == Some(entity)
                    && node.payload.get("attribute").map(String::as_str) == Some(attribute))
        }).map(|trace| trace.turn)
    }

    pub fn turns(&self) -> impl Iterator<Item = &TurnTrace> {
        self.turns.iter()
    }

    pub fn last_turn(&self) -> Option<&TurnTrace> {
        self.turns.back()
    }

    /// 最近一轮的完整轨迹 JSON（编年史归档用）。
    pub fn last_turn_json(&self) -> Option<serde_json::Value> {
        self.turns
            .back()
            .and_then(|trace| serde_json::to_value(trace).ok())
    }

    /// 从编年史回放一轮轨迹：恢复节点、问据、指代与更正覆盖（不重放决策）。
    pub fn push_turn_trace(&mut self, trace: TurnTrace) {
        self.next_node_id = self
            .next_node_id
            .max(trace.nodes.iter().map(|node| node.id.0 + 1).max().unwrap_or(1));
        self.qud.next_id = self
            .qud
            .next_id
            .max(trace.qud.iter().map(|entry| entry.id + 1).max().unwrap_or(1));
        self.trail.next_id = self.trail.next_id.max(
            trace
                .referents
                .iter()
                .map(|event| event.id + 1)
                .max()
                .unwrap_or(1),
        );
        for entry in &trace.qud {
            if let Some(existing) = self.qud.entries.iter_mut().find(|e| e.id == entry.id) {
                *existing = entry.clone();
            } else {
                self.qud.entries.push_back(entry.clone());
                while self.qud.entries.len() > self.qud.cap {
                    self.qud.entries.pop_front();
                }
            }
        }
        for event in &trace.referents {
            if let Some(existing) = self.trail.events.iter_mut().find(|e| e.id == event.id) {
                *existing = event.clone();
            } else {
                self.trail.events.push_back(event.clone());
                while self.trail.events.len() > self.trail.cap {
                    self.trail.events.pop_front();
                }
            }
        }
        for correction in &trace.corrections {
            self.overrides.insert(
                override_key(&correction.entity, &correction.attribute),
                correction.value.clone(),
            );
        }
        self.turn = self.turn.max(trace.turn);
        self.turns.push_back(trace);
        while self.turns.len() > self.max_turns {
            self.turns.pop_front();
        }
    }

    /// 沿 inputs 边回溯：从某节点到根的决策链（新在前）。
    pub fn why(&self, id: NodeId) -> Vec<&TraceNode> {
        let mut chain = Vec::new();
        let mut seen = std::collections::HashSet::new();
        let mut frontier = vec![id];
        while !frontier.is_empty() {
            let mut next = Vec::new();
            for node_id in frontier {
                if !seen.insert(node_id) {
                    continue;
                }
                let Some(node) = self.find_node(node_id) else { continue };
                chain.push(node);
                next.extend(node.inputs.iter().copied());
            }
            frontier = next;
        }
        chain
    }

    pub fn find_node(&self, id: NodeId) -> Option<&TraceNode> {
        self.turns
            .iter()
            .rev()
            .find_map(|turn| turn.nodes.iter().find(|node| node.id == id))
    }

    /// 「为什么」查询的可读回答：决策链 + 状态。
    pub fn render_why(&self, id: NodeId) -> Option<String> {
        let chain = self.why(id);
        if chain.is_empty() {
            return None;
        }
        let mut lines = vec![format!("依据链（节点 #{}）：", id.0)];
        for node in chain {
            let inputs = node
                .inputs
                .iter()
                .map(|input| format!("#{}", input.0))
                .collect::<Vec<_>>()
                .join(", ");
            let inputs = if inputs.is_empty() {
                "（根）".to_string()
            } else {
                format!("← {}", inputs)
            };
            lines.push(format!(
                "  [{}|{}] #{} {} [{}] {}",
                node.layer.code(),
                node.kind,
                node.id.0,
                node.label,
                node.status.label(),
                inputs
            ));
        }
        Some(lines.join("\n"))
    }

    /// 有界摘要（无原文；状态/层/种类分布稳定有序）。
    pub fn receipt(&self) -> TraceReceipt {
        let mut status_counts = BTreeMap::new();
        let mut layer_counts = BTreeMap::new();
        let mut kind_counts = BTreeMap::new();
        let mut corrections = 0;
        let mut stale_turns = 0;
        for turn in &self.turns {
            corrections += turn.corrections.len();
            if turn.stale {
                stale_turns += 1;
            }
            for node in &turn.nodes {
                *status_counts.entry(node.status.label().to_string()).or_insert(0) += 1;
                *layer_counts.entry(node.layer.code().to_string()).or_insert(0) += 1;
                *kind_counts.entry(node.kind.clone()).or_insert(0) += 1;
            }
        }
        TraceReceipt {
            version: self.version,
            turns: self.turns.len(),
            nodes: self.node_count(),
            stale_turns,
            qud_open: self.qud.entries.iter().filter(|e| e.status == QudStatus::Open).count(),
            referents_active: self.trail.active_count(),
            corrections,
            status_counts,
            layer_counts,
            kind_counts,
        }
    }

    /// 可读的追踪报告：模式、问据栈、指代轨迹与逐轮决策摘要。
    pub fn render_report(&self, mode: TraceMode) -> String {
        let receipt = self.receipt();
        let mut lines = vec![format!(
            "分层追踪图 v{}（模式 {}）：{} 轮 / {} 节点 / 更正 {} / 失效轮 {}",
            self.version,
            mode.label(),
            receipt.turns,
            receipt.nodes,
            receipt.corrections,
            receipt.stale_turns
        )];
        lines.push(self.qud.report());
        lines.push(self.trail.report());
        for turn in &self.turns {
            let stale = if turn.stale { "（含失效回答）" } else { "" };
            lines.push(format!(
                "第 {} 轮{}：「{}」{}",
                turn.turn,
                stale,
                turn.input_excerpt,
                turn.reply_excerpt
                    .as_deref()
                    .map(|answer| format!("→「{}」", answer))
                    .unwrap_or_default()
            ));
            for node in &turn.nodes {
                lines.push(format!(
                    "  [{}|{}] #{} {} [{}]",
                    node.layer.code(),
                    node.kind,
                    node.id.0,
                    node.label,
                    node.status.label()
                ));
            }
        }
        lines.join("\n")
    }
}

/// 追踪图配置（`config.toml` 的 `[trace_graph]` 节；缺省整节等价于 Off）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceGraphConfig {
    /// 三档模式："off" | "observe" | "on"（非法值按 off 处理）。
    #[serde(default)]
    pub mode: String,
    /// 会话内保留的轮迹上限。
    #[serde(default = "default_max_turns")]
    pub max_turns: usize,
    /// 单轮决策节点上限（超出折叠为摘要节点）。
    #[serde(default = "default_max_nodes_per_turn")]
    pub max_nodes_per_turn: usize,
    /// 单轮指代事件上限。
    #[serde(default = "default_max_events_per_turn")]
    pub max_events_per_turn: usize,
}

impl Default for TraceGraphConfig {
    fn default() -> Self {
        TraceGraphConfig {
            mode: TraceMode::Off.label().to_string(),
            max_turns: DEFAULT_MAX_TURNS,
            max_nodes_per_turn: DEFAULT_MAX_NODES_PER_TURN,
            max_events_per_turn: DEFAULT_MAX_EVENTS_PER_TURN,
        }
    }
}

impl TraceGraphConfig {
    /// 解析模式（非法值回落 Off，保持行为兼容）。
    pub fn parse_mode(&self) -> TraceMode {
        TraceMode::parse(&self.mode).unwrap_or(TraceMode::Off)
    }

    /// 构造带本配置上限的追踪存储。
    pub fn build_store(&self) -> TraceStore {
        TraceStore::with_limits(self.max_turns, self.max_nodes_per_turn, self.max_events_per_turn)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn payload(pairs: &[(&str, &str)]) -> NodePayload {
        pairs
            .iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect()
    }

    #[test]
    fn trace_mode_parses_and_labels() {
        assert_eq!(TraceMode::parse("off"), Some(TraceMode::Off));
        assert_eq!(TraceMode::parse("OBSERVE"), Some(TraceMode::Observe));
        assert_eq!(TraceMode::parse("on"), Some(TraceMode::On));
        assert_eq!(TraceMode::parse("bogus"), None);
        assert!(TraceMode::Off.records() == false);
        assert!(TraceMode::Observe.records());
        assert!(TraceMode::On.is_on());
    }

    #[test]
    fn qud_predicate_focus_survives_resolution_and_supersede() {
        let mut stack = QudStack::default();
        let first = stack.push_attribute(1, Some("姚明".to_string()), "身高", false);
        assert_eq!(stack.predicate_focus(), Some("身高"));
        stack.resolve_open();
        // 已回答的问据仍承继谓词话题
        assert_eq!(stack.predicate_focus(), Some("身高"));
        // 新属性问据取代旧话题
        stack.push_attribute(2, Some("姚明".to_string()), "国籍", false);
        assert_eq!(stack.predicate_focus(), Some("国籍"));
        // 身份问据取代属性问据：不再承继身高/国籍
        stack.push_identity(3, Some("叶莉".to_string()), false);
        assert_eq!(stack.predicate_focus(), None);
        assert_eq!(stack.entity_focus(), Some("叶莉"));
        assert!(stack.entries().any(|entry| entry.id == first));
    }

    #[test]
    fn trail_overrides_old_binding_and_keeps_version_chain() {
        let mut trail = ReferentTrail::default();
        let first = trail.record(1, "他", "姚明", ReferentSource::Pronoun, "焦点实体");
        let second = trail.record(2, "他", "叶莉", ReferentSource::Correction, "用户改口");
        let resolved = trail.resolve("他").expect("应有有效绑定");
        assert_eq!(resolved.id, second);
        assert_eq!(resolved.referent, "叶莉");
        let history = trail.history("他");
        assert_eq!(history.len(), 2);
        // history 新在前：最近的绑定在 [0]，并覆盖了更早的绑定
        assert_eq!(history[0].id, second);
        assert_eq!(history[0].overrides, Some(first));
        assert_eq!(history[0].status, ReferentStatus::Active);
        assert_eq!(history[1].id, first);
        assert_eq!(history[1].status, ReferentStatus::Overridden);
    }

    #[test]
    fn why_chain_orders_inputs_from_reply_to_surface() {
        let mut store = TraceStore::new();
        let surface = store.record_surface(1, "姚明的身高是多少");
        let interpret =
            store.record_interpret(1, surface, "ask_attribute", Some("姚明"), Some("身高"), Some("姚明的身高是什么"), false);
        let qud = store.record_qud(1, interpret, QudKind::AskAttribute, Some("姚明"), Some("身高"), false);
        let reply = store.record_reply(1, Some("姚明"), Some("身高"), vec![interpret, qud], &["知识图谱中的直接事实".to_string()], Some("姚明的身高是 226 厘米。"));
        let chain = store.why(reply);
        let ids: Vec<NodeId> = chain.iter().map(|node| node.id).collect();
        // BFS 依 inputs 声明序展开：reply ← interpret/qud ← surface
        assert_eq!(ids, vec![reply, interpret, qud, surface]);
        let rendered = store.render_why(reply).expect("应能渲染依据链");
        assert!(rendered.contains("qud_push"), "{}", rendered);
        assert!(rendered.contains("surface"), "{}", rendered);
    }

    #[test]
    fn correction_marks_stale_and_overrides_value() {
        let mut store = TraceStore::new();
        let surface = store.record_surface(1, "姚明的身高是多少");
        let interpret = store.record_interpret(1, surface, "ask_attribute", Some("姚明"), Some("身高"), None, false);
        store.record_reply(1, Some("姚明"), Some("身高"), vec![interpret], &[], Some("226"));
        assert_eq!(store.value_override("姚明", "身高"), None);
        let (correction, marked) = store.record_correction(2, 1, "姚明", "身高", "229", "226");
        assert_eq!(correction.value, "229");
        assert_eq!(marked, 1, "上一轮的回复节点应被标记失效");
        assert_eq!(store.value_override("姚明", "身高"), Some("229"));
        let stale = store
            .turns()
            .find(|turn| turn.turn == 1)
            .expect("第 1 轮应仍在")
            .nodes
            .iter()
            .find(|node| node.kind == "reply")
            .expect("应有回复节点");
        assert_eq!(stale.status, TraceStatus::Stale);
    }

    #[test]
    fn node_budget_folds_overflows_into_summary() {
        let mut store = TraceStore::with_limits(4, 4, 4);
        for index in 0..8 {
            store.record_surface(1, &format!("输入 {}", index));
        }
        let turn = store.last_turn().expect("应有轮迹");
        assert_eq!(turn.nodes.len(), 5, "4 个决策节点 + 1 个折叠摘要");
        assert_eq!(turn.nodes.iter().filter(|node| node.kind == "fold").count(), 1);
    }

    #[test]
    fn turn_budget_is_bounded() {
        let mut store = TraceStore::with_limits(3, 8, 4);
        for turn in 1..=6 {
            store.record_surface(turn, "输入");
        }
        assert_eq!(store.turn_count(), 3);
        assert_eq!(store.turns().map(|turn| turn.turn).collect::<Vec<_>>(), vec![4, 5, 6]);
    }

    #[test]
    fn snapshot_roundtrip_preserves_qud_trail_and_overrides() {
        let mut store = TraceStore::new();
        let surface = store.record_surface(1, "姚明的身高是多少");
        let interpret = store.record_interpret(1, surface, "ask_attribute", Some("姚明"), Some("身高"), None, false);
        store.record_qud(1, interpret, QudKind::AskAttribute, Some("姚明"), Some("身高"), false);
        store.record_mention(1, interpret, "他", "姚明", ReferentSource::Pronoun, "焦点");
        store.record_correction(2, 1, "姚明", "身高", "229", "226");
        let json = serde_json::to_string(&store).expect("应可序列化");
        let restored: TraceStore = serde_json::from_str(&json).expect("应可反序列化");
        assert_eq!(restored, store);
        assert_eq!(restored.predicate_focus(), Some("身高"));
        assert_eq!(restored.referent_trail().resolve("他").map(|e| e.referent.as_str()), Some("姚明"));
        assert_eq!(restored.value_override("姚明", "身高"), Some("229"));
    }

    #[test]
    fn push_turn_trace_replays_state_for_restart() {
        let mut source = TraceStore::new();
        let surface = source.record_surface(1, "姚明的身高是多少");
        let interpret = source.record_interpret(1, surface, "ask_attribute", Some("姚明"), Some("身高"), None, false);
        source.record_qud(1, interpret, QudKind::AskAttribute, Some("姚明"), Some("身高"), false);
        source.record_mention(1, interpret, "他", "姚明", ReferentSource::Pronoun, "焦点");
        source.record_correction(2, 1, "姚明", "身高", "229", "226");
        let archived: Vec<TurnTrace> = source.turns().cloned().collect();

        let mut restored = TraceStore::new();
        for trace in archived {
            restored.push_turn_trace(trace);
        }
        assert_eq!(restored.predicate_focus(), Some("身高"));
        assert_eq!(restored.value_override("姚明", "身高"), Some("229"));
        assert_eq!(restored.referent_trail().resolve("他").map(|e| e.referent.as_str()), Some("姚明"));
        assert_eq!(restored.node_count(), source.node_count());
    }

    #[test]
    fn receipt_is_stable_and_bounded() {
        let mut store = TraceStore::new();
        let surface = store.record_surface(1, "姚明的身高是多少");
        let interpret =
            store.record_interpret(1, surface, "ask_attribute", Some("姚明"), Some("身高"), None, false);
        store.record_qud(1, interpret, QudKind::AskAttribute, Some("姚明"), Some("身高"), false);
        let receipt = store.receipt();
        assert_eq!(receipt.turns, 1);
        assert_eq!(receipt.nodes, 3);
        assert_eq!(receipt.qud_open, 1);
        assert_eq!(receipt.layer_counts.get("L2").copied(), Some(1));
        let json = serde_json::to_string(&receipt).expect("回执应可序列化");
        let reparsed: TraceReceipt = serde_json::from_str(&json).unwrap();
        assert_eq!(reparsed, receipt);
    }

    #[test]
    fn report_mentions_layers_and_modes() {
        let mut store = TraceStore::new();
        let surface = store.record_surface(1, "姚明的身高是多少");
        let interpret =
            store.record_interpret(1, surface, "ask_attribute", Some("姚明"), Some("身高"), None, false);
        store.record_qud(1, interpret, QudKind::AskAttribute, Some("姚明"), Some("身高"), false);
        let report = store.render_report(TraceMode::On);
        assert!(report.contains("模式 on"), "{}", report);
        assert!(report.contains("问据栈"), "{}", report);
        assert!(report.contains("L0"), "{}", report);
        assert!(report.contains("L1"), "{}", report);
        assert!(report.contains("L2"), "{}", report);
    }

    #[test]
    fn last_answered_turn_for_locates_matching_reply() {
        let mut store = TraceStore::new();
        let s1 = store.record_surface(1, "姚明的身高是多少");
        let i1 = store.record_interpret(1, s1, "ask_attribute", Some("姚明"), Some("身高"), None, false);
        store.record_reply(1, Some("姚明"), Some("身高"), vec![i1], &[], Some("226"));
        let s2 = store.record_surface(2, "叶莉是谁");
        let i2 = store.record_interpret(2, s2, "ask_identity", Some("叶莉"), None, None, false);
        store.record_reply(2, Some("叶莉"), None, vec![i2], &[], Some("篮球运动员"));
        assert_eq!(store.last_answered_turn_for("姚明", "身高"), Some(1));
        assert_eq!(store.last_answered_turn_for("叶莉", "身高"), None);
    }

    #[test]
    fn empty_store_reports_cleanly() {
        let store = TraceStore::new();
        assert!(store.is_empty());
        assert_eq!(store.predicate_focus(), None);
        let report = store.render_report(TraceMode::Off);
        assert!(report.contains("（空）"), "{}", report);
        assert!(store.render_why(NodeId(1)).is_none());
    }

    #[test]
    fn alternatives_are_attached_to_nodes() {
        let mut store = TraceStore::new();
        let surface = store.record_surface(1, "他呢");
        store.add_alternative(
            surface,
            Alternative::new("「他」→叶莉", "焦点实体是姚明，叶莉不在焦点栈顶"),
        );
        let node = store.find_node(surface).expect("节点应存在");
        assert_eq!(node.alternatives.len(), 1);
        assert!(node.alternatives[0].reason.contains("姚明"));
    }

    #[test]
    fn payload_helpers_build_deterministic_maps() {
        let map = payload(&[("kind", "ask_attribute"), ("entity", "姚明")]);
        assert_eq!(map.get("entity").map(String::as_str), Some("姚明"));
        // BTreeMap 键有序：同一载荷永远同一字节串
        let keys: Vec<&String> = map.keys().collect();
        assert_eq!(keys, vec!["entity", "kind"]);
    }
}

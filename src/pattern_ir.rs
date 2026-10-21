//! 统一模式中间表示（Pattern IR）与模式库（Pattern Library）。
//!
//! 设计立场：**一次命中 = 一组带绑定、带区间、可排序、可解释的解**，
//! 而不是又一个只返回布尔的正则。本模块把此前散落在三处的模式语言
//! （`config.toml [[prolog_rules]]` 的词性/词面序列、`nlu::IntentConfig`
//! 的 `{槽位}` 模板、教学/诱导的运行时模式）收敛到一套 IR 上：
//!
//!   - [`Elem`]：最小要素（词性 / 词面 / 带排序的槽位 / 任意 / 可选）；
//!   - [`PatternEntry`]：一条有稳定指纹（FNV-1a）的模式——来源、路由、优先级
//!     与 IR 一起构成身份，可落成 RDF（见 [`pattern_quads`]）；
//!   - [`PatternLibrary`]：编译、排序、命中、体检（重复/遮蔽/冲突）与教学；
//!   - [`MatchReport`]：一次匹配的全部候选——胜者、命中未胜、失败原因，
//!     即「为什么是它 / 为什么不是别的」的机器可读底稿。
//!
//! 匹配语义与既有 [`crate::nlp`] 的 `match_pattern_for_prolog` 保持一致
//! （锚定开头、标点跳过、人名合并、词面直配/逐字跨词/包含），保证查询路由
//! 换到本引擎后行为不变；新增的槽位排序、区间与落选原因是增量能力。
//!
//! 排序规则与既有 `analyze_query_intent` 一致：**优先级降序 + 配置序**
//! （稳定排序），特化度只用于展示与体检，不改变路由结果。

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::bindings::BindingSet;
use crate::context_image::fnv1a_hex;
use crate::nlu::{DomainConfig, IntentConfig, SlotConfig};
use crate::{PrologRuleConfig, SystemConfig};

// ---------------------------------------------------------------------------
// 模式 IR
// ---------------------------------------------------------------------------

/// 模式要素：序列中的一步。序列本身由 [`PatternEntry::elems`] 表达。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Elem {
    /// 词性匹配（`nr` / `ns` / `n` / `v` …）。
    Pos { pos: String },
    /// 词面匹配（直配 → 逐字跨词 → 包含，兼容既有语义）。
    Word { surface: String },
    /// 带本体排序的洞：`sort` 约束词性/词类，`hints` 是来自配置的词面提示。
    Slot {
        name: String,
        sort: String,
        hints: Vec<String>,
    },
    /// 任意一个 token。
    Any,
    /// 可选要素（编译配置不会产生，供教学/组合用）。
    Optional(Box<Elem>),
}

/// 无递归的序列化镜像：`Elem` 经它进出 serde，避免递归类型的
/// trait 单态化溢出（E0275）；`Optional` 的内层走 `serde_json::Value`。
#[derive(Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum ElemRepr {
    Pos {
        pos: String,
    },
    Word {
        surface: String,
    },
    Slot {
        name: String,
        sort: String,
        hints: Vec<String>,
    },
    Any,
    Optional {
        inner: serde_json::Value,
    },
}

impl From<&Elem> for ElemRepr {
    fn from(elem: &Elem) -> Self {
        match elem {
            Elem::Pos { pos } => ElemRepr::Pos { pos: pos.clone() },
            Elem::Word { surface } => ElemRepr::Word {
                surface: surface.clone(),
            },
            Elem::Slot { name, sort, hints } => ElemRepr::Slot {
                name: name.clone(),
                sort: sort.clone(),
                hints: hints.clone(),
            },
            Elem::Any => ElemRepr::Any,
            Elem::Optional(inner) => ElemRepr::Optional {
                inner: serde_json::to_value(inner.as_ref())
                    .expect("Elem 到 Value 的序列化不可失败"),
            },
        }
    }
}

impl TryFrom<ElemRepr> for Elem {
    type Error = String;

    fn try_from(repr: ElemRepr) -> Result<Self, String> {
        Ok(match repr {
            ElemRepr::Pos { pos } => Elem::Pos { pos },
            ElemRepr::Word { surface } => Elem::Word { surface },
            ElemRepr::Slot { name, sort, hints } => Elem::Slot { name, sort, hints },
            ElemRepr::Any => Elem::Any,
            ElemRepr::Optional { inner } => Elem::Optional(Box::new(
                serde_json::from_value(inner).map_err(|e| e.to_string())?,
            )),
        })
    }
}

impl Serialize for Elem {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        ElemRepr::from(self).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Elem {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        ElemRepr::deserialize(deserializer)?
            .try_into()
            .map_err(serde::de::Error::custom)
    }
}

impl Elem {
    pub fn pos(pos: impl Into<String>) -> Self {
        Elem::Pos { pos: pos.into() }
    }

    pub fn word(surface: impl Into<String>) -> Self {
        Elem::Word {
            surface: surface.into(),
        }
    }

    pub fn slot(name: impl Into<String>, sort: impl Into<String>) -> Self {
        Elem::Slot {
            name: name.into(),
            sort: sort.into(),
            hints: Vec::new(),
        }
    }

    /// 单要素权重（用于特化度：词面/词性最特化，槽位次之，任意不特化）。
    fn weight(&self) -> f64 {
        match self {
            Elem::Pos { .. } | Elem::Word { .. } => 1.0,
            Elem::Slot { .. } => 0.5,
            Elem::Any => 0.0,
            Elem::Optional(inner) => inner.weight(),
        }
    }

    /// 渲染为可读词面（模式库报告用）。
    pub fn render(&self) -> String {
        match self {
            Elem::Pos { pos } => pos.clone(),
            Elem::Word { surface } => surface.clone(),
            Elem::Slot { name, sort, .. } => {
                if sort.is_empty() || sort == "*" {
                    format!("{{{}}}", name)
                } else {
                    format!("{}:{}", name, sort)
                }
            }
            Elem::Any => "_".to_string(),
            Elem::Optional(inner) => format!("[{}]", inner.render()),
        }
    }
}

/// 配置词表 token → IR 要素（与 `nlp::pattern_from_strings` 同表）。
pub fn parse_config_token(token: &str) -> Elem {
    match token {
        "r" | "n" | "nr" | "ns" | "m" | "q" | "v" | "u" | "f" | "a" | "d" => Elem::pos(token),
        "*" => Elem::Any,
        _ => Elem::word(token),
    }
}

/// 教学语法 token → IR 要素：
/// 词性码（同配置）/ `_` 任意 / `{槽名}` / `{槽名:排序}` / 其余为词面。
pub fn parse_teach_token(token: &str) -> Result<Elem, String> {
    if token == "_" {
        return Ok(Elem::Any);
    }
    if let Some(inner) = token.strip_prefix('{').and_then(|t| t.strip_suffix('}')) {
        let (name, sort) = match inner.split_once(':') {
            Some((name, sort)) => (name.trim(), sort.trim()),
            None => (inner.trim(), "*"),
        };
        if name.is_empty() {
            return Err(format!("槽位名不能为空：{}", token));
        }
        return Ok(Elem::slot(name, if sort.is_empty() { "*" } else { sort }));
    }
    match token {
        "r" | "n" | "nr" | "ns" | "m" | "q" | "v" | "u" | "f" | "a" | "d" => Ok(Elem::pos(token)),
        _ => {
            if token.is_empty() {
                return Err("空词面".to_string());
            }
            Ok(Elem::word(token))
        }
    }
}

/// 教学模式串（空白分隔）→ IR 要素序列。
pub fn parse_teach_pattern(pattern: &str) -> Result<Vec<Elem>, String> {
    let mut elems = Vec::new();
    for token in pattern.split_whitespace() {
        elems.push(parse_teach_token(token)?);
    }
    if elems.is_empty() {
        return Err("模式至少要有一个要素".to_string());
    }
    Ok(elems)
}

// ---------------------------------------------------------------------------
// 模式条目
// ---------------------------------------------------------------------------

fn is_false(value: &bool) -> bool {
    !*value
}

/// 模式所需的对话上下文（可选守卫）。
///
/// 守卫把「这句话什么时候成立」从手写分支移进模式本身：省略/指代问句只有在
/// 焦点、上轮动作或澄清挂账满足时才参与路由；完整句通常不声明守卫。
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ContextGuard {
    /// 需要已有焦点实体。
    #[serde(default, skip_serializing_if = "is_false")]
    pub focus_required: bool,
    /// 焦点实体的类型（同时蕴含 `focus_required`）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub focus_type: Option<String>,
    /// 上一轮对话动作（`ask_attribute` / `assert` …）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_move: Option<String>,
    /// 是否必须存在（或必须不存在）待回答的澄清。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pending_clarify: Option<bool>,
}

impl ContextGuard {
    pub fn focus() -> Self {
        Self {
            focus_required: true,
            ..Self::default()
        }
    }

    pub fn focus_type(sort: impl Into<String>) -> Self {
        Self {
            focus_required: true,
            focus_type: Some(sort.into()),
            ..Self::default()
        }
    }

    pub fn last_move(movement: impl Into<String>) -> Self {
        Self {
            last_move: Some(movement.into()),
            ..Self::default()
        }
    }

    pub fn pending(pending: bool) -> Self {
        Self {
            pending_clarify: Some(pending),
            ..Self::default()
        }
    }

    /// 是否被当前对话上下文满足。
    pub fn satisfied(&self, context: &PatternContext) -> bool {
        if self.focus_required && context.focus.is_none() {
            return false;
        }
        if let Some(focus_type) = &self.focus_type {
            if context.focus_type.as_deref() != Some(focus_type.as_str()) {
                return false;
            }
        }
        if let Some(last_move) = &self.last_move {
            if context.last_move.as_deref() != Some(last_move.as_str()) {
                return false;
            }
        }
        if let Some(pending) = self.pending_clarify {
            if context.pending_clarify != pending {
                return false;
            }
        }
        true
    }

    /// 人类可读渲染（报告 / 图上的守卫说明）。
    pub fn render(&self) -> String {
        let mut parts = Vec::new();
        if self.focus_required || self.focus_type.is_some() {
            match &self.focus_type {
                Some(sort) => parts.push(format!("焦点类型={}", sort)),
                None => parts.push("焦点".to_string()),
            }
        }
        if let Some(movement) = &self.last_move {
            parts.push(format!("上轮={}", movement));
        }
        if let Some(pending) = self.pending_clarify {
            parts.push(if pending {
                "待澄清".to_string()
            } else {
                "无待澄清".to_string()
            });
        }
        if parts.is_empty() {
            "（无守卫）".to_string()
        } else {
            parts.join("；")
        }
    }
}

/// 路由时可见的对话上下文快照。
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct PatternContext {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub focus: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub focus_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_move: Option<String>,
    #[serde(default)]
    pub pending_clarify: bool,
    #[serde(default)]
    pub turn: usize,
}

/// 解析教学命令里的守卫串：`焦点`、`焦点:人物`、`上轮:ask_attribute`、
/// `待澄清`/`无待澄清`，多项用逗号或分号分隔。
pub fn parse_context_guard(spec: &str) -> Result<ContextGuard, String> {
    let spec = spec.trim();
    if spec.is_empty() {
        return Err("守卫不能为空".to_string());
    }
    let mut guard = ContextGuard::default();
    for part in spec.split([',', '，', ';', '；']) {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let (kind, value) = part
            .split_once([':', '：', '=', '＝'])
            .map(|(kind, value)| (kind.trim(), Some(value.trim())))
            .unwrap_or((part, None));
        match (kind, value) {
            ("焦点", None) | ("focus", None) => guard.focus_required = true,
            ("焦点类型", Some(sort)) | ("类型", Some(sort)) | ("focus_type", Some(sort)) => {
                if sort.is_empty() {
                    return Err("焦点类型不能为空".to_string());
                }
                guard.focus_required = true;
                guard.focus_type = Some(sort.to_string());
            }
            ("上轮", Some(movement))
            | ("动作", Some(movement))
            | ("last_move", Some(movement)) => {
                if movement.is_empty() {
                    return Err("上轮动作不能为空".to_string());
                }
                guard.last_move = Some(movement.to_string());
            }
            ("待澄清", None) | ("pending", None) => guard.pending_clarify = Some(true),
            ("无待澄清", None) | ("不待澄清", None) | ("no_pending", None) => {
                guard.pending_clarify = Some(false)
            }
            _ => return Err(format!("无法识别的守卫：{}", part)),
        }
    }
    if guard == ContextGuard::default() {
        return Err("守卫至少要限定一个条件".to_string());
    }
    Ok(guard)
}

/// 一条有稳定身份的模式：IR + 来源 + 路由 + 优先级。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PatternEntry {
    /// 人类可读名称（配置名 / 意图名 / 教学名）。
    pub name: String,
    /// 来源：`config.toml#<规则名>` / `nlu#<领域>#<意图>` / `taught#会话` / `loaded#…`。
    pub source: String,
    /// 命中后路由到的查询类型（NLU 意图条目为意图名，仅展示不路由）。
    pub query_type: String,
    /// 优先级（越大越先匹配；与配置语义一致）。
    pub priority: i32,
    /// 是否参与查询路由（配置查询与教学模式为 true；NLU 意图为 false，
    /// 它们进库只为统一视图与解释，不接管 `analyze_query_intent` 的路由）。
    pub route: bool,
    /// 模式 IR（序列）。
    pub elems: Vec<Elem>,
    /// 可选的对话上下文守卫；不满足时该模式在本次路由中落选。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub guard: Option<ContextGuard>,
    /// 是否作为 NLU 回退模式：主路由全部未命中时才参与。
    ///
    /// 配置查询规则始终优先；NLU 意图只有在与同名配置规则对齐出
    /// 可执行的 `query_type` 后才具备回退资格，避免把天气/聊天意图
    /// 误接进知识查询。
    #[serde(default)]
    pub fallback: bool,
}

impl PatternEntry {
    pub fn new(
        name: impl Into<String>,
        source: impl Into<String>,
        query_type: impl Into<String>,
        priority: i32,
        route: bool,
        elems: Vec<Elem>,
    ) -> Self {
        Self {
            name: name.into(),
            source: source.into(),
            query_type: query_type.into(),
            priority,
            route,
            elems,
            guard: None,
            fallback: false,
        }
    }

    /// 附加上下文守卫（链式；教学命令与测试使用）。
    pub fn with_guard(mut self, guard: ContextGuard) -> Self {
        self.guard = Some(guard);
        self
    }

    /// 条目的上下文是否满足（无守卫恒满足）。
    pub fn context_satisfied(&self, context: &PatternContext) -> bool {
        self.guard
            .as_ref()
            .map(|guard| guard.satisfied(context))
            .unwrap_or(true)
    }

    /// 由配置的一条 `[[prolog_rules]]` 编译。
    pub fn from_prolog_rule(rule: &PrologRuleConfig) -> Self {
        Self::new(
            rule.name.clone(),
            format!("config.toml#{}", rule.name),
            rule.query_type.clone(),
            rule.priority,
            true,
            rule.pattern.iter().map(|t| parse_config_token(t)).collect(),
        )
    }

    /// 由 NLU 的一个意图模式串编译（槽位来自意图的槽配置）。
    pub fn from_nlu_intent(domain: &DomainConfig, intent: &IntentConfig, pattern: &str) -> Self {
        let elems = pattern
            .split_whitespace()
            .map(|token| {
                if let Some(name) = token
                    .strip_prefix('{')
                    .and_then(|t| t.strip_suffix('}'))
                    .map(str::trim)
                {
                    match intent.slots.iter().find(|slot| slot.name == name) {
                        Some(SlotConfig {
                            slot_type,
                            patterns,
                            ..
                        }) => Elem::Slot {
                            name: name.to_string(),
                            sort: slot_type.clone(),
                            hints: patterns.clone(),
                        },
                        None => Elem::slot(name, "*"),
                    }
                } else {
                    Elem::word(token)
                }
            })
            .collect();
        Self::new(
            intent.name.clone(),
            format!("nlu#{}#{}", domain.name, intent.name),
            intent.name.clone(),
            intent.priority,
            false,
            elems,
        )
    }

    /// 模式 IR 的规范 JSON（serde 字段序固定，序列化即规范形）。
    pub fn ir_json(&self) -> String {
        serde_json::to_string(&self.elems).unwrap_or_default()
    }

    /// 仅 IR 的指纹：跨来源判「同一模式语言」（重复/冲突体检用）。
    pub fn ir_fingerprint(&self) -> String {
        fnv1a_hex(self.ir_json().as_bytes())
    }

    /// 条目身份 = FNV(IR + 路由类型 + 来源)：同定义同 id，可作 RDF 主语。
    pub fn id(&self) -> String {
        fnv1a_hex(
            format!(
                "{}\u{0}{}\u{0}{}",
                self.ir_json(),
                self.query_type,
                self.source
            )
            .as_bytes(),
        )
    }

    /// 短 id（展示用 8 位）。
    pub fn short_id(&self) -> String {
        self.id().chars().take(8).collect()
    }

    /// 要素渲染（词面序列，供报告展示）。
    pub fn render(&self) -> String {
        self.elems
            .iter()
            .map(Elem::render)
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// 特化度：要素权重均值（1.0 全为词面/词性，越小越泛）。
    pub fn specificity(&self) -> f64 {
        if self.elems.is_empty() {
            return 0.0;
        }
        let total: f64 = self.elems.iter().map(Elem::weight).sum();
        total / self.elems.len() as f64
    }
}

// ---------------------------------------------------------------------------
// 匹配结果
// ---------------------------------------------------------------------------

/// token 区间（半开，下标为分词序列中的位置）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    /// 词数（0 表示空跨度）。
    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    pub fn is_empty(&self) -> bool {
        self.end <= self.start
    }
}

impl std::fmt::Display for Span {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{},{}）", self.start, self.end)
    }
}

/// 一次命中：条目 + 变量绑定 + 区间证据 + 特化度。
#[derive(Debug, Clone, PartialEq)]
pub struct PatternMatch {
    pub entry: PatternEntry,
    /// 槽位与自动实体槽的绑定（`entity.1`、`entity.2` … 按命中序）。
    pub bindings: BindingSet,
    /// 自动实体槽的有序值（与既有 `analyze_query_intent` 的实体序列同构）。
    pub entities: Vec<String>,
    /// 自动实体槽的 token 区间。
    pub entity_spans: Vec<Span>,
    /// 各要素命中的 token 区间（`None` = 可选未占位）。
    pub elem_spans: Vec<Option<Span>>,
    /// 整体覆盖区间。
    pub span: Span,
}

impl PatternMatch {
    /// 实体序列（与 `nlp::analyze_query_intent` 的返回同构）。
    pub fn entity_values(&self) -> Vec<String> {
        self.entities.clone()
    }
}

// ---------------------------------------------------------------------------
// 模式经验：一次理解的可检索记忆
// ---------------------------------------------------------------------------

/// 一次解析尝试的结果类别。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EpisodeOutcome {
    /// 可路由模式命中并给出了查询解。
    Matched,
    /// 没有可路由模式命中（失败经验同样可检索）。
    Unmatched,
}

impl EpisodeOutcome {
    pub fn label(self) -> &'static str {
        match self {
            EpisodeOutcome::Matched => "命中",
            EpisodeOutcome::Unmatched => "未命中",
        }
    }
}

/// 一次解析经验：输入、词序列、命中模式、绑定与当时的对话焦点。
///
/// 经验不是「背答案」：它记录的是**怎么理解**——哪条模式、绑定到什么、
/// 在什么焦点下发生。相同经验按 [`PatternEpisode::id`] 合并并累计次数，
/// 从而让「支持度」本身成为可查询的数据。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PatternEpisode {
    pub input: String,
    /// 规范化后的词序列（词面 + 词性），用于结构相似检索。
    pub tokens: Vec<(String, String)>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub winner_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub winner_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_type: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub entities: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub bindings: Vec<(String, String)>,
    /// 发生时的对话焦点（无焦点为 None）；检索时作为上下文守卫的弱条件。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub focus: Option<String>,
    /// 焦点实体的类型（人物/地点/…），用于区分同一词面在不同类型下的经验。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub focus_type: Option<String>,
    /// 上一轮的对话动作（`ask_attribute` / `ask_identity` …）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_move: Option<String>,
    /// 当时是否存在待澄清请求（None = 旧账本未记录）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pending_clarify: Option<bool>,
    pub outcome: EpisodeOutcome,
    /// 同一经验被重复观察的次数（支持度）。
    #[serde(default = "one_u32")]
    pub count: u32,
    /// 用户确认「案例联想」的次数（正向反馈）。
    #[serde(default)]
    pub acceptances: u32,
    /// 用户否定「案例联想」的次数（负向反馈；被否定的骨架不再主动推荐）。
    #[serde(default)]
    pub rejections: u32,
    #[serde(default)]
    pub first_turn: usize,
    #[serde(default)]
    pub last_turn: usize,
}

fn one_u32() -> u32 {
    1
}

impl PatternEpisode {
    /// 经验身份：输入 + 命中模式 + 完整对话上下文 + 结果。
    /// 同一问法在不同焦点/上轮动作下是不同经验，避免上下文被后一次覆盖。
    pub fn id(&self) -> String {
        fnv1a_hex(
            format!(
                "{}\u{0}{}\u{0}{}\u{0}{}\u{0}{}\u{0}{}\u{0}{}",
                self.input,
                self.winner_id.as_deref().unwrap_or(""),
                self.focus.as_deref().unwrap_or(""),
                self.focus_type.as_deref().unwrap_or(""),
                self.last_move.as_deref().unwrap_or(""),
                self.pending_clarify
                    .map(|pending| if pending { "1" } else { "0" })
                    .unwrap_or(""),
                self.outcome.label()
            )
            .as_bytes(),
        )
    }

    /// 人类可读的上下文说明（无上下文字段时为 None）。
    pub fn context_note(&self) -> Option<String> {
        let mut parts = Vec::new();
        match (&self.focus, &self.focus_type) {
            (Some(focus), Some(sort)) => parts.push(format!("焦点 {}({})", focus, sort)),
            (Some(focus), None) => parts.push(format!("焦点 {}", focus)),
            (None, Some(sort)) => parts.push(format!("焦点类型 {}", sort)),
            (None, None) => {}
        }
        if let Some(movement) = &self.last_move {
            parts.push(format!("上轮 {}", movement));
        }
        if let Some(true) = self.pending_clarify {
            parts.push("待澄清".to_string());
        }
        if parts.is_empty() {
            None
        } else {
            Some(parts.join("，"))
        }
    }

    /// 上下文与当前对话状态的亲和度（只做弱加分，不压过词面结构）。
    fn context_affinity(&self, context: &PatternContext) -> f64 {
        let mut bonus = 0.0;
        if let (Some(left), Some(right)) = (self.focus.as_deref(), context.focus.as_deref()) {
            if left == right {
                bonus += 0.2;
            }
        }
        if let (Some(left), Some(right)) =
            (self.focus_type.as_deref(), context.focus_type.as_deref())
        {
            if left == right {
                bonus += 0.1;
            }
        }
        if let (Some(left), Some(right)) =
            (self.last_move.as_deref(), context.last_move.as_deref())
        {
            if left == right {
                bonus += 0.1;
            }
        }
        if let Some(pending) = self.pending_clarify {
            if pending == context.pending_clarify {
                bonus += 0.05;
            }
        }
        bonus
    }

    pub fn short_id(&self) -> String {
        self.id().chars().take(8).collect()
    }

    /// 词面集合（去重、排除标点/空白），供相似度与骨架泛化使用。
    fn surface_set(&self) -> BTreeSet<&str> {
        self.tokens
            .iter()
            .map(|(text, _)| text.as_str())
            .filter(|text| !is_skippable(text))
            .collect()
    }
}

/// 相似经验检索结果（只读片段，避免为报告复制整条经验）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpisodeNeighbor {
    pub index: usize,
    pub score: f64,
    /// 与当前输入共享的词面（确定性序）。
    pub shared: Vec<String>,
}

/// 由经验泛化出的候选模式：只用于确认，不直接参与路由。
#[derive(Debug, Clone, PartialEq)]
pub struct CaseSuggestion {
    pub episode: PatternEpisode,
    pub score: f64,
    pub shared: Vec<String>,
    /// 候选条目（`route=false`；确认后应通过教学接口显式激活）。
    pub pattern: PatternEntry,
}

/// 用户对「模式联想」的回答类别。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaseReply {
    /// 确认候选骨架，应转正为可路由模式。
    Accept,
    /// 否定候选骨架，应记入负向反馈。
    Reject,
    /// 暂不确定：关闭本次联想，但不记否定。
    Unclear,
    /// 不是对联想提问的回答（按新输入继续处理）。
    NewInput,
}

/// 把一轮输入分类为对案例联想的确认/否定/不确定。
///
/// 只接受**整句短答**（可带句末语气词），避免把「是什么」「是的，首都在哪」
/// 这类新问题误当成对上一轮候选的确认。
pub fn classify_case_reply(input: &str) -> CaseReply {
    let trimmed = input
        .trim()
        .trim_start_matches(['：', ':', ' ', '，', ','])
        .trim_end_matches(['。', '！', '!', '~', '～', ' ', '？', '?', '，', ',']);
    if trimmed.is_empty() {
        return CaseReply::NewInput;
    }
    let mut core = trimmed;
    for suffix in ['吧', '啊', '呀', '呢', '哦'] {
        if let Some(stripped) = core.strip_suffix(suffix) {
            if !stripped.is_empty() {
                core = stripped;
            }
        }
    }
    let lower = core.to_lowercase();
    let accept = [
        "是", "是的", "对", "对的", "没错", "确认", "好", "好的", "好呀", "可以", "嗯",
        "yes", "y", "true",
    ];
    let reject = [
        "不是", "不对", "没有", "否", "不", "不要", "不用", "别", "错", "算了", "no", "n",
        "false",
    ];
    let unclear = ["不知道", "不确定", "不清楚", "跳过", "下一个", "随便"];
    if accept.contains(&lower.as_str()) {
        CaseReply::Accept
    } else if reject.contains(&lower.as_str()) {
        CaseReply::Reject
    } else if unclear.contains(&lower.as_str()) {
        CaseReply::Unclear
    } else {
        CaseReply::NewInput
    }
}

/// 有界、去重、可检索的模式经验台账。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PatternEpisodeLog {
    entries: Vec<PatternEpisode>,
    #[serde(default = "default_episode_capacity")]
    capacity: usize,
    #[serde(default)]
    clock: usize,
}

fn default_episode_capacity() -> usize {
    64
}

impl Default for PatternEpisodeLog {
    fn default() -> Self {
        Self::with_capacity(default_episode_capacity())
    }
}

impl PatternEpisodeLog {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::new(),
            capacity: capacity.max(1),
            clock: 0,
        }
    }

    pub fn entries(&self) -> &[PatternEpisode] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn clock(&self) -> usize {
        self.clock
    }

    pub fn matched_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|episode| episode.outcome == EpisodeOutcome::Matched)
            .count()
    }

    pub fn unmatched_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|episode| episode.outcome == EpisodeOutcome::Unmatched)
            .count()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// 删除单条经验：按完整 id、短 id、完整原句或至少 4 位十六进制 id 前缀匹配。
    /// 返回删除条数；用于撤销误确认/误否定而不清空整本经验。
    pub fn forget(&mut self, needle: &str) -> usize {
        let needle = needle.trim();
        if needle.is_empty() {
            return 0;
        }
        let prefix = needle.chars().count() >= 4
            && needle.chars().all(|ch| ch.is_ascii_hexdigit())
            && needle.chars().count() < 16;
        let before = self.entries.len();
        self.entries.retain(|episode| {
            let id = episode.id();
            !(id == needle
                || episode.short_id() == needle
                || episode.input.trim() == needle
                || (prefix && id.starts_with(needle)))
        });
        before - self.entries.len()
    }

    /// 记录一次成功理解（同身份经验累计次数并刷新最后轮次）。
    pub fn record_match(
        &mut self,
        input: &str,
        tokens: &[crate::nlp::TaggedWord],
        matched: &PatternMatch,
        focus: Option<&str>,
        turn: usize,
    ) -> PatternEpisode {
        let context = PatternContext {
            focus: focus.map(str::to_string),
            turn,
            ..PatternContext::default()
        };
        self.record_match_with_context(input, tokens, matched, &context)
    }

    /// 记录一次带完整对话上下文的成功理解。
    pub fn record_match_with_context(
        &mut self,
        input: &str,
        tokens: &[crate::nlp::TaggedWord],
        matched: &PatternMatch,
        context: &PatternContext,
    ) -> PatternEpisode {
        let bindings = matched
            .bindings
            .iter()
            .filter_map(|(name, term)| term.render_raw().map(|value| (name.clone(), value)))
            .collect();
        let episode = PatternEpisode {
            input: input.to_string(),
            tokens: token_pairs(tokens),
            winner_id: Some(matched.entry.id()),
            winner_name: Some(matched.entry.name.clone()),
            query_type: Some(matched.entry.query_type.clone()),
            entities: matched.entities.clone(),
            bindings,
            focus: context.focus.clone(),
            focus_type: context.focus_type.clone(),
            last_move: context.last_move.clone(),
            pending_clarify: Some(context.pending_clarify),
            outcome: EpisodeOutcome::Matched,
            count: 1,
            acceptances: 0,
            rejections: 0,
            first_turn: self.effective_turn(context.turn),
            last_turn: self.effective_turn(context.turn),
        };
        self.upsert(episode.clone());
        self.entries
            .iter()
            .find(|entry| entry.id() == episode.id())
            .cloned()
            .unwrap_or(episode)
    }

    /// 记录一次未命中（用于「上次为什么没懂」与候选骨架检索）。
    pub fn record_unmatched(
        &mut self,
        input: &str,
        tokens: &[crate::nlp::TaggedWord],
        focus: Option<&str>,
        turn: usize,
    ) -> PatternEpisode {
        let context = PatternContext {
            focus: focus.map(str::to_string),
            turn,
            ..PatternContext::default()
        };
        self.record_unmatched_with_context(input, tokens, &context)
    }

    /// 记录一次带完整对话上下文的未命中。
    pub fn record_unmatched_with_context(
        &mut self,
        input: &str,
        tokens: &[crate::nlp::TaggedWord],
        context: &PatternContext,
    ) -> PatternEpisode {
        let episode = PatternEpisode {
            input: input.to_string(),
            tokens: token_pairs(tokens),
            winner_id: None,
            winner_name: None,
            query_type: None,
            entities: Vec::new(),
            bindings: Vec::new(),
            focus: context.focus.clone(),
            focus_type: context.focus_type.clone(),
            last_move: context.last_move.clone(),
            pending_clarify: Some(context.pending_clarify),
            outcome: EpisodeOutcome::Unmatched,
            count: 1,
            acceptances: 0,
            rejections: 0,
            first_turn: self.effective_turn(context.turn),
            last_turn: self.effective_turn(context.turn),
        };
        self.upsert(episode.clone());
        self.entries
            .iter()
            .find(|entry| entry.id() == episode.id())
            .cloned()
            .unwrap_or(episode)
    }

    fn effective_turn(&mut self, turn: usize) -> usize {
        let effective = if turn == 0 { self.clock + 1 } else { turn };
        self.clock = self.clock.max(effective);
        effective
    }

    fn upsert(&mut self, episode: PatternEpisode) -> bool {
        let id = episode.id();
        if let Some(existing) = self.entries.iter_mut().find(|entry| entry.id() == id) {
            existing.count = existing.count.saturating_add(1);
            existing.last_turn = existing.last_turn.max(episode.last_turn);
            existing.tokens = episode.tokens;
            existing.entities = episode.entities;
            existing.bindings = episode.bindings;
            existing.acceptances = existing.acceptances.max(episode.acceptances);
            existing.rejections = existing.rejections.max(episode.rejections);
            return false;
        }
        self.entries.push(episode);
        self.enforce_capacity();
        true
    }

    fn enforce_capacity(&mut self) {
        while self.entries.len() > self.capacity {
            // 失败经验先被淘汰；同为失败/成功时再淘汰支持度低、最久未用的。
            let victim = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, episode)| {
                    (
                        u8::from(episode.outcome == EpisodeOutcome::Matched),
                        episode.count,
                        episode.last_turn,
                        episode.id(),
                    )
                })
                .map(|(index, _)| index);
            match victim {
                Some(index) => {
                    self.entries.remove(index);
                }
                None => break,
            }
        }
    }

    /// 并入从图账本恢复的经验（同身份取较大支持度与较新轮次）。
    pub fn merge_persisted(&mut self, episodes: &[PatternEpisode]) -> usize {
        let mut added = 0usize;
        for episode in episodes {
            let id = episode.id();
            if let Some(existing) = self.entries.iter_mut().find(|entry| entry.id() == id) {
                existing.count = existing.count.max(episode.count);
                existing.acceptances = existing.acceptances.max(episode.acceptances);
                existing.rejections = existing.rejections.max(episode.rejections);
                existing.first_turn = existing.first_turn.min(episode.first_turn);
                existing.last_turn = existing.last_turn.max(episode.last_turn);
            } else {
                self.entries.push(episode.clone());
                added += 1;
            }
        }
        self.clock = self
            .clock
            .max(self.entries.iter().map(|episode| episode.last_turn).max().unwrap_or(0));
        self.enforce_capacity();
        added
    }

    /// 按共享词面 + 焦点 + 结果检索相似经验（确定性排序）。
    pub fn nearest(
        &self,
        tokens: &[crate::nlp::TaggedWord],
        focus: Option<&str>,
        limit: usize,
    ) -> Vec<EpisodeNeighbor> {
        let context = PatternContext {
            focus: focus.map(str::to_string),
            ..PatternContext::default()
        };
        self.nearest_with_context(tokens, &context, limit)
    }

    /// 按共享词面 + 完整对话上下文 + 结果检索相似经验（确定性排序）。
    ///
    /// 上下文是**加性弱条件**：词面结构仍是主信号，焦点/类型/上轮动作/
    /// 待澄清只在相近结构之间做消歧，避免同句式在错误对话状态下被召回。
    pub fn nearest_with_context(
        &self,
        tokens: &[crate::nlp::TaggedWord],
        context: &PatternContext,
        limit: usize,
    ) -> Vec<EpisodeNeighbor> {
        if limit == 0 {
            return Vec::new();
        }
        let current: BTreeSet<&str> = tokens
            .iter()
            .map(|token| token.text.as_str())
            .filter(|text| !is_skippable(text))
            .collect();
        let mut neighbors: Vec<EpisodeNeighbor> = Vec::new();
        for (index, episode) in self.entries.iter().enumerate() {
            let stored = episode.surface_set();
            let shared: Vec<String> = episode
                .tokens
                .iter()
                .map(|(text, _)| text.as_str())
                .filter(|text| current.contains(text))
                .fold(Vec::new(), |mut acc, text| {
                    if !acc.iter().any(|seen| seen == text) {
                        acc.push(text.to_string());
                    }
                    acc
                });
            let union = current.union(&stored).count().max(1);
            let overlap = current.intersection(&stored).count() as f64 / union as f64;
            let context_bonus = episode.context_affinity(context);
            let outcome_bonus = if episode.outcome == EpisodeOutcome::Matched {
                0.05
            } else {
                0.0
            };
            // 被用户否定过的成功经验仍可检索（解释为什么不再推荐），
            // 但排序时扣分；净否定越多扣得越重，最多扣到 0。
            let rejection_penalty = episode
                .rejections
                .saturating_sub(episode.acceptances)
                .min(3) as f64
                * 0.2;
            if shared.is_empty() && context_bonus == 0.0 {
                continue;
            }
            neighbors.push(EpisodeNeighbor {
                index,
                score: (overlap + context_bonus + outcome_bonus - rejection_penalty)
                    .clamp(0.0, 1.0),
                shared,
            });
        }
        neighbors.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    self.entries[b.index]
                        .context_affinity(context)
                        .partial_cmp(&self.entries[a.index].context_affinity(context))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| {
                    self.entries[b.index]
                        .count
                        .cmp(&self.entries[a.index].count)
                })
                .then_with(|| self.entries[a.index].id().cmp(&self.entries[b.index].id()))
        });
        neighbors.truncate(limit);
        neighbors
    }

    /// 由最相似的成功经验泛化出「临时模式」：共享词保留为词面，
    /// 当前输入中的新内容词升为槽位。候选项 `route=false`，只能被确认后教学激活。
    pub fn suggest_case(
        &self,
        tokens: &[crate::nlp::TaggedWord],
        focus: Option<&str>,
        min_score: f64,
    ) -> Option<CaseSuggestion> {
        let context = PatternContext {
            focus: focus.map(str::to_string),
            ..PatternContext::default()
        };
        self.suggest_case_with_context(tokens, &context, min_score)
    }

    /// 带完整对话上下文的案例联想。
    pub fn suggest_case_with_context(
        &self,
        tokens: &[crate::nlp::TaggedWord],
        context: &PatternContext,
        min_score: f64,
    ) -> Option<CaseSuggestion> {
        // 取前若干个邻居逐个尝试：最近的一条可能是否定过的骨架、
        // 未命中的经验，或泛化后无法解释当前输入；不应因此放弃次近的经验。
        for neighbor in self.nearest_with_context(tokens, context, 5) {
            if neighbor.score < min_score {
                break;
            }
            let Some(episode) = self.entries.get(neighbor.index).cloned() else {
                continue;
            };
            if episode.outcome != EpisodeOutcome::Matched
                || episode.rejections > episode.acceptances
            {
                continue;
            }
            let shared = episode.surface_set();
            let mut elems = Vec::new();
            let mut slot_number = 0usize;
            for token in tokens {
                let text = token.text.trim();
                if text.is_empty() || is_skippable(text) {
                    continue;
                }
                if shared.contains(text) {
                    elems.push(Elem::word(text));
                    continue;
                }
                let sort = match token.pos.as_str() {
                    "nr" => "人名",
                    "ns" | "nt" => "地点",
                    "n" | "nz" | "vn" => "实体",
                    "m" => "数量",
                    _ => "",
                };
                if sort.is_empty() {
                    elems.push(Elem::word(text));
                } else {
                    slot_number += 1;
                    elems.push(Elem::slot(format!("槽{}", slot_number), sort));
                }
            }
            if elems.is_empty() {
                continue;
            }
            let Some(query_type) = episode.query_type.clone() else {
                continue;
            };
            let mut entry = PatternEntry::new(
                format!(
                    "案例候选：{}",
                    episode.winner_name.as_deref().unwrap_or("经验")
                ),
                format!("case#{}", episode.short_id()),
                query_type,
                5,
                false,
                elems,
            );
            // 当前输入没有显式实体（靠焦点补全的省略/指代问句）时，
            // 候选必须带上焦点守卫；确认转正后也不会在无焦点时误答。
            let has_explicit_entity = tokens
                .iter()
                .any(|token| matches!(token.pos.as_str(), "nr" | "ns" | "nt"));
            if !has_explicit_entity {
                if let Some(sort) = &context.focus_type {
                    entry = entry.with_guard(ContextGuard::focus_type(sort.clone()));
                } else if context.focus.is_some() {
                    entry = entry.with_guard(ContextGuard::focus());
                }
            }
            // 候选必须能解释当前输入；否则继续看下一条经验。
            if match_entry(&entry, tokens).is_err() {
                continue;
            }
            return Some(CaseSuggestion {
                episode,
                score: neighbor.score,
                shared: neighbor.shared,
                pattern: entry,
            });
        }
        None
    }

    /// 记录用户确认了某条案例联想（正向反馈，随经验账本持久化）。
    pub fn confirm_case(&mut self, suggestion: &CaseSuggestion) -> bool {
        let id = suggestion.episode.id();
        if let Some(episode) = self.entries.iter_mut().find(|entry| entry.id() == id) {
            episode.acceptances = episode.acceptances.saturating_add(1);
            return true;
        }
        false
    }

    /// 记录用户否定了某条案例联想（负向反馈；净否定的骨架不再主动推荐）。
    pub fn reject_case(&mut self, suggestion: &CaseSuggestion) -> bool {
        let id = suggestion.episode.id();
        if let Some(episode) = self.entries.iter_mut().find(|entry| entry.id() == id) {
            episode.rejections = episode.rejections.saturating_add(1);
            return true;
        }
        false
    }
}

fn token_pairs(tokens: &[crate::nlp::TaggedWord]) -> Vec<(String, String)> {
    tokens
        .iter()
        .map(|token| (token.text.clone(), token.pos.clone()))
        .collect()
}

/// 单个候选的评估结果（报告用；不携带完整解，避免大对象复制）。
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateOutcome {
    pub id: String,
    pub name: String,
    pub query_type: String,
    pub source: String,
    pub priority: i32,
    pub route: bool,
    pub status: CandidateStatus,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CandidateStatus {
    /// 命中（`rank` 为命中序，从 0 起；按优先级+配置序）。
    Matched {
        rank: usize,
        specificity: f64,
        span: Span,
        bindings: usize,
    },
    /// 未命中（首个失配要素的原因）。
    Failed { reason: String },
}

/// 快速路由的一次扫描：胜者 + 实际完整评估/被首要素预筛跳过的条数。
#[derive(Debug, Clone, PartialEq)]
pub struct RouteScan {
    pub winner: Option<PatternMatch>,
    /// 进入完整 `match_entry` 的候选数。
    pub evaluated: usize,
    /// 首要素必然失配、无需完整匹配的候选数。
    pub skipped: usize,
}

impl RouteScan {
    pub fn scanned(&self) -> usize {
        self.evaluated + self.skipped
    }
}

/// 一次匹配的完整账目：胜者、全部命中、全部失败原因。
#[derive(Debug, Clone, PartialEq)]
pub struct MatchReport {
    pub winner: Option<PatternMatch>,
    pub matched: Vec<CandidateOutcome>,
    pub failed: Vec<CandidateOutcome>,
}

impl MatchReport {
    pub fn matched_count(&self) -> usize {
        self.matched.len()
    }

    /// 同覆盖区间但路由不同的命中（潜在冲突，体检/解释用）。
    pub fn conflicts(&self) -> Vec<(String, String, Span)> {
        let mut out = Vec::new();
        for (i, a) in self.matched.iter().enumerate() {
            let span_a = match a.status {
                CandidateStatus::Matched { span, .. } => span,
                _ => continue,
            };
            for b in self.matched.iter().skip(i + 1) {
                if a.query_type == b.query_type {
                    continue;
                }
                if let CandidateStatus::Matched { span, .. } = b.status {
                    if span == span_a {
                        out.push((a.name.clone(), b.name.clone(), span_a));
                    }
                }
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// 匹配器
// ---------------------------------------------------------------------------

/// 标点/空白 token：匹配推进时跳过（与既有语义一致）。
fn is_skippable(text: &str) -> bool {
    text.trim().is_empty() || matches!(text, "？" | "。" | "，" | "、")
}

const CITY_LITERALS: &[&str] = &[
    "厦门", "北京", "上海", "广州", "深圳", "杭州", "南京", "武汉",
];
const PERSON_LITERALS: &[&str] = &["姚明", "刘德华", "张三", "李四", "王五", "李明"];
const DATE_WORDS: &[&str] = &["明天", "后天", "今天", "昨天", "大后天"];

/// 槽位排序约束：与 `nlu::NLUEngine::extract_slot_value` 同口径
/// （词性集 + 词面提示），锚定在当前位置的 token 上。
fn sort_matches(sort: &str, hints: &[String], token: &crate::nlp::TaggedWord) -> bool {
    match sort {
        "城市" | "地点" => {
            token.pos == "ns"
                || token.pos == "n"
                || CITY_LITERALS.contains(&token.text.as_str())
                || hints.iter().any(|hint| token.text.contains(hint.as_str()))
        }
        "日期" | "时间" => {
            DATE_WORDS.contains(&token.text.as_str())
                || (token.pos == "m"
                    && (token.text.contains('天')
                        || token.text.contains('月')
                        || token.text.contains('年')))
                || hints.iter().any(|hint| token.text.contains(hint.as_str()))
        }
        "人名" => token.pos == "nr" || PERSON_LITERALS.contains(&token.text.as_str()),
        "数量" => token.pos == "m" || token.text.chars().all(|c| c.is_numeric()),
        "实体" => {
            token.pos == "nr"
                || token.pos == "ns"
                || token.pos == "n"
                || PERSON_LITERALS.contains(&token.text.as_str())
        }
        "属性" => matches!(token.pos.as_str(), "n" | "nz" | "vn"),
        _ => true,
    }
}

/// 单要素尝试的结果（只在成功时产生副作用）。
enum Attempt {
    Consumed {
        new_pos: usize,
        span: Span,
        entity: Option<String>,
        binding: Option<(String, String)>,
    },
    /// 可选要素未占位。
    Absent,
}

fn attempt_elem(
    elem: &Elem,
    pattern: &[Elem],
    index: usize,
    tokens: &[crate::nlp::TaggedWord],
    pos: usize,
) -> Result<Attempt, String> {
    let token = &tokens[pos];
    match elem {
        Elem::Optional(inner) => {
            attempt_elem(inner, pattern, index, tokens, pos).or(Ok(Attempt::Absent))
        }
        Elem::Pos { pos: expected } if expected == "nr" => {
            if token.pos != "nr" && token.pos != "unknown" {
                return Err(format!(
                    "第{}要素期望人名(nr)，实得词性「{}」（「{}」）",
                    index + 1,
                    token.pos,
                    token.text
                ));
            }
            let mut parts = vec![token.text.clone()];
            let mut end = pos + 1;
            let next_literal = match pattern.get(index + 1) {
                Some(Elem::Word { surface }) => Some(surface.as_str()),
                _ => None,
            };
            while end < tokens.len() && (tokens[end].pos == "unknown" || tokens[end].pos == "nr") {
                if next_literal == Some(tokens[end].text.as_str()) {
                    break;
                }
                parts.push(tokens[end].text.clone());
                end += 1;
            }
            Ok(Attempt::Consumed {
                new_pos: end,
                span: Span::new(pos, end),
                entity: Some(parts.join("")),
                binding: None,
            })
        }
        Elem::Pos { pos: expected } => {
            if token.pos != *expected {
                return Err(format!(
                    "第{}要素期望词性「{}」，实得「{}」（「{}」）",
                    index + 1,
                    expected,
                    token.pos,
                    token.text
                ));
            }
            let entity = if matches!(expected.as_str(), "n" | "nr" | "ns") {
                Some(token.text.clone())
            } else {
                None
            };
            Ok(Attempt::Consumed {
                new_pos: pos + 1,
                span: Span::new(pos, pos + 1),
                entity,
                binding: None,
            })
        }
        Elem::Word { surface } => {
            let current = token.text.as_str();
            if current == surface {
                return Ok(Attempt::Consumed {
                    new_pos: pos + 1,
                    span: Span::new(pos, pos + 1),
                    entity: None,
                    binding: None,
                });
            }
            // 逐字跨词：词面被切成单字 token 序列时按字推进
            if surface.len() > 1 {
                let chars: Vec<char> = surface.chars().collect();
                let mut temp = pos;
                let mut matched = 0usize;
                for expected_char in &chars {
                    if temp < tokens.len() {
                        let word = tokens[temp].text.as_str();
                        if word.len() == expected_char.len_utf8()
                            && word.starts_with(*expected_char)
                        {
                            matched += 1;
                            temp += 1;
                            continue;
                        }
                    }
                    break;
                }
                if matched == chars.len() {
                    return Ok(Attempt::Consumed {
                        new_pos: temp,
                        span: Span::new(pos, temp),
                        entity: None,
                        binding: None,
                    });
                }
            }
            if current.contains(surface.as_str()) || surface.contains(current) {
                return Ok(Attempt::Consumed {
                    new_pos: pos + 1,
                    span: Span::new(pos, pos + 1),
                    entity: None,
                    binding: None,
                });
            }
            Err(format!(
                "第{}要素期望词「{}」，实得「{}」",
                index + 1,
                surface,
                current
            ))
        }
        Elem::Slot { name, sort, hints } => {
            if sort_matches(sort, hints, token) {
                Ok(Attempt::Consumed {
                    new_pos: pos + 1,
                    span: Span::new(pos, pos + 1),
                    entity: Some(token.text.clone()),
                    binding: Some((name.clone(), token.text.clone())),
                })
            } else {
                Err(format!(
                    "第{}要素词槽「{}」排序不匹配（期望{}，实得词性「{}」（「{}」））",
                    index + 1,
                    name,
                    sort,
                    token.pos,
                    token.text
                ))
            }
        }
        Elem::Any => Ok(Attempt::Consumed {
            new_pos: pos + 1,
            span: Span::new(pos, pos + 1),
            entity: None,
            binding: None,
        }),
    }
}

/// 对一条模式做锚定匹配：成功给解，失败给「第几要素、期望什么」。
fn match_entry(
    entry: &PatternEntry,
    tokens: &[crate::nlp::TaggedWord],
) -> Result<PatternMatch, String> {
    let elems = &entry.elems;
    if elems.is_empty() {
        // 与既有语义一致：空模式命中空解
        return Ok(PatternMatch {
            entry: entry.clone(),
            bindings: BindingSet::new(),
            entities: Vec::new(),
            entity_spans: Vec::new(),
            elem_spans: Vec::new(),
            span: Span::new(0, 0),
        });
    }

    let mut pos = 0usize;
    let mut bindings = BindingSet::new();
    let mut entities = Vec::new();
    let mut entity_spans = Vec::new();
    let mut elem_spans: Vec<Option<Span>> = Vec::with_capacity(elems.len());
    let mut span_start: Option<usize> = None;
    let mut span_end: usize = 0;
    let mut entity_counter = 0usize;

    for (index, elem) in elems.iter().enumerate() {
        while pos < tokens.len() && is_skippable(tokens[pos].text.as_str()) {
            pos += 1;
        }
        if pos >= tokens.len() {
            if matches!(elem, Elem::Optional(_)) {
                elem_spans.push(None);
                continue;
            }
            return Err(format!(
                "第{}要素「{}」未匹配：输入已耗尽",
                index + 1,
                elem.render()
            ));
        }

        match attempt_elem(elem, elems, index, tokens, pos)? {
            Attempt::Absent => elem_spans.push(None),
            Attempt::Consumed {
                new_pos,
                span,
                entity,
                binding,
            } => {
                if let Some(entity) = entity {
                    entity_counter += 1;
                    let key = format!("entity.{}", entity_counter);
                    bindings.bind(&key, &entity);
                    entities.push(entity);
                    entity_spans.push(span);
                }
                if let Some((name, value)) = binding {
                    bindings.bind(name, &value);
                }
                span_start.get_or_insert(span.start);
                span_end = span_end.max(span.end);
                pos = new_pos;
                elem_spans.push(Some(span));
            }
        }
    }

    Ok(PatternMatch {
        entry: entry.clone(),
        bindings,
        entities,
        entity_spans,
        elem_spans,
        span: Span::new(span_start.unwrap_or(0), span_end),
    })
}

/// 首要素预筛：只判断「首要素是否必然失配」，成功集必须是完整匹配的超集。
///
/// 复用 [`attempt_elem`] 的匹配语义（含人名合并、逐字跨词、包含匹配），
/// 因此不会把本可命中的条目误跳过；可选首要素一律保守放行。
fn first_elem_may_miss(entry: &PatternEntry, tokens: &[crate::nlp::TaggedWord]) -> bool {
    let Some(first) = entry.elems.first() else {
        return false;
    };
    let mut pos = 0usize;
    while pos < tokens.len() && is_skippable(tokens[pos].text.as_str()) {
        pos += 1;
    }
    if pos >= tokens.len() {
        return !matches!(first, Elem::Optional(_));
    }
    attempt_elem(first, &entry.elems, 0, tokens, pos).is_err()
}

// ---------------------------------------------------------------------------
// 模式库
// ---------------------------------------------------------------------------

/// 模式库：配置查询 + NLU 意图 + 教学模式的统一登记处。
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PatternLibrary {
    entries: Vec<PatternEntry>,
}

impl PatternLibrary {
    pub fn new() -> Self {
        Self::default()
    }

    /// 由一组 `[[prolog_rules]]` 编译（全部参与路由）。
    pub fn from_prolog_rules(rules: &[PrologRuleConfig]) -> Self {
        Self {
            entries: rules.iter().map(PatternEntry::from_prolog_rule).collect(),
        }
    }

    /// 全量编译：配置查询（可路由）+ NLU 默认意图（同库展示，不路由）。
    pub fn from_config(config: &SystemConfig) -> Self {
        let mut library = Self::from_prolog_rules(&config.prolog_rules);
        for domain in crate::nlu::default_domains() {
            for intent in &domain.intents {
                // NLU 意图与同名配置规则对齐时，复用配置的 query_type，
                // 作为主路由未命中后的回退（route=false 仍不抢配置匹配）。
                let aligned_type = config
                    .prolog_rules
                    .iter()
                    .find(|rule| rule.name == intent.name)
                    .map(|rule| rule.query_type.clone());
                for pattern in &intent.patterns {
                    let mut entry = PatternEntry::from_nlu_intent(&domain, intent, pattern);
                    if let Some(query_type) = &aligned_type {
                        entry.query_type = query_type.clone();
                        entry.fallback = true;
                    }
                    library.entries.push(entry);
                }
            }
        }
        library
    }

    pub fn entries(&self) -> &[PatternEntry] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn route_count(&self) -> usize {
        self.entries.iter().filter(|entry| entry.route).count()
    }

    /// 未对齐到可执行路由的 NLU 意图数（回退条目单独计入 [`Self::fallback_count`]）。
    pub fn nlu_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| !entry.route && !entry.fallback)
            .count()
    }

    pub fn fallback_count(&self) -> usize {
        self.entries.iter().filter(|entry| entry.fallback).count()
    }

    /// 按来源前缀计数（`taught#` / `config.toml#` / `nlu#` …）。
    pub fn count_with_source_prefix(&self, prefix: &str) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.source.starts_with(prefix))
            .count()
    }

    /// 登记一条模式：同 id（IR+路由+来源）原地替换，返回是否为新增。
    pub fn add(&mut self, entry: PatternEntry) -> bool {
        let id = entry.id();
        if let Some(slot) = self.entries.iter_mut().find(|existing| existing.id() == id) {
            *slot = entry;
            false
        } else {
            self.entries.push(entry);
            true
        }
    }

    /// 用户确认一条案例候选：把 `route=false` 的临时骨架转正为可路由模式。
    ///
    /// 转正后的来源为 `taught#案例#<经验 id>`，因此与显式教学一样：
    /// - 参与后续路由与匹配解释；
    /// - 随模式账本落图、跨会话恢复；
    /// - 可被 [`PatternLibrary::forget`] 遗忘。
    ///
    /// 候选本身不可路由（避免未经确认的泛化直接答题）；重复确认同一案例
    /// 幂等（同 id 原地替换）。
    pub fn accept_case(&mut self, suggestion: &CaseSuggestion) -> Result<PatternEntry, String> {
        if suggestion.pattern.route {
            return Err("案例候选已经处于可路由状态".to_string());
        }
        if suggestion.pattern.elems.is_empty() {
            return Err("案例候选没有模式要素".to_string());
        }
        if suggestion.pattern.query_type.trim().is_empty() {
            return Err("案例候选缺少路由类型".to_string());
        }
        let mut entry = suggestion.pattern.clone();
        entry.route = true;
        entry.source = format!("taught#案例#{}", suggestion.episode.short_id());
        entry.name = entry
            .name
            .strip_prefix("案例候选：")
            .map(|name| format!("案例确认：{}", name))
            .unwrap_or_else(|| format!("案例确认：{}", entry.name));
        self.add(entry.clone());
        Ok(entry)
    }

    /// 教一条模式（运行时即刻参与匹配与路由；来源 `taught#会话`）。
    pub fn teach(
        &mut self,
        name: &str,
        pattern: &str,
        query_type: &str,
        priority: i32,
    ) -> Result<PatternEntry, String> {
        self.teach_with_guard(name, pattern, query_type, priority, None)
    }

    /// 教学并附带上下文守卫（省略/指代模式只在相应对话状态下成立）。
    pub fn teach_with_guard(
        &mut self,
        name: &str,
        pattern: &str,
        query_type: &str,
        priority: i32,
        guard: Option<ContextGuard>,
    ) -> Result<PatternEntry, String> {
        if name.trim().is_empty() {
            return Err("模式名称不能为空".to_string());
        }
        if query_type.trim().is_empty() {
            return Err("路由类型（query_type）不能为空".to_string());
        }
        let elems = parse_teach_pattern(pattern)?;
        let mut entry = PatternEntry::new(
            name.trim(),
            "taught#会话",
            query_type.trim(),
            priority,
            true,
            elems,
        );
        if let Some(guard) = guard {
            entry = entry.with_guard(guard);
        }
        self.add(entry.clone());
        Ok(entry)
    }

    /// 忘掉一条教学模式（按 id 前缀或精确名；只删 `taught#` 来源）。
    pub fn forget(&mut self, id_or_name: &str) -> Option<PatternEntry> {
        let key = id_or_name.trim();
        if key.is_empty() {
            return None;
        }
        let position = self.entries.iter().position(|entry| {
            entry.source.starts_with("taught#")
                && (entry.id().starts_with(key) || entry.name == key)
        })?;
        Some(self.entries.remove(position))
    }

    /// 并入外部载入的条目（教学与归纳来源；配置/NLU 每次启动重建）。
    pub fn merge_taught(&mut self, entries: &[PatternEntry]) -> usize {
        let mut merged = 0;
        for entry in entries {
            if (entry.source.starts_with("taught#") || entry.source.starts_with("induced#"))
                && self.add(entry.clone())
            {
                merged += 1;
            }
        }
        merged
    }

    /// 归纳一条模式（来源 `induced#<seed>`；参与路由）。
    ///
    /// 与 [`PatternLibrary::teach`] 同一落库路径，只是来源不同：
    /// 归纳出的模式来自对话诱导/最小泛化，而不是显式教学。
    pub fn induce(
        &mut self,
        name: &str,
        pattern: &str,
        query_type: &str,
        priority: i32,
        seed: &str,
    ) -> Result<PatternEntry, String> {
        if name.trim().is_empty() {
            return Err("模式名称不能为空".to_string());
        }
        if query_type.trim().is_empty() {
            return Err("路由类型（query_type）不能为空".to_string());
        }
        let elems = parse_teach_pattern(pattern)?;
        let entry = PatternEntry::new(
            name.trim(),
            format!("induced#{}", seed.trim()),
            query_type.trim(),
            priority,
            true,
            elems,
        );
        self.add(entry.clone());
        Ok(entry)
    }

    /// 评估序：优先级降序 + 配置序（稳定排序，与既有路由一致）。
    fn evaluation_order(&self) -> Vec<usize> {
        let mut order: Vec<usize> = (0..self.entries.len()).collect();
        order.sort_by(|a, b| self.entries[*b].priority.cmp(&self.entries[*a].priority));
        order
    }

    /// 全量评估：胜者（第一条可路由的命中）+ 命中榜 + 失败原因榜。
    pub fn rank(&self, tokens: &[crate::nlp::TaggedWord]) -> MatchReport {
        self.rank_with_context(tokens, &PatternContext::default())
    }

    /// 带对话上下文的全量评估：守卫不满足的模式进入失败榜并给出原因。
    pub fn rank_with_context(
        &self,
        tokens: &[crate::nlp::TaggedWord],
        context: &PatternContext,
    ) -> MatchReport {
        let mut winner: Option<PatternMatch> = None;
        let mut matched = Vec::new();
        let mut failed = Vec::new();
        let mut rank = 0usize;

        for index in self.evaluation_order() {
            let entry = &self.entries[index];
            if !entry.context_satisfied(context) {
                let guard = entry
                    .guard
                    .as_ref()
                    .map(ContextGuard::render)
                    .unwrap_or_default();
                failed.push(CandidateOutcome {
                    id: entry.id(),
                    name: entry.name.clone(),
                    query_type: entry.query_type.clone(),
                    source: entry.source.clone(),
                    priority: entry.priority,
                    route: entry.route,
                    status: CandidateStatus::Failed {
                        reason: format!("上下文守卫不满足（{}）", guard),
                    },
                });
                continue;
            }
            match match_entry(entry, tokens) {
                Ok(m) => {
                    if winner.is_none() && entry.route {
                        winner = Some(m.clone());
                    }
                    matched.push(CandidateOutcome {
                        id: entry.id(),
                        name: entry.name.clone(),
                        query_type: entry.query_type.clone(),
                        source: entry.source.clone(),
                        priority: entry.priority,
                        route: entry.route,
                        status: CandidateStatus::Matched {
                            rank,
                            specificity: entry.specificity(),
                            span: m.span,
                            bindings: m.bindings.len(),
                        },
                    });
                    rank += 1;
                }
                Err(reason) => failed.push(CandidateOutcome {
                    id: entry.id(),
                    name: entry.name.clone(),
                    query_type: entry.query_type.clone(),
                    source: entry.source.clone(),
                    priority: entry.priority,
                    route: entry.route,
                    status: CandidateStatus::Failed { reason },
                }),
            }
        }

        MatchReport {
            winner,
            matched,
            failed,
        }
    }

    /// 取第一条可路由的命中（查询路径热路径；不必算完整报告）。
    pub fn match_route(&self, tokens: &[crate::nlp::TaggedWord]) -> Option<PatternMatch> {
        self.match_route_with_context(tokens, &PatternContext::default())
    }

    /// 带上下文守卫的路由：守卫不满足的模式在本次路由中被跳过。
    pub fn match_route_with_context(
        &self,
        tokens: &[crate::nlp::TaggedWord],
        context: &PatternContext,
    ) -> Option<PatternMatch> {
        self.scan_with_context(tokens, context, false).winner
    }

    /// 快速路由并返回扫描统计（首要素预筛省掉了多少次完整匹配）。
    pub fn scan_route_with_context(
        &self,
        tokens: &[crate::nlp::TaggedWord],
        context: &PatternContext,
    ) -> RouteScan {
        self.scan_with_context(tokens, context, false)
    }

    /// NLU 回退路由：仅当主路由无命中时使用，按同样优先级与守卫评估。
    pub fn match_fallback(&self, tokens: &[crate::nlp::TaggedWord]) -> Option<PatternMatch> {
        self.match_fallback_with_context(tokens, &PatternContext::default())
    }

    /// 带上下文守卫的 NLU 回退路由。
    pub fn match_fallback_with_context(
        &self,
        tokens: &[crate::nlp::TaggedWord],
        context: &PatternContext,
    ) -> Option<PatternMatch> {
        self.scan_with_context(tokens, context, true).winner
    }

    /// 主路由/回退路由共用的扫描器：保留评估序，首要素必然失配的直接跳过。
    fn scan_with_context(
        &self,
        tokens: &[crate::nlp::TaggedWord],
        context: &PatternContext,
        fallback: bool,
    ) -> RouteScan {
        let mut evaluated = 0usize;
        let mut skipped = 0usize;
        for index in self.evaluation_order() {
            let entry = &self.entries[index];
            let eligible = if fallback {
                entry.fallback
            } else {
                entry.route
            };
            if !eligible || !entry.context_satisfied(context) {
                continue;
            }
            if first_elem_may_miss(entry, tokens) {
                skipped += 1;
                continue;
            }
            evaluated += 1;
            if let Ok(m) = match_entry(entry, tokens) {
                return RouteScan {
                    winner: Some(m),
                    evaluated,
                    skipped,
                };
            }
        }
        RouteScan {
            winner: None,
            evaluated,
            skipped,
        }
    }

    /// 与既有 `nlp::analyze_query_intent` 同构的路由口。
    pub fn route(&self, tokens: &[crate::nlp::TaggedWord]) -> Option<(String, Vec<String>)> {
        self.match_route(tokens)
            .map(|m| (m.entry.query_type.clone(), m.entities.clone()))
    }

    /// 带上下文守卫的路由口。
    pub fn route_with_context(
        &self,
        tokens: &[crate::nlp::TaggedWord],
        context: &PatternContext,
    ) -> Option<(String, Vec<String>)> {
        self.match_route_with_context(tokens, context)
            .map(|m| (m.entry.query_type.clone(), m.entities.clone()))
    }

    /// 按 id 前缀或名称查条目。
    pub fn get(&self, id_or_name: &str) -> Option<&PatternEntry> {
        let key = id_or_name.trim();
        self.entries
            .iter()
            .find(|entry| entry.id().starts_with(key) || entry.name == key)
    }
}

// ---------------------------------------------------------------------------
// 报告渲染
// ---------------------------------------------------------------------------

/// 模式库清单 + 静态体检的渲染。
pub fn render_library_report(library: &PatternLibrary) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    let taught = library.count_with_source_prefix("taught#");
    let _ = writeln!(
        out,
        "模式库：{} 条（可路由查询 {} + NLU 回退 {} + NLU 意图 {} + 教学 {}）",
        library.len(),
        library.route_count(),
        library.fallback_count(),
        library.nlu_count(),
        taught
    );
    let _ = writeln!(
        out,
        " {:<9} {:>4} {:<22} {:<12} {}",
        "#id", "优先", "路由", "名称", "模式"
    );
    for entry in library.entries() {
        let guard = entry
            .guard
            .as_ref()
            .map(|guard| format!("  [守卫：{}]", guard.render()))
            .unwrap_or_default();
        let _ = writeln!(
            out,
            " {:<9} {:>4} {:<22} {:<12} {}{}",
            entry.short_id(),
            entry.priority,
            entry.query_type,
            entry.name,
            entry.render(),
            guard
        );
    }
    let findings = lint(library);
    if findings.is_empty() {
        let _ = writeln!(out, "静态体检：无重复、遮蔽或冲突。");
    } else {
        let _ = writeln!(out, "静态体检：{} 项", findings.len());
        for finding in &findings {
            let _ = writeln!(out, " - {}", finding.render());
        }
    }
    out
}

/// 模式关系报告：泛化/冗余/冲突的图边清单 + 一条可复现的图体检查询 + 端点 id。
///
/// 与 [`render_library_report`] 展示同一份体检结果，但以「模式命名图里的边」
/// 为主视角：每条关系都是可 SPARQL 查询的数据，末尾附上计数查询与端点 id，
/// 便于把结论复制到「SPARQL：…」里复核。
pub fn render_relation_report(library: &PatternLibrary) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    let edges = relation_edges(library.entries());
    let counts = relation_counts(&edges);
    let generalizes = counts.get("泛化").copied().unwrap_or(0);
    let redundant = counts.get("冗余").copied().unwrap_or(0);
    let conflicts = counts.get("冲突").copied().unwrap_or(0);
    let _ = writeln!(
        out,
        "模式关系图 {}：{} 条边（泛化 {} + 冗余 {} + 冲突 {}）",
        PATTERNS_GRAPH,
        edges.len(),
        generalizes,
        redundant,
        conflicts
    );
    if edges.is_empty() {
        let _ = writeln!(out, "（无关系边：模式间无包含、重复或冲突）");
    } else {
        let _ = writeln!(
            out,
            " {:<6} {:<9} {:<9} {}",
            "关系", "主语#id", "宾语#id", "端点"
        );
        for edge in &edges {
            let _ = writeln!(
                out,
                " {:<6} {:<9} {:<9} {}",
                edge.kind.label(),
                &edge.subject_id.chars().take(8).collect::<String>(),
                &edge.object_id.chars().take(8).collect::<String>(),
                edge.render()
            );
        }
    }
    let _ = writeln!(out, "图体检查询（直接执行即得各关系计数）：");
    let _ = writeln!(out, "  {}", relation_count_sparql());
    out
}

/// 匹配报告渲染：胜者带绑定与区间，命中未胜带差额，失败带首个失配。
pub fn render_match_report(input: &str, report: &MatchReport) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    let _ = writeln!(out, "模式匹配「{}」：", input);

    match &report.winner {
        Some(winner) => {
            let _ = writeln!(
                out,
                " 胜者 {}（#{}）优先 {} → 路由 {}，覆盖 {}，特化度 {:.2}",
                winner.entry.name,
                winner.entry.short_id(),
                winner.entry.priority,
                winner.entry.query_type,
                winner.span,
                winner.entry.specificity()
            );
            if let Some(guard) = &winner.entry.guard {
                let _ = writeln!(out, " 上下文守卫：{}", guard.render());
            }
            let bindings: Vec<String> = winner
                .bindings
                .iter()
                .filter(|(_, term)| term.is_bound())
                .map(|(name, term)| match entity_span_for(&winner, name) {
                    Some(span) => format!("{}={}{}", name, term, span),
                    None => format!("{}={}", name, term),
                })
                .collect();
            if bindings.is_empty() {
                let _ = writeln!(out, " 绑定：（无槽位）");
            } else {
                let _ = writeln!(out, " 绑定：{}", bindings.join(" "));
            }
        }
        None => {
            let _ = writeln!(out, " 无可路由命中（教学模式或配置模式均未匹配）。");
        }
    }

    let others: Vec<&CandidateOutcome> = report
        .matched
        .iter()
        .filter(|outcome| report.winner.as_ref().map(|w| w.entry.id()) != Some(outcome.id.clone()))
        .collect();
    if !others.is_empty() {
        let _ = writeln!(out, " 命中未胜（{}）：", others.len());
        for outcome in others.iter().take(5) {
            if let CandidateStatus::Matched {
                rank,
                specificity,
                span,
                ..
            } = &outcome.status
            {
                let _ = writeln!(
                    out,
                    "  {}. {}（#{}）优先 {} → {}，{}，特化度 {:.2}{}",
                    rank + 1,
                    outcome.name,
                    &outcome.id.chars().take(8).collect::<String>(),
                    outcome.priority,
                    outcome.query_type,
                    span,
                    specificity,
                    if outcome.route {
                        ""
                    } else {
                        "（意图展示，不路由）"
                    }
                );
            }
        }
    }

    if !report.failed.is_empty() {
        let _ = writeln!(out, " 未命中（{}，前 3 条原因）：", report.failed.len());
        for outcome in report.failed.iter().take(3) {
            if let CandidateStatus::Failed { reason } = &outcome.status {
                let _ = writeln!(
                    out,
                    "  - {}（#{}）：{}",
                    outcome.name,
                    &outcome.id.chars().take(8).collect::<String>(),
                    reason
                );
            }
        }
    }

    for (a, b, span) in report.conflicts() {
        let _ = writeln!(
            out,
            " 冲突：「{}」与「{}」同覆盖 {} 但路由不同。",
            a, b, span
        );
    }
    out
}

/// 绑定名 → 实体区间（展示绑定的区间证据）。
fn entity_span_for<'a>(m: &'a PatternMatch, name: &str) -> Option<&'a Span> {
    let index: usize = name.strip_prefix("entity.")?.parse().ok()?;
    m.entity_spans.get(index.checked_sub(1)?)
}

/// 模式经验台账的文本报告（总量 + 高频 + 最近）。
pub fn render_episode_report(log: &PatternEpisodeLog) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    let total_support: u32 = log.entries().iter().map(|episode| episode.count).sum();
    let _ = writeln!(
        out,
        "模式经验：{} 条（命中 {} + 未命中 {}），容量 {}，累计支持度 {}。",
        log.len(),
        log.matched_count(),
        log.unmatched_count(),
        log.capacity(),
        total_support
    );
    if log.is_empty() {
        let _ = writeln!(out, "（还没有经验：成功或失败的解析都会留下可检索的记录）");
        return out;
    }
    let mut frequent: Vec<&PatternEpisode> = log.entries().iter().collect();
    frequent.sort_by(|a, b| {
        b.count
            .cmp(&a.count)
            .then_with(|| b.last_turn.cmp(&a.last_turn))
            .then_with(|| a.id().cmp(&b.id()))
    });
    let _ = writeln!(out, " 高频经验：");
    for episode in frequent.iter().take(5) {
        let target = episode
            .winner_name
            .as_deref()
            .map(|name| {
                format!(
                    "{} → {}",
                    name,
                    episode.query_type.as_deref().unwrap_or("")
                )
            })
            .unwrap_or_else(|| "（无可路由命中）".to_string());
        let context = episode
            .context_note()
            .map(|note| format!("，{}", note))
            .unwrap_or_default();
        let feedback = if episode.acceptances > 0 || episode.rejections > 0 {
            format!(
                "，确认×{} 否定×{}",
                episode.acceptances, episode.rejections
            )
        } else {
            String::new()
        };
        let _ = writeln!(
            out,
            "  [{}] {} ×{} 第{}轮{}{}：{}",
            episode.short_id(),
            target,
            episode.count,
            episode.last_turn,
            context,
            feedback,
            episode.input
        );
    }
    let mut recent: Vec<&PatternEpisode> = log.entries().iter().collect();
    recent.sort_by(|a, b| {
        b.last_turn
            .cmp(&a.last_turn)
            .then_with(|| a.id().cmp(&b.id()))
    });
    let _ = writeln!(out, " 最近经验：");
    for episode in recent.iter().take(5) {
        let context = episode
            .context_note()
            .map(|note| format!("，{}", note))
            .unwrap_or_default();
        let _ = writeln!(
            out,
            "  [{}] 第{}轮 {}{}：{}",
            episode.outcome.label(),
            episode.last_turn,
            episode.short_id(),
            context,
            episode.input
        );
    }
    let _ = writeln!(
        out,
        "（可用「忘经验：<短 id 或原句>」删除单条经验，撤销误确认/误否定。）"
    );
    out
}

/// 候选骨架的文本报告：相似经验 + 候选模式 + 显式激活方式。
pub fn render_case_suggestion(input: &str, suggestion: &CaseSuggestion) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    let episode = &suggestion.episode;
    let _ = writeln!(
        out,
        "模式联想「{}」：相似经验 {}（{:.2}，共享：{}）",
        input,
        episode.input,
        suggestion.score,
        if suggestion.shared.is_empty() {
            "无".to_string()
        } else {
            suggestion.shared.join("、")
        }
    );
    let _ = writeln!(
        out,
        " 经验命中：{} → {}（第{}轮，×{}）",
        episode.winner_name.as_deref().unwrap_or("未知模式"),
        episode.query_type.as_deref().unwrap_or("未知路由"),
        episode.last_turn,
        episode.count
    );
    if episode.acceptances > 0 || episode.rejections > 0 {
        let _ = writeln!(
            out,
            " 用户反馈：确认 ×{}，否定 ×{}（净否定的骨架不再主动推荐）",
            episode.acceptances, episode.rejections
        );
    }
    if let Some(context) = episode.context_note() {
        let _ = writeln!(out, " 发生上下文：{}", context);
    }
    let _ = writeln!(out, " 候选模式：{}", suggestion.pattern.render());
    let guard = suggestion
        .pattern
        .guard
        .as_ref()
        .map(|guard| format!("，守卫 {}", guard.render()))
        .unwrap_or_default();
    let _ = writeln!(
        out,
        " 候选路由：{}（来源 {}，未激活{}）",
        suggestion.pattern.query_type,
        suggestion.pattern.source,
        guard
    );
    let _ = writeln!(
        out,
        " 确认方式：确认联想：<句子>（或 教模式：{} | {} | {} | {}）",
        suggestion.pattern.name,
        suggestion.pattern.render(),
        suggestion.pattern.query_type,
        suggestion.pattern.priority
    );
    let _ = writeln!(
        out,
        " 边界：候选只用于确认，不直接回答；确认后才进入可路由模式库。"
    );
    let _ = writeln!(out, " 口语确认：直接回答「是」或「不是」即可。");
    out
}

// ---------------------------------------------------------------------------
// 静态体检：重复 / 遮蔽 / 冲突
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LintKind {
    /// 同一模式语言（IR）+ 同路由，两处定义。
    Redundant,
    /// 同一模式语言（IR）但路由不同。
    Conflict,
    /// 高位模式覆盖低位模式的语言，低位永不胜出。
    Shadowed,
}

/// 单要素包含：`a` 的语言 ⊇ `b` 的语言（保守近似，按位对齐）。
fn elem_subsumes(a: &Elem, b: &Elem) -> bool {
    match (a, b) {
        (Elem::Any, _) => true,
        // 「任意」只被「任意」覆盖
        (_, Elem::Any) => false,
        (Elem::Optional(inner), _) => elem_subsumes(inner, b),
        (Elem::Pos { pos: pa }, Elem::Pos { pos: pb }) => pa == pb,
        // 词性/词面接受集互不可判，保守不覆盖
        (Elem::Pos { .. }, Elem::Word { .. }) | (Elem::Pos { .. }, Elem::Slot { .. }) => false,
        (Elem::Word { surface: wa }, Elem::Word { surface: wb }) => wa == wb,
        (Elem::Word { .. }, Elem::Pos { .. }) | (Elem::Word { .. }, Elem::Slot { .. }) => false,
        (Elem::Slot { sort: sa, .. }, Elem::Pos { pos }) => sort_pos_set(sa)
            .map(|set| set.contains(&pos.as_str()))
            .unwrap_or(false),
        (Elem::Slot { sort: sa, .. }, Elem::Slot { sort: sb, .. }) => sa == sb,
        (Elem::Slot { .. }, Elem::Word { .. }) => false,
        // b 可缺席而 a 不可，单步对齐下不构成覆盖
        (_, Elem::Optional(_)) => false,
    }
}

/// 槽位排序 → 词性集（保守估计；供包含判定）。
fn sort_pos_set(sort: &str) -> Option<&'static [&'static str]> {
    match sort {
        "城市" | "地点" => Some(&["ns", "n"]),
        "日期" | "时间" => Some(&["m"]),
        "人名" => Some(&["nr"]),
        "数量" => Some(&["m"]),
        "实体" => Some(&["nr", "ns", "n"]),
        "属性" => Some(&["n", "nz", "vn"]),
        "*" | "" => Some(&["r", "n", "nr", "ns", "m", "q", "v", "u", "f", "a", "d"]),
        _ => None,
    }
}

/// 模式 A 是否（按位对齐地）覆盖模式 B 的语言。
pub fn subsumes(a: &PatternEntry, b: &PatternEntry) -> bool {
    if a.elems.len() > b.elems.len() {
        return false;
    }
    a.elems
        .iter()
        .enumerate()
        .all(|(i, elem)| elem_subsumes(elem, &b.elems[i]))
        && a.elems != b.elems
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LintFinding {
    pub kind: LintKind,
    pub subject_id: String,
    pub subject_name: String,
    pub object_id: String,
    pub object_name: String,
    pub detail: String,
}

impl LintFinding {
    pub fn render(&self) -> String {
        let kind = match self.kind {
            LintKind::Redundant => "冗余",
            LintKind::Conflict => "冲突",
            LintKind::Shadowed => "遮蔽",
        };
        format!(
            "[{}] {}（#{}）↔ {}（#{}）：{}",
            kind,
            self.subject_name,
            &self.subject_id.chars().take(8).collect::<String>(),
            self.object_name,
            &self.object_id.chars().take(8).collect::<String>(),
            self.detail
        )
    }
}

/// 静态体检：按评估序两两检查重复/冲突/遮蔽（只看参与条目=全部条目）。
pub fn lint(library: &PatternLibrary) -> Vec<LintFinding> {
    let mut findings = Vec::new();
    let order = library.evaluation_order();
    for (pos, &a_index) in order.iter().enumerate() {
        let a = &library.entries[a_index];
        for &b_index in order.iter().skip(pos + 1) {
            let b = &library.entries[b_index];
            if a.ir_fingerprint() == b.ir_fingerprint() {
                let kind = if a.query_type == b.query_type {
                    LintKind::Redundant
                } else {
                    LintKind::Conflict
                };
                let detail = if kind == LintKind::Redundant {
                    format!("IR 完全相同且同路由（{}），两条定义只需其一", a.query_type)
                } else {
                    format!(
                        "IR 完全相同但路由不同：{} vs {}",
                        a.query_type, b.query_type
                    )
                };
                findings.push(LintFinding {
                    kind,
                    subject_id: a.id(),
                    subject_name: a.name.clone(),
                    object_id: b.id(),
                    object_name: b.name.clone(),
                    detail,
                });
                continue;
            }
            if subsumes(a, b) {
                let detail = if a.query_type == b.query_type {
                    format!(
                        "「{}」覆盖「{}」的语言（优先 {} ≥ {}），后者永不胜出",
                        a.name, b.name, a.priority, b.priority
                    )
                } else {
                    format!(
                        "「{}」（{}）命中时「{}」（{}）被遮蔽，路由到 {}",
                        a.name, a.query_type, b.name, b.query_type, a.query_type
                    )
                };
                findings.push(LintFinding {
                    kind: LintKind::Shadowed,
                    subject_id: a.id(),
                    subject_name: a.name.clone(),
                    object_id: b.id(),
                    object_name: b.name.clone(),
                    detail,
                });
            }
        }
    }
    findings
}

// ---------------------------------------------------------------------------
// 模式关系：体检结果落成图边（关系本身是可查询的数据）
// ---------------------------------------------------------------------------

/// 模式之间的关系：静态体检结论的图视角。
///
/// 落图后这些关系就是 `kg:graphs/patterns` 里的普通三元组，
/// 「哪些模式互斥/冗余/互相遮蔽」由一条 SPARQL 就能问出来，
/// 不必再靠手写遍历（见 [`relation_edges`] 与 [`relation_quads`]）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationKind {
    /// 语言包含：subject 覆盖 object 的语言（评估序下 object 被遮蔽）。
    Generalizes,
    /// 同一模式语言且同路由，两处定义只需其一。
    Redundant,
    /// 同一模式语言但路由不同：命中即冲突。
    Conflicts,
}

impl RelationKind {
    pub fn label(self) -> &'static str {
        match self {
            RelationKind::Generalizes => "泛化",
            RelationKind::Redundant => "冗余",
            RelationKind::Conflicts => "冲突",
        }
    }

    /// 关系谓词 IRI（`kg:ledger/` 命名空间）。
    pub fn predicate(self) -> &'static str {
        match self {
            RelationKind::Generalizes => P_GENERALIZES,
            RelationKind::Redundant => P_REDUNDANT,
            RelationKind::Conflicts => P_CONFLICTS,
        }
    }

    /// 是否对称（冗余/冲突双向各一条；泛化有方向）。
    pub fn is_symmetric(self) -> bool {
        !matches!(self, RelationKind::Generalizes)
    }
}

/// 一条模式关系边（端点带稳定 id 与可读名）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RelationEdge {
    pub kind: RelationKind,
    pub subject_id: String,
    pub subject_name: String,
    pub object_id: String,
    pub object_name: String,
}

impl RelationEdge {
    pub fn render(&self) -> String {
        format!(
            "{}（{}）{}（{}）",
            self.subject_name,
            &self.subject_id.chars().take(8).collect::<String>(),
            self.kind.label(),
            self.object_name
        )
    }
}

/// 由条目集计算全部模式关系（即静态体检的图视角；确定性、与顺序无关的集合）。
pub fn relation_edges(entries: &[PatternEntry]) -> Vec<RelationEdge> {
    let library = PatternLibrary {
        entries: entries.to_vec(),
    };
    lint(&library)
        .into_iter()
        .map(|finding| {
            let kind = match finding.kind {
                LintKind::Shadowed => RelationKind::Generalizes,
                LintKind::Redundant => RelationKind::Redundant,
                LintKind::Conflict => RelationKind::Conflicts,
            };
            RelationEdge {
                kind,
                subject_id: finding.subject_id,
                subject_name: finding.subject_name,
                object_id: finding.object_id,
                object_name: finding.object_name,
            }
        })
        .collect()
}

/// 按关系种类计数（图上体检摘要用）。
pub fn relation_counts(edges: &[RelationEdge]) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    for edge in edges {
        *counts.entry(edge.kind.label()).or_insert(0usize) += 1;
    }
    counts
}

/// 图上体检的 SPARQL：直接在模式命名图里数三种关系的边数。
pub fn relation_count_sparql() -> String {
    let predicates = [
        RelationKind::Generalizes.predicate(),
        RelationKind::Redundant.predicate(),
        RelationKind::Conflicts.predicate(),
    ]
    .iter()
    .map(|iri| format!("<{}>", iri))
    .collect::<Vec<_>>()
    .join(", ");
    format!(
        "SELECT ?rel (COUNT(*) AS ?n) WHERE {{ \
         GRAPH <{graph}> {{ ?subject ?rel ?object FILTER(?rel IN ({predicates})) }} \
         }} GROUP BY ?rel",
        graph = PATTERNS_GRAPH,
        predicates = predicates,
    )
}

// ---------------------------------------------------------------------------
// 模式落图：pattern IR → RDF 命名图（kg:graphs/patterns）
// ---------------------------------------------------------------------------

use oxigraph::model::{GraphName, Literal, NamedNode, Quad, Term};

/// 模式账本命名图。
pub const PATTERNS_GRAPH: &str = "http://example.org/kg/graphs/patterns";

use crate::graph_ledger::LEDGER_NS;

type LedgerResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
const PATTERN_CLASS: &str = "http://example.org/kg/ledger/Pattern";
const STAMP_CLASS: &str = "http://example.org/kg/ledger/Stamp";
const STAMP_SUBJECT: &str = "http://example.org/kg/ledger/stamp";

const P_JSON: &str = "http://example.org/kg/ledger/json";
const P_NAME: &str = "http://example.org/kg/ledger/name";
const P_QUERY_TYPE: &str = "http://example.org/kg/ledger/queryType";
const P_PRIORITY: &str = "http://example.org/kg/ledger/priority";
const P_SOURCE: &str = "http://example.org/kg/ledger/source";
const P_IR: &str = "http://example.org/kg/ledger/ir";
const P_GUARD: &str = "http://example.org/kg/ledger/contextGuard";
const P_FINGERPRINT: &str = "http://example.org/kg/ledger/fingerprint";
const P_COUNT: &str = "http://example.org/kg/ledger/count";
const P_GENERALIZES: &str = "http://example.org/kg/ledger/generalizes";
const P_REDUNDANT: &str = "http://example.org/kg/ledger/redundantWith";
const P_CONFLICTS: &str = "http://example.org/kg/ledger/conflictsWith";

/// 模式经验命名图：一次理解的经验与图谱/模式定义分开存放。
pub const EPISODES_GRAPH: &str = "http://example.org/kg/graphs/pattern-episodes";

const EPISODE_CLASS: &str = "http://example.org/kg/ledger/Episode";
const EPISODE_STAMP_SUBJECT: &str = "http://example.org/kg/ledger/episodeStamp";

const P_EPISODE_JSON: &str = "http://example.org/kg/ledger/episodeJson";
const P_EPISODE_INPUT: &str = "http://example.org/kg/ledger/input";
const P_EPISODE_OUTCOME: &str = "http://example.org/kg/ledger/outcome";
const P_EPISODE_LAST_TURN: &str = "http://example.org/kg/ledger/lastTurn";

fn named_node(iri: &str) -> LedgerResult<NamedNode> {
    Ok(NamedNode::new(iri)?)
}

fn literal_quad(
    subject: &NamedNode,
    predicate: &str,
    value: &str,
    graph: &GraphName,
) -> LedgerResult<Quad> {
    Ok(Quad::new(
        subject.clone(),
        named_node(predicate)?,
        Literal::new_simple_literal(value),
        graph.clone(),
    ))
}

/// 模式集指纹（按条目 id 升序折叠，与插入顺序无关）。
pub fn library_fingerprint(entries: &[PatternEntry]) -> String {
    let mut ids: Vec<String> = entries.iter().map(PatternEntry::id).collect();
    ids.sort();
    fnv1a_hex(ids.join("\n").as_bytes())
}

/// 生成模式账本的全部四元组（含物化戳与模式关系边）。
///
/// 模式关系（泛化/冗余/冲突）来自 [`relation_edges`]：对称关系双向各一条，
/// 于是「哪两个模式互斥」在图上是一条反向也走得通的三元组。
pub fn pattern_quads(entries: &[PatternEntry]) -> LedgerResult<Vec<Quad>> {
    let graph = GraphName::NamedNode(named_node(PATTERNS_GRAPH)?);
    let rdf_type = named_node(RDF_TYPE)?;
    let pattern_class = Term::NamedNode(named_node(PATTERN_CLASS)?);
    let edges = relation_edges(entries);
    let mut quads: Vec<Quad> = Vec::with_capacity(entries.len() * 8 + edges.len() * 2 + 3);
    for entry in entries {
        let subject = named_node(&format!("{}pattern/{}", LEDGER_NS, entry.id()))?;
        quads.push(Quad::new(
            subject.clone(),
            rdf_type.clone(),
            pattern_class.clone(),
            graph.clone(),
        ));
        quads.push(literal_quad(
            &subject,
            P_JSON,
            &serde_json::to_string(entry)?,
            &graph,
        )?);
        quads.push(literal_quad(&subject, P_NAME, &entry.name, &graph)?);
        quads.push(literal_quad(
            &subject,
            P_QUERY_TYPE,
            &entry.query_type,
            &graph,
        )?);
        quads.push(literal_quad(
            &subject,
            P_PRIORITY,
            &entry.priority.to_string(),
            &graph,
        )?);
        quads.push(literal_quad(&subject, P_SOURCE, &entry.source, &graph)?);
        quads.push(literal_quad(&subject, P_IR, &entry.render(), &graph)?);
        quads.push(literal_quad(&subject, P_FINGERPRINT, &entry.id(), &graph)?);
        if let Some(guard) = &entry.guard {
            quads.push(literal_quad(&subject, P_GUARD, &guard.render(), &graph)?);
        }
    }
    for edge in &edges {
        let subject = named_node(&format!("{}pattern/{}", LEDGER_NS, edge.subject_id))?;
        let object = named_node(&format!("{}pattern/{}", LEDGER_NS, edge.object_id))?;
        let predicate = named_node(edge.kind.predicate())?;
        quads.push(Quad::new(
            subject.clone(),
            predicate.clone(),
            object.clone(),
            graph.clone(),
        ));
        if edge.kind.is_symmetric() {
            quads.push(Quad::new(object, predicate, subject, graph.clone()));
        }
    }
    let stamp = named_node(STAMP_SUBJECT)?;
    quads.push(Quad::new(
        stamp.clone(),
        rdf_type,
        Term::NamedNode(named_node(STAMP_CLASS)?),
        graph.clone(),
    ));
    quads.push(literal_quad(
        &stamp,
        P_FINGERPRINT,
        &library_fingerprint(entries),
        &graph,
    )?);
    quads.push(literal_quad(
        &stamp,
        P_COUNT,
        &entries.len().to_string(),
        &graph,
    )?);
    Ok(quads)
}

/// 从模式命名图的四元组还原条目（按 id 升序，确定性；物化戳忽略）。
pub fn parse_pattern_graph(quads: impl Iterator<Item = Quad>) -> Vec<PatternEntry> {
    let mut by_subject: BTreeMap<String, String> = BTreeMap::new();
    for quad in quads {
        if quad.predicate.as_str() != P_JSON {
            continue;
        }
        let subject = quad.subject.to_string();
        if let Term::Literal(value) = quad.object {
            by_subject.insert(subject, value.value().to_string());
        }
    }
    by_subject
        .into_values()
        .filter_map(|json| serde_json::from_str::<PatternEntry>(&json).ok())
        .collect()
}

/// 经验集指纹（按经验 id 升序折叠，与插入顺序无关）。
pub fn episode_fingerprint(episodes: &[PatternEpisode]) -> String {
    let mut ids: Vec<String> = episodes.iter().map(PatternEpisode::id).collect();
    ids.sort();
    fnv1a_hex(ids.join("\n").as_bytes())
}

/// 生成模式经验账本的全部四元组（独立命名图 + 物化戳）。
pub fn episode_quads(episodes: &[PatternEpisode]) -> LedgerResult<Vec<Quad>> {
    let graph = GraphName::NamedNode(named_node(EPISODES_GRAPH)?);
    let rdf_type = named_node(RDF_TYPE)?;
    let episode_class = Term::NamedNode(named_node(EPISODE_CLASS)?);
    let mut quads: Vec<Quad> = Vec::with_capacity(episodes.len() * 5 + 3);
    for episode in episodes {
        let subject = named_node(&format!("{}episode/{}", LEDGER_NS, episode.id()))?;
        quads.push(Quad::new(
            subject.clone(),
            rdf_type.clone(),
            episode_class.clone(),
            graph.clone(),
        ));
        quads.push(literal_quad(
            &subject,
            P_EPISODE_JSON,
            &serde_json::to_string(episode)?,
            &graph,
        )?);
        quads.push(literal_quad(
            &subject,
            P_EPISODE_INPUT,
            &episode.input,
            &graph,
        )?);
        quads.push(literal_quad(
            &subject,
            P_EPISODE_OUTCOME,
            episode.outcome.label(),
            &graph,
        )?);
        quads.push(literal_quad(
            &subject,
            P_EPISODE_LAST_TURN,
            &episode.last_turn.to_string(),
            &graph,
        )?);
    }
    let stamp = named_node(EPISODE_STAMP_SUBJECT)?;
    quads.push(Quad::new(
        stamp.clone(),
        rdf_type,
        Term::NamedNode(named_node(STAMP_CLASS)?),
        graph.clone(),
    ));
    quads.push(literal_quad(
        &stamp,
        P_FINGERPRINT,
        &episode_fingerprint(episodes),
        &graph,
    )?);
    quads.push(literal_quad(
        &stamp,
        P_COUNT,
        &episodes.len().to_string(),
        &graph,
    )?);
    Ok(quads)
}

/// 从经验命名图还原经验（按 id 升序，确定性；物化戳忽略）。
pub fn parse_episode_graph(quads: impl Iterator<Item = Quad>) -> Vec<PatternEpisode> {
    let mut by_subject: BTreeMap<String, String> = BTreeMap::new();
    for quad in quads {
        if quad.predicate.as_str() != P_EPISODE_JSON {
            continue;
        }
        let subject = quad.subject.to_string();
        if let Term::Literal(value) = quad.object {
            by_subject.insert(subject, value.value().to_string());
        }
    }
    by_subject
        .into_values()
        .filter_map(|json| serde_json::from_str::<PatternEpisode>(&json).ok())
        .collect()
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bindings::BoundTerm;
    use crate::create_prolog_rules;
    use crate::nlp::{analyze_query_intent, preprocess, TaggedWord};

    fn tagged(pairs: &[(&str, &str)]) -> Vec<TaggedWord> {
        pairs
            .iter()
            .map(|(text, pos)| TaggedWord {
                text: text.to_string(),
                pos: pos.to_string(),
            })
            .collect()
    }

    fn sample_library() -> PatternLibrary {
        let mut library = PatternLibrary::new();
        library.add(PatternEntry::new(
            "首都查询",
            "config.toml#首都查询",
            "capital_query",
            15,
            true,
            vec![
                Elem::pos("ns"),
                Elem::word("的"),
                Elem::word("首都"),
                Elem::word("是"),
                Elem::word("哪里"),
            ],
        ));
        library.add(PatternEntry::new(
            "首都查询2",
            "config.toml#首都查询2",
            "capital_query",
            14,
            true,
            vec![
                Elem::slot("实体", "实体"),
                Elem::word("的"),
                Elem::word("首都"),
            ],
        ));
        library.add(PatternEntry::new(
            "属性查询",
            "config.toml#属性查询",
            "property_query",
            5,
            true,
            vec![Elem::pos("nr"), Elem::word("的"), Elem::pos("n")],
        ));
        library
    }

    #[test]
    fn fingerprint_is_stable_and_route_sensitive() {
        let a = PatternEntry::new("x", "config.toml#x", "q1", 1, true, vec![Elem::pos("ns")]);
        let b = PatternEntry::new("x", "config.toml#x", "q1", 9, true, vec![Elem::pos("ns")]);
        // IR+路由+来源构成身份：优先级变化不改身份（同 id 原地替换）
        assert_eq!(a.id(), b.id());
        let c = PatternEntry::new("x", "config.toml#x", "q2", 1, true, vec![Elem::pos("ns")]);
        assert_ne!(a.id(), c.id());
        // IR 指纹只看 IR
        assert_eq!(a.ir_fingerprint(), c.ir_fingerprint());
        // 序列不同 → 指纹不同
        let d = PatternEntry::new(
            "x",
            "config.toml#x",
            "q1",
            1,
            true,
            vec![Elem::pos("ns"), Elem::word("首都")],
        );
        assert_ne!(a.ir_fingerprint(), d.ir_fingerprint());
        // 指纹稳定（可逐字节断言）
        assert_eq!(a.ir_fingerprint().len(), 16);
        assert_eq!(
            a.ir_fingerprint(),
            PatternEntry::new("y", "s", "q1", 0, true, vec![Elem::pos("ns")]).ir_fingerprint()
        );
    }

    #[test]
    fn config_and_teach_tokens_parse() {
        assert_eq!(parse_config_token("nr"), Elem::pos("nr"));
        assert_eq!(parse_config_token("首都"), Elem::word("首都"));
        assert_eq!(parse_config_token("*"), Elem::Any);
        assert_eq!(
            parse_teach_token("{城市:城市}").unwrap(),
            Elem::Slot {
                name: "城市".to_string(),
                sort: "城市".to_string(),
                hints: Vec::new(),
            }
        );
        assert_eq!(parse_teach_token("_").unwrap(), Elem::Any);
        assert!(parse_teach_token("{}").is_err());
        let elems = parse_teach_pattern("ns 的 首都").unwrap();
        assert_eq!(elems.len(), 3);
        assert!(parse_teach_pattern("   ").is_err());
    }

    #[test]
    fn matcher_binds_entities_with_spans() {
        let library = sample_library();
        let tokens = tagged(&[
            ("法国", "ns"),
            ("的", "u"),
            ("首都", "n"),
            ("是", "v"),
            ("哪里", "r"),
            ("？", "w"),
        ]);
        let m = library.match_route(&tokens).expect("应命中首都查询");
        assert_eq!(m.entry.name, "首都查询");
        assert_eq!(m.entities, vec!["法国"]);
        assert_eq!(
            m.bindings.get("entity.1").map(|t| t.render_raw()),
            Some(Some("法国".to_string()))
        );
        assert_eq!(m.span, Span::new(0, 5));
        assert_eq!(m.entity_spans, vec![Span::new(0, 1)]);
        // 覆盖末尾标点不参与
        assert_eq!(m.elem_spans.len(), 5);
    }

    #[test]
    fn matcher_merges_person_names_and_skips_punctuation() {
        let mut library = PatternLibrary::new();
        library.add(PatternEntry::new(
            "关系查询",
            "config.toml#关系查询",
            "relation_query",
            10,
            true,
            vec![
                Elem::pos("nr"),
                Elem::word("和"),
                Elem::pos("nr"),
                Elem::word("什么"),
                Elem::word("关系"),
            ],
        ));
        let tokens = tagged(&[
            ("张三", "nr"),
            ("和", "unknown"),
            ("李四", "nr"),
            ("什么", "r"),
            ("关系", "unknown"),
        ]);
        let m = library.match_route(&tokens).expect("关系模式应命中");
        assert_eq!(m.entities, vec!["张三", "李四"]);
        assert_eq!(m.entity_spans, vec![Span::new(0, 1), Span::new(2, 3)]);
    }

    #[test]
    fn slot_bindings_come_from_sort_constraints() {
        let mut library = PatternLibrary::new();
        let domain = crate::nlu::default_domains()
            .into_iter()
            .find(|d| d.name == "天气")
            .expect("天气领域存在");
        let intent = domain
            .intents
            .iter()
            .find(|i| i.name == "天气查询")
            .unwrap();
        for pattern in &intent.patterns {
            library.add(PatternEntry::from_nlu_intent(&domain, intent, pattern));
        }
        let tokens = tagged(&[("厦门", "ns"), ("明天", "t"), ("天气", "n")]);
        // NLU 意图不参与路由：winner 为 None，但应出现在命中榜并带槽位绑定
        let report = library.rank(&tokens);
        assert!(report.winner.is_none(), "NLU 意图不接管查询路由");
        assert_eq!(report.matched.len(), 1, "意图应出现在命中榜");
        let entry = library.entries()[0].clone();
        let m = match_entry(&entry, &tokens).expect("意图应命中");
        assert_eq!(
            m.bindings.get("城市").and_then(|t| t.render_raw()),
            Some("厦门".to_string())
        );
        assert_eq!(
            m.bindings.get("日期").and_then(|t| t.render_raw()),
            Some("明天".to_string())
        );
    }

    #[test]
    fn nlu_intents_rank_but_do_not_route() {
        let mut library = PatternLibrary::new();
        let domain = crate::nlu::default_domains()
            .into_iter()
            .find(|d| d.name == "天气")
            .expect("天气领域存在");
        let intent = domain
            .intents
            .iter()
            .find(|i| i.name == "天气查询")
            .unwrap();
        library.add(PatternEntry::from_nlu_intent(
            &domain,
            intent,
            intent.patterns[0].as_str(),
        ));
        let tokens = tagged(&[("厦门", "ns"), ("明天", "t"), ("天气", "n")]);
        let report = library.rank(&tokens);
        assert!(report.winner.is_none(), "NLU 意图不参与路由");
        assert_eq!(report.matched.len(), 1, "但应出现在命中榜");
        assert!(!report.matched[0].route);
        // 槽位绑定在胜者之外：从命中榜拿不到解，需要 match_entry——
        // 直接评估该条目验证槽位排序
        let entry = library.entries()[0].clone();
        let m = match_entry(&entry, &tokens).expect("意图应命中");
        assert_eq!(
            m.bindings.get("城市").and_then(|t| t.render_raw()),
            Some("厦门".to_string())
        );
        assert_eq!(
            m.bindings.get("日期").and_then(|t| t.render_raw()),
            Some("明天".to_string())
        );
    }

    #[test]
    fn anchored_mismatch_reports_first_failed_element() {
        let library = sample_library();
        let tokens = tagged(&[("这", "r"), ("是", "v"), ("什么", "r")]);
        let report = library.rank(&tokens);
        assert!(report.winner.is_none());
        assert_eq!(report.failed.len(), 3);
        let reason = match &report.failed[0].status {
            CandidateStatus::Failed { reason } => reason.clone(),
            other => panic!("期望失败，得到 {:?}", other),
        };
        assert!(
            reason.contains("第1要素"),
            "原因应指出首个失配要素：{}",
            reason
        );
    }

    #[test]
    fn ranking_follows_priority_then_config_order() {
        let library = sample_library();
        let tokens = tagged(&[
            ("阿根廷", "ns"),
            ("的", "u"),
            ("首都", "n"),
            ("是", "v"),
            ("哪里", "r"),
        ]);
        let report = library.rank(&tokens);
        assert_eq!(report.matched.len(), 2, "两条首都模式都应命中");
        // 优先级 15 的完整模式排前
        assert_eq!(report.matched[0].name, "首都查询");
        assert_eq!(report.matched[1].name, "首都查询2");
        assert!(report.winner.is_some());
        // 短模式是长模式前缀以外的独立匹配：特化度不同
        if let CandidateStatus::Matched { specificity, .. } = report.matched[0].status {
            assert!((specificity - 1.0).abs() < f64::EPSILON);
        } else {
            panic!("期望命中状态");
        }
        // 同覆盖同路由 → 无冲突
        assert!(report.conflicts().is_empty());
    }

    #[test]
    fn teaching_adds_and_forgets_patterns() {
        let mut library = sample_library();
        let entry = library
            .teach("老家", "{人} 的 老家", "hometown_query", 20)
            .expect("教学应成功");
        assert_eq!(library.route_count(), 4);
        assert_eq!(
            library.get(&entry.short_id()).map(|e| e.name.as_str()),
            Some("老家")
        );
        let tokens = tagged(&[("姚明", "nr"), ("的", "u"), ("老家", "n")]);
        let m = library
            .match_route(&tokens)
            .expect("教学模式应立即参与路由");
        assert_eq!(m.entry.query_type, "hometown_query");
        assert_eq!(m.entities, vec!["姚明"]);
        // 同 id 重复教学 → 替换不新增
        assert!(!library.add(entry.clone()));
        assert_eq!(library.len(), 4);
        let removed = library.forget("老家").expect("应可遗忘");
        assert_eq!(removed.name, "老家");
        assert!(library.get("老家").is_none());
        // 只删教学来源
        assert!(library.forget("首都查询").is_none());
    }

    #[test]
    fn lint_flags_duplicate_conflict_and_shadow() {
        let mut library = PatternLibrary::new();
        library.add(PatternEntry::new(
            "泛化",
            "config.toml#a",
            "q_gen",
            10,
            true,
            vec![
                Elem::slot("实体", "实体"),
                Elem::word("的"),
                Elem::slot("属性", "属性"),
            ],
        ));
        library.add(PatternEntry::new(
            "特化",
            "config.toml#b",
            "q_spec",
            5,
            true,
            vec![Elem::pos("nr"), Elem::word("的"), Elem::pos("n")],
        ));
        // IR 相同、路由不同 → 冲突
        library.add(PatternEntry::new(
            "复刻",
            "taught#会话",
            "q_other",
            3,
            true,
            vec![Elem::pos("nr"), Elem::word("的"), Elem::pos("n")],
        ));
        // IR+路由+来源不同的完全重复（相对另一条同 IR 同路由）
        library.add(PatternEntry::new(
            "重复",
            "taught#会话2",
            "q_gen",
            9,
            true,
            vec![
                Elem::slot("实体", "实体"),
                Elem::word("的"),
                Elem::slot("属性", "属性"),
            ],
        ));
        let findings = lint(&library);
        assert!(
            findings.iter().any(|f| f.kind == LintKind::Conflict),
            "应检出 IR 同路由不同"
        );
        assert!(
            findings.iter().any(|f| f.kind == LintKind::Redundant),
            "应检出重复定义"
        );
        assert!(
            findings.iter().any(|f| f.kind == LintKind::Shadowed),
            "应检出泛化遮蔽特化：{:?}",
            findings
        );
        // 遮蔽方向：泛化在前（优先 10 > 5）
        let shadow = findings
            .iter()
            .find(|f| f.kind == LintKind::Shadowed)
            .unwrap();
        assert_eq!(shadow.subject_name, "泛化");
        assert_eq!(shadow.object_name, "特化");
    }

    #[test]
    fn subsumption_is_conservative() {
        let general = PatternEntry::new(
            "g",
            "s",
            "q",
            1,
            true,
            vec![Elem::pos("ns"), Elem::word("首都")],
        );
        let specific = PatternEntry::new(
            "sp",
            "s",
            "q",
            1,
            true,
            vec![Elem::pos("ns"), Elem::word("的"), Elem::word("首都")],
        );
        assert!(!subsumes(&general, &specific), "词面错位不构成覆盖");
        let prefix = PatternEntry::new("p", "s", "q", 1, true, vec![Elem::pos("ns")]);
        assert!(subsumes(&prefix, &specific));
        assert!(!subsumes(&specific, &prefix), "长不覆盖短");
        assert!(!subsumes(&prefix, &prefix), "自身不构成严格覆盖");
    }

    #[test]
    fn relation_edges_mirror_lint_findings() {
        let mut library = PatternLibrary::new();
        library.add(PatternEntry::new(
            "泛化",
            "config.toml#a",
            "q_gen",
            10,
            true,
            vec![
                Elem::slot("实体", "实体"),
                Elem::word("的"),
                Elem::slot("属性", "属性"),
            ],
        ));
        library.add(PatternEntry::new(
            "特化",
            "config.toml#b",
            "q_spec",
            5,
            true,
            vec![Elem::pos("nr"), Elem::word("的"), Elem::pos("n")],
        ));
        library.add(PatternEntry::new(
            "复刻",
            "taught#会话",
            "q_other",
            3,
            true,
            vec![Elem::pos("nr"), Elem::word("的"), Elem::pos("n")],
        ));
        let edges = relation_edges(library.entries());
        assert!(
            edges.iter().any(|e| e.kind == RelationKind::Generalizes),
            "应产出泛化边：{:?}",
            edges
        );
        assert!(
            edges.iter().any(|e| e.kind == RelationKind::Conflicts),
            "应产出冲突边：{:?}",
            edges
        );
        // 与 lint 一一对应（图视角就是体检结果的另一面）
        assert_eq!(edges.len(), lint(&library).len());
        for edge in &edges {
            assert_eq!(
                library.get(&edge.subject_id).map(|e| e.name.as_str()),
                Some(edge.subject_name.as_str())
            );
        }
    }

    #[test]
    fn pattern_quads_carry_relations_and_ignore_them_on_parse() {
        let mut library = PatternLibrary::new();
        library.add(PatternEntry::new(
            "A",
            "config.toml#a",
            "q1",
            10,
            true,
            vec![Elem::pos("nr"), Elem::word("的"), Elem::pos("n")],
        ));
        library.add(PatternEntry::new(
            "B",
            "config.toml#b",
            "q2",
            5,
            true,
            vec![Elem::pos("nr"), Elem::word("的"), Elem::pos("n")],
        ));
        let quads = pattern_quads(library.entries()).expect("生成四元组");
        // IR 同、路由不同 → 冲突，对称关系双向各一条
        let conflicts: Vec<_> = quads
            .iter()
            .filter(|q| q.predicate.as_str() == P_CONFLICTS)
            .collect();
        assert_eq!(conflicts.len(), 2, "冲突关系应对称落图");
        // 关系边不干扰条目还原
        let restored = parse_pattern_graph(quads.into_iter());
        assert_eq!(restored.len(), 2);
    }

    #[test]
    fn relation_count_sparql_targets_pattern_graph() {
        let query = relation_count_sparql();
        assert!(query.contains(PATTERNS_GRAPH));
        assert!(query.contains(P_CONFLICTS));
        assert!(query.contains("GROUP BY"));
    }

    #[test]
    fn relation_report_lists_edges_and_query() {
        let mut library = PatternLibrary::new();
        library.add(PatternEntry::new(
            "A",
            "config.toml#a",
            "q1",
            10,
            true,
            vec![Elem::pos("nr"), Elem::word("的"), Elem::pos("n")],
        ));
        library.add(PatternEntry::new(
            "B",
            "config.toml#b",
            "q2",
            5,
            true,
            vec![Elem::pos("nr"), Elem::word("的"), Elem::pos("n")],
        ));
        let report = render_relation_report(&library);
        assert!(report.contains(PATTERNS_GRAPH));
        assert!(report.contains("冲突"));
        assert!(report.contains("图体检查询"));
    }

    #[test]
    fn quads_roundtrip_preserves_entries() {
        let mut library = sample_library();
        library
            .teach("老家", "{人} 的 老家", "hometown_query", 20)
            .unwrap();
        let quads = pattern_quads(library.entries()).expect("生成四元组");
        assert!(!quads.is_empty());
        // 主语都在模式命名图
        for quad in &quads {
            assert_eq!(
                quad.graph_name,
                oxigraph::model::GraphName::NamedNode(
                    oxigraph::model::NamedNode::new(PATTERNS_GRAPH).unwrap()
                )
            );
        }
        let restored = parse_pattern_graph(quads.into_iter());
        assert_eq!(restored.len(), library.len());
        let mut left: Vec<_> = library.entries().to_vec();
        left.sort_by_key(PatternEntry::id);
        let mut right = restored;
        right.sort_by_key(PatternEntry::id);
        assert_eq!(left, right);
        // 物化戳与集合指纹
        assert_eq!(
            library_fingerprint(library.entries()),
            library_fingerprint(&right)
        );
    }

    #[test]
    fn nlu_intent_only_routes_as_fallback_when_config_misses() {
        let config: SystemConfig =
            toml::from_str(include_str!("../config.toml")).expect("config.toml 必须可解析");
        let library = PatternLibrary::from_config(&config);
        let legacy_rules = create_prolog_rules(&config);
        assert!(library.fallback_count() > 0, "应有与配置规则对齐的 NLU 回退意图");

        let tokens = preprocess("姚明年龄", &config.nlp);
        assert!(
            analyze_query_intent(&tokens, &legacy_rules).is_none(),
            "配置规则不应命中这个问法"
        );
        assert!(library.match_route(&tokens).is_none(), "主路由不应命中");
        let fallback = library
            .match_fallback(&tokens)
            .expect("NLU 意图应作为回退命中");
        assert_eq!(fallback.entry.query_type, "age_query");
        assert_eq!(fallback.entities, vec!["姚明".to_string()]);
        assert!(fallback.entry.source.starts_with("nlu#"), "{}", fallback.entry.source);

        // 配置能命中的问法仍走主路由，回退不改变既有行为。
        let configured = preprocess("小明的年龄", &config.nlp);
        assert!(library.match_route(&configured).is_some());
        assert_eq!(
            library.route(&configured),
            analyze_query_intent(&configured, &legacy_rules)
        );
    }

    #[test]
    fn route_parity_with_legacy_analyze_query_intent() {
        let config: SystemConfig =
            toml::from_str(include_str!("../config.toml")).expect("config.toml 必须可解析");
        let library = PatternLibrary::from_config(&config);
        let legacy_rules = create_prolog_rules(&config);

        let corpus: Vec<&str> = vec![
            "法国的首都是哪里",
            "阿根廷首都",
            "小明多大",
            "小明的年龄",
            "张三和李四什么关系",
            "有多少个人",
            "水的化学分子式是什么",
            "红楼梦的作者是谁",
            "什么是光合作用",
            "解释一下什么是相对论",
            "太阳系中最大的行星是哪一颗",
            "北京的海拔高度是多少",
            "这是一句毫无意义的话",
            "你好",
            "谢谢",
        ];
        let mut agreements = 0usize;
        for input in corpus {
            for tokens in [preprocess(input, &config.nlp), manual_tokens(input)] {
                let legacy = analyze_query_intent(&tokens, &legacy_rules);
                let unified = library.route(&tokens);
                assert_eq!(
                    legacy, unified,
                    "路由不一致：输入「{}」legacy={:?} unified={:?}",
                    input, legacy, unified
                );
                if legacy.is_some() {
                    agreements += 1;
                }
            }
        }
        assert!(
            agreements >= 8,
            "语料应覆盖足够多的命中样本，实得 {}",
            agreements
        );
    }

    #[test]
    fn first_elem_prefilter_is_conservative_and_reduces_scan() {
        let config: SystemConfig =
            toml::from_str(include_str!("../config.toml")).expect("config.toml 必须可解析");
        let library = PatternLibrary::from_config(&config);
        let context = PatternContext::default();
        let corpus = [
            "法国的首都是哪里",
            "姚明的年龄",
            "张三和李四什么关系",
            "水的化学分子式是什么",
            "有多少个人",
            "你好",
            "这是一句毫无意义的话",
        ];
        for input in corpus {
            let tokens = preprocess(input, &config.nlp);
            let scan = library.scan_route_with_context(&tokens, &context);
            let mut reference = None;
            for index in library.evaluation_order() {
                let entry = &library.entries[index];
                if !entry.route || !entry.context_satisfied(&context) {
                    continue;
                }
                if let Ok(matched) = match_entry(entry, &tokens) {
                    reference = Some(matched);
                    break;
                }
            }
            assert_eq!(
                scan.winner, reference,
                "预筛不得改变路由结果：输入「{}」",
                input
            );
            assert!(
                scan.scanned() <= library.route_count(),
                "扫描数不能超过可路由条目数：输入「{}」",
                input
            );
        }

        let tokens = preprocess("姚明的年龄", &config.nlp);
        let scan = library.scan_route_with_context(&tokens, &context);
        assert!(scan.winner.is_some(), "该问法应可路由");
        assert!(scan.skipped > 0, "首要素预筛应跳过部分条目");
        assert!(
            scan.evaluated < library.route_count(),
            "完整评估数应少于全库：{} >= {}",
            scan.evaluated,
            library.route_count()
        );
    }

    /// 无词典依赖的确定性 token 序列（仅覆盖语料中的高频词）。
    fn manual_tokens(input: &str) -> Vec<TaggedWord> {
        const LEXICON: &[(&str, &str)] = &[
            ("法国", "ns"),
            ("阿根廷", "ns"),
            ("北京", "ns"),
            ("的", "u"),
            ("首都", "n"),
            ("是", "v"),
            ("哪里", "r"),
            ("小明", "nr"),
            ("多大", "r"),
            ("年龄", "n"),
            ("张三", "nr"),
            ("和", "unknown"),
            ("李四", "nr"),
            ("什么", "r"),
            ("关系", "unknown"),
            ("有", "v"),
            ("多少", "m"),
            ("个", "q"),
            ("人", "n"),
            ("水", "n"),
            ("化学", "n"),
            ("分子式", "n"),
            ("红楼梦", "n"),
            ("作者", "n"),
            ("谁", "r"),
            ("光合作用", "n"),
            ("相对论", "n"),
            ("解释", "v"),
            ("一下", "v"),
            ("太阳系", "ns"),
            ("中", "f"),
            ("最大", "a"),
            ("行星", "n"),
            ("哪一颗", "r"),
            ("海拔", "n"),
            ("高度", "n"),
            ("这", "r"),
            ("一", "m"),
            ("句", "q"),
            ("毫无", "d"),
            ("意义", "n"),
            ("话", "n"),
            ("你好", "l"),
            ("谢谢", "v"),
        ];
        // 词面直配（最长优先），未命中字符归入 unknown 单字
        let mut tokens = Vec::new();
        let chars: Vec<char> = input.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            let rest: String = chars[i..].iter().collect();
            let mut matched = false;
            let mut surfaces: Vec<&(&str, &str)> = LEXICON.iter().collect();
            surfaces.sort_by_key(|(text, _)| std::cmp::Reverse(text.chars().count()));
            for &(text, pos) in surfaces {
                if rest.starts_with(text) {
                    tokens.push(TaggedWord {
                        text: text.to_string(),
                        pos: pos.to_string(),
                    });
                    i += text.chars().count();
                    matched = true;
                    break;
                }
            }
            if !matched {
                tokens.push(TaggedWord {
                    text: chars[i].to_string(),
                    pos: "unknown".to_string(),
                });
                i += 1;
            }
        }
        tokens
    }

    #[test]
    fn report_renders_winner_bindings_and_reasons() {
        let library = sample_library();
        let tokens = tagged(&[
            ("法国", "ns"),
            ("的", "u"),
            ("首都", "n"),
            ("是", "v"),
            ("哪里", "r"),
        ]);
        let report = library.rank(&tokens);
        let text = render_match_report("法国的首都是哪里", &report);
        assert!(text.contains("首都查询"), "{}", text);
        assert!(text.contains("entity.1=法国"), "绑定应带区间证据：{}", text);
        assert!(text.contains("未命中"), "失败原因应展示：{}", text);
        let empty = library.rank(&tagged(&[("这", "r")]));
        let empty_text = render_match_report("这", &empty);
        assert!(empty_text.contains("无可路由命中"), "{}", empty_text);
        let lib_text = render_library_report(&library);
        assert!(lib_text.contains("模式库：3 条"), "{}", lib_text);
    }

    #[test]
    fn bound_terms_are_typed_in_bindings() {
        let mut library = PatternLibrary::new();
        library
            .teach("计数", "有 多少 {物}", "count_query", 8)
            .unwrap();
        let tokens = tagged(&[("有", "v"), ("多少", "m"), ("人", "n")]);
        let m = library.match_route(&tokens).expect("应命中");
        let term = m.bindings.get("物").expect("槽位应绑定");
        // BindingSet::bind 走分类：文本按数值/文本归类
        assert_eq!(term.render_raw().as_deref(), Some("人"));
        assert!(matches!(term, BoundTerm::Text { .. }));
    }

    #[test]
    fn episodes_merge_repeats_and_evict_by_recency() {
        let library = sample_library();
        let tokens = tagged(&[
            ("法国", "ns"),
            ("的", "u"),
            ("首都", "n"),
            ("是", "v"),
            ("哪里", "r"),
        ]);
        let m = library.match_route(&tokens).expect("应命中");
        let mut log = PatternEpisodeLog::with_capacity(3);
        let first = log.record_match("法国的首都是哪里", &tokens, &m, Some("法国"), 2);
        assert_eq!(first.count, 1);
        assert_eq!(first.outcome, EpisodeOutcome::Matched);
        let again = log.record_match("法国的首都是哪里", &tokens, &m, Some("法国"), 3);
        assert_eq!(again.count, 2, "同一经验应累计支持度");
        assert_eq!(log.len(), 1, "相同经验不重复占容量");
        assert_eq!(log.clock(), 3);

        for turn in 4..8 {
            log.record_unmatched(&format!("未知句子{}", turn), &tagged(&[("未知", "n")]), None, turn);
        }
        assert_eq!(log.len(), 3, "容量必须封顶");
        assert_eq!(log.matched_count(), 1);
        assert_eq!(log.unmatched_count(), 2);
        assert!(
            log.entries().iter().any(|episode| episode.input == "法国的首都是哪里"),
            "最近使用的经验不应被淘汰"
        );
        let report = render_episode_report(&log);
        assert!(report.contains("模式经验：3 条"), "{}", report);
        assert!(report.contains("×2"), "{}", report);
    }

    #[test]
    fn episode_quads_roundtrip_in_isolated_graph() {
        let tokens = tagged(&[("这", "r"), ("是", "v")]);
        let mut log = PatternEpisodeLog::with_capacity(8);
        let left = log.record_unmatched("这是", &tokens, Some("焦点"), 1);
        let right = log.record_unmatched("那是", &tokens, None, 2);
        let quads = episode_quads(log.entries()).expect("生成经验四元组");
        assert_eq!(
            quads[0].graph_name,
            oxigraph::model::GraphName::NamedNode(
                oxigraph::model::NamedNode::new(EPISODES_GRAPH).unwrap()
            )
        );
        let restored = parse_episode_graph(quads.into_iter());
        assert_eq!(restored.len(), 2);
        let mut ids: Vec<String> = restored.iter().map(PatternEpisode::id).collect();
        ids.sort();
        let mut expected = vec![left.id(), right.id()];
        expected.sort();
        assert_eq!(ids, expected);
        assert_eq!(
            episode_fingerprint(&restored),
            episode_fingerprint(log.entries()),
            "经验集指纹与插入顺序无关"
        );
    }

    #[test]
    fn case_suggestion_generalizes_unseen_entity_but_stays_inactive() {
        let library = sample_library();
        let stored = tagged(&[
            ("法国", "ns"),
            ("的", "u"),
            ("首都", "n"),
            ("是", "v"),
            ("哪里", "r"),
        ]);
        let matched = library.match_route(&stored).expect("应命中");
        let mut log = PatternEpisodeLog::default();
        log.record_match("法国的首都是哪里", &stored, &matched, Some("法国"), 4);

        let current = tagged(&[
            ("德国", "ns"),
            ("的", "u"),
            ("首都", "n"),
            ("在", "v"),
            ("哪里", "r"),
        ]);
        let suggestion = log
            .suggest_case(&current, Some("法国"), 0.2)
            .expect("相似经验应给出候选骨架");
        assert!(suggestion.score >= 0.2);
        assert_eq!(suggestion.episode.winner_name.as_deref(), Some("首都查询"));
        assert!(!suggestion.pattern.route, "候选不得直接参与路由");
        assert_eq!(suggestion.pattern.source, format!("case#{}", suggestion.episode.short_id()));
        assert!(
            suggestion.pattern.render().contains("槽1:地点"),
            "未见实体应升为地点槽：{}",
            suggestion.pattern.render()
        );
        assert!(
            match_entry(&suggestion.pattern, &current).is_ok(),
            "候选必须能解释当前输入"
        );
        let report = render_case_suggestion("德国的首都在哪里", &suggestion);
        assert!(report.contains("未激活"), "{}", report);
        assert!(report.contains("确认方式"), "{}", report);

        let unmatched = tagged(&[("今天", "t"), ("心情", "n"), ("不错", "a")]);
        assert!(
            log.suggest_case(&unmatched, None, 0.2).is_none(),
            "无共享词且无焦点时不应硬造候选"
        );
    }

    #[test]
    fn accepted_case_becomes_routable_taught_pattern() {
        let mut library = PatternLibrary::new();
        library.add(PatternEntry::new(
            "首都查询",
            "config.toml#首都查询",
            "capital_query",
            15,
            true,
            vec![
                Elem::pos("ns"),
                Elem::word("的"),
                Elem::word("首都"),
                Elem::word("是"),
                Elem::word("哪里"),
            ],
        ));
        let stored = tagged(&[
            ("法国", "ns"),
            ("的", "u"),
            ("首都", "n"),
            ("是", "v"),
            ("哪里", "r"),
        ]);
        let matched = library.match_route(&stored).expect("应命中");
        let mut log = PatternEpisodeLog::default();
        log.record_match("法国的首都是哪里", &stored, &matched, Some("法国"), 4);

        let current = tagged(&[
            ("德国", "ns"),
            ("的", "u"),
            ("首都", "n"),
            ("在", "v"),
            ("哪里", "r"),
        ]);
        let suggestion = log
            .suggest_case(&current, Some("法国"), 0.2)
            .expect("相似经验应给出候选骨架");
        assert!(library.match_route(&current).is_none(), "确认前不应路由");

        let mut active = library.clone();
        let entry = active.accept_case(&suggestion).expect("确认案例应成功");
        assert!(entry.route, "确认后必须可路由");
        assert_eq!(entry.query_type, "capital_query");
        assert!(entry.source.starts_with("taught#案例#"), "{}", entry.source);
        assert_eq!(
            active.accept_case(&suggestion).expect("重复确认幂等"),
            entry
        );
        assert_eq!(
            active.count_with_source_prefix("taught#"),
            1,
            "重复确认不得重复登记"
        );
        let routed = active.match_route(&current).expect("确认后应命中新实体");
        assert_eq!(routed.entry.name, entry.name);
        assert_eq!(routed.entities, vec!["德国".to_string()]);

        let quads = pattern_quads(active.entries()).expect("确认模式应可落图");
        let restored = parse_pattern_graph(quads.into_iter());
        let loaded = restored
            .iter()
            .find(|candidate| candidate.id() == entry.id())
            .expect("确认模式应随账本恢复");
        assert!(loaded.route);
        assert_eq!(loaded.source, entry.source);

        assert!(active.forget(&entry.name).is_some(), "确认模式应可遗忘");
    }

    #[test]
    fn episode_context_scopes_retrieval_and_suggestion() {
        let mut library = PatternLibrary::new();
        library.add(PatternEntry::new(
            "首都查询",
            "config.toml#首都查询",
            "capital_query",
            15,
            true,
            vec![
                Elem::pos("ns"),
                Elem::word("的"),
                Elem::word("首都"),
                Elem::word("是"),
                Elem::word("哪里"),
            ],
        ));
        let stored = tagged(&[
            ("法国", "ns"),
            ("的", "u"),
            ("首都", "n"),
            ("是", "v"),
            ("哪里", "r"),
        ]);
        let matched = library.match_route(&stored).expect("应命中");
        let mut log = PatternEpisodeLog::default();
        let asked = PatternContext {
            focus: Some("法国".to_string()),
            focus_type: Some("地点".to_string()),
            last_move: Some("ask_attribute".to_string()),
            pending_clarify: false,
            turn: 3,
        };
        let greet = PatternContext {
            focus: Some("法国".to_string()),
            focus_type: Some("地点".to_string()),
            last_move: Some("greet".to_string()),
            pending_clarify: false,
            turn: 4,
        };
        log.record_match_with_context("法国的首都是哪里", &stored, &matched, &asked);
        log.record_match_with_context("法国的首都是哪里", &stored, &matched, &greet);
        assert_eq!(log.len(), 2, "不同上轮动作应保留为不同上下文经验");
        assert_ne!(log.entries()[0].id(), log.entries()[1].id());

        let neighbors = log.nearest_with_context(&stored, &asked, 2);
        assert_eq!(neighbors.len(), 2);
        assert_eq!(
            log.entries()[neighbors[0].index].last_move.as_deref(),
            Some("ask_attribute"),
            "上下文匹配的经验应排前"
        );
        let suggestion = log
            .suggest_case_with_context(&stored, &greet, 0.2)
            .expect("相似成功经验应给出候选");
        assert_eq!(suggestion.episode.last_move.as_deref(), Some("greet"));
        let report = render_case_suggestion("法国的首都是哪里", &suggestion);
        assert!(report.contains("发生上下文"), "{}", report);
        assert!(report.contains("上轮 greet"), "{}", report);
        let episode_report = render_episode_report(&log);
        assert!(episode_report.contains("上轮 ask_attribute"), "{}", episode_report);
        assert!(episode_report.contains("焦点 法国(地点)"), "{}", episode_report);
    }

    #[test]
    fn elliptical_case_suggestion_keeps_focus_guard_after_acceptance() {
        let mut library = PatternLibrary::new();
        library.add(PatternEntry::new(
            "身高查询",
            "config.toml#身高查询",
            "height_query",
            10,
            true,
            vec![Elem::pos("nr"), Elem::word("的"), Elem::word("身高")],
        ));
        let stored = tagged(&[("姚明", "nr"), ("的", "u"), ("身高", "n")]);
        let matched = library.match_route(&stored).expect("应命中");
        let focus_context = PatternContext {
            focus: Some("姚明".to_string()),
            focus_type: Some("人物".to_string()),
            last_move: Some("ask_attribute".to_string()),
            ..PatternContext::default()
        };
        let mut log = PatternEpisodeLog::default();
        log.record_match_with_context("姚明的身高", &stored, &matched, &focus_context);

        let elliptical = tagged(&[("身高", "n"), ("呢", "u")]);
        let suggestion = log
            .suggest_case_with_context(&elliptical, &focus_context, 0.2)
            .expect("相似经验应给出省略问法候选");
        assert_eq!(
            suggestion.pattern.guard,
            Some(ContextGuard::focus_type("人物")),
            "省略问法候选必须带焦点类型守卫"
        );
        let mut active = library.clone();
        active.accept_case(&suggestion).expect("确认应成功");
        assert!(
            active.match_route(&elliptical).is_none(),
            "无焦点时守卫必须拦住省略问法"
        );
        assert_eq!(
            active.route_with_context(&elliptical, &focus_context),
            Some(("height_query".to_string(), Vec::new()))
        );
    }

    #[test]
    fn context_guard_parses_and_gates_routing() {
        assert_eq!(
            parse_context_guard("焦点").unwrap(),
            ContextGuard::focus()
        );
        assert_eq!(
            parse_context_guard("焦点类型：人物").unwrap(),
            ContextGuard::focus_type("人物")
        );
        assert_eq!(
            parse_context_guard("上轮=ask_attribute；无待澄清").unwrap(),
            ContextGuard {
                focus_required: false,
                focus_type: None,
                last_move: Some("ask_attribute".to_string()),
                pending_clarify: Some(false),
            }
        );
        assert!(parse_context_guard("").is_err());
        assert!(parse_context_guard("未知守卫").is_err());

        let mut library = PatternLibrary::new();
        library
            .teach_with_guard(
                "省略身高",
                "身高 呢",
                "height_query",
                30,
                Some(ContextGuard::focus()),
            )
            .unwrap();
        let tokens = tagged(&[("身高", "n"), ("呢", "u")]);
        assert!(
            library.match_route(&tokens).is_none(),
            "无焦点时守卫必须拦住省略问句"
        );
        let blocked = library.rank(&tokens);
        assert!(blocked.winner.is_none());
        assert!(
            blocked.failed.iter().any(|outcome| matches!(
                &outcome.status,
                CandidateStatus::Failed { reason } if reason.contains("上下文守卫")
            )),
            "失败榜应披露守卫原因"
        );
        let context = PatternContext {
            focus: Some("姚明".to_string()),
            focus_type: Some("人物".to_string()),
            ..PatternContext::default()
        };
        assert_eq!(
            library.route_with_context(&tokens, &context),
            Some(("height_query".to_string(), Vec::new()))
        );
        let report = render_match_report("身高呢", &library.rank_with_context(&tokens, &context));
        assert!(report.contains("上下文守卫：焦点"), "{}", report);

        let guarded = &library.entries()[0];
        let quads = pattern_quads(library.entries()).expect("守卫条目应可落图");
        assert!(
            quads.iter().any(|quad| quad.predicate.as_str() == P_GUARD),
            "守卫应作为可查询字面量落图"
        );
        let restored = parse_pattern_graph(quads.into_iter());
        assert_eq!(restored[0].guard, guarded.guard);
    }

    #[test]
    fn case_reply_only_accepts_whole_short_answers() {
        for answer in ["是", "是的。", "对呀", "好的", "确认", "YES", "没错啊"] {
            assert_eq!(
                classify_case_reply(answer),
                CaseReply::Accept,
                "「{}」应视为确认",
                answer
            );
        }
        for answer in ["不是", "不对！", "不", "没有", "算了", "不要", "no", "否吧"] {
            assert_eq!(
                classify_case_reply(answer),
                CaseReply::Reject,
                "「{}」应视为否定",
                answer
            );
        }
        for answer in ["不知道", "不确定", "跳过", "下一个"] {
            assert_eq!(
                classify_case_reply(answer),
                CaseReply::Unclear,
                "「{}」应视为暂不确定",
                answer
            );
        }
        for answer in ["是什么", "是的，首都在哪里", "帮助", "", "确认联想：德国"] {
            assert_eq!(
                classify_case_reply(answer),
                CaseReply::NewInput,
                "「{}」不是对联想提问的短答",
                answer
            );
        }
    }

    #[test]
    fn rejected_case_is_not_reoffered_and_feedback_persists() {
        let library = sample_library();
        let stored = tagged(&[
            ("法国", "ns"),
            ("的", "u"),
            ("首都", "n"),
            ("是", "v"),
            ("哪里", "r"),
        ]);
        let matched = library.match_route(&stored).expect("应命中");
        let mut log = PatternEpisodeLog::default();
        log.record_match("法国的首都是哪里", &stored, &matched, Some("法国"), 4);
        let current = tagged(&[
            ("德国", "ns"),
            ("的", "u"),
            ("首都", "n"),
            ("在", "v"),
            ("哪里", "r"),
        ]);
        let suggestion = log
            .suggest_case(&current, Some("法国"), 0.2)
            .expect("首次应给出候选");
        assert!(log.reject_case(&suggestion), "否定应命中经验");
        assert!(
            log.suggest_case(&current, Some("法国"), 0.2).is_none(),
            "净否定后的骨架不应再次主动推荐"
        );
        let latest = log.entries().last().expect("经验仍在账本中");
        assert_eq!(latest.rejections, 1);
        let report = render_episode_report(&log);
        assert!(report.contains("否定×1"), "{}", report);

        let quads = episode_quads(log.entries()).expect("反馈计数应可落图");
        let restored = parse_episode_graph(quads.into_iter());
        assert_eq!(restored.len(), 1);
        assert_eq!(restored[0].rejections, 1);

        assert!(log.confirm_case(&suggestion), "确认也应命中经验");
        assert_eq!(
            log.entries().last().map(|episode| episode.acceptances),
            Some(1)
        );
        assert!(
            log.suggest_case(&current, Some("法国"), 0.2).is_some(),
            "确认抵消净否定后应允许再次联想"
        );
    }

    #[test]
    fn forget_removes_single_episode_by_short_id_or_exact_input() {
        let library = sample_library();
        let stored = tagged(&[
            ("法国", "ns"),
            ("的", "u"),
            ("首都", "n"),
            ("是", "v"),
            ("哪里", "r"),
        ]);
        let matched = library.match_route(&stored).expect("应命中");
        let mut log = PatternEpisodeLog::default();
        let episode = log.record_match("法国的首都是哪里", &stored, &matched, None, 1);
        assert_eq!(log.len(), 1);
        assert_eq!(log.forget("deadbeef"), 0, "未知 id 不应误删");
        assert_eq!(log.forget(&episode.short_id()), 1);
        assert!(log.is_empty(), "按短 id 应删除单条经验");

        log.record_match("法国的首都是哪里", &stored, &matched, None, 2);
        assert_eq!(log.forget("法国的首都是哪里"), 1);
        assert!(log.is_empty(), "按完整原句也应删除单条经验");
    }
}

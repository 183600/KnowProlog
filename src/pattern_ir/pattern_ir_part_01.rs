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
            ("上轮", Some(movement)) | ("动作", Some(movement)) | ("last_move", Some(movement)) =>
            {
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
    /// 精确命中后是否仍允许软路由复核并替换。泛化前缀模式用它让位给更具体
    /// 的近失解释；软层不接管时仍保留本条目，避免降低召回。
    #[serde(default, skip_serializing_if = "is_false")]
    pub defer_to_soft: bool,
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
            defer_to_soft: false,
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
        let mut entry = Self::new(
            rule.name.clone(),
            format!("config.toml#{}", rule.name),
            rule.query_type.clone(),
            rule.priority,
            true,
            rule.pattern.iter().map(|t| parse_config_token(t)).collect(),
        );
        entry.defer_to_soft = rule.defer_to_soft;
        entry
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

    /// 把词面要素归一到同义词表的规范形。
    ///
    /// 运行时先 [`crate::nlp::normalize_words`] 再路由，因此模式里的词面
    /// 也必须用同一个规范形；否则配置写「老婆」、输入归一成「妻子」后，
    /// 规则永远等不到自己的字面。只做一遍替换，与 `normalize_words` 同口径。
    pub fn normalize_literals(&mut self, synonyms: &std::collections::HashMap<String, String>) {
        for elem in &mut self.elems {
            if let Elem::Word { surface } = elem {
                if let Some(canonical) = synonyms.get(surface.as_str()) {
                    *surface = canonical.clone();
                }
            }
        }
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
/// 单要素失配的类别。分类本身就是可行动信息：词面不符可以补别名，
/// 词性不符要改槽位排序，输入耗尽则要看是否缺少尾缀/成分。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MatchFailureKind {
    /// 输入在到达该要素前已耗尽。
    InputExhausted,
    /// 词性不符（含人名 `nr`）。
    PosMismatch,
    /// 固定词面不符。
    WordMismatch,
    /// 槽位排序不符。
    SortMismatch,
}
impl MatchFailureKind {
    pub fn label(self) -> &'static str {
        match self {
            MatchFailureKind::InputExhausted => "输入耗尽",
            MatchFailureKind::PosMismatch => "词性不符",
            MatchFailureKind::WordMismatch => "词面不符",
            MatchFailureKind::SortMismatch => "排序不符",
        }
    }
}
/// 一次**决策前沿**（decision frontier）：匹配器停在哪个要素、期望什么、
/// 实际见到什么，以及此前已经覆盖了哪些要素。
///
/// 这是与 [`PatternMatch`] 对称的失败解。日志/评估一旦拿到它，就能把
/// 「没懂」分解成「第几个要素、缺哪类词面、离命中还差多少」，而不是只留
/// 一句不可行动的失败原因。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatchFailure {
    /// 失配要素下标（0 起）。
    pub index: usize,
    /// 模式要素总数。
    pub total_elems: usize,
    /// 失配发生在输入 token 序列的哪个位置。
    pub position: usize,
    /// 失配类别。
    pub kind: MatchFailureKind,
    /// 期望的要素（人类可读，如 `是`、`人名(nr)`、`{城市:城市}`）。
    pub expected: String,
    /// 实际见到的 token 词面（输入耗尽时为 `None`）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actual: Option<String>,
    /// 实际见到的 token 词性（输入耗尽时为 `None`）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actual_pos: Option<String>,
    /// 失配前已经覆盖的要素区间（`None` = 可选要素未占位）。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub matched_elems: Vec<Option<Span>>,
    /// 失配位置之后还剩多少 token（含当前 token）。
    pub remaining_tokens: usize,
}
impl MatchFailure {
    /// 已完整覆盖的要素数（可选未占位也计入，因为它是被有意跳过的）。
    pub fn covered_elems(&self) -> usize {
        self.matched_elems.len()
    }

    /// 前缀覆盖率：已覆盖要素 / 总要素。用于识别「一步之遥」的缺口。
    pub fn coverage(&self) -> f64 {
        if self.total_elems == 0 {
            return 1.0;
        }
        self.covered_elems() as f64 / self.total_elems as f64
    }

    /// 是否属于「大部分结构已对上」的近似命中（阈值由调用方给定，
    /// 评估侧默认用 [`crate::recall_audit::NEAR_MISS_COVERAGE`]）。
    ///
    /// 线性匹配器只报第一个失配点，因此覆盖率是**前缀**覆盖率；它不宣称
    /// 后缀一定可匹配，只负责把「离命中还差多少」变成可排序的数。
    pub fn is_near_miss(&self, threshold: f64) -> bool {
        self.coverage() >= threshold
    }

    /// 人类可读的一行说明（日志与 CLI 共用）。
    pub fn render(&self) -> String {
        let actual = match (&self.actual, &self.actual_pos) {
            (Some(text), Some(pos)) => format!("「{}」({})", text, pos),
            (Some(text), None) => format!("「{}」", text),
            (None, _) => "输入结束".to_string(),
        };
        format!(
            "第{}要素「{}」未匹配（{}）：实见 {}；此前 {}/{} 要素已覆盖，剩余 {} token",
            self.index + 1,
            self.expected,
            self.kind.label(),
            actual,
            self.covered_elems(),
            self.total_elems,
            self.remaining_tokens
        )
    }
}
impl std::fmt::Display for MatchFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // 保持旧版失败原因逐字不变：调用方/测试把它当稳定文案。
        match self.kind {
            MatchFailureKind::InputExhausted => write!(
                formatter,
                "第{}要素「{}」未匹配：输入已耗尽",
                self.index + 1,
                self.expected
            ),
            MatchFailureKind::PosMismatch => write!(
                formatter,
                "第{}要素{}，实得词性「{}」（「{}」）",
                self.index + 1,
                self.expected,
                self.actual_pos.as_deref().unwrap_or(""),
                self.actual.as_deref().unwrap_or("")
            ),
            MatchFailureKind::WordMismatch => write!(
                formatter,
                "第{}要素期望词「{}」，实得「{}」",
                self.index + 1,
                self.expected,
                self.actual.as_deref().unwrap_or("")
            ),
            MatchFailureKind::SortMismatch => write!(
                formatter,
                "第{}要素{}，实得词性「{}」（「{}」）",
                self.index + 1,
                self.expected,
                self.actual_pos.as_deref().unwrap_or(""),
                self.actual.as_deref().unwrap_or("")
            ),
        }
    }
}
impl std::error::Error for MatchFailure {}
/// 一条失败经验里保存的「最近决策前沿」快照。
///
/// 经验账本记原句与词序列还不够：同一句「没懂」可能离命中只差一个词面，
/// 也可能结构完全不对。把最近的受阻模式与失配要素一起落账，后续才能按
/// 「缺口类型」聚合排序，而不是逐句重放。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpisodeFrontier {
    pub pattern_id: String,
    pub pattern_name: String,
    pub query_type: String,
    /// 已覆盖前缀比例（0..=1）。
    pub coverage: f64,
    pub failure: MatchFailure,
}
impl EpisodeFrontier {
    pub fn render(&self) -> String {
        format!(
            "最近受阻「{}」（{}，覆盖 {:.0}%）：{}",
            self.pattern_name,
            self.query_type,
            self.coverage * 100.0,
            self.failure.render()
        )
    }
}
/// 单要素层面的失配（由 [`attempt_elem`] 产生，由 [`match_entry_detailed`]
/// 补全为带覆盖信息的 [`MatchFailure`]）。
#[derive(Debug, Clone, PartialEq)]
struct ElemFailure {
    kind: MatchFailureKind,
    expected: String,
    actual: Option<String>,
    actual_pos: Option<String>,
}
impl ElemFailure {
    fn new(
        kind: MatchFailureKind,
        expected: impl Into<String>,
        token: &crate::nlp::TaggedWord,
    ) -> Self {
        Self {
            kind,
            expected: expected.into(),
            actual: Some(token.text.clone()),
            actual_pos: Some(token.pos.clone()),
        }
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
    /// 未命中时最近一条受阻模式及其决策前沿；命中时为空。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frontier: Option<EpisodeFrontier>,
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
        if let (Some(left), Some(right)) = (self.last_move.as_deref(), context.last_move.as_deref())
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
        "是", "是的", "对", "对的", "没错", "确认", "好", "好的", "好呀", "可以", "嗯", "yes", "y",
        "true",
    ];
    let reject = [
        "不是", "不对", "没有", "否", "不", "不要", "不用", "别", "错", "算了", "no", "n", "false",
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
    /// 进程内变更计数（不序列化）：供调用方判断是否需要增量落图。
    #[serde(skip)]
    revision: u64,
}
fn default_episode_capacity() -> usize {
    64
}
impl Default for PatternEpisodeLog {
    fn default() -> Self {
        Self::with_capacity(default_episode_capacity())
    }
}

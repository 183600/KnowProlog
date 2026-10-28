//! SSG：词义感知的同义词图（Sense-aware Synonym Graph）。
//!
//! 扁平映射 `surface -> canonical` 在多义词上必然过度合并：只要配置里写了
//! 「苹果 = 苹果公司」，任何语境下的「苹果」都会被改写。本模块把同义关系
//! 升成三元关系 `(词面, 词义, 规范名)`，并把证据、出处、作用域与生命周期
//! 一起记在独立命名图里：
//!
//!   - **理解有方向**：词面 → 规范名；候选分数接近或缺少语境时返回
//!     `Ambiguous`，保持原词面，交给实体链接器或澄清流程；
//!   - **生成有反向**：规范名 → 用户词面；优先复用会话回声，其次已确认
//!     别名，最后回退规范名；每个反向选择都要能再次解析回同一规范名；
//!   - **证据可审计**：每条边带观察次数、来源、置信度、词义与墓碑；
//!   - **持久化隔离**：写入 `kg:graphs/synonym-graph`，不污染知识图谱。
//!
//! 边界：本模块只做读模型与证据日志，不取代实体注册表的权威地位；全局
//! 别名晋升仍必须走 `entity_registry` 的提名/确认流程。默认只消费
//! `Confirmed` 条目，`Proposed` 需要显式打开。

use std::collections::{BTreeMap, BTreeSet, HashMap};

use oxigraph::model::{GraphName, Literal, NamedNode, Quad, Term};
use serde::{Deserialize, Serialize};

/// 同义词图格式版本。
pub const SYNONYM_GRAPH_VERSION: u32 = 1;
/// 同义词图命名图（与知识图谱默认图隔离）。
pub const SYNONYM_GRAPH_IRI: &str = "http://example.org/kg/graphs/synonym-graph";
/// 同义词账本命名空间。
pub const SYNONYM_LEDGER_NS: &str = "http://example.org/kg/ledger/synonym/";
/// 默认条目容量。
pub const DEFAULT_SYNONYM_CAPACITY: usize = 512;
/// 默认最低解析置信度。
pub const DEFAULT_RESOLVE_CONFIDENCE: f64 = 0.5;
/// 默认歧义间距：最高分与次高分相差小于它时保留原词面。
pub const DEFAULT_AMBIGUITY_MARGIN: f64 = 0.12;
/// 单个词面/规范名最大字符数。
pub const MAX_SYNONYM_CHARS: usize = 64;

const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
const ENTRY_CLASS: &str = "http://example.org/kg/ledger/synonym/SynonymEntry";
const STAMP_CLASS: &str = "http://example.org/kg/ledger/synonym/Stamp";
const STAMP_SUBJECT: &str = "http://example.org/kg/ledger/synonym/stamp";

const P_JSON: &str = "http://example.org/kg/ledger/synonym/json";
const P_SURFACE: &str = "http://example.org/kg/ledger/synonym/surface";
const P_CANONICAL: &str = "http://example.org/kg/ledger/synonym/canonical";
const P_KIND: &str = "http://example.org/kg/ledger/synonym/kind";
const P_STATUS: &str = "http://example.org/kg/ledger/synonym/status";
const P_SCOPE: &str = "http://example.org/kg/ledger/synonym/scope";
const P_CONFIDENCE: &str = "http://example.org/kg/ledger/synonym/confidence";
const P_OBSERVATIONS: &str = "http://example.org/kg/ledger/synonym/observations";
const P_REFUTATIONS: &str = "http://example.org/kg/ledger/synonym/refutations";
const P_FIRST_TURN: &str = "http://example.org/kg/ledger/synonym/firstTurn";
const P_LAST_TURN: &str = "http://example.org/kg/ledger/synonym/lastTurn";
const P_SENSE_TYPE: &str = "http://example.org/kg/ledger/synonym/senseType";
const P_SENSE_DOMAIN: &str = "http://example.org/kg/ledger/synonym/senseDomain";
const P_SENSE_RANGE: &str = "http://example.org/kg/ledger/synonym/senseRange";
const P_SOURCES: &str = "http://example.org/kg/ledger/synonym/sources";
const P_VERSION: &str = "http://example.org/kg/ledger/synonym/version";
const P_FINGERPRINT: &str = "http://example.org/kg/ledger/synonym/fingerprint";
const P_CAPACITY: &str = "http://example.org/kg/ledger/synonym/capacity";
const P_COUNT: &str = "http://example.org/kg/ledger/synonym/count";

type SynonymResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// 同义关系作用的层级。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SynonymKind {
    Entity,
    Predicate,
    Type,
}

impl SynonymKind {
    pub fn label(self) -> &'static str {
        match self {
            SynonymKind::Entity => "实体",
            SynonymKind::Predicate => "谓词",
            SynonymKind::Type => "类型",
        }
    }

    fn key(self) -> &'static str {
        match self {
            SynonymKind::Entity => "e",
            SynonymKind::Predicate => "p",
            SynonymKind::Type => "t",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "entity" | "e" | "实体" => Some(SynonymKind::Entity),
            "predicate" | "p" | "谓词" => Some(SynonymKind::Predicate),
            "type" | "t" | "类型" => Some(SynonymKind::Type),
            _ => None,
        }
    }
}

/// 条目生命周期。被否证的条目保留为墓碑，防止同一错误自动复活。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SynonymStatus {
    Proposed,
    Confirmed,
    Refuted,
}

impl Default for SynonymStatus {
    fn default() -> Self {
        SynonymStatus::Proposed
    }
}

impl SynonymStatus {
    pub fn label(self) -> &'static str {
        match self {
            SynonymStatus::Proposed => "待确认",
            SynonymStatus::Confirmed => "已确认",
            SynonymStatus::Refuted => "已否证",
        }
    }
}

/// 作用域：全局证据或仅本会话。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SynonymScope {
    Global,
    Session,
}

impl Default for SynonymScope {
    fn default() -> Self {
        SynonymScope::Global
    }
}

impl SynonymScope {
    pub fn label(self) -> &'static str {
        match self {
            SynonymScope::Global => "全局",
            SynonymScope::Session => "会话",
        }
    }
}

/// 词义约束：实体类型、谓词定义域/值域。缺省表示通配，不阻塞解析。
#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SynonymSense {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub type_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub domain: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub range: Option<String>,
}

impl SynonymSense {
    pub fn new(
        type_name: Option<&str>,
        domain: Option<&str>,
        range: Option<&str>,
    ) -> Self {
        Self {
            type_name: clean_optional(type_name),
            domain: clean_optional(domain),
            range: clean_optional(range),
        }
    }

    pub fn type_only(type_name: &str) -> Self {
        Self::new(Some(type_name), None, None)
    }

    pub fn is_empty(&self) -> bool {
        self.type_name.is_none() && self.domain.is_none() && self.range.is_none()
    }

    pub fn label(&self) -> String {
        if self.is_empty() {
            return "通配词义".to_string();
        }
        let mut parts = Vec::new();
        if let Some(type_name) = &self.type_name {
            parts.push(format!("类型 {}", type_name));
        }
        if let Some(domain) = &self.domain {
            parts.push(format!("定义域 {}", domain));
        }
        if let Some(range) = &self.range {
            parts.push(format!("值域 {}", range));
        }
        parts.join("，")
    }

    fn key(&self) -> String {
        format!(
            "{}|{}|{}",
            self.type_name.as_deref().unwrap_or(""),
            self.domain.as_deref().unwrap_or(""),
            self.range.as_deref().unwrap_or("")
        )
    }
}

/// 解析语境：只携带符号证据，不训练、不猜测。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SynonymContext {
    /// 当前槽位期待的类型（来自谓词画像、焦点实体或类型体系）。
    pub expected_type: Option<String>,
    /// 当前谓词（可用于匹配词义定义域）。
    pub predicate: Option<String>,
    /// 显式定义域（优先于谓词名）。
    pub domain: Option<String>,
    /// 显式值域。
    pub range: Option<String>,
    /// 是否把待确认条目也当作候选（默认否）。
    pub include_proposed: bool,
    /// 最低置信度。
    pub min_confidence: f64,
    /// 歧义间距。
    pub ambiguity_margin: f64,
}

impl SynonymContext {
    pub fn new() -> Self {
        Self {
            min_confidence: DEFAULT_RESOLVE_CONFIDENCE,
            ambiguity_margin: DEFAULT_AMBIGUITY_MARGIN,
            ..Self::default()
        }
    }

    pub fn with_expected_type(mut self, type_name: &str) -> Self {
        self.expected_type = clean_optional(Some(type_name));
        self
    }

    pub fn with_predicate(mut self, predicate: &str) -> Self {
        self.predicate = clean_optional(Some(predicate));
        self
    }

    pub fn with_domain(mut self, domain: &str) -> Self {
        self.domain = clean_optional(Some(domain));
        self
    }

    pub fn with_range(mut self, range: &str) -> Self {
        self.range = clean_optional(Some(range));
        self
    }

    pub fn including_proposed(mut self) -> Self {
        self.include_proposed = true;
        self
    }

    pub fn with_margin(mut self, margin: f64) -> Self {
        self.ambiguity_margin = margin.clamp(0.0, 1.0);
        self
    }

    fn active_domain(&self) -> Option<&str> {
        self.domain
            .as_deref()
            .or(self.predicate.as_deref())
    }
}

/// 一条同义边。`surface == canonical` 时是带词义的自反条目，只用于标注
/// 「规范名本身在某个词义下成立」，不会触发任何改写。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SynonymEntry {
    pub id: String,
    pub surface: String,
    pub canonical: String,
    pub kind: SynonymKind,
    #[serde(default, skip_serializing_if = "SynonymSense::is_empty")]
    pub sense: SynonymSense,
    pub status: SynonymStatus,
    #[serde(default)]
    pub scope: SynonymScope,
    pub confidence: f64,
    pub observations: u32,
    #[serde(default)]
    pub refutations: u32,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub sources: BTreeSet<String>,
    pub first_turn: u64,
    pub last_turn: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

impl SynonymEntry {
    pub fn is_confirmed(&self) -> bool {
        self.status == SynonymStatus::Confirmed
    }

    pub fn is_refuted(&self) -> bool {
        self.status == SynonymStatus::Refuted
    }

    pub fn is_self_entry(&self) -> bool {
        self.surface == self.canonical
    }

    pub fn describe(&self) -> String {
        format!(
            "「{}」→「{}」（{}，{}，{}，{:.2}）",
            self.surface,
            self.canonical,
            self.kind.label(),
            self.sense.label(),
            self.status.label(),
            self.confidence
        )
    }

    fn support_bonus(&self) -> f64 {
        f64::from(self.observations.min(20)) * 0.01
    }

    fn base_score(&self) -> f64 {
        (self.confidence + self.support_bonus()).clamp(0.0, 1.5)
    }
}

/// 一次解析结果。
#[derive(Debug, Clone, PartialEq)]
pub enum SynonymResolution {
    /// 唯一且足够可信：可以改写为 `canonical`。
    Resolved(SynonymReading),
    /// 多个词义分数接近：保持原词面，交给澄清流程。
    Ambiguous {
        surface: String,
        kind: SynonymKind,
        candidates: Vec<SynonymReading>,
    },
    /// 不改写，并给出确定性原因。
    Unresolved {
        surface: String,
        kind: SynonymKind,
        reason: UnresolvedReason,
    },
}

impl SynonymResolution {
    pub fn resolved(&self) -> Option<&SynonymReading> {
        match self {
            SynonymResolution::Resolved(reading) => Some(reading),
            _ => None,
        }
    }

    pub fn canonical(&self) -> Option<&str> {
        self.resolved().map(|reading| reading.canonical.as_str())
    }

    pub fn is_ambiguous(&self) -> bool {
        matches!(self, SynonymResolution::Ambiguous { .. })
    }

    pub fn surface(&self) -> &str {
        match self {
            SynonymResolution::Resolved(reading) => &reading.surface,
            SynonymResolution::Ambiguous { surface, .. }
            | SynonymResolution::Unresolved { surface, .. } => surface,
        }
    }
}

/// 不改写原因。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnresolvedReason {
    EmptySurface,
    NoEntry,
    RefutedOnly,
    CanonicalSurface,
    SenseMismatch,
    BelowConfidence,
}

impl UnresolvedReason {
    pub fn label(self) -> &'static str {
        match self {
            UnresolvedReason::EmptySurface => "词面为空",
            UnresolvedReason::NoEntry => "没有同义证据",
            UnresolvedReason::RefutedOnly => "只有已否证条目",
            UnresolvedReason::CanonicalSurface => "词面本身是规范名",
            UnresolvedReason::SenseMismatch => "词义与语境不兼容",
            UnresolvedReason::BelowConfidence => "置信度不足",
        }
    }
}

/// 一次可审计的解析读法。
#[derive(Debug, Clone, PartialEq)]
pub struct SynonymReading {
    pub entry_id: String,
    pub surface: String,
    pub canonical: String,
    pub kind: SynonymKind,
    pub sense: SynonymSense,
    pub status: SynonymStatus,
    pub score: f64,
    pub confidence: f64,
    pub observations: u32,
    pub reason: String,
}

impl SynonymReading {
    pub fn describe(&self) -> String {
        format!(
            "「{}」→「{}」（{}，{}，得分 {:.2}）",
            self.surface,
            self.canonical,
            self.kind.label(),
            self.sense.label(),
            self.score
        )
    }

    /// 解析结论是否真的改变词面（自反条目只标注词义，不改写）。
    pub fn changes_surface(&self) -> bool {
        self.surface != self.canonical
    }
}

/// 反向生成选择。
#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceChoice {
    pub canonical: String,
    pub surface: String,
    pub source: String,
    pub reversible: bool,
}

/// 观察结果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObserveOutcome {
    Added { id: String },
    Reinforced { id: String },
    BlockedByRefutation { id: String },
    Rejected { reason: &'static str },
}

impl ObserveOutcome {
    pub fn id(&self) -> Option<&str> {
        match self {
            ObserveOutcome::Added { id }
            | ObserveOutcome::Reinforced { id }
            | ObserveOutcome::BlockedByRefutation { id } => Some(id),
            ObserveOutcome::Rejected { .. } => None,
        }
    }

    pub fn changed(&self) -> bool {
        matches!(
            self,
            ObserveOutcome::Added { .. } | ObserveOutcome::Reinforced { .. }
        )
    }
}

/// 冲突报告：同一词面、同一层级、词义不互斥却指向不同规范名。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SynonymConflict {
    pub surface: String,
    pub kind: SynonymKind,
    pub canonicals: BTreeSet<String>,
    pub entry_ids: BTreeSet<String>,
}

/// 词义感知的同义词图。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SynonymGraph {
    pub version: u32,
    pub turn: u64,
    pub revision: u64,
    pub capacity: usize,
    pub entries: BTreeMap<String, SynonymEntry>,
}

impl Default for SynonymGraph {
    fn default() -> Self {
        Self::new(DEFAULT_SYNONYM_CAPACITY)
    }
}

impl SynonymGraph {
    pub fn new(capacity: usize) -> Self {
        Self {
            version: SYNONYM_GRAPH_VERSION,
            turn: 0,
            revision: 0,
            capacity: capacity.max(1),
            entries: BTreeMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn begin_turn(&mut self, turn: u64) {
        self.turn = self.turn.max(turn);
    }

    pub fn entry(&self, id: &str) -> Option<&SynonymEntry> {
        self.entries.get(id)
    }

    pub fn entries(&self) -> impl Iterator<Item = &SynonymEntry> {
        self.entries.values()
    }

    pub fn active_len(&self) -> usize {
        self.entries
            .values()
            .filter(|entry| entry.is_confirmed())
            .count()
    }

    /// 观察一次「词面 ↔ 规范名」。同一 `(词面, 规范名, 层级, 词义)` 累加证据；
    /// 不同词义各自成边，绝不做无上下文的硬合并。
    pub fn observe(
        &mut self,
        turn: u64,
        surface: &str,
        canonical: &str,
        kind: SynonymKind,
        sense: SynonymSense,
        origin: &str,
        confidence: f64,
    ) -> ObserveOutcome {
        self.begin_turn(turn);
        let Some(surface) = clean_surface(surface) else {
            return ObserveOutcome::Rejected {
                reason: "词面为空或过长",
            };
        };
        let Some(canonical) = clean_surface(canonical) else {
            return ObserveOutcome::Rejected {
                reason: "规范名为空或过长",
            };
        };
        if surface == canonical && sense.is_empty() {
            return ObserveOutcome::Rejected {
                reason: "自反条目必须带词义",
            };
        }
        let id = entry_id(kind, &surface, &canonical, &sense);
        if let Some(existing) = self.entries.get_mut(&id) {
            if existing.is_refuted() {
                return ObserveOutcome::BlockedByRefutation { id };
            }
            let old_observations = existing.observations.max(1);
            let incoming = confidence.clamp(0.0, 1.0);
            existing.confidence = ((existing.confidence * f64::from(old_observations))
                + incoming)
                / f64::from(old_observations + 1);
            existing.confidence = existing.confidence.clamp(0.0, 1.0);
            existing.observations = existing.observations.saturating_add(1);
            existing.last_turn = turn;
            if !origin.trim().is_empty() {
                existing.sources.insert(origin.trim().to_string());
            }
            self.revision = self.revision.saturating_add(1);
            return ObserveOutcome::Reinforced { id };
        }
        let entry = SynonymEntry {
            id: id.clone(),
            surface,
            canonical,
            kind,
            sense,
            status: SynonymStatus::Proposed,
            scope: SynonymScope::Global,
            confidence: confidence.clamp(0.0, 1.0),
            observations: 1,
            refutations: 0,
            sources: origin
                .trim()
                .is_empty()
                .then(BTreeSet::new)
                .unwrap_or_else(|| BTreeSet::from([origin.trim().to_string()])),
            first_turn: turn,
            last_turn: turn,
            note: None,
        };
        self.entries.insert(id.clone(), entry);
        self.revision = self.revision.saturating_add(1);
        self.prune();
        ObserveOutcome::Added { id }
    }

    /// 从现有配置资产编译同义词图：扁平同义词、实体画像、谓词画像与类型
    /// 别名。画像里的类型/定义域/值域直接成为词义约束，不需要新增配置段。
    ///
    /// 扁平映射只播种为实体层：谓词别名由 `predicate_profiles` 提供定义域/
    /// 值域，避免同一个词面在谓词层被无词义地重复声明。
    pub fn from_config(config: &crate::SystemConfig, turn: u64) -> Self {
        let mut graph = Self::default();
        if let Some(synonyms) = &config.synonyms {
            let mut pairs: Vec<(&String, &String)> = synonyms.mappings.iter().collect();
            pairs.sort();
            for (surface, canonical) in pairs {
                if let ObserveOutcome::Added { id } | ObserveOutcome::Reinforced { id } = graph
                    .observe(
                        turn,
                        surface,
                        canonical,
                        SynonymKind::Entity,
                        SynonymSense::default(),
                        "config",
                        1.0,
                    )
                {
                    graph.confirm(&id, turn);
                }
            }
        }
        for profile in &config.entity_profiles {
            let senses = senses_for(&profile.etype, &[], &[]);
            for sense in senses {
                if let ObserveOutcome::Added { id } | ObserveOutcome::Reinforced { id } = graph
                    .observe(
                        turn,
                        &profile.name,
                        &profile.name,
                        SynonymKind::Entity,
                        sense.clone(),
                        "entity_profile",
                        1.0,
                    )
                {
                    graph.confirm(&id, turn);
                }
                for alias in &profile.aliases {
                    if let ObserveOutcome::Added { id } | ObserveOutcome::Reinforced { id } =
                        graph.observe(
                            turn,
                            alias,
                            &profile.name,
                            SynonymKind::Entity,
                            sense.clone(),
                            "entity_profile",
                            1.0,
                        )
                    {
                        graph.confirm(&id, turn);
                    }
                }
            }
        }
        for profile in &config.predicate_profiles {
            let senses = senses_for(&[], &profile.domain, &profile.range);
            for sense in senses {
                if let ObserveOutcome::Added { id } | ObserveOutcome::Reinforced { id } = graph
                    .observe(
                        turn,
                        &profile.name,
                        &profile.name,
                        SynonymKind::Predicate,
                        sense.clone(),
                        "predicate_profile",
                        1.0,
                    )
                {
                    graph.confirm(&id, turn);
                }
                for alias in &profile.aliases {
                    if let ObserveOutcome::Added { id } | ObserveOutcome::Reinforced { id } =
                        graph.observe(
                            turn,
                            alias,
                            &profile.name,
                            SynonymKind::Predicate,
                            sense.clone(),
                            "predicate_profile",
                            1.0,
                        )
                    {
                        graph.confirm(&id, turn);
                    }
                }
            }
        }
        for alias in &config.type_system.aliases {
            if let ObserveOutcome::Added { id } | ObserveOutcome::Reinforced { id } = graph
                .observe(
                    turn,
                    &alias.alias,
                    &alias.canonical,
                    SynonymKind::Type,
                    SynonymSense::default(),
                    "type_system",
                    1.0,
                )
            {
                graph.confirm(&id, turn);
            }
        }
        graph
    }

    /// 从扁平映射播种：配置里的映射视为显式承诺（已确认、置信度 1.0）。
    /// 若多个映射把同一词面指向不同规范名，解析时会自然得到 `Ambiguous`。
    pub fn from_flat_mappings(
        mappings: &HashMap<String, String>,
        kind: SynonymKind,
        turn: u64,
    ) -> Self {
        let mut graph = Self::default();
        let mut pairs: Vec<(&String, &String)> = mappings.iter().collect();
        pairs.sort();
        for (surface, canonical) in pairs {
            if let ObserveOutcome::Added { id } | ObserveOutcome::Reinforced { id } = graph
                .observe(
                    turn,
                    surface,
                    canonical,
                    kind,
                    SynonymSense::default(),
                    "config",
                    1.0,
                )
            {
                graph.confirm(&id, turn);
            }
        }
        graph
    }

    /// 确认一条边（显式承诺）。
    pub fn confirm(&mut self, id: &str, turn: u64) -> bool {
        self.begin_turn(turn);
        let Some(entry) = self.entries.get_mut(id) else {
            return false;
        };
        if entry.is_refuted() {
            return false;
        }
        let changed = entry.status != SynonymStatus::Confirmed;
        entry.status = SynonymStatus::Confirmed;
        entry.last_turn = turn;
        if changed {
            self.revision = self.revision.saturating_add(1);
        }
        changed
    }

    /// 否证一条边：保留墓碑与原因，观察不会自动复活它。
    pub fn refute(&mut self, id: &str, turn: u64, note: Option<&str>) -> bool {
        self.begin_turn(turn);
        let Some(entry) = self.entries.get_mut(id) else {
            return false;
        };
        entry.status = SynonymStatus::Refuted;
        entry.refutations = entry.refutations.saturating_add(1);
        entry.last_turn = turn;
        if let Some(note) = note.map(str::trim).filter(|note| !note.is_empty()) {
            entry.note = Some(note.to_string());
        }
        self.revision = self.revision.saturating_add(1);
        true
    }

    /// 按 `(词面, 规范名, 层级)` 查找并确认第一条匹配边。
    pub fn confirm_pair(
        &mut self,
        surface: &str,
        canonical: &str,
        kind: SynonymKind,
        turn: u64,
    ) -> Option<String> {
        let id = self
            .entries
            .values()
            .find(|entry| {
                entry.surface == surface.trim()
                    && entry.canonical == canonical.trim()
                    && entry.kind == kind
            })
            .map(|entry| entry.id.clone())?;
        self.confirm(&id, turn).then_some(id)
    }

    /// 词面 → 规范名（默认类型兼容：大小写无关的相等）。
    pub fn canonicalize(
        &self,
        surface: &str,
        kind: SynonymKind,
        context: &SynonymContext,
    ) -> SynonymResolution {
        self.canonicalize_with(surface, kind, context, |left, right| {
            normalize_type(left) == normalize_type(right)
        })
    }

    /// 词面 → 规范名，允许调用方注入类型体系兼容关系（子类/别名）。
    ///
    /// 规则：
    ///   1. 只消费已确认条目；`include_proposed` 打开后才把待确认条目当候选；
    ///   2. 语境给出的类型/定义域/值域与词义硬冲突的条目直接淘汰；
    ///   3. 最高分低于阈值 → `Unresolved`；与次高分间距小于阈值且规范名
    ///      不同 → `Ambiguous`，保持原词面。
    pub fn canonicalize_with<F>(
        &self,
        surface: &str,
        kind: SynonymKind,
        context: &SynonymContext,
        type_compatible: F,
    ) -> SynonymResolution
    where
        F: Fn(&str, &str) -> bool,
    {
        let Some(surface) = clean_surface(surface) else {
            return SynonymResolution::Unresolved {
                surface: surface.to_string(),
                kind,
                reason: UnresolvedReason::EmptySurface,
            };
        };
        let mut candidates: Vec<(&SynonymEntry, f64)> = Vec::new();
        let mut refuted = false;
        for entry in self.entries.values() {
            if entry.surface != surface || entry.kind != kind {
                continue;
            }
            if entry.is_refuted() {
                refuted = true;
                continue;
            }
            if !entry.is_confirmed() && !context.include_proposed {
                continue;
            }
            let Some(score) = sense_score(entry, context, &type_compatible) else {
                continue;
            };
            candidates.push((entry, score));
        }
        if candidates.is_empty() {
            if refuted {
                return SynonymResolution::Unresolved {
                    surface: surface.clone(),
                    kind,
                    reason: UnresolvedReason::RefutedOnly,
                };
            }
            let canonical = self.entries.values().any(|entry| {
                entry.kind == kind
                    && entry.canonical == surface
                    && !entry.is_refuted()
            });
            return SynonymResolution::Unresolved {
                surface: surface.clone(),
                kind,
                reason: if canonical {
                    UnresolvedReason::CanonicalSurface
                } else {
                    UnresolvedReason::NoEntry
                },
            };
        }
        candidates.sort_by(|(left, left_score), (right, right_score)| {
            right_score
                .partial_cmp(left_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| right.observations.cmp(&left.observations))
                .then_with(|| left.id.cmp(&right.id))
        });
        let (best_entry, best_score) = candidates[0];
        if best_score < context.min_confidence {
            return SynonymResolution::Unresolved {
                surface: surface.clone(),
                kind,
                reason: UnresolvedReason::BelowConfidence,
            };
        }
        let contenders: Vec<(&SynonymEntry, f64)> = candidates
            .iter()
            .copied()
            .filter(|(_, score)| best_score - *score < context.ambiguity_margin)
            .collect();
        let mut canonicals: BTreeSet<&str> = BTreeSet::new();
        for (entry, _) in &contenders {
            canonicals.insert(entry.canonical.as_str());
        }
        if canonicals.len() > 1 {
            let mut readings: Vec<SynonymReading> = contenders
                .iter()
                .map(|(entry, score)| reading_of(entry, *score, "多词义分数接近"))
                .collect();
            readings.sort_by(|left, right| {
                right
                    .score
                    .partial_cmp(&left.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| right.observations.cmp(&left.observations))
                    .then_with(|| left.entry_id.cmp(&right.entry_id))
            });
            readings.truncate(4);
            return SynonymResolution::Ambiguous {
                surface: surface.clone(),
                kind,
                candidates: readings,
            };
        }
        let reason = format!(
            "词义匹配：{}（支持 {}，置信度 {:.2}）",
            best_entry.sense.label(),
            best_entry.observations,
            best_entry.confidence
        );
        SynonymResolution::Resolved(reading_of(best_entry, best_score, &reason))
    }

    /// 规范名 → 用户词面。顺序：会话回声 > 已确认别名 > 规范名。
    ///
    /// 每个候选都必须通过一次反向解析（可逆守卫）；回不来的候选直接弃权。
    pub fn surface_for(
        &self,
        canonical: &str,
        kind: SynonymKind,
        context: &SynonymContext,
        echoes: &[String],
    ) -> SurfaceChoice {
        let canonical = canonical.trim();
        let mut echo_candidates: Vec<(String, f64)> = Vec::new();
        for echo in echoes {
            let Some(echo) = clean_surface(echo) else {
                continue;
            };
            if echo == canonical {
                continue;
            }
            if let Some(reading) = self
                .canonicalize(&echo, kind, context)
                .resolved()
                .filter(|reading| reading.canonical == canonical)
            {
                echo_candidates.push((echo, reading.score));
            }
        }
        echo_candidates.sort_by(|(left_surface, left_score), (right_surface, right_score)| {
            right_score
                .partial_cmp(left_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left_surface.chars().count().cmp(&right_surface.chars().count()))
                .then_with(|| left_surface.cmp(right_surface))
        });
        if let Some((surface, _)) = echo_candidates.into_iter().next() {
            return SurfaceChoice {
                canonical: canonical.to_string(),
                surface,
                source: "会话回声".to_string(),
                reversible: true,
            };
        }
        let mut alias_candidates: Vec<(&SynonymEntry, f64)> = Vec::new();
        for entry in self.entries.values() {
            if entry.kind != kind
                || !entry.is_confirmed()
                || entry.is_self_entry()
                || entry.canonical != canonical
            {
                continue;
            }
            let Some(score) = sense_score(entry, context, &|left, right| {
                normalize_type(left) == normalize_type(right)
            }) else {
                continue;
            };
            if self
                .canonicalize(&entry.surface, kind, context)
                .resolved()
                .map(|reading| reading.canonical == canonical)
                .unwrap_or(false)
            {
                alias_candidates.push((entry, score));
            }
        }
        alias_candidates.sort_by(|(left, left_score), (right, right_score)| {
            right_score
                .partial_cmp(left_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| right.observations.cmp(&left.observations))
                .then_with(|| {
                    left.surface
                        .chars()
                        .count()
                        .cmp(&right.surface.chars().count())
                })
                .then_with(|| left.id.cmp(&right.id))
        });
        if let Some((entry, _)) = alias_candidates.into_iter().next() {
            return SurfaceChoice {
                canonical: canonical.to_string(),
                surface: entry.surface.clone(),
                source: format!(
                    "已确认别名（{}，支持 {}）",
                    entry.sense.label(),
                    entry.observations
                ),
                reversible: true,
            };
        }
        SurfaceChoice {
            canonical: canonical.to_string(),
            surface: canonical.to_string(),
            source: "规范名".to_string(),
            reversible: true,
        }
    }

    /// 词义不互斥却指向不同规范名的冲突（过度合并风险的审计视图）。
    pub fn conflicts(&self) -> Vec<SynonymConflict> {
        let mut grouped: BTreeMap<(String, SynonymKind), Vec<&SynonymEntry>> = BTreeMap::new();
        for entry in self.entries.values() {
            if !entry.is_confirmed() {
                continue;
            }
            grouped
                .entry((entry.surface.clone(), entry.kind))
                .or_default()
                .push(entry);
        }
        let mut conflicts = Vec::new();
        for ((surface, kind), entries) in grouped {
            let mut conflicting: Vec<&SynonymEntry> = Vec::new();
            for (index, entry) in entries.iter().enumerate() {
                for other in entries.iter().skip(index + 1) {
                    if entry.canonical != other.canonical
                        && senses_overlap(&entry.sense, &other.sense)
                    {
                        conflicting.push(entry);
                        conflicting.push(other);
                    }
                }
            }
            if conflicting.is_empty() {
                continue;
            }
            let mut canonicals = BTreeSet::new();
            let mut entry_ids = BTreeSet::new();
            for entry in conflicting {
                canonicals.insert(entry.canonical.clone());
                entry_ids.insert(entry.id.clone());
            }
            conflicts.push(SynonymConflict {
                surface,
                kind,
                canonicals,
                entry_ids,
            });
        }
        conflicts
    }

    /// 容量有界：先淘汰墓碑，再淘汰低证据的待确认条目，最后淘汰最久未用。
    pub fn prune(&mut self) {
        while self.entries.len() > self.capacity {
            let Some(id) = self
                .entries
                .values()
                .min_by(|left, right| {
                    status_rank(left.status)
                        .cmp(&status_rank(right.status))
                        .then_with(|| left.observations.cmp(&right.observations))
                        .then_with(|| left.last_turn.cmp(&right.last_turn))
                        .then_with(|| left.id.cmp(&right.id))
                })
                .map(|entry| entry.id.clone())
            else {
                break;
            };
            self.entries.remove(&id);
        }
    }

    /// 确定性指纹（用于账本戳与回归断言）。
    pub fn fingerprint(&self) -> String {
        let mut lines = Vec::with_capacity(self.entries.len());
        for entry in self.entries.values() {
            let sources: Vec<&str> = entry.sources.iter().map(String::as_str).collect();
            lines.push(format!(
                "{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}",
                entry.id,
                entry.surface,
                entry.canonical,
                entry.kind.key(),
                entry.status.label(),
                entry.scope.label(),
                entry.sense.key(),
                entry.confidence,
                entry.observations,
                entry.refutations,
                entry.last_turn,
                sources.join(",")
            ));
        }
        lines.sort();
        short_hash(&lines.join("\n"))
    }

    /// 生成命名图四元组（含物化戳）。
    pub fn to_quads(&self) -> SynonymResult<Vec<Quad>> {
        let graph = GraphName::NamedNode(NamedNode::new(SYNONYM_GRAPH_IRI)?);
        let rdf_type = NamedNode::new(RDF_TYPE)?;
        let entry_class = Term::NamedNode(NamedNode::new(ENTRY_CLASS)?);
        let mut quads = Vec::with_capacity(self.entries.len() * 8 + 6);
        for entry in self.entries.values() {
            let subject =
                NamedNode::new(format!("{}entry/{}", SYNONYM_LEDGER_NS, entry.id))?;
            quads.push(Quad::new(
                subject.clone(),
                rdf_type.clone(),
                entry_class.clone(),
                graph.clone(),
            ));
            quads.push(literal_quad(
                &subject,
                P_JSON,
                &serde_json::to_string(entry)?,
                &graph,
            )?);
            quads.push(literal_quad(&subject, P_SURFACE, &entry.surface, &graph)?);
            quads.push(literal_quad(&subject, P_CANONICAL, &entry.canonical, &graph)?);
            quads.push(literal_quad(
                &subject,
                P_KIND,
                entry.kind.key(),
                &graph,
            )?);
            quads.push(literal_quad(
                &subject,
                P_STATUS,
                &format!("{:?}", entry.status),
                &graph,
            )?);
            quads.push(literal_quad(
                &subject,
                P_SCOPE,
                &format!("{:?}", entry.scope),
                &graph,
            )?);
            quads.push(literal_quad(
                &subject,
                P_CONFIDENCE,
                &format!("{:.6}", entry.confidence),
                &graph,
            )?);
            quads.push(literal_quad(
                &subject,
                P_OBSERVATIONS,
                &entry.observations.to_string(),
                &graph,
            )?);
            quads.push(literal_quad(
                &subject,
                P_REFUTATIONS,
                &entry.refutations.to_string(),
                &graph,
            )?);
            quads.push(literal_quad(
                &subject,
                P_FIRST_TURN,
                &entry.first_turn.to_string(),
                &graph,
            )?);
            quads.push(literal_quad(
                &subject,
                P_LAST_TURN,
                &entry.last_turn.to_string(),
                &graph,
            )?);
            if !entry.sources.is_empty() {
                let sources: Vec<&str> = entry.sources.iter().map(String::as_str).collect();
                quads.push(literal_quad(
                    &subject,
                    P_SOURCES,
                    &sources.join("\u{1f}"),
                    &graph,
                )?);
            }
            if let Some(type_name) = &entry.sense.type_name {
                quads.push(literal_quad(&subject, P_SENSE_TYPE, type_name, &graph)?);
            }
            if let Some(domain) = &entry.sense.domain {
                quads.push(literal_quad(&subject, P_SENSE_DOMAIN, domain, &graph)?);
            }
            if let Some(range) = &entry.sense.range {
                quads.push(literal_quad(&subject, P_SENSE_RANGE, range, &graph)?);
            }
        }
        let stamp = NamedNode::new(STAMP_SUBJECT)?;
        quads.push(Quad::new(
            stamp.clone(),
            rdf_type,
            Term::NamedNode(NamedNode::new(STAMP_CLASS)?),
            graph.clone(),
        ));
        quads.push(literal_quad(
            &stamp,
            P_VERSION,
            &self.version.to_string(),
            &graph,
        )?);
        quads.push(literal_quad(
            &stamp,
            P_FINGERPRINT,
            &self.fingerprint(),
            &graph,
        )?);
        quads.push(literal_quad(
            &stamp,
            P_CAPACITY,
            &self.capacity.to_string(),
            &graph,
        )?);
        quads.push(literal_quad(
            &stamp,
            P_COUNT,
            &self.entries.len().to_string(),
            &graph,
        )?);
        Ok(quads)
    }

    /// 从命名图四元组还原；损坏条目跳过，不让一条坏 JSON 毁掉整张图。
    pub fn from_quads(quads: impl Iterator<Item = Quad>) -> Self {
        let mut graph = Self::default();
        let mut capacity = None;
        for quad in quads {
            match quad.predicate.as_str() {
                P_CAPACITY => {
                    if let Term::Literal(value) = &quad.object {
                        capacity = value.value().parse::<usize>().ok();
                    }
                }
                P_JSON => {
                    let Term::Literal(value) = &quad.object else {
                        continue;
                    };
                    if let Ok(entry) = serde_json::from_str::<SynonymEntry>(value.value()) {
                        graph.turn = graph.turn.max(entry.last_turn);
                        graph
                            .entries
                            .entry(entry.id.clone())
                            .or_insert(entry);
                    }
                }
                _ => {}
            }
        }
        if let Some(capacity) = capacity {
            graph.capacity = capacity.max(1);
        }
        graph.prune();
        graph
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    pub fn from_json(text: &str) -> Option<Self> {
        let mut graph: SynonymGraph = serde_json::from_str(text).ok()?;
        if graph.version > SYNONYM_GRAPH_VERSION {
            return None;
        }
        graph.capacity = graph.capacity.max(1);
        graph.prune();
        Some(graph)
    }

    pub fn save_to_path(&self, path: &str) -> std::io::Result<()> {
        std::fs::write(path, self.to_json())
    }

    pub fn load_from_path(path: &str) -> Option<Self> {
        let text = std::fs::read_to_string(path).ok()?;
        Self::from_json(&text)
    }

    /// 人读审计报告（确定性排序）。
    pub fn render_report(&self) -> String {
        let mut out = String::new();
        let confirmed = self
            .entries
            .values()
            .filter(|entry| entry.status == SynonymStatus::Confirmed)
            .count();
        let proposed = self
            .entries
            .values()
            .filter(|entry| entry.status == SynonymStatus::Proposed)
            .count();
        let refuted = self
            .entries
            .values()
            .filter(|entry| entry.status == SynonymStatus::Refuted)
            .count();
        out.push_str(&format!(
            "同义词图：{} 条（已确认 {}，待确认 {}，已否证 {}；第 {} 轮，容量 {}）\n",
            self.entries.len(),
            confirmed,
            proposed,
            refuted,
            self.turn,
            self.capacity
        ));
        if self.entries.is_empty() {
            out.push_str("  （空）\n");
            return out;
        }
        for entry in self.entries.values().take(30) {
            out.push_str(&format!(
                "  [{}] {}，来源 {}，轮 {}-{}，观察 {}{}\n",
                entry.status.label(),
                entry.describe(),
                if entry.sources.is_empty() {
                    "未记录".to_string()
                } else {
                    entry.sources.iter().cloned().collect::<Vec<_>>().join(",")
                },
                entry.first_turn,
                entry.last_turn,
                entry.observations,
                entry
                    .note
                    .as_ref()
                    .map(|note| format!("，备注 {}", note))
                    .unwrap_or_default()
            ));
        }
        if self.entries.len() > 30 {
            out.push_str(&format!("  …（还有 {} 条）\n", self.entries.len() - 30));
        }
        let conflicts = self.conflicts();
        if conflicts.is_empty() {
            out.push_str("  冲突：无\n");
        } else {
            out.push_str(&format!("  冲突：{} 组\n", conflicts.len()));
            for conflict in conflicts.iter().take(10) {
                out.push_str(&format!(
                    "    「{}」{} → {}（条目 {}）\n",
                    conflict.surface,
                    conflict.kind.label(),
                    conflict
                        .canonicals
                        .iter()
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(" / "),
                    conflict.entry_ids.len()
                ));
            }
        }
        out
    }
}

/// 画像声明的类型/定义域/值域 → 词义集合。空声明得到通配词义；多值声明
/// 逐项成边（同一词面因此可以按语境选择不同词义）。
fn senses_for(type_names: &[String], domains: &[String], ranges: &[String]) -> Vec<SynonymSense> {
    let type_values: Vec<Option<&str>> = if type_names.is_empty() {
        vec![None]
    } else {
        type_names.iter().map(|value| Some(value.as_str())).collect()
    };
    let domain_values: Vec<Option<&str>> = if domains.is_empty() {
        vec![None]
    } else {
        domains.iter().map(|value| Some(value.as_str())).collect()
    };
    let range_values: Vec<Option<&str>> = if ranges.is_empty() {
        vec![None]
    } else {
        ranges.iter().map(|value| Some(value.as_str())).collect()
    };
    let mut senses = Vec::new();
    for type_name in &type_values {
        for domain in &domain_values {
            for range in &range_values {
                let sense = SynonymSense::new(*type_name, *domain, *range);
                if !senses.contains(&sense) {
                    senses.push(sense);
                }
            }
        }
    }
    senses
}

fn reading_of(entry: &SynonymEntry, score: f64, reason: &str) -> SynonymReading {
    SynonymReading {
        entry_id: entry.id.clone(),
        surface: entry.surface.clone(),
        canonical: entry.canonical.clone(),
        kind: entry.kind,
        sense: entry.sense.clone(),
        status: entry.status,
        score,
        confidence: entry.confidence,
        observations: entry.observations,
        reason: reason.to_string(),
    }
}

/// 词义与语境打分；硬冲突返回 `None`。
fn sense_score<F>(
    entry: &SynonymEntry,
    context: &SynonymContext,
    type_compatible: &F,
) -> Option<f64>
where
    F: Fn(&str, &str) -> bool,
{
    let mut score = entry.base_score();
    if let Some(expected) = context.expected_type.as_deref() {
        match entry.sense.type_name.as_deref() {
            Some(actual) => {
                if !type_compatible(expected, actual) {
                    return None;
                }
                score += 0.30;
            }
            None => score += 0.02,
        }
    }
    if let Some(domain) = context.active_domain() {
        match entry.sense.domain.as_deref() {
            Some(actual) => {
                if normalize_type(domain) != normalize_type(actual) {
                    return None;
                }
                score += 0.20;
            }
            None => score += 0.01,
        }
    }
    if let Some(range) = context.range.as_deref() {
        match entry.sense.range.as_deref() {
            Some(actual) => {
                if normalize_type(range) != normalize_type(actual) {
                    return None;
                }
                score += 0.12;
            }
            None => score += 0.01,
        }
    }
    Some(score)
}

fn senses_overlap(left: &SynonymSense, right: &SynonymSense) -> bool {
    let type_overlap = match (left.type_name.as_deref(), right.type_name.as_deref()) {
        (Some(left), Some(right)) => normalize_type(left) == normalize_type(right),
        _ => true,
    };
    let domain_overlap = match (left.domain.as_deref(), right.domain.as_deref()) {
        (Some(left), Some(right)) => normalize_type(left) == normalize_type(right),
        _ => true,
    };
    let range_overlap = match (left.range.as_deref(), right.range.as_deref()) {
        (Some(left), Some(right)) => normalize_type(left) == normalize_type(right),
        _ => true,
    };
    type_overlap && domain_overlap && range_overlap
}

fn status_rank(status: SynonymStatus) -> u8 {
    match status {
        SynonymStatus::Refuted => 0,
        SynonymStatus::Proposed => 1,
        SynonymStatus::Confirmed => 2,
    }
}

fn normalize_type(value: &str) -> String {
    value.trim().to_lowercase()
}

fn clean_optional(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn clean_surface(value: &str) -> Option<String> {
    let value = value.trim();
    if value.is_empty() || value.chars().count() > MAX_SYNONYM_CHARS {
        return None;
    }
    Some(value.to_string())
}

fn entry_id(kind: SynonymKind, surface: &str, canonical: &str, sense: &SynonymSense) -> String {
    format!(
        "{}-{}",
        kind.key(),
        short_hash(&format!(
            "{}\u{1f}{}\u{1f}{}",
            surface,
            canonical,
            sense.key()
        ))
    )
}

fn short_hash(text: &str) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{:016x}{:x}", hash, text.len())
}

fn literal_quad(
    subject: &NamedNode,
    predicate: &str,
    value: &str,
    graph: &GraphName,
) -> SynonymResult<Quad> {
    Ok(Quad::new(
        subject.clone(),
        NamedNode::new(predicate)?,
        Literal::new_simple_literal(value),
        graph.clone(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn add(
        graph: &mut SynonymGraph,
        surface: &str,
        canonical: &str,
        type_name: Option<&str>,
    ) -> String {
        let outcome = graph.observe(
            1,
            surface,
            canonical,
            SynonymKind::Entity,
            SynonymSense::new(type_name, None, None),
            "test",
            1.0,
        );
        let id = outcome.id().expect("应写入条目").to_string();
        graph.confirm(&id, 1);
        id
    }

    #[test]
    fn ambiguous_surface_without_context_stays_unchanged() {
        let mut graph = SynonymGraph::new(16);
        add(&mut graph, "苹果", "苹果公司", Some("组织"));
        add(&mut graph, "苹果", "苹果", Some("水果"));
        let resolution =
            graph.canonicalize("苹果", SynonymKind::Entity, &SynonymContext::new());
        assert!(resolution.is_ambiguous(), "无语境时应保留原词面");
        match resolution {
            SynonymResolution::Ambiguous { candidates, .. } => {
                assert_eq!(candidates.len(), 2);
            }
            other => panic!("预期 Ambiguous，得到 {:?}", other),
        }
    }

    #[test]
    fn context_type_selects_sense() {
        let mut graph = SynonymGraph::new(16);
        add(&mut graph, "苹果", "苹果公司", Some("组织"));
        add(&mut graph, "苹果", "苹果", Some("水果"));
        let company = graph.canonicalize(
            "苹果",
            SynonymKind::Entity,
            &SynonymContext::new().with_expected_type("组织"),
        );
        assert_eq!(company.canonical(), Some("苹果公司"));
        let fruit = graph.canonicalize(
            "苹果",
            SynonymKind::Entity,
            &SynonymContext::new().with_expected_type("水果"),
        );
        assert_eq!(fruit.canonical(), Some("苹果"));
    }

    #[test]
    fn type_system_closure_can_match_subtypes() {
        let mut graph = SynonymGraph::new(16);
        add(&mut graph, "企鹅", "腾讯", Some("公司"));
        let resolution = graph.canonicalize_with(
            "企鹅",
            SynonymKind::Entity,
            &SynonymContext::new().with_expected_type("组织"),
            |expected, actual| expected == "组织" && actual == "公司",
        );
        assert_eq!(resolution.canonical(), Some("腾讯"));
    }

    #[test]
    fn proposed_entries_are_not_consumed_by_default() {
        let mut graph = SynonymGraph::new(16);
        graph.observe(
            1,
            "阿明",
            "小明",
            SynonymKind::Entity,
            SynonymSense::default(),
            "test",
            1.0,
        );
        let hidden = graph.canonicalize("阿明", SynonymKind::Entity, &SynonymContext::new());
        assert_eq!(hidden.canonical(), None);
        let visible = graph.canonicalize(
            "阿明",
            SynonymKind::Entity,
            &SynonymContext::new().including_proposed(),
        );
        assert_eq!(visible.canonical(), Some("小明"));
    }

    #[test]
    fn self_entry_requires_sense_and_does_not_rewrite() {
        let mut graph = SynonymGraph::new(16);
        let rejected = graph.observe(
            1,
            "苹果",
            "苹果",
            SynonymKind::Entity,
            SynonymSense::default(),
            "test",
            1.0,
        );
        assert!(matches!(rejected, ObserveOutcome::Rejected { .. }));
        add(&mut graph, "苹果", "苹果", Some("水果"));
        let resolution =
            graph.canonicalize("苹果", SynonymKind::Entity, &SynonymContext::new());
        let reading = resolution.resolved().expect("自反条目应解析到自身词义");
        assert_eq!(reading.canonical, "苹果");
        assert!(!reading.changes_surface(), "自反条目不得触发改写");
    }

    #[test]
    fn refuted_entry_is_a_tombstone() {
        let mut graph = SynonymGraph::new(16);
        let id = add(&mut graph, "阿明", "小明", None);
        assert!(graph.refute(&id, 2, Some("用户纠正")));
        let resolution = graph.canonicalize("阿明", SynonymKind::Entity, &SynonymContext::new());
        assert_eq!(
            resolution,
            SynonymResolution::Unresolved {
                surface: "阿明".to_string(),
                kind: SynonymKind::Entity,
                reason: UnresolvedReason::RefutedOnly,
            }
        );
        let outcome = graph.observe(
            3,
            "阿明",
            "小明",
            SynonymKind::Entity,
            SynonymSense::default(),
            "test",
            1.0,
        );
        assert!(matches!(
            outcome,
            ObserveOutcome::BlockedByRefutation { .. }
        ));
    }

    #[test]
    fn reverse_generation_prefers_echo_and_verifies_reversibility() {
        let mut graph = SynonymGraph::new(16);
        add(&mut graph, "老婆", "妻子", None);
        let context = SynonymContext::new();
        let choice = graph.surface_for("妻子", SynonymKind::Entity, &context, &[]);
        assert_eq!(choice.surface, "老婆");
        assert!(choice.reversible);
        let echo = graph.surface_for(
            "妻子",
            SynonymKind::Entity,
            &context,
            &["爱人".to_string(), "老婆".to_string()],
        );
        assert_eq!(echo.surface, "老婆", "回声必须仍能解析回同一规范名");
        let fallback = graph.surface_for("丈夫", SynonymKind::Entity, &context, &[]);
        assert_eq!(fallback.surface, "丈夫");
        assert_eq!(fallback.source, "规范名");
    }

    #[test]
    fn reverse_generation_skips_ambiguous_alias() {
        let mut graph = SynonymGraph::new(16);
        add(&mut graph, "苹果", "苹果公司", Some("组织"));
        add(&mut graph, "苹果", "苹果", Some("水果"));
        let context = SynonymContext::new();
        let choice = graph.surface_for("苹果公司", SynonymKind::Entity, &context, &[]);
        assert_eq!(choice.surface, "苹果公司", "歧义词面不能作为安全反向选择");
    }

    #[test]
    fn conflicts_report_marks_over_merge_risk() {
        let mut graph = SynonymGraph::new(16);
        add(&mut graph, "苹果", "苹果公司", None);
        add(&mut graph, "苹果", "苹果", Some("水果"));
        let conflicts = graph.conflicts();
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].surface, "苹果");
        assert_eq!(conflicts[0].canonicals.len(), 2);
    }

    #[test]
    fn quads_round_trip_preserves_evidence() {
        let mut graph = SynonymGraph::new(32);
        let id = add(&mut graph, "老婆", "妻子", Some("人物"));
        graph.observe(
            4,
            "老婆",
            "妻子",
            SynonymKind::Entity,
            SynonymSense::new(Some("人物"), None, None),
            "dialogue",
            0.8,
        );
        let quads = graph.to_quads().expect("应能生成四元组");
        assert!(quads.iter().all(|quad| {
            quad.graph_name == GraphName::NamedNode(NamedNode::new(SYNONYM_GRAPH_IRI).unwrap())
        }));
        let restored = SynonymGraph::from_quads(quads.into_iter());
        let entry = restored.entry(&id).expect("条目应还原");
        assert_eq!(entry.observations, 2);
        assert_eq!(entry.status, SynonymStatus::Confirmed);
        assert_eq!(entry.sense.type_name.as_deref(), Some("人物"));
        assert!(entry.sources.contains("test"));
        assert!(entry.sources.contains("dialogue"));
        assert_eq!(restored.fingerprint(), graph.fingerprint());
    }

    #[test]
    fn flat_mappings_seed_confirmed_entries() {
        let mut mappings = HashMap::new();
        mappings.insert("老婆".to_string(), "妻子".to_string());
        mappings.insert("老公".to_string(), "丈夫".to_string());
        let graph = SynonymGraph::from_flat_mappings(&mappings, SynonymKind::Entity, 1);
        assert_eq!(graph.active_len(), 2);
        assert_eq!(
            graph
                .canonicalize("老婆", SynonymKind::Entity, &SynonymContext::new())
                .canonical(),
            Some("妻子")
        );
    }

    #[test]
    fn capacity_evicts_tombstones_first() {
        let mut graph = SynonymGraph::new(2);
        let first = add(&mut graph, "甲", "A", None);
        graph.refute(&first, 2, None);
        add(&mut graph, "乙", "B", None);
        add(&mut graph, "丙", "C", None);
        assert_eq!(graph.len(), 2);
        assert!(graph.entry(&first).is_none(), "墓碑应先被淘汰");
    }

    #[test]
    fn config_assets_compile_into_senses() {
        let config: crate::SystemConfig =
            toml::from_str(include_str!("../config.toml")).expect("config.toml 应可解析");
        let graph = SynonymGraph::from_config(&config, 1);
        assert!(graph.active_len() > 0);

        let entity = graph.canonicalize("老婆", SynonymKind::Entity, &SynonymContext::new());
        assert_eq!(entity.canonical(), Some("妻子"));

        let predicate =
            graph.canonicalize("老婆", SynonymKind::Predicate, &SynonymContext::new());
        assert_eq!(predicate.canonical(), Some("配偶"));

        let teacher =
            graph.canonicalize("老师", SynonymKind::Predicate, &SynonymContext::new());
        assert_eq!(teacher.canonical(), Some("师傅"));

        let type_alias =
            graph.canonicalize("Person", SynonymKind::Type, &SynonymContext::new());
        assert_eq!(type_alias.canonical(), Some("人物"));
    }

    #[test]
    fn json_round_trip_is_stable() {
        let mut graph = SynonymGraph::new(8);
        add(&mut graph, "老婆", "妻子", None);
        let json = graph.to_json();
        let restored = SynonymGraph::from_json(&json).expect("应能还原");
        assert_eq!(restored, graph);
    }
}

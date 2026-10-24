//! 镜像对齐记忆：会话级、可逆、带守卫的词形改写文法。
//!
//! 实体链接器回答「这个词面指向哪个已知身份」；本模块记住「这类说法如何
//! 读成身份」，并保证同一条规则可以正着读、反着写：
//!
//!   - 前向：`未名科技公司` --(□ + 公司)--> `未名科技`，但结果必须落到
//!     已确认身份，且通过类型守卫，否则规则不生效（落地律）；
//!   - 反向：`未名科技` --(□ + 公司)--> `未名科技公司`，生成后必须能在
//!     同一上下文下再对齐回原身份（可逆律）；
//!   - 记忆只影响本会话的理解与答句措辞；进入全局别名必须走现有的
//!     实体注册表提名/确认流程（提名律）。
//!
//! 规则形状故意保持有限：字面、前缀、后缀、包裹。字面规则携带目标身份；
//! 形状规则只描述词形，因此可以从一两个例子泛化到未登录词面，但每次
//! 应用都要重新过锚定与类型守卫，绝不由正则直接产生新身份。

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

/// 镜像对齐记忆格式版本。
pub const ALIGNMENT_MEMORY_VERSION: u32 = 1;
/// 默认规则容量。
pub const DEFAULT_ALIGNMENT_CAPACITY: usize = 128;
/// 回声词面容量（每个规范身份保留的最近用法由总容量约束）。
pub const DEFAULT_ECHO_CAPACITY: usize = 64;
/// 单个词缀最多字符数。
pub const MAX_AFFIX_CHARS: usize = 6;
/// 规范名最多字符数（扫描时限制回溯窗口）。
pub const MAX_CANONICAL_CHARS: usize = 24;
/// 整个词面最多字符数。
pub const MAX_SURFACE_CHARS: usize = 48;
const MAX_EVENTS: usize = 64;

/// 对齐对象：实体、谓词或类型。实体进入链接，谓词/类型只提供共用文法。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlignmentKind {
    Entity,
    Predicate,
    Type,
}

impl AlignmentKind {
    pub fn label(self) -> &'static str {
        match self {
            AlignmentKind::Entity => "实体",
            AlignmentKind::Predicate => "谓词",
            AlignmentKind::Type => "类型",
        }
    }
}

/// 规则作用域。跨会话晋升必须走实体注册表，不在这里静默提升。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlignmentScope {
    Session,
    Turn,
    Turns(u32),
}

impl Default for AlignmentScope {
    fn default() -> Self {
        AlignmentScope::Session
    }
}

impl AlignmentScope {
    /// 学会规则的轮次 → 过期轮次（含）。`None` 表示本会话不过期。
    pub fn expires_turn(self, learned_turn: u64) -> Option<u64> {
        match self {
            AlignmentScope::Session => None,
            AlignmentScope::Turn => Some(learned_turn),
            AlignmentScope::Turns(turns) => {
                Some(learned_turn.saturating_add(u64::from(turns.max(1)).saturating_sub(1)))
            }
        }
    }

    pub fn label(self) -> String {
        match self {
            AlignmentScope::Session => "会话".to_string(),
            AlignmentScope::Turn => "本轮".to_string(),
            AlignmentScope::Turns(turns) => format!("{} 轮", turns.max(1)),
        }
    }
}

/// 规则生命周期。被否证的字面规则保留为墓碑，防止同一错误自动复活。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlignmentStatus {
    Active,
    Refuted,
}

impl AlignmentStatus {
    pub fn label(self) -> &'static str {
        match self {
            AlignmentStatus::Active => "生效",
            AlignmentStatus::Refuted => "已否证",
        }
    }
}

/// 有限词形文法。形状规则不带身份，字面规则带目标身份。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(tag = "shape", rename_all = "snake_case")]
pub enum RewriteShape {
    Literal { surface: String },
    Prefix { prefix: String },
    Suffix { suffix: String },
    Wrapper { open: String, close: String },
}

impl RewriteShape {
    /// 从一次「词面 ↔ 规范名」观察归纳形状；不能形成稳定形状时退化为字面规则。
    pub fn from_pair(surface: &str, canonical: &str) -> Option<Self> {
        let surface = surface.trim();
        let canonical = canonical.trim();
        if surface.is_empty() || canonical.is_empty() || surface == canonical {
            return None;
        }
        let first = surface.find(canonical);
        let last = surface.rfind(canonical);
        let (Some(first), Some(last)) = (first, last) else {
            return Some(RewriteShape::Literal {
                surface: surface.to_string(),
            });
        };
        if first != last {
            return Some(RewriteShape::Literal {
                surface: surface.to_string(),
            });
        }
        let prefix = &surface[..first];
        let suffix = &surface[first + canonical.len()..];
        let prefix_chars = prefix.chars().count();
        let suffix_chars = suffix.chars().count();
        if prefix_chars == 0 && suffix_chars == 0 {
            return None;
        }
        if prefix_chars == 0 && valid_affix(suffix) {
            return Some(RewriteShape::Suffix {
                suffix: suffix.to_string(),
            });
        }
        if suffix_chars == 0 && valid_affix(prefix) {
            return Some(RewriteShape::Prefix {
                prefix: prefix.to_string(),
            });
        }
        if let Some((open, close)) = wrapper_pair(prefix, suffix) {
            return Some(RewriteShape::Wrapper { open, close });
        }
        // 两侧都有内容却不是可识别的包裹：不归纳形状，避免「关于…的」这类
        // 句法外壳被当成可泛化词缀。
        Some(RewriteShape::Literal {
            surface: surface.to_string(),
        })
    }

    /// 规范名 → 用户词面。只生成形状，不判断是否落地。
    pub fn apply_to(&self, canonical: &str) -> Option<String> {
        let canonical = canonical.trim();
        if canonical.is_empty() {
            return None;
        }
        let surface = match self {
            RewriteShape::Literal { surface } => surface.clone(),
            RewriteShape::Prefix { prefix } => format!("{}{}", prefix, canonical),
            RewriteShape::Suffix { suffix } => format!("{}{}", canonical, suffix),
            RewriteShape::Wrapper { open, close } => {
                format!("{}{}{}", open, canonical, close)
            }
        };
        (surface != canonical && surface.chars().count() <= MAX_SURFACE_CHARS).then_some(surface)
    }

    /// 用户词面 → 规范名候选（仅形状剥离；字面规则由调用方按目标处理）。
    pub fn strip<'a>(&self, surface: &'a str) -> Option<&'a str> {
        let surface = surface.trim();
        match self {
            RewriteShape::Literal { .. } => None,
            RewriteShape::Prefix { prefix } => surface
                .strip_prefix(prefix.as_str())
                .filter(|rest| !rest.is_empty()),
            RewriteShape::Suffix { suffix } => surface
                .strip_suffix(suffix.as_str())
                .filter(|rest| !rest.is_empty()),
            RewriteShape::Wrapper { open, close } => surface
                .strip_prefix(open.as_str())
                .and_then(|rest| rest.strip_suffix(close.as_str()))
                .filter(|inner| !inner.is_empty()),
        }
    }

    pub fn key(&self) -> String {
        match self {
            RewriteShape::Literal { surface } => format!("literal:{}", surface),
            RewriteShape::Prefix { prefix } => format!("prefix:{}", prefix),
            RewriteShape::Suffix { suffix } => format!("suffix:{}", suffix),
            RewriteShape::Wrapper { open, close } => format!("wrapper:{}:{}", open, close),
        }
    }

    /// 形状的符号记法：`□ + 公司` / `《 □ 》`。
    pub fn notation(&self) -> String {
        match self {
            RewriteShape::Literal { surface } => format!("「{}」", surface),
            RewriteShape::Prefix { prefix } => format!("「{}」+ □", prefix),
            RewriteShape::Suffix { suffix } => format!("□ + 「{}」", suffix),
            RewriteShape::Wrapper { open, close } => {
                format!("「{}」+ □ + 「{}」", open, close)
            }
        }
    }

    pub fn is_literal(&self) -> bool {
        matches!(self, RewriteShape::Literal { .. })
    }
}

fn valid_affix(affix: &str) -> bool {
    let chars: Vec<char> = affix.chars().collect();
    if chars.is_empty() || chars.len() > MAX_AFFIX_CHARS {
        return false;
    }
    chars.iter().all(|ch| {
        (ch.is_alphanumeric() || is_han(*ch))
            && !matches!(
                ch,
                '，' | '。'
                    | '？'
                    | '！'
                    | '、'
                    | '；'
                    | '：'
                    | '“'
                    | '”'
                    | '‘'
                    | '’'
                    | '（'
                    | '）'
                    | '《'
                    | '》'
                    | '「'
                    | '」'
                    | '『'
                    | '』'
                    | '【'
                    | '】'
                    | ' '
            )
    })
}

fn is_han(ch: char) -> bool {
    ('\u{4e00}'..='\u{9fff}').contains(&ch)
}

fn wrapper_pair(prefix: &str, suffix: &str) -> Option<(String, String)> {
    let prefix_chars: Vec<char> = prefix.chars().collect();
    let suffix_chars: Vec<char> = suffix.chars().collect();
    if prefix_chars.len() != 1 || suffix_chars.len() != 1 {
        return None;
    }
    let open = prefix_chars[0];
    let close = suffix_chars[0];
    let paired = matches!(
        (open, close),
        ('《', '》')
            | ('〈', '〉')
            | ('“', '”')
            | ('‘', '’')
            | ('「', '」')
            | ('『', '』')
            | ('(', ')')
            | ('（', '）')
            | ('[', ']')
            | ('【', '】')
            | ('<', '>')
            | ('"', '"')
            | ('\'', '\'')
    );
    paired.then(|| (open.to_string(), close.to_string()))
}

/// 一条可逆对齐规则。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AlignmentRule {
    pub id: String,
    pub shape: RewriteShape,
    /// 规则最初学自的规范身份；形状规则的泛化不改变这个锚点。
    pub target: String,
    pub kind: AlignmentKind,
    /// 类型期待（来自锚点身份的类型证据）；形状规则跨词面复用时作硬守卫。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_type: Option<String>,
    pub scope: AlignmentScope,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_turn: Option<u64>,
    pub status: AlignmentStatus,
    pub support: u32,
    pub refutations: u32,
    pub created_turn: u64,
    pub last_turn: u64,
    pub origin: String,
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub exceptions: BTreeSet<String>,
}

impl AlignmentRule {
    pub fn is_active_at(&self, turn: u64) -> bool {
        self.status == AlignmentStatus::Active
            && self.expires_turn.map_or(true, |expires| turn <= expires)
    }

    pub fn is_expired_at(&self, turn: u64) -> bool {
        self.expires_turn.map_or(false, |expires| turn > expires)
    }

    pub fn matches_literal(&self, surface: &str) -> bool {
        match &self.shape {
            RewriteShape::Literal { surface: literal } => literal == surface,
            _ => false,
        }
    }

    pub fn describe(&self) -> String {
        format!(
            "{} → {}（{}，{}）",
            self.shape.notation(),
            self.target,
            self.kind.label(),
            self.status.label()
        )
    }
}

/// 一次前向对齐结果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlignmentMatch {
    pub rule_id: String,
    pub surface: String,
    pub start: usize,
    pub end: usize,
    pub canonical: String,
    pub kind: AlignmentKind,
    pub expected_type: Option<String>,
    pub support: u32,
    pub literal: bool,
    pub reason: String,
}

impl AlignmentMatch {
    pub fn surface_chars(&self) -> usize {
        self.surface.chars().count()
    }
}

/// 一次学习/否证/清理的审计事件。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AlignmentEvent {
    pub turn: u64,
    pub action: String,
    pub rule_id: String,
    pub detail: String,
}

/// 学习结果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LearnOutcome {
    Learned { rule_id: String, created: bool },
    Refuted { rule_id: String },
    Conflict { existing: String },
    Unchanged,
}

impl LearnOutcome {
    pub fn changed(&self) -> bool {
        matches!(
            self,
            LearnOutcome::Learned { .. } | LearnOutcome::Refuted { .. }
        )
    }

    pub fn rule_id(&self) -> Option<&str> {
        match self {
            LearnOutcome::Learned { rule_id, .. }
            | LearnOutcome::Refuted { rule_id }
            | LearnOutcome::Conflict { existing: rule_id } => Some(rule_id),
            LearnOutcome::Unchanged => None,
        }
    }
}

/// 会话级镜像对齐记忆。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AlignmentMemory {
    pub version: u32,
    pub turn: u64,
    pub revision: u64,
    pub capacity: usize,
    pub rules: BTreeMap<String, AlignmentRule>,
    /// 规范身份 → 本会话用户用过的词面（最近使用排在最前）。
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub echoes: BTreeMap<String, Vec<String>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub events: Vec<AlignmentEvent>,
}

impl Default for AlignmentMemory {
    fn default() -> Self {
        Self::new(DEFAULT_ALIGNMENT_CAPACITY)
    }
}

impl AlignmentMemory {
    pub fn new(capacity: usize) -> Self {
        Self {
            version: ALIGNMENT_MEMORY_VERSION,
            turn: 0,
            revision: 0,
            capacity: capacity.max(1),
            rules: BTreeMap::new(),
            echoes: BTreeMap::new(),
            events: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.rules.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rules.is_empty() && self.echoes.is_empty()
    }

    pub fn active_len(&self, turn: u64) -> usize {
        self.rules
            .values()
            .filter(|rule| rule.is_active_at(turn))
            .count()
    }

    pub fn begin_turn(&mut self, turn: u64) {
        self.turn = self.turn.max(turn);
        self.prune();
    }

    pub fn rule(&self, id: &str) -> Option<&AlignmentRule> {
        self.rules.get(id)
    }

    pub fn active_rules(&self, turn: u64) -> Vec<&AlignmentRule> {
        self.rules
            .values()
            .filter(|rule| rule.is_active_at(turn))
            .collect()
    }

    /// 学习一次「词面 ↔ 规范身份」对齐。同形不同目标的字面规则不静默替换。
    pub fn learn_pair(
        &mut self,
        turn: u64,
        surface: &str,
        canonical: &str,
        kind: AlignmentKind,
        expected_type: Option<&str>,
        origin: &str,
        scope: AlignmentScope,
    ) -> LearnOutcome {
        self.begin_turn(turn);
        let surface = surface.trim();
        let canonical = canonical.trim();
        if surface.is_empty() || canonical.is_empty() || surface == canonical {
            return LearnOutcome::Unchanged;
        }
        let Some(shape) = RewriteShape::from_pair(surface, canonical) else {
            return LearnOutcome::Unchanged;
        };
        let id = rule_id(kind, &shape, canonical);
        if let Some(existing) = self.rules.get_mut(&id) {
            existing.support = existing.support.saturating_add(1);
            existing.last_turn = turn;
            if existing.expected_type.is_none() {
                existing.expected_type = expected_type
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(str::to_string);
            }
            existing.exceptions.remove(surface);
            if existing.status == AlignmentStatus::Refuted {
                self.push_event(turn, "support_refuted", &id, surface.to_string());
                return LearnOutcome::Refuted { rule_id: id };
            }
            self.revision = self.revision.saturating_add(1);
            self.push_event(turn, "support", &id, surface.to_string());
            return LearnOutcome::Learned {
                rule_id: id,
                created: false,
            };
        }
        if let Some(existing) = self.rules.values().find(|rule| {
            rule.kind == kind
                && rule.status == AlignmentStatus::Active
                && rule.matches_literal(surface)
                && rule.target != canonical
        }) {
            return LearnOutcome::Conflict {
                existing: existing.id.clone(),
            };
        }
        let rule = AlignmentRule {
            id: id.clone(),
            shape,
            target: canonical.to_string(),
            kind,
            expected_type: expected_type
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string),
            scope,
            expires_turn: scope.expires_turn(turn),
            status: AlignmentStatus::Active,
            support: 1,
            refutations: 0,
            created_turn: turn,
            last_turn: turn,
            origin: origin.trim().to_string(),
            exceptions: BTreeSet::new(),
        };
        self.rules.insert(id.clone(), rule);
        self.revision = self.revision.saturating_add(1);
        self.push_event(turn, "learn", &id, format!("{} → {}", surface, canonical));
        self.prune();
        LearnOutcome::Learned {
            rule_id: id,
            created: true,
        }
    }

    /// 否决一条规则：保留墓碑，并让该具体词面永久成为规则例外。
    pub fn refute_rule(&mut self, turn: u64, id: &str, surface: &str) -> bool {
        self.begin_turn(turn);
        let Some(rule) = self.rules.get_mut(id) else {
            return false;
        };
        rule.status = AlignmentStatus::Refuted;
        rule.refutations = rule.refutations.saturating_add(1);
        rule.last_turn = turn;
        if !surface.trim().is_empty() {
            rule.exceptions.insert(surface.trim().to_string());
        }
        self.revision = self.revision.saturating_add(1);
        self.push_event(turn, "refute", id, surface.trim().to_string());
        true
    }

    /// 否决「某词面 → 某身份」的全部当前解释。
    pub fn refute_pair(&mut self, turn: u64, surface: &str, canonical: &str) -> usize {
        self.begin_turn(turn);
        let surface = surface.trim();
        let canonical = canonical.trim();
        let ids: Vec<String> = self
            .rules
            .values()
            .filter(|rule| {
                rule.is_active_at(turn)
                    && rule.target == canonical
                    && match &rule.shape {
                        RewriteShape::Literal { .. } => rule.matches_literal(surface),
                        shape => shape
                            .strip(surface)
                            .map(|inner| inner == canonical)
                            .unwrap_or(false),
                    }
            })
            .map(|rule| rule.id.clone())
            .collect();
        let mut count = 0;
        for id in ids {
            if self.refute_rule(turn, &id, surface) {
                count += 1;
            }
        }
        count
    }

    /// 记住用户在一次纠正/回答里实际使用的词面（反向生成的首选）。
    pub fn remember_echo(&mut self, turn: u64, surface: &str, canonical: &str) {
        self.begin_turn(turn);
        let surface = surface.trim();
        let canonical = canonical.trim();
        if surface.is_empty() || canonical.is_empty() || surface == canonical {
            return;
        }
        let entry = self.echoes.entry(canonical.to_string()).or_default();
        entry.retain(|existing| existing != surface);
        entry.insert(0, surface.to_string());
        entry.truncate(8);
        self.revision = self.revision.saturating_add(1);
        self.bound_echoes();
    }

    pub fn echo_surfaces(&self, canonical: &str) -> &[String] {
        self.echoes
            .get(canonical)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// 单次词面的前向对齐（用于反向生成的复核、纠正与 CLI 查询）。
    pub fn canonicalize_with<F, G>(
        &self,
        surface: &str,
        kind: AlignmentKind,
        turn: u64,
        resolve: F,
        type_of: G,
    ) -> Option<AlignmentMatch>
    where
        F: Fn(&str) -> Vec<String>,
        G: Fn(&str) -> Option<String>,
    {
        let surface = surface.trim();
        if surface.is_empty() || surface.chars().count() > MAX_SURFACE_CHARS {
            return None;
        }
        let full_known = resolve(surface);
        let mut best: Option<AlignmentMatch> = None;
        for rule in self.rules.values() {
            if !rule.is_active_at(turn)
                || rule.kind != kind
                || rule.exceptions.contains(surface)
            {
                continue;
            }
            let candidate = if rule.matches_literal(surface) {
                Some(rule.target.as_str())
            } else {
                rule.shape.strip(surface)
            };
            let Some(candidate) = candidate else {
                continue;
            };
            if candidate.is_empty()
                || candidate == surface
                || candidate.chars().count() > MAX_CANONICAL_CHARS
            {
                continue;
            }
            let targets = unique_targets(resolve(candidate));
            if targets.len() != 1 {
                continue;
            }
            let canonical = &targets[0];
            if !full_known.is_empty() && !full_known.iter().any(|known| known == canonical) {
                continue;
            }
            if !type_allows(rule.expected_type.as_deref(), type_of(canonical).as_deref())
                || !rule_allows(rule, canonical, &type_of)
            {
                continue;
            }
            let matched = AlignmentMatch {
                rule_id: rule.id.clone(),
                surface: surface.to_string(),
                start: 0,
                end: surface.chars().count(),
                canonical: canonical.clone(),
                kind,
                expected_type: rule.expected_type.clone(),
                support: rule.support,
                literal: rule.shape.is_literal(),
                reason: format!(
                    "镜像对齐：{}（规则 {}，支持 {}）",
                    rule.shape.notation(),
                    rule.id,
                    rule.support
                ),
            };
            if better_match(&matched, best.as_ref(), rule) {
                best = Some(matched);
            }
        }
        best
    }

    /// 在整段文本里找出所有可落地、类型相容的镜像指称。
    pub fn mention_spans_with<F, G>(
        &self,
        text: &str,
        kind: AlignmentKind,
        turn: u64,
        resolve: F,
        type_of: G,
    ) -> Vec<AlignmentMatch>
    where
        F: Fn(&str) -> Vec<String>,
        G: Fn(&str) -> Option<String>,
    {
        if text.trim().is_empty() {
            return Vec::new();
        }
        let chars: Vec<char> = text.chars().collect();
        let mut by_span: BTreeMap<(usize, usize), Vec<AlignmentMatch>> = BTreeMap::new();
        for rule in self.rules.values() {
            if !rule.is_active_at(turn) || rule.kind != kind {
                continue;
            }
            for (start, end) in rule_spans(rule, &chars) {
                if end <= start || end - start > MAX_SURFACE_CHARS {
                    continue;
                }
                let surface: String = chars[start..end].iter().collect();
                if rule.exceptions.contains(&surface) {
                    continue;
                }
                let (candidate, literal) = match &rule.shape {
                    RewriteShape::Literal { .. } => {
                        if !rule.matches_literal(&surface) {
                            continue;
                        }
                        (rule.target.clone(), true)
                    }
                    shape => {
                        let Some(inner) = shape.strip(&surface) else {
                            continue;
                        };
                        (inner.to_string(), false)
                    }
                };
                if candidate.is_empty()
                    || candidate == surface
                    || candidate.chars().count() > MAX_CANONICAL_CHARS
                {
                    continue;
                }
                let targets = unique_targets(resolve(&candidate));
                if targets.len() != 1 {
                    continue;
                }
                let canonical = &targets[0];
                let full_known = resolve(&surface);
                if !full_known.is_empty() && !full_known.iter().any(|known| known == canonical) {
                    continue;
                }
                if !type_allows(rule.expected_type.as_deref(), type_of(canonical).as_deref())
                    || !rule_allows(rule, canonical, &type_of)
                {
                    continue;
                }
                let matched = AlignmentMatch {
                    rule_id: rule.id.clone(),
                    surface: surface.clone(),
                    start,
                    end,
                    canonical: canonical.clone(),
                    kind,
                    expected_type: rule.expected_type.clone(),
                    support: rule.support,
                    literal,
                    reason: format!(
                        "镜像对齐：{}（{}规则，支持 {}）",
                        rule.shape.notation(),
                        if literal { "字面" } else { "形状" },
                        rule.support
                    ),
                };
                let entry = by_span.entry((start, end)).or_default();
                if !entry.iter().any(|existing| {
                    existing.canonical == matched.canonical && existing.rule_id == matched.rule_id
                }) {
                    entry.push(matched);
                }
            }
        }
        let mut out = Vec::new();
        for ((_, _), mut matches) in by_span {
            let mut canonicals: Vec<String> =
                matches.iter().map(|m| m.canonical.clone()).collect();
            canonicals.sort();
            canonicals.dedup();
            if canonicals.len() != 1 {
                continue;
            }
            matches.sort_by(|left, right| {
                right
                    .literal
                    .cmp(&left.literal)
                    .then_with(|| right.support.cmp(&left.support))
                    .then_with(|| left.rule_id.cmp(&right.rule_id))
            });
            if let Some(best) = matches.into_iter().next() {
                out.push(best);
            }
        }
        out.sort_by(|left, right| {
            left.start
                .cmp(&right.start)
                .then_with(|| right.end.cmp(&left.end))
        });
        let mut selected: Vec<AlignmentMatch> = Vec::new();
        for candidate in out {
            let overlaps = selected.iter().any(|kept| {
                candidate.start < kept.end && kept.start < candidate.end
            });
            if !overlaps {
                selected.push(candidate);
            }
        }
        selected
    }

    /// 反向词面选择：回声 > 指向该身份的规则 > 类型相容的形状规则。
    ///
    /// 生成后立刻用同一文法复核；复核不回来就弃权（可逆律）。
    pub fn surface_for_with<F, G>(
        &self,
        canonical: &str,
        turn: u64,
        resolve: F,
        type_of: G,
    ) -> Option<(String, String)>
    where
        F: Fn(&str) -> Vec<String>,
        G: Fn(&str) -> Option<String>,
    {
        let canonical = canonical.trim();
        if canonical.is_empty() {
            return None;
        }
        for surface in self.echo_surfaces(canonical) {
            if surface == canonical {
                continue;
            }
            if self
                .canonicalize_with(surface, AlignmentKind::Entity, turn, &resolve, &type_of)
                .map(|matched| matched.canonical == canonical)
                .unwrap_or(false)
            {
                return Some((surface.clone(), "回声：本会话用户用过的词面".to_string()));
            }
        }
        let mut candidates: Vec<(&AlignmentRule, String)> = Vec::new();
        for rule in self.rules.values() {
            if !rule.is_active_at(turn) || rule.kind != AlignmentKind::Entity {
                continue;
            }
            if !rule_allows(rule, canonical, &type_of) {
                continue;
            }
            let Some(surface) = rule.shape.apply_to(canonical) else {
                continue;
            };
            if self
                .canonicalize_with(&surface, AlignmentKind::Entity, turn, &resolve, &type_of)
                .map(|matched| matched.canonical == canonical)
                .unwrap_or(false)
            {
                candidates.push((rule, surface));
            }
        }
        candidates.sort_by(|(left_rule, left_surface), (right_rule, right_surface)| {
            let left_specific = left_rule.target == canonical;
            let right_specific = right_rule.target == canonical;
            right_specific
                .cmp(&left_specific)
                .then_with(|| right_rule.support.cmp(&left_rule.support))
                .then_with(|| left_surface.chars().count().cmp(&right_surface.chars().count()))
                .then_with(|| left_rule.id.cmp(&right_rule.id))
        });
        candidates.into_iter().next().map(|(rule, surface)| {
            (
                surface,
                format!("镜像规则：{}（支持 {}）", rule.shape.notation(), rule.support),
            )
        })
    }

    pub fn render_rule(&self, rule: &AlignmentRule) -> String {
        let type_note = rule
            .expected_type
            .as_ref()
            .map(|etype| format!("，类型 {}", etype))
            .unwrap_or_default();
        let origin = if rule.origin.is_empty() {
            String::new()
        } else {
            format!("，来源 {}", rule.origin)
        };
        format!(
            "[{}] {} {} → {}（{}，支持 {}{}{}，作用域 {}{}）",
            rule.id,
            rule.status.label(),
            rule.shape.notation(),
            rule.target,
            rule.kind.label(),
            rule.support,
            if rule.refutations > 0 {
                format!("，否证 {}", rule.refutations)
            } else {
                String::new()
            },
            type_note,
            rule.scope.label(),
            origin
        )
    }

    pub fn render_report(&self) -> String {
        let mut output = format!(
            "镜像对齐记忆：版本 {}，轮次 {}，规则 {}/{} 条生效，回声 {} 个。\n",
            self.version,
            self.turn,
            self.active_len(self.turn),
            self.rules.len(),
            self.echoes.len()
        );
        if self.rules.is_empty() {
            output.push_str("暂无对齐规则；用「对齐纠正：用户词面 = 规范身份」或「记住身份」教学。\n");
        } else {
            for rule in self.rules.values() {
                output.push_str("  ");
                output.push_str(&self.render_rule(rule));
                output.push('\n');
            }
        }
        if !self.echoes.is_empty() {
            output.push_str("回声词面：\n");
            for (canonical, surfaces) in &self.echoes {
                output.push_str(&format!(
                    "  {} ⇐ {}（生成时优先沿用）\n",
                    canonical,
                    surfaces.join("、")
                ));
            }
        }
        output
    }

    fn prune(&mut self) {
        let turn = self.turn;
        let expired: Vec<String> = self
            .rules
            .values()
            .filter(|rule| rule.is_expired_at(turn))
            .map(|rule| rule.id.clone())
            .collect();
        for id in expired {
            self.rules.remove(&id);
            self.push_event(turn, "expire", &id, String::new());
        }
        if self.rules.len() > self.capacity {
            let mut removable: Vec<(bool, u32, u64, String)> = self
                .rules
                .values()
                .map(|rule| {
                    (
                        rule.status == AlignmentStatus::Refuted,
                        rule.support,
                        rule.last_turn,
                        rule.id.clone(),
                    )
                })
                .collect();
            removable.sort_by(|left, right| {
                right
                    .0
                    .cmp(&left.0)
                    .then_with(|| left.1.cmp(&right.1))
                    .then_with(|| left.2.cmp(&right.2))
                    .then_with(|| left.3.cmp(&right.3))
            });
            while self.rules.len() > self.capacity {
                let Some((_, _, _, id)) = removable.pop() else {
                    break;
                };
                if self.rules.remove(&id).is_some() {
                    self.push_event(turn, "evict", &id, String::new());
                }
            }
        }
        self.bound_echoes();
    }

    fn bound_echoes(&mut self) {
        while self.echoes.len() > DEFAULT_ECHO_CAPACITY {
            let stale = self.echoes.keys().next().cloned();
            match stale {
                Some(key) => {
                    self.echoes.remove(&key);
                }
                None => break,
            }
        }
    }

    fn push_event(&mut self, turn: u64, action: &str, rule_id: &str, detail: String) {
        self.events.push(AlignmentEvent {
            turn,
            action: action.to_string(),
            rule_id: rule_id.to_string(),
            detail,
        });
        if self.events.len() > MAX_EVENTS {
            self.events.remove(0);
        }
    }
}

fn rule_id(kind: AlignmentKind, shape: &RewriteShape, target: &str) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for part in [
        format!("{:?}", kind),
        shape.key(),
        target.to_string(),
    ] {
        for byte in part.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        hash ^= 0xff;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("a{:016x}", hash)
}

fn unique_targets(mut targets: Vec<String>) -> Vec<String> {
    targets.retain(|target| !target.trim().is_empty());
    targets.sort();
    targets.dedup();
    targets
}

fn type_allows(expected: Option<&str>, observed: Option<&str>) -> bool {
    let Some(expected) = expected.map(str::trim).filter(|value| !value.is_empty()) else {
        return true;
    };
    let Some(observed) = observed.map(str::trim).filter(|value| !value.is_empty()) else {
        return true;
    };
    let expected_lower = expected.to_lowercase();
    let observed_lower = observed.to_lowercase();
    expected_lower == observed_lower
        || expected_lower.contains(&observed_lower)
        || observed_lower.contains(&expected_lower)
}

fn rule_allows(
    rule: &AlignmentRule,
    canonical: &str,
    type_of: &impl Fn(&str) -> Option<String>,
) -> bool {
    if rule.target == canonical || rule.matches_literal(canonical) {
        return true;
    }
    let expected = rule
        .expected_type
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let observed = type_of(canonical);
    type_allows(expected, observed.as_deref())
        && expected.is_some()
        && observed.is_some()
}

fn better_match(
    candidate: &AlignmentMatch,
    current: Option<&AlignmentMatch>,
    candidate_rule: &AlignmentRule,
) -> bool {
    let Some(current) = current else {
        return true;
    };
    if candidate.literal != current.literal {
        return candidate.literal;
    }
    candidate.support > current.support
        || (candidate.support == current.support
            && candidate.rule_id < current.rule_id)
        || (candidate_rule.id == current.rule_id
            && candidate.support == current.support
            && candidate.canonical < current.canonical)
}

/// 形状规则在字符文本中可能覆盖的跨度；只做候选生成，不在这里判锚定。
fn rule_spans(rule: &AlignmentRule, chars: &[char]) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    match &rule.shape {
        RewriteShape::Literal { surface } => {
            let needle: Vec<char> = surface.chars().collect();
            for start in find_all(chars, &needle) {
                spans.push((start, start + needle.len()));
            }
        }
        RewriteShape::Suffix { suffix } => {
            let needle: Vec<char> = suffix.chars().collect();
            for end in find_all(chars, &needle)
                .into_iter()
                .map(|start| start + needle.len())
            {
                let max_start = end.saturating_sub(MAX_CANONICAL_CHARS);
                for start in (max_start..end).rev() {
                    if end - start <= MAX_CANONICAL_CHARS && end - start <= MAX_SURFACE_CHARS {
                        spans.push((start, end));
                    }
                }
            }
        }
        RewriteShape::Prefix { prefix } => {
            let needle: Vec<char> = prefix.chars().collect();
            for start in find_all(chars, &needle) {
                let body_start = start + needle.len();
                for end in
                    body_start + 1..=(body_start + MAX_CANONICAL_CHARS).min(chars.len())
                {
                    if end - start <= MAX_SURFACE_CHARS {
                        spans.push((start, end));
                    }
                }
            }
        }
        RewriteShape::Wrapper { open, close } => {
            let open_chars: Vec<char> = open.chars().collect();
            let close_chars: Vec<char> = close.chars().collect();
            for start in find_all(chars, &open_chars) {
                let inner_start = start + open_chars.len();
                for inner_end in inner_start + 1..=chars.len() {
                    if inner_end - inner_start > MAX_CANONICAL_CHARS {
                        break;
                    }
                    if inner_end + close_chars.len() > chars.len() {
                        break;
                    }
                    if chars[inner_end..inner_end + close_chars.len()] == close_chars[..] {
                        spans.push((start, inner_end + close_chars.len()));
                        break;
                    }
                }
            }
        }
    }
    spans.sort_unstable();
    spans.dedup();
    spans
}

fn find_all(haystack: &[char], needle: &[char]) -> Vec<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return Vec::new();
    }
    haystack
        .windows(needle.len())
        .enumerate()
        .filter_map(|(index, window)| (window == needle).then_some(index))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resolver(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Vec<String> {
        let pairs: Vec<(String, String)> = pairs
            .iter()
            .map(|(surface, canonical)| (surface.to_string(), canonical.to_string()))
            .collect();
        move |surface: &str| {
            pairs
                .iter()
                .filter(|(known, _)| known == surface)
                .map(|(_, canonical)| canonical.clone())
                .collect()
        }
    }

    fn types(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let pairs: Vec<(String, String)> = pairs
            .iter()
            .map(|(entity, etype)| (entity.to_string(), etype.to_string()))
            .collect();
        move |entity: &str| {
            pairs
                .iter()
                .find(|(known, _)| known == entity)
                .map(|(_, etype)| etype.clone())
        }
    }

    fn learn_suffix_memory() -> AlignmentMemory {
        let mut memory = AlignmentMemory::new(16);
        let outcome = memory.learn_pair(
            1,
            "未名科技公司",
            "未名科技",
            AlignmentKind::Entity,
            Some("组织"),
            "用户纠正",
            AlignmentScope::Session,
        );
        assert!(outcome.changed());
        memory
    }

    #[test]
    fn suffix_shape_generalizes_to_unseen_surface() {
        let memory = learn_suffix_memory();
        let resolve = resolver(&[
            ("未名科技", "未名科技"),
            ("远山科技", "远山科技"),
            ("张三", "张三"),
        ]);
        let type_of = types(&[
            ("未名科技", "组织"),
            ("远山科技", "组织"),
            ("张三", "人物"),
        ]);
        let matched = memory
            .canonicalize_with(
                "远山科技公司",
                AlignmentKind::Entity,
                1,
                &resolve,
                &type_of,
            )
            .expect("形状规则应泛化到未登录词面");
        assert_eq!(matched.canonical, "远山科技");

        let mentions = memory.mention_spans_with(
            "远山科技公司发布了新芯片",
            AlignmentKind::Entity,
            1,
            &resolve,
            &type_of,
        );
        assert_eq!(mentions.len(), 1);
        assert_eq!(mentions[0].surface, "远山科技公司");
        assert_eq!(mentions[0].canonical, "远山科技");
        assert_eq!(mentions[0].start, 0);
        assert_eq!(mentions[0].end, 6);
    }

    #[test]
    fn type_guard_blocks_wrong_kind_of_name() {
        let memory = learn_suffix_memory();
        let resolve = resolver(&[("张三", "张三")]);
        let type_of = types(&[("张三", "人物")]);
        assert!(memory
            .canonicalize_with("张三公司", AlignmentKind::Entity, 1, &resolve, &type_of)
            .is_none());
    }

    #[test]
    fn known_full_surface_wins_over_rewrite() {
        let memory = learn_suffix_memory();
        let resolve = resolver(&[
            ("未名科技", "未名科技"),
            ("未名科技公司", "未名科技公司"),
        ]);
        let type_of = types(&[("未名科技", "组织"), ("未名科技公司", "组织")]);
        assert!(memory
            .canonicalize_with(
                "未名科技公司",
                AlignmentKind::Entity,
                1,
                &resolve,
                &type_of
            )
            .is_none());
    }

    #[test]
    fn ambiguous_anchor_is_refused() {
        let memory = learn_suffix_memory();
        let resolve = resolver(&[
            ("远山科技", "远山科技"),
            ("远山科技", "远山科技集团"),
        ]);
        let type_of = types(&[("远山科技", "组织"), ("远山科技集团", "组织")]);
        assert!(memory
            .canonicalize_with(
                "远山科技公司",
                AlignmentKind::Entity,
                1,
                &resolve,
                &type_of
            )
            .is_none());
    }

    #[test]
    fn wrapper_and_literal_shapes_are_reversible() {
        let mut memory = AlignmentMemory::new(16);
        memory.learn_pair(
            1,
            "《三体》",
            "三体",
            AlignmentKind::Entity,
            Some("作品"),
            "用户问法",
            AlignmentScope::Session,
        );
        memory.learn_pair(
            1,
            "老婆",
            "妻子",
            AlignmentKind::Predicate,
            None,
            "用户说法",
            AlignmentScope::Session,
        );
        let resolve = resolver(&[("三体", "三体"), ("妻子", "妻子")]);
        let type_of = types(&[("三体", "作品"), ("妻子", "人物")]);
        let matched = memory
            .canonicalize_with("《三体》", AlignmentKind::Entity, 1, &resolve, &type_of)
            .unwrap();
        assert_eq!(matched.canonical, "三体");
        let (surface, reason) = memory
            .surface_for_with("三体", 1, &resolve, &type_of)
            .unwrap();
        assert_eq!(surface, "《三体》");
        assert!(reason.contains("镜像规则"));

        let predicate = memory
            .canonicalize_with("老婆", AlignmentKind::Predicate, 1, &resolve, &type_of)
            .unwrap();
        assert_eq!(predicate.canonical, "妻子");
    }

    #[test]
    fn echo_wins_on_reverse_and_must_re_resolve() {
        let mut memory = learn_suffix_memory();
        memory.remember_echo(2, "未名科技公司", "未名科技");
        let resolve = resolver(&[("未名科技", "未名科技")]);
        let type_of = types(&[("未名科技", "组织")]);
        let (surface, reason) = memory
            .surface_for_with("未名科技", 2, &resolve, &type_of)
            .unwrap();
        assert_eq!(surface, "未名科技公司");
        assert!(reason.contains("回声"));

        memory.remember_echo(3, "未名科技集团", "未名科技");
        let resolve_without_shape = resolver(&[("未名科技", "未名科技")]);
        let mut no_rule = AlignmentMemory::new(8);
        no_rule.remember_echo(3, "未名科技集团", "未名科技");
        assert!(
            no_rule
                .surface_for_with("未名科技", 3, &resolve_without_shape, &type_of)
                .is_none(),
            "回声无法再对齐时不得输出"
        );
    }

    #[test]
    fn refutation_is_a_tombstone() {
        let mut memory = learn_suffix_memory();
        let resolve = resolver(&[("未名科技", "未名科技"), ("远山科技", "远山科技")]);
        let type_of = types(&[("未名科技", "组织"), ("远山科技", "组织")]);
        let matched = memory
            .canonicalize_with(
                "未名科技公司",
                AlignmentKind::Entity,
                1,
                &resolve,
                &type_of,
            )
            .unwrap();
        assert!(memory.refute_rule(1, &matched.rule_id, "未名科技公司"));
        assert!(memory
            .canonicalize_with(
                "未名科技公司",
                AlignmentKind::Entity,
                1,
                &resolve,
                &type_of
            )
            .is_none());
        let again = memory.learn_pair(
            1,
            "未名科技公司",
            "未名科技",
            AlignmentKind::Entity,
            Some("组织"),
            "重复学习",
            AlignmentScope::Session,
        );
        assert!(matches!(again, LearnOutcome::Refuted { .. }));
        assert!(memory
            .canonicalize_with(
                "未名科技公司",
                AlignmentKind::Entity,
                1,
                &resolve,
                &type_of
            )
            .is_none());
    }

    #[test]
    fn turn_scope_expires_on_the_next_turn() {
        let mut memory = AlignmentMemory::new(8);
        memory.learn_pair(
            4,
            "张三先生",
            "张三",
            AlignmentKind::Entity,
            Some("人物"),
            "临时说法",
            AlignmentScope::Turns(1),
        );
        let resolve = resolver(&[("张三", "张三")]);
        let type_of = types(&[("张三", "人物")]);
        assert!(memory
            .canonicalize_with("张三先生", AlignmentKind::Entity, 4, &resolve, &type_of)
            .is_some());
        memory.begin_turn(5);
        assert!(memory
            .canonicalize_with("张三先生", AlignmentKind::Entity, 5, &resolve, &type_of)
            .is_none());
    }

    #[test]
    fn conflicting_literal_targets_do_not_silently_replace() {
        let mut memory = AlignmentMemory::new(8);
        memory.learn_pair(
            1,
            "苹果",
            "苹果公司",
            AlignmentKind::Entity,
            Some("组织"),
            "一次澄清",
            AlignmentScope::Session,
        );
        let outcome = memory.learn_pair(
            2,
            "苹果",
            "苹果果实",
            AlignmentKind::Entity,
            Some("食物"),
            "另一次澄清",
            AlignmentScope::Session,
        );
        assert!(matches!(outcome, LearnOutcome::Conflict { .. }));
        let resolve = resolver(&[("苹果公司", "苹果公司"), ("苹果果实", "苹果果实")]);
        let type_of = types(&[("苹果公司", "组织"), ("苹果果实", "食物")]);
        let matched = memory
            .canonicalize_with("苹果", AlignmentKind::Entity, 2, &resolve, &type_of)
            .unwrap();
        assert_eq!(matched.canonical, "苹果公司");
    }

    #[test]
    fn serialization_round_trip_keeps_rules() {
        let mut memory = learn_suffix_memory();
        memory.remember_echo(1, "未名科技公司", "未名科技");
        let json = serde_json::to_string(&memory).unwrap();
        let restored: AlignmentMemory = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, memory);
        assert!(json.contains("suffix"));
        assert!(json.contains("未名科技公司"));
    }

    #[test]
    fn spans_cover_wrapped_and_suffixed_mentions() {
        let mut memory = AlignmentMemory::new(16);
        memory.learn_pair(
            1,
            "张三先生",
            "张三",
            AlignmentKind::Entity,
            Some("人物"),
            "用户称呼",
            AlignmentScope::Session,
        );
        memory.learn_pair(
            1,
            "《三体》",
            "三体",
            AlignmentKind::Entity,
            Some("作品"),
            "书名号",
            AlignmentScope::Session,
        );
        let resolve = resolver(&[
            ("张三", "张三"),
            ("李四", "李四"),
            ("三体", "三体"),
        ]);
        let type_of = types(&[
            ("张三", "人物"),
            ("李四", "人物"),
            ("三体", "作品"),
        ]);
        let mentions = memory.mention_spans_with(
            "张三先生和李四先生都读过《三体》",
            AlignmentKind::Entity,
            1,
            &resolve,
            &type_of,
        );
        let surfaces: Vec<&str> = mentions.iter().map(|m| m.surface.as_str()).collect();
        assert_eq!(surfaces, vec!["张三先生", "李四先生", "《三体》"]);
        assert_eq!(mentions[2].canonical, "三体");
    }
}

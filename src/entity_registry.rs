//! EIR：实体身份注册表（Entity Identity Registry）。
//!
//! 谓词侧由「画像注册表」把准入/别名/互逆写成一本可审计的词典
//! （`predicate_registry`），实体侧需要同构的宪法。名称识别活在提及时空
//! （一次话语里的一串字符：别名、错别字、简称、指代），实体 URI 活在身份
//! 时空（命名空间里的常量）。本模块把两者显式分开：
//!
//! - **提名自由**：任何词面都可以被提名（导入模式、对话澄清、显式命令、
//!   知识图谱观测），提名一律默认「待确认」；
//! - **承诺受审**：把词面绑到规范身份是显式动作（确认 / 拒绝 / 忘记 / 合并），
//!   带证据、可回执、可撤销；
//! - **铸造受律**：`mint_check` 是铸造新 URI 前的检查——折叠同形的名称
//!   只提名别名，绝不新铸第二个 URI。
//!
//! 唯一自动「已确认」的来源是知识图谱里已有的 `name` 属性观测：图谱事实
//! 本身就是证据，其余来源（文本模式、对话、API）只提名，等一次人类签字。
//!
//! 折叠（`fold_surface`）只做无损规约：宽窄折叠、不可见字符剔除、空白折叠、
//! 成对包裹符剥除；不做拼音、繁简、编辑距离——那些属于读侧容错，永不进身份。

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::type_system::{TypeSystem, TypeVerdict};

/// 折叠算法版本：落盘注册表带版本门（高于支持版本拒绝解码）。
pub const FOLD_VERSION: u32 = 1;

/// 提名来源（证据的序数档，不是训练出来的置信度）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NominationOrigin {
    /// 知识图谱 name 属性观测（唯一自动确认的来源）
    Kb,
    /// config.toml 的 [[entity_profiles]] 声明
    Config,
    /// 用户显式命令（确认身份 / 记住身份）
    Dialogue,
    /// 文本导入模式（又称/简称/全称/即）
    Import,
    /// 外部接口来源
    Api,
}

impl NominationOrigin {
    pub const fn label(self) -> &'static str {
        match self {
            NominationOrigin::Kb => "知识图谱",
            NominationOrigin::Config => "配置声明",
            NominationOrigin::Dialogue => "用户确认",
            NominationOrigin::Import => "文本导入",
            NominationOrigin::Api => "外部接口",
        }
    }
}

/// 身份状态：提名是提案（待确认），确认才是承诺。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntityStatus {
    Pending,
    Confirmed,
    Rejected,
}

impl Default for EntityStatus {
    fn default() -> Self {
        EntityStatus::Pending
    }
}

impl EntityStatus {
    pub const fn label(self) -> &'static str {
        match self {
            EntityStatus::Pending => "待确认",
            EntityStatus::Confirmed => "已确认",
            EntityStatus::Rejected => "已拒绝",
        }
    }
}

/// 单个别名词面的记录（首次写法冻结，同折叠形的其它写法进 variants）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AliasRecord {
    /// 首次提名的原词面（承诺以它为基准）
    pub surface: String,
    #[serde(default)]
    pub status: EntityStatus,
    #[serde(default)]
    pub origins: BTreeSet<NominationOrigin>,
    /// 证据来源（去重、稳定排序）
    #[serde(default)]
    pub evidence: Vec<String>,
    /// 被提名次数（同一折叠形的多次提名不重复建条目）
    #[serde(default)]
    pub nominations: usize,
    /// 同折叠形的其它写法（展示用，永不各自铸身份）
    #[serde(default)]
    pub variants: BTreeSet<String>,
}

/// 一个规范身份：类型、状态、别名与合并去向。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntityRecord {
    pub canonical: String,
    #[serde(default)]
    pub status: EntityStatus,
    #[serde(default)]
    pub origins: BTreeSet<NominationOrigin>,
    /// 已知类型（来自图谱 rdf:type 观测或配置声明）
    #[serde(default)]
    pub types: BTreeSet<String>,
    #[serde(default)]
    pub aliases: BTreeMap<String, AliasRecord>,
    /// 来源说明（人类可读）
    #[serde(default)]
    pub sources: BTreeSet<String>,
    /// 已合并去向（重定向链；URI 永不回收）
    #[serde(default)]
    pub merged_into: Option<String>,
    /// 是否在图谱中观测过（证实这个身份确实活在库里）
    #[serde(default)]
    pub observed: bool,
}

/// 合并编年记录（可审计：谁并入谁、搬了多少别名）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MergeRecord {
    pub from: String,
    pub into: String,
    pub moved_aliases: usize,
    /// 合并前来源身份的类型快照（只读审计用，旧快照缺省为空）。
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub from_types: BTreeSet<String>,
    /// 合并前目标身份的类型快照（只读审计用，旧快照缺省为空）。
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub into_types: BTreeSet<String>,
}

/// 词面解析结论（只解析已确认的词面；提名中的提案不参与解析）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EntityResolution {
    Canonical(String),
    Redirected { from: String, to: String },
    Ambiguous(Vec<String>),
    Unknown,
}

impl EntityResolution {
    pub fn canonical(&self) -> Option<&str> {
        match self {
            EntityResolution::Canonical(id) => Some(id),
            EntityResolution::Redirected { to, .. } => Some(to),
            _ => None,
        }
    }

    pub fn label(&self) -> String {
        match self {
            EntityResolution::Canonical(id) => format!("身份「{}」", id),
            EntityResolution::Redirected { from, to } => {
                format!("重定向「{}」→「{}」", from, to)
            }
            EntityResolution::Ambiguous(ids) => format!("歧义（{}）", ids.join(" / ")),
            EntityResolution::Unknown => "未登记".to_string(),
        }
    }
}

/// 铸 URI 前的检查结论。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MintVerdict {
    /// 该名称已有身份（含已确认别名）：复用，不新铸
    Exists { canonical: String, reason: String },
    /// 该名称已合并到其它身份：铸 URI 走重定向
    Redirect { from: String, to: String },
    /// 折叠同形已有身份：只提名别名，绝不新铸第二个 URI
    Variant { canonical: String, folded: String },
    /// 该名称可解析到多个身份：先澄清
    Ambiguous { candidates: Vec<String> },
    /// 未登记：可以铸造新 URI
    Fresh,
}

impl MintVerdict {
    pub fn blocks_mint(&self) -> bool {
        !matches!(self, MintVerdict::Fresh)
    }

    pub fn label(&self) -> String {
        match self {
            MintVerdict::Exists { canonical, reason } => {
                format!("身份已存在「{}」（{}）", canonical, reason)
            }
            MintVerdict::Redirect { from, to } => format!("「{}」已重定向到「{}」", from, to),
            MintVerdict::Variant { canonical, folded } => format!(
                "折叠同形「{}」已属于「{}」：只提名别名，不新铸 URI",
                folded, canonical
            ),
            MintVerdict::Ambiguous { candidates } => {
                format!("可解析到多个身份：{}", candidates.join(" / "))
            }
            MintVerdict::Fresh => "未登记：可以铸造新 URI".to_string(),
        }
    }

    /// 带词面的裁决陈述（命令行输出用）。
    pub fn line(&self, name: &str) -> String {
        match self {
            MintVerdict::Exists { canonical, reason } => {
                format!(
                    "「{}」：已存在身份「{}」（{}）——复用，不新铸。",
                    name, canonical, reason
                )
            }
            MintVerdict::Redirect { from, to } => {
                format!(
                    "「{}」：已随「{}」重定向到「{}」——引用应走重定向。",
                    name, from, to
                )
            }
            MintVerdict::Variant { canonical, folded } => format!(
                "「{}」：与「{}」折叠同形（{}）——只提名别名，不新铸第二个 URI。",
                name, canonical, folded
            ),
            MintVerdict::Ambiguous { candidates } => format!(
                "「{}」：可解析到多个身份（{}）——先澄清，勿贸然铸新。",
                name,
                candidates.join(" / ")
            ),
            MintVerdict::Fresh => {
                format!("「{}」：未登记——可以铸造新 URI，身份由首次观测确立。", name)
            }
        }
    }
}

/// 单次提名的结果序数。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NominationKind {
    /// 自动确认（仅知识图谱观测）
    Confirmed,
    /// 待确认（默认）
    Pending,
    /// 已存在（无变化）
    Duplicate,
    /// 墓碑：已被拒绝，不重复提名
    Tombstoned,
    /// 类型闸门：词面已有互斥类型的身份，转需人工确认
    TypeConflict,
    /// 空词面/空目标：忽略
    Ignored,
}

impl NominationKind {
    pub const fn label(self) -> &'static str {
        match self {
            NominationKind::Confirmed => "已确认",
            NominationKind::Pending => "待确认",
            NominationKind::Duplicate => "已存在",
            NominationKind::Tombstoned => "已拒绝（墓碑）",
            NominationKind::TypeConflict => "类型冲突（需人工确认）",
            NominationKind::Ignored => "已忽略",
        }
    }
}

/// 提名回执：词面、目标、结果与一句话说明。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NominationOutcome {
    pub kind: NominationKind,
    pub surface: String,
    pub canonical: String,
    pub changed: bool,
    pub note: String,
}

/// config.toml 中的 `[[entity_profiles]]` 声明。
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct EntityProfileConfig {
    pub name: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default)]
    pub etype: Vec<String>,
    #[serde(default)]
    pub description: Option<String>,
}

/// 文本模式抽出的别名候选（只提名，不写库）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AliasCandidate {
    /// 待提名的别名词面
    pub surface: String,
    /// 正文里的规范名（可能是图谱已有实体，也可能是新名字）
    pub canonical: String,
    /// 命中的模式词（简称/全称/又称/又名/俗称/别称/即）
    pub pattern: String,
    /// 原文片段（证据）
    pub evidence: String,
}

/// 实体身份注册表。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityRegistry {
    /// 折叠算法版本（版本门）
    #[serde(default = "fold_version_default")]
    pub fold_version: u32,
    #[serde(default)]
    records: BTreeMap<String, EntityRecord>,
    #[serde(default)]
    merges: Vec<MergeRecord>,
}

fn fold_version_default() -> u32 {
    FOLD_VERSION
}

impl Default for EntityRegistry {
    fn default() -> Self {
        Self {
            fold_version: FOLD_VERSION,
            records: BTreeMap::new(),
            merges: Vec::new(),
        }
    }
}

/// 全角 ASCII 与常见不可见字符的无损折叠。
fn fold_char(ch: char) -> Option<char> {
    match ch {
        // 全角 ASCII → 半角
        '\u{FF01}'..='\u{FF5E}' => char::from_u32(ch as u32 - 0xFEE0).or(Some(ch)),
        // 表意空格 → 普通空格
        '\u{3000}' => Some(' '),
        // 零宽字符：不可见，剔除（无损于可读语义）
        '\u{200B}'..='\u{200D}' | '\u{FEFF}' | '\u{2060}' => None,
        other => Some(other),
    }
}

fn is_cjk(ch: char) -> bool {
    matches!(ch,
        '\u{3400}'..='\u{4DBF}'
        | '\u{4E00}'..='\u{9FFF}'
        | '\u{F900}'..='\u{FAFF}'
    )
}

/// 成对包裹符（剥除时要求首尾配对）。
const WRAPPING_PAIRS: [(char, char); 14] = [
    ('《', '》'),
    ('〈', '〉'),
    ('「', '」'),
    ('『', '』'),
    ('〖', '〗'),
    ('【', '】'),
    ('〔', '〕'),
    ('（', '）'),
    ('(', ')'),
    ('[', ']'),
    ('“', '”'),
    ('‘', '’'),
    ('"', '"'),
    ('\'', '\''),
];

/// 无损折叠：trim → 宽窄折叠 → 去不可见字符 → 空白折叠（CJK 相邻空格剔除）
/// → 成对包裹符剥除。确定性、幂等，可无损反推「哪些写法视为同一词面」。
pub fn fold_surface(surface: &str) -> String {
    let mut out = String::with_capacity(surface.len());
    let mut pending_space = false;
    let mut last_cjk = false;
    for ch in surface.trim().chars() {
        let Some(ch) = fold_char(ch) else {
            continue;
        };
        if ch.is_whitespace() {
            if !out.is_empty() {
                pending_space = true;
            }
            continue;
        }
        if pending_space {
            // 空格紧邻汉字（任一侧）时剔除：「ACME 公司」≡「ACME公司」；
            // 两侧都是非汉字时保留单个空格：「John Smith」不被粘连。
            if !(last_cjk || is_cjk(ch)) {
                out.push(' ');
            }
            pending_space = false;
        }
        out.push(ch);
        last_cjk = is_cjk(ch);
    }
    strip_wrapping(&out)
}

fn strip_wrapping(input: &str) -> String {
    let mut current = input.trim().to_string();
    loop {
        let mut chars = current.chars();
        let Some(first) = chars.next() else {
            return current;
        };
        let Some(last) = current.chars().next_back() else {
            return current;
        };
        if first == last && !current.is_empty() && current.chars().count() < 2 {
            return current;
        }
        let matched = WRAPPING_PAIRS
            .iter()
            .any(|(open, close)| first == *open && last == *close);
        if !matched || current.chars().count() < 2 {
            return current;
        }
        let inner: String = current
            .chars()
            .skip(1)
            .take(current.chars().count().saturating_sub(2))
            .collect();
        current = inner.trim().to_string();
    }
}

impl EntityRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn version_supported(&self) -> bool {
        self.fold_version <= FOLD_VERSION
    }

    pub fn len(&self) -> usize {
        self.records
            .values()
            .filter(|record| record.merged_into.is_none())
            .count()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn record(&self, canonical: &str) -> Option<&EntityRecord> {
        self.records.get(canonical.trim())
    }

    pub fn records(&self) -> impl Iterator<Item = &EntityRecord> {
        self.records.values()
    }

    pub fn merges(&self) -> &[MergeRecord] {
        &self.merges
    }

    /// 已确认别名映射（别名 → 规范名），字典序，供链接器同步。
    pub fn confirmed_alias_pairs(&self) -> Vec<(String, String)> {
        let mut pairs = Vec::new();
        for record in self.records.values() {
            if record.merged_into.is_some() {
                continue;
            }
            for alias in record.aliases.values() {
                if alias.status == EntityStatus::Confirmed && alias.surface != record.canonical {
                    pairs.push((alias.surface.clone(), record.canonical.clone()));
                }
            }
        }
        pairs.sort();
        pairs.dedup();
        pairs
    }

    /// 待确认提名（别名 → 规范名，按规范名与词面排序）。
    pub fn pending_aliases(&self) -> Vec<(String, String)> {
        let mut pending = Vec::new();
        for (canonical, record) in &self.records {
            if record.merged_into.is_some() {
                continue;
            }
            for alias in record.aliases.values() {
                if alias.status == EntityStatus::Pending {
                    pending.push((canonical.clone(), alias.surface.clone()));
                }
            }
        }
        pending.sort();
        pending
    }

    /// 已拒绝词面（墓碑）清单：（规范名, 词面），排序。
    pub fn rejected_aliases(&self) -> Vec<(String, String)> {
        let mut rejected = Vec::new();
        for (canonical, record) in &self.records {
            for alias in record.aliases.values() {
                if alias.status == EntityStatus::Rejected {
                    rejected.push((canonical.clone(), alias.surface.clone()));
                }
            }
        }
        rejected.sort();
        rejected
    }

    fn ensure_record(
        &mut self,
        canonical: &str,
        origin: NominationOrigin,
        source: &str,
    ) -> &mut EntityRecord {
        let record = self
            .records
            .entry(canonical.to_string())
            .or_insert_with(|| EntityRecord {
                canonical: canonical.to_string(),
                status: EntityStatus::Pending,
                origins: BTreeSet::new(),
                types: BTreeSet::new(),
                aliases: BTreeMap::new(),
                sources: BTreeSet::new(),
                merged_into: None,
                observed: false,
            });
        record.origins.insert(origin);
        if !source.trim().is_empty() {
            record.sources.insert(source.trim().to_string());
        }
        record
    }

    /// 按规范名或已确认词面解析到记录（跟随重定向链）。
    fn find_target(&self, name: &str) -> Option<String> {
        let name = name.trim();
        if name.is_empty() {
            return None;
        }
        let folded = fold_surface(name);
        if let Some(record) = self.records.get(name) {
            return Some(self.final_target(&record.canonical));
        }
        for record in self.records.values() {
            if fold_surface(&record.canonical) == folded && record.status != EntityStatus::Rejected
            {
                return Some(self.final_target(&record.canonical));
            }
        }
        match self.resolve(name) {
            EntityResolution::Canonical(id) => Some(id),
            EntityResolution::Redirected { to, .. } => Some(to),
            _ => None,
        }
    }

    /// 重定向链的终点（merge 保证无环；异常时停在最后一个未见节点）。
    fn final_target(&self, canonical: &str) -> String {
        let mut cursor = canonical.to_string();
        let mut seen = BTreeSet::new();
        seen.insert(cursor.clone());
        while let Some(next) = self
            .records
            .get(&cursor)
            .and_then(|record| record.merged_into.clone())
        {
            if !seen.insert(next.clone()) {
                break;
            }
            cursor = next;
        }
        cursor
    }

    /// 登记一个知识图谱中的实体（已确认；唯一的自动确认来源）。
    ///
    /// `names` 是图谱里观测到的 name 字面量：与规范名折叠同形的忽略，
    /// 其余自动确认为别名（图谱事实本身就是证据）。
    pub fn observe_kb_entity(
        &mut self,
        canonical: &str,
        types: &BTreeSet<String>,
        names: &[String],
        source: &str,
    ) -> bool {
        let canonical = canonical.trim();
        if canonical.is_empty() {
            return false;
        }
        let existed = self.records.contains_key(canonical);
        let mut changed = !existed;
        {
            let record = self.ensure_record(canonical, NominationOrigin::Kb, source);
            record.status = EntityStatus::Confirmed;
            if !record.observed {
                record.observed = true;
                changed = true;
            }
            for type_name in types {
                if !type_name.trim().is_empty() && record.types.insert(type_name.trim().to_string())
                {
                    changed = true;
                }
            }
        }
        for name in names {
            let name = name.trim();
            if name.is_empty() || fold_surface(name) == fold_surface(canonical) {
                continue;
            }
            let outcome = self.nominate_alias(
                name,
                canonical,
                NominationOrigin::Kb,
                &format!("图谱 name 观测：{}", name),
            );
            changed |= outcome.changed;
        }
        changed
    }

    /// 提名一个别名（默认待确认；知识图谱来源自动确认）。
    pub fn nominate_alias(
        &mut self,
        surface: &str,
        canonical: &str,
        origin: NominationOrigin,
        evidence: &str,
    ) -> NominationOutcome {
        self.nominate_alias_with_types(surface, canonical, origin, evidence, None)
    }

    /// 带类型闸门的提名：词面已有互斥类型的身份时转「需人工确认」。
    pub fn nominate_alias_with_types(
        &mut self,
        surface: &str,
        canonical: &str,
        origin: NominationOrigin,
        evidence: &str,
        types: Option<&TypeSystem>,
    ) -> NominationOutcome {
        let folded = fold_surface(surface);
        let canonical_input = canonical.trim().to_string();
        let surface_input = surface.trim().to_string();
        if folded.is_empty() || canonical_input.is_empty() {
            return NominationOutcome {
                kind: NominationKind::Ignored,
                surface: surface_input,
                canonical: canonical_input,
                changed: false,
                note: "词面与规范名都不能为空".to_string(),
            };
        }
        let target_fold = fold_surface(&canonical_input);
        if folded == target_fold {
            return NominationOutcome {
                kind: NominationKind::Duplicate,
                surface: surface_input,
                canonical: canonical_input,
                changed: false,
                note: "别名与规范名折叠同形，无需登记".to_string(),
            };
        }
        let target = match self.find_target(&canonical_input) {
            Some(target) => target,
            None => {
                self.ensure_record(&canonical_input, origin, "提名目标");
                canonical_input.clone()
            }
        };

        // 类型闸门：词面若已是另一身份且两方类型可证明互斥 → 需人工确认
        if let Some(types) = types {
            if let Some(owner) = self.foreign_identity_of(&folded, &target) {
                let owner_types = self
                    .records
                    .get(&owner)
                    .map(|record| record.types.clone())
                    .unwrap_or_default();
                let target_types = self
                    .records
                    .get(&target)
                    .map(|record| record.types.clone())
                    .unwrap_or_default();
                if !owner_types.is_empty()
                    && !target_types.is_empty()
                    && types.sets_admit(&owner_types, &target_types) == TypeVerdict::Disjoint
                {
                    let note = format!(
                        "词面「{}」已属于「{}」（{}），与「{}」（{}）类型互斥；如确为同名请用「确认身份」显式承诺",
                        surface_input,
                        owner,
                        owner_types.iter().cloned().collect::<Vec<_>>().join("、"),
                        target,
                        target_types.iter().cloned().collect::<Vec<_>>().join("、"),
                    );
                    return NominationOutcome {
                        kind: NominationKind::TypeConflict,
                        surface: surface_input,
                        canonical: target,
                        changed: false,
                        note,
                    };
                }
            }
        }

        let auto_confirm = origin == NominationOrigin::Kb;
        let canonical_key = target.clone();
        let record = self
            .records
            .entry(canonical_key.clone())
            .or_insert_with(|| EntityRecord {
                canonical: canonical_key.clone(),
                status: EntityStatus::Pending,
                origins: BTreeSet::new(),
                types: BTreeSet::new(),
                aliases: BTreeMap::new(),
                sources: BTreeSet::new(),
                merged_into: None,
                observed: false,
            });
        let alias = record
            .aliases
            .entry(folded.clone())
            .or_insert_with(|| AliasRecord {
                surface: surface_input.clone(),
                status: EntityStatus::Pending,
                origins: BTreeSet::new(),
                evidence: Vec::new(),
                nominations: 0,
                variants: BTreeSet::new(),
            });

        // 墓碑：被拒绝的词面不因重复提名而复活（图谱观测例外——事实强于否决）
        if alias.status == EntityStatus::Rejected && !auto_confirm {
            return NominationOutcome {
                kind: NominationKind::Tombstoned,
                surface: alias.surface.clone(),
                canonical: target,
                changed: false,
                note: format!(
                    "「{}」已被拒绝（墓碑），不予自动提名；如已改主意请用「确认身份」显式承诺",
                    alias.surface
                ),
            };
        }
        if alias.status == EntityStatus::Confirmed && !auto_confirm {
            let new_info = !alias.origins.contains(&origin)
                || !alias.evidence.iter().any(|e| e == evidence.trim());
            if new_info {
                alias.nominations += 1;
            }
            alias.origins.insert(origin);
            push_evidence(&mut alias.evidence, evidence);
            if alias.surface != surface_input {
                alias.variants.insert(surface_input.clone());
            }
            return NominationOutcome {
                kind: NominationKind::Duplicate,
                surface: alias.surface.clone(),
                canonical: target,
                changed: false,
                note: format!("「{}」已是已确认别名", alias.surface),
            };
        }

        let was_new = alias.nominations == 0;
        let new_info = was_new
            || !alias.origins.contains(&origin)
            || !alias.evidence.iter().any(|e| e == evidence.trim());
        if new_info {
            alias.nominations += 1;
        }
        alias.origins.insert(origin);
        if alias.surface != surface_input {
            alias.variants.insert(surface_input.clone());
        }
        push_evidence(&mut alias.evidence, evidence);
        let promoted = auto_confirm && alias.status != EntityStatus::Confirmed;
        if auto_confirm {
            alias.status = EntityStatus::Confirmed;
        }
        if alias.status != EntityStatus::Rejected
            && record.status == EntityStatus::Pending
            && auto_confirm
        {
            record.status = EntityStatus::Confirmed;
        }
        NominationOutcome {
            kind: if alias.status == EntityStatus::Confirmed {
                NominationKind::Confirmed
            } else {
                NominationKind::Pending
            },
            surface: alias.surface.clone(),
            canonical: target,
            changed: was_new || promoted,
            note: format!(
                "已提名「{}」→「{}」（{}）",
                alias.surface,
                canonical_key,
                alias.status.label()
            ),
        }
    }

    /// 词面（折叠形）当前属于哪个其它身份（不含目标）。
    fn foreign_identity_of(&self, folded: &str, target: &str) -> Option<String> {
        let mut owners = BTreeSet::new();
        for (canonical, record) in &self.records {
            if canonical == target || record.merged_into.is_some() {
                continue;
            }
            let canonical_hit =
                fold_surface(canonical) == folded && record.status != EntityStatus::Rejected;
            let alias_hit = record.aliases.values().any(|alias| {
                alias.status != EntityStatus::Rejected && fold_surface(&alias.surface) == folded
            });
            if canonical_hit || alias_hit {
                owners.insert(canonical.clone());
            }
        }
        owners.into_iter().next()
    }

    /// 解析词面（折叠后匹配；未确认的提名不参与）。
    pub fn resolve(&self, surface: &str) -> EntityResolution {
        let folded = fold_surface(surface);
        if folded.is_empty() {
            return EntityResolution::Unknown;
        }
        let mut candidates: BTreeSet<String> = BTreeSet::new();
        let mut redirected = false;
        for (canonical, record) in &self.records {
            // 未确认的提名不参与解析；但显式合并过的名字永远重定向
            if record.status != EntityStatus::Confirmed && record.merged_into.is_none() {
                continue;
            }
            let canonical_hit = fold_surface(canonical) == folded;
            let alias_hit = record
                .aliases
                .get(&folded)
                .map(|alias| alias.status == EntityStatus::Confirmed)
                .unwrap_or(false);
            if !canonical_hit && !alias_hit {
                continue;
            }
            if record.merged_into.is_some() {
                redirected = true;
                candidates.insert(self.final_target(canonical));
            } else {
                candidates.insert(canonical.clone());
            }
        }
        match candidates.len() {
            0 => EntityResolution::Unknown,
            1 => {
                let target = candidates.into_iter().next().expect("非空");
                if redirected {
                    EntityResolution::Redirected {
                        from: surface.trim().to_string(),
                        to: target,
                    }
                } else {
                    EntityResolution::Canonical(target)
                }
            }
            _ => EntityResolution::Ambiguous(candidates.into_iter().collect()),
        }
    }

    /// 铸 URI 前的宪法检查：Exists / Redirect / Variant / Ambiguous / Fresh。
    ///
    /// 逐字（或折叠键逐字）登记过的名字 → Exists；只是折叠同形 → Variant
    ///（同物异写只提名别名，绝不新铸第二个 URI）；多解 → Ambiguous。
    pub fn mint_check(&self, name: &str) -> MintVerdict {
        let name = name.trim();
        let folded = fold_surface(name);
        if folded.is_empty() {
            return MintVerdict::Fresh;
        }
        if let Some(record) = self.records.get(name) {
            if record.merged_into.is_some() {
                return MintVerdict::Redirect {
                    from: name.to_string(),
                    to: self.final_target(name),
                };
            }
            return MintVerdict::Exists {
                canonical: name.to_string(),
                reason: "规范身份已在册".to_string(),
            };
        }
        // 逐字登记过的别名（原词面或折叠键）：唯一 → 身份已存在；多解 → 先澄清
        let mut exact_owners: BTreeSet<String> = BTreeSet::new();
        for (canonical, record) in &self.records {
            if record.merged_into.is_some() || record.status == EntityStatus::Rejected {
                continue;
            }
            for (key, alias) in &record.aliases {
                if alias.status != EntityStatus::Confirmed {
                    continue;
                }
                if alias.surface == name || key == name {
                    exact_owners.insert(canonical.clone());
                }
            }
        }
        match exact_owners.len() {
            1 => {
                return MintVerdict::Exists {
                    canonical: exact_owners.into_iter().next().expect("非空"),
                    reason: "已确认别名".to_string(),
                }
            }
            2.. => {
                return MintVerdict::Ambiguous {
                    candidates: exact_owners.into_iter().collect(),
                }
            }
            _ => {}
        }
        match self.resolve(name) {
            EntityResolution::Canonical(canonical) => {
                return MintVerdict::Variant { canonical, folded }
            }
            EntityResolution::Redirected { from, to } => return MintVerdict::Redirect { from, to },
            EntityResolution::Ambiguous(candidates) => {
                return MintVerdict::Ambiguous { candidates }
            }
            EntityResolution::Unknown => {}
        }
        // 折叠同形：规范名或（待确认）提名别名命中同一折叠形 → 只提名别名
        let mut variants: BTreeSet<String> = BTreeSet::new();
        for (canonical, record) in &self.records {
            if record.merged_into.is_some() {
                continue;
            }
            let canonical_hit = fold_surface(canonical) == folded;
            let alias_hit = record.aliases.keys().any(|key| key == &folded);
            if canonical_hit || alias_hit {
                variants.insert(canonical.clone());
            }
        }
        match variants.len() {
            0 => MintVerdict::Fresh,
            1 => MintVerdict::Variant {
                canonical: variants.into_iter().next().expect("非空"),
                folded,
            },
            _ => MintVerdict::Ambiguous {
                candidates: variants.into_iter().collect(),
            },
        }
    }

    /// 显式确认别名（人类签字）：创建/转正记录与别名词面，清墓碑。
    pub fn confirm_alias(&mut self, surface: &str, canonical: &str) -> bool {
        self.confirm_alias_with_origin(surface, canonical, NominationOrigin::Dialogue)
    }

    pub fn confirm_alias_with_origin(
        &mut self,
        surface: &str,
        canonical: &str,
        origin: NominationOrigin,
    ) -> bool {
        let folded = fold_surface(surface);
        let canonical = canonical.trim();
        if folded.is_empty() || canonical.is_empty() {
            return false;
        }
        let target = match self.find_target(canonical) {
            Some(target) => target,
            None => {
                self.ensure_record(canonical, origin, "人工确认");
                canonical.to_string()
            }
        };
        if folded == fold_surface(&target) {
            let record = self.ensure_record(&target, origin, "人工确认");
            let changed = record.status != EntityStatus::Confirmed;
            record.status = EntityStatus::Confirmed;
            return changed;
        }
        let canonical_key = target.clone();
        let record = self
            .records
            .entry(canonical_key)
            .or_insert_with(|| EntityRecord {
                canonical: target.clone(),
                status: EntityStatus::Pending,
                origins: BTreeSet::new(),
                types: BTreeSet::new(),
                aliases: BTreeMap::new(),
                sources: BTreeSet::new(),
                merged_into: None,
                observed: false,
            });
        let record_changed = record.status != EntityStatus::Confirmed;
        record.status = EntityStatus::Confirmed;
        record.origins.insert(origin);
        record.sources.insert("人工确认".to_string());
        let alias = record.aliases.entry(folded).or_insert_with(|| AliasRecord {
            surface: surface.trim().to_string(),
            status: EntityStatus::Pending,
            origins: BTreeSet::new(),
            evidence: Vec::new(),
            nominations: 0,
            variants: BTreeSet::new(),
        });
        let alias_changed = alias.status != EntityStatus::Confirmed;
        alias.status = EntityStatus::Confirmed;
        alias.origins.insert(origin);
        push_evidence(&mut alias.evidence, "人工确认");
        if alias.surface != surface.trim() {
            alias.variants.insert(surface.trim().to_string());
        }
        record_changed || alias_changed
    }

    /// 拒绝别名：落墓碑，不予自动提名，也不参与解析。
    pub fn reject_alias(&mut self, surface: &str, canonical: &str) -> bool {
        let folded = fold_surface(surface);
        let Some(target) = self.find_target(canonical) else {
            return false;
        };
        let Some(record) = self.records.get_mut(&target) else {
            return false;
        };
        let Some(alias) = record.aliases.get_mut(&folded) else {
            return false;
        };
        let changed = alias.status != EntityStatus::Rejected;
        alias.status = EntityStatus::Rejected;
        alias.origins.insert(NominationOrigin::Dialogue);
        push_evidence(&mut alias.evidence, "用户拒绝");
        changed
    }

    /// 忘记别名：撤销已确认别名（含待确认提名），落墓碑。
    /// 返回受影响的规范身份（排序）。
    pub fn forget_alias(&mut self, surface: &str) -> Vec<String> {
        let folded = fold_surface(surface);
        if folded.is_empty() {
            return Vec::new();
        }
        let mut affected = Vec::new();
        for (canonical, record) in &mut self.records {
            if record.merged_into.is_some() {
                continue;
            }
            if let Some(alias) = record.aliases.get_mut(&folded) {
                if alias.status != EntityStatus::Rejected {
                    alias.status = EntityStatus::Rejected;
                    alias.origins.insert(NominationOrigin::Dialogue);
                    push_evidence(&mut alias.evidence, "用户忘记");
                    affected.push(canonical.clone());
                }
            }
        }
        affected.sort();
        affected
    }

    /// 合并身份（重命名 = 合并 + 重定向；URI 永不回收）。
    pub fn merge(&mut self, from: &str, into: &str) -> Result<MergeRecord, String> {
        let from = from.trim();
        let into = into.trim();
        if from.is_empty() || into.is_empty() {
            return Err("合并双方都不能为空".to_string());
        }
        let from_target = match self.find_target(from) {
            Some(target) => target,
            None => {
                self.ensure_record(from, NominationOrigin::Dialogue, "人工合并");
                from.to_string()
            }
        };
        let into_target = match self.find_target(into) {
            Some(target) => target,
            None => {
                self.ensure_record(into, NominationOrigin::Dialogue, "人工合并");
                into.to_string()
            }
        };
        if from_target == into_target {
            return Err(format!("「{}」与「{}」已指向同一身份", from, into));
        }
        // 环检测：沿 into 的链前进，若回到 from 则拒绝
        let mut cursor = into_target.clone();
        let mut seen = BTreeSet::new();
        while seen.insert(cursor.clone()) {
            let Some(next) = self
                .records
                .get(&cursor)
                .and_then(|record| record.merged_into.clone())
            else {
                break;
            };
            if next == from_target {
                return Err(format!(
                    "合并会形成重定向环：{} → … → {}",
                    into_target, from_target
                ));
            }
            cursor = next;
        }
        let into_types_before = self
            .records
            .get(&into_target)
            .map(|record| record.types.clone())
            .unwrap_or_default();
        let (moved_aliases, from_aliases, from_types) = {
            let Some(from_record) = self.records.get_mut(&from_target) else {
                return Err(format!("找不到身份「{}」", from));
            };
            let aliases = std::mem::take(&mut from_record.aliases);
            let types = std::mem::take(&mut from_record.types);
            let types_snapshot = types.clone();
            let origins = from_record.origins.clone();
            let moved = aliases.len();
            from_record.merged_into = Some(into_target.clone());
            let into_record =
                self.records
                    .entry(into_target.clone())
                    .or_insert_with(|| EntityRecord {
                        canonical: into_target.clone(),
                        status: EntityStatus::Confirmed,
                        origins: BTreeSet::new(),
                        types: BTreeSet::new(),
                        aliases: BTreeMap::new(),
                        sources: BTreeSet::new(),
                        merged_into: None,
                        observed: false,
                    });
            into_record.status = EntityStatus::Confirmed;
            into_record.origins.extend(origins);
            into_record.types.extend(types);
            into_record
                .sources
                .insert(format!("合并自「{}」", from_target));
            (moved, aliases, types_snapshot)
        };
        {
            let into_record = self
                .records
                .get_mut(&into_target)
                .ok_or_else(|| format!("找不到身份「{}」", into))?;
            for (folded, incoming) in from_aliases {
                match into_record.aliases.get_mut(&folded) {
                    Some(existing) => {
                        if incoming.status == EntityStatus::Confirmed {
                            existing.status = EntityStatus::Confirmed;
                        }
                        existing.origins.extend(incoming.origins);
                        for evidence in incoming.evidence {
                            push_evidence(&mut existing.evidence, &evidence);
                        }
                        existing.nominations += incoming.nominations;
                        for variant in incoming.variants {
                            if variant != existing.surface {
                                existing.variants.insert(variant);
                            }
                        }
                    }
                    None => {
                        into_record.aliases.insert(folded, incoming);
                    }
                }
            }
        }
        let record = MergeRecord {
            from: from_target,
            into: into_target,
            moved_aliases,
            from_types,
            into_types: into_types_before,
        };
        self.merges.push(record.clone());
        Ok(record)
    }

    /// 从配置声明实体画像（规范名 + 别名 + 类型，均视为已确认的显式声明）。
    pub fn declare(&mut self, config: &EntityProfileConfig) -> bool {
        let name = config.name.trim();
        if name.is_empty() {
            return false;
        }
        let existed = self.records.contains_key(name);
        {
            let record = self.ensure_record(name, NominationOrigin::Config, "配置声明");
            record.status = EntityStatus::Confirmed;
            for type_name in &config.etype {
                let type_name = type_name.trim();
                if !type_name.is_empty() {
                    record.types.insert(type_name.to_string());
                }
            }
        }
        for alias in &config.aliases {
            if fold_surface(alias) != fold_surface(name) {
                self.confirm_alias_with_origin(alias, name, NominationOrigin::Config);
            }
        }
        !existed
    }

    /// 回声选择：本会话用过的无歧义词面 > 已确认持久别名 > 规范名。
    ///
    /// 生成→再识别律：返回的词面必须能被 `resolve` 解析回同一身份，
    /// 否则回退规范名（并说明原因），绝不渲染一个解析不回来的词面。
    pub fn surface_choice(&self, canonical: &str, echo: &[String]) -> (String, String) {
        let canonical = canonical.trim();
        if canonical.is_empty() {
            return (String::new(), "空身份".to_string());
        }
        for surface in echo {
            let surface = surface.trim();
            if surface.is_empty() || surface == canonical {
                continue;
            }
            if let EntityResolution::Canonical(id) = self.resolve(surface) {
                if id == canonical {
                    return (
                        surface.to_string(),
                        format!("回声：本会话用过「{}」", surface),
                    );
                }
            }
        }
        if let Some(record) = self.records.get(canonical) {
            let mut candidates: Vec<&AliasRecord> = record
                .aliases
                .values()
                .filter(|alias| {
                    alias.status == EntityStatus::Confirmed
                        && matches!(
                            self.resolve(&alias.surface),
                            EntityResolution::Canonical(ref id) if id == canonical
                        )
                })
                .collect();
            candidates.sort_by(|left, right| {
                left.surface
                    .chars()
                    .count()
                    .cmp(&right.surface.chars().count())
                    .then_with(|| left.surface.cmp(&right.surface))
            });
            if let Some(alias) = candidates.first() {
                return (alias.surface.clone(), "持久别名".to_string());
            }
        }
        (canonical.to_string(), "规范名".to_string())
    }

    /// 确定性审计视图（可 golden）。
    pub fn audit_lines(&self) -> Vec<String> {
        let confirmed_aliases: usize = self
            .records
            .values()
            .filter(|record| record.merged_into.is_none())
            .map(|record| {
                record
                    .aliases
                    .values()
                    .filter(|alias| alias.status == EntityStatus::Confirmed)
                    .count()
            })
            .sum();
        let pending = self.pending_aliases();
        let rejected = self.rejected_aliases();
        let mut lines = vec![format!(
            "实体身份注册表：{} 个规范身份 / {} 条已确认别名 / {} 条待确认 / {} 条已拒绝（墓碑）/ {} 次合并。",
            self.len(),
            confirmed_aliases,
            pending.len(),
            rejected.len(),
            self.merges.len()
        )];
        for (canonical, record) in &self.records {
            let types = if record.types.is_empty() {
                String::new()
            } else {
                format!(
                    "；类型：{}",
                    record.types.iter().cloned().collect::<Vec<_>>().join("、")
                )
            };
            let origin = if record.origins.is_empty() {
                String::new()
            } else {
                format!(
                    "；来源：{}",
                    record
                        .origins
                        .iter()
                        .map(|origin| origin.label())
                        .collect::<Vec<_>>()
                        .join("、")
                )
            };
            let redirect = record
                .merged_into
                .as_ref()
                .map(|target| format!("；已合并 → {}", target))
                .unwrap_or_default();
            lines.push(format!(
                "- {}（{}{}{}{}）",
                canonical,
                record.status.label(),
                types,
                origin,
                redirect
            ));
            for alias in record.aliases.values() {
                if alias.status == EntityStatus::Rejected {
                    continue;
                }
                let variants = if alias.variants.is_empty() {
                    String::new()
                } else {
                    format!(
                        "；写法：{}",
                        alias
                            .variants
                            .iter()
                            .cloned()
                            .collect::<Vec<_>>()
                            .join("、")
                    )
                };
                let evidence = if alias.evidence.is_empty() {
                    String::new()
                } else {
                    format!("；证据：{}", alias.evidence.join("；"))
                };
                lines.push(format!(
                    "    别名「{}」（{}；{} 次提名{}）{}",
                    alias.surface,
                    alias.status.label(),
                    alias.nominations,
                    variants,
                    evidence
                ));
            }
        }
        if !pending.is_empty() {
            lines.push(format!(
                "待确认提名 {} 条（可用「确认身份：别名 = 规范名」签字）：",
                pending.len()
            ));
            for (canonical, surface) in &pending {
                lines.push(format!("  • 「{}」→「{}」", surface, canonical));
            }
        }
        if !rejected.is_empty() {
            lines.push(format!(
                "墓碑 {} 条（拒绝/忘记的别名不予自动提名）：",
                rejected.len()
            ));
            for (canonical, surface) in &rejected {
                lines.push(format!("  • 「{}」✗「{}」", surface, canonical));
            }
        }
        for record in &self.merges {
            lines.push(format!(
                "  ▸ 合并：{} → {}（搬迁别名 {} 条）",
                record.from, record.into, record.moved_aliases
            ));
        }
        lines
    }

    pub fn report(&self) -> String {
        self.audit_lines().join("\n")
    }

    /// 序列化快照（身份跨会话留存；规范 JSON）。
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// 解码并过版本门：折叠算法版本高于本程序支持时拒绝（不静默降级）。
    pub fn from_json(text: &str) -> Option<Self> {
        let registry: Self = serde_json::from_str(text).ok()?;
        if !registry.version_supported() {
            return None;
        }
        Some(registry)
    }

    pub fn save_to_path(&self, path: &str) -> std::io::Result<()> {
        std::fs::write(path, self.to_json())
    }

    pub fn load_from_path(path: &str) -> Option<Self> {
        let text = std::fs::read_to_string(path).ok()?;
        Self::from_json(&text)
    }
}

fn push_evidence(evidence: &mut Vec<String>, item: &str) {
    let item = item.trim();
    if item.is_empty() || evidence.iter().any(|existing| existing == item) {
        return;
    }
    evidence.push(item.to_string());
    evidence.sort();
}

/// 文本模式抽出的别名关键词（长关键词优先，避免截断）。
const ALIAS_KEYWORDS: [&str; 7] = ["全称", "简称", "又称", "又名", "俗称", "别称", "即"];

/// 从正文抽取别名候选（只提名，不写库；`即` 有词缀保护）。
///
/// 支持形态：`A（简称B）`、`A，简称B`、`A又称B`、`A的别称是B`、
/// `A，即B`、`A的全称是B` 等。canonical 取关键词前的实体片段，
/// surface 取关键词后的别名片段，两侧都做折叠规约并做长度护栏。
pub fn extract_alias_candidates(text: &str) -> Vec<AliasCandidate> {
    let mut candidates: Vec<AliasCandidate> = Vec::new();
    // 以字节位置扫描关键词
    for (position, _) in text.char_indices() {
        for keyword in ALIAS_KEYWORDS {
            if !text[position..].starts_with(keyword) {
                continue;
            }
            if keyword == "即" {
                // 词缀保护：即使/即便/即刻/即将/即令/即可/即为 等不是别名模式
                let next = text[position + keyword.len()..].chars().next();
                if let Some(next) = next {
                    if matches!(next, '使' | '便' | '刻' | '将' | '令' | '可' | '为' | '是')
                    {
                        continue;
                    }
                }
            }
            let prefix = &text[..position];
            let suffix = &text[position + keyword.len()..];
            let Some(canonical) = clean_canonical(prefix) else {
                continue;
            };
            let Some(surface) = clean_alias(suffix) else {
                continue;
            };
            if fold_surface(&canonical) == fold_surface(&surface) {
                continue;
            }
            if canonical.chars().count() > 24 || surface.chars().count() > 24 {
                continue;
            }
            let evidence_start = position.saturating_sub(12);
            let evidence_end =
                (position + keyword.len() + surface.chars().count() + 4).min(text.len());
            let mut start = evidence_start;
            while start > 0 && !text.is_char_boundary(start) {
                start -= 1;
            }
            let mut end = evidence_end;
            while end < text.len() && !text.is_char_boundary(end) {
                end += 1;
            }
            let evidence = text[start..end].trim().to_string();
            let candidate = AliasCandidate {
                surface,
                canonical,
                pattern: keyword.to_string(),
                evidence,
            };
            if !candidates.iter().any(|existing| {
                existing.surface == candidate.surface && existing.canonical == candidate.canonical
            }) {
                candidates.push(candidate);
            }
        }
    }
    candidates.sort_by(|left, right| {
        left.canonical
            .cmp(&right.canonical)
            .then_with(|| left.surface.cmp(&right.surface))
    });
    candidates
}

/// 关键词前的实体片段：截到最近的分隔符，去掉尾部的助词与开引号。
fn clean_canonical(prefix: &str) -> Option<String> {
    const DELIMITERS: [char; 17] = [
        '，', ',', '。', '；', ';', '！', '!', '？', '?', '：', ':', '\n', '（', '(', '《', '「',
        '"',
    ];
    let mut segment =
        prefix.trim_end_matches(|ch: char| DELIMITERS.contains(&ch) || ch.is_whitespace());
    if let Some(position) = segment.rfind(|ch| DELIMITERS.contains(&ch)) {
        segment = &segment[position + ch_len(segment, position)..];
    }
    let segment = segment
        .trim()
        .trim_end_matches(['的', '是', '为', '叫', '作', '、', ' '])
        .trim();
    let canonical = fold_surface(segment);
    if canonical.chars().count() < 2 || canonical.chars().count() > 24 {
        return None;
    }
    Some(canonical)
}

fn ch_len(text: &str, byte_index: usize) -> usize {
    text[byte_index..]
        .chars()
        .next()
        .map(char::len_utf8)
        .unwrap_or(0)
}

/// 关键词后的别名片段：截到分隔符/右包裹符，跳过「是/为」。
fn clean_alias(suffix: &str) -> Option<String> {
    const TERMINATORS: [char; 18] = [
        '，', ',', '。', '；', ';', '！', '!', '？', '?', '：', ':', '\n', '（', '(', ')', '）',
        '》', '」',
    ];
    let suffix = suffix.trim_start();
    let suffix = suffix
        .strip_prefix('是')
        .or_else(|| suffix.strip_prefix('为'))
        .or_else(|| suffix.strip_prefix('叫'))
        .or_else(|| suffix.strip_prefix("称为"))
        .unwrap_or(suffix);
    let suffix = suffix.trim_start();
    let end = suffix
        .char_indices()
        .find(|(_, ch)| TERMINATORS.contains(ch))
        .map(|(index, _)| index)
        .unwrap_or(suffix.len());
    let raw = suffix[..end].trim();
    let alias = fold_surface(raw);
    if alias.is_empty() || alias.chars().count() > 24 {
        return None;
    }
    Some(alias)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::type_system::TypeSystemConfig;

    fn origin_kb() -> NominationOrigin {
        NominationOrigin::Kb
    }

    #[test]
    fn fold_is_lossless_and_idempotent() {
        assert_eq!(fold_surface("  ＡＣＭＥ公司  "), "ACME公司");
        assert_eq!(fold_surface("ACME 公司"), "ACME公司");
        assert_eq!(fold_surface("ACME公司"), "ACME公司");
        assert_eq!(fold_surface("John Smith"), "John Smith", "拉丁词间空格保留");
        assert_eq!(fold_surface("《苹果公司》"), "苹果公司");
        assert_eq!(fold_surface("「『苹果』」"), "苹果");
        assert_eq!(fold_surface("（苹果公司）"), "苹果公司");
        assert_eq!(fold_surface("苹\u{200B}果公司"), "苹果公司");
        assert_eq!(fold_surface("苹果\u{3000}公司"), "苹果公司");
        for name in ["苹果公司", "ACME 公司", "《甲》", " 乙 丙 ", "John Smith"] {
            let once = fold_surface(name);
            assert_eq!(fold_surface(&once), once, "折叠必须幂等：{}", name);
        }
    }

    #[test]
    fn nomination_is_pending_until_confirmed() {
        let mut registry = EntityRegistry::new();
        let outcome =
            registry.nominate_alias("苹果", "苹果公司", NominationOrigin::Import, "又称模式");
        assert_eq!(outcome.kind, NominationKind::Pending);
        assert!(outcome.changed);
        assert_eq!(
            registry.resolve("苹果"),
            EntityResolution::Unknown,
            "提名不参与解析"
        );
        assert_eq!(
            registry.resolve("苹果公司"),
            EntityResolution::Unknown,
            "身份也未确认"
        );

        assert!(registry.confirm_alias("苹果", "苹果公司"));
        assert_eq!(
            registry.resolve("苹果"),
            EntityResolution::Canonical("苹果公司".to_string())
        );
        assert_eq!(
            registry.resolve("ＡＣＭＥ公司"),
            EntityResolution::Unknown,
            "折叠只对已登记词面生效"
        );
    }

    #[test]
    fn kb_observation_auto_confirms_and_is_the_only_automatic_source() {
        let mut registry = EntityRegistry::new();
        let mut types = BTreeSet::new();
        types.insert("组织".to_string());
        assert!(registry.observe_kb_entity("Apple", &types, &["苹果公司".to_string()], "知识图谱"));
        assert_eq!(
            registry.resolve("苹果公司"),
            EntityResolution::Canonical("Apple".to_string())
        );
        let record = registry.record("Apple").unwrap();
        assert_eq!(record.status, EntityStatus::Confirmed);
        let alias = record.aliases.values().next().unwrap();
        assert_eq!(alias.status, EntityStatus::Confirmed);
        assert!(alias.origins.contains(&origin_kb()));
    }

    #[test]
    fn tombstone_blocks_renomination_but_kb_overrides() {
        let mut registry = EntityRegistry::new();
        registry.nominate_alias("苹果", "苹果公司", NominationOrigin::Import, "模式");
        assert!(registry.reject_alias("苹果", "苹果公司"));
        let outcome = registry.nominate_alias("苹果", "苹果公司", NominationOrigin::Import, "模式");
        assert_eq!(outcome.kind, NominationKind::Tombstoned);
        assert!(!outcome.changed);
        let outcome = registry.nominate_alias("苹果", "苹果公司", NominationOrigin::Kb, "图谱");
        assert_eq!(outcome.kind, NominationKind::Confirmed, "图谱事实强于否决");
        assert_eq!(
            registry.resolve("苹果"),
            EntityResolution::Canonical("苹果公司".to_string())
        );
    }

    #[test]
    fn forget_alias_removes_resolution_and_leaves_tombstone() {
        let mut registry = EntityRegistry::new();
        registry.confirm_alias("苹果", "苹果公司");
        assert_eq!(registry.forget_alias("苹果"), vec!["苹果公司".to_string()]);
        assert_eq!(registry.resolve("苹果"), EntityResolution::Unknown);
        assert_eq!(
            registry.forget_alias("苹果"),
            Vec::<String>::new(),
            "重复忘记无变化"
        );
        let outcome = registry.nominate_alias("苹果", "苹果公司", NominationOrigin::Import, "模式");
        assert_eq!(outcome.kind, NominationKind::Tombstoned);
    }

    #[test]
    fn mint_check_is_the_constitution_of_uri_minting() {
        let mut registry = EntityRegistry::new();
        assert_eq!(registry.mint_check("苹果公司"), MintVerdict::Fresh);

        registry.confirm_alias("苹果", "苹果公司");
        assert!(matches!(
            registry.mint_check("苹果"),
            MintVerdict::Exists { ref canonical, .. } if canonical == "苹果公司"
        ));
        assert!(matches!(
            registry.mint_check("苹果公司"),
            MintVerdict::Exists { .. }
        ));
        // 折叠同形：ACME 公司 ≡ ＡＣＭＥ公司
        registry.confirm_alias("ACME公司", "ACME");
        assert!(matches!(
            registry.mint_check("ＡＣＭＥ公司"),
            MintVerdict::Variant { ref canonical, .. } if canonical == "ACME"
        ));
        // 合并后：改名重定向，新名存在
        let record = registry.merge("ACME", "宏基").unwrap();
        assert_eq!(record.moved_aliases, 1);
        assert!(matches!(
            registry.mint_check("ACME"),
            MintVerdict::Redirect { ref to, .. } if to == "宏基"
        ));
        assert!(matches!(
            registry.mint_check("宏基"),
            MintVerdict::Exists { .. }
        ));
    }

    #[test]
    fn merge_moves_aliases_and_redirects_former_names() {
        let mut registry = EntityRegistry::new();
        registry.confirm_alias("苹果", "苹果股份有限公司");
        registry.confirm_alias("苹果有限", "苹果股份有限公司");
        registry.merge("苹果股份有限公司", "苹果公司").unwrap();
        assert_eq!(
            registry.resolve("苹果股份有限公司"),
            EntityResolution::Redirected {
                from: "苹果股份有限公司".to_string(),
                to: "苹果公司".to_string(),
            }
        );
        assert_eq!(
            registry.resolve("苹果"),
            EntityResolution::Canonical("苹果公司".to_string()),
            "别名随合并搬迁"
        );
        assert!(
            registry.merge("苹果公司", "苹果股份有限公司").is_err(),
            "成环拒绝"
        );
        assert!(
            registry.merge("苹果公司", "苹果公司").is_err(),
            "自合并拒绝"
        );
    }

    #[test]
    fn alias_to_ambiguous_surface_is_reported_not_guessed() {
        let mut registry = EntityRegistry::new();
        registry.confirm_alias("苹果", "苹果公司");
        registry.confirm_alias("苹果", "苹果果实");
        assert_eq!(
            registry.resolve("苹果"),
            EntityResolution::Ambiguous(vec!["苹果公司".to_string(), "苹果果实".to_string()])
        );
        assert!(matches!(
            registry.mint_check("苹果"),
            MintVerdict::Ambiguous { .. }
        ));
    }

    #[test]
    fn type_gate_turns_cross_type_alias_into_human_confirmation() {
        let mut types = TypeSystemConfig::default();
        types.disjoint = vec![crate::type_system::TypeDisjointConfig {
            left: "组织".to_string(),
            right: "水果".to_string(),
        }];
        let type_system = TypeSystem::from_config(&types);

        let mut registry = EntityRegistry::new();
        let mut fruit_types = BTreeSet::new();
        fruit_types.insert("水果".to_string());
        let mut company_types = BTreeSet::new();
        company_types.insert("组织".to_string());
        registry.observe_kb_entity("苹果果实", &fruit_types, &[], "知识图谱");
        registry.observe_kb_entity("苹果公司", &company_types, &[], "知识图谱");

        let outcome = registry.nominate_alias_with_types(
            "苹果",
            "苹果公司",
            NominationOrigin::Import,
            "模式",
            Some(&type_system),
        );
        assert_eq!(
            outcome.kind,
            NominationKind::Pending,
            "词面未被占用时正常提名"
        );

        // 词面「苹果果实」要成为「苹果公司」的别名：互斥类型 → 需人工确认
        let outcome = registry.nominate_alias_with_types(
            "苹果果实",
            "苹果公司",
            NominationOrigin::Import,
            "模式",
            Some(&type_system),
        );
        assert_eq!(outcome.kind, NominationKind::TypeConflict);
        assert!(!outcome.changed);
        assert_eq!(
            registry.resolve("苹果果实"),
            EntityResolution::Canonical("苹果果实".to_string()),
            "冲突提名不得改写解析"
        );
    }

    #[test]
    fn declared_config_aliases_are_confirmed() {
        let mut registry = EntityRegistry::new();
        let config = EntityProfileConfig {
            name: "苹果公司".to_string(),
            aliases: vec!["苹果".to_string()],
            etype: vec!["组织".to_string()],
            description: None,
        };
        assert!(registry.declare(&config));
        assert!(!registry.declare(&config), "重复声明不予新增");
        assert_eq!(
            registry.resolve("苹果"),
            EntityResolution::Canonical("苹果公司".to_string())
        );
        assert!(registry.record("苹果公司").unwrap().types.contains("组织"));
    }

    #[test]
    fn surface_choice_prefers_echo_then_persistent_alias_then_canonical() {
        let mut registry = EntityRegistry::new();
        registry.confirm_alias("苹果", "苹果公司");
        registry.confirm_alias("苹果股份有限公司", "苹果公司");

        let (surface, reason) = registry.surface_choice("苹果公司", &[]);
        assert_eq!(surface, "苹果", "最短已确认别名优先（字典序平局）");
        assert_eq!(reason, "持久别名");

        let (surface, reason) =
            registry.surface_choice("苹果公司", &["苹果股份有限公司".to_string()]);
        assert_eq!(surface, "苹果股份有限公司");
        assert!(reason.contains("回声"));

        let (surface, _reason) = registry.surface_choice("苹果公司", &["香蕉".to_string()]);
        assert_eq!(surface, "苹果", "回声解析不回同一身份时不采用");

        // 生成→再识别律：选择的词面必须解析回同一身份
        let chosen = registry.surface_choice("苹果公司", &["苹果".to_string()]);
        assert_eq!(
            registry.resolve(&chosen.0),
            EntityResolution::Canonical("苹果公司".to_string())
        );

        let (surface, reason) = registry.surface_choice("未名实体", &[]);
        assert_eq!(surface, "未名实体");
        assert_eq!(reason, "规范名");
    }

    #[test]
    fn alias_variants_share_one_identity() {
        let mut registry = EntityRegistry::new();
        registry.nominate_alias("ＡＣＭＥ公司", "ACME", NominationOrigin::Import, "模式一");
        let outcome =
            registry.nominate_alias("ACME 公司", "ACME", NominationOrigin::Import, "模式二");
        assert_eq!(outcome.kind, NominationKind::Pending);
        assert!(!outcome.changed, "同折叠形不重复建条目");
        let record = registry.record("ACME").unwrap();
        let alias = record.aliases.values().next().unwrap();
        assert_eq!(alias.surface, "ＡＣＭＥ公司");
        assert!(alias.variants.contains("ACME 公司"));
        assert_eq!(record.aliases.len(), 1);
    }

    #[test]
    fn text_patterns_extract_alias_candidates() {
        let candidates = extract_alias_candidates("苹果公司，简称苹果，是一家科技公司。");
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].surface, "苹果");
        assert_eq!(candidates[0].canonical, "苹果公司");
        assert_eq!(candidates[0].pattern, "简称");

        let candidates = extract_alias_candidates("中国科学院（简称中科院）成立于1949年");
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].canonical, "中国科学院");
        assert_eq!(candidates[0].surface, "中科院");

        let candidates = extract_alias_candidates("腾讯公司又称腾讯");
        assert_eq!(candidates[0].canonical, "腾讯公司");

        let candidates = extract_alias_candidates("阿里巴巴的全称是阿里巴巴集团");
        assert_eq!(candidates[0].surface, "阿里巴巴集团");

        // 词缀保护：即使不是别名模式
        assert!(extract_alias_candidates("即使下雨也要去").is_empty());
        assert!(extract_alias_candidates("没有任何模式的句子").is_empty());
    }

    #[test]
    fn audit_lines_are_deterministic_and_bounded() {
        let mut registry = EntityRegistry::new();
        registry.confirm_alias("苹果", "苹果公司");
        registry.nominate_alias("菠萝", "苹果公司", NominationOrigin::Import, "模式");
        registry.reject_alias("菠萝", "苹果公司");
        let lines = registry.audit_lines();
        assert_eq!(lines, registry.audit_lines());
        assert!(lines[0].contains("实体身份注册表"));
        assert!(lines.iter().any(|line| line.contains("别名「苹果」")));
        assert!(lines.iter().any(|line| line.contains("墓碑 1 条")));
    }

    #[test]
    fn persistence_roundtrip_and_version_gate() {
        let mut registry = EntityRegistry::new();
        registry.confirm_alias("苹果", "苹果公司");
        registry.nominate_alias("菠萝", "苹果公司", NominationOrigin::Import, "模式");
        let json = registry.to_json();
        let restored = EntityRegistry::from_json(&json).expect("往返应成功");
        assert_eq!(restored.resolve("苹果"), registry.resolve("苹果"));
        assert_eq!(
            restored.to_json(),
            registry.to_json(),
            "规范 JSON 逐字节一致"
        );

        let mut future = serde_json::to_value(&registry).unwrap();
        future["fold_version"] = serde_json::json!(FOLD_VERSION + 1);
        assert!(
            EntityRegistry::from_json(&future.to_string()).is_none(),
            "高版本必须拒绝解码"
        );
        assert!(EntityRegistry::from_json("{not json").is_none());
        assert!(
            EntityRegistry::from_json("{}").is_some(),
            "空对象按默认值解码"
        );
    }

    #[test]
    fn empty_and_self_nominations_are_ignored() {
        let mut registry = EntityRegistry::new();
        let outcome = registry.nominate_alias("  ", "苹果公司", NominationOrigin::Import, "模式");
        assert_eq!(outcome.kind, NominationKind::Ignored);
        let outcome =
            registry.nominate_alias("ＡＣＭＥ公司", "ACME公司", NominationOrigin::Import, "模式");
        assert_eq!(
            outcome.kind,
            NominationKind::Duplicate,
            "折叠同形视为自别名"
        );
        assert!(EntityRegistry::new().mint_check("").eq(&MintVerdict::Fresh));
    }

    #[test]
    fn confirmed_alias_pairs_are_sorted_and_deduplicated() {
        let mut registry = EntityRegistry::new();
        registry.confirm_alias("苹果", "苹果公司");
        registry.confirm_alias("苹果", "苹果公司");
        registry.nominate_alias("菠萝", "苹果公司", NominationOrigin::Import, "模式");
        assert_eq!(
            registry.confirmed_alias_pairs(),
            vec![("苹果".to_string(), "苹果公司".to_string())]
        );
        assert_eq!(
            registry.pending_aliases(),
            vec![("苹果公司".to_string(), "菠萝".to_string())]
        );
    }
}

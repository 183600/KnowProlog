//! 对话式可逆模式锻造（Pattern Forge）。
//!
//! 教学模式要求作者直接写出 IR 要素；对话诱导模式只能在固定的
//! 「S 的 P 是 O」骨架上累计支持度。两者之间缺一层：**从一两个例子
//! 归纳出可审计的 token 正则，再展开成现有 [`pattern_ir`] 序列**。
//!
//! 本模块的边界刻意收得很紧：
//!
//! - 词级/类型级而非字符级：复用 `jieba` 分词、词性与既有匹配器，
//!   不切碎人名，也不引入字符正则的回溯风险；
//! - 只归纳词面与实体槽，不猜关系谓词：`姚明的女儿是谁` 得到
//!   `{人物:人名} 的 女儿 是 谁`，而不是把「女儿」泛化成任意属性；
//! - **反例是硬约束**：任一展开命中反例就拒绝学习，并明确报告
//!   「这两句话我分不开」，绝不悄悄泛化；
//! - 正则只活在作者/解释层：运行时仍展开为 ≤16 条无分支 IR 序列，
//!   交给现有线性匹配器；
//! - 新规则先是候选，显式确认后才以 `induced#forge#…` 来源参与路由，
//!   并带正例、反例、探针、命中/反驳计数与撤回入口。

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::context_image::fnv1a_hex;
use crate::nlp::{preprocess, TaggedWord};
use crate::pattern_ir::{Elem, PatternEntry, PatternLibrary};
use crate::NlpConfig;

/// 锻造台账的序列化方言；版本不符时拒绝恢复。
pub const FORGE_DIALECT: &str = "knowprolog-forge/1";
/// 一条正则最多展开成多少条无分支 IR（防止规则爆炸）。
pub const MAX_EXPANSIONS: usize = 16;
/// 单个会话默认最多保留多少条锻造记录（含候选与已退役）。
pub const DEFAULT_MAX_PATTERNS: usize = 32;
/// 默认最多同时激活多少条锻造规则。
pub const DEFAULT_MAX_ACTIVE: usize = 8;
/// 连续被反驳多少次后自动退役（仍需调用方把条目移出模式库）。
pub const MAX_REFUTATIONS: usize = 3;
/// 来源前缀：模式库用它识别锻造条目。
pub const FORGE_SOURCE_PREFIX: &str = "induced#forge#";

// ---------------------------------------------------------------------------
// token 正则
// ---------------------------------------------------------------------------

/// token 正则的一个位置：词面、词性、类型槽，或这些的有限选择。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ForgeAtom {
    /// 固定词面。
    Literal { surface: String },
    /// 固定词性（`nr` / `ns` / `v` …）。
    Pos { pos: String },
    /// 带类型排序的实体槽。
    Slot {
        name: String,
        sort: String,
        #[serde(default)]
        hints: Vec<String>,
    },
    /// 任意一个 token（只应由显式构造产生，归纳不会自动使用）。
    Any,
    /// 有限选择；展开时做笛卡尔积并受 [`MAX_EXPANSIONS`] 约束。
    Alt { options: Vec<ForgeAtom> },
}

impl ForgeAtom {
    fn literal(surface: impl Into<String>) -> Self {
        ForgeAtom::Literal {
            surface: surface.into(),
        }
    }

    fn slot(name: impl Into<String>, sort: impl Into<String>) -> Self {
        ForgeAtom::Slot {
            name: name.into(),
            sort: sort.into(),
            hints: Vec::new(),
        }
    }

    pub fn render(&self) -> String {
        match self {
            ForgeAtom::Literal { surface } => surface.clone(),
            ForgeAtom::Pos { pos } => pos.clone(),
            ForgeAtom::Slot { name, sort, .. } => {
                if sort.is_empty() || sort == "*" {
                    format!("{{{}}}", name)
                } else {
                    format!("{{{}:{}}}", name, sort)
                }
            }
            ForgeAtom::Any => "_".to_string(),
            ForgeAtom::Alt { options } => format!(
                "({})",
                options
                    .iter()
                    .map(ForgeAtom::render)
                    .collect::<Vec<_>>()
                    .join("|")
            ),
        }
    }

    fn expansions(&self) -> Result<Vec<Vec<Elem>>, ForgeError> {
        match self {
            ForgeAtom::Literal { surface } => Ok(vec![vec![Elem::word(surface.clone())]]),
            ForgeAtom::Pos { pos } => Ok(vec![vec![Elem::pos(pos.clone())]]),
            ForgeAtom::Slot { name, sort, hints } => Ok(vec![vec![Elem::Slot {
                name: name.clone(),
                sort: sort.clone(),
                hints: hints.clone(),
            }]]),
            ForgeAtom::Any => Ok(vec![vec![Elem::Any]]),
            ForgeAtom::Alt { options } => {
                let mut out = Vec::new();
                for option in options {
                    for expansion in option.expansions()? {
                        if out.len() >= MAX_EXPANSIONS {
                            return Err(ForgeError::TooManyExpansions);
                        }
                        out.push(expansion);
                    }
                }
                if out.is_empty() {
                    return Err(ForgeError::EmptyAlternative);
                }
                Ok(out)
            }
        }
    }
}

/// 一条可读、可展开、可逆生成的 token 正则。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenRegex {
    pub atoms: Vec<ForgeAtom>,
}

impl TokenRegex {
    pub fn new(atoms: Vec<ForgeAtom>) -> Self {
        Self { atoms }
    }

    pub fn render(&self) -> String {
        self.atoms
            .iter()
            .map(ForgeAtom::render)
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// 用槽位填充值逆生成一句话；未填充的槽保留可读占位符。
    pub fn render_with(&self, fills: &BTreeMap<String, String>) -> String {
        let mut out = String::new();
        for atom in &self.atoms {
            match atom {
                ForgeAtom::Literal { surface } => out.push_str(surface),
                ForgeAtom::Pos { pos } => out.push_str(pos),
                ForgeAtom::Slot { name, .. } => {
                    if let Some(value) = fills.get(name) {
                        out.push_str(value);
                    } else {
                        out.push('{');
                        out.push_str(name);
                        out.push('}');
                    }
                }
                ForgeAtom::Any => out.push('某'),
                ForgeAtom::Alt { options } => {
                    if let Some(first) = options.first() {
                        match first {
                            ForgeAtom::Literal { surface } => out.push_str(surface),
                            ForgeAtom::Slot { name, .. } => {
                                if let Some(value) = fills.get(name) {
                                    out.push_str(value);
                                } else {
                                    out.push('{');
                                    out.push_str(name);
                                    out.push('}');
                                }
                            }
                            other => out.push_str(&other.render()),
                        }
                    }
                }
            }
        }
        out
    }

    /// 展开为无分支 IR 序列（≤ [`MAX_EXPANSIONS`] 条）。
    pub fn expand(&self) -> Result<Vec<Vec<Elem>>, ForgeError> {
        if self.atoms.is_empty() {
            return Err(ForgeError::EmptyPattern);
        }
        let mut acc: Vec<Vec<Elem>> = vec![Vec::new()];
        for atom in &self.atoms {
            let choices = atom.expansions()?;
            let mut next = Vec::new();
            for prefix in &acc {
                for choice in &choices {
                    let mut sequence = prefix.clone();
                    sequence.extend(choice.iter().cloned());
                    if next.len() >= MAX_EXPANSIONS {
                        return Err(ForgeError::TooManyExpansions);
                    }
                    next.push(sequence);
                }
            }
            acc = next;
        }
        Ok(acc)
    }

    /// 用与运行时相同的模式匹配器判断是否命中。
    pub fn matches(&self, tokens: &[TaggedWord]) -> bool {
        let Ok(expansions) = self.expand() else {
            return false;
        };
        expansions_match(&expansions, tokens)
    }
}

/// 展开后的任一条 IR 序列命中即算命中。
pub fn expansions_match(expansions: &[Vec<Elem>], tokens: &[TaggedWord]) -> bool {
    if expansions.is_empty() || tokens.is_empty() {
        return false;
    }
    let mut library = PatternLibrary::new();
    for (index, elems) in expansions.iter().enumerate() {
        library.add(PatternEntry::new(
            format!("锻造探针-{}", index + 1),
            "probe#forge",
            "probe",
            0,
            true,
            elems.clone(),
        ));
    }
    library.match_route(tokens).is_some()
}

// ---------------------------------------------------------------------------
// 请求、结果与错误
// ---------------------------------------------------------------------------

/// 一次锻造请求：一个主正例、可选同句式正例、零到多个反例、目标路由。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForgeRequest {
    pub name: String,
    pub query_type: String,
    pub priority: i32,
    pub positive: String,
    /// 同句式的更多正例：只在 token 数相同的位置上合并词面选择。
    pub extra_positives: Vec<String>,
    pub negatives: Vec<String>,
    pub guard: Option<crate::pattern_ir::ContextGuard>,
}

impl ForgeRequest {
    pub fn new(
        name: impl Into<String>,
        positive: impl Into<String>,
        query_type: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            query_type: query_type.into(),
            priority: 10,
            positive: positive.into(),
            extra_positives: Vec::new(),
            negatives: Vec::new(),
            guard: None,
        }
    }

    /// 追加一个同句式正例（如「姚明的女儿是谁」与「姚明的闺女是谁」）。
    pub fn with_positive(mut self, positive: impl Into<String>) -> Self {
        self.extra_positives.push(positive.into());
        self
    }

    /// 所有正例：主正例在前。
    pub fn positives(&self) -> Vec<&str> {
        std::iter::once(self.positive.as_str())
            .chain(self.extra_positives.iter().map(String::as_str))
            .collect()
    }

    pub fn with_negative(mut self, negative: impl Into<String>) -> Self {
        self.negatives.push(negative.into());
        self
    }

    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    pub fn with_guard(mut self, guard: crate::pattern_ir::ContextGuard) -> Self {
        self.guard = Some(guard);
        self
    }
}

/// 一条判别探针：正例应命中，反例应被排除，逆生成句应命中。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForgeProbe {
    pub sentence: String,
    pub expect_match: bool,
    pub passed: bool,
    pub origin: ProbeOrigin,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProbeOrigin {
    Positive,
    Negative,
    Generated,
}

impl ProbeOrigin {
    pub fn label(self) -> &'static str {
        match self {
            ProbeOrigin::Positive => "正例",
            ProbeOrigin::Negative => "反例",
            ProbeOrigin::Generated => "逆生成",
        }
    }
}

impl ForgeProbe {
    pub fn render(&self) -> String {
        format!(
            "{}「{}」{}",
            self.origin.label(),
            self.sentence,
            if self.passed { "通过" } else { "未通过" }
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForgeStatus {
    Candidate,
    Active,
    Retired,
}

impl ForgeStatus {
    pub fn label(self) -> &'static str {
        match self {
            ForgeStatus::Candidate => "候选",
            ForgeStatus::Active => "已激活",
            ForgeStatus::Retired => "已退役",
        }
    }
}

/// 一条锻造规则：正则 + 例子 + 探针 + 生命周期计数。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForgedPattern {
    pub id: String,
    pub name: String,
    pub query_type: String,
    pub priority: i32,
    pub regex: TokenRegex,
    pub expansions: Vec<Vec<Elem>>,
    pub slot_fills: BTreeMap<String, String>,
    pub positives: Vec<String>,
    pub negatives: Vec<String>,
    pub probes: Vec<ForgeProbe>,
    pub status: ForgeStatus,
    pub created_turn: usize,
    pub last_used_turn: usize,
    pub hits: usize,
    pub refutations: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub guard: Option<crate::pattern_ir::ContextGuard>,
}

impl ForgedPattern {
    pub fn source(&self) -> String {
        format!("{}{}", FORGE_SOURCE_PREFIX, self.id)
    }

    /// 把展开结果物化为统一模式库条目。名称带序号，便于逐条遗忘。
    pub fn materialize(&self) -> Vec<PatternEntry> {
        let active = self.status == ForgeStatus::Active;
        self.expansions
            .iter()
            .enumerate()
            .map(|(index, elems)| {
                let suffix = if self.expansions.len() > 1 {
                    format!("#{}", index + 1)
                } else {
                    String::new()
                };
                let mut entry = PatternEntry::new(
                    format!("锻造：{}{}", self.name, suffix),
                    self.source(),
                    self.query_type.clone(),
                    self.priority,
                    active,
                    elems.clone(),
                );
                entry.guard = self.guard.clone();
                entry
            })
            .collect()
    }

    pub fn render(&self) -> String {
        let mut text = format!(
            "- {}「{}」[{}] → {} · 优先级 {} · 展开 {} · 命中 {} / 反驳 {}\n  正则：{}\n  正例：{}\n  反例：{}",
            self.id,
            self.name,
            self.status.label(),
            self.query_type,
            self.priority,
            self.expansions.len(),
            self.hits,
            self.refutations,
            self.regex.render(),
            if self.positives.is_empty() {
                "（无）".to_string()
            } else {
                self.positives.join("；")
            },
            if self.negatives.is_empty() {
                "（无）".to_string()
            } else {
                self.negatives.join("；")
            }
        );
        if !self.probes.is_empty() {
            text.push_str("\n  探针：");
            text.push_str(
                &self
                    .probes
                    .iter()
                    .map(ForgeProbe::render)
                    .collect::<Vec<_>>()
                    .join("；"),
            );
        }
        text
    }
}

/// 锻造成功后的回执：候选记录 + 是否新建。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForgeOutcome {
    pub pattern: ForgedPattern,
    pub created: bool,
}

impl ForgeOutcome {
    pub fn render(&self) -> String {
        let pattern = &self.pattern;
        let mut lines = vec![
            format!(
                "{}锻造候选 {}「{}」→ {}（正则：{}）",
                if self.created { "" } else { "已更新" },
                pattern.id,
                pattern.name,
                pattern.query_type,
                pattern.regex.render()
            ),
            format!(
                "展开 {} 条无分支模式，{}；正例 {} 条，反例 {} 条已排除。",
                pattern.expansions.len(),
                if pattern.status == ForgeStatus::Candidate {
                    "尚未参与路由"
                } else {
                    "已参与路由"
                },
                pattern.positives.len(),
                pattern.negatives.len()
            ),
        ];
        for probe in &pattern.probes {
            lines.push(format!("  {}", probe.render()));
        }
        if pattern.status == ForgeStatus::Candidate {
            lines.push(format!(
                "确认后上线：确认锻造 {}（只影响本会话，可退役、可遗忘）",
                pattern.id
            ));
        }
        lines.join("\n")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForgeError {
    EmptyName,
    EmptyRoute,
    EmptyPositive,
    NoTokens,
    EmptyPattern,
    EmptyAlternative,
    TooManyExpansions,
    NegativeAccepted { negative: String, pattern: String },
    SelfReject,
    PositiveShapeMismatch { positive: String },
    PositiveNotCovered { positive: String },
    BudgetExhausted { max: usize },
    ActiveBudgetExhausted { max: usize },
    NotFound(String),
    Ambiguous(String),
    NotActive(String),
    AlreadyRetired(String),
    BadSpec(String),
}

impl std::fmt::Display for ForgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ForgeError::EmptyName => write!(f, "模式名称不能为空"),
            ForgeError::EmptyRoute => write!(f, "路由类型（query_type）不能为空"),
            ForgeError::EmptyPositive => write!(f, "正例不能为空"),
            ForgeError::NoTokens => write!(f, "正例没有可用的词元"),
            ForgeError::EmptyPattern => write!(f, "正则至少要有一个要素"),
            ForgeError::EmptyAlternative => write!(f, "选择分支不能为空"),
            ForgeError::TooManyExpansions => write!(
                f,
                "正则展开超过上限 {} 条；请收窄选择分支",
                MAX_EXPANSIONS
            ),
            ForgeError::NegativeAccepted { negative, pattern } => write!(
                f,
                "反例「{}」仍被模式「{}」命中：这两句话我分不开，拒绝学习",
                negative, pattern
            ),
            ForgeError::SelfReject => write!(f, "归纳出的模式没有命中正例，拒绝学习"),
            ForgeError::PositiveShapeMismatch { positive } => write!(
                f,
                "正例「{}」与主正例的 token 数不同；请先用同一句式，或拆成两条规则",
                positive
            ),
            ForgeError::PositiveNotCovered { positive } => write!(
                f,
                "合并后的模式没有覆盖正例「{}」，拒绝学习",
                positive
            ),
            ForgeError::BudgetExhausted { max } => {
                write!(f, "锻造记录已达上限 {} 条；请先退役或遗忘不用的规则", max)
            }
            ForgeError::ActiveBudgetExhausted { max } => {
                write!(f, "同时激活的锻造规则已达上限 {} 条；请先退役不用的规则", max)
            }
            ForgeError::NotFound(key) => write!(f, "没有找到锻造规则「{}」", key),
            ForgeError::Ambiguous(key) => {
                write!(f, "锻造规则「{}」前缀不唯一，请给出更长的 id", key)
            }
            ForgeError::NotActive(id) => write!(f, "锻造规则 {} 尚未激活", id),
            ForgeError::AlreadyRetired(id) => write!(f, "锻造规则 {} 已经退役", id),
            ForgeError::BadSpec(detail) => write!(f, "锻造指令格式错误：{}", detail),
        }
    }
}

impl std::error::Error for ForgeError {}

/// 激活结果：生命周期记录 + 应并入模式库的条目。
#[derive(Debug, Clone, PartialEq)]
pub struct ForgeActivation {
    pub pattern: ForgedPattern,
    pub entries: Vec<PatternEntry>,
}

// ---------------------------------------------------------------------------
// 归纳
// ---------------------------------------------------------------------------

fn is_punctuation(text: &str) -> bool {
    matches!(
        text,
        "？" | "?" | "。" | "，" | "、" | "；" | ";" | "！" | "!" | "："
    )
}

fn is_question_word(text: &str) -> bool {
    matches!(
        text,
        "谁" | "什么" | "哪" | "哪个" | "哪些" | "哪一位" | "多少" | "几" | "怎样" | "如何"
    )
}

fn is_numeric(text: &str) -> bool {
    !text.is_empty()
        && text
            .chars()
            .all(|ch| ch.is_ascii_digit() || "零一二三四五六七八九十百千万两".contains(ch))
}

fn is_date_like(text: &str) -> bool {
    text.contains('年')
        || text.contains('月')
        || text.contains('日')
        || text.contains('号')
        || text.contains('天')
}

fn next_slot_name(counts: &mut HashMap<&'static str, usize>, base: &'static str) -> String {
    let count = counts.entry(base).or_insert(0);
    *count += 1;
    if *count == 1 {
        base.to_string()
    } else {
        format!("{}{}", base, *count)
    }
}

/// 从一个正例归纳保守的 token 正则：实体/数量变槽，其余保持词面。
///
/// 这里只做「新实体、新数量」的泛化，不把属性名词泛化掉：`女儿` 仍是
/// 词面，因此不会把「妻子」「母亲」等不同关系误接到同一路由上。
fn learn_regex(tokens: &[TaggedWord]) -> (TokenRegex, BTreeMap<String, String>) {
    let mut atoms = Vec::new();
    let mut fills = BTreeMap::new();
    let mut counts: HashMap<&'static str, usize> = HashMap::new();
    for token in tokens {
        let text = token.text.trim();
        if text.is_empty() || is_punctuation(text) {
            continue;
        }
        let atom = if token.pos == "nr" && !is_question_word(text) {
            let name = next_slot_name(&mut counts, "人物");
            fills.insert(name.clone(), text.to_string());
            ForgeAtom::slot(name, "人名")
        } else if token.pos == "ns" && !is_question_word(text) {
            let name = next_slot_name(&mut counts, "地点");
            fills.insert(name.clone(), text.to_string());
            ForgeAtom::slot(name, "城市")
        } else if (token.pos == "m" || token.pos == "q") && is_numeric(text) {
            let name = next_slot_name(&mut counts, "数量");
            fills.insert(name.clone(), text.to_string());
            ForgeAtom::slot(name, "数量")
        } else if (token.pos == "m" || token.pos == "q")
            && is_date_like(text)
            && text.chars().any(|ch| ch.is_ascii_digit())
        {
            let name = next_slot_name(&mut counts, "时间");
            fills.insert(name.clone(), text.to_string());
            ForgeAtom::slot(name, "日期")
        } else {
            ForgeAtom::literal(text)
        };
        atoms.push(atom);
    }
    (TokenRegex::new(atoms), fills)
}

fn push_unique_atom(options: &mut Vec<ForgeAtom>, atom: ForgeAtom) {
    if !options.contains(&atom) {
        options.push(atom);
    }
}

/// 合并同一个位置上的两种写法：词面差异进入有限选择，实体槽保持类型约束。
fn merge_atom(left: &ForgeAtom, right: &ForgeAtom) -> ForgeAtom {
    if left == right {
        return left.clone();
    }
    match (left, right) {
        (ForgeAtom::Alt { options }, ForgeAtom::Alt { options: more }) => {
            let mut merged = options.clone();
            for atom in more {
                push_unique_atom(&mut merged, atom.clone());
            }
            ForgeAtom::Alt { options: merged }
        }
        (ForgeAtom::Alt { options }, atom) => {
            let mut merged = options.clone();
            push_unique_atom(&mut merged, atom.clone());
            ForgeAtom::Alt { options: merged }
        }
        (atom, ForgeAtom::Alt { options }) => {
            let mut merged = vec![atom.clone()];
            for option in options {
                push_unique_atom(&mut merged, option.clone());
            }
            ForgeAtom::Alt { options: merged }
        }
        (
            ForgeAtom::Slot { name, sort, hints },
            ForgeAtom::Slot {
                sort: other_sort, ..
            },
        ) if sort == other_sort => ForgeAtom::Slot {
            name: name.clone(),
            sort: sort.clone(),
            hints: hints.clone(),
        },
        (
            ForgeAtom::Slot { name, sort, hints },
            ForgeAtom::Literal { .. },
        )
        | (
            ForgeAtom::Literal { .. },
            ForgeAtom::Slot { name, sort, hints },
        ) => ForgeAtom::Slot {
            name: name.clone(),
            sort: sort.clone(),
            hints: hints.clone(),
        },
        (left, right) => ForgeAtom::Alt {
            options: vec![left.clone(), right.clone()],
        },
    }
}

/// 合并两个同长度正例的 token 正则。
fn merge_regex(
    left: &TokenRegex,
    right: &TokenRegex,
    extra_positive: &str,
) -> Result<TokenRegex, ForgeError> {
    if left.atoms.len() != right.atoms.len() {
        return Err(ForgeError::PositiveShapeMismatch {
            positive: extra_positive.to_string(),
        });
    }
    let atoms = left
        .atoms
        .iter()
        .zip(&right.atoms)
        .map(|(left, right)| merge_atom(left, right))
        .collect();
    Ok(TokenRegex::new(atoms))
}

fn alternative_value(sort: &str, current: &str, config: &NlpConfig) -> Option<String> {
    let candidates: Vec<String> = match sort {
        "人名" => {
            let mut names: Vec<String> = config
                .name_patterns
                .full_names
                .iter()
                .filter(|name| name.as_str() != current)
                .cloned()
                .collect();
            for fallback in ["李明", "王强", "张三", "李四"] {
                if fallback != current && !names.iter().any(|name| name == fallback) {
                    names.push(fallback.to_string());
                }
            }
            names
        }
        "城市" => ["北京", "上海", "广州", "深圳"]
            .iter()
            .filter(|city| **city != current)
            .map(|city| city.to_string())
            .collect(),
        "数量" => ["18", "20", "3", "5"]
            .iter()
            .filter(|number| **number != current)
            .map(|number| number.to_string())
            .collect(),
        "日期" => ["明天", "昨天", "今天"]
            .iter()
            .filter(|day| **day != current)
            .map(|day| day.to_string())
            .collect(),
        _ => Vec::new(),
    };
    candidates.into_iter().next()
}

fn generated_probes(
    pattern: &TokenRegex,
    fills: &BTreeMap<String, String>,
    config: &NlpConfig,
    positives: &[String],
    negatives: &[String],
) -> Vec<ForgeProbe> {
    let mut probes = Vec::new();
    for (slot, current) in fills {
        let Some(sort) = pattern.atoms.iter().find_map(|atom| match atom {
            ForgeAtom::Slot { name, sort, .. } if name == slot => Some(sort.as_str()),
            _ => None,
        }) else {
            continue;
        };
        let Some(alternative) = alternative_value(sort, current, config) else {
            continue;
        };
        let mut replacement = fills.clone();
        replacement.insert(slot.clone(), alternative);
        let sentence = pattern.render_with(&replacement);
        if sentence == current.as_str() || positives.iter().any(|example| example == &sentence) {
            continue;
        }
        let tokens = preprocess(&sentence, config);
        let expect_match = pattern.matches(&tokens);
        // 逆生成句必须命中；若因分词/排序不兼容没命中，就不谎报探针通过。
        if !expect_match {
            continue;
        }
        if negatives.iter().any(|negative| negative == &sentence) {
            continue;
        }
        probes.push(ForgeProbe {
            sentence,
            expect_match: true,
            passed: true,
            origin: ProbeOrigin::Generated,
        });
    }
    probes
}

// ---------------------------------------------------------------------------
// 会话台账
// ---------------------------------------------------------------------------

/// 会话级锻造台账：候选 → 激活 → 退役/遗忘，全程可审计。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternForge {
    dialect: String,
    turn: usize,
    max_patterns: usize,
    max_active: usize,
    patterns: BTreeMap<String, ForgedPattern>,
}

impl Default for PatternForge {
    fn default() -> Self {
        Self {
            dialect: FORGE_DIALECT.to_string(),
            turn: 0,
            max_patterns: DEFAULT_MAX_PATTERNS,
            max_active: DEFAULT_MAX_ACTIVE,
            patterns: BTreeMap::new(),
        }
    }
}

impl PatternForge {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_limits(max_patterns: usize, max_active: usize) -> Self {
        Self {
            max_patterns: max_patterns.max(1),
            max_active: max_active.max(1),
            ..Self::default()
        }
    }

    pub fn turn_count(&self) -> usize {
        self.turn
    }

    pub fn len(&self) -> usize {
        self.patterns.len()
    }

    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }

    pub fn active_count(&self) -> usize {
        self.patterns
            .values()
            .filter(|pattern| pattern.status == ForgeStatus::Active)
            .count()
    }

    pub fn patterns(&self) -> impl Iterator<Item = &ForgedPattern> {
        self.patterns.values()
    }

    pub fn get(&self, id: &str) -> Option<&ForgedPattern> {
        self.patterns.get(id)
    }

    fn resolve(&self, key: &str) -> Result<String, ForgeError> {
        let key = key.trim();
        if key.is_empty() {
            return Err(ForgeError::NotFound(key.to_string()));
        }
        if self.patterns.contains_key(key) {
            return Ok(key.to_string());
        }
        let matches: Vec<String> = self
            .patterns
            .values()
            .filter(|pattern| pattern.id.starts_with(key) || pattern.name == key)
            .map(|pattern| pattern.id.clone())
            .collect();
        match matches.len() {
            0 => Err(ForgeError::NotFound(key.to_string())),
            1 => Ok(matches[0].clone()),
            _ => Err(ForgeError::Ambiguous(key.to_string())),
        }
    }

    /// 从一个主正例、同句式补充正例与零到多个反例锻造候选；反例命中即拒绝。
    pub fn forge(
        &mut self,
        request: &ForgeRequest,
        config: &NlpConfig,
    ) -> Result<ForgeOutcome, ForgeError> {
        let name = request.name.trim();
        let query_type = request.query_type.trim();
        let positives: Vec<String> = request
            .positives()
            .into_iter()
            .map(str::trim)
            .filter(|positive| !positive.is_empty())
            .map(str::to_string)
            .collect();
        let Some(positive) = positives.first().map(String::as_str) else {
            return Err(ForgeError::EmptyPositive);
        };
        if name.is_empty() {
            return Err(ForgeError::EmptyName);
        }
        if query_type.is_empty() {
            return Err(ForgeError::EmptyRoute);
        }
        let positive_tokens = preprocess(positive, config);
        if positive_tokens.is_empty() {
            return Err(ForgeError::NoTokens);
        }
        let (mut regex, slot_fills) = learn_regex(&positive_tokens);
        for extra in positives.iter().skip(1) {
            let tokens = preprocess(extra, config);
            if tokens.is_empty() {
                return Err(ForgeError::NoTokens);
            }
            let (extra_regex, _) = learn_regex(&tokens);
            regex = merge_regex(&regex, &extra_regex, extra)?;
            if !regex.matches(&tokens) {
                return Err(ForgeError::PositiveNotCovered {
                    positive: extra.clone(),
                });
            }
        }
        let expansions = regex.expand()?;
        if !expansions_match(&expansions, &positive_tokens) {
            return Err(ForgeError::SelfReject);
        }

        let negatives: Vec<String> = request
            .negatives
            .iter()
            .map(|negative| negative.trim().to_string())
            .filter(|negative| !negative.is_empty())
            .collect();
        for negative in &negatives {
            if regex.matches(&preprocess(negative, config)) {
                return Err(ForgeError::NegativeAccepted {
                    negative: negative.clone(),
                    pattern: regex.render(),
                });
            }
        }

        let existing = self
            .patterns
            .values()
            .find(|pattern| {
                pattern.query_type == query_type
                    && pattern.regex == regex
                    && pattern.guard == request.guard
            })
            .map(|pattern| pattern.id.clone());

        self.turn += 1;
        let created = existing.is_none();
        if created && self.patterns.len() >= self.max_patterns {
            return Err(ForgeError::BudgetExhausted {
                max: self.max_patterns,
            });
        }
        let id = existing.unwrap_or_else(|| {
            let seed = format!("{}|{}|{:?}", query_type, regex.render(), request.guard);
            format!("forge_{}", fnv1a_hex(seed.as_bytes()))
        });

        let mut probes: Vec<ForgeProbe> = positives
            .iter()
            .map(|positive| ForgeProbe {
                sentence: positive.clone(),
                expect_match: true,
                passed: true,
                origin: ProbeOrigin::Positive,
            })
            .collect();
        for negative in &negatives {
            probes.push(ForgeProbe {
                sentence: negative.clone(),
                expect_match: false,
                passed: true,
                origin: ProbeOrigin::Negative,
            });
        }
        probes.extend(generated_probes(
            &regex,
            &slot_fills,
            config,
            &positives,
            &negatives,
        ));

        let pattern = self
            .patterns
            .entry(id.clone())
            .and_modify(|pattern| {
                pattern.name = name.to_string();
                pattern.priority = request.priority;
                for example in &negatives {
                    if !pattern.negatives.contains(example) {
                        pattern.negatives.push(example.clone());
                    }
                }
                for positive in &positives {
                    if !pattern.positives.contains(positive) {
                        pattern.positives.push(positive.clone());
                    }
                }
                for probe in &probes {
                    if !pattern
                        .probes
                        .iter()
                        .any(|existing| existing.sentence == probe.sentence)
                    {
                        pattern.probes.push(probe.clone());
                    }
                }
            })
            .or_insert_with(|| ForgedPattern {
                id: id.clone(),
                name: name.to_string(),
                query_type: query_type.to_string(),
                priority: request.priority,
                regex: regex.clone(),
                expansions: expansions.clone(),
                slot_fills: slot_fills.clone(),
                positives: positives.clone(),
                negatives: negatives.clone(),
                probes,
                status: ForgeStatus::Candidate,
                created_turn: self.turn,
                last_used_turn: 0,
                hits: 0,
                refutations: 0,
                guard: request.guard.clone(),
            })
            .clone();
        Ok(ForgeOutcome { pattern, created })
    }

    /// 显式确认候选，生成应并入模式库的条目。
    pub fn activate(&mut self, key: &str) -> Result<ForgeActivation, ForgeError> {
        let id = self.resolve(key)?;
        let pattern = self
            .patterns
            .get(&id)
            .ok_or_else(|| ForgeError::NotFound(key.to_string()))?;
        if pattern.status == ForgeStatus::Retired {
            return Err(ForgeError::AlreadyRetired(id));
        }
        if pattern.status != ForgeStatus::Active {
            if self.active_count() >= self.max_active {
                return Err(ForgeError::ActiveBudgetExhausted {
                    max: self.max_active,
                });
            }
            self.turn += 1;
            let pattern = self.patterns.get_mut(&id).expect("已解析的锻造 id");
            pattern.status = ForgeStatus::Active;
            pattern.last_used_turn = self.turn;
        }
        let pattern = self
            .patterns
            .get(&id)
            .expect("刚激活的锻造规则")
            .clone();
        let entries = pattern.materialize();
        Ok(ForgeActivation { pattern, entries })
    }

    /// 退役一条规则；调用方应把对应来源从模式库移除。
    pub fn retire(&mut self, key: &str) -> Result<ForgedPattern, ForgeError> {
        let id = self.resolve(key)?;
        let pattern = self
            .patterns
            .get_mut(&id)
            .ok_or_else(|| ForgeError::NotFound(key.to_string()))?;
        if pattern.status == ForgeStatus::Retired {
            return Err(ForgeError::AlreadyRetired(id));
        }
        self.turn += 1;
        pattern.status = ForgeStatus::Retired;
        pattern.last_used_turn = self.turn;
        Ok(pattern.clone())
    }

    /// 彻底遗忘一条非激活规则（激活规则必须先退役）。
    pub fn forget(&mut self, key: &str) -> Result<ForgedPattern, ForgeError> {
        let id = self.resolve(key)?;
        if self
            .patterns
            .get(&id)
            .map(|pattern| pattern.status == ForgeStatus::Active)
            .unwrap_or(false)
        {
            return Err(ForgeError::NotActive(format!(
                "{}（请先退役再遗忘）",
                id
            )));
        }
        self.turn += 1;
        self.patterns
            .remove(&id)
            .ok_or_else(|| ForgeError::NotFound(key.to_string()))
    }

    /// 记录一次命中（来源串为 `induced#forge#<id>` 或锻造 id）。
    pub fn record_hit(&mut self, source_or_id: &str) -> bool {
        let Some(id) = source_or_id
            .strip_prefix(FORGE_SOURCE_PREFIX)
            .or_else(|| self.patterns.contains_key(source_or_id).then_some(source_or_id))
        else {
            return false;
        };
        let turn = self.turn;
        if let Some(pattern) = self.patterns.get_mut(id) {
            pattern.hits += 1;
            pattern.last_used_turn = turn;
            return true;
        }
        false
    }

    /// 记录一次反驳；达到 [`MAX_REFUTATIONS`] 时自动退役。
    ///
    /// 返回 `Some(retired)` 表示找到了规则，`retired` 说明本次是否触发了
    /// 自动退役（调用方据此把条目移出模式库）。
    pub fn record_refutation(&mut self, source_or_id: &str) -> Option<bool> {
        let id = source_or_id
            .strip_prefix(FORGE_SOURCE_PREFIX)
            .or_else(|| self.patterns.contains_key(source_or_id).then_some(source_or_id))?;
        self.turn += 1;
        let turn = self.turn;
        let pattern = self.patterns.get_mut(id)?;
        pattern.refutations += 1;
        pattern.last_used_turn = turn;
        if pattern.refutations >= MAX_REFUTATIONS && pattern.status == ForgeStatus::Active {
            pattern.status = ForgeStatus::Retired;
            return Some(true);
        }
        Some(false)
    }

    /// 长期未用且从未命中的激活规则自动退役。
    pub fn sweep_stale(&mut self, stale_turns: usize) -> Vec<ForgedPattern> {
        let mut retired = Vec::new();
        let turn = self.turn;
        for pattern in self.patterns.values_mut() {
            if pattern.status == ForgeStatus::Active
                && pattern.hits == 0
                && turn.saturating_sub(pattern.last_used_turn) >= stale_turns
            {
                pattern.status = ForgeStatus::Retired;
                retired.push(pattern.clone());
            }
        }
        retired
    }

    /// 所有激活规则的物化条目（启动恢复用）。
    pub fn active_entries(&self) -> Vec<PatternEntry> {
        self.patterns
            .values()
            .filter(|pattern| pattern.status == ForgeStatus::Active)
            .flat_map(ForgedPattern::materialize)
            .collect()
    }

    pub fn report(&self) -> String {
        let mut lines = vec![format!(
            "锻造台账（{}）：{} 条（候选 {} / 激活 {} / 退役 {}），轮次 {}",
            FORGE_DIALECT.trim_start_matches("knowprolog-forge/"),
            self.patterns.len(),
            self.patterns
                .values()
                .filter(|pattern| pattern.status == ForgeStatus::Candidate)
                .count(),
            self.active_count(),
            self.patterns
                .values()
                .filter(|pattern| pattern.status == ForgeStatus::Retired)
                .count(),
            self.turn
        )];
        if self.patterns.is_empty() {
            lines.push("（空；用「锻造模式：名 | 正例 | 反例 | 路由」从例子学一条）".to_string());
        } else {
            for pattern in self.patterns.values() {
                lines.push(pattern.render());
            }
        }
        lines.join("\n")
    }

    pub fn save_to_path(&self, path: &str) -> std::io::Result<()> {
        if let Some(parent) = Path::new(path).parent() {
            fs::create_dir_all(parent)?;
        }
        let text = serde_json::to_string_pretty(self)
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::Other, error))?;
        fs::write(path, format!("{}\n", text))
    }

    pub fn load_from_path(path: &str) -> Option<Self> {
        let text = fs::read_to_string(path).ok()?;
        let forge: Self = serde_json::from_str(&text).ok()?;
        (forge.dialect == FORGE_DIALECT).then_some(forge)
    }
}

// ---------------------------------------------------------------------------
// 指令解析
// ---------------------------------------------------------------------------

/// 解析交互指令：
/// `<名> | <正例(多个用 ; 分隔)> | <反例(可空，多个用 ; 分隔)> | <路由> | <优先级(可选)> | <守卫(可选)>`
pub fn parse_forge_spec(spec: &str) -> Result<ForgeRequest, ForgeError> {
    let parts: Vec<&str> = spec.split('|').map(str::trim).collect();
    if parts.len() < 4 {
        return Err(ForgeError::BadSpec(
            "至少需要「名 | 正例 | 反例 | 路由」四个字段".to_string(),
        ));
    }
    let name = parts[0];
    let positives = parts[1]
        .split([';', '；'])
        .map(str::trim)
        .filter(|positive| !positive.is_empty())
        .map(str::to_string)
        .collect::<Vec<_>>();
    if name.is_empty() {
        return Err(ForgeError::EmptyName);
    }
    let Some(positive) = positives.first().cloned() else {
        return Err(ForgeError::EmptyPositive);
    };
    let extra_positives = positives.into_iter().skip(1).collect();
    let negatives = parts[2]
        .split([';', '；'])
        .map(str::trim)
        .filter(|negative| !negative.is_empty())
        .map(str::to_string)
        .collect::<Vec<_>>();
    let query_type = parts[3];
    if query_type.is_empty() {
        return Err(ForgeError::EmptyRoute);
    }
    let priority = match parts.get(4).filter(|part| !part.is_empty()) {
        Some(part) => part
            .parse::<i32>()
            .map_err(|_| ForgeError::BadSpec(format!("优先级不是整数：{}", part)))?,
        None => 10,
    };
    let guard = match parts.get(5).filter(|part| !part.is_empty()) {
        Some(part) => Some(
            crate::pattern_ir::parse_context_guard(part)
                .map_err(|error| ForgeError::BadSpec(error))?,
        ),
        None => None,
    };
    Ok(ForgeRequest {
        name: name.to_string(),
        query_type: query_type.to_string(),
        priority,
        positive,
        extra_positives,
        negatives,
        guard,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NamePatterns;
    use std::collections::HashMap;

    fn nlp_config() -> NlpConfig {
        NlpConfig {
            import_keywords: vec!["录入".to_string()],
            pos_dict: HashMap::new(),
            name_patterns: NamePatterns {
                prefixes: vec!["小".to_string(), "老".to_string()],
                suffixes: vec!["总".to_string(), "哥".to_string()],
                full_names: vec![
                    "李明".to_string(),
                    "王强".to_string(),
                    "张三".to_string(),
                    "李四".to_string(),
                ],
                profession_suffix: Vec::new(),
            },
            enable_nlu: Some(false),
            nlu_confidence_threshold: None,
            enable_local_semantic: Some(false),
            disable_ownthink_api: Some(true),
            semantic_similarity_threshold: None,
            max_entity_candidates: None,
            enable_advanced_entity_extraction: None,
            enable_relation_extraction: None,
            context_window_size: None,
        }
    }

    #[test]
    fn single_positive_generalises_entities_and_rejects_negative() {
        let config = nlp_config();
        let request = ForgeRequest::new("女儿问法", "姚明的女儿是谁", "daughter_query")
            .with_negative("姚明今年几岁");
        let mut forge = PatternForge::new();
        let outcome = forge.forge(&request, &config).expect("应当学会");
        assert!(outcome.created);
        assert_eq!(outcome.pattern.status, ForgeStatus::Candidate);
        assert!(outcome.pattern.expansions.len() <= MAX_EXPANSIONS);
        let regex = outcome.pattern.regex.render();
        assert!(regex.contains("{人物:人名}"), "{}", regex);
        assert!(regex.contains("女儿"), "{}", regex);
        assert!(!regex.contains("姚明"), "人名应当变槽：{}", regex);
        assert!(outcome
            .pattern
            .probes
            .iter()
            .any(|probe| probe.origin == ProbeOrigin::Negative && probe.passed));
        assert!(outcome
            .pattern
            .probes
            .iter()
            .any(|probe| probe.origin == ProbeOrigin::Generated && probe.passed));
    }

    #[test]
    fn multiple_positives_learn_surface_choices() {
        let config = nlp_config();
        let request = ForgeRequest::new("女儿问法", "姚明的女儿是谁", "daughter_query")
            .with_positive("姚明的闺女是谁")
            .with_negative("姚明的妻子是谁");
        let mut forge = PatternForge::new();
        let outcome = forge.forge(&request, &config).expect("应当学会两种写法");
        assert_eq!(outcome.pattern.positives.len(), 2);
        assert_eq!(outcome.pattern.expansions.len(), 2);
        let regex = outcome.pattern.regex.render();
        assert!(regex.contains("女儿"), "{}", regex);
        assert!(regex.contains("闺女"), "{}", regex);
        let activation = forge.activate(&outcome.pattern.id).expect("激活");
        let mut library = PatternLibrary::new();
        for entry in &activation.entries {
            library.add(entry.clone());
        }
        for sentence in ["王强的女儿是谁", "王强的闺女是谁"] {
            let routed = library
                .route(&preprocess(sentence, &config))
                .expect("两种写法都应当路由");
            assert_eq!(routed.0, "daughter_query", "{}", sentence);
            assert!(routed.1.contains(&"王强".to_string()), "{}", sentence);
        }
        assert!(library
            .route(&preprocess("王强的妻子是谁", &config))
            .is_none());
    }

    #[test]
    fn mixed_length_positives_are_refused_instead_of_silently_dropped() {
        let config = nlp_config();
        let mut forge = PatternForge::new();
        let error = forge
            .forge(
                &ForgeRequest::new("女儿问法", "姚明的女儿是谁", "daughter_query")
                    .with_positive("姚明的闺女现在是谁"),
                &config,
            )
            .expect_err("不同 token 数应当明确拒绝");
        assert!(matches!(error, ForgeError::PositiveShapeMismatch { .. }));
    }

    #[test]
    fn learned_pattern_routes_new_entity_after_activation() {
        let config = nlp_config();
        let mut forge = PatternForge::new();
        let outcome = forge
            .forge(
                &ForgeRequest::new("女儿问法", "姚明的女儿是谁", "daughter_query")
                    .with_negative("姚明今年几岁"),
                &config,
            )
            .expect("应当学会");
        let activation = forge.activate(&outcome.pattern.id).expect("应当激活");
        assert_eq!(activation.entries.len(), outcome.pattern.expansions.len());
        let mut library = PatternLibrary::new();
        for entry in &activation.entries {
            library.add(entry.clone());
        }
        let tokens = preprocess("王强的女儿是谁", &config);
        let routed = library.route(&tokens).expect("新实体应当路由");
        assert_eq!(routed.0, "daughter_query");
        assert!(routed.1.contains(&"王强".to_string()));
        assert!(forge.record_hit(&outcome.pattern.source()));
        assert_eq!(
            forge
                .get(&outcome.pattern.id)
                .map(|pattern| pattern.hits)
                .unwrap_or_default(),
            1
        );
    }

    #[test]
    fn negative_that_shares_the_skeleton_refuses_learning() {
        let config = nlp_config();
        let mut forge = PatternForge::new();
        let error = forge
            .forge(
                &ForgeRequest::new("女儿问法", "姚明的女儿是谁", "daughter_query")
                    .with_negative("李明的女儿是谁"),
                &config,
            )
            .expect_err("反例必须阻止泛化");
        assert!(matches!(error, ForgeError::NegativeAccepted { .. }));
    }

    #[test]
    fn activation_budget_is_enforced_and_retire_releases_it() {
        let config = nlp_config();
        let mut forge = PatternForge::with_limits(8, 1);
        let first = forge
            .forge(
                &ForgeRequest::new("女儿问法", "姚明的女儿是谁", "daughter_query"),
                &config,
            )
            .expect("第一条");
        forge.activate(&first.pattern.id).expect("第一条激活");
        let second = forge
            .forge(
                &ForgeRequest::new("年龄问法", "姚明今年几岁", "age_query"),
                &config,
            )
            .expect("第二条");
        assert!(matches!(
            forge.activate(&second.pattern.id),
            Err(ForgeError::ActiveBudgetExhausted { .. })
        ));
        forge.retire(&first.pattern.id).expect("退役第一条");
        forge.activate(&second.pattern.id).expect("释放预算后可激活");
    }

    #[test]
    fn refutation_threshold_auto_retires() {
        let config = nlp_config();
        let mut forge = PatternForge::new();
        let outcome = forge
            .forge(
                &ForgeRequest::new("女儿问法", "姚明的女儿是谁", "daughter_query"),
                &config,
            )
            .expect("应当学会");
        forge.activate(&outcome.pattern.id).expect("激活");
        for _ in 0..MAX_REFUTATIONS - 1 {
            assert_eq!(forge.record_refutation(&outcome.pattern.id), Some(false));
        }
        assert_eq!(forge.record_refutation(&outcome.pattern.id), Some(true));
        assert_eq!(
            forge
                .get(&outcome.pattern.id)
                .map(|pattern| pattern.status),
            Some(ForgeStatus::Retired)
        );
    }

    #[test]
    fn spec_parser_reads_negatives_priority_and_guard() {
        let request = parse_forge_spec(
            "女儿问法 | 姚明的女儿是谁 | 姚明今年几岁;姚明的儿子是谁 | daughter_query | 30 | 焦点",
        )
        .expect("应当解析");
        assert_eq!(request.name, "女儿问法");
        assert_eq!(request.query_type, "daughter_query");
        assert_eq!(request.priority, 30);
        assert_eq!(request.negatives.len(), 2);
        assert_eq!(request.guard, Some(crate::pattern_ir::ContextGuard::focus()));
        assert!(parse_forge_spec("只有名字").is_err());
    }

    #[test]
    fn ledger_round_trips_through_json() {
        let config = nlp_config();
        let directory = tempfile::tempdir().expect("临时目录");
        let path = directory.path().join("forge.json");
        let path = path.to_string_lossy().to_string();
        let mut forge = PatternForge::new();
        let outcome = forge
            .forge(
                &ForgeRequest::new("女儿问法", "姚明的女儿是谁", "daughter_query"),
                &config,
            )
            .expect("应当学会");
        forge.activate(&outcome.pattern.id).expect("激活");
        forge.save_to_path(&path).expect("保存");
        let restored = PatternForge::load_from_path(&path).expect("恢复");
        assert_eq!(restored.len(), 1);
        assert_eq!(restored.active_count(), 1);
        assert_eq!(
            restored
                .get(&outcome.pattern.id)
                .map(|pattern| pattern.regex.render()),
            Some(outcome.pattern.regex.render())
        );
        assert!(!restored.active_entries().is_empty());
    }
}

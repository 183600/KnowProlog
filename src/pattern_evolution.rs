//! 模式演化事务与一致性闸门（Pattern Evolution Transaction & Consistency Gate）。
//!
//! 现有 [`crate::pattern_ir`] 的 lint 只看当前快照，[`crate::config_contract`]
//! 只看配置层，[`crate::pattern_forge`] 只看单条规则。本模块把「模式库」当成
//! 可版本化的模型：每次教学、诱导、锻造、确认、遗忘或配置重载都是一次**提案**，
//! 提案必须相对上一代做四类校验，通过后才成为新一代：
//!
//! 1. **结构**：IR/名称/来源非空，新增重复、冲突、遮蔽相对上一代是否变多；
//! 2. **引用/能力**：路由到的 `query_type` 是否仍可执行，词面是否落在同义词
//!    规范形上；
//! 3. **行为**：用金标构念问句与已确认经验回放前代/候选，抓住「已确认问句改道」；
//! 4. **账本**：已确认经验指向的模式是否变成幽灵条目，新增条目是否自证。
//!
//! 判定分四档：提交（Commit）、观察（Observe，先进 probation）、隔离（Isolate，
//! 行为回归）与拒绝（Reject，结构/引用/账本硬错误）。回滚不逆转时间，而是把旧代
//! 条目作为新代提交，因此每一代都能解释「当时相对父代改了什么」。
//!
//! 设计边界：本模块只做确定性符号比较，不训练权重；回放语料来自配置声称可答的
//! 构念问句与经验账本，不宣称开放式召回。来自被提案规则的 seed 经验会单独计数，
//! 不作为泛化证据。

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::config_contract;
use crate::context_image::fnv1a_hex;
use crate::nlp::{normalize_words, preprocess, TaggedWord};
use crate::pattern_ir::{
    self, lint, EpisodeOutcome, LintFinding, LintKind, PatternContext, PatternEntry,
    PatternEpisode, PatternLibrary, PatternMatch,
};
use crate::recall_audit::{self, CaseVariant};
use crate::SystemConfig;

/// 演化台账的序列化方言；版本不符时拒绝恢复。
pub const EVOLUTION_DIALECT: &str = "knowprolog-pattern-evolution/1";
/// 默认保留多少代模式快照。
pub const DEFAULT_MAX_GENERATIONS: usize = 64;
/// 默认保留多少份演化回执。
pub const DEFAULT_MAX_RECEIPTS: usize = 32;

// ---------------------------------------------------------------------------
// 三时钟与代
// ---------------------------------------------------------------------------

/// 一次演化的三时钟 + 经验水位。
///
/// 四个值都由调用方提供：本模块不读挂钟，也不把「第几轮对话」误当成
/// 「知识库第几次写入」。同样的时钟输入必须得到同样的回执。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct EvolutionClock {
    /// 配置层快照序号（热加载一次推进一次）。
    #[serde(default)]
    pub config_epoch: u64,
    /// 知识库修订号（导入/撤销推进）。
    #[serde(default)]
    pub kb_generation: u64,
    /// 模式代序号（本台账内单调递增）。
    #[serde(default)]
    pub pattern_generation: u64,
    /// 经验账本修订号（[`crate::pattern_ir::PatternEpisodeLog::revision`]）。
    #[serde(default)]
    pub episode_revision: u64,
}

impl EvolutionClock {
    pub fn new(
        config_epoch: u64,
        kb_generation: u64,
        pattern_generation: u64,
        episode_revision: u64,
    ) -> Self {
        Self {
            config_epoch,
            kb_generation,
            pattern_generation,
            episode_revision,
        }
    }

    pub fn render(&self) -> String {
        format!(
            "配置代 {} / 知识代 {} / 模式代 {} / 经验修订 {}",
            self.config_epoch, self.kb_generation, self.pattern_generation, self.episode_revision
        )
    }
}

/// 一代模式库的生命周期状态。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenerationStatus {
    /// 正式生效。
    #[default]
    Active,
    /// 观察期：已记录但需要后续经验确认。
    Probation,
    /// 被隔离：保留证据，不参与路由。
    Quarantined,
    /// 已退役：保留历史，不再使用。
    Retired,
}

impl GenerationStatus {
    pub fn label(self) -> &'static str {
        match self {
            GenerationStatus::Active => "生效",
            GenerationStatus::Probation => "观察",
            GenerationStatus::Quarantined => "隔离",
            GenerationStatus::Retired => "退役",
        }
    }
}

/// 一代不可变模式快照：条目顺序保留评估序，指纹与顺序无关。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PatternGeneration {
    pub generation: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<u64>,
    /// 条目集指纹（按条目 id 升序折叠）。
    pub fingerprint: String,
    pub clock: EvolutionClock,
    #[serde(default)]
    pub status: GenerationStatus,
    pub note: String,
    pub entries: Vec<PatternEntry>,
}

impl PatternGeneration {
    /// 从模式库拍一代快照。条目保持原顺序，避免同优先级路由因排序而漂移。
    pub fn snapshot(
        generation: u64,
        parent: Option<u64>,
        library: &PatternLibrary,
        clock: EvolutionClock,
        status: GenerationStatus,
        note: impl Into<String>,
    ) -> Self {
        let entries = library.entries().to_vec();
        let fingerprint = pattern_ir::library_fingerprint(&entries);
        Self {
            generation,
            parent,
            fingerprint,
            clock,
            status,
            note: note.into(),
            entries,
        }
    }

    /// 还原为可路由模式库（保留条目顺序）。
    pub fn library(&self) -> PatternLibrary {
        library_from_entries(&self.entries)
    }

    pub fn ids(&self) -> BTreeSet<String> {
        self.entries.iter().map(PatternEntry::id).collect()
    }

    pub fn render(&self) -> String {
        format!(
            "第{}代（{}，父代 {}，{} 条，指纹 {}）：{}",
            self.generation,
            self.status.label(),
            self.parent
                .map(|parent| parent.to_string())
                .unwrap_or_else(|| "无".to_string()),
            self.entries.len(),
            self.fingerprint,
            self.note
        )
    }
}

fn library_from_entries(entries: &[PatternEntry]) -> PatternLibrary {
    let mut library = PatternLibrary::new();
    for entry in entries {
        library.add(entry.clone());
    }
    library
}

// ---------------------------------------------------------------------------
// 提案
// ---------------------------------------------------------------------------

/// 提案来源类别。类别只用于解释与审计，不改变闸门算法。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProposalKind {
    Teach,
    Induce,
    Forge,
    ConfirmCase,
    Forget,
    ConfigReload,
    Migration,
    Rollback,
}

impl ProposalKind {
    pub fn label(self) -> &'static str {
        match self {
            ProposalKind::Teach => "教学",
            ProposalKind::Induce => "诱导",
            ProposalKind::Forge => "锻造",
            ProposalKind::ConfirmCase => "案例确认",
            ProposalKind::Forget => "遗忘",
            ProposalKind::ConfigReload => "配置重载",
            ProposalKind::Migration => "迁移",
            ProposalKind::Rollback => "回滚",
        }
    }
}

/// 一条提案编辑。所有编辑按顺序应用，后一条可覆盖前一条的结果。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum ProposalEdit {
    Add { entry: PatternEntry },
    RemoveId { id: String },
    RemoveSource { prefix: String },
}

impl ProposalEdit {
    pub fn add(entry: PatternEntry) -> Self {
        ProposalEdit::Add { entry }
    }

    pub fn remove_id(id: impl Into<String>) -> Self {
        ProposalEdit::RemoveId { id: id.into() }
    }

    pub fn remove_source(prefix: impl Into<String>) -> Self {
        ProposalEdit::RemoveSource {
            prefix: prefix.into(),
        }
    }
}

/// 作者显式留出的回放问句（不来自被提案规则自身）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HeldOutCase {
    pub input: String,
    pub expected_query_type: String,
    #[serde(default)]
    pub note: String,
}

/// 一次模式演化提案：相对某一代做增删。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvolutionProposal {
    pub id: String,
    pub kind: ProposalKind,
    pub base_generation: u64,
    #[serde(default)]
    pub base_fingerprint: String,
    #[serde(default)]
    pub note: String,
    #[serde(default)]
    pub edits: Vec<ProposalEdit>,
    #[serde(default)]
    pub held_out: Vec<HeldOutCase>,
}

impl EvolutionProposal {
    pub fn new(
        id: impl Into<String>,
        kind: ProposalKind,
        base: &PatternGeneration,
        note: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            kind,
            base_generation: base.generation,
            base_fingerprint: base.fingerprint.clone(),
            note: note.into(),
            edits: Vec::new(),
            held_out: Vec::new(),
        }
    }

    pub fn with_edit(mut self, edit: ProposalEdit) -> Self {
        self.edits.push(edit);
        self
    }

    pub fn with_held_out(mut self, case: HeldOutCase) -> Self {
        self.held_out.push(case);
        self
    }

    /// 应用提案到指定代，得到候选模式库。基础代或指纹不符时拒绝。
    pub fn apply(&self, base: &PatternGeneration) -> Result<PatternLibrary, EvolutionError> {
        if self.base_generation != base.generation {
            return Err(EvolutionError::BaseGenerationMissing {
                requested: self.base_generation,
                current: base.generation,
            });
        }
        if !self.base_fingerprint.is_empty() && self.base_fingerprint != base.fingerprint {
            return Err(EvolutionError::FingerprintMismatch {
                expected: self.base_fingerprint.clone(),
                actual: base.fingerprint.clone(),
            });
        }
        let mut library = base.library();
        for edit in &self.edits {
            match edit {
                ProposalEdit::Add { entry } => {
                    validate_entry(entry)?;
                    library.add(entry.clone());
                }
                ProposalEdit::RemoveId { id } => {
                    let key = id.trim();
                    if key.is_empty() {
                        return Err(EvolutionError::InvalidEdit("删除 id 不能为空".to_string()));
                    }
                    let matches: Vec<String> = library
                        .entries()
                        .iter()
                        .filter(|entry| entry.id() == key || entry.id().starts_with(key))
                        .map(PatternEntry::id)
                        .collect();
                    if matches.is_empty() {
                        return Err(EvolutionError::EditTargetMissing(key.to_string()));
                    }
                    if matches.len() > 1 {
                        return Err(EvolutionError::AmbiguousEdit {
                            key: key.to_string(),
                            matches: matches.len(),
                        });
                    }
                    let target = &matches[0];
                    let kept: Vec<PatternEntry> = library
                        .entries()
                        .iter()
                        .filter(|entry| entry.id() != *target)
                        .cloned()
                        .collect();
                    library = library_from_entries(&kept);
                }
                ProposalEdit::RemoveSource { prefix } => {
                    let prefix = prefix.trim();
                    if prefix.is_empty() {
                        return Err(EvolutionError::InvalidEdit(
                            "删除来源前缀不能为空".to_string(),
                        ));
                    }
                    let removed = library.remove_source_prefix(prefix);
                    if removed == 0 {
                        return Err(EvolutionError::EditTargetMissing(prefix.to_string()));
                    }
                }
            }
        }
        Ok(library)
    }

    pub fn added_entries(&self) -> Vec<&PatternEntry> {
        self.edits
            .iter()
            .filter_map(|edit| match edit {
                ProposalEdit::Add { entry } => Some(entry),
                _ => None,
            })
            .collect()
    }

    fn added_ids(&self) -> BTreeSet<String> {
        self.added_entries()
            .into_iter()
            .map(PatternEntry::id)
            .collect()
    }

    fn added_sources(&self) -> BTreeSet<String> {
        self.added_entries()
            .into_iter()
            .map(|entry| entry.source.clone())
            .collect()
    }

    fn explicitly_removed_ids(&self) -> BTreeSet<String> {
        self.edits
            .iter()
            .filter_map(|edit| match edit {
                ProposalEdit::RemoveId { id } => Some(id.trim().to_string()),
                _ => None,
            })
            .collect()
    }

    fn explicitly_removed_prefixes(&self) -> Vec<String> {
        self.edits
            .iter()
            .filter_map(|edit| match edit {
                ProposalEdit::RemoveSource { prefix } => Some(prefix.trim().to_string()),
                _ => None,
            })
            .collect()
    }

    fn removes_entry(&self, entry: &PatternEntry) -> bool {
        let id = entry.id();
        self.explicitly_removed_ids()
            .iter()
            .any(|key| id == *key || id.starts_with(key))
            || self
                .explicitly_removed_prefixes()
                .iter()
                .any(|prefix| entry.source.starts_with(prefix))
    }
}

fn validate_entry(entry: &PatternEntry) -> Result<(), EvolutionError> {
    if entry.name.trim().is_empty() {
        return Err(EvolutionError::InvalidEntry("模式名称不能为空".to_string()));
    }
    if entry.source.trim().is_empty() {
        return Err(EvolutionError::InvalidEntry("模式来源不能为空".to_string()));
    }
    if entry.query_type.trim().is_empty() {
        return Err(EvolutionError::InvalidEntry(
            "路由类型（query_type）不能为空".to_string(),
        ));
    }
    if entry.elems.is_empty() {
        return Err(EvolutionError::InvalidEntry("模式至少要有一个要素".to_string()));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// 回放语料
// ---------------------------------------------------------------------------

/// 回放用例标签。强标签来自用户确认/否定，弱标签来自观察与构念问句。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayLabel {
    /// 用户确认过的成功经验（最强）。
    Confirmed,
    /// 仅观察到、未明确反馈的成功经验。
    Observed,
    /// 用户否定过的经验（负例）。
    Negative,
    /// 配置声称可答的构念问句。
    Constructed,
    /// 作者显式留出的问句。
    HeldOut,
}

impl ReplayLabel {
    pub fn label(self) -> &'static str {
        match self {
            ReplayLabel::Confirmed => "确认经验",
            ReplayLabel::Observed => "观察经验",
            ReplayLabel::Negative => "负例",
            ReplayLabel::Constructed => "构念问句",
            ReplayLabel::HeldOut => "留出问句",
        }
    }
}

/// 回放用例来源。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplaySource {
    Episode(String),
    Constructed(CaseVariant),
    HeldOut,
}

/// 一条回放用例：输入、期望、标签与发生时的对话上下文。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayCase {
    pub id: String,
    pub input: String,
    pub expected_query_type: Option<String>,
    pub expected_winner_id: Option<String>,
    pub label: ReplayLabel,
    /// 该经验是否由本次提案新增的规则自身产生（自证，不算泛化证据）。
    pub seed: bool,
    pub source: ReplaySource,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens: Option<Vec<(String, String)>>,
    pub context: PatternContext,
}

impl ReplayCase {
    fn new(
        input: impl Into<String>,
        expected_query_type: Option<String>,
        expected_winner_id: Option<String>,
        label: ReplayLabel,
        seed: bool,
        source: ReplaySource,
        tokens: Option<Vec<(String, String)>>,
        context: PatternContext,
    ) -> Self {
        let input = input.into();
        let id = fnv1a_hex(
            format!(
                "{}\u{0}{}\u{0}{}\u{0}{}\u{0}{}",
                input,
                expected_query_type.as_deref().unwrap_or(""),
                expected_winner_id.as_deref().unwrap_or(""),
                label.label(),
                seed
            )
            .as_bytes(),
        );
        Self {
            id,
            input,
            expected_query_type,
            expected_winner_id,
            label,
            seed,
            source,
            tokens,
            context,
        }
    }
}

/// 构造回放语料：前代构念问句 + 经验账本 + 作者留出问句。
///
/// 构念问句来自前代模式库，因此不会把候选新增规则自己的生成句当成金标。
/// 经验按同一输入/上下文/期望去重；`seed` 判定基于提案新增条目的 id 或来源。
pub fn build_replay_corpus(
    base: &PatternGeneration,
    candidate: &PatternLibrary,
    episodes: &[PatternEpisode],
    proposal: &EvolutionProposal,
    config: &SystemConfig,
) -> Vec<ReplayCase> {
    let synonyms = synonyms(config);
    let (constructed, _) = recall_audit::construct_cases(&base.library(), &synonyms);
    let added_ids = proposal.added_ids();
    let added_sources = proposal.added_sources();
    let base_sources: BTreeMap<String, String> = base
        .entries
        .iter()
        .map(|entry| (entry.id(), entry.source.clone()))
        .collect();
    let candidate_sources: BTreeMap<String, String> = candidate
        .entries()
        .iter()
        .map(|entry| (entry.id(), entry.source.clone()))
        .collect();

    let mut cases: BTreeMap<String, ReplayCase> = BTreeMap::new();

    for case in constructed {
        let replay = ReplayCase::new(
            case.question,
            Some(case.expected_query_type),
            None,
            ReplayLabel::Constructed,
            false,
            ReplaySource::Constructed(case.variant),
            None,
            PatternContext::default(),
        );
        cases.insert(replay.id.clone(), replay);
    }

    for episode in episodes {
        let label = if episode.rejections > episode.acceptances {
            ReplayLabel::Negative
        } else if episode.acceptances > episode.rejections {
            ReplayLabel::Confirmed
        } else {
            ReplayLabel::Observed
        };
        let seed = episode
            .winner_id
            .as_ref()
            .map(|winner| {
                added_ids.contains(winner)
                    || base_sources
                        .get(winner)
                        .map(|source| added_sources.iter().any(|added| source.starts_with(added)))
                        .unwrap_or(false)
                    || candidate_sources
                        .get(winner)
                        .map(|source| added_sources.iter().any(|added| source.starts_with(added)))
                        .unwrap_or(false)
            })
            .unwrap_or(false);
        let expected = match episode.outcome {
            EpisodeOutcome::Matched => episode.query_type.clone(),
            EpisodeOutcome::Unmatched => None,
        };
        let replay = ReplayCase::new(
            episode.input.clone(),
            expected,
            episode.winner_id.clone(),
            label,
            seed,
            ReplaySource::Episode(episode.id()),
            Some(episode.tokens.clone()),
            PatternContext {
                focus: episode.focus.clone(),
                focus_type: episode.focus_type.clone(),
                last_move: episode.last_move.clone(),
                pending_clarify: episode.pending_clarify.unwrap_or(false),
                turn: episode.last_turn,
            },
        );
        cases.insert(replay.id.clone(), replay);
    }

    for held_out in &proposal.held_out {
        let replay = ReplayCase::new(
            held_out.input.clone(),
            Some(held_out.expected_query_type.clone()),
            None,
            ReplayLabel::HeldOut,
            false,
            ReplaySource::HeldOut,
            None,
            PatternContext::default(),
        );
        cases.insert(replay.id.clone(), replay);
    }

    cases.into_values().collect()
}

fn synonyms(config: &SystemConfig) -> HashMap<String, String> {
    config
        .synonyms
        .as_ref()
        .map(|synonyms| synonyms.mappings.clone())
        .unwrap_or_default()
}

fn tokens_for(
    case: &ReplayCase,
    config: &SystemConfig,
    synonyms: &HashMap<String, String>,
) -> Vec<TaggedWord> {
    match &case.tokens {
        Some(pairs) => pairs
            .iter()
            .map(|(text, pos)| TaggedWord {
                text: text.clone(),
                pos: pos.clone(),
            })
            .collect(),
        None => {
            let tagged = preprocess(&case.input, &config.nlp);
            normalize_words(tagged, synonyms)
        }
    }
}

fn route(
    library: &PatternLibrary,
    tokens: &[TaggedWord],
    context: &PatternContext,
) -> Option<PatternMatch> {
    library
        .match_route_with_context(tokens, context)
        .or_else(|| library.match_fallback_with_context(tokens, context))
}

// ---------------------------------------------------------------------------
// 闸门
// ---------------------------------------------------------------------------

/// 四类校验。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateKind {
    Structural,
    Reference,
    Behavior,
    Ledger,
}

impl GateKind {
    pub fn label(self) -> &'static str {
        match self {
            GateKind::Structural => "结构",
            GateKind::Reference => "引用/能力",
            GateKind::Behavior => "行为",
            GateKind::Ledger => "账本",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FindingSeverity {
    Error,
    Warning,
    Info,
}

impl FindingSeverity {
    pub fn label(self) -> &'static str {
        match self {
            FindingSeverity::Error => "ERROR",
            FindingSeverity::Warning => "WARN ",
            FindingSeverity::Info => "INFO ",
        }
    }
}

/// 一条可机器匹配、可人读的闸门发现。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvolutionFinding {
    pub gate: GateKind,
    pub severity: FindingSeverity,
    pub code: String,
    pub subject: String,
    pub message: String,
}

impl EvolutionFinding {
    fn new(
        gate: GateKind,
        severity: FindingSeverity,
        code: impl Into<String>,
        subject: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            gate,
            severity,
            code: code.into(),
            subject: subject.into(),
            message: message.into(),
        }
    }

    pub fn render(&self) -> String {
        format!(
            "{} [{}] {}：{}",
            self.severity.label(),
            self.code,
            self.subject,
            self.message
        )
    }
}

/// 判定：提交、观察、隔离或拒绝。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvolutionVerdict {
    Commit,
    Observe,
    Isolate,
    Reject,
}

impl EvolutionVerdict {
    pub fn label(self) -> &'static str {
        match self {
            EvolutionVerdict::Commit => "提交",
            EvolutionVerdict::Observe => "观察",
            EvolutionVerdict::Isolate => "隔离",
            EvolutionVerdict::Reject => "拒绝",
        }
    }

    /// 是否允许写入下一代（观察期也写入，但状态为 Probation）。
    pub fn may_commit(self) -> bool {
        matches!(self, EvolutionVerdict::Commit | EvolutionVerdict::Observe)
    }
}

/// 单条回放的差异判定。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayVerdict {
    Stable,
    StableMiss,
    Gain,
    Regression,
    Reroute,
    Changed,
    NegativeViolation,
}

impl ReplayVerdict {
    pub fn label(self) -> &'static str {
        match self {
            ReplayVerdict::Stable => "稳定",
            ReplayVerdict::StableMiss => "仍未命中",
            ReplayVerdict::Gain => "新增命中",
            ReplayVerdict::Regression => "回归",
            ReplayVerdict::Reroute => "改道",
            ReplayVerdict::Changed => "变化",
            ReplayVerdict::NegativeViolation => "负例违规",
        }
    }
}

/// 一条回放的机器可读差异。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayOutcome {
    pub case_id: String,
    pub input: String,
    pub label: ReplayLabel,
    pub seed: bool,
    pub expected_query_type: Option<String>,
    pub base_query_type: Option<String>,
    pub candidate_query_type: Option<String>,
    pub base_winner_id: Option<String>,
    pub candidate_winner_id: Option<String>,
    pub verdict: ReplayVerdict,
}

/// 回放汇总。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplaySummary {
    pub cases: usize,
    pub episodes: usize,
    pub constructed: usize,
    pub held_out: usize,
    pub seed_cases: usize,
    pub stable: usize,
    pub stable_misses: usize,
    pub gains: usize,
    pub held_out_gains: usize,
    pub seed_gains: usize,
    pub regressions: usize,
    pub held_out_regressions: usize,
    pub seed_regressions: usize,
    pub reroutes: usize,
    pub changed: usize,
    pub negative_violations: usize,
}

impl ReplaySummary {
    pub fn render(&self) -> String {
        format!(
            "回放 {} 例（经验 {} + 构念 {} + 留出 {}；其中 seed {}）：稳定 {}，仍未命中 {}，新增命中 {}（留出 {} / seed {}），回归 {}（留出 {} / seed {}），改道 {}，变化 {}，负例违规 {}",
            self.cases,
            self.episodes,
            self.constructed,
            self.held_out,
            self.seed_cases,
            self.stable,
            self.stable_misses,
            self.gains,
            self.held_out_gains,
            self.seed_gains,
            self.regressions,
            self.held_out_regressions,
            self.seed_regressions,
            self.reroutes,
            self.changed,
            self.negative_violations
        )
    }
}

/// 演化回执：模型卡 + 发布记录 + 四类闸门结论。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvolutionReceipt {
    pub dialect: String,
    pub proposal_id: String,
    pub kind: ProposalKind,
    pub base_generation: u64,
    pub base_fingerprint: String,
    pub candidate_fingerprint: String,
    pub clock: EvolutionClock,
    pub verdict: EvolutionVerdict,
    pub findings: Vec<EvolutionFinding>,
    pub replay: ReplaySummary,
    pub replay_outcomes: Vec<ReplayOutcome>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub committed_generation: Option<u64>,
}

impl EvolutionReceipt {
    pub fn gate_findings(&self, gate: GateKind) -> impl Iterator<Item = &EvolutionFinding> {
        self.findings.iter().filter(move |finding| finding.gate == gate)
    }

    pub fn gate_tally(&self, gate: GateKind) -> (usize, usize) {
        let errors = self
            .gate_findings(gate)
            .filter(|finding| finding.severity == FindingSeverity::Error)
            .count();
        let warnings = self
            .gate_findings(gate)
            .filter(|finding| finding.severity == FindingSeverity::Warning)
            .count();
        (errors, warnings)
    }

    pub fn render(&self) -> String {
        use std::fmt::Write;
        let mut out = String::new();
        let _ = writeln!(
            out,
            "模式演化回执：提案 {}（{}）→ 判定 {}",
            self.proposal_id,
            self.kind.label(),
            self.verdict.label()
        );
        let _ = writeln!(
            out,
            " 基础第{}代 {} → 候选指纹 {}",
            self.base_generation, self.base_fingerprint, self.candidate_fingerprint
        );
        let _ = writeln!(out, " {}", self.clock.render());
        for gate in [
            GateKind::Structural,
            GateKind::Reference,
            GateKind::Behavior,
            GateKind::Ledger,
        ] {
            let (errors, warnings) = self.gate_tally(gate);
            let _ = writeln!(
                out,
                " {}闸门：错误 {}，警告 {}",
                gate.label(),
                errors,
                warnings
            );
        }
        let _ = writeln!(out, " {}", self.replay.render());
        if !self.findings.is_empty() {
            let _ = writeln!(out, " 发现：");
            for finding in self.findings.iter().take(20) {
                let _ = writeln!(out, "  - {}", finding.render());
            }
            if self.findings.len() > 20 {
                let _ = writeln!(out, "  …另有 {} 条", self.findings.len() - 20);
            }
        }
        if let Some(generation) = self.committed_generation {
            let _ = writeln!(out, " 已写入第{}代。", generation);
        }
        out
    }
}

fn lint_key(finding: &LintFinding) -> String {
    let kind = match finding.kind {
        LintKind::Redundant => "redundant",
        LintKind::Conflict => "conflict",
        LintKind::Shadowed => "shadowed",
    };
    format!(
        "{}\u{0}{}\u{0}{}",
        kind, finding.subject_id, finding.object_id
    )
}

fn new_lint_findings(base: &PatternLibrary, candidate: &PatternLibrary) -> Vec<LintFinding> {
    let existing: BTreeSet<String> = lint(base).iter().map(lint_key).collect();
    lint(candidate)
        .into_iter()
        .filter(|finding| !existing.contains(&lint_key(finding)))
        .collect()
}

fn entry_map(library: &PatternLibrary) -> BTreeMap<String, PatternEntry> {
    library
        .entries()
        .iter()
        .map(|entry| (entry.id(), entry.clone()))
        .collect()
}

/// 对一次提案执行四类闸门。纯函数：不修改台账，也不写图。
pub fn evaluate_proposal(
    base: &PatternGeneration,
    candidate: &PatternLibrary,
    proposal: &EvolutionProposal,
    episodes: &[PatternEpisode],
    config: &SystemConfig,
    clock: EvolutionClock,
) -> EvolutionReceipt {
    let mut findings = Vec::new();
    let base_library = base.library();
    let candidate_entries = entry_map(candidate);
    let base_entries = entry_map(&base_library);
    let candidate_fingerprint = pattern_ir::library_fingerprint(candidate.entries());

    if let Err(error) = proposal.apply(base) {
        findings.push(EvolutionFinding::new(
            GateKind::Structural,
            FindingSeverity::Error,
            "proposal_invalid",
            &proposal.id,
            error.to_string(),
        ));
    }

    // 结构：条目字段、新增 lint、与基础代指纹的关系。
    for entry in candidate.entries() {
        if let Err(error) = validate_entry(entry) {
            findings.push(EvolutionFinding::new(
                GateKind::Structural,
                FindingSeverity::Error,
                "entry_invalid",
                entry.name.clone(),
                error.to_string(),
            ));
        }
    }
    for finding in new_lint_findings(&base_library, candidate) {
        let kind_label = match finding.kind {
            LintKind::Redundant => "新增冗余",
            LintKind::Conflict => "新增冲突",
            LintKind::Shadowed => "新增遮蔽",
        };
        let same_route = candidate_entries
            .get(&finding.subject_id)
            .zip(candidate_entries.get(&finding.object_id))
            .map(|(left, right)| left.query_type == right.query_type)
            .unwrap_or(false);
        let severity = match finding.kind {
            LintKind::Conflict => FindingSeverity::Error,
            LintKind::Shadowed if !same_route => FindingSeverity::Error,
            LintKind::Shadowed | LintKind::Redundant => FindingSeverity::Warning,
        };
        findings.push(EvolutionFinding::new(
            GateKind::Structural,
            severity,
            match finding.kind {
                LintKind::Redundant => "new_redundant",
                LintKind::Conflict => "new_conflict",
                LintKind::Shadowed => "new_shadowed",
            },
            format!("{} ↔ {}", finding.subject_name, finding.object_name),
            format!("{}：{}", kind_label, finding.detail),
        ));
    }

    // 引用/能力：路由类型必须仍可执行；新增词面必须已是同义词规范形。
    for entry in candidate.entries() {
        if entry.route && !config_contract::is_query_type_supported(&entry.query_type) {
            findings.push(EvolutionFinding::new(
                GateKind::Reference,
                FindingSeverity::Error,
                "dangling_query_type",
                entry.name.clone(),
                format!(
                    "路由类型「{}」不在可执行能力表中（配置契约会认为它是悬空引用）",
                    entry.query_type
                ),
            ));
        }
    }
    let synonyms = synonyms(config);
    for entry in proposal.added_entries() {
        for elem in &entry.elems {
            if let crate::pattern_ir::Elem::Word { surface } = elem {
                if let Some(canonical) = synonyms.get(surface) {
                    if canonical != surface {
                        findings.push(EvolutionFinding::new(
                            GateKind::Reference,
                            FindingSeverity::Error,
                            "synonym_literal_not_normalized",
                            entry.name.clone(),
                            format!(
                                "词面「{}」会在输入归一后变成「{}」，规则永远等不到自己",
                                surface, canonical
                            ),
                        ));
                    }
                }
            }
        }
    }

    // 账本：已确认经验指向的模式不能无声消失。
    let base_ids = base.ids();
    let candidate_ids: BTreeSet<String> = candidate_entries.keys().cloned().collect();
    for episode in episodes {
        if episode.outcome != EpisodeOutcome::Matched || episode.acceptances <= episode.rejections {
            continue;
        }
        let Some(winner) = episode.winner_id.as_deref() else {
            continue;
        };
        if base_ids.contains(winner) && !candidate_ids.contains(winner) {
            let explicitly_removed = base_entries
                .get(winner)
                .map(|entry| proposal.removes_entry(entry))
                .unwrap_or(false);
            findings.push(EvolutionFinding::new(
                GateKind::Ledger,
                if explicitly_removed {
                    FindingSeverity::Warning
                } else {
                    FindingSeverity::Error
                },
                if explicitly_removed {
                    "confirmed_pattern_retired"
                } else {
                    "confirmed_pattern_vanished"
                },
                episode.input.clone(),
                format!(
                    "已确认经验指向的模式 #{} 在候选代中不存在{}",
                    &winner.chars().take(8).collect::<String>(),
                    if explicitly_removed {
                        "（提案显式退役）"
                    } else {
                        "，且提案没有显式删除它"
                    }
                ),
            ));
        }
        if !base_ids.contains(winner) && !candidate_ids.contains(winner) {
            findings.push(EvolutionFinding::new(
                GateKind::Ledger,
                FindingSeverity::Warning,
                "preexisting_ghost_episode",
                episode.input.clone(),
                format!(
                    "经验指向的模式 #{} 在前代与候选代都不存在（历史幽灵条目）",
                    &winner.chars().take(8).collect::<String>()
                ),
            ));
        }
    }

    // 行为：前代/候选逐例回放。
    let corpus = build_replay_corpus(base, candidate, episodes, proposal, config);
    let mut summary = ReplaySummary {
        cases: corpus.len(),
        ..ReplaySummary::default()
    };
    let mut outcomes = Vec::with_capacity(corpus.len());
    let added_ids = proposal.added_ids();
    let added_sources = proposal.added_sources();
    for case in &corpus {
        let tokens = tokens_for(case, config, &synonyms);
        let base_route = route(&base_library, &tokens, &case.context);
        let candidate_route = route(candidate, &tokens, &case.context);
        let base_query = base_route
            .as_ref()
            .map(|matched| matched.entry.query_type.clone());
        let candidate_query = candidate_route
            .as_ref()
            .map(|matched| matched.entry.query_type.clone());
        let base_winner = base_route.as_ref().map(|matched| matched.entry.id());
        let candidate_winner = candidate_route.as_ref().map(|matched| matched.entry.id());

        match case.source {
            ReplaySource::Episode(_) => summary.episodes += 1,
            ReplaySource::Constructed(_) => summary.constructed += 1,
            ReplaySource::HeldOut => summary.held_out += 1,
        }
        if case.seed {
            summary.seed_cases += 1;
        }

        let candidate_is_added = candidate_winner
            .as_ref()
            .map(|winner| {
                added_ids.contains(winner)
                    || candidate_entries
                        .get(winner)
                        .map(|entry| added_sources.iter().any(|added| entry.source.starts_with(added)))
                        .unwrap_or(false)
            })
            .unwrap_or(false);

        let verdict = if case.label == ReplayLabel::Negative {
            if candidate_is_added && base_winner != candidate_winner {
                summary.negative_violations += 1;
                findings.push(EvolutionFinding::new(
                    GateKind::Behavior,
                    FindingSeverity::Error,
                    "negative_overgeneralized",
                    case.input.clone(),
                    "被用户否定的经验被本次新增规则重新命中（过泛化）",
                ));
                ReplayVerdict::NegativeViolation
            } else if base_query == candidate_query {
                ReplayVerdict::Stable
            } else {
                ReplayVerdict::Changed
            }
        } else {
            let base_correct = case.expected_query_type.is_some()
                && base_query.as_deref() == case.expected_query_type.as_deref();
            let candidate_correct = case.expected_query_type.is_some()
                && candidate_query.as_deref() == case.expected_query_type.as_deref();
            if base_correct && !candidate_correct {
                summary.regressions += 1;
                if case.seed {
                    summary.seed_regressions += 1;
                } else {
                    summary.held_out_regressions += 1;
                }
                findings.push(EvolutionFinding::new(
                    GateKind::Behavior,
                    FindingSeverity::Error,
                    "confirmed_route_regressed",
                    case.input.clone(),
                    format!(
                        "前代路由到「{}」，候选代变成「{}」；{}",
                        case.expected_query_type.as_deref().unwrap_or(""),
                        candidate_query.as_deref().unwrap_or("未命中"),
                        case.label.label()
                    ),
                ));
                ReplayVerdict::Regression
            } else if !base_correct && candidate_correct {
                summary.gains += 1;
                if case.seed {
                    summary.seed_gains += 1;
                } else {
                    summary.held_out_gains += 1;
                }
                ReplayVerdict::Gain
            } else if base_correct && candidate_correct && base_winner != candidate_winner {
                summary.reroutes += 1;
                findings.push(EvolutionFinding::new(
                    GateKind::Behavior,
                    FindingSeverity::Warning,
                    "confirmed_route_rerouted",
                    case.input.clone(),
                    format!(
                        "路由类型仍为「{}」，但胜者从 #{} 换成 #{}",
                        case.expected_query_type.as_deref().unwrap_or(""),
                        base_winner
                            .as_deref()
                            .map(|id| id.chars().take(8).collect::<String>())
                            .unwrap_or_default(),
                        candidate_winner
                            .as_deref()
                            .map(|id| id.chars().take(8).collect::<String>())
                            .unwrap_or_default()
                    ),
                ));
                ReplayVerdict::Reroute
            } else if base_query == candidate_query && base_winner == candidate_winner {
                if case.expected_query_type.is_some() {
                    summary.stable += 1;
                    ReplayVerdict::Stable
                } else {
                    summary.stable_misses += 1;
                    ReplayVerdict::StableMiss
                }
            } else if !base_correct && !candidate_correct && base_query == candidate_query {
                summary.stable_misses += 1;
                ReplayVerdict::StableMiss
            } else {
                summary.changed += 1;
                ReplayVerdict::Changed
            }
        };

        outcomes.push(ReplayOutcome {
            case_id: case.id.clone(),
            input: case.input.clone(),
            label: case.label,
            seed: case.seed,
            expected_query_type: case.expected_query_type.clone(),
            base_query_type: base_query,
            candidate_query_type: candidate_query,
            base_winner_id: base_winner,
            candidate_winner_id: candidate_winner,
            verdict,
        });
    }

    let has_structural_error = findings
        .iter()
        .any(|finding| finding.gate != GateKind::Behavior && finding.severity == FindingSeverity::Error);
    let has_behavior_error = findings
        .iter()
        .any(|finding| finding.gate == GateKind::Behavior && finding.severity == FindingSeverity::Error);
    let has_warning = findings
        .iter()
        .any(|finding| finding.severity == FindingSeverity::Warning);
    let route_additions = proposal
        .added_entries()
        .iter()
        .any(|entry| entry.route);
    let verdict = if has_structural_error {
        EvolutionVerdict::Reject
    } else if has_behavior_error {
        EvolutionVerdict::Isolate
    } else if has_warning || (route_additions && summary.held_out_gains == 0) {
        EvolutionVerdict::Observe
    } else {
        EvolutionVerdict::Commit
    };

    EvolutionReceipt {
        dialect: EVOLUTION_DIALECT.to_string(),
        proposal_id: proposal.id.clone(),
        kind: proposal.kind,
        base_generation: base.generation,
        base_fingerprint: base.fingerprint.clone(),
        candidate_fingerprint,
        clock,
        verdict,
        findings,
        replay: summary,
        replay_outcomes: outcomes,
        committed_generation: None,
    }
}

// ---------------------------------------------------------------------------
// 台账
// ---------------------------------------------------------------------------

/// 追加式模式演化台账：快照 + 回执，可落 JSON、可前向回滚。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvolutionLedger {
    #[serde(default = "default_dialect")]
    pub dialect: String,
    #[serde(default)]
    pub generations: Vec<PatternGeneration>,
    #[serde(default)]
    pub receipts: Vec<EvolutionReceipt>,
    #[serde(default = "default_max_generations")]
    pub max_generations: usize,
    #[serde(default = "default_max_receipts")]
    pub max_receipts: usize,
}

fn default_dialect() -> String {
    EVOLUTION_DIALECT.to_string()
}

fn default_max_generations() -> usize {
    DEFAULT_MAX_GENERATIONS
}

fn default_max_receipts() -> usize {
    DEFAULT_MAX_RECEIPTS
}

impl Default for EvolutionLedger {
    fn default() -> Self {
        Self {
            dialect: EVOLUTION_DIALECT.to_string(),
            generations: Vec::new(),
            receipts: Vec::new(),
            max_generations: DEFAULT_MAX_GENERATIONS,
            max_receipts: DEFAULT_MAX_RECEIPTS,
        }
    }
}

impl EvolutionLedger {
    pub fn with_limits(max_generations: usize, max_receipts: usize) -> Self {
        Self {
            max_generations: max_generations.max(1),
            max_receipts: max_receipts.max(1),
            ..Self::default()
        }
    }

    pub fn len(&self) -> usize {
        self.generations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.generations.is_empty()
    }

    pub fn current(&self) -> Option<&PatternGeneration> {
        self.generations.last()
    }

    pub fn generation(&self, generation: u64) -> Option<&PatternGeneration> {
        self.generations
            .iter()
            .find(|snapshot| snapshot.generation == generation)
    }

    fn next_generation(&self) -> u64 {
        self.generations
            .iter()
            .map(|snapshot| snapshot.generation)
            .max()
            .map(|max| max.saturating_add(1))
            .unwrap_or(0)
    }

    /// 记录初始代；台账非空时幂等返回当前代。
    pub fn ensure_initial(
        &mut self,
        library: &PatternLibrary,
        clock: EvolutionClock,
        note: impl Into<String>,
    ) -> PatternGeneration {
        if let Some(current) = self.current() {
            return current.clone();
        }
        let generation = PatternGeneration::snapshot(
            0,
            None,
            library,
            clock,
            GenerationStatus::Active,
            note,
        );
        self.generations.push(generation.clone());
        generation
    }

    /// 执行一次提案：先过闸门，只有提交/观察才写入下一代。
    pub fn transact(
        &mut self,
        proposal: &EvolutionProposal,
        episodes: &[PatternEpisode],
        config: &SystemConfig,
        clock: EvolutionClock,
        note: impl Into<String>,
    ) -> Result<EvolutionReceipt, EvolutionError> {
        let base = self
            .generation(proposal.base_generation)
            .cloned()
            .ok_or(EvolutionError::BaseGenerationMissing {
                requested: proposal.base_generation,
                current: self.current().map(|snapshot| snapshot.generation).unwrap_or(0),
            })?;
        let candidate = proposal.apply(&base)?;
        let mut receipt = evaluate_proposal(&base, &candidate, proposal, episodes, config, clock);
        if receipt.verdict.may_commit() {
            let status = match receipt.verdict {
                EvolutionVerdict::Commit => GenerationStatus::Active,
                EvolutionVerdict::Observe => GenerationStatus::Probation,
                _ => GenerationStatus::Quarantined,
            };
            let generation = PatternGeneration::snapshot(
                self.next_generation(),
                Some(base.generation),
                &candidate,
                clock,
                status,
                note,
            );
            receipt.committed_generation = Some(generation.generation);
            self.generations.push(generation);
            self.trim_generations();
        }
        self.receipts.push(receipt.clone());
        self.trim_receipts();
        Ok(receipt)
    }

    /// 前向回滚：把旧代条目作为新代提交，不删除中间代，也不倒转时间。
    pub fn rollback(
        &mut self,
        target: u64,
        clock: EvolutionClock,
        note: impl Into<String>,
    ) -> Result<PatternGeneration, EvolutionError> {
        let current = self
            .current()
            .cloned()
            .ok_or(EvolutionError::EmptyLedger)?;
        let target_generation = self
            .generation(target)
            .cloned()
            .ok_or(EvolutionError::GenerationMissing(target))?;
        let generation = PatternGeneration::snapshot(
            self.next_generation(),
            Some(current.generation),
            &target_generation.library(),
            clock,
            GenerationStatus::Active,
            format!("回滚到第{}代：{}", target, note.into()),
        );
        let receipt = EvolutionReceipt {
            dialect: EVOLUTION_DIALECT.to_string(),
            proposal_id: format!("rollback-{}", generation.generation),
            kind: ProposalKind::Rollback,
            base_generation: current.generation,
            base_fingerprint: current.fingerprint.clone(),
            candidate_fingerprint: generation.fingerprint.clone(),
            clock,
            verdict: EvolutionVerdict::Commit,
            findings: Vec::new(),
            replay: ReplaySummary::default(),
            replay_outcomes: Vec::new(),
            committed_generation: Some(generation.generation),
        };
        self.generations.push(generation.clone());
        self.trim_generations();
        self.receipts.push(receipt);
        self.trim_receipts();
        Ok(generation)
    }

    fn trim_generations(&mut self) {
        while self.generations.len() > self.max_generations.max(1) {
            self.generations.remove(0);
        }
    }

    fn trim_receipts(&mut self) {
        while self.receipts.len() > self.max_receipts.max(1) {
            self.receipts.remove(0);
        }
    }

    /// 人类可读台账：当前代 + 最近回执摘要。
    pub fn render_report(&self) -> String {
        use std::fmt::Write;
        let mut out = String::new();
        let _ = writeln!(
            out,
            "模式演化台账：{} 代，{} 份回执。",
            self.generations.len(),
            self.receipts.len()
        );
        if self.generations.is_empty() {
            let _ = writeln!(out, "（还没有代；用「演化快照」记录当前模式库。）");
            return out;
        }
        if let Some(current) = self.current() {
            let _ = writeln!(out, "当前：{}", current.render());
        }
        let _ = writeln!(out, "最近代：");
        for generation in self.generations.iter().rev().take(6) {
            let _ = writeln!(out, " - {}", generation.render());
        }
        if !self.receipts.is_empty() {
            let _ = writeln!(out, "最近回执：");
            for receipt in self.receipts.iter().rev().take(5) {
                let _ = writeln!(
                    out,
                    " - {}（{}）→ {}，回放 {} 例，回归 {}",
                    receipt.proposal_id,
                    receipt.kind.label(),
                    receipt.verdict.label(),
                    receipt.replay.cases,
                    receipt.replay.regressions
                );
            }
        }
        out
    }

    pub fn save_to_path(&self, path: impl AsRef<Path>) -> Result<(), EvolutionError> {
        let text = serde_json::to_string_pretty(self)
            .map_err(|error| EvolutionError::Json(error.to_string()))?;
        fs::write(path, text).map_err(|error| EvolutionError::Io(error.to_string()))
    }

    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self, EvolutionError> {
        let path = path.as_ref();
        let text = fs::read_to_string(path).map_err(|error| EvolutionError::Io(error.to_string()))?;
        let ledger: Self =
            serde_json::from_str(&text).map_err(|error| EvolutionError::Json(error.to_string()))?;
        if ledger.dialect != EVOLUTION_DIALECT {
            return Err(EvolutionError::BadDialect(ledger.dialect));
        }
        Ok(ledger)
    }
}

// ---------------------------------------------------------------------------
// 错误
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvolutionError {
    EmptyLedger,
    BaseGenerationMissing { requested: u64, current: u64 },
    GenerationMissing(u64),
    FingerprintMismatch { expected: String, actual: String },
    InvalidEntry(String),
    InvalidEdit(String),
    EditTargetMissing(String),
    AmbiguousEdit { key: String, matches: usize },
    BadDialect(String),
    Io(String),
    Json(String),
}

impl std::fmt::Display for EvolutionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvolutionError::EmptyLedger => write!(formatter, "演化台账还没有任何代"),
            EvolutionError::BaseGenerationMissing { requested, current } => write!(
                formatter,
                "提案要求第{}代，但台账当前只到第{}代",
                requested, current
            ),
            EvolutionError::GenerationMissing(generation) => {
                write!(formatter, "台账中没有第{}代", generation)
            }
            EvolutionError::FingerprintMismatch { expected, actual } => write!(
                formatter,
                "提案基于指纹 {}，但当前代指纹是 {}（模式库已漂移）",
                expected, actual
            ),
            EvolutionError::InvalidEntry(detail) => write!(formatter, "模式条目无效：{}", detail),
            EvolutionError::InvalidEdit(detail) => write!(formatter, "提案编辑无效：{}", detail),
            EvolutionError::EditTargetMissing(key) => {
                write!(formatter, "提案要删除的目标不存在：{}", key)
            }
            EvolutionError::AmbiguousEdit { key, matches } => write!(
                formatter,
                "提案删除目标「{}」匹配到 {} 条，请给出更长的 id",
                key, matches
            ),
            EvolutionError::BadDialect(dialect) => {
                write!(formatter, "演化台账方言不受支持：{}", dialect)
            }
            EvolutionError::Io(detail) => write!(formatter, "演化台账读写失败：{}", detail),
            EvolutionError::Json(detail) => write!(formatter, "演化台账 JSON 无效：{}", detail),
        }
    }
}

impl std::error::Error for EvolutionError {}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pattern_ir::Elem;

    fn config() -> SystemConfig {
        toml::from_str(include_str!("../config.toml")).expect("config.toml 必须可解析")
    }

    fn entry(
        name: &str,
        source: &str,
        query_type: &str,
        priority: i32,
        words: &[&str],
    ) -> PatternEntry {
        PatternEntry::new(
            name,
            source,
            query_type,
            priority,
            true,
            words.iter().map(|word| Elem::word(*word)).collect(),
        )
    }

    fn base_library() -> PatternLibrary {
        let mut library = PatternLibrary::new();
        library.add(entry(
            "首都查询",
            "config.toml#首都查询",
            "capital_query",
            10,
            &["北京", "的", "首都", "是", "哪里"],
        ));
        library.add(entry(
            "年龄查询",
            "config.toml#年龄查询",
            "age_query",
            10,
            &["小明", "多大"],
        ));
        library
    }

    fn base_generation() -> PatternGeneration {
        PatternGeneration::snapshot(
            0,
            None,
            &base_library(),
            EvolutionClock::new(1, 2, 0, 3),
            GenerationStatus::Active,
            "初始",
        )
    }

    fn matched_episode(
        input: &str,
        words: &[&str],
        winner: &PatternEntry,
        acceptances: u32,
        rejections: u32,
    ) -> PatternEpisode {
        PatternEpisode {
            input: input.to_string(),
            tokens: words
                .iter()
                .map(|word| ((*word).to_string(), "n".to_string()))
                .collect(),
            winner_id: Some(winner.id()),
            winner_name: Some(winner.name.clone()),
            query_type: Some(winner.query_type.clone()),
            entities: Vec::new(),
            bindings: Vec::new(),
            focus: None,
            focus_type: None,
            last_move: None,
            pending_clarify: Some(false),
            outcome: EpisodeOutcome::Matched,
            count: 1,
            acceptances,
            rejections,
            first_turn: 1,
            last_turn: 1,
            frontier: None,
        }
    }

    #[test]
    fn generation_fingerprint_is_order_independent_and_rollback_is_forward() {
        let base = base_generation();
        let mut reversed = PatternLibrary::new();
        for entry in base.entries.iter().rev() {
            reversed.add(entry.clone());
        }
        let reordered = PatternGeneration::snapshot(
            1,
            Some(0),
            &reversed,
            EvolutionClock::default(),
            GenerationStatus::Active,
            "逆序",
        );
        assert_eq!(base.fingerprint, reordered.fingerprint);

        let mut ledger = EvolutionLedger::default();
        ledger.ensure_initial(&base.library(), base.clock, "初始");
        let rolled = ledger
            .rollback(0, EvolutionClock::new(2, 3, 1, 4), "测试回滚")
            .unwrap();
        assert_eq!(rolled.generation, 1);
        assert_eq!(rolled.parent, Some(0));
        assert_eq!(rolled.fingerprint, base.fingerprint);
        assert_eq!(rolled.entries, base.entries);
        assert_eq!(ledger.receipts.len(), 1);
        assert_eq!(ledger.receipts[0].kind, ProposalKind::Rollback);
    }

    #[test]
    fn structural_gate_rejects_dangling_query_type() {
        let base = base_generation();
        let proposal = EvolutionProposal::new(
            "teach-bad",
            ProposalKind::Teach,
            &base,
            "悬空路由",
        )
        .with_edit(ProposalEdit::add(entry(
            "坏模式",
            "taught#bad",
            "not_a_real_query",
            50,
            &["坏", "模式"],
        )));
        let candidate = proposal.apply(&base).unwrap();
        let receipt = evaluate_proposal(
            &base,
            &candidate,
            &proposal,
            &[],
            &config(),
            EvolutionClock::default(),
        );
        assert_eq!(receipt.verdict, EvolutionVerdict::Reject);
        assert!(receipt
            .findings
            .iter()
            .any(|finding| finding.code == "dangling_query_type"));
    }

    #[test]
    fn behavior_gate_catches_confirmed_route_regression() {
        let base = base_generation();
        let original = base
            .entries
            .iter()
            .find(|entry| entry.query_type == "capital_query")
            .unwrap();
        let mut episode = matched_episode(
            "北京的省会是哪里",
            &["北京", "的", "首都", "是", "哪里"],
            original,
            2,
            0,
        );
        episode.tokens[0].1 = "ns".to_string();
        // 词性泛化 + 不同路由且更高优先级：候选代会把已确认问句改道。
        let hijack = PatternEntry::new(
            "首都改道",
            "taught#hijack",
            "age_query",
            99,
            true,
            vec![
                Elem::pos("ns"),
                Elem::word("的"),
                Elem::word("首都"),
                Elem::word("是"),
                Elem::word("哪里"),
            ],
        );
        let proposal = EvolutionProposal::new(
            "hijack",
            ProposalKind::Teach,
            &base,
            "故意改道",
        )
        .with_edit(ProposalEdit::add(hijack));
        let candidate = proposal.apply(&base).unwrap();
        let receipt = evaluate_proposal(
            &base,
            &candidate,
            &proposal,
            &[episode],
            &config(),
            EvolutionClock::default(),
        );
        assert_eq!(receipt.verdict, EvolutionVerdict::Isolate);
        assert!(receipt.replay.regressions >= 1);
        assert!(receipt
            .replay_outcomes
            .iter()
            .any(|outcome| outcome.verdict == ReplayVerdict::Regression));
    }

    #[test]
    fn seed_episode_does_not_count_as_held_out_generalization() {
        let base = base_generation();
        let new_entry = entry(
            "天气问法",
            "taught#weather",
            "age_query",
            50,
            &["今天", "天气", "如何"],
        );
        let seed = matched_episode(
            "今天天气如何",
            &["今天", "天气", "如何"],
            &new_entry,
            1,
            0,
        );
        let proposal =
            EvolutionProposal::new("weather", ProposalKind::Teach, &base, "新问法")
                .with_edit(ProposalEdit::add(new_entry));
        let candidate = proposal.apply(&base).unwrap();
        let receipt = evaluate_proposal(
            &base,
            &candidate,
            &proposal,
            &[seed],
            &config(),
            EvolutionClock::default(),
        );
        assert!(receipt.replay.seed_cases >= 1);
        assert!(receipt.replay.seed_gains >= 1);
        assert_eq!(receipt.replay.held_out_gains, 0);
        assert_eq!(receipt.verdict, EvolutionVerdict::Observe);
    }

    #[test]
    fn ledger_gate_detects_confirmed_pattern_vanishing() {
        let base = base_generation();
        let original = base
            .entries
            .iter()
            .find(|entry| entry.query_type == "age_query")
            .unwrap();
        let episode = matched_episode("小明多大", &["小明", "多大"], original, 3, 0);
        let proposal = EvolutionProposal::new(
            "drop-age",
            ProposalKind::Forget,
            &base,
            "删除年龄规则",
        )
        .with_edit(ProposalEdit::remove_id(original.id()));
        let candidate = proposal.apply(&base).unwrap();
        let receipt = evaluate_proposal(
            &base,
            &candidate,
            &proposal,
            &[episode],
            &config(),
            EvolutionClock::default(),
        );
        assert!(receipt
            .findings
            .iter()
            .any(|finding| finding.code == "confirmed_pattern_retired"));
        assert_eq!(receipt.verdict, EvolutionVerdict::Isolate);
    }

    #[test]
    fn negative_episode_blocks_overgeneralization() {
        let base = base_generation();
        let original = base
            .entries
            .iter()
            .find(|entry| entry.query_type == "age_query")
            .unwrap();
        let mut negative = matched_episode("小明多大", &["小明", "多大"], original, 0, 2);
        negative.tokens[0].1 = "nr".to_string();
        let aggressive = PatternEntry::new(
            "过泛化年龄",
            "taught#aggressive",
            "capital_query",
            99,
            true,
            vec![Elem::pos("nr"), Elem::word("多大")],
        );
        let proposal =
            EvolutionProposal::new("aggressive", ProposalKind::Teach, &base, "过泛化")
                .with_edit(ProposalEdit::add(aggressive));
        let candidate = proposal.apply(&base).unwrap();
        let receipt = evaluate_proposal(
            &base,
            &candidate,
            &proposal,
            &[negative],
            &config(),
            EvolutionClock::default(),
        );
        assert!(receipt.replay.negative_violations >= 1);
        assert_eq!(receipt.verdict, EvolutionVerdict::Isolate);
    }

    #[test]
    fn transact_persists_roundtrip_and_commits_safe_proposal() {
        let base = base_generation();
        let mut ledger = EvolutionLedger::default();
        ledger.ensure_initial(&base.library(), base.clock, "初始");
        let safe = entry(
            "新年龄问法",
            "taught#age2",
            "age_query",
            5,
            &["李四", "几岁"],
        );
        let proposal =
            EvolutionProposal::new("safe", ProposalKind::Teach, &base, "补充问法")
                .with_edit(ProposalEdit::add(safe))
                .with_held_out(HeldOutCase {
                    input: "李四几岁".to_string(),
                    expected_query_type: "age_query".to_string(),
                    note: "留出".to_string(),
                });
        let receipt = ledger
            .transact(
                &proposal,
                &[],
                &config(),
                EvolutionClock::new(1, 2, 1, 3),
                "补充",
            )
            .unwrap();
        assert_eq!(receipt.verdict, EvolutionVerdict::Commit);
        assert_eq!(receipt.committed_generation, Some(1));
        assert_eq!(ledger.current().unwrap().status, GenerationStatus::Active);
        assert_eq!(ledger.current().unwrap().fingerprint, receipt.candidate_fingerprint);

        let path = std::env::temp_dir().join(format!(
            "knowprolog-evolution-{}.json",
            std::process::id()
        ));
        ledger.save_to_path(&path).unwrap();
        let loaded = EvolutionLedger::load_from_path(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(loaded, ledger);
    }

    #[test]
    fn synonym_literal_check_rejects_unormalized_taught_word() {
        let mut config = config();
        let mut mappings = HashMap::new();
        mappings.insert("老婆".to_string(), "妻子".to_string());
        config.synonyms = Some(crate::SynonymsConfig { mappings });
        let base = base_generation();
        let proposal = EvolutionProposal::new(
            "stale-surface",
            ProposalKind::Teach,
            &base,
            "旧词面",
        )
        .with_edit(ProposalEdit::add(entry(
            "老婆问法",
            "taught#stale",
            "age_query",
            50,
            &["老婆", "多大"],
        )));
        let candidate = proposal.apply(&base).unwrap();
        let receipt = evaluate_proposal(
            &base,
            &candidate,
            &proposal,
            &[],
            &config,
            EvolutionClock::default(),
        );
        assert_eq!(receipt.verdict, EvolutionVerdict::Reject);
        assert!(receipt
            .findings
            .iter()
            .any(|finding| finding.code == "synonym_literal_not_normalized"));
    }

    #[test]
    fn proposal_apply_rejects_fingerprint_drift_and_missing_target() {
        let base = base_generation();
        let mut proposal = EvolutionProposal::new(
            "stale-base",
            ProposalKind::Teach,
            &base,
            "旧基础",
        );
        proposal.base_fingerprint = "deadbeef".to_string();
        assert!(matches!(
            proposal.apply(&base),
            Err(EvolutionError::FingerprintMismatch { .. })
        ));

        let missing = EvolutionProposal::new(
            "missing-target",
            ProposalKind::Forget,
            &base,
            "删不存在",
        )
        .with_edit(ProposalEdit::remove_id("ffffffff"));
        assert!(matches!(
            missing.apply(&base),
            Err(EvolutionError::EditTargetMissing(_))
        ));
    }

    #[test]
    fn report_renders_four_gates_and_replay() {
        let base = base_generation();
        let proposal =
            EvolutionProposal::new("noop", ProposalKind::Migration, &base, "无变化");
        let candidate = proposal.apply(&base).unwrap();
        let receipt = evaluate_proposal(
            &base,
            &candidate,
            &proposal,
            &[],
            &config(),
            EvolutionClock::default(),
        );
        assert_eq!(receipt.verdict, EvolutionVerdict::Commit);
        let text = receipt.render();
        assert!(text.contains("结构闸门"));
        assert!(text.contains("引用/能力闸门"));
        assert!(text.contains("行为闸门"));
        assert!(text.contains("账本闸门"));
        assert!(text.contains("回放"));
    }
}

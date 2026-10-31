//! 证明即记忆：带证明足迹、搜索成本与字节预算的会话级证明缓存。
//!
//! 与 [`crate::delta_cache::DeltaQualifiedCache`] 的分工：
//! - `DeltaQualifiedCache` 回答「这次知识变化是否碰到这条缓存」；
//! - 本模块回答「这次证明值不值得留、留到什么时候、复用省了多少搜索」。
//!
//! 记忆条目不是第二份真相源，而是对已有 [`crate::resolution_engine::Solution`]
//! / 证明视图的索引：
//!
//! - `key` 是规范请求键（去首尾标点与空白），同一次提问才能命中；
//! - `footprint` 记录证明实际读过的实体、谓词与基原子；
//! - `cost` 记录生成证明时的确定性搜索计数，命中时按次数累计「节省步数」；
//! - `payload_bytes` 是调用方给出的规范 JSON 估算值，不是 RSS；
//! - 容量淘汰按 `(命中次数 + 1) × 节省步数 / 字节` 的整数密度比较，
//!   同分先淘汰更久未用者，再按键名稳定决出，全程无浮点、无随机。
//!
//! 认识论纪律：只缓存搜索完整的结果；被截断的搜索可以登记为
//! [`ProofStatus::MayBeIncomplete`]，但调用方不得把它当最终答案复用。
//! 知识增量、别名变化或整体重建按 [`DeltaFootprint`] 保守失效：
//! 宁可多失效，也不返回陈旧证明。

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::context_image::fnv1a_hex;
use crate::delta_cache::DeltaFootprint;

/// 证明记忆格式版本。
pub const PROOF_MEMORY_VERSION: u32 = 1;
/// 默认字节预算（规范 JSON 估算值）。
pub const DEFAULT_MAX_BYTES: usize = 512 * 1024;
/// 默认条目上限。
pub const DEFAULT_MAX_ENTRIES: usize = 64;
/// 请求键保留的字符上限。
pub const MAX_KEY_CHARS: usize = 512;
/// 单条目的固定记账开销（键、状态、计数与索引的保守估计）。
pub const ENTRY_OVERHEAD_BYTES: usize = 96;

/// 证明记忆的容量参数。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProofMemoryConfig {
    pub max_bytes: usize,
    pub max_entries: usize,
}

impl Default for ProofMemoryConfig {
    fn default() -> Self {
        Self {
            max_bytes: DEFAULT_MAX_BYTES,
            max_entries: DEFAULT_MAX_ENTRIES,
        }
    }
}

impl ProofMemoryConfig {
    pub fn new(max_bytes: usize, max_entries: usize) -> Self {
        Self {
            max_bytes: max_bytes.max(1),
            max_entries: max_entries.max(1),
        }
    }
}

/// 证明缓存条目是否完整。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProofStatus {
    /// 搜索完整，可直接复用。
    Complete,
    /// 搜索被截断：只能作为「可能不完整」的线索，不得当最终答案。
    MayBeIncomplete,
}

impl ProofStatus {
    pub fn code(self) -> &'static str {
        match self {
            ProofStatus::Complete => "complete",
            ProofStatus::MayBeIncomplete => "may_be_incomplete",
        }
    }
}

/// 证明实际读过的知识足迹。`watch_any` 表示无法解析出原子时的保守退化。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProofFootprint {
    #[serde(default)]
    pub entities: BTreeSet<String>,
    #[serde(default)]
    pub predicates: BTreeSet<String>,
    #[serde(default)]
    pub atoms: BTreeSet<String>,
    /// 足迹无法表达：任何知识变化都应让它失效。
    #[serde(default)]
    pub watch_any: bool,
}

impl ProofFootprint {
    pub fn new() -> Self {
        Self::default()
    }

    /// 从 `Atom::display()` 形式的文本提取谓词、常量与原子本身。
    pub fn from_canonical_atom(text: &str) -> Option<Self> {
        let trimmed = text.trim();
        let open = trimmed.find('(')?;
        let close = trimmed.rfind(')')?;
        if close < open {
            return None;
        }
        let predicate = trimmed[..open].trim();
        if predicate.is_empty() {
            return None;
        }
        let mut footprint = Self::new();
        footprint.predicates.insert(predicate.to_string());
        footprint.atoms.insert(trimmed.to_string());
        let body = &trimmed[open + 1..close];
        for term in body.split(',') {
            let term = term.trim();
            if term.is_empty() || term.starts_with('?') {
                continue;
            }
            footprint.entities.insert(term.to_string());
        }
        Some(footprint)
    }

    /// 合并一组规范原子文本。一个原子都解析不出来时退化为 `watch_any`。
    pub fn from_canonical_atoms<'a>(texts: impl IntoIterator<Item = &'a str>) -> Self {
        let mut footprint = Self::new();
        for text in texts {
            if let Some(parsed) = Self::from_canonical_atom(text) {
                footprint.merge(&parsed);
            }
        }
        if footprint.atoms.is_empty() {
            footprint.watch_any = true;
        }
        footprint
    }

    pub fn merge(&mut self, other: &Self) {
        self.entities.extend(other.entities.iter().cloned());
        self.predicates.extend(other.predicates.iter().cloned());
        self.atoms.extend(other.atoms.iter().cloned());
        self.watch_any |= other.watch_any;
    }

    pub fn is_empty(&self) -> bool {
        !self.watch_any
            && self.entities.is_empty()
            && self.predicates.is_empty()
            && self.atoms.is_empty()
    }

    /// 保守判定：增量是否可能影响这条证明。
    pub fn touched_by(&self, delta: &DeltaFootprint) -> bool {
        if delta.rebuild || self.watch_any {
            return true;
        }
        if intersects(&self.entities, &delta.names)
            || intersects(&self.predicates, &delta.predicates)
            || intersects(&self.entities, &delta.removed_names)
            || intersects(&self.predicates, &delta.removed_predicates)
        {
            return true;
        }
        if delta
            .aliases
            .iter()
            .any(|(surface, canonical)| {
                self.entities.contains(surface) || self.entities.contains(canonical)
            })
        {
            return true;
        }
        delta.facts.iter().any(|fact| {
            let (names, predicates) = fact_parts(fact);
            names.iter().any(|name| self.entities.contains(name))
                || predicates
                    .iter()
                    .any(|predicate| self.predicates.contains(predicate))
        }) || delta.removed_facts.iter().any(|fact| {
            let (names, predicates) = fact_parts(fact);
            names.iter().any(|name| self.entities.contains(name))
                || predicates
                    .iter()
                    .any(|predicate| self.predicates.contains(predicate))
        })
    }
}

fn intersects(left: &BTreeSet<String>, right: &BTreeSet<String>) -> bool {
    if left.is_empty() || right.is_empty() {
        return false;
    }
    left.iter().any(|value| right.contains(value))
}

fn fact_parts(fact: &crate::prolog_engine::PrologFact) -> (Vec<String>, Vec<String>) {
    match fact {
        crate::prolog_engine::PrologFact::Entity {
            name,
            entity_type,
            properties,
        } => {
            let mut names = vec![name.clone(), entity_type.clone()];
            names.extend(properties.values().cloned());
            (names, properties.keys().cloned().collect())
        }
        crate::prolog_engine::PrologFact::Relation { from, rel_type, to } => {
            (vec![from.clone(), to.clone()], vec![rel_type.clone()])
        }
    }
}

/// 生成证明时的确定性搜索成本。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProofSearchCost {
    pub goals: usize,
    pub goal_attempts: usize,
    pub fact_checks: usize,
    pub rule_attempts: usize,
    pub depth_cutoffs: usize,
    pub budget_cutoffs: usize,
    pub solution_cutoffs: usize,
    pub state_cutoffs: usize,
}

impl ProofSearchCost {
    /// 一次未命中缓存的搜索大约走过的步数。
    pub fn steps(self) -> u64 {
        let units = [
            self.goal_attempts,
            self.fact_checks,
            self.rule_attempts,
            self.depth_cutoffs,
            self.budget_cutoffs,
            self.solution_cutoffs,
            self.state_cutoffs,
        ];
        units
            .into_iter()
            .fold(0u64, |sum, value| sum.saturating_add(value as u64))
    }
}

/// 一条待写入的证明记录。
#[derive(Debug, Clone)]
pub struct ProofRecord<T> {
    pub key: String,
    pub payload: T,
    pub payload_bytes: usize,
    pub footprint: ProofFootprint,
    pub cost: ProofSearchCost,
    pub status: ProofStatus,
}

impl<T> ProofRecord<T> {
    pub fn new(key: impl Into<String>, payload: T, payload_bytes: usize) -> Self {
        Self {
            key: key.into(),
            payload,
            payload_bytes: payload_bytes.max(1),
            footprint: ProofFootprint::new(),
            cost: ProofSearchCost::default(),
            status: ProofStatus::Complete,
        }
    }

    pub fn with_footprint(mut self, footprint: ProofFootprint) -> Self {
        self.footprint = footprint;
        self
    }

    pub fn with_cost(mut self, cost: ProofSearchCost) -> Self {
        self.cost = cost;
        self
    }

    pub fn with_status(mut self, status: ProofStatus) -> Self {
        self.status = status;
        self
    }
}

/// 一条已缓存的证明。
#[derive(Debug, Clone)]
pub struct ProofMemoryEntry<T> {
    pub key: String,
    pub payload: T,
    pub payload_bytes: usize,
    pub footprint: ProofFootprint,
    pub cost: ProofSearchCost,
    pub status: ProofStatus,
    pub generation: u64,
    pub hits: u64,
    pub last_used: u64,
    pub inserted_at: u64,
}

impl<T> ProofMemoryEntry<T> {
    /// 规范 JSON 估算 + 固定索引开销。
    pub fn estimated_bytes(&self) -> usize {
        self.key
            .len()
            .saturating_add(self.payload_bytes)
            .saturating_add(ENTRY_OVERHEAD_BYTES)
    }

    fn utility_numerator(&self) -> u128 {
        (u128::from(self.hits) + 1)
            .saturating_mul(u128::from(self.cost.steps().max(1)))
    }

    fn utility_denominator(&self) -> u128 {
        self.estimated_bytes().max(1) as u128
    }
}

/// 一次命中的只读副本。
#[derive(Debug, Clone)]
pub struct ProofMemoryHit<T> {
    pub payload: T,
    pub payload_bytes: usize,
    pub footprint: ProofFootprint,
    pub cost: ProofSearchCost,
    pub status: ProofStatus,
    pub hits: u64,
}

/// 淘汰原因。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionReason {
    EntryLimit,
    ByteLimit,
}

impl EvictionReason {
    pub fn label(self) -> &'static str {
        match self {
            EvictionReason::EntryLimit => "entry_limit",
            EvictionReason::ByteLimit => "byte_limit",
        }
    }
}

/// 一次淘汰记录。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProofEviction {
    pub key: String,
    pub estimated_bytes: usize,
    pub reason: EvictionReason,
}

/// 一次写入的回执。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProofStoreReceipt {
    pub stored: bool,
    pub oversized: bool,
    pub replaced: bool,
    pub evicted: Vec<ProofEviction>,
}

impl ProofStoreReceipt {
    pub fn evicted_keys(&self) -> Vec<String> {
        self.evicted.iter().map(|entry| entry.key.clone()).collect()
    }
}

/// 一次「查缓存 / 重算并写入」的调用回执。文本只含计数与键，不含答案正文。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProofMemoryAccess {
    pub key: String,
    pub hit: bool,
    pub hits: u64,
    pub payload_bytes: usize,
    pub saved_steps: u64,
    pub status: ProofStatus,
    pub stored: bool,
    pub oversized: bool,
    pub evicted: Vec<String>,
}

impl ProofMemoryAccess {
    pub fn hit(key: String, hit: ProofMemoryHitAccess) -> Self {
        Self {
            key,
            hit: true,
            hits: hit.hits,
            payload_bytes: hit.payload_bytes,
            saved_steps: hit.saved_steps,
            status: hit.status,
            stored: true,
            oversized: false,
            evicted: Vec::new(),
        }
    }

    pub fn miss(key: String) -> Self {
        Self {
            key,
            hit: false,
            hits: 0,
            payload_bytes: 0,
            saved_steps: 0,
            status: ProofStatus::Complete,
            stored: false,
            oversized: false,
            evicted: Vec::new(),
        }
    }

    pub fn with_store(mut self, receipt: &ProofStoreReceipt) -> Self {
        self.stored = receipt.stored;
        self.oversized = receipt.oversized;
        self.evicted = receipt.evicted_keys();
        self
    }

    pub fn render_zh(&self) -> String {
        if self.hit {
            format!(
                "证明记忆：命中（第 {} 次复用，约省 {} 步 / {} 字节）",
                self.hits, self.saved_steps, self.payload_bytes
            )
        } else if self.oversized {
            "证明记忆：未命中（条目超出字节预算，未写入）".to_string()
        } else if self.stored {
            format!("证明记忆：未命中（已写入 {} 字节）", self.payload_bytes)
        } else if !self.evicted.is_empty() {
            format!(
                "证明记忆：未命中（写入时淘汰 {} 条）",
                self.evicted.len()
            )
        } else if self.status == ProofStatus::MayBeIncomplete {
            "证明记忆：未命中（搜索被截断，不缓存最终答案）".to_string()
        } else {
            "证明记忆：未命中".to_string()
        }
    }
}

/// 供 [`ProofMemoryAccess::hit`] 使用的轻量命中摘要。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProofMemoryHitAccess {
    pub hits: u64,
    pub payload_bytes: usize,
    pub saved_steps: u64,
    pub status: ProofStatus,
}

/// 一次知识增量复核的确定性报告。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProofRevalidation {
    pub generation: u64,
    pub retained: usize,
    pub invalidated: usize,
    pub cleared_all: bool,
    pub gap: bool,
    pub invalidated_keys: Vec<String>,
}

/// 证明记忆的累计计数。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProofMemoryStats {
    pub lookups: u64,
    pub hits: u64,
    pub misses: u64,
    pub stores: u64,
    pub evictions: u64,
    pub invalidations: u64,
    pub rejected_oversized: u64,
}

/// 有界证明记忆。`T` 只需可克隆；字节数由调用方给出。
pub struct ProofMemory<T> {
    config: ProofMemoryConfig,
    generation: u64,
    clock: u64,
    entries: BTreeMap<String, ProofMemoryEntry<T>>,
    stats: ProofMemoryStats,
}

impl<T: Clone> ProofMemory<T> {
    pub fn new(config: ProofMemoryConfig) -> Self {
        Self {
            config: ProofMemoryConfig::new(config.max_bytes, config.max_entries),
            generation: 0,
            clock: 0,
            entries: BTreeMap::new(),
            stats: ProofMemoryStats::default(),
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(ProofMemoryConfig::default())
    }

    pub fn config(&self) -> ProofMemoryConfig {
        self.config
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn stats(&self) -> &ProofMemoryStats {
        &self.stats
    }

    pub fn estimated_bytes(&self) -> usize {
        self.entries
            .values()
            .fold(0usize, |sum, entry| sum.saturating_add(entry.estimated_bytes()))
    }

    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    /// 按键读取并记一次命中；只返回完整条目。
    pub fn lookup_complete(&mut self, key: &str) -> Option<ProofMemoryHit<T>> {
        self.stats.lookups = self.stats.lookups.saturating_add(1);
        let stale = self
            .entries
            .get(key)
            .is_some_and(|entry| entry.generation != self.generation);
        if stale {
            self.entries.remove(key);
        }
        let Some(entry) = self.entries.get_mut(key) else {
            self.stats.misses = self.stats.misses.saturating_add(1);
            return None;
        };
        if entry.status != ProofStatus::Complete {
            self.stats.misses = self.stats.misses.saturating_add(1);
            return None;
        }
        self.clock = self.clock.saturating_add(1);
        entry.hits = entry.hits.saturating_add(1);
        entry.last_used = self.clock;
        self.stats.hits = self.stats.hits.saturating_add(1);
        Some(ProofMemoryHit {
            payload: entry.payload.clone(),
            payload_bytes: entry.payload_bytes,
            footprint: entry.footprint.clone(),
            cost: entry.cost,
            status: entry.status,
            hits: entry.hits,
        })
    }

    /// 写入一条证明，按字节与条目上限做确定性淘汰。
    pub fn store(&mut self, record: ProofRecord<T>) -> ProofStoreReceipt {
        self.stats.stores = self.stats.stores.saturating_add(1);
        let key = canonical_key(&record.key);
        let mut receipt = ProofStoreReceipt::default();
        let entry = ProofMemoryEntry {
            key: key.clone(),
            payload: record.payload,
            payload_bytes: record.payload_bytes.max(1),
            footprint: record.footprint,
            cost: record.cost,
            status: record.status,
            generation: self.generation,
            hits: 0,
            last_used: self.next_clock(),
            inserted_at: self.clock,
        };
        if entry.estimated_bytes() > self.config.max_bytes {
            receipt.oversized = true;
            self.stats.rejected_oversized = self.stats.rejected_oversized.saturating_add(1);
            return receipt;
        }
        receipt.replaced = self.entries.remove(&key).is_some();
        self.entries.insert(key, entry);

        while self.entries.len() > self.config.max_entries
            || self.estimated_bytes() > self.config.max_bytes
        {
            let reason = if self.entries.len() > self.config.max_entries {
                EvictionReason::EntryLimit
            } else {
                EvictionReason::ByteLimit
            };
            let Some(victim) = self.select_victim() else {
                break;
            };
            if let Some(removed) = self.entries.remove(&victim) {
                receipt.evicted.push(ProofEviction {
                    key: victim,
                    estimated_bytes: removed.estimated_bytes(),
                    reason,
                });
                self.stats.evictions = self.stats.evictions.saturating_add(1);
            }
        }
        receipt.stored = self.entries.contains_key(&canonical_key(&record.key));
        receipt
    }

    /// 按知识增量逐条复核；触碰即失效，绝不留下可能陈旧的证明。
    pub fn observe_delta(&mut self, delta: &DeltaFootprint) -> ProofRevalidation {
        let mut report = ProofRevalidation {
            generation: delta.generation,
            ..ProofRevalidation::default()
        };
        if delta.rebuild {
            self.invalidate_all(&mut report, true);
            return report;
        }
        if delta.generation <= self.generation {
            return report;
        }
        if delta.generation != self.generation.saturating_add(1) {
            report.gap = true;
            self.invalidate_all(&mut report, true);
            return report;
        }
        let keys: Vec<String> = self.entries.keys().cloned().collect();
        for key in keys {
            let touched = self
                .entries
                .get(&key)
                .is_some_and(|entry| entry.footprint.touched_by(delta));
            if touched {
                self.entries.remove(&key);
                report.invalidated = report.invalidated.saturating_add(1);
                report.invalidated_keys.push(key);
                self.stats.invalidations = self.stats.invalidations.saturating_add(1);
            } else {
                if let Some(entry) = self.entries.get_mut(&key) {
                    entry.generation = delta.generation;
                }
                report.retained = report.retained.saturating_add(1);
            }
        }
        self.generation = delta.generation;
        report
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.generation = self.generation.saturating_add(1);
        self.clock = 0;
    }

    /// 稳定的审计回执：计数、字节口径与按密度排序的条目摘要。
    pub fn render_receipt(&self) -> String {
        let mut lines = vec![format!(
            "证明记忆 v{}：{} 条 / {} 字节（估算口径：规范 JSON + 键 + {} 字节/条开销）",
            PROOF_MEMORY_VERSION,
            self.entries.len(),
            self.estimated_bytes(),
            ENTRY_OVERHEAD_BYTES
        )];
        lines.push(format!(
            "命中 {} / 查询 {}（未命中 {}），写入 {}，淘汰 {}，增量失效 {}，超大拒绝 {}",
            self.stats.hits,
            self.stats.lookups,
            self.stats.misses,
            self.stats.stores,
            self.stats.evictions,
            self.stats.invalidations,
            self.stats.rejected_oversized
        ));
        if self.entries.is_empty() {
            lines.push("当前没有可复用的证明。".to_string());
            return lines.join("\n");
        }
        let mut entries: Vec<&ProofMemoryEntry<T>> = self.entries.values().collect();
        entries.sort_by(|left, right| {
            let left_num = left.utility_numerator();
            let left_den = left.utility_denominator();
            let right_num = right.utility_numerator();
            let right_den = right.utility_denominator();
            right_num
                .saturating_mul(left_den)
                .cmp(&left_num.saturating_mul(right_den))
                .then_with(|| right.hits.cmp(&left.hits))
                .then_with(|| left.key.cmp(&right.key))
        });
        for entry in entries.iter().take(8) {
            lines.push(format!(
                "  {}  命中 {} 次 / {} 字节 / 节省约 {} 步 / {}",
                entry.key,
                entry.hits,
                entry.estimated_bytes(),
                entry.cost.steps().saturating_mul(entry.hits),
                entry.status.code()
            ));
        }
        if entries.len() > 8 {
            lines.push(format!("  ……另有 {} 条未展开", entries.len() - 8));
        }
        lines.join("\n")
    }

    fn invalidate_all(&mut self, report: &mut ProofRevalidation, cleared: bool) {
        report.cleared_all = cleared;
        report.invalidated = report.invalidated.saturating_add(self.entries.len());
        report.invalidated_keys = self.entries.keys().cloned().collect();
        self.stats.invalidations = self
            .stats
            .invalidations
            .saturating_add(self.entries.len() as u64);
        self.entries.clear();
        self.generation = report.generation;
    }

    fn next_clock(&mut self) -> u64 {
        self.clock = self.clock.saturating_add(1);
        self.clock
    }

    fn select_victim(&self) -> Option<String> {
        let mut victim: Option<&ProofMemoryEntry<T>> = None;
        for entry in self.entries.values() {
            let Some(current) = victim else {
                victim = Some(entry);
                continue;
            };
            if utility_less(entry, current) {
                victim = Some(entry);
            }
        }
        victim.map(|entry| entry.key.clone())
    }
}

fn utility_less<T>(left: &ProofMemoryEntry<T>, right: &ProofMemoryEntry<T>) -> bool {
    let left_scaled = left
        .utility_numerator()
        .saturating_mul(right.utility_denominator());
    let right_scaled = right
        .utility_numerator()
        .saturating_mul(left.utility_denominator());
    left_scaled
        .cmp(&right_scaled)
        .then_with(|| left.last_used.cmp(&right.last_used))
        .then_with(|| left.key.cmp(&right.key))
        .is_lt()
}

/// 规范请求键：去首尾标点/空白、折叠内部空白、截断到字符上限。
pub fn canonical_key(text: &str) -> String {
    let trimmed = text.trim().trim_end_matches([
        '？', '?', '。', '！', '!', '，', ',', '；', ';', '：', ':', ' ', '\t', '\n', '\r',
    ]);
    let mut key = String::with_capacity(trimmed.len());
    let mut last_space = false;
    for character in trimmed.chars() {
        if character.is_whitespace() {
            if !last_space {
                key.push(' ');
                last_space = true;
            }
        } else {
            key.push(character);
            last_space = false;
        }
    }
    if key.chars().count() > MAX_KEY_CHARS {
        let digest = fnv1a_hex(key.as_bytes());
        let mut bounded: String = key.chars().take(MAX_KEY_CHARS).collect();
        bounded.push('#');
        bounded.push_str(&digest);
        return bounded;
    }
    key
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::delta_cache::DeltaFootprint;
    use crate::prolog_engine::PrologFact;
    use crate::resolution_engine::SearchStats;

    fn cost(steps: usize) -> ProofSearchCost {
        ProofSearchCost {
            goals: 1,
            goal_attempts: steps,
            fact_checks: 0,
            rule_attempts: 0,
            ..ProofSearchCost::default()
        }
    }

    fn record(key: &str, payload: &str, bytes: usize, steps: usize) -> ProofRecord<String> {
        ProofRecord::new(key, payload.to_string(), bytes)
            .with_cost(cost(steps))
            .with_footprint(ProofFootprint::from_canonical_atom("师傅(张三, 李四)").unwrap())
    }

    #[test]
    fn canonical_key_is_stable_and_bounded() {
        assert_eq!(canonical_key("  张三的师傅是谁？ "), "张三的师傅是谁");
        assert_eq!(canonical_key("a \t b。"), "a b");
        let long = "字".repeat(MAX_KEY_CHARS + 8);
        let bounded = canonical_key(&long);
        assert!(bounded.starts_with(&"字".repeat(MAX_KEY_CHARS)));
        assert!(bounded.contains('#'));
        assert_ne!(canonical_key(&format!("{}甲", long)), canonical_key(&format!("{}乙", long)));
    }

    #[test]
    fn footprint_parses_atoms_and_rejects_unparsed_support() {
        let footprint = ProofFootprint::from_canonical_atom("师傅(张三, 李四)").unwrap();
        assert!(footprint.predicates.contains("师傅"));
        assert!(footprint.entities.contains("张三"));
        assert!(footprint.entities.contains("李四"));
        assert!(!footprint.watch_any);

        let variable = ProofFootprint::from_canonical_atom("师傅(张三, ?X)").unwrap();
        assert!(!variable.entities.contains("?X"));

        let fallback = ProofFootprint::from_canonical_atoms(["张三的师傅"]);
        assert!(fallback.watch_any);
    }

    #[test]
    fn lookup_counts_hits_and_preserves_payload() {
        let mut memory = ProofMemory::new(ProofMemoryConfig::new(64 * 1024, 8));
        let receipt = memory.store(record("师傅(张三, ?X)", "李四", 16, 42));
        assert!(receipt.stored);
        let hit = memory.lookup_complete("师傅(张三, ?X)").unwrap();
        assert_eq!(hit.payload, "李四");
        assert_eq!(hit.hits, 1);
        let hit = memory.lookup_complete("师傅(张三, ?X)").unwrap();
        assert_eq!(hit.hits, 2);
        assert_eq!(memory.stats().lookups, 2);
        assert_eq!(memory.stats().hits, 2);
        assert!(memory.lookup_complete("别的目标").is_none());
        assert_eq!(memory.stats().misses, 1);
    }

    #[test]
    fn byte_budget_evicts_lowest_utility_density() {
        let mut memory = ProofMemory::new(ProofMemoryConfig::new(600, 8));
        memory.store(record("低密度(张三, 李四)", "低", 120, 1));
        memory.store(record("高密度(张三, 李四)", "高", 120, 100));
        for _ in 0..20 {
            memory.lookup_complete("高密度(张三, 李四)").unwrap();
        }
        let receipt = memory.store(record("新条目(张三, 李四)", "新", 120, 50));
        assert!(receipt.stored);
        assert!(
            receipt.evicted_keys().contains(&"低密度(张三, 李四)".to_string()),
            "evicted: {:?}",
            receipt.evicted_keys()
        );
        assert!(memory.lookup_complete("高密度(张三, 李四)").is_some());
    }

    #[test]
    fn oversized_record_is_rejected_without_touching_existing_entries() {
        let mut memory = ProofMemory::new(ProofMemoryConfig::new(300, 4));
        memory.store(record("已有(张三, 李四)", "已有", 64, 10));
        let receipt = memory.store(record("巨大(张三, 李四)", "巨大", 10_000, 10));
        assert!(!receipt.stored);
        assert!(receipt.oversized);
        assert_eq!(memory.len(), 1);
        assert_eq!(memory.stats().rejected_oversized, 1);
    }

    #[test]
    fn delta_only_invalidates_proofs_with_touched_footprint() {
        let mut memory = ProofMemory::new(ProofMemoryConfig::new(64 * 1024, 8));
        memory.store(record("张三的师傅(张三, ?X)", "李四", 16, 20));
        memory.store(
            record("王五的身高(王五, ?X)", "190", 16, 20).with_footprint(
                ProofFootprint::from_canonical_atom("身高(王五, ?X)").unwrap(),
            ),
        );

        let mut delta = DeltaFootprint::new(1);
        delta.touch_name("张三");
        let report = memory.observe_delta(&delta);
        assert_eq!(report.invalidated, 1);
        assert_eq!(report.retained, 1);
        assert_eq!(report.invalidated_keys, vec!["张三的师傅(张三, ?X)"]);
        assert!(memory.lookup_complete("王五的身高(王五, ?X)").is_some());
        assert!(memory.lookup_complete("张三的师傅(张三, ?X)").is_none());
    }

    #[test]
    fn reset_and_generation_gap_clear_everything() {
        let mut memory = ProofMemory::new(ProofMemoryConfig::new(64 * 1024, 8));
        memory.store(record("目标(张三, 李四)", "值", 16, 20));
        let report = memory.observe_delta(&DeltaFootprint::rebuild(7));
        assert!(report.cleared_all);
        assert_eq!(report.invalidated, 1);
        assert!(memory.is_empty());

        let mut memory = ProofMemory::new(ProofMemoryConfig::new(64 * 1024, 8));
        memory.store(record("目标(张三, 李四)", "值", 16, 20));
        let report = memory.observe_delta(&DeltaFootprint::new(9));
        assert!(report.cleared_all);
        assert!(report.gap);
        assert!(memory.is_empty());
    }

    #[test]
    fn aliases_invalidate_only_when_names_are_touched() {
        let mut memory = ProofMemory::new(ProofMemoryConfig::new(64 * 1024, 8));
        memory.store(record("目标(张三, 李四)", "值", 16, 20));
        let mut unrelated = DeltaFootprint::new(1);
        unrelated.aliases = vec![("王五".to_string(), "小王".to_string())];
        let report = memory.observe_delta(&unrelated);
        assert_eq!(report.invalidated, 0);
        assert_eq!(report.retained, 1);

        let mut touched = DeltaFootprint::new(2);
        touched.aliases = vec![("张三".to_string(), "张三丰".to_string())];
        let report = memory.observe_delta(&touched);
        assert_eq!(report.invalidated, 1);
    }

    #[test]
    fn incomplete_entry_never_served_as_complete() {
        let mut memory = ProofMemory::new(ProofMemoryConfig::new(64 * 1024, 8));
        let incomplete = record("未知目标(张三, ?X)", "可能值", 16, 20)
            .with_status(ProofStatus::MayBeIncomplete);
        memory.store(incomplete);
        assert!(memory.lookup_complete("未知目标(张三, ?X)").is_none());
        assert_eq!(memory.stats().misses, 1);
    }

    #[test]
    fn receipt_is_deterministic_for_same_state() {
        let mut memory = ProofMemory::new(ProofMemoryConfig::new(64 * 1024, 8));
        memory.store(record("乙(张三, 李四)", "值", 16, 10));
        memory.store(record("甲(张三, 李四)", "值", 16, 10));
        let first = memory.render_receipt();
        let second = memory.render_receipt();
        assert_eq!(first, second);
        assert!(first.contains("证明记忆 v"));
    }

    #[test]
    fn stats_mapping_matches_search_cost_contract() {
        let stats = SearchStats {
            goals: 2,
            goal_attempts: 3,
            fact_checks: 5,
            rule_attempts: 7,
            ..SearchStats::default()
        };
        let mapped = ProofSearchCost {
            goals: stats.goals,
            goal_attempts: stats.goal_attempts,
            fact_checks: stats.fact_checks,
            rule_attempts: stats.rule_attempts,
            depth_cutoffs: stats.depth_cutoffs,
            budget_cutoffs: stats.budget_cutoffs,
            solution_cutoffs: stats.solution_cutoffs,
            state_cutoffs: stats.state_cutoffs,
        };
        assert_eq!(mapped.steps(), 15);
    }

    #[test]
    fn entity_fact_names_and_predicates_are_seen() {
        let mut memory = ProofMemory::new(ProofMemoryConfig::new(64 * 1024, 8));
        memory.store(record("目标(张三, 李四)", "值", 16, 20));
        let mut delta = DeltaFootprint::new(1);
        delta.extend_facts(&[PrologFact::Entity {
            name: "张三".to_string(),
            entity_type: "人物".to_string(),
            properties: [("职业".to_string(), "教师".to_string())]
                .into_iter()
                .collect(),
        }]);
        let report = memory.observe_delta(&delta);
        assert_eq!(report.invalidated, 1);
    }
}

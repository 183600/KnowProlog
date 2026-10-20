//! ELL：外链名册（External Link Ledger）。
//!
//! 外部知识（ownthink 等）不在本地写权限内，`kb_revision` 管不了它，
//! 于是为外链单独建立一套一致性模型：**逻辑时刻（tick）+ 时限（TTL）+ 显式刷新**。
//! 每次取回都留册：出处（source_uri）、取回时刻（fetched_tick）、内容摘要。
//!
//! 设计要点：
//!   - **可审计**：每条缓存都带来源与取回轮次，回答可逐字复述出处；
//!   - **可撤销**：`refresh` 标记重取、`forget` 删除条目，学习与遗忘都显式；
//!   - **确认即学习**：`confirm_alias` 把「别名 = 规范实体」记为同义链接，
//!     参与后续实体归一（纯符号的 sameAs 别名册）；
//!   - **负缓存**：外链明确回答"查无此实体"时也记入名册，
//!     下一次不重复打网（网络失败**不**写负缓存）；
//!   - **有界 + 确定性淘汰**：容量上限 + LRU（最近使用的在尾部），
//!     所有报告按实体字典序输出，可逐字节断言；
//!   - **持久化**：复用 `ContextImage` 信封（分节哈希 + 原子写），
//!     损坏时整册拒绝、空册降级，绝不按半截数据作答。
//!
//! 时间语义：`tick` 由调用方在每个对话轮次开始时推进（一比一于用户轮次），
//! 与本项目的工程基因一致——不依赖挂钟，测试可逐字节断言。

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;

use crate::context_image::ContextImage;
use crate::perf_ledger::PerfLedger;

/// 持久化分节名。
pub const LEDGER_SECTION: &str = "external_ledger";
/// 默认容量（条目数上限）。
pub const DEFAULT_CAPACITY: usize = 64;
/// 默认时限：0 表示不随逻辑时刻过期（人类知识变化慢，刷新交给显式命令）。
pub const DEFAULT_TTL_TICKS: u64 = 0;

/// 条目状态：有知识 / 确无收录（负缓存）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LedgerStatus {
    Present,
    Absent,
}

/// 外链名册中的一页：某实体在某个逻辑时刻从某个来源取回的知识。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LedgerEntry {
    /// 规范实体名（别名已归一）。
    pub entity: String,
    /// 取回来源（外链 URI，如 ownthink:http://api.ownthink.com/...）。
    pub source_uri: String,
    /// 取回时的逻辑时刻（会话 tick）。
    pub fetched_tick: u64,
    /// 有知识 / 确无收录。
    pub status: LedgerStatus,
    #[serde(default)]
    pub description: String,
    /// 属性值对：[属性, 值]。
    #[serde(default)]
    pub avp: Vec<Vec<String>>,
    #[serde(default)]
    pub tags: Vec<String>,
}

impl LedgerEntry {
    /// 是否为负缓存（来源明确回答"没有这个实体"）。
    pub fn is_absent(&self) -> bool {
        matches!(self.status, LedgerStatus::Absent)
    }

    /// 是否超过时限。`ttl_ticks == 0` 表示永不过期。
    /// 存于时刻 t、时限 q：时刻 t..t+q-1 有效，t+q 起过期。
    pub fn is_stale(&self, now_tick: u64, ttl_ticks: u64) -> bool {
        ttl_ticks > 0 && now_tick.saturating_sub(self.fetched_tick) >= ttl_ticks
    }

    /// 单行摘要（确定性）。
    pub fn summary(&self, now_tick: u64, ttl_ticks: u64) -> String {
        if self.is_absent() {
            format!(
                "- 「{}」← {}（第 {} 轮核验：确无收录{}{}）",
                self.entity,
                self.source_uri,
                self.fetched_tick,
                if self.is_stale(now_tick, ttl_ticks) {
                    "，已过期"
                } else {
                    ""
                },
                if self.tags.is_empty() {
                    String::new()
                } else {
                    format!("，标签：{}", self.tags.join("、"))
                }
            )
        } else {
            format!(
                "- 「{}」← {}（第 {} 轮取回，属性 {} 条{}{}{}）",
                self.entity,
                self.source_uri,
                self.fetched_tick,
                self.avp.len(),
                if self.description.is_empty() {
                    String::new()
                } else {
                    format!("，描述：{}", truncate_chars(&self.description, 40))
                },
                if self.tags.is_empty() {
                    String::new()
                } else {
                    format!("，标签：{}", self.tags.join("、"))
                },
                if self.is_stale(now_tick, ttl_ticks) {
                    "，已过期"
                } else {
                    ""
                }
            )
        }
    }
}

fn truncate_chars(text: &str, limit: usize) -> String {
    let mut chars = text.chars();
    let mut truncated: String = chars.by_ref().take(limit).collect();
    if chars.next().is_some() {
        truncated.push('…');
    }
    truncated
}

/// 名册查询结果。
#[derive(Debug)]
pub enum LedgerLookup<'a> {
    /// 新鲜条目（Present 或 Absent 均可能）。
    Fresh(&'a LedgerEntry),
    /// 条目已存在但超过时限：可作低置信兜底，也应触发重取。
    Stale(&'a LedgerEntry),
    /// 未收录。
    Miss,
}

/// 外链名册加载结果（与 `ImportJournal` 一致的损坏语义）。
#[derive(Debug)]
pub enum LedgerLoad {
    /// 文件不存在（首次运行）。
    Missing,
    /// 成功恢复的条目数。
    Loaded(usize),
    /// 完整性自检失败，整册拒绝（调用方保持空册）。
    Damaged(String),
}

/// 有界、按实体索引、带出处与逻辑时刻的外链缓存。
#[derive(Debug)]
pub struct ExternalLedger {
    entries: BTreeMap<String, LedgerEntry>,
    /// LRU 序：尾部最新，头部最旧（容量溢出时淘汰头部）。
    order: Vec<String>,
    /// 会话逻辑时刻，由调用方每个用户轮次推进一次。
    tick: u64,
    capacity: usize,
    ttl_ticks: u64,
    /// 用户确认的同义链接：别名 → 规范实体。
    aliases: BTreeMap<String, String>,
    perf: Arc<PerfLedger>,
}

impl Default for ExternalLedger {
    fn default() -> Self {
        Self::new()
    }
}

impl ExternalLedger {
    pub fn new() -> Self {
        Self::with_config(
            DEFAULT_CAPACITY,
            DEFAULT_TTL_TICKS,
            Arc::new(PerfLedger::new()),
        )
    }

    pub fn with_perf_ledger(perf: Arc<PerfLedger>) -> Self {
        Self::with_config(DEFAULT_CAPACITY, DEFAULT_TTL_TICKS, perf)
    }

    pub fn with_config(capacity: usize, ttl_ticks: u64, perf: Arc<PerfLedger>) -> Self {
        Self {
            entries: BTreeMap::new(),
            order: Vec::new(),
            tick: 0,
            capacity: capacity.max(1),
            ttl_ticks,
            aliases: BTreeMap::new(),
            perf,
        }
    }

    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// 推进逻辑时刻（每个用户轮次调用一次）。
    pub fn advance_tick(&mut self) -> u64 {
        self.tick += 1;
        self.tick
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn ttl_ticks(&self) -> u64 {
        self.ttl_ticks
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn alias_count(&self) -> usize {
        self.aliases.len()
    }

    /// 解析别名：命中已确认的同义链接时登记计数并返回规范实体。
    pub fn resolve(&mut self, mention: &str) -> String {
        if let Some(target) = self.aliases.get(mention) {
            self.perf.bump("external_ledger_alias_hit");
            return target.clone();
        }
        mention.to_string()
    }

    /// 只读别名查询（不计数，供展示）。
    pub fn alias_target(&self, mention: &str) -> Option<&str> {
        self.aliases.get(mention).map(String::as_str)
    }

    /// 按实体名查询（调用方应先用 `resolve` 归一）。
    pub fn lookup(&mut self, entity: &str) -> LedgerLookup<'_> {
        let now = self.tick;
        let ttl = self.ttl_ticks;
        let stale = self
            .entries
            .get(entity)
            .map(|entry| entry.is_stale(now, ttl))
            .unwrap_or(false);
        if stale {
            self.perf.bump("external_ledger_stale_hit");
            self.touch(entity);
            return LedgerLookup::Stale(&self.entries[entity]);
        }
        match self.entries.get(entity) {
            Some(entry) => {
                let is_absent = entry.is_absent();
                if is_absent {
                    self.perf.bump("external_ledger_absent_hit");
                } else {
                    self.perf.bump("external_ledger_hit");
                }
                self.touch(entity);
                LedgerLookup::Fresh(&self.entries[entity])
            }
            None => {
                self.perf.bump("external_ledger_miss");
                LedgerLookup::Miss
            }
        }
    }

    /// 只读取页（不触碰 LRU、不计数）。
    pub fn entry(&self, entity: &str) -> Option<&LedgerEntry> {
        self.entries.get(entity)
    }

    /// 按实体名字典序枚举条目。
    pub fn entries(&self) -> impl Iterator<Item = &LedgerEntry> {
        self.entries.values()
    }

    /// 写入一页有知识的外链条目（取回时刻 = 当前逻辑时刻）。
    pub fn store_entity(
        &mut self,
        entity: &str,
        source_uri: &str,
        description: String,
        avp: Vec<Vec<String>>,
        tags: Vec<String>,
    ) {
        if entity.trim().is_empty() {
            return;
        }
        let entry = LedgerEntry {
            entity: entity.to_string(),
            source_uri: source_uri.to_string(),
            fetched_tick: self.tick,
            status: LedgerStatus::Present,
            description,
            avp,
            tags,
        };
        self.insert(entry);
        self.perf.bump("external_ledger_store");
    }

    /// 写入一页负缓存：来源明确回答"确无此实体"。
    pub fn store_absent(&mut self, entity: &str, source_uri: &str) {
        if entity.trim().is_empty() {
            return;
        }
        let entry = LedgerEntry {
            entity: entity.to_string(),
            source_uri: source_uri.to_string(),
            fetched_tick: self.tick,
            status: LedgerStatus::Absent,
            description: String::new(),
            avp: Vec::new(),
            tags: Vec::new(),
        };
        self.insert(entry);
        self.perf.bump("external_ledger_store_absent");
    }

    /// 标记重取：删除条目，下次查询将重新打网。返回该条目是否存在。
    pub fn refresh(&mut self, entity: &str) -> bool {
        let removed = self.remove_entry(entity);
        if removed {
            self.perf.bump("external_ledger_refresh");
        }
        removed
    }

    /// 遗忘条目（同义链接保留）。返回该条目是否存在。
    pub fn forget(&mut self, entity: &str) -> bool {
        let removed = self.remove_entry(entity);
        if removed {
            self.perf.bump("external_ledger_forget");
        }
        removed
    }

    /// 确认同义链接：别名 → 规范实体。别名与规范实体相同或为空时不成立。
    pub fn confirm_alias(&mut self, alias: &str, target: &str) -> bool {
        let alias = alias.trim();
        let target = target.trim();
        if alias.is_empty() || target.is_empty() || alias == target {
            return false;
        }
        self.aliases.insert(alias.to_string(), target.to_string());
        self.perf.bump("external_ledger_alias_confirm");
        true
    }

    /// 解除同义链接。返回是否存在。
    pub fn unlink_alias(&mut self, alias: &str) -> bool {
        self.aliases.remove(alias.trim()).is_some()
    }

    /// 确定性报告：条目按实体字典序、别名按别名字典序。
    pub fn report(&self) -> String {
        if self.entries.is_empty() && self.aliases.is_empty() {
            return format!(
                "外链名册：空（逻辑时刻 tick={}，容量 {}，时限 {}）。",
                self.tick,
                self.capacity,
                ttl_label(self.ttl_ticks)
            );
        }
        let mut output = format!(
            "外链名册：{} 条条目（逻辑时刻 tick={}，容量 {}，时限 {}）。\n",
            self.entries.len(),
            self.tick,
            self.capacity,
            ttl_label(self.ttl_ticks)
        );
        for entry in self.entries.values() {
            output.push_str(&entry.summary(self.tick, self.ttl_ticks));
            output.push('\n');
        }
        if !self.aliases.is_empty() {
            output.push_str(&format!("同义链接：{} 条\n", self.aliases.len()));
            for (alias, target) in &self.aliases {
                output.push_str(&format!("- 「{}」→ 「{}」\n", alias, target));
            }
        }
        output.push_str(
            "提示：「外链刷新：实体」下次提问重新取回；「外链删除：实体」移除条目；\
             「同义链接：别名 = 规范实体」把推测记为别名知识。",
        );
        output
    }

    /// 单条目的详细报告（不存在时给出提示）。
    pub fn entry_report(&self, entity: &str) -> String {
        match self.entries.get(entity) {
            None => match self.aliases.get(entity) {
                Some(target) => format!(
                    "外链名册中没有条目「{}」；该名称是「{}」的同义链接。",
                    entity, target
                ),
                None => format!("外链名册中没有条目「{}」。", entity),
            },
            Some(entry) => {
                let mut output = format!("外链条目「{}」\n", entry.entity);
                output.push_str(&format!("- 来源：{}\n", entry.source_uri));
                output.push_str(&format!(
                    "- 取回于：第 {} 轮（逻辑时刻 tick={}）\n",
                    entry.fetched_tick, entry.fetched_tick
                ));
                output.push_str(&format!(
                    "- 状态：{}{}\n",
                    if entry.is_absent() {
                        "确无收录（负缓存）"
                    } else {
                        "有知识"
                    },
                    if entry.is_stale(self.tick, self.ttl_ticks) {
                        "（已过期，可显式刷新）"
                    } else {
                        ""
                    }
                ));
                if !entry.description.is_empty() {
                    output.push_str(&format!("- 描述：{}\n", entry.description));
                }
                if !entry.avp.is_empty() {
                    output.push_str("- 属性：\n");
                    for pair in &entry.avp {
                        if pair.len() >= 2 {
                            output.push_str(&format!("  - {}: {}\n", pair[0], pair[1]));
                        }
                    }
                }
                if !entry.tags.is_empty() {
                    output.push_str(&format!("- 标签：{}\n", entry.tags.join("、")));
                }
                output
            }
        }
    }

    /// 以 `ContextImage` 信封原子落盘（分节哈希 + 原子写）。
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let entries = serde_json::to_value(self.entries.values().collect::<Vec<_>>())
            .map_err(std::io::Error::other)?;
        let aliases = serde_json::to_value(self.aliases.iter().collect::<Vec<_>>())
            .map_err(std::io::Error::other)?;
        let mut image = ContextImage::new();
        image.set_section(LEDGER_SECTION, entries);
        image.set_section("external_ledger_aliases", aliases);
        image.set_section(
            "external_ledger_meta",
            serde_json::json!({ "tick": self.tick }),
        );
        image.seal();
        image.save_atomic(path)
    }

    /// 从磁盘恢复（分节自检失败时整册拒绝、保持原状）。
    pub fn load_into(&mut self, path: &str) -> LedgerLoad {
        if !std::path::Path::new(path).exists() {
            return LedgerLoad::Missing;
        }
        let image = match ContextImage::load(path) {
            Ok(image) => image,
            Err(error) => return LedgerLoad::Damaged(error),
        };
        let audit = image.verify();
        if audit.global_damaged {
            return LedgerLoad::Damaged("名册全局哈希不匹配".to_string());
        }
        if audit
            .damaged_sections
            .iter()
            .any(|section| section == LEDGER_SECTION)
        {
            return LedgerLoad::Damaged("名册分节哈希不匹配".to_string());
        }
        let Some(entries_value) = image.section(LEDGER_SECTION) else {
            return LedgerLoad::Loaded(0);
        };
        let entries: Vec<LedgerEntry> = match serde_json::from_value(entries_value.clone()) {
            Ok(entries) => entries,
            Err(error) => return LedgerLoad::Damaged(format!("名册解析失败：{}", error)),
        };
        let aliases: Vec<(String, String)> = image
            .section("external_ledger_aliases")
            .and_then(|value| serde_json::from_value(value.clone()).ok())
            .unwrap_or_default();
        let restored_tick = image
            .section("external_ledger_meta")
            .and_then(|value| value.get("tick"))
            .and_then(|value| value.as_u64())
            .unwrap_or(0);

        // 恢复也是插入：按 (取回时刻, 实体名) 升序装载，超出容量者淘汰最旧，
        // 保证恢复后的名册与「一路自然缓存而来」的结果逐字节一致。
        let mut ordered = entries;
        ordered.sort_by(|a, b| {
            a.fetched_tick
                .cmp(&b.fetched_tick)
                .then_with(|| a.entity.cmp(&b.entity))
        });
        self.entries.clear();
        self.order.clear();
        for entry in ordered {
            if entry.entity.trim().is_empty() {
                continue;
            }
            self.insert(entry);
        }
        for (alias, target) in aliases {
            if !alias.trim().is_empty() && !target.trim().is_empty() && alias != target {
                self.aliases.insert(alias, target);
            }
        }
        self.tick = self.tick.max(restored_tick);
        LedgerLoad::Loaded(self.entries.len())
    }

    fn insert(&mut self, entry: LedgerEntry) {
        let entity = entry.entity.clone();
        self.entries.insert(entity.clone(), entry);
        self.touch(&entity);
        while self.entries.len() > self.capacity {
            if self.order.is_empty() {
                break;
            }
            let oldest = self.order.remove(0);
            if self.entries.remove(&oldest).is_some() {
                self.perf.bump("external_ledger_evict");
            }
        }
    }

    fn remove_entry(&mut self, entity: &str) -> bool {
        self.order.retain(|name| name != entity);
        self.entries.remove(entity).is_some()
    }

    fn touch(&mut self, entity: &str) {
        self.order.retain(|name| name != entity);
        self.order.push(entity.to_string());
    }
}

fn ttl_label(ttl_ticks: u64) -> String {
    if ttl_ticks == 0 {
        "不过期".to_string()
    } else {
        format!("{} 轮", ttl_ticks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ledger_with_capacity(capacity: usize) -> ExternalLedger {
        ExternalLedger::with_config(capacity, 0, Arc::new(PerfLedger::new()))
    }

    #[test]
    fn store_lookup_roundtrip_with_provenance() {
        let mut ledger = ledger_with_capacity(8);
        ledger.advance_tick();
        ledger.advance_tick();
        ledger.store_entity(
            "法国",
            "ownthink",
            "西欧国家".to_string(),
            vec![vec!["首都".to_string(), "巴黎".to_string()]],
            vec!["国家".to_string()],
        );
        assert_eq!(ledger.len(), 1);
        assert_eq!(ledger.tick(), 2);
        match ledger.lookup("法国") {
            LedgerLookup::Fresh(entry) => {
                assert_eq!(entry.fetched_tick, 2);
                assert_eq!(entry.source_uri, "ownthink");
                assert_eq!(entry.avp[0][1], "巴黎");
            }
            other => panic!("应命中新鲜条目：{:?}", other),
        }
        assert_eq!(ledger.perf.count("external_ledger_store"), 1, "写入计数");
        assert_eq!(ledger.perf.count("external_ledger_hit"), 1);
    }

    #[test]
    fn negative_cache_is_remembered() {
        let mut ledger = ledger_with_capacity(8);
        ledger.advance_tick();
        ledger.store_absent("不存在的实体", "ownthink");
        match ledger.lookup("不存在的实体") {
            LedgerLookup::Fresh(entry) => {
                assert!(entry.is_absent());
            }
            other => panic!("负缓存应命中：{:?}", other),
        }
        assert_eq!(ledger.perf.count("external_ledger_store_absent"), 1);
        assert_eq!(ledger.perf.count("external_ledger_absent_hit"), 1);
        assert!(ledger.report().contains("确无收录"));
    }

    #[test]
    fn ttl_expires_with_logical_ticks() {
        let mut ledger = ExternalLedger::with_config(8, 2, Arc::new(PerfLedger::new()));
        ledger.advance_tick(); // tick = 1
        ledger.store_entity("水", "ownthink", String::new(), Vec::new(), Vec::new());
        assert!(matches!(ledger.lookup("水"), LedgerLookup::Fresh(_)));
        ledger.advance_tick(); // tick = 2：age=1 < 2，仍有效
        assert!(matches!(ledger.lookup("水"), LedgerLookup::Fresh(_)));
        ledger.advance_tick(); // tick = 3：age=2 >= 2，过期
        assert!(matches!(ledger.lookup("水"), LedgerLookup::Stale(_)));
        assert_eq!(ledger.perf.count("external_ledger_stale_hit"), 1);
    }

    #[test]
    fn lru_evicts_oldest_and_touches_recent() {
        let mut ledger = ledger_with_capacity(2);
        ledger.store_entity("a", "src", String::new(), Vec::new(), Vec::new());
        ledger.store_entity("b", "src", String::new(), Vec::new(), Vec::new());
        // 触碰 a，使其成为最近使用；再写入 c 应淘汰 b
        assert!(matches!(ledger.lookup("a"), LedgerLookup::Fresh(_)));
        ledger.store_entity("c", "src", String::new(), Vec::new(), Vec::new());
        assert!(ledger.entry("a").is_some());
        assert!(ledger.entry("b").is_none(), "最久未用的 b 应被淘汰");
        assert!(ledger.entry("c").is_some());
        assert_eq!(ledger.perf.count("external_ledger_evict"), 1);
    }

    #[test]
    fn alias_resolution_and_forget() {
        let mut ledger = ledger_with_capacity(8);
        ledger.store_entity(
            "ownthink:姚明",
            "ownthink",
            String::new(),
            Vec::new(),
            Vec::new(),
        );
        assert!(ledger.confirm_alias("姚明", "ownthink:姚明"));
        assert!(!ledger.confirm_alias("姚明", "姚明"), "同名别名不成立");
        assert_eq!(ledger.resolve("姚明"), "ownthink:姚明");
        assert_eq!(ledger.resolve("姚明"), "ownthink:姚明");
        assert_eq!(ledger.perf.count("external_ledger_alias_hit"), 2);
        assert!(ledger.unlink_alias("姚明"));
        assert_eq!(ledger.resolve("姚明"), "姚明");
    }

    #[test]
    fn refresh_and_forget_are_explicit() {
        let mut ledger = ledger_with_capacity(8);
        ledger.store_entity("法国", "ownthink", String::new(), Vec::new(), Vec::new());
        assert!(ledger.refresh("法国"));
        assert!(ledger.entry("法国").is_none());
        assert!(!ledger.refresh("法国"), "已删除的条目再刷新返回 false");
        ledger.store_entity("法国", "ownthink", String::new(), Vec::new(), Vec::new());
        assert!(ledger.forget("法国"));
        assert_eq!(ledger.perf.count("external_ledger_refresh"), 1);
        assert_eq!(ledger.perf.count("external_ledger_forget"), 1);
    }

    #[test]
    fn report_is_deterministic_and_sorted() {
        let mut ledger = ledger_with_capacity(8);
        ledger.advance_tick();
        ledger.store_entity("b", "src", "描述".to_string(), Vec::new(), Vec::new());
        ledger.store_entity(
            "a",
            "src",
            String::new(),
            Vec::new(),
            vec!["标签".to_string()],
        );
        ledger.confirm_alias("别名", "a");
        let report = ledger.report();
        let first = report.find("「a」").unwrap();
        let second = report.find("「b」").unwrap();
        assert!(first < second, "条目必须按实体字典序：{}", report);
        assert!(report.contains("同义链接：1 条"));
        assert!(report.contains("「别名」→ 「a」"));
        // 纯展示不应改变统计
        let before = ledger.perf.snapshot();
        let _ = ledger.report();
        assert_eq!(before, ledger.perf.snapshot());
    }

    #[test]
    fn persistence_roundtrip_and_tamper_rejection() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("external_ledger.json");
        let path_text = path.to_str().unwrap();

        assert!(matches!(
            ledger_with_capacity(8).load_into(path_text),
            LedgerLoad::Missing
        ));

        let mut ledger = ExternalLedger::with_config(8, 0, Arc::new(PerfLedger::new()));
        ledger.advance_tick();
        ledger.advance_tick();
        ledger.advance_tick();
        ledger.store_entity(
            "法国",
            "ownthink",
            "西欧国家".to_string(),
            vec![vec!["首都".to_string(), "巴黎".to_string()]],
            Vec::new(),
        );
        ledger.store_absent("不存在", "ownthink");
        ledger.confirm_alias("法兰西", "法国");
        ledger.save(path_text).unwrap();

        let mut restored = ExternalLedger::with_config(8, 0, Arc::new(PerfLedger::new()));
        match restored.load_into(path_text) {
            LedgerLoad::Loaded(count) => assert_eq!(count, 2),
            other => panic!("应能恢复名册：{:?}", other),
        }
        assert_eq!(restored.tick(), 3, "逻辑时刻应随册恢复");
        assert_eq!(restored.resolve("法兰西"), "法国");
        let entry = restored.entry("法国").expect("应恢复法国条目");
        assert_eq!(entry.fetched_tick, 3);
        assert_eq!(entry.avp[0][1], "巴黎");
        assert!(restored.entry("不存在").unwrap().is_absent());

        // 篡改条目内容 → 整册拒绝
        let mut value: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path_text).unwrap()).unwrap();
        value["sections"][LEDGER_SECTION][0]["source_uri"] = serde_json::json!("篡改");
        std::fs::write(path_text, value.to_string()).unwrap();
        let mut damaged = ExternalLedger::with_config(8, 0, Arc::new(PerfLedger::new()));
        match damaged.load_into(path_text) {
            LedgerLoad::Damaged(_) => {}
            other => panic!("篡改应被拒绝：{:?}", other),
        }
        assert!(damaged.is_empty(), "拒绝后保持空册");
    }

    #[test]
    fn restore_respects_capacity_by_tick() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("external_ledger.json");
        let path_text = path.to_str().unwrap();

        let mut ledger = ExternalLedger::with_config(8, 0, Arc::new(PerfLedger::new()));
        for entity in ["old", "mid", "new"] {
            ledger.advance_tick();
            ledger.store_entity(entity, "src", String::new(), Vec::new(), Vec::new());
        }
        ledger.save(path_text).unwrap();

        // 容量缩水到 2：恢复时按取回时刻淘汰最久未使用项
        let mut small = ExternalLedger::with_config(2, 0, Arc::new(PerfLedger::new()));
        let LedgerLoad::Loaded(count) = small.load_into(path_text) else {
            panic!("应能恢复");
        };
        assert_eq!(count, 2);
        assert!(small.entry("old").is_none());
        assert!(small.entry("mid").is_some());
        assert!(small.entry("new").is_some());
    }
}

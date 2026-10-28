//! 按知识增量逐条复核的缓存（Delta-Qualified Cache）。
//!
//! [`crate::cache::RevisionLru`] 用全局纪元回答「世界是否变过」；本模块回答
//! 「**这次变化是否碰到这条缓存的依据**」。每个条目携带一张合格证：
//!
//!   - `support_entities` / `support_predicates` / `support_facts`：计算时实际
//!     读过的名称、谓词与事实；
//!   - `patterns`：哪些新事实可能影响条目（例如查询词面是实体名的子串）；
//!   - `closure`：多跳查询的种子与已扩展节点；
//!   - `monotone`：新增事实是否只可能补充答案，还是可能改变答案；
//!   - `exhaustive`：答案是穷尽枚举还是只取了一个见证。
//!
//! 增量到达时逐条判定：无关条目只推进版本（保留计算），相关条目按单调性
//! 标记为「可能不完整」或直接失效；撤回/重置等无法用增量表达的变化仍走
//! 全量清空。遍历顺序由条目 LRU 序决定，判定结果可复现、可审计。

use std::collections::{BTreeSet, HashMap};

use crate::knowledge_delta::{fingerprint_facts, KnowledgeChange};
use crate::prolog_engine::PrologFact;

/// 新事实是否只可能补充答案。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Monotonicity {
    /// 追加事实不会推翻旧答案，只可能让答案更完整。
    Additive,
    /// 追加事实可能改变取值、计数或排序，旧答案必须重算。
    Sensitive,
}

/// 多跳查询的闭包足迹：种子节点与已经扩展到的节点。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ClosureFootprint {
    pub seeds: BTreeSet<String>,
    pub expanded: BTreeSet<String>,
}

impl ClosureFootprint {
    pub fn new(seeds: impl IntoIterator<Item = String>) -> Self {
        Self {
            seeds: seeds.into_iter().collect(),
            expanded: BTreeSet::new(),
        }
    }

    pub fn with_expanded(mut self, expanded: impl IntoIterator<Item = String>) -> Self {
        self.expanded.extend(expanded);
        self
    }

    fn contains(&self, name: &str) -> bool {
        self.seeds.contains(name) || self.expanded.contains(name)
    }
}

/// 缓存条目关心的新事实形状。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FactPattern {
    /// 任意变化都会影响（保守退化）。
    Any,
    /// 精确实体名。
    Entity(String),
    /// 实体名包含给定词面（对应按名称模糊匹配的查询）。
    EntityContains(String),
    /// 精确谓词名。
    Predicate(String),
    /// 关系形状；`None` 表示该位置任意。
    Relation {
        from: Option<String>,
        predicate: Option<String>,
        to: Option<String>,
    },
}

impl FactPattern {
    fn matches_fact(&self, fact: &PrologFact) -> bool {
        match (self, fact) {
            (FactPattern::Any, _) => true,
            (FactPattern::Entity(name), PrologFact::Entity { name: actual, .. }) => actual == name,
            (FactPattern::Entity(name), PrologFact::Relation { from, to, .. }) => {
                from == name || to == name
            }
            (FactPattern::EntityContains(needle), PrologFact::Entity { name, .. }) => {
                contains_fold(name, needle)
            }
            (FactPattern::EntityContains(needle), PrologFact::Relation { from, to, .. }) => {
                contains_fold(from, needle) || contains_fold(to, needle)
            }
            (FactPattern::Predicate(predicate), PrologFact::Relation { rel_type, .. }) => {
                rel_type == predicate
            }
            (FactPattern::Predicate(predicate), PrologFact::Entity { properties, .. }) => {
                properties.contains_key(predicate)
            }
            (
                FactPattern::Relation {
                    from,
                    predicate,
                    to,
                },
                PrologFact::Relation {
                    from: actual_from,
                    rel_type: actual_predicate,
                    to: actual_to,
                },
            ) => {
                from.as_ref().map_or(true, |value| value == actual_from)
                    && predicate
                        .as_ref()
                        .map_or(true, |value| value == actual_predicate)
                    && to.as_ref().map_or(true, |value| value == actual_to)
            }
            (FactPattern::Relation { .. }, PrologFact::Entity { .. }) => false,
        }
    }

    fn matches_name(&self, name: &str) -> bool {
        match self {
            FactPattern::Any => true,
            FactPattern::Entity(expected) => expected == name,
            FactPattern::EntityContains(needle) => contains_fold(name, needle),
            FactPattern::Predicate(_) | FactPattern::Relation { .. } => false,
        }
    }

    fn matches_predicate(&self, predicate: &str) -> bool {
        match self {
            FactPattern::Any => true,
            FactPattern::Predicate(expected) => expected == predicate,
            FactPattern::Relation {
                predicate: Some(expected),
                ..
            } => expected == predicate,
            _ => false,
        }
    }
}

fn contains_fold(haystack: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return true;
    }
    if haystack.contains(needle) {
        return true;
    }
    haystack
        .to_ascii_lowercase()
        .contains(&needle.to_ascii_lowercase())
}

/// 一次知识增量的可比较足迹。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeltaFootprint {
    pub generation: u64,
    /// 涉及的实体名、名称字面量或 IRI 本地名。
    pub names: BTreeSet<String>,
    /// 涉及的谓词本地名。
    pub predicates: BTreeSet<String>,
    /// 新增事实（用于精确形状匹配）。
    pub facts: Vec<PrologFact>,
    /// 别名变化；别名不做「可能不完整」处理，触碰即失效。
    pub aliases: Vec<(String, String)>,
    /// 无法用增量表达的整体重建（撤回、替换、结构迁移）。
    pub rebuild: bool,
}

impl DeltaFootprint {
    pub fn new(generation: u64) -> Self {
        Self {
            generation,
            names: BTreeSet::new(),
            predicates: BTreeSet::new(),
            facts: Vec::new(),
            aliases: Vec::new(),
            rebuild: false,
        }
    }

    pub fn rebuild(generation: u64) -> Self {
        Self {
            rebuild: true,
            ..Self::new(generation)
        }
    }

    pub fn from_facts(generation: u64, facts: &[PrologFact]) -> Self {
        let mut footprint = Self::new(generation);
        footprint.extend_facts(facts);
        footprint
    }

    pub fn from_change(change: &KnowledgeChange) -> Self {
        match change {
            KnowledgeChange::Facts {
                generation, facts, ..
            } => Self::from_facts(*generation, facts),
            KnowledgeChange::Aliases {
                generation,
                aliases,
                ..
            } => {
                let mut footprint = Self::new(*generation);
                footprint.aliases = aliases.clone();
                for (surface, canonical) in aliases {
                    footprint.names.insert(surface.clone());
                    footprint.names.insert(canonical.clone());
                }
                footprint
            }
            KnowledgeChange::Reset { generation, .. } => Self::rebuild(*generation),
        }
    }

    pub fn extend_facts(&mut self, facts: &[PrologFact]) {
        for fact in facts {
            match fact {
                PrologFact::Entity {
                    name,
                    entity_type,
                    properties,
                } => {
                    self.names.insert(name.clone());
                    self.names.insert(entity_type.clone());
                    for (key, value) in properties {
                        self.predicates.insert(key.clone());
                        self.names.insert(value.clone());
                    }
                }
                PrologFact::Relation { from, rel_type, to } => {
                    self.names.insert(from.clone());
                    self.names.insert(to.clone());
                    self.predicates.insert(rel_type.clone());
                }
            }
            self.facts.push(fact.clone());
        }
    }

    pub fn touch_name(&mut self, name: impl Into<String>) {
        self.names.insert(name.into());
    }

    pub fn touch_predicate(&mut self, predicate: impl Into<String>) {
        self.predicates.insert(predicate.into());
    }

    pub fn is_empty(&self) -> bool {
        !self.rebuild
            && self.names.is_empty()
            && self.predicates.is_empty()
            && self.facts.is_empty()
            && self.aliases.is_empty()
    }
}

/// 一条缓存条目的合格证。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qualification {
    pub generation: u64,
    pub support_entities: BTreeSet<String>,
    pub support_predicates: BTreeSet<String>,
    pub support_facts: BTreeSet<String>,
    pub patterns: Vec<FactPattern>,
    pub closure: ClosureFootprint,
    pub monotone: Monotonicity,
    pub exhaustive: bool,
}

impl Qualification {
    pub fn new(generation: u64) -> Self {
        Self {
            generation,
            support_entities: BTreeSet::new(),
            support_predicates: BTreeSet::new(),
            support_facts: BTreeSet::new(),
            patterns: Vec::new(),
            closure: ClosureFootprint::default(),
            monotone: Monotonicity::Sensitive,
            exhaustive: true,
        }
    }

    pub fn additive(generation: u64) -> Self {
        Self {
            monotone: Monotonicity::Additive,
            ..Self::new(generation)
        }
    }

    pub fn sensitive(generation: u64) -> Self {
        Self::new(generation)
    }

    pub fn support_entities(mut self, names: impl IntoIterator<Item = String>) -> Self {
        self.support_entities.extend(names);
        self
    }

    pub fn support_predicates(mut self, predicates: impl IntoIterator<Item = String>) -> Self {
        self.support_predicates.extend(predicates);
        self
    }

    pub fn support_facts(mut self, facts: &[PrologFact]) -> Self {
        for fact in facts {
            self.support_facts.insert(fact_key(fact));
        }
        self
    }

    pub fn watch(mut self, pattern: FactPattern) -> Self {
        self.patterns.push(pattern);
        self
    }

    pub fn closure(mut self, closure: ClosureFootprint) -> Self {
        self.closure = closure;
        self
    }

    pub fn exhaustive(mut self, exhaustive: bool) -> Self {
        self.exhaustive = exhaustive;
        self
    }

    fn touched_by(&self, delta: &DeltaFootprint) -> bool {
        if delta.rebuild {
            return true;
        }
        if self
            .patterns
            .iter()
            .any(|pattern| matches!(pattern, FactPattern::Any))
        {
            return true;
        }
        if delta
            .names
            .iter()
            .any(|name| self.support_entities.contains(name) || self.closure.contains(name))
        {
            return true;
        }
        if delta
            .predicates
            .iter()
            .any(|predicate| self.support_predicates.contains(predicate))
        {
            return true;
        }
        if delta.facts.iter().any(|fact| {
            self.support_facts.contains(&fact_key(fact))
                || self.patterns.iter().any(|pattern| {
                    pattern.matches_fact(fact)
                        || fact_names(fact)
                            .into_iter()
                            .any(|name| pattern.matches_name(&name))
                })
                || fact_predicates(fact).into_iter().any(|predicate| {
                    self.patterns
                        .iter()
                        .any(|pattern| pattern.matches_predicate(&predicate))
                })
        }) {
            return true;
        }
        delta.names.iter().any(|name| {
            self.patterns
                .iter()
                .any(|pattern| pattern.matches_name(name))
        }) || delta.predicates.iter().any(|predicate| {
            self.patterns
                .iter()
                .any(|pattern| pattern.matches_predicate(predicate))
        })
    }

    fn touched_by_alias(&self, delta: &DeltaFootprint) -> bool {
        delta.aliases.iter().any(|(surface, canonical)| {
            self.support_entities.contains(surface)
                || self.support_entities.contains(canonical)
                || self.closure.contains(surface)
                || self.closure.contains(canonical)
                || self
                    .patterns
                    .iter()
                    .any(|pattern| pattern.matches_name(surface) || pattern.matches_name(canonical))
        })
    }
}

/// 增量对单个条目的影响。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Impact {
    Unrelated,
    Supplement,
    Invalidate,
}

fn qualification_impact(qualification: &Qualification, delta: &DeltaFootprint) -> Impact {
    if delta.rebuild {
        return Impact::Invalidate;
    }
    if !delta.aliases.is_empty() && qualification.touched_by_alias(delta) {
        return Impact::Invalidate;
    }
    if qualification.touched_by(delta) {
        return match qualification.monotone {
            Monotonicity::Additive => Impact::Supplement,
            Monotonicity::Sensitive => Impact::Invalidate,
        };
    }
    Impact::Unrelated
}

/// 条目状态：完整答案，或「仍然成立但可能被新知识补充」。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryStatus {
    Complete,
    MayBeIncomplete,
}

impl EntryStatus {
    pub fn label(self) -> &'static str {
        match self {
            EntryStatus::Complete => "complete",
            EntryStatus::MayBeIncomplete => "may_be_incomplete",
        }
    }
}

/// 一次命中：值与合格证一起返回，调用方自行决定是否接受不完整状态。
#[derive(Debug, Clone, Copy)]
pub struct QualifiedRef<'a, V> {
    pub value: &'a V,
    pub status: EntryStatus,
    pub qualification: &'a Qualification,
}

/// 判定理由：用于报告与测试，不参与哈希。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reason {
    Unrelated,
    AdditiveSupplement,
    TouchesSupport,
    GenerationGap,
    Rebuild,
}

impl Reason {
    pub fn label(self) -> &'static str {
        match self {
            Reason::Unrelated => "unrelated",
            Reason::AdditiveSupplement => "additive_supplement",
            Reason::TouchesSupport => "touches_support",
            Reason::GenerationGap => "generation_gap",
            Reason::Rebuild => "rebuild",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionKind {
    Retained,
    MarkedIncomplete,
    Invalidated,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RevalidationDecision {
    pub key: String,
    pub decision: DecisionKind,
    pub reason: Reason,
}

/// 一次增量复核的确定性报告。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RevalidationReport {
    pub generation: u64,
    pub retained: usize,
    pub marked_incomplete: usize,
    pub invalidated: usize,
    pub cleared_all: bool,
    pub gap: bool,
    pub decisions: Vec<RevalidationDecision>,
}

impl RevalidationReport {
    pub fn kept(&self) -> usize {
        self.retained + self.marked_incomplete
    }
}

struct Entry<V> {
    value: V,
    qualification: Qualification,
    status: EntryStatus,
}

/// 有界、按增量逐条复核的 LRU 缓存。
pub struct DeltaQualifiedCache<V> {
    generation: Option<u64>,
    capacity: usize,
    entries: HashMap<String, Entry<V>>,
    order: Vec<String>,
}

impl<V> DeltaQualifiedCache<V> {
    pub fn new(capacity: usize) -> Self {
        Self {
            generation: None,
            capacity: capacity.max(1),
            entries: HashMap::new(),
            order: Vec::new(),
        }
    }

    pub fn generation(&self) -> Option<u64> {
        self.generation
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn get(&mut self, generation: u64, key: &str) -> Option<&V> {
        self.get_qualified(generation, key)
            .filter(|found| found.status == EntryStatus::Complete)
            .map(|found| found.value)
    }

    pub fn get_qualified(&mut self, generation: u64, key: &str) -> Option<QualifiedRef<'_, V>> {
        if self.generation != Some(generation) {
            return None;
        }
        self.touch(key)?;
        let entry = self.entries.get(key)?;
        Some(QualifiedRef {
            value: &entry.value,
            status: entry.status,
            qualification: &entry.qualification,
        })
    }

    pub fn status(&self, key: &str) -> Option<EntryStatus> {
        self.entries.get(key).map(|entry| entry.status)
    }

    pub fn put(&mut self, key: String, value: V, qualification: Qualification) -> usize {
        if self.generation != Some(qualification.generation) {
            self.entries.clear();
            self.order.clear();
            self.generation = Some(qualification.generation);
        }
        let replaced = self.entries.insert(
            key.clone(),
            Entry {
                value,
                qualification,
                status: EntryStatus::Complete,
            },
        );
        self.order.retain(|cached| cached != &key);
        self.order.push(key);
        if replaced.is_none() && self.entries.len() > self.capacity {
            if let Some(oldest) = self.order.first().cloned() {
                self.order.remove(0);
                self.entries.remove(&oldest);
                return 1;
            }
        }
        0
    }

    pub fn observe(&mut self, delta: &DeltaFootprint) -> RevalidationReport {
        let mut report = RevalidationReport {
            generation: delta.generation,
            ..RevalidationReport::default()
        };
        if delta.rebuild {
            report.cleared_all = !self.entries.is_empty();
            report.invalidated = self.entries.len();
            report.decisions = self
                .order
                .iter()
                .filter(|key| self.entries.contains_key(*key))
                .map(|key| RevalidationDecision {
                    key: key.clone(),
                    decision: DecisionKind::Invalidated,
                    reason: Reason::Rebuild,
                })
                .collect();
            self.clear_with_generation(delta.generation);
            return report;
        }
        match self.generation {
            Some(current) if delta.generation <= current => {
                return report;
            }
            Some(current) if delta.generation != current.saturating_add(1) => {
                report.cleared_all = !self.entries.is_empty();
                report.gap = true;
                report.invalidated = self.entries.len();
                report.decisions = self
                    .order
                    .iter()
                    .filter(|key| self.entries.contains_key(*key))
                    .map(|key| RevalidationDecision {
                        key: key.clone(),
                        decision: DecisionKind::Invalidated,
                        reason: Reason::GenerationGap,
                    })
                    .collect();
                self.clear_with_generation(delta.generation);
                return report;
            }
            _ => {}
        }

        let keys: Vec<String> = self.order.clone();
        for key in keys {
            let impact = self
                .entries
                .get(&key)
                .map(|entry| qualification_impact(&entry.qualification, delta));
            let Some(impact) = impact else {
                continue;
            };
            let outcome = match impact {
                Impact::Unrelated => {
                    report.retained += 1;
                    (DecisionKind::Retained, Reason::Unrelated)
                }
                Impact::Supplement => {
                    if let Some(entry) = self.entries.get_mut(&key) {
                        entry.status = EntryStatus::MayBeIncomplete;
                    }
                    report.marked_incomplete += 1;
                    (DecisionKind::MarkedIncomplete, Reason::AdditiveSupplement)
                }
                Impact::Invalidate => {
                    self.entries.remove(&key);
                    report.invalidated += 1;
                    (DecisionKind::Invalidated, Reason::TouchesSupport)
                }
            };
            if let Some(entry) = self.entries.get_mut(&key) {
                entry.qualification.generation = delta.generation;
            }
            report.decisions.push(RevalidationDecision {
                key,
                decision: outcome.0,
                reason: outcome.1,
            });
        }
        self.order.retain(|key| self.entries.contains_key(key));
        self.generation = Some(delta.generation);
        report
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.order.clear();
        self.generation = None;
    }

    fn clear_with_generation(&mut self, generation: u64) {
        self.entries.clear();
        self.order.clear();
        self.generation = Some(generation);
    }

    fn touch(&mut self, key: &str) -> Option<()> {
        if !self.entries.contains_key(key) {
            return None;
        }
        self.order.retain(|cached| cached != key);
        self.order.push(key.to_string());
        Some(())
    }
}

fn fact_key(fact: &PrologFact) -> String {
    fingerprint_facts(std::slice::from_ref(fact))
}

fn fact_names(fact: &PrologFact) -> Vec<String> {
    match fact {
        PrologFact::Entity {
            name,
            entity_type,
            properties,
        } => {
            let mut names = vec![name.clone(), entity_type.clone()];
            names.extend(properties.values().cloned());
            names
        }
        PrologFact::Relation { from, to, .. } => vec![from.clone(), to.clone()],
    }
}

fn fact_predicates(fact: &PrologFact) -> Vec<String> {
    match fact {
        PrologFact::Entity { properties, .. } => properties.keys().cloned().collect(),
        PrologFact::Relation { rel_type, .. } => vec![rel_type.clone()],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prolog_engine::PrologFact;
    use std::collections::HashMap;

    fn entity(name: &str) -> PrologFact {
        PrologFact::Entity {
            name: name.to_string(),
            entity_type: "Person".to_string(),
            properties: HashMap::new(),
        }
    }

    fn relation(from: &str, rel: &str, to: &str) -> PrologFact {
        PrologFact::Relation {
            from: from.to_string(),
            rel_type: rel.to_string(),
            to: to.to_string(),
        }
    }

    fn qualification(generation: u64, names: &[&str]) -> Qualification {
        Qualification::sensitive(generation)
            .support_entities(names.iter().map(|name| name.to_string()))
    }

    #[test]
    fn unrelated_addition_retains_entry_and_rebases_generation() {
        let mut cache = DeltaQualifiedCache::new(4);
        cache.put(
            "q".to_string(),
            "answer".to_string(),
            qualification(3, &["张三"]),
        );
        let delta = DeltaFootprint::from_facts(4, &[entity("王五")]);
        let report = cache.observe(&delta);
        assert_eq!(report.retained, 1);
        assert_eq!(report.invalidated, 0);
        assert_eq!(
            report.decisions,
            vec![RevalidationDecision {
                key: "q".to_string(),
                decision: DecisionKind::Retained,
                reason: Reason::Unrelated,
            }]
        );
        assert_eq!(cache.get(4, "q"), Some(&"answer".to_string()));
        assert_eq!(cache.status("q"), Some(EntryStatus::Complete));
    }

    #[test]
    fn touched_sensitive_entry_is_invalidated() {
        let mut cache = DeltaQualifiedCache::new(4);
        cache.put("q".to_string(), 1, qualification(0, &["张三", "李四"]));
        let report = cache.observe(&DeltaFootprint::from_facts(
            1,
            &[relation("李四", "师傅", "王五")],
        ));
        assert_eq!(report.invalidated, 1);
        assert!(cache.get(1, "q").is_none());
    }

    #[test]
    fn additive_entry_survives_as_may_be_incomplete() {
        let mut cache = DeltaQualifiedCache::new(4);
        cache.put(
            "q".to_string(),
            vec!["旧答案".to_string()],
            Qualification::additive(0)
                .support_entities(["张三".to_string()])
                .exhaustive(false),
        );
        let report = cache.observe(&DeltaFootprint::from_facts(1, &[entity("张三")]));
        assert_eq!(report.marked_incomplete, 1);
        assert_eq!(report.retained, 0);
        assert!(cache.get(1, "q").is_none(), "不完整状态不能冒充完整答案");
        let found = cache.get_qualified(1, "q").expect("旧答案仍可作为基线");
        assert_eq!(found.status, EntryStatus::MayBeIncomplete);
        assert_eq!(found.value, &vec!["旧答案".to_string()]);
        assert!(!found.qualification.exhaustive);
    }

    #[test]
    fn patterns_match_substring_and_predicate_shapes() {
        let mut cache = DeltaQualifiedCache::new(4);
        cache.put(
            "entity".to_string(),
            1,
            Qualification::sensitive(0).watch(FactPattern::EntityContains("小明".to_string())),
        );
        cache.put(
            "predicate".to_string(),
            2,
            Qualification::sensitive(0).watch(FactPattern::Predicate("师傅".to_string())),
        );
        let report = cache.observe(&DeltaFootprint::from_facts(
            1,
            &[relation("王小明", "队友", "李四")],
        ));
        assert_eq!(report.invalidated, 1);
        assert!(cache.get(1, "entity").is_none());
        assert_eq!(cache.get(1, "predicate"), Some(&2));

        let report = cache.observe(&DeltaFootprint::from_facts(
            2,
            &[relation("小红", "师傅", "王五")],
        ));
        assert_eq!(report.invalidated, 1);
        assert!(cache.get(2, "predicate").is_none());
    }

    #[test]
    fn closure_nodes_are_protected_even_without_support_entities() {
        let mut cache = DeltaQualifiedCache::new(4);
        cache.put(
            "chain".to_string(),
            "张三的师傅是李四",
            Qualification::sensitive(0).closure(
                ClosureFootprint::new(["张三".to_string()]).with_expanded(["李四".to_string()]),
            ),
        );
        cache.observe(&DeltaFootprint::from_facts(1, &[entity("李四")]));
        assert!(cache.get(1, "chain").is_none());
    }

    #[test]
    fn aliases_only_invalidate_entries_they_touch() {
        let mut cache = DeltaQualifiedCache::new(4);
        cache.put("zhang".to_string(), 1, qualification(0, &["张三"]));
        cache.put("wang".to_string(), 2, qualification(0, &["王五"]));
        let mut delta = DeltaFootprint::new(1);
        delta.aliases = vec![("明哥".to_string(), "张三".to_string())];
        delta.names.extend(["明哥".to_string(), "张三".to_string()]);
        let report = cache.observe(&delta);
        assert_eq!(report.invalidated, 1);
        assert_eq!(report.retained, 1);
        assert!(cache.get(1, "zhang").is_none());
        assert_eq!(cache.get(1, "wang"), Some(&2));
    }

    #[test]
    fn generation_gap_clears_everything() {
        let mut cache = DeltaQualifiedCache::new(4);
        cache.put("q".to_string(), 1, qualification(1, &["张三"]));
        let report = cache.observe(&DeltaFootprint::from_facts(3, &[entity("王五")]));
        assert!(report.gap);
        assert_eq!(report.invalidated, 1);
        assert!(cache.is_empty());
        assert_eq!(cache.generation(), Some(3));
    }

    #[test]
    fn reset_clears_everything_with_reason() {
        let mut cache = DeltaQualifiedCache::new(4);
        cache.put("q".to_string(), 1, qualification(0, &["张三"]));
        let report = cache.observe(&DeltaFootprint::rebuild(1));
        assert!(report.cleared_all);
        assert_eq!(report.decisions[0].reason, Reason::Rebuild);
        assert!(cache.is_empty());
    }

    #[test]
    fn duplicate_generation_is_ignored() {
        let mut cache = DeltaQualifiedCache::new(4);
        cache.put("q".to_string(), 1, qualification(5, &["张三"]));
        let report = cache.observe(&DeltaFootprint::from_facts(5, &[entity("张三")]));
        assert_eq!(report.retained, 0);
        assert_eq!(report.invalidated, 0);
        assert!(cache.get(5, "q").is_some());
    }

    #[test]
    fn capacity_eviction_is_deterministic_lru() {
        let mut cache = DeltaQualifiedCache::new(2);
        cache.put("a".to_string(), 1, qualification(0, &[]));
        cache.put("b".to_string(), 2, qualification(0, &[]));
        assert_eq!(cache.get(0, "a"), Some(&1));
        assert_eq!(cache.put("c".to_string(), 3, qualification(0, &[])), 1);
        assert!(cache.get(0, "b").is_none());
        assert_eq!(cache.get(0, "a"), Some(&1));
        assert_eq!(cache.get(0, "c"), Some(&3));
    }

    #[test]
    fn knowledge_change_conversion_keeps_generation_and_names() {
        let change = KnowledgeChange::Facts {
            generation: 9,
            facts: vec![relation("张三", "师傅", "李四")],
            source: "test".to_string(),
        };
        let footprint = DeltaFootprint::from_change(&change);
        assert_eq!(footprint.generation, 9);
        assert!(footprint.names.contains("张三"));
        assert!(footprint.names.contains("李四"));
        assert!(footprint.predicates.contains("师傅"));
        assert!(!footprint.rebuild);
        assert_eq!(
            DeltaFootprint::from_change(&KnowledgeChange::Reset {
                generation: 10,
                source: "test".to_string(),
            }),
            DeltaFootprint::rebuild(10)
        );
    }

    #[test]
    fn exact_support_fact_match_invalidates() {
        let fact = relation("张三", "师傅", "李四");
        let mut cache = DeltaQualifiedCache::new(2);
        cache.put(
            "q".to_string(),
            1,
            Qualification::sensitive(0).support_facts(std::slice::from_ref(&fact)),
        );
        let report = cache.observe(&DeltaFootprint::from_facts(1, &[fact]));
        assert_eq!(report.invalidated, 1);
    }
}

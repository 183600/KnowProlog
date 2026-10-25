//! FS：分层事实快照（Fact Snapshot）——让每一轮对话看到同一个世界。
//!
//! 读侧长期存在三个互不相交的「事实」：Oxigraph 里的三元组、
//! `SymbolicReasoner` 的会话事实、`provenance` 的来源账本。它们各自正确，
//! 但回答、生成与审计可能在同一轮看到不同的世界。
//!
//! 本模块把只读事实整理成一份不可变快照：
//!
//! ```text
//! Hypothesis（假设） < Derived（规则派生） < Base（图谱） < Session（会话）
//! ```
//!
//! 同一槽位有多个取值时，按层优先、置信度、来源身份裁决；功能谓词出现
//! 同层多值时保留全部取值并显式标记冲突，而不是静默覆盖。会话更正以
//! 「压制」进入快照：压制只作用于同层或更低层，派生层不能删图谱事实。
//!
//! 快照只读、可版本化、可回放；`OxigraphManager` 只是其中一个 adapter，
//! 写入路径保持原样。

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::provenance::{FactRef, SourceId};
use crate::symbolic_generator::SymbolicGenerator;

pub const FACT_SNAPSHOT_VERSION: u32 = 1;

/// 事实所在层；`Ord` 顺序就是裁决优先级。
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum FactLayer {
    /// 临时假设世界（「如果……」），优先级最低。
    Hypothesis,
    /// 规则推导、带前提的派生事实。
    Derived,
    /// 持久化图谱（Oxigraph 默认图）。
    Base,
    /// 本轮会话中用户确认、更正或补充的事实。
    Session,
}

impl FactLayer {
    pub fn label(self) -> &'static str {
        match self {
            FactLayer::Hypothesis => "假设",
            FactLayer::Derived => "派生",
            FactLayer::Base => "图谱",
            FactLayer::Session => "会话",
        }
    }
}

/// 事实形态：类型断言、属性字面量、实体关系。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FactKind {
    Type,
    Attribute,
    Relation,
}

impl FactKind {
    pub fn label(self) -> &'static str {
        match self {
            FactKind::Type => "类型",
            FactKind::Attribute => "属性",
            FactKind::Relation => "关系",
        }
    }
}

/// 快照里的一条事实。同一事实可以有多层记录，但同一槽位同一取值只保留
/// 裁决后的最佳记录。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FactRecord {
    pub fact: FactRef,
    pub kind: FactKind,
    pub layer: FactLayer,
    pub source: SourceId,
    pub confidence_milli: u32,
    #[serde(default)]
    pub turn: Option<u64>,
    #[serde(default)]
    pub rule: Option<String>,
    #[serde(default)]
    pub premises: Vec<FactRef>,
}

impl FactRecord {
    pub fn new(
        fact: FactRef,
        kind: FactKind,
        layer: FactLayer,
        source: SourceId,
        confidence_milli: u32,
    ) -> Self {
        Self {
            fact,
            kind,
            layer,
            source,
            confidence_milli: confidence_milli.min(1000),
            turn: None,
            rule: None,
            premises: Vec::new(),
        }
    }

    pub fn base(fact: FactRef, kind: FactKind, source: SourceId) -> Self {
        Self::new(fact, kind, FactLayer::Base, source, 1000)
    }

    pub fn session(fact: FactRef, kind: FactKind, source: SourceId, turn: u64) -> Self {
        let mut record = Self::new(fact, kind, FactLayer::Session, source, 1000);
        record.turn = Some(turn);
        record
    }

    pub fn derived(
        fact: FactRef,
        kind: FactKind,
        source: SourceId,
        rule: &str,
        premises: Vec<FactRef>,
        confidence_milli: u32,
    ) -> Self {
        let mut record = Self::new(
            fact,
            kind,
            FactLayer::Derived,
            source,
            confidence_milli,
        );
        record.rule = Some(rule.to_string());
        record.premises = premises;
        record
    }
}

impl FactRecord {
    fn rank(&self) -> (u8, u32, u8, String) {
        (
            self.layer as u8,
            self.confidence_milli,
            u8::from(self.source.is_registered()),
            self.source.render(),
        )
    }

    /// 是否比另一条记录更可采信。
    pub fn outranks(&self, other: &Self) -> bool {
        self.rank() > other.rank()
    }

    pub fn display(&self) -> String {
        format!(
            "{}［{}｜{}｜{}］",
            self.fact.display(),
            self.layer.label(),
            self.kind.label(),
            self.source.render()
        )
    }
}

/// 显式压制：会话更正/否认一条事实。只压制同层或更低层的记录。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FactSuppression {
    pub fact: FactRef,
    pub layer: FactLayer,
    pub source: SourceId,
    #[serde(default)]
    pub turn: Option<u64>,
    #[serde(default)]
    pub reason: String,
}

impl FactSuppression {
    pub fn new(fact: FactRef, layer: FactLayer, source: SourceId) -> Self {
        Self {
            fact,
            layer,
            source,
            turn: None,
            reason: String::new(),
        }
    }

    pub fn with_turn(mut self, turn: u64) -> Self {
        self.turn = Some(turn);
        self
    }

    pub fn with_reason(mut self, reason: &str) -> Self {
        self.reason = reason.to_string();
        self
    }
}

/// 一个功能槽的冲突：同层出现多个取值。`resolved = false` 表示没有依据
/// 在候选之间做选择，只能确定性地挑一个展示并明确报告冲突。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FactConflict {
    pub subject: String,
    pub predicate: String,
    pub layer: FactLayer,
    pub values: Vec<String>,
    pub winner: FactRef,
    pub resolved: bool,
}

impl FactConflict {
    pub fn render(&self) -> String {
        format!(
            "{}「{}」在{}层有 {} 个取值：{}（展示：{}，{}）",
            self.subject,
            self.predicate,
            self.layer.label(),
            self.values.len(),
            self.values.join("、"),
            self.winner.value,
            if self.resolved { "已裁决" } else { "未裁决" }
        )
    }
}

/// 一个槽位的查询视图。`effective` 是功能谓词裁决后的取值记录；
/// `all` 保留快照里该槽位的全部可采信记录（含被覆盖的更低层）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlotView {
    pub subject: String,
    pub predicate: String,
    pub all: Vec<FactRecord>,
    pub effective: Vec<FactRecord>,
    pub conflict: Option<FactConflict>,
}

impl SlotView {
    pub fn values(&self) -> Vec<&str> {
        self.effective
            .iter()
            .map(|record| record.fact.value.as_str())
            .collect()
    }

    pub fn winner(&self) -> Option<&FactRecord> {
        if let Some(conflict) = &self.conflict {
            if let Some(record) = self
                .effective
                .iter()
                .find(|record| record.fact == conflict.winner)
            {
                return Some(record);
            }
        }
        self.effective.first()
    }
}

/// 不可变事实快照。构造后不再接受写入，可安全共享与回放。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FactSnapshot {
    pub version: u32,
    pub revision: u64,
    /// 规则包指纹：规则变更不应让旧派生事实「串味」。
    pub grammar_digest: String,
    pub functional: BTreeSet<String>,
    slots: BTreeMap<(String, String), Vec<FactRecord>>,
    suppressions: Vec<FactSuppression>,
    conflicts: Vec<FactConflict>,
}

impl FactSnapshot {
    pub fn empty(revision: u64) -> Self {
        Self {
            version: FACT_SNAPSHOT_VERSION,
            revision,
            grammar_digest: String::new(),
            functional: BTreeSet::new(),
            slots: BTreeMap::new(),
            suppressions: Vec::new(),
            conflicts: Vec::new(),
        }
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn grammar_digest(&self) -> &str {
        &self.grammar_digest
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    pub fn conflicts(&self) -> &[FactConflict] {
        &self.conflicts
    }

    pub fn conflict_count(&self) -> usize {
        self.conflicts.len()
    }

    pub fn is_functional(&self, predicate: &str) -> bool {
        self.functional.contains(predicate)
    }

    /// 快照中全部仍可采信的事实记录（每个槽位每个取值一条最佳记录）。
    pub fn records(&self) -> Vec<&FactRecord> {
        let mut out: Vec<&FactRecord> = self.slots.values().flatten().collect();
        out.sort_by(|left, right| left.fact.cmp(&right.fact));
        out
    }

    pub fn record_count(&self) -> usize {
        self.slots.values().map(Vec::len).sum()
    }

    /// 当前可回答的有效事实：功能谓词只保留裁决层，非功能谓词保留全部取值。
    pub fn effective_records(&self) -> Vec<&FactRecord> {
        let mut out = Vec::new();
        for records in self.slots.values() {
            out.extend(effective_records(records, self.is_functional(&records[0].fact.predicate)));
        }
        out.sort_by(|left, right| left.fact.cmp(&right.fact));
        out
    }

    pub fn values(&self, subject: &str, predicate: &str) -> Vec<String> {
        self.lookup(subject, predicate)
            .map(|view| view.values().into_iter().map(ToString::to_string).collect())
            .unwrap_or_default()
    }

    pub fn winner(&self, subject: &str, predicate: &str) -> Option<FactRecord> {
        self.lookup(subject, predicate)?.winner().cloned()
    }

    pub fn lookup(&self, subject: &str, predicate: &str) -> Option<SlotView> {
        let key = (subject.to_string(), predicate.to_string());
        let records = self.slots.get(&key)?;
        let functional = self.is_functional(predicate);
        let effective = effective_records(records, functional)
            .into_iter()
            .cloned()
            .collect();
        let conflict = self
            .conflicts
            .iter()
            .find(|conflict| conflict.subject == subject && conflict.predicate == predicate)
            .cloned();
        Some(SlotView {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            all: records.clone(),
            effective,
            conflict,
        })
    }

    pub fn subject_records(&self, subject: &str) -> Vec<&FactRecord> {
        let mut out: Vec<&FactRecord> = self
            .slots
            .iter()
            .filter(|((name, _), _)| name == subject)
            .flat_map(|(_, records)| records.iter())
            .collect();
        out.sort_by(|left, right| left.fact.cmp(&right.fact));
        out
    }

    pub fn subjects(&self) -> Vec<String> {
        let mut out: Vec<String> = self
            .slots
            .keys()
            .map(|(subject, _)| subject.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        out.dedup();
        out
    }

    pub fn suppressions(&self) -> &[FactSuppression] {
        &self.suppressions
    }

    pub fn builder(revision: u64) -> FactSnapshotBuilder {
        FactSnapshotBuilder::new(revision)
    }

    /// 把快照灌入生成器：先类型、再属性、最后关系，保持「同一世界」。
    pub fn observe_into(&self, generator: &mut SymbolicGenerator) {
        for record in self.effective_records() {
            if record.kind == FactKind::Type {
                generator.observe_entity(&record.fact.subject, &record.fact.value, &[]);
            }
        }
        for record in self.effective_records() {
            if record.kind == FactKind::Attribute {
                generator.observe_entity(
                    &record.fact.subject,
                    "",
                    &[(record.fact.predicate.clone(), record.fact.value.clone())],
                );
            }
        }
        for record in self.effective_records() {
            if record.kind == FactKind::Relation {
                generator.observe_triple(
                    &record.fact.subject,
                    &record.fact.predicate,
                    &record.fact.value,
                );
            }
        }
    }

    /// 确定性文本报告：版本、分层计数、功能冲突与压制凭证。
    pub fn render(&self) -> String {
        let mut lines = vec![format!(
            "事实快照 v{}（revision {}，规则指纹 {}）：{} 个槽位、{} 条记录、{} 个冲突",
            self.version,
            self.revision,
            if self.grammar_digest.is_empty() {
                "未登记"
            } else {
                &self.grammar_digest
            },
            self.slots.len(),
            self.record_count(),
            self.conflicts.len()
        )];
        for layer in [
            FactLayer::Session,
            FactLayer::Base,
            FactLayer::Derived,
            FactLayer::Hypothesis,
        ] {
            let count = self
                .slots
                .values()
                .flatten()
                .filter(|record| record.layer == layer)
                .count();
            if count > 0 {
                lines.push(format!("  {}层：{} 条", layer.label(), count));
            }
        }
        for conflict in &self.conflicts {
            lines.push(format!("冲突 {}", conflict.render()));
        }
        for suppression in &self.suppressions {
            lines.push(format!(
                "压制 {}［{}｜{}］{}",
                suppression.fact.display(),
                suppression.layer.label(),
                suppression.source.render(),
                if suppression.reason.is_empty() {
                    String::new()
                } else {
                    format!("：{}", suppression.reason)
                }
            ));
        }
        lines.join("\n")
    }
}

fn effective_records(records: &[FactRecord], functional: bool) -> Vec<&FactRecord> {
    if !functional {
        return records.iter().collect();
    }
    let Some(top) = records.iter().map(|record| record.layer).max() else {
        return Vec::new();
    };
    let mut out: Vec<&FactRecord> = records
        .iter()
        .filter(|record| record.layer == top)
        .collect();
    out.sort_by(|left, right| left.fact.value.cmp(&right.fact.value));
    out
}

/// 快照构造器：只负责收集、裁决与压制，不接触存储。
#[derive(Debug, Clone)]
pub struct FactSnapshotBuilder {
    revision: u64,
    grammar_digest: String,
    functional: BTreeSet<String>,
    slots: BTreeMap<(String, String), BTreeMap<String, FactRecord>>,
    suppressions: Vec<FactSuppression>,
}

impl FactSnapshotBuilder {
    pub fn new(revision: u64) -> Self {
        Self {
            revision,
            grammar_digest: String::new(),
            functional: BTreeSet::new(),
            slots: BTreeMap::new(),
            suppressions: Vec::new(),
        }
    }

    pub fn grammar_digest(&mut self, digest: &str) -> &mut Self {
        self.grammar_digest = digest.trim().to_string();
        self
    }

    pub fn mark_functional(&mut self, predicate: &str) -> &mut Self {
        let predicate = predicate.trim();
        if !predicate.is_empty() {
            self.functional.insert(predicate.to_string());
        }
        self
    }

    pub fn mark_functional_all<I, S>(&mut self, predicates: I) -> &mut Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        for predicate in predicates {
            self.mark_functional(predicate.as_ref());
        }
        self
    }

    pub fn functional_from_config(&mut self, config: &crate::SystemConfig) -> &mut Self {
        for profile in &config.predicate_profiles {
            if profile.functional {
                self.mark_functional(&profile.name);
            }
        }
        self
    }

    /// 添加一条记录；同一槽位同一取值按层/置信度/来源身份保留最佳者。
    pub fn add_record(&mut self, record: FactRecord) -> &mut Self {
        self.slots
            .entry((
                record.fact.subject.clone(),
                record.fact.predicate.clone(),
            ))
            .or_default()
            .entry(record.fact.value.clone())
            .and_modify(|existing| {
                if record.outranks(existing) {
                    *existing = record.clone();
                }
            })
            .or_insert(record);
        self
    }

    pub fn add(
        &mut self,
        fact: FactRef,
        kind: FactKind,
        layer: FactLayer,
        source: SourceId,
        confidence_milli: u32,
    ) -> &mut Self {
        self.add_record(FactRecord::new(
            fact,
            kind,
            layer,
            source,
            confidence_milli,
        ))
    }

    pub fn add_base(&mut self, fact: FactRef, kind: FactKind, source: SourceId) -> &mut Self {
        self.add_record(FactRecord::base(fact, kind, source))
    }

    pub fn add_session(
        &mut self,
        fact: FactRef,
        kind: FactKind,
        source: SourceId,
        turn: u64,
    ) -> &mut Self {
        self.add_record(FactRecord::session(fact, kind, source, turn))
    }

    pub fn suppress(&mut self, suppression: FactSuppression) -> &mut Self {
        self.suppressions.push(suppression);
        self
    }

    /// 完成快照：应用压制、裁决功能冲突。多次构造同输入结果字节一致。
    pub fn build(mut self) -> FactSnapshot {
        let mut suppressions = std::mem::take(&mut self.suppressions);
        suppressions.sort_by(|left, right| {
            left.fact
                .cmp(&right.fact)
                .then_with(|| left.layer.cmp(&right.layer))
                .then_with(|| left.source.cmp(&right.source))
        });
        for suppression in &suppressions {
            let key = (
                suppression.fact.subject.clone(),
                suppression.fact.predicate.clone(),
            );
            if let Some(values) = self.slots.get_mut(&key) {
                if let Some(record) = values.get(&suppression.fact.value) {
                    if record.layer <= suppression.layer {
                        values.remove(&suppression.fact.value);
                    }
                }
                if values.is_empty() {
                    self.slots.remove(&key);
                }
            }
        }

        let functional = self.functional;
        let mut conflicts = Vec::new();
        for ((subject, predicate), values) in &self.slots {
            if !functional.contains(predicate) || values.len() < 2 {
                continue;
            }
            let top = values.values().map(|record| record.layer).max();
            let top_values: Vec<&FactRecord> = values
                .values()
                .filter(|record| Some(record.layer) == top)
                .collect();
            if top_values.len() < 2 {
                continue;
            }
            let mut candidates: Vec<&FactRecord> = top_values;
            candidates.sort_by(|left, right| {
                right
                    .rank()
                    .cmp(&left.rank())
                    .then_with(|| left.fact.value.cmp(&right.fact.value))
            });
            let winner = candidates[0];
            let mut shown: Vec<String> = candidates
                .iter()
                .map(|record| record.fact.value.clone())
                .collect();
            shown.sort();
            conflicts.push(FactConflict {
                subject: subject.clone(),
                predicate: predicate.clone(),
                layer: winner.layer,
                values: shown,
                winner: winner.fact.clone(),
                resolved: false,
            });
        }
        conflicts.sort_by(|left, right| {
            left.subject
                .cmp(&right.subject)
                .then_with(|| left.predicate.cmp(&right.predicate))
        });

        let mut slots: BTreeMap<(String, String), Vec<FactRecord>> = BTreeMap::new();
        for (key, values) in self.slots {
            let mut records: Vec<FactRecord> = values.into_values().collect();
            records.sort_by(|left, right| left.fact.cmp(&right.fact));
            if !records.is_empty() {
                slots.insert(key, records);
            }
        }

        FactSnapshot {
            version: FACT_SNAPSHOT_VERSION,
            revision: self.revision,
            grammar_digest: self.grammar_digest,
            functional,
            slots,
            suppressions,
            conflicts,
        }
    }
}

/// 从已加载的 `PrologFact` 构造基础快照（供会话/测试使用）。
pub fn from_prolog_facts(
    revision: u64,
    facts: &[crate::prolog_engine::PrologFact],
    source: SourceId,
) -> FactSnapshot {
    let mut builder = FactSnapshotBuilder::new(revision);
    for fact in facts {
        match fact {
            crate::prolog_engine::PrologFact::Entity {
                name,
                entity_type,
                properties,
            } => {
                if !entity_type.trim().is_empty() {
                    builder.add_base(
                        FactRef::type_assertion(name, entity_type),
                        FactKind::Type,
                        source.clone(),
                    );
                }
                let mut keys: Vec<&String> = properties.keys().collect();
                keys.sort();
                for key in keys {
                    builder.add_base(
                        FactRef::attribute(name, key, &properties[key]),
                        FactKind::Attribute,
                        source.clone(),
                    );
                }
            }
            crate::prolog_engine::PrologFact::Relation { from, rel_type, to } => {
                builder.add_base(
                    FactRef::relation(from, rel_type, to),
                    FactKind::Relation,
                    source.clone(),
                );
            }
        }
    }
    builder.build()
}

/// 生成一段稳定指纹，可放进快照键（规则/配置变了就不复用旧派生事实）。
pub fn digest_lines<'a, I>(lines: I) -> String
where
    I: IntoIterator<Item = &'a str>,
{
    let mut collected: Vec<&str> = lines.into_iter().collect();
    collected.sort_unstable();
    let joined = collected.join("\n");
    crate::context_image::fnv1a_hex(joined.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provenance::{SourceId, SourceKind};
    use std::collections::HashMap;

    fn source(kind: SourceKind, locator: &str) -> SourceId {
        SourceId::new(kind, locator, locator)
    }

    fn age(value: &str) -> FactRef {
        FactRef::attribute("张三", "age", value)
    }

    #[test]
    fn session_overrides_base_for_functional_slots_without_conflict() {
        let mut builder = FactSnapshotBuilder::new(7);
        builder.mark_functional("age");
        builder.add_base(age("18"), FactKind::Attribute, source(SourceKind::Import, "kb"));
        builder.add_session(
            age("20"),
            FactKind::Attribute,
            source(SourceKind::Conversation, "turn"),
            3,
        );
        let snapshot = builder.build();
        assert_eq!(snapshot.values("张三", "age"), vec!["20"]);
        assert!(snapshot.conflicts().is_empty());
        let view = snapshot.lookup("张三", "age").unwrap();
        assert_eq!(view.all.len(), 2, "低层记录仍可审计");
        assert_eq!(view.winner().unwrap().layer, FactLayer::Session);
    }

    #[test]
    fn same_layer_functional_multi_value_is_an_explicit_conflict() {
        let mut builder = FactSnapshotBuilder::new(1);
        builder.mark_functional("age");
        builder.add_base(age("18"), FactKind::Attribute, source(SourceKind::Import, "a"));
        builder.add_base(age("19"), FactKind::Attribute, source(SourceKind::Import, "b"));
        let snapshot = builder.build();
        assert_eq!(snapshot.conflicts().len(), 1);
        assert!(!snapshot.conflicts()[0].resolved);
        assert_eq!(snapshot.values("张三", "age").len(), 2);
        let view = snapshot.lookup("张三", "age").unwrap();
        assert_eq!(
            view.winner().map(|record| record.fact.clone()),
            Some(snapshot.conflicts()[0].winner.clone()),
            "槽位展示赢家必须与冲突报告一致"
        );
    }

    #[test]
    fn hypothesis_and_derived_never_override_base() {
        let mut builder = FactSnapshotBuilder::new(1);
        builder.mark_functional("age");
        builder.add_base(age("18"), FactKind::Attribute, source(SourceKind::Import, "kb"));
        builder.add(
            age("30"),
            FactKind::Attribute,
            FactLayer::Hypothesis,
            source(SourceKind::Derived, "what-if"),
            1000,
        );
        builder.add(
            age("40"),
            FactKind::Attribute,
            FactLayer::Derived,
            source(SourceKind::Rule, "rule"),
            1000,
        );
        let snapshot = builder.build();
        assert_eq!(snapshot.values("张三", "age"), vec!["18"]);
        assert!(snapshot.conflicts().is_empty());
    }

    #[test]
    fn session_suppression_removes_base_fact_but_not_higher_layer() {
        let mut builder = FactSnapshotBuilder::new(1);
        builder.add_base(age("18"), FactKind::Attribute, source(SourceKind::Import, "kb"));
        builder.add_session(
            age("18"),
            FactKind::Attribute,
            source(SourceKind::Conversation, "turn"),
            4,
        );
        builder.suppress(
            FactSuppression::new(
                age("18"),
                FactLayer::Session,
                source(SourceKind::Manual, "correction"),
            )
            .with_turn(5)
            .with_reason("用户更正"),
        );
        let snapshot = builder.build();
        assert!(snapshot.lookup("张三", "age").is_none());
        assert_eq!(snapshot.suppressions().len(), 1);
    }

    #[test]
    fn non_functional_predicates_keep_all_values() {
        let mut builder = FactSnapshotBuilder::new(1);
        builder.add_base(
            FactRef::attribute("张三", "hobby", "围棋"),
            FactKind::Attribute,
            source(SourceKind::Import, "kb"),
        );
        builder.add_base(
            FactRef::attribute("张三", "hobby", "书法"),
            FactKind::Attribute,
            source(SourceKind::Import, "kb"),
        );
        let snapshot = builder.build();
        assert_eq!(snapshot.values("张三", "hobby"), vec!["书法", "围棋"]);
        assert!(snapshot.conflicts().is_empty());
    }

    #[test]
    fn snapshot_populates_generator_as_one_world() {
        let mut builder = FactSnapshotBuilder::new(1);
        let src = source(SourceKind::Import, "kb");
        builder.add_base(
            FactRef::type_assertion("张三", "Person"),
            FactKind::Type,
            src.clone(),
        );
        builder.add_base(age("18"), FactKind::Attribute, src.clone());
        builder.add_base(
            FactRef::relation("张三", "认识", "李四"),
            FactKind::Relation,
            src,
        );
        let snapshot = builder.build();
        let mut generator = SymbolicGenerator::new(8, 0.5);
        snapshot.observe_into(&mut generator);
        assert!(generator.is_known_entity("张三"));
        assert_eq!(generator.entity_type_of("张三"), Some("Person"));
        assert_eq!(generator.relation_values("张三", "认识"), vec!["李四"]);
        let text = generator
            .profile("张三")
            .expect("快照灌入后应可生成介绍");
        assert!(text.contains("18") || text.contains("李四"), "{}", text);
    }

    #[tokio::test]
    async fn oxigraph_adapter_keeps_fact_kinds_and_functional_marks() {
        let dir = tempfile::TempDir::new().unwrap();
        let manager = crate::OxigraphManager::new(dir.path().to_str().unwrap()).unwrap();
        assert!(manager.mark_predicate_functional("age", true));
        let mut props = HashMap::new();
        props.insert("name".to_string(), "张三".to_string());
        props.insert("age".to_string(), "18".to_string());
        manager
            .save_knowledge(vec![crate::KnowledgeItem::Entity {
                label: "Person".to_string(),
                properties: props,
            }])
            .await
            .unwrap();
        manager
            .save_knowledge(vec![crate::KnowledgeItem::Relationship {
                from_label: String::new(),
                from_id: "张三".to_string(),
                to_label: String::new(),
                to_id: "李四".to_string(),
                rel_type: "认识".to_string(),
                properties: HashMap::new(),
            }])
            .await
            .unwrap();
        let snapshot = manager.load_fact_snapshot().await.unwrap();
        assert!(snapshot.is_functional("age"));
        let kinds: BTreeMap<String, FactKind> = snapshot
            .records()
            .into_iter()
            .map(|record| (record.fact.display(), record.kind))
            .collect();
        assert_eq!(kinds.get("type(张三, Person)"), Some(&FactKind::Type));
        assert_eq!(kinds.get("age(张三, 18)"), Some(&FactKind::Attribute));
        assert_eq!(kinds.get("认识(张三, 李四)"), Some(&FactKind::Relation));
        assert_eq!(snapshot.values("张三", "age"), vec!["18"]);
    }

    #[test]
    fn digest_is_order_insensitive_and_changes_with_content() {
        let first = digest_lines(["b", "a"]);
        let second = digest_lines(["a", "b"]);
        assert_eq!(first, second);
        assert_ne!(first, digest_lines(["a", "c"]));
    }
}

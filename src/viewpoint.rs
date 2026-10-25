use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::knowledge_fusion::{
    sanitize_namespace_id, FusionLedger, NamespaceKind, NamespaceRecord, NamespaceStatus,
    ObjectKind, TrustLevel,
};
use crate::provenance::FactRef;
use crate::query_planner::{QueryPlan, QueryPlanner};
use crate::symbolic_reasoner::{DialogueSnapshot, SnapshotAtom};

pub const DEFAULT_CONTEXT_ID: &str = "default";
pub const SESSION_CONTEXT_ID: &str = "dialogue-session";
pub const HYPOTHESIS_CONTEXT_ID: &str = "dialogue-hypothesis";

const DEFAULT_NAMES: [&str; 8] = [
    "知识库",
    "知识图谱",
    "图谱",
    "数据库",
    "默认图",
    "全局视图",
    "全局",
    "default",
];
const SESSION_NAMES: [&str; 8] = [
    "会话",
    "对话",
    "本次会话",
    "当前会话",
    "这次对话",
    "刚才",
    "上一轮",
    "上次",
];
const HYPOTHESIS_NAMES: [&str; 5] = ["假设", "假想", "反事实", "假设世界", "假设场景"];
const IMPORT_NAMES: [&str; 4] = ["导入来源", "外部来源", "导入", "外部"];
const ALL_NAMES: [&str; 11] = [
    "所有来源",
    "全部来源",
    "各个来源",
    "不同来源",
    "所有说法",
    "全部说法",
    "跨来源",
    "所有语境",
    "全部语境",
    "所有命名空间",
    "各个命名空间",
];
const SOURCE_SUFFIXES: [&str; 9] = [
    "的说法", "的记录", "的资料", "所言", "来源", "说", "称", "表示", "认为",
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LensScope {
    Default,
    All,
    Kind(NamespaceKind),
    Namespace(String),
}

impl LensScope {
    pub fn label_zh(&self) -> String {
        match self {
            LensScope::Default => "知识库".to_string(),
            LensScope::All => "所有语境".to_string(),
            LensScope::Kind(kind) => match kind {
                NamespaceKind::Global => "知识库".to_string(),
                NamespaceKind::Import => "导入来源".to_string(),
                NamespaceKind::Session => "会话".to_string(),
                NamespaceKind::Hypothesis => "假设".to_string(),
            },
            LensScope::Namespace(name) => format!("「{}」的记录", name),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LensCue {
    pub scope: LensScope,
    pub surface: String,
    pub question: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LensContext {
    pub id: String,
    pub label: String,
    pub kind: NamespaceKind,
    pub trust: TrustLevel,
    pub status: NamespaceStatus,
    pub priority: i64,
}

impl LensContext {
    pub fn default_graph() -> Self {
        Self {
            id: DEFAULT_CONTEXT_ID.to_string(),
            label: "知识库".to_string(),
            kind: NamespaceKind::Global,
            trust: TrustLevel::Curated,
            status: NamespaceStatus::Active,
            priority: 0,
        }
    }

    pub fn from_record(record: &NamespaceRecord) -> Self {
        Self {
            id: record.id.clone(),
            label: record.label.clone(),
            kind: record.kind,
            trust: record.trust,
            status: record.status,
            priority: record.priority,
        }
    }

    pub fn is_default(&self) -> bool {
        self.id == DEFAULT_CONTEXT_ID
    }

    pub fn render_short(&self) -> String {
        if self.is_default() {
            return self.label.clone();
        }
        let mut text = format!("{}（{}）", self.label, self.kind.label_zh());
        match self.status {
            NamespaceStatus::Active => {}
            NamespaceStatus::Quarantined => text.push_str("，隔离"),
            NamespaceStatus::Retired => text.push_str("，退役"),
        }
        text
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LensFact {
    pub fact: FactRef,
    pub context: LensContext,
    pub source: String,
    pub object_kind: ObjectKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SlotDirection {
    Subject,
    Object,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SlotIntent {
    Value,
    Exist,
    Count,
    Check,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlotQuery {
    pub key: String,
    pub predicate: String,
    pub predicates: Vec<String>,
    pub direction: SlotDirection,
    pub intent: SlotIntent,
    pub expected: Option<String>,
}

impl SlotQuery {
    pub fn forward(subject: &str, predicate: &str) -> Self {
        Self {
            key: subject.to_string(),
            predicate: predicate.to_string(),
            predicates: vec![predicate.to_string()],
            direction: SlotDirection::Subject,
            intent: SlotIntent::Value,
            expected: None,
        }
    }

    pub fn with_predicate_aliases<I, S>(mut self, aliases: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        for alias in aliases {
            let alias = alias.into();
            if !alias.trim().is_empty() && !self.predicates.contains(&alias) {
                self.predicates.push(alias);
            }
        }
        self
    }

    pub fn slot_label(&self) -> String {
        match self.direction {
            SlotDirection::Subject => format!("{} 的 {}", self.key, self.predicate),
            SlotDirection::Object => format!("{} 为 {} 的实体", self.predicate, self.key),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SlotState {
    Supported,
    Conflict,
    Unknown,
    Unresolved,
}

impl SlotState {
    pub fn label_zh(self) -> &'static str {
        match self {
            SlotState::Supported => "有支撑",
            SlotState::Conflict => "有冲突",
            SlotState::Unknown => "未知",
            SlotState::Unresolved => "语境未解析",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValueWarrant {
    pub value: String,
    pub fact: FactRef,
    pub contexts: Vec<LensContext>,
    pub sources: Vec<String>,
    pub corroborated: bool,
}

impl ValueWarrant {
    pub fn context_labels(&self) -> String {
        self.contexts
            .iter()
            .map(LensContext::render_short)
            .collect::<Vec<_>>()
            .join("、")
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlotView {
    pub scope: LensScope,
    pub scope_label: String,
    pub scope_prefix: String,
    pub query: SlotQuery,
    pub functional: bool,
    pub facts: Vec<LensFact>,
    pub values: Vec<ValueWarrant>,
    pub elsewhere: Vec<LensFact>,
    pub unknown_context: Option<String>,
    pub selected_contexts: Vec<LensContext>,
}

impl SlotView {
    pub fn state(&self) -> SlotState {
        if self.unknown_context.is_some() {
            return SlotState::Unresolved;
        }
        if self.values.is_empty() {
            return SlotState::Unknown;
        }
        if self.functional && self.values.len() > 1 {
            return SlotState::Conflict;
        }
        SlotState::Supported
    }

    pub fn render(&self) -> String {
        self.render_for(self.query.intent, self.query.expected.as_deref())
    }

    pub fn render_for(&self, intent: SlotIntent, expected: Option<&str>) -> String {
        if let Some(error) = &self.unknown_context {
            return error.clone();
        }
        let slot = self.query.slot_label();
        let state = self.state();
        match intent {
            SlotIntent::Check => {
                if let Some(expected) = expected {
                    return self.render_check(&slot, expected);
                }
            }
            SlotIntent::Exist => {
                if state != SlotState::Unknown {
                    return format!(
                        "有。{}有关于{}的记录：{}。",
                        self.scope_prefix,
                        slot,
                        self.render_values_inline()
                    );
                }
            }
            SlotIntent::Count => {
                if state != SlotState::Unknown {
                    return format!(
                        "{}关于{}共有 {} 个取值：{}。",
                        self.scope_prefix,
                        slot,
                        self.values.len(),
                        self.render_values_inline()
                    );
                }
            }
            SlotIntent::Value => {}
        }
        self.render_value_state(&slot)
    }

    fn render_check(&self, slot: &str, expected: &str) -> String {
        if self.values.iter().any(|value| value.value == expected) {
            return format!(
                "是的。{}，{} 是 {}。（出处：{}）",
                self.scope_prefix,
                slot,
                expected,
                self.value_contexts(expected)
            );
        }
        if !self.values.is_empty() {
            if self.state() == SlotState::Conflict {
                return format!(
                    "无法判定。{}关于{}有 {} 种说法：{}。",
                    self.scope_prefix,
                    slot,
                    self.values.len(),
                    self.render_values_inline()
                );
            }
            return format!(
                "不是。{}，{} 是 {}，不是 {}。",
                self.scope_prefix,
                slot,
                self.render_values_inline(),
                expected
            );
        }
        let mut text = format!("{}没有关于{}的记录。", self.scope_prefix, slot);
        if let Some(found) = self.elsewhere.iter().find(|fact| self.display_value(fact) == expected)
        {
            text.push_str(&format!(
                "不过{}里有该说法。",
                found.context.render_short()
            ));
        } else if !self.elsewhere.is_empty() {
            text.push_str(&self.render_elsewhere());
        }
        text
    }

    fn render_value_state(&self, slot: &str) -> String {
        match self.state() {
            SlotState::Unresolved => self.unknown_context.clone().unwrap_or_default(),
            SlotState::Unknown => {
                let mut text = format!("{}没有关于{}的记录。", self.scope_prefix, slot);
                if !self.elsewhere.is_empty() {
                    text.push_str(&self.render_elsewhere());
                }
                text
            }
            SlotState::Conflict => {
                let mut lines = vec![format!(
                    "关于{}，{}有 {} 种说法（未替你做真假裁决）：",
                    slot,
                    self.scope_prefix.trim_end_matches('，'),
                    self.values.len()
                )];
                for value in &self.values {
                    lines.push(format!("  • {}（{}）", value.value, value.context_labels()));
                }
                lines.join("\n")
            }
            SlotState::Supported => {
                let mut text = if self.values.len() == 1 {
                    let value = &self.values[0];
                    format!(
                        "{}，{} 是 {}。（出处：{}）",
                        self.scope_prefix,
                        slot,
                        value.value,
                        value.context_labels()
                    )
                } else {
                    format!(
                        "{}，{} 有：{}。",
                        self.scope_prefix,
                        slot,
                        self.values
                            .iter()
                            .map(|value| format!("{}（{}）", value.value, value.context_labels()))
                            .collect::<Vec<_>>()
                            .join("；")
                    )
                };
                if !self.elsewhere.is_empty() {
                    text.push_str(&self.render_elsewhere());
                }
                text
            }
        }
    }

    fn render_values_inline(&self) -> String {
        self.values
            .iter()
            .map(|value| format!("{}（{}）", value.value, value.context_labels()))
            .collect::<Vec<_>>()
            .join("；")
    }

    fn value_contexts(&self, value: &str) -> String {
        self.values
            .iter()
            .find(|candidate| candidate.value == value)
            .map(ValueWarrant::context_labels)
            .unwrap_or_default()
    }

    fn render_elsewhere(&self) -> String {
        let mut parts = BTreeSet::new();
        for fact in &self.elsewhere {
            parts.insert(format!(
                "{}（{}）",
                self.display_value(fact),
                fact.context.render_short()
            ));
        }
        format!("其他语境里有：{}。", parts.into_iter().collect::<Vec<_>>().join("；"))
    }

    fn display_value(&self, fact: &LensFact) -> String {
        match self.query.direction {
            SlotDirection::Subject => fact.fact.value.clone(),
            SlotDirection::Object => fact.fact.subject.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextSummary {
    pub context: LensContext,
    pub facts: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LensCorpus {
    facts: Vec<LensFact>,
    contexts: Vec<LensContext>,
}

impl LensCorpus {
    pub fn new(default_facts: &[FactRef], ledger: &FusionLedger) -> Self {
        let mut contexts = vec![LensContext::default_graph()];
        let mut by_id: BTreeMap<String, LensContext> = BTreeMap::new();
        by_id.insert(DEFAULT_CONTEXT_ID.to_string(), LensContext::default_graph());
        for record in &ledger.namespaces {
            let context = LensContext::from_record(record);
            by_id.insert(context.id.clone(), context.clone());
            contexts.push(context);
        }

        let mut facts = Vec::new();
        for fact in default_facts {
            facts.push(LensFact {
                fact: fact.clone(),
                context: LensContext::default_graph(),
                source: "默认图".to_string(),
                object_kind: ObjectKind::Literal,
            });
        }
        for assertion in &ledger.assertions {
            let Some(context) = by_id.get(&assertion.namespace) else {
                continue;
            };
            facts.push(LensFact {
                fact: FactRef::new(
                    &assertion.subject,
                    &assertion.predicate,
                    &assertion.object,
                ),
                context: context.clone(),
                source: assertion.source.clone(),
                object_kind: assertion.object_kind,
            });
        }
        let mut corpus = Self { facts, contexts };
        corpus.finish();
        corpus
    }

    /// 在融合来源之外，把持久化对话状态里的会话事实与当前假设也纳入镜头。
    ///
    /// 这两层是运行时信念（不写知识图谱），但用户会自然地问「会话里呢 /
    /// 假设里呢」。`--dialogue-state` 让离线镜头查询看到与交互会话相同的
    /// 会话层和假设世界，而不是把它们误报成未知。
    pub fn with_dialogue(
        default_facts: &[FactRef],
        ledger: &FusionLedger,
        snapshot: &DialogueSnapshot,
    ) -> Self {
        let mut corpus = Self::new(default_facts, ledger);
        corpus.add_virtual_context(
            SESSION_CONTEXT_ID,
            "本次会话",
            NamespaceKind::Session,
            TrustLevel::Verified,
            NamespaceStatus::Active,
            20,
            &snapshot.session_facts,
            "会话事实",
        );
        corpus.add_virtual_context(
            HYPOTHESIS_CONTEXT_ID,
            "当前假设",
            NamespaceKind::Hypothesis,
            TrustLevel::Speculative,
            NamespaceStatus::Active,
            30,
            &snapshot.assumptions,
            "假设世界",
        );
        corpus.finish();
        corpus
    }

    fn add_virtual_context(
        &mut self,
        id: &str,
        label: &str,
        kind: NamespaceKind,
        trust: TrustLevel,
        status: NamespaceStatus,
        priority: i64,
        atoms: &[SnapshotAtom],
        source: &str,
    ) {
        let facts: Vec<LensFact> = atoms
            .iter()
            .filter_map(|atom| {
                if atom.predicate.trim().is_empty() || atom.terms.len() != 2 {
                    return None;
                }
                if atom.terms.iter().any(|term| term.trim().is_empty()) {
                    return None;
                }
                Some(LensFact {
                    fact: FactRef::new(&atom.terms[0], &atom.predicate, &atom.terms[1]),
                    context: LensContext {
                        id: id.to_string(),
                        label: label.to_string(),
                        kind,
                        trust,
                        status,
                        priority,
                    },
                    source: source.to_string(),
                    object_kind: ObjectKind::Literal,
                })
            })
            .collect();
        if facts.is_empty() {
            return;
        }
        let context = self
            .contexts
            .iter()
            .find(|context| context.id == id)
            .cloned()
            .unwrap_or_else(|| LensContext {
                id: id.to_string(),
                label: label.to_string(),
                kind,
                trust,
                status,
                priority,
            });
        if !self.contexts.iter().any(|existing| existing.id == context.id) {
            self.contexts.push(context.clone());
        }
        self.facts.extend(facts.into_iter().map(|fact| LensFact {
            context: context.clone(),
            ..fact
        }));
    }

    fn finish(&mut self) {
        self.facts
            .sort_by(|left, right| fact_sort_key(left).cmp(&fact_sort_key(right)));
        self.facts
            .dedup_by(|left, right| fact_sort_key(left) == fact_sort_key(right));
        self.contexts.sort_by(|left, right| {
            right
                .is_default()
                .cmp(&left.is_default())
                .then_with(|| left.id.cmp(&right.id))
        });
    }

    pub fn contexts(&self) -> &[LensContext] {
        &self.contexts
    }

    pub fn facts(&self) -> &[LensFact] {
        &self.facts
    }

    pub fn aliases(&self) -> Vec<ContextAlias> {
        self.contexts
            .iter()
            .filter(|context| !context.is_default())
            .flat_map(|context| {
                [
                    ContextAlias::new(&context.id, &context.label),
                    ContextAlias::new(&context.id, &context.id),
                ]
            })
            .collect()
    }

    pub fn summaries(&self) -> Vec<ContextSummary> {
        self.contexts
            .iter()
            .map(|context| ContextSummary {
                context: context.clone(),
                facts: self
                    .facts
                    .iter()
                    .filter(|fact| fact.context.id == context.id)
                    .count(),
            })
            .collect()
    }

    pub fn query_slot(
        &self,
        scope: &LensScope,
        query: &SlotQuery,
        functional: bool,
    ) -> SlotView {
        let selected_ids = match self.selected_context_ids(scope) {
            Ok(ids) => ids,
            Err(error) => {
                return SlotView {
                    scope: scope.clone(),
                    scope_label: scope.label_zh(),
                    scope_prefix: format!("在{}中，", scope.label_zh()),
                    query: query.clone(),
                    functional,
                    facts: Vec::new(),
                    values: Vec::new(),
                    elsewhere: Vec::new(),
                    unknown_context: Some(error),
                    selected_contexts: Vec::new(),
                };
            }
        };

        let mut in_scope: Vec<LensFact> = Vec::new();
        let mut elsewhere: Vec<LensFact> = Vec::new();
        for fact in &self.facts {
            if !query.predicates.iter().any(|predicate| predicate == &fact.fact.predicate) {
                continue;
            }
            let key_matches = match query.direction {
                SlotDirection::Subject => fact.fact.subject == query.key,
                SlotDirection::Object => fact.fact.value == query.key,
            };
            if !key_matches {
                continue;
            }
            if selected_ids.contains(&fact.context.id) {
                in_scope.push(fact.clone());
            } else if fact.context.status != NamespaceStatus::Retired {
                elsewhere.push(fact.clone());
            }
        }
        in_scope.sort_by(|left, right| fact_sort_key(left).cmp(&fact_sort_key(right)));
        in_scope.dedup_by(|left, right| fact_sort_key(left) == fact_sort_key(right));
        elsewhere.sort_by(|left, right| fact_sort_key(left).cmp(&fact_sort_key(right)));
        elsewhere.dedup_by(|left, right| fact_sort_key(left) == fact_sort_key(right));

        let values = group_values(&in_scope, query.direction);
        let selected_contexts = self
            .contexts
            .iter()
            .filter(|context| selected_ids.contains(&context.id))
            .cloned()
            .collect();
        let (scope_label, scope_prefix) = scope_render(scope, &self.contexts);
        SlotView {
            scope: scope.clone(),
            scope_label,
            scope_prefix,
            query: query.clone(),
            functional,
            facts: in_scope,
            values,
            elsewhere,
            unknown_context: None,
            selected_contexts,
        }
    }

    /// 列出某个镜头内的原始事实，可按主语/谓词过滤（探索式查看，不做槽位裁决）。
    pub fn list_facts(
        &self,
        scope: &LensScope,
        subject: Option<&str>,
        predicate: Option<&str>,
    ) -> Result<Vec<LensFact>, String> {
        let selected_ids = self.selected_context_ids(scope)?;
        let subject = subject.map(str::trim).filter(|value| !value.is_empty());
        let predicate = predicate
            .map(str::trim)
            .filter(|value| !value.is_empty());
        Ok(self
            .facts
            .iter()
            .filter(|fact| {
                selected_ids.contains(&fact.context.id)
                    && subject.is_none_or(|value| fact.fact.subject == value)
                    && predicate.is_none_or(|value| fact.fact.predicate == value)
            })
            .cloned()
            .collect())
    }

    fn selected_context_ids(&self, scope: &LensScope) -> Result<BTreeSet<String>, String> {
        match scope {
            LensScope::Default => Ok([DEFAULT_CONTEXT_ID.to_string()].into_iter().collect()),
            LensScope::All => Ok(self
                .contexts
                .iter()
                .filter(|context| context.status != NamespaceStatus::Retired)
                .map(|context| context.id.clone())
                .collect()),
            LensScope::Kind(kind) => Ok(self
                .contexts
                .iter()
                .filter(|context| context.kind == *kind && context.status != NamespaceStatus::Retired)
                .map(|context| context.id.clone())
                .collect()),
            LensScope::Namespace(raw) => {
                let wanted = raw.trim();
                let sanitized = sanitize_namespace_id(wanted);
                if let Some(context) = self.contexts.iter().find(|context| {
                    context.id == sanitized
                        || context.id.eq_ignore_ascii_case(wanted)
                        || context.label.eq_ignore_ascii_case(wanted)
                }) {
                    Ok([context.id.clone()].into_iter().collect())
                } else {
                    Err(format!(
                        "没有名为「{}」的语境。可用语境：{}。",
                        wanted,
                        self.contexts
                            .iter()
                            .map(LensContext::render_short)
                            .collect::<Vec<_>>()
                            .join("、")
                    ))
                }
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextAlias {
    pub id: String,
    pub label: String,
}

impl ContextAlias {
    pub fn new(id: &str, label: &str) -> Self {
        Self {
            id: id.to_string(),
            label: label.to_string(),
        }
    }
}

fn fact_sort_key(fact: &LensFact) -> (&str, &str, &str, &str, &str) {
    (
        fact.fact.subject.as_str(),
        fact.fact.predicate.as_str(),
        fact.fact.value.as_str(),
        fact.context.id.as_str(),
        fact.source.as_str(),
    )
}

fn group_values(facts: &[LensFact], direction: SlotDirection) -> Vec<ValueWarrant> {
    let mut groups: BTreeMap<String, Vec<&LensFact>> = BTreeMap::new();
    for fact in facts {
        let value = match direction {
            SlotDirection::Subject => fact.fact.value.clone(),
            SlotDirection::Object => fact.fact.subject.clone(),
        };
        groups.entry(value).or_default().push(fact);
    }
    groups
        .into_iter()
        .map(|(value, members)| {
            let mut contexts: Vec<LensContext> = Vec::new();
            let mut sources: BTreeSet<String> = BTreeSet::new();
            for member in &members {
                if !contexts.iter().any(|context| context.id == member.context.id) {
                    contexts.push(member.context.clone());
                }
                if !member.source.trim().is_empty() {
                    sources.insert(member.source.clone());
                }
            }
            contexts.sort_by(|left, right| left.id.cmp(&right.id));
            let corroborated = contexts.len() >= 2;
            ValueWarrant {
                value,
                fact: members[0].fact.clone(),
                contexts,
                sources: sources.into_iter().collect(),
                corroborated,
            }
        })
        .collect()
}

fn scope_render(scope: &LensScope, contexts: &[LensContext]) -> (String, String) {
    match scope {
        LensScope::Default => ("知识库".to_string(), "知识库中".to_string()),
        LensScope::All => ("所有语境".to_string(), "所有语境中".to_string()),
        LensScope::Kind(kind) => {
            let label = match kind {
                NamespaceKind::Global => "知识库",
                NamespaceKind::Import => "导入来源",
                NamespaceKind::Session => "会话",
                NamespaceKind::Hypothesis => "假设",
            };
            (label.to_string(), format!("{}中", label))
        }
        LensScope::Namespace(raw) => {
            let sanitized = sanitize_namespace_id(raw);
            let label = contexts
                .iter()
                .find(|context| {
                    context.id == sanitized
                        || context.id.eq_ignore_ascii_case(raw)
                        || context.label.eq_ignore_ascii_case(raw)
                })
                .map(|context| context.label.clone())
                .unwrap_or_else(|| raw.trim().to_string());
            (format!("「{}」的记录", label), format!("据「{}」的记录", label))
        }
    }
}

pub fn detect_lens_cue(question: &str, aliases: &[ContextAlias]) -> Option<LensCue> {
    let text = trim_polite_prefix(question.trim());
    if text.is_empty() {
        return None;
    }
    if let Some(cue) = match_named_scope(text, &ALL_NAMES, LensScope::All) {
        return Some(cue);
    }
    if let Some(cue) = match_named_scope(text, &DEFAULT_NAMES, LensScope::Default) {
        return Some(cue);
    }
    if let Some(cue) = match_named_scope(
        text,
        &SESSION_NAMES,
        LensScope::Kind(NamespaceKind::Session),
    ) {
        return Some(cue);
    }
    if let Some(cue) = match_named_scope(
        text,
        &HYPOTHESIS_NAMES,
        LensScope::Kind(NamespaceKind::Hypothesis),
    ) {
        return Some(cue);
    }
    if let Some(cue) = match_named_scope(
        text,
        &IMPORT_NAMES,
        LensScope::Kind(NamespaceKind::Import),
    ) {
        return Some(cue);
    }
    if let Some(cue) = match_alias_prefix(text, aliases) {
        return Some(cue);
    }
    if let Some((name, rest, surface)) = generic_source_prefix(text) {
        let scope = classify_scope_name(&name);
        return Some(LensCue {
            scope,
            surface,
            question: rest.to_string(),
        });
    }
    match_suffix_cue(text, aliases)
}

fn match_named_scope(text: &str, names: &[&str], scope: LensScope) -> Option<LensCue> {
    let mut surfaces: Vec<String> = Vec::new();
    for name in names {
        for prefix in ["", "在", "从"] {
            for suffix in ["", "里", "中", "内", "下"] {
                surfaces.push(format!("{}{}{}", prefix, name, suffix));
            }
        }
    }
    surfaces.sort_by(|left, right| right.chars().count().cmp(&left.chars().count()));
    for surface in surfaces {
        if starts_with_ignore_ascii_case(text, &surface) {
            return Some(LensCue {
                scope,
                surface: surface.clone(),
                question: strip_after_cue(&text[surface.len()..]).to_string(),
            });
        }
    }
    None
}

fn match_alias_prefix(text: &str, aliases: &[ContextAlias]) -> Option<LensCue> {
    let mut candidates: Vec<(String, String)> = Vec::new();
    for alias in aliases {
        let names = [alias.label.as_str(), alias.id.as_str()];
        for name in names {
            if name.trim().is_empty() {
                continue;
            }
            for lead in ["据", "根据", "按照", "按"] {
                for suffix in SOURCE_SUFFIXES {
                    candidates.push((
                        format!("{}{}{}", lead, name, suffix),
                        alias.id.clone(),
                    ));
                }
                candidates.push((format!("{}{}", lead, name), alias.id.clone()));
            }
            for suffix in SOURCE_SUFFIXES {
                candidates.push((format!("{}{}", name, suffix), alias.id.clone()));
            }
            for prefix in ["在", "从", ""] {
                for suffix in ["里", "中", "内", "下"] {
                    candidates.push((
                        format!("{}{}{}", prefix, name, suffix),
                        alias.id.clone(),
                    ));
                }
            }
        }
    }
    candidates.sort_by(|left, right| {
        right
            .0
            .chars()
            .count()
            .cmp(&left.0.chars().count())
            .then_with(|| left.0.cmp(&right.0))
    });
    for (surface, id) in candidates {
        if starts_with_ignore_ascii_case(text, &surface) {
            return Some(LensCue {
                scope: LensScope::Namespace(id),
                surface: surface.clone(),
                question: strip_after_cue(&text[surface.len()..]).to_string(),
            });
        }
    }
    None
}

fn generic_source_prefix(text: &str) -> Option<(String, &str, String)> {
    for lead in ["据", "根据", "按照", "按"] {
        let Some(rest) = text.strip_prefix(lead) else {
            continue;
        };
        let rest = rest.trim_start();
        for suffix in SOURCE_SUFFIXES {
            if let Some(position) = rest.find(suffix) {
                let name = rest[..position].trim();
                if valid_source_name(name) {
                    let after = &rest[position + suffix.len()..];
                    return Some((
                        name.to_string(),
                        strip_after_cue(after),
                        format!("{}{}{}", lead, name, suffix),
                    ));
                }
            }
        }
        if let Some(position) = rest.find(|ch| matches!(ch, '，' | ',' | '：' | ':')) {
            let name = rest[..position].trim();
            if valid_source_name(name) {
                let after = &rest[position..];
                return Some((
                    name.to_string(),
                    strip_after_cue(after),
                    format!("{}{}", lead, name),
                ));
            }
        }
    }
    None
}

fn valid_source_name(name: &str) -> bool {
    !name.is_empty()
        && name.chars().count() <= 32
        && !name.chars().any(|ch| {
            matches!(
                ch,
                '，' | ',' | '。' | '？' | '?' | '！' | '!' | '；' | ';' | '的'
            )
        })
}

fn match_suffix_cue(text: &str, aliases: &[ContextAlias]) -> Option<LensCue> {
    let trimmed = text.trim_end_matches(|ch: char| {
        matches!(ch, '）' | ')' | '。' | '？' | '?' | '！' | '!' | ' ')
    });
    let open = trimmed.rfind(|ch| matches!(ch, '（' | '('))?;
    let inner = &trimmed[open + 1..];
    let (name, _, _) = generic_source_prefix(inner)?;
    let id = aliases
        .iter()
        .find(|alias| {
            alias.id.eq_ignore_ascii_case(name.trim())
                || alias.label.eq_ignore_ascii_case(name.trim())
        })
        .map(|alias| alias.id.clone())
        .unwrap_or_else(|| sanitize_namespace_id(&name));
    Some(LensCue {
        scope: LensScope::Namespace(id),
        surface: trimmed[open..].to_string(),
        question: trimmed[..open].trim().to_string(),
    })
}

fn classify_scope_name(name: &str) -> LensScope {
    let trimmed = name.trim();
    let lower = trimmed.to_ascii_lowercase();
    if ALL_NAMES
        .iter()
        .any(|candidate| candidate.eq_ignore_ascii_case(&lower))
    {
        return LensScope::All;
    }
    if DEFAULT_NAMES
        .iter()
        .any(|candidate| candidate.eq_ignore_ascii_case(&lower))
    {
        return LensScope::Default;
    }
    if SESSION_NAMES
        .iter()
        .any(|candidate| candidate.eq_ignore_ascii_case(&lower))
    {
        return LensScope::Kind(NamespaceKind::Session);
    }
    if HYPOTHESIS_NAMES
        .iter()
        .any(|candidate| candidate.eq_ignore_ascii_case(&lower))
    {
        return LensScope::Kind(NamespaceKind::Hypothesis);
    }
    if IMPORT_NAMES
        .iter()
        .any(|candidate| candidate.eq_ignore_ascii_case(&lower))
    {
        return LensScope::Kind(NamespaceKind::Import);
    }
    if let Some(rest) = trimmed
        .strip_prefix("namespace:")
        .or_else(|| trimmed.strip_prefix("source:"))
    {
        return LensScope::Namespace(rest.trim().to_string());
    }
    LensScope::Namespace(trimmed.to_string())
}

pub fn parse_lens_scope(raw: &str, aliases: &[ContextAlias]) -> LensScope {
    let trimmed = raw.trim();
    let lower = trimmed.to_ascii_lowercase();
    match lower.as_str() {
        "default" | "global" | "知识库" | "默认图" | "图谱" => LensScope::Default,
        "all" | "所有" | "全部" | "所有语境" => LensScope::All,
        "session" | "会话" => LensScope::Kind(NamespaceKind::Session),
        "hypothesis" | "假设" | "反事实" => LensScope::Kind(NamespaceKind::Hypothesis),
        "import" | "导入" | "导入来源" => LensScope::Kind(NamespaceKind::Import),
        _ => {
            if let Some(rest) = trimmed
                .strip_prefix("namespace:")
                .or_else(|| trimmed.strip_prefix("source:"))
            {
                return LensScope::Namespace(rest.trim().to_string());
            }
            if let Some(alias) = aliases.iter().find(|alias| {
                alias.id.eq_ignore_ascii_case(trimmed)
                    || alias.label.eq_ignore_ascii_case(trimmed)
            }) {
                return LensScope::Namespace(alias.id.clone());
            }
            LensScope::Namespace(trimmed.to_string())
        }
    }
}

pub fn extract_slot_query(question: &str) -> Option<SlotQuery> {
    let plan = QueryPlanner::new().parse(question)?;
    match plan {
        QueryPlan::FindObject { segments } if segments.len() == 2 => {
            Some(SlotQuery::forward(&segments[0], &segments[1]))
        }
        QueryPlan::FindObjectByPredicate { subject, predicate } => {
            Some(SlotQuery::forward(&subject, &predicate))
        }
        QueryPlan::Exist { subject, predicate } => {
            let mut query = SlotQuery::forward(&subject, &predicate);
            query.intent = SlotIntent::Exist;
            Some(query)
        }
        QueryPlan::Count { subject, predicate } => {
            let mut query = SlotQuery::forward(&subject, &predicate);
            query.intent = SlotIntent::Count;
            Some(query)
        }
        QueryPlan::CheckRelation {
            subject,
            predicate,
            object,
            ..
        } => {
            let mut query = SlotQuery::forward(&subject, &predicate);
            query.intent = SlotIntent::Check;
            query.expected = Some(object);
            Some(query)
        }
        QueryPlan::CheckTriple {
            subject_segments,
            object,
        } if subject_segments.len() == 2 => {
            let mut query = SlotQuery::forward(&subject_segments[0], &subject_segments[1]);
            query.intent = SlotIntent::Check;
            query.expected = Some(object);
            Some(query)
        }
        QueryPlan::FindSubject { object, predicate } => {
            let mut query = SlotQuery::forward(&object, &predicate);
            query.key = object;
            query.direction = SlotDirection::Object;
            Some(query)
        }
        _ => None,
    }
}

pub fn looks_contextual(question: &str) -> bool {
    const MARKERS: [&str; 22] = [
        "知识库",
        "知识图谱",
        "图谱",
        "数据库",
        "默认图",
        "全局",
        "据",
        "根据",
        "按照",
        "假设",
        "假想",
        "反事实",
        "会话",
        "对话",
        "刚才",
        "上一轮",
        "上次",
        "所有来源",
        "全部来源",
        "各个来源",
        "不同来源",
        "跨来源",
    ];
    MARKERS.iter().any(|marker| question.contains(marker))
}

fn trim_polite_prefix(text: &str) -> &str {
    for prefix in [
        "请问",
        "请告诉我一下",
        "请告诉我",
        "我想知道",
        "我想问一下",
        "帮我查一下",
        "帮我查查",
        "麻烦问一下",
    ] {
        if let Some(rest) = text.strip_prefix(prefix) {
            return rest.trim_start();
        }
    }
    text
}

fn strip_after_cue(rest: &str) -> &str {
    rest.trim_start_matches(|ch: char| {
        ch.is_whitespace()
            || matches!(
                ch,
                '，' | ',' | '：' | ':' | '、' | '；' | ';' | '。' | '的'
            )
    })
}

fn starts_with_ignore_ascii_case(text: &str, prefix: &str) -> bool {
    text.get(..prefix.len())
        .map(|head| head.eq_ignore_ascii_case(prefix))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ledger() -> FusionLedger {
        let mut ledger = FusionLedger::new();
        ledger.upsert_namespace(NamespaceRecord::new(
            "alice",
            "alice",
            NamespaceKind::Import,
            TrustLevel::Imported,
        ));
        ledger.record_assertion(crate::knowledge_fusion::ScopedAssertion::new(
            "alice",
            "张三",
            "师傅",
            "李四",
            ObjectKind::Literal,
            "alice.ttl",
        ));
        ledger.sort_and_dedup();
        ledger
    }

    fn corpus() -> LensCorpus {
        LensCorpus::new(
            &[FactRef::new("张三", "师傅", "王五")],
            &ledger(),
        )
    }

    #[test]
    fn detects_default_cue_and_strips_it() {
        let cue = detect_lens_cue("知识库里张三的师傅是谁？", &[]).expect("应识别知识库镜头");
        assert_eq!(cue.scope, LensScope::Default);
        assert_eq!(cue.question, "张三的师傅是谁？");
        let cue = detect_lens_cue("在知识库中，张三的师傅是谁", &[]).expect("应识别知识库镜头");
        assert_eq!(cue.scope, LensScope::Default);
        assert_eq!(cue.question, "张三的师傅是谁");
    }

    #[test]
    fn detects_source_cue_by_alias_and_by_raw_name() {
        let aliases = corpus().aliases();
        let cue = detect_lens_cue("据alice说，张三的师傅是谁", &aliases).expect("应识别来源");
        assert_eq!(cue.scope, LensScope::Namespace("alice".to_string()));
        assert_eq!(cue.question, "张三的师傅是谁");
        let cue = detect_lens_cue("据bob说，张三的师傅是谁", &aliases).expect("未知来源也应识别");
        assert_eq!(cue.scope, LensScope::Namespace("bob".to_string()));
        assert_eq!(cue.question, "张三的师傅是谁");
    }

    #[test]
    fn detects_kind_and_all_cues() {
        assert_eq!(
            detect_lens_cue("假设里张三的师傅是谁", &[]).unwrap().scope,
            LensScope::Kind(NamespaceKind::Hypothesis)
        );
        assert_eq!(
            detect_lens_cue("本次会话中张三的师傅是谁", &[]).unwrap().scope,
            LensScope::Kind(NamespaceKind::Session)
        );
        assert_eq!(
            detect_lens_cue("所有来源里张三的师傅是谁", &[]).unwrap().scope,
            LensScope::All
        );
        assert_eq!(
            detect_lens_cue("导入来源里张三的师傅是谁", &[]).unwrap().scope,
            LensScope::Kind(NamespaceKind::Import)
        );
    }

    #[test]
    fn plain_question_has_no_cue() {
        assert!(detect_lens_cue("张三的师傅是谁", &[]).is_none());
    }

    #[test]
    fn extracts_forward_reverse_exist_and_check_queries() {
        let query = extract_slot_query("张三的师傅是谁").expect("正向槽位");
        assert_eq!(query.key, "张三");
        assert_eq!(query.predicate, "师傅");
        assert_eq!(query.direction, SlotDirection::Subject);
        assert_eq!(query.intent, SlotIntent::Value);

        let query = extract_slot_query("谁的师傅是李四").expect("反向槽位");
        assert_eq!(query.key, "李四");
        assert_eq!(query.direction, SlotDirection::Object);

        let query = extract_slot_query("张三有师傅吗").expect("存在问句");
        assert_eq!(query.intent, SlotIntent::Exist);

        let query = extract_slot_query("张三的师傅是李四吗").expect("校验问句");
        assert_eq!(query.intent, SlotIntent::Check);
        assert_eq!(query.expected.as_deref(), Some("李四"));
    }

    #[test]
    fn default_lens_does_not_leak_namespace_facts() {
        let corpus = corpus();
        let view = corpus.query_slot(
            &LensScope::Default,
            &SlotQuery::forward("张三", "师傅"),
            true,
        );
        assert_eq!(view.state(), SlotState::Supported);
        assert_eq!(view.values[0].value, "王五");
        assert_eq!(view.elsewhere.len(), 1);
        assert_eq!(view.elsewhere[0].fact.value, "李四");
        let rendered = view.render();
        assert!(rendered.contains("王五"), "{}", rendered);
        assert!(!rendered.contains("是 李四"), "{}", rendered);
        assert!(rendered.contains("其他语境"), "{}", rendered);
    }

    #[test]
    fn namespace_lens_answers_with_provenance_only() {
        let corpus = corpus();
        let view = corpus.query_slot(
            &LensScope::Namespace("alice".to_string()),
            &SlotQuery::forward("张三", "师傅"),
            true,
        );
        assert_eq!(view.state(), SlotState::Supported);
        assert_eq!(view.values[0].value, "李四");
        assert_eq!(view.values[0].context_labels(), "alice（导入）");
        let rendered = view.render();
        assert!(rendered.contains("据「alice」的记录"), "{}", rendered);
        assert!(rendered.contains("李四"), "{}", rendered);
        assert!(!rendered.contains("是 王五"), "{}", rendered);
        assert!(rendered.contains("其他语境"), "{}", rendered);
    }

    #[test]
    fn list_facts_filters_scope_and_slot() {
        let corpus = corpus();
        let all = corpus.list_facts(&LensScope::All, None, None).unwrap();
        assert_eq!(all.len(), 2);
        let alice = corpus
            .list_facts(&LensScope::Namespace("alice".to_string()), None, None)
            .unwrap();
        assert_eq!(alice.len(), 1);
        assert_eq!(alice[0].fact.value, "李四");
        let by_slot = corpus
            .list_facts(&LensScope::All, Some("张三"), Some("师傅"))
            .unwrap();
        assert_eq!(by_slot.len(), 2);
        assert!(corpus
            .list_facts(&LensScope::All, Some("李四"), None)
            .unwrap()
            .is_empty());
        assert!(corpus
            .list_facts(&LensScope::Namespace("bob".to_string()), None, None)
            .unwrap_err()
            .contains("没有名为「bob」的语境"));
    }

    #[test]
    fn dialogue_snapshot_facts_are_visible_in_session_and_hypothesis_lenses() {
        let ledger = FusionLedger::new();
        let snapshot: DialogueSnapshot = serde_json::from_value(serde_json::json!({
            "focus_entities": [],
            "last_predicate": null,
            "history": [],
            "session_facts": [
                {"predicate": "师傅", "terms": ["张三", "李四"]}
            ],
            "assumptions": [
                {"predicate": "师傅", "terms": ["王五", "赵六"]}
            ]
        }))
        .expect("对话快照应可解码");
        let corpus = LensCorpus::with_dialogue(&[], &ledger, &snapshot);
        let session = corpus.query_slot(
            &LensScope::Kind(NamespaceKind::Session),
            &SlotQuery::forward("张三", "师傅"),
            false,
        );
        assert_eq!(session.state(), SlotState::Supported);
        assert_eq!(session.values[0].value, "李四");
        assert_eq!(session.values[0].contexts[0].id, SESSION_CONTEXT_ID);
        let hypothesis = corpus.query_slot(
            &LensScope::Kind(NamespaceKind::Hypothesis),
            &SlotQuery::forward("王五", "师傅"),
            false,
        );
        assert_eq!(hypothesis.state(), SlotState::Supported);
        assert_eq!(hypothesis.values[0].value, "赵六");
        assert_eq!(
            hypothesis.values[0].contexts[0].id,
            HYPOTHESIS_CONTEXT_ID
        );
        let all = corpus.query_slot(
            &LensScope::All,
            &SlotQuery::forward("张三", "师傅"),
            false,
        );
        assert_eq!(all.values[0].value, "李四");
        assert!(corpus
            .summaries()
            .iter()
            .any(|summary| summary.context.kind == NamespaceKind::Session));
        assert!(corpus
            .summaries()
            .iter()
            .any(|summary| summary.context.kind == NamespaceKind::Hypothesis));
    }

    #[test]
    fn empty_dialogue_snapshot_adds_no_virtual_contexts() {
        let ledger = FusionLedger::new();
        let snapshot: DialogueSnapshot = serde_json::from_value(serde_json::json!({
            "focus_entities": [],
            "last_predicate": null,
            "history": []
        }))
        .unwrap();
        let corpus = LensCorpus::with_dialogue(&[], &ledger, &snapshot);
        assert_eq!(corpus.contexts().len(), 1);
        assert_eq!(corpus.contexts()[0].id, DEFAULT_CONTEXT_ID);
    }

    #[test]
    fn all_lens_reports_conflict_without_truth_claim() {
        let corpus = corpus();
        let view = corpus.query_slot(
            &LensScope::All,
            &SlotQuery::forward("张三", "师傅"),
            true,
        );
        assert_eq!(view.state(), SlotState::Conflict);
        assert_eq!(view.values.len(), 2);
        let rendered = view.render();
        assert!(rendered.contains("2 种说法"), "{}", rendered);
        assert!(rendered.contains("未替你做真假裁决"), "{}", rendered);
        assert!(rendered.contains("alice（导入）"), "{}", rendered);
        assert!(rendered.contains("知识库"), "{}", rendered);
    }

    #[test]
    fn unknown_namespace_is_honest_and_lists_available_contexts() {
        let corpus = corpus();
        let view = corpus.query_slot(
            &LensScope::Namespace("bob".to_string()),
            &SlotQuery::forward("张三", "师傅"),
            true,
        );
        assert_eq!(view.state(), SlotState::Unresolved);
        let rendered = view.render();
        assert!(rendered.contains("没有名为「bob」的语境"), "{}", rendered);
        assert!(rendered.contains("alice（导入）"), "{}", rendered);
    }

    #[test]
    fn unknown_default_slot_points_at_other_contexts() {
        let corpus = LensCorpus::new(&[], &ledger());
        let view = corpus.query_slot(
            &LensScope::Default,
            &SlotQuery::forward("张三", "师傅"),
            true,
        );
        assert_eq!(view.state(), SlotState::Unknown);
        let rendered = view.render();
        assert!(rendered.contains("知识库中没有关于张三 的 师傅的记录"), "{}", rendered);
        assert!(rendered.contains("alice（导入）"), "{}", rendered);
    }

    #[test]
    fn reverse_query_uses_object_direction() {
        let corpus = corpus();
        let mut query = SlotQuery::forward("李四", "师傅");
        query.direction = SlotDirection::Object;
        let view = corpus.query_slot(&LensScope::Namespace("alice".to_string()), &query, true);
        assert_eq!(view.state(), SlotState::Supported);
        assert_eq!(view.values[0].value, "张三");
        assert!(view.render().contains("张三"), "{}", view.render());
    }

    #[test]
    fn check_relation_reports_yes_no_and_elsewhere() {
        let corpus = corpus();
        let mut query = SlotQuery::forward("张三", "师傅");
        query.intent = SlotIntent::Check;
        query.expected = Some("李四".to_string());
        let view = corpus.query_slot(&LensScope::Default, &query, true);
        assert!(view.render().contains("不是"), "{}", view.render());
        let mut query = SlotQuery::forward("张三", "师傅");
        query.intent = SlotIntent::Check;
        query.expected = Some("李四".to_string());
        let view = corpus.query_slot(&LensScope::Namespace("alice".to_string()), &query, true);
        assert!(view.render().contains("是的"), "{}", view.render());
    }

    #[test]
    fn retired_namespaces_are_not_in_all_lens() {
        let mut ledger = ledger();
        ledger.namespaces[0].status = NamespaceStatus::Retired;
        let corpus = LensCorpus::new(&[], &ledger);
        let view = corpus.query_slot(&LensScope::All, &SlotQuery::forward("张三", "师傅"), true);
        assert_eq!(view.state(), SlotState::Unknown);
        assert!(view.elsewhere.is_empty());
    }

    #[test]
    fn kind_lens_filters_by_namespace_kind() {
        let mut ledger = ledger();
        ledger.upsert_namespace(NamespaceRecord::new(
            "guess",
            "假设一",
            NamespaceKind::Hypothesis,
            TrustLevel::Speculative,
        ));
        ledger.record_assertion(crate::knowledge_fusion::ScopedAssertion::new(
            "guess",
            "张三",
            "师傅",
            "赵六",
            ObjectKind::Literal,
            "hypothesis",
        ));
        let corpus = LensCorpus::new(&[], &ledger);
        let view = corpus.query_slot(
            &LensScope::Kind(NamespaceKind::Hypothesis),
            &SlotQuery::forward("张三", "师傅"),
            true,
        );
        assert_eq!(view.values[0].value, "赵六");
        let rendered = view.render();
        assert!(rendered.contains("赵六"), "{}", rendered);
        assert!(!rendered.contains("是 李四"), "{}", rendered);
    }

    #[test]
    fn query_is_deterministic_regardless_of_input_order() {
        let mut ledger = ledger();
        ledger.record_assertion(crate::knowledge_fusion::ScopedAssertion::new(
            "alice",
            "张三",
            "师傅",
            "赵六",
            ObjectKind::Literal,
            "alice.ttl",
        ));
        let first = LensCorpus::new(&[FactRef::new("张三", "师傅", "王五")], &ledger);
        let mut reversed = ledger.clone();
        reversed.assertions.reverse();
        let second = LensCorpus::new(&[FactRef::new("张三", "师傅", "王五")], &reversed);
        let query = SlotQuery::forward("张三", "师傅");
        assert_eq!(
            first.query_slot(&LensScope::All, &query, true).render(),
            second.query_slot(&LensScope::All, &query, true).render()
        );
    }

    #[test]
    fn parse_scope_accepts_keywords_prefixes_and_labels() {
        let aliases = corpus().aliases();
        assert_eq!(parse_lens_scope("default", &aliases), LensScope::Default);
        assert_eq!(parse_lens_scope("all", &aliases), LensScope::All);
        assert_eq!(
            parse_lens_scope("session", &aliases),
            LensScope::Kind(NamespaceKind::Session)
        );
        assert_eq!(
            parse_lens_scope("namespace:alice", &aliases),
            LensScope::Namespace("alice".to_string())
        );
        assert_eq!(
            parse_lens_scope("alice", &aliases),
            LensScope::Namespace("alice".to_string())
        );
    }
}

//! CSC：情境语义闭包（Contextual Semantic Closure）。
//!
//! 多轮系统已经有会话事实、前向饱和与回答计划，但三者之间还缺一层显式契约：
//! 一条结论属于哪个情境、由哪些前提推出、撤回一轮后哪些结论必须一起消失。
//! 本模块把「当前世界」做成可寻址、可撤回、可解释的闭包视图：
//!
//! - [`GraphScope`]：基础图 / 会话 / 回合 / 假设 / 命名图。可见性是单向的：
//!   回合视图能看到本会话与基础图，会话视图能看到本会话全部回合，基础视图
//!   只看基础图；跨会话事实不会因为「都在内存里」而互相泄漏。
//! - [`ScopedAssertion`]：每条断言带稳定 id、作用域、来源与宾语种类。同一
//!   三元组可以在多个作用域重复登记；撤回按作用域或按三元组成批进行。
//! - [`ClosureSnapshot`]：对指定作用域重新求值的前向饱和结果。直接事实与
//!   推出事实分开标记；推出事实带规则来源、前提、轮次与整数置信度。
//! - [`ContextualClosure::export_quads`]：把闭包导出为 RDF 四元组。每条事实
//!   落在自己的命名图里，推出事实用 RDF 具体化 + `prov:wasDerivedFrom`
//!   记录前提，来源用 `prov:wasAttributedTo` 记录；不写回知识图谱。
//!
//! 纪律：不新造规则语言（规则仍是 [`Clause`]），不新造推理算法（前向饱和
//! 仍复用 [`ResolutionEngine::forward_chain`]），所有排序、摘要与计数确定。
//! 截断必须显式：事实预算或轮次上限被触达时，快照标记 `truncated`，绝不把
//! 「没展开」说成「推不出」。

use std::collections::{BTreeMap, BTreeSet};

use oxigraph::model::{GraphName, Literal, NamedNode, Quad, Subject, Term as RdfTerm};
use serde::{Deserialize, Serialize};

use crate::answer_plan::{ClaimEvidence, ClaimJustification, PlanClaim, MAX_PLAN_CLAIMS};
use crate::context_image::{canonical_json, fnv1a_hex, short_hash};
use crate::identity::{encode_segment, DEFAULT_BASE_NS};
use crate::knowledge_fusion::ObjectKind;
use crate::resolution_engine::{Atom, Clause, ResolutionEngine, Term};

/// 闭包命名图的 IRI 前缀。
pub const CLOSURE_GRAPH_NS: &str = "http://example.org/kg/graphs/";
/// 闭包具体化节点的 IRI 前缀。
pub const CLOSURE_NS: &str = "http://example.org/kg/closure/";

/// 默认前向饱和轮次。
pub const DEFAULT_MAX_ROUNDS: usize = 8;
/// 默认最多保留的推出事实数。
pub const DEFAULT_MAX_DERIVED: usize = 1_024;
/// 解释文本最多展开的节点数。
pub const MAX_EXPLAIN_NODES: usize = 64;
/// 解释文本最多展开的深度。
pub const MAX_EXPLAIN_DEPTH: usize = 12;
/// 快照文本最多展开的事实数。
pub const MAX_RENDER_FACTS: usize = 24;

const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
const RDF_SUBJECT: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#subject";
const RDF_PREDICATE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate";
const RDF_OBJECT: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#object";
const PROV_ENTITY: &str = "http://www.w3.org/ns/prov#Entity";
const PROV_WAS_DERIVED_FROM: &str = "http://www.w3.org/ns/prov#wasDerivedFrom";
const PROV_WAS_ATTRIBUTED_TO: &str = "http://www.w3.org/ns/prov#wasAttributedTo";

/// 闭包操作的显式错误。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClosureError {
    /// 必填字段为空。
    EmptyField { field: &'static str },
    /// 作用域字段为空。
    EmptyScope { field: &'static str },
    /// 生成的 IRI 不合法。
    InvalidIri { value: String },
}

impl std::fmt::Display for ClosureError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClosureError::EmptyField { field } => write!(formatter, "字段「{}」不能为空", field),
            ClosureError::EmptyScope { field } => {
                write!(formatter, "作用域字段「{}」不能为空", field)
            }
            ClosureError::InvalidIri { value } => write!(formatter, "IRI 不合法：{}", value),
        }
    }
}

impl std::error::Error for ClosureError {}

/// 一个可寻址的图作用域。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GraphScope {
    /// 基础图谱（跨会话只读事实）。
    Base,
    /// 整个会话。
    Session { session: String },
    /// 会话中的某一轮。
    Turn { session: String, turn: u64 },
    /// 一个「如果……」假设世界。
    Hypothesis { id: String },
    /// 调用方自定义的命名图。
    Named { id: String },
}

impl GraphScope {
    pub fn base() -> Self {
        GraphScope::Base
    }

    pub fn session(id: impl Into<String>) -> Self {
        GraphScope::Session { session: id.into() }
    }

    pub fn turn(session: impl Into<String>, turn: u64) -> Self {
        GraphScope::Turn {
            session: session.into(),
            turn,
        }
    }

    pub fn hypothesis(id: impl Into<String>) -> Self {
        GraphScope::Hypothesis { id: id.into() }
    }

    pub fn named(id: impl Into<String>) -> Self {
        GraphScope::Named { id: id.into() }
    }

    pub fn is_base(&self) -> bool {
        matches!(self, GraphScope::Base)
    }

    pub fn session_id(&self) -> Option<&str> {
        match self {
            GraphScope::Session { session } | GraphScope::Turn { session, .. } => {
                Some(session.as_str())
            }
            _ => None,
        }
    }

    /// 稳定的命名图 IRI；不同作用域不会碰撞。
    pub fn iri(&self) -> String {
        match self {
            GraphScope::Base => format!("{}base", CLOSURE_GRAPH_NS),
            GraphScope::Session { session } => {
                format!("{}session/{}", CLOSURE_GRAPH_NS, encode_segment(session))
            }
            GraphScope::Turn { session, turn } => format!(
                "{}session/{}/turn/{}",
                CLOSURE_GRAPH_NS,
                encode_segment(session),
                turn
            ),
            GraphScope::Hypothesis { id } => {
                format!("{}hypothesis/{}", CLOSURE_GRAPH_NS, encode_segment(id))
            }
            GraphScope::Named { id } => {
                format!("{}named/{}", CLOSURE_GRAPH_NS, encode_segment(id))
            }
        }
    }

    pub fn label_zh(&self) -> String {
        match self {
            GraphScope::Base => "基础图".to_string(),
            GraphScope::Session { session } => format!("会话 {}", session),
            GraphScope::Turn { session, turn } => format!("会话 {} 第 {} 轮", session, turn),
            GraphScope::Hypothesis { id } => format!("假设 {}", id),
            GraphScope::Named { id } => format!("命名图 {}", id),
        }
    }

    /// `requested` 视图是否包含本作用域的事实。
    ///
    /// 可见性只向下：会话包含自己的全部回合；回合包含本会话事实与自身；
    /// 假设与命名图只叠加在基础图上，不自动吸收会话内容，避免把别处的
    /// 临时前提混进假设世界。
    pub fn visible_from(&self, requested: &GraphScope) -> bool {
        if self.is_base() {
            return true;
        }
        match requested {
            GraphScope::Base => false,
            GraphScope::Session { session } => match self {
                GraphScope::Session { session: own } => own == session,
                GraphScope::Turn { session: own, .. } => own == session,
                _ => false,
            },
            GraphScope::Turn {
                session,
                turn: requested_turn,
            } => match self {
                GraphScope::Session { session: own } => own == session,
                GraphScope::Turn { session: own, turn } => own == session && turn == requested_turn,
                _ => false,
            },
            GraphScope::Hypothesis { id } => self == &GraphScope::Hypothesis { id: id.clone() },
            GraphScope::Named { id } => self == &GraphScope::Named { id: id.clone() },
        }
    }

    fn validate(&self) -> Result<(), ClosureError> {
        match self {
            GraphScope::Base => Ok(()),
            GraphScope::Session { session } => require_scope(session, "session"),
            GraphScope::Turn { session, .. } => require_scope(session, "session"),
            GraphScope::Hypothesis { id } => require_scope(id, "hypothesis.id"),
            GraphScope::Named { id } => require_scope(id, "named.id"),
        }
    }
}

fn require_scope(value: &str, field: &'static str) -> Result<(), ClosureError> {
    if value.trim().is_empty() {
        Err(ClosureError::EmptyScope { field })
    } else {
        Ok(())
    }
}

/// 一条带作用域与来源的断言。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ScopedAssertion {
    /// 稳定 id（作用域 + 三元组 + 宾语种类 + 来源）。
    pub id: String,
    pub scope: GraphScope,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    #[serde(default)]
    pub object_kind: ObjectKind,
    /// 证据指针：文件、轮次、URL 或调用方自定义标签。
    pub source: String,
}

impl ScopedAssertion {
    fn new(
        scope: GraphScope,
        subject: String,
        predicate: String,
        object: String,
        object_kind: ObjectKind,
        source: String,
    ) -> Self {
        let id = short_hash(&format!(
            "{}|{}|{}|{}|{}|{}",
            scope.iri(),
            subject,
            predicate,
            object_kind.tag(),
            object,
            source
        ));
        Self {
            id,
            scope,
            subject,
            predicate,
            object,
            object_kind,
            source,
        }
    }

    pub fn atom(&self) -> Atom {
        Atom::rel(&self.predicate, &self.subject, &self.object)
    }

    pub fn render_text(&self) -> String {
        format!(
            "[{}] {} -[{}]-> {}（来源：{}）",
            self.scope.label_zh(),
            self.subject,
            self.predicate,
            self.object,
            if self.source.trim().is_empty() {
                "未注明"
            } else {
                self.source.as_str()
            }
        )
    }
}

/// 推出事实使用的一条前提。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ClosurePremise {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

impl ClosurePremise {
    pub fn render_text(&self) -> String {
        format!("{} -[{}]-> {}", self.subject, self.predicate, self.object)
    }

    fn canonical(&self) -> String {
        format!("{}|{}|{}", self.subject, self.predicate, self.object)
    }
}

/// 推出事实的完整依据。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClosureDerivation {
    pub rule_source: String,
    #[serde(default)]
    pub premises: Vec<ClosurePremise>,
    pub depth: usize,
    /// 置信度的万分之一整数表示。
    pub confidence_bps: u16,
}

/// 闭包中的一条事实。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClosureFact {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    /// 事实被登记/推导时所处的作用域。
    pub scope: GraphScope,
    /// 是否至少有一条直接断言（推出但未断言为 false）。
    pub asserted: bool,
    /// 是否来自非基础作用域。
    #[serde(default)]
    pub session: bool,
    #[serde(default)]
    pub object_kind: ObjectKind,
    /// 直接断言的来源（推出事实为空）。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sources: Vec<String>,
    /// 推出事实的规则依据（直接事实为 None）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub derivation: Option<ClosureDerivation>,
}

impl ClosureFact {
    pub fn atom(&self) -> Atom {
        Atom::rel(&self.predicate, &self.subject, &self.object)
    }

    pub fn render_text(&self) -> String {
        format!(
            "{} -[{}]-> {}（{}）",
            self.subject,
            self.predicate,
            self.object,
            if self.asserted { "断言" } else { "推出" }
        )
    }

    pub fn render_sentence_zh(&self) -> String {
        format!("{}的{}是{}", self.subject, self.predicate, self.object)
    }

    pub fn confidence_bps(&self) -> u16 {
        self.derivation
            .as_ref()
            .map(|derivation| derivation.confidence_bps)
            .unwrap_or(10_000)
    }

    fn canonical(&self) -> String {
        format!(
            "{}|{}|{}|{}|{}",
            self.scope.iri(),
            self.subject,
            self.predicate,
            self.object,
            if self.asserted { "a" } else { "d" }
        )
    }

    fn slot_id(&self) -> String {
        format!("csc#{}", short_hash(&self.canonical()))
    }
}

/// 对某个作用域重新求值得到的闭包快照。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClosureSnapshot {
    pub scope: GraphScope,
    /// 直接事实 + 推出事实，按主体/谓词/客体稳定排序。
    pub facts: Vec<ClosureFact>,
    pub rule_count: usize,
    pub asserted_count: usize,
    pub derived_count: usize,
    pub max_rounds: usize,
    pub max_derived: usize,
    /// 是否触达事实预算或轮次上限（显式披露，不把截断说成无解）。
    pub truncated: bool,
    /// 是否触达推出事实数量上限。
    pub fact_budget_hit: bool,
    /// 是否触达前向饱和轮次上限（可能仍有未展开结论）。
    pub depth_limit_hit: bool,
    /// 覆盖作用域与全部事实的规范摘要。
    pub digest: String,
}

impl ClosureSnapshot {
    fn new(
        scope: GraphScope,
        facts: Vec<ClosureFact>,
        rule_count: usize,
        max_rounds: usize,
        max_derived: usize,
        fact_budget_hit: bool,
        depth_limit_hit: bool,
    ) -> Self {
        let asserted_count = facts.iter().filter(|fact| fact.asserted).count();
        let derived_count = facts.len().saturating_sub(asserted_count);
        let mut snapshot = Self {
            scope,
            facts,
            rule_count,
            asserted_count,
            derived_count,
            max_rounds,
            max_derived,
            truncated: fact_budget_hit || depth_limit_hit,
            fact_budget_hit,
            depth_limit_hit,
            digest: String::new(),
        };
        snapshot.digest = snapshot.compute_digest();
        snapshot
    }

    fn compute_digest(&self) -> String {
        let facts = self
            .facts
            .iter()
            .map(|fact| serde_json::to_value(fact).unwrap_or(serde_json::Value::Null))
            .collect::<Vec<_>>();
        let value = serde_json::json!({
            "scope": self.scope,
            "facts": facts,
            "rule_count": self.rule_count,
            "asserted_count": self.asserted_count,
            "derived_count": self.derived_count,
            "max_rounds": self.max_rounds,
            "max_derived": self.max_derived,
            "truncated": self.truncated,
            "fact_budget_hit": self.fact_budget_hit,
            "depth_limit_hit": self.depth_limit_hit,
        });
        fnv1a_hex(canonical_json(&value).as_bytes())
    }

    pub fn fact(&self, subject: &str, predicate: &str, object: &str) -> Option<&ClosureFact> {
        self.facts.iter().find(|fact| {
            fact.subject == subject && fact.predicate == predicate && fact.object == object
        })
    }

    pub fn asserted_facts(&self) -> impl Iterator<Item = &ClosureFact> {
        self.facts.iter().filter(|fact| fact.asserted)
    }

    pub fn derived_facts(&self) -> impl Iterator<Item = &ClosureFact> {
        self.facts.iter().filter(|fact| !fact.asserted)
    }

    /// 沿推导依据展开一条有界、确定性的证明链。
    pub fn explain(&self, subject: &str, predicate: &str, object: &str) -> Option<String> {
        let fact = self.fact(subject, predicate, object)?;
        let mut seen = BTreeSet::new();
        let mut budget = MAX_EXPLAIN_NODES;
        Some(self.explain_fact(fact, 0, &mut seen, &mut budget))
    }

    fn explain_fact(
        &self,
        fact: &ClosureFact,
        depth: usize,
        seen: &mut BTreeSet<String>,
        budget: &mut usize,
    ) -> String {
        let indent = "  ".repeat(depth);
        if *budget == 0 {
            return format!("{}…（解释节点预算已用尽）", indent);
        }
        *budget -= 1;
        let head = format!(
            "{}{}（{}）",
            indent,
            fact.render_text(),
            if fact.asserted {
                "直接断言"
            } else {
                "规则推出"
            }
        );
        let Some(derivation) = &fact.derivation else {
            let sources = if fact.sources.is_empty() {
                "未注明".to_string()
            } else {
                fact.sources.join("、")
            };
            return format!("{}；来源：{}", head, sources);
        };
        let mut lines = vec![format!(
            "{}；由「{}」推出，第 {} 轮，置信度 {:.2}",
            head,
            derivation.rule_source,
            derivation.depth,
            derivation.confidence_bps as f64 / 10_000.0
        )];
        if depth >= MAX_EXPLAIN_DEPTH {
            lines.push(format!("{}  …（解释深度上限）", indent));
            return lines.join("\n");
        }
        for premise in &derivation.premises {
            let key = premise.canonical();
            if !seen.insert(key.clone()) {
                lines.push(format!(
                    "{}  ← {}（已在当前证明链中）",
                    indent,
                    premise.render_text()
                ));
                continue;
            }
            match self.fact(&premise.subject, &premise.predicate, &premise.object) {
                Some(premise_fact) => {
                    lines.push(self.explain_fact(premise_fact, depth + 1, seen, budget));
                }
                None => lines.push(format!(
                    "{}  ← {}（前提不在当前视图）",
                    indent,
                    premise.render_text()
                )),
            }
        }
        lines.join("\n")
    }

    /// 导出该快照的中文审计文本（有界；计数始终覆盖全量）。
    pub fn render_text(&self) -> String {
        let mut output = format!(
            "情境闭包（{}；规则 {} 条；断言 {} 条，推出 {} 条；摘要 {}）\n",
            self.scope.label_zh(),
            self.rule_count,
            self.asserted_count,
            self.derived_count,
            self.digest
        );
        if self.truncated {
            let mut reasons = Vec::new();
            if self.fact_budget_hit {
                reasons.push(format!("推出事实达到上限 {}", self.max_derived));
            }
            if self.depth_limit_hit {
                reasons.push(format!("轮次达到上限 {}", self.max_rounds));
            }
            output.push_str(&format!(
                "  注意：{}；未展开的部分不等于推不出。\n",
                reasons.join("；")
            ));
        }
        for fact in self.facts.iter().take(MAX_RENDER_FACTS) {
            let mark = if fact.asserted { "断言" } else { "推出" };
            output.push_str(&format!(
                "  [{}] {} -[{}]-> {}{}\n",
                mark,
                fact.subject,
                fact.predicate,
                fact.object,
                fact.derivation
                    .as_ref()
                    .map(|derivation| format!(
                        "（由「{}」推出，第 {} 轮）",
                        derivation.rule_source, derivation.depth
                    ))
                    .unwrap_or_default()
            ));
        }
        if self.facts.len() > MAX_RENDER_FACTS {
            output.push_str(&format!(
                "  ……另有 {} 条未展开；摘要覆盖全量。\n",
                self.facts.len() - MAX_RENDER_FACTS
            ));
        }
        output.trim_end().to_string()
    }

    /// 把闭包事实转成回答计划命题：直接事实带证据，推出事实同时带证据与推导。
    pub fn plan_claims(&self, entity: &str, limit: usize) -> Vec<PlanClaim> {
        let limit = limit.min(MAX_PLAN_CLAIMS);
        let mut claims: Vec<PlanClaim> = self
            .facts
            .iter()
            .filter(|fact| fact.subject == entity)
            .map(|fact| self.fact_to_claim(fact))
            .collect();
        claims.sort_by(|left, right| {
            left.predicate
                .cmp(&right.predicate)
                .then_with(|| left.text.cmp(&right.text))
                .then_with(|| left.slot.cmp(&right.slot))
        });
        claims.truncate(limit);
        claims
    }

    fn fact_to_claim(&self, fact: &ClosureFact) -> PlanClaim {
        let evidence = if let Some(derivation) = &fact.derivation {
            derivation
                .premises
                .iter()
                .map(|premise| {
                    let session = self
                        .fact(&premise.subject, &premise.predicate, &premise.object)
                        .map(|premise_fact| premise_fact.session)
                        .unwrap_or(false);
                    ClaimEvidence::new(
                        &premise.subject,
                        &premise.predicate,
                        &premise.object,
                        session,
                    )
                })
                .collect()
        } else {
            vec![ClaimEvidence::new(
                &fact.subject,
                &fact.predicate,
                &fact.object,
                fact.session,
            )]
        };
        let mut claim = PlanClaim::new(
            fact.slot_id(),
            &fact.predicate,
            fact.render_sentence_zh(),
            fact.session,
        )
        .with_evidence(evidence);
        if let Some(derivation) = &fact.derivation {
            let premises = derivation
                .premises
                .iter()
                .map(|premise| {
                    let session = self
                        .fact(&premise.subject, &premise.predicate, &premise.object)
                        .map(|premise_fact| premise_fact.session)
                        .unwrap_or(false);
                    ClaimEvidence::new(
                        &premise.subject,
                        &premise.predicate,
                        &premise.object,
                        session,
                    )
                })
                .collect();
            claim = claim.with_justification(ClaimJustification::new(
                &derivation.rule_source,
                premises,
                derivation.depth,
                derivation.confidence_bps,
            ));
        }
        claim
    }
}

/// 作用域化事实与规则的闭包引擎。
#[derive(Debug, Clone)]
pub struct ContextualClosure {
    assertions: BTreeMap<String, ScopedAssertion>,
    rules: Vec<Clause>,
    rule_keys: BTreeSet<String>,
    max_rounds: usize,
    max_derived: usize,
}

impl Default for ContextualClosure {
    fn default() -> Self {
        Self::new()
    }
}

impl ContextualClosure {
    pub fn new() -> Self {
        Self::with_limits(DEFAULT_MAX_ROUNDS, DEFAULT_MAX_DERIVED)
    }

    pub fn with_limits(max_rounds: usize, max_derived: usize) -> Self {
        Self {
            assertions: BTreeMap::new(),
            rules: Vec::new(),
            rule_keys: BTreeSet::new(),
            max_rounds: max_rounds.max(1),
            max_derived: max_derived.max(1),
        }
    }

    pub fn max_rounds(&self) -> usize {
        self.max_rounds
    }

    pub fn max_derived(&self) -> usize {
        self.max_derived
    }

    pub fn assertion_count(&self) -> usize {
        self.assertions.len()
    }

    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    pub fn assertions(&self) -> impl Iterator<Item = &ScopedAssertion> {
        self.assertions.values()
    }

    pub fn rules(&self) -> &[Clause] {
        &self.rules
    }

    /// 登记一条实体关系断言（宾语按实体处理）。
    pub fn assert(
        &mut self,
        scope: GraphScope,
        subject: &str,
        predicate: &str,
        object: &str,
        source: &str,
    ) -> Result<String, ClosureError> {
        self.assert_with_kind(
            scope,
            subject,
            predicate,
            object,
            ObjectKind::Entity,
            source,
        )
    }

    /// 登记一条字面量断言（属性值、日期、数字等）。
    pub fn assert_literal(
        &mut self,
        scope: GraphScope,
        subject: &str,
        predicate: &str,
        object: &str,
        source: &str,
    ) -> Result<String, ClosureError> {
        self.assert_with_kind(
            scope,
            subject,
            predicate,
            object,
            ObjectKind::Literal,
            source,
        )
    }

    /// 登记断言并显式指定宾语种类；同一 (作用域, 三元组, 来源) 幂等。
    pub fn assert_with_kind(
        &mut self,
        scope: GraphScope,
        subject: &str,
        predicate: &str,
        object: &str,
        object_kind: ObjectKind,
        source: &str,
    ) -> Result<String, ClosureError> {
        require_field(subject, "subject")?;
        require_field(predicate, "predicate")?;
        require_field(object, "object")?;
        scope.validate()?;
        let assertion = ScopedAssertion::new(
            scope,
            subject.trim().to_string(),
            predicate.trim().to_string(),
            object.trim().to_string(),
            object_kind,
            source.trim().to_string(),
        );
        let id = assertion.id.clone();
        self.assertions.entry(id.clone()).or_insert(assertion);
        Ok(id)
    }

    /// 从二元基原子登记断言（变量原子显式拒绝）。
    pub fn assert_atom(
        &mut self,
        scope: GraphScope,
        atom: &Atom,
        source: &str,
    ) -> Result<String, ClosureError> {
        let (subject, predicate, object) = binary_triple(atom).ok_or(ClosureError::EmptyField {
            field: "ground atom",
        })?;
        self.assert(scope, &subject, &predicate, &object, source)
    }

    /// 添加一条 Horn 规则；同一规范形式只保留一份。
    pub fn add_rule(&mut self, clause: Clause) {
        let key = rule_key(&clause);
        if self.rule_keys.insert(key) {
            self.rules.push(clause);
        }
    }

    pub fn add_rules(&mut self, clauses: impl IntoIterator<Item = Clause>) {
        for clause in clauses {
            self.add_rule(clause);
        }
    }

    /// 按来源撤回规则；返回移除条数。
    pub fn remove_rules_by_source(&mut self, source: &str) -> usize {
        let before = self.rules.len();
        self.rules.retain(|clause| clause.source != source);
        self.rule_keys = self.rules.iter().map(rule_key).collect();
        before - self.rules.len()
    }

    /// 撤回整个作用域；下次求值时候选结论自然消失。
    pub fn retract_scope(&mut self, scope: &GraphScope) -> usize {
        let before = self.assertions.len();
        self.assertions
            .retain(|_, assertion| &assertion.scope != scope);
        before - self.assertions.len()
    }

    pub fn retract_assertion(&mut self, id: &str) -> bool {
        self.assertions.remove(id).is_some()
    }

    /// 撤回任意作用域下的同一三元组；返回移除条数。
    pub fn retract_triple(&mut self, subject: &str, predicate: &str, object: &str) -> usize {
        let before = self.assertions.len();
        self.assertions.retain(|_, assertion| {
            !(assertion.subject == subject
                && assertion.predicate == predicate
                && assertion.object == object)
        });
        before - self.assertions.len()
    }

    /// 当前登记过的作用域（含基础图），按稳定顺序返回。
    pub fn scopes(&self) -> Vec<GraphScope> {
        let mut scopes: BTreeSet<GraphScope> = self
            .assertions
            .values()
            .map(|assertion| assertion.scope.clone())
            .collect();
        scopes.insert(GraphScope::Base);
        scopes.into_iter().collect()
    }

    /// 对指定作用域求值：只吸收该视图可见的断言。
    pub fn materialize_for(&self, scope: &GraphScope) -> ClosureSnapshot {
        let mut engine = ResolutionEngine::new(32, 64);
        let mut asserted: BTreeMap<(String, String, String), Vec<&ScopedAssertion>> =
            BTreeMap::new();
        for assertion in self.assertions.values() {
            if !assertion.scope.visible_from(scope) {
                continue;
            }
            engine.add_fact_atom(assertion.atom());
            asserted
                .entry((
                    assertion.subject.clone(),
                    assertion.predicate.clone(),
                    assertion.object.clone(),
                ))
                .or_default()
                .push(assertion);
        }
        for clause in &self.rules {
            engine.add_clause(clause.clone());
        }

        let mut derived = engine.forward_chain(self.max_rounds, self.max_derived);
        derived.sort_by(|left, right| {
            left.depth
                .cmp(&right.depth)
                .then_with(|| left.atom.display().cmp(&right.atom.display()))
        });
        let fact_budget_hit = derived.len() >= self.max_derived;
        let depth_limit_hit =
            !derived.is_empty() && derived.iter().any(|fact| fact.depth >= self.max_rounds);

        let mut facts: Vec<ClosureFact> = Vec::new();
        let mut by_key: BTreeMap<(String, String, String), usize> = BTreeMap::new();
        for (key, group) in asserted {
            let mut scopes: Vec<&GraphScope> =
                group.iter().map(|assertion| &assertion.scope).collect();
            scopes.sort();
            let fact_scope = if scopes.iter().any(|scope| scope.is_base()) {
                GraphScope::Base
            } else {
                scopes[0].clone()
            };
            let mut sources: Vec<String> = group
                .iter()
                .map(|assertion| assertion.source.clone())
                .collect();
            sources.sort();
            sources.dedup();
            let object_kind = group
                .iter()
                .map(|assertion| assertion.object_kind)
                .next()
                .unwrap_or(ObjectKind::Entity);
            let fact = ClosureFact {
                subject: key.0.clone(),
                predicate: key.1.clone(),
                object: key.2.clone(),
                scope: fact_scope.clone(),
                asserted: true,
                session: !fact_scope.is_base(),
                object_kind,
                sources,
                derivation: None,
            };
            by_key.insert(key, facts.len());
            facts.push(fact);
        }

        for item in derived {
            let Some((subject, predicate, object)) = binary_triple(&item.atom) else {
                continue;
            };
            let key = (subject.clone(), predicate.clone(), object.clone());
            if by_key.contains_key(&key) {
                continue;
            }
            let mut premises: Vec<ClosurePremise> = item
                .premises
                .iter()
                .filter_map(binary_triple)
                .map(|(subject, predicate, object)| ClosurePremise {
                    subject,
                    predicate,
                    object,
                })
                .collect();
            premises.sort();
            premises.dedup();
            let session = premises.iter().any(|premise| {
                by_key
                    .get(&(
                        premise.subject.clone(),
                        premise.predicate.clone(),
                        premise.object.clone(),
                    ))
                    .map(|index| facts[*index].session)
                    .unwrap_or(false)
            });
            let derivation = ClosureDerivation {
                rule_source: item.rule_source,
                premises,
                depth: item.depth,
                confidence_bps: confidence_bps(item.confidence),
            };
            let fact = ClosureFact {
                subject,
                predicate,
                object,
                scope: scope.clone(),
                asserted: false,
                session,
                object_kind: ObjectKind::Entity,
                sources: Vec::new(),
                derivation: Some(derivation),
            };
            by_key.insert(key, facts.len());
            facts.push(fact);
        }

        facts.sort_by(|left, right| {
            left.subject
                .cmp(&right.subject)
                .then_with(|| left.predicate.cmp(&right.predicate))
                .then_with(|| left.object.cmp(&right.object))
                .then_with(|| right.asserted.cmp(&left.asserted))
                .then_with(|| left.scope.iri().cmp(&right.scope.iri()))
        });
        ClosureSnapshot::new(
            scope.clone(),
            facts,
            self.rules.len(),
            self.max_rounds,
            self.max_derived,
            fact_budget_hit,
            depth_limit_hit,
        )
    }

    /// 对所有已登记作用域分别求值（基础图总在其中）。
    pub fn materialize_all(&self) -> Vec<ClosureSnapshot> {
        self.scopes()
            .iter()
            .map(|scope| self.materialize_for(scope))
            .collect()
    }

    /// 为指定视图中的实体生成回答计划命题。
    pub fn plan_claims_for(
        &self,
        scope: &GraphScope,
        entity: &str,
        limit: usize,
    ) -> Vec<PlanClaim> {
        self.materialize_for(scope).plan_claims(entity, limit)
    }

    /// 把闭包导出为 RDF 四元组：基础图落默认图，其余作用域落命名图。
    ///
    /// 推出事实额外写入具体化节点：
    /// - `rdf:type prov:Entity` 与 `rdf:subject/predicate/object`；
    /// - 每个前提一条 `prov:wasDerivedFrom`；
    /// - 规则来源、轮次与置信度作为闭包命名空间下的字面量。
    pub fn export_quads(&self, scope: &GraphScope) -> Result<Vec<Quad>, ClosureError> {
        let snapshot = self.materialize_for(scope);
        let mut quads: Vec<Quad> = Vec::new();
        for fact in &snapshot.facts {
            let graph = graph_name(&fact.scope)?;
            let subject = named_node(&entity_iri(&fact.subject))?;
            let predicate = named_node(&property_iri(&fact.predicate))?;
            let object = rdf_object(&fact.object, fact.object_kind)?;
            quads.push(Quad::new(
                Subject::NamedNode(subject.clone()),
                predicate.clone(),
                object.clone(),
                graph.clone(),
            ));

            if let Some(derivation) = &fact.derivation {
                let fact_node = named_node(&format!(
                    "{}derived/{}",
                    CLOSURE_NS,
                    short_hash(&fact.canonical())
                ))?;
                quads.push(Quad::new(
                    Subject::NamedNode(fact_node.clone()),
                    named_node(RDF_TYPE)?,
                    RdfTerm::NamedNode(named_node(PROV_ENTITY)?),
                    graph.clone(),
                ));
                quads.push(Quad::new(
                    Subject::NamedNode(fact_node.clone()),
                    named_node(RDF_SUBJECT)?,
                    RdfTerm::NamedNode(subject.clone()),
                    graph.clone(),
                ));
                quads.push(Quad::new(
                    Subject::NamedNode(fact_node.clone()),
                    named_node(RDF_PREDICATE)?,
                    RdfTerm::NamedNode(predicate.clone()),
                    graph.clone(),
                ));
                quads.push(Quad::new(
                    Subject::NamedNode(fact_node.clone()),
                    named_node(RDF_OBJECT)?,
                    object,
                    graph.clone(),
                ));
                quads.push(literal_quad(
                    &fact_node,
                    &format!("{}ruleSource", CLOSURE_NS),
                    &derivation.rule_source,
                    &graph,
                )?);
                quads.push(literal_quad(
                    &fact_node,
                    &format!("{}depth", CLOSURE_NS),
                    &derivation.depth.to_string(),
                    &graph,
                )?);
                quads.push(literal_quad(
                    &fact_node,
                    &format!("{}confidence", CLOSURE_NS),
                    &format!("{:.4}", derivation.confidence_bps as f64 / 10_000.0),
                    &graph,
                )?);
                for premise in &derivation.premises {
                    let premise_node = named_node(&format!(
                        "{}premise/{}",
                        CLOSURE_NS,
                        short_hash(&premise.canonical())
                    ))?;
                    quads.push(Quad::new(
                        Subject::NamedNode(fact_node.clone()),
                        named_node(PROV_WAS_DERIVED_FROM)?,
                        RdfTerm::NamedNode(premise_node.clone()),
                        graph.clone(),
                    ));
                    quads.push(Quad::new(
                        Subject::NamedNode(premise_node.clone()),
                        named_node(RDF_TYPE)?,
                        RdfTerm::NamedNode(named_node(PROV_ENTITY)?),
                        graph.clone(),
                    ));
                    quads.push(literal_quad(
                        &premise_node,
                        &format!("{}atom", CLOSURE_NS),
                        &premise.render_text(),
                        &graph,
                    )?);
                }
            } else {
                for source in &fact.sources {
                    let source_node =
                        named_node(&format!("{}source/{}", CLOSURE_NS, short_hash(source)))?;
                    quads.push(Quad::new(
                        Subject::NamedNode(subject.clone()),
                        named_node(PROV_WAS_ATTRIBUTED_TO)?,
                        RdfTerm::NamedNode(source_node.clone()),
                        graph.clone(),
                    ));
                    quads.push(literal_quad(
                        &source_node,
                        &format!("{}label", CLOSURE_NS),
                        source,
                        &graph,
                    )?);
                }
            }
        }
        Ok(quads)
    }
}

fn require_field(value: &str, field: &'static str) -> Result<(), ClosureError> {
    if value.trim().is_empty() {
        Err(ClosureError::EmptyField { field })
    } else {
        Ok(())
    }
}

fn binary_triple(atom: &Atom) -> Option<(String, String, String)> {
    if atom.terms.len() != 2 {
        return None;
    }
    let subject = ground_term(&atom.terms[0])?;
    let object = ground_term(&atom.terms[1])?;
    Some((subject, atom.predicate.to_string(), object))
}

fn ground_term(term: &Term) -> Option<String> {
    match term {
        Term::Const(name) => Some(name.to_string()),
        Term::Var(_) => None,
    }
}

fn confidence_bps(confidence: f64) -> u16 {
    (confidence * 10_000.0).round().clamp(0.0, 10_000.0) as u16
}

fn rule_key(clause: &Clause) -> String {
    let body = clause
        .body
        .iter()
        .map(Atom::display)
        .collect::<Vec<_>>()
        .join("&");
    format!(
        "{}:-{}|{:.6}|{}",
        clause.head.display(),
        body,
        clause.confidence,
        clause.source
    )
}

fn entity_iri(name: &str) -> String {
    format!("{}entity/{}", DEFAULT_BASE_NS, encode_segment(name))
}

fn property_iri(predicate: &str) -> String {
    format!("{}{}", DEFAULT_BASE_NS, encode_segment(predicate))
}

fn graph_name(scope: &GraphScope) -> Result<GraphName, ClosureError> {
    if scope.is_base() {
        return Ok(GraphName::DefaultGraph);
    }
    Ok(GraphName::NamedNode(named_node(&scope.iri())?))
}

fn named_node(iri: &str) -> Result<NamedNode, ClosureError> {
    NamedNode::new(iri).map_err(|_| ClosureError::InvalidIri {
        value: iri.to_string(),
    })
}

fn rdf_object(value: &str, kind: ObjectKind) -> Result<RdfTerm, ClosureError> {
    match kind {
        ObjectKind::Entity => Ok(RdfTerm::NamedNode(named_node(&entity_iri(value))?)),
        ObjectKind::Literal => Ok(RdfTerm::Literal(Literal::new_simple_literal(value))),
    }
}

fn literal_quad(
    subject: &NamedNode,
    predicate: &str,
    value: &str,
    graph: &GraphName,
) -> Result<Quad, ClosureError> {
    Ok(Quad::new(
        Subject::NamedNode(subject.clone()),
        named_node(predicate)?,
        RdfTerm::Literal(Literal::new_simple_literal(value)),
        graph.clone(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transitive_rule() -> Clause {
        Clause::new(
            Atom::pattern("师祖", Term::var("X"), Term::var("Z")),
            vec![
                Atom::pattern("师傅", Term::var("X"), Term::var("Y")),
                Atom::pattern("师傅", Term::var("Y"), Term::var("Z")),
            ],
            0.9,
            "传递规则",
        )
    }

    fn closure_with_chain() -> ContextualClosure {
        let mut closure = ContextualClosure::new();
        closure
            .assert(GraphScope::base(), "张三", "师傅", "李四", "kb.txt")
            .unwrap();
        closure
            .assert(GraphScope::base(), "李四", "师傅", "王五", "kb.txt")
            .unwrap();
        closure.add_rule(transitive_rule());
        closure
    }

    #[test]
    fn scope_iris_are_stable_distinct_and_encoded() {
        let base = GraphScope::base().iri();
        let session = GraphScope::session("会话 1").iri();
        let turn = GraphScope::turn("会话 1", 2).iri();
        let other = GraphScope::turn("会话 2", 2).iri();
        assert_eq!(base, GraphScope::base().iri());
        assert!(base.ends_with("/base"));
        assert!(session.contains("session/会话%201"), "{}", session);
        assert!(turn.contains("/turn/2"));
        assert_ne!(turn, other);
        assert_eq!(
            GraphScope::session("").validate(),
            Err(ClosureError::EmptyScope { field: "session" })
        );
    }

    #[test]
    fn visibility_is_one_way_and_session_scoped() {
        let mut closure = ContextualClosure::new();
        closure
            .assert(GraphScope::base(), "甲", "p", "乙", "base")
            .unwrap();
        closure
            .assert(GraphScope::session("s1"), "甲", "p", "丙", "s1")
            .unwrap();
        closure
            .assert(GraphScope::turn("s1", 1), "甲", "p", "丁", "s1t1")
            .unwrap();
        closure
            .assert(GraphScope::turn("s1", 2), "甲", "p", "戊", "s1t2")
            .unwrap();
        closure
            .assert(GraphScope::session("s2"), "甲", "p", "己", "s2")
            .unwrap();

        let base = closure.materialize_for(&GraphScope::base());
        assert!(base.fact("甲", "p", "乙").is_some());
        assert!(base.fact("甲", "p", "丙").is_none());

        let session = closure.materialize_for(&GraphScope::session("s1"));
        assert!(session.fact("甲", "p", "乙").is_some());
        assert!(session.fact("甲", "p", "丙").is_some());
        assert!(session.fact("甲", "p", "丁").is_some());
        assert!(session.fact("甲", "p", "戊").is_some());
        assert!(session.fact("甲", "p", "己").is_none());

        let turn = closure.materialize_for(&GraphScope::turn("s1", 1));
        assert!(turn.fact("甲", "p", "丙").is_some());
        assert!(turn.fact("甲", "p", "丁").is_some());
        assert!(turn.fact("甲", "p", "戊").is_none());
        assert!(turn.fact("甲", "p", "己").is_none());

        let other = closure.materialize_for(&GraphScope::session("s2"));
        assert!(other.fact("甲", "p", "乙").is_some());
        assert!(other.fact("甲", "p", "己").is_some());
        assert!(other.fact("甲", "p", "丙").is_none());
    }

    #[test]
    fn forward_chain_carries_rule_premises_and_explanation() {
        let closure = closure_with_chain();
        let snapshot = closure.materialize_for(&GraphScope::base());
        let derived = snapshot
            .fact("张三", "师祖", "王五")
            .expect("应推出师祖关系");
        assert!(!derived.asserted);
        assert!(!derived.session);
        let derivation = derived.derivation.as_ref().unwrap();
        assert_eq!(derivation.rule_source, "传递规则");
        assert_eq!(derivation.depth, 1);
        assert_eq!(derivation.confidence_bps, 9_000);
        assert_eq!(derivation.premises.len(), 2);
        assert!(derivation
            .premises
            .iter()
            .any(|premise| premise.render_text() == "张三 -[师傅]-> 李四"));
        let explanation = snapshot.explain("张三", "师祖", "王五").unwrap();
        assert!(explanation.contains("传递规则"), "{}", explanation);
        assert!(
            explanation.contains("张三 -[师傅]-> 李四"),
            "{}",
            explanation
        );
        assert!(explanation.contains("直接断言"), "{}", explanation);
    }

    #[test]
    fn retracting_a_turn_contracts_the_closure_without_ghosts() {
        let mut closure = ContextualClosure::new();
        closure
            .assert(GraphScope::base(), "张三", "师傅", "李四", "base")
            .unwrap();
        closure
            .assert(GraphScope::turn("s1", 1), "李四", "师傅", "王五", "turn:1")
            .unwrap();
        closure.add_rule(transitive_rule());
        let turn = GraphScope::turn("s1", 1);
        let before = closure.materialize_for(&turn);
        assert!(before.fact("张三", "师祖", "王五").is_some());

        assert_eq!(closure.retract_scope(&turn), 1);
        let after = closure.materialize_for(&turn);
        assert!(after.fact("张三", "师祖", "王五").is_none());
        assert!(after.fact("张三", "师傅", "李四").is_some());
        assert_eq!(after.derived_count, 0);
        assert_eq!(closure.retract_scope(&turn), 0);
    }

    #[test]
    fn retracting_one_triple_removes_dependent_derivations() {
        let mut closure = closure_with_chain();
        assert!(closure
            .materialize_for(&GraphScope::base())
            .fact("张三", "师祖", "王五")
            .is_some());
        assert_eq!(closure.retract_triple("李四", "师傅", "王五"), 1);
        let after = closure.materialize_for(&GraphScope::base());
        assert!(after.fact("张三", "师祖", "王五").is_none());
        assert_eq!(after.derived_count, 0);
    }

    #[test]
    fn removing_a_rule_removes_its_derivations() {
        let mut closure = closure_with_chain();
        assert_eq!(closure.remove_rules_by_source("传递规则"), 1);
        let after = closure.materialize_for(&GraphScope::base());
        assert!(after.fact("张三", "师祖", "王五").is_none());
        assert_eq!(after.rule_count, 0);
    }

    #[test]
    fn fact_budget_and_round_limit_are_reported_explicitly() {
        let mut closure = ContextualClosure::with_limits(1, 1);
        closure
            .assert(GraphScope::base(), "甲", "p", "乙", "base")
            .unwrap();
        closure
            .assert(GraphScope::base(), "丙", "p", "丁", "base")
            .unwrap();
        closure.add_rule(Clause::new(
            Atom::pattern("q", Term::var("X"), Term::var("Y")),
            vec![Atom::pattern("p", Term::var("X"), Term::var("Y"))],
            1.0,
            "复制规则",
        ));
        let snapshot = closure.materialize_for(&GraphScope::base());
        assert_eq!(snapshot.derived_count, 1);
        assert!(snapshot.fact_budget_hit);
        assert!(snapshot.depth_limit_hit);
        assert!(snapshot.truncated);
        let text = snapshot.render_text();
        assert!(text.contains("未展开的部分不等于推不出"), "{}", text);
        assert!(text.contains("推出事实达到上限 1"), "{}", text);
    }

    #[test]
    fn snapshot_digest_is_independent_of_insertion_order() {
        let mut left = ContextualClosure::new();
        left.assert(GraphScope::base(), "甲", "p", "乙", "b")
            .unwrap();
        left.assert(GraphScope::session("s"), "甲", "p", "丙", "s")
            .unwrap();
        let mut right = ContextualClosure::new();
        right
            .assert(GraphScope::session("s"), "甲", "p", "丙", "s")
            .unwrap();
        right
            .assert(GraphScope::base(), "甲", "p", "乙", "b")
            .unwrap();
        let left_snapshot = left.materialize_for(&GraphScope::session("s"));
        let right_snapshot = right.materialize_for(&GraphScope::session("s"));
        assert_eq!(left_snapshot, right_snapshot);
        assert_eq!(left_snapshot.digest, right_snapshot.digest);
    }

    #[test]
    fn export_quads_uses_named_graphs_and_provenance() {
        let mut closure = ContextualClosure::new();
        closure
            .assert(GraphScope::base(), "张三", "师傅", "李四", "kb.txt")
            .unwrap();
        closure
            .assert(GraphScope::turn("s1", 1), "李四", "师傅", "王五", "turn:1")
            .unwrap();
        closure.add_rule(transitive_rule());
        let turn = GraphScope::turn("s1", 1);
        let quads = closure.export_quads(&turn).unwrap();

        assert!(quads.iter().any(|quad| {
            quad.graph_name == GraphName::DefaultGraph
                && quad.subject.to_string().contains("张三")
                && quad.predicate.as_str().contains("师傅")
        }));
        let turn_graph = GraphName::NamedNode(NamedNode::new(turn.iri()).unwrap());
        assert!(quads.iter().any(|quad| quad.graph_name == turn_graph));
        assert!(quads.iter().any(|quad| {
            quad.predicate.as_str() == PROV_WAS_DERIVED_FROM && quad.graph_name == turn_graph
        }));
        assert!(quads.iter().any(|quad| {
            quad.predicate.as_str() == PROV_WAS_ATTRIBUTED_TO
                && quad.object.to_string().contains("source/")
        }));
        assert!(quads.iter().any(|quad| {
            quad.predicate.as_str().ends_with("label") && quad.object.to_string().contains("kb.txt")
        }));
        assert!(quads.iter().any(|quad| {
            quad.predicate.as_str().ends_with("ruleSource")
                && quad.object.to_string().contains("传递规则")
        }));
    }

    #[test]
    fn plan_claims_distinguish_asserted_and_derived_provenance() {
        let mut closure = ContextualClosure::new();
        closure
            .assert(GraphScope::base(), "张三", "师傅", "李四", "base")
            .unwrap();
        closure
            .assert(GraphScope::turn("s1", 1), "李四", "师傅", "王五", "turn:1")
            .unwrap();
        closure.add_rule(transitive_rule());
        let turn = GraphScope::turn("s1", 1);
        let claims = closure.plan_claims_for(&turn, "张三", 16);
        let direct = claims
            .iter()
            .find(|claim| claim.predicate == "师傅")
            .unwrap();
        assert!(direct.justification.is_none());
        assert_eq!(direct.evidence.len(), 1);
        assert!(!direct.session, "基础图事实不属于会话叠加层");

        let derived = claims
            .iter()
            .find(|claim| claim.predicate == "师祖")
            .unwrap();
        let justification = derived.justification.as_ref().unwrap();
        assert_eq!(justification.rule_source, "传递规则");
        assert_eq!(justification.depth, 1);
        assert_eq!(derived.evidence.len(), 2);
        assert!(
            derived.session,
            "推出依赖会话前提时，命题必须标记为会话来源"
        );
        assert!(derived.evidence.iter().any(|item| item.object == "李四"));
        assert!(derived.evidence.iter().any(|item| item.object == "王五"));
        assert!(derived.text.contains("张三"));
        assert!(derived.text.contains("王五"));
    }

    #[test]
    fn empty_fields_are_rejected_instead_of_silently_ignored() {
        let mut closure = ContextualClosure::new();
        assert_eq!(
            closure.assert(GraphScope::base(), "", "p", "o", "s"),
            Err(ClosureError::EmptyField { field: "subject" })
        );
        assert_eq!(
            closure.assert(GraphScope::session(""), "s", "p", "o", "s"),
            Err(ClosureError::EmptyScope { field: "session" })
        );
        assert_eq!(closure.assertion_count(), 0);
    }
}

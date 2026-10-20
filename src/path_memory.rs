use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::temporal::{TemporalConstraint, TimeSpan, UnknownTimePolicy, Validity};

pub const DEFAULT_MAX_DEPTH: usize = 4;
pub const DEFAULT_MAX_PATHS: usize = 12;
pub const DEFAULT_MAX_EXPANSIONS: usize = 20_000;
pub const DEFAULT_BEAM_WIDTH: usize = 256;

const MAX_BLOCKED_RECORDS: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum EdgeSource {
    Observed,
    Attribute,
    Session,
    Inferred,
    Custom,
}

impl EdgeSource {
    pub fn label(self) -> &'static str {
        match self {
            EdgeSource::Observed => "事实",
            EdgeSource::Attribute => "属性",
            EdgeSource::Session => "会话",
            EdgeSource::Inferred => "推断",
            EdgeSource::Custom => "外部",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PathEdge {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub source: EdgeSource,
    pub confidence_bps: u16,
    /// 有效时间（valid time）：没写时间的边是未知，不是永远为真。
    #[serde(default, skip_serializing_if = "Validity::is_unknown")]
    pub validity: Validity,
}

impl PathEdge {
    pub fn new(subject: &str, predicate: &str, object: &str) -> Self {
        Self {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
            source: EdgeSource::Observed,
            confidence_bps: 10_000,
            validity: Validity::Unknown,
        }
    }

    pub fn with_source(mut self, source: EdgeSource) -> Self {
        self.source = source;
        self
    }

    pub fn with_confidence_bps(mut self, confidence_bps: u16) -> Self {
        self.confidence_bps = confidence_bps.min(10_000);
        self
    }

    pub fn with_validity(mut self, validity: Validity) -> Self {
        self.validity = validity;
        self
    }

    pub fn with_years(mut self, start_year: i32, end_year: i32) -> Self {
        self.validity = Validity::from_years(start_year, end_year);
        self
    }

    pub fn with_time_span(mut self, span: TimeSpan) -> Self {
        self.validity = Validity::Span(span);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PathStep {
    pub from: String,
    pub predicate: String,
    pub to: String,
    pub inverted: bool,
    pub source: EdgeSource,
    pub confidence_bps: u16,
    #[serde(default, skip_serializing_if = "Validity::is_unknown")]
    pub validity: Validity,
}

impl PathStep {
    pub fn display_predicate(&self) -> String {
        let mut notes = Vec::new();
        if self.inverted {
            notes.push("逆向".to_string());
        }
        if self.source != EdgeSource::Observed {
            notes.push(self.source.label().to_string());
        }
        if self.confidence_bps < 10_000 {
            notes.push(format!("置信度 {:.0}%", self.confidence_bps as f64 / 100.0));
        }
        if self.validity.is_known() {
            notes.push(format!("有效期 {}", self.validity.render()));
        }
        if notes.is_empty() {
            self.predicate.clone()
        } else {
            format!("{}({})", self.predicate, notes.join("，"))
        }
    }

    pub fn render(&self) -> String {
        format!(
            "{} --{}--> {}",
            self.from,
            self.display_predicate(),
            self.to
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PathWitness {
    pub steps: Vec<PathStep>,
    pub confidence_bps: u16,
    /// 全路径的有效期：所有步骤有效期的交集；任一步未知则整体未知。
    #[serde(default, skip_serializing_if = "Validity::is_unknown")]
    pub validity: Validity,
}

/// 带有效时间的路径见证；`PathWitness` 本身已携带 `validity`，此别名强调
/// 「时间已进入见证」的语义。
pub type TimedPathWitness = PathWitness;

impl PathWitness {
    pub fn new(steps: Vec<PathStep>) -> Self {
        let confidence_bps = steps
            .iter()
            .map(|step| step.confidence_bps)
            .min()
            .unwrap_or(10_000);
        let validity = steps
            .iter()
            .fold(Validity::Unbounded, |accumulated, step| {
                accumulated.intersection(step.validity)
            });
        Self {
            steps,
            confidence_bps,
            validity,
        }
    }

    /// 路径有效期时间未知（至少一步没写时间）。
    pub fn time_uncertain(&self) -> bool {
        self.validity.is_unknown()
    }

    /// 路径是否存在共同满足的时间点（有效期交集非空）。
    pub fn has_common_time(&self) -> bool {
        self.validity.is_satisfiable() && !self.validity.is_unknown()
    }

    pub fn start(&self) -> Option<&str> {
        self.steps.first().map(|step| step.from.as_str())
    }

    pub fn end(&self) -> Option<&str> {
        self.steps.last().map(|step| step.to.as_str())
    }

    pub fn hop_count(&self) -> usize {
        self.steps.len()
    }

    pub fn render(&self) -> String {
        let Some(first) = self.steps.first() else {
            return String::new();
        };
        let mut output = first.from.clone();
        for step in &self.steps {
            output.push_str(&format!(" --{}--> {}", step.display_predicate(), step.to));
        }
        output
    }

    pub fn key(&self) -> String {
        let mut key = String::new();
        for step in &self.steps {
            key.push_str(&step.from);
            key.push_str(&format!("[{}:{}]->", step.predicate, step.inverted));
            key.push_str(&step.to);
            key.push(';');
        }
        key
    }

    pub fn middle_nodes(&self) -> Vec<String> {
        if self.steps.len() < 2 {
            return Vec::new();
        }
        self.steps[..self.steps.len() - 1]
            .iter()
            .map(|step| step.to.clone())
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlockReason {
    UnknownNode {
        node: String,
    },
    MissingEdge {
        from: String,
        predicate: String,
    },
    NegatedEdge {
        from: String,
        predicate: String,
        to: String,
    },
    PredicateNotAllowed {
        from: String,
        predicate: String,
    },
    SourceNotAllowed {
        from: String,
        predicate: String,
        source: EdgeSource,
    },
    LowConfidence {
        from: String,
        predicate: String,
        confidence_bps: u16,
        minimum_bps: u16,
    },
    ForbiddenNode {
        node: String,
    },
    ExpectedTypeMismatch {
        node: String,
        expected: String,
    },
    /// 时间不合一：步骤有效期与请求时间不相交（跨时间假多跳的阻断点）。
    TemporalDisjoint {
        from: String,
        predicate: String,
        to: String,
        requested: String,
        valid: String,
    },
    ExpansionLimit,
}

impl BlockReason {
    pub fn render(&self) -> String {
        match self {
            BlockReason::UnknownNode { node } => format!("实体「{}」不在知识库中", node),
            BlockReason::MissingEdge { from, predicate } => {
                format!("「{}」没有「{}」边", from, predicate)
            }
            BlockReason::NegatedEdge {
                from,
                predicate,
                to,
            } => format!("「{}」到「{}」的「{}」边被否定记录阻断", from, to, predicate),
            BlockReason::PredicateNotAllowed { from, predicate } => {
                format!("「{}」的「{}」边不在允许的谓词范围内", from, predicate)
            }
            BlockReason::SourceNotAllowed {
                from,
                predicate,
                source,
            } => format!(
                "「{}」的「{}」边来源为「{}」，不在允许范围内",
                from,
                predicate,
                source.label()
            ),
            BlockReason::LowConfidence {
                from,
                predicate,
                confidence_bps,
                minimum_bps,
            } => format!(
                "「{}」的「{}」边置信度 {:.0}% 低于下限 {:.0}%",
                from,
                predicate,
                *confidence_bps as f64 / 100.0,
                *minimum_bps as f64 / 100.0
            ),
            BlockReason::ForbiddenNode { node } => {
                format!("「{}」被约束排除，不能作为路径节点", node)
            }
            BlockReason::ExpectedTypeMismatch { node, expected } => {
                format!("「{}」的类型与期望的「{}」不符", node, expected)
            }
            BlockReason::TemporalDisjoint {
                from,
                predicate,
                to,
                requested,
                valid,
            } => format!(
                "「{}」--{}-->「{}」的有效期（{}）与请求时间（{}）不相交，时间不合一",
                from, predicate, to, valid, requested
            ),
            BlockReason::ExpansionLimit => "达到扩展上限，路径枚举提前停止".to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockedPath {
    pub prefix: Vec<PathStep>,
    pub at: String,
    pub reason: BlockReason,
}

impl BlockedPath {
    pub fn render(&self) -> String {
        if self.prefix.is_empty() {
            return self.reason.render();
        }
        let mut path = self.prefix[0].from.clone();
        for step in &self.prefix {
            path.push_str(&format!(" --{}--> {}", step.display_predicate(), step.to));
        }
        format!("{}；在「{}」处{}", path, self.at, self.reason.render())
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PathBundle {
    pub start: String,
    pub goal: Option<String>,
    pub witnesses: Vec<PathWitness>,
    pub blocked: Vec<BlockedPath>,
    pub expansions: usize,
    pub visited: usize,
    pub truncated: bool,
    /// 本次枚举使用的时间约束（Any 时不序列化，保持旧输出形状）。
    #[serde(default, skip_serializing_if = "TemporalConstraint::is_any")]
    pub temporal: TemporalConstraint,
}

impl PathBundle {
    pub fn is_empty(&self) -> bool {
        self.witnesses.is_empty()
    }

    pub fn len(&self) -> usize {
        self.witnesses.len()
    }

    pub fn answer_groups(&self) -> BTreeMap<String, Vec<usize>> {
        let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (index, witness) in self.witnesses.iter().enumerate() {
            if let Some(end) = witness.end() {
                groups.entry(end.to_string()).or_default().push(index);
            }
        }
        groups
    }

    pub fn intermediary_groups(&self) -> BTreeMap<String, Vec<usize>> {
        let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (index, witness) in self.witnesses.iter().enumerate() {
            for node in witness.middle_nodes() {
                groups.entry(node).or_default().push(index);
            }
        }
        groups
    }

    pub fn render_text(&self) -> String {
        let mut output = String::new();
        let goal = self.goal.clone().unwrap_or_else(|| "?".to_string());
        if self.witnesses.is_empty() {
            output.push_str(&format!(
                "路径工作记忆：没有找到从「{}」到「{}」的路径。",
                self.start, goal
            ));
        } else {
            output.push_str(&format!(
                "路径工作记忆：{} → {}，{} 条路径，扩展 {} 次，访问 {} 个节点。",
                self.start,
                goal,
                self.witnesses.len(),
                self.expansions,
                self.visited
            ));
            for (index, witness) in self.witnesses.iter().enumerate() {
                output.push_str(&format!(
                    "\n  路径 {}（{} 跳，置信度 {:.0}%）：{}",
                    index + 1,
                    witness.hop_count(),
                    witness.confidence_bps as f64 / 100.0,
                    witness.render()
                ));
                if witness.validity.is_known() {
                    output.push_str(&format!("；共同有效期 {}", witness.validity.render()));
                }
            }
        }
        if !self.temporal.is_any() {
            output.push_str(&format!("\n  时间约束：{}", self.temporal.describe()));
            let uncertain = self
                .witnesses
                .iter()
                .filter(|witness| witness.time_uncertain())
                .count();
            if uncertain > 0 {
                output.push_str(&format!(
                    "；其中 {} 条路径含时间未知的边（按背景保留，结论需保留时间 caveat）",
                    uncertain
                ));
            }
        }
        for blocked in &self.blocked {
            output.push_str(&format!("\n  阻断：{}", blocked.render()));
        }
        if self.truncated {
            output.push_str("\n  说明：结果已按上限截断。");
        }
        output
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathPredicate {
    pub predicate: String,
    pub inverted: bool,
}

impl PathPredicate {
    pub fn forward(predicate: &str) -> Self {
        Self {
            predicate: predicate.to_string(),
            inverted: false,
        }
    }

    pub fn inverted(predicate: &str) -> Self {
        Self {
            predicate: predicate.to_string(),
            inverted: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TraversalOptions {
    pub max_depth: usize,
    pub max_paths: usize,
    pub max_expansions: usize,
    pub beam_width: usize,
    pub allowed_predicates: Option<BTreeSet<String>>,
    pub allowed_sources: Option<BTreeSet<EdgeSource>>,
    pub forbidden_nodes: BTreeSet<String>,
    pub expected_end_types: BTreeSet<String>,
    pub min_confidence_bps: u16,
    pub include_inferred: bool,
    /// 路径贯穿的时间约束：AsOf/During 时每一步都要通过时间合一检查。
    pub temporal: TemporalConstraint,
    /// 没写时间的步骤在时间约束下如何处理（默认严格）。
    pub unknown_time: UnknownTimePolicy,
}

impl Default for TraversalOptions {
    fn default() -> Self {
        Self {
            max_depth: DEFAULT_MAX_DEPTH,
            max_paths: DEFAULT_MAX_PATHS,
            max_expansions: DEFAULT_MAX_EXPANSIONS,
            beam_width: DEFAULT_BEAM_WIDTH,
            allowed_predicates: None,
            allowed_sources: None,
            forbidden_nodes: BTreeSet::new(),
            expected_end_types: BTreeSet::new(),
            min_confidence_bps: 0,
            include_inferred: true,
            temporal: TemporalConstraint::Any,
            unknown_time: UnknownTimePolicy::Strict,
        }
    }
}

impl TraversalOptions {
    pub fn with_temporal(mut self, temporal: TemporalConstraint) -> Self {
        self.temporal = temporal;
        self
    }

    pub fn with_unknown_time_policy(mut self, policy: UnknownTimePolicy) -> Self {
        self.unknown_time = policy;
        self
    }

    fn temporal_allows(&self, validity: Validity) -> bool {
        self.temporal.allows(validity, self.unknown_time)
    }
}

#[derive(Debug, Clone, Default)]
pub struct TraversalGraph {
    edges: Vec<PathEdge>,
    edge_keys: BTreeSet<(String, String, String)>,
    out_by_node: BTreeMap<String, Vec<usize>>,
    in_by_node: BTreeMap<String, Vec<usize>>,
    known_nodes: BTreeSet<String>,
    negative_edges: BTreeSet<(String, String, String)>,
    entity_types: BTreeMap<String, BTreeSet<String>>,
}

#[derive(Debug, Clone)]
struct PartialPath {
    node: String,
    steps: Vec<PathStep>,
}

impl TraversalGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_edge(&mut self, subject: &str, predicate: &str, object: &str) {
        self.add_edge_record(PathEdge::new(subject, predicate, object));
    }

    pub fn add_edge_with_source(
        &mut self,
        subject: &str,
        predicate: &str,
        object: &str,
        source: EdgeSource,
    ) {
        self.add_edge_record(PathEdge::new(subject, predicate, object).with_source(source));
    }

    /// 加入一条带有效期的边（ISO 年或年-月-日字符串均可）。
    pub fn add_timed_edge(
        &mut self,
        subject: &str,
        predicate: &str,
        object: &str,
        validity_text: &str,
    ) {
        let validity = crate::temporal::parse_validity(validity_text);
        self.add_edge_record(
            PathEdge::new(subject, predicate, object).with_validity(validity),
        );
    }

    pub fn add_edge_with_years(
        &mut self,
        subject: &str,
        predicate: &str,
        object: &str,
        start_year: i32,
        end_year: i32,
    ) {
        self.add_edge_record(
            PathEdge::new(subject, predicate, object).with_years(start_year, end_year),
        );
    }

    pub fn add_edge_record(&mut self, edge: PathEdge) {
        let key = (
            edge.subject.clone(),
            edge.predicate.clone(),
            edge.object.clone(),
        );
        if !self.edge_keys.insert(key) {
            return;
        }
        let index = self.edges.len();
        self.known_nodes.insert(edge.subject.clone());
        self.known_nodes.insert(edge.object.clone());
        self.out_by_node
            .entry(edge.subject.clone())
            .or_default()
            .push(index);
        self.in_by_node
            .entry(edge.object.clone())
            .or_default()
            .push(index);
        self.edges.push(edge);
    }

    pub fn add_negative_edge(&mut self, subject: &str, predicate: &str, object: &str) {
        self.known_nodes.insert(subject.to_string());
        self.known_nodes.insert(object.to_string());
        self.negative_edges.insert((
            subject.to_string(),
            predicate.to_string(),
            object.to_string(),
        ));
    }

    pub fn add_entity_type(&mut self, entity: &str, entity_type: &str) {
        self.entity_types
            .entry(entity.to_string())
            .or_default()
            .insert(entity_type.to_string());
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn node_count(&self) -> usize {
        self.known_nodes.len()
    }

    pub fn has_node(&self, node: &str) -> bool {
        self.known_nodes.contains(node)
    }

    pub fn edges(&self) -> &[PathEdge] {
        &self.edges
    }

    pub fn entity_types_of(&self, entity: &str) -> Option<&BTreeSet<String>> {
        self.entity_types.get(entity)
    }

    fn step_from_edge(&self, edge: &PathEdge, inverted: bool) -> PathStep {
        if inverted {
            PathStep {
                from: edge.object.clone(),
                predicate: edge.predicate.clone(),
                to: edge.subject.clone(),
                inverted: true,
                source: edge.source,
                confidence_bps: edge.confidence_bps,
                validity: edge.validity,
            }
        } else {
            PathStep {
                from: edge.subject.clone(),
                predicate: edge.predicate.clone(),
                to: edge.object.clone(),
                inverted: false,
                source: edge.source,
                confidence_bps: edge.confidence_bps,
                validity: edge.validity,
            }
        }
    }

    fn step_is_negated(&self, step: &PathStep) -> bool {
        let key = if step.inverted {
            (step.to.clone(), step.predicate.clone(), step.from.clone())
        } else {
            (step.from.clone(), step.predicate.clone(), step.to.clone())
        };
        self.negative_edges.contains(&key)
    }

    fn step_block_reason(
        &self,
        step: &PathStep,
        options: &TraversalOptions,
    ) -> Option<BlockReason> {
        if let Some(allowed) = &options.allowed_predicates {
            if !allowed.contains(&step.predicate) {
                return Some(BlockReason::PredicateNotAllowed {
                    from: step.from.clone(),
                    predicate: step.predicate.clone(),
                });
            }
        }
        if let Some(allowed) = &options.allowed_sources {
            if !allowed.contains(&step.source) {
                return Some(BlockReason::SourceNotAllowed {
                    from: step.from.clone(),
                    predicate: step.predicate.clone(),
                    source: step.source,
                });
            }
        }
        if !options.include_inferred && step.source == EdgeSource::Inferred {
            return Some(BlockReason::SourceNotAllowed {
                from: step.from.clone(),
                predicate: step.predicate.clone(),
                source: step.source,
            });
        }
        if step.confidence_bps < options.min_confidence_bps {
            return Some(BlockReason::LowConfidence {
                from: step.from.clone(),
                predicate: step.predicate.clone(),
                confidence_bps: step.confidence_bps,
                minimum_bps: options.min_confidence_bps,
            });
        }
        if options.forbidden_nodes.contains(&step.to) {
            return Some(BlockReason::ForbiddenNode {
                node: step.to.clone(),
            });
        }
        if self.step_is_negated(step) {
            return Some(BlockReason::NegatedEdge {
                from: step.from.clone(),
                predicate: step.predicate.clone(),
                to: step.to.clone(),
            });
        }
        if !options.temporal_allows(step.validity) {
            return Some(BlockReason::TemporalDisjoint {
                from: step.from.clone(),
                predicate: step.predicate.clone(),
                to: step.to.clone(),
                requested: options.temporal.describe(),
                valid: step.validity.render(),
            });
        }
        None
    }

    fn step_allowed(&self, step: &PathStep, options: &TraversalOptions) -> bool {
        self.step_block_reason(step, options).is_none()
    }

    fn outgoing(&self, node: &str, link: &PathPredicate) -> Vec<PathStep> {
        let mut steps = Vec::new();
        if link.inverted {
            if let Some(indices) = self.in_by_node.get(node) {
                for index in indices {
                    let edge = &self.edges[*index];
                    if edge.predicate == link.predicate {
                        steps.push(self.step_from_edge(edge, true));
                    }
                }
            }
        } else if let Some(indices) = self.out_by_node.get(node) {
            for index in indices {
                let edge = &self.edges[*index];
                if edge.predicate == link.predicate {
                    steps.push(self.step_from_edge(edge, false));
                }
            }
        }
        steps.sort_by(|left, right| {
            left.to
                .cmp(&right.to)
                .then(left.predicate.cmp(&right.predicate))
                .then(left.inverted.cmp(&right.inverted))
        });
        steps
    }

    fn outgoing_all(&self, node: &str) -> Vec<PathStep> {
        let mut steps = Vec::new();
        if let Some(indices) = self.out_by_node.get(node) {
            for index in indices {
                steps.push(self.step_from_edge(&self.edges[*index], false));
            }
        }
        steps.sort_by(|left, right| {
            left.to
                .cmp(&right.to)
                .then(left.predicate.cmp(&right.predicate))
                .then(left.source.cmp(&right.source))
        });
        steps
    }

    fn incoming_reversed(&self, node: &str) -> Vec<PathStep> {
        let mut steps = Vec::new();
        if let Some(indices) = self.in_by_node.get(node) {
            for index in indices {
                steps.push(self.step_from_edge(&self.edges[*index], true));
            }
        }
        steps.sort_by(|left, right| {
            left.to
                .cmp(&right.to)
                .then(left.predicate.cmp(&right.predicate))
                .then(left.source.cmp(&right.source))
        });
        steps
    }

    fn blocked_reasons(
        &self,
        node: &str,
        link: &PathPredicate,
        options: &TraversalOptions,
    ) -> Vec<BlockReason> {
        let mut reasons = Vec::new();
        let mut has_allowed_edge = false;
        let candidates = self.outgoing(node, link);
        for candidate in &candidates {
            match self.step_block_reason(candidate, options) {
                Some(reason) => reasons.push(reason),
                None => has_allowed_edge = true,
            }
        }
        if !link.inverted {
            for (subject, predicate, object) in &self.negative_edges {
                if subject == node && predicate == &link.predicate {
                    reasons.push(BlockReason::NegatedEdge {
                        from: subject.clone(),
                        predicate: predicate.clone(),
                        to: object.clone(),
                    });
                }
            }
        } else {
            for (subject, predicate, object) in &self.negative_edges {
                if object == node && predicate == &link.predicate {
                    reasons.push(BlockReason::NegatedEdge {
                        from: node.to_string(),
                        predicate: predicate.clone(),
                        to: subject.clone(),
                    });
                }
            }
        }
        if !has_allowed_edge && reasons.is_empty() {
            reasons.push(BlockReason::MissingEdge {
                from: node.to_string(),
                predicate: link.predicate.clone(),
            });
        }
        reasons
    }

    fn end_type_allowed(&self, node: &str, options: &TraversalOptions) -> bool {
        if options.expected_end_types.is_empty() {
            return true;
        }
        let Some(types) = self.entity_types.get(node) else {
            return true;
        };
        types
            .iter()
            .any(|entity_type| options.expected_end_types.contains(entity_type))
    }

    fn push_blocked_at(
        &self,
        bundle: &mut PathBundle,
        prefix: &[PathStep],
        at: String,
        reason: BlockReason,
    ) {
        if bundle.blocked.len() >= MAX_BLOCKED_RECORDS {
            bundle.truncated = true;
            return;
        }
        bundle.blocked.push(BlockedPath {
            prefix: prefix.to_vec(),
            at,
            reason,
        });
    }

    fn push_blocked(&self, bundle: &mut PathBundle, prefix: &[PathStep], reason: BlockReason) {
        let at = prefix
            .last()
            .map(|step| step.to.clone())
            .unwrap_or_else(|| bundle.start.clone());
        self.push_blocked_at(bundle, prefix, at, reason);
    }

    pub fn enumerate_chain(
        &self,
        start: &str,
        chain: &[PathPredicate],
        options: &TraversalOptions,
    ) -> PathBundle {
        let mut bundle = PathBundle {
            start: start.to_string(),
            temporal: options.temporal.clone(),
            ..PathBundle::default()
        };
        if !self.known_nodes.contains(start) {
            bundle.blocked.push(BlockedPath {
                prefix: Vec::new(),
                at: start.to_string(),
                reason: BlockReason::UnknownNode {
                    node: start.to_string(),
                },
            });
            return bundle;
        }
        if chain.is_empty() || options.max_depth == 0 {
            return bundle;
        }
        let depth = chain.len().min(options.max_depth);
        let chain = if depth < chain.len() {
            bundle.truncated = true;
            &chain[..depth]
        } else {
            chain
        };

        let mut frontier: Vec<Vec<PathStep>> = vec![Vec::new()];
        let mut expansions = 0usize;
        let mut visited_nodes: BTreeSet<String> = BTreeSet::from([start.to_string()]);
        let mut witness_keys: BTreeSet<String> = BTreeSet::new();

        'levels: for (level, link) in chain.iter().enumerate() {
            let terminal = level + 1 == chain.len();
            let mut next_frontier: Vec<Vec<PathStep>> = Vec::new();
            let mut next_keys: BTreeSet<String> = BTreeSet::new();

            for prefix in &frontier {
                let node = prefix
                    .last()
                    .map(|step| step.to.as_str())
                    .unwrap_or(start);
                let candidates = self.outgoing(node, link);
                let reasons = self.blocked_reasons(node, link, options);
                for reason in reasons {
                    self.push_blocked(&mut bundle, prefix, reason);
                }
                for candidate in candidates {
                    if !self.step_allowed(&candidate, options) {
                        continue;
                    }
                    if expansions >= options.max_expansions {
                        bundle.truncated = true;
                        break 'levels;
                    }
                    if candidate.to == start
                        || prefix
                            .iter()
                            .any(|step| step.to == candidate.to)
                    {
                        continue;
                    }
                    expansions += 1;
                    visited_nodes.insert(candidate.to.clone());
                    let mut path = prefix.clone();
                    path.push(candidate);
                    if terminal {
                        let end = path
                            .last()
                            .map(|step| step.to.clone())
                            .unwrap_or_default();
                        if !self.end_type_allowed(&end, options) {
                            let expected = options
                                .expected_end_types
                                .iter()
                                .cloned()
                                .collect::<Vec<_>>()
                                .join("/");
                            self.push_blocked(
                                &mut bundle,
                                &path[..path.len().saturating_sub(1)],
                                BlockReason::ExpectedTypeMismatch {
                                    node: end.clone(),
                                    expected,
                                },
                            );
                            continue;
                        }
                        let witness = PathWitness::new(path);
                        if !options.temporal_allows(witness.validity) {
                            self.push_blocked(
                                &mut bundle,
                                &witness.steps[..witness.steps.len().saturating_sub(1)],
                                BlockReason::TemporalDisjoint {
                                    from: witness.steps[0].from.clone(),
                                    predicate: witness.steps[0].predicate.clone(),
                                    to: end.clone(),
                                    requested: options.temporal.describe(),
                                    valid: witness.validity.render(),
                                },
                            );
                            continue;
                        }
                        if witness_keys.insert(witness.key()) {
                            bundle.witnesses.push(witness);
                            if bundle.witnesses.len() >= options.max_paths.max(1) {
                                bundle.truncated = true;
                                break 'levels;
                            }
                        }
                    } else {
                        let key = witness_prefix_key(start, &path);
                        if next_keys.insert(key)
                            && next_frontier.len() < options.beam_width.max(1)
                        {
                            next_frontier.push(path);
                        } else if next_frontier.len() >= options.beam_width.max(1) {
                            bundle.truncated = true;
                        }
                    }
                }
            }

            if terminal || next_frontier.is_empty() {
                break;
            }
            frontier = next_frontier;
        }

        bundle.expansions = expansions;
        bundle.visited = visited_nodes.len();
        bundle.witnesses.sort_by(|left, right| {
            left.hop_count()
                .cmp(&right.hop_count())
                .then(right.confidence_bps.cmp(&left.confidence_bps))
                .then(left.key().cmp(&right.key()))
        });
        dedupe_blocked(&mut bundle);
        bundle
    }

    pub fn enumerate_between(
        &self,
        start: &str,
        goal: &str,
        options: &TraversalOptions,
    ) -> PathBundle {
        let mut bundle = PathBundle {
            start: start.to_string(),
            goal: Some(goal.to_string()),
            temporal: options.temporal.clone(),
            ..PathBundle::default()
        };
        if start == goal {
            return bundle;
        }
        let start_known = self.known_nodes.contains(start);
        let goal_known = self.known_nodes.contains(goal);
        if !start_known {
            bundle.blocked.push(BlockedPath {
                prefix: Vec::new(),
                at: start.to_string(),
                reason: BlockReason::UnknownNode {
                    node: start.to_string(),
                },
            });
        }
        if !goal_known {
            bundle.blocked.push(BlockedPath {
                prefix: Vec::new(),
                at: goal.to_string(),
                reason: BlockReason::UnknownNode {
                    node: goal.to_string(),
                },
            });
        }
        if !start_known || !goal_known || options.max_depth == 0 {
            return bundle;
        }

        for (subject, predicate, object) in &self.negative_edges {
            if (subject == start && object == goal) || (subject == goal && object == start) {
                bundle.blocked.push(BlockedPath {
                    prefix: Vec::new(),
                    at: start.to_string(),
                    reason: BlockReason::NegatedEdge {
                        from: start.to_string(),
                        predicate: predicate.clone(),
                        to: goal.to_string(),
                    },
                });
            }
        }

        let mut forward_frontier = vec![PartialPath {
            node: start.to_string(),
            steps: Vec::new(),
        }];
        let mut backward_frontier = vec![PartialPath {
            node: goal.to_string(),
            steps: Vec::new(),
        }];
        let mut forward_seen: BTreeMap<String, Vec<Vec<PathStep>>> = BTreeMap::new();
        let mut backward_seen: BTreeMap<String, Vec<Vec<PathStep>>> = BTreeMap::new();
        forward_seen.insert(start.to_string(), vec![Vec::new()]);
        backward_seen.insert(goal.to_string(), vec![Vec::new()]);

        let mut forward_depth = 0usize;
        let mut backward_depth = 0usize;
        let mut expansions = 0usize;
        let mut visited_nodes: BTreeSet<String> =
            BTreeSet::from([start.to_string(), goal.to_string()]);
        let mut witness_keys: BTreeSet<String> = BTreeSet::new();

        loop {
            let mut advanced = false;

            if !forward_frontier.is_empty()
                && forward_depth < options.max_depth
                && forward_depth + backward_depth < options.max_depth
            {
                let mut next_frontier = Vec::new();
                let mut next_keys: BTreeSet<String> = BTreeSet::new();
                'forward: for partial in &forward_frontier {
                    if expansions >= options.max_expansions {
                        bundle.truncated = true;
                        break;
                    }
                    for candidate in self.outgoing_all(&partial.node) {
                        if let Some(reason) = self.step_block_reason(&candidate, options) {
                            self.push_blocked(&mut bundle, &partial.steps, reason);
                            continue;
                        }
                        if expansions >= options.max_expansions {
                            bundle.truncated = true;
                            break 'forward;
                        }
                        if candidate.to == start
                            || path_nodes(start, &partial.steps).contains(&candidate.to)
                        {
                            continue;
                        }
                        expansions += 1;
                        visited_nodes.insert(candidate.to.clone());
                        let mut steps = partial.steps.clone();
                        steps.push(candidate.clone());
                        let key = witness_prefix_key(start, &steps);
                        if !next_keys.insert(key) {
                            continue;
                        }
                        next_frontier.push(PartialPath {
                            node: candidate.to.clone(),
                            steps,
                        });
                    }
                }
                forward_depth += 1;
                for partial in &next_frontier {
                    if let Some(back_paths) = backward_seen.get(&partial.node) {
                        for back in back_paths {
                            if let Some(witness) = join_bidirectional(start, goal, &partial.steps, back)
                            {
                                if !options.temporal_allows(witness.validity) {
                                    self.push_blocked(
                                        &mut bundle,
                                        &partial.steps,
                                        BlockReason::TemporalDisjoint {
                                            from: start.to_string(),
                                            predicate: witness
                                                .steps
                                                .last()
                                                .map(|step| step.predicate.clone())
                                                .unwrap_or_default(),
                                            to: goal.to_string(),
                                            requested: options.temporal.describe(),
                                            valid: witness.validity.render(),
                                        },
                                    );
                                    continue;
                                }
                                if witness_keys.insert(witness.key()) {
                                    bundle.witnesses.push(witness);
                                    if bundle.witnesses.len() >= options.max_paths.max(1) {
                                        bundle.truncated = true;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
                for partial in &next_frontier {
                    let paths = forward_seen.entry(partial.node.clone()).or_default();
                    if paths.len() < options.max_paths.max(1) {
                        paths.push(partial.steps.clone());
                    }
                }
                forward_frontier = trim_frontier(next_frontier, options.beam_width);
                advanced = true;
            }

            if bundle.witnesses.len() >= options.max_paths.max(1) {
                break;
            }

            if !backward_frontier.is_empty()
                && backward_depth < options.max_depth
                && forward_depth + backward_depth < options.max_depth
            {
                let mut next_frontier = Vec::new();
                let mut next_keys: BTreeSet<String> = BTreeSet::new();
                'backward: for partial in &backward_frontier {
                    if expansions >= options.max_expansions {
                        bundle.truncated = true;
                        break;
                    }
                    for candidate in self.incoming_reversed(&partial.node) {
                        if !self.step_allowed(&candidate, options) {
                            continue;
                        }
                        if expansions >= options.max_expansions {
                            bundle.truncated = true;
                            break 'backward;
                        }
                        if candidate.to == goal
                            || path_nodes(goal, &partial.steps).contains(&candidate.to)
                        {
                            continue;
                        }
                        expansions += 1;
                        visited_nodes.insert(candidate.to.clone());
                        let mut steps = partial.steps.clone();
                        steps.push(candidate.clone());
                        let key = witness_prefix_key(goal, &steps);
                        if !next_keys.insert(key) {
                            continue;
                        }
                        next_frontier.push(PartialPath {
                            node: candidate.to.clone(),
                            steps,
                        });
                    }
                }
                backward_depth += 1;
                for partial in &next_frontier {
                    if let Some(forward_paths) = forward_seen.get(&partial.node) {
                        for forward in forward_paths {
                            if let Some(witness) =
                                join_bidirectional(start, goal, forward, &partial.steps)
                            {
                                if !options.temporal_allows(witness.validity) {
                                    self.push_blocked(
                                        &mut bundle,
                                        forward,
                                        BlockReason::TemporalDisjoint {
                                            from: start.to_string(),
                                            predicate: witness
                                                .steps
                                                .last()
                                                .map(|step| step.predicate.clone())
                                                .unwrap_or_default(),
                                            to: goal.to_string(),
                                            requested: options.temporal.describe(),
                                            valid: witness.validity.render(),
                                        },
                                    );
                                    continue;
                                }
                                if witness_keys.insert(witness.key()) {
                                    bundle.witnesses.push(witness);
                                    if bundle.witnesses.len() >= options.max_paths.max(1) {
                                        bundle.truncated = true;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
                for partial in &next_frontier {
                    let paths = backward_seen.entry(partial.node.clone()).or_default();
                    if paths.len() < options.max_paths.max(1) {
                        paths.push(partial.steps.clone());
                    }
                }
                backward_frontier = trim_frontier(next_frontier, options.beam_width);
                advanced = true;
            }

            if !advanced || bundle.witnesses.len() >= options.max_paths.max(1) {
                break;
            }
        }

        bundle.expansions = expansions;
        bundle.visited = visited_nodes.len();
        bundle.witnesses.sort_by(|left, right| {
            left.hop_count()
                .cmp(&right.hop_count())
                .then(right.confidence_bps.cmp(&left.confidence_bps))
                .then(left.key().cmp(&right.key()))
        });
        if bundle.witnesses.len() > options.max_paths.max(1) {
            bundle.witnesses.truncate(options.max_paths.max(1));
            bundle.truncated = true;
        }
        dedupe_blocked(&mut bundle);
        bundle
    }

    pub fn common_neighbors(
        &self,
        left: &str,
        right: &str,
        options: &TraversalOptions,
    ) -> PathBundle {
        let mut bundle = PathBundle {
            start: left.to_string(),
            goal: Some(right.to_string()),
            temporal: options.temporal.clone(),
            ..PathBundle::default()
        };
        if !self.known_nodes.contains(left) {
            bundle.blocked.push(BlockedPath {
                prefix: Vec::new(),
                at: left.to_string(),
                reason: BlockReason::UnknownNode {
                    node: left.to_string(),
                },
            });
        }
        if !self.known_nodes.contains(right) {
            bundle.blocked.push(BlockedPath {
                prefix: Vec::new(),
                at: right.to_string(),
                reason: BlockReason::UnknownNode {
                    node: right.to_string(),
                },
            });
        }
        if bundle.blocked.len() == 2 || left == right {
            return bundle;
        }

        let mut left_steps: Vec<PathStep> = Vec::new();
        for step in self.outgoing_all(left) {
            match self.step_block_reason(&step, options) {
                Some(reason) => self.push_blocked(&mut bundle, &[], reason),
                None => left_steps.push(step),
            }
        }
        let mut right_steps: Vec<PathStep> = Vec::new();
        for step in self.outgoing_all(right) {
            match self.step_block_reason(&step, options) {
                Some(reason) => self.push_blocked_at(&mut bundle, &[], right.to_string(), reason),
                None => right_steps.push(step),
            }
        }

        let mut reverse_right: BTreeMap<String, Vec<PathStep>> = BTreeMap::new();
        for step in &right_steps {
            if step.to == left || step.to == right {
                continue;
            }
            let reversed = PathStep {
                from: step.to.clone(),
                predicate: step.predicate.clone(),
                to: right.to_string(),
                inverted: !step.inverted,
                source: step.source,
                confidence_bps: step.confidence_bps,
                validity: step.validity,
            };
            reverse_right
                .entry(step.to.clone())
                .or_default()
                .push(reversed);
        }

        let mut witness_keys: BTreeSet<String> = BTreeSet::new();
        let mut visited_nodes: BTreeSet<String> =
            BTreeSet::from([left.to_string(), right.to_string()]);
        for left_step in &left_steps {
            if left_step.to == left || left_step.to == right {
                continue;
            }
            let Some(reverses) = reverse_right.get(&left_step.to) else {
                continue;
            };
            visited_nodes.insert(left_step.to.clone());
            for reverse in reverses {
                let witness = PathWitness::new(vec![left_step.clone(), reverse.clone()]);
                if !options.temporal_allows(witness.validity) {
                    self.push_blocked_at(
                        &mut bundle,
                        std::slice::from_ref(left_step),
                        left_step.to.clone(),
                        BlockReason::TemporalDisjoint {
                            from: left_step.from.clone(),
                            predicate: left_step.predicate.clone(),
                            to: right.to_string(),
                            requested: options.temporal.describe(),
                            valid: witness.validity.render(),
                        },
                    );
                    continue;
                }
                if witness_keys.insert(witness.key()) {
                    bundle.witnesses.push(witness);
                }
            }
        }

        bundle.expansions = left_steps.len() + right_steps.len();
        bundle.visited = visited_nodes.len();
        bundle.witnesses.sort_by(|left, right| {
            left.hop_count()
                .cmp(&right.hop_count())
                .then(left.key().cmp(&right.key()))
        });
        if bundle.witnesses.len() > options.max_paths.max(1) {
            bundle.witnesses.truncate(options.max_paths.max(1));
            bundle.truncated = true;
        }
        dedupe_blocked(&mut bundle);
        bundle
    }
}

fn path_nodes(start: &str, steps: &[PathStep]) -> BTreeSet<String> {
    let mut nodes = BTreeSet::from([start.to_string()]);
    for step in steps {
        nodes.insert(step.to.clone());
    }
    nodes
}

fn witness_prefix_key(start: &str, steps: &[PathStep]) -> String {
    let mut key = start.to_string();
    for step in steps {
        key.push_str(&format!("[{}:{}]->{};", step.predicate, step.inverted, step.to));
    }
    key
}

fn trim_frontier(mut frontier: Vec<PartialPath>, beam_width: usize) -> Vec<PartialPath> {
    frontier.truncate(beam_width.max(1));
    frontier
}

fn join_bidirectional(
    start: &str,
    goal: &str,
    forward: &[PathStep],
    backward: &[PathStep],
) -> Option<PathWitness> {
    let mut steps = forward.to_vec();
    for step in backward.iter().rev() {
        steps.push(PathStep {
            from: step.to.clone(),
            predicate: step.predicate.clone(),
            to: step.from.clone(),
            inverted: !step.inverted,
            source: step.source,
            confidence_bps: step.confidence_bps,
            validity: step.validity,
        });
    }
    let first = steps.first()?;
    let last = steps.last()?;
    if first.from != start || last.to != goal {
        return None;
    }
    let mut nodes = BTreeSet::from([start.to_string()]);
    for step in &steps {
        if !nodes.insert(step.to.clone()) {
            return None;
        }
    }
    for window in steps.windows(2) {
        if window[0].to != window[1].from {
            return None;
        }
    }
    Some(PathWitness::new(steps))
}

fn dedupe_blocked(bundle: &mut PathBundle) {
    let mut seen = BTreeSet::new();
    bundle.blocked.retain(|blocked| {
        seen.insert(format!(
            "{}|{}|{}",
            blocked.at,
            blocked.prefix.len(),
            blocked.reason.render()
        ))
    });
    if bundle.blocked.len() > MAX_BLOCKED_RECORDS {
        bundle.blocked.truncate(MAX_BLOCKED_RECORDS);
        bundle.truncated = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph() -> TraversalGraph {
        let mut graph = TraversalGraph::new();
        graph.add_edge("张三", "师傅", "李四");
        graph.add_edge("李四", "师傅", "王五");
        graph.add_edge("王五", "师傅", "赵六");
        graph
    }

    #[test]
    fn chain_keeps_each_branch_as_its_own_witness() {
        let mut graph = TraversalGraph::new();
        graph.add_edge("张三", "师傅", "李四");
        graph.add_edge("张三", "师傅", "王五");
        graph.add_edge("李四", "师傅", "赵六");
        graph.add_edge("王五", "师傅", "钱七");
        let bundle = graph.enumerate_chain(
            "张三",
            &[PathPredicate::forward("师傅"), PathPredicate::forward("师傅")],
            &TraversalOptions::default(),
        );
        assert_eq!(bundle.witnesses.len(), 2);
        let answers: Vec<&str> = bundle
            .witnesses
            .iter()
            .filter_map(|witness| witness.end())
            .collect();
        assert!(answers.contains(&"赵六"));
        assert!(answers.contains(&"钱七"));
        for witness in &bundle.witnesses {
            let nodes: Vec<&str> = std::iter::once(witness.start().unwrap())
                .chain(witness.steps.iter().map(|step| step.to.as_str()))
                .collect();
            assert!(
                !(nodes.contains(&"赵六") && nodes.contains(&"钱七")),
                "不同分支不得拼成一条路径：{:?}",
                nodes
            );
        }
    }

    #[test]
    fn chain_groups_multiple_witnesses_by_answer() {
        let mut graph = TraversalGraph::new();
        graph.add_edge("张三", "师傅", "李四");
        graph.add_edge("张三", "师傅", "王五");
        graph.add_edge("李四", "师傅", "赵六");
        graph.add_edge("王五", "师傅", "赵六");
        let bundle = graph.enumerate_chain(
            "张三",
            &[PathPredicate::forward("师傅"), PathPredicate::forward("师傅")],
            &TraversalOptions::default(),
        );
        let groups = bundle.answer_groups();
        assert_eq!(groups.get("赵六").map(Vec::len), Some(2));
    }

    #[test]
    fn chain_reports_missing_edge_and_negative_blocker() {
        let mut graph = TraversalGraph::new();
        graph.add_edge("张三", "师傅", "李四");
        graph.add_negative_edge("张三", "师傅", "王五");
        let bundle = graph.enumerate_chain(
            "张三",
            &[PathPredicate::forward("师傅"), PathPredicate::forward("师傅")],
            &TraversalOptions::default(),
        );
        assert!(bundle.witnesses.is_empty());
        assert!(bundle
            .blocked
            .iter()
            .any(|blocked| matches!(blocked.reason, BlockReason::MissingEdge { .. })));
        assert!(bundle
            .blocked
            .iter()
            .any(|blocked| matches!(blocked.reason, BlockReason::NegatedEdge { .. })));
    }

    #[test]
    fn negated_edge_is_not_traversed_even_if_positive_exists() {
        let mut graph = TraversalGraph::new();
        graph.add_edge("张三", "师傅", "李四");
        graph.add_edge("张三", "师傅", "王五");
        graph.add_negative_edge("张三", "师傅", "王五");
        let bundle = graph.enumerate_chain(
            "张三",
            &[PathPredicate::forward("师傅")],
            &TraversalOptions::default(),
        );
        assert_eq!(bundle.witnesses.len(), 1);
        assert_eq!(bundle.witnesses[0].end(), Some("李四"));
        assert!(bundle
            .blocked
            .iter()
            .any(|blocked| matches!(blocked.reason, BlockReason::NegatedEdge { .. })));
    }

    #[test]
    fn between_routes_around_negated_edge() {
        let mut graph = TraversalGraph::new();
        graph.add_edge("甲", "认识", "乙");
        graph.add_edge("乙", "认识", "丁");
        graph.add_edge("甲", "认识", "丙");
        graph.add_edge("丙", "认识", "丁");
        graph.add_negative_edge("乙", "认识", "丁");
        let bundle = graph.enumerate_between(
            "甲",
            "丁",
            &TraversalOptions {
                max_depth: 3,
                max_paths: 8,
                ..TraversalOptions::default()
            },
        );
        assert_eq!(bundle.witnesses.len(), 1);
        assert!(bundle.witnesses[0].steps.iter().all(|step| step.to != "乙"));
        assert!(bundle
            .blocked
            .iter()
            .any(|blocked| matches!(blocked.reason, BlockReason::NegatedEdge { .. })));
    }

    #[test]
    fn confidence_and_source_constraints_filter_edges() {
        let mut graph = TraversalGraph::new();
        graph.add_edge_record(
            PathEdge::new("张三", "师傅", "李四")
                .with_source(EdgeSource::Custom)
                .with_confidence_bps(5_000),
        );
        let low_confidence = graph.enumerate_chain(
            "张三",
            &[PathPredicate::forward("师傅")],
            &TraversalOptions {
                min_confidence_bps: 8_000,
                ..TraversalOptions::default()
            },
        );
        assert!(low_confidence.witnesses.is_empty());
        assert!(low_confidence
            .blocked
            .iter()
            .any(|blocked| matches!(blocked.reason, BlockReason::LowConfidence { .. })));

        let source_filtered = graph.enumerate_chain(
            "张三",
            &[PathPredicate::forward("师傅")],
            &TraversalOptions {
                allowed_sources: Some(BTreeSet::from([EdgeSource::Observed])),
                ..TraversalOptions::default()
            },
        );
        assert!(source_filtered.witnesses.is_empty());
        assert!(source_filtered
            .blocked
            .iter()
            .any(|blocked| matches!(blocked.reason, BlockReason::SourceNotAllowed { .. })));
    }

    #[test]
    fn expected_end_type_prunes_terminal_values() {
        let mut graph = TraversalGraph::new();
        graph.add_edge("张三", "师傅", "李四");
        graph.add_edge("张三", "师傅", "某公司");
        graph.add_entity_type("李四", "Person");
        graph.add_entity_type("某公司", "Organization");
        let bundle = graph.enumerate_chain(
            "张三",
            &[PathPredicate::forward("师傅")],
            &TraversalOptions {
                expected_end_types: BTreeSet::from(["Person".to_string()]),
                ..TraversalOptions::default()
            },
        );
        assert_eq!(bundle.witnesses.len(), 1);
        assert_eq!(bundle.witnesses[0].end(), Some("李四"));
        assert!(bundle
            .blocked
            .iter()
            .any(|blocked| matches!(blocked.reason, BlockReason::ExpectedTypeMismatch { .. })));
    }

    #[test]
    fn common_neighbors_skip_negated_hub() {
        let mut graph = TraversalGraph::new();
        graph.add_edge("张三", "认识", "王五");
        graph.add_edge("李四", "认识", "王五");
        graph.add_negative_edge("张三", "认识", "王五");
        let bundle = graph.common_neighbors("张三", "李四", &TraversalOptions::default());
        assert!(bundle.witnesses.is_empty());
        assert!(bundle
            .blocked
            .iter()
            .any(|blocked| matches!(blocked.reason, BlockReason::NegatedEdge { .. })));
    }

    #[test]
    fn inverted_link_walks_against_edge_direction() {
        let mut graph = TraversalGraph::new();
        graph.add_edge("李四", "师傅", "张三");
        let bundle = graph.enumerate_chain(
            "张三",
            &[PathPredicate::inverted("师傅")],
            &TraversalOptions::default(),
        );
        assert_eq!(bundle.witnesses.len(), 1);
        assert_eq!(bundle.witnesses[0].end(), Some("李四"));
        assert!(bundle.witnesses[0].steps[0].inverted);
    }

    #[test]
    fn between_enumerates_alternate_paths() {
        let mut graph = TraversalGraph::new();
        graph.add_edge("甲", "认识", "乙");
        graph.add_edge("乙", "认识", "丁");
        graph.add_edge("甲", "认识", "丙");
        graph.add_edge("丙", "认识", "丁");
        graph.add_edge("甲", "认识", "丁");
        let bundle = graph.enumerate_between(
            "甲",
            "丁",
            &TraversalOptions {
                max_depth: 3,
                max_paths: 8,
                ..TraversalOptions::default()
            },
        );
        assert_eq!(bundle.witnesses.len(), 3);
        assert_eq!(bundle.witnesses[0].hop_count(), 1);
        assert!(bundle
            .witnesses
            .iter()
            .all(|witness| witness.start() == Some("甲") && witness.end() == Some("丁")));
    }

    #[test]
    fn between_reports_unknown_endpoints() {
        let graph = graph();
        let bundle = graph.enumerate_between(
            "不存在的人",
            "王五",
            &TraversalOptions::default(),
        );
        assert!(bundle.witnesses.is_empty());
        assert!(bundle
            .blocked
            .iter()
            .any(|blocked| matches!(blocked.reason, BlockReason::UnknownNode { .. })));
    }

    #[test]
    fn common_neighbors_finds_the_shared_hub() {
        let mut graph = TraversalGraph::new();
        graph.add_edge("张三", "认识", "王五");
        graph.add_edge("李四", "认识", "王五");
        graph.add_edge("张三", "认识", "赵六");
        let bundle = graph.common_neighbors(
            "张三",
            "李四",
            &TraversalOptions::default(),
        );
        assert_eq!(bundle.witnesses.len(), 1);
        let groups = bundle.intermediary_groups();
        assert_eq!(groups.keys().cloned().collect::<Vec<_>>(), vec!["王五"]);
    }

    #[test]
    fn predicate_filter_blocks_unlisted_edges() {
        let graph = graph();
        let options = TraversalOptions {
            allowed_predicates: Some(BTreeSet::from(["别的关系".to_string()])),
            ..TraversalOptions::default()
        };
        let bundle = graph.enumerate_chain(
            "张三",
            &[PathPredicate::forward("师傅")],
            &options,
        );
        assert!(bundle.witnesses.is_empty());
        assert!(bundle
            .blocked
            .iter()
            .any(|blocked| matches!(blocked.reason, BlockReason::PredicateNotAllowed { .. })));
    }

    #[test]
    fn witness_render_marks_session_source_and_low_confidence() {
        let mut graph = TraversalGraph::new();
        graph.add_edge_record(
            PathEdge::new("张三", "师傅", "李四")
                .with_source(EdgeSource::Session)
                .with_confidence_bps(8_500),
        );
        let bundle = graph.enumerate_chain(
            "张三",
            &[PathPredicate::forward("师傅")],
            &TraversalOptions::default(),
        );
        let text = bundle.witnesses[0].render();
        assert!(text.contains("师傅(会话，置信度 85%)"), "实际：{}", text);
    }

    #[test]
    fn witness_render_is_readable_and_deterministic() {
        let graph = graph();
        let bundle = graph.enumerate_chain(
            "张三",
            &[PathPredicate::forward("师傅"), PathPredicate::forward("师傅")],
            &TraversalOptions::default(),
        );
        let text = bundle.witnesses[0].render();
        assert!(text.contains("张三 --师傅--> 李四 --师傅--> 王五"));
        assert_eq!(text, bundle.witnesses[0].render());
    }

    #[test]
    fn as_of_blocks_cross_time_false_multi_hop() {
        let mut graph = TraversalGraph::new();
        graph.add_edge_record(PathEdge::new("张三", "师傅", "李四").with_years(1990, 1999));
        graph.add_edge_record(PathEdge::new("李四", "师傅", "王五").with_years(2000, 2010));
        let bundle = graph.enumerate_chain(
            "张三",
            &[PathPredicate::forward("师傅"), PathPredicate::forward("师傅")],
            &TraversalOptions::default().with_temporal(TemporalConstraint::as_of(1995)),
        );
        assert!(bundle.witnesses.is_empty());
        assert!(bundle.blocked.iter().any(|blocked| matches!(
            blocked.reason,
            BlockReason::TemporalDisjoint { .. }
        )));
    }

    #[test]
    fn as_of_keeps_path_whose_steps_share_the_year() {
        let mut graph = TraversalGraph::new();
        graph.add_edge_record(PathEdge::new("张三", "师傅", "李四").with_years(1990, 2000));
        graph.add_edge_record(PathEdge::new("李四", "师傅", "王五").with_years(1995, 1999));
        let bundle = graph.enumerate_chain(
            "张三",
            &[PathPredicate::forward("师傅"), PathPredicate::forward("师傅")],
            &TraversalOptions::default().with_temporal(TemporalConstraint::as_of(1995)),
        );
        assert_eq!(bundle.witnesses.len(), 1);
        assert_eq!(bundle.witnesses[0].end(), Some("王五"));
        assert_eq!(bundle.witnesses[0].validity, Validity::from_years(1995, 1999));
        assert!(bundle.witnesses[0].has_common_time());
    }

    #[test]
    fn unknown_time_is_strict_by_default_but_can_be_background() {
        let mut graph = TraversalGraph::new();
        graph.add_edge("张三", "师傅", "李四");
        let strict = graph.enumerate_chain(
            "张三",
            &[PathPredicate::forward("师傅")],
            &TraversalOptions::default().with_temporal(TemporalConstraint::as_of(1995)),
        );
        assert!(strict.witnesses.is_empty());
        assert!(strict.blocked.iter().any(|blocked| matches!(
            blocked.reason,
            BlockReason::TemporalDisjoint { .. }
        )));

        let background = graph.enumerate_chain(
            "张三",
            &[PathPredicate::forward("师傅")],
            &TraversalOptions::default()
                .with_temporal(TemporalConstraint::as_of(1995))
                .with_unknown_time_policy(UnknownTimePolicy::Background),
        );
        assert_eq!(background.witnesses.len(), 1);
        assert!(background.witnesses[0].time_uncertain());
    }

    #[test]
    fn during_rejects_path_without_common_intersection() {
        let mut graph = TraversalGraph::new();
        graph.add_edge_record(PathEdge::new("甲", "认识", "乙").with_years(1990, 1995));
        graph.add_edge_record(PathEdge::new("乙", "认识", "丙").with_years(1994, 2000));
        graph.add_edge_record(PathEdge::new("丙", "认识", "丁").with_years(1999, 2005));
        let bundle = graph.enumerate_chain(
            "甲",
            &[
                PathPredicate::forward("认识"),
                PathPredicate::forward("认识"),
                PathPredicate::forward("认识"),
            ],
            &TraversalOptions::default().with_temporal(TemporalConstraint::during(
                TimeSpan::from_bounds(
                    Some(crate::temporal::TimePoint::year(1990)),
                    Some(crate::temporal::TimePoint::year(2005)),
                    crate::temporal::TimeGranularity::Year,
                )
                .unwrap(),
            )),
        );
        assert!(
            bundle.witnesses.is_empty(),
            "每条边都与查询区间相交，但三步没有共同时间点：{:?}",
            bundle.witnesses
        );
        assert!(bundle.blocked.iter().any(|blocked| matches!(
            blocked.reason,
            BlockReason::TemporalDisjoint { .. }
        )));
    }

    #[test]
    fn timed_edge_helper_parses_chinese_validity() {
        let mut graph = TraversalGraph::new();
        graph.add_timed_edge("张三", "师傅", "李四", "1990年到1999年");
        let bundle = graph.enumerate_chain(
            "张三",
            &[PathPredicate::forward("师傅")],
            &TraversalOptions::default().with_temporal(TemporalConstraint::as_of(1995)),
        );
        assert_eq!(bundle.witnesses.len(), 1);
        assert!(bundle.witnesses[0].render().contains("有效期 1990年"));
    }
}

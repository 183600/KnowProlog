//! 图间身份对齐：用邻域结构指纹发现跨来源的同一实体。
//!
//! `knowledge_fusion` 假设不同来源已经使用相同的词面，`ontology_alignment`
//! 的结构通道则只在单张全局术语表上比较一跳邻居。本模块补上两者之间缺失的
//! 一层：按命名空间隔离构图，用 Weisfeiler–Lehman 式的颜色迭代把「词面不同、
//! 邻域结构相同」的实体放进同一候选桶，再用类型门、互惠最优、领先幅度与
//! 度数门槛把候选收紧成可审计的身份证书。
//!
//! 核心约定：
//!
//! - **结构指纹不看词面**：颜色只由类型门之外的有向边（谓词 + 邻居颜色）与
//!   字面量谓词集合决定，因此 `苹果公司` 与 `Apple Inc.` 可以同桶；
//! - **类型门最后裁决**：类型集合不参与颜色；两边类型非空且不等时直接拒绝，
//!   缺类型时最多停在提案，绝不自动合并；
//! - **互惠最优**：只有双方都把对方排在第一才可能接受，且领先第二候选必须
//!   达到 `accept_margin_bps`；同分候选一律进提案，不偷偷挑一个；
//! - **枢纽截断标记**：邻居超过 `hub_cap` 时按稳定键截断并记录，带截断的
//!   证书最多是提案；
//! - **确定性**：所有遍历按命名空间优先级、词面与稳定哈希排序，同输入同字节。

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::coref_closure::{IdentityEdge, IdentityKind};
use crate::knowledge_fusion::{
    sanitize_namespace_id, FusionLedger, NamespaceRecord, ObjectKind, ScopedAssertion,
};

/// 图间身份对齐报告方言。
pub const GRAPH_IDENTITY_DIALECT: &str = "knowprolog-graph-identity/1";
/// 没有第二候选时的领先幅度（basis point）。
pub const UNCONTESTED_MARGIN_BPS: u16 = 10_000;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
const TYPE_PREDICATE: &str = "type";

/// 对齐选项；默认值偏保守（两轮指纹、要求互惠最优、至少一条实体边）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AlignmentOptions {
    /// 颜色迭代轮数（不含初始颜色）。
    pub rounds: usize,
    /// 至少需要一致的颜色轮数。
    pub min_matched_rounds: usize,
    /// 每个方向保留的邻居上限；超出即标记截断。
    pub hub_cap: usize,
    /// 可接受实体至少需要的非类型边数。
    pub min_degree: usize,
    /// 每个实体保留的候选上限。
    pub max_candidates: usize,
    /// 单个颜色桶允许的最大节点数；更大的桶整体跳过。
    pub max_bucket_nodes: usize,
    /// 最佳候选领先第二候选的最小幅度。
    pub accept_margin_bps: u16,
    /// 接受所需的最低置信度。
    pub min_confidence_bps: u16,
}

impl Default for AlignmentOptions {
    fn default() -> Self {
        Self {
            rounds: 2,
            min_matched_rounds: 2,
            hub_cap: 16,
            min_degree: 1,
            max_candidates: 8,
            max_bucket_nodes: 256,
            accept_margin_bps: 1_000,
            min_confidence_bps: 7_000,
        }
    }
}

impl AlignmentOptions {
    pub fn with_rounds(mut self, rounds: usize) -> Self {
        self.rounds = rounds.clamp(1, 6);
        self.min_matched_rounds = self.min_matched_rounds.min(self.rounds + 1);
        self
    }

    pub fn with_min_confidence_bps(mut self, value: u16) -> Self {
        self.min_confidence_bps = value.min(10_000);
        self
    }

    pub fn with_accept_margin_bps(mut self, value: u16) -> Self {
        self.accept_margin_bps = value.min(10_000);
        self
    }
}

/// 证书状态：可接受、需确认、已拒绝。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlignmentStatus {
    Accepted,
    Proposed,
    Rejected,
}

impl AlignmentStatus {
    pub fn label_zh(self) -> &'static str {
        match self {
            AlignmentStatus::Accepted => "接受",
            AlignmentStatus::Proposed => "提案",
            AlignmentStatus::Rejected => "拒绝",
        }
    }
}

/// 阻止自动接受的具体原因；顺序固定，便于逐字节比较。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlignmentBlocker {
    TypeConflict,
    MissingTypes,
    InsufficientRounds,
    HubTruncated,
    Sparse,
    AmbiguousMargin,
    NonReciprocal,
    LowConfidence,
}

impl AlignmentBlocker {
    pub fn label_zh(self) -> &'static str {
        match self {
            AlignmentBlocker::TypeConflict => "类型互斥",
            AlignmentBlocker::MissingTypes => "缺少类型",
            AlignmentBlocker::InsufficientRounds => "结构轮数不足",
            AlignmentBlocker::HubTruncated => "枢纽邻居被截断",
            AlignmentBlocker::Sparse => "结构过于稀疏",
            AlignmentBlocker::AmbiguousMargin => "领先幅度不足",
            AlignmentBlocker::NonReciprocal => "非互惠最优",
            AlignmentBlocker::LowConfidence => "置信度不足",
        }
    }

    fn is_hard(self) -> bool {
        matches!(
            self,
            AlignmentBlocker::TypeConflict | AlignmentBlocker::InsufficientRounds
        )
    }
}

/// 一张可审计的身份证书：为什么把两个来源中的两个词面视为同一实体。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AlignmentCertificate {
    pub from_scope: String,
    pub from: String,
    pub to_scope: String,
    pub to: String,
    pub status: AlignmentStatus,
    pub canonical: String,
    pub alias: String,
    pub matched_rounds: usize,
    pub total_rounds: usize,
    pub confidence_bps: u16,
    pub margin_bps: u16,
    pub from_types: Vec<String>,
    pub to_types: Vec<String>,
    pub shared_predicates: Vec<String>,
    pub from_degree: usize,
    pub to_degree: usize,
    pub from_truncated: bool,
    pub to_truncated: bool,
    pub fingerprint: String,
    pub blockers: Vec<AlignmentBlocker>,
}

impl AlignmentCertificate {
    pub fn is_accepted(&self) -> bool {
        self.status == AlignmentStatus::Accepted
    }

    pub fn render_zh(&self) -> String {
        let mut lines = vec![format!(
            "[{}] {}:{} ≈ {}:{} → 规范「{}」",
            self.status.label_zh(),
            self.from_scope,
            self.from,
            self.to_scope,
            self.to,
            self.canonical
        )];
        let from_types = if self.from_types.is_empty() {
            "无".to_string()
        } else {
            self.from_types.join("、")
        };
        let to_types = if self.to_types.is_empty() {
            "无".to_string()
        } else {
            self.to_types.join("、")
        };
        lines.push(format!(
            "  类型：{} / {}；结构：{}/{} 轮一致；指纹 {}",
            from_types,
            to_types,
            self.matched_rounds,
            self.total_rounds,
            self.fingerprint
        ));
        let shared = if self.shared_predicates.is_empty() {
            "无".to_string()
        } else {
            self.shared_predicates.join("、")
        };
        lines.push(format!(
            "  度：{} / {}；共享谓词：{}；置信 {}/10000；领先 {} bp",
            self.from_degree, self.to_degree, shared, self.confidence_bps, self.margin_bps
        ));
        if self.from_truncated || self.to_truncated {
            lines.push("  注意：至少一侧的枢纽邻居被截断，证据不完整".to_string());
        }
        if !self.blockers.is_empty() {
            let reasons = self
                .blockers
                .iter()
                .map(|blocker| blocker.label_zh())
                .collect::<Vec<_>>()
                .join("、");
            lines.push(format!("  未决原因：{}", reasons));
        }
        lines.join("\n")
    }
}

/// 对齐计划：证书、统计与可复用的投影映射。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AlignmentPlan {
    pub dialect: String,
    pub certificates: Vec<AlignmentCertificate>,
    pub scopes: usize,
    pub nodes: usize,
    pub candidates: usize,
    pub accepted: usize,
    pub proposed: usize,
    pub rejected: usize,
    pub oversized_buckets: usize,
    #[serde(default)]
    scope_order: Vec<String>,
}

impl AlignmentPlan {
    pub fn accepted(&self) -> impl Iterator<Item = &AlignmentCertificate> {
        self.certificates
            .iter()
            .filter(|certificate| certificate.is_accepted())
    }

    pub fn proposed(&self) -> impl Iterator<Item = &AlignmentCertificate> {
        self.certificates
            .iter()
            .filter(|certificate| certificate.status == AlignmentStatus::Proposed)
    }

    pub fn rejected(&self) -> impl Iterator<Item = &AlignmentCertificate> {
        self.certificates
            .iter()
            .filter(|certificate| certificate.status == AlignmentStatus::Rejected)
    }

    /// 已接受证书的传递投影：`(命名空间, 词面) → 规范词面`。
    ///
    /// 同一等价类选命名空间优先级最高（信任 → 优先级 → id）的词面作为规范名；
    /// 完全同优先级时再按词面字典序破平，保证结果可复现。
    pub fn projection_map(&self) -> BTreeMap<(String, String), String> {
        let mut adjacency: BTreeMap<(String, String), BTreeSet<(String, String)>> = BTreeMap::new();
        for certificate in self.accepted() {
            let left = (certificate.from_scope.clone(), certificate.from.clone());
            let right = (certificate.to_scope.clone(), certificate.to.clone());
            adjacency.entry(left.clone()).or_default().insert(right.clone());
            adjacency.entry(right).or_default().insert(left);
        }
        let scope_rank: BTreeMap<&str, usize> = self
            .scope_order
            .iter()
            .enumerate()
            .map(|(index, scope)| (scope.as_str(), index))
            .collect();
        let mut mapping = BTreeMap::new();
        let mut visited: BTreeSet<(String, String)> = BTreeSet::new();
        for start in adjacency.keys() {
            if visited.contains(start) {
                continue;
            }
            let mut component = Vec::new();
            let mut queue = vec![start.clone()];
            visited.insert(start.clone());
            while let Some(current) = queue.pop() {
                component.push(current.clone());
                if let Some(neighbors) = adjacency.get(&current) {
                    for neighbor in neighbors {
                        if visited.insert(neighbor.clone()) {
                            queue.push(neighbor.clone());
                        }
                    }
                }
            }
            component.sort_by(|left, right| {
                scope_rank
                    .get(left.0.as_str())
                    .copied()
                    .unwrap_or(usize::MAX)
                    .cmp(
                        &scope_rank
                            .get(right.0.as_str())
                            .copied()
                            .unwrap_or(usize::MAX),
                    )
                    .then_with(|| left.1.cmp(&right.1))
                    .then_with(|| left.0.cmp(&right.0))
            });
            let canonical = component
                .first()
                .map(|(_, surface)| surface.clone())
                .unwrap_or_default();
            for key in component {
                mapping.insert(key, canonical.clone());
            }
        }
        mapping
    }

    /// 把断言中的实体词面投影到规范词面；字面量宾语保持原样。
    ///
    /// 返回结果按断言 id 排序去重，可直接交给融合层预览。
    pub fn project_assertions(&self, assertions: &[ScopedAssertion]) -> Vec<ScopedAssertion> {
        let mapping = self.projection_map();
        let mut projected = Vec::new();
        for assertion in assertions {
            let scope = sanitize_namespace_id(&assertion.namespace);
            let subject = mapping
                .get(&(scope.clone(), assertion.subject.clone()))
                .cloned()
                .unwrap_or_else(|| assertion.subject.clone());
            let object = match assertion.object_kind {
                ObjectKind::Entity => mapping
                    .get(&(scope, assertion.object.clone()))
                    .cloned()
                    .unwrap_or_else(|| assertion.object.clone()),
                ObjectKind::Literal => assertion.object.clone(),
            };
            projected.push(ScopedAssertion::new(
                &assertion.namespace,
                &subject,
                &assertion.predicate,
                &object,
                assertion.object_kind,
                &assertion.source,
            ));
        }
        projected.sort_by(|left, right| left.id.cmp(&right.id));
        projected.dedup_by(|left, right| left.id == right.id);
        projected
    }

    /// 已接受证书导出的 `sameAs` 身份边，可直接交给 `CorefGraph`。
    pub fn identity_edges(&self) -> Vec<IdentityEdge> {
        let mut edges = BTreeSet::new();
        for certificate in self.accepted() {
            if certificate.alias == certificate.canonical {
                continue;
            }
            edges.insert((
                certificate.alias.clone(),
                certificate.canonical.clone(),
                certificate.confidence_bps,
                certificate.source_label(),
            ));
        }
        edges
            .into_iter()
            .map(|(alias, canonical, confidence, source)| {
                IdentityEdge::new(
                    &alias,
                    &canonical,
                    IdentityKind::Structural,
                    &canonical,
                )
                .with_source(&source)
                .with_confidence(f64::from(confidence) / 10_000.0)
            })
            .collect()
    }

    /// 可写入别名册的 `(别名, 规范名)` 对（去重、稳定排序）。
    pub fn aliases(&self) -> Vec<(String, String)> {
        let mut pairs = BTreeSet::new();
        for ((_, surface), canonical) in self.projection_map() {
            if surface != canonical {
                pairs.insert((surface, canonical));
            }
        }
        pairs.into_iter().collect()
    }

    pub fn render_zh(&self) -> String {
        let mut lines = vec![format!(
            "图间身份对齐：命名空间 {} / 实体 {} / 候选 {} / 接受 {} / 提案 {} / 拒绝 {}",
            self.scopes,
            self.nodes,
            self.candidates,
            self.accepted,
            self.proposed,
            self.rejected
        )];
        if self.oversized_buckets > 0 {
            lines.push(format!(
                "  {} 个结构桶超过规模上限，已整体跳过（保守退化）",
                self.oversized_buckets
            ));
        }
        let mut certificates = self.certificates.clone();
        certificates.sort_by(|left, right| {
            left.status
                .cmp(&right.status)
                .then_with(|| left.from_scope.cmp(&right.from_scope))
                .then_with(|| left.from.cmp(&right.from))
                .then_with(|| left.to_scope.cmp(&right.to_scope))
                .then_with(|| left.to.cmp(&right.to))
        });
        for certificate in certificates {
            lines.push(certificate.render_zh());
        }
        lines.join("\n")
    }
}

impl AlignmentCertificate {
    fn source_label(&self) -> String {
        format!("结构对齐 {}→{}", self.from_scope, self.to_scope)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct NodeKey {
    scope: String,
    surface: String,
}

impl NodeKey {
    fn new(scope: &str, surface: &str) -> Self {
        Self {
            scope: scope.to_string(),
            surface: surface.to_string(),
        }
    }
}

#[derive(Debug, Default)]
struct NodeBuild {
    types: BTreeSet<String>,
    out_entity: Vec<(String, NodeKey)>,
    in_entity: Vec<(String, NodeKey)>,
    out_literal: Vec<String>,
}

impl NodeBuild {
    fn degree(&self) -> usize {
        self.out_entity.len() + self.in_entity.len() + self.out_literal.len()
    }

    fn predicates(&self) -> BTreeSet<String> {
        let mut predicates = BTreeSet::new();
        for (predicate, _) in &self.out_entity {
            predicates.insert(predicate.clone());
        }
        for (predicate, _) in &self.in_entity {
            predicates.insert(predicate.clone());
        }
        for predicate in &self.out_literal {
            predicates.insert(predicate.clone());
        }
        predicates
    }

    fn shared_predicates(&self, other: &Self) -> Vec<String> {
        self.predicates()
            .intersection(&other.predicates())
            .cloned()
            .collect()
    }
}

#[derive(Debug, Clone)]
struct Fingerprint {
    colors: Vec<u64>,
    truncated: bool,
}

impl Fingerprint {
    fn trace(&self) -> String {
        self.colors
            .iter()
            .map(|color| format!("{:016x}", color))
            .collect::<Vec<_>>()
            .join(":")
    }

    fn final_color(&self) -> u64 {
        self.colors.last().copied().unwrap_or(0)
    }
}

#[derive(Debug, Clone)]
struct Candidate {
    from: NodeKey,
    to: NodeKey,
    matched_rounds: usize,
    confidence_bps: u16,
}

impl Candidate {
    fn reversed(&self) -> Self {
        Self {
            from: self.to.clone(),
            to: self.from.clone(),
            matched_rounds: self.matched_rounds,
            confidence_bps: self.confidence_bps,
        }
    }
}

#[derive(Debug, Default)]
struct ChosenPair {
    first: Option<Candidate>,
    second: Option<Candidate>,
}

/// 从命名空间与断言构建对齐计划。
pub fn align_assertions(
    namespaces: &[NamespaceRecord],
    assertions: &[ScopedAssertion],
    options: &AlignmentOptions,
) -> AlignmentPlan {
    let mut active: Vec<&NamespaceRecord> = namespaces
        .iter()
        .filter(|record| record.participates(false))
        .collect();
    active.sort_by(|left, right| left.precedence().cmp(&right.precedence()));
    let scope_order: Vec<String> = active.iter().map(|record| record.id.clone()).collect();
    let active_ids: BTreeSet<&str> = active.iter().map(|record| record.id.as_str()).collect();

    let mut nodes: BTreeMap<NodeKey, NodeBuild> = BTreeMap::new();
    let mut seen_assertions: BTreeSet<&str> = BTreeSet::new();
    for assertion in assertions {
        if !seen_assertions.insert(assertion.id.as_str()) {
            continue;
        }
        if !active_ids.contains(assertion.namespace.as_str()) {
            continue;
        }
        let scope = &assertion.namespace;
        let subject = NodeKey::new(scope, &assertion.subject);
        nodes.entry(subject.clone()).or_default();
        if assertion.predicate == TYPE_PREDICATE {
            if let Some(node) = nodes.get_mut(&subject) {
                node.types.insert(assertion.object.clone());
            }
            continue;
        }
        match assertion.object_kind {
            ObjectKind::Entity => {
                let object = NodeKey::new(scope, &assertion.object);
                nodes.entry(object.clone()).or_default();
                if let Some(node) = nodes.get_mut(&subject) {
                    node.out_entity
                        .push((assertion.predicate.clone(), object.clone()));
                }
                if let Some(node) = nodes.get_mut(&object) {
                    node.in_entity
                        .push((assertion.predicate.clone(), subject.clone()));
                }
            }
            ObjectKind::Literal => {
                if let Some(node) = nodes.get_mut(&subject) {
                    node.out_literal.push(assertion.predicate.clone());
                }
            }
        }
    }

    let fingerprints = fingerprint_nodes(&nodes, options);
    let scope_rank = scope_order
        .iter()
        .enumerate()
        .map(|(index, scope)| (scope.clone(), index))
        .collect::<BTreeMap<_, _>>();

    let mut buckets: BTreeMap<String, Vec<NodeKey>> = BTreeMap::new();
    for (key, fingerprint) in &fingerprints {
        buckets
            .entry(fingerprint.trace())
            .or_default()
            .push(key.clone());
    }

    let mut per_node: BTreeMap<NodeKey, Vec<Candidate>> = BTreeMap::new();
    let mut oversized_buckets = 0usize;
    let mut candidate_count = 0usize;
    for (_trace, mut bucket) in buckets {
        if bucket.len() > options.max_bucket_nodes {
            oversized_buckets += 1;
            continue;
        }
        bucket.sort_by(|left, right| compare_nodes(left, right, &scope_rank));
        for (index, left) in bucket.iter().enumerate() {
            for right in bucket.iter().skip(index + 1) {
                if left.scope == right.scope || left.surface == right.surface {
                    continue;
                }
                let (left_build, right_build) = match (nodes.get(left), nodes.get(right)) {
                    (Some(left_build), Some(right_build)) => (left_build, right_build),
                    _ => continue,
                };
                let matched_rounds =
                    matched_rounds(&fingerprints[left], &fingerprints[right]);
                let confidence = confidence_bps(
                    left_build,
                    right_build,
                    matched_rounds,
                    options,
                );
                let candidate = Candidate {
                    from: left.clone(),
                    to: right.clone(),
                    matched_rounds,
                    confidence_bps: confidence,
                };
                per_node
                    .entry(left.clone())
                    .or_default()
                    .push(candidate.clone());
                per_node
                    .entry(right.clone())
                    .or_default()
                    .push(candidate.reversed());
                candidate_count += 2;
            }
        }
    }

    let mut best: BTreeMap<NodeKey, (Candidate, u16)> = BTreeMap::new();
    for (key, mut candidates) in per_node {
        candidates.sort_by(|left, right| compare_candidates(left, right, &scope_rank));
        candidates.truncate(options.max_candidates);
        let Some(first) = candidates.first().cloned() else {
            continue;
        };
        let margin = candidates
            .get(1)
            .map(|second| first.confidence_bps.saturating_sub(second.confidence_bps))
            .unwrap_or(UNCONTESTED_MARGIN_BPS);
        best.insert(key, (first, margin));
    }

    let mut chosen: BTreeMap<(NodeKey, NodeKey), ChosenPair> = BTreeMap::new();
    for (key, (candidate, _)) in &best {
        let (first, second) = if compare_nodes(key, &candidate.to, &scope_rank).is_lt() {
            (key.clone(), candidate.to.clone())
        } else {
            (candidate.to.clone(), key.clone())
        };
        let slot = chosen.entry((first.clone(), second.clone())).or_default();
        if &first == key {
            slot.first = Some(candidate.clone());
        } else {
            slot.second = Some(candidate.clone());
        }
    }

    let mut certificates = Vec::new();
    for ((first, second), slot) in chosen {
        let (source, target, candidate) = if let Some(candidate) = slot.first.clone() {
            (first.clone(), second.clone(), candidate)
        } else if let Some(candidate) = slot.second.clone() {
            (second.clone(), first.clone(), candidate)
        } else {
            continue;
        };
        let reverse_ok = if source == first {
            slot.second
                .as_ref()
                .map(|reverse| reverse.to == source)
                .unwrap_or(false)
        } else {
            slot.first
                .as_ref()
                .map(|reverse| reverse.to == source)
                .unwrap_or(false)
        };
        let source_margin = best
            .get(&source)
            .map(|(_, margin)| *margin)
            .unwrap_or(UNCONTESTED_MARGIN_BPS);
        let target_margin = best
            .get(&target)
            .map(|(_, margin)| *margin)
            .unwrap_or(UNCONTESTED_MARGIN_BPS);
        let source_build = nodes.get(&source).expect("源节点已构建");
        let target_build = nodes.get(&target).expect("目标节点已构建");
        let source_fingerprint = &fingerprints[&source];
        let target_fingerprint = &fingerprints[&target];

        let mut blockers = Vec::new();
        let types_equal = !source_build.types.is_empty()
            && source_build.types == target_build.types;
        if !types_equal {
            if source_build.types.is_empty() || target_build.types.is_empty() {
                blockers.push(AlignmentBlocker::MissingTypes);
            } else {
                blockers.push(AlignmentBlocker::TypeConflict);
            }
        }
        let matched_rounds = candidate.matched_rounds;
        if matched_rounds < options.min_matched_rounds {
            blockers.push(AlignmentBlocker::InsufficientRounds);
        }
        if source_fingerprint.truncated || target_fingerprint.truncated {
            blockers.push(AlignmentBlocker::HubTruncated);
        }
        if source_build.degree() < options.min_degree
            || target_build.degree() < options.min_degree
        {
            blockers.push(AlignmentBlocker::Sparse);
        }
        if !reverse_ok {
            blockers.push(AlignmentBlocker::NonReciprocal);
        }
        if source_margin < options.accept_margin_bps
            || (reverse_ok && target_margin < options.accept_margin_bps)
        {
            blockers.push(AlignmentBlocker::AmbiguousMargin);
        }
        if candidate.confidence_bps < options.min_confidence_bps {
            blockers.push(AlignmentBlocker::LowConfidence);
        }
        blockers.sort();
        blockers.dedup();
        let status = if blockers.iter().any(|blocker| blocker.is_hard()) {
            AlignmentStatus::Rejected
        } else if blockers.is_empty() {
            AlignmentStatus::Accepted
        } else {
            AlignmentStatus::Proposed
        };

        let (canonical, alias) = canonical_pair(&source, &target, &scope_rank);
        certificates.push(AlignmentCertificate {
            from_scope: source.scope.clone(),
            from: source.surface.clone(),
            to_scope: target.scope.clone(),
            to: target.surface.clone(),
            status,
            canonical,
            alias,
            matched_rounds,
            total_rounds: source_fingerprint.colors.len(),
            confidence_bps: candidate.confidence_bps,
            margin_bps: source_margin.min(target_margin),
            from_types: source_build.types.iter().cloned().collect(),
            to_types: target_build.types.iter().cloned().collect(),
            shared_predicates: source_build.shared_predicates(target_build),
            from_degree: source_build.degree(),
            to_degree: target_build.degree(),
            from_truncated: source_fingerprint.truncated,
            to_truncated: target_fingerprint.truncated,
            fingerprint: format!("{:016x}", source_fingerprint.final_color()),
            blockers,
        });
    }

    certificates.sort_by(|left, right| {
        left.status
            .cmp(&right.status)
            .then_with(|| left.from_scope.cmp(&right.from_scope))
            .then_with(|| left.from.cmp(&right.from))
            .then_with(|| left.to_scope.cmp(&right.to_scope))
            .then_with(|| left.to.cmp(&right.to))
    });
    let accepted = certificates
        .iter()
        .filter(|certificate| certificate.status == AlignmentStatus::Accepted)
        .count();
    let proposed = certificates
        .iter()
        .filter(|certificate| certificate.status == AlignmentStatus::Proposed)
        .count();
    let rejected = certificates
        .iter()
        .filter(|certificate| certificate.status == AlignmentStatus::Rejected)
        .count();

    AlignmentPlan {
        dialect: GRAPH_IDENTITY_DIALECT.to_string(),
        certificates,
        scopes: active.len(),
        nodes: nodes.len(),
        candidates: candidate_count,
        accepted,
        proposed,
        rejected,
        oversized_buckets,
        scope_order,
    }
}

/// 从融合账本构建对齐计划；只读取参与默认视图的命名空间。
pub fn align_ledger(ledger: &FusionLedger, options: &AlignmentOptions) -> AlignmentPlan {
    align_assertions(&ledger.namespaces, &ledger.assertions, options)
}

fn fingerprint_nodes(
    nodes: &BTreeMap<NodeKey, NodeBuild>,
    options: &AlignmentOptions,
) -> BTreeMap<NodeKey, Fingerprint> {
    let initial = color_of(&["node".to_string()]);
    let mut colors: BTreeMap<NodeKey, u64> = nodes
        .keys()
        .map(|key| (key.clone(), initial))
        .collect();
    let mut traces: BTreeMap<NodeKey, Vec<u64>> = colors
        .keys()
        .map(|key| (key.clone(), vec![initial]))
        .collect();
    let mut truncated: BTreeSet<NodeKey> = BTreeSet::new();

    for _ in 0..options.rounds {
        let mut next = BTreeMap::new();
        for (key, node) in nodes {
            let previous = colors[key];
            let mut parts = vec![format!("p|{:016x}", previous)];
            let mut outgoing: Vec<(String, u64)> = node
                .out_entity
                .iter()
                .map(|(predicate, target)| (predicate.clone(), colors[target]))
                .collect();
            outgoing.sort();
            let out_truncated = outgoing.len() > options.hub_cap;
            if out_truncated {
                outgoing.truncate(options.hub_cap);
                truncated.insert(key.clone());
            }
            for (predicate, color) in &outgoing {
                parts.push(format!("o|{}|{:016x}", predicate, color));
            }
            parts.push(format!("oc|{}", node.out_entity.len()));

            let mut incoming: Vec<(String, u64)> = node
                .in_entity
                .iter()
                .map(|(predicate, source)| (predicate.clone(), colors[source]))
                .collect();
            incoming.sort();
            let in_truncated = incoming.len() > options.hub_cap;
            if in_truncated {
                incoming.truncate(options.hub_cap);
                truncated.insert(key.clone());
            }
            for (predicate, color) in &incoming {
                parts.push(format!("i|{}|{:016x}", predicate, color));
            }
            parts.push(format!("ic|{}", node.in_entity.len()));

            let literal_predicates: BTreeSet<&String> = node.out_literal.iter().collect();
            for predicate in literal_predicates {
                parts.push(format!("l|{}", predicate));
            }
            parts.push(format!("lc|{}", node.out_literal.len()));
            parts.sort();
            next.insert(key.clone(), color_of(&parts));
        }
        for (key, color) in &next {
            traces.get_mut(key).expect("轨迹已初始化").push(*color);
        }
        colors = next;
    }

    traces
        .into_iter()
        .map(|(key, colors)| {
            (
                key.clone(),
                Fingerprint {
                    colors,
                    truncated: truncated.contains(&key),
                },
            )
        })
        .collect()
}

fn matched_rounds(left: &Fingerprint, right: &Fingerprint) -> usize {
    left.colors
        .iter()
        .zip(right.colors.iter())
        .filter(|(left, right)| left == right)
        .count()
}

fn confidence_bps(
    left: &NodeBuild,
    right: &NodeBuild,
    matched_rounds: usize,
    options: &AlignmentOptions,
) -> u16 {
    let mut score = 3_000u32;
    score += matched_rounds.min(4) as u32 * 1_200;
    if !left.types.is_empty() && left.types == right.types {
        score += 1_800;
    }
    if !left.shared_predicates(right).is_empty() {
        score += 600;
    }
    if left.degree() >= options.min_degree && right.degree() >= options.min_degree {
        score += 600;
    }
    if left.degree() == right.degree() {
        score += 300;
    }
    score.min(9_900) as u16
}

fn canonical_pair(
    left: &NodeKey,
    right: &NodeKey,
    scope_rank: &BTreeMap<String, usize>,
) -> (String, String) {
    let left_rank = scope_rank
        .get(&left.scope)
        .copied()
        .unwrap_or(usize::MAX);
    let right_rank = scope_rank
        .get(&right.scope)
        .copied()
        .unwrap_or(usize::MAX);
    let left_first = left_rank < right_rank
        || (left_rank == right_rank && left.surface <= right.surface);
    if left_first {
        (left.surface.clone(), right.surface.clone())
    } else {
        (right.surface.clone(), left.surface.clone())
    }
}

fn compare_nodes(
    left: &NodeKey,
    right: &NodeKey,
    scope_rank: &BTreeMap<String, usize>,
) -> std::cmp::Ordering {
    scope_rank
        .get(&left.scope)
        .copied()
        .unwrap_or(usize::MAX)
        .cmp(
            &scope_rank
                .get(&right.scope)
                .copied()
                .unwrap_or(usize::MAX),
        )
        .then_with(|| left.surface.cmp(&right.surface))
        .then_with(|| left.scope.cmp(&right.scope))
}

fn compare_candidates(
    left: &Candidate,
    right: &Candidate,
    scope_rank: &BTreeMap<String, usize>,
) -> std::cmp::Ordering {
    right
        .confidence_bps
        .cmp(&left.confidence_bps)
        .then_with(|| compare_nodes(&left.to, &right.to, scope_rank))
}

fn color_of(parts: &[String]) -> u64 {
    let mut hash = FNV_OFFSET;
    for part in parts {
        let bytes = part.as_bytes();
        hash ^= bytes.len() as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge_fusion::{NamespaceKind, TrustLevel};

    fn namespace(id: &str, trust: TrustLevel) -> NamespaceRecord {
        NamespaceRecord::new(id, id, NamespaceKind::Import, trust)
    }

    fn assertion(
        namespace: &str,
        subject: &str,
        predicate: &str,
        object: &str,
        kind: ObjectKind,
    ) -> ScopedAssertion {
        ScopedAssertion::new(namespace, subject, predicate, object, kind, "test")
    }

    fn store() -> Vec<NamespaceRecord> {
        vec![
            namespace("alice", TrustLevel::Curated),
            namespace("bob", TrustLevel::Imported),
        ]
    }

    fn graph_pairs() -> Vec<ScopedAssertion> {
        vec![
            assertion("alice", "苹果公司", "type", "Company", ObjectKind::Entity),
            assertion(
                "alice",
                "苹果公司",
                "创始人",
                "史蒂夫·乔布斯",
                ObjectKind::Entity,
            ),
            assertion(
                "alice",
                "史蒂夫·乔布斯",
                "type",
                "Person",
                ObjectKind::Entity,
            ),
            assertion("bob", "Apple Inc.", "type", "Company", ObjectKind::Entity),
            assertion(
                "bob",
                "Apple Inc.",
                "创始人",
                "Steve Jobs",
                ObjectKind::Entity,
            ),
            assertion("bob", "Steve Jobs", "type", "Person", ObjectKind::Entity),
        ]
    }

    #[test]
    fn structural_alignment_accepts_cross_scope_company() {
        let plan = align_assertions(&store(), &graph_pairs(), &AlignmentOptions::default());
        let certificate = plan
            .certificates
            .iter()
            .find(|certificate| certificate.from.contains("苹果") || certificate.to.contains("苹果"))
            .expect("公司应出现身份证书");
        assert_eq!(certificate.status, AlignmentStatus::Accepted);
        assert_eq!(certificate.from_types, vec!["Company".to_string()]);
        assert!(certificate.shared_predicates.contains(&"创始人".to_string()));
        assert!(certificate.confidence_bps >= 7_000);
        assert!(certificate.matched_rounds >= 2);
    }

    #[test]
    fn type_conflict_is_rejected_not_merged() {
        let namespaces = store();
        let assertions = vec![
            assertion("alice", "苹果", "type", "水果", ObjectKind::Entity),
            assertion("alice", "苹果", "产地", "山东", ObjectKind::Literal),
            assertion("bob", "苹果公司", "type", "公司", ObjectKind::Entity),
            assertion("bob", "苹果公司", "产地", "山东", ObjectKind::Literal),
        ];
        let plan = align_assertions(&namespaces, &assertions, &AlignmentOptions::default());
        let certificate = plan
            .certificates
            .iter()
            .find(|certificate| {
                certificate.from == "苹果" && certificate.to == "苹果公司"
            })
            .expect("同构但类型互斥的实体应留下拒绝证书");
        assert_eq!(certificate.status, AlignmentStatus::Rejected);
        assert!(certificate
            .blockers
            .contains(&AlignmentBlocker::TypeConflict));
        assert!(plan.projection_map().is_empty());
    }

    #[test]
    fn missing_types_are_proposed_only() {
        let namespaces = store();
        let assertions = vec![
            assertion("alice", "甲", "关系", "乙", ObjectKind::Entity),
            assertion("bob", "丙", "关系", "丁", ObjectKind::Entity),
        ];
        let plan = align_assertions(&namespaces, &assertions, &AlignmentOptions::default());
        let certificate = plan
            .certificates
            .iter()
            .find(|certificate| certificate.from == "甲" && certificate.to == "丙")
            .expect("无类型实体应留下提案证书");
        assert_eq!(certificate.status, AlignmentStatus::Proposed);
        assert!(certificate
            .blockers
            .contains(&AlignmentBlocker::MissingTypes));
    }

    #[test]
    fn tied_candidates_are_ambiguous() {
        let namespaces = store();
        let assertions = vec![
            assertion("alice", "甲一", "type", "Person", ObjectKind::Entity),
            assertion("alice", "甲一", "年龄", "18", ObjectKind::Literal),
            assertion("alice", "甲二", "type", "Person", ObjectKind::Entity),
            assertion("alice", "甲二", "年龄", "19", ObjectKind::Literal),
            assertion("bob", "乙", "type", "Person", ObjectKind::Entity),
            assertion("bob", "乙", "年龄", "20", ObjectKind::Literal),
        ];
        let plan = align_assertions(&namespaces, &assertions, &AlignmentOptions::default());
        let certificate = plan
            .certificates
            .iter()
            .find(|certificate| certificate.to == "乙")
            .expect("乙应出现证书");
        assert_eq!(certificate.status, AlignmentStatus::Proposed);
        assert!(certificate
            .blockers
            .contains(&AlignmentBlocker::AmbiguousMargin));
    }

    #[test]
    fn projection_uses_most_trusted_scope() {
        let plan = align_assertions(&store(), &graph_pairs(), &AlignmentOptions::default());
        let aliases = plan.aliases();
        assert!(aliases.contains(&("Apple Inc.".to_string(), "苹果公司".to_string())));
        let projected = plan.project_assertions(&graph_pairs());
        assert!(projected.iter().any(|assertion| {
            assertion.namespace == "bob"
                && assertion.subject == "苹果公司"
                && assertion.predicate == "type"
        }));
        assert!(projected.iter().any(|assertion| {
            assertion.namespace == "alice" && assertion.subject == "苹果公司"
        }));
    }

    #[test]
    fn identity_edges_use_structural_kind() {
        let plan = align_assertions(&store(), &graph_pairs(), &AlignmentOptions::default());
        let edges = plan.identity_edges();
        assert!(!edges.is_empty());
        let edge = edges
            .iter()
            .find(|edge| edge.from == "Apple Inc.")
            .expect("应导出 Apple Inc. 的结构身份边");
        assert_eq!(edge.kind, IdentityKind::Structural);
        assert_eq!(edge.to, "苹果公司");
        assert!(edge.source.contains("结构对齐"));
    }

    #[test]
    fn output_is_deterministic() {
        let first = align_assertions(&store(), &graph_pairs(), &AlignmentOptions::default());
        let second = align_assertions(&store(), &graph_pairs(), &AlignmentOptions::default());
        assert_eq!(first.render_zh(), second.render_zh());
        assert_eq!(first, second);
    }
}

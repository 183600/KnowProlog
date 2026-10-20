//! 证明日志：把归结证明具体化为可回放、可比较、可叙述的凭证。
//!
//! 设计边界：
//!   - 证明 id 只由规范逻辑形式（原子、规则来源、置信度、绑定）计算，
//!     绝不包含自然语言标签；换一种说法叙述同一证明，id 不变；
//!   - 日志不假装完整：超过节点/深度预算的子树被折叠，但保留被折叠
//!     子树的规范摘要，读者能区分「省略」与「本来就没有」；
//!   - 不是所有回答都有形式证明。没有证明的管线显式记为
//!     [`ProofOrigin::Unattributed`]，绝不伪造一棵证明树；
//!   - 会话日志是真相源：本模块只从 [`Chronicle`] 投影问答对，
//!     不维护第二份可变状态。

use std::collections::BTreeMap;
use std::io;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::chronicle::{answer_key, question_key, Chronicle, ChronicleEntry};
use crate::context_image::{canonical_json, fnv1a_hex};
use crate::resolution_engine::{ProofNode, ProofSource, Solution};
use crate::symbolic_reasoner::{ProofSolutionView, ResolutionProofView};
use crate::view_render::{ProofViewNode, ViewProofSource};

/// 默认保留的证明节点上限（折叠而不是丢弃语义）。
pub const DEFAULT_MAX_PROOF_NODES: usize = 64;
/// 默认证明深度上限。
pub const DEFAULT_MAX_PROOF_DEPTH: usize = 12;

/// 证明被记录时的来源。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ProofOrigin {
    /// 归结引擎给出的形式证明。
    Resolution,
    /// 没有可展开形式证明的回答（显式诚实，不伪造证明树）。
    Unattributed { pipeline: String },
}

impl ProofOrigin {
    pub fn is_resolution(&self) -> bool {
        matches!(self, ProofOrigin::Resolution)
    }

    fn tag(&self) -> String {
        match self {
            ProofOrigin::Resolution => "resolution".to_string(),
            ProofOrigin::Unattributed { pipeline } => format!("unattributed:{}", pipeline),
        }
    }
}

/// 证明节点的规范来源。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ProofSourceRecord {
    Fact,
    Rule { source: String },
}

/// 被折叠的子树：省略了多少节点，以及它们的规范摘要。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProofElision {
    pub omitted_nodes: usize,
    pub digest: String,
}

/// 证明日志中的一个节点。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProofNodeRecord {
    /// 规范逻辑形式（`Atom::display()`）。
    pub atom: String,
    /// 展示用标签（自然语言）；不参与证明 id。
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub label: String,
    pub source: ProofSourceRecord,
    pub confidence: f64,
    pub children: Vec<ProofNodeRecord>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub elided: Option<ProofElision>,
}

impl ProofNodeRecord {
    pub fn node_count(&self) -> usize {
        1 + self.children.iter().map(Self::node_count).sum::<usize>()
    }

    pub fn describe(&self) -> String {
        let label = if self.label.is_empty() {
            self.atom.as_str()
        } else {
            self.label.as_str()
        };
        match &self.source {
            ProofSourceRecord::Fact => format!("{}（事实）", label),
            ProofSourceRecord::Rule { source } => {
                format!("{}（由 {}，置信度 {:.2}）", label, source, self.confidence)
            }
        }
    }
}

/// 生成证明时所属搜索的确定性上下文（与证明 id 解耦，可单独变化）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchRecord {
    pub status: String,
    pub truncated: bool,
    pub goals: usize,
    pub goal_attempts: usize,
    pub fact_checks: usize,
    pub rule_attempts: usize,
    pub depth_cutoffs: usize,
    pub budget_cutoffs: usize,
    pub solution_cutoffs: usize,
    pub state_cutoffs: usize,
    /// 同一搜索树的代表性失败前沿（`FailureFrontier::describe`）。
    pub frontier: Vec<String>,
    pub frontier_entries: usize,
    pub frontier_omitted: usize,
}

impl SearchRecord {
    pub fn from_view(view: &ResolutionProofView) -> Self {
        Self {
            status: view.search_status.label().to_string(),
            truncated: view.search_truncated,
            goals: view.stats.goals,
            goal_attempts: view.stats.goal_attempts,
            fact_checks: view.stats.fact_checks,
            rule_attempts: view.stats.rule_attempts,
            depth_cutoffs: view.stats.depth_cutoffs,
            budget_cutoffs: view.stats.budget_cutoffs,
            solution_cutoffs: view.stats.solution_cutoffs,
            state_cutoffs: view.stats.state_cutoffs,
            frontier: view.frontier.iter().map(|entry| entry.describe()).collect(),
            frontier_entries: view.stats.frontier_entries,
            frontier_omitted: view.stats.frontier_omitted,
        }
    }
}

/// 一条可持久化的证明凭证。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProofCertificate {
    /// 稳定 id：只依赖规范证明内容。
    pub id: String,
    /// 被证明的目标（规范显示形式）。
    pub goal: String,
    /// 查询变量的绑定（按变量名排序）。
    pub bindings: Vec<(String, String)>,
    /// 证明的总置信度；无形式证明的管线为 None。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
    pub origin: ProofOrigin,
    /// 生成证明的搜索上下文；旧凭证没有该字段，读取时为 None。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search: Option<SearchRecord>,
    pub nodes: Vec<ProofNodeRecord>,
    /// 完整证明的节点总数（含被折叠部分）。
    pub step_count: usize,
    pub truncated: bool,
    pub omitted_nodes: usize,
    /// 完整证明的规范摘要（折叠不影响它）。
    pub digest: String,
}

/// 证明记录预算。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProofLimits {
    pub max_nodes: usize,
    pub max_depth: usize,
}

impl Default for ProofLimits {
    fn default() -> Self {
        Self {
            max_nodes: DEFAULT_MAX_PROOF_NODES,
            max_depth: DEFAULT_MAX_PROOF_DEPTH,
        }
    }
}

#[derive(Debug, Clone)]
struct RawProofNode {
    atom: String,
    label: String,
    source: ProofSourceRecord,
    confidence: f64,
    children: Vec<RawProofNode>,
}

impl RawProofNode {
    fn from_resolution(node: &ProofNode) -> Self {
        Self {
            atom: node.atom.display(),
            label: String::new(),
            source: match &node.source {
                ProofSource::Fact => ProofSourceRecord::Fact,
                ProofSource::Rule { source } => ProofSourceRecord::Rule {
                    source: source.clone(),
                },
            },
            confidence: node.confidence,
            children: node.children.iter().map(Self::from_resolution).collect(),
        }
    }

    fn from_view(node: &ProofViewNode) -> Self {
        Self {
            atom: node.canonical.clone(),
            label: node.label.clone(),
            source: match &node.source {
                ViewProofSource::Fact => ProofSourceRecord::Fact,
                ViewProofSource::Rule(source) => ProofSourceRecord::Rule {
                    source: source.clone(),
                },
            },
            confidence: node.confidence,
            children: node.children.iter().map(Self::from_view).collect(),
        }
    }

    fn node_count(&self) -> usize {
        1 + self.children.iter().map(Self::node_count).sum::<usize>()
    }

    fn write_canonical(&self, out: &mut String) {
        out.push_str(&self.atom);
        out.push('[');
        match &self.source {
            ProofSourceRecord::Fact => out.push_str("fact"),
            ProofSourceRecord::Rule { source } => {
                out.push_str("rule:");
                out.push_str(source);
            }
        }
        out.push('|');
        out.push_str(&format!("{:.6}", self.confidence));
        out.push(']');
        for child in &self.children {
            out.push('(');
            child.write_canonical(out);
            out.push(')');
        }
    }

    fn canonical(&self) -> String {
        let mut out = String::new();
        self.write_canonical(&mut out);
        out
    }

    fn digest(&self) -> String {
        fnv1a_hex(self.canonical().as_bytes())
    }
}

impl ProofCertificate {
    /// 从归结解构造凭证。
    pub fn from_solution(goal: &str, solution: &Solution, limits: ProofLimits) -> Self {
        Self::from_resolution(
            goal,
            &solution.bindings,
            solution.proofs.as_slice(),
            solution.confidence,
            limits,
        )
    }

    /// 从归结证明森林构造凭证。
    pub fn from_resolution(
        goal: &str,
        bindings: &std::collections::HashMap<String, String>,
        proofs: &[ProofNode],
        confidence: f64,
        limits: ProofLimits,
    ) -> Self {
        let mut bindings: Vec<(String, String)> = bindings
            .iter()
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect();
        bindings.sort();
        let roots: Vec<RawProofNode> = proofs.iter().map(RawProofNode::from_resolution).collect();
        Self::assemble(
            goal,
            bindings,
            Some(confidence),
            ProofOrigin::Resolution,
            roots,
            limits,
        )
    }

    /// 从结构化证明视图构造凭证：同一视图中的每个解各一条。
    pub fn from_view(view: &ResolutionProofView, limits: ProofLimits) -> Vec<Self> {
        view.solutions
            .iter()
            .map(|solution: &ProofSolutionView| {
                let roots: Vec<RawProofNode> = solution
                    .proofs
                    .iter()
                    .map(RawProofNode::from_view)
                    .collect();
                let mut certificate = Self::assemble(
                    &view.goal_display,
                    solution.bindings.clone(),
                    Some(solution.confidence),
                    ProofOrigin::Resolution,
                    roots,
                    limits,
                );
                certificate.search = Some(SearchRecord::from_view(view));
                certificate
            })
            .collect()
    }

    /// 显式标记「没有可展开形式证明」的回答。
    pub fn unattributed(goal: &str, pipeline: &str) -> Self {
        let origin = ProofOrigin::Unattributed {
            pipeline: pipeline.to_string(),
        };
        let digest = fnv1a_hex(format!("unattributed|{}", goal.trim()).as_bytes());
        let id = format!(
            "proof:{}",
            fnv1a_hex(format!("{}|{}", origin.tag(), digest).as_bytes())
        );
        Self {
            id,
            goal: goal.trim().to_string(),
            bindings: Vec::new(),
            confidence: None,
            origin,
            search: None,
            nodes: Vec::new(),
            step_count: 0,
            truncated: false,
            omitted_nodes: 0,
            digest,
        }
    }

    fn assemble(
        goal: &str,
        bindings: Vec<(String, String)>,
        confidence: Option<f64>,
        origin: ProofOrigin,
        roots: Vec<RawProofNode>,
        limits: ProofLimits,
    ) -> Self {
        let step_count = roots.iter().map(RawProofNode::node_count).sum::<usize>();
        let canonical_roots: Vec<String> = roots.iter().map(RawProofNode::canonical).collect();
        let digest = fnv1a_hex(canonical_roots.join(";").as_bytes());
        let binding_text = bindings
            .iter()
            .map(|(name, value)| format!("{}={}", name, value))
            .collect::<Vec<_>>()
            .join("|");
        let id = format!(
            "proof:{}",
            fnv1a_hex(
                format!(
                    "{}|{}|{}|{}",
                    origin.tag(),
                    goal.trim(),
                    binding_text,
                    digest
                )
                .as_bytes()
            )
        );

        let max_nodes = limits.max_nodes.max(1);
        let max_depth = limits.max_depth.max(1);
        let mut budget = max_nodes;
        let mut nodes = Vec::new();
        for root in &roots {
            if budget == 0 {
                continue;
            }
            nodes.push(build_truncated(root, 1, &mut budget, max_depth));
        }
        let retained = nodes.iter().map(ProofNodeRecord::node_count).sum::<usize>();
        let omitted_nodes = step_count.saturating_sub(retained);
        let truncated = omitted_nodes > 0;

        Self {
            id,
            goal: goal.trim().to_string(),
            bindings,
            confidence,
            origin,
            search: None,
            nodes,
            step_count,
            truncated,
            omitted_nodes,
            digest,
        }
    }

    /// 已保留的节点数。
    pub fn retained_nodes(&self) -> usize {
        self.nodes.iter().map(ProofNodeRecord::node_count).sum()
    }

    /// 完整证明的规范文本（折叠不影响它，可用于 diff）。
    pub fn canonical_body(&self) -> String {
        canonical_json(&json!({
            "goal": self.goal,
            "bindings": self.bindings,
            "origin": self.origin,
            "digest": self.digest,
            "step_count": self.step_count,
        }))
    }

    /// 叙述证明：Brief 只给结论，Normal 给树，Deep 给完整树与规范摘要。
    pub fn narrate(&self, detail: ProofDetail) -> String {
        let mut lines = Vec::new();
        let confidence = self
            .confidence
            .map(|value| format!("，置信度 {:.2}", value))
            .unwrap_or_default();
        let truncation = if self.truncated {
            format!("，已折叠 {} 步（摘要 {}）", self.omitted_nodes, self.digest)
        } else {
            String::new()
        };
        lines.push(format!(
            "证明 {}：{}（{} 步{}）{}",
            self.id, self.goal, self.step_count, confidence, truncation
        ));
        match &self.origin {
            ProofOrigin::Unattributed { pipeline } => {
                lines.push(format!(
                    "这一条来自 {} 管线，没有可展开的形式证明。",
                    pipeline
                ));
                return lines.join("\n");
            }
            ProofOrigin::Resolution => {}
        }
        if !self.bindings.is_empty() {
            lines.push(format!(
                "绑定：{}",
                self.bindings
                    .iter()
                    .map(|(name, value)| format!("{}={}", name, value))
                    .collect::<Vec<_>>()
                    .join("；")
            ));
        }
        if detail == ProofDetail::Brief {
            return lines.join("\n");
        }
        if let Some(search) = &self.search {
            lines.push(format!(
                "搜索上下文：{}（{} 个目标、{} 次目标展开、{} 次事实检查、{} 次规则尝试；{}）",
                search.status,
                search.goals,
                search.goal_attempts,
                search.fact_checks,
                search.rule_attempts,
                if search.truncated {
                    "结果可能不完整"
                } else {
                    "搜索完整"
                }
            ));
            if !search.frontier.is_empty() {
                let shown = search.frontier.iter().take(3).cloned().collect::<Vec<_>>();
                let omitted =
                    search.frontier.len().saturating_sub(shown.len()) + search.frontier_omitted;
                lines.push(format!(
                    "同树失败前沿：{}{}",
                    shown.join("；"),
                    if omitted > 0 {
                        format!("；另有 {} 条未展开", omitted)
                    } else {
                        String::new()
                    }
                ));
            }
        }
        let limit = match detail {
            ProofDetail::Brief => 0,
            ProofDetail::Normal => 24,
            ProofDetail::Deep => usize::MAX,
        };
        let mut tree = Vec::new();
        for root in &self.nodes {
            if tree.len() >= limit {
                tree.push("…（其余节点已省略）".to_string());
                break;
            }
            render_tree(root, "", true, &mut tree, limit);
        }
        if tree.is_empty() {
            tree.push("（无证明节点）".to_string());
        }
        lines.push(tree.join("\n"));
        if detail == ProofDetail::Deep {
            lines.push(format!("规范摘要：{}", self.digest));
        }
        lines.join("\n")
    }

    /// 写入编年史（kind = proof）。
    pub fn record(&self, chronicle: &mut Chronicle) -> io::Result<u64> {
        let payload = serde_json::to_value(self)
            .map_err(|error| io::Error::new(io::ErrorKind::Other, error))?;
        chronicle.append("proof", payload)
    }

    /// 从编年史记录解析（兼容 `{"certificate": {...}}` 包装）。
    pub fn from_entry(entry: &ChronicleEntry) -> Option<Self> {
        if entry.kind != "proof" {
            return None;
        }
        let value = entry
            .payload
            .get("certificate")
            .cloned()
            .unwrap_or_else(|| entry.payload.clone());
        serde_json::from_value(value).ok()
    }
}

/// 批量写入证明凭证，返回对应 seq。
pub fn record_proofs(
    chronicle: &mut Chronicle,
    proofs: &[ProofCertificate],
) -> io::Result<Vec<u64>> {
    proofs.iter().map(|proof| proof.record(chronicle)).collect()
}

/// 叙述深度。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProofDetail {
    Brief,
    Normal,
    Deep,
}

fn build_truncated(
    raw: &RawProofNode,
    depth: usize,
    budget: &mut usize,
    max_depth: usize,
) -> ProofNodeRecord {
    *budget = budget.saturating_sub(1);
    let mut children = Vec::new();
    let mut omitted_nodes = 0usize;
    let mut omitted_digests = Vec::new();
    for child in &raw.children {
        let count = child.node_count();
        if depth < max_depth && count <= *budget {
            children.push(build_truncated(child, depth + 1, budget, max_depth));
        } else {
            omitted_nodes += count;
            omitted_digests.push(format!("{}:{}", child.digest(), count));
        }
    }
    let elided = (omitted_nodes > 0).then(|| ProofElision {
        omitted_nodes,
        digest: fnv1a_hex(omitted_digests.join(";").as_bytes()),
    });
    ProofNodeRecord {
        atom: raw.atom.clone(),
        label: raw.label.clone(),
        source: raw.source.clone(),
        confidence: raw.confidence,
        children,
        elided,
    }
}

fn render_tree(
    node: &ProofNodeRecord,
    prefix: &str,
    last: bool,
    lines: &mut Vec<String>,
    limit: usize,
) {
    if lines.len() >= limit {
        return;
    }
    lines.push(format!(
        "{}{} {}",
        prefix,
        if last { "└─" } else { "├─" },
        node.describe()
    ));
    let next_prefix = format!("{}{}", prefix, if last { "   " } else { "│  " });
    if let Some(elided) = &node.elided {
        lines.push(format!(
            "{}{} …（折叠 {} 步，摘要 {}）",
            next_prefix,
            if node.children.is_empty() {
                "└─"
            } else {
                "├─"
            },
            elided.omitted_nodes,
            elided.digest
        ));
    }
    for (index, child) in node.children.iter().enumerate() {
        let child_last = index + 1 == node.children.len() && node.elided.is_none();
        render_tree(child, &next_prefix, child_last, lines, limit);
        if lines.len() >= limit {
            break;
        }
    }
}

/// 从日志投影出的一轮问答。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JournalTurn {
    pub turn_seq: u64,
    pub reply_seq: Option<u64>,
    pub question: String,
    pub question_key: String,
    pub answer: Option<String>,
    pub answer_key: Option<String>,
    pub kb_revision: u64,
    pub proof_id: Option<String>,
}

impl JournalTurn {
    pub fn is_answered(&self) -> bool {
        self.answer.is_some()
    }
}

#[derive(Debug, Clone)]
struct PendingTurn {
    seq: u64,
    question: String,
    key: String,
    revision: u64,
}

/// 编年史的只读问答视图（日志是真相，投影只是解释）。
#[derive(Debug, Clone, Default)]
pub struct JournalProjection {
    pub turns: Vec<JournalTurn>,
    pub proofs: Vec<ProofCertificate>,
}

impl JournalProjection {
    pub fn from_chronicle(chronicle: &Chronicle) -> Self {
        let mut projection = JournalProjection::default();
        let mut pending: Option<PendingTurn> = None;

        let flush = |pending: &mut Option<PendingTurn>, turns: &mut Vec<JournalTurn>| {
            if let Some(turn) = pending.take() {
                turns.push(JournalTurn {
                    turn_seq: turn.seq,
                    reply_seq: None,
                    question: turn.question,
                    question_key: turn.key,
                    answer: None,
                    answer_key: None,
                    kb_revision: turn.revision,
                    proof_id: None,
                });
            }
        };

        for entry in chronicle.entries() {
            match entry.kind.as_str() {
                "turn" => {
                    flush(&mut pending, &mut projection.turns);
                    let mode = entry
                        .payload
                        .get("mode")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    let user = entry
                        .payload
                        .get("user")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .trim();
                    if mode == "query" && !user.is_empty() {
                        pending = Some(PendingTurn {
                            seq: entry.seq,
                            question: user.to_string(),
                            key: question_key(user),
                            revision: entry.kb_revision,
                        });
                    }
                }
                "reply" => {
                    let Some(turn) = pending.take() else {
                        continue;
                    };
                    let payload_question = entry
                        .payload
                        .get("question")
                        .and_then(Value::as_str)
                        .map(str::trim)
                        .filter(|question| !question.is_empty());
                    let question = payload_question.unwrap_or(&turn.question).to_string();
                    let key = entry
                        .payload
                        .get("question_key")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                        .unwrap_or_else(|| question_key(&question));
                    let answer = entry
                        .payload
                        .get("answer")
                        .and_then(Value::as_str)
                        .map(str::to_string);
                    projection.turns.push(JournalTurn {
                        turn_seq: turn.seq,
                        reply_seq: Some(entry.seq),
                        question,
                        question_key: key,
                        answer_key: answer.as_deref().map(answer_key),
                        answer,
                        kb_revision: entry.kb_revision,
                        proof_id: entry
                            .payload
                            .get("proof_id")
                            .and_then(Value::as_str)
                            .map(str::to_string),
                    });
                }
                "proof" => {
                    if let Some(certificate) = ProofCertificate::from_entry(entry) {
                        projection.proofs.push(certificate);
                    }
                }
                _ => {}
            }
        }
        flush(&mut pending, &mut projection.turns);
        projection
    }

    pub fn unanswered(&self) -> Vec<&JournalTurn> {
        self.turns
            .iter()
            .filter(|turn| !turn.is_answered())
            .collect()
    }

    pub fn find_proof(&self, id: &str) -> Option<&ProofCertificate> {
        self.proofs.iter().find(|proof| proof.id == id)
    }

    pub fn latest_proof(&self) -> Option<&ProofCertificate> {
        self.proofs.last()
    }

    /// 同题异答检测：相邻两次回答不同即视为一次「改口」。
    pub fn consistency(&self, current_revision: u64) -> ConsistencyReport {
        let mut by_key: BTreeMap<&str, Vec<&JournalTurn>> = BTreeMap::new();
        for turn in &self.turns {
            if turn.is_answered() && !turn.question_key.is_empty() {
                by_key
                    .entry(turn.question_key.as_str())
                    .or_default()
                    .push(turn);
            }
        }

        let mut findings = Vec::new();
        let mut stale = Vec::new();
        for records in by_key.values() {
            for pair in records.windows(2) {
                let previous = pair[0];
                let current = pair[1];
                if previous.answer_key == current.answer_key {
                    continue;
                }
                let kind = if previous.kb_revision == current.kb_revision {
                    ChangeKind::SameRevisionConflict
                } else {
                    ChangeKind::RevisionChanged
                };
                findings.push(ConsistencyFinding {
                    question_key: current.question_key.clone(),
                    question: current.question.clone(),
                    previous: previous.clone(),
                    current: current.clone(),
                    kind,
                });
            }
            if let Some(last) = records.last() {
                if last.kb_revision < current_revision {
                    stale.push(StaleAnswer {
                        question: last.question.clone(),
                        answer: last.answer.clone().unwrap_or_default(),
                        answered_at: last.kb_revision,
                        current_revision,
                    });
                }
            }
        }
        findings.sort_by_key(|finding| finding.current.reply_seq.unwrap_or(u64::MAX));
        stale.sort_by(|left, right| left.question.cmp(&right.question));

        ConsistencyReport {
            reviewed_answers: self.turns.iter().filter(|turn| turn.is_answered()).count(),
            question_keys: by_key.len(),
            findings,
            stale,
            unanswered: self.unanswered().len(),
            current_revision,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeKind {
    /// 知识版本变化后答案随之变化（可解释的修订）。
    RevisionChanged,
    /// 同一知识版本下两次回答不一致（强冲突）。
    SameRevisionConflict,
}

impl ChangeKind {
    pub fn label(self) -> &'static str {
        match self {
            ChangeKind::RevisionChanged => "知识版本变化",
            ChangeKind::SameRevisionConflict => "同一知识版本下答案冲突",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConsistencyFinding {
    pub question_key: String,
    pub question: String,
    pub previous: JournalTurn,
    pub current: JournalTurn,
    pub kind: ChangeKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StaleAnswer {
    pub question: String,
    pub answer: String,
    pub answered_at: u64,
    pub current_revision: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConsistencyReport {
    pub reviewed_answers: usize,
    pub question_keys: usize,
    pub findings: Vec<ConsistencyFinding>,
    pub stale: Vec<StaleAnswer>,
    pub unanswered: usize,
    pub current_revision: u64,
}

impl ConsistencyReport {
    pub fn has_findings(&self) -> bool {
        !self.findings.is_empty()
    }

    pub fn same_revision_conflicts(&self) -> usize {
        self.findings
            .iter()
            .filter(|finding| finding.kind == ChangeKind::SameRevisionConflict)
            .count()
    }

    pub fn render(&self) -> String {
        let mut lines = vec![format!(
            "一致性检查：审查 {} 条已答问题（{} 个问题键，当前 KB 修订 {}）；改口 {} 处（其中同版本冲突 {}），过期回答 {} 条，未答问题 {} 个。",
            self.reviewed_answers,
            self.question_keys,
            self.current_revision,
            self.findings.len(),
            self.same_revision_conflicts(),
            self.stale.len(),
            self.unanswered
        )];
        if self.findings.is_empty() && self.stale.is_empty() && self.unanswered == 0 {
            lines.push("没有发现同题异答、过期回答或未答问题。".to_string());
            return lines.join("\n");
        }
        for finding in &self.findings {
            lines.push(format!(
                "- 问题「{}」：第 {} 条（rev {}）→ 第 {} 条（rev {}），{}。前答：{}；后答：{}",
                truncate(&finding.question, 32),
                finding
                    .previous
                    .reply_seq
                    .unwrap_or(finding.previous.turn_seq),
                finding.previous.kb_revision,
                finding
                    .current
                    .reply_seq
                    .unwrap_or(finding.current.turn_seq),
                finding.current.kb_revision,
                finding.kind.label(),
                truncate(finding.previous.answer.as_deref().unwrap_or(""), 48),
                truncate(finding.current.answer.as_deref().unwrap_or(""), 48),
            ));
        }
        for stale in &self.stale {
            lines.push(format!(
                "- 过期回答「{}」：答于 rev {}，当前 rev {}（知识库已变化，回答可能已不成立）。答：{}",
                truncate(&stale.question, 32),
                stale.answered_at,
                stale.current_revision,
                truncate(&stale.answer, 48)
            ));
        }
        if self.unanswered > 0 {
            lines.push(format!(
                "- 未答问题：{} 个（日志中有提问但没有对应回答）。",
                self.unanswered
            ));
        }
        lines.join("\n")
    }
}

fn truncate(text: &str, limit: usize) -> String {
    let mut out: String = text.chars().take(limit).collect();
    if text.chars().count() > limit {
        out.push('…');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chronicle::FsyncPolicy;
    use crate::resolution_engine::Atom;
    use tempfile::TempDir;

    fn leaf(atom: &str) -> ProofNode {
        ProofNode {
            atom: Atom::rel(atom, "张三", "李四"),
            source: ProofSource::Fact,
            confidence: 1.0,
            children: Vec::new(),
        }
    }

    fn chain(depth: usize) -> ProofNode {
        let mut node = leaf("师傅");
        for index in 1..depth {
            node = ProofNode {
                atom: Atom::rel("祖先", "张三", "李四"),
                source: ProofSource::Rule {
                    source: format!("规则{}", index),
                },
                confidence: 0.9,
                children: vec![node],
            };
        }
        node
    }

    #[test]
    fn proof_id_is_stable_and_ignores_narrative_labels() {
        let proof = chain(3);
        let first = ProofCertificate::from_resolution(
            "祖先(张三, 李四)",
            &std::collections::HashMap::new(),
            &[proof.clone()],
            0.81,
            ProofLimits::default(),
        );
        let second = ProofCertificate::from_resolution(
            "祖先(张三, 李四)",
            &std::collections::HashMap::new(),
            &[proof],
            0.81,
            ProofLimits::default(),
        );
        assert_eq!(first.id, second.id);
        assert_eq!(first.digest, second.digest);
        assert!(first.id.starts_with("proof:"));

        let view_node = ProofViewNode {
            label: "自然语言：张三的祖先".to_string(),
            canonical: "祖先(张三, 李四)".to_string(),
            source: ViewProofSource::Fact,
            confidence: 1.0,
            children: Vec::new(),
        };
        let view = ResolutionProofView {
            phrase: "测试".to_string(),
            goal_display: "祖先(张三, 李四)".to_string(),
            facts: 1,
            clauses: 0,
            max_depth: 6,
            search_status: crate::resolution_engine::SearchStatus::Proved,
            search_truncated: true,
            frontier: Vec::new(),
            stats: crate::resolution_engine::SearchStats {
                goals: 1,
                goal_attempts: 7,
                fact_checks: 3,
                rule_attempts: 2,
                ..crate::resolution_engine::SearchStats::default()
            },
            solutions: vec![ProofSolutionView {
                bindings: vec![],
                confidence: 1.0,
                proofs: vec![view_node.clone()],
            }],
        };
        let from_view = ProofCertificate::from_view(&view, ProofLimits::default());
        let search = from_view[0]
            .search
            .as_ref()
            .expect("视图凭证应带搜索上下文");
        assert_eq!(search.status, "得证");
        assert!(search.truncated);
        assert_eq!(search.goal_attempts, 7);
        let narrative = from_view[0].narrate(ProofDetail::Normal);
        assert!(narrative.contains("搜索上下文："), "实际：{}", narrative);
        assert!(narrative.contains("结果可能不完整"), "实际：{}", narrative);

        let relabeled = {
            let mut node = view_node;
            node.label = "完全不同的措辞".to_string();
            ResolutionProofView {
                solutions: vec![ProofSolutionView {
                    bindings: vec![],
                    confidence: 1.0,
                    proofs: vec![node],
                }],
                search_truncated: false,
                stats: crate::resolution_engine::SearchStats {
                    goal_attempts: 999,
                    ..crate::resolution_engine::SearchStats::default()
                },
                ..view
            }
        };
        let relabeled = ProofCertificate::from_view(&relabeled, ProofLimits::default());
        assert_eq!(
            from_view[0].id, relabeled[0].id,
            "叙述措辞与搜索统计都不得改变证明 id"
        );
    }

    #[test]
    fn truncation_is_bounded_and_keeps_digest() {
        let proof = chain(20);
        let limits = ProofLimits {
            max_nodes: 4,
            max_depth: 3,
        };
        let certificate = ProofCertificate::from_resolution(
            "祖先(张三, 李四)",
            &std::collections::HashMap::new(),
            &[proof],
            0.9_f64.powi(19),
            limits,
        );
        assert!(certificate.truncated);
        assert!(certificate.omitted_nodes > 0);
        assert!(certificate.retained_nodes() <= 4);
        assert_eq!(certificate.step_count, 20);
        let narrative = certificate.narrate(ProofDetail::Normal);
        assert!(narrative.contains("折叠"));
        assert!(narrative.contains(&certificate.digest));
        assert!(!narrative.is_empty());
    }

    #[test]
    fn unattributed_proof_is_honest() {
        let proof = ProofCertificate::unattributed("姚明的身高", "外部知识 API");
        assert!(!proof.origin.is_resolution());
        assert!(proof.nodes.is_empty());
        assert!(proof
            .narrate(ProofDetail::Deep)
            .contains("没有可展开的形式证明"));
    }

    #[test]
    fn proof_roundtrips_through_chronicle() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("proof.chronicle.jsonl");
        let (mut chronicle, _) =
            Chronicle::open(path.to_str().unwrap(), FsyncPolicy::None).unwrap();
        let mut certificate = ProofCertificate::from_resolution(
            "师傅(张三, 李四)",
            &std::collections::HashMap::new(),
            &[leaf("师傅")],
            1.0,
            ProofLimits::default(),
        );
        certificate.search = Some(SearchRecord {
            status: "得证".to_string(),
            truncated: false,
            goals: 1,
            goal_attempts: 6,
            fact_checks: 2,
            rule_attempts: 3,
            depth_cutoffs: 0,
            budget_cutoffs: 0,
            solution_cutoffs: 0,
            state_cutoffs: 0,
            frontier: vec!["父母(张三, ?X)（缺少事实或规则，深度 1）".to_string()],
            frontier_entries: 1,
            frontier_omitted: 0,
        });
        certificate.record(&mut chronicle).unwrap();
        drop(chronicle);

        let (reloaded, audit) = Chronicle::open(path.to_str().unwrap(), FsyncPolicy::None).unwrap();
        assert!(audit.is_clean());
        let projection = JournalProjection::from_chronicle(&reloaded);
        assert_eq!(projection.proofs.len(), 1);
        assert_eq!(projection.proofs[0].id, certificate.id);
        assert_eq!(
            projection.proofs[0].search, certificate.search,
            "搜索上下文必须随凭证持久化"
        );
        assert!(projection.find_proof(&certificate.id).is_some());
        assert!(projection.find_proof("proof:missing").is_none());
    }

    #[test]
    fn consistency_distinguishes_revision_change_from_conflict() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("consistency.chronicle.jsonl");
        let (mut chronicle, _) =
            Chronicle::open(path.to_str().unwrap(), FsyncPolicy::None).unwrap();
        chronicle
            .append_turn("姚明的身高是多少？", "query", None, 0)
            .unwrap();
        chronicle
            .append_reply("姚明身高 226 厘米。", &[], &[])
            .unwrap();
        chronicle
            .append_turn("姚明的身高是多少呢", "query", None, 1)
            .unwrap();
        chronicle
            .append_reply("姚明身高 229 厘米。", &[], &[])
            .unwrap();
        chronicle.set_kb_revision(7);
        chronicle
            .append_turn("姚明的身高是多少呢？", "query", None, 2)
            .unwrap();
        chronicle
            .append_reply("姚明身高 226 厘米。", &[], &[])
            .unwrap();

        let projection = JournalProjection::from_chronicle(&chronicle);
        assert_eq!(projection.turns.len(), 3);
        let report = projection.consistency(8);
        assert_eq!(report.findings.len(), 2);
        assert_eq!(report.same_revision_conflicts(), 1);
        assert!(report
            .findings
            .iter()
            .any(|finding| finding.kind == ChangeKind::RevisionChanged));
        assert_eq!(report.stale.len(), 1);
        assert!(report.render().contains("同一知识版本下答案冲突"));
    }

    #[test]
    fn identical_answers_do_not_trigger_disclosure() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("same.chronicle.jsonl");
        let (mut chronicle, _) =
            Chronicle::open(path.to_str().unwrap(), FsyncPolicy::None).unwrap();
        for index in 0..2 {
            chronicle
                .append_turn("姚明的身高是多少", "query", None, index)
                .unwrap();
            chronicle.append_reply("226 厘米", &[], &[]).unwrap();
        }
        let report = JournalProjection::from_chronicle(&chronicle).consistency(0);
        assert!(report.findings.is_empty());
        assert!(!report.has_findings());
        assert_eq!(report.reviewed_answers, 2);
    }

    #[test]
    fn unanswered_question_is_projected_not_lost() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("open.chronicle.jsonl");
        let (mut chronicle, _) =
            Chronicle::open(path.to_str().unwrap(), FsyncPolicy::None).unwrap();
        chronicle
            .append_turn("谁发明了电话", "query", None, 0)
            .unwrap();
        let projection = JournalProjection::from_chronicle(&chronicle);
        assert_eq!(projection.turns.len(), 1);
        assert_eq!(projection.unanswered().len(), 1);
        assert!(projection.consistency(0).render().contains("未答问题"));
    }
}

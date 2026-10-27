//! 开放谓词闭包（Open Closure）：把「看见的关系」变成可承诺、可撤回的语义假设。
//!
//! 开放信息抽取不预设谓词表，于是任何词面都可能是关系；但传递闭包需要一条
//! 语义承诺。本模块把这条承诺放在会话里，而不是直接写进知识库或律典：
//!
//!   - 观察：复用 `FrameExtractor` 把句子抽成带极性/模态的关系帧，只收直陈事实；
//!   - 试探：对每个谓词枚举有界二跳链，区分「新增结论」与「已有结论」；
//!   - 提问：只在会话里问一次「这个谓词可传递吗」，用户确认/拒绝后才改变口径；
//!   - 回答：未确认时给条件答案与见证路径；确认后给带见证的闭包答案；
//!   - 撤回：确认、拒绝与事实撤回都记在会话账本里，JSON 可复现，不污染图谱。
//!
//! 开放世界原则：没有直接记录不等于否定。只有显式否定事实才会成为反例，
//! 二跳链也不会因为「目标边缺失」被误判为不成立。

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::pattern_closure::{evaluate_closure, ClosureBudget, ClosureOp, ClosureTruncation};
use crate::relation_frame::{FrameExtractor, Modality, Polarity, RelationFrame};

/// 会话账本方言标识，写入 JSON 以便未来迁移。
pub const OPEN_CLOSURE_DIALECT: &str = "knowprolog-open-closure/1";

/// 一次探针最多展示多少条支持链/反例链。
pub const MAX_PROBE_CHAINS: usize = 3;

/// 文本回答最多展开多少个闭包目标。
pub const MAX_RENDER_TARGETS: usize = 12;

/// 谓词传递假设的会话状态。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HypothesisStatus {
    /// 尚未询问/决定：只能给条件答案。
    Unknown,
    /// 用户确认：本会话内参与闭包。
    Confirmed,
    /// 用户拒绝：只返回直接记录。
    Rejected,
}

impl HypothesisStatus {
    pub fn label(self) -> &'static str {
        match self {
            HypothesisStatus::Unknown => "未确认",
            HypothesisStatus::Confirmed => "已确认",
            HypothesisStatus::Rejected => "已拒绝",
        }
    }

    pub fn tag(self) -> &'static str {
        match self {
            HypothesisStatus::Unknown => "unknown",
            HypothesisStatus::Confirmed => "confirmed",
            HypothesisStatus::Rejected => "rejected",
        }
    }
}

/// 一条会话内开放事实。`negated` 表示显式否定，仅用于识别反例与阻断推导。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenFact {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    #[serde(default)]
    pub negated: bool,
    #[serde(default)]
    pub sources: BTreeSet<String>,
}

impl OpenFact {
    fn key(&self) -> (&str, &str, &str, bool) {
        (
            self.subject.as_str(),
            self.predicate.as_str(),
            self.object.as_str(),
            self.negated,
        )
    }

    pub fn render_text(&self) -> String {
        format!(
            "{}{} -[{}]-> {}",
            if self.negated { "¬" } else { "" },
            self.subject,
            self.predicate,
            self.object
        )
    }
}

/// 一个谓词的会话假设与决定理由。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PredicateHypothesis {
    pub status: HypothesisStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

impl Default for PredicateHypothesis {
    fn default() -> Self {
        Self {
            status: HypothesisStatus::Unknown,
            reason: None,
        }
    }
}

/// 二跳链 `a → b → c` 的见证。`adds_new_fact` 表示 `a → c` 尚未直接记录。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TransitiveChain {
    pub path: Vec<String>,
    #[serde(default)]
    pub adds_new_fact: bool,
    #[serde(default)]
    pub blocked_by_negation: bool,
}

impl TransitiveChain {
    pub fn target(&self) -> Option<&str> {
        self.path.last().map(String::as_str)
    }

    pub fn render_text(&self) -> String {
        let mut text = self.path.join(" → ");
        if self.adds_new_fact {
            if let Some(target) = self.target() {
                text.push_str(&format!("（新增 {}）", target));
            }
        }
        if self.blocked_by_negation {
            text.push_str("（目标被显式否定）");
        }
        text
    }
}

/// 传递性探针：问题、支持链与显式反例。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransitiveProbe {
    pub predicate: String,
    pub question: String,
    pub support: Vec<TransitiveChain>,
    pub counterexamples: Vec<TransitiveChain>,
    pub open_world_note: String,
}

impl TransitiveProbe {
    pub fn render_text(&self) -> String {
        let mut output = format!("{}\n", self.question);
        if !self.support.is_empty() {
            output.push_str("  支持链：\n");
            for chain in &self.support {
                output.push_str(&format!("    - {}\n", chain.render_text()));
            }
        }
        if !self.counterexamples.is_empty() {
            output.push_str("  显式反例：\n");
            for chain in &self.counterexamples {
                output.push_str(&format!("    - {}\n", chain.render_text()));
            }
        }
        output.push_str(&format!("  {}\n", self.open_world_note));
        output
    }
}

/// 闭包推导出的目标（不含直接记录）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InferredTarget {
    pub target: String,
    pub path: Vec<String>,
    pub hops: usize,
}

impl InferredTarget {
    fn from_path(path: Vec<String>) -> Self {
        let hops = path.len().saturating_sub(1);
        let target = path.last().cloned().unwrap_or_default();
        Self {
            target,
            path,
            hops,
        }
    }

    pub fn render_text(&self) -> String {
        format!(
            "{}（{} 跳：{}）",
            self.target,
            self.hops,
            self.path.join(" → ")
        )
    }
}

/// 回答口径。条件答案是「若谓词可传递」的结论，不能被当成已确认事实。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAnswerKind {
    NoEvidence,
    DirectOnly,
    Conditional,
    Confirmed,
    Rejected,
}

impl OpenAnswerKind {
    pub fn label(self) -> &'static str {
        match self {
            OpenAnswerKind::NoEvidence => "无记录",
            OpenAnswerKind::DirectOnly => "仅直接记录",
            OpenAnswerKind::Conditional => "条件答案",
            OpenAnswerKind::Confirmed => "已确认闭包",
            OpenAnswerKind::Rejected => "已拒绝闭包",
        }
    }
}

/// 一次「所有」查询的完整结果：直接值、闭包值、见证、阻断与探针。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenClosureAnswer {
    pub subject: String,
    pub predicate: String,
    pub kind: OpenAnswerKind,
    pub hypothesis: HypothesisStatus,
    pub direct: Vec<String>,
    pub inferred: Vec<InferredTarget>,
    pub blocked: Vec<String>,
    pub conflicts: Vec<String>,
    pub probe: Option<TransitiveProbe>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncated: Option<ClosureTruncation>,
    #[serde(default)]
    pub witnesses_truncated: bool,
    pub reached_count: usize,
    pub expansions: usize,
    pub visited: usize,
    #[serde(default)]
    pub lower_bound: bool,
    pub note: String,
}

impl OpenClosureAnswer {
    pub fn all_targets(&self) -> Vec<&str> {
        let mut targets: Vec<&str> = self
            .direct
            .iter()
            .map(String::as_str)
            .chain(self.inferred.iter().map(|target| target.target.as_str()))
            .collect();
        targets.sort_unstable();
        targets.dedup();
        targets
    }

    pub fn is_empty(&self) -> bool {
        self.direct.is_empty() && self.inferred.is_empty()
    }

    pub fn render_text(&self) -> String {
        let mut output = format!(
            "「{}」的「{}」：{}（假设{}）\n",
            self.subject,
            self.predicate,
            self.kind.label(),
            self.hypothesis.label()
        );
        if self.direct.is_empty() {
            output.push_str("  直接记录：无\n");
        } else {
            output.push_str(&format!("  直接记录：{}\n", self.direct.join("、")));
        }
        if !self.inferred.is_empty() {
            output.push_str(&format!("  闭包推导：{} 个\n", self.inferred.len()));
            for target in self.inferred.iter().take(MAX_RENDER_TARGETS) {
                output.push_str(&format!("    - {}\n", target.render_text()));
            }
            if self.inferred.len() > MAX_RENDER_TARGETS {
                output.push_str(&format!(
                    "    ……另有 {} 个结果未展开\n",
                    self.inferred.len() - MAX_RENDER_TARGETS
                ));
            }
        }
        if !self.blocked.is_empty() {
            output.push_str(&format!(
                "  被显式否定阻断：{}\n",
                self.blocked.join("、")
            ));
        }
        for conflict in &self.conflicts {
            output.push_str(&format!("  冲突：{}\n", conflict));
        }
        if let Some(truncation) = self.truncated {
            output.push_str(&format!(
                "  说明：闭包搜索在{}处停止，结果计数是下界。\n",
                truncation.label()
            ));
        } else if self.witnesses_truncated {
            output.push_str("  说明：见证列表已截断，实际可达目标可能更多。\n");
        }
        output.push_str(&format!("  {}\n", self.note));
        if let Some(probe) = &self.probe {
            output.push_str(&probe.render_text());
        }
        output
    }
}

/// 会话级开放事实与假设账本。所有集合有序，JSON 输出可复现。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenClosureSession {
    #[serde(default = "default_dialect")]
    pub dialect: String,
    #[serde(default)]
    facts: Vec<OpenFact>,
    #[serde(default)]
    hypotheses: BTreeMap<String, PredicateHypothesis>,
    #[serde(default)]
    probes_asked: BTreeSet<String>,
}

fn default_dialect() -> String {
    OPEN_CLOSURE_DIALECT.to_string()
}

impl Default for OpenClosureSession {
    fn default() -> Self {
        Self::new()
    }
}

impl OpenClosureSession {
    pub fn new() -> Self {
        Self {
            dialect: default_dialect(),
            facts: Vec::new(),
            hypotheses: BTreeMap::new(),
            probes_asked: BTreeSet::new(),
        }
    }

    pub fn facts(&self) -> &[OpenFact] {
        &self.facts
    }

    pub fn len(&self) -> usize {
        self.facts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.facts.is_empty()
    }

    pub fn hypotheses(&self) -> &BTreeMap<String, PredicateHypothesis> {
        &self.hypotheses
    }

    pub fn hypothesis(&self, predicate: &str) -> PredicateHypothesis {
        self.hypotheses
            .get(predicate.trim())
            .cloned()
            .unwrap_or_default()
    }

    pub fn status(&self, predicate: &str) -> HypothesisStatus {
        self.hypothesis(predicate).status
    }

    pub fn probe_was_asked(&self, predicate: &str) -> bool {
        self.probes_asked.contains(predicate.trim())
    }

    /// 登记一条开放事实；正负事实分开保存，便于把显式否定当反例。
    pub fn observe_triple(
        &mut self,
        subject: &str,
        predicate: &str,
        object: &str,
        source: &str,
    ) -> bool {
        self.observe_polarity(subject, predicate, object, false, source)
    }

    pub fn observe_negative(
        &mut self,
        subject: &str,
        predicate: &str,
        object: &str,
        source: &str,
    ) -> bool {
        self.observe_polarity(subject, predicate, object, true, source)
    }

    fn observe_polarity(
        &mut self,
        subject: &str,
        predicate: &str,
        object: &str,
        negated: bool,
        source: &str,
    ) -> bool {
        let subject = subject.trim();
        let predicate = predicate.trim();
        let object = object.trim();
        if subject.is_empty() || predicate.is_empty() || object.is_empty() {
            return false;
        }
        let source = if source.trim().is_empty() {
            "会话观察".to_string()
        } else {
            source.trim().to_string()
        };

        if let Some(existing) = self.facts.iter_mut().find(|fact| {
            fact.subject == subject
                && fact.predicate == predicate
                && fact.object == object
                && fact.negated == negated
        }) {
            return existing.sources.insert(source);
        }

        self.facts.push(OpenFact {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
            negated,
            sources: BTreeSet::from([source]),
        });
        self.sort_facts();
        true
    }

    /// 撤回一条事实（正负都删）；返回是否真的删除了记录。
    pub fn retract_triple(&mut self, subject: &str, predicate: &str, object: &str) -> bool {
        let subject = subject.trim();
        let predicate = predicate.trim();
        let object = object.trim();
        let before = self.facts.len();
        self.facts.retain(|fact| {
            !(fact.subject == subject && fact.predicate == predicate && fact.object == object)
        });
        before != self.facts.len()
    }

    /// 吸收关系帧：只收直陈/必然、非条件、非时间限定的事实；
    /// 否定帧保留为显式反例。返回新增记录数。
    pub fn observe_frames(&mut self, frames: &[RelationFrame]) -> usize {
        let mut added = 0usize;
        for frame in frames {
            if !matches!(frame.modality, Modality::Asserted | Modality::Necessary) {
                continue;
            }
            if frame
                .qualifiers
                .iter()
                .any(|qualifier| qualifier == "条件/因果" || qualifier == "时间限定")
            {
                continue;
            }
            let source = if frame.evidence.sentence.trim().is_empty() {
                frame.id.clone()
            } else {
                frame.evidence.sentence.trim().to_string()
            };
            let negated = frame.polarity == Polarity::Negative;
            if self.observe_polarity(
                &frame.subject,
                &frame.predicate,
                &frame.object,
                negated,
                &source,
            ) {
                added += 1;
            }
        }
        added
    }

    /// 从自然语言文本吸收开放事实；返回新增记录数。
    pub fn observe_text(&mut self, text: &str) -> usize {
        let frames = FrameExtractor::new().extract(text);
        self.observe_frames(&frames)
    }

    /// 用户确认某谓词在本会话内可传递。重复确认返回 false。
    pub fn confirm_transitive(&mut self, predicate: &str, reason: &str) -> bool {
        self.set_hypothesis(predicate, HypothesisStatus::Confirmed, reason)
    }

    /// 用户拒绝某谓词的传递假设。重复拒绝返回 false。
    pub fn reject_transitive(&mut self, predicate: &str, reason: &str) -> bool {
        self.set_hypothesis(predicate, HypothesisStatus::Rejected, reason)
    }

    /// 清空假设，回到未确认状态，并允许再次询问。
    pub fn reset_hypothesis(&mut self, predicate: &str) -> bool {
        let predicate = predicate.trim();
        let removed = self.hypotheses.remove(predicate).is_some();
        self.probes_asked.remove(predicate);
        removed
    }

    fn set_hypothesis(&mut self, predicate: &str, status: HypothesisStatus, reason: &str) -> bool {
        let predicate = predicate.trim();
        if predicate.is_empty() {
            return false;
        }
        if self.status(predicate) == status {
            return false;
        }
        self.hypotheses.insert(
            predicate.to_string(),
            PredicateHypothesis {
                status,
                reason: if reason.trim().is_empty() {
                    None
                } else {
                    Some(reason.trim().to_string())
                },
            },
        );
        true
    }

    /// 回答「`subject` 的所有 `predicate`」。
    ///
    /// 未确认假设时给条件答案，并（每个谓词每会话最多一次）附上传递性探针；
    /// 已确认时给闭包答案；已拒绝时只给直接记录。
    pub fn answer(
        &mut self,
        subject: &str,
        predicate: &str,
        budget: ClosureBudget,
    ) -> OpenClosureAnswer {
        let subject = subject.trim().to_string();
        let predicate = predicate.trim().to_string();
        let mut answer = self.build_answer(&subject, &predicate, budget, true);
        if let Some(probe) = answer.probe.take() {
            answer.probe = Some(probe);
            self.probes_asked.insert(predicate);
        }
        answer
    }

    /// 只读查询：不消耗「每会话只问一次」的探针额度。
    pub fn peek_answer(
        &self,
        subject: &str,
        predicate: &str,
        budget: ClosureBudget,
    ) -> OpenClosureAnswer {
        self.build_answer(
            subject.trim(),
            predicate.trim(),
            budget,
            false,
        )
    }

    fn build_answer(
        &self,
        subject: &str,
        predicate: &str,
        budget: ClosureBudget,
        allow_probe: bool,
    ) -> OpenClosureAnswer {
        let status = self.status(predicate);
        let adjacency = self.positive_adjacency(predicate);
        let negatives = self.negative_objects(predicate);
        let direct: Vec<String> = adjacency
            .get(subject)
            .map(|values| values.iter().cloned().collect())
            .unwrap_or_default();

        let chains = two_hop_chains(&adjacency, &negatives);
        let has_query_chain = chains
            .iter()
            .any(|chain| chain.path.first().map(String::as_str) == Some(subject));
        let mut support: Vec<TransitiveChain> = chains
            .iter()
            .filter(|chain| !chain.blocked_by_negation)
            .cloned()
            .collect();
        support.sort_by(|left, right| {
            let left_matches = left.path.first().map(String::as_str) == Some(subject);
            let right_matches = right.path.first().map(String::as_str) == Some(subject);
            right_matches
                .cmp(&left_matches)
                .then_with(|| right.adds_new_fact.cmp(&left.adds_new_fact))
                .then_with(|| left.path.cmp(&right.path))
        });
        support.truncate(MAX_PROBE_CHAINS);
        let mut counterexamples: Vec<TransitiveChain> = chains
            .into_iter()
            .filter(|chain| chain.blocked_by_negation)
            .collect();
        counterexamples.truncate(MAX_PROBE_CHAINS);

        let mut conflicts: Vec<String> = counterexamples
            .iter()
            .map(|chain| {
                format!(
                    "二跳链 {} 的终点被显式否定，不能据传递性推出该目标",
                    chain.path.join(" → ")
                )
            })
            .collect();

        let query_probe_available = status == HypothesisStatus::Unknown
            && !support.is_empty()
            && has_query_chain
            && !self.probes_asked.contains(predicate);
        let probe = if allow_probe && query_probe_available {
            Some(build_probe(predicate, &support, &counterexamples))
        } else {
            None
        };

        if status == HypothesisStatus::Rejected {
            return OpenClosureAnswer {
                subject: subject.to_string(),
                predicate: predicate.to_string(),
                kind: OpenAnswerKind::Rejected,
                hypothesis: status,
                direct,
                inferred: Vec::new(),
                blocked: Vec::new(),
                conflicts,
                probe: None,
                truncated: None,
                witnesses_truncated: false,
                reached_count: 0,
                expansions: 0,
                visited: 0,
                lower_bound: false,
                note: "已在本会话拒绝该谓词的传递假设，只保留直接记录；知识库未被修改。"
                    .to_string(),
            };
        }

        let should_close = status == HypothesisStatus::Confirmed
            || (status == HypothesisStatus::Unknown && has_query_chain);
        if !should_close {
            let kind = if direct.is_empty() {
                OpenAnswerKind::NoEvidence
            } else {
                OpenAnswerKind::DirectOnly
            };
            let note = if support.is_empty() {
                "会话中没有可检验的二跳链；缺少直接记录不代表否定。".to_string()
            } else {
                "该谓词在会话中有二跳链，但本次查询起点没有可延伸路径；缺少直接记录不代表否定。".to_string()
            };
            return OpenClosureAnswer {
                subject: subject.to_string(),
                predicate: predicate.to_string(),
                kind,
                hypothesis: status,
                direct,
                inferred: Vec::new(),
                blocked: Vec::new(),
                conflicts,
                probe: None,
                truncated: None,
                witnesses_truncated: false,
                reached_count: 0,
                expansions: 0,
                visited: 0,
                lower_bound: false,
                note,
            };
        }

        let law = format!("session:transitive:{}", predicate);
        let outcome = match evaluate_closure(
            subject,
            ClosureOp::Plus,
            budget,
            Some(law.as_str()),
            |node| {
                adjacency
                    .get(node)
                    .map(|values| values.iter().cloned().collect())
                    .unwrap_or_default()
            },
        ) {
            Ok(outcome) => outcome,
            Err(_) => {
                return OpenClosureAnswer {
                    subject: subject.to_string(),
                    predicate: predicate.to_string(),
                    kind: OpenAnswerKind::NoEvidence,
                    hypothesis: status,
                    direct,
                    inferred: Vec::new(),
                    blocked: Vec::new(),
                    conflicts,
                    probe: None,
                    truncated: None,
                    witnesses_truncated: false,
                    reached_count: 0,
                    expansions: 0,
                    visited: 0,
                    lower_bound: false,
                    note: "闭包预算不可用。".to_string(),
                };
            }
        };

        let direct_set: BTreeSet<&str> = direct.iter().map(String::as_str).collect();
        let mut inferred: Vec<InferredTarget> = Vec::new();
        let mut blocked: Vec<String> = Vec::new();
        let mut blocked_set: BTreeSet<String> = BTreeSet::new();
        for (target, path) in &outcome.witnesses {
            if path.len() <= 2 {
                continue;
            }
            if direct_set.contains(target.as_str()) {
                continue;
            }
            if negatives.contains(target) {
                if blocked_set.insert(target.clone()) {
                    blocked.push(target.clone());
                }
                continue;
            }
            inferred.push(InferredTarget::from_path(path.clone()));
        }
        inferred.sort_by(|left, right| {
            left.hops
                .cmp(&right.hops)
                .then_with(|| left.target.cmp(&right.target))
        });
        blocked.sort();
        if !blocked.is_empty() {
            conflicts.push(format!(
                "以下目标已有显式否定，已从闭包结果中剔除：{}",
                blocked.join("、")
            ));
        }

        let kind = match status {
            HypothesisStatus::Confirmed => OpenAnswerKind::Confirmed,
            HypothesisStatus::Unknown => OpenAnswerKind::Conditional,
            HypothesisStatus::Rejected => OpenAnswerKind::Rejected,
        };
        let lower_bound = outcome.count_is_lower_bound();
        let note = match status {
            HypothesisStatus::Confirmed => {
                "传递性已由本会话确认（未写入知识库或律典）；结果均附带见证路径。".to_string()
            }
            HypothesisStatus::Unknown => {
                "这是条件答案：仅在「该谓词可传递」的假设下成立；确认后才会作为会话结论复用。"
                    .to_string()
            }
            HypothesisStatus::Rejected => String::new(),
        };

        OpenClosureAnswer {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            kind,
            hypothesis: status,
            direct,
            inferred,
            blocked,
            conflicts,
            probe,
            truncated: outcome.truncated,
            witnesses_truncated: outcome.witnesses_truncated,
            reached_count: outcome.reached_count,
            expansions: outcome.expansions,
            visited: outcome.visited,
            lower_bound,
            note,
        }
    }

    fn positive_adjacency(&self, predicate: &str) -> BTreeMap<String, BTreeSet<String>> {
        let mut adjacency: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for fact in &self.facts {
            if fact.negated || fact.predicate != predicate {
                continue;
            }
            adjacency
                .entry(fact.subject.clone())
                .or_default()
                .insert(fact.object.clone());
        }
        adjacency
    }

    fn negative_objects(&self, predicate: &str) -> BTreeSet<String> {
        self.facts
            .iter()
            .filter(|fact| fact.negated && fact.predicate == predicate)
            .map(|fact| fact.object.clone())
            .collect()
    }

    /// 简要账本：谓词、正负事实数与假设状态。
    pub fn render_text(&self) -> String {
        let positives = self.facts.iter().filter(|fact| !fact.negated).count();
        let negatives = self.len().saturating_sub(positives);
        let mut output = format!(
            "开放闭包会话：{} 条记录（肯定 {} / 否定 {}），{} 个谓词。\n",
            self.len(),
            positives,
            negatives,
            self.predicates().len()
        );
        if self.facts.is_empty() {
            output.push_str("尚无开放事实。\n");
            return output;
        }
        output.push_str("谓词：\n");
        for predicate in self.predicates() {
            let status = self.status(&predicate);
            let positive = self
                .facts
                .iter()
                .filter(|fact| fact.predicate == predicate && !fact.negated)
                .count();
            let negative = self
                .facts
                .iter()
                .filter(|fact| fact.predicate == predicate && fact.negated)
                .count();
            output.push_str(&format!(
                "  「{}」 {}（肯定 {} / 否定 {}）\n",
                predicate,
                status.label(),
                positive,
                negative
            ));
        }
        output
    }

    pub fn predicates(&self) -> BTreeSet<String> {
        self.facts
            .iter()
            .map(|fact| fact.predicate.clone())
            .collect()
    }

    fn sort_facts(&mut self) {
        self.facts.sort_by(|left, right| left.key().cmp(&right.key()));
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    pub fn from_json(text: &str) -> Option<Self> {
        let mut session: Self = serde_json::from_str(text).ok()?;
        if session.dialect.trim().is_empty() {
            session.dialect = default_dialect();
        }
        session.sort_facts();
        Some(session)
    }

    pub fn save_to_path(&self, path: &str) -> std::io::Result<()> {
        std::fs::write(path, self.to_json())
    }

    pub fn load_from_path(path: &str) -> Option<Self> {
        let text = std::fs::read_to_string(path).ok()?;
        Self::from_json(&text)
    }
}

fn build_probe(
    predicate: &str,
    support: &[TransitiveChain],
    counterexamples: &[TransitiveChain],
) -> TransitiveProbe {
    let examples = support
        .iter()
        .map(|chain| chain.render_text())
        .collect::<Vec<_>>();
    let question = if examples.is_empty() {
        format!("「{}」是可传递关系吗？", predicate)
    } else {
        format!(
            "「{}」看起来可传递（例如 {}）。要把「{}」登记为本会话的传递关系吗？",
            predicate,
            examples.join("；"),
            predicate
        )
    };
    TransitiveProbe {
        predicate: predicate.to_string(),
        question,
        support: support.to_vec(),
        counterexamples: counterexamples.to_vec(),
        open_world_note: "开放世界：缺少 a → c 的直接记录不是反例；只有显式否定才计入反例。"
            .to_string(),
    }
}

fn two_hop_chains(
    adjacency: &BTreeMap<String, BTreeSet<String>>,
    negatives: &BTreeSet<String>,
) -> Vec<TransitiveChain> {
    let mut chains: Vec<TransitiveChain> = Vec::new();
    for (start, middles) in adjacency {
        let direct_targets = adjacency.get(start);
        for middle in middles {
            let Some(targets) = adjacency.get(middle) else {
                continue;
            };
            for target in targets {
                if target == start {
                    continue;
                }
                let adds_new_fact = direct_targets
                    .map(|values| !values.contains(target))
                    .unwrap_or(true);
                chains.push(TransitiveChain {
                    path: vec![start.clone(), middle.clone(), target.clone()],
                    adds_new_fact,
                    blocked_by_negation: negatives.contains(target),
                });
            }
        }
    }
    chains.sort_by(|left, right| {
        right
            .adds_new_fact
            .cmp(&left.adds_new_fact)
            .then_with(|| left.path.cmp(&right.path))
    });
    chains
}

/// 把用户提问解析成 `(subject, predicate)`。
///
/// 显式格式 `主语|谓词` 最稳；也接受常见中文闭包问法：
/// 「张三的所有师兄是谁？」「张三的师兄有哪些？」。
pub fn parse_query(text: &str) -> Result<(String, String), String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err("查询为空".to_string());
    }
    if let Some((subject, predicate)) = split_explicit_query(trimmed) {
        return validate_query(subject, predicate);
    }
    let cleaned = trimmed
        .trim_matches(|ch: char| {
            matches!(
                ch,
                '？' | '?' | '。' | '！' | '!' | ' ' | '\t' | '\n' | '\r'
            )
        })
        .trim();
    let cleaned = cleaned
        .strip_prefix("查询")
        .unwrap_or(cleaned)
        .trim();

    const TAILS: [&str; 12] = [
        "分别是谁",
        "都是谁",
        "是哪些",
        "有哪些",
        "有那些",
        "是什么",
        "是哪个",
        "都有谁",
        "有谁",
        "是谁",
        "多少",
        "呢",
    ];

    if let Some(position) = cleaned.find("的所有") {
        let subject = cleaned[..position].trim();
        let rest = cleaned[position + "的所有".len()..].trim();
        let predicate = strip_query_tail(rest, &TAILS);
        return validate_query(subject, predicate);
    }
    if let Some(position) = cleaned.find('的') {
        let subject = cleaned[..position].trim();
        let rest = cleaned[position + '的'.len_utf8()..].trim();
        let predicate = strip_query_tail(rest, &TAILS);
        if predicate != rest {
            return validate_query(subject, predicate);
        }
    }
    Err("无法解析查询；请用 SUBJECT|PREDICATE 或「X 的所有 Y 是谁？」".to_string())
}

fn split_explicit_query(text: &str) -> Option<(&str, &str)> {
    for separator in ['|', '｜'] {
        if let Some(position) = text.find(separator) {
            let subject = text[..position].trim();
            let predicate = text[position + separator.len_utf8()..].trim();
            return Some((subject, predicate));
        }
    }
    None
}

fn strip_query_tail<'a>(text: &'a str, tails: &[&str]) -> &'a str {
    for tail in tails {
        if let Some(stripped) = text.strip_suffix(tail) {
            return stripped.trim();
        }
    }
    text.trim()
}

fn validate_query(subject: &str, predicate: &str) -> Result<(String, String), String> {
    let subject = subject.trim();
    let predicate = predicate.trim();
    if subject.is_empty() {
        return Err("查询缺少主语".to_string());
    }
    if predicate.is_empty() {
        return Err("查询缺少谓词".to_string());
    }
    if subject.chars().count() > 64 || predicate.chars().count() > 64 {
        return Err("主语或谓词过长".to_string());
    }
    Ok((subject.to_string(), predicate.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session() -> OpenClosureSession {
        let mut session = OpenClosureSession::new();
        session.observe_triple("张三", "师兄", "李四", "测试");
        session.observe_triple("李四", "师兄", "王五", "测试");
        session
    }

    #[test]
    fn unknown_hypothesis_answers_conditionally_with_probe() {
        let mut session = session();
        let answer = session.answer("张三", "师兄", ClosureBudget::default());
        assert_eq!(answer.kind, OpenAnswerKind::Conditional);
        assert_eq!(answer.hypothesis, HypothesisStatus::Unknown);
        assert_eq!(answer.direct, vec!["李四"]);
        assert_eq!(
            answer
                .inferred
                .iter()
                .map(|target| target.target.as_str())
                .collect::<Vec<_>>(),
            vec!["王五"]
        );
        assert_eq!(answer.inferred[0].hops, 2);
        let probe = answer.probe.expect("二跳链应触发探针");
        assert_eq!(probe.predicate, "师兄");
        assert!(probe.support[0].adds_new_fact);
        assert!(probe.question.contains("可传递"));
    }

    #[test]
    fn confirmation_turns_conditional_into_closure_and_rejection_retracts_it() {
        let mut session = session();
        assert!(session.confirm_transitive("师兄", "用户确认"));
        let answer = session.answer("张三", "师兄", ClosureBudget::default());
        assert_eq!(answer.kind, OpenAnswerKind::Confirmed);
        assert_eq!(answer.hypothesis, HypothesisStatus::Confirmed);
        assert!(answer.probe.is_none());
        assert_eq!(answer.all_targets(), vec!["李四", "王五"]);
        assert!(answer.render_text().contains("张三 → 李四 → 王五"));

        assert!(session.reject_transitive("师兄", "用户拒绝"));
        let rejected = session.answer("张三", "师兄", ClosureBudget::default());
        assert_eq!(rejected.kind, OpenAnswerKind::Rejected);
        assert_eq!(rejected.direct, vec!["李四"]);
        assert!(rejected.inferred.is_empty());
        assert!(rejected.probe.is_none());
    }

    #[test]
    fn probe_is_asked_at_most_once_per_predicate() {
        let mut session = session();
        let first = session.answer("张三", "师兄", ClosureBudget::default());
        assert!(first.probe.is_some());
        let second = session.answer("张三", "师兄", ClosureBudget::default());
        assert!(second.probe.is_none());
        assert!(second.render_text().contains("条件答案"));
        let peeked = session.peek_answer("张三", "师兄", ClosureBudget::default());
        assert!(peeked.probe.is_none());
    }

    #[test]
    fn explicit_negation_blocks_the_inferred_target() {
        let mut session = session();
        session.observe_negative("李四", "师兄", "王五", "测试反例");
        let answer = session.answer("张三", "师兄", ClosureBudget::default());
        assert_eq!(answer.kind, OpenAnswerKind::Conditional);
        assert_eq!(answer.direct, vec!["李四"]);
        assert!(answer.inferred.is_empty());
        assert_eq!(answer.blocked, vec!["王五"]);
        assert!(!answer.conflicts.is_empty());
    }

    #[test]
    fn negation_of_target_is_a_counterexample_not_an_open_world_gap() {
        let mut session = session();
        session.observe_triple("王五", "师兄", "赵六", "测试");
        session.observe_negative("张三", "师兄", "王五", "明确否认");
        let answer = session.peek_answer("张三", "师兄", ClosureBudget::default());
        assert_eq!(answer.kind, OpenAnswerKind::Conditional);
        assert!(answer.blocked.contains(&"王五".to_string()));
        assert!(answer
            .inferred
            .iter()
            .all(|target| target.target != "王五"));
        assert!(!answer.conflicts.is_empty());
    }

    #[test]
    fn text_observation_uses_open_extraction_without_predicate_tables() {
        let mut session = OpenClosureSession::new();
        let added = session.observe_text("张三是李四的师兄。李四是王五的师兄。");
        assert!(added >= 2, "实际抽取 {:?}", session.facts());
        assert!(session.facts().iter().any(|fact| fact.predicate == "师兄"));
        let answer = session.peek_answer("王五", "师兄", ClosureBudget::default());
        assert_eq!(answer.direct, vec!["李四"]);
        assert_eq!(
            answer
                .inferred
                .iter()
                .map(|target| target.target.as_str())
                .collect::<Vec<_>>(),
            vec!["张三"]
        );
    }

    #[test]
    fn budget_truncation_is_disclosed_and_retraction_removes_support() {
        let mut session = OpenClosureSession::new();
        session.observe_triple("甲", "p", "乙", "测试");
        session.observe_triple("乙", "p", "丙", "测试");
        session.observe_triple("丙", "p", "丁", "测试");
        session.confirm_transitive("p", "测试");
        let answer = session.answer(
            "甲",
            "p",
            ClosureBudget {
                max_depth: 1,
                ..ClosureBudget::default()
            },
        );
        assert_eq!(answer.truncated, Some(ClosureTruncation::Depth));
        assert!(answer.lower_bound);

        assert!(session.retract_triple("乙", "p", "丙"));
        let after = session.answer("甲", "p", ClosureBudget::default());
        assert!(after.inferred.is_empty());
    }

    #[test]
    fn parse_query_accepts_explicit_and_chinese_forms() {
        assert_eq!(
            parse_query("张三|师兄").unwrap(),
            ("张三".to_string(), "师兄".to_string())
        );
        assert_eq!(
            parse_query("张三的所有师兄是谁？").unwrap(),
            ("张三".to_string(), "师兄".to_string())
        );
        assert_eq!(
            parse_query("王五的师兄有哪些？").unwrap(),
            ("王五".to_string(), "师兄".to_string())
        );
        assert_eq!(
            parse_query(" 查询 李四的所有老师都是谁 ").unwrap(),
            ("李四".to_string(), "老师".to_string())
        );
        assert!(parse_query("没有分隔符的句子").is_err());
    }

    #[test]
    fn json_round_trip_preserves_facts_hypotheses_and_probe_fatigue() {
        let mut session = session();
        let first = session.answer("张三", "师兄", ClosureBudget::default());
        assert!(first.probe.is_some());
        assert!(session.confirm_transitive("师兄", "用户确认"));
        let restored = OpenClosureSession::from_json(&session.to_json()).expect("会话应可回读");
        assert_eq!(restored, session);
        assert_eq!(restored.status("师兄"), HypothesisStatus::Confirmed);
        assert!(restored.probe_was_asked("师兄"));
    }
}

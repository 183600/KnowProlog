//! 有界推导视图：把已有的归结证明投影成对话可消费的平铺结构。
//!
//! 设计边界：
//!   - 不新造推理算法：视图直接来自 [`ResolutionProofView`] 的事实/规则证明树，
//!     或来自对话知识源中的直接事实边；没有证明就显式标记，绝不补造步骤；
//!   - 证明不进入对话快照：这里只产出短生命周期视图，节点数、深度、字节与
//!     渲染长度都有上限，避免递归证明树在会话记忆中无界增长；
//!   - 平铺存储：渲染与序列化都消费 [`DerivationStep`] 列表，不递归遍历；
//!   - 三态忠实：得证 / 已穷尽无解 / 搜索被截断互不混淆，失败前沿随视图返回。

use serde::{Deserialize, Serialize};

use crate::resolution_engine::{FailureFrontier, SearchStatus};
use crate::symbolic_reasoner::ResolutionProofView;
use crate::view_render::{ProofViewNode, ViewProofSource};

/// 单个推导最多保留的证明节点数。
pub const MAX_DERIVATION_STEPS: usize = 96;
/// 单个推导最多保留的证明深度。
pub const MAX_DERIVATION_DEPTH: usize = 24;
/// 单个推导保留的展示字段总字节上限。
pub const MAX_DERIVATION_BYTES: usize = 256 * 1024;
/// 对用户渲染时最多展示的步骤数。
pub const MAX_RENDER_STEPS: usize = 12;
/// 对用户渲染时最多展示的失败前沿条数。
pub const MAX_RENDER_FRONTIER: usize = 4;
/// 单次解释文本的字符上限。
pub const MAX_EXPLANATION_CHARS: usize = 1800;

/// 推导的三种认识论状态。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DerivationStatus {
    /// 找到了至少一个完整证明。
    Proved,
    /// 搜索已穷尽仍无解（开世界假设：推不出不等于不存在）。
    NoSolution,
    /// 搜索被深度/预算/解数上限截断，不能据此断言不存在。
    Truncated,
}

impl DerivationStatus {
    pub fn code(self) -> &'static str {
        match self {
            DerivationStatus::Proved => "proved",
            DerivationStatus::NoSolution => "no_solution",
            DerivationStatus::Truncated => "truncated",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            DerivationStatus::Proved => "得证",
            DerivationStatus::NoSolution => "无解（已穷尽）",
            DerivationStatus::Truncated => "未得证（搜索被截断）",
        }
    }
}

/// 单步证明的来源。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DerivationSource {
    /// 图谱事实或会话事实。
    Fact,
    /// 一条具名规则（链接预测规则、关联规则、会话规则或语义律）。
    Rule { source: String },
}

/// 平铺后的一个证明节点。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DerivationStep {
    /// 在证明树中的缩进层级（根为 0）。
    pub depth: usize,
    /// 自然语言展示标签。
    pub claim: String,
    /// 规范逻辑形式（稳定身份，不随措辞变化）。
    pub canonical: String,
    pub source: DerivationSource,
    pub confidence: f64,
}

impl DerivationStep {
    fn source_text(&self) -> String {
        match &self.source {
            DerivationSource::Fact => "事实".to_string(),
            DerivationSource::Rule { source } => {
                format!("由 {}，本步置信度 {:.2}", source, self.confidence)
            }
        }
    }
}

/// 一次可解释的推导（平铺、有界、可渲染）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Derivation {
    /// 被证明的目标（逻辑形式）。
    pub goal: String,
    pub status: DerivationStatus,
    /// 证明的总置信度（无完整证明时为 None）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
    /// 查询变量绑定（按变量名排序）。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub bindings: Vec<(String, String)>,
    /// 平铺后的证明节点（前序遍历）。
    pub steps: Vec<DerivationStep>,
    /// 本次搜索实际遇到的代表性知识缺口。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub frontier: Vec<String>,
    /// 因容量上限未保留的缺口条数。
    #[serde(default)]
    pub frontier_omitted: usize,
    /// 知识库事实数与规则数（帮助读者判断证明的上下文）。
    pub facts: usize,
    pub rules: usize,
    /// 因节点/深度/字节预算未展示的节点数。
    #[serde(default)]
    pub omitted_steps: usize,
    /// 节点/深度/字节预算是否截断了证明。
    #[serde(default)]
    pub truncated: bool,
    /// 同一目标的其他独立证明数（只计数，不展开，避免解释无界增长）。
    #[serde(default)]
    pub alternatives: usize,
}

impl Derivation {
    pub fn is_proved(&self) -> bool {
        self.status == DerivationStatus::Proved
    }

    /// 主答案绑定值：优先「答案」变量，否则第一个非空绑定。
    pub fn answer_value(&self) -> Option<&str> {
        self.bindings
            .iter()
            .find(|(name, value)| name == "答案" && !value.is_empty())
            .or_else(|| self.bindings.iter().find(|(_, value)| !value.is_empty()))
            .map(|(_, value)| value.as_str())
    }

    /// 从结构化归结视图取一条证明。
    ///
    /// `expected_answer` 非空时必须能在某个解的绑定中找到该值，否则返回
    /// `None`（宁可不解释，也不把别的解的证明挂到当前回答上）。
    pub fn from_resolution_view(
        view: &ResolutionProofView,
        expected_answer: Option<&str>,
    ) -> Option<Self> {
        let selected = if view.solutions.is_empty() {
            None
        } else if let Some(expected) = expected_answer {
            view.solutions.iter().find(|solution| {
                solution.bindings.is_empty()
                    || solution
                        .bindings
                        .iter()
                        .any(|(_, value)| !value.is_empty() && expected.contains(value))
            })
        } else {
            view.solutions.first()
        };
        if !view.solutions.is_empty() && selected.is_none() {
            return None;
        }

        let mut derivation = Derivation {
            goal: clip_chars(&view.goal_display, 240),
            status: if selected.is_some() {
                DerivationStatus::Proved
            } else if view.search_status == SearchStatus::Incomplete || view.search_truncated {
                DerivationStatus::Truncated
            } else {
                DerivationStatus::NoSolution
            },
            confidence: None,
            bindings: Vec::new(),
            steps: Vec::new(),
            frontier: view
                .frontier
                .iter()
                .take(MAX_RENDER_FRONTIER)
                .map(FailureFrontier::describe)
                .collect(),
            frontier_omitted: view
                .frontier
                .len()
                .saturating_sub(MAX_RENDER_FRONTIER)
                + view.stats.frontier_omitted,
            facts: view.facts,
            rules: view.clauses,
            omitted_steps: 0,
            truncated: view.search_truncated,
            alternatives: view.solutions.len().saturating_sub(1),
        };

        if let Some(solution) = selected {
            let mut budget = StepBudget::new();
            for proof in &solution.proofs {
                budget.push_forest(proof, &mut derivation);
            }
            derivation.bindings = solution.bindings.clone();
            derivation.confidence = Some(solution.confidence);
        }
        Some(derivation)
    }

    /// 没有证明的显式视图：用于「为什么没有记录」这类追问，
    /// 只携带本次检索/搜索实际遇到的缺口，不伪造任何步骤。
    pub fn missing(goal: impl Into<String>, frontier: Vec<String>) -> Self {
        let omitted = frontier.len().saturating_sub(MAX_RENDER_FRONTIER);
        Derivation {
            goal: goal.into(),
            status: DerivationStatus::NoSolution,
            confidence: None,
            bindings: Vec::new(),
            steps: Vec::new(),
            frontier: frontier.into_iter().take(MAX_RENDER_FRONTIER).collect(),
            frontier_omitted: omitted,
            facts: 0,
            rules: 0,
            omitted_steps: 0,
            truncated: false,
            alternatives: 0,
        }
    }

    /// 直接事实链：每个 `(subject, predicate, object)` 是一条图谱事实边。
    /// 用于解释「查表/检索得到的属性值」这类没有归结规则参与的答案。
    pub fn from_direct_facts(
        goal: impl Into<String>,
        facts: impl IntoIterator<Item = (String, String, String)>,
    ) -> Self {
        let mut steps = Vec::new();
        let mut first_value = None;
        for (subject, predicate, object) in facts {
            if steps.len() >= MAX_DERIVATION_STEPS {
                break;
            }
            if first_value.is_none() {
                first_value = Some(object.clone());
            }
            steps.push(DerivationStep {
                depth: 0,
                claim: clip_chars(
                    &format!("{} 的 {} 是 {}", subject, predicate, object),
                    240,
                ),
                canonical: clip_chars(&format!("{}({}, {})", predicate, subject, object), 240),
                source: DerivationSource::Fact,
                confidence: 1.0,
            });
        }
        let count = steps.len();
        Derivation {
            goal: goal.into(),
            status: if count == 0 {
                DerivationStatus::NoSolution
            } else {
                DerivationStatus::Proved
            },
            confidence: if count == 0 { None } else { Some(1.0) },
            bindings: first_value
                .map(|value| vec![("答案".to_string(), value)])
                .unwrap_or_default(),
            steps,
            frontier: Vec::new(),
            frontier_omitted: 0,
            facts: count,
            rules: 0,
            omitted_steps: 0,
            truncated: false,
            alternatives: 0,
        }
    }

    /// 渲染成中文解释（受步数与字符上限约束）。
    pub fn render_explanation(&self) -> String {
        let mut output = String::new();
        match self.status {
            DerivationStatus::Proved => {
                output.push_str(&format!(
                    "这个结论有一条可展开的推导（{} 条事实、{} 条规则）。\n",
                    self.facts, self.rules
                ));
                if !self.goal.is_empty() {
                    output.push_str(&format!("  目标：{}\n", self.goal));
                }
                if let Some(value) = self.answer_value() {
                    output.push_str(&format!("  结论：{}\n", value));
                }
                if self.steps.is_empty() {
                    output.push_str("  证明节点已达到显示预算，未展开具体步骤。\n");
                } else {
                    output.push_str("  证明步骤：\n");
                    for (index, step) in self
                        .steps
                        .iter()
                        .take(MAX_RENDER_STEPS)
                        .enumerate()
                    {
                        let indent = "  ".repeat(step.depth.min(6));
                        output.push_str(&format!(
                            "    {}{}. {}（{}）\n",
                            indent,
                            index + 1,
                            step.claim,
                            step.source_text()
                        ));
                    }
                    if self.steps.len() > MAX_RENDER_STEPS {
                        output.push_str(&format!(
                            "    ……另有 {} 个证明节点未展开。\n",
                            self.steps.len() - MAX_RENDER_STEPS
                        ));
                    }
                }
                if self.omitted_steps > 0 {
                    output.push_str(&format!(
                        "  说明：至少 {} 个证明节点因节点/深度/字节预算未展开。\n",
                        self.omitted_steps
                    ));
                }
                if let Some(confidence) = self.confidence {
                    output.push_str(&format!("  证明置信度：{:.2}。\n", confidence));
                }
                if self.alternatives > 0 {
                    output.push_str(&format!(
                        "  另有 {} 条独立证明（同一目标的不同解，本次只展开一条）。\n",
                        self.alternatives
                    ));
                }
            }
            DerivationStatus::NoSolution => {
                output.push_str(
                    "没有可展开的完整证明：这次归结搜索已穷尽所有分支，仍未能证明该目标。\n",
                );
                output.push_str("  这只能说明当前事实与规则推不出它；开世界假设下，推不出不等于不存在。\n");
                self.render_frontier(&mut output);
            }
            DerivationStatus::Truncated => {
                output.push_str(
                    "没有完整的证明：这次归结搜索触发了深度或预算上限，结果被截断。\n",
                );
                output.push_str("  因此不能据此断言目标不存在，也不能把部分搜索当作证明。\n");
                self.render_frontier(&mut output);
            }
        }
        truncate_chars(output.trim_end().to_string(), MAX_EXPLANATION_CHARS)
    }

    fn render_frontier(&self, output: &mut String) {
        if self.frontier.is_empty() {
            return;
        }
        output.push_str("  知识缺口（本次搜索实际失败叶子）：\n");
        for entry in &self.frontier {
            output.push_str(&format!("    - {}\n", entry));
        }
        if self.frontier_omitted > 0 {
            output.push_str(&format!(
                "    - ……另有 {} 条缺口未展开\n",
                self.frontier_omitted
            ));
        }
    }
}

struct StepBudget {
    bytes: usize,
}

impl StepBudget {
    fn new() -> Self {
        StepBudget { bytes: 0 }
    }

    fn push_forest(&mut self, root: &ProofViewNode, out: &mut Derivation) {
        let mut stack = vec![(root, 0usize)];
        while let Some((node, depth)) = stack.pop() {
            let label = if node.label.is_empty() {
                &node.canonical
            } else {
                &node.label
            };
            let cost = label.len() + node.canonical.len();
            if out.steps.len() >= MAX_DERIVATION_STEPS
                || depth > MAX_DERIVATION_DEPTH
                || self.bytes.saturating_add(cost) > MAX_DERIVATION_BYTES
            {
                out.truncated = true;
                out.omitted_steps += 1;
                continue;
            }
            self.bytes = self.bytes.saturating_add(cost);
            out.steps.push(DerivationStep {
                depth,
                claim: clip_chars(label, 240),
                canonical: clip_chars(&node.canonical, 240),
                source: match &node.source {
                    ViewProofSource::Fact => DerivationSource::Fact,
                    ViewProofSource::Rule(source) => DerivationSource::Rule {
                        source: clip_chars(source, 120),
                    },
                },
                confidence: node.confidence,
            });
            for child in node.children.iter().rev() {
                stack.push((child, depth + 1));
            }
        }
    }
}

fn clip_chars(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_string();
    }
    let mut out: String = text.chars().take(limit).collect();
    out.push('…');
    out
}

fn truncate_chars(text: String, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text;
    }
    let mut out: String = text.chars().take(limit.saturating_sub(1)).collect();
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resolution_engine::{FrontierReason, SearchStats};
    use crate::symbolic_reasoner::ProofSolutionView;

    fn node(label: &str, source: ViewProofSource, children: Vec<ProofViewNode>) -> ProofViewNode {
        ProofViewNode {
            label: label.to_string(),
            canonical: label.to_string(),
            source,
            confidence: 1.0,
            children,
        }
    }

    fn view(status: SearchStatus, solutions: Vec<ProofSolutionView>) -> ResolutionProofView {
        ResolutionProofView {
            phrase: "测试问句".to_string(),
            goal_display: "师傅(张三, ?X)".to_string(),
            facts: 7,
            clauses: 2,
            max_depth: 8,
            search_status: status,
            search_truncated: false,
            frontier: Vec::new(),
            stats: SearchStats::default(),
            solutions,
        }
    }

    #[test]
    fn flattens_resolution_proof_with_sources_and_bindings() {
        let fact = node("师傅(张三, 李四)", ViewProofSource::Fact, Vec::new());
        let proof = node(
            "师傅(李四, 王五)",
            ViewProofSource::Rule("链接预测规则#7".to_string()),
            vec![fact],
        );
        let solution = ProofSolutionView {
            bindings: vec![("答案".to_string(), "王五".to_string())],
            confidence: 0.83,
            proofs: vec![proof],
        };
        let derivation =
            Derivation::from_resolution_view(&view(SearchStatus::Proved, vec![solution]), Some("王五"))
                .expect("应能取到与回答匹配的证明");
        assert_eq!(derivation.status, DerivationStatus::Proved);
        assert_eq!(derivation.answer_value(), Some("王五"));
        assert_eq!(derivation.steps.len(), 2);
        assert_eq!(derivation.steps[0].depth, 0);
        assert_eq!(derivation.steps[1].depth, 1);
        assert!(matches!(
            derivation.steps[0].source,
            DerivationSource::Rule { .. }
        ));
        assert_eq!(derivation.steps[1].source, DerivationSource::Fact);

        let text = derivation.render_explanation();
        assert!(text.contains("王五"), "{}", text);
        assert!(text.contains("链接预测规则#7"), "{}", text);
        assert!(text.contains("李四"), "{}", text);
        assert!(text.contains("证明置信度"), "{}", text);
    }

    #[test]
    fn counts_alternative_proofs_without_expanding_them() {
        let proof = |value: &str| ProofSolutionView {
            bindings: vec![("答案".to_string(), value.to_string())],
            confidence: 1.0,
            proofs: vec![node(value, ViewProofSource::Fact, Vec::new())],
        };
        let view = view(
            SearchStatus::Proved,
            vec![proof("王五"), proof("赵六"), proof("孙七")],
        );
        let derivation =
            Derivation::from_resolution_view(&view, Some("王五")).expect("应选到匹配答案的解");
        assert_eq!(derivation.answer_value(), Some("王五"));
        assert_eq!(derivation.alternatives, 2);
        assert!(derivation.render_explanation().contains("另有 2 条独立证明"));
    }

    #[test]
    fn refuses_to_attach_a_proof_for_a_different_answer() {
        let solution = ProofSolutionView {
            bindings: vec![("答案".to_string(), "王五".to_string())],
            confidence: 1.0,
            proofs: vec![node("师傅(张三, 王五)", ViewProofSource::Fact, Vec::new())],
        };
        assert!(Derivation::from_resolution_view(
            &view(SearchStatus::Proved, vec![solution]),
            Some("赵六")
        )
        .is_none());
    }

    #[test]
    fn bounds_deep_proofs_instead_of_recursing_forever() {
        let mut proof = node("事实(底层)", ViewProofSource::Fact, Vec::new());
        for depth in 0..40 {
            proof = node(
                &format!("第{}层", depth),
                ViewProofSource::Rule("递归规则".to_string()),
                vec![proof],
            );
        }
        let solution = ProofSolutionView {
            bindings: vec![("答案".to_string(), "底层".to_string())],
            confidence: 1.0,
            proofs: vec![proof],
        };
        let derivation =
            Derivation::from_resolution_view(&view(SearchStatus::Proved, vec![solution]), None)
                .unwrap();
        assert!(derivation.steps.len() <= MAX_DERIVATION_STEPS);
        assert!(derivation.truncated);
        assert!(derivation.omitted_steps > 0);
        let text = derivation.render_explanation();
        assert!(text.contains("预算"), "{}", text);
    }

    #[test]
    fn renders_exhausted_search_with_frontier() {
        let mut view = view(SearchStatus::Exhausted, Vec::new());
        view.frontier = vec![FailureFrontier {
            goal: "师傅(张三, ?X)".to_string(),
            depth: 0,
            reason: FrontierReason::MissingFactOrRule,
        }];
        let derivation = Derivation::from_resolution_view(&view, Some("王五")).unwrap();
        assert_eq!(derivation.status, DerivationStatus::NoSolution);
        let text = derivation.render_explanation();
        assert!(text.contains("已穷尽"), "{}", text);
        assert!(text.contains("知识缺口"), "{}", text);
        assert!(text.contains("推不出不等于不存在"), "{}", text);
    }

    #[test]
    fn direct_facts_are_explainable_without_resolution() {
        let derivation = Derivation::from_direct_facts(
            "姚明的身高",
            vec![(
                "姚明".to_string(),
                "身高".to_string(),
                "226 厘米".to_string(),
            )],
        );
        assert!(derivation.is_proved());
        assert_eq!(derivation.answer_value(), Some("226 厘米"));
        let text = derivation.render_explanation();
        assert!(text.contains("226 厘米"), "{}", text);
        assert!(text.contains("事实"), "{}", text);
    }
}

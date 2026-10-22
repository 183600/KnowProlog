//! 配置兑现（Config Fulfillment）：声明有没有被运行时执行。
//!
//! 已有的两套校验各管一面：`config_contract` 证明配置内部自洽，
//! `shape_gate` / `law_book` 证明数据与声明不冲突。它们都回答
//! 「写下来的东西对不对」，没有回答「写下来的东西有没有被做到」。
//!
//! 本模块把 `[[predicate_profiles]]` 里的蕴涵声明编译成**证明义务**：
//!
//! - 对称 `P(a,b) ⇒ P(b,a)`；
//! - 互逆 `P(a,b) ≡ Q(b,a)`；
//! - 传递 `P(a,b) ∧ P(b,c) ⇒ P(a,c)`。
//!
//! 每条义务从图里取一条最小见证，生成两句问话：一句**对照探针**只验证
//! 直接方向可读，一句**义务探针**问被蕴涵的方向。两句都走真实的
//! `SymbolicReasoner` 回答管线（读句 → 规划 → 检索/归结），而不是直接调用
//! 某个内部函数。判决只有四种：
//!
//! | 判决 | 含义 |
//! |---|---|
//! | [`Verdict::Fulfilled`] | 答出了被蕴涵的事实 |
//! | [`Verdict::HonestUnknown`] | 没答出，但没有给出错误断言（能力缺口） |
//! | [`Verdict::Wrong`] | 给出了确定回答，但不含被蕴涵的事实 |
//! | [`Verdict::Unchecked`] | 对照探针失败 / 搜索截断，不能归因 |
//!
//! 「未兑现」不等于「说谎」：只有 `Wrong` 是矛盾，其余是能力缺口或未检验。
//! 图上没有见证时记 [`Verdict::Unwitnessed`]，明确表示「未检验」而不是「为真」。
//!
//! 设计纪律：
//! - 只读，不写知识库；
//! - 评估时关闭链接预测/关联规则，避免统计猜测替声明「圆谎」；
//! - 义务按稳定 id 排序，报告与摘要可逐字节复现；
//! - 默认最多 256 条义务，超出显式标记截断（截断 ≠ 通过）。

use std::collections::{BTreeMap, BTreeSet, HashMap};

use serde::{Deserialize, Serialize};

use crate::predicate_registry::PredicateRegistry;
use crate::prolog_engine::PrologFact;
use crate::resolution_engine::SearchStatus;
use crate::symbolic_reasoner::SymbolicReasoner;
use crate::ReasoningConfig;

/// 兑现报告的契约版本（格式变化时递增）。
pub const FULFILLMENT_VERSION: u32 = 1;

/// 默认义务上限：防止大图一次生成不可读的报告。
pub const DEFAULT_MAX_OBLIGATIONS: usize = 256;

/// 每条义务最多取几条见证（同一谓词事实很多时只审计前几条）。
const MAX_WITNESSES: usize = 8;

/// 未登记谓词报告里最多保留几个样例主语/事实。
const MAX_SAMPLES: usize = 5;

/// 报告里单条回答的最大字符数（避免异常输出无限增长）。
const ANSWER_CLIP: usize = 240;

/// 一条规范化后的图事实。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Fact {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

impl Fact {
    fn new(subject: impl Into<String>, predicate: impl Into<String>, object: impl Into<String>) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
        }
    }

    /// 便于报告阅读的见证行。
    fn render(&self) -> String {
        format!("{} -{}-> {}", self.subject, self.predicate, self.object)
    }
}

/// 义务种类。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObligationKind {
    Symmetric,
    Inverse,
    Transitive,
}

impl ObligationKind {
    pub fn label(self) -> &'static str {
        match self {
            ObligationKind::Symmetric => "对称",
            ObligationKind::Inverse => "互逆",
            ObligationKind::Transitive => "传递",
        }
    }

    pub fn tag(self) -> &'static str {
        match self {
            ObligationKind::Symmetric => "symmetric",
            ObligationKind::Inverse => "inverse",
            ObligationKind::Transitive => "transitive",
        }
    }
}

/// 一次探针执行的结果。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProbeResult {
    pub query: String,
    pub expected: Vec<String>,
    /// 从回答文本与归结解里抽出的确定取值（排序去重）。
    pub observed: Vec<String>,
    pub verdict: Verdict,
    /// 原始回答的有界摘要；未作答时为 `None`。
    pub answer: Option<String>,
    pub detail: String,
}

/// 义务/探针判决。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// 兑现：给出了被蕴涵的事实。
    Fulfilled,
    /// 诚实不知道：没有答出，但没有给出错误断言。
    HonestUnknown,
    /// 说错：给出了确定回答，却不含被蕴涵的事实。
    Wrong,
    /// 未检验：对照探针失败、搜索截断或无法形式化。
    Unchecked,
    /// 图上没有满足前提的见证，这条义务本轮未被检验。
    Unwitnessed,
}

impl Verdict {
    pub fn label(self) -> &'static str {
        match self {
            Verdict::Fulfilled => "兑现",
            Verdict::HonestUnknown => "诚实不知道",
            Verdict::Wrong => "说错",
            Verdict::Unchecked => "未检验",
            Verdict::Unwitnessed => "无见证",
        }
    }

    pub fn tag(self) -> &'static str {
        match self {
            Verdict::Fulfilled => "fulfilled",
            Verdict::HonestUnknown => "honest_unknown",
            Verdict::Wrong => "wrong",
            Verdict::Unchecked => "unchecked",
            Verdict::Unwitnessed => "unwitnessed",
        }
    }

    pub fn is_failure(self) -> bool {
        matches!(self, Verdict::Wrong)
    }

    pub fn is_gap(self) -> bool {
        matches!(
            self,
            Verdict::HonestUnknown | Verdict::Unchecked | Verdict::Unwitnessed
        )
    }
}

/// 一句可执行的最小问话。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Probe {
    pub query: String,
    pub expected: Vec<String>,
}

/// 一条证明义务：声明 + 图见证 → 最小问句。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Obligation {
    /// 稳定 id（不含随机数/哈希迭代序）。
    pub id: String,
    pub kind: ObligationKind,
    /// 被声明的规范谓词。
    pub predicate: String,
    /// 互逆伙伴（仅互逆义务）。
    pub related: Option<String>,
    pub witness: Vec<Fact>,
    pub expected: Vec<String>,
    /// 对照探针：只验证直接方向可读；失败时义务探针的结果不可归因。
    pub control: Option<Probe>,
    /// 义务探针：问被蕴涵的方向/闭包。
    pub probe: Option<Probe>,
    pub detail: String,
}

/// 功能谓词的数据冲突（同一主语多个取值）。这不是运行时谎言，而是
/// 图与声明已经矛盾；在兑现报告里单列，避免被探针结果掩盖。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FunctionalConflict {
    pub predicate: String,
    pub subject: String,
    pub values: Vec<String>,
}

/// 图里出现、但配置画像不认识的谓词。
///
/// 这些事实不会进入声明律，也不会被义务生成器看见；单列出来是为了让
/// 「系统不认识这个词」从一句提示变成可指认的条数、主语与见证。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnregisteredPredicate {
    pub predicate: String,
    pub occurrences: usize,
    pub subjects: Vec<String>,
    pub samples: Vec<Fact>,
}

/// 义务表（不跑运行时，可单独测试/打印）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Plan {
    pub obligations: Vec<Obligation>,
    pub functional_conflicts: Vec<FunctionalConflict>,
    pub unregistered: Vec<UnregisteredPredicate>,
    /// 规范化后的事实条数（含属性事实与关系事实）。
    pub graph_facts: usize,
    /// 义务数超过上限时被截断。
    pub truncated: bool,
}

/// 一条义务的兑现结果。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObligationReport {
    pub obligation: Obligation,
    pub verdict: Verdict,
    pub control: Option<ProbeResult>,
    pub probe: Option<ProbeResult>,
    pub detail: String,
}

/// 兑现报告。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FulfillmentReport {
    pub version: u32,
    pub graph_facts: usize,
    pub obligations: usize,
    pub fulfilled: usize,
    pub honest_unknown: usize,
    pub wrong: usize,
    pub unchecked: usize,
    pub unwitnessed: usize,
    pub functional_conflicts: Vec<FunctionalConflict>,
    pub unregistered: Vec<UnregisteredPredicate>,
    pub results: Vec<ObligationReport>,
    pub truncated: bool,
    pub digest: String,
}

impl FulfillmentReport {
    /// 没有「说错」，也没有功能谓词数据冲突。
    pub fn is_clean(&self) -> bool {
        self.wrong == 0 && self.functional_conflicts.is_empty()
    }

    /// 存在未兑现/未检验/未登记谓词；`--strict` 据此决定是否非零退出。
    pub fn has_gaps(&self) -> bool {
        self.honest_unknown > 0
            || self.unchecked > 0
            || self.unwitnessed > 0
            || !self.unregistered.is_empty()
    }

    pub fn render_text(&self, verbose: bool) -> String {
        let mut output = format!(
            "配置兑现（CF v{}）：图事实 {} 条，义务 {} 条；兑现 {} / 诚实不知道 {} / 说错 {} / 未检验 {} / 无见证 {}。\n",
            self.version,
            self.graph_facts,
            self.obligations,
            self.fulfilled,
            self.honest_unknown,
            self.wrong,
            self.unchecked,
            self.unwitnessed
        );
        if self.truncated {
            output.push_str("  注意：义务数超过上限，报告已截断；截断不等于通过。\n");
        }
        if self.obligations == 0 {
            output.push_str(
                "没有可生成的义务：在 config.toml 的 [[predicate_profiles]] 中声明 symmetric / inverse / transitive。\n",
            );
        }

        let mut shown = 0usize;
        for result in &self.results {
            if result.verdict == Verdict::Fulfilled && !verbose {
                continue;
            }
            shown += 1;
            let obligation = &result.obligation;
            output.push_str(&format!(
                "  [{}] {} {}  {}\n",
                result.verdict.label(),
                obligation.kind.label(),
                obligation.id,
                result.detail
            ));
            if !obligation.witness.is_empty() {
                output.push_str(&format!(
                    "      见证：{}\n",
                    obligation
                        .witness
                        .iter()
                        .map(Fact::render)
                        .collect::<Vec<_>>()
                        .join("；")
                ));
            }
            if let Some(control) = &result.control {
                output.push_str(&format!(
                    "      对照：{} → 期望 {}，观察 {}（{}）\n",
                    control.query,
                    render_values(&control.expected),
                    render_values(&control.observed),
                    control.verdict.label()
                ));
            }
            if let Some(probe) = &result.probe {
                output.push_str(&format!(
                    "      义务：{} → 期望 {}，观察 {}（{}）\n",
                    probe.query,
                    render_values(&probe.expected),
                    render_values(&probe.observed),
                    probe.verdict.label()
                ));
                if let Some(answer) = &probe.answer {
                    output.push_str(&format!("      回答：{}\n", answer));
                }
            }
        }
        if !verbose {
            if self.fulfilled > 0 {
                output.push_str(&format!(
                    "  ……另有 {} 条已兑现义务（省略明细；用 --verbose 查看全部）。\n",
                    self.fulfilled
                ));
            }
            if shown == 0 {
                output.push_str("  所有带见证的义务都已兑现。\n");
            }
        }

        if !self.functional_conflicts.is_empty() {
            output.push_str(&format!(
                "功能谓词数据冲突：{} 处（同一主语出现多个取值，声明与图已矛盾）。\n",
                self.functional_conflicts.len()
            ));
            for conflict in &self.functional_conflicts {
                output.push_str(&format!(
                    "  [冲突] {} 的 {} 有 {} 个取值：{}\n",
                    conflict.subject,
                    conflict.predicate,
                    conflict.values.len(),
                    conflict.values.join("、")
                ));
            }
        }

        if !self.unregistered.is_empty() {
            output.push_str(&format!(
                "未登记谓词：{} 个（这些事实不进入声明律，义务生成器看不见它们）。\n",
                self.unregistered.len()
            ));
            for entry in &self.unregistered {
                output.push_str(&format!(
                    "  [未登记] {}：{} 条；主语 {}；样例 {}\n",
                    entry.predicate,
                    entry.occurrences,
                    render_values(&entry.subjects),
                    entry
                        .samples
                        .iter()
                        .map(Fact::render)
                        .collect::<Vec<_>>()
                        .join("；")
                ));
            }
        }

        output.push_str(&format!("digest {}\n", self.digest));
        output
    }
}

/// 生成义务表（只读配置 + 图，不跑运行时）。
pub fn plan(
    registry: &PredicateRegistry,
    facts: &[PrologFact],
    max_obligations: usize,
) -> Plan {
    let graph = NormalizedGraph::from_facts(registry, facts);
    let mut obligations: BTreeMap<String, Obligation> = BTreeMap::new();
    let mut functional_conflicts: Vec<FunctionalConflict> = Vec::new();
    let max = max_obligations.max(1);

    for canonical in registry.canonical_names() {
        let Some(profile) = registry.profile(canonical) else {
            continue;
        };
        if !profile.declared {
            continue;
        }
        let pairs = graph.edges.get(canonical).cloned().unwrap_or_default();

        if profile.symmetric {
            if pairs.is_empty() {
                insert_unwitnessed(
                    &mut obligations,
                    ObligationKind::Symmetric,
                    canonical,
                    None,
                );
            } else {
                for (subject, object) in pairs.iter().take(MAX_WITNESSES) {
                    let id = format!("sym:{}:{}->{}", canonical, subject, object);
                    let obligation = Obligation {
                        id: id.clone(),
                        kind: ObligationKind::Symmetric,
                        predicate: canonical.to_string(),
                        related: None,
                        witness: vec![Fact::new(subject, canonical, object)],
                        expected: vec![subject.clone()],
                        control: Some(Probe {
                            query: format!("{}的{}是谁", subject, canonical),
                            expected: vec![object.clone()],
                        }),
                        probe: Some(Probe {
                            query: format!("{}的{}是谁", object, canonical),
                            expected: vec![subject.clone()],
                        }),
                        detail: format!(
                            "「{}」声明对称：{} ⇒ {}({}, {})，检验反向问句",
                            canonical,
                            Fact::new(subject, canonical, object).render(),
                            canonical,
                            object,
                            subject
                        ),
                    };
                    insert_obligation(&mut obligations, obligation);
                }
            }
        }

        if let Some(inverse) = profile.inverse.as_deref() {
            let inverse = inverse.trim();
            if !inverse.is_empty() && inverse != canonical {
                if pairs.is_empty() {
                    insert_unwitnessed(
                        &mut obligations,
                        ObligationKind::Inverse,
                        canonical,
                        Some(inverse),
                    );
                } else {
                    for (subject, object) in pairs.iter().take(MAX_WITNESSES) {
                        let id = format!(
                            "inv:{}|{}:{}->{}",
                            canonical, inverse, subject, object
                        );
                        let obligation = Obligation {
                            id: id.clone(),
                            kind: ObligationKind::Inverse,
                            predicate: canonical.to_string(),
                            related: Some(inverse.to_string()),
                            witness: vec![Fact::new(subject, canonical, object)],
                            expected: vec![subject.clone()],
                            control: Some(Probe {
                                query: format!("{}的{}是谁", subject, canonical),
                                expected: vec![object.clone()],
                            }),
                            probe: Some(Probe {
                                query: format!("{}的{}是谁", object, inverse),
                                expected: vec![subject.clone()],
                            }),
                            detail: format!(
                                "「{}」与「{}」声明互逆：{}(a, b) ≡ {}(b, a)，检验换向后的问题",
                                canonical, inverse, canonical, inverse
                            ),
                        };
                        insert_obligation(&mut obligations, obligation);
                    }
                }
            }
        }

        if profile.transitive {
            // 两跳见证：a -P-> b -P-> c，义务是 a 的 P 应包含 c。
            let mut successors: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
            for (subject, object) in &pairs {
                successors.entry(subject.as_str()).or_default().insert(object.as_str());
            }
            let mut closures: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
            for (subject, objects) in &successors {
                for object in objects {
                    let Some(next) = successors.get(*object) else {
                        continue;
                    };
                    for target in next {
                        if *subject != *target {
                            closures
                                .entry(*subject)
                                .or_default()
                                .insert(*target);
                        }
                    }
                }
            }
            if closures.is_empty() {
                insert_unwitnessed(
                    &mut obligations,
                    ObligationKind::Transitive,
                    canonical,
                    None,
                );
            } else {
                for (subject, expected) in closures.iter().take(MAX_WITNESSES) {
                    let expected: Vec<String> =
                        expected.iter().map(|value| (*value).to_string()).collect();
                    let id = format!("tra:{}:{}", canonical, subject);
                    let chain = format!(
                        "{}的{}的{}是谁",
                        subject, canonical, canonical
                    );
                    let obligation = Obligation {
                        id: id.clone(),
                        kind: ObligationKind::Transitive,
                        predicate: canonical.to_string(),
                        related: None,
                        witness: expected
                            .iter()
                            .map(|target| {
                                Fact::new(
                                    (*subject).to_string(),
                                    canonical.to_string(),
                                    target.clone(),
                                )
                            })
                            .collect(),
                        expected: expected.clone(),
                        control: Some(Probe {
                            query: chain,
                            expected: expected.clone(),
                        }),
                        probe: Some(Probe {
                            query: format!("{}的{}是谁", subject, canonical),
                            expected: expected.clone(),
                        }),
                        detail: format!(
                            "「{}」声明传递：两跳闭包中的 {} 应出现在 {} 的直接回答里",
                            canonical,
                            render_values(&expected),
                            subject
                        ),
                    };
                    insert_obligation(&mut obligations, obligation);
                }
            }
        }

        if profile.functional {
            let mut values: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
            for (subject, object) in &pairs {
                values.entry(subject.as_str()).or_default().insert(object.as_str());
            }
            for (subject, objects) in values {
                if objects.len() > 1 {
                    functional_conflicts.push(FunctionalConflict {
                        predicate: canonical.to_string(),
                        subject: subject.to_string(),
                        values: objects.iter().map(|value| (*value).to_string()).collect(),
                    });
                }
            }
        }
    }

    let mut obligations: Vec<Obligation> = obligations.into_values().collect();
    let truncated = obligations.len() > max;
    if truncated {
        obligations.truncate(max);
    }
    functional_conflicts.sort_by(|a, b| {
        a.predicate
            .cmp(&b.predicate)
            .then_with(|| a.subject.cmp(&b.subject))
    });

    Plan {
        obligations,
        functional_conflicts,
        unregistered: graph.unregistered.into_values().collect(),
        graph_facts: graph.fact_count,
        truncated,
    }
}

/// 评估配置兑现。使用默认推理配置，但关闭链接预测与关联规则：
/// 否则统计猜测可能替一条没人执行的声明「圆谎」，测到的就不是声明本身。
pub fn evaluate(
    registry: &PredicateRegistry,
    facts: &[PrologFact],
    max_obligations: usize,
) -> FulfillmentReport {
    evaluate_with_config(
        registry,
        facts,
        &ReasoningConfig::default(),
        max_obligations,
    )
}

/// 用给定推理配置评估；仍会关闭链接预测/关联规则，理由见 [`evaluate`]。
pub fn evaluate_with_config(
    registry: &PredicateRegistry,
    facts: &[PrologFact],
    reasoning: &ReasoningConfig,
    max_obligations: usize,
) -> FulfillmentReport {
    let plan = plan(registry, facts, max_obligations);
    let mut reasoning = reasoning.clone();
    reasoning.enable_link_prediction = false;
    reasoning.enable_association_rules = false;

    let mut reasoner = SymbolicReasoner::new(reasoning, HashMap::new());
    reasoner.observe_facts(facts);
    reasoner.align_predicate_pairs(&registry.alias_pairs());
    reasoner.configure_laws(registry);

    let mut results = Vec::with_capacity(plan.obligations.len());
    for obligation in plan.obligations {
        results.push(evaluate_obligation(&reasoner, obligation));
    }

    let mut fulfilled = 0usize;
    let mut honest_unknown = 0usize;
    let mut wrong = 0usize;
    let mut unchecked = 0usize;
    let mut unwitnessed = 0usize;
    for result in &results {
        match result.verdict {
            Verdict::Fulfilled => fulfilled += 1,
            Verdict::HonestUnknown => honest_unknown += 1,
            Verdict::Wrong => wrong += 1,
            Verdict::Unchecked => unchecked += 1,
            Verdict::Unwitnessed => unwitnessed += 1,
        }
    }

    let mut report = FulfillmentReport {
        version: FULFILLMENT_VERSION,
        graph_facts: plan.graph_facts,
        obligations: results.len(),
        fulfilled,
        honest_unknown,
        wrong,
        unchecked,
        unwitnessed,
        functional_conflicts: plan.functional_conflicts,
        unregistered: plan.unregistered,
        results,
        truncated: plan.truncated,
        digest: String::new(),
    };
    report.digest = digest(&report);
    report
}

fn evaluate_obligation(reasoner: &SymbolicReasoner, obligation: Obligation) -> ObligationReport {
    let Some(probe) = obligation.probe.clone() else {
        return ObligationReport {
            detail: obligation.detail.clone(),
            obligation,
            verdict: Verdict::Unwitnessed,
            control: None,
            probe: None,
        };
    };
    let control = obligation
        .control
        .as_ref()
        .map(|control| run_probe(reasoner, control));
    let probe_result = run_probe(reasoner, &probe);

    // 对照探针不通过时，义务探针的失败可能只是「谓词根本没被读懂」，
    // 不能把两类失败混成同一条「没兑现」。
    let verdict = match &control {
        Some(control) if control.verdict != Verdict::Fulfilled => Verdict::Unchecked,
        _ => probe_result.verdict,
    };
    let detail = match &control {
        Some(control) if control.verdict != Verdict::Fulfilled => format!(
            "对照探针未兑现（{}：期望 {}，观察 {}），义务探针结果不可归因",
            control.verdict.label(),
            render_values(&control.expected),
            render_values(&control.observed)
        ),
        _ => obligation.detail.clone(),
    };

    ObligationReport {
        obligation,
        verdict,
        control,
        probe: Some(probe_result),
        detail,
    }
}

fn run_probe(reasoner: &SymbolicReasoner, probe: &Probe) -> ProbeResult {
    let answer = reasoner.answer(&probe.query);
    let view = reasoner.resolution_proof_view(&probe.query);
    let mut observed: BTreeSet<String> = BTreeSet::new();
    let mut truncated = false;

    if let Some(view) = &view {
        truncated = view.search_truncated || view.search_status == SearchStatus::Incomplete;
        for solution in &view.solutions {
            if let Some(value) = answer_binding(&solution.bindings) {
                observed.insert(normalize_value(value));
            }
        }
    }
    if let Some(text) = &answer {
        observed.extend(extract_answer_values(text));
    }
    observed.retain(|value| !value.is_empty());

    let expected: Vec<String> = probe
        .expected
        .iter()
        .map(|value| normalize_value(value))
        .filter(|value| !value.is_empty())
        .collect();
    let matched = !expected.is_empty()
        && expected
            .iter()
            .all(|item| observed.contains(item));
    let unknown_answer = answer
        .as_deref()
        .map(is_honest_unknown_answer)
        .unwrap_or(false);
    let contains_expected = answer.as_deref().map_or(false, |text| {
        !contains_negation(text)
            && expected
                .iter()
                .all(|item| text.contains(item.as_str()))
    });

    let (verdict, detail) = if matched || contains_expected {
        (
            Verdict::Fulfilled,
            format!(
                "回答包含被蕴涵的取值：{}",
                render_values(&expected)
            ),
        )
    } else if truncated {
        (
            Verdict::Unchecked,
            "搜索被截断，不能据此判错（截断不等于无解）".to_string(),
        )
    } else if unknown_answer {
        (
            Verdict::HonestUnknown,
            "管线明确表示没有记录/不知道；这是能力缺口，不是错误断言".to_string(),
        )
    } else if observed.is_empty() {
        if answer.is_none() {
            (
                Verdict::HonestUnknown,
                "管线没有给出任何回答；能力缺口，不是错误断言".to_string(),
            )
        } else {
            (
                Verdict::Unchecked,
                "回答存在，但没有抽出可核对的确定取值".to_string(),
            )
        }
    } else {
        (
            Verdict::Wrong,
            format!(
                "回答给出了 {}，但不含被蕴涵的 {}",
                render_values(&observed.iter().cloned().collect::<Vec<_>>()),
                render_values(&expected)
            ),
        )
    };

    ProbeResult {
        query: probe.query.clone(),
        expected,
        observed: observed.into_iter().collect(),
        verdict,
        answer: answer.map(|text| clip(&text, ANSWER_CLIP)),
        detail,
    }
}

/// 从回答文本里抽出「… 是 X。」这类确定取值。
///
/// 只认带空格的「 是 」：这是本系统回答模板的固定分隔符，比裸「是」
/// 更不容易把疑问词或解释句误当取值。
fn extract_answer_values(answer: &str) -> BTreeSet<String> {
    let mut values = BTreeSet::new();
    for line in answer.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Some(index) = line.rfind(" 是 ") else {
            continue;
        };
        let mut rest = line[index + " 是 ".len()..].trim();
        rest = rest.trim_end_matches(['。', '；', ';', '，', ',']);
        rest = rest.trim_start_matches(['：', ':']);
        for value in rest.split(['、', '，', ',', '；', ';']) {
            let value = normalize_value(value);
            if !value.is_empty() && value.chars().count() <= 40 {
                values.insert(value);
            }
        }
    }
    values
}

fn normalize_value(value: &str) -> String {
    value
        .trim()
        .trim_matches(|ch: char| {
            matches!(
                ch,
                '「' | '」' | '“' | '”' | '"' | '\'' | '（' | '）' | '(' | ')' | ' '
            )
        })
        .trim_end_matches(['。', '；', ';'])
        .to_string()
}

fn is_honest_unknown_answer(answer: &str) -> bool {
    [
        "没有记录",
        "没有找到",
        "暂未找到",
        "未找到",
        "无法确定",
        "不知道",
        "查无",
    ]
    .iter()
    .any(|marker| answer.contains(marker))
}

fn contains_negation(answer: &str) -> bool {
    answer.contains("不是") || answer.contains("并非")
}

fn answer_binding(bindings: &[(String, String)]) -> Option<&str> {
    bindings
        .iter()
        .find(|(name, value)| name == "答案" && !value.is_empty())
        .map(|(_, value)| value.as_str())
        .or_else(|| {
            bindings
                .iter()
                .rev()
                .find(|(name, value)| !name.starts_with('V') && !value.is_empty())
                .map(|(_, value)| value.as_str())
        })
}

fn render_values(values: &[String]) -> String {
    if values.is_empty() {
        "（空）".to_string()
    } else {
        values.join("、")
    }
}

fn clip(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_string();
    }
    let mut clipped: String = text.chars().take(limit).collect();
    clipped.push('…');
    clipped
}

fn insert_obligation(obligations: &mut BTreeMap<String, Obligation>, obligation: Obligation) {
    obligations.entry(obligation.id.clone()).or_insert(obligation);
}

fn insert_unwitnessed(
    obligations: &mut BTreeMap<String, Obligation>,
    kind: ObligationKind,
    predicate: &str,
    related: Option<&str>,
) {
    let id = match related {
        Some(related) => format!("unwitnessed:{}:{}|{}", kind.tag(), predicate, related),
        None => format!("unwitnessed:{}:{}", kind.tag(), predicate),
    };
    let detail = match related {
        Some(related) => format!(
            "声明了「{}」与「{}」互逆，但图上没有该谓词的见证，本轮未检验",
            predicate, related
        ),
        None => format!(
            "声明了「{}」的{}性质，但图上没有该谓词的见证，本轮未检验",
            predicate,
            kind.label()
        ),
    };
    insert_obligation(
        obligations,
        Obligation {
            id,
            kind,
            predicate: predicate.to_string(),
            related: related.map(str::to_string),
            witness: Vec::new(),
            expected: Vec::new(),
            control: None,
            probe: None,
            detail,
        },
    );
}

/// 规范化后的图：谓词别名/互逆方向已经按注册表归一到槽位。
struct NormalizedGraph {
    edges: BTreeMap<String, BTreeSet<(String, String)>>,
    unregistered: BTreeMap<String, UnregisteredPredicate>,
    fact_count: usize,
}

impl NormalizedGraph {
    fn from_facts(registry: &PredicateRegistry, facts: &[PrologFact]) -> Self {
        let mut graph = Self {
            edges: BTreeMap::new(),
            unregistered: BTreeMap::new(),
            fact_count: 0,
        };
        for fact in facts {
            match fact {
                PrologFact::Relation { from, rel_type, to } => {
                    match registry.lookup(rel_type) {
                        Some((profile, inverted)) => {
                            let (subject, object) = if inverted {
                                (to.as_str(), from.as_str())
                            } else {
                                (from.as_str(), to.as_str())
                            };
                            graph.insert(profile.canonical.as_str(), subject, object);
                        }
                        None => graph.observe_unregistered(rel_type, from, to),
                    }
                }
                PrologFact::Entity {
                    name, properties, ..
                } => {
                    for (key, value) in properties {
                        // 属性键若不是声明谓词（如 name），不进入谓词义务；
                        // 只有已登记画像的属性才参与对称/互逆/功能检查。
                        if let Some((profile, inverted)) = registry.lookup(key) {
                            let (subject, object) = if inverted {
                                (value.as_str(), name.as_str())
                            } else {
                                (name.as_str(), value.as_str())
                            };
                            graph.insert(profile.canonical.as_str(), subject, object);
                        }
                    }
                }
            }
        }
        graph
    }

    fn insert(&mut self, predicate: &str, subject: &str, object: &str) {
        self.fact_count += 1;
        self.edges
            .entry(predicate.to_string())
            .or_default()
            .insert((subject.to_string(), object.to_string()));
    }

    fn observe_unregistered(&mut self, predicate: &str, subject: &str, object: &str) {
        self.fact_count += 1;
        let entry = self
            .unregistered
            .entry(predicate.to_string())
            .or_insert_with(|| UnregisteredPredicate {
                predicate: predicate.to_string(),
                occurrences: 0,
                subjects: Vec::new(),
                samples: Vec::new(),
            });
        entry.occurrences += 1;
        if entry.subjects.len() < MAX_SAMPLES && !entry.subjects.iter().any(|s| s == subject) {
            entry.subjects.push(subject.to_string());
        }
        if entry.samples.len() < MAX_SAMPLES {
            let sample = Fact::new(subject, predicate, object);
            if !entry.samples.contains(&sample) {
                entry.samples.push(sample);
            }
        }
    }
}

fn digest(report: &FulfillmentReport) -> String {
    let mut material = String::new();
    material.push_str(&format!("cf:{}\n", report.version));
    material.push_str(&format!("facts:{}\n", report.graph_facts));
    for result in &report.results {
        let obligation = &result.obligation;
        material.push_str(&format!(
            "{}|{}|{}|{}|{}\n",
            result.verdict.tag(),
            obligation.kind.tag(),
            obligation.id,
            obligation.expected.join("、"),
            obligation
                .witness
                .iter()
                .map(Fact::render)
                .collect::<Vec<_>>()
                .join("；")
        ));
        if let Some(probe) = &result.probe {
            material.push_str(&format!(
                "probe:{}|{}|{}\n",
                probe.query,
                probe.observed.join("、"),
                probe.verdict.tag()
            ));
        }
    }
    for conflict in &report.functional_conflicts {
        material.push_str(&format!(
            "conflict:{}|{}|{}\n",
            conflict.predicate,
            conflict.subject,
            conflict.values.join("、")
        ));
    }
    for entry in &report.unregistered {
        material.push_str(&format!(
            "unregistered:{}|{}\n",
            entry.predicate, entry.occurrences
        ));
    }
    fnv1a(material.as_bytes())
}

/// FNV-1a：只用于报告摘要，不用于安全用途。
fn fnv1a(bytes: &[u8]) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{:016x}", hash)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::predicate_registry::PredicateProfileConfig;

    fn profile(
        name: &str,
        symmetric: bool,
        inverse: Option<&str>,
        transitive: bool,
        functional: bool,
    ) -> PredicateProfileConfig {
        PredicateProfileConfig {
            name: name.to_string(),
            symmetric,
            inverse: inverse.map(str::to_string),
            transitive,
            functional,
            ..PredicateProfileConfig::default()
        }
    }

    fn registry() -> PredicateRegistry {
        let mut registry = PredicateRegistry::new();
        registry.declare(&profile("配偶", true, None, false, false));
        registry.declare(&profile("师傅", false, Some("徒弟"), true, false));
        registry.declare(&profile("年龄", false, None, false, true));
        registry
    }

    fn relation(from: &str, predicate: &str, to: &str) -> PrologFact {
        PrologFact::Relation {
            from: from.to_string(),
            rel_type: predicate.to_string(),
            to: to.to_string(),
        }
    }

    fn entity(name: &str, properties: &[(&str, &str)]) -> PrologFact {
        PrologFact::Entity {
            name: name.to_string(),
            entity_type: "人物".to_string(),
            properties: properties
                .iter()
                .map(|(key, value)| (key.to_string(), value.to_string()))
                .collect(),
        }
    }

    #[test]
    fn plan_builds_symmetric_inverse_and_transitive_obligations() {
        let facts = vec![
            relation("叶莉", "配偶", "姚明"),
            relation("张三", "师傅", "李四"),
            relation("李四", "师傅", "王五"),
        ];
        let plan = plan(&registry(), &facts, DEFAULT_MAX_OBLIGATIONS);
        let kinds: Vec<ObligationKind> = plan.obligations.iter().map(|o| o.kind).collect();
        assert!(kinds.contains(&ObligationKind::Symmetric));
        assert!(kinds.contains(&ObligationKind::Inverse));
        assert!(kinds.contains(&ObligationKind::Transitive));

        let symmetric = plan
            .obligations
            .iter()
            .find(|o| o.kind == ObligationKind::Symmetric)
            .expect("应有对称义务");
        assert_eq!(symmetric.probe.as_ref().unwrap().query, "姚明的配偶是谁");
        assert_eq!(symmetric.probe.as_ref().unwrap().expected, vec!["叶莉"]);
        assert_eq!(symmetric.control.as_ref().unwrap().query, "叶莉的配偶是谁");

        let inverse = plan
            .obligations
            .iter()
            .find(|o| o.kind == ObligationKind::Inverse)
            .expect("应有互逆义务");
        assert_eq!(inverse.probe.as_ref().unwrap().query, "李四的徒弟是谁");
        assert_eq!(inverse.probe.as_ref().unwrap().expected, vec!["张三"]);

        let transitive = plan
            .obligations
            .iter()
            .find(|o| o.kind == ObligationKind::Transitive)
            .expect("应有传递义务");
        assert_eq!(transitive.probe.as_ref().unwrap().query, "张三的师傅是谁");
        assert_eq!(transitive.probe.as_ref().unwrap().expected, vec!["王五"]);
        assert_eq!(
            transitive.control.as_ref().unwrap().query,
            "张三的师傅的师傅是谁"
        );
    }

    #[test]
    fn aliases_and_inverse_surfaces_are_normalized() {
        let mut registry = PredicateRegistry::new();
        registry.declare(&PredicateProfileConfig {
            name: "师傅".to_string(),
            aliases: vec!["老师".to_string()],
            inverse: Some("徒弟".to_string()),
            transitive: true,
            ..PredicateProfileConfig::default()
        });
        registry.declare(&PredicateProfileConfig {
            name: "徒弟".to_string(),
            aliases: vec!["学生".to_string()],
            inverse: Some("师傅".to_string()),
            transitive: true,
            ..PredicateProfileConfig::default()
        });
        let facts = vec![
            relation("张三", "老师", "李四"),
            relation("李四", "学生", "王五"),
        ];
        let plan = plan(&registry, &facts, DEFAULT_MAX_OBLIGATIONS);
        assert!(
            plan.obligations
                .iter()
                .any(|o| o.predicate == "师傅" && o.kind == ObligationKind::Inverse),
            "别名与互逆词面应归一为师傅槽位：{:?}",
            plan.obligations
        );
        assert!(plan.unregistered.is_empty());
    }

    #[test]
    fn symmetric_and_inverse_obligations_are_fulfilled_by_pipeline() {
        let facts = vec![
            relation("叶莉", "配偶", "姚明"),
            relation("张三", "师傅", "李四"),
        ];
        let report = evaluate(&registry(), &facts, DEFAULT_MAX_OBLIGATIONS);
        for result in &report.results {
            if matches!(
                result.obligation.kind,
                ObligationKind::Symmetric | ObligationKind::Inverse
            ) {
                assert_eq!(
                    result.verdict,
                    Verdict::Fulfilled,
                    "{}：{:?}",
                    result.obligation.id,
                    result
                );
            }
        }
        assert_eq!(report.wrong, 0);
        assert_eq!(report.functional_conflicts, Vec::new());
    }

    #[test]
    fn no_witness_is_unwitnessed_not_fulfilled() {
        let plan = plan(&registry(), &[], DEFAULT_MAX_OBLIGATIONS);
        assert!(plan.obligations.iter().all(|o| o.probe.is_none()));
        assert!(plan.obligations.iter().all(|o| o.expected.is_empty()));
        let report = evaluate(&registry(), &[], DEFAULT_MAX_OBLIGATIONS);
        assert_eq!(report.fulfilled, 0);
        assert!(report.unwitnessed >= 3, "对称/互逆/传递都应有未检验义务");
        assert!(report.has_gaps());
    }

    #[test]
    fn transitive_closure_gap_is_reported_as_wrong() {
        // 当前回答管线在有直接事实时只返回直接取值，不把传递闭包并入。
        // 这条测试固定的是「审计能看见这个缺口」，不是把这个缺口当成正确。
        let facts = vec![
            relation("张三", "师傅", "李四"),
            relation("李四", "师傅", "王五"),
        ];
        let report = evaluate(&registry(), &facts, DEFAULT_MAX_OBLIGATIONS);
        let transitive = report
            .results
            .iter()
            .find(|result| result.obligation.kind == ObligationKind::Transitive)
            .expect("应有传递义务");
        assert_eq!(
            transitive.control.as_ref().unwrap().verdict,
            Verdict::Fulfilled,
            "对照的两跳问句应可答：{:?}",
            transitive.control
        );
        assert_eq!(
            transitive.verdict,
            Verdict::Wrong,
            "直接回答只有李四时应判为未兑现：{:?}",
            transitive
        );
        assert!(report.wrong >= 1);
        assert!(!report.is_clean());
    }

    #[test]
    fn functional_multi_value_is_reported_as_conflict() {
        let facts = vec![
            entity("姚明", &[("年龄", "41")]),
            entity("姚明", &[("年龄", "42")]),
        ];
        let plan = plan(&registry(), &facts, DEFAULT_MAX_OBLIGATIONS);
        assert_eq!(plan.functional_conflicts.len(), 1);
        let conflict = &plan.functional_conflicts[0];
        assert_eq!(conflict.predicate, "年龄");
        assert_eq!(conflict.subject, "姚明");
        assert_eq!(conflict.values, vec!["41", "42"]);

        let report = evaluate(&registry(), &facts, DEFAULT_MAX_OBLIGATIONS);
        assert!(!report.is_clean());
    }

    #[test]
    fn unregistered_predicate_is_visible_with_witness() {
        let facts = vec![
            relation("张三", "暗恋", "李四"),
            relation("王五", "暗恋", "赵六"),
        ];
        let plan = plan(&registry(), &facts, DEFAULT_MAX_OBLIGATIONS);
        assert_eq!(plan.unregistered.len(), 1);
        let entry = &plan.unregistered[0];
        assert_eq!(entry.predicate, "暗恋");
        assert_eq!(entry.occurrences, 2);
        assert_eq!(entry.subjects, vec!["张三", "王五"]);
        assert_eq!(entry.samples.len(), 2);
    }

    #[test]
    fn report_is_deterministic_and_digest_is_stable() {
        let facts = vec![
            relation("叶莉", "配偶", "姚明"),
            relation("张三", "师傅", "李四"),
            relation("李四", "师傅", "王五"),
        ];
        let first = evaluate(&registry(), &facts, DEFAULT_MAX_OBLIGATIONS);
        let second = evaluate(&registry(), &facts, DEFAULT_MAX_OBLIGATIONS);
        assert_eq!(first.digest, second.digest);
        assert_eq!(first.results, second.results);
        assert_eq!(first, second);
    }

    #[test]
    fn plan_respects_obligation_budget_and_marks_truncation() {
        let facts = vec![
            relation("叶莉", "配偶", "姚明"),
            relation("张三", "师傅", "李四"),
            relation("李四", "师傅", "王五"),
        ];
        let plan = plan(&registry(), &facts, 1);
        assert_eq!(plan.obligations.len(), 1);
        assert!(plan.truncated);
    }

    #[test]
    fn default_config_declares_the_expected_traits() {
        let source = include_str!("../config.toml");
        let config: crate::SystemConfig =
            toml::from_str(source).expect("config.toml 应能解析");
        let mut registry = PredicateRegistry::new();
        for profile in &config.predicate_profiles {
            registry.declare(profile);
        }
        let mut symmetric = 0;
        let mut inverse = 0;
        let mut transitive = 0;
        let mut functional = 0;
        for name in registry.canonical_names() {
            let Some(profile) = registry.profile(name) else {
                continue;
            };
            symmetric += usize::from(profile.symmetric);
            inverse += usize::from(profile.inverse.is_some());
            transitive += usize::from(profile.transitive);
            functional += usize::from(profile.functional);
        }
        assert_eq!(symmetric, 1, "配偶");
        assert_eq!(inverse, 1, "师傅/徒弟归一到一个槽位");
        assert_eq!(transitive, 1, "师傅");
        assert_eq!(functional, 3, "年龄/居住于/出生于");
    }

    #[test]
    fn render_text_groups_non_fulfilled_results() {
        let facts = vec![relation("张三", "师傅", "李四"), relation("李四", "师傅", "王五")];
        let report = evaluate(&registry(), &facts, DEFAULT_MAX_OBLIGATIONS);
        let text = report.render_text(false);
        assert!(text.contains("配置兑现"));
        assert!(text.contains("说错"));
        assert!(text.contains("传递"));
        assert!(text.contains("当前回答管线") || text.contains("直接回答"));
    }
}

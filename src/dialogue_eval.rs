//! 对话 NLU 评测台（Dialogue NLU Eval）：用可回归的数字度量对话理解质量。
//!
//! 与 `embedding_eval` 的关系：后者度量图谱侧（链接预测 / 类比），本模块度量
//! 对话侧（意图判定 / 实体链接 / 槽位填充 / 澄清决策 / 接管决策 / 答案生成 /
//! 生成忠信度）。两者共享同一套棘轮哲学：确定性金标语料 → 确定性指标 →
//! 基线 JSON → 指标退化即失败。
//!
//! 指标口径（全部零新依赖、纯符号可解释）：
//! - 意图：12 类 [`MoveKind`] 的逐类 P/R/F1 + 宏平均 + 混淆对；
//! - 实体链接：每轮显式提及实体集合的 micro-P/R/F1；
//! - 槽位：属性槽的 micro-F1；
//! - 澄清决策：二分类 F1 + over-ask 率（无歧义却追问的比例，越低越好）；
//! - 接管决策：该接管 / 该让位的二分类 F1（让位正确性不被答案指标掩盖）；
//! - 答案：回答首段（回答生成段）的中文按字符 char-F1（SQuAD 式）+ exact match 率，
//!   追问建议等组合片段不计入答案指标，而由忠信度与对话符合性套件约束；
//! - 忠信度：内容型回答中「引号词条」可回溯到知识库、当前输入或对话历史的比例
//!   （precision 型）；纯寒暄的示例引文不参与审计。
//!
//! 两种知识模式：
//! - [`EvalKnowledge::Scripted`]：脚本化知识替身，隔离策略层行为；
//! - [`EvalKnowledge::Reasoner`]：真实 [`SymbolicReasoner`]（由金标事实构建），
//!   端到端度量产品行为。模式参与语料摘要，避免跨模式误比基线。

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::dialogue_manager::{DialogueKnowledge, DialogueManager, MoveKind, UtteranceFrame};
use crate::prolog_engine::PrologFact;
use crate::prompt_spec::{PromptMode, PromptStats};
use crate::symbolic_reasoner::SymbolicReasoner;
use crate::ReasoningConfig;

/// 评测使用的知识后端。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvalKnowledge {
    /// 脚本化知识替身（策略层隔离）。
    Scripted,
    /// 由金标事实构建的真实推理器（端到端）。
    Reasoner,
}

impl EvalKnowledge {
    pub fn label(&self) -> &'static str {
        match self {
            EvalKnowledge::Scripted => "scripted",
            EvalKnowledge::Reasoner => "reasoner",
        }
    }
}

/// 精确率 / 召回率 / F1 的计数载体。
///
/// 空分母约定：没有预测时 precision = 1.0，没有金标时 recall = 1.0，
/// 两者都空时 F1 = 1.0——「没有需要判定的样本」不构成错误。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Prf {
    pub tp: u32,
    pub fp: u32,
    pub fn_: u32,
}

impl Prf {
    pub fn precision(&self) -> f64 {
        ratio(self.tp, self.tp + self.fp)
    }

    pub fn recall(&self) -> f64 {
        ratio(self.tp, self.tp + self.fn_)
    }

    pub fn f1(&self) -> f64 {
        let (precision, recall) = (self.precision(), self.recall());
        if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        }
    }

    fn observe(&mut self, tp: u32, fp: u32, fn_: u32) {
        self.tp += tp;
        self.fp += fp;
        self.fn_ += fn_;
    }

    /// 布尔二分类观测。
    fn observe_bool(&mut self, gold: bool, predicted: bool) {
        match (gold, predicted) {
            (true, true) => self.observe(1, 0, 0),
            (false, true) => self.observe(0, 1, 0),
            (true, false) => self.observe(0, 0, 1),
            (false, false) => {}
        }
    }

    /// 集合观测（按集合差计 micro 计数）。
    fn observe_sets(&mut self, gold: &HashSet<String>, predicted: &HashSet<String>) {
        let tp = gold.intersection(predicted).count() as u32;
        let fp = predicted.difference(gold).count() as u32;
        let fn_ = gold.difference(predicted).count() as u32;
        self.observe(tp, fp, fn_);
    }
}

fn ratio(numerator: u32, denominator: u32) -> f64 {
    if denominator == 0 {
        1.0
    } else {
        numerator as f64 / denominator as f64
    }
}

/// 意图判定指标。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IntentMetrics {
    pub accuracy: f64,
    pub micro_f1: f64,
    /// 只对金标中出现过的类别求宏平均（支持数为 0 的类别不进平均）。
    pub macro_f1: f64,
    /// 类别标签 → 指标。
    pub per_class: BTreeMap<String, Prf>,
    /// 金标标签 → 预测标签 → 次数（只记非零项）。
    pub confusion: BTreeMap<String, BTreeMap<String, u32>>,
}

/// 答案生成指标。
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AnswerMetrics {
    /// 有参考回答的轮数。
    pub reference_turns: u32,
    /// 其中系统实际给出回答的轮数（未让位、非空文本）。
    pub answered: u32,
    /// 中文按字符的 F1 宏平均（未回答的轮按 0 计）。
    pub char_f1: f64,
    /// 归一化后逐字相等的比例。
    pub exact_match_rate: f64,
}

/// 生成忠信度：回答引用的词条必须可回溯。
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ProvenanceMetrics {
    /// 检查过的回答数（成功接管的轮）。
    pub replies: u32,
    /// 回答中「…」引号词条总数。
    pub quoted_tokens: u32,
    /// 其中可回溯到知识库或对话历史的词条数。
    pub traceable_tokens: u32,
    /// traceable / quoted（无词条时为 1.0）。
    pub precision: f64,
}

/// 提示接地指标：模板所需槽位是否全部拿到绑定（事前约束，而非事后审计）。
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GroundingMetrics {
    /// 尝试走模板渲染的策略次数。
    pub attempted: u32,
    /// required 槽位全部绑定，成功渲染。
    pub grounded: u32,
    /// required 槽位缺失，回退组合生成器。
    pub ungrounded: u32,
    /// 双跑模式下模板与组合生成器文本不一致的次数（应为 0）。
    pub parity_mismatches: u32,
    /// grounded / attempted（无尝试时为 1.0）。
    pub slot_rate: f64,
}

/// 一次完整评测的报告（可序列化，供基线棘轮与 CI 消费）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DialogueEvalReport {
    pub knowledge_mode: String,
    pub corpus_digest: String,
    pub scenarios: u32,
    pub turns: u32,
    pub intents: IntentMetrics,
    pub entity: Prf,
    pub slot: Prf,
    pub clarify: Prf,
    /// 无歧义却追问的比例（越低越好）。
    pub over_ask_rate: f64,
    pub takeover: Prf,
    pub answers: AnswerMetrics,
    pub provenance: ProvenanceMetrics,
    /// 模板接地闸门指标。
    #[serde(default)]
    pub grounding: GroundingMetrics,
}

impl DialogueEvalReport {
    /// 确定性文本报告。
    pub fn render_text(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "对话 NLU 评测（{} 模式）：{} 个场景 / {} 轮，语料摘要 {}。\n",
            self.knowledge_mode, self.scenarios, self.turns, self.corpus_digest
        ));
        out.push_str(&format!(
            "意图：macro-F1 {:.4}，micro-F1 {:.4}，准确率 {:.4}\n",
            self.intents.macro_f1, self.intents.micro_f1, self.intents.accuracy
        ));
        for (label, prf) in &self.intents.per_class {
            let support = prf.tp + prf.fn_;
            let predicted = prf.tp + prf.fp;
            if support == 0 && predicted == 0 {
                continue;
            }
            out.push_str(&format!(
                "  - {}: F1 {:.4}（P {:.4} / R {:.4}，支持 {} / 预测 {}）\n",
                label,
                prf.f1(),
                prf.precision(),
                prf.recall(),
                support,
                predicted
            ));
        }
        for (gold, predicted_map) in &self.intents.confusion {
            for (predicted, count) in predicted_map {
                if gold == predicted || *count == 0 {
                    continue;
                }
                out.push_str(&format!("  ! 混淆 {} → {} ×{}\n", gold, predicted, count));
            }
        }
        out.push_str(&format!(
            "实体链接：micro-F1 {:.4}（P {:.4} / R {:.4}）\n",
            self.entity.f1(),
            self.entity.precision(),
            self.entity.recall()
        ));
        out.push_str(&format!("槽位填充：micro-F1 {:.4}\n", self.slot.f1()));
        out.push_str(&format!(
            "澄清决策：F1 {:.4}，over-ask 率 {:.4}\n",
            self.clarify.f1(),
            self.over_ask_rate
        ));
        out.push_str(&format!("接管决策：F1 {:.4}\n", self.takeover.f1()));
        out.push_str(&format!(
            "答案生成：char-F1 {:.4}，EM {:.4}，覆盖 {}/{} 轮\n",
            self.answers.char_f1,
            self.answers.exact_match_rate,
            self.answers.answered,
            self.answers.reference_turns
        ));
        out.push_str(&format!(
            "生成忠信度：{}/{} 个引号词条可回溯（precision {:.4}）\n",
            self.provenance.traceable_tokens,
            self.provenance.quoted_tokens,
            self.provenance.precision
        ));
        out.push_str(&format!(
            "槽位接地：{}/{} 次模板渲染通过闸门（slot_rate {:.4}），回退 {} 次，双跑差异 {} 次\n",
            self.grounding.grounded,
            self.grounding.attempted,
            self.grounding.slot_rate,
            self.grounding.ungrounded,
            self.grounding.parity_mismatches
        ));
        if let Some((label, f1)) = self.weakest_intent_class() {
            out.push_str(&format!("最弱意图类别：{}（F1 {:.4}）\n", label, f1));
        }
        out
    }

    /// 金标支持数 > 0 的最低 F1 类别（供定点攻坚）。
    pub fn weakest_intent_class(&self) -> Option<(String, f64)> {
        self.intents
            .per_class
            .iter()
            .filter(|(_, prf)| prf.tp + prf.fn_ > 0)
            .map(|(label, prf)| (label.clone(), prf.f1()))
            .min_by(|left, right| left.1.partial_cmp(&right.1).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// 标量指标摊平（相对比较用；键稳定、可扩展）。
    pub fn flatten_metrics(&self) -> BTreeMap<String, f64> {
        let mut metrics = BTreeMap::new();
        metrics.insert("intent.macro_f1".to_string(), self.intents.macro_f1);
        metrics.insert("intent.micro_f1".to_string(), self.intents.micro_f1);
        metrics.insert("intent.accuracy".to_string(), self.intents.accuracy);
        for (label, prf) in &self.intents.per_class {
            metrics.insert(format!("intent.class.{}.f1", label), prf.f1());
        }
        metrics.insert("entity.micro_f1".to_string(), self.entity.f1());
        metrics.insert("entity.precision".to_string(), self.entity.precision());
        metrics.insert("entity.recall".to_string(), self.entity.recall());
        metrics.insert("slot.micro_f1".to_string(), self.slot.f1());
        metrics.insert("clarify.f1".to_string(), self.clarify.f1());
        metrics.insert("clarify.over_ask_rate".to_string(), self.over_ask_rate);
        metrics.insert("takeover.f1".to_string(), self.takeover.f1());
        metrics.insert("answer.char_f1".to_string(), self.answers.char_f1);
        metrics.insert("answer.em_rate".to_string(), self.answers.exact_match_rate);
        metrics.insert(
            "answer.answered_rate".to_string(),
            ratio(self.answers.answered, self.answers.reference_turns),
        );
        metrics.insert("provenance.precision".to_string(), self.provenance.precision);
        metrics.insert("grounding.slot_rate".to_string(), self.grounding.slot_rate);
        metrics
    }
}

/// 越大越差的指标（棘轮对它们反向判退化）。
pub const LOWER_IS_BETTER: [&str; 1] = ["clarify.over_ask_rate"];

/// 可写入磁盘的评测基线。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DialogueEvalBaseline {
    pub version: u32,
    pub corpus_digest: String,
    pub knowledge_mode: String,
    /// 标量指标摊平表。
    pub metrics: BTreeMap<String, f64>,
}

pub const DIALOGUE_BASELINE_VERSION: u32 = 2;

impl DialogueEvalBaseline {
    pub fn from_report(report: &DialogueEvalReport) -> Self {
        Self {
            version: DIALOGUE_BASELINE_VERSION,
            corpus_digest: report.corpus_digest.clone(),
            knowledge_mode: report.knowledge_mode.clone(),
            metrics: report.flatten_metrics(),
        }
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), std::io::Error> {
        let mut text = serde_json::to_string_pretty(self)?;
        text.push('\n');
        std::fs::write(path, text)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let text = std::fs::read_to_string(path)?;
        let baseline: DialogueEvalBaseline = serde_json::from_str(&text)?;
        Ok(baseline)
    }
}

/// 单项指标变化。
#[derive(Debug, Clone, PartialEq)]
pub struct DialogueMetricDelta {
    pub metric: String,
    pub baseline: f64,
    pub current: f64,
}

/// 棘轮判定。
#[derive(Debug, Clone, PartialEq)]
pub struct DialogueRatchetVerdict {
    pub passed: bool,
    pub digest_matches: bool,
    pub regressions: Vec<DialogueMetricDelta>,
    pub gains: Vec<DialogueMetricDelta>,
}

impl DialogueRatchetVerdict {
    pub fn render_text(&self) -> String {
        let mut out = String::new();
        if self.passed {
            out.push_str("棘轮检查通过：无指标低于基线。\n");
        } else {
            out.push_str("棘轮检查未通过：\n");
            if !self.digest_matches {
                out.push_str("  - 语料摘要或知识模式不匹配：金标语料已漂移，基线不可直接比较\n");
            }
            for regression in &self.regressions {
                let direction = if LOWER_IS_BETTER.contains(&regression.metric.as_str()) {
                    "（越低越好）"
                } else {
                    ""
                };
                out.push_str(&format!(
                    "  - {}: {:.6} → {:.6}{}\n",
                    regression.metric, regression.baseline, regression.current, direction
                ));
            }
        }
        if !self.gains.is_empty() {
            out.push_str(&format!("有 {} 项指标高于基线。\n", self.gains.len()));
        }
        out
    }
}

/// 与基线比较：语料摘要 / 知识模式不同，或任何指标退化超 ε，即不通过。
pub fn compare_baseline(
    baseline: &DialogueEvalBaseline,
    current: &DialogueEvalReport,
    epsilon: f64,
) -> DialogueRatchetVerdict {
    let digest_matches = baseline.corpus_digest == current.corpus_digest
        && baseline.knowledge_mode == current.knowledge_mode;
    let current_metrics = current.flatten_metrics();
    let mut regressions = Vec::new();
    let mut gains = Vec::new();
    for (metric, baseline_value) in &baseline.metrics {
        let current_value = current_metrics.get(metric).copied().unwrap_or(0.0);
        let lower_is_better = LOWER_IS_BETTER.contains(&metric.as_str());
        let (is_regression, is_gain) = if lower_is_better {
            (current_value > baseline_value + epsilon, current_value < baseline_value - epsilon)
        } else {
            (current_value < baseline_value - epsilon, current_value > baseline_value + epsilon)
        };
        if is_regression {
            regressions.push(DialogueMetricDelta {
                metric: metric.clone(),
                baseline: *baseline_value,
                current: current_value,
            });
        } else if is_gain {
            gains.push(DialogueMetricDelta {
                metric: metric.clone(),
                baseline: *baseline_value,
                current: current_value,
            });
        }
    }
    DialogueRatchetVerdict {
        passed: digest_matches && regressions.is_empty(),
        digest_matches,
        regressions,
        gains,
    }
}

// ---------------------------------------------------------------------------
// 金标语料
// ---------------------------------------------------------------------------

/// 一轮金标：输入 + 期望解读 + 期望行为。
#[derive(Debug, Clone)]
pub struct GoldTurn {
    pub input: &'static str,
    pub kind: MoveKind,
    /// 输入中显式提及（且应被链接到）的实体集合。
    pub entities: &'static [&'static str],
    /// 期望填充的属性槽（无槽为 None）。
    pub attribute: Option<&'static str>,
    /// 参考回答（char-F1 / EM 的对照；无参考则不参与答案指标）。
    pub reference: Option<&'static str>,
    /// 期望系统发起澄清（消歧）追问。
    pub clarify: bool,
    /// 期望系统本轮接管生成回答（false = 应让位给其它管线）。
    pub expect_takeover: bool,
}

/// 一个金标场景：小知识库 + 顺序多轮脚本。
#[derive(Debug, Clone)]
pub struct GoldScenario {
    pub name: &'static str,
    /// 实体 → (属性, 取值)。
    pub entities: &'static [(&'static str, &'static [(&'static str, &'static str)])],
    /// 实体 → 画像（身份问句的参考回答；也供脚本假想知识使用）。
    pub profiles: &'static [(&'static str, &'static str)],
    pub turns: &'static [GoldTurn],
}

/// 种子金标语料：覆盖 12 类 MoveKind、指代 / 省略继承、消歧追问、
/// 未知属性坦白、断言 / 是非问让位与寒暄收尾。
pub fn seed_corpus() -> &'static [GoldScenario] {
    static SEED: &[GoldScenario] = &[
        GoldScenario {
            name: "篮球家庭",
            entities: &[
                (
                    "姚明",
                    &[("身高", "226"), ("国籍", "中国"), ("出生地", "上海")],
                ),
                ("叶莉", &[("身高", "190"), ("国籍", "中国")]),
            ],
            profiles: &[
                ("姚明", "姚明是中国篮球运动员。"),
                ("叶莉", "叶莉是中国篮球运动员。"),
            ],
            turns: &[
                GoldTurn {
                    input: "你好",
                    kind: MoveKind::Greet,
                    entities: &[],
                    attribute: None,
                    reference: None,
                    clarify: false,
                    expect_takeover: true,
                },
                GoldTurn {
                    input: "姚明是谁",
                    kind: MoveKind::AskIdentity,
                    entities: &["姚明"],
                    attribute: None,
                    reference: Some("姚明是中国篮球运动员。"),
                    clarify: false,
                    expect_takeover: true,
                },
                GoldTurn {
                    input: "他的身高呢",
                    kind: MoveKind::AskAttribute,
                    entities: &[],
                    attribute: Some("身高"),
                    reference: Some("姚明的身高是226。"),
                    clarify: false,
                    expect_takeover: true,
                },
                GoldTurn {
                    input: "身高呢",
                    kind: MoveKind::AskAttribute,
                    entities: &[],
                    attribute: Some("身高"),
                    reference: Some("姚明的身高是226。"),
                    clarify: false,
                    expect_takeover: true,
                },
                GoldTurn {
                    input: "叶莉是谁",
                    kind: MoveKind::AskIdentity,
                    entities: &["叶莉"],
                    attribute: None,
                    reference: Some("叶莉是中国篮球运动员。"),
                    clarify: false,
                    expect_takeover: true,
                },
                GoldTurn {
                    input: "身高呢",
                    kind: MoveKind::AskAttribute,
                    entities: &[],
                    attribute: Some("身高"),
                    reference: None,
                    clarify: true,
                    expect_takeover: true,
                },
                GoldTurn {
                    input: "那叶莉呢",
                    kind: MoveKind::AskIdentity,
                    entities: &["叶莉"],
                    attribute: None,
                    reference: Some("叶莉是中国篮球运动员。"),
                    clarify: false,
                    expect_takeover: true,
                },
                GoldTurn {
                    input: "还有呢",
                    kind: MoveKind::ContinueRequest,
                    entities: &[],
                    attribute: None,
                    reference: None,
                    clarify: false,
                    expect_takeover: true,
                },
                GoldTurn {
                    input: "我们聊了什么？",
                    kind: MoveKind::MetaRecap,
                    entities: &[],
                    attribute: None,
                    reference: None,
                    clarify: false,
                    expect_takeover: true,
                },
                GoldTurn {
                    input: "你怎么知道的？",
                    kind: MoveKind::MetaWhy,
                    entities: &[],
                    attribute: None,
                    reference: None,
                    clarify: false,
                    expect_takeover: true,
                },
                GoldTurn {
                    input: "详细一点",
                    kind: MoveKind::ElaborateRequest,
                    entities: &[],
                    attribute: None,
                    reference: None,
                    clarify: false,
                    expect_takeover: true,
                },
                GoldTurn {
                    input: "谢谢",
                    kind: MoveKind::Thanks,
                    entities: &[],
                    attribute: None,
                    reference: None,
                    clarify: false,
                    expect_takeover: true,
                },
                GoldTurn {
                    input: "再见",
                    kind: MoveKind::Farewell,
                    entities: &[],
                    attribute: None,
                    reference: None,
                    clarify: false,
                    expect_takeover: true,
                },
            ],
        },
        GoldScenario {
            name: "未知与坦白",
            entities: &[("诸葛亮", &[("出生地", "琅琊")])],
            profiles: &[("诸葛亮", "诸葛亮是三国时期蜀汉丞相。")],
            turns: &[
                GoldTurn {
                    input: "诸葛亮是谁",
                    kind: MoveKind::AskIdentity,
                    entities: &["诸葛亮"],
                    attribute: None,
                    reference: Some("诸葛亮是三国时期蜀汉丞相。"),
                    clarify: false,
                    expect_takeover: true,
                },
                GoldTurn {
                    input: "他的身高是多少",
                    kind: MoveKind::AskAttribute,
                    entities: &[],
                    attribute: Some("身高"),
                    reference: None,
                    clarify: false,
                    expect_takeover: true,
                },
                GoldTurn {
                    input: "他的出生地呢",
                    kind: MoveKind::AskAttribute,
                    entities: &[],
                    attribute: Some("出生地"),
                    reference: Some("诸葛亮的出生地是琅琊。"),
                    clarify: false,
                    expect_takeover: true,
                },
                GoldTurn {
                    input: "谢谢",
                    kind: MoveKind::Thanks,
                    entities: &[],
                    attribute: None,
                    reference: None,
                    clarify: false,
                    expect_takeover: true,
                },
            ],
        },
        GoldScenario {
            name: "断言与让位",
            entities: &[("姚明", &[("身高", "226")])],
            profiles: &[("姚明", "姚明是中国篮球运动员。")],
            turns: &[
                GoldTurn {
                    input: "姚明是篮球运动员",
                    kind: MoveKind::Assert,
                    entities: &["姚明"],
                    attribute: None,
                    reference: None,
                    clarify: false,
                    expect_takeover: false,
                },
                GoldTurn {
                    input: "姚明是篮球运动员吗",
                    kind: MoveKind::AskBinary,
                    entities: &["姚明"],
                    attribute: None,
                    reference: None,
                    clarify: false,
                    expect_takeover: false,
                },
                GoldTurn {
                    input: "今天天气怎么样",
                    kind: MoveKind::Other,
                    entities: &[],
                    attribute: None,
                    reference: None,
                    clarify: false,
                    expect_takeover: false,
                },
                GoldTurn {
                    input: "你好",
                    kind: MoveKind::Greet,
                    entities: &[],
                    attribute: None,
                    reference: None,
                    clarify: false,
                    expect_takeover: true,
                },
            ],
        },
    ];
    SEED
}

// ---------------------------------------------------------------------------
// 脚本化知识替身
// ---------------------------------------------------------------------------

/// 由金标场景构建的确定性知识替身（子串提及检测 + 结构化问答）。
#[derive(Debug, Default, Clone)]
pub struct GoldKnowledge {
    entities: Vec<String>,
    attributes: BTreeMap<String, Vec<String>>,
    values: BTreeMap<(String, String), Vec<String>>,
    profiles: BTreeMap<String, String>,
}

impl GoldKnowledge {
    pub fn from_scenario(scenario: &GoldScenario) -> Self {
        let mut knowledge = GoldKnowledge::default();
        for (entity, attributes) in scenario.entities {
            knowledge.entities.push((*entity).to_string());
            knowledge.attributes.insert(
                (*entity).to_string(),
                attributes.iter().map(|(key, _)| (*key).to_string()).collect(),
            );
            for (attribute, value) in *attributes {
                knowledge
                    .values
                    .entry(((*entity).to_string(), (*attribute).to_string()))
                    .or_default()
                    .push((*value).to_string());
            }
        }
        knowledge
            .entities
            .sort_by(|left, right| right.chars().count().cmp(&left.chars().count()).then(left.cmp(right)));
        for (entity, profile) in scenario.profiles {
            knowledge
                .profiles
                .insert((*entity).to_string(), (*profile).to_string());
        }
        knowledge
    }

    /// 引号词条是否可回溯到知识库（实体 / 属性 / 取值 / 画像内容）。
    pub fn traceable(&self, token: &str) -> bool {
        self.entities.iter().any(|entity| entity == token)
            || self
                .attributes
                .values()
                .flatten()
                .any(|attribute| attribute == token)
            || self
                .values
                .iter()
                .any(|((entity, attribute), values)| {
                    entity == token
                        || attribute == token
                        || values.iter().any(|value| value == token)
                })
            || self
                .profiles
                .iter()
                .any(|(entity, profile)| entity == token || profile.contains(token))
    }
}

impl DialogueKnowledge for GoldKnowledge {
    fn entities_in(&self, text: &str) -> Vec<String> {
        self.entities
            .iter()
            .filter(|entity| text.contains(entity.as_str()))
            .cloned()
            .collect()
    }

    fn known_entity(&self, entity: &str) -> bool {
        self.entities.iter().any(|known| known == entity)
    }

    fn answer(&self, question: &str) -> Option<String> {
        let question = question
            .trim()
            .trim_end_matches(['？', '?', '。'])
            .trim();
        for (entity, profile) in &self.profiles {
            if question == format!("{}是谁", entity) || question == format!("{}是什么", entity) {
                return Some(profile.clone());
            }
        }
        let (entity, attribute) = parse_attribute_question(question)?;
        let values = self.attribute_values(&entity, &attribute);
        if values.is_empty() {
            None
        } else {
            Some(format!(
                "「{}」的「{}」是{}。",
                entity,
                attribute,
                values.join("、")
            ))
        }
    }

    fn known_attributes(&self, entity: &str) -> Vec<String> {
        self.attributes.get(entity).cloned().unwrap_or_default()
    }

    fn attribute_values(&self, entity: &str, attribute: &str) -> Vec<String> {
        self.values
            .get(&(entity.to_string(), attribute.to_string()))
            .cloned()
            .unwrap_or_default()
    }

    fn describe(&self, entity: &str) -> Option<String> {
        self.profiles.get(entity).cloned()
    }

    fn traceable_token(&self, token: &str) -> bool {
        self.traceable(token)
    }
}

/// 解析「X的A是什么」形态的规范化问句。
fn parse_attribute_question(question: &str) -> Option<(String, String)> {
    let (head, tail) = question.split_once('的')?;
    let attribute = tail.strip_suffix("是什么").unwrap_or(tail).trim();
    if head.is_empty() || attribute.is_empty() {
        return None;
    }
    Some((head.trim().to_string(), attribute.to_string()))
}

/// 由金标场景构建真实推理器（端到端评测用）。
pub fn reasoner_for_scenario(
    scenario: &GoldScenario,
    config: ReasoningConfig,
    synonyms: HashMap<String, String>,
) -> SymbolicReasoner {
    let facts: Vec<PrologFact> = scenario
        .entities
        .iter()
        .map(|(name, attributes)| {
            let mut properties = HashMap::new();
            for (attribute, value) in *attributes {
                properties.insert((*attribute).to_string(), (*value).to_string());
            }
            PrologFact::Entity {
                name: (*name).to_string(),
                entity_type: "人物".to_string(),
                properties,
            }
        })
        .collect();
    let mut reasoner = SymbolicReasoner::new(config, synonyms);
    reasoner.observe_facts(&facts);
    reasoner
}

// ---------------------------------------------------------------------------
// 评测执行
// ---------------------------------------------------------------------------

/// 逐场景、逐轮执行金标脚本，输出确定性报告。
///
/// 模板运行模式为 [`PromptMode::On`]（接地失败自动回退组合生成器，
/// 因此答案文本保持一致），同时收集 [`GroundingMetrics`]。
pub fn evaluate_scenarios(scenarios: &[GoldScenario]) -> DialogueEvalReport {
    evaluate_scenarios_with(scenarios, |scenario| {
        Box::new(GoldKnowledge::from_scenario(scenario))
    })
}

/// 与 [`evaluate_scenarios`] 相同，但知识后端由调用方提供
/// （例如用真实 `SymbolicReasoner` 跑端到端评测）。
pub fn evaluate_scenarios_with<F>(
    scenarios: &[GoldScenario],
    mut build_knowledge: F,
) -> DialogueEvalReport
where
    F: FnMut(&GoldScenario) -> Box<dyn DialogueKnowledge>,
{
    let mut acc = Acc::default();
    for scenario in scenarios {
        acc.scenarios += 1;
        let knowledge = build_knowledge(scenario);
        let mut manager = DialogueManager::new();
        manager.set_prompt_mode(PromptMode::On);
        let mut history: Vec<String> = Vec::new();
        for turn in scenario.turns {
            acc.turns += 1;
            let frame = manager.analyze(turn.input, &*knowledge);
            acc.observe_intent(turn, &frame);
            let predicted_entities: HashSet<String> =
                knowledge.entities_in(turn.input).into_iter().collect();
            acc.entity.observe_sets(&set_of(turn.entities.iter().copied()), &predicted_entities);
            let predicted_slot: HashSet<String> = frame.attribute.clone().into_iter().collect();
            acc.slot.observe_sets(
                &set_of(turn.attribute.as_ref().into_iter().copied()),
                &predicted_slot,
            );

            // 写前日志：当前输入先进入历史，引号词条可回溯到本轮问句
            history.push(turn.input.to_string());
            let reply = manager
                .plan_pure_dialogue(turn.input, &*knowledge)
                .or_else(|| manager.plan_reply(turn.input, &*knowledge));
            acc.takeover.observe_bool(turn.expect_takeover, reply.is_some());
            let predicted_clarify = reply
                .as_ref()
                .map(|reply| {
                    reply
                        .plan
                        .iter()
                        .any(|plan| plan == "clarify_entity" || plan == "clarify_entity_no_focus")
                })
                .unwrap_or(false);
            acc.observe_clarify(turn.clarify, predicted_clarify);
            acc.observe_answer(turn, reply.as_ref().map(|reply| reply.text.as_str()));
            if let Some(reply) = &reply {
                acc.observe_provenance(&reply.text, reply.plan.first(), &*knowledge, &history);
            }
            if let Some(reply) = &reply {
                history.push(reply.text.clone());
            }
        }
        acc.prompt_stats.absorb(manager.prompt_stats());
    }
    let digest = corpus_digest(scenarios, EvalKnowledge::Scripted);
    acc.finalize(EvalKnowledge::Scripted, digest)
}

/// 跑种子语料：`Scripted` 用脚本知识替身，`Reasoner` 用真实推理器。
pub fn evaluate_seed(mode: EvalKnowledge) -> DialogueEvalReport {
    evaluate_seed_with(mode, PromptMode::On)
}

/// 与 [`evaluate_seed`] 相同，但可指定模板运行模式
/// （`--prompt-mode off|parity|on`，供提示模板对比使用）。
pub fn evaluate_seed_with(mode: EvalKnowledge, prompt_mode: PromptMode) -> DialogueEvalReport {
    match mode {
        EvalKnowledge::Scripted => {
            let mut report = evaluate_scenarios(seed_corpus());
            if prompt_mode != PromptMode::On {
                report = reevaluate_scripted(prompt_mode);
            }
            report
        }
        EvalKnowledge::Reasoner => {
            let config = ReasoningConfig {
                enable_link_prediction: false,
                ..ReasoningConfig::default()
            };
            let mut acc = Acc::default();
            for scenario in seed_corpus() {
                acc.scenarios += 1;
                let knowledge = reasoner_for_scenario(scenario, config.clone(), HashMap::new());
                let mut manager = DialogueManager::new();
                manager.set_prompt_mode(prompt_mode);
                let mut history: Vec<String> = Vec::new();
                for turn in scenario.turns {
                    acc.turns += 1;
                    let frame = manager.analyze(turn.input, &knowledge);
                    acc.observe_intent(turn, &frame);
                    let predicted_entities: HashSet<String> =
                        knowledge.entities_in(turn.input).into_iter().collect();
                    acc.entity.observe_sets(&set_of(turn.entities.iter().copied()), &predicted_entities);
                    let predicted_slot: HashSet<String> =
                        frame.attribute.clone().into_iter().collect();
                    acc.slot.observe_sets(
                        &set_of(turn.attribute.as_ref().into_iter().copied()),
                        &predicted_slot,
                    );
                    history.push(turn.input.to_string());
                    let reply = manager
                        .plan_pure_dialogue(turn.input, &knowledge)
                        .or_else(|| manager.plan_reply(turn.input, &knowledge));
                    acc.takeover.observe_bool(turn.expect_takeover, reply.is_some());
                    let predicted_clarify = reply
                        .as_ref()
                        .map(|reply| {
                            reply.plan.iter().any(|plan| {
                                plan == "clarify_entity" || plan == "clarify_entity_no_focus"
                            })
                        })
                        .unwrap_or(false);
                    acc.observe_clarify(turn.clarify, predicted_clarify);
                    acc.observe_answer(turn, reply.as_ref().map(|reply| reply.text.as_str()));
                    if let Some(reply) = &reply {
                        acc.observe_provenance(&reply.text, reply.plan.first(), &knowledge, &history);
                    }
                    if let Some(reply) = &reply {
                        history.push(reply.text.clone());
                    }
                }
                acc.prompt_stats.absorb(manager.prompt_stats());
            }
            let digest = corpus_digest(seed_corpus(), mode);
            acc.finalize(mode, digest)
        }
    }
}

/// Scripted 模式下按指定模板模式重跑种子语料（与 reasoner 模式同构）。
fn reevaluate_scripted(prompt_mode: PromptMode) -> DialogueEvalReport {
    let mut acc = Acc::default();
    for scenario in seed_corpus() {
        acc.scenarios += 1;
        let knowledge = GoldKnowledge::from_scenario(scenario);
        let mut manager = DialogueManager::new();
        manager.set_prompt_mode(prompt_mode);
        let mut history: Vec<String> = Vec::new();
        for turn in scenario.turns {
            acc.turns += 1;
            let frame = manager.analyze(turn.input, &knowledge);
            acc.observe_intent(turn, &frame);
            let predicted_entities: HashSet<String> =
                knowledge.entities_in(turn.input).into_iter().collect();
            acc.entity.observe_sets(&set_of(turn.entities.iter().copied()), &predicted_entities);
            let predicted_slot: HashSet<String> = frame.attribute.clone().into_iter().collect();
            acc.slot.observe_sets(
                &set_of(turn.attribute.as_ref().into_iter().copied()),
                &predicted_slot,
            );
            history.push(turn.input.to_string());
            let reply = manager
                .plan_pure_dialogue(turn.input, &knowledge)
                .or_else(|| manager.plan_reply(turn.input, &knowledge));
            acc.takeover.observe_bool(turn.expect_takeover, reply.is_some());
            let predicted_clarify = reply
                .as_ref()
                .map(|reply| {
                    reply.plan.iter().any(|plan| {
                        plan == "clarify_entity" || plan == "clarify_entity_no_focus"
                    })
                })
                .unwrap_or(false);
            acc.observe_clarify(turn.clarify, predicted_clarify);
            acc.observe_answer(turn, reply.as_ref().map(|reply| reply.text.as_str()));
            if let Some(reply) = &reply {
                acc.observe_provenance(&reply.text, reply.plan.first(), &knowledge, &history);
            }
            if let Some(reply) = &reply {
                history.push(reply.text.clone());
            }
        }
        acc.prompt_stats.absorb(manager.prompt_stats());
    }
    let digest = corpus_digest(seed_corpus(), EvalKnowledge::Scripted);
    acc.finalize(EvalKnowledge::Scripted, digest)
}

fn set_of<'a, I: IntoIterator<Item = &'a str>>(items: I) -> HashSet<String> {
    items.into_iter().map(|item| item.to_string()).collect()
}

#[derive(Default)]
struct Acc {
    scenarios: u32,
    turns: u32,
    intent_correct: u32,
    gold_counts: BTreeMap<String, u32>,
    predicted_counts: BTreeMap<String, u32>,
    confusion: BTreeMap<String, BTreeMap<String, u32>>,
    entity: Prf,
    slot: Prf,
    clarify: Prf,
    over_ask_fp: u32,
    over_ask_tn: u32,
    takeover: Prf,
    answer_reference_turns: u32,
    answer_answered: u32,
    answer_char_f1_sum: f64,
    answer_exact: u32,
    provenance_replies: u32,
    provenance_quoted: u32,
    provenance_traceable: u32,
    prompt_stats: PromptStats,
}

impl Acc {
    fn observe_intent(&mut self, turn: &GoldTurn, frame: &UtteranceFrame) {
        let gold = turn.kind.label().to_string();
        let predicted = frame.kind.label().to_string();
        *self.gold_counts.entry(gold.clone()).or_insert(0) += 1;
        *self.predicted_counts.entry(predicted.clone()).or_insert(0) += 1;
        *self
            .confusion
            .entry(gold.clone())
            .or_default()
            .entry(predicted.clone())
            .or_insert(0) += 1;
        if gold == predicted {
            self.intent_correct += 1;
        }
    }

    fn observe_answer(&mut self, turn: &GoldTurn, reply: Option<&str>) {
        let Some(reference) = turn.reference else {
            return;
        };
        self.answer_reference_turns += 1;
        let predicted = answer_segment(reply.unwrap_or(""));
        if !predicted.is_empty() {
            self.answer_answered += 1;
        }
        let f1 = char_f1(predicted, reference);
        self.answer_char_f1_sum += f1;
        if normalize_answer(predicted) == normalize_answer(reference) {
            self.answer_exact += 1;
        }
    }

    /// 澄清决策：同时维护二分类指标与 over-ask 计数（FP / TN）。
    fn observe_clarify(&mut self, gold: bool, predicted: bool) {
        self.clarify.observe_bool(gold, predicted);
        match (gold, predicted) {
            (false, true) => self.over_ask_fp += 1,
            (false, false) => self.over_ask_tn += 1,
            _ => {}
        }
    }

    /// 只审计「内容型」回答：纯寒暄（greet/thanks/farewell）里的示例引文
    /// 是教学话术而非事实声明，不参与忠信度统计。
    fn observe_provenance(
        &mut self,
        text: &str,
        leading_plan: Option<&String>,
        knowledge: &dyn DialogueKnowledge,
        history: &[String],
    ) {
        if matches!(leading_plan.map(String::as_str), Some("greet" | "thanks" | "farewell")) {
            return;
        }
        self.provenance_replies += 1;
        for token in quoted_tokens(text) {
            self.provenance_quoted += 1;
            let from_history = history.iter().any(|line| line.contains(token.as_str()));
            if from_history || knowledge.traceable_token(&token) {
                self.provenance_traceable += 1;
            }
        }
    }

    fn finalize(mut self, mode: EvalKnowledge, digest: String) -> DialogueEvalReport {
        let mut per_class = BTreeMap::new();
        for (label, gold) in &self.gold_counts {
            let predicted = self.predicted_counts.get(label).copied().unwrap_or(0);
            let tp = self
                .confusion
                .get(label)
                .and_then(|row| row.get(label))
                .copied()
                .unwrap_or(0);
            per_class.insert(
                label.clone(),
                Prf {
                    tp,
                    fp: predicted.saturating_sub(tp),
                    fn_: gold.saturating_sub(tp),
                },
            );
        }
        for (label, predicted) in &self.predicted_counts {
            per_class.entry(label.clone()).or_insert(Prf {
                tp: 0,
                fp: *predicted,
                fn_: 0,
            });
        }
        let gold_total: u32 = self.gold_counts.values().sum();
        let predicted_total: u32 = self.predicted_counts.values().sum();
        let tp_micro: u32 = per_class.values().map(|prf| prf.tp).sum();
        let micro_f1 = Prf {
            tp: tp_micro,
            fp: predicted_total.saturating_sub(tp_micro),
            fn_: gold_total.saturating_sub(tp_micro),
        }
        .f1();
        let supported: Vec<&Prf> = self
            .gold_counts
            .keys()
            .filter_map(|label| per_class.get(label))
            .collect();
        let macro_f1 = if supported.is_empty() {
            1.0
        } else {
            supported.iter().map(|prf| prf.f1()).sum::<f64>() / supported.len() as f64
        };
        let over_ask_denominator = self.over_ask_fp + self.over_ask_tn;
        self.answer_char_f1_sum = if self.answer_reference_turns == 0 {
            0.0
        } else {
            self.answer_char_f1_sum / self.answer_reference_turns as f64
        };
        DialogueEvalReport {
            knowledge_mode: mode.label().to_string(),
            corpus_digest: digest,
            scenarios: self.scenarios,
            turns: self.turns,
            intents: IntentMetrics {
                accuracy: ratio(self.intent_correct, self.turns),
                micro_f1,
                macro_f1,
                per_class,
                confusion: self.confusion,
            },
            entity: self.entity,
            slot: self.slot,
            clarify: self.clarify,
            over_ask_rate: if over_ask_denominator == 0 {
                0.0
            } else {
                self.over_ask_fp as f64 / over_ask_denominator as f64
            },
            takeover: self.takeover,
            answers: AnswerMetrics {
                reference_turns: self.answer_reference_turns,
                answered: self.answer_answered,
                char_f1: self.answer_char_f1_sum,
                exact_match_rate: ratio(self.answer_exact, self.answer_reference_turns),
            },
            provenance: ProvenanceMetrics {
                replies: self.provenance_replies,
                quoted_tokens: self.provenance_quoted,
                traceable_tokens: self.provenance_traceable,
                precision: ratio(self.provenance_traceable, self.provenance_quoted),
            },
            grounding: GroundingMetrics {
                attempted: self.prompt_stats.attempted() as u32,
                grounded: self.prompt_stats.grounded as u32,
                ungrounded: self.prompt_stats.ungrounded as u32,
                parity_mismatches: self.prompt_stats.parity_mismatches as u32,
                slot_rate: self.prompt_stats.grounding_rate(),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// 文本指标
// ---------------------------------------------------------------------------

/// 提取回答段：跳过空行与以「：」结尾的引导性标题行，取首个实质行。
/// 两种知识模式共用（脚本替身无标题行；真实推理器有「根据知识图谱：」等标题）。
pub fn answer_segment(reply: &str) -> &str {
    for line in reply.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.ends_with('：') || trimmed.ends_with(':') {
            continue;
        }
        return trimmed;
    }
    ""
}

/// 归一化：去空白与常见中英文标点，用于 char-F1 / EM。
pub fn normalize_answer(text: &str) -> String {
    text.chars()
        .filter(|ch| {
            !ch.is_whitespace()
                && !"，。！？；：、（）「」『』《》【】…—～,./!?;:()[]{}\"'`·-~".contains(*ch)
        })
        .collect()
}

/// 中文按字符的 SQuAD 式 F1（token 多重集交集）。
pub fn char_f1(predicted: &str, reference: &str) -> f64 {
    let predicted_chars = char_bag(&normalize_answer(predicted));
    let reference_chars = char_bag(&normalize_answer(reference));
    if predicted_chars.is_empty() && reference_chars.is_empty() {
        return 1.0;
    }
    let common: usize = predicted_chars
        .iter()
        .map(|(ch, count)| {
            let reference_count = reference_chars.get(ch).copied().unwrap_or(0);
            (*count).min(reference_count)
        })
        .sum();
    if common == 0 {
        return 0.0;
    }
    let precision = common as f64 / predicted_chars.values().sum::<usize>().max(1) as f64;
    let recall = common as f64 / reference_chars.values().sum::<usize>().max(1) as f64;
    2.0 * precision * recall / (precision + recall)
}

fn char_bag(text: &str) -> HashMap<char, usize> {
    let mut bag = HashMap::new();
    for ch in text.chars() {
        *bag.entry(ch).or_insert(0) += 1;
    }
    bag
}

/// 提取回答中所有「…」引号词条（与对话符合性套件 C4 同口径）。
pub fn quoted_tokens(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find('「') {
        let after = &rest[start + '「'.len_utf8()..];
        let Some(end) = after.find('」') else { break };
        tokens.push(after[..end].to_string());
        rest = &after[end + '」'.len_utf8()..];
    }
    tokens
}

// ---------------------------------------------------------------------------
// 语料摘要
// ---------------------------------------------------------------------------

/// 确定性语料摘要（FNV-1a 64）：金标语料或知识模式变化必须让基线失效。
pub fn corpus_digest(scenarios: &[GoldScenario], mode: EvalKnowledge) -> String {
    let mut canonical = String::new();
    canonical.push_str("mode=");
    canonical.push_str(mode.label());
    canonical.push('\n');
    for scenario in scenarios {
        canonical.push_str("scenario=");
        canonical.push_str(scenario.name);
        canonical.push('\n');
        for (entity, attributes) in scenario.entities {
            canonical.push_str("entity=");
            canonical.push_str(entity);
            for (attribute, value) in *attributes {
                canonical.push_str("|");
                canonical.push_str(attribute);
                canonical.push('=');
                canonical.push_str(value);
            }
            canonical.push('\n');
        }
        for (entity, profile) in scenario.profiles {
            canonical.push_str("profile=");
            canonical.push_str(entity);
            canonical.push('|');
            canonical.push_str(profile);
            canonical.push('\n');
        }
        for turn in scenario.turns {
            canonical.push_str("turn=");
            canonical.push_str(turn.input);
            canonical.push('|');
            canonical.push_str(turn.kind.label());
            canonical.push('|');
            canonical.push_str(&turn.entities.join(","));
            canonical.push('|');
            canonical.push_str(turn.attribute.unwrap_or("-"));
            canonical.push('|');
            canonical.push_str(turn.reference.unwrap_or("-"));
            canonical.push('|');
            canonical.push_str(if turn.clarify { "clarify" } else { "-" });
            canonical.push('|');
            canonical.push_str(if turn.expect_takeover { "takeover" } else { "defer" });
            canonical.push('\n');
        }
    }
    fnv1a_hex(canonical.as_bytes())
}

fn fnv1a_hex(bytes: &[u8]) -> String {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{:016x}", hash)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prf_math_and_empty_denominator_conventions() {
        let mut prf = Prf::default();
        assert_eq!(prf.f1(), 1.0, "没有样本时不应人为扣分");
        prf.observe(1, 1, 0);
        assert!((prf.precision() - 0.5).abs() < 1e-12);
        assert!((prf.recall() - 1.0).abs() < 1e-12);

        let mut bools = Prf::default();
        bools.observe_bool(true, true);
        bools.observe_bool(false, true);
        bools.observe_bool(true, false);
        assert_eq!(bools.tp, 1);
        assert_eq!(bools.fp, 1);
        assert_eq!(bools.fn_, 1);
    }

    #[test]
    fn char_f1_and_normalization_are_deterministic() {
        assert_eq!(normalize_answer("226 厘米。"), "226厘米");
        assert_eq!(char_f1("226", "226"), 1.0);
        assert_eq!(char_f1("", "226"), 0.0);
        let partial = char_f1("姚明的身高是226。", "226");
        assert!(partial > 0.0 && partial < 1.0, "partial={partial}");
    }

    #[test]
    fn scripted_seed_corpus_meets_dialogue_contract() {
        let report = evaluate_seed(EvalKnowledge::Scripted);
        assert_eq!(report.knowledge_mode, "scripted");
        assert!(!report.corpus_digest.is_empty());
        assert_eq!(report.intents.accuracy, 1.0, "{}", report.render_text());
        assert_eq!(report.intents.macro_f1, 1.0, "{}", report.render_text());
        assert_eq!(report.takeover.f1(), 1.0, "{}", report.render_text());
        assert_eq!(report.over_ask_rate, 0.0, "{}", report.render_text());
        assert_eq!(report.provenance.precision, 1.0, "{}", report.render_text());
        assert_eq!(report.grounding.slot_rate, 1.0, "{}", report.render_text());
        assert_eq!(report.grounding.ungrounded, 0, "{}", report.render_text());
        assert_eq!(
            report.grounding.parity_mismatches, 0,
            "{}",
            report.render_text()
        );
        assert!(report.grounding.attempted > 0);
        assert!(report.answers.reference_turns >= 4);
        assert!(
            report.answers.char_f1 > 0.5,
            "char-F1 过低：{}",
            report.render_text()
        );
    }

    #[test]
    fn reasoner_seed_corpus_runs_end_to_end() {
        let report = evaluate_seed(EvalKnowledge::Reasoner);
        assert_eq!(report.knowledge_mode, "reasoner");
        assert_eq!(report.turns, seed_corpus().iter().map(|s| s.turns.len()).sum::<usize>() as u32);
        assert!(
            report.intents.accuracy >= 0.9,
            "真实推理器下意图判定应稳定：{}",
            report.render_text()
        );
        assert_eq!(
            report.provenance.precision, 1.0,
            "真实推理器的回答引用必须可回溯：{}",
            report.render_text()
        );
        for (metric, value) in report.flatten_metrics() {
            assert!(
                (0.0..=1.0).contains(&value),
                "指标 {} 越界：{}",
                metric,
                value
            );
        }
    }

    #[test]
    fn baseline_ratchet_detects_regressions_and_drift() {
        let report = evaluate_seed(EvalKnowledge::Scripted);
        let baseline = DialogueEvalBaseline::from_report(&report);
        let verdict = compare_baseline(&baseline, &report, 0.0);
        assert!(verdict.passed, "{}", verdict.render_text());
        assert!(verdict.digest_matches);

        let mut degraded = report.clone();
        degraded.answers.char_f1 -= 0.1;
        degraded.over_ask_rate += 0.2;
        let verdict = compare_baseline(&baseline, &degraded, 0.0);
        assert!(!verdict.passed);
        assert!(verdict
            .regressions
            .iter()
            .any(|delta| delta.metric == "answer.char_f1"));
        assert!(verdict
            .regressions
            .iter()
            .any(|delta| delta.metric == "clarify.over_ask_rate"));

        let mut drifted = report.clone();
        drifted.corpus_digest = "0000000000000000".to_string();
        let verdict = compare_baseline(&baseline, &drifted, 0.0);
        assert!(!verdict.passed && !verdict.digest_matches);
    }

    #[test]
    fn baseline_roundtrips_through_disk() {
        let report = evaluate_seed(EvalKnowledge::Scripted);
        let baseline = DialogueEvalBaseline::from_report(&report);
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("nlu_baseline.json");
        baseline.save(&path).unwrap();
        let loaded = DialogueEvalBaseline::load(&path).unwrap();
        assert_eq!(loaded.version, baseline.version);
        assert_eq!(loaded.corpus_digest, baseline.corpus_digest);
        assert_eq!(loaded.knowledge_mode, baseline.knowledge_mode);
        assert_eq!(loaded.metrics.len(), baseline.metrics.len());
        for (metric, value) in &baseline.metrics {
            let loaded_value = loaded.metrics.get(metric).copied().unwrap_or(f64::NAN);
            assert!(
                (loaded_value - value).abs() < 1e-12,
                "{} 经磁盘往返漂移：{} vs {}",
                metric,
                loaded_value,
                value
            );
        }
    }

    #[test]
    fn prompt_modes_agree_on_seed_corpus_metrics() {
        let off = evaluate_seed_with(EvalKnowledge::Scripted, PromptMode::Off);
        let parity = evaluate_seed_with(EvalKnowledge::Scripted, PromptMode::Parity);
        let on = evaluate_seed_with(EvalKnowledge::Scripted, PromptMode::On);
        for (name, report) in [("off", &off), ("parity", &parity), ("on", &on)] {
            assert_eq!(report.answers, off.answers, "{} 模式改变了回答指标", name);
            assert_eq!(report.over_ask_rate, off.over_ask_rate, "{}", name);
            assert_eq!(report.takeover, off.takeover, "{}", name);
            assert_eq!(report.grounding.ungrounded, 0, "{}", name);
            assert_eq!(report.grounding.parity_mismatches, 0, "{}", name);
        }
        assert_eq!(off.grounding.attempted, 0, "Off 模式不应尝试模板");
        assert!(parity.grounding.attempted > 0);
        assert_eq!(on.grounding.attempted, parity.grounding.attempted);
        assert_eq!(on.grounding.grounded, parity.grounding.grounded);
    }
}

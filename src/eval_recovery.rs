//! EVR：失败恢复评测（silent-failure 棘轮）。
//!
//! 无监督学习系统里，「失败被静默吞掉」是最贵的缺陷：错误率和知识空腔
//! 混成同一种「不知道」，用户与开发者都无从着手。本模块把恢复行为
//! 变成可回归的指标：
//!
//! - `silent_failure_rate`：On 模式下未被命名的失败占比（目标 0，越低越好）；
//! - `named_failure_rate` / `fault_named_rate`：失败/引擎故障被显式命名的比例；
//! - `receipt_coverage`：失败回执覆盖率（每个失败都有可追溯 id）；
//! - `evidence_rate`：推测必须带证据链的比例（不幻觉的硬条件）；
//! - `ratification_rate`：用户裁定（是/不是）能关闭挂账回执的比例；
//! - `legacy_off_purity`：Off 模式零副作用（旧行为逐字、不起台账）；
//! - `ledger_digest`：整个评测的台账摘要（确定性回归锚点）。
//!
//! 语料是确定性的种子场景；基线 JSON + 棘轮判定与 `dialogue_eval` 同构。

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::context_image::fnv1a_hex;
use crate::dialogue_manager::{DialogueKnowledge, DialogueManager};
use crate::epistemic::EpistemicStatus;
use crate::recovery::{RecoveryMode, RecoveryNomination};

/// 种子知识替身：只提供恢复评测需要的通道（答案 / 提名 / 线索 / 故障）。
#[derive(Default, Clone)]
struct SeedKnowledge {
    entities: Vec<String>,
    known: Vec<String>,
    attributes: BTreeMap<String, Vec<String>>,
    answers: BTreeMap<String, String>,
    values: BTreeMap<(String, String), Vec<String>>,
    nominations: BTreeMap<(String, String), Vec<RecoveryNomination>>,
    clues: BTreeMap<String, Vec<String>>,
    fault: Option<String>,
}

impl SeedKnowledge {
    fn baseline() -> Self {
        let mut knowledge = SeedKnowledge::default();
        for entity in ["姚明", "叶莉"] {
            knowledge.entities.push(entity.to_string());
            knowledge.known.push(entity.to_string());
        }
        knowledge.attributes.insert(
            "姚明".to_string(),
            vec!["身高".to_string(), "国籍".to_string()],
        );
        knowledge.attributes.insert(
            "叶莉".to_string(),
            vec!["身高".to_string(), "国籍".to_string()],
        );
        knowledge.answers.insert(
            "姚明的身高是什么".to_string(),
            "姚明的身高是 226 厘米。".to_string(),
        );
        knowledge
            .values
            .insert(("姚明".to_string(), "国籍".to_string()), vec!["中国".to_string()]);
        knowledge
    }

    fn with_fault(mut self, fault: &str) -> Self {
        self.fault = Some(fault.to_string());
        self
    }

    fn with_nomination(mut self, entity: &str, attribute: &str, value: &str, score: f64) -> Self {
        self.nominations
            .entry((entity.to_string(), attribute.to_string()))
            .or_default()
            .push(RecoveryNomination {
                value: value.to_string(),
                score,
                evidence: vec!["链接预测：同位置分布统计".to_string()],
            });
        self
    }

    fn with_clue(mut self, entity: &str, clue: &str) -> Self {
        self.clues
            .entry(entity.to_string())
            .or_default()
            .push(clue.to_string());
        self
    }
}

impl DialogueKnowledge for SeedKnowledge {
    fn entities_in(&self, text: &str) -> Vec<String> {
        let mut found: Vec<String> = self
            .entities
            .iter()
            .filter(|entity| text.contains(entity.as_str()))
            .cloned()
            .collect();
        found.sort_by_key(|entity| std::cmp::Reverse(entity.chars().count()));
        found
    }

    fn known_entity(&self, entity: &str) -> bool {
        self.known.iter().any(|known| known == entity)
    }

    fn answer(&self, question: &str) -> Option<String> {
        self.answers.get(question).cloned()
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

    fn recovery_nominations(&self, entity: &str, attribute: &str) -> Vec<RecoveryNomination> {
        self.nominations
            .get(&(entity.to_string(), attribute.to_string()))
            .cloned()
            .unwrap_or_default()
    }

    fn recovery_clues(&self, entity: &str) -> Vec<String> {
        self.clues.get(entity).cloned().unwrap_or_default()
    }

    fn engine_fault(&self) -> Option<String> {
        self.fault.clone()
    }
}

/// 种子场景：名 + 知识 + 逐轮输入（确定性）。
fn scenarios() -> Vec<(&'static str, SeedKnowledge, Vec<&'static str>)> {
    vec![
        (
            "engine_fault",
            SeedKnowledge::baseline().with_fault("Prolog consult 崩溃：index out of bounds"),
            vec!["姚明的身高是多少"],
        ),
        (
            "absent_clarify_then_answer",
            SeedKnowledge::baseline().with_clue("姚明", "国籍"),
            vec!["姚明的体重是多少", "他体重140公斤"],
        ),
        (
            "absent_admit_after_pending",
            SeedKnowledge::baseline(),
            vec!["姚明的体重是多少", "姚明的体重是多少"],
        ),
        (
            "speculation_refuted",
            SeedKnowledge::baseline().with_nomination("姚明", "体重", "140公斤", 0.72),
            vec!["姚明的体重是多少", "不是"],
        ),
        (
            "speculation_confirmed",
            SeedKnowledge::baseline().with_nomination("姚明", "体重", "140公斤", 0.72),
            vec!["姚明的体重是多少", "是"],
        ),
    ]
}

/// 语料摘要：场景名 + 输入 + 知识指纹（确定性）。
fn corpus_digest() -> String {
    let mut text = String::new();
    for (name, knowledge, inputs) in scenarios() {
        text.push_str(name);
        text.push('\u{1}');
        for input in inputs {
            text.push_str(input);
            text.push('\u{2}');
        }
        text.push_str(&serde_json::to_string(&knowledge.answers).unwrap_or_default());
        text.push_str(&serde_json::to_string(&knowledge.fault).unwrap_or_default());
        text.push('\n');
    }
    fnv1a_hex(text.as_bytes())
}

/// 状态措辞是否被显式命名（silent failure 的判据）。
fn status_is_named(status: EpistemicStatus, text: &str) -> bool {
    let markers: &[&str] = match status {
        EpistemicStatus::Certain => return true,
        EpistemicStatus::Provisional => &["推测"],
        EpistemicStatus::Conflict => &["矛盾", "说法"],
        EpistemicStatus::Absent => &["确定没有", "没有记录"],
        EpistemicStatus::Unknown => &["暂时", "没有"],
        EpistemicStatus::Unparseable => &["解析"],
        EpistemicStatus::EngineFault => &["出错", "故障"],
    };
    markers.iter().any(|marker| text.contains(marker))
}

/// 一轮场景执行的观察。
struct TurnObservation {
    status: Option<EpistemicStatus>,
    level: Option<String>,
    receipt: bool,
    evidence: usize,
    text_len: usize,
    named: bool,
    fault: bool,
    fault_named: bool,
    plan: Vec<String>,
}

/// 跑一个场景（模式固定），返回逐轮观察。
fn run_scenario(knowledge: &SeedKnowledge, mode: RecoveryMode, inputs: &[&str]) -> Vec<TurnObservation> {
    let mut manager = DialogueManager::new();
    manager.set_recovery_mode(mode);
    let mut observations = Vec::new();
    for input in inputs {
        let reply = manager
            .plan_pure_dialogue(input, knowledge)
            .or_else(|| manager.plan_reply(input, knowledge));
        let Some(reply) = reply else {
            observations.push(TurnObservation {
                status: None,
                level: None,
                receipt: false,
                evidence: 0,
                text_len: 0,
                named: false,
                fault: false,
                fault_named: false,
                plan: Vec::new(),
            });
            continue;
        };
        let notice = manager.last_recovery();
        let status = notice.map(|notice| notice.status);
        observations.push(TurnObservation {
            status,
            level: notice.map(|notice| notice.level.clone()),
            receipt: notice.and_then(|notice| notice.receipt_id.clone()).is_some(),
            evidence: notice.map(|notice| notice.evidence.len()).unwrap_or(0),
            text_len: reply.text.chars().count(),
            named: status.is_some_and(|status| status_is_named(status, &reply.text)),
            fault: status == Some(EpistemicStatus::EngineFault),
            fault_named: status == Some(EpistemicStatus::EngineFault)
                && status_is_named(EpistemicStatus::EngineFault, &reply.text),
            plan: reply.plan,
        });
    }
    observations
}

/// 恢复评测报告（可 JSON、可棘轮）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalRecoveryReport {
    pub corpus_digest: String,
    pub scenarios: usize,
    pub turns: usize,
    pub failures: usize,
    pub silent_failures: usize,
    pub silent_failure_rate: f64,
    pub named_failure_rate: f64,
    pub receipt_coverage: f64,
    pub evidence_rate: f64,
    pub ratification_rate: f64,
    pub fault_named_rate: f64,
    pub legacy_off_purity: f64,
    pub ladder: BTreeMap<String, usize>,
    pub ledger_digest: String,
}

impl EvalRecoveryReport {
    pub fn render_text(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "失败恢复评测：{} 个场景 / {} 轮；失败 {} 次（静默 {}）。\n",
            self.scenarios, self.turns, self.failures, self.silent_failures
        ));
        out.push_str(&format!(
            "  silent_failure_rate {:.4}（越低越好）；named_failure_rate {:.4}；fault_named_rate {:.4}\n",
            self.silent_failure_rate, self.named_failure_rate, self.fault_named_rate
        ));
        out.push_str(&format!(
            "  receipt_coverage {:.4}；evidence_rate {:.4}；ratification_rate {:.4}；legacy_off_purity {:.4}\n",
            self.receipt_coverage, self.evidence_rate, self.ratification_rate, self.legacy_off_purity
        ));
        let ladder: Vec<String> = self
            .ladder
            .iter()
            .map(|(level, count)| format!("{}×{}", level, count))
            .collect();
        out.push_str(&format!("  恢复档位分布：{}\n", ladder.join("、")));
        out.push_str(&format!("  台账摘要：{}", self.ledger_digest));
        out
    }

    pub fn flatten_metrics(&self) -> BTreeMap<String, f64> {
        let mut metrics = BTreeMap::new();
        metrics.insert("recovery.silent_failure_rate".to_string(), self.silent_failure_rate);
        metrics.insert("recovery.named_failure_rate".to_string(), self.named_failure_rate);
        metrics.insert("recovery.receipt_coverage".to_string(), self.receipt_coverage);
        metrics.insert("recovery.evidence_rate".to_string(), self.evidence_rate);
        metrics.insert("recovery.ratification_rate".to_string(), self.ratification_rate);
        metrics.insert("recovery.fault_named_rate".to_string(), self.fault_named_rate);
        metrics.insert("recovery.legacy_off_purity".to_string(), self.legacy_off_purity);
        for (level, count) in &self.ladder {
            metrics.insert(format!("recovery.ladder.{}.count", level), *count as f64);
        }
        metrics
    }
}

/// 越大越差的指标（棘轮反向判定）。
pub const LOWER_IS_BETTER: [&str; 1] = ["recovery.silent_failure_rate"];

/// 跑种子语料：On 模式统计失败命名/回执/证据/裁定；Off 模式验证零副作用。
pub fn evaluate_seed() -> EvalRecoveryReport {
    let corpus = scenarios();
    let mut turns = 0usize;
    let mut failures = 0usize;
    let mut silent = 0usize;
    let mut named = 0usize;
    let mut with_receipt = 0usize;
    let mut provisional = 0usize;
    let mut with_evidence = 0usize;
    let mut verdicts = 0usize;
    let mut ratified = 0usize;
    let mut faults = 0usize;
    let mut faults_named = 0usize;
    let mut ladder: BTreeMap<String, usize> = BTreeMap::new();
    let mut digest_inputs = String::new();
    let mut ledger_digests = Vec::new();

    for (name, knowledge, inputs) in &corpus {
        let observations = run_scenario(knowledge, RecoveryMode::On, inputs);
        turns += observations.len();
        for observation in &observations {
            digest_inputs.push_str(name);
            digest_inputs.push_str(&observation.plan.join("+"));
            digest_inputs.push_str(&format!("|{}|{}", observation.evidence, observation.text_len));
            // 裁定/接收轮是回执的关闭动作，不是新失败：只参与 ratification 统计。
            let is_closure = observation
                .plan
                .first()
                .is_some_and(|plan| plan.starts_with("recovery_"));
            if is_closure {
                continue;
            }
            if let Some(status) = observation.status {
                if status.is_failure() {
                    failures += 1;
                    if observation.named {
                        named += 1;
                    } else {
                        silent += 1;
                    }
                    if observation.receipt {
                        with_receipt += 1;
                    }
                }
                if status == EpistemicStatus::Provisional {
                    provisional += 1;
                    if observation.evidence > 0 {
                        with_evidence += 1;
                    }
                }
                if observation.fault {
                    faults += 1;
                    if observation.fault_named {
                        faults_named += 1;
                    }
                }
                if let Some(level) = &observation.level {
                    *ladder.entry(level.clone()).or_insert(0) += 1;
                }
            }
        }
        // 裁定轮：是/不是 之后挂账应清零
        let mut manager = DialogueManager::new();
        manager.set_recovery_mode(RecoveryMode::On);
        for input in inputs {
            let _ = manager
                .plan_pure_dialogue(input, knowledge)
                .or_else(|| manager.plan_reply(input, knowledge));
            if *input == "是" || *input == "不是" {
                verdicts += 1;
                if manager.recovery_state().pending.is_empty() {
                    ratified += 1;
                }
            }
        }
        ledger_digests.push(manager.failure_ledger().digest());
    }

    // Off 纯度：所有场景在 Off 模式下不得留下任何回执/挂账。
    let mut off_pure = 0usize;
    for (_name, knowledge, inputs) in &corpus {
        let mut legacy = DialogueManager::new();
        for input in inputs {
            let _ = legacy
                .plan_pure_dialogue(input, knowledge)
                .or_else(|| legacy.plan_reply(input, knowledge));
        }
        if legacy.failure_ledger().is_empty()
            && legacy.recovery_state().pending.is_empty()
            && legacy.last_recovery().is_none()
        {
            off_pure += 1;
        }
    }

    let ratio = |part: usize, whole: usize| if whole == 0 { 1.0 } else { part as f64 / whole as f64 };
    EvalRecoveryReport {
        corpus_digest: corpus_digest(),
        scenarios: corpus.len(),
        turns,
        failures,
        silent_failures: silent,
        silent_failure_rate: ratio(silent, failures),
        named_failure_rate: ratio(named, failures),
        receipt_coverage: ratio(with_receipt, failures),
        evidence_rate: ratio(with_evidence, provisional),
        ratification_rate: ratio(ratified, verdicts),
        fault_named_rate: ratio(faults_named, faults),
        legacy_off_purity: ratio(off_pure, corpus.len()),
        ladder,
        ledger_digest: fnv1a_hex(ledger_digests.join("\u{1}").as_bytes()),
    }
}

// ---------------------------------------------------------------------------
// 基线棘轮
// ---------------------------------------------------------------------------

/// 可写入磁盘的恢复评测基线。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalRecoveryBaseline {
    pub version: u32,
    pub corpus_digest: String,
    pub metrics: BTreeMap<String, f64>,
}

pub const RECOVERY_BASELINE_VERSION: u32 = 1;

impl EvalRecoveryBaseline {
    pub fn from_report(report: &EvalRecoveryReport) -> Self {
        Self {
            version: RECOVERY_BASELINE_VERSION,
            corpus_digest: report.corpus_digest.clone(),
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
        let baseline: EvalRecoveryBaseline = serde_json::from_str(&text)?;
        Ok(baseline)
    }
}

/// 单项指标变化。
#[derive(Debug, Clone, PartialEq)]
pub struct RecoveryMetricDelta {
    pub metric: String,
    pub baseline: f64,
    pub current: f64,
}

/// 棘轮判定。
#[derive(Debug, Clone, PartialEq)]
pub struct RecoveryRatchetVerdict {
    pub passed: bool,
    pub digest_matches: bool,
    pub regressions: Vec<RecoveryMetricDelta>,
    pub gains: Vec<RecoveryMetricDelta>,
}

impl RecoveryRatchetVerdict {
    pub fn render_text(&self) -> String {
        if self.passed {
            return "恢复棘轮检查通过：无指标低于基线。\n".to_string();
        }
        let mut out = String::from("恢复棘轮检查未通过：\n");
        if !self.digest_matches {
            out.push_str("  - 语料摘要不匹配：种子场景已漂移，基线不可直接比较\n");
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
        out
    }
}

/// 与基线比较：语料摘要不同或任何指标退化超 ε，即不通过。
pub fn compare_baseline(
    baseline: &EvalRecoveryBaseline,
    current: &EvalRecoveryReport,
    epsilon: f64,
) -> RecoveryRatchetVerdict {
    let digest_matches = baseline.corpus_digest == current.corpus_digest;
    let current_metrics = current.flatten_metrics();
    let mut regressions = Vec::new();
    let mut gains = Vec::new();
    for (metric, baseline_value) in &baseline.metrics {
        let current_value = current_metrics.get(metric).copied().unwrap_or(0.0);
        let lower_is_better = LOWER_IS_BETTER.contains(&metric.as_str());
        let (is_regression, is_gain) = if lower_is_better {
            (
                current_value > baseline_value + epsilon,
                current_value < baseline_value - epsilon,
            )
        } else {
            (
                current_value < baseline_value - epsilon,
                current_value > baseline_value + epsilon,
            )
        };
        if is_regression {
            regressions.push(RecoveryMetricDelta {
                metric: metric.clone(),
                baseline: *baseline_value,
                current: current_value,
            });
        } else if is_gain {
            gains.push(RecoveryMetricDelta {
                metric: metric.clone(),
                baseline: *baseline_value,
                current: current_value,
            });
        }
    }
    RecoveryRatchetVerdict {
        passed: digest_matches && regressions.is_empty(),
        digest_matches,
        regressions,
        gains,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_report_is_silent_failure_free() {
        let report = evaluate_seed();
        assert_eq!(report.silent_failure_rate, 0.0, "{}", report.render_text());
        assert_eq!(report.receipt_coverage, 1.0, "{}", report.render_text());
        assert_eq!(report.evidence_rate, 1.0, "{}", report.render_text());
        assert_eq!(report.ratification_rate, 1.0, "{}", report.render_text());
        assert_eq!(report.fault_named_rate, 1.0, "{}", report.render_text());
        assert_eq!(report.legacy_off_purity, 1.0, "{}", report.render_text());
        assert!(report.failures > 0, "语料必须真的包含失败");
        assert!(
            report.ladder.contains_key("L2") && report.ladder.contains_key("L3"),
            "阶梯应覆盖联想与澄清：{:?}",
            report.ladder
        );
    }

    #[test]
    fn report_is_deterministic() {
        let first = evaluate_seed();
        let second = evaluate_seed();
        assert_eq!(first, second);
        assert_eq!(first.render_text(), second.render_text());
    }

    #[test]
    fn baseline_roundtrip_and_regression_detection() {
        let report = evaluate_seed();
        let baseline = EvalRecoveryBaseline::from_report(&report);
        let verdict = compare_baseline(&baseline, &report, 0.0);
        assert!(verdict.passed, "{}", verdict.render_text());

        // 退化：静默失败率上升必须被判为退化（越低越好）
        let mut degraded = report.clone();
        degraded.silent_failure_rate = 0.25;
        let verdict = compare_baseline(&baseline, &degraded, 0.0);
        assert!(!verdict.passed);
        assert!(verdict
            .regressions
            .iter()
            .any(|delta| delta.metric == "recovery.silent_failure_rate"));

        // 语料漂移：摘要不同则不可直接比较
        let mut drifted = report.clone();
        drifted.corpus_digest = "deadbeef".to_string();
        let verdict = compare_baseline(&baseline, &drifted, 0.0);
        assert!(!verdict.passed && !verdict.digest_matches);
    }

    #[test]
    fn baseline_file_roundtrip() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("recovery_baseline.json");
        let report = evaluate_seed();
        let baseline = EvalRecoveryBaseline::from_report(&report);
        baseline.save(&path).unwrap();
        let loaded = EvalRecoveryBaseline::load(&path).unwrap();
        assert_eq!(baseline, loaded);
        let verdict = compare_baseline(&loaded, &report, 1e-9);
        assert!(verdict.passed, "{}", verdict.render_text());
    }
}

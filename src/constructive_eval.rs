//! 建设性多跳的离线恢复基准（M0）。
//!
//! 基准刻意把「训练规则」与「评测事实」分开：规则只在训练图上挖掘，评测图
//! 抽掉目标边，并保留合成目标所需的全部体部事实。这样能直接回答两个问题：
//!
//! 1. 单跳链接预测在缺少中间边时能恢复多少目标；
//! 2. 允许虚拟事实继续参与下一跳后，恢复率提升多少。
//!
//! 所有图、规则、预算与排序都是确定性的；没有随机数、没有训练权重。

use serde::{Deserialize, Serialize};

use crate::constructive::{ConstructionConfig, ConstructiveReasoner, SupportAudit};
use crate::link_prediction::{LinkPredictor, Triple};

pub const BENCH_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConstructionCase {
    pub seed: String,
    pub predicate: String,
    pub expected: String,
    pub baseline_recovered: bool,
    pub constructive_recovered: bool,
    pub constructive_status: String,
    pub derived_target_facts: usize,
    pub wrong_target_facts: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConstructionMetrics {
    pub cases: usize,
    pub baseline_recovered: usize,
    pub constructive_recovered: usize,
    pub recovered_over_baseline: usize,
    pub derived_target_facts: usize,
    pub wrong_target_facts: usize,
    pub truncated_cases: usize,
    pub support_audit_clean: bool,
}

impl ConstructionMetrics {
    pub fn recall_bps(&self) -> u16 {
        ratio_bps(self.constructive_recovered, self.cases)
    }

    pub fn baseline_recall_bps(&self) -> u16 {
        ratio_bps(self.baseline_recovered, self.cases)
    }

    pub fn precision_bps(&self) -> u16 {
        let total = self.derived_target_facts + self.wrong_target_facts;
        if total == 0 {
            return 0;
        }
        ratio_bps(self.derived_target_facts, total)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConstructionEvalReport {
    pub version: u32,
    pub cases: Vec<ConstructionCase>,
    pub metrics: ConstructionMetrics,
    pub support_audit: SupportAuditSnapshot,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SupportAuditSnapshot {
    pub inspected: usize,
    pub invalid: usize,
    pub cyclic: usize,
}

impl SupportAuditSnapshot {
    fn from_audit(audit: &SupportAudit) -> Self {
        Self {
            inspected: audit.inspected,
            invalid: audit.invalid.len(),
            cyclic: audit.cyclic.len(),
        }
    }
}

impl ConstructionEvalReport {
    pub fn render_text(&self) -> String {
        let metrics = &self.metrics;
        let mut output = format!(
            "建设性多跳离线基准（v{}）：{} 个缺边多跳目标\n",
            self.version, metrics.cases
        );
        output.push_str(&format!(
            "  单跳链接预测恢复：{}/{}（{:.1}%）\n",
            metrics.baseline_recovered,
            metrics.cases,
            f64::from(metrics.baseline_recall_bps()) / 100.0
        ));
        output.push_str(&format!(
            "  建设性多跳恢复：{}/{}（{:.1}%），其中 {} 个是链接预测未覆盖的\n",
            metrics.constructive_recovered,
            metrics.cases,
            f64::from(metrics.recall_bps()) / 100.0,
            metrics.recovered_over_baseline
        ));
        output.push_str(&format!(
            "  目标推断精度：{:.1}%（派生 {} 条，错误 {} 条）\n",
            f64::from(metrics.precision_bps()) / 100.0,
            metrics.derived_target_facts,
            metrics.wrong_target_facts
        ));
        output.push_str(&format!(
            "  证明审计：{} 条虚拟事实，无效 {}，循环 {}；截断用例 {}\n",
            self.support_audit.inspected,
            self.support_audit.invalid,
            self.support_audit.cyclic,
            metrics.truncated_cases
        ));
        for case in &self.cases {
            output.push_str(&format!(
                "  - {} 的 {}：期望 {}；链接预测 {}，建设性 {}（{}，派生 {}，错误 {}）\n",
                case.seed,
                case.predicate,
                case.expected,
                if case.baseline_recovered { "命中" } else { "未命中" },
                if case.constructive_recovered { "命中" } else { "未命中" },
                case.constructive_status,
                case.derived_target_facts,
                case.wrong_target_facts
            ));
        }
        output.push_str(
            "说明：虚拟事实只在查询内参与多跳，不写基础图谱；关闭建设性开关时本基准不改变回答路径。",
        );
        output
    }

    pub fn json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }
}

pub fn evaluate() -> ConstructionEvalReport {
    let training = training_triples();
    let evaluation = evaluation_triples();
    let mut baseline = LinkPredictor::new();
    for triple in &training {
        baseline.add_triple(&triple.subject, &triple.predicate, &triple.object);
    }
    baseline.mine_rules(2, 0.5);
    for triple in &evaluation {
        baseline.add_triple(&triple.subject, &triple.predicate, &triple.object);
    }

    let config = ConstructionConfig {
        enabled: true,
        max_depth: 4,
        min_confidence_bps: 3_500,
        ..ConstructionConfig::default()
    };
    let rules = baseline.rules().to_vec();
    let mut engine = ConstructiveReasoner::from_predictor(
        config,
        1,
        &evaluation,
        &rules,
    );

    let mut cases = Vec::new();
    let mut baseline_recovered = 0usize;
    let mut constructive_recovered = 0usize;
    let mut derived_total = 0usize;
    let mut wrong_total = 0usize;
    let mut truncated_cases = 0usize;
    for (seed, predicate, expected) in [
        ("评测甲", "祖传", "评测丁"),
        ("评测戊", "祖传", "评测辛"),
    ] {
        let predictions = baseline.predict_for_entity(seed, Some(predicate), 16);
        let baseline_hit = predictions
            .iter()
            .any(|prediction| prediction.triple.object == expected);
        let report = engine.derive(seed, predicate, false);
        let derived: Vec<String> = report
            .target_facts
            .iter()
            .map(|fact| fact.key.object.clone())
            .collect();
        let constructive_hit = derived.iter().any(|value| value == expected);
        let wrong = derived.iter().filter(|value| value.as_str() != expected).count();
        if baseline_hit {
            baseline_recovered += 1;
        }
        if constructive_hit {
            constructive_recovered += 1;
        }
        if report.truncated {
            truncated_cases += 1;
        }
        derived_total += derived.len();
        wrong_total += wrong;
        cases.push(ConstructionCase {
            seed: seed.to_string(),
            predicate: predicate.to_string(),
            expected: expected.to_string(),
            baseline_recovered: baseline_hit,
            constructive_recovered: constructive_hit,
            constructive_status: report.status.label().to_string(),
            derived_target_facts: derived.len(),
            wrong_target_facts: wrong,
        });
    }
    let audit = engine.support_audit();
    let recover_over = cases
        .iter()
        .filter(|case| case.constructive_recovered && !case.baseline_recovered)
        .count();
    engine.end_query();

    ConstructionEvalReport {
        version: BENCH_VERSION,
        cases,
        metrics: ConstructionMetrics {
            cases: 2,
            baseline_recovered,
            constructive_recovered,
            recovered_over_baseline: recover_over,
            derived_target_facts: derived_total,
            wrong_target_facts: wrong_total,
            truncated_cases,
            support_audit_clean: audit.is_clean() && audit.inspected > 0,
        },
        support_audit: SupportAuditSnapshot::from_audit(&audit),
    }
}

fn training_triples() -> Vec<Triple> {
    vec![
        Triple::new("训练甲", "同门", "训练乙"),
        Triple::new("训练乙", "师承", "训练丙"),
        Triple::new("训练甲", "传承", "训练丙"),
        Triple::new("训练丁", "同门", "训练戊"),
        Triple::new("训练戊", "师承", "训练己"),
        Triple::new("训练丁", "传承", "训练己"),
        Triple::new("训练庚", "传承", "训练辛"),
        Triple::new("训练辛", "再传", "训练壬"),
        Triple::new("训练庚", "祖传", "训练壬"),
        Triple::new("训练癸", "传承", "训练子"),
        Triple::new("训练子", "再传", "训练丑"),
        Triple::new("训练癸", "祖传", "训练丑"),
    ]
}

fn evaluation_triples() -> Vec<Triple> {
    vec![
        Triple::new("评测甲", "同门", "评测乙"),
        Triple::new("评测乙", "师承", "评测丙"),
        Triple::new("评测丙", "再传", "评测丁"),
        Triple::new("评测戊", "同门", "评测己"),
        Triple::new("评测己", "师承", "评测庚"),
        Triple::new("评测庚", "再传", "评测辛"),
    ]
}

fn ratio_bps(numerator: usize, denominator: usize) -> u16 {
    if denominator == 0 {
        return 0;
    }
    ((numerator as f64 / denominator as f64) * 10_000.0).round() as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructive_recovers_targets_that_one_hop_prediction_misses() {
        let report = evaluate();
        assert_eq!(report.metrics.cases, 2);
        assert_eq!(report.metrics.baseline_recovered, 0);
        assert_eq!(report.metrics.constructive_recovered, 2);
        assert_eq!(report.metrics.recovered_over_baseline, 2);
        assert_eq!(report.metrics.wrong_target_facts, 0);
        assert_eq!(report.metrics.recall_bps(), 10_000);
        assert_eq!(report.metrics.precision_bps(), 10_000);
        assert!(report.metrics.support_audit_clean);
    }

    #[test]
    fn report_is_deterministic_and_explains_the_comparison() {
        let left = evaluate();
        let right = evaluate();
        assert_eq!(left, right);
        let text = left.render_text();
        assert!(text.contains("单跳链接预测恢复：0/2"), "{}", text);
        assert!(text.contains("建设性多跳恢复：2/2"), "{}", text);
        assert!(text.contains("不写基础图谱"), "{}", text);
        assert!(left.json().contains("\"constructive_recovered\": 2"));
    }

    #[test]
    fn evaluation_cases_are_not_part_of_training() {
        let training: std::collections::HashSet<Triple> =
            training_triples().into_iter().collect();
        for triple in evaluation_triples() {
            assert!(!training.contains(&triple));
        }
        assert!(evaluation_triples()
            .iter()
            .all(|triple| !triple.predicate.contains("祖传")));
    }
}

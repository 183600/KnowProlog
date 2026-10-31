//! TF：回答分类学课程（Taxonomy Fuzz）。
//!
//! QDF 逐项变异知识质量，本模块补另一个维度：**回答图式 × 追问转移 ×
//! 知识缺口**的确定性组合。每个用例都是一份最小会话契约，经由与 CLI /
//! 会话契约完全相同的入口回放，再用结构不变量做 oracle：
//!
//! - 已回答的回合必须有可解析的回答图式；
//! - 直接属性问句在知识完整时闭合，在缺属性时显式欠「回答 + 依据」；
//! - 未知实体让位为未回答，而不是伪造一个图式；
//! - 「还有呢 / 你怎么知道的」必须留在同一回答回合（轮次递增）；
//! - 换主体必须开启新回合（轮次回到 1），回顾必须落到 `recap`；
//! - 相同种子两次运行得到逐字节一致的报告。
//!
//! 与手写 fixture 的分工：fixture 钉住产品措辞与具体策略；本模块枚举组合，
//! 失败会缩减成「世界 × 追问计划 × 不变量」的最小签名，可直接转成会话契约。

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::answer_taxonomy::{AnswerSchema, Obligation};
use crate::context_image::fnv1a_hex;
use crate::query_scenario::{ScenarioEntity, ScenarioRelation};
use crate::session_contract::{
    run_contract, ContractTurn, SessionContract, SESSION_CONTRACT_VERSION,
};

/// 课程格式版本；不变量变化时递增。
pub const TAXONOMY_FUZZ_VERSION: u32 = 1;
/// 默认课程种子。
pub const TAXONOMY_FUZZ_SEED: u64 = 20261031;

/// 知识世界变异。
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum WorldVariant {
    /// 实体与所问属性都存在。
    Complete,
    /// 实体存在，但所问属性缺失。
    MissingAttribute,
    /// 实体不存在。
    MissingEntity,
    /// 只有关系，没有字面量属性。
    RelationOnly,
}

impl WorldVariant {
    pub fn label(self) -> &'static str {
        match self {
            WorldVariant::Complete => "complete",
            WorldVariant::MissingAttribute => "missing_attribute",
            WorldVariant::MissingEntity => "missing_entity",
            WorldVariant::RelationOnly => "relation_only",
        }
    }

    fn question(self) -> &'static str {
        match self {
            WorldVariant::MissingAttribute => "姚明的体重是多少",
            WorldVariant::MissingEntity => "张三的身高是多少",
            WorldVariant::Complete | WorldVariant::RelationOnly => "姚明的身高是多少",
        }
    }

    fn entities(self) -> Vec<ScenarioEntity> {
        let mut ming = BTreeMap::new();
        let mut li = BTreeMap::new();
        if !matches!(self, WorldVariant::RelationOnly) {
            ming.insert("身高".to_string(), "226".to_string());
            ming.insert("国籍".to_string(), "中国".to_string());
            li.insert("身高".to_string(), "190".to_string());
            li.insert("国籍".to_string(), "中国".to_string());
        }
        vec![
            ScenarioEntity {
                name: "姚明".to_string(),
                entity_type: Some("Person".to_string()),
                attributes: ming,
            },
            ScenarioEntity {
                name: "叶莉".to_string(),
                entity_type: Some("Person".to_string()),
                attributes: li,
            },
        ]
    }

    fn relations(self) -> Vec<ScenarioRelation> {
        match self {
            WorldVariant::RelationOnly => vec![ScenarioRelation {
                from: "姚明".to_string(),
                predicate: "师傅".to_string(),
                to: "叶莉".to_string(),
            }],
            _ => Vec::new(),
        }
    }
}

/// 追问计划：从同一个直接问句出发的受控转移。
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum FollowUpPlan {
    /// 只问一次。
    Direct,
    /// 直接问句后说「还有呢」。
    Continue,
    /// 直接问句后追问依据。
    Why,
    /// 直接问句后换主体。
    NewSubject,
    /// 直接问句后回顾。
    Recap,
    /// 直接问句后显式比较两个主体。
    Compare,
}

impl FollowUpPlan {
    pub fn label(self) -> &'static str {
        match self {
            FollowUpPlan::Direct => "direct",
            FollowUpPlan::Continue => "continue",
            FollowUpPlan::Why => "why",
            FollowUpPlan::NewSubject => "new_subject",
            FollowUpPlan::Recap => "recap",
            FollowUpPlan::Compare => "compare",
        }
    }

    fn followup(self) -> Option<&'static str> {
        match self {
            FollowUpPlan::Direct => None,
            FollowUpPlan::Continue => Some("还有呢"),
            FollowUpPlan::Why => Some("你怎么知道的？"),
            FollowUpPlan::NewSubject => Some("叶莉的国籍是多少"),
            FollowUpPlan::Recap => Some("我们聊过什么？"),
            FollowUpPlan::Compare => Some("比较姚明和叶莉的身高"),
        }
    }
}

const WORLDS: [WorldVariant; 4] = [
    WorldVariant::Complete,
    WorldVariant::MissingAttribute,
    WorldVariant::MissingEntity,
    WorldVariant::RelationOnly,
];

const PLANS: [FollowUpPlan; 6] = [
    FollowUpPlan::Direct,
    FollowUpPlan::Continue,
    FollowUpPlan::Why,
    FollowUpPlan::NewSubject,
    FollowUpPlan::Recap,
    FollowUpPlan::Compare,
];

/// 一个可复现的最小失败。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TaxonomyFuzzFailure {
    pub case: usize,
    pub world: String,
    pub plan: String,
    pub input: String,
    pub invariant: String,
    pub detail: String,
}

impl TaxonomyFuzzFailure {
    /// 最小签名：同一世界、同一追问计划、同一不变量只保留一次。
    pub fn signature(&self) -> (String, String, String) {
        (
            self.world.clone(),
            self.plan.clone(),
            self.invariant.clone(),
        )
    }
}

/// 课程报告（确定性，可按种子复现）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaxonomyFuzzReport {
    pub version: u32,
    pub seed: u64,
    pub cases: usize,
    pub passed: usize,
    pub failed: usize,
    pub checks: usize,
    pub failures: Vec<TaxonomyFuzzFailure>,
    pub digest: String,
}

impl TaxonomyFuzzReport {
    pub fn minimal_failures(&self) -> Vec<TaxonomyFuzzFailure> {
        let mut seen = BTreeSet::new();
        let mut minimal = Vec::new();
        for failure in &self.failures {
            if seen.insert(failure.signature()) {
                minimal.push(failure.clone());
            }
        }
        minimal
    }

    pub fn render_text(&self) -> String {
        let minimal = self.minimal_failures();
        let mut lines = vec![format!(
            "TF 回答分类学课程 v{}：种子 {}，用例 {}（通过 {} / 失败 {}），检查 {} 次，最小失败签名 {} 条。",
            self.version,
            self.seed,
            self.cases,
            self.passed,
            self.failed,
            self.checks,
            minimal.len()
        )];
        lines.push(format!(
            "  组合：{} 个知识世界 × {} 个追问计划 = {} 份会话契约。",
            WORLDS.len(),
            PLANS.len(),
            WORLDS.len() * PLANS.len()
        ));
        if minimal.is_empty() {
            lines.push(
                "  所有图式、缺口与追问转移通过：直接回答闭合、缺属性显式欠账、续聊留在同一回合、换主体开启新回合。"
                    .to_string(),
            );
        } else {
            for failure in minimal.iter().take(12) {
                lines.push(format!(
                    "  [{}] {} / {} / {}：{}（{}）",
                    failure.case,
                    failure.world,
                    failure.plan,
                    failure.input,
                    failure.invariant,
                    failure.detail
                ));
            }
            if minimal.len() > 12 {
                lines.push(format!("  ……其余 {} 条省略", minimal.len() - 12));
            }
        }
        lines.push(format!("摘要：{}", self.digest));
        lines.join("\n")
    }
}

/// 跑完整课程。
pub fn run(seed: u64) -> TaxonomyFuzzReport {
    let mut failures = Vec::new();
    let mut checks = 0usize;
    let mut passed = 0usize;
    let mut case = 0usize;
    for world in WORLDS {
        for plan in PLANS {
            case += 1;
            let report = run_contract(&course_contract(seed, case, world, plan));
            let mut case_failures = Vec::new();
            inspect_case(case, world, plan, &report, &mut checks, &mut case_failures);
            if case_failures.is_empty() {
                passed += 1;
            } else {
                failures.extend(case_failures);
            }
        }
    }
    let failed = case - passed;
    let mut report = TaxonomyFuzzReport {
        version: TAXONOMY_FUZZ_VERSION,
        seed,
        cases: case,
        passed,
        failed,
        checks,
        failures,
        digest: String::new(),
    };
    report.digest = report_digest(&report);
    report
}

fn course_contract(
    seed: u64,
    case: usize,
    world: WorldVariant,
    plan: FollowUpPlan,
) -> SessionContract {
    let mut inputs = vec![world.question().to_string()];
    if let Some(followup) = plan.followup() {
        inputs.push(followup.to_string());
    }
    SessionContract {
        format_version: SESSION_CONTRACT_VERSION,
        name: format!("tf-{}-{:02}-{}", seed, case, world.label()),
        description: Some(format!(
            "回答分类学课程：{} × {}",
            world.label(),
            plan.label()
        )),
        entities: world.entities(),
        relations: world.relations(),
        aliases: Vec::new(),
        turns: inputs
            .into_iter()
            .map(|input| ContractTurn {
                input,
                expect: Default::default(),
                set_revision: None,
                add_entities: Vec::new(),
                add_relations: Vec::new(),
                add_aliases: Vec::new(),
            })
            .collect(),
        require_stable: true,
        expect_digest: None,
    }
}

fn inspect_case(
    case: usize,
    world: WorldVariant,
    plan: FollowUpPlan,
    report: &crate::session_contract::SessionReport,
    checks: &mut usize,
    failures: &mut Vec<TaxonomyFuzzFailure>,
) {
    let mut check = |invariant: &str, input: &str, condition: bool, detail: String| {
        *checks += 1;
        if !condition {
            failures.push(TaxonomyFuzzFailure {
                case,
                world: world.label().to_string(),
                plan: plan.label().to_string(),
                input: input.to_string(),
                invariant: invariant.to_string(),
                detail,
            });
        }
    };
    check(
        "session_stable",
        world.question(),
        report.stable,
        "两次独立回放的会话摘要不一致".to_string(),
    );
    check(
        "all_steps_present",
        world.question(),
        report.steps.len() == if plan.followup().is_some() { 2 } else { 1 },
        format!("实际 {} 步", report.steps.len()),
    );
    for step in &report.steps {
        let receipt = &step.receipt;
        if receipt.answered {
            check(
                "answered_has_schema",
                &receipt.input,
                !receipt.answer_schema.is_empty(),
                "已回答但回答图式为空".to_string(),
            );
            check(
                "schema_is_known",
                &receipt.input,
                AnswerSchema::from_label(&receipt.answer_schema).is_some(),
                format!("未知回答图式「{}」", receipt.answer_schema),
            );
        }
        let mut seen = BTreeSet::new();
        for obligation in &receipt.missing_obligations {
            check(
                "obligation_is_known",
                &receipt.input,
                Obligation::from_label(obligation).is_some(),
                format!("未知槽位「{}」", obligation),
            );
            check(
                "obligation_is_unique",
                &receipt.input,
                seen.insert(obligation.clone()),
                format!("槽位「{}」重复", obligation),
            );
        }
    }
    let Some(direct) = report.steps.first().map(|step| &step.receipt) else {
        return;
    };
    match world {
        WorldVariant::Complete => {
            check(
                "complete_is_answered",
                &direct.input,
                direct.answered,
                "知识完整却没有回答".to_string(),
            );
            check(
                "complete_schema_attribute",
                &direct.input,
                direct.answer_schema == AnswerSchema::Attribute.label(),
                format!("实际图式「{}」", direct.answer_schema),
            );
            check(
                "complete_episode_closed",
                &direct.input,
                direct.missing_obligations.is_empty(),
                format!("仍欠 {:?}", direct.missing_obligations),
            );
            check(
                "complete_answer_has_value",
                &direct.input,
                direct.answer_excerpt.contains("226"),
                format!("回答摘要「{}」", direct.answer_excerpt),
            );
        }
        WorldVariant::MissingAttribute | WorldVariant::RelationOnly => {
            check(
                "gap_is_answered",
                &direct.input,
                direct.answered,
                "缺口应显式回答，而不是静默让位".to_string(),
            );
            check(
                "gap_is_reported",
                &direct.input,
                direct.gaps.iter().any(|gap| gap == "knowledge_gap"),
                format!("缺口 {:?}", direct.gaps),
            );
            check(
                "gap_owes_answer",
                &direct.input,
                direct
                    .missing_obligations
                    .iter()
                    .any(|obligation| obligation == Obligation::Answer.label()),
                format!("欠账 {:?}", direct.missing_obligations),
            );
            check(
                "gap_owes_evidence",
                &direct.input,
                direct
                    .missing_obligations
                    .iter()
                    .any(|obligation| obligation == Obligation::Evidence.label()),
                format!("欠账 {:?}", direct.missing_obligations),
            );
            check(
                "gap_episode_open",
                &direct.input,
                !direct.missing_obligations.is_empty(),
                "缺口回合不应闭合".to_string(),
            );
        }
        WorldVariant::MissingEntity => {
            check(
                "missing_entity_yields",
                &direct.input,
                !direct.answered,
                "未知实体不应伪造回答".to_string(),
            );
            check(
                "missing_entity_has_gap",
                &direct.input,
                direct.gaps.iter().any(|gap| gap == "unhandled"),
                format!("缺口 {:?}", direct.gaps),
            );
        }
    }
    let Some(followup) = report.steps.get(1).map(|step| &step.receipt) else {
        return;
    };
    match plan {
        FollowUpPlan::Direct => {}
        FollowUpPlan::Continue | FollowUpPlan::Why => {
            if direct.answered {
                check(
                    "followup_is_answered",
                    &followup.input,
                    followup.answered,
                    "直接回答后的追问让位了".to_string(),
                );
            }
            if direct.answered && followup.answered {
                check(
                    "followup_keeps_schema",
                    &followup.input,
                    followup.answer_schema == AnswerSchema::Attribute.label(),
                    format!("实际图式「{}」", followup.answer_schema),
                );
                check(
                    "followup_keeps_episode",
                    &followup.input,
                    followup.episode_turns >= 2,
                    format!("回合轮次 {}", followup.episode_turns),
                );
            }
            if plan == FollowUpPlan::Why && direct.answered && followup.answered {
                check(
                    "why_owes_evidence_or_explains",
                    &followup.input,
                    followup
                        .missing_obligations
                        .iter()
                        .any(|obligation| obligation == Obligation::Evidence.label())
                        || followup.evidence.iter().any(|line| !line.is_empty()),
                    format!(
                        "欠账 {:?}；证据 {:?}",
                        followup.missing_obligations, followup.evidence
                    ),
                );
            }
        }
        FollowUpPlan::NewSubject => {
            if followup.answered {
                check(
                    "new_subject_keeps_schema",
                    &followup.input,
                    followup.answer_schema == AnswerSchema::Attribute.label(),
                    format!("实际图式「{}」", followup.answer_schema),
                );
                check(
                    "new_subject_starts_new_episode",
                    &followup.input,
                    followup.episode_turns == 1,
                    format!("回合轮次 {}", followup.episode_turns),
                );
                check(
                    "new_subject_moves_focus",
                    &followup.input,
                    followup.focus_after.as_deref() == Some("叶莉"),
                    format!("焦点 {:?}", followup.focus_after),
                );
            }
        }
        FollowUpPlan::Recap => {
            if followup.answered {
                check(
                    "recap_schema",
                    &followup.input,
                    followup.answer_schema == AnswerSchema::Recap.label(),
                    format!("实际图式「{}」", followup.answer_schema),
                );
                check(
                    "recap_is_closed",
                    &followup.input,
                    followup.missing_obligations.is_empty(),
                    format!("仍欠 {:?}", followup.missing_obligations),
                );
            }
        }
        FollowUpPlan::Compare => {
            if world == WorldVariant::Complete {
                check(
                    "compare_is_answered",
                    &followup.input,
                    followup.answered,
                    "完整知识下比较问句让位了".to_string(),
                );
            }
            if followup.answered {
                check(
                    "compare_schema",
                    &followup.input,
                    followup.answer_schema == AnswerSchema::Comparison.label(),
                    format!("实际图式「{}」", followup.answer_schema),
                );
                check(
                    "compare_starts_new_episode",
                    &followup.input,
                    followup.episode_turns == 1,
                    format!("回合轮次 {}", followup.episode_turns),
                );
            }
        }
    }
}

fn report_digest(report: &TaxonomyFuzzReport) -> String {
    let material = format!(
        "v{}|{}|{}|{}|{}|{}|{}",
        report.version,
        report.seed,
        report.cases,
        report.passed,
        report.failed,
        report.checks,
        report
            .failures
            .iter()
            .map(|failure| format!(
                "{}:{}:{}:{}:{}:{}",
                failure.case,
                failure.world,
                failure.plan,
                failure.input,
                failure.invariant,
                failure.detail
            ))
            .collect::<Vec<_>>()
            .join("\n")
    );
    fnv1a_hex(material.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn taxonomy_course_is_clean_and_stable() {
        let report = run(TAXONOMY_FUZZ_SEED);
        assert_eq!(report.cases, WORLDS.len() * PLANS.len());
        assert_eq!(report.failed, 0, "{}", report.render_text());
        assert!(report.checks >= 60, "检查次数过少：{}", report.checks);
        let again = run(TAXONOMY_FUZZ_SEED);
        assert_eq!(report.digest, again.digest);
        assert_eq!(report.render_text(), again.render_text());
    }

    #[test]
    fn seeds_are_part_of_the_report_identity() {
        let first = run(1);
        let second = run(2);
        assert_ne!(first.digest, second.digest);
        assert_eq!(first.failed, 0, "{}", first.render_text());
        assert_eq!(second.failed, 0, "{}", second.render_text());
    }

    #[test]
    fn every_world_and_plan_pair_is_covered() {
        let mut request = Vec::new();
        for world in WORLDS {
            for plan in PLANS {
                let contract = course_contract(TAXONOMY_FUZZ_SEED, 0, world, plan);
                request.push((contract.entities.len(), contract.turns.len()));
            }
        }
        assert_eq!(request.len(), WORLDS.len() * PLANS.len());
        assert!(request.iter().all(|(entities, turns)| *entities == 2 && *turns >= 1));
        let report = run(TAXONOMY_FUZZ_SEED);
        assert!(report.minimal_failures().is_empty());
    }
}

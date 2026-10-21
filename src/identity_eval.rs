//! 身份识别评测台（Identity Eval）：作用域 × 身份卡 × 会话台账。
//!
//! 与 `link_eval` 的分工：后者评「文本里的指称指哪个实体」；本模块评
//! 「谁在问谁的身份、允许声称什么、更正/遗忘后是否守住作用域」。
//!
//! 四层指标：
//! - act 层：System / User / Entity / Other 四类身份域的准确率与混淆矩阵；
//! - content 层：身份卡 claim 的 precision / recall / F1 与状态准确率；
//! - scope 层：claim 来源是否匹配身份域、证据覆盖、越权声称、未知时 abstain；
//! - session 层：快照恢复后用户名字是否保留、新会话是否泄漏、更正后旧名是否失效、
//!   遗忘后历史值是否真的清空。
//!
//! 棘轮约定：`overclaim_rate` / `scope_leak_rate` / `cross_session_leak_rate` /
//! `stale_mention_rate` / `missing_cards` 越低越好，其余越高越好。

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::context_image::fnv1a_hex;
use crate::contextual_reference::{SenseCard, SenseSource};
use crate::dialogue_manager::{DialogueKnowledge, DialogueManager, ReferenceView};
use crate::entity_linker::EntityLinker;
use crate::identity_dialogue::{
    classify, entity_card, user_card, IdentityCard, IdentityScope, IdentityStatus,
    UserIdentityLedger,
};

/// 一轮身份金标。
#[derive(Debug, Clone)]
pub struct IdentityGoldTurn {
    pub input: &'static str,
    /// 期望的身份作用域。
    pub scope: IdentityScope,
    /// 期望的身份卡状态；`None` 表示这一轮不应产生身份卡。
    pub status: Option<IdentityStatus>,
    /// 期望的身份卡 claim（`key=value` 集合）。
    pub claims: &'static [&'static str],
    /// 实体身份轮的图谱主体（未知实体也要能构造 abstain 卡）。
    pub subject: Option<&'static str>,
}

/// 一个身份评测场景：小图谱 + 顺序多轮 + 会话终态检查。
#[derive(Debug, Clone)]
pub struct IdentityGoldScenario {
    pub name: &'static str,
    pub entities: &'static [(&'static str, Option<&'static str>)],
    pub aliases: &'static [(&'static str, &'static str)],
    pub attributes: &'static [(&'static str, &'static str, &'static str)],
    pub relations: &'static [(&'static str, &'static str, &'static str)],
    pub turns: &'static [IdentityGoldTurn],
    /// 更正 / 遗忘后不得再出现在任何身份卡里的旧词面。
    pub stale_names: &'static [&'static str],
}

const SYSTEM_CLAIMS: &[&str] = &[
    "system:name=KnowProlog",
    "system:role=符号主义中文知识图谱对话系统",
    "system:abilities=图谱查询、规则推理、多轮对话与确定性模板生成",
    "system:limits=不具备意识、情感或主观体验，也不声称拥有图谱之外的知识",
    "system:privacy=只记住用户在本会话中明确自述的名字，并可随时忘记",
];

const ENTITY_ENTITIES: &[(&str, Option<&str>)] = &[
    ("姚明", Some("人物")),
    ("叶莉", Some("人物")),
    ("火箭队", Some("组织")),
];
const ENTITY_ALIASES: &[(&str, &str)] = &[("小姚", "姚明")];
const ENTITY_ATTRIBUTES: &[(&str, &str, &str)] = &[
    ("姚明", "身高", "226"),
    ("姚明", "国籍", "中国"),
    ("叶莉", "身高", "190"),
];
const ENTITY_RELATIONS: &[(&str, &str, &str)] = &[("姚明", "效力于", "火箭队")];
const YAO_CLAIMS: &[&str] = &[
    "entity:identity=姚明",
    "entity:type=人物",
    "entity:alias=小姚",
    "entity:attribute:身高=身高=226",
    "entity:attribute:国籍=国籍=中国",
    "entity:relation:效力于=效力于=火箭队",
];

/// 种子语料：系统宪法、用户台账、实体卡、未知克制、跨会话作用域。
pub fn seed_corpus() -> &'static [IdentityGoldScenario] {
    static SEED: &[IdentityGoldScenario] = &[
        IdentityGoldScenario {
            name: "系统身份：宪法事实与能力边界",
            entities: &[],
            aliases: &[],
            attributes: &[],
            relations: &[],
            turns: &[
                IdentityGoldTurn {
                    input: "你是谁",
                    scope: IdentityScope::System,
                    status: Some(IdentityStatus::Known),
                    claims: SYSTEM_CLAIMS,
                    subject: None,
                },
                IdentityGoldTurn {
                    input: "你叫什么名字",
                    scope: IdentityScope::System,
                    status: Some(IdentityStatus::Known),
                    claims: SYSTEM_CLAIMS,
                    subject: None,
                },
                IdentityGoldTurn {
                    input: "你有意识吗",
                    scope: IdentityScope::System,
                    status: Some(IdentityStatus::Known),
                    claims: SYSTEM_CLAIMS,
                    subject: None,
                },
            ],
            stale_names: &[],
        },
        IdentityGoldScenario {
            name: "用户身份：显式自述、更正与回忆",
            entities: &[],
            aliases: &[],
            attributes: &[],
            relations: &[],
            turns: &[
                IdentityGoldTurn {
                    input: "我是谁",
                    scope: IdentityScope::User,
                    status: Some(IdentityStatus::Unknown),
                    claims: &[],
                    subject: None,
                },
                IdentityGoldTurn {
                    input: "我叫张三",
                    scope: IdentityScope::User,
                    status: Some(IdentityStatus::Known),
                    claims: &["user:name=张三"],
                    subject: None,
                },
                IdentityGoldTurn {
                    input: "我是谁",
                    scope: IdentityScope::User,
                    status: Some(IdentityStatus::Known),
                    claims: &["user:name=张三"],
                    subject: None,
                },
                IdentityGoldTurn {
                    input: "你记得我叫什么吗",
                    scope: IdentityScope::User,
                    status: Some(IdentityStatus::Known),
                    claims: &["user:name=张三"],
                    subject: None,
                },
                IdentityGoldTurn {
                    input: "不对，我叫李四",
                    scope: IdentityScope::User,
                    status: Some(IdentityStatus::Corrected),
                    claims: &["user:name=李四"],
                    subject: None,
                },
                IdentityGoldTurn {
                    input: "我是谁",
                    scope: IdentityScope::User,
                    status: Some(IdentityStatus::Corrected),
                    claims: &["user:name=李四"],
                    subject: None,
                },
            ],
            stale_names: &["张三"],
        },
        IdentityGoldScenario {
            name: "用户身份：遗忘后不再使用",
            entities: &[],
            aliases: &[],
            attributes: &[],
            relations: &[],
            turns: &[
                IdentityGoldTurn {
                    input: "我叫王五",
                    scope: IdentityScope::User,
                    status: Some(IdentityStatus::Known),
                    claims: &["user:name=王五"],
                    subject: None,
                },
                IdentityGoldTurn {
                    input: "忘掉我的名字",
                    scope: IdentityScope::User,
                    status: Some(IdentityStatus::Forgotten),
                    claims: &[],
                    subject: None,
                },
                IdentityGoldTurn {
                    input: "我是谁",
                    scope: IdentityScope::User,
                    status: Some(IdentityStatus::Forgotten),
                    claims: &[],
                    subject: None,
                },
            ],
            stale_names: &["王五"],
        },
        IdentityGoldScenario {
            name: "实体身份：图谱事实 + 别名注册表",
            entities: ENTITY_ENTITIES,
            aliases: ENTITY_ALIASES,
            attributes: ENTITY_ATTRIBUTES,
            relations: ENTITY_RELATIONS,
            turns: &[
                IdentityGoldTurn {
                    input: "姚明是谁",
                    scope: IdentityScope::Entity,
                    status: Some(IdentityStatus::Known),
                    claims: YAO_CLAIMS,
                    subject: Some("姚明"),
                },
                IdentityGoldTurn {
                    input: "小姚是谁",
                    scope: IdentityScope::Entity,
                    status: Some(IdentityStatus::Known),
                    claims: YAO_CLAIMS,
                    subject: Some("姚明"),
                },
                IdentityGoldTurn {
                    input: "叶莉是谁",
                    scope: IdentityScope::Entity,
                    status: Some(IdentityStatus::Known),
                    claims: &[
                        "entity:identity=叶莉",
                        "entity:type=人物",
                        "entity:attribute:身高=身高=190",
                    ],
                    subject: Some("叶莉"),
                },
                IdentityGoldTurn {
                    input: "姚明的身高是多少",
                    scope: IdentityScope::Other,
                    status: None,
                    claims: &[],
                    subject: None,
                },
            ],
            stale_names: &[],
        },
        IdentityGoldScenario {
            name: "未知实体：克制 abstain，不造身份卡",
            entities: ENTITY_ENTITIES,
            aliases: ENTITY_ALIASES,
            attributes: ENTITY_ATTRIBUTES,
            relations: ENTITY_RELATIONS,
            turns: &[
                IdentityGoldTurn {
                    input: "未名科技公司是谁",
                    scope: IdentityScope::Entity,
                    status: Some(IdentityStatus::Unknown),
                    claims: &[],
                    subject: Some("未名科技公司"),
                },
                IdentityGoldTurn {
                    input: "诸葛亮是谁",
                    scope: IdentityScope::Entity,
                    status: Some(IdentityStatus::Unknown),
                    claims: &[],
                    subject: Some("诸葛亮"),
                },
                IdentityGoldTurn {
                    input: "诸葛亮的身高是多少",
                    scope: IdentityScope::Other,
                    status: None,
                    claims: &[],
                    subject: None,
                },
            ],
            stale_names: &[],
        },
    ];
    SEED
}

/// 评测知识源：小图谱 + 别名注册表（与链接器同源）。
pub struct IdentityEvalKnowledge {
    entities: BTreeMap<String, Option<String>>,
    aliases: BTreeMap<String, String>,
    attributes: BTreeMap<(String, String), Vec<String>>,
    relations: BTreeMap<(String, String), Vec<String>>,
    linker: EntityLinker,
}

impl IdentityEvalKnowledge {
    pub fn from_scenario(scenario: &IdentityGoldScenario) -> Self {
        let mut knowledge = Self {
            entities: BTreeMap::new(),
            aliases: BTreeMap::new(),
            attributes: BTreeMap::new(),
            relations: BTreeMap::new(),
            linker: EntityLinker::new(),
        };
        for (entity, etype) in scenario.entities {
            knowledge
                .entities
                .insert((*entity).to_string(), etype.map(str::to_string));
            knowledge.linker.register_entity(entity, *etype);
        }
        for (alias, canonical) in scenario.aliases {
            knowledge
                .aliases
                .insert((*alias).to_string(), (*canonical).to_string());
            knowledge.linker.register_alias(alias, canonical);
        }
        for (entity, attribute, value) in scenario.attributes {
            knowledge
                .attributes
                .entry(((*entity).to_string(), (*attribute).to_string()))
                .or_default()
                .push((*value).to_string());
        }
        for (entity, predicate, object) in scenario.relations {
            knowledge
                .relations
                .entry(((*entity).to_string(), (*predicate).to_string()))
                .or_default()
                .push((*object).to_string());
        }
        knowledge
    }

    fn canonical(&self, surface: &str) -> Option<&str> {
        let surface = surface.trim();
        if self.entities.contains_key(surface) {
            return self.entities.get_key_value(surface).map(|(name, _)| name.as_str());
        }
        self.aliases
            .get(surface)
            .and_then(|canonical| self.entities.get_key_value(canonical))
            .map(|(name, _)| name.as_str())
    }

    fn out_predicates(&self, entity: &str) -> Vec<String> {
        let mut out: Vec<String> = self
            .attributes
            .keys()
            .filter(|(name, _)| name == entity)
            .map(|(_, predicate)| predicate.clone())
            .chain(
                self.relations
                    .keys()
                    .filter(|(name, _)| name == entity)
                    .map(|(_, predicate)| predicate.clone()),
            )
            .collect();
        out.sort();
        out.dedup();
        out
    }
}

impl SenseSource for IdentityEvalKnowledge {
    fn sense_card(&self, entity: &str) -> Option<SenseCard> {
        let canonical = self.canonical(entity)?;
        let etype = self.entities.get(canonical).cloned().flatten();
        Some(SenseCard {
            entity: canonical.to_string(),
            etype,
            out_predicates: self.out_predicates(canonical),
            in_predicates: Vec::new(),
            anchors: Vec::new(),
            degree: 0,
        })
    }
}

impl DialogueKnowledge for IdentityEvalKnowledge {
    fn entities_in(&self, text: &str) -> Vec<String> {
        let mut surfaces: Vec<(String, String)> = self
            .entities
            .keys()
            .map(|entity| (entity.clone(), entity.clone()))
            .chain(
                self.aliases
                    .iter()
                    .map(|(alias, canonical)| (alias.clone(), canonical.clone())),
            )
            .collect();
        surfaces.sort_by(|(left, _), (right, _)| {
            right
                .chars()
                .count()
                .cmp(&left.chars().count())
                .then(left.cmp(right))
        });
        let mut out = Vec::new();
        for (surface, canonical) in surfaces {
            if text.contains(&surface) && !out.contains(&canonical) {
                out.push(canonical);
            }
        }
        out
    }

    fn known_entity(&self, entity: &str) -> bool {
        self.canonical(entity).is_some()
    }

    fn answer(&self, _question: &str) -> Option<String> {
        None
    }

    fn known_attributes(&self, entity: &str) -> Vec<String> {
        self.out_predicates(entity)
    }

    fn attribute_values(&self, entity: &str, attribute: &str) -> Vec<String> {
        self.attributes
            .get(&(entity.to_string(), attribute.to_string()))
            .cloned()
            .unwrap_or_default()
    }

    fn entity_type(&self, entity: &str) -> Option<String> {
        self.canonical(entity)
            .and_then(|canonical| self.entities.get(canonical).cloned().flatten())
    }

    fn entity_attributes(&self, entity: &str) -> Vec<(String, String)> {
        let Some(canonical) = self.canonical(entity) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for ((name, predicate), values) in &self.attributes {
            if name == canonical {
                for value in values {
                    out.push((predicate.clone(), value.clone()));
                }
            }
        }
        out.sort();
        out
    }

    fn entity_relations(&self, entity: &str) -> Vec<(String, String)> {
        let Some(canonical) = self.canonical(entity) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for ((name, predicate), values) in &self.relations {
            if name == canonical {
                for value in values {
                    out.push((predicate.clone(), value.clone()));
                }
            }
        }
        out.sort();
        out
    }

    fn describe(&self, entity: &str) -> Option<String> {
        let canonical = self.canonical(entity)?;
        let etype = self.entities.get(canonical).cloned().flatten();
        Some(match etype {
            Some(etype) => format!("{}是图谱中的{}。", canonical, etype),
            None => format!("{}是图谱中的实体。", canonical),
        })
    }

    fn reference_view(&self) -> Option<ReferenceView<'_>> {
        Some(ReferenceView {
            linker: &self.linker,
            source: self,
        })
    }
}

/// 结构化评测报告。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IdentityEvalReport {
    pub corpus_digest: String,
    pub scenarios: u32,
    pub turns: u32,
    pub act_accuracy: f64,
    pub act_confusion: BTreeMap<String, BTreeMap<String, u32>>,
    pub claim_tp: u32,
    pub claim_fp: u32,
    pub claim_fn: u32,
    pub claim_precision: f64,
    pub claim_recall: f64,
    pub claim_f1: f64,
    pub status_accuracy: f64,
    pub abstain_expected: u32,
    pub abstain_correct: u32,
    pub abstain_rate: f64,
    pub overclaim_rate: f64,
    pub scope_leak_rate: f64,
    pub evidence_coverage: f64,
    pub missing_cards: u32,
    pub session_retention_rate: f64,
    pub cross_session_leak_rate: f64,
    pub stale_mention_rate: f64,
    pub forget_clear_rate: f64,
}

impl IdentityEvalReport {
    pub fn render_text(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "身份识别评测：{} 个场景 / {} 轮，语料摘要 {}。\n",
            self.scenarios, self.turns, self.corpus_digest
        ));
        out.push_str(&format!(
            "act：作用域准确率 {:.4}\n",
            self.act_accuracy
        ));
        for (gold, predicted) in &self.act_confusion {
            for (label, count) in predicted {
                if gold == label || *count == 0 {
                    continue;
                }
                out.push_str(&format!("  ! 混淆 {} → {} ×{}\n", gold, label, count));
            }
        }
        out.push_str(&format!(
            "content：claim P {:.4} / R {:.4} / F1 {:.4}（tp {} / fp {} / fn {}），状态准确率 {:.4}\n",
            self.claim_precision,
            self.claim_recall,
            self.claim_f1,
            self.claim_tp,
            self.claim_fp,
            self.claim_fn,
            self.status_accuracy
        ));
        out.push_str(&format!(
            "scope：abstain {}/{}（{:.4}），overclaim {:.4}，scope leak {:.4}，证据覆盖 {:.4}，缺失身份卡 {}\n",
            self.abstain_correct,
            self.abstain_expected,
            self.abstain_rate,
            self.overclaim_rate,
            self.scope_leak_rate,
            self.evidence_coverage,
            self.missing_cards
        ));
        out.push_str(&format!(
            "session：快照保留 {:.4}，跨会话泄漏 {:.4}，旧名残留 {:.4}，遗忘清空 {:.4}\n",
            self.session_retention_rate,
            self.cross_session_leak_rate,
            self.stale_mention_rate,
            self.forget_clear_rate
        ));
        out
    }

    pub fn flatten_metrics(&self) -> BTreeMap<String, f64> {
        let mut metrics = BTreeMap::new();
        metrics.insert("identity.act_accuracy".to_string(), self.act_accuracy);
        metrics.insert("identity.claim_precision".to_string(), self.claim_precision);
        metrics.insert("identity.claim_recall".to_string(), self.claim_recall);
        metrics.insert("identity.claim_f1".to_string(), self.claim_f1);
        metrics.insert("identity.status_accuracy".to_string(), self.status_accuracy);
        metrics.insert("identity.abstain_rate".to_string(), self.abstain_rate);
        metrics.insert("identity.overclaim_rate".to_string(), self.overclaim_rate);
        metrics.insert("identity.scope_leak_rate".to_string(), self.scope_leak_rate);
        metrics.insert(
            "identity.evidence_coverage".to_string(),
            self.evidence_coverage,
        );
        metrics.insert("identity.missing_cards".to_string(), self.missing_cards as f64);
        metrics.insert(
            "identity.session_retention_rate".to_string(),
            self.session_retention_rate,
        );
        metrics.insert(
            "identity.cross_session_leak_rate".to_string(),
            self.cross_session_leak_rate,
        );
        metrics.insert(
            "identity.stale_mention_rate".to_string(),
            self.stale_mention_rate,
        );
        metrics.insert(
            "identity.forget_clear_rate".to_string(),
            self.forget_clear_rate,
        );
        metrics
    }
}

/// 越大越差的指标。
pub const LOWER_IS_BETTER: [&str; 5] = [
    "identity.overclaim_rate",
    "identity.scope_leak_rate",
    "identity.missing_cards",
    "identity.cross_session_leak_rate",
    "identity.stale_mention_rate",
];

/// 可写入磁盘的评测基线。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IdentityEvalBaseline {
    pub version: u32,
    pub corpus_digest: String,
    pub metrics: BTreeMap<String, f64>,
}

pub const IDENTITY_BASELINE_VERSION: u32 = 1;

impl IdentityEvalBaseline {
    pub fn from_report(report: &IdentityEvalReport) -> Self {
        Self {
            version: IDENTITY_BASELINE_VERSION,
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
        let baseline: IdentityEvalBaseline = serde_json::from_str(&text)?;
        Ok(baseline)
    }
}

/// 单项指标变化。
#[derive(Debug, Clone, PartialEq)]
pub struct IdentityMetricDelta {
    pub metric: String,
    pub baseline: f64,
    pub current: f64,
}

/// 棘轮判定。
#[derive(Debug, Clone, PartialEq)]
pub struct IdentityRatchetVerdict {
    pub passed: bool,
    pub digest_matches: bool,
    pub regressions: Vec<IdentityMetricDelta>,
    pub gains: Vec<IdentityMetricDelta>,
}

impl IdentityRatchetVerdict {
    pub fn render_text(&self) -> String {
        if self.passed {
            return "身份评测棘轮检查通过：无指标低于基线。\n".to_string();
        }
        let mut out = String::from("身份评测棘轮检查未通过：\n");
        if !self.digest_matches {
            out.push_str("  - 语料摘要不匹配：金标语料已漂移，基线不可直接比较\n");
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
        if !self.gains.is_empty() {
            out.push_str(&format!("有 {} 项指标高于基线。\n", self.gains.len()));
        }
        out
    }
}

pub fn compare_baseline(
    baseline: &IdentityEvalBaseline,
    current: &IdentityEvalReport,
    epsilon: f64,
) -> IdentityRatchetVerdict {
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
            regressions.push(IdentityMetricDelta {
                metric: metric.clone(),
                baseline: *baseline_value,
                current: current_value,
            });
        } else if is_gain {
            gains.push(IdentityMetricDelta {
                metric: metric.clone(),
                baseline: *baseline_value,
                current: current_value,
            });
        }
    }
    IdentityRatchetVerdict {
        passed: digest_matches && regressions.is_empty(),
        digest_matches,
        regressions,
        gains,
    }
}

fn ratio(numerator: u32, denominator: u32) -> f64 {
    if denominator == 0 {
        1.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn observe_turn(
    manager: &mut DialogueManager,
    knowledge: &IdentityEvalKnowledge,
    turn: &IdentityGoldTurn,
) -> Option<IdentityCard> {
    let act = classify(turn.input);
    match act {
        crate::identity_dialogue::IdentityAct::AskEntity => {
            let _ = manager.plan_pure_dialogue(turn.input, knowledge);
            let _ = manager.plan_reply(turn.input, knowledge);
            manager.last_identity_card().cloned().or_else(|| {
                turn.subject
                    .map(|subject| entity_card(knowledge, subject))
            })
        }
        crate::identity_dialogue::IdentityAct::AskSystem
        | crate::identity_dialogue::IdentityAct::AskUser
        | crate::identity_dialogue::IdentityAct::Introduce { .. }
        | crate::identity_dialogue::IdentityAct::ForgetUser => {
            let _ = manager.plan_pure_dialogue(turn.input, knowledge);
            manager.last_identity_card().cloned()
        }
        crate::identity_dialogue::IdentityAct::Other => None,
    }
}

fn card_contains(card: &IdentityCard, surface: &str) -> bool {
    card.subject.contains(surface)
        || card.render().contains(surface)
        || card
            .claims
            .iter()
            .any(|claim| claim.value.contains(surface))
}

/// 跑完整种子评测。
pub fn evaluate_seed() -> IdentityEvalReport {
    evaluate(seed_corpus())
}

/// 跑给定场景集。结果只依赖语料与实现，不含时间/随机量。
pub fn evaluate(scenarios: &[IdentityGoldScenario]) -> IdentityEvalReport {
    let mut act_total = 0u32;
    let mut act_correct = 0u32;
    let mut act_confusion: BTreeMap<String, BTreeMap<String, u32>> = BTreeMap::new();
    let mut claim_tp = 0u32;
    let mut claim_fp = 0u32;
    let mut claim_fn = 0u32;
    let mut status_total = 0u32;
    let mut status_correct = 0u32;
    let mut abstain_expected = 0u32;
    let mut abstain_correct = 0u32;
    let mut overclaims = 0u32;
    let mut claim_total = 0u32;
    let mut scope_leaks = 0u32;
    let mut evidence_missing = 0u32;
    let mut missing_cards = 0u32;
    let mut unexpected_cards = 0u32;
    let mut retention_expected = 0u32;
    let mut retention_correct = 0u32;
    let mut cross_expected = 0u32;
    let mut cross_leaks = 0u32;
    let mut stale_expected = 0u32;
    let mut stale_hits = 0u32;
    let mut forget_expected = 0u32;
    let mut forget_correct = 0u32;
    let mut turns = 0u32;
    let mut digest_input = String::new();

    for scenario in scenarios {
        digest_input.push_str(scenario.name);
        digest_input.push('\n');
        for (entity, etype) in scenario.entities {
            digest_input.push_str(&format!("entity={:?}|{:?}\n", entity, etype));
        }
        for (alias, canonical) in scenario.aliases {
            digest_input.push_str(&format!("alias={}|{}\n", alias, canonical));
        }
        for (entity, key, value) in scenario.attributes {
            digest_input.push_str(&format!("attr={}|{}={}\n", entity, key, value));
        }
        for (entity, key, value) in scenario.relations {
            digest_input.push_str(&format!("rel={}|{}={}\n", entity, key, value));
        }
        for stale in scenario.stale_names {
            digest_input.push_str(&format!("stale={}\n", stale));
        }

        let knowledge = IdentityEvalKnowledge::from_scenario(scenario);
        let mut manager = DialogueManager::new();

        for turn in scenario.turns {
            turns += 1;
            digest_input.push_str(&format!(
                "turn={}|{:?}|{:?}|{:?}|{:?}\n",
                turn.input, turn.scope, turn.status, turn.claims, turn.subject
            ));
            let act = classify(turn.input);
            let predicted_scope = act.scope();
            act_total += 1;
            if predicted_scope == turn.scope {
                act_correct += 1;
            }
            *act_confusion
                .entry(turn.scope.label().to_string())
                .or_default()
                .entry(predicted_scope.label().to_string())
                .or_insert(0) += 1;

            let card = observe_turn(&mut manager, &knowledge, turn);
            if let Some(expected_status) = turn.status {
                status_total += 1;
                let actual_status = card.as_ref().map(|card| card.status);
                if actual_status == Some(expected_status) {
                    status_correct += 1;
                }
                let expected: BTreeSet<String> =
                    turn.claims.iter().map(|claim| claim.to_string()).collect();
                let predicted: BTreeSet<String> = card
                    .as_ref()
                    .map(|card| card.claim_signatures().into_iter().collect())
                    .unwrap_or_default();
                claim_tp += expected.intersection(&predicted).count() as u32;
                claim_fp += predicted.difference(&expected).count() as u32;
                claim_fn += expected.difference(&predicted).count() as u32;
                match &card {
                    Some(card) => {
                        claim_total += card.claims.len() as u32;
                        scope_leaks += card.scope_leaks() as u32;
                        evidence_missing += card.missing_evidence() as u32;
                    }
                    None => missing_cards += 1,
                }
                let abstain = matches!(
                    expected_status,
                    IdentityStatus::Unknown | IdentityStatus::Forgotten
                );
                if abstain {
                    abstain_expected += 1;
                    let actual_claims_empty = card
                        .as_ref()
                        .map(|card| card.claims.is_empty())
                        .unwrap_or(true);
                    if actual_status == Some(expected_status) && actual_claims_empty {
                        abstain_correct += 1;
                    }
                    if card
                        .as_ref()
                        .map(|card| !card.claims.is_empty())
                        .unwrap_or(false)
                        || matches!(
                            actual_status,
                            Some(IdentityStatus::Known) | Some(IdentityStatus::Corrected)
                        )
                    {
                        overclaims += 1;
                    }
                }
            } else if card.is_some() {
                unexpected_cards += 1;
            }
        }

        let final_ledger: &UserIdentityLedger = manager.identity_ledger();
        let final_card = user_card(final_ledger);
        if final_ledger.status() == IdentityStatus::Forgotten {
            forget_expected += 1;
            if final_ledger.current().is_none() && final_ledger.history().is_empty() {
                forget_correct += 1;
            }
        }

        if let Some(name) = final_ledger.current() {
            retention_expected += 1;
            let snapshot = manager.snapshot();
            let mut restored = DialogueManager::new();
            restored.restore(&snapshot);
            let retained = restored.identity_ledger().current() == Some(name);
            let _ = restored.plan_pure_dialogue("我是谁", &knowledge);
            let recalled = restored
                .last_identity_card()
                .map(|card| {
                    card.claims
                        .iter()
                        .any(|claim| claim.key == "user:name" && claim.value == name)
                })
                .unwrap_or(false);
            if retained && recalled {
                retention_correct += 1;
            }
        }

        cross_expected += 1;
        let mut fresh = DialogueManager::new();
        let _ = fresh.plan_pure_dialogue("我是谁", &knowledge);
        let fresh_card = fresh.last_identity_card().cloned();
        let leaked = fresh_card
            .as_ref()
            .map(|card| {
                card.status != IdentityStatus::Unknown
                    || !card.claims.is_empty()
                    || scenario
                        .stale_names
                        .iter()
                        .any(|stale| card_contains(card, stale))
            })
            .unwrap_or(false);
        if leaked {
            cross_leaks += 1;
        }

        for stale in scenario.stale_names {
            stale_expected += 1;
            if card_contains(&final_card, stale)
                || fresh_card
                    .as_ref()
                    .map(|card| card_contains(card, stale))
                    .unwrap_or(false)
            {
                stale_hits += 1;
            }
        }
        digest_input.push('\n');
    }

    let claim_precision = ratio(claim_tp, claim_tp + claim_fp);
    let claim_recall = ratio(claim_tp, claim_tp + claim_fn);
    let claim_f1 = if claim_precision + claim_recall == 0.0 {
        0.0
    } else {
        2.0 * claim_precision * claim_recall / (claim_precision + claim_recall)
    };
    let _ = unexpected_cards;

    IdentityEvalReport {
        corpus_digest: fnv1a_hex(digest_input.as_bytes()),
        scenarios: scenarios.len() as u32,
        turns,
        act_accuracy: ratio(act_correct, act_total),
        act_confusion,
        claim_tp,
        claim_fp,
        claim_fn,
        claim_precision,
        claim_recall,
        claim_f1,
        status_accuracy: ratio(status_correct, status_total),
        abstain_expected,
        abstain_correct,
        abstain_rate: ratio(abstain_correct, abstain_expected),
        overclaim_rate: ratio(overclaims, abstain_expected),
        scope_leak_rate: ratio(scope_leaks, claim_total),
        evidence_coverage: ratio(claim_total - evidence_missing, claim_total),
        missing_cards,
        session_retention_rate: ratio(retention_correct, retention_expected),
        cross_session_leak_rate: ratio(cross_leaks, cross_expected),
        stale_mention_rate: ratio(stale_hits, stale_expected),
        forget_clear_rate: ratio(forget_correct, forget_expected),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_identity_eval_is_clean_and_structured() {
        let report = evaluate_seed();
        assert_eq!(report.act_accuracy, 1.0, "{}", report.render_text());
        assert_eq!(report.claim_precision, 1.0, "{}", report.render_text());
        assert_eq!(report.claim_recall, 1.0, "{}", report.render_text());
        assert_eq!(report.claim_f1, 1.0, "{}", report.render_text());
        assert_eq!(report.status_accuracy, 1.0, "{}", report.render_text());
        assert_eq!(report.abstain_rate, 1.0, "{}", report.render_text());
        assert_eq!(report.overclaim_rate, 0.0, "{}", report.render_text());
        assert_eq!(report.scope_leak_rate, 0.0, "{}", report.render_text());
        assert_eq!(report.evidence_coverage, 1.0, "{}", report.render_text());
        assert_eq!(report.missing_cards, 0, "{}", report.render_text());
        assert_eq!(
            report.session_retention_rate, 1.0,
            "{}",
            report.render_text()
        );
        assert_eq!(
            report.cross_session_leak_rate, 0.0,
            "{}",
            report.render_text()
        );
        assert_eq!(report.stale_mention_rate, 0.0, "{}", report.render_text());
        assert_eq!(report.forget_clear_rate, 1.0, "{}", report.render_text());
        assert_eq!(
            report
                .act_confusion
                .get("system")
                .and_then(|row| row.get("system")),
            Some(&3)
        );
    }

    #[test]
    fn report_is_deterministic() {
        let first = evaluate_seed();
        let second = evaluate_seed();
        assert_eq!(first, second);
        assert_eq!(first.corpus_digest, second.corpus_digest);
    }

    #[test]
    fn ratchet_rejects_scope_leak_and_overclaim() {
        let report = evaluate_seed();
        let baseline = IdentityEvalBaseline::from_report(&report);
        let mut broken = report.clone();
        broken.scope_leak_rate = 0.5;
        broken.overclaim_rate = 0.5;
        let verdict = compare_baseline(&baseline, &broken, 1e-6);
        assert!(!verdict.passed);
        assert!(verdict
            .regressions
            .iter()
            .any(|delta| delta.metric == "identity.scope_leak_rate"));
        assert!(verdict
            .regressions
            .iter()
            .any(|delta| delta.metric == "identity.overclaim_rate"));
    }

    #[test]
    fn corpus_digest_changes_with_annotations() {
        let baseline = evaluate_seed();
        let mut scenarios = seed_corpus().to_vec();
        scenarios[0].name = "系统身份（改名）";
        let changed = evaluate(&scenarios);
        assert_ne!(baseline.corpus_digest, changed.corpus_digest);
    }
}

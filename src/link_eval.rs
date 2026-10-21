//! 会话级实体链接评测（Link Eval）：候选 → 消歧 → 绑定 → 复用。
//!
//! 与 `dialogue_eval` 的分工：后者把「文本里字面命中了哪些规范名」当作实体
//! 指标；本模块评估真正的链接决策——指称跨度、候选表、五态裁决
//! （Bound / Unique / Ambiguous / Description / Nil）、澄清选项、绑定记忆
//! 与跨轮一致性。语料是手写金标，指标是整数计数 + 确定性摘要，不引入
//! 训练、权重或新词表。
//!
//! 指标分三层：
//! - mention 级：候选召回、动作准确率、链接准确率、OOV 提名率；
//! - turn 级：澄清召回/选项覆盖、过度追问、静默错链；
//! - session 级：澄清回执的即时绑定率、同词面重复追问、话语链翻转。
//!
//! 棘轮约定：除了 `silent_mislink_rate` / `over_ask_rate` 等「越低越好」的
//! 指标，其余指标越高越好。

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::context_image::fnv1a_hex;
use crate::contextual_reference::{
    parse_clarify_act, BindingLedger, BindingScope, ClarifyAct, ContextualResolver, SenseCard,
    SenseSource, Verdict,
};
use crate::entity_linker::{oov_suspects, EntityLinker, LinkedMention};
use crate::entity_registry::fold_surface;

/// 金标动作：链接到规范实体 / 发起显式澄清 / 拒绝越权链接。
#[derive(Debug, Clone, Copy)]
pub enum GoldAction {
    /// 唯一裁决（或绑定命中）到给定实体。
    Link(&'static str),
    /// 升级为显式歧义，选项至少覆盖给定候选。
    Clarify(&'static [&'static str]),
    /// 零候选：不静默造实体、不链接到近似实体。
    Reject,
    /// 对上一轮澄清提问的回答；期望把该指称绑定到给定实体。
    Answer(&'static str),
}

/// 一轮金标：输入 + 指称跨度 + 期望动作 + 问题线索 + 话语链。
#[derive(Debug, Clone, Copy)]
pub struct LinkGoldTurn {
    pub input: &'static str,
    pub mention: &'static str,
    /// 指称在输入中的字符偏移（0 起）。
    pub start: usize,
    pub action: GoldAction,
    /// 问题谓词/类型线索，喂给 `link_lattice` 与 CRD 表决。
    pub hints: &'static [&'static str],
    /// 同一话语链的稳定 id；用于跨轮翻转统计。
    pub chain: Option<&'static str>,
    /// `Reject` 轮是否应被 OOV 提名（只提名、不落库）。
    pub oov_suspect: bool,
}

/// 一个金标场景：小图谱 + 顺序多轮脚本。
#[derive(Debug, Clone)]
pub struct LinkGoldScenario {
    pub name: &'static str,
    /// 规范实体 → 登记类型。
    pub entities: &'static [(&'static str, Option<&'static str>)],
    /// 别名 → 规范实体。
    pub aliases: &'static [(&'static str, &'static str)],
    /// 图谱三元组（主语, 谓词, 宾语）。
    pub triples: &'static [(&'static str, &'static str, &'static str)],
    pub turns: &'static [LinkGoldTurn],
}

const APPLE_ENTITIES: &[(&str, Option<&str>)] = &[
    ("苹果", Some("水果")),
    ("苹果公司", Some("组织")),
    ("乔布斯", Some("人物")),
    ("库克", Some("人物")),
];

const APPLE_ALIASES: &[(&str, &str)] = &[("苹果", "苹果公司")];

const APPLE_TRIPLES: &[(&str, &str, &str)] = &[
    ("苹果", "颜色", "红色"),
    ("苹果", "产地", "山东"),
    ("苹果公司", "创始人", "乔布斯"),
    ("苹果公司", "CEO", "库克"),
    ("乔布斯", "国籍", "美国"),
];

/// 种子语料：歧义澄清 → 绑定复用、谓词线索消歧、别名/模糊容错、OOV 克制。
pub fn seed_corpus() -> &'static [LinkGoldScenario] {
    static SEED: &[LinkGoldScenario] = &[
        LinkGoldScenario {
            name: "苹果双义：澄清后绑定复用",
            entities: APPLE_ENTITIES,
            aliases: APPLE_ALIASES,
            triples: APPLE_TRIPLES,
            turns: &[
                LinkGoldTurn {
                    input: "苹果是什么",
                    mention: "苹果",
                    start: 0,
                    action: GoldAction::Clarify(&["苹果", "苹果公司"]),
                    hints: &[],
                    chain: None,
                    oov_suspect: false,
                },
                LinkGoldTurn {
                    input: "2",
                    mention: "2",
                    start: 0,
                    action: GoldAction::Answer("苹果公司"),
                    hints: &[],
                    chain: None,
                    oov_suspect: false,
                },
                LinkGoldTurn {
                    input: "苹果的创始人是谁",
                    mention: "苹果",
                    start: 0,
                    action: GoldAction::Link("苹果公司"),
                    hints: &[],
                    chain: Some("苹果"),
                    oov_suspect: false,
                },
                LinkGoldTurn {
                    input: "苹果的CEO是谁",
                    mention: "苹果",
                    start: 0,
                    action: GoldAction::Link("苹果公司"),
                    hints: &[],
                    chain: Some("苹果"),
                    oov_suspect: false,
                },
                LinkGoldTurn {
                    input: "苹果公司是什么",
                    mention: "苹果公司",
                    start: 0,
                    action: GoldAction::Link("苹果公司"),
                    hints: &[],
                    chain: Some("苹果公司"),
                    oov_suspect: false,
                },
            ],
        },
        LinkGoldScenario {
            name: "苹果双义：谓词线索分义",
            entities: APPLE_ENTITIES,
            aliases: APPLE_ALIASES,
            triples: APPLE_TRIPLES,
            turns: &[
                LinkGoldTurn {
                    input: "苹果的颜色是什么",
                    mention: "苹果",
                    start: 0,
                    action: GoldAction::Link("苹果"),
                    hints: &["颜色"],
                    chain: Some("水果苹果"),
                    oov_suspect: false,
                },
                LinkGoldTurn {
                    input: "苹果的产地是什么",
                    mention: "苹果",
                    start: 0,
                    action: GoldAction::Link("苹果"),
                    hints: &["产地"],
                    chain: Some("水果苹果"),
                    oov_suspect: false,
                },
                LinkGoldTurn {
                    input: "苹果的创始人是谁",
                    mention: "苹果",
                    start: 0,
                    action: GoldAction::Link("苹果公司"),
                    hints: &[],
                    chain: Some("公司苹果"),
                    oov_suspect: false,
                },
            ],
        },
        LinkGoldScenario {
            name: "别名与错别字容错",
            entities: &[
                ("刘德华", Some("人物")),
                ("张艺谋", Some("人物")),
                ("无间道", Some("电影")),
            ],
            aliases: &[("华仔", "刘德华")],
            triples: &[
                ("刘德华", "出演", "无间道"),
                ("张艺谋", "执导", "无间道"),
            ],
            turns: &[
                LinkGoldTurn {
                    input: "华仔是谁",
                    mention: "华仔",
                    start: 0,
                    action: GoldAction::Link("刘德华"),
                    hints: &["人物"],
                    chain: Some("刘德华"),
                    oov_suspect: false,
                },
                LinkGoldTurn {
                    input: "刘德桦是谁",
                    mention: "刘德桦",
                    start: 0,
                    action: GoldAction::Link("刘德华"),
                    hints: &["人物"],
                    chain: Some("刘德华"),
                    oov_suspect: false,
                },
                LinkGoldTurn {
                    input: "无间道是谁执导的",
                    mention: "无间道",
                    start: 0,
                    action: GoldAction::Link("无间道"),
                    hints: &["执导"],
                    chain: Some("无间道"),
                    oov_suspect: false,
                },
            ],
        },
        LinkGoldScenario {
            name: "未登录词只提名不落库",
            entities: APPLE_ENTITIES,
            aliases: APPLE_ALIASES,
            triples: APPLE_TRIPLES,
            turns: &[
                LinkGoldTurn {
                    input: "未名科技公司是什么",
                    mention: "未名科技公司",
                    start: 0,
                    action: GoldAction::Reject,
                    hints: &[],
                    chain: None,
                    oov_suspect: true,
                },
                LinkGoldTurn {
                    input: "张三是什么",
                    mention: "张三",
                    start: 0,
                    action: GoldAction::Reject,
                    hints: &[],
                    chain: None,
                    oov_suspect: false,
                },
            ],
        },
    ];
    SEED
}

/// 由金标场景构建阅读侧义项来源：类型、谓词、取值、邻居锚点。
#[derive(Debug, Default, Clone)]
struct GoldSense {
    cards: HashMap<String, SenseCard>,
    values: HashMap<(String, String), Vec<String>>,
    values_in: HashMap<(String, String), Vec<String>>,
}

impl GoldSense {
    fn from_scenario(scenario: &LinkGoldScenario) -> Self {
        let mut sense = GoldSense::default();
        for (entity, etype) in scenario.entities {
            sense.cards.insert(
                (*entity).to_string(),
                SenseCard {
                    entity: (*entity).to_string(),
                    etype: etype.map(str::to_string),
                    ..SenseCard::default()
                },
            );
        }
        for (subject, predicate, object) in scenario.triples {
            for (entity, other, incoming) in [
                (*subject, *object, false),
                (*object, *subject, true),
            ] {
                let card = sense
                    .cards
                    .entry(entity.to_string())
                    .or_insert_with(|| SenseCard {
                        entity: entity.to_string(),
                        ..SenseCard::default()
                    });
                card.degree += 1;
                if !card.anchors.iter().any(|anchor| anchor == other) {
                    card.anchors.push(other.to_string());
                }
                let bucket = if incoming {
                    &mut card.in_predicates
                } else {
                    &mut card.out_predicates
                };
                if !bucket.iter().any(|known| known == predicate) {
                    bucket.push((*predicate).to_string());
                }
            }
            sense
                .values
                .entry(((*subject).to_string(), (*predicate).to_string()))
                .or_default()
                .push((*object).to_string());
            sense
                .values_in
                .entry(((*object).to_string(), (*predicate).to_string()))
                .or_default()
                .push((*subject).to_string());
        }
        for card in sense.cards.values_mut() {
            card.out_predicates.sort();
            card.in_predicates.sort();
            card.anchors.sort();
        }
        sense
    }
}

impl SenseSource for GoldSense {
    fn sense_card(&self, entity: &str) -> Option<SenseCard> {
        self.cards.get(entity).cloned()
    }

    fn predicate_values(&self, entity: &str, predicate: &str) -> Vec<String> {
        self.values
            .get(&(entity.to_string(), predicate.to_string()))
            .cloned()
            .unwrap_or_default()
    }

    fn predicate_values_in(&self, entity: &str, predicate: &str) -> Vec<String> {
        self.values_in
            .get(&(entity.to_string(), predicate.to_string()))
            .cloned()
            .unwrap_or_default()
    }
}

/// 评测用链接器：金标实体、别名与三元组全部登记，不做任何额外注入。
fn linker_for_scenario(scenario: &LinkGoldScenario) -> EntityLinker {
    let mut linker = EntityLinker::new();
    for (entity, etype) in scenario.entities {
        linker.register_entity(entity, *etype);
    }
    for (surface, predicate, object) in scenario.triples {
        linker.observe_triple(surface, predicate, object);
    }
    for (alias, canonical) in scenario.aliases {
        linker.register_alias(alias, canonical);
    }
    linker
}

/// 评测报告（可 JSON、可棘轮）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinkEvalReport {
    pub corpus_digest: String,
    pub scenarios: u32,
    pub turns: u32,
    /// 参与动作判定的轮次（不含澄清回答轮）。
    pub mentions: u32,
    /// 金标期望实体/选项出现在候选表中的比例。
    pub candidate_recall: f64,
    /// 动作（Link / Clarify / Reject）完全正确的比例。
    pub action_accuracy: f64,
    /// 期望链接的轮次中选对规范实体的比例。
    pub link_accuracy: f64,
    /// 期望澄清的轮次中实际发起澄清的比例。
    pub clarify_recall: f64,
    /// 澄清问题覆盖金标选项的比例。
    pub clarify_option_coverage: f64,
    /// 期望「不该链接」却链接了的比例（越低越好）。
    pub silent_mislink_rate: f64,
    /// 期望链接却发起澄清的比例（越低越好）。
    pub over_ask_rate: f64,
    /// Reject 轮中 OOV 被提名的比例。
    pub oov_recall: f64,
    /// 澄清回答被正确解析并写入绑定账本的比例。
    pub binding_accuracy: f64,
    /// 澄清后的首个相关链接是否命中绑定（Bound）。
    pub immediate_binding_rate: f64,
    /// 同一话语链上规范实体发生改变的次数（越低越好）。
    pub chain_flips: u32,
    /// 同一词面在同一场景内被重复追问的次数（越低越好）。
    pub repeated_asks: u32,
}

impl LinkEvalReport {
    pub fn render_text(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "会话实体链接评测：{} 个场景 / {} 轮（{} 个动作判定），语料摘要 {}。\n",
            self.scenarios, self.turns, self.mentions, self.corpus_digest
        ));
        out.push_str(&format!(
            "mention：candidate_recall {:.4}；action_accuracy {:.4}；link_accuracy {:.4}；oov_recall {:.4}\n",
            self.candidate_recall, self.action_accuracy, self.link_accuracy, self.oov_recall
        ));
        out.push_str(&format!(
            "turn：clarify_recall {:.4}；option_coverage {:.4}；silent_mislink_rate {:.4}；over_ask_rate {:.4}\n",
            self.clarify_recall,
            self.clarify_option_coverage,
            self.silent_mislink_rate,
            self.over_ask_rate
        ));
        out.push_str(&format!(
            "session：binding_accuracy {:.4}；immediate_binding_rate {:.4}；chain_flips {}；repeated_asks {}\n",
            self.binding_accuracy, self.immediate_binding_rate, self.chain_flips, self.repeated_asks
        ));
        out.trim_end().to_string()
    }

    pub fn flatten_metrics(&self) -> BTreeMap<String, f64> {
        let mut metrics = BTreeMap::new();
        metrics.insert("link.candidate_recall".to_string(), self.candidate_recall);
        metrics.insert("link.action_accuracy".to_string(), self.action_accuracy);
        metrics.insert("link.link_accuracy".to_string(), self.link_accuracy);
        metrics.insert("link.clarify_recall".to_string(), self.clarify_recall);
        metrics.insert(
            "link.clarify_option_coverage".to_string(),
            self.clarify_option_coverage,
        );
        metrics.insert(
            "link.silent_mislink_rate".to_string(),
            self.silent_mislink_rate,
        );
        metrics.insert("link.over_ask_rate".to_string(), self.over_ask_rate);
        metrics.insert("link.oov_recall".to_string(), self.oov_recall);
        metrics.insert("link.binding_accuracy".to_string(), self.binding_accuracy);
        metrics.insert(
            "link.immediate_binding_rate".to_string(),
            self.immediate_binding_rate,
        );
        metrics.insert("link.chain_flips".to_string(), self.chain_flips as f64);
        metrics.insert("link.repeated_asks".to_string(), self.repeated_asks as f64);
        metrics
    }
}

/// 越大越差的指标（棘轮反向判定）。
pub const LOWER_IS_BETTER: [&str; 4] = [
    "link.silent_mislink_rate",
    "link.over_ask_rate",
    "link.chain_flips",
    "link.repeated_asks",
];

/// 可写入磁盘的链接评测基线。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinkEvalBaseline {
    pub version: u32,
    pub corpus_digest: String,
    pub metrics: BTreeMap<String, f64>,
}

pub const LINK_BASELINE_VERSION: u32 = 1;

impl LinkEvalBaseline {
    pub fn from_report(report: &LinkEvalReport) -> Self {
        Self {
            version: LINK_BASELINE_VERSION,
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
        let baseline: LinkEvalBaseline = serde_json::from_str(&text)?;
        Ok(baseline)
    }
}

/// 单项指标变化。
#[derive(Debug, Clone, PartialEq)]
pub struct LinkMetricDelta {
    pub metric: String,
    pub baseline: f64,
    pub current: f64,
}

/// 棘轮判定。
#[derive(Debug, Clone, PartialEq)]
pub struct LinkRatchetVerdict {
    pub passed: bool,
    pub digest_matches: bool,
    pub regressions: Vec<LinkMetricDelta>,
    pub gains: Vec<LinkMetricDelta>,
}

impl LinkRatchetVerdict {
    pub fn render_text(&self) -> String {
        if self.passed {
            return "链接棘轮检查通过：无指标低于基线。\n".to_string();
        }
        let mut out = String::from("链接棘轮检查未通过：\n");
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
        out
    }
}

/// 与基线比较：语料摘要不同或任何指标退化超 ε，即不通过。
pub fn compare_baseline(
    baseline: &LinkEvalBaseline,
    current: &LinkEvalReport,
    epsilon: f64,
) -> LinkRatchetVerdict {
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
            regressions.push(LinkMetricDelta {
                metric: metric.clone(),
                baseline: *baseline_value,
                current: current_value,
            });
        } else if is_gain {
            gains.push(LinkMetricDelta {
                metric: metric.clone(),
                baseline: *baseline_value,
                current: current_value,
            });
        }
    }
    LinkRatchetVerdict {
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

fn lattice_for<'a>(mentions: &'a [LinkedMention], turn: &LinkGoldTurn) -> Option<&'a LinkedMention> {
    mentions.iter().find(|mention| {
        mention.start == turn.start
            && (mention.mention == turn.mention || mention.mention.contains(turn.mention))
    })
}

fn candidate_entities(mention: &LinkedMention) -> BTreeSet<String> {
    mention
        .candidates
        .iter()
        .map(|candidate| candidate.entity.clone())
        .collect()
}

fn question_options(ask: &crate::contextual_reference::ClarifyRequest) -> BTreeSet<String> {
    ask.question
        .options
        .iter()
        .map(|option| option.entity.clone())
        .collect()
}

/// 运行种子语料，输出确定性报告。
pub fn evaluate_seed() -> LinkEvalReport {
    evaluate(seed_corpus())
}

/// 在给定金标语料上逐场景、逐轮执行链接与澄清记忆。
pub fn evaluate(scenarios: &[LinkGoldScenario]) -> LinkEvalReport {
    let mut corpus_digest_input = String::new();
    let mut scenarios_count = 0u32;
    let mut turns = 0u32;
    let mut mentions = 0u32;

    let mut candidate_expected = 0u32;
    let mut candidate_hit = 0u32;
    let mut actions_correct = 0u32;
    let mut link_expected = 0u32;
    let mut links_correct = 0u32;
    let mut clarify_expected = 0u32;
    let mut clarify_asked = 0u32;
    let mut clarify_options_expected = 0u32;
    let mut clarify_options_hit = 0u32;
    let mut non_link_expected = 0u32;
    let mut silent_mislinks = 0u32;
    let mut over_asks = 0u32;
    let mut oov_expected = 0u32;
    let mut oov_nominated = 0u32;
    let mut answers_expected = 0u32;
    let mut answers_bound = 0u32;
    let mut immediate_expected = 0u32;
    let mut immediate_bound = 0u32;
    let mut chain_flips = 0u32;
    let mut repeated_asks = 0u32;

    for scenario in scenarios {
        scenarios_count += 1;
        corpus_digest_input.push_str(scenario.name);
        corpus_digest_input.push('\u{1}');
        let linker = linker_for_scenario(scenario);
        let sense = GoldSense::from_scenario(scenario);
        let resolver = ContextualResolver::new();
        let mut ledger = BindingLedger::new();
        let mut pending: Option<crate::contextual_reference::ClarifyRequest> = None;
        let mut asked_surfaces: BTreeSet<String> = BTreeSet::new();
        let mut chain_entities: HashMap<String, String> = HashMap::new();
        let mut demoted_binding: Option<(String, String)> = None;

        for turn in scenario.turns {
            turns += 1;
            corpus_digest_input.push_str(turn.input);
            corpus_digest_input.push('\u{2}');
            ledger.begin_turn();

            if let Some(ask) = pending.take() {
                if let GoldAction::Answer(expected) = turn.action {
                    answers_expected += 1;
                    let act = parse_clarify_act(turn.input, &ask.question);
                    let bound = match act {
                        Some(ClarifyAct::Choose { entity, .. })
                        | Some(ClarifyAct::Restate { entity }) => {
                            ledger.bind(
                                &ask.key,
                                &entity,
                                BindingScope::Topic,
                                &format!("评测澄清回答：{}", turn.input),
                            );
                            (entity == expected).then_some(entity)
                        }
                        Some(ClarifyAct::Correct { to, .. }) => {
                            ledger.correct(
                                &ask.key,
                                &to,
                                &format!("评测澄清纠正：{}", turn.input),
                            );
                            (to == expected).then_some(to)
                        }
                        Some(ClarifyAct::Unknown) | Some(ClarifyAct::Decline) | None => None,
                    };
                    if let Some(entity) = bound {
                        answers_bound += 1;
                        demoted_binding = Some((ask.mention.clone(), entity));
                    }
                    let _ = expected;
                    continue;
                }
            }

            mentions += 1;
            let hints: Vec<String> = turn.hints.iter().map(|hint| (*hint).to_string()).collect();
            let outcome = resolver.resolve_text(
                turn.input,
                &linker,
                &sense,
                &ledger,
                &hints,
                None,
            );
            let lattice = linker.link_lattice(turn.input, &hints);
            let receipt_list = linker.mention_receipts(turn.input, &hints, &[], None);
            let suspects = oov_suspects(turn.input, &receipt_list);

            let mention = lattice_for(&lattice, turn);
            let candidates = mention.map(candidate_entities).unwrap_or_default();
            let resolution = mention.and_then(|mention| {
                outcome
                    .resolutions
                    .iter()
                    .find(|resolution| resolution.sketch.start == mention.start)
            });

            let matched_mention = mention.is_some();
            let predicted = if let Some(ask) = &outcome.ask {
                Predicted::Clarify(question_options(ask))
            } else if let Some(chosen) = resolution.and_then(|resolution| resolution.chosen.clone())
            {
                Predicted::Link(chosen)
            } else {
                Predicted::Reject
            };

            let expected_candidates: BTreeSet<String> = match turn.action {
                GoldAction::Link(entity) => [entity.to_string()].into_iter().collect(),
                GoldAction::Clarify(options) => {
                    options.iter().map(|option| (*option).to_string()).collect()
                }
                GoldAction::Reject | GoldAction::Answer(_) => BTreeSet::new(),
            };
            if !expected_candidates.is_empty() {
                candidate_expected += expected_candidates.len() as u32;
                candidate_hit += expected_candidates
                    .iter()
                    .filter(|entity| candidates.contains(*entity))
                    .count() as u32;
            }

            match (turn.action, &predicted) {
                (GoldAction::Link(expected), Predicted::Link(actual)) if expected == actual => {
                    actions_correct += 1;
                }
                (GoldAction::Clarify(options), Predicted::Clarify(actual))
                    if options
                        .iter()
                        .all(|option| actual.iter().any(|candidate| candidate.as_str() == *option)) =>
                {
                    actions_correct += 1;
                }
                (GoldAction::Reject, Predicted::Reject) => actions_correct += 1,
                _ => {}
            }

            match turn.action {
                GoldAction::Link(expected) => {
                    link_expected += 1;
                    if matches!(predicted, Predicted::Link(ref actual) if actual.as_str() == expected) {
                        links_correct += 1;
                    }
                    if matches!(predicted, Predicted::Clarify(_)) {
                        over_asks += 1;
                    }
                    if let (Some(chain_id), Predicted::Link(actual)) =
                        (turn.chain, &predicted)
                    {
                        if let Some(previous) = chain_entities.get(chain_id) {
                            if previous != actual {
                                chain_flips += 1;
                            }
                        }
                        chain_entities.insert(chain_id.to_string(), actual.clone());
                    }
                    if let Some((surface, entity)) = &demoted_binding {
                        if turn.mention == surface || matched_mention && candidates.contains(entity)
                        {
                            immediate_expected += 1;
                            if matches!(resolution.map(|r| r.verdict), Some(Verdict::Bound))
                                && matches!(
                                    predicted,
                                    Predicted::Link(ref actual) if actual.as_str() == entity.as_str()
                                )
                            {
                                immediate_bound += 1;
                            }
                        }
                    }
                }
                GoldAction::Clarify(options) => {
                    clarify_expected += 1;
                    clarify_options_expected += options.len() as u32;
                    if let Some(ask) = &outcome.ask {
                        clarify_asked += 1;
                        let actual = question_options(ask);
                        clarify_options_hit += options
                            .iter()
                            .filter(|option| {
                                actual.iter().any(|candidate| candidate.as_str() == **option)
                            })
                            .count() as u32;
                        let key = fold_surface(&ask.mention);
                        if !asked_surfaces.insert(key) {
                            repeated_asks += 1;
                        }
                    }
                    if matches!(predicted, Predicted::Link(_)) {
                        silent_mislinks += 1;
                    }
                }
                GoldAction::Reject => {
                    non_link_expected += 1;
                    if matches!(predicted, Predicted::Link(_)) {
                        silent_mislinks += 1;
                    }
                    if turn.oov_suspect {
                        oov_expected += 1;
                        if suspects
                            .iter()
                            .any(|suspect| suspect.text == turn.mention)
                        {
                            oov_nominated += 1;
                        }
                    }
                }
                GoldAction::Answer(_) => {}
            }

            if let Some(ask) = outcome.ask {
                pending = Some(ask);
            }
            if let Some((_, entity)) = &demoted_binding {
                if matches!(predicted, Predicted::Link(ref actual) if actual.as_str() == entity.as_str())
                    && matches!(resolution.map(|r| r.verdict), Some(Verdict::Bound))
                {
                    demoted_binding = None;
                }
            }
        }
        corpus_digest_input.push('\n');
    }

    LinkEvalReport {
        corpus_digest: fnv1a_hex(corpus_digest_input.as_bytes()),
        scenarios: scenarios_count,
        turns,
        mentions,
        candidate_recall: ratio(candidate_hit, candidate_expected),
        action_accuracy: ratio(actions_correct, mentions),
        link_accuracy: ratio(links_correct, link_expected),
        clarify_recall: ratio(clarify_asked, clarify_expected),
        clarify_option_coverage: ratio(clarify_options_hit, clarify_options_expected),
        silent_mislink_rate: ratio(silent_mislinks, non_link_expected),
        over_ask_rate: ratio(over_asks, link_expected),
        oov_recall: ratio(oov_nominated, oov_expected),
        binding_accuracy: ratio(answers_bound, answers_expected),
        immediate_binding_rate: ratio(immediate_bound, immediate_expected),
        chain_flips,
        repeated_asks,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Predicted {
    Link(String),
    Clarify(BTreeSet<String>),
    Reject,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_links_are_correct_and_auditable() {
        let report = evaluate_seed();
        assert_eq!(report.candidate_recall, 1.0, "{}", report.render_text());
        assert_eq!(report.action_accuracy, 1.0, "{}", report.render_text());
        assert_eq!(report.link_accuracy, 1.0, "{}", report.render_text());
        assert_eq!(report.clarify_recall, 1.0, "{}", report.render_text());
        assert_eq!(
            report.clarify_option_coverage, 1.0,
            "{}",
            report.render_text()
        );
        assert_eq!(report.silent_mislink_rate, 0.0, "{}", report.render_text());
        assert_eq!(report.over_ask_rate, 0.0, "{}", report.render_text());
        assert_eq!(report.oov_recall, 1.0, "{}", report.render_text());
        assert_eq!(report.binding_accuracy, 1.0, "{}", report.render_text());
        assert_eq!(
            report.immediate_binding_rate, 1.0,
            "{}",
            report.render_text()
        );
        assert_eq!(report.chain_flips, 0, "{}", report.render_text());
        assert_eq!(report.repeated_asks, 0, "{}", report.render_text());
    }

    #[test]
    fn report_is_deterministic() {
        let first = evaluate_seed();
        let second = evaluate_seed();
        assert_eq!(first, second);
        assert_eq!(first.corpus_digest, second.corpus_digest);
    }

    #[test]
    fn ratchet_rejects_lower_is_better_regression() {
        let report = evaluate_seed();
        let mut baseline = LinkEvalBaseline::from_report(&report);
        baseline
            .metrics
            .insert("link.silent_mislink_rate".to_string(), 0.0);
        let mut broken = report.clone();
        broken.silent_mislink_rate = 0.5;
        let verdict = compare_baseline(&baseline, &broken, 1e-6);
        assert!(!verdict.passed);
        assert!(verdict
            .regressions
            .iter()
            .any(|delta| delta.metric == "link.silent_mislink_rate"));
    }

    #[test]
    fn corpus_digest_changes_with_annotations() {
        let baseline = evaluate_seed();
        let mut scenarios = seed_corpus().to_vec();
        scenarios[0].name = "苹果双义（改名）";
        let changed = evaluate(&scenarios);
        assert_ne!(baseline.corpus_digest, changed.corpus_digest);
    }
}

//! 构念召回（Constructed Recall）：把「应该能答多少句」构造成可复现的分母。
//!
//! 在线时系统知道自己答了多少句，却不知道**本该**答多少句：分子可得，分母
//! 不可得。本模块不依赖人工标注，而是从配置里**声称可答**的路由规则出发，
//! 为每条规则构造一组表面变体：
//!
//! - `canonical`：规则字面本身（seen）；
//! - `synonym`：用 `[synonyms.mappings]` 的反向变体替换词面（unseen）；
//! - `entity`：换一个同词性/同排序的实体值（unseen）；
//! - `suffix`：加一个语气尾缀「吗」（unseen）。
//!
//! 分母 = 去重后的构念问句数；分子 = 路由到预期 `query_type` 的问句数。
//! `seen` 与 `unseen` 分开报告，两者之差就是**表面泛化缺口**的主估计。
//!
//! 诚实的边界：这不是开放式召回。分母只覆盖「配置已经声称会答的结构」，
//! 不覆盖尚未写进配置的句式；后者要靠 [`crate::frontier`] 的在线缺口聚合
//! 去发现。因此本模块回答的是「已有承诺兑现得怎样」，而不是「世界被覆盖了
//! 多少」——把这两件事混成一个数，才是真正的分母自欺。
//!
//! 指标必须与负例/精度一起看：本模块只测召回，不奖励把字面改宽；路由到
//! 错误意图会记为 `wrong_route`，无命中记为 `unanswered`，两者都拉低召回。

use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};

use crate::context_image::fnv1a_hex;
use crate::frontier::{self, GapCount};
use crate::nlp::{normalize_words, preprocess};
use crate::pattern_ir::{Elem, MatchFailure, PatternContext, PatternEntry, PatternLibrary};
use crate::SystemConfig;

pub const RECALL_DIALECT: &str = "knowprolog-recall/1";
pub const RECALL_BASELINE_VERSION: u32 = 1;
/// 默认近似命中阈值，与前沿报告保持同一口径。
pub const DEFAULT_NEAR_MISS: f64 = frontier::NEAR_MISS_COVERAGE;
/// 报告里最多保留多少条未命中样例。
pub const SHOWN_MISSES: usize = 40;

/// 构念问句的来源变体。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaseVariant {
    /// 规则字面本身。
    Canonical,
    /// 同义词表里的另一词面。
    Synonym,
    /// 同词性/同排序的另一个实体值。
    Entity,
    /// 追加语气尾缀。
    Suffix,
}

impl CaseVariant {
    pub fn label(self) -> &'static str {
        match self {
            CaseVariant::Canonical => "字面",
            CaseVariant::Synonym => "同义词",
            CaseVariant::Entity => "实体替换",
            CaseVariant::Suffix => "语气尾缀",
        }
    }
}

/// 分母里的一条构念问句。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecallCase {
    pub question: String,
    pub expected_query_type: String,
    pub rule_name: String,
    pub expected_pattern: String,
    pub priority: i32,
    pub variant: CaseVariant,
    /// 是否规则字面本身；`false` 即留出表面。
    pub seen: bool,
}

/// 路由来源：主路由、NLU 回退、或完全未命中。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RouteKind {
    Primary,
    Fallback,
    Unanswered,
}

impl RouteKind {
    pub fn label(self) -> &'static str {
        match self {
            RouteKind::Primary => "主路由",
            RouteKind::Fallback => "回退",
            RouteKind::Unanswered => "未命中",
        }
    }
}

/// 一条未命中的构念问句及其最近前沿。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecallMiss {
    pub question: String,
    pub expected_query_type: String,
    pub actual_query_type: Option<String>,
    pub route: RouteKind,
    pub variant: CaseVariant,
    pub coverage: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure: Option<MatchFailure>,
}

/// 一个切片的召回。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecallSlice {
    pub cases: usize,
    pub correct: usize,
    pub recall: f64,
}

impl RecallSlice {
    fn new(cases: usize, correct: usize) -> Self {
        Self {
            cases,
            correct,
            recall: if cases == 0 {
                0.0
            } else {
                correct as f64 / cases as f64
            },
        }
    }
}

/// 单个路由类型的召回。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RouteRecall {
    pub query_type: String,
    pub cases: usize,
    pub correct: usize,
    pub recall: f64,
    pub seen_cases: usize,
    pub seen_correct: usize,
    pub unseen_cases: usize,
    pub unseen_correct: usize,
}

/// 构念召回报告。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecallReport {
    pub dialect: String,
    /// 构念分母的确定性摘要；基线只与同分母比较。
    pub corpus_digest: String,
    pub near_miss_threshold: f64,
    /// 参与构念的配置路由规则数。
    pub claims: usize,
    /// 去重后的分母。
    pub cases: usize,
    /// 路由到预期 query_type（主路由或回退）。
    pub correct: usize,
    /// 有路由但路由到错误 query_type。
    pub wrong_route: usize,
    /// 完全无路由。
    pub unanswered: usize,
    /// 主路由直接命中预期类型。
    pub primary_correct: usize,
    /// 回退命中预期类型。
    pub fallback_correct: usize,
    /// 同一问句被多条规则声称、且预期类型不同的情况（已按优先级去重）。
    pub ambiguous: usize,
    pub seen: RecallSlice,
    pub unseen: RecallSlice,
    /// seen_recall - unseen_recall；正数表示表面泛化有损失。
    pub generalization_gap: f64,
    pub by_route: Vec<RouteRecall>,
    /// 未命中问句的最近缺口，按频次排序。
    pub gaps: Vec<GapCount>,
    pub misses: Vec<RecallMiss>,
}

impl RecallReport {
    pub fn recall(&self) -> f64 {
        if self.cases == 0 {
            0.0
        } else {
            self.correct as f64 / self.cases as f64
        }
    }

    pub fn answered(&self) -> usize {
        self.correct + self.wrong_route
    }

    pub fn primary_recall(&self) -> f64 {
        if self.cases == 0 {
            0.0
        } else {
            self.primary_correct as f64 / self.cases as f64
        }
    }

    pub fn render_text(&self) -> String {
        let mut output = String::new();
        output.push_str(&format!("构念召回 {}\n", self.dialect));
        output.push_str(&format!(
            "分母：{} 条问句（来自 {} 条配置规则；歧义去重 {} 条）\n",
            self.cases, self.claims, self.ambiguous
        ));
        output.push_str(&format!(
            "命中：{}（召回 {:.1}%）；错误路由：{}；无路由：{}\n",
            self.correct,
            self.recall() * 100.0,
            self.wrong_route,
            self.unanswered
        ));
        output.push_str(&format!(
            "主路由命中：{}（{:.1}%）；回退补救：{}\n",
            self.primary_correct,
            self.primary_recall() * 100.0,
            self.fallback_correct
        ));
        output.push_str(&format!(
            "seen：{}/{}（{:.1}%）；unseen：{}/{}（{:.1}%）；泛化缺口：{:+.1} 个百分点\n",
            self.seen.correct,
            self.seen.cases,
            self.seen.recall * 100.0,
            self.unseen.correct,
            self.unseen.cases,
            self.unseen.recall * 100.0,
            self.generalization_gap * 100.0
        ));
        if !self.by_route.is_empty() {
            output.push_str("按路由：\n");
            for route in &self.by_route {
                output.push_str(&format!(
                    "  {}：{}/{}（{:.1}%，seen {}/{}，unseen {}/{}）\n",
                    route.query_type,
                    route.correct,
                    route.cases,
                    route.recall * 100.0,
                    route.seen_correct,
                    route.seen_cases,
                    route.unseen_correct,
                    route.unseen_cases
                ));
            }
        }
        if self.gaps.is_empty() {
            output.push_str("未命中缺口：无\n");
        } else {
            output.push_str("未命中缺口：\n");
            for gap in self.gaps.iter().take(frontier::SHOWN_GAPS) {
                output.push_str(&format!(
                    "  {}× {}（{}）\n",
                    gap.count,
                    gap.label(),
                    gap.kind.label()
                ));
            }
        }
        if !self.misses.is_empty() {
            output.push_str(&format!("未命中样例（最多 {} 条）：\n", SHOWN_MISSES));
            for miss in &self.misses {
                output.push_str(&format!(
                    "  · {} → 期望 {}，实得 {}（{}，覆盖 {:.0}%）\n",
                    miss.question,
                    miss.expected_query_type,
                    miss.actual_query_type.as_deref().unwrap_or("<无>"),
                    miss.route.label(),
                    miss.coverage * 100.0
                ));
            }
        }
        output
    }
}

/// 从配置构念并评估召回（使用配置编译出的模式库）。
pub fn evaluate_config(config: &SystemConfig, near_miss_threshold: f64) -> RecallReport {
    let library = PatternLibrary::from_config(config);
    evaluate(config, &library, near_miss_threshold)
}

/// 用给定模式库评估构念召回。
pub fn evaluate(
    config: &SystemConfig,
    library: &PatternLibrary,
    near_miss_threshold: f64,
) -> RecallReport {
    let synonyms = config
        .synonyms
        .as_ref()
        .map(|synonyms| synonyms.mappings.clone())
        .unwrap_or_default();
    let (cases, ambiguous) = construct_cases(library, &synonyms);
    let claims = library.entries().iter().filter(|entry| entry.route).count();
    let digest = corpus_digest(&cases);
    let context = PatternContext::default();

    let mut correct = 0usize;
    let mut wrong_route = 0usize;
    let mut unanswered = 0usize;
    let mut primary_correct = 0usize;
    let mut fallback_correct = 0usize;
    let mut seen_cases = 0usize;
    let mut seen_correct = 0usize;
    let mut unseen_cases = 0usize;
    let mut unseen_correct = 0usize;
    let mut routes: BTreeMap<String, RouteRecall> = BTreeMap::new();
    let mut gaps: BTreeMap<(String, String, crate::pattern_ir::MatchFailureKind), GapCount> =
        BTreeMap::new();
    let mut misses = Vec::new();

    for case in &cases {
        if case.seen {
            seen_cases += 1;
        } else {
            unseen_cases += 1;
        }
        let tagged = preprocess(&case.question, &config.nlp);
        let tokens = normalize_words(tagged, &synonyms);
        let primary = library.match_route_with_context(&tokens, &context);
        let fallback = if primary.is_none() {
            library.match_fallback_with_context(&tokens, &context)
        } else {
            None
        };
        let (route_kind, actual_query_type) = match (&primary, &fallback) {
            (Some(matched), _) => (RouteKind::Primary, Some(matched.entry.query_type.clone())),
            (None, Some(matched)) => (RouteKind::Fallback, Some(matched.entry.query_type.clone())),
            (None, None) => (RouteKind::Unanswered, None),
        };
        let is_correct = actual_query_type.as_deref() == Some(case.expected_query_type.as_str());

        let route_entry = routes
            .entry(case.expected_query_type.clone())
            .or_insert_with(|| RouteRecall {
                query_type: case.expected_query_type.clone(),
                cases: 0,
                correct: 0,
                recall: 0.0,
                seen_cases: 0,
                seen_correct: 0,
                unseen_cases: 0,
                unseen_correct: 0,
            });
        route_entry.cases += 1;
        if case.seen {
            route_entry.seen_cases += 1;
        } else {
            route_entry.unseen_cases += 1;
        }

        if is_correct {
            correct += 1;
            route_entry.correct += 1;
            if case.seen {
                seen_correct += 1;
                route_entry.seen_correct += 1;
            } else {
                unseen_correct += 1;
                route_entry.unseen_correct += 1;
            }
            match route_kind {
                RouteKind::Primary => primary_correct += 1,
                RouteKind::Fallback => fallback_correct += 1,
                RouteKind::Unanswered => {}
            }
        } else {
            if route_kind == RouteKind::Unanswered {
                unanswered += 1;
            } else {
                wrong_route += 1;
            }
            let frontier_report =
                frontier::analyze_tokens(&case.question, &tokens, library, &context);
            let failure = frontier_report
                .closest_blocked()
                .and_then(|row| row.failure.clone());
            let coverage = failure.as_ref().map(MatchFailure::coverage).unwrap_or(0.0);
            if let Some(failure) = &failure {
                let key = (
                    failure.expected.clone(),
                    failure.actual.clone().unwrap_or_default(),
                    failure.kind,
                );
                let gap = gaps.entry(key).or_insert_with(|| GapCount {
                    kind: failure.kind,
                    expected: failure.expected.clone(),
                    actual: failure.actual.clone().unwrap_or_default(),
                    count: 0,
                    examples: Vec::new(),
                });
                gap.count += 1;
                if gap.examples.len() < frontier::GAP_EXAMPLES
                    && !gap.examples.iter().any(|item| item == &case.question)
                {
                    gap.examples.push(case.question.clone());
                }
            }
            if misses.len() < SHOWN_MISSES {
                misses.push(RecallMiss {
                    question: case.question.clone(),
                    expected_query_type: case.expected_query_type.clone(),
                    actual_query_type,
                    route: route_kind,
                    variant: case.variant,
                    coverage,
                    failure,
                });
            }
        }
    }

    let mut by_route: Vec<RouteRecall> = routes.into_values().collect();
    for route in &mut by_route {
        route.recall = if route.cases == 0 {
            0.0
        } else {
            route.correct as f64 / route.cases as f64
        };
    }
    by_route.sort_by(|left, right| {
        left.recall
            .partial_cmp(&right.recall)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| right.cases.cmp(&left.cases))
            .then_with(|| left.query_type.cmp(&right.query_type))
    });

    let mut gap_list: Vec<GapCount> = gaps.into_values().collect();
    gap_list.sort_by(|left, right| {
        right
            .count
            .cmp(&left.count)
            .then_with(|| left.expected.cmp(&right.expected))
            .then_with(|| left.actual.cmp(&right.actual))
    });

    let seen = RecallSlice::new(seen_cases, seen_correct);
    let unseen = RecallSlice::new(unseen_cases, unseen_correct);
    RecallReport {
        dialect: RECALL_DIALECT.to_string(),
        corpus_digest: digest,
        near_miss_threshold,
        claims,
        cases: cases.len(),
        correct,
        wrong_route,
        unanswered,
        primary_correct,
        fallback_correct,
        ambiguous,
        generalization_gap: seen.recall - unseen.recall,
        seen,
        unseen,
        by_route,
        gaps: gap_list,
        misses,
    }
}

/// 构造分母：规则字面 + 同义词/实体/尾缀变体，按问句去重。
pub fn construct_cases(
    library: &PatternLibrary,
    synonyms: &HashMap<String, String>,
) -> (Vec<RecallCase>, usize) {
    let mut canonical: BTreeMap<String, RecallCase> = BTreeMap::new();
    let mut ambiguous = 0usize;

    let mut synonym_pairs: Vec<(&String, &String)> = synonyms.iter().collect();
    synonym_pairs.sort_by(
        |(left_variant, left_canonical), (right_variant, right_canonical)| {
            left_canonical
                .cmp(right_canonical)
                .then_with(|| left_variant.cmp(right_variant))
        },
    );

    for entry in library.entries() {
        if !entry.route || entry.elems.is_empty() {
            continue;
        }
        let question = render_question(entry, None, None);
        if question.trim().is_empty() {
            continue;
        }
        insert_case(
            &mut canonical,
            &mut ambiguous,
            RecallCase {
                question: question.clone(),
                expected_query_type: entry.query_type.clone(),
                rule_name: entry.name.clone(),
                expected_pattern: entry.render(),
                priority: entry.priority,
                variant: CaseVariant::Canonical,
                seen: true,
            },
        );

        // 同义词：规则字面若是某个映射的规范形，反向变体就是留出表面。
        for (index, elem) in entry.elems.iter().enumerate() {
            let Elem::Word { surface } = elem else {
                continue;
            };
            for (variant, canonical_surface) in &synonym_pairs {
                if canonical_surface.as_str() != surface {
                    continue;
                }
                let question = render_question(entry, Some((index, variant.as_str())), None);
                insert_case(
                    &mut canonical,
                    &mut ambiguous,
                    RecallCase {
                        question,
                        expected_query_type: entry.query_type.clone(),
                        rule_name: entry.name.clone(),
                        expected_pattern: entry.render(),
                        priority: entry.priority,
                        variant: CaseVariant::Synonym,
                        seen: false,
                    },
                );
            }
        }

        // 实体替换：同词性/同排序的另一个值，检验槽位而不是字面。
        for (index, elem) in entry.elems.iter().enumerate() {
            let Some(alternate) = alternate_sample(elem) else {
                continue;
            };
            let question = render_question(entry, None, Some((index, alternate.as_str())));
            insert_case(
                &mut canonical,
                &mut ambiguous,
                RecallCase {
                    question,
                    expected_query_type: entry.query_type.clone(),
                    rule_name: entry.name.clone(),
                    expected_pattern: entry.render(),
                    priority: entry.priority,
                    variant: CaseVariant::Entity,
                    seen: false,
                },
            );
        }

        // 语气尾缀：不应改变前缀路由；变了就是匹配器/预筛出了问题。
        if !question.ends_with(['?', '？', '吗', '呢']) {
            insert_case(
                &mut canonical,
                &mut ambiguous,
                RecallCase {
                    question: format!("{}吗", question),
                    expected_query_type: entry.query_type.clone(),
                    rule_name: entry.name.clone(),
                    expected_pattern: entry.render(),
                    priority: entry.priority,
                    variant: CaseVariant::Suffix,
                    seen: false,
                },
            );
        }
    }

    (canonical.into_values().collect(), ambiguous)
}

fn insert_case(cases: &mut BTreeMap<String, RecallCase>, ambiguous: &mut usize, case: RecallCase) {
    match cases.get_mut(&case.question) {
        None => {
            cases.insert(case.question.clone(), case);
        }
        Some(existing) => {
            if existing.expected_query_type != case.expected_query_type {
                *ambiguous += 1;
                if case.priority > existing.priority {
                    cases.insert(case.question.clone(), case);
                }
            } else if case.priority > existing.priority {
                // 同一预期类型：保留优先级更高者，但分母只算一次。
                cases.insert(case.question.clone(), case);
            }
        }
    }
}

fn render_question(
    entry: &PatternEntry,
    word_override: Option<(usize, &str)>,
    elem_override: Option<(usize, &str)>,
) -> String {
    let mut output = String::new();
    for (index, elem) in entry.elems.iter().enumerate() {
        if let Some((override_index, surface)) = word_override {
            if override_index == index {
                output.push_str(surface);
                continue;
            }
        }
        if let Some((override_index, value)) = elem_override {
            if override_index == index {
                output.push_str(value);
                continue;
            }
        }
        output.push_str(&sample_for_elem(elem));
    }
    output
}

fn sample_for_elem(elem: &Elem) -> String {
    match elem {
        Elem::Word { surface } => surface.clone(),
        Elem::Pos { pos } => sample_for_pos(pos).to_string(),
        Elem::Slot { sort, .. } => sample_for_sort(sort).to_string(),
        Elem::Any => "什么".to_string(),
        Elem::Optional(inner) => sample_for_elem(inner),
    }
}

fn alternate_sample(elem: &Elem) -> Option<String> {
    match elem {
        Elem::Pos { pos } => alternate_for_pos(pos).map(str::to_string),
        Elem::Slot { sort, .. } => alternate_for_sort(sort).map(str::to_string),
        _ => None,
    }
}

fn sample_for_pos(pos: &str) -> &'static str {
    match pos {
        "nr" => "张三",
        "ns" => "北京",
        "n" | "nz" => "苹果",
        "vn" => "研究",
        "m" => "3",
        "q" => "个",
        "v" => "是",
        "u" => "的",
        "f" => "上",
        "a" => "好",
        "d" => "很",
        "t" => "今天",
        "r" => "哪里",
        _ => "什么",
    }
}

fn alternate_for_pos(pos: &str) -> Option<&'static str> {
    match pos {
        "nr" => Some("李四"),
        "ns" => Some("上海"),
        "n" | "nz" => Some("书"),
        "vn" => Some("学习"),
        "m" => Some("5"),
        // 用配置词表稳定标注为 v/r 的词；否则测到的是标注缺口，
        // 会把「表面变体」指标和「词性标注」指标混在一起。
        "v" => Some("是"),
        "t" => Some("昨天"),
        "r" => Some("谁"),
        // 功能词/量词不换：换掉就不再是同一句法位置，会污染表面泛化估计。
        _ => None,
    }
}

fn sample_for_sort(sort: &str) -> &'static str {
    match sort {
        "城市" | "地点" => "北京",
        "日期" | "时间" => "今天",
        "人名" => "张三",
        "数量" => "3",
        "实体" => "苹果",
        "属性" => "颜色",
        _ => "什么",
    }
}

fn alternate_for_sort(sort: &str) -> Option<&'static str> {
    match sort {
        "城市" | "地点" => Some("上海"),
        "日期" | "时间" => Some("昨天"),
        "人名" => Some("李四"),
        "数量" => Some("5"),
        "实体" => Some("书"),
        "属性" => Some("大小"),
        _ => None,
    }
}

fn corpus_digest(cases: &[RecallCase]) -> String {
    let mut text = String::new();
    for case in cases {
        text.push_str(&case.question);
        text.push('\u{0}');
        text.push_str(&case.expected_query_type);
        text.push('\u{0}');
        text.push_str(case.variant.label());
        text.push('\n');
    }
    fnv1a_hex(text.as_bytes())
}

// ---------------------------------------------------------------------------
// 基线棘轮
// ---------------------------------------------------------------------------

/// 可写入磁盘的构念召回基线。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecallBaseline {
    pub version: u32,
    pub corpus_digest: String,
    pub metrics: BTreeMap<String, f64>,
}

impl RecallBaseline {
    pub fn from_report(report: &RecallReport) -> Self {
        Self {
            version: RECALL_BASELINE_VERSION,
            corpus_digest: report.corpus_digest.clone(),
            metrics: report.flatten_metrics(),
        }
    }

    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<(), std::io::Error> {
        let mut text = serde_json::to_string_pretty(self)?;
        text.push('\n');
        std::fs::write(path, text)
    }

    pub fn load(
        path: impl AsRef<std::path::Path>,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let text = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&text)?)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RecallMetricDelta {
    pub metric: String,
    pub baseline: f64,
    pub current: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RecallRatchetVerdict {
    pub passed: bool,
    pub digest_matches: bool,
    pub regressions: Vec<RecallMetricDelta>,
    pub gains: Vec<RecallMetricDelta>,
}

impl RecallRatchetVerdict {
    pub fn render_text(&self) -> String {
        if self.passed {
            return "召回棘轮检查通过：无指标低于基线。\n".to_string();
        }
        let mut output = String::from("召回棘轮检查未通过：\n");
        if !self.digest_matches {
            output.push_str("  - 构念分母摘要不匹配：规则/变体生成已漂移，基线不可直接比较\n");
        }
        for regression in &self.regressions {
            output.push_str(&format!(
                "  - {}: {:.6} → {:.6}\n",
                regression.metric, regression.baseline, regression.current
            ));
        }
        output
    }
}

impl RecallReport {
    /// 扁平指标（棘轮比较用）。全部方向都是越高越好。
    pub fn flatten_metrics(&self) -> BTreeMap<String, f64> {
        let mut metrics = BTreeMap::new();
        metrics.insert("recall".to_string(), self.recall());
        metrics.insert("seen_recall".to_string(), self.seen.recall);
        metrics.insert("unseen_recall".to_string(), self.unseen.recall);
        metrics.insert("primary_recall".to_string(), self.primary_recall());
        metrics.insert(
            "answered_rate".to_string(),
            if self.cases == 0 {
                0.0
            } else {
                self.answered() as f64 / self.cases as f64
            },
        );
        metrics.insert(
            "fallback_rescue_rate".to_string(),
            if self.cases == 0 {
                0.0
            } else {
                self.fallback_correct as f64 / self.cases as f64
            },
        );
        metrics
    }
}

/// 与基线比较：分母摘要不同或任何指标退化超 ε，即不通过。
pub fn compare_baseline(
    baseline: &RecallBaseline,
    current: &RecallReport,
    epsilon: f64,
) -> RecallRatchetVerdict {
    let digest_matches = baseline.corpus_digest == current.corpus_digest;
    let current_metrics = current.flatten_metrics();
    let mut regressions = Vec::new();
    let mut gains = Vec::new();
    for (metric, baseline_value) in &baseline.metrics {
        let current_value = current_metrics.get(metric).copied().unwrap_or(0.0);
        if current_value < baseline_value - epsilon {
            regressions.push(RecallMetricDelta {
                metric: metric.clone(),
                baseline: *baseline_value,
                current: current_value,
            });
        } else if current_value > baseline_value + epsilon {
            gains.push(RecallMetricDelta {
                metric: metric.clone(),
                baseline: *baseline_value,
                current: current_value,
            });
        }
    }
    RecallRatchetVerdict {
        passed: digest_matches && regressions.is_empty(),
        digest_matches,
        regressions,
        gains,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> SystemConfig {
        toml::from_str(include_str!("../config.toml")).expect("config.toml 必须可解析")
    }

    #[test]
    fn constructed_denominator_is_nonempty_and_has_both_slices() {
        let config = config();
        let library = PatternLibrary::from_config(&config);
        let synonyms = config
            .synonyms
            .as_ref()
            .map(|synonyms| synonyms.mappings.clone())
            .unwrap_or_default();
        let (cases, _) = construct_cases(&library, &synonyms);
        assert!(cases.len() >= 30, "分母过小：{}", cases.len());
        assert!(cases.iter().any(|case| case.seen));
        assert!(cases.iter().any(|case| !case.seen));
        assert!(cases
            .iter()
            .any(|case| case.variant == CaseVariant::Synonym));
        assert!(cases.iter().any(|case| case.variant == CaseVariant::Suffix));
        // 分母必须确定：同一配置重复构造得到同一序列。
        let (again, _) = construct_cases(&library, &synonyms);
        assert_eq!(cases, again);
    }

    #[test]
    fn report_splits_seen_unseen_and_keeps_digest_stable() {
        let config = config();
        let report = evaluate_config(&config, DEFAULT_NEAR_MISS);
        assert!(report.cases > 0);
        assert_eq!(report.seen.cases + report.unseen.cases, report.cases);
        assert_eq!(
            report.correct + report.wrong_route + report.unanswered,
            report.cases
        );
        assert_eq!(report.corpus_digest.len(), 16);
        let baseline = RecallBaseline::from_report(&report);
        let verdict = compare_baseline(&baseline, &report, 0.0);
        assert!(verdict.passed, "{}", verdict.render_text());
        assert!(verdict.regressions.is_empty());
    }

    #[test]
    fn baseline_flags_a_degraded_report() {
        let config = config();
        let report = evaluate_config(&config, DEFAULT_NEAR_MISS);
        let baseline = RecallBaseline::from_report(&report);
        let mut degraded = report.clone();
        degraded.correct = degraded.correct.saturating_sub(1);
        degraded.seen.correct = degraded.seen.correct.saturating_sub(1);
        let verdict = compare_baseline(&baseline, &degraded, 0.0);
        assert!(!verdict.passed);
        assert!(verdict
            .regressions
            .iter()
            .any(|delta| delta.metric == "recall"));
    }
}

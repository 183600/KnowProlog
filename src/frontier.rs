//! 决策前沿（Decision Frontier）：把「没懂」分解到具体要素。
//!
//! 既有日志记的是「说了什么、决定了什么」，但一次失败的匹配只剩一句
//! 「未命中」。本模块复用 [`crate::pattern_ir::match_entry_detailed`] 的
//! **同一段匹配语义**，把每条候选的失败位置、期望要素、实际词面与已覆盖
//! 前缀收集成可排序、可聚合、可序列化的前沿报告。
//!
//! 设计约束：
//!
//! - **不另写匹配器**：前沿只是 [`match_entry_detailed`] 的投影；语义漂移
//!   会立刻污染所有统计；
//! - **只读**：不改路由、不改分词、不写知识库；
//! - **可聚合**：单句报告用于解释，语料聚合用于给「补哪一个词面/句式」排序；
//! - **诚实**：前缀覆盖率不是「成功率」，它只回答「离第一个失配点有多远」。

use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};

use crate::nlp::{normalize_words, preprocess, TaggedWord};
use crate::pattern_ir::{
    match_entry_detailed, MatchFailure, MatchFailureKind, PatternContext, PatternLibrary,
};
use crate::NlpConfig;

/// 前沿报告的序列化方言；版本不符时调用方应拒绝把旧报告当新报告。
pub const FRONTIER_DIALECT: &str = "knowprolog-frontier/1";
/// 默认的「近似命中」覆盖率阈值：三分之二以上的前缀已对上。
pub const NEAR_MISS_COVERAGE: f64 = 0.75;
/// 单句报告里最多展示多少个近似命中。
pub const SHOWN_NEAR_MISSES: usize = 8;
/// 聚合报告里最多展示多少条高频缺口。
pub const SHOWN_GAPS: usize = 12;
/// 每条缺口保留多少个不同例句。
pub const GAP_EXAMPLES: usize = 3;

/// 词序列的稳定投影（只保留词面与词性，便于日志/JSON 对齐）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenView {
    pub text: String,
    pub pos: String,
}

/// 一条候选模式在本次输入上的前沿。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrontierRow {
    pub id: String,
    pub name: String,
    pub query_type: String,
    pub source: String,
    pub priority: i32,
    /// 是否主路由候选。
    pub route: bool,
    /// 是否 NLU 回退候选。
    pub fallback: bool,
    /// 模式的可读词面序列。
    pub pattern: String,
    pub matched: bool,
    /// 未命中时的结构化前沿；命中时为 `None`。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure: Option<MatchFailure>,
    /// 命中时绑定的实体（失败时为空）。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub entities: Vec<String>,
}

impl FrontierRow {
    /// 已覆盖前缀比例：命中为 1.0，失败由前沿给出。
    pub fn coverage(&self) -> f64 {
        if self.matched {
            1.0
        } else {
            self.failure
                .as_ref()
                .map(MatchFailure::coverage)
                .unwrap_or(0.0)
        }
    }

    /// 是否属于大前缀已对上的近似命中。
    pub fn is_near_miss(&self, threshold: f64) -> bool {
        !self.matched && self.coverage() >= threshold
    }
}

/// 单句的决策前沿报告。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrontierReport {
    pub dialect: String,
    pub input: String,
    pub tokens: Vec<TokenView>,
    /// 参与本次路由/回退评估的候选数。
    pub candidates: usize,
    pub matched: Vec<FrontierRow>,
    pub blocked: Vec<FrontierRow>,
    /// 胜者模式名（主路由优先，其次回退）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub winner: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub winner_query_type: Option<String>,
    /// 生成 `near_misses` 时使用的覆盖率阈值。
    pub near_miss_threshold: f64,
    /// 覆盖率降序的近似命中（只含未命中候选）。
    pub near_misses: Vec<FrontierRow>,
}

impl FrontierReport {
    pub fn routed(&self) -> bool {
        self.winner.is_some()
    }

    /// 最接近命中的受阻候选（覆盖率优先，其次优先级）。
    pub fn closest_blocked(&self) -> Option<&FrontierRow> {
        self.blocked.iter().max_by(|left, right| {
            left.coverage()
                .partial_cmp(&right.coverage())
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.priority.cmp(&right.priority))
                .then_with(|| right.name.cmp(&left.name))
        })
    }

    pub fn render_report(&self) -> String {
        let mut output = String::new();
        output.push_str(&format!("决策前沿 {}\n", self.dialect));
        output.push_str(&format!("输入：{}\n", self.input));
        let tokens = self
            .tokens
            .iter()
            .map(|token| format!("{}/{}", token.text, token.pos))
            .collect::<Vec<_>>()
            .join(" ");
        output.push_str(&format!("词序列：{}\n", tokens));
        match (&self.winner, &self.winner_query_type) {
            (Some(name), Some(query_type)) => {
                output.push_str(&format!("路由：命中「{}」（{}）\n", name, query_type));
            }
            _ => output.push_str("路由：未命中\n"),
        }
        output.push_str(&format!(
            "候选：{} 条（命中 {}，受阻 {}）\n",
            self.candidates,
            self.matched.len(),
            self.blocked.len()
        ));
        if let Some(best) = self.closest_blocked() {
            if let Some(failure) = &best.failure {
                output.push_str(&format!(
                    "最近缺口：{}  {:.0}%\n",
                    best.name,
                    failure.coverage() * 100.0
                ));
                output.push_str(&format!("  {}\n", failure.render()));
                output.push_str(&format!(
                    "  期望「{}」，实见「{}」\n",
                    failure.expected,
                    failure.actual.as_deref().unwrap_or("<输入结束>")
                ));
            }
        }
        if self.near_misses.is_empty() {
            output.push_str(&format!(
                "近似命中（覆盖率 ≥ {:.0}%）：0\n",
                self.near_miss_threshold * 100.0
            ));
        } else {
            output.push_str(&format!(
                "近似命中（覆盖率 ≥ {:.0}%）：{}\n",
                self.near_miss_threshold * 100.0,
                self.near_misses.len()
            ));
            for row in self.near_misses.iter().take(SHOWN_NEAR_MISSES) {
                if let Some(failure) = &row.failure {
                    output.push_str(&format!(
                        "  {:.0}%  {}：第{}要素期望「{}」，实见「{}」\n",
                        failure.coverage() * 100.0,
                        row.name,
                        failure.index + 1,
                        failure.expected,
                        failure.actual.as_deref().unwrap_or("<输入结束>")
                    ));
                }
            }
        }
        output
    }
}

/// 一条可排序的缺口：同一「期望 → 实见」在语料里出现多少次。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GapCount {
    pub kind: MatchFailureKind,
    pub expected: String,
    /// 实际词面；输入耗尽时为空串。
    pub actual: String,
    pub count: usize,
    /// 命中的例句（去重、有界）。
    pub examples: Vec<String>,
}

impl GapCount {
    pub fn label(&self) -> String {
        format!(
            "期望「{}」实见「{}」",
            self.expected,
            if self.actual.is_empty() {
                "<输入结束>"
            } else {
                self.actual.as_str()
            }
        )
    }
}

/// 语料的决策前沿聚合。分母是**语料里的问句数**，分子是其中成功路由的
/// 问句数；`near_miss_unmatched` 单独列出，避免把「差一点」混进「完全没懂」。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrontierAggregate {
    pub dialect: String,
    pub questions: usize,
    pub routed: usize,
    pub unmatched: usize,
    pub near_miss_unmatched: usize,
    pub near_miss_threshold: f64,
    /// 未命中问句的最近缺口，按出现次数降序。
    pub gaps: Vec<GapCount>,
    /// 未命中且近似命中的例句（有界）。
    pub near_miss_examples: Vec<String>,
}

impl FrontierAggregate {
    pub fn recall(&self) -> f64 {
        if self.questions == 0 {
            0.0
        } else {
            self.routed as f64 / self.questions as f64
        }
    }

    pub fn near_miss_rate(&self) -> f64 {
        if self.unmatched == 0 {
            0.0
        } else {
            self.near_miss_unmatched as f64 / self.unmatched as f64
        }
    }

    pub fn render_report(&self) -> String {
        let mut output = String::new();
        output.push_str(&format!("决策前沿聚合 {}\n", self.dialect));
        output.push_str(&format!(
            "问句：{}，命中：{}，未命中：{}，召回：{:.1}%\n",
            self.questions,
            self.routed,
            self.unmatched,
            self.recall() * 100.0
        ));
        output.push_str(&format!(
            "未命中中近似命中（覆盖率 ≥ {:.0}%）：{}（{:.1}%）\n",
            self.near_miss_threshold * 100.0,
            self.near_miss_unmatched,
            self.near_miss_rate() * 100.0
        ));
        if self.gaps.is_empty() {
            output.push_str("高频缺口：无\n");
        } else {
            output.push_str("高频缺口：\n");
            for gap in self.gaps.iter().take(SHOWN_GAPS) {
                output.push_str(&format!(
                    "  {}× {}（{}）\n",
                    gap.count,
                    gap.label(),
                    gap.kind.label()
                ));
                for example in &gap.examples {
                    output.push_str(&format!("      · {}\n", example));
                }
            }
        }
        if !self.near_miss_examples.is_empty() {
            output.push_str("近似命中例句：\n");
            for example in &self.near_miss_examples {
                output.push_str(&format!("  · {}\n", example));
            }
        }
        output
    }
}

/// 分析单句：分词、词面归一（与运行时同序），再收集所有路由/回退候选的前沿。
pub fn analyze(
    question: &str,
    library: &PatternLibrary,
    nlp: &NlpConfig,
    synonyms: &HashMap<String, String>,
) -> FrontierReport {
    analyze_with_threshold(question, library, nlp, synonyms, NEAR_MISS_COVERAGE)
}

/// 分析单句，并指定「近似命中」的覆盖率阈值。
pub fn analyze_with_threshold(
    question: &str,
    library: &PatternLibrary,
    nlp: &NlpConfig,
    synonyms: &HashMap<String, String>,
    near_miss_threshold: f64,
) -> FrontierReport {
    analyze_with_context_and_threshold(
        question,
        library,
        nlp,
        synonyms,
        &PatternContext::default(),
        near_miss_threshold,
    )
}

/// 带对话上下文的前沿分析（守卫不满足的候选不进入报告）。
pub fn analyze_with_context(
    question: &str,
    library: &PatternLibrary,
    nlp: &NlpConfig,
    synonyms: &HashMap<String, String>,
    context: &PatternContext,
) -> FrontierReport {
    analyze_with_context_and_threshold(
        question,
        library,
        nlp,
        synonyms,
        context,
        NEAR_MISS_COVERAGE,
    )
}

/// 带上下文与自定义阈值的完整入口。
pub fn analyze_with_context_and_threshold(
    question: &str,
    library: &PatternLibrary,
    nlp: &NlpConfig,
    synonyms: &HashMap<String, String>,
    context: &PatternContext,
    near_miss_threshold: f64,
) -> FrontierReport {
    let tagged = preprocess(question, nlp);
    let tokens = normalize_words(tagged, synonyms);
    analyze_tokens_with_threshold(question, &tokens, library, context, near_miss_threshold)
}

/// 已有词序列时的前沿分析（调用方负责与运行时保持同样的归一化）。
pub fn analyze_tokens(
    question: &str,
    tokens: &[TaggedWord],
    library: &PatternLibrary,
    context: &PatternContext,
) -> FrontierReport {
    analyze_tokens_with_threshold(question, tokens, library, context, NEAR_MISS_COVERAGE)
}

/// 已有词序列时的前沿分析，并指定近似命中阈值。
pub fn analyze_tokens_with_threshold(
    question: &str,
    tokens: &[TaggedWord],
    library: &PatternLibrary,
    context: &PatternContext,
    near_miss_threshold: f64,
) -> FrontierReport {
    let tokens_view: Vec<TokenView> = tokens
        .iter()
        .map(|token| TokenView {
            text: token.text.clone(),
            pos: token.pos.clone(),
        })
        .collect();

    let winner = library
        .match_route_with_context(tokens, context)
        .or_else(|| library.match_fallback_with_context(tokens, context));

    let mut matched = Vec::new();
    let mut blocked = Vec::new();
    let mut candidates = 0usize;

    for entry in library.entries() {
        if !(entry.route || entry.fallback) || !entry.context_satisfied(context) {
            continue;
        }
        candidates += 1;
        match match_entry_detailed(entry, tokens) {
            Ok(solution) => matched.push(FrontierRow {
                id: entry.id(),
                name: entry.name.clone(),
                query_type: entry.query_type.clone(),
                source: entry.source.clone(),
                priority: entry.priority,
                route: entry.route,
                fallback: entry.fallback,
                pattern: entry.render(),
                matched: true,
                failure: None,
                entities: solution.entities,
            }),
            Err(failure) => blocked.push(FrontierRow {
                id: entry.id(),
                name: entry.name.clone(),
                query_type: entry.query_type.clone(),
                source: entry.source.clone(),
                priority: entry.priority,
                route: entry.route,
                fallback: entry.fallback,
                pattern: entry.render(),
                matched: false,
                failure: Some(failure),
                entities: Vec::new(),
            }),
        }
    }

    let by_priority = |left: &FrontierRow, right: &FrontierRow| {
        right
            .priority
            .cmp(&left.priority)
            .then_with(|| left.name.cmp(&right.name))
    };
    matched.sort_by(by_priority);
    blocked.sort_by(by_priority);

    let mut near_misses: Vec<FrontierRow> = blocked
        .iter()
        .filter(|row| row.is_near_miss(near_miss_threshold))
        .cloned()
        .collect();
    near_misses.sort_by(|left, right| {
        right
            .coverage()
            .partial_cmp(&left.coverage())
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| right.priority.cmp(&left.priority))
            .then_with(|| left.name.cmp(&right.name))
    });
    near_misses.truncate(SHOWN_NEAR_MISSES);

    let winner_name = winner.as_ref().map(|solution| solution.entry.name.clone());
    let winner_query_type = winner
        .as_ref()
        .map(|solution| solution.entry.query_type.clone());

    FrontierReport {
        dialect: FRONTIER_DIALECT.to_string(),
        input: question.to_string(),
        tokens: tokens_view,
        candidates,
        matched,
        blocked,
        winner: winner_name,
        winner_query_type,
        near_miss_threshold,
        near_misses,
    }
}

/// 对一批问句做前沿聚合：失败问句按最近缺口归并、排序。
pub fn analyze_corpus<'a, I>(
    questions: I,
    library: &PatternLibrary,
    nlp: &NlpConfig,
    synonyms: &HashMap<String, String>,
    near_miss_threshold: f64,
) -> FrontierAggregate
where
    I: IntoIterator<Item = &'a str>,
{
    let mut aggregate = FrontierAggregate {
        dialect: FRONTIER_DIALECT.to_string(),
        questions: 0,
        routed: 0,
        unmatched: 0,
        near_miss_unmatched: 0,
        near_miss_threshold,
        gaps: Vec::new(),
        near_miss_examples: Vec::new(),
    };
    let mut gaps: BTreeMap<(String, String, MatchFailureKind), GapCount> = BTreeMap::new();

    for question in questions {
        let question = question.trim();
        if question.is_empty() {
            continue;
        }
        aggregate.questions += 1;
        let report = analyze(question, library, nlp, synonyms);
        if report.routed() {
            aggregate.routed += 1;
            continue;
        }
        aggregate.unmatched += 1;
        let Some(best) = report.closest_blocked() else {
            continue;
        };
        let Some(failure) = &best.failure else {
            continue;
        };
        if failure.coverage() >= near_miss_threshold {
            aggregate.near_miss_unmatched += 1;
            if aggregate.near_miss_examples.len() < SHOWN_GAPS {
                aggregate.near_miss_examples.push(format!(
                    "{}（{:.0}%：第{}要素期望「{}」）",
                    question,
                    failure.coverage() * 100.0,
                    failure.index + 1,
                    failure.expected
                ));
            }
        }
        let actual = failure.actual.clone().unwrap_or_default();
        let key = (failure.expected.clone(), actual.clone(), failure.kind);
        let entry = gaps.entry(key).or_insert_with(|| GapCount {
            kind: failure.kind,
            expected: failure.expected.clone(),
            actual,
            count: 0,
            examples: Vec::new(),
        });
        entry.count += 1;
        if entry.examples.len() < GAP_EXAMPLES
            && !entry.examples.iter().any(|item| item == question)
        {
            entry.examples.push(question.to_string());
        }
    }

    aggregate.gaps = gaps.into_values().collect();
    aggregate.gaps.sort_by(|left, right| {
        right
            .count
            .cmp(&left.count)
            .then_with(|| left.expected.cmp(&right.expected))
            .then_with(|| left.actual.cmp(&right.actual))
    });
    aggregate
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pattern_ir::{Elem, PatternEntry};

    fn library() -> PatternLibrary {
        let mut library = PatternLibrary::new();
        library.add(PatternEntry::new(
            "首都查询",
            "config.toml#首都查询",
            "capital_query",
            15,
            true,
            vec![
                Elem::pos("ns"),
                Elem::word("的"),
                Elem::word("首都"),
                Elem::word("是"),
                Elem::word("哪里"),
            ],
        ));
        library.add(PatternEntry::new(
            "首都查询2",
            "config.toml#首都查询2",
            "capital_query",
            14,
            true,
            vec![Elem::pos("ns"), Elem::word("首都")],
        ));
        library
    }

    fn nlp() -> NlpConfig {
        let config: crate::SystemConfig =
            toml::from_str(include_str!("../config.toml")).expect("config.toml 必须可解析");
        config.nlp
    }

    fn tagged(pairs: &[(&str, &str)]) -> Vec<TaggedWord> {
        pairs
            .iter()
            .map(|(text, pos)| TaggedWord {
                text: text.to_string(),
                pos: pos.to_string(),
            })
            .collect()
    }

    #[test]
    fn frontier_names_the_blocking_element_and_prefix() {
        let tokens = tagged(&[
            ("法国", "ns"),
            ("的", "u"),
            ("首都", "n"),
            ("在", "v"),
            ("哪里", "r"),
        ]);
        let report = analyze_tokens("法国的首都在哪里", &tokens, &library(), &Default::default());
        assert!(!report.routed());
        assert_eq!(report.candidates, 2);
        let best = report.closest_blocked().expect("应有受阻候选");
        assert_eq!(best.name, "首都查询");
        assert_eq!(best.coverage(), 0.6);
        let failure = best.failure.as_ref().unwrap();
        assert_eq!(failure.index, 3);
        assert_eq!(failure.expected, "是");
        assert_eq!(failure.actual.as_deref(), Some("在"));
        assert_eq!(failure.covered_elems(), 3);
        assert!(report.render_report().contains("第4要素"));
        assert_eq!(report.near_miss_threshold, NEAR_MISS_COVERAGE);
        assert!(
            report.near_misses.is_empty(),
            "0.6 覆盖率低于默认 0.75 阈值"
        );

        // 阈值必须真正作用于单句报告，而不只是聚合报告。
        let loose = analyze_tokens_with_threshold(
            "法国的首都在哪里",
            &tokens,
            &library(),
            &Default::default(),
            0.5,
        );
        assert_eq!(loose.near_miss_threshold, 0.5);
        assert!(loose.near_misses.iter().any(|row| row.name == "首都查询"));
    }

    #[test]
    fn aggregate_ranks_repeated_gaps_and_counts_near_misses() {
        let nlp = nlp();
        let report = analyze_corpus(
            ["法国的首都在哪里", "德国的首都在哪里", "德国首都", "你好"],
            &library(),
            &nlp,
            &HashMap::new(),
            0.5,
        );
        assert_eq!(report.questions, 4);
        assert_eq!(report.routed, 1, "「德国首都」应命中短模式");
        assert_eq!(report.unmatched, 3);
        assert_eq!(report.gaps[0].expected, "是");
        assert_eq!(report.gaps[0].actual, "在");
        assert_eq!(report.gaps[0].count, 2);
        assert!(report.near_miss_unmatched >= 2);
        assert!(report.render_report().contains("召回"));
    }

    #[test]
    fn route_winner_is_reported_without_reimplementing_matcher() {
        let tokens = tagged(&[
            ("法国", "ns"),
            ("的", "u"),
            ("首都", "n"),
            ("是", "v"),
            ("哪里", "r"),
        ]);
        let report = analyze_tokens("法国的首都是哪里", &tokens, &library(), &Default::default());
        assert!(report.routed());
        assert_eq!(report.winner.as_deref(), Some("首都查询"));
        assert_eq!(report.winner_query_type.as_deref(), Some("capital_query"));
        // 长模式命中；短模式（ns + 首都）停在「的」，同句里两种前沿并存。
        assert_eq!(report.matched.len(), 1, "{:?}", report.matched);
        assert!(report.blocked.iter().any(|row| row.name == "首都查询2"));
        assert!(report.near_misses.is_empty());
    }
}

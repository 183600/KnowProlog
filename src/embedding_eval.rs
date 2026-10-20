//! 符号嵌入评估台（Eval Harness）：统一的评估协议，给确定性符号构造
//! （RWR/PPR、PPMI、VSA 全息）的嵌入发一张可复现的「成绩单」，遵守三条戒律：
//!
//!  1. **确定性**：留出划分只用 FNV-1a 哈希（`canonical key % 1000`），
//!     无 RNG、无 HashMap 序依赖；指标对输入顺序不敏感，同输入同字节。
//!  2. **可审计**：划分摘要（split digest）入基线，指标可逐字节 diff；
//!     报告分层到 谓词 × 主体度数，样本不足（n < 5）显式标注，不下结论。
//!  3. **防退化**：基线 JSON + ε 棘轮。指标只许涨不许跌，跌了给出具体通道与差值。
//!
//! 评估方式为标准过滤式链接预测（object prediction）：
//!   对每条留出三元组 (s, p, o)，在被全图过滤掉其他真值的候选集上排序，
//!   统计 MRR / Hits@1 / Hits@3 / Hits@10。五个通道：
//!   - `ppr`：图游走重启的结构分量（`GraphWalkEmbedding`）
//!   - `ppmi`：可命名特征的稀疏嵌入余弦（`SymbolicEmbedding`）
//!   - `holo`：全息记忆的键控软召回（`HolographicMemory`，谓词敏感的硬核查通道）
//!   - `rule`：Horn 规则挖掘（`LinkPredictor`，组合/逆向/共邻）
//!   - `fused`：四通道倒数排名融合（RRF，k=60，无权重学习）
//!
//! 另给 Holo top-1 置信分做固定分桶校准（命中率按分桶单调即为有信号）；
//! 并用「训练对已知 + 留出对补全」的同谓词结构题（A:B::C:?）评估
//! PPMI / Holo / RRF 的类比泛化能力（金标不在训练侧出现，防记忆式捡漏）。

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::context_image::fnv1a_hex;
use crate::graph_walk_embedding::GraphWalkEmbedding;
use crate::holographic_memory::HolographicMemory;
use crate::link_prediction::LinkPredictor;
use crate::prolog_engine::PrologFact;
use crate::semantic_embedding::SymbolicEmbedding;

/// 基线 JSON 的格式版本。
pub const EVAL_BASELINE_VERSION: u32 = 1;
/// 倒数排名融合的平滑常数（标准 RRF k=60）。
pub const DEFAULT_RRF_K: f64 = 60.0;
/// 指标保留的小数位数（消除浮点尾差，保证跨进程可 diff）。
const RATE_DECIMALS: i32 = 6;
/// 排序量化尺度：分数乘该值取整后参与排序，浮点尾差不改排名。
const SCORE_QUANTUM: f64 = 1e9;
/// 报告里「样本不足」的阈值。
pub const LOW_SAMPLE_THRESHOLD: usize = 5;

/// 评估用三元组（独立小类型，避免与推理模块耦合）。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EvalTriple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

impl EvalTriple {
    pub fn new(subject: &str, predicate: &str, object: &str) -> Self {
        Self {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
        }
    }

    /// 规范键（制表符分隔，无歧义），用于哈希与摘要。
    pub fn key(&self) -> String {
        format!("{}\t{}\t{}", self.subject, self.predicate, self.object)
    }

    /// 稳定哈希桶：FNV-1a 前 16 位十六进制 → `% 1000`。
    fn bucket(&self) -> u64 {
        bucket_of(&self.key())
    }
}

fn bucket_of(key: &str) -> u64 {
    let hex = fnv1a_hex(key.as_bytes());
    u64::from_str_radix(&hex[..16], 16).unwrap_or(0) % 1000
}

/// 从整图关系事实提取评估三元组（实体属性不参与链接预测）。
pub fn triples_from_prolog_facts(facts: &[PrologFact]) -> Vec<EvalTriple> {
    facts
        .iter()
        .filter_map(|fact| match fact {
            PrologFact::Relation { from, rel_type, to } => {
                Some(EvalTriple::new(from, rel_type, to))
            }
            _ => None,
        })
        .collect()
}

/// 每个谓词在划分中的样本数。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PredicateSamples {
    pub train: usize,
    pub test: usize,
}

/// 确定性留出划分结果。
#[derive(Debug, Clone)]
pub struct HoldoutSplit {
    pub train: Vec<EvalTriple>,
    pub test: Vec<EvalTriple>,
    /// 划分摘要：对（比例 + 每条三元组的 T/E 指派）做 FNV-1a，入基线文件。
    pub digest: String,
    /// 划分比例（千分比）。
    pub test_ratio_per_mille: u32,
    pub predicate_samples: BTreeMap<String, PredicateSamples>,
}

/// 确定性留出划分。
///
/// - 输入先去重、按规范序排序，与调用方给的顺序无关；
/// - `bucket(t) < ratio` 进测试集（ratio=0 表示不设测试集）；
/// - 谓词保底：ratio>0 时每个谓词至少留一条测试三元组（取桶最小者，
///   并列时按规范序），避免低频谓词在评估中完全缺席；
/// - 摘要覆盖指派结果，基线可检测「划分漂移」。
pub fn deterministic_split(triples: &[EvalTriple], test_ratio_per_mille: u32) -> HoldoutSplit {
    let ratio = test_ratio_per_mille.min(1000);
    let mut unique: Vec<EvalTriple> = triples.to_vec();
    unique.sort();
    unique.dedup();

    let mut assignments: Vec<(EvalTriple, bool, u64)> = unique
        .into_iter()
        .map(|triple| {
            let bucket = triple.bucket();
            let is_test = ratio > 0 && bucket < ratio as u64;
            (triple, is_test, bucket)
        })
        .collect();

    if ratio > 0 {
        let mut by_predicate: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (index, (triple, _, _)) in assignments.iter().enumerate() {
            by_predicate
                .entry(triple.predicate.clone())
                .or_default()
                .push(index);
        }
        for indices in by_predicate.values() {
            if indices.iter().any(|&index| assignments[index].1) {
                continue;
            }
            if let Some(&best) = indices
                .iter()
                .min_by_key(|&&index| (assignments[index].2, assignments[index].0.clone()))
            {
                assignments[best].1 = true;
            }
        }
    }

    let mut train = Vec::new();
    let mut test = Vec::new();
    let mut predicate_samples: BTreeMap<String, PredicateSamples> = BTreeMap::new();
    let mut digest_input = format!("eval-split-v1 ratio={}\n", ratio);
    for (triple, is_test, _) in &assignments {
        let samples = predicate_samples
            .entry(triple.predicate.clone())
            .or_default();
        if *is_test {
            test.push(triple.clone());
            samples.test += 1;
            digest_input.push('T');
        } else {
            train.push(triple.clone());
            samples.train += 1;
            digest_input.push('E');
        }
        digest_input.push('\t');
        digest_input.push_str(&triple.key());
        digest_input.push('\n');
    }

    HoldoutSplit {
        train,
        test,
        digest: fnv1a_hex(digest_input.as_bytes()),
        test_ratio_per_mille: ratio,
        predicate_samples,
    }
}

/// 类比评估的通道 ID（与链接预测的通道分开放置，基线按前缀区分）。
pub const ANALOGY_PPMI_ID: &str = "analogy_ppmi";
pub const ANALOGY_HOLO_ID: &str = "analogy_holo";
pub const ANALOGY_FUSED_ID: &str = "analogy_fused";

/// 一道结构生成的类比题：`a : b :: c : ?`，金标答案 `d`。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnalogyQuestion {
    pub predicate: String,
    pub a: String,
    pub b: String,
    pub c: String,
    pub d: String,
}

/// 从留出集确定性生成同谓词类比题：已知对 `(a,p,b)` 来自训练，
/// 待补全对 `(c,p,·)` 来自测试（金标 `d` 只有测试集见过，防止记忆式捡漏）。
///
/// 每个谓词的训练对与测试对都按规范序排序；第 i 道题复用 `train_pairs[i % n]`，
/// 因此题目只依赖两侧集合内容与排序，不依赖调用顺序；每谓词至多 `cap` 道。
/// 同主语（a==c）或同客体（b==d）的退化题直接跳过。
pub fn generate_analogy_questions(
    train: &[EvalTriple],
    test: &[EvalTriple],
    cap: usize,
) -> Vec<AnalogyQuestion> {
    fn collect(triples: &[EvalTriple]) -> BTreeMap<&str, BTreeSet<(&str, &str)>> {
        let mut pairs: BTreeMap<&str, BTreeSet<(&str, &str)>> = BTreeMap::new();
        for triple in triples {
            if triple.subject == triple.object {
                continue;
            }
            pairs
                .entry(triple.predicate.as_str())
                .or_default()
                .insert((triple.subject.as_str(), triple.object.as_str()));
        }
        pairs
    }

    let train_pairs = collect(train);
    let test_pairs = collect(test);

    let mut questions = Vec::new();
    for (predicate, targets) in test_pairs {
        let Some(known) = train_pairs.get(predicate) else {
            continue;
        };
        let known: Vec<(&str, &str)> = known.iter().copied().collect();
        let mut taken = 0;
        for (index, &(c, d)) in targets.iter().enumerate() {
            if taken >= cap {
                break;
            }
            let (a, b) = known[index % known.len()];
            if a == c || b == d {
                continue;
            }
            questions.push(AnalogyQuestion {
                predicate: predicate.to_string(),
                a: a.to_string(),
                b: b.to_string(),
                c: c.to_string(),
                d: d.to_string(),
            });
            taken += 1;
        }
    }
    questions
}

/// 排序键：量化分数（NaN/负无穷 → i64::MIN）。
fn score_key(score: f64) -> i64 {
    if score.is_nan() {
        i64::MIN
    } else {
        (score * SCORE_QUANTUM).round() as i64
    }
}

fn round_metric(value: f64) -> f64 {
    let factor = 10f64.powi(RATE_DECIMALS);
    (value * factor).round() / factor
}

/// 单通道评估指标。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalMetrics {
    pub queries: usize,
    pub mrr: f64,
    pub hits1_rate: f64,
    pub hits3_rate: f64,
    pub hits10_rate: f64,
}

impl Default for EvalMetrics {
    fn default() -> Self {
        Self {
            queries: 0,
            mrr: 0.0,
            hits1_rate: 0.0,
            hits3_rate: 0.0,
            hits10_rate: 0.0,
        }
    }
}

impl EvalMetrics {
    /// 指定 k 的命中率（k ∈ {1, 3, 10}）。
    pub fn hit_rate(&self, k: usize) -> f64 {
        match k {
            1 => self.hits1_rate,
            3 => self.hits3_rate,
            10 => self.hits10_rate,
            _ => f64::NAN,
        }
    }
}

#[derive(Debug, Default, Clone)]
struct MetricAcc {
    queries: usize,
    reciprocal_rank: f64,
    hits1: usize,
    hits3: usize,
    hits10: usize,
}

impl MetricAcc {
    fn observe(&mut self, rank: usize) {
        if rank == 0 {
            return;
        }
        self.queries += 1;
        self.reciprocal_rank += 1.0 / rank as f64;
        if rank <= 1 {
            self.hits1 += 1;
        }
        if rank <= 3 {
            self.hits3 += 1;
        }
        if rank <= 10 {
            self.hits10 += 1;
        }
    }

    fn finalize(&self) -> EvalMetrics {
        if self.queries == 0 {
            return EvalMetrics::default();
        }
        let queries = self.queries as f64;
        EvalMetrics {
            queries: self.queries,
            mrr: round_metric(self.reciprocal_rank / queries),
            hits1_rate: round_metric(self.hits1 as f64 / queries),
            hits3_rate: round_metric(self.hits3 as f64 / queries),
            hits10_rate: round_metric(self.hits10 as f64 / queries),
        }
    }
}

/// 评估通道（稳定 ID 即报告与基线中的键）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvalChannel {
    Ppr,
    Ppmi,
    Holo,
    Rule,
    Fused,
}

impl EvalChannel {
    pub const ALL: [EvalChannel; 5] = [
        EvalChannel::Ppr,
        EvalChannel::Ppmi,
        EvalChannel::Holo,
        EvalChannel::Rule,
        EvalChannel::Fused,
    ];

    /// 稳定 ASCII 标识（报告、基线 JSON、测试断言共用）。
    pub fn id(self) -> &'static str {
        match self {
            EvalChannel::Ppr => "ppr",
            EvalChannel::Ppmi => "ppmi",
            EvalChannel::Holo => "holo",
            EvalChannel::Rule => "rule",
            EvalChannel::Fused => "fused",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            EvalChannel::Ppr => "PPR（图游走结构）",
            EvalChannel::Ppmi => "PPMI（可命名特征）",
            EvalChannel::Holo => "HOLO（全息软召回）",
            EvalChannel::Rule => "RULE（Horn规则挖掘）",
            EvalChannel::Fused => "FUSED（四通道RRF）",
        }
    }
}

/// 评估配置（各参数与推理侧同源，默认值贴近 config.toml）。
#[derive(Debug, Clone)]
pub struct EvalConfig {
    pub test_ratio_per_mille: u32,
    pub damping: f64,
    pub ppr_iterations: usize,
    pub similarity_floor: f64,
    pub holo_dimension: usize,
    pub rrf_k: f64,
    pub low_sample_threshold: usize,
    pub rule_min_support: usize,
    pub rule_min_confidence: f64,
    pub analogy_max_per_predicate: usize,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            test_ratio_per_mille: 200,
            damping: 0.85,
            ppr_iterations: 12,
            similarity_floor: 0.0,
            holo_dimension: 4096,
            rrf_k: DEFAULT_RRF_K,
            low_sample_threshold: LOW_SAMPLE_THRESHOLD,
            rule_min_support: 2,
            rule_min_confidence: 0.6,
            analogy_max_per_predicate: 50,
        }
    }
}

/// Holo top-1 置信校准分桶。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalibrationBin {
    pub label: String,
    pub count: usize,
    pub hits: usize,
}

impl CalibrationBin {
    pub fn hit_rate(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.hits as f64 / self.count as f64
        }
    }
}

/// 固定分桶边界（余弦分天然落在 [-1, 1]）。
const CALIBRATION_BINS: [(&str, f64, f64); 5] = [
    ("[-1.00,-0.20)", -1.0, -0.2),
    ("[-0.20,0.00)", -0.2, 0.0),
    ("[0.00,0.10)", 0.0, 0.1),
    ("[0.10,0.20)", 0.1, 0.2),
    ("[0.20,1.00]", 0.2, 1.0),
];

fn calibration_bin(score: f64) -> usize {
    for (index, (_, _, high)) in CALIBRATION_BINS.iter().enumerate() {
        if score < *high || index + 1 == CALIBRATION_BINS.len() {
            return index;
        }
    }
    CALIBRATION_BINS.len() - 1
}

/// 完整评估报告。
#[derive(Debug, Clone)]
pub struct EvalReport {
    pub triples: usize,
    pub train: usize,
    pub test: usize,
    /// 候选实体数（主语 ∪ 宾语，过滤式评估的排序规模）。
    pub candidates: usize,
    pub test_ratio_per_mille: u32,
    pub split_digest: String,
    /// 通道 ID → 指标。
    pub channels: BTreeMap<String, EvalMetrics>,
    /// 谓词 → 通道 ID → 指标。
    pub per_predicate: BTreeMap<String, BTreeMap<String, EvalMetrics>>,
    /// 谓词 → 样本数（train/test）。
    pub predicate_samples: BTreeMap<String, PredicateSamples>,
    /// 主体度数分桶 → 通道 ID → 指标。
    pub per_degree: BTreeMap<String, BTreeMap<String, EvalMetrics>>,
    /// 类比题数量（A:B::C:?）。
    pub analogy_queries: usize,
    /// 类比通道 ID → 指标（`analogy_*`）。
    pub analogy: BTreeMap<String, EvalMetrics>,
    /// Holo top-1 校准分桶。
    pub calibration: Vec<CalibrationBin>,
    /// 主体=客体的自环测试三元组（不参与排序）。
    pub skipped_self_loops: usize,
}

impl EvalReport {
    /// 检查该报告是否可用于棘轮比较的现成基线。
    pub fn channel_metrics(&self, channel: EvalChannel) -> Option<&EvalMetrics> {
        self.channels.get(channel.id())
    }

    /// 确定性文本报告（同输入同字节）。
    pub fn render_text(&self, config: &EvalConfig) -> String {
        let mut output = String::new();
        output.push_str("符号嵌入评估报告（确定性留出 · 可逐字节复现）\n");
        output.push_str("================================================\n");
        output.push_str(&format!(
            "三元组: {}（训练 {} / 测试 {}，测试比例 {}‰）\n",
            self.triples, self.train, self.test, self.test_ratio_per_mille
        ));
        output.push_str(&format!(
            "候选实体: {}（排序规模；Hits@k 在 k≥候选数 时无区分度）\n",
            self.candidates
        ));
        output.push_str(&format!("划分摘要: {}\n", self.split_digest));
        if self.candidates <= 10 && self.test > 0 {
            output.push_str("提示: 候选实体过少，Hits@k 指标仅供参考，建议先扩充知识库。\n");
        }
        if self.skipped_self_loops > 0 {
            output.push_str(&format!(
                "跳过自环测试三元组: {}\n",
                self.skipped_self_loops
            ));
        }

        output.push_str("\n通道指标（过滤式链接预测，按客体排序）:\n");
        output.push_str(&format!(
            "  {:<6} {:>6} {:>8} {:>8} {:>8} {:>8}\n",
            "通道", "查询", "MRR", "Hits@1", "Hits@3", "Hits@10"
        ));
        for channel in EvalChannel::ALL {
            if let Some(metrics) = self.channel_metrics(channel) {
                output.push_str(&format!(
                    "  {:<6} {:>6} {:>8.4} {:>7.1}% {:>7.1}% {:>7.1}%\n",
                    channel.id(),
                    metrics.queries,
                    metrics.mrr,
                    metrics.hits1_rate * 100.0,
                    metrics.hits3_rate * 100.0,
                    metrics.hits10_rate * 100.0
                ));
            }
        }

        output.push_str(&format!(
            "\n按谓词（Hits@10；n<{} 标注为样本不足）:\n",
            config.low_sample_threshold
        ));
        output.push_str(&format!(
            "  {:<16} {:>5} {:>5} {:>8} {:>8} {:>8} {:>8} {:>8}\n",
            "谓词", "训练", "测试", "ppr", "ppmi", "holo", "rule", "fused"
        ));
        for (predicate, samples) in &self.predicate_samples {
            let metrics = self.per_predicate.get(predicate);
            let mut row = format!(
                "  {:<16} {:>5} {:>5}",
                truncate_label(predicate, 16),
                samples.train,
                samples.test
            );
            for channel in EvalChannel::ALL {
                let value = metrics
                    .and_then(|map| map.get(channel.id()))
                    .map(|m| format!("{:>7.1}%", m.hits10_rate * 100.0))
                    .unwrap_or_else(|| "      -".to_string());
                row.push(' ');
                row.push_str(&value);
            }
            if samples.test < config.low_sample_threshold {
                row.push_str("  [样本不足]");
            }
            if samples.test == 0 {
                row.push_str(" [无测试样本]");
            }
            output.push_str(&row);
            output.push('\n');
        }

        output.push_str("\n按主体度数（Hits@10）:\n");
        output.push_str(&format!(
            "  {:<8} {:>5} {:>5} {:>8} {:>8} {:>8} {:>8} {:>8}\n",
            "度数", "查询", "—", "ppr", "ppmi", "holo", "rule", "fused"
        ));
        for (bucket, metrics) in &self.per_degree {
            let queries = metrics.values().next().map(|m| m.queries).unwrap_or(0);
            let mut row = format!("  {:<8} {:>5} {:>5}", bucket, queries, "");
            for channel in EvalChannel::ALL {
                let value = metrics
                    .get(channel.id())
                    .map(|m| format!("{:>7.1}%", m.hits10_rate * 100.0))
                    .unwrap_or_else(|| "      -".to_string());
                row.push(' ');
                row.push_str(&value);
            }
            output.push_str(&row);
            output.push('\n');
        }

        if self.analogy_queries > 0 {
            output.push_str(&format!(
                "\n类比 A:B::C:?（同谓词结构题，{} 道；过滤 (C,p,·) 已知真值）:\n",
                self.analogy_queries
            ));
            output.push_str(&format!(
                "  {:<14} {:>8} {:>8} {:>8} {:>8}\n",
                "方法", "MRR", "Hits@1", "Hits@3", "Hits@10"
            ));
            for (id, label) in [
                (ANALOGY_PPMI_ID, "ppmi"),
                (ANALOGY_HOLO_ID, "holo"),
                (ANALOGY_FUSED_ID, "fused"),
            ] {
                if let Some(metrics) = self.analogy.get(id) {
                    output.push_str(&format!(
                        "  {:<12} {:>8.4} {:>7.1}% {:>7.1}% {:>7.1}%\n",
                        label,
                        metrics.mrr,
                        metrics.hits1_rate * 100.0,
                        metrics.hits3_rate * 100.0,
                        metrics.hits10_rate * 100.0
                    ));
                }
            }
        } else if self.triples > 0 {
            output.push_str("\n类比 A:B::C:?：留出集为空或无同谓词配对，未生成题目。\n");
        }

        output.push_str("\nHolo top-1 置信校准:\n");
        for bin in &self.calibration {
            output.push_str(&format!(
                "  {:<15} {:>5} 条，命中 {:>4}（{:>5.1}%）\n",
                bin.label,
                bin.count,
                bin.hits,
                bin.hit_rate() * 100.0
            ));
        }

        output
    }
}

fn truncate_label(text: &str, max_chars: usize) -> String {
    let mut result = String::new();
    for (index, ch) in text.chars().enumerate() {
        if index >= max_chars {
            break;
        }
        result.push(ch);
    }
    if text.chars().count() > max_chars {
        result.push('…');
    }
    result
}

fn degree_bucket(degree: usize) -> &'static str {
    match degree {
        0 => "度0",
        1 => "度1",
        2..=3 => "度2-3",
        4..=7 => "度4-7",
        _ => "度8+",
    }
}

/// 在候选集上评估留出集；`all_triples` 用于实体词表与过滤式真值过滤。
pub fn evaluate_split(
    split: &HoldoutSplit,
    all_triples: &[EvalTriple],
    config: &EvalConfig,
) -> EvalReport {
    // 实体词表：全图主语 ∪ 宾语（只用名字，不泄漏任何留出边）。
    let mut vocabulary: BTreeSet<&str> = BTreeSet::new();
    let mut full_set: HashSet<(String, String, String)> = HashSet::new();
    for triple in all_triples {
        if triple.subject == triple.object {
            continue;
        }
        vocabulary.insert(triple.subject.as_str());
        vocabulary.insert(triple.object.as_str());
        full_set.insert((
            triple.subject.clone(),
            triple.predicate.clone(),
            triple.object.clone(),
        ));
    }
    let candidates: Vec<String> = vocabulary.into_iter().map(str::to_string).collect();
    let index_of: HashMap<&str, usize> = candidates
        .iter()
        .enumerate()
        .map(|(index, name)| (name.as_str(), index))
        .collect();
    let candidate_count = candidates.len().max(1);

    // 各通道观测训练集后重建（观测顺序 = 规范序，保证确定性）。
    let mut walks = GraphWalkEmbedding::new(config.damping, config.ppr_iterations, 0.0);
    let mut embedding = SymbolicEmbedding::new(0.0);
    let mut holo = HolographicMemory::new(
        config.holo_dimension,
        config.similarity_floor.max(0.0),
        candidate_count,
    );
    let mut predictor = LinkPredictor::new();
    for triple in &split.train {
        walks.observe_triple(&triple.subject, &triple.predicate, &triple.object);
        embedding.observe_triple(&triple.subject, &triple.predicate, &triple.object);
        holo.observe_triple(&triple.subject, &triple.predicate, &triple.object);
        predictor.add_triple(&triple.subject, &triple.predicate, &triple.object);
    }
    walks.rebuild();
    embedding.rebuild();
    predictor.mine_rules(config.rule_min_support, config.rule_min_confidence);

    // 训练集度数（结构分层用；不引入留出边）。
    let mut adjacency: HashMap<&str, HashSet<&str>> = HashMap::new();
    for triple in &split.train {
        adjacency
            .entry(triple.subject.as_str())
            .or_default()
            .insert(triple.object.as_str());
        adjacency
            .entry(triple.object.as_str())
            .or_default()
            .insert(triple.subject.as_str());
    }

    let mut overall: BTreeMap<&'static str, MetricAcc> = BTreeMap::new();
    let mut per_predicate: BTreeMap<String, BTreeMap<&'static str, MetricAcc>> = BTreeMap::new();
    let mut per_degree: BTreeMap<&'static str, BTreeMap<&'static str, MetricAcc>> = BTreeMap::new();
    let mut calibration = vec![(0usize, 0usize); CALIBRATION_BINS.len()];
    let mut skipped_self_loops = 0usize;

    let n = candidates.len();
    for triple in &split.test {
        if triple.subject == triple.object {
            skipped_self_loops += 1;
            continue;
        }
        let Some(&subject_index) = index_of.get(triple.subject.as_str()) else {
            continue;
        };
        let Some(&object_index) = index_of.get(triple.object.as_str()) else {
            continue;
        };

        // 排除集合：主体自身 + 已登记的其他真值客体（过滤式评估）。
        let mut excluded = vec![false; n];
        excluded[subject_index] = true;
        for (index, candidate) in candidates.iter().enumerate() {
            if index == object_index || index == subject_index {
                continue;
            }
            if full_set.contains(&(
                triple.subject.clone(),
                triple.predicate.clone(),
                candidate.clone(),
            )) {
                excluded[index] = true;
            }
        }

        // 通道 1：PPR 结构分量。
        let mut ppr_scores = vec![i64::MIN; n];
        let mut ppr_raw = vec![0.0f64; n];
        let mut ppr_hits = walks.related(&triple.subject, n);
        if ppr_hits.is_empty() {
            // 大图只保留单源按需计算时的回退。
            ppr_hits = walks.ppr(&triple.subject);
        }
        for (name, score) in ppr_hits {
            if let Some(&index) = index_of.get(name.as_str()) {
                ppr_scores[index] = score_key(score);
                ppr_raw[index] = score;
            }
        }

        // 通道 2：PPMI 余弦（未嵌入的候选视为无证据，排最后而不是按名字捡漏）。
        let mut ppmi_scores = vec![i64::MIN; n];
        for (index, candidate) in candidates.iter().enumerate() {
            if index == subject_index || !embedding.is_embedded(candidate) {
                continue;
            }
            let score = embedding.similarity(&triple.subject, candidate);
            ppmi_scores[index] = score_key(score);
        }

        // 通道 3：全息键控召回（谓词敏感）。
        let mut holo_scores = vec![i64::MIN; n];
        let mut holo_raw = vec![0.0f64; n];
        for hit in holo.recall_object(&triple.subject, &triple.predicate) {
            if let Some(&index) = index_of.get(hit.symbol.as_str()) {
                holo_scores[index] = score_key(hit.score);
                holo_raw[index] = hit.score;
            }
        }

        // 通道 4：Horn 规则挖掘（谓词敏感；同一客体取最高置信度）。
        let mut rule_scores = vec![i64::MIN; n];
        for prediction in predictor.predict_for_entity(&triple.subject, Some(&triple.predicate), n)
        {
            if prediction.triple.subject != triple.subject {
                continue;
            }
            if let Some(&index) = index_of.get(prediction.triple.object.as_str()) {
                let key = score_key(prediction.confidence);
                if key > rule_scores[index] {
                    rule_scores[index] = key;
                }
            }
        }

        let ppr_ranking = rank_channel(&candidates, &ppr_scores, &excluded);
        let ppmi_ranking = rank_channel(&candidates, &ppmi_scores, &excluded);
        let holo_ranking = rank_channel(&candidates, &holo_scores, &excluded);
        let rule_ranking = rank_channel(&candidates, &rule_scores, &excluded);
        let non_excluded = excluded.iter().filter(|flag| !**flag).count().max(1);

        // 通道 5：RRF 融合（只用「有证据」通道的名次；无任何证据的候选排最后）。
        let mut fused_scores = vec![i64::MIN; n];
        for index in 0..n {
            if excluded[index] {
                continue;
            }
            let mut sum = 0.0;
            let mut has_evidence = false;
            for ranking in [&ppr_ranking, &ppmi_ranking, &holo_ranking, &rule_ranking] {
                if ranking.ranks[index] > 0 {
                    sum += 1.0 / (config.rrf_k + ranking.ranks[index] as f64);
                    has_evidence = true;
                }
            }
            if has_evidence {
                fused_scores[index] = score_key(sum);
            }
        }
        let fused_ranking = rank_channel(&candidates, &fused_scores, &excluded);

        // 校准：Holo 第一名候选的原始余弦分 vs 是否命中。
        if let Some(top_index) = (0..n).find(|&index| holo_ranking.ranks[index] == 1) {
            let bin = calibration_bin(holo_raw[top_index]);
            calibration[bin].0 += 1;
            if top_index == object_index {
                calibration[bin].1 += 1;
            }
        }

        let ranks = [
            (
                EvalChannel::Ppr,
                effective_rank(&ppr_ranking, object_index, non_excluded),
            ),
            (
                EvalChannel::Ppmi,
                effective_rank(&ppmi_ranking, object_index, non_excluded),
            ),
            (
                EvalChannel::Holo,
                effective_rank(&holo_ranking, object_index, non_excluded),
            ),
            (
                EvalChannel::Rule,
                effective_rank(&rule_ranking, object_index, non_excluded),
            ),
            (
                EvalChannel::Fused,
                effective_rank(&fused_ranking, object_index, non_excluded),
            ),
        ];
        let predicate_entry = per_predicate.entry(triple.predicate.clone()).or_default();
        let degree = adjacency
            .get(triple.subject.as_str())
            .map(|neighbors| neighbors.len())
            .unwrap_or(0);
        let degree_entry = per_degree.entry(degree_bucket(degree)).or_default();
        for (channel, rank) in ranks {
            overall.entry(channel.id()).or_default().observe(rank);
            predicate_entry
                .entry(channel.id())
                .or_default()
                .observe(rank);
            degree_entry.entry(channel.id()).or_default().observe(rank);
        }
    }

    // ===== 类比评估（A:B::C:?）：已知对来自训练，待补对来自留出，过滤 (c,p,·) 已知真值 =====
    let analogy_questions =
        generate_analogy_questions(&split.train, &split.test, config.analogy_max_per_predicate);
    let mut analogy_acc: BTreeMap<&'static str, MetricAcc> = BTreeMap::new();
    {
        for question in &analogy_questions {
            let Some(&gold_index) = index_of.get(question.d.as_str()) else {
                continue;
            };
            let Some(&third_index) = index_of.get(question.c.as_str()) else {
                continue;
            };
            let mut excluded = vec![false; n];
            excluded[third_index] = true;
            for candidate in &candidates {
                if full_set.contains(&(
                    question.c.clone(),
                    question.predicate.clone(),
                    candidate.clone(),
                )) {
                    let index = index_of[candidate.as_str()];
                    if index != gold_index {
                        excluded[index] = true;
                    }
                }
            }

            let mut ppmi_scores = vec![i64::MIN; n];
            for (name, score) in embedding.analogy(&question.a, &question.b, &question.c, n) {
                if let Some(&index) = index_of.get(name.as_str()) {
                    ppmi_scores[index] = score_key(score);
                }
            }
            let mut holo_scores = vec![i64::MIN; n];
            for hit in holo.analogy(&question.a, &question.b, &question.c) {
                if let Some(&index) = index_of.get(hit.symbol.as_str()) {
                    holo_scores[index] = score_key(hit.score);
                }
            }

            let ppmi_ranking = rank_channel(&candidates, &ppmi_scores, &excluded);
            let holo_ranking = rank_channel(&candidates, &holo_scores, &excluded);
            let non_excluded = excluded.iter().filter(|flag| !**flag).count().max(1);
            let mut fused_scores = vec![i64::MIN; n];
            for index in 0..n {
                if excluded[index] {
                    continue;
                }
                let mut sum = 0.0;
                let mut has_evidence = false;
                for ranking in [&ppmi_ranking, &holo_ranking] {
                    if ranking.ranks[index] > 0 {
                        sum += 1.0 / (config.rrf_k + ranking.ranks[index] as f64);
                        has_evidence = true;
                    }
                }
                if has_evidence {
                    fused_scores[index] = score_key(sum);
                }
            }
            let fused_ranking = rank_channel(&candidates, &fused_scores, &excluded);
            for (id, ranking) in [
                (ANALOGY_PPMI_ID, &ppmi_ranking),
                (ANALOGY_HOLO_ID, &holo_ranking),
                (ANALOGY_FUSED_ID, &fused_ranking),
            ] {
                analogy_acc.entry(id).or_default().observe(effective_rank(
                    ranking,
                    gold_index,
                    non_excluded,
                ));
            }
        }
    }
    let analogy = finalize_named(
        &analogy_acc,
        &[ANALOGY_PPMI_ID, ANALOGY_HOLO_ID, ANALOGY_FUSED_ID],
    );

    let channels = finalize_named(&overall, &EvalChannel::ALL.map(EvalChannel::id));
    let per_predicate = per_predicate
        .into_iter()
        .map(|(predicate, map)| {
            (
                predicate,
                finalize_named(&map, &EvalChannel::ALL.map(EvalChannel::id)),
            )
        })
        .collect();
    let per_degree = per_degree
        .into_iter()
        .map(|(bucket, map)| {
            (
                bucket.to_string(),
                finalize_named(&map, &EvalChannel::ALL.map(EvalChannel::id)),
            )
        })
        .collect();
    let calibration = CALIBRATION_BINS
        .iter()
        .enumerate()
        .map(|(index, (label, _, _))| CalibrationBin {
            label: label.to_string(),
            count: calibration[index].0,
            hits: calibration[index].1,
        })
        .collect();

    EvalReport {
        triples: {
            let mut unique = all_triples.to_vec();
            unique.sort();
            unique.dedup();
            unique.len()
        },
        train: split.train.len(),
        test: split.test.len(),
        candidates: n,
        test_ratio_per_mille: split.test_ratio_per_mille,
        split_digest: split.digest.clone(),
        channels,
        per_predicate,
        predicate_samples: split.predicate_samples.clone(),
        per_degree,
        analogy_queries: analogy_questions.len(),
        analogy,
        calibration,
        skipped_self_loops,
    }
}

/// 单通道名次：`ranks[i] = 0` 表示该候选没有任何证据（量化分为 i64::MIN），
/// 统一按「排最后」处理，避免零信息时靠名字字典序捡漏命中。
struct ChannelRanking {
    ranks: Vec<usize>,
}

fn rank_channel(candidates: &[String], scores: &[i64], excluded: &[bool]) -> ChannelRanking {
    let mut order: Vec<usize> = (0..candidates.len())
        .filter(|&index| !excluded[index] && scores[index] != i64::MIN)
        .collect();
    order.sort_by(|&left, &right| {
        scores[right]
            .cmp(&scores[left])
            .then_with(|| candidates[left].cmp(&candidates[right]))
    });
    let mut ranks = vec![0usize; candidates.len()];
    for (position, index) in order.into_iter().enumerate() {
        ranks[index] = position + 1;
    }
    ChannelRanking { ranks }
}

/// 目标名次：无证据候选（ranks=0）计为并列最后一名。
fn effective_rank(ranking: &ChannelRanking, index: usize, non_excluded: usize) -> usize {
    if ranking.ranks[index] > 0 {
        ranking.ranks[index]
    } else {
        non_excluded.max(1)
    }
}

/// 按给定 ID 顺序收口观测表；缺失 ID 输出全零指标（通道始终齐全）。
fn finalize_named(map: &BTreeMap<&str, MetricAcc>, ids: &[&str]) -> BTreeMap<String, EvalMetrics> {
    ids.iter()
        .map(|id| {
            let metrics = map.get(id).map(MetricAcc::finalize).unwrap_or_default();
            ((*id).to_string(), metrics)
        })
        .collect()
}

/// 一步完成：划分 + 评估。
pub fn evaluate(triples: &[EvalTriple], config: &EvalConfig) -> EvalReport {
    let split = deterministic_split(triples, config.test_ratio_per_mille);
    evaluate_split(&split, triples, config)
}

/// 可写入磁盘的评估基线。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalBaseline {
    pub version: u32,
    pub split_digest: String,
    pub test_ratio_per_mille: u32,
    /// 通道 ID → 指标。
    pub channels: BTreeMap<String, EvalMetrics>,
    /// 类比通道 ID → 指标；该字段缺失时视为空。
    #[serde(default)]
    pub analogy: BTreeMap<String, EvalMetrics>,
}

impl EvalBaseline {
    pub fn from_report(report: &EvalReport) -> Self {
        Self {
            version: EVAL_BASELINE_VERSION,
            split_digest: report.split_digest.clone(),
            test_ratio_per_mille: report.test_ratio_per_mille,
            channels: report.channels.clone(),
            analogy: report.analogy.clone(),
        }
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), std::io::Error> {
        let mut text = serde_json::to_string_pretty(self)?;
        text.push('\n');
        std::fs::write(path, text)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let text = std::fs::read_to_string(path)?;
        let baseline: EvalBaseline = serde_json::from_str(&text)?;
        Ok(baseline)
    }
}

/// 单项指标变化。
#[derive(Debug, Clone, PartialEq)]
pub struct MetricDelta {
    pub channel: String,
    pub metric: &'static str,
    pub baseline: f64,
    pub current: f64,
}

/// 棘轮判定。
#[derive(Debug, Clone, PartialEq)]
pub struct RatchetVerdict {
    pub passed: bool,
    pub digest_matches: bool,
    pub drops: Vec<MetricDelta>,
    pub gains: Vec<MetricDelta>,
}

impl RatchetVerdict {
    /// 确定性文本结论。
    pub fn render_text(&self) -> String {
        let mut output = String::new();
        if self.passed {
            output.push_str("棘轮检查通过：无指标低于基线。\n");
        } else {
            output.push_str("棘轮检查未通过：\n");
            if !self.digest_matches {
                output.push_str("  - 划分摘要不匹配：留出集已漂移，基线不可直接比较\n");
            }
            for drop in &self.drops {
                output.push_str(&format!(
                    "  - {} {}: {:.6} → {:.6}（-{:.6}）\n",
                    drop.channel,
                    drop.metric,
                    drop.baseline,
                    drop.current,
                    drop.baseline - drop.current
                ));
            }
        }
        if !self.gains.is_empty() {
            output.push_str(&format!("有 {} 项指标高于基线。\n", self.gains.len()));
        }
        output
    }
}

/// 与基线比较：任何指标跌幅超过 ε、或划分摘要不同，即不通过。
pub fn compare_baseline(
    baseline: &EvalBaseline,
    current: &EvalReport,
    epsilon: f64,
) -> RatchetVerdict {
    let digest_matches = baseline.split_digest == current.split_digest
        && baseline.test_ratio_per_mille == current.test_ratio_per_mille;
    let mut drops = Vec::new();
    let mut gains = Vec::new();

    compare_metric_maps(
        &baseline.channels,
        &current.channels,
        epsilon,
        &mut drops,
        &mut gains,
    );
    compare_metric_maps(
        &baseline.analogy,
        &current.analogy,
        epsilon,
        &mut drops,
        &mut gains,
    );

    RatchetVerdict {
        passed: digest_matches && drops.is_empty(),
        digest_matches,
        drops,
        gains,
    }
}

/// 比较两组通道指标：跌幅超 ε 记入 `drops`，涨幅超 ε 记入 `gains`。
fn compare_metric_maps(
    baseline: &BTreeMap<String, EvalMetrics>,
    current: &BTreeMap<String, EvalMetrics>,
    epsilon: f64,
    drops: &mut Vec<MetricDelta>,
    gains: &mut Vec<MetricDelta>,
) {
    for (channel, baseline_metrics) in baseline {
        let current_metrics = current.get(channel).cloned().unwrap_or_default();
        let pairs = [
            ("mrr", baseline_metrics.mrr, current_metrics.mrr),
            (
                "hits1_rate",
                baseline_metrics.hits1_rate,
                current_metrics.hits1_rate,
            ),
            (
                "hits3_rate",
                baseline_metrics.hits3_rate,
                current_metrics.hits3_rate,
            ),
            (
                "hits10_rate",
                baseline_metrics.hits10_rate,
                current_metrics.hits10_rate,
            ),
        ];
        for (metric, baseline_value, current_value) in pairs {
            let delta = current_value - baseline_value;
            if delta < -epsilon {
                drops.push(MetricDelta {
                    channel: channel.clone(),
                    metric,
                    baseline: baseline_value,
                    current: current_value,
                });
            } else if delta > epsilon {
                gains.push(MetricDelta {
                    channel: channel.clone(),
                    metric,
                    baseline: baseline_value,
                    current: current_value,
                });
            }
        }
    }
    // 基线中有、当前完全缺失的通道视为证据丢失。
    for channel in baseline.keys() {
        if current.contains_key(channel) {
            continue;
        }
        drops.push(MetricDelta {
            channel: channel.clone(),
            metric: "channel_missing",
            baseline: 1.0,
            current: 0.0,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_triples() -> Vec<EvalTriple> {
        let mut triples = Vec::new();
        for index in 0..40 {
            let subject = format!("人物{:02}", index);
            let object = format!("师傅{:02}", index);
            triples.push(EvalTriple::new(&subject, "师傅", &object));
            triples.push(EvalTriple::new(&object, "徒弟", &subject));
        }
        for index in 0..10 {
            triples.push(EvalTriple::new(
                &format!("公司{:02}", index),
                "位于",
                &format!("城市{:02}", index % 4),
            ));
        }
        triples
    }

    fn fast_config() -> EvalConfig {
        EvalConfig {
            test_ratio_per_mille: 300,
            ppr_iterations: 6,
            holo_dimension: 256,
            ..Default::default()
        }
    }

    #[test]
    fn split_is_deterministic_under_input_shuffle() {
        let triples = sample_triples();
        let first = deterministic_split(&triples, 300);

        let mut shuffled = triples.clone();
        shuffled.reverse();
        shuffled.rotate_left(7);
        let second = deterministic_split(&shuffled, 300);

        assert_eq!(first.train, second.train);
        assert_eq!(first.test, second.test);
        assert_eq!(first.digest, second.digest);
        assert_eq!(first.predicate_samples, second.predicate_samples);
    }

    #[test]
    fn split_covers_every_predicate_and_respects_ratio_zero() {
        let triples = vec![
            EvalTriple::new("甲", "认识", "乙"),
            EvalTriple::new("乙", "认识", "丙"),
            EvalTriple::new("丙", "认识", "丁"),
            EvalTriple::new("独苗", "特殊关系", "孤例"),
        ];
        let split = deterministic_split(&triples, 1);
        // 谓词保底：即使桶不命中，每个谓词也至少有一条测试。
        for samples in split.predicate_samples.values() {
            assert!(
                samples.test >= 1,
                "谓词保底失败: {:?}",
                split.predicate_samples
            );
        }
        // 去重后 4 条，全部进测试也不会超过总数。
        assert_eq!(split.train.len() + split.test.len(), 4);

        let zero = deterministic_split(&triples, 0);
        assert!(zero.test.is_empty(), "ratio=0 不应产生测试集");
        assert_eq!(zero.train.len(), 4);
    }

    #[test]
    fn evaluate_is_deterministic_and_order_invariant() {
        let triples = sample_triples();
        let config = fast_config();
        let first = evaluate(&triples, &config);

        let mut shuffled = triples.clone();
        shuffled.reverse();
        shuffled.rotate_left(5);
        let second = evaluate(&shuffled, &config);

        assert_eq!(first.split_digest, second.split_digest);
        assert_eq!(first.channels, second.channels);
        assert_eq!(first.per_predicate, second.per_predicate);
        assert_eq!(first.per_degree, second.per_degree);
        assert_eq!(first.calibration, second.calibration);

        // 同输入再跑一次：逐项相同（逐字节可复现）。
        let third = evaluate(&triples, &config);
        assert_eq!(first.channels, third.channels);
    }

    #[test]
    fn metrics_respond_to_training_evidence() {
        // 结构桥：p00 与 q00 共享邻居 r00，留出 p00 -师傅-> q00。
        // 有训练边时 PPR 应从 p00 扩散到 q00；训练为空时任何通道都无证据。
        let mut triples = vec![
            EvalTriple::new("p00", "师傅", "q00"),
            EvalTriple::new("p00", "住在", "r00"),
            EvalTriple::new("q00", "住在", "r00"),
        ];
        for index in 1..30 {
            triples.push(EvalTriple::new(
                &format!("p{:02}", index),
                "师傅",
                &format!("q{:02}", index),
            ));
            triples.push(EvalTriple::new(
                &format!("p{:02}", index),
                "住在",
                &format!("r{:02}", index),
            ));
            triples.push(EvalTriple::new(
                &format!("q{:02}", index),
                "住在",
                &format!("r{:02}", index),
            ));
        }
        let all = triples.clone();
        let target = triples[0].clone();
        let config = fast_config();

        let mut rich_train = triples.clone();
        rich_train.retain(|triple| triple != &target);
        let rich_split = HoldoutSplit {
            train: rich_train,
            test: vec![target.clone()],
            digest: "rich".to_string(),
            test_ratio_per_mille: 0,
            predicate_samples: BTreeMap::new(),
        };
        let rich = evaluate_split(&rich_split, &all, &config);

        let empty_split = HoldoutSplit {
            train: Vec::new(),
            test: vec![target.clone()],
            digest: "empty".to_string(),
            test_ratio_per_mille: 0,
            predicate_samples: BTreeMap::new(),
        };
        let empty = evaluate_split(&empty_split, &all, &config);

        let rich_ppr = rich.channel_metrics(EvalChannel::Ppr).unwrap();
        assert_eq!(rich_ppr.hits10_rate, 1.0, "共享邻居应让 PPR 命中 Top-10");
        assert!(rich_ppr.mrr > 0.0);

        for channel in EvalChannel::ALL {
            let metrics = empty.channel_metrics(channel).unwrap();
            // 无证据候选统一记「并列最后一名」：命中为 0，MRR 只剩 1/候选数 的兜底。
            assert_eq!(metrics.hits10_rate, 0.0, "{:?} 不应有命中收益", channel);
            assert!(
                metrics.mrr < 0.05,
                "{:?} 不应有排名收益: {}",
                channel,
                metrics.mrr
            );
        }
        assert!(
            rich_ppr.mrr > empty.channel_metrics(EvalChannel::Ppr).unwrap().mrr,
            "有结构证据时 MRR 应更高"
        );
    }

    #[test]
    fn rule_channel_completes_transitive_chain() {
        // 训练集同时包含传递链 A→B→C 与直达边 A→C（规则的支持度来源）；
        // 另有一条 D→E→F 的链留出 D→F：组合规则应把 F 排在首位。
        let train = vec![
            EvalTriple::new("A", "师傅", "B"),
            EvalTriple::new("B", "师傅", "C"),
            EvalTriple::new("A", "师傅", "C"),
            EvalTriple::new("D", "师傅", "E"),
            EvalTriple::new("E", "师傅", "F"),
        ];
        let target = EvalTriple::new("D", "师傅", "F");
        let all = {
            let mut triples = train.clone();
            triples.push(target.clone());
            triples
        };
        let split = HoldoutSplit {
            train,
            test: vec![target],
            digest: "rule-chain".to_string(),
            test_ratio_per_mille: 0,
            predicate_samples: BTreeMap::new(),
        };
        let config = EvalConfig {
            test_ratio_per_mille: 0,
            ppr_iterations: 4,
            holo_dimension: 256,
            rule_min_support: 1,
            rule_min_confidence: 0.4,
            ..Default::default()
        };
        let report = evaluate_split(&split, &all, &config);
        let rule = report.channel_metrics(EvalChannel::Rule).unwrap();
        assert_eq!(rule.hits1_rate, 1.0, "组合规则应命中传递边");
        assert_eq!(rule.mrr, 1.0);
        let fused = report.channel_metrics(EvalChannel::Fused).unwrap();
        assert_eq!(fused.hits1_rate, 1.0, "融合通道应继承规则证据");
    }

    #[test]
    fn analogy_questions_are_deterministic_and_skip_degenerate_pairs() {
        let train = vec![EvalTriple::new("A", "p", "1")];
        let test = vec![
            EvalTriple::new("X", "p", "2"),
            EvalTriple::new("Y", "p", "3"),
            EvalTriple::new("Z", "q", "4"),
        ];
        let first = generate_analogy_questions(&train, &test, 10);
        let second = generate_analogy_questions(&train, &test, 10);
        assert_eq!(first, second, "同输入应生成完全相同的题");
        assert_eq!(first.len(), 2, "训练里没有 q，Z 题应被跳过");
        assert_eq!(first[0].a, "A");
        assert_eq!(first[0].b, "1");
        assert_eq!(first[0].c, "X");
        assert_eq!(first[0].d, "2");
        assert_eq!(first[1].c, "Y");
        assert_eq!(first[1].d, "3");
        assert!(first.iter().all(|q| q.a != q.c && q.b != q.d));

        let degenerate = vec![EvalTriple::new("A", "p", "2")];
        assert!(
            generate_analogy_questions(&train, &degenerate, 10).is_empty(),
            "同主语题应跳过"
        );

        let many: Vec<EvalTriple> = (0..4)
            .map(|index| EvalTriple::new(&format!("t{index}"), "p", &format!("u{index}")))
            .collect();
        assert_eq!(
            generate_analogy_questions(&train, &many, 2).len(),
            2,
            "cap 应生效"
        );
        assert!(generate_analogy_questions(&train, &many, 0).is_empty());
        assert!(
            generate_analogy_questions(&[], &test, 10).is_empty(),
            "没有已知对就没有类比题"
        );
    }

    #[test]
    fn analogy_channels_generalize_held_out_target() {
        // 训练：甲父亲乙；丙老师丁；丁/乙喜欢戊。留出：丙父亲丁。
        // 题 甲:乙 :: 丙:? → 丁；丁未在训练里与父亲相连，考的是向量算术。
        let train = vec![
            EvalTriple::new("甲", "父亲", "乙"),
            EvalTriple::new("丙", "老师", "丁"),
            EvalTriple::new("丁", "喜欢", "戊"),
            EvalTriple::new("乙", "喜欢", "戊"),
        ];
        let test = vec![EvalTriple::new("丙", "父亲", "丁")];
        let all = {
            let mut all = train.clone();
            all.extend(test.clone());
            all
        };
        let split = HoldoutSplit {
            train,
            test,
            digest: "analogy".to_string(),
            test_ratio_per_mille: 0,
            predicate_samples: BTreeMap::new(),
        };
        let config = EvalConfig {
            ppr_iterations: 4,
            holo_dimension: 256,
            ..Default::default()
        };
        let report = evaluate_split(&split, &all, &config);
        assert_eq!(report.analogy_queries, 1);
        let ppmi = report.analogy.get(ANALOGY_PPMI_ID).unwrap();
        assert_eq!(ppmi.queries, 1);
        assert_eq!(
            ppmi.hits1_rate, 1.0,
            "v(乙)-v(甲)+v(丙) 应最接近留出目标 v(丁)"
        );
        let fused = report.analogy.get(ANALOGY_FUSED_ID).unwrap();
        assert_eq!(fused.hits10_rate, 1.0);
        let text = report.render_text(&config);
        assert!(text.contains("类比 A:B::C:?"), "报告应包含类比表");
    }

    #[test]
    fn per_predicate_report_marks_low_samples() {
        let triples = vec![
            EvalTriple::new("甲", "认识", "乙"),
            EvalTriple::new("乙", "认识", "丙"),
            EvalTriple::new("独苗", "特殊关系", "孤例"),
        ];
        let config = fast_config();
        let report = evaluate(&triples, &config);
        let text = report.render_text(&config);
        assert!(text.contains("划分摘要"), "报告应包含划分摘要");
        assert!(text.contains("特殊关系"), "报告应包含谓词分层");
        assert!(text.contains("样本不足"), "低频谓词应标注样本不足");
        assert!(text.contains("Holo top-1 置信校准"), "报告应包含校准表");
    }

    #[test]
    fn baseline_roundtrip_and_ratchet() {
        let triples = sample_triples();
        let config = fast_config();
        let report = evaluate(&triples, &config);
        let baseline = EvalBaseline::from_report(&report);

        let verdict = compare_baseline(&baseline, &report, 1e-6);
        assert!(verdict.passed, "同报告对自身基线应通过");
        assert!(verdict.digest_matches);
        assert!(verdict.drops.is_empty());

        // 人为降低一个通道的指标：应被棘轮抓住。
        let mut damaged = report.clone();
        let fused = damaged.channels.get_mut("fused").unwrap();
        fused.hits10_rate = (fused.hits10_rate - 0.25).max(0.0);
        let verdict = compare_baseline(&baseline, &damaged, 1e-6);
        assert!(!verdict.passed, "指标下滑必须被检出");
        assert!(verdict
            .drops
            .iter()
            .any(|drop| drop.channel == "fused" && drop.metric == "hits10_rate"));

        // 划分漂移：摘要不一致直接不通过。
        let mut drifted = damaged.clone();
        drifted.split_digest = "漂移".to_string();
        let verdict = compare_baseline(&baseline, &drifted, 1e-6);
        assert!(!verdict.passed);
        assert!(!verdict.digest_matches);
    }

    #[test]
    fn baseline_file_roundtrip() {
        let triples = sample_triples();
        let report = evaluate(&triples, &fast_config());
        let baseline = EvalBaseline::from_report(&report);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("eval_baseline.json");
        baseline.save(&path).unwrap();
        let loaded = EvalBaseline::load(&path).unwrap();
        assert_eq!(baseline, loaded);
    }

    #[test]
    fn empty_test_set_yields_zero_metrics_without_panic() {
        let triples = sample_triples();
        let mut config = fast_config();
        config.test_ratio_per_mille = 0;
        let report = evaluate(&triples, &config);
        assert_eq!(report.test, 0);
        for channel in EvalChannel::ALL {
            let metrics = report.channel_metrics(channel).unwrap();
            assert_eq!(metrics.queries, 0);
            assert_eq!(metrics.mrr, 0.0);
        }
        assert!(report.calibration.iter().all(|bin| bin.count == 0));
    }

    #[test]
    fn calibration_counts_are_consistent() {
        let triples = sample_triples();
        let report = evaluate(&triples, &fast_config());
        let total: usize = report.calibration.iter().map(|bin| bin.count).sum();
        let queries = report.channel_metrics(EvalChannel::Holo).unwrap().queries;
        // 校准只统计 Holo 真正给出候选的查询，因此总数不超过查询数。
        assert!(total > 0, "应至少有一次 Holo 召回");
        assert!(total <= queries, "校准分桶数不能超过查询数");
        assert!(report.calibration.iter().all(|bin| bin.hits <= bin.count));
    }

    #[test]
    fn prolog_fact_conversion_keeps_only_relations() {
        let facts = vec![
            PrologFact::Relation {
                from: "甲".to_string(),
                rel_type: "认识".to_string(),
                to: "乙".to_string(),
            },
            PrologFact::Entity {
                name: "甲".to_string(),
                entity_type: "Person".to_string(),
                properties: HashMap::new(),
            },
        ];
        let triples = triples_from_prolog_facts(&facts);
        assert_eq!(triples, vec![EvalTriple::new("甲", "认识", "乙")]);
    }

    #[test]
    fn score_quantization_ignores_float_tails() {
        assert_eq!(score_key(0.1234567891), score_key(0.1234567892));
        assert!(score_key(0.5) > score_key(0.4));
        assert_eq!(score_key(f64::NAN), i64::MIN);
    }
}

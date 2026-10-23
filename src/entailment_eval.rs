//! 语义蕴含评测（Entailment Eval）：把蕴含判定的承诺固化成金标语料与
//! 变形关系（metamorphic relations），并提供可存档、可比较的指标基线。
//!
//! 评测只用确定性的内存图谱，不读磁盘、不联网：
//!   • 金标层：陈述句 → 期望判定（蕴含 / 矛盾 / 未知），覆盖直接事实、
//!     属性记录、类型断言、否定、封闭世界、容错链接与同义改写；
//!   • 变形层：不依赖人工金标也能检验语义不变量——确定性、知识单调增、
//!     支持撤回、否定对偶、插入顺序无关；
//!   • 基线层：把指标摊平成可比对的表，发布闸门用它做棘轮。

use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};

use crate::context_image::{canonical_json, fnv1a_hex};
use crate::entity_linker::EntityLinker;
use crate::link_prediction::LinkPredictor;
use crate::ontology_alignment::{OntologyAligner, TermKind};
use crate::semantic_entailment::{
    entail, EntailmentResult, Knowledge, Proposition, PropositionParser, Verdict,
};
use crate::slot_ledger::SlotPolicy;
use crate::symbolic_generator::SymbolicGenerator;

pub const ENTAILMENT_EVAL_VERSION: u32 = 1;

#[derive(Debug, Clone)]
struct Triple {
    subject: &'static str,
    predicate: &'static str,
    object: &'static str,
}

const FACTS: &[Triple] = &[
    Triple { subject: "张三", predicate: "师傅", object: "李四" },
    Triple { subject: "李四", predicate: "师傅", object: "王五" },
    Triple { subject: "张三", predicate: "朋友", object: "王五" },
    Triple { subject: "张三", predicate: "出生地", object: "北京" },
    Triple { subject: "李四", predicate: "出生地", object: "上海" },
];

struct Fixture {
    aligner: OntologyAligner,
    linker: EntityLinker,
    predictor: LinkPredictor,
    generator: SymbolicGenerator,
    policy: SlotPolicy,
}

impl Fixture {
    fn new() -> Self {
        Self::build(FACTS.iter(), true)
    }

    fn without_friend() -> Self {
        Self::build(
            FACTS.iter().filter(|triple| triple.predicate != "朋友"),
            true,
        )
    }

    fn reversed() -> Self {
        Self::build(FACTS.iter().rev(), true)
    }

    fn build<'a>(facts: impl Iterator<Item = &'a Triple>, with_synonyms: bool) -> Self {
        let mut aligner = OntologyAligner::new(HashMap::new(), 0.72);
        let mut linker = EntityLinker::new();
        let mut predictor = LinkPredictor::new();
        let mut generator = SymbolicGenerator::new(6, 0.3);

        aligner.register_term("人物", TermKind::Type);
        aligner.register_term("地点", TermKind::Type);
        for (name, etype) in [
            ("张三", "人物"),
            ("李四", "人物"),
            ("王五", "人物"),
            ("赵六", "人物"),
            ("北京", "地点"),
            ("上海", "地点"),
        ] {
            aligner.register_term(name, TermKind::Entity);
            linker.register_entity(name, Some(etype));
            generator.observe_entity(name, etype, &[]);
        }
        aligner.register_term("年龄", TermKind::Attribute);
        generator.observe_entity("张三", "人物", &[("年龄".to_string(), "30".to_string())]);
        if with_synonyms {
            aligner.register_synonym("老师", "师傅");
            aligner.register_synonym("岁数", "年龄");
            generator.register_predicate_alias("老师", "师傅");
            generator.register_predicate_alias("岁数", "年龄");
            linker.register_alias("老师", "师傅");
        }

        for triple in facts {
            aligner.observe_triple(triple.subject, triple.predicate, triple.object);
            predictor.add_triple(triple.subject, triple.predicate, triple.object);
            linker.observe_triple(triple.subject, triple.predicate, triple.object);
            generator.observe_triple(triple.subject, triple.predicate, triple.object);
        }

        let policy = SlotPolicy::new().with_functional(["年龄", "出生地"]);

        Self {
            aligner,
            linker,
            predictor,
            generator,
            policy,
        }
    }

    fn parse(&self, text: &str) -> Vec<Proposition> {
        PropositionParser::new(
            &self.aligner,
            &self.linker,
            self.generator.predicate_surfaces(),
        )
        .parse(text)
    }

    fn knowledge(&self) -> Knowledge<'_> {
        Knowledge {
            predictor: &self.predictor,
            generator: &self.generator,
            min_rule_confidence: 0.5,
            max_rule_candidates: 16,
            slot_policy: Some(&self.policy),
        }
    }

    fn verdict(&self, text: &str) -> Verdict {
        self.results(text)
            .first()
            .map(|result| result.verdict)
            .unwrap_or(Verdict::Unknown)
    }

    fn results(&self, text: &str) -> Vec<EntailmentResult> {
        let knowledge = self.knowledge();
        self.parse(text)
            .into_iter()
            .map(|proposition| entail(&proposition, &knowledge))
            .collect()
    }
}

struct GoldenCase {
    text: &'static str,
    expected: Verdict,
    tag: &'static str,
}

const GOLDEN: &[GoldenCase] = &[
    GoldenCase { text: "张三的师傅是李四。", expected: Verdict::Entailed, tag: "direct" },
    GoldenCase { text: "李四是张三的师傅。", expected: Verdict::Entailed, tag: "inverse" },
    GoldenCase { text: "李四的师傅是王五。", expected: Verdict::Entailed, tag: "direct" },
    GoldenCase { text: "王五的师傅是赵六。", expected: Verdict::Unknown, tag: "missing" },
    GoldenCase { text: "张三的师傅是王五。", expected: Verdict::Unknown, tag: "multi_value" },
    GoldenCase { text: "张三的师傅不是王五。", expected: Verdict::Unknown, tag: "multi_value" },
    GoldenCase { text: "张三的朋友是王五。", expected: Verdict::Entailed, tag: "direct" },
    GoldenCase { text: "张三的朋友是李四。", expected: Verdict::Unknown, tag: "multi_value" },
    GoldenCase { text: "张三的出生地是北京。", expected: Verdict::Entailed, tag: "direct" },
    GoldenCase { text: "张三的出生地是上海。", expected: Verdict::Contradicted, tag: "functional" },
    GoldenCase { text: "张三的出生地不是上海。", expected: Verdict::Entailed, tag: "closed_world" },
    GoldenCase { text: "张三的年龄是30。", expected: Verdict::Entailed, tag: "attribute" },
    GoldenCase { text: "张三的年龄是31。", expected: Verdict::Contradicted, tag: "functional" },
    GoldenCase { text: "张三的年龄不是31。", expected: Verdict::Entailed, tag: "closed_world" },
    GoldenCase { text: "张三是人物。", expected: Verdict::Entailed, tag: "type" },
    GoldenCase { text: "张三是地点。", expected: Verdict::Contradicted, tag: "type" },
    GoldenCase { text: "赵六的师傅是李四。", expected: Verdict::Unknown, tag: "unknown_entity" },
    GoldenCase { text: "张山的师傅是李四。", expected: Verdict::Entailed, tag: "fuzzy_link" },
    GoldenCase { text: "张三的老师是李四。", expected: Verdict::Entailed, tag: "synonym" },
    GoldenCase { text: "张三的岁数是30。", expected: Verdict::Entailed, tag: "synonym" },
];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntailmentEvalReport {
    pub version: u32,
    pub corpus_digest: String,
    pub cases: usize,
    pub passed_cases: usize,
    pub entailed_cases: usize,
    pub contradicted_cases: usize,
    pub unknown_cases: usize,
    pub metamorphic_checks: usize,
    pub metamorphic_passed: usize,
    pub failures: Vec<String>,
}

impl EntailmentEvalReport {
    pub fn gold_accuracy(&self) -> f64 {
        ratio(self.passed_cases, self.cases)
    }

    pub fn metamorphic_pass_rate(&self) -> f64 {
        ratio(self.metamorphic_passed, self.metamorphic_checks)
    }

    pub fn passed(&self) -> bool {
        self.failures.is_empty() && self.passed_cases == self.cases
    }

    pub fn metrics(&self) -> BTreeMap<String, f64> {
        let mut metrics = BTreeMap::new();
        metrics.insert("gold.accuracy".to_string(), self.gold_accuracy());
        metrics.insert("gold.entailed_rate".to_string(), ratio(self.entailed_cases, self.cases));
        metrics.insert(
            "gold.contradicted_rate".to_string(),
            ratio(self.contradicted_cases, self.cases),
        );
        metrics.insert("gold.unknown_rate".to_string(), ratio(self.unknown_cases, self.cases));
        metrics.insert(
            "metamorphic.pass_rate".to_string(),
            self.metamorphic_pass_rate(),
        );
        metrics
    }

    pub fn digest(&self) -> String {
        let mut values = serde_json::Map::new();
        values.insert("version".to_string(), serde_json::json!(self.version));
        values.insert("corpus".to_string(), serde_json::json!(self.corpus_digest));
        values.insert("cases".to_string(), serde_json::json!(self.cases));
        values.insert(
            "passed".to_string(),
            serde_json::json!(self.passed_cases),
        );
        values.insert(
            "metamorphic".to_string(),
            serde_json::json!(self.metamorphic_passed),
        );
        values.insert("failures".to_string(), serde_json::json!(self.failures));
        fnv1a_hex(canonical_json(&serde_json::Value::Object(values)).as_bytes())
    }

    pub fn render_text(&self) -> String {
        let mut lines = vec![
            "语义蕴含评测（entailment eval）".to_string(),
            format!(
                "金标：{}/{} 通过（准确率 {:.3}）；蕴含 {} / 矛盾 {} / 未知 {}",
                self.passed_cases,
                self.cases,
                self.gold_accuracy(),
                self.entailed_cases,
                self.contradicted_cases,
                self.unknown_cases
            ),
            format!(
                "变形关系：{}/{} 通过（{:.3}）",
                self.metamorphic_passed,
                self.metamorphic_checks,
                self.metamorphic_pass_rate()
            ),
        ];
        for failure in &self.failures {
            lines.push(format!("  - {}", failure));
        }
        lines.join("\n")
    }
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        return 0.0;
    }
    numerator as f64 / denominator as f64
}

fn corpus_digest() -> String {
    let cases: Vec<serde_json::Value> = GOLDEN
        .iter()
        .map(|case| {
            serde_json::json!({
                "text": case.text,
                "expected": case.expected.label(),
                "tag": case.tag,
            })
        })
        .collect();
    fnv1a_hex(canonical_json(&serde_json::json!(cases)).as_bytes())
}

fn run_gold(fixture: &Fixture) -> (usize, usize, usize, usize, Vec<String>) {
    let mut passed = 0usize;
    let mut entailed = 0usize;
    let mut contradicted = 0usize;
    let mut unknown = 0usize;
    let mut failures = Vec::new();
    for case in GOLDEN {
        let results = fixture.results(case.text);
        let actual = results
            .first()
            .map(|result| result.verdict)
            .unwrap_or(Verdict::Unknown);
        match actual {
            Verdict::Entailed => entailed += 1,
            Verdict::Contradicted => contradicted += 1,
            Verdict::Unknown => unknown += 1,
        }
        if actual == case.expected && !results.is_empty() {
            passed += 1;
        } else {
            failures.push(format!(
                "[{}] 「{}」期望「{}」，实际「{}」",
                case.tag,
                case.text,
                case.expected.label(),
                actual.label()
            ));
        }
    }
    (passed, entailed, contradicted, unknown, failures)
}

fn check_determinism(fixture: &Fixture) -> Result<(), String> {
    for case in GOLDEN {
        let first = fixture.verdict(case.text);
        let second = fixture.verdict(case.text);
        if first != second {
            return Err(format!(
                "确定性：同一输入两次判定不一致（{} vs {}）：「{}」",
                first.label(),
                second.label(),
                case.text
            ));
        }
    }
    Ok(())
}

fn check_monotonicity(fixture: &Fixture) -> Result<(), String> {
    let before: Vec<Verdict> = GOLDEN.iter().map(|case| fixture.verdict(case.text)).collect();
    let mut grown = Fixture::new();
    grown
        .predictor
        .add_triple("王五", "出生地", "北京");
    grown
        .generator
        .observe_triple("王五", "出生地", "北京");
    let after: Vec<Verdict> = GOLDEN.iter().map(|case| grown.verdict(case.text)).collect();
    for (case, (old, new)) in GOLDEN.iter().zip(before.iter().zip(after.iter())) {
        let allowed = *old == *new
            || (*old == Verdict::Unknown && *new == Verdict::Entailed)
            || (*old == Verdict::Unknown && *new == Verdict::Contradicted);
        if !allowed {
            return Err(format!(
                "知识单调增改变了既有判定（{} → {}）：「{}」",
                old.label(),
                new.label(),
                case.text
            ));
        }
    }
    Ok(())
}

fn check_retraction(fixture: &Fixture) -> Result<(), String> {
    let with_friend = fixture.verdict("张三的朋友是王五。");
    if with_friend != Verdict::Entailed {
        return Err(format!(
            "支持撤回：前提事实未被蕴含（实际 {}）",
            with_friend.label()
        ));
    }
    let without = Fixture::without_friend();
    match without.verdict("张三的朋友是王五。") {
        Verdict::Unknown | Verdict::Contradicted => Ok(()),
        verdict => Err(format!(
            "撤回事实后仍判「{}」，知识没有真正失效",
            verdict.label()
        )),
    }
}

fn check_negation_duality(fixture: &Fixture) -> Result<(), String> {
    for case in GOLDEN {
        if case.expected != Verdict::Entailed
            || case.tag == "type"
            || case.tag == "closed_world"
            || case.tag == "fuzzy_link"
            || case.tag == "synonym"
        {
            continue;
        }
        let negative = negative_form(case.text);
        let verdict = fixture.verdict(&negative);
        if verdict != Verdict::Contradicted {
            return Err(format!(
                "否定对偶：肯定句蕴含时否定句应为矛盾，实际「{}」：「{}」",
                verdict.label(),
                negative
            ));
        }
    }
    Ok(())
}

fn negative_form(text: &str) -> String {
    if let Some(rest) = text.strip_prefix("张三的") {
        for predicate in ["师傅", "朋友", "出生地"] {
            if let Some(tail) = rest.strip_prefix(predicate) {
                return format!("张三的{}不是{}", predicate, tail.trim_start_matches('是'));
            }
        }
    }
    text.replace("是", "不是")
}

fn check_order_independence(fixture: &Fixture) -> Result<(), String> {
    let reversed = Fixture::reversed();
    for case in GOLDEN {
        let original = fixture.verdict(case.text);
        let permuted = reversed.verdict(case.text);
        if original != permuted {
            return Err(format!(
                "插入顺序无关性：原序「{}」vs 逆序「{}」：「{}」",
                original.label(),
                permuted.label(),
                case.text
            ));
        }
    }
    Ok(())
}

pub fn evaluate_seed() -> EntailmentEvalReport {
    let fixture = Fixture::new();
    let (passed, entailed, contradicted, unknown, mut failures) = run_gold(&fixture);
    let checks: [(&str, Result<(), String>); 5] = [
        ("determinism", check_determinism(&fixture)),
        ("monotonicity", check_monotonicity(&fixture)),
        ("retraction", check_retraction(&fixture)),
        ("negation_duality", check_negation_duality(&fixture)),
        ("order_independence", check_order_independence(&fixture)),
    ];
    let mut metamorphic_passed = 0usize;
    for (name, outcome) in checks {
        match outcome {
            Ok(()) => metamorphic_passed += 1,
            Err(error) => failures.push(format!("[{}] {}", name, error)),
        }
    }
    EntailmentEvalReport {
        version: ENTAILMENT_EVAL_VERSION,
        corpus_digest: corpus_digest(),
        cases: GOLDEN.len(),
        passed_cases: passed,
        entailed_cases: entailed,
        contradicted_cases: contradicted,
        unknown_cases: unknown,
        metamorphic_checks: 5,
        metamorphic_passed,
        failures,
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntailmentEvalBaseline {
    pub version: u32,
    pub corpus_digest: String,
    pub metrics: BTreeMap<String, f64>,
}

impl EntailmentEvalBaseline {
    pub fn from_report(report: &EntailmentEvalReport) -> Self {
        Self {
            version: ENTAILMENT_EVAL_VERSION,
            corpus_digest: report.corpus_digest.clone(),
            metrics: report.metrics(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EntailmentMetricDelta {
    pub metric: String,
    pub baseline: f64,
    pub current: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EntailmentRatchetVerdict {
    pub passed: bool,
    pub digest_matches: bool,
    pub regressions: Vec<EntailmentMetricDelta>,
    pub gains: Vec<EntailmentMetricDelta>,
}

pub fn compare_baseline(
    baseline: &EntailmentEvalBaseline,
    current: &EntailmentEvalReport,
    epsilon: f64,
) -> EntailmentRatchetVerdict {
    let digest_matches = baseline.corpus_digest == current.corpus_digest;
    let current_metrics = current.metrics();
    let mut regressions = Vec::new();
    let mut gains = Vec::new();
    for (metric, baseline_value) in &baseline.metrics {
        let current_value = current_metrics.get(metric).copied().unwrap_or(0.0);
        if current_value < baseline_value - epsilon {
            regressions.push(EntailmentMetricDelta {
                metric: metric.clone(),
                baseline: *baseline_value,
                current: current_value,
            });
        } else if current_value > baseline_value + epsilon {
            gains.push(EntailmentMetricDelta {
                metric: metric.clone(),
                baseline: *baseline_value,
                current: current_value,
            });
        }
    }
    EntailmentRatchetVerdict {
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
    fn golden_corpus_is_fully_satisfied() {
        let report = evaluate_seed();
        assert_eq!(
            report.passed_cases,
            report.cases,
            "{}",
            report.render_text()
        );
        assert!(report.passed(), "{}", report.render_text());
        assert!(report.gold_accuracy() >= 1.0);
    }

    #[test]
    fn metamorphic_relations_hold() {
        let report = evaluate_seed();
        assert_eq!(
            report.metamorphic_passed,
            report.metamorphic_checks,
            "{}",
            report.render_text()
        );
    }

    #[test]
    fn report_digest_is_deterministic() {
        let first = evaluate_seed();
        let second = evaluate_seed();
        assert_eq!(first, second);
        assert_eq!(first.digest(), second.digest());
    }

    #[test]
    fn baseline_ratchet_detects_regression_and_drift() {
        let report = evaluate_seed();
        let baseline = EntailmentEvalBaseline::from_report(&report);
        let verdict = compare_baseline(&baseline, &report, 1e-9);
        assert!(verdict.passed, "{:?}", verdict);

        let mut worsened = report.clone();
        worsened.passed_cases = worsened.passed_cases.saturating_sub(1);
        let verdict = compare_baseline(&baseline, &worsened, 1e-9);
        assert!(!verdict.passed);
        assert!(!verdict.regressions.is_empty());

        let mut drifted = report.clone();
        drifted.corpus_digest = "drifted".to_string();
        let verdict = compare_baseline(&baseline, &drifted, 1e-9);
        assert!(!verdict.passed);
        assert!(!verdict.digest_matches);
    }
}

//! FCJP：帧约束联合解析（Frame-Constrained Joint Parsing）。
//!
//! 现有链路是 `文本 → 分词 → 关系帧` 的单向流水线：分词的一次性错误会被
//! 关系抽取完整继承，关系帧携带的定义域/值域、参数跨度与极性却回不到分词层。
//! 本模块补上反向通道，并把「分析」与「综合」放在同一个可审计中间表示上：
//!
//! 1. 先抽原始关系帧 `F0`（带字符跨度、极性、模态、谓词）；
//! 2. 用 `F0` 的谓词定义域/值域生成 `SegmentOptions` 软约束；
//! 3. 用 `ContextualSegmenter` 重新分词，得到候选读法；
//! 4. 用帧参数跨度把 OOV 参数「焊」成原子 token（只在本模块内，不回写词表）；
//! 5. 用 token 读法归一化帧参数，用 `SchemaView::edge_verdict` 做类型裁决；
//! 6. 若参数未对齐/类型互斥/歧义未决，把失败帧的约束回灌，最多 3 轮；
//! 7. 输出 `JointParse`：分词、原始帧、归一化帧、对齐证据、类型理由、
//!    候选新词、歧义与轮次轨迹。
//!
//! 设计约束：纯符号、零训练、同输入同字节；只读诊断，不写知识图谱；
//! 候选新词只暂存在本次解析结果里，用户确认后才应进入读法/词法记忆。

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use serde::{Deserialize, Serialize};

use crate::context_image::fnv1a_hex;
use crate::contextual_segmentation::{
    ContextualSegmenter, ReadingKind, ReadingScore, SegmentOptions, SegmentResult, SegmentedToken,
};
use crate::relation_frame::{FrameExtractor, Polarity, RelationFrame};
use crate::schema_view::SchemaView;
use crate::type_system::TypeVerdict;

/// 联合解析结果方言标识。
pub const JOINT_PARSE_DIALECT: &str = "knowprolog-fcjp/1";
/// 默认总轮次（基础分词 + 帧约束重分词 + 一次失败回灌）。
pub const DEFAULT_MAX_ROUNDS: usize = 3;
/// 默认参与回灌的最低帧置信度。
pub const DEFAULT_MIN_FRAME_CONFIDENCE: f64 = 0.5;
/// 参数最多允许焊接的字符数，防止把整句吞成一个 token。
pub const DEFAULT_MAX_ARGUMENT_CHARS: usize = 16;

const NORMALIZE_BONUS: f64 = 0.20;
const COMPATIBLE_BONUS: f64 = 0.30;
const ALIGNED_BONUS: f64 = 0.40;
const CONFLICT_PENALTY: f64 = 1.50;
const AMBIGUITY_MARGIN: f64 = 0.50;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArgumentRole {
    Subject,
    Object,
}

impl ArgumentRole {
    pub fn label(self) -> &'static str {
        match self {
            ArgumentRole::Subject => "主语",
            ArgumentRole::Object => "宾语",
        }
    }
}

/// 显式联合上下文：不引入隐向量，每个字段都可打印、可回滚、可测试。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct JointContext {
    pub focus_entities: Vec<String>,
    pub expected_types: Vec<String>,
    pub last_predicate: Option<String>,
}

impl JointContext {
    pub fn new(
        focus_entities: Vec<String>,
        expected_types: Vec<String>,
        last_predicate: Option<String>,
    ) -> Self {
        let mut context = Self {
            focus_entities: Vec::new(),
            expected_types: Vec::new(),
            last_predicate: None,
        };
        context.observe_focus(&focus_entities);
        context.observe_types(&expected_types);
        context.last_predicate = last_predicate
            .map(|predicate| predicate.trim().to_string())
            .filter(|predicate| !predicate.is_empty());
        context
    }

    fn observe_focus(&mut self, entities: &[String]) {
        for entity in entities {
            let entity = entity.trim();
            if entity.is_empty() || self.focus_entities.iter().any(|known| known == entity) {
                continue;
            }
            self.focus_entities.push(entity.to_string());
        }
        self.focus_entities.truncate(5);
    }

    fn observe_types(&mut self, types: &[String]) {
        for etype in types {
            let etype = etype.trim();
            if etype.is_empty() || self.expected_types.iter().any(|known| known == etype) {
                continue;
            }
            self.expected_types.push(etype.to_string());
        }
        self.expected_types.sort();
        self.expected_types.dedup();
    }

    /// 把解析结果回写为下一轮种子：焦点实体、类型证据、最高置信谓词。
    /// 只回写没有硬冲突的帧，避免类型互斥/未对齐的坏帧污染下一轮语境。
    pub fn observe_parse(&mut self, frames: &[JointFrame]) {
        let healthy: Vec<&JointFrame> = frames
            .iter()
            .filter(|frame| !frame.is_hard_failure())
            .collect();
        if healthy.is_empty() {
            return;
        }
        let mut entities = Vec::new();
        let mut types = Vec::new();
        for frame in &healthy {
            entities.push(frame.subject.canonical.clone());
            entities.push(frame.object.canonical.clone());
            types.extend(frame.subject_types.iter().cloned());
            types.extend(frame.object_types.iter().cloned());
        }
        self.observe_focus(&entities);
        self.observe_types(&types);
        if let Some(frame) = healthy.iter().max_by(|left, right| {
            left.original
                .confidence
                .total_cmp(&right.original.confidence)
                .then_with(|| right.original.id.cmp(&left.original.id))
        }) {
            self.last_predicate = Some(frame.normalized.predicate.clone());
        }
    }

    fn seed_options(&self, ambiguity_margin: f64) -> SegmentOptions {
        SegmentOptions {
            context_terms: self.focus_entities.clone(),
            expected_types: self.expected_types.clone(),
            predicate: self.last_predicate.clone(),
            ambiguity_margin,
            ..SegmentOptions::default()
        }
    }
}

/// 联合解析选项；`type_hints` 由宿主注入（图谱实体类型/链接器/本体），
/// 模块本身不读存储，因此单元测试可以完全离线复现。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointParseOptions {
    pub context: JointContext,
    pub type_hints: BTreeMap<String, BTreeSet<String>>,
    pub max_rounds: usize,
    pub min_frame_confidence: f64,
    pub max_argument_chars: usize,
    pub ambiguity_margin: f64,
}

impl Default for JointParseOptions {
    fn default() -> Self {
        Self {
            context: JointContext::default(),
            type_hints: BTreeMap::new(),
            max_rounds: DEFAULT_MAX_ROUNDS,
            min_frame_confidence: DEFAULT_MIN_FRAME_CONFIDENCE,
            max_argument_chars: DEFAULT_MAX_ARGUMENT_CHARS,
            ambiguity_margin: crate::contextual_segmentation::DEFAULT_AMBIGUITY_MARGIN,
        }
    }
}

impl JointParseOptions {
    pub fn with_context(mut self, context: JointContext) -> Self {
        self.context = context;
        self
    }

    pub fn with_type_hints(mut self, hints: BTreeMap<String, BTreeSet<String>>) -> Self {
        self.type_hints = hints;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JointArgument {
    pub role: ArgumentRole,
    pub surface: String,
    pub span: Option<(usize, usize)>,
    pub canonical: String,
    pub token_span: Option<(usize, usize)>,
    pub types: BTreeSet<String>,
    pub evidence: Vec<String>,
    pub welded: bool,
}

impl JointArgument {
    fn unresolved(role: ArgumentRole, surface: &str, span: Option<(usize, usize)>) -> Self {
        Self {
            role,
            surface: surface.to_string(),
            span,
            canonical: surface.to_string(),
            token_span: None,
            types: BTreeSet::new(),
            evidence: Vec::new(),
            welded: false,
        }
    }

    pub fn normalized(&self) -> bool {
        self.canonical != self.surface
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JointFrame {
    pub original: RelationFrame,
    pub normalized: RelationFrame,
    pub subject: JointArgument,
    pub object: JointArgument,
    pub subject_types: BTreeSet<String>,
    pub object_types: BTreeSet<String>,
    pub subject_verdict: TypeVerdict,
    pub object_verdict: TypeVerdict,
    pub overall_verdict: TypeVerdict,
    pub reasons: Vec<String>,
    pub ambiguities: Vec<String>,
    pub conflicts: Vec<String>,
    pub aligned: bool,
    pub welded: bool,
}

impl JointFrame {
    pub fn is_hard_failure(&self) -> bool {
        !self.conflicts.is_empty()
    }

    pub fn render_text(&self) -> String {
        self.normalized.render_text()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NewWordCandidate {
    pub surface: String,
    pub span: (usize, usize),
    pub role: ArgumentRole,
    pub evidence: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JointRound {
    pub round: usize,
    pub digest: String,
    pub score: f64,
    pub frame_count: usize,
    pub aligned: usize,
    pub normalized: usize,
    pub compatible: usize,
    pub conflicts: usize,
    pub predicate: Option<String>,
    pub expected_types: Vec<String>,
    pub context_fingerprint: String,
    pub segmentation: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JointParse {
    pub dialect: String,
    pub text: String,
    pub base: SegmentResult,
    pub segmentation: SegmentResult,
    pub frames: Vec<JointFrame>,
    pub rounds: Vec<JointRound>,
    pub candidates: Vec<NewWordCandidate>,
    pub ambiguities: Vec<String>,
    pub context: JointContext,
    pub score: f64,
    pub converged: bool,
}

impl JointParse {
    pub fn render_report(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "联合解析（FCJP：帧约束重分词 + 本体类型裁决 + 读法归一；纯符号、零训练、只读）"
        );
        let _ = writeln!(
            out,
            "轮次：{}；收敛：{}；帧：{}；候选新词：{}；方言：{}",
            self.rounds.len(),
            if self.converged { "是" } else { "否" },
            self.frames.len(),
            self.candidates.len(),
            self.dialect
        );
        let _ = writeln!(out, "基础分词：{}", self.base.render_segmentation());
        let _ = writeln!(out, "选定分词：{}", self.segmentation.render_segmentation());
        let _ = writeln!(
            out,
            "选定语境：{}（期望谓词 {}；期望类型 {}）",
            if self.segmentation.context.fingerprint.is_empty() {
                "∅".to_string()
            } else {
                self.segmentation.context.fingerprint.clone()
            },
            self.segmentation
                .context
                .predicate
                .as_deref()
                .unwrap_or("∅"),
            if self.segmentation.context.expected_types.is_empty() {
                "∅".to_string()
            } else {
                self.segmentation.context.expected_types.join("、")
            }
        );
        out.push_str("帧：\n");
        if self.frames.is_empty() {
            out.push_str("  （未抽到关系帧）\n");
        }
        for (index, frame) in self.frames.iter().enumerate() {
            let _ = writeln!(
                out,
                "  {}. {}（原始：{}；置信度 {:.2}）",
                index + 1,
                frame.normalized.render_text(),
                frame.original.render_text(),
                frame.original.confidence
            );
            render_argument(&mut out, "主语", &frame.subject, frame.subject_verdict);
            render_argument(&mut out, "宾语", &frame.object, frame.object_verdict);
            let _ = writeln!(
                out,
                "     类型裁决：{}（主语 {}；宾语 {}）",
                type_verdict_label(frame.overall_verdict),
                type_verdict_label(frame.subject_verdict),
                type_verdict_label(frame.object_verdict)
            );
            if !frame.reasons.is_empty() {
                for reason in &frame.reasons {
                    let _ = writeln!(out, "     理由：{}", reason);
                }
            }
            for ambiguity in &frame.ambiguities {
                let _ = writeln!(out, "     歧义：{}", ambiguity);
            }
            for conflict in &frame.conflicts {
                let _ = writeln!(out, "     冲突：{}", conflict);
            }
        }
        if !self.candidates.is_empty() {
            out.push_str("候选新词（仅本次诊断，不写记忆）：\n");
            for candidate in &self.candidates {
                let _ = writeln!(
                    out,
                    "  {}[{},{}) {}；{}",
                    candidate.surface,
                    candidate.span.0,
                    candidate.span.1,
                    candidate.role.label(),
                    candidate.evidence.join("；")
                );
            }
        }
        if !self.ambiguities.is_empty() {
            out.push_str("歧义：\n");
            for ambiguity in &self.ambiguities {
                let _ = writeln!(out, "  {}", ambiguity);
            }
        }
        let _ = writeln!(
            out,
            "回写上下文：焦点 {}；类型 {}；上一谓词 {}",
            if self.context.focus_entities.is_empty() {
                "∅".to_string()
            } else {
                self.context.focus_entities.join("、")
            },
            if self.context.expected_types.is_empty() {
                "∅".to_string()
            } else {
                self.context.expected_types.join("、")
            },
            self.context.last_predicate.as_deref().unwrap_or("∅")
        );
        out.push_str("轮次轨迹：\n");
        for round in &self.rounds {
            let _ = writeln!(
                out,
                "  #{} 分数={:.4} 帧={} 对齐={} 归一={} 相容={} 冲突={} 谓词={} 类型={} 分词={} 指纹={}",
                round.round,
                round.score,
                round.frame_count,
                round.aligned,
                round.normalized,
                round.compatible,
                round.conflicts,
                round.predicate.as_deref().unwrap_or("∅"),
                if round.expected_types.is_empty() {
                    "∅".to_string()
                } else {
                    round.expected_types.join("、")
                },
                round.segmentation.join(" | "),
                if round.context_fingerprint.is_empty() {
                    "∅".to_string()
                } else {
                    round.context_fingerprint.clone()
                }
            );
        }
        out
    }
}

fn render_argument(out: &mut String, label: &str, argument: &JointArgument, verdict: TypeVerdict) {
    let span = argument
        .span
        .map(|(start, end)| format!("[{},{})", start, end))
        .unwrap_or_else(|| "[未对齐]".to_string());
    let types = if argument.types.is_empty() {
        "∅".to_string()
    } else {
        argument
            .types
            .iter()
            .cloned()
            .collect::<Vec<_>>()
            .join("、")
    };
    let _ = writeln!(
        out,
        "     {}：{}{} → {}；类型 {{{}}}；裁决 {}；{}",
        label,
        argument.surface,
        span,
        argument.canonical,
        types,
        type_verdict_label(verdict),
        if argument.welded {
            "帧锚定焊接"
        } else {
            "原生 token"
        }
    );
    for evidence in &argument.evidence {
        let _ = writeln!(out, "       证据：{}", evidence);
    }
}

fn type_verdict_label(verdict: TypeVerdict) -> &'static str {
    match verdict {
        TypeVerdict::Compatible => "相容",
        TypeVerdict::Disjoint => "互斥",
        TypeVerdict::Unknown => "未知（开放世界）",
    }
}

/// 对一段文本执行联合解析：先抽帧，再让帧约束分词，最后做类型裁决。
pub fn joint_parse(
    segmenter: &mut ContextualSegmenter,
    schema: &SchemaView,
    text: &str,
    options: &JointParseOptions,
) -> JointParse {
    let frames = FrameExtractor::new().extract(text);
    joint_parse_with_frames(segmenter, schema, text, frames, options)
}

/// 与 [`joint_parse`] 相同，但接受外部帧（测试/生成回译验证可注入候选帧）。
pub fn joint_parse_with_frames(
    segmenter: &mut ContextualSegmenter,
    schema: &SchemaView,
    text: &str,
    frames: Vec<RelationFrame>,
    options: &JointParseOptions,
) -> JointParse {
    let text = text.to_string();
    let base_options = options.context.seed_options(options.ambiguity_margin);
    let base = segmenter.segment(&text, &base_options);
    if text.trim().is_empty() {
        return empty_parse(text, base, options);
    }

    let max_rounds = options.max_rounds.clamp(1, 6);
    let mut evaluated: Vec<(
        JointRound,
        SegmentResult,
        Vec<JointFrame>,
        Vec<NewWordCandidate>,
    )> = Vec::new();
    let mut seen_digests = BTreeSet::new();
    let mut current_options = base_options.clone();
    let mut current_result = base.clone();
    let mut round_index = 0usize;
    let mut next_from_failure = false;

    loop {
        let (round_frames, candidates, hard_failures) = evaluate_frames(
            segmenter,
            schema,
            &text,
            &frames,
            &mut current_result,
            options,
        );
        let score = round_score(&current_result, &round_frames);
        let digest = round_digest(round_index, &current_options, &round_frames);
        if !seen_digests.insert(digest.clone()) {
            break;
        }
        let round = JointRound {
            round: round_index,
            digest,
            score,
            frame_count: round_frames.len(),
            aligned: round_frames.iter().filter(|frame| frame.aligned).count(),
            normalized: round_frames
                .iter()
                .filter(|frame| frame.subject.normalized() || frame.object.normalized())
                .count(),
            compatible: round_frames
                .iter()
                .filter(|frame| frame.overall_verdict == TypeVerdict::Compatible)
                .count(),
            conflicts: hard_failures,
            predicate: current_options.predicate.clone(),
            expected_types: current_options.expected_types.clone(),
            context_fingerprint: current_result.context.fingerprint.clone(),
            segmentation: current_result
                .tokens
                .iter()
                .map(|token| token.text.clone())
                .collect(),
        };
        evaluated.push((round, current_result.clone(), round_frames, candidates));

        if round_index + 1 >= max_rounds {
            break;
        }
        let next = if hard_failures == 0 && !next_from_failure {
            frame_constraints(
                &frames,
                &options.context,
                options.min_frame_confidence,
                schema,
            )
        } else {
            failure_constraints(
                &evaluated.last().expect("just pushed").2,
                &options.context,
                schema,
            )
        };
        let Some(mut next) = next else {
            break;
        };
        next.ambiguity_margin = options.ambiguity_margin;
        if next.context_terms == current_options.context_terms
            && next.expected_types == current_options.expected_types
            && next.predicate == current_options.predicate
        {
            break;
        }
        current_options = next;
        current_result = segmenter.segment(&text, &current_options);
        round_index += 1;
        next_from_failure = true;
    }

    let rounds: Vec<JointRound> = evaluated
        .iter()
        .map(|(round, _, _, _)| round.clone())
        .collect();
    let (best_round, best_result, best_frames, best_candidates) = evaluated
        .into_iter()
        .max_by(|left, right| {
            left.0
                .score
                .total_cmp(&right.0.score)
                .then_with(|| right.0.round.cmp(&left.0.round))
        })
        .expect("at least one round");

    let mut context = options.context.clone();
    context.observe_parse(&best_frames);
    let mut ambiguities: Vec<String> = best_frames
        .iter()
        .flat_map(|frame| frame.ambiguities.iter().cloned())
        .collect();
    if let Some(report) = &best_result.ambiguity {
        if let Some(question) = &report.clarification {
            ambiguities.push(question.clone());
        }
    }
    ambiguities.sort();
    ambiguities.dedup();

    let mut candidates = best_candidates;
    candidates.sort_by(|left, right| {
        left.span
            .cmp(&right.span)
            .then_with(|| left.surface.cmp(&right.surface))
            .then_with(|| left.role.cmp(&right.role))
    });
    candidates.dedup_by(|left, right| left.span == right.span && left.surface == right.surface);

    let converged = best_frames.iter().all(|frame| !frame.is_hard_failure());
    let score = best_round.score;
    JointParse {
        dialect: JOINT_PARSE_DIALECT.to_string(),
        text,
        base,
        segmentation: best_result,
        frames: best_frames,
        rounds,
        candidates,
        ambiguities,
        context,
        score,
        converged,
    }
}

fn empty_parse(text: String, base: SegmentResult, options: &JointParseOptions) -> JointParse {
    JointParse {
        dialect: JOINT_PARSE_DIALECT.to_string(),
        text,
        segmentation: base.clone(),
        base,
        frames: Vec::new(),
        rounds: Vec::new(),
        candidates: Vec::new(),
        ambiguities: Vec::new(),
        context: options.context.clone(),
        score: 0.0,
        converged: true,
    }
}

fn round_score(result: &SegmentResult, frames: &[JointFrame]) -> f64 {
    let aligned = frames.iter().filter(|frame| frame.aligned).count() as f64;
    let normalized = frames
        .iter()
        .filter(|frame| frame.subject.normalized() || frame.object.normalized())
        .count() as f64;
    let compatible = frames
        .iter()
        .filter(|frame| frame.overall_verdict == TypeVerdict::Compatible)
        .count() as f64;
    let conflicts = frames
        .iter()
        .filter(|frame| frame.is_hard_failure())
        .count() as f64;
    result.score
        + ALIGNED_BONUS * aligned
        + NORMALIZE_BONUS * normalized
        + COMPATIBLE_BONUS * compatible
        - CONFLICT_PENALTY * conflicts
}

fn round_digest(round: usize, options: &SegmentOptions, frames: &[JointFrame]) -> String {
    let mut material = format!(
        "{}|{}|{}|{}",
        round,
        options.context_terms.join("\u{1f}"),
        options.expected_types.join("\u{1f}"),
        options.predicate.as_deref().unwrap_or("")
    );
    for frame in frames {
        let _ = write!(
            material,
            "|{}>{}>{}>{}",
            frame.normalized.subject,
            frame.normalized.predicate,
            frame.normalized.object,
            frame.conflicts.len()
        );
    }
    fnv1a_hex(material.as_bytes())
}

fn frame_constraints(
    frames: &[RelationFrame],
    context: &JointContext,
    min_confidence: f64,
    schema: &SchemaView,
) -> Option<SegmentOptions> {
    let anchor = frames
        .iter()
        .filter(|frame| frame.confidence >= min_confidence)
        .max_by(|left, right| {
            left.confidence
                .total_cmp(&right.confidence)
                .then_with(|| right.id.cmp(&left.id))
        })?;
    let mut terms = context.focus_entities.clone();
    terms.push(anchor.subject.clone());
    terms.push(anchor.object.clone());
    let mut expected = context.expected_types.clone();
    expected.extend(frame_domain_range(schema, anchor));
    Some(SegmentOptions {
        context_terms: terms,
        expected_types: expected,
        predicate: Some(anchor.predicate.clone()),
        ..SegmentOptions::default()
    })
}

fn failure_constraints(
    frames: &[JointFrame],
    context: &JointContext,
    schema: &SchemaView,
) -> Option<SegmentOptions> {
    let failed = frames
        .iter()
        .filter(|frame| frame.is_hard_failure())
        .max_by(|left, right| {
            left.original
                .confidence
                .total_cmp(&right.original.confidence)
                .then_with(|| right.original.id.cmp(&left.original.id))
        })?;
    let mut terms = context.focus_entities.clone();
    terms.push(failed.subject.canonical.clone());
    terms.push(failed.object.canonical.clone());
    let mut expected = context.expected_types.clone();
    expected.extend(frame_domain_range(schema, &failed.original));
    Some(SegmentOptions {
        context_terms: terms,
        expected_types: expected,
        predicate: Some(failed.normalized.predicate.clone()),
        ..SegmentOptions::default()
    })
}

fn frame_domain_range(schema: &SchemaView, frame: &RelationFrame) -> Vec<String> {
    let predicate = schema.canonical_property(&frame.predicate);
    let mut types: Vec<String> = schema.domain_classes(&predicate).into_iter().collect();
    types.extend(schema.range_classes(&predicate));
    types.sort();
    types.dedup();
    types
}

fn evaluate_frames(
    segmenter: &ContextualSegmenter,
    schema: &SchemaView,
    text: &str,
    frames: &[RelationFrame],
    result: &mut SegmentResult,
    options: &JointParseOptions,
) -> (Vec<JointFrame>, Vec<NewWordCandidate>, usize) {
    let mut requests: Vec<FrameRequests> = Vec::with_capacity(frames.len());
    for frame in frames {
        let subject = argument_request(
            text,
            frame,
            ArgumentRole::Subject,
            options.max_argument_chars,
        );
        let object = argument_request(
            text,
            frame,
            ArgumentRole::Object,
            options.max_argument_chars,
        );
        requests.push(FrameRequests { subject, object });
    }

    let mut weld_log: BTreeMap<(usize, usize), WeldRecord> = BTreeMap::new();
    let mut unique: BTreeMap<(usize, usize), String> = BTreeMap::new();
    for request in &requests {
        for argument in [&request.subject, &request.object] {
            if let (Some(span), None) = (argument.span, &argument.error) {
                unique
                    .entry(span)
                    .or_insert_with(|| argument.surface.clone());
            }
        }
    }
    for (span, surface) in unique {
        let preserved = result
            .tokens
            .iter()
            .find(|token| token.start < span.1 && token.end > span.0 && token.text == surface)
            .or_else(|| {
                result
                    .tokens
                    .iter()
                    .find(|token| token.start < span.1 && token.end > span.0)
            })
            .cloned();
        let was_atomic = exact_token(result, span).is_some();
        let welded = weld_span(result, span, &surface, preserved.as_ref());
        weld_log.insert(
            span,
            WeldRecord {
                was_atomic: was_atomic && !welded,
            },
        );
    }

    let mut evaluated = Vec::with_capacity(frames.len());
    let mut candidates = Vec::new();
    let mut hard_failures = 0usize;
    for (frame, requests) in frames.iter().zip(requests) {
        let (joint, frame_candidates) = evaluate_frame(
            segmenter, schema, text, frame, requests, result, options, &weld_log,
        );
        if joint.is_hard_failure() {
            hard_failures += 1;
        }
        candidates.extend(frame_candidates);
        evaluated.push(joint);
    }
    (evaluated, candidates, hard_failures)
}

fn evaluate_frame(
    segmenter: &ContextualSegmenter,
    schema: &SchemaView,
    text: &str,
    frame: &RelationFrame,
    requests: FrameRequests,
    result: &mut SegmentResult,
    options: &JointParseOptions,
    weld_log: &BTreeMap<(usize, usize), WeldRecord>,
) -> (JointFrame, Vec<NewWordCandidate>) {
    let predicate = schema.canonical_property(&frame.predicate);
    let subject_slot = schema.domain_classes(&predicate);
    let object_slot = schema.range_classes(&predicate);
    let mut reasons = Vec::new();
    let mut ambiguities = Vec::new();
    let mut conflicts = Vec::new();

    let mut subject = resolve_argument(
        segmenter,
        schema,
        &requests.subject,
        &subject_slot,
        &options.type_hints,
        result,
        &mut reasons,
        &mut ambiguities,
        &mut conflicts,
        weld_log,
    );
    let mut object = resolve_argument(
        segmenter,
        schema,
        &requests.object,
        &object_slot,
        &options.type_hints,
        result,
        &mut reasons,
        &mut ambiguities,
        &mut conflicts,
        weld_log,
    );

    let subject_types = argument_types(
        &subject.canonical,
        &subject.surface,
        &options.type_hints,
        result,
        subject.token_span,
    );
    let object_types = argument_types(
        &object.canonical,
        &object.surface,
        &options.type_hints,
        result,
        object.token_span,
    );
    subject.types = subject_types.clone();
    object.types = object_types.clone();

    let edge = schema.edge_verdict(&subject_types, &predicate, &object_types);
    reasons.extend(edge.reasons.clone());
    if edge.overall == TypeVerdict::Disjoint {
        conflicts.push(format!(
            "类型互斥：{} 的主语/宾语类型与谓词「{}」的定义域/值域不相容",
            frame.render_text(),
            predicate
        ));
    }
    if subject_slot.is_empty() {
        reasons.push(format!("{}：未声明定义域，按开放世界放行", predicate));
    }
    if object_slot.is_empty() {
        reasons.push(format!("{}：未声明值域，按开放世界放行", predicate));
    }

    let mut normalized = RelationFrame::new(&subject.canonical, &predicate, &object.canonical);
    normalized.id = frame.id.clone();
    normalized.polarity = frame.polarity;
    normalized.modality = frame.modality;
    normalized.temporal = frame.temporal.clone();
    normalized.qualifiers = frame.qualifiers.clone();
    normalized.evidence = frame.evidence.clone();
    normalized.extraction = frame.extraction.clone();
    normalized.confidence = frame.confidence;

    let aligned = subject.token_span.is_some() && object.token_span.is_some();
    let welded = subject.welded || object.welded;
    let mut candidates = Vec::new();
    if let Some(candidate) = new_word_candidate(
        &subject,
        result,
        segmenter,
        &options.type_hints,
        frame.confidence,
        "主语",
    ) {
        candidates.push(candidate);
    }
    if let Some(candidate) = new_word_candidate(
        &object,
        result,
        segmenter,
        &options.type_hints,
        frame.confidence,
        "宾语",
    ) {
        candidates.push(candidate);
    }

    if subject.normalized() {
        reasons.push(format!(
            "读法归一：主语「{}」→「{}」",
            subject.surface, subject.canonical
        ));
    }
    if object.normalized() {
        reasons.push(format!(
            "读法归一：宾语「{}」→「{}」",
            object.surface, object.canonical
        ));
    }
    if welded {
        reasons.push("帧锚定：参数跨度与分词边界不一致，已在本次解析内焊为原子 token".to_string());
    }
    if !aligned {
        conflicts.push(format!(
            "参数未对齐：{} 的字符跨度无法落到 token 边界",
            frame.render_text()
        ));
    }
    if text.trim().is_empty() {
        conflicts.push("空文本".to_string());
    }

    (
        JointFrame {
            original: frame.clone(),
            normalized,
            subject,
            object,
            subject_types,
            object_types,
            subject_verdict: edge.subject,
            object_verdict: edge.object,
            overall_verdict: edge.overall,
            reasons,
            ambiguities,
            conflicts,
            aligned,
            welded,
        },
        candidates,
    )
}

#[allow(clippy::too_many_arguments)]
fn resolve_argument(
    segmenter: &ContextualSegmenter,
    schema: &SchemaView,
    request: &ArgumentRequest,
    slot_types: &BTreeSet<String>,
    hints: &BTreeMap<String, BTreeSet<String>>,
    result: &mut SegmentResult,
    reasons: &mut Vec<String>,
    ambiguities: &mut Vec<String>,
    conflicts: &mut Vec<String>,
    weld_log: &BTreeMap<(usize, usize), WeldRecord>,
) -> JointArgument {
    let mut argument = JointArgument::unresolved(request.role, &request.surface, request.span);
    if let Some(error) = &request.error {
        conflicts.push(format!(
            "{}「{}」{}",
            request.role.label(),
            request.surface,
            error
        ));
        return argument;
    }
    let Some(span) = request.span else {
        conflicts.push(format!(
            "{}「{}」没有字符跨度",
            request.role.label(),
            request.surface
        ));
        return argument;
    };
    let Some(token_index) = exact_token(result, span) else {
        conflicts.push(format!(
            "{}「{}」[{},{}) 未对齐到原子 token",
            request.role.label(),
            request.surface,
            span.0,
            span.1
        ));
        return argument;
    };
    let token = result.tokens[token_index].clone();
    let choice = select_reading(
        segmenter,
        schema,
        &request.surface,
        slot_types,
        hints,
        &token,
    );
    if let Some(ambiguity) = &choice.ambiguity {
        ambiguities.push(ambiguity.clone());
    }
    if choice.canonical != token.canonical
        || choice.kind != token.kind
        || choice.etype != token.etype
    {
        let token = &mut result.tokens[token_index];
        token.canonical = choice.canonical.clone();
        token.kind = choice.kind;
        token.etype = choice.etype.clone();
        token.evidence.extend(choice.evidence.iter().cloned());
        token.evidence.sort();
        token.evidence.dedup();
        reasons.push(format!(
            "{}读法：{} → {}（{}）",
            request.role.label(),
            request.surface,
            choice.canonical,
            choice.kind.label()
        ));
    }
    argument.canonical = choice.canonical;
    argument.token_span = Some((token.start, token.end));
    argument.evidence = choice.evidence;
    argument.welded = weld_log
        .get(&span)
        .map(|record| !record.was_atomic)
        .unwrap_or(false);
    argument
}

#[derive(Debug, Clone)]
struct ReadingChoice {
    canonical: String,
    kind: ReadingKind,
    etype: Option<String>,
    evidence: Vec<String>,
    ambiguity: Option<String>,
}

fn select_reading(
    segmenter: &ContextualSegmenter,
    schema: &SchemaView,
    surface: &str,
    slot_types: &BTreeSet<String>,
    hints: &BTreeMap<String, BTreeSet<String>>,
    token: &SegmentedToken,
) -> ReadingChoice {
    let mut candidates: Vec<(String, ReadingKind, Vec<String>)> = Vec::new();
    candidates.push((token.canonical.clone(), token.kind, token.evidence.clone()));
    for (canonical, kind) in segmenter.surface_readings(surface) {
        candidates.push((canonical, kind, Vec::new()));
    }
    candidates.retain(|(_, kind, _)| *kind != ReadingKind::Predicate);
    candidates.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
    candidates.dedup_by(|left, right| left.0 == right.0 && left.1 == right.1);

    let mut scored: Vec<(f64, String, ReadingKind, BTreeSet<String>, Vec<String>)> = Vec::new();
    for (canonical, kind, evidence) in candidates {
        let types = reading_types(&canonical, surface, kind, hints, token);
        let verdict = schema.sets_admit(&types, slot_types);
        let mut score = match verdict {
            TypeVerdict::Compatible => 3.0,
            TypeVerdict::Unknown => 0.5,
            TypeVerdict::Disjoint => -4.0,
        };
        score += match kind {
            ReadingKind::Entity => 0.4,
            ReadingKind::Type => 0.2,
            ReadingKind::Composition => 0.1,
            ReadingKind::Unknown | ReadingKind::Predicate => 0.0,
        };
        if canonical == token.canonical && kind == token.kind {
            score += 0.2;
        }
        if canonical == surface {
            score += 0.05;
        }
        scored.push((score, canonical, kind, types, evidence));
    }

    if scored.is_empty() {
        return ReadingChoice {
            canonical: surface.to_string(),
            kind: ReadingKind::Unknown,
            etype: None,
            evidence: vec!["词表无读法，保留原词面".to_string()],
            ambiguity: None,
        };
    }
    scored.sort_by(|left, right| {
        right
            .0
            .total_cmp(&left.0)
            .then_with(|| left.1.cmp(&right.1))
            .then_with(|| left.2.cmp(&right.2))
    });
    let best = scored[0].clone();
    let ambiguity = scored.get(1).and_then(|second| {
        ((best.0 - second.0).abs() <= AMBIGUITY_MARGIN).then(|| {
            format!(
                "「{}」有多个候选读法：{} / {}；当前按 {}（类型 {}）",
                surface,
                best.1,
                second.1,
                best.1,
                if best.3.is_empty() {
                    "∅".to_string()
                } else {
                    best.3.iter().cloned().collect::<Vec<_>>().join("、")
                }
            )
        })
    });
    let etype = best.3.iter().next().cloned();
    let mut evidence = best.4;
    evidence.push(format!(
        "读法评分：{}（{}）",
        best.1,
        type_verdict_label(schema.sets_admit(&best.3, slot_types))
    ));
    ReadingChoice {
        canonical: best.1,
        kind: best.2,
        etype,
        evidence,
        ambiguity,
    }
}

fn reading_types(
    canonical: &str,
    surface: &str,
    kind: ReadingKind,
    hints: &BTreeMap<String, BTreeSet<String>>,
    token: &SegmentedToken,
) -> BTreeSet<String> {
    let mut types = BTreeSet::new();
    if canonical == surface {
        if let Some(hint) = hints.get(surface) {
            types.extend(hint.iter().cloned());
        }
        if token.canonical == canonical {
            if let Some(etype) = &token.etype {
                types.insert(etype.clone());
            }
        }
    } else if let Some(hint) = hints.get(canonical) {
        types.extend(hint.iter().cloned());
    }
    if kind == ReadingKind::Type && !canonical.trim().is_empty() {
        types.insert(canonical.trim().to_string());
    }
    types
}

fn argument_types(
    canonical: &str,
    surface: &str,
    hints: &BTreeMap<String, BTreeSet<String>>,
    result: &SegmentResult,
    span: Option<(usize, usize)>,
) -> BTreeSet<String> {
    let mut types = BTreeSet::new();
    if canonical == surface {
        if let Some(hint) = hints.get(surface) {
            types.extend(hint.iter().cloned());
        }
    } else if let Some(hint) = hints.get(canonical) {
        types.extend(hint.iter().cloned());
    }
    if let Some(span) = span {
        if let Some(token) = result
            .tokens
            .iter()
            .find(|token| token.start == span.0 && token.end == span.1)
        {
            if token.canonical == canonical {
                if let Some(etype) = &token.etype {
                    types.insert(etype.clone());
                }
            }
        }
    }
    types
}

fn new_word_candidate(
    argument: &JointArgument,
    result: &SegmentResult,
    segmenter: &ContextualSegmenter,
    hints: &BTreeMap<String, BTreeSet<String>>,
    confidence: f64,
    role: &str,
) -> Option<NewWordCandidate> {
    let span = argument.token_span?;
    let token = result
        .tokens
        .iter()
        .find(|token| token.start == span.0 && token.end == span.1)?;
    if argument.canonical != argument.surface {
        return None;
    }
    if !hints.contains_key(&argument.canonical)
        && !segmenter
            .surface_readings(&argument.surface)
            .iter()
            .any(|(_, kind)| *kind == ReadingKind::Entity)
    {
        let mut evidence = vec![format!(
            "{}「{}」未登录，帧置信度 {:.2}",
            role, argument.surface, confidence
        )];
        if argument.welded {
            evidence.push("参数跨度跨分词边界，已按帧证据焊接".to_string());
        }
        evidence.push(format!("当前读法 {}", token.kind.label()));
        return Some(NewWordCandidate {
            surface: argument.surface.clone(),
            span,
            role: argument.role,
            evidence,
        });
    }
    None
}

#[derive(Debug, Clone)]
struct FrameRequests {
    subject: ArgumentRequest,
    object: ArgumentRequest,
}

#[derive(Debug, Clone)]
struct ArgumentRequest {
    role: ArgumentRole,
    surface: String,
    span: Option<(usize, usize)>,
    error: Option<String>,
}

fn argument_request(
    text: &str,
    frame: &RelationFrame,
    role: ArgumentRole,
    max_chars: usize,
) -> ArgumentRequest {
    let surface = match role {
        ArgumentRole::Subject => frame.subject.trim(),
        ArgumentRole::Object => frame.object.trim(),
    }
    .to_string();
    if surface.is_empty() {
        return ArgumentRequest {
            role,
            surface,
            span: None,
            error: Some("为空".to_string()),
        };
    }
    if is_function_argument(&surface) {
        return ArgumentRequest {
            role,
            surface,
            span: None,
            error: Some("是功能词/标点，不能作为关系参数".to_string()),
        };
    }
    if surface.chars().count() > max_chars.max(1) {
        return ArgumentRequest {
            role,
            surface,
            span: None,
            error: Some(format!("超过 {} 字符的参数上限", max_chars.max(1))),
        };
    }
    let Some(mut span) = absolute_char_span(text, frame, role) else {
        return ArgumentRequest {
            role,
            surface,
            span: None,
            error: Some("在原文中找不到字符跨度".to_string()),
        };
    };
    if char_slice(text, span.0, span.1) != surface {
        if let Some(found) = find_char_span_from(text, &surface, frame.evidence.start) {
            span = found;
        } else {
            return ArgumentRequest {
                role,
                surface,
                span: Some(span),
                error: Some("跨度文本与参数词面不一致".to_string()),
            };
        }
    }
    ArgumentRequest {
        role,
        surface,
        span: Some(span),
        error: None,
    }
}

#[derive(Debug, Clone)]
struct WeldRecord {
    was_atomic: bool,
}

fn absolute_char_span(
    text: &str,
    frame: &RelationFrame,
    role: ArgumentRole,
) -> Option<(usize, usize)> {
    let (surface, relative) = match role {
        ArgumentRole::Subject => (&frame.subject, frame.evidence.subject_span),
        ArgumentRole::Object => (&frame.object, frame.evidence.object_span),
    };
    let surface = surface.trim();
    if surface.is_empty() {
        return None;
    }
    let sentence = &frame.evidence.sentence;
    let relative = relative.or_else(|| {
        sentence
            .find(surface)
            .map(|start| (start, start + surface.len()))
    })?;
    let abs_start = frame.evidence.start.checked_add(relative.0)?;
    let abs_end = frame.evidence.start.checked_add(relative.1)?;
    if let (Some(start), Some(end)) = (byte_to_char(text, abs_start), byte_to_char(text, abs_end)) {
        if start < end {
            return Some((start, end));
        }
    }
    find_char_span_from(text, surface, frame.evidence.start)
}

fn find_char_span_from(text: &str, surface: &str, byte_start: usize) -> Option<(usize, usize)> {
    let byte_start = byte_start.min(text.len());
    let haystack = &text[byte_start..];
    let relative = haystack.find(surface)?;
    let absolute = byte_start + relative;
    let start = text[..absolute].chars().count();
    Some((start, start + surface.chars().count()))
}

fn byte_to_char(text: &str, byte: usize) -> Option<usize> {
    if byte > text.len() || !text.is_char_boundary(byte) {
        return None;
    }
    Some(text[..byte].chars().count())
}

fn char_slice(text: &str, start: usize, end: usize) -> String {
    text.chars()
        .skip(start)
        .take(end.saturating_sub(start))
        .collect()
}

fn exact_token(result: &SegmentResult, span: (usize, usize)) -> Option<usize> {
    result
        .tokens
        .iter()
        .position(|token| token.start == span.0 && token.end == span.1)
}

fn weld_span(
    result: &mut SegmentResult,
    span: (usize, usize),
    surface: &str,
    preserved: Option<&SegmentedToken>,
) -> bool {
    if exact_token(result, span).is_some() {
        return false;
    }
    let text = result.text.clone();
    let tokens = std::mem::take(&mut result.tokens);
    let mut welded_tokens = Vec::with_capacity(tokens.len() + 1);
    let mut inserted = false;
    for token in tokens {
        if token.end <= span.0 || token.start >= span.1 {
            welded_tokens.push(token);
            continue;
        }
        if token.start < span.0 {
            welded_tokens.push(fragment_token(&text, token.start, span.0, &token));
        }
        if !inserted {
            welded_tokens.push(welded_token(surface, span, preserved));
            inserted = true;
        }
        if token.end > span.1 {
            welded_tokens.push(fragment_token(&text, span.1, token.end, &token));
        }
    }
    if !inserted {
        let index = welded_tokens.partition_point(|token| token.start < span.0);
        welded_tokens.insert(index, welded_token(surface, span, preserved));
    }
    welded_tokens.sort_by_key(|token| (token.start, token.end));
    result.tokens = welded_tokens;
    true
}

fn welded_token(
    surface: &str,
    span: (usize, usize),
    preserved: Option<&SegmentedToken>,
) -> SegmentedToken {
    let mut token = SegmentedToken {
        text: surface.to_string(),
        start: span.0,
        end: span.1,
        canonical: surface.to_string(),
        kind: preserved
            .map(|token| token.kind)
            .unwrap_or(ReadingKind::Unknown),
        etype: preserved.and_then(|token| token.etype.clone()),
        evidence: vec![format!("帧锚定：参数「{}」焊为原子 token", surface)],
        score: ReadingScore::default(),
    };
    if let Some(preserved) = preserved {
        if preserved.text == surface {
            token.canonical = preserved.canonical.clone();
            token.kind = preserved.kind;
            token.etype = preserved.etype.clone();
            token.score = preserved.score.clone();
            token.evidence.extend(preserved.evidence.iter().cloned());
            token.evidence.sort();
            token.evidence.dedup();
        }
    }
    token
}

fn fragment_token(
    text: &str,
    start: usize,
    end: usize,
    original: &SegmentedToken,
) -> SegmentedToken {
    let fragment = char_slice(text, start, end);
    SegmentedToken {
        text: fragment.clone(),
        start,
        end,
        canonical: fragment,
        kind: ReadingKind::Unknown,
        etype: None,
        evidence: vec![format!("从「{}」切分（帧边界）", original.text)],
        score: ReadingScore::default(),
    }
}

fn is_function_argument(surface: &str) -> bool {
    let surface = surface.trim();
    if surface.is_empty() {
        return true;
    }
    if surface.chars().all(|ch| {
        ch.is_whitespace()
            || matches!(
                ch,
                '，' | '。'
                    | '！'
                    | '？'
                    | '；'
                    | '：'
                    | '、'
                    | '（'
                    | '）'
                    | '【'
                    | '】'
                    | '《'
                    | '》'
                    | '"'
                    | '\''
                    | ','
                    | '.'
                    | '!'
                    | '?'
                    | ';'
                    | ':'
                    | '('
                    | ')'
                    | '['
                    | ']'
            )
    }) {
        return true;
    }
    matches!(
        surface,
        "的" | "了"
            | "是"
            | "和"
            | "与"
            | "及"
            | "在"
            | "被"
            | "把"
            | "对"
            | "为"
            | "于"
            | "中"
            | "上"
            | "下"
            | "个"
            | "这"
            | "那"
            | "其"
            | "之"
            | "而"
            | "或"
            | "也"
            | "都"
            | "就"
            | "很"
            | "不"
            | "没"
            | "有"
            | "吗"
            | "呢"
            | "吧"
            | "啊"
            | "一个"
            | "一种"
    )
}

/// 生成回译验证：要求重新解析出的帧与期望帧逐一对应（含极性/模态）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrameVerification {
    pub passed: bool,
    pub matched: usize,
    pub missing: Vec<String>,
    pub extra: Vec<String>,
    pub reasons: Vec<String>,
}

impl FrameVerification {
    pub fn render_report(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "回译验证：{}（匹配 {}；缺失 {}；多余 {}）",
            if self.passed { "通过" } else { "未通过" },
            self.matched,
            self.missing.len(),
            self.extra.len()
        );
        for missing in &self.missing {
            let _ = writeln!(out, "  缺失：{}", missing);
        }
        for extra in &self.extra {
            let _ = writeln!(out, "  多余：{}", extra);
        }
        for reason in &self.reasons {
            let _ = writeln!(out, "  理由：{}", reason);
        }
        out
    }
}

/// 用联合解析结果验证生成候选：主语、谓词、宾语、极性、模态必须一致。
pub fn verify_frames(
    schema: &SchemaView,
    parsed: &JointParse,
    expected: &[RelationFrame],
) -> FrameVerification {
    let parsed_keys: Vec<String> = parsed
        .frames
        .iter()
        .map(|frame| frame_key(schema, &frame.normalized))
        .collect();
    let mut used = vec![false; parsed_keys.len()];
    let mut missing = Vec::new();
    let mut matched = 0usize;
    let mut reasons = Vec::new();
    for frame in expected {
        let key = frame_key(schema, frame);
        if let Some(index) = parsed_keys
            .iter()
            .enumerate()
            .find(|(index, parsed_key)| {
                !used[*index] && **parsed_key == key && !parsed.frames[*index].is_hard_failure()
            })
            .map(|(index, _)| index)
        {
            used[index] = true;
            matched += 1;
        } else {
            if parsed_keys.iter().enumerate().any(|(index, parsed_key)| {
                **parsed_key == key && parsed.frames[index].is_hard_failure()
            }) {
                reasons.push(format!(
                    "期望帧「{}」虽然命中，但联合解析存在硬冲突",
                    frame.render_text()
                ));
            }
            missing.push(frame.render_text());
        }
    }
    let extra: Vec<String> = parsed
        .frames
        .iter()
        .enumerate()
        .filter(|(index, _)| !used[*index])
        .map(|(_, frame)| frame.normalized.render_text())
        .collect();
    if !missing.is_empty() {
        reasons.push("生成句没有恢复全部期望帧".to_string());
    }
    if !extra.is_empty() {
        reasons.push("生成句带出了期望之外的帧".to_string());
    }
    if parsed.frames.is_empty() {
        reasons.push("联合解析没有抽到任何帧".to_string());
    }
    FrameVerification {
        passed: missing.is_empty() && extra.is_empty() && !parsed.frames.is_empty(),
        matched,
        missing,
        extra,
        reasons,
    }
}

fn frame_key(schema: &SchemaView, frame: &RelationFrame) -> String {
    format!(
        "{}|{}|{}|{}|{}",
        frame.subject.trim(),
        schema.canonical_property(&frame.predicate),
        frame.object.trim(),
        polarity_code(frame.polarity),
        modality_code(frame.modality)
    )
}

fn polarity_code(polarity: Polarity) -> &'static str {
    match polarity {
        Polarity::Affirmative => "affirmative",
        Polarity::Negative => "negative",
    }
}

fn modality_code(modality: crate::relation_frame::Modality) -> &'static str {
    match modality {
        crate::relation_frame::Modality::Asserted => "asserted",
        crate::relation_frame::Modality::Possible => "possible",
        crate::relation_frame::Modality::Necessary => "necessary",
        crate::relation_frame::Modality::Uncertain => "uncertain",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contextual_segmentation::ContextualSegmenter;
    use crate::relation_frame::{EvidenceSpan, Modality, Polarity};
    use crate::schema_view::SchemaView;
    use std::collections::BTreeSet;

    fn schema_with(domain: &[&str], range: &[&str], disjoint: &[(&str, &str)]) -> SchemaView {
        let mut schema = SchemaView::new();
        for class in domain {
            schema.declare_domain("创始人", class);
        }
        for class in range {
            schema.declare_range("创始人", class);
        }
        for (left, right) in disjoint {
            schema.declare_disjoint(left, right);
        }
        schema
    }

    fn hints(entries: &[(&str, &[&str])]) -> BTreeMap<String, BTreeSet<String>> {
        entries
            .iter()
            .map(|(entity, types)| {
                (
                    entity.to_string(),
                    types.iter().map(|etype| etype.to_string()).collect(),
                )
            })
            .collect()
    }

    fn frame_with_spans(text: &str, subject: &str, predicate: &str, object: &str) -> RelationFrame {
        let mut frame = RelationFrame::new(subject, predicate, object);
        frame.evidence = EvidenceSpan {
            sentence_index: 0,
            sentence: text.to_string(),
            start: 0,
            end: text.len(),
            subject_span: Some(byte_span(text, subject)),
            object_span: Some(byte_span(text, object)),
            polarity_span: None,
        };
        frame
    }

    fn byte_span(text: &str, surface: &str) -> (usize, usize) {
        let start = text.find(surface).expect("surface in text");
        (start, start + surface.len())
    }

    fn segmenter_with_apple() -> ContextualSegmenter {
        let mut segmenter = ContextualSegmenter::new();
        segmenter.observe_type("水果");
        segmenter.observe_type("公司");
        segmenter.observe_type("人物");
        segmenter.observe_entity("苹果", Some("水果"));
        segmenter.observe_entity("苹果公司", Some("公司"));
        segmenter.observe_alias("苹果", "苹果公司");
        segmenter.observe_entity("乔布斯", Some("人物"));
        segmenter
    }

    #[test]
    fn absolute_spans_handle_cjk_and_emoji() {
        let text = "😀张三的师傅是李四";
        let frame = frame_with_spans(text, "张三", "师傅", "李四");
        assert_eq!(
            absolute_char_span(text, &frame, ArgumentRole::Subject),
            Some((1, 3))
        );
        assert_eq!(
            absolute_char_span(text, &frame, ArgumentRole::Object),
            Some((7, 9))
        );
        assert_eq!(char_slice(text, 1, 3), "张三");
    }

    #[test]
    fn frame_constraints_normalize_ambiguous_surface() {
        let mut segmenter = segmenter_with_apple();
        let schema = schema_with(&["公司"], &["人物"], &[("水果", "公司")]);
        let text = "苹果的创始人是乔布斯";
        let options = JointParseOptions {
            type_hints: hints(&[
                ("苹果", &["水果"]),
                ("苹果公司", &["公司"]),
                ("乔布斯", &["人物"]),
            ]),
            ..JointParseOptions::default()
        };
        let parse = joint_parse(&mut segmenter, &schema, text, &options);
        assert!(!parse.frames.is_empty(), "应抽到创始人帧：{:?}", parse);
        let frame = &parse.frames[0];
        assert_eq!(frame.subject.canonical, "苹果公司", "帧约束应选公司读法");
        assert_eq!(frame.normalized.subject, "苹果公司");
        assert!(!frame.is_hard_failure(), "冲突：{:?}", frame.conflicts);
        assert!(parse
            .segmentation
            .tokens
            .iter()
            .any(|token| token.text == "苹果" && token.canonical == "苹果公司"));
    }

    #[test]
    fn disjoint_types_are_a_hard_gate() {
        let mut segmenter = ContextualSegmenter::new();
        segmenter.observe_entity("苹果", Some("水果"));
        segmenter.observe_entity("乔布斯", Some("人物"));
        let schema = schema_with(&["公司"], &["人物"], &[("水果", "公司")]);
        let text = "苹果的创始人是乔布斯";
        let options = JointParseOptions {
            type_hints: hints(&[("苹果", &["水果"]), ("乔布斯", &["人物"])]),
            ..JointParseOptions::default()
        };
        let parse = joint_parse(&mut segmenter, &schema, text, &options);
        assert!(!parse.frames.is_empty());
        let frame = &parse.frames[0];
        assert_eq!(frame.overall_verdict, TypeVerdict::Disjoint);
        assert!(!frame.conflicts.is_empty());
        assert!(!parse.converged);
        assert!(parse.context.focus_entities.is_empty());
    }

    #[test]
    fn verification_rejects_frames_with_hard_conflicts() {
        let mut segmenter = ContextualSegmenter::new();
        segmenter.observe_entity("苹果", Some("水果"));
        segmenter.observe_entity("乔布斯", Some("人物"));
        let schema = schema_with(&["公司"], &["人物"], &[("水果", "公司")]);
        let text = "苹果的创始人是乔布斯";
        let options = JointParseOptions {
            type_hints: hints(&[("苹果", &["水果"]), ("乔布斯", &["人物"])]),
            ..JointParseOptions::default()
        };
        let parse = joint_parse(&mut segmenter, &schema, text, &options);
        let expected = vec![RelationFrame::new("苹果", "创始人", "乔布斯")];
        let verification = verify_frames(&schema, &parse, &expected);
        assert!(!verification.passed);
        assert!(verification
            .reasons
            .iter()
            .any(|reason| reason.contains("硬冲突")));
    }

    #[test]
    fn welding_keeps_span_and_reports_oov_candidate() {
        let mut segmenter = ContextualSegmenter::new();
        let text = "纳西姆塔勒布认识李四";
        let frame = frame_with_spans(text, "纳西姆塔勒布", "认识", "李四");
        let schema = SchemaView::new();
        let options = JointParseOptions::default();
        let parse = joint_parse_with_frames(&mut segmenter, &schema, text, vec![frame], &options);
        assert_eq!(parse.frames.len(), 1);
        let frame = &parse.frames[0];
        assert!(frame.aligned);
        assert!(frame.subject.welded);
        assert_eq!(frame.subject.canonical, "纳西姆塔勒布");
        assert!(parse
            .segmentation
            .tokens
            .iter()
            .any(|token| token.text == "纳西姆塔勒布"
                && token.start == frame.subject.span.unwrap().0
                && token.end == frame.subject.span.unwrap().1));
        assert!(!parse.candidates.is_empty());
    }

    #[test]
    fn extractor_welds_oov_person_and_nominates_candidate() {
        let mut segmenter = ContextualSegmenter::new();
        segmenter.observe_entity("李四", Some("人物"));
        let schema = SchemaView::new();
        let text = "纳西姆塔勒布认识李四";
        let parse = joint_parse(&mut segmenter, &schema, text, &JointParseOptions::default());
        assert!(!parse.frames.is_empty(), "应抽到 OOV 人名帧：{:?}", parse);
        let frame = &parse.frames[0];
        assert_eq!(frame.subject.canonical, "纳西姆塔勒布");
        let span = frame.subject.span.expect("subject span");
        assert!(parse.segmentation.tokens.iter().any(|token| {
            token.start == span.0 && token.end == span.1 && token.text == "纳西姆塔勒布"
        }));
        assert!(
            parse
                .candidates
                .iter()
                .any(|candidate| candidate.surface == "纳西姆塔勒布"),
            "OOV 人名应进入候选新词：{:?}",
            parse.candidates
        );
    }

    #[test]
    fn verify_frames_round_trips_generation() {
        let mut segmenter = ContextualSegmenter::new();
        let schema = SchemaView::new();
        let text = "张三的师傅是李四";
        let parse = joint_parse(&mut segmenter, &schema, text, &JointParseOptions::default());
        assert!(!parse.frames.is_empty(), "应抽到师傅帧：{:?}", parse);
        let expected = vec![RelationFrame::new("张三", "师傅", "李四")];
        let verification = verify_frames(&schema, &parse, &expected);
        assert!(verification.passed, "{:?}", verification);

        let negative =
            vec![RelationFrame::new("张三", "师傅", "李四").with_polarity(Polarity::Negative)];
        let verification = verify_frames(&schema, &parse, &negative);
        assert!(!verification.passed);
        assert!(!verification.missing.is_empty());
    }

    #[test]
    fn modality_and_polarity_are_preserved() {
        let mut segmenter = ContextualSegmenter::new();
        let schema = SchemaView::new();
        let text = "张三不认识李四";
        let mut frame = frame_with_spans(text, "张三", "认识", "李四");
        frame.polarity = Polarity::Negative;
        frame.modality = Modality::Asserted;
        let options = JointParseOptions::default();
        let parse = joint_parse_with_frames(&mut segmenter, &schema, text, vec![frame], &options);
        assert_eq!(parse.frames[0].normalized.polarity, Polarity::Negative);
        assert_eq!(parse.frames[0].normalized.modality, Modality::Asserted);
    }
}

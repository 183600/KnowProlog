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
    fn overlapping_argument_surfaces_are_disambiguated_by_order() {
        let text = "苹果公司的创始人是苹果";
        let mut frame = RelationFrame::new("苹果公司", "创始人", "苹果");
        frame.evidence = EvidenceSpan {
            sentence_index: 0,
            sentence: text.to_string(),
            start: 0,
            end: text.len(),
            subject_span: None,
            object_span: None,
            polarity_span: None,
        };
        let schema = schema_with(&["公司"], &["水果"], &[]);
        let mut segmenter = segmenter_with_apple();
        let parse = joint_parse_with_frames(
            &mut segmenter,
            &schema,
            text,
            vec![frame],
            &JointParseOptions::default(),
        );
        let frame = &parse.frames[0];
        assert_eq!(frame.subject.span, Some((0, 4)));
        assert_eq!(frame.object.span, Some((9, 11)));
        assert!(frame.conflicts.is_empty(), "{:?}", frame.conflicts);
    }

    #[test]
    fn repeated_surface_gets_two_spans_in_order() {
        let text = "张三认识张三";
        let mut frame = RelationFrame::new("张三", "认识", "张三");
        frame.evidence = EvidenceSpan {
            sentence_index: 0,
            sentence: text.to_string(),
            start: 0,
            end: text.len(),
            subject_span: None,
            object_span: None,
            polarity_span: None,
        };
        let mut segmenter = ContextualSegmenter::new();
        segmenter.observe_entity("张三", Some("人物"));
        let parse = joint_parse_with_frames(
            &mut segmenter,
            &SchemaView::new(),
            text,
            vec![frame],
            &JointParseOptions::default(),
        );
        let frame = &parse.frames[0];
        assert_eq!(frame.subject.span, Some((0, 2)));
        assert_eq!(frame.object.span, Some((4, 6)));
        assert!(frame.conflicts.is_empty(), "{:?}", frame.conflicts);
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

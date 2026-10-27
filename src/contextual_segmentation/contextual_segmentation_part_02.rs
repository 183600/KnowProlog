fn reconstruct_path(
    paths: &[Vec<PathEntry>],
    candidates: &[Candidate],
    chars: &[char],
    end: usize,
    rank: usize,
) -> ResolvedPath {
    let mut tokens: Vec<SegmentedToken> = Vec::new();
    let mut cursor = end;
    let mut current_rank = rank;
    let mut score = 0.0;
    while cursor > 0 {
        let Some(entry) = paths[cursor].get(current_rank) else {
            break;
        };
        if entry.candidate == usize::MAX {
            break;
        }
        let candidate = &candidates[entry.candidate];
        if let Some(reading) = candidate.readings.get(entry.reading) {
            tokens.push(SegmentedToken {
                text: candidate.surface.clone(),
                start: candidate.start,
                end: candidate.end,
                canonical: reading.canonical.clone(),
                kind: reading.kind,
                etype: reading.etype.clone(),
                evidence: reading.evidence.clone(),
                score: reading.score.clone(),
            });
            score += reading.score.total;
        }
        cursor = candidate.start;
        current_rank = entry.prev_rank;
    }
    tokens.reverse();
    let _ = chars;
    ResolvedPath { tokens, score }
}
fn path_key(tokens: &[SegmentedToken]) -> String {
    tokens
        .iter()
        .map(|token| format!("{}:{}:{}", token.start, token.kind.code(), token.canonical))
        .collect::<Vec<_>>()
        .join("|")
}
fn ambiguity_report(
    text: &str,
    best: &[SegmentedToken],
    alternative: &[SegmentedToken],
    margin: f64,
    threshold: f64,
) -> Option<AmbiguityReport> {
    if alternative.is_empty() {
        return None;
    }
    let limit = best.len().min(alternative.len());
    let mut divergent_index = limit;
    for index in 0..limit {
        if best[index].text != alternative[index].text
            || best[index].canonical != alternative[index].canonical
            || best[index].kind != alternative[index].kind
        {
            divergent_index = index;
            break;
        }
    }
    if divergent_index == limit && best.len() == alternative.len() {
        return None;
    }
    let best_token = best.get(divergent_index);
    let alternative_token = alternative.get(divergent_index);
    let content_divergence = best_token
        .map(|token| !is_function_surface(&token.text))
        .unwrap_or(false)
        || alternative_token
            .map(|token| !is_function_surface(&token.text))
            .unwrap_or(false)
        || best_token
            .map(|token| token.kind != ReadingKind::Unknown)
            .unwrap_or(false)
        || alternative_token
            .map(|token| token.kind != ReadingKind::Unknown)
            .unwrap_or(false);
    let options = vec![
        best_token
            .map(SegmentedToken::display_reading)
            .unwrap_or_else(|| "（空）".to_string()),
        alternative_token
            .map(SegmentedToken::display_reading)
            .unwrap_or_else(|| "（空）".to_string()),
    ];
    let triggered = margin <= threshold && content_divergence;
    let clarification = triggered.then(|| {
        format!(
            "「{}」的第 {} 个词有两种读法：{} / {}。你指的是哪一种？",
            text,
            divergent_index + 1,
            options[0],
            options[1]
        )
    });
    Some(AmbiguityReport {
        margin,
        threshold,
        divergent_index,
        content_divergence,
        options,
        clarification,
    })
}
fn is_function_surface(surface: &str) -> bool {
    let chars: Vec<char> = surface.chars().collect();
    chars.len() == 1 && is_function_char(chars[0])
}
fn is_function_char(ch: char) -> bool {
    matches!(
        ch,
        '的' | '了'
            | '是'
            | '在'
            | '和'
            | '与'
            | '及'
            | '或'
            | '有'
            | '被'
            | '把'
            | '对'
            | '为'
            | '从'
            | '到'
            | '就'
            | '也'
            | '都'
            | '很'
            | '吗'
            | '呢'
            | '吧'
            | '啊'
            | '呀'
            | '么'
            | '个'
            | '一'
            | '谁'
            | '什'
            | '怎'
            | '多'
            | '少'
            | '几'
            | '这'
            | '那'
            | '哪'
            | '我'
            | '你'
            | '他'
            | '她'
            | '它'
    )
}
fn jieba_spans(text: &str, char_len: usize) -> HashSet<(usize, usize)> {
    let mut spans = HashSet::new();
    let mut offset = 0usize;
    for word in shared_jieba().cut(text, false) {
        let len = word.chars().count();
        if len > 0 && offset + len <= char_len {
            spans.insert((offset, offset + len));
        }
        offset += len;
    }
    spans
}
fn surface_ngrams(surface: &str, min: usize, max: usize) -> Vec<String> {
    let chars: Vec<char> = surface.chars().collect();
    let mut ngrams = Vec::new();
    if chars.is_empty() {
        return ngrams;
    }
    let max = max.min(chars.len());
    for len in min..=max {
        for start in 0..=chars.len() - len {
            ngrams.push(chars[start..start + len].iter().collect());
        }
    }
    ngrams.sort();
    ngrams.dedup();
    ngrams
}
fn find_char_span(surface: &str, needle: &str) -> Option<(usize, usize)> {
    let haystack: Vec<char> = surface.chars().collect();
    let needle: Vec<char> = needle.chars().collect();
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    for start in 0..=haystack.len() - needle.len() {
        if haystack[start..start + needle.len()] == needle[..] {
            return Some((start, start + needle.len()));
        }
    }
    None
}
fn is_punctuation(ch: char) -> bool {
    ch.is_ascii_punctuation()
        || matches!(
            ch,
            '，' | '。'
                | '？'
                | '！'
                | '、'
                | '；'
                | '：'
                | '“'
                | '”'
                | '‘'
                | '’'
                | '（'
                | '）'
                | '《'
                | '》'
                | '…'
                | '—'
                | '·'
                | '【'
                | '】'
                | '「'
                | '」'
        )
}
#[cfg(test)]
mod tests {
    use super::*;

    fn sample_segmenter() -> ContextualSegmenter {
        let mut segmenter = ContextualSegmenter::new();
        segmenter.observe_entity("苹果", Some("水果"));
        segmenter.observe_entity("红富士", Some("水果"));
        segmenter.observe_entity("苹果公司", Some("公司"));
        segmenter.observe_entity("乔布斯", Some("人物"));
        segmenter.observe_entity("星尘计划", Some("项目"));
        segmenter.observe_entity("火星计划", Some("项目"));
        segmenter.observe_entity("张三", Some("人物"));
        segmenter.observe_entity("李四", Some("人物"));
        segmenter.observe_triple("苹果", "品种", "红富士");
        segmenter.observe_triple("苹果公司", "创始人", "乔布斯");
        segmenter.observe_triple("星尘计划", "目标", "火星计划");
        segmenter.observe_alias("苹果", "苹果公司");
        segmenter.rebuild_index();
        segmenter
    }

    #[test]
    fn context_selects_between_same_surface_readings() {
        let mut segmenter = sample_segmenter();
        let fruit = segmenter.segment(
            "苹果",
            &SegmentOptions {
                context_terms: vec!["红富士".to_string()],
                expected_types: vec!["水果".to_string()],
                ..Default::default()
            },
        );
        assert_eq!(fruit.tokens.len(), 1);
        assert_eq!(fruit.tokens[0].canonical, "苹果");
        assert_eq!(fruit.tokens[0].etype.as_deref(), Some("水果"));

        let company = segmenter.segment(
            "苹果",
            &SegmentOptions {
                context_terms: vec!["乔布斯".to_string()],
                expected_types: vec!["公司".to_string()],
                ..Default::default()
            },
        );
        assert_eq!(company.tokens.len(), 1);
        assert_eq!(company.tokens[0].canonical, "苹果公司");
        assert_eq!(company.tokens[0].etype.as_deref(), Some("公司"));
    }

    #[test]
    fn neutral_ambiguity_keeps_both_readings_and_asks() {
        let mut segmenter = sample_segmenter();
        let result = segmenter.segment("苹果", &SegmentOptions::default());
        assert_eq!(result.tokens.len(), 1);
        assert!(!result.alternatives.is_empty());
        let ambiguity = result.ambiguity.as_ref().expect("歧义应被暴露");
        assert!(ambiguity.content_divergence);
        assert!(ambiguity.margin <= ambiguity.threshold);
        assert!(ambiguity
            .options
            .iter()
            .any(|option| option.contains("苹果公司")));
        assert!(ambiguity
            .clarification
            .as_deref()
            .unwrap_or_default()
            .contains("两种读法"));
    }

    #[test]
    fn composition_keeps_oov_suffix_word_whole() {
        let mut segmenter = sample_segmenter();
        let result = segmenter.segment(
            "银河计划启动",
            &SegmentOptions {
                context_terms: vec!["星尘计划".to_string()],
                expected_types: vec!["项目".to_string()],
                ..Default::default()
            },
        );
        let texts = result.token_texts();
        assert!(
            texts.contains(&"银河计划"),
            "已知子词「计划」应支撑未登录组合：{:?}",
            texts
        );
        let token = result
            .tokens
            .iter()
            .find(|token| token.text == "银河计划")
            .unwrap();
        assert_eq!(token.kind, ReadingKind::Composition);
        assert_eq!(token.etype.as_deref(), Some("项目"));
        assert!(token
            .evidence
            .iter()
            .any(|evidence| evidence.contains("计划")));
    }

    #[test]
    fn known_entities_are_not_glued_together() {
        let mut segmenter = sample_segmenter();
        let result = segmenter.segment(
            "张三李四",
            &SegmentOptions {
                expected_types: vec!["人物".to_string()],
                ..Default::default()
            },
        );
        assert_eq!(result.token_texts(), vec!["张三", "李四"]);
        assert!(result
            .tokens
            .iter()
            .all(|token| token.kind == ReadingKind::Entity));
    }

    #[test]
    fn reading_memory_flips_neutral_choice() {
        let mut segmenter = sample_segmenter();
        let first = segmenter.segment("苹果", &SegmentOptions::default());
        assert_eq!(first.tokens[0].canonical, "苹果");
        let fingerprint = first.context.fingerprint.clone();
        assert!(segmenter.memory_mut().correct(
            "苹果",
            "苹果",
            ReadingKind::Entity,
            "苹果公司",
            ReadingKind::Entity,
            &fingerprint
        ));
        let second = segmenter.segment("苹果", &SegmentOptions::default());
        assert_eq!(second.tokens[0].canonical, "苹果公司");
        assert_eq!(
            segmenter
                .memory()
                .preferred("苹果", &fingerprint)
                .map(|preference| preference.canonical.as_str()),
            Some("苹果公司")
        );
    }

    #[test]
    fn segmentation_is_deterministic_and_faithful() {
        let mut segmenter = sample_segmenter();
        let options = SegmentOptions {
            context_terms: vec!["红富士".to_string()],
            ..Default::default()
        };
        let first = segmenter.segment("苹果的品种是红富士", &options);
        let second = segmenter.segment("苹果的品种是红富士", &options);
        assert_eq!(first.tokens, second.tokens);
        assert_eq!(
            first.score.total_cmp(&second.score),
            std::cmp::Ordering::Equal
        );
        assert_eq!(
            first.token_texts().concat(),
            "苹果的品种是红富士",
            "路径必须逐字覆盖原文"
        );
        for (index, token) in first.tokens.iter().enumerate() {
            assert_eq!(token.start, first.tokens[index].start);
        }
    }

    #[test]
    fn composition_requires_a_known_subword() {
        let mut segmenter = sample_segmenter();
        let result = segmenter.segment(
            "甲乙丙丁",
            &SegmentOptions {
                expected_types: vec!["项目".to_string()],
                ..Default::default()
            },
        );
        assert!(result
            .tokens
            .iter()
            .all(|token| token.kind != ReadingKind::Composition));
    }

    #[test]
    fn schema_disjoint_types_are_penalized() {
        let mut segmenter = sample_segmenter();
        let mut schema = SchemaView::new();
        schema.declare_disjoint("水果", "公司");
        segmenter.set_schema(schema);
        let result = segmenter.segment(
            "苹果",
            &SegmentOptions {
                context_terms: vec!["红富士".to_string()],
                expected_types: vec!["公司".to_string()],
                ..Default::default()
            },
        );
        let fruit = result
            .alternatives
            .iter()
            .flat_map(|alternative| alternative.readings.iter())
            .any(|reading| reading.contains("苹果（苹果）"));
        let company = result.tokens[0].canonical == "苹果公司";
        assert!(
            company || fruit,
            "模式互斥应让语境与类型证据参与比较：{:?}",
            result.tokens
        );
    }

    #[test]
    fn type_aliases_and_triples_are_deduplicated() {
        let mut segmenter = sample_segmenter();
        let types = segmenter.type_count();
        let triples = segmenter.triple_count();
        segmenter.observe_type_alias("企业", "公司");
        assert_eq!(segmenter.type_count(), types);
        assert_eq!(
            segmenter.reading_kind("企业", "公司"),
            Some(ReadingKind::Type)
        );
        segmenter.observe_triple("张三", "师傅", "李四");
        segmenter.observe_triple("张三", "师傅", "李四");
        assert_eq!(segmenter.triple_count(), triples + 1);
    }

    #[test]
    fn reading_memory_json_roundtrip_is_lossless() {
        let mut memory = ReadingMemory::new(8);
        memory.confirm("苹果", "苹果公司", ReadingKind::Entity, "ctx-1");
        memory.correct(
            "苹果",
            "苹果",
            ReadingKind::Entity,
            "苹果公司",
            ReadingKind::Entity,
            "ctx-1",
        );
        let json = memory.to_json().expect("记忆应可序列化");
        let restored = ReadingMemory::from_json(&json).expect("记忆应可反序列化");
        assert_eq!(restored, memory);
        let report = restored.render_report();
        assert!(!report.contains('\u{1f}'), "{}", report);
        assert!(report.contains("「苹果」@ctx-1"), "{}", report);
        assert_eq!(
            restored
                .preferred("苹果", "ctx-1")
                .map(|preference| preference.canonical.as_str()),
            Some("苹果公司")
        );
    }

    #[test]
    fn reading_kind_exposes_surface_senses() {
        let segmenter = sample_segmenter();
        assert_eq!(
            segmenter.reading_kind("苹果", "苹果"),
            Some(ReadingKind::Entity)
        );
        assert_eq!(
            segmenter.reading_kind("苹果", "苹果公司"),
            Some(ReadingKind::Entity)
        );
        assert_eq!(
            segmenter.reading_kind("水果", "水果"),
            Some(ReadingKind::Type)
        );
        assert_eq!(segmenter.reading_kind("苹果", "不存在"), None);
    }

    #[test]
    fn report_contains_scores_and_evidence() {
        let mut segmenter = sample_segmenter();
        let result = segmenter.segment(
            "银河计划启动",
            &SegmentOptions {
                context_terms: vec!["星尘计划".to_string()],
                expected_types: vec!["项目".to_string()],
                ..Default::default()
            },
        );
        let report = result.render_report();
        assert!(report.contains("读法"));
        assert!(report.contains("词表="));
        assert!(report.contains("语境="));
        assert!(report.contains("银河计划"));
    }
}

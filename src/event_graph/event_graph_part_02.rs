impl EventDialogue {
    pub fn new(graph: EventGraph) -> Self {
        Self {
            graph,
            focus: EventFocus::new(DEFAULT_FOCUS_DEPTH),
        }
    }

    pub fn with_focus_depth(graph: EventGraph, depth: usize) -> Self {
        Self {
            graph,
            focus: EventFocus::new(depth),
        }
    }

    pub fn graph(&self) -> &EventGraph {
        &self.graph
    }

    pub fn focus(&self) -> &EventFocus {
        &self.focus
    }

    pub fn ask(&mut self, question: &str) -> EventAnswer {
        let answer = self.graph.ask(question, self.focus.last());
        if answer.status == EventAnswerStatus::Answered {
            let next_focus = if answer.kind == EventQuestionKind::WhatNext {
                answer
                    .events
                    .iter()
                    .rev()
                    .find(|item| item.relation == "后续")
                    .map(|item| item.event.id.clone())
                    .or_else(|| answer.focus.clone())
            } else {
                answer.focus.clone()
            };
            if let Some(focus) = next_focus {
                self.focus.push(focus);
            }
        }
        answer
    }
}
fn build_links(events: &[EventNode]) -> Vec<EventLink> {
    let mut links = Vec::new();
    if events.is_empty() {
        return links;
    }
    let mut by_sentence: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (index, event) in events.iter().enumerate() {
        by_sentence
            .entry(event.evidence.sentence_index)
            .or_default()
            .push(index);
    }
    let sentences: Vec<usize> = by_sentence.keys().copied().collect();

    for (position, sentence_index) in sentences.iter().enumerate() {
        let current = &by_sentence[sentence_index];
        let sentence = &events[current[0]].evidence.sentence;
        let sentence_start = events[current[0]].evidence.sentence_start;
        let cause_position = first_marker(sentence, CAUSE_MARKERS).map(|p| sentence_start + p);
        let result_position = first_marker(sentence, RESULT_MARKERS).map(|p| sentence_start + p);
        let order_position = first_marker(sentence, ORDER_MARKERS).map(|p| sentence_start + p);

        if let Some(result_global) = result_position {
            let causes: Vec<usize> = current
                .iter()
                .copied()
                .filter(|index| {
                    events[*index].evidence.anchor < result_global
                        && cause_position
                            .map(|cause| events[*index].evidence.anchor > cause)
                            .unwrap_or(true)
                })
                .collect();
            let effects: Vec<usize> = current
                .iter()
                .copied()
                .filter(|index| events[*index].evidence.anchor > result_global)
                .collect();
            let causes = if causes.is_empty() {
                previous_events(&sentences, position, &by_sentence)
            } else {
                causes
            };
            for cause in &causes {
                for effect in &effects {
                    push_link(
                        &mut links,
                        events,
                        *cause,
                        *effect,
                        EventLinkKind::Causes,
                        0.9,
                        sentence,
                    );
                }
            }
        } else if let Some(cause_global) = cause_position {
            let causes: Vec<usize> = current
                .iter()
                .copied()
                .filter(|index| events[*index].evidence.anchor > cause_global)
                .collect();
            let effects = next_events(&sentences, position, &by_sentence);
            for cause in &causes {
                for effect in &effects {
                    push_link(
                        &mut links,
                        events,
                        *cause,
                        *effect,
                        EventLinkKind::Causes,
                        0.85,
                        sentence,
                    );
                }
            }
        }

        if let Some(order_global) = order_position {
            let following: Vec<usize> = current
                .iter()
                .copied()
                .filter(|index| events[*index].evidence.anchor > order_global)
                .collect();
            let following = if following.is_empty() {
                current.clone()
            } else {
                following
            };
            let previous = previous_events(&sentences, position, &by_sentence);
            for before in &previous {
                for after in &following {
                    push_link(
                        &mut links,
                        events,
                        *before,
                        *after,
                        EventLinkKind::Before,
                        0.8,
                        sentence,
                    );
                }
            }
        }
    }

    for left in 0..events.len() {
        for right in (left + 1)..events.len() {
            let (Some(left_time), Some(right_time)) = (
                events[left]
                    .time
                    .as_ref()
                    .and_then(TemporalScope::to_time_span),
                events[right]
                    .time
                    .as_ref()
                    .and_then(TemporalScope::to_time_span),
            ) else {
                continue;
            };
            if events[left].evidence.sentence_index == events[right].evidence.sentence_index {
                continue;
            }
            if let (Some(left_end), Some(right_start)) = (left_time.end, right_time.start) {
                if left_end < right_start {
                    push_link(
                        &mut links,
                        events,
                        left,
                        right,
                        EventLinkKind::Before,
                        0.75,
                        "显式时间比较",
                    );
                    continue;
                }
            }
            if let (Some(right_end), Some(left_start)) = (right_time.end, left_time.start) {
                if right_end < left_start {
                    push_link(
                        &mut links,
                        events,
                        right,
                        left,
                        EventLinkKind::Before,
                        0.75,
                        "显式时间比较",
                    );
                    continue;
                }
            }
            if left_time.overlaps(&right_time) {
                push_link(
                    &mut links,
                    events,
                    left,
                    right,
                    EventLinkKind::Simultaneous,
                    0.7,
                    "显式时间重叠",
                );
            }
        }
    }

    links
}
fn push_link(
    links: &mut Vec<EventLink>,
    events: &[EventNode],
    from: usize,
    to: usize,
    kind: EventLinkKind,
    confidence: f64,
    evidence: &str,
) {
    if from == to {
        return;
    }
    if links
        .iter()
        .any(|link| link.from == events[from].id && link.to == events[to].id && link.kind == kind)
    {
        return;
    }
    links.push(EventLink {
        from: events[from].id.clone(),
        to: events[to].id.clone(),
        kind,
        evidence: Some(evidence.chars().take(80).collect()),
        confidence,
    });
}
fn previous_events(
    sentences: &[usize],
    position: usize,
    by_sentence: &BTreeMap<usize, Vec<usize>>,
) -> Vec<usize> {
    if position == 0 {
        return Vec::new();
    }
    by_sentence[&sentences[position - 1]].clone()
}
fn next_events(
    sentences: &[usize],
    position: usize,
    by_sentence: &BTreeMap<usize, Vec<usize>>,
) -> Vec<usize> {
    if position + 1 >= sentences.len() {
        return Vec::new();
    }
    by_sentence[&sentences[position + 1]].clone()
}
fn first_marker(sentence: &str, markers: &[&str]) -> Option<usize> {
    markers
        .iter()
        .filter_map(|marker| sentence.find(marker))
        .min()
}
fn contains_any(text: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| text.contains(needle))
}
fn strip_leading_connectors(mut text: &str) -> &str {
    loop {
        let trimmed = text.trim_start_matches(|ch: char| {
            ch.is_whitespace() || matches!(ch, '，' | ',' | '：' | ':')
        });
        let mut changed = false;
        for connector in LEADING_CONNECTORS {
            if let Some(rest) = trimmed.strip_prefix(connector) {
                text = rest;
                changed = true;
                break;
            }
        }
        if !changed {
            return trimmed;
        }
    }
}
const EVENT_TRIGGERS: &[&str] = &[
    "出生于",
    "毕业于",
    "成立于",
    "坐落于",
    "隶属于",
    "工作于",
    "出生于",
    "去世",
    "逝世",
    "病逝",
    "加入",
    "入职",
    "离职",
    "辞职",
    "加盟",
    "创立",
    "创建",
    "成立",
    "建立",
    "创办",
    "出生",
    "前往",
    "到达",
    "抵达",
    "去往",
    "来到",
    "返回",
    "离开",
    "迁往",
    "迁居",
    "位于",
    "认识",
    "结识",
    "参加",
    "举办",
    "举行",
    "召开",
    "发布",
    "推出",
    "收购",
    "投资",
    "赢得",
    "获得",
    "成为",
    "担任",
    "签订",
    "合作",
    "访问",
    "结婚",
    "离婚",
    "毕业",
    "完成",
    "开始",
    "结束",
    "发生",
    "宣布",
];
const SUBJECT_PREFIXES: &[&str] = &[
    "因为",
    "由于",
    "鉴于",
    "所以",
    "因此",
    "于是",
    "因而",
    "故而",
    "后来",
    "随后",
    "然后",
    "接着",
    "最后",
    "此后",
    "之后",
    "接下来",
    "同时",
    "另外",
    "此外",
    "但是",
    "而且",
    "并且",
    "都",
    "也",
    "还",
    "就",
    "又",
    "曾",
    "已经",
    "正在",
    "将",
    "要",
    "会",
    "可以",
    "能够",
    "决定",
    "计划",
];
const RELATIVE_TIME_PREFIXES: &[&str] = &[
    "大前天",
    "大后天",
    "上个月",
    "下个月",
    "上星期",
    "下星期",
    "去年",
    "今年",
    "明年",
    "前天",
    "昨天",
    "今天",
    "明天",
    "后天",
];
fn extract_trigger_frames(text: &str, ranges: &[(usize, usize)]) -> Vec<RelationFrame> {
    let mut frames = Vec::new();
    for (sentence_index, (range_start, range_end)) in ranges.iter().enumerate() {
        let raw_sentence = &text[*range_start..*range_end];
        let sentence = raw_sentence.trim();
        if sentence.is_empty() {
            continue;
        }
        let sentence_start = *range_start + (raw_sentence.len() - raw_sentence.trim_start().len());
        let sentence_temporal = parse_temporal(sentence);
        for (clause_start, clause_end) in split_clause_ranges(sentence) {
            let clause = &sentence[clause_start..clause_end];
            let clause = clause.trim();
            if clause.is_empty() {
                continue;
            }
            let clause_leading = &sentence[clause_start..clause_end];
            let clause_trim_start =
                clause_start + (clause_leading.len() - clause_leading.trim_start().len());
            let Some((trigger, trigger_position)) = first_trigger(clause) else {
                continue;
            };
            let Some((subject, subject_start)) = subject_before(clause, trigger_position) else {
                continue;
            };
            let Some((object, object_start, locative, recipient)) =
                object_after(clause, trigger_position + trigger.len())
            else {
                continue;
            };
            if subject == object {
                continue;
            }
            let prefix = &clause[..trigger_position];
            let negative = ["没有", "不是", "不", "未", "没"]
                .iter()
                .any(|cue| prefix.contains(cue))
                && !prefix.contains("不丹");
            let modality = if contains_any(sentence, &["据说", "听说", "据称", "未必", "不一定"])
            {
                Modality::Uncertain
            } else if contains_any(sentence, &["可能", "也许", "或许"]) {
                Modality::Possible
            } else if contains_any(sentence, &["一定", "必然", "必定"]) {
                Modality::Necessary
            } else {
                Modality::Asserted
            };
            let temporal = parse_temporal(clause).or_else(|| sentence_temporal.clone());
            let mut frame = RelationFrame::new(&subject, trigger, &object);
            frame.polarity = if negative {
                Polarity::Negative
            } else {
                Polarity::Affirmative
            };
            frame.modality = modality;
            frame.temporal = temporal;
            let subject_global = sentence_start + clause_trim_start + subject_start;
            frame.evidence = EvidenceSpan {
                sentence_index,
                sentence: sentence.to_string(),
                start: subject_global,
                end: sentence_start + clause_trim_start + object_start + object.len(),
                subject_span: Some((0, subject.len())),
                object_span: Some((object_start.saturating_sub(subject_start), object.len())),
                polarity_span: None,
            };
            frame.extraction = vec!["事件触发词".to_string()];
            frame.qualifiers = vec![if locative {
                "地点角色".to_string()
            } else if recipient {
                "接受者角色".to_string()
            } else {
                "受事角色".to_string()
            }];
            frame.confidence = if locative || recipient { 0.78 } else { 0.74 };
            frame.id = format!(
                "frame:{}",
                fnv1a_hex(
                    format!(
                        "trigger|{}|{}|{}|{}|{}",
                        sentence_index,
                        subject_global,
                        trigger,
                        object,
                        text.len()
                    )
                    .as_bytes()
                )
            );
            frames.push(frame);
        }
    }
    frames
}
fn split_clause_ranges(sentence: &str) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut start = 0usize;
    for (index, ch) in sentence.char_indices() {
        if matches!(ch, '，' | ',') {
            ranges.push((start, index));
            start = index + ch.len_utf8();
        }
    }
    ranges.push((start, sentence.len()));
    ranges
}
fn first_trigger(sentence: &str) -> Option<(&'static str, usize)> {
    let mut best: Option<(&'static str, usize)> = None;
    for trigger in EVENT_TRIGGERS {
        if let Some(position) = sentence.find(trigger) {
            let replace = match best {
                None => true,
                Some((known, known_position)) => {
                    position < known_position
                        || (position == known_position && trigger.len() > known.len())
                }
            };
            if replace {
                best = Some((trigger, position));
            }
        }
    }
    best
}
fn subject_before(sentence: &str, trigger_position: usize) -> Option<(String, usize)> {
    let left = &sentence[..trigger_position];
    let mut segment_start = 0usize;
    for (index, ch) in left.char_indices() {
        if matches!(
            ch,
            '，' | ',' | '；' | ';' | '：' | ':' | '。' | '！' | '？' | '!' | '?' | ' ' | '\t'
        ) {
            segment_start = index + ch.len_utf8();
        }
    }
    let segment = &left[segment_start..];
    let stripped = strip_subject_prefixes(segment);
    let mut offset = segment_start + (segment.len() - stripped.len());
    let mut candidate = stripped.trim();
    offset += stripped.len() - stripped.trim_start().len();
    if let Some(position) = candidate
        .char_indices()
        .find_map(|(index, ch)| matches!(ch, '在' | '于' | '到' | '往' | '从').then_some(index))
    {
        if position > 0 {
            candidate = candidate[..position].trim_end();
        }
    }
    let candidate = candidate.trim_matches(|ch: char| {
        ch.is_whitespace()
            || matches!(
                ch,
                '「' | '」' | '『' | '』' | '“' | '”' | '‘' | '’' | '《' | '》' | '\'' | '"'
            )
    });
    if candidate.is_empty() || candidate.chars().count() > 40 {
        return None;
    }
    Some((candidate.to_string(), offset))
}
fn object_after(sentence: &str, after_trigger: usize) -> Option<(String, usize, bool, bool)> {
    if after_trigger >= sentence.len() {
        return None;
    }
    let right = &sentence[after_trigger..];
    let leading = right.len() - right.trim_start().len();
    let mut offset = after_trigger + leading;
    let mut segment = right.trim_start();
    let mut locative = false;
    let mut recipient = false;
    loop {
        let mut changed = false;
        for prefix in [
            "正在", "已经", "了", "着", "过", "于", "在", "到", "往", "给", "向", "对",
        ] {
            if let Some(rest) = segment.strip_prefix(prefix) {
                if matches!(prefix, "于" | "在" | "到" | "往") {
                    locative = true;
                }
                if matches!(prefix, "给" | "向" | "对") {
                    recipient = true;
                }
                let trimmed = rest.trim_start();
                offset += segment.len() - trimmed.len();
                segment = trimmed;
                changed = true;
                break;
            }
        }
        if !changed {
            break;
        }
    }
    let end = segment
        .char_indices()
        .find_map(|(index, ch)| {
            matches!(
                ch,
                '，' | ',' | '；' | ';' | '。' | '！' | '？' | '!' | '?' | '\n'
            )
            .then_some(index)
        })
        .unwrap_or(segment.len());
    let candidate = segment[..end].trim();
    if candidate.is_empty() || candidate.chars().count() > 40 {
        return None;
    }
    Some((candidate.to_string(), offset, locative, recipient))
}
fn strip_subject_prefixes(mut text: &str) -> &str {
    loop {
        let trimmed = text.trim_start_matches(|ch: char| {
            ch.is_whitespace() || matches!(ch, '，' | ',' | '：' | ':' | '、' | '；' | ';')
        });
        let without_time = strip_time_prefix(trimmed);
        if without_time.len() != trimmed.len() {
            text = without_time;
            continue;
        }
        let mut changed = false;
        for prefix in SUBJECT_PREFIXES {
            if let Some(rest) = trimmed.strip_prefix(prefix) {
                text = rest;
                changed = true;
                break;
            }
        }
        if !changed {
            return trimmed;
        }
    }
}
fn strip_time_prefix(text: &str) -> &str {
    for prefix in RELATIVE_TIME_PREFIXES {
        if let Some(rest) = text.strip_prefix(prefix) {
            return rest.trim_start_matches(|ch: char| {
                ch.is_whitespace() || matches!(ch, '，' | ',' | '、' | '：' | ':')
            });
        }
    }
    let Some(nian) = text.find('年') else {
        return text;
    };
    if nian > 12 {
        return text;
    }
    let digits: String = text[..nian]
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect();
    if digits.is_empty() || !digits.chars().all(|ch| ch.is_ascii_digit()) {
        return text;
    }
    let mut rest = &text[nian + '年'.len_utf8()..];
    for unit in ['月', '日', '号'] {
        let digits_len = rest
            .char_indices()
            .take_while(|(_, ch)| ch.is_ascii_digit() || ch.is_whitespace())
            .last()
            .map(|(index, ch)| index + ch.len_utf8())
            .unwrap_or(0);
        if digits_len > 0 && rest[digits_len..].starts_with(unit) {
            rest = &rest[digits_len + unit.len_utf8()..];
        }
    }
    rest.trim_start_matches(|ch: char| {
        ch.is_whitespace() || matches!(ch, '，' | ',' | '、' | '：' | ':')
    })
}
fn extract_fallback_clauses(
    text: &str,
    ranges: &[(usize, usize)],
    extractor: &FrameExtractor,
) -> (Vec<RelationFrame>, usize) {
    let mut frames = Vec::new();
    let mut clauses = 0usize;
    for (sentence_index, (range_start, range_end)) in ranges.iter().enumerate() {
        let raw_sentence = &text[*range_start..*range_end];
        if !contains_any(
            raw_sentence,
            &[
                "因为",
                "由于",
                "所以",
                "因此",
                "于是",
                "后来",
                "随后",
                "然后",
                "接着",
                "最后",
                "此后",
                "之后",
                "接下来",
            ],
        ) {
            continue;
        }
        let sentence = raw_sentence.trim();
        let mut clause_start = *range_start;
        let mut boundary = Vec::new();
        for (offset, ch) in raw_sentence.char_indices() {
            if matches!(ch, '，' | ',') {
                boundary.push((clause_start, *range_start + offset));
                clause_start = *range_start + offset + ch.len_utf8();
            }
        }
        boundary.push((clause_start, *range_end));
        for (start, end) in boundary {
            if start >= end {
                continue;
            }
            let clause_raw = &text[start..end];
            let clause_trimmed = clause_raw.trim();
            if clause_trimmed.is_empty() {
                continue;
            }
            let clause_trim_start = start + (clause_raw.len() - clause_raw.trim_start().len());
            let stripped = strip_leading_connectors(clause_trimmed);
            let body = stripped.trim();
            if body.chars().count() < 3 {
                continue;
            }
            let body_start = clause_trim_start + (clause_trimmed.len() - stripped.len());
            let (mut clause_frames, _) = extractor.extract_with_stats(body);
            if clause_frames.is_empty() {
                continue;
            }
            clauses += 1;
            if body_start + body.len() > text.len() {
                continue;
            }
            for frame in &mut clause_frames {
                frame.evidence.sentence_index = sentence_index;
                frame.evidence.sentence = sentence.to_string();
                frame.evidence.start = body_start;
                frame.evidence.end = body_start + body.len();
            }
            frames.extend(clause_frames);
        }
    }
    (frames, clauses)
}
fn sentence_ranges(text: &str) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut start = 0usize;
    for (index, ch) in text.char_indices() {
        if matches!(ch, '。' | '！' | '？' | '；' | '!' | '?' | ';' | '\n') {
            if !text[start..index].trim().is_empty() {
                ranges.push((start, index));
            }
            start = index + ch.len_utf8();
        }
    }
    if !text[start..].trim().is_empty() {
        ranges.push((start, text.len()));
    }
    ranges
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_events_with_roles_time_and_stable_ids() {
        let text = "1991年，张三认识李四。";
        let first = EventGraph::build(text);
        let second = EventGraph::build(text);
        assert_eq!(first.events.len(), 1, "{:?}", first.stats);
        assert_eq!(first.events[0].id, second.events[0].id);
        let event = &first.events[0];
        assert_eq!(event.argument(EventRole::Agent), Some("张三"));
        assert_eq!(event.argument(EventRole::Patient), Some("李四"));
        assert_eq!(event.time_text().as_deref(), Some("1991"));
        assert!(event.evidence.sentence.contains("张三认识李四"));
        assert!(!event.id.is_empty());
    }

    #[test]
    fn explicit_causal_connectors_create_directed_links() {
        let text = "因为张三认识李四，所以王五加入公司。";
        let graph = EventGraph::build(text);
        assert!(graph.events.iter().any(|event| event.trigger == "认识"));
        assert!(graph.events.iter().any(|event| event.trigger == "加入"));
        let cause = graph
            .events
            .iter()
            .find(|event| event.trigger == "认识")
            .expect("原因事件");
        let effect = graph
            .events
            .iter()
            .find(|event| event.trigger == "加入")
            .expect("结果事件");
        assert!(
            graph.links.iter().any(|link| {
                link.from == cause.id && link.to == effect.id && link.kind == EventLinkKind::Causes
            }),
            "{:?}",
            graph.links
        );
        let answer = graph.ask("为什么王五加入公司？", None);
        assert_eq!(answer.status, EventAnswerStatus::Answered, "{:?}", answer);
        assert!(answer.answer.contains("张三"));
    }

    #[test]
    fn explicit_times_order_events_and_answer_when() {
        let text = "1999年，张三加入公司。1991年，张三认识李四。";
        let graph = EventGraph::build(text);
        assert_eq!(graph.events.len(), 2, "{:?}", graph.stats);
        let ordered = graph.ordered_events();
        assert_eq!(ordered[0].trigger, "认识");
        assert_eq!(ordered[1].trigger, "加入");
        assert!(graph
            .links
            .iter()
            .any(|link| link.kind == EventLinkKind::Before));
        let answer = graph.ask("张三什么时候认识李四？", None);
        assert_eq!(answer.status, EventAnswerStatus::Answered);
        assert!(answer.answer.contains("1991"));
    }

    #[test]
    fn missing_cause_abstains_instead_of_guessing() {
        let graph = EventGraph::build("1991年，张三认识李四。");
        let answer = graph.ask("为什么张三认识李四？", None);
        assert_eq!(answer.status, EventAnswerStatus::Abstained);
        assert!(answer.reason.as_deref().unwrap_or("").contains("显式因果"));
    }

    #[test]
    fn dialogue_focus_carries_across_turns() {
        let graph = EventGraph::build("1991年，张三认识李四。1992年，张三加入公司。");
        let mut dialogue = EventDialogue::new(graph);
        let first = dialogue.ask("谁认识李四？");
        assert_eq!(first.status, EventAnswerStatus::Answered);
        assert!(first.answer.contains("张三"));
        let next = dialogue.ask("后来呢？");
        assert_eq!(next.status, EventAnswerStatus::Answered, "{:?}", next);
        assert!(next.answer.contains("加入"), "{:?}", next);
        assert!(dialogue.focus().ids().len() >= 2);
    }

    #[test]
    fn nquads_materialize_only_into_a_named_event_graph() {
        let graph = EventGraph::build("1991年，张三认识李四。");
        let nquads = graph.to_nquads();
        let graph_iri = graph.default_graph_iri();
        assert!(nquads.contains(&graph_iri));
        assert!(nquads.contains("event#Event"));
        assert!(nquads.contains("rdf-syntax-ns#Statement"));
        assert!(nquads.contains("event#projectsTo"));
        assert!(nquads
            .lines()
            .all(|line| line.trim_end().ends_with(&format!("<{}> .", graph_iri))));
        let custom = graph.to_nquads_in_graph("http://example.org/custom");
        assert!(custom
            .lines()
            .all(|line| line.ends_with("<http://example.org/custom> .")));
    }

    #[test]
    fn coreference_candidates_are_reported_not_merged() {
        let graph =
            EventGraph::build("1991年，张三认识李四。1991年，张三认识李四。1999年，张三认识李四。");
        assert_eq!(graph.events.len(), 3, "{:?}", graph.stats);
        let candidates = graph.coreference_candidates();
        assert!(candidates
            .iter()
            .any(|candidate| candidate.score >= 0.8 && candidate.reason.contains("时间")));
        assert!(candidates
            .iter()
            .all(|candidate| candidate.left != candidate.right));
    }

    #[test]
    fn prolog_program_ships_derived_rules_alongside_facts() {
        let graph = EventGraph::build("1991年，张三认识李四。");
        let program = graph.to_prolog_program();
        assert!(program.contains("event_role("));
        assert!(program.contains("event_before(A, B) :- event_link(A, before, B)."));
        assert!(program.contains("event_participant(E, P)"));
        let mut engine = crate::prolog_engine::PrologEngine::new();
        engine
            .consult_source(program)
            .expect("事件程序应可被 Prolog 引擎装载");
        let participants = engine.query("event_participant(E, P)");
        assert!(!participants.is_empty(), "角色规则应可查询");
    }

    #[test]
    fn prolog_projection_keeps_event_identity_and_evidence() {
        let graph = EventGraph::build("1991年，张三认识李四。");
        let lines = graph.to_prolog_lines();
        assert!(lines.iter().any(|line| line.starts_with("event('evt:")));
        assert!(lines.iter().any(|line| line.contains("event_role(")));
        assert!(lines.iter().any(|line| line.contains("event_projects_to(")));
        assert!(lines.iter().any(|line| line.contains("event_evidence(")));
    }
}

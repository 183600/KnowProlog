/// 描述性指称识别：关系名词（「姚明的妻子」）与作品-角色（「演《无间道》的演员」）。
pub fn detect_descriptions(text: &str, source: &dyn SenseSource) -> Vec<DescriptionQuery> {
    let chars: Vec<char> = text.chars().collect();
    let mut out: Vec<DescriptionQuery> = Vec::new();
    let mut seen: Vec<(String, String)> = Vec::new();

    // 1) 关系名词：头实体（≤4 字的已登记实体）+ 的 + 关系词
    for relation in RELATION_NOUNS {
        let relation_chars: Vec<char> = relation.chars().collect();
        let rel_len = relation_chars.len();
        if rel_len == 0 || chars.len() < rel_len {
            continue;
        }
        for index in 0..=chars.len() - rel_len {
            if chars[index..index + rel_len] != relation_chars[..] {
                continue;
            }
            if index == 0 || chars[index - 1] != '的' {
                continue;
            }
            // 只有关系名词处于问句尾部时才是描述性指称。
            // 「张三的师傅的师傅是谁」中的第一个「的师傅」是链式路径的一环，
            // 若在此改写会把两跳问句截成单跳，必须放行给推理器。
            let tail: String = chars[index + rel_len..].iter().collect();
            let tail = tail
                .trim()
                .trim_end_matches(['？', '?', '。', '！', '!', ' '])
                .trim_end_matches('呢')
                .trim_end_matches(['？', '?', '。', '！', '!', ' ']);
            if !tail.is_empty()
                && !matches!(
                    tail,
                    "是谁" | "是什么" | "是哪个" | "是哪位" | "是哪里" | "叫什么" | "叫啥"
                )
            {
                continue;
            }
            let de_index = index - 1;
            let mut head: Option<String> = None;
            for head_len in (1..=4.min(de_index)).rev() {
                let candidate: String = chars[de_index - head_len..de_index].iter().collect();
                if candidate.chars().all(is_han_char) && source.known_entity(&candidate) {
                    head = Some(candidate);
                    break;
                }
            }
            let Some(head) = head else {
                continue;
            };
            if seen.iter().any(|(h, r)| h == &head && r == relation) {
                continue;
            }
            seen.push((head.clone(), (*relation).to_string()));
            let rewritten = format!("{}的{}是谁", head, relation);
            out.push(DescriptionQuery {
                kind: DescriptionKind::Relational,
                head: Some(head.clone()),
                relation: Some((*relation).to_string()),
                role: None,
                title: None,
                rewritten: Some(rewritten.clone()),
                note: format!(
                    "描述性指称（关系名词）：「{}的{}」→ 路径查询「{}」",
                    head, relation, rewritten
                ),
            });
        }
    }

    // 2) 作品-角色：动词 + 《作品》 + 的 + 角色词
    let mut index = 0;
    while index < chars.len() {
        if chars[index] != '《' {
            index += 1;
            continue;
        }
        let Some(close_offset) = chars[index + 1..].iter().position(|ch| *ch == '》') else {
            index += 1;
            continue;
        };
        let close = index + 1 + close_offset;
        let title: String = chars[index + 1..close].iter().collect();
        // 角色词紧随「》的」之后
        if chars.get(close + 1) == Some(&'的') {
            let tail: String = chars[close + 2..].iter().collect();
            if let Some(role) = ROLE_WORDS
                .iter()
                .filter(|role| tail.starts_with(*role))
                .max_by_key(|role| role.chars().count())
            {
                let verb = if index > 0 {
                    DESCRIPTION_VERBS
                        .iter()
                        .find(|(verb, _)| {
                            chars.get(index - 1) == Some(&verb.chars().next().unwrap())
                        })
                        .map(|(_, relation)| (*relation).to_string())
                        .unwrap_or_else(|| "出演".to_string())
                } else {
                    "出演".to_string()
                };
                let rewritten = format!("{}的{}是谁", title, role);
                out.push(DescriptionQuery {
                    kind: DescriptionKind::ByTitle,
                    head: None,
                    relation: Some(verb),
                    role: Some((*role).to_string()),
                    title: Some(title.clone()),
                    rewritten: Some(rewritten.clone()),
                    note: format!(
                        "描述性指称（作品-角色）：「《{}》的{}」→ 路径查询「{}」",
                        title, role, rewritten
                    ),
                });
            }
        }
        index = close + 1;
    }

    out
}
/// 澄清回答解析：`Choose / Restate / Correct / Unknown / Decline`。
///
/// 返回 `None` 表示这不像对当前问题的回答（应放行给常规管线）。
pub fn parse_clarify_act(input: &str, question: &ClarifyQuestion) -> Option<ClarifyAct> {
    let trimmed = input
        .trim()
        .trim_end_matches(['？', '?', '。', '！', '!', ' '])
        .trim();
    if trimmed.is_empty() {
        return None;
    }
    const DECLINE: [&str; 8] = [
        "别问了",
        "不问了",
        "算了",
        "取消",
        "跳过",
        "不用了",
        "没事",
        "换个话题",
    ];
    const UNKNOWN: [&str; 8] = [
        "不知道",
        "不清楚",
        "不确定",
        "都行",
        "随便",
        "无所谓",
        "你说呢",
        "都可能是",
    ];
    if DECLINE.iter().any(|marker| trimmed.contains(marker)) {
        return Some(ClarifyAct::Decline);
    }
    if UNKNOWN.iter().any(|marker| trimmed.contains(marker)) {
        return Some(ClarifyAct::Unknown);
    }
    // 新问题混入（「那苹果的CEO呢」）：不把它当成对澄清问题的一次回答。
    if trimmed.ends_with('呢')
        || ["什么", "为什么", "怎么", "谁", "哪", "多少"]
            .iter()
            .any(|marker| trimmed.contains(marker))
    {
        return None;
    }

    // 序号：1 / 第1个 / 一
    if let Some(index) = parse_option_index(trimmed, question.options.len()) {
        return Some(ClarifyAct::Choose {
            index,
            entity: question.options[index].entity.clone(),
        });
    }

    let mentioned: Vec<&ClarifyOption> = {
        let mut hits: Vec<(&ClarifyOption, usize)> = question
            .options
            .iter()
            .filter_map(|option| trimmed.find(&option.entity).map(|index| (option, index)))
            .collect();
        hits.sort_by(|a, b| {
            a.1.cmp(&b.1)
                .then_with(|| b.0.entity.chars().count().cmp(&a.0.entity.chars().count()))
        });
        hits.into_iter().map(|(option, _)| option).collect()
    };
    let after_is = extract_after_is(trimmed);

    if trimmed.contains("不是") {
        let negated = mentioned.first();
        let affirmed = mentioned
            .last()
            .filter(|option| negated.map(|n| n.entity != option.entity).unwrap_or(true));
        if let (Some(negated), Some(affirmed)) = (negated, affirmed) {
            return Some(ClarifyAct::Correct {
                from: negated.entity.clone(),
                to: affirmed.entity.clone(),
            });
        }
        if let Some(negated) = negated {
            // 「不是 A」：只有两个选项时取另一个；否则需更多信息
            if question.options.len() == 2 {
                let other = question
                    .options
                    .iter()
                    .find(|option| option.entity != negated.entity)?;
                return Some(ClarifyAct::Choose {
                    index: question
                        .options
                        .iter()
                        .position(|option| option.entity == other.entity)?,
                    entity: other.entity.clone(),
                });
            }
            if let Some(entity) = after_is {
                return Some(ClarifyAct::Restate { entity });
            }
            return Some(ClarifyAct::Unknown);
        }
        if let Some(entity) = after_is {
            return Some(ClarifyAct::Restate { entity });
        }
        return None;
    }

    if let Some(option) = mentioned.first() {
        let index = question
            .options
            .iter()
            .position(|candidate| candidate.entity == option.entity)?;
        return Some(ClarifyAct::Choose {
            index,
            entity: option.entity.clone(),
        });
    }

    // 类型线索（「是公司」「水果」）：短回答里出现的唯一类型标签
    if trimmed.chars().count() <= 10 {
        let typed: Vec<&ClarifyOption> = question
            .options
            .iter()
            .filter(|option| {
                option
                    .etype
                    .as_deref()
                    .map(|etype| type_cue_hit(etype, trimmed))
                    .unwrap_or(false)
            })
            .collect();
        if typed.len() == 1 {
            let option = typed[0];
            let index = question
                .options
                .iter()
                .position(|candidate| candidate.entity == option.entity)?;
            return Some(ClarifyAct::Choose {
                index,
                entity: option.entity.clone(),
            });
        }
    }

    // 「就是那个演……」式重述
    if let Some(entity) = after_is {
        return Some(ClarifyAct::Restate { entity });
    }
    None
}
/// 按下标重写指称（跨字节安全；越界/不匹配时回退为首次出现替换）。
pub fn rewrite_at(text: &str, start: usize, mention_len: usize, replacement: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    if mention_len == 0 || start + mention_len > chars.len() {
        return text.to_string();
    }
    let mut out: String = chars[..start].iter().collect();
    out.push_str(replacement);
    out.extend(chars[start + mention_len..].iter());
    out
}
/// 按词面重写首次出现（用于无法确定跨度的场合）。
pub fn rewrite_mention(text: &str, mention: &str, replacement: &str) -> String {
    match text.find(mention) {
        Some(byte_index) => {
            let start = text[..byte_index].chars().count();
            rewrite_at(text, start, mention.chars().count(), replacement)
        }
        None => text.to_string(),
    }
}
fn parse_option_index(trimmed: &str, len: usize) -> Option<usize> {
    let core = trimmed
        .trim_start_matches(['第', '选'])
        .trim_end_matches(['个', '项', '号', '是', '的', '。', '.'])
        .trim();
    if core.is_empty() {
        return None;
    }
    if core.chars().all(|ch| ch.is_ascii_digit()) {
        let value: usize = core.parse().ok()?;
        if value >= 1 && value <= len {
            return Some(value - 1);
        }
        return None;
    }
    let chinese = ["一", "二", "三", "四", "五"];
    chinese
        .iter()
        .position(|word| core == *word)
        .filter(|index| *index < len)
}
/// 类型标签的词面线索（用户说「是公司」也能命中「组织」）。
fn type_cue_hit(etype: &str, text: &str) -> bool {
    if text.contains(etype) {
        return true;
    }
    let cues: &[&str] = if etype.contains("组织") {
        &["公司", "企业", "机构", "集团", "组织"]
    } else if etype.contains("人物") || etype.to_lowercase().contains("person") {
        &["人物", "人"]
    } else if etype.contains("地点") {
        &["地点", "城市", "地方"]
    } else if etype.contains("时间") {
        &["时间", "日期", "时候"]
    } else {
        &[]
    };
    cues.iter().any(|cue| text.contains(cue))
}
fn extract_after_is(text: &str) -> Option<String> {
    let position = text.rfind('是')?;
    let tail: String = text[position + '是'.len_utf8()..].chars().collect();
    let tail = tail.split(['，', ',', '、']).next().unwrap_or("");
    let cleaned = tail
        .trim()
        .trim_start_matches('的')
        .trim_end_matches(['的', '啊', '吧', '呀', '呢', '哦', '了', '！', '!', '。'])
        .trim();
    if cleaned.is_empty() || cleaned.chars().count() > 12 {
        return None;
    }
    Some(cleaned.to_string())
}
fn discriminating_predicate(
    options: &[&CandidateVote],
    source: &dyn SenseSource,
) -> Option<String> {
    let mut predicates: Vec<String> = Vec::new();
    for vote in options {
        if let Some(card) = source.sense_card(&vote.entity) {
            for predicate in card.all_predicates() {
                if !predicates.contains(&predicate) {
                    predicates.push(predicate);
                }
            }
        }
    }
    predicates.sort();
    for predicate in predicates {
        let present: Vec<bool> = options
            .iter()
            .map(|vote| {
                source
                    .sense_card(&vote.entity)
                    .map(|card| card.has_predicate(&predicate))
                    .unwrap_or(false)
            })
            .collect();
        if present.iter().all(|p| *p) {
            // 都有该谓词：看取值是否不同
            let values: Vec<Vec<String>> = options
                .iter()
                .map(|vote| {
                    let mut out = source.predicate_values(&vote.entity, &predicate);
                    out.extend(source.predicate_values_in(&vote.entity, &predicate));
                    out.sort();
                    out.dedup();
                    out
                })
                .collect();
            let distinct: Vec<&Vec<String>> = {
                let mut distinct: Vec<&Vec<String>> = Vec::new();
                for value in &values {
                    if !value.is_empty() && !distinct.contains(&value) {
                        distinct.push(value);
                    }
                }
                distinct
            };
            if distinct.len() >= 2 {
                return Some(predicate);
            }
            continue;
        }
        if present.iter().any(|p| *p) {
            return Some(predicate);
        }
    }
    None
}
fn discriminating_anchor(options: &[&CandidateVote], source: &dyn SenseSource) -> Option<String> {
    let mut anchors: Vec<String> = Vec::new();
    for vote in options {
        if let Some(card) = source.sense_card(&vote.entity) {
            for anchor in card.anchors {
                if !anchors.contains(&anchor) {
                    anchors.push(anchor);
                }
            }
        }
    }
    anchors.sort();
    for anchor in anchors {
        let present: Vec<bool> = options
            .iter()
            .map(|vote| {
                source
                    .sense_card(&vote.entity)
                    .map(|card| card.anchors.contains(&anchor))
                    .unwrap_or(false)
            })
            .collect();
        if present.iter().any(|p| *p) && !present.iter().all(|p| *p) {
            return Some(anchor);
        }
    }
    None
}
fn relation_expectation(relation: &str) -> Option<&'static [&'static str]> {
    const PERSON: [&str; 2] = ["人", "person"];
    const ORG: [&str; 4] = ["组织", "公司", "机构", "organization"];
    match relation {
        "创始人" | "董事长" | "总裁" | "总经理" | "CEO" | "校长" | "院长" => {
            Some(&ORG)
        }
        _ => Some(&PERSON),
    }
}
fn type_matches(etype: &str, expectations: &[String]) -> bool {
    let lowered = etype.to_lowercase();
    expectations.iter().any(|keyword| {
        etype.contains(keyword.as_str()) || lowered.contains(&keyword.to_lowercase())
    })
}
fn is_person_type(etype: &str) -> bool {
    etype.contains('人') || etype.to_lowercase().contains("person")
}
fn looks_like_person_name(surface: &str) -> bool {
    let chars: Vec<char> = surface.chars().collect();
    (2..=4).contains(&chars.len())
        && chars.iter().all(|ch| is_han_char(*ch))
        && infer_suffix_type(surface).is_none()
}
fn shares_bigram(a: &str, b: &str) -> bool {
    let grams = |text: &str| -> Vec<(char, char)> {
        let chars: Vec<char> = text.chars().collect();
        chars
            .windows(2)
            .map(|window| (window[0], window[1]))
            .collect()
    };
    let a_grams = grams(a);
    let b_grams = grams(b);
    a_grams
        .iter()
        .any(|gram| b_grams.iter().any(|other| other == gram))
}
fn is_han_char(ch: char) -> bool {
    ('\u{4e00}'..='\u{9fff}').contains(&ch)
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[derive(Default)]
    struct MapSenseSource {
        cards: HashMap<String, SenseCard>,
        values: HashMap<(String, String), Vec<String>>,
        values_in: HashMap<(String, String), Vec<String>>,
    }

    impl MapSenseSource {
        fn add(
            &mut self,
            entity: &str,
            etype: &str,
            out: &[&str],
            anchors: &[&str],
            degree: usize,
        ) {
            self.cards.insert(
                entity.to_string(),
                SenseCard {
                    entity: entity.to_string(),
                    etype: Some(etype.to_string()),
                    out_predicates: out.iter().map(|p| p.to_string()).collect(),
                    in_predicates: Vec::new(),
                    anchors: anchors.iter().map(|a| a.to_string()).collect(),
                    degree,
                },
            );
        }

        fn add_value(&mut self, entity: &str, predicate: &str, value: &str) {
            self.values
                .entry((entity.to_string(), predicate.to_string()))
                .or_default()
                .push(value.to_string());
        }
    }

    impl SenseSource for MapSenseSource {
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

    fn apple_linker() -> EntityLinker {
        let mut linker = EntityLinker::new();
        linker.register_entity("苹果", Some("水果"));
        linker.register_entity("苹果公司", Some("组织"));
        linker.register_alias("苹果", "苹果公司");
        linker
    }

    fn resolutions(text: &str, linker: &EntityLinker, source: &dyn SenseSource) -> CrdOutcome {
        ContextualResolver::new().resolve_text(
            text,
            linker,
            source,
            &BindingLedger::new(),
            &[],
            None,
        )
    }

    #[test]
    fn ambiguous_mention_asks_type_discriminating_question() {
        let linker = apple_linker();
        let mut source = MapSenseSource::default();
        source.add("苹果", "水果", &["颜色"], &[], 2);
        source.add("苹果公司", "组织", &["创始人"], &[], 3);
        let outcome = resolutions("苹果是什么", &linker, &source);
        let ask = outcome.ask.expect("应升级为显式歧义");
        assert_eq!(ask.mention, "苹果");
        assert_eq!(ask.question.options.len(), 2);
        assert!(matches!(ask.question.feature, DiscriminatingFeature::Type));
        let rendered = ask.question.render();
        assert!(rendered.contains("水果"));
        assert!(rendered.contains("组织") || rendered.contains("公司"));
    }

    #[test]
    fn relational_noun_constraint_resolves_executive_query() {
        let linker = apple_linker();
        let mut source = MapSenseSource::default();
        source.add("苹果", "水果", &["颜色"], &[], 2);
        source.add("苹果公司", "组织", &["创始人"], &[], 3);
        let outcome = resolutions("苹果的CEO是谁", &linker, &source);
        assert!(outcome.ask.is_none(), "关系名词 CEO 期待组织，足以裁决");
        assert_eq!(outcome.resolutions[0].verdict, Verdict::Unique);
        assert_eq!(outcome.resolutions[0].chosen.as_deref(), Some("苹果公司"));
        assert_eq!(outcome.rewrite.as_deref(), Some("苹果公司的CEO是谁"));
    }

    #[test]
    fn bound_mention_wins_without_asking() {
        let linker = apple_linker();
        let mut source = MapSenseSource::default();
        source.add("苹果", "水果", &[], &[], 0);
        source.add("苹果公司", "组织", &[], &[], 0);
        let mut ledger = BindingLedger::new();
        ledger.begin_turn();
        let confirmation = ledger
            .bind("苹果", "苹果公司", BindingScope::Topic, "第一轮澄清")
            .expect("绑定成功");
        assert_eq!(confirmation.binding.id, "rb-1-1");
        let outcome = ContextualResolver::new().resolve_text(
            "苹果的CEO是谁",
            &linker,
            &source,
            &ledger,
            &[],
            None,
        );
        assert!(outcome.ask.is_none());
        assert_eq!(outcome.rewrite.as_deref(), Some("苹果公司的CEO是谁"));
        assert!(outcome.notes.iter().any(|note| note.contains("rb-1-1")));
        let resolution = &outcome.resolutions[0];
        assert_eq!(resolution.verdict, Verdict::Bound);
        assert_eq!(resolution.chosen.as_deref(), Some("苹果公司"));
    }

    #[test]
    fn binding_expires_on_turn_and_topic_boundaries() {
        let mut ledger = BindingLedger::new();
        ledger.begin_turn();
        ledger.bind("它", "张三", BindingScope::Turn, "本回合");
        ledger.bind("苹果", "苹果公司", BindingScope::Topic, "话题");
        ledger.bind("小明", "小明", BindingScope::Session, "会话");
        ledger.begin_turn();
        assert!(ledger.lookup("它").is_none(), "Turn 绑定随回合过期");
        assert!(ledger.lookup("苹果").is_some());
        ledger.end_topic();
        assert!(ledger.lookup("苹果").is_none(), "Topic 绑定随话题过期");
        assert!(ledger.lookup("小明").is_some(), "Session 绑定保留");
    }

    #[test]
    fn override_marks_old_binding_and_correction_rebinds() {
        let mut ledger = BindingLedger::new();
        ledger.begin_turn();
        ledger.bind("苹果", "苹果公司", BindingScope::Topic, "第一轮");
        let confirmation = ledger
            .correct("苹果", "苹果果实", "用户纠错：不是苹果公司")
            .expect("改绑成功");
        assert_eq!(confirmation.binding.entity, "苹果果实");
        assert_eq!(
            ledger.lookup("苹果").map(|b| b.entity.as_str()),
            Some("苹果果实")
        );
        let report = ledger.report();
        assert!(report.contains("已覆盖"));
        assert!(report.contains("苹果公司"));
    }

    #[test]
    fn second_independent_confirmation_nominates_alias_once() {
        let mut ledger = BindingLedger::new();
        ledger.begin_turn();
        ledger.bind("苹果", "苹果公司", BindingScope::Topic, "第一轮");
        assert!(ledger.take_alias_nominations().is_empty());
        ledger.begin_turn();
        let confirmation = ledger
            .bind("苹果", "苹果公司", BindingScope::Topic, "第二轮")
            .expect("追加确认");
        assert!(confirmation.second_signature);
        let nominations = ledger.take_alias_nominations();
        assert_eq!(nominations.len(), 1);
        assert_eq!(nominations[0].surface, "苹果");
        assert_eq!(nominations[0].canonical, "苹果公司");
        assert_eq!(nominations[0].confirmations, 2);
        assert!(
            ledger.take_alias_nominations().is_empty(),
            "同一绑定只提名一次"
        );
    }

    #[test]
    fn ledger_snapshot_roundtrip() {
        let mut ledger = BindingLedger::new();
        ledger.begin_turn();
        ledger.bind("苹果", "苹果公司", BindingScope::Session, "快照");
        let json = ledger.snapshot_json().expect("序列化");
        let restored = BindingLedger::from_snapshot_json(&json).expect("反序列化");
        assert_eq!(restored.turn(), ledger.turn());
        assert_eq!(
            restored.lookup("苹果").map(|b| b.entity.as_str()),
            Some("苹果公司")
        );
    }

    #[test]
    fn type_mismatch_is_hard_rejected() {
        let mut linker = EntityLinker::new();
        linker.register_entity("张三", Some("人物"));
        linker.register_entity("张三公司", Some("组织"));
        linker.register_alias("张三", "张三公司");
        let mut source = MapSenseSource::default();
        source.add("张三", "人物", &[], &[], 0);
        source.add("张三公司", "组织", &[], &[], 0);
        let outcome = resolutions("那位张三是谁", &linker, &source);
        let resolution = &outcome.resolutions[0];
        let rejected = resolution
            .candidates
            .iter()
            .find(|vote| vote.entity == "张三公司")
            .expect("应有组织候选");
        assert!(rejected.hard_rejected.is_some());
        assert_eq!(resolution.verdict, Verdict::Unique);
        assert_eq!(resolution.chosen.as_deref(), Some("张三"));
    }

    #[test]
    fn person_guard_rejects_single_char_fuzzy_link() {
        let mut linker = EntityLinker::new();
        linker.register_entity("王大明", Some("人物"));
        let mut source = MapSenseSource::default();
        source.add("王大明", "人物", &[], &[], 0);
        let outcome = resolutions("王小明的生日是哪天", &linker, &source);
        if let Some(resolution) = outcome.resolutions.first() {
            assert_eq!(resolution.verdict, Verdict::Nil);
            let rejected = resolution
                .candidates
                .iter()
                .find(|vote| vote.entity == "王大明")
                .expect("模糊候选应进入表决视图");
            assert!(
                rejected
                    .hard_rejected
                    .as_deref()
                    .unwrap_or("")
                    .contains("人名护栏"),
                "人名护栏应否决仅单字近似的候选：{:?}",
                rejected.hard_rejected
            );
        } else {
            panic!("应至少有模糊提及进入视图");
        }
    }

    #[test]
    fn person_guard_allows_typo_with_shared_bigram() {
        let mut linker = EntityLinker::new();
        linker.register_entity("刘德华", Some("人物"));
        let mut source = MapSenseSource::default();
        source.add("刘德华", "人物", &[], &[], 0);
        let outcome = resolutions("刘德桦的出生地是哪里", &linker, &source);
        let resolution = outcome.resolutions.first().expect("应有模糊提及");
        assert_eq!(resolution.verdict, Verdict::Unique);
        assert_eq!(resolution.chosen.as_deref(), Some("刘德华"));
    }

    #[test]
    fn quantifier_and_appositive_add_type_expectation() {
        let mut linker = EntityLinker::new();
        linker.register_entity("梁朝伟", Some("人物"));
        linker.register_entity("梁朝伟公司", Some("组织"));
        linker.register_alias("梁朝伟", "梁朝伟公司");
        let mut source = MapSenseSource::default();
        source.add("梁朝伟", "人物", &[], &[], 0);
        source.add("梁朝伟公司", "组织", &[], &[], 0);
        let mention = &linker.link_lattice("演员梁朝伟是谁", &[])[0];
        let sketch = sketch_mention("演员梁朝伟是谁", mention);
        assert_eq!(sketch.appositive_role.as_deref(), Some("演员"));
        assert!(sketch.expectation_types.contains(&"人物".to_string()));
    }

    #[test]
    fn learned_classifier_supplements_quantifier_expectation() {
        let mut linker = EntityLinker::new();
        linker.register_entity("梁朝伟", Some("人物"));
        linker.register_entity("梁朝伟公司", Some("组织"));
        linker.register_alias("梁朝伟", "梁朝伟公司");
        let mut memory = LexicalMemory::new(16);
        memory.observe_annotated(
            1,
            &[
                crate::lexical_memory::annotated_word(
                    "名",
                    "q",
                    crate::nlp::TagSource::JiebaTag,
                    0,
                    1,
                ),
                crate::lexical_memory::annotated_word(
                    "教授",
                    "n",
                    crate::nlp::TagSource::JiebaTag,
                    1,
                    3,
                ),
            ],
        );
        let text = "那名梁朝伟是谁";
        let mention = &linker.link_lattice(text, &[])[0];
        let sketch = sketch_mention_with_memory(text, mention, Some(&memory));
        assert_eq!(sketch.quantifier.as_deref(), Some("那名"));
        assert!(sketch.expectation_types.contains(&"人物".to_string()));
    }

    #[test]
    fn parse_clarify_act_variants() {
        let question = ClarifyQuestion {
            mention: "苹果".to_string(),
            feature: DiscriminatingFeature::Type,
            options: vec![
                ClarifyOption {
                    entity: "苹果".to_string(),
                    label: "苹果（水果）".to_string(),
                    etype: Some("水果".to_string()),
                },
                ClarifyOption {
                    entity: "苹果公司".to_string(),
                    label: "苹果公司（组织）".to_string(),
                    etype: Some("组织".to_string()),
                },
            ],
        };
        assert_eq!(
            parse_clarify_act("1", &question),
            Some(ClarifyAct::Choose {
                index: 0,
                entity: "苹果".to_string()
            })
        );
        assert_eq!(
            parse_clarify_act("第二个", &question),
            Some(ClarifyAct::Choose {
                index: 1,
                entity: "苹果公司".to_string()
            })
        );
        assert_eq!(
            parse_clarify_act("苹果公司", &question),
            Some(ClarifyAct::Choose {
                index: 1,
                entity: "苹果公司".to_string()
            })
        );
        assert_eq!(
            parse_clarify_act("是公司", &question),
            Some(ClarifyAct::Choose {
                index: 1,
                entity: "苹果公司".to_string()
            })
        );
        assert_eq!(
            parse_clarify_act("不是苹果，是苹果公司", &question),
            Some(ClarifyAct::Correct {
                from: "苹果".to_string(),
                to: "苹果公司".to_string()
            })
        );
        assert_eq!(
            parse_clarify_act("不是苹果", &question),
            Some(ClarifyAct::Choose {
                index: 1,
                entity: "苹果公司".to_string()
            })
        );
        assert_eq!(
            parse_clarify_act("不知道", &question),
            Some(ClarifyAct::Unknown)
        );
        assert_eq!(
            parse_clarify_act("别问了", &question),
            Some(ClarifyAct::Decline)
        );
        assert_eq!(parse_clarify_act("那苹果的CEO呢", &question), None);
    }

    #[test]
    fn description_relational_and_by_title() {
        let mut source = MapSenseSource::default();
        source.add("姚明", "人物", &[], &[], 0);
        let descriptions = detect_descriptions("姚明的妻子是谁", &source);
        assert_eq!(descriptions.len(), 1);
        assert_eq!(descriptions[0].kind, DescriptionKind::Relational);
        assert_eq!(descriptions[0].head.as_deref(), Some("姚明"));
        assert_eq!(descriptions[0].relation.as_deref(), Some("妻子"));
        assert_eq!(descriptions[0].rewritten.as_deref(), Some("姚明的妻子是谁"));

        let nested = detect_descriptions("姚明的妻子的丈夫是谁", &source);
        assert!(
            nested
                .iter()
                .all(|description| description.kind != DescriptionKind::Relational),
            "{nested:?}"
        );
        let assertion = detect_descriptions("姚明的妻子是叶莉", &source);
        assert!(
            assertion
                .iter()
                .all(|description| description.kind != DescriptionKind::Relational),
            "{assertion:?}"
        );

        let descriptions = detect_descriptions("演《无间道》的演员是谁", &source);
        assert_eq!(descriptions.len(), 1);
        assert_eq!(descriptions[0].kind, DescriptionKind::ByTitle);
        assert_eq!(descriptions[0].title.as_deref(), Some("无间道"));
        assert_eq!(descriptions[0].role.as_deref(), Some("演员"));
        assert_eq!(
            descriptions[0].rewritten.as_deref(),
            Some("无间道的演员是谁")
        );
    }

    #[test]
    fn predicate_feature_splits_same_type_candidates() {
        let mut linker = EntityLinker::new();
        linker.register_entity("张三丰", Some("人物"));
        linker.register_entity("张三丰演员", Some("人物"));
        linker.register_alias("张三丰", "张三丰演员");
        let mut source = MapSenseSource::default();
        source.add("张三丰", "人物", &["徒弟"], &[], 1);
        source.add("张三丰演员", "人物", &["演员"], &[], 1);
        source.add_value("张三丰", "徒弟", "太极");
        source.add_value("张三丰演员", "演员", "无间道");
        let outcome = resolutions("张三丰是谁", &linker, &source);
        let ask = outcome.ask.expect("同类型候选应由谓词区分");
        assert!(matches!(
            ask.question.feature,
            DiscriminatingFeature::Predicate { .. }
        ));
        let labels: Vec<&str> = ask
            .question
            .options
            .iter()
            .map(|option| option.label.as_str())
            .collect();
        assert!(
            labels.iter().any(|label| label.contains("徒弟=太极")),
            "谓词取值应出现在澄清选项里：{:?}",
            labels
        );
        assert!(
            labels.iter().any(|label| label.contains("人物")),
            "无取值候选应回退到类型标签：{:?}",
            labels
        );
    }

    #[test]
    fn unique_verdict_rewrites_alias_to_canonical() {
        let mut linker = EntityLinker::new();
        linker.register_entity("苹果公司", Some("组织"));
        linker.register_alias("果子公司", "苹果公司");
        let mut source = MapSenseSource::default();
        source.add("苹果公司", "组织", &[], &[], 0);
        let outcome = resolutions("果子公司有多少员工", &linker, &source);
        assert!(outcome.ask.is_none());
        assert_eq!(outcome.rewrite.as_deref(), Some("苹果公司有多少员工"));
        assert_eq!(outcome.resolutions[0].verdict, Verdict::Unique);
    }

    #[test]
    fn rewrite_at_is_char_safe() {
        assert_eq!(
            rewrite_at("苹果的CEO是谁", 0, 2, "苹果公司"),
            "苹果公司的CEO是谁"
        );
        assert_eq!(rewrite_at("abc", 0, 9, "x"), "abc");
        assert_eq!(rewrite_at("说苹果", 1, 2, "苹果公司"), "说苹果公司");
    }

    #[test]
    fn vote_quantization_prefers_exact_over_alias_with_margin() {
        let mut linker = EntityLinker::new();
        linker.register_entity("苹果", Some("水果"));
        linker.register_entity("苹果公司", Some("组织"));
        linker.register_alias("苹果", "苹果公司");
        let mut source = MapSenseSource::default();
        source.add("苹果", "水果", &[], &[], 0);
        source.add("苹果公司", "组织", &[], &[], 0);
        // 无类型期待时两候选票差 = 精确 4 - 别名 3 = 1 < 边际 2 → 仍应追问
        let outcome = resolutions("苹果是什么", &linker, &source);
        assert!(outcome.ask.is_some(), "边际不足时不应静默选一个");
        // 边际降为 1 时精确匹配足以裁决
        let outcome = ContextualResolver::new().with_margin(1).resolve_text(
            "苹果是什么",
            &linker,
            &source,
            &BindingLedger::new(),
            &[],
            None,
        );
        assert!(outcome.ask.is_none());
        assert_eq!(outcome.resolutions[0].chosen.as_deref(), Some("苹果"));
    }
}

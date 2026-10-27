/// 描述「前一行的取值相对后一行」的确定关系与差值。
fn relation_words(
    before: &QueryRow,
    before_quantity: &Quantity,
    after: &QueryRow,
    after_quantity: &Quantity,
    positive: &str,
    superlative: &str,
) -> String {
    let comparison = before_quantity.compare(after_quantity);
    match comparison {
        Comparison::Greater | Comparison::Less => {
            let (difference, unit) = before_quantity
                .difference_in_own_unit(after_quantity)
                .map(|(value, unit)| (value.abs(), unit.unwrap_or("")))
                .unwrap_or((0.0, ""));
            // 方向由实际比较结果决定，不依赖可能已过时的排序方向。
            let word = if comparison == Comparison::Greater {
                positive
            } else {
                comparison_lower_word(positive)
            };
            format!(
                "{}（{}）比 {}（{}）{}，相差 {}{}。",
                before.subject,
                before.value,
                after.subject,
                after.value,
                word,
                quantity::format_number(difference),
                unit
            )
        }
        Comparison::Equal => format!(
            "{}（{}）与 {}（{}）并列最{}，取值相同。",
            before.subject, before.value, after.subject, after.value, superlative
        ),
        Comparison::Uncertain => format!(
            "{}（{}）与 {}（{}）的约数区间重叠，无法确定谁更{}。",
            before.subject, before.value, after.subject, after.value, superlative
        ),
        Comparison::Incomparable(reason) => format!(
            "{}（{}）与 {}（{}）不可比（{}）。",
            before.subject,
            before.value,
            after.subject,
            after.value,
            reason.describe()
        ),
    }
}
/// 生成排名结论。直接消费 [`quantity::rank`] 产出的名次元数据，
/// 不再自行重排一遍；不可比行不会混进前 N 名。
fn rank_summary(
    entries: &[quantity::Ranked<QueryRow>],
    limit: usize,
    attribute: &str,
    descending: bool,
    knowledge: &dyn DialogueKnowledge,
) -> String {
    let label = knowledge.predicate_label(attribute);
    let Some(first) = entries.first() else {
        return format!("没有可用于排名的「{}」记录。", label);
    };
    let word = superlative_word(first.quantity.dimension(), &label, descending);
    let limit = limit.max(1);

    if limit == 1 {
        if first.tied {
            let cohort: Vec<&quantity::Ranked<QueryRow>> = entries
                .iter()
                .take_while(|entry| entry.rank == first.rank)
                .collect();
            let names: Vec<&str> = cohort
                .iter()
                .map(|entry| entry.item.subject.as_str())
                .collect();
            if first.uncertain {
                return format!(
                    "「{}」的「{}」约数区间重叠，无法确定谁最{}。",
                    names.join("」「"),
                    label,
                    word
                );
            }
            let values: Vec<&str> = cohort
                .iter()
                .map(|entry| entry.item.value.as_str())
                .collect();
            return format!(
                "「{}」并列最{}（{}）。",
                names.join("」「"),
                word,
                values.join("、")
            );
        }
        return format!(
            "「{}」的「{}」最{}（{}）。",
            first.item.subject, label, word, first.item.value
        );
    }

    let mut output = format!(
        "按「{}」{}的前{}名：",
        label,
        if descending {
            "从高到低"
        } else {
            "从低到高"
        },
        limit
    );
    // 名次是竞赛式名次：并列后跳号；按名次截止可把并列项完整列出。
    for entry in entries.iter().take_while(|entry| entry.rank <= limit) {
        let marker = if entry.uncertain {
            "（约数，不确定）"
        } else if entry.tied {
            "（并列）"
        } else {
            ""
        };
        output.push_str(&format!(
            "\n  {}. {}{}（{}）",
            entry.rank, entry.item.subject, marker, entry.item.value
        ));
    }
    output
}
/// 当前知识里至少有一个可解析数值的属性，供「按哪个属性排名」的澄清列出，
/// 避免用户面对一个不知道有哪些可选项的空问题。
fn rankable_attributes(knowledge: &dyn DialogueKnowledge) -> Vec<String> {
    let mut labels: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for entity in knowledge.all_entities() {
        let mut facts = knowledge.entity_attributes(&entity);
        facts.extend(knowledge.entity_relations(&entity));
        for (key, value) in facts {
            if quantity::parse_quantity(&value).is_some() {
                labels.insert(knowledge.predicate_label(&key));
            }
        }
    }
    labels.into_iter().collect()
}
/// 把「身高」这类短答解析成图谱中的谓词键；找不到就返回 None。
fn resolve_attribute_hint(text: &str, knowledge: &dyn DialogueKnowledge) -> Option<String> {
    let cleaned = text.trim_matches(|ch: char| {
        matches!(ch, '。' | '，' | ',' | '！' | '!' | '？' | '?' | ' ' | '的')
    });
    if cleaned.is_empty() {
        return None;
    }
    let mut attributes: BTreeMap<String, String> = BTreeMap::new();
    for entity in knowledge.all_entities() {
        for (key, _) in knowledge.entity_attributes(&entity) {
            attributes
                .entry(key.clone())
                .or_insert_with(|| knowledge.predicate_label(&key));
        }
        for (key, _) in knowledge.entity_relations(&entity) {
            attributes
                .entry(key.clone())
                .or_insert_with(|| knowledge.predicate_label(&key));
        }
    }
    if let Some(canonical) = knowledge.canonical_predicate(cleaned) {
        if attributes.contains_key(&canonical) {
            return Some(canonical);
        }
    }
    for (key, label) in &attributes {
        if key == cleaned || label == cleaned || cleaned.contains(label.as_str()) {
            return Some(key.clone());
        }
    }
    None
}
fn has_rank_interrogative(text: &str) -> bool {
    ["谁", "哪个", "哪个人", "哪位", "哪些", "什么"]
        .iter()
        .any(|marker| text.contains(marker))
}
fn parse_extreme(text: &str, board_active: bool) -> Option<ExtremeSpec> {
    if let Some((limit, attribute)) = parse_top_n(text) {
        return Some(ExtremeSpec {
            attribute,
            descending: true,
            limit,
        });
    }
    let position = text.find('最')?;
    let after = &text[position + '最'.len_utf8()..];
    let (descending, consumed) = if after.starts_with("年长") {
        (true, "年长".len())
    } else if after.starts_with("年轻") {
        (false, "年轻".len())
    } else {
        let character = after.chars().next()?;
        let descending = match character {
            '高' | '大' | '重' | '多' | '长' | '远' | '快' | '强' | '贵' | '早' | '深'
            | '厚' | '宽' => true,
            '矮' | '小' | '轻' | '少' | '短' | '近' | '慢' | '弱' | '晚' | '浅' | '薄'
            | '窄' | '便' => false,
            _ => return None,
        };
        (descending, character.len_utf8())
    };
    // 「谁最高兴」不是排名：「高」后面必须是一个句法边界，而不是另一个实词。
    if let Some(next) = after[consumed..].chars().next() {
        if !matches!(
            next,
            '的' | '是'
                | '呢'
                | '吗'
                | '啊'
                | '呀'
                | '哦'
                | '嘛'
                | '个'
                | '名'
                | '位'
                | '条'
                | '项'
                | '种'
                | '人'
                | '中'
                | '里'
                | '在'
                | '有'
                | '？'
                | '?'
                | '。'
                | '，'
                | ','
                | ' '
        ) {
            return None;
        }
    }
    if !board_active && !has_rank_interrogative(text) {
        return None;
    }

    let mut candidate = text[..position].trim();
    while let Some(stripped) = candidate
        .strip_suffix('的')
        .or_else(|| candidate.strip_suffix('是'))
        .or_else(|| candidate.strip_suffix('为'))
    {
        candidate = stripped.trim_end();
    }
    for marker in ["请问", "那么", "到底", "究竟", "帮我看看", "看看"] {
        if let Some(stripped) = candidate.strip_prefix(marker) {
            candidate = stripped.trim_start();
        }
    }
    for marker in [
        "哪几个人",
        "哪个人",
        "哪一位",
        "哪几个",
        "哪位",
        "哪个",
        "谁",
        "什么",
    ] {
        while let Some(stripped) = candidate.strip_prefix(marker) {
            candidate = stripped
                .trim_start_matches(['的', '是', '为', ' '])
                .trim();
        }
    }
    let attribute = if candidate.is_empty() {
        None
    } else {
        Some(candidate.to_string())
    };
    let limit = parse_rank_limit(&after[consumed..]).unwrap_or(1);
    Some(ExtremeSpec {
        attribute,
        descending,
        limit,
    })
}
fn parse_top_n(text: &str) -> Option<(usize, Option<String>)> {
    let position = text.find('前')?;
    let after = &text[position + '前'.len_utf8()..];
    let (limit, consumed) = leading_count(after)?;
    let suffix = &after[consumed..];
    if !(suffix.is_empty()
        || suffix.starts_with(['名', '个', '位', '条', '项', '种', '人', '是']))
    {
        return None;
    }
    let mut candidate = text[..position].trim();
    while let Some(stripped) = candidate.strip_suffix('的') {
        candidate = stripped.trim_end();
    }
    for marker in ["哪几个人", "哪个人", "哪一位", "哪几个", "哪位", "哪个", "谁"] {
        while let Some(stripped) = candidate.strip_prefix(marker) {
            candidate = stripped.trim_start_matches(['的', ' ']).trim();
        }
    }
    let attribute = if candidate.is_empty() {
        None
    } else {
        Some(candidate.to_string())
    };
    Some((limit, attribute))
}
fn parse_rank_limit(after: &str) -> Option<usize> {
    let rest = after.trim_start_matches(['的', ' ']);
    let (limit, consumed) = leading_count(rest)?;
    let suffix = &rest[consumed..];
    if suffix.starts_with(['个', '名', '位', '条', '项', '种', '人']) {
        Some(limit)
    } else {
        None
    }
}
fn leading_count(text: &str) -> Option<(usize, usize)> {
    let mut end = 0;
    for (index, character) in text.char_indices() {
        let is_number = character.is_ascii_digit()
            || matches!(
                character,
                '零' | '〇'
                    | '一'
                    | '二'
                    | '两'
                    | '三'
                    | '四'
                    | '五'
                    | '六'
                    | '七'
                    | '八'
                    | '九'
                    | '十'
                    | '百'
                    | '千'
            );
        if is_number {
            end = index + character.len_utf8();
        } else {
            break;
        }
    }
    if end == 0 {
        return None;
    }
    let quantity = quantity::parse_quantity(&text[..end])?;
    let value = quantity.base_value();
    if value <= 0.0 || value.fract().abs() > 1e-9 || value > 100.0 {
        return None;
    }
    Some((value as usize, end))
}
fn sort_value(row: &QueryRow, sort: &SortSpec, _knowledge: &dyn DialogueKnowledge) -> String {
    if let Some(column) = &sort.column {
        if row.predicate.contains(column) || column.contains(row.predicate.as_str()) {
            return row.value.clone();
        }
    }
    row.value.clone()
}
/// 一次数量排序的完整结果：展示行 + 保留名次元数据的可比项 + 跳过数。
struct QuantityRanking {
    /// 可比项（已排序）在前，跨量纲与非数值行在后。
    rows: Vec<QueryRow>,
    /// 可比项及其名次/并列/不确定标记，排名摘要只消费这一份。
    ranked: Vec<quantity::Ranked<QueryRow>>,
    skipped: usize,
}
/// 按数量语义排序：同量纲内换算后比较，跨量纲/非数值行显式排到末尾。
fn rank_rows(
    rows: Vec<QueryRow>,
    sort: &SortSpec,
    knowledge: &dyn DialogueKnowledge,
) -> QuantityRanking {
    let mut parsed: Vec<(QueryRow, Quantity)> = Vec::new();
    let mut unparsed: Vec<QueryRow> = Vec::new();
    for row in rows {
        let raw = sort_value(&row, sort, knowledge);
        match quantity::parse_quantity(&raw) {
            Some(value) => parsed.push((row, value)),
            None => unparsed.push(row),
        }
    }
    let outcome = quantity::rank(
        parsed,
        &RankSpec {
            descending: sort.descending,
            offset: 0,
            limit: usize::MAX,
        },
    );
    let skipped = outcome.skipped.len() + unparsed.len();
    let ranked = outcome.entries;
    let mut sorted: Vec<QueryRow> = ranked.iter().map(|entry| entry.item.clone()).collect();
    sorted.extend(outcome.skipped);
    sorted.extend(unparsed);
    QuantityRanking {
        rows: sorted,
        ranked,
        skipped,
    }
}
/// 排序并返回未参与数值排序的行数；名次摘要走 [`rank_rows`] 的元数据。
fn sort_rows_by_quantity(
    rows: &mut Vec<QueryRow>,
    sort: &SortSpec,
    knowledge: &dyn DialogueKnowledge,
) -> usize {
    let ranking = rank_rows(std::mem::take(rows), sort, knowledge);
    *rows = ranking.rows;
    ranking.skipped
}
fn clean(input: &str) -> String {
    input
        .trim()
        .trim_end_matches(|c: char| {
            matches!(
                c,
                '？' | '?' | '。' | '！' | '!' | '，' | ',' | '；' | ';' | '~' | ' '
            )
        })
        .trim()
        .to_string()
}
fn is_undo(text: &str) -> bool {
    matches!(
        text,
        "回到上一步" | "上一步" | "撤销" | "回退" | "撤销上一步" | "退回一步" | "往回一步"
    )
}
fn is_reset(text: &str) -> bool {
    matches!(
        text,
        "重新开始" | "清空查询" | "清空查询板" | "新查询" | "清空" | "重置查询"
    )
}
fn is_explain(text: &str) -> bool {
    [
        "依据是什么",
        "有什么依据",
        "依据",
        "证据呢",
        "证据",
        "出处",
        "来源是什么",
        "怎么得出",
        "根据什么",
        "为什么",
        "凭什么",
    ]
    .iter()
    .any(|marker| text.contains(marker))
}
fn is_count(text: &str) -> bool {
    [
        "一共多少",
        "总共多少",
        "有多少条",
        "多少条",
        "有几条",
        "几条",
        "总数",
        "共几",
    ]
    .iter()
    .any(|marker| text.contains(marker))
}
fn is_next_page(text: &str) -> bool {
    matches!(
        text,
        "下一页"
            | "下页"
            | "翻页"
            | "还有吗"
            | "还有呢"
            | "还有别的吗"
            | "还有其它吗"
            | "还有其他的吗"
            | "继续看"
            | "再多看几个"
            | "更多"
            | "往后翻"
            | "再下一页"
    )
}
fn is_prev_page(text: &str) -> bool {
    matches!(text, "上一页" | "上页" | "往前翻" | "前一页")
}
fn is_identity_request(text: &str) -> bool {
    matches!(
        text,
        "他是谁"
            | "她是谁"
            | "它是什么"
            | "这个人是谁"
            | "这个实体是什么"
            | "介绍一下"
            | "介绍这个"
            | "这个呢"
            | "他呢"
            | "她呢"
            | "它呢"
    )
}
fn parse_scan(text: &str, knowledge: &dyn DialogueKnowledge) -> Option<ScanSpec> {
    if !is_scan_request(text) {
        return None;
    }
    let body = strip_scan_prefix(text);
    if leading_entity(&body, knowledge).is_some() {
        return None;
    }
    let entity_type = detect_type(text, knowledge);
    let predicate = extract_predicate(text);
    Some(ScanSpec {
        predicate,
        entity_type,
    })
}
fn is_scan_request(text: &str) -> bool {
    const STRONG: [&str; 6] = ["列出", "列举", "罗列", "帮我列出", "所有", "全部"];
    if STRONG.iter().any(|prefix| text.starts_with(prefix)) {
        return true;
    }
    const SOFT: [&str; 6] = ["显示", "查看", "看看", "看一下", "给我看看", "请列出"];
    if SOFT.iter().any(|prefix| text.starts_with(prefix)) {
        return text.contains("所有")
            || text.contains("全部")
            || text.contains("有哪些")
            || text.contains("哪些有")
            || text.contains("哪些实体")
            || text.ends_with("列表")
            || text.ends_with("清单");
    }
    if text.contains("有哪些")
        || text.contains("哪些有")
        || text.contains("哪些实体")
        || text.contains("哪几个实体")
        || text.contains("哪些人")
        || text.contains("哪几个人")
    {
        return true;
    }
    text.ends_with("列表") || text.ends_with("清单")
}
fn strip_scan_prefix(text: &str) -> String {
    let mut body = text.to_string();
    for prefix in [
        "帮我列出",
        "给我看看",
        "请列出",
        "列出",
        "列举",
        "罗列",
        "显示",
        "查看",
        "看看",
        "看一下",
    ] {
        if let Some(rest) = body.strip_prefix(prefix) {
            body = rest.trim().to_string();
            break;
        }
    }
    body
}
/// 请求以具体实体开头时（如「列出姚明的身高」），交给实体问答而不是全表扫描。
fn leading_entity(body: &str, knowledge: &dyn DialogueKnowledge) -> Option<String> {
    let mut best: Option<String> = None;
    for entity in knowledge.all_entities() {
        if entity.is_empty() {
            continue;
        }
        let Some(rest) = body.strip_prefix(&entity) else {
            continue;
        };
        let rest = rest.trim_start();
        if !(rest.is_empty()
            || rest.starts_with(['的', '有', '是', '在', '和', '与', '跟', '，', ',']))
        {
            continue;
        }
        if best
            .as_ref()
            .is_none_or(|current| current.chars().count() < entity.chars().count())
        {
            best = Some(entity);
        }
    }
    best
}
fn detect_type(text: &str, knowledge: &dyn DialogueKnowledge) -> Option<String> {
    use crate::symbolic_generator::localized_type_label;
    let types = knowledge.all_types();
    for entity_type in &types {
        if text.contains(entity_type.as_str()) {
            return Some(entity_type.clone());
        }
    }
    for entity_type in &types {
        let label = localized_type_label(entity_type);
        if label != entity_type.as_str() && text.contains(label) {
            return Some(entity_type.clone());
        }
    }
    const ALIASES: [(&str, &str); 12] = [
        ("人物", "Person"),
        ("人", "Person"),
        ("公司", "Company"),
        ("企业", "Company"),
        ("城市", "City"),
        ("国家", "Country"),
        ("电影", "Film"),
        ("书籍", "Book"),
        ("学校", "School"),
        ("组织", "Organization"),
        ("球员", "Person"),
        ("运动员", "Person"),
    ];
    for (alias, canonical) in ALIASES {
        if !text.contains(alias) {
            continue;
        }
        if types.iter().any(|entity_type| entity_type == canonical) {
            return Some(canonical.to_string());
        }
        let wanted = localized_type_label(canonical);
        if let Some(found) = types
            .iter()
            .find(|entity_type| localized_type_label(entity_type) == wanted)
        {
            return Some(found.clone());
        }
    }
    None
}
fn extract_predicate(text: &str) -> Option<String> {
    let mut body = strip_scan_prefix(text);
    for trim in ["一下", "所有", "全部", "每个", "各个", "各"] {
        while let Some(rest) = body.strip_prefix(trim) {
            body = rest.trim().to_string();
        }
    }

    let mut candidate = None;
    if let Some((_, rest)) = body.split_once("有哪些") {
        candidate = Some(rest.to_string());
    } else if let Some((_, rest)) = body.split_once("哪些有") {
        candidate = Some(rest.to_string());
    } else if let Some(index) = body.find("哪些") {
        let rest = &body[index + "哪些".len()..];
        if let Some((_, tail)) = rest.split_once('有') {
            candidate = Some(tail.to_string());
        }
    }

    let mut candidate = candidate.unwrap_or_else(|| {
        if let Some((_, tail)) = body.split_once('有') {
            if !tail.is_empty() && !tail.starts_with("哪些") {
                tail.to_string()
            } else {
                body.clone()
            }
        } else {
            body.clone()
        }
    });

    for suffix in ["的实体", "的人", "的记录", "的条目"] {
        if let Some(stripped) = candidate.strip_suffix(suffix) {
            candidate = stripped.trim().to_string();
        }
    }
    if let Some((_, tail)) = candidate.rsplit_once('的') {
        candidate = tail.to_string();
    }
    let candidate = candidate
        .trim()
        .trim_end_matches(|c: char| {
            matches!(
                c,
                '？' | '?' | '。' | '！' | '!' | '吗' | '呢' | '吧' | '啊' | '的' | ' '
            )
        })
        .trim();
    for suffix in ["是什么", "是多少", "有哪些", "有谁", "列表", "清单"] {
        if let Some(stripped) = candidate.strip_suffix(suffix) {
            return normalize_predicate(stripped);
        }
    }
    normalize_predicate(candidate)
}
fn normalize_predicate(candidate: &str) -> Option<String> {
    let mut candidate = candidate
        .trim()
        .trim_matches(|c: char| matches!(c, '「' | '」' | '"' | '\'' | '“' | '”'))
        .trim();
    // 「哪些实体」「哪几个人」这类整表问法没有谓词，归一到具体谓词或 None；
    // 否则「哪些实体」会被当成谓词去查值，结果永远为空。
    for prefix in [
        "哪些实体",
        "哪几个实体",
        "哪些",
        "哪几个",
        "哪几",
        "所有的",
        "全体",
    ] {
        if let Some(rest) = candidate.strip_prefix(prefix) {
            candidate = rest.trim();
        }
    }
    if candidate.is_empty()
        || matches!(
            candidate,
            "实体" | "东西" | "记录" | "数据" | "内容" | "人" | "人物" | "条目" | "成员"
        )
    {
        return None;
    }
    Some(candidate.to_string())
}
fn parse_filter(text: &str) -> Option<FilterSpec> {
    let mut rest = None;
    for prefix in ["只看", "只要", "筛选", "过滤", "限定", "其中"] {
        if let Some(stripped) = text.strip_prefix(prefix) {
            rest = Some(stripped.trim());
            break;
        }
    }
    let rest = rest?;
    let rest = rest
        .trim_start_matches(['是', '有', '的', ' '])
        .trim_end_matches(['的', '呢', '吧', ' '])
        .trim();
    if rest.is_empty() {
        return None;
    }
    if let Some(numeric) = parse_numeric_filter(rest) {
        return Some(numeric);
    }
    Some(FilterSpec::Text(rest.to_string()))
}
fn parse_numeric_filter(text: &str) -> Option<FilterSpec> {
    const OPERATORS: [(&str, CompareOp); 18] = [
        ("不低于", CompareOp::Ge),
        ("不少于", CompareOp::Ge),
        ("不小于", CompareOp::Ge),
        ("不超过", CompareOp::Le),
        ("不大于", CompareOp::Le),
        ("大于等于", CompareOp::Ge),
        ("小于等于", CompareOp::Le),
        ("高于", CompareOp::Gt),
        ("低于", CompareOp::Lt),
        ("超过", CompareOp::Gt),
        ("大于", CompareOp::Gt),
        ("小于", CompareOp::Lt),
        ("至少", CompareOp::Ge),
        ("至多", CompareOp::Le),
        ("等于", CompareOp::Eq),
        ("不到", CompareOp::Lt),
        (">=", CompareOp::Ge),
        ("<=", CompareOp::Le),
    ];
    for (marker, op) in OPERATORS {
        if let Some((column, value)) = text.split_once(marker) {
            if let Some(threshold) = quantity::parse_quantity(value) {
                let column = column
                    .trim()
                    .trim_end_matches('的')
                    .trim_end_matches("数值")
                    .trim();
                return Some(FilterSpec::Numeric {
                    column: if column.is_empty() {
                        None
                    } else {
                        Some(column.to_string())
                    },
                    op,
                    threshold,
                });
            }
        }
    }
    for (marker, op) in [
        (">", CompareOp::Gt),
        ("<", CompareOp::Lt),
        ("=", CompareOp::Eq),
    ] {
        if let Some((column, value)) = text.split_once(marker) {
            if let Some(threshold) = quantity::parse_quantity(value) {
                let column = column.trim().trim_end_matches('的').trim();
                return Some(FilterSpec::Numeric {
                    column: if column.is_empty() {
                        None
                    } else {
                        Some(column.to_string())
                    },
                    op,
                    threshold,
                });
            }
        }
    }
    None
}
fn parse_sort(text: &str) -> Option<SortSpec> {
    let descending = text.contains("从高到低")
        || text.contains("从大到小")
        || text.contains("降序")
        || text.contains("倒序");
    let ascending = text.contains("从低到高")
        || text.contains("从小到大")
        || text.contains("升序")
        || text.contains("正序");
    let mut column = None;
    if let Some(index) = text.find('按') {
        let rest = &text[index + '按'.len_utf8()..];
        let mut end = rest.len();
        for marker in [
            "从低到高",
            "从高到低",
            "从小到大",
            "从大到小",
            "升序",
            "降序",
            "倒序",
            "正序",
            "排序",
            "排一下",
            "排",
            "呢",
            "吧",
            "。",
        ] {
            if let Some(position) = rest.find(marker) {
                end = end.min(position);
            }
        }
        let candidate = rest[..end].trim().trim_end_matches('的').trim();
        if !candidate.is_empty() && !matches!(candidate, "照" | "顺序") {
            column = Some(candidate.to_string());
        }
    } else if !(text.contains("排序") || text.contains("排一下") || text.ends_with('排')) {
        return None;
    }
    if !(text.contains('按')
        || ascending
        || descending
        || text.contains("排序")
        || text.ends_with('排'))
    {
        return None;
    }
    Some(SortSpec { column, descending })
}
fn parse_compare(text: &str) -> Option<String> {
    for prefix in ["跟", "和", "与", "同"] {
        if let Some(rest) = text.strip_prefix(prefix) {
            let mut other = rest.trim();
            for suffix in ["相比呢", "相比", "比较呢", "比较", "比呢", "比", "呢"] {
                if let Some(stripped) = other.strip_suffix(suffix) {
                    other = stripped.trim();
                    break;
                }
            }
            if !other.is_empty() {
                return Some(other.to_string());
            }
        }
    }
    None
}
fn parse_ordinal(text: &str) -> Option<usize> {
    let index = text.find('第')?;
    let rest = &text[index + '第'.len_utf8()..];
    let digits: String = rest
        .chars()
        .take_while(|c| {
            !matches!(
                c,
                '个' | '条'
                    | '行'
                    | '位'
                    | '页'
                    | '呢'
                    | '的'
                    | ' '
                    | '高'
                    | '矮'
                    | '大'
                    | '小'
                    | '重'
                    | '轻'
                    | '多'
                    | '少'
                    | '长'
                    | '短'
                    | '远'
                    | '近'
                    | '快'
                    | '慢'
            )
        })
        .collect();
    chinese_number(digits.trim()).map(|ordinal| ordinal.saturating_sub(1))
}
fn parse_page_number(text: &str) -> Option<usize> {
    let index = text.find('第')?;
    let rest = &text[index + '第'.len_utf8()..];
    let digits: String = rest
        .chars()
        .take_while(|c| !matches!(c, '页' | '呢' | '的' | ' '))
        .collect();
    if !rest.chars().any(|c| c == '页') {
        return None;
    }
    let page = chinese_number(digits.trim())?;
    Some(page.saturating_sub(1))
}
fn parse_select_entity(text: &str) -> Option<String> {
    for prefix in ["选择", "切换到", "切到", "选", "看"] {
        if let Some(rest) = text.strip_prefix(prefix) {
            let candidate = rest.trim().trim_end_matches(['呢', '吧', '的', ' ']).trim();
            if !candidate.is_empty()
                && !matches!(
                    candidate,
                    "一下" | "看" | "这个" | "那个" | "第一个" | "第二个"
                )
            {
                return Some(candidate.to_string());
            }
        }
    }
    None
}
fn chinese_number(raw: &str) -> Option<usize> {
    let text = raw.trim();
    if text.is_empty() {
        return None;
    }
    if text.chars().all(|c| c.is_ascii_digit()) {
        return text.parse().ok();
    }
    let mut total = 0usize;
    let mut current = 0usize;
    let mut seen = false;
    for character in text.chars() {
        match character {
            '零' => {
                seen = true;
            }
            '一' => {
                current = 1;
                seen = true;
            }
            '二' | '两' => {
                current = 2;
                seen = true;
            }
            '三' => {
                current = 3;
                seen = true;
            }
            '四' => {
                current = 4;
                seen = true;
            }
            '五' => {
                current = 5;
                seen = true;
            }
            '六' => {
                current = 6;
                seen = true;
            }
            '七' => {
                current = 7;
                seen = true;
            }
            '八' => {
                current = 8;
                seen = true;
            }
            '九' => {
                current = 9;
                seen = true;
            }
            '十' => {
                if current == 0 {
                    current = 1;
                }
                total += current * 10;
                current = 0;
                seen = true;
            }
            '百' => {
                if current == 0 {
                    current = 1;
                }
                total += current * 100;
                current = 0;
                seen = true;
            }
            _ => return None,
        }
    }
    if !seen {
        return None;
    }
    Some(total + current)
}

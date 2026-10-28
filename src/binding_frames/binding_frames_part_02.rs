fn template_names(template: &str) -> Result<Vec<String>, String> {
    let mut names = Vec::new();
    let mut chars = template.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '{' => {
                let mut name = String::new();
                let mut closed = false;
                for next in chars.by_ref() {
                    if next == '}' {
                        closed = true;
                        break;
                    }
                    if next == '{' {
                        return Err("答案模板出现嵌套 `{`".to_string());
                    }
                    name.push(next);
                }
                if !closed {
                    return Err("答案模板存在未闭合的槽位".to_string());
                }
                let name = name.trim().to_string();
                if name.is_empty() {
                    return Err("答案模板出现空槽位 `{}`".to_string());
                }
                names.push(name);
            }
            '}' => return Err("答案模板出现未配对的右花括号".to_string()),
            _ => {}
        }
    }
    Ok(names)
}
fn is_builtin_var(name: &str) -> bool {
    matches!(
        name,
        "values"
            | "value"
            | "objects"
            | "object"
            | "count"
            | "left_value"
            | "right_value"
            | "comparison"
            | "winner"
            | "raw"
            | "slot"
            | "slot_label"
    )
}
fn render_template(template: &str, vars: &BTreeMap<String, String>) -> Option<String> {
    let mut output = String::new();
    let mut chars = template.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '{' => {
                let mut name = String::new();
                let mut closed = false;
                for next in chars.by_ref() {
                    if next == '}' {
                        closed = true;
                        break;
                    }
                    name.push(next);
                }
                if !closed {
                    return None;
                }
                let value = vars.get(name.trim())?;
                output.push_str(value);
            }
            '}' => return None,
            other => output.push(other),
        }
    }
    Some(output)
}

// ----------------------------- 运行层 -----------------------------
/// 槽位来源：可打印、可审计。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum FrameOrigin {
    #[default]
    Explicit,
    Inherited,
    Clarified,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct StoredBinding {
    raw: String,
    origin: FrameOrigin,
    turn: u64,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct FrameSnapshot {
    bindings: BTreeMap<String, String>,
    turn: u64,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct PendingFrame {
    schema: String,
    bindings: BTreeMap<String, String>,
    origins: BTreeMap<String, FrameOrigin>,
    missing: Vec<String>,
    candidates: Vec<String>,
    raw: String,
    prompt: String,
    turn: u64,
}
/// 会话级绑定记忆：环境、上一帧、待澄清帧与计数。
///
/// 空态不写入快照（`DialogueManagerSnapshot` 用 `Option` + skip），
/// 因此未启用绑定帧的会话字节不变。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameMemory {
    #[serde(default)]
    turn: u64,
    #[serde(default)]
    env: BTreeMap<String, StoredBinding>,
    #[serde(default)]
    last: BTreeMap<String, FrameSnapshot>,
    #[serde(default)]
    pending: Option<PendingFrame>,
    #[serde(default)]
    answered: u64,
    #[serde(default)]
    inherited_slots: u64,
    #[serde(default)]
    clarifications_asked: u64,
    #[serde(default)]
    clarifications_bound: u64,
    #[serde(default)]
    clarifications_cancelled: u64,
}
impl Default for FrameMemory {
    fn default() -> Self {
        Self {
            turn: 0,
            env: BTreeMap::new(),
            last: BTreeMap::new(),
            pending: None,
            answered: 0,
            inherited_slots: 0,
            clarifications_asked: 0,
            clarifications_bound: 0,
            clarifications_cancelled: 0,
        }
    }
}
impl FrameMemory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.turn == 0
            && self.env.is_empty()
            && self.last.is_empty()
            && self.pending.is_none()
            && self.answered == 0
            && self.inherited_slots == 0
            && self.clarifications_asked == 0
            && self.clarifications_bound == 0
            && self.clarifications_cancelled == 0
    }

    pub fn turn(&self) -> u64 {
        self.turn
    }

    pub fn answered(&self) -> u64 {
        self.answered
    }

    pub fn pending_prompt(&self) -> Option<&str> {
        self.pending.as_ref().map(|pending| pending.prompt.as_str())
    }

    pub fn report(&self) -> String {
        format!(
            "绑定帧记忆：轮次 {}，回答 {} 次，继承槽 {} 次，澄清 {} 次（绑定 {}、撤销 {}），待澄清 {}。",
            self.turn,
            self.answered,
            self.inherited_slots,
            self.clarifications_asked,
            self.clarifications_bound,
            self.clarifications_cancelled,
            if self.pending.is_some() { 1 } else { 0 }
        )
    }

    fn prune(&mut self, turn: u64) {
        if let Some(pending) = &self.pending {
            if turn.saturating_sub(pending.turn) > FRAME_PENDING_TTL {
                self.pending = None;
            }
        }
        self.env
            .retain(|_, binding| turn.saturating_sub(binding.turn) <= FRAME_ENV_TTL);
        self.last
            .retain(|_, snapshot| turn.saturating_sub(snapshot.turn) <= FRAME_ENV_TTL * 2);
    }

    fn commit(
        &mut self,
        frame: &CompiledFrame,
        bindings: &BTreeMap<String, String>,
        origins: &BTreeMap<String, FrameOrigin>,
        turn: u64,
    ) {
        self.turn = turn;
        self.answered += 1;
        for (name, raw) in bindings {
            if !frame.is_input_slot(name) {
                continue;
            }
            let origin = origins.get(name).copied().unwrap_or(FrameOrigin::Explicit);
            if origin == FrameOrigin::Inherited {
                self.inherited_slots += 1;
            }
            self.env.insert(
                name.clone(),
                StoredBinding {
                    raw: raw.clone(),
                    origin,
                    turn,
                },
            );
        }
        self.last.insert(
            frame.id.clone(),
            FrameSnapshot {
                bindings: bindings.clone(),
                turn,
            },
        );
        while self.env.len() > 64 {
            let oldest = self
                .env
                .iter()
                .min_by_key(|(name, binding)| (binding.turn, name.as_str()))
                .map(|(name, _)| name.clone());
            match oldest {
                Some(name) => {
                    self.env.remove(&name);
                }
                None => break,
            }
        }
    }
}
#[derive(Debug, Clone, Default)]
struct Resolved {
    bindings: BTreeMap<String, String>,
    origins: BTreeMap<String, FrameOrigin>,
    missing: Vec<String>,
}
struct QuestionPlan {
    prompt: String,
    pending: PendingFrame,
}
struct Executed {
    text: String,
    bindings: BTreeMap<String, String>,
    evidence: Vec<String>,
    entity: Option<String>,
    attribute: Option<String>,
}
/// 一轮帧规划结果（对话层据此记账，不直接打印）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameOutcome {
    pub text: String,
    pub plan: Vec<String>,
    pub provenance: Vec<String>,
    pub kind: MoveKind,
    pub entity: Option<String>,
    pub attribute: Option<String>,
}
fn normalize_utterance(input: &str) -> String {
    let mut text = input.trim();
    const POLITE: [&str; 7] = [
        "请问",
        "请告诉我",
        "我想知道",
        "我想问一下",
        "帮我查一下",
        "帮我查查",
        "麻烦问一下",
    ];
    loop {
        let mut changed = false;
        for prefix in POLITE {
            if let Some(rest) = text.strip_prefix(prefix) {
                let rest = rest.trim();
                if !rest.is_empty() {
                    text = rest;
                    changed = true;
                    break;
                }
            }
        }
        if !changed {
            break;
        }
    }
    text.trim()
        .trim_end_matches(|ch: char| {
            matches!(ch, '？' | '?' | '。' | '！' | '!' | ' ' | '～' | '~' | '…')
        })
        .trim()
        .to_string()
}
fn normalize_slot_value(
    slot: &FrameSlotConfig,
    raw: &str,
    knowledge: &dyn DialogueKnowledge,
) -> Option<String> {
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }
    match slot.sort {
        SlotSort::Entity => {
            if knowledge.known_entity(raw) {
                Some(raw.to_string())
            } else {
                knowledge
                    .entities_in(raw)
                    .into_iter()
                    .filter(|entity| entity == raw)
                    .next()
            }
        }
        SlotSort::Number => quantity::parse_quantity(raw).map(|_| raw.to_string()),
        SlotSort::Predicate | SlotSort::Attribute => Some(
            knowledge
                .canonical_predicate(raw)
                .unwrap_or_else(|| raw.to_string()),
        ),
        SlotSort::Text => Some(raw.to_string()),
    }
}
fn display_value(raw: &str, knowledge: &dyn DialogueKnowledge) -> String {
    knowledge.predicate_label(raw)
}
/// 比较结论与胜者。胜者为 `None` 表示不可比或约数区间重叠，调用方不得
/// 把它渲染成任何一方获胜。
struct AttributeComparison {
    winner: Option<String>,
    comparison: String,
}
/// 属性比较的唯一出口：单位换算、量纲校验、约数不确定性与差异量都在这里。
fn compare_attribute_values(
    left: &str,
    right: &str,
    attribute_label: &str,
    left_raw: &str,
    right_raw: &str,
) -> AttributeComparison {
    let (Some(left_quantity), Some(right_quantity)) =
        (quantity::parse_quantity(left_raw), quantity::parse_quantity(right_raw))
    else {
        return AttributeComparison {
            winner: None,
            comparison: format!(
                "「{}」和「{}」的{}至少一侧不是数值，无法比较",
                left, right, attribute_label
            ),
        };
    };
    let word = left_quantity.dimension().comparative_word(attribute_label);
    match left_quantity.compare(&right_quantity) {
        Comparison::Greater => {
            winner_with_difference(left, word, &left_quantity, &right_quantity)
        }
        Comparison::Less => {
            winner_with_difference(right, word, &right_quantity, &left_quantity)
        }
        Comparison::Equal => AttributeComparison {
            winner: Some(left.to_string()),
            comparison: format!("{}和{}一样{}", left, right, word),
        },
        Comparison::Uncertain => AttributeComparison {
            winner: None,
            comparison: format!(
                "「{}」和「{}」的{}约数区间重叠，无法确定谁更{}",
                left, right, attribute_label, word
            ),
        },
        Comparison::Incomparable(reason) => AttributeComparison {
            winner: None,
            comparison: format!(
                "「{}」和「{}」的{}不可比（{}）",
                left,
                right,
                attribute_label,
                reason.describe()
            ),
        },
    }
}
fn winner_with_difference(
    winner: &str,
    word: &str,
    winner_quantity: &Quantity,
    loser_quantity: &Quantity,
) -> AttributeComparison {
    // 差值写回胜者的原始单位（226 厘米 vs 216 厘米 → 10 厘米）。
    let difference = winner_quantity
        .difference_in_own_unit(loser_quantity)
        .ok()
        .map(|(value, unit)| (value.abs(), unit.unwrap_or("")));
    let detail = difference
        .filter(|(value, _)| *value > 1e-9)
        .map(|(value, unit)| {
            format!("（相差{}{}）", quantity::format_number(value), unit)
        })
        .unwrap_or_default();
    AttributeComparison {
        winner: Some(winner.to_string()),
        comparison: format!("{}更{}{}", winner, word, detail),
    }
}
fn relation_objects(
    knowledge: &dyn DialogueKnowledge,
    subject: &str,
    predicate: &str,
) -> Vec<String> {
    let mut forward: Vec<String> = knowledge
        .entity_relations(subject)
        .into_iter()
        .filter(|(candidate, _)| predicate_matches(knowledge, candidate, predicate))
        .map(|(_, object)| object)
        .collect();
    forward.sort();
    forward.dedup();
    if !forward.is_empty() {
        return forward;
    }
    reverse_relation_objects(knowledge, subject, predicate)
}
fn reverse_relation_objects(
    knowledge: &dyn DialogueKnowledge,
    object: &str,
    predicate: &str,
) -> Vec<String> {
    let mut subjects: Vec<String> = knowledge
        .all_entities()
        .into_iter()
        .filter(|candidate| {
            knowledge.entity_relations(candidate).into_iter().any(
                |(candidate_predicate, candidate_object)| {
                    candidate_object == object
                        && predicate_matches(knowledge, &candidate_predicate, predicate)
                },
            )
        })
        .collect();
    subjects.sort();
    subjects.dedup();
    subjects
}
fn predicate_matches(knowledge: &dyn DialogueKnowledge, candidate: &str, predicate: &str) -> bool {
    candidate == predicate
        || knowledge.predicate_label(candidate) == predicate
        || knowledge.predicate_label(predicate) == candidate
        || knowledge.canonical_predicate(predicate).as_deref() == Some(candidate)
}

/// 绑定帧执行图谱模式时的单轮规模上限：超过就截断，不把穷举当答案。
const KG_FRAME_MAX_SOLUTIONS: usize = 64;
const KG_FRAME_MAX_EXPANSIONS: usize = 20_000;
const KG_FACT_ENTITY_LIMIT: usize = 512;

/// 把知识源暴露的实体、属性和出向关系物化成图事实。
///
/// 从本轮槽位种子出发，再吸收 `all_entities` 中有界枚举的实体；关系宾语
/// 继续入队，因此两跳模式不需要第二套执行器。返回值第二项表示实体枚举
/// 是否完整：只有完整时才敢把零解说成「无记录」。
fn kg_facts_from_knowledge(
    knowledge: &dyn DialogueKnowledge,
    seeds: &[String],
) -> (Vec<GraphFact>, bool) {
    let all_entities = knowledge.all_entities();
    let complete = all_entities.len() <= KG_FACT_ENTITY_LIMIT;
    let mut queue = VecDeque::new();
    let mut seen = BTreeSet::new();
    for seed in seeds {
        if seen.insert(seed.clone()) {
            queue.push_back(seed.clone());
        }
    }
    for entity in all_entities.into_iter().take(KG_FACT_ENTITY_LIMIT) {
        if seen.insert(entity.clone()) {
            queue.push_back(entity);
        }
    }

    let mut facts = BTreeSet::new();
    while let Some(entity) = queue.pop_front() {
        if let Some(entity_type) = knowledge.entity_type(&entity) {
            facts.insert((
                0u8,
                entity.clone(),
                "type".to_string(),
                entity_type,
            ));
        }
        for (predicate, value) in knowledge.entity_attributes(&entity) {
            facts.insert((1u8, entity.clone(), predicate, value));
        }
        for (predicate, object) in knowledge.entity_relations(&entity) {
            facts.insert((2u8, entity.clone(), predicate, object.clone()));
            if seen.len() < KG_FACT_ENTITY_LIMIT && seen.insert(object.clone()) {
                queue.push_back(object);
            }
        }
    }

    let facts = facts
        .into_iter()
        .map(|(source, subject, predicate, object)| match source {
            0 => GraphFact::type_assertion(subject, object),
            1 => GraphFact::attribute(subject, predicate, object),
            _ => GraphFact::relation(subject, predicate, object),
        })
        .collect();
    (facts, complete)
}
fn clarification_candidates(
    _frame: &CompiledFrame,
    slot: &FrameSlotConfig,
    resolved: &Resolved,
    knowledge: &dyn DialogueKnowledge,
) -> Vec<String> {
    let mut candidates = match slot.candidates {
        CandidateSource::None => Vec::new(),
        CandidateSource::Entities => knowledge.all_entities(),
        CandidateSource::Attributes | CandidateSource::Predicates => resolved
            .bindings
            .values()
            .filter(|value| knowledge.known_entity(value))
            .flat_map(|entity| knowledge.known_attributes(entity))
            .map(|attribute| display_value(&attribute, knowledge))
            .collect(),
    };
    candidates.sort();
    candidates.dedup();
    candidates.truncate(FRAME_MAX_CANDIDATES);
    candidates
}
enum PendingReply {
    Candidate(String),
    OutOfRange,
    Unmatched,
}
fn pending_reply(
    pending: &PendingFrame,
    slot: &FrameSlotConfig,
    input: &str,
    knowledge: &dyn DialogueKnowledge,
) -> PendingReply {
    if let Some(index) = parse_choice_index(input) {
        if index >= 1 {
            if let Some(candidate) = pending.candidates.get(index - 1) {
                if let Some(value) = normalize_slot_value(slot, candidate, knowledge) {
                    return PendingReply::Candidate(value);
                }
            }
        }
        return PendingReply::OutOfRange;
    }
    if let Some(candidate) = pending
        .candidates
        .iter()
        .find(|candidate| candidate.as_str() == input)
    {
        if let Some(value) = normalize_slot_value(slot, candidate, knowledge) {
            return PendingReply::Candidate(value);
        }
    }
    if pending.candidates.is_empty() {
        if let Some(value) = normalize_slot_value(slot, input, knowledge) {
            return PendingReply::Candidate(value);
        }
    }
    PendingReply::Unmatched
}
fn is_cancel_reply(input: &str) -> bool {
    matches!(
        input,
        "都不是"
            | "算了"
            | "取消"
            | "不用了"
            | "不查了"
            | "不问了"
            | "结束澄清"
            | "停止澄清"
            | "退出澄清"
    )
}
fn parse_choice_index(text: &str) -> Option<usize> {
    let spec = text
        .trim()
        .trim_start_matches('第')
        .trim_end_matches(|ch: char| matches!(ch, '.' | '。' | '、' | '个' | ')' | '）' | ' '));
    if spec.is_empty() {
        return None;
    }
    if spec.chars().all(|ch| ch.is_ascii_digit()) {
        return spec.parse().ok();
    }
    match spec {
        "一" => Some(1),
        "二" => Some(2),
        "三" => Some(3),
        "四" => Some(4),
        "五" => Some(5),
        "六" => Some(6),
        "七" => Some(7),
        "八" => Some(8),
        "九" => Some(9),
        "十" => Some(10),
        _ => None,
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    fn config(source: &str) -> DialogueFramesConfig {
        toml::from_str(source).expect("测试配置应可解析")
    }

    fn program() -> FrameProgram {
        FrameProgram::compile(&config(r#"
enabled = true

[[frames]]
id = "person.attribute"
priority = 20
patterns = ["{entity}的{attribute}是什么", "{attribute}呢", "{attribute}是多少"]
realize = { answer = "{entity}的{attribute}是{values}。" }

[[frames.slots]]
name = "entity"
sort = "entity"
required = true
inherit = true
candidates = "entities"

[[frames.slots]]
name = "attribute"
sort = "attribute"
required = true
inherit = true
candidates = "attributes"
ask = "你想问谁的「{slot}」？"

[[frames.slots]]
name = "value"
sort = "text"

[frames.goal]
kind = "attribute"
entity = "entity"
attribute = "attribute"
output = "value"

[[frames]]
id = "person.compare"
priority = 30
patterns = ["{left}和{right}的{attribute}谁更高"]
realize = { answer = "{left}的{attribute}是{left_value}，{right}的{attribute}是{right_value}；{comparison}。" }

[[frames.slots]]
name = "left"
sort = "entity"
required = true
candidates = "entities"

[[frames.slots]]
name = "right"
sort = "entity"
required = true
candidates = "entities"

[[frames.slots]]
name = "attribute"
sort = "attribute"
required = true
inherit = true
candidates = "attributes"

[[frames.slots]]
name = "winner"
sort = "text"

[frames.goal]
kind = "compare"
left = "left"
right = "right"
attribute = "attribute"
output = "winner"
"#)).expect("配置应可编译")
    }

    struct FakeKnowledge {
        entities: Vec<String>,
        attributes: BTreeMap<String, Vec<(String, String)>>,
        relations: BTreeMap<String, Vec<(String, String)>>,
    }

    impl FakeKnowledge {
        fn new() -> Self {
            let mut attributes = BTreeMap::new();
            attributes.insert(
                "姚明".to_string(),
                vec![
                    ("height".to_string(), "226".to_string()),
                    ("weight".to_string(), "140".to_string()),
                ],
            );
            attributes.insert(
                "叶莉".to_string(),
                vec![("height".to_string(), "190".to_string())],
            );
            let mut relations = BTreeMap::new();
            relations.insert(
                "张三".to_string(),
                vec![("师傅".to_string(), "李四".to_string())],
            );
            relations.insert(
                "李四".to_string(),
                vec![("师傅".to_string(), "王五".to_string())],
            );
            Self {
                entities: vec![
                    "姚明".to_string(),
                    "叶莉".to_string(),
                    "张三".to_string(),
                    "李四".to_string(),
                    "王五".to_string(),
                ],
                attributes,
                relations,
            }
        }
    }

    impl DialogueKnowledge for FakeKnowledge {
        fn entities_in(&self, text: &str) -> Vec<String> {
            self.entities
                .iter()
                .filter(|entity| text.contains(entity.as_str()))
                .cloned()
                .collect()
        }

        fn known_entity(&self, entity: &str) -> bool {
            self.entities.iter().any(|candidate| candidate == entity)
        }

        fn answer(&self, _question: &str) -> Option<String> {
            None
        }

        fn known_attributes(&self, entity: &str) -> Vec<String> {
            let mut values: Vec<String> = self
                .attributes
                .get(entity)
                .map(|pairs| pairs.iter().map(|(key, _)| key.clone()).collect())
                .unwrap_or_default();
            values.extend(
                self.relations
                    .get(entity)
                    .map(|pairs| {
                        pairs
                            .iter()
                            .map(|(key, _)| key.clone())
                            .collect::<Vec<String>>()
                    })
                    .unwrap_or_default(),
            );
            values.sort();
            values.dedup();
            values
        }

        fn attribute_values(&self, entity: &str, attribute: &str) -> Vec<String> {
            self.attributes
                .get(entity)
                .map(|pairs| {
                    pairs
                        .iter()
                        .filter(|(key, _)| {
                            key == attribute
                                || crate::symbolic_generator::canonical_predicate_key(attribute)
                                    == Some(key.as_str())
                        })
                        .map(|(_, value)| value.clone())
                        .collect()
                })
                .unwrap_or_default()
        }

        fn all_entities(&self) -> Vec<String> {
            self.entities.clone()
        }

        fn entity_relations(&self, entity: &str) -> Vec<(String, String)> {
            self.relations.get(entity).cloned().unwrap_or_default()
        }

        fn canonical_predicate(&self, surface: &str) -> Option<String> {
            crate::symbolic_generator::canonical_predicate_key(surface)
                .map(str::to_string)
                .or_else(|| (surface == "师傅" || surface == "徒弟").then(|| surface.to_string()))
        }

        fn predicate_label(&self, predicate: &str) -> String {
            crate::symbolic_generator::localized_predicate_key(predicate)
                .map(str::to_string)
                .unwrap_or_else(|| predicate.to_string())
        }
    }

    #[test]
    fn compile_rejects_undeclared_template_variable() {
        let config = config(
            r#"
enabled = true
[[frames]]
id = "bad"
patterns = ["{entity}是谁"]
realize = { answer = "{missing}" }
[[frames.slots]]
name = "entity"
sort = "entity"
required = true
[frames.goal]
kind = "attribute"
entity = "entity"
attribute = "const:name"
output = "value"
"#,
        );
        let errors = FrameProgram::compile(&config).expect_err("应拒绝未声明变量");
        assert!(errors
            .iter()
            .any(|error| error.code == "frame_output_undeclared"));
        assert!(errors
            .iter()
            .any(|error| error.code == "frame_template_undeclared"));
    }

    #[test]
    fn compile_rejects_required_slot_without_source() {
        let config = config(
            r#"
enabled = true
[[frames]]
id = "bad"
patterns = ["{attribute}呢"]
[[frames.slots]]
name = "entity"
sort = "entity"
required = true
[[frames.slots]]
name = "attribute"
sort = "attribute"
required = true
[[frames.slots]]
name = "value"
sort = "text"
[frames.goal]
kind = "attribute"
entity = "entity"
attribute = "attribute"
output = "value"
"#,
        );
        let errors = FrameProgram::compile(&config).expect_err("应拒绝不可达必填槽");
        assert!(errors
            .iter()
            .any(|error| error.code == "frame_required_unreachable"));
    }

    #[test]
    fn full_question_executes_attribute_goal() {
        let program = program();
        let mut memory = FrameMemory::new();
        let outcome = program
            .plan(&mut memory, "姚明的身高是什么？", &FakeKnowledge::new(), 1)
            .expect("应回答");
        assert!(outcome.text.contains("226"), "{}", outcome.text);
        assert_eq!(memory.answered(), 1);
        assert!(outcome
            .plan
            .iter()
            .any(|plan| plan == "frame:person.attribute"));
    }

    #[test]
    fn followup_inherits_entity_from_environment() {
        let program = program();
        let mut memory = FrameMemory::new();
        let knowledge = FakeKnowledge::new();
        program
            .plan(&mut memory, "姚明的身高是什么", &knowledge, 1)
            .unwrap();
        let outcome = program
            .plan(&mut memory, "体重呢", &knowledge, 2)
            .expect("应继承实体");
        assert!(outcome.text.contains("140"), "{}", outcome.text);
        assert!(outcome
            .provenance
            .iter()
            .any(|note| note.contains("person.attribute")));
    }

    #[test]
    fn clarification_binds_by_candidate_index() {
        let program = program();
        let mut memory = FrameMemory::new();
        let knowledge = FakeKnowledge::new();
        let question = program
            .plan(&mut memory, "身高是多少", &knowledge, 1)
            .expect("缺实体应澄清");
        assert!(question.text.contains("1. 叶莉"), "{}", question.text);
        let outcome = program
            .plan(&mut memory, "1", &knowledge, 2)
            .expect("应完成绑定");
        assert!(outcome.text.contains("叶莉"), "{}", outcome.text);
        assert!(outcome
            .plan
            .iter()
            .any(|plan| plan == "frame_clarification_bound"));
    }

    #[test]
    fn unknown_entity_falls_back_without_state_change() {
        let program = program();
        let mut memory = FrameMemory::new();
        assert!(program
            .plan(&mut memory, "某甲的身高是什么", &FakeKnowledge::new(), 1)
            .is_none());
        assert!(memory.is_empty());
    }

    #[test]
    fn compare_uses_numeric_order_not_lexicographic() {
        let program = program();
        let mut memory = FrameMemory::new();
        let outcome = program
            .plan(
                &mut memory,
                "姚明和叶莉的身高谁更高",
                &FakeKnowledge::new(),
                1,
            )
            .expect("应比较成功");
        assert!(outcome.text.contains("226"), "{}", outcome.text);
        assert!(outcome.text.contains("190"), "{}", outcome.text);
        assert!(outcome.text.contains("姚明更高"), "{}", outcome.text);
    }

    struct UnitKnowledge;

    impl DialogueKnowledge for UnitKnowledge {
        fn entities_in(&self, text: &str) -> Vec<String> {
            ["姚明", "叶莉"]
                .iter()
                .filter(|entity| text.contains(**entity))
                .map(|entity| (*entity).to_string())
                .collect()
        }

        fn known_entity(&self, entity: &str) -> bool {
            matches!(entity, "姚明" | "叶莉")
        }

        fn answer(&self, _question: &str) -> Option<String> {
            None
        }

        fn known_attributes(&self, entity: &str) -> Vec<String> {
            match entity {
                "姚明" | "叶莉" => vec!["height".to_string()],
                _ => Vec::new(),
            }
        }

        fn attribute_values(&self, entity: &str, attribute: &str) -> Vec<String> {
            match (entity, attribute) {
                ("姚明", "height") => vec!["226 厘米".to_string()],
                ("叶莉", "height") => vec!["1.90 米".to_string()],
                _ => Vec::new(),
            }
        }

        fn all_entities(&self) -> Vec<String> {
            vec!["姚明".to_string(), "叶莉".to_string()]
        }

        fn canonical_predicate(&self, surface: &str) -> Option<String> {
            crate::symbolic_generator::canonical_predicate_key(surface).map(str::to_string)
        }

        fn predicate_label(&self, predicate: &str) -> String {
            crate::symbolic_generator::localized_predicate_key(predicate)
                .map(str::to_string)
                .unwrap_or_else(|| predicate.to_string())
        }
    }

    #[test]
    fn compare_parses_units_from_real_knowledge_values() {
        let program = program();
        let mut memory = FrameMemory::new();
        let outcome = program
            .plan(
                &mut memory,
                "姚明和叶莉的身高谁更高",
                &UnitKnowledge,
                1,
            )
            .expect("带单位的值应能比较");
        assert!(outcome.text.contains("姚明更高"), "{}", outcome.text);
        assert!(
            outcome.text.contains("相差36厘米"),
            "比较应给出原始单位的同量纲差值：{}",
            outcome.text
        );
    }

    #[test]
    fn compare_refuses_cross_dimension_and_uncertain_approximations() {
        let cross = compare_attribute_values(
            "法国",
            "珠峰",
            "面积",
            "67万平方公里",
            "8848米",
        );
        assert!(cross.winner.is_none());
        assert!(cross.comparison.contains("量纲不同"), "{}", cross.comparison);

        let approximate =
            compare_attribute_values("甲", "乙", "年龄", "46亿年", "约46亿年");
        assert!(approximate.winner.is_none());
        assert!(
            approximate.comparison.contains("无法确定"),
            "{}",
            approximate.comparison
        );
    }

    #[test]
    fn memory_and_pending_survive_serde_roundtrip() {
        let program = program();
        let mut memory = FrameMemory::new();
        let knowledge = FakeKnowledge::new();
        program
            .plan(&mut memory, "身高是多少", &knowledge, 1)
            .unwrap();
        let json = serde_json::to_string(&memory).expect("应可序列化");
        let restored: FrameMemory = serde_json::from_str(&json).expect("应可反序列化");
        assert_eq!(memory, restored);
        assert!(restored.pending_prompt().is_some());
    }

    #[test]
    fn old_frames_without_optional_fields_still_parse() {
        let config = config(
            r#"
enabled = true
[[frames]]
id = "minimal"
patterns = ["{entity}的{attribute}是什么"]
[[frames.slots]]
name = "entity"
sort = "entity"
required = true
[[frames.slots]]
name = "attribute"
sort = "attribute"
required = true
[[frames.slots]]
name = "value"
sort = "text"
[frames.goal]
kind = "attribute"
entity = "entity"
attribute = "attribute"
output = "value"
"#,
        );
        let program = FrameProgram::compile(&config).expect("应可编译");
        let mut memory = FrameMemory::new();
        assert!(program
            .plan(&mut memory, "姚明的高度是什么", &FakeKnowledge::new(), 1)
            .is_none());
    }
}

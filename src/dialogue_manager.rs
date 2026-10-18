//! 本体驱动的对话管理器：把「对话」本身当作一个本体来建模，
//! 用逻辑程序（Horn 规则 + 分层否定）推导对话策略，再由模板组合生成回答。
//!
//! 设计要点（纯符号主义，无神经网络、无权重训练）：
//! - 对话本体：每一轮是一个 `DialogueTurn` 实例，携带发言、意图（move）、
//!   焦点实体、属性、回答出处（provenance）与命中的策略；
//! - 策略程序：`POLICY_SOURCE` 用 Prolog 风格规则显式声明「什么状态下说什么话」，
//!   每轮把当前信息状态固化为 ABox 事实，交给 `LogicProgram` 做自底向上求值，
//!   推导出 `respond(...)` 策略集合，再由优先级选择落地；
//! - 组合生成：回答不是一次查表，而是由「结论 + 出处说明 + 追问建议」等
//!   discourse 片段按策略组装；支持多轮指代/省略、元话语（回顾、解释依据）、
//!   追问建议与主动续聊。
//!
//! 与 `SymbolicReasoner` 的关系：管理器不重复存储领域知识，通过
//! `DialogueKnowledge` 接口向推理器要「答案 / 实体 / 实体属性表」。


use serde::{Deserialize, Serialize};

use crate::logic_program::{LAtom, LTerm, LogicProgram};
use crate::symbolic_reasoner::SymbolicReasoner;

/// 一轮话语的语用类型（对话本体的「动作类」）。
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MoveKind {
    Greet,
    Thanks,
    Farewell,
    AskAttribute,
    AskIdentity,
    AskBinary,
    Assert,
    MetaRecap,
    MetaWhy,
    ContinueRequest,
    ElaborateRequest,
    Other,
}

impl MoveKind {
    pub fn label(&self) -> &'static str {
        match self {
            MoveKind::Greet => "greet",
            MoveKind::Thanks => "thanks",
            MoveKind::Farewell => "farewell",
            MoveKind::AskAttribute => "ask_attribute",
            MoveKind::AskIdentity => "ask_identity",
            MoveKind::AskBinary => "ask_binary",
            MoveKind::Assert => "assert",
            MoveKind::MetaRecap => "meta_recap",
            MoveKind::MetaWhy => "meta_why",
            MoveKind::ContinueRequest => "continue",
            MoveKind::ElaborateRequest => "elaborate",
            MoveKind::Other => "other",
        }
    }
}

/// 一次话语的本体化解读：动作 + 实体 + 属性 + 规范化问句。
#[derive(Clone, Debug)]
pub struct UtteranceFrame {
    pub raw: String,
    pub kind: MoveKind,
    pub entity: Option<String>,
    pub attribute: Option<String>,
    pub question: Option<String>,
    // 实体来自指代消解或省略补全（而非用户显式提及）
    pub inherited: bool,
}

impl UtteranceFrame {
    fn new(raw: &str, kind: MoveKind) -> Self {
        UtteranceFrame {
            raw: raw.to_string(),
            kind,
            entity: None,
            attribute: None,
            question: None,
            inherited: false,
        }
    }
}

/// 对话本体的一个实例：一轮完整交互。
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DialogueTurn {
    pub index: usize,
    pub user: String,
    pub kind: MoveKind,
    pub entity: Option<String>,
    pub attribute: Option<String>,
    pub answer_excerpt: Option<String>,
    pub provenance: Vec<String>,
    pub plan: Vec<String>,
}

/// 对话管理器快照（随会话持久化）。
///
/// 状态充分性：快照必须覆盖全部可观察状态——`turns`、`focus`、`externals`
/// 之外的 `last_reply` 也要持久化，恢复后的 `summary()` 与中断运行逐字一致。
/// `last_reply` 带 serde 默认值，快照缺少该字段时按默认值恢复。
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DialogueManagerSnapshot {
    pub turns: Vec<DialogueTurn>,
    pub focus: Vec<String>,
    pub externals: Vec<String>,
    #[serde(default)]
    pub last_reply: Vec<String>,
}

/// 管理器的生成结果：文本 + 命中的策略 + 语用类型。
#[derive(Clone, Debug)]
pub struct DialogueReply {
    pub text: String,
    pub plan: Vec<String>,
    pub kind: MoveKind,
}

/// 管理器向知识源提出的接口（由 `SymbolicReasoner` 或测试替身实现）。
pub trait DialogueKnowledge {
    /// 文本中出现的已知实体（规范名，按长度降序更佳）。
    fn entities_in(&self, text: &str) -> Vec<String>;
    /// 实体是否在本体/图谱中登记。
    fn known_entity(&self, entity: &str) -> bool;
    /// 对规范化问句作答。
    fn answer(&self, question: &str) -> Option<String>;
    /// 实体已知的属性键/关系谓词（供追问建议）。
    fn known_attributes(&self, entity: &str) -> Vec<String>;
    /// 字面量属性的取值（结构化检索，缺省为空）。
    fn attribute_values(&self, entity: &str, attribute: &str) -> Vec<String> {
        let _ = (entity, attribute);
        Vec::new()
    }
    /// 实体的生成式简介（画像段落），无法生成时缺省为 None。
    fn describe(&self, entity: &str) -> Option<String> {
        let _ = entity;
        None
    }
}

impl DialogueKnowledge for SymbolicReasoner {
    fn entities_in(&self, text: &str) -> Vec<String> {
        self.entities_mentioned(text)
    }

    fn known_entity(&self, entity: &str) -> bool {
        self.generator().is_known_entity(entity)
    }

    fn answer(&self, question: &str) -> Option<String> {
        SymbolicReasoner::answer(self, question)
    }

    fn known_attributes(&self, entity: &str) -> Vec<String> {
        let mut attributes = self.generator().attribute_keys(entity);
        for predicate in self.generator().relation_predicates(entity) {
            if !attributes.contains(&predicate) {
                attributes.push(predicate);
            }
        }
        attributes
    }

    fn attribute_values(&self, entity: &str, attribute: &str) -> Vec<String> {
        self.generator().attribute_values(entity, attribute)
    }

    fn describe(&self, entity: &str) -> Option<String> {
        self.generator().profile(entity)
    }
}

/// 对话策略程序：显式声明「在什么信息状态下选择什么话语行为」。
/// 每轮把状态固化为 ABox，求值后得到 `respond(...)` 集合。
pub const POLICY_SOURCE: &str = r#"
# —— 对话策略本体（Horn 规则 + 分层否定）——
# 元话语：回顾与解释依据
respond(recap) :- kind(T, meta_recap).
respond(explain_why) :- kind(T, meta_why), has_evidence(T).
respond(why_no_evidence) :- kind(T, meta_why), not has_evidence(T).
# 对话控制：继续说 / 展开细节
respond(elaborate) :- kind(T, elaborate), has_focus(T).
respond(continue_topic) :- kind(T, continue), has_focus(T).
respond(continue_no_focus) :- kind(T, continue), not has_focus(T).
# 知识问答：先消歧，再直答或坦白不知道
respond(clarify_entity) :- kind(T, ask), has_attribute(T), not has_explicit_entity(T), multiple_focus(T).
respond(clarify_entity_no_focus) :- kind(T, ask), has_attribute(T), not has_explicit_entity(T), not has_focus(T).
respond(direct_answer) :- kind(T, ask), entity_known(T), answer_found(T).
respond(admit_ignorance) :- kind(T, ask), entity_known(T), not answer_found(T).
respond(offer_related) :- kind(T, ask), entity_known(T), answer_found(T), has_more_attributes(T).
respond(ask_back) :- kind(T, ask), entity_known(T), not answer_found(T), has_more_attributes(T).
# 寒暄
respond(greet) :- kind(T, greet).
respond(thanks) :- kind(T, thanks).
respond(farewell) :- kind(T, farewell).
"#;

/// 策略优先级：同一轮推导出多条策略时取第一个可落地者。
const REPLY_PRIORITY: [&str; 13] = [
    "recap",
    "explain_why",
    "why_no_evidence",
    "continue_topic",
    "continue_no_focus",
    "elaborate",
    "clarify_entity",
    "clarify_entity_no_focus",
    "direct_answer",
    "admit_ignorance",
    "greet",
    "thanks",
    "farewell",
];

const RECAP_PATTERNS: [&str; 10] = [
    "我们聊了什么",
    "我们聊过什么",
    "我们说到哪",
    "刚才聊了什么",
    "刚才说了什么",
    "聊了什么",
    "说了什么",
    "回顾一下",
    "对话总结",
    "总结一下我们的对话",
];

const WHY_PATTERNS: [&str; 9] = [
    "你怎么知道",
    "你凭什么",
    "为什么这么说",
    "为什么这样说",
    "为什么这样回答",
    "为什么这么回答",
    "依据是什么",
    "有什么依据",
    "怎么得出结论",
];

const CONTINUE_PATTERNS: [&str; 8] = [
    "还有呢",
    "还有吗",
    "再多说点",
    "再多说些",
    "再说说",
    "再说点",
    "展开说说",
    "展开讲",
];

const ELABORATE_PATTERNS: [&str; 8] = [
    "详细一点",
    "详细说说",
    "具体说说",
    "具体一点",
    "说说细节",
    "讲讲细节",
    "详细说明",
    "多给点细节",
];

const GREETINGS: [&str; 12] = [
    "你好", "您好", "你好呀", "嗨", "哈喽", "hi", "hello", "早上好", "中午好", "下午好",
    "晚上好", "在吗",
];

/// 本体化对话管理器。
pub struct DialogueManager {
    turns: Vec<DialogueTurn>,
    focus: Vec<String>,
    externals: Vec<String>,
    policy: LogicProgram,
    last_reply: Vec<String>,
}

impl Default for DialogueManager {
    fn default() -> Self {
        Self::new()
    }
}

impl DialogueManager {
    pub fn new() -> Self {
        let policy = LogicProgram::from_source(POLICY_SOURCE).expect("对话策略程序应可解析");
        DialogueManager {
            turns: Vec::new(),
            focus: Vec::new(),
            externals: Vec::new(),
            policy,
            last_reply: Vec::new(),
        }
    }

    pub fn policy_source() -> &'static str {
        POLICY_SOURCE
    }

    pub fn turn_count(&self) -> usize {
        self.turns.len()
    }

    /// 已记录的对话本体实例（知识/续聊轮），供外部审计与一致性测试。
    pub fn turns(&self) -> &[DialogueTurn] {
        &self.turns
    }

    pub fn focus_entity(&self) -> Option<&str> {
        self.focus.last().map(String::as_str)
    }

    pub fn focus_stack(&self) -> &[String] {
        &self.focus
    }

    pub fn last_reply_plan(&self) -> &[String] {
        &self.last_reply
    }

    /// 记录一次用户输入（含被其他管线处理的命令），供回顾时统计。
    pub fn observe_input(&mut self, input: &str) {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return;
        }
        self.externals.push(trimmed.to_string());
        if self.externals.len() > 24 {
            self.externals.remove(0);
        }
    }

    /// 对话管理器的对外自述（对话本体状态）。
    pub fn summary(&self) -> String {
        let focus = if self.focus.is_empty() {
            "（空）".to_string()
        } else {
            self.focus.join(" → ")
        };
        let last = if self.last_reply.is_empty() {
            "（无）".to_string()
        } else {
            self.last_reply.join(" + ")
        };
        format!(
            "对话本体：{} 轮对话实例；焦点栈 [{}]；最近策略 [{}]；外部指令 {} 条；策略程序 {} 条规则。",
            self.turns.len(),
            focus,
            last,
            self.externals.len(),
            self.policy.rule_count()
        )
    }

    pub fn snapshot(&self) -> DialogueManagerSnapshot {
        DialogueManagerSnapshot {
            turns: self.turns.clone(),
            focus: self.focus.clone(),
            externals: self.externals.clone(),
            last_reply: self.last_reply.clone(),
        }
    }

    pub fn restore(&mut self, snapshot: &DialogueManagerSnapshot) {
        self.turns = snapshot.turns.clone();
        self.focus = snapshot.focus.clone();
        self.externals = snapshot.externals.clone();
        self.last_reply = snapshot.last_reply.clone();
    }

    pub fn save_to_path(&self, path: &str) -> std::io::Result<()> {
        if let Some(parent) = std::path::Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let json = serde_json::to_string_pretty(&self.snapshot())
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::Other, error))?;
        std::fs::write(path, json)
    }

    pub fn restore_from_path(&mut self, path: &str) -> bool {
        let Ok(content) = std::fs::read_to_string(path) else {
            return false;
        };
        let Ok(snapshot) = serde_json::from_str::<DialogueManagerSnapshot>(&content) else {
            return false;
        };
        self.restore(&snapshot);
        true
    }

    /// 纯对话行为（元话语 / 续聊 / 展开 / 寒暄）：可在管线早段拦截。
    pub fn plan_pure_dialogue(
        &mut self,
        input: &str,
        knowledge: &dyn DialogueKnowledge,
    ) -> Option<DialogueReply> {
        let frame = self.analyze(input, knowledge);
        match frame.kind {
            MoveKind::MetaRecap
            | MoveKind::MetaWhy
            | MoveKind::Greet
            | MoveKind::Thanks
            | MoveKind::Farewell
            | MoveKind::ContinueRequest
            | MoveKind::ElaborateRequest => self.execute(frame, knowledge),
            _ => None,
        }
    }

    /// 知识问句的生成式回答：仅在实体已知时才接管（否则让位给既有管线）。
    pub fn plan_reply(
        &mut self,
        input: &str,
        knowledge: &dyn DialogueKnowledge,
    ) -> Option<DialogueReply> {
        let frame = self.analyze(input, knowledge);
        match frame.kind {
            MoveKind::AskAttribute | MoveKind::AskIdentity => self.execute(frame, knowledge),
            _ => None,
        }
    }

    /// 本体化解读：先识别元话语与寒暄，再解析属性/身份问句与断言，
    /// 最后用焦点栈补全指代与省略。
    pub fn analyze(&self, input: &str, knowledge: &dyn DialogueKnowledge) -> UtteranceFrame {
        let raw = input.trim();
        let stripped = raw
            .trim_end_matches(['？', '?', '。', '！', '!', ' '])
            .trim();
        if RECAP_PATTERNS.iter().any(|pattern| stripped.contains(pattern)) {
            return UtteranceFrame::new(raw, MoveKind::MetaRecap);
        }
        if WHY_PATTERNS.iter().any(|pattern| stripped.contains(pattern)) {
            return UtteranceFrame::new(raw, MoveKind::MetaWhy);
        }
        if CONTINUE_PATTERNS.contains(&stripped) {
            return UtteranceFrame::new(raw, MoveKind::ContinueRequest);
        }
        if ELABORATE_PATTERNS.contains(&stripped) {
            return UtteranceFrame::new(raw, MoveKind::ElaborateRequest);
        }
        let lowered = stripped.to_lowercase();
        if GREETINGS.iter().any(|greeting| lowered == *greeting) {
            return UtteranceFrame::new(raw, MoveKind::Greet);
        }
        if ["谢谢", "多谢", "感谢", "辛苦了"]
            .iter()
            .any(|pattern| stripped.contains(pattern))
        {
            return UtteranceFrame::new(raw, MoveKind::Thanks);
        }
        if ["再见", "拜拜", "下次聊", "结束对话", "退出对话"]
            .iter()
            .any(|pattern| stripped.contains(pattern))
        {
            return UtteranceFrame::new(raw, MoveKind::Farewell);
        }

        let explicit: Vec<String> = knowledge.entities_in(raw);
        let focus = self.focus.last().cloned();
        let question_markers = ["谁", "什么", "哪", "多少", "几", "吗", "呢", "怎么"];
        let has_question = question_markers
            .iter()
            .any(|marker| stripped.contains(marker));

        // 属性问句：S的P（是什么/是多少/是谁/怎么样/呢……），
        // 也接受"姚明的身高"这类无问词的简写，但不吸收含"是"的陈述。
        if has_question || !stripped.contains('是') {
            if let Some((head, attribute)) = split_entity_attribute(stripped) {
                let head_entity = explicit
                    .iter()
                    .find(|entity| head.contains(entity.as_str()))
                    .cloned();
                let entity = head_entity.clone().or_else(|| {
                    if head.is_empty() || contains_pronoun(head) {
                        focus.clone()
                    } else {
                        None
                    }
                });
                if let Some(entity) = entity {
                    let inherited = head_entity.is_none();
                    // 统一规范化为「S的P是什么」，让查询规划器稳定解析
                    let question = format!("{}的{}是什么", entity, attribute);
                    return UtteranceFrame {
                        raw: raw.to_string(),
                        kind: MoveKind::AskAttribute,
                        entity: Some(entity),
                        attribute: Some(attribute.to_string()),
                        question: Some(question),
                        inherited,
                    };
                }
            }
        }

        // 身份问句：X是谁 / 什么是X / X是什么
        if let Some(entity) = explicit.first().cloned() {
            if stripped.ends_with("是谁")
                || stripped.ends_with("是什么")
                || stripped.contains("什么是")
            {
                return UtteranceFrame {
                    raw: raw.to_string(),
                    kind: MoveKind::AskIdentity,
                    entity: Some(entity),
                    attribute: None,
                    question: Some(raw.to_string()),
                    inherited: false,
                };
            }
        }

        // 指代问句：他/她/它+X
        if contains_pronoun(stripped) {
            if let Some(entity) = focus.clone() {
                if has_question {
                    return UtteranceFrame {
                        raw: raw.to_string(),
                        kind: MoveKind::AskIdentity,
                        entity: Some(entity.clone()),
                        attribute: None,
                        question: Some(format!("{}是什么", entity)),
                        inherited: true,
                    };
                }
            }
        }

        // 省略问句：X呢 / 属性呢
        if let Some(remainder) = stripped.strip_suffix('呢') {
            let remainder = remainder.trim();
            if let Some(entity) = explicit.first().cloned() {
                // 显式实体的省略问句（「那叶莉呢」）：规范化问句，保证查询可命中
                return UtteranceFrame {
                    raw: raw.to_string(),
                    kind: MoveKind::AskIdentity,
                    entity: Some(entity.clone()),
                    attribute: None,
                    question: Some(format!("{}是什么", entity)),
                    inherited: false,
                };
            }
            if !remainder.is_empty() && !contains_pronoun(remainder) {
                // 属性省略：如"身高呢" —— 由焦点实体补全，交给策略层决定直答或消歧
                let question = match &focus {
                    Some(entity) => format!("{}的{}是什么", entity, remainder),
                    None => raw.to_string(),
                };
                return UtteranceFrame {
                    raw: raw.to_string(),
                    kind: MoveKind::AskAttribute,
                    entity: focus.clone(),
                    attribute: Some(remainder.to_string()),
                    question: Some(question),
                    inherited: true,
                };
            }
            if let Some(entity) = focus.clone() {
                return UtteranceFrame {
                    raw: raw.to_string(),
                    kind: MoveKind::AskIdentity,
                    entity: Some(entity.clone()),
                    attribute: None,
                    question: Some(format!("{}是什么", entity)),
                    inherited: true,
                };
            }
        }

        // 是非问句：交给既有蕴含/溯因管线处理
        if stripped.ends_with('吗') {
            return UtteranceFrame {
                raw: raw.to_string(),
                kind: MoveKind::AskBinary,
                entity: explicit.first().cloned(),
                attribute: None,
                question: Some(raw.to_string()),
                inherited: false,
            };
        }

        // 陈述句：交给既有语义蕴含管线
        if !has_question && explicit.first().is_some() && stripped.contains('是') {
            return UtteranceFrame {
                raw: raw.to_string(),
                kind: MoveKind::Assert,
                entity: explicit.first().cloned(),
                attribute: None,
                question: None,
                inherited: false,
            };
        }

        UtteranceFrame::new(raw, MoveKind::Other)
    }

    /// 执行策略：固化 ABox → 逻辑程序求值 → 选策略 → 组合生成 → 记录对话实例。
    fn execute(
        &mut self,
        frame: UtteranceFrame,
        knowledge: &dyn DialogueKnowledge,
    ) -> Option<DialogueReply> {
        let mut entity = frame.entity.clone();
        let pure_dialogue = matches!(
            frame.kind,
            MoveKind::Greet
                | MoveKind::Thanks
                | MoveKind::Farewell
                | MoveKind::MetaRecap
                | MoveKind::MetaWhy
        );
        if entity.is_none() && !pure_dialogue {
            entity = self.focus.last().cloned();
        }
        let known = entity
            .as_deref()
            .map(|name| knowledge.known_entity(name))
            .unwrap_or(false);
        // 只有属性没有实体（且无焦点）时不外抛：由策略层生成澄清问题；
        // 实体存在但图谱不认识时让位给既有 NLU / 外部知识管线。
        let attribute_only =
            matches!(frame.kind, MoveKind::AskAttribute) && entity.is_none() && frame.attribute.is_some();
        if matches!(frame.kind, MoveKind::AskAttribute | MoveKind::AskIdentity) && !known && !attribute_only
        {
            return None;
        }

        let known_attributes: Vec<String> = match (&entity, known) {
            (Some(name), true) => knowledge.known_attributes(name),
            _ => Vec::new(),
        };
        let asked = frame.attribute.clone();
        let more_attributes: Vec<String> = known_attributes
            .iter()
            .filter(|attribute| Some(attribute.as_str()) != asked.as_deref())
            .filter(|attribute| !self.already_asked(&entity, attribute))
            .cloned()
            .collect();
        let mut answer: Option<String> = if known {
            frame
                .question
                .as_deref()
                .and_then(|question| knowledge.answer(question))
        } else {
            None
        };
        // 结构化兜底：链接预测/推导链答不上来时，直接检索字面量属性取值
        if answer.is_none() && known {
            if let (Some(name), Some(attribute)) = (entity.as_deref(), frame.attribute.as_deref()) {
                answer = self.lookup_attribute(knowledge, name, attribute);
            }
        }
        // 身份问句兜底：用符号生成器输出结构化画像段落
        if answer.is_none() && known && matches!(frame.kind, MoveKind::AskIdentity) {
            if let Some(name) = entity.as_deref() {
                answer = knowledge.describe(name);
            }
        }

        // —— ABox：本轮的信息状态 ——
        let turn_id = format!("t{}", self.turns.len() + 1);
        let kind_atom = match frame.kind {
            MoveKind::MetaRecap => "meta_recap",
            MoveKind::MetaWhy => "meta_why",
            MoveKind::Greet => "greet",
            MoveKind::Thanks => "thanks",
            MoveKind::Farewell => "farewell",
            MoveKind::ContinueRequest => "continue",
            MoveKind::ElaborateRequest => "elaborate",
            MoveKind::AskAttribute | MoveKind::AskIdentity | MoveKind::AskBinary => "ask",
            MoveKind::Assert => "assert",
            MoveKind::Other => "other",
        };
        let mut abox = vec![format!("kind({}, {})", turn_id, kind_atom)];
        if !self.focus.is_empty() {
            abox.push(format!("has_focus({})", turn_id));
        }
        if self.focus.len() >= 2 {
            abox.push(format!("multiple_focus({})", turn_id));
        }
        if self.last_answered_turn().is_some() {
            abox.push(format!("has_evidence({})", turn_id));
        }
        if known {
            abox.push(format!("entity_known({})", turn_id));
        }
        if answer.is_some() {
            abox.push(format!("answer_found({})", turn_id));
        }
        if frame.attribute.is_some() {
            abox.push(format!("has_attribute({})", turn_id));
        }
        if frame.entity.is_some() && !frame.inherited {
            abox.push(format!("has_explicit_entity({})", turn_id));
        }
        if !more_attributes.is_empty() {
            abox.push(format!("has_more_attributes({})", turn_id));
        }

        self.policy.clear_facts();
        for fact in &abox {
            let _ = self.policy.add_fact_str(fact);
        }
        if self.policy.evaluate().is_err() {
            return None;
        }
        let mut plans: Vec<String> = self
            .policy
            .query("respond(X)")
            .ok()?
            .into_iter()
            .filter_map(|env| env.get("X").cloned())
            .collect();
        plans.sort_by_key(|name| {
            REPLY_PRIORITY
                .iter()
                .position(|candidate| candidate == name)
                .unwrap_or(usize::MAX)
        });
        plans.dedup();
        let chosen = plans.first()?.clone();

        let text = match chosen.as_str() {
            "recap" => self.recap_text(),
            "explain_why" => self.explain_last(),
            "why_no_evidence" => {
                "我还没有答过什么问题，暂时没有可解释的依据。你可以先问我一个知识问题，再问我「你怎么知道的」。"
                    .to_string()
            }
            "continue_topic" => self.continue_text(entity.as_deref(), knowledge),
            "continue_no_focus" => {
                "我还不知道你想接着聊哪个实体。先给我一个名字吧，比如「姚明」。".to_string()
            }
            "elaborate" => self.elaborate_text(entity.as_deref(), knowledge),
            "clarify_entity" => self.clarify_text(frame.attribute.as_deref()),
            "clarify_entity_no_focus" => format!(
                "你想问谁的「{}」？先告诉我实体名吧。",
                frame.attribute.as_deref().unwrap_or("这个属性")
            ),
            "direct_answer" => self.compose_direct(
                answer.as_deref().unwrap_or(""),
                entity.as_deref().unwrap_or(""),
                &more_attributes,
            ),
            "admit_ignorance" => self.compose_ignorance(
                entity.as_deref().unwrap_or(""),
                frame.attribute.as_deref(),
                &more_attributes,
            ),
            "greet" => self.greet_text(),
            "thanks" => self.thanks_text(),
            "farewell" => self.farewell_text(),
            _ => return None,
        };

        let provenance = answer
            .as_deref()
            .map(classify_provenance)
            .unwrap_or_default();
        // 知识/续聊轮进入对话本体实例；元话语与寒暄只影响策略，不记入"聊了什么"
        let content_turn = matches!(
            frame.kind,
            MoveKind::AskAttribute
                | MoveKind::AskIdentity
                | MoveKind::AskBinary
                | MoveKind::ContinueRequest
                | MoveKind::ElaborateRequest
        );
        if content_turn {
            self.turns.push(DialogueTurn {
                index: self.turns.len() + 1,
                user: raw_input(&frame),
                kind: frame.kind.clone(),
                entity: entity.clone(),
                attribute: frame.attribute.clone(),
                answer_excerpt: answer.as_deref().map(|text| excerpt(text, 40)),
                provenance,
                plan: plans.clone(),
            });
        }
        if let Some(entity) = &entity {
            self.push_focus(entity);
        }
        self.last_reply = plans.clone();
        if self.externals.last().map(String::as_str) == Some(frame.raw.trim()) {
            self.externals.pop();
        }
        Some(DialogueReply {
            text,
            plan: plans,
            kind: frame.kind,
        })
    }

    fn already_asked(&self, entity: &Option<String>, attribute: &str) -> bool {
        let Some(entity) = entity else {
            return false;
        };
        self.turns.iter().any(|turn| {
            turn.entity.as_deref() == Some(entity.as_str())
                && turn.attribute.as_deref() == Some(attribute)
        })
    }

    fn last_answered_turn(&self) -> Option<&DialogueTurn> {
        self.turns
            .iter()
            .rev()
            .find(|turn| turn.answer_excerpt.is_some())
    }

    fn push_focus(&mut self, entity: &str) {
        if entity.is_empty() || self.focus.last().map(String::as_str) == Some(entity) {
            return;
        }
        self.focus.retain(|focus| focus != entity);
        self.focus.push(entity.to_string());
        while self.focus.len() > 5 {
            self.focus.remove(0);
        }
    }

    /// 多轮回顾：把对话本体实例组织成一段连贯自述。
    pub fn recap_text(&self) -> String {
        if self.turns.is_empty() {
            let mut text = String::from("我们还没有正式开始知识问答。");
            if !self.externals.is_empty() {
                text.push_str(&format!(
                    "你刚才输入了 {} 条指令，但还没有聊到具体实体。",
                    self.externals.len()
                ));
            }
            text.push_str("你可以问我某个人物的属性，比如「姚明的身高是多少」。");
            return text;
        }
        let mut text = format!("我们一共聊了 {} 轮。", self.turns.len());
        let recent: Vec<String> = self
            .turns
            .iter()
            .rev()
            .take(3)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .map(|turn| {
                let topic = match (&turn.entity, &turn.attribute) {
                    (Some(entity), Some(attribute)) => {
                        format!("「{}」的「{}」", entity, attribute)
                    }
                    (Some(entity), None) => format!("「{}」", entity),
                    _ => format!("「{}」", excerpt(&turn.user, 16)),
                };
                format!("第 {} 轮你问的是{}", turn.index, topic)
            })
            .collect();
        text.push_str(&recent.join("；"));
        text.push('。');
        if let Some(focus) = self.focus.last() {
            text.push_str(&format!("当前话题焦点是「{}」。", focus));
        }
        if !self.externals.is_empty() {
            text.push_str(&format!("另外你还输入了 {} 条其他指令。", self.externals.len()));
        }
        text
    }

    /// 元话语：解释上一轮回答的依据（把推理出处「讲」出来）。
    pub fn explain_last(&self) -> String {
        let Some(turn) = self.last_answered_turn() else {
            return "我还没有答过什么问题，暂时没有可解释的依据。先问我一个知识问题吧。"
                .to_string();
        };
        let provenance = if turn.provenance.is_empty() {
            vec!["知识图谱中的事实".to_string()]
        } else {
            turn.provenance.clone()
        };
        let topic = match (&turn.entity, &turn.attribute) {
            (Some(entity), Some(attribute)) => format!("「{}」的「{}」", entity, attribute),
            (Some(entity), None) => format!("「{}」", entity),
            _ => format!("「{}」", excerpt(&turn.user, 16)),
        };
        format!(
            "刚才关于{}的回答，依据是：{}。这些结论都来自知识图谱与规则程序（逐条符号匹配、逻辑推导），没有任何神经网络参与。",
            topic,
            provenance.join("、")
        )
    }

    /// 单条属性的结构化作答：先走推导链/预测，再退到字面量取值检索。
    fn lookup_attribute(
        &self,
        knowledge: &dyn DialogueKnowledge,
        entity: &str,
        attribute: &str,
    ) -> Option<String> {
        let question = format!("{}的{}是什么", entity, attribute);
        if let Some(answer) = knowledge.answer(&question) {
            return Some(answer);
        }
        let values = knowledge.attribute_values(entity, attribute);
        if values.is_empty() {
            None
        } else {
            Some(format!(
                "「{}」的「{}」是{}。",
                entity,
                attribute,
                values.join("、")
            ))
        }
    }

    fn continue_text(&self, entity: Option<&str>, knowledge: &dyn DialogueKnowledge) -> String {
        let Some(entity) = entity else {
            return "先给我一个实体名，我们就有话题可继续了。".to_string();
        };
        let attributes = knowledge.known_attributes(entity);
        for attribute in attributes.iter().filter(|attribute| {
            !self
                .turns
                .iter()
                .any(|turn| turn.entity.as_deref() == Some(entity) && turn.attribute.as_deref() == Some(attribute.as_str()))
        }) {
            if let Some(answer) = self.lookup_attribute(knowledge, entity, attribute) {
                return format!("接着说说「{}」的「{}」：{}", entity, attribute, answer.trim());
            }
        }
        format!(
            "关于「{}」我知道的都讲得差不多了；换个实体，或者直接考我一个问题吧。",
            entity
        )
    }

    fn elaborate_text(&self, entity: Option<&str>, knowledge: &dyn DialogueKnowledge) -> String {
        let Some(entity) = entity else {
            return "先告诉我你想展开讲哪个实体。".to_string();
        };
        let mut parts = Vec::new();
        for attribute in knowledge.known_attributes(entity).iter().take(3) {
            if let Some(answer) = self.lookup_attribute(knowledge, entity, attribute) {
                parts.push(format!("{}：{}", attribute, answer.trim()));
            }
        }
        if parts.is_empty() {
            return format!("关于「{}」我暂时没有可以展开的细节。", entity);
        }
        let body = parts
            .iter()
            .map(|part| format!("  • {}", part))
            .collect::<Vec<_>>()
            .join("\n");
        format!("关于「{}」，我知道这些：\n{}", entity, body)
    }

    fn clarify_text(&self, attribute: Option<&str>) -> String {
        let attribute = attribute.unwrap_or("这个属性");
        let candidates: Vec<String> = self.focus.iter().rev().take(3).cloned().collect();
        match candidates.split_last() {
            Some((last, rest)) if !rest.is_empty() => format!(
                "你想问的「{}」是指「{}」还是「{}」？告诉我名字我就接着答。",
                attribute,
                rest.join("」「"),
                last
            ),
            _ => format!("你想问谁的「{}」？先告诉我实体名吧。", attribute),
        }
    }

    fn compose_direct(&self, answer: &str, entity: &str, more: &[String]) -> String {
        let mut text = answer.trim().to_string();
        if !more.is_empty() {
            let listed = more
                .iter()
                .take(3)
                .cloned()
                .collect::<Vec<_>>()
                .join("」「");
            let variants = [
                format!(
                    "我还知道它的「{}」等 {} 项信息，想了解可以直接问。",
                    listed,
                    more.len()
                ),
                format!(
                    "关于「{}」，我还能补充「{}」等 {} 项。",
                    entity,
                    listed,
                    more.len()
                ),
                format!(
                    "除此之外，「{}」的「{}」我也知道（共 {} 项），继续问我就好。",
                    entity,
                    listed,
                    more.len()
                ),
            ];
            text.push('\n');
            text.push_str(&variants[self.turns.len() % variants.len()]);
        }
        text
    }

    fn compose_ignorance(&self, entity: &str, attribute: Option<&str>, more: &[String]) -> String {
        let mut text = match attribute {
            Some(attribute) => format!(
                "关于「{}」的「{}」，我的知识图谱里暂时没有记录。",
                entity, attribute
            ),
            None => format!("我的知识图谱里暂时没有关于「{}」的介绍信息。", entity),
        };
        if !more.is_empty() {
            let listed = more
                .iter()
                .take(3)
                .cloned()
                .collect::<Vec<_>>()
                .join("」「");
            text.push_str(&format!(
                "\n不过我知道它的「{}」等 {} 项信息，你想了解哪一个？",
                listed,
                more.len()
            ));
        } else {
            text.push_str("\n你可以换一种问法，或者直接告诉我相关事实，我会把它记住。");
        }
        text
    }

    fn greet_text(&self) -> String {
        if self.turns.is_empty() {
            "你好！我是 KnowProlog 的符号主义对话层：知识来自图谱，推理依靠规则与逻辑程序，回答由模板组合生成。你可以直接问「姚明的身高是多少」这样的问题。"
                .to_string()
        } else if let Some(focus) = self.focus.last() {
            format!("又见面了！我们接着聊「{}」也行，换个话题也行。", focus)
        } else {
            "又见面了！继续问我吧。".to_string()
        }
    }

    fn thanks_text(&self) -> String {
        match self.focus.last() {
            Some(focus) => format!("不客气！关于「{}」还有想问的，随时说。", focus),
            None => "不客气！随时继续问我。".to_string(),
        }
    }

    fn farewell_text(&self) -> String {
        format!(
            "好的，再见！这次我们聊了 {} 轮，对话本体已经记入记忆。",
            self.turns.len()
        )
    }
}

fn raw_input(frame: &UtteranceFrame) -> String {
    frame.raw.trim().to_string()
}

/// 把回答文本归类为可宣讲的推理出处（纯符号标记识别）。
fn classify_provenance(answer: &str) -> Vec<String> {
    let mut tags: Vec<String> = Vec::new();
    if answer.contains("链接预测") || answer.contains("预测") {
        tags.push("知识图谱链接预测（符号统计）".to_string());
    }
    if answer.contains("溯因") || answer.contains("假设") {
        tags.push("溯因假设生成（会话级推测）".to_string());
    }
    if answer.contains("归结") || answer.contains("证明") {
        tags.push("一阶归结证明".to_string());
    }
    if answer.contains("本体") || answer.contains("继承") {
        tags.push("OWL 本体推理（层级闭包）".to_string());
    }
    if answer.contains("软推理") || answer.contains("加权") || answer.contains("概率") {
        tags.push("加权软逻辑 / 概率推理".to_string());
    }
    if answer.contains("规则") {
        tags.push("会话规则后向链".to_string());
    }
    if tags.is_empty() {
        tags.push("知识图谱中的直接事实".to_string());
    }
    tags
}

fn excerpt(text: &str, limit: usize) -> String {
    let first_line = text.lines().next().unwrap_or("").trim();
    let mut out: String = first_line.chars().take(limit).collect();
    if first_line.chars().count() > limit {
        out.push('…');
    }
    out
}

fn contains_pronoun(text: &str) -> bool {
    let cleaned = text.replace("其他", "").replace("其中", "").replace("其实", "");
    ["他们", "她们", "它们", "这个人", "那位", "这位", "他", "她", "它"]
        .iter()
        .any(|pronoun| cleaned.contains(pronoun))
}

fn split_entity_attribute(text: &str) -> Option<(&str, &str)> {
    let index = text.find('的')?;
    let head = text[..index].trim();
    let mut tail = text[index + '的'.len_utf8()..].trim();
    for suffix in [
        "是什么", "是多少", "是谁", "是哪个", "是哪里", "怎么样", "多少", "呢", "吗",
    ] {
        if let Some(stripped) = tail.strip_suffix(suffix) {
            tail = stripped.trim();
            break;
        }
    }
    if tail.is_empty() || tail.contains('是') || tail.chars().count() > 16 {
        return None;
    }
    Some((head, tail))
}

/// 构造 `LogicProgram` 时使用的辅助类型别名占位（保持模块内 API 稳定）。
pub type DialogueFact = LAtom;

/// 便捷构造：把实体断言为对话事实（供扩展使用）。
pub fn entity_fact(entity: &str) -> LAtom {
    LAtom::new("entity", vec![LTerm::atom(entity)])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    struct MockKnowledge {
        entities: Vec<String>,
        known: Vec<String>,
        attributes: BTreeMap<String, Vec<String>>,
        answers: BTreeMap<String, String>,
        values: BTreeMap<(String, String), Vec<String>>,
        profiles: BTreeMap<String, String>,
    }

    impl MockKnowledge {
        fn new() -> Self {
            MockKnowledge {
                entities: Vec::new(),
                known: Vec::new(),
                attributes: BTreeMap::new(),
                answers: BTreeMap::new(),
                values: BTreeMap::new(),
                profiles: BTreeMap::new(),
            }
        }

        fn with_value(mut self, entity: &str, attribute: &str, value: &str) -> Self {
            self.values
                .insert((entity.to_string(), attribute.to_string()), vec![value.to_string()]);
            self
        }

        fn with_profile(mut self, entity: &str, profile: &str) -> Self {
            self.profiles
                .insert(entity.to_string(), profile.to_string());
            self
        }

        fn with_entity(mut self, entity: &str, attributes: &[&str]) -> Self {
            self.entities.push(entity.to_string());
            self.known.push(entity.to_string());
            self.attributes.insert(
                entity.to_string(),
                attributes.iter().map(|a| a.to_string()).collect(),
            );
            self
        }

        fn with_answer(mut self, question: &str, answer: &str) -> Self {
            self.answers
                .insert(question.to_string(), answer.to_string());
            self
        }
    }

    impl DialogueKnowledge for MockKnowledge {
        fn entities_in(&self, text: &str) -> Vec<String> {
            let mut found: Vec<String> = self
                .entities
                .iter()
                .filter(|entity| text.contains(entity.as_str()))
                .cloned()
                .collect();
            found.sort_by_key(|entity| std::cmp::Reverse(entity.chars().count()));
            found
        }

        fn known_entity(&self, entity: &str) -> bool {
            self.known.iter().any(|known| known == entity)
        }

        fn answer(&self, question: &str) -> Option<String> {
            self.answers.get(question).cloned()
        }

        fn known_attributes(&self, entity: &str) -> Vec<String> {
            self.attributes.get(entity).cloned().unwrap_or_default()
        }

        fn attribute_values(&self, entity: &str, attribute: &str) -> Vec<String> {
            self.values
                .get(&(entity.to_string(), attribute.to_string()))
                .cloned()
                .unwrap_or_default()
        }

        fn describe(&self, entity: &str) -> Option<String> {
            self.profiles.get(entity).cloned()
        }
    }

    fn mock() -> MockKnowledge {
        MockKnowledge::new()
            .with_entity("姚明", &["身高", "国籍", "出生地"])
            .with_answer("姚明是谁", "姚明是中国篮球运动员。")
            .with_answer("姚明的身高是什么", "姚明的身高是 226 厘米。")
            .with_answer("姚明的国籍是什么", "姚明的国籍是中国。")
    }

    #[test]
    fn test_policy_source_is_stratified() {
        let mut program = LogicProgram::from_source(POLICY_SOURCE).expect("策略程序应可解析");
        program.evaluate().expect("策略程序应可求值");
        assert!(program.rule_count() >= 10);
    }

    #[test]
    fn test_analyze_attribute_question() {
        let manager = DialogueManager::new();
        let frame = manager.analyze("姚明的身高是多少？", &mock());
        assert_eq!(frame.kind, MoveKind::AskAttribute);
        assert_eq!(frame.entity.as_deref(), Some("姚明"));
        assert_eq!(frame.attribute.as_deref(), Some("身高"));
    }

    #[test]
    fn test_analyze_meta_and_smalltalk() {
        let manager = DialogueManager::new();
        let knowledge = mock();
        assert_eq!(
            manager.analyze("我们聊了什么？", &knowledge).kind,
            MoveKind::MetaRecap
        );
        assert_eq!(
            manager.analyze("你怎么知道的？", &knowledge).kind,
            MoveKind::MetaWhy
        );
        assert_eq!(manager.analyze("你好", &knowledge).kind, MoveKind::Greet);
        assert_eq!(manager.analyze("谢谢！", &knowledge).kind, MoveKind::Thanks);
        assert_eq!(manager.analyze("还有呢", &knowledge).kind, MoveKind::ContinueRequest);
    }

    #[test]
    fn test_direct_answer_plan_and_followup_offer() {
        let knowledge = mock();
        let mut manager = DialogueManager::new();
        let reply = manager
            .plan_reply("姚明的身高是多少", &knowledge)
            .expect("已知实体且有答案时应接管");
        assert!(reply.plan.iter().any(|plan| plan == "direct_answer"), "策略：{:?}", reply.plan);
        assert!(reply.plan.iter().any(|plan| plan == "offer_related"));
        assert!(reply.text.contains("226"));
        assert!(reply.text.contains("国籍"), "应附带追问建议：{}", reply.text);
        assert_eq!(manager.turn_count(), 1);
        assert_eq!(manager.focus_entity(), Some("姚明"));
    }

    #[test]
    fn test_admit_ignorance_with_suggestions() {
        let knowledge = mock();
        let mut manager = DialogueManager::new();
        let reply = manager
            .plan_reply("姚明的体重是多少", &knowledge)
            .expect("已知实体即使无答案也应接管（生成式坦白）");
        assert!(
            reply.plan.iter().any(|plan| plan == "admit_ignorance"),
            "策略：{:?}",
            reply.plan
        );
        assert!(reply.plan.iter().any(|plan| plan == "ask_back"));
        assert!(reply.text.contains("没有记录"));
        assert!(reply.text.contains("身高") && reply.text.contains("国籍"));
    }

    #[test]
    fn test_unknown_entity_defers_to_other_pipelines() {
        let knowledge = mock();
        let mut manager = DialogueManager::new();
        assert!(manager.plan_reply("诸葛亮的身高是多少", &knowledge).is_none());
        assert_eq!(manager.turn_count(), 0);
    }

    #[test]
    fn test_pronoun_and_ellipsis_followup() {
        let knowledge = mock();
        let mut manager = DialogueManager::new();
        manager
            .plan_reply("姚明是谁", &knowledge)
            .expect("身份问句应可回答");
        let frame = manager.analyze("他的身高呢", &knowledge);
        assert_eq!(frame.kind, MoveKind::AskAttribute);
        assert_eq!(frame.entity.as_deref(), Some("姚明"));
        assert!(frame.inherited);
        let reply = manager
            .plan_reply("他的身高呢", &knowledge)
            .expect("指代消解后应能回答");
        assert!(reply.text.contains("226"));
    }

    #[test]
    fn test_recap_and_explain_why() {
        let knowledge = mock();
        let mut manager = DialogueManager::new();
        manager.plan_reply("姚明的身高是多少", &knowledge).unwrap();
        let recap = manager
            .plan_pure_dialogue("我们聊了什么？", &knowledge)
            .expect("回顾元话语应被接管");
        assert!(recap.text.contains("聊了 1 轮"), "{}", recap.text);
        assert!(recap.text.contains("姚明"));
        let why = manager
            .plan_pure_dialogue("你怎么知道的？", &knowledge)
            .expect("依据元话语应被接管");
        assert!(why.text.contains("依据"));
        assert!(why.text.contains("知识图谱"));
    }

    #[test]
    fn test_continue_topic_picks_unasked_attribute() {
        let knowledge = mock();
        let mut manager = DialogueManager::new();
        manager.plan_reply("姚明的身高是多少", &knowledge).unwrap();
        let reply = manager
            .plan_pure_dialogue("还有呢", &knowledge)
            .expect("续聊应被接管");
        assert!(reply.plan.iter().any(|plan| plan == "continue_topic"), "{:?}", reply.plan);
        assert!(reply.text.contains("国籍"), "应挑未问过的属性：{}", reply.text);
    }

    #[test]
    fn test_continue_without_focus() {
        let knowledge = mock();
        let mut manager = DialogueManager::new();
        let reply = manager
            .plan_pure_dialogue("还有呢", &knowledge)
            .expect("无焦点时也应给出可行动作");
        assert!(reply.plan.iter().any(|plan| plan == "continue_no_focus"));
        assert!(reply.text.contains("实体"));
    }

    #[test]
    fn test_structured_fallbacks_fill_missing_answers() {
        let knowledge = MockKnowledge::new()
            .with_entity("姚明", &["身高", "国籍"])
            .with_value("姚明", "国籍", "中国")
            .with_profile("姚明", "姚明是中国篮球运动员。");
        let mut manager = DialogueManager::new();
        let reply = manager
            .plan_reply("姚明的国籍是什么", &knowledge)
            .expect("属性取值检索应兜底作答");
        assert!(reply.plan.iter().any(|plan| plan == "direct_answer"));
        assert!(reply.text.contains("中国"));
        let mut manager = DialogueManager::new();
        let reply = manager
            .plan_reply("姚明是谁", &knowledge)
            .expect("身份问句应回落到画像生成");
        assert!(reply.text.contains("中国篮球运动员"));
    }

    #[test]
    fn test_attribute_without_any_entity_asks_which_entity() {
        let knowledge = mock();
        let mut manager = DialogueManager::new();
        let reply = manager
            .plan_reply("首都呢", &knowledge)
            .expect("缺实体时应生成澄清问题而非放弃");
        assert!(
            reply
                .plan
                .iter()
                .any(|plan| plan == "clarify_entity_no_focus"),
            "策略：{:?}",
            reply.plan
        );
        assert!(reply.text.contains("首都"));
        assert!(reply.text.contains("实体名"));
    }

    #[test]
    fn test_snapshot_roundtrip() {
        let knowledge = mock();
        let mut manager = DialogueManager::new();
        manager.plan_reply("姚明是谁", &knowledge).unwrap();
        let path = std::env::temp_dir().join(format!(
            "knowprolog_dialogue_manager_{}.json",
            std::process::id()
        ));
        let path_str = path.to_string_lossy().to_string();
        manager.save_to_path(&path_str).expect("应能保存");
        let mut restored = DialogueManager::new();
        assert!(restored.restore_from_path(&path_str), "应能恢复");
        assert_eq!(restored.turn_count(), 1);
        assert_eq!(restored.focus_entity(), Some("姚明"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_snapshot_preserves_observable_state() {
        let knowledge = mock();
        let mut manager = DialogueManager::new();
        manager.plan_reply("姚明的身高是多少", &knowledge).unwrap();
        manager.plan_pure_dialogue("还有呢", &knowledge).unwrap();
        let snapshot = manager.snapshot();
        let mut restored = DialogueManager::new();
        restored.restore(&snapshot);
        assert_eq!(restored.snapshot(), snapshot);
        assert_eq!(restored.summary(), manager.summary());
        assert_eq!(restored.last_reply_plan(), manager.last_reply_plan());
    }

    #[test]
    fn test_snapshot_deserializes_without_last_reply_field() {
        let stored = r#"{
            "turns": [],
            "focus": ["姚明"],
            "externals": []
        }"#;
        let snapshot: DialogueManagerSnapshot =
            serde_json::from_str(stored).expect("缺少 last_reply 字段的快照应可反序列化");
        assert!(snapshot.last_reply.is_empty());
        let mut manager = DialogueManager::new();
        manager.restore(&snapshot);
        assert_eq!(manager.focus_entity(), Some("姚明"));
    }

    #[test]
    fn test_external_inputs_are_tracked_and_consumed() {
        let knowledge = mock();
        let mut manager = DialogueManager::new();
        manager.observe_input("你好");
        manager.observe_input("谢谢");
        manager.plan_pure_dialogue("谢谢", &knowledge).unwrap();
        let summary = manager.summary();
        assert!(summary.contains("外部指令 1 条"), "{}", summary);
    }

    #[test]
    fn test_polite_replies_use_focus() {
        let knowledge = mock();
        let mut manager = DialogueManager::new();
        manager.plan_reply("姚明是谁", &knowledge).unwrap();
        let thanks = manager.plan_pure_dialogue("谢谢！", &knowledge).unwrap();
        assert!(thanks.text.contains("姚明"));
        let farewell = manager
            .plan_pure_dialogue("再见", &knowledge)
            .unwrap();
        assert!(farewell.text.contains("1 轮"));
    }

    #[test]
    fn test_clarify_entity_with_multiple_focus() {
        let knowledge = MockKnowledge::new()
            .with_entity("姚明", &["身高"])
            .with_entity("叶莉", &["身高"])
            .with_answer("姚明是谁", "姚明是中国篮球运动员。")
            .with_answer("叶莉是谁", "叶莉是中国篮球运动员。");
        let mut manager = DialogueManager::new();
        manager.plan_reply("姚明是谁", &knowledge).unwrap();
        manager.plan_reply("叶莉是谁", &knowledge).unwrap();
        let reply = manager
            .plan_reply("身高呢", &knowledge)
            .expect("属性问句缺实体时应触发消歧");
        assert!(
            reply.plan.iter().any(|plan| plan == "clarify_entity"),
            "策略：{:?}",
            reply.plan
        );
        assert!(reply.text.contains("叶莉") && reply.text.contains("姚明"));
    }
}

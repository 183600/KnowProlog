//! 对话身份层：区分「谁在问谁的身份」，并把身份回答收敛为可审计的身份卡。
//!
//! 与 `identity`（实体 URI / 别名注册表）和 `entity_linker`（指称消歧）的分工：
//! 后者回答「文本里的某个词指哪个实体」；本模块回答三个作用域问题——
//! - 系统身份（第二人称）：「你是谁 / 你叫什么 / 你有意识吗」；
//! - 用户身份（第一人称）：「我是谁 / 我叫什么 / 我叫张三 / 忘记我的名字」；
//! - 实体身份（第三人称）：「姚明是谁」。
//!
//! 三条硬约束：
//! 1. 用户身份只来自用户的明确自述，不推断职业、住址或偏好；
//! 2. 用户自述只活在会话台账里，绝不写入图谱或实体注册表；
//! 3. 身份卡中的每条 claim 都带来源；未知时克制 abstain，不编造。
//!
//! 本模块零依赖、确定性输出；中文渲染只是身份卡的模板化视图，评测读取结构化
//! claim，不解析自然语言句子。

use serde::{Deserialize, Serialize};

use crate::dialogue_manager::DialogueKnowledge;

/// 身份作用域：系统 / 用户 / 图谱实体 / 非身份话语。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IdentityScope {
    System,
    User,
    Entity,
    Other,
}

impl IdentityScope {
    pub fn label(&self) -> &'static str {
        match self {
            IdentityScope::System => "system",
            IdentityScope::User => "user",
            IdentityScope::Entity => "entity",
            IdentityScope::Other => "other",
        }
    }
}

/// 一轮话语的身份动作（只做确定性模式识别，不依赖知识库）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IdentityAct {
    /// 询问系统身份：你是谁 / 你叫什么 / 你有意识吗。
    AskSystem,
    /// 询问用户在本会话中的身份：我是谁 / 你记得我叫什么吗。
    AskUser,
    /// 用户明确自述名字：我叫张三 / 我的名字是张三。
    Introduce { name: String },
    /// 用户要求忘记名字：忘记我的名字 / 不要记住我。
    ForgetUser,
    /// 第三人称实体身份问句：姚明是谁 / 什么是量子力学。
    AskEntity,
    /// 非身份话语。
    Other,
}

impl IdentityAct {
    pub fn scope(&self) -> IdentityScope {
        match self {
            IdentityAct::AskSystem => IdentityScope::System,
            IdentityAct::AskUser | IdentityAct::Introduce { .. } | IdentityAct::ForgetUser => {
                IdentityScope::User
            }
            IdentityAct::AskEntity => IdentityScope::Entity,
            IdentityAct::Other => IdentityScope::Other,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            IdentityAct::AskSystem => "ask_system",
            IdentityAct::AskUser => "ask_user",
            IdentityAct::Introduce { .. } => "introduce",
            IdentityAct::ForgetUser => "forget_user",
            IdentityAct::AskEntity => "ask_entity",
            IdentityAct::Other => "other",
        }
    }
}

/// 身份卡状态：已知 / 未知 / 歧义 / 已更正 / 已遗忘。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IdentityStatus {
    Known,
    Unknown,
    Ambiguous,
    Corrected,
    Forgotten,
}

impl IdentityStatus {
    pub fn label(&self) -> &'static str {
        match self {
            IdentityStatus::Known => "known",
            IdentityStatus::Unknown => "unknown",
            IdentityStatus::Ambiguous => "ambiguous",
            IdentityStatus::Corrected => "corrected",
            IdentityStatus::Forgotten => "forgotten",
        }
    }
}

/// claim 的来源类型（评测据此检查作用域是否越界）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClaimSource {
    /// 系统宪法常量。
    Constitution,
    /// 用户在本会话中的明确自述。
    UserStatement,
    /// 图谱事实。
    Graph,
    /// 身份注册表（别名 / 重定向）。
    Registry,
}

impl ClaimSource {
    pub fn label(&self) -> &'static str {
        match self {
            ClaimSource::Constitution => "系统宪法",
            ClaimSource::UserStatement => "用户明确自述",
            ClaimSource::Graph => "图谱事实",
            ClaimSource::Registry => "身份注册表",
        }
    }

    pub fn scope(&self) -> IdentityScope {
        match self {
            ClaimSource::Constitution => IdentityScope::System,
            ClaimSource::UserStatement => IdentityScope::User,
            ClaimSource::Graph | ClaimSource::Registry => IdentityScope::Entity,
        }
    }
}

/// 身份卡的一条可审计陈述。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IdentityClaim {
    /// 稳定的机器键（`entity:type`、`user:name`……）。
    pub key: String,
    /// 中文展示名。
    pub label: String,
    /// 陈述值。
    pub value: String,
    pub source: ClaimSource,
    /// 来源细节（轮次 / 图谱谓词 / 注册表词面）。
    pub evidence: String,
}

impl IdentityClaim {
    pub fn new(
        key: impl Into<String>,
        label: impl Into<String>,
        value: impl Into<String>,
        source: ClaimSource,
        evidence: impl Into<String>,
    ) -> Self {
        Self {
            key: key.into(),
            label: label.into(),
            value: value.into(),
            source,
            evidence: evidence.into(),
        }
    }

    /// 评测口径：`key=value`。
    pub fn signature(&self) -> String {
        format!("{}={}", self.key, self.value)
    }
}

/// 结构化身份卡：作用域 + 主体 + 状态 + claim 列表。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IdentityCard {
    pub scope: IdentityScope,
    pub subject: String,
    pub status: IdentityStatus,
    pub claims: Vec<IdentityClaim>,
}

impl IdentityCard {
    pub fn new(scope: IdentityScope, subject: impl Into<String>, status: IdentityStatus) -> Self {
        Self {
            scope,
            subject: subject.into(),
            status,
            claims: Vec::new(),
        }
    }

    pub fn with_claims(mut self, claims: Vec<IdentityClaim>) -> Self {
        self.claims = claims;
        self
    }

    pub fn claim_signatures(&self) -> Vec<String> {
        self.claims.iter().map(IdentityClaim::signature).collect()
    }

    /// 作用域越界：claim 的来源与卡片作用域不一致。
    pub fn scope_leaks(&self) -> usize {
        self.claims
            .iter()
            .filter(|claim| claim.source.scope() != self.scope)
            .count()
    }

    pub fn missing_evidence(&self) -> usize {
        self.claims
            .iter()
            .filter(|claim| claim.evidence.trim().is_empty())
            .count()
    }

    /// 确定性中文渲染。实体卡以「追加段」形式呈现，便于接在旧回答之后。
    pub fn render(&self) -> String {
        match self.scope {
            IdentityScope::System => self.render_system(),
            IdentityScope::User => self.render_user(),
            IdentityScope::Entity => self.render_entity(),
            IdentityScope::Other => String::new(),
        }
    }

    fn render_claims(&self) -> String {
        let mut out = String::new();
        for claim in &self.claims {
            out.push_str(&format!(
                "- {}：{}（依据：{}）\n",
                claim.label,
                claim.value,
                if claim.evidence.trim().is_empty() {
                    claim.source.label()
                } else {
                    claim.evidence.as_str()
                }
            ));
        }
        out
    }

    fn render_system(&self) -> String {
        let mut out = String::from("我是 KnowProlog，一个符号主义的中文知识图谱对话系统。\n");
        out.push_str("【身份卡·系统】\n");
        out.push_str(&self.render_claims());
        out.trim_end().to_string()
    }

    fn render_user(&self) -> String {
        match self.status {
            IdentityStatus::Unknown => "你还没有在这个会话里告诉我你的名字。你可以说「我叫……」；我只会记住你明确说出的名字，并且可以随时忘记。".to_string(),
            IdentityStatus::Forgotten => "我已经忘记你的名字了。本会话里我不会再使用它；如果愿意，你可以重新告诉我。".to_string(),
            IdentityStatus::Corrected => {
                let name = self.claims.iter().find(|claim| claim.key == "user:name").map(|claim| claim.value.as_str()).unwrap_or("");
                let mut out = format!("收到，已把称呼更正为「{}」。之前的称呼在本会话里不再使用。\n", name);
                out.push_str("【身份卡·用户】\n");
                out.push_str(&self.render_claims());
                out.trim_end().to_string()
            }
            _ => {
                let name = self.claims.iter().find(|claim| claim.key == "user:name").map(|claim| claim.value.as_str()).unwrap_or("");
                let mut out = format!("你在这个会话里告诉我你叫「{}」。\n", name);
                out.push_str("【身份卡·用户】\n");
                out.push_str(&self.render_claims());
                out.trim_end().to_string()
            }
        }
    }

    fn render_entity(&self) -> String {
        let subject = if self.subject.is_empty() {
            "这个实体"
        } else {
            self.subject.as_str()
        };
        match self.status {
            IdentityStatus::Unknown => format!(
                "知识库里没有「{}」的身份记录，我不会为它编造身份卡。",
                subject
            ),
            IdentityStatus::Ambiguous => {
                let candidates: Vec<&str> = self
                    .claims
                    .iter()
                    .filter(|claim| claim.key == "entity:ambiguous")
                    .map(|claim| claim.value.as_str())
                    .collect();
                if candidates.is_empty() {
                    format!("「{}」对应多个身份，需要先澄清你指的是哪一个。", subject)
                } else {
                    format!(
                        "「{}」对应多个身份：{}。请先澄清你指的是哪一个。",
                        subject,
                        candidates.join(" / ")
                    )
                }
            }
            _ => {
                let mut out = String::from("【身份卡·实体】\n");
                out.push_str(&self.render_claims());
                out.trim_end().to_string()
            }
        }
    }
}

/// 一次用户名字变更的会话内编年记录。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UserNameChange {
    pub previous: Option<String>,
    pub current: String,
    pub turn: usize,
}

/// 用户名字的变更结果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RememberOutcome {
    Remembered,
    Corrected { previous: String },
    Unchanged,
}

/// 用户身份台账：当前值 + 更正历史 + 遗忘墓碑。
///
/// 只接受显式自述；`forget` 会清空当前值与历史值（隐私边界），只留下
/// 「已遗忘」状态。台账随会话快照持久化，不进入图谱。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct UserIdentityLedger {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    current: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    history: Vec<UserNameChange>,
    #[serde(default)]
    forgotten: bool,
    #[serde(default)]
    revision: u64,
}

impl UserIdentityLedger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.current.is_none() && self.history.is_empty() && !self.forgotten
    }

    pub fn current(&self) -> Option<&str> {
        self.current.as_deref()
    }

    pub fn forgotten(&self) -> bool {
        self.forgotten
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn history(&self) -> &[UserNameChange] {
        &self.history
    }

    pub fn status(&self) -> IdentityStatus {
        if self.forgotten {
            return IdentityStatus::Forgotten;
        }
        let Some(current) = self.current.as_deref() else {
            return IdentityStatus::Unknown;
        };
        match self.history.last() {
            Some(change)
                if change.current == current
                    && change.previous.as_deref().is_some_and(|old| old != current) =>
            {
                IdentityStatus::Corrected
            }
            _ => IdentityStatus::Known,
        }
    }

    /// 记住一个显式自述的名字。相同名字重复自述不产生新历史。
    pub fn remember(&mut self, name: &str, turn: usize) -> RememberOutcome {
        let name = name.trim();
        if name.is_empty() {
            return RememberOutcome::Unchanged;
        }
        if self.forgotten {
            self.forgotten = false;
            self.current = None;
            self.history.clear();
        }
        if self.current.as_deref() == Some(name) {
            return RememberOutcome::Unchanged;
        }
        let previous = self.current.replace(name.to_string());
        self.history.push(UserNameChange {
            previous: previous.clone(),
            current: name.to_string(),
            turn,
        });
        if self.history.len() > 32 {
            self.history.remove(0);
        }
        self.revision = self.revision.saturating_add(1);
        match previous {
            Some(previous) => RememberOutcome::Corrected { previous },
            None => RememberOutcome::Remembered,
        }
    }

    /// 忘记名字：清空当前值与全部历史值（只保留遗忘墓碑）。
    pub fn forget(&mut self) -> bool {
        let had = self.current.is_some() || !self.history.is_empty() || self.forgotten;
        self.current = None;
        self.history.clear();
        self.forgotten = true;
        self.revision = self.revision.saturating_add(1);
        had && self.current.is_none()
    }

    fn last_turn(&self) -> Option<usize> {
        self.history.last().map(|change| change.turn)
    }
}

const SELF_PREFIXES: [&str; 14] = [
    "你好，",
    "你好,",
    "您好，",
    "您好,",
    "请记住，",
    "请记住,",
    "记住，",
    "记住,",
    "不对，",
    "不对,",
    "不是，",
    "不是,",
    "更正：",
    "更正:",
];

const SELF_MARKERS: [&str; 11] = [
    "我的名字改成",
    "我的名字叫",
    "我的名字是",
    "我改名叫",
    "改名为",
    "称呼我为",
    "你可以叫我",
    "请叫我",
    "你叫我",
    "叫我",
    "我叫",
];

const USER_QUESTION_MARKERS: [&str; 10] = [
    "我是谁",
    "我叫什么",
    "我的名字是什么",
    "我的名字叫什么",
    "你记得我是谁",
    "你记得我叫什么",
    "你还记得我",
    "知道我是谁",
    "我姓什么",
    "我叫什么名字",
];

const SYSTEM_QUESTION_MARKERS: [&str; 18] = [
    "你是谁",
    "你叫什么",
    "你的名字",
    "你是什么",
    "你是做什么",
    "你是干嘛",
    "你能做什么",
    "你会做什么",
    "你有哪些能力",
    "你会什么",
    "你有意识",
    "你有自我意识",
    "你是人吗",
    "你是机器人吗",
    "介绍一下你自己",
    "介绍你自己",
    "自我介绍",
    "你了解你自己",
];

const FORGET_MARKERS: [&str; 8] = [
    "忘记我的名字",
    "忘掉我的名字",
    "忘了我的名字",
    "忘记我",
    "忘掉我",
    "忘了我",
    "不要记住我",
    "别记得我",
];

fn strip_sentence_end(text: &str) -> &str {
    text.trim()
        .trim_end_matches(['？', '?', '。', '！', '!', ' ', '～', '~'])
        .trim()
}

fn strip_self_prefix(mut body: &str) -> &str {
    loop {
        let mut stripped = false;
        for prefix in SELF_PREFIXES {
            if let Some(rest) = body.strip_prefix(prefix) {
                body = rest.trim_start();
                stripped = true;
                break;
            }
        }
        if !stripped {
            return body;
        }
    }
}

fn extract_self_name(text: &str) -> Option<String> {
    let body = strip_self_prefix(text);
    let mut found: Option<(usize, &str)> = None;
    for marker in SELF_MARKERS {
        if let Some(index) = body.rfind(marker) {
            match found {
                Some((best, _)) if best >= index => {}
                _ => found = Some((index, marker)),
            }
        }
    }
    let (index, marker) = found?;
    let rest = body[index + marker.len()..].trim();
    let cut = rest
        .find(|ch: char| {
            matches!(
                ch,
                '，' | ',' | '。' | '！' | '!' | '？' | '?' | '；' | ';' | '\n' | '\t'
            )
        })
        .map(|position| &rest[..position])
        .unwrap_or(rest);
    let name = cut
        .trim()
        .trim_matches(|ch| matches!(ch, '「' | '」' | '『' | '』' | '“' | '”' | '"' | '\''))
        .trim()
        .trim_end_matches(['吧', '呀', '啊', '哦', '呢'])
        .trim();
    if name.is_empty() || name.chars().count() > 24 {
        return None;
    }
    if name.chars().any(|ch| ch.is_whitespace()) && name.split_whitespace().count() != 2 {
        return None;
    }
    const REJECT: [&str; 10] = ["？", "?", "谁", "什么", "哪", "吗", "呢", "的", "是", "叫"];
    if REJECT.iter().any(|marker| name.contains(marker)) {
        return None;
    }
    if name.starts_with(['你', '他', '她', '它', '这', '那']) {
        return None;
    }
    Some(name.to_string())
}

fn is_entity_identity(text: &str) -> bool {
    text.ends_with("是谁")
        || text.ends_with("是什么")
        || text.ends_with("是哪位")
        || text.contains("什么是")
}

/// 确定性身份动作分类。
pub fn classify(input: &str) -> IdentityAct {
    let stripped = strip_sentence_end(input);
    if stripped.is_empty() {
        return IdentityAct::Other;
    }
    let lowered = stripped.to_lowercase();

    if FORGET_MARKERS
        .iter()
        .any(|marker| stripped.contains(marker))
    {
        return IdentityAct::ForgetUser;
    }
    if USER_QUESTION_MARKERS
        .iter()
        .any(|marker| stripped.contains(marker))
    {
        return IdentityAct::AskUser;
    }
    if let Some(name) = extract_self_name(stripped) {
        return IdentityAct::Introduce { name };
    }
    if SYSTEM_QUESTION_MARKERS
        .iter()
        .any(|marker| stripped.contains(marker) || lowered.contains(&marker.to_lowercase()))
    {
        return IdentityAct::AskSystem;
    }
    if is_entity_identity(stripped) {
        return IdentityAct::AskEntity;
    }
    IdentityAct::Other
}

/// 系统身份卡：来自固定的「宪法常量」，不查询也不编造外部知识。
pub fn system_card() -> IdentityCard {
    let claims = vec![
        IdentityClaim::new(
            "system:name",
            "系统名称",
            "KnowProlog",
            ClaimSource::Constitution,
            "系统宪法",
        ),
        IdentityClaim::new(
            "system:role",
            "系统角色",
            "符号主义中文知识图谱对话系统",
            ClaimSource::Constitution,
            "系统宪法",
        ),
        IdentityClaim::new(
            "system:abilities",
            "能力范围",
            "图谱查询、规则推理、多轮对话与确定性模板生成",
            ClaimSource::Constitution,
            "系统宪法",
        ),
        IdentityClaim::new(
            "system:limits",
            "能力边界",
            "不具备意识、情感或主观体验，也不声称拥有图谱之外的知识",
            ClaimSource::Constitution,
            "系统宪法",
        ),
        IdentityClaim::new(
            "system:privacy",
            "隐私约定",
            "只记住用户在本会话中明确自述的名字，并可随时忘记",
            ClaimSource::Constitution,
            "系统宪法",
        ),
    ];
    IdentityCard::new(IdentityScope::System, "KnowProlog", IdentityStatus::Known)
        .with_claims(claims)
}

/// 用户身份卡：只读会话台账；未知 / 已遗忘时不给任何身份 claim。
pub fn user_card(ledger: &UserIdentityLedger) -> IdentityCard {
    let status = ledger.status();
    let subject = "用户";
    if status == IdentityStatus::Unknown || status == IdentityStatus::Forgotten {
        return IdentityCard::new(IdentityScope::User, subject, status);
    }
    let Some(name) = ledger.current() else {
        return IdentityCard::new(IdentityScope::User, subject, IdentityStatus::Unknown);
    };
    let turn = ledger.last_turn().unwrap_or(0);
    let evidence = if turn > 0 {
        format!("第 {} 轮用户明确自述，仅本会话", turn)
    } else {
        "用户明确自述，仅本会话".to_string()
    };
    let claim = IdentityClaim::new(
        "user:name",
        "会话内称呼",
        name,
        ClaimSource::UserStatement,
        evidence,
    );
    IdentityCard::new(IdentityScope::User, subject, status).with_claims(vec![claim])
}

/// 实体身份卡：类型 / 别名 / 关键事实全部来自现有图谱与身份注册表。
pub fn entity_card(knowledge: &dyn DialogueKnowledge, entity: &str) -> IdentityCard {
    let entity = entity.trim();
    if entity.is_empty() {
        return IdentityCard::new(IdentityScope::Entity, entity, IdentityStatus::Unknown);
    }

    let aliases = knowledge.reference_view().map(|view| {
        let registry = view.linker.identity_registry();
        match registry.resolve(entity) {
            crate::identity::Resolution::Ambiguous(ids) => Err(ids),
            _ => Ok(registry.surfaces_of(entity)),
        }
    });
    let aliases = match aliases {
        Some(Err(candidates)) => {
            let mut card =
                IdentityCard::new(IdentityScope::Entity, entity, IdentityStatus::Ambiguous);
            for candidate in candidates {
                card.claims.push(IdentityClaim::new(
                    "entity:ambiguous",
                    "候选身份",
                    candidate,
                    ClaimSource::Registry,
                    "身份注册表歧义",
                ));
            }
            return card;
        }
        Some(Ok(aliases)) => aliases,
        None => Vec::new(),
    };

    if !knowledge.known_entity(entity) {
        return IdentityCard::new(IdentityScope::Entity, entity, IdentityStatus::Unknown);
    }

    let mut claims = vec![IdentityClaim::new(
        "entity:identity",
        "规范身份",
        entity,
        ClaimSource::Graph,
        "图谱规范实体",
    )];
    if let Some(etype) = knowledge.entity_type(entity) {
        claims.push(IdentityClaim::new(
            "entity:type",
            "类型",
            etype,
            ClaimSource::Graph,
            "图谱类型断言",
        ));
    }
    for alias in aliases.into_iter().filter(|alias| alias != entity).take(8) {
        claims.push(IdentityClaim::new(
            "entity:alias",
            "别名",
            alias,
            ClaimSource::Registry,
            "身份注册表",
        ));
    }
    for (attribute, value) in knowledge.entity_attributes(entity).into_iter().take(6) {
        let label = format!("事实·{}", knowledge.predicate_label(&attribute));
        claims.push(IdentityClaim::new(
            format!("entity:attribute:{}", attribute),
            label,
            format!("{}={}", attribute, value),
            ClaimSource::Graph,
            format!("图谱属性 {}", attribute),
        ));
    }
    for (predicate, object) in knowledge.entity_relations(entity).into_iter().take(6) {
        let label = format!("关系·{}", knowledge.predicate_label(&predicate));
        claims.push(IdentityClaim::new(
            format!("entity:relation:{}", predicate),
            label,
            format!("{}={}", predicate, object),
            ClaimSource::Graph,
            format!("图谱关系 {}", predicate),
        ));
    }
    IdentityCard::new(IdentityScope::Entity, entity, IdentityStatus::Known).with_claims(claims)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialogue_manager::DialogueKnowledge;

    #[derive(Default)]
    struct FakeKnowledge {
        entities: Vec<String>,
        types: Vec<(String, String)>,
        attributes: Vec<(String, String, String)>,
        relations: Vec<(String, String, String)>,
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
            self.entities.iter().any(|known| known == entity)
        }

        fn answer(&self, _question: &str) -> Option<String> {
            None
        }

        fn known_attributes(&self, _entity: &str) -> Vec<String> {
            Vec::new()
        }

        fn entity_type(&self, entity: &str) -> Option<String> {
            self.types
                .iter()
                .find(|(name, _)| name == entity)
                .map(|(_, etype)| etype.clone())
        }

        fn entity_attributes(&self, entity: &str) -> Vec<(String, String)> {
            self.attributes
                .iter()
                .filter(|(name, _, _)| name == entity)
                .map(|(_, key, value)| (key.clone(), value.clone()))
                .collect()
        }

        fn entity_relations(&self, entity: &str) -> Vec<(String, String)> {
            self.relations
                .iter()
                .filter(|(name, _, _)| name == entity)
                .map(|(_, key, value)| (key.clone(), value.clone()))
                .collect()
        }
    }

    #[test]
    fn classifies_three_identity_scopes_before_entity_questions() {
        assert_eq!(classify("你是谁？"), IdentityAct::AskSystem);
        assert_eq!(classify("你有意识吗"), IdentityAct::AskSystem);
        assert_eq!(classify("我是谁"), IdentityAct::AskUser);
        assert_eq!(classify("你记得我叫什么吗"), IdentityAct::AskUser);
        assert_eq!(classify("姚明是谁"), IdentityAct::AskEntity);
        assert_eq!(classify("什么是量子力学"), IdentityAct::AskEntity);
        assert_eq!(classify("姚明的身高是多少"), IdentityAct::Other);
    }

    #[test]
    fn extracts_explicit_self_introductions_only() {
        assert_eq!(
            classify("我叫张三。"),
            IdentityAct::Introduce {
                name: "张三".to_string()
            }
        );
        assert_eq!(
            classify("我的名字是李四"),
            IdentityAct::Introduce {
                name: "李四".to_string()
            }
        );
        assert_eq!(
            classify("请叫我小明吧"),
            IdentityAct::Introduce {
                name: "小明".to_string()
            }
        );
        assert_eq!(
            classify("不对，我叫王五"),
            IdentityAct::Introduce {
                name: "王五".to_string()
            }
        );
        assert_eq!(classify("我叫什么"), IdentityAct::AskUser);
        assert_eq!(classify("我叫你别走"), IdentityAct::Other);
    }

    #[test]
    fn user_ledger_tracks_correction_and_forgets_cleanly() {
        let mut ledger = UserIdentityLedger::new();
        assert_eq!(ledger.status(), IdentityStatus::Unknown);
        assert_eq!(ledger.remember("张三", 1), RememberOutcome::Remembered);
        assert_eq!(ledger.status(), IdentityStatus::Known);
        assert_eq!(
            ledger.remember("李四", 2),
            RememberOutcome::Corrected {
                previous: "张三".to_string()
            }
        );
        assert_eq!(ledger.status(), IdentityStatus::Corrected);
        assert_eq!(ledger.current(), Some("李四"));
        assert!(ledger.forget());
        assert_eq!(ledger.status(), IdentityStatus::Forgotten);
        assert_eq!(ledger.current(), None);
        assert!(ledger.history().is_empty(), "遗忘必须清空历史值");
        let card = user_card(&ledger);
        assert!(card.claims.is_empty());
        assert!(!card.render().contains("张三"));
        assert!(!card.render().contains("李四"));
    }

    #[test]
    fn system_card_is_constitutional_and_scoped() {
        let card = system_card();
        assert_eq!(card.scope, IdentityScope::System);
        assert_eq!(card.scope_leaks(), 0);
        assert!(card.render().contains("KnowProlog"));
        assert!(card.render().contains("不具备意识"));
        assert!(card.claims.iter().all(|claim| !claim.evidence.is_empty()));
    }

    #[test]
    fn entity_card_uses_graph_and_abstains_when_unknown() {
        let knowledge = FakeKnowledge {
            entities: vec!["姚明".to_string()],
            types: vec![("姚明".to_string(), "人物".to_string())],
            attributes: vec![("姚明".to_string(), "身高".to_string(), "226".to_string())],
            relations: vec![(
                "姚明".to_string(),
                "效力于".to_string(),
                "火箭队".to_string(),
            )],
        };
        let card = entity_card(&knowledge, "姚明");
        assert_eq!(card.status, IdentityStatus::Known);
        assert_eq!(card.scope_leaks(), 0);
        assert!(card
            .claim_signatures()
            .contains(&"entity:type=人物".to_string()));
        assert!(card
            .claim_signatures()
            .contains(&"entity:attribute:身高=身高=226".to_string()));
        let text = card.render();
        assert!(text.contains("【身份卡·实体】"));
        assert!(text.contains("依据：图谱属性 身高"));

        let unknown = entity_card(&knowledge, "诸葛亮");
        assert_eq!(unknown.status, IdentityStatus::Unknown);
        assert!(unknown.claims.is_empty());
        assert!(unknown.render().contains("不会为它编造身份卡"));
    }

    #[test]
    fn user_and_entity_cards_do_not_mix_scopes() {
        let mut ledger = UserIdentityLedger::new();
        ledger.remember("张三", 1);
        let knowledge = FakeKnowledge {
            entities: vec!["姚明".to_string()],
            types: vec![("姚明".to_string(), "人物".to_string())],
            attributes: Vec::new(),
            relations: Vec::new(),
        };
        let user = user_card(&ledger);
        let entity = entity_card(&knowledge, "姚明");
        assert_eq!(user.scope_leaks(), 0);
        assert_eq!(entity.scope_leaks(), 0);
        assert_eq!(user.status, IdentityStatus::Known);
        assert!(!user.render().contains("姚明"));
        assert!(!entity.render().contains("张三"));
    }
}

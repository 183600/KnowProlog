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

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::binding_frames::{FrameMemory, FrameProgram};
use crate::bindings::BindingSet;
use crate::derivation::Derivation;
use crate::epistemic::{EpistemicStatus, LedgerLoad};
use crate::logic_program::{LAtom, LTerm, LogicProgram};
use crate::prompt_spec::{self, PromptMode, PromptStats, RenderOutcome, PROMPT_SPEC_VERSION};
use crate::quality::{ClaimWarrant, QualityIndex};
use crate::query_board::QueryBoard;
use crate::recovery::{self, PendingKind, RecoveryMode, RecoveryRequest, RecoveryState};
use crate::recovery::{RecoveryNomination, ReframedReading, VerificationOutcome};
use crate::symbolic_reasoner::SymbolicReasoner;
use crate::trace_graph::{NodeId, QudKind, ReferentSource, TraceMode, TraceStore, TurnTrace};
use crate::turn_runtime::{OpenTurn, OpenTurnBook, TurnFault};

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
    MetaReview,
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
            MoveKind::MetaReview => "meta_review",
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
    /// 本轮回答依据的质量担保（来源/冲突/缺口/类型）；快照未包含该字段时按 None 恢复。
    #[serde(default)]
    pub warrant: Option<ClaimWarrant>,
    /// 回答时所依据的知识版本（多轮复核用）；快照未包含该字段时按 None 恢复。
    #[serde(default)]
    pub kb_revision: Option<u64>,
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
    /// 失败回执台账与挂账问题（认识论状态也是可观察状态的一部分）。
    #[serde(default)]
    pub recovery: RecoveryState,
    /// 交互式查询板：当前查询程序的帧栈（工作记忆）。
    #[serde(default)]
    pub board: QueryBoard,
    /// 分层追踪图（Off/空态时为 None，快照字节与引入追踪前逐字节一致）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trace: Option<TraceStore>,
    /// 未完成轮次（取消 / 超时 / 故障）：空态省略，旧快照字节不变。
    #[serde(default, skip_serializing_if = "OpenTurnBook::is_empty")]
    pub open_turns: OpenTurnBook,
    /// 绑定帧记忆：空态省略，未启用绑定帧的会话字节不变。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frames: Option<FrameMemory>,
}

/// 管理器的生成结果：文本 + 命中的策略 + 语用类型。
#[derive(Clone, Debug)]
pub struct DialogueReply {
    pub text: String,
    pub plan: Vec<String>,
    pub kind: MoveKind,
}

/// 最近一次恢复动作的对外视图（不持久化；派生于台账与挂账）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecoveryNotice {
    /// 本轮对外的认识论定性。
    pub status: crate::epistemic::EpistemicStatus,
    /// 恢复阶梯档位（`L0`–`L4`）。
    pub level: String,
    /// 关联回执 id（如有）。
    pub receipt_id: Option<String>,
    /// 证据链（推测/冲突时非空；逐字可回溯）。
    pub evidence: Vec<String>,
    /// 挂账追问（如有）。
    pub question: Option<String>,
}

/// 管理器向知识源提出的接口（由 `SymbolicReasoner` 或测试替身实现）。
pub trait DialogueKnowledge {
    /// 文本中出现的已知实体（规范名，按长度降序更佳）。
    fn entities_in(&self, text: &str) -> Vec<String>;
    /// 实体是否在本体/图谱中登记。
    fn known_entity(&self, entity: &str) -> bool;
    /// 对规范化问句作答。
    fn answer(&self, question: &str) -> Option<String>;
    /// 为问句重建一条有界推导；`expected_answer` 用于把证明锚定到上一轮
    /// 的实际回答，匹配不上时返回 None（宁可不解释，也不挂错证明）。
    fn derivation(&self, question: &str, expected_answer: Option<&str>) -> Option<Derivation> {
        let _ = (question, expected_answer);
        None
    }
    /// 实体已知的属性键/关系谓词（供追问建议）。
    fn known_attributes(&self, entity: &str) -> Vec<String>;
    /// 字面量属性的取值（结构化检索，缺省为空）。
    fn attribute_values(&self, entity: &str, attribute: &str) -> Vec<String> {
        let _ = (entity, attribute);
        Vec::new()
    }

    /// 图谱中的全部规范实体（查询板扫描用；缺省不可枚举）。
    fn all_entities(&self) -> Vec<String> {
        Vec::new()
    }

    /// 图谱中的全部实体类型（查询板按类型筛选用；缺省不可枚举）。
    fn all_types(&self) -> Vec<String> {
        Vec::new()
    }

    /// 某类型的全部成员实体（缺省不可枚举）。
    fn entities_of_type(&self, entity_type: &str) -> Vec<String> {
        let _ = entity_type;
        Vec::new()
    }

    /// 实体的规范类型（缺省未知）。
    fn entity_type(&self, entity: &str) -> Option<String> {
        let _ = entity;
        None
    }

    /// 实体的字面量属性对（缺省级联属性键与取值）。
    fn entity_attributes(&self, entity: &str) -> Vec<(String, String)> {
        let mut pairs = Vec::new();
        for key in self.known_attributes(entity) {
            for value in self.attribute_values(entity, &key) {
                pairs.push((key.clone(), value));
            }
        }
        pairs.sort();
        pairs.dedup();
        pairs
    }

    /// 实体的出向关系对（缺省为空）。
    fn entity_relations(&self, entity: &str) -> Vec<(String, String)> {
        let _ = entity;
        Vec::new()
    }

    /// 谓词词面归一（缺省恒等）。
    fn canonical_predicate(&self, surface: &str) -> Option<String> {
        let _ = surface;
        None
    }

    /// 谓词的人类可读词面（缺省原样返回）。
    fn predicate_label(&self, predicate: &str) -> String {
        predicate.to_string()
    }

    /// 知识版本（查询板据此标注过期帧；缺省 0）。
    fn revision(&self) -> u64 {
        0
    }
    /// 实体的生成式简介（画像段落），无法生成时缺省为 None。
    fn describe(&self, entity: &str) -> Option<String> {
        let _ = entity;
        None
    }
    /// 审计 / 评测用：回答中引用的词条是否可在知识源中回溯到出处。
    /// 缺省不可判定（false），由具体知识源按需实现。
    fn traceable_token(&self, token: &str) -> bool {
        let _ = token;
        false
    }

    /// 恢复通道 L1：换帧重解（缺省无替代问法）。
    fn recovery_reframe(&self, entity: &str, attribute: &str) -> Vec<ReframedReading> {
        let _ = (entity, attribute);
        Vec::new()
    }

    /// 恢复通道 L2：联想通道提名（缺省无提名）。
    fn recovery_nominations(&self, entity: &str, attribute: &str) -> Vec<RecoveryNomination> {
        let _ = (entity, attribute);
        Vec::new()
    }

    /// 恢复通道 L2：符号通道验证一条提名（缺省无结论）。
    fn recovery_verify(&self, entity: &str, attribute: &str, value: &str) -> VerificationOutcome {
        let _ = (entity, attribute, value);
        VerificationOutcome::NoVerdict
    }

    /// 恢复通道 L3：实体可追问的相关线索（缺省无）。
    fn recovery_clues(&self, entity: &str) -> Vec<String> {
        let _ = entity;
        Vec::new()
    }

    /// 引擎层故障（Prolog / 存储 / 策略求值）：非空时走 EngineFault 路径，
    /// 绝不与「知识里没有」合并。缺省无故障。
    fn engine_fault(&self) -> Option<String> {
        None
    }
}

/// 把 [`DialogueKnowledge`] 适配为恢复引擎所需的 [`recovery::RecoveryKnowledge`]。
struct KnowledgeRecovery<'a> {
    knowledge: &'a dyn DialogueKnowledge,
}

impl recovery::RecoveryKnowledge for KnowledgeRecovery<'_> {
    fn reframe(&self, entity: &str, attribute: &str) -> Vec<ReframedReading> {
        self.knowledge.recovery_reframe(entity, attribute)
    }

    fn nominations(&self, entity: &str, attribute: &str) -> Vec<RecoveryNomination> {
        self.knowledge.recovery_nominations(entity, attribute)
    }

    fn verify(&self, entity: &str, attribute: &str, value: &str) -> VerificationOutcome {
        self.knowledge.recovery_verify(entity, attribute, value)
    }

    fn related(&self, entity: &str) -> Vec<String> {
        self.knowledge.recovery_clues(entity)
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

    fn derivation(&self, question: &str, expected_answer: Option<&str>) -> Option<Derivation> {
        SymbolicReasoner::answer_derivation(self, question, expected_answer)
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

    fn all_entities(&self) -> Vec<String> {
        self.generator().all_entities()
    }

    fn all_types(&self) -> Vec<String> {
        self.generator().all_types()
    }

    fn entities_of_type(&self, entity_type: &str) -> Vec<String> {
        self.generator().entities_of_type(entity_type)
    }

    fn entity_type(&self, entity: &str) -> Option<String> {
        self.generator().entity_type_of(entity).map(str::to_string)
    }

    fn entity_attributes(&self, entity: &str) -> Vec<(String, String)> {
        self.generator().attribute_pairs(entity)
    }

    fn entity_relations(&self, entity: &str) -> Vec<(String, String)> {
        self.generator().relation_pairs(entity)
    }

    fn canonical_predicate(&self, surface: &str) -> Option<String> {
        self.generator().canonical_predicate(surface).or_else(|| {
            crate::symbolic_generator::canonical_predicate_key(surface).map(str::to_string)
        })
    }

    fn predicate_label(&self, predicate: &str) -> String {
        crate::symbolic_generator::localized_predicate_key(predicate)
            .map(str::to_string)
            .unwrap_or_else(|| predicate.to_string())
    }

    fn revision(&self) -> u64 {
        SymbolicReasoner::revision(self)
    }

    fn describe(&self, entity: &str) -> Option<String> {
        self.generator().profile(entity)
    }

    /// 词条可回溯：登记实体 / 谓词词面 / 任一实体的属性键或取值。
    fn traceable_token(&self, token: &str) -> bool {
        self.generator().is_known_entity(token)
            || self.generator().predicate_surfaces().contains_key(token)
            || self.attribute_token_known(token)
    }

    /// 联想提名来自链接预测（符号统计，非神经网络）。
    fn recovery_nominations(&self, entity: &str, attribute: &str) -> Vec<RecoveryNomination> {
        SymbolicReasoner::recovery_nominations(self, entity, attribute)
    }

    fn recovery_verify(&self, entity: &str, attribute: &str, value: &str) -> VerificationOutcome {
        SymbolicReasoner::recovery_verify(self, entity, attribute, value)
    }

    fn recovery_clues(&self, entity: &str) -> Vec<String> {
        SymbolicReasoner::recovery_clues(self, entity)
    }

    fn recovery_reframe(&self, entity: &str, attribute: &str) -> Vec<ReframedReading> {
        SymbolicReasoner::recovery_reframe(self, entity, attribute)
    }

    fn engine_fault(&self) -> Option<String> {
        SymbolicReasoner::engine_fault(self)
    }
}

/// 对话策略程序：显式声明「在什么信息状态下选择什么话语行为」。
/// 每轮把状态固化为 ABox，求值后得到 `respond(...)` 集合。
pub const POLICY_SOURCE: &str = r#"
# —— 对话策略本体（Horn 规则 + 分层否定）——
# 元话语：回顾、复核与解释依据
respond(recap) :- kind(T, meta_recap).
respond(review_answer) :- kind(T, meta_review).
respond(explain_why) :- kind(T, meta_why), has_evidence(T).
respond(explain_why) :- kind(T, meta_why), why_not(T).
respond(explain_why) :- kind(T, meta_why), step_request(T).
respond(why_no_evidence) :- kind(T, meta_why), not has_evidence(T), not why_not(T), not step_request(T).
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
const REPLY_PRIORITY: [&str; 14] = [
    "recap",
    "review_answer",
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

const REVIEW_PATTERNS: [&str; 10] = [
    "你确定吗",
    "你确定么",
    "确定吗",
    "刚才说的还算吗",
    "刚才说的还算数吗",
    "现在还成立吗",
    "还成立吗",
    "复核一下",
    "复核",
    "有没有变化",
];

const WHY_NOT_PATTERNS: [&str; 8] = [
    "为什么没有",
    "为什么查不到",
    "为什么不知道",
    "为什么找不到",
    "怎么没有记录",
    "为什么推不出",
    "为什么无法回答",
    "为什么答不上",
];

const WHY_PATTERNS: [&str; 13] = [
    "你怎么知道",
    "你凭什么",
    "为什么这么说",
    "为什么这样说",
    "为什么这样回答",
    "为什么这么回答",
    "依据是什么",
    "有什么依据",
    "怎么得出结论",
    "怎么推出来的",
    "怎么推导",
    "推导过程",
    "证明过程",
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
    "你好",
    "您好",
    "你好呀",
    "嗨",
    "哈喽",
    "hi",
    "hello",
    "早上好",
    "中午好",
    "下午好",
    "晚上好",
    "在吗",
];

/// 本体化对话管理器。
pub struct DialogueManager {
    turns: Vec<DialogueTurn>,
    focus: Vec<String>,
    externals: Vec<String>,
    policy: LogicProgram,
    last_reply: Vec<String>,
    /// 提示模板运行模式（默认 Off）。
    prompt_mode: PromptMode,
    /// 提示模板运行统计（诊断计数，不参与对话快照的语义状态）。
    prompt_stats: PromptStats,
    /// 失败恢复模式（默认 Off：原样输出；Observe 只记台账；On 完整阶梯）。
    recovery_mode: RecoveryMode,
    /// 失败台账 + 待定挂账（随快照持久化）。
    recovery: RecoveryState,
    /// 最近一次恢复动作（运行时视图，不进快照）。
    last_recovery: Option<RecoveryNotice>,
    /// 质量护照索引（只读派生数据；由宿主按知识版本注入，不进快照）。
    /// 用 `Arc` 共享：多会话（HTTP actor）注入同一份索引时不复制整张查询表。
    quality: Option<Arc<QualityIndex>>,
    /// 交互式查询板：记住当前查询程序、结果行与证据（随快照持久化）。
    board: QueryBoard,
    /// 分层追踪模式（默认 Off：不记录、不介入；Observe 只记决策；On 参与理解与生成）。
    trace_mode: TraceMode,
    /// 分层追踪图：问据栈、指代轨迹与逐轮决策节点（随快照持久化）。
    trace: TraceStore,
    /// 未完成轮次台账：让「上一问停在推理阶段」成为可继续的一等记忆。
    open_turns: OpenTurnBook,
    /// 配置编译的绑定帧程序（None：完全不接管）。
    frame_program: Option<Arc<FrameProgram>>,
    /// 绑定帧的会话记忆（槽位环境 / 上一帧 / 待澄清帧）。
    frame_memory: FrameMemory,
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
            prompt_mode: PromptMode::Off,
            prompt_stats: PromptStats::default(),
            recovery_mode: RecoveryMode::Off,
            recovery: RecoveryState::new(),
            last_recovery: None,
            quality: None,
            board: QueryBoard::new(),
            trace_mode: TraceMode::Off,
            trace: TraceStore::new(),
            open_turns: OpenTurnBook::new(),
            frame_program: None,
            frame_memory: FrameMemory::new(),
        }
    }

    pub fn policy_source() -> &'static str {
        POLICY_SOURCE
    }

    /// 设置模板运行模式（Off / Parity / On），支持链式调用。
    pub fn set_prompt_mode(&mut self, mode: PromptMode) -> &mut Self {
        self.prompt_mode = mode;
        self
    }

    pub fn prompt_mode(&self) -> PromptMode {
        self.prompt_mode
    }

    pub fn prompt_stats(&self) -> &PromptStats {
        &self.prompt_stats
    }

    pub fn reset_prompt_stats(&mut self) {
        self.prompt_stats = PromptStats::default();
    }

    /// 双跑差异计数（Parity 模式下 spec 渲染与组合生成器文本不一致的次数）。
    pub fn prompt_parity_mismatches(&self) -> usize {
        self.prompt_stats.parity_mismatches
    }

    /// 可读的提示资产状态行。
    pub fn prompt_report(&self) -> String {
        format!(
            "提示资产 v{}：{}",
            PROMPT_SPEC_VERSION,
            self.prompt_stats.render_text()
        )
    }

    /// 设置失败恢复模式（Off / Observe / On），支持链式调用。
    pub fn set_recovery_mode(&mut self, mode: RecoveryMode) -> &mut Self {
        self.recovery_mode = mode;
        self
    }

    pub fn recovery_mode(&self) -> RecoveryMode {
        self.recovery_mode
    }

    /// 设置分层追踪模式（Off / Observe / On），支持链式调用。
    pub fn set_trace_mode(&mut self, mode: TraceMode) -> &mut Self {
        self.trace_mode = mode;
        self
    }

    pub fn trace_mode(&self) -> TraceMode {
        self.trace_mode
    }

    /// 分层追踪图（只读）：问据栈、指代轨迹与逐轮决策节点。
    pub fn trace_store(&self) -> &TraceStore {
        &self.trace
    }

    /// 分层追踪图（可变引用；宿主用于编年史回放重建）。
    pub fn trace_store_mut(&mut self) -> &mut TraceStore {
        &mut self.trace
    }

    /// 可读的追踪报告（模式、问据栈、指代轨迹与逐轮决策）。
    pub fn trace_report(&self) -> String {
        self.trace.render_report(self.trace_mode)
    }

    /// 「为什么」查询：沿追踪图 inputs 边回溯某节点的决策链。
    pub fn trace_why(&self, node_id: u64) -> Option<String> {
        self.trace.render_why(NodeId(node_id))
    }

    /// 最近一轮的完整轨迹 JSON（编年史归档用；无记录时为 None）。
    pub fn last_turn_trace_json(&self) -> Option<serde_json::Value> {
        self.trace.last_turn_json()
    }

    /// 从编年史回放一轮轨迹（重启后重建追踪工作记忆）。
    pub fn replay_turn_trace(&mut self, trace: TurnTrace) {
        self.trace.push_turn_trace(trace);
    }

    /// 注入质量索引（只读；`None` 时完全走常规回答路径）。
    /// 宿主应在知识版本推进后重建索引，使「带出处/披露冲突/追问缺口」生效。
    pub fn set_quality_index(&mut self, index: Option<QualityIndex>) -> &mut Self {
        self.quality = index.map(Arc::new);
        self
    }

    /// 注入共享质量索引（供多会话服务复用同一份只读派生数据）。
    pub fn set_quality_index_shared(&mut self, index: Option<Arc<QualityIndex>>) -> &mut Self {
        self.quality = index;
        self
    }

    pub fn quality_index(&self) -> Option<&QualityIndex> {
        self.quality.as_deref()
    }

    /// 注入配置编译的绑定帧程序；None 或空程序时完全走常规管线。
    pub fn set_frame_program(&mut self, program: Option<Arc<FrameProgram>>) -> &mut Self {
        self.frame_program = program;
        self
    }

    pub fn frame_program(&self) -> Option<&FrameProgram> {
        self.frame_program.as_deref()
    }

    pub fn frame_memory(&self) -> &FrameMemory {
        &self.frame_memory
    }

    pub fn frame_report(&self) -> String {
        self.frame_memory.report()
    }

    /// 当前索引对应的知识版本（未注入时为 None）。
    pub fn quality_revision(&self) -> Option<u64> {
        self.quality.as_ref().map(|index| index.revision)
    }

    /// 失败台账（只读），供 CLI 报告与评测。
    pub fn failure_ledger(&self) -> &crate::epistemic::FailureLedger {
        &self.recovery.ledger
    }

    /// 恢复状态（只读）。
    pub fn recovery_state(&self) -> &RecoveryState {
        &self.recovery
    }

    /// 最近一次恢复动作的对外视图（服务层 `epistemic` 字段用；Off 模式下为 None）。
    pub fn last_recovery(&self) -> Option<&RecoveryNotice> {
        self.last_recovery.as_ref()
    }

    /// 清除最近恢复视图（服务层每轮开始时调用，防止跨轮陈旧泄漏）。
    pub fn clear_last_recovery(&mut self) {
        self.last_recovery = None;
    }

    /// 把一次让位（defer）记为 Unknown 失败回执：知识管线未能接管输入。
    ///
    /// Off 模式零副作用；返回是否产生了回执。让位和确答一样，
    /// 有状态、有 id、可被自省检索。
    pub fn note_deferred(&mut self, input: &str, knowledge: &dyn DialogueKnowledge) -> bool {
        if self.recovery_mode.eq(&RecoveryMode::Off) {
            return false;
        }
        let detail = excerpt(input.trim(), 40);
        if detail.is_empty() {
            return false;
        }
        self.recovery.begin_turn();
        let request = RecoveryRequest {
            status: EpistemicStatus::Unknown,
            entity: None,
            attribute: None,
            detail: &detail,
        };
        let attempt = recovery::attempt(
            &mut self.recovery,
            self.recovery_mode,
            &request,
            &KnowledgeRecovery { knowledge },
        );
        if attempt.receipt_id.is_none() {
            return false;
        }
        self.note_recovery(&attempt);
        true
    }

    /// 可读的失败恢复状态行（无回执时为确定性空态描述）。
    pub fn recovery_report(&self) -> String {
        format!(
            "失败恢复：模式 {}；{}；挂账 {} 个待定问题。\n{}",
            self.recovery_mode.label(),
            format!(
                "回执 {} 条（未结 {}）",
                self.recovery.ledger.len(),
                self.recovery.ledger.open_count()
            ),
            self.recovery.pending.len(),
            self.recovery.ledger.render_text()
        )
    }

    /// 把失败台账写入路径（供跨会话持久化，`--recovery-ledger` 使用）。
    pub fn save_recovery_ledger(&self, path: &str) -> std::io::Result<()> {
        self.recovery.ledger.save(path)
    }

    /// 从路径恢复失败台账（缺失/损坏时不改动现有台账，安全降级）。
    pub fn load_recovery_ledger(&mut self, path: &str) -> LedgerLoad {
        let (load, ledger) = crate::epistemic::FailureLedger::load_full(path);
        if let Some(ledger) = ledger {
            self.recovery.ledger = ledger;
        }
        load
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

    /// 插件配方回答后标注下一轮焦点。正向问句把答案实体置顶，
    /// 逆向问句仍保留被查询的主语；只移动焦点，不新增对话轮。
    pub fn note_plugin_focus(&mut self, entity: &str) {
        self.push_focus(entity.trim());
    }

    pub fn focus_stack(&self) -> &[String] {
        &self.focus
    }

    pub fn last_reply_plan(&self) -> &[String] {
        &self.last_reply
    }

    pub fn board(&self) -> &QueryBoard {
        &self.board
    }

    pub fn board_summary(&self) -> String {
        self.board.summary()
    }

    /// 记录一次用户输入（含被其他管线处理的命令），供回顾时统计。
    ///
    /// 同时推进未完成轮次的逻辑时钟：TTL 按轮次而不是挂钟计算。
    pub fn observe_input(&mut self, input: &str) {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return;
        }
        self.begin_turn();
        self.externals.push(trimmed.to_string());
        if self.externals.len() > 24 {
            self.externals.remove(0);
        }
    }

    /// 记录一次由插件配方给出的回答：不经过策略求值，但纳入同一对话本体，
    /// 让焦点栈、回顾、续聊与「为什么这么答」都能看到外部专家的贡献。
    pub fn note_plugin_turn(
        &mut self,
        input: &str,
        kind: MoveKind,
        entity: Option<&str>,
        attribute: Option<&str>,
        answer: &str,
        plan: Vec<String>,
        provenance: Vec<String>,
    ) {
        self.turns.push(DialogueTurn {
            index: self.turns.len() + 1,
            user: input.trim().to_string(),
            kind: kind.clone(),
            entity: entity.map(str::to_string),
            attribute: attribute.map(str::to_string),
            answer_excerpt: Some(excerpt(answer, 40)),
            provenance,
            plan: plan.clone(),
            warrant: None,
            kb_revision: self.quality.as_ref().map(|index| index.revision),
        });
        if let Some(entity) = entity {
            self.push_focus(entity);
        }
        self.last_reply = plan;
        if self.externals.last().map(String::as_str) == Some(input.trim()) {
            self.externals.pop();
        }
        // On/Observe 模式：插件代答也纳入分层追踪（记 Plugin 指代来源）。
        if !self.trace_mode.is_off() {
            let turn_no = self.trace.turn_count() + 1;
            let surface = self.trace.record_surface(turn_no, input);
            let kind_label = format!("{:?}", kind);
            let interpret = self.trace.record_interpret(
                turn_no,
                surface,
                &kind_label,
                entity,
                attribute,
                None,
                false,
            );
            if let Some(entity) = entity {
                self.trace.record_mention(
                    turn_no,
                    interpret,
                    entity,
                    entity,
                    ReferentSource::Plugin,
                    "插件配方代答",
                );
            }
            self.trace.record_reply(
                turn_no,
                entity,
                attribute,
                vec![interpret],
                &[],
                Some(answer),
            );
            self.trace.resolve_open_qud();
        }
    }

    /// 推进一个逻辑轮次：未完成轮次按 TTL 过期。
    pub fn begin_turn(&mut self) -> usize {
        self.open_turns.begin_tick()
    }

    /// 未完成轮次台账（只读）。
    pub fn open_turns(&self) -> &OpenTurnBook {
        &self.open_turns
    }

    /// 最新的未完成轮次。
    pub fn latest_open_turn(&self) -> Option<&OpenTurn> {
        self.open_turns.latest()
    }

    /// 记录一次运行时失败：原始输入、阶段、故障与焦点都进入一等记忆。
    pub fn record_open_turn(
        &mut self,
        input: &str,
        turn: u64,
        fault: &TurnFault,
        focus: Option<&str>,
        attribute: Option<&str>,
    ) -> u64 {
        self.open_turns.record(
            input,
            turn,
            fault.phase,
            fault.kind,
            &fault.detail,
            focus,
            attribute,
        )
    }

    /// 重放失败后重建未完成轮次（attempts 递增）。
    pub fn record_open_turn_attempts(
        &mut self,
        input: &str,
        turn: u64,
        fault: &TurnFault,
        focus: Option<&str>,
        attribute: Option<&str>,
        attempts: u32,
    ) -> u64 {
        self.open_turns.record_attempts(
            input,
            turn,
            fault.phase,
            fault.kind,
            &fault.detail,
            focus,
            attribute,
            attempts,
        )
    }

    /// 取走最新未完成轮次（继续语义：先消费，再重放）。
    pub fn take_open_turn(&mut self) -> Option<OpenTurn> {
        self.open_turns.take_latest()
    }

    /// 放弃最新未完成轮次。
    pub fn dismiss_open_turn(&mut self) -> Option<OpenTurn> {
        self.open_turns.dismiss_latest()
    }

    pub fn clear_open_turns(&mut self) {
        self.open_turns.clear();
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
        let mut summary = format!(
            "对话本体：{} 轮对话实例；焦点栈 [{}]；最近策略 [{}]；外部指令 {} 条；策略程序 {} 条规则。",
            self.turns.len(),
            focus,
            last,
            self.externals.len(),
            self.policy.rule_count()
        );
        if !self.recovery_mode.eq(&RecoveryMode::Off)
            || !self.recovery.ledger.is_empty()
            || !self.recovery.pending.is_empty()
        {
            summary.push_str(&format!(
                " 失败恢复：模式 {}，回执 {} 条（未结 {}），挂账 {} 个。",
                self.recovery_mode.label(),
                self.recovery.ledger.len(),
                self.recovery.ledger.open_count(),
                self.recovery.pending.len()
            ));
        }
        if self.board.is_active() {
            summary.push_str(&format!(" {}", self.board.summary()));
        }
        if let Some(open) = self.latest_open_turn() {
            summary.push_str(&format!(
                " 未完成轮次：{} 个；最新停在「{}」（{}，第 {} 次）。",
                self.open_turns.len(),
                open.phase.label(),
                open.fault.label(),
                open.attempts
            ));
        }
        if !self.frame_memory.is_empty() {
            summary.push_str(&format!(" {}", self.frame_memory.report()));
        }
        summary
    }

    pub fn snapshot(&self) -> DialogueManagerSnapshot {
        DialogueManagerSnapshot {
            turns: self.turns.clone(),
            focus: self.focus.clone(),
            externals: self.externals.clone(),
            last_reply: self.last_reply.clone(),
            recovery: self.recovery.clone(),
            board: self.board.clone(),
            trace: if self.trace.is_empty() {
                None
            } else {
                Some(self.trace.clone())
            },
            open_turns: self.open_turns.clone(),
            frames: (!self.frame_memory.is_empty()).then(|| self.frame_memory.clone()),
        }
    }

    pub fn restore(&mut self, snapshot: &DialogueManagerSnapshot) {
        self.turns = snapshot.turns.clone();
        self.focus = snapshot.focus.clone();
        self.externals = snapshot.externals.clone();
        self.last_reply = snapshot.last_reply.clone();
        self.recovery = snapshot.recovery.clone();
        self.board = snapshot.board.clone();
        self.trace = snapshot.trace.clone().unwrap_or_default();
        self.open_turns = snapshot.open_turns.clone();
        self.frame_memory = snapshot.frames.clone().unwrap_or_default();
    }

    pub fn save_to_path(&self, path: &str) -> std::io::Result<()> {
        if let Some(parent) = std::path::Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        // 规范字节：同一快照永远同一字节串（不依赖任何内部容器迭代序）。
        let bytes = crate::canonical::canonical_bytes(&self.snapshot())
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::Other, error.to_string()))?;
        std::fs::write(path, bytes)
    }

    /// 带解码回执的恢复：失败原因是可回答的（io/parse/invariant），
    /// 不折叠为 `false`。
    pub fn restore_from_path_report(
        &mut self,
        path: &str,
    ) -> Result<crate::canonical::DecodeReceipt, crate::canonical::DecodeError> {
        use crate::canonical::{self, DecodeError};
        let bytes = std::fs::read(path).map_err(|error| DecodeError::Io(error.to_string()))?;
        let (snapshot, receipt) = canonical::decode_json::<DialogueManagerSnapshot>(
            &bytes,
            "dialogue_manager",
            1,
            canonical::DEFAULT_MAX_DECODE_BYTES,
        )?;
        self.restore(&snapshot);
        Ok(receipt)
    }

    pub fn restore_from_path(&mut self, path: &str) -> bool {
        self.restore_from_path_report(path).is_ok()
    }

    /// 纯对话行为（元话语 / 续聊 / 展开 / 寒暄）：可在管线早段拦截。
    ///
    /// 恢复模式开启时，先尝试结算挂账（「是/不是」确认推测、具体线索关闭澄清），
    /// 这保证「推测 → 用户裁定 → 回执关闭」的闭环。
    pub fn plan_pure_dialogue(
        &mut self,
        input: &str,
        knowledge: &dyn DialogueKnowledge,
    ) -> Option<DialogueReply> {
        if let Some(reply) = self.try_trace_correction(input, knowledge) {
            return Some(reply);
        }
        if let Some(reply) = self.plan_frame_reply(input, knowledge) {
            return Some(reply);
        }
        if !self.recovery_mode.eq(&RecoveryMode::Off) {
            if let Some(reply) = self.try_resolve_pending(input) {
                return Some(reply);
            }
        }
        if self.board.is_active() {
            if let Some(reply) = self.plan_board_reply(input, knowledge) {
                return Some(reply);
            }
        }
        let frame = self.analyze(input, knowledge);
        match frame.kind {
            MoveKind::MetaRecap
            | MoveKind::MetaReview
            | MoveKind::MetaWhy
            | MoveKind::Greet
            | MoveKind::Thanks
            | MoveKind::Farewell
            | MoveKind::ContinueRequest
            | MoveKind::ElaborateRequest => self.execute(frame, knowledge),
            _ => None,
        }
    }

    /// 挂账结算：是/不是裁定推测挂账；具体而短的陈述关闭澄清挂账。
    fn try_resolve_pending(&mut self, input: &str) -> Option<DialogueReply> {
        if self.recovery.pending.is_empty() {
            return None;
        }
        let stripped = input
            .trim()
            .trim_end_matches(['？', '?', '。', '！', '!', ' '])
            .trim();
        if stripped.is_empty() {
            return None;
        }
        const YES: [&str; 6] = ["是", "是的", "对", "对的", "没错", "嗯"];
        const NO: [&str; 5] = ["不是", "不对", "错了", "错", "不"];
        let pending = self.recovery.latest_pending().cloned()?;
        let (text, plan) = if YES.contains(&stripped) {
            (
                recovery::resolve_by_yes(&mut self.recovery, true)?,
                "recovery_confirm",
            )
        } else if NO.contains(&stripped) {
            (
                recovery::resolve_by_yes(&mut self.recovery, false)?,
                "recovery_refute",
            )
        } else {
            // 澄清回答：排除问句与命令式输入，只收短陈述。
            if pending.kind != PendingKind::Clarification || stripped.chars().count() > 24 {
                return None;
            }
            if [
                "？",
                "?",
                "导入",
                "导出",
                "帮助",
                "退出",
                "为什么",
                "你怎么",
                "什么",
                "谁",
                "哪",
                "多少",
                "几",
                "吗",
                "呢",
                "怎么",
                "吧",
                "哈哈",
            ]
            .iter()
            .any(|marker| input.contains(marker))
            {
                return None;
            }
            (
                recovery::resolve_clarification(&mut self.recovery, stripped)?,
                "recovery_acknowledge",
            )
        };
        let level = match pending.kind {
            PendingKind::Provisional => "L2",
            PendingKind::Clarification => "L3",
        };
        self.last_recovery = Some(RecoveryNotice {
            status: pending.status,
            level: level.to_string(),
            receipt_id: Some(pending.receipt_id.clone()),
            evidence: Vec::new(),
            question: Some(pending.question.clone()),
        });
        self.last_reply = vec![plan.to_string()];
        Some(DialogueReply {
            text,
            plan: vec![plan.to_string()],
            kind: MoveKind::Other,
        })
    }

    /// 绑定帧回答：模式 -> 槽位绑定 -> 结构化目标 -> 模板措辞。
    /// 任何一步不成立都返回 None，调用方继续常规管线（零状态变化）。
    fn plan_frame_reply(
        &mut self,
        input: &str,
        knowledge: &dyn DialogueKnowledge,
    ) -> Option<DialogueReply> {
        let program = self.frame_program.clone()?;
        if program.is_empty() || !program.is_enabled() {
            return None;
        }
        let turn = self.turns.len() as u64 + 1;
        let outcome = program.plan(&mut self.frame_memory, input, knowledge, turn)?;
        self.turns.push(DialogueTurn {
            index: self.turns.len() + 1,
            user: input.trim().to_string(),
            kind: outcome.kind.clone(),
            entity: outcome.entity.clone(),
            attribute: outcome.attribute.clone(),
            answer_excerpt: Some(excerpt(&outcome.text, 40)),
            provenance: outcome.provenance.clone(),
            plan: outcome.plan.clone(),
            warrant: None,
            kb_revision: Some(knowledge.revision()),
        });
        if let Some(entity) = outcome.entity.as_deref() {
            self.push_focus(entity);
        }
        self.last_reply = outcome.plan.clone();
        if self.externals.last().map(String::as_str) == Some(input.trim()) {
            self.externals.pop();
        }
        Some(DialogueReply {
            text: outcome.text,
            plan: outcome.plan,
            kind: outcome.kind,
        })
    }

    /// 知识问句的生成式回答：仅在实体已知时才接管（否则让位给常规管线）。
    pub fn plan_reply(
        &mut self,
        input: &str,
        knowledge: &dyn DialogueKnowledge,
    ) -> Option<DialogueReply> {
        if let Some(reply) = self.try_trace_correction(input, knowledge) {
            return Some(reply);
        }
        if let Some(reply) = self.plan_frame_reply(input, knowledge) {
            return Some(reply);
        }
        if let Some(reply) = self.plan_board_reply(input, knowledge) {
            return Some(reply);
        }
        let frame = self.analyze(input, knowledge);
        match frame.kind {
            MoveKind::AskAttribute | MoveKind::AskIdentity => self.execute(frame, knowledge),
            _ => None,
        }
    }

    pub fn plan_board_reply(
        &mut self,
        input: &str,
        knowledge: &dyn DialogueKnowledge,
    ) -> Option<DialogueReply> {
        let revision = knowledge.revision();
        let outcome = self.board.apply(input, knowledge, revision)?;
        self.turns.push(DialogueTurn {
            index: self.turns.len() + 1,
            user: input.trim().to_string(),
            kind: MoveKind::Other,
            entity: outcome.focus.clone(),
            attribute: None,
            answer_excerpt: Some(excerpt(&outcome.text, 40)),
            provenance: outcome.provenance,
            plan: outcome.plan.clone(),
            warrant: None,
            kb_revision: Some(revision),
        });
        if let Some(entity) = outcome.focus {
            self.push_focus(&entity);
        }
        self.last_reply = outcome.plan.clone();
        if self.externals.last().map(String::as_str) == Some(input.trim()) {
            self.externals.pop();
        }
        Some(DialogueReply {
            text: outcome.text,
            plan: outcome.plan,
            kind: MoveKind::Other,
        })
    }

    /// 本体化解读：先识别元话语与寒暄，再解析属性/身份问句与断言，
    /// 最后用焦点栈补全指代与省略。
    pub fn analyze(&self, input: &str, knowledge: &dyn DialogueKnowledge) -> UtteranceFrame {
        let raw = input.trim();
        let stripped = raw
            .trim_end_matches(['？', '?', '。', '！', '!', ' '])
            .trim();
        if RECAP_PATTERNS
            .iter()
            .any(|pattern| stripped.contains(pattern))
        {
            return UtteranceFrame::new(raw, MoveKind::MetaRecap);
        }
        if REVIEW_PATTERNS
            .iter()
            .any(|pattern| stripped.contains(pattern))
        {
            return UtteranceFrame::new(raw, MoveKind::MetaReview);
        }
        if let Some(step) = parse_step_request(stripped) {
            let mut frame = UtteranceFrame::new(raw, MoveKind::MetaWhy);
            frame.question = Some(format!("step:{}", step));
            return frame;
        }
        if WHY_NOT_PATTERNS
            .iter()
            .any(|pattern| stripped.contains(pattern))
        {
            let mut frame = UtteranceFrame::new(raw, MoveKind::MetaWhy);
            frame.question = Some("why_not".to_string());
            return frame;
        }
        if WHY_PATTERNS
            .iter()
            .any(|pattern| stripped.contains(pattern))
        {
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
            // On 模式谓词焦点继承：「他呢」在属性问据下读作「他（=<焦点实体>）的<属性>呢」，
            // 而不是退回身份问句——上一轮问身高，这一轮仍问身高。
            if let Some(remainder) = stripped.strip_suffix('呢') {
                let remainder = remainder.trim();
                if self.trace_mode.is_on() && !remainder.is_empty() && contains_pronoun(remainder) {
                    if let Some(attribute) = self.trace.predicate_focus() {
                        let entity = self
                            .trace
                            .entity_focus()
                            .map(str::to_string)
                            .or_else(|| focus.clone());
                        if let Some(entity) = entity {
                            return UtteranceFrame {
                                raw: raw.to_string(),
                                kind: MoveKind::AskAttribute,
                                entity: Some(entity.clone()),
                                attribute: Some(attribute.to_string()),
                                question: Some(format!("{}的{}是什么", entity, attribute)),
                                inherited: true,
                            };
                        }
                    }
                }
            }
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
                // On 模式谓词焦点继承：「那叶莉呢」在属性问据下读作「叶莉的<属性>呢」，
                // 而不是退回身份问句（「叶莉是什么」）。
                if self.trace_mode.is_on() {
                    if let Some(attribute) = self.trace.predicate_focus() {
                        return UtteranceFrame {
                            raw: raw.to_string(),
                            kind: MoveKind::AskAttribute,
                            entity: Some(entity.clone()),
                            attribute: Some(attribute.to_string()),
                            question: Some(format!("{}的{}是什么", entity, attribute)),
                            inherited: false,
                        };
                    }
                }
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

        // 是非问句：交给常规蕴含/溯因管线处理
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

        // 陈述句：交给常规语义蕴含管线
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
        self.last_recovery = None;
        let mut entity = frame.entity.clone();
        let pure_dialogue = matches!(
            frame.kind,
            MoveKind::Greet
                | MoveKind::Thanks
                | MoveKind::Farewell
                | MoveKind::MetaRecap
                | MoveKind::MetaReview
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
        // 实体存在但图谱不认识时让位给常规 NLU / 外部知识管线。
        let attribute_only = matches!(frame.kind, MoveKind::AskAttribute)
            && entity.is_none()
            && frame.attribute.is_some();

        // —— 失败恢复：引擎层故障优先于一切知识判定 ——
        // On 模式下故障必须显式说「我出错了」；Observe 只入台账；Off 不介入。
        let mut recovery_attempt: Option<recovery::RecoveryAttempt> = None;
        let ask_like = matches!(
            frame.kind,
            MoveKind::AskAttribute | MoveKind::AskIdentity | MoveKind::AskBinary
        );
        if !self.recovery_mode.eq(&RecoveryMode::Off) && ask_like {
            self.recovery.begin_turn();
            if let Some(fault) = knowledge.engine_fault() {
                let fault = fault.trim().to_string();
                let request = RecoveryRequest {
                    status: EpistemicStatus::EngineFault,
                    entity: entity.as_deref(),
                    attribute: frame.attribute.as_deref(),
                    detail: &fault,
                };
                let attempt = recovery::attempt(
                    &mut self.recovery,
                    self.recovery_mode,
                    &request,
                    &KnowledgeRecovery { knowledge },
                );
                self.note_recovery(&attempt);
                if self.recovery_mode == RecoveryMode::On {
                    let text = if attempt.text.is_empty() {
                        crate::epistemic::render_statement(
                            EpistemicStatus::EngineFault,
                            entity.as_deref(),
                            frame.attribute.as_deref(),
                            None,
                        )
                    } else {
                        attempt.text.clone()
                    };
                    let plans = vec!["engine_fault".to_string()];
                    self.turns.push(DialogueTurn {
                        index: self.turns.len() + 1,
                        user: raw_input(&frame),
                        kind: frame.kind.clone(),
                        entity: entity.clone(),
                        attribute: frame.attribute.clone(),
                        answer_excerpt: None,
                        provenance: vec![format!("引擎故障：{}", fault)],
                        plan: plans.clone(),
                        warrant: None,
                        kb_revision: self.quality.as_ref().map(|index| index.revision),
                    });
                    if let Some(entity) = &entity {
                        self.push_focus(entity);
                    }
                    self.last_reply = plans.clone();
                    if self.externals.last().map(String::as_str) == Some(frame.raw.trim()) {
                        self.externals.pop();
                    }
                    return Some(DialogueReply {
                        text,
                        plan: plans,
                        kind: frame.kind.clone(),
                    });
                }
                recovery_attempt = Some(attempt);
            }
        }

        if matches!(frame.kind, MoveKind::AskAttribute | MoveKind::AskIdentity)
            && !known
            && !attribute_only
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
        // On 模式：用户更正后的会话内覆盖值优先于图谱旧值（改口在本会话内立即生效）。
        let mut used_correction = false;
        if self.trace_mode.is_on() {
            if let (Some(name), Some(attribute)) = (entity.as_deref(), frame.attribute.as_deref()) {
                if let Some(value) = self.trace.value_override(name, attribute) {
                    answer = Some(value.to_string());
                    used_correction = true;
                }
            }
        }

        // —— ABox：本轮的信息状态 ——
        let turn_id = format!("t{}", self.turns.len() + 1);
        let kind_atom = match frame.kind {
            MoveKind::MetaRecap => "meta_recap",
            MoveKind::MetaReview => "meta_review",
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
        if frame.question.as_deref() == Some("why_not") {
            abox.push(format!("why_not({})", turn_id));
        }
        if frame
            .question
            .as_deref()
            .is_some_and(|question| question.starts_with("step:"))
        {
            abox.push(format!("step_request({})", turn_id));
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
        let warrant = self.quality.as_ref().and_then(|index| {
            entity
                .as_deref()
                .and_then(|name| index.warrant_for(name, frame.attribute.as_deref()))
        });

        let text = match chosen.as_str() {
            "recap" => self.recap_text(),
            "review_answer" => {
                let (text, tag) = self.review_text(knowledge);
                plans.push(tag.to_string());
                text
            }
            "explain_why" => {
                if frame.question.as_deref() == Some("why_not") {
                    self.explain_last_failure(knowledge)
                } else if let Some(step) = frame
                    .question
                    .as_deref()
                    .and_then(|question| question.strip_prefix("step:"))
                    .and_then(|step| step.parse::<usize>().ok())
                {
                    self.explain_step(knowledge, step)
                } else {
                    self.explain_last_with(knowledge)
                }
            }
            "why_no_evidence" => self
                .failure_self_description()
                .unwrap_or_else(|| {
                    "我还没有答过什么问题，暂时没有可解释的依据。你可以先问我一个知识问题，再问我「你怎么知道的」。"
                        .to_string()
                }),
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
            "admit_ignorance" => {
                // 失败恢复：缺记录时走 L2（候选验证）/ L3（一问澄清）/ L4（承认）。
                if recovery_attempt.is_none() && !self.recovery_mode.eq(&RecoveryMode::Off) {
                    let status = if known {
                        EpistemicStatus::Absent
                    } else {
                        EpistemicStatus::Unknown
                    };
                    let request = RecoveryRequest {
                        status,
                        entity: entity.as_deref(),
                        attribute: frame.attribute.as_deref(),
                        detail: "",
                    };
                    let attempt = recovery::attempt(
                        &mut self.recovery,
                        self.recovery_mode,
                        &request,
                        &KnowledgeRecovery { knowledge },
                    );
                    self.note_recovery(&attempt);
                    recovery_attempt = Some(attempt);
                }
                self.compose_ignorance(
                    entity.as_deref().unwrap_or(""),
                    frame.attribute.as_deref(),
                    &more_attributes,
                )
            }
            "greet" => self.greet_text(),
            "thanks" => self.thanks_text(),
            "farewell" => self.farewell_text(),
            _ => return None,
        };
        // 提示资产接管：接地闸门通过则用模板渲染（On），双跑模式计数（Parity），
        // 任一 required 槽位未绑定则回退上面的组合生成器（结构保证不幻觉）。
        let text = self.apply_prompt_spec(
            &chosen,
            &frame,
            entity.as_deref(),
            answer.as_deref(),
            &more_attributes,
            text,
        );
        // On 模式：恢复引擎的文本（含证据链/追问/回执 id）取代生成器措辞；
        // Observe 模式只记台账，文本不变。
        let text = match &recovery_attempt {
            Some(attempt) if self.recovery_mode == RecoveryMode::On && !attempt.text.is_empty() => {
                attempt.text.clone()
            }
            _ => text,
        };
        let (text, plans) =
            self.apply_warrant(text, plans, warrant.as_ref(), &frame.kind, answer.is_some());

        let mut provenance = answer
            .as_deref()
            .map(classify_provenance)
            .unwrap_or_default();
        if used_correction {
            provenance.push("用户更正".to_string());
        }
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
                provenance: provenance.clone(),
                plan: plans.clone(),
                warrant,
                kb_revision: self.quality.as_ref().map(|index| index.revision),
            });
        }
        if let Some(entity) = &entity {
            self.push_focus(entity);
        }
        self.last_reply = plans.clone();
        if self.externals.last().map(String::as_str) == Some(frame.raw.trim()) {
            self.externals.pop();
        }
        let reply = DialogueReply {
            text: text.clone(),
            plan: plans,
            kind: frame.kind.clone(),
        };
        self.record_turn_trace(&frame, &entity, Some(&reply), &provenance);
        Some(reply)
    }

    /// 记录一次恢复动作的对外视图。
    fn note_recovery(&mut self, attempt: &recovery::RecoveryAttempt) {
        self.last_recovery = Some(RecoveryNotice {
            status: attempt.status,
            level: attempt.level.label().to_string(),
            receipt_id: attempt.receipt_id.clone(),
            evidence: attempt.evidence.clone(),
            question: attempt.question.clone(),
        });
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

    /// 用户改口拦截（仅 On 模式）：识别「不对，叶莉的身高是 190」这类更正，
    /// 覆盖会话内取值并把上一轮相关回复判为失效。
    fn try_trace_correction(
        &mut self,
        input: &str,
        knowledge: &dyn DialogueKnowledge,
    ) -> Option<DialogueReply> {
        if !self.trace_mode.is_on() {
            return None;
        }
        let stripped = input
            .trim()
            .trim_end_matches(['？', '?', '。', '！', '!', ' '])
            .trim();
        if stripped.is_empty() || stripped.chars().count() > 40 {
            return None;
        }
        const PREFIXES: [&str; 6] = ["不对，", "不是，", "错了，", "更正：", "改成", "应该是"];
        let body = PREFIXES
            .iter()
            .find_map(|prefix| stripped.strip_prefix(prefix))
            .map(str::trim)?;
        if body.is_empty() || body.contains('？') || body.contains('?') {
            return None;
        }
        // 上一轮必须是有实体+属性的知识回答，改口才有对象。
        let previous = self
            .turns
            .iter()
            .rev()
            .find(|turn| {
                turn.answer_excerpt.is_some() && turn.entity.is_some() && turn.attribute.is_some()
            })
            .cloned()?;
        let entity = previous.entity.clone()?;
        let attribute = previous.attribute.clone()?;
        // 从句式里抽出更正值：「…是 190」/「…为 190」/「…190」。
        let value = body
            .split(['是', '为'])
            .next_back()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| body.to_string());
        let previous_answer = previous.answer_excerpt.clone().unwrap_or_default();
        let turn_no = self.trace.turn_count() + 1;
        let (correction, marked) = self.trace.record_correction(
            turn_no,
            previous.index,
            &entity,
            &attribute,
            &value,
            &previous_answer,
        );
        // 更正轮也进入对话本体实例。
        self.turns.push(DialogueTurn {
            index: self.turns.len() + 1,
            user: stripped.to_string(),
            kind: MoveKind::Assert,
            entity: Some(entity.clone()),
            attribute: Some(attribute.clone()),
            answer_excerpt: Some(excerpt(&value, 40)),
            provenance: vec!["用户更正".to_string()],
            plan: vec!["trace_correction".to_string()],
            warrant: None,
            kb_revision: knowledge.revision().into(),
        });
        self.push_focus(&entity);
        let text = if marked > 0 {
            format!(
                "收到，已把「{}」的「{}」更新为「{}」，上一轮的说法作废。",
                entity, attribute, value
            )
        } else {
            format!(
                "收到，已记下「{}」的「{}」为「{}」。",
                entity, attribute, value
            )
        };
        let plan = vec!["trace_correction".to_string()];
        self.last_reply = plan.clone();
        if self.externals.last().map(String::as_str) == Some(stripped) {
            self.externals.pop();
        }
        let _ = correction;
        let reply = DialogueReply {
            text,
            plan,
            kind: MoveKind::Assert,
        };
        let frame = UtteranceFrame {
            raw: input.trim().to_string(),
            kind: MoveKind::Assert,
            entity: Some(entity),
            attribute: Some(attribute),
            question: None,
            inherited: false,
        };
        self.record_turn_trace(
            &frame,
            &frame.entity,
            Some(&reply),
            &["用户更正".to_string()],
        );
        Some(reply)
    }

    /// 把本轮决策写入分层追踪图（仅 Off 之外的模式）。
    ///
    /// 记录顺序：L0 表层 → L1 解读 → L2 指代/问据（谓词承继）→ L2 回答/让位。
    /// `reply == None` 表示让位给其它管线（记 defer 节点）。
    fn record_turn_trace(
        &mut self,
        frame: &UtteranceFrame,
        entity: &Option<String>,
        reply: Option<&DialogueReply>,
        provenance: &[String],
    ) {
        if self.trace_mode.is_off() {
            return;
        }
        // 若本轮已有节点（如更正已开启本轮），复用该轮号，避免产生空轮迹。
        let turn_no = if self
            .trace
            .turns()
            .last()
            .map(|turn| !turn.nodes.is_empty())
            .unwrap_or(false)
        {
            self.trace.current_turn()
        } else {
            self.trace.turn_count() + 1
        };
        let surface = self.trace.record_surface(turn_no, &frame.raw);
        let kind = format!("{:?}", frame.kind);
        let interpret = self.trace.record_interpret(
            turn_no,
            surface,
            &kind,
            entity.as_deref(),
            frame.attribute.as_deref(),
            frame.question.as_deref(),
            frame.inherited,
        );
        // 指代/省略登记：显式实体、代词、省略各有出处。
        if let Some(entity) = entity.as_deref() {
            let source = if frame.inherited {
                if contains_pronoun(&frame.raw) {
                    ReferentSource::Pronoun
                } else {
                    ReferentSource::Ellipsis
                }
            } else {
                ReferentSource::Explicit
            };
            let surface_form = if frame.inherited {
                frame.raw.trim().to_string()
            } else {
                entity.to_string()
            };
            self.trace.record_mention(
                turn_no,
                interpret,
                &surface_form,
                entity,
                source,
                frame.question.as_deref().unwrap_or(""),
            );
            self.trace.record_qud(
                turn_no,
                interpret,
                match frame.kind {
                    MoveKind::AskIdentity => QudKind::AskIdentity,
                    MoveKind::AskBinary => QudKind::AskBinary,
                    _ => QudKind::AskAttribute,
                },
                Some(entity),
                frame.attribute.as_deref(),
                frame.inherited,
            );
            if frame.inherited {
                if let Some(attribute) = frame.attribute.as_deref() {
                    let origin = self
                        .trace
                        .predicate_focus()
                        .unwrap_or("谓词焦点")
                        .to_string();
                    self.trace
                        .record_inheritance(turn_no, interpret, attribute, &origin);
                }
            }
        }
        match reply {
            Some(reply) => {
                let excerpt = excerpt(&reply.text, 80);
                self.trace.record_reply(
                    turn_no,
                    entity.as_deref(),
                    frame.attribute.as_deref(),
                    vec![interpret],
                    provenance,
                    Some(&excerpt),
                );
                self.trace.resolve_open_qud();
            }
            None => {
                self.trace
                    .record_defer(turn_no, interpret, "知识管线未接管本轮输入");
            }
        }
    }

    /// 复核上一轮回答：比较当时保存的担保与当前知识版本。
    ///
    /// 关键约束：删掉的事实绝不能被说成「现在仍为真」；新增冲突必须显式
    /// 披露；版本推进但事实未变时只更新「仍成立」而不制造虚假变化。
    fn review_text(&self, knowledge: &dyn DialogueKnowledge) -> (String, &'static str) {
        let Some(turn) = self.last_answered_turn() else {
            return (
                "我还没有给出过可复核的知识回答。先问我一个具体问题吧。".to_string(),
                "review_empty",
            );
        };
        let Some(entity) = turn.entity.as_deref() else {
            return (
                "上一轮没有落到具体实体，暂时无法复核。".to_string(),
                "review_empty",
            );
        };
        let attribute = turn.attribute.as_deref();
        let current_answer = if let Some(attribute) = attribute {
            self.lookup_attribute(knowledge, entity, attribute)
        } else {
            knowledge
                .answer(&format!("{}是什么", entity))
                .or_else(|| knowledge.describe(entity))
        };
        let current_warrant = self
            .quality
            .as_ref()
            .and_then(|index| index.warrant_for(entity, attribute));
        let current_revision = self.quality_revision();
        let subject = match attribute {
            Some(attribute) => format!("「{}」的「{}」", entity, attribute),
            None => format!("「{}」", entity),
        };
        if current_answer.is_none() {
            return (
                format!(
                    "复核结果：{}现在已没有记录，上一轮的说法不再作为当前事实。",
                    subject
                ),
                "review_stale",
            );
        }
        if let (Some(old), Some(new)) = (turn.warrant.as_ref(), current_warrant.as_ref()) {
            if old.known_values != new.known_values {
                let old_values = if old.known_values.is_empty() {
                    "（无）".to_string()
                } else {
                    old.known_values.join("、")
                };
                let new_values = if new.known_values.is_empty() {
                    "（无）".to_string()
                } else {
                    new.known_values.join("、")
                };
                return (
                    format!(
                        "复核结果：知识已变化。{}上一轮记录为「{}」，现在记录为「{}」，以现在为准。",
                        subject, old_values, new_values
                    ),
                    "review_changed",
                );
            }
            let new_conflicts: Vec<&crate::predicate_registry::ImportConflict> = new
                .conflicts
                .iter()
                .filter(|conflict| !old.conflicts.contains(*conflict))
                .collect();
            if !new_conflicts.is_empty() {
                let incoming = new_conflicts
                    .iter()
                    .map(|conflict| conflict.incoming.clone())
                    .collect::<Vec<_>>()
                    .join("、");
                return (
                    format!(
                        "复核结果：{}仍为「{}」，但出现了新的不同说法「{}」，需要你确认哪个为准。",
                        subject,
                        old.known_values.join("、"),
                        incoming
                    ),
                    "review_conflict",
                );
            }
            // 缺口变化也是知识变化：删掉必备属性/类型后，答案本身可能仍为真，
            // 但「当时完整、现在有缺口」必须可见，不能只说「版本推进了」。
            let conflict_gap = |gap: &&String| gap.starts_with("存在未裁决冲突");
            let new_gaps: Vec<&String> = new
                .gaps
                .iter()
                .filter(|gap| !old.gaps.contains(gap) && !conflict_gap(gap))
                .collect();
            let resolved_gaps: Vec<&String> = old
                .gaps
                .iter()
                .filter(|gap| !new.gaps.contains(gap) && !conflict_gap(gap))
                .collect();
            if !new_gaps.is_empty() || !resolved_gaps.is_empty() {
                let mut changes = Vec::new();
                if !new_gaps.is_empty() {
                    changes.push(format!(
                        "新增缺口「{}」",
                        new_gaps
                            .iter()
                            .take(2)
                            .map(|gap| gap.as_str())
                            .collect::<Vec<_>>()
                            .join("」「")
                    ));
                }
                if !resolved_gaps.is_empty() {
                    changes.push(format!(
                        "已补齐「{}」",
                        resolved_gaps
                            .iter()
                            .take(2)
                            .map(|gap| gap.as_str())
                            .collect::<Vec<_>>()
                            .join("」「")
                    ));
                }
                let still = if old.known_values.is_empty() {
                    "该回答仍成立".to_string()
                } else {
                    format!("仍为「{}」", old.known_values.join("、"))
                };
                return (
                    format!(
                        "复核结果：{}{}，但知识质量有变化：{}。事实本身不变，缺口可用于下一次补全。",
                        subject,
                        still,
                        changes.join("；")
                    ),
                    "review_gap",
                );
            }
            if current_revision != Some(old.revision) {
                return (
                    format!(
                        "复核结果：知识版本已从 {} 推进到 {}，但{}仍然成立。",
                        old.revision,
                        current_revision.unwrap_or(old.revision),
                        subject
                    ),
                    "review_confirmed",
                );
            }
            return (
                format!("复核结果：知识版本没有变化，{}仍然成立。", subject),
                "review_confirmed",
            );
        }
        let excerpt = current_answer
            .as_deref()
            .map(|answer| excerpt(answer, 60))
            .unwrap_or_default();
        (
            format!("复核结果：现有记录仍支持上一轮的回答：{}", excerpt),
            "review_confirmed",
        )
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
            text.push_str(&format!(
                "另外你还输入了 {} 条其他指令。",
                self.externals.len()
            ));
        }
        text
    }

    /// 失败自省：恢复模式开启且台账非空时，把失败回执讲给用户听。
    ///
    /// 绝不编造依据：没有成功回答时不假装有，只有回执可引用。
    fn failure_self_description(&self) -> Option<String> {
        if self.recovery_mode.eq(&RecoveryMode::Off) || self.recovery.ledger.is_empty() {
            return None;
        }
        let receipt = self
            .recovery
            .ledger
            .open()
            .next()
            .or_else(|| self.recovery.ledger.last())?;
        let framing = if receipt.is_open() {
            "还有一条没结清的失败回执"
        } else {
            "最近一条失败回执"
        };
        Some(format!(
            "我还没有答过带依据的问题；这里有一条失败回执——{}：{}。失败回执和成功回答一样会被记录（状态、档位、证据都可回溯），我不会为此编造依据。",
            framing,
            receipt.render_line()
        ))
    }

    /// 元话语：解释上一轮回答的依据（把推理出处「讲」出来）。
    ///
    /// 该入口不依赖知识源，保留给只读调用方；带知识源的路径见
    /// [`Self::explain_last_with`]，会优先重建一条可展开的推导。
    pub fn explain_last(&self) -> String {
        self.explain_last_inner(None)
    }

    /// 带知识源的依据解释：先尝试从上一轮问句重建有界推导，
    /// 重建不到（或证明与回答不匹配）时诚实退回出处标签。
    pub fn explain_last_with(&self, knowledge: &dyn DialogueKnowledge) -> String {
        self.explain_last_inner(Some(knowledge))
    }

    fn explain_last_inner(&self, knowledge: Option<&dyn DialogueKnowledge>) -> String {
        let Some(turn) = self.last_answered_turn() else {
            return self.failure_self_description().unwrap_or_else(|| {
                "我还没有答过什么问题，暂时没有可解释的依据。先问我一个知识问题吧。".to_string()
            });
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
        if let Some(knowledge) = knowledge {
            if let Some(derivation) = self.explain_derivation(turn, knowledge) {
                let mut text = format!(
                    "刚才关于{}的回答，{}",
                    topic,
                    derivation.render_explanation()
                );
                if let Some(recorded) = turn.kb_revision {
                    let current = knowledge.revision();
                    if recorded != current {
                        text.push_str(&format!(
                            "\n说明：这条证明按当前知识版本 v{} 重建（回答时是 v{}）；若知识已变化，原回答仍保留在对话记忆里。",
                            current, recorded
                        ));
                    }
                }
                text.push_str(&format!("\n出处标签：{}。", provenance.join("、")));
                return text;
            }
        }
        format!(
            "刚才关于{}的回答，依据是：{}。这些结论都来自知识图谱与规则程序（逐条符号匹配、逻辑推导），没有任何神经网络参与。",
            topic,
            provenance.join("、")
        )
    }

    /// 「为什么没有/查不到」：不改口，也不把失败说成不存在；
    /// 只展开最近一次未答上的知识问句所对应的失败前沿。
    pub fn explain_last_failure(&self, knowledge: &dyn DialogueKnowledge) -> String {
        let Some(turn) = self.turns.iter().rev().find(|turn| {
            turn.answer_excerpt.is_none()
                && matches!(
                    turn.kind,
                    MoveKind::AskAttribute | MoveKind::AskIdentity | MoveKind::AskBinary
                )
        }) else {
            return "最近没有没答上的知识问句；上一轮是有结论的，可以问「怎么推出来的」。".to_string();
        };
        let topic = match (&turn.entity, &turn.attribute) {
            (Some(entity), Some(attribute)) => format!("「{}」的「{}」", entity, attribute),
            (Some(entity), None) => format!("「{}」", entity),
            _ => format!("「{}」", excerpt(&turn.user, 16)),
        };
        let Some(question) = turn_question(turn) else {
            return format!(
                "刚才关于{}的问题没有结论；这个问句没有可形式化的实体与属性槽位，因此没有可展开的失败前沿。",
                topic
            );
        };
        match knowledge.derivation(&question, None) {
            Some(derivation) => {
                let mut text = format!(
                    "刚才关于{}没有作答，这里不改口：{}",
                    topic,
                    derivation.render_explanation()
                );
                if let Some(recorded) = turn.kb_revision {
                    let current = knowledge.revision();
                    if recorded != current {
                        text.push_str(&format!(
                            "\n说明：失败前沿按当前知识版本 v{} 重建（当时是 v{}）。",
                            current, recorded
                        ));
                    }
                }
                text.push_str("\n失败前沿只说明当前事实或规则的缺口，不是「它不存在」的证明。");
                text
            }
            None => format!(
                "刚才关于{}没有作答；这个问句无法形式化为归结目标，因此没有可展开的失败前沿（属于检索或模板管线未命中）。",
                topic
            ),
        }
    }

    /// 单步追问：只展开上一轮推导的第 `step` 步（从 1 开始）。
    pub fn explain_step(&self, knowledge: &dyn DialogueKnowledge, step: usize) -> String {
        let Some(derivation) = self.last_derivation(knowledge) else {
            return "上一轮没有可展开的推导，暂时不能按步骤解释。可以先问「怎么推出来的」。"
                .to_string();
        };
        let total = derivation.steps.len();
        match derivation.render_step(step) {
            Some(line) => {
                let mut text = line;
                if derivation.alternatives > 0 {
                    text.push_str(&format!(
                        "\n完整证明共 {} 步；另有 {} 条独立证明未展开。",
                        total, derivation.alternatives
                    ));
                } else {
                    text.push_str(&format!("\n完整证明共 {} 步。", total));
                }
                if derivation.truncated || derivation.omitted_steps > 0 {
                    text.push_str("部分节点因预算未展开，步骤数可能少于真实证明。");
                }
                text
            }
            None => format!(
                "上一轮推导只展开了 {} 步，没有第 {} 步；可以问「怎么推出来的」查看全貌。",
                total, step
            ),
        }
    }

    /// 只读：为最近一条已答轮次重建推导（与 [`Self::explain_last_with`] 同源）。
    /// 供 HTTP/CLI 在「为什么」轮次返回结构化证明，不写入会话状态。
    pub fn last_derivation(&self, knowledge: &dyn DialogueKnowledge) -> Option<Derivation> {
        let turn = self.last_answered_turn()?;
        self.explain_derivation(turn, knowledge)
    }

    /// 为上一轮回答重建一条与实际答案锚定的推导：
    /// 先走知识源的归结证明；没有形式证明时，再按实体属性/关系边
    /// 组装直接事实链（只取回答文本中确实出现的取值）。
    fn explain_derivation(
        &self,
        turn: &DialogueTurn,
        knowledge: &dyn DialogueKnowledge,
    ) -> Option<Derivation> {
        let question = turn_question(turn);
        let expected = turn.answer_excerpt.as_deref();
        if let Some(question) = question.as_deref() {
            if let Some(derivation) = knowledge.derivation(question, expected) {
                if derivation.is_proved() {
                    return Some(derivation);
                }
            }
        }
        self.direct_fact_derivation(turn, knowledge)
    }

    fn direct_fact_derivation(
        &self,
        turn: &DialogueTurn,
        knowledge: &dyn DialogueKnowledge,
    ) -> Option<Derivation> {
        let entity = turn.entity.as_deref()?;
        let attribute = turn.attribute.as_deref()?;
        let expected = turn.answer_excerpt.as_deref()?;
        let mut facts: Vec<(String, String, String)> = Vec::new();

        for value in knowledge.attribute_values(entity, attribute) {
            if !value.is_empty() && expected.contains(value.as_str()) {
                facts.push((
                    entity.to_string(),
                    surface_attribute(attribute).to_string(),
                    value,
                ));
            }
        }
        let canonical = knowledge
            .canonical_predicate(attribute)
            .unwrap_or_else(|| attribute.to_string());
        for (predicate, object) in knowledge.entity_relations(entity) {
            if object.is_empty() || !expected.contains(object.as_str()) {
                continue;
            }
            let label = knowledge.predicate_label(&predicate);
            if predicate == attribute || label == attribute || predicate == canonical {
                facts.push((entity.to_string(), label, object));
            }
        }
        facts.sort();
        facts.dedup();
        if facts.is_empty() {
            return None;
        }
        Some(Derivation::from_direct_facts(
            format!(
                "{} 的 {}",
                entity,
                surface_attribute(attribute)
            ),
            facts,
        ))
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
            !self.turns.iter().any(|turn| {
                turn.entity.as_deref() == Some(entity)
                    && turn.attribute.as_deref() == Some(attribute.as_str())
            })
        }) {
            if let Some(answer) = self.lookup_attribute(knowledge, entity, attribute) {
                return format!(
                    "接着说说「{}」的「{}」：{}",
                    entity,
                    attribute,
                    answer.trim()
                );
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
                parts.push(format!(
                    "{}：{}",
                    surface_attribute(attribute),
                    answer.trim()
                ));
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

    /// 把质量担保翻译成话语行为：冲突披露 → 类型澄清 → 一个缺口 → 引用方式。
    ///
    /// 顺序固定，最多追加四条（每类至多一条），因此回答不会因质量问题无限膨胀。
    /// 担保只改「怎么说」，不改答案本身，也不把低分/缺来源说成事实为假。
    fn apply_warrant(
        &self,
        mut text: String,
        mut plans: Vec<String>,
        warrant: Option<&ClaimWarrant>,
        kind: &MoveKind,
        answered: bool,
    ) -> (String, Vec<String>) {
        let Some(warrant) = warrant else {
            return (text, plans);
        };
        let ask_like = matches!(
            kind,
            MoveKind::AskAttribute | MoveKind::AskIdentity | MoveKind::AskBinary
        );
        if !ask_like {
            return (text, plans);
        }
        if let Some(note) = warrant.conflict_note() {
            text.push('\n');
            text.push_str(&note);
            plans.push("quality_disclose_conflict".to_string());
        }
        if warrant.needs_type_clarification() && matches!(kind, MoveKind::AskIdentity) {
            text.push_str("\n另外，我还不能确定它的类型（人物/地点/组织等），你能补充一下吗？");
            plans.push("quality_clarify_type".to_string());
        }
        if let Some(gap) = warrant.actionable_gap() {
            text.push_str(&format!("\n补充线索：{}。", gap));
            plans.push("quality_offer_gap".to_string());
        }
        if answered {
            text.push_str(&warrant.citation_note());
            plans.push(if warrant.has_source() {
                "quality_cite_source".to_string()
            } else {
                "quality_no_source".to_string()
            });
        }
        (text, plans)
    }

    fn compose_direct(&self, answer: &str, entity: &str, more: &[String]) -> String {
        let mut text = answer.trim().to_string();
        if !more.is_empty() {
            let listed = more
                .iter()
                .take(3)
                .map(|attribute| surface_attribute(attribute).to_string())
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
                .map(|attribute| surface_attribute(attribute).to_string())
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

    /// 选策略 → 选 spec 变体 → 绑定槽位；未外化的策略返回 None。
    ///
    /// 槽位值必须与组合生成器使用的是同一批运行时值，
    /// 这样 Parity 模式的逐字比对才有意义（差异 = 模板资产的真差异）。
    fn prompt_bindings(
        &self,
        policy: &str,
        frame: &UtteranceFrame,
        entity: Option<&str>,
        answer: Option<&str>,
        more: &[String],
    ) -> Option<(&'static str, BindingSet)> {
        let listed = || {
            more.iter()
                .take(3)
                .map(|attribute| surface_attribute(attribute).to_string())
                .collect::<Vec<_>>()
                .join("」「")
        };
        match policy {
            "direct_answer" => {
                if more.is_empty() {
                    let mut bindings = BindingSet::new();
                    if let Some(answer) = answer {
                        bindings.bind("answer", answer.trim());
                    }
                    Some(("direct_answer/plain", bindings))
                } else {
                    let ids = [
                        "direct_answer/more/0",
                        "direct_answer/more/1",
                        "direct_answer/more/2",
                    ];
                    let id = ids[self.turns.len() % ids.len()];
                    let mut bindings = BindingSet::new();
                    if let Some(answer) = answer {
                        bindings.bind("answer", answer.trim());
                    }
                    bindings.bind("entity", entity.unwrap_or(""));
                    bindings.bind("listed", &listed());
                    bindings.bind("count", &more.len().to_string());
                    Some((id, bindings))
                }
            }
            "admit_ignorance" => {
                let id = match (frame.attribute.is_some(), more.is_empty()) {
                    (true, true) => "admit_ignorance/attribute/plain",
                    (true, false) => "admit_ignorance/attribute/more",
                    (false, true) => "admit_ignorance/identity/plain",
                    (false, false) => "admit_ignorance/identity/more",
                };
                let mut bindings = BindingSet::new();
                bindings.bind("entity", entity.unwrap_or(""));
                if let Some(attribute) = frame.attribute.as_deref() {
                    bindings.bind("attribute", attribute);
                }
                if !more.is_empty() {
                    bindings.bind("listed", &listed());
                    bindings.bind("count", &more.len().to_string());
                }
                Some((id, bindings))
            }
            "clarify_entity" => {
                let candidates: Vec<String> = self.focus.iter().rev().take(3).cloned().collect();
                match candidates.split_last() {
                    Some((last, rest)) if !rest.is_empty() => {
                        let mut bindings = BindingSet::new();
                        if let Some(attribute) = frame.attribute.as_deref() {
                            bindings.bind("attribute", attribute);
                        }
                        bindings.bind("candidates", &rest.join("」「"));
                        bindings.bind("last", last);
                        Some(("clarify_entity/multi", bindings))
                    }
                    _ => {
                        let mut bindings = BindingSet::new();
                        if let Some(attribute) = frame.attribute.as_deref() {
                            bindings.bind("attribute", attribute);
                        }
                        Some(("clarify_entity/single", bindings))
                    }
                }
            }
            "clarify_entity_no_focus" => {
                let mut bindings = BindingSet::new();
                if let Some(attribute) = frame.attribute.as_deref() {
                    bindings.bind("attribute", attribute);
                }
                Some(("clarify_entity_no_focus/plain", bindings))
            }
            "greet" => {
                if self.turns.is_empty() {
                    Some(("greet/first", BindingSet::new()))
                } else if let Some(focus) = self.focus.last() {
                    let mut bindings = BindingSet::new();
                    bindings.bind("focus", focus);
                    Some(("greet/returning_focus", bindings))
                } else {
                    Some(("greet/returning_plain", BindingSet::new()))
                }
            }
            "thanks" => match self.focus.last() {
                Some(focus) => {
                    let mut bindings = BindingSet::new();
                    bindings.bind("focus", focus);
                    Some(("thanks/focus", bindings))
                }
                None => Some(("thanks/plain", BindingSet::new())),
            },
            "farewell" => {
                let mut bindings = BindingSet::new();
                bindings.bind("turn_count", &self.turns.len().to_string());
                Some(("farewell/plain", bindings))
            }
            "why_no_evidence" => Some(("why_no_evidence/plain", BindingSet::new())),
            "continue_no_focus" => Some(("continue_no_focus/plain", BindingSet::new())),
            "explain_why" => {
                if frame.question.as_deref() == Some("why_not")
                    || frame
                        .question
                        .as_deref()
                        .is_some_and(|question| question.starts_with("step:"))
                {
                    return None;
                }
                let turn = self.last_answered_turn()?;
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
                let mut bindings = BindingSet::new();
                bindings.bind("topic", &topic);
                bindings.bind("provenance", &provenance.join("、"));
                Some(("explain_why/plain", bindings))
            }
            _ => None,
        }
    }

    /// 模板接管：Parity 双跑计数、On 用模板文本、接地失败回退组合生成器文本。
    fn apply_prompt_spec(
        &mut self,
        policy: &str,
        frame: &UtteranceFrame,
        entity: Option<&str>,
        answer: Option<&str>,
        more: &[String],
        composed: String,
    ) -> String {
        if self.prompt_mode == PromptMode::Off {
            return composed;
        }
        let Some((spec_id, bindings)) = self.prompt_bindings(policy, frame, entity, answer, more)
        else {
            self.prompt_stats.composed += 1;
            return composed;
        };
        self.prompt_stats.checked += 1;
        match prompt_spec::render(spec_id, &bindings) {
            RenderOutcome::Rendered { text, .. } => {
                self.prompt_stats.grounded += 1;
                if text != composed {
                    self.prompt_stats.parity_mismatches += 1;
                    self.prompt_stats.record_mismatch(format!(
                        "{}：模板与组合生成器文本不一致\n    模板: {}\n    组合: {}",
                        spec_id,
                        text.replace('\n', "⏎"),
                        composed.replace('\n', "⏎")
                    ));
                }
                match self.prompt_mode {
                    PromptMode::On => text,
                    _ => composed,
                }
            }
            RenderOutcome::Ungrounded { missing, .. } => {
                self.prompt_stats.ungrounded += 1;
                self.prompt_stats.record_missing(&missing);
                composed
            }
            RenderOutcome::Invalid { message, .. } => {
                self.prompt_stats.ungrounded += 1;
                self.prompt_stats
                    .record_mismatch(format!("{} 资产无效：{}", spec_id, message));
                composed
            }
        }
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

/// 属性键的展示词面：英文内置键（name/age 等）本地化为中文，
/// 未知键保持原样（查询仍用原始键，只有展示走词面）。
fn surface_attribute(attribute: &str) -> &str {
    crate::symbolic_generator::localized_predicate_key(attribute).unwrap_or(attribute)
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

/// 解析「第 N 步」追问（阿拉伯数字与 1–99 的中文数字）。
fn parse_step_request(text: &str) -> Option<usize> {
    let start = text.find('第')?;
    let tail = &text[start + '第'.len_utf8()..];
    let end = tail.find('步')?;
    let raw = tail[..end].trim();
    if raw.is_empty() {
        return None;
    }
    if let Ok(step) = raw.parse::<usize>() {
        return (step > 0).then_some(step);
    }
    let digit = |ch: char| match ch {
        '零' => Some(0),
        '一' => Some(1),
        '二' => Some(2),
        '三' => Some(3),
        '四' => Some(4),
        '五' => Some(5),
        '六' => Some(6),
        '七' => Some(7),
        '八' => Some(8),
        '九' => Some(9),
        _ => None,
    };
    if let Some(pos) = raw.find('十') {
        let tens = if pos == 0 {
            1
        } else {
            raw[..pos].chars().next().and_then(digit)?
        };
        let ones = if pos + '十'.len_utf8() == raw.len() {
            0
        } else {
            raw[pos + '十'.len_utf8()..].chars().next().and_then(digit)?
        };
        return Some(tens * 10 + ones).filter(|step| *step > 0);
    }
    let mut value = 0usize;
    for ch in raw.chars() {
        value = value * 10 + digit(ch)?;
    }
    (value > 0).then_some(value)
}

fn turn_question(turn: &DialogueTurn) -> Option<String> {
    match (&turn.entity, &turn.attribute, &turn.kind) {
        (Some(entity), Some(attribute), _) => Some(format!("{}的{}是什么", entity, attribute)),
        (Some(entity), None, MoveKind::AskIdentity) => Some(format!("{}是谁", entity)),
        _ => None,
    }
}

fn excerpt(text: &str, limit: usize) -> String {
    // 推理器的多行回答常以「根据知识图谱：」这类标题开头；摘要取第一条
    // 实质内容行，避免「依据」只剩标题、无法与答案取值对齐。
    let lines: Vec<&str> = text.lines().map(str::trim).collect();
    let first_line = lines
        .iter()
        .copied()
        .find(|line| !line.is_empty() && !line.ends_with('：') && !line.ends_with(':'))
        .or_else(|| lines.iter().copied().find(|line| !line.is_empty()))
        .unwrap_or("");
    let mut out: String = first_line.chars().take(limit).collect();
    if first_line.chars().count() > limit {
        out.push('…');
    }
    out
}

fn contains_pronoun(text: &str) -> bool {
    let cleaned = text
        .replace("其他", "")
        .replace("其中", "")
        .replace("其实", "");
    [
        "他们",
        "她们",
        "它们",
        "这个人",
        "那位",
        "这位",
        "他",
        "她",
        "它",
    ]
    .iter()
    .any(|pronoun| cleaned.contains(pronoun))
}

fn split_entity_attribute(text: &str) -> Option<(&str, &str)> {
    let index = text.find('的')?;
    let head = text[..index].trim();
    let mut tail = text[index + '的'.len_utf8()..].trim();
    // 连续剥离（「…到底是多少呢」需要先剥「呢」再剥「是多少」）。
    for _ in 0..3 {
        let mut stripped_once = false;
        for suffix in [
            "是什么",
            "是多少",
            "是谁",
            "是哪个",
            "是哪里",
            "怎么样",
            "多少",
            "呢",
            "吗",
        ] {
            if let Some(stripped) = tail.strip_suffix(suffix) {
                tail = stripped.trim();
                stripped_once = true;
                break;
            }
        }
        if !stripped_once {
            break;
        }
    }
    // 语气/强调填充词不承载属性：先把「到底是谁」剥成「到底」，再归一成属性。
    for filler in ["到底", "究竟", "来着", "呀", "啊", "嘛"] {
        if let Some(stripped) = tail.strip_suffix(filler) {
            let stripped = stripped.trim();
            if !stripped.is_empty() {
                tail = stripped;
                break;
            }
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
        derivations: BTreeMap<String, Derivation>,
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
                derivations: BTreeMap::new(),
            }
        }

        fn with_value(mut self, entity: &str, attribute: &str, value: &str) -> Self {
            self.values.insert(
                (entity.to_string(), attribute.to_string()),
                vec![value.to_string()],
            );
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

        fn with_derivation(mut self, question: &str, derivation: Derivation) -> Self {
            self.derivations
                .insert(question.to_string(), derivation);
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

        fn derivation(&self, question: &str, expected: Option<&str>) -> Option<Derivation> {
            let derivation = self.derivations.get(question)?;
            if let Some(expected) = expected {
                if let Some(value) = derivation.answer_value() {
                    if !expected.contains(value) {
                        return None;
                    }
                }
            }
            Some(derivation.clone())
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

        fn all_entities(&self) -> Vec<String> {
            let mut entities = self.entities.clone();
            entities.sort();
            entities.dedup();
            entities
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
    fn test_analyze_attribute_question_ignores_emphasis_fillers() {
        let knowledge = mock();
        let manager = DialogueManager::new();
        for input in ["姚明的身高到底是多少？", "姚明的身高究竟是多少呢？"] {
            let frame = manager.analyze(input, &knowledge);
            assert_eq!(frame.kind, MoveKind::AskAttribute, "{}", input);
            assert_eq!(frame.entity.as_deref(), Some("姚明"), "{}", input);
            assert_eq!(frame.attribute.as_deref(), Some("身高"), "{}", input);
        }

        let mut manager = DialogueManager::new();
        let reply = manager
            .plan_reply("姚明的身高到底是多少呢？", &knowledge)
            .expect("填充词不应阻断属性问句");
        assert!(reply.text.contains("226"), "{}", reply.text);
    }

    #[test]
    fn test_note_plugin_turn_updates_focus_and_conversation_body() {
        let mut manager = DialogueManager::new();
        manager.note_plugin_turn(
            "阿凡达是谁导演的",
            MoveKind::AskAttribute,
            Some("阿凡达"),
            Some("导演"),
            "「阿凡达」的导演是詹姆斯·卡梅隆。",
            vec!["plugin:zh.film:film.directed_by".to_string()],
            vec!["插件 zh.film / 配方 film.directed_by".to_string()],
        );
        assert_eq!(manager.turn_count(), 1);
        assert_eq!(manager.focus_entity(), Some("阿凡达"));
        let turn = &manager.turns()[0];
        assert_eq!(turn.entity.as_deref(), Some("阿凡达"));
        assert_eq!(turn.attribute.as_deref(), Some("导演"));
        assert!(turn
            .answer_excerpt
            .as_deref()
            .unwrap_or_default()
            .contains("詹姆斯"));
        assert_eq!(
            manager.last_reply_plan(),
            &["plugin:zh.film:film.directed_by".to_string()]
        );
        assert_eq!(
            manager.analyze("还有呢", &mock()).kind,
            MoveKind::ContinueRequest
        );
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
        assert_eq!(
            manager.analyze("还有呢", &knowledge).kind,
            MoveKind::ContinueRequest
        );
    }

    #[test]
    fn test_direct_answer_plan_and_followup_offer() {
        let knowledge = mock();
        let mut manager = DialogueManager::new();
        let reply = manager
            .plan_reply("姚明的身高是多少", &knowledge)
            .expect("已知实体且有答案时应接管");
        assert!(
            reply.plan.iter().any(|plan| plan == "direct_answer"),
            "策略：{:?}",
            reply.plan
        );
        assert!(reply.plan.iter().any(|plan| plan == "offer_related"));
        assert!(reply.text.contains("226"));
        assert!(
            reply.text.contains("国籍"),
            "应附带追问建议：{}",
            reply.text
        );
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
        assert!(manager
            .plan_reply("诸葛亮的身高是多少", &knowledge)
            .is_none());
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
    fn test_explain_why_rebuilds_a_real_derivation() {
        let derivation = Derivation::from_direct_facts(
            "姚明的身高",
            vec![(
                "姚明".to_string(),
                "身高".to_string(),
                "226 厘米".to_string(),
            )],
        );
        let knowledge = MockKnowledge::new()
            .with_entity("姚明", &["身高"])
            .with_answer("姚明的身高是什么", "姚明的身高是 226 厘米。")
            .with_derivation("姚明的身高是什么", derivation);
        let mut manager = DialogueManager::new();
        manager
            .plan_reply("姚明的身高是多少", &knowledge)
            .expect("属性问句应被接管");
        let why = manager
            .plan_pure_dialogue("怎么推出来的？", &knowledge)
            .expect("推导追问应被接管");
        assert!(why.text.contains("证明步骤"), "{}", why.text);
        assert!(why.text.contains("226 厘米"), "{}", why.text);
        assert!(why.text.contains("事实"), "{}", why.text);
        assert!(why.text.contains("出处标签"), "{}", why.text);
    }

    #[test]
    fn test_step_followup_explains_one_proof_node() {
        let derivation = Derivation::from_direct_facts(
            "姚明的身高",
            vec![
                (
                    "姚明".to_string(),
                    "身高".to_string(),
                    "226 厘米".to_string(),
                ),
                (
                    "姚明".to_string(),
                    "体重".to_string(),
                    "140 公斤".to_string(),
                ),
            ],
        );
        let knowledge = MockKnowledge::new()
            .with_entity("姚明", &["身高"])
            .with_answer("姚明的身高是什么", "姚明的身高是 226 厘米。")
            .with_derivation("姚明的身高是什么", derivation);
        let mut manager = DialogueManager::new();
        manager
            .plan_reply("姚明的身高是多少", &knowledge)
            .expect("属性问句应被接管");
        let step = manager
            .plan_pure_dialogue("第二步是什么？", &knowledge)
            .expect("步骤追问应被接管");
        assert!(step.text.contains("第 2 步"), "{}", step.text);
        assert!(step.text.contains("140 公斤"), "{}", step.text);
        assert!(step.text.contains("完整证明共 2 步"), "{}", step.text);

        let out_of_range = manager
            .plan_pure_dialogue("第九步呢？", &knowledge)
            .expect("越界步骤也应给出可行动作");
        assert!(out_of_range.text.contains("没有第 9 步"), "{}", out_of_range.text);
    }

    #[test]
    fn test_explain_why_falls_back_to_direct_fact_edges() {
        let knowledge = MockKnowledge::new()
            .with_entity("姚明", &["身高"])
            .with_value("姚明", "身高", "226 厘米")
            .with_answer("姚明的身高是什么", "姚明的身高是 226 厘米。");
        let mut manager = DialogueManager::new();
        manager
            .plan_reply("姚明的身高是多少", &knowledge)
            .expect("属性问句应被接管");
        let why = manager
            .plan_pure_dialogue("依据是什么？", &knowledge)
            .expect("依据追问应被接管");
        assert!(why.text.contains("证明步骤"), "{}", why.text);
        assert!(why.text.contains("姚明 的 身高 是 226 厘米"), "{}", why.text);
    }

    #[test]
    fn test_explain_why_not_uses_failure_frontier_without_prior_answer() {
        let knowledge = MockKnowledge::new()
            .with_entity("姚明", &["身高"])
            .with_answer("姚明的身高是什么", "姚明的身高是 226 厘米。")
            .with_derivation(
                "姚明的体重是什么",
                Derivation::missing(
                    "姚明的体重",
                    vec!["体重(姚明, ?X)（缺少事实或规则，深度 0）".to_string()],
                ),
            );
        let mut manager = DialogueManager::new();
        let reply = manager
            .plan_reply("姚明的体重是多少", &knowledge)
            .expect("已知实体即使无答案也应坦白");
        assert!(reply.text.contains("没有记录"), "{}", reply.text);
        let why = manager
            .plan_pure_dialogue("为什么没有记录？", &knowledge)
            .expect("失败追问应被接管");
        assert!(why.text.contains("没有作答"), "{}", why.text);
        assert!(why.text.contains("体重"), "{}", why.text);
        assert!(why.text.contains("知识缺口"), "{}", why.text);
        assert!(why.text.contains("不是「它不存在」"), "{}", why.text);
    }

    #[test]
    fn test_continue_topic_picks_unasked_attribute() {
        let knowledge = mock();
        let mut manager = DialogueManager::new();
        manager.plan_reply("姚明的身高是多少", &knowledge).unwrap();
        let reply = manager
            .plan_pure_dialogue("还有呢", &knowledge)
            .expect("续聊应被接管");
        assert!(
            reply.plan.iter().any(|plan| plan == "continue_topic"),
            "{:?}",
            reply.plan
        );
        assert!(
            reply.text.contains("国籍"),
            "应挑未问过的属性：{}",
            reply.text
        );
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
    fn test_snapshot_roundtrip_writes_canonical_bytes_with_receipt() {
        let knowledge = mock();
        let mut manager = DialogueManager::new();
        manager.plan_reply("姚明是谁", &knowledge).unwrap();
        let path = std::env::temp_dir().join(format!(
            "knowprolog_dialogue_manager_canonical_{}.json",
            std::process::id()
        ));
        let path_str = path.to_string_lossy().to_string();
        manager.save_to_path(&path_str).expect("应能保存");
        let bytes = std::fs::read(&path).unwrap();
        let snapshot = manager.snapshot();
        assert_eq!(
            crate::canonical::canonical_bytes(&snapshot).unwrap(),
            bytes,
            "落盘必须是规范字节（L2 字节不动点）"
        );

        let mut restored = DialogueManager::new();
        let receipt = restored
            .restore_from_path_report(&path_str)
            .expect("应能恢复");
        assert!(receipt.canonical);
        assert_eq!(receipt.kind, "dialogue_manager");
        assert_eq!(restored.snapshot(), snapshot);

        // 失败都带稳定错误码：坏字节/缺失文件分类明确
        let bad = std::env::temp_dir().join(format!(
            "knowprolog_dialogue_manager_bad_{}.json",
            std::process::id()
        ));
        std::fs::write(&bad, b"{broken").unwrap();
        let error = DialogueManager::new()
            .restore_from_path_report(bad.to_str().unwrap())
            .unwrap_err();
        assert_eq!(error.code(), "parse");
        let error = DialogueManager::new()
            .restore_from_path_report("/nonexistent/definitely/not/here.json")
            .unwrap_err();
        assert_eq!(error.code(), "io");
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&bad);
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
        let farewell = manager.plan_pure_dialogue("再见", &knowledge).unwrap();
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

    fn parity_knowledge() -> MockKnowledge {
        MockKnowledge::new()
            .with_entity("姚明", &["身高", "国籍", "出生地"])
            .with_entity("叶莉", &["身高", "国籍"])
            .with_answer("姚明是谁", "姚明是中国篮球运动员。")
            .with_answer("叶莉是谁", "叶莉是中国篮球运动员。")
            .with_answer("姚明的身高是什么", "姚明的身高是 226 厘米。")
            .with_answer("姚明的国籍是什么", "姚明的国籍是中国。")
            .with_answer("叶莉的身高是什么", "叶莉的身高是 190 厘米。")
    }

    const PARITY_TRANSCRIPT: [&str; 15] = [
        "你怎么知道的？",
        "谢谢！",
        "你好",
        "姚明是谁",
        "他的身高呢",
        "你怎么知道的？",
        "我们聊了什么？",
        "还有呢",
        "详细一点",
        "姚明的体重是多少",
        "叶莉是谁",
        "叶莉的身高是多少",
        "身高呢",
        "谢谢！",
        "再见",
    ];

    fn replay_texts(mode: PromptMode) -> Vec<String> {
        let knowledge = parity_knowledge();
        let mut manager = DialogueManager::new();
        manager.set_prompt_mode(mode);
        PARITY_TRANSCRIPT
            .iter()
            .map(|input| {
                manager
                    .plan_pure_dialogue(input, &knowledge)
                    .or_else(|| manager.plan_reply(input, &knowledge))
                    .unwrap_or_else(|| panic!("输入「{}」应被接管", input))
                    .text
            })
            .collect()
    }

    #[test]
    fn test_prompt_specs_reproduce_replies_byte_for_byte() {
        let knowledge = parity_knowledge();
        let mut manager = DialogueManager::new();
        manager.set_prompt_mode(PromptMode::Parity);
        for input in PARITY_TRANSCRIPT {
            manager
                .plan_pure_dialogue(input, &knowledge)
                .or_else(|| manager.plan_reply(input, &knowledge))
                .unwrap_or_else(|| panic!("输入「{}」应被接管", input));
        }
        let stats = manager.prompt_stats();
        assert!(stats.checked >= 10, "覆盖不足：{}", stats.render_text());
        assert_eq!(
            stats.parity_mismatches, 0,
            "模板与组合生成器文本存在差异：{:#?}",
            stats.mismatch_examples
        );
        assert_eq!(stats.ungrounded, 0, "缺槽位：{:?}", stats.missing_slots);
        assert_eq!(
            stats.composed, 3,
            "recap/continue/elaborate 应保持组合生成器"
        );
    }

    #[test]
    fn test_prompt_on_mode_is_byte_identical_on_seed_transcript() {
        let off = replay_texts(PromptMode::Off);
        let on = replay_texts(PromptMode::On);
        let parity = replay_texts(PromptMode::Parity);
        assert_eq!(off, on, "On 模式应逐字等价于 Off 模式渲染");
        assert_eq!(off, parity, "Parity 模式对外输出应与 Off 模式一致");
    }

    #[test]
    fn test_prompt_stats_are_observable() {
        let knowledge = parity_knowledge();
        let mut manager = DialogueManager::new();
        manager.set_prompt_mode(PromptMode::On);
        manager.plan_reply("姚明的身高是多少", &knowledge).unwrap();
        let stats = manager.prompt_stats();
        assert_eq!(stats.grounded, 1);
        assert_eq!(stats.ungrounded, 0);
        assert!((stats.grounding_rate() - 1.0).abs() < 1e-12);
        assert!(manager.prompt_report().contains("提示资产 v1"));
        manager.reset_prompt_stats();
        assert_eq!(manager.prompt_stats().attempted(), 0);
    }

    #[test]
    fn test_query_board_drives_multi_turn_edit_and_snapshot() {
        let knowledge = MockKnowledge::new()
            .with_entity("姚明", &["身高"])
            .with_value("姚明", "身高", "226")
            .with_entity("王治郅", &["身高"])
            .with_value("王治郅", "身高", "214")
            .with_entity("易建联", &["身高"])
            .with_value("易建联", "身高", "213");
        let mut manager = DialogueManager::new();
        let reply = manager
            .plan_reply("列出所有人的身高", &knowledge)
            .expect("查询板应接管集合查询");
        assert!(reply.text.contains("姚明"));
        assert!(manager.board().is_active());

        let selected = manager
            .plan_reply("第二个呢", &knowledge)
            .expect("序号选择应留在查询板内");
        assert!(selected.text.contains("当前选中"));
        assert_eq!(manager.turns().last().unwrap().plan, vec!["board:select"]);

        let snapshot = manager.snapshot();
        let mut restored = DialogueManager::new();
        restored.restore(&snapshot);
        assert!(restored.board().is_active());
        assert_eq!(
            restored.board().frame_count(),
            manager.board().frame_count()
        );
        assert_eq!(restored.turn_count(), manager.turn_count());

        let sorted = restored
            .plan_reply("按身高从低到高排", &knowledge)
            .expect("排序应留在查询板内");
        assert!(sorted.text.contains("易建联"));
        let undone = restored
            .plan_pure_dialogue("回到上一步", &knowledge)
            .expect("撤销应回到上一帧");
        assert!(undone.text.contains("已回到第"));
    }

    #[test]
    fn test_trace_mode_off_is_byte_compatible_and_records_nothing() {
        let knowledge = mock();
        let mut manager = DialogueManager::new();
        manager.plan_reply("姚明的身高是多少", &knowledge).unwrap();
        assert_eq!(manager.trace_mode(), TraceMode::Off);
        assert!(manager.trace_store().is_empty());
        // 默认 Off 的省略问句仍读作身份问句（既有契约）。
        let frame = manager.analyze("那姚明呢", &knowledge);
        assert_eq!(frame.kind, MoveKind::AskIdentity);
        assert!(!frame.inherited);
    }

    #[test]
    fn test_trace_on_inherits_predicate_focus_for_ellipsis() {
        let knowledge = MockKnowledge::new()
            .with_entity("姚明", &["身高", "国籍"])
            .with_answer("姚明的身高是什么", "姚明的身高是 226 厘米。")
            .with_entity("叶莉", &["身高"])
            .with_answer("叶莉的身高是什么", "叶莉的身高是 190 厘米。");
        let mut manager = DialogueManager::new();
        manager.set_trace_mode(TraceMode::On);
        manager.plan_reply("姚明的身高是多少", &knowledge).unwrap();
        // 「叶莉呢」在属性问据下读作「叶莉的身高呢」。
        let frame = manager.analyze("叶莉呢", &knowledge);
        assert_eq!(frame.kind, MoveKind::AskAttribute, "应承继谓词焦点");
        assert_eq!(frame.entity.as_deref(), Some("叶莉"));
        assert_eq!(frame.attribute.as_deref(), Some("身高"));
        assert!(!frame.inherited, "实体显式出现");
        let reply = manager.plan_reply("叶莉呢", &knowledge).unwrap();
        assert!(reply.text.contains("190"), "{}", reply.text);
        assert_eq!(manager.trace_store().predicate_focus(), Some("身高"));
    }

    #[test]
    fn test_trace_on_pronoun_inherits_predicate_focus() {
        let knowledge = mock();
        let mut manager = DialogueManager::new();
        manager.set_trace_mode(TraceMode::On);
        manager.plan_reply("姚明的身高是多少", &knowledge).unwrap();
        let frame = manager.analyze("他呢", &knowledge);
        assert_eq!(frame.kind, MoveKind::AskAttribute);
        assert_eq!(frame.entity.as_deref(), Some("姚明"));
        assert_eq!(frame.attribute.as_deref(), Some("身高"));
        assert!(frame.inherited);
    }

    #[test]
    fn test_trace_on_correction_overrides_value_and_marks_stale() {
        let knowledge = mock();
        let mut manager = DialogueManager::new();
        manager.set_trace_mode(TraceMode::On);
        let first = manager.plan_reply("姚明的身高是多少", &knowledge).unwrap();
        assert!(first.text.contains("226"));
        let correction = manager
            .plan_reply("不对，姚明的身高是 190", &knowledge)
            .expect("改口应被拦截");
        assert!(correction.text.contains("190"), "{}", correction.text);
        assert!(correction.text.contains("作废"), "{}", correction.text);
        // 会话内覆盖值优先于图谱旧值。
        let again = manager.plan_reply("姚明的身高是多少", &knowledge).unwrap();
        assert!(again.text.contains("190"), "{}", again.text);
        assert!(!again.text.contains("226"), "{}", again.text);
        assert!(manager.trace_report().contains("更正"));
        // 更正与改口回复必须落在同一轮迹（不留空轮、不拆成两轮）。
        let correction_turn = manager
            .trace_store()
            .turns()
            .find(|turn| !turn.corrections.is_empty())
            .expect("应有更正轮迹");
        assert!(
            correction_turn
                .nodes
                .iter()
                .any(|node| node.kind == "reply"),
            "更正轮应同时含回复节点：{:?}",
            correction_turn
                .nodes
                .iter()
                .map(|n| &n.kind)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_trace_snapshot_roundtrip_preserves_focus_and_overrides() {
        let knowledge = mock();
        let mut manager = DialogueManager::new();
        manager.set_trace_mode(TraceMode::On);
        manager.plan_reply("姚明的身高是多少", &knowledge).unwrap();
        manager
            .plan_reply("不对，姚明的身高是 190", &knowledge)
            .unwrap();
        let snapshot = manager.snapshot();
        let mut restored = DialogueManager::new();
        restored.set_trace_mode(TraceMode::On);
        restored.restore(&snapshot);
        assert_eq!(restored.trace_store().predicate_focus(), Some("身高"));
        assert_eq!(
            restored.trace_store().value_override("姚明", "身高"),
            Some("190")
        );
        assert_eq!(
            restored.trace_store().turn_count(),
            manager.trace_store().turn_count()
        );
    }

    #[test]
    fn open_turn_is_recorded_and_survives_snapshot() {
        let mut manager = DialogueManager::new();
        manager.begin_turn();
        let fault = crate::turn_runtime::TurnFault::new(
            crate::turn_runtime::TurnFaultKind::Deadline,
            crate::turn_runtime::TurnPhase::Reason,
            "超出预算",
            12,
        );
        manager.record_open_turn("姚明的身高是多少", 1, &fault, Some("姚明"), Some("身高"));
        let open = manager.latest_open_turn().expect("应记录未完成轮次");
        assert_eq!(open.input, "姚明的身高是多少");
        assert!(manager.summary().contains("未完成轮次"));

        let snapshot = manager.snapshot();
        let mut restored = DialogueManager::new();
        restored.restore(&snapshot);
        assert_eq!(
            restored.latest_open_turn().map(|open| open.fault),
            Some(crate::turn_runtime::TurnFaultKind::Deadline)
        );

        let taken = restored.take_open_turn().expect("应可取走");
        assert_eq!(taken.attempts, 1);
        assert!(restored.latest_open_turn().is_none());
    }

    #[test]
    fn open_turns_expire_after_ttl_ticks() {
        let mut manager = DialogueManager::new();
        manager.begin_turn();
        let fault = crate::turn_runtime::TurnFault::new(
            crate::turn_runtime::TurnFaultKind::Store,
            crate::turn_runtime::TurnPhase::Recall,
            "",
            1,
        );
        manager.record_open_turn("姚明是谁", 1, &fault, None, None);
        for _ in 0..crate::turn_runtime::OPEN_TURN_TTL_TURNS {
            manager.begin_turn();
        }
        assert!(manager.latest_open_turn().is_none());
    }

    #[test]
    fn frame_program_survives_snapshot_and_inherits_entity() {
        let source = r#"
enabled = true

[[frames]]
id = "person.attribute"
patterns = ["{entity}的{attribute}是什么", "{attribute}呢"]
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

[[frames.slots]]
name = "value"
sort = "text"

[frames.goal]
kind = "attribute"
entity = "entity"
attribute = "attribute"
output = "value"
"#;
        let config: crate::binding_frames::DialogueFramesConfig =
            toml::from_str(source).expect("帧配置应可解析");
        let program = Arc::new(FrameProgram::compile(&config).expect("帧配置应可编译"));
        let knowledge = MockKnowledge::new()
            .with_entity("姚明", &["身高", "体重"])
            .with_value("姚明", "身高", "226")
            .with_value("姚明", "体重", "140");
        let mut manager = DialogueManager::new();
        manager.set_frame_program(Some(program.clone()));
        let reply = manager
            .plan_reply("姚明的身高是什么", &knowledge)
            .expect("完整绑定应直接回答");
        assert!(reply.text.contains("226"), "{}", reply.text);

        let snapshot = manager.snapshot();
        assert!(snapshot.frames.is_some());
        let mut restored = DialogueManager::new();
        restored.restore(&snapshot);
        restored.set_frame_program(Some(program));
        let reply = restored
            .plan_reply("体重呢", &knowledge)
            .expect("恢复后应继承实体");
        assert!(reply.text.contains("140"), "{}", reply.text);
        assert!(restored.frame_report().contains("回答 2 次"));
        assert!(restored.summary().contains("绑定帧记忆"));
    }
}

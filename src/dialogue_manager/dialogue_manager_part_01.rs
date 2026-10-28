use std::sync::Arc;
use serde::{Deserialize, Serialize};
use crate::binding_frames::{FrameMemory, FrameProgram};
use crate::bindings::BindingSet;
use crate::context_window::{select_window, ContextCue, WindowBudget, WindowSelection};
use crate::contextual_reference::{
    parse_clarify_act, rewrite_at, BindingLedger, BindingScope, ClarifyAct, ClarifyRequest,
    ContextualResolver, SenseSource,
};
use crate::derivation::Derivation;
use crate::entity_linker::EntityLinker;
use crate::epistemic::{EpistemicStatus, LedgerLoad};
use crate::identity_dialogue::{
    self, IdentityAct, IdentityCard, IdentityStatus, UserIdentityLedger,
};
use crate::logic_program::{LAtom, LTerm, LogicProgram};
use crate::probabilistic_query_intent::{EpistemicMode, EpistemicTarget, ProbabilisticQueryIntent};
use crate::prompt_spec::{self, PromptMode, PromptStats, RenderOutcome, PROMPT_SPEC_VERSION};
use crate::quality::{ClaimWarrant, QualityIndex};
use crate::query_board::QueryBoard;
use crate::query_planner::{QueryPlan, QueryPlanner};
use crate::recovery::{self, PendingKind, RecoveryMode, RecoveryRequest, RecoveryState};
use crate::recovery::{RecoveryNomination, ReframedReading, VerificationOutcome};
use crate::symbolic_reasoner::SymbolicReasoner;
use crate::trace_graph::{NodeId, QudKind, ReferentSource, TraceMode, TraceStore, TurnTrace};
use crate::turn_runtime::{OpenTurn, OpenTurnBook, TurnFault};
use crate::working_memory::{
    AttentionBudget, AttentionItem, AttentionKind, AttentionQuery, ContextPacket, WorkingMemory,
    WORKING_MEMORY_CAPACITY,
};
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
    /// 语境指称绑定：空态省略，未发生澄清的会话字节不变。
    #[serde(default, skip_serializing_if = "BindingLedger::is_empty")]
    pub references: BindingLedger,
    /// 用户身份台账：只存用户明确自述的名字；空态省略，旧快照字节不变。
    #[serde(default, skip_serializing_if = "UserIdentityLedger::is_empty")]
    pub identity: UserIdentityLedger,
    /// 上一轮情态问句的算子与槽位：让「那王五呢」沿用「可能/最可能」，
    /// 而不是退化成一次普通身份追问。空态省略，旧快照字节不变。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub epistemic_followup: Option<EpistemicFollowup>,
    /// 类型对话记忆：焦点类型、成员游标与已说槽位（空态省略）。
    #[serde(default, skip_serializing_if = "crate::type_memory::TypeDialogueMemory::is_empty")]
    pub type_memory: crate::type_memory::TypeDialogueMemory,
}
/// 管理器的生成结果：文本 + 命中的策略 + 语用类型。
#[derive(Clone, Debug)]
pub struct DialogueReply {
    pub text: String,
    pub plan: Vec<String>,
    pub kind: MoveKind,
    /// 元解释轮附带的有界推导（普通回答为 None，不进入会话快照）。
    pub derivation: Option<Derivation>,
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
/// 语境指称消歧的只读视图：链接器候选 + 义项来源。
///
/// 只有实现方显式提供时，[`DialogueManager`] 才会走「候选 → 裁决 → 澄清 →
/// 绑定」的链接栈；缺省（`None`）保持纯字面 `entities_in` 行为。
pub struct ReferenceView<'a> {
    pub linker: &'a EntityLinker,
    pub source: &'a dyn SenseSource,
}
/// 一轮语境链接的落点。
enum ReferenceDecision {
    /// 候选分裂：返回澄清问题文本与指称词面。
    Clarify(String, String),
    /// 已裁决/已绑定：用规范实体改写后的输入继续下游。
    Rewrite(String),
    /// 本轮跳过链接（用户拒绝澄清或无需改写）。
    Skip,
}
/// 管理器向知识源提出的接口（由 `SymbolicReasoner` 或测试替身实现）。
pub trait DialogueKnowledge {
    /// 文本中出现的已知实体（规范名，按长度降序更佳）。
    fn entities_in(&self, text: &str) -> Vec<String>;
    /// 实体是否在本体/图谱中登记。
    fn known_entity(&self, entity: &str) -> bool;
    /// 对规范化问句作答（可包含链接预测等推测性内容）。
    fn answer(&self, question: &str) -> Option<String>;

    /// 只返回可确证的回答；推测内容留给恢复阶梯以「候选 + 确认」呈现。
    /// 缺省与 [`Self::answer`] 相同，保持既有知识源行为不变。
    fn answer_certain(&self, question: &str) -> Option<String> {
        self.answer(question)
    }
    /// 已采纳的习得重写直接给出的回答；缺省不接管。
    fn learned_rewrite_answer(&self, question: &str) -> Option<String> {
        let _ = question;
        None
    }
    /// 类型画像/成员/对比问句（缺省不接管，保持旧知识源行为）。
    fn type_memory_answer(&self, question: &str) -> Option<String> {
        let _ = question;
        None
    }
    /// 带会话焦点的类型问答；缺省回落到无状态实现。
    fn type_memory_answer_in_context(
        &self,
        question: &str,
        memory: &mut crate::type_memory::TypeDialogueMemory,
    ) -> Option<crate::type_memory::TypeAnswer> {
        let _ = memory;
        self.type_memory_answer(question)
            .map(|text| crate::type_memory::TypeAnswer { text, focus: None })
    }
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

    /// 语境指称消歧视图（链接器 + 义项卡）。缺省不启用 CRD。
    fn reference_view(&self) -> Option<ReferenceView<'_>> {
        None
    }

    /// 问题谓词/类型线索（供 CRD 表决）。缺省无线索。
    fn reference_hints(&self, text: &str) -> Vec<String> {
        let _ = text;
        Vec::new()
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

    fn reference_view(&self) -> Option<ReferenceView<'_>> {
        Some(ReferenceView {
            linker: self.linker(),
            source: self,
        })
    }

    fn reference_hints(&self, text: &str) -> Vec<String> {
        SymbolicReasoner::reference_hints(self, text)
    }

    fn known_entity(&self, entity: &str) -> bool {
        self.generator().is_known_entity(entity)
    }

    fn answer(&self, question: &str) -> Option<String> {
        SymbolicReasoner::answer(self, question)
    }

    fn answer_certain(&self, question: &str) -> Option<String> {
        SymbolicReasoner::answer_certain(self, question)
    }

    fn learned_rewrite_answer(&self, question: &str) -> Option<String> {
        SymbolicReasoner::answer_learned_rewrite(self, question)
    }

    fn type_memory_answer(&self, question: &str) -> Option<String> {
        crate::type_memory::answer(self.generator(), question).map(|answer| answer.text)
    }

    fn type_memory_answer_in_context(
        &self,
        question: &str,
        memory: &mut crate::type_memory::TypeDialogueMemory,
    ) -> Option<crate::type_memory::TypeAnswer> {
        crate::type_memory::answer_in_context(
            self.generator(),
            Some(self.schema_view()),
            memory,
            question,
        )
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
/// 多轮情态承接的记忆：只保存重建问句所需的算子与槽位，不保存答案。
///
/// 这是对话状态而非知识状态：它绝不能参与事实概率计算，也不能替代
/// `ProbabilisticQueryIntent` 对当前轮的理解。它的唯一用途是当用户说
/// 「那王五呢」时，把上一轮的情态算子与谓词槽位重新施加到新实体上。
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EpistemicFollowup {
    /// [`EpistemicMode::code`] 的字符串形式（快照可读、可审计）。
    pub mode: String,
    /// 基原子问句的固定主语；ranking 问句的主语槽位。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subject: Option<String>,
    #[serde(default)]
    pub predicate: String,
    /// 基原子问句的固定宾语；`SubjectsOf` 的宾语槽位。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub object: Option<String>,
    #[serde(default)]
    pub negated: bool,
    /// true：候选排序（`ObjectsOf` / `SubjectsOf`）；false：基原子。
    #[serde(default)]
    pub ranking: bool,
}
impl EpistemicFollowup {
    fn from_intent(intent: &ProbabilisticQueryIntent) -> Option<Self> {
        let mode = intent.mode.code().to_string();
        match &intent.target {
            Some(EpistemicTarget::GroundClaim(claim)) => Some(Self {
                mode,
                subject: Some(claim.subject.clone()),
                predicate: claim.predicate.clone(),
                object: Some(claim.object.clone()),
                negated: claim.negated,
                ranking: false,
            }),
            Some(EpistemicTarget::ObjectsOf { subject, predicate }) => Some(Self {
                mode,
                subject: Some(subject.clone()),
                predicate: predicate.clone(),
                object: None,
                negated: false,
                ranking: true,
            }),
            Some(EpistemicTarget::SubjectsOf { object, predicate }) => Some(Self {
                mode,
                subject: None,
                predicate: predicate.clone(),
                object: Some(object.clone()),
                negated: false,
                ranking: true,
            }),
            // 比较问需要同时替换两个具名候选，单实体省略无法无歧义补全。
            Some(EpistemicTarget::Comparison { .. }) => None,
            None => Self::from_core(&intent.core_question, mode),
        }
    }

    /// 情态层只抽出了算子、没抽出目标时，用与推理器相同的 QueryPlanner
    /// 读指称计划；只接受单跳目标，多跳链留给普通管线。
    fn from_core(question: &str, mode: String) -> Option<Self> {
        let plan = QueryPlanner::new().parse(question)?;
        match plan {
            QueryPlan::FindObject { segments } if segments.len() == 2 => Some(Self {
                mode,
                subject: Some(segments[0].clone()),
                predicate: segments[1].clone(),
                object: None,
                negated: false,
                ranking: true,
            }),
            QueryPlan::FindObjectByPredicate { subject, predicate } => Some(Self {
                mode,
                subject: Some(subject),
                predicate,
                object: None,
                negated: false,
                ranking: true,
            }),
            QueryPlan::FindSubject { object, predicate } => Some(Self {
                mode,
                subject: None,
                predicate,
                object: Some(object),
                negated: false,
                ranking: true,
            }),
            QueryPlan::CheckTriple {
                subject_segments,
                object,
            } if subject_segments.len() == 2 => Some(Self {
                mode,
                subject: Some(subject_segments[0].clone()),
                predicate: subject_segments[1].clone(),
                object: Some(object),
                negated: false,
                ranking: false,
            }),
            QueryPlan::CheckRelation {
                subject,
                predicate,
                object,
                negated,
            } => Some(Self {
                mode,
                subject: Some(subject),
                predicate,
                object: Some(object),
                negated,
                ranking: false,
            }),
            _ => None,
        }
    }

    /// 把新实体代回上一轮槽位，重建一条完整情态问句。
    fn render_followup(&self, entity: &str) -> Option<String> {
        let mode = EpistemicMode::from_code(&self.mode)?;
        let predicate = self.predicate.trim();
        if predicate.is_empty() {
            return None;
        }
        if self.ranking {
            if self.subject.is_some() {
                return match mode {
                    EpistemicMode::Possibility => {
                        Some(format!("{}的{}可能是谁", entity, predicate))
                    }
                    EpistemicMode::MostLikely => {
                        Some(format!("谁最可能是{}的{}", entity, predicate))
                    }
                    _ => None,
                };
            }
            if self.object.is_some() {
                return match mode {
                    EpistemicMode::Possibility => {
                        Some(format!("{}可能是谁的{}", entity, predicate))
                    }
                    EpistemicMode::MostLikely => {
                        Some(format!("{}最可能是谁的{}", entity, predicate))
                    }
                    _ => None,
                };
            }
            return None;
        }
        let subject = self.subject.as_deref()?;
        let negated = self.negated;
        match mode {
            EpistemicMode::Probability => Some(format!(
                "{}有多大的可能{}是{}的{}",
                subject,
                if negated { "不" } else { "" },
                entity,
                predicate
            )),
            EpistemicMode::Certainty => Some(format!(
                "{}{}{}的{}吗",
                subject,
                if negated { "一定不是" } else { "一定是" },
                entity,
                predicate
            )),
            EpistemicMode::Possibility => Some(format!(
                "{}{}{}的{}吗",
                subject,
                if negated { "不可能" } else { "可能" },
                entity,
                predicate
            )),
            EpistemicMode::MostLikely => Some(format!(
                "{}{}{}的{}吗",
                subject,
                if negated { "最不可能" } else { "最可能" },
                entity,
                predicate
            )),
            _ => None,
        }
    }
}
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
    /// 语境指称绑定账本（澄清回答 → 后续同词面直接复用）。
    references: BindingLedger,
    /// 用户身份台账（随快照持久化；不写入图谱）。
    identity: UserIdentityLedger,
    /// 上一轮情态问句的算子与槽位（随快照持久化）。
    epistemic_followup: Option<EpistemicFollowup>,
    /// 类型对话记忆：焦点类型、成员游标与已说槽位（随快照持久化）。
    type_memory: crate::type_memory::TypeDialogueMemory,
    /// 最近一次身份卡（运行时只读视图，不进快照）。
    last_identity_card: Option<IdentityCard>,
    /// 待回答的歧义澄清请求（不持久化；恢复会话后需重新提问）。
    pending_reference: Option<ClarifyRequest>,
    /// 无状态消歧器（状态都在 `references`）。
    reference_resolver: ContextualResolver,
}
impl Default for DialogueManager {
    fn default() -> Self {
        Self::new()
    }
}

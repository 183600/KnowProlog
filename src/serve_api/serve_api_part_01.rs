use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, oneshot, Mutex};
use crate::abduction::{clarification_batch_scope, classify_answer, AnswerKind};
use crate::binding_frames::FrameProgram;
use crate::cache::RevisionLru;
use crate::canonical::{self, DecodeError, DecodeReceipt};
use crate::dialogue_manager::{
    DialogueKnowledge, DialogueManager, DialogueManagerSnapshot, MoveKind,
};
use crate::event_graph::{EventDialogue, EventGraph};
use crate::knowledge_delta::{KnowledgeBase, KnowledgeCatchup, KnowledgeChange};
use crate::ontology_reasoning::OntologyProgram;
use crate::perf_ledger::PerfLedger;
use crate::plugin::{PluginOutcome, PluginRuntime};
use crate::prolog_engine::PrologFact;
use crate::prompt_spec::{self, PromptMode};
use crate::quality::QualityIndex;
use crate::reading::{Reading, Residual};
use crate::recovery::RecoveryMode;
use crate::schema_view::SchemaView;
use crate::shared_memory::{
    MemoryScope, NewClaim, PublishError, SharedMemoryBoard, SharedMemoryView,
    DEFAULT_MAX_CLAIMS,
};
use crate::startup_ledger::StartupLedger;
use crate::symbolic_reasoner::{DialogueSnapshot, SymbolicReasoner};
use crate::turn_runtime::{
    CancelToken, OpenTurn, TurnBudget, TurnControl, TurnFault, TurnFaultKind, TurnPhase,
};
use crate::working_memory::{
    AttentionBudget, AttentionItem, AttentionKind, WORKING_MEMORY_CAPACITY,
};
use crate::{NlpConfig, ReasoningConfig};
const MAX_HEADER_BYTES: usize = 16 * 1024;
const DEFAULT_MAX_BODY_BYTES: usize = 256 * 1024;
/// 冷存储信封的体量上限（解码绝不因巨量输入拖垮进程）。
const SESSION_ENVELOPE_MAX_BYTES: usize = 16 * 1024 * 1024;
const CONNECTION_TIMEOUT: Duration = Duration::from_secs(30);
/// 等待会话 actor 返回的硬上限；超时先触发协作取消，再报 504。
const TURN_WAIT_TIMEOUT: Duration = Duration::from_secs(90);
const HISTORY_LIMIT: usize = 64;
/// 无状态问答缓存容量（条目数）；纯函数结果，命中与重算必须逐字节一致。
const STATELESS_CACHE_CAPACITY: usize = 64;
/// OpenAI 兼容层报告的模型名（纯符号推理，非神经网络）。
pub const DEFAULT_MODEL: &str = "knowprolog-symbolic";
/// 缓存纪元：失效必须由显式版本驱动，绝不用 TTL / 挂钟。
///
/// - `kb_revision`：知识库版本（写路径推进；无写路径时等于预热基座版本）；
/// - `identity_epoch`：身份准入纪元（新实体/别名准入时推进）。
///
/// 任何一元推进都会让全部缓存条目不可命中（[`crate::cache::RevisionLru`]）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheEpoch {
    pub kb_revision: u64,
    pub identity_epoch: u64,
}
/// 插件本体并入统一推理器的回执：公理、别名、逆向词面的新增条数。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PluginOntologyReceipt {
    pub lessons: usize,
    pub aliases: usize,
    pub inverse_surfaces: usize,
    pub rules: usize,
}
impl PluginOntologyReceipt {
    pub fn is_empty(self) -> bool {
        self.lessons == 0 && self.aliases == 0 && self.inverse_surfaces == 0 && self.rules == 0
    }
}
/// 无状态问答缓存键：按序的全部用户消息（trim 后）+ 生效策略。
///
/// 只做「完全相等」命中，不做相似/近似匹配：键由长度前缀拼接，
/// 任何字符差异都会产生不同键。同义词归约由下游推理器（对齐器）完成，
/// 缓存层不猜测语义等价。
fn stateless_answer_key(inputs: &[String], policy: Option<&SessionPolicy>) -> String {
    let mut key = String::new();
    for input in inputs {
        let trimmed = input.trim();
        key.push_str(&trimmed.len().to_string());
        key.push('#');
        key.push_str(trimmed);
        key.push('\u{1f}');
    }
    key.push_str("\u{1e}");
    if let Some(policy) = policy {
        key.push_str(&serde_json::to_string(policy).unwrap_or_default());
    }
    key
}
/// 回答风格（由 system 指令白名单解析）。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PolicyStyle {
    #[default]
    Normal,
    Concise,
    Detailed,
}
impl PolicyStyle {
    pub fn label(self) -> &'static str {
        match self {
            PolicyStyle::Normal => "normal",
            PolicyStyle::Concise => "concise",
            PolicyStyle::Detailed => "detailed",
        }
    }
}
/// 会话级策略：从 OpenAI 兼容请求的 `system` / `developer` 消息确定性解析，
/// 只认白名单短语，未识别的子句进 `warnings` 原样回显（不猜、不静默丢弃）。
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SessionPolicy {
    #[serde(default)]
    pub style: PolicyStyle,
    /// 每个回答追加「依据：…」，内容来自真实策略轨迹（非生成文本）。
    #[serde(default)]
    pub require_evidence: bool,
    /// 让位轮给出确定性澄清问句（status = needs_clarification）。
    #[serde(default)]
    pub strict_clarify: bool,
    /// 命中的规则标签（按出现顺序去重）。
    #[serde(default)]
    pub recognized: Vec<String>,
    /// 未被识别的子句（截断回显，提示调用方表述更明确）。
    #[serde(default)]
    pub warnings: Vec<String>,
}
const STYLE_CONCISE_PHRASES: &[&str] = &[
    "简短",
    "简洁",
    "一句话",
    "尽量短",
    "concise",
    "brief",
    "short",
];
const STYLE_DETAILED_PHRASES: &[&str] = &[
    "详细",
    "展开",
    "深入",
    "in depth",
    "detailed",
    "elaborate",
    "verbose",
];
const EVIDENCE_PHRASES: &[&str] = &[
    "依据",
    "来源",
    "出处",
    "证据",
    "引用",
    "evidence",
    "cite",
    "citation",
    "source",
    "provenance",
    "justify",
];
const STRICT_CLARIFY_PHRASES: &[&str] = &[
    "澄清",
    "追问",
    "不要猜",
    "不要臆测",
    "先确认",
    "clarify",
    "ask before",
    "do not guess",
    "don't guess",
    "no guessing",
    "confirm",
];
impl SessionPolicy {
    /// 从 `system` / `developer` 消息文本解析；空输入返回默认策略。
    pub fn from_system_texts(texts: &[String]) -> SessionPolicy {
        let mut policy = SessionPolicy::default();
        for text in texts {
            for clause in text.split(['。', '.', '；', ';', '\n', '！', '!', '？', '?']) {
                let clause = clause.trim();
                if clause.is_empty() {
                    continue;
                }
                let lowered = clause.to_lowercase();
                let mut matched = false;
                if STYLE_CONCISE_PHRASES.iter().any(|p| lowered.contains(p)) {
                    policy.style = PolicyStyle::Concise;
                    Self::record(&mut policy.recognized, "style:concise");
                    matched = true;
                }
                if STYLE_DETAILED_PHRASES.iter().any(|p| lowered.contains(p)) {
                    policy.style = PolicyStyle::Detailed;
                    Self::record(&mut policy.recognized, "style:detailed");
                    matched = true;
                }
                if EVIDENCE_PHRASES.iter().any(|p| lowered.contains(p)) {
                    policy.require_evidence = true;
                    Self::record(&mut policy.recognized, "require_evidence");
                    matched = true;
                }
                if STRICT_CLARIFY_PHRASES.iter().any(|p| lowered.contains(p)) {
                    policy.strict_clarify = true;
                    Self::record(&mut policy.recognized, "strict_clarify");
                    matched = true;
                }
                if !matched {
                    policy.warnings.push(truncate_chars(clause, 32));
                }
            }
        }
        policy
    }

    fn record(recognized: &mut Vec<String>, label: &str) {
        if !recognized.iter().any(|existing| existing == label) {
            recognized.push(label.to_string());
        }
    }

    pub fn is_default(&self) -> bool {
        self.style == PolicyStyle::Normal
            && !self.require_evidence
            && !self.strict_clarify
            && self.recognized.is_empty()
            && self.warnings.is_empty()
    }

    pub fn render_text(&self) -> String {
        if self.is_default() {
            return "默认策略（风格 normal）".to_string();
        }
        format!(
            "风格 {}·依据 {}·严格澄清 {}·命中 [{}]",
            self.style.label(),
            if self.require_evidence { "开" } else { "关" },
            if self.strict_clarify { "开" } else { "关" },
            self.recognized.join(", ")
        )
    }

    /// 把策略作用到一轮结果上：风格 → 依据 → 严格澄清。
    fn apply(&self, input: &str, turn: &mut TurnResponse, knowledge: &dyn DialogueKnowledge) {
        if self.strict_clarify && turn.status == "deferred" {
            turn.status = "needs_clarification".to_string();
            turn.source = "policy".to_string();
            turn.plan = vec!["strict_clarify".to_string()];
            turn.reply = format!(
                "我还不能确定如何回答「{}」，请补充实体名称或换一种问法。",
                truncate_chars(input.trim(), 24)
            );
        }
        if !turn.reply.is_empty() && turn.status != "needs_clarification" {
            match self.style {
                PolicyStyle::Concise => turn.reply = first_sentence(&turn.reply),
                PolicyStyle::Detailed => {
                    let already_offers =
                        turn.reply.contains("还可以") || turn.reply.contains("可继续");
                    if !already_offers {
                        if let Some(focus) = &turn.focus {
                            let attributes = knowledge.known_attributes(focus);
                            if !attributes.is_empty() {
                                let list = attributes
                                    .iter()
                                    .take(4)
                                    .cloned()
                                    .collect::<Vec<_>>()
                                    .join("、");
                                turn.reply.push_str(&format!("\n（可继续了解：{}）", list));
                            }
                        }
                    }
                }
                PolicyStyle::Normal => {}
            }
        }
        if self.require_evidence && !turn.reply.is_empty() {
            let evidence = match turn.source.as_str() {
                "dialogue" => match &turn.focus {
                    Some(focus) => format!(
                        "\n依据：对话策略「{}」· 焦点「{}」",
                        turn.plan.join("→"),
                        focus
                    ),
                    None => format!("\n依据：对话策略「{}」", turn.plan.join("→")),
                },
                "reasoner" => "\n依据：知识图谱符号推理".to_string(),
                "policy" => "\n依据：服务策略 strict_clarify（未猜测）".to_string(),
                _ => return,
            };
            turn.reply.push_str(&evidence);
        }
    }
}
/// 截断到前 `limit` 个字符（按字符边界，尾部加省略号）。
fn truncate_chars(text: &str, limit: usize) -> String {
    let mut out: String = text.chars().take(limit).collect();
    if text.chars().count() > limit {
        out.push('…');
    }
    out
}
/// 取出首条实质句（跳过空行与以「：」结尾的标题行，止于 。！？或换行）。
fn first_sentence(text: &str) -> String {
    let substantive = text
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty() && !line.ends_with('：') && !line.ends_with(':'))
        .unwrap_or("");
    if substantive.is_empty() {
        return text.trim().to_string();
    }
    let mut out = String::new();
    for ch in substantive.chars() {
        out.push(ch);
        if matches!(ch, '。' | '！' | '？') {
            break;
        }
    }
    out
}
/// 一轮对话的服务级结果。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TurnResponse {
    pub session_id: String,
    pub turn: u32,
    /// `answered` = 本轮给出了回答；`deferred` = 让位（无回答）。
    pub status: String,
    /// `dialogue` = 对话管理器接管；`reasoner` = 推理器直接作答；`none` = 让位。
    pub source: String,
    /// 语用类型（[`crate::dialogue_manager::MoveKind::label`]）；让位时为 None。
    pub kind: Option<String>,
    /// 命中的对话策略（按优先级）。
    pub plan: Vec<String>,
    pub reply: String,
    pub focus: Option<String>,
    /// 生效的会话级策略（默认策略时省略）。
    #[serde(default, skip_serializing_if = "SessionPolicy::is_default")]
    pub policy: SessionPolicy,
    /// 本轮的认识论定性（仅恢复模式开启且本轮发生失败/恢复动作时出现）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub epistemic: Option<EpistemicView>,
    /// 本轮的结构化读法诊断：区分「没这个词」「没这个实体」「没这条事实」。
    /// 只在让位/失败轮生成（成功轮不额外分词）；客户端据此决定补词、登记
    /// 实体、补事实还是改写问句，而不是只看到「让位了」。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lexical: Option<LexicalView>,
    /// 接地清单：本轮输入中链接到知识图谱的实体（词面 ↔ URI ↔ 类型 ↔ 角色）。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub entities: Vec<GroundingCitation>,
    /// 交互式查询板状态（空板时省略；客户端可据此继续翻页/筛选/选行）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub board: Option<crate::query_board::BoardView>,
    /// 运行时故障（取消 / 超时 / 存储 / 引擎 / 崩溃 / 快照）；正常轮省略。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fault: Option<TurnFault>,
    /// 未完成轮次（原始输入、失败阶段、尝试次数）；可显式继续。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub open_turn: Option<OpenTurn>,
    /// 本轮是否由继续未完成轮次触发。
    #[serde(default)]
    pub resumed: bool,
    /// 未完成轮次是否仍可继续（TTL 与尝试次数均未耗尽）。
    #[serde(default)]
    pub resumable: bool,
    /// 「为什么」轮次的有界推导视图：事实/规则边、绑定与失败前沿。
    /// 只在元解释轮出现，且与回答一样受节点/深度/字节预算约束。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub derivation: Option<crate::derivation::Derivation>,
}
/// 一条结构化的读法残差：稳定枚举代码 + 局部文本 + 可执行修复动作。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LexicalResidualView {
    /// 稳定类别（`unknown_token` / `unknown_entity` / `unknown_predicate` / ...）。
    pub kind: String,
    pub label: String,
    /// 失败阶段（`tag` / `plan` / `link` / `execute` / ...）。
    pub stage: String,
    /// `info` / `warning` / `blocking`。
    pub severity: String,
    /// 稳定修复动作（`add_word` / `supply_entity` / `supply_predicate` / ...）。
    pub repair: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub surface: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected: Option<String>,
    pub detail: String,
}
/// 一轮的读法状态与有界残差表；回答内容不变，只把「为什么读不懂」外化。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LexicalView {
    /// `ready` / `partial` / `unreadable`。
    pub status: String,
    pub label: String,
    pub blocking: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub residuals: Vec<LexicalResidualView>,
}
impl LexicalView {
    pub fn from_reading(reading: &Reading) -> Self {
        Self {
            status: reading.status().code().to_string(),
            label: reading.status().label().to_string(),
            blocking: reading.blocking_count(),
            residuals: reading.residuals.iter().map(Self::residual).collect(),
        }
    }

    fn residual(residual: &Residual) -> LexicalResidualView {
        LexicalResidualView {
            kind: residual.kind.code().to_string(),
            label: residual.kind.label().to_string(),
            stage: residual.stage.code().to_string(),
            severity: residual.severity.code().to_string(),
            repair: residual.repair.code().to_string(),
            surface: residual.surface.clone(),
            expected: residual.expected.clone(),
            detail: residual.detail.clone(),
        }
    }
}
/// 一条接地引用：回答依据的实体身份（URI 可寻址、词面可回溯、类型可审计）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroundingCitation {
    /// 实体 URI（identity::entity_uri 确定性生成，本地名可无损反解）
    pub uri: String,
    /// 输入中的原词面（可能含错别字，模糊链接时与规范名不同）
    pub surface: String,
    /// 知识图谱登记类型（未登记时省略）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub etype: Option<String>,
    /// `focus` = 本轮对话焦点；`mention` = 普通指称
    pub role: String,
}
/// 一轮回答的认识论视图：状态、档位、回执与开放计数。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EpistemicView {
    /// 状态码（`certain`/`provisional`/`conflict`/`absent`/`unknown`/`unparseable`/`engine_fault`）。
    pub status: String,
    /// 状态中文标签。
    pub label: String,
    /// 恢复阶梯档位（`L0`–`L4`）。
    pub level: String,
    /// 关联回执 id（如有）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub receipt: Option<String>,
    /// 证据链（推测/冲突时出现；逐字可回溯）。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence: Vec<String>,
    /// 挂账追问（如有）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub question: Option<String>,
    /// 会话内未结回执数 + 挂账数（透明度计数）。
    pub open_receipts: usize,
    pub pending_followups: usize,
}
/// 单个会话：对话管理器 + 独立推理器 + 有界历史 + 会话级策略。
///
/// 只在会话 actor 线程内被创建与访问（`SymbolicReasoner` 含 `Rc`，不是 `Send`）。
struct Session {
    id: String,
    turns: u32,
    manager: DialogueManager,
    reasoner: SymbolicReasoner,
    history: Vec<String>,
    policy: SessionPolicy,
    /// 服务级插件运行时（多会话共享；模式在服务启动时确定，HTTP 不提供热切换）。
    plugin: Option<Arc<parking_lot::Mutex<PluginRuntime>>>,
    /// 本轮预算（默认 30 秒 Enforce；测试可注入过期预算）。
    turn_budget: TurnBudget,
    /// 回滚失败后的会话毒化标记：绝不带着半状态继续推理。
    poisoned: Option<String>,
    knowledge: Arc<KnowledgeBase>,
    knowledge_generation: u64,
    perf: Arc<PerfLedger>,
}
/// 一轮开始前的可观察状态：失败轮必须原样回滚，不能出现「回答没给出，焦点却跳走」。
struct SessionCheckpoint {
    turns: u32,
    history: Vec<String>,
    policy: SessionPolicy,
    manager: DialogueManagerSnapshot,
    reasoner: DialogueSnapshot,
}
impl Session {
    /// 插件配方直答：只读既有知识源，模板必须全部槽位接地；取不到值则让位核心管线。
    /// 与 CLI 共用 [`SymbolicReasoner::answer_plugin_recipe`]，回指/方向/依据不另写一份。
    fn plugin_recipe_answer(
        &mut self,
        raw_input: &str,
        outcome: Option<&PluginOutcome>,
    ) -> Option<(String, Vec<String>, String)> {
        let outcome = outcome?;
        let plugin = self.plugin.as_ref()?.clone();
        let focus = self.manager.focus_entity().map(str::to_string);
        let mut runtime = plugin.lock();
        let reply = self
            .reasoner
            .answer_plugin_recipe(outcome, &mut runtime, focus.as_deref())?;
        drop(runtime);
        let plan = reply.plan.clone();
        self.manager.note_plugin_turn(
            raw_input,
            MoveKind::AskAttribute,
            reply.entity.as_deref(),
            reply.attribute.as_deref(),
            &reply.text,
            plan.clone(),
            reply.provenance,
        );
        if let Some(focus) = reply.focus.as_deref() {
            self.manager.note_plugin_focus(focus);
        }
        Some((reply.text, plan, reply.kind.to_string()))
    }

    /// 把对话层的确认/更正同步为推理前提（幂等；不写知识图谱）。
    fn sync_recovery_confirmed_facts(&mut self) -> usize {
        let state = self.manager.recovery_state().clone();
        for fact in &state.confirmed {
            self.reasoner.note_dialogue_verdict(
                &fact.entity,
                &fact.attribute,
                &fact.value,
                true,
                &fact.receipt_id,
            );
        }
        for candidate in &state.refuted {
            self.reasoner.note_dialogue_verdict(
                &candidate.entity,
                &candidate.attribute,
                &candidate.value,
                false,
                &candidate.receipt_id,
            );
        }
        state
            .confirmed
            .iter()
            .filter(|fact| {
                self.reasoner
                    .adopt_session_triple(&fact.entity, &fact.attribute, &fact.value)
            })
            .count()
    }

    fn sync_knowledge(&mut self) {
        let current = self.knowledge.generation();
        if current == self.knowledge_generation {
            return;
        }
        match self.knowledge.changes_since(self.knowledge_generation) {
            KnowledgeCatchup::UpToDate => {
                self.knowledge_generation = current;
            }
            KnowledgeCatchup::Incremental(changes) => {
                let mut facts = 0usize;
                let mut aliases = 0usize;
                for change in &changes {
                    match change {
                        KnowledgeChange::Facts { facts: batch, .. } => {
                            self.reasoner.observe_facts(batch);
                            facts += batch.len();
                        }
                        KnowledgeChange::Aliases { aliases: batch, .. } => {
                            self.reasoner.align_entity_aliases(batch);
                            aliases += batch.len();
                        }
                        KnowledgeChange::Reset { .. } => {
                            let snapshot = self.knowledge.snapshot();
                            self.reasoner.rebuild_from_graph(&snapshot.facts);
                            self.reasoner.align_entity_aliases(&snapshot.aliases);
                        }
                    }
                }
                if let Some(last) = changes.last() {
                    self.knowledge_generation = last.generation();
                }
                self.perf.bump("knowledge_session_appends");
                self.perf.bump_by("knowledge_session_facts", facts as u64);
                self.perf.bump_by("knowledge_session_aliases", aliases as u64);
            }
            KnowledgeCatchup::Rebuild => {
                let snapshot = self.knowledge.snapshot();
                self.reasoner.rebuild_from_graph(&snapshot.facts);
                self.reasoner.align_entity_aliases(&snapshot.aliases);
                self.knowledge_generation = snapshot.generation;
                self.perf.bump("knowledge_session_rebuilds");
            }
        }
    }

    fn engine_stamp(&self) -> EngineStamp {
        let snapshot = self.knowledge.snapshot();
        EngineStamp {
            crate_version: env!("CARGO_PKG_VERSION").to_string(),
            kb_fingerprint: snapshot.fingerprint,
            kb_facts: snapshot.facts.len(),
            kb_generation: snapshot.generation,
            prompt_mode: self.manager.prompt_mode().label().to_string(),
            recovery_mode: self.manager.recovery_mode().label().to_string(),
        }
    }

    fn checkpoint(&self) -> SessionCheckpoint {
        SessionCheckpoint {
            turns: self.turns,
            history: self.history.clone(),
            policy: self.policy.clone(),
            manager: self.manager.snapshot(),
            reasoner: self.reasoner.dialogue_snapshot(),
        }
    }

    fn rollback(&mut self, checkpoint: SessionCheckpoint) -> Result<(), String> {
        self.turns = checkpoint.turns;
        self.history = checkpoint.history;
        self.policy = checkpoint.policy;
        self.manager.restore(&checkpoint.manager);
        self.reasoner
            .restore_dialogue_snapshot(&checkpoint.reasoner)
            .map_err(|error| format!("轮次回滚失败：{}", error))
    }

    fn open_turn_view(&self) -> Option<OpenTurn> {
        self.manager.latest_open_turn().cloned()
    }

    fn open_turn_resumable(&self) -> bool {
        self.open_turn_view()
            .map(|open| open.can_resume(self.manager.open_turns().tick()))
            .unwrap_or(false)
    }

    /// 普通一轮：默认预算 + 独立取消句柄（测试与直接调用路径）。
    #[cfg(test)]
    fn run_turn(&mut self, input: &str) -> TurnResponse {
        self.run_turn_with_cancel(input, CancelToken::new())
    }

    #[cfg(test)]
    fn run_turn_with_cancel(&mut self, input: &str, cancel: CancelToken) -> TurnResponse {
        self.run_turn_with_cancel_and_progress(input, cancel, None)
    }

    fn run_turn_with_cancel_and_progress(
        &mut self,
        input: &str,
        cancel: CancelToken,
        progress: Option<mpsc::UnboundedSender<TurnPhase>>,
    ) -> TurnResponse {
        self.manager.begin_turn();
        let mut control = TurnControl::with_cancel(
            u64::from(self.turns).saturating_add(1),
            self.turn_budget,
            cancel,
        );
        if let Some(progress) = progress {
            control = control.with_progress(progress);
        }
        self.run_turn_checked(input, &mut control, false, 1)
    }

    /// 继续最近一次未完成轮次：先取走再重放；失败后以 attempts+1 重建。
    fn resume_turn(&mut self, cancel: CancelToken) -> Result<TurnResponse, SessionError> {
        self.manager.begin_turn();
        let tick = self.manager.open_turns().tick();
        let can_resume = self
            .manager
            .latest_open_turn()
            .map(|open| open.can_resume(tick))
            .unwrap_or(false);
        if !can_resume {
            return Err(SessionError::NoOpenTurn);
        }
        let open = self
            .manager
            .take_open_turn()
            .ok_or(SessionError::NoOpenTurn)?;
        let mut control = TurnControl::with_cancel(
            u64::from(self.turns).saturating_add(1),
            self.turn_budget,
            cancel,
        );
        let mut response = self.run_turn_checked(
            &open.input,
            &mut control,
            true,
            open.attempts.saturating_add(1),
        );
        response.resumed = true;
        Ok(response)
    }

    fn run_turn_checked(
        &mut self,
        input: &str,
        control: &mut TurnControl,
        resumed: bool,
        attempts: u32,
    ) -> TurnResponse {
        self.turns = self.turns.saturating_add(1);
        if let Some(detail) = self.poisoned.clone() {
            let fault = TurnFault::new(
                TurnFaultKind::Snapshot,
                control.phase(),
                detail,
                control.elapsed_ms(),
            );
            let focus = self.manager.focus_entity().map(str::to_string);
            return self.fault_response(input, fault, focus, attempts, resumed);
        }
        let checkpoint = self.checkpoint();
        let focus_before = self.manager.focus_entity().map(str::to_string);
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.run_turn_inner(input, control)
        }));
        match outcome {
            Ok(Ok(response)) => response,
            Ok(Err(fault)) => {
                let rollback = self.rollback(checkpoint);
                if let Err(error) = rollback {
                    self.poisoned = Some(error);
                }
                self.fault_response(input, fault, focus_before, attempts, resumed)
            }
            Err(payload) => {
                let detail = panic_payload(payload.as_ref());
                let fault = TurnFault::panic(control.phase(), &detail, control.elapsed_ms());
                if let Err(error) = self.rollback(checkpoint) {
                    self.poisoned = Some(error);
                }
                self.fault_response(input, fault, focus_before, attempts, resumed)
            }
        }
    }

    fn fault_response(
        &mut self,
        input: &str,
        fault: TurnFault,
        focus_before: Option<String>,
        attempts: u32,
        resumed: bool,
    ) -> TurnResponse {
        self.manager.record_open_turn_attempts(
            input,
            u64::from(self.turns),
            &fault,
            focus_before.as_deref(),
            None,
            attempts,
        );
        let open_turn = self.open_turn_view();
        let resumable = self.open_turn_resumable();
        TurnResponse {
            session_id: self.id.clone(),
            turn: self.turns,
            status: fault.status().to_string(),
            source: "runtime".to_string(),
            kind: None,
            plan: vec![format!("turn_{}", fault.status())],
            reply: fault.render(),
            focus: self.manager.focus_entity().map(str::to_string),
            policy: self.policy.clone(),
            epistemic: None,
            lexical: None,
            entities: Vec::new(),
            board: self.manager.board().view(self.reasoner.revision()),
            fault: Some(fault),
            open_turn,
            resumed,
            resumable,
            derivation: None,
        }
    }

    fn run_turn_inner(
        &mut self,
        input: &str,
        control: &mut TurnControl,
    ) -> Result<TurnResponse, TurnFault> {
        control.enter(TurnPhase::Prepare)?;
        self.sync_knowledge();
        let raw_input = input.trim().to_string();
        if self.history.len() >= HISTORY_LIMIT {
            self.history.remove(0);
        }
        self.history.push(raw_input.clone());
        self.manager.clear_last_recovery();
        // 上一轮确认/更正过的事实必须进入本轮推理前提（幂等，不写图谱）。
        let adopted = self.sync_recovery_confirmed_facts();
        // 补全回合与假设世界命令使用原始措辞，避免被插件改写；
        // 带「那么」的一次性反事实句仍让位给 what-if 管线。
        let completion_command = self.reasoner.handle_completion_command(&raw_input);
        let constructive_command = self.reasoner.handle_constructive_command(&raw_input);
        let assumption_command = self.reasoner.handle_assumption_command(&raw_input);
        let predictive_command = self.reasoner.handle_predictive_command(&raw_input);

        // 解释空间澄清进行中：严格澄清策略下的短答直接结算为确认/否定/暂缓，
        // 不再走普通问答或对话策略，避免「不是」被当成新问题。
        let clarification_answer = if self.policy.strict_clarify
            && self.reasoner.clarification_active()
            && is_clarification_answer(&raw_input)
        {
            self.reasoner
                .answer_clarification(&raw_input)
                .map(|text| (text, self.reasoner.clarification_active()))
        } else {
            None
        };
        // 悬问回填：上一轮刚追问过的洞，本轮短答直接绑定并重解原目标。
        // 系统命令/补全命令不参与裸回答，避免劫持正常控制流；澄清短答
        // 已由上面的通道结算，不能再被悬问窗口二次消费。
        let pending_answer = if clarification_answer.is_none()
            && completion_command.is_none()
            && constructive_command.is_none()
            && assumption_command.is_none()
            && predictive_command.is_none()
        {
            self.reasoner.try_answer_pending_question(&raw_input)
        } else {
            None
        };
        if pending_answer.is_none() && clarification_answer.is_none() {
            // 裸回答窗口只开一轮：本轮没绑定成功，后续输入不得被旧追问劫持。
            self.reasoner.clear_pending_awaiting();
        }
        // 上一轮的确认/更正若补上了悬问的洞，立即给出延期回答。
        let deferred_after_adopt = if adopted > 0 {
            self.reasoner.resolve_pending_after_assertion()
        } else {
            None
        };

        // 插件改写预通道：词表/本体归一只影响管线输入，原始措辞仍用于接地引用。
        let plugin_outcome: Option<PluginOutcome> = self
            .plugin
            .as_ref()
            .map(|runtime| runtime.lock().rewrite(&raw_input));
        control.enter(TurnPhase::Rewrite)?;
        let input = plugin_outcome
            .as_ref()
            .map(|outcome| outcome.trace.effective.clone())
            .unwrap_or_else(|| raw_input.clone());
        let plugin_answer = self.plugin_recipe_answer(&raw_input, plugin_outcome.as_ref());

        control.enter(TurnPhase::Recall)?;
        let dialogue = if completion_command.is_some()
            || constructive_command.is_some()
            || assumption_command.is_some()
            || predictive_command.is_some()
        {
            None
        } else {
            self.manager
                .plan_pure_dialogue(&input, &self.reasoner)
                .or_else(|| self.manager.plan_reply(&input, &self.reasoner))
        };

        control.enter(TurnPhase::Reason)?;
        let (status, source, kind, plan, mut reply, derivation) =
            if let Some(text) = completion_command {
                (
                    "answered".to_string(),
                    "completion".to_string(),
                    Some("completion_command".to_string()),
                    vec!["completion_command".to_string()],
                    text,
                    None,
                )
            } else if let Some(text) = constructive_command {
                (
                    "answered".to_string(),
                    "constructive".to_string(),
                    Some("constructive_command".to_string()),
                    vec!["constructive_command".to_string()],
                    text,
                    None,
                )
            } else if let Some(text) = assumption_command {
                (
                    "answered".to_string(),
                    "assumption".to_string(),
                    Some("assumption_world".to_string()),
                    vec!["assumption_world".to_string()],
                    text,
                    None,
                )
            } else if let Some(text) = predictive_command {
                (
                    "answered".to_string(),
                    "predictive".to_string(),
                    Some("predictive_command".to_string()),
                    vec!["predictive_command".to_string()],
                    text,
                    None,
                )
            } else if let Some((text, still_active)) = clarification_answer {
                (
                    if still_active {
                        "needs_clarification".to_string()
                    } else {
                        "answered".to_string()
                    },
                    "clarification".to_string(),
                    Some("clarification".to_string()),
                    vec!["clarification".to_string()],
                    text,
                    None,
                )
            } else if let Some(text) = pending_answer {
                (
                    "answered".to_string(),
                    "pending_question".to_string(),
                    Some("pending_question".to_string()),
                    vec!["pending_question".to_string()],
                    text,
                    None,
                )
            } else if let Some((text, plan, kind)) = plugin_answer {
                (
                    "answered".to_string(),
                    "plugin".to_string(),
                    Some(kind),
                    plan,
                    text,
                    None,
                )
            } else if let Some(reply) = dialogue {
                let text = self
                    .plugin
                    .as_ref()
                    .map(|runtime| runtime.lock().style_wrap(&reply.text))
                    .unwrap_or(reply.text);
                (
                    "answered".to_string(),
                    "dialogue".to_string(),
                    Some(reply.kind.label().to_string()),
                    reply.plan,
                    text,
                    reply.derivation,
                )
            } else if let Some(answer) = self.reasoner.answer(&input) {
                (
                    "answered".to_string(),
                    "reasoner".to_string(),
                    None,
                    Vec::new(),
                    answer,
                    None,
                )
            } else if let Some((text, plan)) = self
                .plugin
                .as_ref()
                .and_then(|runtime| runtime.lock().policy_reply(&input))
            {
                let text = self
                    .plugin
                    .as_ref()
                    .map(|runtime| runtime.lock().style_wrap(&text))
                    .unwrap_or(text);
                (
                    "answered".to_string(),
                    "plugin".to_string(),
                    Some("policy".to_string()),
                    vec![plan],
                    text,
                    None,
                )
            } else {
                // 让位也是失败：以 Unknown 回执记账（Off 模式零副作用）
                self.manager.note_deferred(&input, &self.reasoner);
                if let Some(prompt) = self.reasoner.open_pending_question(&input) {
                    (
                        "deferred".to_string(),
                        "pending_question".to_string(),
                        Some("pending_question".to_string()),
                        vec!["pending_question".to_string()],
                        prompt,
                        None,
                    )
                } else {
                    (
                        "deferred".to_string(),
                        "none".to_string(),
                        None,
                        Vec::new(),
                        String::new(),
                        None,
                    )
                }
            };
        // 失败即立悬问：回答/对话路径诚实报告「没有记录」时，把
        // 「目标 + 缺哪条前提」留给下一轮；成功回答则关闭同题悬问。
        if (source == "reasoner" || source == "dialogue") && !reply.is_empty() {
            if is_no_record_answer(&reply) {
                if let Some(prompt) = self.reasoner.open_pending_question(&input) {
                    reply.push('\n');
                    reply.push_str(&prompt);
                }
            } else {
                self.reasoner.dismiss_pending_question(&input);
            }
        }
        if let Some(deferred) = deferred_after_adopt {
            if !reply.is_empty() {
                reply.push('\n');
            }
            reply.push_str(&deferred);
        }
        control.enter(TurnPhase::Generate)?;
        if !reply.is_empty() && self.history.len() < HISTORY_LIMIT {
            self.history.push(reply.clone());
        }
        let epistemic = match self.manager.last_recovery() {
            Some(notice) => Some(EpistemicView {
                status: notice.status.code().to_string(),
                label: notice.status.label().to_string(),
                level: notice.level.clone(),
                receipt: notice.receipt_id.clone(),
                evidence: notice.evidence.clone(),
                question: notice.question.clone(),
                open_receipts: self.manager.failure_ledger().open_count(),
                pending_followups: self.manager.recovery_state().pending.len(),
            }),
            None => None,
        };
        let focus = self.manager.focus_entity().map(str::to_string);
        // 让位轮才做结构化读法诊断：成功轮不额外分词，回答路径保持零开销。
        let lexical = if status == "deferred" {
            self.reasoner
                .read_question(&input)
                .map(|reading| LexicalView::from_reading(&reading))
        } else {
            None
        };
        // 接地引用：本轮输入链接到的实体（与回答同源，可审计、可寻址）
        let entities = self
            .reasoner
            .grounding_mentions(&input)
            .into_iter()
            .filter(|grounded| {
                grounded.score >= crate::symbolic_reasoner::SymbolicReasoner::LINK_ACCEPT_THRESHOLD
            })
            .map(|grounded| GroundingCitation {
                uri: crate::identity::entity_uri(
                    crate::identity::DEFAULT_BASE_NS,
                    &grounded.entity,
                ),
                role: if focus.as_deref() == Some(grounded.entity.as_str()) {
                    "focus".to_string()
                } else {
                    "mention".to_string()
                },
                surface: grounded.mention,
                etype: grounded.etype,
            })
            .collect();
        let mut response = TurnResponse {
            session_id: self.id.clone(),
            turn: self.turns,
            status,
            source,
            kind,
            plan,
            reply,
            focus,
            policy: self.policy.clone(),
            epistemic,
            lexical,
            entities,
            board: self.manager.board().view(self.reasoner.revision()),
            fault: None,
            open_turn: self.open_turn_view(),
            resumed: false,
            resumable: self.open_turn_resumable(),
            derivation,
        };
        self.policy.apply(&input, &mut response, &self.reasoner);
        // 轮末预取：只把焦点实体下一轮可能用到的链接预测提前算好（有界），
        // 回答文本与轮次语义不变；失败/澄清轮不预取，避免为不确定目标付费。
        if response.status == "answered" {
            self.reasoner.prefetch_answer_drafts();
        }
        control.enter(TurnPhase::Commit)?;
        Ok(response)
    }
}
/// 从 panic payload 提取可读消息（actor 必须存活，错误不能吞掉）。
fn panic_payload(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(text) = payload.downcast_ref::<&str>() {
        (*text).to_string()
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else {
        "未知 panic".to_string()
    }
}

// ---------------------------------------------------------------------------
// 会话信封：可移植的会话序列化格式
// ---------------------------------------------------------------------------
/// 信封格式标识与版本（导入时逐项校验）。
pub const ENVELOPE_FORMAT: &str = "knowprolog-session-envelope";
pub const ENVELOPE_VERSION: u32 = 1;
/// 引擎指纹：证明信封产自同版本、同知识库、同运行模式的引擎。
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineStamp {
    pub crate_version: String,
    /// 知识库事实内容的 FNV-1a 指纹（与加载顺序无关的确定性哈希）。
    pub kb_fingerprint: String,
    pub kb_facts: usize,
    #[serde(default)]
    pub kb_generation: u64,
    pub prompt_mode: String,
    pub recovery_mode: String,
}
impl EngineStamp {
    fn mismatched_fields(&self, other: &EngineStamp) -> Vec<String> {
        let mut fields = Vec::new();
        if self.crate_version != other.crate_version {
            fields.push("crate_version".to_string());
        }
        if self.kb_fingerprint != other.kb_fingerprint {
            fields.push("kb_fingerprint".to_string());
        }
        if self.kb_facts != other.kb_facts {
            fields.push("kb_facts".to_string());
        }
        if self.kb_generation != other.kb_generation {
            fields.push("kb_generation".to_string());
        }
        if self.prompt_mode != other.prompt_mode {
            fields.push("prompt_mode".to_string());
        }
        if self.recovery_mode != other.recovery_mode {
            fields.push("recovery_mode".to_string());
        }
        fields
    }

    fn mismatched_fields_for_resume(&self, current: &EngineStamp) -> Vec<String> {
        let mut fields = Vec::new();
        if self.crate_version != current.crate_version {
            fields.push("crate_version".to_string());
        }
        if self.prompt_mode != current.prompt_mode {
            fields.push("prompt_mode".to_string());
        }
        if self.recovery_mode != current.recovery_mode {
            fields.push("recovery_mode".to_string());
        }
        if self.kb_generation > current.kb_generation {
            fields.push("kb_generation".to_string());
        }
        fields
    }
}
/// 会话信封：策略 + 轮数 + 服务层历史 + 管理器/推理器快照 + 引擎指纹 + 摘要。
///
/// `digest` 是对除自身以外全部字段序列化结果的 FNV-1a 摘要；导入时重算比对，
/// 任何字段被篡改都会被拒绝。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SessionEnvelope {
    pub format: String,
    pub version: u32,
    pub session_id: String,
    pub turns: u32,
    #[serde(default)]
    pub history: Vec<String>,
    #[serde(default)]
    pub policy: SessionPolicy,
    pub manager: DialogueManagerSnapshot,
    pub reasoner: DialogueSnapshot,
    pub engine: EngineStamp,
    #[serde(default)]
    pub digest: String,
}
impl SessionEnvelope {
    /// 计算（不写入）摘要。摘要对**规范字节**计算（对象键递归排序），
    /// 因此 `HashMap` 迭代序不会泄漏进摘要：同一状态跨进程永远同一摘要。
    fn compute_digest(&self) -> String {
        let mut copy = self.clone();
        copy.digest = String::new();
        canonical::canonical_digest(&copy).unwrap_or_default()
    }

    /// 就地封缄：写入正确摘要。
    pub fn seal(&mut self) {
        self.digest = self.compute_digest();
    }

    pub fn digest_valid(&self) -> bool {
        !self.digest.is_empty() && self.digest == self.compute_digest()
    }

    /// 解码宪法入口：体量门 → 解析 → 格式门 → 版本门 → 摘要门。
    ///
    /// 任一门失败都返回带稳定错误码的 [`DecodeError`]，且不产生任何副作用；
    /// 全部通过才返回信封与 [`DecodeReceipt`]（含「字节是否已规范」）。
    pub fn decode(bytes: &[u8], max_bytes: usize) -> Result<(Self, DecodeReceipt), DecodeError> {
        let (envelope, mut receipt) =
            canonical::decode_json::<Self>(bytes, "session_envelope", ENVELOPE_VERSION, max_bytes)?;
        if envelope.format != ENVELOPE_FORMAT {
            return Err(DecodeError::Format(truncate_chars(&envelope.format, 32)));
        }
        if envelope.version != ENVELOPE_VERSION {
            return Err(DecodeError::Version {
                found: envelope.version,
                supported: ENVELOPE_VERSION,
            });
        }
        if !envelope.digest_valid() {
            return Err(DecodeError::Digest {
                found: truncate_chars(&envelope.digest, 32),
                expected: truncate_chars(&envelope.compute_digest(), 32),
            });
        }
        receipt.version = envelope.version;
        Ok((envelope, receipt))
    }
}
pub use crate::knowledge_delta::fingerprint_facts;
/// 会话 id → 冷存储文件名（过滤路径分隔符等，防目录穿越）。
fn session_file_name(session_id: &str) -> String {
    let mut name: String = session_id
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' || ch == '.' {
                ch
            } else {
                '_'
            }
        })
        .collect();
    if name.is_empty() {
        name.push('_');
    }
    name.truncate(128);
    name
}
/// 导入信封时恢复会话所需的全部可序列化状态。
struct SessionRestore {
    turns: u32,
    history: Vec<String>,
    policy: SessionPolicy,
    manager: DialogueManagerSnapshot,
    reasoner: DialogueSnapshot,
}
/// actor 命令：会话线程是唯一能触碰 [`Session`] 的执行者。
enum SessionCommand {
    Turn {
        input: String,
        policy: Option<SessionPolicy>,
        cancel: CancelToken,
        progress: Option<mpsc::UnboundedSender<TurnPhase>>,
        reply: oneshot::Sender<TurnResponse>,
    },
    Resume {
        cancel: CancelToken,
        reply: oneshot::Sender<Result<TurnResponse, SessionError>>,
    },
    DiscardOpenTurn {
        reply: oneshot::Sender<Option<OpenTurn>>,
    },
    Snapshot {
        reply: oneshot::Sender<serde_json::Value>,
    },
    Export {
        reply: oneshot::Sender<SessionEnvelope>,
    },
    ReadTraces {
        reply: oneshot::Sender<serde_json::Value>,
    },
    Semantic {
        reply: oneshot::Sender<serde_json::Value>,
    },
    Layers {
        reply: oneshot::Sender<serde_json::Value>,
    },
}
/// 会话句柄：HTTP 层可见的唯一部分（`Send + Sync`）。
struct SessionHandle {
    tx: mpsc::Sender<SessionCommand>,
    /// 最近一次被访问的活动序号（确定性 LRU，不依赖挂钟）。
    last_active: Arc<AtomicU64>,
}
impl SessionHandle {
    fn touch(&self, clock: &AtomicU64) {
        self.last_active
            .store(clock.fetch_add(1, Ordering::Relaxed), Ordering::Relaxed);
    }
}
/// 会话访问错误：区分「不存在」「actor 已退出」「冷会话不可恢复」。
/// `Stale` 携带解码错误分类：为什么被拒（parse/version/digest/fingerprint…）
/// 是可回答的，而不是折叠成一个「不行」。
#[derive(Debug)]
enum SessionError {
    NotFound,
    Unavailable,
    /// 会话收件箱已满：拒绝排队，让客户端退避后重试。
    Busy,
    /// 等待 actor 超过硬上限：已触发协作取消。
    Timeout,
    /// 没有可继续的未完成轮次（或已过期 / 尝试次数耗尽）。
    NoOpenTurn,
    Stale(DecodeError),
}
/// 预热基座：进程启动时对 `(facts, reasoning config, synonyms)` 只做一次
/// `observe_facts` 全量重放，之后每个会话按值分叉（字段级深拷贝），不重放。
///
/// 这是 LLM 权重预加载的符号等价物：知识派生状态（对齐剖面/谓词规则/关联/
/// 向量/全息码本/链接索引）只构建一次；分叉输出与「全新观察同一批事实」
/// 逐字节一致（`fork_matches_fresh_observe_byte_for_byte` 守护）。
///
/// 事实正文不进入基座的可变部分：store 仍是唯一事实源；若知识版本变化
/// （未来的写入路径），调用方应重建整个 [`WarmBase`]（单飞），而不是修补它。
pub struct WarmBase {
    reasoner: parking_lot::Mutex<SymbolicReasoner>,
}
impl WarmBase {
    /// 构建基座：全量观察一次事实（唯一一次重放），并计入启动分期。
    fn build(
        facts: &[PrologFact],
        reasoning: &ReasoningConfig,
        synonyms: &HashMap<String, String>,
        perf: &Arc<PerfLedger>,
        startup: &StartupLedger,
    ) -> Self {
        let _span = startup.span("warm_build");
        let mut reasoner = SymbolicReasoner::with_perf_ledger(
            reasoning.clone(),
            synonyms.clone(),
            Arc::clone(perf),
        );
        reasoner.observe_facts(facts);
        perf.bump("warm_base_builds");
        Self {
            reasoner: parking_lot::Mutex::new(reasoner),
        }
    }

    /// 分叉一个新会话的推理器。锁只在克隆期间持有（毫秒级），
    /// 并发建会话按到达顺序串行分叉，但不会重放事实。
    fn fork(&self) -> SymbolicReasoner {
        self.reasoner.lock().fork_session()
    }

    /// 为后续分叉注入 NLP 配置（失败轮读法回执需要词性来源）。
    fn set_nlp_config(&self, config: NlpConfig) {
        self.reasoner.lock().set_nlp_config(config);
    }

    /// 为后续分叉注入双向语义规则包（生成侧与抽取侧共享模板）。
    fn set_semantic_grammar(&self, grammar: crate::semantic_grammar::RulePack) {
        self.reasoner.lock().attach_semantic_grammar(grammar);
    }

    fn observe_facts(&self, facts: &[PrologFact]) {
        self.reasoner.lock().observe_facts(facts);
    }

    fn rebuild_from_graph(&self, facts: &[PrologFact]) {
        self.reasoner.lock().rebuild_from_graph(facts);
    }

    /// 知识版本（分叉基座当前携带的 revision，供审计与测试）。
    pub fn revision(&self) -> u64 {
        self.reasoner.lock().revision()
    }
}
/// 构造会话（可在任意线程调用；返回的 [`Session`] 只应留在该线程）。
#[allow(clippy::too_many_arguments)]
fn build_session(
    id: String,
    warm: &WarmBase,
    knowledge: Arc<KnowledgeBase>,
    perf: &Arc<PerfLedger>,
    prompt_mode: PromptMode,
    recovery_mode: RecoveryMode,
    quality: Option<Arc<QualityIndex>>,
    plugin: Option<Arc<parking_lot::Mutex<PluginRuntime>>>,
    frame_program: Option<Arc<FrameProgram>>,
    turn_budget: TurnBudget,
    restore: Option<SessionRestore>,
) -> Session {
    let reasoner = warm.fork();
    let knowledge_generation = knowledge.generation();
    perf.bump("session_forks");
    let mut manager = DialogueManager::new();
    manager.set_prompt_mode(prompt_mode);
    manager.set_recovery_mode(recovery_mode);
    manager.set_frame_program(frame_program);
    // 质量担保是只读派生数据：多会话共享同一份索引，不写图谱、不进快照。
    manager.set_quality_index_shared(quality);
    let mut session = Session {
        id,
        turns: 0,
        manager,
        reasoner,
        history: Vec::new(),
        policy: SessionPolicy::default(),
        plugin,
        turn_budget,
        poisoned: None,
        knowledge,
        knowledge_generation,
        perf: Arc::clone(perf),
    };
    if let Some(restore) = restore {
        session.turns = restore.turns;
        session.history = restore.history;
        session.policy = restore.policy;
        session.manager.restore(&restore.manager);
        // 信封层已按规范字节 + 摘要把关；此处仍把本体胶囊校验失败当作
        // 「快照不可恢复」上报（零半应用），不让被改写的公理进入会话。
        if let Err(error) = session
            .reasoner
            .restore_dialogue_snapshot(&restore.reasoner)
        {
            eprintln!(
                "会话快照拒绝（本体胶囊校验失败，其余状态未应用）：{}",
                error
            );
        }
    }
    session
}
/// 服务共享状态：预热基座 + 会话 actor 表 + 计数（`Send + Sync`）。
pub struct ApiState {
    knowledge: Arc<KnowledgeBase>,
    /// 知识派生状态的预热基座：会话从这里分叉，不逐会话重放事实。
    warm: Arc<WarmBase>,
    /// 无状态问答缓存：键为「按序全部用户消息 + 策略」，纪元失效（纯函数结果）。
    stateless_answers: Arc<parking_lot::Mutex<RevisionLru<TurnResponse, CacheEpoch>>>,
    /// 知识写入纪元：每次事实/别名写入推进，旧缓存整体失效。
    kb_revision: AtomicU64,
    knowledge_write: parking_lot::Mutex<()>,
    /// 质量护照索引（由宿主注入；只读派生数据，供会话生成出处/冲突/缺口话语）。
    quality: Option<Arc<QualityIndex>>,
    /// 插件运行时（由宿主注入；多会话共享，模式在注入时确定）。
    plugin: Option<Arc<parking_lot::Mutex<PluginRuntime>>>,
    /// 身份准入纪元：新实体/别名准入时 +1，全部缓存条目整体失效（可审计）。
    identity_epoch: AtomicU64,
    perf: Arc<PerfLedger>,
    /// 启动分期台账（进程启动 / 会话分叉 / 冷换入）。
    startup: Arc<StartupLedger>,
    sessions: Mutex<BTreeMap<String, SessionHandle>>,
    next_session_id: AtomicU64,
    turns_total: AtomicU64,
    /// 冷会话换出次数（容量压力下把 LRU 会话冻结到磁盘）。
    evictions_total: AtomicU64,
    /// 会话活动时钟：每次命中 +1，用于确定性 LRU（不依赖挂钟）。
    activity_clock: Arc<AtomicU64>,
    /// 冷会话目录：配置后，容量已满时把 LRU 会话冻结到磁盘，
    /// 再次访问时反序列化重建（上下文缓存的符号等价物）。
    session_dir: Option<std::path::PathBuf>,
    max_sessions: usize,
    max_body_bytes: usize,
    prompt_mode: PromptMode,
    recovery_mode: RecoveryMode,
    /// 配置编译的绑定帧程序（多会话共享只读程序，记忆按会话隔离）。
    frame_program: Option<Arc<FrameProgram>>,
    /// 每轮预算（复制给每个会话 actor；默认 30 秒 Enforce）。
    turn_budget: TurnBudget,
    /// 跨会话共享语义白板：默认私有，显式发布到全局或指定会话。
    shared_memory: Arc<SharedMemoryBoard>,
}

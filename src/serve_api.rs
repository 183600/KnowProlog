//! 会话式 REST 服务（零新依赖的最小 HTTP/1.1 网关）。
//!
//! 把「多轮对话记忆」做成服务级能力：每个会话一个 [`DialogueManager`] +
//! 独立 [`SymbolicReasoner`]，共享启动时加载的知识事实；SSE 按句流式输出
//! 等价于符号版的 token streaming；`/health` 与 `/metrics` 暴露确定性计数。
//!
//! `system` / `developer` 消息被解析为确定性 [`SessionPolicy`]（白名单短语），
//! 生效于该会话的后续轮次：简短/详细风格、必须给依据、严格澄清。
//!
//! # 会话 actor（并发结构）
//!
//! 每个会话由**专属 OS 线程**拥有（actor）：HTTP 层只持有一个
//! `mpsc::Sender`，通过 oneshot 等待结果。其结果是：
//! - [`ApiState`] 是 `Send + Sync`，accept 循环 `tokio::spawn` 每连接，
//!   跨会话真并行、同会话严格 FIFO（命令按到达顺序处理）；
//! - 一个慢查询/慢 SSE 连接不会阻塞其他连接的 accept；
//! - 会话关闭 = 丢弃 `Sender`，actor 处理完已排队命令后线程自然退出。
//!
//! # 会话信封（序列化 + 冷存储）
//!
//! `GET /sessions/{id}/export` 把会话导出为可移植 JSON 信封：策略 + 轮数 +
//! 历史 + 对话管理器快照（含失败回执台账）+ 推理器会话记忆（焦点/会话事实/
//! 教学/会话规则/话语帧）+ 引擎指纹（crate 版本、知识库 FNV-1a 指纹、事实数、
//! 运行模式）+ 信封摘要（FNV-1a）。`POST /sessions/import` 反序列化重建：
//! 版本 / 摘要 / 引擎指纹 / 尺寸四重校验全部通过才产生副作用；陌生引擎或
//! 不同知识库的信封默认 409 拒收，`?accept_foreign=1` 显式接收并回执差异字段。
//!
//! 配置 `--session-dir` 后，`max_sessions` 变为「内存驻留上限」：容量已满时
//! 最近最少使用的会话被冻结到磁盘（换出），再次访问时自动反序列化重建
//! （换入）——LLM 服务 context cache 的确定性符号等价物；DELETE/关闭会同时
//! 清理冷存储，已删除会话不会复活。
//!
//! 端点：
//! - `GET  /`                      服务索引（端点清单）
//! - `GET  /health`                健康检查
//! - `GET  /metrics`               会话数 / 轮数 / 性能台账 / 启动分期明细
//! - `POST /sessions`              新建会话 → `{"session_id": "s-1"}`
//! - `GET  /sessions/{id}`         会话快照（焦点栈 / 历史摘要 / 策略）
//! - `GET  /sessions/{id}/layers`  四层知识容器清单（基础/会话/推测/假设）
//! - `GET  /sessions/{id}/export`  导出会话信封（可移植 JSON）
//! - `POST /sessions/import`       导入会话信封重建会话
//! - `DELETE /sessions/{id}`       关闭会话
//! - `POST /sessions/{id}/turns`   一轮对话；`Accept: text/event-stream`
//!   或 `?stream=1` 时按句 SSE 流式返回。轮次有墙钟预算（默认 30 秒 Enforce），
//!   失败/取消/超时会回滚可观察状态并留下可继续的未完成轮次
//! - `POST /sessions/{id}/resume`  继续最近的未完成轮次（消费一次，失败 attempts+1）
//! - `GET  /v1/models`             OpenAI 兼容：模型清单
//! - `GET  /v1/prompts`            提示资产注册表（spec + lint + 覆盖）
//! - `POST /v1/events`             无状态事件图抽取 / 事件焦点问答
//! - `POST /v1/chat/completions`   OpenAI 兼容聊天补全（`stream: true` 时
//!   按句 SSE 流式）；无扩展字段时按无状态语义重放用户消息，带
//!   `"session_id"` 扩展时只执行最后一条用户消息并复用服务端记忆
//!
//! 设计约束：
//! - 零新依赖：tokio 已在依赖中，HTTP/1.1 手写最小实现；
//! - 恢复等价性承诺「可观察输出逐字一致」（回答/状态/焦点/策略），不承诺
//!   内部派生缓存一致——缓存是快照的派生数据；
//! - 请求体上限、连接超时、会话数上限全部显式配置，防御性默认值。

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
use crate::ontology_reasoning::OntologyProgram;
use crate::perf_ledger::PerfLedger;
use crate::plugin::{PluginOutcome, PluginRuntime};
use crate::prolog_engine::PrologFact;
use crate::prompt_spec::{self, PromptMode};
use crate::quality::QualityIndex;
use crate::reading::{Reading, Residual};
use crate::recovery::RecoveryMode;
use crate::startup_ledger::StartupLedger;
use crate::symbolic_reasoner::{DialogueSnapshot, SymbolicReasoner};
use crate::turn_runtime::{
    CancelToken, OpenTurn, TurnBudget, TurnControl, TurnFault, TurnFaultKind, TurnPhase,
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
}

impl PluginOntologyReceipt {
    pub fn is_empty(self) -> bool {
        self.lessons == 0 && self.aliases == 0 && self.inverse_surfaces == 0
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

        // 悬问回填：上一轮刚追问过的洞，本轮短答直接绑定并重解原目标。
        // 系统命令/补全命令不参与裸回答，避免劫持正常控制流。
        let pending_answer = if completion_command.is_none()
            && constructive_command.is_none()
            && assumption_command.is_none()
            && predictive_command.is_none()
        {
            self.reasoner.try_answer_pending_question(&raw_input)
        } else {
            None
        };
        if pending_answer.is_none() {
            // 裸回答窗口只开一轮：本轮没绑定成功，后续输入不得被旧追问劫持。
            self.reasoner.clear_pending_awaiting();
        }
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
        if self.prompt_mode != other.prompt_mode {
            fields.push("prompt_mode".to_string());
        }
        if self.recovery_mode != other.recovery_mode {
            fields.push("recovery_mode".to_string());
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

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv1a_step(hash: u64, bytes: &[u8]) -> u64 {
    let mut hash = hash;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn fnv1a_field(hash: u64, value: &str) -> u64 {
    fnv1a_step(fnv1a_step(hash, value.as_bytes()), b"\x1f")
}

/// 事实集的确定性指纹：HashMap 属性按键排序，保证跨进程稳定。
pub fn fingerprint_facts(facts: &[PrologFact]) -> String {
    let mut hash = FNV_OFFSET;
    for fact in facts {
        match fact {
            PrologFact::Entity {
                name,
                entity_type,
                properties,
            } => {
                hash = fnv1a_step(hash, b"E\x1f");
                hash = fnv1a_field(hash, name);
                hash = fnv1a_field(hash, entity_type);
                let mut keys: Vec<&String> = properties.keys().collect();
                keys.sort();
                for key in keys {
                    hash = fnv1a_field(hash, key);
                    hash = fnv1a_field(hash, &properties[key]);
                }
            }
            PrologFact::Relation { from, rel_type, to } => {
                hash = fnv1a_step(hash, b"R\x1f");
                hash = fnv1a_field(hash, from);
                hash = fnv1a_field(hash, rel_type);
                hash = fnv1a_field(hash, to);
            }
        }
        hash = fnv1a_step(hash, b"\n");
    }
    format!("{:016x}", hash)
}

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
    facts: Arc<Vec<PrologFact>>,
    kb_fingerprint: String,
    /// 知识派生状态的预热基座：会话从这里分叉，不逐会话重放事实。
    warm: Arc<WarmBase>,
    /// 无状态问答缓存：键为「按序全部用户消息 + 策略」，纪元失效（纯函数结果）。
    stateless_answers: Arc<parking_lot::Mutex<RevisionLru<TurnResponse, CacheEpoch>>>,
    /// 预热基座构建时的知识版本（缓存纪元的一部分）。
    kb_revision: u64,
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
}

impl ApiState {
    pub fn new(
        facts: Vec<PrologFact>,
        reasoning: ReasoningConfig,
        synonyms: HashMap<String, String>,
        perf: Arc<PerfLedger>,
        max_sessions: usize,
        startup: Arc<StartupLedger>,
    ) -> Self {
        let kb_fingerprint = fingerprint_facts(&facts);
        // 预热基座：整进程唯一的全量重放点（会话创建只做字段级分叉）
        let warm = Arc::new(WarmBase::build(
            &facts, &reasoning, &synonyms, &perf, &startup,
        ));
        let kb_revision = warm.revision();
        ApiState {
            facts: Arc::new(facts),
            kb_fingerprint,
            warm,
            stateless_answers: Arc::new(parking_lot::Mutex::new(RevisionLru::new(
                STATELESS_CACHE_CAPACITY,
            ))),
            kb_revision,
            quality: None,
            plugin: None,
            identity_epoch: AtomicU64::new(0),
            perf,
            startup,
            sessions: Mutex::new(BTreeMap::new()),
            next_session_id: AtomicU64::new(1),
            turns_total: AtomicU64::new(0),
            evictions_total: AtomicU64::new(0),
            activity_clock: Arc::new(AtomicU64::new(1)),
            session_dir: None,
            max_sessions: max_sessions.max(1),
            max_body_bytes: DEFAULT_MAX_BODY_BYTES,
            prompt_mode: PromptMode::On,
            recovery_mode: RecoveryMode::Observe,
            frame_program: None,
            turn_budget: TurnBudget::enforced(crate::turn_runtime::DEFAULT_TURN_BUDGET),
        }
    }

    /// 共享的启动分期台账（主程序打印摘要 / `/metrics` 输出明细）。
    pub fn startup_ledger(&self) -> Arc<StartupLedger> {
        Arc::clone(&self.startup)
    }

    /// 预热基座的知识版本（分叉一致性审计）。
    pub fn warm_revision(&self) -> u64 {
        self.warm.revision()
    }

    /// 注入 NLP 配置：后续分叉的会话在回答失败时记录有界读法回执。
    /// 必须在启动会话 actor 之前调用；不改变任何回答文本。
    pub fn set_nlp_config(&mut self, config: NlpConfig) {
        self.warm.set_nlp_config(config);
    }

    /// 注入双向语义规则包：之后由本基座分叉的会话在生成时复用同一份模板。
    pub fn set_semantic_grammar(&mut self, grammar: crate::semantic_grammar::RulePack) {
        self.warm.set_semantic_grammar(grammar);
    }

    /// 宿主注入配置态本体（`[type_system]` + `[[predicate_profiles]]` 归约的公理）：
    /// 在建会话前调用，之后分叉的每个会话都携带同一份单一本体。
    /// 返回新登记的公理条数（重复调用按整句去重，返回 0）。
    pub fn configure_ontology(&self, program: &OntologyProgram) -> usize {
        self.warm
            .reasoner
            .lock()
            .configure_ontology_program(program)
    }

    /// 宿主注入已启用插件的本体贡献：谓词别名、逆向词面与结构性公理
    /// 全部进入**同一个**暖基座推理器，之后分叉的每个会话同源携带。
    /// 与 `configure_ontology` 一样是单调、幂等的启动期操作。
    pub fn configure_plugin_runtime(&self, runtime: &PluginRuntime) -> PluginOntologyReceipt {
        let sync = self.warm.reasoner.lock().apply_plugin_runtime(runtime);
        PluginOntologyReceipt {
            lessons: sync.lessons,
            aliases: sync.aliases,
            inverse_surfaces: sync.inverse_surfaces,
        }
    }

    /// 宿主注入谓词对齐（与交互启动同一注册入口）：画像别名/换向词面生效后，
    /// 谓词画像律典一并接入选路。必须在建会话前调用；返回生效的对齐条数。
    pub fn align_predicate_pairs(
        &self,
        pairs: &[(String, String, bool)],
        laws: &crate::predicate_registry::PredicateRegistry,
    ) -> usize {
        let mut reasoner = self.warm.reasoner.lock();
        let applied = reasoner.align_predicate_pairs(pairs);
        reasoner.configure_laws(laws);
        drop(reasoner);
        // 谓词对齐与律典重配都改变选路/结论：旧的无状态答案可能按旧词面缓存，
        // 必须整体失效（律典即使无新别名也重放，故此处无条件推进纪元）。
        self.bump_identity_epoch();
        applied
    }

    /// 宿主注入实体身份注册表里已确认的持久别名（别名与规范名同权），
    /// 使 CLI 里签过字的指称在 HTTP 会话同样可解。返回生效条数。
    pub fn align_entity_aliases(&self, aliases: &[(String, String)]) -> usize {
        let applied = self.warm.reasoner.lock().align_entity_aliases(aliases);
        // 新别名让旧词面指向新实体：缓存里的旧答案可能已改口。
        if applied > 0 {
            self.bump_identity_epoch();
        }
        applied
    }

    /// 宿主注入律典（跨重启的采纳/退役决定），返回载入的显式决定条数。
    pub fn load_law_book(&self, laws: crate::law_book::LawBook) -> usize {
        let applied = self.warm.reasoner.lock().load_persisted_laws(laws);
        // 采纳/退役定律会改变推理结论，缓存须随之失效。
        if applied > 0 {
            self.bump_identity_epoch();
        }
        applied
    }

    /// 当前缓存纪元（kb 版本 + 身份准入纪元）。
    pub fn cache_epoch(&self) -> CacheEpoch {
        CacheEpoch {
            kb_revision: self.kb_revision,
            identity_epoch: self.identity_epoch.load(Ordering::Relaxed),
        }
    }

    /// 推进身份准入纪元（准入新词面后调用）：所有缓存条目整体失效。
    pub fn bump_identity_epoch(&self) {
        self.identity_epoch.fetch_add(1, Ordering::Relaxed);
        self.perf.bump("identity_epoch_bumps");
    }

    /// 无状态问答缓存查询：命中即返回缓存响应（调用方负责重戳会话 id）。
    fn cached_stateless_answer(&self, epoch: CacheEpoch, key: &str) -> Option<TurnResponse> {
        let mut cache = self.stateless_answers.lock();
        let hit = cache.get(epoch, key).cloned();
        if hit.is_some() {
            self.perf.bump("answer_cache_hits");
        } else {
            self.perf.bump("answer_cache_misses");
        }
        hit
    }

    /// 写入无状态问答缓存；返回被淘汰条目数并计入台账。
    fn store_stateless_answer(&self, epoch: CacheEpoch, key: String, response: TurnResponse) {
        let evicted = self.stateless_answers.lock().put(epoch, key, response);
        self.perf.bump_by("answer_cache_evictions", evicted as u64);
    }

    /// 设置冷会话目录（在 `Arc::new` 之前调用）。
    ///
    /// 配置后 `max_sessions` 语义变为「内存驻留上限」：容量已满时新建/恢复会话
    /// 会把最近最少使用的会话冻结到该目录，而不是直接拒绝；被冻结的会话在
    /// 下一次访问时自动反序列化重建。
    pub fn set_session_dir(&mut self, dir: impl Into<std::path::PathBuf>) {
        self.session_dir = Some(dir.into());
    }

    pub fn session_dir(&self) -> Option<&std::path::Path> {
        self.session_dir.as_deref()
    }

    /// 设置模板运行模式（在 `Arc::new` 之前调用；默认 [`PromptMode::On`]，
    /// 接地失败会自动回退组合生成器）。
    pub fn set_prompt_mode(&mut self, mode: PromptMode) {
        self.prompt_mode = mode;
    }

    pub fn prompt_mode(&self) -> PromptMode {
        self.prompt_mode
    }

    /// 设置失败恢复模式（在 `Arc::new` 之前调用；默认 [`RecoveryMode::Observe`]：
    /// 只记失败台账不改回答文本，On 才启用完整恢复阶梯）。
    pub fn set_recovery_mode(&mut self, mode: RecoveryMode) {
        self.recovery_mode = mode;
    }

    pub fn recovery_mode(&self) -> RecoveryMode {
        self.recovery_mode
    }

    /// 注入轮次预算（须在 `Arc::new` 之前；之后创建的会话继承该预算）。
    pub fn set_turn_budget(&mut self, budget: TurnBudget) {
        self.turn_budget = budget;
    }

    pub fn turn_budget(&self) -> TurnBudget {
        self.turn_budget
    }

    /// 注入质量护照索引（在 `Arc::new` 之前调用）。
    ///
    /// 索引只影响「怎么说」（出处/冲突披露/缺口追问），不改变真值与存储；
    /// 会话分叉时共享同一份 `Arc`，不复制整张查询表。
    pub fn set_quality_index(&mut self, index: Option<Arc<QualityIndex>>) {
        self.quality = index;
    }

    pub fn quality_index(&self) -> Option<&QualityIndex> {
        self.quality.as_deref()
    }

    /// 注入配置编译的绑定帧程序（在 `Arc::new` 之前调用）。
    /// 程序只读且跨会话共享，绑定记忆仍在各自 [`DialogueManager`] 内。
    pub fn set_frame_program(&mut self, program: Option<Arc<FrameProgram>>) {
        self.frame_program = program;
    }

    pub fn frame_program(&self) -> Option<&FrameProgram> {
        self.frame_program.as_deref()
    }

    /// 注入插件运行时（在 `Arc::new` 之前调用）。
    ///
    /// 运行时被所有会话共享：词表/本体/改写/配方/风格一致地作用于每个会话，
    /// 而对话记忆仍按会话隔离。注入时自动把已启用插件的本体贡献并入暖基座
    /// （幂等）；模式在注入时确定，运行中不提供 HTTP 热切换。
    pub fn set_plugin_runtime(&mut self, runtime: Option<Arc<parking_lot::Mutex<PluginRuntime>>>) {
        if let Some(runtime) = &runtime {
            self.configure_plugin_runtime(&runtime.lock());
        }
        self.plugin = runtime;
    }

    /// 当前引擎指纹（知识库指纹 + 运行模式）。
    pub fn engine_stamp(&self) -> EngineStamp {
        EngineStamp {
            crate_version: env!("CARGO_PKG_VERSION").to_string(),
            kb_fingerprint: self.kb_fingerprint.clone(),
            kb_facts: self.facts.len(),
            prompt_mode: self.prompt_mode.label().to_string(),
            recovery_mode: self.recovery_mode.label().to_string(),
        }
    }

    /// 启动一个会话 actor 线程；失败返回错误描述。
    fn spawn_actor(
        &self,
        id: String,
        restore: Option<SessionRestore>,
    ) -> Result<SessionHandle, &'static str> {
        let (tx, mut rx) = mpsc::channel::<SessionCommand>(64);
        let warm = Arc::clone(&self.warm);
        let perf = Arc::clone(&self.perf);
        let startup = Arc::clone(&self.startup);
        let prompt_mode = self.prompt_mode;
        let recovery_mode = self.recovery_mode;
        let quality = self.quality.clone();
        let plugin = self.plugin.clone();
        let frame_program = self.frame_program.clone();
        let turn_budget = self.turn_budget;
        let engine = Arc::new(self.engine_stamp());
        let thread_name = format!("kp-session-{}", id);
        std::thread::Builder::new()
            .name(thread_name)
            .spawn(move || {
                // 只计量「分叉到就绪」：guard 在 session 构造完成时立即落账，
                // 绝不能覆盖整个 actor 生命周期（那会拖到线程退出才记账）
                let mut session = {
                    let _span = startup.span("session_fork");
                    build_session(
                        id,
                        &warm,
                        &perf,
                        prompt_mode,
                        recovery_mode,
                        quality,
                        plugin,
                        frame_program,
                        turn_budget,
                        restore,
                    )
                };
                while let Some(command) = rx.blocking_recv() {
                    match command {
                        SessionCommand::Turn {
                            input,
                            policy,
                            cancel,
                            progress,
                            reply,
                        } => {
                            if let Some(policy) = policy {
                                session.policy = policy;
                            }
                            let response =
                                session.run_turn_with_cancel_and_progress(&input, cancel, progress);
                            let _ = reply.send(response);
                        }
                        SessionCommand::Resume { cancel, reply } => {
                            let response = session.resume_turn(cancel);
                            let _ = reply.send(response);
                        }
                        SessionCommand::DiscardOpenTurn { reply } => {
                            let _ = reply.send(session.manager.dismiss_open_turn());
                        }
                        SessionCommand::Snapshot { reply } => {
                            let _ = reply.send(serde_json::json!({
                                "session_id": session.id,
                                "turns": session.turns,
                                "focus": session.manager.focus_entity(),
                                "focus_stack": session.manager.focus_stack(),
                                "history": session.history,
                                "summary": session.manager.summary(),
                                "policy": session.policy,
                                "board": session.manager.board().view(session.reasoner.revision()),
                                "open_turn": session.manager.latest_open_turn(),
                                "resumable": session.open_turn_resumable(),
                                "poisoned": session.poisoned,
                            }));
                        }
                        SessionCommand::Export { reply } => {
                            let mut envelope = SessionEnvelope {
                                format: ENVELOPE_FORMAT.to_string(),
                                version: ENVELOPE_VERSION,
                                session_id: session.id.clone(),
                                turns: session.turns,
                                history: session.history.clone(),
                                policy: session.policy.clone(),
                                manager: session.manager.snapshot(),
                                reasoner: session.reasoner.dialogue_snapshot(),
                                engine: (*engine).clone(),
                                digest: String::new(),
                            };
                            envelope.seal();
                            let _ = reply.send(envelope);
                        }
                        SessionCommand::ReadTraces { reply } => {
                            let _ = reply.send(serde_json::json!({
                                "session_id": session.id,
                                "report": session.reasoner.read_trace_ledger_report(),
                                "receipts": session.reasoner.read_trace_receipts(),
                            }));
                        }
                        SessionCommand::Layers { reply } => {
                            let manifest = session.reasoner.knowledge_layer_manifest();
                            let _ = reply.send(serde_json::json!({
                                "session_id": session.id,
                                "turns": session.turns,
                                "manifest": manifest,
                            }));
                        }
                    }
                }
            })
            .map_err(|_| "failed to spawn session actor")?;
        Ok(SessionHandle {
            tx,
            last_active: Arc::new(AtomicU64::new(0)),
        })
    }

    /// 冷会话文件路径（id 已通过导入校验/服务端生成，再做防穿越过滤）。
    fn session_file(&self, session_id: &str) -> Option<std::path::PathBuf> {
        let dir = self.session_dir.as_ref()?;
        Some(dir.join(format!("{}.json", session_file_name(session_id))))
    }

    /// 把 LRU 会话冻结到磁盘并从内存移除（容量压力下的换出）。
    async fn freeze_lru_locked(
        &self,
        sessions: &mut BTreeMap<String, SessionHandle>,
    ) -> Result<(), SessionError> {
        let Some(id) = sessions
            .iter()
            .min_by_key(|(_, handle)| handle.last_active.load(Ordering::Relaxed))
            .map(|(id, _)| id.clone())
        else {
            return Err(SessionError::NotFound);
        };
        let envelope = {
            let tx = sessions
                .get(&id)
                .map(|handle| handle.tx.clone())
                .ok_or(SessionError::NotFound)?;
            let (reply_tx, reply_rx) = oneshot::channel();
            tx.send(SessionCommand::Export { reply: reply_tx })
                .await
                .map_err(|_| SessionError::Unavailable)?;
            reply_rx.await.map_err(|_| SessionError::Unavailable)?
        };
        let Some(path) = self.session_file(&id) else {
            return Err(SessionError::Unavailable);
        };
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|_| SessionError::Unavailable)?;
            }
        }
        let bytes = canonical::canonical_bytes(&envelope).map_err(|_| SessionError::Unavailable)?;
        // 先写临时文件再原子重命名：崩溃/并发下不会留下半个信封
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, bytes).map_err(|_| SessionError::Unavailable)?;
        std::fs::rename(&tmp, &path).map_err(|_| SessionError::Unavailable)?;
        sessions.remove(&id);
        self.evictions_total.fetch_add(1, Ordering::Relaxed);
        self.perf.bump("session_evictions");
        Ok(())
    }

    /// 尝试从磁盘反序列化恢复冷会话；返回是否已驻留内存。
    ///
    /// 解码走 [`SessionEnvelope::decode`] 宪法：失败带分类错误码并计入
    /// `decode_reject_<code>`；成功计入 `decode_ok`（非规范字节另计
    /// `decode_non_canonical`，供未来 `--fix` 审计）。
    async fn resume_from_disk(&self, session_id: &str) -> Result<bool, SessionError> {
        let Some(path) = self.session_file(session_id) else {
            return Ok(false);
        };
        let read_start = Instant::now();
        let Ok(bytes) = std::fs::read(&path) else {
            return Ok(false);
        };
        let (envelope, receipt) = match SessionEnvelope::decode(&bytes, SESSION_ENVELOPE_MAX_BYTES)
        {
            Ok(decoded) => decoded,
            Err(error) => {
                self.perf.bump(&error.metric_key());
                return Err(SessionError::Stale(error));
            }
        };
        self.startup.record_since("session_resume_read", read_start);
        if envelope.session_id != session_id {
            let error = DecodeError::Invariant(format!(
                "冷存储会话 id 不匹配：{} != {}",
                envelope.session_id, session_id
            ));
            self.perf.bump(&error.metric_key());
            return Err(SessionError::Stale(error));
        }
        let foreign = envelope.engine.mismatched_fields(&self.engine_stamp());
        if !foreign.is_empty() {
            let error = DecodeError::Fingerprint(foreign);
            self.perf.bump(&error.metric_key());
            return Err(SessionError::Stale(error));
        }
        self.perf.bump("decode_ok");
        if !receipt.canonical {
            self.perf.bump("decode_non_canonical");
        }
        let restore = SessionRestore {
            turns: envelope.turns,
            history: envelope.history,
            policy: envelope.policy,
            manager: envelope.manager,
            reasoner: envelope.reasoner,
        };
        let mut sessions = self.sessions.lock().await;
        if sessions.contains_key(session_id) {
            return Ok(true);
        }
        if sessions.len() >= self.max_sessions {
            self.freeze_lru_locked(&mut sessions).await?;
        }
        let handle = self
            .spawn_actor(session_id.to_string(), Some(restore))
            .map_err(|_| SessionError::Unavailable)?;
        sessions.insert(session_id.to_string(), handle);
        self.perf.bump("session_resumes");
        Ok(true)
    }

    /// 取会话命令通道：内存命中 → 触活；冷会话 → 反序列化换入。
    async fn session_tx(
        &self,
        session_id: &str,
    ) -> Result<mpsc::Sender<SessionCommand>, SessionError> {
        {
            let sessions = self.sessions.lock().await;
            if let Some(handle) = sessions.get(session_id) {
                handle.touch(&self.activity_clock);
                return Ok(handle.tx.clone());
            }
        }
        if self.resume_from_disk(session_id).await? {
            let sessions = self.sessions.lock().await;
            if let Some(handle) = sessions.get(session_id) {
                handle.touch(&self.activity_clock);
                return Ok(handle.tx.clone());
            }
        }
        Err(SessionError::NotFound)
    }

    async fn create_session(&self) -> Result<String, &'static str> {
        let mut sessions = self.sessions.lock().await;
        if sessions.len() >= self.max_sessions {
            if self.session_dir.is_none() {
                return Err("too many sessions");
            }
            self.freeze_lru_locked(&mut sessions)
                .await
                .map_err(|_| "failed to freeze least-recently-used session")?;
        }
        // 跳过已占用的 id：导入的信封可以使用任意 id（含 s-N），
        // 服务端生成的 id 绝不能覆盖已存在（尤其是刚导入）的会话。
        let id = loop {
            let candidate = format!("s-{}", self.next_session_id.fetch_add(1, Ordering::Relaxed));
            if !sessions.contains_key(&candidate) {
                break candidate;
            }
        };
        let handle = self.spawn_actor(id.clone(), None)?;
        sessions.insert(id.clone(), handle);
        Ok(id)
    }

    async fn take_turn(&self, session_id: &str, input: &str) -> Result<TurnResponse, SessionError> {
        self.take_turn_with_policy(session_id, input, None).await
    }

    /// 执行一轮；给出 `policy` 时先更新会话策略再执行。
    ///
    /// 收件箱满时立即拒绝（429），不无限排队；等待超时先触发协作取消再报 504。
    async fn take_turn_with_policy(
        &self,
        session_id: &str,
        input: &str,
        policy: Option<&SessionPolicy>,
    ) -> Result<TurnResponse, SessionError> {
        let tx = self.session_tx(session_id).await?;
        let cancel = CancelToken::new();
        let (reply_tx, reply_rx) = oneshot::channel();
        match tx.try_send(SessionCommand::Turn {
            input: input.to_string(),
            policy: policy.cloned(),
            cancel: cancel.clone(),
            progress: None,
            reply: reply_tx,
        }) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(_)) => return Err(SessionError::Busy),
            Err(mpsc::error::TrySendError::Closed(_)) => return Err(SessionError::Unavailable),
        }
        match tokio::time::timeout(TURN_WAIT_TIMEOUT, reply_rx).await {
            Ok(Ok(response)) => {
                self.turns_total.fetch_add(1, Ordering::Relaxed);
                Ok(response)
            }
            Ok(Err(_)) => Err(SessionError::Unavailable),
            Err(_) => {
                cancel.cancel();
                Err(SessionError::Timeout)
            }
        }
    }

    /// 继续会话里最近的未完成轮次（消费一次；失败后 attempts+1 重建）。
    async fn take_resume(&self, session_id: &str) -> Result<TurnResponse, SessionError> {
        let tx = self.session_tx(session_id).await?;
        let cancel = CancelToken::new();
        let (reply_tx, reply_rx) = oneshot::channel();
        match tx.try_send(SessionCommand::Resume {
            cancel: cancel.clone(),
            reply: reply_tx,
        }) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(_)) => return Err(SessionError::Busy),
            Err(mpsc::error::TrySendError::Closed(_)) => return Err(SessionError::Unavailable),
        }
        match tokio::time::timeout(TURN_WAIT_TIMEOUT, reply_rx).await {
            Ok(Ok(Ok(turn))) => {
                self.turns_total.fetch_add(1, Ordering::Relaxed);
                Ok(turn)
            }
            Ok(Ok(Err(error))) => Err(error),
            Ok(Err(_)) => Err(SessionError::Unavailable),
            Err(_) => {
                cancel.cancel();
                Err(SessionError::Timeout)
            }
        }
    }

    /// 显式放弃最新未完成轮次；返回被放弃的轮次（没有则 `None`）。
    async fn discard_open_turn(&self, session_id: &str) -> Result<Option<OpenTurn>, SessionError> {
        let tx = self.session_tx(session_id).await?;
        let (reply_tx, reply_rx) = oneshot::channel();
        tx.send(SessionCommand::DiscardOpenTurn { reply: reply_tx })
            .await
            .map_err(|_| SessionError::Unavailable)?;
        reply_rx.await.map_err(|_| SessionError::Unavailable)
    }

    /// 启动一轮并返回阶段进度接收端；调用方负责等待最终回执并计量轮次。
    async fn take_turn_stream(
        &self,
        session_id: &str,
        input: &str,
    ) -> Result<
        (
            mpsc::UnboundedReceiver<TurnPhase>,
            oneshot::Receiver<TurnResponse>,
            CancelToken,
        ),
        SessionError,
    > {
        let tx = self.session_tx(session_id).await?;
        let cancel = CancelToken::new();
        let (progress_tx, progress_rx) = mpsc::unbounded_channel();
        let (reply_tx, reply_rx) = oneshot::channel();
        match tx.try_send(SessionCommand::Turn {
            input: input.to_string(),
            policy: None,
            cancel: cancel.clone(),
            progress: Some(progress_tx),
            reply: reply_tx,
        }) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(_)) => return Err(SessionError::Busy),
            Err(mpsc::error::TrySendError::Closed(_)) => return Err(SessionError::Unavailable),
        }
        Ok((progress_rx, reply_rx, cancel))
    }

    async fn session_snapshot(&self, session_id: &str) -> Result<serde_json::Value, SessionError> {
        let tx = self.session_tx(session_id).await?;
        let (reply_tx, reply_rx) = oneshot::channel();
        tx.send(SessionCommand::Snapshot { reply: reply_tx })
            .await
            .map_err(|_| SessionError::Unavailable)?;
        reply_rx.await.map_err(|_| SessionError::Unavailable)
    }

    /// 四层知识容器清单（基础 / 会话 / 推测 / 假设）。
    async fn session_layers(&self, session_id: &str) -> Result<serde_json::Value, SessionError> {
        let tx = self.session_tx(session_id).await?;
        let (reply_tx, reply_rx) = oneshot::channel();
        tx.send(SessionCommand::Layers { reply: reply_tx })
            .await
            .map_err(|_| SessionError::Unavailable)?;
        reply_rx.await.map_err(|_| SessionError::Unavailable)
    }

    /// 会话级读法回执：只含 digest/计数/有限枚举，不返回原始问句。
    async fn read_traces(&self, session_id: &str) -> Result<serde_json::Value, SessionError> {
        let tx = self.session_tx(session_id).await?;
        let (reply_tx, reply_rx) = oneshot::channel();
        tx.send(SessionCommand::ReadTraces { reply: reply_tx })
            .await
            .map_err(|_| SessionError::Unavailable)?;
        reply_rx.await.map_err(|_| SessionError::Unavailable)
    }

    /// 导出会话信封（在 actor 内取一致快照）。
    async fn export_session(&self, session_id: &str) -> Result<SessionEnvelope, SessionError> {
        let tx = self.session_tx(session_id).await?;
        let (reply_tx, reply_rx) = oneshot::channel();
        tx.send(SessionCommand::Export { reply: reply_tx })
            .await
            .map_err(|_| SessionError::Unavailable)?;
        reply_rx.await.map_err(|_| SessionError::Unavailable)
    }

    /// 导入结果：新会话 id + 被显式接收的陌生字段回执 + 解码回执。
    async fn import_session(
        &self,
        restore: SessionRestore,
        session_id: String,
        foreign_fields: Vec<String>,
        receipt: &DecodeReceipt,
    ) -> Result<serde_json::Value, ImportError> {
        let mut sessions = self.sessions.lock().await;
        if sessions.contains_key(&session_id) {
            return Err(ImportError::Conflict(format!(
                "session {} already exists",
                session_id
            )));
        }
        // 导入 id 形如 s-N 时把自增游标推过 N：后续 create_session 不会撞上它
        if let Some(number) = session_id
            .strip_prefix("s-")
            .and_then(|suffix| suffix.parse::<u64>().ok())
        {
            self.next_session_id
                .fetch_max(number.saturating_add(1), Ordering::Relaxed);
        }
        if sessions.len() >= self.max_sessions {
            if self.session_dir.is_none() {
                return Err(ImportError::TooMany);
            }
            self.freeze_lru_locked(&mut sessions)
                .await
                .map_err(|_| ImportError::Conflict("failed to freeze session".to_string()))?;
        }
        let handle = self
            .spawn_actor(session_id.clone(), Some(restore))
            .map_err(|message| ImportError::Conflict(message.to_string()))?;
        sessions.insert(session_id.clone(), handle);
        Ok(serde_json::json!({
            "object": "session.imported",
            "session_id": session_id,
            "foreign_accepted": !foreign_fields.is_empty(),
            "foreign_fields": foreign_fields,
            "decode": {
                "code": "decode_ok",
                "kind": receipt.kind,
                "version": receipt.version,
                "canonical": receipt.canonical,
                "migrations": receipt.migrations,
            },
        }))
    }

    async fn close_session(&self, session_id: &str) -> bool {
        let removed = self.sessions.lock().await.remove(session_id).is_some();
        match self.session_file(session_id) {
            // 冷会话（已换出）也必须可删除，否则会被下次访问复活
            Some(path) => {
                let file_removed = std::fs::remove_file(&path).is_ok();
                removed || file_removed
            }
            None => removed,
        }
    }

    async fn session_count(&self) -> usize {
        self.sessions.lock().await.len()
    }

    /// 冷存储中的会话数（`.json` 信封文件计数，供 `/metrics` 观测换出水位）。
    fn cold_session_count(&self) -> usize {
        let Some(dir) = &self.session_dir else {
            return 0;
        };
        std::fs::read_dir(dir)
            .map(|entries| {
                entries
                    .filter_map(|entry| entry.ok())
                    .filter(|entry| {
                        entry
                            .path()
                            .extension()
                            .map(|ext| ext == "json")
                            .unwrap_or(false)
                    })
                    .count()
            })
            .unwrap_or(0)
    }
}

/// 导入失败的具体原因（映射到 HTTP 状态与错误码）。
#[derive(Debug)]
enum ImportError {
    BadRequest(String),
    Conflict(String),
    TooMany,
}

/// 启动服务（阻塞直到监听套接字出错）。
///
/// 套接字绑定完成后打印启动分期摘要（stdout 一行）：进程启动各相耗时 +
/// 已在建会话（若有）。计时只观测，不影响服务行为。
pub async fn serve(state: Arc<ApiState>, addr: &str) -> std::io::Result<()> {
    let bind_start = Instant::now();
    let listener = TcpListener::bind(addr).await?;
    state.startup.record_since("bind", bind_start);
    if let Some(line) = state.startup.summary_line() {
        println!("{}", line);
    }
    run(listener, state).await
}

/// 在已绑定的监听器上运行（测试可直接绑定 `127.0.0.1:0`）。
///
/// [`ApiState`] 是 `Send + Sync`（会话在专属 actor 线程内），因此每个连接
/// 独立 `tokio::spawn`：跨会话真并行、同会话由 actor 串行化；每个连接仍有
/// [`CONNECTION_TIMEOUT`] 兜底，慢客户端不会阻塞新连接。
pub async fn run(listener: TcpListener, state: Arc<ApiState>) -> std::io::Result<()> {
    loop {
        let (stream, _peer) = listener.accept().await?;
        let state = Arc::clone(&state);
        tokio::spawn(async move {
            let _ = tokio::time::timeout(CONNECTION_TIMEOUT, serve_connection(stream, state)).await;
        });
    }
}

struct Request {
    method: String,
    path: String,
    query: BTreeMap<String, String>,
    headers: BTreeMap<String, String>,
    body: Vec<u8>,
}

impl Request {
    fn wants_sse(&self) -> bool {
        self.headers
            .get("accept")
            .map(|value| value.contains("text/event-stream"))
            .unwrap_or(false)
            || self
                .query
                .get("stream")
                .map(|value| value == "1" || value == "true")
                .unwrap_or(false)
    }

    fn body_text(&self) -> String {
        String::from_utf8_lossy(&self.body).to_string()
    }
}

struct Response {
    status: u16,
    content_type: &'static str,
    body: Vec<u8>,
}

impl Response {
    fn json(status: u16, value: &serde_json::Value) -> Self {
        Response {
            status,
            content_type: "application/json; charset=utf-8",
            body: serde_json::to_vec(value).unwrap_or_else(|_| b"{}".to_vec()),
        }
    }

    fn error(status: u16, message: &str) -> Self {
        Response::json(status, &serde_json::json!({"error": message}))
    }
}

async fn serve_connection(mut stream: TcpStream, state: Arc<ApiState>) -> std::io::Result<()> {
    let request = match read_request(&mut stream, state.max_body_bytes).await {
        Ok(request) => request,
        Err(RequestReadError::BodyTooLarge) => {
            write_response(&mut stream, Response::error(413, "body too large")).await?;
            return Ok(());
        }
        Err(RequestReadError::Malformed(message)) => {
            write_response(&mut stream, Response::error(400, &message)).await?;
            return Ok(());
        }
    };
    if request.wants_sse() && request.method == "POST" {
        if let Some(session_id) = turn_route(&request.path) {
            return stream_turn(&mut stream, &state, &session_id, &request.body_text()).await;
        }
    }
    if request.method == "POST" && request.path == "/v1/chat/completions" {
        match parse_chat_request(&request) {
            Ok(chat) if chat.stream => {
                return stream_chat_completion(&mut stream, &state, &chat).await;
            }
            Ok(_) => {}
            Err(message) => {
                write_response(&mut stream, openai_error(400, &message)).await?;
                return Ok(());
            }
        }
    }
    let response = route(&state, &request).await;
    write_response(&mut stream, response).await
}

/// `POST /sessions/{id}/turns` → Some(id)。
fn turn_route(path: &str) -> Option<String> {
    let segments: Vec<&str> = path
        .split('/')
        .filter(|segment| !segment.is_empty())
        .collect();
    match segments.as_slice() {
        ["sessions", id, "turns"] => Some((*id).to_string()),
        _ => None,
    }
}

async fn route(state: &Arc<ApiState>, request: &Request) -> Response {
    let segments: Vec<&str> = request
        .path
        .split('/')
        .filter(|segment| !segment.is_empty())
        .collect();
    match (request.method.as_str(), segments.as_slice()) {
        ("GET", []) => Response::json(
            200,
            &serde_json::json!({
                "service": "knowprolog",
                "version": env!("CARGO_PKG_VERSION"),
                "endpoints": [
                    "GET /health",
                    "GET /metrics",
                    "POST /sessions",
                    "GET /sessions/{id}",
                    "GET /sessions/{id}/layers",
                    "GET /sessions/{id}/export",
                    "GET /sessions/{id}/read-traces",
                    "POST /sessions/import",
                    "DELETE /sessions/{id}",
                    "POST /sessions/{id}/turns",
                    "POST /sessions/{id}/resume",
                    "DELETE /sessions/{id}/open-turn",
                    "GET /v1/models",
                    "GET /v1/prompts",
                    "POST /v1/events",
                    "POST /v1/chat/completions"
                ]
            }),
        ),
        ("GET", ["health"]) => Response::json(
            200,
            &serde_json::json!({
                "status": "ok",
                "version": env!("CARGO_PKG_VERSION"),
                "sessions": state.session_count().await,
                "turns": state.turns_total.load(Ordering::Relaxed),
                "quality": state.quality_index().is_some(),
            }),
        ),
        ("GET", ["metrics"]) => Response::json(
            200,
            &serde_json::json!({
                "sessions": state.session_count().await,
                "turns_total": state.turns_total.load(Ordering::Relaxed),
                "quality": state.quality_index().map(|index| serde_json::json!({
                    "entities": index.len(),
                    "revision": index.revision,
                    "digest": index.digest,
                })),
                "max_sessions": state.max_sessions,
                "session_dir": state.session_dir.as_ref().map(|dir| dir.display().to_string()),
                "cold_sessions": state.cold_session_count(),
                "evictions_total": state.evictions_total.load(Ordering::Relaxed),
                "warm_revision": state.warm_revision(),
                "perf": state.perf.snapshot(),
                "startup": state.startup.snapshot(),
            }),
        ),
        ("POST", ["sessions"]) => match state.create_session().await {
            Ok(session_id) => Response::json(201, &serde_json::json!({"session_id": session_id})),
            Err(message) => Response::error(429, message),
        },
        ("POST", ["sessions", "import"]) => handle_import(state, request).await,
        ("POST", ["v1", "events"]) => handle_events(request).await,
        ("GET", ["sessions", id]) => match state.session_snapshot(id).await {
            Ok(snapshot) => Response::json(200, &snapshot),
            Err(error) => session_error_response(error),
        },
        ("GET", ["sessions", id, "layers"]) => match state.session_layers(id).await {
            Ok(layers) => Response::json(200, &layers),
            Err(error) => session_error_response(error),
        },
        ("GET", ["sessions", id, "export"]) => match state.export_session(id).await {
            Ok(envelope) => Response::json(
                200,
                &serde_json::to_value(&envelope).unwrap_or_else(|_| serde_json::json!({})),
            ),
            Err(error) => session_error_response(error),
        },
        ("GET", ["sessions", id, "read-traces"]) => match state.read_traces(id).await {
            Ok(report) => Response::json(200, &report),
            Err(error) => session_error_response(error),
        },
        ("DELETE", ["sessions", id]) => {
            if state.close_session(id).await {
                Response::json(200, &serde_json::json!({"closed": id}))
            } else {
                Response::error(404, "session not found")
            }
        }
        ("POST", ["sessions", id, "turns"]) => {
            let input = match parse_turn_input(request) {
                Ok(input) => input,
                Err(message) => return Response::error(400, &message),
            };
            match state.take_turn(id, &input).await {
                Ok(turn) => Response::json(
                    200,
                    &serde_json::to_value(&turn).unwrap_or_else(|_| serde_json::json!({})),
                ),
                Err(error) => session_error_response(error),
            }
        }
        ("POST", ["sessions", id, "resume"]) => match state.take_resume(id).await {
            Ok(turn) => Response::json(
                200,
                &serde_json::to_value(&turn).unwrap_or_else(|_| serde_json::json!({})),
            ),
            Err(error) => session_error_response(error),
        },
        ("DELETE", ["sessions", id, "open-turn"]) => match state.discard_open_turn(id).await {
            Ok(discarded) => Response::json(
                200,
                &serde_json::json!({
                    "discarded": discarded.is_some(),
                    "open_turn": discarded,
                }),
            ),
            Err(error) => session_error_response(error),
        },
        ("GET", ["v1", "models"]) => Response::json(
            200,
            &serde_json::json!({
                "object": "list",
                "data": [{
                    "id": DEFAULT_MODEL,
                    "object": "model",
                    "created": 0,
                    "owned_by": "knowprolog",
                }]
            }),
        ),
        ("GET", ["v1", "prompts"]) => {
            let specs = prompt_spec::default_specs();
            let lint = prompt_spec::lint_default();
            let errors = lint
                .iter()
                .filter(|issue| issue.severity == prompt_spec::LintSeverity::Error)
                .count();
            let (covered, composed) = prompt_spec::coverage();
            let spec_json: Vec<serde_json::Value> = specs
                .iter()
                .map(|spec| {
                    serde_json::json!({
                        "id": spec.id,
                        "policy": spec.policy,
                        "template": spec.template,
                        "required": spec.required,
                        "optional": spec.optional,
                        "example_rendered": spec.example_rendered,
                        "description": spec.description,
                    })
                })
                .collect();
            Response::json(
                200,
                &serde_json::json!({
                    "object": "list",
                    "version": prompt_spec::PROMPT_SPEC_VERSION,
                    "modes": ["off", "parity", "on"],
                    "serving_mode": state.prompt_mode.label(),
                    "count": specs.len(),
                    "data": spec_json,
                    "lint": {
                        "errors": errors,
                        "issues": lint,
                    },
                    "coverage": {
                        "covered": covered,
                        "composed": composed,
                    },
                }),
            )
        }
        ("POST", ["v1", "chat", "completions"]) => {
            let chat = match parse_chat_request(request) {
                Ok(chat) => chat,
                Err(message) => return openai_error(400, &message),
            };
            match run_chat_turns(state, &chat).await {
                Ok((turn, prompt_chars)) => {
                    Response::json(200, &chat_completion_response(&chat, &turn, prompt_chars))
                }
                Err((status, message)) => openai_error(status, &message),
            }
        }
        (_method, _) => Response::error(404, "not found"),
    }
}

fn session_error_response(error: SessionError) -> Response {
    match error {
        SessionError::NotFound => Response::error(404, "session not found"),
        SessionError::Unavailable => Response::error(500, "session actor unavailable"),
        SessionError::Busy => Response::error(429, "session busy; retry shortly"),
        SessionError::Timeout => Response::error(504, "turn deadline exceeded; cancellation requested"),
        SessionError::NoOpenTurn => Response::error(
            409,
            "no resumable turn (none, expired, or attempts exhausted)",
        ),
        SessionError::Stale(error) => Response::error(
            409,
            &format!(
                "session snapshot is stale or from a different engine (decode_{}); re-import it explicitly",
                error.code()
            ),
        ),
    }
}

/// 解码失败 → HTTP 响应：保留人类可读措辞，附加稳定错误码。
fn import_decode_response(state: &Arc<ApiState>, error: &DecodeError) -> Response {
    state.perf.bump(&error.metric_key());
    match error {
        DecodeError::TooLarge { .. } => {
            Response::error(413, "envelope too large [decode_too_large]")
        }
        DecodeError::Parse(message) => Response::error(
            400,
            &format!("invalid envelope JSON: {} [decode_parse]", message),
        ),
        DecodeError::Format(formatted) => Response::error(
            400,
            &format!("unknown envelope format: {} [decode_format]", formatted),
        ),
        DecodeError::Version { found, supported } => Response::error(
            409,
            &format!(
                "unsupported envelope version: {} (expected {}) [decode_version]",
                found, supported
            ),
        ),
        DecodeError::Digest { .. } => Response::error(
            409,
            "envelope digest mismatch (tampered or truncated) [decode_digest]",
        ),
        other => Response::error(
            400,
            &format!(
                "envelope decode failed: {} [decode_{}]",
                other,
                other.code()
            ),
        ),
    }
}

/// 导入信封：解码（解析/格式/版本/摘要/尺寸）→ 校验（会话 id/引擎）→
/// 零副作用重建会话。解码回执随 201 响应返回（可审计的接收凭据）。
/// 无状态事件抽取/问答：`POST /v1/events`。
///
/// 请求：`{"text":"...","question":"..."}`；`question` 可省略，省略时只返回事件图。
/// 事件图按请求现算，不写入会话或默认 RDF 图；N-Quads 需调用 CLI `events --nquads`。
async fn handle_events(request: &Request) -> Response {
    let payload: serde_json::Value = match serde_json::from_str(&request.body_text()) {
        Ok(value) => value,
        Err(error) => return Response::error(400, &format!("invalid JSON: {error}")),
    };
    let Some(text) = payload.get("text").and_then(serde_json::Value::as_str) else {
        return Response::error(400, "missing string field: text");
    };
    if text.trim().is_empty() {
        return Response::error(400, "text must not be empty");
    }
    let graph = EventGraph::build(text);
    let Some(question) = payload.get("question") else {
        return Response::json(200, &serde_json::json!({"status": "ok", "graph": graph}));
    };
    let Some(question) = question.as_str() else {
        return Response::error(400, "question must be a string");
    };
    if question.trim().is_empty() {
        return Response::error(400, "question must not be empty");
    }
    let mut dialogue = EventDialogue::new(graph.clone());
    let answer = dialogue.ask(question);
    Response::json(
        200,
        &serde_json::json!({
            "status": answer.status.tag(),
            "answer": answer,
            "graph": graph,
        }),
    )
}

async fn handle_import(state: &Arc<ApiState>, request: &Request) -> Response {
    // 信封解码一律用信封上限：导出的会话必须能原样导回，
    // 默认 256KB 体量只约束对话轮，不约束冷存储信封。
    let (envelope, receipt) =
        match SessionEnvelope::decode(&request.body, SESSION_ENVELOPE_MAX_BYTES) {
            Ok(decoded) => decoded,
            Err(error) => return import_decode_response(state, &error),
        };
    state.perf.bump("decode_ok");
    if !receipt.canonical {
        state.perf.bump("decode_non_canonical");
    }
    let accept_foreign = request
        .query
        .get("accept_foreign")
        .map(|value| value == "1" || value == "true")
        .unwrap_or(false);
    match validate_envelope(state, envelope, accept_foreign) {
        Ok((restore, session_id, foreign_fields)) => {
            match state
                .import_session(restore, session_id, foreign_fields, &receipt)
                .await
            {
                Ok(body) => Response::json(201, &body),
                Err(ImportError::BadRequest(message)) => Response::error(400, &message),
                Err(ImportError::Conflict(message)) => Response::error(409, &message),
                Err(ImportError::TooMany) => Response::error(429, "too many sessions"),
            }
        }
        Err(ImportError::BadRequest(message)) => Response::error(400, &message),
        Err(ImportError::Conflict(message)) => Response::error(409, &message),
        Err(ImportError::TooMany) => Response::error(429, "too many sessions"),
    }
}

/// 导入校验：会话 id 合法性 + 引擎指纹（解码门已由 [`SessionEnvelope::decode`] 完成）。
///
/// 任一项失败都**不**产生任何副作用（会话表不变）。
fn validate_envelope(
    state: &ApiState,
    envelope: SessionEnvelope,
    accept_foreign: bool,
) -> Result<(SessionRestore, String, Vec<String>), ImportError> {
    if envelope.session_id.trim().is_empty() || envelope.session_id.len() > 128 {
        return Err(ImportError::BadRequest(
            "envelope session_id must be 1..=128 bytes".to_string(),
        ));
    }
    if !envelope
        .session_id
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '_')
    {
        return Err(ImportError::BadRequest(
            "envelope session_id may only contain [A-Za-z0-9_-]".to_string(),
        ));
    }
    let foreign_fields = envelope.engine.mismatched_fields(&state.engine_stamp());
    if !foreign_fields.is_empty() && !accept_foreign {
        return Err(ImportError::Conflict(format!(
            "foreign envelope (engine mismatch: {}); retry with ?accept_foreign=1 to import",
            foreign_fields.join(", ")
        )));
    }
    let restore = SessionRestore {
        turns: envelope.turns,
        history: envelope.history,
        policy: envelope.policy,
        manager: envelope.manager,
        reasoner: envelope.reasoner,
    };
    Ok((restore, envelope.session_id, foreign_fields))
}

/// OpenAI 兼容的聊天请求（`session_id` 为本服务的扩展字段）。
struct ChatRequest {
    model: String,
    /// 提供时复用该服务端会话，只执行最后一条用户消息。
    session_id: Option<String>,
    /// 无状态模式下的用户消息序列（按顺序重放到一个临时会话）。
    inputs: Vec<String>,
    /// 由 `system` / `developer` 消息解析出的会话级策略（无则使用默认）。
    policy: Option<SessionPolicy>,
    stream: bool,
}

fn parse_chat_request(request: &Request) -> Result<ChatRequest, String> {
    let value: serde_json::Value = serde_json::from_slice(&request.body)
        .map_err(|error| format!("invalid JSON: {}", error))?;
    let messages = value
        .get("messages")
        .and_then(|messages| messages.as_array())
        .ok_or_else(|| "missing messages array".to_string())?;
    let mut inputs = Vec::new();
    let mut system_texts = Vec::new();
    for message in messages {
        let role = message
            .get("role")
            .and_then(|role| role.as_str())
            .unwrap_or("");
        if role != "user" && role != "system" && role != "developer" {
            continue;
        }
        let content = message
            .get("content")
            .and_then(|content| content.as_str())
            .ok_or_else(|| "message content must be a string".to_string())?;
        if content.trim().is_empty() {
            continue;
        }
        if role == "user" {
            inputs.push(content.to_string());
        } else {
            system_texts.push(content.to_string());
        }
    }
    if inputs.is_empty() {
        return Err("no non-empty user messages".to_string());
    }
    Ok(ChatRequest {
        model: value
            .get("model")
            .and_then(|model| model.as_str())
            .unwrap_or(DEFAULT_MODEL)
            .to_string(),
        session_id: value
            .get("session_id")
            .and_then(|session| session.as_str())
            .map(str::to_string),
        inputs,
        policy: if system_texts.is_empty() {
            None
        } else {
            Some(SessionPolicy::from_system_texts(&system_texts))
        },
        stream: value
            .get("stream")
            .and_then(|stream| stream.as_bool())
            .unwrap_or(false),
    })
}

/// 执行聊天请求，返回最后一轮结果与提示字符数（用作 token 近似）。
///
/// - 带 `session_id`：只执行最后一条用户消息（服务端持有记忆）；
/// - 无 `session_id`：建临时会话并按顺序重放全部用户消息，然后关闭会话。
async fn run_chat_turns(
    state: &Arc<ApiState>,
    chat: &ChatRequest,
) -> Result<(TurnResponse, usize), (u16, String)> {
    let prompt_chars: usize = chat.inputs.iter().map(|input| input.chars().count()).sum();
    if let Some(session_id) = &chat.session_id {
        let input = chat
            .inputs
            .last()
            .expect("inputs non-empty by construction");
        return match state
            .take_turn_with_policy(session_id, input, chat.policy.as_ref())
            .await
        {
            Ok(turn) => Ok((turn, prompt_chars)),
            Err(error) => Err(session_error_http(error)),
        };
    }
    // 无状态路径是纯函数：键完全相等才命中缓存（纪元不符一律未命中）。
    // 命中时不建会话（临时会话本就随响应关闭，id 不可续用），
    // 只发一个新 id 供响应回执；响应载荷（回答/焦点/回执）来自缓存。
    let epoch = state.cache_epoch();
    let key = stateless_answer_key(&chat.inputs, chat.policy.as_ref());
    if let Some(mut cached) = state.cached_stateless_answer(epoch, &key) {
        cached.session_id = format!(
            "s-{}",
            state.next_session_id.fetch_add(1, Ordering::Relaxed)
        );
        return Ok((cached, prompt_chars));
    }
    let temp_id = state
        .create_session()
        .await
        .map_err(|message| (429, message.to_string()))?;
    let mut last = None;
    let mut failure: Option<(u16, String)> = None;
    for input in &chat.inputs {
        match state
            .take_turn_with_policy(&temp_id, input, chat.policy.as_ref())
            .await
        {
            Ok(turn) => last = Some(turn),
            // 临时会话刚建好就不可用，说明 actor 已退出；保留分类而不是折叠成
            // 无区分度的 "turn failed"，让无状态路径与会话路径的错误保真一致。
            Err(error) => {
                failure = Some(session_error_http(error));
                break;
            }
        }
    }
    state.close_session(&temp_id).await;
    if let Some(error) = failure {
        return Err(error);
    }
    let turn = last.ok_or_else(|| (500, "turn failed".to_string()))?;
    // 只缓存纯函数结果：无会话状态残留，命中与重算逐字节一致
    state.store_stateless_answer(epoch, key, turn.clone());
    Ok((turn, prompt_chars))
}

/// 会话访问错误的 HTTP 归类：会话路径与无状态路径共用同一张表，
/// 避免两处各自硬编码导致同一原因返回不同状态码/措辞。
fn session_error_http(error: SessionError) -> (u16, String) {
    match error {
        SessionError::NotFound => (404, "session not found".to_string()),
        SessionError::Unavailable => (500, "session actor unavailable".to_string()),
        SessionError::Busy => (429, "session busy; retry shortly".to_string()),
        SessionError::Timeout => (
            504,
            "turn deadline exceeded; cancellation requested".to_string(),
        ),
        SessionError::NoOpenTurn => (
            409,
            "no resumable turn (none, expired, or attempts exhausted)".to_string(),
        ),
        SessionError::Stale(error) => (
            409,
            format!("session snapshot is stale (decode_{})", error.code()),
        ),
    }
}

fn completion_id(session_id: &str, turn: u32) -> String {
    format!("chatcmpl-{}-{}", session_id, turn)
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

fn chat_completion_response(
    chat: &ChatRequest,
    turn: &TurnResponse,
    prompt_chars: usize,
) -> serde_json::Value {
    let completion_chars = turn.reply.chars().count();
    serde_json::json!({
        "id": completion_id(&turn.session_id, turn.turn),
        "object": "chat.completion",
        "created": now_secs(),
        "model": chat.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": turn.reply},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_chars,
            "completion_tokens": completion_chars,
            "total_tokens": prompt_chars + completion_chars,
        },
        "knowprolog": {
            "session_id": turn.session_id,
            "turn": turn.turn,
            "status": turn.status,
            "source": turn.source,
            "kind": turn.kind,
            "plan": turn.plan,
            "focus": turn.focus,
            "policy": turn.policy,
            "entities": turn.entities,
            "lexical": turn.lexical,
        }
    })
}

fn openai_error(status: u16, message: &str) -> Response {
    let kind = match status {
        429 => "rate_limit_error",
        500 => "server_error",
        _ => "invalid_request_error",
    };
    Response::json(
        status,
        &serde_json::json!({
            "error": {
                "message": message,
                "type": kind,
                "code": status,
            }
        }),
    )
}

/// OpenAI 兼容流式补全：首块 role、按句 content、末块 finish_reason、`[DONE]`。
async fn stream_chat_completion(
    stream: &mut TcpStream,
    state: &Arc<ApiState>,
    chat: &ChatRequest,
) -> std::io::Result<()> {
    let (turn, _) = match run_chat_turns(state, chat).await {
        Ok(result) => result,
        Err((status, message)) => {
            write_response(stream, openai_error(status, &message)).await?;
            return Ok(());
        }
    };
    let id = completion_id(&turn.session_id, turn.turn);
    let created = now_secs();
    let header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
    stream.write_all(header.as_bytes()).await?;

    let base = |delta: serde_json::Value, finish: Option<&str>| {
        serde_json::json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": chat.model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish,
            }],
        })
    };
    let mut send = async |value: &serde_json::Value| -> std::io::Result<()> {
        stream
            .write_all(format!("data: {}\n\n", value).as_bytes())
            .await
    };
    send(&base(serde_json::json!({"role": "assistant"}), None)).await?;
    for chunk in sentence_chunks(&turn.reply) {
        send(&base(serde_json::json!({"content": chunk}), None)).await?;
    }
    send(&base(serde_json::json!({}), Some("stop"))).await?;
    stream.write_all(b"data: [DONE]\n\n").await?;
    stream.flush().await?;
    let _ = stream.shutdown().await;
    Ok(())
}

/// 轮次请求体：JSON（`input` / `message` / `text` / `prompt`）或纯文本。
fn parse_turn_input(request: &Request) -> Result<String, String> {
    let body = request.body_text();
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return Err("empty body: expected {\"input\": \"...\"} or plain text".to_string());
    }
    let input = if trimmed.starts_with('{') {
        let value: serde_json::Value =
            serde_json::from_str(trimmed).map_err(|error| format!("invalid JSON: {}", error))?;
        ["input", "message", "text", "prompt"]
            .iter()
            .find_map(|key| value.get(key).and_then(|value| value.as_str()))
            .ok_or_else(|| "missing string field: input".to_string())?
            .to_string()
    } else {
        trimmed.to_string()
    };
    if input.trim().is_empty() {
        return Err("empty input".to_string());
    }
    Ok(input)
}

/// SSE：按句流式输出符号版「token streaming」。
async fn stream_turn(
    stream: &mut TcpStream,
    state: &Arc<ApiState>,
    session_id: &str,
    body: &str,
) -> std::io::Result<()> {
    let request = Request {
        method: "POST".to_string(),
        path: format!("/sessions/{}/turns", session_id),
        query: BTreeMap::new(),
        headers: BTreeMap::new(),
        body: body.as_bytes().to_vec(),
    };
    let input = match parse_turn_input(&request) {
        Ok(input) => input,
        Err(message) => {
            write_response(stream, Response::error(400, &message)).await?;
            return Ok(());
        }
    };
    let (mut progress, mut reply, cancel) = match state.take_turn_stream(session_id, &input).await {
        Ok(parts) => parts,
        Err(error) => {
            write_response(stream, session_error_response(error)).await?;
            return Ok(());
        }
    };

    let header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
    stream.write_all(header.as_bytes()).await?;
    let deadline = tokio::time::sleep(TURN_WAIT_TIMEOUT);
    tokio::pin!(deadline);
    let mut progress_open = true;
    let turn = loop {
        tokio::select! {
            phase = progress.recv(), if progress_open => {
                match phase {
                    Some(phase) => {
                        let event = serde_json::json!({
                            "session_id": session_id,
                            "phase": phase.code(),
                            "phase_label": phase.label(),
                        });
                        if let Err(error) = stream
                            .write_all(format!("data: {}\n\n", event).as_bytes())
                            .await
                        {
                            cancel.cancel();
                            return Err(error);
                        }
                    }
                    None => progress_open = false,
                }
            }
            result = &mut reply => {
                match result {
                    Ok(turn) => break turn,
                    Err(_) => {
                        let event = serde_json::json!({
                            "session_id": session_id,
                            "done": true,
                            "error": "session unavailable",
                        });
                        stream
                            .write_all(
                                format!("data: {}\n\ndata: [DONE]\n\n", event).as_bytes(),
                            )
                            .await?;
                        stream.flush().await?;
                        let _ = stream.shutdown().await;
                        return Ok(());
                    }
                }
            }
            _ = &mut deadline => {
                cancel.cancel();
                let event = serde_json::json!({
                    "session_id": session_id,
                    "done": true,
                    "error": "turn timeout",
                    "status": "timeout",
                });
                stream
                    .write_all(format!("data: {}\n\ndata: [DONE]\n\n", event).as_bytes())
                    .await?;
                stream.flush().await?;
                let _ = stream.shutdown().await;
                return Ok(());
            }
        }
    };
    // 回执可能先于进度事件被 select 选中：先把已排队的阶段事件全部补发，
    // 保证客户端总能在正文之前看到 prepare/rewrite/... 的阶段回执。
    while let Ok(phase) = progress.try_recv() {
        let event = serde_json::json!({
            "session_id": session_id,
            "phase": phase.code(),
            "phase_label": phase.label(),
        });
        stream
            .write_all(format!("data: {}\n\n", event).as_bytes())
            .await?;
    }
    state.turns_total.fetch_add(1, Ordering::Relaxed);
    for chunk in sentence_chunks(&turn.reply) {
        let event = serde_json::json!({
            "session_id": turn.session_id,
            "turn": turn.turn,
            "delta": chunk,
        });
        stream
            .write_all(format!("data: {}\n\n", event).as_bytes())
            .await?;
    }
    let done = serde_json::json!({
        "session_id": turn.session_id,
        "turn": turn.turn,
        "done": true,
        "status": turn.status,
        "source": turn.source,
        "kind": turn.kind,
        "focus": turn.focus,
        "entities": turn.entities,
        "lexical": turn.lexical,
        "fault": turn.fault,
        "open_turn": turn.open_turn,
        "resumed": turn.resumed,
        "resumable": turn.resumable,
    });
    stream
        .write_all(format!("data: {}\n\ndata: [DONE]\n\n", done).as_bytes())
        .await?;
    stream.flush().await?;
    let _ = stream.shutdown().await;
    Ok(())
}

/// 按句切分回答（。！？与换行），供 SSE 事件流。
fn sentence_chunks(text: &str) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }
    let mut chunks = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        current.push(ch);
        if matches!(ch, '。' | '！' | '？' | '\n') {
            chunks.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

/// 请求读取失败：体量超限单独成类，映射 413（而非一律 400）。
enum RequestReadError {
    Malformed(String),
    BodyTooLarge,
}

fn malformed(message: impl Into<String>) -> RequestReadError {
    RequestReadError::Malformed(message.into())
}

async fn read_request(
    stream: &mut TcpStream,
    max_body: usize,
) -> Result<Request, RequestReadError> {
    let mut buffer = Vec::with_capacity(4096);
    let mut chunk = [0u8; 4096];
    let header_end = loop {
        if let Some(position) = find_subsequence(&buffer, b"\r\n\r\n") {
            break position;
        }
        if buffer.len() > MAX_HEADER_BYTES {
            return Err(malformed("header too large"));
        }
        let read = stream
            .read(&mut chunk)
            .await
            .map_err(|error| malformed(format!("read error: {}", error)))?;
        if read == 0 {
            return Err(malformed("connection closed before header"));
        }
        buffer.extend_from_slice(&chunk[..read]);
    };

    let header_text = String::from_utf8_lossy(&buffer[..header_end]).to_string();
    let mut lines = header_text.split("\r\n");
    let request_line = lines
        .next()
        .ok_or_else(|| malformed("missing request line"))?;
    let mut parts = request_line.split_whitespace();
    let method = parts
        .next()
        .ok_or_else(|| malformed("missing method"))?
        .to_string();
    let target = parts
        .next()
        .ok_or_else(|| malformed("missing target"))?
        .to_string();
    let (path, query) = match target.split_once('?') {
        Some((path, query)) => (path.to_string(), parse_query(query)),
        None => (target, BTreeMap::new()),
    };
    let mut headers = BTreeMap::new();
    for line in lines {
        if let Some((name, value)) = line.split_once(':') {
            headers.insert(name.trim().to_lowercase(), value.trim().to_string());
        }
    }
    let content_length: usize = headers
        .get("content-length")
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    // 信封导入允许到信封上限（导出必须能原样导回）；其余路由维持默认体量。
    let limit = if path == "/sessions/import" {
        SESSION_ENVELOPE_MAX_BYTES
    } else {
        max_body
    };
    if content_length > limit {
        return Err(RequestReadError::BodyTooLarge);
    }
    let mut body = buffer[header_end + 4..].to_vec();
    while body.len() < content_length {
        let read = stream
            .read(&mut chunk)
            .await
            .map_err(|error| malformed(format!("read error: {}", error)))?;
        if read == 0 {
            return Err(malformed("connection closed before body"));
        }
        body.extend_from_slice(&chunk[..read]);
    }
    body.truncate(content_length);
    Ok(Request {
        method,
        path,
        query,
        headers,
        body,
    })
}

fn parse_query(query: &str) -> BTreeMap<String, String> {
    let mut parsed = BTreeMap::new();
    for pair in query.split('&') {
        if let Some((key, value)) = pair.split_once('=') {
            parsed.insert(key.to_string(), value.to_string());
        }
    }
    parsed
}

fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn status_reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        201 => "Created",
        400 => "Bad Request",
        404 => "Not Found",
        409 => "Conflict",
        413 => "Payload Too Large",
        429 => "Too Many Requests",
        504 => "Gateway Timeout",
        _ => "Internal Server Error",
    }
}

/// 服务端短答是否在回答澄清提问（是/不是/不知道、更正、批量）。
///
/// 只接受短句：长句里以「好」「是」开头的普通新问题不应被当成确认，
/// 否则严格澄清会劫持正常问答。批量短语单独识别（`classify_answer` 不覆盖）。
fn is_clarification_answer(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() || trimmed.chars().count() > 24 {
        return false;
    }
    clarification_batch_scope(trimmed).is_some()
        || !matches!(classify_answer(trimmed), AnswerKind::Unrecognized)
}

/// 回答是否表示「知识库没有依据」：只有这种失败才立悬问。
fn is_no_record_answer(text: &str) -> bool {
    text.contains("没有记录")
        || text.contains("没有相关记录")
        || text.contains("没有可用依据")
        || text.contains("数量为 0")
}

async fn write_response(stream: &mut TcpStream, response: Response) -> std::io::Result<()> {
    let header = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        response.status,
        status_reason(response.status),
        response.content_type,
        response.body.len()
    );
    stream.write_all(header.as_bytes()).await?;
    stream.write_all(&response.body).await?;
    stream.flush().await?;
    let _ = stream.shutdown().await;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send<T: Send>() {}
    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn turn_response_is_send() {
        assert_send::<TurnResponse>();
    }

    #[test]
    fn api_state_is_send_and_sync() {
        // actor 化后的核心契约：HTTP 层可以跨线程/跨任务共享服务状态
        assert_send_sync::<ApiState>();
    }

    #[tokio::test]
    async fn plugin_recipe_answers_across_http_sessions() {
        let manifest: crate::plugin::PluginManifest = toml::from_str(
            r#"
id = "test.height"
version = "0.1.0"
api = 1
name = "身高包"
default_mode = "on"

[[contributions]]
kind = "recipe"
id = "height.recipe"
patterns = ["{entity}有多高"]
question = "{entity}的身高是什么"
attribute = "身高"
template = "height.template"

[[contributions]]
kind = "template"
id = "height.template"
template = "「{entity}」的身高是{answer}。"
"#,
        )
        .expect("测试清单应可解析");
        let registry = crate::plugin::PluginRegistry::from_manifests(vec![manifest]);
        let runtime = crate::plugin::PluginRuntime::with_settings(
            registry,
            &crate::plugin::PluginSettings::default(),
        );
        assert_eq!(runtime.enabled_count(), 1);

        let mut yao = HashMap::new();
        yao.insert("身高".to_string(), "226".to_string());
        let facts = vec![PrologFact::Entity {
            name: "姚明".to_string(),
            entity_type: "人物".to_string(),
            properties: yao,
        }];
        let reasoning = ReasoningConfig {
            enable_link_prediction: false,
            ..ReasoningConfig::default()
        };
        let mut state = ApiState::new(
            facts,
            reasoning,
            HashMap::new(),
            PerfLedger::shared(),
            4,
            StartupLedger::shared(),
        );
        state.set_plugin_runtime(Some(Arc::new(parking_lot::Mutex::new(runtime))));
        let state = Arc::new(state);

        let session_id = state.create_session().await.expect("会话应创建成功");
        let turn = state
            .take_turn(&session_id, "姚明有多高")
            .await
            .expect("轮次应成功");
        assert_eq!(turn.source, "plugin");
        assert!(turn.reply.contains("226"), "{}", turn.reply);
        assert!(
            turn.plan.iter().any(|plan| plan.contains("height.recipe")),
            "{:?}",
            turn.plan
        );
    }

    /// 插件本体进入服务暖基座：逆向配方 + 子类类型约束 + 依据槽位都可跨会话工作。
    #[tokio::test]
    async fn plugin_typed_inverse_recipe_reaches_http_sessions() {
        let manifest: crate::plugin::PluginManifest = toml::from_str(
            r#"
id = "test.film"
version = "0.1.0"
api = 1
name = "电影包"
default_mode = "on"

[[contributions]]
kind = "ontology"
id = "film.ontology"
predicates = [
  { name = "导演", inverse = "执导", domain = ["电影"], range = ["人物"] },
]
types = [
  { name = "电影", subclass_of = ["影视作品"] },
  { name = "动画电影", subclass_of = ["电影"] },
]

[[contributions]]
kind = "recipe"
id = "film.works_by_director"
patterns = ["{entity}导演过哪些电影", "那{entity}还导演过什么"]
question = "{entity}导演过哪些作品"
attribute = "导演"
direction = "inverse"
answer_types = ["电影"]
template = "film.works.template"

[[contributions]]
kind = "template"
id = "film.works.template"
template = "「{entity}」导演过的作品：{answer}。依据：{evidence}"
"#,
        )
        .expect("测试清单应可解析");
        let registry = crate::plugin::PluginRegistry::from_manifests(vec![manifest]);
        let runtime = crate::plugin::PluginRuntime::with_settings(
            registry,
            &crate::plugin::PluginSettings::default(),
        );
        let facts = vec![
            PrologFact::Entity {
                name: "阿凡达".into(),
                entity_type: "电影".into(),
                properties: HashMap::new(),
            },
            PrologFact::Entity {
                name: "卡梅隆".into(),
                entity_type: "人物".into(),
                properties: HashMap::new(),
            },
            PrologFact::Relation {
                from: "阿凡达".into(),
                rel_type: "导演".into(),
                to: "卡梅隆".into(),
            },
        ];
        let reasoning = ReasoningConfig {
            enable_link_prediction: false,
            ..ReasoningConfig::default()
        };
        let mut state = ApiState::new(
            facts,
            reasoning,
            HashMap::new(),
            PerfLedger::shared(),
            4,
            StartupLedger::shared(),
        );
        let receipt = state.configure_plugin_runtime(&runtime);
        assert!(receipt.lessons > 0, "插件 TBox 必须并入暖基座");
        assert!(receipt.inverse_surfaces > 0, "逆关系词面必须登记");
        state.set_plugin_runtime(Some(Arc::new(parking_lot::Mutex::new(runtime))));
        let state = Arc::new(state);

        let session_id = state.create_session().await.expect("会话应创建成功");
        let turn = state
            .take_turn(&session_id, "卡梅隆导演过哪些电影")
            .await
            .expect("轮次应成功");
        assert_eq!(turn.source, "plugin", "{}", turn.reply);
        assert!(turn.reply.contains("阿凡达"), "{}", turn.reply);
        assert!(
            turn.reply.contains("导演") && turn.reply.contains("依据"),
            "依据必须接地：{}",
            turn.reply
        );

        // 多轮：省略主语时用对话焦点补全，再沿同一本体做逆向检索。
        let followup = state
            .take_turn(&session_id, "那他还导演过什么")
            .await
            .expect("追问轮次应成功");
        assert_eq!(followup.source, "plugin", "{}", followup.reply);
        assert!(followup.reply.contains("阿凡达"), "{}", followup.reply);
    }

    /// 正向配方回答后焦点转到答案实体：下一轮「这位导演…」可无缝续问。
    #[tokio::test]
    async fn plugin_forward_answer_moves_focus_for_role_followup() {
        let manifest: crate::plugin::PluginManifest = toml::from_str(
            r#"
id = "test.film"
version = "0.1.0"
api = 1
name = "电影包"
default_mode = "on"

[[contributions]]
kind = "reference"
id = "film.reference"
quantifiers = ["这位", "那位"]
role_words = ["导演"]

[[contributions]]
kind = "recipe"
id = "film.directed_by"
patterns = ["{entity}是谁导演的"]
question = "{entity}的导演是什么"
attribute = "导演"
direction = "forward"
answer_types = ["人物"]
template = "film.directed.template"

[[contributions]]
kind = "recipe"
id = "film.works_by_director"
patterns = ["{entity}还拍过什么", "那{entity}还拍过什么"]
question = "{entity}导演过哪些作品"
attribute = "导演"
direction = "inverse"
answer_types = ["电影"]
template = "film.works.template"

[[contributions]]
kind = "template"
id = "film.directed.template"
template = "「{entity}」的导演是{answer}。依据：{evidence}"

[[contributions]]
kind = "template"
id = "film.works.template"
template = "「{entity}」导演过的作品：{answer}。依据：{evidence}"
"#,
        )
        .expect("测试清单应可解析");
        let registry = crate::plugin::PluginRegistry::from_manifests(vec![manifest]);
        let runtime = crate::plugin::PluginRuntime::with_settings(
            registry,
            &crate::plugin::PluginSettings::default(),
        );
        let facts = vec![
            PrologFact::Entity {
                name: "阿凡达".into(),
                entity_type: "电影".into(),
                properties: HashMap::new(),
            },
            PrologFact::Entity {
                name: "卡梅隆".into(),
                entity_type: "人物".into(),
                properties: HashMap::new(),
            },
            PrologFact::Relation {
                from: "阿凡达".into(),
                rel_type: "导演".into(),
                to: "卡梅隆".into(),
            },
        ];
        let reasoning = ReasoningConfig {
            enable_link_prediction: false,
            ..ReasoningConfig::default()
        };
        let mut state = ApiState::new(
            facts,
            reasoning,
            HashMap::new(),
            PerfLedger::shared(),
            4,
            StartupLedger::shared(),
        );
        state.set_plugin_runtime(Some(Arc::new(parking_lot::Mutex::new(runtime))));
        let state = Arc::new(state);

        let session_id = state.create_session().await.expect("会话应创建成功");
        let first = state
            .take_turn(&session_id, "阿凡达是谁导演的")
            .await
            .expect("首轮应成功");
        assert_eq!(first.source, "plugin", "{}", first.reply);
        assert!(first.reply.contains("卡梅隆"), "{}", first.reply);

        let followup = state
            .take_turn(&session_id, "这位导演还拍过什么")
            .await
            .expect("追问轮次应成功");
        assert_eq!(followup.source, "plugin", "{}", followup.reply);
        assert!(followup.reply.contains("阿凡达"), "{}", followup.reply);
    }

    fn test_state(max_sessions: usize) -> Arc<ApiState> {
        let mut yao = HashMap::new();
        yao.insert("身高".to_string(), "226".to_string());
        yao.insert("国籍".to_string(), "中国".to_string());
        let mut ye = HashMap::new();
        ye.insert("身高".to_string(), "190".to_string());
        let facts = vec![
            PrologFact::Entity {
                name: "姚明".to_string(),
                entity_type: "人物".to_string(),
                properties: yao,
            },
            PrologFact::Entity {
                name: "叶莉".to_string(),
                entity_type: "人物".to_string(),
                properties: ye,
            },
        ];
        let reasoning = ReasoningConfig {
            enable_link_prediction: false,
            ..ReasoningConfig::default()
        };
        Arc::new(ApiState::new(
            facts,
            reasoning,
            HashMap::new(),
            PerfLedger::shared(),
            max_sessions,
            StartupLedger::shared(),
        ))
    }

    /// 配置态本体注入暖基座：分叉出的每个会话同源携带，且注入幂等。
    #[test]
    fn configure_ontology_reaches_forked_sessions() {
        use crate::ontology_reasoning::OntologyProgram;
        use crate::predicate_registry::PredicateProfileConfig;
        use crate::type_system::{TypeSubclassConfig, TypeSystemConfig};

        let state = test_state(2);
        let program = OntologyProgram::from_config(
            &TypeSystemConfig {
                subclasses: vec![TypeSubclassConfig {
                    sub: "学生".into(),
                    sup: "人物".into(),
                }],
                ..TypeSystemConfig::default()
            },
            &[PredicateProfileConfig {
                name: "师傅".into(),
                domain: vec!["人物".into()],
                ..PredicateProfileConfig::default()
            }],
        );
        assert_eq!(state.configure_ontology(&program), 2, "子类1 + 定义域1");
        assert_eq!(state.configure_ontology(&program), 0, "重复注入必须去重");
        let fork = state.warm.fork();
        assert_eq!(
            fork.taught_ontology_lessons().len(),
            2,
            "分叉会话应携带配置本体"
        );
        let snapshot = fork.dialogue_snapshot();
        assert!(
            snapshot.ontology.is_some(),
            "配置本体应随会话快照导出（生成 = 序列化）"
        );
    }

    fn test_state_with_quality() -> Arc<ApiState> {
        use crate::predicate_registry::PredicateRegistry;
        use crate::quality::{build_report, QualityIndex};
        use crate::type_system::TypeSystem;

        let mut yao = HashMap::new();
        yao.insert("身高".to_string(), "226".to_string());
        let mut ye = HashMap::new();
        ye.insert("身高".to_string(), "190".to_string());
        let facts = vec![
            PrologFact::Entity {
                name: "姚明".to_string(),
                entity_type: "人物".to_string(),
                properties: yao,
            },
            PrologFact::Entity {
                name: "叶莉".to_string(),
                entity_type: "人物".to_string(),
                properties: ye,
            },
        ];
        let mut registry = PredicateRegistry::new();
        registry.observe("姚明", "身高", "226", Some("体育百科"));
        let report = build_report(&facts, &registry, &TypeSystem::new(), &[]);
        let index = QualityIndex::from_report(&report, &facts, &registry);
        let reasoning = ReasoningConfig {
            enable_link_prediction: false,
            ..ReasoningConfig::default()
        };
        let mut state = ApiState::new(
            facts,
            reasoning,
            HashMap::new(),
            PerfLedger::shared(),
            4,
            StartupLedger::shared(),
        );
        let index = index.with_revision(state.warm_revision());
        state.set_quality_index(Some(Arc::new(index)));
        Arc::new(state)
    }

    #[tokio::test]
    async fn quality_index_reaches_http_sessions() {
        let state = test_state_with_quality();
        assert!(state.quality_index().is_some());
        let session_id = state.create_session().await.expect("会话应创建成功");
        let turn = state
            .take_turn(&session_id, "姚明的身高是多少")
            .await
            .expect("轮次应成功");
        assert!(turn.reply.contains("226"), "{}", turn.reply);
        assert!(
            turn.plan.iter().any(|plan| plan == "quality_cite_source"),
            "HTTP 回答必须带出处：{:?} / {}",
            turn.plan,
            turn.reply
        );
        assert!(turn.reply.contains("出处：体育百科"), "{}", turn.reply);
        assert_eq!(
            state.quality_index().map(|index| index.revision),
            Some(state.warm_revision())
        );
    }

    /// 注册表对齐（谓词别名/换向词面 + 实体持久别名）必须进暖基座：
    /// CLI 里签过字的说法，走 HTTP 发问同样要能命中同一份知识。
    #[tokio::test]
    async fn registry_alignment_reaches_http_sessions() {
        let mut yao = HashMap::new();
        yao.insert("身高".to_string(), "226".to_string());
        let facts = vec![
            PrologFact::Entity {
                name: "姚明".to_string(),
                entity_type: "人物".to_string(),
                properties: yao,
            },
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "师傅".to_string(),
                to: "李四".to_string(),
            },
        ];
        let reasoning = ReasoningConfig {
            enable_link_prediction: false,
            ..ReasoningConfig::default()
        };
        let state = ApiState::new(
            facts,
            reasoning,
            HashMap::new(),
            PerfLedger::shared(),
            4,
            StartupLedger::shared(),
        );
        let applied = state.align_predicate_pairs(
            &[
                ("老师".to_string(), "师傅".to_string(), false),
                ("徒弟".to_string(), "师傅".to_string(), true),
            ],
            &crate::predicate_registry::PredicateRegistry::default(),
        );
        assert_eq!(applied, 2, "正向别名与换向词面都应生效");
        let entities = state.align_entity_aliases(&[("明哥".to_string(), "姚明".to_string())]);
        assert_eq!(entities, 1, "实体持久别名应生效");
        let state = Arc::new(state);

        let session_id = state.create_session().await.expect("会话应创建成功");
        let forward = state
            .take_turn(&session_id, "张三的老师是谁")
            .await
            .expect("正向别名轮次应成功");
        assert!(forward.reply.contains("李四"), "{}", forward.reply);

        let inverse = state
            .take_turn(&session_id, "李四的徒弟是谁")
            .await
            .expect("换向词面轮次应成功");
        assert!(inverse.reply.contains("张三"), "{}", inverse.reply);

        let alias = state
            .take_turn(&session_id, "明哥的身高是多少")
            .await
            .expect("实体别名轮次应成功");
        assert!(
            alias.reply.contains("226"),
            "{:?} / {:?} / {}",
            alias.source,
            alias.plan,
            alias.reply
        );
    }

    /// 构造最小合法信封（`insert_order` 只改变焦点类型表的插入序，
    /// 焦点栈本身固定，用于验证摘要与 HashMap 迭代序无关）。
    fn envelope_with_focus(insert_order: &[&str]) -> SessionEnvelope {
        use crate::symbolic_reasoner::DialogueContext;
        let keys = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛"];
        let mut context = DialogueContext::new(16);
        for key in &keys {
            context.record_entities_typed(&[(key.to_string(), Some("人物".to_string()))]);
        }
        let mut snapshot = context.snapshot();
        let mut focus_types = HashMap::new();
        for key in insert_order {
            focus_types.insert(key.to_string(), "人物".to_string());
        }
        snapshot.focus_types = focus_types;
        SessionEnvelope {
            format: ENVELOPE_FORMAT.to_string(),
            version: ENVELOPE_VERSION,
            session_id: "s-test".to_string(),
            turns: 2,
            history: vec!["姚明是谁".to_string()],
            policy: SessionPolicy::default(),
            manager: DialogueManager::new().snapshot(),
            reasoner: snapshot,
            engine: EngineStamp {
                crate_version: env!("CARGO_PKG_VERSION").to_string(),
                kb_fingerprint: "test-fingerprint".to_string(),
                kb_facts: 2,
                prompt_mode: "on".to_string(),
                recovery_mode: "observe".to_string(),
            },
            digest: String::new(),
        }
    }

    #[test]
    fn envelope_digest_is_canonical_and_hashmap_order_invariant() {
        let forward_keys = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛"];
        let reversed_keys: Vec<&str> = forward_keys.iter().rev().copied().collect();
        let mut forward = envelope_with_focus(&forward_keys);
        let mut backward = envelope_with_focus(&reversed_keys);
        forward.seal();
        backward.seal();
        assert!(forward.digest_valid() && backward.digest_valid());
        assert_eq!(
            forward.digest, backward.digest,
            "摘要不得依赖 HashMap 迭代序（跨进程换入不得误判 Stale）"
        );

        // L1/L2/L3/L4：规范字节往返 + 摘要交换
        let bytes = canonical::canonical_bytes(&forward).unwrap();
        let (decoded, receipt) = SessionEnvelope::decode(&bytes, 1 << 20).unwrap();
        assert!(receipt.canonical, "自产字节必须是规范形");
        assert_eq!(decoded.digest, forward.digest);
        assert!(decoded.digest_valid());
        assert_eq!(canonical::canonical_bytes(&decoded).unwrap(), bytes);
        assert_eq!(
            canonical::canonical_digest(&decoded).unwrap(),
            canonical::canonical_digest(&forward).unwrap()
        );
    }

    #[test]
    fn envelope_decode_error_taxonomy_is_stable() {
        let mut base = envelope_with_focus(&["甲"]);
        base.seal();
        let canonical_bytes = canonical::canonical_bytes(&base).unwrap();

        assert_eq!(
            SessionEnvelope::decode(b"{not json", 1 << 20)
                .unwrap_err()
                .code(),
            "parse"
        );
        assert_eq!(
            SessionEnvelope::decode(&canonical_bytes, 8)
                .unwrap_err()
                .code(),
            "too_large"
        );

        let mut wrong_format = base.clone();
        wrong_format.format = "other-format".to_string();
        wrong_format.seal();
        assert_eq!(
            SessionEnvelope::decode(&canonical::canonical_bytes(&wrong_format).unwrap(), 1 << 20)
                .unwrap_err()
                .code(),
            "format"
        );

        let mut future = base.clone();
        future.version = ENVELOPE_VERSION + 1;
        future.seal();
        assert_eq!(
            SessionEnvelope::decode(&canonical::canonical_bytes(&future).unwrap(), 1 << 20)
                .unwrap_err()
                .code(),
            "version"
        );

        let mut tampered = base.clone();
        tampered.history.push("伪造的历史".to_string());
        assert_eq!(
            SessionEnvelope::decode(&canonical::canonical_bytes(&tampered).unwrap(), 1 << 20)
                .unwrap_err()
                .code(),
            "digest"
        );
    }

    /// 服务跑在专属 OS 线程的 current-thread 运行时里（会话 actor 由 `run`
    /// 自行派生）；测试只通过 TCP 交互。
    fn start_server(max_sessions: usize) -> (std::net::SocketAddr, std::thread::JoinHandle<()>) {
        start_server_with_recovery(max_sessions, RecoveryMode::Observe)
    }

    #[tokio::test]
    async fn tampered_cold_session_surfaces_decode_code_without_side_effects() {
        let dir = temp_session_dir("decode");
        let (addr, _server) =
            start_server_with_options(1, RecoveryMode::Observe, Some(dir.clone()), None);
        let first = create_session(addr).await;
        take_turn_body(addr, &first, "姚明是谁").await;
        let _second = create_session(addr).await;
        let path = dir.join(format!("{}.json", first));
        assert!(path.exists(), "容量 1 应把 first 冻结到磁盘");

        // 换出写规范字节：解码回执必须标记 canonical（L2 字节不动点）
        let frozen_bytes = std::fs::read(&path).unwrap();
        let (frozen, receipt) = SessionEnvelope::decode(&frozen_bytes, 1 << 20).unwrap();
        assert!(receipt.canonical, "换出的冷信封必须写规范字节");
        assert!(frozen.digest_valid());
        assert_eq!(
            canonical::canonical_bytes(&frozen).unwrap(),
            frozen_bytes,
            "规范字节再编码逐字节一致"
        );

        // 篡改 turns 但不重算摘要 → 换入必须被拒，且给出稳定错误码
        let mut value: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        value["turns"] = serde_json::json!(99);
        std::fs::write(&path, value.to_string()).unwrap();
        let (status, _, body) =
            raw_request(addr, "GET", &format!("/sessions/{}", first), &[], "").await;
        assert_eq!(status, 409, "{}", body);
        assert!(body.contains("decode_digest"), "{}", body);

        let (_, _, metrics) = raw_request(addr, "GET", "/metrics", &[], "").await;
        let metrics: serde_json::Value = serde_json::from_str(&metrics).unwrap();
        assert_eq!(metrics["sessions"], 1, "被拒的冷换入不得产生会话");
        assert_eq!(metrics["perf"]["decode_reject_digest"], 1, "{}", metrics);
        assert_eq!(
            metrics["perf"]["decode_ok"].as_u64().unwrap_or(0),
            0,
            "{}",
            metrics
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    fn start_server_with_recovery(
        max_sessions: usize,
        recovery: RecoveryMode,
    ) -> (std::net::SocketAddr, std::thread::JoinHandle<()>) {
        start_server_with_options(max_sessions, recovery, None, None)
    }

    fn start_server_with_budget(
        max_sessions: usize,
        budget: TurnBudget,
    ) -> (std::net::SocketAddr, std::thread::JoinHandle<()>) {
        start_server_with_options(max_sessions, RecoveryMode::Observe, None, Some(budget))
    }

    /// 与服务启动路径一致：给暖基座注入 NLP 配置，失败轮才能生成读法回执。
    fn test_nlp_config() -> NlpConfig {
        NlpConfig {
            import_keywords: Vec::new(),
            pos_dict: HashMap::new(),
            name_patterns: crate::NamePatterns {
                prefixes: Vec::new(),
                suffixes: Vec::new(),
                full_names: Vec::new(),
                profession_suffix: Vec::new(),
            },
            enable_nlu: None,
            nlu_confidence_threshold: None,
            enable_local_semantic: None,
            disable_ownthink_api: None,
            semantic_similarity_threshold: None,
            max_entity_candidates: None,
            enable_advanced_entity_extraction: None,
            enable_relation_extraction: None,
            context_window_size: None,
        }
    }

    fn start_server_with_options(
        max_sessions: usize,
        recovery: RecoveryMode,
        session_dir: Option<std::path::PathBuf>,
        turn_budget: Option<TurnBudget>,
    ) -> (std::net::SocketAddr, std::thread::JoinHandle<()>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            runtime.block_on(async move {
                let listener = TcpListener::from_std(listener).unwrap();
                let mut state = test_state(max_sessions);
                // Arc::get_mut 不适用（已 Arc）：状态在启动前唯一持有，先解包
                let inner = Arc::get_mut(&mut state).expect("测试启动前状态唯一");
                inner.set_recovery_mode(recovery);
                inner.set_nlp_config(test_nlp_config());
                if let Some(dir) = session_dir {
                    inner.set_session_dir(dir);
                }
                if let Some(budget) = turn_budget {
                    inner.set_turn_budget(budget);
                }
                let _ = run(listener, state).await;
            });
        });
        (addr, handle)
    }

    fn temp_session_dir(label: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("kp-serve-{}-{}", std::process::id(), label));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    async fn raw_request(
        addr: std::net::SocketAddr,
        method: &str,
        path: &str,
        headers: &[(&str, &str)],
        body: &str,
    ) -> (u16, String, String) {
        let mut stream = TcpStream::connect(addr).await.unwrap();
        let mut request = format!("{} {} HTTP/1.1\r\nHost: localhost\r\n", method, path);
        for (name, value) in headers {
            request.push_str(&format!("{}: {}\r\n", name, value));
        }
        request.push_str(&format!("Content-Length: {}\r\n\r\n", body.len()));
        request.push_str(body);
        stream.write_all(request.as_bytes()).await.unwrap();
        let mut response = Vec::new();
        stream.read_to_end(&mut response).await.unwrap();
        let response = String::from_utf8_lossy(&response).to_string();
        let status = response
            .lines()
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|code| code.parse().ok())
            .unwrap_or(0);
        let (head, body) = response
            .split_once("\r\n\r\n")
            .map(|(head, body)| (head.to_string(), body.to_string()))
            .unwrap_or((response.clone(), String::new()));
        (status, head, body)
    }

    async fn create_session(addr: std::net::SocketAddr) -> String {
        let (status, _, body) = raw_request(addr, "POST", "/sessions", &[], "").await;
        assert_eq!(status, 201, "{}", body);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        value["session_id"].as_str().unwrap().to_string()
    }

    #[tokio::test]
    async fn health_metrics_and_index() {
        let (addr, _server) = start_server(4);
        let (status, _, body) = raw_request(addr, "GET", "/health", &[], "").await;
        assert_eq!(status, 200);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["status"], "ok");

        let session_id = create_session(addr).await;
        let (status, _, body) = raw_request(
            addr,
            "POST",
            &format!("/sessions/{}/turns", session_id),
            &[("Content-Type", "application/json")],
            r#"{"input": "姚明的身高是多少"}"#,
        )
        .await;
        assert_eq!(status, 200, "{}", body);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["status"], "answered");
        assert!(value["reply"].as_str().unwrap().contains("226"), "{}", body);

        let (status, _, body) = raw_request(addr, "GET", "/metrics", &[], "").await;
        assert_eq!(status, 200);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["sessions"], 1);
        assert_eq!(value["turns_total"], 1);

        let (status, _, body) = raw_request(addr, "GET", "/", &[], "").await;
        assert_eq!(status, 200);
        assert!(body.contains("knowprolog"));
    }

    #[tokio::test]
    async fn read_traces_route_returns_bounded_receipts_without_raw_question() {
        let (addr, _server) = start_server(4);
        let session_id = create_session(addr).await;

        let (status, _, body) = raw_request(
            addr,
            "GET",
            &format!("/sessions/{}/read-traces", session_id),
            &[],
            "",
        )
        .await;
        assert_eq!(status, 200, "{}", body);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["session_id"], session_id);
        assert!(value["report"]
            .as_str()
            .is_some_and(|report| report.contains("读法台账")));
        assert_eq!(value["receipts"].as_array().map(Vec::len), Some(0));

        let question = "未名科技公司的老板是谁";
        let (status, _, body) = raw_request(
            addr,
            "POST",
            &format!("/sessions/{}/turns", session_id),
            &[("Content-Type", "application/json")],
            &serde_json::json!({ "input": question }).to_string(),
        )
        .await;
        assert_eq!(status, 200, "{}", body);

        let (status, _, body) = raw_request(
            addr,
            "GET",
            &format!("/sessions/{}/read-traces", session_id),
            &[],
            "",
        )
        .await;
        assert_eq!(status, 200, "{}", body);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        let receipts = value["receipts"].as_array().expect("回执必须是数组");
        assert_eq!(receipts.len(), 1, "失败轮应留下一条回执：{}", body);
        assert_eq!(receipts[0]["answered"], false);
        assert!(
            receipts[0]["failure_kinds"]
                .as_array()
                .is_some_and(|kinds| kinds.iter().any(|kind| kind == "unknown_entity")),
            "未知实体应被分类：{}",
            body
        );
        assert!(
            !body.contains(question),
            "读法回执不得泄露原始问句：{}",
            body
        );

        let (status, _, _) = raw_request(addr, "GET", "/sessions/nope/read-traces", &[], "").await;
        assert_eq!(status, 404);
    }

    #[tokio::test]
    async fn recovery_mode_surfaces_epistemic_view() {
        let (addr, _server) = start_server_with_recovery(4, RecoveryMode::On);
        let session_id = create_session(addr).await;

        // 已知实体 + 图谱里没有的属性：On 模式必须给出认识论视图与回执
        let (status, _, body) = raw_request(
            addr,
            "POST",
            &format!("/sessions/{}/turns", session_id),
            &[],
            "姚明的体重是多少",
        )
        .await;
        assert_eq!(status, 200, "{}", body);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        let epistemic = &value["epistemic"];
        assert_eq!(epistemic["status"], "absent", "{}", body);
        assert_eq!(epistemic["label"], "确无");
        assert!(
            epistemic["receipt"]
                .as_str()
                .is_some_and(|receipt| receipt.starts_with('r')),
            "缺记录必须有回执 id：{}",
            body
        );
        assert!(value["reply"]
            .as_str()
            .is_some_and(|reply| !reply.contains("暂时") || reply.contains("没有记录")));

        // 无失败的动作轮只携带当前视图
        let (_, _, body) = raw_request(
            addr,
            "POST",
            &format!("/sessions/{}/turns", session_id),
            &[],
            "姚明的身高是多少",
        )
        .await;
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert!(
            value.get("epistemic").is_none(),
            "成功回答轮不得携带陈旧认识论视图：{}",
            body
        );

        // 让位（defer）也必须可命名：Unknown 回执而非无声丢弃
        let (_, _, body) = raw_request(
            addr,
            "POST",
            &format!("/sessions/{}/turns", session_id),
            &[],
            "随便聊聊吧",
        )
        .await;
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["status"], "deferred", "{}", body);
        assert_eq!(value["epistemic"]["status"], "unknown", "{}", body);
        assert!(value["epistemic"]["receipt"].as_str().is_some(), "{}", body);
    }

    #[tokio::test]
    async fn session_layers_expose_four_level_manifest() {
        let (addr, _server) = start_server(4);
        let session_id = create_session(addr).await;
        take_turn_body(addr, &session_id, "姚明是谁").await;

        let (status, _, body) = raw_request(
            addr,
            "GET",
            &format!("/sessions/{}/layers", session_id),
            &[],
            "",
        )
        .await;
        assert_eq!(status, 200, "{}", body);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["session_id"], session_id);
        let manifest = &value["manifest"];
        assert_eq!(manifest["manifest_version"], 1);
        let layers = manifest["layers"].as_array().expect("layers 数组");
        assert_eq!(layers.len(), 4);
        assert_eq!(layers[0]["id"], "L0");
        assert!(layers[0]["count"].as_u64().unwrap_or(0) > 0);
        assert_eq!(layers[3]["id"], "L3");
        assert_eq!(
            manifest["digest"].as_str().map(str::len),
            Some(16),
            "{}",
            body
        );
    }

    #[tokio::test]
    async fn session_lifecycle_and_focus() {
        let (addr, _server) = start_server(4);
        let session_id = create_session(addr).await;

        let (status, _, body) = raw_request(
            addr,
            "POST",
            &format!("/sessions/{}/turns", session_id),
            &[],
            "姚明是谁",
        )
        .await;
        assert_eq!(status, 200);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["source"], "dialogue");
        assert_eq!(value["focus"], "姚明");

        // 指代继承：第二轮用「他的身高呢」应命中同一焦点
        let (status, _, body) = raw_request(
            addr,
            "POST",
            &format!("/sessions/{}/turns", session_id),
            &[],
            "他的身高呢",
        )
        .await;
        assert_eq!(status, 200);
        assert!(body.contains("226"), "{}", body);

        let (status, _, body) =
            raw_request(addr, "GET", &format!("/sessions/{}", session_id), &[], "").await;
        assert_eq!(status, 200);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["turns"], 2);
        assert_eq!(value["focus_stack"][0], "姚明");

        let (status, _, _) = raw_request(
            addr,
            "DELETE",
            &format!("/sessions/{}", session_id),
            &[],
            "",
        )
        .await;
        assert_eq!(status, 200);
        let (status, _, _) = raw_request(
            addr,
            "POST",
            &format!("/sessions/{}/turns", session_id),
            &[],
            "你好",
        )
        .await;
        assert_eq!(status, 404);
    }

    /// 接地引用：回答附 `entities`（词面 ↔ URI ↔ 类型 ↔ 角色），身份可寻址。
    #[tokio::test]
    async fn grounding_citations_expose_uri_type_and_role() {
        let (addr, _server) = start_server(2);
        let session_id = create_session(addr).await;
        let (status, _, body) = raw_request(
            addr,
            "POST",
            &format!("/sessions/{}/turns", session_id),
            &[],
            "姚明的身高是多少",
        )
        .await;
        assert_eq!(status, 200);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        let entities = value["entities"].as_array().expect("应附接地清单");
        let yao = entities
            .iter()
            .find(|entity| entity["surface"] == "姚明")
            .expect("应接地到姚明");
        assert_eq!(yao["uri"], "http://example.org/kg/entity/姚明");
        assert_eq!(yao["etype"], "人物");
        assert_eq!(yao["role"], "focus");
    }

    #[tokio::test]
    async fn unknown_input_defers_and_sessions_are_isolated() {
        let (addr, _server) = start_server(4);
        let first = create_session(addr).await;
        let second = create_session(addr).await;

        let (status, _, body) = raw_request(
            addr,
            "POST",
            &format!("/sessions/{}/turns", first),
            &[],
            "姚明是谁",
        )
        .await;
        assert_eq!(status, 200);
        assert!(body.contains("姚明"));

        // 第二个会话没有焦点：指代句不得借用第一个会话的状态
        let (status, _, body) = raw_request(
            addr,
            "POST",
            &format!("/sessions/{}/turns", second),
            &[],
            "他的身高呢",
        )
        .await;
        assert_eq!(status, 200);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["status"], "deferred", "{}", body);

        // 未知输入：让位而非幻觉
        let (status, _, body) = raw_request(
            addr,
            "POST",
            &format!("/sessions/{}/turns", second),
            &[],
            r#"{"input": "今天天气怎么样"}"#,
        )
        .await;
        assert_eq!(status, 200);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["status"], "deferred");
        assert_eq!(value["source"], "none");
    }

    #[tokio::test]
    async fn sse_streams_sentences_then_done() {
        let (addr, _server) = start_server(4);
        let session_id = create_session(addr).await;
        let (status, head, body) = raw_request(
            addr,
            "POST",
            &format!("/sessions/{}/turns", session_id),
            &[("Accept", "text/event-stream")],
            "姚明的身高是多少",
        )
        .await;
        assert_eq!(status, 200);
        assert!(head.contains("text/event-stream"), "{}", head);
        assert!(body.contains("data: "), "{}", body);
        assert!(body.contains("[DONE]"), "{}", body);
        assert!(body.contains("226"), "{}", body);
        assert!(
            body.contains("\"phase\":\"prepare\""),
            "SSE 必须先回执阶段进度：{}",
            body
        );
    }

    #[tokio::test]
    async fn sse_deadline_done_event_carries_fault_and_open_turn() {
        let (addr, _server) = start_server_with_budget(1, TurnBudget::enforced(Duration::ZERO));
        let session_id = create_session(addr).await;
        let (status, _, body) = raw_request(
            addr,
            "POST",
            &format!("/sessions/{}/turns", session_id),
            &[("Accept", "text/event-stream")],
            "姚明的身高是多少",
        )
        .await;
        assert_eq!(status, 200, "{}", body);
        assert!(body.contains("\"phase\":\"prepare\""), "{}", body);
        assert!(body.contains("\"status\":\"deadline\""), "{}", body);
        assert!(body.contains("\"fault\""), "{}", body);
        assert!(body.contains("\"open_turn\""), "{}", body);
        assert!(body.contains("\"resumable\":true"), "{}", body);
        assert!(body.contains("[DONE]"), "{}", body);
    }

    #[tokio::test]
    async fn openai_compat_models_and_completion() {
        let (addr, _server) = start_server(4);
        let (status, _, body) = raw_request(addr, "GET", "/v1/models", &[], "").await;
        assert_eq!(status, 200);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["data"][0]["id"], DEFAULT_MODEL);

        // 无状态：临时会话执行后即关闭，不占会话额度
        let (status, _, body) = raw_request(
            addr,
            "POST",
            "/v1/chat/completions",
            &[("Content-Type", "application/json")],
            r#"{"model": "knowprolog-symbolic", "messages": [{"role": "user", "content": "姚明的身高是多少"}]}"#,
        )
        .await;
        assert_eq!(status, 200, "{}", body);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["object"], "chat.completion");
        assert_eq!(value["model"], DEFAULT_MODEL);
        assert!(
            value["choices"][0]["message"]["content"]
                .as_str()
                .unwrap()
                .contains("226"),
            "{}",
            body
        );
        let (_, _, metrics) = raw_request(addr, "GET", "/metrics", &[], "").await;
        let metrics: serde_json::Value = serde_json::from_str(&metrics).unwrap();
        assert_eq!(metrics["sessions"], 0);

        // 服务端记忆：带 session_id 时只执行最后一条用户消息
        let session_id = create_session(addr).await;
        let (status, _, _) = raw_request(
            addr,
            "POST",
            &format!("/sessions/{}/turns", session_id),
            &[],
            "姚明是谁",
        )
        .await;
        assert_eq!(status, 200);
        let payload = serde_json::json!({
            "messages": [
                {"role": "user", "content": "姚明是谁"},
                {"role": "assistant", "content": "他是姚明。"},
                {"role": "user", "content": "他的身高呢"},
            ],
            "session_id": session_id,
        })
        .to_string();
        let (status, _, body) = raw_request(
            addr,
            "POST",
            "/v1/chat/completions",
            &[("Content-Type", "application/json")],
            &payload,
        )
        .await;
        assert_eq!(status, 200, "{}", body);
        assert!(body.contains("226"), "{}", body);
    }

    /// 无状态问答缓存：第二次同键请求命中缓存（回答逐字节一致），
    /// 命中路径不消耗会话额度；纪元计数与启动分期可从 /metrics 审计。
    #[tokio::test]
    async fn stateless_answer_cache_hits_are_byte_identical_and_audited() {
        let (addr, _server) = start_server(4);
        let payload = r#"{"messages": [
            {"role": "user", "content": "姚明是谁"},
            {"role": "user", "content": "姚明的身高是多少"}
        ]}"#;

        let (status, _, first_body) = raw_request(
            addr,
            "POST",
            "/v1/chat/completions",
            &[("Content-Type", "application/json")],
            payload,
        )
        .await;
        assert_eq!(status, 200, "{}", first_body);
        let first: serde_json::Value = serde_json::from_str(&first_body).unwrap();

        let (status, _, second_body) = raw_request(
            addr,
            "POST",
            "/v1/chat/completions",
            &[("Content-Type", "application/json")],
            payload,
        )
        .await;
        assert_eq!(status, 200, "{}", second_body);
        let second: serde_json::Value = serde_json::from_str(&second_body).unwrap();

        // 命中输出 ≡ 重算输出：回答/模型/选择逐字节一致
        assert_eq!(
            first["choices"], second["choices"],
            "缓存命中的回答必须与首次计算逐字节一致"
        );
        assert_eq!(first["model"], second["model"]);
        assert_ne!(
            first["id"], second["id"],
            "完成 id 按请求的新会话戳记，不共享"
        );
        assert!(
            first["choices"][0]["message"]["content"]
                .as_str()
                .unwrap()
                .contains("226"),
            "{}",
            first_body
        );

        // 命中路径不驻留会话（临时会话本就关闭）；台账可审计
        let (_, _, metrics) = raw_request(addr, "GET", "/metrics", &[], "").await;
        let metrics: serde_json::Value = serde_json::from_str(&metrics).unwrap();
        assert_eq!(metrics["sessions"], 0);
        assert_eq!(metrics["perf"]["answer_cache_misses"], 1);
        assert_eq!(metrics["perf"]["answer_cache_hits"], 1);
        assert_eq!(metrics["perf"]["session_forks"], 1, "命中不应再分叉会话");

        // 启动分期：进程启动相与会话分叉相都有明细（结构断言，不断言毫秒）
        let phases: Vec<&str> = metrics["startup"]
            .as_array()
            .expect("startup 明细应存在")
            .iter()
            .filter_map(|entry| entry["phase"].as_str())
            .collect();
        // 测试直接注入已绑定的监听器，故 bind 相不在场；进程内相与会话相必须可见
        assert!(phases.contains(&"warm_build"), "{:?}", phases);
        assert!(phases.contains(&"session_fork"), "{:?}", phases);
        assert!(metrics["warm_revision"].as_u64().unwrap() >= 1);
    }

    /// 身份准入纪元推进必须让无状态问答缓存整体失效：
    /// 非当前纪元的条目不可命中（即便键相同），新纪元写入后恢复命中。
    #[tokio::test]
    async fn identity_epoch_bump_invalidates_stateless_cache() {
        let state = test_state(4);
        let response = TurnResponse {
            session_id: "s-1".to_string(),
            turn: 1,
            status: "answered".to_string(),
            source: "reasoner".to_string(),
            kind: None,
            plan: Vec::new(),
            reply: "姚明的身高是226cm".to_string(),
            focus: None,
            policy: SessionPolicy::default(),
            epistemic: None,
            lexical: None,
            entities: Vec::new(),
            board: None,
            fault: None,
            open_turn: None,
            resumed: false,
            resumable: false,
            derivation: None,
        };
        let key = stateless_answer_key(&["姚明的身高是多少".to_string()], None);

        let epoch = state.cache_epoch();
        state.store_stateless_answer(epoch, key.clone(), response.clone());
        assert!(state.cached_stateless_answer(epoch, &key).is_some());

        state.bump_identity_epoch();
        let bumped = state.cache_epoch();
        assert_ne!(epoch, bumped, "身份准入必须推进纪元");
        assert!(
            state.cached_stateless_answer(bumped, &key).is_none(),
            "非当前纪元的条目不得命中"
        );

        state.store_stateless_answer(bumped, key.clone(), response);
        assert!(
            state.cached_stateless_answer(bumped, &key).is_some(),
            "新纪元写入后恢复命中"
        );
        assert_eq!(state.perf.count("identity_epoch_bumps"), 1);
        assert_eq!(state.perf.count("answer_cache_hits"), 2);
        assert_eq!(state.perf.count("answer_cache_misses"), 1);
    }

    /// 运行时注入谓词对齐/实体别名也必须推进纪元并失效无状态缓存：
    /// CLI「确认身份」等签过字的说法在热路径注入后应立即生效且不污染旧缓存。
    #[tokio::test]
    async fn alignment_injection_invalidates_stateless_cache() {
        let state = test_state(4);
        let response = TurnResponse {
            session_id: "s-1".to_string(),
            turn: 1,
            status: "answered".to_string(),
            source: "reasoner".to_string(),
            kind: None,
            plan: Vec::new(),
            reply: "旧答案".to_string(),
            focus: None,
            policy: SessionPolicy::default(),
            epistemic: None,
            lexical: None,
            entities: Vec::new(),
            board: None,
            fault: None,
            open_turn: None,
            resumed: false,
            resumable: false,
            derivation: None,
        };
        let key = stateless_answer_key(&["明哥的身高是多少".to_string()], None);
        let epoch = state.cache_epoch();
        state.store_stateless_answer(epoch, key.clone(), response.clone());
        assert!(state.cached_stateless_answer(epoch, &key).is_some());

        state.align_entity_aliases(&[("明哥".to_string(), "姚明".to_string())]);
        let bumped = state.cache_epoch();
        assert_ne!(epoch, bumped);
        assert!(state.cached_stateless_answer(bumped, &key).is_none());

        state.align_predicate_pairs(
            &[("老师".to_string(), "师傅".to_string(), false)],
            &crate::predicate_registry::PredicateRegistry::default(),
        );
        let bumped2 = state.cache_epoch();
        assert_ne!(bumped, bumped2);
        assert!(state.cached_stateless_answer(bumped2, &key).is_none());
        assert!(state.perf.count("identity_epoch_bumps") >= 2);
    }

    #[tokio::test]
    async fn openai_compat_streaming_and_errors() {
        let (addr, _server) = start_server(4);
        let (status, head, body) = raw_request(
            addr,
            "POST",
            "/v1/chat/completions",
            &[("Content-Type", "application/json")],
            r#"{"messages": [{"role": "user", "content": "姚明的身高是多少"}], "stream": true}"#,
        )
        .await;
        assert_eq!(status, 200, "{}", body);
        assert!(head.contains("text/event-stream"), "{}", head);
        assert!(body.contains("chat.completion.chunk"), "{}", body);
        assert!(body.contains("[DONE]"), "{}", body);
        assert!(body.contains("226"), "{}", body);

        // 无用户消息 → 400
        let (status, _, _) = raw_request(
            addr,
            "POST",
            "/v1/chat/completions",
            &[],
            r#"{"messages": [{"role": "assistant", "content": "你好"}]}"#,
        )
        .await;
        assert_eq!(status, 400);
        // 未知会话 → 404
        let (status, _, _) = raw_request(
            addr,
            "POST",
            "/v1/chat/completions",
            &[],
            r#"{"messages": [{"role": "user", "content": "你好"}], "session_id": "s-404"}"#,
        )
        .await;
        assert_eq!(status, 404);
    }

    #[tokio::test]
    async fn rejects_bad_requests_and_limits_sessions() {
        let (addr, _server) = start_server(1);
        let session_id = create_session(addr).await;
        let (status, _, body) = raw_request(addr, "POST", "/sessions", &[], "").await;
        assert_eq!(status, 429, "{}", body);

        let (status, _, _) = raw_request(
            addr,
            "POST",
            &format!("/sessions/{}/turns", session_id),
            &[],
            "{not json",
        )
        .await;
        assert_eq!(status, 400);

        let (status, _, _) = raw_request(addr, "GET", "/sessions/missing", &[], "").await;
        assert_eq!(status, 404);
    }

    #[test]
    fn session_policy_parses_whitelist_and_echoes_unknown_clauses() {
        let texts = vec![
            "请用简短的一句话回答；回答必须给出依据；遇到不确定时先澄清，不要猜测。像莎士比亚一样说话".to_string(),
            "Answer concisely with citations".to_string(),
        ];
        let policy = SessionPolicy::from_system_texts(&texts);
        assert_eq!(policy.style, PolicyStyle::Concise);
        assert!(policy.require_evidence);
        assert!(policy.strict_clarify);
        assert!(policy.recognized.contains(&"style:concise".to_string()));
        assert!(policy.recognized.contains(&"require_evidence".to_string()));
        assert!(policy.recognized.contains(&"strict_clarify".to_string()));
        assert_eq!(policy.warnings.len(), 1, "{:?}", policy.warnings);
        assert!(policy.warnings[0].starts_with("像莎士比亚"));
        assert!(policy.render_text().contains("风格 concise"));

        let empty = SessionPolicy::from_system_texts(&[]);
        assert!(empty.is_default());
        let detailed = SessionPolicy::from_system_texts(&["请详细展开".to_string()]);
        assert_eq!(detailed.style, PolicyStyle::Detailed);
    }

    #[tokio::test]
    async fn deferred_turn_exposes_structured_lexical_diagnosis() {
        let (addr, _server) = start_server(2);
        let payload = serde_json::json!({
            "messages": [{"role": "user", "content": "今天天气怎么样"}]
        })
        .to_string();
        let (status, _, body) = raw_request(
            addr,
            "POST",
            "/v1/chat/completions",
            &[("Content-Type", "application/json")],
            &payload,
        )
        .await;
        assert_eq!(status, 200, "{}", body);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        let lexical = &value["knowprolog"]["lexical"];
        assert_eq!(lexical["status"], "unreadable", "{}", body);
        assert!(lexical["blocking"].as_u64().unwrap_or(0) >= 1);
        let residuals = lexical["residuals"].as_array().expect("应附残差表");
        assert!(residuals.iter().any(|residual| {
            residual["kind"] == "unsupported_question" && residual["repair"] == "rewrite_question"
        }));
        assert!(residuals
            .iter()
            .any(|residual| residual["kind"] == "unknown_token"));

        // 成功轮不额外分词，因而不带 lexical 字段。
        let payload = serde_json::json!({
            "messages": [{"role": "user", "content": "姚明的身高是多少"}]
        })
        .to_string();
        let (_, _, body) = raw_request(
            addr,
            "POST",
            "/v1/chat/completions",
            &[("Content-Type", "application/json")],
            &payload,
        )
        .await;
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert!(
            value["knowprolog"]["lexical"].is_null(),
            "成功轮不应生成读法诊断：{}",
            body
        );
    }

    #[tokio::test]
    async fn system_policy_shapes_chat_completion_turns() {
        let (addr, _server) = start_server(4);
        let payload = serde_json::json!({
            "messages": [
                {"role": "system", "content": "请用简短的一句话回答；必须给出依据。"},
                {"role": "user", "content": "姚明的身高是多少"},
            ]
        })
        .to_string();
        let (status, _, body) = raw_request(
            addr,
            "POST",
            "/v1/chat/completions",
            &[("Content-Type", "application/json")],
            &payload,
        )
        .await;
        assert_eq!(status, 200, "{}", body);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["knowprolog"]["policy"]["style"], "concise");
        assert_eq!(value["knowprolog"]["policy"]["require_evidence"], true);
        let content = value["choices"][0]["message"]["content"].as_str().unwrap();
        assert!(content.contains("226"), "{}", content);
        assert!(content.contains("依据：对话策略"), "{}", content);
        assert!(
            content.lines().count() <= 2,
            "简短策略不应多行：{}",
            content
        );

        // 详细策略：给出可继续了解的真实属性清单
        let payload = serde_json::json!({
            "messages": [
                {"role": "system", "content": "Please answer in depth."},
                {"role": "user", "content": "姚明是谁"},
            ]
        })
        .to_string();
        let (_, _, body) = raw_request(
            addr,
            "POST",
            "/v1/chat/completions",
            &[("Content-Type", "application/json")],
            &payload,
        )
        .await;
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        let content = value["choices"][0]["message"]["content"].as_str().unwrap();
        assert!(content.contains("可继续了解："), "{}", content);
        assert!(content.contains("身高"), "{}", content);

        // 严格澄清：让位轮变成确定性澄清问句
        let payload = serde_json::json!({
            "messages": [
                {"role": "system", "content": "遇到不确定时先澄清，不要猜测。"},
                {"role": "user", "content": "今天天气怎么样"},
            ]
        })
        .to_string();
        let (_, _, body) = raw_request(
            addr,
            "POST",
            "/v1/chat/completions",
            &[("Content-Type", "application/json")],
            &payload,
        )
        .await;
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["knowprolog"]["status"], "needs_clarification");
        assert_eq!(value["knowprolog"]["source"], "policy");
        assert!(value["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .contains("请补充实体名称"));
    }

    #[tokio::test]
    async fn prompts_endpoint_exposes_registry() {
        let (addr, _server) = start_server(2);
        let (status, _, body) = raw_request(addr, "GET", "/v1/prompts", &[], "").await;
        assert_eq!(status, 200);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["version"], prompt_spec::PROMPT_SPEC_VERSION);
        assert_eq!(value["count"], prompt_spec::default_specs().len());
        assert_eq!(value["lint"]["errors"], 0, "{}", body);
        assert_eq!(value["serving_mode"], "on");
        assert!(value["coverage"]["covered"].as_array().unwrap().len() >= 8);
        let first = &value["data"][0];
        assert!(first["id"].is_string() && first["template"].is_string());

        let (status, _, body) = raw_request(addr, "GET", "/", &[], "").await;
        assert_eq!(status, 200);
        assert!(body.contains("/v1/prompts"), "{}", body);
    }

    async fn take_turn_body(
        addr: std::net::SocketAddr,
        session_id: &str,
        input: &str,
    ) -> serde_json::Value {
        let (status, _, body) = raw_request(
            addr,
            "POST",
            &format!("/sessions/{}/turns", session_id),
            &[],
            input,
        )
        .await;
        assert_eq!(status, 200, "{}", body);
        serde_json::from_str(&body).unwrap()
    }

    async fn export_envelope(addr: std::net::SocketAddr, session_id: &str) -> serde_json::Value {
        let (status, _, body) = raw_request(
            addr,
            "GET",
            &format!("/sessions/{}/export", session_id),
            &[],
            "",
        )
        .await;
        assert_eq!(status, 200, "{}", body);
        serde_json::from_str(&body).unwrap()
    }

    #[tokio::test]
    async fn session_envelope_export_import_roundtrip_is_behaviorally_exact() {
        let (addr, _server) = start_server(4);
        let session_id = create_session(addr).await;
        take_turn_body(addr, &session_id, "姚明是谁").await;
        take_turn_body(addr, &session_id, "他的身高呢").await;

        let envelope = export_envelope(addr, &session_id).await;
        assert_eq!(envelope["format"], ENVELOPE_FORMAT);
        assert_eq!(envelope["version"], ENVELOPE_VERSION);
        assert_eq!(envelope["session_id"], session_id);
        assert_eq!(envelope["turns"], 2);
        assert_eq!(envelope["engine"]["kb_facts"], 2);
        assert_eq!(
            envelope["digest"].as_str().map(str::len),
            Some(16),
            "{}",
            envelope
        );
        // 信封是纯 JSON 值，摘要字段可独立复核
        let mut check: SessionEnvelope = serde_json::from_value(envelope.clone()).unwrap();
        let digest = check.digest.clone();
        check.seal();
        assert_eq!(check.digest, digest, "导出摘要必须自洽可复核");
        assert!(check.digest_valid());

        // 对照组：导出后原会话继续走一轮
        let original = take_turn_body(addr, &session_id, "姚明的国籍是什么").await;
        assert!(original["reply"].as_str().unwrap().contains("中国"));

        // 关闭原会话 → 反序列化信封重建同 id 会话
        let (status, _, _) = raw_request(
            addr,
            "DELETE",
            &format!("/sessions/{}", session_id),
            &[],
            "",
        )
        .await;
        assert_eq!(status, 200);
        let payload = serde_json::to_string(&envelope).unwrap();
        let (status, _, body) = raw_request(
            addr,
            "POST",
            "/sessions/import",
            &[("Content-Type", "application/json")],
            &payload,
        )
        .await;
        assert_eq!(status, 201, "{}", body);
        let imported: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(imported["session_id"], session_id);
        assert_eq!(imported["foreign_accepted"], false);
        assert_eq!(imported["decode"]["code"], "decode_ok", "{}", body);
        assert_eq!(imported["decode"]["version"], ENVELOPE_VERSION);
        assert_eq!(imported["decode"]["canonical"], true, "{}", body);

        // 恢复后的同一下一轮输出必须逐字一致（含轮号/状态/焦点/策略）
        let restored = take_turn_body(addr, &session_id, "姚明的国籍是什么").await;
        assert_eq!(original, restored, "恢复等价性：可观察输出逐字一致");
    }

    #[tokio::test]
    async fn import_gates_tampering_version_and_foreign_engine_with_zero_side_effects() {
        let (addr, _server) = start_server(4);
        let session_id = create_session(addr).await;
        take_turn_body(addr, &session_id, "姚明是谁").await;
        let envelope = export_envelope(addr, &session_id).await;

        let session_count = |value: &serde_json::Value| value["sessions"].as_u64().unwrap();

        // 1) 篡改内容但不重算摘要 → 409，且零副作用
        let mut tampered: SessionEnvelope = serde_json::from_value(envelope.clone()).unwrap();
        tampered.history.push("伪造的历史".to_string());
        let body = serde_json::to_string(&tampered).unwrap();
        let (status, _, response) = raw_request(addr, "POST", "/sessions/import", &[], &body).await;
        assert_eq!(status, 409, "{}", response);
        assert!(response.contains("digest"), "{}", response);
        let (_, _, metrics) = raw_request(addr, "GET", "/metrics", &[], "").await;
        let metrics: serde_json::Value = serde_json::from_str(&metrics).unwrap();
        assert_eq!(session_count(&metrics), 1, "校验失败不得产生会话");

        // 2) 未知版本 → 409
        let mut future: SessionEnvelope = serde_json::from_value(envelope.clone()).unwrap();
        future.version = ENVELOPE_VERSION + 1;
        future.seal();
        let body = serde_json::to_string(&future).unwrap();
        let (status, _, response) = raw_request(addr, "POST", "/sessions/import", &[], &body).await;
        assert_eq!(status, 409, "{}", response);
        assert!(response.contains("version"), "{}", response);

        // 3) 陌生引擎（crate 版本不同）→ 默认拒收
        let mut foreign: SessionEnvelope = serde_json::from_value(envelope.clone()).unwrap();
        foreign.session_id = "s-foreign".to_string();
        foreign.engine.crate_version = "0.0.0-foreign".to_string();
        foreign.seal();
        let body = serde_json::to_string(&foreign).unwrap();
        let (status, _, response) = raw_request(addr, "POST", "/sessions/import", &[], &body).await;
        assert_eq!(status, 409, "{}", response);
        assert!(response.contains("foreign envelope"), "{}", response);
        assert!(response.contains("crate_version"), "{}", response);

        // 4) 显式 accept_foreign → 201，并带回执字段，重建的会话可继续对话
        let (status, _, response) = raw_request(
            addr,
            "POST",
            "/sessions/import?accept_foreign=1",
            &[("Content-Type", "application/json")],
            &body,
        )
        .await;
        assert_eq!(status, 201, "{}", response);
        let imported: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert_eq!(imported["foreign_accepted"], true);
        assert_eq!(imported["foreign_fields"][0], "crate_version");
        let turn = take_turn_body(addr, &"s-foreign".to_string(), "他的身高呢").await;
        assert!(
            turn["reply"].as_str().unwrap().contains("226"),
            "外来信封恢复后应保留焦点记忆：{}",
            turn
        );

        // 5) id 冲突 → 409（不覆盖在用会话）
        let conflict_body = serde_json::to_string(&{
            let mut value: SessionEnvelope = serde_json::from_value(envelope.clone()).unwrap();
            value.seal();
            value
        })
        .unwrap();
        let (status, _, response) =
            raw_request(addr, "POST", "/sessions/import", &[], &conflict_body).await;
        assert_eq!(status, 409, "{}", response);
        assert!(response.contains("already exists"), "{}", response);

        // 其余会话不受任何导入尝试影响
        let turn = take_turn_body(addr, &session_id, "他的身高呢").await;
        assert!(turn["reply"].as_str().unwrap().contains("226"), "{}", turn);
    }

    /// 导入的信封可以占用服务端生成空间里的 id（如 s-2）：新建会话必须跳过
    /// 已占用的 id，绝不覆盖（尤其是刚导入的）会话。
    #[tokio::test]
    async fn create_session_never_overwrites_imported_ids() {
        let (addr, _server) = start_server(4);
        let original = create_session(addr).await; // s-1，游标推进到 2
        take_turn_body(addr, &original, "姚明是谁").await;
        let mut envelope: SessionEnvelope =
            serde_json::from_value(export_envelope(addr, &original).await).unwrap();
        envelope.session_id = "s-2".to_string(); // 模拟携带 s-2 的历史信封
        envelope.seal();

        let (status, _, _) =
            raw_request(addr, "DELETE", &format!("/sessions/{}", original), &[], "").await;
        assert_eq!(status, 200);
        let payload = serde_json::to_string(&envelope).unwrap();
        let (status, _, body) = raw_request(
            addr,
            "POST",
            "/sessions/import",
            &[("Content-Type", "application/json")],
            &payload,
        )
        .await;
        assert_eq!(status, 201, "{}", body);

        let next = create_session(addr).await;
        assert_ne!(next, "s-2", "服务端生成的 id 不得撞上已导入的会话");

        // 导入的会话仍然可用（焦点记忆未被覆盖）
        let turn = take_turn_body(addr, &"s-2".to_string(), "他的身高呢").await;
        assert!(turn["reply"].as_str().unwrap().contains("226"), "{}", turn);
    }

    /// 导出的信封必须能原样导回：超过默认 256KB 体量上限的信封
    /// 在 `/sessions/import` 上按信封上限（16MB）放行。
    #[tokio::test]
    async fn import_accepts_envelopes_larger_than_default_body_limit() {
        let (addr, _server) = start_server(4);
        let session_id = create_session(addr).await;
        take_turn_body(addr, &session_id, "姚明是谁").await;
        let mut envelope: SessionEnvelope =
            serde_json::from_value(export_envelope(addr, &session_id).await).unwrap();
        envelope.history.push("长".repeat(300_000));
        envelope.seal();
        let payload = serde_json::to_string(&envelope).unwrap();
        assert!(
            payload.len() > DEFAULT_MAX_BODY_BYTES,
            "测试信封必须超过默认体量上限：{} 字节",
            payload.len()
        );

        let (status, _, _) = raw_request(
            addr,
            "DELETE",
            &format!("/sessions/{}", session_id),
            &[],
            "",
        )
        .await;
        assert_eq!(status, 200);
        let (status, _, body) = raw_request(
            addr,
            "POST",
            "/sessions/import",
            &[("Content-Type", "application/json")],
            &payload,
        )
        .await;
        assert_eq!(status, 201, "{}", body);

        // 大信封重建后仍可对话：焦点记忆完好
        let turn = take_turn_body(addr, &session_id, "他的身高呢").await;
        assert!(turn["reply"].as_str().unwrap().contains("226"), "{}", turn);
    }

    /// 普通路由（非信封导入）体量超限时明确返回 413，而不是 400。
    #[tokio::test]
    async fn oversized_body_on_regular_route_gets_413() {
        let (addr, _server) = start_server(2);
        let mut stream = TcpStream::connect(addr).await.unwrap();
        // 只声明超限的 Content-Length，不真正发送 256KB 体：
        // 服务端在读取体之前就必须拒绝，避免连接中止导致测试抖动。
        let request = format!(
            "POST /sessions HTTP/1.1\r\nHost: localhost\r\nContent-Length: {}\r\n\r\n",
            DEFAULT_MAX_BODY_BYTES + 1
        );
        stream.write_all(request.as_bytes()).await.unwrap();
        let mut response = Vec::new();
        stream.read_to_end(&mut response).await.unwrap();
        let text = String::from_utf8_lossy(&response);
        assert!(text.starts_with("HTTP/1.1 413"), "{}", text);
    }

    #[tokio::test]
    async fn concurrent_sessions_are_isolated_actors() {
        let (addr, _server) = start_server(4);
        let first = create_session(addr).await;
        let second = create_session(addr).await;
        let first_path = format!("/sessions/{}/turns", first);
        let second_path = format!("/sessions/{}/turns", second);
        let (a, b) = tokio::join!(
            raw_request(addr, "POST", &first_path, &[], "姚明的身高是多少"),
            raw_request(addr, "POST", &second_path, &[], "姚明的国籍是什么"),
        );
        assert_eq!(a.0, 200, "{}", a.2);
        assert!(a.2.contains("226"), "{}", a.2);
        assert_eq!(b.0, 200, "{}", b.2);
        assert!(b.2.contains("中国"), "{}", b.2);
        // 指代各自解析到各会话自己的焦点，不串台
        let a2 = take_turn_body(addr, &first, "他的国籍呢").await;
        let b2 = take_turn_body(addr, &second, "他的身高呢").await;
        assert!(a2["reply"].as_str().unwrap().contains("中国"), "{}", a2);
        assert!(b2["reply"].as_str().unwrap().contains("226"), "{}", b2);
    }

    #[tokio::test]
    async fn cold_sessions_freeze_and_resume_across_capacity() {
        let dir = temp_session_dir("cold");
        let (addr, _server) =
            start_server_with_options(1, RecoveryMode::Observe, Some(dir.clone()), None);
        let first = create_session(addr).await;
        take_turn_body(addr, &first, "姚明是谁").await;

        // 容量 1：新建第二个会话必须把 LRU（first）冻结到磁盘
        let second = create_session(addr).await;
        assert!(
            dir.join(format!("{}.json", first)).exists(),
            "冷会话应落盘：{:?}",
            dir
        );
        let (_, _, metrics) = raw_request(addr, "GET", "/metrics", &[], "").await;
        let metrics: serde_json::Value = serde_json::from_str(&metrics).unwrap();
        assert_eq!(metrics["sessions"], 1);
        assert_eq!(metrics["evictions_total"], 1, "{}", metrics);
        assert_eq!(metrics["cold_sessions"], 1, "{}", metrics);
        assert!(metrics["session_dir"]
            .as_str()
            .unwrap()
            .contains("kp-serve"));

        take_turn_body(addr, &second, "姚明是谁").await;

        // 访问冷会话 → 反序列化换入（焦点/轮号随快照恢复），并冻结当前 LRU
        let resumed = take_turn_body(addr, &first, "他的身高呢").await;
        assert!(
            resumed["reply"].as_str().unwrap().contains("226"),
            "焦点记忆应随反序列化恢复：{}",
            resumed
        );
        assert_eq!(resumed["turn"], 2, "轮号应连续：{}", resumed);
        assert_eq!(resumed["focus"], "姚明");
        assert!(
            dir.join(format!("{}.json", second)).exists(),
            "换入 first 时应把 second 冻结"
        );

        // DELETE 清除内存与冷存储：已删除会话不得从磁盘复活
        let (status, _, _) =
            raw_request(addr, "DELETE", &format!("/sessions/{}", first), &[], "").await;
        assert_eq!(status, 200);
        assert!(
            !dir.join(format!("{}.json", first)).exists(),
            "删除会话应清理冷存储"
        );
        let (status, _, _) = raw_request(
            addr,
            "POST",
            &format!("/sessions/{}/turns", first),
            &[],
            "你好",
        )
        .await;
        assert_eq!(status, 404, "已删除的会话不得复活");

        // second 仍在冷存储，可恢复并继续多轮记忆
        let reopened = take_turn_body(addr, &second, "他的国籍呢").await;
        assert!(
            reopened["reply"].as_str().unwrap().contains("中国"),
            "{}",
            reopened
        );
        assert_eq!(reopened["turn"], 2, "{}", reopened);

        // 冷会话也必须可删除：再次换出 second 后 DELETE 应清理磁盘且不得复活
        let third = create_session(addr).await;
        assert!(
            dir.join(format!("{}.json", second)).exists(),
            "新建 third 应换出 second"
        );
        let (status, _, _) =
            raw_request(addr, "DELETE", &format!("/sessions/{}", second), &[], "").await;
        assert_eq!(status, 200, "冷会话删除应成功");
        assert!(!dir.join(format!("{}.json", second)).exists());
        let (status, _, _) = raw_request(
            addr,
            "POST",
            &format!("/sessions/{}/turns", second),
            &[],
            "你好",
        )
        .await;
        assert_eq!(status, 404, "已删除的冷会话不得复活");
        let _ = third;

        let _ = std::fs::remove_dir_all(&dir);
    }

    fn test_session(budget: TurnBudget) -> Session {
        let state = test_state(4);
        build_session(
            "s-test".to_string(),
            &state.warm,
            &state.perf,
            state.prompt_mode,
            state.recovery_mode,
            None,
            None,
            None,
            budget,
            None,
        )
    }

    #[test]
    fn deadline_rolls_back_and_keeps_a_resumable_turn() {
        let budget = TurnBudget::enforced(Duration::from_millis(1));
        let mut session = test_session(budget);
        let mut control = TurnControl::starting_at(
            1,
            budget,
            Instant::now()
                .checked_sub(Duration::from_secs(1))
                .expect("测试起点"),
        );
        let response = session.run_turn_checked("姚明的身高是多少", &mut control, false, 1);
        assert_eq!(response.status, "deadline", "{}", response.reply);
        assert!(response.fault.is_some());
        let open = response
            .open_turn
            .as_ref()
            .expect("超时轮必须留下未完成记忆");
        assert_eq!(open.input, "姚明的身高是多少");
        assert_eq!(open.attempts, 1);
        assert!(response.resumable);
        assert_eq!(session.turns, 1, "失败轮仍占一个轮号");
        assert!(
            session.manager.focus_entity().is_none(),
            "失败轮不得提交焦点"
        );
        assert_eq!(session.manager.turn_count(), 0, "失败轮不得进入对话本体");

        // 预算恢复后继续同一问：先取走再重放，成功即清空未完成记忆。
        session.turn_budget = TurnBudget::unlimited();
        let resumed = session
            .resume_turn(CancelToken::new())
            .expect("未完成轮次应可继续");
        assert!(resumed.resumed);
        assert_eq!(resumed.status, "answered", "{}", resumed.reply);
        assert!(resumed.reply.contains("226"), "{}", resumed.reply);
        assert!(resumed.open_turn.is_none());
        assert!(session.manager.latest_open_turn().is_none());
    }

    #[test]
    fn cancellation_is_typed_and_rolls_back_observable_state() {
        let mut session = test_session(TurnBudget::unlimited());
        let cancel = CancelToken::new();
        cancel.cancel();
        let response = session.run_turn_with_cancel("姚明的身高是多少", cancel);
        assert_eq!(response.status, "cancelled", "{}", response.reply);
        assert_eq!(
            response.fault.as_ref().map(|fault| fault.kind),
            Some(TurnFaultKind::Cancelled)
        );
        assert!(response.open_turn.is_some());
        assert!(session.manager.focus_entity().is_none());
        assert_eq!(session.manager.turn_count(), 0);
    }

    #[test]
    fn plain_turn_still_commits_with_default_budget() {
        let mut session = test_session(TurnBudget::unlimited());
        let response = session.run_turn("姚明的身高是多少");
        assert_eq!(response.status, "answered", "{}", response.reply);
        assert!(response.reply.contains("226"), "{}", response.reply);
        assert!(response.fault.is_none());
        assert!(response.open_turn.is_none());
        assert_eq!(session.manager.turn_count(), 1);
    }

    #[test]
    fn assumption_world_spans_turns_and_rolls_back_on_exit() {
        let mut session = test_session(TurnBudget::unlimited());
        let entered = session.run_turn("假设：姚明的身高是230");
        assert_eq!(entered.status, "answered", "{}", entered.reply);
        assert_eq!(entered.source, "assumption", "{}", entered.reply);
        assert!(entered.reply.contains("假设世界"), "{}", entered.reply);

        let inside = session.run_turn("姚明的身高是多少");
        assert_eq!(inside.status, "answered", "{}", inside.reply);
        assert!(inside.reply.contains("230"), "{}", inside.reply);

        let exited = session.run_turn("退出假设");
        assert_eq!(exited.source, "assumption", "{}", exited.reply);
        assert!(exited.reply.contains("已退出"), "{}", exited.reply);
        let outside = session.run_turn("姚明的身高是多少");
        assert!(outside.reply.contains("226"), "{}", outside.reply);
        assert!(!outside.reply.contains("230"), "{}", outside.reply);
    }

    #[test]
    fn completion_commands_are_available_over_session_turns() {
        let mut facts = vec![
            PrologFact::Relation {
                from: "张三".into(),
                rel_type: "师傅".into(),
                to: "李四".into(),
            },
            PrologFact::Relation {
                from: "李四".into(),
                rel_type: "师傅".into(),
                to: "王五".into(),
            },
            PrologFact::Relation {
                from: "王五".into(),
                rel_type: "师傅".into(),
                to: "赵六".into(),
            },
            PrologFact::Relation {
                from: "张三".into(),
                rel_type: "师祖".into(),
                to: "王五".into(),
            },
        ];
        for (from, to) in [
            ("孙八", "周九"),
            ("周九", "吴十"),
            ("郑十一", "钱十二"),
            ("钱十二", "孙十三"),
        ] {
            facts.push(PrologFact::Relation {
                from: from.into(),
                rel_type: "师傅".into(),
                to: to.into(),
            });
        }
        let reasoning = ReasoningConfig {
            enable_link_prediction: true,
            link_prediction_min_confidence: 0.1,
            link_prediction_min_support: 1,
            completion_min_score: 0.3,
            ..ReasoningConfig::default()
        };
        let state = Arc::new(ApiState::new(
            facts,
            reasoning,
            HashMap::new(),
            PerfLedger::shared(),
            4,
            StartupLedger::shared(),
        ));
        let mut session = build_session(
            "s-test".to_string(),
            &state.warm,
            &state.perf,
            state.prompt_mode,
            state.recovery_mode,
            None,
            None,
            None,
            TurnBudget::unlimited(),
            None,
        );

        let opened = session.run_turn("补全回合：孙八");
        assert_eq!(opened.source, "completion", "{}", opened.reply);
        assert!(opened.reply.contains("补全回合"), "{}", opened.reply);
        assert!(opened.reply.contains("师祖"), "{}", opened.reply);

        let confirmed = session.run_turn("确认补全：孙八|师祖|吴十");
        assert_eq!(confirmed.source, "completion", "{}", confirmed.reply);
        assert!(
            confirmed.reply.contains("已确认补全"),
            "{}",
            confirmed.reply
        );

        let answer = session.run_turn("孙八的师祖是谁");
        assert_eq!(answer.status, "answered", "{}", answer.reply);
        assert!(answer.reply.contains("吴十"), "{}", answer.reply);

        let report = session.run_turn("补全报告");
        assert_eq!(report.source, "completion", "{}", report.reply);
        assert!(report.reply.contains("已确认 1"), "{}", report.reply);
    }

    #[test]
    fn predictive_commands_are_available_over_session_turns() {
        let facts = vec![
            PrologFact::Relation {
                from: "张三".into(),
                rel_type: "师傅".into(),
                to: "李四".into(),
            },
            PrologFact::Relation {
                from: "李四".into(),
                rel_type: "师傅".into(),
                to: "王五".into(),
            },
            PrologFact::Relation {
                from: "王五".into(),
                rel_type: "师傅".into(),
                to: "赵六".into(),
            },
            PrologFact::Relation {
                from: "张三".into(),
                rel_type: "师祖".into(),
                to: "王五".into(),
            },
        ];
        let reasoning = ReasoningConfig {
            enable_link_prediction: true,
            link_prediction_min_confidence: 0.5,
            link_prediction_min_support: 1,
            ..ReasoningConfig::default()
        };
        let state = Arc::new(ApiState::new(
            facts,
            reasoning,
            HashMap::new(),
            PerfLedger::shared(),
            4,
            StartupLedger::shared(),
        ));
        let mut session = build_session(
            "s-predictive".to_string(),
            &state.warm,
            &state.perf,
            state.prompt_mode,
            state.recovery_mode,
            None,
            None,
            None,
            TurnBudget::unlimited(),
            None,
        );

        let opened = session.run_turn("预测展开：李四");
        assert_eq!(opened.source, "predictive", "{}", opened.reply);
        assert!(
            opened.reply.contains("推测，不是已确认事实"),
            "{}",
            opened.reply
        );
        assert!(opened.reply.contains("赵六"), "{}", opened.reply);

        let id = session
            .reasoner
            .predictive_ledger()
            .pending()
            .iter()
            .find(|hypothesis| hypothesis.key.object == "赵六")
            .map(|hypothesis| hypothesis.id.clone())
            .expect("应存在指向赵六的待定推测");
        let confirmed = session.run_turn(&format!("确认预测：{}", id));
        assert_eq!(confirmed.source, "predictive", "{}", confirmed.reply);
        assert!(confirmed.reply.contains("已确认预测"), "{}", confirmed.reply);
        let profile = session
            .reasoner
            .generate_about("李四")
            .expect("确认后的会话事实应参与介绍");
        assert!(profile.contains("赵六"), "{}", profile);

        let report = session.run_turn("预测台账");
        assert_eq!(report.source, "predictive", "{}", report.reply);
        assert!(report.reply.contains("已确认 1"), "{}", report.reply);
    }

    #[test]
    fn why_turn_carries_bounded_derivation_over_http_response() {
        let mut session = test_session(TurnBudget::unlimited());
        let answer = session.run_turn("姚明的身高是多少");
        assert_eq!(answer.status, "answered", "{}", answer.reply);
        assert!(answer.derivation.is_none(), "普通回答不附带证明");

        let why = session.run_turn("怎么推出来的？");
        assert_eq!(why.kind.as_deref(), Some("meta_why"), "{}", why.reply);
        let derivation = why.derivation.as_ref().expect("元解释轮应返回结构化推导");
        assert!(derivation.is_proved(), "{}", why.reply);
        assert!(
            derivation
                .steps
                .iter()
                .any(|step| step.claim.contains("226")),
            "{:?}",
            derivation.steps
        );
        let encoded = serde_json::to_string(derivation).expect("推导视图应可序列化");
        assert!(encoded.contains("steps"));
        assert!(encoded.len() < crate::derivation::MAX_DERIVATION_BYTES);
    }

    #[test]
    fn resume_without_open_turn_is_rejected() {
        let mut session = test_session(TurnBudget::unlimited());
        assert!(matches!(
            session.resume_turn(CancelToken::new()),
            Err(SessionError::NoOpenTurn)
        ));
    }

    #[tokio::test]
    async fn resume_route_reports_missing_open_turn() {
        let (addr, _server) = start_server(1);
        let session_id = create_session(addr).await;
        let (status, _, body) = raw_request(
            addr,
            "POST",
            &format!("/sessions/{}/resume", session_id),
            &[],
            "",
        )
        .await;
        assert_eq!(status, 409, "{}", body);
        assert!(body.contains("resumable"), "{}", body);
    }

    #[tokio::test]
    async fn discard_open_turn_route_clears_resume_state() {
        let (addr, _server) = start_server_with_budget(1, TurnBudget::enforced(Duration::ZERO));
        let session_id = create_session(addr).await;
        let turn = take_turn_body(addr, &session_id, "姚明的身高是多少").await;
        assert_eq!(turn["status"], "deadline", "{}", turn);

        let (status, _, body) = raw_request(
            addr,
            "DELETE",
            &format!("/sessions/{}/open-turn", session_id),
            &[],
            "",
        )
        .await;
        assert_eq!(status, 200, "{}", body);
        let discarded: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(discarded["discarded"], true, "{}", body);
        assert_eq!(
            discarded["open_turn"]["input"], "姚明的身高是多少",
            "{}",
            body
        );

        let (status, _, body) =
            raw_request(addr, "GET", &format!("/sessions/{}", session_id), &[], "").await;
        assert_eq!(status, 200, "{}", body);
        let snapshot: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert!(snapshot["open_turn"].is_null(), "{}", body);
        assert_eq!(snapshot["resumable"], false, "{}", body);

        let (status, _, body) = raw_request(
            addr,
            "POST",
            &format!("/sessions/{}/resume", session_id),
            &[],
            "",
        )
        .await;
        assert_eq!(status, 409, "{}", body);
    }

    #[tokio::test]
    async fn pending_question_roundtrip_over_http_session_turns() {
        let facts = vec![
            PrologFact::Relation {
                from: "张三".into(),
                rel_type: "师傅".into(),
                to: "李四".into(),
            },
            PrologFact::Relation {
                from: "王五".into(),
                rel_type: "师傅".into(),
                to: "赵六".into(),
            },
        ];
        let reasoning = ReasoningConfig {
            enable_link_prediction: false,
            ..ReasoningConfig::default()
        };
        let state = Arc::new(ApiState::new(
            facts,
            reasoning,
            HashMap::new(),
            PerfLedger::shared(),
            2,
            StartupLedger::shared(),
        ));
        let session_id = state.create_session().await.expect("会话应创建成功");

        let ask = state
            .take_turn(&session_id, "张三的师傅的师傅是谁")
            .await
            .expect("缺失前提时应返回追问");
        assert!(ask.reply.contains("李四 的 师傅"), "{}", ask.reply);
        assert!(
            ask.reply.contains("没有记录"),
            "先诚实报告缺依据，再追问：{}",
            ask.reply
        );

        let answered = state
            .take_turn(&session_id, "王五")
            .await
            .expect("短答应回填悬问");
        assert_eq!(answered.source, "pending_question", "{}", answered.reply);
        assert!(
            answered.reply.contains("现在可以回答了"),
            "{}",
            answered.reply
        );
        assert!(answered.reply.contains("王五"), "{}", answered.reply);
    }

    #[tokio::test]
    async fn http_deadline_is_resumable_until_attempts_exhausted() {
        let (addr, _server) = start_server_with_budget(1, TurnBudget::enforced(Duration::ZERO));
        let session_id = create_session(addr).await;
        let turn = take_turn_body(addr, &session_id, "姚明的身高是多少").await;
        assert_eq!(turn["status"], "deadline", "{}", turn);
        assert_eq!(turn["resumable"], true, "{}", turn);
        assert_eq!(turn["open_turn"]["input"], "姚明的身高是多少", "{}", turn);
        assert_eq!(turn["open_turn"]["attempts"], 1, "{}", turn);

        let (status, _, body) =
            raw_request(addr, "GET", &format!("/sessions/{}", session_id), &[], "").await;
        assert_eq!(status, 200, "{}", body);
        let snapshot: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(snapshot["open_turn"]["attempts"], 1, "{}", body);
        assert_eq!(snapshot["resumable"], true, "{}", body);

        for attempts in 2..=3 {
            let (status, _, body) = raw_request(
                addr,
                "POST",
                &format!("/sessions/{}/resume", session_id),
                &[],
                "",
            )
            .await;
            assert_eq!(status, 200, "{}", body);
            let resumed: serde_json::Value = serde_json::from_str(&body).unwrap();
            assert_eq!(resumed["status"], "deadline", "{}", body);
            assert_eq!(resumed["resumed"], true, "{}", body);
            assert_eq!(resumed["open_turn"]["attempts"], attempts, "{}", body);
            assert_eq!(
                resumed["resumable"],
                attempts < 3,
                "attempts={} body={}",
                attempts,
                body
            );
        }

        let (status, _, body) = raw_request(
            addr,
            "POST",
            &format!("/sessions/{}/resume", session_id),
            &[],
            "",
        )
        .await;
        assert_eq!(status, 409, "{}", body);
    }
    #[tokio::test]
    async fn events_endpoint_extracts_and_answers() {
        let request = Request {
            method: "POST".to_string(),
            path: "/v1/events".to_string(),
            query: BTreeMap::new(),
            headers: BTreeMap::new(),
            body: r#"{"text":"1991年，张三认识李四。","question":"谁认识李四"}"#
                .as_bytes()
                .to_vec(),
        };
        let response = handle_events(&request).await;
        assert_eq!(
            response.status,
            200,
            "{}",
            String::from_utf8_lossy(&response.body)
        );
        let value: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(value["status"], "answered");
        assert!(value["answer"]["answer"].as_str().unwrap().contains("张三"));
        assert_eq!(value["graph"]["events"].as_array().unwrap().len(), 1);
    }
}

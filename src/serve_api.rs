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
//! - `GET  /sessions/{id}/export`  导出会话信封（可移植 JSON）
//! - `POST /sessions/import`       导入会话信封重建会话
//! - `DELETE /sessions/{id}`       关闭会话
//! - `POST /sessions/{id}/turns`   一轮对话；`Accept: text/event-stream`
//!   或 `?stream=1` 时按句 SSE 流式返回
//! - `GET  /v1/models`             OpenAI 兼容：模型清单
//! - `GET  /v1/prompts`            提示资产注册表（spec + lint + 覆盖）
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

use crate::cache::RevisionLru;
use crate::canonical::{self, DecodeError, DecodeReceipt};
use crate::dialogue_manager::{DialogueKnowledge, DialogueManager, DialogueManagerSnapshot};
use crate::recovery::RecoveryMode;
use crate::perf_ledger::PerfLedger;
use crate::prolog_engine::PrologFact;
use crate::prompt_spec::{self, PromptMode};
use crate::startup_ledger::StartupLedger;
use crate::symbolic_reasoner::{DialogueSnapshot, SymbolicReasoner};
use crate::ReasoningConfig;

const MAX_HEADER_BYTES: usize = 16 * 1024;
const DEFAULT_MAX_BODY_BYTES: usize = 256 * 1024;
/// 冷存储信封的体量上限（解码绝不因巨量输入拖垮进程）。
const SESSION_ENVELOPE_MAX_BYTES: usize = 16 * 1024 * 1024;
const CONNECTION_TIMEOUT: Duration = Duration::from_secs(30);
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
    "简短", "简洁", "一句话", "尽量短", "concise", "brief", "short",
];
const STYLE_DETAILED_PHRASES: &[&str] = &[
    "详细", "展开", "深入", "in depth", "detailed", "elaborate", "verbose",
];
const EVIDENCE_PHRASES: &[&str] = &[
    "依据", "来源", "出处", "证据", "引用", "evidence", "cite", "citation", "source",
    "provenance", "justify",
];
const STRICT_CLARIFY_PHRASES: &[&str] = &[
    "澄清", "追问", "不要猜", "不要臆测", "先确认", "clarify", "ask before",
    "do not guess", "don't guess", "no guessing", "confirm",
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
                    let already_offers = turn.reply.contains("还可以") || turn.reply.contains("可继续");
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
    /// 接地清单：本轮输入中链接到知识图谱的实体（词面 ↔ URI ↔ 类型 ↔ 角色）。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub entities: Vec<GroundingCitation>,
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
}

impl Session {
    fn run_turn(&mut self, input: &str) -> TurnResponse {
        let input = input.trim().to_string();
        self.turns += 1;
        if self.history.len() >= HISTORY_LIMIT {
            self.history.remove(0);
        }
        self.history.push(input.clone());
        self.manager.clear_last_recovery();

        let dialogue = self
            .manager
            .plan_pure_dialogue(&input, &self.reasoner)
            .or_else(|| self.manager.plan_reply(&input, &self.reasoner));
        let (status, source, kind, plan, reply) = if let Some(reply) = dialogue {
            (
                "answered".to_string(),
                "dialogue".to_string(),
                Some(reply.kind.label().to_string()),
                reply.plan,
                reply.text,
            )
        } else if let Some(answer) = self.reasoner.answer(&input) {
            (
                "answered".to_string(),
                "reasoner".to_string(),
                None,
                Vec::new(),
                answer,
            )
        } else {
            // 让位也是失败：以 Unknown 回执记账（Off 模式零副作用）
            self.manager.note_deferred(&input, &self.reasoner);
            (
                "deferred".to_string(),
                "none".to_string(),
                None,
                Vec::new(),
                String::new(),
            )
        };
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
            entities,
        };
        self.policy.apply(&input, &mut response, &self.reasoner);
        response
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
    pub fn decode(
        bytes: &[u8],
        max_bytes: usize,
    ) -> Result<(Self, DecodeReceipt), DecodeError> {
        let (envelope, mut receipt) = canonical::decode_json::<Self>(
            bytes,
            "session_envelope",
            ENVELOPE_VERSION,
            max_bytes,
        )?;
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
        reply: oneshot::Sender<TurnResponse>,
    },
    Snapshot {
        reply: oneshot::Sender<serde_json::Value>,
    },
    Export {
        reply: oneshot::Sender<SessionEnvelope>,
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
        let mut reasoner =
            SymbolicReasoner::with_perf_ledger(reasoning.clone(), synonyms.clone(), Arc::clone(perf));
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
    restore: Option<SessionRestore>,
) -> Session {
    let reasoner = warm.fork();
    perf.bump("session_forks");
    let mut manager = DialogueManager::new();
    manager.set_prompt_mode(prompt_mode);
    manager.set_recovery_mode(recovery_mode);
    let mut session = Session {
        id,
        turns: 0,
        manager,
        reasoner,
        history: Vec::new(),
        policy: SessionPolicy::default(),
    };
    if let Some(restore) = restore {
        session.turns = restore.turns;
        session.history = restore.history;
        session.policy = restore.policy;
        session.manager.restore(&restore.manager);
        session.reasoner.restore_dialogue_snapshot(&restore.reasoner);
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
            &facts,
            &reasoning,
            &synonyms,
            &perf,
            &startup,
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
        self.perf
            .bump_by("answer_cache_evictions", evicted as u64);
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
        let engine = Arc::new(self.engine_stamp());
        let thread_name = format!("kp-session-{}", id);
        std::thread::Builder::new()
            .name(thread_name)
            .spawn(move || {
                // 只计量「分叉到就绪」：guard 在 session 构造完成时立即落账，
                // 绝不能覆盖整个 actor 生命周期（那会拖到线程退出才记账）
                let mut session = {
                    let _span = startup.span("session_fork");
                    build_session(id, &warm, &perf, prompt_mode, recovery_mode, restore)
                };
                while let Some(command) = rx.blocking_recv() {
                    match command {
                        SessionCommand::Turn {
                            input,
                            policy,
                            reply,
                        } => {
                            if let Some(policy) = policy {
                                session.policy = policy;
                            }
                            let response = session.run_turn(&input);
                            let _ = reply.send(response);
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
        let id = format!("s-{}", self.next_session_id.fetch_add(1, Ordering::Relaxed));
        let handle = self.spawn_actor(id.clone(), None)?;
        sessions.insert(id.clone(), handle);
        Ok(id)
    }

    async fn take_turn(&self, session_id: &str, input: &str) -> Result<TurnResponse, SessionError> {
        self.take_turn_with_policy(session_id, input, None).await
    }

    /// 执行一轮；给出 `policy` 时先更新会话策略再执行。
    async fn take_turn_with_policy(
        &self,
        session_id: &str,
        input: &str,
        policy: Option<&SessionPolicy>,
    ) -> Result<TurnResponse, SessionError> {
        let tx = self.session_tx(session_id).await?;
        let (reply_tx, reply_rx) = oneshot::channel();
        tx.send(SessionCommand::Turn {
            input: input.to_string(),
            policy: policy.cloned(),
            reply: reply_tx,
        })
        .await
        .map_err(|_| SessionError::Unavailable)?;
        let response = reply_rx.await.map_err(|_| SessionError::Unavailable)?;
        self.turns_total.fetch_add(1, Ordering::Relaxed);
        Ok(response)
    }

    async fn session_snapshot(
        &self,
        session_id: &str,
    ) -> Result<serde_json::Value, SessionError> {
        let tx = self.session_tx(session_id).await?;
        let (reply_tx, reply_rx) = oneshot::channel();
        tx.send(SessionCommand::Snapshot { reply: reply_tx })
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
    state
        .startup
        .record_since("bind", bind_start);
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
        Err(message) => {
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
    let segments: Vec<&str> = path.split('/').filter(|segment| !segment.is_empty()).collect();
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
                    "GET /sessions/{id}/export",
                    "POST /sessions/import",
                    "DELETE /sessions/{id}",
                    "POST /sessions/{id}/turns",
                    "GET /v1/models",
                    "GET /v1/prompts",
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
            }),
        ),
        ("GET", ["metrics"]) => Response::json(
            200,
            &serde_json::json!({
                "sessions": state.session_count().await,
                "turns_total": state.turns_total.load(Ordering::Relaxed),
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
        ("GET", ["sessions", id]) => match state.session_snapshot(id).await {
            Ok(snapshot) => Response::json(200, &snapshot),
            Err(error) => session_error_response(error),
        },
        ("GET", ["sessions", id, "export"]) => match state.export_session(id).await {
            Ok(envelope) => Response::json(
                200,
                &serde_json::to_value(&envelope).unwrap_or_else(|_| serde_json::json!({})),
            ),
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
        SessionError::Stale(error) => Response::error(
            409,
            &format!(
                "session snapshot is stale or from a different engine (decode_{}); re-import it explicitly",
                error.code()
            ),
        ),
    }
}

/// 解码失败 → HTTP 响应：保留原有人类可读措辞，附加稳定错误码。
fn import_decode_response(state: &Arc<ApiState>, error: &DecodeError) -> Response {
    state.perf.bump(&error.metric_key());
    match error {
        DecodeError::TooLarge { .. } => Response::error(413, "envelope too large [decode_too_large]"),
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
async fn handle_import(state: &Arc<ApiState>, request: &Request) -> Response {
    let (envelope, receipt) = match SessionEnvelope::decode(&request.body, state.max_body_bytes) {
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
        let role = message.get("role").and_then(|role| role.as_str()).unwrap_or("");
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
        stream: value.get("stream").and_then(|stream| stream.as_bool()).unwrap_or(false),
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
        let input = chat.inputs.last().expect("inputs non-empty by construction");
        return match state
            .take_turn_with_policy(session_id, input, chat.policy.as_ref())
            .await
        {
            Ok(turn) => Ok((turn, prompt_chars)),
            Err(SessionError::NotFound) => Err((404, "session not found".to_string())),
            Err(SessionError::Unavailable) => {
                Err((500, "session actor unavailable".to_string()))
            }
            Err(SessionError::Stale(error)) => Err((
                409,
                format!("session snapshot is stale (decode_{})", error.code()),
            )),
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
    for input in &chat.inputs {
        last = state
            .take_turn_with_policy(&temp_id, input, chat.policy.as_ref())
            .await
            .ok();
    }
    state.close_session(&temp_id).await;
    let turn = last.ok_or_else(|| (500, "turn failed".to_string()))?;
    // 只缓存纯函数结果：无会话状态残留，命中与重算逐字节一致
    state.store_stateless_answer(epoch, key, turn.clone());
    Ok((turn, prompt_chars))
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
        stream.write_all(format!("data: {}\n\n", value).as_bytes()).await
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
    let turn = match state.take_turn(session_id, &input).await {
        Ok(turn) => turn,
        Err(error) => {
            write_response(stream, session_error_response(error)).await?;
            return Ok(());
        }
    };

    let header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
    stream.write_all(header.as_bytes()).await?;
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

async fn read_request(stream: &mut TcpStream, max_body: usize) -> Result<Request, String> {
    let mut buffer = Vec::with_capacity(4096);
    let mut chunk = [0u8; 4096];
    let header_end = loop {
        if let Some(position) = find_subsequence(&buffer, b"\r\n\r\n") {
            break position;
        }
        if buffer.len() > MAX_HEADER_BYTES {
            return Err("header too large".to_string());
        }
        let read = stream
            .read(&mut chunk)
            .await
            .map_err(|error| format!("read error: {}", error))?;
        if read == 0 {
            return Err("connection closed before header".to_string());
        }
        buffer.extend_from_slice(&chunk[..read]);
    };

    let header_text = String::from_utf8_lossy(&buffer[..header_end]).to_string();
    let mut lines = header_text.split("\r\n");
    let request_line = lines.next().ok_or_else(|| "missing request line".to_string())?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next().ok_or_else(|| "missing method".to_string())?.to_string();
    let target = parts.next().ok_or_else(|| "missing target".to_string())?.to_string();
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
    if content_length > max_body {
        return Err("body too large".to_string());
    }
    let mut body = buffer[header_end + 4..].to_vec();
    while body.len() < content_length {
        let read = stream
            .read(&mut chunk)
            .await
            .map_err(|error| format!("read error: {}", error))?;
        if read == 0 {
            return Err("connection closed before body".to_string());
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
        _ => "Internal Server Error",
    }
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
            SessionEnvelope::decode(
                &canonical::canonical_bytes(&wrong_format).unwrap(),
                1 << 20
            )
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
            start_server_with_options(1, RecoveryMode::Observe, Some(dir.clone()));
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
        start_server_with_options(max_sessions, recovery, None)
    }

    fn start_server_with_options(
        max_sessions: usize,
        recovery: RecoveryMode,
        session_dir: Option<std::path::PathBuf>,
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
                if let Some(dir) = session_dir {
                    inner.set_session_dir(dir);
                }
                let _ = run(listener, state).await;
            });
        });
        (addr, handle)
    }

    fn temp_session_dir(label: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "kp-serve-{}-{}",
            std::process::id(),
            label
        ));
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

        // 无失败的动作轮不重复携带陈旧视图
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

        let (status, _, body) = raw_request(addr, "GET", &format!("/sessions/{}", session_id), &[], "").await;
        assert_eq!(status, 200);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["turns"], 2);
        assert_eq!(value["focus_stack"][0], "姚明");

        let (status, _, _) = raw_request(addr, "DELETE", &format!("/sessions/{}", session_id), &[], "").await;
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
            entities: Vec::new(),
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
        assert!(content.lines().count() <= 2, "简短策略不应多行：{}", content);

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

    async fn export_envelope(
        addr: std::net::SocketAddr,
        session_id: &str,
    ) -> serde_json::Value {
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
        let (status, _, response) =
            raw_request(addr, "POST", "/sessions/import", &[], &body).await;
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
        let (status, _, response) =
            raw_request(addr, "POST", "/sessions/import", &[], &body).await;
        assert_eq!(status, 409, "{}", response);
        assert!(response.contains("version"), "{}", response);

        // 3) 陌生引擎（crate 版本不同）→ 默认拒收
        let mut foreign: SessionEnvelope = serde_json::from_value(envelope.clone()).unwrap();
        foreign.session_id = "s-foreign".to_string();
        foreign.engine.crate_version = "0.0.0-foreign".to_string();
        foreign.seal();
        let body = serde_json::to_string(&foreign).unwrap();
        let (status, _, response) =
            raw_request(addr, "POST", "/sessions/import", &[], &body).await;
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

        // 原有会话不受任何导入尝试影响
        let turn = take_turn_body(addr, &session_id, "他的身高呢").await;
        assert!(turn["reply"].as_str().unwrap().contains("226"), "{}", turn);
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
            start_server_with_options(1, RecoveryMode::Observe, Some(dir.clone()));
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
        assert!(metrics["session_dir"].as_str().unwrap().contains("kp-serve"));

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
        let (status, _, _) = raw_request(
            addr,
            "DELETE",
            &format!("/sessions/{}", first),
            &[],
            "",
        )
        .await;
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
        let (status, _, _) = raw_request(
            addr,
            "DELETE",
            &format!("/sessions/{}", second),
            &[],
            "",
        )
        .await;
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
}

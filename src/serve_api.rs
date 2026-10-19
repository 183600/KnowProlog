//! 会话式 REST 服务（零新依赖的最小 HTTP/1.1 网关）。
//!
//! 把「多轮对话记忆」做成服务级能力：每个会话一个 [`DialogueManager`] +
//! 独立 [`SymbolicReasoner`]，共享启动时加载的知识事实；SSE 按句流式输出
//! 等价于符号版的 token streaming；`/health` 与 `/metrics` 暴露确定性计数。
//!
//! `system` / `developer` 消息被解析为确定性 [`SessionPolicy`]（白名单短语），
//! 生效于该会话的后续轮次：简短/详细风格、必须给依据、严格澄清。
//!
//! 端点：
//! - `GET  /`                      服务索引（端点清单）
//! - `GET  /health`                健康检查
//! - `GET  /metrics`               会话数 / 轮数 / 性能台账
//! - `POST /sessions`              新建会话 → `{"session_id": "s-1"}`
//! - `GET  /sessions/{id}`         会话快照（焦点栈 / 历史摘要 / 策略）
//! - `DELETE /sessions/{id}`       关闭会话
//! - `POST /sessions/{id}/turns`   一轮对话；`Accept: text/event-stream`
//!   或 `?stream=1` 时按句 SSE 流式返回
//! - `GET  /v1/models`             OpenAI 兼容：模型清单
//! - `GET  /v1/prompts`            提示资产注册表（spec + lint + 覆盖）
//! - `POST /v1/chat/completions`   OpenAI 兼容聊天补全（`stream: true` 时
//!   按句 SSE 流式）；无扩展字段时按无状态语义重放用户消息，带
//!   `"session_id"` 扩展时只执行最后一条用户消息并沿用服务端记忆
//!
//! 设计约束：
//! - 零新依赖：tokio 已在依赖中，HTTP/1.1 手写最小实现；
//! - 正确优先于并发：推理器内部含 `Rc`（求解器编译产物），会话不是 `Send`，
//!   因此连接在单一 accept 循环中串行处理，且同一会话的轮次天然串行化；
//!   每个连接有 30s 超时兜底，真正的并行留给后续把推理器送进专属工作线程；
//! - 请求体上限、连接超时、会话数上限全部显式配置，防御性默认值。

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Mutex;

use crate::dialogue_manager::{DialogueKnowledge, DialogueManager};
use crate::perf_ledger::PerfLedger;
use crate::prolog_engine::PrologFact;
use crate::prompt_spec::{self, PromptMode};
use crate::symbolic_reasoner::SymbolicReasoner;
use crate::ReasoningConfig;

const MAX_HEADER_BYTES: usize = 16 * 1024;
const DEFAULT_MAX_BODY_BYTES: usize = 256 * 1024;
const CONNECTION_TIMEOUT: Duration = Duration::from_secs(30);
const HISTORY_LIMIT: usize = 64;
/// OpenAI 兼容层报告的模型名（纯符号推理，非神经网络）。
pub const DEFAULT_MODEL: &str = "knowprolog-symbolic";

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
}

/// 单个会话：对话管理器 + 独立推理器 + 有界历史 + 会话级策略。
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
        let mut response = TurnResponse {
            session_id: self.id.clone(),
            turn: self.turns,
            status,
            source,
            kind,
            plan,
            reply,
            focus: self.manager.focus_entity().map(str::to_string),
            policy: self.policy.clone(),
        };
        self.policy.apply(&input, &mut response, &self.reasoner);
        response
    }
}

/// 服务共享状态：知识事实 + 会话表 + 计数。
pub struct ApiState {
    facts: Arc<Vec<PrologFact>>,
    reasoning: ReasoningConfig,
    synonyms: HashMap<String, String>,
    perf: Arc<PerfLedger>,
    sessions: Mutex<BTreeMap<String, Session>>,
    next_session_id: AtomicU64,
    turns_total: AtomicU64,
    max_sessions: usize,
    max_body_bytes: usize,
    prompt_mode: PromptMode,
}

impl ApiState {
    pub fn new(
        facts: Vec<PrologFact>,
        reasoning: ReasoningConfig,
        synonyms: HashMap<String, String>,
        perf: Arc<PerfLedger>,
        max_sessions: usize,
    ) -> Self {
        ApiState {
            facts: Arc::new(facts),
            reasoning,
            synonyms,
            perf,
            sessions: Mutex::new(BTreeMap::new()),
            next_session_id: AtomicU64::new(1),
            turns_total: AtomicU64::new(0),
            max_sessions: max_sessions.max(1),
            max_body_bytes: DEFAULT_MAX_BODY_BYTES,
            prompt_mode: PromptMode::On,
        }
    }

    /// 设置模板运行模式（在 `Arc::new` 之前调用；默认 [`PromptMode::On`]，
    /// 接地失败会自动回退组合生成器）。
    pub fn set_prompt_mode(&mut self, mode: PromptMode) {
        self.prompt_mode = mode;
    }

    pub fn prompt_mode(&self) -> PromptMode {
        self.prompt_mode
    }

    fn build_session(&self, id: String) -> Session {
        let mut reasoner = SymbolicReasoner::with_perf_ledger(
            self.reasoning.clone(),
            self.synonyms.clone(),
            self.perf.clone(),
        );
        reasoner.observe_facts(&self.facts);
        let mut manager = DialogueManager::new();
        manager.set_prompt_mode(self.prompt_mode);
        Session {
            id,
            turns: 0,
            manager,
            reasoner,
            history: Vec::new(),
            policy: SessionPolicy::default(),
        }
    }

    async fn create_session(&self) -> Result<String, &'static str> {
        let mut sessions = self.sessions.lock().await;
        if sessions.len() >= self.max_sessions {
            return Err("too many sessions");
        }
        let id = format!("s-{}", self.next_session_id.fetch_add(1, Ordering::Relaxed));
        let session = self.build_session(id.clone());
        sessions.insert(id.clone(), session);
        Ok(id)
    }

    async fn take_turn(&self, session_id: &str, input: &str) -> Option<TurnResponse> {
        self.take_turn_with_policy(session_id, input, None).await
    }

    /// 执行一轮；给出 `policy` 时先更新会话策略再执行。
    async fn take_turn_with_policy(
        &self,
        session_id: &str,
        input: &str,
        policy: Option<&SessionPolicy>,
    ) -> Option<TurnResponse> {
        let mut sessions = self.sessions.lock().await;
        let session = sessions.get_mut(session_id)?;
        if let Some(policy) = policy {
            session.policy = policy.clone();
        }
        let response = session.run_turn(input);
        self.turns_total.fetch_add(1, Ordering::Relaxed);
        Some(response)
    }

    async fn session_snapshot(&self, session_id: &str) -> Option<serde_json::Value> {
        let sessions = self.sessions.lock().await;
        let session = sessions.get(session_id)?;
        Some(serde_json::json!({
            "session_id": session.id,
            "turns": session.turns,
            "focus": session.manager.focus_entity(),
            "focus_stack": session.manager.focus_stack(),
            "history": session.history,
            "summary": session.manager.summary(),
            "policy": session.policy,
        }))
    }

    async fn close_session(&self, session_id: &str) -> bool {
        self.sessions.lock().await.remove(session_id).is_some()
    }

    async fn session_count(&self) -> usize {
        self.sessions.lock().await.len()
    }
}

/// 启动服务（阻塞直到监听套接字出错）。
pub async fn serve(state: Arc<ApiState>, addr: &str) -> std::io::Result<()> {
    let listener = TcpListener::bind(addr).await?;
    run(listener, state).await
}

/// 在已绑定的监听器上运行（测试可直接绑定 `127.0.0.1:0`）。
///
/// 会话（及其推理器）不是 `Send`，连接按到达顺序串行处理；
/// 每个连接有 [`CONNECTION_TIMEOUT`] 兜底，避免慢客户端永久占用。
pub async fn run(listener: TcpListener, state: Arc<ApiState>) -> std::io::Result<()> {
    loop {
        let (stream, _peer) = listener.accept().await?;
        let _ = tokio::time::timeout(
            CONNECTION_TIMEOUT,
            serve_connection(stream, Arc::clone(&state)),
        )
        .await;
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
                "perf": state.perf.snapshot(),
            }),
        ),
        ("POST", ["sessions"]) => match state.create_session().await {
            Ok(session_id) => Response::json(201, &serde_json::json!({"session_id": session_id})),
            Err(message) => Response::error(429, message),
        },
        ("GET", ["sessions", id]) => match state.session_snapshot(id).await {
            Some(snapshot) => Response::json(200, &snapshot),
            None => Response::error(404, "session not found"),
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
                Some(turn) => Response::json(
                    200,
                    &serde_json::to_value(&turn).unwrap_or_else(|_| serde_json::json!({})),
                ),
                None => Response::error(404, "session not found"),
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

/// OpenAI 兼容的聊天请求（`session_id` 为本服务的扩展字段）。
struct ChatRequest {
    model: String,
    /// 提供时沿用该服务端会话，只执行最后一条用户消息。
    session_id: Option<String>,
    /// 无状态模式下的用户消息序列（按顺序重放到一个临时会话）。
    inputs: Vec<String>,
    /// 由 `system` / `developer` 消息解析出的会话级策略（无则沿用默认）。
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
            Some(turn) => Ok((turn, prompt_chars)),
            None => Err((404, "session not found".to_string())),
        };
    }
    let temp_id = state
        .create_session()
        .await
        .map_err(|message| (429, message.to_string()))?;
    let mut last = None;
    for input in &chat.inputs {
        last = state
            .take_turn_with_policy(&temp_id, input, chat.policy.as_ref())
            .await;
    }
    state.close_session(&temp_id).await;
    last.map(|turn| (turn, prompt_chars))
        .ok_or_else(|| (500, "turn failed".to_string()))
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
        Some(turn) => turn,
        None => {
            write_response(stream, Response::error(404, "session not found")).await?;
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

    #[test]
    fn turn_response_is_send() {
        assert_send::<TurnResponse>();
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
        ))
    }

    /// 服务状态不是 `Send`（推理器含 `Rc`），因此把服务跑在专属 OS 线程的
    /// current-thread 运行时里；测试只通过 TCP 交互，不跨线程共享状态。
    fn start_server(max_sessions: usize) -> (std::net::SocketAddr, std::thread::JoinHandle<()>) {
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
                let state = test_state(max_sessions);
                let _ = run(listener, state).await;
            });
        });
        (addr, handle)
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
}

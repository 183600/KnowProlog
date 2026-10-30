async fn handle_knowledge_changes(state: &Arc<ApiState>, request: &Request) -> Response {
    let since = request
        .query
        .get("since")
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(0);
    let generation = state.knowledge.generation();
    match state.knowledge.journal_since(since) {
        Some(changes) => Response::json(
            200,
            &serde_json::json!({
                "from": since,
                "generation": generation,
                "requires_rebuild": changes.iter().any(|change| {
                    matches!(change, KnowledgeChange::Reset { .. })
                }),
                "changes": changes.iter().map(knowledge_change_to_json).collect::<Vec<_>>(),
            }),
        ),
        None => Response::json(
            409,
            &serde_json::json!({
                "error": "knowledge journal no longer covers this generation",
                "code": "journal_gap",
                "from": since,
                "generation": generation,
            }),
        ),
    }
}
async fn handle_knowledge_replace(state: &Arc<ApiState>, request: &Request) -> Response {
    let body = match request.body_text() {
        Ok(body) => body,
        Err(error) => return utf8_body_error(error),
    };
    let payload: serde_json::Value = match serde_json::from_str(&body) {
        Ok(value) => value,
        Err(error) => return Response::error(400, &format!("invalid JSON: {error}")),
    };
    let facts = match parse_knowledge_facts(&payload) {
        Ok(facts) => facts,
        Err(message) => return Response::error(400, &message),
    };
    let source = payload
        .get("source")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("http");
    let changed = state.replace_facts(facts, source);
    let snapshot = state.knowledge.snapshot();
    Response::json(
        200,
        &serde_json::json!({
            "status": if changed { "replaced" } else { "unchanged" },
            "replaced": changed,
            "source": source,
            "scope": "process",
            "generation": snapshot.generation,
            "facts": snapshot.facts.len(),
            "fingerprint": snapshot.fingerprint,
        }),
    )
}
async fn handle_knowledge_aliases(state: &Arc<ApiState>, request: &Request) -> Response {
    let body = match request.body_text() {
        Ok(body) => body,
        Err(error) => return utf8_body_error(error),
    };
    let payload: serde_json::Value = match serde_json::from_str(&body) {
        Ok(value) => value,
        Err(error) => return Response::error(400, &format!("invalid JSON: {error}")),
    };
    let aliases = match parse_knowledge_aliases(&payload) {
        Ok(aliases) => aliases,
        Err(message) => return Response::error(400, &message),
    };
    let applied = state.align_entity_aliases(&aliases);
    let snapshot = state.knowledge.snapshot();
    let status = if applied > 0 { 201 } else { 200 };
    Response::json(
        status,
        &serde_json::json!({
            "status": if applied > 0 { "admitted" } else { "unchanged" },
            "applied": applied,
            "generation": snapshot.generation,
            "aliases": snapshot.aliases.len(),
        }),
    )
}
/// 无状态事件抽取/问答：`POST /v1/events`。
///
/// 请求：`{"text":"...","question":"..."}`；`question` 可省略，省略时只返回事件图。
/// 事件图按请求现算，不写入会话或默认 RDF 图；N-Quads 需调用 CLI `events --nquads`。
async fn handle_events(request: &Request) -> Response {
    let body = match request.body_text() {
        Ok(body) => body,
        Err(error) => return utf8_body_error(error),
    };
    let payload: serde_json::Value = match serde_json::from_str(&body) {
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
/// 无状态共指闭包：`POST /v1/coref`。
///
/// 请求体与 CLI `coref` 的证据 JSON 相同，可额外带 `resolve` / `closure` /
/// `predicate` / `operator` / `law` 字段；只读，不写会话或知识库。
async fn handle_coref(request: &Request) -> Response {
    let body = match request.body_text() {
        Ok(body) => body,
        Err(error) => return utf8_body_error(error),
    };
    let payload: serde_json::Value = match serde_json::from_str(&body) {
        Ok(value) => value,
        Err(error) => return Response::error(400, &format!("invalid JSON: {error}")),
    };
    let evidence: crate::coref_closure::CorefEvidence =
        match serde_json::from_value(payload.clone()) {
            Ok(evidence) => evidence,
            Err(error) => {
                return Response::error(400, &format!("invalid coref evidence: {error}"));
            }
        };
    let resolve = payload.get("resolve").and_then(serde_json::Value::as_str);
    let closure = payload.get("closure").and_then(serde_json::Value::as_str);
    let predicate = payload.get("predicate").and_then(serde_json::Value::as_str);
    let law = payload.get("law").and_then(serde_json::Value::as_str);
    let operator = match payload.get("operator").and_then(serde_json::Value::as_str) {
        Some(raw) => match crate::pattern_closure::ClosureOp::parse(raw) {
            Some(operator) => operator,
            None => {
                return Response::error(
                    400,
                    &format!("invalid operator: {raw} (use plus, star or exact:N)"),
                );
            }
        },
        None => crate::pattern_closure::ClosureOp::Plus,
    };

    let graph = evidence.into_graph();
    let budget = crate::coref_closure::CorefBudget::default();
    let build = graph.build(&budget);
    let mut response = serde_json::json!({
        "status": "ok",
        "identity": build,
    });
    if let Some(surface) = resolve {
        response["resolution"] =
            serde_json::to_value(build.resolve(surface)).unwrap_or(serde_json::Value::Null);
    }
    if let Some(start) = closure {
        let quotient = graph.quotient(&budget);
        match quotient.closure(
            start,
            operator,
            crate::pattern_closure::ClosureBudget::default(),
            law,
            predicate,
        ) {
            Ok(outcome) => {
                response["closure"] = coref_closure_json(&outcome);
            }
            Err(error) => return Response::error(400, &error.to_string()),
        }
    }
    response["quotient"] =
        serde_json::to_value(graph.quotient(&budget)).unwrap_or(serde_json::Value::Null);
    Response::json(200, &response)
}

fn coref_closure_json(outcome: &crate::coref_closure::QuotientOutcome) -> serde_json::Value {
    let targets: Vec<serde_json::Value> = outcome
        .targets()
        .into_iter()
        .map(|target| {
            serde_json::json!({
                "target": target,
                "hops": outcome.outcome.hop_count_of(target),
                "witness": outcome.render_witness(target),
            })
        })
        .collect();
    serde_json::json!({
        "start_surface": outcome.start_surface,
        "start": outcome.start,
        "operator": outcome.outcome.operator.label(),
        "reached_count": outcome.outcome.reached_count,
        "complete": outcome.outcome.complete(),
        "truncated": outcome.outcome.truncated.map(|truncated| truncated.label()),
        "targets": targets,
    })
}

/// 把共享白板声明转成低可信度证据索引；冲突中的说法不替用户拍板，进一步降权。
fn shared_attention_items(view: &SharedMemoryView) -> Vec<AttentionItem> {
    let conflicted: std::collections::BTreeSet<&str> = view
        .conflicts
        .iter()
        .flat_map(|certificate| certificate.claim_ids.iter().map(String::as_str))
        .collect();
    // 只索引最近发布的有界窗口：快照生成绝不随白板容量线性膨胀。
    let mut recent: Vec<&crate::shared_memory::SharedClaim> = view.claims.iter().collect();
    recent.sort_by_key(|claim| std::cmp::Reverse(claim.revision));
    recent.truncate(WORKING_MEMORY_CAPACITY);
    recent
        .into_iter()
        .map(|claim| {
            let mut text = format!(
                "{}的{}是{}",
                claim.subject, claim.predicate, claim.value
            );
            if !claim.source.is_empty() {
                text.push_str(&format!("（来源：{}）", claim.source));
            }
            AttentionItem::new(
                format!("shared:{}", claim.id),
                AttentionKind::Evidence,
                text,
            )
            .with_entities([claim.subject.clone()])
            .with_predicates([claim.predicate.clone()])
            .recency(claim.revision)
            .trust(if conflicted.contains(claim.id.as_str()) {
                40
            } else {
                65
            })
        })
        .collect()
}
/// 读取共享语义白板：无会话参数只读全局；指定会话时叠加该会话私有声明。
async fn handle_shared_memory_view(state: &Arc<ApiState>, session_id: Option<&str>) -> Response {
    let view = match session_id {
        Some(id) => {
            if let Err(error) = state.session_tx(id).await {
                return session_error_response(error);
            }
            state.shared_memory.snapshot_for(id)
        }
        None => state.shared_memory.snapshot(),
    };
    Response::json(
        200,
        &serde_json::to_value(view.as_ref()).unwrap_or_else(|_| serde_json::json!({})),
    )
}
/// 发布一条共享声明。作用域由路由显式决定，绝不信任请求体里的 `session_id`：
/// `/v1/memory` 恒为全局，`/sessions/{id}/memory` 恒为该会话私有。
async fn handle_shared_memory_publish(
    state: &Arc<ApiState>,
    request: &Request,
    session_id: Option<&str>,
) -> Response {
    if let Some(id) = session_id {
        if let Err(error) = state.session_tx(id).await {
            return session_error_response(error);
        }
    }
    let body = match request.body_text() {
        Ok(body) => body,
        Err(error) => return utf8_body_error(error),
    };
    let payload: serde_json::Value = match serde_json::from_str(&body) {
        Ok(value) => value,
        Err(error) => return Response::error(400, &format!("invalid JSON: {error}")),
    };
    let Some(subject) = payload.get("subject").and_then(serde_json::Value::as_str) else {
        return Response::error(400, "missing string field: subject");
    };
    let Some(predicate) = payload
        .get("predicate")
        .and_then(serde_json::Value::as_str)
    else {
        return Response::error(400, "missing string field: predicate");
    };
    let Some(value) = payload.get("value").and_then(serde_json::Value::as_str) else {
        return Response::error(400, "missing string field: value");
    };
    let source = payload
        .get("source")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("http");
    let scope = session_id
        .map(MemoryScope::session)
        .unwrap_or(MemoryScope::Global);
    let claim = NewClaim::new(subject, predicate, value, scope).with_source(source);
    match state.shared_memory.publish(claim) {
        Ok(outcome) => {
            let status_code = match outcome.status() {
                "published" | "conflict" => 201,
                "saturated" => 409,
                _ => 200,
            };
            let mut body = serde_json::to_value(&outcome)
                .ok()
                .and_then(|value| value.as_object().cloned())
                .unwrap_or_default();
            body.insert(
                "status".to_string(),
                serde_json::Value::String(outcome.status().to_string()),
            );
            body.insert(
                "memory_revision".to_string(),
                serde_json::json!(outcome.revision()),
            );
            Response::json(status_code, &serde_json::Value::Object(body))
        }
        Err(error) => shared_memory_error_response(&error),
    }
}
/// 撤回声明：作用域由路由决定，跨作用域一律 404（不泄漏其它会话是否存在该声明）。
async fn handle_shared_memory_retract(
    state: &Arc<ApiState>,
    claim_id: &str,
    session_id: Option<&str>,
) -> Response {
    if let Some(id) = session_id {
        if let Err(error) = state.session_tx(id).await {
            return session_error_response(error);
        }
    }
    let scope = session_id
        .map(MemoryScope::session)
        .unwrap_or(MemoryScope::Global);
    match state.shared_memory.retract_scoped(claim_id, &scope) {
        Some(claim) => Response::json(
            200,
            &serde_json::json!({
                "status": "retracted",
                "claim": claim,
                "memory_revision": state.shared_memory.revision(),
            }),
        ),
        None => Response::json(
            404,
            &serde_json::json!({
                "error": "claim not found in this scope",
                "code": "not_found",
            }),
        ),
    }
}
fn shared_memory_error_response(error: &PublishError) -> Response {
    Response::json(
        400,
        &serde_json::json!({
            "error": error.to_string(),
            "code": error.code(),
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
    let body = request.body_text().map_err(|error| {
        format!(
            "请求体不是合法 UTF-8（{}，偏移 {}）",
            error.code(),
            error.offset
        )
    })?;
    let value: serde_json::Value =
        serde_json::from_str(&body).map_err(|error| format!("invalid JSON: {}", error))?;
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
    let body = request.body_text().map_err(|error| {
        format!(
            "请求体不是合法 UTF-8（{}，偏移 {}）",
            error.code(),
            error.offset
        )
    })?;
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
    include!("serve_api__tests_01.rs");
    include!("serve_api__tests_02.rs");
}

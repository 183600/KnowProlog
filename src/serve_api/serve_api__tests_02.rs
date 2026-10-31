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

    /// 服务暖基座也要恢复交互导入落盘的措辞/问法，并让旧缓存失效。
    #[test]
    fn load_import_frames_reaches_warm_reasoner_and_bumps_epoch() {
        use crate::import_frames::{
            FactKind, FrameAtom, FrameObservation, ImportFrame, ImportFrameLedger,
        };

        let state = test_state(2);
        let epoch = state.cache_epoch();
        let mut ledger = ImportFrameLedger::new();
        let frame = ImportFrame {
            id: "ifr#年龄".to_string(),
            kind: FactKind::Attribute,
            predicate: "年龄".to_string(),
            atoms: vec![
                FrameAtom::Subject,
                FrameAtom::Literal {
                    surface: "今年".to_string(),
                },
                FrameAtom::Object,
                FrameAtom::Literal {
                    surface: "岁".to_string(),
                },
            ],
            example_subject: "张三".to_string(),
            example_object: "30".to_string(),
            observations: vec![FrameObservation {
                witness: "张三今年30岁".to_string(),
                import_digest: "digest".to_string(),
                turn: 1,
            }],
            ask_activated: true,
        };
        ledger.frames.insert(frame.id.clone(), frame);

        assert_eq!(state.load_import_frames(ledger.clone()), 1);
        assert_ne!(state.cache_epoch(), epoch);
        assert_eq!(
            state
                .warm
                .reasoner
                .lock()
                .import_frames()
                .expect("暖基座应持有导入帧")
                .len(),
            1
        );

        let after = state.cache_epoch();
        assert_eq!(state.load_import_frames(ImportFrameLedger::new()), 0);
        assert_eq!(state.cache_epoch(), after, "空台账不应推进纪元");
    }

    /// 与 CLI 共用存储时，服务也必须载入已学习的重写规则，否则
    /// 「张三到底师傅是谁」这类说法只在交互/ask 入口可用。
    #[test]
    fn load_rewrite_book_reaches_warm_reasoner_and_bumps_epoch() {
        use crate::rewrite_book::{RewriteBook, RewriteRule, RewriteStatus};

        let state = test_state(2);
        let epoch = state.cache_epoch();
        let mut book = RewriteBook::new();
        book.rules.push(RewriteRule {
            id: "learned-shell".to_string(),
            plan_kind: "FindObjectByPredicate".to_string(),
            surface: "{S}到底{P}是谁".to_string(),
            canonical: "{S}的{P}是谁".to_string(),
            status: RewriteStatus::Active,
            support: 3,
            predicates: vec!["师傅".to_string()],
            entities: Vec::new(),
            examples: Vec::new(),
            refutations: 0,
            source: "test".to_string(),
        });

        assert_eq!(state.load_rewrite_book(book), 1);
        assert_ne!(state.cache_epoch(), epoch);
        assert_eq!(
            state
                .warm
                .reasoner
                .lock()
                .rewrite_book()
                .expect("暖基座应持有重写手册")
                .len(),
            1
        );

        let after = state.cache_epoch();
        assert_eq!(state.load_rewrite_book(RewriteBook::new()), 0);
        assert_eq!(state.cache_epoch(), after, "空手册不应推进纪元");
    }

    /// 与 CLI 共用存储时，服务也要恢复词法记忆（词性证据/量词期待），
    /// 否则同一问法在 HTTP 入口得到的解析软约束不同。
    #[test]
    fn apply_lexical_memory_reaches_warm_reasoner_and_bumps_epoch() {
        use crate::lexical_memory::{LexicalMemory, TokenEvidence};

        let state = test_state(2);
        let epoch = state.cache_epoch();
        let mut memory = LexicalMemory::new(8);
        memory.tokens.insert(
            "t1".to_string(),
            TokenEvidence {
                id: "t1".to_string(),
                surface: "姚明".to_string(),
                pos: "nr".to_string(),
                source: "test".to_string(),
                count: 1,
                first_turn: 1,
                last_turn: 1,
                type_votes: std::collections::BTreeMap::from([("人物".to_string(), 1)]),
            },
        );

        assert_eq!(state.apply_lexical_memory(memory), 1);
        assert_ne!(state.cache_epoch(), epoch);
        assert_eq!(state.warm.reasoner.lock().lexical_memory().len(), 1);

        let after = state.cache_epoch();
        assert_eq!(state.apply_lexical_memory(LexicalMemory::new(8)), 0);
        assert_eq!(state.cache_epoch(), after, "空记忆不应推进纪元");
    }

    /// `[frame_semantics]` 在交互入口会注入框架本，服务启动同样要带，
    /// 否则同一配置下 HTTP 生成缺少角色标注。
    #[test]
    fn set_frame_book_reaches_warm_reasoner() {
        use crate::frame_semantics::FrameBook;

        let state = test_state(2);
        let mut state = Arc::try_unwrap(state).ok().expect("测试状态应独占");
        assert!(state.warm.reasoner.lock().frame_book().is_none());
        state.set_frame_book(FrameBook::with_core_schemas());
        assert!(state.warm.reasoner.lock().frame_book().is_some());
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

        let budgeted = SessionPolicy::from_system_texts(&["回答控制在100字以内".to_string()]);
        assert_eq!(budgeted.max_answer_chars, Some(100));
        assert!(budgeted.recognized.contains(&"budget:100".to_string()));
        assert!(
            budgeted.render_text().contains("回答预算 100 字"),
            "{}",
            budgeted.render_text()
        );
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
            Arc::clone(&state.knowledge),
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

    /// 严格澄清会话中的短答必须由澄清通道结算：确认/否定/暂缓不能再被
    /// 普通问答或悬问回填消费，也不能给用户抛回一个新的未回答问题。
    #[test]
    fn strict_clarification_short_answer_settles_through_clarification_channel() {
        let mut session = test_session(TurnBudget::unlimited());
        session.reasoner.observe_facts(&[
            PrologFact::Relation {
                from: "张三".to_string(),
                rel_type: "师傅".to_string(),
                to: "李四".to_string(),
            },
            PrologFact::Relation {
                from: "王五".to_string(),
                rel_type: "师傅".to_string(),
                to: "赵六".to_string(),
            },
        ]);
        session.policy.strict_clarify = true;
        session
            .reasoner
            .hypothesis_answer("赵六的师傅是谁")
            .expect("缺失关系应能生成候选假设");
        session
            .reasoner
            .start_clarification()
            .expect("有待定假设应能开始澄清");

        let response = session.run_turn("是");
        let expected_status = if session.reasoner.clarification_active() {
            "needs_clarification"
        } else {
            "answered"
        };
        assert_eq!(response.status, expected_status, "{}", response.reply);
        assert_eq!(response.source, "clarification", "{}", response.reply);
        assert_eq!(response.plan, vec!["clarification".to_string()]);
        assert!(
            response.reply.contains("澄清确认") || response.reply.contains("已确认假设"),
            "短答应结算为确认而不是新问题：{}",
            response.reply
        );
        assert!(
            !response.reply.contains("请补充实体名称"),
            "澄清短答不应落入严格澄清的让位话术：{}",
            response.reply
        );
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
            Arc::clone(&state.knowledge),
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
            Arc::clone(&state.knowledge),
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
        assert!(
            confirmed.reply.contains("已确认预测"),
            "{}",
            confirmed.reply
        );
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

    #[tokio::test]
    async fn coref_endpoint_resolves_aliases_and_closes_relations() {
        let request = Request {
            method: "POST".to_string(),
            path: "/v1/coref".to_string(),
            query: BTreeMap::new(),
            headers: BTreeMap::new(),
            body: serde_json::json!({
                "identity": [
                    {"from": "老李", "to": "李四", "kind": "alias"},
                    {"from": "他", "to": "李四", "kind": "pronoun", "confidence": 0.8}
                ],
                "non_identity": [
                    {"from": "姚明的妻子", "to": "叶莉", "kind": "description"}
                ],
                "relations": [
                    {"from": "李四", "to": "王五", "predicate": "师傅"},
                    {"from": "王五", "to": "赵六", "predicate": "师傅"}
                ],
                "resolve": "老李",
                "closure": "老李",
                "predicate": "师傅",
                "law": "师傅"
            })
            .to_string()
            .into_bytes(),
        };
        let response = handle_coref(&request).await;
        assert_eq!(
            response.status,
            200,
            "{}",
            String::from_utf8_lossy(&response.body)
        );
        let value: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
        assert_eq!(value["status"], "ok");
        assert_eq!(value["resolution"]["canonical"], "李四");
        assert_eq!(value["closure"]["reached_count"], 2);
        assert_eq!(value["closure"]["targets"][0]["target"], "王五");
        assert_eq!(value["closure"]["targets"][1]["target"], "赵六");
        assert_eq!(value["quotient"]["self_loops"].as_array().map(Vec::len), Some(0));
    }

    #[tokio::test]
    async fn coref_route_is_reachable_over_http() {
        let (addr, _server) = start_server(2);
        let body = serde_json::json!({
            "identity": [{"from": "老李", "to": "李四"}]
        })
        .to_string();
        let (status, _, body) = raw_request(
            addr,
            "POST",
            "/v1/coref",
            &[("Content-Type", "application/json")],
            &body,
        )
        .await;
        assert_eq!(status, 200, "{}", body);
        let value: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert_eq!(value["identity"]["canonical_of"]["老李"], "李四");
        assert_eq!(value["identity"]["active_edges"], 1);
    }

    #[tokio::test]
    async fn coref_endpoint_rejects_unknown_operator_without_panicking() {
        let request = Request {
            method: "POST".to_string(),
            path: "/v1/coref".to_string(),
            query: BTreeMap::new(),
            headers: BTreeMap::new(),
            body: r#"{
                "identity": [{"from": "老李", "to": "李四"}],
                "closure": "老李",
                "operator": "sometimes"
            }"#
            .as_bytes()
            .to_vec(),
        };
        let response = handle_coref(&request).await;
        assert_eq!(response.status, 400);
        let value: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
        assert!(value["error"].as_str().unwrap().contains("invalid operator"));
    }

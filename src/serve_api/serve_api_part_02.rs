impl ApiState {
    pub fn new(
        facts: Vec<PrologFact>,
        reasoning: ReasoningConfig,
        synonyms: HashMap<String, String>,
        perf: Arc<PerfLedger>,
        max_sessions: usize,
        startup: Arc<StartupLedger>,
    ) -> Self {
        let knowledge = Arc::new(KnowledgeBase::new(facts.clone()));
        // 预热基座：整进程唯一的全量重放点（会话创建只做字段级分叉）
        let warm = Arc::new(WarmBase::build(
            &facts, &reasoning, &synonyms, &perf, &startup,
        ));
        ApiState {
            knowledge,
            warm,
            stateless_answers: Arc::new(parking_lot::Mutex::new(RevisionLru::new(
                STATELESS_CACHE_CAPACITY,
            ))),
            kb_revision: AtomicU64::new(0),
            knowledge_write: parking_lot::Mutex::new(()),
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
            shared_memory: Arc::new(SharedMemoryBoard::new(DEFAULT_MAX_CLAIMS)),
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

    /// 注入配置态 RDFS 模式视图（含类型别名与谓词画像），
    /// 与 `configure_ontology` 一样在暖基座上单调合并，分叉会话同源携带。
    pub fn configure_schema(&self, schema: &SchemaView) -> bool {
        self.warm.reasoner.lock().configure_schema_view(schema)
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
        let _write = self.knowledge_write.lock();
        self.warm.reasoner.lock().align_entity_aliases(aliases);
        let admitted = self.knowledge.record_aliases(aliases, "host");
        if !admitted.is_empty() {
            self.bump_knowledge_epoch();
            self.bump_identity_epoch();
        }
        admitted.len()
    }

    pub fn assert_facts(&self, facts: &[PrologFact], source: &str) -> usize {
        let _write = self.knowledge_write.lock();
        let missing = self.knowledge.missing_facts(facts);
        if missing.is_empty() {
            return 0;
        }
        self.warm.observe_facts(&missing);
        let admitted = self.knowledge.assert_facts(&missing, source);
        if admitted.is_empty() {
            return 0;
        }
        self.bump_knowledge_epoch();
        admitted.len()
    }

    pub fn replace_facts(&self, facts: Vec<PrologFact>, source: &str) -> bool {
        let _write = self.knowledge_write.lock();
        if self.knowledge.snapshot().facts.as_slice() == facts.as_slice() {
            return false;
        }
        self.warm.rebuild_from_graph(&facts);
        let changed = self.knowledge.replace_facts(facts, source);
        if changed {
            self.bump_knowledge_epoch();
        }
        changed
    }

    fn bump_knowledge_epoch(&self) {
        self.kb_revision.fetch_add(1, Ordering::Relaxed);
        self.perf.bump("knowledge_epoch_bumps");
    }

    pub fn knowledge_snapshot(&self) -> crate::knowledge_delta::KnowledgeSnapshot {
        self.knowledge.snapshot()
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
            kb_revision: self.kb_revision.load(Ordering::Relaxed),
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
        let snapshot = self.knowledge.snapshot();
        EngineStamp {
            crate_version: env!("CARGO_PKG_VERSION").to_string(),
            kb_fingerprint: snapshot.fingerprint,
            kb_facts: snapshot.facts.len(),
            kb_generation: snapshot.generation,
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
        let shared_memory = Arc::clone(&self.shared_memory);
        let knowledge = Arc::clone(&self.knowledge);
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
                        Arc::clone(&knowledge),
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
                            session.sync_knowledge();
                            let shared = shared_memory.snapshot_for(&session.id);
                            let evidence = shared_attention_items(shared.as_ref());
                            let _ = reply.send(serde_json::json!({
                                "session_id": session.id,
                                "turns": session.turns,
                                "focus": session.manager.focus_entity(),
                                "focus_stack": session.manager.focus_stack(),
                                "history": session.history,
                                "summary": session.manager.summary(),
                                "policy": session.policy,
                                "board": session.manager.board().view(session.reasoner.revision()),
                                "working_memory": session.manager.context_packet_with(
                                    AttentionBudget::default(),
                                    evidence,
                                ),
                                "shared_memory": {
                                    "revision": shared.revision,
                                    "claims": shared.claims.len(),
                                    "conflicts": shared.conflicts.len(),
                                    "digest": shared.digest,
                                },
                                "open_turn": session.manager.latest_open_turn(),
                                "resumable": session.open_turn_resumable(),
                                "poisoned": session.poisoned,
                            }));
                        }
                        SessionCommand::Export { reply } => {
                            session.sync_knowledge();
                            let mut envelope = SessionEnvelope {
                                format: ENVELOPE_FORMAT.to_string(),
                                version: ENVELOPE_VERSION,
                                session_id: session.id.clone(),
                                turns: session.turns,
                                history: session.history.clone(),
                                policy: session.policy.clone(),
                                manager: session.manager.snapshot(),
                                reasoner: session.reasoner.dialogue_snapshot(),
                                engine: session.engine_stamp(),
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
                            session.sync_knowledge();
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
        let foreign = envelope
            .engine
            .mismatched_fields_for_resume(&self.engine_stamp());
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
        // 私有声明随会话生命周期结束；否则同 id 重新导入/创建会继承旧记忆。
        self.shared_memory.purge_session(session_id);
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

    fn body_text(&self) -> Result<String, crate::text_identity::Utf8DecodeError> {
        crate::text_identity::decode_utf8_strict(&self.body).map(str::to_string)
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
fn utf8_body_error(error: crate::text_identity::Utf8DecodeError) -> Response {
    Response::json(
        400,
        &serde_json::json!({
            "error": format!("请求体不是合法 UTF-8：{}", error),
            "code": error.code(),
            "offset": error.offset,
            "len": error.len,
        }),
    )
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
            let body = match request.body_text() {
                Ok(body) => body,
                Err(error) => {
                    write_response(&mut stream, utf8_body_error(error)).await?;
                    return Ok(());
                }
            };
            return stream_turn(&mut stream, &state, &session_id, &body).await;
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
                    "GET /sessions/{id}/memory",
                    "POST /sessions/{id}/memory",
                    "DELETE /sessions/{id}/memory/{claim}",
                    "GET /v1/memory",
                    "POST /v1/memory",
                    "DELETE /v1/memory/{claim}",
                    "GET /v1/knowledge",
                    "GET /v1/knowledge/changes",
                    "POST /v1/knowledge/facts",
                    "PUT /v1/knowledge/facts",
                    "POST /v1/knowledge/aliases",
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
        ("GET", ["metrics"]) => {
            let memory = state.shared_memory.snapshot_all();
            let knowledge = state.knowledge.snapshot();
            Response::json(
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
                    "knowledge": {
                        "generation": knowledge.generation,
                        "facts": knowledge.facts.len(),
                        "aliases": knowledge.aliases.len(),
                        "fingerprint": knowledge.fingerprint,
                        "journal": knowledge.journal_len,
                        "cache_epoch": state.cache_epoch().kb_revision,
                    },
                    "shared_memory": {
                        "revision": memory.revision,
                        "claims": memory.claims.len(),
                        "conflicts": memory.conflicts.len(),
                        "evictions": memory.evictions,
                        "digest": memory.digest,
                    },
                    "perf": state.perf.snapshot(),
                    "startup": state.startup.snapshot(),
                }),
            )
        }
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
        ("GET", ["sessions", id, "memory"]) => {
            handle_shared_memory_view(state, Some(id)).await
        }
        ("POST", ["sessions", id, "memory"]) => {
            handle_shared_memory_publish(state, request, Some(id)).await
        }
        ("GET", ["v1", "memory"]) => {
            let session = request.query.get("session").map(String::as_str);
            handle_shared_memory_view(state, session).await
        }
        ("POST", ["v1", "memory"]) => handle_shared_memory_publish(state, request, None).await,
        ("GET", ["v1", "knowledge"]) => handle_knowledge_view(state, request).await,
        ("GET", ["v1", "knowledge", "changes"]) => {
            handle_knowledge_changes(state, request).await
        }
        ("POST", ["v1", "knowledge", "facts"]) => handle_knowledge_assert(state, request).await,
        ("PUT", ["v1", "knowledge", "facts"]) => handle_knowledge_replace(state, request).await,
        ("POST", ["v1", "knowledge", "aliases"]) => {
            handle_knowledge_aliases(state, request).await
        }
        ("DELETE", ["sessions", id, "memory", claim_id]) => {
            handle_shared_memory_retract(state, claim_id, Some(id)).await
        }
        ("DELETE", ["v1", "memory", claim_id]) => {
            handle_shared_memory_retract(state, claim_id, None).await
        }
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
const MAX_KNOWLEDGE_FACTS_PER_REQUEST: usize = 256;
const MAX_KNOWLEDGE_ALIASES_PER_REQUEST: usize = 256;
#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum KnowledgeFactInput {
    Entity {
        name: String,
        #[serde(default)]
        entity_type: String,
        #[serde(default)]
        properties: HashMap<String, String>,
    },
    Relation {
        from: String,
        rel_type: String,
        to: String,
    },
}
impl KnowledgeFactInput {
    fn into_fact(self) -> Result<PrologFact, String> {
        match self {
            KnowledgeFactInput::Entity {
                name,
                entity_type,
                properties,
            } => {
                let name = name.trim().to_string();
                if name.is_empty() {
                    return Err("entity name must not be empty".to_string());
                }
                let entity_type = entity_type.trim();
                let entity_type = if entity_type.is_empty() {
                    "Entity".to_string()
                } else {
                    entity_type.to_string()
                };
                if properties.len() > 64 {
                    return Err("entity properties must be at most 64 entries".to_string());
                }
                let mut normalized = HashMap::with_capacity(properties.len());
                for (key, value) in properties {
                    let key = key.trim();
                    if key.is_empty() {
                        return Err("entity property key must not be empty".to_string());
                    }
                    normalized.insert(key.to_string(), value);
                }
                Ok(PrologFact::Entity {
                    name,
                    entity_type,
                    properties: normalized,
                })
            }
            KnowledgeFactInput::Relation { from, rel_type, to } => {
                let from = from.trim().to_string();
                let rel_type = rel_type.trim().to_string();
                let to = to.trim().to_string();
                if from.is_empty() || rel_type.is_empty() || to.is_empty() {
                    return Err("relation from/rel_type/to must not be empty".to_string());
                }
                Ok(PrologFact::Relation { from, rel_type, to })
            }
        }
    }
}
#[derive(Debug, Deserialize)]
struct KnowledgeAliasInput {
    surface: String,
    canonical: String,
}
fn parse_knowledge_facts(payload: &serde_json::Value) -> Result<Vec<PrologFact>, String> {
    let items = payload
        .get("facts")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "missing array field: facts".to_string())?;
    if items.is_empty() {
        return Err("facts must not be empty".to_string());
    }
    if items.len() > MAX_KNOWLEDGE_FACTS_PER_REQUEST {
        return Err(format!(
            "facts must contain at most {} entries",
            MAX_KNOWLEDGE_FACTS_PER_REQUEST
        ));
    }
    items
        .iter()
        .map(|item| {
            serde_json::from_value::<KnowledgeFactInput>(item.clone())
                .map_err(|error| format!("invalid fact: {error}"))
                .and_then(KnowledgeFactInput::into_fact)
        })
        .collect()
}
fn parse_knowledge_aliases(payload: &serde_json::Value) -> Result<Vec<(String, String)>, String> {
    let items = payload
        .get("aliases")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| "missing array field: aliases".to_string())?;
    if items.is_empty() {
        return Err("aliases must not be empty".to_string());
    }
    if items.len() > MAX_KNOWLEDGE_ALIASES_PER_REQUEST {
        return Err(format!(
            "aliases must contain at most {} entries",
            MAX_KNOWLEDGE_ALIASES_PER_REQUEST
        ));
    }
    items
        .iter()
        .map(|item| {
            let alias = serde_json::from_value::<KnowledgeAliasInput>(item.clone())
                .map_err(|error| format!("invalid alias: {error}"))?;
            let surface = alias.surface.trim().to_string();
            let canonical = alias.canonical.trim().to_string();
            if surface.is_empty() || canonical.is_empty() {
                return Err("alias surface/canonical must not be empty".to_string());
            }
            if surface == canonical {
                return Err("alias surface must differ from canonical".to_string());
            }
            Ok((surface, canonical))
        })
        .collect()
}
fn fact_to_json(fact: &PrologFact) -> serde_json::Value {
    match fact {
        PrologFact::Entity {
            name,
            entity_type,
            properties,
        } => {
            let properties: BTreeMap<&String, &String> = properties.iter().collect();
            serde_json::json!({
                "kind": "entity",
                "name": name,
                "entity_type": entity_type,
                "properties": properties,
            })
        }
        PrologFact::Relation { from, rel_type, to } => serde_json::json!({
            "kind": "relation",
            "from": from,
            "rel_type": rel_type,
            "to": to,
        }),
    }
}
async fn handle_knowledge_view(state: &Arc<ApiState>, request: &Request) -> Response {
    let snapshot = state.knowledge.snapshot();
    let include_facts = request
        .query
        .get("include_facts")
        .map(|value| value == "1" || value == "true")
        .unwrap_or(false);
    let mut body = serde_json::json!({
        "generation": snapshot.generation,
        "facts": snapshot.facts.len(),
        "aliases": snapshot.aliases,
        "fingerprint": snapshot.fingerprint,
        "journal": snapshot.journal_len,
        "cache_epoch": state.cache_epoch().kb_revision,
    });
    if include_facts {
        body["data"] = serde_json::Value::Array(snapshot.facts.iter().map(fact_to_json).collect());
    }
    Response::json(200, &body)
}
async fn handle_knowledge_assert(state: &Arc<ApiState>, request: &Request) -> Response {
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
    let applied = state.assert_facts(&facts, source);
    let snapshot = state.knowledge.snapshot();
    let status = if applied > 0 { 201 } else { 200 };
    Response::json(
        status,
        &serde_json::json!({
            "status": if applied > 0 { "applied" } else { "unchanged" },
            "applied": applied,
            "source": source,
            "scope": "process",
            "generation": snapshot.generation,
            "facts": snapshot.facts.len(),
            "fingerprint": snapshot.fingerprint,
        }),
    )
}
fn knowledge_change_to_json(change: &KnowledgeChange) -> serde_json::Value {
    match change {
        KnowledgeChange::Facts {
            generation,
            facts,
            source,
        } => serde_json::json!({
            "generation": generation,
            "kind": "facts",
            "source": source,
            "facts": facts.iter().map(fact_to_json).collect::<Vec<_>>(),
        }),
        KnowledgeChange::Aliases {
            generation,
            aliases,
            source,
        } => serde_json::json!({
            "generation": generation,
            "kind": "aliases",
            "source": source,
            "aliases": aliases.iter().map(|(surface, canonical)| {
                serde_json::json!({"surface": surface, "canonical": canonical})
            }).collect::<Vec<_>>(),
        }),
        KnowledgeChange::Reset { generation, source } => serde_json::json!({
            "generation": generation,
            "kind": "reset",
            "source": source,
        }),
    }
}

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
            references: BindingLedger::new(),
            identity: UserIdentityLedger::new(),
            epistemic_followup: None,
            type_memory: crate::type_memory::TypeDialogueMemory::new(),
            last_identity_card: None,
            pending_reference: None,
            reference_resolver: ContextualResolver::new(),
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
            "失败恢复：模式 {}；{}；挂账 {} 个待定问题；会话确认 {} 条；否定证据 {} 条。\n{}",
            self.recovery_mode.label(),
            format!(
                "回执 {} 条（未结 {}）",
                self.recovery.ledger.len(),
                self.recovery.ledger.open_count()
            ),
            self.recovery.pending.len(),
            self.recovery.confirmed.len(),
            self.recovery.refuted.len(),
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

    /// 语境指称绑定账本（只读；供宿主持久化与审计）。
    pub fn reference_ledger(&self) -> &BindingLedger {
        &self.references
    }

    /// 用户身份台账（只读；名字只在本会话内有效）。
    pub fn identity_ledger(&self) -> &UserIdentityLedger {
        &self.identity
    }

    /// 最近一轮身份回答的结构化身份卡（运行时视图，不进快照）。
    pub fn last_identity_card(&self) -> Option<&IdentityCard> {
        self.last_identity_card.as_ref()
    }

    /// 待回答的歧义澄清请求（只读；无挂起时为 None）。
    pub fn pending_reference(&self) -> Option<&ClarifyRequest> {
        self.pending_reference.as_ref()
    }

    /// 插件配方回答后标注下一轮焦点。正向问句把答案实体置顶，
    /// 逆向问句仍保留被查询的主语；只移动焦点，不新增对话轮。
    pub fn note_plugin_focus(&mut self, entity: &str) {
        self.push_focus(entity.trim());
    }

    /// 跨会话焦点兜底：快照没有恢复出焦点时，用编年史最近的实体线程移动焦点。
    /// 只影响下一轮的指代先验，不新增对话轮，也不写任何事实。
    pub fn note_restored_focus(&mut self, entity: &str) -> bool {
        let entity = entity.trim();
        if entity.is_empty() {
            return false;
        }
        self.push_focus(entity);
        true
    }

    pub fn focus_stack(&self) -> &[String] {
        &self.focus
    }

    /// 从既有状态派生工作记忆索引（不复制事实正文，不改变任何状态）。
    pub fn working_memory(&self) -> WorkingMemory {
        let mut memory = WorkingMemory::new(WORKING_MEMORY_CAPACITY);
        let latest_open = self.open_turns.latest().map(|open| open.id);
        let latest_pending = self
            .recovery
            .latest_pending()
            .map(|pending| pending.receipt_id.clone());

        for turn in &self.turns {
            let answered = turn.answer_excerpt.is_some();
            let mut item = AttentionItem::new(
                format!("turn:{}", turn.index),
                AttentionKind::Turn,
                turn.user.clone(),
            )
            .recency(turn.index as u64)
            .trust(if answered { 80 } else { 50 })
            .open(!answered);
            if let Some(entity) = &turn.entity {
                item = item.with_entities([entity.clone()]);
            }
            if let Some(attribute) = &turn.attribute {
                item = item.with_predicates([attribute.clone()]);
            }
            memory.push(item);
        }

        let focus_recency = self.turns.len() as u64;
        for (index, entity) in self.focus.iter().enumerate() {
            memory.push(
                AttentionItem::new(
                    format!("focus:{}", entity),
                    AttentionKind::Focus,
                    entity.clone(),
                )
                .with_entities([entity.clone()])
                .recency(focus_recency.saturating_add(index as u64 + 1))
                .trust(100)
                .pinned(true),
            );
        }

        for open in self.open_turns.entries() {
            let pinned = latest_open == Some(open.id);
            let mut item = AttentionItem::new(
                format!("open:{}", open.id),
                AttentionKind::OpenQuestion,
                open.input.clone(),
            )
            .recency(open.turn)
            .trust(100)
            .open(true)
            .pinned(pinned);
            if let Some(focus) = &open.focus {
                item = item.with_entities([focus.clone()]);
            }
            if let Some(attribute) = &open.attribute {
                item = item.with_predicates([attribute.clone()]);
            }
            memory.push(item);
        }

        for pending in self.recovery.pending.values() {
            let pinned = latest_pending.as_deref() == Some(pending.receipt_id.as_str());
            let mut item = AttentionItem::new(
                format!("pending:{}", pending.receipt_id),
                AttentionKind::OpenQuestion,
                pending.question.clone(),
            )
            .recency(pending.created_tick)
            .trust(90)
            .open(true)
            .pinned(pinned);
            if let Some(entity) = &pending.entity {
                item = item.with_entities([entity.clone()]);
            }
            if let Some(attribute) = &pending.attribute {
                item = item.with_predicates([attribute.clone()]);
            }
            memory.push(item);
        }

        for fact in &self.recovery.confirmed {
            memory.push(
                AttentionItem::new(
                    format!(
                        "fact:{}\u{1}{}\u{1}{}",
                        fact.entity, fact.attribute, fact.value
                    ),
                    AttentionKind::SessionFact,
                    format!("{}的{}是{}", fact.entity, fact.attribute, fact.value),
                )
                .with_entities([fact.entity.clone()])
                .with_predicates([fact.attribute.clone()])
                .recency(fact.tick)
                .trust(100),
            );
        }

        if let Some(name) = self.identity.current() {
            memory.push(
                AttentionItem::new("identity:user", AttentionKind::Identity, name)
                    .recency(focus_recency.saturating_add(1))
                    .trust(100),
            );
        }

        memory
    }

    /// 当前轮的注意力查询：焦点 + 最近谓词 + 未完成问题/待补全目标。
    pub fn attention_query(&self) -> AttentionQuery {
        let mut query = AttentionQuery::new()
            .with_focus(self.focus.iter().cloned())
            .with_predicates(
                self.turns
                    .last()
                    .and_then(|turn| turn.attribute.clone())
                    .into_iter(),
            );
        if let Some(open) = self.latest_open_turn() {
            if let Some(attribute) = &open.attribute {
                query.predicates.push(attribute.clone());
            }
            if let Some(focus) = &open.focus {
                query.goals.push(focus.clone());
            }
        }
        if let Some(pending) = self.recovery.latest_pending() {
            if let Some(attribute) = &pending.attribute {
                query.predicates.push(attribute.clone());
            }
            if let Some(entity) = &pending.entity {
                query.goals.push(entity.clone());
            }
        }
        query
    }

    /// 当前轮的上下文包：焦点、谓词与最近的未完成问题共同决定注意力。
    pub fn context_packet(&self, budget: AttentionBudget) -> ContextPacket {
        self.context_packet_with(budget, std::iter::empty())
    }

    /// 在工作记忆之上叠加外部证据（如共享白板声明）后选包；外部项只是索引，
    /// 不改写会话状态，也不自动升格为图谱事实。
    pub fn context_packet_with(
        &self,
        budget: AttentionBudget,
        external: impl IntoIterator<Item = AttentionItem>,
    ) -> ContextPacket {
        let mut memory = self.working_memory();
        for item in external {
            memory.push(item);
        }
        memory.select(&self.attention_query(), budget)
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
            || !self.recovery.confirmed.is_empty()
            || !self.recovery.refuted.is_empty()
        {
            summary.push_str(&format!(
                " 失败恢复：模式 {}，回执 {} 条（未结 {}），挂账 {} 个，会话确认 {} 条，否定证据 {} 条。",
                self.recovery_mode.label(),
                self.recovery.ledger.len(),
                self.recovery.ledger.open_count(),
                self.recovery.pending.len(),
                self.recovery.confirmed.len(),
                self.recovery.refuted.len()
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
            references: self.references.clone(),
            identity: self.identity.clone(),
            epistemic_followup: self.epistemic_followup.clone(),
            type_memory: self.type_memory.clone(),
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
        self.references = snapshot.references.clone();
        self.identity = snapshot.identity.clone();
        self.epistemic_followup = snapshot.epistemic_followup.clone();
        self.type_memory = snapshot.type_memory.clone();
        self.last_identity_card = None;
        self.pending_reference = None;
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

    /// 记录上一轮情态问句的算子与槽位；只保存可回答的已知实体问句。
    fn remember_epistemic_followup(&mut self, input: &str, knowledge: &dyn DialogueKnowledge) {
        let Some(intent) = ProbabilisticQueryIntent::analyze(input) else {
            return;
        };
        let Some(memory) = EpistemicFollowup::from_intent(&intent) else {
            self.epistemic_followup = None;
            return;
        };
        let anchors = [memory.subject.as_deref(), memory.object.as_deref()];
        if anchors
            .into_iter()
            .flatten()
            .any(|entity| !knowledge.known_entity(entity))
        {
            self.epistemic_followup = None;
            return;
        }
        self.epistemic_followup = Some(memory);
    }

    /// 「那王五呢」：沿用上一轮的情态算子与谓词槽位，换掉锚点实体。
    ///
    /// 只接受「(那/换成)? 实体 呢」这种纯省略，不接「那王五的师傅呢」这类
    /// 新属性问句；重建后还会再跑一次 PQI，确认情态没有被模板降级。
    fn plan_epistemic_followup(
        &mut self,
        input: &str,
        knowledge: &dyn DialogueKnowledge,
    ) -> Option<DialogueReply> {
        let memory = self.epistemic_followup.clone()?;
        let entity = Self::epistemic_followup_entity(input, knowledge)?;
        let question = memory.render_followup(&entity)?;
        let intent = ProbabilisticQueryIntent::analyze(&question)?;
        let mode = EpistemicMode::from_code(&memory.mode)?;
        if intent.mode != mode {
            return None;
        }
        let text = knowledge
            .answer_certain(&question)
            .or_else(|| knowledge.answer(&question))?;
        let plan = vec!["epistemic_followup".to_string(), memory.mode.clone()];
        let provenance = vec![format!("承接上一轮{}：{}", mode.label(), question)];
        self.turns.push(DialogueTurn {
            index: self.turns.len() + 1,
            user: input.trim().to_string(),
            kind: MoveKind::AskBinary,
            entity: Some(entity.clone()),
            attribute: Some(memory.predicate.clone()),
            answer_excerpt: Some(excerpt(&text, 40)),
            provenance,
            plan: plan.clone(),
            warrant: None,
            kb_revision: Some(knowledge.revision()),
        });
        self.push_focus(&entity);
        self.last_reply = plan.clone();
        if self.externals.last().map(String::as_str) == Some(input.trim()) {
            self.externals.pop();
        }
        self.remember_epistemic_followup(&question, knowledge);
        Some(DialogueReply {
            text,
            plan,
            kind: MoveKind::AskBinary,
            derivation: None,
        })
    }

    fn epistemic_followup_entity(input: &str, knowledge: &dyn DialogueKnowledge) -> Option<String> {
        let stripped = input
            .trim()
            .trim_end_matches(['？', '?', '！', '!', '。', '.', '，', ',', ' '])
            .trim();
        let mut body = stripped.strip_suffix('呢')?.trim();
        let mut marked = false;
        for prefix in [
            "如果换成",
            "要是换成",
            "那换成",
            "换成",
            "换为",
            "那么",
            "那",
        ] {
            if let Some(rest) = body.strip_prefix(prefix) {
                body = rest.trim();
                marked = true;
            }
        }
        // 裸「王五呢」仍交给普通省略/身份管线；只有明确的承接标记
        //（那/那么/换成…）才把它读成上一轮情态问句的换槽。
        if !marked
            || body.is_empty()
            || body.chars().any(|ch| {
                matches!(
                    ch,
                    '谁' | '什'
                        | '么'
                        | '哪'
                        | '吗'
                        | '的'
                        | '是'
                        | '怎'
                        | '几'
                        | '多'
                        | '？'
                        | '?'
                )
            })
        {
            return None;
        }
        let entities = knowledge.entities_in(body);
        (entities.len() == 1 && entities[0] == body).then(|| body.to_string())
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
        self.last_identity_card = None;
        if let Some(answer) =
            knowledge.type_memory_answer_in_context(input, &mut self.type_memory)
        {
            let plans = vec!["type_memory".to_string()];
            let provenance = answer
                .focus
                .as_ref()
                .map(|focus| vec![format!("类型焦点：{}", focus)])
                .unwrap_or_default();
            self.turns.push(DialogueTurn {
                index: self.turns.len() + 1,
                user: input.trim().to_string(),
                kind: MoveKind::Other,
                entity: None,
                attribute: None,
                answer_excerpt: Some(excerpt(&answer.text, 40)),
                provenance,
                plan: plans.clone(),
                warrant: None,
                kb_revision: Some(knowledge.revision()),
            });
            self.last_reply = plans.clone();
            if self.externals.last().map(String::as_str) == Some(input.trim()) {
                self.externals.pop();
            }
            return Some(DialogueReply {
                text: answer.text,
                plan: plans,
                kind: MoveKind::Other,
                derivation: None,
            });
        }
        self.type_memory.note_non_type_turn();
        if let Some(reply) = self.plan_epistemic_followup(input, knowledge) {
            return Some(reply);
        }
        self.remember_epistemic_followup(input, knowledge);
        if let Some(reply) = self.plan_identity_dialogue(input, knowledge) {
            return Some(reply);
        }
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
        let negative_body = (pending.kind == PendingKind::Provisional)
            .then(|| negative_correction_body(stripped))
            .flatten();
        let correction = negative_body.filter(|body| {
            body.contains('是')
                || body.contains('为')
                || pending.value.as_deref().map(str::trim) != Some(*body)
        });
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
        } else if let Some(correction) = correction {
            (
                recovery::resolve_by_correction(&mut self.recovery, correction)?,
                "recovery_correct",
            )
        } else if negative_body.is_some() {
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
            derivation: None,
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
            derivation: None,
        })
    }

    /// 知识问句的生成式回答：仅在实体已知时才接管（否则让位给常规管线）。
    pub fn plan_reply(
        &mut self,
        input: &str,
        knowledge: &dyn DialogueKnowledge,
    ) -> Option<DialogueReply> {
        self.last_identity_card = None;
        if let Some(text) = knowledge.learned_rewrite_answer(input) {
            let plan = vec!["learned_rewrite".to_string()];
            let entity = knowledge.entities_in(input).into_iter().next();
            self.turns.push(DialogueTurn {
                index: self.turns.len() + 1,
                user: input.trim().to_string(),
                kind: MoveKind::AskAttribute,
                entity: entity.clone(),
                attribute: None,
                answer_excerpt: Some(excerpt(&text, 40)),
                provenance: Vec::new(),
                plan: plan.clone(),
                warrant: None,
                kb_revision: Some(knowledge.revision()),
            });
            if let Some(entity) = entity {
                self.push_focus(&entity);
            }
            self.last_reply = plan.clone();
            return Some(DialogueReply {
                text,
                plan,
                kind: MoveKind::AskAttribute,
                derivation: None,
            });
        }
        if let Some(reply) = self.plan_epistemic_followup(input, knowledge) {
            return Some(reply);
        }
        self.remember_epistemic_followup(input, knowledge);
        // 「A 是 B 的 C」交给蕴含/校验管线；指代重写会把 A 误当成待消解
        // 的称谓并改写成「B 的 C」，从而翻转判断方向。
        if is_entity_copula_relation_statement(input, knowledge) {
            return None;
        }
        if let Some(reply) = self.plan_identity_dialogue(input, knowledge) {
            return Some(reply);
        }
        if let Some(reply) = self.try_trace_correction(input, knowledge) {
            return Some(reply);
        }
        if let Some(reply) = self.plan_frame_reply(input, knowledge) {
            return Some(reply);
        }
        if let Some(reply) = self.plan_board_reply(input, knowledge) {
            return Some(reply);
        }
        // 语境指称栈：先结算上一轮澄清，再对本轮做绑定 / 改写 / 追问。
        // `reference_view` 缺省为 None，因此未接入链接器的知识源行为不变。
        let effective_input = match self.resolve_reference_input(input, knowledge) {
            Some(ReferenceDecision::Clarify(text, mention)) => {
                return Some(self.reference_clarify_reply(input, text, mention));
            }
            Some(ReferenceDecision::Rewrite(text)) => text,
            Some(ReferenceDecision::Skip) | None => input.to_string(),
        };
        let frame = self.analyze(&effective_input, knowledge);
        match frame.kind {
            MoveKind::AskAttribute | MoveKind::AskIdentity => self.execute(frame, knowledge),
            _ => None,
        }
    }

    /// 身份域对话：系统 / 用户身份完全在对话层闭合；实体身份交给知识管线。
    ///
    /// 第一/第二人称绝不进入实体焦点栈，也不会被当作图谱实体查询；用户自述
    /// 只写入会话台账（随快照持久化），不触碰图谱。
    fn plan_identity_dialogue(
        &mut self,
        input: &str,
        knowledge: &dyn DialogueKnowledge,
    ) -> Option<DialogueReply> {
        self.last_recovery = None;
        let act = identity_dialogue::classify(input);
        let turn_no = self.turns.len() + 1;
        let (card, plan, kind, text) = match act {
            IdentityAct::AskSystem => (
                identity_dialogue::system_card(),
                "identity_system".to_string(),
                MoveKind::AskIdentity,
                String::new(),
            ),
            IdentityAct::AskUser => {
                let card = identity_dialogue::user_card(&self.identity);
                let plan = match card.status {
                    IdentityStatus::Unknown => "identity_user_unknown",
                    IdentityStatus::Forgotten => "identity_user_forgotten",
                    _ => "identity_user_known",
                };
                (card, plan.to_string(), MoveKind::AskIdentity, String::new())
            }
            IdentityAct::Introduce { name } => {
                let outcome = self.identity.remember(&name, turn_no);
                let card = identity_dialogue::user_card(&self.identity);
                let (plan, text) = match outcome {
                    identity_dialogue::RememberOutcome::Corrected { .. } => (
                        "identity_user_correct",
                        format!(
                            "收到，已把称呼更正为「{}」。之前的称呼在本会话里不再使用。",
                            name
                        ),
                    ),
                    identity_dialogue::RememberOutcome::Unchanged => (
                        "identity_user_unchanged",
                        format!("我记得，你叫「{}」。", name),
                    ),
                    identity_dialogue::RememberOutcome::Remembered => (
                        "identity_user_remember",
                        format!(
                            "好，我记住了：你叫「{}」。这个称呼只保存在本会话里，你可以随时让我忘记。",
                            name
                        ),
                    ),
                };
                (card, plan.to_string(), MoveKind::Assert, text)
            }
            IdentityAct::ForgetUser => {
                let had_identity = !self.identity.is_empty();
                self.identity.forget();
                let card = identity_dialogue::user_card(&self.identity);
                let text = if had_identity {
                    String::new()
                } else {
                    "你还没有在这个会话里告诉我你的名字；我仍然不会保留任何称呼。".to_string()
                };
                (
                    card,
                    "identity_user_forget".to_string(),
                    MoveKind::Assert,
                    text,
                )
            }
            IdentityAct::AskEntity | IdentityAct::Other => return None,
        };
        let text = if text.is_empty() { card.render() } else { text };
        let plans = vec![plan];
        self.turns.push(DialogueTurn {
            index: turn_no,
            user: input.trim().to_string(),
            kind: kind.clone(),
            entity: None,
            attribute: None,
            answer_excerpt: Some(excerpt(&text, 40)),
            provenance: Vec::new(),
            plan: plans.clone(),
            warrant: None,
            kb_revision: Some(knowledge.revision()),
        });
        self.last_reply = plans.clone();
        self.last_identity_card = Some(card);
        if self.externals.last().map(String::as_str) == Some(input.trim()) {
            self.externals.pop();
        }
        Some(DialogueReply {
            text,
            plan: plans,
            kind,
            derivation: None,
        })
    }

    /// 一轮链接裁决的结果。
    fn resolve_reference_input(
        &mut self,
        input: &str,
        knowledge: &dyn DialogueKnowledge,
    ) -> Option<ReferenceDecision> {
        let view = knowledge.reference_view()?;
        self.references.begin_turn();
        if let Some(pending) = self.pending_reference.take() {
            match parse_clarify_act(input, &pending.question) {
                Some(ClarifyAct::Choose { entity, .. }) | Some(ClarifyAct::Restate { entity }) => {
                    self.references.bind(
                        &pending.key,
                        &entity,
                        BindingScope::Topic,
                        &format!("澄清回答：{}", input.trim()),
                    );
                    return Some(ReferenceDecision::Rewrite(rewrite_at(
                        &pending.original_input,
                        pending.start,
                        pending.mention.chars().count(),
                        &entity,
                    )));
                }
                Some(ClarifyAct::Correct { to, .. }) => {
                    self.references.correct(
                        &pending.key,
                        &to,
                        &format!("澄清纠正：{}", input.trim()),
                    );
                    return Some(ReferenceDecision::Rewrite(rewrite_at(
                        &pending.original_input,
                        pending.start,
                        pending.mention.chars().count(),
                        &to,
                    )));
                }
                Some(ClarifyAct::Unknown) | Some(ClarifyAct::Decline) => {
                    return Some(ReferenceDecision::Skip);
                }
                None => {}
            }
        }
        let hints = knowledge.reference_hints(input);
        let outcome = self.reference_resolver.resolve_text(
            input,
            view.linker,
            view.source,
            &self.references,
            &hints,
            self.focus.last().map(String::as_str),
        );
        if let Some(ask) = outcome.ask {
            let text = ask.question.render();
            let mention = ask.mention.clone();
            self.pending_reference = Some(ask);
            return Some(ReferenceDecision::Clarify(text, mention));
        }
        if let Some(rewrite) = outcome.rewrite {
            if rewrite != input {
                return Some(ReferenceDecision::Rewrite(rewrite));
            }
        }
        Some(ReferenceDecision::Skip)
    }

    fn reference_clarify_reply(
        &mut self,
        input: &str,
        text: String,
        mention: String,
    ) -> DialogueReply {
        let plans = vec!["clarify_reference".to_string()];
        self.turns.push(DialogueTurn {
            index: self.turns.len() + 1,
            user: input.trim().to_string(),
            kind: MoveKind::AskIdentity,
            entity: Some(mention),
            attribute: None,
            answer_excerpt: Some(excerpt(&text, 40)),
            provenance: Vec::new(),
            plan: plans.clone(),
            warrant: None,
            kb_revision: None,
        });
        self.last_reply = plans.clone();
        DialogueReply {
            text,
            plan: plans,
            kind: MoveKind::AskIdentity,
            derivation: None,
        }
    }

    pub fn plan_board_reply(
        &mut self,
        input: &str,
        knowledge: &dyn DialogueKnowledge,
    ) -> Option<DialogueReply> {
        // 「A 是 B 的 C」属于真值判断，不能让查询板按「B 的 C」重扫一遍：
        // 否则「张三是李四的师傅吗」会被答成「李四的师傅是王五」。
        if is_entity_copula_relation_statement(input, knowledge) {
            return None;
        }
        let revision = knowledge.revision();
        let attribute_hint = self.turns.last().and_then(|turn| turn.attribute.clone());
        let outcome =
            self.board
                .apply_with_hint(input, knowledge, revision, attribute_hint.as_deref())?;
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
            derivation: None,
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

        // 第一/第二人称身份问句先于实体链接：它们不是图谱实体，也不应污染焦点栈。
        match identity_dialogue::classify(stripped) {
            IdentityAct::AskSystem => {
                let mut frame = UtteranceFrame::new(raw, MoveKind::AskIdentity);
                frame.question = Some("identity:system".to_string());
                return frame;
            }
            IdentityAct::AskUser => {
                let mut frame = UtteranceFrame::new(raw, MoveKind::AskIdentity);
                frame.question = Some("identity:user".to_string());
                return frame;
            }
            IdentityAct::Introduce { name } => {
                let mut frame = UtteranceFrame::new(raw, MoveKind::Assert);
                frame.question = Some(format!("identity:introduce:{}", name));
                return frame;
            }
            IdentityAct::ForgetUser => {
                let mut frame = UtteranceFrame::new(raw, MoveKind::Assert);
                frame.question = Some("identity:forget".to_string());
                return frame;
            }
            IdentityAct::AskEntity | IdentityAct::Other => {}
        }

        let explicit: Vec<String> = knowledge.entities_in(raw);
        let focus = self.focus.last().cloned();
        let question_markers = ["谁", "什么", "哪", "多少", "几", "吗", "呢", "怎么"];
        let has_question = question_markers
            .iter()
            .any(|marker| stripped.contains(marker));

        // 地点类实体的「多大」问的是面积而不是年龄：北京有多大。
        // 类型未知时仍交给通用规划器（人物默认年龄），不猜测。
        if let Some(prefix) = stripped.strip_suffix("多大") {
            let subject = prefix.trim().trim_end_matches('有').trim();
            let entity = explicit
                .iter()
                .find(|entity| !subject.is_empty() && subject.contains(entity.as_str()))
                .cloned()
                .or_else(|| {
                    if subject.is_empty() || contains_pronoun(subject) {
                        focus.clone()
                    } else {
                        None
                    }
                });
            if let Some(entity) = entity {
                if knowledge
                    .entity_type(&entity)
                    .is_some_and(|etype| is_place_type(&etype))
                {
                    return UtteranceFrame {
                        raw: raw.to_string(),
                        kind: MoveKind::AskAttribute,
                        entity: Some(entity.clone()),
                        attribute: Some("面积".to_string()),
                        question: Some(format!("{}的面积是多少", entity)),
                        inherited: false,
                    };
                }
            }
        }

        // 地点类实体的「有多少人」问的是人口，不是计数未知名词。
        for marker in ["有多少人口", "有多少人", "多少人口"] {
            let Some(prefix) = stripped.strip_suffix(marker) else {
                continue;
            };
            let subject = prefix.trim().trim_end_matches('有').trim();
            let entity = explicit
                .iter()
                .find(|entity| !subject.is_empty() && subject.contains(entity.as_str()))
                .cloned()
                .or_else(|| {
                    if subject.is_empty() || contains_pronoun(subject) {
                        focus.clone()
                    } else {
                        None
                    }
                });
            if let Some(entity) = entity {
                if knowledge
                    .entity_type(&entity)
                    .is_some_and(|etype| is_place_type(&etype))
                {
                    return UtteranceFrame {
                        raw: raw.to_string(),
                        kind: MoveKind::AskAttribute,
                        entity: Some(entity.clone()),
                        attribute: Some("人口".to_string()),
                        question: Some(format!("{}的人口是多少", entity)),
                        inherited: false,
                    };
                }
            }
        }

        // 属性问句：S的P（是什么/是多少/是谁/怎么样/呢……），
        // 也接受"姚明的身高"这类无问词的简写，但不吸收含"是"的陈述。
        // 「A 是 B 的 C」是系表判断（校验方向），不是「A是B」的属性问句。
        if (has_question || !stripped.contains('是')) && !is_copula_relation_statement(stripped) {
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
}

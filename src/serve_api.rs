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
//! - `GET  /metrics/semantic`      跨会话语义检索 KPI 聚合（窗口/告警/预算）
//! - `POST /sessions`              新建会话 → `{"session_id": "s-1"}`
//! - `GET  /sessions/{id}`         会话快照（焦点栈 / 历史摘要 / 策略）
//! - `GET  /sessions/{id}/layers`  四层知识容器清单（基础/会话/推测/假设）
//! - `GET  /sessions/{id}/semantic` 会话级语义检索观测（不含原始问句）
//! - `GET  /sessions/{id}/export`  导出会话信封（可移植 JSON）
//! - `POST /sessions/import`       导入会话信封重建会话
//! - `DELETE /sessions/{id}`       关闭会话
//! - `GET/POST /sessions/{id}/memory` 会话可见的共享语义白板（全局 + 会话私有）
//! - `DELETE /sessions/{id}/memory/{claim}` 撤回该会话私有声明
//! - `GET/POST /v1/memory`          全局共享语义白板（`?session=` 只读叠加会话视图）
//! - `DELETE /v1/memory/{claim}`    撤回全局共享声明
//! - `GET  /v1/knowledge`          进程内知识台账（事实/别名版本与指纹）
//! - `GET  /v1/knowledge/changes`  增量变更日志（`?since=N`，缺口显式 409）
//! - `POST /v1/knowledge/facts`    追加全局事实（下一轮对所有活跃会话可见）
//! - `PUT  /v1/knowledge/facts`    整表替换全局事实（懒重建，可更正/撤回）
//! - `POST /v1/knowledge/aliases`  准入实体别名（下一轮对所有活跃会话可见）
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

include!("serve_api/serve_api_part_01.rs");
include!("serve_api/serve_api_part_02.rs");
include!("serve_api/serve_api_part_03.rs");

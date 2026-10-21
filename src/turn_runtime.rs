//! TRN：有界、可取消的轮次运行时（Bounded, Cancellable Turn Runtime）。
//!
//! 对话系统已经有「知识失败」的恢复阶梯，但一轮对话本身还缺生命周期：
//! 开始、阶段预算、取消、提交点、失败回执。没有这些，客户端断开或用户换话题
//! 时，会话 actor 仍会把整轮算完并提交半成品状态；引擎超时也会被折叠成
//! 「查无」，与真正的不知道混为一谈。
//!
//! 本模块把一轮对话变成可观察的工作单元：
//!
//! - [`TurnPhase`]：阶段边界（准备 / 改写 / 召回 / 推理 / 生成 / 提交）；
//! - [`TurnBudget`]：墙钟只用于 liveness，不参与知识真假；`Observe` 只记录
//!   超时，`Enforce` 才真正中止；
//! - [`CancelToken`]：可克隆的取消句柄（客户端断开、换话题、服务关闭）；
//! - [`TurnFault`]：封闭的类型化失败（取消 / 超时 / 存储 / 引擎 / 崩溃 /
//!   快照），每种都有自己的稳定状态码与可重试性；
//! - [`OpenTurnBook`]：未完成轮次是一等记忆（原始输入、失败阶段、焦点、
//!   已尝试次数、TTL），下一轮可显式继续或放弃。
//!
//! 设计纪律：不引入概率、采样或训练；所有状态有限、确定性、可审计。
//! `TurnFault` 的措辞绝不说「没有记录」——运行时故障不是知识论上的未知。

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// 未完成轮次上限（超出丢弃最旧）。
pub const OPEN_TURN_CAPACITY: usize = 8;
/// 未完成轮次的有效轮数（TTL，按逻辑轮次而非挂钟）。
pub const OPEN_TURN_TTL_TURNS: u64 = 3;
/// 同一输入最多自动继续尝试的次数。
pub const MAX_RESUME_ATTEMPTS: u32 = 3;
/// 未完成轮次 detail 的字符上限（防止异常信息无限增长）。
pub const DETAIL_LIMIT: usize = 160;
/// 默认轮次预算（只限制 liveness；知识真假仍由证据决定）。
pub const DEFAULT_TURN_BUDGET: Duration = Duration::from_secs(30);

/// 一轮对话的阶段边界。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnPhase {
    /// 读入、历史与临时状态准备。
    Prepare,
    /// 插件词表 / 本体改写。
    Rewrite,
    /// 对话召回与上下文补全。
    Recall,
    /// 符号推理 / 检索。
    Reason,
    /// 生成、策略修饰与接地。
    Generate,
    /// 原子提交（成功后不再回滚）。
    Commit,
}

impl TurnPhase {
    pub fn code(self) -> &'static str {
        match self {
            TurnPhase::Prepare => "prepare",
            TurnPhase::Rewrite => "rewrite",
            TurnPhase::Recall => "recall",
            TurnPhase::Reason => "reason",
            TurnPhase::Generate => "generate",
            TurnPhase::Commit => "commit",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            TurnPhase::Prepare => "准备",
            TurnPhase::Rewrite => "改写",
            TurnPhase::Recall => "召回",
            TurnPhase::Reason => "推理",
            TurnPhase::Generate => "生成",
            TurnPhase::Commit => "提交",
        }
    }
}

impl Default for TurnPhase {
    fn default() -> Self {
        TurnPhase::Prepare
    }
}

/// 类型化轮次故障。封闭枚举：新增错误必须显式出现在这里。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnFaultKind {
    /// 客户端离开 / 换话题 / 服务关闭。
    Cancelled,
    /// 阶段预算耗尽（墙钟只用于 liveness）。
    Deadline,
    /// 显式预算（尝试次数、片段数等）耗尽。
    BudgetExhausted,
    /// 存储读取失败。
    Store,
    /// Prolog / 策略求值失败（引擎故障，绝非无解）。
    Prolog,
    /// 推理进程 panic（actor 必须存活，会话标记为可恢复）。
    Panic,
    /// 状态快照 / 回滚失败（会话不得带着半状态继续）。
    Snapshot,
}

impl TurnFaultKind {
    /// 稳定状态码（API / 台账 / 基线用）。
    pub fn code(self) -> &'static str {
        match self {
            TurnFaultKind::Cancelled => "cancelled",
            TurnFaultKind::Deadline => "deadline",
            TurnFaultKind::BudgetExhausted => "budget_exhausted",
            TurnFaultKind::Store => "store_fault",
            TurnFaultKind::Prolog => "engine_fault",
            TurnFaultKind::Panic => "panic",
            TurnFaultKind::Snapshot => "snapshot_fault",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            TurnFaultKind::Cancelled => "已取消",
            TurnFaultKind::Deadline => "超出预算",
            TurnFaultKind::BudgetExhausted => "预算耗尽",
            TurnFaultKind::Store => "存储故障",
            TurnFaultKind::Prolog => "推理引擎故障",
            TurnFaultKind::Panic => "推理崩溃",
            TurnFaultKind::Snapshot => "快照故障",
        }
    }

    /// 是否值得原轮重试（取消与预算耗尽不是同一件事）。
    pub fn retryable(self) -> bool {
        match self {
            TurnFaultKind::Cancelled => true,
            TurnFaultKind::Deadline => true,
            TurnFaultKind::BudgetExhausted => false,
            TurnFaultKind::Store => true,
            TurnFaultKind::Prolog => true,
            TurnFaultKind::Panic => false,
            TurnFaultKind::Snapshot => false,
        }
    }

    /// 对外措辞：故障绝不伪装成「没有记录」。
    pub fn statement(self) -> &'static str {
        match self {
            TurnFaultKind::Cancelled => {
                "这一轮已经取消，没有提交任何结论；原始问题仍然保留，可以继续。"
            }
            TurnFaultKind::Deadline => {
                "这一轮在预算内没有完成，我已经停下且没有提交半成品；可以继续这一问。"
            }
            TurnFaultKind::BudgetExhausted => {
                "这一轮的重试预算已经用完；换个问法或补充线索后再试。"
            }
            TurnFaultKind::Store => {
                "我读取知识存储时出错了（已记录）；这不是图谱里没有记录，请稍后重试。"
            }
            TurnFaultKind::Prolog => {
                "我的推理引擎出错了（已记录）；这不是图谱里没有记录，请稍后重试。"
            }
            TurnFaultKind::Panic => {
                "我这边推理时崩溃了（已记录）；会话仍然可用，可以继续这一问。"
            }
            TurnFaultKind::Snapshot => {
                "我无法安全保存这一轮的状态（已记录）；为避免半状态污染，本轮没有提交。"
            }
        }
    }
}

/// 一次轮次故障（类型 + 阶段 + 耗时 + 可重试性）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnFault {
    pub kind: TurnFaultKind,
    pub phase: TurnPhase,
    #[serde(default)]
    pub detail: String,
    #[serde(default)]
    pub elapsed_ms: u64,
    #[serde(default)]
    pub retryable: bool,
}

impl TurnFault {
    pub fn new(kind: TurnFaultKind, phase: TurnPhase, detail: impl Into<String>, elapsed_ms: u64) -> Self {
        Self {
            kind,
            phase,
            detail: truncate_chars(&detail.into(), DETAIL_LIMIT),
            elapsed_ms,
            retryable: kind.retryable(),
        }
    }

    pub fn deadline(phase: TurnPhase, elapsed_ms: u64) -> Self {
        Self::new(
            TurnFaultKind::Deadline,
            phase,
            format!("阶段「{}」超出预算", phase.label()),
            elapsed_ms,
        )
    }

    pub fn cancelled(phase: TurnPhase, elapsed_ms: u64) -> Self {
        Self::new(
            TurnFaultKind::Cancelled,
            phase,
            format!("阶段「{}」被取消", phase.label()),
            elapsed_ms,
        )
    }

    pub fn panic(phase: TurnPhase, payload: &str, elapsed_ms: u64) -> Self {
        Self::new(TurnFaultKind::Panic, phase, payload, elapsed_ms)
    }

    pub fn status(&self) -> &'static str {
        self.kind.code()
    }

    /// 对外文本：固定模板 + 阶段/耗时细节。
    pub fn render(&self) -> String {
        if self.detail.is_empty() {
            self.kind.statement().to_string()
        } else {
            format!("{}（{}：{}）", self.kind.statement(), self.phase.label(), self.detail)
        }
    }
}

/// 预算执法模式：默认 Observe，测试或显式配置才 Enforce。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnEnforcement {
    /// 只记录超时，不改变回答（先收集真实分布）。
    Observe,
    /// 超时即中止并拒绝提交。
    Enforce,
}

impl Default for TurnEnforcement {
    fn default() -> Self {
        TurnEnforcement::Observe
    }
}

/// 一轮的墙钟预算。`None` 表示不设上限（仍可取消）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TurnBudget {
    pub wall: Option<Duration>,
    pub enforcement: TurnEnforcement,
}

impl Default for TurnBudget {
    fn default() -> Self {
        Self::observed(DEFAULT_TURN_BUDGET)
    }
}

impl TurnBudget {
    pub const fn unlimited() -> Self {
        Self {
            wall: None,
            enforcement: TurnEnforcement::Observe,
        }
    }

    pub const fn observed(wall: Duration) -> Self {
        Self {
            wall: Some(wall),
            enforcement: TurnEnforcement::Observe,
        }
    }

    pub const fn enforced(wall: Duration) -> Self {
        Self {
            wall: Some(wall),
            enforcement: TurnEnforcement::Enforce,
        }
    }

    pub const fn with_enforcement(mut self, enforcement: TurnEnforcement) -> Self {
        self.enforcement = enforcement;
        self
    }

    pub fn wall_ms(&self) -> Option<u64> {
        self.wall.map(|wall| wall.as_millis().min(u64::MAX as u128) as u64)
    }
}

/// 可克隆的取消句柄：所有 clone 共享同一原子标志。
#[derive(Debug, Clone, Default)]
pub struct CancelToken {
    flag: Arc<AtomicBool>,
}

impl CancelToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.flag.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Acquire)
    }

    pub fn reset(&self) {
        self.flag.store(false, Ordering::Release);
    }
}

/// 单轮控制器：阶段入口检查取消与预算。
#[derive(Debug)]
pub struct TurnControl {
    turn: u64,
    budget: TurnBudget,
    started: Instant,
    phase: TurnPhase,
    cancel: CancelToken,
    observed_overrun: Option<TurnPhase>,
}

impl TurnControl {
    pub fn new(turn: u64, budget: TurnBudget) -> Self {
        Self::starting_at(turn, budget, Instant::now())
    }

    /// 使用外部取消句柄（HTTP 客户端断开 / 服务关闭时由等待方触发）。
    pub fn with_cancel(turn: u64, budget: TurnBudget, cancel: CancelToken) -> Self {
        let mut control = Self::new(turn, budget);
        control.cancel = cancel;
        control
    }

    /// 测试与重放可注入起点；生产路径只用 [`Self::new`]。
    pub fn starting_at(turn: u64, budget: TurnBudget, started: Instant) -> Self {
        Self {
            turn,
            budget,
            started,
            phase: TurnPhase::Prepare,
            cancel: CancelToken::new(),
            observed_overrun: None,
        }
    }

    pub fn turn(&self) -> u64 {
        self.turn
    }

    pub fn phase(&self) -> TurnPhase {
        self.phase
    }

    pub fn budget(&self) -> TurnBudget {
        self.budget
    }

    pub fn cancel_handle(&self) -> CancelToken {
        self.cancel.clone()
    }

    pub fn cancel(&self) {
        self.cancel.cancel();
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel.is_cancelled()
    }

    pub fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }

    pub fn elapsed_ms(&self) -> u64 {
        self.elapsed().as_millis().min(u64::MAX as u128) as u64
    }

    /// 本次运行中观察到的超时阶段（Observe 模式下可上报）。
    pub fn observed_overrun(&self) -> Option<TurnPhase> {
        self.observed_overrun
    }

    /// 阶段入口：先切换阶段，再做一次取消 / 预算检查。
    pub fn enter(&mut self, phase: TurnPhase) -> Result<(), TurnFault> {
        self.phase = phase;
        self.check()
    }

    /// 只检查不切换阶段（长时间调用返回后的提交前检查）。
    pub fn check(&mut self) -> Result<(), TurnFault> {
        if self.cancel.is_cancelled() {
            return Err(TurnFault::cancelled(self.phase, self.elapsed_ms()));
        }
        if let Some(wall) = self.budget.wall {
            if self.started.elapsed() >= wall {
                match self.budget.enforcement {
                    TurnEnforcement::Observe => {
                        self.observed_overrun.get_or_insert(self.phase);
                    }
                    TurnEnforcement::Enforce => {
                        return Err(TurnFault::deadline(self.phase, self.elapsed_ms()));
                    }
                }
            }
        }
        Ok(())
    }
}

/// 未完成轮次：下一轮可继续或放弃的一等记忆。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenTurn {
    /// 台账内唯一 id（`1` 起，与快照水位一致）。
    pub id: u64,
    /// 原始输入（未改写、未裁剪）。
    pub input: String,
    /// 失败阶段。
    pub phase: TurnPhase,
    /// 失败类型。
    pub fault: TurnFaultKind,
    /// 失败细节（有界）。
    #[serde(default)]
    pub detail: String,
    /// 失败时的焦点实体（用于诊断，不用于伪造回答）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub focus: Option<String>,
    /// 失败时正在追问的属性（如有）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attribute: Option<String>,
    /// 已尝试次数（首次失败为 1）。
    pub attempts: u32,
    /// 创建时的逻辑轮次（TTL 计量）。
    pub created_tick: u64,
    /// 发生失败的对话轮号。
    pub turn: u64,
}

impl OpenTurn {
    pub fn can_resume(&self, tick: u64) -> bool {
        self.attempts < MAX_RESUME_ATTEMPTS
            && tick.saturating_sub(self.created_tick) < OPEN_TURN_TTL_TURNS
    }

    pub fn is_expired(&self, tick: u64) -> bool {
        tick.saturating_sub(self.created_tick) >= OPEN_TURN_TTL_TURNS
    }

    /// 对外提示（不替代回答，只说明停在哪、可以怎么办）。
    pub fn render_notice(&self) -> String {
        format!(
            "上一轮在「{}」阶段因「{}」未完成（第 {} 次）；原始问题仍保留，可以继续这一问。",
            self.phase.label(),
            self.fault.label(),
            self.attempts
        )
    }
}

/// 未完成轮次台账：有界、确定性排序、可随对话快照持久化。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenTurnBook {
    #[serde(default)]
    next_id: u64,
    #[serde(default)]
    tick: u64,
    #[serde(default)]
    entries: BTreeMap<u64, OpenTurn>,
}

impl OpenTurnBook {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn tick(&self) -> u64 {
        self.tick
    }

    pub fn entries(&self) -> impl Iterator<Item = &OpenTurn> {
        self.entries.values()
    }

    /// 推进一个逻辑轮次并按 TTL 清理过期条目；返回清理条数。
    pub fn begin_tick(&mut self) -> usize {
        self.tick = self.tick.saturating_add(1);
        let tick = self.tick;
        let before = self.entries.len();
        self.entries.retain(|_, open| !open.is_expired(tick));
        before - self.entries.len()
    }

    pub fn latest(&self) -> Option<&OpenTurn> {
        let tick = self.tick;
        self.entries
            .values()
            .filter(|open| !open.is_expired(tick))
            .max_by(|left, right| {
                left.created_tick
                    .cmp(&right.created_tick)
                    .then_with(|| left.id.cmp(&right.id))
            })
    }

    /// 取走最新未完成轮次（继续语义：先消费，再重放；失败后重建 attempts+1）。
    pub fn take_latest(&mut self) -> Option<OpenTurn> {
        let id = self.latest().map(|open| open.id)?;
        self.entries.remove(&id)
    }

    /// 放弃最新未完成轮次。
    pub fn dismiss_latest(&mut self) -> Option<OpenTurn> {
        self.take_latest()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// 记录一次未完成轮次（首次 attempts = 1）。
    pub fn record(
        &mut self,
        input: &str,
        turn: u64,
        phase: TurnPhase,
        fault: TurnFaultKind,
        detail: &str,
        focus: Option<&str>,
        attribute: Option<&str>,
    ) -> u64 {
        self.record_attempts(input, turn, phase, fault, detail, focus, attribute, 1)
    }

    /// 重放失败后重建未完成轮次；attempts 递增且不超过上限 + 1（保留终态可见性）。
    pub fn record_attempts(
        &mut self,
        input: &str,
        turn: u64,
        phase: TurnPhase,
        fault: TurnFaultKind,
        detail: &str,
        focus: Option<&str>,
        attribute: Option<&str>,
        attempts: u32,
    ) -> u64 {
        self.next_id = self.next_id.saturating_add(1);
        let id = self.next_id;
        let attempts = attempts.max(1).min(MAX_RESUME_ATTEMPTS.saturating_add(1));
        self.entries.insert(
            id,
            OpenTurn {
                id,
                input: truncate_chars(input, 512),
                phase,
                fault,
                detail: truncate_chars(detail, DETAIL_LIMIT),
                focus: focus.map(str::to_string),
                attribute: attribute.map(str::to_string),
                attempts,
                created_tick: self.tick,
                turn,
            },
        );
        self.evict_overflow();
        id
    }

    fn evict_overflow(&mut self) {
        while self.entries.len() > OPEN_TURN_CAPACITY {
            let Some(id) = self.entries.keys().next().copied() else {
                break;
            };
            self.entries.remove(&id);
        }
    }
}

fn truncate_chars(text: &str, limit: usize) -> String {
    let mut out: String = text.chars().take(limit).collect();
    if text.chars().count() > limit {
        out.push('…');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enforce_deadline_names_the_phase() {
        let started = Instant::now()
            .checked_sub(Duration::from_secs(5))
            .expect("测试起点");
        let mut control = TurnControl::starting_at(
            1,
            TurnBudget::enforced(Duration::from_millis(10)),
            started,
        );
        let fault = control.enter(TurnPhase::Reason).expect_err("应超时");
        assert_eq!(fault.kind, TurnFaultKind::Deadline);
        assert_eq!(fault.phase, TurnPhase::Reason);
        assert!(fault.retryable);
        assert!(fault.render().contains("推理"));
        assert!(!fault.render().contains("没有记录"));
    }

    #[test]
    fn observe_mode_records_without_rejecting() {
        let started = Instant::now()
            .checked_sub(Duration::from_secs(5))
            .expect("测试起点");
        let mut control = TurnControl::starting_at(
            2,
            TurnBudget::observed(Duration::from_millis(10)),
            started,
        );
        control.enter(TurnPhase::Recall).expect("Observe 不中止");
        assert_eq!(control.observed_overrun(), Some(TurnPhase::Recall));
    }

    #[test]
    fn cancellation_wins_over_budget_and_is_shared() {
        let mut control = TurnControl::new(3, TurnBudget::unlimited());
        let handle = control.cancel_handle();
        handle.cancel();
        let fault = control.enter(TurnPhase::Generate).expect_err("应取消");
        assert_eq!(fault.kind, TurnFaultKind::Cancelled);
        assert_eq!(fault.phase, TurnPhase::Generate);
        assert!(fault.retryable);
    }

    #[test]
    fn open_turn_retries_are_bounded_and_ttl_expires() {
        let mut book = OpenTurnBook::new();
        book.begin_tick();
        book.record(
            "姚明的身高是多少",
            1,
            TurnPhase::Reason,
            TurnFaultKind::Deadline,
            "超时",
            Some("姚明"),
            Some("身高"),
        );
        let first = book.take_latest().expect("应有未完成轮次");
        assert_eq!(first.attempts, 1);
        assert!(first.can_resume(book.tick()));

        book.record_attempts(
            &first.input,
            2,
            TurnPhase::Reason,
            TurnFaultKind::Deadline,
            "超时",
            Some("姚明"),
            Some("身高"),
            first.attempts + 1,
        );
        let latest = book.latest().expect("重试失败后应重建");
        assert_eq!(latest.attempts, 2);

        book.begin_tick();
        book.begin_tick();
        book.begin_tick();
        assert!(book.is_empty(), "TTL 后必须自动过期");
    }

    #[test]
    fn open_turn_capacity_keeps_newest_entries() {
        let mut book = OpenTurnBook::new();
        for index in 0..(OPEN_TURN_CAPACITY + 3) {
            book.record(
                &format!("问题{}", index),
                1,
                TurnPhase::Reason,
                TurnFaultKind::Store,
                "",
                None,
                None,
            );
        }
        assert_eq!(book.len(), OPEN_TURN_CAPACITY);
        assert_eq!(
            book.entries().next().map(|open| open.input.as_str()),
            Some("问题3")
        );
        let removed = book.dismiss_latest().expect("应可放弃");
        assert_eq!(removed.input, format!("问题{}", OPEN_TURN_CAPACITY + 2));
    }

    #[test]
    fn open_turn_roundtrips_through_json() {
        let mut book = OpenTurnBook::new();
        book.begin_tick();
        book.record(
            "李四的老师是谁",
            7,
            TurnPhase::Generate,
            TurnFaultKind::Panic,
            "boom",
            Some("李四"),
            Some("老师"),
        );
        let json = serde_json::to_string(&book).expect("应可序列化");
        let restored: OpenTurnBook = serde_json::from_str(&json).expect("应可反序列化");
        assert_eq!(restored, book);
        assert_eq!(
            restored.latest().map(|open| open.fault),
            Some(TurnFaultKind::Panic)
        );
    }

    #[test]
    fn engine_fault_never_renders_as_unknown() {
        let fault = TurnFault::new(
            TurnFaultKind::Prolog,
            TurnPhase::Reason,
            "consult failed",
            12,
        );
        let text = fault.render();
        assert!(text.contains("推理引擎"));
        assert!(!text.starts_with("我的知识图谱里暂时没有"));
        assert_ne!(
            text,
            crate::epistemic::render_statement(
                crate::epistemic::EpistemicStatus::Unknown,
                Some("姚明"),
                Some("身高"),
                None,
            )
        );
    }
}

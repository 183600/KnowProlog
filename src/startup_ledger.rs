//! 启动分期台账（Startup Phase Ledger）。
//!
//! 符号系统的成本结构与 LLM 的「首 token prefill」同构：进程启动要扫全图、
//! 建派生索引，每个新会话还要把事实重放一遍。本台账把这三处成本显式分期：
//!   - 进程启动：`config_load` → `store_open` → `facts_load` → `warm_build` → `bind`；
//!   - 会话创建：`session_fork`（从预热基座分叉到就绪）；
//!   - 冷会话换入：`session_resume_read`（磁盘信封读取 + 反序列化）。
//!
//! 设计约定：
//!   - 计时只观测、不断言：测试只检查「相名存在、计数、单调累计」等结构事实，
//!     绝不写毫秒硬阈值（CI 机器抖动不属于回归）；
//!   - 报告确定性：相按名称字典序输出，同名相合并（`count`/`total_micros`/`max_micros`）；
//!   - 微秒整数记账，避免浮点尾差；零新依赖。

use std::sync::Arc;
use std::time::Instant;

use parking_lot::Mutex;
use serde::Serialize;

/// 单个分期的聚合统计（同名分期多次发生时累计）。
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct PhaseStats {
    pub phase: String,
    pub count: u64,
    pub total_micros: u64,
    pub max_micros: u64,
}

impl PhaseStats {
    /// 人类可读的单相摘要：`1.234ms`（count==1）或 `1.234ms x3`。
    pub fn display(&self) -> String {
        let millis = self.total_micros / self.count.max(1);
        let base = format!("{}.{:03}ms", millis / 1000, millis % 1000);
        if self.count > 1 {
            format!("{} x{}", base, self.count)
        } else {
            base
        }
    }
}

/// 线程安全的分期台账（`Arc` 共享：启动线程与会话线程写入同一实例）。
#[derive(Debug, Default)]
pub struct StartupLedger {
    phases: Mutex<Vec<PhaseStats>>,
}

impl StartupLedger {
    pub fn new() -> Self {
        Self::default()
    }

    /// 便于共享：`let startup = StartupLedger::shared();`
    pub fn shared() -> Arc<Self> {
        Arc::new(Self::new())
    }

    /// 记一笔分期耗时（微秒）。同名分期累计，保留最大值。
    pub fn record(&self, phase: &str, micros: u64) {
        let mut phases = self.phases.lock();
        if let Some(entry) = phases.iter_mut().find(|entry| entry.phase == phase) {
            entry.count += 1;
            entry.total_micros = entry.total_micros.saturating_add(micros);
            entry.max_micros = entry.max_micros.max(micros);
        } else {
            phases.push(PhaseStats {
                phase: phase.to_string(),
                count: 1,
                total_micros: micros,
                max_micros: micros,
            });
        }
    }

    /// 记一笔「从 `start` 到现在」的分期耗时（跨 await 时手动调用）。
    pub fn record_since(&self, phase: &str, start: Instant) {
        self.record(phase, elapsed_micros(start));
    }

    /// 同步区间计时：guard 析构时自动记账。
    pub fn span(&self, phase: &'static str) -> StartupSpan<'_> {
        StartupSpan {
            ledger: self,
            phase,
            start: Instant::now(),
        }
    }

    /// 相名的字典序快照（报告确定性的来源）。
    pub fn snapshot(&self) -> Vec<PhaseStats> {
        let mut phases = self.phases.lock().clone();
        phases.sort_by(|a, b| a.phase.cmp(&b.phase));
        phases
    }

    /// 单行摘要：`启动分期: bind=0.123ms config_load=1.234ms ...`；
    /// 台账为空时返回 None（调用方决定是否打印）。
    pub fn summary_line(&self) -> Option<String> {
        let phases = self.snapshot();
        if phases.is_empty() {
            return None;
        }
        let body = phases
            .iter()
            .map(|stats| format!("{}={}", stats.phase, stats.display()))
            .collect::<Vec<_>>()
            .join(" ");
        Some(format!("启动分期: {}", body))
    }

    /// 多行明细（`/metrics` 之外的人工审计面）。
    pub fn report(&self) -> String {
        match self.summary_line() {
            Some(line) => line,
            None => "启动分期：暂无记录。".to_string(),
        }
    }
}

/// `Instant` 差值转微秒（饱和到 `u64`，极端久挂也不回绕）。
fn elapsed_micros(start: Instant) -> u64 {
    start.elapsed().as_micros().min(u64::MAX as u128) as u64
}

/// 区间计时守卫：Drop 时把 `Instant` 差值记入台账。
pub struct StartupSpan<'a> {
    ledger: &'a StartupLedger,
    phase: &'static str,
    start: Instant,
}

impl Drop for StartupSpan<'_> {
    fn drop(&mut self) {
        self.ledger.record(self.phase, elapsed_micros(self.start));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phases_aggregate_and_report_in_name_order() {
        let ledger = StartupLedger::new();
        ledger.record("facts_load", 2_000);
        ledger.record("bind", 500);
        ledger.record("facts_load", 3_000);
        let snapshot = ledger.snapshot();
        assert_eq!(snapshot.len(), 2, "同名相必须合并");
        assert_eq!(snapshot[0].phase, "bind", "报告按相名字典序");
        assert_eq!(snapshot[1].phase, "facts_load");
        assert_eq!(snapshot[1].count, 2);
        assert_eq!(snapshot[1].total_micros, 5_000);
        assert_eq!(snapshot[1].max_micros, 3_000);

        let line = ledger.summary_line().expect("有记录必有摘要");
        assert!(line.starts_with("启动分期: "));
        let bind = line.find("bind=").unwrap();
        let facts = line.find("facts_load=").unwrap();
        assert!(bind < facts, "摘要与明细同序：{}", line);
        assert!(line.contains("facts_load=2.500ms x2"), "{}", line);
    }

    #[test]
    fn span_records_on_drop_and_empty_ledger_is_quiet() {
        let ledger = StartupLedger::new();
        assert!(ledger.summary_line().is_none());
        {
            let _span = ledger.span("warm_build");
        }
        let stats = ledger.snapshot();
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].phase, "warm_build");
        assert_eq!(stats[0].count, 1);
    }

    #[test]
    fn display_formats_micros_as_millis() {
        let stats = PhaseStats {
            phase: "x".to_string(),
            count: 1,
            total_micros: 1_234,
            max_micros: 1_234,
        };
        assert_eq!(stats.display(), "1.234ms");
        let stats = PhaseStats {
            phase: "x".to_string(),
            count: 3,
            total_micros: 3_000,
            max_micros: 1_500,
        };
        assert_eq!(stats.display(), "1.000ms x3");
    }
}

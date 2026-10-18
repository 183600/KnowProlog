//! PPL：操作计数台账（Perf Ledger）。
//!
//! 记录确定性的操作计数（不依赖挂钟，避免 CI 机器抖动）——
//! 观察次数、各类重建次数、缓存命中/未命中、图扫描次数、序列化字节数……
//! 计数可跨模块共享（`Arc<PerfLedger>`），可序列化（BTreeMap 序），可逐字节 diff。
//!
//! 设计约定：
//!   - 键是稳定的 ASCII 标识符（`graph_snapshot_scan` 等），报告按字典序输出；
//!   - 计数只增不减（除非显式 `reset`），因此测试可以取样差值做预算断言；
//!   - 零新依赖，`parking_lot::Mutex` 提供内部可变性，任意 `&self` 可记账。

use std::collections::BTreeMap;
use std::sync::Arc;

use parking_lot::Mutex;

/// 线程安全、可共享的操作计数台账。
#[derive(Debug, Default)]
pub struct PerfLedger {
    counters: Mutex<BTreeMap<String, u64>>,
}

impl PerfLedger {
    pub fn new() -> Self {
        Self::default()
    }

    /// 便于构造共享台账：`let perf = PerfLedger::shared();`
    pub fn shared() -> Arc<Self> {
        Arc::new(Self::new())
    }

    /// 计数 +1。
    pub fn bump(&self, key: &str) {
        self.bump_by(key, 1);
    }

    /// 计数 +delta（delta 为 0 时不落键，保证空报告干净）。
    pub fn bump_by(&self, key: &str, delta: u64) {
        if delta == 0 {
            return;
        }
        *self.counters.lock().entry(key.to_string()).or_insert(0) += delta;
    }

    /// 当前计数（键不存在为 0）。
    pub fn count(&self, key: &str) -> u64 {
        self.counters.lock().get(key).copied().unwrap_or(0)
    }

    /// 计数快照：BTreeMap 字典序，天然确定。
    pub fn snapshot(&self) -> BTreeMap<String, u64> {
        self.counters.lock().clone()
    }

    pub fn is_empty(&self) -> bool {
        self.counters.lock().is_empty()
    }

    /// 清零（供测试隔离使用；生产路径不应调用）。
    pub fn reset(&self) {
        self.counters.lock().clear();
    }

    /// 合并另一份快照（同键相加）。
    pub fn merge_snapshot(&self, other: &BTreeMap<String, u64>) {
        let mut counters = self.counters.lock();
        for (key, value) in other {
            *counters.entry(key.clone()).or_insert(0) += value;
        }
    }

    /// 确定性报告：标题 + 每行 `- key: value`（字典序）。
    pub fn report_with_title(&self, title: &str) -> String {
        let counters = self.counters.lock();
        if counters.is_empty() {
            return format!("{}：暂无计数。", title);
        }
        let mut output = format!("{}：共 {} 项计数。\n", title, counters.len());
        for (key, value) in counters.iter() {
            output.push_str(&format!("- {}: {}\n", key, value));
        }
        output
    }

    pub fn report(&self) -> String {
        self.report_with_title("性能台账")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters_accumulate_and_report_deterministically() {
        let ledger = PerfLedger::new();
        ledger.bump("b_second");
        ledger.bump("a_first");
        ledger.bump_by("a_first", 2);
        ledger.bump_by("zero", 0);
        assert_eq!(ledger.count("a_first"), 3);
        assert_eq!(ledger.count("b_second"), 1);
        assert_eq!(ledger.count("missing"), 0);
        assert_eq!(ledger.count("zero"), 0, "零增量不应落键");

        let report = ledger.report();
        assert!(report.starts_with("性能台账"));
        let first = report.find("a_first").unwrap();
        let second = report.find("b_second").unwrap();
        assert!(first < second, "报告必须按字典序：{}", report);
    }

    #[test]
    fn snapshot_and_reset_are_isolated() {
        let ledger = PerfLedger::new();
        ledger.bump_by("hits", 5);
        let snapshot = ledger.snapshot();
        assert_eq!(snapshot.get("hits"), Some(&5));
        ledger.reset();
        assert!(ledger.is_empty());

        let merged = PerfLedger::new();
        merged.merge_snapshot(&snapshot);
        assert_eq!(merged.count("hits"), 5);
    }
}

//! 轮次资源账本（Resource Ledger）。
//!
//! 一轮回答同时消耗多种彼此独立的预算：上下文字符、输出字符、证据条数、
//! 图扫描次数与推理步数。过去它们散落在各模块的局部上限里，既不能相加，
//! 也无法在回答后说明「哪里用完了」。本模块提供一个只做整数加减的账本：
//!
//! - [`ResourceVector`]：同一组资源的上限或用量，`usize::MAX` 表示不限；
//! - [`ResourceLedger`]：按 [`ResourceKind`] 申请与退还预算，拒绝时留下
//!   [`ResourceDenial`]，不静默截断；
//! - [`ResourceLedger::render_receipt`]：确定性回执，同样输入同样输出。
//!
//! 账本不持有业务数据，也不依赖挂钟、随机数或浮点。调用方先申请预算，
//! 成功后再执行副作用；失败时账本保持原用量，便于重试或降级。

use serde::{Deserialize, Serialize};

/// 账本可记录的资源类别闭集。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourceKind {
    /// 送进生成/推理的上下文字符数。
    ContextChars,
    /// 最终回答的字符数。
    OutputChars,
    /// 引用的证据条数。
    EvidenceItems,
    /// 图谱扫描次数。
    GraphScans,
    /// 推理步数。
    ReasoningSteps,
}

impl ResourceKind {
    pub const ALL: [ResourceKind; 5] = [
        ResourceKind::ContextChars,
        ResourceKind::OutputChars,
        ResourceKind::EvidenceItems,
        ResourceKind::GraphScans,
        ResourceKind::ReasoningSteps,
    ];

    pub fn code(self) -> &'static str {
        match self {
            ResourceKind::ContextChars => "context_chars",
            ResourceKind::OutputChars => "output_chars",
            ResourceKind::EvidenceItems => "evidence_items",
            ResourceKind::GraphScans => "graph_scans",
            ResourceKind::ReasoningSteps => "reasoning_steps",
        }
    }

    pub fn label_zh(self) -> &'static str {
        match self {
            ResourceKind::ContextChars => "上下文字符",
            ResourceKind::OutputChars => "输出字符",
            ResourceKind::EvidenceItems => "证据条数",
            ResourceKind::GraphScans => "图扫描",
            ResourceKind::ReasoningSteps => "推理步数",
        }
    }
}

/// 一组资源的上限或用量。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceVector {
    pub context_chars: usize,
    pub output_chars: usize,
    pub evidence_items: usize,
    pub graph_scans: usize,
    pub reasoning_steps: usize,
}

impl Default for ResourceVector {
    fn default() -> Self {
        Self::zero()
    }
}

impl ResourceVector {
    pub const fn zero() -> Self {
        Self {
            context_chars: 0,
            output_chars: 0,
            evidence_items: 0,
            graph_scans: 0,
            reasoning_steps: 0,
        }
    }

    /// 所有类别都不限。
    pub const fn unlimited() -> Self {
        Self {
            context_chars: usize::MAX,
            output_chars: usize::MAX,
            evidence_items: usize::MAX,
            graph_scans: usize::MAX,
            reasoning_steps: usize::MAX,
        }
    }

    /// 只限制输出字符，其余类别不限。
    pub const fn output_only(output_chars: usize) -> Self {
        let mut limits = Self::unlimited();
        limits.output_chars = output_chars;
        limits
    }

    pub fn get(self, kind: ResourceKind) -> usize {
        match kind {
            ResourceKind::ContextChars => self.context_chars,
            ResourceKind::OutputChars => self.output_chars,
            ResourceKind::EvidenceItems => self.evidence_items,
            ResourceKind::GraphScans => self.graph_scans,
            ResourceKind::ReasoningSteps => self.reasoning_steps,
        }
    }

    pub fn set(&mut self, kind: ResourceKind, value: usize) {
        match kind {
            ResourceKind::ContextChars => self.context_chars = value,
            ResourceKind::OutputChars => self.output_chars = value,
            ResourceKind::EvidenceItems => self.evidence_items = value,
            ResourceKind::GraphScans => self.graph_scans = value,
            ResourceKind::ReasoningSteps => self.reasoning_steps = value,
        }
    }

    /// 逐项饱和相加；用于合并多个阶段的用量。
    pub fn saturating_add(self, other: Self) -> Self {
        let mut sum = Self::zero();
        for kind in ResourceKind::ALL {
            sum.set(kind, self.get(kind).saturating_add(other.get(kind)));
        }
        sum
    }

    /// 逐项饱和相减；用于退还预算。
    pub fn saturating_sub(self, other: Self) -> Self {
        let mut rest = Self::zero();
        for kind in ResourceKind::ALL {
            rest.set(kind, self.get(kind).saturating_sub(other.get(kind)));
        }
        rest
    }

    /// 用量是否不超过上限。
    pub fn fits_within(self, limits: Self) -> bool {
        ResourceKind::ALL
            .iter()
            .all(|kind| self.get(*kind) <= limits.get(*kind))
    }

    pub fn is_unlimited(self, kind: ResourceKind) -> bool {
        self.get(kind) == usize::MAX
    }
}

/// 一次被拒绝的预算申请；用量不因拒绝而改变。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceDenial {
    pub kind: ResourceKind,
    pub requested: usize,
    pub used: usize,
    pub limit: usize,
}

impl ResourceDenial {
    pub fn remaining(&self) -> usize {
        self.limit.saturating_sub(self.used)
    }

    pub fn render_zh(&self) -> String {
        if self.limit == usize::MAX {
            format!(
                "{}：申请 {}，当前不限",
                self.kind.label_zh(),
                self.requested
            )
        } else {
            format!(
                "{}：申请 {}，已用 {}/{}，剩余 {}",
                self.kind.label_zh(),
                self.requested,
                self.used,
                self.limit,
                self.remaining()
            )
        }
    }
}

/// 每轮共享的整数资源账本。
///
/// `limits` 中 `usize::MAX` 表示该类资源不限。拒绝记录按申请顺序保存，
/// 最多保留 [`ResourceLedger::MAX_DENIALS`] 条，避免异常循环撑大会话快照。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceLedger {
    limits: ResourceVector,
    used: ResourceVector,
    denials: Vec<ResourceDenial>,
}

impl ResourceLedger {
    pub const MAX_DENIALS: usize = 32;

    pub fn new(limits: ResourceVector) -> Self {
        Self {
            limits,
            used: ResourceVector::zero(),
            denials: Vec::new(),
        }
    }

    pub fn limits(&self) -> ResourceVector {
        self.limits
    }

    pub fn used(&self) -> ResourceVector {
        self.used
    }

    pub fn denials(&self) -> &[ResourceDenial] {
        &self.denials
    }

    pub fn remaining(&self, kind: ResourceKind) -> usize {
        let limit = self.limits.get(kind);
        if limit == usize::MAX {
            usize::MAX
        } else {
            limit.saturating_sub(self.used.get(kind))
        }
    }

    pub fn used_of(&self, kind: ResourceKind) -> usize {
        self.used.get(kind)
    }

    pub fn exhausted(&self, kind: ResourceKind) -> bool {
        self.remaining(kind) == 0
    }

    /// 申请预算；成功才累加用量，失败留下显式拒绝记录。
    pub fn try_charge(&mut self, kind: ResourceKind, amount: usize) -> Result<(), ResourceDenial> {
        let used = self.used.get(kind);
        let limit = self.limits.get(kind);
        let next = used.saturating_add(amount);
        if next > limit {
            let denial = ResourceDenial {
                kind,
                requested: amount,
                used,
                limit,
            };
            if self.denials.len() < Self::MAX_DENIALS {
                self.denials.push(denial.clone());
            }
            return Err(denial);
        }
        self.used.set(kind, next);
        Ok(())
    }

    /// 退还预算；不会把用量减到 0 以下。
    pub fn release(&mut self, kind: ResourceKind, amount: usize) {
        let used = self.used.get(kind);
        self.used.set(kind, used.saturating_sub(amount));
    }

    /// 是否仍在所有上限之内。
    pub fn fits_within_limits(&self) -> bool {
        self.used.fits_within(self.limits)
    }

    /// 确定性回执：固定按 [`ResourceKind::ALL`] 的顺序渲染。
    pub fn render_receipt(&self) -> String {
        let rows = ResourceKind::ALL
            .iter()
            .map(|kind| {
                let used = self.used.get(*kind);
                let limit = self.limits.get(*kind);
                if limit == usize::MAX {
                    format!("{} {}（不限）", kind.label_zh(), used)
                } else {
                    let remaining = limit.saturating_sub(used);
                    format!(
                        "{} {}/{}（剩余 {}）",
                        kind.label_zh(),
                        used,
                        limit,
                        remaining
                    )
                }
            })
            .collect::<Vec<_>>()
            .join("；");
        if self.denials.is_empty() {
            format!("资源账本：{}。", rows)
        } else {
            let denied = self
                .denials
                .iter()
                .map(ResourceDenial::render_zh)
                .collect::<Vec<_>>()
                .join("；");
            format!("资源账本：{}。拒绝：{}。", rows, denied)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits() -> ResourceVector {
        ResourceVector {
            context_chars: 120,
            output_chars: 40,
            evidence_items: 2,
            graph_scans: 1,
            reasoning_steps: 3,
        }
    }

    #[test]
    fn charges_accumulate_without_crossing_the_limit() {
        let mut ledger = ResourceLedger::new(limits());
        assert!(ledger.try_charge(ResourceKind::OutputChars, 30).is_ok());
        assert_eq!(ledger.remaining(ResourceKind::OutputChars), 10);
        let denial = ledger
            .try_charge(ResourceKind::OutputChars, 11)
            .unwrap_err();
        assert_eq!(denial.requested, 11);
        assert_eq!(denial.used, 30);
        assert_eq!(denial.remaining(), 10);
        assert_eq!(ledger.used_of(ResourceKind::OutputChars), 30);
        assert_eq!(ledger.denials().len(), 1);
        assert!(ledger.fits_within_limits());
    }

    #[test]
    fn release_returns_budget_but_never_below_zero() {
        let mut ledger = ResourceLedger::new(limits());
        ledger.try_charge(ResourceKind::ReasoningSteps, 2).unwrap();
        ledger.release(ResourceKind::ReasoningSteps, 1);
        assert_eq!(ledger.used_of(ResourceKind::ReasoningSteps), 1);
        ledger.release(ResourceKind::ReasoningSteps, 9);
        assert_eq!(ledger.used_of(ResourceKind::ReasoningSteps), 0);
        assert_eq!(ledger.remaining(ResourceKind::ReasoningSteps), 3);
    }

    #[test]
    fn unlimited_resources_accept_any_charge() {
        let mut ledger = ResourceLedger::new(ResourceVector::output_only(5));
        ledger
            .try_charge(ResourceKind::ContextChars, usize::MAX)
            .unwrap();
        assert_eq!(ledger.remaining(ResourceKind::ContextChars), usize::MAX);
        assert!(ledger.render_receipt().contains("上下文字符"));
        assert!(ledger.render_receipt().contains("不限"));
        assert!(ledger.try_charge(ResourceKind::OutputChars, 6).is_err());
    }

    #[test]
    fn receipt_is_deterministic_and_lists_denials_in_order() {
        let mut ledger = ResourceLedger::new(limits());
        ledger
            .try_charge(ResourceKind::OutputChars, 41)
            .unwrap_err();
        ledger.try_charge(ResourceKind::GraphScans, 2).unwrap_err();
        let first = ledger.render_receipt();
        for _ in 0..8 {
            assert_eq!(first, ledger.render_receipt());
        }
        assert!(first.contains("输出字符：申请 41"));
        assert!(first.contains("图扫描：申请 2"));
        let output_at = first.find("输出字符").unwrap();
        let graph_at = first.find("图扫描：申请").unwrap();
        assert!(output_at < graph_at, "拒绝记录必须保持申请顺序");
    }

    #[test]
    fn denial_log_is_bounded_but_usage_stays_unchanged() {
        let mut ledger = ResourceLedger::new(ResourceVector::zero());
        for _ in 0..(ResourceLedger::MAX_DENIALS + 5) {
            assert!(ledger.try_charge(ResourceKind::OutputChars, 1).is_err());
        }
        assert_eq!(ledger.denials().len(), ResourceLedger::MAX_DENIALS);
        assert_eq!(ledger.used(), ResourceVector::zero());
    }
}

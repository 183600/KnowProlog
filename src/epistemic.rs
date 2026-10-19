//! EPI：认识论状态与失败回执（Epistemic Failure Accounting）。
//!
//! 系统里「不知道」长期是一个字符串（`cannot_understand` / `没有记录`），
//! 而真实的认识论状态至少有七种：确证 / 推测 / 冲突 / 确无 / 查无 /
//! 没解析 / 引擎故障。本模块把「失败」变成与正常回答同构的第一类对象：
//!
//! - [`EpistemicStatus`]：结论的确定性分类；**EngineFault 绝不允许伪装成
//!   Unknown**（「我出错了」与「图谱里没有」是两件事）；
//! - [`FailureReceipt`]：一次失败的结构化回执（状态 + 实体/属性 + 恢复档位 +
//!   证据 + 待确认问题 + 用户裁定），可上屏、可入台账、可被元对话逐字检索；
//! - [`FailureLedger`]：有界 FIFO（默认 64）、确定性序、规范 JSON + FNV-1a
//!   digest、`ContextImage` 信封原子落盘（损坏整体拒绝，绝不按坏账自省）；
//! - [`render_statement`]：认识论状态 → 措辞的固定模板（槽位绑定，不生成）。
//!
//! 失败回执不写图谱、不改导入语义——它只影响「怎么说」与「追问什么」。
//! 自省不是生成，是查询：`why_no_evidence` 会逐字渲染这里的最新回执。

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::context_image::{canonical_json, fnv1a_hex, ContextImage};

/// 失败台账容量上限（FIFO，超出丢弃最旧）。
pub const LEDGER_CAPACITY: usize = 64;
/// 单条回执 detail 的字符上限（防止异常信息把台账撑爆）。
pub const DETAIL_LIMIT: usize = 200;
/// 回执正文展示条数上限。
pub const RECEIPT_DISPLAY_CAP: usize = 8;

/// 台账落盘分节名。
pub const LEDGER_SECTION: &str = "failure_ledger";

/// 一条结论的认识论状态。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EpistemicStatus {
    /// 符号通道证实（正常回答）。
    Certain,
    /// 联想通道提名、符号通道未证实：只能以推测语气输出。
    Provisional,
    /// 两条通道给出矛盾说法：披露争议，不替用户拍板。
    Conflict,
    /// 实体已知，但确证没有该属性（缺的是事实，不是检索）。
    Absent,
    /// 查无记录（实体或属性无从谈起）。
    Unknown,
    /// 帧解析 / 形式化失败：问题没被理解成可推理的形式。
    Unparseable,
    /// 引擎层故障（Prolog / 存储 / 策略求值）：认识论之外，绝不与 Unknown 合并。
    EngineFault,
}

impl EpistemicStatus {
    /// ASCII 代码（API / 回执 / 基线用，稳定）。
    pub fn code(self) -> &'static str {
        match self {
            EpistemicStatus::Certain => "certain",
            EpistemicStatus::Provisional => "provisional",
            EpistemicStatus::Conflict => "conflict",
            EpistemicStatus::Absent => "absent",
            EpistemicStatus::Unknown => "unknown",
            EpistemicStatus::Unparseable => "unparseable",
            EpistemicStatus::EngineFault => "engine_fault",
        }
    }

    /// 中文短名（报告用）。
    pub fn label(self) -> &'static str {
        match self {
            EpistemicStatus::Certain => "确证",
            EpistemicStatus::Provisional => "推测",
            EpistemicStatus::Conflict => "冲突",
            EpistemicStatus::Absent => "确无",
            EpistemicStatus::Unknown => "查无",
            EpistemicStatus::Unparseable => "未解析",
            EpistemicStatus::EngineFault => "引擎故障",
        }
    }

    /// 非确证即失败（需要回执）。
    pub fn is_failure(self) -> bool {
        !matches!(self, EpistemicStatus::Certain)
    }

    pub fn is_engine_fault(self) -> bool {
        matches!(self, EpistemicStatus::EngineFault)
    }
}

/// 认识论状态 → 措辞的固定模板（每个词要么是脚手架，要么是绑定值）。
///
/// 硬约束：EngineFault 的文案绝不出现「没有记录」——故障不得伪装成不知道；
/// Absent 与 Unknown 的差异必须可见（「确定没有」vs「暂无记录」）。
pub fn render_statement(
    status: EpistemicStatus,
    entity: Option<&str>,
    attribute: Option<&str>,
    value: Option<&str>,
) -> String {
    let entity = entity.unwrap_or("");
    let attribute = attribute.unwrap_or("");
    let value = value.unwrap_or("");
    match status {
        EpistemicStatus::Certain => value.to_string(),
        EpistemicStatus::Provisional => match (entity.is_empty(), attribute.is_empty()) {
            (false, false) => format!(
                "关于「{}」的「{}」，我只有推测：{}（还没证实，请以你的信息为准）。",
                entity, attribute, value
            ),
            (false, true) => format!(
                "关于「{}」，我只有个推测：{}（还没证实，请以你的信息为准）。",
                entity, value
            ),
            _ => format!("我只有个推测：{}（还没证实，请以你的信息为准）。", value),
        },
        EpistemicStatus::Conflict => match (entity.is_empty(), attribute.is_empty()) {
            (false, false) => format!(
                "关于「{}」的「{}」，我听到了互相矛盾的说法：{}。我不替你拍板，请核实来源。",
                entity, attribute, value
            ),
            _ => format!("关于这件事，我听到了互相矛盾的说法：{}。需要你帮忙核实。", value),
        },
        EpistemicStatus::Absent => match (entity.is_empty(), attribute.is_empty()) {
            (false, false) => format!(
                "我确认「{}」的「{}」在图谱里没有记录——是确定没有，不是没查到。",
                entity, attribute
            ),
            (false, true) => format!(
                "我确认图谱里没有「{}」的可介绍信息——是确定没有，不是没查到。",
                entity
            ),
            _ => "我确认图谱里没有这条记录——是确定没有，不是没查到。".to_string(),
        },
        EpistemicStatus::Unknown => match (entity.is_empty(), attribute.is_empty()) {
            (false, false) => format!(
                "我的知识图谱里暂时没有关于「{}」的「{}」的记录。",
                entity, attribute
            ),
            (false, true) => format!("我的知识图谱里暂时没有关于「{}」的记录。", entity),
            _ => "我的知识图谱里暂时没有这条记录。".to_string(),
        },
        EpistemicStatus::Unparseable => {
            "我没能把这句话解析成可推理的问题，换个说法再问一次吧。".to_string()
        }
        EpistemicStatus::EngineFault => {
            "我这边刚出错了（已记录），不是知识图谱里没有——请稍后再试。".to_string()
        }
    }
}

/// 用户对回执的裁定。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReceiptResolution {
    /// 推测被确认。
    Confirmed,
    /// 推测被否定。
    Refuted,
    /// 澄清/线索已被接收（不作真假判定）。
    Acknowledged,
}

impl ReceiptResolution {
    pub fn label(self) -> &'static str {
        match self {
            ReceiptResolution::Confirmed => "已确认",
            ReceiptResolution::Refuted => "已否认",
            ReceiptResolution::Acknowledged => "已接收",
        }
    }
}

/// 一条失败回执（结构化、可引用、可裁定）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FailureReceipt {
    /// 台账内唯一 id（`r1` 起，与持久化水位对应）。
    pub id: String,
    /// 发生时的对话/会话轮次水位。
    pub turn: u64,
    pub status: EpistemicStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entity: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attribute: Option<String>,
    /// 恢复阶梯档位（`L0`–`L4`）。
    pub level: String,
    /// 失败细节（有界；引擎故障时是错误摘要）。
    #[serde(default)]
    pub detail: String,
    /// 证据链（推测/冲突时非空；逐字可回溯）。
    #[serde(default)]
    pub evidence: Vec<String>,
    /// 挂账的追问（至多一个；确认/否认后关闭）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub question: Option<String>,
    /// 用户裁定（None = 未结）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolution: Option<ReceiptResolution>,
}

impl FailureReceipt {
    /// 尚未被用户裁定的回执。
    pub fn is_open(&self) -> bool {
        self.resolution.is_none()
    }

    /// 身份键：同一 (状态, 实体, 属性) 的未结回执不重复入账（防刷屏）。
    pub fn identity(&self) -> (EpistemicStatus, Option<String>, Option<String>) {
        (self.status, self.entity.clone(), self.attribute.clone())
    }

    /// 主题短语（自省 / 报告用）。
    pub fn topic(&self) -> String {
        match (&self.entity, &self.attribute) {
            (Some(entity), Some(attribute)) => format!("「{}」的「{}」", entity, attribute),
            (Some(entity), None) => format!("「{}」", entity),
            _ => "这次提问".to_string(),
        }
    }

    /// 单条回执的单行渲染（确定性）。
    pub fn render_line(&self) -> String {
        let mut line = format!(
            "#{} [{}] {}：{}",
            self.id,
            self.status.code(),
            self.topic(),
            self.status.label()
        );
        if !self.detail.is_empty() {
            line.push_str(&format!("（{}）", self.detail.replace('\n', " ")));
        }
        if !self.evidence.is_empty() {
            line.push_str(&format!("；证据：{}", self.evidence.join("；")));
        }
        if let Some(question) = &self.question {
            line.push_str(&format!("；追问：{}", question));
        }
        if let Some(resolution) = self.resolution {
            line.push_str(&format!("；裁定：{}", resolution.label()));
        }
        line
    }
}

/// 有界 FIFO 失败台账（确定性序、digest 锁定、信封落盘）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FailureLedger {
    #[serde(default)]
    receipts: Vec<FailureReceipt>,
    /// 持久化水位：下一个回执 id 从这里递增，跨重启不重号。
    #[serde(default)]
    next_id: u64,
}

impl Default for FailureLedger {
    fn default() -> Self {
        Self::new()
    }
}

/// 台账加载结果：缺失 / 已加载（条数）/ 损坏。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LedgerLoad {
    Missing,
    Loaded(usize),
    Damaged(String),
}

impl FailureLedger {
    pub fn new() -> Self {
        FailureLedger {
            receipts: Vec::new(),
            next_id: 1,
        }
    }

    /// 记录一条回执：自动分配 id、按容量 FIFO；同身份未结回执只保留一条。
    ///
    /// 返回回执 id。相同 (状态, 实体, 属性) 的未结回执再次出现时，不重复入账，
    /// 返回已有 id（引擎/存储故障路径只报一次，不刷屏）。
    pub fn record(&mut self, mut receipt: FailureReceipt) -> String {
        let identity = receipt.identity();
        if let Some(existing) = self
            .receipts
            .iter()
            .find(|candidate| candidate.is_open() && candidate.identity() == identity)
        {
            return existing.id.clone();
        }
        receipt.id = format!("r{}", self.next_id);
        self.next_id += 1;
        if receipt.detail.chars().count() > DETAIL_LIMIT {
            receipt.detail = receipt.detail.chars().take(DETAIL_LIMIT).collect();
        }
        self.receipts.push(receipt);
        while self.receipts.len() > LEDGER_CAPACITY {
            self.receipts.remove(0);
        }
        self.receipts.last().map(|r| r.id.clone()).unwrap_or_default()
    }

    pub fn len(&self) -> usize {
        self.receipts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.receipts.is_empty()
    }

    pub fn receipts(&self) -> &[FailureReceipt] {
        &self.receipts
    }

    pub fn last(&self) -> Option<&FailureReceipt> {
        self.receipts.last()
    }

    pub fn get(&self, id: &str) -> Option<&FailureReceipt> {
        self.receipts.iter().find(|receipt| receipt.id == id)
    }

    /// 未结回执（新 → 旧）。
    pub fn open(&self) -> impl Iterator<Item = &FailureReceipt> {
        self.receipts.iter().rev().filter(|receipt| receipt.is_open())
    }

    pub fn open_count(&self) -> usize {
        self.open().count()
    }

    /// 裁定一条回执；返回是否存在且此前未结。
    pub fn resolve(&mut self, id: &str, resolution: ReceiptResolution) -> bool {
        let Some(receipt) = self
            .receipts
            .iter_mut()
            .find(|receipt| receipt.id == id)
        else {
            return false;
        };
        if receipt.resolution.is_some() {
            return false;
        }
        receipt.resolution = Some(resolution);
        true
    }

    /// 按身份找最近的未结回执（供确认/否认挂账匹配）。
    pub fn open_matching(
        &self,
        entity: Option<&str>,
        attribute: Option<&str>,
    ) -> Option<&FailureReceipt> {
        self.receipts
            .iter()
            .rev()
            .find(|receipt| receipt.is_open())
            .filter(|receipt| {
                (entity.is_none() || receipt.entity.as_deref() == entity)
                    && (attribute.is_none() || receipt.attribute.as_deref() == attribute)
            })
    }

    /// 统计各状态的失败分布（确定性序：状态升序）。
    pub fn status_counts(&self) -> Vec<(EpistemicStatus, usize)> {
        let mut counts: std::collections::BTreeMap<EpistemicStatus, usize> =
            std::collections::BTreeMap::new();
        for receipt in &self.receipts {
            *counts.entry(receipt.status).or_insert(0) += 1;
        }
        counts.into_iter().collect()
    }

    /// 规范内容摘要（费用：规范 JSON + FNV-1a 64）。
    pub fn digest(&self) -> String {
        let value = serde_json::to_value(self).unwrap_or(Value::Null);
        fnv1a_hex(canonical_json(&value).as_bytes())
    }

    /// 台账正文（新 → 旧，确定性）。
    pub fn render_text(&self) -> String {
        if self.receipts.is_empty() {
            return "失败台账：空（暂无结构化失败回执）。".to_string();
        }
        let mut lines = vec![format!(
            "失败台账：{} 条回执（未结 {}），摘要 {}。",
            self.receipts.len(),
            self.open_count(),
            self.digest()
        )];
        let distribution: Vec<String> = self
            .status_counts()
            .into_iter()
            .map(|(status, count)| format!("{}×{}", status.label(), count))
            .collect();
        lines.push(format!("分布：{}", distribution.join("、")));
        for receipt in self.receipts.iter().rev().take(RECEIPT_DISPLAY_CAP) {
            lines.push(format!("  • {}", receipt.render_line()));
        }
        if self.receipts.len() > RECEIPT_DISPLAY_CAP {
            lines.push(format!(
                "  ……其余 {} 条省略",
                self.receipts.len() - RECEIPT_DISPLAY_CAP
            ));
        }
        lines.join("\n")
    }

    /// `ContextImage` 信封原子落盘（分节自检；损坏整体拒绝）。
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let value = serde_json::to_value(self)
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::Other, error))?;
        let mut image = ContextImage::new();
        image.set_section(LEDGER_SECTION, value);
        image.set_section("ledger_meta", Value::from(self.receipts.len() as u64));
        image.seal();
        image.save_atomic(path)
    }

    /// 读取台账：缺失 / 已加载 / 损坏（损坏时调用方以空台账安全降级）。
    pub fn load(path: &str) -> LedgerLoad {
        Self::load_full(path).0
    }

    /// 读取台账本体：返回（加载结果, 台账）。仅 `Loaded` 时给出台账内容，
    /// 且已做过容量裁剪与水位恢复。
    pub fn load_full(path: &str) -> (LedgerLoad, Option<FailureLedger>) {
        if !std::path::Path::new(path).exists() {
            return (LedgerLoad::Missing, None);
        }
        let image = match ContextImage::load(path) {
            Ok(image) => image,
            Err(error) => return (LedgerLoad::Damaged(error), None),
        };
        let audit = image.verify();
        if audit.global_damaged {
            return (LedgerLoad::Damaged("台账全局哈希不匹配".to_string()), None);
        }
        if audit
            .damaged_sections
            .iter()
            .any(|section| section == LEDGER_SECTION)
        {
            return (LedgerLoad::Damaged("台账分节哈希不匹配".to_string()), None);
        }
        let Some(value) = image.section(LEDGER_SECTION) else {
            return (LedgerLoad::Loaded(0), Some(FailureLedger::new()));
        };
        match serde_json::from_value::<FailureLedger>(value.clone()) {
            Ok(ledger) => {
                let restored = FailureLedger::restore_from(ledger);
                let count = restored.len();
                (LedgerLoad::Loaded(count), Some(restored))
            }
            Err(error) => (
                LedgerLoad::Damaged(format!("台账解析失败：{}", error)),
                None,
            ),
        }
    }

    /// 用磁盘内容替换当前台账（调用方在 `Loaded` 时使用）。
    pub fn restore_from(value: FailureLedger) -> Self {
        let mut ledger = value;
        while ledger.receipts.len() > LEDGER_CAPACITY {
            ledger.receipts.remove(0);
        }
        if ledger.next_id == 0 {
            ledger.next_id = 1;
        }
        ledger
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn receipt(
        status: EpistemicStatus,
        entity: Option<&str>,
        attribute: Option<&str>,
    ) -> FailureReceipt {
        FailureReceipt {
            id: String::new(),
            turn: 1,
            status,
            entity: entity.map(str::to_string),
            attribute: attribute.map(str::to_string),
            level: "L4".to_string(),
            detail: String::new(),
            evidence: Vec::new(),
            question: None,
            resolution: None,
        }
    }

    #[test]
    fn engine_fault_never_masquerades_as_unknown() {
        let fault = render_statement(EpistemicStatus::EngineFault, Some("姚明"), Some("身高"), None);
        assert!(
            !fault.contains("没有记录"),
            "引擎故障绝不能伪装成不知道：{}",
            fault
        );
        let absent = render_statement(EpistemicStatus::Absent, Some("姚明"), Some("体重"), None);
        assert!(absent.contains("没有记录") && absent.contains("确定没有"));
        let unknown = render_statement(EpistemicStatus::Unknown, Some("姚明"), Some("体重"), None);
        assert!(unknown.contains("暂时") && unknown.contains("没有"));
        assert_ne!(absent, unknown);
    }

    #[test]
    fn ledger_ids_are_monotonic_and_bounded_fifo() {
        let mut ledger = FailureLedger::new();
        let first = ledger.record(receipt(EpistemicStatus::Absent, Some("A"), Some("x")));
        let second = ledger.record(receipt(EpistemicStatus::Unknown, Some("B"), Some("y")));
        assert_eq!(first, "r1");
        assert_eq!(second, "r2");
        assert_eq!(ledger.len(), 2);

        // 同身份未结回执只入账一次
        let again = ledger.record(receipt(EpistemicStatus::Absent, Some("A"), Some("x")));
        assert_eq!(again, "r1");
        assert_eq!(ledger.len(), 2);

        // 容量：FIFO 丢最旧
        for index in 0..LEDGER_CAPACITY {
            ledger.record(receipt(
                EpistemicStatus::Unparseable,
                Some(&format!("E{}", index)),
                None,
            ));
        }
        assert_eq!(ledger.len(), LEDGER_CAPACITY);
        assert!(ledger.get("r1").is_none() && ledger.get("r2").is_none(), "最旧回执应被淘汰");
        assert!(ledger.get("r3").is_some());
    }

    #[test]
    fn resolve_closes_receipt_and_digest_changes() {
        let mut ledger = FailureLedger::new();
        let id = ledger.record(receipt(EpistemicStatus::Provisional, Some("姚明"), Some("体重")));
        let before = ledger.digest();
        assert_eq!(ledger.open_count(), 1);
        assert!(ledger.resolve(&id, ReceiptResolution::Refuted));
        assert_eq!(ledger.open_count(), 0);
        assert!(!ledger.resolve(&id, ReceiptResolution::Confirmed), "重复裁定应被拒绝");
        assert_ne!(before, ledger.digest(), "裁定必须改变摘要");
        assert!(ledger.render_text().contains("已否认"));
    }

    #[test]
    fn determinism_same_bytes_and_digest() {
        let build = || {
            let mut ledger = FailureLedger::new();
            ledger.record(receipt(EpistemicStatus::Absent, Some("A"), Some("x")));
            ledger.record(receipt(EpistemicStatus::EngineFault, Some("B"), None));
            ledger
        };
        let left = build();
        let right = build();
        assert_eq!(left.digest(), right.digest());
        assert_eq!(left.render_text(), right.render_text());
    }

    #[test]
    fn ledger_roundtrip_and_tamper_rejection() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("failure_ledger.json");
        let path_text = path.to_str().unwrap();

        assert!(matches!(FailureLedger::load(path_text), LedgerLoad::Missing));

        let mut ledger = FailureLedger::new();
        ledger.record(receipt(EpistemicStatus::Absent, Some("姚明"), Some("体重")));
        ledger.save(path_text).unwrap();
        assert_eq!(FailureLedger::load(path_text), LedgerLoad::Loaded(1));

        let mut value: Value =
            serde_json::from_str(&std::fs::read_to_string(path_text).unwrap()).unwrap();
        value["sections"][LEDGER_SECTION]["receipts"][0]["status"] =
            serde_json::json!("unknown");
        std::fs::write(path_text, value.to_string()).unwrap();
        assert!(matches!(
            FailureLedger::load(path_text),
            LedgerLoad::Damaged(_)
        ));
    }

    #[test]
    fn watermark_survives_restore() {
        let mut ledger = FailureLedger::new();
        ledger.record(receipt(EpistemicStatus::Absent, Some("A"), Some("x")));
        let json = serde_json::to_string(&ledger).unwrap();
        let restored: FailureLedger = serde_json::from_str(&json).unwrap();
        let mut restored = FailureLedger::restore_from(restored);
        let id = restored.record(receipt(EpistemicStatus::Unknown, Some("B"), Some("y")));
        assert_eq!(id, "r2", "跨重启不得重号");
    }
}

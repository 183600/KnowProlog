//! IJG：导入日志（Import Journal）。
//!
//! 跨重启的撤销栈：每次成功导入压入 `ImportReceipt`，`撤销导入` 弹出栈顶
//! 并逐条回撤三元组。多步撤销即连续弹栈；重启后仍可继续撤销。
//!
//! 落盘格式复用 `ContextImage` 信封（单文件、分节、逐节哈希、原子写）：
//!   - 完整性自检失败 → `JournalLoad::Damaged`，调用方以空日志安全降级
//!     （宁可不撤销，也不按损坏的动作误删数据）；
//!   - 日志有界（默认保留最近 32 笔），避免随导入批次无限膨胀。

use serde_json::Value;

use crate::context_image::ContextImage;
use crate::predicate_registry::ImportReceipt;

pub const JOURNAL_SECTION: &str = "import_journal";
pub const DEFAULT_MAX_ENTRIES: usize = 32;

/// 导入日志加载结果：缺失 / 已加载 / 损坏。
#[derive(Debug)]
pub enum JournalLoad {
    Missing,
    Loaded(ImportJournal),
    Damaged(String),
}

/// 有界 LIFO 导入日志（旧 → 新）。
#[derive(Debug)]
pub struct ImportJournal {
    entries: Vec<ImportReceipt>,
    max_entries: usize,
}

impl Default for ImportJournal {
    fn default() -> Self {
        Self::new()
    }
}

impl ImportJournal {
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_MAX_ENTRIES)
    }

    pub fn with_capacity(max_entries: usize) -> Self {
        Self {
            entries: Vec::new(),
            max_entries: max_entries.max(1),
        }
    }

    /// 从已恢复的条目重建（截断到容量上限，保留最新的）。
    pub fn from_entries(entries: Vec<ImportReceipt>) -> Self {
        let mut journal = Self::new();
        for receipt in entries {
            journal.push(receipt);
        }
        journal
    }

    /// 压入一条可撤销回执；不可撤销（无动作）的回执不入栈。
    pub fn push(&mut self, receipt: ImportReceipt) {
        if !receipt.can_undo() {
            return;
        }
        self.entries.push(receipt);
        while self.entries.len() > self.max_entries {
            self.entries.remove(0);
        }
    }

    /// 弹出最近一笔（撤销入口）。
    pub fn pop(&mut self) -> Option<ImportReceipt> {
        self.entries.pop()
    }

    pub fn last(&self) -> Option<&ImportReceipt> {
        self.entries.last()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn entries(&self) -> &[ImportReceipt] {
        &self.entries
    }

    /// 可撤销清单（新 → 旧），供「导入历史」展示。
    pub fn summary(&self) -> String {
        if self.entries.is_empty() {
            return "导入日志：空（暂无跨会话可撤销的导入）。".to_string();
        }
        let mut output = format!("导入日志：{} 笔可撤销（新 → 旧）。\n", self.entries.len());
        for (index, receipt) in self.entries.iter().rev().enumerate() {
            output.push_str(&format!(
                "- #{}：新增 {} 条，合并 {} 条，重复 {} 条，撤销动作 {} 条",
                index + 1,
                receipt.added,
                receipt.merged,
                receipt.duplicates,
                receipt.actions.len()
            ));
            if !receipt.conflicts.is_empty() {
                output.push_str(&format!("，冲突 {} 处", receipt.conflicts.len()));
            }
            if !receipt.provisional_predicates.is_empty() {
                output.push_str(&format!(
                    "，新谓词 {}",
                    receipt.provisional_predicates.join("、")
                ));
            }
            output.push('\n');
        }
        output.push_str("提示：「撤销导入」逐笔弹出栈顶并回撤三元组，重启后仍然可用。");
        output
    }

    /// 以 ContextImage 信封原子落盘。
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let entries = serde_json::to_value(&self.entries)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let mut image = ContextImage::new();
        image.set_section(JOURNAL_SECTION, entries);
        image.set_section("journal_meta", Value::from(self.entries.len() as u64));
        image.seal();
        image.save_atomic(path)
    }

    /// 读取导入日志：分节自检保证只有完整日志才被采信。
    pub fn load(path: &str) -> JournalLoad {
        if !std::path::Path::new(path).exists() {
            return JournalLoad::Missing;
        }
        let image = match ContextImage::load(path) {
            Ok(image) => image,
            Err(error) => return JournalLoad::Damaged(error),
        };
        let audit = image.verify();
        if audit.global_damaged {
            return JournalLoad::Damaged("日志全局哈希不匹配".to_string());
        }
        if audit
            .damaged_sections
            .iter()
            .any(|section| section == JOURNAL_SECTION)
        {
            return JournalLoad::Damaged("日志分节哈希不匹配".to_string());
        }
        let Some(value) = image.section(JOURNAL_SECTION) else {
            return JournalLoad::Loaded(ImportJournal::new());
        };
        match serde_json::from_value::<Vec<ImportReceipt>>(value.clone()) {
            Ok(entries) => JournalLoad::Loaded(ImportJournal::from_entries(entries)),
            Err(error) => JournalLoad::Damaged(format!("日志解析失败：{}", error)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::predicate_registry::{ImportAction, ImportConflict};

    fn receipt(added: usize, marker: &str) -> ImportReceipt {
        ImportReceipt {
            added,
            merged: 0,
            duplicates: 0,
            conflicts: Vec::new(),
            schema_violations: Vec::new(),
            provisional_predicates: Vec::new(),
            pending_admissions: Vec::new(),
            actions: (0..added)
                .map(|index| ImportAction::Literal {
                    subject: format!("{}{}", marker, index),
                    predicate: "http://example.org/kg/name".to_string(),
                    value: marker.to_string(),
                })
                .collect(),
            registry_changed: false,
        }
    }

    #[test]
    fn push_pop_is_lifo_and_ignores_noop() {
        let mut journal = ImportJournal::new();
        journal.push(ImportReceipt::default());
        assert!(journal.is_empty(), "无动作回执不入栈");
        journal.push(receipt(2, "a"));
        journal.push(receipt(1, "b"));
        assert_eq!(journal.len(), 2);
        let popped = journal.pop().unwrap();
        assert_eq!(popped.actions.len(), 1);
        assert_eq!(journal.len(), 1);
        assert!(journal.summary().contains("1 笔可撤销"));
        assert!(journal.pop().is_some());
        assert!(journal.pop().is_none());
    }

    #[test]
    fn journal_is_bounded() {
        let mut journal = ImportJournal::with_capacity(3);
        for index in 0..5 {
            journal.push(receipt(1, &format!("m{}", index)));
        }
        assert_eq!(journal.len(), 3, "只保留最近 3 笔");
        let newest = journal.pop().unwrap();
        assert_eq!(newest.actions[0].value(), "m4");
    }

    #[test]
    fn roundtrip_tamper_and_missing() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("import_journal.json");
        let path_text = path.to_str().unwrap();

        assert!(matches!(ImportJournal::load(path_text), JournalLoad::Missing));

        let mut journal = ImportJournal::new();
        journal.push(receipt(3, "x"));
        journal.save(path_text).unwrap();

        let JournalLoad::Loaded(restored) = ImportJournal::load(path_text) else {
            panic!("应能恢复导入日志");
        };
        assert_eq!(restored.len(), 1);
        assert_eq!(restored.last().unwrap().added, 3);

        // 篡改分节内容：整体拒绝（宁可不撤销，也不误撤）
        let mut value: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path_text).unwrap()).unwrap();
        value["sections"][JOURNAL_SECTION][0]["added"] = serde_json::json!(999);
        std::fs::write(path_text, value.to_string()).unwrap();
        match ImportJournal::load(path_text) {
            JournalLoad::Damaged(_) => {}
            other => panic!("篡改应被拒绝：{:?}", other),
        }
    }

    #[test]
    fn version_gate_rejects_future_image() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("import_journal.json");
        let path_text = path.to_str().unwrap();
        std::fs::write(
            path_text,
            serde_json::json!({
                "image_version": crate::context_image::IMAGE_VERSION + 1,
                "sections": {},
                "section_hashes": {},
                "canonical_hash": ""
            })
            .to_string(),
        )
        .unwrap();
        match ImportJournal::load(path_text) {
            JournalLoad::Damaged(message) => assert!(message.contains("高于支持版本")),
            other => panic!("未来版本应被拒绝：{:?}", other),
        }
    }

    trait ActionValue {
        fn value(&self) -> &str;
    }

    impl ActionValue for ImportAction {
        fn value(&self) -> &str {
            match self {
                ImportAction::Named { .. } => "",
                ImportAction::Literal { value, .. } => value,
            }
        }
    }

    #[test]
    fn conflict_roundtrip_is_stable() {
        let conflict = ImportConflict {
            subject: "s".to_string(),
            predicate: "p".to_string(),
            existing: vec!["1".to_string()],
            incoming: "2".to_string(),
        };
        let text = serde_json::to_string(&conflict).unwrap();
        let restored: ImportConflict = serde_json::from_str(&text).unwrap();
        assert_eq!(restored, conflict);
    }
}

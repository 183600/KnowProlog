//! KLM：知识容器层次清单（Knowledge Layer Manifest）。
//!
//! 回答只读「合成视图」，但用户需要知道某条结论究竟来自哪一层：
//!
//! ```text
//! L0 基础层   图谱关系 + 字面量属性（只读，随知识版本变化）
//! L1 会话层   用户确认/更正/采纳的会话事实与否定边（可撤回，不写图谱）
//! L2 推测层   补全回合中待确认的候选（有证据、可裁定）
//! L3 假设层   当前「如果……」世界的临时前提（退出即丢弃）
//! ```
//!
//! 清单是确定性导出：每层给出条目数、全量摘要与少量样例；摘要覆盖该层
//! 全部条目（样例不参与摘要的二次裁剪），同一状态永远得到同一摘要。
//! 这不是训练权重，也不是自由生成——只是把符号系统的前提分层摊开，供
//! 用户逐层审计、复制与比较。

use serde::{Deserialize, Serialize};

use crate::context_image::{canonical_json, fnv1a_hex};

/// 清单格式版本。
pub const MANIFEST_VERSION: u32 = 1;
/// 每层展示的样例上限（摘要仍覆盖全量）。
pub const SAMPLE_LIMIT: usize = 8;

/// 一层知识容器的可见状态。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerEntry {
    /// 稳定层 id（`L0`–`L3`）。
    pub id: String,
    /// 层名（基础层 / 会话层 / 推测层 / 假设层）。
    pub name: String,
    /// 写入纪律（只读 / 会话可写 / 待确认 / 临时）。
    pub discipline: String,
    /// 当前是否为空。
    pub empty: bool,
    /// 条目数。
    pub count: usize,
    /// 全量条目的规范摘要（FNV-1a 64）。
    pub digest: String,
    /// 有界样例（排序后前 [`SAMPLE_LIMIT`] 条）。
    pub sample: Vec<String>,
}

impl LayerEntry {
    /// 构造一层：入参按字典序排序、去重后计算全量摘要。
    pub fn new(
        id: &str,
        name: &str,
        discipline: &str,
        items: impl IntoIterator<Item = String>,
    ) -> Self {
        let mut items: Vec<String> = items
            .into_iter()
            .map(|item| item.trim().to_string())
            .filter(|item| !item.is_empty())
            .collect();
        items.sort();
        items.dedup();
        let digest = fnv1a_hex(canonical_json(&serde_json::Value::Array(
            items
                .iter()
                .map(|item| serde_json::Value::String(item.clone()))
                .collect(),
        ))
        .as_bytes());
        Self {
            id: id.to_string(),
            name: name.to_string(),
            discipline: discipline.to_string(),
            empty: items.is_empty(),
            count: items.len(),
            digest,
            sample: items.into_iter().take(SAMPLE_LIMIT).collect(),
        }
    }
}

/// 一次导出：知识版本 + 四层状态 + 合成摘要。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KnowledgeLayerManifest {
    pub manifest_version: u32,
    /// 基础图谱的知识版本（写路径推进；只读分叉与父会话共享）。
    pub revision: u64,
    pub layers: Vec<LayerEntry>,
    /// 覆盖版本与全部层摘要的合成摘要。
    pub digest: String,
}

impl KnowledgeLayerManifest {
    pub fn new(revision: u64, layers: Vec<LayerEntry>) -> Self {
        let mut manifest = Self {
            manifest_version: MANIFEST_VERSION,
            revision,
            layers,
            digest: String::new(),
        };
        manifest.digest = manifest.compute_digest();
        manifest
    }

    fn compute_digest(&self) -> String {
        let value = serde_json::json!({
            "manifest_version": self.manifest_version,
            "revision": self.revision,
            "layers": self.layers.iter().map(|layer| serde_json::json!({
                "id": layer.id,
                "count": layer.count,
                "digest": layer.digest,
            })).collect::<Vec<_>>(),
        });
        fnv1a_hex(canonical_json(&value).as_bytes())
    }

    pub fn layer(&self, id: &str) -> Option<&LayerEntry> {
        self.layers.iter().find(|layer| layer.id == id)
    }

    /// Dockerfile 风格的层次清单（人读；摘要可复制比对）。
    pub fn render_text(&self) -> String {
        let mut output = format!(
            "知识容器层次清单（manifest v{}，知识版本 {}，合成摘要 {}）\n",
            self.manifest_version, self.revision, self.digest
        );
        for layer in &self.layers {
            output.push_str(&format!(
                "{} {}（{}）：{} 条，摘要 {}{}\n",
                layer.id,
                layer.name,
                layer.discipline,
                layer.count,
                layer.digest,
                if layer.empty { "，空" } else { "" }
            ));
            for item in &layer.sample {
                output.push_str(&format!("    {}\n", item));
            }
            if layer.count > layer.sample.len() {
                output.push_str(&format!(
                    "    ……（另有 {} 条；样例有界，摘要覆盖全量）\n",
                    layer.count - layer.sample.len()
                ));
            }
        }
        output.push_str(
            "说明：回答只读合成视图；L1 可撤回、L2 待确认、L3 退出即丢弃，均不写基础图谱。",
        );
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_layers_are_deterministic_and_sorted() {
        let left = LayerEntry::new("L0", "基础层", "只读", Vec::<String>::new());
        let right = LayerEntry::new("L0", "基础层", "只读", vec!["".to_string()]);
        assert_eq!(left, right);
        assert!(left.empty);
        assert_eq!(left.count, 0);
    }

    #[test]
    fn sample_is_bounded_but_digest_covers_all_items() {
        let mut items: Vec<String> = (0..20).map(|index| format!("事实{}", index)).collect();
        items.reverse();
        let layer = LayerEntry::new("L0", "基础层", "只读", items);
        assert_eq!(layer.count, 20);
        assert_eq!(layer.sample.len(), SAMPLE_LIMIT);
        assert_eq!(
            layer.sample.first().map(String::as_str),
            Some("事实0"),
            "样例必须按字典序稳定"
        );

        let mut reordered: Vec<String> = (0..20).map(|index| format!("事实{}", index)).collect();
        reordered.swap(0, 19);
        let same = LayerEntry::new("L0", "基础层", "只读", reordered);
        assert_eq!(layer, same, "排序去重后同一内容必须同摘要");
    }

    #[test]
    fn manifest_changes_when_any_layer_changes() {
        let base = KnowledgeLayerManifest::new(
            1,
            vec![LayerEntry::new("L0", "基础层", "只读", ["a".to_string()])],
        );
        let changed = KnowledgeLayerManifest::new(
            1,
            vec![LayerEntry::new("L0", "基础层", "只读", ["b".to_string()])],
        );
        assert_ne!(base.digest, changed.digest);
        let bumped = KnowledgeLayerManifest::new(
            2,
            vec![LayerEntry::new("L0", "基础层", "只读", ["a".to_string()])],
        );
        assert_ne!(base.digest, bumped.digest);
    }

    #[test]
    fn render_marks_empty_layers_and_counts_overflow() {
        let manifest = KnowledgeLayerManifest::new(
            7,
            vec![
                LayerEntry::new("L0", "基础层", "只读", Vec::<String>::new()),
                LayerEntry::new(
                    "L1",
                    "会话层",
                    "会话可写",
                    (0..10).map(|index| format!("会话事实{}", index)),
                ),
            ],
        );
        let text = manifest.render_text();
        assert!(text.contains("知识版本 7"), "{}", text);
        assert!(text.contains("空"), "{}", text);
        assert!(text.contains("另有 2 条"), "{}", text);
    }
}

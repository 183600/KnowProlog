use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::context_image::fnv1a_hex;

use super::anchor::{Anchor, AnchorDraft};
use super::asset::{
    AnchorId, AssetId, AssetRecord, MediaError, MediaLimits, Modality, MAX_LABEL_CHARS,
    MAX_MEDIA_TYPE_CHARS,
};

/// 已存资产（字节不可变，锚点有界追加）。
#[derive(Debug, Clone)]
pub(crate) struct StoredAsset {
    pub(crate) record: AssetRecord,
    pub(crate) bytes: Arc<[u8]>,
    pub(crate) anchors: Vec<Anchor>,
}

/// 有界媒体库：内容寻址、发布幂等、删除后引用失效。
#[derive(Debug)]
pub struct MediaStore {
    pub(crate) assets: BTreeMap<AssetId, StoredAsset>,
    total_bytes: usize,
    pub(crate) next_sequence: u64,
    next_ordinals: BTreeMap<Modality, u64>,
    next_anchor_sequence: u64,
    pub(crate) limits: MediaLimits,
    root: Option<PathBuf>,
}

impl Default for MediaStore {
    fn default() -> Self {
        Self::new(MediaLimits::default())
    }
}

impl MediaStore {
    pub fn new(limits: MediaLimits) -> Self {
        Self {
            assets: BTreeMap::new(),
            total_bytes: 0,
            next_sequence: 1,
            next_ordinals: BTreeMap::new(),
            next_anchor_sequence: 1,
            limits,
            root: None,
        }
    }

    /// 绑定媒体根目录；发布时原子写入 `<root>/<hash>.bin`。
    pub fn with_root(root: impl Into<PathBuf>, limits: MediaLimits) -> Self {
        Self {
            root: Some(root.into()),
            ..Self::new(limits)
        }
    }

    pub fn limits(&self) -> MediaLimits {
        self.limits
    }

    pub fn root(&self) -> Option<&Path> {
        self.root.as_deref()
    }

    pub fn len(&self) -> usize {
        self.assets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.assets.is_empty()
    }

    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    pub fn asset(&self, id: &AssetId) -> Option<&AssetRecord> {
        self.assets.get(id).map(|stored| &stored.record)
    }

    pub fn bytes(&self, id: &AssetId) -> Option<&[u8]> {
        self.assets.get(id).map(|stored| stored.bytes.as_ref())
    }

    pub fn anchors(&self, id: &AssetId) -> Option<&[Anchor]> {
        self.assets.get(id).map(|stored| stored.anchors.as_slice())
    }

    pub fn anchor(&self, asset_id: &AssetId, anchor_id: &AnchorId) -> Option<&Anchor> {
        self.assets
            .get(asset_id)?
            .anchors
            .iter()
            .find(|anchor| anchor.id == *anchor_id)
    }

    /// 按发布次序返回全部资产。
    pub fn assets_in_order(&self) -> Vec<&AssetRecord> {
        let mut records: Vec<&AssetRecord> =
            self.assets.values().map(|stored| &stored.record).collect();
        records.sort_by_key(|record| record.sequence);
        records
    }

    /// 发布资产：同字节内容幂等归并；先检查全部上限，再做 IO。
    pub fn publish(
        &mut self,
        modality: Modality,
        media_type: &str,
        label: &str,
        bytes: &[u8],
    ) -> Result<AssetRecord, MediaError> {
        if bytes.is_empty() {
            return Err(MediaError::EmptyAsset);
        }
        let media_type = media_type.trim();
        if media_type.is_empty() {
            return Err(MediaError::EmptyMediaType);
        }
        let media_type_len = media_type.chars().count();
        if media_type_len > MAX_MEDIA_TYPE_CHARS {
            return Err(MediaError::MediaTypeTooLong {
                actual: media_type_len,
                limit: MAX_MEDIA_TYPE_CHARS,
            });
        }
        let content_hash = fnv1a_hex(bytes);
        let id = AssetId(format!("asset-{}", content_hash));
        if let Some(existing) = self.assets.get(&id) {
            return Ok(existing.record.clone());
        }
        if bytes.len() > self.limits.max_asset_bytes {
            return Err(MediaError::AssetTooLarge {
                actual: bytes.len(),
                limit: self.limits.max_asset_bytes,
            });
        }
        if self.assets.len() >= self.limits.max_assets {
            return Err(MediaError::AssetCapacityExceeded {
                limit: self.limits.max_assets,
            });
        }
        let total = self.total_bytes.saturating_add(bytes.len());
        if total > self.limits.max_total_bytes {
            return Err(MediaError::TotalBytesExceeded {
                actual: total,
                limit: self.limits.max_total_bytes,
            });
        }
        let sequence = self.next_sequence;
        let ordinal = self.next_ordinals.get(&modality).copied().unwrap_or(1);
        let label = if label.trim().is_empty() {
            format!("{} {}", modality.label_zh(), ordinal)
        } else {
            label.trim().to_string()
        };
        let label_len = label.chars().count();
        if label_len > MAX_LABEL_CHARS {
            return Err(MediaError::LabelTooLong {
                actual: label_len,
                limit: MAX_LABEL_CHARS,
            });
        }
        if let Some(root) = &self.root {
            persist_bytes(root, &content_hash, bytes)?;
        }
        let record = AssetRecord {
            id: id.clone(),
            modality,
            media_type: media_type.to_string(),
            content_hash,
            byte_len: bytes.len(),
            label,
            ordinal,
            sequence,
        };
        self.next_sequence = self.next_sequence.saturating_add(1);
        self.next_ordinals
            .insert(modality, ordinal.saturating_add(1));
        self.total_bytes = total;
        self.assets.insert(
            id,
            StoredAsset {
                record: record.clone(),
                bytes: Arc::from(bytes),
                anchors: Vec::new(),
            },
        );
        Ok(record)
    }

    /// 追加锚点；定位符必须与资产模态相容，文本与数量受硬上限约束。
    pub fn add_anchor(
        &mut self,
        asset_id: &AssetId,
        draft: AnchorDraft,
    ) -> Result<Anchor, MediaError> {
        let (modality, anchor_count) = {
            let stored = self
                .assets
                .get(asset_id)
                .ok_or_else(|| MediaError::UnknownAsset(asset_id.as_str().to_string()))?;
            (stored.record.modality, stored.anchors.len())
        };
        if anchor_count >= self.limits.max_anchors_per_asset {
            return Err(MediaError::AnchorCapacityExceeded {
                limit: self.limits.max_anchors_per_asset,
            });
        }
        draft.locator.validate_for(modality)?;
        let text_len = draft.text.chars().count();
        if text_len > self.limits.max_anchor_text_chars {
            return Err(MediaError::AnchorTextTooLong {
                actual: text_len,
                limit: self.limits.max_anchor_text_chars,
            });
        }
        if draft.extractor.trim().is_empty() {
            return Err(MediaError::EmptyExtractor);
        }
        if !draft.confidence.is_finite() || !(0.0..=1.0).contains(&draft.confidence) {
            return Err(MediaError::InvalidConfidence);
        }
        let anchor = Anchor {
            id: AnchorId(format!("a{}", self.next_anchor_sequence)),
            locator: draft.locator,
            text: draft.text,
            extractor: draft.extractor.trim().to_string(),
            confidence: draft.confidence,
            entities: draft.entities,
        };
        self.next_anchor_sequence = self.next_anchor_sequence.saturating_add(1);
        if let Some(stored) = self.assets.get_mut(asset_id) {
            stored.anchors.push(anchor.clone());
        }
        Ok(anchor)
    }

    /// 删除资产；旧 `AssetRef` / `MediaCitation` 从此不可用。
    pub fn remove(&mut self, id: &AssetId) -> bool {
        match self.assets.remove(id) {
            Some(stored) => {
                self.total_bytes = self.total_bytes.saturating_sub(stored.record.byte_len);
                true
            }
            None => false,
        }
    }
}

fn persist_bytes(root: &Path, content_hash: &str, bytes: &[u8]) -> Result<(), MediaError> {
    std::fs::create_dir_all(root).map_err(|error| MediaError::Io(error.to_string()))?;
    let path = root.join(format!("{}.bin", content_hash));
    if path.exists() {
        let existing = std::fs::read(&path).map_err(|error| MediaError::Io(error.to_string()))?;
        let actual = fnv1a_hex(&existing);
        if actual != content_hash {
            return Err(MediaError::PersistedHashMismatch {
                expected: content_hash.to_string(),
                actual,
            });
        }
        return Ok(());
    }
    let temporary = root.join(format!(".{}.{}.tmp", content_hash, std::process::id()));
    std::fs::write(&temporary, bytes).map_err(|error| MediaError::Io(error.to_string()))?;
    std::fs::rename(&temporary, &path).map_err(|error| {
        let _ = std::fs::remove_file(&temporary);
        MediaError::Io(error.to_string())
    })
}

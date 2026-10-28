use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::context_image::ContextImage;

use super::asset::{
    AssetId, AssetRecord, AssetRef, MediaError, Modality, UnavailableReason,
    DEFAULT_MAX_SESSION_ASSETS, MEDIA_EVIDENCE_VERSION, MEDIA_SECTION,
};
use super::store::MediaStore;

/// 会话媒体台账快照：只有逻辑引用，绝不包含媒体字节或 base64。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MediaLedgerSnapshot {
    pub version: u32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub assets: Vec<AssetRef>,
}

impl Default for MediaLedgerSnapshot {
    fn default() -> Self {
        Self {
            version: MEDIA_EVIDENCE_VERSION,
            assets: Vec::new(),
        }
    }
}

/// 恢复时被丢弃的引用。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DroppedAsset {
    pub reference: AssetRef,
    pub reason: DropReason,
}

impl DroppedAsset {
    pub fn render_zh(&self) -> String {
        format!(
            "{}：{}",
            self.reference.display_zh(),
            self.reason.label_zh()
        )
    }
}

/// 引用被丢弃的原因。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DropReason {
    Missing,
    HashMismatch,
    Stale,
    Capacity,
}

impl DropReason {
    pub fn code(self) -> &'static str {
        match self {
            DropReason::Missing => "missing",
            DropReason::HashMismatch => "hash_mismatch",
            DropReason::Stale => "stale",
            DropReason::Capacity => "capacity",
        }
    }

    pub fn label_zh(self) -> &'static str {
        match self {
            DropReason::Missing => "媒体已不可用",
            DropReason::HashMismatch => "内容哈希不符",
            DropReason::Stale => "旧世代引用已失效",
            DropReason::Capacity => "会话媒体容量已满",
        }
    }

    fn from_unavailable(reason: UnavailableReason) -> Self {
        match reason {
            UnavailableReason::Missing => DropReason::Missing,
            UnavailableReason::HashMismatch { .. } => DropReason::HashMismatch,
            UnavailableReason::Stale { .. } => DropReason::Stale,
        }
    }
}

/// 会话媒体台账：有界、按引用去重、最近使用者为焦点。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionMedia {
    references: Vec<AssetRef>,
    capacity: usize,
}

impl Default for SessionMedia {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_SESSION_ASSETS)
    }
}

impl SessionMedia {
    pub fn new(capacity: usize) -> Self {
        Self {
            references: Vec::new(),
            capacity,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.references.len()
    }

    pub fn is_empty(&self) -> bool {
        self.references.is_empty()
    }

    pub fn refs(&self) -> &[AssetRef] {
        &self.references
    }

    /// 最近使用/关注的资产（焦点）。
    pub fn focus(&self) -> Option<&AssetRef> {
        self.references.last()
    }

    pub fn contains(&self, id: &AssetId) -> bool {
        self.references
            .iter()
            .any(|reference| reference.asset_id == *id)
    }

    /// 记住一个资产；重复引用的最近使用顺序被刷新，容量按去重后计数。
    pub fn remember(&mut self, record: &AssetRecord) -> Result<(), MediaError> {
        self.remember_ref(AssetRef::from_record(record))
    }

    fn remember_ref(&mut self, reference: AssetRef) -> Result<(), MediaError> {
        if let Some(position) = self
            .references
            .iter()
            .position(|existing| existing.asset_id == reference.asset_id)
        {
            self.references.remove(position);
        } else if self.references.len() >= self.capacity {
            return Err(MediaError::SessionCapacityExceeded {
                limit: self.capacity,
            });
        }
        self.references.push(reference);
        Ok(())
    }

    pub fn forget(&mut self, id: &AssetId) -> bool {
        let before = self.references.len();
        self.references
            .retain(|reference| reference.asset_id != *id);
        self.references.len() != before
    }

    /// 当前焦点中最近的、仍可解析且模态匹配的资产。
    pub fn focus_for(&self, store: &MediaStore, modality: Modality) -> Option<AssetRef> {
        self.references.iter().rev().find_map(|reference| {
            reference
                .resolve(store)
                .ok()
                .filter(|record| record.modality == modality)
                .map(|_| reference.clone())
        })
    }

    pub fn snapshot(&self) -> MediaLedgerSnapshot {
        MediaLedgerSnapshot {
            version: MEDIA_EVIDENCE_VERSION,
            assets: self.references.clone(),
        }
    }

    pub fn to_value(&self) -> Result<Value, MediaError> {
        serde_json::to_value(self.snapshot())
            .map_err(|error| MediaError::SnapshotDamaged(error.to_string()))
    }

    pub fn from_value(value: &Value) -> Result<Self, MediaError> {
        Self::from_value_with_capacity(value, DEFAULT_MAX_SESSION_ASSETS)
    }

    pub fn from_value_with_capacity(value: &Value, capacity: usize) -> Result<Self, MediaError> {
        let snapshot: MediaLedgerSnapshot = serde_json::from_value(value.clone())
            .map_err(|error| MediaError::SnapshotDamaged(error.to_string()))?;
        if snapshot.version != MEDIA_EVIDENCE_VERSION {
            return Err(MediaError::SnapshotVersionUnsupported {
                found: snapshot.version,
                supported: MEDIA_EVIDENCE_VERSION,
            });
        }
        if snapshot.assets.len() > capacity {
            return Err(MediaError::SessionCapacityExceeded { limit: capacity });
        }
        Ok(Self {
            references: snapshot.assets,
            capacity,
        })
    }

    /// 把台账写入 `ContextImage`；空台账不落节，保持旧快照字节不变。
    pub fn install_into(&self, image: &mut ContextImage) -> Result<(), MediaError> {
        if self.references.is_empty() {
            image.remove_section(MEDIA_SECTION);
            return Ok(());
        }
        image.set_section(MEDIA_SECTION, self.to_value()?);
        Ok(())
    }

    /// 从 `ContextImage` 恢复；缺失分节视为空台账。
    pub fn restore_from(
        image: &ContextImage,
        store: &MediaStore,
    ) -> Result<(Self, Vec<DroppedAsset>), MediaError> {
        match image.section(MEDIA_SECTION) {
            Some(value) => {
                let parsed = Self::from_value(value)?;
                let snapshot = parsed.snapshot();
                Ok(Self::restore(&snapshot, store))
            }
            None => Ok((Self::default(), Vec::new())),
        }
    }

    /// 恢复并逐条核验内容哈希与世代；不可用引用被丢弃而非带病入会话。
    pub fn restore(
        snapshot: &MediaLedgerSnapshot,
        store: &MediaStore,
    ) -> (Self, Vec<DroppedAsset>) {
        Self::restore_with_capacity(snapshot, store, DEFAULT_MAX_SESSION_ASSETS)
    }

    pub fn restore_with_capacity(
        snapshot: &MediaLedgerSnapshot,
        store: &MediaStore,
        capacity: usize,
    ) -> (Self, Vec<DroppedAsset>) {
        let mut session = Self::new(capacity);
        let mut dropped = Vec::new();
        for reference in &snapshot.assets {
            match reference.resolve(store) {
                Ok(_) => {
                    if session.remember_ref(reference.clone()).is_err() {
                        dropped.push(DroppedAsset {
                            reference: reference.clone(),
                            reason: DropReason::Capacity,
                        });
                    }
                }
                Err(reason) => dropped.push(DroppedAsset {
                    reference: reference.clone(),
                    reason: DropReason::from_unavailable(reason),
                }),
            }
        }
        (session, dropped)
    }
}

use std::fmt;

use serde::{Deserialize, Serialize};

/// 媒体证据格式版本。
pub const MEDIA_EVIDENCE_VERSION: u32 = 1;
/// 会话媒体台账在 `ContextImage` 中的分节名。
pub const MEDIA_SECTION: &str = "media_ledger";
/// 默认资产数量上限。
pub const DEFAULT_MAX_ASSETS: usize = 64;
/// 单个资产字节上限（8 MiB）。
pub const DEFAULT_MAX_ASSET_BYTES: usize = 8 * 1024 * 1024;
/// 全部资产字节上限（64 MiB）。
pub const DEFAULT_MAX_TOTAL_BYTES: usize = 64 * 1024 * 1024;
/// 单个资产的锚点数量上限。
pub const DEFAULT_MAX_ANCHORS_PER_ASSET: usize = 64;
/// 单条锚点文本字符上限。
pub const DEFAULT_MAX_ANCHOR_TEXT: usize = 4096;
/// 会话最多记住的资产引用数。
pub const DEFAULT_MAX_SESSION_ASSETS: usize = 32;
/// 资产标签字符上限。
pub const MAX_LABEL_CHARS: usize = 80;
/// MIME 类型字符上限。
pub const MAX_MEDIA_TYPE_CHARS: usize = 120;
/// 澄清提示中最多展示的候选数。
pub const CANDIDATE_DISPLAY_CAP: usize = 8;
/// 图区域坐标采用千分位（0..=1000），与像素尺寸解耦。
pub const IMAGE_COORDINATE_SCALE: u32 = 1000;

/// 媒体模态（区别于 `relation_frame::Modality` 的语言学情态）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Modality {
    Image,
    Table,
    Audio,
    Document,
}

impl Modality {
    pub fn code(self) -> &'static str {
        match self {
            Modality::Image => "image",
            Modality::Table => "table",
            Modality::Audio => "audio",
            Modality::Document => "document",
        }
    }

    pub fn label_zh(self) -> &'static str {
        match self {
            Modality::Image => "图片",
            Modality::Table => "表格",
            Modality::Audio => "录音",
            Modality::Document => "文档",
        }
    }
}

/// 内容寻址的资产 id（`asset-<fnv1a>`）。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AssetId(pub(crate) String);

impl AssetId {
    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn parse(value: impl Into<String>) -> Option<Self> {
        let value = value.into();
        (!value.trim().is_empty()).then_some(Self(value))
    }
}

/// 资产内唯一的锚点 id（`a1`、`a2`…）。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AnchorId(pub(crate) String);

impl AnchorId {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// 发布后的资产元数据；字节本身在 `MediaStore` 内，不进入 RDF/快照。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssetRecord {
    pub id: AssetId,
    pub modality: Modality,
    pub media_type: String,
    pub content_hash: String,
    pub byte_len: usize,
    /// 面向用户的稳定标签（`图片 1`、`表格 2`…）。
    pub label: String,
    /// 同模态内的发布序号（从 1 开始）；与默认标签和「第 N 张」一致。
    pub ordinal: u64,
    /// 全局发布次序（单调递增）；用于「最近一张」与世代判定。
    pub sequence: u64,
}

impl AssetRecord {
    pub fn display_zh(&self) -> String {
        format!("{}（{}）", self.label, self.modality.label_zh())
    }
}

/// 会话持有的逻辑资产引用；只含 id/哈希/世代/标签。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssetRef {
    pub asset_id: AssetId,
    pub content_hash: String,
    pub sequence: u64,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub label: String,
}

impl AssetRef {
    pub fn from_record(record: &AssetRecord) -> Self {
        Self {
            asset_id: record.id.clone(),
            content_hash: record.content_hash.clone(),
            sequence: record.sequence,
            label: record.label.clone(),
        }
    }

    pub fn display_zh(&self) -> &str {
        if self.label.is_empty() {
            self.asset_id.as_str()
        } else {
            &self.label
        }
    }

    /// 按内容哈希与世代解析；删除或重发后旧引用永久失效。
    pub fn resolve<'a>(
        &self,
        store: &'a super::store::MediaStore,
    ) -> Result<&'a AssetRecord, UnavailableReason> {
        let record = store
            .asset(&self.asset_id)
            .ok_or(UnavailableReason::Missing)?;
        if record.content_hash != self.content_hash {
            return Err(UnavailableReason::HashMismatch {
                expected: self.content_hash.clone(),
                actual: record.content_hash.clone(),
            });
        }
        if record.sequence != self.sequence {
            return Err(UnavailableReason::Stale {
                expected_sequence: self.sequence,
                actual_sequence: record.sequence,
            });
        }
        Ok(record)
    }
}

/// 旧引用不可用的原因。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "reason")]
pub enum UnavailableReason {
    Missing,
    HashMismatch {
        expected: String,
        actual: String,
    },
    Stale {
        expected_sequence: u64,
        actual_sequence: u64,
    },
}

impl UnavailableReason {
    pub fn code(&self) -> &'static str {
        match self {
            UnavailableReason::Missing => "missing",
            UnavailableReason::HashMismatch { .. } => "hash_mismatch",
            UnavailableReason::Stale { .. } => "stale",
        }
    }

    pub fn label_zh(&self) -> &'static str {
        match self {
            UnavailableReason::Missing => "媒体已删除或未载入",
            UnavailableReason::HashMismatch { .. } => "媒体内容与引用哈希不符",
            UnavailableReason::Stale { .. } => "媒体已被重新发布，旧引用失效",
        }
    }
}

impl fmt::Display for UnavailableReason {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnavailableReason::Missing => write!(formatter, "媒体已不可用"),
            UnavailableReason::HashMismatch { expected, actual } => write!(
                formatter,
                "媒体哈希不符（引用 {}，实际 {}）",
                expected, actual
            ),
            UnavailableReason::Stale {
                expected_sequence,
                actual_sequence,
            } => write!(
                formatter,
                "媒体世代已变化（引用 {}，当前 {}）",
                expected_sequence, actual_sequence
            ),
        }
    }
}

/// 媒体库的硬上限。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MediaLimits {
    pub max_assets: usize,
    pub max_asset_bytes: usize,
    pub max_total_bytes: usize,
    pub max_anchors_per_asset: usize,
    pub max_anchor_text_chars: usize,
}

impl Default for MediaLimits {
    fn default() -> Self {
        Self {
            max_assets: DEFAULT_MAX_ASSETS,
            max_asset_bytes: DEFAULT_MAX_ASSET_BYTES,
            max_total_bytes: DEFAULT_MAX_TOTAL_BYTES,
            max_anchors_per_asset: DEFAULT_MAX_ANCHORS_PER_ASSET,
            max_anchor_text_chars: DEFAULT_MAX_ANCHOR_TEXT,
        }
    }
}

/// 媒体层类型化错误；任何资源超限都显式失败，不静默截断。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MediaError {
    EmptyAsset,
    EmptyMediaType,
    LabelTooLong { actual: usize, limit: usize },
    MediaTypeTooLong { actual: usize, limit: usize },
    AssetTooLarge { actual: usize, limit: usize },
    TotalBytesExceeded { actual: usize, limit: usize },
    AssetCapacityExceeded { limit: usize },
    AnchorCapacityExceeded { limit: usize },
    AnchorTextTooLong { actual: usize, limit: usize },
    EmptyExtractor,
    InvalidConfidence,
    InvalidLocator(&'static str),
    InvalidTable(&'static str),
    TableTooLarge { cells: usize, limit: usize },
    LocatorModalityMismatch {
        modality: Modality,
        locator: &'static str,
    },
    UnknownAsset(String),
    SessionCapacityExceeded { limit: usize },
    SnapshotVersionUnsupported { found: u32, supported: u32 },
    SnapshotDamaged(String),
    Io(String),
    PersistedHashMismatch { expected: String, actual: String },
}

impl MediaError {
    pub fn code(&self) -> &'static str {
        match self {
            MediaError::EmptyAsset => "empty_asset",
            MediaError::EmptyMediaType => "empty_media_type",
            MediaError::LabelTooLong { .. } => "label_too_long",
            MediaError::MediaTypeTooLong { .. } => "media_type_too_long",
            MediaError::AssetTooLarge { .. } => "asset_too_large",
            MediaError::TotalBytesExceeded { .. } => "total_bytes_exceeded",
            MediaError::AssetCapacityExceeded { .. } => "asset_capacity_exceeded",
            MediaError::AnchorCapacityExceeded { .. } => "anchor_capacity_exceeded",
            MediaError::AnchorTextTooLong { .. } => "anchor_text_too_long",
            MediaError::EmptyExtractor => "empty_extractor",
            MediaError::InvalidConfidence => "invalid_confidence",
            MediaError::InvalidLocator(_) => "invalid_locator",
            MediaError::InvalidTable(_) => "invalid_table",
            MediaError::TableTooLarge { .. } => "table_too_large",
            MediaError::LocatorModalityMismatch { .. } => "locator_modality_mismatch",
            MediaError::UnknownAsset(_) => "unknown_asset",
            MediaError::SessionCapacityExceeded { .. } => "session_capacity_exceeded",
            MediaError::SnapshotVersionUnsupported { .. } => "snapshot_version_unsupported",
            MediaError::SnapshotDamaged(_) => "snapshot_damaged",
            MediaError::Io(_) => "io",
            MediaError::PersistedHashMismatch { .. } => "persisted_hash_mismatch",
        }
    }
}

impl fmt::Display for MediaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MediaError::EmptyAsset => write!(formatter, "不能发布空的媒体资产"),
            MediaError::EmptyMediaType => write!(formatter, "媒体类型不能为空"),
            MediaError::LabelTooLong { actual, limit } => {
                write!(formatter, "资产标签过长（{} > {} 字符）", actual, limit)
            }
            MediaError::MediaTypeTooLong { actual, limit } => {
                write!(formatter, "媒体类型过长（{} > {} 字符）", actual, limit)
            }
            MediaError::AssetTooLarge { actual, limit } => {
                write!(formatter, "单个资产过大（{} > {} 字节）", actual, limit)
            }
            MediaError::TotalBytesExceeded { actual, limit } => {
                write!(formatter, "媒体库总字节超限（{} > {} 字节）", actual, limit)
            }
            MediaError::AssetCapacityExceeded { limit } => {
                write!(formatter, "资产数量已达上限 {}", limit)
            }
            MediaError::AnchorCapacityExceeded { limit } => {
                write!(formatter, "锚点数量已达上限 {}", limit)
            }
            MediaError::AnchorTextTooLong { actual, limit } => {
                write!(formatter, "锚点文本过长（{} > {} 字符）", actual, limit)
            }
            MediaError::EmptyExtractor => write!(formatter, "抽取器名称不能为空"),
            MediaError::InvalidConfidence => write!(formatter, "置信度必须位于 0..=1"),
            MediaError::InvalidLocator(reason) => write!(formatter, "非法定位符：{}", reason),
            MediaError::InvalidTable(reason) => write!(formatter, "非法表格：{}", reason),
            MediaError::TableTooLarge { cells, limit } => {
                write!(formatter, "表格单元格过多（{} > {}）", cells, limit)
            }
            MediaError::LocatorModalityMismatch { modality, locator } => write!(
                formatter,
                "定位符 {} 不适用于 {} 资产",
                locator,
                modality.label_zh()
            ),
            MediaError::UnknownAsset(id) => write!(formatter, "未知媒体资产：{}", id),
            MediaError::SessionCapacityExceeded { limit } => {
                write!(formatter, "会话媒体引用已达上限 {}", limit)
            }
            MediaError::SnapshotVersionUnsupported { found, supported } => write!(
                formatter,
                "媒体台账版本不支持（文件 {}，当前 {}）",
                found, supported
            ),
            MediaError::SnapshotDamaged(reason) => write!(formatter, "媒体台账损坏：{}", reason),
            MediaError::Io(reason) => write!(formatter, "媒体文件读写失败：{}", reason),
            MediaError::PersistedHashMismatch { expected, actual } => write!(
                formatter,
                "媒体根目录中的内容与哈希不符（期望 {}，实际 {}）",
                expected, actual
            ),
        }
    }
}

impl std::error::Error for MediaError {}

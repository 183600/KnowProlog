use serde::{Deserialize, Serialize};

use super::asset::{AnchorId, AssetRecord, MediaError, Modality, IMAGE_COORDINATE_SCALE};
use super::support::MediaCitation;

/// 锚点的定位符。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Locator {
    /// 覆盖整个资产。
    Whole,
    /// 图片区域；坐标均为千分位（0..=1000），与像素解耦。
    ImageRegion {
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    },
    /// 表格单元格（行列从 1 开始）。
    TableCell { row: u32, column: u32 },
    /// 表格整行（从 1 开始）。
    TableRow { row: u32 },
    /// 音频区间（毫秒）。
    AudioSpan { start_ms: u64, end_ms: u64 },
    /// 文本区间（字符下标，半开区间）。
    TextSpan { start: usize, end: usize },
}

impl Locator {
    pub fn label_zh(&self) -> String {
        match self {
            Locator::Whole => "整个资产".to_string(),
            Locator::ImageRegion {
                x,
                y,
                width,
                height,
            } => format!("图区域 x={}‰ y={}‰ {}×{}‰", x, y, width, height),
            Locator::TableCell { row, column } => {
                format!("表格第 {} 行第 {} 列", row, column)
            }
            Locator::TableRow { row } => format!("表格第 {} 行", row),
            Locator::AudioSpan { start_ms, end_ms } => format!(
                "录音 {:.1}–{:.1} 秒",
                *start_ms as f64 / 1000.0,
                *end_ms as f64 / 1000.0
            ),
            Locator::TextSpan { start, end } => {
                format!("文本第 {}–{} 字", start.saturating_add(1), end)
            }
        }
    }

    pub(crate) fn validate_for(&self, modality: Modality) -> Result<(), MediaError> {
        match self {
            Locator::Whole => Ok(()),
            Locator::ImageRegion {
                x,
                y,
                width,
                height,
            } => {
                if modality != Modality::Image {
                    return Err(MediaError::LocatorModalityMismatch {
                        modality,
                        locator: "image_region",
                    });
                }
                if *width == 0 || *height == 0 {
                    return Err(MediaError::InvalidLocator("图区域宽高必须大于 0"));
                }
                if x
                    .checked_add(*width)
                    .is_none_or(|end| end > IMAGE_COORDINATE_SCALE)
                    || y
                        .checked_add(*height)
                        .is_none_or(|end| end > IMAGE_COORDINATE_SCALE)
                {
                    return Err(MediaError::InvalidLocator(
                        "图区域坐标必须落在 0..=1000 千分位内",
                    ));
                }
                Ok(())
            }
            Locator::TableCell { row, column } => {
                if modality != Modality::Table {
                    return Err(MediaError::LocatorModalityMismatch {
                        modality,
                        locator: "table_cell",
                    });
                }
                if *row == 0 || *column == 0 {
                    return Err(MediaError::InvalidLocator("表格行列从 1 开始"));
                }
                Ok(())
            }
            Locator::TableRow { row } => {
                if modality != Modality::Table {
                    return Err(MediaError::LocatorModalityMismatch {
                        modality,
                        locator: "table_row",
                    });
                }
                if *row == 0 {
                    return Err(MediaError::InvalidLocator("表格行号从 1 开始"));
                }
                Ok(())
            }
            Locator::AudioSpan { start_ms, end_ms } => {
                if modality != Modality::Audio {
                    return Err(MediaError::LocatorModalityMismatch {
                        modality,
                        locator: "audio_span",
                    });
                }
                if end_ms <= start_ms {
                    return Err(MediaError::InvalidLocator("音频区间必须非空且递增"));
                }
                Ok(())
            }
            Locator::TextSpan { start, end } => {
                if !matches!(modality, Modality::Document | Modality::Audio) {
                    return Err(MediaError::LocatorModalityMismatch {
                        modality,
                        locator: "text_span",
                    });
                }
                if end <= start {
                    return Err(MediaError::InvalidLocator("文本区间必须非空且递增"));
                }
                Ok(())
            }
        }
    }

    pub(crate) fn image_side(&self) -> Option<Side> {
        let Locator::ImageRegion {
            x,
            y,
            width,
            height,
        } = self
        else {
            return None;
        };
        let horizontal = (2 * u64::from(*x) + u64::from(*width)) * 3;
        let vertical = (2 * u64::from(*y) + u64::from(*height)) * 3;
        let left = horizontal < 2000;
        let right = horizontal > 4000;
        let top = vertical < 2000;
        let bottom = vertical > 4000;
        match (left, right, top, bottom) {
            (true, false, _, _) => Some(Side::Left),
            (false, true, _, _) => Some(Side::Right),
            (_, _, true, false) => Some(Side::Top),
            (_, _, false, true) => Some(Side::Bottom),
            (false, false, false, false) => Some(Side::Center),
            _ => None,
        }
    }
}

/// 图片方位（千分位几何判定；只用于「左边 / 右边 / 上面…」的定位提示）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Side {
    Left,
    Right,
    Top,
    Bottom,
    Center,
}

impl Side {
    pub fn label_zh(self) -> &'static str {
        match self {
            Side::Left => "左边",
            Side::Right => "右边",
            Side::Top => "上面",
            Side::Bottom => "下面",
            Side::Center => "中间",
        }
    }
}

/// 从用户指称中解析出的定位提示。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LocatorHint {
    TableCell { row: u32, column: u32 },
    TableRow { row: u32 },
    TableColumn { column: u32 },
    ImageSide(Side),
    LastSentence,
    Sentence(usize),
}

impl LocatorHint {
    pub fn label_zh(self) -> String {
        match self {
            LocatorHint::TableCell { row, column } => {
                format!("表格第 {} 行第 {} 列", row, column)
            }
            LocatorHint::TableRow { row } => format!("表格第 {} 行", row),
            LocatorHint::TableColumn { column } => format!("表格第 {} 列", column),
            LocatorHint::ImageSide(side) => side.label_zh().to_string(),
            LocatorHint::LastSentence => "最后一句".to_string(),
            LocatorHint::Sentence(index) => format!("第 {} 句", index),
        }
    }
}

/// 一条符号锚点：媒体上「哪一块」说了什么。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Anchor {
    pub id: AnchorId,
    pub locator: Locator,
    /// 抽取/人工标注出的符号文本（OCR、图注、单元格、转录）。
    pub text: String,
    /// 抽取器名称（`ocr`、`table_parser`、`asr`、`manual`…）。
    pub extractor: String,
    /// 抽取置信度 0..=1；冲突披露时保留，不当作真值。
    pub confidence: f64,
    /// 实体链接结果（可空）。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub entities: Vec<String>,
}

impl Anchor {
    pub fn citation(&self, asset: &AssetRecord) -> MediaCitation {
        MediaCitation {
            asset_id: asset.id.clone(),
            asset_sequence: asset.sequence,
            asset_label: asset.label.clone(),
            content_hash: asset.content_hash.clone(),
            anchor_id: self.id.clone(),
            locator: self.locator.clone(),
            extractor: self.extractor.clone(),
            confidence: self.confidence,
        }
    }

    pub fn matches_hint(&self, hint: &LocatorHint) -> bool {
        match hint {
            LocatorHint::TableCell { row, column } => match &self.locator {
                Locator::TableCell { row: r, column: c } => r == row && c == column,
                Locator::TableRow { row: r } => r == row,
                _ => false,
            },
            LocatorHint::TableRow { row } => match &self.locator {
                Locator::TableCell { row: r, .. } | Locator::TableRow { row: r } => r == row,
                _ => false,
            },
            LocatorHint::TableColumn { column } => match &self.locator {
                Locator::TableCell { column: c, .. } => c == column,
                _ => false,
            },
            LocatorHint::ImageSide(side) => self
                .locator
                .image_side()
                .is_some_and(|actual| actual == *side),
            LocatorHint::LastSentence | LocatorHint::Sentence(_) => !self.text.trim().is_empty(),
        }
    }

    /// 按定位提示抽取要引用的那句话/那个单元格文本。
    pub fn focus_text(&self, hint: Option<&LocatorHint>) -> String {
        match hint {
            Some(LocatorHint::LastSentence) => split_sentences(&self.text)
                .last()
                .map(|sentence| sentence.to_string())
                .unwrap_or_else(|| self.text.clone()),
            Some(LocatorHint::Sentence(index)) => {
                let sentences = split_sentences(&self.text);
                let index = index.saturating_sub(1);
                sentences
                    .get(index)
                    .map(|sentence| sentence.to_string())
                    .unwrap_or_else(|| self.text.clone())
            }
            _ => self.text.clone(),
        }
    }
}

/// 添加锚点的草稿；id 由 `MediaStore` 分配，调用方不能伪造。
#[derive(Debug, Clone, PartialEq)]
pub struct AnchorDraft {
    pub locator: Locator,
    pub text: String,
    pub extractor: String,
    pub confidence: f64,
    pub entities: Vec<String>,
}

impl AnchorDraft {
    pub fn new(
        locator: Locator,
        text: impl Into<String>,
        extractor: impl Into<String>,
        confidence: f64,
    ) -> Self {
        Self {
            locator,
            text: text.into(),
            extractor: extractor.into(),
            confidence,
            entities: Vec::new(),
        }
    }

    pub fn with_entities(mut self, entities: Vec<String>) -> Self {
        self.entities = entities;
        self
    }
}

/// 按中文/英文句末标点切句；保留标点，供引用时逐字回显。
pub fn split_sentences(text: &str) -> Vec<&str> {
    let mut sentences = Vec::new();
    let mut start = 0;
    for (index, character) in text.char_indices() {
        if matches!(character, '。' | '！' | '？' | '!' | '?' | '；' | ';') {
            let end = index + character.len_utf8();
            let sentence = text[start..end].trim();
            if !sentence.is_empty() {
                sentences.push(sentence);
            }
            start = end;
        }
    }
    let tail = text[start..].trim();
    if !tail.is_empty() {
        sentences.push(tail);
    }
    sentences
}

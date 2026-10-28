use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::text_identity::TextIdentity;

use super::anchor::{Anchor, Locator};
use super::asset::{AnchorId, AssetId, AssetRecord, AssetRef, UnavailableReason};
use super::store::MediaStore;

/// 一条带精确定位的媒体引用。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MediaCitation {
    pub asset_id: AssetId,
    pub asset_sequence: u64,
    pub asset_label: String,
    pub content_hash: String,
    pub anchor_id: AnchorId,
    pub locator: Locator,
    pub extractor: String,
    pub confidence: f64,
}

impl MediaCitation {
    pub fn render_zh(&self) -> String {
        format!(
            "{} · {} · {}（置信度 {:.2}）",
            self.asset_label,
            self.locator.label_zh(),
            self.extractor,
            self.confidence
        )
    }

    pub fn asset_ref(&self) -> AssetRef {
        AssetRef {
            asset_id: self.asset_id.clone(),
            content_hash: self.content_hash.clone(),
            sequence: self.asset_sequence,
            label: self.asset_label.clone(),
        }
    }

    /// 用当前媒体库核对引用是否仍可用。
    pub fn resolve<'a>(
        &self,
        store: &'a MediaStore,
    ) -> Result<(&'a AssetRecord, &'a Anchor), UnavailableReason> {
        let record = self.asset_ref().resolve(store)?;
        let anchor = store
            .anchor(&self.asset_id, &self.anchor_id)
            .ok_or(UnavailableReason::Missing)?;
        Ok((record, anchor))
    }
}

/// 一条媒体观察（观察值 + 可核对引用）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MediaClaim {
    pub value: String,
    pub citation: MediaCitation,
}

impl MediaClaim {
    pub fn from_anchor(value: impl Into<String>, asset: &AssetRecord, anchor: &Anchor) -> Self {
        Self {
            value: value.into(),
            citation: anchor.citation(asset),
        }
    }
}

/// 五态证据格。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SupportState {
    /// 只有图谱文本证据。
    TextOnly,
    /// 只有媒体观察，图谱没有对应记录。
    MediaOnly,
    /// 图谱与媒体归一到同一取值。
    Corroborated,
    /// 出现两个及以上不同取值。
    Conflicted,
    /// 两路都没有证据。
    Missing,
}

impl SupportState {
    pub fn code(self) -> &'static str {
        match self {
            SupportState::TextOnly => "text_only",
            SupportState::MediaOnly => "media_only",
            SupportState::Corroborated => "corroborated",
            SupportState::Conflicted => "conflicted",
            SupportState::Missing => "missing",
        }
    }

    pub fn label_zh(self) -> &'static str {
        match self {
            SupportState::TextOnly => "仅图谱",
            SupportState::MediaOnly => "仅媒体",
            SupportState::Corroborated => "相互印证",
            SupportState::Conflicted => "相互冲突",
            SupportState::Missing => "没有证据",
        }
    }

    pub fn is_answerable(self) -> bool {
        !matches!(self, SupportState::Missing)
    }
}

/// 归一到同一个取值键的证据组。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SupportValue {
    pub key: String,
    pub display: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub text_values: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub media: Vec<MediaClaim>,
}

impl SupportValue {
    pub fn has_text(&self) -> bool {
        !self.text_values.is_empty()
    }

    pub fn has_media(&self) -> bool {
        !self.media.is_empty()
    }

    pub fn citations(&self) -> Vec<&MediaCitation> {
        self.media.iter().map(|claim| &claim.citation).collect()
    }
}

/// 一次跨模态支持判定。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SupportReport {
    pub entity: String,
    pub attribute: String,
    pub state: SupportState,
    pub values: Vec<SupportValue>,
    pub text_count: usize,
    pub media_count: usize,
}

impl SupportReport {
    pub fn answerable(&self) -> bool {
        self.state.is_answerable()
    }

    pub fn primary_value(&self) -> Option<&SupportValue> {
        if self.state == SupportState::Conflicted {
            None
        } else {
            self.values.first()
        }
    }

    pub fn citations(&self) -> Vec<&MediaCitation> {
        self.values
            .iter()
            .flat_map(SupportValue::citations)
            .collect()
    }

    /// 固定模板渲染；冲突时并列所有说法，不替用户拍板。
    pub fn render_zh(&self) -> String {
        let subject = subject_phrase(&self.entity, &self.attribute);
        match self.state {
            SupportState::Corroborated => {
                let value = self.values.first();
                let text = value.map(text_values_phrase).unwrap_or_default();
                let media = value.map(media_phrase).unwrap_or_default();
                format!(
                    "{}，图谱与媒体一致：{}。图谱记录 {}；媒体引用 {}。",
                    subject, text, text, media
                )
            }
            SupportState::TextOnly => {
                let text = self
                    .values
                    .first()
                    .map(text_values_phrase)
                    .unwrap_or_default();
                format!(
                    "{}，图谱记录：{}。目前没有可核对的媒体证据。",
                    subject, text
                )
            }
            SupportState::MediaOnly => {
                let value = self.values.first();
                let media = value.map(media_phrase).unwrap_or_default();
                let display = value.map(|value| value.display.as_str()).unwrap_or("");
                format!(
                    "{}，只有媒体证据：『{}』（引用：{}）。图谱中没有对应记录，仅依据你提供的媒体。",
                    subject, display, media
                )
            }
            SupportState::Conflicted => {
                let alternatives = self
                    .values
                    .iter()
                    .map(describe_value)
                    .collect::<Vec<_>>()
                    .join("；");
                let has_text = self.values.iter().any(SupportValue::has_text);
                let has_media = self.values.iter().any(SupportValue::has_media);
                let conflict = match (has_text, has_media) {
                    (true, true) => "图谱与媒体的说法不一致",
                    (true, false) => "图谱自身存在互相矛盾的说法",
                    (false, true) => "媒体证据之间存在互相矛盾的说法",
                    (false, false) => "存在互相矛盾的说法",
                };
                format!(
                    "{}，{}：{}。我不替你拍板，请核实来源。",
                    subject, conflict, alternatives
                )
            }
            SupportState::Missing => format!("{}，当前没有图谱或媒体证据。", subject),
        }
    }
}

fn subject_phrase(entity: &str, attribute: &str) -> String {
    match (entity.is_empty(), attribute.is_empty()) {
        (false, false) => format!("关于「{}」的「{}」", entity, attribute),
        (false, true) => format!("关于「{}」", entity),
        (true, false) => format!("关于「{}」", attribute),
        (true, true) => "关于这个结论".to_string(),
    }
}

fn text_values_phrase(value: &SupportValue) -> String {
    value
        .text_values
        .iter()
        .map(|item| format!("『{}』", item))
        .collect::<Vec<_>>()
        .join("、")
}

fn media_phrase(value: &SupportValue) -> String {
    value
        .media
        .iter()
        .map(|claim| claim.citation.render_zh())
        .collect::<Vec<_>>()
        .join("；")
}

fn describe_value(value: &SupportValue) -> String {
    match (value.has_text(), value.has_media()) {
        (true, true) => format!(
            "图谱与媒体都说『{}』（{}）",
            value.display,
            media_phrase(value)
        ),
        (true, false) => format!("图谱说『{}』", value.display),
        (false, true) => format!("媒体说『{}』（{}）", value.display, media_phrase(value)),
        (false, false) => format!("『{}』", value.display),
    }
}

/// 将取值归一为保守召回键；空白、宽窄、不可见字符与 ASCII 大小写不产生幽灵冲突。
pub fn value_key(value: &str) -> String {
    let identity = TextIdentity::new(value);
    let match_key = identity.match_key();
    if !match_key.is_empty() {
        return match_key.to_string();
    }
    if !identity.identity().is_empty() {
        return identity.identity().to_string();
    }
    let trimmed = value.trim();
    if trimmed.is_empty() {
        "<空>".to_string()
    } else {
        trimmed.to_string()
    }
}

/// 图谱文本与媒体观察 → 五态证据格。
pub fn assess_support(
    entity: impl Into<String>,
    attribute: impl Into<String>,
    text_values: &[String],
    media: &[MediaClaim],
) -> SupportReport {
    let mut groups: BTreeMap<String, SupportValue> = BTreeMap::new();
    for value in text_values {
        let key = value_key(value);
        let group = groups.entry(key.clone()).or_insert_with(|| SupportValue {
            key,
            display: String::new(),
            text_values: Vec::new(),
            media: Vec::new(),
        });
        if !group.text_values.iter().any(|existing| existing == value) {
            group.text_values.push(value.clone());
        }
    }
    for claim in media {
        let key = value_key(&claim.value);
        let group = groups.entry(key.clone()).or_insert_with(|| SupportValue {
            key,
            display: String::new(),
            text_values: Vec::new(),
            media: Vec::new(),
        });
        group.media.push(claim.clone());
    }
    let mut values: Vec<SupportValue> = groups.into_values().collect();
    for value in &mut values {
        value.text_values.sort();
        value.media.sort_by(|left, right| {
            (
                left.citation.asset_sequence,
                &left.citation.anchor_id,
                &left.value,
            )
                .cmp(&(
                    right.citation.asset_sequence,
                    &right.citation.anchor_id,
                    &right.value,
                ))
        });
        value.display = value
            .text_values
            .first()
            .cloned()
            .or_else(|| value.media.first().map(|claim| claim.value.clone()))
            .unwrap_or_default();
    }
    let state = match values.len() {
        0 => SupportState::Missing,
        1 => {
            let value = &values[0];
            match (value.has_text(), value.has_media()) {
                (true, true) => SupportState::Corroborated,
                (true, false) => SupportState::TextOnly,
                (false, true) => SupportState::MediaOnly,
                (false, false) => SupportState::Missing,
            }
        }
        _ => SupportState::Conflicted,
    };
    SupportReport {
        entity: entity.into(),
        attribute: attribute.into(),
        state,
        values,
        text_count: text_values.len(),
        media_count: media.len(),
    }
}

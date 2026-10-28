use super::anchor::{Anchor, LocatorHint};
use super::asset::{AssetRecord, AssetRef, CANDIDATE_DISPLAY_CAP};
use super::selector::{MediaSelector, SelectorOrdinal};
use super::store::MediaStore;

/// 歧义澄清时的编号候选。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaCandidate {
    pub position: usize,
    pub asset: AssetRecord,
    pub anchor_count: usize,
}

impl MediaCandidate {
    pub fn render_zh(&self) -> String {
        format!(
            "{}. {} · {} · {} 个锚点",
            self.position, self.asset.label, self.asset.media_type, self.anchor_count
        )
    }
}

/// 指称解析结果。
#[derive(Debug, Clone, PartialEq)]
pub enum MediaSelection {
    Selected {
        asset: AssetRecord,
        anchors: Vec<Anchor>,
    },
    AssetWithoutAnchor {
        asset: AssetRecord,
        hint: LocatorHint,
    },
    Ambiguous {
        candidates: Vec<MediaCandidate>,
    },
    Missing {
        selector: MediaSelector,
    },
}

impl MediaSelection {
    pub fn asset(&self) -> Option<&AssetRecord> {
        match self {
            MediaSelection::Selected { asset, .. }
            | MediaSelection::AssetWithoutAnchor { asset, .. } => Some(asset),
            _ => None,
        }
    }

    pub fn anchors(&self) -> &[Anchor] {
        match self {
            MediaSelection::Selected { anchors, .. } => anchors,
            _ => &[],
        }
    }

    /// 歧义时给出编号澄清提示；其余情况不插话。
    pub fn clarification_prompt(&self) -> Option<String> {
        let MediaSelection::Ambiguous { candidates } = self else {
            return None;
        };
        let mut lines: Vec<String> = candidates
            .iter()
            .take(CANDIDATE_DISPLAY_CAP)
            .map(MediaCandidate::render_zh)
            .collect();
        if candidates.len() > CANDIDATE_DISPLAY_CAP {
            lines.push(format!(
                "…另有 {} 个候选",
                candidates.len() - CANDIDATE_DISPLAY_CAP
            ));
        }
        Some(format!(
            "你说的媒体有 {} 个候选，请用编号确认：\n{}",
            candidates.len(),
            lines.join("\n")
        ))
    }

    pub fn describe_zh(&self) -> String {
        match self {
            MediaSelection::Selected { asset, anchors } => {
                format!("已选中 {}，匹配 {} 个锚点", asset.display_zh(), anchors.len())
            }
            MediaSelection::AssetWithoutAnchor { asset, hint } => format!(
                "已找到 {}，但没有「{}」对应的锚点",
                asset.display_zh(),
                hint.label_zh()
            ),
            MediaSelection::Ambiguous { candidates } => {
                format!("有 {} 个媒体候选，需要先澄清", candidates.len())
            }
            MediaSelection::Missing { selector } => {
                format!("没有找到与「{}」匹配的媒体", selector.raw)
            }
        }
    }
}

impl MediaStore {
    /// 选择资产与锚点。歧义返回编号候选，绝不静默取第一个。
    pub fn select(
        &self,
        selector: &MediaSelector,
        focus: Option<&AssetRef>,
    ) -> MediaSelection {
        let mut candidates: Vec<&super::store::StoredAsset> = self
            .assets
            .values()
            .filter(|stored| {
                selector.modalities.is_empty()
                    || selector.modalities.contains(&stored.record.modality)
            })
            .collect();
        candidates.sort_by_key(|stored| stored.record.sequence);
        if candidates.is_empty() {
            return MediaSelection::Missing {
                selector: selector.clone(),
            };
        }
        let selected: Option<&super::store::StoredAsset> = match selector.ordinal {
            SelectorOrdinal::Any => {
                let focus_match = focus.and_then(|reference| {
                    candidates.iter().copied().find(|stored| {
                        stored.record.id == reference.asset_id
                            && stored.record.sequence == reference.sequence
                    })
                });
                if let Some(found) = focus_match {
                    Some(found)
                } else if candidates.len() == 1 {
                    candidates.first().copied()
                } else {
                    return MediaSelection::Ambiguous {
                        candidates: candidates
                            .iter()
                            .enumerate()
                            .map(|(index, stored)| MediaCandidate {
                                position: index + 1,
                                asset: stored.record.clone(),
                                anchor_count: stored.anchors.len(),
                            })
                            .collect(),
                    };
                }
            }
            SelectorOrdinal::Latest => candidates.last().copied(),
            SelectorOrdinal::Previous => {
                if let Some(reference) = focus {
                    candidates
                        .iter()
                        .copied()
                        .filter(|stored| stored.record.sequence < reference.sequence)
                        .max_by_key(|stored| stored.record.sequence)
                } else if candidates.len() >= 2 {
                    candidates.get(candidates.len() - 2).copied()
                } else {
                    None
                }
            }
            SelectorOrdinal::Index(position) => position
                .checked_sub(1)
                .and_then(|index| candidates.get(index).copied()),
        };
        let Some(stored) = selected else {
            return MediaSelection::Missing {
                selector: selector.clone(),
            };
        };
        let anchors: Vec<Anchor> = match &selector.locator {
            Some(hint) => stored
                .anchors
                .iter()
                .filter(|anchor| anchor.matches_hint(hint))
                .cloned()
                .collect(),
            None => stored.anchors.clone(),
        };
        if let Some(hint) = selector.locator {
            if anchors.is_empty() {
                return MediaSelection::AssetWithoutAnchor {
                    asset: stored.record.clone(),
                    hint,
                };
            }
        }
        MediaSelection::Selected {
            asset: stored.record.clone(),
            anchors,
        }
    }
}

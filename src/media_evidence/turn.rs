use super::asset::AssetRef;
use super::selection::MediaSelection;
use super::selector::MediaSelector;
use super::session::SessionMedia;
use super::store::MediaStore;
use super::support::{assess_support, MediaCitation, MediaClaim, SupportReport, SupportState};

/// 一次媒体指认 + 跨模态支持判定 + 可上屏回答。
#[derive(Debug, Clone, PartialEq)]
pub struct MediaTurn {
    pub selector: MediaSelector,
    pub selection: MediaSelection,
    pub support: Option<SupportReport>,
    /// 用户可见回答；歧义时就是编号澄清提示。
    pub response: String,
    /// 歧义时的结构化澄清提示（其余情况为空）。
    pub clarification: Option<String>,
    /// 本轮是否成功更新了会话焦点。
    pub focus_updated: bool,
}

impl MediaTurn {
    pub fn state(&self) -> Option<SupportState> {
        self.support.as_ref().map(|support| support.state)
    }

    pub fn citations(&self) -> Vec<&MediaCitation> {
        self.support
            .as_ref()
            .map(SupportReport::citations)
            .unwrap_or_default()
    }

    pub fn asset_ref(&self) -> Option<AssetRef> {
        self.selection.asset().map(AssetRef::from_record)
    }

    pub fn is_grounded(&self) -> bool {
        self.support
            .as_ref()
            .map(SupportReport::answerable)
            .unwrap_or(false)
    }

    pub fn render_zh(&self) -> &str {
        &self.response
    }
}

/// 指认媒体 → 读取锚点 → 与图谱文本做五态判定 → 生成带定位引用的回答。
///
/// 只消费已抽取的符号锚点；不会看图、不会跑 OCR/ASR、不会把媒体观察写回图谱。
pub fn ground_media_turn(
    selector: &MediaSelector,
    store: &MediaStore,
    session: &mut SessionMedia,
    entity: &str,
    attribute: &str,
    text_values: &[String],
) -> MediaTurn {
    let focus = match selector.modalities.first() {
        Some(modality) => session.focus_for(store, *modality),
        None => session.focus().cloned(),
    };
    let selection = store.select(selector, focus.as_ref());
    match &selection {
        MediaSelection::Selected { asset, anchors } => {
            let observations: Vec<MediaClaim> = anchors
                .iter()
                .filter_map(|anchor| {
                    let text = anchor.focus_text(selector.locator.as_ref());
                    if text.trim().is_empty() {
                        None
                    } else {
                        Some(MediaClaim::from_anchor(text, asset, anchor))
                    }
                })
                .collect();
            let support = assess_support(entity, attribute, text_values, &observations);
            let response = if observations.is_empty() {
                format!(
                    "已找到 {}，但它还没有可引用的符号锚点；当前无法用媒体证据回答。",
                    asset.display_zh()
                )
            } else {
                support.render_zh()
            };
            let focus_updated = session.remember(asset).is_ok();
            MediaTurn {
                selector: selector.clone(),
                selection,
                support: Some(support),
                response,
                clarification: None,
                focus_updated,
            }
        }
        MediaSelection::AssetWithoutAnchor { .. } | MediaSelection::Missing { .. } => MediaTurn {
            selector: selector.clone(),
            response: selection.describe_zh(),
            selection,
            support: None,
            clarification: None,
            focus_updated: false,
        },
        MediaSelection::Ambiguous { .. } => {
            let clarification = selection.clarification_prompt();
            let response = clarification
                .clone()
                .unwrap_or_else(|| selection.describe_zh());
            MediaTurn {
                selector: selector.clone(),
                selection,
                support: None,
                response,
                clarification,
                focus_updated: false,
            }
        }
    }
}

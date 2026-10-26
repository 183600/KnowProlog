//! TXA：文本锚点（Text Anchors）——让「位置」自带坐标系，让归一化可逆。
//!
//! 中文输入里同一个 `3..5` 可能指三种不同的东西：原文字节、原文字符、
//! 归一化后字符。调用方拿错坐标系时，轻则高亮错位，重则 `raw[3..5]`
//! 落在多字节字符中间直接 panic。本模块把这三件事拆成显式类型：
//!
//! - [`CharSpan`]：Unicode 标量（`char`）序号区间；
//! - [`ByteSpan`]：UTF-8 字节区间；
//! - [`AnchoredSpan`]：同一条证据在原串与归一化串中的四种坐标一次性给全。
//!
//! [`TextProjection`] 只做**保守归一化**：宽窄折叠、不可见字符剔除、空白
//! 折叠与首尾裁剪。不做 NFKC、不做繁简转换、不改写实体身份——这些会改变
//! 「说的是谁」，应由实体注册表的显式折叠/合并处理。每个输出字符都记住
//! 它来自原文哪一段字符，因此归一化后的正则命中可以安全地投回原文：
//!
//! ```text
//! 原文：  张 三 ？ 身 高        （15 字节）
//! 归一化：张 三 ?  身 高        （13 字节）
//! 归一化字节 7..13  ──投影──▶  原文字节 9..15  = "身高"
//! ```
//!
//! 设计约束：
//!   - 只读：不改变任何调用方的默认行为，投影失败必须显式返回错误；
//!   - 自描述：跨度类型名里写明单位，不再靠注释猜；
//!   - 有界：归一化审计只保留固定数量的编辑，避免异常输入撑爆回执；
//!   - 确定：同一输入得到同一投影与同一编辑序，便于回放与测试。

use std::fmt;

use serde::{Deserialize, Serialize};

/// 文本锚点格式版本。
pub const TEXT_ANCHOR_VERSION: u32 = 1;
/// 归一化审计最多保留的编辑条数；超出只累计计数。
pub const MAX_AUDIT_EDITS: usize = 64;

/// Unicode 标量（`char`）序号区间，左闭右开。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CharSpan {
    pub start: usize,
    pub end: usize,
}

impl CharSpan {
    pub fn new(start: usize, end: usize) -> Self {
        Self {
            start: start.min(end),
            end: end.max(start),
        }
    }

    pub fn chars(self) -> usize {
        self.end.saturating_sub(self.start)
    }

    pub fn is_empty(self) -> bool {
        self.start >= self.end
    }
}

/// UTF-8 字节区间，左闭右开。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ByteSpan {
    pub start: usize,
    pub end: usize,
}

impl ByteSpan {
    pub fn new(start: usize, end: usize) -> Self {
        Self {
            start: start.min(end),
            end: end.max(start),
        }
    }

    pub fn len(self) -> usize {
        self.end.saturating_sub(self.start)
    }

    pub fn is_empty(self) -> bool {
        self.start >= self.end
    }
}

/// 同一条证据在原文与归一化文本中的坐标。
///
/// 四个区间缺一不可：只给字节偏移的接口无法同时服务正则（字节）、
/// 高亮（字符）与切片（字节边界校验）。字段名即单位，避免再猜。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnchoredSpan {
    /// 原文 UTF-8 字节区间。
    pub raw: ByteSpan,
    /// 原文 Unicode 标量区间。
    pub raw_chars: CharSpan,
    /// 归一化文本 UTF-8 字节区间。
    pub normalized: ByteSpan,
    /// 归一化文本 Unicode 标量区间。
    pub normalized_chars: CharSpan,
}

impl AnchoredSpan {
    pub fn raw_text<'a>(&self, raw: &'a str) -> Option<&'a str> {
        raw.get(self.raw.start..self.raw.end)
    }

    pub fn normalized_text<'a>(&self, normalized: &'a str) -> Option<&'a str> {
        normalized.get(self.normalized.start..self.normalized.end)
    }

    pub fn is_empty(&self) -> bool {
        self.raw.is_empty()
    }
}

/// 归一化时发生的一类编辑。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EditKind {
    /// 全角 ASCII 折叠为半角。
    WidthFold,
    /// 剔除不可见字符（零宽空格等）。
    Invisible,
    /// 空白折叠/统一（制表、换行、表意空格 → 单个半角空格）。
    Whitespace,
    /// 首尾空白裁剪。
    Trim,
}

impl EditKind {
    pub fn code(self) -> &'static str {
        match self {
            EditKind::WidthFold => "width_fold",
            EditKind::Invisible => "invisible",
            EditKind::Whitespace => "whitespace",
            EditKind::Trim => "trim",
        }
    }
}

/// 一条归一化编辑：原文区间 → 归一化文本中的落点。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NormalizationEdit {
    pub kind: EditKind,
    pub raw: CharSpan,
    /// 编辑后对应字符在归一化串中的位置；被剔除时为插入点。
    pub normalized: CharSpan,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replacement: Option<char>,
}

/// 一次归一化的可审计回执（不含原文，可安全进入日志）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectionReceipt {
    pub version: u32,
    pub raw_chars: usize,
    pub normalized_chars: usize,
    pub removed_chars: usize,
    pub edits: Vec<NormalizationEdit>,
    /// 超出 [`MAX_AUDIT_EDITS`] 而未逐条保留的编辑数。
    pub omitted_edits: usize,
}

impl ProjectionReceipt {
    pub fn width_folds(&self) -> usize {
        self.edits
            .iter()
            .filter(|edit| edit.kind == EditKind::WidthFold)
            .count()
    }

    pub fn invisible_removed(&self) -> usize {
        self.edits
            .iter()
            .filter(|edit| edit.kind == EditKind::Invisible)
            .count()
    }
}

/// 归一化选项。默认值即 [`TextProjection::new`] 使用的保守策略。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProjectionOptions {
    pub fold_width: bool,
    pub strip_invisible: bool,
    pub collapse_whitespace: bool,
    pub trim: bool,
}

impl Default for ProjectionOptions {
    fn default() -> Self {
        Self {
            fold_width: true,
            strip_invisible: true,
            collapse_whitespace: true,
            trim: true,
        }
    }
}

/// 投影失败的原因。调用方拿不到「大概对」的区间：越界/半个字符必须报错。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnchorError {
    /// 字节偏移不在 UTF-8 字符边界上。
    NotCharBoundary { offset: usize },
    /// 区间超出文本长度。
    OutOfBounds { offset: usize, len: usize },
}

impl fmt::Display for AnchorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AnchorError::NotCharBoundary { offset } => {
                write!(f, "字节偏移 {} 不在字符边界上", offset)
            }
            AnchorError::OutOfBounds { offset, len } => {
                write!(f, "字节偏移 {} 超出文本长度 {}", offset, len)
            }
        }
    }
}

impl std::error::Error for AnchorError {}

/// 原文 → 归一化文本的投影表。
///
/// 生命周期绑定原文，避免复制长文本；归一化串只在构造时生成一次。
pub struct TextProjection<'a> {
    raw: &'a str,
    normalized: String,
    /// 每个归一化字符对应的原文 Unicode 标量区间。
    sources: Vec<CharSpan>,
    edits: Vec<NormalizationEdit>,
    omitted_edits: usize,
    raw_char_bytes: Vec<usize>,
    normalized_char_bytes: Vec<usize>,
}

impl<'a> TextProjection<'a> {
    pub fn new(raw: &'a str, options: ProjectionOptions) -> Self {
        let raw_len = raw.chars().count();
        let mut raw_char_bytes = Vec::with_capacity(raw_len + 1);
        raw_char_bytes.push(0);
        for (offset, ch) in raw.char_indices() {
            raw_char_bytes.push(offset + ch.len_utf8());
        }

        let mut normalized = String::new();
        let mut sources = Vec::new();
        let mut edits = Vec::new();
        let mut omitted_edits = 0usize;
        let mut pending_space: Option<CharSpan> = None;

        let emit_edit = |edit: NormalizationEdit,
                             edits: &mut Vec<NormalizationEdit>,
                             omitted: &mut usize| {
            if edits.len() < MAX_AUDIT_EDITS {
                edits.push(edit);
            } else {
                *omitted += 1;
            }
        };

        for (index, ch) in raw.chars().enumerate() {
            if options.strip_invisible && is_invisible(ch) {
                emit_edit(
                    NormalizationEdit {
                        kind: EditKind::Invisible,
                        raw: CharSpan::new(index, index + 1),
                        normalized: CharSpan::new(sources.len(), sources.len()),
                        replacement: None,
                    },
                    &mut edits,
                    &mut omitted_edits,
                );
                continue;
            }

            if options.collapse_whitespace && ch.is_whitespace() {
                pending_space = Some(match pending_space {
                    Some(span) => CharSpan::new(span.start, index + 1),
                    None => CharSpan::new(index, index + 1),
                });
                continue;
            }

            if let Some(span) = pending_space.take() {
                if normalized.is_empty() {
                    if options.trim {
                        emit_edit(
                            NormalizationEdit {
                                kind: EditKind::Trim,
                                raw: span,
                                normalized: CharSpan::new(0, 0),
                                replacement: None,
                            },
                            &mut edits,
                            &mut omitted_edits,
                        );
                    } else {
                        push_char(
                            &mut normalized,
                            &mut sources,
                            ' ',
                            span,
                            Some(EditKind::Whitespace),
                            &mut edits,
                            &mut omitted_edits,
                        );
                    }
                } else {
                    push_char(
                        &mut normalized,
                        &mut sources,
                        ' ',
                        span,
                        Some(EditKind::Whitespace),
                        &mut edits,
                        &mut omitted_edits,
                    );
                }
            }

            let folded = if options.fold_width {
                fold_width(ch)
            } else {
                None
            };
            let out = folded.unwrap_or(ch);
            push_char(
                &mut normalized,
                &mut sources,
                out,
                CharSpan::new(index, index + 1),
                folded.map(|_| EditKind::WidthFold),
                &mut edits,
                &mut omitted_edits,
            );
        }

        if let Some(span) = pending_space.take() {
            if options.trim {
                let point = sources.len();
                emit_edit(
                    NormalizationEdit {
                        kind: EditKind::Trim,
                        raw: span,
                        normalized: CharSpan::new(point, point),
                        replacement: None,
                    },
                    &mut edits,
                    &mut omitted_edits,
                );
            } else {
                push_char(
                    &mut normalized,
                    &mut sources,
                    ' ',
                    span,
                    Some(EditKind::Whitespace),
                    &mut edits,
                    &mut omitted_edits,
                );
            }
        }

        let mut normalized_char_bytes = Vec::with_capacity(sources.len() + 1);
        normalized_char_bytes.push(0);
        for (offset, ch) in normalized.char_indices() {
            normalized_char_bytes.push(offset + ch.len_utf8());
        }

        Self {
            raw,
            normalized,
            sources,
            edits,
            omitted_edits,
            raw_char_bytes,
            normalized_char_bytes,
        }
    }

    pub fn raw(&self) -> &'a str {
        self.raw
    }

    pub fn normalized(&self) -> &str {
        &self.normalized
    }

    pub fn receipt(&self) -> ProjectionReceipt {
        ProjectionReceipt {
            version: TEXT_ANCHOR_VERSION,
            raw_chars: self.raw_char_bytes.len().saturating_sub(1),
            normalized_chars: self.sources.len(),
            removed_chars: self
                .raw_char_bytes
                .len()
                .saturating_sub(1)
                .saturating_sub(self.sources.len()),
            edits: self.edits.clone(),
            omitted_edits: self.omitted_edits,
        }
    }

    /// 把归一化串中的字节区间投回原文的四种坐标。
    ///
    /// 供正则/字节索引调用方使用；`start`/`end` 必须是归一化串的字符边界。
    pub fn surface_bytes_to_raw(&self, span: ByteSpan) -> Result<AnchoredSpan, AnchorError> {
        let chars = self.surface_bytes_to_chars(span)?;
        self.surface_chars_to_raw(chars)
    }

    /// 把归一化串中的字符区间投回原文。
    pub fn surface_chars_to_raw(&self, span: CharSpan) -> Result<AnchoredSpan, AnchorError> {
        let surface = self.clamp_chars(span)?;
        let raw_chars = self.surface_chars_to_raw_chars(surface);
        self.build_anchor(surface, raw_chars)
    }

    /// 把原文中的字节区间投影到归一化串（用于「原文高亮 → 归一化匹配」）。
    pub fn raw_bytes_to_surface(&self, span: ByteSpan) -> Result<AnchoredSpan, AnchorError> {
        let chars = self.raw_bytes_to_chars(span)?;
        let surface_chars = self.raw_chars_to_surface_chars(chars);
        self.build_anchor(surface_chars, chars)
    }

    /// 安全切出原文字符串；越界或半个字符返回 `None`。
    pub fn raw_slice(&self, span: ByteSpan) -> Option<&'a str> {
        self.raw.get(span.start..span.end)
    }

    /// 安全切出归一化字符串；越界或半个字符返回 `None`。
    pub fn normalized_slice(&self, span: ByteSpan) -> Option<&str> {
        self.normalized.get(span.start..span.end)
    }

    /// 原文某个位置附近在归一化串中的落点（插入点也返回，不 panic）。
    pub fn raw_point_to_surface(&self, raw_char: usize) -> CharSpan {
        let point = CharSpan::new(raw_char, raw_char);
        self.raw_chars_to_surface_chars(point)
    }

    fn build_anchor(
        &self,
        surface: CharSpan,
        raw: CharSpan,
    ) -> Result<AnchoredSpan, AnchorError> {
        let normalized = self.chars_to_bytes(&self.normalized_char_bytes, surface)?;
        let raw_bytes = self.chars_to_bytes(&self.raw_char_bytes, raw)?;
        Ok(AnchoredSpan {
            raw: ByteSpan::new(raw_bytes.start, raw_bytes.end),
            raw_chars: raw,
            normalized: ByteSpan::new(normalized.start, normalized.end),
            normalized_chars: surface,
        })
    }

    fn surface_bytes_to_chars(&self, span: ByteSpan) -> Result<CharSpan, AnchorError> {
        self.bytes_to_chars(&self.normalized_char_bytes, span, self.normalized.len())
    }

    fn raw_bytes_to_chars(&self, span: ByteSpan) -> Result<CharSpan, AnchorError> {
        self.bytes_to_chars(&self.raw_char_bytes, span, self.raw.len())
    }

    fn bytes_to_chars(
        &self,
        starts: &[usize],
        span: ByteSpan,
        len: usize,
    ) -> Result<CharSpan, AnchorError> {
        if span.start > len {
            return Err(AnchorError::OutOfBounds {
                offset: span.start,
                len,
            });
        }
        if span.end > len {
            return Err(AnchorError::OutOfBounds {
                offset: span.end,
                len,
            });
        }
        let start = position_at(starts, span.start).ok_or(AnchorError::NotCharBoundary {
            offset: span.start,
        })?;
        let end = position_at(starts, span.end).ok_or(AnchorError::NotCharBoundary {
            offset: span.end,
        })?;
        Ok(CharSpan::new(start, end))
    }

    fn chars_to_bytes(&self, starts: &[usize], span: CharSpan) -> Result<ByteSpan, AnchorError> {
        if span.start > starts.len().saturating_sub(1) {
            return Err(AnchorError::OutOfBounds {
                offset: span.start,
                len: starts.len().saturating_sub(1),
            });
        }
        if span.end > starts.len().saturating_sub(1) {
            return Err(AnchorError::OutOfBounds {
                offset: span.end,
                len: starts.len().saturating_sub(1),
            });
        }
        Ok(ByteSpan::new(starts[span.start], starts[span.end]))
    }

    fn clamp_chars(&self, span: CharSpan) -> Result<CharSpan, AnchorError> {
        let len = self.sources.len();
        if span.start > len {
            return Err(AnchorError::OutOfBounds {
                offset: span.start,
                len,
            });
        }
        if span.end > len {
            return Err(AnchorError::OutOfBounds {
                offset: span.end,
                len,
            });
        }
        Ok(CharSpan::new(span.start, span.end))
    }

    fn surface_chars_to_raw_chars(&self, surface: CharSpan) -> CharSpan {
        if self.sources.is_empty() {
            return CharSpan::new(0, 0);
        }
        if surface.is_empty() {
            let point = if surface.start < self.sources.len() {
                self.sources[surface.start].start
            } else {
                self.sources.last().map(|span| span.end).unwrap_or(0)
            };
            return CharSpan::new(point, point);
        }
        let start = self.sources[surface.start].start;
        let end = self.sources[surface.end - 1].end;
        CharSpan::new(start, end)
    }

    fn raw_chars_to_surface_chars(&self, raw: CharSpan) -> CharSpan {
        let len = self.sources.len();
        if len == 0 {
            return CharSpan::new(0, 0);
        }
        let first = self.sources.partition_point(|span| span.end <= raw.start);
        if first >= len {
            return CharSpan::new(len, len);
        }
        let last = self.sources.partition_point(|span| span.start < raw.end);
        if first < last {
            CharSpan::new(first, last)
        } else if raw.is_empty() || self.sources[first].start >= raw.end {
            CharSpan::new(first, first)
        } else {
            CharSpan::new(last, last)
        }
    }
}

fn push_char(
    normalized: &mut String,
    sources: &mut Vec<CharSpan>,
    ch: char,
    raw: CharSpan,
    edit: Option<EditKind>,
    edits: &mut Vec<NormalizationEdit>,
    omitted_edits: &mut usize,
) {
    let normalized_span = CharSpan::new(sources.len(), sources.len() + 1);
    if let Some(kind) = edit {
        let replacement = if kind == EditKind::Whitespace { Some(' ') } else { Some(ch) };
        let edit = NormalizationEdit {
            kind,
            raw,
            normalized: normalized_span,
            replacement,
        };
        if edits.len() < MAX_AUDIT_EDITS {
            edits.push(edit);
        } else {
            *omitted_edits += 1;
        }
    }
    normalized.push(ch);
    sources.push(raw);
}

fn position_at(starts: &[usize], offset: usize) -> Option<usize> {
    starts.binary_search(&offset).ok()
}

fn is_invisible(ch: char) -> bool {
    matches!(
        ch,
        '\u{00AD}' | '\u{180E}' | '\u{200B}' | '\u{200C}' | '\u{2060}' | '\u{FEFF}'
    )
}

fn fold_width(ch: char) -> Option<char> {
    match ch {
        '\u{FF01}'..='\u{FF5E}' => char::from_u32(ch as u32 - 0xFEE0),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fullwidth_punctuation_is_folded_but_projects_back() {
        let raw = "张三？身高";
        let projection = TextProjection::new(raw, ProjectionOptions::default());
        assert_eq!(projection.normalized(), "张三?身高");

        let surface = projection.normalized().find("身高").unwrap();
        let anchor = projection
            .surface_bytes_to_raw(ByteSpan::new(surface, surface + "身高".len()))
            .expect("应能投影");
        assert_eq!(anchor.raw_text(raw), Some("身高"));
        assert_eq!(anchor.raw, ByteSpan::new(9, 15));
        assert_eq!(anchor.raw_chars, CharSpan::new(3, 5));
        assert_eq!(anchor.normalized, ByteSpan::new(7, 13));
        assert_eq!(anchor.normalized_text(projection.normalized()), Some("身高"));
    }

    #[test]
    fn zero_width_removal_keeps_raw_evidence_contiguous() {
        let raw = "张\u{200B}三的身高";
        let projection = TextProjection::new(raw, ProjectionOptions::default());
        assert_eq!(projection.normalized(), "张三的身高");

        let anchor = projection
            .surface_bytes_to_raw(ByteSpan::new(0, "张三".len()))
            .expect("应能投影");
        assert_eq!(anchor.raw_text(raw), Some("张\u{200B}三"));
        assert_eq!(anchor.raw_chars, CharSpan::new(0, 3));
        assert_eq!(anchor.normalized, ByteSpan::new(0, 6));

        let receipt = projection.receipt();
        assert_eq!(receipt.invisible_removed(), 1);
        assert_eq!(receipt.edits[0].kind, EditKind::Invisible);
    }

    #[test]
    fn whitespace_collapse_maps_whole_run_to_one_surface_space() {
        let raw = "张三\n\n身高";
        let projection = TextProjection::new(raw, ProjectionOptions::default());
        assert_eq!(projection.normalized(), "张三 身高");

        let start = projection.normalized().find("身高").unwrap();
        let anchor = projection
            .surface_bytes_to_raw(ByteSpan::new(start, start + "身高".len()))
            .expect("应能投影");
        assert_eq!(anchor.raw_text(raw), Some("身高"));
        assert_eq!(anchor.raw_chars, CharSpan::new(4, 6));

        let surface_space = projection.raw_bytes_to_surface(ByteSpan::new(6, 8));
        let anchor = surface_space.expect("换行应投影到那个空格");
        assert_eq!(anchor.normalized_chars, CharSpan::new(2, 3));
        assert_eq!(anchor.normalized_text(projection.normalized()), Some(" "));
    }

    #[test]
    fn leading_and_trailing_trim_do_not_skew_offsets() {
        let raw = "  张三  ";
        let projection = TextProjection::new(raw, ProjectionOptions::default());
        assert_eq!(projection.normalized(), "张三");

        let anchor = projection
            .surface_bytes_to_raw(ByteSpan::new(0, "张三".len()))
            .expect("应能投影");
        assert_eq!(anchor.raw_text(raw), Some("张三"));
        assert_eq!(anchor.raw, ByteSpan::new(2, 8));
        assert_eq!(projection.receipt().edits.iter().filter(|e| e.kind == EditKind::Trim).count(), 2);
    }

    #[test]
    fn fullwidth_digits_fold_without_losing_original_text() {
        let raw = "年龄４２岁";
        let projection = TextProjection::new(raw, ProjectionOptions::default());
        assert_eq!(projection.normalized(), "年龄42岁");

        let start = projection.normalized().find("42").unwrap();
        let anchor = projection
            .surface_bytes_to_raw(ByteSpan::new(start, start + 2))
            .expect("应能投影");
        assert_eq!(anchor.raw_text(raw), Some("４２"));
        assert_eq!(anchor.raw, ByteSpan::new(6, 12));
        assert_eq!(projection.receipt().width_folds(), 2);
    }

    #[test]
    fn multibyte_boundaries_are_checked_not_silently_clamped() {
        let raw = "😀张三";
        let projection = TextProjection::new(raw, ProjectionOptions::default());
        assert!(projection
            .surface_bytes_to_raw(ByteSpan::new(1, 5))
            .is_err());
        let anchor = projection
            .surface_bytes_to_raw(ByteSpan::new(4, 10))
            .expect("应从表情后的字符边界开始");
        assert_eq!(anchor.raw_text(raw), Some("张三"));
        assert_eq!(anchor.raw_chars, CharSpan::new(1, 3));
    }

    #[test]
    fn normalization_is_idempotent_and_receipts_are_stable() {
        let raw = " 张\u{200B}三？ 身高 ";
        let first = TextProjection::new(raw, ProjectionOptions::default());
        let second = TextProjection::new(first.normalized(), ProjectionOptions::default());
        assert_eq!(first.normalized(), second.normalized());
        assert_eq!(second.receipt().removed_chars, 0);
        assert_eq!(first.receipt(), first.receipt());
    }

    #[test]
    fn out_of_bounds_and_empty_projection_are_explicit() {
        let projection = TextProjection::new("张三", ProjectionOptions::default());
        let err = projection
            .surface_bytes_to_raw(ByteSpan::new(0, 7))
            .unwrap_err();
        assert!(matches!(err, AnchorError::OutOfBounds { .. }));

        let empty = TextProjection::new("", ProjectionOptions::default());
        assert_eq!(empty.normalized(), "");
        let anchor = empty
            .surface_bytes_to_raw(ByteSpan::new(0, 0))
            .expect("空文本空区间应可投影");
        assert_eq!(anchor, AnchoredSpan::default());
    }
}

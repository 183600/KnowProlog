//! TXI：文本身份层（Text Identity Layer）。
//!
//! 同一段中文在输入、日志、缓存与共享记忆里可能带着不同的「非语义噪声」：
//! BOM、零宽空格、全角标点、重复空白、ASCII 大小写。它们看起来不同、哈希
//! 也不同，但指的是同一个实体或同一条声明。反过来，用户看到的字面又绝不能
//! 被归一化改写——否则回显、引用与审计都会失真。
//!
//! 本模块把三件事拆成显式契约：
//!
//! - [`decode_utf8_strict`]：非法字节返回带绝对偏移的类型化错误，绝不静默
//!   替换成 `�`；
//! - [`Utf8StreamDecoder`]：任意字节分块（SSE、HTTP body、文件尾）都能增量
//!   解码，跨块的半个汉字不会 panic，也永远不会被截断成错误文本；
//! - [`TextIdentity`]：同时给出 `display`（原样回显）、`identity`（保守身份键）
//!   与 `match_key`（激进召回键）。去重/缓存写身份键，模糊召回只读匹配键。
//!
//! 归一化是保守的：宽窄折叠、不可见字符剔除、空白折叠、Unicode 小写。它
//! **不做** NFC/NFKC、繁简转换或拼音折叠——那些会改变「说的是谁」，必须由
//! 实体注册表的显式合并处理。`identity` 保留标点与词边界；`match_key` 才
//! 剔除标点与空白，只用于召回，不写回身份。
//!
//! 设计约束：零新依赖、确定性、幂等；显示文本只读。

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::context_image::fnv1a_hex;

/// 文本身份层格式版本。
pub const TEXT_IDENTITY_VERSION: u32 = 1;

/// UTF-8 解码失败的类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Utf8ErrorKind {
    /// 非法起始字节或非法续字节序列。
    InvalidSequence,
    /// 输入在结尾处截断（还差续字节）。
    IncompleteSequence,
}

impl Utf8ErrorKind {
    pub fn code(self) -> &'static str {
        match self {
            Utf8ErrorKind::InvalidSequence => "invalid_sequence",
            Utf8ErrorKind::IncompleteSequence => "incomplete_sequence",
        }
    }
}

/// 严格的 UTF-8 解码错误：偏移是相对整段输入的绝对字节位置。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Utf8DecodeError {
    pub kind: Utf8ErrorKind,
    /// 首个非法字节的绝对偏移。
    pub offset: usize,
    /// 非法序列长度；截断时为 `None`。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub len: Option<usize>,
}

impl Utf8DecodeError {
    pub fn invalid(offset: usize, len: usize) -> Self {
        Self {
            kind: Utf8ErrorKind::InvalidSequence,
            offset,
            len: Some(len),
        }
    }

    pub fn incomplete(offset: usize, remaining: usize) -> Self {
        Self {
            kind: Utf8ErrorKind::IncompleteSequence,
            offset,
            len: (remaining > 0).then_some(remaining),
        }
    }

    pub fn code(&self) -> &'static str {
        self.kind.code()
    }

    pub fn end(&self) -> usize {
        self.offset.saturating_add(self.len.unwrap_or(0))
    }
}

impl fmt::Display for Utf8DecodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            Utf8ErrorKind::InvalidSequence => write!(
                formatter,
                "非法 UTF-8 字节序列（偏移 {}，长度 {}）",
                self.offset,
                self.len.unwrap_or(0)
            ),
            Utf8ErrorKind::IncompleteSequence => {
                write!(formatter, "UTF-8 序列在偏移 {} 处被截断", self.offset)
            }
        }
    }
}

impl std::error::Error for Utf8DecodeError {}

/// 严格解码整段字节；非法序列绝不替换为 `�`。
pub fn decode_utf8_strict(bytes: &[u8]) -> Result<&str, Utf8DecodeError> {
    std::str::from_utf8(bytes).map_err(|error| match error.error_len() {
        Some(len) => Utf8DecodeError::invalid(error.valid_up_to(), len),
        None => Utf8DecodeError::incomplete(
            error.valid_up_to(),
            bytes.len().saturating_sub(error.valid_up_to()),
        ),
    })
}

/// 增量 UTF-8 解码器：跨块保留至多 3 个未完成字节。
///
/// 出现非法序列后解码器进入毒化状态：后续 `push` 与 `finish` 返回同一错误，
/// 避免调用方在错误之后继续拼出「半对」的文本。
#[derive(Debug, Clone, Default)]
pub struct Utf8StreamDecoder {
    pending: Vec<u8>,
    /// 已成功输出的字节数（绝对偏移）。
    consumed: usize,
    /// 已接收的字节数（含未完成尾巴）。
    received: usize,
    poisoned: Option<Utf8DecodeError>,
}

impl Utf8StreamDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// 已成功输出的字节数。
    pub fn consumed(&self) -> usize {
        self.consumed
    }

    /// 已接收的字节总数。
    pub fn received(&self) -> usize {
        self.received
    }

    /// 尚未凑成完整字符的尾巴长度（0..=3）。
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    pub fn error(&self) -> Option<Utf8DecodeError> {
        self.poisoned
    }

    /// 送入一个字节块，返回本次可安全输出的文本（可能为空）。
    pub fn push(&mut self, chunk: &[u8]) -> Result<String, Utf8DecodeError> {
        if let Some(error) = self.poisoned {
            return Err(error);
        }
        self.received = self.received.saturating_add(chunk.len());
        if chunk.is_empty() {
            return Ok(String::new());
        }
        self.pending.extend_from_slice(chunk);
        match std::str::from_utf8(&self.pending) {
            Ok(text) => {
                let text = text.to_string();
                self.consumed = self.consumed.saturating_add(self.pending.len());
                self.pending.clear();
                Ok(text)
            }
            Err(error) => {
                let valid = error.valid_up_to();
                let emitted = match std::str::from_utf8(&self.pending[..valid]) {
                    Ok(prefix) => prefix.to_string(),
                    Err(_) => {
                        let offset = self.consumed;
                        self.pending.clear();
                        let failure = Utf8DecodeError::invalid(offset, valid.max(1));
                        self.poisoned = Some(failure);
                        return Err(failure);
                    }
                };
                match error.error_len() {
                    Some(len) => {
                        let offset = self.consumed.saturating_add(valid);
                        self.consumed = offset;
                        self.pending.clear();
                        let failure = Utf8DecodeError::invalid(offset, len);
                        self.poisoned = Some(failure);
                        Err(failure)
                    }
                    None => {
                        self.consumed = self.consumed.saturating_add(valid);
                        self.pending.drain(..valid);
                        Ok(emitted)
                    }
                }
            }
        }
    }

    /// 结束输入：若仍留有半个字符，返回 `IncompleteSequence`。
    pub fn finish(&mut self) -> Result<(), Utf8DecodeError> {
        if let Some(error) = self.poisoned {
            return Err(error);
        }
        if self.pending.is_empty() {
            return Ok(());
        }
        let failure = Utf8DecodeError::incomplete(self.consumed, self.pending.len());
        self.poisoned = Some(failure);
        Err(failure)
    }
}

/// 一段文本的三种键：显示、身份、召回。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextIdentity {
    version: u32,
    display: String,
    identity: String,
    match_key: String,
}

impl TextIdentity {
    pub fn new(display: impl Into<String>) -> Self {
        let display = display.into();
        let identity = identity_form(&display);
        let match_key = match_form(&identity);
        Self {
            version: TEXT_IDENTITY_VERSION,
            display,
            identity,
            match_key,
        }
    }

    pub fn version(&self) -> u32 {
        self.version
    }

    /// 用户看到的原始字面；任何情况下都不改写。
    pub fn display(&self) -> &str {
        &self.display
    }

    /// 保守身份键：可用于去重、缓存与共享记忆的身份。
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// 激进召回键：只用于模糊查找；不得写回身份。
    pub fn match_key(&self) -> &str {
        &self.match_key
    }

    /// 身份键的稳定摘要（FNV-1a 64）。
    pub fn identity_digest(&self) -> String {
        fnv1a_hex(self.identity.as_bytes())
    }

    pub fn same_identity(&self, other: &Self) -> bool {
        self.identity == other.identity
    }

    pub fn same_recall(&self, other: &Self) -> bool {
        self.match_key == other.match_key
    }

    pub fn is_empty(&self) -> bool {
        self.identity.is_empty()
    }
}

fn identity_form(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut pending_space = false;
    for ch in text.chars() {
        if is_invisible(ch) {
            continue;
        }
        let folded = fold_width(ch).unwrap_or(ch);
        if folded.is_whitespace() {
            if !output.is_empty() {
                pending_space = true;
            }
            continue;
        }
        if pending_space {
            output.push(' ');
            pending_space = false;
        }
        for lower in folded.to_lowercase() {
            output.push(lower);
        }
    }
    output
}

fn match_form(identity: &str) -> String {
    let mut output = String::with_capacity(identity.len());
    for ch in identity.chars() {
        if ch.is_whitespace() || is_recall_noise(ch) {
            continue;
        }
        output.push(ch);
    }
    output
}

fn is_recall_noise(ch: char) -> bool {
    ch.is_ascii_punctuation()
        || matches!(
            ch,
            '。' | '，'
                | '、'
                | '；'
                | '：'
                | '？'
                | '！'
                | '“'
                | '”'
                | '‘'
                | '’'
                | '（'
                | '）'
                | '【'
                | '】'
                | '《'
                | '》'
                | '「'
                | '」'
                | '『'
                | '』'
                | '…'
                | '—'
                | '–'
                | '·'
                | '～'
        )
}

fn is_invisible(ch: char) -> bool {
    // C0/C1 控制字符（含 NUL、分隔符、换行/制表）不参与身份键：
    // 否则它们既能制造「显示不同、身份相同」的幽灵，也能伪造内部分隔符。
    if ch.is_control() {
        return true;
    }
    matches!(
        ch,
        '\u{00AD}'
            | '\u{034F}'
            | '\u{061C}'
            | '\u{115F}'
            | '\u{1160}'
            | '\u{17B4}'
            | '\u{17B5}'
            | '\u{180B}'..='\u{180E}'
            | '\u{200B}'..='\u{200F}'
            | '\u{202A}'..='\u{202E}'
            | '\u{2060}'..='\u{2064}'
            | '\u{2066}'..='\u{206F}'
            | '\u{3164}'
            | '\u{FE00}'..='\u{FE0F}'
            | '\u{FEFF}'
            | '\u{FFA0}'
            | '\u{FFF0}'..='\u{FFF8}'
            | '\u{1BCA0}'..='\u{1BCA3}'
            | '\u{1D173}'..='\u{1D17A}'
            | '\u{E0100}'..='\u{E01EF}'
    )
}

fn fold_width(ch: char) -> Option<char> {
    match ch {
        '\u{3000}' => Some(' '),
        '\u{FF01}'..='\u{FF5E}' => char::from_u32(ch as u32 - 0xFEE0),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_decode_reports_absolute_offsets_and_never_replaces() {
        assert_eq!(decode_utf8_strict("张三".as_bytes()).unwrap(), "张三");

        let error = decode_utf8_strict(b"\xE5\xBC\xA0\xFF").unwrap_err();
        assert_eq!(error.kind, Utf8ErrorKind::InvalidSequence);
        assert_eq!(error.offset, 3);
        assert_eq!(error.len, Some(1));

        let truncated = decode_utf8_strict(b"\xE5\xBC").unwrap_err();
        assert_eq!(truncated.kind, Utf8ErrorKind::IncompleteSequence);
        assert_eq!(truncated.offset, 0);
        assert!(!truncated.to_string().contains('�'));
    }

    #[test]
    fn streaming_decoder_accepts_every_byte_boundary() {
        let text = "张三？身高 42😀岁";
        let bytes = text.as_bytes();
        let mut decoder = Utf8StreamDecoder::new();
        let mut output = String::new();
        for byte in bytes {
            output.push_str(&decoder.push(std::slice::from_ref(byte)).unwrap());
        }
        decoder.finish().unwrap();
        assert_eq!(output, text);
        assert_eq!(decoder.consumed(), bytes.len());
        assert_eq!(decoder.received(), bytes.len());
        assert_eq!(decoder.pending_len(), 0);
    }

    #[test]
    fn streaming_decoder_keeps_incomplete_tail_until_next_chunk() {
        let bytes = "张三".as_bytes();
        let split = 4;
        let mut decoder = Utf8StreamDecoder::new();
        assert_eq!(decoder.push(&bytes[..split]).unwrap(), "张");
        assert_eq!(decoder.pending_len(), 1);
        assert_eq!(decoder.push(&bytes[split..]).unwrap(), "三");
        decoder.finish().unwrap();
    }

    #[test]
    fn streaming_decoder_reports_cross_chunk_error_offsets() {
        let mut decoder = Utf8StreamDecoder::new();
        assert_eq!(decoder.push(b"ab").unwrap(), "ab");
        let error = decoder.push(b"\xFF").unwrap_err();
        assert_eq!(error.offset, 2);
        assert_eq!(error.kind, Utf8ErrorKind::InvalidSequence);
        assert_eq!(decoder.push(b"c").unwrap_err(), error, "毒化后错误稳定");
    }

    #[test]
    fn streaming_decoder_finish_detects_truncation() {
        let mut decoder = Utf8StreamDecoder::new();
        assert_eq!(decoder.push(b"\xE5\xBC").unwrap(), "");
        let error = decoder.finish().unwrap_err();
        assert_eq!(error.kind, Utf8ErrorKind::IncompleteSequence);
        assert_eq!(error.offset, 0);
    }

    #[test]
    fn identity_folds_noise_but_display_is_untouched() {
        let left = TextIdentity::new("  张\u{200B}三？ ");
        let right = TextIdentity::new("张三?");
        assert_eq!(left.display(), "  张\u{200B}三？ ");
        assert_eq!(left.identity(), "张三?");
        assert_eq!(right.display(), "张三?");
        assert!(left.same_identity(&right));
        assert!(left.same_recall(&right));
    }

    #[test]
    fn identity_folds_bom_width_and_ascii_case() {
        let left = TextIdentity::new("\u{FEFF}ＡＢＣ　公司");
        let right = TextIdentity::new("abc 公司");
        assert_eq!(left.identity(), "abc 公司");
        assert!(left.same_identity(&right));
        assert_eq!(left.identity_digest(), right.identity_digest());
    }

    #[test]
    fn identity_keeps_punctuation_while_match_key_strips_it() {
        let left = TextIdentity::new("张三，你好。");
        let right = TextIdentity::new("张三你好");
        assert_ne!(left.identity(), right.identity());
        assert_eq!(left.match_key(), "张三你好");
        assert!(left.same_recall(&right));
        assert!(!left.same_identity(&right));
    }

    #[test]
    fn identity_is_idempotent_and_deterministic() {
        let first = TextIdentity::new(" 张\u{200B}三？？ 身高 ");
        let second = TextIdentity::new(first.identity());
        assert_eq!(first.identity(), second.identity());
        assert_eq!(first.match_key(), second.match_key());
        assert_eq!(first, TextIdentity::new(" 张\u{200B}三？？ 身高 "));
        assert_eq!(first.version(), TEXT_IDENTITY_VERSION);
    }

    #[test]
    fn control_characters_cannot_forge_internal_separators() {
        let clean = TextIdentity::new("张三");
        let forged = TextIdentity::new("张\u{1}三");
        assert!(clean.same_identity(&forged));
        assert_eq!(forged.identity(), "张三");
        assert_eq!(forged.display(), "张\u{1}三");
    }

    #[test]
    fn empty_and_whitespace_only_text_has_empty_identity() {
        let identity = TextIdentity::new(" \u{200B}\u{FEFF} \n");
        assert!(identity.is_empty());
        assert_eq!(identity.identity(), "");
        assert_eq!(identity.match_key(), "");
        assert_eq!(identity.display(), " \u{200B}\u{FEFF} \n");
    }
}

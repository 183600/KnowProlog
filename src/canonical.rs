//! CAN：规范字节与解码宪法（Canonical Bytes & Decode Constitution）。
//!
//! 符号系统相对黑箱 KV 的机会不在规模，而在「记忆」是受定律约束的数据：
//! 外部字节 → 内部状态（解码）必须有版本门、错误分类学与回执；内部状态 →
//! 字节（编码）必须只产出唯一规范形。合起来才是一条可断言的闭环：
//!
//! - **L1 往返恒等**：`decode(encode(x)) ≡ x`
//! - **L2 字节不动点**：`encode(decode(b)) ≡ b`（对规范字节 `b`）
//! - **L3 编码幂等**：`encode(x) ≡ encode(decode(encode(x)))`
//! - **L4 摘要交换**：`digest(x) ≡ digest(decode(encode(x)))`
//!
//! 规范字节 = `serde_json::to_value` 规约（对象不依赖 `HashMap` 迭代序）后走
//! [`crate::context_image::canonical_json`]（递归键排序、紧凑无空白）。任何
//! 缓存/摘要/落盘都应经由本模块，避免「碰巧正确」。
//!
//! 设计约束：零新依赖；解码绝不 panic；错误一律带稳定错误码（供 `/metrics`
//! 与回执使用）；读宽容、写严格——非规范字节仍可解码，只是回执标记
//! `canonical=false`，下一次写入自动规范。

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::context_image::{canonical_json, fnv1a_hex};

/// 单件工件的默认体量上限（防御性：解码绝不因巨量输入拖垮进程）。
pub const DEFAULT_MAX_DECODE_BYTES: usize = 16 * 1024 * 1024;

/// 解码错误分类学：每一步失败都有稳定错误码（不静默折叠为「不行」）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    /// 读取字节失败（文件不存在、权限、IO）。
    Io(String),
    /// 字节不是合法 JSON（截断、语法错误）。
    Parse(String),
    /// JSON 合法但格式标识不对（陌生工件）。
    Format(String),
    /// 版本门：发现版本高于（或低于迁移链）支持范围。
    Version { found: u32, supported: u32 },
    /// 摘要不匹配（被改写、截断或非规范重排）。
    Digest { found: String, expected: String },
    /// 引擎/知识库指纹不匹配（陌生引擎的信封）。
    Fingerprint(Vec<String>),
    /// 反序列化后违反结构不变量。
    Invariant(String),
    /// 体量超限。
    TooLarge { found: usize, limit: usize },
}

impl DecodeError {
    /// 稳定错误码（ASCII，供指标键名与回执使用）。
    pub const fn code(&self) -> &'static str {
        match self {
            DecodeError::Io(_) => "io",
            DecodeError::Parse(_) => "parse",
            DecodeError::Format(_) => "format",
            DecodeError::Version { .. } => "version",
            DecodeError::Digest { .. } => "digest",
            DecodeError::Fingerprint(_) => "fingerprint",
            DecodeError::Invariant(_) => "invariant",
            DecodeError::TooLarge { .. } => "too_large",
        }
    }

    /// 计数台账键名：`decode_reject_<code>`。
    pub fn metric_key(&self) -> String {
        format!("decode_reject_{}", self.code())
    }
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecodeError::Io(message) => write!(formatter, "读取失败：{}", message),
            DecodeError::Parse(message) => write!(formatter, "解析失败：{}", message),
            DecodeError::Format(message) => write!(formatter, "未知格式：{}", message),
            DecodeError::Version { found, supported } => write!(
                formatter,
                "版本 {} 高于支持版本 {}（请升级程序）",
                found, supported
            ),
            DecodeError::Digest { found, expected } => {
                write!(formatter, "摘要不匹配（发现 {}，期望 {}）", found, expected)
            }
            DecodeError::Fingerprint(fields) => {
                write!(formatter, "引擎指纹不匹配：{}", fields.join("、"))
            }
            DecodeError::Invariant(message) => write!(formatter, "结构不变量失败：{}", message),
            DecodeError::TooLarge { found, limit } => {
                write!(formatter, "体量超限：{} > {}", found, limit)
            }
        }
    }
}

impl std::error::Error for DecodeError {}

/// 解码回执：成功解码时说明「读了什么、版本几何、字节是否已规范、是否迁移」。
///
/// 回执可序列化：导入响应用它给出可审计的接收凭据。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecodeReceipt {
    /// 工件类别（`session_envelope`、`context_image`……）。
    pub kind: String,
    /// 读到的格式版本。
    pub version: u32,
    /// 输入字节是否已是规范形（`false` = 读宽容）；下一次写入自动规范。
    pub canonical: bool,
    /// 迁移链条目（`"1 -> 2"`）；空 = 直接解码。
    #[serde(default)]
    pub migrations: Vec<String>,
}

impl DecodeReceipt {
    pub fn new(kind: impl Into<String>, version: u32, canonical: bool) -> Self {
        Self {
            kind: kind.into(),
            version,
            canonical,
            migrations: Vec::new(),
        }
    }
}

/// 规范字节：`to_value` 规约对象序后紧凑序列化。同一状态永远同一字节串。
///
/// 非有限浮点（NaN/Infinity）无法进入 `Value`，归为 [`DecodeError::Invariant`]。
pub fn canonical_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>, DecodeError> {
    Ok(canonical_text(value)?.into_bytes())
}

/// 规范文本（[`canonical_bytes`] 的字符串形式）。
pub fn canonical_text<T: Serialize>(value: &T) -> Result<String, DecodeError> {
    let value = serde_json::to_value(value)
        .map_err(|error| DecodeError::Invariant(format!("无法规约为 JSON 值：{}", error)))?;
    Ok(canonical_json(&value))
}

/// 规范摘要（FNV-1a 64，十六进制小写）：内容相等 ⇒ 摘要相等（L4 的基础）。
pub fn canonical_digest<T: Serialize>(value: &T) -> Result<String, DecodeError> {
    Ok(fnv1a_hex(canonical_text(value)?.as_bytes()))
}

/// 通用解码：体量门 → 解析 → 规范回执。不做任何格式/版本裁决（各工件自己判）。
///
/// 返回 `(T, DecodeReceipt)`；`receipt.canonical` 表示磁盘字节是否已是规范形。
pub fn decode_json<T: DeserializeOwned + Serialize>(
    bytes: &[u8],
    kind: &str,
    version: u32,
    max_bytes: usize,
) -> Result<(T, DecodeReceipt), DecodeError> {
    if bytes.len() > max_bytes {
        return Err(DecodeError::TooLarge {
            found: bytes.len(),
            limit: max_bytes,
        });
    }
    let value: T =
        serde_json::from_slice(bytes).map_err(|error| DecodeError::Parse(format!("{}", error)))?;
    let canonical_text = canonical_text(&value)?;
    let trimmed = trim_ascii_whitespace(bytes);
    let canonical = trimmed == canonical_text.as_bytes();
    Ok((
        value,
        DecodeReceipt {
            kind: kind.to_string(),
            version,
            canonical,
            migrations: Vec::new(),
        },
    ))
}

/// 去掉首尾 ASCII 空白（磁盘文件常见尾随换行）。
fn trim_ascii_whitespace(bytes: &[u8]) -> &[u8] {
    let start = bytes
        .iter()
        .position(|byte| !byte.is_ascii_whitespace())
        .unwrap_or(bytes.len());
    let end = bytes
        .iter()
        .rposition(|byte| !byte.is_ascii_whitespace())
        .map(|index| index + 1)
        .unwrap_or(start);
    &bytes[start..end]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct Sample {
        name: String,
        #[serde(default)]
        props: HashMap<String, String>,
    }

    fn sample_in_order(keys: &[&str]) -> Sample {
        let mut props = HashMap::new();
        for key in keys {
            props.insert(key.to_string(), format!("值-{}", key));
        }
        Sample {
            name: "样例".to_string(),
            props,
        }
    }

    #[test]
    fn canonical_bytes_are_hashmap_order_invariant() {
        let forward = sample_in_order(&["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛"]);
        let reverse = sample_in_order(&["辛", "庚", "己", "戊", "丁", "丙", "乙", "甲"]);
        assert_eq!(forward, reverse, "仅插入序不同，内容相等");
        assert_eq!(
            canonical_bytes(&forward).unwrap(),
            canonical_bytes(&reverse).unwrap(),
            "同一内容必须同一字节串（HashMap 迭代序不得泄漏到字节）"
        );
        assert_eq!(
            canonical_digest(&forward).unwrap(),
            canonical_digest(&reverse).unwrap()
        );
    }

    #[test]
    fn decode_roundtrip_satisfies_all_four_laws() {
        let original = sample_in_order(&["a", "b", "c"]);
        // L1：decode(encode(x)) ≡ x
        let bytes = canonical_bytes(&original).unwrap();
        let (decoded, receipt) = decode_json::<Sample>(&bytes, "sample", 1, 1024).unwrap();
        assert_eq!(decoded, original);
        assert!(receipt.canonical, "规范字节必须被识别");
        assert_eq!(receipt.kind, "sample");
        // L2：encode(decode(b)) ≡ b
        assert_eq!(canonical_bytes(&decoded).unwrap(), bytes);
        // L3：编码幂等
        let twice = canonical_bytes(&decoded).unwrap();
        let (again, _) = decode_json::<Sample>(&twice, "sample", 1, 1024).unwrap();
        assert_eq!(canonical_bytes(&again).unwrap(), twice);
        // L4：摘要交换
        assert_eq!(
            canonical_digest(&original).unwrap(),
            canonical_digest(&decoded).unwrap()
        );
    }

    #[test]
    fn pretty_input_decodes_with_non_canonical_receipt() {
        let original = sample_in_order(&["甲", "乙"]);
        let pretty = serde_json::to_string_pretty(&original).unwrap();
        let (decoded, receipt) =
            decode_json::<Sample>(pretty.as_bytes(), "sample", 1, 4096).unwrap();
        assert_eq!(decoded, original);
        assert!(!receipt.canonical, "pretty 字节不是规范形");
        // 读宽容：非规范输入再编码即规范
        let canonical = canonical_bytes(&decoded).unwrap();
        let (_, again) = decode_json::<Sample>(&canonical, "sample", 1, 4096).unwrap();
        assert!(again.canonical);
    }

    #[test]
    fn decode_errors_have_stable_codes_and_never_panic() {
        let cases: Vec<(DecodeError, &str)> = vec![
            (
                decode_json::<Sample>(b"{not json", "k", 1, 4096).unwrap_err(),
                "parse",
            ),
            (
                decode_json::<Sample>(b"", "k", 1, 4096).unwrap_err(),
                "parse",
            ),
            (
                decode_json::<Sample>(b"{\"name\":\"x\"}", "k", 1, 4).unwrap_err(),
                "too_large",
            ),
            (DecodeError::Invariant("样例".to_string()), "invariant"),
            (
                DecodeError::Fingerprint(vec!["kb_fingerprint".to_string()]),
                "fingerprint",
            ),
            (DecodeError::Format("陌生格式".to_string()), "format"),
            (DecodeError::Io("没有这个文件".to_string()), "io"),
            (
                DecodeError::Version {
                    found: 9,
                    supported: 1,
                },
                "version",
            ),
            (
                DecodeError::Digest {
                    found: "0".to_string(),
                    expected: "1".to_string(),
                },
                "digest",
            ),
        ];
        for (error, code) in cases {
            assert_eq!(error.code(), code, "{}", error);
            assert!(error.metric_key().starts_with("decode_reject_"));
        }
    }

    #[test]
    fn error_display_is_chinese_and_actionable() {
        let error = DecodeError::Version {
            found: 3,
            supported: 1,
        };
        assert!(error.to_string().contains("高于支持版本"));
        let error = DecodeError::Digest {
            found: "a".to_string(),
            expected: "b".to_string(),
        };
        assert!(error.to_string().contains("摘要不匹配"));
    }
}

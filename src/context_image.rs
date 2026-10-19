//! CIM：上下文镜像（Context Image）。
//!
//! 会话状态的可比较、可自检、可原子落盘的统一信封。设计目标对应 LLM 的
//! "会话缓存可保存/恢复"，但这里是白箱镜像：
//!   - 单文件分节：`sections: BTreeMap<String, Value>`，每节独立；
//!   - 规范序列化：对象键强制字典序、无多余空白——同一内容永远同一字节串；
//!   - 逐节哈希 + 全局哈希（自实现 FNV-1a 64，零新依赖；仅完整性自检，
//!     非安全用途）：某节损坏只废该节，其余节仍可用（安全回退重建）；
//!   - 原子写：临时文件 + rename，写一半的镜像不会替换旧镜像；
//!   - 版本门：`image_version` 高于支持版本直接拒绝，低版本留给迁移。

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::canonical::{self, DecodeError, DecodeReceipt};

/// 当前支持的镜像格式版本。
pub const IMAGE_VERSION: u32 = 1;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// FNV-1a 64 位哈希（十六进制小写）。
pub fn fnv1a_hex(bytes: &[u8]) -> String {
    let mut hash = FNV_OFFSET;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    format!("{:016x}", hash)
}

/// 规范 JSON 文本：对象键按字典序递归排序，紧凑无空白。
///
/// 不依赖 serde_json 的 Map 后端是否启用 `preserve_order`（feature 统一可能
/// 影响全局），因此手工递归序列化，保证语言级确定。
pub fn canonical_json(value: &Value) -> String {
    let mut output = String::new();
    write_canonical(value, &mut output);
    output
}

fn write_canonical(value: &Value, output: &mut String) {
    match value {
        Value::Null => output.push_str("null"),
        Value::Bool(true) => output.push_str("true"),
        Value::Bool(false) => output.push_str("false"),
        Value::Number(number) => output.push_str(&number.to_string()),
        Value::String(text) => {
            output.push_str(&serde_json::to_string(text).unwrap_or_else(|_| "\"\"".to_string()))
        }
        Value::Array(items) => {
            output.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                write_canonical(item, output);
            }
            output.push(']');
        }
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            output.push('{');
            for (index, key) in keys.into_iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                output.push_str(
                    &serde_json::to_string(key).unwrap_or_else(|_| "\"\"".to_string()),
                );
                output.push(':');
                write_canonical(&map[key], output);
            }
            output.push('}');
        }
    }
}

/// 单节规范文本：`name\u{1}canonical-json`。
fn section_canonical(name: &str, value: &Value) -> String {
    format!("{}\u{1}{}", name, canonical_json(value))
}

/// 单节哈希。
pub fn section_hash(name: &str, value: &Value) -> String {
    fnv1a_hex(section_canonical(name, value).as_bytes())
}

/// 分节会话镜像。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextImage {
    pub image_version: u32,
    pub sections: BTreeMap<String, Value>,
    #[serde(default)]
    pub section_hashes: BTreeMap<String, String>,
    #[serde(default)]
    pub canonical_hash: String,
}

/// 镜像自检结果：逐节损坏清单 + 全局是否损坏。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ImageAudit {
    pub damaged_sections: Vec<String>,
    pub global_damaged: bool,
}

impl ImageAudit {
    pub fn is_clean(&self) -> bool {
        self.damaged_sections.is_empty() && !self.global_damaged
    }
}

impl Default for ContextImage {
    fn default() -> Self {
        Self::new()
    }
}

impl ContextImage {
    pub fn new() -> Self {
        Self {
            image_version: IMAGE_VERSION,
            sections: BTreeMap::new(),
            section_hashes: BTreeMap::new(),
            canonical_hash: String::new(),
        }
    }

    pub fn set_section(&mut self, name: &str, value: Value) {
        self.sections.insert(name.to_string(), value);
    }

    pub fn section(&self, name: &str) -> Option<&Value> {
        self.sections.get(name)
    }

    pub fn remove_section(&mut self, name: &str) -> Option<Value> {
        self.section_hashes.remove(name);
        self.sections.remove(name)
    }

    pub fn section_names(&self) -> Vec<&str> {
        self.sections.keys().map(String::as_str).collect()
    }

    /// 重算逐节哈希与全局哈希。
    pub fn seal(&mut self) {
        self.section_hashes = self
            .sections
            .iter()
            .map(|(name, value)| (name.clone(), section_hash(name, value)))
            .collect();
        self.canonical_hash = self.compute_canonical_hash();
    }

    /// 封口后返回自身（链式构造）。
    pub fn sealed(mut self) -> Self {
        self.seal();
        self
    }

    /// 全局规范哈希：仅覆盖 `image_version + sections + section_hashes`。
    fn compute_canonical_hash(&self) -> String {
        let mut root = serde_json::Map::new();
        root.insert(
            "image_version".to_string(),
            Value::from(self.image_version as u64),
        );
        root.insert(
            "sections".to_string(),
            Value::Object(self.sections.clone().into_iter().collect()),
        );
        root.insert(
            "section_hashes".to_string(),
            Value::Object(
                self.section_hashes
                    .clone()
                    .into_iter()
                    .map(|(key, value)| (key, Value::String(value)))
                    .collect(),
            ),
        );
        fnv1a_hex(canonical_json(&Value::Object(root)).as_bytes())
    }

    /// 逐节自检：坏掉的节只报该节，不牵连其余节。
    pub fn verify(&self) -> ImageAudit {
        let mut damaged: Vec<String> = Vec::new();
        for (name, value) in &self.sections {
            match self.section_hashes.get(name) {
                Some(expected) if *expected == section_hash(name, value) => {}
                _ => damaged.push(name.clone()),
            }
        }
        for name in self.section_hashes.keys() {
            if !self.sections.contains_key(name) {
                damaged.push(name.clone());
            }
        }
        damaged.sort();
        damaged.dedup();
        ImageAudit {
            damaged_sections: damaged,
            global_damaged: self.canonical_hash != self.compute_canonical_hash(),
        }
    }

    /// 确定性 JSON 文本（BTreeMap 序 + pretty）；仅作「人读导出」，
    /// 不用于落盘或摘要。
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// 规范文本（对象键递归排序、紧凑无空白）：落盘与摘要的唯一来源。
    pub fn canonical_text(&self) -> Result<String, DecodeError> {
        canonical::canonical_text(self)
    }

    /// 原子保存：先写 `{path}.tmp` 再 rename，任何时刻磁盘上只有完整镜像。
    ///
    /// 写入的是规范字节：同一镜像重复保存逐字节一致（L2 字节不动点），
    /// 摘要只对规范字节计算（L4 摘要交换）。
    pub fn save_atomic(&self, path: &str) -> std::io::Result<()> {
        let target = Path::new(path);
        if let Some(parent) = target.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let text = self
            .canonical_text()
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::Other, error.to_string()))?;
        let tmp_path = format!("{}.tmp", path);
        {
            let mut file = std::fs::File::create(&tmp_path)?;
            file.write_all(text.as_bytes())?;
            file.sync_all()?;
        }
        std::fs::rename(&tmp_path, target)
    }

    /// 读取镜像并返回解码回执（解析 + 版本门；哈希裁决仍由 `verify` 决定）。
    ///
    /// 读宽容：pretty 排布的文件仍可解码（`receipt.canonical == false`），
    /// 下一次 `save_atomic` 自动规范。
    pub fn decode(path: &str) -> Result<(Self, DecodeReceipt), DecodeError> {
        let bytes = std::fs::read(path).map_err(|error| DecodeError::Io(error.to_string()))?;
        Self::decode_bytes(&bytes)
    }

    /// 从字节解码（审计/测试可复用，避免重复读盘）。
    pub fn decode_bytes(bytes: &[u8]) -> Result<(Self, DecodeReceipt), DecodeError> {
        let (image, receipt) = canonical::decode_json::<Self>(
            bytes,
            "context_image",
            IMAGE_VERSION,
            canonical::DEFAULT_MAX_DECODE_BYTES,
        )?;
        if image.image_version > IMAGE_VERSION {
            return Err(DecodeError::Version {
                found: image.image_version,
                supported: IMAGE_VERSION,
            });
        }
        Ok((image, receipt))
    }

    /// 读取镜像（只做解析与版本门，不做哈希裁决；调用方用 `verify` 决定信任哪些节）。
    pub fn load(path: &str) -> Result<Self, String> {
        Self::decode(path)
            .map(|(image, _)| image)
            .map_err(|error| error.to_string())
    }

    /// 读取并要求完全自检通过。
    pub fn load_verified(path: &str) -> Result<Self, String> {
        let image = Self::load(path)?;
        let audit = image.verify();
        if audit.global_damaged {
            return Err("镜像全局哈希不匹配（内容已被改写或损坏）".to_string());
        }
        if !audit.damaged_sections.is_empty() {
            return Err(format!(
                "镜像分节哈希不匹配：{}",
                audit.damaged_sections.join("、")
            ));
        }
        Ok(image)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn sample() -> ContextImage {
        let mut image = ContextImage::new();
        image.set_section("predicates", json!({"b": 2, "a": 1}));
        image.set_section("journal", json!([{"added": 1}]));
        image.seal();
        image
    }

    #[test]
    fn canonical_json_sorts_object_keys_recursively() {
        let left = json!({"b": {"y": 1, "x": 2}, "a": [ {"q": 1, "p": 2} ]});
        let right = json!({"a": [ {"p": 2, "q": 1} ], "b": {"x": 2, "y": 1}});
        assert_eq!(canonical_json(&left), canonical_json(&right));
        assert_eq!(
            canonical_json(&json!({"b":1,"a":2})),
            r#"{"a":2,"b":1}"#
        );
    }

    #[test]
    fn image_roundtrip_via_atomic_save() {
        let image = sample();
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("context_image.json");
        let path_text = path.to_str().unwrap();
        image.save_atomic(path_text).unwrap();
        assert!(!Path::new(&format!("{}.tmp", path_text)).exists(), "临时文件不应残留");

        let loaded = ContextImage::load_verified(path_text).unwrap();
        assert_eq!(loaded, image);
        assert!(loaded.verify().is_clean());
        assert_eq!(
            loaded.canonical_text().unwrap(),
            image.canonical_text().unwrap(),
            "落盘字节稳定"
        );
        // 再存一次字节不变
        loaded.save_atomic(path_text).unwrap();
        assert_eq!(
            std::fs::read_to_string(path_text).unwrap(),
            image.canonical_text().unwrap()
        );
    }

    #[test]
    fn pretty_image_decodes_with_non_canonical_receipt_then_normalizes() {
        let image = sample();
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("image.json");
        let path_text = path.to_str().unwrap();
        std::fs::write(path_text, image.to_json()).unwrap();

        let (loaded, receipt) = ContextImage::decode(path_text).unwrap();
        assert_eq!(loaded, image);
        assert!(!receipt.canonical, "人读 pretty 不是规范形");
        assert_eq!(receipt.kind, "context_image");

        image.save_atomic(path_text).unwrap();
        let (_, receipt) = ContextImage::decode(path_text).unwrap();
        assert!(receipt.canonical, "规范化落盘后必须是规范形");
        assert_eq!(
            std::fs::read_to_string(path_text).unwrap(),
            image.canonical_text().unwrap()
        );
    }

    #[test]
    fn tampered_section_is_isolated() {
        let mut image = sample();
        image
            .sections
            .insert("predicates".to_string(), json!({"a": 999}));
        let audit = image.verify();
        assert_eq!(audit.damaged_sections, vec!["predicates".to_string()]);
        assert!(audit.global_damaged, "全局哈希也应报损");
        // 其余节仍可单独取用（安全回退重建的依据）
        assert!(image.section("journal").is_some());
    }

    #[test]
    fn version_gate_rejects_too_new_image() {
        let mut image = sample();
        image.image_version = IMAGE_VERSION + 1;
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("image.json");
        let path_text = path.to_str().unwrap();
        std::fs::write(path_text, image.to_json()).unwrap();
        let error = ContextImage::load(path_text).unwrap_err();
        assert!(error.contains("高于支持版本"), "{}", error);
        let error = ContextImage::decode(path_text).unwrap_err();
        assert_eq!(error.code(), "version");
        assert_eq!(
            error,
            DecodeError::Version {
                found: IMAGE_VERSION + 1,
                supported: IMAGE_VERSION
            }
        );
    }

    #[test]
    fn missing_section_hash_is_damage() {
        let mut image = sample();
        image.section_hashes.remove("journal");
        let audit = image.verify();
        assert_eq!(audit.damaged_sections, vec!["journal".to_string()]);
    }
}

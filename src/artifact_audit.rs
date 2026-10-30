//! ART：工件审计（Artifact Audit）——知识版的 `rustfmt --check`。
//!
//! 解码宪法（[`crate::canonical`]）需要一个可审计的
//! 执行面：对已知持久化工件执行「解码 → 规范重编码 → 与磁盘比对」闭环，
//! 把每个文件归入四类之一：
//!
//! - [`ArtifactStatus::Canonical`]：磁盘字节已是规范形（幂等、可 diff）；
//! - [`ArtifactStatus::NonCanonical`]：可解码但字节非规范（按 pretty 排布落盘），
//!   `--fix` 可原子重写；
//! - [`ArtifactStatus::Repairable`]：追加式哈希链（编年史）尾行截断——损坏
//!   只在末尾，可信前缀无损；`--fix` 仅截掉坏尾，绝不裁剪链中；
//! - [`ArtifactStatus::Damaged`]：解码或自检失败（parse/digest/version/…），
//!   **绝不触碰**——与导入日志「宁可不撤销，也不误撤」同一伦理；
//! - [`ArtifactStatus::Missing`]：文件不存在。
//!
//! 只做无损重写：内容按 JSON 值语义保持不变，只改字节排布。`fix` 只处理
//! `NonCanonical`/`Repairable`，返回新的审计结果（重写后必为 `Canonical`）。

use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::canonical::{self, DecodeError};
use crate::chronicle::{Chronicle, FsyncPolicy};
use crate::context_image::{ContextImage, IMAGE_VERSION};
use crate::dialogue_manager::DialogueManagerSnapshot;
use crate::dialogue_uri::DialogueSession;
use crate::embedding_eval::EvalBaseline;
use crate::entity_registry::{EntityRegistry, FOLD_VERSION};
use crate::predicate_registry::PredicateRegistry;
use crate::serve_api::SessionEnvelope;
use crate::symbolic_reasoner::DialogueSnapshot;

/// 已知工件类别（决定「怎样解码才算没坏」）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ArtifactKind {
    /// CIM 分节信封（导入日志/外链名册/失败台账共用）。
    ContextImage,
    /// 会话信封（冷存储 + HTTP 导入导出）。
    SessionEnvelope,
    /// 推理器对话记忆（`dialogue_state.json`）。
    DialogueState,
    /// 对话管理器快照（`*.manager.json`）。
    DialogueManager,
    /// 对话工件会话标识侧车（`*.session.json`）。
    DialogueSession,
    /// 谓词画像注册表。
    PredicateRegistry,
    /// 实体身份注册表。
    EntityRegistry,
    /// Eval* 棘轮基线。
    EvalBaseline,
    /// 追加式哈希链对话档案（JSONL，`.chronicle.jsonl`）。
    Chronicle,
    /// 其它 JSON（只查解析与键序）。
    Json,
}

impl ArtifactKind {
    pub const fn label(self) -> &'static str {
        match self {
            ArtifactKind::ContextImage => "context_image",
            ArtifactKind::SessionEnvelope => "session_envelope",
            ArtifactKind::DialogueState => "dialogue_state",
            ArtifactKind::DialogueManager => "dialogue_manager",
            ArtifactKind::DialogueSession => "dialogue_session",
            ArtifactKind::PredicateRegistry => "predicate_registry",
            ArtifactKind::EntityRegistry => "entity_registry",
            ArtifactKind::EvalBaseline => "eval_baseline",
            ArtifactKind::Chronicle => "chronicle",
            ArtifactKind::Json => "json",
        }
    }

    /// 从文件名推断工件类别（默认扫描与 `--path` 均可用）。
    pub fn infer(path: &str) -> Self {
        let name = path.rsplit(['/', '\\']).next().unwrap_or(path);
        if name.ends_with(".manager.json") {
            return ArtifactKind::DialogueManager;
        }
        if name.ends_with(".chronicle.jsonl") {
            return ArtifactKind::Chronicle;
        }
        if name.ends_with(".session.json") {
            return ArtifactKind::DialogueSession;
        }
        match name {
            "import_journal.json" | "external_ledger.json" => ArtifactKind::ContextImage,
            "dialogue_state.json" => ArtifactKind::DialogueState,
            "predicate_registry.json" => ArtifactKind::PredicateRegistry,
            "entity_registry.json" => ArtifactKind::EntityRegistry,
            "eval_baseline.json"
            | "nlu_eval_baseline.json"
            | "recovery_baseline.json"
            | "quality_baseline.json" => ArtifactKind::EvalBaseline,
            _ => ArtifactKind::Json,
        }
    }
}

/// 单个工件的审计结论。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArtifactStatus {
    Canonical,
    NonCanonical,
    /// 哈希链仅尾行损坏：可信前缀无损，`--fix` 可安全截尾续写。
    Repairable(String),
    Damaged(String),
    Missing,
}

impl ArtifactStatus {
    pub const fn label(&self) -> &'static str {
        match self {
            ArtifactStatus::Canonical => "canonical",
            ArtifactStatus::NonCanonical => "non_canonical",
            ArtifactStatus::Repairable(_) => "repairable",
            ArtifactStatus::Damaged(_) => "damaged",
            ArtifactStatus::Missing => "missing",
        }
    }

    pub fn detail(&self) -> String {
        match self {
            ArtifactStatus::Repairable(reason) | ArtifactStatus::Damaged(reason) => reason.clone(),
            _ => String::new(),
        }
    }
}

/// 单件审计结果（可序列化为报告条目）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactAudit {
    pub path: String,
    pub kind: ArtifactKind,
    pub status: ArtifactStatus,
}

impl ArtifactAudit {
    pub fn is_clean(&self) -> bool {
        matches!(
            self.status,
            ArtifactStatus::Canonical | ArtifactStatus::Missing
        )
    }
}

/// 审计一个工件：缺失 → Missing；解码失败 → Damaged；字节非规范 → NonCanonical。
pub fn audit(path: &str, kind: ArtifactKind) -> ArtifactAudit {
    if kind == ArtifactKind::Chronicle {
        return audit_chronicle(path, kind);
    }
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return ArtifactAudit {
                path: path.to_string(),
                kind,
                status: ArtifactStatus::Missing,
            }
        }
        Err(error) => {
            return ArtifactAudit {
                path: path.to_string(),
                kind,
                status: ArtifactStatus::Damaged(format!("io: {}", error)),
            }
        }
    };
    match verify(&bytes, kind) {
        Ok((_, true)) => ArtifactAudit {
            path: path.to_string(),
            kind,
            status: ArtifactStatus::Canonical,
        },
        Ok((_, false)) => ArtifactAudit {
            path: path.to_string(),
            kind,
            status: ArtifactStatus::NonCanonical,
        },
        Err(error) => ArtifactAudit {
            path: path.to_string(),
            kind,
            status: ArtifactStatus::Damaged(format!("decode_{}: {}", error.code(), error)),
        },
    }
}

/// 文件名推断类别后审计。
pub fn audit_inferred(path: &str) -> ArtifactAudit {
    audit(path, ArtifactKind::infer(path))
}

/// 哈希链档案专用审计：`Chronicle::audit_file` 只读校验，不看字节排布。
fn audit_chronicle(path: &str, kind: ArtifactKind) -> ArtifactAudit {
    if !std::path::Path::new(path).exists() {
        return ArtifactAudit {
            path: path.to_string(),
            kind,
            status: ArtifactStatus::Missing,
        };
    }
    match Chronicle::audit_file(path) {
        Ok(audit) if audit.is_clean() => ArtifactAudit {
            path: path.to_string(),
            kind,
            status: ArtifactStatus::Canonical,
        },
        Ok(audit) if audit.tail_incomplete => ArtifactAudit {
            path: path.to_string(),
            kind,
            status: ArtifactStatus::Repairable(audit.verdict()),
        },
        Ok(audit) => ArtifactAudit {
            path: path.to_string(),
            kind,
            status: ArtifactStatus::Damaged(audit.verdict()),
        },
        Err(error) => ArtifactAudit {
            path: path.to_string(),
            kind,
            status: ArtifactStatus::Damaged(format!("io: {}", error)),
        },
    }
}

/// `--fix` 的哈希链分支：仅对「尾行截断」截尾续写；链中损坏一律拒绝。
fn fix_chronicle(path: &str, kind: ArtifactKind) -> Result<ArtifactAudit, String> {
    let audit = audit_chronicle(path, kind);
    match &audit.status {
        ArtifactStatus::Canonical | ArtifactStatus::Missing => Ok(audit),
        ArtifactStatus::Damaged(reason) => Err(format!(
            "damaged artifacts are never rewritten ({}): {}",
            path, reason
        )),
        ArtifactStatus::Repairable(_) => {
            let (mut chronicle, _) = Chronicle::open(path, FsyncPolicy::None)
                .map_err(|error| format!("{}: {}", path, error))?;
            chronicle
                .repair_tail()
                .map_err(|error| format!("{}: {}", path, error))?;
            Ok(audit_chronicle(path, kind))
        }
        ArtifactStatus::NonCanonical => {
            Err(format!("{}: chronicle 审计不应产生 non_canonical", path))
        }
    }
}

/// `--fix`：只原子重写 `NonCanonical`；`Damaged`/`Missing` 原样返回。
///
/// 返回重写后的审计结论（成功重写即 `Canonical`）。`Damaged` 返回 `Err`，
/// 调用方据此刻入退出码。
pub fn fix(path: &str, kind: ArtifactKind) -> Result<ArtifactAudit, String> {
    if kind == ArtifactKind::Chronicle {
        return fix_chronicle(path, kind);
    }
    let audit = audit(path, kind);
    match &audit.status {
        ArtifactStatus::Canonical | ArtifactStatus::Missing => Ok(audit),
        ArtifactStatus::Damaged(reason) => Err(format!(
            "damaged artifacts are never rewritten ({}): {}",
            path, reason
        )),
        ArtifactStatus::Repairable(reason) => Err(format!(
            "repairable status is unexpected for {} ({}): {}",
            kind.label(),
            path,
            reason
        )),
        ArtifactStatus::NonCanonical => {
            let bytes = std::fs::read(path).map_err(|error| format!("{}: {}", path, error))?;
            let (canonical_text, _) =
                verify(&bytes, kind).map_err(|error| format!("{}: {}", path, error))?;
            write_atomic(path, canonical_text.as_bytes())
                .map_err(|error| format!("{}: {}", path, error))?;
            Ok(ArtifactAudit {
                path: path.to_string(),
                kind,
                status: ArtifactStatus::Canonical,
            })
        }
    }
}

/// 原子写：临时文件 + rename，任何时刻磁盘上只有完整工件。
fn write_atomic(path: &str, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    let target = std::path::Path::new(path);
    if let Some(parent) = target.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let tmp = format!("{}.tmp", path);
    {
        let mut file = std::fs::File::create(&tmp)?;
        file.write_all(bytes)?;
        file.sync_all()?;
    }
    std::fs::rename(&tmp, target)
}

/// 解码 + 规范重编码：返回（规范文本, 输入是否已规范）。
fn verify(bytes: &[u8], kind: ArtifactKind) -> Result<(String, bool), DecodeError> {
    let limit = canonical::DEFAULT_MAX_DECODE_BYTES;
    match kind {
        ArtifactKind::ContextImage => {
            let (image, receipt) = ContextImage::decode_bytes(bytes)?;
            let audit = image.verify();
            if audit.global_damaged {
                return Err(DecodeError::Digest {
                    found: "镜像全局哈希损坏".to_string(),
                    expected: "自检通过".to_string(),
                });
            }
            if !audit.damaged_sections.is_empty() {
                return Err(DecodeError::Invariant(format!(
                    "损坏分节：{}",
                    audit.damaged_sections.join("、")
                )));
            }
            Ok((image.canonical_text()?, receipt.canonical))
        }
        ArtifactKind::SessionEnvelope => {
            let (envelope, receipt) = SessionEnvelope::decode(bytes, limit)?;
            Ok((canonical::canonical_text(&envelope)?, receipt.canonical))
        }
        ArtifactKind::DialogueState => typed::<DialogueSnapshot>(bytes, kind.label(), 1, limit),
        ArtifactKind::DialogueManager => {
            typed::<DialogueManagerSnapshot>(bytes, kind.label(), 1, limit)
        }
        ArtifactKind::DialogueSession => DialogueSession::verify_bytes(bytes),
        ArtifactKind::PredicateRegistry => {
            typed::<PredicateRegistry>(bytes, kind.label(), 1, limit)
        }
        ArtifactKind::EntityRegistry => {
            // 折叠算法版本门：高于支持版本属于「陌生工件」，不以规范字节通过
            let (registry, receipt) =
                canonical::decode_json::<EntityRegistry>(bytes, kind.label(), FOLD_VERSION, limit)?;
            if !registry.version_supported() {
                return Err(DecodeError::Version {
                    found: registry.fold_version,
                    supported: FOLD_VERSION,
                });
            }
            Ok((canonical::canonical_text(&registry)?, receipt.canonical))
        }
        ArtifactKind::EvalBaseline => typed::<EvalBaseline>(bytes, kind.label(), 1, limit),
        ArtifactKind::Chronicle => Err(DecodeError::Invariant(
            "chronicle 不以 JSON 规范重编码审计（走哈希链校验）".to_string(),
        )),
        ArtifactKind::Json => {
            let (value, receipt) =
                canonical::decode_json::<serde_json::Value>(bytes, kind.label(), 1, limit)?;
            Ok((canonical::canonical_text(&value)?, receipt.canonical))
        }
    }
}

/// 通用类型化解码（工件类型已知、只需解析与键序核查）。
fn typed<T: DeserializeOwned + Serialize>(
    bytes: &[u8],
    kind: &str,
    version: u32,
    limit: usize,
) -> Result<(String, bool), DecodeError> {
    let (value, receipt) = canonical::decode_json::<T>(bytes, kind, version, limit)?;
    Ok((canonical::canonical_text(&value)?, receipt.canonical))
}

/// 版本门辅助（供未来迁移链使用）：`found > supported` 即拒绝。
pub fn version_gate(found: u32, supported: u32) -> Result<(), DecodeError> {
    if found > supported {
        return Err(DecodeError::Version { found, supported });
    }
    Ok(())
}

/// `IMAGE_VERSION` 的显式再导出（审计报告用，避免调用方到处引用）。
pub const SUPPORTED_IMAGE_VERSION: u32 = IMAGE_VERSION;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbolic_reasoner::DialogueContext;

    fn temp_path(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "kp-artifacts-{}-{}.json",
            std::process::id(),
            label
        ))
    }

    fn sample_context_image() -> ContextImage {
        let mut image = ContextImage::new();
        image.set_section("payload", serde_json::json!({"b": 2, "a": 1}));
        image.seal();
        image
    }

    #[test]
    fn pretty_artifact_is_audited_non_canonical_then_fixed() {
        let path = temp_path("pretty");
        let path_text = path.to_string_lossy().to_string();
        let image = sample_context_image();
        std::fs::write(&path_text, image.to_json()).unwrap();

        let audit = audit(&path_text, ArtifactKind::ContextImage);
        assert_eq!(audit.status, ArtifactStatus::NonCanonical);
        assert!(!audit.is_clean());

        let fixed = fix(&path_text, ArtifactKind::ContextImage).unwrap();
        assert_eq!(fixed.status, ArtifactStatus::Canonical);
        assert_eq!(
            std::fs::read_to_string(&path_text).unwrap(),
            image.canonical_text().unwrap()
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn damaged_artifact_is_never_rewritten() {
        let path = temp_path("damaged");
        let path_text = path.to_string_lossy().to_string();
        let mut image = sample_context_image();
        std::fs::write(&path_text, image.to_json()).unwrap();
        // 篡改分节但不更新哈希 → Damaged
        image
            .sections
            .insert("payload".to_string(), serde_json::json!({"a": 999}));
        std::fs::write(&path_text, image.to_json()).unwrap();
        let before = std::fs::read(&path_text).unwrap();

        let audit = audit(&path_text, ArtifactKind::ContextImage);
        assert!(
            matches!(audit.status, ArtifactStatus::Damaged(_)),
            "{}",
            audit.status.label()
        );
        let error = fix(&path_text, ArtifactKind::ContextImage).unwrap_err();
        assert!(error.contains("never rewritten"), "{}", error);
        assert_eq!(
            std::fs::read(&path_text).unwrap(),
            before,
            "损坏工件不得被改写"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn missing_artifact_is_reported_not_created() {
        let path = temp_path("missing");
        let path_text = path.to_string_lossy().to_string();
        let audit = audit(&path_text, ArtifactKind::Json);
        assert_eq!(audit.status, ArtifactStatus::Missing);
        assert!(audit.is_clean());
        assert!(
            !std::path::Path::new(&path_text).exists(),
            "审计不得创建文件"
        );
        assert!(fix(&path_text, ArtifactKind::Json).is_ok());
        assert!(
            !std::path::Path::new(&path_text).exists(),
            "fix 不得凭空创建"
        );
    }

    #[test]
    fn session_envelope_artifact_is_classified_by_decode_constitution() {
        use crate::serve_api::{EngineStamp, ENVELOPE_FORMAT, ENVELOPE_VERSION};
        let path = temp_path("envelope");
        let path_text = path.to_string_lossy().to_string();
        let mut context = DialogueContext::new(8);
        context.record_entities_typed(&[("姚明".to_string(), Some("人物".to_string()))]);
        let mut envelope = SessionEnvelope {
            format: ENVELOPE_FORMAT.to_string(),
            version: ENVELOPE_VERSION,
            session_id: "s-audit".to_string(),
            turns: 1,
            history: Vec::new(),
            policy: Default::default(),
            manager: crate::dialogue_manager::DialogueManager::new().snapshot(),
            reasoner: context.snapshot(),
            engine: EngineStamp {
                crate_version: env!("CARGO_PKG_VERSION").to_string(),
                kb_fingerprint: "audit".to_string(),
                kb_facts: 1,
                kb_generation: 0,
                prompt_mode: "on".to_string(),
                recovery_mode: "observe".to_string(),
            },
            digest: String::new(),
        };
        envelope.seal();
        // pretty 落盘 → NonCanonical → fix 后 Canonical
        std::fs::write(&path_text, serde_json::to_string_pretty(&envelope).unwrap()).unwrap();
        assert_eq!(
            audit(&path_text, ArtifactKind::SessionEnvelope).status,
            ArtifactStatus::NonCanonical
        );
        assert_eq!(
            fix(&path_text, ArtifactKind::SessionEnvelope)
                .unwrap()
                .status,
            ArtifactStatus::Canonical
        );
        // 篡改内容 → Damaged，fix 拒绝
        let mut tampered: SessionEnvelope =
            serde_json::from_str(&std::fs::read_to_string(&path_text).unwrap()).unwrap();
        tampered.history.push("伪造".to_string());
        std::fs::write(&path_text, serde_json::to_string(&tampered).unwrap()).unwrap();
        assert!(matches!(
            audit(&path_text, ArtifactKind::SessionEnvelope).status,
            ArtifactStatus::Damaged(_)
        ));
        assert!(fix(&path_text, ArtifactKind::SessionEnvelope).is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn kind_inference_matches_known_file_names() {
        assert_eq!(
            ArtifactKind::infer("/tmp/x/import_journal.json"),
            ArtifactKind::ContextImage
        );
        assert_eq!(
            ArtifactKind::infer("oxigraph_data/dialogue_state.json.manager.json"),
            ArtifactKind::DialogueManager
        );
        assert_eq!(
            ArtifactKind::infer("oxigraph_data/dialogue_state.json"),
            ArtifactKind::DialogueState
        );
        assert_eq!(
            ArtifactKind::infer("nlu_eval_baseline.json"),
            ArtifactKind::EvalBaseline
        );
        assert_eq!(
            ArtifactKind::infer("oxigraph_data/dialogue_state.json.chronicle.jsonl"),
            ArtifactKind::Chronicle
        );
        assert_eq!(
            ArtifactKind::infer("oxigraph_data/dialogue_state.json.chronicle.jsonl.session.json"),
            ArtifactKind::DialogueSession
        );
        assert_eq!(
            ArtifactKind::infer("oxigraph_data/entity_registry.json"),
            ArtifactKind::EntityRegistry
        );
        assert_eq!(ArtifactKind::infer("random.json"), ArtifactKind::Json);
    }

    #[test]
    fn dialogue_session_sidecar_is_canonical_and_fixable() {
        let path = temp_path("dialogue-session");
        let path_text = path.to_string_lossy().to_string();
        let session = DialogueSession::open_or_create(&path_text).unwrap();
        assert!(!session.id().is_empty());
        assert_eq!(
            audit(&path_text, ArtifactKind::DialogueSession).status,
            ArtifactStatus::Canonical
        );

        std::fs::write(
            &path_text,
            "{\n  \"session_version\": 1,\n  \"session_id\": \"s-audit\"\n}\n",
        )
        .unwrap();
        assert_eq!(
            audit(&path_text, ArtifactKind::DialogueSession).status,
            ArtifactStatus::NonCanonical
        );
        let fixed = fix(&path_text, ArtifactKind::DialogueSession).unwrap();
        assert_eq!(fixed.status, ArtifactStatus::Canonical);

        std::fs::write(
            &path_text,
            "{\"session_id\":\"\",\"session_version\":1}",
        )
        .unwrap();
        assert!(matches!(
            audit(&path_text, ArtifactKind::DialogueSession).status,
            ArtifactStatus::Damaged(_)
        ));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn chronicle_tail_damage_is_repairable_not_damaged() {
        use crate::chronicle::{Chronicle, FsyncPolicy};
        let path = temp_path("chronicle");
        let path_text = path.to_string_lossy().to_string();
        {
            let (mut chronicle, _) = Chronicle::open(&path_text, FsyncPolicy::None).unwrap();
            chronicle
                .append("turn", serde_json::json!({"user": "姚明是谁"}))
                .unwrap();
            chronicle
                .append("turn", serde_json::json!({"user": "他的身高呢"}))
                .unwrap();
        }
        // 干净档案 → Canonical
        assert_eq!(
            audit(&path_text, ArtifactKind::Chronicle).status,
            ArtifactStatus::Canonical
        );
        // 砍掉最后一行的一半 → Repairable，fix 后恢复 Canonical 且前缀无损
        let bytes = std::fs::read(&path_text).unwrap();
        std::fs::write(&path_text, &bytes[..bytes.len() - 8]).unwrap();
        let audit = audit(&path_text, ArtifactKind::Chronicle);
        assert!(
            matches!(audit.status, ArtifactStatus::Repairable(_)),
            "尾行截断应报可修复：{}",
            audit.status.label()
        );
        let fixed = fix(&path_text, ArtifactKind::Chronicle).unwrap();
        assert_eq!(fixed.status, ArtifactStatus::Canonical);
        let (repaired, chain) = Chronicle::open(&path_text, FsyncPolicy::None).unwrap();
        assert!(chain.is_clean());
        assert_eq!(repaired.len(), 1, "只保留可信前缀");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn chronicle_mid_chain_damage_is_damaged_and_never_fixed() {
        use crate::chronicle::{Chronicle, FsyncPolicy};
        let path = temp_path("chronicle-mid");
        let path_text = path.to_string_lossy().to_string();
        {
            let (mut chronicle, _) = Chronicle::open(&path_text, FsyncPolicy::None).unwrap();
            chronicle
                .append("turn", serde_json::json!({"user": "第一轮"}))
                .unwrap();
            chronicle
                .append("turn", serde_json::json!({"user": "第二轮"}))
                .unwrap();
        }
        // 篡改首行内容（哈希不匹配）→ Damaged；fix 拒绝且字节不动
        let text = std::fs::read_to_string(&path_text).unwrap();
        let tampered = text.replacen("第一轮", "被篡改", 1);
        std::fs::write(&path_text, tampered).unwrap();
        let before = std::fs::read(&path_text).unwrap();
        let audit = audit(&path_text, ArtifactKind::Chronicle);
        assert!(
            matches!(audit.status, ArtifactStatus::Damaged(_)),
            "链中损坏应报损坏：{}",
            audit.status.label()
        );
        assert!(fix(&path_text, ArtifactKind::Chronicle).is_err());
        assert_eq!(
            std::fs::read(&path_text).unwrap(),
            before,
            "链中损坏不得被裁剪"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn typed_kinds_roundtrip_through_audit_and_fix() {
        let path = temp_path("manager");
        let path_text = path.to_string_lossy().to_string();
        let manager = crate::dialogue_manager::DialogueManager::new();
        let snapshot = manager.snapshot();
        std::fs::write(&path_text, serde_json::to_string_pretty(&snapshot).unwrap()).unwrap();
        assert_eq!(
            audit(&path_text, ArtifactKind::DialogueManager).status,
            ArtifactStatus::NonCanonical
        );
        assert_eq!(
            fix(&path_text, ArtifactKind::DialogueManager)
                .unwrap()
                .status,
            ArtifactStatus::Canonical
        );
        assert_eq!(
            canonical::canonical_text(&snapshot).unwrap(),
            std::fs::read_to_string(&path_text).unwrap()
        );
        let _ = std::fs::remove_file(&path);

        let path = temp_path("registry");
        let path_text = path.to_string_lossy().to_string();
        let registry = PredicateRegistry::default();
        std::fs::write(&path_text, registry.to_json()).unwrap();
        let report = audit(&path_text, ArtifactKind::PredicateRegistry);
        assert!(
            report.status == ArtifactStatus::NonCanonical
                || report.status == ArtifactStatus::Canonical,
            "{}",
            report.status.label()
        );
        assert_eq!(
            fix(&path_text, ArtifactKind::PredicateRegistry)
                .unwrap()
                .status,
            ArtifactStatus::Canonical
        );
        let _ = std::fs::remove_file(&path);

        // 实体身份注册表：规范重写 + 折叠版本门（高版本 → Damaged，永不改写）
        let path = temp_path("entity-registry");
        let path_text = path.to_string_lossy().to_string();
        let mut entity_registry = EntityRegistry::new();
        entity_registry.confirm_alias("苹果", "苹果公司");
        std::fs::write(&path_text, entity_registry.to_json()).unwrap();
        assert_eq!(
            audit(&path_text, ArtifactKind::EntityRegistry).status,
            ArtifactStatus::NonCanonical
        );
        assert_eq!(
            fix(&path_text, ArtifactKind::EntityRegistry)
                .unwrap()
                .status,
            ArtifactStatus::Canonical
        );
        let mut future: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path_text).unwrap()).unwrap();
        future["fold_version"] = serde_json::json!(FOLD_VERSION + 1);
        std::fs::write(&path_text, future.to_string()).unwrap();
        assert!(matches!(
            audit(&path_text, ArtifactKind::EntityRegistry).status,
            ArtifactStatus::Damaged(_)
        ));
        assert!(fix(&path_text, ArtifactKind::EntityRegistry).is_err());
        let _ = std::fs::remove_file(&path);

        let _ = version_gate(1, 1);
        assert!(version_gate(2, 1).is_err());
    }
}

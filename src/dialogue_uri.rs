//! DUR：对话工件 URI 层（Dialogue Artifact URI）。
//!
//! 编年史记录「第几轮说了什么」，本模块给这些产物补上稳定地址：
//! 会话、轮次、回答、命题、陈述、提及与证据各自有确定性 URI。URI 由
//! 会话标识 + 序号/下标/内容摘要构成，重启后只要编年史和会话标识还在，
//! 同一个对象就能被重新定位；回答文本按内容寻址得到不可变陈述 URI，
//! 后续轮次可精确引用而不依赖整段历史重放。
//!
//! 设计边界：
//! - 会话标识持久化在编年史旁的旁车文件，路径改名不改变身份；
//! - 旧编年史不需要迁移，投影时按当前会话标识重建工件索引；
//! - 不引入外部依赖，摘要沿用规范 JSON + FNV-1a 64（完整性自检，非密码学）；
//! - 未知命名空间显式报错，绝不把陌生 URI 静默当成实体。

use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::context_image::fnv1a_hex;
use crate::identity::encode_segment;

pub const DEFAULT_DIALOGUE_NS: &str = "http://example.org/kg/dialogue/";
pub const DIALOGUE_SESSION_VERSION: u32 = 1;
pub const SESSION_FILE_SUFFIX: &str = ".session.json";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    Session,
    Turn,
    Reply,
    Claim,
    Statement,
    Mention,
    Evidence,
}

impl ArtifactKind {
    pub const fn code(self) -> &'static str {
        match self {
            ArtifactKind::Session => "session",
            ArtifactKind::Turn => "turn",
            ArtifactKind::Reply => "reply",
            ArtifactKind::Claim => "claim",
            ArtifactKind::Statement => "statement",
            ArtifactKind::Mention => "mention",
            ArtifactKind::Evidence => "evidence",
        }
    }

    pub const fn label_zh(self) -> &'static str {
        match self {
            ArtifactKind::Session => "会话",
            ArtifactKind::Turn => "用户轮次",
            ArtifactKind::Reply => "系统回答",
            ArtifactKind::Claim => "回答命题",
            ArtifactKind::Statement => "内容寻址陈述",
            ArtifactKind::Mention => "实体提及",
            ArtifactKind::Evidence => "回答依据",
        }
    }

    pub fn from_code(code: &str) -> Option<Self> {
        match code {
            "session" => Some(ArtifactKind::Session),
            "turn" => Some(ArtifactKind::Turn),
            "reply" => Some(ArtifactKind::Reply),
            "claim" => Some(ArtifactKind::Claim),
            "statement" => Some(ArtifactKind::Statement),
            "mention" => Some(ArtifactKind::Mention),
            "evidence" => Some(ArtifactKind::Evidence),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UriError {
    ForeignNamespace(String),
    UnknownKind(String),
    Malformed(String),
    UnknownArtifact(String),
}

impl std::fmt::Display for UriError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UriError::ForeignNamespace(uri) => {
                write!(formatter, "不是对话工件命名空间：{}", uri)
            }
            UriError::UnknownKind(kind) => write!(formatter, "未知工件类型：{}", kind),
            UriError::Malformed(detail) => write!(formatter, "工件 URI 格式错误：{}", detail),
            UriError::UnknownArtifact(uri) => write!(formatter, "工件不存在：{}", uri),
        }
    }
}

impl std::error::Error for UriError {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DialogueRef {
    Session {
        session: String,
    },
    Turn {
        session: String,
        seq: u64,
    },
    Reply {
        session: String,
        seq: u64,
    },
    Claim {
        session: String,
        reply_seq: u64,
        index: u32,
    },
    Statement {
        session: String,
        digest: String,
    },
    Mention {
        session: String,
        turn_seq: u64,
        index: u32,
    },
    Evidence {
        session: String,
        digest: String,
    },
}

impl DialogueRef {
    pub fn kind(&self) -> ArtifactKind {
        match self {
            DialogueRef::Session { .. } => ArtifactKind::Session,
            DialogueRef::Turn { .. } => ArtifactKind::Turn,
            DialogueRef::Reply { .. } => ArtifactKind::Reply,
            DialogueRef::Claim { .. } => ArtifactKind::Claim,
            DialogueRef::Statement { .. } => ArtifactKind::Statement,
            DialogueRef::Mention { .. } => ArtifactKind::Mention,
            DialogueRef::Evidence { .. } => ArtifactKind::Evidence,
        }
    }

    pub fn session(&self) -> &str {
        match self {
            DialogueRef::Session { session }
            | DialogueRef::Turn { session, .. }
            | DialogueRef::Reply { session, .. }
            | DialogueRef::Claim { session, .. }
            | DialogueRef::Statement { session, .. }
            | DialogueRef::Mention { session, .. }
            | DialogueRef::Evidence { session, .. } => session,
        }
    }

    pub fn uri(&self) -> String {
        match self {
            DialogueRef::Session { session } => session_uri(session),
            DialogueRef::Turn { session, seq } => turn_uri(session, *seq),
            DialogueRef::Reply { session, seq } => reply_uri(session, *seq),
            DialogueRef::Claim {
                session,
                reply_seq,
                index,
            } => claim_uri(session, *reply_seq, *index),
            DialogueRef::Statement { session, digest } => statement_uri(session, digest),
            DialogueRef::Mention {
                session,
                turn_seq,
                index,
            } => mention_uri(session, *turn_seq, *index),
            DialogueRef::Evidence { session, digest } => evidence_uri(session, digest),
        }
    }
}

pub fn session_uri(session: &str) -> String {
    format!(
        "{}{}/{}",
        DEFAULT_DIALOGUE_NS,
        ArtifactKind::Session.code(),
        encode_segment(session.trim())
    )
}

pub fn turn_uri(session: &str, seq: u64) -> String {
    format!(
        "{}{}/{}/{}/{}",
        DEFAULT_DIALOGUE_NS,
        ArtifactKind::Session.code(),
        encode_segment(session.trim()),
        ArtifactKind::Turn.code(),
        seq
    )
}

pub fn reply_uri(session: &str, seq: u64) -> String {
    format!(
        "{}{}/{}/{}/{}",
        DEFAULT_DIALOGUE_NS,
        ArtifactKind::Session.code(),
        encode_segment(session.trim()),
        ArtifactKind::Reply.code(),
        seq
    )
}

pub fn claim_uri(session: &str, reply_seq: u64, index: u32) -> String {
    format!(
        "{}{}/{}/{}/{}/{}",
        DEFAULT_DIALOGUE_NS,
        ArtifactKind::Session.code(),
        encode_segment(session.trim()),
        ArtifactKind::Claim.code(),
        reply_seq,
        index
    )
}

pub fn statement_uri(session: &str, digest: &str) -> String {
    format!(
        "{}{}/{}/{}/{}",
        DEFAULT_DIALOGUE_NS,
        ArtifactKind::Session.code(),
        encode_segment(session.trim()),
        ArtifactKind::Statement.code(),
        digest
    )
}

pub fn mention_uri(session: &str, turn_seq: u64, index: u32) -> String {
    format!(
        "{}{}/{}/{}/{}/{}",
        DEFAULT_DIALOGUE_NS,
        ArtifactKind::Session.code(),
        encode_segment(session.trim()),
        ArtifactKind::Mention.code(),
        turn_seq,
        index
    )
}

pub fn evidence_uri(session: &str, digest: &str) -> String {
    format!(
        "{}{}/{}/{}/{}",
        DEFAULT_DIALOGUE_NS,
        ArtifactKind::Session.code(),
        encode_segment(session.trim()),
        ArtifactKind::Evidence.code(),
        digest
    )
}

pub fn content_digest(text: &str) -> String {
    fnv1a_hex(text.as_bytes())
}

pub fn statement_uri_for_text(session: &str, text: &str) -> String {
    statement_uri(session, &content_digest(text))
}

pub fn parse(uri: &str) -> Result<DialogueRef, UriError> {
    parse_with_namespace(DEFAULT_DIALOGUE_NS, uri)
}

pub fn parse_with_namespace(namespace: &str, uri: &str) -> Result<DialogueRef, UriError> {
    let rest = uri
        .strip_prefix(namespace)
        .ok_or_else(|| UriError::ForeignNamespace(uri.to_string()))?;
    let parts: Vec<&str> = rest.split('/').collect();
    if parts.len() < 2 || parts[0] != ArtifactKind::Session.code() {
        return Err(UriError::Malformed(uri.to_string()));
    }
    let session = crate::identity::decode_segment(parts[1])
        .filter(|session| !session.is_empty())
        .ok_or_else(|| UriError::Malformed(uri.to_string()))?;
    if parts.len() == 2 {
        return Ok(DialogueRef::Session { session });
    }
    let kind = ArtifactKind::from_code(parts[2])
        .ok_or_else(|| UriError::UnknownKind(parts[2].to_string()))?;
    match (kind, parts.len()) {
        (ArtifactKind::Session, _) => Err(UriError::Malformed(uri.to_string())),
        (ArtifactKind::Turn, 4) => Ok(DialogueRef::Turn {
            session,
            seq: parse_seq(parts[3], uri)?,
        }),
        (ArtifactKind::Reply, 4) => Ok(DialogueRef::Reply {
            session,
            seq: parse_seq(parts[3], uri)?,
        }),
        (ArtifactKind::Claim, 5) => Ok(DialogueRef::Claim {
            session,
            reply_seq: parse_seq(parts[3], uri)?,
            index: parse_index(parts[4], uri)?,
        }),
        (ArtifactKind::Statement, 4) => Ok(DialogueRef::Statement {
            session,
            digest: parse_digest(parts[3], uri)?,
        }),
        (ArtifactKind::Mention, 5) => Ok(DialogueRef::Mention {
            session,
            turn_seq: parse_seq(parts[3], uri)?,
            index: parse_index(parts[4], uri)?,
        }),
        (ArtifactKind::Evidence, 4) => Ok(DialogueRef::Evidence {
            session,
            digest: parse_digest(parts[3], uri)?,
        }),
        _ => Err(UriError::Malformed(uri.to_string())),
    }
}

fn parse_seq(segment: &str, uri: &str) -> Result<u64, UriError> {
    let seq: u64 = segment
        .parse()
        .map_err(|_| UriError::Malformed(uri.to_string()))?;
    if seq == 0 || seq.to_string() != segment {
        return Err(UriError::Malformed(uri.to_string()));
    }
    Ok(seq)
}

fn parse_index(segment: &str, uri: &str) -> Result<u32, UriError> {
    let index: u32 = segment
        .parse()
        .map_err(|_| UriError::Malformed(uri.to_string()))?;
    if index == 0 || index.to_string() != segment {
        return Err(UriError::Malformed(uri.to_string()));
    }
    Ok(index)
}

fn parse_digest(segment: &str, uri: &str) -> Result<String, UriError> {
    if segment.is_empty()
        || segment.len() > 64
        || !segment.bytes().all(|byte| byte.is_ascii_hexdigit())
        || segment != segment.to_ascii_lowercase()
    {
        return Err(UriError::Malformed(uri.to_string()));
    }
    Ok(segment.to_string())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct SessionRecord {
    session_version: u32,
    session_id: String,
}

#[derive(Debug, Clone)]
pub struct DialogueSession {
    id: String,
    path: PathBuf,
    created: bool,
}

impl DialogueSession {
    pub fn open_or_create(path: &str) -> std::io::Result<Self> {
        let path_buf = PathBuf::from(path);
        match Self::load(path)? {
            Some(session) => Ok(session),
            None => Self::create(&path_buf),
        }
    }

    pub fn load(path: &str) -> std::io::Result<Option<Self>> {
        let path_buf = PathBuf::from(path);
        let bytes = match std::fs::read(&path_buf) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error),
        };
        let record: SessionRecord = serde_json::from_slice(&bytes).map_err(|error| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("会话标识文件损坏（{}）：{}", path, error),
            )
        })?;
        if record.session_version > DIALOGUE_SESSION_VERSION {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "会话标识版本 {} 高于支持版本 {}",
                    record.session_version, DIALOGUE_SESSION_VERSION
                ),
            ));
        }
        let id = record.session_id.trim().to_string();
        if id.is_empty() || id.chars().any(char::is_control) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("会话标识文件中的 session_id 非法：{}", path),
            ));
        }
        Ok(Some(Self {
            id,
            path: path_buf,
            created: false,
        }))
    }

    fn create(path: &Path) -> std::io::Result<Self> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let id = uuid::Uuid::new_v4().to_string();
        let record = SessionRecord {
            session_version: DIALOGUE_SESSION_VERSION,
            session_id: id.clone(),
        };
        let body = serde_json::to_vec(&record)
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
        let tmp = PathBuf::from(format!(
            "{}.tmp-{}",
            path.to_string_lossy(),
            uuid::Uuid::new_v4()
        ));
        {
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&tmp)?;
            file.write_all(&body)?;
            file.sync_all()?;
        }
        match std::fs::hard_link(&tmp, path) {
            Ok(()) => {
                let _ = std::fs::remove_file(&tmp);
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                let _ = std::fs::remove_file(&tmp);
                return Self::load(&path.to_string_lossy())?.ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        format!("会话标识文件创建后被删除：{}", path.display()),
                    )
                });
            }
            Err(error) => {
                let _ = std::fs::remove_file(&tmp);
                return Err(error);
            }
        }
        Ok(Self {
            id,
            path: path.to_path_buf(),
            created: true,
        })
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn uri(&self) -> String {
        session_uri(&self.id)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn was_created(&self) -> bool {
        self.created
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uri_constructors_roundtrip_through_parser() {
        let session = "会话/一 号";
        let refs = vec![
            DialogueRef::Session {
                session: session.to_string(),
            },
            DialogueRef::Turn {
                session: session.to_string(),
                seq: 42,
            },
            DialogueRef::Reply {
                session: session.to_string(),
                seq: 43,
            },
            DialogueRef::Claim {
                session: session.to_string(),
                reply_seq: 43,
                index: 2,
            },
            DialogueRef::Statement {
                session: session.to_string(),
                digest: content_digest("姚明的身高是 226 厘米。"),
            },
            DialogueRef::Mention {
                session: session.to_string(),
                turn_seq: 42,
                index: 1,
            },
            DialogueRef::Evidence {
                session: session.to_string(),
                digest: content_digest("身高"),
            },
        ];
        for reference in refs {
            let uri = reference.uri();
            assert!(uri.starts_with(DEFAULT_DIALOGUE_NS), "{}", uri);
            assert!(!uri.contains(' '), "{}", uri);
            assert_eq!(parse(&uri).unwrap(), reference, "{}", uri);
            assert_eq!(parse(&uri).unwrap().uri(), uri);
        }
    }

    #[test]
    fn parser_rejects_foreign_or_malformed_uris() {
        assert!(matches!(
            parse("http://example.org/kg/entity/张三"),
            Err(UriError::ForeignNamespace(_))
        ));
        assert!(matches!(
            parse("http://example.org/kg/dialogue/session/s/turn/0"),
            Err(UriError::Malformed(_))
        ));
        assert!(matches!(
            parse("http://example.org/kg/dialogue/session/s/turn/01"),
            Err(UriError::Malformed(_))
        ));
        assert!(matches!(
            parse("http://example.org/kg/dialogue/session/s/unknown/1"),
            Err(UriError::UnknownKind(_))
        ));
        assert!(matches!(
            parse("http://example.org/kg/dialogue/session/s/claim/1/0"),
            Err(UriError::Malformed(_))
        ));
        assert!(matches!(
            parse("http://example.org/kg/dialogue/session/s/statement/XYZ"),
            Err(UriError::Malformed(_))
        ));
        assert!(matches!(
            parse("http://example.org/kg/dialogue/session/s/reply/1/extra"),
            Err(UriError::Malformed(_))
        ));
    }

    #[test]
    fn content_address_is_stable_for_same_text_and_distinct_for_changes() {
        assert_eq!(content_digest("同一句话"), content_digest("同一句话"));
        assert_ne!(content_digest("同一句话"), content_digest("同一句话。"));
        assert_eq!(
            statement_uri_for_text("s-1", "同一句话"),
            statement_uri_for_text("s-1", "同一句话")
        );
        assert_ne!(
            statement_uri_for_text("s-1", "同一句话"),
            statement_uri_for_text("s-2", "同一句话")
        );
    }

    #[test]
    fn session_identity_survives_reopen_and_path_move() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("dialogue.session.json");
        let path_text = path.to_string_lossy().to_string();
        let first = DialogueSession::open_or_create(&path_text).unwrap();
        assert!(first.was_created());
        let id = first.id().to_string();
        assert!(!id.is_empty());
        assert_eq!(first.uri(), session_uri(&id));

        let reopened = DialogueSession::open_or_create(&path_text).unwrap();
        assert!(!reopened.was_created());
        assert_eq!(reopened.id(), id);
        assert_eq!(reopened.uri(), first.uri());

        let moved = dir.path().join("renamed.session.json");
        std::fs::rename(&path, &moved).unwrap();
        let after_move = DialogueSession::open_or_create(&moved.to_string_lossy()).unwrap();
        assert_eq!(after_move.id(), id, "路径改名不得改变会话身份");
    }

    #[test]
    fn session_file_rejects_corruption_and_future_versions() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("dialogue.session.json");
        std::fs::write(&path, b"{not json").unwrap();
        assert!(DialogueSession::open_or_create(&path.to_string_lossy()).is_err());

        std::fs::write(
            &path,
            format!(
                "{{\"session_version\":{},\"session_id\":\"s-1\"}}",
                DIALOGUE_SESSION_VERSION + 1
            ),
        )
        .unwrap();
        let error = DialogueSession::load(&path.to_string_lossy()).unwrap_err();
        assert!(error.to_string().contains("高于支持版本"));
    }
}

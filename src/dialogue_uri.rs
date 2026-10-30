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

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::chronicle::{Chronicle, ChronicleEntry, EntityRef};

use crate::context_image::fnv1a_hex;
use crate::identity::{encode_segment, entity_uri, DEFAULT_BASE_NS};

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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MentionArtifact {
    pub index: u32,
    pub uri: String,
    pub surface: String,
    pub entity: Option<String>,
    pub entity_uri: Option<String>,
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnArtifact {
    pub seq: u64,
    pub uri: String,
    pub user: String,
    pub mode: String,
    pub focus: Option<String>,
    pub focus_uri: Option<String>,
    pub kb_revision: u64,
    pub mentions: Vec<MentionArtifact>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaimArtifact {
    pub index: u32,
    pub uri: String,
    pub statement_uri: String,
    pub digest: String,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceArtifact {
    pub uri: String,
    pub digest: String,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplyArtifact {
    pub seq: u64,
    pub uri: String,
    pub turn_seq: Option<u64>,
    pub question: Option<String>,
    pub answer: String,
    pub plan: Vec<String>,
    pub proof_id: Option<String>,
    pub kb_revision: u64,
    pub claims: Vec<ClaimArtifact>,
    pub evidence: Vec<EvidenceArtifact>,
}

#[derive(Debug, Clone)]
pub enum ResolvedArtifact<'a> {
    Session,
    Turn(&'a TurnArtifact),
    Reply(&'a ReplyArtifact),
    Claim {
        reply: &'a ReplyArtifact,
        claim: &'a ClaimArtifact,
    },
    Statement {
        digest: &'a str,
        text: &'a str,
    },
    Mention {
        turn: &'a TurnArtifact,
        mention: &'a MentionArtifact,
    },
    Evidence {
        digest: &'a str,
        text: &'a str,
    },
}

impl ResolvedArtifact<'_> {
    pub fn kind(&self) -> ArtifactKind {
        match self {
            ResolvedArtifact::Session => ArtifactKind::Session,
            ResolvedArtifact::Turn(_) => ArtifactKind::Turn,
            ResolvedArtifact::Reply(_) => ArtifactKind::Reply,
            ResolvedArtifact::Claim { .. } => ArtifactKind::Claim,
            ResolvedArtifact::Statement { .. } => ArtifactKind::Statement,
            ResolvedArtifact::Mention { .. } => ArtifactKind::Mention,
            ResolvedArtifact::Evidence { .. } => ArtifactKind::Evidence,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct DialogueArtifactIndex {
    session: String,
    turns: BTreeMap<u64, TurnArtifact>,
    replies: BTreeMap<u64, ReplyArtifact>,
    statements: BTreeMap<String, String>,
    evidences: BTreeMap<String, String>,
}

impl DialogueArtifactIndex {
    pub fn build(session: &str, entries: &[ChronicleEntry]) -> Self {
        let mut index = Self {
            session: session.trim().to_string(),
            ..Self::default()
        };
        for entry in entries {
            match entry.kind.as_str() {
                "turn" => index.add_turn(entry),
                "reply" => index.add_reply(entry),
                _ => {}
            }
        }
        index
    }

    pub fn from_chronicle(session: &DialogueSession, chronicle: &Chronicle) -> Self {
        Self::build(session.id(), chronicle.entries())
    }

    pub fn session(&self) -> &str {
        &self.session
    }

    pub fn session_uri(&self) -> String {
        session_uri(&self.session)
    }

    pub fn turn_count(&self) -> usize {
        self.turns.len()
    }

    pub fn reply_count(&self) -> usize {
        self.replies.len()
    }

    pub fn claim_count(&self) -> usize {
        self.replies.values().map(|reply| reply.claims.len()).sum()
    }

    pub fn evidence_count(&self) -> usize {
        self.replies
            .values()
            .map(|reply| reply.evidence.len())
            .sum()
    }

    pub fn turn(&self, seq: u64) -> Option<&TurnArtifact> {
        self.turns.get(&seq)
    }

    pub fn reply(&self, seq: u64) -> Option<&ReplyArtifact> {
        self.replies.get(&seq)
    }

    pub fn latest_turn(&self) -> Option<&TurnArtifact> {
        self.turns.values().next_back()
    }

    pub fn latest_query_turn(&self) -> Option<&TurnArtifact> {
        self.turns.values().rev().find(|turn| turn.mode == "query")
    }

    pub fn latest_reply(&self) -> Option<&ReplyArtifact> {
        self.replies.values().next_back()
    }

    pub fn claim(&self, reply_seq: u64, index: u32) -> Option<&ClaimArtifact> {
        self.reply(reply_seq)?
            .claims
            .iter()
            .find(|claim| claim.index == index)
    }

    pub fn statement(&self, digest: &str) -> Option<&str> {
        self.statements.get(digest).map(String::as_str)
    }

    pub fn evidence(&self, digest: &str) -> Option<&str> {
        self.evidences.get(digest).map(String::as_str)
    }

    pub fn resolve(&self, uri: &str) -> Result<ResolvedArtifact<'_>, UriError> {
        let reference = parse(uri)?;
        if reference.session() != self.session {
            return Err(UriError::UnknownArtifact(uri.to_string()));
        }
        match reference {
            DialogueRef::Session { .. } => Ok(ResolvedArtifact::Session),
            DialogueRef::Turn { seq, .. } => self
                .turns
                .get(&seq)
                .map(ResolvedArtifact::Turn)
                .ok_or_else(|| UriError::UnknownArtifact(uri.to_string())),
            DialogueRef::Reply { seq, .. } => self
                .replies
                .get(&seq)
                .map(ResolvedArtifact::Reply)
                .ok_or_else(|| UriError::UnknownArtifact(uri.to_string())),
            DialogueRef::Claim {
                reply_seq, index, ..
            } => {
                let reply = self
                    .replies
                    .get(&reply_seq)
                    .ok_or_else(|| UriError::UnknownArtifact(uri.to_string()))?;
                let claim = reply
                    .claims
                    .iter()
                    .find(|claim| claim.index == index)
                    .ok_or_else(|| UriError::UnknownArtifact(uri.to_string()))?;
                Ok(ResolvedArtifact::Claim { reply, claim })
            }
            DialogueRef::Statement { digest, .. } => self
                .statements
                .get_key_value(&digest)
                .map(|(digest, text)| ResolvedArtifact::Statement {
                    digest: digest.as_str(),
                    text: text.as_str(),
                })
                .ok_or_else(|| UriError::UnknownArtifact(uri.to_string())),
            DialogueRef::Mention {
                turn_seq, index, ..
            } => {
                let turn = self
                    .turns
                    .get(&turn_seq)
                    .ok_or_else(|| UriError::UnknownArtifact(uri.to_string()))?;
                let mention = turn
                    .mentions
                    .iter()
                    .find(|mention| mention.index == index)
                    .ok_or_else(|| UriError::UnknownArtifact(uri.to_string()))?;
                Ok(ResolvedArtifact::Mention { turn, mention })
            }
            DialogueRef::Evidence { digest, .. } => self
                .evidences
                .get_key_value(&digest)
                .map(|(digest, text)| ResolvedArtifact::Evidence {
                    digest: digest.as_str(),
                    text: text.as_str(),
                })
                .ok_or_else(|| UriError::UnknownArtifact(uri.to_string())),
        }
    }

    pub fn resolve_selector(&self, selector: &str) -> Result<ResolvedArtifact<'_>, String> {
        let spec = selector.trim();
        if spec.is_empty() {
            return Err("用法：工件：<URI|上一轮|第2点|依据>".to_string());
        }
        if spec.starts_with("http://")
            || spec.starts_with("https://")
            || spec.starts_with(DEFAULT_DIALOGUE_NS)
        {
            return self.resolve(spec).map_err(|error| error.to_string());
        }
        match spec {
            "上一轮" | "上一轮回答" | "上一条回答" | "最新回答" | "刚才的回答" => {
                self.latest_reply()
                    .map(ResolvedArtifact::Reply)
                    .ok_or_else(|| "还没有可引用的回答。".to_string())
            }
            "上一轮问题" | "上一个问题" | "最新问题" | "刚才的问题" => self
                .latest_query_turn()
                .map(ResolvedArtifact::Turn)
                .ok_or_else(|| "还没有可引用的查询轮次。".to_string()),
            "依据" | "证据" | "出处" => self
                .latest_reply()
                .map(ResolvedArtifact::Reply)
                .ok_or_else(|| "还没有可解释依据的回答。".to_string()),
            _ => {
                let index =
                    parse_ordinal(spec).ok_or_else(|| format!("无法识别工件选择「{}」。", spec))?;
                let reply = self
                    .latest_reply()
                    .ok_or_else(|| "还没有可引用的回答。".to_string())?;
                let claim = reply
                    .claims
                    .iter()
                    .find(|claim| claim.index == index)
                    .ok_or_else(|| {
                        format!(
                            "最新回答只有 {} 点，没有第 {} 点。",
                            reply.claims.len(),
                            index
                        )
                    })?;
                Ok(ResolvedArtifact::Claim { reply, claim })
            }
        }
    }

    pub fn render_report(&self) -> String {
        let mut output = String::new();
        output.push_str("对话工件索引\n");
        output.push_str(&format!("  会话：{}\n", self.session_uri()));
        output.push_str(&format!(
            "  计数：轮次 {}，回答 {}，命题 {}，依据 {}\n",
            self.turn_count(),
            self.reply_count(),
            self.claim_count(),
            self.evidence_count()
        ));
        let mut recent: Vec<(u64, String)> = Vec::new();
        for turn in self.turns.values() {
            recent.push((
                turn.seq,
                format!(
                    "  [轮次 {}] {}  用户：{}",
                    turn.seq,
                    turn.uri,
                    compact(&turn.user, 40)
                ),
            ));
        }
        for reply in self.replies.values() {
            let question = reply
                .question
                .as_deref()
                .map(|question| format!("，问题：{}", compact(question, 24)))
                .unwrap_or_default();
            recent.push((
                reply.seq,
                format!(
                    "  [回答 {}] {}{}  回答：{}",
                    reply.seq,
                    reply.uri,
                    question,
                    compact(&reply.answer, 40)
                ),
            ));
        }
        recent.sort_by_key(|(seq, _)| *seq);
        if recent.is_empty() {
            output.push_str("  最近：（还没有可寻址的对话工件）\n");
        } else {
            output.push_str("  最近：\n");
            for (_, line) in recent.iter().rev().take(8).rev() {
                output.push_str(line);
                output.push('\n');
            }
        }
        output.push_str(
            "  用法：工件：上一轮 / 工件：第2点 / 工件：依据 / 工件：实体：姚明 / 工件：<URI>\n",
        );
        output
    }

    pub fn render_selector(&self, selector: &str) -> String {
        let spec = selector.trim();
        if matches!(spec, "依据" | "证据" | "出处") {
            return self.render_latest_evidence();
        }
        if let Some(entity) = spec
            .strip_prefix("实体：")
            .or_else(|| spec.strip_prefix("实体:"))
            .or_else(|| spec.strip_prefix("实体 "))
            .or_else(|| spec.strip_prefix("entity:"))
            .or_else(|| spec.strip_prefix("entity "))
        {
            return self.render_entity_mentions(entity.trim());
        }
        if spec.starts_with(&format!("{}entity/", DEFAULT_BASE_NS)) {
            return self.render_entity_mentions(spec);
        }
        match self.resolve_selector(spec) {
            Ok(resolved) => self.render_resolved(&resolved),
            Err(error) => format!("无法解析工件选择「{}」：{}", spec, error),
        }
    }

    pub fn render_resolved(&self, resolved: &ResolvedArtifact<'_>) -> String {
        let mut output = String::new();
        match resolved {
            ResolvedArtifact::Session => {
                output.push_str(&format!("工件：{}\n", self.session_uri()));
                output.push_str(&format!("  类型：{}\n", ArtifactKind::Session.label_zh()));
                output.push_str(&self.render_report());
            }
            ResolvedArtifact::Turn(turn) => {
                output.push_str(&format!("工件：{}\n", turn.uri));
                output.push_str(&format!(
                    "  类型：{}（seq {}，KB 修订 {}）\n",
                    ArtifactKind::Turn.label_zh(),
                    turn.seq,
                    turn.kb_revision
                ));
                append_block(&mut output, "用户", &turn.user);
                append_block(&mut output, "模式", &turn.mode);
                if let Some(focus) = turn.focus.as_deref() {
                    match turn.focus_uri.as_deref() {
                        Some(uri) => {
                            append_block(&mut output, "焦点", &format!("{}（{}）", focus, uri))
                        }
                        None => append_block(&mut output, "焦点", focus),
                    }
                }
                if !turn.mentions.is_empty() {
                    output.push_str("  提及：\n");
                    for mention in &turn.mentions {
                        let entity =
                            match (mention.entity.as_deref(), mention.entity_uri.as_deref()) {
                                (Some(name), Some(uri)) => format!("{}（{}）", name, uri),
                                (Some(name), None) => name.to_string(),
                                _ => "（未链接）".to_string(),
                            };
                        output.push_str(&format!(
                            "    {}. {}  {} → {}（{}）\n",
                            mention.index, mention.uri, mention.surface, entity, mention.source
                        ));
                    }
                }
            }
            ResolvedArtifact::Reply(reply) => {
                output.push_str(&format!("工件：{}\n", reply.uri));
                let linked = reply
                    .turn_seq
                    .map(|seq| format!("，关联轮次 seq {}", seq))
                    .unwrap_or_default();
                output.push_str(&format!(
                    "  类型：{}（seq {}，KB 修订 {}{}）\n",
                    ArtifactKind::Reply.label_zh(),
                    reply.seq,
                    reply.kb_revision,
                    linked
                ));
                if let Some(question) = reply.question.as_deref() {
                    append_block(&mut output, "问题", question);
                }
                append_block(&mut output, "回答", &reply.answer);
                if !reply.plan.is_empty() {
                    append_block(&mut output, "计划", &reply.plan.join(" → "));
                }
                if !reply.claims.is_empty() {
                    output.push_str(&format!("  命题（{}）：\n", reply.claims.len()));
                    for claim in &reply.claims {
                        output.push_str(&format!(
                            "    第{}点：{}\n      陈述：{}\n      文本：{}\n",
                            claim.index, claim.uri, claim.statement_uri, claim.text
                        ));
                    }
                }
                if !reply.evidence.is_empty() {
                    output.push_str(&format!("  依据（{}）：\n", reply.evidence.len()));
                    for evidence in &reply.evidence {
                        output.push_str(&format!("    {}  {}\n", evidence.uri, evidence.text));
                    }
                }
                if let Some(proof_id) = reply.proof_id.as_deref() {
                    append_block(&mut output, "证明", proof_id);
                }
            }
            ResolvedArtifact::Claim { reply, claim } => {
                output.push_str(&format!("工件：{}\n", claim.uri));
                output.push_str(&format!(
                    "  类型：{}（回答 seq {}，第 {} 点）\n",
                    ArtifactKind::Claim.label_zh(),
                    reply.seq,
                    claim.index
                ));
                append_block(&mut output, "文本", &claim.text);
                append_block(&mut output, "陈述", &claim.statement_uri);
                append_block(&mut output, "回答", &reply.uri);
            }
            ResolvedArtifact::Statement { digest, text } => {
                output.push_str(&format!("工件：{}\n", statement_uri(&self.session, digest)));
                output.push_str(&format!(
                    "  类型：{}（摘要 {}）\n",
                    ArtifactKind::Statement.label_zh(),
                    digest
                ));
                append_block(&mut output, "文本", text);
            }
            ResolvedArtifact::Mention { turn, mention } => {
                output.push_str(&format!("工件：{}\n", mention.uri));
                output.push_str(&format!(
                    "  类型：{}（轮次 seq {}，第 {} 条）\n",
                    ArtifactKind::Mention.label_zh(),
                    turn.seq,
                    mention.index
                ));
                append_block(&mut output, "词面", &mention.surface);
                let entity = match (mention.entity.as_deref(), mention.entity_uri.as_deref()) {
                    (Some(name), Some(uri)) => format!("{}（{}）", name, uri),
                    (Some(name), None) => name.to_string(),
                    _ => "（未链接）".to_string(),
                };
                append_block(&mut output, "实体", &entity);
                append_block(&mut output, "来源", &mention.source);
                append_block(&mut output, "轮次", &turn.uri);
            }
            ResolvedArtifact::Evidence { digest, text } => {
                output.push_str(&format!("工件：{}\n", evidence_uri(&self.session, digest)));
                output.push_str(&format!(
                    "  类型：{}（摘要 {}）\n",
                    ArtifactKind::Evidence.label_zh(),
                    digest
                ));
                append_block(&mut output, "内容", text);
            }
        }
        output
    }

    pub fn mentions_of_entity(&self, entity: &str) -> Vec<(&TurnArtifact, &MentionArtifact)> {
        let needle = entity.trim();
        if needle.is_empty() {
            return Vec::new();
        }
        let mut hits = Vec::new();
        for turn in self.turns.values() {
            for mention in &turn.mentions {
                let matches = mention.surface == needle
                    || mention.entity.as_deref() == Some(needle)
                    || mention.entity_uri.as_deref() == Some(needle);
                if matches {
                    hits.push((turn, mention));
                }
            }
        }
        hits
    }

    pub fn replies_for_turn(&self, turn_seq: u64) -> Vec<&ReplyArtifact> {
        self.replies
            .values()
            .filter(|reply| reply.turn_seq == Some(turn_seq))
            .collect()
    }

    pub fn render_entity_mentions(&self, entity: &str) -> String {
        let entity = entity.trim();
        if entity.is_empty() {
            return "用法：工件：实体：<实体名或实体 URI>".to_string();
        }
        let hits = self.mentions_of_entity(entity);
        if hits.is_empty() {
            return format!("没有找到实体「{}」的对话提及。", entity);
        }
        let entity_uri = if entity.starts_with(DEFAULT_BASE_NS) {
            entity.to_string()
        } else {
            hits.iter()
                .find_map(|(_, mention)| mention.entity_uri.clone())
                .unwrap_or_else(|| entity_uri(DEFAULT_BASE_NS, entity))
        };
        let mut output = format!(
            "实体「{}」的对话工件：{} 条提及\n  实体 URI：{}\n",
            entity,
            hits.len(),
            entity_uri
        );
        for (position, (turn, mention)) in hits.iter().enumerate() {
            output.push_str(&format!(
                "  {}. {}  {}（第 {} 条，{}）\n",
                position + 1,
                mention.uri,
                mention.surface,
                mention.index,
                mention.source
            ));
            output.push_str(&format!("     轮次：{}\n", turn.uri));
            let replies = self.replies_for_turn(turn.seq);
            if replies.is_empty() {
                output.push_str("     回答：无\n");
            } else {
                for reply in replies {
                    output.push_str(&format!(
                        "     回答：{}  {}\n",
                        reply.uri,
                        compact(&reply.answer, 40)
                    ));
                }
            }
        }
        output
    }

    pub fn render_latest_evidence(&self) -> String {
        let Some(reply) = self.latest_reply() else {
            return "还没有可解释依据的回答。".to_string();
        };
        if reply.evidence.is_empty() {
            return format!("最新回答 {} 没有登记依据。", reply.uri);
        }
        let mut output = format!("最新回答 {} 的依据：\n", reply.uri);
        for (position, evidence) in reply.evidence.iter().enumerate() {
            output.push_str(&format!(
                "  {}. {}  {}\n",
                position + 1,
                evidence.uri,
                evidence.text
            ));
        }
        output
    }

    fn add_turn(&mut self, entry: &ChronicleEntry) {
        let user = payload_str(entry, "user").unwrap_or_default().to_string();
        let mode = payload_str(entry, "mode").unwrap_or("unknown").to_string();
        let focus = payload_str(entry, "focus")
            .map(str::trim)
            .filter(|focus| !focus.is_empty())
            .map(str::to_string);
        let refs: Vec<EntityRef> = entry
            .payload
            .get("refs")
            .and_then(|value| serde_json::from_value(value.clone()).ok())
            .unwrap_or_default();
        let mut mentions = Vec::new();
        for reference in &refs {
            let surface = reference.surface.trim();
            if surface.is_empty() {
                continue;
            }
            let index = mentions.len() as u32 + 1;
            let entity = reference
                .entity
                .as_deref()
                .map(str::trim)
                .filter(|entity| !entity.is_empty())
                .map(str::to_string);
            mentions.push(MentionArtifact {
                index,
                uri: mention_uri(&self.session, entry.seq, index),
                surface: surface.to_string(),
                entity_uri: entity
                    .as_deref()
                    .map(|entity| entity_uri(DEFAULT_BASE_NS, entity)),
                entity,
                source: reference.source.clone(),
            });
        }
        if let Some(focus) = focus.as_deref() {
            if !mentions.iter().any(|mention| mention.surface == focus) {
                let index = mentions.len() as u32 + 1;
                mentions.push(MentionArtifact {
                    index,
                    uri: mention_uri(&self.session, entry.seq, index),
                    surface: focus.to_string(),
                    entity: None,
                    entity_uri: None,
                    source: "focus".to_string(),
                });
            }
        }
        self.turns.insert(
            entry.seq,
            TurnArtifact {
                seq: entry.seq,
                uri: turn_uri(&self.session, entry.seq),
                user,
                mode,
                focus_uri: focus
                    .as_deref()
                    .map(|focus| entity_uri(DEFAULT_BASE_NS, focus)),
                focus,
                kb_revision: entry.kb_revision,
                mentions,
            },
        );
    }

    fn add_reply(&mut self, entry: &ChronicleEntry) {
        let answer = payload_str(entry, "answer").unwrap_or_default().to_string();
        let observed = entry.payload.get("observed_seq").and_then(Value::as_u64);
        let turn_seq = observed
            .filter(|seq| self.turns.contains_key(seq))
            .or_else(|| {
                self.turns
                    .range(..entry.seq)
                    .next_back()
                    .map(|(seq, _)| *seq)
            });
        let question = payload_str(entry, "question")
            .map(str::trim)
            .filter(|question| !question.is_empty())
            .map(str::to_string)
            .or_else(|| {
                turn_seq
                    .and_then(|seq| self.turns.get(&seq))
                    .map(|turn| turn.user.clone())
                    .filter(|user| !user.is_empty())
            });
        let plan = payload_strings(entry, "plan");
        let provenance = payload_strings(entry, "provenance");
        let proof_id = payload_str(entry, "proof_id")
            .map(str::trim)
            .filter(|proof| !proof.is_empty())
            .map(str::to_string);
        let mut claims = Vec::new();
        for (position, text) in answer_units(&answer).into_iter().enumerate() {
            let index = position as u32 + 1;
            let digest = content_digest(&text);
            let statement = statement_uri(&self.session, &digest);
            claims.push(ClaimArtifact {
                index,
                uri: claim_uri(&self.session, entry.seq, index),
                statement_uri: statement,
                digest: digest.clone(),
                text: text.clone(),
            });
            self.statements.entry(digest).or_insert(text);
        }
        let mut evidence = Vec::new();
        let mut seen = std::collections::BTreeSet::new();
        for text in provenance {
            let text = text.trim();
            if text.is_empty() {
                continue;
            }
            let digest = content_digest(text);
            if !seen.insert(digest.clone()) {
                continue;
            }
            evidence.push(EvidenceArtifact {
                uri: evidence_uri(&self.session, &digest),
                digest: digest.clone(),
                text: text.to_string(),
            });
            self.evidences
                .entry(digest)
                .or_insert_with(|| text.to_string());
        }
        self.replies.insert(
            entry.seq,
            ReplyArtifact {
                seq: entry.seq,
                uri: reply_uri(&self.session, entry.seq),
                turn_seq,
                question,
                answer,
                plan,
                proof_id,
                kb_revision: entry.kb_revision,
                claims,
                evidence,
            },
        );
    }
}

fn payload_str<'a>(entry: &'a ChronicleEntry, key: &str) -> Option<&'a str> {
    entry.payload.get(key).and_then(Value::as_str)
}

fn payload_strings(entry: &ChronicleEntry, key: &str) -> Vec<String> {
    entry
        .payload
        .get(key)
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn answer_units(answer: &str) -> Vec<String> {
    let mut units: Vec<String> = answer
        .lines()
        .map(str::trim)
        .map(strip_list_marker)
        .filter(|line| !line.is_empty())
        .map(str::to_string)
        .collect();
    if units.is_empty() {
        let trimmed = answer.trim();
        if !trimmed.is_empty() {
            units.push(trimmed.to_string());
        }
    }
    units
}

fn strip_list_marker(line: &str) -> &str {
    let line = line.trim();
    for marker in ["- ", "* ", "• ", "· "] {
        if let Some(rest) = line.strip_prefix(marker) {
            return rest.trim_start();
        }
    }
    let bytes = line.as_bytes();
    let mut end = 0;
    while end < bytes.len() && bytes[end].is_ascii_digit() {
        end += 1;
    }
    if end == 0 || end >= line.len() {
        return line;
    }
    let separator = line[end..].chars().next().unwrap_or_default();
    if !matches!(separator, '.' | '、' | ')' | '）') {
        return line;
    }
    let rest = &line[end + separator.len_utf8()..];
    if rest.is_empty() || rest.chars().next().is_some_and(|ch| ch.is_ascii_digit()) {
        return line;
    }
    rest.trim_start()
}

fn parse_ordinal(spec: &str) -> Option<u32> {
    let spec = spec.trim();
    let digits = spec
        .strip_prefix('第')
        .unwrap_or(spec)
        .trim()
        .trim_end_matches(['点', '条', '项', '个', '句'])
        .trim();
    if let Ok(number) = digits.parse::<u32>() {
        return (number > 0).then_some(number);
    }
    chinese_number(digits)
}

fn chinese_number(text: &str) -> Option<u32> {
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    if text == "十" {
        return Some(10);
    }
    if let Some((tens, ones)) = text.split_once('十') {
        let tens = if tens.is_empty() {
            1
        } else {
            chinese_digit(tens)?
        };
        let ones = if ones.is_empty() {
            0
        } else {
            chinese_digit(ones)?
        };
        let value = tens * 10 + ones;
        return (value > 0).then_some(value);
    }
    let value = chinese_digit(text)?;
    (value > 0).then_some(value)
}

fn chinese_digit(text: &str) -> Option<u32> {
    if text.chars().count() != 1 {
        return None;
    }
    match text.chars().next()? {
        '零' => Some(0),
        '一' => Some(1),
        '二' | '两' => Some(2),
        '三' => Some(3),
        '四' => Some(4),
        '五' => Some(5),
        '六' => Some(6),
        '七' => Some(7),
        '八' => Some(8),
        '九' => Some(9),
        _ => None,
    }
}

fn compact(text: &str, limit: usize) -> String {
    let joined = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut compact: String = joined.chars().take(limit).collect();
    if joined.chars().count() > limit {
        compact.push('…');
    }
    compact
}

fn append_block(output: &mut String, label: &str, text: &str) {
    if text.is_empty() {
        output.push_str(&format!("  {}：（空）\n", label));
        return;
    }
    let mut lines = text.lines();
    if let Some(first) = lines.next() {
        output.push_str(&format!("  {}：{}\n", label, first));
    }
    for line in lines {
        output.push_str(&format!("    {}\n", line));
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
    fn answer_units_keep_content_and_strip_only_list_markers() {
        let answer = "1. 第一点\n- 第二点\n2、第三点\n1.5 米不是列表\n• 第四点";
        assert_eq!(
            answer_units(answer),
            vec![
                "第一点".to_string(),
                "第二点".to_string(),
                "第三点".to_string(),
                "1.5 米不是列表".to_string(),
                "第四点".to_string(),
            ]
        );
        assert_eq!(answer_units("  "), Vec::<String>::new());
        assert_eq!(answer_units("单句回答"), vec!["单句回答".to_string()]);
    }

    #[test]
    fn ordinal_parsing_accepts_arabic_and_chinese() {
        assert_eq!(parse_ordinal("第2点"), Some(2));
        assert_eq!(parse_ordinal("第二点"), Some(2));
        assert_eq!(parse_ordinal("第十二条"), Some(12));
        assert_eq!(parse_ordinal("二十"), Some(20));
        assert_eq!(parse_ordinal("第十"), Some(10));
        assert_eq!(parse_ordinal("第0点"), None);
        assert_eq!(parse_ordinal("第N点"), None);
    }

    fn sample_index(dir: &tempfile::TempDir) -> (DialogueSession, DialogueArtifactIndex, String) {
        let session_path = dir.path().join("dialogue.session.json");
        let session = DialogueSession::open_or_create(&session_path.to_string_lossy()).unwrap();
        let chronicle_path = dir.path().join("dialogue.chronicle.jsonl");
        let chronicle_path_text = chronicle_path.to_string_lossy().to_string();
        let (mut chronicle, _) =
            Chronicle::open(&chronicle_path_text, crate::chronicle::FsyncPolicy::None).unwrap();
        chronicle
            .append_turn_with_refs(
                "姚明多高",
                "query",
                None,
                0,
                &[EntityRef::linked("姚明", "姚明")],
            )
            .unwrap();
        chronicle
            .append_reply(
                "1. 姚明的身高是 226 厘米。\n2. 他是篮球运动员。",
                &["direct_answer".to_string()],
                &["身高".to_string(), "职业".to_string()],
            )
            .unwrap();
        chronicle
            .append("trace", serde_json::json!({"ignored": true}))
            .unwrap();
        let index = DialogueArtifactIndex::from_chronicle(&session, &chronicle);
        (session, index, chronicle_path_text)
    }

    #[test]
    fn index_projects_turns_replies_claims_mentions_and_evidence() {
        let dir = tempfile::TempDir::new().unwrap();
        let (session, index, _) = sample_index(&dir);
        assert_eq!(index.turn_count(), 1);
        assert!(index.latest_turn().unwrap().focus.is_none());
        assert_eq!(index.reply_count(), 1);
        assert_eq!(index.claim_count(), 2);
        assert_eq!(index.evidence_count(), 2);

        let turn = index.latest_turn().unwrap();
        assert_eq!(turn.seq, 1);
        assert_eq!(turn.user, "姚明多高");
        assert_eq!(turn.mentions.len(), 1);
        assert_eq!(turn.mentions[0].surface, "姚明");
        assert_eq!(turn.mentions[0].entity.as_deref(), Some("姚明"));
        assert_eq!(
            turn.mentions[0].entity_uri.as_deref(),
            Some(entity_uri(DEFAULT_BASE_NS, "姚明").as_str())
        );
        assert!(matches!(
            index.resolve(&turn.uri).unwrap(),
            ResolvedArtifact::Turn(_)
        ));

        let reply = index.latest_reply().unwrap();
        assert_eq!(reply.seq, 2);
        assert_eq!(reply.turn_seq, Some(1));
        assert_eq!(reply.question.as_deref(), Some("姚明多高"));
        assert_eq!(reply.claims.len(), 2);
        assert_eq!(reply.claims[0].text, "姚明的身高是 226 厘米。");
        assert_eq!(reply.claims[1].text, "他是篮球运动员。");
        assert_eq!(reply.claims[1].index, 2);
        assert_eq!(
            reply.claims[0].statement_uri,
            statement_uri_for_text(session.id(), "姚明的身高是 226 厘米。")
        );
        assert_eq!(
            index.statement(&reply.claims[0].digest).unwrap(),
            "姚明的身高是 226 厘米。"
        );

        let claim = index.resolve(&reply.claims[1].uri).unwrap();
        match claim {
            ResolvedArtifact::Claim {
                reply: parent,
                claim,
            } => {
                assert_eq!(parent.seq, 2);
                assert_eq!(claim.text, "他是篮球运动员。");
            }
            other => panic!("应解析为命题：{:?}", other.kind()),
        }
        assert!(matches!(
            index.resolve(&reply.evidence[0].uri).unwrap(),
            ResolvedArtifact::Evidence { .. }
        ));
        assert!(matches!(
            index.resolve(&turn.mentions[0].uri).unwrap(),
            ResolvedArtifact::Mention { .. }
        ));
        assert!(index.render_report().contains("对话工件索引"));
        assert!(index.render_selector("第2点").contains("篮球运动员"));
        assert!(index.render_selector("第二点").contains("篮球运动员"));
        assert!(index.render_selector("上一轮").contains("226 厘米"));
        assert!(index.render_selector("依据").contains("身高"));
        assert!(index.render_selector("上一轮问题").contains("姚明多高"));
        assert!(index
            .render_selector(&reply.claims[0].uri)
            .contains("226 厘米"));
        assert!(index.render_selector("第9点").contains("没有第 9 点"));
    }

    #[test]
    fn index_rebuild_after_restart_keeps_the_same_addresses() {
        let dir = tempfile::TempDir::new().unwrap();
        let (session, first, chronicle_path) = sample_index(&dir);
        let reply_uri = first.latest_reply().unwrap().uri.clone();
        let claim_uri = first.latest_reply().unwrap().claims[1].uri.clone();
        let digest = first.latest_reply().unwrap().claims[1].digest.clone();
        let session_id = session.id().to_string();
        let session_path = session.path().to_string_lossy().to_string();
        drop(first);
        drop(session);

        let reopened_session = DialogueSession::open_or_create(&session_path).unwrap();
        assert_eq!(reopened_session.id(), session_id);
        let (chronicle, audit) =
            Chronicle::open(&chronicle_path, crate::chronicle::FsyncPolicy::None).unwrap();
        assert!(audit.is_clean());
        let restored = DialogueArtifactIndex::from_chronicle(&reopened_session, &chronicle);
        assert_eq!(restored.latest_reply().unwrap().uri, reply_uri);
        assert_eq!(restored.claim(2, 2).unwrap().uri, claim_uri);
        assert_eq!(restored.statement(&digest), Some("他是篮球运动员。"));
        assert!(restored.render_selector("上一轮").contains("226 厘米"));
    }

    #[test]
    fn resolve_rejects_other_sessions_and_unknown_artifacts() {
        let dir = tempfile::TempDir::new().unwrap();
        let (_, index, _) = sample_index(&dir);
        let foreign = reply_uri("another-session", 2);
        assert!(matches!(
            index.resolve(&foreign),
            Err(UriError::UnknownArtifact(_))
        ));
        let missing = claim_uri(index.session(), 2, 9);
        assert!(matches!(
            index.resolve(&missing),
            Err(UriError::UnknownArtifact(_))
        ));
    }

    #[test]
    fn entity_mentions_cross_reference_turns_and_replies() {
        let dir = tempfile::TempDir::new().unwrap();
        let (_, index, _) = sample_index(&dir);
        let hits = index.mentions_of_entity("姚明");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0.seq, 1);
        assert_eq!(hits[0].1.index, 1);
        assert_eq!(index.replies_for_turn(1).len(), 1);

        let by_name = index.render_selector("实体：姚明");
        assert!(by_name.contains("1 条提及"), "{}", by_name);
        assert!(by_name.contains(&hits[0].1.uri), "{}", by_name);
        assert!(by_name.contains(&hits[0].0.uri), "{}", by_name);
        assert!(by_name.contains("226 厘米"), "{}", by_name);

        let entity_uri = entity_uri(DEFAULT_BASE_NS, "姚明");
        let by_uri = index.render_selector(&entity_uri);
        assert!(by_uri.contains("1 条提及"), "{}", by_uri);
        assert!(by_uri.contains(&entity_uri), "{}", by_uri);
        assert!(index
            .render_selector("实体：不存在")
            .contains("没有找到实体"));
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

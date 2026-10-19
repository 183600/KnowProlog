//! Chronicle：对话编年史——追加式、哈希链、可回放/可检索的会话档案。
//!
//! 持久化哲学（与 CIM 快照互补）：
//!   - 快照（dialogue_state / context_image）是「最后一帧」，覆盖写、崩溃即丢整段；
//!   - 编年史是「真相源」：每一轮追加一行 JSONL，`hash = FNV(canonical(entry))`、
//!     且 `prev` 指向上一条的哈希，形成防篡改链。崩溃最多丢进行中的一轮；
//!   - 坏行只废尾巴不废全档：截断/翻字节/拔高版本三类损坏可区分检出，
//!     绝不静默采信；调用方用 `ChainAudit` 决定降级（保前缀 / 报坏下标 / 整体拒绝）；
//!   - 语义时戳是 `seq`（单调、可断言），不使用挂钟时间，保证同样的轮次序列
//!     在两个文件里逐字节一致，可做 golden 测试。
//!
//! 与 LLM 的类比：这不是「权重记忆」，而是白箱的、带证据链的对话日志——
//! 可重新加载、可回放任意水位、可全文检索命中哪一轮、可校验未被篡改。

use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::context_image::{canonical_json, fnv1a_hex};

/// 当前支持的编年史格式版本。
pub const CHRONICLE_VERSION: u32 = 1;

/// 每轮输入写入后是否立即 fsync。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FsyncPolicy {
    /// 每轮落盘即 fsync：崩溃最多丢进行中的一轮（交互式默认）。
    EachAppend,
    /// 交给操作系统：崩溃可能丢最近若干轮（批处理/测试用）。
    None,
}

/// 一条编年史记录。
///
/// `hash` 覆盖除自身以外的全部字段（含 `prev`），因此任何字段被改写都会
/// 在自检时被发现；`prev` 指向上一条的 `hash`，任何重排/删除都会断链。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChronicleEntry {
    pub seq: u64,
    pub version: u32,
    pub kind: String,
    pub payload: Value,
    /// 上一条的哈希；首条为空串。
    pub prev: String,
    /// 知识库修订号（导入/撤销自增），用于区分「重演」时 KB 是否已漂移。
    #[serde(default)]
    pub kb_revision: u64,
    /// 本条的链哈希。
    pub hash: String,
}

impl ChronicleEntry {
    /// 计算链哈希（规范 JSON + FNV-1a）。
    pub fn compute_hash(&self) -> String {
        let mut root = serde_json::Map::new();
        root.insert("kb_revision".to_string(), Value::from(self.kb_revision));
        root.insert("kind".to_string(), Value::from(self.kind.clone()));
        root.insert("payload".to_string(), self.payload.clone());
        root.insert("prev".to_string(), Value::from(self.prev.clone()));
        root.insert("seq".to_string(), Value::from(self.seq));
        root.insert("version".to_string(), Value::from(self.version as u64));
        fnv1a_hex(canonical_json(&Value::Object(root)).as_bytes())
    }

    /// 同一条记录（不含哈希字段）的规范文本，用于检索与展示。
    pub fn canonical_body(&self) -> String {
        canonical_json(&self.payload)
    }
}

/// 损坏类型：三类主要损坏各自可辨（截尾 / 翻字节 / 版本拔高），另有断链/乱序。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DamageKind {
    /// 行不是合法 JSON（含截断的半行）。
    BadJson,
    /// 行版本高于本程序支持版本。
    VersionTooNew,
    /// 重算哈希与记录不符（内容被改写）。
    HashMismatch,
    /// `prev` 与上一条哈希不符（重排/删行）。
    PrevMismatch,
    /// `seq` 不连续。
    SeqMismatch,
}

impl DamageKind {
    pub fn label(self) -> &'static str {
        match self {
            DamageKind::BadJson => "非法JSON（疑似截断/损坏）",
            DamageKind::VersionTooNew => "版本高于支持版本",
            DamageKind::HashMismatch => "内容哈希不匹配（被改写）",
            DamageKind::PrevMismatch => "链指针不匹配（重排/删行）",
            DamageKind::SeqMismatch => "序号不连续",
        }
    }
}

/// 首个损坏的行（0 基下标）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DamagedLine {
    pub line: usize,
    pub kind: DamageKind,
    pub detail: String,
}

/// 自检结果。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChainAudit {
    /// 解析器看到的行数。
    pub lines: usize,
    /// 链上可信的前缀长度（首个损坏之前的条数）。
    pub valid: usize,
    /// 首个损坏；None 表示全链可信。
    pub first_damage: Option<DamagedLine>,
    /// 损坏发生在最后一行（尾行截断的典型形态，前缀可安全保留）。
    pub tail_incomplete: bool,
}

impl ChainAudit {
    pub fn is_clean(&self) -> bool {
        self.first_damage.is_none()
    }

    /// 可安全保留的前缀条数。
    pub fn salvage_len(&self) -> usize {
        self.valid
    }

    /// 人类可读的一行结论。
    pub fn verdict(&self) -> String {
        match &self.first_damage {
            None => format!("全链可信（{} 条记录，哈希链完整）", self.valid),
            Some(damage) if self.tail_incomplete => format!(
                "尾行损坏（第 {} 行，{}，{}）：已保留前缀 {} 条，可截尾续写",
                damage.line + 1,
                damage.kind.label(),
                damage.detail,
                self.valid
            ),
            Some(damage) => format!(
                "链中损坏（第 {} 行，{}，{}）：只信任前 {} 条，其余拒绝采信",
                damage.line + 1,
                damage.kind.label(),
                damage.detail,
                self.valid
            ),
        }
    }
}

/// 检索命中：状态编号 + 命中次数 + 摘要。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChronicleMatch {
    pub seq: u64,
    pub kind: String,
    pub hits: usize,
    pub excerpt: String,
}

/// 追加式哈希链对话档案。
pub struct Chronicle {
    path: PathBuf,
    entries: Vec<ChronicleEntry>,
    fsync: FsyncPolicy,
    kb_revision: u64,
}

impl Chronicle {
    /// 打开（或创建）档案。返回已通过链校验的可信前缀与自检报告。
    ///
    /// 文件不存在视为空档案；存在但损坏时**只加载可信前缀**，
    /// 由调用方根据 `ChainAudit` 走截断重写或拒绝路径（绝不静默全量采信）。
    pub fn open(path: &str, fsync: FsyncPolicy) -> std::io::Result<(Self, ChainAudit)> {
        let bytes = match std::fs::read(path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok((
                    Chronicle {
                        path: PathBuf::from(path),
                        entries: Vec::new(),
                        fsync,
                        kb_revision: 0,
                    },
                    ChainAudit::default(),
                ))
            }
            Err(error) => return Err(error),
        };
        // 尾行可能被截断到多字节字符中间：lossy 转换保证仍能定位坏行。
        let text = String::from_utf8_lossy(&bytes);
        let (entries, audit) = Self::verify_text(&text);
        // 修订号随链延续：重启后未发生导入时不会误报 KB 漂移
        let kb_revision = entries.last().map(|entry| entry.kb_revision).unwrap_or(0);
        Ok((
            Chronicle {
                path: PathBuf::from(path),
                entries,
                fsync,
                kb_revision,
            },
            audit,
        ))
    }

    /// 打开并要求全链可信；任何损坏都返回错误（用于只读校验入口）。
    pub fn open_verified(path: &str) -> Result<Self, String> {
        let (chronicle, audit) =
            Self::open(path, FsyncPolicy::None).map_err(|error| format!("读取失败：{}", error))?;
        if !audit.is_clean() {
            return Err(audit.verdict());
        }
        Ok(chronicle)
    }

    /// 静态解析 + 链校验：返回可信前缀与自检报告。
    pub fn verify_text(text: &str) -> (Vec<ChronicleEntry>, ChainAudit) {
        let raw_lines: Vec<&str> = text.lines().collect();
        let mut entries: Vec<ChronicleEntry> = Vec::new();
        let mut damage: Option<DamagedLine> = None;
        for (index, line) in raw_lines.iter().enumerate() {
            if damage.is_some() {
                break;
            }
            if line.trim().is_empty() {
                damage = Some(DamagedLine {
                    line: index,
                    kind: DamageKind::BadJson,
                    detail: "空行".to_string(),
                });
                break;
            }
            let entry = match serde_json::from_str::<ChronicleEntry>(line) {
                Ok(entry) => entry,
                Err(error) => {
                    damage = Some(DamagedLine {
                        line: index,
                        kind: DamageKind::BadJson,
                        detail: error.to_string(),
                    });
                    break;
                }
            };
            if entry.version > CHRONICLE_VERSION {
                damage = Some(DamagedLine {
                    line: index,
                    kind: DamageKind::VersionTooNew,
                    detail: format!(
                        "记录版本 {} 高于支持版本 {}",
                        entry.version, CHRONICLE_VERSION
                    ),
                });
                break;
            }
            let expected_hash = entry.compute_hash();
            if entry.hash != expected_hash {
                damage = Some(DamagedLine {
                    line: index,
                    kind: DamageKind::HashMismatch,
                    detail: format!("存储 {} ≠ 重算 {}", entry.hash, expected_hash),
                });
                break;
            }
            match entries.last() {
                None => {
                    if entry.seq != 1 {
                        damage = Some(DamagedLine {
                            line: index,
                            kind: DamageKind::SeqMismatch,
                            detail: format!("首条序号应为 1，实为 {}", entry.seq),
                        });
                        break;
                    }
                }
                Some(previous) => {
                    if entry.seq != previous.seq + 1 {
                        damage = Some(DamagedLine {
                            line: index,
                            kind: DamageKind::SeqMismatch,
                            detail: format!("序号 {} 应为 {}", entry.seq, previous.seq + 1),
                        });
                        break;
                    }
                    if entry.prev != previous.hash {
                        damage = Some(DamagedLine {
                            line: index,
                            kind: DamageKind::PrevMismatch,
                            detail: "prev 与上一条哈希不符".to_string(),
                        });
                        break;
                    }
                }
            }
            entries.push(entry);
        }
        let lines = raw_lines.len();
        let valid = entries.len();
        let tail_incomplete = damage
            .as_ref()
            .map(|damage| damage.line + 1 == lines)
            .unwrap_or(false);
        (
            entries,
            ChainAudit {
                lines,
                valid,
                first_damage: damage,
                tail_incomplete,
            },
        )
    }

    /// 审计磁盘上的档案而不改变自身状态。
    pub fn audit_file(path: &str) -> std::io::Result<ChainAudit> {
        match std::fs::read(path) {
            Ok(bytes) => Ok(Self::verify_text(&String::from_utf8_lossy(&bytes)).1),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                Ok(ChainAudit::default())
            }
            Err(error) => Err(error),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn entries(&self) -> &[ChronicleEntry] {
        &self.entries
    }

    pub fn last_seq(&self) -> u64 {
        self.entries.last().map(|entry| entry.seq).unwrap_or(0)
    }

    pub fn last_hash(&self) -> &str {
        self.entries
            .last()
            .map(|entry| entry.hash.as_str())
            .unwrap_or("")
    }

    /// 水位：(最后序号, 最后哈希)。可写入快照以计算「快照之后新增的尾巴」。
    pub fn watermark(&self) -> (u64, String) {
        (self.last_seq(), self.last_hash().to_string())
    }

    pub fn kb_revision(&self) -> u64 {
        self.kb_revision
    }

    pub fn set_kb_revision(&mut self, revision: u64) {
        self.kb_revision = revision;
    }

    /// 追加一条记录；返回其 seq。写入后按策略 fsync。
    pub fn append(&mut self, kind: &str, payload: Value) -> std::io::Result<u64> {
        let seq = self.last_seq() + 1;
        let prev = self.last_hash().to_string();
        let mut entry = ChronicleEntry {
            seq,
            version: CHRONICLE_VERSION,
            kind: kind.to_string(),
            payload,
            prev,
            kb_revision: self.kb_revision,
            hash: String::new(),
        };
        entry.hash = entry.compute_hash();
        self.write_line(&entry)?;
        self.entries.push(entry);
        Ok(seq)
    }

    /// 便捷追加：一轮用户输入。
    pub fn append_turn(
        &mut self,
        user: &str,
        mode: &str,
        focus: Option<&str>,
        dialogue_turns: usize,
    ) -> std::io::Result<u64> {
        self.append(
            "turn",
            serde_json::json!({
                "user": user,
                "mode": mode,
                "focus": focus,
                "dialogue_turns": dialogue_turns as u64,
            }),
        )
    }

    /// 便捷追加：一次系统回答（尽力而为；未接管的管线不记录）。
    pub fn append_reply(
        &mut self,
        answer: &str,
        plan: &[String],
        provenance: &[String],
    ) -> std::io::Result<u64> {
        self.append(
            "reply",
            serde_json::json!({
                "answer": answer,
                "plan": plan,
                "provenance": provenance,
            }),
        )
    }

    fn write_line(&self, entry: &ChronicleEntry) -> std::io::Result<()> {
        if let Some(parent) = self.path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let line = serde_json::to_string(entry)
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::Other, error))?;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        file.write_all(line.as_bytes())?;
        file.write_all(b"\n")?;
        if self.fsync == FsyncPolicy::EachAppend {
            file.sync_all()?;
        }
        Ok(())
    }

    /// 原子重写整档（临时文件 + rename）。重写损坏尾巴 / 截断超水位后调用。
    pub fn rewrite_atomic(&self) -> std::io::Result<()> {
        if let Some(parent) = self.path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let mut body = String::new();
        for entry in &self.entries {
            let line = serde_json::to_string(entry)
                .map_err(|error| std::io::Error::new(std::io::ErrorKind::Other, error))?;
            body.push_str(&line);
            body.push('\n');
        }
        let tmp = format!("{}.tmp", self.path.display());
        {
            let mut file = std::fs::File::create(&tmp)?;
            file.write_all(body.as_bytes())?;
            file.sync_all()?;
        }
        std::fs::rename(&tmp, &self.path)
    }

    /// 重写：把磁盘内容重写为当前可信前缀（用于尾行截断后的续写）。
    pub fn repair_tail(&self) -> std::io::Result<()> {
        self.rewrite_atomic()
    }

    /// 截断到指定序号（含），并原子重写。返回实际保留条数。
    pub fn truncate_after(&mut self, seq: u64) -> std::io::Result<usize> {
        self.entries.retain(|entry| entry.seq <= seq);
        self.rewrite_atomic()?;
        Ok(self.entries.len())
    }

    /// 回放：序号 ≤ `until` 的记录（调试器「回放到第 N 轮」）。
    pub fn replay_to(&self, until: u64) -> Vec<&ChronicleEntry> {
        self.entries
            .iter()
            .filter(|entry| entry.seq <= until)
            .collect()
    }

    /// 全文检索：对 kind + 规范 payload 做大小写不敏感的计数命中。
    /// 返回按 seq 升序的命中清单，命中可审计（哪一轮、几条）。
    pub fn search(&self, query: &str) -> Vec<ChronicleMatch> {
        let needle = query.trim().to_lowercase();
        if needle.is_empty() {
            return Vec::new();
        }
        let mut matches: Vec<ChronicleMatch> = Vec::new();
        for entry in &self.entries {
            let haystack = format!("{} {}", entry.kind, entry.canonical_body()).to_lowercase();
            let hits = haystack.matches(&needle).count();
            if hits == 0 {
                continue;
            }
            matches.push(ChronicleMatch {
                seq: entry.seq,
                kind: entry.kind.clone(),
                hits,
                excerpt: Self::excerpt(entry),
            });
        }
        matches
    }

    fn excerpt(entry: &ChronicleEntry) -> String {
        let payload = &entry.payload;
        let user = payload
            .get("user")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let answer = payload
            .get("answer")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let text = if !user.is_empty() { user } else { answer };
        if text.is_empty() {
            return entry.canonical_body().chars().take(60).collect();
        }
        let mut excerpt: String = text.chars().take(60).collect();
        if text.chars().count() > 60 {
            excerpt.push('…');
        }
        excerpt
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn sample_chronicle(dir: &TempDir) -> (Chronicle, String) {
        let path = dir.path().join("dialogue.chronicle.jsonl");
        let path_text = path.to_string_lossy().to_string();
        let (mut chronicle, audit) = Chronicle::open(&path_text, FsyncPolicy::None).unwrap();
        assert!(audit.is_clean());
        chronicle
            .append_turn("姚明是谁", "query", None, 0)
            .unwrap();
        chronicle.set_kb_revision(7);
        chronicle
            .append_turn("他的身高呢", "query", Some("姚明"), 1)
            .unwrap();
        chronicle
            .append_reply("姚明的身高是 226 厘米。", &["direct_answer".to_string()], &["身高".to_string()])
            .unwrap();
        (chronicle, path_text)
    }

    #[test]
    fn append_reopen_roundtrip_is_byte_stable() {
        let dir = TempDir::new().unwrap();
        let (chronicle, path_text) = sample_chronicle(&dir);
        let bytes_before = std::fs::read(&path_text).unwrap();
        assert_eq!(chronicle.len(), 3);
        assert_eq!(chronicle.last_seq(), 3);
        drop(chronicle);

        let (reloaded, audit) = Chronicle::open(&path_text, FsyncPolicy::None).unwrap();
        assert!(audit.is_clean(), "{}", audit.verdict());
        assert_eq!(reloaded.len(), 3);
        assert_eq!(reloaded.watermark().0, 3);
        assert_eq!(std::fs::read(&path_text).unwrap(), bytes_before, "只读打开不得改写字节");
        assert_eq!(reloaded.entries()[1].kb_revision, 7);
        assert_eq!(reloaded.kb_revision(), 7, "修订号应随链延续到重启后的会话");
    }

    #[test]
    fn same_turns_produce_byte_identical_archives() {
        let dir = TempDir::new().unwrap();
        let (left, left_path) = sample_chronicle(&dir);
        let other = TempDir::new().unwrap();
        let (right, right_path) = sample_chronicle(&other);
        assert_eq!(left.entries(), right.entries());
        assert_eq!(
            std::fs::read(&left_path).unwrap(),
            std::fs::read(&right_path).unwrap(),
            "同样的轮次序列必须落盘为同一字节串（seq 为语义时戳）"
        );
    }

    #[test]
    fn middle_line_tamper_is_detected_with_index() {
        let dir = TempDir::new().unwrap();
        let (chronicle, path_text) = sample_chronicle(&dir);
        drop(chronicle);
        let text = std::fs::read_to_string(&path_text).unwrap();
        let mut lines: Vec<String> = text.lines().map(str::to_string).collect();
        lines[1] = lines[1].replace("他的身高呢", "他的体重呢");
        std::fs::write(&path_text, lines.join("\n") + "\n").unwrap();

        let (salvaged, audit) = Chronicle::open(&path_text, FsyncPolicy::None).unwrap();
        let damage = audit.first_damage.as_ref().expect("应检出损坏");
        assert_eq!(damage.line, 1);
        assert_eq!(damage.kind, DamageKind::HashMismatch);
        assert!(!audit.tail_incomplete, "中行损坏不是尾行截断");
        assert_eq!(salvaged.len(), 1, "只信任损坏行之前的前缀");
        assert!(audit.verdict().contains("第 2 行"));
    }

    #[test]
    fn tail_truncation_salvages_prefix_and_allows_repair() {
        let dir = TempDir::new().unwrap();
        let (chronicle, path_text) = sample_chronicle(&dir);
        drop(chronicle);
        let text = std::fs::read_to_string(&path_text).unwrap();
        let mut lines: Vec<&str> = text.lines().collect();
        let last = lines.pop().unwrap();
        let cut = &last[..last.len() / 2];
        std::fs::write(&path_text, format!("{}\n{}", lines.join("\n"), cut)).unwrap();

        let (mut salvaged, audit) = Chronicle::open(&path_text, FsyncPolicy::None).unwrap();
        assert!(audit.tail_incomplete, "{}", audit.verdict());
        assert_eq!(audit.salvage_len(), 2);
        let damage = audit.first_damage.as_ref().unwrap();
        assert_eq!(damage.line, 2);
        assert!(matches!(
            damage.kind,
            DamageKind::BadJson | DamageKind::HashMismatch
        ));

        salvaged.repair_tail().unwrap();
        let (repaired, audit) = Chronicle::open(&path_text, FsyncPolicy::None).unwrap();
        assert!(audit.is_clean());
        assert_eq!(repaired.len(), 2);
        let seq = salvaged
            .append_turn("他的国籍呢", "query", Some("姚明"), 2)
            .unwrap();
        assert_eq!(seq, 3, "截断后续写应从可信前缀继续编号");
        assert!(Chronicle::audit_file(&path_text).unwrap().is_clean());
    }

    #[test]
    fn version_bump_is_rejected_not_trusted() {
        let dir = TempDir::new().unwrap();
        let (chronicle, path_text) = sample_chronicle(&dir);
        drop(chronicle);
        let text = std::fs::read_to_string(&path_text).unwrap();
        let mut lines: Vec<String> = text.lines().map(str::to_string).collect();
        let mut entry: ChronicleEntry = serde_json::from_str(&lines[2]).unwrap();
        entry.version = CHRONICLE_VERSION + 1;
        entry.hash = entry.compute_hash();
        lines[2] = serde_json::to_string(&entry).unwrap();
        std::fs::write(&path_text, lines.join("\n") + "\n").unwrap();

        let (_, audit) = Chronicle::open(&path_text, FsyncPolicy::None).unwrap();
        let damage = audit.first_damage.as_ref().unwrap();
        assert_eq!(damage.line, 2);
        assert_eq!(damage.kind, DamageKind::VersionTooNew);
        assert_eq!(audit.valid, 2);
        assert!(matches!(
            Chronicle::open_verified(&path_text),
            Err(message) if message.contains("版本高于支持版本")
        ));
    }

    #[test]
    fn reordering_breaks_chain() {
        let dir = TempDir::new().unwrap();
        let (chronicle, path_text) = sample_chronicle(&dir);
        drop(chronicle);
        let text = std::fs::read_to_string(&path_text).unwrap();
        let mut lines: Vec<String> = text.lines().map(str::to_string).collect();
        lines.swap(1, 2);
        std::fs::write(&path_text, lines.join("\n") + "\n").unwrap();

        let (_, audit) = Chronicle::open(&path_text, FsyncPolicy::None).unwrap();
        let damage = audit.first_damage.as_ref().unwrap();
        assert_eq!(damage.line, 1);
        assert!(matches!(
            damage.kind,
            DamageKind::SeqMismatch | DamageKind::PrevMismatch
        ));
    }

    #[test]
    fn search_hits_are_auditable_and_ordered() {
        let dir = TempDir::new().unwrap();
        let (mut chronicle, _) = sample_chronicle(&dir);
        chronicle.append_turn("姚明和叶莉的关系", "query", Some("姚明"), 3).unwrap();
        let matches = chronicle.search("姚明");
        assert_eq!(matches.len(), 4, "{:?}", matches);
        assert_eq!(matches[0].seq, 1);
        assert_eq!(matches[1].seq, 2, "焦点字段命中");
        assert_eq!(matches[2].seq, 3, "回答正文命中");
        assert_eq!(matches[3].seq, 4, "按 seq 升序");
        assert!(matches[3].excerpt.contains("姚明"));

        // ASCII 大小写不敏感；空查询不命中
        chronicle.append("note", serde_json::json!({"text": "Use CIM Envelope"})).unwrap();
        assert_eq!(chronicle.search("envelope").len(), 1);
        assert!(chronicle.search("   ").is_empty());
    }

    #[test]
    fn replay_and_truncate_respect_watermark() {
        let dir = TempDir::new().unwrap();
        let (mut chronicle, path_text) = sample_chronicle(&dir);
        chronicle.append_turn("第三个问题", "query", Some("叶莉"), 3).unwrap();

        let replayed: Vec<u64> = chronicle.replay_to(2).iter().map(|e| e.seq).collect();
        assert_eq!(replayed, vec![1, 2]);

        let kept = chronicle.truncate_after(2).unwrap();
        assert_eq!(kept, 2);
        assert_eq!(chronicle.last_seq(), 2);
        let (reloaded, audit) = Chronicle::open(&path_text, FsyncPolicy::None).unwrap();
        assert!(audit.is_clean());
        assert_eq!(reloaded.len(), 2);
        assert_eq!(reloaded.last_hash(), chronicle.last_hash());
        assert!(!Path::new(&format!("{}.tmp", path_text)).exists());
    }

    #[test]
    fn each_append_policy_persists_before_next_turn() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("crash.chronicle.jsonl");
        let path_text = path.to_string_lossy().to_string();
        let (mut chronicle, _) = Chronicle::open(&path_text, FsyncPolicy::EachAppend).unwrap();
        chronicle.append_turn("第一轮", "query", None, 0).unwrap();
        chronicle.append_turn("第二轮", "query", Some("焦点"), 1).unwrap();

        // 模拟崩溃后重启：不调用任何保存逻辑，直接重新打开
        let (reloaded, audit) = Chronicle::open(&path_text, FsyncPolicy::None).unwrap();
        assert!(audit.is_clean());
        assert_eq!(reloaded.len(), 2);
        assert_eq!(reloaded.replay_to(1).len(), 1);
    }

    #[test]
    fn empty_and_missing_files_are_clean() {
        let dir = TempDir::new().unwrap();
        let missing = dir.path().join("missing.jsonl");
        let (chronicle, audit) =
            Chronicle::open(missing.to_str().unwrap(), FsyncPolicy::None).unwrap();
        assert!(chronicle.is_empty());
        assert!(audit.is_clean());
        assert!(Chronicle::audit_file(missing.to_str().unwrap())
            .unwrap()
            .is_clean());
    }
}

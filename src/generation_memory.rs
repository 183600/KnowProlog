//! GRM：生成记忆（Generation Memory）——把已说过的回答投影成可查询的 RDF 命名图。
//!
//! 编年史回答「第几轮说了什么」，DUR 回答「每个工件在哪里」。本模块补上第三问：
//! 「这些工件能否和知识图谱在同一条查询里组合」。每一轮对话落成独立的命名图，
//! 图中保留轮次、回答、命题、陈述、提及与依据的完整结构，同时附带规范 JSON，
//! 保证读回后与 `DialogueArtifactIndex` 逐字段一致。
//!
//! 设计边界：
//! - **按轮隔离**：一个轮次一个命名图，会话图只放索引与计数；忘记某轮只需删图；
//! - **幂等重写**：同步先清空该会话的全部生成记忆图，再全量写入，不推进知识版本；
//! - **默认图零污染**：所有四元组都落在 `kg:graphs/dialogue/` 下，全图扫描看不到；
//! - **有界投影**：调用方按最近 N 轮投影，避免记忆无限膨胀；
//! - **可追溯**：命题指向内容寻址陈述，回答指向轮次，生成史不再是黑箱文本。

use std::collections::{BTreeMap, BTreeSet};

use oxigraph::model::{GraphName, Literal, NamedNode, Quad, Subject, Term};
use serde::{Deserialize, Serialize};

use crate::chronicle::{answer_key, Chronicle};
use crate::dialogue_uri::{
    session_uri, turn_uri, ClaimArtifact, DialogueArtifactIndex, DialogueSession, ReplyArtifact,
    TurnArtifact,
};
use crate::identity::encode_segment;

/// 生成记忆格式版本。
pub const GENERATION_MEMORY_VERSION: u32 = 1;
/// 默认投影的最近轮次数。
pub const DEFAULT_MEMORY_TURNS: usize = 8;
/// 生成记忆命名图前缀（按会话隔离）。
pub const DIALOGUE_MEMORY_GRAPH_NS: &str = "http://example.org/kg/graphs/dialogue/";
/// 生成记忆账本专用命名空间（与知识图谱 `kg:` 词面隔离）。
pub const DIALOGUE_MEMORY_NS: &str = "http://example.org/kg/ledger/dialogue/";

const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
const SESSION_CLASS: &str = "http://example.org/kg/ledger/dialogue/Session";
const TURN_CLASS: &str = "http://example.org/kg/ledger/dialogue/Turn";
const REPLY_CLASS: &str = "http://example.org/kg/ledger/dialogue/Reply";
const CLAIM_CLASS: &str = "http://example.org/kg/ledger/dialogue/Claim";
const STATEMENT_CLASS: &str = "http://example.org/kg/ledger/dialogue/Statement";
const MENTION_CLASS: &str = "http://example.org/kg/ledger/dialogue/Mention";
const EVIDENCE_CLASS: &str = "http://example.org/kg/ledger/dialogue/Evidence";

const P_SESSION: &str = "http://example.org/kg/ledger/dialogue/session";
const P_VERSION: &str = "http://example.org/kg/ledger/dialogue/version";
const P_SEQ: &str = "http://example.org/kg/ledger/dialogue/seq";
const P_INDEX: &str = "http://example.org/kg/ledger/dialogue/index";
const P_USER: &str = "http://example.org/kg/ledger/dialogue/user";
const P_MODE: &str = "http://example.org/kg/ledger/dialogue/mode";
const P_FOCUS: &str = "http://example.org/kg/ledger/dialogue/focus";
const P_FOCUS_URI: &str = "http://example.org/kg/ledger/dialogue/focusUri";
const P_KB_REVISION: &str = "http://example.org/kg/ledger/dialogue/kbRevision";
const P_QUESTION: &str = "http://example.org/kg/ledger/dialogue/question";
const P_ANSWER: &str = "http://example.org/kg/ledger/dialogue/answer";
const P_PLAN: &str = "http://example.org/kg/ledger/dialogue/plan";
const P_PROOF_ID: &str = "http://example.org/kg/ledger/dialogue/proofId";
const P_TEXT: &str = "http://example.org/kg/ledger/dialogue/text";
const P_DIGEST: &str = "http://example.org/kg/ledger/dialogue/digest";
const P_STATEMENT: &str = "http://example.org/kg/ledger/dialogue/statement";
const P_SURFACE: &str = "http://example.org/kg/ledger/dialogue/surface";
const P_ENTITY: &str = "http://example.org/kg/ledger/dialogue/entity";
const P_ENTITY_URI: &str = "http://example.org/kg/ledger/dialogue/entityUri";
const P_SOURCE: &str = "http://example.org/kg/ledger/dialogue/source";
const P_TURN: &str = "http://example.org/kg/ledger/dialogue/turn";
const P_TURN_SEQ: &str = "http://example.org/kg/ledger/dialogue/turnSeq";
const P_REPLY: &str = "http://example.org/kg/ledger/dialogue/reply";
const P_HAS_TURN: &str = "http://example.org/kg/ledger/dialogue/hasTurn";
const P_HAS_REPLY: &str = "http://example.org/kg/ledger/dialogue/hasReply";
const P_HAS_CLAIM: &str = "http://example.org/kg/ledger/dialogue/hasClaim";
const P_HAS_EVIDENCE: &str = "http://example.org/kg/ledger/dialogue/hasEvidence";
const P_HAS_MENTION: &str = "http://example.org/kg/ledger/dialogue/hasMention";
const P_TURN_COUNT: &str = "http://example.org/kg/ledger/dialogue/turnCount";
const P_REPLY_COUNT: &str = "http://example.org/kg/ledger/dialogue/replyCount";
const P_LATEST_SEQ: &str = "http://example.org/kg/ledger/dialogue/latestSeq";
const P_JSON: &str = "http://example.org/kg/ledger/dialogue/json";
/// 更正关系：回答 → 其更正的先前命题 URI，按轮隔离、内容寻址。
const P_CORRECTS: &str = "http://example.org/kg/ledger/dialogue/corrects";

type MemoryResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// RDF 生成记忆配置（默认关闭；开启后每个轮次边界做一次有界投影）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerationMemoryConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_memory_turns")]
    pub max_turns: usize,
}

fn default_memory_turns() -> usize {
    DEFAULT_MEMORY_TURNS
}

impl Default for GenerationMemoryConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_turns: DEFAULT_MEMORY_TURNS,
        }
    }
}

/// 一个轮次及其回答（回答按 seq 升序）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryTurn {
    pub turn: TurnArtifact,
    pub replies: Vec<ReplyArtifact>,
}

/// 生成记忆：会话内可寻址的轮次、回答、命题、陈述、提及与依据。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerationMemory {
    pub session: String,
    pub turns: BTreeMap<u64, MemoryTurn>,
    /// 没有关联轮次的回答（仍保留，避免静默丢历史）。
    pub unattached_replies: BTreeMap<u64, ReplyArtifact>,
}

/// 会话图 IRI（索引与计数，不含正文）。
pub fn session_graph_iri(session: &str) -> String {
    format!(
        "{}{}/session",
        DIALOGUE_MEMORY_GRAPH_NS,
        encode_segment(session.trim())
    )
}

/// 会话生成记忆的命名图前缀（用于按会话查找与删除）。
pub fn session_graph_prefix(session: &str) -> String {
    format!(
        "{}{}/",
        DIALOGUE_MEMORY_GRAPH_NS,
        encode_segment(session.trim())
    )
}

/// 会话生成记忆的命名图 IRI 集合（用于 SPARQL GRAPH 子句）。
/// 返回最近 `max_turns` 轮的图 IRI，`0` 表示全部。
/// 这提供了一个有界的图集合，防止查询膨胀至全会话图。
pub fn session_graph_iris(max_turns: usize, session: &str) -> Vec<String> {
    let prefix = session_graph_prefix(session);
    if max_turns == 0 {
        // 完整列表由 OxigraphManager::generation_memory_graphs 从 store 实时获取
        // 此处返回占位，实际查询由 manager 层处理
        vec![]
    } else {
        (1..=max_turns)
            .map(|seq| format!("{}turn/{}", prefix, seq))
            .collect()
    }
}

/// 轮次命名图 IRI。
pub fn turn_graph_iri(session: &str, turn_seq: u64) -> String {
    format!("{}turn/{}", session_graph_prefix(session), turn_seq)
}

/// 无轮次回答的命名图 IRI。
pub fn reply_graph_iri(session: &str, reply_seq: u64) -> String {
    format!("{}reply/{}", session_graph_prefix(session), reply_seq)
}

/// 四元组是否属于生成记忆命名图。
pub fn is_memory_graph(iri: &str) -> bool {
    iri.starts_with(DIALOGUE_MEMORY_GRAPH_NS)
}

impl GenerationMemory {
    /// 空记忆（保留会话标识，用于「尚未记录」与「已忘记」）。
    pub fn empty(session: impl Into<String>) -> Self {
        GenerationMemory {
            session: session.into().trim().to_string(),
            ..Self::default()
        }
    }

    /// 从工件索引投影最近 `max_turns` 轮；`0` 表示不设上限。
    pub fn from_index(index: &DialogueArtifactIndex, max_turns: usize) -> Self {
        let session = index.session().trim().to_string();
        let turn_seqs: Vec<u64> = if max_turns == 0 {
            index.turns().map(|turn| turn.seq).collect()
        } else {
            let mut seqs: Vec<u64> = index
                .turns()
                .map(|turn| turn.seq)
                .rev()
                .take(max_turns)
                .collect();
            seqs.reverse();
            seqs
        };
        let selected: BTreeSet<u64> = turn_seqs.iter().copied().collect();
        let mut turns: BTreeMap<u64, MemoryTurn> = BTreeMap::new();
        for turn in index.turns().filter(|turn| selected.contains(&turn.seq)) {
            turns.insert(
                turn.seq,
                MemoryTurn {
                    turn: turn.clone(),
                    replies: Vec::new(),
                },
            );
        }
        let mut unattached_replies: BTreeMap<u64, ReplyArtifact> = BTreeMap::new();
        for reply in index.replies() {
            match reply.turn_seq {
                Some(turn_seq) if selected.contains(&turn_seq) => {
                    if let Some(stored) = turns.get_mut(&turn_seq) {
                        stored.replies.push(reply.clone());
                    }
                }
                Some(_) => {}
                None => {
                    unattached_replies.insert(reply.seq, reply.clone());
                }
            }
        }
        for stored in turns.values_mut() {
            stored.replies.sort_by_key(|reply| reply.seq);
        }
        if max_turns != 0 {
            let keep: BTreeSet<u64> = unattached_replies
                .keys()
                .rev()
                .take(max_turns)
                .copied()
                .collect();
            unattached_replies.retain(|seq, _| keep.contains(seq));
        }
        GenerationMemory {
            session,
            turns,
            unattached_replies,
        }
    }

    /// 便捷投影：直接从编年史构建。
    pub fn from_chronicle(
        session: &DialogueSession,
        chronicle: &Chronicle,
        max_turns: usize,
    ) -> Self {
        let index = DialogueArtifactIndex::from_chronicle(session, chronicle);
        Self::from_index(&index, max_turns)
    }

    pub fn session_uri(&self) -> String {
        session_uri(&self.session)
    }

    pub fn turn_count(&self) -> usize {
        self.turns.len()
    }

    pub fn reply_count(&self) -> usize {
        self.turns
            .values()
            .map(|turn| turn.replies.len())
            .sum::<usize>()
            + self.unattached_replies.len()
    }

    pub fn claim_count(&self) -> usize {
        self.reply_iter()
            .map(|reply| reply.claims.len())
            .sum::<usize>()
    }

    pub fn evidence_count(&self) -> usize {
        self.reply_iter()
            .map(|reply| reply.evidence.len())
            .sum::<usize>()
    }

    pub fn is_empty(&self) -> bool {
        self.turns.is_empty() && self.unattached_replies.is_empty()
    }

    /// 全部回答（先轮次内、后无轮次；各段按 seq 升序）。
    pub fn reply_iter(&self) -> impl Iterator<Item = &ReplyArtifact> {
        self.turns
            .values()
            .flat_map(|turn| turn.replies.iter())
            .chain(self.unattached_replies.values())
    }

    /// 最近 `limit` 轮；`0` 表示全部。返回按 seq 升序。
    pub fn recent_turns(&self, limit: usize) -> Vec<&MemoryTurn> {
        if limit == 0 {
            return self.turns.values().collect();
        }
        let mut turns: Vec<&MemoryTurn> = self.turns.values().rev().take(limit).collect();
        turns.reverse();
        turns
    }

    /// 最近一条回答。
    pub fn latest_reply(&self) -> Option<&ReplyArtifact> {
        self.reply_iter().max_by_key(|reply| reply.seq)
    }

    /// 最近一条回答的命题计划。
    pub fn latest_plan(&self) -> Vec<String> {
        self.latest_reply()
            .map(|reply| reply.plan.clone())
            .unwrap_or_default()
    }

    /// 按规范化文本查找已说过的命题（忽略空白/标点/句末语气词）。
    pub fn find_claim(&self, text: &str) -> Option<&ClaimArtifact> {
        let needle = answer_key(text);
        if needle.is_empty() {
            return None;
        }
        self.reply_iter()
            .flat_map(|reply| reply.claims.iter())
            .find(|claim| answer_key(&claim.text) == needle)
    }

    /// 这句话是否已经作为某个回答的命题说过。
    pub fn already_said(&self, text: &str) -> bool {
        self.find_claim(text).is_some()
    }

    /// 整条回答文本是否已作为某条回答的完整正文说过（按规范化文本比对）。
    pub fn already_replied(&self, answer: &str) -> bool {
        let needle = answer_key(answer);
        if needle.is_empty() {
            return false;
        }
        self.reply_iter()
            .any(|reply| answer_key(&reply.answer) == needle)
    }

    /// 投影为 RDF 四元组（确定性排序、去重）。
    pub fn to_quads(&self) -> MemoryResult<Vec<Quad>> {
        if self.session.trim().is_empty() {
            return Err("生成记忆的会话标识不能为空".into());
        }
        let mut quads: Vec<Quad> = Vec::new();
        self.add_session_quads(&mut quads)?;
        for stored in self.turns.values() {
            let graph =
                GraphName::NamedNode(named_node(&turn_graph_iri(&self.session, stored.turn.seq))?);
            add_turn_quads(&mut quads, &graph, &self.session, &stored.turn)?;
            for reply in &stored.replies {
                add_reply_quads(&mut quads, &graph, &self.session, reply)?;
            }
        }
        for reply in self.unattached_replies.values() {
            let graph =
                GraphName::NamedNode(named_node(&reply_graph_iri(&self.session, reply.seq))?);
            add_reply_quads(&mut quads, &graph, &self.session, reply)?;
        }
        Ok(normalize_quads(quads))
    }

    /// 从四元组恢复记忆；会话不一致或工件结构损坏时显式报错。
    pub fn from_quads(quads: impl IntoIterator<Item = Quad>) -> Result<Self, String> {
        Self::from_quads_in(None, quads)
    }

    /// 从指定会话的四元组恢复记忆；混入其它会话时显式报错。
    pub fn from_quads_in_session(
        session: &str,
        quads: impl IntoIterator<Item = Quad>,
    ) -> Result<Self, String> {
        Self::from_quads_in(Some(session), quads)
    }

    fn from_quads_in(
        expected_session: Option<&str>,
        quads: impl IntoIterator<Item = Quad>,
    ) -> Result<Self, String> {
        let expected = expected_session
            .map(str::trim)
            .filter(|session| !session.is_empty());
        let mut session: Option<String> = None;
        let mut json_by_subject: BTreeMap<String, String> = BTreeMap::new();
        let mut turn_subjects: BTreeSet<String> = BTreeSet::new();
        let mut reply_subjects: BTreeSet<String> = BTreeSet::new();
        for quad in quads {
            let subject = match &quad.subject {
                Subject::NamedNode(node) => node.as_str().to_string(),
                _ => continue,
            };
            let predicate = quad.predicate.as_str();
            if predicate == P_SESSION {
                if let Term::Literal(literal) = &quad.object {
                    let value = literal.value().trim().to_string();
                    match &session {
                        Some(existing) if existing != &value => {
                            return Err(format!(
                                "生成记忆混入多个会话：「{}」与「{}」",
                                existing, value
                            ))
                        }
                        Some(_) => {}
                        None => session = Some(value),
                    }
                }
                continue;
            }
            if predicate == P_JSON {
                if let Term::Literal(literal) = &quad.object {
                    json_by_subject.insert(subject, literal.value().to_string());
                }
                continue;
            }
            if predicate == RDF_TYPE {
                if let Term::NamedNode(class) = &quad.object {
                    match class.as_str() {
                        TURN_CLASS => {
                            turn_subjects.insert(subject);
                        }
                        REPLY_CLASS => {
                            reply_subjects.insert(subject);
                        }
                        _ => {}
                    }
                }
            }
        }
        let session = match session {
            Some(session) => {
                if let Some(expected) = expected {
                    if session != expected {
                        return Err(format!(
                            "生成记忆属于会话「{}」，不是「{}」",
                            session, expected
                        ));
                    }
                }
                session
            }
            None => match expected {
                Some(expected) => expected.to_string(),
                None => String::new(),
            },
        };
        let mut turns: BTreeMap<u64, MemoryTurn> = BTreeMap::new();
        let mut replies: BTreeMap<u64, ReplyArtifact> = BTreeMap::new();
        for subject in &turn_subjects {
            let body = json_by_subject
                .get(subject)
                .ok_or_else(|| format!("轮次工件 {} 缺少规范 JSON，拒绝猜测其内容", subject))?;
            let turn: TurnArtifact = serde_json::from_str(body)
                .map_err(|error| format!("轮次工件 {} 解码失败：{}", subject, error))?;
            if turn.uri != *subject {
                return Err(format!(
                    "轮次工件 URI 不一致：四元组 {}，JSON {}",
                    subject, turn.uri
                ));
            }
            turns.insert(
                turn.seq,
                MemoryTurn {
                    turn,
                    replies: Vec::new(),
                },
            );
        }
        for subject in &reply_subjects {
            let body = json_by_subject
                .get(subject)
                .ok_or_else(|| format!("回答工件 {} 缺少规范 JSON，拒绝猜测其内容", subject))?;
            let reply: ReplyArtifact = serde_json::from_str(body)
                .map_err(|error| format!("回答工件 {} 解码失败：{}", subject, error))?;
            if reply.uri != *subject {
                return Err(format!(
                    "回答工件 URI 不一致：四元组 {}，JSON {}",
                    subject, reply.uri
                ));
            }
            replies.insert(reply.seq, reply);
        }
        let mut unattached_replies: BTreeMap<u64, ReplyArtifact> = BTreeMap::new();
        for (seq, reply) in replies {
            match reply.turn_seq.and_then(|turn_seq| turns.get_mut(&turn_seq)) {
                Some(stored) => stored.replies.push(reply),
                None => {
                    unattached_replies.insert(seq, reply);
                }
            }
        }
        for stored in turns.values_mut() {
            stored.replies.sort_by_key(|reply| reply.seq);
        }
        if turns.is_empty() && unattached_replies.is_empty() && session.is_empty() {
            return Err("生成记忆缺少会话标识".to_string());
        }
        Ok(GenerationMemory {
            session,
            turns,
            unattached_replies,
        })
    }

    /// 确定性的检索报告（CLI / 对话内查看）。
    pub fn render_report(&self) -> String {
        let mut output = String::new();
        output.push_str("生成记忆（RDF 命名图）\n");
        output.push_str(&format!("  会话：{}\n", self.session_uri()));
        output.push_str(&format!(
            "  图前缀：{}\n",
            session_graph_prefix(&self.session)
        ));
        output.push_str(&format!(
            "  计数：轮次 {}，回答 {}，命题 {}，依据 {}\n",
            self.turn_count(),
            self.reply_count(),
            self.claim_count(),
            self.evidence_count()
        ));
        let recent = self.recent_turns(DEFAULT_MEMORY_TURNS);
        if recent.is_empty() {
            output.push_str("  最近：（还没有投影任何轮次；可先运行 --sync）\n");
            return output;
        }
        output.push_str("  最近：\n");
        for stored in recent {
            output.push_str(&format!(
                "  [轮次 {}] {}\n",
                stored.turn.seq,
                compact(&stored.turn.user, 40)
            ));
            for reply in &stored.replies {
                output.push_str(&format!(
                    "    回答 {}：{}（命题 {}，依据 {}）\n",
                    reply.seq,
                    compact(&reply.answer, 48),
                    reply.claims.len(),
                    reply.evidence.len()
                ));
            }
        }
        output
    }

    fn add_session_quads(&self, quads: &mut Vec<Quad>) -> MemoryResult<()> {
        let graph = GraphName::NamedNode(named_node(&session_graph_iri(&self.session))?);
        let subject = named_node(&self.session_uri())?;
        push_class(quads, &subject, SESSION_CLASS, &graph)?;
        push_literal(quads, &subject, P_SESSION, &self.session, &graph)?;
        push_literal(
            quads,
            &subject,
            P_VERSION,
            GENERATION_MEMORY_VERSION.to_string(),
            &graph,
        )?;
        push_literal(
            quads,
            &subject,
            P_TURN_COUNT,
            self.turn_count().to_string(),
            &graph,
        )?;
        push_literal(
            quads,
            &subject,
            P_REPLY_COUNT,
            self.reply_count().to_string(),
            &graph,
        )?;
        let latest = self
            .turns
            .keys()
            .next_back()
            .copied()
            .into_iter()
            .chain(self.unattached_replies.keys().next_back().copied())
            .max()
            .unwrap_or(0);
        push_literal(quads, &subject, P_LATEST_SEQ, latest.to_string(), &graph)?;
        for stored in self.turns.values() {
            push_node(
                quads,
                &subject,
                P_HAS_TURN,
                &named_node(&stored.turn.uri)?,
                &graph,
            )?;
            for reply in &stored.replies {
                push_node(
                    quads,
                    &subject,
                    P_HAS_REPLY,
                    &named_node(&reply.uri)?,
                    &graph,
                )?;
            }
        }
        for reply in self.unattached_replies.values() {
            push_node(
                quads,
                &subject,
                P_HAS_REPLY,
                &named_node(&reply.uri)?,
                &graph,
            )?;
        }
        Ok(())
    }
}

fn named_node(iri: &str) -> MemoryResult<NamedNode> {
    Ok(NamedNode::new(iri)?)
}

fn push_literal(
    quads: &mut Vec<Quad>,
    subject: &NamedNode,
    predicate: &str,
    value: impl Into<String>,
    graph: &GraphName,
) -> MemoryResult<()> {
    quads.push(Quad::new(
        subject.clone(),
        named_node(predicate)?,
        Literal::new_simple_literal(value.into()),
        graph.clone(),
    ));
    Ok(())
}

fn push_node(
    quads: &mut Vec<Quad>,
    subject: &NamedNode,
    predicate: &str,
    object: &NamedNode,
    graph: &GraphName,
) -> MemoryResult<()> {
    quads.push(Quad::new(
        subject.clone(),
        named_node(predicate)?,
        object.clone(),
        graph.clone(),
    ));
    Ok(())
}

fn push_class(
    quads: &mut Vec<Quad>,
    subject: &NamedNode,
    class: &str,
    graph: &GraphName,
) -> MemoryResult<()> {
    push_node(quads, subject, RDF_TYPE, &named_node(class)?, graph)
}

fn add_turn_quads(
    quads: &mut Vec<Quad>,
    graph: &GraphName,
    session: &str,
    turn: &TurnArtifact,
) -> MemoryResult<()> {
    let subject = named_node(&turn.uri)?;
    push_class(quads, &subject, TURN_CLASS, graph)?;
    push_literal(quads, &subject, P_SESSION, session, graph)?;
    push_literal(quads, &subject, P_SEQ, turn.seq.to_string(), graph)?;
    push_literal(quads, &subject, P_USER, &turn.user, graph)?;
    push_literal(quads, &subject, P_MODE, &turn.mode, graph)?;
    push_literal(
        quads,
        &subject,
        P_KB_REVISION,
        turn.kb_revision.to_string(),
        graph,
    )?;
    if let Some(focus) = turn.focus.as_deref() {
        push_literal(quads, &subject, P_FOCUS, focus, graph)?;
    }
    if let Some(focus_uri) = turn.focus_uri.as_deref() {
        push_node(quads, &subject, P_FOCUS_URI, &named_node(focus_uri)?, graph)?;
    }
    push_literal(
        quads,
        &subject,
        P_JSON,
        serde_json::to_string(turn).map_err(|error| error.to_string())?,
        graph,
    )?;
    for mention in &turn.mentions {
        let mention_node = named_node(&mention.uri)?;
        push_class(quads, &mention_node, MENTION_CLASS, graph)?;
        push_node(quads, &subject, P_HAS_MENTION, &mention_node, graph)?;
        push_node(quads, &mention_node, P_TURN, &subject, graph)?;
        push_literal(quads, &mention_node, P_SESSION, session, graph)?;
        push_literal(
            quads,
            &mention_node,
            P_INDEX,
            mention.index.to_string(),
            graph,
        )?;
        push_literal(quads, &mention_node, P_SURFACE, &mention.surface, graph)?;
        push_literal(quads, &mention_node, P_SOURCE, &mention.source, graph)?;
        if let Some(entity) = mention.entity.as_deref() {
            push_literal(quads, &mention_node, P_ENTITY, entity, graph)?;
        }
        if let Some(entity_uri) = mention.entity_uri.as_deref() {
            push_node(
                quads,
                &mention_node,
                P_ENTITY_URI,
                &named_node(entity_uri)?,
                graph,
            )?;
        }
        push_literal(quads, &mention_node, P_JSON, mention_json(mention)?, graph)?;
    }
    Ok(())
}

fn mention_json(mention: &crate::dialogue_uri::MentionArtifact) -> Result<String, String> {
    serde_json::to_string(mention).map_err(|error| error.to_string())
}

fn add_reply_quads(
    quads: &mut Vec<Quad>,
    graph: &GraphName,
    session: &str,
    reply: &ReplyArtifact,
) -> MemoryResult<()> {
    let subject = named_node(&reply.uri)?;
    push_class(quads, &subject, REPLY_CLASS, graph)?;
    push_literal(quads, &subject, P_SESSION, session, graph)?;
    push_literal(quads, &subject, P_SEQ, reply.seq.to_string(), graph)?;
    push_literal(
        quads,
        &subject,
        P_KB_REVISION,
        reply.kb_revision.to_string(),
        graph,
    )?;
    push_literal(quads, &subject, P_ANSWER, &reply.answer, graph)?;
    if let Some(question) = reply.question.as_deref() {
        push_literal(quads, &subject, P_QUESTION, question, graph)?;
    }
    for step in &reply.plan {
        push_literal(quads, &subject, P_PLAN, step, graph)?;
    }
    if let Some(proof_id) = reply.proof_id.as_deref() {
        push_literal(quads, &subject, P_PROOF_ID, proof_id, graph)?;
    }
    if let Some(turn_seq) = reply.turn_seq {
        push_literal(quads, &subject, P_TURN_SEQ, turn_seq.to_string(), graph)?;
        push_node(
            quads,
            &subject,
            P_TURN,
            &named_node(&turn_uri(session, turn_seq))?,
            graph,
        )?;
    }
    push_literal(
        quads,
        &subject,
        P_JSON,
        serde_json::to_string(reply).map_err(|error| error.to_string())?,
        graph,
    )?;
    let mut statements: BTreeSet<String> = BTreeSet::new();
    for claim in &reply.claims {
        add_claim_quads(quads, graph, session, &subject, claim, &mut statements)?;
    }
    for (position, evidence) in reply.evidence.iter().enumerate() {
        let evidence_node = named_node(&evidence.uri)?;
        push_class(quads, &evidence_node, EVIDENCE_CLASS, graph)?;
        push_node(quads, &subject, P_HAS_EVIDENCE, &evidence_node, graph)?;
        push_node(quads, &evidence_node, P_REPLY, &subject, graph)?;
        push_literal(quads, &evidence_node, P_SESSION, session, graph)?;
        push_literal(
            quads,
            &evidence_node,
            P_INDEX,
            (position + 1).to_string(),
            graph,
        )?;
        push_literal(quads, &evidence_node, P_DIGEST, &evidence.digest, graph)?;
        push_literal(quads, &evidence_node, P_TEXT, &evidence.text, graph)?;
        push_literal(
            quads,
            &evidence_node,
            P_JSON,
            serde_json::to_string(evidence).map_err(|error| error.to_string())?,
            graph,
        )?;
    }
    for corrected in &reply.corrects {
        let corrected_node = named_node(corrected)?;
        push_node(quads, &subject, P_CORRECTS, &corrected_node, graph)?;
    }
    Ok(())
}

fn add_claim_quads(
    quads: &mut Vec<Quad>,
    graph: &GraphName,
    session: &str,
    reply: &NamedNode,
    claim: &ClaimArtifact,
    statements: &mut BTreeSet<String>,
) -> MemoryResult<()> {
    let subject = named_node(&claim.uri)?;
    push_class(quads, &subject, CLAIM_CLASS, graph)?;
    push_node(quads, reply, P_HAS_CLAIM, &subject, graph)?;
    push_node(quads, &subject, P_REPLY, reply, graph)?;
    push_literal(quads, &subject, P_SESSION, session, graph)?;
    push_literal(quads, &subject, P_INDEX, claim.index.to_string(), graph)?;
    push_literal(quads, &subject, P_DIGEST, &claim.digest, graph)?;
    push_literal(quads, &subject, P_TEXT, &claim.text, graph)?;
    push_node(
        quads,
        &subject,
        P_STATEMENT,
        &named_node(&claim.statement_uri)?,
        graph,
    )?;
    push_literal(
        quads,
        &subject,
        P_JSON,
        serde_json::to_string(claim).map_err(|error| error.to_string())?,
        graph,
    )?;
    if statements.insert(claim.statement_uri.clone()) {
        let statement = named_node(&claim.statement_uri)?;
        push_class(quads, &statement, STATEMENT_CLASS, graph)?;
        push_literal(quads, &statement, P_SESSION, session, graph)?;
        push_literal(quads, &statement, P_DIGEST, &claim.digest, graph)?;
        push_literal(quads, &statement, P_TEXT, &claim.text, graph)?;
    }
    Ok(())
}

fn quad_key(quad: &Quad) -> String {
    format!(
        "{}\u{1}{}\u{1}{}\u{1}{}",
        quad.subject, quad.predicate, quad.object, quad.graph_name
    )
}

fn normalize_quads(mut quads: Vec<Quad>) -> Vec<Quad> {
    quads.sort_by_key(quad_key);
    quads.dedup();
    quads
}

fn compact(text: &str, limit: usize) -> String {
    let joined = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut compact: String = joined.chars().take(limit).collect();
    if joined.chars().count() > limit {
        compact.push('…');
    }
    compact
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chronicle::{EntityRef, FsyncPolicy};
    use crate::perf_ledger::PerfLedger;
    use crate::OxigraphManager;
    use std::sync::Arc;
    use tempfile::TempDir;

    fn sample_index(dir: &TempDir) -> (DialogueArtifactIndex, DialogueSession, Chronicle) {
        let session_path = dir.path().join("dialogue.session.json");
        let session = DialogueSession::open_or_create(&session_path.to_string_lossy()).unwrap();
        let chronicle_path = dir.path().join("dialogue.chronicle.jsonl");
        let (mut chronicle, audit) =
            Chronicle::open(&chronicle_path.to_string_lossy(), FsyncPolicy::None).unwrap();
        assert!(audit.is_clean());
        chronicle
            .append_turn_with_refs(
                "姚明是谁",
                "query",
                None,
                0,
                &[EntityRef::linked("姚明", "姚明")],
            )
            .unwrap();
        chronicle
            .append_reply_with_context(
                Some("姚明是谁"),
                "姚明是篮球运动员。\n他的身高是 226 厘米。",
                &["direct_answer".to_string()],
                &["身高".to_string()],
                Some("proof-1"),
            )
            .unwrap();
        chronicle
            .append_turn_with_refs("他的身高呢", "query", Some("姚明"), 1, &[])
            .unwrap();
        chronicle
            .append_reply(
                "姚明的身高是 226 厘米。",
                &["direct_answer".to_string()],
                &["身高".to_string()],
            )
            .unwrap();
        let index = DialogueArtifactIndex::from_chronicle(&session, &chronicle);
        (index, session, chronicle)
    }

    #[test]
    fn projection_roundtrips_and_isolates_per_turn_graphs() {
        let dir = TempDir::new().unwrap();
        let (index, _session, _chronicle) = sample_index(&dir);
        let memory = GenerationMemory::from_index(&index, 0);
        assert_eq!(memory.turn_count(), 2);
        assert_eq!(memory.reply_count(), 2);
        assert_eq!(memory.claim_count(), 3);
        assert_eq!(memory.evidence_count(), 2);
        assert!(memory.already_said("姚明是篮球运动员"));
        assert!(!memory.already_said("姚明是足球运动员"));
        assert_eq!(memory.latest_plan(), vec!["direct_answer".to_string()]);

        let quads = memory.to_quads().unwrap();
        assert!(!quads.is_empty());
        let prefix = session_graph_prefix(&memory.session);
        for quad in &quads {
            match &quad.graph_name {
                GraphName::NamedNode(graph) => {
                    assert!(graph.as_str().starts_with(&prefix), "{}", graph.as_str())
                }
                other => panic!("生成记忆不得写入非命名图：{:?}", other),
            }
        }
        let turn_graphs: BTreeSet<String> = quads
            .iter()
            .filter_map(|quad| match &quad.graph_name {
                GraphName::NamedNode(graph) if graph.as_str().contains("/turn/") => {
                    Some(graph.as_str().to_string())
                }
                _ => None,
            })
            .collect();
        assert_eq!(turn_graphs.len(), 2, "每轮应有独立命名图");
        assert_eq!(
            turn_graphs,
            BTreeSet::from([
                turn_graph_iri(&memory.session, 1),
                turn_graph_iri(&memory.session, 3),
            ])
        );

        let loaded = GenerationMemory::from_quads(quads).unwrap();
        assert_eq!(loaded, memory, "RDF 投影必须无损往返");
    }

    #[test]
    fn bounded_projection_keeps_only_recent_turns() {
        let dir = TempDir::new().unwrap();
        let (index, _session, _chronicle) = sample_index(&dir);
        let memory = GenerationMemory::from_index(&index, 1);
        assert_eq!(memory.turn_count(), 1);
        assert_eq!(memory.turns.keys().copied().collect::<Vec<_>>(), vec![3]);
        assert_eq!(memory.reply_count(), 1);
        assert_eq!(memory.recent_turns(0).len(), 1);
    }

    #[test]
    fn config_defaults_off_and_parses_explicit_values() {
        let defaults: GenerationMemoryConfig = toml::from_str("").unwrap();
        assert!(!defaults.enabled);
        assert_eq!(defaults.max_turns, DEFAULT_MEMORY_TURNS);
        let explicit: GenerationMemoryConfig =
            toml::from_str("enabled = true\nmax_turns = 3").unwrap();
        assert!(explicit.enabled);
        assert_eq!(explicit.max_turns, 3);
    }

    #[test]
    fn corrupted_projection_is_rejected_not_guessed() {
        let dir = TempDir::new().unwrap();
        let (index, _session, _chronicle) = sample_index(&dir);
        let memory = GenerationMemory::from_index(&index, 0);
        let mut quads = memory.to_quads().unwrap();
        quads.retain(|quad| quad.predicate.as_str() != P_JSON);
        let error = GenerationMemory::from_quads(quads).unwrap_err();
        assert!(error.contains("缺少规范 JSON"), "{}", error);
    }

    #[tokio::test]
    async fn manager_sync_recall_forget_keeps_default_graph_clean() {
        let dir = TempDir::new().unwrap();
        let (index, _session, _chronicle) = sample_index(&dir);
        let memory = GenerationMemory::from_index(&index, 0);
        let store_dir = dir.path().join("store");
        let manager = OxigraphManager::with_perf_ledger(
            &store_dir.to_string_lossy(),
            Arc::new(PerfLedger::new()),
        )
        .unwrap();
        let revision = manager.revision();

        let written = manager.sync_generation_memory(&memory).await.unwrap();
        assert!(written > 0);
        assert_eq!(manager.revision(), revision, "账本写入不得推进知识版本");

        let loaded = manager
            .load_generation_memory(&memory.session)
            .await
            .unwrap();
        assert_eq!(loaded, memory);

        let default_graph = manager
            .query_sparql("SELECT ?s WHERE { ?s ?p ?o }")
            .await
            .unwrap();
        assert!(default_graph.data.is_empty(), "生成记忆不得污染默认图");

        let answers = manager
            .query_sparql(&format!(
                "SELECT ?a WHERE {{ GRAPH ?g {{ ?r <{}> ?a }} }}",
                P_ANSWER
            ))
            .await
            .unwrap();
        assert_eq!(answers.data.len(), 2, "同一查询应能跨轮读取回答");

        let removed = manager
            .forget_generation_memory(&memory.session)
            .await
            .unwrap();
        assert_eq!(removed, 3, "会话图 + 两个轮次图");
        let forgotten = manager
            .load_generation_memory(&memory.session)
            .await
            .unwrap();
        assert!(forgotten.is_empty());
        assert_eq!(manager.revision(), revision);
    }

    #[test]
    fn corrects_quads_emitted_and_roundtrip() {
        let dir = TempDir::new().unwrap();
        let (index, _session, _chronicle) = sample_index(&dir);
        let mut memory = GenerationMemory::from_index(&index, 0);
        let corrected = index.reply(2).unwrap().claims[0].uri.clone();
        let latest_seq = memory.latest_reply().unwrap().seq;
        for stored in memory.turns.values_mut() {
            for reply in stored.replies.iter_mut() {
                if reply.seq == latest_seq {
                    reply.corrects = vec![corrected.clone()];
                }
            }
        }
        let quads = memory.to_quads().unwrap();
        assert!(
            quads.iter().any(|quad| {
                matches!(&quad.graph_name, GraphName::NamedNode(_))
                    && quad.predicate.as_str() == P_CORRECTS
                    && matches!(
                        &quad.object,
                        Term::NamedNode(node) if node.as_str() == corrected
                    )
            }),
            "应发出 P_CORRECTS 四元组，使更正关系可被 SPARQL 查询"
        );

        // P_JSON 载荷随 corrects 一并保留，圆满往返不丢更正关系。
        let loaded = GenerationMemory::from_quads(quads).unwrap();
        assert_eq!(loaded, memory);
        assert_eq!(loaded.latest_reply().unwrap().corrects, vec![corrected]);
    }

    #[test]
    fn already_replied_matches_full_answer_text() {
        let dir = TempDir::new().unwrap();
        let (index, _session, _chronicle) = sample_index(&dir);
        let memory = GenerationMemory::from_index(&index, 0);
        // 样本里第二条回答的完整正文。
        let reply = index.reply(2).unwrap();
        assert!(memory.already_replied(&reply.answer));
        // 不同标点/空白视为相同。
        assert!(memory.already_replied(&format!("{} ", reply.answer.trim())));
        // 改动正文就不命中。
        assert!(!memory.already_replied(&format!("{}\n多余一句。", reply.answer)));
        assert!(!memory.already_replied("完全没说过的回答内容。"));
    }
}

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
//! - **可审计**：每次投影在会话图里留一份「投影收据」——窗口多大、编年史共几轮、
//!   保留了哪些 seq、丢弃了哪些 seq——上下文选择不再是黑箱截断；
//! - **可追溯**：命题指向内容寻址陈述，回答指向轮次，生成史不再是黑箱文本。

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

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

pub const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
pub const SESSION_CLASS: &str = "http://example.org/kg/ledger/dialogue/Session";
pub const TURN_CLASS: &str = "http://example.org/kg/ledger/dialogue/Turn";
pub const REPLY_CLASS: &str = "http://example.org/kg/ledger/dialogue/Reply";
pub const CLAIM_CLASS: &str = "http://example.org/kg/ledger/dialogue/Claim";
pub const STATEMENT_CLASS: &str = "http://example.org/kg/ledger/dialogue/Statement";
pub const MENTION_CLASS: &str = "http://example.org/kg/ledger/dialogue/Mention";
pub const EVIDENCE_CLASS: &str = "http://example.org/kg/ledger/dialogue/Evidence";

pub const P_SESSION: &str = "http://example.org/kg/ledger/dialogue/session";
pub const P_VERSION: &str = "http://example.org/kg/ledger/dialogue/version";
pub const P_SEQ: &str = "http://example.org/kg/ledger/dialogue/seq";
pub const P_INDEX: &str = "http://example.org/kg/ledger/dialogue/index";
pub const P_USER: &str = "http://example.org/kg/ledger/dialogue/user";
pub const P_MODE: &str = "http://example.org/kg/ledger/dialogue/mode";
pub const P_FOCUS: &str = "http://example.org/kg/ledger/dialogue/focus";
pub const P_FOCUS_URI: &str = "http://example.org/kg/ledger/dialogue/focusUri";
pub const P_KB_REVISION: &str = "http://example.org/kg/ledger/dialogue/kbRevision";
pub const P_QUESTION: &str = "http://example.org/kg/ledger/dialogue/question";
pub const P_ANSWER: &str = "http://example.org/kg/ledger/dialogue/answer";
pub const P_PLAN: &str = "http://example.org/kg/ledger/dialogue/plan";
pub const P_PROOF_ID: &str = "http://example.org/kg/ledger/dialogue/proofId";
pub const P_TEXT: &str = "http://example.org/kg/ledger/dialogue/text";
pub const P_DIGEST: &str = "http://example.org/kg/ledger/dialogue/digest";
pub const P_STATEMENT: &str = "http://example.org/kg/ledger/dialogue/statement";
/// 命题文本的规范化键（`answer_key` 口径），供 SPARQL 与内存路径同口径匹配。
pub const P_TEXT_KEY: &str = "http://example.org/kg/ledger/dialogue/textKey";
pub const P_SURFACE: &str = "http://example.org/kg/ledger/dialogue/surface";
pub const P_ENTITY: &str = "http://example.org/kg/ledger/dialogue/entity";
pub const P_ENTITY_URI: &str = "http://example.org/kg/ledger/dialogue/entityUri";
pub const P_SOURCE: &str = "http://example.org/kg/ledger/dialogue/source";
pub const P_TURN: &str = "http://example.org/kg/ledger/dialogue/turn";
pub const P_TURN_SEQ: &str = "http://example.org/kg/ledger/dialogue/turnSeq";
pub const P_REPLY: &str = "http://example.org/kg/ledger/dialogue/reply";
pub const P_HAS_TURN: &str = "http://example.org/kg/ledger/dialogue/hasTurn";
pub const P_HAS_REPLY: &str = "http://example.org/kg/ledger/dialogue/hasReply";
pub const P_HAS_CLAIM: &str = "http://example.org/kg/ledger/dialogue/hasClaim";
pub const P_HAS_EVIDENCE: &str = "http://example.org/kg/ledger/dialogue/hasEvidence";
pub const P_HAS_MENTION: &str = "http://example.org/kg/ledger/dialogue/hasMention";
pub const P_TURN_COUNT: &str = "http://example.org/kg/ledger/dialogue/turnCount";
pub const P_REPLY_COUNT: &str = "http://example.org/kg/ledger/dialogue/replyCount";
pub const P_LATEST_SEQ: &str = "http://example.org/kg/ledger/dialogue/latestSeq";
pub const P_JSON: &str = "http://example.org/kg/ledger/dialogue/json";
/// 更正关系：回答 → 其更正的先前命题 URI，按轮隔离、内容寻址。
pub const P_CORRECTS: &str = "http://example.org/kg/ledger/dialogue/corrects";
/// 投影收据：会话图里的上下文选择记录（窗口/来源/保留/丢弃）。
pub const P_PROJECTION_JSON: &str = "http://example.org/kg/ledger/dialogue/projectionJson";
pub const P_WINDOW: &str = "http://example.org/kg/ledger/dialogue/projectionWindow";
pub const P_SOURCE_TURNS: &str = "http://example.org/kg/ledger/dialogue/projectionSourceTurns";
pub const P_DROPPED_TURNS: &str = "http://example.org/kg/ledger/dialogue/projectionDroppedTurns";
pub const P_DROPPED_SEQS: &str = "http://example.org/kg/ledger/dialogue/projectionDroppedSeqs";
pub const P_SOURCE_REPLIES: &str = "http://example.org/kg/ledger/dialogue/projectionSourceReplies";
pub const P_DROPPED_REPLIES: &str =
    "http://example.org/kg/ledger/dialogue/projectionDroppedReplies";

type MemoryResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// 会话内承载工件图的 IRI 列表：轮次图 + 无轮次回答图（升序）。
///
/// 编年史 `seq` 是跨条目（turn/reply/trace）的全局计数，轮次图 IRI 因此是
/// 稀疏的（如 1、3、5）：**绝不能用 `1..=N` 枚举图 IRI**，那会漏掉真实轮次、
/// 又掺入不存在的图。图集合必须从记忆投影本身取，见
/// [`GenerationMemory::graph_iris`]。
///
/// 组装 `GRAPH ?g { 主语的会话过滤 + 模式 } [FILTER(?g IN (图列表))]` 的
/// WHERE 主体：会话过滤直接落在模式主语（`?claim`）上，避免引入自由变量
/// 与工件模式做笛卡尔积；跨会话图不会被扫到。图列表为空（空记忆）时省略
/// 过滤器，会话过滤天然无解，同样不泄漏。
fn memory_graph_body(memory: &GenerationMemory, subject: &str, pattern: &str) -> String {
    let session = escape_sparql_literal(&memory.session);
    let iris = memory.graph_iris();
    let filter = if iris.is_empty() {
        String::new()
    } else {
        let list = iris
            .iter()
            .map(|iri| format!("<{}>", iri))
            .collect::<Vec<_>>()
            .join(", ");
        format!(" FILTER(?g IN ({}))", list)
    };
    format!(
        "GRAPH ?g {{ {} <{}> \"{}\" . {} }}{}",
        subject, P_SESSION, session, pattern, filter
    )
}

/// SPARQL SELECT 查询：找出记忆里哪条回答的哪个命题说过指定文本，
/// 返回 `(?seq, ?text)`；`?seq` 是回答关联的轮次 seq（无轮次回答时不绑定）。
///
/// 匹配走命题的规范化键（`answer_key` 口径，折叠空白/标点/大小写），与内存
/// 侧 `GenerationMemory::find_claim` 同口径。返回空串表示待查文本规范化后
/// 为空，没有可执行的查询；否则查询可直接传给 `oxigraph_manager.query_sparql()`
/// 执行。
pub fn find_claim_by_text_sparql(memory: &GenerationMemory, text: &str) -> String {
    let normalized = answer_key(text);
    if normalized.is_empty() {
        return String::new();
    }
    let escaped = escape_sparql_literal(&normalized);
    format!(
        "SELECT ?seq ?text WHERE {{ {} }}",
        memory_graph_body(
            memory,
            "?claim",
            &format!(
                "?claim a <{}> . ?claim <{}> ?text . ?claim <{}> \"{}\" . \
                 OPTIONAL {{ ?claim <{}> ?reply . ?reply <{}> ?seq . }}",
                CLAIM_CLASS, P_TEXT, P_TEXT_KEY, escaped, P_REPLY, P_TURN_SEQ
            )
        )
    )
}

/// SPARQL ASK 查询：特定文本是否已作为命题出现在记忆的任意回答里。
///
/// 与 [`find_claim_by_text_sparql`] 同一口径（命题类陈述、规范化键匹配、
/// 只在本记忆的命名图内搜索）；返回空串表示待查文本为空，无查询可执行。
pub fn already_said_sparql_query(memory: &GenerationMemory, text: &str) -> String {
    let normalized = answer_key(text);
    if normalized.is_empty() {
        return String::new();
    }
    let escaped = escape_sparql_literal(&normalized);
    format!(
        "ASK {{ {} }}",
        memory_graph_body(
            memory,
            "?claim",
            &format!(
                "?claim a <{}> . ?claim <{}> \"{}\" .",
                CLAIM_CLASS, P_TEXT_KEY, escaped
            )
        )
    )
}

/// SPARQL SELECT 查询：列出记忆中全部已说命题的 `(?digest, ?text)`。
///
/// `?digest` 是命题的内容摘要（FNV-1a 十六进制），`?text` 是命题完整文本；
/// 只含命题类陈述，不含依据文本。查询可直接传给
/// `oxigraph_manager.query_sparql()` 执行。
pub fn list_all_claims_sparql(memory: &GenerationMemory) -> String {
    format!(
        "SELECT ?digest ?text WHERE {{ {} }}",
        memory_graph_body(
            memory,
            "?claim",
            &format!(
                "?claim a <{}> . ?claim <{}> ?digest . ?claim <{}> ?text .",
                CLAIM_CLASS, P_DIGEST, P_TEXT
            )
        )
    )
}

/// SPARQL ASK 查询：记忆里是否说过同时包含指定实体词面与属性词面的命题。
///
/// 词面用 `CONTAINS` 做大小写不敏感的包含匹配，对应生成侧「实体+属性兜底」
/// 的保守口径；实体或属性为空时返回空串，不产出查询。
pub fn entity_attribute_said_sparql_query(
    memory: &GenerationMemory,
    entity: &str,
    attribute: &str,
) -> String {
    let entity = entity.trim();
    let attribute = attribute.trim();
    if entity.is_empty() || attribute.is_empty() {
        return String::new();
    }
    let escaped_entity = escape_sparql_literal(entity);
    let escaped_attribute = escape_sparql_literal(attribute);
    format!(
        "ASK {{ {} }}",
        memory_graph_body(
            memory,
            "?claim",
            &format!(
                "?claim a <{}> . ?claim <{}> ?text . \
                 FILTER(CONTAINS(LCASE(?text), LCASE(\"{}\")) && CONTAINS(LCASE(?text), LCASE(\"{}\")))",
                CLAIM_CLASS, P_TEXT, escaped_entity, escaped_attribute
            )
        )
    )
}

/// SPARQL 数值排序前缀：轮次/回答/命题序号以简单字面量落图，多位数词面
/// 按字典序会错排（"10" < "2"），排序前先投成 `xsd:integer`。
pub const XSD_PREFIX: &str = "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>";

/// SPARQL SELECT 查询：最近一条回答（回答 seq 最大）的全部命题，
/// 返回 `(?replySeq, ?claimIndex, ?text)`，按回答 seq 降序、命题序号升序。
///
/// 「第 N 个命题」的 SPARQL 召回用它取最近一条回答的命题切片，与内存侧
/// `render_plan_step` 同口径：结果第一组行就是最近一条回答，组内按序号升序。
pub fn latest_reply_claims_sparql(memory: &GenerationMemory) -> String {
    format!(
        "{} SELECT ?replySeq ?claimIndex ?text WHERE {{ {} }} \
         ORDER BY DESC(xsd:integer(?replySeq)) ASC(xsd:integer(?claimIndex))",
        XSD_PREFIX,
        memory_graph_body(
            memory,
            "?reply",
            &format!(
                "?reply a <{}> . ?reply <{}> ?replySeq . \
                 ?reply <{}> ?claim . ?claim <{}> ?claimIndex . ?claim <{}> ?text .",
                REPLY_CLASS, P_SEQ, P_HAS_CLAIM, P_INDEX, P_TEXT
            )
        )
    )
}

/// SPARQL SELECT 查询：定位「说过指定文本」的回答与命题（带位置信息），
/// 返回 `(?replySeq, ?claimIndex, ?turnSeq, ?text)`，按回答 seq 升序、命题
/// 序号升序——与内存侧 `find_claim`「最早一次提及优先」的顺序同口径。
///
/// 无轮次回答的 `?turnSeq` 不绑定（缺席即无轮次关联）；文本为空时返回空串。
pub fn locate_claim_by_text_sparql(memory: &GenerationMemory, text: &str) -> String {
    let normalized = answer_key(text);
    if normalized.is_empty() {
        return String::new();
    }
    let escaped = escape_sparql_literal(&normalized);
    format!(
        "{} SELECT ?replySeq ?claimIndex ?turnSeq ?text WHERE {{ {} }} \
         ORDER BY ASC(xsd:integer(?replySeq)) ASC(xsd:integer(?claimIndex))",
        XSD_PREFIX,
        memory_graph_body(
            memory,
            "?claim",
            &format!(
                "?claim a <{}> . ?claim <{}> ?text . ?claim <{}> \"{}\" . \
                 ?claim <{}> ?claimIndex . ?claim <{}> ?reply . ?reply <{}> ?replySeq . \
                 OPTIONAL {{ ?reply <{}> ?turnSeq . }}",
                CLAIM_CLASS, P_TEXT, P_TEXT_KEY, escaped, P_INDEX, P_REPLY, P_SEQ, P_TURN_SEQ
            )
        )
    )
}

/// SPARQL SELECT 查询：本记忆中的命题总数（单行 `?total`，十进制词面）。
///
/// 「没有说过」的召回用它如实报检索量，与内存侧 `claim_count()` 同口径。
pub fn count_claims_sparql(memory: &GenerationMemory) -> String {
    format!(
        "SELECT (COUNT(?claim) AS ?total) WHERE {{ {} }}",
        memory_graph_body(memory, "?claim", &format!("?claim a <{}> .", CLAIM_CLASS))
    )
}

/// SPARQL SELECT 查询：本记忆里的更正关系（更正回答 → 被更正命题文本），
/// 返回 `(?replySeq, ?answer, ?correctedText)`，按回答 seq 升序、被更正文本
/// 升序——与内存侧「按回答顺序列出」一致。
///
/// 更正回答与被更正命题可能落在不同轮次图：用两个 GRAPH 子句分别绑定并
/// 各自受会话图列表约束。被更正命题所在图被有界投影丢弃时，`?correctedText`
/// 不绑定、该行内连接消失，与内存侧「找不到命题文本就不列出」同口径。
pub fn corrections_sparql_query(memory: &GenerationMemory) -> String {
    let session = escape_sparql_literal(&memory.session);
    let iris = memory.graph_iris();
    let filter_for = |variable: &str| {
        if iris.is_empty() {
            String::new()
        } else {
            let list = iris
                .iter()
                .map(|iri| format!("<{}>", iri))
                .collect::<Vec<_>>()
                .join(", ");
            format!(" FILTER(?{} IN ({}))", variable, list)
        }
    };
    format!(
        "{} SELECT ?replySeq ?answer ?correctedText WHERE {{ \
           GRAPH ?g1 {{ ?reply <{}> \"{}\" . ?reply a <{}> . \
             ?reply <{}> ?corrected . ?reply <{}> ?replySeq . \
             OPTIONAL {{ ?reply <{}> ?answer . }} }}{} \
           GRAPH ?g2 {{ ?corrected <{}> ?correctedText . }}{} \
         }} \
         ORDER BY ASC(xsd:integer(?replySeq)) ASC(?correctedText)",
        XSD_PREFIX,
        P_SESSION,
        session,
        REPLY_CLASS,
        P_CORRECTS,
        P_SEQ,
        P_ANSWER,
        filter_for("g1"),
        P_TEXT,
        filter_for("g2")
    )
}

/// SPARQL SELECT 查询：最近一次投影的收据标量（会话图内），
/// 返回 `(?window, ?sourceTurns, ?droppedTurns, ?droppedSeqs, ?sourceReplies,
/// ?droppedReplies)`。
///
/// 「保留了哪些轮、丢弃了哪些轮、为什么」的机器可读口径：外部工具不必解析
/// JSON 收据即可用 SPARQL 审计上下文选择。图与主语都由会话标识确定性推出，
/// 查询天然有界；无收据（空记忆或旧版本投影）时查询无解。
pub fn projection_receipt_sparql(memory: &GenerationMemory) -> String {
    let graph = session_graph_iri(&memory.session);
    let subject = session_uri(&memory.session);
    format!(
        "SELECT ?window ?sourceTurns ?droppedTurns ?droppedSeqs ?sourceReplies ?droppedReplies WHERE {{ \
           GRAPH <{}> {{ \
             <{}> <{}> ?window . \
             <{}> <{}> ?sourceTurns . \
             <{}> <{}> ?droppedTurns . \
             <{}> <{}> ?droppedSeqs . \
             <{}> <{}> ?sourceReplies . \
             <{}> <{}> ?droppedReplies . \
           }} \
         }}",
        graph,
        subject,
        P_WINDOW,
        subject,
        P_SOURCE_TURNS,
        subject,
        P_DROPPED_TURNS,
        subject,
        P_DROPPED_SEQS,
        subject,
        P_SOURCE_REPLIES,
        subject,
        P_DROPPED_REPLIES
    )
}

/// SPARQL SELECT 查询：最近一次记录的实体焦点，返回 `(?focus, ?turnSeq)`。
///
/// 与内存侧 `latest_focus` 同口径：轮次按 seq 降序取第一条带焦点的；
/// seq 投成 `xsd:integer` 比较，避免多位数词面按字典序错排。
pub fn latest_focus_sparql_query(memory: &GenerationMemory) -> String {
    format!(
        "{} SELECT ?focus ?turnSeq WHERE {{ {} }} \
         ORDER BY DESC(xsd:integer(?turnSeq)) LIMIT 1",
        XSD_PREFIX,
        memory_graph_body(
            memory,
            "?turn",
            &format!(
                "?turn a <{}> . ?turn <{}> ?turnSeq . ?turn <{}> ?focus .",
                TURN_CLASS, P_SEQ, P_FOCUS
            )
        )
    )
}

/// 把待插入 SPARQL 简单字面量的文本转义：反斜杠与双引号做转义，控制字符
/// 改写为 SPARQL 转义序列，避免用户文本破坏查询语法或注入额外子句。
/// 仅用于生成记忆内部拼装的受控查询，不处理任意 IRI。
fn escape_sparql_literal(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            other if other.is_control() => {
                let _ = write!(out, "\\u{:04x}", other as u32);
            }
            other => out.push(other),
        }
    }
    out
}

/// RDF 生成记忆配置（serde 默认关闭；显式开启后每个轮次边界做一次有界投影）。
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

/// 投影收据：一次有界投影的上下文选择记录，随会话图落 RDF、可查询可审计。
///
/// 「保留了哪些轮、丢弃了哪些轮、为什么」都写进会话命名图：`window` 是投影
/// 时的轮次上限（0 = 未设上限，全量投影）；`source_turns`/`source_replies`
/// 是工件索引里的轮次/回答总数；`kept_turns`/`dropped_turns` 是保留与丢弃的
/// 轮次 seq（升序）；`dropped_replies` 是因窗口被丢弃的回答数。丢弃原因恒为
/// 「超出最近 N 轮窗口」，不再单独建模。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectionReceipt {
    /// 投影窗口（最近 N 轮）；0 表示未设上限。
    pub window: usize,
    /// 工件索引中的轮次总数。
    pub source_turns: usize,
    /// 保留进记忆的轮次 seq（升序）。
    pub kept_turns: Vec<u64>,
    /// 因窗口被丢弃的轮次 seq（升序）。
    pub dropped_turns: Vec<u64>,
    /// 工件索引中的回答总数。
    pub source_replies: usize,
    /// 因窗口被丢弃的回答数（含超出保留数的无轮次回答）。
    pub dropped_replies: usize,
}

impl ProjectionReceipt {
    /// 丢弃轮次的紧凑词面（逗号连接；无丢弃为空串）。
    pub fn dropped_seqs_joined(&self) -> String {
        self.dropped_turns
            .iter()
            .map(|seq| seq.to_string())
            .collect::<Vec<_>>()
            .join(",")
    }

    /// 是否为全量投影（未设窗口且无丢弃）。
    pub fn is_unbounded(&self) -> bool {
        self.window == 0
    }

    /// 确定性单行摘要（CLI 报告与召回措辞共用同一口径）。
    pub fn render_line(&self) -> String {
        if self.is_unbounded() {
            return format!(
                "投影窗口：未设上限（max_turns=0）；编年史共 {} 轮全部在记忆中，回答 {} 条。",
                self.source_turns, self.source_replies
            );
        }
        if self.dropped_turns.is_empty() && self.dropped_replies == 0 {
            return format!(
                "投影窗口：最近 {} 轮；编年史共 {} 轮全部保留，回答 {} 条。",
                self.window, self.source_turns, self.source_replies
            );
        }
        let dropped_seqs = self.dropped_seqs_joined();
        let seqs = if dropped_seqs.is_empty() {
            String::new()
        } else {
            format!("（seq {}）", dropped_seqs)
        };
        format!(
            "投影窗口：最近 {} 轮；编年史共 {} 轮，保留 {} 轮，丢弃 {} 轮{}，丢弃回答 {} 条。",
            self.window,
            self.source_turns,
            self.kept_turns.len(),
            self.dropped_turns.len(),
            seqs,
            self.dropped_replies
        )
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
    /// 最近一次投影的收据；空记忆或旧投影（无收据版本）为 `None`。
    #[serde(default)]
    pub projection: Option<ProjectionReceipt>,
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

/// 编年史里「忘记生成记忆」水位条目的种类。
pub const FORGET_WATERMARK_KIND: &str = "forget_memory";

/// 读取编年史的「忘记生成记忆」水位：最后一次水位条目的 seq。
///
/// 水位之后的条目才参与生成记忆投影（`from_chronicle_since`）；没有水位
/// 条目时返回 0（全量投影）。跨重启持久——水位随编年史落盘。
pub fn chronicle_watermark(chronicle: &Chronicle) -> u64 {
    chronicle
        .entries()
        .iter()
        .filter(|entry| entry.kind == FORGET_WATERMARK_KIND)
        .map(|entry| entry.seq)
        .next_back()
        .unwrap_or(0)
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
        let s = session.into().trim().to_string();
        let session = if s.is_empty() {
            "unknown".to_string()
        } else {
            s
        };
        GenerationMemory {
            session,
            ..Self::default()
        }
    }

    /// 清空记忆内容，保留会话标识。用于会话重启或测试场景。
    ///
    /// 此操作仅重置内存状态，不影响 RDF 存储中的已有记录。
    /// 若要在 RDF 存储中删除会话，请使用 `OxigraphManager::forget_generation_memory`。
    pub fn clear(&mut self) {
        self.turns.clear();
        self.unattached_replies.clear();
        self.projection = None;
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
        let kept_replies =
            turns.values().map(|turn| turn.replies.len()).sum::<usize>() + unattached_replies.len();
        let kept_turns: BTreeSet<u64> = turns.keys().copied().collect();
        let receipt = ProjectionReceipt {
            window: max_turns,
            source_turns: index.turn_count(),
            kept_turns: kept_turns.iter().copied().collect(),
            dropped_turns: index
                .turns()
                .map(|turn| turn.seq)
                .filter(|seq| !kept_turns.contains(seq))
                .collect(),
            source_replies: index.reply_count(),
            dropped_replies: index.reply_count().saturating_sub(kept_replies),
        };
        GenerationMemory {
            session,
            turns,
            unattached_replies,
            projection: Some(receipt),
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

    /// 便捷投影：从编年史的水位之后构建（「忘记生成记忆」后的增量口径）。
    ///
    /// `min_seq` 之前的条目不参与投影，已忘记的历史不会被重新写回图；
    /// 投影收据如实记录水位后的轮次总数。
    pub fn from_chronicle_since(
        session: &DialogueSession,
        chronicle: &Chronicle,
        min_seq: u64,
        max_turns: usize,
    ) -> Self {
        let index = DialogueArtifactIndex::from_chronicle_since(session, chronicle, min_seq);
        Self::from_index(&index, max_turns)
    }

    /// 向记忆中添加一个轮次（幂等：已存在则不重复）。
    ///
    /// 返回被写入的轮次 seq，若 seq 已存在则返回 None。
    pub fn add_turn(
        &mut self,
        seq: u64,
        user: String,
        mode: String,
        focus: Option<String>,
    ) -> Option<u64> {
        if self.turns.contains_key(&seq) {
            return None;
        }
        self.turns.insert(
            seq,
            MemoryTurn {
                turn: TurnArtifact {
                    seq,
                    uri: turn_graph_iri(&self.session, seq),
                    user,
                    mode,
                    focus,
                    focus_uri: None,
                    kb_revision: 0,
                    mentions: Vec::new(),
                },
                replies: Vec::new(),
            },
        );
        Some(seq)
    }

    /// 向指定轮次添加一个回答（幂等：重复调用不重复写入）。
    ///
    /// 如果轮次不存在，会自动创建；如果该轮次已有该 seq 的回答则不重复写入。
    ///
    /// 注意：如果返回 `None`，调用方仍保有 `reply` 的所有权。
    pub fn add_reply(&mut self, turn_seq: u64, reply: ReplyArtifact) -> Option<&ReplyArtifact> {
        let reply_seq = reply.seq;
        let turn = self.turns.get_mut(&turn_seq);
        match turn {
            Some(stored) => {
                // 检查是否已存在相同 seq 的回答（按 uri 唯一性）
                if stored.replies.iter().any(|r| r.seq == reply_seq) {
                    return None;
                }
                stored.replies.push(reply);
                stored.replies.sort_by_key(|r| r.seq);
                Some(stored.replies.last().unwrap())
            }
            None => {
                // 轮次不存在，作为无轮次回答保存
                self.unattached_replies.insert(reply_seq, reply);
                Some(self.unattached_replies.get(&reply_seq).unwrap())
            }
        }
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

    /// 记忆中实际保留的轮次 seq（升序）。编年史 seq 是跨条目全局计数，
    /// 轮次图 IRI 因此稀疏；SPARQL 的图集合必须从这里取，不得用 `1..=N` 猜。
    pub fn kept_turn_seqs(&self) -> Vec<u64> {
        self.turns.keys().copied().collect()
    }

    /// 记忆承载工件图的 IRI 列表（轮次图 + 无轮次回答图，升序、去重）。
    ///
    /// 这是「有界查询」的图集合来源：所有面向记忆的 SPARQL 构造器都用它
    /// 约束 `?g`，保证查询只落在本会话本轮投影的图上。
    pub fn graph_iris(&self) -> Vec<String> {
        let mut iris: Vec<String> = self
            .turns
            .keys()
            .map(|seq| turn_graph_iri(&self.session, *seq))
            .chain(
                self.unattached_replies
                    .keys()
                    .map(|seq| reply_graph_iri(&self.session, *seq)),
            )
            .collect();
        iris.sort();
        iris.dedup();
        iris
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

    /// 定位被一次用户更正撤回的先前命题 URI（按回答、命题顺序升序，去重）。
    ///
    /// 语义与追踪图的更正一致：「上一轮的说法作废」。匹配保守，宁漏勿错：
    ///
    /// 1. **片段锚定**（优先）：旧回答片段是上一条被更正回答首行的截断，
    ///    取最近一条规范化回答包含该片段（或互为前缀）的回答，作废其全部
    ///    内容命题（排除以冒号结尾的小节标题）；
    /// 2. **实体+属性兜底**：片段锚定失败时，回退到同时包含实体与属性名的
    ///    历史命题（可能覆盖更早轮对同一事实的陈述）；
    /// 3. 实体名为空（没有锚点）时不匹配任何命题，宁可空着也不猜。
    pub fn corrected_claim_uris(
        &self,
        entity: &str,
        attribute: &str,
        previous_answer: &str,
    ) -> Vec<String> {
        let entity = entity.trim();
        if entity.is_empty() {
            return Vec::new();
        }
        let attribute = attribute.trim();
        let excerpt = previous_answer.trim();
        let excerpt = excerpt.strip_suffix('…').unwrap_or(excerpt).trim();
        let prev_key = (!excerpt.is_empty())
            .then(|| answer_key(excerpt))
            .filter(|key| !key.is_empty());
        // 阶段一：旧回答片段锚定最近一条对应回答，整条作废（与追踪图同语义）。
        if let Some(prev_key) = &prev_key {
            let replies: Vec<&crate::dialogue_uri::ReplyArtifact> = self.reply_iter().collect();
            for reply in replies.iter().rev() {
                let reply_key = answer_key(&reply.answer);
                if reply_key.is_empty() {
                    continue;
                }
                if reply_key.contains(prev_key.as_str()) || prev_key.starts_with(&reply_key) {
                    return reply
                        .claims
                        .iter()
                        .filter(|claim| is_proposition(&claim.text))
                        .map(|claim| claim.uri.clone())
                        .collect();
                }
            }
        }
        // 阶段二：实体+属性兜底（两者都非空才启用，避免属性名单独误伤）。
        if !attribute.is_empty() {
            let mut uris: Vec<String> = Vec::new();
            for claim in self.reply_iter().flat_map(|reply| reply.claims.iter()) {
                if claim.text.contains(entity)
                    && claim.text.contains(attribute)
                    && is_proposition(&claim.text)
                    && !uris.contains(&claim.uri)
                {
                    uris.push(claim.uri.clone());
                }
            }
            return uris;
        }
        Vec::new()
    }

    /// 丢弃末尾「已提问但还没有回答」的轮次（写前日志下的在途回合），
    /// 供召回在回答前剔除当前问句，避免回顾把自己复述成历史。
    ///
    /// 只删最后一轮，且必须同时满足「无回答」与「用户词面一致」（按
    /// `answer_key` 规范化，折叠空白/标点/大小写）；不满足时不删，崩溃
    /// 轮次与历史轮次原样保留。若投影收据记录了该轮，同步从保留名单去掉，
    /// 使「投影保留最近 K 轮」与实际渲染的轮数一致。
    pub fn drop_trailing_unanswered_turn(&mut self, user: &str) -> bool {
        let needle = crate::chronicle::answer_key(user);
        if needle.is_empty() {
            return false;
        }
        let Some((&seq, stored)) = self.turns.iter().next_back() else {
            return false;
        };
        if !stored.replies.is_empty() || crate::chronicle::answer_key(&stored.turn.user) != needle {
            return false;
        }
        self.turns.remove(&seq);
        if let Some(receipt) = self.projection.as_mut() {
            receipt.kept_turns.retain(|kept| *kept != seq);
        }
        true
    }

    /// 投影为 RDF 四元组（确定性排序、去重）。
    pub fn to_quads(&self) -> MemoryResult<Vec<Quad>> {
        if self.session.trim().is_empty() {
            return Err("生成记忆的会话标识不能为空".into());
        }
        let mut quads: Vec<Quad> = Vec::new();
        self.add_session_quads(&mut quads)?;
        if let Some(receipt) = &self.projection {
            add_projection_quads(&mut quads, &self.session, receipt)?;
        }
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
        let mut projection_json: Option<String> = None;
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
            if predicate == P_PROJECTION_JSON {
                if let Term::Literal(literal) = &quad.object {
                    match &projection_json {
                        Some(existing) if existing != literal.value() => {
                            return Err("生成记忆含多份投影收据，拒绝猜测以哪份为准".to_string());
                        }
                        Some(_) => {}
                        None => projection_json = Some(literal.value().to_string()),
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
        let projection = match projection_json {
            Some(body) => Some(
                serde_json::from_str::<ProjectionReceipt>(&body)
                    .map_err(|error| format!("投影收据解码失败：{}", error))?,
            ),
            None => None,
        };
        Ok(GenerationMemory {
            session,
            turns,
            unattached_replies,
            projection,
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
        match &self.projection {
            Some(receipt) => output.push_str(&format!("  {}\n", receipt.render_line())),
            None => output.push_str("  投影窗口：（无收据；空记忆或旧版本投影）\n"),
        }
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

/// 投影收据落图：会话图内的上下文选择记录。
///
/// JSON 载荷保证无损读回，标量四元组（窗口/来源/丢弃计数/丢弃 seq 词面）
/// 让「这次投影丢了什么」可以用 SPARQL 直接问出来。
fn add_projection_quads(
    quads: &mut Vec<Quad>,
    session: &str,
    receipt: &ProjectionReceipt,
) -> MemoryResult<()> {
    let graph = GraphName::NamedNode(named_node(&session_graph_iri(session))?);
    let subject = named_node(&session_uri(session))?;
    push_literal(
        quads,
        &subject,
        P_PROJECTION_JSON,
        serde_json::to_string(receipt).map_err(|error| error.to_string())?,
        &graph,
    )?;
    push_literal(
        quads,
        &subject,
        P_WINDOW,
        receipt.window.to_string(),
        &graph,
    )?;
    push_literal(
        quads,
        &subject,
        P_SOURCE_TURNS,
        receipt.source_turns.to_string(),
        &graph,
    )?;
    push_literal(
        quads,
        &subject,
        P_DROPPED_TURNS,
        receipt.dropped_turns.len().to_string(),
        &graph,
    )?;
    push_literal(
        quads,
        &subject,
        P_DROPPED_SEQS,
        receipt.dropped_seqs_joined(),
        &graph,
    )?;
    push_literal(
        quads,
        &subject,
        P_SOURCE_REPLIES,
        receipt.source_replies.to_string(),
        &graph,
    )?;
    push_literal(
        quads,
        &subject,
        P_DROPPED_REPLIES,
        receipt.dropped_replies.to_string(),
        &graph,
    )?;
    Ok(())
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
    let text_key = crate::chronicle::answer_key(&claim.text);
    if !text_key.is_empty() {
        push_literal(quads, &subject, P_TEXT_KEY, text_key, graph)?;
    }
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

/// 命题 vs 小节标题：以冒号结尾的回答行（「根据知识图谱：」）不是可撤回的命题。
fn is_proposition(text: &str) -> bool {
    !text.trim_end().ends_with(['：', ':'])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chronicle::{EntityRef, FsyncPolicy};
    use crate::perf_ledger::PerfLedger;
    use crate::OxigraphManager;
    use serde_json::Value as JsonValue;
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
                &[],
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
                &[],
            )
            .unwrap();
        let index = DialogueArtifactIndex::from_chronicle(&session, &chronicle);
        (index, session, chronicle)
    }

    #[test]
    fn forget_watermark_excludes_history_from_projection() {
        let dir = TempDir::new().unwrap();
        let (index, session, mut chronicle) = sample_index(&dir);
        let _ = index;

        // 无水位条目：全量投影。
        assert_eq!(chronicle_watermark(&chronicle), 0);
        let full = GenerationMemory::from_chronicle(&session, &chronicle, 0);
        assert_eq!(full.turn_count(), 2);
        assert!(full.already_said("姚明是篮球运动员"));

        // 忘记生成记忆：追加水位条目（seq 记入编年史）。
        let watermark_seq = chronicle
            .append(
                FORGET_WATERMARK_KIND,
                serde_json::json!({"forgotten_graphs": 3}),
            )
            .unwrap();
        assert_eq!(chronicle_watermark(&chronicle), watermark_seq);

        // 水位之后的新一轮问答。
        chronicle
            .append_turn_with_refs("叶莉是谁", "query", None, 2, &[])
            .unwrap();
        chronicle
            .append_reply("叶莉是篮球运动员。", &[], &[], &[])
            .unwrap();

        // 水位之后投影：只含新一轮，已忘记的历史不再算「说过」。
        let since = GenerationMemory::from_chronicle_since(
            &session,
            &chronicle,
            chronicle_watermark(&chronicle),
            0,
        );
        assert_eq!(since.turn_count(), 1);
        assert_eq!(since.reply_count(), 1);
        assert!(since.already_said("叶莉是篮球运动员"));
        assert!(!since.already_said("姚明是篮球运动员"));
        assert!(!since.already_replied("姚明是篮球运动员。\n他的身高是 226 厘米。"));
        let receipt = since.projection.as_ref().unwrap();
        assert_eq!(receipt.source_turns, 1, "收据只统计水位后的轮次");

        // 无关种类不算水位；多条水位取最后一条。
        chronicle.append("trace", serde_json::json!({})).unwrap();
        let later = chronicle
            .append(FORGET_WATERMARK_KIND, serde_json::json!({}))
            .unwrap();
        assert_eq!(chronicle_watermark(&chronicle), later);
    }

    #[test]
    fn drop_trailing_unanswered_turn_removes_only_matching_inflight_turn() {
        let dir = TempDir::new().unwrap();
        let (index, _session, _chronicle) = sample_index(&dir);
        let mut memory = GenerationMemory::from_index(&index, 0);
        let base_turns = memory.turn_count();

        // 在途回合：已写前日志、还没有回答；词面与当前输入一致才删除。
        memory.add_turn(
            9,
            "我们刚才聊了什么？".to_string(),
            "query".to_string(),
            None,
        );
        assert!(memory.drop_trailing_unanswered_turn("我们刚才聊了什么？"));
        assert_eq!(memory.turn_count(), base_turns);

        // 词面不一致（即便同样没有回答）不删除：不误伤崩溃轮次。
        memory.add_turn(9, "还没答的问题".to_string(), "query".to_string(), None);
        assert!(!memory.drop_trailing_unanswered_turn("我们刚才聊了什么？"));
        assert_eq!(memory.turn_count(), base_turns + 1);

        // 已有回答的末轮不删除。
        memory.clear();
        let mut fresh = GenerationMemory::from_index(&index, 0);
        assert!(!fresh.drop_trailing_unanswered_turn("姚明是谁"));
        assert_eq!(fresh.turn_count(), base_turns);

        // 有界投影收据同步收缩：保留名单不再含被剔除的 seq。
        let (index, _session, _chronicle) = sample_index(&dir);
        let mut bounded = GenerationMemory::from_index(&index, 2);
        let kept_before = bounded.projection.as_ref().unwrap().kept_turns.clone();
        bounded.add_turn(
            kept_before.last().copied().unwrap_or(0).saturating_add(1),
            "我们刚才聊了什么".to_string(),
            "query".to_string(),
            None,
        );
        let inflight_seq = *bounded.turns.keys().next_back().unwrap();
        assert!(bounded.drop_trailing_unanswered_turn("我们刚才聊了什么"));
        let receipt = bounded.projection.as_ref().unwrap();
        assert!(
            !receipt.kept_turns.contains(&inflight_seq),
            "{:?}",
            receipt.kept_turns
        );

        // 空记忆与空词面直接返回 false。
        let mut empty = GenerationMemory::empty("s-1");
        assert!(!empty.drop_trailing_unanswered_turn("随便什么"));
        assert!(!empty.drop_trailing_unanswered_turn("   "));
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
        // 收据如实记录窗口与丢弃：编年史 2 轮，保留 seq 3，丢弃 seq 1。
        let receipt = memory.projection.as_ref().unwrap();
        assert_eq!(receipt.window, 1);
        assert_eq!(receipt.source_turns, 2);
        assert_eq!(receipt.kept_turns, vec![3]);
        assert_eq!(receipt.dropped_turns, vec![1]);
        assert_eq!(receipt.source_replies, 2);
        assert_eq!(receipt.dropped_replies, 1);
        assert_eq!(receipt.dropped_seqs_joined(), "1");
        assert!(!receipt.is_unbounded());
    }

    #[test]
    fn unbounded_projection_receipt_reports_no_drops() {
        let dir = TempDir::new().unwrap();
        let (index, _session, _chronicle) = sample_index(&dir);
        let memory = GenerationMemory::from_index(&index, 0);
        let receipt = memory.projection.as_ref().unwrap();
        assert_eq!(receipt.window, 0);
        assert_eq!(receipt.source_turns, 2);
        assert_eq!(receipt.kept_turns, vec![1, 3]);
        assert!(receipt.dropped_turns.is_empty());
        assert_eq!(receipt.dropped_replies, 0);
        assert!(receipt.is_unbounded());
        let line = receipt.render_line();
        assert!(line.contains("未设上限"), "{}", line);
        assert!(!line.contains("丢弃"), "{}", line);

        let bounded_line = GenerationMemory::from_index(&index, 1)
            .projection
            .unwrap()
            .render_line();
        assert!(bounded_line.contains("最近 1 轮"), "{}", bounded_line);
        assert!(
            bounded_line.contains("丢弃 1 轮（seq 1）"),
            "{}",
            bounded_line
        );
    }

    #[test]
    fn projection_receipt_roundtrips_through_quads_and_store() {
        let dir = TempDir::new().unwrap();
        let (index, _session, _chronicle) = sample_index(&dir);
        let memory = GenerationMemory::from_index(&index, 1);
        let loaded = GenerationMemory::from_quads(memory.to_quads().unwrap()).unwrap();
        assert_eq!(loaded.projection, memory.projection, "收据必须无损往返");
        assert_eq!(loaded, memory);
    }

    #[test]
    fn corrupted_projection_receipt_is_rejected_not_guessed() {
        let dir = TempDir::new().unwrap();
        let (index, _session, _chronicle) = sample_index(&dir);
        let memory = GenerationMemory::from_index(&index, 0);
        let mut quads = memory.to_quads().unwrap();
        for quad in quads.iter_mut() {
            if quad.predicate.as_str() == P_PROJECTION_JSON {
                quad.object = Literal::new_simple_literal("{\"window\": not-json}").into();
            }
        }
        let error = GenerationMemory::from_quads(quads).unwrap_err();
        assert!(error.contains("投影收据解码失败"), "{}", error);
    }

    #[test]
    fn config_defaults_off_and_parses_explicit_values() {
        assert!(
            !GenerationMemoryConfig::default().enabled,
            "Default 与 serde 默认一致：生成记忆默认关闭"
        );
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
    fn latest_reply_projects_into_memory_reply_fragment() {
        let dir = TempDir::new().unwrap();
        let (index, _session, _chronicle) = sample_index(&dir);
        let memory = GenerationMemory::from_index(&index, 0);
        // 与 interactive.rs 注入的供给闭包同一构造路径：最近一条回答 → 切片。
        let reply = memory.latest_reply().unwrap();
        let fragment = crate::symbolic_reasoner::MemoryReplyFragment {
            reply_seq: reply.seq,
            claims: reply
                .claims
                .iter()
                .map(|claim| (claim.index, claim.text.clone()))
                .collect(),
        };
        assert_eq!(fragment.reply_seq, 4);
        assert_eq!(
            fragment.claims,
            vec![(1u32, "姚明的身高是 226 厘米。".to_string())]
        );
        // 空记忆不产切片：复述回退落回既有指引，不伪装成答案。
        let empty = GenerationMemory::empty("s");
        assert!(empty.latest_reply().is_none());
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

    #[test]
    fn corrected_claim_uris_anchor_on_previous_answer_excerpt() {
        let dir = TempDir::new().unwrap();
        let (index, _session, _chronicle) = sample_index(&dir);
        let memory = GenerationMemory::from_index(&index, 0);
        // 追踪图更正携带的旧回答片段 = 上一条被更正回答首行（可能截断）。
        let previous = "姚明的身高是 226 厘米。";
        let uris = memory.corrected_claim_uris("姚明", "身高", previous);
        assert_eq!(
            uris,
            vec![index.reply(4).unwrap().claims[0].uri.clone()],
            "片段应精确锚定被更正回答的命题"
        );
        // 片段被截断（超过 40 字符时带省略号）仍按前缀关系命中。
        let truncated = format!("{}…", &previous[..previous.len() - '。'.len_utf8()]);
        assert_eq!(
            memory.corrected_claim_uris("姚明", "身高", &truncated),
            uris
        );
        // 实体+属性兜底也要求实体匹配：没有叶莉的命题就不得误伤姚明的。
        assert!(memory.corrected_claim_uris("叶莉", "身高", "").is_empty());
    }

    #[test]
    fn corrected_claim_uris_fall_back_to_entity_and_attribute() {
        let dir = TempDir::new().unwrap();
        let (index, _session, _chronicle) = sample_index(&dir);
        let memory = GenerationMemory::from_index(&index, 0);
        // 片段缺失（追踪图没拿到旧回答）时退化为实体+属性匹配。
        let uris = memory.corrected_claim_uris("姚明", "身高", "");
        assert_eq!(uris, vec![index.reply(4).unwrap().claims[0].uri.clone()]);
        // 片段对不上任何回答（例如中间插了别的轮次），只要实体+属性仍在历史
        // 命题里，就按兜底规则命中。
        let uris = memory.corrected_claim_uris("姚明", "身高", "完全无关的旧回答");
        assert_eq!(uris, vec![index.reply(4).unwrap().claims[0].uri.clone()]);
    }

    #[test]
    fn corrected_claim_uris_skip_headers_and_require_anchors() {
        let dir = TempDir::new().unwrap();
        let (index, _session, _chronicle) = sample_index(&dir);
        let memory = GenerationMemory::from_index(&index, 0);
        // 无实体锚点：宁可空着也不猜。
        assert!(memory
            .corrected_claim_uris("", "身高", "姚明的身高是 226 厘米")
            .is_empty());
        let empty = GenerationMemory::empty("s");
        assert!(empty.corrected_claim_uris("姚明", "身高", "").is_empty());

        // 标题行（以冒号结尾）不是可撤回命题：片段锚定整条作废时也要排除。
        let session_path = dir.path().join("header.session.json");
        let session = DialogueSession::open_or_create(&session_path.to_string_lossy()).unwrap();
        let chronicle_path = dir.path().join("header.chronicle.jsonl");
        let (mut chronicle, audit) =
            Chronicle::open(&chronicle_path.to_string_lossy(), FsyncPolicy::None).unwrap();
        assert!(audit.is_clean());
        chronicle
            .append_turn_with_refs("叶莉的身高呢", "query", Some("叶莉"), 0, &[])
            .unwrap();
        chronicle
            .append_reply(
                "根据知识图谱：\n叶莉的身高是 190 厘米。",
                &["direct_answer".to_string()],
                &[],
                &[],
            )
            .unwrap();
        let header_index = DialogueArtifactIndex::from_chronicle(&session, &chronicle);
        let header_memory = GenerationMemory::from_index(&header_index, 0);
        // 追踪图的旧回答片段取首条实质行（跳过标题），命题里只剩实质行。
        let uris = header_memory.corrected_claim_uris("叶莉", "身高", "叶莉的身高是 190 厘米");
        let reply = header_index.reply(2).unwrap();
        assert_eq!(
            uris,
            reply
                .claims
                .iter()
                .filter(|claim| claim.text.contains("身高"))
                .map(|claim| claim.uri.clone())
                .collect::<Vec<_>>(),
            "标题行不得进入更正负载"
        );
        assert_eq!(uris.len(), 1);
    }

    #[test]
    fn escape_sparql_literal_neutralizes_quotes_and_backslashes() {
        assert_eq!(
            escape_sparql_literal("he said \"hi\""),
            "he said \\\"hi\\\""
        );
        assert_eq!(
            escape_sparql_literal("path\\file \"x\""),
            "path\\\\file \\\"x\\\""
        );
        assert_eq!(escape_sparql_literal("a\nb\tc"), "a\\nb\\tc");
        assert_eq!(escape_sparql_literal("plain"), "plain");
        assert_eq!(escape_sparql_literal(""), "");
    }

    #[test]
    fn already_said_sparql_query_escapes_and_bounds_to_memory_graphs() {
        let dir = TempDir::new().unwrap();
        let (index, _session, _chronicle) = sample_index(&dir);
        let memory = GenerationMemory::from_index(&index, 0);
        let unbounded = already_said_sparql_query(&memory, "姚明\"说'话");
        assert!(!unbounded.is_empty());
        // 引号被转义（出现反斜杠），查询结构保持 ASK + 命名图遍历。
        assert!(unbounded.contains('\\'), "{}", unbounded);
        assert!(unbounded.contains(P_TEXT_KEY), "{}", unbounded);
        assert!(unbounded.contains("GRAPH ?g"), "{}", unbounded);
        // 会话过滤内嵌于图模式，跨会话图不会被扫到。
        assert!(
            unbounded.contains(&escape_sparql_literal(&memory.session)),
            "{}",
            unbounded
        );

        // 有界：?g 被约束到记忆实际承载的图 IRI（稀疏 seq 1、3，不得枚举 1..=N）。
        assert!(unbounded.contains("FILTER(?g IN"), "{}", unbounded);
        assert!(
            unbounded.contains(&turn_graph_iri(&memory.session, 1)),
            "{}",
            unbounded
        );
        assert!(
            unbounded.contains(&turn_graph_iri(&memory.session, 3)),
            "{}",
            unbounded
        );
        assert!(
            !unbounded.contains(&turn_graph_iri(&memory.session, 2)),
            "{}",
            unbounded
        );

        // 空文本不产出生成查询。
        let blank = GenerationMemory::empty("s-1");
        assert!(already_said_sparql_query(&blank, "   ").is_empty());
        assert!(already_said_sparql_query(&blank, "").is_empty());
    }

    #[test]
    fn entity_attribute_said_sparql_query_escapes_and_rejects_empty() {
        let dir = TempDir::new().unwrap();
        let (index, _session, _chronicle) = sample_index(&dir);
        let memory = GenerationMemory::from_index(&index, 0);
        let q = entity_attribute_said_sparql_query(&memory, "姚\"明", "身高");
        assert!(q.contains('\\'), "{}", q);
        assert!(q.contains("身高"), "{}", q);
        assert!(q.contains("CONTAINS(LCASE(?text)"), "{}", q);
        assert!(q.contains("ASK {"), "{}", q);

        // 任一为空不产出生成查询。
        let blank = GenerationMemory::empty("s-1");
        assert!(entity_attribute_said_sparql_query(&blank, "", "身高").is_empty());
        assert!(entity_attribute_said_sparql_query(&blank, "姚明", "",).is_empty());
        assert!(entity_attribute_said_sparql_query(&blank, "   ", "身高").is_empty());
    }

    #[test]
    fn list_all_claims_sparql_bounds_to_memory_graphs() {
        let dir = TempDir::new().unwrap();
        let (index, _session, _chronicle) = sample_index(&dir);
        let memory = GenerationMemory::from_index(&index, 0);
        let bounded = list_all_claims_sparql(&memory);
        assert!(bounded.contains("SELECT ?digest ?text"), "{}", bounded);
        assert!(bounded.contains("GRAPH ?g"), "{}", bounded);
        assert!(bounded.contains("FILTER(?g IN"), "{}", bounded);
        assert!(
            bounded.contains(&turn_graph_iri(&memory.session, 1)),
            "{}",
            bounded
        );
        assert!(
            bounded.contains(&turn_graph_iri(&memory.session, 3)),
            "{}",
            bounded
        );
        // 命题而非轮次：类别与摘要谓词都落在 claim 上。
        assert!(bounded.contains(CLAIM_CLASS), "{}", bounded);
        assert!(bounded.contains(P_DIGEST), "{}", bounded);
        assert!(!bounded.contains(TURN_CLASS), "{}", bounded);

        // 空记忆仍产出可执行查询（会话过滤天然无解）。
        let blank = list_all_claims_sparql(&GenerationMemory::empty("s-1"));
        assert!(blank.contains("GRAPH ?g"), "{}", blank);
        assert!(!blank.contains("FILTER(?g IN"), "{}", blank);
    }

    #[test]
    fn find_claim_by_text_sparql_escapes_and_bounds_to_memory_graphs() {
        let dir = TempDir::new().unwrap();
        let (index, _session, _chronicle) = sample_index(&dir);
        let memory = GenerationMemory::from_index(&index, 0);
        let bounded = find_claim_by_text_sparql(&memory, "姚明\"说'话");
        assert!(!bounded.is_empty());
        assert!(bounded.contains('\\'), "{}", bounded);
        assert!(bounded.contains("SELECT ?seq ?text"), "{}", bounded);
        assert!(bounded.contains("GRAPH ?g"), "{}", bounded);
        assert!(bounded.contains("FILTER(?g IN"), "{}", bounded);
        assert!(
            bounded.contains(&turn_graph_iri(&memory.session, 3)),
            "{}",
            bounded
        );
        assert!(bounded.contains(CLAIM_CLASS), "{}", bounded);
        // 命题→回答→轮次 seq 的链路，使 ?seq 可定位轮次。
        assert!(bounded.contains(P_REPLY), "{}", bounded);
        assert!(bounded.contains(P_TURN_SEQ), "{}", bounded);

        // 空文本不产生查询。
        let blank = GenerationMemory::empty("s-1");
        assert!(find_claim_by_text_sparql(&blank, "   ").is_empty());
        assert!(find_claim_by_text_sparql(&blank, "").is_empty());
    }

    #[test]
    fn locate_claim_by_text_sparql_escapes_and_reports_position() {
        let dir = TempDir::new().unwrap();
        let (index, _session, _chronicle) = sample_index(&dir);
        let memory = GenerationMemory::from_index(&index, 0);
        let bounded = locate_claim_by_text_sparql(&memory, "姚明\"说'话");
        assert!(!bounded.is_empty());
        assert!(bounded.contains('\\'), "{}", bounded);
        assert!(bounded.contains("SELECT ?replySeq ?claimIndex ?turnSeq ?text"));
        assert!(bounded.contains("GRAPH ?g"), "{}", bounded);
        assert!(bounded.contains("FILTER(?g IN"), "{}", bounded);
        assert!(bounded.contains(P_INDEX), "{}", bounded);
        assert!(bounded.contains(P_REPLY), "{}", bounded);
        // 回答 seq 参与排序：xsd:integer 投影保证多位数词面不按字典序错排。
        assert!(
            bounded.contains("ORDER BY ASC(xsd:integer(?replySeq))"),
            "{}",
            bounded
        );

        // 空文本不产生查询。
        let blank = GenerationMemory::empty("s-1");
        assert!(locate_claim_by_text_sparql(&blank, "   ").is_empty());
        assert!(locate_claim_by_text_sparql(&blank, "").is_empty());
    }

    #[test]
    fn latest_reply_claims_sparql_anchors_to_reply_class_and_orders_desc() {
        let dir = TempDir::new().unwrap();
        let (index, _session, _chronicle) = sample_index(&dir);
        let memory = GenerationMemory::from_index(&index, 0);
        let query = latest_reply_claims_sparql(&memory);
        assert!(
            query.contains("SELECT ?replySeq ?claimIndex ?text"),
            "{}",
            query
        );
        assert!(query.contains(REPLY_CLASS), "{}", query);
        assert!(query.contains(P_HAS_CLAIM), "{}", query);
        assert!(
            query.contains("ORDER BY DESC(xsd:integer(?replySeq))"),
            "{}",
            query
        );
        assert!(query.contains("GRAPH ?g"), "{}", query);
        assert!(query.contains("FILTER(?g IN"), "{}", query);

        // 空记忆仍产出可执行查询（会话过滤天然无解）。
        let blank = latest_reply_claims_sparql(&GenerationMemory::empty("s-1"));
        assert!(blank.contains("GRAPH ?g"), "{}", blank);
        assert!(!blank.contains("FILTER(?g IN"), "{}", blank);
    }

    #[test]
    fn count_claims_sparql_aggregates_claim_class_only() {
        let dir = TempDir::new().unwrap();
        let (index, _session, _chronicle) = sample_index(&dir);
        let memory = GenerationMemory::from_index(&index, 0);
        let query = count_claims_sparql(&memory);
        assert!(query.contains("(COUNT(?claim) AS ?total)"), "{}", query);
        assert!(query.contains(CLAIM_CLASS), "{}", query);
        assert!(!query.contains(REPLY_CLASS), "{}", query);
        assert!(query.contains("FILTER(?g IN"), "{}", query);
    }

    #[test]
    fn corrections_sparql_query_joins_two_graphs_with_session_bounds() {
        let dir = TempDir::new().unwrap();
        let (index, _session, _chronicle) = sample_index(&dir);
        let memory = GenerationMemory::from_index(&index, 0);
        let query = corrections_sparql_query(&memory);
        assert!(
            query.contains("SELECT ?replySeq ?answer ?correctedText"),
            "{}",
            query
        );
        // 双图绑定：更正回答与被更正命题可以分属不同轮次图。
        assert!(query.contains("GRAPH ?g1"), "{}", query);
        assert!(query.contains("GRAPH ?g2"), "{}", query);
        assert!(query.contains(P_CORRECTS), "{}", query);
        // 两个图变量各自受会话图列表约束。
        assert!(query.contains("FILTER(?g1 IN"), "{}", query);
        assert!(query.contains("FILTER(?g2 IN"), "{}", query);
        assert!(
            query.contains(&turn_graph_iri(&memory.session, 1)),
            "{}",
            query
        );

        // 空记忆：省略图列表过滤，会话过滤天然无解。
        let blank = corrections_sparql_query(&GenerationMemory::empty("s-1"));
        assert!(!blank.contains("FILTER(?g1 IN"), "{}", blank);
        assert!(!blank.contains("FILTER(?g2 IN"), "{}", blank);
    }

    #[test]
    fn latest_focus_sparql_query_orders_turns_desc_with_limit() {
        let dir = TempDir::new().unwrap();
        let (index, _session, _chronicle) = sample_index(&dir);
        let memory = GenerationMemory::from_index(&index, 0);
        let query = latest_focus_sparql_query(&memory);
        assert!(query.contains("SELECT ?focus ?turnSeq"), "{}", query);
        assert!(query.contains(TURN_CLASS), "{}", query);
        assert!(query.contains(P_FOCUS), "{}", query);
        assert!(
            query.contains("ORDER BY DESC(xsd:integer(?turnSeq)) LIMIT 1"),
            "{}",
            query
        );
    }

    #[tokio::test]
    async fn projection_receipt_sparql_reports_window_and_drops() {
        let dir = TempDir::new().unwrap();
        let (index, _session, _chronicle) = sample_index(&dir);
        let memory = GenerationMemory::from_index(&index, 1);
        let store_dir = dir.path().join("store");
        let manager = OxigraphManager::with_perf_ledger(
            &store_dir.to_string_lossy(),
            Arc::new(PerfLedger::new()),
        )
        .unwrap();
        manager.sync_generation_memory(&memory).await.unwrap();

        let rows = manager
            .query_sparql(&projection_receipt_sparql(&memory))
            .await
            .unwrap();
        assert_eq!(rows.data.len(), 1, "{:?}", rows.data);
        assert_eq!(
            rows.data[0].get("window").and_then(JsonValue::as_str),
            Some("1")
        );
        assert_eq!(
            rows.data[0].get("sourceTurns").and_then(JsonValue::as_str),
            Some("2")
        );
        assert_eq!(
            rows.data[0].get("droppedTurns").and_then(JsonValue::as_str),
            Some("1")
        );
        assert_eq!(
            rows.data[0].get("droppedSeqs").and_then(JsonValue::as_str),
            Some("1")
        );

        // 全量投影：无丢弃、窗口未设上限。
        let full = GenerationMemory::from_index(&index, 0);
        manager.sync_generation_memory(&full).await.unwrap();
        let full_rows = manager
            .query_sparql(&projection_receipt_sparql(&full))
            .await
            .unwrap();
        assert_eq!(
            full_rows.data[0].get("window").and_then(JsonValue::as_str),
            Some("0")
        );
        assert_eq!(
            full_rows.data[0]
                .get("droppedTurns")
                .and_then(JsonValue::as_str),
            Some("0")
        );
    }

    #[tokio::test]
    async fn positional_sparql_builders_execute_against_projected_memory() {
        let dir = TempDir::new().unwrap();
        let (index, _session, _chronicle) = sample_index(&dir);
        let memory = GenerationMemory::from_index(&index, 0);
        let store_dir = dir.path().join("store");
        let manager = OxigraphManager::with_perf_ledger(
            &store_dir.to_string_lossy(),
            Arc::new(PerfLedger::new()),
        )
        .unwrap();
        manager.sync_generation_memory(&memory).await.unwrap();

        // 最近一条回答的命题：行按（回答 seq 降序、命题序号升序）排序，
        // 第一组（replySeq 相同的连续行）就是最近回答（seq 4）。
        let rows = manager
            .query_sparql(&latest_reply_claims_sparql(&memory))
            .await
            .unwrap();
        assert_eq!(rows.data.len(), 3, "{:?}", rows.data);
        assert_eq!(
            rows.data[0].get("replySeq").and_then(JsonValue::as_str),
            Some("4")
        );
        assert_eq!(
            rows.data[0].get("claimIndex").and_then(JsonValue::as_str),
            Some("1")
        );
        assert_eq!(
            rows.data[0].get("text").and_then(JsonValue::as_str),
            Some("姚明的身高是 226 厘米。")
        );
        // 后续组是更早回答（seq 2），组内命题序号升序。
        assert_eq!(
            rows.data[1].get("replySeq").and_then(JsonValue::as_str),
            Some("2")
        );
        assert_eq!(
            rows.data[2].get("claimIndex").and_then(JsonValue::as_str),
            Some("2")
        );

        // 定位已说命题：给出回答 seq、命题序号、所属轮次与原文。
        let located = manager
            .query_sparql(&locate_claim_by_text_sparql(
                &memory,
                " 姚明是篮球运动员 。",
            ))
            .await
            .unwrap();
        assert_eq!(located.data.len(), 1, "{:?}", located.data);
        assert_eq!(
            located.data[0].get("replySeq").and_then(JsonValue::as_str),
            Some("2")
        );
        assert_eq!(
            located.data[0]
                .get("claimIndex")
                .and_then(JsonValue::as_str),
            Some("1")
        );
        assert_eq!(
            located.data[0].get("turnSeq").and_then(JsonValue::as_str),
            Some("1")
        );

        // 命题计数与内存侧 claim_count 同口径（聚合返回 xsd:integer，JSON 为数字）。
        let counted = manager
            .query_sparql(&count_claims_sparql(&memory))
            .await
            .unwrap();
        let total = counted.data[0].get("total").cloned().unwrap_or_default();
        assert_eq!(
            total
                .as_u64()
                .or_else(|| { total.as_str().and_then(|s| s.parse::<u64>().ok()) }),
            Some(3),
            "{:?}",
            counted.data
        );

        // 最近焦点：第 3 轮记录了焦点「姚明」。
        let focused = manager
            .query_sparql(&latest_focus_sparql_query(&memory))
            .await
            .unwrap();
        assert_eq!(focused.data.len(), 1, "{:?}", focused.data);
        assert_eq!(
            focused.data[0].get("focus").and_then(JsonValue::as_str),
            Some("姚明")
        );
        assert_eq!(
            focused.data[0].get("turnSeq").and_then(JsonValue::as_str),
            Some("3")
        );

        // 更正关系：让最近回答更正第 2 条回答的命题后，双图联接能取回文本。
        let mut corrected_memory = memory.clone();
        let corrected_uri = corrected_memory.reply_iter().next().unwrap().claims[0]
            .uri
            .clone();
        let latest_seq = corrected_memory.latest_reply().unwrap().seq;
        for stored in corrected_memory.turns.values_mut() {
            for reply in stored.replies.iter_mut() {
                if reply.seq == latest_seq {
                    reply.corrects = vec![corrected_uri.clone()];
                }
            }
        }
        manager
            .sync_generation_memory(&corrected_memory)
            .await
            .unwrap();
        let corrections = manager
            .query_sparql(&corrections_sparql_query(&corrected_memory))
            .await
            .unwrap();
        assert_eq!(corrections.data.len(), 1, "{:?}", corrections.data);
        assert_eq!(
            corrections.data[0]
                .get("correctedText")
                .and_then(JsonValue::as_str),
            Some("姚明是篮球运动员。")
        );
        assert_eq!(
            corrections.data[0]
                .get("replySeq")
                .and_then(JsonValue::as_str),
            Some("4")
        );
    }

    #[tokio::test]
    async fn sparql_builders_execute_against_projected_memory() {
        let dir = TempDir::new().unwrap();
        let (index, _session, _chronicle) = sample_index(&dir);
        let memory = GenerationMemory::from_index(&index, 0);
        let store_dir = dir.path().join("store");
        let manager = OxigraphManager::with_perf_ledger(
            &store_dir.to_string_lossy(),
            Arc::new(PerfLedger::new()),
        )
        .unwrap();
        manager.sync_generation_memory(&memory).await.unwrap();

        // 精确查找：命中第 1 轮（seq=1）的命题。
        let found = manager
            .query_sparql(&find_claim_by_text_sparql(&memory, "姚明是篮球运动员"))
            .await
            .unwrap();
        assert_eq!(found.data.len(), 1, "{:?}", found.data);
        assert_eq!(
            found.data[0].get("text").and_then(JsonValue::as_str),
            Some("姚明是篮球运动员。")
        );
        assert_eq!(
            found.data[0].get("seq").and_then(JsonValue::as_str),
            Some("1"),
            "命题应能定位到第 1 轮"
        );

        // ASK：说过返回真（规范化键与内存路径同口径），没说过返回假。
        let said = manager
            .query_sparql(&already_said_sparql_query(
                &memory,
                "姚明的身高是 226 厘米。",
            ))
            .await
            .unwrap();
        assert_eq!(said.data[0].get("boolean"), Some(&JsonValue::Bool(true)));
        let unsaid = manager
            .query_sparql(&already_said_sparql_query(&memory, "姚明是足球运动员"))
            .await
            .unwrap();
        assert_eq!(unsaid.data[0].get("boolean"), Some(&JsonValue::Bool(false)));

        // 枚举全部命题：只含命题，不含依据文本。
        let claims = manager
            .query_sparql(&list_all_claims_sparql(&memory))
            .await
            .unwrap();
        assert_eq!(claims.data.len(), 3, "{:?}", claims.data);

        // 实体+属性兜底口径：包含两个词面即命中。
        let combined = manager
            .query_sparql(&entity_attribute_said_sparql_query(&memory, "姚明", "身高"))
            .await
            .unwrap();
        assert_eq!(
            combined.data[0].get("boolean"),
            Some(&JsonValue::Bool(true))
        );

        // 跨会话隔离：另一会话同词面不串扰。
        let other_dir = TempDir::new().unwrap();
        let other_session = DialogueSession::open_or_create(
            &other_dir
                .path()
                .join("dialogue.session.json")
                .to_string_lossy(),
        )
        .unwrap();
        let (mut other_chronicle, _) = Chronicle::open(
            &other_dir
                .path()
                .join("dialogue.chronicle.jsonl")
                .to_string_lossy(),
            FsyncPolicy::None,
        )
        .unwrap();
        other_chronicle
            .append_turn("姚明是谁", "query", None, 0)
            .unwrap();
        other_chronicle
            .append_reply("姚明是篮球运动员。", &[], &[], &[])
            .unwrap();
        let other_memory = GenerationMemory::from_chronicle(&other_session, &other_chronicle, 0);
        manager.sync_generation_memory(&other_memory).await.unwrap();
        let scoped = manager
            .query_sparql(&find_claim_by_text_sparql(&memory, "姚明是篮球运动员"))
            .await
            .unwrap();
        assert_eq!(
            scoped.data.len(),
            1,
            "本会话查询不得命中其它会话的同词面命题：{:?}",
            scoped.data
        );
        assert_eq!(
            scoped.data[0].get("seq").and_then(JsonValue::as_str),
            Some("1")
        );
    }
}

//! GRL：生成记忆召回（Generation Recall）——用 RDF 命名图回答「我们说到哪了」。
//!
//! 编年史与对话快照记住「发生过什么」，`DialogueArtifactIndex` 记住「每个工件
//! 在哪里」，`GenerationMemory` 把这些投影成按轮隔离的命名图。本模块补上最后一
//! 环：**这些图能否直接回答元话语问题**。五类问题只读图作答，不重新检索知识：
//!
//! - 「我们刚才聊到哪了 / 回顾一下」：跨重启、有界、带计数的会话回顾；
//! - 「这句话说过吗 / 你是不是说过 X」：对已说命题做规范化比对（`answer_key`），
//!   命中说清是哪一轮回答的第几点，未命中也明确说「没有说过」，不返回空结果；
//! - 「上一轮的第 N 个命题是什么」：按序号取最近一条回答的命题切片；
//! - 「当前焦点是谁」：最近一次记录的实体焦点；
//! - 「上次说的和这次冲突吗」：本会话记录的更正关系（回答 → 被更正命题）。
//!
//! 每类意图都有两条等价路径：SPARQL 查询直查命名图（跨进程、受会话图边界
//! 约束、序号排序投 `xsd:integer`），与内存侧同口径逐字一致（只多一个路径
//! 标记）；SPARQL 失败或语义无法在图上表达（指示词、空记忆）时回退内存。
//!
//! 设计边界：
//! - **只读**：召回不写任何命名图，也不推进知识版本；
//! - **不接管**：裸「第2点」仍是既有回答计划的风格指令，本层只接带
//!   「命题/观点」或「上一轮/上一个回答」限定的说法，两者词面不重叠；
//! - **默认关闭**：`generation_memory.enabled = false` 时调用方不进入本层，
//!   回答路径与关闭前逐字节一致；
//! - **坦白不知道**：记忆为空、没有依据时显式说「还没有记录」，绝不伪装成答案；
//! - **确定性**：排序固定、文案不含挂钟时间，同一份记忆对同一问题逐字节可复现。

use crate::dialogue_uri::ClaimArtifact;
use crate::generation_memory::{self, GenerationMemory};
use crate::OxigraphManager;
use serde_json::Value;

/// 回顾默认展示的最近轮次数（记忆本身已在投影时有界）。
pub const RECAP_TURN_LIMIT: usize = 3;
/// 回顾里问答正文的截断长度（字符）。
const EXCERPT_LIMIT: usize = 42;
/// 「这句话说过吗」里指代本会话最近一条回答命题的指示词。
const DEMONSTRATIVES: &[&str] = &[
    "这句话",
    "这句",
    "这个",
    "这段话",
    "这个命题",
    "上面的",
    "上面的话",
    "上述",
];
/// 回顾意图的问句词面（与对话管理器的 `RECAP_PATTERNS` 互补：那边答内存窗口，
/// 这边答 RDF 命名图；两边都不覆盖的词面才落在这里）。
const RECAP_PATTERNS: &[&str] = &[
    "刚才聊了什么",
    "刚才聊了啥",
    "刚才聊了点什么",
    "刚才说了什么",
    "刚才说过什么",
    "刚才问了什么",
    "刚才答了什么",
    "聊了什么",
    "聊了点什么",
    "聊了啥",
    "聊过什么",
    "我们聊过什么",
    "聊到哪",
    "说到哪",
    "讲到哪",
    "回顾一下",
    "回顾我们的对话",
    "回顾这段对话",
    "对话总结",
    "总结一下对话",
    "总结我们的对话",
];
/// 焦点意图的问句词面。
const FOCUS_PATTERNS: &[&str] = &[
    "当前焦点",
    "现在焦点",
    "话题焦点",
    "在聊谁",
    "在说谁",
    "现在在聊什么",
    "当前话题",
];
/// 「是不是说过 X」的中缀形式（都已含「说过」）。
const SAID_INFIXES: &[&str] = &["是不是说过", "是否说过", "曾经说过", "之前说过", "刚才说过"];
/// 「说过吗：X」的显式前缀形式（冒号可带空格）。
const SAID_PREFIXES: &[&str] = &["说过吗", "讲过吗"];
/// 「X 说过吗」的后缀形式。
const SAID_SUFFIXES: &[&str] = &[
    "说过吗",
    "说过没有",
    "说过没",
    "讲过吗",
    "讲过没有",
    "讲过没",
];
/// 更正/冲突意图的核心词（必须同时出现上下文词才归入召回，避免误伤）。
const CORRECTION_WORDS: &[&str] = &["更正", "冲突", "矛盾", "不一致"];
/// 更正意图的上下文词（把「和上次/之前/先前冲突」区别成更正而非回顾）。
const CORRECTION_CONTEXTS: &[&str] = &["上次", "之前", "先前", "上一轮", "刚才", "上一条"];
/// 待查文本本身是疑问词时不构成「说过吗」（「你刚才说过什么」是回顾）。
const INTERROGATIVES: &[&str] = &[
    "什么",
    "啥",
    "些什么",
    "些什么啊",
    "哪些",
    "哪个",
    "多少",
    "几点",
];
/// 问句末尾的语气词（抽取待查文本时逐层剥除）。
const TRAILING_PARTICLES: &[&str] = &["没有", "吗", "么", "呢", "啊", "呀", "吧", "哈", "嘛"];
/// 「上一轮/上一个回答」这类限定词：它们让「第N点」从风格指令变成召回问题。
const PLAN_STEP_CONTEXTS: &[&str] = &["上一轮", "上一个回答", "上一条回答", "刚才", "上次", "之前"];
/// 带「命题/观点」的点序号说法：与 `StyleDirective` 接受的词面不重叠。
const PLAN_STEP_NOUNS: &[&str] = &["个命题", "个观点", "个要点", "条命题", "条观点", "条要点"];
/// 点序号说法在轮次限定下还需要问句标记，避免「我们刚才聊到第2点了」被误接。
const PLAN_STEP_QUESTIONS: &[&str] = &[
    "是什么",
    "讲的什么",
    "说的什么",
    "内容",
    "讲讲",
    "说说",
    "哪个",
    "哪些",
];

/// 召回意图：只描述「记忆里有什么」，绝不描述知识问句。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecallIntent {
    /// 回顾整个会话（有界、带计数）。
    Recap,
    /// 某句话是否已经作为回答命题说过；`None` 指「这句话/这个」= 最近一条命题。
    AlreadySaid(Option<String>),
    /// 最近一条回答的第 N 个命题（从 1 开始）。
    PlanStep(usize),
    /// 当前实体焦点。
    Focus,
    /// 本会话记录的更正关系（回答 → 其更正的先前命题）。
    Corrections,
}

impl RecallIntent {
    /// 稳定标签：进入编年史计划字段，用于区分召回声。
    pub fn tag(&self) -> &'static str {
        match self {
            RecallIntent::Recap => "recap",
            RecallIntent::AlreadySaid(_) => "already_said",
            RecallIntent::PlanStep(_) => "plan_step",
            RecallIntent::Focus => "focus",
            RecallIntent::Corrections => "corrections",
        }
    }
}

/// 召回回答：确定性文本 + 编年史计划标签 + 依据意图。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecallReply {
    pub intent: RecallIntent,
    pub text: String,
    pub plan: Vec<String>,
}

impl RecallReply {
    pub fn render(&self) -> &str {
        &self.text
    }
}

/// 分类：输入是否是本层负责的元话语问题（纯函数、可先行调用）。
///
/// 调用顺序即优先级：命题序号 > 说过吗 > 回顾 > 焦点。不能判定为元话语的
/// 输入一律返回 `None`，由既有管线继续处理——默认关闭时这层根本不被调用。
pub fn classify(input: &str) -> Option<RecallIntent> {
    let stripped = strip_question(input);
    if stripped.is_empty() {
        return None;
    }
    if let Some(index) = parse_plan_step(stripped) {
        return Some(RecallIntent::PlanStep(index));
    }
    if let Some(target) = parse_said_target(stripped) {
        return Some(RecallIntent::AlreadySaid(target));
    }
    if CORRECTION_WORDS.iter().any(|word| stripped.contains(word))
        && CORRECTION_CONTEXTS
            .iter()
            .any(|context| stripped.contains(context))
    {
        return Some(RecallIntent::Corrections);
    }
    if RECAP_PATTERNS
        .iter()
        .any(|pattern| stripped.contains(pattern))
    {
        return Some(RecallIntent::Recap);
    }
    if FOCUS_PATTERNS
        .iter()
        .any(|pattern| stripped.contains(pattern))
    {
        return Some(RecallIntent::Focus);
    }
    None
}

/// 召回：先分类，再用生成记忆作答；分类失败返回 `None`（不抢答）。
pub fn recall(memory: &GenerationMemory, input: &str) -> Option<RecallReply> {
    let intent = classify(input)?;
    Some(recall_intent(memory, intent))
}

/// 用已判定的意图作答（调用方已分类时避免重复解析）。
pub fn recall_intent(memory: &GenerationMemory, intent: RecallIntent) -> RecallReply {
    let text = match &intent {
        RecallIntent::Recap => render_recap(memory),
        RecallIntent::AlreadySaid(target) => render_said(memory, target.as_deref()),
        RecallIntent::PlanStep(index) => render_plan_step(memory, *index),
        RecallIntent::Focus => render_focus(memory),
        RecallIntent::Corrections => render_corrections(memory),
    };
    let plan = vec!["generation_recall".to_string(), intent.tag().to_string()];
    RecallReply { intent, text, plan }
}

/// SPARQL 召回「说过吗」：命中报位置与原文，未命中报检索量与有界投影范围，
/// 与内存侧 `render_said` 同口径。指示词追问（无显式文本）与空记忆返回
/// `None`，由内存召回补上「这句话指最近第 N 点」「生成记忆为空」的措辞。
async fn recall_already_said_sparql(
    manager: &OxigraphManager,
    memory: &GenerationMemory,
    target: Option<&str>,
) -> Option<String> {
    // 指示词（「这句话说过吗」）需要先解析最近命题，交回内存召回。
    let target = target.filter(|t| !t.trim().is_empty())?;
    if memory.is_empty() {
        return None;
    }
    let query = generation_memory::locate_claim_by_text_sparql(memory, target);
    if query.is_empty() {
        return None;
    }
    let located = manager.query_sparql(&query).await.ok()?;
    // 行按（回答 seq 升序、命题序号升序）排序：首行即最早一次提及，
    // 与内存侧 `find_claim` 的命中顺序一致。
    if let Some(row) = located.data.first() {
        let claim_index = row.get("claimIndex").and_then(Value::as_str)?;
        let reply_seq = row.get("replySeq").and_then(Value::as_str)?;
        let text = row.get("text").and_then(Value::as_str)?;
        let turn_part = match row.get("turnSeq").and_then(Value::as_str) {
            Some(turn_seq) => format!("（关联第 {} 轮）", turn_seq),
            None => String::new(),
        };
        return Some(format!(
            "这句话说过：第 {} 点出自第 {} 条回答{}——「{}」。（SPARQL 确认）",
            claim_index, reply_seq, turn_part, text
        ));
    }
    // 未命中：报检索量；有界投影下披露范围（收据与图同源，直接读内存侧）。
    let total = manager
        .query_sparql(&generation_memory::count_claims_sparql(memory))
        .await
        .ok()
        .and_then(|counted| {
            counted
                .data
                .first()
                .and_then(|row| row.get("total"))
                .cloned()
        })
        .and_then(|total| total.as_u64().map(|n| n as usize))
        .unwrap_or_else(|| memory.claim_count());
    let bound = match memory.projection.as_ref() {
        Some(receipt) if !receipt.dropped_turns.is_empty() => format!(
            "；记忆为最近 {} 轮的有界投影，更早 {} 轮不在检索范围",
            receipt.kept_turns.len(),
            receipt.dropped_turns.len()
        ),
        _ => String::new(),
    };
    Some(format!(
        "这句话在本会话的生成记忆里没有说过（已检索 {} 条命题{}）。（SPARQL 确认）",
        total, bound
    ))
}

/// SPARQL 召回「第 N 个命题」：与内存侧 `render_plan_step` 同口径——最近一条
/// 回答的第 N 点；越界说清共几点，没有回答显式说明。查询失败或结果异常返回
/// `None`，回退内存召回。
async fn recall_plan_step_sparql(
    manager: &OxigraphManager,
    memory: &GenerationMemory,
    index: usize,
) -> Option<String> {
    let query = generation_memory::latest_reply_claims_sparql(memory);
    if query.is_empty() {
        return None;
    }
    let result = manager.query_sparql(&query).await.ok()?;
    // 行已按（回答 seq 降序、命题序号升序）排序：第一组行即最近一条回答。
    let latest_seq = result
        .data
        .first()
        .and_then(|row| row.get("replySeq"))
        .and_then(Value::as_str)
        .and_then(|s| s.parse::<u64>().ok())?;
    let claims: Vec<(u32, &str)> = result
        .data
        .iter()
        .filter(|row| {
            row.get("replySeq")
                .and_then(Value::as_str)
                .and_then(|s| s.parse::<u64>().ok())
                == Some(latest_seq)
        })
        .filter_map(|row| {
            let claim_index = row
                .get("claimIndex")
                .and_then(Value::as_str)?
                .parse::<u32>()
                .ok()?;
            let text = row.get("text").and_then(Value::as_str)?;
            Some((claim_index, text))
        })
        .collect();
    if claims.is_empty() {
        return None;
    }
    match claims
        .iter()
        .find(|(claim_index, _)| *claim_index == index as u32)
    {
        Some((_, text)) => Some(format!(
            "第 {} 点出自第 {} 条回答：「{}」。（SPARQL 召回）",
            index, latest_seq, text
        )),
        None => Some(format!(
            "最近一条回答（第 {} 条）一共 {} 点，没有第 {} 点。（SPARQL 召回）",
            latest_seq,
            claims.len(),
            index
        )),
    }
}

/// SPARQL 召回回顾：跨轮读取回答；排序按 seq 数值升序（字面量投 xsd:integer，
/// 避免多位数词面按字典序错排）。图集合受会话边界约束，不扫全图。
async fn recall_recap_sparql(
    manager: &OxigraphManager,
    memory: &GenerationMemory,
) -> Option<String> {
    let query = format!(
        "{} SELECT ?turnSeq ?user ?replySeq ?answer WHERE {{\
            GRAPH ?g {{\
                ?turn a <{}> .\
                ?turn <{}> ?turnSeq .\
                ?turn <{}> ?user .\
                OPTIONAL {{\
                    ?reply a <{}> .\
                    ?reply <{}> ?replySeq .\
                    ?reply <{}> ?answer .\
                    ?reply <{}> ?turnSeq .\
                }}\
            }}\
            FILTER(?g IN ({}))\
        }} ORDER BY ASC(xsd:integer(?turnSeq)) ASC(xsd:integer(?replySeq))",
        generation_memory::XSD_PREFIX,
        generation_memory::TURN_CLASS,
        generation_memory::P_SEQ,
        generation_memory::P_USER,
        generation_memory::REPLY_CLASS,
        generation_memory::P_SEQ,
        generation_memory::P_ANSWER,
        generation_memory::P_TURN_SEQ,
        memory
            .graph_iris()
            .iter()
            .map(|iri| format!("<{}>", iri))
            .collect::<Vec<_>>()
            .join(", ")
    );
    match manager.query_sparql(&query).await {
        Ok(result) => {
            if result.data.is_empty() {
                return Some("生成记忆里还没有投影任何轮次（SPARQL 确认）。".to_string());
            }
            let mut text = String::new();
            text.push_str(&format!(
                "生成记忆回顾（SPARQL 命名图）：{}、{} 条回答、{} 条命题、{} 条依据。\n",
                projection_scope(memory),
                memory.reply_count(),
                memory.claim_count(),
                memory.evidence_count()
            ));
            let mut current_turn = 0u64;
            for row in &result.data {
                let turn_seq = row
                    .get("turnSeq")
                    .and_then(|v| v.as_str())
                    .and_then(|s| s.parse::<u64>().ok())
                    .unwrap_or(0);
                let user = row.get("user").and_then(|v| v.as_str()).unwrap_or("");
                let reply_seq = row
                    .get("replySeq")
                    .and_then(|v| v.as_str())
                    .and_then(|s| s.parse::<u64>().ok());
                let answer = row.get("answer").and_then(|v| v.as_str()).unwrap_or("");
                if turn_seq != current_turn {
                    current_turn = turn_seq;
                    text.push_str(&format!("第 {} 轮你问「{}」。", turn_seq, user));
                }
                if let Some(rseq) = reply_seq {
                    if !answer.is_empty() {
                        let excerpt = if answer.chars().count() > 40 {
                            format!("{}…", answer.chars().take(40).collect::<String>())
                        } else {
                            answer.to_string()
                        };
                        text.push_str(&format!("  回答 {}：{}\n", rseq, excerpt));
                    }
                }
            }
            Some(text)
        }
        Err(_) => None,
    }
}

/// SPARQL 召回更正关系：与内存侧 `render_corrections` 同口径——列出
/// 更正回答与其撤回的先前命题；被更正命题所在图被丢弃的行自然消失。
async fn recall_corrections_sparql(
    manager: &OxigraphManager,
    memory: &GenerationMemory,
) -> Option<String> {
    let query = generation_memory::corrections_sparql_query(memory);
    let result = manager.query_sparql(&query).await.ok()?;
    let corrections: Vec<(String, String)> = result
        .data
        .iter()
        .filter_map(|row| {
            let answer = row.get("answer").and_then(Value::as_str).unwrap_or("");
            let corrected = row.get("correctedText").and_then(Value::as_str)?;
            Some((excerpt(answer, 40), corrected.to_string()))
        })
        .collect();
    if corrections.is_empty() {
        return Some(
            "本会话还没有记录更正关系；回答未标记更正任何先前命题。（SPARQL 确认）".to_string(),
        );
    }
    let mut out = format!("本会话共记录 {} 条更正关系。\n", corrections.len());
    for (position, (reply_excerpt, corrected)) in corrections.iter().enumerate() {
        out.push_str(&format!(
            "  {}. 更正回答「{}」撤回了先前命题「{}」。\n",
            position + 1,
            reply_excerpt,
            corrected
        ));
    }
    out.push_str("（SPARQL 确认）");
    Some(out)
}

/// SPARQL 召回当前焦点：焦点与轮次取自图（seq 数值降序取一）；提及计数与
/// 内存侧同源（同一份已加载记忆）。无焦点行时返回 `None`，回退内存召回。
async fn recall_focus_sparql(
    manager: &OxigraphManager,
    memory: &GenerationMemory,
) -> Option<String> {
    let query = generation_memory::latest_focus_sparql_query(memory);
    let result = manager.query_sparql(&query).await.ok()?;
    let row = result.data.first()?;
    let focus = row.get("focus").and_then(Value::as_str)?;
    let turn_seq = row.get("turnSeq").and_then(Value::as_str)?;
    Some(format!(
        "当前实体焦点是「{}」（第 {} 轮记录；本会话共 {} 次实体提及）。（SPARQL 确认）",
        focus,
        turn_seq,
        mention_count(memory)
    ))
}

/// SPARQL-based recall dispatcher：使用已加载的记忆，使图查询受限于该会话
/// 实际承载的命名图（`graph_iris()`），避免全图扫描。
pub async fn recall_sparql(
    manager: &OxigraphManager,
    memory: &GenerationMemory,
    input: &str,
) -> Option<RecallReply> {
    let intent = classify(input)?;
    let intent_tag = intent.tag().to_string();
    let text = match &intent {
        RecallIntent::AlreadySaid(target) => {
            recall_already_said_sparql(manager, memory, target.as_deref()).await
        }
        RecallIntent::PlanStep(index) => recall_plan_step_sparql(manager, memory, *index).await,
        RecallIntent::Recap => recall_recap_sparql(manager, memory).await,
        RecallIntent::Corrections => recall_corrections_sparql(manager, memory).await,
        RecallIntent::Focus => recall_focus_sparql(manager, memory).await,
    };
    text.map(|text| RecallReply {
        intent,
        text,
        plan: vec!["generation_recall_sparql".to_string(), intent_tag],
    })
}

/// 投影范围词（内存与 SPARQL 回顾共用同一口径）：有界投影如实披露
/// 「编年史共几轮、另有几轮更早轮次不在记忆中」，不谎称全量。
fn projection_scope(memory: &GenerationMemory) -> String {
    match memory.projection.as_ref() {
        Some(receipt) if receipt.dropped_turns.is_empty() => {
            format!("本会话共 {} 轮", memory.turn_count())
        }
        Some(receipt) => format!(
            "投影保留最近 {} 轮（编年史共 {} 轮，另有 {} 轮更早轮次不在记忆中）",
            receipt.kept_turns.len(),
            receipt.source_turns,
            receipt.dropped_turns.len()
        ),
        None => format!("本会话共 {} 轮", memory.turn_count()),
    }
}

/// 会话回顾：计数 + 最近若干轮的问答切片 + 当前焦点；空记忆显式说明。
///
/// 有界投影不得谎称「本会话共 N 轮」：投影收据记录了丢弃的轮次，
/// 措辞必须披露「另有 K 轮更早轮次不在记忆中」。
fn render_recap(memory: &GenerationMemory) -> String {
    if memory.is_empty() {
        return "生成记忆里还没有投影任何轮次；先用「dialogue-memory --sync」从编年史投影，再问这句话。"
            .to_string();
    }
    let mut text = format!(
        "生成记忆回顾（RDF 命名图）：{}、{} 条回答、{} 条命题、{} 条依据。",
        projection_scope(memory),
        memory.reply_count(),
        memory.claim_count(),
        memory.evidence_count()
    );
    for stored in memory.recent_turns(RECAP_TURN_LIMIT) {
        let replies = stored
            .replies
            .iter()
            .map(|reply| format!("答「{}」", excerpt(&reply.answer, EXCERPT_LIMIT)))
            .collect::<Vec<_>>()
            .join("；");
        text.push_str(&format!(
            "第 {} 轮你问「{}」{}。",
            stored.turn.seq,
            excerpt(&stored.turn.user, EXCERPT_LIMIT),
            if replies.is_empty() {
                "，本轮还没有登记回答".to_string()
            } else {
                format!("，我{}", replies)
            },
        ));
    }
    match latest_focus(memory) {
        Some((focus, _)) => text.push_str(&format!("当前实体焦点是「{}」。", focus)),
        None => text.push_str("本会话还没有记录实体焦点。"),
    }
    text
}

/// 「这句话说过吗」：命中报位置与原文，未命中明确否定，空记忆先提示投影。
fn render_said(memory: &GenerationMemory, target: Option<&str>) -> String {
    if memory.is_empty() {
        return "生成记忆为空，无法确认是否说过；先用「dialogue-memory --sync」投影再用本句追问。"
            .to_string();
    }
    let Some(target) = target.filter(|target| !target.is_empty()) else {
        // 「这句话说过吗」指向最近一条命题：没有命题就没有「这句话」。
        return match memory.latest_reply().and_then(|reply| reply.claims.last()) {
            Some(claim) => match locate_reply(memory, claim) {
                Some(reply) => format!(
                    "「这句话」指我最近说过的第 {} 点（第 {} 条回答）：「{}」。",
                    claim.index, reply.seq, claim.text
                ),
                None => format!(
                    "「这句话」指我最近说过的第 {} 点：「{}」。",
                    claim.index, claim.text
                ),
            },
            None => "还没有可指代的「这句话」：最近一条回答里没有命题。".to_string(),
        };
    };
    match memory.find_claim(target) {
        Some(claim) => match locate_reply(memory, claim) {
            Some(reply) => format!(
                "这句话说过：第 {} 点出自第 {} 条回答（关联第 {} 轮）——「{}」。",
                claim.index,
                reply.seq,
                reply.turn_seq.unwrap_or(0),
                claim.text
            ),
            None => format!("这句话说过：第 {} 点——「{}」。", claim.index, claim.text),
        },
        None => {
            // 有界投影下「没有说过」必须披露检索范围：更早轮次不在记忆里，
            // 未命中不等于从未说过。
            let bound = match memory.projection.as_ref() {
                Some(receipt) if !receipt.dropped_turns.is_empty() => format!(
                    "；记忆为最近 {} 轮的有界投影，更早 {} 轮不在检索范围",
                    receipt.kept_turns.len(),
                    receipt.dropped_turns.len()
                ),
                _ => String::new(),
            };
            format!(
                "这句话在本会话的生成记忆里没有说过（已检索 {} 条命题{}）。",
                memory.claim_count(),
                bound
            )
        }
    }
}

/// 「第 N 个命题」：命中给原文，越界说清共几点，无回答显式说明。
fn render_plan_step(memory: &GenerationMemory, index: usize) -> String {
    let Some(reply) = memory.latest_reply() else {
        return "生成记忆里还没有可回顾的回答。".to_string();
    };
    match reply
        .claims
        .iter()
        .find(|claim| claim.index == index as u32)
    {
        Some(claim) => format!(
            "第 {} 点出自第 {} 条回答：「{}」。",
            index, reply.seq, claim.text
        ),
        None => format!(
            "最近一条回答（第 {} 条）一共 {} 点，没有第 {} 点。",
            reply.seq,
            reply.claims.len(),
            index
        ),
    }
}

/// 当前焦点：优先轮次记录的 focus，其次最近一次实体提及。
fn render_focus(memory: &GenerationMemory) -> String {
    if memory.is_empty() {
        return "生成记忆里还没有投影任何轮次，没有焦点可回顾。".to_string();
    }
    match latest_focus(memory) {
        Some((focus, seq)) => format!(
            "当前实体焦点是「{}」（第 {} 轮记录；本会话共 {} 次实体提及）。",
            focus,
            seq,
            mention_count(memory)
        ),
        None => "本会话还没有记录实体焦点。".to_string(),
    }
}

/// 「更正/冲突」：列出本会话记录的更正关系（更正回答 → 被更正命题），
/// 空记忆或无更正时诚实说明。
fn render_corrections(memory: &GenerationMemory) -> String {
    let mut corrections: Vec<(String, String)> = Vec::new();
    for reply in memory.reply_iter() {
        for corrected_uri in &reply.corrects {
            let corrected_text = memory
                .reply_iter()
                .flat_map(|r| r.claims.iter())
                .find(|claim| claim.uri == *corrected_uri)
                .map(|claim| claim.text.clone());
            if let Some(text) = corrected_text {
                corrections.push((excerpt(&reply.answer, 40), text));
            }
        }
    }
    if corrections.is_empty() {
        return "本会话还没有记录更正关系；回答未标记更正任何先前命题。".to_string();
    }
    let mut out = format!("本会话共记录 {} 条更正关系。\n", corrections.len());
    for (position, (reply_excerpt, corrected)) in corrections.into_iter().enumerate() {
        out.push_str(&format!(
            "  {}. 更正回答「{}」撤回了先前命题「{}」。\n",
            position + 1,
            reply_excerpt,
            corrected
        ));
    }
    out
}

/// 定位命题所属的回答（用于把「第几点」讲清是哪条回答的）。
fn locate_reply<'a>(
    memory: &'a GenerationMemory,
    claim: &ClaimArtifact,
) -> Option<&'a crate::dialogue_uri::ReplyArtifact> {
    memory
        .reply_iter()
        .find(|reply| reply.claims.iter().any(|stored| stored.uri == claim.uri))
}

/// 最近的实体焦点：（焦点名, 轮次 seq）。轮次按 seq 降序取第一条 focus。
fn latest_focus(memory: &GenerationMemory) -> Option<(String, u64)> {
    memory.recent_turns(0).into_iter().rev().find_map(|stored| {
        stored
            .turn
            .focus
            .clone()
            .map(|focus| (focus, stored.turn.seq))
    })
}

/// 投影内全部实体提及计数（确定性：固定遍历累加）。
fn mention_count(memory: &GenerationMemory) -> usize {
    memory
        .recent_turns(0)
        .into_iter()
        .map(|stored| stored.turn.mentions.len())
        .sum()
}

/// 去掉问句尾饰与首尾空白，得到可模式匹配的词面。
fn strip_question(input: &str) -> &str {
    input
        .trim()
        .trim_end_matches(['？', '?', '。', '！', '!', '；', ';', '，', ',', ' '])
        .trim()
}

/// 截断展示用正文（与 `generation_memory` 的截断同口径：折叠空白 + 省略号）。
fn excerpt(text: &str, limit: usize) -> String {
    let joined = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut out: String = joined.chars().take(limit).collect();
    if joined.chars().count() > limit {
        out.push('…');
    }
    out
}

/// 解析「第N点/第N条/第N个命题」类说法；只接与风格指令不重叠的词面。
fn parse_plan_step(text: &str) -> Option<usize> {
    let ordinal = parse_step_ordinal(text)?;
    let tail = ordinal.tail.trim();
    // 「第N个命题/观点」自带名词标记，不需要轮次限定。
    if PLAN_STEP_NOUNS.iter().any(|noun| tail.starts_with(noun)) {
        return Some(ordinal.value);
    }
    // 「上一轮/上一个回答」限定的点序号：点/条标记之后必须是问句尾巴。
    let has_context = PLAN_STEP_CONTEXTS
        .iter()
        .any(|context| text.contains(context));
    let has_question = PLAN_STEP_QUESTIONS
        .iter()
        .any(|question| text.contains(question));
    if !has_context || !has_question {
        return None;
    }
    let marker = tail
        .strip_prefix("点")
        .or_else(|| tail.strip_prefix("条"))
        .unwrap_or(tail)
        .trim();
    if marker.is_empty()
        || marker.starts_with("呢")
        || PLAN_STEP_QUESTIONS
            .iter()
            .any(|question| marker.starts_with(question))
    {
        return Some(ordinal.value);
    }
    None
}

/// 点序号解析结果：数值 + 序号词面 + 序号（含「第」与数字）之后的剩余文本。
struct StepOrdinal {
    value: usize,
    tail: String,
}

/// 扫描「第N」：N 支持阿拉伯数字与中文数字（到 99），必须跟随
/// 「点/条/个命题/个观点」之类标记或轮次限定，否则不算点序号。
fn parse_step_ordinal(text: &str) -> Option<StepOrdinal> {
    let start = text.find('第')?;
    let rest = &text[start + '第'.len_utf8()..];
    let mut digits = String::new();
    for ch in rest.chars() {
        if ch.is_ascii_digit() {
            digits.push(ch);
            if digits.len() == 2 {
                break;
            }
        } else {
            break;
        }
    }
    let (value, tail) = if !digits.is_empty() {
        (
            digits.parse::<usize>().ok()?,
            rest[digits.len()..].to_string(),
        )
    } else {
        let (marker, tail) = split_chinese_ordinal(rest)?;
        (chinese_number(marker)?, tail.to_string())
    };
    if value == 0 {
        return None;
    }
    Some(StepOrdinal { value, tail })
}

/// 切分句首中文序数词（「二」「十」「十二」），返回（词面, 剩余文本）。
fn split_chinese_ordinal(text: &str) -> Option<(&str, &str)> {
    let first = text.chars().next()?;
    let first_width = first.len_utf8();
    if first == '十' {
        return Some((&text[..first_width], &text[first_width..]));
    }
    if chinese_digit(first).is_none() {
        return None;
    }
    if let Some(second) = text[first_width..].chars().next() {
        let second_width = second.len_utf8();
        if second == '十' {
            if let Some(ones) = text[first_width + second_width..].chars().next() {
                if chinese_digit(ones).is_some() {
                    let width = first_width + second_width + ones.len_utf8();
                    return Some((&text[..width], &text[width..]));
                }
            }
            let width = first_width + second_width;
            return Some((&text[..width], &text[width..]));
        }
    }
    Some((&text[..first_width], &text[first_width..]))
}

/// 单个中文数字（0–9）。
fn chinese_digit(ch: char) -> Option<usize> {
    match ch {
        '零' => Some(0),
        '一' => Some(1),
        '二' => Some(2),
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

/// 中文整数（含「十」「二十」），与 `dialogue_uri::chinese_number` 同口径。
fn chinese_number(text: &str) -> Option<usize> {
    if text == "十" {
        return Some(10);
    }
    if let Some((tens, ones)) = text.split_once('十') {
        let tens = if tens.is_empty() {
            1
        } else {
            chinese_digit(tens.chars().next()?)?
        };
        let ones = if ones.is_empty() {
            0
        } else {
            chinese_digit(ones.chars().next()?)?
        };
        return Some(tens * 10 + ones);
    }
    chinese_digit(text.chars().next()?)
}

/// 抽取待查文本：显式前缀 > 中缀「是不是说过」> 后缀「X 说过吗」。
///
/// 返回值：外层 `None` = 不是「说过吗」意图（调用方继续匹配别的意图）；
/// 内层 `None` = 指示词追问（「这句话说过吗」指向最近一条命题）。
fn parse_said_target(text: &str) -> Option<Option<String>> {
    for prefix in SAID_PREFIXES {
        if let Some(rest) = text.strip_prefix(prefix) {
            let rest = rest.trim().trim_start_matches(['：', ':', ' ']).trim();
            let target = strip_particles(rest);
            return Some((!target.is_empty()).then(|| target.to_string()));
        }
    }
    for infix in SAID_INFIXES {
        if let Some(position) = text.find(infix) {
            let raw = text[position + infix.len()..].trim();
            // 「你刚才说过什么」不是说过吗，是回顾。
            if INTERROGATIVES.iter().any(|word| raw == *word) {
                return None;
            }
            let target = strip_particles(strip_lead(raw));
            if target.is_empty() {
                return None;
            }
            return Some(Some(target.to_string()));
        }
    }
    for suffix in SAID_SUFFIXES {
        if let Some(head) = text.strip_suffix(suffix) {
            let head = head.trim();
            if DEMONSTRATIVES.iter().any(|word| head == *word) {
                return Some(None);
            }
            let target = strip_particles(strip_lead(head));
            return Some((!target.is_empty()).then(|| target.to_string()));
        }
    }
    None
}

/// 剥掉待查文本前的引导词（「你说的」「你之前说的」等）。
fn strip_lead(head: &str) -> &str {
    let mut rest = head.trim();
    for prefix in [
        "你刚才说的",
        "你之前说的",
        "你说的",
        "你讲的",
        "你刚才说",
        "你说",
        "你刚才",
        "你之前",
    ] {
        if let Some(stripped) = rest.strip_prefix(prefix) {
            rest = stripped.trim();
            break;
        }
    }
    rest
}

/// 逐层剥除句末语气词与标点（「…吗」「…没有」），保留正文本身。
fn strip_particles(text: &str) -> &str {
    let mut rest = text.trim();
    loop {
        let stripped = rest
            .trim_end_matches(['？', '?', '。', '！', '!', '，', ',', '；', ';', ' '])
            .trim();
        let stripped = strip_suffix_particles(stripped);
        if stripped == rest {
            return rest;
        }
        rest = stripped;
    }
}

/// 剥掉末尾语气词（「吗」「没有」「呢」…），一次一个，直到没有为止。
fn strip_suffix_particles(text: &str) -> &str {
    let mut rest = text;
    loop {
        let mut stripped_one = rest;
        for particle in TRAILING_PARTICLES {
            if let Some(head) = rest.strip_suffix(particle) {
                stripped_one = head.trim_end();
                break;
            }
        }
        if stripped_one == rest {
            return rest;
        }
        rest = stripped_one;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chronicle::{answer_key, Chronicle, EntityRef, FsyncPolicy};
    use crate::dialogue_uri::{DialogueArtifactIndex, DialogueSession};
    use crate::perf_ledger::PerfLedger;
    use crate::OxigraphManager;
    use std::sync::Arc;
    use tempfile::TempDir;

    fn sample_memory(dir: &TempDir) -> GenerationMemory {
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
            .append_turn_with_refs("他的升高呢", "query", Some("姚明"), 1, &[])
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
        GenerationMemory::from_index(&index, 0)
    }

    #[test]
    fn classify_does_not_steal_style_directives_or_questions() {
        assert_eq!(classify("小明多大？"), None);
        assert_eq!(classify("姚明和李四什么关系"), None);
        // 裸「第2点」仍是回答计划风格指令的辖域，召回不得接管。
        assert_eq!(classify("第2点"), None);
        assert_eq!(classify("第二点是什么"), None);
        assert_eq!(classify("风格：第2点"), None);
        assert_eq!(classify("详细一点"), None);
        assert_eq!(classify("我们刚才聊到第2点了"), None);
        // 带命题限定或轮次限定的说法归召回。
        assert_eq!(
            classify("上一轮的第2点是什么"),
            Some(RecallIntent::PlanStep(2))
        );
        assert_eq!(classify("第2个命题是什么"), Some(RecallIntent::PlanStep(2)));
        assert_eq!(classify("刚才聊了什么"), Some(RecallIntent::Recap));
        assert_eq!(classify("当前焦点是谁"), Some(RecallIntent::Focus));
        assert_eq!(
            classify("说过吗：姚明是篮球运动员"),
            Some(RecallIntent::AlreadySaid(Some(
                "姚明是篮球运动员".to_string()
            )))
        );
    }

    #[test]
    fn classify_extracts_said_target_from_every_form() {
        assert_eq!(
            classify("你是不是说过姚明是篮球运动员"),
            Some(RecallIntent::AlreadySaid(Some(
                "姚明是篮球运动员".to_string()
            )))
        );
        assert_eq!(
            classify("你之前说过姚明的身高是226厘米吗？"),
            Some(RecallIntent::AlreadySaid(Some(
                "姚明的身高是226厘米".to_string()
            )))
        );
        assert_eq!(
            classify("姚明是篮球运动员 说过吗"),
            Some(RecallIntent::AlreadySaid(Some(
                "姚明是篮球运动员".to_string()
            )))
        );
        assert_eq!(
            classify("这句话说过吗"),
            Some(RecallIntent::AlreadySaid(None))
        );
        assert_eq!(
            classify("姚明是足球运动员 说过没有"),
            Some(RecallIntent::AlreadySaid(Some(
                "姚明是足球运动员".to_string()
            )))
        );
        // 疑问词追问不是「说过吗」：按回顾继续。
        assert_eq!(classify("你刚才说过什么"), Some(RecallIntent::Recap));
    }

    #[test]
    fn already_said_answers_with_location_or_honest_negative() {
        let dir = TempDir::new().unwrap();
        let memory = sample_memory(&dir);
        let said = recall(&memory, "说过吗：姚明是篮球运动员").unwrap();
        assert_eq!(said.intent.tag(), "already_said");
        assert!(said.text.contains("这句话说过"), "{}", said.text);
        assert!(said.text.contains("姚明是篮球运动员"), "{}", said.text);

        // 规范化比对：空白/标点不影响「说过」判定。
        let said_normalized = recall(&memory, "说过吗： 姚明是篮球运动员。").unwrap();
        assert_eq!(said.text, said_normalized.text);

        let unsaid = recall(&memory, "说过吗：姚明是足球运动员").unwrap();
        assert!(unsaid.text.contains("没有说过"), "{}", unsaid.text);
        assert!(unsaid.text.contains('3'), "{}", unsaid.text);

        let demonstrative = recall(&memory, "这句话说过吗").unwrap();
        assert!(
            demonstrative.text.contains("指我最近说过"),
            "{}",
            demonstrative.text
        );
    }

    #[test]
    fn plan_step_slices_latest_reply_and_reports_out_of_range() {
        let dir = TempDir::new().unwrap();
        let memory = sample_memory(&dir);
        let step = recall(&memory, "上一个回答的第1个命题是什么").unwrap();
        assert!(step.text.contains("姚明的身高是 226 厘米"), "{}", step.text);

        let missing = recall(&memory, "上一个回答的第3个命题是什么").unwrap();
        assert!(
            missing.text.contains("一共 1 点，没有第 3 点"),
            "{}",
            missing.text
        );
    }

    #[test]
    fn recap_and_focus_are_deterministic() {
        let dir = TempDir::new().unwrap();
        let memory = sample_memory(&dir);
        let first = recall(&memory, "我们刚才聊了点什么").unwrap();
        let second = recall(&memory, "回顾一下我们的对话").unwrap();
        assert_eq!(first.text, second.text, "同一记忆的回顾必须逐字节一致");
        assert!(first.text.contains("共 2 轮"), "{}", first.text);
        assert!(first.text.contains("姚明是谁"), "{}", first.text);
        assert!(
            first.text.contains("当前实体焦点是「姚明」"),
            "{}",
            first.text
        );

        let focus = recall(&memory, "当前焦点是谁").unwrap();
        assert!(focus.text.contains("「姚明」"), "{}", focus.text);
    }

    #[test]
    fn bounded_projection_recap_and_said_disclose_the_window() {
        let dir = TempDir::new().unwrap();
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
            .append_reply("姚明是篮球运动员。", &[], &[], &[])
            .unwrap();
        chronicle
            .append_turn_with_refs("他的身高呢", "query", Some("姚明"), 1, &[])
            .unwrap();
        chronicle
            .append_reply("姚明的身高是 226 厘米。", &[], &[], &[])
            .unwrap();
        let index = DialogueArtifactIndex::from_chronicle(&session, &chronicle);
        let memory = GenerationMemory::from_index(&index, 1);

        // 有界投影的回顾不得谎称「本会话共 1 轮」。
        let recap = recall(&memory, "我们刚才聊了点什么").unwrap();
        assert!(recap.text.contains("投影保留最近 1 轮"), "{}", recap.text);
        assert!(
            recap
                .text
                .contains("编年史共 2 轮，另有 1 轮更早轮次不在记忆中"),
            "{}",
            recap.text
        );

        // 「没有说过」在有界投影下必须披露检索范围。
        let miss = recall(&memory, "说过吗：姚明是篮球运动员").unwrap();
        assert!(miss.text.contains("没有说过"), "{}", miss.text);
        assert!(miss.text.contains("更早 1 轮不在检索范围"), "{}", miss.text);

        // 无丢弃的投影沿用「本会话共 N 轮」。
        let full = GenerationMemory::from_index(&index, 0);
        let full_recap = recall(&full, "我们刚才聊了点什么").unwrap();
        assert!(
            full_recap.text.contains("本会话共 2 轮"),
            "{}",
            full_recap.text
        );
        let full_miss = recall(&full, "说过吗：姚明是足球运动员").unwrap();
        assert!(!full_miss.text.contains("有界投影"), "{}", full_miss.text);
    }

    #[test]
    fn empty_memory_says_instead_of_faking() {
        let empty = GenerationMemory::empty("session-x");
        assert!(recall(&empty, "刚才聊了什么")
            .unwrap()
            .text
            .contains("还没有投影"));
        assert!(recall(&empty, "说过吗：姚明是谁")
            .unwrap()
            .text
            .contains("生成记忆为空"));
        assert!(recall(&empty, "上一个回答的第1个命题是什么")
            .unwrap()
            .text
            .contains("还没有可回顾的回答"));
        assert!(recall(&empty, "当前焦点是谁")
            .unwrap()
            .text
            .contains("还没有投影任何轮次"));
    }

    #[tokio::test]
    async fn sparql_recap_discloses_bounded_projection_window() {
        let dir = TempDir::new().unwrap();
        let session_path = dir.path().join("dialogue.session.json");
        let session = DialogueSession::open_or_create(&session_path.to_string_lossy()).unwrap();
        let chronicle_path = dir.path().join("dialogue.chronicle.jsonl");
        let (mut chronicle, _) =
            Chronicle::open(&chronicle_path.to_string_lossy(), FsyncPolicy::None).unwrap();
        chronicle
            .append_turn_with_refs("姚明是谁", "query", None, 0, &[])
            .unwrap();
        chronicle
            .append_reply("姚明是篮球运动员。", &[], &[], &[])
            .unwrap();
        chronicle
            .append_turn_with_refs("他的身高呢", "query", Some("姚明"), 1, &[])
            .unwrap();
        chronicle
            .append_reply("姚明的身高是 226 厘米。", &[], &[], &[])
            .unwrap();
        let index = DialogueArtifactIndex::from_chronicle(&session, &chronicle);
        // 有界投影：只保留最近 1 轮。
        let memory = GenerationMemory::from_index(&index, 1);
        let manager = OxigraphManager::with_perf_ledger(
            &dir.path().join("store").to_string_lossy(),
            Arc::new(PerfLedger::new()),
        )
        .unwrap();
        manager.sync_generation_memory(&memory).await.unwrap();
        let loaded = manager
            .load_generation_memory(&memory.session)
            .await
            .unwrap();

        let recap = recall_sparql(&manager, &loaded, "我们刚才聊了点什么")
            .await
            .unwrap();
        // SPARQL 回顾与内存侧共用同一范围词：不得谎称「本会话共 1 轮」。
        assert!(
            recap
                .text
                .contains("投影保留最近 1 轮（编年史共 2 轮，另有 1 轮更早轮次不在记忆中）"),
            "{}",
            recap.text
        );
        assert!(
            !recap.text.contains("姚明是谁"),
            "被窗口丢弃的轮次不得出现在回顾里：{}",
            recap.text
        );
        assert!(
            recap.text.contains("姚明的身高是 226 厘米"),
            "{}",
            recap.text
        );
    }

    #[test]
    fn recall_returns_none_for_non_meta_input() {
        let dir = TempDir::new().unwrap();
        let memory = sample_memory(&dir);
        assert!(recall(&memory, "姚明的身高是多少").is_none());
        assert!(recall(&memory, "").is_none());
        assert!(recall(&memory, "帮助").is_none());
    }

    #[tokio::test]
    async fn recall_reads_what_was_synced_to_named_graphs() {
        let dir = TempDir::new().unwrap();
        let memory = sample_memory(&dir);
        let manager = OxigraphManager::with_perf_ledger(
            &dir.path().join("store").to_string_lossy(),
            Arc::new(PerfLedger::new()),
        )
        .unwrap();
        manager.sync_generation_memory(&memory).await.unwrap();
        let loaded = manager
            .load_generation_memory(&memory.session)
            .await
            .unwrap();
        assert_eq!(loaded, memory, "召回前必须能无损读回记忆");

        let misspelled = recall(&loaded, "说过吗：姚明的升高是226厘米").unwrap();
        // 升高/身高不属规范化差异，未说过要如实说。
        assert!(misspelled.text.contains("没有说过"), "{}", misspelled.text);

        // 规范化只折叠空白/标点/大小写：数字与汉字差异仍算没说过。
        let said = recall(&loaded, "说过吗： 姚明的身高是 226 厘米。").unwrap();
        assert!(said.text.contains("这句话说过"), "{}", said.text);
    }

    #[tokio::test]
    async fn recall_sparql_bounds_to_session_graphs_and_isolates_sessions() {
        let dir = TempDir::new().unwrap();
        let memory = sample_memory(&dir);
        let manager = OxigraphManager::with_perf_ledger(
            &dir.path().join("store").to_string_lossy(),
            Arc::new(PerfLedger::new()),
        )
        .unwrap();
        manager.sync_generation_memory(&memory).await.unwrap();
        let loaded = manager
            .load_generation_memory(&memory.session)
            .await
            .unwrap();

        // SPARQL 已说查询：命中本会话命题，受图边界约束。
        let said = recall_sparql(&manager, &loaded, "说过吗：姚明是篮球运动员")
            .await
            .unwrap();
        assert!(said.text.contains("这句话说过"), "{}", said.text);

        let unsaid = recall_sparql(&manager, &loaded, "说过吗：姚明是足球运动员")
            .await
            .unwrap();
        assert!(unsaid.text.contains("没有说过"), "{}", unsaid.text);

        // 回顾：跨轮读取回答。
        let recap = recall_sparql(&manager, &loaded, "我们刚才聊了点什么")
            .await
            .unwrap();
        assert!(recap.text.contains("SPARQL 命名图"), "{}", recap.text);

        // 第 N 个命题：SPARQL 召回与内存侧同口径——最近一条回答（seq 4）的第 N 点。
        let step = recall_sparql(&manager, &loaded, "上一个回答的第1个命题是什么")
            .await
            .unwrap();
        let memory_step = recall(&loaded, "上一个回答的第1个命题是什么").unwrap();
        assert_eq!(
            step.text,
            format!("{}（SPARQL 召回）", memory_step.text),
            "SPARQL 召回必须与内存召回同口径：{} vs {}",
            step.text,
            memory_step.text
        );

        // 越界：说清最近一条回答共几点，而不是拿全局命题列表顶数。
        let missing = recall_sparql(&manager, &loaded, "上一个回答的第3个命题是什么")
            .await
            .unwrap();
        assert!(
            missing.text.contains("一共 1 点，没有第 3 点"),
            "{}",
            missing.text
        );

        // 更正关系：SPARQL 召回与内存侧同口径（未标记更正时如实说无）。
        let no_corrections = recall_sparql(&manager, &loaded, "上次和这次冲突吗")
            .await
            .unwrap();
        assert!(
            no_corrections.text.contains("还没有记录更正关系")
                && no_corrections.text.contains("SPARQL 确认"),
            "{}",
            no_corrections.text
        );

        // 当前焦点：SPARQL 召回最近一次焦点（第 3 轮「姚明」）。
        let focus = recall_sparql(&manager, &loaded, "当前焦点是谁")
            .await
            .unwrap();
        let memory_focus = recall(&loaded, "当前焦点是谁").unwrap();
        assert_eq!(
            focus.text,
            format!("{}（SPARQL 确认）", memory_focus.text),
            "{} vs {}",
            focus.text,
            memory_focus.text
        );

        // 标记更正后：SPARQL 召回列出更正关系，与内存侧逐字一致。
        let mut corrected_memory = loaded.clone();
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
        let corrected_loaded = manager
            .load_generation_memory(&corrected_memory.session)
            .await
            .unwrap();
        let corrections_sparql_reply =
            recall_sparql(&manager, &corrected_loaded, "上次和这次矛盾吗")
                .await
                .unwrap();
        let corrections_memory_reply = recall(&corrected_loaded, "上次和这次矛盾吗").unwrap();
        assert_eq!(
            corrections_sparql_reply.text,
            format!("{}（SPARQL 确认）", corrections_memory_reply.text),
            "{} vs {}",
            corrections_sparql_reply.text,
            corrections_memory_reply.text
        );
        assert!(corrections_sparql_reply
            .text
            .contains("共记录 1 条更正关系"));

        // 跨会话隔离：另一会话同词面不串入本会话的 SPARQL 召回。
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
            .append_reply("姚明是足球运动员。", &[], &[], &[])
            .unwrap();
        let other_memory = GenerationMemory::from_chronicle(&other_session, &other_chronicle, 0);
        manager.sync_generation_memory(&other_memory).await.unwrap();
        let other_loaded = manager
            .load_generation_memory(other_session.id())
            .await
            .unwrap();

        // 本会话查询「姚明是足球运动员」应仍为未说过（其他会话的不串入）。
        let scoped = recall_sparql(&manager, &loaded, "说过吗：姚明是足球运动员")
            .await
            .unwrap();
        assert!(
            scoped.text.contains("没有说过"),
            "本会话 SPARQL 召回不得命中其它会话：{}",
            scoped.text
        );

        // 其它会话自己的查询应命中。
        let other_said = recall_sparql(&manager, &other_loaded, "说过吗：姚明是足球运动员")
            .await
            .unwrap();
        assert!(
            other_said.text.contains("这句话说过"),
            "{}",
            other_said.text
        );
    }

    #[test]
    fn normalization_ignores_punctuation_and_case() {
        let dir = TempDir::new().unwrap();
        let memory = sample_memory(&dir);
        assert!(answer_key("姚明是篮球运动员。") == answer_key("姚明是篮球运动员"));
        assert!(memory.already_said("姚明是篮球运动员。"));
        assert!(recall(&memory, "说过吗：姚明是篮球运动员！").is_some());
        assert!(!memory.already_said("姚明是足球运动员"));
    }

    #[test]
    fn classify_detects_corrections_only_with_context() {
        assert_eq!(
            classify("上次说的和这次冲突吗"),
            Some(RecallIntent::Corrections)
        );
        assert_eq!(
            classify("之前和上一轮矛盾"),
            Some(RecallIntent::Corrections)
        );
        assert_eq!(
            classify("上一条和刚才不一致"),
            Some(RecallIntent::Corrections)
        );
        // 单纯回顾不应被当成更正。
        assert_eq!(classify("刚才聊了什么"), Some(RecallIntent::Recap));
        // 无上下文的冲突词不归入更正，交给常规管线。
        assert_eq!(classify("这两个说法冲突"), None);
        assert_eq!(classify("姚明是不是篮球运动员"), None);
    }

    #[test]
    fn corrections_recall_lists_relations_and_honest_empty() {
        let empty = GenerationMemory::empty("s");
        let r = recall(&empty, "上次和这次冲突吗").unwrap();
        assert_eq!(r.intent.tag(), "corrections");
        assert!(r.text.contains("还没有记录更正关系"));

        let dir = TempDir::new().unwrap();
        let mut memory = sample_memory(&dir);
        let corrected = memory.reply_iter().next().unwrap().claims[0].uri.clone();
        let latest_seq = memory.latest_reply().unwrap().seq;
        for stored in memory.turns.values_mut() {
            for reply in stored.replies.iter_mut() {
                if reply.seq == latest_seq {
                    reply.corrects = vec![corrected.clone()];
                }
            }
        }
        let r = recall(&memory, "上次和这一次矛盾吗").unwrap();
        assert_eq!(r.intent.tag(), "corrections");
        assert!(r.text.contains("共记录 1 条更正关系"));
        assert!(r.text.contains("撤回了先前命题"));
    }
}

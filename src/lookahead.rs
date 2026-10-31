//! ABF：澄清分支前瞻（Abductive Branch Futures）。
//!
//! 澄清会话每问一次，系统其实已经知道用户下一句最可能属于哪几种短答：
//! 确认、否定、暂缓、批量确认/否定，必要时还有更正。每种短答之后的
//! 失败前沿、解释空间与下一问都可以在**不写会话状态**的探针引擎上先算好，
//! 作为下一轮回答的派生证据包。
//!
//! 本模块只做三件事：
//!
//! - [`BranchKey`]：把用户短答归一成确定的分支键（同义短答命中同一键）；
//! - [`BranchSeal`]：给证据包盖封印（知识版本、引擎摘要、目标、承诺、问题队列）；
//! - [`LookaheadBoard`]：有界、确定性、按封印整体失效的前瞻缓存。
//!
//! 纪律：前瞻不是会话真相。命中只复用派生证据包，回答仍由当前真实状态生成；
//! 封印不符、引擎摘要不符或任何不确定一律同步回退到正常路径。前瞻不写图谱、
//! 不确认假设、不进入会话快照，知识重建后一律视为空。

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::abduction::{ClarificationReply, ClarificationSession};
use crate::resolution_engine::{Atom, Clause, ResolutionEngine};

/// 默认前瞻缓存容量（一个澄清问题最多几条短答分支）。
pub const DEFAULT_CAPACITY: usize = 8;
/// 默认预计算的分支数：先覆盖最常见的「是 / 不是」。
pub const DEFAULT_BRANCHES: usize = 2;
/// 单个问题最多预计算的分支数。
pub const MAX_BRANCHES: usize = 5;

/// 用户对当前澄清问题的确定性分支。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BranchKey {
    /// 确认当前假设。
    Confirm(String),
    /// 否定当前假设。
    Reject(String),
    /// 暂缓当前假设。
    Defer(String),
    /// 确认当前问题与队列中的全部假设。
    ConfirmAll(Vec<String>),
    /// 否定当前问题与队列中的全部假设。
    RejectAll(Vec<String>),
    /// 从当前假设更正到另一条假设。
    Correct { rejected: String, confirmed: String },
}

impl BranchKey {
    /// 从澄清回答结论构造分支键；无法归一成确定分支的回答返回 `None`。
    pub fn from_reply(reply: &ClarificationReply) -> Option<Self> {
        match reply {
            ClarificationReply::Confirmed(hypothesis) => {
                Some(Self::Confirm(hypothesis.atom.display()))
            }
            ClarificationReply::Rejected(hypothesis) => {
                Some(Self::Reject(hypothesis.atom.display()))
            }
            ClarificationReply::Deferred(hypothesis) => {
                Some(Self::Defer(hypothesis.atom.display()))
            }
            ClarificationReply::Corrected {
                rejected,
                confirmed,
            } => Some(Self::Correct {
                rejected: rejected.atom.display(),
                confirmed: confirmed.atom.display(),
            }),
            ClarificationReply::ConfirmedAll(batch) => Some(Self::ConfirmAll(
                batch.iter().map(|h| h.atom.display()).collect(),
            )),
            ClarificationReply::RejectedAll(batch) => Some(Self::RejectAll(
                batch.iter().map(|h| h.atom.display()).collect(),
            )),
            ClarificationReply::Unrecognized | ClarificationReply::Finished => None,
        }
    }

    /// 稳定 ASCII 标识（台账/测试用）。
    pub fn code(&self) -> &'static str {
        match self {
            Self::Confirm(_) => "confirm",
            Self::Reject(_) => "reject",
            Self::Defer(_) => "defer",
            Self::ConfirmAll(_) => "confirm_all",
            Self::RejectAll(_) => "reject_all",
            Self::Correct { .. } => "correct",
        }
    }

    /// 分支涉及的假设数（更正为 2）。
    pub fn width(&self) -> usize {
        match self {
            Self::Confirm(_) | Self::Reject(_) | Self::Defer(_) => 1,
            Self::ConfirmAll(items) | Self::RejectAll(items) => items.len(),
            Self::Correct { .. } => 2,
        }
    }
}

/// 与真实回答路径使用同一套分类器做只读预演：返回该输入会落到哪个分支。
pub fn preview_key(session: &ClarificationSession, text: &str) -> Option<BranchKey> {
    let mut probe = session.clone();
    BranchKey::from_reply(&probe.respond(text))
}

/// 为当前问题生成待预计算的规范短答。
///
/// 返回 `(分支键, 规范短答)`；调用方用规范短答在探针会话上重放，
/// 因此前瞻证据与真实回答共享完全相同的分类与推进逻辑。
pub fn seed_branch_answers(
    session: &ClarificationSession,
    limit: usize,
) -> Vec<(BranchKey, &'static str)> {
    let Some(current) = session.current() else {
        return Vec::new();
    };
    let current_atom = current.atom.display();
    let mut seeds = vec![
        (BranchKey::Confirm(current_atom.clone()), "是"),
        (BranchKey::Reject(current_atom), "不是"),
        (BranchKey::Defer(current.atom.display()), "不知道"),
    ];
    if !session.queue().is_empty() {
        let mut batch = Vec::with_capacity(session.remaining());
        if let Some(current) = session.current() {
            batch.push(current.atom.display());
        }
        batch.extend(session.queue().iter().map(|h| h.atom.display()));
        seeds.push((BranchKey::ConfirmAll(batch.clone()), "全部确认"));
        seeds.push((BranchKey::RejectAll(batch), "全部否定"));
    }
    seeds.truncate(limit.min(MAX_BRANCHES));
    seeds
}

/// 分支证据包的封印：任何一项变化都不得复用旧证据。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BranchSeal {
    /// 知识版本（观察事实确有变化时自增）。
    pub revision: u64,
    /// 已处理澄清轮数（回答推进后必须换封印）。
    pub turn: usize,
    /// 当前焦点实体。
    pub focus: String,
    /// 归结引擎状态摘要（事实 + 规则，排序后稳定）。
    pub engine: u64,
    /// 溯因目标与查询变量。
    pub goals: u64,
    /// 待定假设池。
    pub pending: u64,
    /// 用户已确认/否定/暂缓的承诺。
    pub commitments: u64,
    /// 当前问题原子。
    pub current: String,
    /// 尚未提问的队列（保序）。
    pub queue: u64,
    /// 当前解释空间（前沿 + 场景，含是否有界）。
    pub space: u64,
}

impl BranchSeal {
    #[allow(clippy::too_many_arguments)]
    pub fn compute(
        revision: u64,
        turn: usize,
        focus: &str,
        engine: u64,
        goals: &[String],
        pending: &[String],
        commitments: &[String],
        current: Option<&str>,
        queue: &[String],
        space: &[String],
    ) -> Self {
        Self {
            revision,
            turn,
            focus: focus.to_string(),
            engine,
            goals: digest_list(goals),
            pending: digest_list(pending),
            commitments: digest_list(commitments),
            current: current.unwrap_or("").to_string(),
            queue: digest_list(queue),
            space: digest_list(space),
        }
    }
}

/// 前瞻证据包：一条分支被重放后的派生状态与引擎摘要。
#[derive(Debug, Clone)]
pub struct BranchEvidence {
    /// 探针会话（已重放回答并完成重排；不写真实会话）。
    pub session: ClarificationSession,
    /// 重放后、重排前的会话承诺摘要；真实路径必须逐位一致才可复用。
    pub state: u64,
    /// 重放后探针引擎的摘要；真实路径必须逐位一致才可复用。
    pub engine: u64,
    /// 重放后的解释空间大小。
    pub explanation_count: usize,
    /// 解释空间是否因上限被截断。
    pub explanation_bounded: bool,
    /// 重放后的下一问原子（无则 `None`）。
    pub next_question: Option<String>,
    /// 重放后的剩余问题数。
    pub remaining: usize,
    /// 重放后原目标是否已可证。
    pub goal_proved: bool,
}

impl BranchEvidence {
    pub fn new(session: ClarificationSession, engine: u64, goal_proved: bool) -> Self {
        let state = session_state_digest(&session);
        Self::with_state(session, engine, state, goal_proved)
    }

    pub fn with_state(
        session: ClarificationSession,
        engine: u64,
        state: u64,
        goal_proved: bool,
    ) -> Self {
        let (explanation_count, explanation_bounded) = session
            .explanation_space()
            .map(|space| (space.len(), space.is_bounded()))
            .unwrap_or((0, false));
        Self {
            next_question: session.current().map(|h| h.atom.display()),
            remaining: session.remaining(),
            session,
            state,
            engine,
            explanation_count,
            explanation_bounded,
            goal_proved,
        }
    }
}

/// 有界、确定性、按封印整体失效的前瞻缓存。
#[derive(Debug, Clone)]
pub struct LookaheadBoard {
    capacity: usize,
    seal: Option<BranchSeal>,
    entries: BTreeMap<BranchKey, BranchEvidence>,
    order: Vec<BranchKey>,
}

impl Default for LookaheadBoard {
    fn default() -> Self {
        Self::new(DEFAULT_CAPACITY)
    }
}

impl LookaheadBoard {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            seal: None,
            entries: BTreeMap::new(),
            order: Vec::new(),
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn seal(&self) -> Option<&BranchSeal> {
        self.seal.as_ref()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// 只读命中：封印完全一致且键存在才返回；命中会推进确定性的 LRU 序。
    pub fn lookup(&mut self, seal: &BranchSeal, key: &BranchKey) -> Option<&BranchEvidence> {
        if self.seal.as_ref() != Some(seal) || !self.entries.contains_key(key) {
            return None;
        }
        self.order.retain(|cached| cached != key);
        self.order.push(key.clone());
        self.entries.get(key)
    }

    /// 取走一条证据：命中即从板上删除，避免同一证据被重复消费。
    pub fn take(&mut self, seal: &BranchSeal, key: &BranchKey) -> Option<BranchEvidence> {
        if self.seal.as_ref() != Some(seal) {
            return None;
        }
        self.order.retain(|cached| cached != key);
        self.entries.remove(key)
    }

    /// 写入一条证据；封印变化时整体清空，容量溢出淘汰最久未用者。
    ///
    /// 返回被淘汰的条目数（0 或 1）。
    pub fn insert(&mut self, seal: &BranchSeal, key: BranchKey, evidence: BranchEvidence) -> usize {
        if self.seal.as_ref() != Some(seal) {
            self.entries.clear();
            self.order.clear();
            self.seal = Some(seal.clone());
        }
        self.entries.insert(key.clone(), evidence);
        self.order.retain(|cached| cached != &key);
        self.order.push(key);
        if self.entries.len() > self.capacity {
            if let Some(oldest) = self.order.first().cloned() {
                self.order.remove(0);
                self.entries.remove(&oldest);
                return 1;
            }
        }
        0
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.order.clear();
        self.seal = None;
    }
}

/// 会话承诺摘要：回答推进后、重排之前的可观测状态。
///
/// 真实路径重放同一回答后必须与此摘要一致，才能采用分支证据。
pub fn session_state_digest(session: &ClarificationSession) -> u64 {
    let mut parts = vec![
        format!("asked:{}", session.questions_asked()),
        format!("remaining:{}", session.remaining()),
        format!(
            "current:{}",
            session
                .current()
                .map(|h| h.atom.display())
                .unwrap_or_default()
        ),
    ];
    parts.push("queue".to_string());
    parts.extend(session.queue().iter().map(|h| h.atom.display()));
    parts.push("confirmed".to_string());
    parts.extend(session.confirmed.iter().map(|h| h.atom.display()));
    parts.push("rejected".to_string());
    parts.extend(session.rejected.iter().map(|h| h.atom.display()));
    parts.push("deferred".to_string());
    parts.extend(session.deferred.iter().map(|h| h.atom.display()));
    digest_list(&parts)
}

/// 归结引擎的确定性摘要：排序后的事实与规则，不依赖 `HashSet` 遍历序。
pub fn engine_digest(engine: &ResolutionEngine) -> u64 {
    let mut facts: Vec<String> = engine
        .all_facts()
        .map(|atom| format!("fact:{}", atom.display()))
        .collect();
    facts.sort();
    let mut clauses: Vec<Vec<String>> = engine
        .all_clauses()
        .iter()
        .map(clause_digest_parts)
        .collect();
    clauses.sort();
    let mut parts: Vec<String> = Vec::with_capacity(facts.len() + clauses.len() * 5 + 2);
    parts.push(format!("facts:{}", facts.len()));
    parts.extend(facts);
    parts.push(format!("clauses:{}", clauses.len()));
    for clause in clauses {
        parts.extend(clause);
    }
    digest_list(&parts)
}

/// 把一条子句拆成带标签的独立片段：每个片段都会被 [`digest_list`] 长度前缀，
/// 避免用户输入中的分隔符造成拼接歧义。
fn clause_digest_parts(clause: &Clause) -> Vec<String> {
    let mut parts = Vec::with_capacity(clause.body.len() + 4);
    parts.push(format!("head:{}", clause.head.display()));
    parts.push(format!("body:{}", clause.body.len()));
    for body in &clause.body {
        parts.push(format!("goal:{}", body.display()));
    }
    parts.push(format!("confidence:{}", clause.confidence.to_bits()));
    parts.push(format!("source:{}", clause.source));
    parts
}

/// 长度前缀的稳定 FNV-1a 摘要：同输入同输出，不依赖随机种子。
pub fn digest_list(items: &[String]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for item in items {
        hash ^= item.len() as u64;
        hash = hash.wrapping_mul(0x100_0000_01b3);
        for byte in item.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abduction::{Hypothesis, HypothesisKind};

    fn hypothesis(atom: Atom) -> Hypothesis {
        Hypothesis {
            goal: atom.clone(),
            atom,
            kind: HypothesisKind::MissingFact,
            confidence: 0.5,
            reasons: Vec::new(),
            conflicts: Vec::new(),
            contradicts_known: false,
        }
    }

    fn session() -> ClarificationSession {
        ClarificationSession::start(&[
            hypothesis(Atom::rel("师傅", "张三", "李四")),
            hypothesis(Atom::rel("师傅", "李四", "王五")),
        ])
    }

    fn seal(seed: u64) -> BranchSeal {
        BranchSeal::compute(
            seed,
            0,
            "张三",
            seed,
            &["师祖(张三, ?X)".to_string()],
            &["师傅(张三, 李四)".to_string()],
            &[],
            Some("师傅(张三, 李四)"),
            &["师傅(李四, 王五)".to_string()],
            &["frontier".to_string()],
        )
    }

    #[test]
    fn preview_normalizes_synonyms_to_the_same_branch() {
        let session = session();
        let current = session.current().expect("应有当前问题").atom.display();
        for text in ["是", "是的", "对", "没错"] {
            assert_eq!(
                preview_key(&session, text),
                Some(BranchKey::Confirm(current.clone())),
                "短答「{}」应归一到确认分支",
                text
            );
        }
        for text in ["不是", "不", "否", "不对"] {
            assert_eq!(
                preview_key(&session, text),
                Some(BranchKey::Reject(current.clone())),
                "短答「{}」应归一到否定分支",
                text
            );
        }
        for text in ["不知道", "不确定", "不清楚"] {
            assert_eq!(
                preview_key(&session, text),
                Some(BranchKey::Defer(current.clone())),
                "短答「{}」应归一到暂缓分支",
                text
            );
        }
    }

    #[test]
    fn preview_keeps_batch_branches_distinct_and_ordered() {
        let session = session();
        let current = session.current().unwrap().atom.display();
        let queue: Vec<String> = session.queue().iter().map(|h| h.atom.display()).collect();
        let mut batch = vec![current];
        batch.extend(queue.clone());
        assert_eq!(
            preview_key(&session, "全部确认"),
            Some(BranchKey::ConfirmAll(batch.clone()))
        );
        assert_eq!(
            preview_key(&session, "全部否定"),
            Some(BranchKey::RejectAll(batch))
        );
    }

    #[test]
    fn seeds_cover_common_branches_and_respect_limit() {
        let session = session();
        let seeds = seed_branch_answers(&session, MAX_BRANCHES);
        assert_eq!(seeds.len(), 5);
        assert_eq!(seeds[0].0.code(), "confirm");
        assert_eq!(seeds[1].0.code(), "reject");
        assert_eq!(seeds[2].0.code(), "defer");
        assert_eq!(seeds[3].0.code(), "confirm_all");
        assert_eq!(seeds[4].0.code(), "reject_all");
        assert_eq!(seed_branch_answers(&session, 2).len(), 2);
        assert!(seed_branch_answers(&session, 0).is_empty());
    }

    #[test]
    fn board_hit_requires_exact_seal_and_evicts_deterministically() {
        let mut board = LookaheadBoard::new(2);
        let base = seal(1);
        let key = BranchKey::Confirm("师傅(张三, 李四)".to_string());
        let evidence = BranchEvidence::new(session(), 9, false);
        assert_eq!(board.insert(&base, key.clone(), evidence.clone()), 0);
        assert!(board.lookup(&base, &key).is_some());
        assert!(board.lookup(&seal(2), &key).is_none(), "不同封印不得命中");

        let other = BranchKey::Reject("师傅(张三, 李四)".to_string());
        let third = BranchKey::Defer("师傅(张三, 李四)".to_string());
        board.insert(&base, other.clone(), evidence.clone());
        assert_eq!(board.insert(&base, third.clone(), evidence), 1);
        assert!(
            board.lookup(&base, &key).is_none(),
            "容量溢出应淘汰最久未用条目"
        );
        assert!(board.lookup(&base, &other).is_some());
        assert!(board.lookup(&base, &third).is_some());
    }

    #[test]
    fn new_seal_invalidates_the_whole_board() {
        let mut board = LookaheadBoard::new(4);
        let key = BranchKey::Confirm("a".to_string());
        board.insert(
            &seal(1),
            key.clone(),
            BranchEvidence::new(session(), 1, false),
        );
        assert!(board.lookup(&seal(2), &key).is_none());
        board.insert(
            &seal(2),
            BranchKey::Reject("b".to_string()),
            BranchEvidence::new(session(), 2, false),
        );
        assert_eq!(board.len(), 1, "新封印写入应整体清空旧条目");
        assert!(board.lookup(&seal(2), &key).is_none());
    }

    #[test]
    fn digest_is_order_sensitive_and_engine_digest_tracks_facts() {
        assert_ne!(
            digest_list(&["a".to_string(), "b".to_string()]),
            digest_list(&["b".to_string(), "a".to_string()])
        );
        assert_ne!(
            digest_list(&["ab".to_string()]),
            digest_list(&["a".to_string(), "b".to_string()]),
            "长度前缀必须避免拼接歧义"
        );

        let mut engine = ResolutionEngine::new(4, 4);
        let empty = engine_digest(&engine);
        engine.add_fact("师傅", "张三", "李四");
        assert_ne!(engine_digest(&engine), empty);
        let with_fact = engine_digest(&engine);
        engine.add_fact("师傅", "张三", "李四");
        assert_eq!(engine_digest(&engine), with_fact, "幂等写入不改变摘要");
        engine.add_clause(Clause::new(
            Atom::pattern(
                "师祖",
                crate::resolution_engine::Term::var("X"),
                crate::resolution_engine::Term::var("Z"),
            ),
            vec![Atom::pattern(
                "师傅",
                crate::resolution_engine::Term::var("X"),
                crate::resolution_engine::Term::var("Z"),
            )],
            0.9,
            "测试",
        ));
        assert_ne!(engine_digest(&engine), with_fact, "规则变化必须换摘要");
    }
}

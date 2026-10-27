//! WKM：有界注意工作记忆（Bounded Attentional Working Memory）。
//!
//! 多轮系统里「最近 N 轮」只是时间窗口，不是注意力：当前焦点、未完成的问题、
//! 用户刚确认的事实、待澄清的指代，可能都比上一句寒暄更重要。本模块把「这一
//! 轮该想起什么」变成一次可解释的确定性选择：
//!
//! - [`AttentionItem`]：只保存对既有状态的引用（id、类型、文本、实体、谓词、
//!   可信度、逻辑新近性、是否悬而未决），不复制第二份事实正文；
//! - [`WorkingMemory`]：有界、按 id 去重、单写者；插入超过容量时淘汰最旧项；
//! - [`AttentionQuery`]：当前焦点、谓词与目标依赖；
//! - [`select_attention`]：整数显著性打分（显式提及、焦点、目标依赖、谓词/
//!   类型匹配、悬问、可信度、新近性），同分按新近性与稳定 id 决出；
//! - [`ContextPacket`]：选中的 `[M#]` 引用、丢弃原因、字符用量与稳定摘要。
//!
//! 纪律：纯函数选择、无浮点、无随机；显示文本不改写；实体比较走
//! [`crate::text_identity::TextIdentity`] 的身份键，零宽/全角噪声不会造成
//! 「看起来一样、哈希不同」的幽灵。工作记忆是索引，不是第二份真相源。

use serde::{Deserialize, Serialize};

use crate::context_image::fnv1a_hex;
use crate::text_identity::TextIdentity;

/// 工作记忆格式版本。
pub const WORKING_MEMORY_VERSION: u32 = 1;
/// 默认工作记忆容量（槽位，不是字符数）。
pub const WORKING_MEMORY_CAPACITY: usize = 128;
/// 单项文本保留上限（字符）；超出部分不影响实体/谓词匹配。
pub const ITEM_TEXT_LIMIT: usize = 512;
/// 包内引用标记上限（`M1`..`M99`）。
pub const MAX_PACKET_ITEMS: usize = 99;

/// 工作记忆条目的类型（封闭枚举）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionKind {
    /// 一轮对话。
    Turn,
    /// 当前焦点实体。
    Focus,
    /// 未完成/待回答的问题。
    OpenQuestion,
    /// 用户确认或更正的会话事实。
    SessionFact,
    /// 检索证据或出处。
    Evidence,
    /// 用户身份自述。
    Identity,
}

impl AttentionKind {
    pub fn code(self) -> &'static str {
        match self {
            AttentionKind::Turn => "turn",
            AttentionKind::Focus => "focus",
            AttentionKind::OpenQuestion => "open_question",
            AttentionKind::SessionFact => "session_fact",
            AttentionKind::Evidence => "evidence",
            AttentionKind::Identity => "identity",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            AttentionKind::Turn => "对话轮",
            AttentionKind::Focus => "焦点",
            AttentionKind::OpenQuestion => "未完成问题",
            AttentionKind::SessionFact => "会话事实",
            AttentionKind::Evidence => "证据",
            AttentionKind::Identity => "身份",
        }
    }

    fn priority(self) -> i64 {
        match self {
            AttentionKind::Focus => 12,
            AttentionKind::OpenQuestion => 10,
            AttentionKind::SessionFact => 8,
            AttentionKind::Identity => 7,
            AttentionKind::Evidence => 5,
            AttentionKind::Turn => 3,
        }
    }
}

/// 一条工作记忆索引。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttentionItem {
    /// 稳定 id（`turn:3`、`focus:姚明`、`open:2`…）；同 id 插入即更新。
    pub id: String,
    pub kind: AttentionKind,
    /// 供回显的原文（有界）；工作记忆不据此改写任何状态。
    pub text: String,
    #[serde(default)]
    pub entities: Vec<String>,
    #[serde(default)]
    pub predicates: Vec<String>,
    /// 可信度 0..=100（用户确认 100，普通回答 80，悬问 50）。
    pub trust: u8,
    /// 逻辑新近性（轮号或单调序号），不使用挂钟。
    pub recency: u64,
    /// 是否悬而未决（未回答的问题、待澄清指代）。
    pub open: bool,
    /// 是否必须进入上下文包（当前焦点、正在追问的问题）。
    pub pinned: bool,
}

impl AttentionItem {
    pub fn new(id: impl Into<String>, kind: AttentionKind, text: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            kind,
            text: truncate_chars(&text.into(), ITEM_TEXT_LIMIT),
            entities: Vec::new(),
            predicates: Vec::new(),
            trust: 50,
            recency: 0,
            open: false,
            pinned: false,
        }
    }

    pub fn with_entities(
        mut self,
        entities: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.entities = entities.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_predicates(
        mut self,
        predicates: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.predicates = predicates.into_iter().map(Into::into).collect();
        self
    }

    pub fn trust(mut self, trust: u8) -> Self {
        self.trust = trust.min(100);
        self
    }

    pub fn recency(mut self, recency: u64) -> Self {
        self.recency = recency;
        self
    }

    pub fn open(mut self, open: bool) -> Self {
        self.open = open;
        self
    }

    pub fn pinned(mut self, pinned: bool) -> Self {
        self.pinned = pinned;
        self
    }

    pub fn char_count(&self) -> usize {
        self.text.chars().count()
    }

    fn identity_matches(&self, entity: &str) -> bool {
        let needle = TextIdentity::new(entity);
        if needle.is_empty() {
            return false;
        }
        self.entities
            .iter()
            .any(|candidate| TextIdentity::new(candidate).same_identity(&needle))
            || TextIdentity::new(&self.text).identity().contains(needle.identity())
    }

    fn predicate_matches(&self, predicate: &str) -> bool {
        let needle = TextIdentity::new(predicate);
        if needle.is_empty() {
            return false;
        }
        self.predicates
            .iter()
            .any(|candidate| TextIdentity::new(candidate).same_identity(&needle))
            || TextIdentity::new(&self.text).identity().contains(needle.identity())
    }
}

/// 当前轮的注意力查询（焦点 / 谓词 / 目标依赖）。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttentionQuery {
    #[serde(default)]
    pub focus: Vec<String>,
    #[serde(default)]
    pub predicates: Vec<String>,
    /// 当前目标依赖的实体或谓词（例如待补全的属性）。
    #[serde(default)]
    pub goals: Vec<String>,
}

impl AttentionQuery {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_focus(mut self, focus: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.focus = focus.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_predicates(
        mut self,
        predicates: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.predicates = predicates.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_goals(mut self, goals: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.goals = goals.into_iter().map(Into::into).collect();
        self
    }

    pub fn is_empty(&self) -> bool {
        self.focus.is_empty() && self.predicates.is_empty() && self.goals.is_empty()
    }
}

/// 上下文包预算：槽位与字符双上限；`min_recent` 用逻辑新近性保底。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttentionBudget {
    pub max_items: usize,
    pub max_chars: usize,
    pub min_recent: usize,
}

impl Default for AttentionBudget {
    fn default() -> Self {
        Self {
            max_items: 12,
            max_chars: 800,
            min_recent: 1,
        }
    }
}

impl AttentionBudget {
    pub fn new(max_items: usize, max_chars: usize, min_recent: usize) -> Self {
        Self {
            max_items: max_items.min(MAX_PACKET_ITEMS),
            max_chars,
            min_recent: min_recent.min(max_items),
        }
    }

    pub fn disabled() -> Self {
        Self {
            max_items: 0,
            max_chars: 0,
            min_recent: 0,
        }
    }
}

/// 打分理由（可审计；按固定顺序去重）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionReason {
    Pinned,
    FocusMatch,
    PredicateMatch,
    GoalMatch,
    OpenQuestion,
    Trusted,
    Recency,
    KindPriority,
}

impl AttentionReason {
    pub fn label(self) -> &'static str {
        match self {
            AttentionReason::Pinned => "必须保留",
            AttentionReason::FocusMatch => "焦点匹配",
            AttentionReason::PredicateMatch => "谓词匹配",
            AttentionReason::GoalMatch => "目标依赖",
            AttentionReason::OpenQuestion => "悬而未决",
            AttentionReason::Trusted => "高可信",
            AttentionReason::Recency => "较新",
            AttentionReason::KindPriority => "类型优先级",
        }
    }
}

/// 包内一条被选中的记忆。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttendedItem {
    /// 稳定引用标记：`M1`、`M2`……
    pub marker: String,
    pub id: String,
    pub kind: AttentionKind,
    pub text: String,
    #[serde(default)]
    pub entities: Vec<String>,
    #[serde(default)]
    pub predicates: Vec<String>,
    pub score: i64,
    pub reasons: Vec<AttentionReason>,
}

/// 一条被丢弃的记忆及原因。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DroppedItem {
    pub id: String,
    pub kind: AttentionKind,
    pub score: i64,
    pub reason: DropReason,
}

/// 丢弃原因闭集。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DropReason {
    /// 槽位预算已满。
    ItemBudget,
    /// 字符预算不足以容纳。
    CharBudget,
    /// 预算为 0（显式关闭）。
    Disabled,
}

impl DropReason {
    pub fn label(self) -> &'static str {
        match self {
            DropReason::ItemBudget => "槽位预算",
            DropReason::CharBudget => "字符预算",
            DropReason::Disabled => "工作记忆关闭",
        }
    }
}

/// 一次注意力选择的完整结果（可直接进入快照/回执）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextPacket {
    pub version: u32,
    pub items: Vec<AttendedItem>,
    pub dropped: Vec<DroppedItem>,
    pub considered: usize,
    pub retained_chars: usize,
    /// 是否因为必须保留的条目而超出预算。
    pub over_budget: bool,
    /// 覆盖选中条目（marker/id/score）的稳定摘要。
    pub digest: String,
}

impl ContextPacket {
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// 按 `[M#]` 标记查找。
    pub fn item(&self, marker: &str) -> Option<&AttendedItem> {
        self.items.iter().find(|item| item.marker == marker)
    }

    pub fn references(&self) -> impl Iterator<Item = &AttendedItem> {
        self.items.iter()
    }

    pub fn dropped_reason(&self, id: &str) -> Option<DropReason> {
        self.dropped
            .iter()
            .find(|item| item.id == id)
            .map(|item| item.reason)
    }

    pub fn render_receipt(&self, budget: AttentionBudget) -> String {
        let reasons = if self.dropped.is_empty() {
            "无丢弃".to_string()
        } else {
            let mut items = 0usize;
            let mut chars = 0usize;
            let mut disabled = 0usize;
            for dropped in &self.dropped {
                match dropped.reason {
                    DropReason::ItemBudget => items += 1,
                    DropReason::CharBudget => chars += 1,
                    DropReason::Disabled => disabled += 1,
                }
            }
            format!(
                "丢弃：槽位预算 {}、字符预算 {}、工作记忆关闭 {}",
                items, chars, disabled
            )
        };
        format!(
            "工作记忆：保留 {}/{} 条，字符 {}/{}；{}；{}。",
            self.items.len(),
            self.considered,
            self.retained_chars,
            budget.max_chars,
            reasons,
            if self.over_budget {
                "必须保留条目已超出预算"
            } else {
                "未超预算"
            }
        )
    }
}

/// 有界工作记忆：单写者、按 id 更新、超出容量淘汰最旧项。
#[derive(Debug, Clone)]
pub struct WorkingMemory {
    capacity: usize,
    items: Vec<AttentionItem>,
    clock: u64,
}

impl Default for WorkingMemory {
    fn default() -> Self {
        Self::new(WORKING_MEMORY_CAPACITY)
    }
}

impl WorkingMemory {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            items: Vec::new(),
            clock: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn items(&self) -> &[AttentionItem] {
        &self.items
    }

    pub fn clock(&self) -> u64 {
        self.clock
    }

    /// 插入或更新同 id 条目；返回是否真正占用了槽位。
    pub fn push(&mut self, item: AttentionItem) -> bool {
        if self.capacity == 0 {
            return false;
        }
        if let Some(existing) = self.items.iter_mut().find(|entry| entry.id == item.id) {
            *existing = item;
            return true;
        }
        if self.items.len() >= self.capacity {
            let oldest = self
                .items
                .iter()
                .enumerate()
                .min_by(|(left_index, left), (right_index, right)| {
                    left.recency
                        .cmp(&right.recency)
                        .then_with(|| left_index.cmp(right_index))
                })
                .map(|(index, _)| index)
                .unwrap_or(0);
            self.items.remove(oldest);
        }
        self.clock = self.clock.saturating_add(1);
        let mut item = item;
        if item.recency == 0 {
            item.recency = self.clock;
        }
        self.items.push(item);
        true
    }

    pub fn extend(&mut self, items: impl IntoIterator<Item = AttentionItem>) {
        for item in items {
            self.push(item);
        }
    }

    pub fn remove(&mut self, id: &str) -> Option<AttentionItem> {
        let index = self.items.iter().position(|item| item.id == id)?;
        Some(self.items.remove(index))
    }

    pub fn clear(&mut self) {
        self.items.clear();
    }

    /// 按查询与预算选出上下文包。
    pub fn select(&self, query: &AttentionQuery, budget: AttentionBudget) -> ContextPacket {
        select_attention(&self.items, query, budget)
    }
}

/// 纯函数：从工作记忆条目中选出有界上下文包。
pub fn select_attention(
    items: &[AttentionItem],
    query: &AttentionQuery,
    budget: AttentionBudget,
) -> ContextPacket {
    let considered = items.len();
    if budget.max_items == 0 {
        return ContextPacket {
            version: WORKING_MEMORY_VERSION,
            items: Vec::new(),
            dropped: items
                .iter()
                .map(|item| DroppedItem {
                    id: item.id.clone(),
                    kind: item.kind,
                    score: 0,
                    reason: DropReason::Disabled,
                })
                .collect(),
            considered,
            retained_chars: 0,
            over_budget: false,
            digest: packet_digest(&[]),
        };
    }

    let max_recency = items.iter().map(|item| item.recency).max().unwrap_or(0);
    let mut scored: Vec<(usize, i64, Vec<AttentionReason>)> = items
        .iter()
        .enumerate()
        .map(|(index, item)| {
            let (score, mut reasons) =
                score_item(item, index, max_recency, considered, query);
            reasons.sort();
            reasons.dedup();
            (index, score, reasons)
        })
        .collect();
    scored.sort_by(|left, right| {
        right
            .1
            .cmp(&left.1)
            .then_with(|| items[right.0].recency.cmp(&items[left.0].recency))
            .then_with(|| items[left.0].id.cmp(&items[right.0].id))
    });

    let min_recent = budget.min_recent.min(considered).min(budget.max_items);
    let mut mandatory: Vec<usize> = scored
        .iter()
        .filter(|(index, _, _)| items[*index].pinned)
        .map(|(index, _, _)| *index)
        .collect();
    mandatory.sort_by(|left, right| {
        items[*right]
            .recency
            .cmp(&items[*left].recency)
            .then_with(|| items[*left].id.cmp(&items[*right].id))
    });
    mandatory.truncate(budget.max_items);
    for (index, _, _) in scored.iter().take(min_recent) {
        if !mandatory.contains(index) {
            mandatory.push(*index);
        }
    }

    let mut selected: Vec<usize> = Vec::new();
    let mut retained_chars = 0usize;
    let mut over_budget = false;
    for index in mandatory {
        if selected.contains(&index) {
            continue;
        }
        selected.push(index);
        retained_chars = retained_chars.saturating_add(items[index].char_count());
    }
    if retained_chars > budget.max_chars {
        over_budget = true;
    }

    for (index, _, _) in &scored {
        if selected.len() >= budget.max_items {
            break;
        }
        if selected.contains(index) {
            continue;
        }
        let chars = items[*index].char_count();
        if retained_chars.saturating_add(chars) > budget.max_chars {
            continue;
        }
        selected.push(*index);
        retained_chars = retained_chars.saturating_add(chars);
    }

    selected.sort_unstable();
    let selected_set = &selected;
    let mut packet_items = Vec::with_capacity(selected.len());
    for (position, index) in selected.iter().enumerate() {
        let item = &items[*index];
        let (score, reasons) = scored
            .iter()
            .find(|(candidate, _, _)| candidate == index)
            .map(|(_, score, reasons)| (*score, reasons.clone()))
            .unwrap_or((0, Vec::new()));
        packet_items.push(AttendedItem {
            marker: format!("M{}", position + 1),
            id: item.id.clone(),
            kind: item.kind,
            text: item.text.clone(),
            entities: item.entities.clone(),
            predicates: item.predicates.clone(),
            score,
            reasons,
        });
    }

    let dropped = items
        .iter()
        .enumerate()
        .filter(|(index, _)| !selected_set.contains(index))
        .map(|(index, item)| {
            let score = scored
                .iter()
                .find(|(candidate, _, _)| *candidate == index)
                .map(|(_, score, _)| *score)
                .unwrap_or(0);
            DroppedItem {
                id: item.id.clone(),
                kind: item.kind,
                score,
                reason: if selected.len() >= budget.max_items {
                    DropReason::ItemBudget
                } else {
                    DropReason::CharBudget
                },
            }
        })
        .collect();

    ContextPacket {
        version: WORKING_MEMORY_VERSION,
        digest: packet_digest(&packet_items),
        items: packet_items,
        dropped,
        considered,
        retained_chars,
        over_budget,
    }
}

fn score_item(
    item: &AttentionItem,
    index: usize,
    max_recency: u64,
    total: usize,
    query: &AttentionQuery,
) -> (i64, Vec<AttentionReason>) {
    let mut score = item.kind.priority();
    let mut reasons = vec![AttentionReason::KindPriority];
    if item.pinned {
        score += 100;
        reasons.push(AttentionReason::Pinned);
    }
    let mut focus_hits = 0i64;
    for focus in &query.focus {
        if item.identity_matches(focus) {
            focus_hits += 1;
        }
    }
    if focus_hits > 0 {
        score += 40 * focus_hits.min(4);
        reasons.push(AttentionReason::FocusMatch);
    }
    let mut predicate_hits = 0i64;
    for predicate in &query.predicates {
        if item.predicate_matches(predicate) {
            predicate_hits += 1;
        }
    }
    if predicate_hits > 0 {
        score += 24 * predicate_hits.min(4);
        reasons.push(AttentionReason::PredicateMatch);
    }
    let mut goal_hits = 0i64;
    for goal in &query.goals {
        if item.identity_matches(goal) || item.predicate_matches(goal) {
            goal_hits += 1;
        }
    }
    if goal_hits > 0 {
        score += 16 * goal_hits.min(4);
        reasons.push(AttentionReason::GoalMatch);
    }
    if item.open {
        score += 20;
        reasons.push(AttentionReason::OpenQuestion);
    }
    if item.trust >= 80 {
        score += i64::from(item.trust) / 10;
        reasons.push(AttentionReason::Trusted);
    }
    if max_recency > 0 {
        let recency = i64::try_from(item.recency).unwrap_or(i64::MAX);
        let max = i64::try_from(max_recency).unwrap_or(i64::MAX);
        let bonus = if max == 0 { 0 } else { (recency * 12) / max };
        if bonus > 0 {
            score += bonus;
            reasons.push(AttentionReason::Recency);
        }
    }
    if total > 1 && index + 1 == total {
        score += 1;
    }
    (score, reasons)
}

fn packet_digest(items: &[AttendedItem]) -> String {
    let value = serde_json::json!({
        "version": WORKING_MEMORY_VERSION,
        "items": items.iter().map(|item| serde_json::json!({
            "marker": item.marker,
            "id": item.id,
            "score": item.score,
        })).collect::<Vec<_>>(),
    });
    fnv1a_hex(crate::context_image::canonical_json(&value).as_bytes())
}

fn truncate_chars(text: &str, limit: usize) -> String {
    let mut output: String = text.chars().take(limit).collect();
    if text.chars().count() > limit {
        output.push('…');
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(id: &str, kind: AttentionKind, text: &str) -> AttentionItem {
        AttentionItem::new(id, kind, text)
    }

    #[test]
    fn focus_match_beats_recency_and_noise_is_normalized() {
        let items = vec![
            item("turn:1", AttentionKind::Turn, "姚明的身高是多少")
                .with_entities(["姚明"])
                .with_predicates(["身高"])
                .recency(1),
            item("turn:2", AttentionKind::Turn, "谢谢").recency(2),
            item("turn:3", AttentionKind::Turn, "今天天气怎么样").recency(3),
        ];
        let query = AttentionQuery::new().with_focus(["姚\u{200B}明"]);
        let packet = select_attention(&items, &query, AttentionBudget::new(2, 200, 1));
        assert_eq!(packet.items.len(), 2);
        assert_eq!(packet.items[0].id, "turn:1", "{}", packet.render_receipt(AttentionBudget::new(2, 200, 1)));
        assert!(packet.items[0]
            .reasons
            .contains(&AttentionReason::FocusMatch));
        assert_eq!(packet.dropped_reason("turn:2"), Some(DropReason::ItemBudget));
    }

    #[test]
    fn pinned_items_are_kept_even_when_over_budget() {
        let items = vec![
            item("focus:1", AttentionKind::Focus, "姚明")
                .with_entities(["姚明"])
                .pinned(true)
                .recency(9),
            item("turn:1", AttentionKind::Turn, "谢谢").recency(1),
        ];
        let packet = select_attention(
            &items,
            &AttentionQuery::new(),
            AttentionBudget::new(1, 100, 0),
        );
        assert_eq!(packet.items.len(), 1);
        assert_eq!(packet.items[0].id, "focus:1");
        assert_eq!(packet.dropped_reason("turn:1"), Some(DropReason::ItemBudget));
        assert!(!packet.over_budget);
    }

    #[test]
    fn char_budget_is_hard_and_reported() {
        let items = vec![
            item("turn:1", AttentionKind::Turn, "很长的旧问题很长的旧问题").recency(1),
            item("turn:2", AttentionKind::Turn, "好").recency(2),
        ];
        let packet = select_attention(
            &items,
            &AttentionQuery::new().with_focus(["不存在"]),
            AttentionBudget::new(2, 1, 0),
        );
        assert_eq!(packet.items.len(), 1);
        assert_eq!(packet.items[0].id, "turn:2");
        assert_eq!(packet.dropped_reason("turn:1"), Some(DropReason::CharBudget));
        assert!(!packet.over_budget);
        assert_eq!(packet.retained_chars, 1);
    }

    #[test]
    fn disabled_budget_drops_everything_explicitly() {
        let items = vec![item("turn:1", AttentionKind::Turn, "姚明")];
        let packet = select_attention(&items, &AttentionQuery::new(), AttentionBudget::disabled());
        assert!(packet.is_empty());
        assert_eq!(packet.dropped.len(), 1);
        assert_eq!(packet.dropped[0].reason, DropReason::Disabled);
        assert!(packet
            .render_receipt(AttentionBudget::disabled())
            .contains("工作记忆关闭"));
    }

    #[test]
    fn selection_is_deterministic_and_markers_are_stable() {
        let items: Vec<AttentionItem> = (1..=8)
            .map(|index| {
                item(
                    &format!("turn:{}", index),
                    AttentionKind::Turn,
                    &format!("问题{}", index),
                )
                .with_entities(["姚明"])
                .recency(index)
            })
            .collect();
        let query = AttentionQuery::new().with_focus(["姚明"]);
        let budget = AttentionBudget::new(4, 500, 1);
        let first = select_attention(&items, &query, budget);
        for _ in 0..16 {
            assert_eq!(first, select_attention(&items, &query, budget));
        }
        assert_eq!(
            first
                .items
                .iter()
                .map(|item| item.marker.as_str())
                .collect::<Vec<_>>(),
            vec!["M1", "M2", "M3", "M4"]
        );
        assert_eq!(first.digest.len(), 16);
    }

    #[test]
    fn working_memory_is_bounded_and_updates_by_id() {
        let mut memory = WorkingMemory::new(2);
        assert!(memory.push(item("a", AttentionKind::Turn, "旧")));
        assert!(memory.push(item("b", AttentionKind::Turn, "新").recency(2)));
        assert!(memory.push(
            item("a", AttentionKind::Turn, "更新后的旧")
                .recency(3)
                .trust(100)
        ));
        assert_eq!(memory.len(), 2);
        assert_eq!(
            memory
                .items()
                .iter()
                .find(|entry| entry.id == "a")
                .map(|entry| entry.text.as_str()),
            Some("更新后的旧")
        );
        assert!(memory.push(item("c", AttentionKind::Turn, "最新").recency(1)));
        assert_eq!(memory.len(), 2);
        assert!(memory.items().iter().any(|entry| entry.id == "c"));
        assert!(memory.remove("c").is_some());
        assert_eq!(memory.len(), 1);
    }

    #[test]
    fn zero_capacity_memory_never_panics() {
        let mut memory = WorkingMemory::new(0);
        assert!(!memory.push(item("a", AttentionKind::Turn, "x")));
        assert!(memory.is_empty());
        let packet = memory.select(&AttentionQuery::new(), AttentionBudget::default());
        assert!(packet.is_empty());
    }

    #[test]
    fn packet_roundtrips_through_json() {
        let items = vec![item("turn:1", AttentionKind::Turn, "姚明的身高")
            .with_entities(["姚明"])
            .with_predicates(["身高"])
            .trust(90)
            .recency(1)];
        let packet = select_attention(
            &items,
            &AttentionQuery::new().with_focus(["姚明"]),
            AttentionBudget::new(4, 100, 0),
        );
        let json = serde_json::to_string(&packet).unwrap();
        let restored: ContextPacket = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, packet);
    }
}

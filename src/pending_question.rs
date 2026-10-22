//! 悬问账本（Pending Question Ledger, PQL）。
//!
//! 对话里「没答上的问题」是一等公民：当一个多跳目标在中途证不下去时，
//! 系统记住的是 **目标 + 当前缺失的那条前提**，而不是一段「没有记录」的文本。
//! 用户下一轮补上这条前提后，系统重新证明原目标并给出延期回答。
//!
//! 设计边界：
//! - 只记录可确定生成的缺失原子（主语 + 谓词），不猜取值；
//! - 容量有界，可显式放弃；不设 TTL，忘记必须是显式动作；
//! - 纯符号、无网络、无模型调用；
//! - 空账本不落盘（旧快照字节保持不变）。

use serde::{Deserialize, Serialize};

use crate::function_words::contains_question_marker;

/// 同时挂起的悬问上限：超出后淘汰最早的一条，避免会话状态无限增长。
pub const MAX_PENDING_QUESTIONS: usize = 8;

/// 显式说「不知道」后搁置的悬问上限。
pub const MAX_SHELVED_QUESTIONS: usize = 8;

/// 裸回答允许的最大字符数（再长就不再当作短答，避免吞掉正常语句）。
pub const MAX_BARE_ANSWER_CHARS: usize = 32;

/// 一个逻辑洞：目标证明在中途卡住时缺失的那条原子。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PendingHole {
    /// 缺前提的主语（已绑定到具体实体）。
    pub subject: String,
    /// 谓词词面（用于追问与显示）。
    pub predicate: String,
    /// 规范化槽位（用于写入会话事实）。
    #[serde(default)]
    pub slot: String,
    /// 谓词是否为逆向词面（逆向时补充取值的存储方向要翻转）。
    #[serde(default)]
    pub inverted: bool,
    /// 期望取值类型（来自谓词画像值域；未知为空）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_type: Option<String>,
}

impl PendingHole {
    pub fn new(subject: &str, predicate: &str, slot: &str, inverted: bool) -> Self {
        Self {
            subject: subject.trim().to_string(),
            predicate: predicate.trim().to_string(),
            slot: slot.trim().to_string(),
            inverted,
            expected_type: None,
        }
    }

    /// 洞的中文短语：「李四的师傅」。
    pub fn phrase(&self) -> String {
        format!("{} 的 {}", self.subject, self.predicate)
    }

    /// 补全后的完整句子：「李四的师傅是王五」。
    pub fn fill_sentence(&self, object: &str) -> String {
        format!("{} 是 {}", self.phrase(), object.trim())
    }
}

/// 一条悬问：原始目标 + 缺失前提 + 已走通的前缀见证。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PendingQuestion {
    /// 会话内稳定编号，便于报告与显式放弃。
    pub id: u64,
    /// 用户原问句（延期回答时逐字重放）。
    pub question: String,
    /// 目标链的中文形式：「张三 的 师傅 的 师傅」。
    pub goal: String,
    /// 当前要补的那条前提。
    pub hole: PendingHole,
    /// 已证明的前缀边：「张三 的 师傅 是 李四」。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub support: Vec<String>,
    /// 已追问次数（用于报告，不用于自动过期）。
    #[serde(default)]
    pub asks: u32,
}

/// 悬问账本：有界、可持久化、可显式清空。
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct PendingQuestionLedger {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    questions: Vec<PendingQuestion>,
    #[serde(default, skip_serializing_if = "is_zero")]
    next_id: u64,
    /// 用户明确说「不知道」后搁置的悬问：不自动重开，但保留可重开。
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    shelved: Vec<PendingQuestion>,
    /// 最近一次真正追问过的悬问编号。
    ///
    /// 台账里可能同时挂着多条未答之问；裸回答必须绑定到「刚刚被问到的」
    /// 那一条，而不是最早的一条。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    active_id: Option<u64>,
    /// 上一轮是否刚把追问说出口。只有紧接的一轮才接受裸回答。
    #[serde(default, skip_serializing_if = "is_false")]
    awaiting_answer: bool,
}

fn is_zero(value: &u64) -> bool {
    *value == 0
}

fn is_false(value: &bool) -> bool {
    !*value
}

impl PendingQuestionLedger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.questions.is_empty() && self.shelved.is_empty()
    }

    pub fn len(&self) -> usize {
        self.questions.len()
    }

    pub fn shelved_len(&self) -> usize {
        self.shelved.len()
    }

    pub fn has_active(&self) -> bool {
        !self.questions.is_empty()
    }

    pub fn questions(&self) -> &[PendingQuestion] {
        &self.questions
    }

    pub fn shelved(&self) -> &[PendingQuestion] {
        &self.shelved
    }

    pub fn first(&self) -> Option<&PendingQuestion> {
        self.questions.first()
    }

    /// 当前被追问的悬问：优先取最近追问过的那条，否则退回最早的一条。
    pub fn active(&self) -> Option<&PendingQuestion> {
        self.active_id
            .and_then(|id| self.questions.iter().find(|pending| pending.id == id))
            .or_else(|| self.questions.first())
    }

    pub fn active_id(&self) -> Option<u64> {
        self.active().map(|pending| pending.id)
    }

    /// 上一轮是否刚追问过某个洞；只有此时裸取值才允许绑定。
    pub fn awaiting_answer(&self) -> bool {
        self.awaiting_answer && self.active().is_some()
    }

    /// 把某条悬问标为「刚刚问过」，下一轮短答将绑定到它。
    pub fn mark_asked(&mut self, id: u64) {
        if self.questions.iter().any(|pending| pending.id == id) {
            self.active_id = Some(id);
            self.awaiting_answer = true;
        }
    }

    /// 本轮没有再追问：关闭裸回答窗口，但保留悬问本身。
    pub fn clear_awaiting(&mut self) {
        self.awaiting_answer = false;
    }

    fn normalize_active(&mut self) {
        if self.questions.is_empty() {
            self.active_id = None;
            self.awaiting_answer = false;
            return;
        }
        let valid = self
            .active_id
            .is_some_and(|id| self.questions.iter().any(|pending| pending.id == id));
        if !valid {
            self.active_id = self.questions.first().map(|pending| pending.id);
            self.awaiting_answer = false;
        }
    }

    pub fn clear(&mut self) {
        self.questions.clear();
        self.shelved.clear();
        self.active_id = None;
        self.awaiting_answer = false;
    }

    /// 记录一条悬问。同一「目标 + 洞」重复失败时只累计追问次数，不重复入账。
    pub fn record(
        &mut self,
        question: &str,
        goal: &str,
        hole: PendingHole,
        support: Vec<String>,
    ) -> PendingQuestion {
        if let Some(existing) = self.questions.iter_mut().find(|pending| {
            pending.question == question
                && pending.hole.subject == hole.subject
                && pending.hole.predicate == hole.predicate
                && pending.hole.inverted == hole.inverted
        }) {
            existing.asks = existing.asks.saturating_add(1);
            // 支持边按当前知识重算：用户补过前提后，旧前缀可能已不是最新。
            if !support.is_empty() {
                existing.support = support;
            }
            let id = existing.id;
            self.active_id = Some(id);
            return existing.clone();
        }
        if self.questions.len() >= MAX_PENDING_QUESTIONS {
            self.questions.remove(0);
        }
        self.next_id = self.next_id.saturating_add(1);
        let pending = PendingQuestion {
            id: self.next_id,
            question: question.trim().to_string(),
            goal: goal.trim().to_string(),
            hole,
            support,
            asks: 1,
        };
        // 用户重新问起一条已搁置的悬问：以新的活动条目为准，不保留重复副本。
        self.shelved.retain(|shelved| {
            !(shelved.question == pending.question
                && shelved.hole.subject == pending.hole.subject
                && shelved.hole.predicate == pending.hole.predicate
                && shelved.hole.inverted == pending.hole.inverted)
        });
        self.active_id = Some(pending.id);
        self.questions.push(pending.clone());
        pending
    }

    pub fn remove(&mut self, id: u64) -> Option<PendingQuestion> {
        let position = self.questions.iter().position(|pending| pending.id == id)?;
        let removed = self.questions.remove(position);
        self.normalize_active();
        Some(removed)
    }

    /// 用户明确说「不知道」：移入搁置区，保留原目标与洞，等待显式重开。
    pub fn suspend(&mut self, id: u64) -> Option<PendingQuestion> {
        let removed = self.remove(id)?;
        self.shelved.push(removed.clone());
        if self.shelved.len() > MAX_SHELVED_QUESTIONS {
            let excess = self.shelved.len() - MAX_SHELVED_QUESTIONS;
            self.shelved.drain(0..excess);
        }
        Some(removed)
    }

    /// 显式重开全部搁置悬问，返回重开的条数。
    pub fn reopen_shelved(&mut self) -> usize {
        if self.shelved.is_empty() {
            return 0;
        }
        let count = self.shelved.len();
        let mut revived = std::mem::take(&mut self.shelved);
        for pending in &mut revived {
            pending.asks = pending.asks.saturating_add(1);
        }
        self.questions.append(&mut revived);
        if self.questions.len() > MAX_PENDING_QUESTIONS {
            let excess = self.questions.len() - MAX_PENDING_QUESTIONS;
            self.questions.drain(0..excess);
        }
        self.awaiting_answer = false;
        self.normalize_active();
        count
    }

    /// 按原问句移除悬问（活动区与搁置区都收），返回是否删除了条目。
    pub fn dismiss_question(&mut self, question: &str) -> bool {
        let question = question.trim();
        let before = self.questions.len() + self.shelved.len();
        self.questions.retain(|pending| pending.question != question);
        self.shelved.retain(|pending| pending.question != question);
        self.normalize_active();
        self.questions.len() + self.shelved.len() < before
    }

    pub fn remove_first(&mut self) -> Option<PendingQuestion> {
        if self.questions.is_empty() {
            None
        } else {
            let removed = self.questions.remove(0);
            self.normalize_active();
            Some(removed)
        }
    }

    /// 替换整个队列（恢复快照/批量重排时使用），保持编号单调不回退。
    pub fn replace(&mut self, questions: Vec<PendingQuestion>) {
        let max_id = questions.iter().map(|pending| pending.id).max().unwrap_or(0);
        self.next_id = self.next_id.max(max_id);
        self.questions = questions;
        if self.questions.len() > MAX_PENDING_QUESTIONS {
            let excess = self.questions.len() - MAX_PENDING_QUESTIONS;
            self.questions.drain(0..excess);
        }
        self.normalize_active();
    }

    /// 恢复快照时连同「当前追问」与「是否等待短答」一起恢复。
    pub fn restore_from(&mut self, saved: &Self) {
        let mut restored = saved.clone();
        if restored.questions.len() > MAX_PENDING_QUESTIONS {
            let excess = restored.questions.len() - MAX_PENDING_QUESTIONS;
            restored.questions.drain(0..excess);
        }
        if restored.shelved.len() > MAX_SHELVED_QUESTIONS {
            let excess = restored.shelved.len() - MAX_SHELVED_QUESTIONS;
            restored.shelved.drain(0..excess);
        }
        restored.next_id = saved
            .next_id
            .max(
                restored
                    .questions
                    .iter()
                    .chain(restored.shelved.iter())
                    .map(|pending| pending.id)
                    .max()
                    .unwrap_or(0),
            );
        restored.normalize_active();
        *self = restored;
    }

    /// 追问话术：把「缺哪条前提」说成一个用户可以短答的问题。
    pub fn render_ask(pending: &PendingQuestion) -> String {
        let mut output = String::new();
        if pending.support.is_empty() {
            output.push_str(&format!(
                "「{}」目前还没有足够的前提。\n",
                pending.question
            ));
        } else {
            output.push_str(&format!(
                "关于「{}」，我现在能证到：{}。\n",
                pending.question,
                pending.support.join("；")
            ));
        }
        output.push_str(&format!(
            "还缺一条前提：「{}」。\n",
            pending.hole.phrase()
        ));
        output.push_str(&format!(
            "{} 是谁？直接回复取值即可，我就能回答原来的问题。",
            pending.hole.phrase()
        ));
        if let Some(expected) = &pending.hole.expected_type {
            output.push_str(&format!("（期望类型：{}）", expected));
        }
        output
    }

    /// 台账报告：列出所有未答之问，供用户查看或显式放弃。
    pub fn render_report(&self) -> String {
        if self.is_empty() {
            return "悬问台账为空：当前没有欠用户答案的问题。".to_string();
        }
        let mut output = if self.questions.is_empty() {
            "悬问台账：当前没有正在追问的问题。\n".to_string()
        } else {
            format!("悬问台账：{} 条未答之问。\n", self.questions.len())
        };
        for pending in &self.questions {
            let marker = if self.active_id == Some(pending.id) {
                if self.awaiting_answer {
                    " ← 当前追问（可直接回短答）"
                } else {
                    " ← 上次追问"
                }
            } else {
                ""
            };
            output.push_str(&format!(
                "  {}. 「{}」缺「{}」（已追问 {} 次）{}\n",
                pending.id,
                pending.question,
                pending.hole.phrase(),
                pending.asks,
                marker
            ));
        }
        if !self.shelved.is_empty() {
            output.push_str(&format!("已搁置 {} 条（不自动重开）：\n", self.shelved.len()));
            for pending in &self.shelved {
                output.push_str(&format!(
                    "  {}. 「{}」缺「{}」\n",
                    pending.id,
                    pending.question,
                    pending.hole.phrase()
                ));
            }
            output.push_str("输入「重开悬问」可恢复搁置的问题。\n");
        }
        output.push_str("回复取值即可回填当前追问；输入「放弃悬问」清空台账。");
        output
    }
}

/// 用户对悬问的回应分类。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnswerBinding {
    /// 可以绑定到洞的取值。
    Fill(String),
    /// 用户明确表示不知道：显式搁置，不当作否定事实。
    Unknown,
    /// 不是对悬问的回答：让位给正常对话管线。
    NotAnAnswer,
}

/// 用户是否用了显式回填前缀（例如「回答悬问：王五」）。
///
/// 显式前缀不受「仅紧接一轮可裸答」的限制：用户可以在任何时候主动
/// 指名回填，而不必等系统再次追问。
pub fn has_explicit_answer_prefix(raw: &str) -> bool {
    let text = raw.trim();
    ["回答悬问：", "回答悬问:", "回填悬问：", "回填悬问:"]
        .iter()
        .any(|prefix| text.starts_with(prefix))
}

/// 把用户的一轮输入解释为对某个洞的回答。
///
/// 只在系统刚刚追问过这个洞时调用；宁可让位，也不劫持正常问句/命令。
pub fn interpret_answer(raw: &str, hole: &PendingHole) -> AnswerBinding {
    let mut text = raw.trim();
    for prefix in ["回答悬问：", "回答悬问:", "回填悬问：", "回填悬问:"] {
        if let Some(rest) = text.strip_prefix(prefix) {
            text = rest.trim();
            break;
        }
    }
    if text.is_empty() {
        return AnswerBinding::NotAnAnswer;
    }
    let normalized = text.trim_matches(|ch: char| ch.is_whitespace() || matches!(ch, '。' | '！' | '!'));
    if matches!(
        normalized,
        "不知道" | "不清楚" | "不确定" | "没有" | "没信息" | "不知道答案" | "放弃"
    ) {
        return AnswerBinding::Unknown;
    }
    if contains_question_marker(text) || text.contains('?') || text.contains('？') {
        return AnswerBinding::NotAnAnswer;
    }
    // 否定陈述不能被读成肯定回填：「李四不是王五的师傅」不是答案。
    if text.contains("不是") || text.contains("并非") || text.contains("没有") {
        return AnswerBinding::NotAnAnswer;
    }
    // 带前缀的命令式输入（「记住：…」「教学：…」）不是对悬问的短答。
    if text.contains('：') || text.contains(':') {
        return AnswerBinding::NotAnAnswer;
    }
    // 显式整句：「李四的师傅是王五」。
    if let Some((left, right)) = split_fact(text) {
        if left_matches_hole(left, hole) {
            let object = clean_value(right);
            if !object.is_empty() {
                return AnswerBinding::Fill(object);
            }
        }
        // 逆向洞也接受按存储方向陈述的事实：问「王五的徒弟」时，
        // 「赵六的师傅是王五」与「王五的徒弟是赵六」是同一回事。
        if let Some(object) = inverted_fact_subject(left, right, hole) {
            return AnswerBinding::Fill(object);
        }
        // 是完整陈述句但填的不是这个洞：让位，不把它当裸取值。
        return AnswerBinding::NotAnAnswer;
    }
    // 短答「是王五」/「叫王五」。
    let mut value = normalized;
    for prefix in ["就是", "叫做", "叫", "是"] {
        if let Some(rest) = value.strip_prefix(prefix) {
            value = rest.trim();
            break;
        }
    }
    for suffix in ["吧", "呀", "啊"] {
        if let Some(rest) = value.strip_suffix(suffix) {
            value = rest.trim_end();
        }
    }
    let value = value.trim();
    if value.is_empty()
        || value.chars().count() > MAX_BARE_ANSWER_CHARS
        || value.contains(['是', '为', '，', ',', '：', ':', '；', ';', '叫'])
        || value.contains(hole.subject.as_str())
        || (!hole.predicate.is_empty() && value.contains(hole.predicate.as_str()))
    {
        return AnswerBinding::NotAnAnswer;
    }
    AnswerBinding::Fill(value.to_string())
}

fn split_fact(text: &str) -> Option<(&str, &str)> {
    let (left, right) = text
        .rsplit_once('是')
        .or_else(|| text.rsplit_once('为'))?;
    if left.trim().is_empty() {
        return None;
    }
    Some((left.trim(), right.trim()))
}

/// 陈述句左侧是否恰好就是悬问的那个洞。
///
/// 用整句包含关系判断会把「李四的师傅的爸爸是王五」误当成
/// 「李四的师傅是王五」；这里要求去掉空白后逐字相等。
fn left_matches_hole(left: &str, hole: &PendingHole) -> bool {
    fn compact(text: &str) -> String {
        text.chars()
            .filter(|ch| !ch.is_whitespace() && !matches!(ch, '的'))
            .collect()
    }
    let left = compact(left);
    let subject = compact(&hole.subject);
    if left.is_empty() || subject.is_empty() {
        return false;
    }
    let mut expected = vec![format!("{}{}", subject, compact(&hole.predicate))];
    if !hole.slot.is_empty() && hole.slot != hole.predicate {
        expected.push(format!("{}{}", subject, compact(&hole.slot)));
    }
    expected.iter().any(|phrase| phrase == &left)
}

/// 逆向洞的存储方向陈述：「赵六的师傅是王五」→ 取值「赵六」。
fn inverted_fact_subject(left: &str, right: &str, hole: &PendingHole) -> Option<String> {
    if !hole.inverted {
        return None;
    }
    fn compact(text: &str) -> String {
        text.chars()
            .filter(|ch| !ch.is_whitespace() && !matches!(ch, '的'))
            .collect()
    }
    if compact(right) != compact(&hole.subject) {
        return None;
    }
    let slot = compact(if hole.slot.is_empty() {
        &hole.predicate
    } else {
        &hole.slot
    });
    if slot.is_empty() {
        return None;
    }
    let compacted = compact(left);
    let object = compacted
        .strip_suffix(slot.as_str())?
        .trim()
        .to_string();
    if object.is_empty() || object == compact(&hole.subject) {
        return None;
    }
    Some(object)
}

fn clean_value(value: &str) -> String {
    value
        .trim()
        .trim_matches(|ch: char| matches!(ch, '。' | '！' | '!' | '，' | ','))
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hole() -> PendingHole {
        PendingHole::new("李四", "师傅", "师傅", false)
    }

    #[test]
    fn bare_value_binds() {
        assert_eq!(
            interpret_answer("王五", &hole()),
            AnswerBinding::Fill("王五".to_string())
        );
        assert_eq!(
            interpret_answer("是王五。", &hole()),
            AnswerBinding::Fill("王五".to_string())
        );
        assert_eq!(
            interpret_answer("王五吧", &hole()),
            AnswerBinding::Fill("王五".to_string())
        );
    }

    #[test]
    fn explicit_fact_binds_only_matching_hole() {
        assert_eq!(
            interpret_answer("李四的师傅是王五", &hole()),
            AnswerBinding::Fill("王五".to_string())
        );
        assert_eq!(
            interpret_answer("张三的师傅是李四", &hole()),
            AnswerBinding::NotAnAnswer
        );
        assert_eq!(
            interpret_answer("李四的师傅的爸爸是王五", &hole()),
            AnswerBinding::NotAnAnswer,
            "复合陈述不能冒充洞的直接填充"
        );
    }

    #[test]
    fn question_and_command_like_input_does_not_bind() {
        assert_eq!(
            interpret_answer("王五是谁", &hole()),
            AnswerBinding::NotAnAnswer
        );
        assert_eq!(
            interpret_answer("张三的师傅的师傅是谁", &hole()),
            AnswerBinding::NotAnAnswer
        );
        assert_eq!(
            interpret_answer("记住：李四的师傅是王五", &hole()),
            AnswerBinding::NotAnAnswer
        );
    }

    #[test]
    fn negation_and_full_statement_are_not_swallowed_as_bare_values() {
        assert_eq!(
            interpret_answer("李四的师傅不是王五", &hole()),
            AnswerBinding::NotAnAnswer
        );
        assert_eq!(
            interpret_answer("李四的师傅是不是王五", &hole()),
            AnswerBinding::NotAnAnswer
        );
        assert_eq!(
            interpret_answer("李四没有师傅", &hole()),
            AnswerBinding::NotAnAnswer
        );
        assert_eq!(
            interpret_answer("李四的师傅叫王五", &hole()),
            AnswerBinding::NotAnAnswer,
            "不是「X 是 Y」的整句也不该被当成裸取值"
        );
        assert_eq!(
            interpret_answer("叫王五", &hole()),
            AnswerBinding::Fill("王五".to_string())
        );
    }

    #[test]
    fn unknown_is_explicit_and_not_a_negative_fact() {
        assert_eq!(interpret_answer("不知道", &hole()), AnswerBinding::Unknown);
        assert_eq!(interpret_answer("不清楚。", &hole()), AnswerBinding::Unknown);
    }

    #[test]
    fn ledger_dedups_same_hole_and_keeps_ids_monotonic() {
        let mut ledger = PendingQuestionLedger::new();
        let first = ledger.record(
            "张三的师傅的师傅是谁？",
            "张三 的 师傅 的 师傅",
            hole(),
            vec!["张三 的 师傅 是 李四".to_string()],
        );
        let again = ledger.record(
            "张三的师傅的师傅是谁？",
            "张三 的 师傅 的 师傅",
            hole(),
            Vec::new(),
        );
        assert_eq!(ledger.len(), 1);
        assert_eq!(first.id, again.id);
        assert_eq!(again.asks, 2);
        let second = ledger.record(
            "张三的师傅的师傅是谁？",
            "张三 的 师傅 的 师傅",
            PendingHole::new("王五", "师傅", "师傅", false),
            Vec::new(),
        );
        assert_eq!(ledger.len(), 2);
        assert!(second.id > first.id);
    }

    #[test]
    fn ledger_is_bounded() {
        let mut ledger = PendingQuestionLedger::new();
        for index in 0..(MAX_PENDING_QUESTIONS + 2) {
            ledger.record(
                &format!("问题{}", index),
                "目标",
                PendingHole::new(&format!("实体{}", index), "师傅", "师傅", false),
                Vec::new(),
            );
        }
        assert_eq!(ledger.len(), MAX_PENDING_QUESTIONS);
        assert_eq!(ledger.first().unwrap().hole.subject, "实体2");
    }

    #[test]
    fn ask_text_names_the_missing_premise() {
        let mut ledger = PendingQuestionLedger::new();
        let pending = ledger.record(
            "张三的师傅的师傅是谁？",
            "张三 的 师傅 的 师傅",
            hole(),
            vec!["张三 的 师傅 是 李四".to_string()],
        );
        let text = PendingQuestionLedger::render_ask(&pending);
        assert!(text.contains("李四 的 师傅"), "{}", text);
        assert!(text.contains("张三的师傅的师傅是谁"), "{}", text);
        assert!(text.contains("张三 的 师傅 是 李四"), "{}", text);
    }

    #[test]
    fn empty_ledger_serializes_to_nothing_extra() {
        let ledger = PendingQuestionLedger::new();
        let value = serde_json::to_value(&ledger).unwrap();
        assert_eq!(value, serde_json::json!({}));
    }

    #[test]
    fn active_question_is_the_last_one_asked() {
        let mut ledger = PendingQuestionLedger::new();
        let first = ledger.record(
            "张三的师傅的师傅是谁？",
            "张三 的 师傅 的 师傅",
            hole(),
            Vec::new(),
        );
        ledger.mark_asked(first.id);
        let second = ledger.record(
            "王五的师傅的师傅是谁？",
            "王五 的 师傅 的 师傅",
            PendingHole::new("赵六", "师傅", "师傅", false),
            Vec::new(),
        );
        ledger.mark_asked(second.id);
        assert_eq!(ledger.active().map(|pending| pending.id), Some(second.id));
        assert!(ledger.awaiting_answer());

        // 追问窗口未关闭时，裸回答瞄向最后问过的那条。
        ledger.clear_awaiting();
        assert!(!ledger.awaiting_answer());
        assert_eq!(
            ledger.active().map(|pending| pending.hole.subject.as_str()),
            Some("赵六")
        );

        // 删除当前追问后，活动项回退到仍挂着的那条，且不再接受裸回答。
        ledger.remove(second.id);
        assert_eq!(ledger.active().map(|pending| pending.id), Some(first.id));
        assert!(!ledger.awaiting_answer());
    }

    #[test]
    fn restore_keeps_awaiting_state_and_normalizes_dangling_active() {
        let mut ledger = PendingQuestionLedger::new();
        let pending = ledger.record(
            "张三的师傅的师傅是谁？",
            "张三 的 师傅 的 师傅",
            hole(),
            Vec::new(),
        );
        ledger.mark_asked(pending.id);
        let mut restored = PendingQuestionLedger::new();
        restored.restore_from(&ledger);
        assert!(restored.awaiting_answer());
        assert_eq!(restored.active().map(|item| item.id), Some(pending.id));

        let mut dangling = ledger.clone();
        dangling.active_id = Some(pending.id + 99);
        dangling.awaiting_answer = true;
        let mut normalized = PendingQuestionLedger::new();
        normalized.restore_from(&dangling);
        assert!(!normalized.awaiting_answer());
        assert_eq!(normalized.active().map(|item| item.id), Some(pending.id));
    }

    #[test]
    fn explicit_prefix_is_recognized_without_awaiting_window() {
        assert!(has_explicit_answer_prefix("回答悬问：王五"));
        assert!(has_explicit_answer_prefix(" 回填悬问:王五 "));
        assert!(!has_explicit_answer_prefix("王五"));
        assert_eq!(
            interpret_answer("回答悬问：王五", &hole()),
            AnswerBinding::Fill("王五".to_string())
        );
    }

    #[test]
    fn inverted_hole_accepts_both_statement_directions() {
        let inverted = PendingHole::new("王五", "徒弟", "师傅", true);
        assert_eq!(
            interpret_answer("王五的徒弟是赵六", &inverted),
            AnswerBinding::Fill("赵六".to_string())
        );
        assert_eq!(
            interpret_answer("赵六的师傅是王五", &inverted),
            AnswerBinding::Fill("赵六".to_string())
        );
        assert_eq!(
            interpret_answer("赵六的徒弟是王五", &inverted),
            AnswerBinding::NotAnAnswer,
            "反方向语句不等价，不能回填"
        );
        assert_eq!(
            interpret_answer("赵六的师傅是李四", &inverted),
            AnswerBinding::NotAnAnswer
        );
    }

    #[test]
    fn shelved_question_is_kept_but_not_active_until_reopened() {
        let mut ledger = PendingQuestionLedger::new();
        let pending = ledger.record(
            "张三的师傅的师傅是谁？",
            "张三 的 师傅 的 师傅",
            hole(),
            Vec::new(),
        );
        ledger.mark_asked(pending.id);
        ledger.suspend(pending.id);
        assert!(!ledger.has_active());
        assert!(!ledger.is_empty(), "搁置不等于忘记");
        assert_eq!(ledger.shelved_len(), 1);
        assert!(!ledger.awaiting_answer());
        assert!(ledger.render_report().contains("已搁置 1 条"));

        let encoded = serde_json::to_value(&ledger).unwrap();
        assert_equal_roundtrip(&ledger, &encoded);

        assert_eq!(ledger.reopen_shelved(), 1);
        assert!(ledger.has_active());
        assert!(ledger.shelved().is_empty());
        assert!(!ledger.awaiting_answer(), "重开后还需要再追问一次");
        assert_eq!(ledger.active().map(|item| item.id), Some(pending.id));
    }

    fn assert_equal_roundtrip(ledger: &PendingQuestionLedger, value: &serde_json::Value) {
        let decoded: PendingQuestionLedger = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(&decoded, ledger);
    }
}

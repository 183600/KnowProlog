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
}

fn is_zero(value: &u64) -> bool {
    *value == 0
}

impl PendingQuestionLedger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.questions.is_empty()
    }

    pub fn len(&self) -> usize {
        self.questions.len()
    }

    pub fn questions(&self) -> &[PendingQuestion] {
        &self.questions
    }

    pub fn first(&self) -> Option<&PendingQuestion> {
        self.questions.first()
    }

    pub fn clear(&mut self) {
        self.questions.clear();
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
            if existing.support.is_empty() && !support.is_empty() {
                existing.support = support;
            }
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
        self.questions.push(pending.clone());
        pending
    }

    pub fn remove(&mut self, id: u64) -> Option<PendingQuestion> {
        let position = self.questions.iter().position(|pending| pending.id == id)?;
        Some(self.questions.remove(position))
    }

    pub fn remove_first(&mut self) -> Option<PendingQuestion> {
        if self.questions.is_empty() {
            None
        } else {
            Some(self.questions.remove(0))
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
        if self.questions.is_empty() {
            return "悬问台账为空：当前没有欠用户答案的问题。".to_string();
        }
        let mut output = format!("悬问台账：{} 条未答之问。\n", self.questions.len());
        for pending in &self.questions {
            output.push_str(&format!(
                "  {}. 「{}」缺「{}」（已追问 {} 次）\n",
                pending.id,
                pending.question,
                pending.hole.phrase(),
                pending.asks
            ));
        }
        output.push_str("回复取值即可回填；输入「放弃悬问」清空台账。");
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
    // 带前缀的命令式输入（「记住：…」「教学：…」）不是对悬问的短答。
    if text.contains('：') || text.contains(':') {
        return AnswerBinding::NotAnAnswer;
    }
    // 显式整句：「李四的师傅是王五」。
    if let Some((left, right)) = split_fact(text) {
        let subject_matches = left.contains(hole.subject.as_str());
        let predicate_matches = left.contains(hole.predicate.as_str())
            || (!hole.slot.is_empty() && left.contains(hole.slot.as_str()));
        if subject_matches && predicate_matches {
            let object = clean_value(right);
            if !object.is_empty() {
                return AnswerBinding::Fill(object);
            }
        }
        // 是完整陈述句但填的不是这个洞：让位，不把它当裸取值。
        return AnswerBinding::NotAnAnswer;
    }
    // 短答「是王五」。
    let mut value = normalized;
    if let Some(rest) = value.strip_prefix('是') {
        value = rest.trim();
    }
    for suffix in ["吧", "呀", "啊"] {
        if let Some(rest) = value.strip_suffix(suffix) {
            value = rest.trim_end();
        }
    }
    let value = value.trim();
    if value.is_empty()
        || value.chars().count() > MAX_BARE_ANSWER_CHARS
        || value.contains(|ch: char| matches!(ch, '是' | '为' | '，' | ',' | '：' | ':' | '；' | ';'))
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
}

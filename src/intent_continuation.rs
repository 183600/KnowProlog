use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::soft_route::{SoftCandidate, SoftDecision, SoftRouteOutcome};

pub const DIALECT: &str = "knowprolog-intent-continuation/1";

const MAX_OPTIONS: usize = 3;
const DEFAULT_TTL_TURNS: u64 = 3;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IntentOption {
    pub pattern_id: String,
    pub pattern_name: String,
    pub pattern: String,
    pub query_type: String,
    #[serde(default)]
    pub entities: Vec<String>,
    pub confidence: f64,
}

impl IntentOption {
    fn from_candidate(candidate: &SoftCandidate) -> Self {
        Self {
            pattern_id: candidate.entry.id(),
            pattern_name: candidate.entry.name.clone(),
            pattern: candidate.entry.render(),
            query_type: candidate.entry.query_type.clone(),
            entities: candidate.alignment.entities.clone(),
            confidence: candidate.confidence,
        }
    }

    fn label(&self) -> String {
        let mut label = if self.pattern_name.trim().is_empty() {
            self.query_type.clone()
        } else {
            self.pattern_name.clone()
        };
        if label != self.query_type {
            label.push_str(&format!("（{}）", self.query_type));
        }
        if !self.entities.is_empty() {
            label.push_str(&format!("，实体：{}", self.entities.join("、")));
        }
        label
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PendingIntent {
    pub dialect: String,
    pub original_input: String,
    pub turn: u64,
    pub options: Vec<IntentOption>,
}

impl PendingIntent {
    pub fn from_soft_outcome(input: &str, outcome: &SoftRouteOutcome, turn: u64) -> Option<Self> {
        if outcome.decision != SoftDecision::Clarify {
            return None;
        }
        let options = options_from_outcome(outcome);
        if options.len() < 2 {
            return None;
        }
        Some(Self {
            dialect: DIALECT.to_string(),
            original_input: input.trim().to_string(),
            turn,
            options,
        })
    }

    pub fn prompt(&self) -> String {
        let mut prompt = format!(
            "「{}」有 {} 种可能的问法，你想问哪一种？",
            self.original_input,
            self.options.len()
        );
        for (index, option) in self.options.iter().enumerate() {
            prompt.push_str(&format!("\n  {}. {}", index + 1, option.label()));
        }
        prompt.push_str("\n回复序号选择；回复「都不是」放弃这次追问，也可以直接问新问题。");
        prompt
    }

    pub fn parse_reply(&self, text: &str) -> IntentReply {
        let reply = text
            .trim()
            .trim_end_matches(|ch: char| {
                matches!(
                    ch,
                    '。' | '.' | '，' | ',' | '、' | '！' | '!' | '？' | '?' | '～' | '~'
                )
            })
            .trim();
        if reply.is_empty() {
            return IntentReply::NewInput;
        }
        const CANCEL: [&str; 12] = [
            "都不是",
            "都不要",
            "算了",
            "取消",
            "放弃",
            "不用了",
            "不查了",
            "不问了",
            "结束澄清",
            "停止澄清",
            "退出澄清",
            "quit",
        ];
        if CANCEL.contains(&reply) || reply == "exit" {
            return IntentReply::Cancel;
        }
        if reply == "前者" {
            return if self.options.is_empty() {
                IntentReply::OutOfRange
            } else {
                IntentReply::Choice(0)
            };
        }
        if reply == "后者" {
            return if self.options.len() >= 2 {
                IntentReply::Choice(1)
            } else {
                IntentReply::OutOfRange
            };
        }
        if let Some(index) = parse_ordinal(reply) {
            if index == 0 || index > self.options.len() {
                return IntentReply::OutOfRange;
            }
            return IntentReply::Choice(index - 1);
        }
        let bare = reply.trim_matches(|ch: char| {
            matches!(ch, '「' | '」' | '『' | '』' | '"' | '\'' | '“' | '”')
        });
        let matches: Vec<usize> = self
            .options
            .iter()
            .enumerate()
            .filter(|(_, option)| {
                option.pattern_name == bare
                    || option.query_type == bare
                    || option.pattern == bare
                    || option.pattern_id == bare
            })
            .map(|(index, _)| index)
            .collect();
        if matches.len() == 1 {
            IntentReply::Choice(matches[0])
        } else {
            IntentReply::NewInput
        }
    }

    pub fn option(&self, index: usize) -> Option<&IntentOption> {
        self.options.get(index)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntentReply {
    Choice(usize),
    OutOfRange,
    Cancel,
    NewInput,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IntentContinuation {
    #[serde(default)]
    dialect: String,
    #[serde(default)]
    turn: u64,
    #[serde(default = "default_ttl")]
    ttl: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pending: Option<PendingIntent>,
    #[serde(default)]
    asked: u64,
    #[serde(default)]
    chosen: u64,
    #[serde(default)]
    cancelled: u64,
    #[serde(default)]
    expired: u64,
    #[serde(default)]
    replaced: u64,
}

fn default_ttl() -> u64 {
    DEFAULT_TTL_TURNS
}

impl Default for IntentContinuation {
    fn default() -> Self {
        Self {
            dialect: DIALECT.to_string(),
            turn: 0,
            ttl: DEFAULT_TTL_TURNS,
            pending: None,
            asked: 0,
            chosen: 0,
            cancelled: 0,
            expired: 0,
            replaced: 0,
        }
    }
}

impl IntentContinuation {
    pub fn begin_turn(&mut self) -> bool {
        self.turn = self.turn.saturating_add(1);
        let expired = self
            .pending
            .as_ref()
            .map(|pending| self.turn.saturating_sub(pending.turn) >= self.ttl)
            .unwrap_or(false);
        if expired {
            self.pending = None;
            self.expired += 1;
        }
        expired
    }

    pub fn turn(&self) -> u64 {
        self.turn
    }

    pub fn pending(&self) -> Option<&PendingIntent> {
        self.pending.as_ref()
    }

    pub fn has_pending(&self) -> bool {
        self.pending.is_some()
    }

    pub fn offer(&mut self, input: &str, outcome: &SoftRouteOutcome) -> Option<&PendingIntent> {
        let pending = PendingIntent::from_soft_outcome(input, outcome, self.turn)?;
        if self.pending.is_some() {
            self.replaced += 1;
        }
        self.asked += 1;
        self.pending = Some(pending);
        self.pending.as_ref()
    }

    pub fn choose(&mut self, index: usize) -> Option<IntentOption> {
        let option = self.pending.as_ref()?.option(index)?.clone();
        self.pending = None;
        self.chosen += 1;
        Some(option)
    }

    pub fn cancel(&mut self) -> Option<PendingIntent> {
        let pending = self.pending.take();
        if pending.is_some() {
            self.cancelled += 1;
        }
        pending
    }

    pub fn take_pending(&mut self) -> Option<PendingIntent> {
        self.pending.take()
    }

    pub fn asked(&self) -> u64 {
        self.asked
    }

    pub fn chosen(&self) -> u64 {
        self.chosen
    }

    pub fn cancelled(&self) -> u64 {
        self.cancelled
    }

    pub fn expired(&self) -> u64 {
        self.expired
    }

    pub fn replaced(&self) -> u64 {
        self.replaced
    }

    pub fn render_report(&self) -> String {
        let mut report = match self.pending.as_ref() {
            Some(pending) => pending.prompt(),
            None => "当前没有待裁决意图。".to_string(),
        };
        report.push_str(&format!(
            "\n统计：发起 {}，选择 {}，放弃 {}，过期 {}，覆盖 {}。",
            self.asked, self.chosen, self.cancelled, self.expired, self.replaced
        ));
        report
    }

    pub fn load_from_path(path: &str) -> Self {
        match std::fs::read_to_string(path) {
            Ok(text) => serde_json::from_str(&text).unwrap_or_else(|error| {
                eprintln!("意图续体读取失败（已忽略，不影响知识库）：{}", error);
                Self::default()
            }),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Self::default(),
            Err(error) => {
                eprintln!("意图续体读取失败（已忽略，不影响知识库）：{}", error);
                Self::default()
            }
        }
    }

    pub fn save_to_path(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::Other, error))?;
        let temporary = format!("{}.tmp", path);
        std::fs::write(&temporary, json)?;
        std::fs::rename(&temporary, path)
    }
}

pub fn has_distinct_options(outcome: &SoftRouteOutcome) -> bool {
    options_from_outcome(outcome).len() >= 2
}

fn options_from_outcome(outcome: &SoftRouteOutcome) -> Vec<IntentOption> {
    let mut options = Vec::new();
    let mut seen_types = BTreeSet::new();
    let mut seen_ids = BTreeSet::new();
    let selected_id = outcome
        .selected
        .as_ref()
        .map(|selected| selected.entry.id());
    for candidate in outcome.clarification_options() {
        let id = candidate.entry.id();
        if options.is_empty() {
            if selected_id.as_deref() != Some(id.as_str()) {
                continue;
            }
        } else if !candidate.eligible() {
            continue;
        }
        if !seen_ids.insert(id) {
            continue;
        }
        if !seen_types.insert(candidate.query_type().to_string()) {
            continue;
        }
        options.push(IntentOption::from_candidate(candidate));
        if options.len() >= MAX_OPTIONS {
            break;
        }
    }
    options
}

fn parse_ordinal(text: &str) -> Option<usize> {
    let mut digits = text.trim();
    for _ in 0..3 {
        let mut changed = false;
        for prefix in ["我想问", "我想选", "选择", "选", "第", "问"] {
            if let Some(rest) = digits.strip_prefix(prefix) {
                digits = rest.trim();
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    loop {
        let stripped = digits.trim_end_matches(|ch: char| {
            matches!(
                ch,
                '个' | '条' | '项' | '种' | '号' | '吧' | '。' | '.' | '，' | ','
            )
        });
        if stripped == digits {
            break;
        }
        digits = stripped.trim_end();
    }
    if digits.is_empty() {
        return None;
    }
    if digits.chars().all(|ch| ch.is_ascii_digit()) {
        return digits.parse().ok();
    }
    chinese_numeral(digits)
}

fn chinese_numeral(text: &str) -> Option<usize> {
    let digit = |ch: char| -> Option<usize> {
        Some(match ch {
            '一' => 1,
            '二' | '两' => 2,
            '三' => 3,
            '四' => 4,
            '五' => 5,
            '六' => 6,
            '七' => 7,
            '八' => 8,
            '九' => 9,
            _ => return None,
        })
    };
    let chars: Vec<char> = text.chars().collect();
    match chars.as_slice() {
        ['十'] => Some(10),
        [single] => digit(*single),
        ['十', ones] => Some(10 + digit(*ones)?),
        [tens, '十'] => Some(digit(*tens)? * 10),
        [tens, '十', ones] => Some(digit(*tens)? * 10 + digit(*ones)?),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bindings::BindingSet;
    use crate::pattern_ir::PatternEntry;
    use crate::soft_route::{Alignment, SoftCandidate, DIALECT as SOFT_DIALECT};

    fn candidate(
        name: &str,
        query_type: &str,
        entities: &[&str],
        confidence: f64,
        blocked: Option<&str>,
    ) -> SoftCandidate {
        let entry = PatternEntry::new(name, "test", query_type, 10, true, Vec::new());
        let total_tokens = entities.len().max(1);
        SoftCandidate {
            entry,
            confidence,
            alignment: Alignment {
                total_elems: 1,
                matched_elems: 1,
                partial_elems: 0,
                matched_tokens: total_tokens,
                total_tokens,
                extra_tokens: 0,
                score: 0.9,
                exact: false,
                elem_spans: Vec::new(),
                entity_spans: Vec::new(),
                entities: entities.iter().map(|entity| entity.to_string()).collect(),
                bindings: BindingSet::default(),
            },
            blocked: blocked.map(str::to_string),
            evidence: Vec::new(),
        }
    }

    fn outcome(decision: SoftDecision) -> SoftRouteOutcome {
        SoftRouteOutcome {
            dialect: SOFT_DIALECT.to_string(),
            decision,
            selected: Some(candidate("师傅查询", "master_query", &["张三"], 0.62, None)),
            alternatives: vec![
                candidate("老师查询", "teacher_query", &["张三"], 0.58, None),
                candidate("师傅别名", "master_query", &["张三"], 0.55, None),
                candidate("失配查询", "other_query", &["张三"], 0.52, Some("闸门")),
            ],
            reason: "两种读法接近".to_string(),
        }
    }

    #[test]
    fn builds_one_option_per_distinct_query_type() {
        let pending =
            PendingIntent::from_soft_outcome("张三的师傅是谁", &outcome(SoftDecision::Clarify), 7)
                .expect("应能建立续体");
        assert_eq!(pending.options.len(), 2);
        assert_eq!(pending.options[0].query_type, "master_query");
        assert_eq!(pending.options[1].query_type, "teacher_query");
        assert_eq!(pending.turn, 7);
        assert!(pending.prompt().contains("你想问哪一种"));
    }

    #[test]
    fn ignores_accept_decisions() {
        assert!(PendingIntent::from_soft_outcome(
            "张三的师傅是谁",
            &outcome(SoftDecision::Accept),
            1
        )
        .is_none());
    }

    #[test]
    fn parses_numeric_and_named_replies() {
        let pending =
            PendingIntent::from_soft_outcome("张三的师傅是谁", &outcome(SoftDecision::Clarify), 1)
                .expect("应能建立续体");
        assert_eq!(pending.parse_reply("2"), IntentReply::Choice(1));
        assert_eq!(pending.parse_reply("第二个"), IntentReply::Choice(1));
        assert_eq!(pending.parse_reply("前者"), IntentReply::Choice(0));
        assert_eq!(pending.parse_reply("后者"), IntentReply::Choice(1));
        assert_eq!(pending.parse_reply("老师查询"), IntentReply::Choice(1));
        assert_eq!(pending.parse_reply("teacher_query"), IntentReply::Choice(1));
        assert_eq!(pending.parse_reply("都不是"), IntentReply::Cancel);
        assert_eq!(pending.parse_reply("9"), IntentReply::OutOfRange);
        assert_eq!(pending.parse_reply("张三的师傅是谁"), IntentReply::NewInput);
    }

    #[test]
    fn choose_and_cancel_track_state() {
        let mut continuation = IntentContinuation::default();
        continuation.begin_turn();
        continuation
            .offer("张三的师傅是谁", &outcome(SoftDecision::Clarify))
            .expect("应能登记续体");
        assert!(continuation.has_pending());
        assert_eq!(
            continuation.choose(1).map(|option| option.query_type),
            Some("teacher_query".to_string())
        );
        assert!(!continuation.has_pending());
        assert_eq!(continuation.chosen(), 1);

        continuation.begin_turn();
        continuation
            .offer("张三的师傅是谁", &outcome(SoftDecision::Clarify))
            .expect("应能再次登记");
        assert!(continuation.cancel().is_some());
        assert_eq!(continuation.cancelled(), 1);
    }

    #[test]
    fn expires_after_ttl_without_state_leak() {
        let mut continuation = IntentContinuation::default();
        assert!(!continuation.begin_turn());
        continuation
            .offer("张三的师傅是谁", &outcome(SoftDecision::Clarify))
            .expect("应能登记续体");
        for _ in 0..DEFAULT_TTL_TURNS - 1 {
            assert!(!continuation.begin_turn());
            assert!(continuation.has_pending());
        }
        assert!(continuation.begin_turn());
        assert!(!continuation.has_pending());
        assert_eq!(continuation.expired(), 1);
    }

    #[test]
    fn persists_and_restores_pending_intent() {
        let directory = tempfile::tempdir().expect("临时目录");
        let path = directory.path().join("pending_intent.json");
        let path = path.to_string_lossy();
        let mut continuation = IntentContinuation::default();
        continuation.begin_turn();
        continuation
            .offer("张三的师傅是谁", &outcome(SoftDecision::Clarify))
            .expect("应能登记续体");
        continuation.save_to_path(&path).expect("应能保存");
        let restored = IntentContinuation::load_from_path(&path);
        assert!(restored.has_pending());
        assert_eq!(
            restored.pending().map(|pending| pending.options.len()),
            Some(2)
        );
        assert_eq!(restored.turn(), 1);
    }
}

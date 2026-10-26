//! CXT：有界上下文窗口（Bounded Context Window）。
//!
//! 多轮对话里「最近 N 轮」不是记忆：它既不知道哪一轮支撑当前焦点，也没有
//! 字符预算。本模块把窗口选择拆成可审计的三步：
//!
//! 1. [`ContextCue`]：每轮只保留检索所需的线索（焦点实体、谓词、是否悬问、
//!    字符数），不复制整段对话；
//! 2. [`WindowBudget`]：轮数与字符双重封顶，最近的若干轮可强制保留，但
//!    超出预算会明确标记；
//! 3. [`select_window`]：按「与当前焦点的重叠 → 新近性 → 稳定 id」确定性地
//!    选槽，返回保留/丢弃清单与字符用量，便于回放和解释。
//!
//! 设计约束：
//!   - 纯函数：同一输入、同一预算永远得到同一窗口；
//!   - 有界：不扫描全库，只在线索列表上做一次排序与一次贪心；
//!   - 诚实：预算不足不是静默截断，而是写进 [`WindowSelection::over_budget`]；
//!   - 可解释：每条被丢弃的线索都有 [`DropReason`]。

use serde::{Deserialize, Serialize};

/// 上下文窗口选择器版本。
pub const CONTEXT_WINDOW_VERSION: u32 = 1;

/// 一轮对话的可检索线索（不含完整回答，避免窗口本身膨胀）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextCue {
    pub id: usize,
    /// 用户原话（用于渲染与最后的词面兜底匹配）。
    pub text: String,
    /// 本轮显式实体/焦点。
    #[serde(default)]
    pub focus: Vec<String>,
    /// 本轮谓词/属性。
    #[serde(default)]
    pub predicates: Vec<String>,
    /// 本轮是否仍有未解决的问题。
    #[serde(default)]
    pub open: bool,
}

impl ContextCue {
    pub fn new(id: usize, text: impl Into<String>) -> Self {
        Self {
            id,
            text: text.into(),
            focus: Vec::new(),
            predicates: Vec::new(),
            open: false,
        }
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

    pub fn open(mut self, open: bool) -> Self {
        self.open = open;
        self
    }

    pub fn char_count(&self) -> usize {
        self.text.chars().count()
    }
}

/// 窗口预算：轮数与字符双上限；`min_recent` 是必须保留的最近轮数。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct WindowBudget {
    pub max_items: usize,
    pub max_chars: usize,
    pub min_recent: usize,
}

impl Default for WindowBudget {
    fn default() -> Self {
        Self {
            max_items: 8,
            max_chars: 480,
            min_recent: 1,
        }
    }
}

impl WindowBudget {
    pub fn new(max_items: usize, max_chars: usize, min_recent: usize) -> Self {
        Self {
            max_items,
            max_chars,
            min_recent: min_recent.min(max_items),
        }
    }
}

/// 丢弃原因闭集。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DropReason {
    /// 轮数预算已满。
    ItemBudget,
    /// 字符预算不足以容纳该轮。
    CharBudget,
    /// 预算为 0（显式关闭窗口）。
    Disabled,
}

impl DropReason {
    pub fn label(self) -> &'static str {
        match self {
            DropReason::ItemBudget => "轮数预算",
            DropReason::CharBudget => "字符预算",
            DropReason::Disabled => "窗口关闭",
        }
    }
}

/// 一条被丢弃的线索。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DroppedCue {
    pub id: usize,
    pub reason: DropReason,
}

/// 一次窗口选择的结果。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WindowSelection {
    pub version: u32,
    /// 保留的线索 id，按时间先后排列。
    pub kept: Vec<usize>,
    pub dropped: Vec<DroppedCue>,
    pub considered: usize,
    /// 保留线索的字符总数。
    pub retained_chars: usize,
    /// 是否因为强制保留最近轮次而超出字符预算。
    pub over_budget: bool,
}

impl WindowSelection {
    pub fn keep(&self, id: usize) -> bool {
        self.kept.binary_search(&id).is_ok()
    }

    pub fn dropped_reason(&self, id: usize) -> Option<DropReason> {
        self.dropped
            .iter()
            .find(|item| item.id == id)
            .map(|item| item.reason)
    }

    pub fn render_receipt(&self, budget: WindowBudget) -> String {
        let reasons = if self.dropped.is_empty() {
            "无丢弃".to_string()
        } else {
            let mut item = 0usize;
            let mut chars = 0usize;
            let mut disabled = 0usize;
            for dropped in &self.dropped {
                match dropped.reason {
                    DropReason::ItemBudget => item += 1,
                    DropReason::CharBudget => chars += 1,
                    DropReason::Disabled => disabled += 1,
                }
            }
            format!(
                "丢弃：轮数预算 {}、字符预算 {}、窗口关闭 {}",
                item, chars, disabled
            )
        };
        format!(
            "上下文窗口：保留 {}/{} 条，字符 {}/{}；{}；{}。",
            self.kept.len(),
            self.considered,
            self.retained_chars,
            budget.max_chars,
            reasons,
            if self.over_budget {
                "强制保留最近轮次，已超出字符预算"
            } else {
                "未超预算"
            }
        )
    }
}

/// 选择有界上下文窗口。
///
/// `focus` 与 `predicates` 是当前轮的检索线索；为空时退化为「最近 N 轮」。
/// 最近 `budget.min_recent` 轮无条件优先，即使因此超出字符预算（会标记
/// `over_budget`）；其余槽位按相关性与新近性贪心填充。
pub fn select_window(
    cues: &[ContextCue],
    focus: &[String],
    predicates: &[String],
    budget: WindowBudget,
) -> WindowSelection {
    let considered = cues.len();
    if budget.max_items == 0 || considered == 0 {
        return WindowSelection {
            version: CONTEXT_WINDOW_VERSION,
            kept: Vec::new(),
            dropped: cues
                .iter()
                .map(|cue| DroppedCue {
                    id: cue.id,
                    reason: DropReason::Disabled,
                })
                .collect(),
            considered,
            retained_chars: 0,
            over_budget: false,
        };
    }

    let required = budget.min_recent.min(considered).min(budget.max_items);
    let required_start = considered - required;
    let mut selected: Vec<usize> = (required_start..considered).collect();
    let mut retained_chars: usize = selected.iter().map(|index| cues[*index].char_count()).sum();
    let over_budget = retained_chars > budget.max_chars;

    let mut candidates: Vec<usize> = (0..required_start).collect();
    candidates.sort_by(|left, right| {
        let left_score = cue_score(&cues[*left], *left, considered, focus, predicates);
        let right_score = cue_score(&cues[*right], *right, considered, focus, predicates);
        right_score.cmp(&left_score).then_with(|| right.cmp(left))
    });

    for index in candidates {
        if selected.len() >= budget.max_items {
            break;
        }
        let chars = cues[index].char_count();
        if retained_chars.saturating_add(chars) > budget.max_chars {
            continue;
        }
        selected.push(index);
        retained_chars += chars;
    }

    selected.sort_unstable();
    let mut kept_ids: Vec<usize> = selected.iter().map(|index| cues[*index].id).collect();
    kept_ids.sort_unstable();

    let dropped = cues
        .iter()
        .enumerate()
        .filter(|(index, _)| !selected.contains(index))
        .map(|(_, cue)| DroppedCue {
            id: cue.id,
            reason: if selected.len() >= budget.max_items {
                DropReason::ItemBudget
            } else {
                DropReason::CharBudget
            },
        })
        .collect();

    WindowSelection {
        version: CONTEXT_WINDOW_VERSION,
        kept: kept_ids,
        dropped,
        considered,
        retained_chars,
        over_budget,
    }
}

fn cue_score(
    cue: &ContextCue,
    index: usize,
    total: usize,
    focus: &[String],
    predicates: &[String],
) -> i64 {
    let mut overlap = 0i64;
    for term in focus {
        let term = term.as_str();
        if term.is_empty() {
            continue;
        }
        if cue.focus.iter().any(|item| item == term) || cue.text.contains(term) {
            overlap += 2;
        }
    }
    for term in predicates {
        let term = term.as_str();
        if term.is_empty() {
            continue;
        }
        if cue.predicates.iter().any(|item| item == term) || cue.text.contains(term) {
            overlap += 1;
        }
    }
    let recency = index as i64 + 1;
    let open_bonus = if cue.open { total as i64 + 1 } else { 0 };
    overlap * (2 * (total as i64 + 1)) + open_bonus + recency
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cues() -> Vec<ContextCue> {
        vec![
            ContextCue::new(1, "姚明的身高是多少")
                .with_focus(["姚明"])
                .with_predicates(["身高"]),
            ContextCue::new(2, "今天天气怎么样"),
            ContextCue::new(3, "姚明的国籍是什么")
                .with_focus(["姚明"])
                .with_predicates(["国籍"]),
            ContextCue::new(4, "谢谢"),
        ]
    }

    #[test]
    fn without_cues_the_window_is_recent_turns_in_order() {
        let selection = select_window(&cues(), &[], &[], WindowBudget::new(2, 100, 1));
        assert_eq!(selection.kept, vec![3, 4]);
        assert_eq!(selection.dropped_reason(1), Some(DropReason::ItemBudget));
        assert!(!selection.over_budget);
    }

    #[test]
    fn focus_relevance_can_pull_an_older_turn_into_the_window() {
        let focus = vec!["姚明".to_string()];
        let selection = select_window(&cues(), &focus, &[], WindowBudget::new(2, 100, 1));
        assert_eq!(selection.kept, vec![3, 4], "最近一轮 + 更近的相关轮");
        assert_eq!(selection.dropped_reason(1), Some(DropReason::ItemBudget));
    }

    #[test]
    fn predicates_break_focus_ties_without_floating_point() {
        let cues = vec![
            ContextCue::new(1, "姚明的身高")
                .with_focus(["姚明"])
                .with_predicates(["身高"]),
            ContextCue::new(2, "姚明的国籍")
                .with_focus(["姚明"])
                .with_predicates(["国籍"]),
            ContextCue::new(3, "谢谢"),
        ];
        let selection = select_window(
            &cues,
            &["姚明".to_string()],
            &["国籍".to_string()],
            WindowBudget::new(2, 100, 1),
        );
        assert_eq!(selection.kept, vec![2, 3], "谓词命中优先于同焦点的旧轮");
    }

    #[test]
    fn character_budget_is_hard_for_optional_turns() {
        let cues = vec![
            ContextCue::new(1, "姚明的身高是多少呀呀呀呀呀呀呀呀"),
            ContextCue::new(2, "谢谢"),
            ContextCue::new(3, "好"),
        ];
        let selection = select_window(
            &cues,
            &["姚明".to_string()],
            &[],
            WindowBudget::new(3, 2, 1),
        );
        assert_eq!(selection.kept, vec![3]);
        assert_eq!(selection.dropped_reason(1), Some(DropReason::CharBudget));
        assert_eq!(selection.dropped_reason(2), Some(DropReason::CharBudget));
        assert!(!selection.over_budget);
    }

    #[test]
    fn required_recent_turn_may_exceed_budget_but_is_reported() {
        let cues = vec![
            ContextCue::new(1, "旧"),
            ContextCue::new(2, "这一轮特别特别特别特别特别特别特别特别特别长"),
        ];
        let selection = select_window(&cues, &[], &[], WindowBudget::new(2, 4, 1));
        assert_eq!(selection.kept, vec![2]);
        assert!(selection.over_budget, "强制最近轮次必须如实标记超预算");
        assert_eq!(selection.dropped_reason(1), Some(DropReason::CharBudget));
    }

    #[test]
    fn open_questions_get_a_stable_tie_break_bonus() {
        let cues = vec![
            ContextCue::new(1, "问题甲").open(true),
            ContextCue::new(2, "问题乙").open(false),
            ContextCue::new(3, "最近"),
        ];
        let selection = select_window(&cues, &[], &[], WindowBudget::new(2, 100, 1));
        assert_eq!(selection.kept, vec![1, 3]);
        assert_eq!(selection.dropped_reason(2), Some(DropReason::ItemBudget));
    }

    #[test]
    fn empty_budget_disables_the_window_explicitly() {
        let selection = select_window(&cues(), &[], &[], WindowBudget::new(0, 0, 0));
        assert!(selection.kept.is_empty());
        assert_eq!(selection.dropped.len(), 4);
        assert!(selection
            .dropped
            .iter()
            .all(|item| item.reason == DropReason::Disabled));
        assert!(selection
            .render_receipt(WindowBudget::new(0, 0, 0))
            .contains("窗口关闭"));
    }

    #[test]
    fn selection_is_repeatable_and_kept_ids_are_chronological() {
        let focus = vec!["姚明".to_string()];
        let first = select_window(&cues(), &focus, &[], WindowBudget::new(3, 100, 1));
        for _ in 0..32 {
            assert_eq!(
                first,
                select_window(&cues(), &focus, &[], WindowBudget::new(3, 100, 1))
            );
        }
        let mut sorted = first.kept.clone();
        sorted.sort_unstable();
        assert_eq!(first.kept, sorted);
        assert!(first
            .render_receipt(WindowBudget::new(3, 100, 1))
            .contains("保留"));
    }
}

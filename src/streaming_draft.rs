//! 输入侧增量草稿（Streaming Draft）。
//!
//! 用户输入尚未结束时，系统只应展示「不会被后续字符改写」的理解。本模块
//! 冻结一份提及词表快照，并用两条规则计算稳定水位：
//!
//! - 已确认词面按 [`MentionLexicon::confirmed_spans`] 选路；
//! - 任何可能在尾部继续变长的词面都会把水位拉回它的起点，因此「北京」
//!   在词表同时含「北京大学」时不会被提前封存。
//!
//! 水位之前的文本与指称被密封为 `sealed`，水位之后仍是 `pending`。密封是
//! 单调的：后续片段只能追加新区域，不能改写已经提交的提及跨度。草稿本身
//! 不写会话、不写图谱；真正的轮次仍在完整输入后由既有管线重新执行。

use std::collections::HashMap;

use crate::mention_lexicon::{LexiconSpan, LexiconStatus, MentionLexicon};

/// 一个已经密封的提及；`start` / `end` 是字符偏移，不是字节偏移。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DraftMention {
    pub surface: String,
    pub canonical: String,
    pub pos: String,
    pub etype: Option<String>,
    pub origin: String,
    pub start: usize,
    pub end: usize,
}

impl DraftMention {
    fn from_span(span: LexiconSpan) -> Self {
        Self {
            surface: span.surface,
            canonical: span.canonical,
            pos: span.pos,
            etype: span.etype,
            origin: span.origin,
            start: span.start,
            end: span.end,
        }
    }

    pub fn char_len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }
}

/// 一次追加后的增量回执：新增的稳定文本、新密封的提及与仍未稳定的尾部。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DraftUpdate {
    /// 草稿创建时冻结的词表版本。
    pub revision: u64,
    /// 本次新越过稳定水位的文本。
    pub added_text: String,
    /// 从输入开始到稳定水位的全部文本。
    pub stable_prefix: String,
    /// 稳定水位之后、可能仍会被改写的尾部文本。
    pub pending: String,
    /// 本次新密封的提及。
    pub mentions: Vec<DraftMention>,
    /// 累计稳定字符数。
    pub stable_chars: usize,
    /// 当前输入总字符数。
    pub total_chars: usize,
}

impl DraftUpdate {
    pub fn is_quiescent(&self) -> bool {
        self.added_text.is_empty() && self.mentions.is_empty()
    }
}

/// 基于冻结词表的增量输入草稿。
#[derive(Debug, Clone)]
pub struct StreamingDraft {
    lexicon: MentionLexicon,
    term_chars: Vec<Vec<char>>,
    by_first: HashMap<char, Vec<usize>>,
    revision: u64,
    text: String,
    chars: Vec<char>,
    committed: usize,
    sealed: Vec<DraftMention>,
}

impl StreamingDraft {
    /// 冻结当前词表；之后对原词表的修改不会影响这份草稿。
    pub fn new(lexicon: &MentionLexicon, revision: u64) -> Self {
        let mut term_chars = Vec::new();
        let mut by_first: HashMap<char, Vec<usize>> = HashMap::new();
        for entry in lexicon.entries() {
            if entry.status != LexiconStatus::Confirmed {
                continue;
            }
            let chars: Vec<char> = entry.surface.chars().collect();
            if chars.len() < 2 {
                continue;
            }
            if let Some(first) = chars.first().copied() {
                by_first
                    .entry(first)
                    .or_insert_with(Vec::new)
                    .push(term_chars.len());
            }
            term_chars.push(chars);
        }
        Self {
            lexicon: lexicon.clone(),
            term_chars,
            by_first,
            revision,
            text: String::new(),
            chars: Vec::new(),
            committed: 0,
            sealed: Vec::new(),
        }
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn stable_chars(&self) -> usize {
        self.committed
    }

    pub fn total_chars(&self) -> usize {
        self.chars.len()
    }

    pub fn sealed_mentions(&self) -> &[DraftMention] {
        &self.sealed
    }

    pub fn stable_prefix(&self) -> String {
        self.chars.iter().take(self.committed).collect()
    }

    pub fn pending(&self) -> String {
        self.chars.iter().skip(self.committed).collect()
    }

    /// 追加一个输入片段；返回新稳定文本与新密封提及。
    pub fn push(&mut self, fragment: &str) -> DraftUpdate {
        let previous = self.committed;
        self.text.push_str(fragment);
        self.chars.extend(fragment.chars());
        let total = self.chars.len();
        let stable = self.stable_end().max(previous).min(total);
        let mut mentions = Vec::new();
        if stable > previous {
            for span in self.lexicon.confirmed_spans(&self.text) {
                if span.start >= previous && span.end <= stable {
                    let mention = DraftMention::from_span(span);
                    self.sealed.push(mention.clone());
                    mentions.push(mention);
                }
            }
            self.committed = stable;
        }
        self.update(previous, mentions)
    }

    /// 输入结束：把剩余尾部全部密封；之后不再追加。
    pub fn seal_all(&mut self) -> DraftUpdate {
        let previous = self.committed;
        let total = self.chars.len();
        let mut mentions = Vec::new();
        if total > previous {
            for span in self.lexicon.confirmed_spans(&self.text) {
                if span.start >= previous && span.end <= total {
                    let mention = DraftMention::from_span(span);
                    self.sealed.push(mention.clone());
                    mentions.push(mention);
                }
            }
            self.committed = total;
        }
        self.update(previous, mentions)
    }

    fn update(&self, previous: usize, mentions: Vec<DraftMention>) -> DraftUpdate {
        DraftUpdate {
            revision: self.revision,
            added_text: self.chars[previous..self.committed].iter().collect(),
            stable_prefix: self.stable_prefix(),
            pending: self.pending(),
            mentions,
            stable_chars: self.committed,
            total_chars: self.chars.len(),
        }
    }

    /// 最早一个「还可能继续变长」的词面起点；它之前的内容才可封存。
    fn stable_end(&self) -> usize {
        let total = self.chars.len();
        let mut stable = total;
        for start in 0..total {
            let Some(bucket) = self.by_first.get(&self.chars[start]) else {
                continue;
            };
            let remaining = total - start;
            for &index in bucket {
                let term = &self.term_chars[index];
                if term.len() <= remaining {
                    continue;
                }
                if term[..remaining] == self.chars[start..] {
                    stable = stable.min(start);
                    break;
                }
            }
            if stable == 0 {
                break;
            }
        }
        stable
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lexicon() -> MentionLexicon {
        let mut lexicon = MentionLexicon::new();
        lexicon.confirm("姚明", "姚明", Some("Person"), "test");
        lexicon.confirm("北京", "北京", Some("Place"), "test");
        lexicon.confirm("北京大学", "北京大学", Some("Organization"), "test");
        lexicon
    }

    #[test]
    fn ambiguous_tail_is_not_sealed_until_it_stops_growing() {
        let mut draft = StreamingDraft::new(&lexicon(), 7);

        let first = draft.push("北京");
        assert_eq!(first.stable_chars, 0, "{:?}", first);
        assert_eq!(first.added_text, "");
        assert_eq!(first.pending, "北京");
        assert!(first.mentions.is_empty());
        assert!(first.stable_prefix.is_empty());

        let second = draft.push("大学");
        assert_eq!(second.revision, 7);
        assert_eq!(second.stable_prefix, "北京大学");
        assert_eq!(second.added_text, "北京大学");
        assert_eq!(second.pending, "");
        assert_eq!(second.mentions.len(), 1);
        assert_eq!(second.mentions[0].surface, "北京大学");
        assert_eq!(second.mentions[0].start, 0);
        assert_eq!(second.mentions[0].end, 4);
        assert_eq!(second.mentions[0].etype.as_deref(), Some("Organization"));
    }

    #[test]
    fn unambiguous_mention_seals_while_tail_remains_pending() {
        let mut draft = StreamingDraft::new(&lexicon(), 1);
        let update = draft.push("姚明在");
        assert_eq!(update.stable_prefix, "姚明在");
        assert_eq!(update.pending, "");
        assert_eq!(update.mentions.len(), 1);
        assert_eq!(update.mentions[0].surface, "姚明");
        assert_eq!(update.mentions[0].start, 0);
        assert_eq!(update.mentions[0].end, 2);
        assert_eq!(update.mentions[0].char_len(), 2);

        let tail = draft.push("北京");
        assert_eq!(tail.stable_chars, update.stable_chars);
        assert_eq!(tail.pending, "北京");
        assert!(tail.mentions.is_empty());
        assert_eq!(draft.sealed_mentions().len(), 1);
    }

    #[test]
    fn stable_prefix_never_shrinks_under_fragment_appends() {
        let mut draft = StreamingDraft::new(&lexicon(), 1);
        let mut previous = 0;
        let mut sealed = Vec::new();
        for fragment in ["姚", "明", "在", "北", "京", "大", "学"] {
            let update = draft.push(fragment);
            assert!(
                update.stable_chars >= previous,
                "稳定水位回退：{} -> {}",
                previous,
                update.stable_chars
            );
            previous = update.stable_chars;
            sealed.extend(update.mentions.into_iter().map(|mention| mention.surface));
        }
        assert_eq!(
            sealed,
            vec!["姚明".to_string(), "北京大学".to_string()],
            "每个提及只密封一次"
        );
        assert_eq!(draft.sealed_mentions().len(), 2);
        assert_eq!(draft.pending(), "");
    }

    #[test]
    fn seal_all_commits_the_remaining_tail_and_pending_terms_stay_out() {
        let mut lexicon = lexicon();
        lexicon.nominate("苹果", Some("Fruit"), "test");
        let mut draft = StreamingDraft::new(&lexicon, 3);
        let update = draft.push("北京");
        assert_eq!(update.stable_chars, 0);

        let final_update = draft.seal_all();
        assert_eq!(final_update.stable_prefix, "北京");
        assert_eq!(final_update.added_text, "北京");
        assert_eq!(final_update.pending, "");
        assert_eq!(final_update.mentions.len(), 1);
        assert_eq!(final_update.mentions[0].surface, "北京");
        assert!(draft.sealed_mentions().iter().all(|item| item.surface != "苹果"));

        let apple = draft.push("苹果");
        assert!(apple.mentions.is_empty(), "待定词面不参与密封");
    }

    #[test]
    fn empty_fragments_are_quiescent_and_offsets_use_chars_not_bytes() {
        let mut draft = StreamingDraft::new(&lexicon(), 9);
        let empty = draft.push("");
        assert!(empty.is_quiescent());
        assert_eq!(empty.total_chars, 0);

        let update = draft.push("看姚明");
        assert_eq!(update.mentions[0].start, 1);
        assert_eq!(update.mentions[0].end, 3);
        assert_eq!(update.added_text, "看姚明");
    }
}
